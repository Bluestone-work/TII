"""
MAPPO 策略级 LSTM 模型（Policy-level LSTM）

架构：
  Actor  (去中心化): [local_obs(B), neighbor_obs(K*F)] ──► LSTM ──► Linear ──► 动作分布
  Critic (集中式):   global_state(N×B)                ──► MLP  ──► V(s)

观测布局（默认启用邻居观测）：
  [0 : B]             = local
  [B : B + K*F]       = neighbor
  [.. + 1]            = reset_flag
  [.. + N*B]          = global（仅 Critic 使用）
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType


class MAPPOMLPModel(TorchModelV2, nn.Module):
    """
    去中心化 Actor（LSTM）+ 集中式 Critic（MLP）。

    LSTM 在 Actor 侧积累跨 step 的时序隐状态，使模型能区分
    "自身移动导致的视角变化" 与 "动态障碍物真实接近"，提升动态避障能力。
    Critic 已享有全局特权信息，不需要时序记忆，保持轻量 MLP。
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **custom_model_kwargs,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        custom_cfg = model_config.get("custom_model_config", {})

        self.num_agents = int(custom_cfg.get("num_agents", 2))
        self.action_mode = "interaction_mode"
        self.use_high_level_branch = True
        self.scan_history_len = int(custom_cfg.get("scan_history_len", 4))
        self.scan_dim = int(custom_cfg.get("obstacle_top_k", 0))
        self.angular_bins = int(custom_cfg.get("angular_bins", 0))
        self.scan_emb_dim = int(custom_cfg.get("scan_emb_dim", 128))
        # Effective angular bins: prefer explicit angular_bins, fall back to obstacle_top_k
        if self.angular_bins < 8:
            self.angular_bins = self.scan_dim
        self.scan_raw_dim = self.scan_history_len * self.angular_bins
        # After 1D CNN encoding, the scan slot shrinks to scan_emb_dim
        self.scan_slot_dim = self.scan_emb_dim if self.angular_bins > self.scan_dim else self.scan_raw_dim
        self._use_scan_cnn = bool(self.angular_bins > self.scan_dim)
        self.option_state_dim = 0
        self.interaction_base_ego_dim = int(custom_cfg.get("interaction_base_ego_dim", 8))
        self.action_mask_dim = 0
        self.tracking_target_dim = 0
        self.interaction_ego_state_dim = int(custom_cfg.get("interaction_ego_state_dim", 8))
        self.base_safety_feature_dim = int(custom_cfg.get("base_safety_feature_dim", 8))
        self.predictive_feature_dim = 0
        self.gap_feature_dim = 0
        self.temporal_delta_dim = int(custom_cfg.get("temporal_delta_dim", 4))
        self.neighbor_prediction_dim = 0
        self.obstacle_motion_dim = 0
        # 环境侧观测槽位规则：最多保留 min(num_agents-1, 5) 个邻居，每个邻居 5 维
        # 这里不能直接默认 num_agents-1，否则当 num_agents > 6 时会与 env 的 obs 布局不一致。
        env_max_neighbors = max(0, min(self.num_agents - 1, 5))
        self.max_neighbors = int(custom_cfg.get("max_neighbors", env_max_neighbors))
        self.neighbor_feature_dim = int(custom_cfg.get("neighbor_feature_dim", 5))
        self.use_neighbor_obs = bool(custom_cfg.get("use_neighbor_obs", True))

        self.reset_flag_dim = 1
        total_obs_dim = obs_space.shape[0]
        base_denom = 1 + self.num_agents

        # 观测空间自动解析：
        # total = base_obs + neighbor_slots + reset_flag + global_state
        #       = B + (K*F) + 1 + (N*B)
        #       = (N+1)*B + K*F + 1
        #
        # 为兼容“修改过观测空间但未同步 custom_model_config”的情况，
        # 优先使用配置；若不匹配，则在合理候选 K 中自动反推。
        configured_neighbor_dim = (
            self.neighbor_feature_dim * self.max_neighbors if self.use_neighbor_obs else 0
        )
        candidate_neighbor_dims = []
        if self.use_neighbor_obs:
            candidate_neighbor_dims.append(configured_neighbor_dim)
            for k in range(env_max_neighbors + 1):
                candidate_neighbor_dims.append(k * self.neighbor_feature_dim)
        else:
            candidate_neighbor_dims.append(0)

        # 去重并保持顺序
        seen = set()
        candidate_neighbor_dims = [
            d for d in candidate_neighbor_dims if not (d in seen or seen.add(d))
        ]

        matched = None
        for neighbor_dim in candidate_neighbor_dims:
            base_numer = total_obs_dim - neighbor_dim - self.reset_flag_dim
            if base_numer > 0 and base_numer % base_denom == 0:
                matched = (neighbor_dim, base_numer // base_denom)
                break

        if matched is None:
            raise ValueError(
                "[MAPPOMLPModel] obs 维度不匹配，无法从观测空间反推出布局: "
                f"total_obs_dim={total_obs_dim}, num_agents={self.num_agents}, "
                f"candidate_neighbor_dims={candidate_neighbor_dims}, "
                f"neighbor_feature_dim={self.neighbor_feature_dim}"
            )

        self.neighbor_dim, self.base_obs_dim = matched
        self.max_neighbors = self.neighbor_dim // self.neighbor_feature_dim if self.neighbor_feature_dim > 0 else 0

        # Effective dimensions after scan CNN replaces raw scan with embedding
        if self._use_scan_cnn:
            self.eff_base_obs_dim = self.base_obs_dim - self.scan_raw_dim + self.scan_emb_dim
        else:
            self.eff_base_obs_dim = self.base_obs_dim
        self.eff_actor_obs_dim = self.eff_base_obs_dim + self.neighbor_dim
        self.eff_actor_total_dim = self.eff_actor_obs_dim + self.reset_flag_dim

        self.actor_obs_dim = self.base_obs_dim + self.neighbor_dim
        self.actor_total_dim = self.actor_obs_dim + self.reset_flag_dim
        self.global_state_dim = self.num_agents * self.base_obs_dim
        self.interaction_ego_start = self.scan_slot_dim + 2 + 2
        self.safety_start = self.interaction_ego_start
        self.safety_end = self.safety_start + self.base_safety_feature_dim
        self.ego_state_start = self.safety_start
        self.ego_state_end = self.ego_state_start + self.interaction_base_ego_dim
        self.temporal_delta_start = self.safety_end
        self.temporal_delta_end = self.temporal_delta_start + self.temporal_delta_dim
        self.option_state_start = self.temporal_delta_end
        self.option_state_end = self.option_state_start
        self.action_mask_start = self.option_state_end
        self.action_mask_end = self.action_mask_start
        self.tracking_target_start = self.action_mask_end
        self.tracking_target_end = self.tracking_target_start

        # ── 1D CNN Angular Scan Encoder ─────────────────────────────────────
        if self._use_scan_cnn:
            self.scan_cnn = nn.Sequential(
                nn.Conv1d(self.scan_history_len, 32, kernel_size=7, stride=2, padding=3),
                nn.GELU(),
                nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
                nn.GELU(),
                nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )
        else:
            self.scan_cnn = nn.Identity()

        hidden_dim = int(custom_cfg.get("hidden_dim", 256))
        self.lstm_hidden_dim = int(custom_cfg.get("lstm_hidden_dim", hidden_dim))
        self.actor_source_dim = (
            self.base_safety_feature_dim + self.temporal_delta_dim + self.neighbor_dim
            if self.use_high_level_branch else (
                self.eff_actor_obs_dim if self._use_scan_cnn else self.actor_obs_dim
            )
        )
        self.actor_input_dim = hidden_dim if self.use_high_level_branch else self.actor_source_dim

        # ── Actor LSTM（去中心化，仅看 local + neighbor）─────────────────────
        # 1 层单向 LSTM；hidden state 跨 step 保持，积累时序记忆
        self.actor_input_norm = nn.LayerNorm(self.actor_source_dim)
        self.actor_pre_net = (
            nn.Sequential(
                nn.Linear(self.actor_source_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            if self.use_high_level_branch else nn.Identity()
        )
        self.lstm_cell = nn.LSTMCell(self.actor_input_dim, self.lstm_hidden_dim)
        # LayerNorm 稳定 LSTM 输出，加速收敛
        self.lstm_norm = nn.LayerNorm(self.lstm_hidden_dim)

        self.actor_head = nn.Linear(self.lstm_hidden_dim, num_outputs)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.constant_(self.actor_head.bias, 0.0)

        # ── Critic MLP（集中式，全局状态，无需时序）──────────────────────────
        self.critic_net = nn.Sequential(
            nn.Linear(self.global_state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.critic_head = nn.Linear(hidden_dim, 1)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.constant_(self.critic_head.bias, 0.0)

        self._cur_value = None

        lstm_p   = sum(p.numel() for p in self.lstm_cell.parameters())
        actor_p  = sum(p.numel() for p in self.actor_head.parameters())
        critic_p = sum(p.numel() for p in list(self.critic_net.parameters())
                                         + list(self.critic_head.parameters()))
        print(
            f"\n[MAPPOMLPModel · MAPPO-LSTM] 参数量: {lstm_p+actor_p+critic_p:,}\n"
            f"  obs_space={obs_space.shape}  base_obs_dim={self.base_obs_dim}"
            f"  neighbor_dim={self.neighbor_dim}  actor_obs_dim={self.actor_obs_dim}"
            f"  num_agents={self.num_agents}\n"
            f"  LSTM (1层单向): {lstm_p:,}  input={self.actor_input_dim}  "
            f"hidden={self.lstm_hidden_dim}\n"
            f"  Actor head:    {actor_p:,}\n"
            f"  Critic MLP:    {critic_p:,}  "
            f"(in={self.global_state_dim} = {self.num_agents}×{self.base_obs_dim})"
        )

    @override(TorchModelV2)
    def get_initial_state(self) -> List[TensorType]:
        """返回 LSTM 初始隐状态 [h0, c0]，各自 shape=[lstm_hidden_dim]。
        RLlib 会自动在 batch 维度上扩展，并在 episode 结束时重置。"""
        w = next(self.lstm_cell.parameters())
        return [
            w.new_zeros(self.lstm_hidden_dim),   # h0
            w.new_zeros(self.lstm_hidden_dim),   # c0
        ]

    def load_state_dict(self, state_dict, strict=True):
        compat_state = dict(state_dict)

        legacy_key_map = {
            "lstm.weight_ih_l0": "lstm_cell.weight_ih",
            "lstm.weight_hh_l0": "lstm_cell.weight_hh",
            "lstm.bias_ih_l0": "lstm_cell.bias_ih",
            "lstm.bias_hh_l0": "lstm_cell.bias_hh",
        }
        for old_key, new_key in legacy_key_map.items():
            if old_key in compat_state and new_key not in compat_state:
                compat_state[new_key] = compat_state.pop(old_key)

        current_state = nn.Module.state_dict(self)
        filtered_state = {}
        skipped = []
        for key, value in compat_state.items():
            if key not in current_state:
                skipped.append(f"{key} (missing)")
                continue
            if current_state[key].shape != value.shape:
                skipped.append(
                    f"{key} {tuple(value.shape)} -> {tuple(current_state[key].shape)}"
                )
                continue
            filtered_state[key] = value

        if skipped:
            print(
                "⚠️  检测到旧版 checkpoint，以下权重因结构变更被跳过加载:\n"
                + "\n".join(f"   - {item}" for item in skipped)
            )

        return nn.Module.load_state_dict(self, filtered_state, strict=False)

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        obs = input_dict["obs_flat"].float()   # [B*T, obs_dim]

        # ── 1D CNN angular scan encoder ────────────────────────────────────
        if self._use_scan_cnn:
            raw_scan = obs[:, :self.scan_raw_dim]               # [B*T, hist*ang_bins]
            rest = obs[:, self.scan_raw_dim:]                   # non-scan remainder
            scan_2d = raw_scan.view(-1, self.scan_history_len, self.angular_bins)
            scan_emb = self.scan_cnn(scan_2d)                   # [B*T, scan_emb_dim]
            obs = torch.cat([scan_emb, rest], dim=-1)

        local_obs = obs[:, :self.eff_base_obs_dim]
        neighbor_obs = obs[:, self.eff_base_obs_dim:self.eff_actor_obs_dim]
        if self.use_high_level_branch:
            ego_state = self._extract_interaction_actor_features(local_obs)
            actor_obs = torch.cat([ego_state, neighbor_obs], dim=-1)
        else:
            actor_obs = obs[:, :self.eff_actor_obs_dim]
        actor_obs = self.actor_input_norm(actor_obs)
        actor_obs = self.actor_pre_net(actor_obs)
        reset_flag = obs[:, self.eff_actor_obs_dim:self.eff_actor_total_dim]
        global_start = self.eff_actor_total_dim
        global_obs = obs[:, global_start : global_start + self.global_state_dim]  # [B*T, N*base]

        # ── Actor：LSTM 时序编码 ──────────────────────────────────────────────
        # RLlib 把 batch 中所有序列展平为 [B*T, obs_dim]，B = seq_lens.shape[0]
        B = seq_lens.shape[0]
        T = actor_obs.shape[0] // B                           # max_seq_len

        local_3d = actor_obs.view(B, T, self.actor_input_dim)    # [B, T, actor_input]
        reset_3d = reset_flag.view(B, T, self.reset_flag_dim)

        h_t = state[0]
        c_t = state[1]
        outputs = []
        for t in range(T):
            reset_mask = (reset_3d[:, t, 0] > 0.5).unsqueeze(-1)
            h_t = torch.where(reset_mask, torch.zeros_like(h_t), h_t)
            c_t = torch.where(reset_mask, torch.zeros_like(c_t), c_t)
            h_t, c_t = self.lstm_cell(local_3d[:, t, :], (h_t, c_t))
            outputs.append(h_t)

        lstm_out = torch.stack(outputs, dim=1)
        # lstm_out: [B, T, hidden]  →  展平为 [B*T, hidden]
        lstm_feat = self.lstm_norm(lstm_out.reshape(-1, self.lstm_hidden_dim))

        action_out = self.actor_head(lstm_feat)               # [B*T, num_outputs]

        # Action masks are kept as diagnostic observation features only.
        # Do not alter logits; invalid option usage is learned from reward.

        # ── Critic：MLP，直接用全局状态，无需时序 ──────────────────────────────
        self._cur_value = self._critic_from_global_obs(global_obs)  # [B*T]

        return action_out, [h_t, c_t]

    def _extract_interaction_actor_features(self, local_obs: torch.Tensor) -> torch.Tensor:
        """Build the minimal actor input from risk summary and temporal deltas."""
        risk_summary = local_obs[:, self.safety_start:self.safety_end]
        temporal_delta = local_obs[:, self.temporal_delta_start:self.temporal_delta_end]
        return torch.cat([risk_summary, temporal_delta], dim=-1)

    def _extract_action_mask(self, local_obs: torch.Tensor) -> torch.Tensor:
        """Extract action mask from local observation.

        Layout is assembled explicitly from local observation slots.
        """
        if self.action_mask_end > local_obs.shape[1]:
            return None
        return local_obs[:, self.action_mask_start:self.action_mask_end].float()

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._cur_value is not None, "必须先调用 forward()"
        return self._cur_value

    def _critic_from_global_obs(self, global_obs: torch.Tensor) -> torch.Tensor:
        """Critic 前向；global_obs shape=[B, num_agents, base_obs_dim]。"""
        global_flat = global_obs.reshape(global_obs.shape[0], -1)
        return self.critic_head(self.critic_net(global_flat)).squeeze(-1)

    def compute_counterfactual_values(
        self,
        obs: torch.Tensor,
        agent_indices: torch.Tensor,
    ) -> torch.Tensor:
        """按样本的当前 agent index 置零其全局槽位，估计 leave-one-out V(s\\i)。"""
        obs = obs.float()
        batch_size = obs.shape[0]
        if batch_size == 0:
            return obs.new_zeros((0,))

        global_start = self.actor_total_dim
        global_obs = obs[:, global_start : global_start + self.global_state_dim]
        global_obs = global_obs.view(batch_size, self.num_agents, self.base_obs_dim).clone()

        if self.num_agents <= 1:
            return self._critic_from_global_obs(global_obs)

        row_index = torch.arange(batch_size, device=obs.device)
        masked_agent_idx = (
            agent_indices.to(device=obs.device).long().clamp_(0, self.num_agents - 1)
        )
        global_obs[row_index, masked_agent_idx, :] = 0.0
        return self._critic_from_global_obs(global_obs)


# 注册名称
MODEL_NAME_MLP = "mappo_mlp"
