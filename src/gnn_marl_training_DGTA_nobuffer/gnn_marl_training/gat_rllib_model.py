"""
RLlib GAT actor model.

Actor trunk follows the MAPPO-MLP-LSTM baseline and adds a lightweight
social-risk GAT branch. The graph branch consumes oracle social-risk tokens
from local observations by default, so navigation and local scene modeling stay
on the same path as the MLP baseline.
"""
import math
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from gnn_marl_training.gat_policy import GATEncoder


class GATRLlibModel(TorchModelV2, nn.Module):
    """
    MAPPO actor with:
    1. MLP-compatible actor trunk: [local_obs, neighbor_obs] -> LSTM
    2. Social GAT side branch: oracle social-risk tokens -> ego-centric graph
    3. Residual fusion: social risk only modulates the post-LSTM policy feature
    4. Centralized critic: MLP by default, optional GAT critic for ablations
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config: ModelConfigDict,
        name: str,
        **custom_model_kwargs
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        custom_cfg = model_config.get("custom_model_config", {})

        self.num_agents = int(custom_cfg.get("num_agents", 2))
        self.neighbor_feature_dim = int(custom_cfg.get("neighbor_feature_dim", 7))
        self.max_neighbors = int(custom_cfg.get("max_neighbors", self.num_agents - 1))
        self.reset_flag_dim = 1
        self.use_feature_norm = bool(custom_cfg.get("use_feature_norm", True))
        self.dropout_p = float(custom_cfg.get("dropout", 0.0))
        self.use_max_pool_critic = bool(custom_cfg.get("use_max_pool_critic", True))
        self.actor_graph_mode = str(custom_cfg.get("actor_graph_mode", "social_risk")).strip().lower()
        self.graph_ablation = str(custom_cfg.get("graph_ablation", "dual_graph")).strip().lower()
        self.critic_mode = str(custom_cfg.get("critic_mode", "mlp")).strip().lower()
        self.risk_bias_scale = float(custom_cfg.get("risk_bias_scale", 2.5))
        self.scan_history_len = int(custom_cfg.get("scan_history_len", 4))
        self.base_safety_feature_dim = int(custom_cfg.get("base_safety_feature_dim", 7))
        self.predictive_feature_dim = int(custom_cfg.get("predictive_feature_dim", 6))
        self.gap_feature_dim = int(custom_cfg.get("gap_feature_dim", 0))
        self.neighbor_prediction_dim = int(custom_cfg.get("neighbor_prediction_dim", 0))
        self.neighbor_prediction_feature_dim = int(
            custom_cfg.get("neighbor_prediction_feature_dim", 6)
        )
        self.obstacle_motion_dim = int(custom_cfg.get("obstacle_motion_dim", 0))
        self.obstacle_motion_feature_dim = int(
            custom_cfg.get("obstacle_motion_feature_dim", 7)
        )
        self.enable_token_token_edges = bool(custom_cfg.get("enable_token_token_edges", False))
        self.target_obs_dim = int(custom_cfg.get("target_obs_dim", 6))
        self.local_map_dim = int(custom_cfg.get("local_map_dim", 0))
        self.agent_id_dim = int(custom_cfg.get("agent_id_dim", 8))
        self.non_scan_feature_dim = (
            self.target_obs_dim
            + 2
            + self.base_safety_feature_dim
            + self.neighbor_prediction_dim
            + self.obstacle_motion_dim
            + self.agent_id_dim
        )
        # No scan pool in new architecture — local map CNN replaces it
        self.scan_dim = 0
        self.scan_total_dim = 0

        self.obs_dim = obs_space.shape[0]
        self.neighbor_dim = self.neighbor_feature_dim * self.max_neighbors
        base_numer = self.obs_dim - self.local_map_dim - self.neighbor_dim - self.reset_flag_dim
        base_denom = 1 + self.num_agents
        if base_numer <= 0 or base_numer % base_denom != 0:
            raise ValueError(
                "[GATRLlibModel] obs 维度不匹配: "
                f"obs_dim={self.obs_dim}, local_map_dim={self.local_map_dim}, "
                f"neighbor_dim={self.neighbor_dim}, "
                f"num_agents={self.num_agents}, reset_flag_dim={self.reset_flag_dim}"
            )
        self.base_obs_dim = base_numer // base_denom

        if self.non_scan_feature_dim != self.base_obs_dim:
            raise ValueError(
                "[GATRLlibModel] base_obs 布局不匹配: "
                f"non_scan_feature_dim={self.non_scan_feature_dim}, "
                f"base_obs_dim={self.base_obs_dim}"
            )

        # Local map CNN
        self.local_map_cnn_out_dim = 64 if self.local_map_dim > 0 else 0
        if self.local_map_dim > 0:
            self.local_map_grid_size = int(math.isqrt(self.local_map_dim // 2))
            self.local_map_frames = 2
            self.local_map_cnn = nn.Sequential(
                nn.Conv2d(self.local_map_frames, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(),
                nn.Linear(32 * 4 * 4, self.local_map_cnn_out_dim),
                nn.ReLU(),
            )
        else:
            self.local_map_cnn = None

        self.actor_obs_dim = self.base_obs_dim + self.local_map_cnn_out_dim + self.neighbor_dim
        self.actor_total_dim = self.actor_obs_dim + self.reset_flag_dim
        self.global_state_dim = self.num_agents * self.base_obs_dim
        # Raw obs offset where global_state begins (after base_obs + local_map + neighbor + reset_flag)
        self.global_state_raw_offset = (
            self.base_obs_dim + self.local_map_dim + self.neighbor_dim + self.reset_flag_dim
        )

        if self.neighbor_prediction_dim > 0:
            if self.neighbor_prediction_dim % self.neighbor_prediction_feature_dim != 0:
                raise ValueError(
                    "[GATRLlibModel] neighbor_prediction_dim 与 feature_dim 不整除: "
                    f"neighbor_prediction_dim={self.neighbor_prediction_dim}, "
                    f"neighbor_prediction_feature_dim={self.neighbor_prediction_feature_dim}"
                )
            self.social_graph_top_k = (
                self.neighbor_prediction_dim // self.neighbor_prediction_feature_dim
            )
        else:
            self.social_graph_top_k = 0

        if self.obstacle_motion_dim > 0:
            if self.obstacle_motion_dim % self.obstacle_motion_feature_dim != 0:
                raise ValueError(
                    "[GATRLlibModel] obstacle_motion_dim 与 feature_dim 不整除: "
                    f"obstacle_motion_dim={self.obstacle_motion_dim}, "
                    f"obstacle_motion_feature_dim={self.obstacle_motion_feature_dim}"
                )
            self.obstacle_graph_top_k = (
                self.obstacle_motion_dim // self.obstacle_motion_feature_dim
            )
        else:
            self.obstacle_graph_top_k = 0
        self.graph_token_top_k = self.social_graph_top_k + self.obstacle_graph_top_k
        if self.graph_ablation not in {"social_only", "obstacle_only", "dual_graph"}:
            raise ValueError(
                f"[GATRLlibModel] 未知 graph_ablation={self.graph_ablation}，"
                "允许: social_only | obstacle_only | dual_graph"
            )
        self.use_social_graph_tokens = self.graph_ablation in {"social_only", "dual_graph"}
        self.use_obstacle_graph_tokens = self.graph_ablation in {"obstacle_only", "dual_graph"}
        if self.graph_ablation == "social_only" and self.social_graph_top_k <= 0:
            raise ValueError("[GATRLlibModel] graph_ablation=social_only 但 social token 维度为 0")
        if self.graph_ablation == "obstacle_only" and self.obstacle_graph_top_k <= 0:
            raise ValueError("[GATRLlibModel] graph_ablation=obstacle_only 但 obstacle token 维度为 0")

        self.social_token_start = (
            self.target_obs_dim
            + 2
            + self.base_safety_feature_dim
        )
        self.social_token_end = self.social_token_start + self.neighbor_prediction_dim
        if self.social_token_end > self.base_obs_dim:
            raise ValueError(
                "[GATRLlibModel] 社交风险 token 切片越界: "
                f"social_token_end={self.social_token_end}, base_obs_dim={self.base_obs_dim}"
            )
        self.obstacle_token_start = self.social_token_end
        self.obstacle_token_end = self.obstacle_token_start + self.obstacle_motion_dim
        if self.obstacle_token_end > self.base_obs_dim:
            raise ValueError(
                "[GATRLlibModel] 动态障碍 token 切片越界: "
                f"obstacle_token_end={self.obstacle_token_end}, base_obs_dim={self.base_obs_dim}"
            )

        hidden_dim = int(custom_cfg.get("hidden_dim", 128))
        gat_hidden_dim = int(custom_cfg.get("gat_hidden_dim", 128))
        lstm_hidden_dim = int(custom_cfg.get("lstm_hidden_dim", 256))
        n_gat_heads = int(custom_cfg.get("n_gat_heads", 4))

        # Actor trunk: same structure as the MLP baseline.
        self.actor_input_norm = (
            nn.LayerNorm(self.actor_obs_dim) if self.use_feature_norm else nn.Identity()
        )
        self.local_input_norm = (
            nn.LayerNorm(self.base_obs_dim) if self.use_feature_norm else nn.Identity()
        )
        self.neighbor_input_norm = (
            nn.LayerNorm(self.neighbor_feature_dim) if self.use_feature_norm else nn.Identity()
        )
        self.lstm_cell = nn.LSTMCell(self.actor_obs_dim, lstm_hidden_dim)
        self.lstm_norm = nn.LayerNorm(lstm_hidden_dim)
        self.lstm_hidden_dim = lstm_hidden_dim

        # Social GAT side branch.
        self.social_ego_encoder = nn.Sequential(
            nn.Linear(self.base_obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.social_token_input_norm = (
            nn.LayerNorm(self.neighbor_prediction_feature_dim)
            if self.use_feature_norm and self.social_graph_top_k > 0
            else nn.Identity()
        )
        self.social_token_encoder = nn.Sequential(
            nn.Linear(self.neighbor_prediction_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.obstacle_token_input_norm = (
            nn.LayerNorm(self.obstacle_motion_feature_dim)
            if self.use_feature_norm and self.obstacle_graph_top_k > 0
            else nn.Identity()
        )
        self.obstacle_token_encoder = nn.Sequential(
            nn.Linear(self.obstacle_motion_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.neighbor_token_encoder = nn.Sequential(
            nn.Linear(self.neighbor_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.social_gat = GATEncoder(
            input_dim=hidden_dim,
            hidden_dim=gat_hidden_dim,
            n_heads=n_gat_heads,
        )
        self.obstacle_gat = GATEncoder(
            input_dim=hidden_dim,
            hidden_dim=gat_hidden_dim,
            n_heads=n_gat_heads,
        )
        self.social_proj = nn.Sequential(
            nn.Linear(self.social_gat.output_dim, lstm_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(lstm_hidden_dim),
        )
        self.obstacle_proj = nn.Sequential(
            nn.Linear(self.obstacle_gat.output_dim, lstm_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(lstm_hidden_dim),
        )
        self.dual_graph_fusion = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(lstm_hidden_dim),
        )
        self.graph_gate = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.graph_gate[0].bias, -1.0)
        self.graph_delta = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
        )
        self.policy_fusion_norm = nn.LayerNorm(lstm_hidden_dim)
        self.actor_head = nn.Linear(lstm_hidden_dim, num_outputs)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.constant_(self.actor_head.bias, 0.0)

        if self.critic_mode == "gat":
            self.central_node_encoder = nn.Sequential(
                nn.Linear(self.base_obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.central_gat = GATEncoder(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                n_heads=n_gat_heads,
            )
            self.critic_feat_dim = hidden_dim * (2 if self.use_max_pool_critic else 1)
            self.central_critic_head = nn.Linear(self.critic_feat_dim, 1)
            nn.init.orthogonal_(self.central_critic_head.weight, gain=1.0)
            nn.init.constant_(self.central_critic_head.bias, 0)
        else:
            self.critic_net = nn.Sequential(
                nn.Linear(self.global_state_dim, lstm_hidden_dim),
                nn.Tanh(),
                nn.Linear(lstm_hidden_dim, lstm_hidden_dim),
                nn.Tanh(),
            )
            self.central_critic_head = nn.Linear(lstm_hidden_dim, 1)
            nn.init.orthogonal_(self.central_critic_head.weight, gain=1.0)
            nn.init.constant_(self.central_critic_head.bias, 0)

        self._cur_value = None

    def _encode_social_prediction_tokens(
        self,
        local_obs_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if self.social_graph_top_k <= 0:
            return None, None, None

        batch_size = local_obs_raw.shape[0]
        social_tokens_raw = local_obs_raw[:, self.social_token_start:self.social_token_end].view(
            batch_size,
            self.social_graph_top_k,
            self.neighbor_prediction_feature_dim,
        )
        social_tokens = self.social_token_input_norm(social_tokens_raw)
        social_flat = social_tokens.reshape(-1, self.neighbor_prediction_feature_dim)
        social_features = self.social_token_encoder(social_flat).view(
            batch_size,
            self.social_graph_top_k,
            -1,
        )

        social_valid = (social_tokens_raw.abs().sum(dim=-1) > 1e-6).float()
        # Attention bias from proximity (dist_norm at index 4: closer = higher bias)
        proximity = (1.0 - social_tokens_raw[:, :, 4]).clamp(0.0, 1.0)
        social_bias = proximity * social_valid
        return social_features, social_valid, social_bias

    def _encode_obstacle_prediction_tokens(
        self,
        local_obs_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if self.obstacle_graph_top_k <= 0:
            return None, None, None

        batch_size = local_obs_raw.shape[0]
        obstacle_tokens_raw = local_obs_raw[
            :,
            self.obstacle_token_start:self.obstacle_token_end,
        ].view(
            batch_size,
            self.obstacle_graph_top_k,
            self.obstacle_motion_feature_dim,
        )
        obstacle_tokens = self.obstacle_token_input_norm(obstacle_tokens_raw)
        obstacle_flat = obstacle_tokens.reshape(-1, self.obstacle_motion_feature_dim)
        obstacle_features = self.obstacle_token_encoder(obstacle_flat).view(
            batch_size,
            self.obstacle_graph_top_k,
            -1,
        )

        obstacle_valid = (obstacle_tokens_raw.abs().sum(dim=-1) > 1e-6).float()
        cur_pos = obstacle_tokens_raw[:, :, :2]
        cur_vel = obstacle_tokens_raw[:, :, 2:4]
        future_pos = obstacle_tokens_raw[:, :, 4:6]
        cur_dist = torch.norm(cur_pos, dim=-1).clamp(min=1e-6)
        future_dist = torch.norm(future_pos, dim=-1).clamp(min=1e-6)
        speed = torch.norm(cur_vel, dim=-1).clamp(min=0.0)
        distance_risk = (1.0 / (1.0 + cur_dist)).clamp(0.0, 1.0)
        future_risk = (1.0 / (1.0 + future_dist)).clamp(0.0, 1.0)
        crossing_risk = (
            torch.abs(future_pos[:, :, 1]).lt(0.35).float()
            * (speed / 1.5).clamp(0.0, 1.0)
        )
        obstacle_bias = (
            0.45 * distance_risk
            + 0.35 * future_risk
            + 0.20 * crossing_risk
        ) * obstacle_valid
        return obstacle_features, obstacle_valid, obstacle_bias

    def _assemble_ego_token_graph(
        self,
        ego_context: torch.Tensor,
        token_features: torch.Tensor,
        token_valid: torch.Tensor,
        bias_score: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = ego_context.shape[0]
        device = ego_context.device
        token_count = int(token_features.shape[1])
        n_nodes = 1 + token_count
        graph_nodes = torch.cat([ego_context.unsqueeze(1), token_features], dim=1)
        adj = torch.zeros(batch_size, n_nodes, n_nodes, device=device)
        adj[:, 0, 0] = 1.0
        for k in range(token_count):
            adj[:, 0, k + 1] = token_valid[:, k]
            adj[:, k + 1, 0] = token_valid[:, k]
            adj[:, k + 1, k + 1] = token_valid[:, k]
        # token-to-token edges: valid tokens attend to each other
        if self.enable_token_token_edges:
            for i in range(token_count):
                for j in range(token_count):
                    if i != j:
                        adj[:, i + 1, j + 1] = token_valid[:, i] * token_valid[:, j]

        attention_bias = torch.zeros(batch_size, n_nodes, n_nodes, device=device)
        attention_bias[:, 0, 1:] = self.risk_bias_scale * bias_score
        attention_bias[:, 1:, 0] = 0.5 * self.risk_bias_scale * bias_score
        return graph_nodes, adj, attention_bias

    def _build_actor_prediction_graph(
        self,
        ego_context: torch.Tensor,
        local_obs_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        token_features_list = []
        token_valid_list = []
        bias_score_list = []

        if self.use_social_graph_tokens:
            social_features, social_valid, social_bias = self._encode_social_prediction_tokens(local_obs_raw)
            if social_features is not None:
                token_features_list.append(social_features)
                token_valid_list.append(social_valid)
                bias_score_list.append(social_bias)

        if self.use_obstacle_graph_tokens:
            obstacle_features, obstacle_valid, obstacle_bias = self._encode_obstacle_prediction_tokens(local_obs_raw)
            if obstacle_features is not None:
                token_features_list.append(obstacle_features)
                token_valid_list.append(obstacle_valid)
                bias_score_list.append(obstacle_bias)

        if not token_features_list:
            raise ValueError(
                f"[GATRLlibModel] graph_ablation={self.graph_ablation} 但没有可用 token"
            )

        token_features = torch.cat(token_features_list, dim=1)
        token_valid = torch.cat(token_valid_list, dim=1)
        bias_score = torch.cat(bias_score_list, dim=1)
        return self._assemble_ego_token_graph(ego_context, token_features, token_valid, bias_score)

    def _build_social_graph_from_predictions(
        self,
        ego_context: torch.Tensor,
        local_obs_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return self._build_actor_prediction_graph(ego_context, local_obs_raw)

    def _build_social_prediction_graph(
        self,
        ego_context: torch.Tensor,
        local_obs_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        social_features, social_valid, social_bias = self._encode_social_prediction_tokens(local_obs_raw)
        if social_features is None:
            raise ValueError("[GATRLlibModel] social graph token 不可用")
        return self._assemble_ego_token_graph(ego_context, social_features, social_valid, social_bias)

    def _build_obstacle_prediction_graph(
        self,
        ego_context: torch.Tensor,
        local_obs_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        obstacle_features, obstacle_valid, obstacle_bias = self._encode_obstacle_prediction_tokens(local_obs_raw)
        if obstacle_features is None:
            raise ValueError("[GATRLlibModel] obstacle graph token 不可用")
        return self._assemble_ego_token_graph(ego_context, obstacle_features, obstacle_valid, obstacle_bias)

    def _encode_social_graph_branch(
        self,
        ego_context: torch.Tensor,
        local_obs_raw: torch.Tensor,
    ) -> torch.Tensor:
        graph_nodes, adj, attention_bias = self._build_social_prediction_graph(ego_context, local_obs_raw)
        social_gat_out = self.social_gat(graph_nodes, adj, attention_bias=attention_bias)
        return self.social_proj(social_gat_out[:, 0, :])

    def _encode_obstacle_graph_branch(
        self,
        ego_context: torch.Tensor,
        local_obs_raw: torch.Tensor,
    ) -> torch.Tensor:
        graph_nodes, adj, attention_bias = self._build_obstacle_prediction_graph(ego_context, local_obs_raw)
        obstacle_gat_out = self.obstacle_gat(graph_nodes, adj, attention_bias=attention_bias)
        return self.obstacle_proj(obstacle_gat_out[:, 0, :])

    def _fuse_graph_features(
        self,
        social_feat: torch.Tensor | None,
        obstacle_feat: torch.Tensor | None,
    ) -> torch.Tensor:
        if social_feat is not None and obstacle_feat is not None:
            return self.dual_graph_fusion(torch.cat([social_feat, obstacle_feat], dim=-1))
        if social_feat is not None:
            return social_feat
        if obstacle_feat is not None:
            return obstacle_feat
        raise ValueError("[GATRLlibModel] dual-graph fusion 时没有可用图分支特征")

    def _build_social_graph_from_neighbors(
        self,
        ego_context: torch.Tensor,
        neighbor_obs_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_size = ego_context.shape[0]
        device = ego_context.device
        n_nodes = 1 + self.max_neighbors

        neighbor_tokens = self.neighbor_input_norm(neighbor_obs_raw)
        token_flat = neighbor_tokens.reshape(-1, self.neighbor_feature_dim)
        token_features = self.neighbor_token_encoder(token_flat).view(
            batch_size,
            self.max_neighbors,
            -1,
        )

        token_valid = (neighbor_obs_raw.abs().sum(dim=-1) > 1e-6).float()
        if self.neighbor_feature_dim >= 5:
            dist = neighbor_obs_raw[:, :, 4].clamp(min=1e-6)
            token_valid = token_valid * (dist > 1e-6).float()
            rel_pos = neighbor_obs_raw[:, :, :2]
            rel_vel = neighbor_obs_raw[:, :, 2:4]
            approach = (-torch.sum(rel_pos * rel_vel, dim=-1) / dist).clamp(min=0.0, max=2.0)
            distance_risk = (1.0 / (1.0 + dist)).clamp(0.0, 1.0)
            bias_score = (distance_risk + 0.5 * approach) * token_valid
            attention_bias = torch.zeros(batch_size, n_nodes, n_nodes, device=device)
            attention_bias[:, 0, 1:] = self.risk_bias_scale * bias_score
            attention_bias[:, 1:, 0] = 0.5 * self.risk_bias_scale * bias_score
        else:
            attention_bias = None

        graph_nodes = torch.cat([ego_context.unsqueeze(1), token_features], dim=1)
        adj = torch.zeros(batch_size, n_nodes, n_nodes, device=device)
        adj[:, 0, 0] = 1.0
        for k in range(self.max_neighbors):
            adj[:, 0, k + 1] = token_valid[:, k]
            adj[:, k + 1, 0] = token_valid[:, k]
            adj[:, k + 1, k + 1] = token_valid[:, k]
        # token-to-token edges
        if self.enable_token_token_edges:
            for i in range(self.max_neighbors):
                for j in range(self.max_neighbors):
                    if i != j:
                        adj[:, i + 1, j + 1] = token_valid[:, i] * token_valid[:, j]
        return graph_nodes, adj, attention_bias

    def _encode_neighbor_social_branch(
        self,
        ego_context: torch.Tensor,
        neighbor_obs_raw: torch.Tensor,
    ) -> torch.Tensor:
        """neighbor 模式下的 social 分支：用通信邻居观测构图并投影到策略特征维度。"""
        graph_nodes, adj, attention_bias = self._build_social_graph_from_neighbors(
            ego_context,
            neighbor_obs_raw,
        )
        social_gat_out = self.social_gat(graph_nodes, adj, attention_bias=attention_bias)
        return self.social_proj(social_gat_out[:, 0, :])

    def load_state_dict(self, state_dict, strict=True):
        """兼容旧 checkpoint：仅加载 shape 匹配的权重。"""
        current_state = nn.Module.state_dict(self)
        filtered_state = {}
        skipped = []
        for key, value in state_dict.items():
            if key not in current_state:
                skipped.append(f"{key} (missing)")
                continue
            if current_state[key].shape != value.shape:
                skipped.append(f"{key} {tuple(value.shape)} -> {tuple(current_state[key].shape)}")
                continue
            filtered_state[key] = value

        if skipped:
            print(
                "⚠️  [GATRLlibModel] 检测到结构变更，以下权重被跳过:\n"
                + "\n".join(f"   - {item}" for item in skipped)
            )
        return nn.Module.load_state_dict(self, filtered_state, strict=False)

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType
    ) -> Tuple[TensorType, List[TensorType]]:
        obs = input_dict["obs_flat"].float()
        batch_size = obs.shape[0]

        # Parse obs layout: [base_obs | local_map(2048) | neighbor | reset_flag | global_state]
        local_obs_raw = obs[:, :self.base_obs_dim]
        offset = self.base_obs_dim

        if self.local_map_cnn is not None:
            local_map_flat = obs[:, offset:offset + self.local_map_dim]
            offset += self.local_map_dim
            gs = self.local_map_grid_size
            local_map_2d = local_map_flat.view(batch_size, self.local_map_frames, gs, gs)
            local_map_feat = self.local_map_cnn(local_map_2d)
        else:
            local_map_feat = obs.new_zeros(batch_size, 0)

        neighbor_obs_flat = obs[:, offset:offset + self.neighbor_dim]
        offset += self.neighbor_dim

        # Build actor_obs: [base_obs_normed | local_map_feat(64) | neighbor_obs]
        actor_obs = torch.cat([
            self.actor_input_norm(torch.cat([local_obs_raw, local_map_feat, neighbor_obs_flat], dim=-1)),
        ], dim=-1)

        reset_flag = obs[:, offset:offset + self.reset_flag_dim]

        batch_seqs = seq_lens.shape[0]
        max_seq_len = actor_obs.shape[0] // batch_seqs
        actor_obs_seq = actor_obs.view(batch_seqs, max_seq_len, self.actor_obs_dim)
        reset_seq = reset_flag.view(batch_seqs, max_seq_len, self.reset_flag_dim)

        if len(state) < 2 or state[0].nelement() == 0 or state[0].shape[0] != batch_seqs:
            h_t = actor_obs.new_zeros(batch_seqs, self.lstm_hidden_dim)
            c_t = actor_obs.new_zeros(batch_seqs, self.lstm_hidden_dim)
        else:
            h_t = state[0]
            c_t = state[1]

        outputs = []
        for t in range(max_seq_len):
            reset_mask = (reset_seq[:, t, 0] > 0.5).unsqueeze(-1)
            h_t = torch.where(reset_mask, torch.zeros_like(h_t), h_t)
            c_t = torch.where(reset_mask, torch.zeros_like(c_t), c_t)
            h_t, c_t = self.lstm_cell(actor_obs_seq[:, t, :], (h_t, c_t))
            outputs.append(h_t)

        lstm_out = torch.stack(outputs, dim=1).reshape(-1, self.lstm_hidden_dim)
        policy_feat = self.lstm_norm(lstm_out)
        new_state = [h_t, c_t]

        social_feat = None
        obstacle_feat = None
        graph_feat = torch.zeros_like(policy_feat)
        ego_context = self.social_ego_encoder(self.local_input_norm(local_obs_raw))

        if self.actor_graph_mode == "neighbor" and self.max_neighbors > 0 and self.neighbor_dim > 0:
            # neighbor 模式：social 分支来自通信邻居图，obstacle 分支仍来自预测 token。
            # 关键：必须尊重 graph_ablation，否则三种消融跑的是同一个网络。
            neighbor_obs_raw = neighbor_obs_flat.view(
                batch_size,
                self.max_neighbors,
                self.neighbor_feature_dim,
            )
            if self.use_social_graph_tokens:
                social_feat = self._encode_neighbor_social_branch(ego_context, neighbor_obs_raw)
            if self.use_obstacle_graph_tokens:
                obstacle_feat = self._encode_obstacle_graph_branch(ego_context, local_obs_raw)
            graph_feat = self._fuse_graph_features(social_feat, obstacle_feat)
        else:
            if self.use_social_graph_tokens:
                social_feat = self._encode_social_graph_branch(ego_context, local_obs_raw)
            if self.use_obstacle_graph_tokens:
                obstacle_feat = self._encode_obstacle_graph_branch(ego_context, local_obs_raw)
            graph_feat = self._fuse_graph_features(social_feat, obstacle_feat)

        graph_gate = self.graph_gate(torch.cat([policy_feat, graph_feat], dim=-1))
        fused_policy = self.policy_fusion_norm(
            policy_feat + graph_gate * self.graph_delta(graph_feat)
        )

        action_out = self.actor_head(fused_policy)
        action_out = torch.clamp(action_out, -10.0, 10.0)
        if torch.isnan(action_out).any():
            print("⚠️ WARNING: NaN in action output! Replacing with zeros.")
            action_out = torch.where(torch.isnan(action_out), torch.zeros_like(action_out), action_out)

        global_obs_flat = obs[:, self.global_state_raw_offset:self.global_state_raw_offset + self.global_state_dim]
        global_obs = global_obs_flat.view(batch_size, self.num_agents, self.base_obs_dim)
        self._cur_value = self._critic_from_global_obs(global_obs)
        return action_out, new_state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._cur_value is not None, "必须先调用 forward()"
        return self._cur_value

    @override(TorchModelV2)
    def get_initial_state(self) -> List[TensorType]:
        return [
            torch.zeros(self.lstm_hidden_dim, dtype=torch.float32),
            torch.zeros(self.lstm_hidden_dim, dtype=torch.float32),
        ]

    def _critic_from_global_obs(
        self,
        global_obs: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = global_obs.shape[0]

        if self.critic_mode == "gat":
            if node_mask is None:
                node_mask = torch.ones(
                    batch_size,
                    self.num_agents,
                    device=global_obs.device,
                    dtype=global_obs.dtype,
                )
            else:
                node_mask = node_mask.to(device=global_obs.device, dtype=global_obs.dtype)

            node_feats = self.central_node_encoder(
                global_obs.reshape(-1, self.base_obs_dim)
            ).view(batch_size, self.num_agents, -1)
            node_feats = node_feats * node_mask.unsqueeze(-1)

            full_adj = torch.ones(
                batch_size,
                self.num_agents,
                self.num_agents,
                device=global_obs.device,
                dtype=global_obs.dtype,
            )
            full_adj = full_adj * (node_mask.unsqueeze(1) * node_mask.unsqueeze(2))

            gat_out = self.central_gat(node_feats, full_adj)
            gat_out = gat_out * node_mask.unsqueeze(-1)

            active_count = node_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled_mean = gat_out.sum(dim=1) / active_count
            if self.use_max_pool_critic:
                masked_out = gat_out.masked_fill(node_mask.unsqueeze(-1) <= 0.0, float("-inf"))
                pooled_max = masked_out.max(dim=1).values
                pooled_max = torch.where(
                    torch.isfinite(pooled_max),
                    pooled_max,
                    torch.zeros_like(pooled_max),
                )
                pooled = torch.cat([pooled_mean, pooled_max], dim=-1)
            else:
                pooled = pooled_mean
        else:
            if node_mask is not None:
                global_obs = global_obs * node_mask.unsqueeze(-1).to(global_obs.dtype)
            global_obs_flat = global_obs.reshape(batch_size, -1)
            pooled = self.critic_net(global_obs_flat)

        value = self.central_critic_head(pooled).squeeze(-1)
        return torch.clamp(value, -100.0, 100.0)

    def compute_counterfactual_values(
        self,
        obs: torch.Tensor,
        agent_indices: torch.Tensor,
    ) -> torch.Tensor:
        """按样本屏蔽当前 agent 的全局节点，估计 leave-one-out V(s\\i)。"""
        obs = obs.float()
        batch_size = obs.shape[0]
        if batch_size == 0:
            return obs.new_zeros((0,))

        global_obs_flat = obs[:, self.global_state_raw_offset:self.global_state_raw_offset + self.global_state_dim]
        global_obs = global_obs_flat.view(batch_size, self.num_agents, self.base_obs_dim)

        if self.num_agents <= 1:
            return self._critic_from_global_obs(global_obs)

        node_mask = torch.ones(
            batch_size,
            self.num_agents,
            device=obs.device,
            dtype=obs.dtype,
        )
        row_index = torch.arange(batch_size, device=obs.device)
        masked_agent_idx = (
            agent_indices.to(device=obs.device).long().clamp_(0, self.num_agents - 1)
        )
        node_mask[row_index, masked_agent_idx] = 0.0
        return self._critic_from_global_obs(global_obs, node_mask=node_mask)


MODEL_NAME = "gat_model"
