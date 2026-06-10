"""
自定义 RLlib 模型：集成 GAT 策略网络
基于 ego-graph 的图注意力机制
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from gnn_marl_training.gat_policy import GATEncoder


class GATRLlibModel(TorchModelV2, nn.Module):
    """
    RLlib 自定义模型：基于 GAT 的多智能体策略
    
    架构:
    1. Ego节点编码器 (处理自身观测)
    2. 邻居节点编码器 (处理每个邻居的特征)
    3. GAT 编码器 (在ego-graph上聚合邻居信息)
    4. 融合层 (局部特征 + 图注意力特征)
    5. LSTM (时序记忆)
    6. Actor-Critic 头
    
    关键: 从观测中提取邻居特征，构建ego-centric图，
    使用图注意力机制学习对不同邻居的关注度
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
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        
        custom_cfg = model_config.get('custom_model_config', {})

        # 从配置获取参数
        self.num_agents = int(custom_cfg.get('num_agents', 2))
        self.neighbor_feature_dim = 5  # 每个邻居: 相对位置(2)+相对速度(2)+距离(1)
        self.max_neighbors = int(custom_cfg.get('max_neighbors', self.num_agents - 1))
        self.reset_flag_dim = 1
        
        # 观测空间分解
        self.obs_dim = obs_space.shape[0]
        self.neighbor_dim = self.neighbor_feature_dim * self.max_neighbors
        # 从 obs_space 自动推断 base_obs_dim（与 MAPPOMLPModel 保持一致的推算逻辑）
        # obs 布局: [base_obs(B)] + [neighbor(K*5)] + [reset_flag(1)] + [global(N*B)]
        # => total = B*(1+N) + K*5 + 1  =>  B = (total - K*5 - 1) / (1+N)
        self.base_obs_dim = (self.obs_dim - self.neighbor_dim - self.reset_flag_dim) // (1 + self.num_agents)
        # Actor 实际使用的维度（本体obs + 邻居通信obs）
        self.actor_obs_dim = self.base_obs_dim + self.neighbor_dim
        self.actor_total_dim = self.actor_obs_dim + self.reset_flag_dim
        # 集中式 Critic 的全局状态维度（N × 42，附在观测末尾）
        self.global_state_dim = self.num_agents * self.base_obs_dim
        
        # 网络参数
        hidden_dim = int(custom_cfg.get('hidden_dim', 128))
        gat_hidden_dim = int(custom_cfg.get('gat_hidden_dim', 128))
        lstm_hidden_dim = int(custom_cfg.get('lstm_hidden_dim', 256))
        n_gat_heads = int(custom_cfg.get('n_gat_heads', 4))
        
        # ======= 网络层 =======
        
        # 1. Ego节点编码器: base_obs(40) → hidden_dim
        self.ego_encoder = nn.Sequential(
            nn.Linear(self.base_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 2. 邻居节点编码器: per_neighbor_features(5) → hidden_dim
        #    每个邻居独立编码（共享权重），使GAT能在同维度空间做注意力
        self.neighbor_node_encoder = nn.Sequential(
            nn.Linear(self.neighbor_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 3. GAT编码器: 在ego-graph上做图注意力聚合
        self.gat_encoder = GATEncoder(
            input_dim=hidden_dim,
            hidden_dim=gat_hidden_dim,
            n_heads=n_gat_heads
        )
        
        # 4. 融合层: 局部特征 + GAT输出
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + self.gat_encoder.output_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 5. LSTM
        self.lstm_cell = nn.LSTMCell(hidden_dim, lstm_hidden_dim)
        self.lstm_hidden_dim = lstm_hidden_dim
        
        # 6. Actor 头 (策略)
        self.actor = nn.Linear(lstm_hidden_dim, num_outputs)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0)
        
        # 7. 集中式 Critic（真正的 MAPPO）
        # ┌─ Actor  只看 o_i（去中心化，部署时直接用）
        # └─ Critic 看 s = [o_0, ..., o_{N-1}]（集中式，只在训练时存在）
        #
        # 结构：每个 agent 的 40 维基础 obs → 共享编码器 → 全连接 GAT → 均值聚合 → V(s)
        # 全连接图代表训练时 Critic 拥有所有机器人的完整信息（特权信息）
        self.central_node_encoder = nn.Sequential(
            nn.Linear(self.base_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.central_gat = GATEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_heads=n_gat_heads
        )
        self.central_critic_head = nn.Linear(hidden_dim, 1)
        nn.init.orthogonal_(self.central_critic_head.weight, gain=1.0)
        nn.init.constant_(self.central_critic_head.bias, 0)

        # 内部状态
        self._cur_value = None
    
    def _build_ego_graph(
        self,
        ego_features: torch.Tensor,    # [batch, hidden_dim]
        neighbor_obs: torch.Tensor      # [batch, max_neighbors, neighbor_feature_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建 ego-centric 图并编码节点
        
        图结构 (star topology):
        - 中心节点: ego (自身)
        - 叶节点: 通信范围内的邻居
        - 边: ego ↔ 每个有效邻居 + 所有节点自环
        
        Returns:
            graph_nodes: [batch, 1+max_neighbors, hidden_dim]
            adj: [batch, 1+max_neighbors, 1+max_neighbors]
        """
        batch_size = ego_features.shape[0]
        device = ego_features.device
        n_nodes = 1 + self.max_neighbors
        
        # 编码邻居节点 (共享权重)
        neighbor_flat = neighbor_obs.reshape(-1, self.neighbor_feature_dim)
        neighbor_encoded = self.neighbor_node_encoder(neighbor_flat)
        neighbor_features = neighbor_encoded.view(batch_size, self.max_neighbors, -1)
        
        # 拼接 ego + neighbors 作为图节点
        graph_nodes = torch.cat([ego_features.unsqueeze(1), neighbor_features], dim=1)
        
        # 构建邻接矩阵
        adj = torch.zeros(batch_size, n_nodes, n_nodes, device=device)
        
        # Ego节点自环
        adj[:, 0, 0] = 1.0
        
        # 检测有效邻居 (非零特征 = 在通信范围内的邻居)
        neighbor_valid = (neighbor_obs.abs().sum(dim=-1) > 1e-6).float()
        
        # 建立 ego ↔ neighbor 双向边 + neighbor 自环
        for k in range(self.max_neighbors):
            adj[:, 0, k + 1] = neighbor_valid[:, k]
            adj[:, k + 1, 0] = neighbor_valid[:, k]
            adj[:, k + 1, k + 1] = neighbor_valid[:, k]
        
        return graph_nodes, adj
    
    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType
    ) -> Tuple[TensorType, List[TensorType]]:
        """
        前向传播
        
        核心流程:
        1. 解析观测 → ego特征 + 邻居特征
        2. 编码ego节点
        3. 构建ego-graph并应用GAT图注意力
        4. 融合局部特征和图特征
        5. LSTM时序编码
        6. 输出动作分布参数 + 价值估计
        """
        obs = input_dict["obs_flat"]  # [batch, obs_dim]
        batch_size = obs.shape[0]
        
        # ===== 1. 解析观测 =====
        local_obs = obs[:, :self.base_obs_dim]
        neighbor_obs_flat = obs[:, self.base_obs_dim:self.base_obs_dim + self.neighbor_dim]
        reset_flag = obs[:, self.actor_obs_dim:self.actor_total_dim]
        
        # ===== 2. 编码Ego节点 =====
        ego_features = self.ego_encoder(local_obs)  # [batch, hidden_dim]
        
        # ===== 3. 构建Ego-Graph + GAT =====
        if self.max_neighbors > 0 and self.neighbor_dim > 0:
            neighbor_obs = neighbor_obs_flat.view(
                batch_size, self.max_neighbors, self.neighbor_feature_dim
            )
            graph_nodes, adj = self._build_ego_graph(ego_features, neighbor_obs)
            gat_output = self.gat_encoder(graph_nodes, adj)
            ego_gat_output = gat_output[:, 0, :]  # 取ego节点输出
        else:
            ego_gat_output = torch.zeros(
                batch_size, self.gat_encoder.output_dim, device=obs.device
            )
        
        # ===== 4. 融合 =====
        fused = torch.cat([ego_features, ego_gat_output], dim=-1)
        fused = self.fusion(fused)
        
        # ===== 5. LSTM =====
        batch_seqs = seq_lens.shape[0]
        max_seq_len = fused.shape[0] // batch_seqs
        fused_seq = fused.view(batch_seqs, max_seq_len, -1)
        reset_seq = reset_flag.view(batch_seqs, max_seq_len, self.reset_flag_dim)

        if len(state) < 2 or state[0].nelement() == 0 or state[0].shape[0] != batch_seqs:
            h_t = fused.new_zeros(batch_seqs, self.lstm_hidden_dim)
            c_t = fused.new_zeros(batch_seqs, self.lstm_hidden_dim)
        else:
            h_t = state[0]
            c_t = state[1]

        outputs = []
        for t in range(max_seq_len):
            reset_mask = (reset_seq[:, t, 0] > 0.5).unsqueeze(-1)
            h_t = torch.where(reset_mask, torch.zeros_like(h_t), h_t)
            c_t = torch.where(reset_mask, torch.zeros_like(c_t), c_t)
            h_t, c_t = self.lstm_cell(fused_seq[:, t, :], (h_t, c_t))
            outputs.append(h_t)

        lstm_out = torch.stack(outputs, dim=1).reshape(-1, self.lstm_hidden_dim)
        new_state = [h_t, c_t]
        
        # ===== 6. Actor =====
        action_out = self.actor(lstm_out)
        action_out = torch.clamp(action_out, -10.0, 10.0)
        
        # NaN安全检查
        if torch.isnan(action_out).any():
            print("⚠️ WARNING: NaN in action output! Replacing with zeros.")
            action_out = torch.where(
                torch.isnan(action_out), torch.zeros_like(action_out), action_out
            )
        
        # ===== 7. 集中式 Critic (MAPPO) =====
        # 从观测末尾取出全局状态 [B, N*40]，对 Actor 路径完全透明
        global_obs_flat = obs[:, self.actor_total_dim : self.actor_total_dim + self.global_state_dim]
        global_obs = global_obs_flat.view(batch_size, self.num_agents, self.base_obs_dim)
        # 全连接图：集中训练时 Critic 掌握所有机器人的完整信息
        full_adj = torch.ones(batch_size, self.num_agents, self.num_agents, device=obs.device)
        node_feats = self.central_node_encoder(
            global_obs.reshape(-1, self.base_obs_dim)
        ).view(batch_size, self.num_agents, -1)          # [B, N, hidden_dim]
        gat_out = self.central_gat(node_feats, full_adj) # [B, N, hidden_dim]
        pooled  = gat_out.mean(dim=1)                    # [B, hidden_dim] 均值聚合
        self._cur_value = self.central_critic_head(pooled).squeeze(-1)
        self._cur_value = torch.clamp(self._cur_value, -100.0, 100.0)

        return action_out, new_state
    
    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        """返回最后计算的价值"""
        assert self._cur_value is not None, "必须先调用 forward()"
        return self._cur_value
    
    @override(TorchModelV2)
    def get_initial_state(self) -> List[TensorType]:
        """返回 LSTM 初始状态（PyTorch tensor）"""
        # 返回 [(lstm_hidden_dim,), (lstm_hidden_dim,)]
        return [
            torch.zeros(self.lstm_hidden_dim, dtype=torch.float32),
            torch.zeros(self.lstm_hidden_dim, dtype=torch.float32)
        ]


# 模型注册名称
MODEL_NAME = "gat_model"
