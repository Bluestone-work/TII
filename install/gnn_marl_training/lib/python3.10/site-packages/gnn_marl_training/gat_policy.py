"""
Graph Attention Policy Network
基于图注意力机制的策略网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class GraphAttentionLayer(nn.Module):
    """
    单层图注意力机制
    
    参考: Veličković et al. "Graph Attention Networks" (ICLR 2018)
    """
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # 可学习参数
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            h: 节点特征 [batch, n_nodes, in_features]
            adj: 邻接矩阵 [batch, n_nodes, n_nodes] (0/1)
            attention_bias: 可选的先验注意力偏置 [batch, n_nodes, n_nodes]
        
        Returns:
            h_prime: 更新后的节点特征 [batch, n_nodes, out_features]
        """
        batch_size, n_nodes, _ = h.size()
        
        # 线性变换: [batch, n_nodes, in_features] @ [in_features, out_features]
        #           = [batch, n_nodes, out_features]
        Wh = torch.matmul(h, self.W)  
        
        # 计算注意力系数
        # 1. 拼接特征: 对于每条边 (i, j)，拼接 [Wh_i || Wh_j]
        #    [batch, n_nodes, out_features] -> [batch, n_nodes, 1, out_features]
        Wh_i = Wh.unsqueeze(2)  # broadcast source
        Wh_j = Wh.unsqueeze(1)  # broadcast target
        
        # 2. 拼接并计算得分
        #    [batch, n_nodes, n_nodes, 2*out_features]
        concat = torch.cat([Wh_i.expand(-1, -1, n_nodes, -1),
                           Wh_j.expand(-1, n_nodes, -1, -1)], dim=-1)
        
        # 3. 注意力得分: [batch, n_nodes, n_nodes, 2*out_features] @ [2*out_features, 1]
        #                = [batch, n_nodes, n_nodes, 1] -> [batch, n_nodes, n_nodes]
        e = self.leakyrelu(torch.matmul(concat, self.a).squeeze(-1))
        
        if attention_bias is not None:
            e = e + attention_bias

        # 4. Mask: 只保留有边的注意力
        #    adj==0 的位置用 -1e9 (softmax后接近0)
        mask = (adj == 0)
        e = e.masked_fill(mask, -1e9)
        
        # 5. Softmax归一化 (按行，即每个节点对所有邻居归一化)
        attention = F.softmax(e, dim=2)  # [batch, n_nodes, n_nodes]
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 6. 聚合邻居特征: [batch, n_nodes, n_nodes] @ [batch, n_nodes, out_features]
        #                  = [batch, n_nodes, out_features]
        h_prime = torch.matmul(attention, Wh)
        
        return h_prime


class MultiHeadGATLayer(nn.Module):
    """
    多头图注意力层
    """
    
    def __init__(self, in_features: int, out_features: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.out_features = out_features
        
        # 多个注意力头
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, dropout=dropout)
            for _ in range(n_heads)
        ])
    
    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        多头注意力 + 拼接
        
        Returns: [batch, n_nodes, n_heads * out_features]
        """
        h_list = [att(h, adj, attention_bias=attention_bias) for att in self.attentions]
        h_multi = torch.cat(h_list, dim=-1)  # 拼接多头
        return h_multi


class GATEncoder(nn.Module):
    """
    GAT 编码器: 2层 GAT
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_heads: int = 4):
        super().__init__()
        
        # 第一层: multi-head attention
        self.gat1 = MultiHeadGATLayer(input_dim, hidden_dim, n_heads=n_heads)
        
        # 第二层: single-head attention (聚合)
        self.gat2 = GraphAttentionLayer(hidden_dim * n_heads, hidden_dim)
        
        self.output_dim = hidden_dim
    
    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        图编码
        
        Args:
            node_features: [batch, n_nodes, input_dim]
            adj: [batch, n_nodes, n_nodes]
        
        Returns:
            encoded: [batch, n_nodes, output_dim]
        """
        # 第一层
        h = self.gat1(node_features, adj, attention_bias=attention_bias)
        h = F.elu(h)

        # 第二层
        h = self.gat2(h, adj, attention_bias=attention_bias)
        h = F.elu(h)
        
        return h


class GATPolicy(nn.Module):
    """
    完整的 GAT-based 策略网络（集中式多智能体版本）

    ⚠️  注意：此类在 RLlib 训练流程中【未被使用】。
    实际训练使用的是 gat_rllib_model.py 中的 GATRLlibModel（去中心化，每智能体独立前向传播）。
    本类保留仅供研究对比：集中式 GATPolicy 一次处理全部智能体 [batch, n_agents, obs_dim]，
    而 GATRLlibModel 每次只处理单个智能体 [batch, obs_dim]。

    架构:
    1. 特征编码器 (MLP): 提取每个机器人的局部特征
    2. GAT编码器: 聚合邻居信息
    3. LSTM: 时序记忆
    4. Actor/Critic 头
    """
    
    def __init__(
        self, 
        obs_dim: int, 
        action_dim: int = 2,
        hidden_dim: int = 128,
        gat_hidden_dim: int = 128,
        lstm_hidden_dim: int = 256,
        n_gat_heads: int = 4
    ):
        super().__init__()
        
        # 1. 局部特征编码器 (MLP)
        self.feature_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 2. GAT编码器 (聚合邻居信息)
        self.gat_encoder = GATEncoder(
            input_dim=hidden_dim,
            hidden_dim=gat_hidden_dim,
            n_heads=n_gat_heads
        )
        
        # 3. 融合层 (局部特征 + 图特征)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + gat_hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 4. LSTM (时序记忆)
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
        
        # 5. Actor 头 (策略)
        self.actor_mean = nn.Linear(lstm_hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # 6. Critic 头 (价值函数)
        self.critic = nn.Linear(lstm_hidden_dim, 1)
        
        self.lstm_hidden_dim = lstm_hidden_dim
    
    def forward(
        self, 
        obs_batch: torch.Tensor,      # [batch, n_agents, obs_dim]
        adj_batch: torch.Tensor,       # [batch, n_agents, n_agents]
        lstm_state: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """
        前向传播
        
        Args:
            obs_batch: 观测 [batch, n_agents, obs_dim]
            adj_batch: 邻接矩阵 [batch, n_agents, n_agents]
            lstm_state: LSTM状态 (h, c)，每个 [1, batch*n_agents, lstm_hidden_dim]
        
        Returns:
            action_mean: 动作均值 [batch*n_agents, action_dim]
            value: 价值 [batch*n_agents, 1]
            new_lstm_state: 新的LSTM状态
        """
        batch_size, n_agents, obs_dim = obs_batch.size()
        
        # 1. 局部特征编码
        local_features = self.feature_encoder(obs_batch)  # [batch, n_agents, hidden_dim]
        
        # 2. 图注意力聚合
        graph_features = self.gat_encoder(local_features, adj_batch)  # [batch, n_agents, gat_hidden_dim]
        
        # 3. 融合局部特征和图特征
        fused = torch.cat([local_features, graph_features], dim=-1)  # [batch, n_agents, hidden_dim + gat_hidden_dim]
        fused = self.fusion(fused)  # [batch, n_agents, hidden_dim]
        
        # 4. 展平为 [batch*n_agents, hidden_dim] 用于 LSTM
        fused_flat = fused.view(batch_size * n_agents, 1, -1)  # [batch*n_agents, 1, hidden_dim]
        
        # 5. LSTM
        if lstm_state is None:
            lstm_out, new_state = self.lstm(fused_flat)
        else:
            lstm_out, new_state = self.lstm(fused_flat, lstm_state)
        
        lstm_out = lstm_out.squeeze(1)  # [batch*n_agents, lstm_hidden_dim]
        
        # 6. Actor和Critic
        action_mean = self.actor_mean(lstm_out)  # [batch*n_agents, action_dim]
        value = self.critic(lstm_out)            # [batch*n_agents, 1]
        
        return action_mean, value, new_state
    
    def get_action(
        self, 
        obs: torch.Tensor,           # [n_agents, obs_dim]
        adj: torch.Tensor,            # [n_agents, n_agents]
        lstm_state: Tuple = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        获取动作 (单个batch)
        
        Returns:
            action: [n_agents, action_dim]
            new_lstm_state
        """
        # 添加batch维度
        obs_batch = obs.unsqueeze(0)    # [1, n_agents, obs_dim]
        adj_batch = adj.unsqueeze(0)    # [1, n_agents, n_agents]
        
        with torch.no_grad():
            action_mean, _, new_state = self.forward(obs_batch, adj_batch, lstm_state)
        
        if deterministic:
            action = action_mean
        else:
            # 采样
            std = torch.exp(self.actor_logstd)
            action = torch.normal(action_mean, std)
        
        # 裁剪到归一化动作范围 [-1, 1]（底层环境自动映射为速度）
        action = torch.clamp(action, -1.0, 1.0)
        
        return action, new_state
    
    def init_hidden_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """初始化LSTM隐状态"""
        h = torch.zeros(1, batch_size, self.lstm_hidden_dim)
        c = torch.zeros(1, batch_size, self.lstm_hidden_dim)
        return (h, c)


# ===== 测试代码 =====
if __name__ == "__main__":
    # 测试 GAT 层
    print("测试 GraphAttentionLayer...")
    gat = GraphAttentionLayer(in_features=64, out_features=128)
    
    h = torch.randn(2, 3, 64)  # batch=2, n_nodes=3, features=64
    adj = torch.tensor([
        [[1, 1, 0],
         [1, 1, 1],
         [0, 1, 1]],
        [[1, 1, 1],
         [1, 1, 0],
         [1, 0, 1]]
    ], dtype=torch.float32)
    
    h_out = gat(h, adj)
    print(f"输入: {h.shape}, 输出: {h_out.shape}")
    assert h_out.shape == (2, 3, 128)
    
    # 测试完整策略
    print("\n测试 GATPolicy...")
    policy = GATPolicy(obs_dim=42, action_dim=2)

    obs_batch = torch.randn(4, 3, 42)  # batch=4, n_agents=3, obs_dim=42
    adj_batch = torch.randint(0, 2, (4, 3, 3)).float()
    
    action_mean, value, state = policy(obs_batch, adj_batch)
    print(f"Action mean: {action_mean.shape}, Value: {value.shape}")
    assert action_mean.shape == (12, 2)  # 4*3 = 12
    assert value.shape == (12, 1)
    
    print("\n✅ 所有测试通过！")
