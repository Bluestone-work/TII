"""
Advanced Critic Network based on 2025 SOTA papers

实现了以下改进：
1. 层次化图注意力（Hierarchical GAT）- 2025 MDPI paper
2. 更深的网络结构（4层 + 残差连接 + LayerNorm）
3. 自适应正则化（Adaptive Regularization）
4. 双 Critic 架构（TD3 思想）
5. Value Clipping with Huber Loss

参考论文：
- Multi-Agent Hierarchical Graph Attention Actor–Critic (2025)
- Adaptive Regularized Multi-Agent Soft Actor-Critic (2024)
- Actor-Attention-Critic for Multi-Agent RL (2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadAttentionPooling(nn.Module):
    """
    多头注意力池化层
    用于从多个智能体的特征中提取全局特征

    基于 Actor-Attention-Critic (AACC) 思想
    """
    def __init__(self, feature_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        assert feature_dim % num_heads == 0

        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, num_agents, feature_dim]
            mask: [batch, num_agents] (1 for valid, 0 for masked)

        Returns:
            out: [batch, feature_dim] - 全局特征
        """
        batch_size, num_agents, feature_dim = x.shape

        # Multi-head projection
        Q = self.query(x).view(batch_size, num_agents, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, num_agents, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, num_agents, self.num_heads, self.head_dim)

        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch, heads, num_agents, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            # mask: [batch, num_agents] -> [batch, 1, 1, num_agents]
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, V)  # [batch, heads, num_agents, head_dim]

        # Global pooling (mean over agents)
        out = out.mean(dim=2)  # [batch, heads, head_dim]

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, feature_dim)

        return self.out_proj(out)


class HierarchicalGATLayer(nn.Module):
    """
    层次化图注意力层

    基于 Multi-Agent Hierarchical GAT Actor-Critic (2025)

    两层注意力：
    1. Agent-level: 智能体间的局部交互
    2. Global-level: 全局聚合
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.residual = residual

        # Agent-level attention
        self.W_agent = nn.Linear(in_features, out_features * num_heads)
        self.a_agent = nn.Parameter(torch.zeros(1, num_heads, 2 * out_features))
        nn.init.xavier_uniform_(self.a_agent)

        # Layer normalization
        self.ln1 = nn.LayerNorm(out_features * num_heads)

        # Global attention pooling
        self.global_attn = MultiHeadAttentionPooling(
            out_features * num_heads,
            num_heads=num_heads,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

        # Residual projection
        if residual and in_features != out_features * num_heads:
            self.residual_proj = nn.Linear(in_features, out_features * num_heads)
        else:
            self.residual_proj = None

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, num_agents, in_features]
            adj: [batch, num_agents, num_agents] (邻接矩阵)
            mask: [batch, num_agents] (节点mask)

        Returns:
            node_features: [batch, num_agents, out_features * num_heads]
            global_feature: [batch, out_features * num_heads]
        """
        batch_size, num_agents, _ = x.shape

        # Agent-level attention
        Wx = self.W_agent(x)  # [batch, num_agents, out * heads]
        Wx = Wx.view(batch_size, num_agents, self.num_heads, self.out_features)

        # Compute attention scores
        # [batch, num_agents, 1, heads, out]
        Wx_i = Wx.unsqueeze(2)
        # [batch, 1, num_agents, heads, out]
        Wx_j = Wx.unsqueeze(1)

        # Concatenate and compute scores
        # [batch, num_agents, num_agents, heads, 2*out]
        concat = torch.cat([
            Wx_i.expand(-1, -1, num_agents, -1, -1),
            Wx_j.expand(-1, num_agents, -1, -1, -1)
        ], dim=-1)

        # [batch, num_agents, num_agents, heads]
        e = torch.matmul(concat, self.a_agent.transpose(-2, -1)).squeeze(-1)
        e = F.leaky_relu(e, 0.2)

        # Mask invalid edges
        adj_mask = adj.unsqueeze(-1).expand(-1, -1, -1, self.num_heads)  # [batch, n, n, heads]
        e = e.masked_fill(adj_mask == 0, -1e9)

        # Softmax over neighbors
        alpha = F.softmax(e, dim=2)  # [batch, n, n, heads]
        alpha = self.dropout(alpha)

        # Aggregate
        # [batch, n, n, heads, 1] * [batch, 1, n, heads, out]
        # -> [batch, n, n, heads, out] -> sum over dim=2
        alpha_expanded = alpha.unsqueeze(-1)
        Wx_expanded = Wx.unsqueeze(1)
        h = (alpha_expanded * Wx_expanded).sum(dim=2)

        # Reshape: [batch, n, heads, out] -> [batch, n, heads*out]
        h = h.view(batch_size, num_agents, -1)

        # Layer norm
        h = self.ln1(h)

        # Residual connection
        if self.residual:
            if self.residual_proj is not None:
                h = h + self.residual_proj(x)
            else:
                h = h + x

        # Global pooling
        global_feat = self.global_attn(h, mask)

        return h, global_feat


class DeepResidualMLP(nn.Module):
    """
    深度残差 MLP

    4 层 + 残差连接 + LayerNorm
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # Input projection for residual
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection
        if self.input_proj is not None:
            identity = self.input_proj(x)
        else:
            identity = x

        # Layer 1
        out = F.relu(self.ln1(self.fc1(x)))
        out = self.dropout(out)

        # Layer 2 with residual
        out2 = F.relu(self.ln2(self.fc2(out)))
        out2 = self.dropout(out2)
        out = out + out2  # Residual

        # Layer 3 with residual
        out3 = F.relu(self.ln3(self.fc3(out)))
        out3 = self.dropout(out3)
        out = out + out3  # Residual

        # Layer 4
        out = F.relu(self.ln4(self.fc4(out)))
        out = self.dropout(out)

        # Final residual from input
        out = out + identity

        return out


class HierarchicalGATCritic(nn.Module):
    """
    层次化图注意力 Critic 网络

    架构：
    1. Node encoder (per-agent features)
    2. 2-layer Hierarchical GAT
    3. Deep Residual MLP
    4. Value head with clipping

    基于 2025 年 MDPI 论文的层次化 GAT 设计
    """
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        num_agents: int = 4,
        num_heads: int = 4,
        num_gat_layers: int = 2,
        dropout: float = 0.1,
        use_dual_critic: bool = False,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents
        self.use_dual_critic = use_dual_critic

        # Node encoder (encode individual agent observations)
        self.node_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Hierarchical GAT layers
        self.gat_layers = nn.ModuleList([
            HierarchicalGATLayer(
                in_features=hidden_dim if i == 0 else hidden_dim * num_heads,
                out_features=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                residual=True,
            )
            for i in range(num_gat_layers)
        ])

        # Deep residual MLP
        mlp_input_dim = hidden_dim * num_heads
        self.value_mlp = DeepResidualMLP(
            input_dim=mlp_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

        # Dual critic (optional, for TD3-style training)
        if use_dual_critic:
            self.value_mlp2 = DeepResidualMLP(
                input_dim=mlp_input_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            self.value_head2 = nn.Linear(hidden_dim, 1)
            nn.init.orthogonal_(self.value_head2.weight, gain=1.0)
            nn.init.constant_(self.value_head2.bias, 0.0)

    def forward(
        self,
        global_obs: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        return_both: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            global_obs: [batch, num_agents, obs_dim] or [batch, num_agents * obs_dim]
            node_mask: [batch, num_agents] (1 for valid, 0 for masked)
            return_both: 如果是 dual critic，是否返回两个值

        Returns:
            value: [batch] - 状态价值
        """
        batch_size = global_obs.shape[0]

        # Reshape if flattened
        if global_obs.dim() == 2:
            global_obs = global_obs.view(batch_size, self.num_agents, self.obs_dim)

        # Default mask (all agents valid)
        if node_mask is None:
            node_mask = torch.ones(
                batch_size, self.num_agents,
                device=global_obs.device,
                dtype=torch.float32
            )

        # Encode individual agent observations
        node_features = self.node_encoder(
            global_obs.reshape(-1, self.obs_dim)
        ).view(batch_size, self.num_agents, self.hidden_dim)

        # Apply node mask
        node_features = node_features * node_mask.unsqueeze(-1)

        # Full adjacency (all-to-all communication)
        adj = torch.ones(
            batch_size, self.num_agents, self.num_agents,
            device=global_obs.device
        )

        # Hierarchical GAT layers
        for gat_layer in self.gat_layers:
            node_features, global_feature = gat_layer(
                node_features, adj, node_mask
            )

        # Use final global feature
        # global_feature: [batch, hidden_dim * num_heads]

        # Deep MLP
        h = self.value_mlp(global_feature)

        # Value head
        value = self.value_head(h).squeeze(-1)

        # Clip value to prevent explosion
        value = torch.clamp(value, -100.0, 100.0)

        # Dual critic
        if self.use_dual_critic and return_both:
            h2 = self.value_mlp2(global_feature)
            value2 = self.value_head2(h2).squeeze(-1)
            value2 = torch.clamp(value2, -100.0, 100.0)
            return value, value2

        return value

    def get_min_value(self, global_obs: torch.Tensor, node_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        获取两个 Critic 的最小值（TD3 思想）
        """
        if not self.use_dual_critic:
            return self.forward(global_obs, node_mask)

        v1, v2 = self.forward(global_obs, node_mask, return_both=True)
        return torch.min(v1, v2)


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 10.0) -> torch.Tensor:
    """
    Huber Loss（对离群值鲁棒）

    Args:
        pred: 预测值
        target: 目标值
        delta: Huber delta 参数（控制何时从平方变为线性）

    Returns:
        loss: Huber loss
    """
    error = pred - target
    abs_error = torch.abs(error)

    quadratic = torch.where(
        abs_error < delta,
        0.5 * error ** 2,
        torch.zeros_like(error)
    )
    linear = torch.where(
        abs_error >= delta,
        delta * (abs_error - 0.5 * delta),
        torch.zeros_like(error)
    )

    return (quadratic + linear).mean()


class AdaptiveVFCoeffScheduler:
    """
    自适应 VF Loss 权重调度器

    根据 VF Loss 和 Policy Loss 的比值动态调整权重
    """
    def __init__(
        self,
        init_coeff: float = 1.0,
        min_coeff: float = 0.1,
        max_coeff: float = 2.0,
        target_ratio: float = 100.0,
    ):
        self.coeff = init_coeff
        self.min_coeff = min_coeff
        self.max_coeff = max_coeff
        self.target_ratio = target_ratio

    def update(self, vf_loss: float, policy_loss: float) -> float:
        """
        根据当前 loss 比值更新权重

        Args:
            vf_loss: Value function loss
            policy_loss: Policy loss (取绝对值)

        Returns:
            new_coeff: 更新后的权重
        """
        ratio = vf_loss / max(abs(policy_loss), 1e-6)

        if ratio > self.target_ratio * 2:
            # VF Loss 太大，降低权重
            self.coeff *= 0.9
        elif ratio < self.target_ratio * 0.5:
            # VF Loss 合理，可以增加权重
            self.coeff *= 1.05

        self.coeff = max(self.min_coeff, min(self.max_coeff, self.coeff))
        return self.coeff


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 测试 Hierarchical GAT Critic
    batch_size = 16
    num_agents = 4
    obs_dim = 80

    # 创建模型
    critic = HierarchicalGATCritic(
        obs_dim=obs_dim,
        hidden_dim=256,
        num_agents=num_agents,
        num_heads=4,
        num_gat_layers=2,
        dropout=0.1,
        use_dual_critic=True,
    )

    # 测试数据
    global_obs = torch.randn(batch_size, num_agents, obs_dim)
    node_mask = torch.ones(batch_size, num_agents)
    node_mask[:, -1] = 0  # Mask last agent

    # Forward pass
    value = critic(global_obs, node_mask)
    print(f"Value shape: {value.shape}")  # [batch_size]
    print(f"Value range: [{value.min():.2f}, {value.max():.2f}]")

    # Dual critic
    v1, v2 = critic(global_obs, node_mask, return_both=True)
    print(f"Dual values: V1 range [{v1.min():.2f}, {v1.max():.2f}], "
          f"V2 range [{v2.min():.2f}, {v2.max():.2f}]")

    # Min value (TD3 style)
    v_min = critic.get_min_value(global_obs, node_mask)
    print(f"Min value range: [{v_min.min():.2f}, {v_min.max():.2f}]")

    # 测试 Huber Loss
    pred = torch.randn(batch_size) * 20
    target = torch.randn(batch_size) * 15

    mse_loss = F.mse_loss(pred, target)
    huber = huber_loss(pred, target, delta=10.0)

    print(f"\nLoss comparison:")
    print(f"  MSE Loss: {mse_loss:.4f}")
    print(f"  Huber Loss: {huber:.4f}")

    # 测试自适应权重调度
    scheduler = AdaptiveVFCoeffScheduler(init_coeff=1.0)

    print(f"\nAdaptive VF Coeff Scheduler:")
    for i in range(10):
        vf_loss = 5.0 + i * 0.5  # 模拟 VF Loss 增长
        policy_loss = -0.01
        new_coeff = scheduler.update(vf_loss, policy_loss)
        print(f"  Iter {i}: VF Loss={vf_loss:.2f}, Coeff={new_coeff:.3f}")

    # 参数统计
    total_params = sum(p.numel() for p in critic.parameters())
    trainable_params = sum(p.numel() for p in critic.parameters() if p.requires_grad)

    print(f"\nModel statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
