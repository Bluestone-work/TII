#!/bin/bash
# GNN-MAPPO 快速验证测试（10个episodes）

echo "=========================================="
echo "🧪 GNN-MAPPO 快速验证"
echo "=========================================="

source /home/wj/anaconda3/bin/activate ros2
source install/setup.sh

echo "运行10个训练episodes以验证配置..."
echo ""

python src/gnn_marl_training/gnn_marl_training/train_gnn_mappo_full.py \
    --num_agents 2 \
    --communication_range 5.0 \
    --num_workers 1 \
    --train_steps 10000 \
    --checkpoint_freq 1 \
    --collision_penalty_weight 2.0 \
    --proximity_penalty_weight 0.5 \
    --cooperation_reward_weight 0.3

echo ""
echo "✅ 快速验证完成！"
echo ""
echo "如果没有错误，可以开始完整训练："
echo "  ./run_curriculum.sh"
