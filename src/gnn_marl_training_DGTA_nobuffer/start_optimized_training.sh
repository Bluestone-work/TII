#!/bin/bash
# 快速启动优化后的课程训练
# 使用方法: ./start_optimized_training.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="/home/wj/work/multi-robot-exploration-rl"

echo "========================================="
echo "多机器人导航 - 优化训练启动"
echo "========================================="
echo ""
echo "课程配置："
echo "  Stage 1: 2车 + 3动障 (circle_swap) - 100k steps"
echo "  Stage 2: 4车 + 4动障 (circle_swap) - 300k steps"
echo "  Stage 3: 6车 + 4动障 (circle_swap) - 400k steps  ← 新增缓冲"
echo "  Stage 4: 8车 + 5动障 (circle_swap) - 600k steps"
echo "  Stage 5: 4车 + 3动障 (circle_swap) - 200k steps  ← 新增预热"
echo "  Stage 6: 4车 + 3动障 (intersection) - 400k steps"
echo ""
echo "改进点："
echo "  ✓ 渐进式难度曲线 (避免4→8车的跳跃)"
echo "  ✓ 自适应学习率策略 (3e-4 → 1e-4)"
echo "  ✓ 密度自适应奖励函数 (8车避碰权重↑41%)"
echo ""
echo "预计总训练时间: ~12-16小时 (取决于硬件)"
echo ""

# 询问用户确认
read -p "是否开始训练? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消。"
    exit 0
fi

# 检查必要文件
if [[ ! -f "$SCRIPT_DIR/run_curriculum.sh" ]]; then
    echo "[错误] 找不到 run_curriculum.sh"
    exit 1
fi

echo ""
echo "正在启动训练..."
echo "日志将保存到: $WORKSPACE/curriculum_logs/"
echo "训练结果将保存到: $WORKSPACE/ray_results/"
echo ""

# 启动训练
cd "$SCRIPT_DIR"
./run_curriculum.sh \
    --start_stage 1 \
    --end_stage 6 \
    --model_type gat \
    --gat_actor_graph neighbor \
    --gat_critic_mode mlp \
    --action_mode continuous \
    --graph_ablation dual_graph \
    --ppo_profile auto \
    --num_workers 1 \
    --train_batch_size 5000 \
    --checkpoint_freq 2000 \
    --enable_visualization \
    --tracking_viz_interval 4

echo ""
echo "========================================="
echo "训练完成！"
echo "========================================="
echo ""
echo "检查结果："
echo "  1. 查看训练日志: ls -lh $WORKSPACE/curriculum_logs/"
echo "  2. 查看checkpoint: ls -lh $WORKSPACE/ray_results/"
echo "  3. TensorBoard可视化: tensorboard --logdir=$WORKSPACE/ray_results"
echo ""
echo "下一步："
echo "  1. 评估最佳checkpoint的效果"
echo "  2. 如需微调，从特定Stage恢复训练"
echo "  3. 准备Sim2Real迁移"
echo ""
