#!/bin/bash
# 测试屏幕录制功能 - Stage2单次测试
set -euo pipefail

cd /home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training_DGTA_nobuffer

echo "========================================="
echo "  屏幕录制测试 (Stage2 × 1 episode)"
echo "========================================="
echo ""
echo "⚠️  请确保:"
echo "  1. Gazebo窗口已打开并可见"
echo "  2. RViz窗口(如需要)已打开"
echo "  3. 窗口位置固定,不要移动"
echo ""
read -p "按 Enter 开始测试..."

./run_test.sh \
    -c ../../ray_results/GNN_MAPPO_Stage2_Cont_EnvStage2/best \
    --test_stage 2 \
    --num_episodes 1 \
    --save_gif \
    --record_screen

echo ""
echo "测试完成!查看结果:"
echo "  轨迹GIF: ls -lh ../../test_results/gifs_stage2/*.gif"
echo "  屏幕GIF: ls -lh ../../test_results/screen_gifs/*.gif"
