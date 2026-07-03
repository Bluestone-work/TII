#!/bin/bash
# 一键测试6个训练好的stage环境,自动保存GIF轨迹动画
set -euo pipefail

WORKSPACE="/home/wj/work/multi-robot-exploration-rl"
PKG_DIR="$WORKSPACE/src/gnn_marl_training_DGTA_nobuffer"
GIF_BASE="$WORKSPACE/test_results/stage_gifs"

cd "$PKG_DIR"

echo "========================================="
echo "  6 Stage 环境测试 + GIF 轨迹动画保存"
echo "========================================="
echo ""
echo "自动查找 ray_results/ 下6个stage的checkpoint"
echo "每个stage测试 5 episodes,保存轨迹GIF到 $GIF_BASE"
echo ""
read -p "按 Enter 开始,或 Ctrl+C 取消..."

mkdir -p "$GIF_BASE"

# 用 --all_stages 模式一次性跑6个
./run_test.sh \
    -c dummy \
    --all_stages \
    --num_episodes 5 \
    --explore \
    --save_gif \
    2>&1 | tee "$WORKSPACE/test_results/test_all_stages.log"

echo ""
echo "========================================="
echo "  测试完成!"
echo "========================================="
echo "日志: $WORKSPACE/test_results/test_all_stages.log"
echo "GIF: $GIF_BASE/gifs_stage*/*.gif"
echo ""
echo "查看GIF:"
echo "  ls -lh $GIF_BASE/gifs_stage*/*.gif"
