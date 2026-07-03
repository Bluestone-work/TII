#!/bin/bash
# 快速GIF生成脚本 - 各stage用自己的模型,每个只测1个episode
set -euo pipefail

WORKSPACE="/home/wj/work/multi-robot-exploration-rl"
PKG_DIR="$WORKSPACE/src/gnn_marl_training_DGTA_nobuffer"
GIF_DIR="$WORKSPACE/test_results/quick_gifs"

NUM_EPISODES="${1:-1}"  # 默认1个episode,可通过参数覆盖

cd "$PKG_DIR"

echo "========================================="
echo "  快速GIF生成 (各stage自己的模型)"
echo "========================================="
echo "每个stage: ${NUM_EPISODES} episode"
echo "预计时长: ~$((NUM_EPISODES * 2)) 分钟/stage × 6 = ~$((NUM_EPISODES * 12)) 分钟"
echo ""
echo "GIF保存到: $GIF_DIR/gifs_stage*/"
echo ""
read -p "按 Enter 开始,或 Ctrl+C 取消..."

mkdir -p "$GIF_DIR"

./run_test.sh \
    --all_stages \
    --num_episodes "$NUM_EPISODES" \
    --save_gif \
    2>&1 | tee "$GIF_DIR/quick_test.log"

echo ""
echo "========================================="
echo "  快速测试完成!"
echo "========================================="
echo ""
echo "生成的GIF:"
find "$WORKSPACE/test_results/stage_gifs" -name "*.gif" -type f -printf "  %p  (%s bytes)\n" | sort

echo ""
echo "查看GIF:"
echo "  cd $WORKSPACE/test_results/stage_gifs"
echo "  ls -lh gifs_stage*/*.gif"
echo ""
echo "指标摘要:"
grep -E "Stage [1-6].*测试环境|总计到达|总计碰撞|平均步数" "$GIF_DIR/quick_test.log" | tail -30
