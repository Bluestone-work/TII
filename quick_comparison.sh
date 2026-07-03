#!/bin/bash
# 用Stage6最终模型测试所有6个stage环境
# 验证最终模型的泛化能力
set -euo pipefail

WORKSPACE="/home/wj/work/multi-robot-exploration-rl"
PKG_DIR="$WORKSPACE/src/gnn_marl_training_DGTA_nobuffer"
RESULTS_DIR="$WORKSPACE/test_results"

NUM_EPISODES=3  # 每stage测3次,增加统计可靠性

cd "$PKG_DIR"

echo "========================================="
echo "  Stage6最终模型 - 全Stage测试"
echo "========================================="
echo ""
echo "用Stage6最终模型测试所有6个stage环境"
echo "共 6 stages × ${NUM_EPISODES} episodes = $(( 6 * NUM_EPISODES )) episodes"
echo "预计时长: ~20-30 分钟"
echo ""
echo "测试模式: deterministic (无探索噪声)"
echo "GIF保存: test_results/final_model_gifs/gifs_stage*/"
echo ""
read -p "按 Enter 开始,或 Ctrl+C 取消..."

mkdir -p "$RESULTS_DIR/final_model_gifs"

STAGE6_MODEL="$WORKSPACE/ray_results/GNN_MAPPO_Stage2_Cont_EnvStage6/best"
if [[ ! -d "$STAGE6_MODEL" ]]; then
    echo "❌ Stage6模型不存在: $STAGE6_MODEL"
    exit 1
fi

echo ""
echo "========================================="
echo "  开始测试 (使用Stage6最终模型)"
echo "========================================="
./run_test.sh \
    --all_stages \
    --num_episodes "$NUM_EPISODES" \
    --save_gif \
    --final_model "$STAGE6_MODEL" \
    2>&1 | tee "$RESULTS_DIR/final_model_gifs/test_log.txt"

echo ""
echo "========================================="
echo "  测试完成!"
echo "========================================="
echo ""
echo "GIF保存位置:"
find "$RESULTS_DIR" -name "*.gif" -newer "$RESULTS_DIR/final_model_gifs/test_log.txt" 2>/dev/null | head -20
echo ""
echo "关键指标:"
grep -E "测试 Stage|总计到达|总计碰撞|平均步数|平均回报" "$RESULTS_DIR/final_model_gifs/test_log.txt" | tail -30
