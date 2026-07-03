#!/bin/bash
# 课程学习有效性对比实验:
#   实验A — 各stage用自己专属的checkpoint(验证专属性)
#   实验B — 所有stage都用Stage6最终模型(验证泛化能力)
set -euo pipefail

WORKSPACE="/home/wj/work/multi-robot-exploration-rl"
PKG_DIR="$WORKSPACE/src/gnn_marl_training_DGTA_nobuffer"
RESULTS_DIR="$WORKSPACE/test_results"

cd "$PKG_DIR"

echo "========================================="
echo "  课程学习有效性对比实验"
echo "========================================="
echo ""
echo "实验A: 各stage用自己训练的checkpoint"
echo "实验B: 所有stage都用Stage6最终模型"
echo ""
echo "目标: 对比'专属模型'vs'最终模型泛化'的性能差异"
echo ""
echo "每个实验: 6 stages × 5 episodes = 30 episodes"
echo "预计时长: ~60-90分钟(串行)"
echo ""
read -p "按 Enter 开始,或 Ctrl+C 取消..."

mkdir -p "$RESULTS_DIR"/{individual_models,final_model_only}

# 先找Stage6最终模型
STAGE6_MODEL=$(find "$WORKSPACE/ray_results" -type d -name "best" | grep "EnvStage6" | head -1)
if [[ -z "$STAGE6_MODEL" ]]; then
    echo "❌ 未找到Stage6模型,尝试查找最新checkpoint..."
    STAGE6_MODEL=$(find "$WORKSPACE/ray_results" -type d -name "checkpoint_*" | grep "EnvStage6" | sort -V | tail -1)
    if [[ -z "$STAGE6_MODEL" ]]; then
        echo "❌ 完全找不到Stage6模型,无法进行对比实验"
        exit 1
    fi
fi
echo "✓ Stage6最终模型: $STAGE6_MODEL"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
echo "========================================="
echo "  实验A: 各stage用自己的模型"
echo "========================================="
./run_test.sh \
    --all_stages \
    --num_episodes 5 \
    --explore \
    --save_gif \
    2>&1 | tee "$RESULTS_DIR/individual_models/test_log.txt"

echo ""
echo "实验A完成,GIF和指标已保存到: $RESULTS_DIR/individual_models/"
echo ""
read -p "按 Enter 继续实验B,或 Ctrl+C 停止..."

# ──────────────────────────────────────────────────────────────────────────────
echo "========================================="
echo "  实验B: 所有stage都用Stage6最终模型"
echo "========================================="
./run_test.sh \
    --all_stages \
    --num_episodes 5 \
    --explore \
    --save_gif \
    --final_model "$STAGE6_MODEL" \
    2>&1 | tee "$RESULTS_DIR/final_model_only/test_log.txt"

echo ""
echo "实验B完成,GIF和指标已保存到: $RESULTS_DIR/final_model_only/"

# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "========================================="
echo "  对比实验完成!"
echo "========================================="
echo ""
echo "结果对比:"
echo "  实验A(专属模型): $RESULTS_DIR/individual_models/test_log.txt"
echo "  实验B(最终模型): $RESULTS_DIR/final_model_only/test_log.txt"
echo ""
echo "提取关键指标对比:"
echo "─────────────────────────────────────────"
for exp in individual_models final_model_only; do
    log="$RESULTS_DIR/$exp/test_log.txt"
    if [[ -f "$log" ]]; then
        echo "【${exp}】"
        grep -E "总计到达|总计碰撞|平均步数|平均最小间距" "$log" | tail -6
        echo ""
    fi
done

echo "查看GIF轨迹动画:"
echo "  实验A: ls $WORKSPACE/test_results/stage_gifs/gifs_stage*/*.gif"
echo "  实验B: ls $WORKSPACE/test_results/stage_gifs/gifs_stage*/*.gif"
echo ""
echo "结论性分析:"
echo "  若实验A >> 实验B → 课程学习必要,各阶段模型有专属优势"
echo "  若实验B ≈ 实验A  → 最终模型泛化能力强,可跨阶段部署"
