#!/bin/bash
# 试点2: 图结构消融 — social_only vs dual_graph
# Stage2(4车), 单种子, 快速验证。注意: gat_actor_graph 必须为 local_risk,否则 graph_ablation 失效(见 MEMORY bug)
set -euo pipefail

WORKSPACE="/home/wj/work/multi-robot-exploration-rl"
PKG_DIR="$WORKSPACE/src/gnn_marl_training_DGTA_nobuffer"
EXP_DIR="$WORKSPACE/experiments/pilot_ablation"
SEED=42
TRAIN_STEPS=50000

cd "$PKG_DIR"

echo "========================================="
echo "试点2: 图结构消融 (social_only vs dual_graph)"
echo "  Stage2 (4车), seed=$SEED, steps=$TRAIN_STEPS"
echo "  ⚠️ gat_actor_graph=local_risk (避开 graph_ablation 失效 bug)"
echo "========================================="

for ABLATION in dual_graph social_only; do
    echo ""
    echo ">>> 图消融变体: $ABLATION 训练中..."
    TRAIN_VERBOSE=0 ENV_VERBOSE=0 RAY_memory_usage_threshold=0.97 \
    ./run_curriculum.sh \
        --model_type gat \
        --graph_ablation "$ABLATION" \
        --gat_actor_graph local_risk \
        --start_stage 2 --end_stage 2 \
        --num_agents 4 \
        --num_workers 1 \
        --train_steps $TRAIN_STEPS \
        --seed $SEED \
        --run_suffix "pilot_abl_${ABLATION}_seed${SEED}" \
        2>&1 | tee "$EXP_DIR/logs/${ABLATION}_seed${SEED}.log"
done

echo ""
echo "✓ 试点2 完成。结果 CSV 在 ray_results/pilot_abl_*/"
