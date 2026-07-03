#!/bin/bash
# 试点3: 参数敏感性 — communication_range 扫描(3点验证流程)
# Stage2(4车), 单种子, 快速验证
set -euo pipefail

WORKSPACE="/home/wj/work/multi-robot-exploration-rl"
PKG_DIR="$WORKSPACE/src/gnn_marl_training_DGTA_nobuffer"
EXP_DIR="$WORKSPACE/experiments/pilot_param_sweep"
SEED=42
TRAIN_STEPS=50000

cd "$PKG_DIR"

echo "========================================="
echo "试点3: 参数敏感性 (communication_range 扫描)"
echo "  Stage2 (4车), seed=$SEED, steps=$TRAIN_STEPS"
echo "  扫描点: 2.0 / 3.5 / 6.0 (米)"
echo "========================================="

for COMM_RANGE in 2.0 3.5 6.0; do
    echo ""
    echo ">>> communication_range=$COMM_RANGE 训练中..."
    TRAIN_VERBOSE=0 ENV_VERBOSE=0 RAY_memory_usage_threshold=0.97 \
    ./run_curriculum.sh \
        --model_type gat \
        --graph_ablation dual_graph \
        --gat_actor_graph local_risk \
        --communication_range "$COMM_RANGE" \
        --start_stage 2 --end_stage 2 \
        --num_agents 4 \
        --num_workers 1 \
        --train_steps $TRAIN_STEPS \
        --seed $SEED \
        --run_suffix "pilot_comm_${COMM_RANGE}_seed${SEED}" \
        2>&1 | tee "$EXP_DIR/logs/comm_${COMM_RANGE}_seed${SEED}.log"
done

echo ""
echo "✓ 试点3 完成。结果 CSV 在 ray_results/pilot_comm_*/"
