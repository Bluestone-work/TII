#!/bin/bash
# 试点1: 对比实验 — MLP baseline vs Full GAT
# Stage2(4车 circle_swap), 单种子, 快速验证(5万步)
set -euo pipefail

WORKSPACE="/home/wj/work/multi-robot-exploration-rl"
PKG_DIR="$WORKSPACE/src/gnn_marl_training_DGTA_nobuffer"
EXP_DIR="$WORKSPACE/experiments/pilot_comparison"
SEED=42
TRAIN_STEPS=50000   # 试点用小步数快速验证流程

cd "$PKG_DIR"

echo "========================================="
echo "试点1: 对比实验 (MLP vs Full GAT)"
echo "  Stage2 (4车), seed=$SEED, steps=$TRAIN_STEPS"
echo "========================================="

# --- 实验 A: MLP baseline ---
echo ""
echo ">>> [1/2] MLP baseline 训练中..."
TRAIN_VERBOSE=0 ENV_VERBOSE=0 RAY_memory_usage_threshold=0.97 \
./run_curriculum.sh \
    --model_type mlp \
    --start_stage 2 --end_stage 2 \
    --num_agents 4 \
    --num_workers 1 \
    --train_steps $TRAIN_STEPS \
    --seed $SEED \
    --run_suffix "pilot_cmp_mlp_seed${SEED}" \
    2>&1 | tee "$EXP_DIR/logs/mlp_seed${SEED}.log"

# --- 实验 B: Full GAT (dual_graph) ---
echo ""
echo ">>> [2/2] Full GAT 训练中..."
TRAIN_VERBOSE=0 ENV_VERBOSE=0 RAY_memory_usage_threshold=0.97 \
./run_curriculum.sh \
    --model_type gat \
    --graph_ablation dual_graph \
    --gat_actor_graph local_risk \
    --start_stage 2 --end_stage 2 \
    --num_agents 4 \
    --num_workers 1 \
    --train_steps $TRAIN_STEPS \
    --seed $SEED \
    --run_suffix "pilot_cmp_gat_seed${SEED}" \
    2>&1 | tee "$EXP_DIR/logs/gat_seed${SEED}.log"

echo ""
echo "✓ 试点1 完成。结果 CSV 在 ray_results/pilot_cmp_*/"
