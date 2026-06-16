#!/bin/bash
set -euo pipefail

WORKSPACE="/home/wj/work/multi-robot-exploration-rl"
RUN_SCRIPT="$WORKSPACE/src/gnn_marl_training_DGTA_nobuffer/run_curriculum.sh"

NUM_AGENTS="${NUM_AGENTS:-4}"
TRAIN_STEPS="${TRAIN_STEPS:-300000}"
START_STAGE="${START_STAGE:-2}"
END_STAGE="${END_STAGE:-2}"
NUM_WORKERS="${NUM_WORKERS:-1}"
ACTION_MODE="${ACTION_MODE:-continuous}"
MODEL_TYPE="${MODEL_TYPE:-gat}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-20}"
HEADLESS_SIM="${HEADLESS_SIM:-1}"
ENABLE_VISUALIZATION="${ENABLE_VISUALIZATION:-0}"

COMMON_ARGS=(
  --model_type "$MODEL_TYPE"
  --action_mode "$ACTION_MODE"
  --num_agents "$NUM_AGENTS"
  --num_workers "$NUM_WORKERS"
  --start_stage "$START_STAGE"
  --end_stage "$END_STAGE"
  --train_steps "$TRAIN_STEPS"
  --checkpoint_freq "$CHECKPOINT_FREQ"
)

if [[ "$HEADLESS_SIM" == "1" ]]; then
  COMMON_ARGS+=(--headless_sim)
fi
if [[ "$ENABLE_VISUALIZATION" == "1" ]]; then
  COMMON_ARGS+=(--enable_visualization)
else
  COMMON_ARGS+=(--disable_visualization)
fi

PROFILES=(
  "social_only"
  "obstacle_only"
  "dual_graph"
)

for graph_ablation in "${PROFILES[@]}"; do
  run_suffix="dgta_serial_${graph_ablation}_seed1"
  echo "[launch] graph_ablation=$graph_ablation run_suffix=$run_suffix checkpoint_freq=$CHECKPOINT_FREQ"
  "$RUN_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --graph_ablation "$graph_ablation" \
    --run_suffix "$run_suffix"
  echo "[done] graph_ablation=$graph_ablation"
  echo
  sleep 5
done
