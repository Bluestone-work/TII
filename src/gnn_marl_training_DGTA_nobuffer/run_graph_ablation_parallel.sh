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
  --parallel_safe
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
  "social:social_only:41:11445"
  "obstacle:obstacle_only:42:11446"
  "dual:dual_graph:43:11447"
)

LOG_DIR="$WORKSPACE/curriculum_logs/dgta_graph_parallel"
mkdir -p "$LOG_DIR"
PIDS=()

launch_profile() {
  local label="$1"
  local graph_ablation="$2"
  local ros_domain_id="$3"
  local gazebo_port="$4"
  local run_suffix="dgta_${label}_${graph_ablation}_seed1"
  local log_file="$LOG_DIR/${run_suffix}.launcher.log"

  echo "[launch] label=$label graph_ablation=$graph_ablation ros_domain_id=$ros_domain_id gazebo_port=$gazebo_port checkpoint_freq=$CHECKPOINT_FREQ log=$log_file"

  local cmd=(
    "$RUN_SCRIPT"
    "${COMMON_ARGS[@]}"
    --graph_ablation "$graph_ablation"
    --run_suffix "$run_suffix"
    --ros_domain_id "$ros_domain_id"
    --gazebo_port "$gazebo_port"
  )

  "${cmd[@]}" >"$log_file" 2>&1 &
  PIDS+=("$!")
}

for spec in "${PROFILES[@]}"; do
  IFS=':' read -r label graph_ablation ros_domain_id gazebo_port <<<"$spec"
  launch_profile "$label" "$graph_ablation" "$ros_domain_id" "$gazebo_port"
done

echo "[info] launched ${#PIDS[@]} parallel DGTA graph ablation runs"
echo "[info] logs: $LOG_DIR"
echo "[info] periodic checkpoints enabled via --checkpoint_freq $CHECKPOINT_FREQ"
echo "[info] pids: ${PIDS[*]}"
wait
