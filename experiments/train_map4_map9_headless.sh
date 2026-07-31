#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC="$ROOT/src/gnn_marl_training_DGTA_nobuffer"
TRAIN_SCRIPT="$SRC/gnn_marl_training/train_gnn_mappo_full.py"
KILL_SCRIPT="$ROOT/kill_all_ros.sh"

CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
ROS2_CONDA_ENV="${ROS2_CONDA_ENV:-ros2}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
WS_SETUP="${WS_SETUP:-$ROOT/install/setup.bash}"

NUM_AGENTS="${NUM_AGENTS:-4}"
SEED="${SEED:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-300000}"
EARLY_STOP_MIN_STEPS="${EARLY_STOP_MIN_STEPS:-200000}"
EARLY_STOP_PATIENCE_ITERS="${EARLY_STOP_PATIENCE_ITERS:-25}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2000}"
NUM_WORKERS="${NUM_WORKERS:-1}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-0}"
RAY_NUM_GPUS="${RAY_NUM_GPUS:--1.0}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/artifacts/ray_results}"
GAZEBO_PORT="${GAZEBO_PORT:-11345}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-113}"
GAZEBO_WAIT_SEC="${GAZEBO_WAIT_SEC:-45}"

export PYTHONPATH="$SRC:${PYTHONPATH:-}"
export GAZEBO_MASTER_URI="http://127.0.0.1:$GAZEBO_PORT"
export ROS_DOMAIN_ID
export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-burger}"
export GAZEBO_MODEL_DATABASE_URI=""

setup_env() {
  set +u
  [[ -f "$CONDA_SH" ]] && source "$CONDA_SH"
  command -v conda >/dev/null 2>&1 && conda activate "$ROS2_CONDA_ENV" || true
  source "$ROS_SETUP"
  [[ -f "$WS_SETUP" ]] && source "$WS_SETUP"
  set -u
}

cleanup() {
  KILL_ALL_ROS_SCOPE=port_only GAZEBO_PORT="$GAZEBO_PORT" bash "$KILL_SCRIPT" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

latest_checkpoint() {
  local root="$1"
  find "$root" -type d -name 'checkpoint_*' 2>/dev/null | sort -V | tail -1
}

launch_world() {
  local map_number="$1"
  local num_obstacles="$2"
  local obs_speed_scale="$3"
  cleanup
  ros2 launch start_rl_environment_tb3 main_headless.launch.py \
    map_number:="$map_number" \
    robot_number:="$NUM_AGENTS" \
    num_obstacles:="$num_obstacles" \
    obs_speed_scale:="$obs_speed_scale" &
  ROS_LAUNCH_PID=$!
  sleep "$GAZEBO_WAIT_SEC"
}

run_train() {
  local map_number="$1"
  local run_suffix="$2"
  local resume_ckpt="${3:-}"
  local num_obstacles="$4"
  local obs_speed_scale="$5"

  launch_world "$map_number" "$num_obstacles" "$obs_speed_scale"

  cmd=(
    python3 "$TRAIN_SCRIPT"
    --model_type gat
    --graph_ablation dual_graph
    --num_agents "$NUM_AGENTS"
    --map_number "$map_number"
    --num_obstacles "$num_obstacles"
    --obs_speed_scale "$obs_speed_scale"
    --train_steps "$TRAIN_STEPS"
    --early_stop_min_steps "$EARLY_STOP_MIN_STEPS"
    --early_stop_patience_iters "$EARLY_STOP_PATIENCE_ITERS"
    --train_batch_size "$TRAIN_BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --ray_num_cpus "$RAY_NUM_CPUS"
    --ray_num_gpus "$RAY_NUM_GPUS"
    --safety_filter_enable 0
    --allow_shield_during_training 0
    --action_mode continuous
    --ppo_profile continuous
    --output_dir "$OUTPUT_DIR"
    --run_name_suffix "$run_suffix"
    --seed "$SEED"
  )

  if [[ -n "$resume_ckpt" ]]; then
    cmd+=(--resume_checkpoint "$resume_ckpt")
  fi

  "${cmd[@]}"
}

setup_env
mkdir -p "$OUTPUT_DIR"

run_train 4 "map4_seed${SEED}" "" 2 0.25
MAP4_CKPT="$(latest_checkpoint "$OUTPUT_DIR")"
if [[ -z "$MAP4_CKPT" ]]; then
  echo "[FATAL] No Map 4 checkpoint found under $OUTPUT_DIR" >&2
  exit 2
fi

run_train 9 "map9_after_map4_seed${SEED}" "$MAP4_CKPT" 6 0.25

echo "[done] outputs: $OUTPUT_DIR"
