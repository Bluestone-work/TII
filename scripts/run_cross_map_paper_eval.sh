#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEFAULT_SRC="$ROOT/src/gnn_marl_training_DGTA_nobuffer"
STRICT_TAG="ablation_strict_fourway_3seed_100k_20260714"
STRICT_EVAL_SRC="${STRICT_EVAL_SRC:-}"
CKPT="${CKPT:?CKPT must point to an RLlib checkpoint directory. Checkpoints are not included in this code release.}"
SRC="${EVAL_SRC_OVERRIDE:-$DEFAULT_SRC}"
if [[ -z "${EVAL_SRC_OVERRIDE:-}" && -n "$STRICT_EVAL_SRC" && "$CKPT" == *"/$STRICT_TAG/"* && -d "$STRICT_EVAL_SRC/gnn_marl_training" ]]; then
  SRC="$STRICT_EVAL_SRC"
fi
TEST_SCRIPT="$SRC/gnn_marl_training/test_gnn_mappo.py"
LOCAL_MAP_SCRIPT="$ROOT/scripts/plot_lidar_local_map.py"
AGG_SCRIPT="$ROOT/scripts/aggregate_cross_map_eval.py"
ATTN_PLOT_SCRIPT="$ROOT/scripts/plot_gat_attention.py"
RISK_PLOT_SCRIPT="$ROOT/scripts/plot_risk_field.py"
SCIENCE_PLOT_SCRIPT="$ROOT/scripts/plot_evaluation_science.py"
KILL_SCRIPT="$ROOT/kill_all_ros.sh"

NUM_EPISODES="${NUM_EPISODES:-20}"
MAPS="${MAPS:-3 4 5 9}"
NUM_AGENTS="${NUM_AGENTS:-4}"
MAP3_NUM_DYNAMIC_OBS="${MAP3_NUM_DYNAMIC_OBS:-6}"
MAP4_NUM_DYNAMIC_OBS="${MAP4_NUM_DYNAMIC_OBS:-2}"
MAP5_NUM_DYNAMIC_OBS="${MAP5_NUM_DYNAMIC_OBS:-0}"
MAP9_NUM_DYNAMIC_OBS="${MAP9_NUM_DYNAMIC_OBS:-6}"
OBS_SPEED_SCALE="${OBS_SPEED_SCALE:-0.25}"
OBS_SPEED_MPS="${OBS_SPEED_MPS:-}"
SAFETY_FILTER_ENABLE="${SAFETY_FILTER_ENABLE:-1}"
CAPTURE_ATTENTION="${CAPTURE_ATTENTION:-1}"
VIZ_STRIDE="${VIZ_STRIDE:-20}"
ATTN_AGENT="${ATTN_AGENT:-agent_0}"
ATTN_EPISODE="${ATTN_EPISODE:-0}"
EVAL_SEED="${EVAL_SEED:-0}"
TRAINING_SEED="${TRAINING_SEED:-}"
METHOD_LABEL="${METHOD_LABEL:-gnn_mappo_relative_motion}"
SCENARIO_SUFFIX="${SCENARIO_SUFFIX:-}"
EXPERIMENT_CONDITION="${EXPERIMENT_CONDITION:-}"
FIXED_SCENARIO_PACK="${FIXED_SCENARIO_PACK:-}"
FIXED_SCENARIO_START_INDEX="${FIXED_SCENARIO_START_INDEX:-0}"
RESUME_RESULTS="${RESUME_RESULTS:-0}"
COLLECT_LOCAL_MAP="${COLLECT_LOCAL_MAP:-0}"
GENERATE_SCIENCE_PLOTS="${GENERATE_SCIENCE_PLOTS:-0}"
COLLECT_CF_TRACE="${COLLECT_CF_TRACE:-0}"
COLLECT_TRAJECTORY="${COLLECT_TRAJECTORY:-0}"
TRAJECTORY_STRIDE="${TRAJECTORY_STRIDE:-5}"
ROS_DOMAIN_ID_EVAL="${ROS_DOMAIN_ID_EVAL:-113}"
GAZEBO_PORT_EVAL="${GAZEBO_PORT_EVAL:-11813}"
EVAL_CPU_THREADS="${EVAL_CPU_THREADS:-2}"
SENSOR_NOISE_STD="${SENSOR_NOISE_STD:-0.0}"
EXTRA_TEST_ARGS="${EXTRA_TEST_ARGS:-}"
CONTROLLER="${CONTROLLER:-policy}"
OUT_DIR="${OUT_DIR:-$ROOT/artifacts/eval/cross_map_eval_$(date +%Y%m%d_%H%M%S)}"
RESULTS_JSONL="$OUT_DIR/episodes.jsonl"
RESULTS_CSV="$OUT_DIR/episodes.csv"
CF_TRACE_JSONL="$OUT_DIR/cf_credit_trace.jsonl.gz"
TRAJECTORY_JSONL="$OUT_DIR/trajectories.jsonl"

CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
ROS2_CONDA_ENV="${ROS2_CONDA_ENV:-ros2}"
ROS_SETUP="/opt/ros/humble/setup.bash"
WS_SETUP="$ROOT/install/setup.bash"

export CONDA_SH ROS2_CONDA_ENV ROS_SETUP WS_SETUP SRC ROS_DOMAIN_ID_EVAL GAZEBO_PORT_EVAL
export EVAL_CPU_THREADS
# Tolerate occasional Gazebo multi-robot spawn-settle jitter by re-resetting the
# episode up to N times before the strict fixed-start-tolerance check fails.
# Default 1 == legacy crash-on-first-drift behavior (unchanged for existing runs).
FIXED_SPAWN_MAX_RETRIES="${FIXED_SPAWN_MAX_RETRIES:-1}"
export FIXED_SPAWN_MAX_RETRIES

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/local_map_obs" "$OUT_DIR/attention_risk"

setup_env_prefix='set +u; [[ -f "$CONDA_SH" ]] && source "$CONDA_SH"; command -v conda >/dev/null 2>&1 && conda activate "$ROS2_CONDA_ENV" || true; source "$ROS_SETUP"; [[ -f "$WS_SETUP" ]] && source "$WS_SETUP"; set -u; export PYTHONPATH="$SRC:${PYTHONPATH:-}"; export ROS_DOMAIN_ID="$ROS_DOMAIN_ID_EVAL"; export GAZEBO_MASTER_URI="http://127.0.0.1:$GAZEBO_PORT_EVAL"; export GAZEBO_MODEL_DATABASE_URI=""; export TURTLEBOT3_MODEL=burger; export OMP_NUM_THREADS="$EVAL_CPU_THREADS"; export MKL_NUM_THREADS="$EVAL_CPU_THREADS"; export OPENBLAS_NUM_THREADS="$EVAL_CPU_THREADS"; export NUMEXPR_NUM_THREADS="$EVAL_CPU_THREADS"; export VECLIB_MAXIMUM_THREADS="$EVAL_CPU_THREADS"; export BLIS_NUM_THREADS="$EVAL_CPU_THREADS"; export MKL_DYNAMIC=FALSE; export no_proxy=localhost,127.0.0.1; unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY'

map_label() {
  case "$1" in
    3) echo "map3_corridor_swap" ;;
    4) echo "map4_intersection" ;;
    5) echo "map5_warehouse_aisles" ;;
    9) echo "map9_warehouse_dynamic" ;;
    *) echo "map$1" ;;
  esac
}

map_max_steps() {
  case "$1" in
    3)     echo "${MAX_STEPS_BIG:-2000}" ;;
    4)     echo "${MAX_STEPS_MAP4:-1200}" ;;
    5)     echo "${MAX_STEPS_MAP5:-3000}" ;;
    9)     echo "${MAX_STEPS_MAP9:-1000}" ;;
    *)     echo "${MAX_STEPS_DEFAULT:-1000}" ;;
  esac
}

map_dynamic_obstacles() {
  case "$1" in
    3) echo "$MAP3_NUM_DYNAMIC_OBS" ;;
    4) echo "$MAP4_NUM_DYNAMIC_OBS" ;;
    5) echo "$MAP5_NUM_DYNAMIC_OBS" ;;
    9) echo "$MAP9_NUM_DYNAMIC_OBS" ;;
    *) echo 0 ;;
  esac
}

cleanup_eval_port() {
  KILL_ALL_ROS_SCOPE=port_only \
  GAZEBO_PORT="$GAZEBO_PORT_EVAL" \
  GAZEBO_MASTER_URI="http://127.0.0.1:$GAZEBO_PORT_EVAL" \
  bash "$KILL_SCRIPT" >/dev/null 2>&1 || true
}

stop_ros_launch() {
  if [[ -n "${ROS_PID:-}" ]]; then
    # The ros2 launch parent can exit before all of its process-group members.
    # Address the group directly so orphaned ROS nodes cannot survive and
    # answer services during the next fixed-scenario chunk.
    kill -TERM -- "-$ROS_PID" 2>/dev/null || true
    for _ in {1..10}; do
      kill -0 -- "-$ROS_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$ROS_PID" 2>/dev/null || true
    wait "$ROS_PID" 2>/dev/null || true
  fi
  ROS_PID=""
  cleanup_eval_port
}

wait_for_gazebo() {
  local log="$1"
  local ready_timeout="${GAZEBO_READY_TIMEOUT:-180}"
  local waited=0
  while (( waited < ready_timeout )); do
    local spawned=0
    if [[ -f "$log" ]]; then
      spawned="$(grep -cE 'Successfully spawned entity \[tb3_[0-9]+\]' "$log" 2>/dev/null || true)"
      spawned="${spawned:-0}"
    fi
    if (( spawned >= NUM_AGENTS )); then
      sleep 8
      return 0
    fi
    if ! kill -0 "$ROS_PID" 2>/dev/null; then
      echo "[FATAL] ros2 launch exited early; log=$log" >&2
      tail -n 120 "$log" >&2 || true
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
    if (( waited % 20 == 0 )); then
      echo "[wait] Gazebo ${waited}s/${ready_timeout}s spawned=${spawned}/${NUM_AGENTS} log=$log"
    fi
  done
  echo "[FATAL] Gazebo did not become ready in ${ready_timeout}s; log=$log" >&2
  tail -n 160 "$log" >&2 || true
  return 1
}

run_map() {
  local map="$1"
  local label
  label="$(map_label "$map")"
  local ros_log="$OUT_DIR/logs/${label}_ros.log"
  local eval_log="$OUT_DIR/logs/${label}_eval.log"
  local obs_log="$OUT_DIR/logs/${label}_local_map.log"
  local dyn_obs
  dyn_obs="$(map_dynamic_obstacles "$map")"
  local gazebo_speed_scale="$OBS_SPEED_SCALE"
  local test_speed_arg="--obs_speed_scale '$OBS_SPEED_SCALE'"
  local fixed_scenario_file=""
  local fixed_dynamic_file=""
  local fixed_launch_arg=""
  local obstacle_seed="$EVAL_SEED"
  if [[ -n "$OBS_SPEED_MPS" ]]; then
    gazebo_speed_scale="$(awk -v speed="$OBS_SPEED_MPS" 'BEGIN { printf "%.9f", speed / 0.22 }')"
    test_speed_arg="--obs_speed_mps '$OBS_SPEED_MPS'"
  fi
  if [[ -n "$FIXED_SCENARIO_PACK" ]]; then
    fixed_scenario_file="$FIXED_SCENARIO_PACK/fixed_scenarios/map${map}_scenarios.json"
    fixed_dynamic_file="$FIXED_SCENARIO_PACK/dynamic_initial_states/map${map}_dynamic_initial_states.json"
    [[ -f "$fixed_scenario_file" && -f "$fixed_dynamic_file" ]] || {
      echo "[FATAL] fixed scenario inputs are missing for map $map" >&2
      return 2
    }
    obstacle_seed="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['scenarios'][int(sys.argv[2])]['seed'])" "$fixed_scenario_file" "$FIXED_SCENARIO_START_INDEX")"
    [[ "$obstacle_seed" =~ ^[0-9]+$ ]] || {
      echo "[FATAL] invalid fixed scenario seed for map $map index=$FIXED_SCENARIO_START_INDEX" >&2
      return 2
    }
    dyn_obs="$(python3 -c "import json,sys; print(int(json.load(open(sys.argv[1]))['map_config']['num_dynamic_obstacles']))" "$fixed_scenario_file")"
    fixed_launch_arg="fixed_initial_states_file:=$fixed_dynamic_file"
  fi

  echo
  echo "===== EVAL $label ====="
  cleanup_eval_port

  setsid bash -lc "$setup_env_prefix; exec ros2 launch start_rl_environment_tb3 main_headless.launch.py map_number:=$map robot_number:=$NUM_AGENTS num_obstacles:=$dyn_obs obs_speed_scale:=$gazebo_speed_scale obstacle_seed:=$obstacle_seed $fixed_launch_arg enable_rviz:=false" >"$ros_log" 2>&1 &
  ROS_PID=$!
  echo "[ros] pid=$ROS_PID log=$ros_log"
  wait_for_gazebo "$ros_log"

  if [[ "$COLLECT_LOCAL_MAP" == "1" ]]; then
    echo "[obs] collecting local-map heatmaps for $label"
    bash -lc "$setup_env_prefix; python3 '$LOCAL_MAP_SCRIPT' --robot tb3_0 --samples 120 --stride 2 --timeout-s 90 --out-dir '$OUT_DIR/local_map_obs' --prefix '${label}_tb3_0'" >"$obs_log" 2>&1 || {
      echo "[WARN] local-map capture failed for $label; log=$obs_log" >&2
    }
  fi

  echo "[eval] running policy episodes for $label"
  local max_steps
  max_steps="$(map_max_steps "$map")"
  local attn_npz="$OUT_DIR/attention_risk/${label}_attn_risk.npz"
  local training_seed_arg=""
  local experiment_condition_arg=""
  local cf_trace_arg=""
  local trajectory_arg=""
  local scenario_arg=""
  if [[ -n "$TRAINING_SEED" ]]; then
    training_seed_arg="--training_seed '$TRAINING_SEED'"
  fi
  if [[ -n "$EXPERIMENT_CONDITION" ]]; then
    experiment_condition_arg="--experiment_condition '$EXPERIMENT_CONDITION'"
  fi
  if [[ "$COLLECT_CF_TRACE" == "1" ]]; then
    cf_trace_arg="--cf_trace_jsonl '$CF_TRACE_JSONL'"
  fi
  if [[ "$COLLECT_TRAJECTORY" == "1" ]]; then
    trajectory_arg="--trajectory_jsonl '$TRAJECTORY_JSONL' --trajectory_stride '$TRAJECTORY_STRIDE'"
  fi
  if [[ -n "$FIXED_SCENARIO_PACK" ]]; then
    scenario_arg="--fixed_scenario_file '$fixed_scenario_file' --fixed_dynamic_state_file '$fixed_dynamic_file' --fixed_scenario_start_index '$FIXED_SCENARIO_START_INDEX' --paired_dynamic_obstacles 1"
  fi
  # 固定场景模式下不传 --num_dynamic_obstacles（协议已规定）
  local dyn_obs_arg=""
  local speed_arg=""
  local noise_arg=""
  local shield_arg=""
  if [[ -z "$FIXED_SCENARIO_PACK" ]]; then
    dyn_obs_arg="--num_dynamic_obstacles '$dyn_obs'"
    speed_arg="$test_speed_arg"
    noise_arg="--sensor_noise_std '$SENSOR_NOISE_STD'"
    shield_arg="--shield_enable '$SAFETY_FILTER_ENABLE'"
  fi
  set +e
  bash -lc "$setup_env_prefix; python3 '$TEST_SCRIPT' --checkpoint_path '$CKPT' --num_episodes '$NUM_EPISODES' --seed '$EVAL_SEED' $training_seed_arg $experiment_condition_arg $cf_trace_arg $trajectory_arg $scenario_arg $EXTRA_TEST_ARGS --controller '$CONTROLLER' --num_agents '$NUM_AGENTS' --map_number '$map' $dyn_obs_arg $speed_arg $noise_arg $shield_arg --auto_reset_agents 0 --min_active_agents_to_continue 0 --max_episode_steps '$max_steps' --max_failed_agents_before_cutoff 0 --diag_steps 0 --capture_attention '$CAPTURE_ATTENTION' --viz_stride '$VIZ_STRIDE' --attention_out '$attn_npz' --results_jsonl '$RESULTS_JSONL' --results_csv '$RESULTS_CSV' --method_label '$METHOD_LABEL' --scenario_label '${label}${SCENARIO_SUFFIX}'" 2>&1 | tee "$eval_log"
  local rc=${PIPESTATUS[0]}
  set -e

  stop_ros_launch
  if (( rc != 0 )); then
    echo "[FATAL] eval failed for $label rc=$rc log=$eval_log" >&2
    return "$rc"
  fi

  if [[ "$CAPTURE_ATTENTION" == "1" && -f "$attn_npz" ]]; then
    bash -lc "$setup_env_prefix; python3 '$ATTN_PLOT_SCRIPT' '$attn_npz' --out-dir '$OUT_DIR/attention_risk' --prefix '$label' --agent '$ATTN_AGENT' --episode '$ATTN_EPISODE'" \
      >"$OUT_DIR/logs/${label}_attn_plot.log" 2>&1 || echo "[WARN] attention plot failed for $label" >&2
    bash -lc "$setup_env_prefix; python3 '$RISK_PLOT_SCRIPT' '$attn_npz' --out-dir '$OUT_DIR/attention_risk' --prefix '$label'" \
      >"$OUT_DIR/logs/${label}_risk_plot.log" 2>&1 || echo "[WARN] risk-field plot failed for $label" >&2
  fi
}

main() {
  if [[ ! -f "$CKPT/algorithm_state.pkl" ]]; then
    echo "[FATAL] invalid checkpoint: $CKPT" >&2
    exit 2
  fi
  echo "[INFO] output=$OUT_DIR"
  echo "[INFO] checkpoint=$CKPT"
  echo "[INFO] maps=$MAPS episodes_per_map=$NUM_EPISODES dyn_obs={map3:$MAP3_NUM_DYNAMIC_OBS,map4:$MAP4_NUM_DYNAMIC_OBS,map5:$MAP5_NUM_DYNAMIC_OBS,map9:$MAP9_NUM_DYNAMIC_OBS} speed_scale=$OBS_SPEED_SCALE speed_mps=${OBS_SPEED_MPS:-derived} safety_filter=$SAFETY_FILTER_ENABLE viz_stride=$VIZ_STRIDE cf_trace=$COLLECT_CF_TRACE trajectory=$COLLECT_TRAJECTORY trajectory_stride=$TRAJECTORY_STRIDE method=$METHOD_LABEL scenario_suffix=$SCENARIO_SUFFIX fixed_pack=${FIXED_SCENARIO_PACK:-none} fixed_start=$FIXED_SCENARIO_START_INDEX"
  echo "[INFO] ROS_DOMAIN_ID=$ROS_DOMAIN_ID_EVAL GAZEBO_PORT=$GAZEBO_PORT_EVAL eval_cpu_threads=$EVAL_CPU_THREADS"

  if [[ "$RESUME_RESULTS" == "1" ]]; then
    touch "$RESULTS_JSONL" "$RESULTS_CSV"
  else
    : > "$RESULTS_JSONL"
    : > "$RESULTS_CSV"
  fi
  if [[ "$COLLECT_CF_TRACE" == "1" ]]; then
    rm -f "$CF_TRACE_JSONL"
  fi
  if [[ "$COLLECT_TRAJECTORY" == "1" ]]; then
    rm -f "$TRAJECTORY_JSONL"
  fi
  for map in $MAPS; do
    run_map "$map"
  done

  bash -lc "$setup_env_prefix; python3 '$AGG_SCRIPT' '$RESULTS_JSONL' --outdir '$OUT_DIR'" 2>&1 | tee "$OUT_DIR/aggregate.log"
  if [[ "$GENERATE_SCIENCE_PLOTS" == "1" ]]; then
    bash -lc "$setup_env_prefix; python3 '$SCIENCE_PLOT_SCRIPT' '$RESULTS_JSONL' --risk-dir '$OUT_DIR/attention_risk' --out-dir '$OUT_DIR/scientific_figures'" 2>&1 | tee "$OUT_DIR/scientific_figures.log"
  fi
  echo "[INFO] complete output=$OUT_DIR"
}

trap stop_ros_launch EXIT
main "$@"
