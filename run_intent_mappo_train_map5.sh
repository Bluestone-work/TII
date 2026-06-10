#!/bin/bash
# Intent-aware MAPPO launcher for map 5.

if [ -z "${BASH_VERSION:-}" ]; then
    exec /bin/bash "$0" "$@"
fi

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
info()    { echo -e "${CYAN}[INFO]${RESET} $*"; }
success() { echo -e "${GREEN}[OK]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET} $*"; }
error()   { echo -e "${RED}[ERR]${RESET} $*" >&2; }
banner()  { echo -e "\n${BOLD}${CYAN}$*${RESET}\n"; }

WORKSPACE="/home/wj/work/multi-robot-exploration-rl"
CONDA_SETUP="/home/wj/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="ros2"
ROS_SETUP="/opt/ros/humble/setup.bash"
WS_SETUP="${WORKSPACE}/install/setup.bash"
KILL_SCRIPT="${WORKSPACE}/kill_all_ros.sh"
TRAIN_ENTRY_SRC="${WORKSPACE}/src/intent_marl_training/intent_marl_training/train_intent_mappo.py"

ENV_STAGE=1
MAP_NUMBER=5
NUM_AGENTS=4
NUM_WORKERS=1
TRAIN_STEPS=300000
CHECKPOINT_FREQ=20
TRAIN_BATCH_SIZE=4000
ROLLOUT_FRAGMENT_LENGTH=200
RUN_NAME="intent_mappo_map5"
OUTPUT_DIR="${WORKSPACE}/ray_results"
SAMPLE_TIMEOUT_S=1800

COMMUNICATION_RANGE=3.5
INTENT_TOP_K=3
INTENT_HISTORY_STEPS=4
INTENT_DT=0.1

ENABLE_YIELDING=1
ENABLE_YIELD_ACTION=1
ENABLE_YIELD_REWARD=1
YIELD_OBS_DIM=5
YIELD_DIST_THRESHOLD=1.10
YIELD_TTC_THRESHOLD=2.20
YIELD_RELEASE_DIST=1.45
YIELD_RELEASE_TTC=3.00
YIELD_COMMIT_STEPS=6
YIELD_TIE_MARGIN=0.30
YIELD_LINEAR_SCALE=0.25
YIELD_LINEAR_STOP=0.02
YIELD_LINEAR_STOP_TTC=0.85
YIELD_TURN_BIAS=0.20
YIELD_OBEY_SPEED_THRESH=0.08
PRIORITY_MOVE_SPEED_THRESH=0.10
YIELD_COMPLIANCE_REWARD=0.22
YIELD_VIOLATION_PENALTY=0.35
PRIORITY_PROGRESS_REWARD=0.10
PRIORITY_IDLE_PENALTY=0.08

ENABLE_SHIELD=1
SHIELD_TRIGGER_DIST=0.55
SHIELD_HARD_DIST=0.34
MAX_REVERSE_SPEED=0.08
MAX_LINEAR_SPEED=0.22
MAX_ANGULAR_SPEED=1.2

ENABLE_VISUALIZATION=1
ENABLE_INTENT_VISUALIZATION=1
TRACKING_VIZ_INTERVAL=4
INTENT_VIZ_INTERVAL=4
INTENT_VIZ_HORIZON_SEC=1.2
INTENT_VIZ_TOPIC="/intent_marl/intent_markers"
SHIELD_VIZ_TOPIC="/intent_marl/shield_markers"
ENV_LOG_LEVEL="WARNING"
SIM_WAIT_WALL_TIMEOUT=2.5

LAUNCH_ENV=1
GAZEBO_WAIT_SEC=60
GAZEBO_GRACE_SEC=5

LOG_DIR="${WORKSPACE}/train_logs"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_LOG="${LOG_DIR}/intent_env_map${MAP_NUMBER}_${STAMP}.log"
TRAIN_LOG="${LOG_DIR}/intent_train_${STAMP}.log"

EXTRA_ARGS=()
ENV_STARTED=0

usage() {
    cat <<'EOF'
Intent MAPPO training launcher for map 5.

Usage:
  ./run_intent_mappo_train_map5.sh
  ./run_intent_mappo_train_map5.sh --train_steps 500000 --num_agents 4
  ./run_intent_mappo_train_map5.sh --skip_launch_env
  ./run_intent_mappo_train_map5.sh -- --lr 1e-4 --sample_timeout_s 2400

Options:
  --env_stage N
  --num_agents N
  --num_workers N
  --train_steps N
  --checkpoint_freq N
  --train_batch_size N
  --rollout_fragment_length N
  --run_name NAME
  --output_dir DIR
  --sample_timeout_s N
  --communication_range X
  --intent_top_k N
  --intent_history_steps N
  --intent_dt X
  --enable_yielding / --disable_yielding
  --enable_yield_action / --disable_yield_action
  --enable_yield_reward / --disable_yield_reward
  --yield_obs_dim N
  --yield_dist_threshold X
  --yield_ttc_threshold X
  --yield_release_dist X
  --yield_release_ttc X
  --yield_commit_steps N
  --yield_tie_margin X
  --yield_linear_scale X
  --yield_linear_stop X
  --yield_linear_stop_ttc X
  --yield_turn_bias X
  --yield_obey_speed_thresh X
  --priority_move_speed_thresh X
  --yield_compliance_reward X
  --yield_violation_penalty X
  --priority_progress_reward X
  --priority_idle_penalty X
  --enable_safety_shield / --disable_safety_shield
  --shield_trigger_dist X
  --shield_hard_dist X
  --max_reverse_speed X
  --max_linear_speed X
  --max_angular_speed X
  --enable_visualization / --disable_visualization
  --enable_intent_visualization / --disable_intent_visualization
  --tracking_viz_interval N
  --intent_viz_interval N
  --intent_viz_horizon_sec X
  --intent_viz_topic TOPIC
  --shield_viz_topic TOPIC
  --env_log_level LEVEL
  --sim_wait_wall_timeout X
  --skip_launch_env
  --gazebo_wait_sec N
  --gazebo_grace_sec N
  --help

Notes:
  1) Map is fixed to 5 in this launcher.
  2) Use "-- <extra args>" to pass additional train_intent_mappo args.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env_stage) ENV_STAGE="$2"; shift 2 ;;
        --num_agents) NUM_AGENTS="$2"; shift 2 ;;
        --num_workers) NUM_WORKERS="$2"; shift 2 ;;
        --train_steps) TRAIN_STEPS="$2"; shift 2 ;;
        --checkpoint_freq) CHECKPOINT_FREQ="$2"; shift 2 ;;
        --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --rollout_fragment_length) ROLLOUT_FRAGMENT_LENGTH="$2"; shift 2 ;;
        --run_name) RUN_NAME="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --sample_timeout_s) SAMPLE_TIMEOUT_S="$2"; shift 2 ;;
        --communication_range) COMMUNICATION_RANGE="$2"; shift 2 ;;
        --intent_top_k) INTENT_TOP_K="$2"; shift 2 ;;
        --intent_history_steps) INTENT_HISTORY_STEPS="$2"; shift 2 ;;
        --intent_dt) INTENT_DT="$2"; shift 2 ;;
        --enable_yielding) ENABLE_YIELDING=1; shift 1 ;;
        --disable_yielding) ENABLE_YIELDING=0; shift 1 ;;
        --enable_yield_action) ENABLE_YIELD_ACTION=1; shift 1 ;;
        --disable_yield_action) ENABLE_YIELD_ACTION=0; shift 1 ;;
        --enable_yield_reward) ENABLE_YIELD_REWARD=1; shift 1 ;;
        --disable_yield_reward) ENABLE_YIELD_REWARD=0; shift 1 ;;
        --yield_obs_dim) YIELD_OBS_DIM="$2"; shift 2 ;;
        --yield_dist_threshold) YIELD_DIST_THRESHOLD="$2"; shift 2 ;;
        --yield_ttc_threshold) YIELD_TTC_THRESHOLD="$2"; shift 2 ;;
        --yield_release_dist) YIELD_RELEASE_DIST="$2"; shift 2 ;;
        --yield_release_ttc) YIELD_RELEASE_TTC="$2"; shift 2 ;;
        --yield_commit_steps) YIELD_COMMIT_STEPS="$2"; shift 2 ;;
        --yield_tie_margin) YIELD_TIE_MARGIN="$2"; shift 2 ;;
        --yield_linear_scale) YIELD_LINEAR_SCALE="$2"; shift 2 ;;
        --yield_linear_stop) YIELD_LINEAR_STOP="$2"; shift 2 ;;
        --yield_linear_stop_ttc) YIELD_LINEAR_STOP_TTC="$2"; shift 2 ;;
        --yield_turn_bias) YIELD_TURN_BIAS="$2"; shift 2 ;;
        --yield_obey_speed_thresh) YIELD_OBEY_SPEED_THRESH="$2"; shift 2 ;;
        --priority_move_speed_thresh) PRIORITY_MOVE_SPEED_THRESH="$2"; shift 2 ;;
        --yield_compliance_reward) YIELD_COMPLIANCE_REWARD="$2"; shift 2 ;;
        --yield_violation_penalty) YIELD_VIOLATION_PENALTY="$2"; shift 2 ;;
        --priority_progress_reward) PRIORITY_PROGRESS_REWARD="$2"; shift 2 ;;
        --priority_idle_penalty) PRIORITY_IDLE_PENALTY="$2"; shift 2 ;;
        --enable_safety_shield) ENABLE_SHIELD=1; shift 1 ;;
        --disable_safety_shield) ENABLE_SHIELD=0; shift 1 ;;
        --shield_trigger_dist) SHIELD_TRIGGER_DIST="$2"; shift 2 ;;
        --shield_hard_dist) SHIELD_HARD_DIST="$2"; shift 2 ;;
        --max_reverse_speed) MAX_REVERSE_SPEED="$2"; shift 2 ;;
        --max_linear_speed) MAX_LINEAR_SPEED="$2"; shift 2 ;;
        --max_angular_speed) MAX_ANGULAR_SPEED="$2"; shift 2 ;;
        --enable_visualization) ENABLE_VISUALIZATION=1; shift 1 ;;
        --disable_visualization) ENABLE_VISUALIZATION=0; shift 1 ;;
        --enable_intent_visualization) ENABLE_INTENT_VISUALIZATION=1; shift 1 ;;
        --disable_intent_visualization) ENABLE_INTENT_VISUALIZATION=0; shift 1 ;;
        --tracking_viz_interval) TRACKING_VIZ_INTERVAL="$2"; shift 2 ;;
        --intent_viz_interval) INTENT_VIZ_INTERVAL="$2"; shift 2 ;;
        --intent_viz_horizon_sec) INTENT_VIZ_HORIZON_SEC="$2"; shift 2 ;;
        --intent_viz_topic) INTENT_VIZ_TOPIC="$2"; shift 2 ;;
        --shield_viz_topic) SHIELD_VIZ_TOPIC="$2"; shift 2 ;;
        --env_log_level) ENV_LOG_LEVEL="$2"; shift 2 ;;
        --sim_wait_wall_timeout) SIM_WAIT_WALL_TIMEOUT="$2"; shift 2 ;;
        --skip_launch_env) LAUNCH_ENV=0; shift 1 ;;
        --gazebo_wait_sec) GAZEBO_WAIT_SEC="$2"; shift 2 ;;
        --gazebo_grace_sec) GAZEBO_GRACE_SEC="$2"; shift 2 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) EXTRA_ARGS+=("$1"); shift 1 ;;
    esac
done

source_env() {
    set +u
    if [[ -f "${CONDA_SETUP}" ]]; then
        source "${CONDA_SETUP}"
        conda activate "${CONDA_ENV}" >/dev/null 2>&1 || { set -u; error "Failed to activate conda env ${CONDA_ENV}."; exit 1; }
    fi
    [[ -f "${ROS_SETUP}" ]] && source "${ROS_SETUP}" || { set -u; error "Missing ROS setup: ${ROS_SETUP}"; exit 1; }
    [[ -f "${WS_SETUP}" ]] && source "${WS_SETUP}" || warn "Workspace setup not found: ${WS_SETUP}"
    set -u
}

check_env() {
    banner "=== Intent MAPPO training check (map 5) ==="
    command -v ros2 >/dev/null 2>&1 || { error "ros2 not found."; exit 1; }
    [[ -f "${TRAIN_ENTRY_SRC}" ]] || { error "Training entry script not found: ${TRAIN_ENTRY_SRC}"; exit 1; }

    if (( LAUNCH_ENV == 1 )) && ! ros2 pkg prefix start_rl_environment_tb3 >/dev/null 2>&1; then
        error "Package start_rl_environment_tb3 unavailable. Cannot launch env."
        exit 1
    fi

    if ! python3 - <<'PY'
import importlib

missing = []
for mod_name in ("ray",):
    try:
        importlib.import_module(mod_name)
    except Exception as exc:  # noqa: BLE001
        missing.append(f"{mod_name} ({exc})")

ok_env_lib = False
for mod_name in ("gymnasium", "gym"):
    try:
        importlib.import_module(mod_name)
        ok_env_lib = True
        break
    except Exception:
        pass
if not ok_env_lib:
    missing.append("gymnasium/gym")

if missing:
    raise SystemExit("Missing python deps: " + ", ".join(missing))
PY
    then
        error "Python deps missing in current interpreter. Activate env and install deps first."
        error "Example: conda activate ${CONDA_ENV} && pip install gymnasium ray[rllib]"
        exit 1
    fi

    info "python3: $(command -v python3)"
    success "Environment check passed."
}

stop_ros_env() {
    [[ -f "${KILL_SCRIPT}" ]] || { warn "kill script not found: ${KILL_SCRIPT}"; return 0; }
    info "Stopping old ROS/Gazebo..."
    bash "${KILL_SCRIPT}" 2>/dev/null || true
    sleep 2
}

start_target_env() {
    banner "Launch env map5 for training"
    info "log: ${ENV_LOG}"

    local inner_cmd
    inner_cmd="set +u"
    inner_cmd+="; source '${ROS_SETUP}'"
    inner_cmd+="; [[ -f '${WS_SETUP}' ]] && source '${WS_SETUP}'"
    inner_cmd+="; set -u"
    inner_cmd+="; ros2 launch start_rl_environment_tb3 main.launch.py"
    inner_cmd+=" map_number:=5 robot_number:=${NUM_AGENTS}"
    inner_cmd+=" num_obstacles:=0 obs_speed_scale:=0.0"
    inner_cmd+=" 2>&1 | tee '${ENV_LOG}'"

    if command -v gnome-terminal >/dev/null 2>&1; then
        gnome-terminal --title="[Intent MAPPO] env map5" -- bash -c "${inner_cmd}" &
    elif command -v xterm >/dev/null 2>&1; then
        xterm -title "[Intent MAPPO] env map5" -e bash -c "${inner_cmd}" &
    else
        warn "No GUI terminal found, running env in background."
        bash -c "${inner_cmd}" &
    fi

    info "Waiting Gazebo ready (max ${GAZEBO_WAIT_SEC}s)..."
    local waited=0
    while [[ ${waited} -lt ${GAZEBO_WAIT_SEC} ]]; do
        if ros2 topic list 2>/dev/null | grep -Eq "/tb3_0/(scan|odom)"; then
            echo ""
            success "Gazebo ready (${waited}s)."
            sleep "${GAZEBO_GRACE_SEC}"
            ENV_STARTED=1
            return 0
        fi
        if [[ -f "${ENV_LOG}" ]] && grep -q "Successfully spawned entity \[tb3_0\]" "${ENV_LOG}"; then
            echo ""
            success "Robot spawn detected (${waited}s)."
            sleep "${GAZEBO_GRACE_SEC}"
            ENV_STARTED=1
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
        printf "\r  waiting... %ds / %ds" "${waited}" "${GAZEBO_WAIT_SEC}"
    done
    echo ""
    warn "Gazebo wait timeout. Continue anyway."
    ENV_STARTED=1
}

run_training() {
    local cmd=(
        python3 "${TRAIN_ENTRY_SRC}"
        --env_stage "${ENV_STAGE}"
        --map_number "5"
        --num_agents "${NUM_AGENTS}"
        --num_workers "${NUM_WORKERS}"
        --train_steps "${TRAIN_STEPS}"
        --checkpoint_freq "${CHECKPOINT_FREQ}"
        --train_batch_size "${TRAIN_BATCH_SIZE}"
        --rollout_fragment_length "${ROLLOUT_FRAGMENT_LENGTH}"
        --sample_timeout_s "${SAMPLE_TIMEOUT_S}"
        --communication_range "${COMMUNICATION_RANGE}"
        --intent_top_k "${INTENT_TOP_K}"
        --intent_history_steps "${INTENT_HISTORY_STEPS}"
        --intent_dt "${INTENT_DT}"
        --yield_obs_dim "${YIELD_OBS_DIM}"
        --yield_dist_threshold "${YIELD_DIST_THRESHOLD}"
        --yield_ttc_threshold "${YIELD_TTC_THRESHOLD}"
        --yield_release_dist "${YIELD_RELEASE_DIST}"
        --yield_release_ttc "${YIELD_RELEASE_TTC}"
        --yield_commit_steps "${YIELD_COMMIT_STEPS}"
        --yield_tie_margin "${YIELD_TIE_MARGIN}"
        --yield_linear_scale "${YIELD_LINEAR_SCALE}"
        --yield_linear_stop "${YIELD_LINEAR_STOP}"
        --yield_linear_stop_ttc "${YIELD_LINEAR_STOP_TTC}"
        --yield_turn_bias "${YIELD_TURN_BIAS}"
        --yield_obey_speed_thresh "${YIELD_OBEY_SPEED_THRESH}"
        --priority_move_speed_thresh "${PRIORITY_MOVE_SPEED_THRESH}"
        --yield_compliance_reward "${YIELD_COMPLIANCE_REWARD}"
        --yield_violation_penalty "${YIELD_VIOLATION_PENALTY}"
        --priority_progress_reward "${PRIORITY_PROGRESS_REWARD}"
        --priority_idle_penalty "${PRIORITY_IDLE_PENALTY}"
        --shield_trigger_dist "${SHIELD_TRIGGER_DIST}"
        --shield_hard_dist "${SHIELD_HARD_DIST}"
        --max_reverse_speed "${MAX_REVERSE_SPEED}"
        --max_linear_speed "${MAX_LINEAR_SPEED}"
        --max_angular_speed "${MAX_ANGULAR_SPEED}"
        --tracking_viz_interval "${TRACKING_VIZ_INTERVAL}"
        --intent_viz_interval "${INTENT_VIZ_INTERVAL}"
        --intent_viz_horizon_sec "${INTENT_VIZ_HORIZON_SEC}"
        --intent_viz_topic "${INTENT_VIZ_TOPIC}"
        --shield_viz_topic "${SHIELD_VIZ_TOPIC}"
        --env_log_level "${ENV_LOG_LEVEL}"
        --sim_wait_wall_timeout "${SIM_WAIT_WALL_TIMEOUT}"
        --output_dir "${OUTPUT_DIR}"
        --run_name "${RUN_NAME}"
    )

    if (( ENABLE_SHIELD == 1 )); then
        cmd+=(--enable_safety_shield)
    else
        cmd+=(--no-enable_safety_shield)
    fi
    if (( ENABLE_YIELDING == 1 )); then
        cmd+=(--enable_yielding)
    else
        cmd+=(--no-enable_yielding)
    fi
    if (( ENABLE_YIELD_ACTION == 1 )); then
        cmd+=(--enable_yield_action)
    else
        cmd+=(--no-enable_yield_action)
    fi
    if (( ENABLE_YIELD_REWARD == 1 )); then
        cmd+=(--enable_yield_reward)
    else
        cmd+=(--no-enable_yield_reward)
    fi
    if (( ENABLE_VISUALIZATION == 1 )); then
        cmd+=(--enable_visualization)
    else
        cmd+=(--no-enable_visualization)
    fi
    if (( ENABLE_INTENT_VISUALIZATION == 1 )); then
        cmd+=(--enable_intent_visualization)
    else
        cmd+=(--no-enable_intent_visualization)
    fi
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi

    banner "Run training (map 5)"
    info "Command: ${cmd[*]}"
    info "Training log: ${TRAIN_LOG}"

    set +e
    "${cmd[@]}" 2>&1 | tee "${TRAIN_LOG}"
    local exit_code=${PIPESTATUS[0]}
    set -e

    if [[ ${exit_code} -ne 0 ]]; then
        error "Training failed (code=${exit_code}). See ${TRAIN_LOG}"
        return "${exit_code}"
    fi
}

cleanup_on_interrupt() {
    echo ""
    warn "Interrupted."
    if (( LAUNCH_ENV == 1 && ENV_STARTED == 1 )); then
        stop_ros_env
    fi
    exit 130
}
trap cleanup_on_interrupt SIGINT SIGTERM

main() {
    source_env
    check_env

    if (( LAUNCH_ENV == 1 )); then
        stop_ros_env
        start_target_env
    fi

    run_training

    if (( LAUNCH_ENV == 1 && ENV_STARTED == 1 )); then
        stop_ros_env
    fi

    banner "=== training finished ==="
    info "train log: ${TRAIN_LOG}"
    if (( LAUNCH_ENV == 1 )); then
        info "env log:   ${ENV_LOG}"
    fi
}

main
