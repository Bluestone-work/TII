#!/bin/bash
# =============================================================================
# One-click ORCA/DWA small tuner with configurable map for BC design.
# Flow: stop old env -> launch target map env -> tune (20 episodes/trial) -> cleanup.
#
# Usage:
#   ./run_bc_orca_dwa_tune_map1.sh
#   ./run_bc_orca_dwa_tune_map1.sh --map_number 1
#   ./run_bc_orca_dwa_tune_map1.sh --num_trials 20 --episodes 20 --num_agents 4 --map_number 3
#   ./run_bc_orca_dwa_tune_map1.sh --enable_visualization
# =============================================================================

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
ROS_SETUP="/opt/ros/humble/setup.bash"
WS_SETUP="${WORKSPACE}/install/setup.bash"
KILL_SCRIPT="${WORKSPACE}/kill_all_ros.sh"

NUM_AGENTS=4
MAP_NUMBER=1
EPISODES=20
NUM_TRIALS=16
SEED=42
SCORE_MODE="safety_first"
RANK_MODE="lexicographic"
MIN_SUCCESS_RATE=0.35
MAX_COLLISION_RATE=0.20
MAX_TIMEOUT_RATE=0.35
INTENT_HORIZON_SEC=1.8
INTENT_DT_SEC=0.2
INTENT_SAFE_MARGIN=0.12
INTENT_COMMIT_STEPS=4
INTENT_REPLAN_INTERVAL_STEPS=2
INTENT_MAX_STALENESS_STEPS=20
ENABLE_VISUALIZATION=1
TRACKING_VIZ_INTERVAL=4
OUTPUT_DIR="${WORKSPACE}/bc_tune_results"
TAG=""
GAZEBO_WAIT_SEC=60
GAZEBO_GRACE_SEC=5

LOG_DIR="${WORKSPACE}/bc_tune_logs"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

cleanup() {
    echo ""
    warn "Interrupted. Cleaning ROS/Gazebo..."
    bash "${KILL_SCRIPT}" 2>/dev/null || true
    exit 130
}
trap cleanup SIGINT SIGTERM

usage() {
    cat <<'EOF'
One-click ORCA/DWA tuner with configurable map.

Usage:
  ./run_bc_orca_dwa_tune_map1.sh
  ./run_bc_orca_dwa_tune_map1.sh --map_number 1
  ./run_bc_orca_dwa_tune_map1.sh --num_trials 20 --episodes 20 --num_agents 4 --map_number 3
  ./run_bc_orca_dwa_tune_map1.sh --min_success_rate 0.45 --max_collision_rate 0.18
  ./run_bc_orca_dwa_tune_map1.sh --intent_horizon_sec 2.2 --intent_commit_steps 5
  ./run_bc_orca_dwa_tune_map1.sh --enable_visualization
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_agents) NUM_AGENTS="$2"; shift 2 ;;
        --map_number) MAP_NUMBER="$2"; shift 2 ;;
        --episodes) EPISODES="$2"; shift 2 ;;
        --num_trials) NUM_TRIALS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --score_mode) SCORE_MODE="$2"; shift 2 ;;
        --rank_mode) RANK_MODE="$2"; shift 2 ;;
        --min_success_rate) MIN_SUCCESS_RATE="$2"; shift 2 ;;
        --max_collision_rate) MAX_COLLISION_RATE="$2"; shift 2 ;;
        --max_timeout_rate) MAX_TIMEOUT_RATE="$2"; shift 2 ;;
        --intent_horizon_sec) INTENT_HORIZON_SEC="$2"; shift 2 ;;
        --intent_dt_sec) INTENT_DT_SEC="$2"; shift 2 ;;
        --intent_safe_margin) INTENT_SAFE_MARGIN="$2"; shift 2 ;;
        --intent_commit_steps) INTENT_COMMIT_STEPS="$2"; shift 2 ;;
        --intent_replan_interval_steps) INTENT_REPLAN_INTERVAL_STEPS="$2"; shift 2 ;;
        --intent_max_staleness_steps) INTENT_MAX_STALENESS_STEPS="$2"; shift 2 ;;
        --tracking_viz_interval) TRACKING_VIZ_INTERVAL="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --enable_visualization) ENABLE_VISUALIZATION=1; shift 1 ;;
        --disable_visualization) ENABLE_VISUALIZATION=0; shift 1 ;;
        --gazebo_wait_sec) GAZEBO_WAIT_SEC="$2"; shift 2 ;;
        --gazebo_grace_sec) GAZEBO_GRACE_SEC="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) error "Unknown arg: $1"; exit 1 ;;
    esac
done

if ! [[ "${MAP_NUMBER}" =~ ^[1-5]$ ]]; then
    error "--map_number must be one of: 1 2 3 4 5"
    exit 1
fi

source_env() {
    set +u
    if [[ -f "${CONDA_SETUP}" ]]; then
        source "${CONDA_SETUP}"
        conda activate ros2 >/dev/null 2>&1 || warn "Failed to activate conda env ros2."
    fi
    [[ -f "${ROS_SETUP}" ]] && source "${ROS_SETUP}" || { set -u; error "Missing ROS setup: ${ROS_SETUP}"; exit 1; }
    [[ -f "${WS_SETUP}"  ]] && source "${WS_SETUP}"  || warn "Workspace not built yet: ${WS_SETUP}"
    set -u
}

check_env() {
    banner "=== ORCA/DWA tuner check ==="
    [[ -f "${KILL_SCRIPT}" ]] || { error "Missing kill script: ${KILL_SCRIPT}"; exit 1; }
    command -v ros2 >/dev/null 2>&1 || { error "ros2 not found"; exit 1; }

    if ! ros2 pkg prefix gnn_bc_tools >/dev/null 2>&1; then
        error "Package gnn_bc_tools unavailable. Build first:"
        echo "  cd ${WORKSPACE} && colcon build --packages-select gnn_marl_training gnn_bc_tools && source install/setup.bash"
        exit 1
    fi
    success "Env check passed."
}

stop_ros_env() {
    info "Stopping old ROS/Gazebo..."
    bash "${KILL_SCRIPT}" 2>/dev/null || true
    sleep 2
}

start_target_env() {
    local log="${LOG_DIR}/map${MAP_NUMBER}_ros.log"
    banner "Start map${MAP_NUMBER} env for tuning"
    info "map=${MAP_NUMBER}, robots=${NUM_AGENTS}, obstacles=0, speed=0.0"

    local inner_cmd
    inner_cmd="set +u"
    inner_cmd+="; source '${ROS_SETUP}'"
    inner_cmd+="; [[ -f '${WS_SETUP}' ]] && source '${WS_SETUP}'"
    inner_cmd+="; set -u"
    inner_cmd+="; ros2 launch start_rl_environment_tb3 main.launch.py"
    inner_cmd+=" map_number:=${MAP_NUMBER} robot_number:=${NUM_AGENTS}"
    inner_cmd+=" num_obstacles:=0 obs_speed_scale:=0.0"
    inner_cmd+=" 2>&1 | tee '${log}'"

    if command -v gnome-terminal >/dev/null 2>&1; then
        gnome-terminal --title="[BC Tune] map${MAP_NUMBER} env" -- bash -c "${inner_cmd}" &
        info "Started env in gnome-terminal."
    elif command -v xterm >/dev/null 2>&1; then
        xterm -title "[BC Tune] map${MAP_NUMBER} env" -e bash -c "${inner_cmd}" &
        info "Started env in xterm."
    else
        warn "No GUI terminal found, running in background."
        bash -c "${inner_cmd}" &
    fi

    info "Waiting Gazebo ready (max ${GAZEBO_WAIT_SEC}s)..."
    local waited=0
    while [[ ${waited} -lt ${GAZEBO_WAIT_SEC} ]]; do
        if ros2 topic list 2>/dev/null | grep -Eq "/tb3_0/(scan|odom)"; then
            echo ""
            success "Gazebo ready (${waited}s)."
            sleep "${GAZEBO_GRACE_SEC}"
            return 0
        fi
        if [[ -f "${log}" ]] && grep -q "Successfully spawned entity \[tb3_0\]" "${log}"; then
            echo ""
            success "Robot spawn detected (${waited}s)."
            sleep "${GAZEBO_GRACE_SEC}"
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
        printf "\r  waiting... %ds / %ds" "${waited}" "${GAZEBO_WAIT_SEC}"
    done
    echo ""
    warn "Gazebo wait timeout. Continue anyway."
}

run_tuner() {
    local log="${LOG_DIR}/tune_map${MAP_NUMBER}.log"
    local cmd=(
        ros2 run gnn_bc_tools tune_orca_dwa_map1
        --map_number "${MAP_NUMBER}"
        --episodes "${EPISODES}"
        --num_trials "${NUM_TRIALS}"
        --num_agents "${NUM_AGENTS}"
        --seed "${SEED}"
        --score_mode "${SCORE_MODE}"
        --rank_mode "${RANK_MODE}"
        --min_success_rate "${MIN_SUCCESS_RATE}"
        --max_collision_rate "${MAX_COLLISION_RATE}"
        --max_timeout_rate "${MAX_TIMEOUT_RATE}"
        --intent_horizon_sec "${INTENT_HORIZON_SEC}"
        --intent_dt_sec "${INTENT_DT_SEC}"
        --intent_safe_margin "${INTENT_SAFE_MARGIN}"
        --intent_commit_steps "${INTENT_COMMIT_STEPS}"
        --intent_replan_interval_steps "${INTENT_REPLAN_INTERVAL_STEPS}"
        --intent_max_staleness_steps "${INTENT_MAX_STALENESS_STEPS}"
        --tracking_viz_interval "${TRACKING_VIZ_INTERVAL}"
        --output_dir "${OUTPUT_DIR}"
    )
    if [[ -n "${TAG}" ]]; then
        cmd+=(--tag "${TAG}")
    fi
    if (( ENABLE_VISUALIZATION == 1 )); then
        cmd+=(--enable_visualization)
    fi

    banner "Run tuner"
    info "Command: ${cmd[*]}"
    info "Log: ${log}"

    set +e
    "${cmd[@]}" 2>&1 | tee "${log}"
    local exit_code=${PIPESTATUS[0]}
    set -e

    if [[ ${exit_code} -ne 0 ]]; then
        error "Tuner failed (code=${exit_code}). See ${log}"
        return "${exit_code}"
    fi
}

main() {
    source_env
    check_env
    stop_ros_env
    start_target_env
    run_tuner
    stop_ros_env
    banner "=== tuning finished ==="
    info "Logs: ${LOG_DIR}"
    info "Results: ${OUTPUT_DIR}"
}

main
