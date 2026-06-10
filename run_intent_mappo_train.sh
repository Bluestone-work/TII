#!/bin/bash
# One-click launcher for intent-aware MAPPO training.
# Flow:
#   1) optional: stop old ROS/Gazebo and launch target map env
#   2) run intent_marl_training/train_intent_mappo
#   3) optional: cleanup launched ROS/Gazebo

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
MAP_NUMBER=3
NUM_AGENTS=4
NUM_WORKERS=1
TRAIN_STEPS=500000
CHECKPOINT_FREQ=20
TRAIN_BATCH_SIZE=4000
ROLLOUT_FRAGMENT_LENGTH=200
RUN_NAME="intent_mappo"
OUTPUT_DIR="${WORKSPACE}/ray_results"

COMMUNICATION_RANGE=3.5
INTENT_TOP_K=3
INTENT_HISTORY_STEPS=4
INTENT_DT=0.1
ENABLE_YIELDING=1
ENABLE_YIELD_ACTION=0
ENABLE_YIELD_REWARD=1
YIELD_OBS_DIM=14
YIELD_DIST_THRESHOLD=1.35
YIELD_TTC_THRESHOLD=3.00
YIELD_RELEASE_DIST=1.80
YIELD_RELEASE_TTC=3.60
YIELD_COMMIT_STEPS=10
YIELD_TIE_MARGIN=0.35
YIELD_LINEAR_SCALE=0.12
YIELD_LINEAR_STOP=0.00
YIELD_LINEAR_STOP_TTC=1.20
YIELD_TURN_BIAS=0.45
YIELD_OBEY_SPEED_THRESH=0.06
PRIORITY_MOVE_SPEED_THRESH=0.08
YIELD_COMPLIANCE_REWARD=0.60
YIELD_VIOLATION_PENALTY=1.20
PRIORITY_PROGRESS_REWARD=0.25
PRIORITY_IDLE_PENALTY=0.25
SOCIAL_CONTROLLER_ENABLE=0
SOCIAL_ACTION_CURRICULUM_ENABLE=1
SOCIAL_ACTION_WARMUP_EPISODES=2
SOCIAL_ACTION_RAMP_EPISODES=10
SOCIAL_ACTION_MIN_SCALE=0.05
SOCIAL_ACTION_EMERGENCY_SEVERITY=0.72
SOCIAL_ACTION_EMERGENCY_DIST=0.72
SOCIAL_PREFERRED_SIDE="right"
SOCIAL_COLLISION_PENALTY=6.0
SOCIAL_COLLISION_PARTNER_PENALTY=3.0
SOCIAL_COLLISION_REWARD_CAP=-6.0
SOCIAL_RESERVATION_STEPS=24
SOCIAL_RESERVATION_RELEASE_DIST=2.40
SOCIAL_HANDOFF_WAIT_STEPS=48
SOCIAL_HANDOFF_COOLDOWN_STEPS=120
SOCIAL_HANDOFF_PARTNER_STALL_STEPS=14
SOCIAL_STARVATION_PENALTY=0.60
SOCIAL_HANDOFF_BONUS=0.45
ZONE_RESERVATION_ENABLE=0
ZONE_OWNER_HOLD_STEPS=30
ZONE_RELEASE_DIST=2.15
ENABLE_SHIELD=0
SHIELD_TURN_GAIN=0.85
SHIELD_TRIGGER_DIST=0.70
SHIELD_HARD_DIST=0.38
MAX_REVERSE_SPEED=0.08
COLLISION_PENALTY=90.0
NEAR_COLLISION_PENALTY_SCALE=3.0
FRONT_SAFETY_PENALTY_SCALE=1.4
NEIGHBOR_SAFETY_PENALTY_SCALE=2.8
BASE_SHIELD_ENABLE=0
BASE_TRACKING_ASSIST_ENABLE=0
LOCAL_EXECUTOR_ENABLE=0
LOCAL_EXECUTOR_NOMINAL_SPEED=0.18
LOCAL_EXECUTOR_OBSTACLE_GAIN=0.80
LOCAL_EXECUTOR_NEIGHBOR_GAIN=1.20
LOCAL_EXECUTOR_TANGENTIAL_GAIN=0.70
LOCAL_EXECUTOR_OBSTACLE_INFLUENCE_DIST=0.95
LOCAL_EXECUTOR_NEIGHBOR_INFLUENCE_DIST=1.55
LOCAL_EXECUTOR_FRONT_SLOW_DIST=0.75
LOCAL_EXECUTOR_HARD_STOP_DIST=0.24
LOCAL_EXECUTOR_TURN_IN_PLACE_ANGLE=0.95
LOCAL_EXECUTOR_HEADING_GAIN=1.55
LOCAL_EXECUTOR_PREFERRED_SIDE="right"
LOCAL_EXECUTOR_SCAN_STRIDE=6
LOCAL_EXECUTOR_ACTION_BLEND=0.18
LOCAL_EXECUTOR_ANGULAR_ACTION_BLEND=0.22
LOCAL_EXECUTOR_REVERSE_ESCAPE_GAIN=0.75
BASE_ZONE_MANAGER_ENABLE=0
BASE_ZONE_OWNER_HOLD_STEPS=30
BASE_ZONE_RELEASE_DIST=2.15
BASE_ZONE_QUEUE_SPACING=0.38
BASE_ZONE_OWNER_PROGRESS_EPSILON=0.04
BASE_ZONE_OWNER_STALL_STEPS=20
BASE_ZONE_FORCE_HANDOFF_WAIT_STEPS=54
BASE_HYBRID_CONTROL_ENABLE=0

ENABLE_VISUALIZATION=1
ENABLE_INTENT_VISUALIZATION=1
TRACKING_VIZ_INTERVAL=4
INTENT_VIZ_INTERVAL=4
INTENT_VIZ_HORIZON_SEC=1.2
INTENT_VIZ_TOPIC="/intent_marl/intent_markers"
SHIELD_VIZ_TOPIC="/intent_marl/shield_markers"
ENV_LOG_LEVEL="INFO"

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
Intent MAPPO training launcher.

Usage:
  ./run_intent_mappo_train.sh
  ./run_intent_mappo_train.sh --env_stage 1 --map_number 3 --num_agents 4
  ./run_intent_mappo_train.sh --skip_launch_env --train_steps 200000
  ./run_intent_mappo_train.sh -- --lr 1e-4 --sample_timeout_s 1800

Options:
  --env_stage N
  --map_number N
  --num_agents N
  --num_workers N
  --train_steps N
  --checkpoint_freq N
  --train_batch_size N
  --rollout_fragment_length N
  --run_name NAME
  --output_dir DIR
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
  --social_action_curriculum_enable / --no-social_action_curriculum_enable
  --social_action_warmup_episodes N
  --social_action_ramp_episodes N
  --social_action_min_scale X
  --social_action_emergency_severity X
  --social_action_emergency_dist X
  --social_collision_penalty X
  --social_collision_partner_penalty X
  --social_collision_reward_cap X
  --social_reservation_steps N
  --social_reservation_release_dist X
  --social_handoff_wait_steps N
  --social_handoff_cooldown_steps N
  --social_handoff_partner_stall_steps N
  --social_starvation_penalty X
  --social_handoff_bonus X
  --zone_reservation_enable / --no-zone_reservation_enable
  --zone_owner_hold_steps N
  --zone_release_dist X
  --collision_penalty X
  --near_collision_penalty_scale X
  --front_safety_penalty_scale X
  --neighbor_safety_penalty_scale X
  --base_shield_enable / --no-base_shield_enable
  --base_tracking_assist_enable / --no-base_tracking_assist_enable
  --local_executor_enable / --no-local_executor_enable
  --local_executor_nominal_speed X
  --local_executor_obstacle_gain X
  --local_executor_neighbor_gain X
  --local_executor_tangential_gain X
  --local_executor_obstacle_influence_dist X
  --local_executor_neighbor_influence_dist X
  --local_executor_front_slow_dist X
  --local_executor_hard_stop_dist X
  --local_executor_turn_in_place_angle X
  --local_executor_heading_gain X
  --local_executor_preferred_side {right,left}
  --local_executor_scan_stride N
  --local_executor_action_blend X
  --local_executor_angular_action_blend X
  --local_executor_reverse_escape_gain X
  --base_zone_manager_enable / --no-base_zone_manager_enable
  --base_zone_owner_hold_steps N
  --base_zone_release_dist X
  --base_zone_queue_spacing X
  --base_zone_owner_progress_epsilon X
  --base_zone_owner_stall_steps N
  --base_zone_force_handoff_wait_steps N
  --base_hybrid_control_enable / --no-base_hybrid_control_enable
  --enable_safety_shield / --disable_safety_shield
  --shield_trigger_dist X
  --shield_hard_dist X
  --max_reverse_speed X
  --enable_visualization / --disable_visualization
  --enable_intent_visualization / --disable_intent_visualization
  --tracking_viz_interval N
  --intent_viz_interval N
  --intent_viz_horizon_sec X
  --intent_viz_topic TOPIC
  --shield_viz_topic TOPIC
  --env_log_level LEVEL
  --skip_launch_env
  --gazebo_wait_sec N
  --gazebo_grace_sec N
  --help

Notes:
  1) Use "-- <extra args>" to pass through any additional train_intent_mappo args.
  2) When --skip_launch_env is set, this script only runs training and will not stop ROS.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env_stage) ENV_STAGE="$2"; shift 2 ;;
        --map_number) MAP_NUMBER="$2"; shift 2 ;;
        --num_agents) NUM_AGENTS="$2"; shift 2 ;;
        --num_workers) NUM_WORKERS="$2"; shift 2 ;;
        --train_steps) TRAIN_STEPS="$2"; shift 2 ;;
        --checkpoint_freq) CHECKPOINT_FREQ="$2"; shift 2 ;;
        --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --rollout_fragment_length) ROLLOUT_FRAGMENT_LENGTH="$2"; shift 2 ;;
        --run_name) RUN_NAME="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
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
        --social_action_curriculum_enable) SOCIAL_ACTION_CURRICULUM_ENABLE=1; shift 1 ;;
        --no-social_action_curriculum_enable) SOCIAL_ACTION_CURRICULUM_ENABLE=0; shift 1 ;;
        --social_action_warmup_episodes) SOCIAL_ACTION_WARMUP_EPISODES="$2"; shift 2 ;;
        --social_action_ramp_episodes) SOCIAL_ACTION_RAMP_EPISODES="$2"; shift 2 ;;
        --social_action_min_scale) SOCIAL_ACTION_MIN_SCALE="$2"; shift 2 ;;
        --social_action_emergency_severity) SOCIAL_ACTION_EMERGENCY_SEVERITY="$2"; shift 2 ;;
        --social_action_emergency_dist) SOCIAL_ACTION_EMERGENCY_DIST="$2"; shift 2 ;;
        --social_collision_penalty) SOCIAL_COLLISION_PENALTY="$2"; shift 2 ;;
        --social_collision_partner_penalty) SOCIAL_COLLISION_PARTNER_PENALTY="$2"; shift 2 ;;
        --social_collision_reward_cap) SOCIAL_COLLISION_REWARD_CAP="$2"; shift 2 ;;
        --social_reservation_steps) SOCIAL_RESERVATION_STEPS="$2"; shift 2 ;;
        --social_reservation_release_dist) SOCIAL_RESERVATION_RELEASE_DIST="$2"; shift 2 ;;
        --social_handoff_wait_steps) SOCIAL_HANDOFF_WAIT_STEPS="$2"; shift 2 ;;
        --social_handoff_cooldown_steps) SOCIAL_HANDOFF_COOLDOWN_STEPS="$2"; shift 2 ;;
        --social_handoff_partner_stall_steps) SOCIAL_HANDOFF_PARTNER_STALL_STEPS="$2"; shift 2 ;;
        --social_starvation_penalty) SOCIAL_STARVATION_PENALTY="$2"; shift 2 ;;
        --social_handoff_bonus) SOCIAL_HANDOFF_BONUS="$2"; shift 2 ;;
        --zone_reservation_enable) ZONE_RESERVATION_ENABLE=1; shift 1 ;;
        --no-zone_reservation_enable) ZONE_RESERVATION_ENABLE=0; shift 1 ;;
        --zone_owner_hold_steps) ZONE_OWNER_HOLD_STEPS="$2"; shift 2 ;;
        --zone_release_dist) ZONE_RELEASE_DIST="$2"; shift 2 ;;
        --collision_penalty) COLLISION_PENALTY="$2"; shift 2 ;;
        --near_collision_penalty_scale) NEAR_COLLISION_PENALTY_SCALE="$2"; shift 2 ;;
        --front_safety_penalty_scale) FRONT_SAFETY_PENALTY_SCALE="$2"; shift 2 ;;
        --neighbor_safety_penalty_scale) NEIGHBOR_SAFETY_PENALTY_SCALE="$2"; shift 2 ;;
        --base_shield_enable) BASE_SHIELD_ENABLE=1; shift 1 ;;
        --no-base_shield_enable) BASE_SHIELD_ENABLE=0; shift 1 ;;
        --base_tracking_assist_enable) BASE_TRACKING_ASSIST_ENABLE=1; shift 1 ;;
        --no-base_tracking_assist_enable) BASE_TRACKING_ASSIST_ENABLE=0; shift 1 ;;
        --local_executor_enable) LOCAL_EXECUTOR_ENABLE=1; shift 1 ;;
        --no-local_executor_enable) LOCAL_EXECUTOR_ENABLE=0; shift 1 ;;
        --local_executor_nominal_speed) LOCAL_EXECUTOR_NOMINAL_SPEED="$2"; shift 2 ;;
        --local_executor_obstacle_gain) LOCAL_EXECUTOR_OBSTACLE_GAIN="$2"; shift 2 ;;
        --local_executor_neighbor_gain) LOCAL_EXECUTOR_NEIGHBOR_GAIN="$2"; shift 2 ;;
        --local_executor_tangential_gain) LOCAL_EXECUTOR_TANGENTIAL_GAIN="$2"; shift 2 ;;
        --local_executor_obstacle_influence_dist) LOCAL_EXECUTOR_OBSTACLE_INFLUENCE_DIST="$2"; shift 2 ;;
        --local_executor_neighbor_influence_dist) LOCAL_EXECUTOR_NEIGHBOR_INFLUENCE_DIST="$2"; shift 2 ;;
        --local_executor_front_slow_dist) LOCAL_EXECUTOR_FRONT_SLOW_DIST="$2"; shift 2 ;;
        --local_executor_hard_stop_dist) LOCAL_EXECUTOR_HARD_STOP_DIST="$2"; shift 2 ;;
        --local_executor_turn_in_place_angle) LOCAL_EXECUTOR_TURN_IN_PLACE_ANGLE="$2"; shift 2 ;;
        --local_executor_heading_gain) LOCAL_EXECUTOR_HEADING_GAIN="$2"; shift 2 ;;
        --local_executor_preferred_side) LOCAL_EXECUTOR_PREFERRED_SIDE="$2"; shift 2 ;;
        --local_executor_scan_stride) LOCAL_EXECUTOR_SCAN_STRIDE="$2"; shift 2 ;;
        --local_executor_action_blend) LOCAL_EXECUTOR_ACTION_BLEND="$2"; shift 2 ;;
        --local_executor_angular_action_blend) LOCAL_EXECUTOR_ANGULAR_ACTION_BLEND="$2"; shift 2 ;;
        --local_executor_reverse_escape_gain) LOCAL_EXECUTOR_REVERSE_ESCAPE_GAIN="$2"; shift 2 ;;
        --base_zone_manager_enable) BASE_ZONE_MANAGER_ENABLE=1; shift 1 ;;
        --no-base_zone_manager_enable) BASE_ZONE_MANAGER_ENABLE=0; shift 1 ;;
        --base_zone_owner_hold_steps) BASE_ZONE_OWNER_HOLD_STEPS="$2"; shift 2 ;;
        --base_zone_release_dist) BASE_ZONE_RELEASE_DIST="$2"; shift 2 ;;
        --base_zone_queue_spacing) BASE_ZONE_QUEUE_SPACING="$2"; shift 2 ;;
        --base_zone_owner_progress_epsilon) BASE_ZONE_OWNER_PROGRESS_EPSILON="$2"; shift 2 ;;
        --base_zone_owner_stall_steps) BASE_ZONE_OWNER_STALL_STEPS="$2"; shift 2 ;;
        --base_zone_force_handoff_wait_steps) BASE_ZONE_FORCE_HANDOFF_WAIT_STEPS="$2"; shift 2 ;;
        --base_hybrid_control_enable) BASE_HYBRID_CONTROL_ENABLE=1; shift 1 ;;
        --no-base_hybrid_control_enable) BASE_HYBRID_CONTROL_ENABLE=0; shift 1 ;;
        --enable_safety_shield) ENABLE_SHIELD=1; shift 1 ;;
        --disable_safety_shield) ENABLE_SHIELD=0; shift 1 ;;
        --shield_trigger_dist) SHIELD_TRIGGER_DIST="$2"; shift 2 ;;
        --shield_hard_dist) SHIELD_HARD_DIST="$2"; shift 2 ;;
        --max_reverse_speed) MAX_REVERSE_SPEED="$2"; shift 2 ;;
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
    banner "=== Intent MAPPO training check ==="
    command -v ros2 >/dev/null 2>&1 || { error "ros2 not found."; exit 1; }
    [[ -f "${TRAIN_ENTRY_SRC}" ]] || {
        error "Training entry script not found: ${TRAIN_ENTRY_SRC}"
        exit 1
    }

    if (( LAUNCH_ENV == 1 )) && ! ros2 pkg prefix start_rl_environment_tb3 >/dev/null 2>&1; then
        error "Package start_rl_environment_tb3 unavailable. Cannot launch env."
        exit 1
    fi

    if ! python3 - <<'PY'
import importlib

missing = []
try:
    importlib.import_module("ray")
except Exception as exc:  # noqa: BLE001
    missing.append(f"ray ({exc})")

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
    ENV_LOG="${LOG_DIR}/intent_env_map${MAP_NUMBER}_${STAMP}.log"
    banner "Launch env map${MAP_NUMBER} for training"
    info "log: ${ENV_LOG}"

    local inner_cmd
    inner_cmd="set +u"
    inner_cmd+="; source '${ROS_SETUP}'"
    inner_cmd+="; [[ -f '${WS_SETUP}' ]] && source '${WS_SETUP}'"
    inner_cmd+="; set -u"
    inner_cmd+="; ros2 launch start_rl_environment_tb3 main.launch.py"
    inner_cmd+=" map_number:=${MAP_NUMBER} robot_number:=${NUM_AGENTS}"
    inner_cmd+=" num_obstacles:=0 obs_speed_scale:=0.0"
    inner_cmd+=" 2>&1 | tee '${ENV_LOG}'"

    if command -v gnome-terminal >/dev/null 2>&1; then
        gnome-terminal --title="[Intent MAPPO] env map${MAP_NUMBER}" -- bash -c "${inner_cmd}" &
    elif command -v xterm >/dev/null 2>&1; then
        xterm -title "[Intent MAPPO] env map${MAP_NUMBER}" -e bash -c "${inner_cmd}" &
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
        --map_number "${MAP_NUMBER}"
        --num_agents "${NUM_AGENTS}"
        --num_workers "${NUM_WORKERS}"
        --train_steps "${TRAIN_STEPS}"
        --checkpoint_freq "${CHECKPOINT_FREQ}"
        --train_batch_size "${TRAIN_BATCH_SIZE}"
        --rollout_fragment_length "${ROLLOUT_FRAGMENT_LENGTH}"
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
        --social_action_warmup_episodes "${SOCIAL_ACTION_WARMUP_EPISODES}"
        --social_action_ramp_episodes "${SOCIAL_ACTION_RAMP_EPISODES}"
        --social_action_min_scale "${SOCIAL_ACTION_MIN_SCALE}"
        --social_action_emergency_severity "${SOCIAL_ACTION_EMERGENCY_SEVERITY}"
        --social_action_emergency_dist "${SOCIAL_ACTION_EMERGENCY_DIST}"
        --social_collision_penalty "${SOCIAL_COLLISION_PENALTY}"
        --social_collision_partner_penalty "${SOCIAL_COLLISION_PARTNER_PENALTY}"
        --social_collision_reward_cap "${SOCIAL_COLLISION_REWARD_CAP}"
        --social_reservation_steps "${SOCIAL_RESERVATION_STEPS}"
        --social_reservation_release_dist "${SOCIAL_RESERVATION_RELEASE_DIST}"
        --social_handoff_wait_steps "${SOCIAL_HANDOFF_WAIT_STEPS}"
        --social_handoff_cooldown_steps "${SOCIAL_HANDOFF_COOLDOWN_STEPS}"
        --social_handoff_partner_stall_steps "${SOCIAL_HANDOFF_PARTNER_STALL_STEPS}"
        --social_starvation_penalty "${SOCIAL_STARVATION_PENALTY}"
        --social_handoff_bonus "${SOCIAL_HANDOFF_BONUS}"
        --zone_owner_hold_steps "${ZONE_OWNER_HOLD_STEPS}"
        --zone_release_dist "${ZONE_RELEASE_DIST}"
        --collision_penalty "${COLLISION_PENALTY}"
        --near_collision_penalty_scale "${NEAR_COLLISION_PENALTY_SCALE}"
        --front_safety_penalty_scale "${FRONT_SAFETY_PENALTY_SCALE}"
        --neighbor_safety_penalty_scale "${NEIGHBOR_SAFETY_PENALTY_SCALE}"
        --local_executor_nominal_speed "${LOCAL_EXECUTOR_NOMINAL_SPEED}"
        --local_executor_obstacle_gain "${LOCAL_EXECUTOR_OBSTACLE_GAIN}"
        --local_executor_neighbor_gain "${LOCAL_EXECUTOR_NEIGHBOR_GAIN}"
        --local_executor_tangential_gain "${LOCAL_EXECUTOR_TANGENTIAL_GAIN}"
        --local_executor_obstacle_influence_dist "${LOCAL_EXECUTOR_OBSTACLE_INFLUENCE_DIST}"
        --local_executor_neighbor_influence_dist "${LOCAL_EXECUTOR_NEIGHBOR_INFLUENCE_DIST}"
        --local_executor_front_slow_dist "${LOCAL_EXECUTOR_FRONT_SLOW_DIST}"
        --local_executor_hard_stop_dist "${LOCAL_EXECUTOR_HARD_STOP_DIST}"
        --local_executor_turn_in_place_angle "${LOCAL_EXECUTOR_TURN_IN_PLACE_ANGLE}"
        --local_executor_heading_gain "${LOCAL_EXECUTOR_HEADING_GAIN}"
        --local_executor_preferred_side "${LOCAL_EXECUTOR_PREFERRED_SIDE}"
        --local_executor_scan_stride "${LOCAL_EXECUTOR_SCAN_STRIDE}"
        --local_executor_action_blend "${LOCAL_EXECUTOR_ACTION_BLEND}"
        --local_executor_angular_action_blend "${LOCAL_EXECUTOR_ANGULAR_ACTION_BLEND}"
        --local_executor_reverse_escape_gain "${LOCAL_EXECUTOR_REVERSE_ESCAPE_GAIN}"
        --base_zone_owner_hold_steps "${BASE_ZONE_OWNER_HOLD_STEPS}"
        --base_zone_release_dist "${BASE_ZONE_RELEASE_DIST}"
        --base_zone_queue_spacing "${BASE_ZONE_QUEUE_SPACING}"
        --base_zone_owner_progress_epsilon "${BASE_ZONE_OWNER_PROGRESS_EPSILON}"
        --base_zone_owner_stall_steps "${BASE_ZONE_OWNER_STALL_STEPS}"
        --base_zone_force_handoff_wait_steps "${BASE_ZONE_FORCE_HANDOFF_WAIT_STEPS}"
        --social_preferred_side "${SOCIAL_PREFERRED_SIDE}"
        --shield_turn_gain "${SHIELD_TURN_GAIN}"
        --shield_trigger_dist "${SHIELD_TRIGGER_DIST}"
        --shield_hard_dist "${SHIELD_HARD_DIST}"
        --max_reverse_speed "${MAX_REVERSE_SPEED}"
        --tracking_viz_interval "${TRACKING_VIZ_INTERVAL}"
        --intent_viz_interval "${INTENT_VIZ_INTERVAL}"
        --intent_viz_horizon_sec "${INTENT_VIZ_HORIZON_SEC}"
        --intent_viz_topic "${INTENT_VIZ_TOPIC}"
        --shield_viz_topic "${SHIELD_VIZ_TOPIC}"
        --env_log_level "${ENV_LOG_LEVEL}"
        --output_dir "${OUTPUT_DIR}"
        --run_name "${RUN_NAME}"
    )

    if (( ENABLE_SHIELD == 1 )); then
        cmd+=(--enable_safety_shield)
    else
        cmd+=(--no-enable_safety_shield)
    fi
    if (( BASE_SHIELD_ENABLE == 1 )); then
        cmd+=(--base_shield_enable)
    else
        cmd+=(--no-base_shield_enable)
    fi
    if (( BASE_TRACKING_ASSIST_ENABLE == 1 )); then
        cmd+=(--base_tracking_assist_enable)
    else
        cmd+=(--no-base_tracking_assist_enable)
    fi
    if (( LOCAL_EXECUTOR_ENABLE == 1 )); then
        cmd+=(--local_executor_enable)
    else
        cmd+=(--no-local_executor_enable)
    fi
    if (( BASE_ZONE_MANAGER_ENABLE == 1 )); then
        cmd+=(--base_zone_manager_enable)
    else
        cmd+=(--no-base_zone_manager_enable)
    fi
    if (( BASE_HYBRID_CONTROL_ENABLE == 1 )); then
        cmd+=(--base_hybrid_control_enable)
    else
        cmd+=(--no-base_hybrid_control_enable)
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
    if (( SOCIAL_CONTROLLER_ENABLE == 1 )); then
        cmd+=(--social_controller_enable)
    else
        cmd+=(--no-social_controller_enable)
    fi
    if (( SOCIAL_ACTION_CURRICULUM_ENABLE == 1 )); then
        cmd+=(--social_action_curriculum_enable)
    else
        cmd+=(--no-social_action_curriculum_enable)
    fi
    if (( ZONE_RESERVATION_ENABLE == 1 )); then
        cmd+=(--zone_reservation_enable)
    else
        cmd+=(--no-zone_reservation_enable)
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

    banner "Run training"
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
