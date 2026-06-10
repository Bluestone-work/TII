#!/bin/bash
# End-to-End MAPPO curriculum launcher (simplified)

if [ -z "${BASH_VERSION:-}" ]; then
    exec /bin/bash "$0" "$@"
fi

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; RESET='\033[0m'
info()    { echo -e "${CYAN}[INFO]${RESET} $*"; }
success() { echo -e "${GREEN}[OK]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET} $*"; }
error()   { echo -e "${RED}[ERR]${RESET} $*" >&2; }

WORKSPACE="/home/wj/work/multi-robot-exploration-rl"
CONDA_SETUP="/home/wj/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="ros2"
ROS_SETUP="/opt/ros/humble/setup.bash"
WS_SETUP="${WORKSPACE}/install/setup.bash"
KILL_SCRIPT="${WORKSPACE}/kill_all_ros.sh"
TRAIN_SCRIPT="${WORKSPACE}/src/intent_marl_training_e2e/intent_marl_training_e2e/train_e2e_mappo.py"
LOG_DIR="${WORKSPACE}/curriculum_logs_e2e"
OUTPUT_DIR="${WORKSPACE}/ray_results"
mkdir -p "${LOG_DIR}"

MAX_SEQ_LEN=64
PROGRESS_REWARD_SCALE=6.0
PATH_PROGRESS_REWARD_SCALE=4.0
GOAL_REWARD=60.0
TIME_PENALTY=0.006
IDLE_PENALTY_SCALE=0.030

FRONT_SAFETY_DIST=0.55
LIDAR_NEAR_COLLISION_DIST=0.45
LIDAR_NEAR_COLLISION_PENALTY_SCALE=0.8
NEAR_WALL_PENALTY_DIST=0.30
NEIGHBOR_SAFETY_DIST=0.72

E2E_INTERACTION_DIST=1.25
E2E_NEIGHBOR_SAFE_DIST=0.72
E2E_NEIGHBOR_PENALTY_SCALE=0.35
E2E_ESCAPE_REWARD_SCALE=1.0
E2E_APPROACH_PENALTY_SCALE=0.05
E2E_CLEARANCE_REWARD_SCALE=0.10
E2E_MAX_ESCAPE_DELTA=0.15

# Minimal knobs
STAGES="1,6,7"
CYCLES=2
TRAIN_STEPS=200000
RUN_NAME="e2e_mappo_v2"
RESUME_CKPT=""

NUM_AGENTS=2
NUM_WORKERS=1
TRAIN_BATCH_SIZE=2000
ROLLOUT_FRAGMENT_LENGTH=100
SAMPLE_TIMEOUT_S=1200
CHECKPOINT_FREQ=20
COMM_MODE="centralized_oracle"
COMMUNICATION_RANGE=3.5

ENABLE_VIS=0
LAUNCH_ENV=1
GAZEBO_WAIT_SEC=60
GAZEBO_GRACE_SEC=5

# Collision-avoidance focused defaults
COLLISION_PENALTY=60.0
FRONT_SAFETY_PENALTY_SCALE=0.25
NEIGHBOR_SAFETY_PENALTY_SCALE=0

LAST_STAGE_CKPT=""

STAGE_MAP_1=3
STAGE_MAP_6=6
STAGE_MAP_7=6

usage() {
cat <<'USAGE'
Usage:
  ./run_e2e_curriculum.sh
  ./run_e2e_curriculum.sh --stages 1,6,7 --cycles 2 --steps 120000
  ./run_e2e_curriculum.sh --resume /abs/path/checkpoint_000123

Options:
  --stages LIST          课程阶段列表，逗号分隔（仅支持 1,6,7），默认 6,7
  --cycles N             阶段序列循环次数，默认 1
  --steps N              每阶段训练步数，默认 200000
  --run_name NAME        训练前缀名，默认 e2e_mappo
  --output_dir DIR       checkpoint 输出目录
  --resume PATH          从已有 checkpoint 开始
  --num_workers N        RLlib workers，默认 1
  --collision_penalty X
  --front_safety_penalty_scale X
  --neighbor_safety_penalty_scale X
  --enable_visualization / --disable_visualization
  --skip_launch_env      不重启 Gazebo/ROS 环境
  -h, --help
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stages) STAGES="$2"; shift 2 ;;
        --cycles) CYCLES="$2"; shift 2 ;;
        --steps) TRAIN_STEPS="$2"; shift 2 ;;
        --run_name) RUN_NAME="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --resume) RESUME_CKPT="$2"; shift 2 ;;
        --num_workers) NUM_WORKERS="$2"; shift 2 ;;
        --collision_penalty) COLLISION_PENALTY="$2"; shift 2 ;;
        --front_safety_penalty_scale) FRONT_SAFETY_PENALTY_SCALE="$2"; shift 2 ;;
        --neighbor_safety_penalty_scale) NEIGHBOR_SAFETY_PENALTY_SCALE="$2"; shift 2 ;;
        --enable_visualization) ENABLE_VIS=1; shift ;;
        --disable_visualization) ENABLE_VIS=0; shift ;;
        --skip_launch_env) LAUNCH_ENV=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) error "Unknown arg: $1"; usage; exit 1 ;;
    esac
done

setup_env() {
    set +u
    [[ -f "$CONDA_SETUP" ]] && source "$CONDA_SETUP" && conda activate "$CONDA_ENV" || true
    [[ -f "$ROS_SETUP" ]] && source "$ROS_SETUP" || { set -u; error "missing ROS setup: $ROS_SETUP"; exit 1; }
    [[ -f "$WS_SETUP" ]] && source "$WS_SETUP" || warn "workspace setup not found: $WS_SETUP"
    set -u
}

stage_to_map() {
    local s="$1"
    case "$s" in
        1) echo "$STAGE_MAP_1" ;;
        6) echo "$STAGE_MAP_6" ;;
        7) echo "$STAGE_MAP_7" ;;
        *) return 1 ;;
    esac
}

parse_stage_sequence() {
    local list="$1"
    local cycles="$2"
    local raw=()
    local raw_valid=()
    local out=()

    IFS=',' read -r -a raw <<< "$list"
    for s in "${raw[@]}"; do
        s="${s//[[:space:]]/}"
        [[ -z "$s" ]] && continue
        case "$s" in
            1|6|7) raw_valid+=("$s") ;;
            *) error "unsupported stage: $s (only 1,6,7)"; exit 1 ;;
        esac
    done

    if [[ ${#raw_valid[@]} -eq 0 ]]; then
        error "stage list is empty"
        exit 1
    fi

    for ((c=0; c<cycles; c++)); do
        for s in "${raw_valid[@]}"; do
            out+=("$s")
        done
    done

    echo "${out[@]}"
}

stop_ros_env() {
    (( LAUNCH_ENV == 0 )) && return 0
    bash "$KILL_SCRIPT" >/dev/null 2>&1 || true
    sleep 2
}

start_ros_env() {
    local stage="$1"
    (( LAUNCH_ENV == 0 )) && return 0

    local map_num
    map_num="$(stage_to_map "$stage")" || { error "invalid stage for map: $stage"; exit 1; }
    local ros_log="${LOG_DIR}/stage${stage}_ros.log"

    info "launch env for stage=${stage}, map=${map_num}"

    local inner
    inner="set +u"
    inner+="; source '${CONDA_SETUP}'"
    inner+="; conda activate '${CONDA_ENV}'"
    inner+="; source '${ROS_SETUP}'"
    inner+="; [[ -f '${WS_SETUP}' ]] && source '${WS_SETUP}'"
    inner+="; set -u"
    inner+="; ros2 launch start_rl_environment_tb3 main.launch.py"
    inner+=" map_number:=${map_num} robot_number:=${NUM_AGENTS} num_obstacles:=0 obs_speed_scale:=0.0"
    inner+=" 2>&1 | tee '${ros_log}'"

    if command -v gnome-terminal >/dev/null 2>&1; then
        gnome-terminal --title="[E2E Stage ${stage}]" -- bash -c "$inner" &
    else
        bash -c "$inner" &
    fi

    local waited=0
    while [[ $waited -lt $GAZEBO_WAIT_SEC ]]; do
        if ros2 topic list 2>/dev/null | grep -Eq "/tb3_0/(scan|odom)"; then
            success "gazebo ready in ${waited}s (topic check)"
            sleep "$GAZEBO_GRACE_SEC"
            return 0
        fi
        if [[ -f "$ros_log" ]] && [[ $(grep -c "Spawn status: SpawnEntity: Successfully spawned entity" "$ros_log" 2>/dev/null || true) -ge "$NUM_AGENTS" ]]; then
            success "gazebo ready in ${waited}s (spawn log check)"
            sleep "$GAZEBO_GRACE_SEC"
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
    done
    warn "gazebo wait timeout (${GAZEBO_WAIT_SEC}s), continue"
}

normalize_ckpt_path() {
    local raw="${1:-}"
    raw="$(echo "$raw" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [[ -z "$raw" ]] && { echo ""; return 0; }
    [[ -e "$raw" ]] && { echo "$raw"; return 0; }
    local parsed
    parsed=$(echo "$raw" | sed -n 's/.*path=\([^),]*\).*/\1/p' | head -1)
    [[ -n "$parsed" && -e "$parsed" ]] && { echo "$parsed"; return 0; }
    echo "$raw"
}

find_latest_ckpt() {
    local stage="$1"
    local run_dir
    run_dir=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "${RUN_NAME}_stage${stage}_n${NUM_AGENTS}*" 2>/dev/null | sort | tail -1)
    [[ -z "$run_dir" ]] && { echo ""; return 1; }
    find "$run_dir" -maxdepth 4 -type d -name "checkpoint_*" 2>/dev/null | sort -V | tail -1
}

run_stage() {
    local stage="$1"
    local restore="$2"
    local log_file="${LOG_DIR}/stage${stage}_train.log"

    local cmd=(
        python3 -u "$TRAIN_SCRIPT"
        --env_stage "$stage"
        --num_agents "$NUM_AGENTS"
        --num_workers "$NUM_WORKERS"
        --train_steps "$TRAIN_STEPS"
        --checkpoint_freq "$CHECKPOINT_FREQ"

        --lr "2.5e-4"
        --train_batch_size "$TRAIN_BATCH_SIZE"
        --rollout_fragment_length "$ROLLOUT_FRAGMENT_LENGTH"
        --sample_timeout_s "$SAMPLE_TIMEOUT_S"
        --max_seq_len "$MAX_SEQ_LEN"

        --comm_mode "$COMM_MODE"
        --communication_range "$COMMUNICATION_RANGE"

        --progress_reward_scale "$PROGRESS_REWARD_SCALE"
        --path_progress_reward_scale "$PATH_PROGRESS_REWARD_SCALE"
        --goal_reward "$GOAL_REWARD"
        --collision_penalty "$COLLISION_PENALTY"
        --time_penalty "$TIME_PENALTY"
        --idle_penalty_scale "$IDLE_PENALTY_SCALE"

        --front_safety_dist "$FRONT_SAFETY_DIST"
        --front_safety_penalty_scale "$FRONT_SAFETY_PENALTY_SCALE"
        --lidar_near_collision_dist "$LIDAR_NEAR_COLLISION_DIST"
        --lidar_near_collision_penalty_scale "$LIDAR_NEAR_COLLISION_PENALTY_SCALE"
        --near_wall_penalty_dist "$NEAR_WALL_PENALTY_DIST"

        --neighbor_safety_dist "$NEIGHBOR_SAFETY_DIST"
        --neighbor_safety_penalty_scale "$NEIGHBOR_SAFETY_PENALTY_SCALE"

        --e2e_interaction_dist "$E2E_INTERACTION_DIST"
        --e2e_neighbor_safe_dist "$E2E_NEIGHBOR_SAFE_DIST"
        --e2e_neighbor_penalty_scale "$E2E_NEIGHBOR_PENALTY_SCALE"
        --e2e_escape_reward_scale "$E2E_ESCAPE_REWARD_SCALE"
        --e2e_approach_penalty_scale "$E2E_APPROACH_PENALTY_SCALE"
        --e2e_clearance_reward_scale "$E2E_CLEARANCE_REWARD_SCALE"
        --e2e_max_escape_delta "$E2E_MAX_ESCAPE_DELTA"

        --run_name "$RUN_NAME"
        --output_dir "$OUTPUT_DIR"
    )

    if (( ENABLE_VIS == 1 )); then
        cmd+=(--enable_visualization)
    else
        cmd+=(--no-enable_visualization)
    fi

    if [[ -n "$restore" ]]; then
        cmd+=(--restore_checkpoint "$restore")
    fi

    info "train stage=${stage}, log=${log_file}"
    set +e
    "${cmd[@]}" 2>&1 | tee "$log_file"
    local rc=${PIPESTATUS[0]}
    set -e
    (( rc == 0 )) || { error "stage ${stage} failed (code=${rc})"; return $rc; }

    local ckpt
    ckpt=$(grep -oP '(?<=final_checkpoint: ).*' "$log_file" | tail -1 || true)
    ckpt="$(normalize_ckpt_path "$ckpt")"
    if [[ -z "$ckpt" ]]; then
        ckpt=$(find_latest_ckpt "$stage" || true)
    fi
    LAST_STAGE_CKPT="$ckpt"
}

cleanup() {
    warn "interrupted, cleaning up"
    stop_ros_env
    exit 130
}
trap cleanup SIGINT SIGTERM

main() {
    setup_env

    [[ -f "$TRAIN_SCRIPT" ]] || { error "train script not found: $TRAIN_SCRIPT"; exit 1; }
    [[ -f "$KILL_SCRIPT" ]] || { error "kill script not found: $KILL_SCRIPT"; exit 1; }

    local stage_seq
    stage_seq=( $(parse_stage_sequence "$STAGES" "$CYCLES") )

    info "stages=${stage_seq[*]} cycles=${CYCLES} steps_per_stage=${TRAIN_STEPS}"
    info "output_dir=${OUTPUT_DIR}"

    local current_ckpt
    current_ckpt="$(normalize_ckpt_path "$RESUME_CKPT")"

    local last_map=""
    for s in "${stage_seq[@]}"; do
        local map_num
        map_num="$(stage_to_map "$s")"

        if [[ "$map_num" != "$last_map" ]]; then
            stop_ros_env
            start_ros_env "$s"
            last_map="$map_num"
        fi

        run_stage "$s" "$current_ckpt"
        current_ckpt="$LAST_STAGE_CKPT"
        [[ -n "$current_ckpt" ]] && success "stage ${s} checkpoint: $current_ckpt" || warn "stage ${s} no checkpoint parsed"
    done

    stop_ros_env
    success "curriculum finished"
    info "final checkpoint: ${current_ckpt:-N/A}"
    info "logs: ${LOG_DIR}"
}

main
