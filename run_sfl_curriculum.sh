#!/bin/bash
# Curriculum launcher for SFL IPPO navigation training.

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
TRAIN_SCRIPT="${WORKSPACE}/src/sfl_nav_training/sfl_nav_training/train_sfl_ippo.py"
OUTPUT_DIR="${WORKSPACE}/ray_results"
LOG_DIR="${WORKSPACE}/curriculum_logs_sfl"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

# Curriculum schedule
STAGES="1,6,7"
CYCLES=1
TRAIN_STEPS=200000
RUN_NAME="sfl_ippo_curriculum"
RESUME_CKPT=""

# RL and env args
NUM_AGENTS=2
NUM_WORKERS=1
CHECKPOINT_FREQ=20
TRAIN_BATCH_SIZE=4096
ROLLOUT_FRAGMENT_LENGTH=256
SAMPLE_TIMEOUT_S=1200
LR="2.5e-4"
ANNEAL_LR=1
FC_DIM=512
HIDDEN_SIZE=512
MAX_SEQ_LEN=64
USE_LAYER_NORM=0
COMMUNICATION_RANGE=3.5
COMM_MODE="centralized_oracle"
END_EP_ON_COLLISION=1

# Observation and reward args
LIDAR_NUM_BEAMS=200
LIDAR_MAX_RANGE=6.0
LIDAR_MIN_RANGE=0.0
REW_LAMBDA=0.5
GOAL_REW=4.0
DT_REW=-0.01
COLL_REW=-4.0
LIDAR_THRESH=0.1
LIDAR_REW=-0.1
AGENT_COLLISION_DIST=0.6

# Visualization / launch control
ENABLE_VIS=1
TRACKING_VIZ_INTERVAL=2
ENV_LOG_LEVEL="INFO"
LAUNCH_ENV=1
GAZEBO_WAIT_SEC=60
GAZEBO_GRACE_SEC=5

EXTRA_ARGS=()
LAST_CKPT=""
declare -a PHASE_SUMMARY=()

usage() {
cat <<'USAGE'
Usage:
  ./run_sfl_curriculum.sh
  ./run_sfl_curriculum.sh --stages 1,6,7 --cycles 2 --steps 120000
  ./run_sfl_curriculum.sh --resume /abs/path/checkpoint_000123
  ./run_sfl_curriculum.sh --disable_visualization --skip_launch_env
  ./run_sfl_curriculum.sh -- --grad_clip 0.7

Options:
  --stages LIST                 Curriculum stage list, comma-separated (supports: 1,6,7)
  --cycles N                    Number of times to repeat the stage list
  --steps N                     Train steps per stage
  --run_name NAME               Run name prefix
  --output_dir DIR              Checkpoint output root
  --resume PATH                 Initial restore checkpoint

  --num_agents N
  --num_workers N
  --checkpoint_freq N
  --train_batch_size N
  --rollout_fragment_length N
  --sample_timeout_s N
  --lr X
  --anneal_lr / --no-anneal_lr
  --fc_dim N
  --hidden_size N
  --max_seq_len N
  --use_layer_norm / --no-use_layer_norm
  --communication_range X
  --comm_mode MODE
  --end_episode_on_collision_event / --no-end_episode_on_collision_event

  --lidar_num_beams N
  --lidar_max_range X
  --lidar_min_range X
  --rew_lambda X
  --goal_rew X
  --dt_rew X
  --coll_rew X
  --lidar_thresh X
  --lidar_rew X
  --agent_collision_dist X

  --enable_visualization / --disable_visualization
  --tracking_viz_interval N
  --env_log_level LEVEL
  --skip_launch_env

  -h, --help

Notes:
  1) For stage 6/7, train_sfl_ippo.py requires --num_agents 2.
  2) Any args after "--" are passed through to train_sfl_ippo.py for every phase.
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

        --num_agents) NUM_AGENTS="$2"; shift 2 ;;
        --num_workers) NUM_WORKERS="$2"; shift 2 ;;
        --checkpoint_freq) CHECKPOINT_FREQ="$2"; shift 2 ;;
        --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --rollout_fragment_length) ROLLOUT_FRAGMENT_LENGTH="$2"; shift 2 ;;
        --sample_timeout_s) SAMPLE_TIMEOUT_S="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --anneal_lr) ANNEAL_LR=1; shift ;;
        --no-anneal_lr) ANNEAL_LR=0; shift ;;
        --fc_dim) FC_DIM="$2"; shift 2 ;;
        --hidden_size) HIDDEN_SIZE="$2"; shift 2 ;;
        --max_seq_len) MAX_SEQ_LEN="$2"; shift 2 ;;
        --use_layer_norm) USE_LAYER_NORM=1; shift ;;
        --no-use_layer_norm) USE_LAYER_NORM=0; shift ;;
        --communication_range) COMMUNICATION_RANGE="$2"; shift 2 ;;
        --comm_mode) COMM_MODE="$2"; shift 2 ;;
        --end_episode_on_collision_event) END_EP_ON_COLLISION=1; shift ;;
        --no-end_episode_on_collision_event) END_EP_ON_COLLISION=0; shift ;;

        --lidar_num_beams) LIDAR_NUM_BEAMS="$2"; shift 2 ;;
        --lidar_max_range) LIDAR_MAX_RANGE="$2"; shift 2 ;;
        --lidar_min_range) LIDAR_MIN_RANGE="$2"; shift 2 ;;
        --rew_lambda) REW_LAMBDA="$2"; shift 2 ;;
        --goal_rew) GOAL_REW="$2"; shift 2 ;;
        --dt_rew) DT_REW="$2"; shift 2 ;;
        --coll_rew) COLL_REW="$2"; shift 2 ;;
        --lidar_thresh) LIDAR_THRESH="$2"; shift 2 ;;
        --lidar_rew) LIDAR_REW="$2"; shift 2 ;;
        --agent_collision_dist) AGENT_COLLISION_DIST="$2"; shift 2 ;;

        --enable_visualization) ENABLE_VIS=1; shift ;;
        --disable_visualization) ENABLE_VIS=0; shift ;;
        --tracking_viz_interval) TRACKING_VIZ_INTERVAL="$2"; shift 2 ;;
        --env_log_level) ENV_LOG_LEVEL="$2"; shift 2 ;;
        --skip_launch_env) LAUNCH_ENV=0; shift ;;

        --) shift; EXTRA_ARGS+=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) error "Unknown option: $1"; usage; exit 2 ;;
    esac
done

setup_env() {
    set +u
    if [[ -f "${CONDA_SETUP}" ]]; then
        # shellcheck disable=SC1090
        source "${CONDA_SETUP}"
        conda activate "${CONDA_ENV}" || warn "conda activate failed: ${CONDA_ENV}"
    else
        warn "conda setup not found: ${CONDA_SETUP}"
    fi

    [[ -f "${ROS_SETUP}" ]] || { set -u; error "missing ROS setup: ${ROS_SETUP}"; exit 1; }
    # shellcheck disable=SC1090
    source "${ROS_SETUP}"

    if [[ -f "${WS_SETUP}" ]]; then
        # shellcheck disable=SC1090
        source "${WS_SETUP}"
    else
        warn "workspace setup not found: ${WS_SETUP}"
    fi
    set -u
}

check_env() {
    [[ -f "${TRAIN_SCRIPT}" ]] || { error "missing train script: ${TRAIN_SCRIPT}"; exit 1; }
    [[ -f "${KILL_SCRIPT}" ]] || { error "missing kill script: ${KILL_SCRIPT}"; exit 1; }
    command -v python3 >/dev/null 2>&1 || { error "python3 not found"; exit 1; }
    command -v ros2 >/dev/null 2>&1 || { error "ros2 not found"; exit 1; }
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

parse_stage_sequence() {
    local list="$1"
    local cycles="$2"
    local raw=()
    local valid=()
    local s

    IFS=',' read -r -a raw <<< "$list"
    for s in "${raw[@]}"; do
        s="${s//[[:space:]]/}"
        [[ -z "$s" ]] && continue
        case "$s" in
            1|6|7) valid+=("$s") ;;
            *) error "unsupported stage: $s (only 1,6,7)"; exit 1 ;;
        esac
    done

    [[ ${#valid[@]} -gt 0 ]] || { error "stage list is empty"; exit 1; }

    local c
    for ((c=0; c<cycles; c++)); do
        for s in "${valid[@]}"; do
            printf '%s\n' "$s"
        done
    done
}

stage_to_map() {
    local stage="$1"
    case "$stage" in
        1) echo "3" ;;
        6|7) echo "6" ;;
        *) return 1 ;;
    esac
}

stop_ros_env() {
    (( LAUNCH_ENV == 0 )) && return 0
    bash "${KILL_SCRIPT}" >/dev/null 2>&1 || true
    sleep 2
}

start_ros_env() {
    local stage="$1"
    local phase_idx="$2"
    (( LAUNCH_ENV == 0 )) && return 0

    local map_num
    map_num="$(stage_to_map "$stage")" || { error "invalid stage: $stage"; exit 1; }

    local ros_log="${LOG_DIR}/phase$(printf '%02d' "$phase_idx")_stage${stage}_ros.log"
    info "launch env for phase=${phase_idx}, stage=${stage}, map=${map_num}, agents=${NUM_AGENTS}"

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
        gnome-terminal --title="[SFL phase ${phase_idx} stage ${stage}]" -- bash -c "$inner" &
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
        if [[ -f "$ros_log" ]] && [[ $(grep -c "Successfully spawned entity" "$ros_log" 2>/dev/null || true) -ge "$NUM_AGENTS" ]]; then
            success "gazebo ready in ${waited}s (spawn log check)"
            sleep "$GAZEBO_GRACE_SEC"
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
    done

    warn "gazebo wait timeout (${GAZEBO_WAIT_SEC}s), continue"
}

find_latest_ckpt_in_run() {
    local run_dir="$1"
    [[ -d "$run_dir" ]] || return 1
    find "$run_dir" -maxdepth 4 -type d -name "checkpoint_*" 2>/dev/null | sort -V | tail -1
}

run_stage() {
    local stage="$1"
    local phase_idx="$2"
    local restore_ckpt="${3:-}"

    local phase_name
    phase_name="phase$(printf '%02d' "$phase_idx")"
    local phase_run_name="${RUN_NAME}_${phase_name}"
    local log_file="${LOG_DIR}/${phase_name}_stage${stage}_train.log"

    local cmd=(
        python3 -u "${TRAIN_SCRIPT}"
        --env_stage "$stage"
        --num_agents "$NUM_AGENTS"
        --num_workers "$NUM_WORKERS"
        --train_steps "$TRAIN_STEPS"
        --checkpoint_freq "$CHECKPOINT_FREQ"
        --lr "$LR"
        --train_batch_size "$TRAIN_BATCH_SIZE"
        --rollout_fragment_length "$ROLLOUT_FRAGMENT_LENGTH"
        --sample_timeout_s "$SAMPLE_TIMEOUT_S"
        --fc_dim "$FC_DIM"
        --hidden_size "$HIDDEN_SIZE"
        --max_seq_len "$MAX_SEQ_LEN"
        --communication_range "$COMMUNICATION_RANGE"
        --comm_mode "$COMM_MODE"
        --tracking_viz_interval "$TRACKING_VIZ_INTERVAL"
        --env_log_level "$ENV_LOG_LEVEL"
        --lidar_num_beams "$LIDAR_NUM_BEAMS"
        --lidar_max_range "$LIDAR_MAX_RANGE"
        --lidar_min_range "$LIDAR_MIN_RANGE"
        --rew_lambda "$REW_LAMBDA"
        --goal_rew "$GOAL_REW"
        --dt_rew "$DT_REW"
        --coll_rew "$COLL_REW"
        --lidar_thresh "$LIDAR_THRESH"
        --lidar_rew "$LIDAR_REW"
        --agent_collision_dist "$AGENT_COLLISION_DIST"
        --run_name "$phase_run_name"
        --output_dir "$OUTPUT_DIR"
    )

    if (( ANNEAL_LR == 1 )); then
        cmd+=(--anneal_lr)
    else
        cmd+=(--no-anneal_lr)
    fi

    if (( USE_LAYER_NORM == 1 )); then
        cmd+=(--use_layer_norm)
    else
        cmd+=(--no-use_layer_norm)
    fi

    if (( END_EP_ON_COLLISION == 1 )); then
        cmd+=(--end_episode_on_collision_event)
    else
        cmd+=(--no-end_episode_on_collision_event)
    fi

    if (( ENABLE_VIS == 1 )); then
        cmd+=(--enable_visualization)
    else
        cmd+=(--no-enable_visualization)
    fi

    if [[ -n "$restore_ckpt" ]]; then
        cmd+=(--restore_checkpoint "$restore_ckpt")
    fi

    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi

    info "phase=${phase_idx} stage=${stage}"
    info "command: ${cmd[*]}"
    info "log: ${log_file}"

    set +e
    "${cmd[@]}" 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}
    set -e

    if [[ $exit_code -ne 0 ]]; then
        error "phase=${phase_idx} stage=${stage} failed, code=${exit_code}"
        return $exit_code
    fi

    local ckpt_path=""
    ckpt_path=$(sed -n 's/^final_checkpoint:[[:space:]]*//p' "$log_file" | tail -1)
    ckpt_path=$(normalize_ckpt_path "$ckpt_path")

    if [[ -z "$ckpt_path" ]]; then
        local run_dir="${OUTPUT_DIR}/${phase_run_name}_stage${stage}_n${NUM_AGENTS}"
        ckpt_path=$(find_latest_ckpt_in_run "$run_dir" || true)
    fi

    LAST_CKPT="$ckpt_path"

    if [[ -n "$LAST_CKPT" ]]; then
        success "phase=${phase_idx} stage=${stage} checkpoint=${LAST_CKPT}"
    else
        warn "phase=${phase_idx} stage=${stage} checkpoint not found"
    fi

    PHASE_SUMMARY+=("phase ${phase_idx} stage ${stage}: ${LAST_CKPT:-<none>}" )
}

cleanup() {
    warn "received interrupt, cleaning ROS/Gazebo"
    stop_ros_env
    exit 130
}
trap cleanup SIGINT SIGTERM

main() {
    check_env
    setup_env

    RESUME_CKPT="$(normalize_ckpt_path "$RESUME_CKPT")"
    if [[ -n "$RESUME_CKPT" && ! -e "$RESUME_CKPT" ]]; then
        error "resume checkpoint does not exist: $RESUME_CKPT"
        exit 1
    fi

    mapfile -t STAGE_SEQ < <(parse_stage_sequence "$STAGES" "$CYCLES")
    [[ ${#STAGE_SEQ[@]} -gt 0 ]] || { error "parsed stage sequence is empty"; exit 1; }

    local s
    for s in "${STAGE_SEQ[@]}"; do
        if [[ "$s" =~ ^(6|7)$ ]] && [[ "$NUM_AGENTS" != "2" ]]; then
            error "stage ${s} requires --num_agents 2 for train_sfl_ippo.py"
            exit 1
        fi
    done

    info "stages: ${STAGE_SEQ[*]}"
    info "train_steps_per_stage: ${TRAIN_STEPS}"
    info "num_agents: ${NUM_AGENTS}, num_workers: ${NUM_WORKERS}"
    info "run_name: ${RUN_NAME}"
    info "output_dir: ${OUTPUT_DIR}"
    [[ -n "$RESUME_CKPT" ]] && info "initial restore checkpoint: ${RESUME_CKPT}"

    local current_ckpt="$RESUME_CKPT"
    local total=${#STAGE_SEQ[@]}
    local i
    for ((i=0; i<total; i++)); do
        local stage="${STAGE_SEQ[$i]}"
        local phase_idx=$((i + 1))

        stop_ros_env
        start_ros_env "$stage" "$phase_idx"
        LAST_CKPT=""
        run_stage "$stage" "$phase_idx" "$current_ckpt"
        current_ckpt="$LAST_CKPT"

        if (( i < total - 1 )); then
            info "sleep 5s before next phase"
            sleep 5
        fi
    done

    stop_ros_env

    success "SFL curriculum training finished"
    local summary
    for summary in "${PHASE_SUMMARY[@]}"; do
        info "$summary"
    done
    [[ -n "$current_ckpt" ]] && success "final checkpoint: ${current_ckpt}"
    info "logs: ${LOG_DIR}"
}

main "$@"
