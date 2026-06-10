#!/bin/bash

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
BASE_SCRIPT="${WORKSPACE}/run_intent_mappo_train.sh"
OUTPUT_DIR="${WORKSPACE}/ray_results"
BASE_RUN_NAME="intent_msa3c_curriculum_map6"
MODEL_NUM_AGENTS=4
NUM_WORKERS=1
TRAIN_BATCH_SIZE=2000
ROLLOUT_FRAGMENT_LENGTH=100
CHECKPOINT_FREQ=5
COMMUNICATION_RANGE=3.5
INTENT_TOP_K=3
INTENT_HISTORY_STEPS=4
INTENT_DT=0.1
ENV_LOG_LEVEL="INFO"
ENABLE_VISUALIZATION=0
ENABLE_INTENT_VISUALIZATION=0
MSA3C_AVOID_ENTER_FRONT_DIST=1.10
MSA3C_AVOID_EXIT_FRONT_DIST=1.35
MSA3C_AVOID_ENTER_NEIGHBOR_DIST=1.70
MSA3C_AVOID_EXIT_NEIGHBOR_DIST=2.00
MSA3C_AVOID_ENTER_TTC=3.20
MSA3C_AVOID_EXIT_TTC=4.00
MSA3C_AVOID_ENTER_CONFLICT_DIST=2.10
MSA3C_AVOID_EXIT_CONFLICT_DIST=2.70
MSA3C_AVOID_HOLD_STEPS=12
MSA3C_CORRIDOR_HALF_WIDTH=0.34
MSA3C_CORRIDOR_NEIGHBOR_HALF_WIDTH=0.46
MSA3C_CORRIDOR_CHECK_DIST=1.70
MSA3C_EMERGENCY_FRONT_DIST=0.36
MSA3C_EMERGENCY_NEIGHBOR_DIST=0.52
MSA3C_EMERGENCY_TTC=0.90
STAGE6_STEPS=20000
STAGE7_STEPS=20000
STAGE8_STEPS=60000
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
MSA3C curriculum launcher.

Stages:
  6: 2-robot head-on
  7: 2-robot crossing
  8: 4-robot intersection

Usage:
  ./run_msa3c_curriculum_train.sh
  ./run_msa3c_curriculum_train.sh --stage6_steps 10000 --stage7_steps 10000 --stage8_steps 40000
  ./run_msa3c_curriculum_train.sh --enable_visualization --enable_intent_visualization
  ./run_msa3c_curriculum_train.sh -- --lr 1e-4

Options:
  --base_run_name NAME
  --output_dir DIR
  --model_num_agents N
  --num_workers N
  --train_batch_size N
  --rollout_fragment_length N
  --checkpoint_freq N
  --communication_range X
  --intent_top_k N
  --intent_history_steps N
  --intent_dt X
  --stage6_steps N
  --stage7_steps N
  --stage8_steps N
  --env_log_level LEVEL
  --enable_visualization / --disable_visualization
  --enable_intent_visualization / --disable_intent_visualization
  --help

Notes:
  1) The script relaunches the ROS/Gazebo world for each stage to keep robot count consistent.
  2) Extra args after "--" are passed through to train_intent_mappo.py in every stage.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base_run_name) BASE_RUN_NAME="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --model_num_agents) MODEL_NUM_AGENTS="$2"; shift 2 ;;
        --num_workers) NUM_WORKERS="$2"; shift 2 ;;
        --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --rollout_fragment_length) ROLLOUT_FRAGMENT_LENGTH="$2"; shift 2 ;;
        --checkpoint_freq) CHECKPOINT_FREQ="$2"; shift 2 ;;
        --communication_range) COMMUNICATION_RANGE="$2"; shift 2 ;;
        --intent_top_k) INTENT_TOP_K="$2"; shift 2 ;;
        --intent_history_steps) INTENT_HISTORY_STEPS="$2"; shift 2 ;;
        --intent_dt) INTENT_DT="$2"; shift 2 ;;
        --stage6_steps) STAGE6_STEPS="$2"; shift 2 ;;
        --stage7_steps) STAGE7_STEPS="$2"; shift 2 ;;
        --stage8_steps) STAGE8_STEPS="$2"; shift 2 ;;
        --env_log_level) ENV_LOG_LEVEL="$2"; shift 2 ;;
        --enable_visualization) ENABLE_VISUALIZATION=1; shift 1 ;;
        --disable_visualization) ENABLE_VISUALIZATION=0; shift 1 ;;
        --enable_intent_visualization) ENABLE_INTENT_VISUALIZATION=1; shift 1 ;;
        --disable_intent_visualization) ENABLE_INTENT_VISUALIZATION=0; shift 1 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) error "Unknown option: $1"; usage; exit 2 ;;
    esac
done

[[ -x "${BASE_SCRIPT}" ]] || { error "Missing launcher: ${BASE_SCRIPT}"; exit 1; }
mkdir -p "${OUTPUT_DIR}"

find_latest_checkpoint() {
    local run_dir="$1"
    if [[ ! -d "${run_dir}" ]]; then
        return 1
    fi
    local ckpt
    ckpt=$(find "${run_dir}" -type f -name algorithm_state.pkl -printf '%T@ %h\n' | sort -nr | head -n1 | cut -d' ' -f2-)
    [[ -n "${ckpt}" ]] || return 1
    printf '%s\n' "${ckpt}"
}

run_stage() {
    local stage="$1"
    local num_agents="$2"
    local steps="$3"
    local phase_name="$4"
    local restore_ckpt="${5:-}"

    local stage_run_name="${BASE_RUN_NAME}_${phase_name}"
    local run_dir="${OUTPUT_DIR}/${stage_run_name}_stage${stage}_n${num_agents}"

    banner "Curriculum stage ${stage} :: ${phase_name}"
    info "num_agents=${num_agents} steps=${steps} model_slots=${MODEL_NUM_AGENTS}"
    if [[ -n "${restore_ckpt}" ]]; then
        info "restore=${restore_ckpt}"
    fi

    local cmd=(
        bash "${BASE_SCRIPT}"
        --env_stage "${stage}"
        --map_number 6
        --num_agents "${num_agents}"
        --num_workers "${NUM_WORKERS}"
        --train_steps "${steps}"
        --checkpoint_freq "${CHECKPOINT_FREQ}"
        --train_batch_size "${TRAIN_BATCH_SIZE}"
        --rollout_fragment_length "${ROLLOUT_FRAGMENT_LENGTH}"
        --run_name "${stage_run_name}"
        --output_dir "${OUTPUT_DIR}"
        --communication_range "${COMMUNICATION_RANGE}"
        --intent_top_k "${INTENT_TOP_K}"
        --intent_history_steps "${INTENT_HISTORY_STEPS}"
        --intent_dt "${INTENT_DT}"
        --disable_yielding
        --disable_yield_action
        --disable_yield_reward
        --disable_safety_shield
        --no-base_shield_enable
        --no-base_tracking_assist_enable
        --no-local_executor_enable
        --no-base_zone_manager_enable
        --no-base_hybrid_control_enable
        # --disable_visualization
        # --disable_intent_visualization
        --env_log_level "${ENV_LOG_LEVEL}"
        --
        --msa3c_action_mode
        --msa3c_forward_only
        --msa3c_mode_switch_enable
        --msa3c_avoid_enter_front_dist "${MSA3C_AVOID_ENTER_FRONT_DIST}"
        --msa3c_avoid_exit_front_dist "${MSA3C_AVOID_EXIT_FRONT_DIST}"
        --msa3c_avoid_enter_neighbor_dist "${MSA3C_AVOID_ENTER_NEIGHBOR_DIST}"
        --msa3c_avoid_exit_neighbor_dist "${MSA3C_AVOID_EXIT_NEIGHBOR_DIST}"
        --msa3c_avoid_enter_ttc "${MSA3C_AVOID_ENTER_TTC}"
        --msa3c_avoid_exit_ttc "${MSA3C_AVOID_EXIT_TTC}"
        --msa3c_avoid_enter_conflict_dist "${MSA3C_AVOID_ENTER_CONFLICT_DIST}"
        --msa3c_avoid_exit_conflict_dist "${MSA3C_AVOID_EXIT_CONFLICT_DIST}"
        --msa3c_avoid_hold_steps "${MSA3C_AVOID_HOLD_STEPS}"
        --msa3c_corridor_half_width "${MSA3C_CORRIDOR_HALF_WIDTH}"
        --msa3c_corridor_neighbor_half_width "${MSA3C_CORRIDOR_NEIGHBOR_HALF_WIDTH}"
        --msa3c_corridor_check_dist "${MSA3C_CORRIDOR_CHECK_DIST}"
        --msa3c_emergency_front_dist "${MSA3C_EMERGENCY_FRONT_DIST}"
        --msa3c_emergency_neighbor_dist "${MSA3C_EMERGENCY_NEIGHBOR_DIST}"
        --msa3c_emergency_ttc "${MSA3C_EMERGENCY_TTC}"
        --model_num_agents "${MODEL_NUM_AGENTS}"
    )

    if (( ENABLE_VISUALIZATION == 1 )); then
        cmd+=(--enable_visualization)
    fi
    if (( ENABLE_INTENT_VISUALIZATION == 1 )); then
        cmd+=(--enable_intent_visualization)
    fi
    if [[ -n "${restore_ckpt}" ]]; then
        cmd+=(--restore_checkpoint "${restore_ckpt}")
    fi
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi

    info "command: ${cmd[*]}"
    "${cmd[@]}"

    local ckpt
    ckpt=$(find_latest_checkpoint "${run_dir}") || {
        error "No checkpoint found in ${run_dir}"
        exit 1
    }
    success "stage ${stage} checkpoint: ${ckpt}"
    STAGE_CHECKPOINT="${ckpt}"
}

main() {
    local stage_ckpt=""
    run_stage 6 2 "${STAGE6_STEPS}" "headon" ""
    stage_ckpt="${STAGE_CHECKPOINT}"
    run_stage 7 2 "${STAGE7_STEPS}" "cross" "${stage_ckpt}"
    stage_ckpt="${STAGE_CHECKPOINT}"
    run_stage 8 4 "${STAGE8_STEPS}" "intersection" "${stage_ckpt}"

    banner "Curriculum training finished"
    info "final checkpoint: ${STAGE_CHECKPOINT}"
}

main
