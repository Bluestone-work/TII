#!/bin/bash

set -euo pipefail

WORKSPACE="/home/wj/work/multi-robot-exploration-rl"
RUN_SCRIPT="$WORKSPACE/run_curriculum.sh"
LAUNCH_LOG_DIR="$WORKSPACE/train_logs"
RVIZ_CONFIG_DIR="$LAUNCH_LOG_DIR/rviz_configs"

MODEL_SET="both"          # both | mlp | gat
START_STAGE=1
END_STAGE=4
ACTION_MODE="continuous"
MLP_USE_COMM_OBS=0
GAT_ACTOR_GRAPH="local_risk"
GAT_CRITIC_MODE="mlp"
NUM_WORKERS=1
TRAIN_STEPS=""
TRAIN_BATCH_SIZE=""
CHECKPOINT_FREQ=""
BASE_DOMAIN_ID=11
BASE_GAZEBO_PORT=11345
RAY_NUM_CPUS_PER_JOB=6
RAY_NUM_GPUS_PER_JOB=0
RAY_OBJECT_STORE_MEMORY_MB=512
CUDA_VISIBLE_DEVICES_OVERRIDE=""
RENDER_MODE="rviz"        # headless | rviz | full_gui
COUNTERFACTUAL_ADVANTAGE_COEF=""
COUNTERFACTUAL_CREDIT_CLIP=""
RESUME_CKPT=""
EXACT_RESUME=0
JOB_TAG=""

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
info()    { echo -e "${CYAN}[INFO]${RESET} $*"; }
success() { echo -e "${GREEN}[✓]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[⚠]${RESET} $*"; }
error()   { echo -e "${RED}[✗]${RESET} $*" >&2; }

usage() {
    cat <<'EOF'
用法:
  ./run_parallel.sh
  ./run_parallel.sh --model_set both --start_stage 1 --end_stage 4
  ./run_parallel.sh --model_set gat --action_mode continuous --train_steps 300000

参数:
  --model_set         both | mlp | gat
  --start_stage       起始课程阶段
  --end_stage         结束课程阶段
  --action_mode       continuous | discrete_primitive
  --mlp_use_comm_obs  0 | 1，MLP 是否也吃通信邻居观测
    --gat_actor_graph   social_risk | local_risk | neighbor
  --gat_critic_mode   mlp | gat
  --num_workers       每组实验的 Ray worker 数
  --train_steps       覆盖每组训练步数
  --train_batch_size  覆盖每组 batch size
  --checkpoint_freq   覆盖每组 checkpoint 保存频率（按迭代）
  --base_domain_id    第一组实验使用的 ROS_DOMAIN_ID
  --base_gazebo_port  第一组实验使用的 Gazebo 端口
  --ray_num_cpus_per_job   每组 Ray 集群 CPU 上限
  --ray_num_gpus_per_job   每组 Ray 集群 GPU 上限
  --ray_object_store_memory_mb  每组 Ray object store 内存上限（MB）
  --cuda_visible_devices   给所有并行实验设置 CUDA_VISIBLE_DEVICES
  --render_mode      headless | rviz | full_gui
  --counterfactual_advantage_coef  混入 advantage 的 counterfactual 系数 λ
  --counterfactual_credit_clip     counterfactual credit 标准化后的裁剪阈值
  --resume            透传给 run_curriculum.sh 的 checkpoint 路径
  --exact_resume      Stage 1 恢复时按“原样续训”处理，不自动降 LR/clip
  --job_tag           追加到实验标签末尾，避免并行实验结果目录冲突
  -h, --help          显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_set) MODEL_SET="$2"; shift 2 ;;
        --start_stage) START_STAGE="$2"; shift 2 ;;
        --end_stage) END_STAGE="$2"; shift 2 ;;
        --action_mode) ACTION_MODE="$2"; shift 2 ;;
        --mlp_use_comm_obs) MLP_USE_COMM_OBS="$2"; shift 2 ;;
        --gat_actor_graph) GAT_ACTOR_GRAPH="$2"; shift 2 ;;
        --gat_critic_mode) GAT_CRITIC_MODE="$2"; shift 2 ;;
        --num_workers) NUM_WORKERS="$2"; shift 2 ;;
        --train_steps) TRAIN_STEPS="$2"; shift 2 ;;
        --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --checkpoint_freq) CHECKPOINT_FREQ="$2"; shift 2 ;;
        --base_domain_id) BASE_DOMAIN_ID="$2"; shift 2 ;;
        --base_gazebo_port) BASE_GAZEBO_PORT="$2"; shift 2 ;;
        --ray_num_cpus_per_job) RAY_NUM_CPUS_PER_JOB="$2"; shift 2 ;;
        --ray_num_gpus_per_job) RAY_NUM_GPUS_PER_JOB="$2"; shift 2 ;;
        --ray_object_store_memory_mb) RAY_OBJECT_STORE_MEMORY_MB="$2"; shift 2 ;;
        --cuda_visible_devices) CUDA_VISIBLE_DEVICES_OVERRIDE="$2"; shift 2 ;;
        --render_mode) RENDER_MODE="$2"; shift 2 ;;
        --counterfactual_advantage_coef) COUNTERFACTUAL_ADVANTAGE_COEF="$2"; shift 2 ;;
        --counterfactual_credit_clip) COUNTERFACTUAL_CREDIT_CLIP="$2"; shift 2 ;;
        --resume) RESUME_CKPT="$2"; shift 2 ;;
        --exact_resume) EXACT_RESUME=1; shift 1 ;;
        --job_tag) JOB_TAG="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) error "未知参数: $1"; usage; exit 1 ;;
    esac
done

if [[ "$GAT_ACTOR_GRAPH" != "social_risk" && "$GAT_ACTOR_GRAPH" != "local_risk" && "$GAT_ACTOR_GRAPH" != "neighbor" ]]; then
    error "--gat_actor_graph 只支持 social_risk | local_risk | neighbor"
    exit 1
fi

[[ -x "$RUN_SCRIPT" ]] || { error "未找到可执行训练脚本: $RUN_SCRIPT"; exit 1; }
mkdir -p "$LAUNCH_LOG_DIR"
mkdir -p "$RVIZ_CONFIG_DIR"

if [[ "$MODEL_SET" == "gat" || "$MODEL_SET" == "both" ]] && (( END_STAGE < 2 )); then
    warn "END_STAGE=$END_STAGE 会让 GAT 停留在课程 Stage 1，通信分支不会真正开启。"
fi

declare -a JOB_PIDS=()
declare -a JOB_LABELS=()
declare -a JOB_LOGS=()
declare -a JOB_GAZEBO_PORTS=()
declare -a JOB_DOMAIN_IDS=()
CLEANUP_RUNNING=0

port_listener_pids() {
    local port="${1:-}"
    [[ -n "$port" ]] || return 0
    lsof -tiTCP:"$port" -sTCP:LISTEN -Pn 2>/dev/null || true
}

assert_port_available() {
    local port="${1:-}"
    local label="${2:-job}"
    [[ -n "$port" ]] || return 0

    local pids=""
    pids="$(port_listener_pids "$port")"
    if [[ -n "$pids" ]]; then
        error "[$label] GAZEBO_PORT=${port} 已被占用。为避免误杀其它实验，run_parallel.sh 不会自动清理该端口。"
        echo "$pids" | while read -r pid; do
            [[ -n "$pid" ]] || continue
            ps -fp "$pid" || true
        done
        exit 1
    fi
}

report_port_state() {
    local port="${1:-}"
    local label="${2:-job}"
    [[ -n "$port" ]] || return 0

    local pids=""
    pids="$(port_listener_pids "$port")"
    if [[ -n "$pids" ]]; then
        warn "[$label] 清理后 GAZEBO_PORT=${port} 仍有监听进程，未自动强杀以避免影响其它实验。"
        echo "$pids" | while read -r pid; do
            [[ -n "$pid" ]] || continue
            ps -fp "$pid" || true
        done
    fi
}

terminate_job() {
    local pid="${1:-}"
    [[ -n "$pid" ]] || return 0
    if kill -0 "$pid" 2>/dev/null; then
        kill -TERM -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

cleanup_parallel() {
    local exit_code="${1:-0}"
    if (( CLEANUP_RUNNING == 1 )); then
        return 0
    fi
    CLEANUP_RUNNING=1

    echo ""
    if [[ "$exit_code" -eq 0 ]]; then
        info "并行实验退出，开始清理残留进程..."
    else
        warn "并行实验异常退出，开始清理残留进程..."
    fi

    local i pid port domain label
    for i in "${!JOB_PIDS[@]}"; do
        pid="${JOB_PIDS[$i]}"
        label="${JOB_LABELS[$i]:-job_$i}"
        port="${JOB_GAZEBO_PORTS[$i]:-}"
        domain="${JOB_DOMAIN_IDS[$i]:-}"
        info "清理 [$label] pid=${pid:-n/a} ROS_DOMAIN_ID=${domain:-n/a} GAZEBO_PORT=${port:-n/a}"
        terminate_job "$pid"
        report_port_state "$port" "$label"
    done
}

on_signal() {
    cleanup_parallel 130
    exit 130
}

on_exit() {
    local exit_code=$?
    cleanup_parallel "$exit_code"
}

trap on_signal SIGINT SIGTERM
trap on_exit EXIT

launch_job() {
    local model="$1"
    local domain_id="$2"
    local gazebo_port="$3"
    local suffix="$4"
    if [[ -n "$JOB_TAG" ]]; then
        suffix="${suffix}_$(echo "$JOB_TAG" | tr ' /:' '___' | sed 's/[^A-Za-z0-9._-]/_/g')"
    fi
    local launch_log="$LAUNCH_LOG_DIR/parallel_${suffix}.log"
    local rviz_config="$RVIZ_CONFIG_DIR/${suffix}.rviz"
    local rviz_node_name="rviz2_${suffix}"
    local rviz_x=80
    local rviz_y=120

    if [[ "$suffix" == gat_* ]]; then
        rviz_x=860
    fi

    local -a cmd=(
        "$RUN_SCRIPT"
        --model_type "$model"
        --start_stage "$START_STAGE"
        --end_stage "$END_STAGE"
        --action_mode "$ACTION_MODE"
        --num_workers "$NUM_WORKERS"
        --ros_domain_id "$domain_id"
        --gazebo_port "$gazebo_port"
        --run_suffix "$suffix"
        --ray_num_cpus "$RAY_NUM_CPUS_PER_JOB"
        --ray_num_gpus "$RAY_NUM_GPUS_PER_JOB"
        --ray_object_store_memory_mb "$RAY_OBJECT_STORE_MEMORY_MB"
    )

    if [[ "$model" == "gat" ]]; then
        cmd+=(--gat_actor_graph "$GAT_ACTOR_GRAPH" --gat_critic_mode "$GAT_CRITIC_MODE")
    elif [[ "$model" == "mlp" ]]; then
        cmd+=(--mlp_use_comm_obs "$MLP_USE_COMM_OBS")
    fi

    if [[ -n "$COUNTERFACTUAL_ADVANTAGE_COEF" ]]; then
        cmd+=(--counterfactual_advantage_coef "$COUNTERFACTUAL_ADVANTAGE_COEF")
    fi
    if [[ -n "$COUNTERFACTUAL_CREDIT_CLIP" ]]; then
        cmd+=(--counterfactual_credit_clip "$COUNTERFACTUAL_CREDIT_CLIP")
    fi

    case "$RENDER_MODE" in
        headless)
            cmd+=(--headless_sim --disable_rviz)
            ;;
        rviz)
            sed \
                -e "s/^  X: .*/  X: ${rviz_x}/" \
                -e "s/^  Y: .*/  Y: ${rviz_y}/" \
                "/home/wj/work/multi-robot-exploration-rl/src/start_rl_environment_tb3/rviz/multi_robot.rviz" \
                > "$rviz_config"
            cmd+=(--headless_sim --enable_rviz --rviz_config "$rviz_config" --rviz_node_name "$rviz_node_name")
            ;;
        full_gui)
            sed \
                -e "s/^  X: .*/  X: ${rviz_x}/" \
                -e "s/^  Y: .*/  Y: ${rviz_y}/" \
                "/home/wj/work/multi-robot-exploration-rl/src/start_rl_environment_tb3/rviz/multi_robot.rviz" \
                > "$rviz_config"
            cmd+=(--enable_rviz --rviz_config "$rviz_config" --rviz_node_name "$rviz_node_name")
            ;;
        *)
            error "--render_mode 只支持 headless | rviz | full_gui"
            exit 1
            ;;
    esac

    if [[ -n "$TRAIN_STEPS" ]]; then
        cmd+=(--train_steps "$TRAIN_STEPS")
    fi
    if [[ -n "$TRAIN_BATCH_SIZE" ]]; then
        cmd+=(--train_batch_size "$TRAIN_BATCH_SIZE")
    fi
    if [[ -n "$CHECKPOINT_FREQ" ]]; then
        cmd+=(--checkpoint_freq "$CHECKPOINT_FREQ")
    fi
    if [[ -n "$RESUME_CKPT" ]]; then
        cmd+=(--resume "$RESUME_CKPT")
    fi
    if (( EXACT_RESUME == 1 )); then
        cmd+=(--exact_resume)
    fi

    info "启动实验 [$suffix]"
    info "  model=$model stage=${START_STAGE}->${END_STAGE} action=$ACTION_MODE"
    if [[ "$model" == "gat" ]]; then
        info "  gat_actor_graph=$GAT_ACTOR_GRAPH gat_critic_mode=$GAT_CRITIC_MODE"
    elif [[ "$model" == "mlp" ]]; then
        info "  mlp_use_comm_obs=$MLP_USE_COMM_OBS"
    fi
    info "  ROS_DOMAIN_ID=$domain_id GAZEBO_PORT=$gazebo_port"
    info "  stdout/stderr -> $launch_log"

    assert_port_available "$gazebo_port" "$suffix"

    CUDA_VISIBLE_DEVICES_OVERRIDE="$CUDA_VISIBLE_DEVICES_OVERRIDE" \
    KILL_ALL_ROS_SCOPE=port_only \
    setsid bash -lc '
        cd "$1"
        shift
        if [[ -n "${CUDA_VISIBLE_DEVICES_OVERRIDE:-}" ]]; then
            export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_OVERRIDE"
        fi
        exec "$@"
    ' bash "$WORKSPACE" "${cmd[@]}" >"$launch_log" 2>&1 &

    JOB_PIDS+=("$!")
    JOB_LABELS+=("$suffix")
    JOB_LOGS+=("$launch_log")
    JOB_GAZEBO_PORTS+=("$gazebo_port")
    JOB_DOMAIN_IDS+=("$domain_id")
}

next_domain="$BASE_DOMAIN_ID"
next_port="$BASE_GAZEBO_PORT"

case "$MODEL_SET" in
    both)
        launch_job "mlp" "$next_domain" "$next_port" "mlp_s${START_STAGE}_${ACTION_MODE}"
        next_domain=$((next_domain + 1))
        next_port=$((next_port + 10))
        launch_job "gat" "$next_domain" "$next_port" "gat_s${START_STAGE}_${ACTION_MODE}"
        ;;
    mlp|gat)
        launch_job "$MODEL_SET" "$next_domain" "$next_port" "${MODEL_SET}_s${START_STAGE}_${ACTION_MODE}"
        ;;
    *)
        error "--model_set 只支持 both | mlp | gat"
        exit 1
        ;;
esac

echo ""
success "已启动 ${#JOB_PIDS[@]} 组实验"

for i in "${!JOB_PIDS[@]}"; do
    echo "  ${JOB_LABELS[$i]}  pid=${JOB_PIDS[$i]}  log=${JOB_LOGS[$i]}"
    echo "  结果目录: $WORKSPACE/ray_results/${JOB_LABELS[$i]}"
    echo "  课程日志: $WORKSPACE/curriculum_logs/${JOB_LABELS[$i]}"
done

echo ""
info "实时查看日志示例:"
for log in "${JOB_LOGS[@]}"; do
    echo "  tail -f $log"
done

echo ""
wait
