#!/bin/bash
# =============================================================================
# 课程学习一键启动脚本
# 自动按 Stage 1→4 依次：重启环境 → 训练 → 保存 checkpoint → 进入下一阶段
#
# 用法：
#   ./run_curriculum.sh                         # 从 Stage 1 开始白板训练
#   ./run_curriculum.sh --start_stage 1 --resume <ckpt_path> # 【新增】Stage 1 导入权重微调
#   ./run_curriculum.sh --start_stage 2         # 从 Stage 2 恢复（需指定 checkpoint）
#   ./run_curriculum.sh --model_type gat        # 使用 GAT 模型
#   ./run_curriculum.sh --action_mode discrete_primitive --ppo_profile auto
# =============================================================================

if [ -z "${BASH_VERSION:-}" ]; then
    exec /bin/bash "$0" "$@"
fi

set -euo pipefail

# ─── 颜色输出 ────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
info()    { echo -e "${CYAN}[INFO]${RESET} $*"; }
success() { echo -e "${GREEN}[✓]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[⚠]${RESET} $*"; }
error()   { echo -e "${RED}[✗]${RESET} $*" >&2; }
banner()  { echo -e "\n${BOLD}${CYAN}$*${RESET}\n"; }

# ─── 可配置参数（可通过命令行覆盖）──────────────────────────────────────────
MODEL_TYPE="gat"          # gat | mlp
MLP_USE_COMM_OBS=0        # 0 | 1
GAT_ACTOR_GRAPH="neighbor"    # local_risk | neighbor
GAT_CRITIC_MODE="mlp"     # mlp | gat
NUM_AGENTS=4              # 机器人数量
NUM_WORKERS=1             # Ray 并行 Worker 数
TRAIN_STEPS=1000000        # 每个阶段训练步数
TRAIN_BATCH_SIZE=5000       # PPO 训练批大小
CHECKPOINT_FREQ=2000        # 每隔多少次迭代保存一次 checkpoint
HIDDEN_DIM=512            # 隐藏层维度
SAMPLE_TIMEOUT_S=1200     # RLlib env runner 采样超时
ROLLOUT_FRAGMENT_LENGTH=1000  
BATCH_MODE="truncate_episodes" 
ENABLE_VISUALIZATION=1       
TRACKING_VIZ_INTERVAL=4      
START_STAGE=2  
END_STAGE=2             
RESUME_CKPT=""            
EXACT_RESUME=0
ENABLE_CONTROL_FUSION=0   
ACTION_SMOOTH_ALPHA=0.2   
SAFETY_FILTER_ENABLE=0    
ROLLING_LOOKAHEAD_DIST=0.8  
OBSTACLE_FILTER_RANGE=1.2   
OBSTACLE_FILTER_FOV_DEG=360 
OBSTACLE_TOP_K=0
ACTION_MODE="continuous"
GRAPH_ABLATION="dual_graph"
PPO_PROFILE="auto"               
COUNTERFACTUAL_ADVANTAGE_COEF="0.15"
COUNTERFACTUAL_CREDIT_CLIP="2.5"
HIGH_CONFLICT_PROB="0.85"
OBS_SPEED_SCALE_OVERRIDE=""
PREDICTIVE_SOCIAL_PENALTY_SCALE=""
PREDICTIVE_FRONT_PENALTY_SCALE=""
SOCIAL_PROXIMITY_RISK_SCALE=""
SUBGOAL_DETOUR_LATERAL_GAIN=""
SUBGOAL_PROGRESS_REWARD_SCALE=""
DETOUR_PROGRESS_RELAX=""
RISK_AWARE_FORWARD_PENALTY_SCALE=""
SAFE_TURN_REWARD_SCALE=""
HEAD_ON_AVOIDANCE_REWARD_SCALE=""
TEAM_REWARD_LAMBDA=""
ROS_DOMAIN_ID_OVERRIDE=""
GAZEBO_PORT="11345"
RUN_SUFFIX=""
MANAGE_ROS_DAEMON=0
RAY_NUM_CPUS=0
RAY_NUM_GPUS=-1
RAY_OBJECT_STORE_MEMORY_MB=0
PARALLEL_SAFE=0
HEADLESS_SIM=0
ENABLE_RVIZ=0
RVIZ_CONFIG_OVERRIDE=""
RVIZ_NODE_NAME="rviz2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$SCRIPT_DIR"
WORKSPACE="/home/wj/work/multi-robot-exploration-rl"
GAZEBO_WAIT_SEC=120
GAZEBO_GRACE_SEC=5
CONDA_SH="${CONDA_SH:-/home/wj/anaconda3/etc/profile.d/conda.sh}"
ROS2_CONDA_ENV="${ROS2_CONDA_ENV:-ros2}"
REQUIRED_PY_VER="3.10"

ensure_python_abi_compatible() {
    local py_ver py_bin
    py_ver="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo unknown)"
    py_bin="$(command -v python3 2>/dev/null || echo unknown)"
    if [[ "$py_ver" != "$REQUIRED_PY_VER" ]]; then
        error "检测到 python3=${py_ver} (${py_bin})，ROS Humble 需要 Python ${REQUIRED_PY_VER}"
        exit 1
    fi
}

bootstrap_python_env() {
    if [[ -f "$CONDA_SH" ]]; then
        set +u
        source "$CONDA_SH" || true
        if command -v conda &>/dev/null; then
            conda activate "$ROS2_CONDA_ENV" >/dev/null 2>&1 || true
        fi
        set -u
    fi
    ensure_python_abi_compatible
}

ROS_SETUP="/opt/ros/humble/setup.bash"
WS_SETUP="${WORKSPACE}/install/setup.bash"
bootstrap_python_env
export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"
export PYTHONPATH="${CODEBASE_ROOT}:${PYTHONPATH:-}"
set +u
[[ -f "$ROS_SETUP" ]] && source "$ROS_SETUP"  || { set -u; error "ROS2 Humble 未找到"; exit 1; }
[[ -f "$WS_SETUP"  ]] && source "$WS_SETUP"   
set -u

sanitize_ament_prefix_path() {
    local original="${AMENT_PREFIX_PATH:-}"
    [[ -z "$original" ]] && return 0

    local filtered=()
    local dropped=()
    local prefix expected_pkg marker

    IFS=':' read -r -a _ament_prefixes <<< "$original"
    for prefix in "${_ament_prefixes[@]}"; do
        [[ -z "$prefix" ]] && continue

        if [[ "$prefix" == "/opt/ros/"* ]]; then
            filtered+=("$prefix")
            continue
        fi

        expected_pkg="$(basename "$prefix")"
        marker="$prefix/share/ament_index/resource_index/packages/$expected_pkg"
        if [[ -f "$marker" ]] && readlink -f "$marker" >/dev/null 2>&1; then
            filtered+=("$prefix")
        else
            dropped+=("$prefix")
        fi
    done

    if ((${#filtered[@]} > 0)); then
        export AMENT_PREFIX_PATH
        AMENT_PREFIX_PATH="$(IFS=:; echo "${filtered[*]}")"
    fi

    if ((${#dropped[@]} > 0)); then
        for prefix in "${dropped[@]}"; do
            warn "检测到失效 AMENT prefix，已跳过: $prefix"
        done
    fi
}

sanitize_ament_prefix_path

normalize_gat_actor_graph() {
    case "${GAT_ACTOR_GRAPH}" in
        social_risk)
            warn "检测到旧参数 gat_actor_graph=social_risk，自动映射为 local_risk"
            GAT_ACTOR_GRAPH="local_risk"
            ;;
        local_risk|neighbor)
            ;;
        *)
            error "无效的 --gat_actor_graph: ${GAT_ACTOR_GRAPH}（允许: local_risk | neighbor）"
            exit 1
            ;;
    esac
}

require_gui_display() {
    (( HEADLESS_SIM == 1 )) && return 0

    local display="${DISPLAY:-}"
    local xauthority="${XAUTHORITY:-}"
    [[ -n "$display" ]] || {
        error "GUI 模式需要可用的 DISPLAY，但当前未设置。"
        error "请在本机图形桌面终端运行，或先导出正确的 DISPLAY/XAUTHORITY。"
        exit 1
    }

    if command -v xdpyinfo >/dev/null 2>&1; then
        if ! xdpyinfo -display "$display" >/dev/null 2>&1; then
            error "GUI 模式无法连接显示服务器 DISPLAY=${display}"
            [[ -n "$xauthority" ]] && error "当前 XAUTHORITY=${xauthority}"
            error "请先确保 'xdpyinfo' 能成功连接当前桌面会话，再启动 Gazebo/RViz。"
            exit 1
        fi
        return 0
    fi

    if command -v xset >/dev/null 2>&1; then
        if ! xset q >/dev/null 2>&1; then
            error "GUI 模式无法访问显示服务器 DISPLAY=${display}"
            [[ -n "$xauthority" ]] && error "当前 XAUTHORITY=${xauthority}"
            error "请先修复图形显示权限，再启动 Gazebo/RViz。"
            exit 1
        fi
        return 0
    fi

    warn "未找到 xdpyinfo/xset，无法预检 DISPLAY；将继续尝试启动 GUI。"
}

build_gui_display_check_cmd() {
    cat <<'EOF'
if [[ -z "${DISPLAY:-}" ]]; then
    echo "[FATAL] GUI 模式需要 DISPLAY，但当前未设置。" >&2
    exit 86
fi
if command -v xdpyinfo >/dev/null 2>&1; then
    xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1 || {
        echo "[FATAL] 无法连接显示服务器 DISPLAY=${DISPLAY}" >&2
        [[ -n "${XAUTHORITY:-}" ]] && echo "[FATAL] XAUTHORITY=${XAUTHORITY}" >&2
        exit 86
    }
elif command -v xset >/dev/null 2>&1; then
    xset q >/dev/null 2>&1 || {
        echo "[FATAL] 无法访问显示服务器 DISPLAY=${DISPLAY}" >&2
        [[ -n "${XAUTHORITY:-}" ]] && echo "[FATAL] XAUTHORITY=${XAUTHORITY}" >&2
        exit 86
    }
else
    python3 - <<'PY'
import os, sys
try:
    import tkinter as tk
except Exception as exc:
    print(f"[FATAL] GUI 预检失败：既没有 xdpyinfo/xset，也无法导入 tkinter: {exc}", file=sys.stderr)
    sys.exit(86)
try:
    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()
    root.destroy()
except Exception as exc:
    print(f"[FATAL] 无法连接显示服务器 DISPLAY={os.environ.get('DISPLAY','')}: {exc}", file=sys.stderr)
    xa = os.environ.get("XAUTHORITY", "")
    if xa:
        print(f"[FATAL] XAUTHORITY={xa}", file=sys.stderr)
    sys.exit(86)
PY
fi
EOF
}

# ─── 解析命令行参数 ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_type)   MODEL_TYPE="$2";   shift 2 ;;
        --mlp_use_comm_obs) MLP_USE_COMM_OBS="$2"; shift 2 ;;
        --gat_actor_graph) GAT_ACTOR_GRAPH="$2"; shift 2 ;;
        --gat_critic_mode) GAT_CRITIC_MODE="$2"; shift 2 ;;
        --num_agents)   NUM_AGENTS="$2";   shift 2 ;;
        --num_workers)  NUM_WORKERS="$2";  shift 2 ;;
        --train_steps)  TRAIN_STEPS="$2";  shift 2 ;;
        --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --checkpoint_freq) CHECKPOINT_FREQ="$2"; shift 2 ;;
        --sample_timeout_s) SAMPLE_TIMEOUT_S="$2"; shift 2 ;;
        --rollout_fragment_length) ROLLOUT_FRAGMENT_LENGTH="$2"; shift 2 ;;
        --batch_mode)   BATCH_MODE="$2";   shift 2 ;;
        --enable_visualization) ENABLE_VISUALIZATION=1; shift 1 ;;
        --disable_visualization) ENABLE_VISUALIZATION=0; shift 1 ;;
        --tracking_viz_interval) TRACKING_VIZ_INTERVAL="$2"; shift 2 ;;
        --rolling_lookahead_dist) ROLLING_LOOKAHEAD_DIST="$2"; shift 2 ;;
        --obstacle_filter_range) OBSTACLE_FILTER_RANGE="$2"; shift 2 ;;
        --obstacle_filter_fov_deg) OBSTACLE_FILTER_FOV_DEG="$2"; shift 2 ;;
        --obstacle_top_k) OBSTACLE_TOP_K="$2"; shift 2 ;;
        --action_mode) ACTION_MODE="$2"; shift 2 ;;
        --graph_ablation) GRAPH_ABLATION="$2"; shift 2 ;;
        --ppo_profile) PPO_PROFILE="$2"; shift 2 ;;
        --counterfactual_advantage_coef) COUNTERFACTUAL_ADVANTAGE_COEF="$2"; shift 2 ;;
        --counterfactual_credit_clip) COUNTERFACTUAL_CREDIT_CLIP="$2"; shift 2 ;;
        --high_conflict_prob) HIGH_CONFLICT_PROB="$2"; shift 2 ;;
        --obs_speed_scale_override) OBS_SPEED_SCALE_OVERRIDE="$2"; shift 2 ;;
        --predictive_social_penalty_scale) PREDICTIVE_SOCIAL_PENALTY_SCALE="$2"; shift 2 ;;
        --predictive_front_penalty_scale) PREDICTIVE_FRONT_PENALTY_SCALE="$2"; shift 2 ;;
        --social_proximity_risk_scale) SOCIAL_PROXIMITY_RISK_SCALE="$2"; shift 2 ;;
        --subgoal_detour_lateral_gain) SUBGOAL_DETOUR_LATERAL_GAIN="$2"; shift 2 ;;
        --subgoal_progress_reward_scale) SUBGOAL_PROGRESS_REWARD_SCALE="$2"; shift 2 ;;
        --detour_progress_relax) DETOUR_PROGRESS_RELAX="$2"; shift 2 ;;
        --risk_aware_forward_penalty_scale) RISK_AWARE_FORWARD_PENALTY_SCALE="$2"; shift 2 ;;
        --safe_turn_reward_scale) SAFE_TURN_REWARD_SCALE="$2"; shift 2 ;;
        --head_on_avoidance_reward_scale) HEAD_ON_AVOIDANCE_REWARD_SCALE="$2"; shift 2 ;;
        --team_reward_lambda) TEAM_REWARD_LAMBDA="$2"; shift 2 ;;
        --hidden_dim)   HIDDEN_DIM="$2";   shift 2 ;;
        --start_stage)  START_STAGE="$2";  shift 2 ;;
        --end_stage)    END_STAGE="$2";    shift 2 ;;
        --resume)       RESUME_CKPT="$2";  shift 2 ;;
        --exact_resume) EXACT_RESUME=1; shift 1 ;;
        --ros_domain_id) ROS_DOMAIN_ID_OVERRIDE="$2"; shift 2 ;;
        --gazebo_port) GAZEBO_PORT="$2"; shift 2 ;;
        --run_suffix) RUN_SUFFIX="$2"; shift 2 ;;
        --manage_ros_daemon) MANAGE_ROS_DAEMON=1; shift 1 ;;
        --ray_num_cpus) RAY_NUM_CPUS="$2"; shift 2 ;;
        --ray_num_gpus) RAY_NUM_GPUS="$2"; shift 2 ;;
        --ray_object_store_memory_mb) RAY_OBJECT_STORE_MEMORY_MB="$2"; shift 2 ;;
        --parallel_safe) PARALLEL_SAFE=1; shift 1 ;;
        --headless_sim) HEADLESS_SIM=1; shift 1 ;;
        --enable_rviz) ENABLE_RVIZ=1; shift 1 ;;
        --disable_rviz) ENABLE_RVIZ=0; shift 1 ;;
        --rviz_config) RVIZ_CONFIG_OVERRIDE="$2"; shift 2 ;;
        --rviz_node_name) RVIZ_NODE_NAME="$2"; shift 2 ;;
        -h|--help) exit 0 ;;
        *) error "未知参数: $1"; exit 1 ;;
    esac
done

normalize_gat_actor_graph

# ─── 各阶段配置（必须与 train_gnn_mappo_full.py::ENV_CURRICULUM 保持同步）────
# Stage 1a/1b/1c：circle_swap (map 8) 的 2/4/8 车 curriculum
# 短 stage 1（2车收敛快），动障递增逼出泛化协作
# Stage 2: intersection(map 4) 十字路口
declare -A STAGE_MAP_NUM=(       [1]=8     [2]=8     [3]=8     [4]=4 )
declare -A STAGE_NUM_AGENTS=(    [1]=2     [2]=4     [3]=8     [4]=4 )
declare -A STAGE_OBS_NUM=(       [1]=3     [2]=4     [3]=5     [4]=3 )
declare -A STAGE_OBS_SPD=(       [1]=0.5   [2]=0.6   [3]=0.7   [4]=0.6 )
declare -A STAGE_TRAIN_STEPS=(   [1]=100000 [2]=400000 [3]=800000 [4]=600000 )
declare -A STAGE_NAME=(
    [1]="Stage 1 · Circle Swap 2 车 + 3 动障（基础避障）"
    [2]="Stage 2 · Circle Swap 4 车 + 4 动障（多体交互）"
    [3]="Stage 3 · Circle Swap 8 车 + 5 动障（高密度协调）"
    [4]="Stage 4 · 十字交汇 4 车 + 3 动障（场景泛化）"
)

# ─── 【新增】课程学习防休克超参数（防止切图灾难性遗忘） ───────────────
# 学习率：Stage 1 白板训练用 3e-4；Stage 2 切到全新 intersection 交互场景，
# 属"换任务类型"而非"同类微调"，需更高 lr 实质学习（2e-4）；Stage 3/4 才回到微调档。
declare -A STAGE_LR=(       [1]="3e-4" [2]="3e-4" [3]="2e-4" [4]="5e-5" )
# PPO Clip：Stage 2 新场景需要更大策略更新自由度（0.20）；Stage 3 中等保护；Stage 4 收紧保护泛化策略
declare -A STAGE_CLIP=(     [1]="0.20" [2]="0.20" [3]="0.15" [4]="0.10" )

TRAIN_SCRIPT="$CODEBASE_ROOT/gnn_marl_training/train_gnn_mappo_full.py"
KILL_SCRIPT="$WORKSPACE/kill_all_ros.sh"
RUN_SUFFIX_SANITIZED="$(echo "$RUN_SUFFIX" | tr ' /:' '___' | sed 's/[^A-Za-z0-9._-]/_/g')"
RAY_RESULTS_BASE="$WORKSPACE/ray_results"
LOG_DIR_BASE="$WORKSPACE/curriculum_logs"
if [[ -n "$RUN_SUFFIX_SANITIZED" ]]; then
    RAY_RESULTS="$RAY_RESULTS_BASE/$RUN_SUFFIX_SANITIZED"
    LOG_DIR="$LOG_DIR_BASE/$RUN_SUFFIX_SANITIZED"
else
    RAY_RESULTS="$RAY_RESULTS_BASE"
    LOG_DIR="$LOG_DIR_BASE"
fi
mkdir -p "$LOG_DIR"
mkdir -p "$RAY_RESULTS"

if [[ -n "$ROS_DOMAIN_ID_OVERRIDE" ]]; then
    export ROS_DOMAIN_ID="$ROS_DOMAIN_ID_OVERRIDE"
fi
export GAZEBO_MASTER_URI="http://127.0.0.1:${GAZEBO_PORT}"
export GAZEBO_MODEL_DATABASE_URI="${GAZEBO_MODEL_DATABASE_URI:-}"

ROS_PID=""
TRAIN_PID=""

cleanup() {
    echo ""
    warn "收到中断信号，正在清理..."
    [[ -n "$TRAIN_PID" ]] && kill "$TRAIN_PID" 2>/dev/null || true
    if [[ -n "$ROS_PID" ]] && kill -0 "$ROS_PID" 2>/dev/null; then
        kill -TERM -- "-$ROS_PID" 2>/dev/null || kill "$ROS_PID" 2>/dev/null || true
        wait "$ROS_PID" 2>/dev/null || true
    fi
    info "已清理，退出。"
    exit 130
}
trap cleanup SIGINT SIGTERM

check_env() {
    banner "═══ 环境检查 ═══"
    [[ -f "$TRAIN_SCRIPT" ]]   || { error "训练脚本不存在: $TRAIN_SCRIPT"; exit 1; }
    
    if [[ -n "$RESUME_CKPT" ]]; then
        RESUME_CKPT="$(normalize_ckpt_path "$RESUME_CKPT")"
    fi
    success "环境检查通过"
    [[ -n "$RESUME_CKPT" ]] && info "  导入权重自: $RESUME_CKPT"
    [[ -n "${ROS_DOMAIN_ID:-}" ]] && info "  ROS_DOMAIN_ID: ${ROS_DOMAIN_ID}"
    info "  GAZEBO_MASTER_URI: ${GAZEBO_MASTER_URI}"
    if (( HEADLESS_SIM == 1 )); then
        info "  渲染模式: headless"
    else
        info "  渲染模式: gui"
        info "  DISPLAY: ${DISPLAY:-<unset>}"
        [[ -n "${XAUTHORITY:-}" ]] && info "  XAUTHORITY: ${XAUTHORITY}"
    fi
    info "  Ray 输出目录: $RAY_RESULTS"
    info "  课程日志目录: $LOG_DIR"
    require_gui_display
    echo ""
}

stop_ros_env() {
    info "正在停止当前脚本启动的 ROS2/Gazebo 进程..."
    if [[ -n "$ROS_PID" ]] && kill -0 "$ROS_PID" 2>/dev/null; then
        kill -TERM -- "-$ROS_PID" 2>/dev/null || kill "$ROS_PID" 2>/dev/null || true
        wait "$ROS_PID" 2>/dev/null || true
    fi
    # 失败重启后可能残留同名 map_server / lifecycle_manager，仅清端口不够。
    KILL_ALL_ROS_SCOPE=global GAZEBO_PORT="$GAZEBO_PORT" GAZEBO_MASTER_URI="$GAZEBO_MASTER_URI" \
        bash "$KILL_SCRIPT" 2>/dev/null || true
    ROS_PID=""
    sleep 2
}

start_ros_env() {
    local stage=$1
    local map_num=${STAGE_MAP_NUM[$stage]}
    local obs_num=${STAGE_OBS_NUM[$stage]}
    local obs_spd=${STAGE_OBS_SPD[$stage]}
    # Per-stage agent count from curriculum config
    if [[ -n "${STAGE_NUM_AGENTS[$stage]+x}" ]]; then
        NUM_AGENTS=${STAGE_NUM_AGENTS[$stage]}
    fi
    if [[ -n "$OBS_SPEED_SCALE_OVERRIDE" ]]; then
        obs_spd="$OBS_SPEED_SCALE_OVERRIDE"
    fi
    local log="$LOG_DIR/stage${stage}_ros.log"
    local launch_file="main.launch.py"
    if (( HEADLESS_SIM == 1 )); then
        launch_file="main_headless.launch.py"
    fi

    banner "  启动 Gazebo 环境 (Stage $stage · ${STAGE_NAME[$stage]})"

    # 启动前做完整清理，避免残留的 map_server / lifecycle_manager 与新 launch 冲突。
    KILL_ALL_ROS_SCOPE=global GAZEBO_PORT="$GAZEBO_PORT" GAZEBO_MASTER_URI="$GAZEBO_MASTER_URI" \
        bash "$KILL_SCRIPT" 2>/dev/null || true

    local inner_cmd
    inner_cmd="set +u; [[ -f '${CONDA_SH}' ]] && source '${CONDA_SH}'"
    inner_cmd+="; command -v conda >/dev/null 2>&1 && conda activate '${ROS2_CONDA_ENV}' >/dev/null 2>&1 || true"
    inner_cmd+="; source '${ROS_SETUP}'; [[ -f '${WS_SETUP}' ]] && source '${WS_SETUP}'"
    inner_cmd+="; unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY"
    inner_cmd+="; export no_proxy=localhost,127.0.0.1; export GAZEBO_MODEL_DATABASE_URI=''"
    if [[ -n "${ROS_DOMAIN_ID:-}" ]]; then
        inner_cmd+="; export ROS_DOMAIN_ID='${ROS_DOMAIN_ID}'"
    fi
    if (( HEADLESS_SIM == 0 )) && [[ -n "${DISPLAY:-}" ]]; then
        inner_cmd+="; export DISPLAY='${DISPLAY}'"
        if [[ -n "${XAUTHORITY:-}" ]]; then
            inner_cmd+="; export XAUTHORITY='${XAUTHORITY}'"
        fi
    fi
    inner_cmd+="; export GAZEBO_MASTER_URI='${GAZEBO_MASTER_URI}'"
    if (( HEADLESS_SIM == 0 )); then
        local gui_check_cmd
        gui_check_cmd="$(build_gui_display_check_cmd)"
        inner_cmd+="; ${gui_check_cmd}"
    fi
    inner_cmd+="; set -u; echo '=== Stage ${stage}: ${STAGE_NAME[$stage]} ==='"
    inner_cmd+="; stdbuf -oL -eL ros2 launch start_rl_environment_tb3 ${launch_file}"
    inner_cmd+=" map_number:=${map_num} robot_number:=${NUM_AGENTS}"
    inner_cmd+=" num_obstacles:=${obs_num} obs_speed_scale:=${obs_spd}"
    if (( HEADLESS_SIM == 1 )); then
        inner_cmd+=" enable_rviz:=$([[ $ENABLE_RVIZ -eq 1 ]] && echo true || echo false)"
    fi
    if (( ENABLE_RVIZ == 1 )); then
        if [[ -n "$RVIZ_CONFIG_OVERRIDE" ]]; then
            inner_cmd+=" rviz_config:='${RVIZ_CONFIG_OVERRIDE}'"
        fi
        inner_cmd+=" rviz_node_name:='${RVIZ_NODE_NAME}'"
    fi
    inner_cmd+=" 2>&1 | tee '${log}'"

    setsid bash -c "${inner_cmd}" &
    ROS_PID=$!

    if (( MANAGE_ROS_DAEMON == 1 )); then
        timeout 5s ros2 daemon stop  >/dev/null 2>&1 || true
        timeout 5s ros2 daemon start >/dev/null 2>&1 || true
    fi
    sleep 1

    info "等待 Gazebo 就绪（最多 ${GAZEBO_WAIT_SEC}s）..."
    local waited=0
    local topics=""
    local spawned_count=0
    local ros2_cmd="ROS2CLI_NODE_STRATEGY=direct ros2 topic list 2>/dev/null"
    if [[ -n "${ROS_DOMAIN_ID:-}" ]]; then
        ros2_cmd="export ROS_DOMAIN_ID=${ROS_DOMAIN_ID}; ${ros2_cmd}"
    fi
    while [[ $waited -lt $GAZEBO_WAIT_SEC ]]; do
        if [[ -n "$ROS_PID" ]] && ! kill -0 "$ROS_PID" 2>/dev/null; then
            error "launch 进程已退出，Gazebo 启动失败。"
            return 1
        fi
        if [[ -f "$log" ]] && grep -q "Unable to start server\\[bind: Address already in use\\]" "$log"; then
            KILL_ALL_ROS_SCOPE=port_only GAZEBO_PORT="$GAZEBO_PORT" GAZEBO_MASTER_URI="$GAZEBO_MASTER_URI" \
                bash "$KILL_SCRIPT" 2>/dev/null || true
            error "Gazebo 端口冲突：${GAZEBO_MASTER_URI} 已被占用。"
            return 1
        fi
        if [[ -f "$log" ]] && grep -q "process has died .*gzserver" "$log"; then
            error "gzserver 启动失败，请检查日志: $log"
            return 1
        fi
        topics="$(timeout 3s bash -lc "$ros2_cmd" || true)"
        if [[ -f "$log" ]]; then
            spawned_count="$(grep -c "Spawn status: SpawnEntity: Successfully spawned entity \\[tb3_" "$log" 2>/dev/null || true)"
            [[ -z "$spawned_count" ]] && spawned_count=0
        else
            spawned_count=0
        fi
        if [[ "$spawned_count" -ge "$NUM_AGENTS" ]]; then
            success "Gazebo 就绪！（${waited}s，已生成 ${spawned_count}/${NUM_AGENTS} 台机器人）"
            sleep ${GAZEBO_GRACE_SEC}
            return 0
        fi
        if [[ $waited -ge 4 ]] && echo "$topics" | grep -Eq "/tb3_0/(scan|odom)|/model_states_demo|/clock"; then
            success "Gazebo 已完成基础 ROS 发布（${waited}s，spawn=${spawned_count}/${NUM_AGENTS}）"
            sleep ${GAZEBO_GRACE_SEC}
            return 0
        fi
        if (( waited > 0 && waited % 10 == 0 )); then
            info "Gazebo 等待诊断: waited=${waited}s spawn=${spawned_count}/${NUM_AGENTS} domain=${ROS_DOMAIN_ID:-<unset>} topics_hint=$(echo "$topics" | tr '\n' ' ' | cut -c1-200)"
        fi
        sleep 2
        waited=$((waited + 2))
        printf "\r  等待中... %ds / %ds" "$waited" "$GAZEBO_WAIT_SEC"
    done
    error "等待超时，Gazebo 未就绪。"
    return 1
}

find_latest_ckpt() {
    local stage=$1
    local pattern="*Stage${stage}*"
    local run_dir
    run_dir=$(find "$RAY_RESULTS" -maxdepth 1 -type d -name "$pattern" 2>/dev/null \
              | sort -t_ -k1 | tail -1)
    [[ -z "$run_dir" ]] && return 1
    local ckpt
    ckpt=$(find "$run_dir" -maxdepth 3 -type d -name "checkpoint_*" 2>/dev/null \
           | sort -V | tail -1)
    echo "$ckpt"
}

is_valid_checkpoint_dir() {
    local path="${1:-}"
    [[ -d "$path" ]] || return 1
    [[ -f "$path/algorithm_state.pkl" || -f "$path/rllib_checkpoint.json" ]] && return 0
    [[ "$(basename "$path")" =~ ^checkpoint_[0-9]+$ ]] && return 0
    return 1
}

find_checkpoint_under() {
    local root="${1:-}"
    [[ -d "$root" ]] || return 1

    local best=""
    best=$(find "$root" -maxdepth 4 -type d -name best 2>/dev/null | sort | tail -1 || true)
    if [[ -n "$best" ]] && is_valid_checkpoint_dir "$best"; then
        echo "$best"
        return 0
    fi

    local ckpt=""
    ckpt=$(find "$root" -maxdepth 4 -type d -name 'checkpoint_*' 2>/dev/null | sort -V | tail -1 || true)
    if [[ -n "$ckpt" ]] && is_valid_checkpoint_dir "$ckpt"; then
        echo "$ckpt"
        return 0
    fi

    return 1
}

normalize_ckpt_path() {
    local raw="${1:-}"
    raw="$(echo "$raw" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [[ -z "$raw" ]] && { echo ""; return 0; }
    local parsed
    parsed=$(echo "$raw" | sed -n 's/.*path=\([^),]*\).*/\1/p' | head -1)
    if [[ -n "$parsed" ]]; then
        raw="$parsed"
    fi
    if [[ -e "$raw" ]]; then
        if is_valid_checkpoint_dir "$raw"; then
            echo "$raw"
            return 0
        fi
        local nested=""
        nested=$(find_checkpoint_under "$raw" || true)
        if [[ -n "$nested" ]]; then
            warn "检测到 resume 路径并非 checkpoint 目录，自动改为: $nested"
            echo "$nested"
            return 0
        fi
        echo "$raw"
        return 0
    fi
    echo "$raw"
}

build_ray_temp_dir() {
    local suffix="${RUN_SUFFIX_SANITIZED:-default}"
    local compact
    compact="$(echo "$suffix" | tr -cd '[:alnum:]_-' | tr '[:upper:]' '[:lower:]')"
    [[ -z "$compact" ]] && compact="default"

    local token
    if [[ ${#compact} -le 12 ]]; then
        token="$compact"
    else
        local prefix="${compact:0:6}"
        local digest
        digest="$(printf '%s' "$compact" | sha1sum | cut -c1-8)"
        token="${prefix}_${digest}"
    fi
    echo "/tmp/ray_${token}"
}

script_supports_arg() {
    local arg_name="$1"
    grep -q -- "$arg_name" "$TRAIN_SCRIPT" 2>/dev/null
}

run_stage() {
    local stage=$1
    local resume="${2:-}"
    local log="$LOG_DIR/stage${stage}_train.log"
    local ckpt_out=""
    local map_num=${STAGE_MAP_NUM[$stage]}
    local obs_num=${STAGE_OBS_NUM[$stage]}
    local obs_spd=${STAGE_OBS_SPD[$stage]}
    # Per-stage agent count from curriculum config
    if [[ -n "${STAGE_NUM_AGENTS[$stage]+x}" ]]; then
        NUM_AGENTS=${STAGE_NUM_AGENTS[$stage]}
    fi
    if [[ -n "$OBS_SPEED_SCALE_OVERRIDE" ]]; then
        obs_spd="$OBS_SPEED_SCALE_OVERRIDE"
    fi

    # Per-stage train_steps: use curriculum-specific value, fallback to global TRAIN_STEPS
    local train_steps=${STAGE_TRAIN_STEPS[$stage]:-$TRAIN_STEPS}

    # 【核心修改】动态获取当前阶段防休克参数
    local current_lr=${STAGE_LR[$stage]:-3e-4}
    local current_clip=${STAGE_CLIP[$stage]:-0.15}

    # 【智能微调保护】如果你在 Stage 1 导入了权重，说明这不是白板训练，调低 LR 和 Clip 保护权重
    if [[ "$stage" -eq 1 && -n "$resume" && "$EXACT_RESUME" -ne 1 ]]; then
        current_lr="1e-4"
        current_clip="0.10"
        info "检测到 Stage 1 导入了预训练权重，自动启用防休克微调模式 (LR=${current_lr}, Clip=${current_clip})"
    elif [[ "$stage" -eq 1 && -n "$resume" && "$EXACT_RESUME" -eq 1 ]]; then
        info "检测到 Stage 1 精确续训，保留当前阶段默认超参 (LR=${current_lr}, Clip=${current_clip})"
    fi

    banner "═══ Stage $stage · ${STAGE_NAME[$stage]} ═══"
    info "当前超参 -> LR: $current_lr | Clip: $current_clip"

    # 构建 python 命令
    local cmd=(
        python3 "$TRAIN_SCRIPT"
        --model_type  "$MODEL_TYPE"
        --env_stage   "$stage"
        --num_agents  "$NUM_AGENTS"
        --num_workers "$NUM_WORKERS"
        --train_steps "$train_steps"
        --train_batch_size "$TRAIN_BATCH_SIZE"
        --checkpoint_freq "$CHECKPOINT_FREQ"
        --hidden_dim  "$HIDDEN_DIM"
        # 传递动态参数到 Python
        --lr "$current_lr"
        --clip_param "$current_clip"
    )

    if script_supports_arg "--sample_timeout_s"; then cmd+=(--sample_timeout_s "$SAMPLE_TIMEOUT_S"); fi
    if script_supports_arg "--rollout_fragment_length"; then cmd+=(--rollout_fragment_length "$ROLLOUT_FRAGMENT_LENGTH"); fi
    if script_supports_arg "--batch_mode"; then cmd+=(--batch_mode "$BATCH_MODE"); fi
    if script_supports_arg "--map_number"; then cmd+=(--map_number "$map_num"); fi
    if script_supports_arg "--num_obstacles"; then cmd+=(--num_obstacles "$obs_num"); fi
    if script_supports_arg "--obs_speed_scale"; then cmd+=(--obs_speed_scale "$obs_spd"); fi
    if script_supports_arg "--action_mode"; then cmd+=(--action_mode "$ACTION_MODE"); fi
    if script_supports_arg "--graph_ablation"; then cmd+=(--graph_ablation "$GRAPH_ABLATION"); fi
    if script_supports_arg "--ppo_profile"; then cmd+=(--ppo_profile "$PPO_PROFILE"); fi
    if script_supports_arg "--counterfactual_advantage_coef"; then
        cmd+=(--counterfactual_advantage_coef "$COUNTERFACTUAL_ADVANTAGE_COEF")
    fi
    if script_supports_arg "--counterfactual_credit_clip"; then
        cmd+=(--counterfactual_credit_clip "$COUNTERFACTUAL_CREDIT_CLIP")
    fi
    if script_supports_arg "--high_conflict_prob"; then
        cmd+=(--high_conflict_prob "$HIGH_CONFLICT_PROB")
    fi
    if [[ -n "$PREDICTIVE_SOCIAL_PENALTY_SCALE" ]] && script_supports_arg "--predictive_social_penalty_scale"; then
        cmd+=(--predictive_social_penalty_scale "$PREDICTIVE_SOCIAL_PENALTY_SCALE")
    fi
    if [[ -n "$PREDICTIVE_FRONT_PENALTY_SCALE" ]] && script_supports_arg "--predictive_front_penalty_scale"; then
        cmd+=(--predictive_front_penalty_scale "$PREDICTIVE_FRONT_PENALTY_SCALE")
    fi
    if [[ -n "$SOCIAL_PROXIMITY_RISK_SCALE" ]] && script_supports_arg "--social_proximity_risk_scale"; then
        cmd+=(--social_proximity_risk_scale "$SOCIAL_PROXIMITY_RISK_SCALE")
    fi
    if [[ -n "$SUBGOAL_DETOUR_LATERAL_GAIN" ]] && script_supports_arg "--subgoal_detour_lateral_gain"; then
        cmd+=(--subgoal_detour_lateral_gain "$SUBGOAL_DETOUR_LATERAL_GAIN")
    fi
    if [[ -n "$SUBGOAL_PROGRESS_REWARD_SCALE" ]] && script_supports_arg "--subgoal_progress_reward_scale"; then
        cmd+=(--subgoal_progress_reward_scale "$SUBGOAL_PROGRESS_REWARD_SCALE")
    fi
    if [[ -n "$DETOUR_PROGRESS_RELAX" ]] && script_supports_arg "--detour_progress_relax"; then
        cmd+=(--detour_progress_relax "$DETOUR_PROGRESS_RELAX")
    fi
    if [[ -n "$RISK_AWARE_FORWARD_PENALTY_SCALE" ]] && script_supports_arg "--risk_aware_forward_penalty_scale"; then
        cmd+=(--risk_aware_forward_penalty_scale "$RISK_AWARE_FORWARD_PENALTY_SCALE")
    fi
    if [[ -n "$SAFE_TURN_REWARD_SCALE" ]] && script_supports_arg "--safe_turn_reward_scale"; then
        cmd+=(--safe_turn_reward_scale "$SAFE_TURN_REWARD_SCALE")
    fi
    if [[ -n "$HEAD_ON_AVOIDANCE_REWARD_SCALE" ]] && script_supports_arg "--head_on_avoidance_reward_scale"; then
        cmd+=(--head_on_avoidance_reward_scale "$HEAD_ON_AVOIDANCE_REWARD_SCALE")
    fi
    if [[ -n "$TEAM_REWARD_LAMBDA" ]] && script_supports_arg "--team_reward_lambda"; then
        cmd+=(--team_reward_lambda "$TEAM_REWARD_LAMBDA")
    fi
    if script_supports_arg "--output_dir"; then cmd+=(--output_dir "$RAY_RESULTS"); fi
    if script_supports_arg "--run_name_suffix" && [[ -n "$RUN_SUFFIX_SANITIZED" ]]; then
        cmd+=(--run_name_suffix "$RUN_SUFFIX_SANITIZED")
    fi
    if script_supports_arg "--ray_num_cpus"; then cmd+=(--ray_num_cpus "$RAY_NUM_CPUS"); fi
    if script_supports_arg "--ray_num_gpus"; then cmd+=(--ray_num_gpus "$RAY_NUM_GPUS"); fi
    if script_supports_arg "--ray_temp_dir"; then cmd+=(--ray_temp_dir "$(build_ray_temp_dir)"); fi
    if (( PARALLEL_SAFE == 1 )) && script_supports_arg "--skip_ray_stop"; then cmd+=(--skip_ray_stop); fi
    if script_supports_arg "--ray_object_store_memory_mb"; then
        cmd+=(--ray_object_store_memory_mb "$RAY_OBJECT_STORE_MEMORY_MB")
    fi
    if script_supports_arg "--obstacle_filter_range"; then cmd+=(--obstacle_filter_range "$OBSTACLE_FILTER_RANGE"); fi
    if script_supports_arg "--obstacle_filter_fov_deg"; then cmd+=(--obstacle_filter_fov_deg "$OBSTACLE_FILTER_FOV_DEG"); fi
    if script_supports_arg "--obstacle_top_k"; then cmd+=(--obstacle_top_k "$OBSTACLE_TOP_K"); fi
    if script_supports_arg "--rolling_lookahead_dist"; then cmd+=(--rolling_lookahead_dist "$ROLLING_LOOKAHEAD_DIST"); fi
    if script_supports_arg "--tracking_viz_interval"; then cmd+=(--tracking_viz_interval "$TRACKING_VIZ_INTERVAL"); fi
    if script_supports_arg "--action_smooth_alpha"; then cmd+=(--action_smooth_alpha "$ACTION_SMOOTH_ALPHA"); fi
    if script_supports_arg "--safety_filter_enable"; then cmd+=(--safety_filter_enable "$SAFETY_FILTER_ENABLE"); fi
    
    if (( ENABLE_VISUALIZATION == 1 )) && script_supports_arg "--enable_visualization"; then cmd+=(--enable_visualization); fi
    if (( ENABLE_VISUALIZATION == 0 )) && script_supports_arg "--disable_visualization"; then cmd+=(--disable_visualization); fi
    if [[ "$MODEL_TYPE" == "mlp" ]] && script_supports_arg "--mlp_use_comm_obs"; then
        cmd+=(--mlp_use_comm_obs "$MLP_USE_COMM_OBS")
    fi

    # 【核心修复】GAT 模式自动课程，防止在 Stage 1 导入权重时意外跳转到协作通信
    if [[ "$MODEL_TYPE" == "gat" ]]; then
        if script_supports_arg "--gat_actor_graph"; then cmd+=(--gat_actor_graph "$GAT_ACTOR_GRAPH"); fi
        if script_supports_arg "--gat_critic_mode"; then cmd+=(--gat_critic_mode "$GAT_CRITIC_MODE"); fi
        if [[ -n "$resume" && "$stage" -gt 1 ]]; then
            # 只有在环境切换到 Stage 2 及以上，且导入了权重时，才正式开启通信机制
            cmd+=(--curriculum_stage 2)
        else
            # 只要是 Stage 1（无论是否导入权重），都保持独立的无通信状态
            cmd+=(--curriculum_stage 1)
        fi
    fi

    [[ -n "$resume" ]] && cmd+=(--resume_checkpoint "$resume")

    info "训练命令: ${cmd[*]}"
    if [[ -f "$log" && -s "$log" ]]; then
        {
            echo ""
            echo "===== RESUME $(date '+%Y-%m-%d %H:%M:%S') | stage=${stage} ====="
            echo "resume_checkpoint=${resume:-<none>}"
        } >> "$log"
    fi
    
    set +e
    "${cmd[@]}" 2>&1 | tee -a "$log"
    local exit_code=${PIPESTATUS[0]}
    set -e

    if [[ $exit_code -ne 0 ]]; then
        error "Stage $stage 训练异常退出 (code=$exit_code)，检查日志: $log"
        return $exit_code
    fi

    ckpt_out=$(grep -oP '(?<=最佳 Checkpoint: ).*' "$log" | tail -1 || true)
    ckpt_out=$(normalize_ckpt_path "$ckpt_out")
    if [[ -z "$ckpt_out" ]]; then
        ckpt_out=$(find_latest_ckpt "$stage" || true)
    fi

    success "Stage $stage 训练完成！"
    if [[ -n "$ckpt_out" ]]; then
        success "最佳 Checkpoint: $ckpt_out"
        echo "$ckpt_out" > "$LOG_DIR/stage${stage}_best_ckpt.txt"
    fi

    LAST_CKPT="$ckpt_out"
}

main() {
    check_env
    local current_ckpt="$RESUME_CKPT"

    for stage in $(seq "$START_STAGE" "$END_STAGE"); do
        stop_ros_env
        if start_ros_env "$stage"; then
            :
        else
            local ros_exit=$?
            error "Stage $stage 环境启动失败，停止当前 ROS/Gazebo 环境。"
            stop_ros_env
            exit "$ros_exit"
        fi
        
        LAST_CKPT=""
        if run_stage "$stage" "$current_ckpt"; then
            :
        else
            local train_exit=$?
            error "Stage $stage 训练失败，停止当前 ROS/Gazebo 环境。"
            stop_ros_env
            exit "$train_exit"
        fi
        current_ckpt="$LAST_CKPT"

        if (( stage < END_STAGE )); then
            info "等待 5s 后启动下一阶段..."
            sleep 5
        fi
    done

    stop_ros_env
    banner "═══ 课程学习全部完成！═══"
}

main

#./run_curriculum.sh --start_stage 1 --resume /home/wj/work/multi-robot-exploration-rl/ray_results/MAPPO_MLP_LSTM_Stage2_Disc/best
