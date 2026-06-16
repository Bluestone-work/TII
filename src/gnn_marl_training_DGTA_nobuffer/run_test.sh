#!/bin/bash
# =============================================================================
# GNN-MAPPO 一键测试脚本 (适配 params.pkl 自动克隆版)
# 自动：启动指定阶段的环境 → 运行推理测试 → 打印结果 → 清理环境
#
# 用法示例：
#   ./run_test.sh -c /path/to/checkpoint                      # 默认测试 Stage 4 (最难)
#   ./run_test.sh -c /path/to/checkpoint --test_stage 1       # 测试 Stage 1 (静态)
#   ./run_test.sh -c /path/to/checkpoint -e                   # 开启探索噪声 (打破会车死锁)
#   ./run_test.sh -c /path/to/checkpoint --num_agents 3       # 告诉 Gazebo 生成 3 辆车
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
CHECKPOINT=""             # 模型权重路径 (必须)
NUM_AGENTS=""             # 机器人数量 (默认自动从 checkpoint 推断)
NUM_EPISODES=5            # 测试回合数
TEST_STAGE=4              # 默认在最难的 Stage 4 进行测试
EXPLORE=0                 # 探索标志位：0=关闭，1=开启

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$SCRIPT_DIR"
WORKSPACE="/home/wj/work/multi-robot-exploration-rl"
GAZEBO_WAIT_SEC=120       # 等待 Gazebo 就绪的最长时间（秒）
GAZEBO_GRACE_SEC=8        # 强制等待 Gazebo 完全启动的额外秒数
CONDA_SH="${CONDA_SH:-/home/wj/anaconda3/etc/profile.d/conda.sh}"
ROS2_CONDA_ENV="${ROS2_CONDA_ENV:-ros2}"
REQUIRED_PY_VER="3.10"

# ─── Python 环境检查与挂载 ──────────────────────────────────────────────────
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
# CRITICAL: prepend CODEBASE_ROOT AFTER sourcing install/setup.bash, so this codebase wins over the older install-tree package.
export PYTHONPATH="${CODEBASE_ROOT}:${PYTHONPATH:-}"

# ─── 解析命令行参数 ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--checkpoint)  CHECKPOINT="$2";   shift 2 ;;
        --num_agents)     NUM_AGENTS="$2";   shift 2 ;;
        --num_episodes)   NUM_EPISODES="$2"; shift 2 ;;
        --test_stage)     TEST_STAGE="$2";   shift 2 ;;
        -e|--explore)     EXPLORE=1;         shift 1 ;; # 新增探索参数
        -h|--help)
            echo "用法: ./run_test.sh -c <checkpoint_path> [OPTIONS]"
            echo "  -c, --checkpoint     模型权重路径 (必需)"
            echo "  --num_agents         启动 Gazebo 时的机器人数量 (默认2)"
            echo "  --num_episodes       测试回合数 (默认5)"
            echo "  --test_stage         测试环境复杂度 1-4 (默认4)"
            echo "  -e, --explore        【强烈建议】开启探索噪声，打破狭窄空间死锁"
            exit 0 ;;
        *) error "未知参数: $1"; exit 1 ;;
    esac
done

if [[ -z "$CHECKPOINT" ]]; then
    error "必须提供 Checkpoint 路径！例如: ./run_test.sh -c /path/to/checkpoint"
    exit 1
fi

# ─── 阶段配置（复用训练配置）────────────────────────────────────────────────
declare -A STAGE_MAP_NUM=(  [1]=5 [2]=4 [3]=3 [4]=3 )
declare -A STAGE_OBS_NUM=(  [1]=0 [2]=2 [3]=6 [4]=8 )
declare -A STAGE_OBS_SPD=(  [1]=0.0 [2]=0.35 [3]=0.9 [4]=1.3 )
declare -A STAGE_NAME=(
    [1]="Stage 1 · 静态入门"
    [2]="Stage 2 · 十字交汇（多体交互）"
    [3]="Stage 3 · 慢速动态障碍"
    [4]="Stage 4 · 完整任务"
)

# ─── 路径 ────────────────────────────────────────────────────────────────────
TEST_SCRIPT="$CODEBASE_ROOT/gnn_marl_training/test_gnn_mappo.py"
KILL_SCRIPT="$WORKSPACE/kill_all_ros.sh"

ROS_PID=""
TEST_PID=""

# ─── 清理函数（Ctrl+C 安全退出）─────────────────────────────────────────────
cleanup() {
    echo ""
    warn "测试结束/收到中断信号，正在清理环境..."
    [[ -n "$TEST_PID" ]] && kill "$TEST_PID" 2>/dev/null || true
    bash "$KILL_SCRIPT" 2>/dev/null || true
    info "已清理，退出。"
    exit 130
}
trap cleanup SIGINT SIGTERM

# ─── 环境检查 ─────────────────────────────────────────────────────────────────
check_env() {
    banner "═══ 测试环境检查 ═══"
    [[ -f "$TEST_SCRIPT" ]]   || { error "测试脚本不存在: $TEST_SCRIPT"; exit 1; }
    [[ -f "$KILL_SCRIPT" ]]   || { error "kill 脚本不存在: $KILL_SCRIPT";  exit 1; }
    if [[ ! -e "$CHECKPOINT" ]]; then
        error "Checkpoint 路径不存在: $CHECKPOINT"; exit 1
    fi
    if [[ -z "$NUM_AGENTS" ]]; then
        local inferred
        inferred="$(python3 - "$CHECKPOINT" <<'PY'
import os
import sys
from ray.rllib.algorithms.algorithm import Algorithm
from gnn_marl_training.counterfactual_ppo_policy import register_counterfactual_policy

ckpt = os.path.abspath(os.path.expanduser(sys.argv[1]))
register_counterfactual_policy()
algo = Algorithm.from_checkpoint(ckpt)
try:
    print(int(algo.config.env_config.get("num_agents", 2)))
finally:
    algo.stop()
PY
)"
        if [[ "$inferred" =~ ^[0-9]+$ ]]; then
            NUM_AGENTS="$inferred"
        else
            NUM_AGENTS=2
            warn "无法从 checkpoint 推断 num_agents，回退为 2"
        fi
    fi
    success "环境检查通过"
    info "  仿真环境机器人数量: $NUM_AGENTS"
    info "  测试回合:           $NUM_EPISODES"
    info "  探索模式 (Explore): $( ((EXPLORE==1)) && echo "开启" || echo "关闭" )"
    info "  测试环境:           Stage $TEST_STAGE (${STAGE_NAME[$TEST_STAGE]})"
    info "  权重路径:           $CHECKPOINT"
    echo ""
}

# ─── 停止旧环境 ───────────────────────────────────────────────────────────────
stop_ros_env() {
    info "正在停止旧的 ROS2/Gazebo 进程..."
    bash "$KILL_SCRIPT" 2>/dev/null || true
    sleep 2
}

# ─── 启动 Gazebo 环境 ─────────────────────────────────────────────────────────
start_ros_env() {
    local stage=$1
    local map_num=${STAGE_MAP_NUM[$stage]}
    local obs_num=${STAGE_OBS_NUM[$stage]}
    local obs_spd=${STAGE_OBS_SPD[$stage]}

    banner "  启动测试仿真环境 (Stage $stage · ${STAGE_NAME[$stage]})"

    local log="/tmp/run_test_gazebo_$$.log"

    local inner_cmd
    inner_cmd="set +u; [[ -f '${CONDA_SH}' ]] && source '${CONDA_SH}'"
    inner_cmd+="; command -v conda >/dev/null 2>&1 && conda activate '${ROS2_CONDA_ENV}' >/dev/null 2>&1 || true"
    inner_cmd+="; source '${ROS_SETUP}'; [[ -f '${WS_SETUP}' ]] && source '${WS_SETUP}'"
    inner_cmd+="; unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY"
    inner_cmd+="; export no_proxy=localhost,127.0.0.1; export GAZEBO_MODEL_DATABASE_URI=''"
    inner_cmd+="; export TURTLEBOT3_MODEL=burger"
    inner_cmd+="; set -u; echo '=== 启动测试环境 ==='"
    inner_cmd+="; ros2 launch start_rl_environment_tb3 main.launch.py"
    inner_cmd+=" map_number:=${map_num} robot_number:=${NUM_AGENTS}"
    inner_cmd+=" num_obstacles:=${obs_num} obs_speed_scale:=${obs_spd}"

    # Always run in background with log capture (gnome-terminal loses stdout)
    bash -c "${inner_cmd}" > "$log" 2>&1 &
    ROS_PID=$!

    timeout 5s ros2 daemon stop  >/dev/null 2>&1 || true
    timeout 5s ros2 daemon start >/dev/null 2>&1 || true
    sleep 1

    info "等待 Gazebo 就绪（最多 ${GAZEBO_WAIT_SEC}s）..."
    local waited=0
    local topics="" spawned_count=0
    while [[ $waited -lt $GAZEBO_WAIT_SEC ]]; do
        # Check if launch process died
        if ! kill -0 "$ROS_PID" 2>/dev/null; then
            error "launch 进程已退出，Gazebo 启动失败。查看日志: $log"
            tail -20 "$log" 2>/dev/null
            return 1
        fi
        topics="$(timeout 3s bash -lc 'ROS2CLI_NODE_STRATEGY=direct ros2 topic list 2>/dev/null' || true)"
        # Check spawn count from log
        spawned_count="$(grep -c "Successfully spawned entity \[tb3_" "$log" 2>/dev/null || true)"
        [[ -z "$spawned_count" ]] && spawned_count=0
        if [[ "$spawned_count" -ge "$NUM_AGENTS" ]]; then
            echo ""
            success "Gazebo 就绪！(${waited}s, spawn=${spawned_count}/${NUM_AGENTS})"
            sleep ${GAZEBO_GRACE_SEC}
            return 0
        fi
        # Fallback: if scan topic exists + waited enough, proceed anyway
        if [[ $waited -ge 15 ]] && echo "$topics" | grep -Eq "/tb3_0/(scan|odom)"; then
            echo ""
            success "Gazebo 就绪 (topics detected, spawn=${spawned_count}/${NUM_AGENTS}, ${waited}s)"
            sleep ${GAZEBO_GRACE_SEC}
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
        if (( waited % 10 == 0 )); then
            info "等待中... ${waited}s / ${GAZEBO_WAIT_SEC}s (spawn=${spawned_count}/${NUM_AGENTS})"
        fi
    done
    echo ""
    error "等待超时，Gazebo 未就绪。最后日志:"
    tail -30 "$log" 2>/dev/null
    return 1
}

# ─── 运行测试 ─────────────────────────────────────────────────────────────────
run_test() {
    banner "═══ 开始推理测试 ═══"
    local map_num=${STAGE_MAP_NUM[$TEST_STAGE]}
    local obs_num=${STAGE_OBS_NUM[$TEST_STAGE]}
    local obs_spd=${STAGE_OBS_SPD[$TEST_STAGE]}

    # 新版 Python 脚本极简了参数
    local py_bin="${ROS2_CONDA_ENV_PYTHON:-/home/wj/anaconda3/envs/${ROS2_CONDA_ENV}/bin/python}"
    if [[ ! -x "$py_bin" ]]; then py_bin="python3"; fi
    local cmd=(
        "$py_bin" "$TEST_SCRIPT"
        --checkpoint_path "$CHECKPOINT"
        --num_episodes "$NUM_EPISODES"
        --map_number "$map_num"
        --num_dynamic_obstacles "$obs_num"
        --obs_speed_scale "$obs_spd"
    )

    # 动态附加 explore 标志
    if (( EXPLORE == 1 )); then
        cmd+=(--explore)
    fi

    info "测试命令: ${cmd[*]}"
    echo ""

    set +e
    "${cmd[@]}"
    local exit_code=$?
    set -e

    if [[ $exit_code -ne 0 ]]; then
        error "测试异常退出 (code=$exit_code)"
    else
        success "测试顺利完成！"
    fi
}

# ─── 主流程 ───────────────────────────────────────────────────────────────────
main() {
    check_env
    stop_ros_env
    
    # 启动 Gazebo
    if start_ros_env "$TEST_STAGE"; then
        # 运行推理
        run_test
    else
        error "仿真环境启动失败，中止测试。"
    fi

    # 清理
    stop_ros_env
    banner "═══ 测试脚本执行完毕 ═══"
}

main
