#!/bin/bash
# =============================================================================
# GNN-MAPPO 一键测试脚本 (适配 params.pkl 自动克隆版)
# 自动：默认按 checkpoint 训练环境启动 Gazebo 与推理测试；也可显式覆盖为指定 Stage preset
#
# 用法示例：
#   ./run_test.sh -c /path/to/checkpoint                      # 默认复用 checkpoint 训练环境
#   ./run_test.sh -c /path/to/checkpoint --test_stage 1       # 强制覆盖为 Stage 1 (静态)，默认启动 Gazebo GUI + RViz
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
CHECKPOINT=""             # 模型权重路径/运行目录
NUM_AGENTS=""             # 机器人数量 (默认自动从 checkpoint 推断)
NUM_EPISODES=5            # 测试回合数
TEST_STAGE=""             # 显式指定时才覆盖 stage preset；默认复用 checkpoint 训练环境
EXPLORE=0                 # 探索标志位：0=关闭，1=开启
TEST_MAX_EPISODE_STEPS=2500
REQUIRE_ALL_DONE=1
DIAG_STEPS=15
FIXED_BENCHMARK_SET=""
BENCHMARK_CSV=""
USE_LATEST=0
LATEST_PATTERN=""
MAP_NUMBER_OVERRIDE=""

RUN_MAP_NUM=""
RUN_OBS_NUM=""
RUN_OBS_SPD_SCALE=""
RUN_ENV_DESC=""
USE_STAGE_OVERRIDE=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$SCRIPT_DIR"
WORKSPACE="/home/wj/work/multi-robot-exploration-rl"
RAY_RESULTS_ROOT="${WORKSPACE}/ray_results"
GAZEBO_WAIT_SEC=60        # 等待 Gazebo 就绪的最长时间（秒）
GAZEBO_GRACE_SEC=5        # 强制等待 Gazebo 完全启动的额外秒数
CONDA_SH="${CONDA_SH:-/home/wj/anaconda3/etc/profile.d/conda.sh}"
ROS2_CONDA_ENV="${ROS2_CONDA_ENV:-ros2}"
REQUIRED_PY_VER="3.10"

normalize_ckpt_path() {
    local raw="${1:-}"
    raw="$(echo "$raw" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [[ -z "$raw" ]] && { echo ""; return 0; }

    if [[ -L "$raw" ]]; then
        raw="$(readlink -f "$raw" 2>/dev/null || echo "$raw")"
    fi

    if [[ -f "$raw" && ! -d "$raw" ]]; then
        local parsed
        parsed="$(sed -n 's/.*path=\([^),]*\).*/\1/p' "$raw" | head -1)"
        if [[ -z "$parsed" ]]; then
            parsed="$(head -n 1 "$raw" | tr -d '\r' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
        fi
        if [[ -n "$parsed" && -e "$parsed" ]]; then
            raw="$parsed"
        fi
    fi

    if [[ -d "$raw" ]]; then
        if [[ -f "$raw/algorithm_state.pkl" || -f "$raw/rllib_checkpoint.json" || "$(basename "$raw")" =~ ^checkpoint_[0-9]+$ ]]; then
            echo "$raw"
            return 0
        fi
        local nested
        nested="$(find "$raw" -maxdepth 4 -type d -name 'checkpoint_*' 2>/dev/null | sort -V | tail -1)"
        if [[ -n "$nested" ]]; then
            echo "$nested"
            return 0
        fi
    fi

    echo "$raw"
}

find_latest_checkpoint() {
    local search_root="${1:-$RAY_RESULTS_ROOT}"
    local pattern="${2:-}"
    [[ -d "$search_root" ]] || return 1

    python3 - "$search_root" "$pattern" <<'PY'
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser().resolve()
pattern = (sys.argv[2] or "").strip().lower()

def is_ckpt_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "algorithm_state.pkl").is_file():
        return True
    meta = path / "rllib_checkpoint.json"
    if meta.is_file():
        try:
            import json
            payload = json.load(open(meta, "r", encoding="utf-8"))
            if str(payload.get("type", "")).strip().lower() == "algorithm":
                return True
        except Exception:
            pass
    return False

candidates = []
for path in root.rglob("*"):
    try:
        if not is_ckpt_dir(path):
            continue
        path_str = str(path).lower()
        parent_str = str(path.parent).lower()
        if pattern and pattern not in path_str and pattern not in parent_str:
            continue
        mtime = path.stat().st_mtime
        candidates.append((mtime, str(path)))
    except Exception:
        continue

if not candidates:
    sys.exit(1)

candidates.sort(key=lambda item: (item[0], item[1]))
print(candidates[-1][1])
PY
}

resolve_checkpoint_input() {
    local raw="${1:-}"
    local resolved=""

    if [[ -z "$raw" ]]; then
        return 1
    fi

    raw="$(normalize_ckpt_path "$raw")"
    if [[ -d "$raw" ]]; then
        if [[ -e "$raw/best" ]]; then
            resolved="$(normalize_ckpt_path "$raw/best")"
            if [[ -e "$resolved" ]]; then
                echo "$resolved"
                return 0
            fi
        fi

        resolved="$(find_latest_checkpoint "$raw" "" 2>/dev/null || true)"
        if [[ -n "$resolved" ]]; then
            echo "$(normalize_ckpt_path "$resolved")"
            return 0
        fi
    fi

    echo "$raw"
}

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

# ─── 解析命令行参数 ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--checkpoint|--checkpoint_path) CHECKPOINT="$2"; shift 2 ;;
        --run_dir|--run-dir) CHECKPOINT="$2"; shift 2 ;;
        --latest) USE_LATEST=1; shift 1 ;;
        --latest_pattern|--latest-pattern) USE_LATEST=1; LATEST_PATTERN="$2"; shift 2 ;;
	    --num_agents) NUM_AGENTS="$2"; shift 2 ;;
	    --num_episodes) NUM_EPISODES="$2"; shift 2 ;;
	    --test_stage) TEST_STAGE="$2"; shift 2 ;;
	    --map_number) MAP_NUMBER_OVERRIDE="$2"; shift 2 ;;
	    --test_max_episode_steps) TEST_MAX_EPISODE_STEPS="$2"; shift 2 ;;
        --require_all_done) REQUIRE_ALL_DONE="$2"; shift 2 ;;
        --diag_steps) DIAG_STEPS="$2"; shift 2 ;;
        --fixed_benchmark_set) FIXED_BENCHMARK_SET="$2"; shift 2 ;;
        --benchmark_csv) BENCHMARK_CSV="$2"; shift 2 ;;
        -e|--explore) EXPLORE=1; shift 1 ;; # 新增探索参数
        -h|--help)
            echo "用法: ./run_test.sh [-c <checkpoint_or_run_dir> | --latest] [OPTIONS]"
            echo "  -c, --checkpoint, --checkpoint_path  checkpoint 路径，或训练 run 目录"
            echo "  --run_dir            与 -c 等价，显式表示传入的是训练 run 目录"
            echo "  --latest             自动从 ${RAY_RESULTS_ROOT} 选择最新 checkpoint/best"
            echo "  --latest_pattern     仅在路径包含该关键词时参与 latest 选择"
	    echo "  --num_agents         启动 Gazebo 时的机器人数量 (默认2)"
	    echo "  --num_episodes       测试回合数 (默认5)"
	    echo "  --test_stage         显式覆盖为 Stage preset (1-4)"
	    echo "  --map_number         覆盖地图编号，同时用于 Gazebo 和测试 env"
	    echo "  --test_max_episode_steps  单回合最大步数 (默认2500)"
            echo "  --require_all_done   1=所有机器人完成才结束回合，0=沿用 checkpoint 语义"
            echo "  --diag_steps         前 N 步打印诊断信息 (默认15)"
            echo "  --fixed_benchmark_set  固定 benchmark 场景集名称（交由 test_gnn_mappo.py 解析）"
            echo "  --benchmark_csv      固定 benchmark 逐场景结果 CSV 输出路径"
            echo "  -e, --explore        【强烈建议】开启探索噪声，打破狭窄空间死锁"
            echo "  默认: Gazebo 与测试环境都复用 checkpoint 的训练配置"
            exit 0 ;;
        *) error "未知参数: $1"; exit 1 ;;
    esac
done

if (( USE_LATEST == 1 )); then
    CHECKPOINT="$(find_latest_checkpoint "$RAY_RESULTS_ROOT" "$LATEST_PATTERN" 2>/dev/null || true)"
    if [[ -z "$CHECKPOINT" ]]; then
        error "未找到最新 checkpoint。搜索目录: $RAY_RESULTS_ROOT 关键词: ${LATEST_PATTERN:-<none>}"
        exit 1
    fi
fi

if [[ -z "$CHECKPOINT" ]]; then
    error "必须提供 Checkpoint/Run 路径，或使用 --latest"
    exit 1
fi

CHECKPOINT="$(resolve_checkpoint_input "$CHECKPOINT")"

# ─── 阶段配置（复用训练配置）────────────────────────────────────────────────
declare -A STAGE_MAP_NUM=(  [1]=5 [2]=3 [3]=3 [4]=3 )
declare -A STAGE_OBS_NUM=(  [1]=0 [2]=2 [3]=6 [4]=8 )
declare -A STAGE_OBS_SPD=(  [1]=0.0 [2]=0.35 [3]=0.9 [4]=1.3 )
declare -A STAGE_NAME=(
    [1]="Stage 1 · 静态入门"
    [2]="Stage 2 · 静态变长"
    [3]="Stage 3 · 慢速动态障碍"
    [4]="Stage 4 · 完整任务"
)

infer_runtime_env_from_checkpoint() {
    python3 - "$CHECKPOINT" <<'PY'
import os
import sys
import json
from ray import cloudpickle

ckpt = os.path.abspath(os.path.expanduser(sys.argv[1]))

def resolve_state_file(path: str) -> str:
    if os.path.isdir(path):
        meta = os.path.join(path, "rllib_checkpoint.json")
        if os.path.isfile(meta):
            try:
                payload = json.load(open(meta, "r", encoding="utf-8"))
                state_file = payload.get("state_file")
                if state_file and os.path.isfile(state_file):
                    return state_file
            except Exception:
                pass
        candidate = os.path.join(path, "algorithm_state.pkl")
        if os.path.isfile(candidate):
            return candidate
    return path

state_file = resolve_state_file(ckpt)
with open(state_file, "rb") as f:
    state = cloudpickle.load(f)
cfg = dict((state or {}).get("config") or {})
env = dict(cfg.get("env_config") or {})
map_number = int(env.get("map_number", 3))
num_dynamic_obstacles = int(env.get("num_dynamic_obstacles", 8))
obs_speed = float(env.get("obs_speed", 0.3))
obs_speed_scale = obs_speed / 0.3 if abs(0.3) > 1e-9 else 1.0
print(f"{map_number}\t{num_dynamic_obstacles}\t{obs_speed_scale:.6f}")
PY
}

resolve_test_runtime() {
    if [[ -n "$TEST_STAGE" ]]; then
        USE_STAGE_OVERRIDE=1
        RUN_MAP_NUM="${STAGE_MAP_NUM[$TEST_STAGE]}"
        RUN_OBS_NUM="${STAGE_OBS_NUM[$TEST_STAGE]}"
        RUN_OBS_SPD_SCALE="${STAGE_OBS_SPD[$TEST_STAGE]}"
        RUN_ENV_DESC="Stage $TEST_STAGE (${STAGE_NAME[$TEST_STAGE]})"
        if [[ -n "$MAP_NUMBER_OVERRIDE" ]]; then
            RUN_MAP_NUM="$MAP_NUMBER_OVERRIDE"
            RUN_ENV_DESC="${RUN_ENV_DESC} + map ${MAP_NUMBER_OVERRIDE}"
        fi
        return 0
    fi

    local inferred
    inferred="$(infer_runtime_env_from_checkpoint 2>/dev/null || true)"
    if [[ -n "$inferred" ]]; then
        IFS=$'\t' read -r RUN_MAP_NUM RUN_OBS_NUM RUN_OBS_SPD_SCALE <<< "$inferred"
    fi

    if [[ ! "$RUN_MAP_NUM" =~ ^[0-9]+$ ]] || [[ ! "$RUN_OBS_NUM" =~ ^[0-9]+$ ]] || [[ -z "$RUN_OBS_SPD_SCALE" ]]; then
        warn "无法从 checkpoint 推断训练环境，回退为 Stage 4 preset"
        USE_STAGE_OVERRIDE=1
        TEST_STAGE=4
        RUN_MAP_NUM="${STAGE_MAP_NUM[$TEST_STAGE]}"
        RUN_OBS_NUM="${STAGE_OBS_NUM[$TEST_STAGE]}"
        RUN_OBS_SPD_SCALE="${STAGE_OBS_SPD[$TEST_STAGE]}"
        RUN_ENV_DESC="Stage $TEST_STAGE (${STAGE_NAME[$TEST_STAGE]})"
        return 0
    fi

    USE_STAGE_OVERRIDE=0
    RUN_ENV_DESC="Checkpoint 训练环境"
    if [[ -n "$MAP_NUMBER_OVERRIDE" ]]; then
        RUN_MAP_NUM="$MAP_NUMBER_OVERRIDE"
        USE_STAGE_OVERRIDE=1
        RUN_ENV_DESC="${RUN_ENV_DESC} + map ${MAP_NUMBER_OVERRIDE}"
    fi
}

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
import json
from ray import cloudpickle

ckpt = os.path.abspath(os.path.expanduser(sys.argv[1]))

def resolve_state_file(path: str) -> str:
    if os.path.isdir(path):
        meta = os.path.join(path, "rllib_checkpoint.json")
        if os.path.isfile(meta):
            try:
                payload = json.load(open(meta, "r", encoding="utf-8"))
                state_file = payload.get("state_file")
                if state_file and os.path.isfile(state_file):
                    return state_file
            except Exception:
                pass
        candidate = os.path.join(path, "algorithm_state.pkl")
        if os.path.isfile(candidate):
            return candidate
    return path

state_file = resolve_state_file(ckpt)
with open(state_file, "rb") as f:
    state = cloudpickle.load(f)
cfg = dict((state or {}).get("config") or {})
env = dict(cfg.get("env_config") or {})
print(int(env.get("num_agents", 2)))
PY
)"
        if [[ "$inferred" =~ ^[0-9]+$ ]]; then
            NUM_AGENTS="$inferred"
        else
            NUM_AGENTS=2
            warn "无法从 checkpoint 推断 num_agents，回退为 2"
        fi
    fi
    resolve_test_runtime
    success "环境检查通过"
    info "  仿真环境机器人数量: $NUM_AGENTS"
    info "  测试回合:           $NUM_EPISODES"
    info "  回合最大步数:       $TEST_MAX_EPISODE_STEPS"
    info "  所有机器人完成才结束: $REQUIRE_ALL_DONE"
    info "  诊断步数:           $DIAG_STEPS"
    info "  探索模式 (Explore): $( ((EXPLORE==1)) && echo "开启" || echo "关闭" )"
    info "  测试环境:           $RUN_ENV_DESC"
    info "  地图编号:           $RUN_MAP_NUM"
    info "  动态障碍物数量:     $RUN_OBS_NUM"
    info "  动态障碍物速度系数: $RUN_OBS_SPD_SCALE"
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
    local map_num="$RUN_MAP_NUM"
    local obs_num="$RUN_OBS_NUM"
    local obs_spd="$RUN_OBS_SPD_SCALE"

    banner "  启动测试仿真环境 (${RUN_ENV_DESC})"

    local inner_cmd
    inner_cmd="set +u; [[ -f '${CONDA_SH}' ]] && source '${CONDA_SH}'"
    inner_cmd+="; command -v conda >/dev/null 2>&1 && conda activate '${ROS2_CONDA_ENV}' >/dev/null 2>&1 || true"
    inner_cmd+="; source '${ROS_SETUP}'; [[ -f '${WS_SETUP}' ]] && source '${WS_SETUP}'"
    inner_cmd+="; unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY"
    inner_cmd+="; export no_proxy=localhost,127.0.0.1; export GAZEBO_MODEL_DATABASE_URI=''"
    inner_cmd+="; set -u; echo '=== 启动测试环境 ==='"
    # main.launch.py starts Gazebo GUI and RViz by default for visual inspection.
    inner_cmd+="; ros2 launch start_rl_environment_tb3 main.launch.py"
    # 这里我们保留了 $NUM_AGENTS，因为启动 Gazebo 必须得知道要放几台车进去
    inner_cmd+=" map_number:=${map_num} robot_number:=${NUM_AGENTS}"
    inner_cmd+=" num_obstacles:=${obs_num} obs_speed_scale:=${obs_spd}"

    local launched=0
    local has_gui=0
    if [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]]; then has_gui=1; fi

    if (( has_gui == 1 )) && command -v gnome-terminal &>/dev/null; then
        if gnome-terminal --title="[Test] ROS2 环境" -- bash -c "${inner_cmd}"; then
            launched=1
        fi
    fi
    if (( launched == 0 )) && (( has_gui == 1 )) && command -v xterm &>/dev/null; then
        if xterm -title "[Test] ROS2 环境" -e bash -c "${inner_cmd}" &>/dev/null & then
            launched=1
        fi
    fi
    if (( launched == 0 )); then
        bash -c "${inner_cmd}" &
        ROS_PID=$!
    fi

    timeout 5s ros2 daemon stop  >/dev/null 2>&1 || true
    timeout 5s ros2 daemon start >/dev/null 2>&1 || true
    sleep 1

    info "等待 Gazebo 就绪（最多 ${GAZEBO_WAIT_SEC}s）..."
    local waited=0
    local topics=""
    while [[ $waited -lt $GAZEBO_WAIT_SEC ]]; do
        topics="$(timeout 3s bash -lc 'ROS2CLI_NODE_STRATEGY=direct ros2 topic list 2>/dev/null' || true)"
        if echo "$topics" | grep -Eq "/tb3_0/(scan|odom)"; then
            echo ""
            success "Gazebo 就绪！(${waited}s)"
            sleep ${GAZEBO_GRACE_SEC}
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
        printf "\r  等待中... %ds / %ds" "$waited" "$GAZEBO_WAIT_SEC"
    done
    echo ""
    error "等待超时，Gazebo 未就绪。"
    return 1
}

# ─── 运行测试 ─────────────────────────────────────────────────────────────────
run_test() {
    banner "═══ 开始推理测试 ═══"
    local cmd=(
        python3 "$TEST_SCRIPT"
        --checkpoint_path "$CHECKPOINT"
        --num_episodes "$NUM_EPISODES"
        --test_max_episode_steps "$TEST_MAX_EPISODE_STEPS"
        --require_all_done "$REQUIRE_ALL_DONE"
        --diag_steps "$DIAG_STEPS"
    )

    if (( USE_STAGE_OVERRIDE == 1 )); then
        cmd+=(
            --map_number "$RUN_MAP_NUM"
            --num_dynamic_obstacles "$RUN_OBS_NUM"
            --obs_speed_scale "$RUN_OBS_SPD_SCALE"
        )
    fi

    if [[ -n "$FIXED_BENCHMARK_SET" ]]; then
        cmd+=(--fixed_benchmark_set "$FIXED_BENCHMARK_SET")
    fi
    if [[ -n "$BENCHMARK_CSV" ]]; then
        cmd+=(--benchmark_csv "$BENCHMARK_CSV")
    fi

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
    if start_ros_env; then
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
