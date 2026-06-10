#!/bin/bash
# =============================================================================
# GNN-MAPPO 一键测试脚本
# 自动：启动指定阶段环境 → 运行推理测试 → 保存日志 → 汇总结果 → 清理环境
#
# 示例：
#   ./run_test.sh -c /path/to/checkpoint
#   ./run_test.sh -c /path/to/checkpoint --all_stages --repeat_runs 3
#   ./run_test.sh -c /path/to/checkpoint --test_stages 1,3,4 --num_episodes 10
# =============================================================================

if [ -z "${BASH_VERSION:-}" ]; then
    exec /bin/bash "$0" "$@"
fi

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
info()    { echo -e "${CYAN}[INFO]${RESET} $*"; }
success() { echo -e "${GREEN}[✓]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[⚠]${RESET} $*"; }
error()   { echo -e "${RED}[✗]${RESET} $*" >&2; }
banner()  { echo -e "\n${BOLD}${CYAN}$*${RESET}\n"; }

CHECKPOINT=""
NUM_AGENTS="4"
NUM_EPISODES=5
TEST_MAX_EPISODE_STEPS=2500
TEST_STAGE=4
TEST_STAGES=""
RUN_ALL_STAGES=0
TEST_STAGE_EXPLICIT=0
FIXED_BENCHMARK_SET=""
BENCHMARK_CSV=""
MAP_NUMBER_OVERRIDE=""
REPEAT_RUNS=1
EXPLORE=0
DIAG_STEPS=15
SHIELD_ENABLE=0
SHIELD_FRONT_SLOW_DIST=0.60
SHIELD_FRONT_STOP_DIST=0.26
SHIELD_NEIGHBOR_SLOW_DIST=0.45
SHIELD_LINEAR_SLOW=0.12
SHIELD_LINEAR_STOP=0.04
SHIELD_TURN_BIAS=0.35
TURN_IN_PLACE_FRONT_DIST=0.40
TURN_IN_PLACE_ANGLE_THRESH=0.45
TURN_IN_PLACE_W=0.90
ROLLING_LOOKAHEAD_DIST=0.4
SHIELD_OVERRIDE_SET=0
TEST_ROS_DOMAIN_ID=71
TEST_GAZEBO_PORT=11845
HEADLESS_SIM=0
ENABLE_RVIZ=1
RENDER_MODE="full_gui"   # headless | rviz | full_gui
SAVE_RVIZ_GIF=0
RVIZ_GIF_FPS=10
RVIZ_GIF_SCALE="960:-1"
RVIZ_GIF_DURATION=0
RVIZ_GIF_DIR=""
SUMMARY_DIR=""
SUMMARY_CSV=""
RUN_TAG=""

WORKSPACE="/home/wj/work/multi-robot-exploration-rl"
GAZEBO_WAIT_SEC=60
GAZEBO_GRACE_SEC=5
CONDA_SH="${CONDA_SH:-/home/wj/anaconda3/etc/profile.d/conda.sh}"
ROS2_CONDA_ENV="${ROS2_CONDA_ENV:-ros2}"
REQUIRED_PY_VER="3.10"
LOG_DIR="$WORKSPACE/train_logs"
SUMMARY_ROOT="$WORKSPACE/train_logs/test_summaries"

TEST_SCRIPT="$WORKSPACE/src/gnn_marl_training/gnn_marl_training/test_gnn_mappo.py"
KILL_SCRIPT="$WORKSPACE/kill_all_ros.sh"
ROS_SETUP="/opt/ros/humble/setup.bash"
WS_SETUP="${WORKSPACE}/install/setup.bash"

TEST_ENV_LOG=""
ROS_PID=""
TEST_PID=""
RVIZ_GIF_PID=""
RVIZ_GIF_PATH=""
declare -a STAGE_LIST=()

# 必须与 train_gnn_mappo_full.py::ENV_CURRICULUM 保持同步
declare -A STAGE_MAP_NUM=(  [1]=5 [2]=3 [3]=3 [4]=3 )
declare -A STAGE_OBS_NUM=(  [1]=0 [2]=2 [3]=6 [4]=8 )
declare -A STAGE_OBS_SPD=(  [1]=0.0 [2]=0.35 [3]=0.9 [4]=1.3 )
declare -A STAGE_NAME=(
    [1]="Stage 1 · 静态入门"
    [2]="Stage 2 · 静态变长"
    [3]="Stage 3 · 慢速动态障碍"
    [4]="Stage 4 · 完整任务"
)

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

infer_checkpoint_num_agents() {
    local ckpt_path="$1"
    python3 - "$ckpt_path" <<'PY'
import json
import os
import pickle
import sys

start = os.path.abspath(sys.argv[1])
candidates = []
cur = start
for _ in range(6):
    candidates.append(cur)
    parent = os.path.dirname(cur)
    if parent == cur:
        break
    cur = parent

def load_mapping(path):
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)

for base in candidates:
    for name in ("params.json", "params.pkl"):
        path = os.path.join(base, name)
        if not os.path.exists(path):
            continue
        try:
            data = load_mapping(path)
        except Exception:
            continue
        env_config = data.get("env_config", {}) if isinstance(data, dict) else {}
        num_agents = env_config.get("num_agents")
        if isinstance(num_agents, int) and num_agents > 0:
            print(num_agents)
            raise SystemExit(0)
print("")
PY
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

    if is_valid_checkpoint_dir "$root"; then
        echo "$root"
        return 0
    fi

    local best=""
    best=$(
        find "$root" -maxdepth 6 -type d \( -name best -o -name final -o -name 'checkpoint_*' \) \
            -printf '%T@ %p\n' 2>/dev/null \
            | sort -nr \
            | awk 'NR==1 {print $2}'
    )
    if [[ -n "$best" ]] && is_valid_checkpoint_dir "$best"; then
        echo "$best"
        return 0
    fi

    return 1
}

normalize_ckpt_path() {
    local raw="${1:-}"
    raw="$(echo "$raw" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [[ -z "$raw" ]] && { echo ""; return 0; }

    local parsed=""
    parsed=$(echo "$raw" | sed -n 's/.*path=\([^),]*\).*/\1/p' | head -1)
    if [[ -n "$parsed" ]]; then
        raw="$parsed"
    fi

    if [[ -f "$raw" ]]; then
        local base
        base="$(basename "$raw")"
        if [[ "$base" == "rllib_checkpoint.json" || "$base" == "algorithm_state.pkl" ]]; then
            raw="$(dirname "$raw")"
        fi
    fi

    if [[ -d "$raw" ]]; then
        local nested=""
        nested="$(find_checkpoint_under "$raw" || true)"
        if [[ -n "$nested" ]]; then
            if [[ "$nested" != "$raw" ]]; then
                warn "检测到输入路径不是最终 checkpoint，自动改为: $nested"
            fi
            echo "$nested"
            return 0
        fi
    fi

    echo "$raw"
}

infer_checkpoint_stage() {
    local ckpt_path="$1"
    local abs_path
    abs_path="$(readlink -f "$ckpt_path" 2>/dev/null || python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$ckpt_path")"

    if [[ "$abs_path" =~ [Ee]nv[Ss]tage([1-4]) ]]; then
        echo "${BASH_REMATCH[1]}"
        return 0
    fi
    if [[ "$abs_path" =~ (^|[/_-])[Ss]tage([1-4])([/_-]|$) ]]; then
        echo "${BASH_REMATCH[2]}"
        return 0
    fi

    echo ""
    return 0
}

build_default_summary_dir() {
    local ckpt_name
    ckpt_name="$(basename "$CHECKPOINT" | tr ' /:' '___' | sed 's/[^A-Za-z0-9._-]/_/g')"
    echo "$SUMMARY_ROOT/${ckpt_name}_$(date +%Y%m%d_%H%M%S)"
}

parse_stage_list() {
    local raw="$1"
    local token
    IFS=',' read -r -a tokens <<< "$raw"
    STAGE_LIST=()
    for token in "${tokens[@]}"; do
        token="$(echo "$token" | xargs)"
        [[ -z "$token" ]] && continue
        [[ "$token" =~ ^[1-4]$ ]] || { error "非法阶段: $token"; exit 1; }
        STAGE_LIST+=("$token")
    done
}

init_summary_csv() {
    mkdir -p "$SUMMARY_DIR"
    SUMMARY_CSV="$SUMMARY_DIR/test_summary.csv"
    cat > "$SUMMARY_CSV" <<'EOF'
stage,repeat,num_episodes,total_success,total_collision,avg_reward,avg_steps,avg_min_dist,avg_social_risk,avg_front_risk,reached_agents,collided_agents,truncated_episodes,exit_code,log_path,checkpoint,fixed_benchmark_set,benchmark_csv
EOF
}

append_summary_row() {
    local stage="$1"
    local repeat="$2"
    local exit_code="$3"
    local log_path="$4"

    python3 - "$SUMMARY_CSV" "$stage" "$repeat" "$NUM_EPISODES" "$exit_code" "$log_path" "$CHECKPOINT" "$FIXED_BENCHMARK_SET" "$BENCHMARK_CSV" <<'PY'
import csv, re, sys

summary_csv, stage, repeat, num_episodes, exit_code, log_path, checkpoint, fixed_benchmark_set, benchmark_csv = sys.argv[1:]

try:
    text = open(log_path, "r", encoding="utf-8", errors="ignore").read()
except Exception:
    text = ""

patterns = {
    "total_success": r"总计到达:\s*([0-9]+)",
    "total_collision": r"总计碰撞:\s*([0-9]+)",
    "avg_reward": r"平均回报:\s*([-0-9.]+)",
    "avg_steps": r"平均步数:\s*([-0-9.]+)",
    "avg_min_dist": r"平均最小间距:\s*([-0-9.]+)",
    "avg_social_risk": r"平均社交风险:\s*([-0-9.]+)",
    "avg_front_risk": r"平均前向风险:\s*([-0-9.]+)",
    "reached_agents": r"到达过目标的 agent 数:\s*([0-9]+)",
    "collided_agents": r"发生过碰撞的 agent 数:\s*([0-9]+)",
    "truncated_episodes": r"被时间截断的 Episode:\s*([0-9]+)",
}

row = {
    "stage": stage,
    "repeat": repeat,
    "num_episodes": num_episodes,
    "exit_code": exit_code,
    "log_path": log_path,
    "checkpoint": checkpoint,
    "fixed_benchmark_set": fixed_benchmark_set,
    "benchmark_csv": benchmark_csv,
}
for key, pattern in patterns.items():
    m = re.search(pattern, text)
    row[key] = m.group(1) if m else ""

with open(summary_csv, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "stage", "repeat", "num_episodes", "total_success", "total_collision",
        "avg_reward", "avg_steps", "avg_min_dist", "avg_social_risk",
        "avg_front_risk", "reached_agents", "collided_agents",
        "truncated_episodes", "exit_code", "log_path", "checkpoint",
        "fixed_benchmark_set", "benchmark_csv"
    ])
    writer.writerow(row)
PY
}

cleanup() {
    echo ""
    warn "测试结束/收到中断信号，正在清理环境..."
    [[ -n "$TEST_PID" ]] && kill "$TEST_PID" 2>/dev/null || true
    if [[ -n "$RVIZ_GIF_PID" ]] && kill -0 "$RVIZ_GIF_PID" 2>/dev/null; then
        kill -INT "$RVIZ_GIF_PID" 2>/dev/null || true
        wait "$RVIZ_GIF_PID" 2>/dev/null || true
    fi
    KILL_ALL_ROS_SCOPE=port_only GAZEBO_PORT="$TEST_GAZEBO_PORT" GAZEBO_MASTER_URI="$GAZEBO_MASTER_URI" \
        bash "$KILL_SCRIPT" 2>/dev/null || true
    info "已清理，退出。"
    exit 130
}
trap cleanup SIGINT SIGTERM
trap stop_ros_env EXIT

check_env() {
    banner "═══ 测试环境检查 ═══"
    [[ -f "$TEST_SCRIPT" ]] || { error "测试脚本不存在: $TEST_SCRIPT"; exit 1; }
    [[ -f "$KILL_SCRIPT" ]] || { error "kill 脚本不存在: $KILL_SCRIPT"; exit 1; }
    [[ -e "$CHECKPOINT" ]] || { error "Checkpoint 路径不存在: $CHECKPOINT"; exit 1; }

    CHECKPOINT="$(normalize_ckpt_path "$CHECKPOINT")"
    [[ -e "$CHECKPOINT" ]] || { error "未找到可用 checkpoint: $CHECKPOINT"; exit 1; }

    if [[ -z "$NUM_AGENTS" ]]; then
        NUM_AGENTS="$(infer_checkpoint_num_agents "$CHECKPOINT")"
        if [[ -z "$NUM_AGENTS" ]]; then
            warn "未能从 checkpoint 自动推断 num_agents，回退为 2"
            NUM_AGENTS=2
        else
            info "从 checkpoint 自动推断 num_agents=${NUM_AGENTS}"
        fi
    fi

    [[ "$NUM_EPISODES" =~ ^[0-9]+$ ]] || { error "--num_episodes 必须是非负整数"; exit 1; }
    [[ "$REPEAT_RUNS" =~ ^[0-9]+$ ]] || { error "--repeat_runs 必须是非负整数"; exit 1; }
    [[ "$DIAG_STEPS" =~ ^[0-9]+$ ]] || { error "--diag_steps 必须是非负整数"; exit 1; }
    [[ "$TEST_ROS_DOMAIN_ID" =~ ^[0-9]+$ ]] || { error "--ros_domain_id 必须是非负整数"; exit 1; }
    [[ "$TEST_GAZEBO_PORT" =~ ^[0-9]+$ ]] || { error "--gazebo_port 必须是非负整数"; exit 1; }
    [[ "$RVIZ_GIF_FPS" =~ ^[0-9]+$ ]] || { error "--rviz_gif_fps 必须是非负整数"; exit 1; }
    [[ "$RVIZ_GIF_DURATION" =~ ^[0-9]+$ ]] || { error "--rviz_gif_duration 必须是非负整数"; exit 1; }

    case "$RENDER_MODE" in
        headless)
            HEADLESS_SIM=1
            ENABLE_RVIZ=0
            ;;
        rviz)
            HEADLESS_SIM=1
            ENABLE_RVIZ=1
            ;;
        full_gui)
            HEADLESS_SIM=0
            ENABLE_RVIZ=1
            ;;
        *)
            error "--render_mode 只支持 headless | rviz | full_gui"
            exit 1
            ;;
    esac

    if (( RUN_ALL_STAGES == 0 )) && [[ -z "$TEST_STAGES" ]] && (( TEST_STAGE_EXPLICIT == 0 )); then
        local inferred_stage=""
        inferred_stage="$(infer_checkpoint_stage "$CHECKPOINT")"
        if [[ "$inferred_stage" =~ ^[1-4]$ ]]; then
            TEST_STAGE="$inferred_stage"
            info "未显式指定测试阶段，已从 checkpoint 路径推断 TEST_STAGE=${TEST_STAGE}"
        else
            warn "未能从 checkpoint 路径推断阶段，继续使用默认 TEST_STAGE=${TEST_STAGE}"
        fi
    fi

    if (( RUN_ALL_STAGES == 1 )); then
        STAGE_LIST=(1 2 3 4)
    elif [[ -n "$TEST_STAGES" ]]; then
        parse_stage_list "$TEST_STAGES"
    else
        [[ "$TEST_STAGE" =~ ^[1-4]$ ]] || { error "--test_stage 必须是 1~4"; exit 1; }
        STAGE_LIST=("$TEST_STAGE")
    fi

    if [[ -z "$SUMMARY_DIR" ]]; then
        SUMMARY_DIR="$(build_default_summary_dir)"
    fi
    if [[ -z "$RVIZ_GIF_DIR" ]]; then
        RVIZ_GIF_DIR="$SUMMARY_DIR/rviz_gifs"
    fi
    RUN_TAG="$(basename "$SUMMARY_DIR")"
    init_summary_csv

    if (( SAVE_RVIZ_GIF == 1 )) && (( ENABLE_RVIZ == 0 )); then
        warn "当前 render_mode=$RENDER_MODE 不会启动 RViz，自动关闭 GIF 录制。"
        SAVE_RVIZ_GIF=0
    fi

    success "环境检查通过"
    info "  仿真环境机器人数量: $NUM_AGENTS"
    info "  测试回合:           $NUM_EPISODES"
    info "  重复次数:           $REPEAT_RUNS"
    info "  诊断步数:           $DIAG_STEPS"
    info "  探索模式:           $( ((EXPLORE==1)) && echo "开启" || echo "关闭" )"
    if (( SHIELD_OVERRIDE_SET == 1 )); then
        info "  测试Shield:         显式覆盖为 ${SHIELD_ENABLE}"
    else
        info "  测试Shield:         跟随 checkpoint 配置"
    fi
    info "  测试阶段:           ${STAGE_LIST[*]}"
    for stage in "${STAGE_LIST[@]}"; do
        local display_map="${STAGE_MAP_NUM[$stage]}"
        if [[ -n "$MAP_NUMBER_OVERRIDE" ]]; then
            display_map="$MAP_NUMBER_OVERRIDE"
        fi
        info "    Stage ${stage} 配置: map=${display_map} obs=${STAGE_OBS_NUM[$stage]} obs_speed_scale=${STAGE_OBS_SPD[$stage]}"
    done
    info "  ROS_DOMAIN_ID:      $TEST_ROS_DOMAIN_ID"
    info "  GAZEBO_MASTER_URI:  $GAZEBO_MASTER_URI"
    info "  启动模式:           $RENDER_MODE"
    info "  RViz GIF录制:       $( ((SAVE_RVIZ_GIF==1)) && echo "开启" || echo "关闭" )"
    if (( SAVE_RVIZ_GIF == 1 )); then
        info "  GIF参数:            fps=$RVIZ_GIF_FPS scale=$RVIZ_GIF_SCALE duration=${RVIZ_GIF_DURATION}s(0=全程)"
        info "  GIF目录:            $RVIZ_GIF_DIR"
    fi
    info "  权重路径:           $CHECKPOINT"
    info "  汇总目录:           $SUMMARY_DIR"
    echo ""
}

stop_ros_env() {
    info "正在停止旧的 ROS2/Gazebo 进程..."
    if [[ -n "$RVIZ_GIF_PID" ]] && kill -0 "$RVIZ_GIF_PID" 2>/dev/null; then
        info "停止 RViz GIF 录制..."
        kill -INT "$RVIZ_GIF_PID" 2>/dev/null || true
        wait "$RVIZ_GIF_PID" 2>/dev/null || true
    fi
    RVIZ_GIF_PID=""
    RVIZ_GIF_PATH=""
    KILL_ALL_ROS_SCOPE=port_only GAZEBO_PORT="$TEST_GAZEBO_PORT" GAZEBO_MASTER_URI="$GAZEBO_MASTER_URI" \
        bash "$KILL_SCRIPT" 2>/dev/null || true
    pkill -f "gzclient|gzserver|rviz2.*${RUN_TAG}|rviz2-17|main_headless.launch.py|main.launch.py" 2>/dev/null || true
    sleep 2
}

start_rviz_gif_capture() {
    local stage="$1"
    local repeat="$2"

    if (( SAVE_RVIZ_GIF != 1 )); then
        return 0
    fi
    if (( ENABLE_RVIZ != 1 )); then
        return 0
    fi
    if ! command -v ffmpeg >/dev/null 2>&1; then
        warn "未找到 ffmpeg，跳过 RViz GIF 录制。"
        return 0
    fi
    if [[ -z "${DISPLAY:-}" ]]; then
        warn "DISPLAY 未设置，跳过 RViz GIF 录制。"
        return 0
    fi

    mkdir -p "$RVIZ_GIF_DIR"
    RVIZ_GIF_PATH="$RVIZ_GIF_DIR/rviz_stage${stage}_run${repeat}.gif"

    local -a cmd=(
        ffmpeg
        -y
        -loglevel error
        -f x11grab
        -framerate "$RVIZ_GIF_FPS"
        -i "$DISPLAY"
        -vf "fps=$RVIZ_GIF_FPS,scale=$RVIZ_GIF_SCALE:flags=lanczos"
    )
    if (( RVIZ_GIF_DURATION > 0 )); then
        cmd+=( -t "$RVIZ_GIF_DURATION" )
    fi
    cmd+=( "$RVIZ_GIF_PATH" )

    info "开始录制 RViz GIF: $RVIZ_GIF_PATH"
    "${cmd[@]}" >/dev/null 2>&1 &
    RVIZ_GIF_PID=$!
}

start_ros_env() {
    local stage="$1"
    local repeat="$2"
    local map_num=${STAGE_MAP_NUM[$stage]}
    if [[ -n "$MAP_NUMBER_OVERRIDE" ]]; then
        map_num="$MAP_NUMBER_OVERRIDE"
    fi
    local obs_num=${STAGE_OBS_NUM[$stage]}
    local obs_spd=${STAGE_OBS_SPD[$stage]}
    local launch_file="main.launch.py"
    if (( HEADLESS_SIM == 1 )); then
        launch_file="main_headless.launch.py"
    fi

    local log="$SUMMARY_DIR/ros_stage${stage}.log"
    TEST_ENV_LOG="$log"
    mkdir -p "$SUMMARY_DIR"

    banner "  启动测试仿真环境 (Stage $stage · ${STAGE_NAME[$stage]})"

    local inner_cmd
    inner_cmd="set +u; [[ -f '${CONDA_SH}' ]] && source '${CONDA_SH}'"
    inner_cmd+="; command -v conda >/dev/null 2>&1 && conda activate '${ROS2_CONDA_ENV}' >/dev/null 2>&1 || true"
    inner_cmd+="; source '${ROS_SETUP}'; [[ -f '${WS_SETUP}' ]] && source '${WS_SETUP}'"
    inner_cmd+="; __sanitize_ament_prefix_path() {"
    inner_cmd+=" local original=\"\${AMENT_PREFIX_PATH:-}\";"
    inner_cmd+=" [[ -z \"\$original\" ]] && return 0;"
    inner_cmd+=" local filtered=(); local prefix expected_pkg marker;"
    inner_cmd+=" IFS=':' read -r -a prefixes <<< \"\$original\";"
    inner_cmd+=" for prefix in \"\${prefixes[@]}\"; do"
    inner_cmd+=" [[ -z \"\$prefix\" ]] && continue;"
    inner_cmd+=" if [[ \"\$prefix\" == /opt/ros/* ]]; then filtered+=(\"\$prefix\"); continue; fi;"
    inner_cmd+=" expected_pkg=\"\$(basename \"\$prefix\")\";"
    inner_cmd+=" marker=\"\$prefix/share/ament_index/resource_index/packages/\$expected_pkg\";"
    inner_cmd+=" [[ -f \"\$marker\" ]] && filtered+=(\"\$prefix\");"
    inner_cmd+=" done;"
    inner_cmd+=" if (( \${#filtered[@]} > 0 )); then export AMENT_PREFIX_PATH=\"\$(IFS=:; echo \"\${filtered[*]}\")\"; fi;"
    inner_cmd+=" }; __sanitize_ament_prefix_path"
    inner_cmd+="; unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY"
    inner_cmd+="; export no_proxy=localhost,127.0.0.1; export GAZEBO_MODEL_DATABASE_URI=''"
    inner_cmd+="; export ROS_DOMAIN_ID='${TEST_ROS_DOMAIN_ID}'"
    inner_cmd+="; export GAZEBO_MASTER_URI='${GAZEBO_MASTER_URI}'"
    inner_cmd+="; set -u; echo '=== 启动测试环境 ==='"
    inner_cmd+="; ros2 launch start_rl_environment_tb3 ${launch_file}"
    inner_cmd+=" map_number:=${map_num} robot_number:=${NUM_AGENTS}"
    inner_cmd+=" num_obstacles:=${obs_num} obs_speed_scale:=${obs_spd}"
    if (( HEADLESS_SIM == 1 )); then
        inner_cmd+=" enable_rviz:=$([[ $ENABLE_RVIZ -eq 1 ]] && echo true || echo false)"
        if (( ENABLE_RVIZ == 1 )); then
            inner_cmd+=" rviz_node_name:='rviz2_test_${RUN_TAG}_s${stage}'"
        fi
    fi
    inner_cmd+=" 2>&1 | tee '${log}'"

    bash -c "${inner_cmd}" &
    ROS_PID=$!

    timeout 5s ros2 daemon stop >/dev/null 2>&1 || true
    timeout 5s ros2 daemon start >/dev/null 2>&1 || true
    sleep 1

    info "等待 Gazebo 就绪（最多 ${GAZEBO_WAIT_SEC}s）..."
    local waited=0
    local topics=""
    while [[ $waited -lt $GAZEBO_WAIT_SEC ]]; do
        if [[ -f "$log" ]] && grep -q "Unable to start server\\[bind: Address already in use\\]" "$log"; then
            echo ""
            error "Gazebo 端口冲突：${GAZEBO_MASTER_URI} 已被占用。"
            return 1
        fi
        topics="$(timeout 3s bash -lc 'export ROS_DOMAIN_ID='"${TEST_ROS_DOMAIN_ID}"'; export GAZEBO_MASTER_URI='"${GAZEBO_MASTER_URI}"'; ROS2CLI_NODE_STRATEGY=direct ros2 topic list 2>/dev/null' || true)"
        if echo "$topics" | grep -Eq "/tb3_0/(scan|odom)"; then
            echo ""
            success "Gazebo 就绪！(${waited}s)"
            sleep ${GAZEBO_GRACE_SEC}
            start_rviz_gif_capture "$stage" "$repeat"
            return 0
        fi
        if [[ -f "$log" ]] && grep -q "Successfully spawned entity \[tb3_0\]" "$log"; then
            echo ""
            success "Gazebo 已完成机器人生成（日志检测，${waited}s）"
            sleep ${GAZEBO_GRACE_SEC}
            start_rviz_gif_capture "$stage" "$repeat"
            return 0
        fi
        if [[ -n "$ROS_PID" ]] && ! kill -0 "$ROS_PID" 2>/dev/null; then
            echo ""
            error "测试环境启动进程已提前退出。"
            [[ -f "$log" ]] && { warn "最近启动日志:"; tail -n 40 "$log"; }
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
        printf "\r  等待中... %ds / %ds" "$waited" "$GAZEBO_WAIT_SEC"
    done

    echo ""
    error "等待超时，Gazebo 未就绪。"
    [[ -f "$log" ]] && { warn "最近启动日志:"; tail -n 40 "$log"; }
    return 1
}

run_test() {
    local stage="$1"
    local repeat="$2"
    local map_num=${STAGE_MAP_NUM[$stage]}
    if [[ -n "$MAP_NUMBER_OVERRIDE" ]]; then
        map_num="$MAP_NUMBER_OVERRIDE"
    fi
    local obs_num=${STAGE_OBS_NUM[$stage]}
    local obs_spd=${STAGE_OBS_SPD[$stage]}
    local log_path="$SUMMARY_DIR/test_stage${stage}_run${repeat}.log"

    banner "═══ 开始推理测试 (Stage ${stage} / Run ${repeat}) ═══"

    local cmd=(
        python3 "$TEST_SCRIPT"
        --checkpoint_path "$CHECKPOINT"
        --num_episodes "$NUM_EPISODES"
        --test_max_episode_steps "$TEST_MAX_EPISODE_STEPS"
        --diag_steps "$DIAG_STEPS"
        --map_number "$map_num"
        --num_dynamic_obstacles "$obs_num"
        --obs_speed_scale "$obs_spd"
        --rolling_lookahead_dist "$ROLLING_LOOKAHEAD_DIST"
    )
    if (( SHIELD_OVERRIDE_SET == 1 )); then
        cmd+=(--shield_enable "$SHIELD_ENABLE")
        cmd+=(--shield_front_slow_dist "$SHIELD_FRONT_SLOW_DIST")
        cmd+=(--shield_front_stop_dist "$SHIELD_FRONT_STOP_DIST")
        cmd+=(--shield_neighbor_slow_dist "$SHIELD_NEIGHBOR_SLOW_DIST")
        cmd+=(--shield_linear_slow "$SHIELD_LINEAR_SLOW")
        cmd+=(--shield_linear_stop "$SHIELD_LINEAR_STOP")
        cmd+=(--shield_turn_bias "$SHIELD_TURN_BIAS")
        cmd+=(--turn_in_place_front_dist "$TURN_IN_PLACE_FRONT_DIST")
        cmd+=(--turn_in_place_angle_thresh "$TURN_IN_PLACE_ANGLE_THRESH")
        cmd+=(--turn_in_place_w "$TURN_IN_PLACE_W")
    fi
    if (( EXPLORE == 1 )); then
        cmd+=(--explore)
    fi
    if [[ -n "$FIXED_BENCHMARK_SET" ]]; then
        BENCHMARK_CSV="$SUMMARY_DIR/benchmark_${FIXED_BENCHMARK_SET}_stage${stage}_run${repeat}.csv"
        cmd+=(--fixed_benchmark_set "$FIXED_BENCHMARK_SET")
        cmd+=(--benchmark_csv "$BENCHMARK_CSV")
    else
        BENCHMARK_CSV=""
    fi

    info "测试命令: ${cmd[*]}"
    info "测试日志: $log_path"
    echo ""

    set +e
    "${cmd[@]}" 2>&1 | tee "$log_path"
    local exit_code=${PIPESTATUS[0]}
    set -e

    append_summary_row "$stage" "$repeat" "$exit_code" "$log_path"

    if [[ $exit_code -ne 0 ]]; then
        error "测试异常退出 (code=$exit_code)"
    else
        success "测试顺利完成！"
    fi
}

main() {
    bootstrap_python_env
    export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"
    export ROS_DOMAIN_ID="$TEST_ROS_DOMAIN_ID"
    export GAZEBO_MASTER_URI="http://127.0.0.1:${TEST_GAZEBO_PORT}"
    set +u
    [[ -f "$ROS_SETUP" ]] && source "$ROS_SETUP" || { set -u; error "ROS2 Humble 未找到"; exit 1; }
    [[ -f "$WS_SETUP"  ]] && source "$WS_SETUP"
    set -u

    check_env

    local stage repeat
    for stage in "${STAGE_LIST[@]}"; do
        for repeat in $(seq 1 "$REPEAT_RUNS"); do
            stop_ros_env
            if start_ros_env "$stage" "$repeat"; then
                run_test "$stage" "$repeat"
            else
                error "仿真环境启动失败，中止当前测试。"
                append_summary_row "$stage" "$repeat" "125" "$SUMMARY_DIR/ros_stage${stage}.log"
            fi
            stop_ros_env
        done
    done

    info "测试汇总: $SUMMARY_CSV"
    banner "═══ 测试脚本执行完毕 ═══"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--checkpoint)  CHECKPOINT="$2"; shift 2 ;;
        --num_agents)     NUM_AGENTS="$2"; shift 2 ;;
        --num_episodes)   NUM_EPISODES="$2"; shift 2 ;;
        --test_max_episode_steps) TEST_MAX_EPISODE_STEPS="$2"; shift 2 ;;
        --test_stage)     TEST_STAGE="$2"; TEST_STAGE_EXPLICIT=1; shift 2 ;;
        --fixed_benchmark_set) FIXED_BENCHMARK_SET="$2"; shift 2 ;;
        --rolling_lookahead_dist) ROLLING_LOOKAHEAD_DIST="$2"; shift 2 ;;
        --map_number)     MAP_NUMBER_OVERRIDE="$2"; shift 2 ;;
        --num_stage)      TEST_STAGE="$2"; TEST_STAGE_EXPLICIT=1; shift 2 ;;
        --test_stages)    TEST_STAGES="$2"; shift 2 ;;
        --all_stages)     RUN_ALL_STAGES=1; shift 1 ;;
        --repeat_runs)    REPEAT_RUNS="$2"; shift 2 ;;
        --diag_steps)     DIAG_STEPS="$2"; shift 2 ;;
        --shield_enable)  SHIELD_ENABLE="$2"; SHIELD_OVERRIDE_SET=1; shift 2 ;;
        --shield_front_slow_dist) SHIELD_FRONT_SLOW_DIST="$2"; SHIELD_OVERRIDE_SET=1; shift 2 ;;
        --shield_front_stop_dist) SHIELD_FRONT_STOP_DIST="$2"; SHIELD_OVERRIDE_SET=1; shift 2 ;;
        --shield_neighbor_slow_dist) SHIELD_NEIGHBOR_SLOW_DIST="$2"; SHIELD_OVERRIDE_SET=1; shift 2 ;;
        --shield_linear_slow) SHIELD_LINEAR_SLOW="$2"; SHIELD_OVERRIDE_SET=1; shift 2 ;;
        --shield_linear_stop) SHIELD_LINEAR_STOP="$2"; SHIELD_OVERRIDE_SET=1; shift 2 ;;
        --shield_turn_bias) SHIELD_TURN_BIAS="$2"; SHIELD_OVERRIDE_SET=1; shift 2 ;;
        --turn_in_place_front_dist) TURN_IN_PLACE_FRONT_DIST="$2"; SHIELD_OVERRIDE_SET=1; shift 2 ;;
        --turn_in_place_angle_thresh) TURN_IN_PLACE_ANGLE_THRESH="$2"; SHIELD_OVERRIDE_SET=1; shift 2 ;;
        --turn_in_place_w) TURN_IN_PLACE_W="$2"; SHIELD_OVERRIDE_SET=1; shift 2 ;;
        --ros_domain_id)  TEST_ROS_DOMAIN_ID="$2"; shift 2 ;;
        --gazebo_port)    TEST_GAZEBO_PORT="$2"; shift 2 ;;
        --render_mode)    RENDER_MODE="$2"; shift 2 ;;
        --headless_sim)   RENDER_MODE="headless"; shift 1 ;;
        --gui_sim)        RENDER_MODE="full_gui"; shift 1 ;;
        --save_rviz_gif)  SAVE_RVIZ_GIF=1; shift 1 ;;
        --rviz_gif_fps)   RVIZ_GIF_FPS="$2"; shift 2 ;;
        --rviz_gif_scale) RVIZ_GIF_SCALE="$2"; shift 2 ;;
        --rviz_gif_duration) RVIZ_GIF_DURATION="$2"; shift 2 ;;
        --rviz_gif_dir)   RVIZ_GIF_DIR="$2"; shift 2 ;;
        --summary_dir)    SUMMARY_DIR="$2"; shift 2 ;;
        -e|--explore)     EXPLORE=1; shift 1 ;;
        -h|--help)
            echo "用法: ./run_test.sh -c <checkpoint_path> [OPTIONS]"
            echo "  -c, --checkpoint     模型权重路径 (必需)"
            echo "  --num_agents         启动 Gazebo 时的机器人数量 (默认从 checkpoint 推断)"
            echo "  --num_episodes       每轮测试回合数 (默认5)"
            echo "  --test_max_episode_steps  单回合最大步数 (默认2500)"
            echo "  --test_stage         单阶段测试，1~4 (默认4)"
            echo "  --fixed_benchmark_set  固定基准场景集合，例如 fixed50_v1"
            echo "  --rolling_lookahead_dist 覆盖测试前瞻距离 (默认0.4)"
            echo "  --map_number         覆盖 Stage preset 的地图编号，同时用于 Gazebo 和测试 env"
            echo "  --num_stage          --test_stage 的别名"
            echo "  --test_stages        多阶段测试列表，例如 1,2,3,4"
            echo "  --all_stages         依次测试 Stage 1~4"
            echo "  --repeat_runs        每个阶段重复测试次数 (默认1)"
            echo "  --diag_steps         前 N 步打印诊断信息 (默认15, 设为0关闭)"
            echo "  --shield_enable      显式覆盖 checkpoint 中的 shield 开关"
            echo "  --shield_front_slow_dist     显式覆盖 checkpoint 中的 shield 参数"
            echo "  --shield_front_stop_dist     显式覆盖 checkpoint 中的 shield 参数"
            echo "  --shield_neighbor_slow_dist  显式覆盖 checkpoint 中的 shield 参数"
            echo "  --shield_linear_slow         显式覆盖 checkpoint 中的 shield 参数"
            echo "  --shield_linear_stop         显式覆盖 checkpoint 中的 shield 参数"
            echo "  --shield_turn_bias           显式覆盖 checkpoint 中的 shield 参数"
            echo "  --turn_in_place_front_dist   显式覆盖 checkpoint 中的 shield 参数"
            echo "  --turn_in_place_angle_thresh 显式覆盖 checkpoint 中的 shield 参数"
            echo "  --turn_in_place_w            显式覆盖 checkpoint 中的 shield 参数"
            echo "  --ros_domain_id      测试专用 ROS_DOMAIN_ID (默认71)"
            echo "  --gazebo_port        测试专用 Gazebo 端口 (默认11845)"
            echo "  --render_mode        headless | rviz | full_gui (默认full_gui)"
            echo "  --headless_sim       使用 main_headless.launch.py 并关闭 RViz"
            echo "  --gui_sim            兼容旧参数，等价于 --render_mode full_gui"
            echo "  --save_rviz_gif      保存 RViz 画面为 GIF（需 render_mode=rviz/full_gui）"
            echo "  --rviz_gif_fps       GIF 帧率 (默认10)"
            echo "  --rviz_gif_scale     GIF 尺寸，如 960:-1 (默认960:-1)"
            echo "  --rviz_gif_duration  GIF 最长秒数，0表示覆盖整个测试过程 (默认0)"
            echo "  --rviz_gif_dir       GIF 输出目录 (默认 summary_dir/rviz_gifs)"
            echo "  --summary_dir        汇总目录"
            echo "  -e, --explore        开启探索噪声"
            exit 0
            ;;
        *) error "未知参数: $1"; exit 1 ;;
    esac
done

if [[ -z "$CHECKPOINT" ]]; then
    error "必须提供 Checkpoint 路径！例如: ./run_test.sh -c /path/to/checkpoint"
    exit 1
fi
CHECKPOINT="$(readlink -f "$CHECKPOINT" 2>/dev/null || python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$CHECKPOINT")"

main
