#!/bin/bash
# =============================================================================
# Option 原子能力一键测试脚本
# 自动：按场景启动 Gazebo → 运行 option_tester → 收日志 → 清理环境
#
# 示例：
#   ./run_option_test.sh --scenario head_on_two_agents --option backoff
#   ./run_option_test.sh --scenario l_corner --option detour_right --gui
#   ./run_option_test.sh --all                                    # 测试所有场景×所有option
#   ./run_option_test.sh --all --gui --num-episodes 5             # GUI 下全量测试
#   ./run_option_test.sh --all --scenarios narrow_corridor,...    # 只测指定场景
#   ./run_option_test.sh --all --options backoff,detour_left      # 只测指定 option
#   ./run_option_test.sh --list-scenarios
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

# ─── 默认参数 ────────────────────────────────────────────────────────────────────
SCENARIO="single_follow"
OPTION="follow_path"
BACKGROUND_OPTION="follow_path"
NUM_EPISODES=3
MAX_OPTION_STEPS=14
MAX_EPISODE_STEPS=40
OUTPUT_DIR="option_test_results"
SEED=""
RESPECT_ACTION_MASK=0
DISABLE_SAFETY_OVERRIDES=0
DISABLE_REPLAN=0
ROLLING_LOOKAHEAD_DIST=0.8
OBSTACLE_FILTER_RANGE=1.5
OBSTACLE_TOP_K=9
LIST_SCENARIOS=0
RUN_ALL=0
ALL_SCENARIOS_FILTER=""
ALL_OPTIONS_FILTER=""
HEADLESS_SIM=1
TEST_ROS_DOMAIN_ID=72
TEST_GAZEBO_PORT=11846

WORKSPACE="/home/wj/work/multi-robot-exploration-rl"
GAZEBO_WAIT_SEC=60
GAZEBO_GRACE_SEC=5
CONDA_SH="${CONDA_SH:-/home/wj/anaconda3/etc/profile.d/conda.sh}"
ROS2_CONDA_ENV="${ROS2_CONDA_ENV:-ros2}"
REQUIRED_PY_VER="3.10"

TESTER_SCRIPT="$WORKSPACE/src/gnn_marl_training/gnn_marl_training/option_tester.py"
SCENARIO_SCRIPT="$WORKSPACE/src/gnn_marl_training/gnn_marl_training/option_test_scenarios.py"
KILL_SCRIPT="$WORKSPACE/kill_all_ros.sh"
ROS_SETUP="/opt/ros/humble/setup.bash"
WS_SETUP="${WORKSPACE}/install/setup.bash"

ROS_PID=""
TEST_PID=""

# 全量测试时收集每个 combo 的 summary 路径
declare -a ALL_SUMMARY_PATHS=()

# ─── Python 环境 ─────────────────────────────────────────────────────────────────
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

# ─── 从 Python 查询场景 / option 元数据 ──────────────────────────────────────────
query_scenario_meta() {
    local name="$1"
    python3 - "$name" "$SCENARIO_SCRIPT" <<'PY'
import sys
from pathlib import Path

name = sys.argv[1]
scenario_path = Path(sys.argv[2])

sys.path.insert(0, str(scenario_path.parents[1]))
from gnn_marl_training.option_test_scenarios import get_scenario

spec = get_scenario(name)
print(f"{spec.map_number}\t{spec.num_agents}\t{spec.description}")
PY
}

query_all_scenarios() {
    python3 - "$SCENARIO_SCRIPT" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).parents[1]))
from gnn_marl_training.option_test_scenarios import list_scenarios
for name in list_scenarios():
    print(name)
PY
}

query_all_options() {
    python3 - <<'PY'
from gnn_marl_training.option_feasibility import OPTION_NAMES
for name in OPTION_NAMES:
    print(name)
PY
}

# ─── 清理 ────────────────────────────────────────────────────────────────────────
cleanup() {
    echo ""
    warn "测试结束/收到中断信号，正在清理环境..."
    [[ -n "$TEST_PID" ]] && kill "$TEST_PID" 2>/dev/null || true
    KILL_ALL_ROS_SCOPE=port_only GAZEBO_PORT="$TEST_GAZEBO_PORT" GAZEBO_MASTER_URI="$GAZEBO_MASTER_URI" \
        bash "$KILL_SCRIPT" 2>/dev/null || true
    info "已清理，退出。"
    exit 130
}
trap cleanup SIGINT SIGTERM

# ─── 环境检查（单场景模式）──────────────────────────────────────────────────────
check_single_env() {
    banner "═══ Option 测试环境检查 ═══"
    [[ -f "$TESTER_SCRIPT" ]]  || { error "测试脚本不存在: $TESTER_SCRIPT"; exit 1; }
    [[ -f "$KILL_SCRIPT" ]]    || { error "kill 脚本不存在: $KILL_SCRIPT";   exit 1; }

    if (( LIST_SCENARIOS == 1 )); then
        print_scenario_list
        exit 0
    fi

    local meta
    meta="$(query_scenario_meta "$SCENARIO")" || {
        error "无法解析场景: $SCENARIO"
        exit 1
    }
    IFS=$'\t' read -r RUN_MAP_NUM RUN_NUM_AGENTS RUN_DESC <<< "$meta"

    [[ "$NUM_EPISODES"       =~ ^[0-9]+$ ]] || { error "--num-episodes 必须是非负整数"; exit 1; }
    [[ "$MAX_OPTION_STEPS"   =~ ^[0-9]+$ ]] || { error "--max-option-steps 必须是非负整数"; exit 1; }
    [[ "$MAX_EPISODE_STEPS"  =~ ^[0-9]+$ ]] || { error "--max-episode-steps 必须是非负整数"; exit 1; }
    [[ "$TEST_ROS_DOMAIN_ID" =~ ^[0-9]+$ ]] || { error "--ros_domain_id 必须是非负整数"; exit 1; }
    [[ "$TEST_GAZEBO_PORT"   =~ ^[0-9]+$ ]] || { error "--gazebo_port 必须是非负整数"; exit 1; }

    OUTPUT_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$OUTPUT_DIR")"

    success "环境检查通过"
    info "  场景:               $SCENARIO — $RUN_DESC"
    info "  Option:             $OPTION"
    info "  背景 Option:        $BACKGROUND_OPTION"
    info "  地图编号:           $RUN_MAP_NUM"
    info "  机器人数量:         $RUN_NUM_AGENTS"
    info "  测试次数:           $NUM_EPISODES"
    info "  Option 最大步数:    $MAX_OPTION_STEPS"
    info "  Episode 最大步数:   $MAX_EPISODE_STEPS"
    info "  输出目录:           $OUTPUT_DIR"
    if (( RESPECT_ACTION_MASK == 1 )); then
        info "  Action Mask:        仅可行时执行"
    fi
    if (( DISABLE_SAFETY_OVERRIDES == 1 )); then
        info "  Safety Overrides:   关闭"
    fi
    if (( DISABLE_REPLAN == 1 )); then
        info "  Replan:             关闭"
    fi
    info "  启动模式:           $( ((HEADLESS_SIM==1)) && echo "headless" || echo "headless + RViz" )"
    info "  ROS_DOMAIN_ID:      $TEST_ROS_DOMAIN_ID"
    info "  GAZEBO_MASTER_URI:  $GAZEBO_MASTER_URI"
    echo ""
}

# ─── 列出场景 ────────────────────────────────────────────────────────────────────
print_scenario_list() {
    python3 - "$SCENARIO_SCRIPT" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).parents[1]))
from gnn_marl_training.option_test_scenarios import list_scenarios, get_scenario

for name in list_scenarios():
    spec = get_scenario(name)
    print(f"  {name:30s}  map={spec.map_number}  agents={spec.num_agents}  {spec.description}")
PY
}

# ─── 停止 ROS 环境 ───────────────────────────────────────────────────────────────
stop_ros_env() {
    info "正在停止旧的 ROS2/Gazebo 进程..."
    KILL_ALL_ROS_SCOPE=port_only GAZEBO_PORT="$TEST_GAZEBO_PORT" GAZEBO_MASTER_URI="$GAZEBO_MASTER_URI" \
        bash "$KILL_SCRIPT" 2>/dev/null || true
    sleep 2
}

# ─── 启动 Gazebo 仿真 ───────────────────────────────────────────────────────────
# 始终使用 main_headless.launch.py（ExecuteProcess 直接起 gzserver，env 传递可靠）
# GUI 模式通过 enable_rviz:=true 打开 RViz 可视化
start_ros_env() {
    local map_num="$1"
    local num_agents="$2"
    local enable_rviz="false"
    local rviz_name="rviz2_opt_test"
    if (( HEADLESS_SIM == 0 )); then
        enable_rviz="true"
        rviz_name="rviz2_opt_test_$(date +%H%M%S)"
    fi

    banner "  启动仿真环境 (map=${map_num}, robots=${num_agents}, rviz=${enable_rviz})"

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
    inner_cmd+="; export no_proxy=localhost,127.0.0.1"
    inner_cmd+="; export TURTLEBOT3_MODEL=\${TURTLEBOT3_MODEL:-burger}"
    inner_cmd+="; export ROS_DOMAIN_ID='${TEST_ROS_DOMAIN_ID}'"
    inner_cmd+="; export GAZEBO_MASTER_URI='${GAZEBO_MASTER_URI}'"
    inner_cmd+="; set -u; echo '=== 启动 Option 测试仿真环境 ==='"
    inner_cmd+="; ros2 launch start_rl_environment_tb3 main_headless.launch.py"
    inner_cmd+=" map_number:=${map_num} robot_number:=${num_agents}"
    inner_cmd+=" num_obstacles:=0 obs_speed_scale:=0.0"
    inner_cmd+=" enable_rviz:=${enable_rviz}"
    if [[ "$enable_rviz" == "true" ]]; then
        inner_cmd+=" rviz_node_name:=${rviz_name}"
    fi

    bash -c "${inner_cmd}" &
    ROS_PID=$!

    timeout 5s ros2 daemon stop  >/dev/null 2>&1 || true
    timeout 5s ros2 daemon start >/dev/null 2>&1 || true
    sleep 1

    info "等待 Gazebo 就绪（最多 ${GAZEBO_WAIT_SEC}s）..."
    local waited=0
    local topics=""
    while [[ $waited -lt $GAZEBO_WAIT_SEC ]]; do
        if [[ -n "$ROS_PID" ]] && ! kill -0 "$ROS_PID" 2>/dev/null; then
            echo ""
            error "仿真环境启动进程已提前退出。"
            return 1
        fi
        topics="$(timeout 3s bash -lc 'export ROS_DOMAIN_ID='"${TEST_ROS_DOMAIN_ID}"'; export GAZEBO_MASTER_URI='"${GAZEBO_MASTER_URI}"'; ROS2CLI_NODE_STRATEGY=direct ros2 topic list 2>/dev/null' || true)"
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

# ─── 执行单个 (scenario, option) 测试 ────────────────────────────────────────────
# 参数: $1=scenario_name  $2=option_name  $3=output_dir  $4=quiet_mode(0/1)
_run_one_combo() {
    local scenario_name="$1"
    local option_name="$2"
    local output_dir="$3"
    local quiet_mode="${4:-0}"

    local log_path="$output_dir/${scenario_name}__${option_name}.log"

    if (( quiet_mode == 0 )); then
        banner "═══ Option 测试 (${scenario_name} / ${option_name}) ═══"
    fi

    local cmd=(
        python3 "$TESTER_SCRIPT"
        --scenario "$scenario_name"
        --option "$option_name"
        --background-option "$BACKGROUND_OPTION"
        --num-episodes "$NUM_EPISODES"
        --max-option-steps "$MAX_OPTION_STEPS"
        --max-episode-steps "$MAX_EPISODE_STEPS"
        --output-dir "$output_dir"
        --rolling-lookahead-dist "$ROLLING_LOOKAHEAD_DIST"
        --obstacle-filter-range "$OBSTACLE_FILTER_RANGE"
        --obstacle-top-k "$OBSTACLE_TOP_K"
    )

    if [[ -n "$SEED" ]]; then
        cmd+=(--seed "$SEED")
    fi
    if (( RESPECT_ACTION_MASK == 1 )); then
        cmd+=(--respect-action-mask)
    fi
    if (( DISABLE_SAFETY_OVERRIDES == 1 )); then
        cmd+=(--disable-safety-overrides)
    fi
    if (( DISABLE_REPLAN == 1 )); then
        cmd+=(--disable-replan)
    fi

    if (( quiet_mode == 0 )); then
        info "测试命令: ${cmd[*]}"
        info "测试日志: $log_path"
        echo ""
    fi

    set +e
    "${cmd[@]}" 2>&1 | tee "$log_path"
    local exit_code=${PIPESTATUS[0]}
    set -e

    # 找到 tester 生成的带时间戳子目录
    local run_subdir
    run_subdir="$(find "$output_dir" -maxdepth 1 -type d -name "${scenario_name}_${option_name}_*" -print -quit 2>/dev/null || true)"
    local summary_json=""
    if [[ -n "$run_subdir" ]]; then
        summary_json="$run_subdir/summary.json"
    fi

    if (( quiet_mode == 1 )); then
        # 静默模式：一行结果
        if [[ $exit_code -ne 0 ]]; then
            error "[${scenario_name}/${option_name}] 异常退出 (code=$exit_code)"
            return $exit_code
        elif [[ -n "$summary_json" ]] && [[ -f "$summary_json" ]]; then
            python3 - "$summary_json" "$scenario_name" "$option_name" <<'PY'
import json, sys
d = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(f"  {d.get('scenario','?'):28s} {d.get('option_name','?'):16s}  "
      f"success={d.get('success_rate',0):.0%}  "
      f"feasible={d.get('initial_feasible_rate',0):.0%}  "
      f"collision={d.get('collision_rate',0):.0%}  "
      f"steps={d.get('mean_steps',0):.1f}  "
      f"progress={d.get('mean_progress_gain',0):.3f}")
PY
            ALL_SUMMARY_PATHS+=("$summary_json")
            return 0
        else
            warn "[${scenario_name}/${option_name}] 未生成 summary.json"
            return 1
        fi
    else
        if [[ $exit_code -ne 0 ]]; then
            error "测试异常退出 (code=$exit_code)"
        else
            success "测试顺利完成！"
        fi
        echo ""

        if [[ -n "$summary_json" ]] && [[ -f "$summary_json" ]]; then
            _print_one_summary "$summary_json"
        else
            warn "未找到 summary.json，请检查测试是否正常完成。"
        fi
    fi
}

# ─── 打印单个 summary ────────────────────────────────────────────────────────────
_print_one_summary() {
    local summary_json="$1"

    banner "═══ Option 测试结果汇总 ═══"

    python3 - "$summary_json" <<'PY'
import json, sys

data = json.loads(open(sys.argv[1], encoding="utf-8").read())

print(f"  场景:        {data.get('scenario', '?')}")
print(f"  Option:      {data.get('option_name', '?')}")
print(f"  输出目录:    {data.get('output_dir', '?')}")
print(f"  Episodes:    {data.get('episodes', '?')}")
print(f"  成功率:      {data.get('success_rate', 0):.1%}")
print(f"  初始可行率:  {data.get('initial_feasible_rate', 0):.1%}")
print(f"  碰撞率:      {data.get('collision_rate', 0):.1%}")
print(f"  平均步数:    {data.get('mean_steps', 0):.1f}")
print(f"  平均进度增益: {data.get('mean_progress_gain', 0):.3f}")
print(f"  前向间隙增益: {data.get('mean_front_clearance_gain', 0):.3f}")
print(f"  社交风险降幅: {data.get('mean_social_risk_drop', 0):.3f}")
print(f"  TTC 增益:    {data.get('mean_ttc_gain', 0):.3f}")
print(f"  Safety 覆盖:  {data.get('mean_safety_override_count', 0):.1f}")
print(f"  Emergency:    {data.get('mean_emergency_override_count', 0):.1f}")

failures = data.get("failure_reason_counts", {})
if failures:
    print(f"  失败原因分布:")
    for reason, count in sorted(failures.items(), key=lambda x: -x[1]):
        print(f"    - {reason}: {count}")
PY
    echo ""

    info "输出文件:"
    local f run_dir
    run_dir="$(dirname "$summary_json")"
    for f in "$run_dir"/*; do
        if [[ -f "$f" ]]; then
            printf "  %s  (%s)\n" "$(basename "$f")" "$(du -h "$f" | cut -f1)"
        fi
    done
}

# ─── 全量测试：所有场景 × 所有 option ───────────────────────────────────────────
run_all_tests() {
    local -a scenarios=()
    local -a options=()
    local total=0 passed=0 failed=0

    # 解析场景列表
    if [[ -n "$ALL_SCENARIOS_FILTER" ]]; then
        IFS=',' read -r -a scenarios <<< "$ALL_SCENARIOS_FILTER"
    else
        mapfile -t scenarios < <(query_all_scenarios)
    fi

    # 解析 option 列表
    if [[ -n "$ALL_OPTIONS_FILTER" ]]; then
        IFS=',' read -r -a options <<< "$ALL_OPTIONS_FILTER"
    else
        mapfile -t options < <(query_all_options)
    fi

    total=$(( ${#scenarios[@]} * ${#options[@]} ))

    banner "═══ 全量 Option 测试 ═══"
    info "场景数: ${#scenarios[@]}"
    info "Option 数: ${#options[@]}"
    info "总组合数: $total"
    info "每组合 Episodes: $NUM_EPISODES"
    info "启动模式: $( ((HEADLESS_SIM==1)) && echo "headless" || echo "headless + RViz" )"
    info "输出目录: $OUTPUT_DIR"
    echo ""

    local scenario_name opt_name meta map_num num_agents desc
    local combo_dir="$OUTPUT_DIR"

    for scenario_name in "${scenarios[@]}"; do
        meta="$(query_scenario_meta "$scenario_name")" || {
            error "无法解析场景: $scenario_name，跳过"
            continue
        }
        IFS=$'\t' read -r map_num num_agents desc <<< "$meta"

        banner "── 场景: ${scenario_name} (map=${map_num}, agents=${num_agents}) ──"
        info "$desc"

        stop_ros_env

        if ! start_ros_env "$map_num" "$num_agents"; then
            error "Gazebo 启动失败，跳过场景 $scenario_name 的所有 option"
            failed=$((failed + ${#options[@]}))
            continue
        fi

        for opt_name in "${options[@]}"; do
            info ">>> ${scenario_name} / ${opt_name}"
            if _run_one_combo "$scenario_name" "$opt_name" "$combo_dir" 1; then
                passed=$((passed + 1))
            else
                failed=$((failed + 1))
            fi
            # 每个 combo 之间留一点间隔让 env 释放资源
            sleep 1
        done

        stop_ros_env
    done

    # ─── 打印全量汇总表 ──────────────────────────────────────────────────────
    echo ""
    banner "═══ 全量测试完成 ═══"
    info "通过: $passed / 失败: $failed / 总计: $total"

    if [[ ${#ALL_SUMMARY_PATHS[@]} -gt 0 ]]; then
        _print_combined_summary "${ALL_SUMMARY_PATHS[@]}"
    fi
}

# ─── 全量汇总表 ──────────────────────────────────────────────────────────────────
_print_combined_summary() {
    local -a paths=("$@")

    banner "═══ 全量 Option 测试汇总表 ═══"

    # 用 Python 生成一张完整矩阵表格
    python3 - "${paths[@]}" <<'PY'
import json, sys
from collections import defaultdict
from pathlib import Path

paths = sys.argv[1:]
rows = []
for p in paths:
    try:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        rows.append(data)
    except Exception:
        pass

if not rows:
    print("无有效 summary 数据")
    raise SystemExit(0)

# 按 scenario 分组，每个 scenario 下列出所有 option
by_scenario = defaultdict(dict)
for r in rows:
    by_scenario[r.get("scenario", "?")][r.get("option_name", "?")] = r

all_options = sorted(set(r.get("option_name", "?") for r in rows))

# 表头
header = f"  {'Scenario':28s}"
for opt in all_options:
    header += f" {opt:>14s}"
print(header)
print("  " + "-" * (28 + 16 * len(all_options)))

for scenario in sorted(by_scenario.keys()):
    line = f"  {scenario:28s}"
    for opt in all_options:
        r = by_scenario[scenario].get(opt)
        if r is None:
            line += f" {'--':>14s}"
        else:
            sr = r.get("success_rate", 0.0)
            cr = r.get("collision_rate", 0.0)
            tag = "✓" if sr >= 0.99 else ("◐" if sr >= 0.5 else ("✗" if cr > 0 else "·"))
            line += f" {tag} {sr:>4.0%} {r.get('mean_steps',0):>4.0f}s"
    print(line)

print()
print("  图例: ✓=成功率≥99%  ◐=≥50%  ✗=有碰撞(成功率<50%)  ·=其他失败  s=mean_steps")
print()

# 全局统计
all_sr = [r.get("success_rate", 0.0) for r in rows]
all_cr = [r.get("collision_rate", 0.0) for r in rows]
all_fr = [r.get("initial_feasible_rate", 0.0) for r in rows]
print(f"  全局成功率:     {sum(all_sr)/len(all_sr):.1%}")
print(f"  全局初始可行率: {sum(all_fr)/len(all_fr):.1%}")
print(f"  全局碰撞率:     {sum(all_cr)/len(all_cr):.1%}")

# 按 option 聚合
print()
print(f"  {'Option':16s}  success  feasible  collision  steps")
print("  " + "-" * 58)
by_option = defaultdict(list)
for r in rows:
    by_option[r.get("option_name", "?")].append(r)
for opt in sorted(by_option.keys()):
    group = by_option[opt]
    sr = sum(r.get("success_rate", 0.0) for r in group) / len(group)
    fr = sum(r.get("initial_feasible_rate", 0.0) for r in group) / len(group)
    cr = sum(r.get("collision_rate", 0.0) for r in group) / len(group)
    ms = sum(r.get("mean_steps", 0.0) for r in group) / len(group)
    print(f"  {opt:16s}  {sr:.0%}     {fr:.0%}      {cr:.0%}       {ms:.1f}")
PY
    echo ""

    # 写出汇总 CSV
    local all_csv="$OUTPUT_DIR/all_summary.csv"
    python3 - "$all_csv" "${paths[@]}" <<'PY'
import csv, json, sys
from pathlib import Path

csv_path = sys.argv[1]
paths = sys.argv[2:]

fieldnames = [
    "scenario", "option_name", "episodes", "success_rate", "initial_feasible_rate",
    "mask_allow_on_start_rate", "collision_rate", "mean_steps", "mean_progress_gain",
    "mean_front_clearance_gain", "mean_social_risk_drop", "mean_ttc_gain",
    "mean_safety_override_count", "mean_emergency_override_count",
    "failure_reason_counts",
]

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for p in paths:
        try:
            data = json.loads(Path(p).read_text(encoding="utf-8"))
            row = {k: data.get(k, "") for k in fieldnames}
            if isinstance(row.get("failure_reason_counts"), dict):
                row["failure_reason_counts"] = json.dumps(row["failure_reason_counts"], ensure_ascii=False)
            writer.writerow(row)
        except Exception:
            pass

print(f"汇总 CSV: {csv_path}")
PY
    info "汇总 CSV: $all_csv"
}

# ─── 单场景测试 ──────────────────────────────────────────────────────────────────
run_single_test() {
    RUN_DIR="$OUTPUT_DIR"
    mkdir -p "$RUN_DIR"

    stop_ros_env

    if start_ros_env "$RUN_MAP_NUM" "$RUN_NUM_AGENTS"; then
        _run_one_combo "$SCENARIO" "$OPTION" "$RUN_DIR" 0
    else
        error "仿真环境启动失败，中止测试。"
    fi

    stop_ros_env
}

# ─── 主入口 ──────────────────────────────────────────────────────────────────────
main() {
    bootstrap_python_env
    export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"
    export ROS_DOMAIN_ID="$TEST_ROS_DOMAIN_ID"
    export GAZEBO_MASTER_URI="http://127.0.0.1:${TEST_GAZEBO_PORT}"
    set +u
    [[ -f "$ROS_SETUP" ]] && source "$ROS_SETUP" || { set -u; error "ROS2 Humble 未找到"; exit 1; }
    [[ -f "$WS_SETUP"  ]] && source "$WS_SETUP"
    set -u

    if (( LIST_SCENARIOS == 1 )); then
        print_scenario_list
        exit 0
    fi

    # 基础校验
    [[ -f "$TESTER_SCRIPT" ]] || { error "测试脚本不存在: $TESTER_SCRIPT"; exit 1; }
    [[ -f "$KILL_SCRIPT"   ]] || { error "kill 脚本不存在: $KILL_SCRIPT";   exit 1; }
    [[ "$NUM_EPISODES"       =~ ^[0-9]+$ ]] || { error "--num-episodes 必须是非负整数"; exit 1; }
    [[ "$MAX_OPTION_STEPS"   =~ ^[0-9]+$ ]] || { error "--max-option-steps 必须是非负整数"; exit 1; }
    [[ "$MAX_EPISODE_STEPS"  =~ ^[0-9]+$ ]] || { error "--max-episode-steps 必须是非负整数"; exit 1; }
    [[ "$TEST_ROS_DOMAIN_ID" =~ ^[0-9]+$ ]] || { error "--ros_domain_id 必须是非负整数"; exit 1; }
    [[ "$TEST_GAZEBO_PORT"   =~ ^[0-9]+$ ]] || { error "--gazebo_port 必须是非负整数"; exit 1; }
    OUTPUT_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$OUTPUT_DIR")"

    if (( RUN_ALL == 1 )); then
        run_all_tests
    else
        check_single_env
        run_single_test
    fi

    banner "═══ Option 测试脚本执行完毕 ═══"
}

# ─── 参数解析 ────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenario)                SCENARIO="$2";              shift 2 ;;
        --option)                  OPTION="$2";                shift 2 ;;
        --background-option)       BACKGROUND_OPTION="$2";     shift 2 ;;
        --num-episodes)            NUM_EPISODES="$2";          shift 2 ;;
        --max-option-steps)        MAX_OPTION_STEPS="$2";      shift 2 ;;
        --max-episode-steps)       MAX_EPISODE_STEPS="$2";     shift 2 ;;
        --output-dir)              OUTPUT_DIR="$2";            shift 2 ;;
        --seed)                    SEED="$2";                  shift 2 ;;
        --respect-action-mask)     RESPECT_ACTION_MASK=1;      shift 1 ;;
        --disable-safety-overrides) DISABLE_SAFETY_OVERRIDES=1; shift 1 ;;
        --disable-replan)          DISABLE_REPLAN=1;           shift 1 ;;
        --rolling-lookahead-dist)  ROLLING_LOOKAHEAD_DIST="$2"; shift 2 ;;
        --obstacle-filter-range)   OBSTACLE_FILTER_RANGE="$2"; shift 2 ;;
        --obstacle-top-k)          OBSTACLE_TOP_K="$2";        shift 2 ;;
        --list-scenarios)          LIST_SCENARIOS=1;           shift 1 ;;
        --all)                     RUN_ALL=1;                  shift 1 ;;
        --scenarios)               ALL_SCENARIOS_FILTER="$2";  shift 2 ;;
        --options)                 ALL_OPTIONS_FILTER="$2";    shift 2 ;;
        --ros_domain_id)           TEST_ROS_DOMAIN_ID="$2";    shift 2 ;;
        --gazebo_port)             TEST_GAZEBO_PORT="$2";      shift 2 ;;
        --headless)                HEADLESS_SIM=1;             shift 1 ;;
        --gui)                     HEADLESS_SIM=0;             shift 1 ;;
        -h|--help)
            echo "用法: ./run_option_test.sh [OPTIONS]"
            echo ""
            echo "场景选择:"
            echo "  --scenario              测试场景名称 (默认 single_follow)"
            echo "  --list-scenarios        列出所有可用场景并退出"
            echo ""
            echo "Option 配置:"
            echo "  --option                要测试的原子 option (默认 follow_path)"
            echo "  --background-option     背景机器人使用的 option (默认 follow_path)"
            echo ""
            echo "全量测试:"
            echo "  --all                   测试所有场景 × 所有 option 的组合"
            echo "  --scenarios S1,S2,...   配合 --all: 只测指定场景 (逗号分隔)"
            echo "  --options O1,O2,...     配合 --all: 只测指定 option (逗号分隔)"
            echo ""
            echo "测试控制:"
            echo "  --num-episodes          每个场景重复测试次数 (默认 3)"
            echo "  --max-option-steps      ego option 最大持续执行步数 (默认 14)"
            echo "  --max-episode-steps     单次 episode 最大步数 (默认 40)"
            echo "  --output-dir            输出目录 (默认 option_test_results)"
            echo "  --seed                  随机种子"
            echo "  --respect-action-mask   若当前 option 不可行则不强制执行"
            echo "  --disable-safety-overrides  关闭 emergency/safety override"
            echo "  --disable-replan        在 action mask 中禁用 replan"
            echo ""
            echo "环境参数:"
            echo "  --rolling-lookahead-dist  rolling lookahead 距离 (默认 0.8)"
            echo "  --obstacle-filter-range  局部障碍观测半径 (默认 1.5)"
            echo "  --obstacle-top-k         Top-K 障碍编码数量 (默认 9)"
            echo ""
            echo "仿真控制:"
            echo "  --ros_domain_id          ROS_DOMAIN_ID (默认 72)"
            echo "  --gazebo_port            Gazebo 端口 (默认 11846)"
            echo "  --headless               headless 模式 (默认, 纯 gzserver)"
            echo "  --gui                    开启 RViz 可视化 (headless + RViz)"
            echo ""
            echo "示例:"
            echo "  ./run_option_test.sh --scenario head_on_two_agents --option backoff"
            echo "  ./run_option_test.sh --all                                          # 全量测试"
            echo "  ./run_option_test.sh --all --gui --num-episodes 5                   # GUI 全量"
            echo "  ./run_option_test.sh --all --scenarios l_corner,narrow_corridor     # 部分场景"
            echo "  ./run_option_test.sh --all --options backoff,detour_left,detour_right"
            echo "  ./run_option_test.sh --list-scenarios"
            exit 0
            ;;
        *) error "未知参数: $1"; exit 1 ;;
    esac
done

main
