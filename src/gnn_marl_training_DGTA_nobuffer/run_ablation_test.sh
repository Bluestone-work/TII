#!/bin/bash
# =============================================================================
# 3 个 ablation × stage 4 批量测试脚本
# 每个 ablation 跑 N 个 episode，输出对比表
# =============================================================================

if [ -z "${BASH_VERSION:-}" ]; then
    exec /bin/bash "$0" "$@"
fi

set -uo pipefail

# ─── 配置 ─────────────────────────────────────────────────────────────────────
NUM_EPISODES=${NUM_EPISODES:-5}
TEST_STAGE=${TEST_STAGE:-4}
WORKSPACE="/home/wj/work/multi-robot-exploration-rl"
RAY_ROOT="$WORKSPACE/ray_results"
RESULTS_DIR="$WORKSPACE/test_results/ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

declare -A ABLATIONS=(
    [dual_graph]="$RAY_ROOT/dgta_dual_dual_graph_seed1/GNN_MAPPO_Stage1_Cont_dgta_dual_dual_graph_seed1_EnvStage2"
    [social_only]="$RAY_ROOT/dgta_social_social_only_seed1"
    [obstacle_only]="$RAY_ROOT/dgta_obstacle_obstacle_only_seed1"
)

# ─── 颜色 ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
info()    { echo -e "${CYAN}[INFO]${RESET} $*"; }
ok()      { echo -e "${GREEN}[OK]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET} $*"; }
banner()  { echo -e "\n${BOLD}${CYAN}========== $* ==========${RESET}\n"; }

# ─── 解析每个 ablation 的实际 trial 子目录 ────────────────────────────────────
resolve_trial_dir() {
    local root="$1"
    if [[ -f "$root/algorithm_state.pkl" ]]; then echo "$root"; return; fi
    local child
    child=$(find "$root" -maxdepth 2 -name "algorithm_state.pkl" 2>/dev/null | head -1)
    [[ -n "$child" ]] && dirname "$child"
}

# ─── 启动 Gazebo 一次，按 stage 复用 ───────────────────────────────────────────
RUN_TEST_SH="$WORKSPACE/src/gnn_marl_training_DGTA_nobuffer/run_test.sh"

# ─── 主循环 ──────────────────────────────────────────────────────────────────
banner "ABLATION × STAGE $TEST_STAGE  ·  EPISODES=$NUM_EPISODES"

declare -A SUMMARY_SUCCESS SUMMARY_COLLISION SUMMARY_AVGSTEPS SUMMARY_AVGREWARD SUMMARY_MINDIST SUMMARY_TRUNC

for name in dual_graph social_only obstacle_only; do
    raw="${ABLATIONS[$name]}"
    ckpt=$(resolve_trial_dir "$raw")
    if [[ -z "$ckpt" || ! -d "$ckpt" ]]; then
        warn "[$name] checkpoint 目录无效: $raw"
        continue
    fi
    log="$RESULTS_DIR/${name}.log"
    banner "[$name]  checkpoint=$ckpt"
    info "log -> $log"

    bash "$RUN_TEST_SH" \
        -c "$ckpt" \
        --num_agents 4 \
        --num_episodes "$NUM_EPISODES" \
        --test_stage "$TEST_STAGE" \
        2>&1 | tee "$log" || warn "[$name] 测试出错（继续下一个）"

    # 抽取总体结果行
    total_success=$(grep -E "总计到达:" "$log" | tail -1 | sed -E 's/.*总计到达: *([0-9]+) *次.*/\1/' || echo "-")
    total_coll=$(grep -E "总计碰撞:" "$log" | tail -1 | sed -E 's/.*总计碰撞: *([0-9]+) *次.*/\1/' || echo "-")
    avg_steps=$(grep -E "平均回报" "$log" | tail -1 | sed -E 's/.*平均步数: *([0-9.]+).*/\1/' || echo "-")
    avg_reward=$(grep -E "平均回报" "$log" | tail -1 | sed -E 's/.*平均回报: *(-?[0-9.]+).*/\1/' || echo "-")
    avg_min_dist=$(grep -E "平均最小间距" "$log" | tail -1 | sed -E 's/.*平均最小间距: *([0-9.]+).*/\1/' || echo "-")
    trunc=$(grep -E "被时间截断" "$log" | tail -1 | sed -E 's/.*被时间截断的 Episode: *([0-9]+).*/\1/' || echo "-")

    SUMMARY_SUCCESS[$name]="$total_success"
    SUMMARY_COLLISION[$name]="$total_coll"
    SUMMARY_AVGSTEPS[$name]="$avg_steps"
    SUMMARY_AVGREWARD[$name]="$avg_reward"
    SUMMARY_MINDIST[$name]="$avg_min_dist"
    SUMMARY_TRUNC[$name]="$trunc"
done

# ─── 汇总表 ──────────────────────────────────────────────────────────────────
SUMMARY="$RESULTS_DIR/SUMMARY.txt"
{
    echo "================================================================="
    echo "  ABLATION 测试汇总  (stage=$TEST_STAGE, episodes=$NUM_EPISODES)"
    echo "================================================================="
    printf "%-16s | %-7s | %-7s | %-8s | %-9s | %-9s | %-6s\n" \
        "ablation" "success" "collide" "avgSteps" "avgReward" "minDist" "trunc"
    echo "-----------------+---------+---------+----------+-----------+-----------+--------"
    for name in dual_graph social_only obstacle_only; do
        printf "%-16s | %-7s | %-7s | %-8s | %-9s | %-9s | %-6s\n" \
            "$name" \
            "${SUMMARY_SUCCESS[$name]:--}" \
            "${SUMMARY_COLLISION[$name]:--}" \
            "${SUMMARY_AVGSTEPS[$name]:--}" \
            "${SUMMARY_AVGREWARD[$name]:--}" \
            "${SUMMARY_MINDIST[$name]:--}" \
            "${SUMMARY_TRUNC[$name]:--}"
    done
    echo "================================================================="
} | tee "$SUMMARY"

banner "结果已保存到 $RESULTS_DIR"
