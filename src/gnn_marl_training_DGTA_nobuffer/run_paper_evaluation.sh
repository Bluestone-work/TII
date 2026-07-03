#!/bin/bash
# =============================================================================
# 论文评估批处理脚本 (2026-06-29)
#
# 把 N 个 checkpoint × M 个测试 stage 串行跑完，每个组合启停一次 Gazebo
# （Gazebo 单实例约束，无法并行）
#
# 输出：results.jsonl  每行一个 episode 的指标
#       论文 Table 1 / Figure 7 可由 results.jsonl 直接聚合
#
# 用法:
#   ./run_paper_evaluation.sh                   # 默认 (重训完成后用此)
#   ./run_paper_evaluation.sh --episodes 30     # 减少 episode 数
#   ./run_paper_evaluation.sh --dry-run         # 看会跑哪些组合
# =============================================================================

if [ -z "${BASH_VERSION:-}" ]; then
    exec /bin/bash "$0" "$@"
fi

set -uo pipefail

# ─── 默认配置 ─────────────────────────────────────────────────────────────────
NUM_EPISODES=${NUM_EPISODES:-30}    # 每个 (method × stage) 跑多少 episode
DRY_RUN=0
WORKSPACE="/home/wj/work/multi-robot-exploration-rl"
RAY_ROOT="$WORKSPACE/ray_results"
RUN_TEST_SH="$WORKSPACE/src/gnn_marl_training_DGTA_nobuffer/run_test.sh"
TS=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$WORKSPACE/paper_results/eval_${TS}"
RESULTS_JSONL="$RESULTS_DIR/episodes.jsonl"
SUMMARY_CSV="$RESULTS_DIR/summary.csv"
LOG_DIR="$RESULTS_DIR/logs"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# ─── 颜色 ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; RED='\033[0;31m'
BOLD='\033[1m'; RESET='\033[0m'
info()    { echo -e "${CYAN}[INFO]${RESET} $*"; }
ok()      { echo -e "${GREEN}[OK]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET} $*"; }
err()     { echo -e "${RED}[ERR]${RESET} $*" >&2; }
banner()  { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════════════════════════════════${RESET}\n${BOLD}${CYAN}  $*${RESET}\n${BOLD}${CYAN}══════════════════════════════════════════════════════════════════════${RESET}\n"; }

# ─── 参数解析 ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --episodes)  NUM_EPISODES="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=1; shift ;;
        -h|--help)
            echo "用法: $0 [--episodes N] [--dry-run]"
            echo "  --episodes N    每个 (method × stage) 跑多少 episode (默认 30)"
            echo "  --dry-run       只打印会跑哪些组合，不真跑"
            exit 0 ;;
        *) err "未知参数: $1"; exit 1 ;;
    esac
done

# ─── 评估矩阵：method → ckpt_root ────────────────────────────────────────────
# 每个 method 对应一个 ray_results/ 下的 checkpoint 目录（指 best/）
# 如果同一 method 有多 seed，写多个用空格隔开（评估时分别跑，最后聚合）
declare -A METHOD_CKPTS

# 注: best/ 在 ray_results/.../EnvStage2/best/ 这一层
RAYRES="$RAY_ROOT"

METHOD_CKPTS[ours_dual]="$RAYRES/dgta_dual_dual_graph_seed1/GNN_MAPPO_Stage1_Cont_dgta_dual_dual_graph_seed1_EnvStage2/best"
METHOD_CKPTS[social_only]="$RAYRES/dgta_social_social_only_seed1/GNN_MAPPO_Stage1_Cont_dgta_social_social_only_seed1_EnvStage2/best"
METHOD_CKPTS[obstacle_only]="$RAYRES/dgta_obstacle_obstacle_only_seed1/GNN_MAPPO_Stage1_Cont_dgta_obstacle_obstacle_only_seed1_EnvStage2/best"
METHOD_CKPTS[mlp_lstm]="$RAYRES/p0_cmp_mlp_seed42/MAPPO_MLP_LSTM_Stage2_Cont_p0_cmp_mlp_seed42/best"

# ── 占位：等重训完成后追加新模型 ──
# 重训出的带 CF 的 Ours: 训练完成后取代 dgta_dual_dual_graph_seed1 路径
# 例：METHOD_CKPTS[ours_dual_cf]="$RAYRES/<new training run>/best"

# ─── 测试场景：stage 编号 (按 run_test.sh::STAGE_MAP_NUM 定义) ──────────────
# run_test.sh 已有的映射：
#   Stage 1: map=8, 2车基础避障
#   Stage 2: map=8, 4车多体
#   Stage 3: map=8, 8车高密度
#   Stage 4: map=4, 4车十字交汇泛化
# 论文主战场：Stage 2 (Circle-Swap-4) 因为现有 ckpt 都是 4-agent 训出来的
TEST_STAGES=(2)
# 如有 8 车模型，加进来:  TEST_STAGES=(2 3)

# ─── 列出评估计划 ────────────────────────────────────────────────────────────
banner "评估计划"
info "结果目录: $RESULTS_DIR"
info "JSONL    : $RESULTS_JSONL"
info "Episodes : $NUM_EPISODES per (method × stage)"
info ""
info "Methods × Stages:"
i=0
declare -a PLAN
for method in "${!METHOD_CKPTS[@]}"; do
    ckpt="${METHOD_CKPTS[$method]}"
    if [[ ! -d "$ckpt" ]]; then
        warn "  [$method] 跳过: ckpt 不存在 $ckpt"
        continue
    fi
    for stage in "${TEST_STAGES[@]}"; do
        i=$((i+1))
        PLAN+=("$method|$stage|$ckpt")
        info "  $i. $method × stage=$stage"
    done
done
TOTAL=${#PLAN[@]}
info ""
info "总计: $TOTAL 个评估任务，每个 ~$NUM_EPISODES episodes (约 5-15 分钟)"
info "估计总耗时: $((TOTAL * 10)) 分钟 (单 Gazebo 串行)"

if (( DRY_RUN == 1 )); then
    info ""
    ok "Dry-run 完成。去掉 --dry-run 即可真跑"
    exit 0
fi

# ─── 执行 ────────────────────────────────────────────────────────────────────
banner "开始执行"
START_TS=$(date +%s)
SUCCESS=0
FAILED=0
SKIPPED=0

# CSV 表头
echo "method,stage,episodes_done,total_success,total_collision,avg_reward,avg_steps,avg_min_dist,truncated_eps,wall_time_sec,status" > "$SUMMARY_CSV"

for idx in "${!PLAN[@]}"; do
    IFS='|' read -r method stage ckpt <<< "${PLAN[$idx]}"
    n=$((idx + 1))
    log="$LOG_DIR/${method}_stage${stage}.log"

    banner "[$n/$TOTAL]  method=$method  stage=$stage"
    info "ckpt: $ckpt"
    info "log : $log"

    task_start=$(date +%s)
    # 调用 run_test.sh，它会自己启停 Gazebo
    bash "$RUN_TEST_SH" \
        -c "$ckpt" \
        --num_episodes "$NUM_EPISODES" \
        --test_stage "$stage" \
        --results_jsonl "$RESULTS_JSONL" \
        --method_label "$method" \
        --scenario_label "stage${stage}" \
        > "$log" 2>&1
    rc=$?
    task_dur=$(($(date +%s) - task_start))

    if (( rc != 0 )); then
        FAILED=$((FAILED + 1))
        err "[$method × stage=$stage] 失败 (rc=$rc, ${task_dur}s)，日志见: $log"
        echo "$method,$stage,0,0,0,,,,,$task_dur,failed" >> "$SUMMARY_CSV"
    else
        # 从 log 抓总体数字写入 CSV
        all_success=$(grep -oP "总计到达: \K[0-9]+" "$log" | tail -1)
        all_collision=$(grep -oP "总计碰撞: \K[0-9]+" "$log" | tail -1)
        avg_reward=$(grep -oP "平均回报: \K-?[0-9.]+" "$log" | tail -1)
        avg_steps=$(grep -oP "平均步数: \K[0-9.]+" "$log" | tail -1)
        avg_min_dist=$(grep -oP "平均最小间距: \K[0-9.]+" "$log" | tail -1)
        trunc=$(grep -oP "被时间截断的 Episode: \K[0-9]+" "$log" | tail -1)

        echo "$method,$stage,$NUM_EPISODES,${all_success:-?},${all_collision:-?},${avg_reward:-?},${avg_steps:-?},${avg_min_dist:-?},${trunc:-?},$task_dur,ok" >> "$SUMMARY_CSV"
        SUCCESS=$((SUCCESS + 1))
        ok "[$method × stage=$stage] 完成 (${task_dur}s)"
        info "    success=$all_success  collision=$all_collision  avg_reward=$avg_reward  avg_min_dist=$avg_min_dist  truncated=$trunc"
    fi

    # Gazebo 之间留 5 秒缓冲清理
    sleep 5
done

TOTAL_DUR=$(($(date +%s) - START_TS))

banner "评估完成"
info "总耗时: ${TOTAL_DUR}s ($((TOTAL_DUR / 60)) 分钟)"
info "成功 : $SUCCESS / $TOTAL"
info "失败 : $FAILED / $TOTAL"
info ""
ok "结果目录: $RESULTS_DIR"
ok "  - episodes.jsonl  (per-episode 指标, 论文 Figure/Table 用)"
ok "  - summary.csv     (per-method 总体指标)"
ok "  - logs/           (各任务原始日志)"
echo ""
info "查看 summary.csv:"
echo ""
column -t -s, "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"

exit $((FAILED > 0 ? 1 : 0))
