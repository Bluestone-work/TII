#!/bin/bash
# P0 实验队列: 全长度(Stage2=30万步), 单种子(42), 串行
# 复用正在跑的 pilot_smoke_gat 作为 dual_graph 基线, 只补 5 个新配置
# 用法: nohup bash run_p0_queue.sh > p0_queue.log 2>&1 &
set -uo pipefail

WORKSPACE="/home/wj/work/multi-robot-exploration-rl"
PKG_DIR="$WORKSPACE/src/gnn_marl_training_DGTA_nobuffer"
PROGRESS="$WORKSPACE/experiments/p0_progress.log"
SEED=42
COMMON="--start_stage 2 --end_stage 2 --num_agents 4 --num_workers 1 --seed $SEED"

cd "$PKG_DIR"

log_progress() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS"; }

cleanup_env() {
    ray stop --force >/dev/null 2>&1 || true
    pkill -f "from multiprocessing.spawn" >/dev/null 2>&1 || true
    pkill -f "gzserver" >/dev/null 2>&1 || true
    sleep 3
}

# 等正在跑的冒烟测试(dual_graph 基线)自然完成, 不杀它
log_progress "P0 队列启动。等待 pilot_smoke_gat(dual_graph基线)完成..."
while pgrep -f "pilot_smoke_gat" >/dev/null 2>&1 || pgrep -f train_gnn_mappo_full >/dev/null 2>&1; do
    sleep 60
done
log_progress "冒烟测试已完成, 开始 P0 剩余 5 个配置。"
cleanup_env

# 实验配置: "标签|额外参数|run_suffix"
declare -a EXPERIMENTS=(
    "MLP-baseline|--model_type mlp|p0_cmp_mlp_seed${SEED}"
    "GAT-social_only|--model_type gat --graph_ablation social_only --gat_actor_graph local_risk|p0_abl_social_only_seed${SEED}"
    "GAT-obstacle_only|--model_type gat --graph_ablation obstacle_only --gat_actor_graph local_risk|p0_abl_obstacle_only_seed${SEED}"
    "GAT-comm2.0|--model_type gat --graph_ablation dual_graph --gat_actor_graph local_risk --communication_range 2.0|p0_comm_2.0_seed${SEED}"
    "GAT-comm6.0|--model_type gat --graph_ablation dual_graph --gat_actor_graph local_risk --communication_range 6.0|p0_comm_6.0_seed${SEED}"
)

TOTAL=${#EXPERIMENTS[@]}
IDX=0
for entry in "${EXPERIMENTS[@]}"; do
    IDX=$((IDX+1))
    IFS='|' read -r label extra suffix <<< "$entry"
    log_progress ">>> [$IDX/$TOTAL] 启动: $label (suffix=$suffix)"

    TRAIN_VERBOSE=0 ENV_VERBOSE=0 RAY_memory_usage_threshold=0.97 \
    ./run_curriculum.sh $COMMON $extra --run_suffix "$suffix" \
        > "$WORKSPACE/experiments/logs_${suffix}.log" 2>&1
    rc=$?

    if [[ $rc -eq 0 ]]; then
        log_progress "<<< [$IDX/$TOTAL] 完成: $label"
    else
        log_progress "!!! [$IDX/$TOTAL] 失败: $label (exit=$rc), 继续下一个"
    fi
    cleanup_env
done

log_progress "✅ P0 队列全部完成 ($TOTAL 个配置)。运行 aggregate_pilot_results.py 查看汇总。"
