#!/bin/bash
# 测试单个episode，查看done信号是否正确

set -euo pipefail

WORKSPACE="/home/wj/work/multi-robot-exploration-rl"
cd "$WORKSPACE"

# 找到Stage1的checkpoint
STAGE1_CKPT=$(find ray_results -type d -name "best" | \
    grep "GNN_MAPPO_Stage1_Cont_EnvStage1" | \
    grep -v "seed\|ablation\|no_action_mask\|circle_A\|p0_" | \
    head -1)

if [[ -z "$STAGE1_CKPT" ]]; then
    echo "❌ 找不到Stage1 checkpoint"
    exit 1
fi

echo "使用checkpoint: $STAGE1_CKPT"
echo "测试单个episode，观察done信号..."

cd src/gnn_marl_training_DGTA_nobuffer

# 运行单个episode
./run_test.sh \
    -c "$WORKSPACE/$STAGE1_CKPT" \
    --test_stage 1 \
    --num_episodes 1
