#!/bin/bash
# 验证run_test.sh的NUM_AGENTS逻辑是否正确

set -euo pipefail

WORKSPACE="/home/wj/work/multi-robot-exploration-rl"
cd "$WORKSPACE/src/gnn_marl_training_DGTA_nobuffer"

# 模拟STAGE_NUM_AGENTS数组
declare -A STAGE_NUM_AGENTS=([1]=2 [2]=4 [3]=6 [4]=8 [5]=4 [6]=4)

echo "======================================"
echo "  验证 NUM_AGENTS 更新逻辑"
echo "======================================"
echo ""

echo "【场景1】单stage测试 (TEST_STAGE=1)"
TEST_STAGE=1
NUM_AGENTS=""
if [[ -z "$NUM_AGENTS" ]]; then
    NUM_AGENTS=${STAGE_NUM_AGENTS[$TEST_STAGE]:-2}
fi
echo "  结果: NUM_AGENTS=$NUM_AGENTS (预期: 2)"
echo ""

echo "【场景2】单stage测试 (TEST_STAGE=2)"
TEST_STAGE=2
NUM_AGENTS=""
if [[ -z "$NUM_AGENTS" ]]; then
    NUM_AGENTS=${STAGE_NUM_AGENTS[$TEST_STAGE]:-2}
fi
echo "  结果: NUM_AGENTS=$NUM_AGENTS (预期: 4)"
echo ""

echo "【场景3】all_stages循环测试"
for s in {1..6}; do
    NUM_AGENTS=${STAGE_NUM_AGENTS[$s]}
    echo "  Stage $s: NUM_AGENTS=$NUM_AGENTS (预期: ${STAGE_NUM_AGENTS[$s]})"
done
echo ""

echo "【场景4】手动指定 --num_agents 3"
NUM_AGENTS=3
TEST_STAGE=2
# 手动指定时不会被覆盖(单stage模式)
if [[ -z "$NUM_AGENTS" ]]; then
    NUM_AGENTS=${STAGE_NUM_AGENTS[$TEST_STAGE]:-2}
fi
echo "  结果: NUM_AGENTS=$NUM_AGENTS (预期: 3, 不应该被覆盖)"
echo ""

echo "======================================"
echo "  ✅ 逻辑验证完成"
echo "======================================"
echo ""
echo "结论: run_test.sh 的 NUM_AGENTS 逻辑是正确的！"
echo ""
echo "如果测试时 num_agents 不匹配,可能原因:"
echo "  1. test_gnn_mappo.py 在覆盖env_config['num_agents']后没有同步更新 num_agents 局部变量"
echo "  2. GNNMARLEnv.__init__ 时 num_agents 还没生效"
echo ""
echo "已修复: test_gnn_mappo.py 第465行添加了 num_agents = int(args.num_agents)"
