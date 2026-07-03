#!/bin/bash
# 一键诊断脚本: 从当前训练/测试的 checkpoint 采集数据 → 可视化分析 → 生成报告

set -e

echo "================================================================"
echo "避碰策略定量诊断 - 一键流程"
echo "================================================================"
echo ""

# ===== 配置 =====
CHECKPOINT_DIR="${1:-}"  # 第一个参数: checkpoint 目录,可选
NUM_EPISODES=20          # 采集 episode 数
ENV_STAGE=2              # 环境 stage
NUM_AGENTS=4             # 机器人数
MAX_STEPS=1500           # 单 episode 最大步数
OUTPUT_DIR="./diagnosis_output_$(date +%Y%m%d_%H%M%S)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECT_SCRIPT="$SCRIPT_DIR/collect_episode_diagnostics.py"
VISUALIZE_SCRIPT="$SCRIPT_DIR/visualize_diagnostics.py"

# ===== 检查依赖 =====
echo "[检查] Python 依赖..."
python3 -c "import pandas, matplotlib, seaborn" 2>/dev/null || {
    echo "❌ 缺少依赖: pandas, matplotlib, seaborn"
    echo "   安装: pip install pandas matplotlib seaborn"
    exit 1
}
echo "✓ 依赖满足"
echo ""

# ===== 检查 Gazebo =====
echo "[检查] Gazebo 是否运行..."
if ! pgrep -x "gzserver" > /dev/null; then
    echo "⚠️  gzserver 未运行!"
    echo "   请先启动 Gazebo 环境:"
    echo "   ros2 launch ... map_number:=... robot_number:=$NUM_AGENTS"
    echo ""
    read -p "   已启动 Gazebo? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "   中断."
        exit 1
    fi
fi
echo "✓ Gazebo 运行中"
echo ""

# ===== 步骤 1: 采集数据 =====
DATA_FILE="$OUTPUT_DIR/diagnosis_data.jsonl"
mkdir -p "$OUTPUT_DIR"

echo "================================================================"
echo "步骤 1: 采集 episode 数据"
echo "================================================================"
echo "  配置:"
echo "    env_stage   = $ENV_STAGE"
echo "    num_agents  = $NUM_AGENTS"
echo "    num_episodes= $NUM_EPISODES"
echo "    max_steps   = $MAX_STEPS"
if [ -n "$CHECKPOINT_DIR" ]; then
    echo "    checkpoint  = $CHECKPOINT_DIR"
    CHECKPOINT_ARG="--checkpoint $CHECKPOINT_DIR"
else
    echo "    checkpoint  = (随机策略)"
    CHECKPOINT_ARG=""
fi
echo "    输出        = $DATA_FILE"
echo ""

python3 "$COLLECT_SCRIPT" \
    $CHECKPOINT_ARG \
    --num_episodes $NUM_EPISODES \
    --env_stage $ENV_STAGE \
    --num_agents $NUM_AGENTS \
    --max_steps $MAX_STEPS \
    --output "$DATA_FILE" \
    --action_mode continuous

if [ ! -f "$DATA_FILE" ]; then
    echo "❌ 数据采集失败"
    exit 1
fi

echo ""
echo "✓ 数据采集完成: $DATA_FILE"
echo "  记录数: $(wc -l < "$DATA_FILE")"
echo ""

# ===== 步骤 2: 可视化分析 =====
echo "================================================================"
echo "步骤 2: 可视化分析"
echo "================================================================"
echo ""

python3 "$VISUALIZE_SCRIPT" \
    --input "$DATA_FILE" \
    --output_dir "$OUTPUT_DIR"

if [ ! -f "$OUTPUT_DIR/diagnosis_report.html" ]; then
    echo "❌ 可视化失败"
    exit 1
fi

echo ""
echo "✓ 分析完成"
echo ""

# ===== 总结 =====
echo "================================================================"
echo "诊断完成!"
echo "================================================================"
echo ""
echo "📁 输出目录: $OUTPUT_DIR"
echo ""
echo "📊 主要产出:"
echo "   • diagnosis_report.html          (HTML 报告,包含所有图表)"
echo "   • diagnosis_data.jsonl           (原始数据)"
echo "   • reward_vs_distance_*.png       (奖励-距离分析)"
echo "   • collision_analysis.png         (碰撞分析)"
echo "   • reward_vs_velocity.png         (速度-奖励分析)"
echo ""
echo "🌐 查看报告:"
echo "   浏览器打开: file://$(realpath "$OUTPUT_DIR/diagnosis_report.html")"
echo ""

# 尝试自动打开浏览器 (Linux)
if command -v xdg-open &> /dev/null; then
    echo "   (尝试自动打开浏览器...)"
    xdg-open "$OUTPUT_DIR/diagnosis_report.html" 2>/dev/null || true
fi

echo ""
echo "💡 下一步:"
echo "   1. 查看报告里的 '⚠️' 警告项"
echo "   2. 对照 DIAGNOSIS_REPORT.md 的改进方案"
echo "   3. 决定改哪些 (建议先改 奖励重平衡 + 扇区距离)"
echo ""
echo "================================================================"
