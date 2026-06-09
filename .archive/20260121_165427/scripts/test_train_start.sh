#!/bin/bash

##############################################################################
# 快速测试训练启动 - 运行10步验证
##############################################################################

echo "🧪 快速测试训练启动"
echo ""

source /opt/ros/humble/setup.bash
source install/setup.bash

echo "🚀 启动训练（10秒测试）..."
echo "   地图: map1"
echo "   机器人数: 1"
echo "   模式: 固定位置"
echo ""

# 启动训练，10秒后自动终止
timeout 10 ros2 launch start_reinforcement_learning start_learning.launch.py \
    map_number:=1 \
    robot_number:=1 \
    use_random_mode:=false \
    2>&1 | tee /tmp/quick_test_train.log &

TRAIN_PID=$!
echo "   训练 PID: $TRAIN_PID"

# 等待进程
sleep 10

echo ""
echo "🔍 检查训练日志..."
if grep -q "Map number\|Robot number\|开始训练\|Episode" /tmp/quick_test_train.log; then
    echo "✅ 训练节点成功启动！"
    echo ""
    echo "📋 日志片段:"
    grep -E "Map number|Robot number|Episode|训练" /tmp/quick_test_train.log | head -10
else
    echo "❌ 训练可能未正常启动"
    echo ""
    echo "📋 完整日志:"
    tail -30 /tmp/quick_test_train.log
fi

echo ""
echo "✅ 测试完成"
echo ""
echo "💡 如果看到 'Episode' 关键词，说明训练正在运行"
echo "💡 完整训练请运行: ./train_stage.sh 1 500"
