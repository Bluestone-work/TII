#!/bin/bash

# 启动带距离场功能的训练环境
# 使用方法: ./start_with_distance_field.sh [map_number] [robot_number]

# 默认参数
MAP_NUM=${1:-3}         # 默认 corridor_swap
ROBOT_NUM=${2:-4}       # 默认 4 个机器人

# 地图文件映射
case $MAP_NUM in
    1) MAP_FILE="map1.yaml" ;;
    2) MAP_FILE="map2.yaml" ;;
    3) MAP_FILE="corridor_swap.yaml" ;;
    4) MAP_FILE="intersection.yaml" ;;
    5) MAP_FILE="warehouse_aisles.yaml" ;;
    *) echo "错误: 无效的地图编号 $MAP_NUM"; exit 1 ;;
esac

MAP_PATH="/home/wj/work/multi-robot-exploration-rl/src/start_reinforcement_learning/maps/$MAP_FILE"

cd /home/wj/work/multi-robot-exploration-rl
source install/setup.bash

echo "================================================"
echo "🗺️  启动距离场导航训练"
echo "================================================"
echo "地图编号: $MAP_NUM ($MAP_FILE)"
echo "机器人数量: $ROBOT_NUM"
echo "地图路径: $MAP_PATH"
echo "================================================"

# 1. 启动 Gazebo 环境
echo "🚀 [1/3] 启动 Gazebo 环境..."
gnome-terminal --title="Gazebo Environment" -- bash -c "
    source install/setup.bash && \
    ros2 launch start_rl_environment main.launch.py map_number:=$MAP_NUM robot_number:=$ROBOT_NUM; \
    exec bash
" &

sleep 15  # 等待 Gazebo 和地图服务器完全启动

# 2. 验证地图服务器
echo "🗺️  [2/3] 验证地图服务器状态..."
source install/setup.bash

# 检查 /map 话题是否存在
if ros2 topic list | grep -q "/map"; then
    echo "✓ /map 话题已发布"
else
    echo "⚠️  /map 话题未找到，地图服务器可能未启动"
fi

sleep 2

# 3. 启动训练
echo "🤖 [3/3] 启动 MATD3 训练（带距离场）..."
echo ""
echo "⏳ 训练即将开始..."
echo "   - 观测维度: 98 (49 基础 + 49 距离场)"
echo "   - MapSubscriber 正在订阅 /map 话题"
echo ""

sleep 2

ros2 launch start_reinforcement_learning start_learning.launch.py \
    map_number:=$MAP_NUM \
    robot_number:=$ROBOT_NUM \
    goal_termination_mode:=any \
    stuck_enabled:=true \
    stuck_min_progress:=0.02 \
    stuck_max_steps:=40 \
    stuck_check_after_steps:=20 \
    stuck_penalty:=-10.0

echo ""
echo "训练已结束。"
