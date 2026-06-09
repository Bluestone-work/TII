#!/bin/bash
# 测试导航修复

echo "========================================"
echo "测试ORCA导航修复（使用waypoint目标）"
echo "========================================"
echo ""
echo "✅ 修复内容："
echo "  1. DWA使用实际waypoint而非ORCA速度投射"
echo "  2. 路径显示球体缩小到0.06米（原0.3米）"
echo "  3. 路径插值显示（0.3米间隔）"
echo ""
echo "🎯 预期效果："
echo "  - 机器人直接朝waypoint前进（不再原地摇摆）"
echo "  - 路径显示为彩色小球链"
echo "  - 每条路径约每0.3米一个球"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# 激活ROS2环境并启动
source /home/wj/anaconda3/etc/profile.d/conda.sh
conda activate ros2
source /home/wj/work/multi-robot-exploration-rl/install/setup.bash

# 启动导航
ros2 launch start_orca_nav start_orca_nav.launch.py \
    map_name:=corridor_swap \
    robot_number:=4 \
    use_rviz:=true
