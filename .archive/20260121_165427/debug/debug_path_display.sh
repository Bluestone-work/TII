#!/bin/bash
# 调试脚本 - 检查路径显示问题

source /home/wj/anaconda3/etc/profile.d/conda.sh
conda activate ros2
cd /home/wj/work/multi-robot-exploration-rl
source install/setup.bash

echo "=========================================="
echo "检查Gazebo路径显示问题"
echo "=========================================="
echo ""

echo "1. 检查Gazebo服务是否可用："
ros2 service list | grep -E "(spawn|delete)_entity"
echo ""

echo "2. 检查导航节点日志（路径相关）："
echo "   等待5秒..."
sleep 5
ros2 topic echo /robot0/goal_pose --once &
sleep 2
echo ""

echo "3. 查看Gazebo中的实体："
gz model -l 2>/dev/null || gazebo model -l 2>/dev/null || echo "   Gazebo命令行工具不可用"
echo ""

echo "4. 检查导航节点是否在运行："
ros2 node list | grep orca
echo ""

echo "完成！如果路径没显示，请查看上述输出找出问题。"
