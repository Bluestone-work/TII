#!/bin/bash
# 调试脚本 - 查看路径可视化话题

source /home/wj/anaconda3/etc/profile.d/conda.sh
conda activate ros2
source install/setup.bash

echo "=========================================="
echo "检查路径可视化话题"
echo "=========================================="
echo ""

echo "1. 检查marker话题是否存在："
ros2 topic list | grep -E "(path_marker|goal_marker)"
echo ""

echo "2. 检查robot0的路径marker (按Ctrl+C停止)："
echo ""
ros2 topic echo /robot0/path_marker --once
