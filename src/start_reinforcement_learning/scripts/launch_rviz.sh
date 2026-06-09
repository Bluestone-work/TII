#!/bin/bash

# 启动RViz并加载多机器人奖励可视化配置

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_FILE="$SCRIPT_DIR/../config/multi_robot_rewards.rviz"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 启动RViz
echo "启动RViz，配置文件: $CONFIG_FILE"
rviz2 -d "$CONFIG_FILE"
