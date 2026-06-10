#!/bin/bash
# 手动发送单个机器人的目标点
# Usage: ./send_single_goal.sh <robot_id> <x> <y>

ROBOT_ID=${1:-0}
X=${2:-5.0}
Y=${3:-3.0}

if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: $0 <robot_id> <x> <y>"
    echo ""
    echo "Examples:"
    echo "  $0 0 5.0 3.0    # Send robot0 to (5.0, 3.0)"
    echo "  $0 1 -5.0 -3.0  # Send robot1 to (-5.0, -3.0)"
    echo ""
    echo "Corridor_swap map valid range: x[-10, 10], y[-10, 10]"
    exit 0
fi

echo "Sending goal to robot$ROBOT_ID: ($X, $Y)"

source /home/wj/anaconda3/etc/profile.d/conda.sh
conda activate ros2
source install/setup.bash

ros2 topic pub --once /robot${ROBOT_ID}/goal_pose geometry_msgs/msg/PoseStamped "{
  header: {frame_id: 'map'},
  pose: {
    position: {x: $X, y: $Y, z: 0.0},
    orientation: {w: 1.0}
  }
}"

echo "✅ Goal sent to robot$ROBOT_ID"
