#!/bin/bash
# 自动给所有机器人发送目标点
# 使用corridor_swap地图的合适位置

ROBOT_NUM=${1:-4}

echo "========================================"
echo "Sending Goals to $ROBOT_NUM Robots"
echo "========================================"
echo ""

# 激活环境
source /home/wj/anaconda3/etc/profile.d/conda.sh
conda activate ros2
source install/setup.bash

# 为每个机器人定义不同的目标点
# 这些坐标是corridor_swap地图的有效位置

declare -a GOALS=(
    # robot0: 左上角
    "-6.5 -3.8"
    # robot1: 右上角  
    "6.7 6.2"
    # robot2: 左下角
    "-2.6 -6.1"
    # robot3: 右下角
    "-2.4 -7.5"
)

echo "📍 Sending goals to robots..."
echo ""

for i in $(seq 0 $((ROBOT_NUM-1))); do
    GOAL=(${GOALS[$i]})
    X=${GOAL[0]}
    Y=${GOAL[1]}
    
    echo "Robot $i → Goal: ($X, $Y)"
    
    ros2 topic pub --once /robot${i}/goal_pose geometry_msgs/msg/PoseStamped "{
      header: {
        stamp: {sec: 0, nanosec: 0},
        frame_id: 'map'
      },
      pose: {
        position: {x: $X, y: $Y, z: 0.0},
        orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
      }
    }" &
    
    sleep 0.5
done

wait

echo ""
echo "✅ All goals sent!"
echo ""
echo "💡 Tips:"
echo "  - Check RViz to see goal markers (red, green, blue, yellow)"
echo "  - Watch ORCA node logs for path planning messages"
echo "  - Robots should start moving towards their goals"
echo ""
echo "To monitor navigation:"
echo "  ros2 topic echo /robot0/goal_pose"
echo "  ros2 topic echo /my_bot0/cmd_vel"
