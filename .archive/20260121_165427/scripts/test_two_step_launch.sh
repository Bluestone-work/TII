#!/bin/bash

# Quick test of two-step launch pattern (stage 1, 5 episodes)
# Tests: single Gazebo, single RViz, all robots controllable

echo "=== Two-Step Launch Test ==="
echo "Stage: 1 (1 robot, map1)"
echo "Episodes: 5"
echo ""

# Set ROS2 workspace
source /home/wj/work/multi-robot-exploration-rl/install/setup.bash

# Step 1: Launch environment
echo "[1/2] Launching Gazebo environment..."
ros2 launch start_rl_environment main.launch.py \
    map_number:=1 \
    num_robots:=1 \
    use_rviz:=true \
    use_distance_field:=true &
GAZEBO_PID=$!
echo "Environment PID: $GAZEBO_PID"

# Step 2: Wait for initialization
echo "[2/2] Waiting for environment (15s)..."
sleep 15

# Verify /map topic
echo "Verifying /map topic..."
if ros2 topic list | grep -q "/map"; then
    echo "✓ /map topic available"
else
    echo "✗ /map topic not found (environment may not be ready)"
fi

# Launch training
echo ""
echo "Starting training (5 episodes)..."
ros2 launch start_reinforcement_learning start_learning.launch.py \
    max_episodes:=5 \
    num_robots:=1 \
    use_random_mode:=true \
    use_distance_field:=true

# Cleanup
echo ""
echo "Cleaning up..."
kill $GAZEBO_PID
sleep 2
killall -9 gzserver gzclient rviz2 2>/dev/null

echo "Test complete!"
