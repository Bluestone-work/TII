#!/bin/bash
# 快速测试Nav2集成

echo "========================================"
echo "Testing Nav2 Integration"
echo "========================================"
echo ""

# 1. Check if Nav2 packages are installed
echo "1. Checking Nav2 packages..."
MISSING_PKGS=""

for pkg in nav2_planner nav2_map_server nav2_lifecycle_manager; do
    if ! ros2 pkg list | grep -q "^${pkg}$"; then
        MISSING_PKGS="$MISSING_PKGS $pkg"
    fi
done

if [ -n "$MISSING_PKGS" ]; then
    echo "❌ Missing Nav2 packages:$MISSING_PKGS"
    echo ""
    echo "Install with:"
    echo "  sudo apt update"
    echo "  sudo apt install ros-humble-nav2-bringup"
    exit 1
else
    echo "✅ Nav2 packages found"
fi

echo ""

# 2. Check map files
echo "2. Checking map files..."
MAP_DIR="./src/start_rl_environment/maps"
if [ -d "$MAP_DIR" ]; then
    MAP_COUNT=$(ls -1 "$MAP_DIR"/*.yaml 2>/dev/null | wc -l)
    echo "✅ Found $MAP_COUNT map files:"
    ls -1 "$MAP_DIR"/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/^/     - /'
else
    echo "❌ Map directory not found: $MAP_DIR"
    exit 1
fi

echo ""

# 3. Check if package is built
echo "3. Checking package build..."
if [ -f "./install/start_orca_nav/lib/start_orca_nav/orca_nav_node_nav2" ]; then
    echo "✅ orca_nav_node_nav2 executable found"
else
    echo "❌ orca_nav_node_nav2 not built"
    echo ""
    echo "Build with:"
    echo "  colcon build --packages-select start_orca_nav"
    exit 1
fi

echo ""

# 4. Check launch files
echo "4. Checking launch files..."
LAUNCH_DIR="./src/start_orca_nav/launch"
for launch in orca_nav2.launch.py nav2_multi_robot.launch.py; do
    if [ -f "$LAUNCH_DIR/$launch" ]; then
        echo "  ✅ $launch"
    else
        echo "  ❌ $launch missing"
    fi
done

echo ""
echo "========================================"
echo "✅ All checks passed!"
echo "========================================"
echo ""
echo "Ready to start:"
echo "  ./start_orca_nav2.sh -m corridor_swap -r 4"
echo ""
echo "Send goals with:"
echo "  ros2 topic pub --once /robot0/goal_pose geometry_msgs/msg/PoseStamped \\"
echo "    '{header: {frame_id: map}, pose: {position: {x: 5.0, y: 3.0}, orientation: {w: 1.0}}}'"
