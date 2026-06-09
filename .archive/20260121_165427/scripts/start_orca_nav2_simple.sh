#!/bin/bash
# 简化版启动脚本：假设Gazebo已经在运行，只启动ORCA控制节点

# 默认参数
MAP_NAME="corridor_swap"
ROBOT_NUM=4
USE_RVIZ="true"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--map)
            MAP_NAME="$2"
            shift 2
            ;;
        -r|--robots)
            ROBOT_NUM="$2"
            shift 2
            ;;
        --no-rviz)
            USE_RVIZ="false"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -m, --map NAME      Map name (default: corridor_swap)"
            echo "  -r, --robots NUM    Number of robots (default: 4)"
            echo "  --no-rviz           Don't launch RViz"
            echo "  -h, --help          Show this help"
            echo ""
            echo "⚠️  Prerequisites:"
            echo "  1. Gazebo must be running with robots"
            echo "  2. Each robot should have Nav2 stack running (optional)"
            echo ""
            echo "To start complete system, use start_orca_nav.sh first"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAP_FILE="${SCRIPT_DIR}/src/start_rl_environment/maps/${MAP_NAME}.yaml"

echo "========================================"
echo "ORCA Navigation (Simple Mode)"
echo "========================================"
echo "Map: $MAP_NAME"
echo "Robots: $ROBOT_NUM"
echo "RViz: $USE_RVIZ"
echo ""
echo "⚠️  Make sure Gazebo is already running!"
echo ""

# 检查Gazebo是否在运行
if ! pgrep -x "gzserver" > /dev/null; then
    echo "❌ Error: Gazebo is not running!"
    echo ""
    echo "Please start Gazebo first with:"
    echo "  ./start_gazebo.sh -m $MAP_NAME -r $ROBOT_NUM"
    echo ""
    echo "Or use the complete launch script:"
    echo "  ./start_orca_nav.sh -m 3 -r $ROBOT_NUM"
    exit 1
fi

echo "✅ Gazebo is running"
echo ""

# 激活ROS2环境
source /home/wj/anaconda3/etc/profile.d/conda.sh
conda activate ros2
source install/setup.bash

# 启动ORCA导航节点（不包含Nav2栈）
ros2 launch start_orca_nav orca_nav2_simple.launch.py \
    robot_number:=$ROBOT_NUM \
    map_file:=$MAP_FILE \
    use_rviz:=$USE_RVIZ
