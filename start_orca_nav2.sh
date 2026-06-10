#!/bin/bash
# 启动使用Nav2全局规划的ORCA导航系统

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
            echo "Available maps: corridor_swap, intersection, map1, map2, warehouse_aisles"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# 设置地图文件路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAP_FILE="${SCRIPT_DIR}/src/start_rl_environment/maps/${MAP_NAME}.yaml"

# 检查地图文件是否存在
if [ ! -f "$MAP_FILE" ]; then
    echo "❌ Error: Map file not found: $MAP_FILE"
    echo "Available maps in src/start_rl_environment/maps/:"
    ls -1 "${SCRIPT_DIR}/src/start_rl_environment/maps/"*.yaml 2>/dev/null | xargs -n1 basename | sed 's/.yaml//'
    exit 1
fi

echo "========================================"
echo "ORCA Navigation with Nav2 Global Planner"
echo "========================================"
echo "Map: $MAP_NAME"
echo "Robots: $ROBOT_NUM"
echo "RViz: $USE_RVIZ"
echo "Map file: $MAP_FILE"
echo ""
echo "Architecture:"
echo "  1. Gazebo + RL Environment (with Nav2 stack)"
echo "  2. ORCA Multi-Robot Avoidance"
echo "  3. DWA Local Planner (laser-based)"
echo ""
echo "Starting in 2 seconds..."
echo "========================================"
echo ""

sleep 2

# 激活ROS2环境
source /home/wj/anaconda3/etc/profile.d/conda.sh
conda activate ros2
source install/setup.bash

# 首先启动Gazebo和RL环境（包含Nav2栈）
echo "Step 1: Starting Gazebo and RL Environment..."

# 将map_name转换为map_number
case $MAP_NAME in
    "map1") MAP_NUM=1 ;;
    "map2") MAP_NUM=2 ;;
    "corridor_swap") MAP_NUM=3 ;;
    "intersection") MAP_NUM=4 ;;
    "warehouse_aisles") MAP_NUM=5 ;;
    *) MAP_NUM=3 ;;  # 默认使用corridor_swap
esac

ros2 launch start_rl_environment main.launch.py \
    map_number:=$MAP_NUM \
    robot_number:=$ROBOT_NUM &

GAZEBO_PID=$!

# 等待Gazebo启动
echo "Waiting for Gazebo to start..."
sleep 15

# 检查Gazebo是否成功启动
if ! pgrep -x "gzserver" > /dev/null; then
    echo "❌ Error: Gazebo failed to start!"
    exit 1
fi

echo "✅ Gazebo started"
echo ""
echo "Step 2: Starting ORCA Navigation Node..."

# 启动ORCA导航节点（使用Nav2的规划服务）
ros2 launch start_orca_nav orca_nav2_simple.launch.py \
    robot_number:=$ROBOT_NUM \
    map_file:=$MAP_FILE \
    use_rviz:=$USE_RVIZ

# 清理
kill $GAZEBO_PID 2>/dev/null
