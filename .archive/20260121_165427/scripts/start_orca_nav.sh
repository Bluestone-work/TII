#!/bin/bash

##############################################################################
# ORCA 多机器人导航启动脚本 (极速优化版)
# 核心改进：智能检测替代固定等待，大幅缩短启动时间
##############################################################################

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        ORCA 多机器人导航 - Multi-Robot Navigation            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# === 默认参数配置 ===
MAP_NUM=3               # 默认地图: 走廊交换
ROBOT_NUM=4             # 默认机器人数量
ROBOT_RADIUS=0.35       # 机器人半径
MAX_LINEAR_SPEED=0.8    # [优化] 默认速度改为 0.8，避免移动过慢
MAX_ANGULAR_SPEED=2.0   # 最大角速度
NEIGHBOR_DISTANCE=5.0   # 感知距离
TIME_HORIZON=2.0        # 避障预判时间
NAVIGATION_MODE=orca    # 模式: orca 或 nav2
GOAL_TOLERANCE=0.3      # 到达阈值

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--map) MAP_NUM="$2"; shift 2 ;;
        -r|--robots) ROBOT_NUM="$2"; shift 2 ;;
        --radius) ROBOT_RADIUS="$2"; shift 2 ;;
        --max-speed) MAX_LINEAR_SPEED="$2"; shift 2 ;;
        --mode) NAVIGATION_MODE="$2"; shift 2 ;;
        --neighbor-distance) NEIGHBOR_DISTANCE="$2"; shift 2 ;;
        --time-horizon) TIME_HORIZON="$2"; shift 2 ;;
        --goal-tolerance) GOAL_TOLERANCE="$2"; shift 2 ;;
        -h|--help)
            echo "用法: $0 [选项]"
            exit 0 ;;
        *) echo -e "${RED}未知参数: $1${NC}"; exit 1 ;;
    esac
done

# Nav2 模式警告
if [ "$NAVIGATION_MODE" = "nav2" ]; then
    echo -e "${RED}⚠️  警告: Nav2 模式需要额外的 Nav2 Stack 运行！${NC}"
    echo -e "${YELLOW}当前环境未自动启动 Nav2 Bringup，请确保已手动启动。${NC}"
    sleep 1 # [优化] 缩短警告时间
fi

# 地图选择
case $MAP_NUM in
    1) MAP_FILE="map1.yaml"; MAP_NAME="开放空间" ;;
    2) MAP_FILE="map2.yaml"; MAP_NAME="复杂环境" ;;
    3) MAP_FILE="corridor_swap.yaml"; MAP_NAME="走廊交换" ;;
    4) MAP_FILE="intersection.yaml"; MAP_NAME="十字路口" ;;
    5) MAP_FILE="warehouse_aisles.yaml"; MAP_NAME="仓库过道" ;;
    *) echo -e "${RED}错误: 无效的地图编号 $MAP_NUM${NC}"; exit 1 ;;
esac

# 切换工作目录并 Source 环境
cd /home/wj/work/multi-robot-exploration-rl
source install/setup.bash

echo -e "${GREEN}📋 配置摘要:${NC} 地图=${YELLOW}$MAP_NAME${NC} | 机器人=${YELLOW}$ROBOT_NUM${NC} | 速度=${YELLOW}$MAX_LINEAR_SPEED${NC} | 模式=${YELLOW}$NAVIGATION_MODE${NC}"

# 创建日志目录
mkdir -p orca_logs

# === 快速预清理 ===
# [优化] 不再显示清理过程，后台静默执行
pkill -9 -f "ros2-daemon" 2>/dev/null || true
pkill -9 -f "gz sim" 2>/dev/null || true
pkill -9 -f "gzserver" 2>/dev/null || true

# === 步骤 1: 启动 Gazebo ===
echo -e "${YELLOW}[1/3] 启动 Gazebo 仿真环境...${NC}"
ros2 launch start_rl_environment main.launch.py \
    map_number:=$MAP_NUM \
    robot_number:=$ROBOT_NUM \
    > orca_logs/gazebo_$(date +%Y%m%d_%H%M%S).log 2>&1 &
GAZEBO_PID=$!

# === [核心优化] 智能等待机器人就绪 ===
echo -e "${YELLOW}⏳ 正在检测机器人话题 (智能等待)...${NC}"
MAX_RETRIES=40
ALL_READY=false

# 预先 source 确保环境正常
source install/setup.bash

for ((i=1; i<=MAX_RETRIES; i++)); do
    # 快速获取当前所有话题
    TOPIC_LIST=$(ros2 topic list 2>/dev/null || true)
    
    READY_COUNT=0
    for j in $(seq 0 $((ROBOT_NUM-1))); do
        # 检测里程计话题是否存在
        if echo "$TOPIC_LIST" | grep -q "/my_bot$j/odom"; then
            READY_COUNT=$((READY_COUNT+1))
        fi
    done
    
    # 动态进度显示
    PERCENT=$((READY_COUNT * 100 / ROBOT_NUM))
    echo -ne "\r   检测中 [${i}s]: ${CYAN}$READY_COUNT/$ROBOT_NUM${NC} 机器人 (${PERCENT}%)   "
    
    if [ $READY_COUNT -eq $ROBOT_NUM ]; then
        echo -e "\n${GREEN}✓ 所有机器人已就绪！(仅耗时 ${i}s)${NC}"
        ALL_READY=true
        break
    fi
    
    sleep 1
done

if [ "$ALL_READY" = false ]; then
    echo -e "\n${RED}⚠️  检测超时: 部分机器人可能未启动成功，尝试强行继续...${NC}"
fi

# === 步骤 2: 启动导航服务 ===
echo -e "${YELLOW}[2/3] 启动导航服务 (模式: ${GREEN}$NAVIGATION_MODE${YELLOW})...${NC}"

# 如果是Nav2模式，启动Nav2 Stack
if [ "$NAVIGATION_MODE" = "nav2" ]; then
    echo -e "${CYAN}启动 Nav2 Stack (Planner + Controller)...${NC}"
    
    # 为每个机器人启动Nav2 Bringup
    for i in $(seq 0 $((ROBOT_NUM-1))); do
        NAMESPACE="my_bot$i"
        echo -e "  启动 Nav2 for ${NAMESPACE}..."
        
        ros2 launch nav2_bringup navigation_launch.py \
            namespace:=$NAMESPACE \
            use_sim_time:=true \
            > orca_logs/nav2_${NAMESPACE}_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    done
    
    echo -e "${GREEN}✓ Nav2 Stack 启动完成，等待服务就绪...${NC}"
    sleep 3
fi

# 启动ORCA导航节点
ros2 launch start_orca_nav start_orca_nav.launch.py \
    robot_number:=$ROBOT_NUM \
    robot_radius:=$ROBOT_RADIUS \
    max_linear_speed:=$MAX_LINEAR_SPEED \
    max_angular_speed:=$MAX_ANGULAR_SPEED \
    neighbor_distance:=$NEIGHBOR_DISTANCE \
    time_horizon:=$TIME_HORIZON \
    goal_tolerance:=$GOAL_TOLERANCE \
    navigation_mode:=$NAVIGATION_MODE \
    use_rviz:=true \
    > orca_logs/navigation_$(date +%Y%m%d_%H%M%S).log 2>&1 &
ORCA_PID=$!

echo -e "${GREEN}✓ 导航节点已启动 (PID: $ORCA_PID)${NC}"
# [优化] 等待节点初始化，从 3s 缩短为 1s
sleep 1

# === 步骤 3: 发送测试目标 ===
echo -e "${CYAN}📍 正在分发测试目标点...${NC}"

for i in $(seq 0 $((ROBOT_NUM-1))); do
    # 生成随机目标 (根据地图1的安全范围大致生成)
    GOAL_X=$(awk -v seed=$RANDOM 'BEGIN{srand(seed); print -8.0 + rand()*16.0}')
    GOAL_Y=$(awk -v seed=$RANDOM 'BEGIN{srand(seed); print -8.0 + rand()*16.0}')
    
    echo -e "   ${CYAN}robot$i${NC} → [${GOAL_X::5}, ${GOAL_Y::5}]"
    
    # 后台异步发送，不阻塞主线程
    ros2 topic pub --once /robot$i/goal_pose geometry_msgs/msg/PoseStamped "{
      header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'map'},
      pose: {position: {x: $GOAL_X, y: $GOAL_Y, z: 0.0}, orientation: {w: 1.0}}
    }" > /dev/null 2>&1 &
    
    # [优化] 极速发送间隔，从 0.5s 缩短为 0.1s
    sleep 0.1
done

echo -e "${GREEN}✓ 系统启动完成！${NC}"
echo -e "${YELLOW}⚠️  按 Ctrl+C 停止所有进程${NC}"
echo ""
echo -e "${CYAN}🎨 机器人颜色对应：${NC}"
echo -e "   ${RED}robot0${NC} → 🔴 红色路径和目标点"
echo -e "   ${GREEN}robot1${NC} → 🟢 绿色路径和目标点"
echo -e "   ${CYAN}robot2${NC} → 🔵 蓝色路径和目标点"
echo -e "   ${YELLOW}robot3${NC} → 🟡 黄色路径和目标点"
echo ""

# === 进程守护 ===
wait $ORCA_PID

# === 退出清理 ===
echo -e "\n${YELLOW}🛑 正在清理环境...${NC}"
kill $ORCA_PID 2>/dev/null || true
kill $GAZEBO_PID 2>/dev/null || true
pkill -f "gz sim" 2>/dev/null || true
pkill -f "gzserver" 2>/dev/null || true
pkill -f "robot_state_publisher" 2>/dev/null || true

echo -e "${GREEN}再见！${NC}"