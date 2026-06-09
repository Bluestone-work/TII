#!/bin/bash

##############################################################################
# ORCA导航测试脚本 - 修复版
# 功能：测试ORCA和Nav2两种模式，验证bug修复
##############################################################################

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         ORCA导航测试 - Bug修复验证                           ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# 切换到工作目录
cd /home/wj/work/multi-robot-exploration-rl

# 清理旧进程
echo -e "${YELLOW}清理旧进程...${NC}"
./kill_all_ros.sh
sleep 2

# 选择测试模式
echo ""
echo -e "${CYAN}选择测试模式:${NC}"
echo "  1) ORCA模式 (推荐)"
echo "  2) Nav2模式 (需要Nav2 Stack)"
echo ""
read -p "请选择 [1-2]: " choice

case $choice in
    1)
        MODE="orca"
        echo -e "${GREEN}测试模式: ORCA${NC}"
        ;;
    2)
        MODE="nav2"
        echo -e "${YELLOW}测试模式: Nav2 (启动Nav2 Stack)${NC}"
        ;;
    *)
        echo -e "${RED}无效选择，使用默认ORCA模式${NC}"
        MODE="orca"
        ;;
esac

# 启动导航
echo ""
echo -e "${CYAN}启动导航系统...${NC}"
./start_orca_nav.sh -m 3 -r 1 --mode $MODE &

# 等待系统启动
echo -e "${YELLOW}等待系统初始化（30秒）...${NC}"
sleep 30

# 检查进程状态
echo ""
echo -e "${CYAN}检查进程状态:${NC}"
if pgrep -f "gzserver" > /dev/null; then
    echo -e "${GREEN}✓ Gazebo运行中${NC}"
else
    echo -e "${RED}✗ Gazebo未运行${NC}"
fi

if pgrep -f "orca_nav_node" > /dev/null; then
    echo -e "${GREEN}✓ ORCA导航节点运行中${NC}"
else
    echo -e "${RED}✗ ORCA导航节点未运行${NC}"
fi

# 检查话题
echo ""
echo -e "${CYAN}检查ROS2话题:${NC}"
echo -e "${YELLOW}机器人话题:${NC}"
ros2 topic list | grep my_bot0 | head -5

# 测试odom数据
echo ""
echo -e "${CYAN}测试odom数据接收:${NC}"
timeout 3 ros2 topic echo /my_bot0/odom --once > /tmp/odom_test.txt 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Odom数据正常接收${NC}"
    cat /tmp/odom_test.txt | grep -A 3 "position:"
else
    echo -e "${RED}✗ Odom数据未接收（3秒超时）${NC}"
    echo -e "${YELLOW}这可能说明机器人spawn有问题${NC}"
fi

# 查看最新日志
echo ""
echo -e "${CYAN}最新导航日志 (最后20行):${NC}"
LATEST_LOG=$(ls -t orca_logs/navigation_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo -e "${YELLOW}文件: $LATEST_LOG${NC}"
    tail -20 "$LATEST_LOG"
else
    echo -e "${RED}未找到日志文件${NC}"
fi

# 发送测试目标
echo ""
echo -e "${CYAN}发送测试目标点...${NC}"
ros2 topic pub --once /robot0/goal_pose geometry_msgs/msg/PoseStamped \
    "{header: {frame_id: 'map'}, pose: {position: {x: 5.0, y: 4.0, z: 0.0}}}"

echo ""
echo -e "${GREEN}等待10秒观察机器人行为...${NC}"
sleep 10

# 再次查看日志
echo ""
echo -e "${CYAN}发送目标后的日志 (最后30行):${NC}"
tail -30 "$LATEST_LOG" | grep -E "goal|waypoint|ORCA|control_loop|odom"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  测试完成！请检查上方输出确认bug是否修复                    ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}提示:${NC}"
echo "  - 如果看到 'odom received' 日志，说明ORCA模式odom接收正常"
echo "  - 如果看到 'waypoint' 和 'ORCA_vel' 日志，说明控制循环正常运行"
echo "  - 如果机器人在Gazebo中移动，说明导航成功"
echo ""
echo -e "${CYAN}停止所有进程: ${NC}./kill_all_ros.sh"
