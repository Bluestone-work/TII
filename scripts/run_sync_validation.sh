#!/bin/bash
# 快速启动数据同步验证工具

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          数据同步验证工具 - 快速启动                          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# 检查ROS2环境
if [ -z "$ROS_DISTRO" ]; then
    echo -e "${YELLOW}⚠️  未检测到ROS2环境，尝试source...${NC}"
    if [ -f "/opt/ros/humble/setup.bash" ]; then
        source /opt/ros/humble/setup.bash
        echo -e "${GREEN}✅ 已加载ROS2 Humble${NC}"
    else
        echo -e "${RED}❌ 找不到ROS2安装，请手动source${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ 检测到ROS2: $ROS_DISTRO${NC}"
fi

# 获取机器人数量
ROBOT_COUNT=${1:-3}
echo -e "${BLUE}🤖 机器人数量: $ROBOT_COUNT${NC}"
echo ""

# 功能菜单
echo -e "${GREEN}请选择要运行的工具:${NC}"
echo "  1) TF树监控器 (单次检查)"
echo "  2) TF树监控器 (持续监控)"
echo "  3) 数据同步可视化器 (RViz)"
echo "  4) 全套工具 (TF监控 + 可视化 + RViz)"
echo "  5) 查看使用文档"
echo "  q) 退出"
echo ""
read -p "输入选项 [1-5/q]: " choice

case $choice in
    1)
        echo -e "${BLUE}🔍 启动TF树监控器 (单次检查)...${NC}"
        cd "$WORKSPACE_ROOT"
        python3 src/start_reinforcement_learning/start_reinforcement_learning/env_logic/tf_monitor.py $ROBOT_COUNT
        ;;
    
    2)
        echo -e "${BLUE}🔍 启动TF树监控器 (持续监控)...${NC}"
        cd "$WORKSPACE_ROOT"
        python3 src/start_reinforcement_learning/start_reinforcement_learning/env_logic/tf_monitor.py $ROBOT_COUNT <<< "y"
        ;;
    
    3)
        echo -e "${BLUE}🎨 启动数据同步可视化器...${NC}"
        cd "$WORKSPACE_ROOT"
        
        # 为每个机器人启动一个实例
        for ((i=0; i<$ROBOT_COUNT; i++)); do
            echo -e "${GREEN}  启动 Robot $i 可视化器${NC}"
            python3 src/start_reinforcement_learning/start_reinforcement_learning/env_logic/sync_visualizer.py $i &
            PIDS[$i]=$!
            sleep 0.5
        done
        
        echo ""
        echo -e "${YELLOW}✅ 所有可视化器已启动${NC}"
        echo -e "${YELLOW}   在RViz中添加MarkerArray话题:${NC}"
        for ((i=0; i<$ROBOT_COUNT; i++)); do
            echo -e "${YELLOW}     - /robot${i}/sync_visualization${NC}"
        done
        echo ""
        echo -e "${BLUE}📍 启动RViz...${NC}"
        rviz2 &
        RVIZ_PID=$!
        
        echo ""
        echo -e "${RED}按 Ctrl+C 停止所有进程${NC}"
        
        # 等待用户中断
        wait $RVIZ_PID
        
        # 清理
        echo -e "${YELLOW}🧹 清理进程...${NC}"
        for pid in "${PIDS[@]}"; do
            kill $pid 2>/dev/null
        done
        ;;
    
    4)
        echo -e "${BLUE}🚀 启动全套工具...${NC}"
        cd "$WORKSPACE_ROOT"
        
        # 启动TF监控器（持续模式）
        echo -e "${GREEN}1/3: 启动TF监控器${NC}"
        gnome-terminal --title="TF Monitor" -- bash -c "
            cd '$WORKSPACE_ROOT' && \
            python3 src/start_reinforcement_learning/start_reinforcement_learning/env_logic/tf_monitor.py $ROBOT_COUNT <<< 'y'; \
            exec bash
        " 2>/dev/null || xterm -title "TF Monitor" -e "
            cd '$WORKSPACE_ROOT' && \
            python3 src/start_reinforcement_learning/start_reinforcement_learning/env_logic/tf_monitor.py $ROBOT_COUNT <<< 'y'; \
            exec bash
        " &
        
        sleep 2
        
        # 启动可视化器
        echo -e "${GREEN}2/3: 启动可视化器${NC}"
        for ((i=0; i<$ROBOT_COUNT; i++)); do
            python3 src/start_reinforcement_learning/start_reinforcement_learning/env_logic/sync_visualizer.py $i &
            sleep 0.5
        done
        
        sleep 2
        
        # 启动RViz
        echo -e "${GREEN}3/3: 启动RViz${NC}"
        rviz2 &
        
        echo ""
        echo -e "${GREEN}✅ 全套工具已启动！${NC}"
        echo -e "${YELLOW}   - TF监控器: 在新终端窗口中${NC}"
        echo -e "${YELLOW}   - 可视化器: 后台运行${NC}"
        echo -e "${YELLOW}   - RViz: 已启动${NC}"
        echo ""
        echo -e "${BLUE}📋 下一步:${NC}"
        echo "   1. 在RViz中添加MarkerArray话题"
        echo "   2. 启动你的训练脚本"
        echo "   3. 观察数据同步状态"
        ;;
    
    5)
        echo -e "${BLUE}📚 打开使用文档...${NC}"
        if [ -f "$WORKSPACE_ROOT/docs/DATA_SYNC_VALIDATION_GUIDE.md" ]; then
            if command -v code &> /dev/null; then
                code "$WORKSPACE_ROOT/docs/DATA_SYNC_VALIDATION_GUIDE.md"
            elif command -v gedit &> /dev/null; then
                gedit "$WORKSPACE_ROOT/docs/DATA_SYNC_VALIDATION_GUIDE.md" &
            else
                cat "$WORKSPACE_ROOT/docs/DATA_SYNC_VALIDATION_GUIDE.md"
            fi
        else
            echo -e "${RED}❌ 找不到文档文件${NC}"
        fi
        ;;
    
    q|Q)
        echo -e "${YELLOW}👋 再见！${NC}"
        exit 0
        ;;
    
    *)
        echo -e "${RED}❌ 无效选项${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✨ 完成！${NC}"
