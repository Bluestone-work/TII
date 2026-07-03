#!/bin/bash
# 快速诊断脚本 - 检查路径穿过障碍物的原因

echo "================================================================================"
echo "                  路径穿过障碍物问题 - 快速诊断"
echo "================================================================================"

LOG_FILE="${1:-debug.log}"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ 未找到日志文件: $LOG_FILE"
    echo ""
    echo "请运行训练并保存日志:"
    echo "  ./run_curriculum.sh 2>&1 | tee debug.log"
    echo ""
    echo "然后执行:"
    echo "  ./diagnose_path_through_obstacles.sh debug.log"
    exit 1
fi

echo ""
echo "1. 检查障碍物spawn情况"
echo "----------------------------------------"
grep "成功spawn.*棕色方块" "$LOG_FILE" | tail -5

echo ""
echo "2. 检查blocked_points数量"
echo "----------------------------------------"
grep "blocked_points=" "$LOG_FILE" | tail -10

echo ""
echo "3. 检查规划器是否被调用"
echo "----------------------------------------"
grep "🔧.*plan_with_dynamic_obstacles 被调用" "$LOG_FILE" | head -5

echo ""
echo "4. 检查障碍物膨胀参数"
echo "----------------------------------------"
grep "🗺️.*动态障碍物膨胀" "$LOG_FILE" | head -5

echo ""
echo "5. 检查障碍物坐标转换"
echo "----------------------------------------"
grep "障碍物\[.*\]:" "$LOG_FILE" | head -10

echo ""
echo "6. 检查规划结果"
echo "----------------------------------------"
grep -E "✅.*A\*规划成功|❌.*A\* plan 失败" "$LOG_FILE" | tail -10

echo ""
echo "7. 检查blocked_points为空的情况"
echo "----------------------------------------"
grep -A 10 "blocked_points为空" "$LOG_FILE" | head -20

echo ""
echo "================================================================================"
echo "                              诊断建议"
echo "================================================================================"

# 统计
SPAWN_COUNT=$(grep -c "成功spawn.*棕色方块" "$LOG_FILE")
BLOCKED_ZERO=$(grep -c "blocked_points=0" "$LOG_FILE")
BLOCKED_NONZERO=$(grep "blocked_points=[1-9]" "$LOG_FILE" | wc -l)
PLAN_SUCCESS=$(grep -c "A\*规划成功" "$LOG_FILE")
PLAN_FAIL=$(grep -c "A\* plan 失败" "$LOG_FILE")

echo ""
echo "统计信息:"
echo "  障碍物spawn次数: $SPAWN_COUNT"
echo "  blocked_points=0 次数: $BLOCKED_ZERO"
echo "  blocked_points>0 次数: $BLOCKED_NONZERO"
echo "  规划成功次数: $PLAN_SUCCESS"
echo "  规划失败次数: $PLAN_FAIL"

echo ""
if [ $BLOCKED_ZERO -gt 0 ]; then
    echo "⚠️  发现blocked_points为空的情况 ($BLOCKED_ZERO 次)"
    echo "   → 这是主要问题！规划器没有收到障碍物信息"
    echo "   → 检查 parent_env 和 spawned_static_obstacles 的同步"
fi

if [ $PLAN_FAIL -gt 0 ]; then
    echo "⚠️  发现规划失败 ($PLAN_FAIL 次)"
    echo "   → 规划失败会退化为直线，导致穿过障碍物"
    echo "   → 可能原因: 起点/终点被障碍物覆盖，或无可行路径"
fi

if [ $PLAN_SUCCESS -gt 0 ] && [ $BLOCKED_NONZERO -gt 0 ]; then
    echo "✅ 有成功的规划 ($PLAN_SUCCESS 次)，且blocked_points非空"
    echo "   → 如果仍然穿过障碍物，可能是:"
    echo "   → 1. waypoint简化时出错"
    echo "   → 2. 重规划时没有考虑静态障碍物"
    echo "   → 3. rviz可视化延迟"
fi

echo ""
echo "================================================================================"
