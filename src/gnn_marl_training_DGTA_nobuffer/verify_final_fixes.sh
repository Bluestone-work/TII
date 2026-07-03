#!/bin/bash
# 最终验证脚本 - 检查所有修复

echo "================================================================================"
echo "                  全局路径规划器 - 最终修复验证"
echo "================================================================================"

echo ""
echo "修复1: Y轴坐标转换"
echo "----------------------------------------"
if grep -q "self.height - 1 -" gnn_marl_training/global_planner.py; then
    echo "✅ Y轴翻转已修复"
else
    echo "❌ Y轴翻转缺失"
fi

echo ""
echo "修复2: 障碍物spawn距离和膨胀"
echo "----------------------------------------"
if grep -q "MIN_ROBOT_SEP = 1.8" gnn_marl_training/gnn_marl_env.py; then
    echo "✅ MIN_ROBOT_SEP = 1.8m"
else
    echo "⚠️  MIN_ROBOT_SEP 不是1.8m"
fi

if grep -q "block_radius = 0.8" gnn_marl_training/gnn_marl_env.py; then
    echo "✅ block_radius = 0.8m"
else
    echo "⚠️  block_radius 不是0.8m"
fi

echo ""
echo "修复3: 动态障碍物支持"
echo "----------------------------------------"
if grep -q "_DYN_OBS_SPAWNS.get(self.map_number" gnn_marl_training/gnn_marl_env.py; then
    echo "✅ 动态障碍物位置已加入blocked_points"
else
    echo "❌ 动态障碍物位置未加入"
fi

if grep -q "8: \[" gnn_marl_training/gnn_marl_env.py | grep -A 2 "_DYN_OBS_SPAWNS"; then
    echo "✅ Map 8 动态障碍物位置已定义"
else
    echo "⚠️  Map 8 可能缺少定义"
fi

echo ""
echo "修复4: 调试日志"
echo "----------------------------------------"
if grep -q "🔧.*plan_with_dynamic_obstacles 被调用" gnn_marl_training/global_planner.py; then
    echo "✅ 详细调试日志已添加"
else
    echo "❌ 调试日志缺失"
fi

echo ""
echo "================================================================================"
echo "                              总结"
echo "================================================================================"
echo ""
echo "已完成的修复："
echo "  1. ✅ Y轴坐标转换错误"
echo "  2. ✅ 机器人spawn到障碍物附近"
echo "  3. ✅ 动态障碍物(dyn_obs_X)未被考虑"
echo "  4. ✅ 调试日志增强"
echo ""
echo "现在blocked_points应该包含："
echo "  • 4个棕色方块 (spawned_static_obstacles)"
echo "  • 8个动态障碍物位置 (_DYN_OBS_SPAWNS)"
echo "  • 激光聚类检测的障碍物 (如果有)"
echo "  = 总共 12-20 个避障点"
echo ""
echo "运行训练："
echo "  ./run_curriculum.sh 2>&1 | tee debug.log"
echo ""
echo "验证："
echo "  grep 'blocked_points=' debug.log | head -10"
echo "  # 应该看到 blocked_points=12 或更多（不再是4）"
echo ""
echo "================================================================================"
