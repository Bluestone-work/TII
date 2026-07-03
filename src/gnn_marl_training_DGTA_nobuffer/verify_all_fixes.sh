#!/bin/bash
# 全局路径规划器修复验证脚本

echo "================================================================================"
echo "                    全局路径规划器修复验证"
echo "================================================================================"

echo ""
echo "检查修复是否已应用..."
echo ""

# 1. 检查Y轴坐标转换
echo "1. Y轴坐标转换修复:"
if grep -q "self.height - 1 -" gnn_marl_training/global_planner.py; then
    echo "   ✅ world_to_grid() 已添加Y轴翻转"
else
    echo "   ❌ world_to_grid() 缺少Y轴翻转"
fi

if grep -q "(self.height - 1 - grid_y)" gnn_marl_training/global_planner.py; then
    echo "   ✅ grid_to_world() 已添加Y轴翻转"
else
    echo "   ❌ grid_to_world() 缺少Y轴翻转"
fi

# 2. 检查MIN_ROBOT_SEP
echo ""
echo "2. 障碍物spawn距离:"
if grep -q "MIN_ROBOT_SEP = 1.8" gnn_marl_training/gnn_marl_env.py; then
    echo "   ✅ MIN_ROBOT_SEP = 1.8m (已从1.0m增加)"
else
    echo "   ⚠️  MIN_ROBOT_SEP 不是1.8m"
    grep "MIN_ROBOT_SEP" gnn_marl_training/gnn_marl_env.py | head -1
fi

# 3. 检查block_radius
echo ""
echo "3. 障碍物膨胀半径:"
if grep -q "block_radius = 0.8" gnn_marl_training/gnn_marl_env.py; then
    echo "   ✅ block_radius = 0.8m (已从1.0m减小)"
else
    echo "   ⚠️  block_radius 不是0.8m"
    grep "block_radius = " gnn_marl_training/gnn_marl_env.py | head -1
fi

# 4. 检查起点保护区域
echo ""
echo "4. 起点/终点保护区域:"
if grep -q "radius = 2.*# 从1增加到2" gnn_marl_training/global_planner.py; then
    echo "   ✅ 保护区域扩大到5x5 (radius=2)"
else
    echo "   ⚠️  保护区域可能未扩大"
fi

# 5. 检查调试日志
echo ""
echo "5. 调试日志增强:"
if grep -q "🔧 \[AStarPlanner\] plan_with_dynamic_obstacles 被调用" gnn_marl_training/global_planner.py; then
    echo "   ✅ 入口日志已添加"
else
    echo "   ❌ 入口日志缺失"
fi

if grep -q "flush=True" gnn_marl_training/global_planner.py; then
    echo "   ✅ 日志已添加 flush=True"
else
    echo "   ❌ 日志缺少 flush=True"
fi

# 6. 检查Python缓存
echo ""
echo "6. Python缓存:"
if find . -name "*.pyc" -o -name "__pycache__" | grep -q .; then
    echo "   ⚠️  发现Python缓存，建议清理"
    echo "   运行: find . -name '*.pyc' -delete && find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null"
else
    echo "   ✅ 无Python缓存"
fi

echo ""
echo "================================================================================"
echo "                          修复完成度检查"
echo "================================================================================"

PASS=0
FAIL=0

# 统计检查结果
if grep -q "self.height - 1 -" gnn_marl_training/global_planner.py && \
   grep -q "(self.height - 1 - grid_y)" gnn_marl_training/global_planner.py; then
    ((PASS++))
else
    ((FAIL++))
fi

if grep -q "MIN_ROBOT_SEP = 1.8" gnn_marl_training/gnn_marl_env.py; then
    ((PASS++))
else
    ((FAIL++))
fi

if grep -q "block_radius = 0.8" gnn_marl_training/gnn_marl_env.py; then
    ((PASS++))
else
    ((FAIL++))
fi

if grep -q "radius = 2.*# 从1增加到2" gnn_marl_training/global_planner.py; then
    ((PASS++))
else
    ((FAIL++))
fi

if grep -q "🔧 \[AStarPlanner\] plan_with_dynamic_obstacles 被调用" gnn_marl_training/global_planner.py; then
    ((PASS++))
else
    ((FAIL++))
fi

echo ""
echo "通过: $PASS/5"
echo "失败: $FAIL/5"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "✅ 所有修复已正确应用！可以运行训练。"
    echo ""
    echo "运行命令:"
    echo "  ./run_curriculum.sh 2>&1 | tee debug.log"
    echo ""
    echo "观察日志关键字:"
    echo "  grep '🔧\|🗺️\|✅.*A\*规划成功\|❌.*A\* plan 失败' debug.log"
    exit 0
else
    echo "❌ 有 $FAIL 个修复未正确应用，请检查。"
    exit 1
fi
