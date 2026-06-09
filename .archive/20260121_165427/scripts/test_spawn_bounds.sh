#!/bin/bash

##############################################################################
# 测试修复后的spawn区域配置
# 验证机器人和目标是否在围墙内生成
##############################################################################

echo "🧪 测试 Map1 和 Map2 的 Spawn 区域限制"
echo ""

# ROS2环境
source /opt/ros/humble/setup.bash
source install/setup.bash

# 测试Map1
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📍 测试 Map1 (围墙范围: X=[-2.3, 3.5], Y=[-11, -0.1])"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 启动Gazebo (map1)
echo "🚀 启动 Gazebo (map1) 并生成3个机器人..."
ros2 launch start_rl_environment main.launch.py \
    map_name:=map1 \
    robot_number:=3 \
    > /dev/null 2>&1 &
GAZEBO_PID=$!
echo "   PID: $GAZEBO_PID"

sleep 10

# 检查初始spawn位置
echo ""
echo "🔍 检查初始Spawn位置 (通过start_robots.launch.py生成)..."
python3 << 'PYTHON_CHECK_INITIAL'
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import time

class OdomChecker(Node):
    def __init__(self):
        super().__init__('odom_checker')
        self.positions = {}
        self.subs = []
        for i in range(3):
            sub = self.create_subscription(
                Odometry,
                f'/my_bot{i}/odom',
                lambda msg, idx=i: self.odom_callback(msg, idx),
                10
            )
            self.subs.append(sub)
    
    def odom_callback(self, msg, idx):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.positions[idx] = (x, y)

rclpy.init()
checker = OdomChecker()

# 等待接收位置
for _ in range(20):
    rclpy.spin_once(checker, timeout_sec=0.2)
    if len(checker.positions) >= 3:
        break

print("\n✅ 初始机器人位置 (应在围墙内: X∈[-2.3, 3.5], Y∈[-11, -0.1]):")
for i in range(3):
    if i in checker.positions:
        x, y = checker.positions[i]
        in_bounds = (-2.3 <= x <= 3.5) and (-11.0 <= y <= -0.1)
        status = "✓" if in_bounds else "✗ 越界!"
        print(f"   Robot {i}: x={x:6.2f}, y={y:6.2f}  {status}")
    else:
        print(f"   Robot {i}: 未收到位置数据")

rclpy.shutdown()
PYTHON_CHECK_INITIAL

sleep 2

# 测试随机模式重置
echo ""
echo "🔄 测试随机模式重置 (restart_environment.py)..."
python3 << 'PYTHON_TEST1'
import rclpy
from start_reinforcement_learning.env_logic.logic import Env
import time

rclpy.init()

try:
    # 创建环境（随机模式）
    env = Env(number_of_robots=3, map_number=1, use_random_mode=True)
    
    # Reset环境以生成机器人和目标
    print("   执行 reset...")
    obs = env.reset()
    
    time.sleep(2)
    
    # 检查机器人位置
    print("\n✅ 随机重置后的机器人位置 (应在围墙内: X∈[-2.3, 3.5], Y∈[-11, -0.1]):")
    for i in range(len(env.current_pose_x)):
        x = env.current_pose_x[i]
        y = env.current_pose_y[i]
        in_bounds = (-2.3 <= x <= 3.5) and (-11.0 <= y <= -0.1)
        status = "✓" if in_bounds else "✗ 越界!"
        print(f"   Robot {i}: x={x:6.2f}, y={y:6.2f}  {status}")
    
    # 检查目标位置
    print("\n🎯 目标位置 (应在围墙内: X∈[-2.3, 3.5], Y∈[-11, -0.1]):")
    for i, goal in enumerate(env.current_goal_locations):
        x, y = goal
        in_bounds = (-2.3 <= x <= 3.5) and (-11.0 <= y <= -0.1)
        status = "✓" if in_bounds else "✗ 越界!"
        print(f"   Goal  {i}: x={x:6.2f}, y={y:6.2f}  {status}")
    
    print("\n✅ Map1 测试完成\n")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
finally:
    rclpy.shutdown()
PYTHON_TEST1

# 清理Gazebo
echo "🛑 关闭 Gazebo..."
kill $GAZEBO_PID 2>/dev/null
sleep 3



# 测试Map2
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📍 测试 Map2 (围墙范围: X=[-1.9, 10.2], Y=[-11.3, -0.1])"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "🚀 启动 Gazebo (map2) 并生成3个机器人..."
ros2 launch start_rl_environment main.launch.py \
    map_name:=map2 \
    robot_number:=3 \
    > /dev/null 2>&1 &
GAZEBO_PID=$!
echo "   PID: $GAZEBO_PID"

sleep 10

# 检查初始spawn位置
echo ""
echo "🔍 检查初始Spawn位置 (通过start_robots.launch.py生成)..."
python3 << 'PYTHON_CHECK_INITIAL2'
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import time

class OdomChecker(Node):
    def __init__(self):
        super().__init__('odom_checker')
        self.positions = {}
        self.subs = []
        for i in range(3):
            sub = self.create_subscription(
                Odometry,
                f'/my_bot{i}/odom',
                lambda msg, idx=i: self.odom_callback(msg, idx),
                10
            )
            self.subs.append(sub)
    
    def odom_callback(self, msg, idx):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.positions[idx] = (x, y)

rclpy.init()
checker = OdomChecker()

# 等待接收位置
for _ in range(20):
    rclpy.spin_once(checker, timeout_sec=0.2)
    if len(checker.positions) >= 3:
        break

print("\n✅ 初始机器人位置 (应在围墙内: X∈[-1.9, 10.2], Y∈[-11.3, -0.1]):")
for i in range(3):
    if i in checker.positions:
        x, y = checker.positions[i]
        in_bounds = (-1.9 <= x <= 10.2) and (-11.3 <= y <= -0.1)
        status = "✓" if in_bounds else "✗ 越界!"
        print(f"   Robot {i}: x={x:6.2f}, y={y:6.2f}  {status}")
    else:
        print(f"   Robot {i}: 未收到位置数据")

rclpy.shutdown()
PYTHON_CHECK_INITIAL2

sleep 2

# 测试随机模式
echo ""
echo "🔄 测试随机模式重置 (restart_environment.py)..."
python3 << 'PYTHON_TEST2'
import rclpy
from start_reinforcement_learning.env_logic.logic import Env
import time

rclpy.init()

try:
    # 创建环境（随机模式）
    env = Env(number_of_robots=3, map_number=2, use_random_mode=True)
    
    # Reset环境
    print("   执行 reset...")
    obs = env.reset()
    
    time.sleep(2)
    
    # 检查机器人位置
    print("\n✅ 随机重置后的机器人位置 (应在围墙内: X∈[-1.9, 10.2], Y∈[-11.3, -0.1]):")
    for i in range(len(env.current_pose_x)):
        x = env.current_pose_x[i]
        y = env.current_pose_y[i]
        in_bounds = (-1.9 <= x <= 10.2) and (-11.3 <= y <= -0.1)
        status = "✓" if in_bounds else "✗ 越界!"
        print(f"   Robot {i}: x={x:6.2f}, y={y:6.2f}  {status}")
    
    # 检查目标位置
    print("\n🎯 目标位置 (应在围墙内: X∈[-1.9, 10.2], Y∈[-11.3, -0.1]):")
    for i, goal in enumerate(env.current_goal_locations):
        x, y = goal
        in_bounds = (-1.9 <= x <= 10.2) and (-11.3 <= y <= -0.1)
        status = "✓" if in_bounds else "✗ 越界!"
        print(f"   Goal  {i}: x={x:6.2f}, y={y:6.2f}  {status}")
    
    print("\n✅ Map2 测试完成\n")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
finally:
    rclpy.shutdown()
PYTHON_TEST2

# 清理
echo "🛑 关闭 Gazebo..."
kill $GAZEBO_PID 2>/dev/null
sleep 2

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 测试完成！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
