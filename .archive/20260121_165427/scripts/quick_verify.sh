#!/bin/bash

##############################################################################
# 快速验证 - 检查Map1机器人spawn位置
##############################################################################

echo "🧪 快速验证 Map1 初始Spawn"
echo ""

source /opt/ros/humble/setup.bash
source install/setup.bash

# 启动Gazebo
echo "🚀 启动 Gazebo (map1, 3机器人)..."
ros2 launch start_rl_environment main.launch.py \
    map_name:=map1 \
    robot_number:=3 \
    > /tmp/gazebo_spawn.log 2>&1 &

GAZEBO_PID=$!
echo "   Gazebo PID: $GAZEBO_PID"

# 等待启动
echo "⏳ 等待10秒让机器人spawn..."
sleep 10

# 检查机器人位置
echo ""
echo "🔍 检查机器人位置..."
python3 << 'EOF'
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import time

class QuickCheck(Node):
    def __init__(self):
        super().__init__('quick_check')
        self.positions = {}
        self.subs = []
        for i in range(3):
            sub = self.create_subscription(
                Odometry,
                f'/my_bot{i}/odom',
                lambda msg, idx=i: self.cb(msg, idx),
                10
            )
            self.subs.append(sub)
    
    def cb(self, msg, idx):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.positions[idx] = (x, y)

rclpy.init()
node = QuickCheck()

# 收集位置数据
for _ in range(25):
    rclpy.spin_once(node, timeout_sec=0.2)
    if len(node.positions) >= 3:
        break

print("\n" + "="*70)
print("Map1 围墙范围: X ∈ [-2.3, 3.5], Y ∈ [-11.0, -0.1]")
print("="*70)

all_good = True
for i in range(3):
    if i in node.positions:
        x, y = node.positions[i]
        in_x = -2.3 <= x <= 3.5
        in_y = -11.0 <= y <= -0.1
        in_bounds = in_x and in_y
        
        if in_bounds:
            status = "✓ 正确"
            symbol = "✅"
        else:
            status = "✗ 越界!"
            symbol = "❌"
            all_good = False
        
        print(f"{symbol} Robot{i}: ({x:6.2f}, {y:6.2f})  {status}")
    else:
        print(f"⚠️  Robot{i}: 未收到数据")
        all_good = False

print("="*70)

if all_good:
    print("\n🎉 验证通过！所有机器人都在围墙内\n")
else:
    print("\n⚠️  验证失败！存在越界机器人\n")

rclpy.shutdown()
EOF

# 清理
echo ""
echo "🛑 关闭 Gazebo..."
kill $GAZEBO_PID 2>/dev/null
sleep 2

echo ""
echo "✅ 验证完成"
echo ""
