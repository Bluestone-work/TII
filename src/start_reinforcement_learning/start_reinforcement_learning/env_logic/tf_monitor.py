#!/usr/bin/env python3
"""
TF树监控工具 - 用于诊断TF变换问题
帮助验证强化学习观测与仿真场景的对应关系
"""

import rclpy
from rclpy.node import Node
import tf2_ros
import time
import sys
from rclpy.clock import Clock, ClockType


class TFMonitor(Node):
    def __init__(self, number_of_robots=3):
        super().__init__('tf_monitor_main')
        
        self.number_of_robots = number_of_robots
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        print("\n" + "="*80)
        print("🔍 TF树监控工具已启动")
        print(f"   监控机器人数量: {number_of_robots}")
        print("="*80 + "\n")
        
        # 给TF树一些时间积累数据
        time.sleep(2.0)
        
    def discover_frames(self):
        """发现所有可用的TF frames"""
        print("\n📡 扫描TF树...")
        all_frames_str = self.tf_buffer.all_frames_as_string()
        print(all_frames_str)
        
        # 解析frames
        frames = set()
        for line in all_frames_str.split('\n'):
            if 'Frame' in line:
                parts = line.split()
                if len(parts) >= 2:
                    frame_name = parts[1].strip(':')
                    frames.add(frame_name)
        
        return frames
    
    def test_transform(self, target_frame, source_frame):
        """测试两个frame之间的变换"""
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, 
                source_frame, 
                rclpy.time.Time()
            )
            
            tx = transform.transform.translation.x
            ty = transform.transform.translation.y
            tz = transform.transform.translation.z
            
            print(f"  ✅ {target_frame} <- {source_frame}: "
                  f"位移=({tx:.3f}, {ty:.3f}, {tz:.3f})")
            return True
            
        except (tf2_ros.LookupException, 
                tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            print(f"  ❌ {target_frame} <- {source_frame}: {type(e).__name__}")
            return False
    
    def monitor_robot_transforms(self):
        """监控所有机器人的TF变换"""
        print(f"\n🤖 测试机器人TF变换 (总共 {self.number_of_robots} 个机器人):")
        print("-" * 80)
        
        # 可能的frame命名模式
        odom_patterns = [
            lambda i: f"tb3_{i}/odom",
            lambda i: f"robot{i}/odom",
            lambda i: f"bot{i}/odom",
            lambda i: "odom",  # 无命名空间
        ]
        
        base_patterns = [
            lambda i: f"tb3_{i}/base_link",
            lambda i: f"robot{i}/base_link",
            lambda i: f"bot{i}/base_link",
            lambda i: "base_link",
        ]
        
        for i in range(self.number_of_robots):
            print(f"\nRobot {i}:")
            
            # 测试 map -> odom
            found_odom = False
            for pattern in odom_patterns:
                odom_frame = pattern(i)
                if self.test_transform('map', odom_frame):
                    found_odom = True
                    
                    # 继续测试 odom -> base_link
                    for base_pattern in base_patterns:
                        base_frame = base_pattern(i)
                        self.test_transform(odom_frame, base_frame)
                    break
            
            if not found_odom:
                print(f"  ⚠️  未找到 Robot {i} 的 odom frame")
    
    def continuous_monitor(self, interval=5.0):
        """持续监控模式"""
        print(f"\n🔄 进入持续监控模式 (每 {interval}秒 更新一次)")
        print("   按 Ctrl+C 退出\n")
        
        try:
            while rclpy.ok():
                print(f"\n{'='*80}")
                print(f"时间: {self.get_clock().now().to_msg()}")
                print(f"{'='*80}")
                
                self.monitor_robot_transforms()
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\n👋 监控已停止")
    
    def check_data_freshness(self):
        """检查数据新鲜度"""
        print(f"\n⏰ 数据新鲜度检查:")
        print("-" * 80)
        
        now = self.get_clock().now()
        
        for i in range(self.number_of_robots):
            odom_frame = f"tb3_{i}/odom"
            
            try:
                # 获取最新的变换
                transform = self.tf_buffer.lookup_transform(
                    'map', 
                    odom_frame, 
                    rclpy.time.Time()
                )
                
                stamp = transform.header.stamp
                age_sec = (now.nanoseconds - (stamp.sec * 1e9 + stamp.nanosec)) / 1e9
                
                if age_sec < 0.2:
                    status = "✅ 新鲜"
                elif age_sec < 0.5:
                    status = "⚠️  稍旧"
                else:
                    status = "❌ 过时"
                
                print(f"  Robot {i}: {status} (年龄: {age_sec*1000:.1f}ms)")
                
            except Exception as e:
                print(f"  Robot {i}: ❌ 无法获取 ({type(e).__name__})")


def main():
    rclpy.init()
    
    # 从命令行参数获取机器人数量
    num_robots = 3
    if len(sys.argv) > 1:
        try:
            num_robots = int(sys.argv[1])
        except ValueError:
            print(f"警告: 无效的机器人数量 '{sys.argv[1]}', 使用默认值 3")
    
    monitor = TFMonitor(number_of_robots=num_robots)
    
    # 发现所有frames
    frames = monitor.discover_frames()
    print(f"\n📋 发现的frames ({len(frames)} 个):")
    for frame in sorted(frames):
        print(f"   - {frame}")
    
    # 测试机器人变换
    monitor.monitor_robot_transforms()
    
    # 检查数据新鲜度
    monitor.check_data_freshness()
    
    # 询问是否进入持续监控模式
    print("\n" + "="*80)
    response = input("是否进入持续监控模式? (y/n): ")
    if response.lower() == 'y':
        monitor.continuous_monitor()
    else:
        print("\n👋 监控已完成")
    
    monitor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
