#!/usr/bin/env python3
"""
数据同步可视化工具
在RViz中可视化观测、奖励与机器人状态的对应关系
"""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import math


class SyncVisualization(Node):
    def __init__(self, robot_id=0):
        super().__init__(f'sync_viz_robot_{robot_id}')
        
        self.robot_id = robot_id
        
        # 订阅传感器数据
        self.odom_sub = self.create_subscription(
            Odometry,
            f'/tb3_{robot_id}/odom',
            self.odom_callback,
            10
        )
        
        self.scan_sub = self.create_subscription(
            LaserScan,
            f'/tb3_{robot_id}/scan',
            self.scan_callback,
            10
        )
        
        # 发布可视化markers
        self.marker_pub = self.create_publisher(
            MarkerArray,
            f'/robot{robot_id}/sync_visualization',
            10
        )
        
        # 定时发布（10Hz）
        self.timer = self.create_timer(0.1, self.publish_visualization)
        
        # 数据存储
        self.last_odom = None
        self.last_scan = None
        self.last_odom_time = None
        self.last_scan_time = None
        
        print(f"✅ Robot {robot_id} 同步可视化已启动")
    
    def odom_callback(self, msg):
        self.last_odom = msg
        self.last_odom_time = self.get_clock().now()
    
    def scan_callback(self, msg):
        self.last_scan = msg
        self.last_scan_time = self.get_clock().now()
    
    def publish_visualization(self):
        if self.last_odom is None:
            return
        
        marker_array = MarkerArray()
        now = self.get_clock().now()
        
        # 1. 数据新鲜度指示器 - 圆环
        freshness_marker = Marker()
        freshness_marker.header.frame_id = "map"
        freshness_marker.header.stamp = now.to_msg()
        freshness_marker.ns = f"robot_{self.robot_id}_freshness"
        freshness_marker.id = 0
        freshness_marker.type = Marker.CYLINDER
        freshness_marker.action = Marker.ADD
        
        # 位置：机器人上方
        freshness_marker.pose.position.x = self.last_odom.pose.pose.position.x
        freshness_marker.pose.position.y = self.last_odom.pose.pose.position.y
        freshness_marker.pose.position.z = 0.5
        freshness_marker.pose.orientation.w = 1.0
        
        # 大小
        freshness_marker.scale.x = 0.6
        freshness_marker.scale.y = 0.6
        freshness_marker.scale.z = 0.05
        
        # 颜色：根据数据年龄变化
        odom_age = (now.nanoseconds - self.last_odom_time.nanoseconds) / 1e9 if self.last_odom_time else 999
        scan_age = (now.nanoseconds - self.last_scan_time.nanoseconds) / 1e9 if self.last_scan_time else 999
        max_age = max(odom_age, scan_age)
        
        if max_age < 0.1:
            # 新鲜 - 绿色
            freshness_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.6)
        elif max_age < 0.3:
            # 稍旧 - 黄色
            freshness_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.6)
        else:
            # 过时 - 红色
            freshness_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.6)
        
        marker_array.markers.append(freshness_marker)
        
        # 2. 激光雷达可视化 - 点云
        if self.last_scan is not None:
            scan_marker = Marker()
            scan_marker.header.frame_id = f"my_bot{self.robot_id}/base_scan"
            scan_marker.header.stamp = now.to_msg()
            scan_marker.ns = f"robot_{self.robot_id}_scan"
            scan_marker.id = 1
            scan_marker.type = Marker.POINTS
            scan_marker.action = Marker.ADD
            
            scan_marker.scale.x = 0.05
            scan_marker.scale.y = 0.05
            
            # 将激光点转换为点列表
            angle = self.last_scan.angle_min
            for r in self.last_scan.ranges:
                if r > self.last_scan.range_min and r < self.last_scan.range_max:
                    p = Point()
                    p.x = r * math.cos(angle)
                    p.y = r * math.sin(angle)
                    p.z = 0.0
                    scan_marker.points.append(p)
                    
                    # 颜色：根据距离
                    color = ColorRGBA()
                    if r < 0.5:
                        color.r, color.g, color.b = 1.0, 0.0, 0.0  # 红色 - 危险
                    elif r < 1.0:
                        color.r, color.g, color.b = 1.0, 1.0, 0.0  # 黄色 - 警告
                    else:
                        color.r, color.g, color.b = 0.0, 1.0, 0.0  # 绿色 - 安全
                    color.a = 0.8
                    scan_marker.colors.append(color)
                
                angle += self.last_scan.angle_increment
            
            marker_array.markers.append(scan_marker)
        
        # 3. 时间戳文本
        text_marker = Marker()
        text_marker.header.frame_id = "map"
        text_marker.header.stamp = now.to_msg()
        text_marker.ns = f"robot_{self.robot_id}_timestamp"
        text_marker.id = 2
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        
        text_marker.pose.position.x = self.last_odom.pose.pose.position.x
        text_marker.pose.position.y = self.last_odom.pose.pose.position.y
        text_marker.pose.position.z = 0.8
        text_marker.pose.orientation.w = 1.0
        
        text_marker.text = f"R{self.robot_id}\nOdom:{odom_age*1000:.0f}ms\nScan:{scan_age*1000:.0f}ms"
        text_marker.scale.z = 0.15
        text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        
        marker_array.markers.append(text_marker)
        
        # 发布
        self.marker_pub.publish(marker_array)


def main():
    rclpy.init()
    
    import sys
    robot_id = 0
    if len(sys.argv) > 1:
        try:
            robot_id = int(sys.argv[1])
        except ValueError:
            print(f"警告: 无效的机器人ID '{sys.argv[1]}', 使用默认值 0")
    
    node = SyncVisualization(robot_id=robot_id)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print(f"\n👋 Robot {robot_id} 同步可视化已停止")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
