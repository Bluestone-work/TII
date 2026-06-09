from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import math

class WaypointVisualizer:
    """路径点可视化辅助类"""
    
    def __init__(self, node, topic_name='/waypoint_markers'):
        self.node = node
        self.marker_pub = self.node.create_publisher(
            MarkerArray, 
            topic_name, 
            10
        )
        self.node.get_logger().info(f"✅ 路径点可视化器已启动，话题: {topic_name}")

    def get_robot_color(self, robot_id):
        """根据机器人ID生成不同颜色"""
        # 预定义颜色表 (R, G, B)
        colors = [
            (0.0, 1.0, 0.0),  # 0: 绿色
            (0.0, 1.0, 1.0),  # 1: 青色
            (1.0, 0.0, 1.0),  # 2: 紫色
            (1.0, 1.0, 0.0),  # 3: 黄色
            (1.0, 0.5, 0.0),  # 4: 橙色
            (0.5, 0.0, 1.0),  # 5: 深紫
            (0.0, 0.5, 1.0),  # 6: 这里的蓝
        ]
        # 如果超出列表，循环使用
        c = colors[robot_id % len(colors)]
        return ColorRGBA(r=c[0], g=c[1], b=c[2], a=0.8)

    def publish_waypoints(self, waypoints, robot_id=0, namespace='waypoints'):
        """发布关键路径点"""
        if not waypoints:
            return

        marker_array = MarkerArray()
        current_time = self.node.get_clock().now().to_msg()
        
        # 获取当前机器人的专属颜色
        robot_color = self.get_robot_color(robot_id)

        # 1. 路径线段 (Line Strip)
        line_marker = Marker()
        line_marker.header.frame_id = "map"
        line_marker.header.stamp = current_time
        line_marker.ns = f"{namespace}_line" # 命名空间
        line_marker.id = robot_id            # 线条ID
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        
        line_marker.scale.x = 0.05
        line_marker.color = robot_color      # 使用专属颜色
        line_marker.pose.orientation.w = 1.0
        
        for wp in waypoints:
            p = Point()
            p.x, p.y, p.z = float(wp[0]), float(wp[1]), 0.05
            line_marker.points.append(p)
        
        marker_array.markers.append(line_marker)
        
        # 2. 路径点球体 (Spheres)
        for i, wp in enumerate(waypoints):
            sphere = Marker()
            sphere.header.frame_id = "map"
            sphere.header.stamp = current_time
            sphere.ns = f"{namespace}_spheres"
            sphere.id = i  # 每个球体独立的ID
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            
            sphere.pose.position.x = float(wp[0])
            sphere.pose.position.y = float(wp[1])
            sphere.pose.position.z = 0.2
            sphere.pose.orientation.w = 1.0
            
            # 起点(蓝)和终点(红)保持特殊颜色，中间点使用机器人专属颜色
            if i == 0: 
                sphere.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0) # 起点蓝
                sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.3
            elif i == len(waypoints) - 1: 
                sphere.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0) # 终点红
                sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.3
            else: 
                sphere.color = robot_color # 中间点跟随机器人颜色
                sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.15
            
            marker_array.markers.append(sphere)
            
        self.marker_pub.publish(marker_array)
    
    def clear_waypoints(self, namespace='waypoints'):
        """清除指定命名空间下的所有标记"""
        marker_array = MarkerArray()
        # 清除线条和球体
        for ns_suffix in ["_line", "_spheres", "_text", "current_target"]:
            delete_marker = Marker()
            delete_marker.header.frame_id = "map"
            # 注意：这里拼接完整的命名空间
            delete_marker.ns = f"{namespace}{ns_suffix}"
            delete_marker.action = Marker.DELETEALL
            marker_array.markers.append(delete_marker)
        
        self.marker_pub.publish(marker_array)