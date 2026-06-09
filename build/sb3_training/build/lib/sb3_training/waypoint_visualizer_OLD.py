"""
Gazebo路径点可视化器
在仿真环境中显示全局路径和关键路径点
"""
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


class WaypointVisualizer(Node):
    """路径点可视化节点"""
    
    def __init__(self):
        super().__init__('waypoint_visualizer')
        
        # 发布Marker
        self.marker_pub = self.create_publisher(
            MarkerArray, 
            '/waypoint_markers', 
            10
        )
        
        self.get_logger().info("✅ 路径点可视化器已启动")
    
    def publish_waypoints(self, waypoints, robot_id=0, namespace='waypoints'):
        """
        发布关键路径点到Gazebo
        
        Args:
            waypoints: 路径点列表 [(x1,y1), (x2,y2), ...]
            robot_id: 机器人ID
            namespace: Marker命名空间
        """
        marker_array = MarkerArray()
        
        # 1. 路径线段
        line_marker = Marker()
        line_marker.header.frame_id = "map"
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.ns = f"{namespace}_line"
        line_marker.id = robot_id * 1000
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        
        line_marker.scale.x = 0.05  # 线宽
        line_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)  # 绿色
        line_marker.pose.orientation.w = 1.0
        
        for wp in waypoints:
            p = Point()
            p.x, p.y, p.z = wp[0], wp[1], 0.05
            line_marker.points.append(p)
        
        marker_array.markers.append(line_marker)
        
        # 2. 路径点球体
        for i, wp in enumerate(waypoints):
            sphere = Marker()
            sphere.header.frame_id = "map"
            sphere.header.stamp = self.get_clock().now().to_msg()
            sphere.ns = f"{namespace}_spheres"
            sphere.id = robot_id * 1000 + i + 1
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            
            sphere.pose.position.x = wp[0]
            sphere.pose.position.y = wp[1]
            sphere.pose.position.z = 0.2  # 悬浮高度
            sphere.pose.orientation.w = 1.0
            
            # 起点蓝色，终点红色，中间黄色
            if i == 0:
                sphere.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)  # 蓝色起点
                sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.3
            elif i == len(waypoints) - 1:
                sphere.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # 红色终点
                sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.3
            else:
                sphere.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.9)  # 黄色中间点
                sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.2
            
            marker_array.markers.append(sphere)
        
        # 3. 路径点编号文字
        for i, wp in enumerate(waypoints):
            text = Marker()
            text.header.frame_id = "map"
            text.header.stamp = self.get_clock().now().to_msg()
            text.ns = f"{namespace}_text"
            text.id = robot_id * 1000 + i + 100
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            
            text.pose.position.x = wp[0]
            text.pose.position.y = wp[1]
            text.pose.position.z = 0.5  # 文字高度
            text.pose.orientation.w = 1.0
            
            text.text = f"WP{i}"
            text.scale.z = 0.2  # 文字大小
            text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)  # 白色
            
            marker_array.markers.append(text)
        
        # 发布
        self.marker_pub.publish(marker_array)
        self.get_logger().info(f"📍 已发布 {len(waypoints)} 个路径点标记")
    
    def clear_waypoints(self, robot_id=0, namespace='waypoints'):
        """清除所有路径点标记"""
        marker_array = MarkerArray()
        
        # 删除标记
        for ns in [f"{namespace}_line", f"{namespace}_spheres", f"{namespace}_text"]:
            delete_marker = Marker()
            delete_marker.header.frame_id = "map"
            delete_marker.ns = ns
            delete_marker.action = Marker.DELETEALL
            marker_array.markers.append(delete_marker)
        
        self.marker_pub.publish(marker_array)
        self.get_logger().info("🗑️ 已清除所有路径点标记")
    
    def highlight_current_waypoint(self, waypoint, robot_id=0):
        """高亮当前目标路径点"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "current_target"
        marker.id = robot_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        marker.pose.position.x = waypoint[0]
        marker.pose.position.y = waypoint[1]
        marker.pose.position.z = 0.01
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = marker.scale.y = 0.5  # 圆圈半径
        marker.scale.z = 0.02  # 厚度
        marker.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.5)  # 紫色高亮
        
        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)
        self.get_logger().info(f"🔮 已高亮当前目标路径点: {waypoint}")

def test_visualizer():
    """测试可视化"""
    rclpy.init()
    viz = WaypointVisualizer()
    
    # 测试路径点
    waypoints = [
        (0.0, 0.0),
        (2.0, 1.0),
        (4.0, 3.0),
        (5.0, 5.0),
        (7.0, 7.0)
    ]
    
    viz.publish_waypoints(waypoints, robot_id=0)
    viz.get_logger().info("测试标记已发布，保持5秒...")
    
    import time
    time.sleep(5)
    
    viz.highlight_current_waypoint(waypoints[1], robot_id=0)
    time.sleep(5)
    
    viz.clear_waypoints()
    
    viz.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    test_visualizer()
