from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


class WaypointVisualizer:
    """路径/rolling subgoal 可视化辅助类（ROS2 / RViz）。

    设计目标：
    - 全局路径与 tracking 状态分 topic 发布，互不影响
    - 保留美化版显示效果
    - 兼容现有 publish_waypoints()/publish_tracking_state() 调用方式
    """

    def __init__(
        self,
        node,
        topic_name='/waypoint_markers',
        tracking_topic_name='/rolling_subgoal_markers',
    ):
        self.node = node
        self.path_topic_name = topic_name
        self.tracking_topic_name = tracking_topic_name

        self.path_pub = self.node.create_publisher(MarkerArray, self.path_topic_name, 10)
        self.tracking_pub = self.node.create_publisher(MarkerArray, self.tracking_topic_name, 10)
        self.node.get_logger().info(
            f"✅ 路径可视化器已启动，路径话题: {self.path_topic_name}, tracking话题: {self.tracking_topic_name}"
        )

    def get_robot_color(self, robot_id):
        """根据机器人 ID 生成主色。"""
        colors = [
            (0.17, 0.84, 0.44),  # emerald
            (0.20, 0.72, 1.00),  # sky
            (0.80, 0.42, 1.00),  # violet
            (1.00, 0.78, 0.20),  # amber
            (1.00, 0.50, 0.24),  # orange
            (0.95, 0.32, 0.55),  # rose
            (0.45, 0.62, 1.00),  # indigo-blue
        ]
        c = colors[robot_id % len(colors)]
        return ColorRGBA(r=float(c[0]), g=float(c[1]), b=float(c[2]), a=0.92)

    @staticmethod
    def _point(x, y, z=0.0):
        p = Point()
        p.x = float(x)
        p.y = float(y)
        p.z = float(z)
        return p

    def publish_waypoints(self, waypoints, robot_id=0, namespace='waypoints'):
        """发布全局路径：细线 + 起终点 + 中间关键点，单独走路径话题。"""
        if not waypoints:
            return

        marker_array = MarkerArray()
        current_time = self.node.get_clock().now().to_msg()
        robot_color = self.get_robot_color(robot_id)

        # 1) 柔和底线，让路径更有层次
        base_line = Marker()
        base_line.header.frame_id = 'map'
        base_line.header.stamp = current_time
        base_line.ns = f'{namespace}_path_base'
        base_line.id = robot_id
        base_line.type = Marker.LINE_STRIP
        base_line.action = Marker.ADD
        base_line.scale.x = 0.10
        base_line.color = ColorRGBA(r=0.08, g=0.10, b=0.14, a=0.22)
        base_line.pose.orientation.w = 1.0
        for wp in waypoints:
            base_line.points.append(self._point(wp[0], wp[1], 0.03))
        marker_array.markers.append(base_line)

        # 2) 主路径线
        line_marker = Marker()
        line_marker.header.frame_id = 'map'
        line_marker.header.stamp = current_time
        line_marker.ns = f'{namespace}_path_main'
        line_marker.id = robot_id
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.05
        line_marker.color = robot_color
        line_marker.pose.orientation.w = 1.0
        for wp in waypoints:
            line_marker.points.append(self._point(wp[0], wp[1], 0.05))
        marker_array.markers.append(line_marker)

        # 3) 路径点球体
        for i, wp in enumerate(waypoints):
            sphere = Marker()
            sphere.header.frame_id = 'map'
            sphere.header.stamp = current_time
            sphere.ns = f'{namespace}_path_points_r{robot_id}'
            sphere.id = i
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(wp[0])
            sphere.pose.position.y = float(wp[1])
            sphere.pose.position.z = 0.12
            sphere.pose.orientation.w = 1.0

            if i == 0:
                sphere.color = ColorRGBA(r=0.20, g=0.58, b=1.00, a=0.98)
                sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.22
            elif i == len(waypoints) - 1:
                sphere.color = ColorRGBA(r=1.00, g=0.28, b=0.28, a=0.98)
                sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.24
            else:
                sphere.color = ColorRGBA(r=robot_color.r, g=robot_color.g, b=robot_color.b, a=0.62)
                sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.11
            marker_array.markers.append(sphere)

        # 4) 可选标签：仅标起点和终点，避免画面过乱
        start_text = Marker()
        start_text.header.frame_id = 'map'
        start_text.header.stamp = current_time
        start_text.ns = f'{namespace}_path_text_start'
        start_text.id = robot_id
        start_text.type = Marker.TEXT_VIEW_FACING
        start_text.action = Marker.ADD
        start_text.pose.position.x = float(waypoints[0][0])
        start_text.pose.position.y = float(waypoints[0][1])
        start_text.pose.position.z = 0.42
        start_text.pose.orientation.w = 1.0
        start_text.scale.z = 0.14
        start_text.color = ColorRGBA(r=0.72, g=0.88, b=1.00, a=0.95)
        start_text.text = f'R{robot_id} start'
        marker_array.markers.append(start_text)

        goal_text = Marker()
        goal_text.header.frame_id = 'map'
        goal_text.header.stamp = current_time
        goal_text.ns = f'{namespace}_path_text_goal'
        goal_text.id = robot_id
        goal_text.type = Marker.TEXT_VIEW_FACING
        goal_text.action = Marker.ADD
        goal_text.pose.position.x = float(waypoints[-1][0])
        goal_text.pose.position.y = float(waypoints[-1][1])
        goal_text.pose.position.z = 0.44
        goal_text.pose.orientation.w = 1.0
        goal_text.scale.z = 0.15
        goal_text.color = ColorRGBA(r=1.00, g=0.62, b=0.62, a=0.96)
        goal_text.text = f'R{robot_id} goal'
        marker_array.markers.append(goal_text)

        self.path_pub.publish(marker_array)

    def publish_tracking_state(
        self,
        robot_pos,
        target_pos,
        nominal_target_pos=None,
        projection_pos=None,
        robot_id=0,
        namespace='waypoints',
        label=None,
    ):
        """发布当前跟踪状态。

        `target_pos` 表示实际控制器正在跟踪的目标点。
        `nominal_target_pos` 表示路径滚动前瞻得到的名义 rolling_subgoal。
        """
        if robot_pos is None or target_pos is None:
            return

        marker_array = MarkerArray()
        current_time = self.node.get_clock().now().to_msg()
        robot_color = self.get_robot_color(robot_id)
        ns_prefix = f'{namespace}_tracking_r{robot_id}'
        nominal_differs = False
        if nominal_target_pos is not None:
            dx = float(target_pos[0]) - float(nominal_target_pos[0])
            dy = float(target_pos[1]) - float(nominal_target_pos[1])
            nominal_differs = (dx * dx + dy * dy) > 1e-6

        # A. 机器人到实际 tracking target 的引导箭头
        guide = Marker()
        guide.header.frame_id = 'map'
        guide.header.stamp = current_time
        guide.ns = f'{ns_prefix}_actual_guide'
        guide.id = 0
        guide.type = Marker.ARROW
        guide.action = Marker.ADD
        guide.pose.orientation.w = 1.0
        guide.scale.x = 0.05
        guide.scale.y = 0.10
        guide.scale.z = 0.12
        guide.color = ColorRGBA(r=robot_color.r, g=robot_color.g, b=robot_color.b, a=0.70)
        guide.points = [
            self._point(robot_pos[0], robot_pos[1], 0.12),
            self._point(target_pos[0], target_pos[1], 0.12),
        ]
        marker_array.markers.append(guide)

        # B. 实际 tracking target 外层光晕
        halo = Marker()
        halo.header.frame_id = 'map'
        halo.header.stamp = current_time
        halo.ns = f'{ns_prefix}_actual_halo'
        halo.id = 1
        halo.type = Marker.SPHERE
        halo.action = Marker.ADD
        halo.pose.position.x = float(target_pos[0])
        halo.pose.position.y = float(target_pos[1])
        halo.pose.position.z = 0.16
        halo.pose.orientation.w = 1.0
        halo.scale.x = halo.scale.y = halo.scale.z = 0.34
        halo.color = ColorRGBA(r=1.00, g=0.74, b=0.18, a=0.22)
        marker_array.markers.append(halo)

        # C. 实际 tracking target 主球
        subgoal = Marker()
        subgoal.header.frame_id = 'map'
        subgoal.header.stamp = current_time
        subgoal.ns = f'{ns_prefix}_actual_main'
        subgoal.id = 2
        subgoal.type = Marker.SPHERE
        subgoal.action = Marker.ADD
        subgoal.pose.position.x = float(target_pos[0])
        subgoal.pose.position.y = float(target_pos[1])
        subgoal.pose.position.z = 0.17
        subgoal.pose.orientation.w = 1.0
        subgoal.scale.x = subgoal.scale.y = subgoal.scale.z = 0.20
        subgoal.color = ColorRGBA(r=1.00, g=0.78, b=0.24, a=0.96)
        marker_array.markers.append(subgoal)

        # D. 机器人当前位置小球
        robot_marker = Marker()
        robot_marker.header.frame_id = 'map'
        robot_marker.header.stamp = current_time
        robot_marker.ns = f'{ns_prefix}_robot'
        robot_marker.id = 3
        robot_marker.type = Marker.SPHERE
        robot_marker.action = Marker.ADD
        robot_marker.pose.position.x = float(robot_pos[0])
        robot_marker.pose.position.y = float(robot_pos[1])
        robot_marker.pose.position.z = 0.11
        robot_marker.pose.orientation.w = 1.0
        robot_marker.scale.x = robot_marker.scale.y = robot_marker.scale.z = 0.12
        robot_marker.color = ColorRGBA(r=robot_color.r, g=robot_color.g, b=robot_color.b, a=0.90)
        marker_array.markers.append(robot_marker)

        # E. 路径最近投影点（可选）
        if projection_pos is not None:
            proj = Marker()
            proj.header.frame_id = 'map'
            proj.header.stamp = current_time
            proj.ns = f'{ns_prefix}_projection'
            proj.id = 4
            proj.type = Marker.SPHERE
            proj.action = Marker.ADD
            proj.pose.position.x = float(projection_pos[0])
            proj.pose.position.y = float(projection_pos[1])
            proj.pose.position.z = 0.14
            proj.pose.orientation.w = 1.0
            proj.scale.x = proj.scale.y = proj.scale.z = 0.14
            proj.color = ColorRGBA(r=0.22, g=0.95, b=1.00, a=0.92)
            marker_array.markers.append(proj)

            proj_link = Marker()
            proj_link.header.frame_id = 'map'
            proj_link.header.stamp = current_time
            proj_link.ns = f'{ns_prefix}_projection_link'
            proj_link.id = 5
            proj_link.type = Marker.LINE_STRIP
            proj_link.action = Marker.ADD
            proj_link.pose.orientation.w = 1.0
            proj_link.scale.x = 0.02
            proj_link.color = ColorRGBA(r=0.22, g=0.95, b=1.00, a=0.52)
            proj_link.points = [
                self._point(robot_pos[0], robot_pos[1], 0.09),
                self._point(projection_pos[0], projection_pos[1], 0.09),
            ]
            marker_array.markers.append(proj_link)

        # F. 名义 rolling_subgoal。只有与实际 tracking target 分叉时才单独画出。
        if nominal_target_pos is not None and nominal_differs:
            nominal_link = Marker()
            nominal_link.header.frame_id = 'map'
            nominal_link.header.stamp = current_time
            nominal_link.ns = f'{ns_prefix}_nominal_link'
            nominal_link.id = 7
            nominal_link.type = Marker.LINE_STRIP
            nominal_link.action = Marker.ADD
            nominal_link.pose.orientation.w = 1.0
            nominal_link.scale.x = 0.025
            nominal_link.color = ColorRGBA(r=0.18, g=0.92, b=1.00, a=0.72)
            nominal_link.points = [
                self._point(robot_pos[0], robot_pos[1], 0.07),
                self._point(nominal_target_pos[0], nominal_target_pos[1], 0.07),
            ]
            marker_array.markers.append(nominal_link)

            nominal_marker = Marker()
            nominal_marker.header.frame_id = 'map'
            nominal_marker.header.stamp = current_time
            nominal_marker.ns = f'{ns_prefix}_nominal_main'
            nominal_marker.id = 8
            nominal_marker.type = Marker.CUBE
            nominal_marker.action = Marker.ADD
            nominal_marker.pose.position.x = float(nominal_target_pos[0])
            nominal_marker.pose.position.y = float(nominal_target_pos[1])
            nominal_marker.pose.position.z = 0.18
            nominal_marker.pose.orientation.w = 1.0
            nominal_marker.scale.x = nominal_marker.scale.y = nominal_marker.scale.z = 0.14
            nominal_marker.color = ColorRGBA(r=0.18, g=0.92, b=1.00, a=0.96)
            marker_array.markers.append(nominal_marker)

            delta_link = Marker()
            delta_link.header.frame_id = 'map'
            delta_link.header.stamp = current_time
            delta_link.ns = f'{ns_prefix}_nominal_delta'
            delta_link.id = 9
            delta_link.type = Marker.LINE_STRIP
            delta_link.action = Marker.ADD
            delta_link.pose.orientation.w = 1.0
            delta_link.scale.x = 0.03
            delta_link.color = ColorRGBA(r=1.00, g=0.30, b=0.30, a=0.85)
            delta_link.points = [
                self._point(nominal_target_pos[0], nominal_target_pos[1], 0.22),
                self._point(target_pos[0], target_pos[1], 0.22),
            ]
            marker_array.markers.append(delta_link)

            nominal_text = Marker()
            nominal_text.header.frame_id = 'map'
            nominal_text.header.stamp = current_time
            nominal_text.ns = f'{ns_prefix}_nominal_label'
            nominal_text.id = 10
            nominal_text.type = Marker.TEXT_VIEW_FACING
            nominal_text.action = Marker.ADD
            nominal_text.pose.position.x = float(nominal_target_pos[0])
            nominal_text.pose.position.y = float(nominal_target_pos[1])
            nominal_text.pose.position.z = 0.38
            nominal_text.pose.orientation.w = 1.0
            nominal_text.scale.z = 0.13
            nominal_text.color = ColorRGBA(r=0.82, g=0.98, b=1.00, a=0.92)
            nominal_text.text = 'nominal rolling_subgoal'
            marker_array.markers.append(nominal_text)

        # G. 文字标签
        text = Marker()
        text.header.frame_id = 'map'
        text.header.stamp = current_time
        text.ns = f'{ns_prefix}_label'
        text.id = 6
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(target_pos[0])
        text.pose.position.y = float(target_pos[1])
        text.pose.position.z = 0.46
        text.pose.orientation.w = 1.0
        text.scale.z = 0.16
        text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.95)
        text.text = label if label is not None else f'R{robot_id} actual tracking target'
        marker_array.markers.append(text)

        self.tracking_pub.publish(marker_array)

    def clear_waypoints(self, namespace='waypoints'):
        """清除路径与 tracking 两类标记。"""
        current_time = self.node.get_clock().now().to_msg()

        path_array = MarkerArray()
        for ns_suffix in ['_path_base', '_path_main', '_path_points_r0', '_path_points_r1', '_path_points_r2', '_path_points_r3', '_path_text_start', '_path_text_goal']:
            delete_marker = Marker()
            delete_marker.header.frame_id = 'map'
            delete_marker.header.stamp = current_time
            delete_marker.ns = f'{namespace}{ns_suffix}'
            delete_marker.action = Marker.DELETEALL
            path_array.markers.append(delete_marker)
        self.path_pub.publish(path_array)

        tracking_array = MarkerArray()
        # 用更稳妥的方式：对每个常见机器人 id 发 DELETEALL
        for robot_id in range(16):
            prefix = f'{namespace}_tracking_r{robot_id}'
            for suffix in [
                '_actual_guide', '_actual_halo', '_actual_main', '_robot',
                '_projection', '_projection_link', '_label',
                '_nominal_link', '_nominal_main', '_nominal_delta', '_nominal_label'
            ]:
                delete_marker = Marker()
                delete_marker.header.frame_id = 'map'
                delete_marker.header.stamp = current_time
                delete_marker.ns = f'{prefix}{suffix}'
                delete_marker.action = Marker.DELETEALL
                tracking_array.markers.append(delete_marker)
        self.tracking_pub.publish(tracking_array)
