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

        # 记录每个 robot_id 上一次发布的 (ns, id)，用于精准 DELETE 残留 marker。
        # 不再使用 DELETEALL —— 在 RViz2 中 DELETEALL 会清空整个 display（无视 ns），
        # 多机器人共享同一 topic 时会互相把对方的路径删掉，且与 ADD marker 同 (ns,id)
        # 还会触发 "Duplicate Marker" 警告。
        self._path_markers = {}      # robot_id -> [(ns, id), ...]
        self._tracking_markers = {}  # robot_id -> [(ns, id), ...]
        self._completion_markers = {}  # robot_id -> [(ns, id), ...]

        self.node.get_logger().info(
            f" 路径可视化器已启动，路径话题: {self.path_topic_name}, tracking话题: {self.tracking_topic_name}"
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

        current_time = self.node.get_clock().now().to_msg()
        robot_color = self.get_robot_color(robot_id)

        # 先用*单独一条*消息精准 DELETE 该 robot_id 上一轮发布的 markers（防止路径点数减少时残留）。
        # 关键：DELETE 与本轮 ADD 分两条消息发布 —— 否则同一条 MarkerArray 里
        # DELETE(_path_base, id) 与新的 ADD(_path_base, id) 会共用同一 (ns, id)，
        # RViz2 的重复检测按整条消息扫描（不区分 action），仍会报 "Duplicate Marker"。
        prev = self._path_markers.get(robot_id, [])
        if prev:
            delete_array = MarkerArray()
            for ns, mid in prev:
                dm = Marker()
                dm.header.frame_id = 'map'
                dm.header.stamp = current_time
                dm.ns = ns
                dm.id = mid
                dm.action = Marker.DELETE
                delete_array.markers.append(dm)
            self.path_pub.publish(delete_array)

        marker_array = MarkerArray()
        new_markers = []  # 本轮发布的 (ns, id)，发布成功后覆盖记录

        def _track(marker):
            """登记本轮 ADD 的 marker，并加入待发布数组。"""
            new_markers.append((marker.ns, marker.id))
            marker_array.markers.append(marker)

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
        _track(base_line)

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
        _track(line_marker)

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
            _track(sphere)

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
        _track(start_text)

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
        _track(goal_text)

        self.path_pub.publish(marker_array)
        # 记录本轮发布的 marker，供下一轮精准 DELETE 残留
        self._path_markers[robot_id] = new_markers

    def publish_tracking_state(
        self,
        robot_pos,
        target_pos,
        projection_pos=None,
        robot_id=0,
        namespace='waypoints',
        label=None,
    ):
        """发布 rolling subgoal / 当前跟踪目标的高颜值可视化，单独走 tracking 话题。"""
        if robot_pos is None or target_pos is None:
            return

        marker_array = MarkerArray()
        current_time = self.node.get_clock().now().to_msg()
        robot_color = self.get_robot_color(robot_id)
        ns_prefix = f'{namespace}_tracking_r{robot_id}'

        # A. 机器人到 subgoal 的引导箭头（淡色）
        guide = Marker()
        guide.header.frame_id = 'map'
        guide.header.stamp = current_time
        guide.ns = f'{ns_prefix}_guide'
        guide.id = 0
        guide.type = Marker.ARROW
        guide.action = Marker.ADD
        guide.pose.orientation.w = 1.0
        guide.scale.x = 0.05
        guide.scale.y = 0.10
        guide.scale.z = 0.12
        guide.color = ColorRGBA(r=robot_color.r, g=robot_color.g, b=robot_color.b, a=0.55)
        guide.points = [
            self._point(robot_pos[0], robot_pos[1], 0.12),
            self._point(target_pos[0], target_pos[1], 0.12),
        ]
        marker_array.markers.append(guide)

        # B. subgoal 外层光晕
        halo = Marker()
        halo.header.frame_id = 'map'
        halo.header.stamp = current_time
        halo.ns = f'{ns_prefix}_subgoal_halo'
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

        # C. subgoal 主球
        subgoal = Marker()
        subgoal.header.frame_id = 'map'
        subgoal.header.stamp = current_time
        subgoal.ns = f'{ns_prefix}_subgoal_main'
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

        # F. 文字标签
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
        text.text = label if label is not None else f'R{robot_id} rolling_subgoal'
        marker_array.markers.append(text)

        self.tracking_pub.publish(marker_array)
        # 记录本轮 tracking marker 的 (ns, id)，供 clear_waypoints 精准删除
        self._tracking_markers[robot_id] = [
            (m.ns, m.id) for m in marker_array.markers
        ]

    def publish_goal_completed(self, goal_pos, robot_id=0, namespace='waypoints'):
        """发布到达后的绿色终点标记。"""
        if goal_pos is None:
            return

        current_time = self.node.get_clock().now().to_msg()
        prev = self._completion_markers.get(robot_id, [])
        if prev:
            delete_array = MarkerArray()
            for ns, mid in prev:
                dm = Marker()
                dm.header.frame_id = 'map'
                dm.header.stamp = current_time
                dm.ns = ns
                dm.id = mid
                dm.action = Marker.DELETE
                delete_array.markers.append(dm)
            self.path_pub.publish(delete_array)

        marker_array = MarkerArray()
        ns_prefix = f'{namespace}_completed_r{robot_id}'

        halo = Marker()
        halo.header.frame_id = 'map'
        halo.header.stamp = current_time
        halo.ns = f'{ns_prefix}_halo'
        halo.id = 0
        halo.type = Marker.SPHERE
        halo.action = Marker.ADD
        halo.pose.position.x = float(goal_pos[0])
        halo.pose.position.y = float(goal_pos[1])
        halo.pose.position.z = 0.18
        halo.pose.orientation.w = 1.0
        halo.scale.x = halo.scale.y = halo.scale.z = 0.48
        halo.color = ColorRGBA(r=0.05, g=0.95, b=0.30, a=0.28)
        marker_array.markers.append(halo)

        core = Marker()
        core.header.frame_id = 'map'
        core.header.stamp = current_time
        core.ns = f'{ns_prefix}_core'
        core.id = 1
        core.type = Marker.SPHERE
        core.action = Marker.ADD
        core.pose.position.x = float(goal_pos[0])
        core.pose.position.y = float(goal_pos[1])
        core.pose.position.z = 0.22
        core.pose.orientation.w = 1.0
        core.scale.x = core.scale.y = core.scale.z = 0.26
        core.color = ColorRGBA(r=0.08, g=1.00, b=0.36, a=0.98)
        marker_array.markers.append(core)

        text = Marker()
        text.header.frame_id = 'map'
        text.header.stamp = current_time
        text.ns = f'{ns_prefix}_text'
        text.id = 2
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(goal_pos[0])
        text.pose.position.y = float(goal_pos[1])
        text.pose.position.z = 0.58
        text.pose.orientation.w = 1.0
        text.scale.z = 0.17
        text.color = ColorRGBA(r=0.68, g=1.00, b=0.76, a=0.98)
        text.text = f'R{robot_id} reached'
        marker_array.markers.append(text)

        self.path_pub.publish(marker_array)
        self._completion_markers[robot_id] = [
            (m.ns, m.id) for m in marker_array.markers
        ]

    def _delete_markers(self, publisher, marker_records, current_time):
        """按 (ns, id) 精准删除已登记的 marker。避免 DELETEALL 清空共享 topic 上的其它路径。"""
        if not marker_records:
            return
        arr = MarkerArray()
        for records in list(marker_records.values()):
            for ns, mid in records:
                dm = Marker()
                dm.header.frame_id = 'map'
                dm.header.stamp = current_time
                dm.ns = ns
                dm.id = mid
                dm.action = Marker.DELETE
                arr.markers.append(dm)
        if arr.markers:
            publisher.publish(arr)

    def clear_waypoints(self, namespace='waypoints'):
        """清除本可视化器发布过的路径与 tracking 标记（精准 DELETE，不影响其它机器人）。"""
        current_time = self.node.get_clock().now().to_msg()

        self._delete_markers(self.path_pub, self._path_markers, current_time)
        self._path_markers = {}

        self._delete_markers(self.tracking_pub, self._tracking_markers, current_time)
        self._tracking_markers = {}

        self._delete_markers(self.path_pub, self._completion_markers, current_time)
        self._completion_markers = {}
