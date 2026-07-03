#!/usr/bin/env python3
"""
动态仓库环境 - 每个episode随机生成静态障碍物
基于circle_swap_arena，添加随机货架/箱子
"""

import os
import random
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from geometry_msgs.msg import Pose
import math

class WarehouseObstacleSpawner(Node):
    """
    仓库动态障碍物生成器
    功能：在每个episode开始时，随机生成不同数量/位置/大小的静态障碍物
    """

    def __init__(self):
        super().__init__('warehouse_obstacle_spawner')

        # Gazebo服务客户端
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_client = self.create_client(DeleteEntity, '/delete_entity')

        # 等待服务可用
        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待spawn_entity服务...')

        # 仓库参数
        self.arena_size = 8.0  # 8m x 8m (circle_swap_arena大小)
        self.safe_margin = 1.8  # 距离边界的安全距离（增加）
        self.min_obstacle_spacing = 1.5  # 障碍物之间最小间距（增加）

        # 障碍物类型定义
        self.obstacle_types = {
            'small_box': {
                'size': (0.4, 0.4, 0.5),  # 小箱子
                'model_name': 'warehouse_box_small'
            },
            'medium_box': {
                'size': (0.6, 0.6, 0.8),  # 中箱子
                'model_name': 'warehouse_box_medium'
            },
            'large_box': {
                'size': (0.8, 0.8, 1.0),  # 大箱子
                'model_name': 'warehouse_box_large'
            },
            'shelf': {
                'size': (0.5, 2.0, 1.5),  # 货架（长条形）
                'model_name': 'warehouse_shelf'
            },
            'pallet': {
                'size': (1.0, 1.2, 0.3),  # 托盘
                'model_name': 'warehouse_pallet'
            }
        }

        # 当前episode的障碍物列表
        self.current_obstacles = []

        self.get_logger().info('仓库障碍物生成器初始化完成')

    def generate_random_obstacles(self, min_count=3, max_count=8):
        """
        生成随机障碍物配置

        Args:
            min_count: 最少障碍物数量
            max_count: 最多障碍物数量

        Returns:
            list: 障碍物配置列表 [(type, x, y, yaw), ...]
        """
        num_obstacles = random.randint(min_count, max_count)
        obstacles = []
        occupied_positions = []

        # 保护中心区域（机器人spawn位置）- 增加到2.5m
        protected_radius = 2.5

        attempts = 0
        max_attempts = 100

        while len(obstacles) < num_obstacles and attempts < max_attempts:
            attempts += 1

            # 随机选择障碍物类型
            obs_type = random.choice(list(self.obstacle_types.keys()))
            obs_config = self.obstacle_types[obs_type]

            # 随机位置（在arena范围内）
            x = random.uniform(-self.arena_size/2 + self.safe_margin,
                             self.arena_size/2 - self.safe_margin)
            y = random.uniform(-self.arena_size/2 + self.safe_margin,
                             self.arena_size/2 - self.safe_margin)

            # 检查是否在保护区域内
            dist_from_center = math.sqrt(x**2 + y**2)
            if dist_from_center < protected_radius:
                continue

            # 检查与已有障碍物的距离
            too_close = False
            for (ox, oy) in occupied_positions:
                dist = math.sqrt((x - ox)**2 + (y - oy)**2)
                if dist < self.min_obstacle_spacing:
                    too_close = True
                    break

            if too_close:
                continue

            # 随机朝向（对货架很重要）
            yaw = random.uniform(0, 2 * math.pi)

            obstacles.append((obs_type, x, y, yaw))
            occupied_positions.append((x, y))

            self.get_logger().info(
                f'生成障碍物 {len(obstacles)}/{num_obstacles}: {obs_type} @ ({x:.2f}, {y:.2f})'
            )

        return obstacles

    def spawn_obstacle(self, obs_type, x, y, yaw, index):
        """生成单个障碍物到Gazebo"""
        obs_config = self.obstacle_types[obs_type]
        model_name = f"{obs_config['model_name']}_{index}"

        # 构造SDF模型
        sdf = self._generate_sdf(obs_type, obs_config)

        # 创建spawn请求
        request = SpawnEntity.Request()
        request.name = model_name
        request.xml = sdf
        request.robot_namespace = ''

        # 设置位姿
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = obs_config['size'][2] / 2  # 高度的一半
        pose.orientation.z = math.sin(yaw / 2)
        pose.orientation.w = math.cos(yaw / 2)
        request.initial_pose = pose

        # 调用服务
        future = self.spawn_client.call_async(request)
        self.current_obstacles.append(model_name)

        return future

    def _generate_sdf(self, obs_type, config):
        """生成SDF模型XML"""
        sx, sy, sz = config['size']

        sdf_template = f'''<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{config['model_name']}">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <box>
            <size>{sx} {sy} {sz}</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>{sx} {sy} {sz}</size>
          </box>
        </geometry>
        <material>
          <ambient>0.7 0.5 0.3 1</ambient>
          <diffuse>0.7 0.5 0.3 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>'''
        return sdf_template

    def clear_all_obstacles(self):
        """清除所有当前障碍物"""
        for model_name in self.current_obstacles:
            request = DeleteEntity.Request()
            request.name = model_name
            self.delete_client.call_async(request)
            self.get_logger().info(f'删除障碍物: {model_name}')

        self.current_obstacles.clear()

    def reset_environment(self):
        """重置环境 - 清除旧障碍物，生成新障碍物"""
        self.get_logger().info('===== 重置仓库环境 =====')

        # 清除旧障碍物
        self.clear_all_obstacles()

        # 等待一小段时间确保删除完成
        rclpy.spin_once(self, timeout_sec=0.5)

        # 生成新障碍物
        obstacles = self.generate_random_obstacles(min_count=4, max_count=10)

        for i, (obs_type, x, y, yaw) in enumerate(obstacles):
            self.spawn_obstacle(obs_type, x, y, yaw, i)

        self.get_logger().info(f'环境重置完成，生成了 {len(obstacles)} 个障碍物')


def main(args=None):
    rclpy.init(args=args)
    spawner = WarehouseObstacleSpawner()

    # 初始生成一批障碍物
    spawner.reset_environment()

    try:
        rclpy.spin(spawner)
    except KeyboardInterrupt:
        pass
    finally:
        spawner.clear_all_obstacles()
        spawner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
