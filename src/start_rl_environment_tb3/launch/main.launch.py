#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node

def launch_map_server(context, *args, **kwargs):
    # 使用新包名
    pkg_share = get_package_share_directory('start_rl_environment_tb3')
    
    # 1. 获取命令行传入的 map_number
    map_num = context.perform_substitution(LaunchConfiguration('map_number'))
    
    # 2. 映射逻辑 (必须与 start_robots.launch.py 保持一致!)
    map_mapping = {
        '1': 'map1', 
        '2': 'map2', 
        '3': 'corridor_swap', 
        '4': 'intersection',
        '5': 'warehouse_aisles',
        '6': 'interaction_hub',
        '7': 'interaction_hub_mini'
    }
    # 默认为 map1
    map_name = map_mapping.get(map_num, 'map1')
    
    # 3. 构建 yaml 文件路径
    map_yaml_path = os.path.join(pkg_share, 'maps', f'{map_name}.yaml')
    
    print(f"[INFO] Map Server loading: {map_yaml_path}")
    # 4. 定义节点
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'yaml_filename': map_yaml_path,
            'use_sim_time': True  # 重要：仿真模式下必须为 True
        }]
    )

    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'node_names': ['map_server']
        }]
    )
    
    return [map_server_node, lifecycle_manager_node]

def generate_launch_description():
    # 使用新包名
    pkg = get_package_share_directory('start_rl_environment_tb3')
    start_world = os.path.join(pkg, 'launch', 'start_world.launch.py')
    start_robots = os.path.join(pkg, 'launch', 'start_robots.launch.py')

    map_number = LaunchConfiguration('map_number')
    robot_number = LaunchConfiguration('robot_number')
    spawn_mode = LaunchConfiguration('spawn_mode')
    seed = LaunchConfiguration('seed')
    min_separation = LaunchConfiguration('min_separation')
    poses_goals_yaml = LaunchConfiguration('poses_goals_yaml')
    
    rviz_config_file = LaunchConfiguration('rviz_config')
    rviz_node_name = LaunchConfiguration('rviz_node_name')
    rviz_node = Node(
            package='rviz2',
            executable='rviz2',
            name=rviz_node_name,
            arguments=['-d', rviz_config_file],
            output='screen',
            parameters=[{'use_sim_time': True}]
        )

    return LaunchDescription([
        DeclareLaunchArgument('map_number', default_value='1'),
        DeclareLaunchArgument('robot_number', default_value='3'),
        DeclareLaunchArgument('spawn_mode', default_value='fixed'),
        DeclareLaunchArgument('seed', default_value='0'),
        DeclareLaunchArgument('min_separation', default_value='0.8'),
        DeclareLaunchArgument('poses_goals_yaml', default_value=''),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=PathJoinSubstitution([pkg, 'rviz', 'multi_robot.rviz'])
        ),
        DeclareLaunchArgument('rviz_node_name', default_value='rviz2'),
        DeclareLaunchArgument('num_obstacles', default_value='8',
                              description='动态障碍物激活数量 0~8'),
        DeclareLaunchArgument('obs_speed_scale', default_value='1.0',  # 必须带小数点，否则 ROS2 解析为 INTEGER
                              description='障碍物速度全局缩放'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(start_world),
            launch_arguments={'map_number': map_number}.items()
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(start_robots),
            launch_arguments={
                'map_number': map_number,
                'robot_number': robot_number,
                'spawn_mode': spawn_mode,
                'seed': seed,
                'min_separation': min_separation,
                'poses_goals_yaml': poses_goals_yaml,
            }.items()
        ),
        # 使用 OpaqueFunction 来动态启动 map_server
        OpaqueFunction(function=launch_map_server),
        rviz_node,

        # 动态障碍物控制节点（所有地图均启用，map_number 传入以选择对应轨迹）
        # 注意：不传 use_sim_time，节点内部使用 Wall Clock 驱动定时器，
        #       避免 Gazebo 启动竞态导致障碍物不动的问题。
        Node(
            package='start_rl_environment_tb3',
            executable='obstacle_mover.py',
            name='obstacle_mover',
            output='screen',
            parameters=[{
                'map_number':   LaunchConfiguration('map_number'),
                'num_obstacles': LaunchConfiguration('num_obstacles'),
                'speed_scale':   LaunchConfiguration('obs_speed_scale'),
            }],
        ),
    ])
