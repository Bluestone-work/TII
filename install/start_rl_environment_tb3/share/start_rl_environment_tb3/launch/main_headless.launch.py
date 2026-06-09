#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无GUI版本的launch文件，专门用于强化学习训练
速度更快，不会卡在GUI加载
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, ExecuteProcess
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

def launch_map_server(context, *args, **kwargs):
    pkg_share = get_package_share_directory('start_rl_environment_tb3')
    
    # 1. 获取命令行传入的 map_number
    map_num = context.perform_substitution(LaunchConfiguration('map_number'))
    
    # 2. 映射逻辑
    map_mapping = {
        '1': 'map1', 
        '2': 'map2', 
        '3': 'corridor_swap', 
        '4': 'intersection',
        '5': 'warehouse_aisles',
        '6': 'interaction_hub',
        '7': 'interaction_hub_mini'
    }
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
            'use_sim_time': True
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

def launch_gzserver(context, *args, **kwargs):
    """只启动Gazebo服务器，不启动GUI"""
    pkg = get_package_share_directory('start_rl_environment_tb3')
    map_number = context.perform_substitution(LaunchConfiguration('map_number')).strip()
    gazebo_master_uri = os.environ.get('GAZEBO_MASTER_URI', 'http://127.0.0.1:11345')
    
    worlds = {
        '1': os.path.join(pkg, 'worlds', 'map1.world'),
        '2': os.path.join(pkg, 'worlds', 'map2.world'),
        '3': os.path.join(pkg, 'worlds', 'corridor_swap.world'),
        '4': os.path.join(pkg, 'worlds', 'intersection.world'),
        '5': os.path.join(pkg, 'worlds', 'warehouse_aisles.world'),
        '6': os.path.join(pkg, 'worlds', 'interaction_hub.world'),
        '7': os.path.join(pkg, 'worlds', 'interaction_hub_mini.world'),
    }
    
    world_path = worlds.get(map_number, worlds['1'])
    print(f"[INFO] Loading world (HEADLESS): {world_path}")
    print(f"[INFO] Using GAZEBO_MASTER_URI: {gazebo_master_uri}")
    
    # 只启动gzserver，不启动gzclient
    gzserver = ExecuteProcess(
        cmd=['gzserver', world_path, 
             '--seed=0',
             '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so',
             '-s', 'libgazebo_ros_force_system.so',
             '--verbose'],
        output='screen',
        additional_env={
            'GAZEBO_MASTER_URI': gazebo_master_uri,
        }
    )
    
    return [gzserver]

def generate_launch_description():
    pkg = get_package_share_directory('start_rl_environment_tb3')
    start_robots = os.path.join(pkg, 'launch', 'start_robots.launch.py')

    map_number = LaunchConfiguration('map_number')
    robot_number = LaunchConfiguration('robot_number')
    enable_rviz = LaunchConfiguration('enable_rviz')
    rviz_config = LaunchConfiguration('rviz_config')
    rviz_node_name = LaunchConfiguration('rviz_node_name')

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name=rviz_node_name,
        arguments=['-d', rviz_config],
        output='screen',
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(enable_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument('map_number', default_value='1'),
        DeclareLaunchArgument('robot_number', default_value='3'),
        DeclareLaunchArgument('enable_rviz', default_value='false'),
        DeclareLaunchArgument('num_obstacles', default_value='8',
                              description='动态障碍物激活数量 0~8'),
        DeclareLaunchArgument('obs_speed_scale', default_value='1.0',
                              description='障碍物速度全局缩放'),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=os.path.join(pkg, 'rviz', 'multi_robot.rviz'),
        ),
        DeclareLaunchArgument('rviz_node_name', default_value='rviz2'),
        
        # 启动Gazebo服务器（无GUI）
        OpaqueFunction(function=launch_gzserver),
        
        # 启动机器人
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(start_robots),
            launch_arguments={
                'map_number': map_number,
                'robot_number': robot_number,
            }.items()
        ),
        
        # 启动地图服务器
        OpaqueFunction(function=launch_map_server),
        rviz_node,

        # headless 训练也需要显式启动动态障碍控制节点，否则障碍物会保持静止。
        Node(
            package='start_rl_environment_tb3',
            executable='obstacle_mover.py',
            name='obstacle_mover',
            output='screen',
            parameters=[{
                'map_number': LaunchConfiguration('map_number'),
                'num_obstacles': LaunchConfiguration('num_obstacles'),
                'speed_scale': LaunchConfiguration('obs_speed_scale'),
            }],
        ),
    ])
