#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def launch_setup(context, *args, **kwargs):
    # 1. 获取包路径 - 使用新包名
    pkg = get_package_share_directory('start_rl_environment_tb3')
    gazebo_ros_pkg = get_package_share_directory('gazebo_ros')
    
    # 2. 解析参数
    map_number = context.perform_substitution(LaunchConfiguration('map_number')).strip()
    world_file_override = context.perform_substitution(LaunchConfiguration('world_file')).strip()

    # 3. 定义 World 路径
    worlds = {
        '1': os.path.join(pkg, 'worlds', 'map1.world'),
        '2': os.path.join(pkg, 'worlds', 'map2.world'),
        '3': os.path.join(pkg, 'worlds', 'corridor_swap.world'),
        '4': os.path.join(pkg, 'worlds', 'intersection.world'),
        '5': os.path.join(pkg, 'worlds', 'warehouse_aisles.world'),
        '6': os.path.join(pkg, 'worlds', 'interaction_hub.world'),
        '7': os.path.join(pkg, 'worlds', 'interaction_hub_mini.world'),
    }

    # 4. 确定最终路径
    world_path = worlds.get('1')
    if world_file_override:
        world_path = world_file_override
    elif map_number in worlds:
        world_path = worlds[map_number]
    
    print(f"[INFO] Loading world for TurtleBot3: {world_path}")

    # 5. 检查文件是否存在 (调试关键)
    if not os.path.exists(world_path):
        print(f"[ERROR] World file does not exist: {world_path}")

    gazebo_launch = os.path.join(gazebo_ros_pkg, 'launch', 'gazebo.launch.py')
    
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gazebo_launch),
            launch_arguments=list({
                'world': world_path,
                # 仅保留详细日志；factory 插件由 gazebo_ros 启动链自身处理，避免重复加载导致 gzserver 崩溃
                'extra_gazebo_args': '--verbose'
            }.items())
        )
    ]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('map_number', default_value='1'),
        DeclareLaunchArgument('world_file', default_value=''),
        OpaqueFunction(function=launch_setup)
    ])
