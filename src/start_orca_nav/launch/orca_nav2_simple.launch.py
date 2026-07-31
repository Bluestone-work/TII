#!/usr/bin/env python3
"""
Simplified launch: assumes Gazebo and Nav2 are already running
Only launches ORCA navigation node
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory('start_orca_nav')
    env_pkg_dir = get_package_share_directory('start_rl_environment')
    
    # Arguments
    robot_number_arg = DeclareLaunchArgument(
        'robot_number',
        default_value='4',
        description='Number of robots'
    )
    
    map_file_arg = DeclareLaunchArgument(
        'map_file',
        default_value=os.path.join(env_pkg_dir, 'maps', 'corridor_swap.yaml'),
        description='Map file path'
    )
    
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz'
    )
    
    # Get configurations
    robot_number = LaunchConfiguration('robot_number')
    map_file = LaunchConfiguration('map_file')
    use_rviz = LaunchConfiguration('use_rviz')
    
    # ORCA navigation node
    orca_nav_node = Node(
        package='start_orca_nav',
        executable='orca_nav_node_nav2',
        name='orca_nav_node',
        output='screen',
        parameters=[{
            'robot_number': robot_number,
            'robot_radius': 0.35,
            'neighbor_distance': 5.0,
            'time_horizon': 2.0,
            'goal_tolerance': 0.3,
            'map_file': map_file,
            'use_sim_time': True
        }]
    )
    
    # RViz
    rviz_config = os.path.join(pkg_dir, 'rviz', 'multi_robot_nav.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        condition=IfCondition(use_rviz),
        parameters=[{'use_sim_time': True}]
    )
    
    return LaunchDescription([
        robot_number_arg,
        map_file_arg,
        use_rviz_arg,
        orca_nav_node,
        rviz_node
    ])
