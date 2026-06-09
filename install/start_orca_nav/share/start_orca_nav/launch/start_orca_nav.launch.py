#!/usr/bin/env python3
"""
Launch file for ORCA multi-robot navigation
"""

import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare launch arguments
    robot_number_arg = DeclareLaunchArgument(
        'robot_number',
        default_value='4',
        description='Number of robots'
    )
    
    robot_radius_arg = DeclareLaunchArgument(
        'robot_radius',
        default_value='0.35',
        description='Robot radius for collision avoidance'
    )
    
    max_linear_speed_arg = DeclareLaunchArgument(
        'max_linear_speed',
        default_value='0.22',
        description='Maximum linear speed'
    )
    
    max_angular_speed_arg = DeclareLaunchArgument(
        'max_angular_speed',
        default_value='2.0',
        description='Maximum angular speed'
    )
    
    neighbor_distance_arg = DeclareLaunchArgument(
        'neighbor_distance',
        default_value='5.0',
        description='Distance to consider other robots as neighbors'
    )
    
    time_horizon_arg = DeclareLaunchArgument(
        'time_horizon',
        default_value='2.0',
        description='ORCA time horizon for collision prediction'
    )
    
    navigation_mode_arg = DeclareLaunchArgument(
        'navigation_mode',
        default_value='orca',
        description='Navigation mode: orca (ORCA+DWA+Theta*) or nav2 (Pure Nav2)'
    )
    
    goal_tolerance_arg = DeclareLaunchArgument(
        'goal_tolerance',
        default_value='0.3',
        description='Goal reached tolerance in meters'
    )
    
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='启动RViz可视化'
    )
    
    # ORCA navigation node
    orca_nav_node = Node(
        package='start_orca_nav',
        executable='orca_nav_node',
        name='orca_nav_node',
        output='screen',
        parameters=[{
            'robot_number': LaunchConfiguration('robot_number'),
            'robot_radius': LaunchConfiguration('robot_radius'),
            'max_linear_speed': LaunchConfiguration('max_linear_speed'),
            'max_angular_speed': LaunchConfiguration('max_angular_speed'),
            'neighbor_distance': LaunchConfiguration('neighbor_distance'),
            'time_horizon': LaunchConfiguration('time_horizon'),
            'navigation_mode': LaunchConfiguration('navigation_mode'),
            'goal_tolerance': LaunchConfiguration('goal_tolerance'),
        }]
    )
    
    # RViz node
    pkg_dir = get_package_share_directory('start_orca_nav')
    rviz_config_file = os.path.join(pkg_dir, 'rviz', 'multi_robot_nav.rviz')
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen',
        condition=launch.conditions.IfCondition(LaunchConfiguration('use_rviz'))
    )
    
    return LaunchDescription([
        robot_number_arg,
        robot_radius_arg,
        max_linear_speed_arg,
        max_angular_speed_arg,
        neighbor_distance_arg,
        time_horizon_arg,
        navigation_mode_arg,
        goal_tolerance_arg,
        use_rviz_arg,
        orca_nav_node,
        # rviz_node,
    ])
