#!/usr/bin/env python3
"""
Launch Nav2 stack for multi-robot system
Each robot gets its own Nav2 stack (map_server, planner_server, etc.)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, PushRosNamespace
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get package directories
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    pkg_dir = get_package_share_directory('start_orca_nav')
    
    # Declare arguments
    robot_number_arg = DeclareLaunchArgument(
        'robot_number',
        default_value='4',
        description='Number of robots'
    )
    
    map_arg = DeclareLaunchArgument(
        'map',
        default_value=os.path.join(
            get_package_share_directory('start_rl_environment'),
            'maps', 'corridor_swap.yaml'
        ),
        description='Full path to map yaml file'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    
    # Get values
    robot_number = LaunchConfiguration('robot_number')
    map_yaml = LaunchConfiguration('map')
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    # Create launch description
    ld = LaunchDescription()
    
    # Add arguments
    ld.add_action(robot_number_arg)
    ld.add_action(map_arg)
    ld.add_action(use_sim_time_arg)
    
    # For each robot, launch a Nav2 stack
    # We'll do this dynamically in the launch file or statically for 4 robots
    
    # Robot 0
    robot0_group = GroupAction([
        PushRosNamespace('my_bot0'),
        
        # Map server
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{
                'yaml_filename': map_yaml,
                'use_sim_time': use_sim_time
            }]
        ),
        
        # Planner server
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'planner_plugins': ['GridBased'],
                'GridBased': {
                    'plugin': 'nav2_navfn_planner/NavfnPlanner',
                    'tolerance': 0.5,
                    'use_astar': False,
                    'allow_unknown': True
                }
            }]
        ),
        
        # Lifecycle manager
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'autostart': True,
                'node_names': ['map_server', 'planner_server']
            }]
        )
    ])
    
    # Robot 1
    robot1_group = GroupAction([
        PushRosNamespace('my_bot1'),
        
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{
                'yaml_filename': map_yaml,
                'use_sim_time': use_sim_time
            }]
        ),
        
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'planner_plugins': ['GridBased'],
                'GridBased': {
                    'plugin': 'nav2_navfn_planner/NavfnPlanner',
                    'tolerance': 0.5,
                    'use_astar': False,
                    'allow_unknown': True
                }
            }]
        ),
        
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'autostart': True,
                'node_names': ['map_server', 'planner_server']
            }]
        )
    ])
    
    # Robot 2
    robot2_group = GroupAction([
        PushRosNamespace('my_bot2'),
        
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{
                'yaml_filename': map_yaml,
                'use_sim_time': use_sim_time
            }]
        ),
        
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'planner_plugins': ['GridBased'],
                'GridBased': {
                    'plugin': 'nav2_navfn_planner/NavfnPlanner',
                    'tolerance': 0.5,
                    'use_astar': False,
                    'allow_unknown': True
                }
            }]
        ),
        
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'autostart': True,
                'node_names': ['map_server', 'planner_server']
            }]
        )
    ])
    
    # Robot 3
    robot3_group = GroupAction([
        PushRosNamespace('my_bot3'),
        
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{
                'yaml_filename': map_yaml,
                'use_sim_time': use_sim_time
            }]
        ),
        
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'planner_plugins': ['GridBased'],
                'GridBased': {
                    'plugin': 'nav2_navfn_planner/NavfnPlanner',
                    'tolerance': 0.5,
                    'use_astar': False,
                    'allow_unknown': True
                }
            }]
        ),
        
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'autostart': True,
                'node_names': ['map_server', 'planner_server']
            }]
        )
    ])
    
    # Add robot groups
    ld.add_action(robot0_group)
    ld.add_action(robot1_group)
    ld.add_action(robot2_group)
    ld.add_action(robot3_group)
    
    return ld
