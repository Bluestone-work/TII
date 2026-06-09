#!/usr/bin/env python3
import os
import re
import tempfile
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def launch_setup(context, *args, **kwargs):
    """在运行时动态修改SDF并生成节点"""
    
    # 获取参数值
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')
    x_pos = LaunchConfiguration('x')
    y_pos = LaunchConfiguration('y')
    z_pos = LaunchConfiguration('z')
    turn_around = LaunchConfiguration('rotation')
    robot_name_prefix = LaunchConfiguration('robot_name_prefix')
    
    robot_name_value = context.perform_substitution(robot_name)
    robot_name_prefix_value = context.perform_substitution(robot_name_prefix)
    
    # 读取TurtleBot3模型文件
    TURTLEBOT3_MODEL = os.environ.get('TURTLEBOT3_MODEL', 'burger')
    turtlebot3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')
    
    # SDF文件路径
    model_folder = 'turtlebot3_' + TURTLEBOT3_MODEL
    sdf_file = os.path.join(turtlebot3_gazebo_dir, 'models', model_folder, 'model.sdf')
    
    # URDF文件路径
    urdf_file = os.path.join(turtlebot3_gazebo_dir, 'urdf', f'turtlebot3_{TURTLEBOT3_MODEL}.urdf')
    
    # 读取并修改SDF内容，取消注释namespace并设置为robot名称
    with open(sdf_file, 'r') as f:
        sdf_content = f.read()
    
    # 替换所有被注释的 <namespace> 标签
    modified_sdf = re.sub(
        r'<!-- <namespace>/tb3</namespace> -->',
        f'<namespace>/{robot_name_value}</namespace>',
        sdf_content
    )
    
    # 修改激光雷达的frame_id，加上命名空间前缀
    modified_sdf = re.sub(
        r'<frame_name>base_scan</frame_name>',
        f'<frame_name>{robot_name_value}/base_scan</frame_name>',
        modified_sdf
    )
    
    # 修改diff_drive插件的frame名称，确保TF树连接正确
    # odometry_frame: odom -> tb3_X/odom
    modified_sdf = re.sub(
        r'<odometry_frame>odom</odometry_frame>',
        f'<odometry_frame>{robot_name_value}/odom</odometry_frame>',
        modified_sdf
    )
    # robot_base_frame: base_footprint -> tb3_X/base_footprint  
    modified_sdf = re.sub(
        r'<robot_base_frame>base_footprint</robot_base_frame>',
        f'<robot_base_frame>{robot_name_value}/base_footprint</robot_base_frame>',
        modified_sdf
    )#不加这个多智能体识别不到
    
    # 创建临时SDF文件
    temp_sdf = tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False, prefix=f'{robot_name_value}_')
    temp_sdf.write(modified_sdf)
    temp_sdf_path = temp_sdf.name
    temp_sdf.close()
    
    print(f"[spawn_robots] Created modified SDF for {robot_name_value} at {temp_sdf_path}")
    
    # 读取URDF内容用于robot_state_publisher
    with open(urdf_file, 'r') as f:
        robot_description_content = f.read()
    
    # Robot State Publisher - 发布TF树
    params = {
        'frame_prefix': robot_name_prefix_value,
        'robot_description': robot_description_content,
        'use_sim_time': True
    }
    
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace=robot_name_value,
        output='screen',
        parameters=[params]
    )
    
    # Spawn Entity - 使用修改后的SDF文件
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_entity',
        namespace=robot_name_value,
        arguments=[
            '-file', temp_sdf_path,
            '-entity', robot_name_value,
            '-robot_namespace', robot_name_value,
            '-x', context.perform_substitution(x_pos),
            '-y', context.perform_substitution(y_pos),
            '-z', context.perform_substitution(z_pos),
            '-Y', context.perform_substitution(turn_around)
        ],
        output='screen'
    )
    
    return [node_robot_state_publisher, spawn_entity]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('robot_name', default_value='tb3_0'),
        DeclareLaunchArgument('robot_name_prefix', default_value='tb3_0/'),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('z', default_value='0.01'),
        DeclareLaunchArgument('rotation', default_value='0.0'),
        
        OpaqueFunction(function=launch_setup)
    ])
