import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node

def generate_launch_description():
    # 1. 获取参数
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    robot_name = LaunchConfiguration('robot_name') # 例如：my_bot0
    x_pos = LaunchConfiguration('x')
    y_pos = LaunchConfiguration('y')
    z_pos = LaunchConfiguration('z')
    turn_around = LaunchConfiguration('rotation')
    robot_name_prefix = LaunchConfiguration('robot_name_prefix') # 例如：my_bot0/

    # 2. 定位 Xacro 文件
    pkg_path = os.path.join(get_package_share_directory('start_rl_environment'))
    xacro_file = os.path.join(pkg_path, 'description', 'robot.urdf.xacro')

    # 3. 【核心修复】使用 Command 动态生成 URDF，并传入 robot_name 参数
    # 这样 lidara.xacro 里的 $(arg robot_name) 才能变成 'my_bot0' 而不是默认的 'my_bot'
    robot_description_content = Command([
        'xacro ', xacro_file, 
        ' robot_name:=', robot_name
    ])

    # 4. Robot State Publisher 参数
    # frame_prefix 确保发布的静态 TF (base_link等) 也有 my_bot0/ 前缀
    params = {
        'frame_prefix': robot_name_prefix, 
        'robot_description': robot_description_content, 
        'use_sim_time': use_sim_time
    }
    
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace=robot_name, 
        output='screen',
        parameters=[params]
    )

    # 5. Spawn Entity (在 Gazebo 中生成模型)
    spawn_entity = Node(
        package='gazebo_ros', 
        executable='spawn_entity.py', 
        name='spawn_entity', 
        namespace=robot_name,
        arguments=[
            '-topic', 'robot_description', 
            '-entity', robot_name,
            '-robot_namespace', robot_name, 
            '-x', x_pos, '-y', y_pos, '-z', z_pos, '-Y', turn_around
        ],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('robot_name', default_value='my_bot0'),
        DeclareLaunchArgument('robot_name_prefix', default_value='my_bot0/'),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('z', default_value='0.1'),
        DeclareLaunchArgument('rotation', default_value='0.0'),
        
        node_robot_state_publisher,
        spawn_entity
    ])