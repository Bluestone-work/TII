from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """启动RecurrentPPO训练"""
    
    # 声明参数
    robot_number_arg = DeclareLaunchArgument(
        'robot_number',
        default_value='1',
        description='Number of robots'
    )
    
    map_number_arg = DeclareLaunchArgument(
        'map_number',
        default_value='3',
        description='Map number to use'
    )
    
    total_timesteps_arg = DeclareLaunchArgument(
        'total_timesteps',
        default_value='1000000',
        description='Total training timesteps'
    )
    
    # 训练节点
    training_node = Node(
        package='sb3_training',
        executable='train_ppo',
        name='ppo_training',
        output='screen',
        parameters=[{
            'robot_number': LaunchConfiguration('robot_number'),
            'map_number': LaunchConfiguration('map_number'),
        }],
        arguments=[
            '--robot_number', LaunchConfiguration('robot_number'),
            '--map_number', LaunchConfiguration('map_number'),
            '--total_timesteps', LaunchConfiguration('total_timesteps'),
        ]
    )
    
    return LaunchDescription([
        robot_number_arg,
        map_number_arg,
        total_timesteps_arg,
        training_node,
    ])
