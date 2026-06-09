from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Define launch parameters
    map_number_arg = DeclareLaunchArgument(
        'map_number',
        default_value='1',
        description='地图编号: 1=map1, 2=map2, 3=corridor_swap, 4=intersection, 5=warehouse_aisles'
    )
    robot_number_arg = DeclareLaunchArgument(
        'robot_number',
        default_value='3',
        description='机器人数量'
    )
    use_random_mode_arg = DeclareLaunchArgument(
        'use_random_mode',
        default_value='false',
        description='位置模式: true=随机分布, false=固定位置'
    )

    goal_termination_mode_arg = DeclareLaunchArgument(
        'goal_termination_mode',
        default_value='all',
        description="目标终止模式: 'any' 任意机器人到达即结束, 'all' 全部到达才结束"
    )
    stuck_enabled_arg = DeclareLaunchArgument(
        'stuck_enabled',
        default_value='true',
        description='是否启用卡住检测(无进展提前截断)'
    )
    stuck_min_progress_arg = DeclareLaunchArgument(
        'stuck_min_progress',
        default_value='0.02',
        description='卡住检测: 单步最小进展(米)，低于则计入卡住'
    )
    stuck_max_steps_arg = DeclareLaunchArgument(
        'stuck_max_steps',
        default_value='40',
        description='卡住检测: 连续多少步无进展则判定卡住'
    )
    stuck_check_after_steps_arg = DeclareLaunchArgument(
        'stuck_check_after_steps',
        default_value='20',
        description='卡住检测: episode开始多少步后才开始检查'
    )
    stuck_penalty_arg = DeclareLaunchArgument(
        'stuck_penalty',
        default_value='-10.0',
        description='卡住检测: 卡住时对卡住机器人施加的惩罚'
    )

    # Create the training node (只启动训练，不启动环境)
    matd3_node = Node(
        package='start_reinforcement_learning',
        executable='run_matd3',
        namespace='matd3_ns',
        name='matd3_node',
        output='screen',
        parameters=[
            {'map_number': LaunchConfiguration('map_number')},
            {'robot_number': LaunchConfiguration('robot_number')},
            {'use_random_mode': LaunchConfiguration('use_random_mode')},
            {'goal_termination_mode': LaunchConfiguration('goal_termination_mode')},
            {'stuck_enabled': LaunchConfiguration('stuck_enabled')},
            {'stuck_min_progress': LaunchConfiguration('stuck_min_progress')},
            {'stuck_max_steps': LaunchConfiguration('stuck_max_steps')},
            {'stuck_check_after_steps': LaunchConfiguration('stuck_check_after_steps')},
            {'stuck_penalty': LaunchConfiguration('stuck_penalty')},
        ]
    )
    
    return LaunchDescription([
        map_number_arg,
        robot_number_arg,
        use_random_mode_arg,
        goal_termination_mode_arg,
        stuck_enabled_arg,
        stuck_min_progress_arg,
        stuck_max_steps_arg,
        stuck_check_after_steps_arg,
        stuck_penalty_arg,
        matd3_node
    ])



