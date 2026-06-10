#!/bin/bash
# 同时发送多个机器人的目标点
# Usage: ./send_multi_goals.sh

# 显示帮助
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: $0 [preset_name]"
    echo ""
    echo "Presets:"
    echo "  test1              - 两个机器人交叉路径测试（默认）"
    echo "  test2              - 两个机器人对向测试"
    echo "  test3              - 两个机器人同向测试"
    echo "  test4              - 四个机器人交叉测试"
    echo "  corridor_swap_2    - Corridor Swap 2机器人位置交换"
    echo "  corridor_swap_4    - Corridor Swap 4机器人位置交换"
    echo "  intersection_2     - Intersection 2机器人位置交换"
    echo "  intersection_4     - Intersection 4机器人位置交换"
    echo "  custom             - 自定义目标（需修改脚本）"
    echo ""
    echo "Examples:"
    echo "  $0                      # 使用默认test1"
    echo "  $0 corridor_swap_2      # Corridor Swap地图2机器人交换"
    echo "  $0 intersection_4       # Intersection地图4机器人交换"
    exit 0
fi

PRESET=${1:-test1}

source /home/wj/anaconda3/etc/profile.d/conda.sh
conda activate ros2
source install/setup.bash

echo "🎯 使用预设: $PRESET"
echo ""

case $PRESET in
    test1)
        echo "📍 Test1: 两个机器人交叉路径（推荐）"
        echo "  robot0: (-7.5, 4) → (-7.5, -3)  (上 → 下)"
        echo "  robot1: (-7.5, -3) → (-7.5, 4)  (下 → 上)"
        echo "  两个机器人会在走廊中间相遇，测试ORCA避碰"
        echo ""
        
        # 同时发送两个目标
        ros2 topic pub --once /robot0/goal_pose geometry_msgs/msg/PoseStamped "{
          header: {frame_id: 'map'},
          pose: {
            position: {x: -7.5, y: -3.0, z: 0.0},
            orientation: {w: 1.0}
          }
        }" &
        
        ros2 topic pub --once /robot1/goal_pose geometry_msgs/msg/PoseStamped "{
          header: {frame_id: 'map'},
          pose: {
            position: {x: -7.5, y: 4.0, z: 0.0},
            orientation: {w: 1.0}
          }
        }" &
        
        wait
        ;;
        
    test2)
        echo "📍 Test2: 两个机器人对向测试"
        echo "  robot0: (-8, 0) → (-6, 0)   (左 → 右)"
        echo "  robot1: (-6, 0) → (-8, 0)   (右 → 左)"
        echo ""
        
        ros2 topic pub --once /robot0/goal_pose geometry_msgs/msg/PoseStamped "{
          header: {frame_id: 'map'},
          pose: {
            position: {x: -6.0, y: 0.0, z: 0.0},
            orientation: {w: 1.0}
          }
        }" &
        
        ros2 topic pub --once /robot1/goal_pose geometry_msgs/msg/PoseStamped "{
          header: {frame_id: 'map'},
          pose: {
            position: {x: -8.0, y: 0.0, z: 0.0},
            orientation: {w: 1.0}
          }
        }" &
        
        wait
        ;;
        
    test3)
        echo "📍 Test3: 两个机器人同向测试"
        echo "  robot0: (-8, 1) → (-5, -2)  (后车)"
        echo "  robot1: (-7, 0) → (-5, -2)  (前车，同目标)"
        echo ""
        
        ros2 topic pub --once /robot0/goal_pose geometry_msgs/msg/PoseStamped "{
          header: {frame_id: 'map'},
          pose: {
            position: {x: -5.0, y: -2.0, z: 0.0},
            orientation: {w: 1.0}
          }
        }" &
        
        ros2 topic pub --once /robot1/goal_pose geometry_msgs/msg/PoseStamped "{
          header: {frame_id: 'map'},
          pose: {
            position: {x: -5.0, y: -2.0, z: 0.0},
            orientation: {w: 1.0}
          }
        }" &
        
        wait
        ;;
        
    test4)
        echo "📍 Test4: 四个机器人交叉测试"
        echo "  robot0: (-8, 4) → (-5, -3)"
        echo "  robot1: (-5, 4) → (-8, -3)"
        echo "  robot2: (-8, -3) → (-5, 4)"
        echo "  robot3: (-5, -3) → (-8, 4)"
        echo ""
        
        ros2 topic pub --once /robot0/goal_pose geometry_msgs/msg/PoseStamped "{
          header: {frame_id: 'map'},
          pose: {
            position: {x: -5.0, y: -3.0, z: 0.0},
            orientation: {w: 1.0}
          }
        }" &
        
        ros2 topic pub --once /robot1/goal_pose geometry_msgs/msg/PoseStamped "{
          header: {frame_id: 'map'},
          pose: {
            position: {x: -8.0, y: -3.0, z: 0.0},
            orientation: {w: 1.0}
          }
        }" &
        
        ros2 topic pub --once /robot2/goal_pose geometry_msgs/msg/PoseStamped "{
          header: {frame_id: 'map'},
          pose: {
            position: {x: -5.0, y: 4.0, z: 0.0},
            orientation: {w: 1.0}
          }
        }" &
        
        ros2 topic pub --once /robot3/goal_pose geometry_msgs/msg/PoseStamped "{
          header: {frame_id: 'map'},
          pose: {
            position: {x: -8.0, y: 4.0, z: 0.0},
            orientation: {w: 1.0}
          }
        }" &
        
        wait
        ;;
        
    corridor_swap_2)
        echo "📍 Corridor Swap: 2机器人位置交换"
        echo "  robot0: 左侧(-7.5, 0) → 右侧(7.5, 0)"
        echo "  robot1: 右侧(7.5, 0) → 左侧(-7.5, 0)"
        echo "  测试走廊中对向会车和避让"
        echo ""
        
        ros2 topic pub --once /robot0/goal_pose geometry_msgs/msg/PoseStamped "{
          header: {frame_id: 'map'},
          pose: {
            position: {x: 7.5, y: 0.0, z: 0.0},
            orientation: {w: 1.0}
          }
        }" &
        
        ros2 topic pub --once /robot1/goal_pose geometry_msgs/msg/PoseStamped "{
          header: {frame_id: 'map'},
          pose: {
            position: {x: -7.5, y: 0.0, z: 0.0},
            orientation: {w: 1.0}
          }
        }" &
        
        wait
        ;;
        
    corridor_swap_4)
        echo "📍 Corridor Swap: 4机器人四角交换"
        echo "  robot0: 左中(-7.5, 0) → 右中(7.5, 0)"
        echo "  robot1: 右中(7.5, 0) → 左中(-7.5, 0)"
        echo "  robot2: 左上(-7.5, 4) → 右下(7.5, -4)"
        echo "  robot3: 左下(-7.5, -4) → 右上(7.5, 4)"
        echo "  测试复杂多机器人交互"
        echo ""
        
        ros2 topic pub --once /robot0/goal_pose geometry_msgs/msg/PoseStamped "{
          header: {frame_id: 'map'},
          pose: {
            position: {x: 7.5, y: 0.0, z: 0.0},
            orientation: {w: 1.0}
          }
        }" &
        
        ros2 topic pub --once /robot1/goal_pose geometry_msgs/msg/PoseStamped "{
          header: {frame_id: 'map'},
          pose: {
            position: {x: -7.5, y: 0.0, z: 0.0},
            orientation: {w: 1.0}
          }
        }" &
        
        ros2 topic pub --once /robot2/goal_pose geometry_msgs/msg/PoseStamped "{
          header: {frame_id: 'map'},
          pose: {
            position: {x: 7.5, y: -4.0, z: 0.0},
            orientation: {w: 1.0}
          }
        }" &
        
        ros2 topic pub --once /robot3/goal_pose geometry_msgs/msg/PoseStamped "{
          header: {frame_id: 'map'},
          pose: {
            position: {x: 7.5, y: 4.0, z: 0.0},
            orientation: {w: 1.0}
          }
        }" &
        
        wait
        ;;
        
    intersection_2)
        echo "📍 Intersection: 2机器人位置交换"
        echo "  robot0: 左侧(-7.0, 0) → 右侧(7.0, 0)"
        echo "  robot1: 右侧(7.0, 0) → 左侧(-7.0, 0)"
        echo "  测试十字路口对向通过"
        echo ""
        
        ros2 topic pub --once /robot0/goal_pose geometry_msgs/msg/PoseStamped "{
          header: {frame_id: 'map'},
          pose: {
            position: {x: 7.0, y: 0.0, z: 0.0},
            orientation: {w: 1.0}
          }
        }" &
        
        ros2 topic pub --once /robot1/goal_pose geometry_msgs/msg/PoseStamped "{
          header: {frame_id: 'map'},
          pose: {
            position: {x: -7.0, y: 0.0, z: 0.0},
            orientation: {w: 1.0}
          }
        }" &
        
        wait
        ;;
        
    intersection_4)
        echo "📍 Intersection: 4机器人四方向交换"
        echo "  robot0: 左侧(-7.0, 0) → 右侧(7.0, 0)"
        echo "  robot1: 右侧(7.0, 0) → 左侧(-7.0, 0)"
        echo "  robot2: 下方(0, -7.0) → 上方(0, 7.0)"
        echo "  robot3: 上方(0, 7.0) → 下方(0, -7.0)"
        echo "  测试四方向同时通过十字路口"
        echo ""
        
        ros2 topic pub --once /robot0/goal_pose geometry_msgs/msg/PoseStamped "{
          header: {frame_id: 'map'},
          pose: {
            position: {x: 7.0, y: 0.0, z: 0.0},
            orientation: {w: 1.0}
          }
        }" &
        
        ros2 topic pub --once /robot1/goal_pose geometry_msgs/msg/PoseStamped "{
          header: {frame_id: 'map'},
          pose: {
            position: {x: -7.0, y: 0.0, z: 0.0},
            orientation: {w: 1.0}
          }
        }" &
        
        ros2 topic pub --once /robot2/goal_pose geometry_msgs/msg/PoseStamped "{
          header: {frame_id: 'map'},
          pose: {
            position: {x: 0.0, y: 7.0, z: 0.0},
            orientation: {w: 1.0}
          }
        }" &
        
        ros2 topic pub --once /robot3/goal_pose geometry_msgs/msg/PoseStamped "{
          header: {frame_id: 'map'},
          pose: {
            position: {x: 0.0, y: -7.0, z: 0.0},
            orientation: {w: 1.0}
          }
        }" &
        
        wait
        ;;
    
    custom)
        echo "📍 Custom: 自定义目标"
        echo "请修改脚本中的custom部分设置您的目标"
        echo ""
        
        # 在这里添加你的自定义目标
        ros2 topic pub --once /robot0/goal_pose geometry_msgs/msg/PoseStamped "{
          header: {frame_id: 'map'},
          pose: {
            position: {x: 0.0, y: 0.0, z: 0.0},
            orientation: {w: 1.0}
          }
        }" &
        
        ros2 topic pub --once /robot1/goal_pose geometry_msgs/msg/PoseStamped "{
          header: {frame_id: 'map'},
          pose: {
            position: {x: 0.0, y: 0.0, z: 0.0},
            orientation: {w: 1.0}
          }
        }" &
        
        wait
        ;;
        
    *)
        echo "❌ 未知预设: $PRESET"
        echo "使用 -h 或 --help 查看可用预设"
        exit 1
        ;;
esac

echo ""
echo "✅ 所有目标已发送！"
echo "💡 提示: 使用以下命令监控机器人状态:"
echo "   tail -f /tmp/orca_narrow.log | grep -E '(robot0|robot1).*(DWA|ORCA|reached)'"
