# start_rl_environment_tb3

ROS2/Gazebo TurtleBot3 simulation environment for multi-robot navigation experiments.

## Launch

Headless launch for training:

```bash
ros2 launch start_rl_environment_tb3 main_headless.launch.py map_number:=4 robot_number:=4
```

RViz launch for visual debugging:

```bash
ros2 launch start_rl_environment_tb3 main.launch.py map_number:=4 robot_number:=4
```

## Maps

| `map_number` | world |
| --- | --- |
| 1 | `map1` |
| 2 | `map2` |
| 3 | `corridor_swap` |
| 4 | `intersection` |
| 5 | `warehouse_aisles` |
| 6 | `interaction_hub` |
| 7 | `interaction_hub_mini` |
| 8 | `circle_swap_arena` |
| 9 | `warehouse_dynamic` |

The package includes the map, world, URDF, and obstacle-motion assets required by the provided training and evaluation scripts.
