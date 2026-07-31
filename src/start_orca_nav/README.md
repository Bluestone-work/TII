# start_orca_nav

ORCA-based multi-robot navigation baseline.

## Build

```bash
colcon build --packages-select start_orca_nav
source install/setup.bash
```

## Launch

```bash
ros2 launch start_orca_nav start_orca_nav.launch.py robot_number:=4
```

## Main parameters

| parameter | default | description |
| --- | --- | --- |
| `robot_number` | 4 | number of robots |
| `robot_radius` | 0.35 | robot collision radius in meters |
| `max_linear_speed` | 0.22 | maximum linear speed in m/s |
| `max_angular_speed` | 2.0 | maximum angular speed in rad/s |
| `neighbor_distance` | 5.0 | neighbor detection range in meters |
| `time_horizon` | 2.0 | ORCA time horizon in seconds |
| `use_nav2` | true | whether to use Nav2 global guidance |
