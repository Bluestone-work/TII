Gazebo Classic (.world) 场景（用于 Multi-Goal Navigation：拥堵/死锁主线A）
生成日期：2026-01-07

文件说明
- corridor_swap.world：两侧房间+中间窄通道（会车/Swap，最容易复现 deadlock）
- intersection.world：十字路口交汇（吞吐/让行）
- warehouse_aisles.world：仓库货架通道（拥堵）

坐标范围与尺寸
- 世界边界：约 20m x 20m（墙在 x/y = ±10m）
- 墙高：1.5m；墙厚：0.2m

如何在你的仓库中使用（典型）
1) 将 .world 放到你的 start_rl_environment/worlds/ 目录（或仓库使用的 worlds 目录）
2) 修改 main.launch.py / gazebo.launch.py 中的 world 参数指向该文件：
   world:=<path>/corridor_swap.world
3) 起点/目标点请配合 poses_goals.yaml（上一份压缩包里提供了模板与随机采样脚本）

注意
- Gazebo world 与 Nav2 2D map 是两种不同文件：
  - .world/.sdf：3D 物理仿真环境
  - .pgm/.png + .yaml：2D 栅格地图（Nav2/map_server 用）
  你可以在该 world 上跑 slam_toolbox 再 map_saver 生成 2D map，实现 world 与 map 一致。
