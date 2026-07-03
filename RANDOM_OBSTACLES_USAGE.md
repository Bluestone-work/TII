# 障碍物随机初始化功能说明

## 功能概述

新增了每次 episode reset 时随机初始化静态和动态障碍物位置的功能，避免障碍物重叠导致的碰撞倾倒问题，并增加训练场景的多样性。

## 新增参数

### 环境参数 (gnn_marl_env.py)

1. **num_static_obstacles** (int, 默认: 8)
   - 静态障碍物数量（0-8）
   - 静态障碍物不会移动，为灰色圆柱体

2. **random_obstacles** (bool, 默认: False)
   - 是否每次reset时随机初始化障碍物位置
   - 启用后会确保障碍物之间、障碍物与机器人spawn点之间有足够间隔

3. **num_dynamic_obstacles** (int, 已存在，默认: 8)
   - 动态障碍物数量（0-8）
   - 动态障碍物可以被 obstacle_mover 控制移动，为红色圆柱体

### 训练脚本参数 (train_gnn_mappo_full.py)

```bash
--num_obstacles N            # 动态障碍物数量（覆盖stage默认值）
--num_static_obstacles N     # 静态障碍物数量（默认8）
--random_obstacles           # 启用随机初始化（flag参数，无需赋值）
```

### 课程学习脚本参数 (run_curriculum.sh)

```bash
--num_obstacles N            # 动态障碍物数量
--num_static_obstacles N     # 静态障碍物数量
--random_obstacles           # 启用随机初始化```

## 使用示例

### 1. 基本使用（训练脚本）

```bash
# 使用随机障碍物初始化，动态障碍物3个，静态障碍物5个
python3 train_gnn_mappo_full.py \
  --num_agents 4 \
  --num_obstacles 3 \
  --num_static_obstacles 5 \
  --random_obstacles \
  --env_stage 1
```

### 2. 课程学习脚本使用

```bash
# 启用随机障碍物，自定义障碍物数量
./run_curriculum.sh \
  --run_suffix "map9_random_obstacles_test" \
  --start_stage 1 \
  --end_stage 4 \
  --num_obstacles 4 \
  --num_static_obstacles 6 \
  --random_obstacles \
  --gat_critic_mode gat \
  --graph_ablation dual_graph
```

### 3. 仅修改障碍物数量（不启用随机初始化）

```bash
# 使用固定位置，但只spawn部分障碍物
./run_curriculum.sh \
  --run_suffix "map9_fixed_4obs" \
  --start_stage 1 \
  --num_obstacles 4 \
  --num_static_obstacles 4
```

## 实现细节

### 随机spawn逻辑

1. **碰撞检测**
   - 障碍物之间最小间隔: 0.6m
   - 障碍物与机器人spawn点最小间隔: 1.0m
   - 最多尝试100次寻找有效位置

2. **spawn顺序**
   - 先spawn静态障碍物（固定在原地）
   - 再spawn动态障碍物（可被obstacle_mover移动）
   - 未使用的障碍物移到地图外(100, 100)

3. **支持地图**
   - Map 9 (warehouse_dynamic): spawn区域 [-3, 3] × [-3, 3]
   - Map 8 (circle_swap_arena): spawn区域 [-2.5, 2.5] × [-2.5, 2.5]
   - 其他地图暂不支持随机spawn

### 障碍物参数

- **静态障碍物**
  - 半径: 0.20m
  - 高度: 0.8m
  - 颜色: 灰色
  - 名称: static_obs_0 到 static_obs_7

- **动态障碍物**
  - 半径: 0.22m
  - 高度: 0.8m
  - 质量: 5.0kg
  - 颜色: 红色
  - 名称: dyn_obs_0 到 dyn_obs_7

## 注意事项

1. **只有 robot_id==0 时执行spawn**
   - 避免多个智能体重复spawn障碍物
   - 所有智能体共享同一组障碍物

2. **与 obstacle_mover 的交互**
   - 动态障碍物spawn后会被 obstacle_mover 节点控制移动
   - 静态障碍物保持在spawn位置不动

3. **默认行为**
   - 如果不指定 `--random_obstacles`，障碍物使用world文件中的固定位置
   - 未使用的障碍物会被移到地图外，不影响训练

4. **重叠问题已解决**
   - 随机spawn时会确保障碍物之间有足够间隔
   - 如果100次尝试都失败，会打印警告并跳过该障碍物

## 调试建议

1. **验证spawn位置**
   ```bash
   # 在Gazebo中观察障碍物是否正确spawn且不重叠
   # 红色圆柱 = 动态障碍物
   # 灰色圆柱 = 静态障碍物
   ```

2. **检查日志**
   ```bash
   # 查找spawn相关警告
   grep "无法为.*找到有效位置" curriculum_logs/*/stage*_train.log
   ```

3. **测试不同配置**
   ```bash
   # 少量障碍物（容易spawn）
   --num_obstacles 2 --num_static_obstacles 2 --random_obstacles

   # 大量障碍物（可能难以spawn）
   --num_obstacles 6 --num_static_obstacles 6 --random_obstacles
   ```

## 性能影响

- 每次reset增加约50-100ms的spawn开销
- 障碍物越多，寻找有效位置的时间越长
- 建议：动态+静态障碍物总数 ≤ 12个

## 与原有功能的兼容性

- ✅ 兼容所有现有的stage配置
- ✅ 兼容课程学习脚本
- ✅ 兼容失败场景重采样
- ✅ 兼容高冲突模式
- ✅ 可选功能，不影响现有训练流程
