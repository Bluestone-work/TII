# 动态障碍物问题修复总结

## 问题确认 ✅

从日志诊断结果：
- ✅ 成功spawn 4/4 个棕色方块（静态障碍物）
- ✅ blocked_points=4（A*规划器收到了4个静态障碍物）
- ❌ **但环境中还有其他4个动态障碍物（dyn_obs_X）没有被考虑**

## 根本原因

**World文件中预定义的动态障碍物不在 blocked_points 中**

1. World文件中有8个 `dyn_obs_X` 障碍物
2. 这些障碍物由 `obstacle_mover.py` 控制移动
3. 但它们的位置**不在** `spawned_static_obstacles` 列表中
4. A*规划时只考虑了4个棕色方块，忽略了8个动态障碍物
5. **导致路径穿过动态障碍物**

## 已实现的临时修复

在reset时，尝试从激光雷达聚类中获取动态障碍物位置：

```python
# 【关键修复】添加从激光雷达聚类中检测到的动态障碍物
if hasattr(self, '_obstacle_cluster_history') and self._obstacle_cluster_history:
    recent_clusters = self._obstacle_cluster_history[-1]
    for cluster in recent_clusters:
        if isinstance(cluster, dict) and 'center' in cluster:
            cx, cy = cluster['center']
            blocked_points.append((float(cx), float(cy)))
```

### 局限性

- ⚠️ Reset时激光雷达数据为空，无法检测到障碍物
- ⚠️ 只能检测视野内的障碍物
- ⚠️ 需要等几帧后才有聚类数据

## 推荐的完整解决方案

### 方案1: 订阅 /model_states（最准确）

```python
from gazebo_msgs.msg import ModelStates

class IndependentRobotEnv:
    def __init__(self, ...):
        # 订阅Gazebo的model states
        self.model_states_sub = self.node.create_subscription(
            ModelStates,
            '/model_states',
            self._model_states_callback,
            10
        )
        self.dynamic_obstacle_positions = {}
    
    def _model_states_callback(self, msg):
        """获取所有dyn_obs_X的实时位置"""
        for i, name in enumerate(msg.name):
            if name.startswith('dyn_obs'):
                pose = msg.pose[i]
                self.dynamic_obstacle_positions[name] = (
                    pose.position.x,
                    pose.position.y
                )
    
    def reset(self, ...):
        # 在规划时加入动态障碍物
        blocked_points = [...]  # 静态障碍物
        
        # 添加动态障碍物
        for pos in self.dynamic_obstacle_positions.values():
            blocked_points.append(pos)
```

### 方案2: 使用固定的保守估计位置（简单但不精确）

在 `_DYN_OBS_SPAWNS` 中定义动态障碍物的典型活动区域，作为静态避障点。

优点：简单，不需要额外订阅
缺点：不准确，可能过度保守或遗漏

### 方案3: 延迟规划（不推荐）

Reset后等待几帧，让激光雷达聚类数据积累，然后再规划。

缺点：增加reset延迟，影响训练效率

## 立即可用的workaround

如果你现在就要测试，最快的方法是：

### 临时禁用动态障碍物

修改 `obstacle_mover.py`，让动态障碍物保持在场地边缘不动（初始位置±3.5m），这样它们不会干扰中心区域的路径规划。

或者在launch文件中不启动 `obstacle_mover.py` 节点。

## 下一步

1. **验证当前修复**：重新运行，看激光聚类是否能捕获动态障碍物
2. **实现方案1**：订阅 /model_states 获取精确位置（推荐）
3. **调整参数**：如果动态障碍物影响太大，可以减少数量或降低速度

---
修复日期: 2026-07-03
问题类型: 动态障碍物未被路径规划考虑
