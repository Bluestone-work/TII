# 修复说明 - Spawn区域限制与课程学习

## 问题修复

### 1. 训练脚本Bug修复

**问题**: `start_curriculum_training.sh` 使用了 `--curriculum-learning` 参数，但 `matd3_main.py` 不支持该参数，导致训练直接失败退出。

**修复**: 
- 移除了不支持的 `--curriculum-learning` 参数
- 改用 `ros2 launch` 直接启动训练，通过launch参数传递配置
- 修改了脚本以使用 ROS2 launch 文件

**文件**: [start_curriculum_training.sh](start_curriculum_training.sh#L97-L104)

### 2. Spawn区域限制修复

**问题**: Map1 和 Map2 的机器人和目标点会生成在围墙外部。

**原因分析**:
通过检查 world 文件中的 `grey_wall` 定义：

**Map1** ([map1.world](src/start_rl_environment/worlds/map1.world)):
- 围墙形成矩形区域: X ∈ [-2.3, 3.5], Y ∈ [-11.0, -0.1]
- 但原配置使用 X ∈ [-9, 9], Y ∈ [-8.5, 8.5]，严重越界

**Map2** ([map2.world](src/start_rl_environment/worlds/map2.world)):
- 围墙形成L型区域: 
  - 左区: X ∈ [-1.9, 10.2], Y ∈ [-11.3, -0.1]
  - 右下区: X ∈ [2.6, 10.1], Y ∈ [-9.5, -0.1]
- 原配置同样越界

**修复**:
修改了 [spawn_presets.yaml](src/start_reinforcement_learning/config/spawn_presets.yaml):

```yaml
map1:
  # 围墙范围: X=[-2.3, 3.5], Y=[-11, -0.1]
  start_regions:
  - [-1.5, -10.3, 2.7, -3.5]   # 下半部分
  - [-1.5, -7.0, 2.7, -0.7]    # 上半部分
  goal_regions:
  - [-1.5, -10.3, 2.7, -3.5]
  - [-1.5, -7.0, 2.7, -0.7]

map2:
  # 围墙范围: X=[-1.9, 10.2], Y=[-11.3, -0.1]
  start_regions:
  - [-1.2, -10.6, 9.5, -6.0]   # 下半部分
  - [-1.2, -5.5, 9.5, -0.7]    # 上半部分
  - [3.2, -9.0, 9.5, -0.7]     # 右下区域
  goal_regions:
  - [-1.2, -10.6, 9.5, -6.0]
  - [-1.2, -5.5, 9.5, -0.7]
  - [3.2, -9.0, 9.5, -0.7]
```

留出了 0.5-0.8m 的安全边距，避免机器人卡在墙里。

## 新增工具

### 1. 测试脚本: `test_spawn_bounds.sh`

验证修复后的spawn区域是否正确限制在围墙内。

**使用方法**:
```bash
./test_spawn_bounds.sh
```

**功能**:
- 分别测试 Map1 和 Map2
- 生成3个随机机器人和目标
- 检查所有位置是否在围墙内
- 输出详细的坐标和边界检查结果

**示例输出**:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📍 测试 Map1 (围墙范围: X=[-2.3, 3.5], Y=[-11, -0.1])
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 机器人位置 (应在围墙内: X∈[-2.3, 3.5], Y∈[-11, -0.1]):
   Robot 0: x= -1.23, y= -8.45  ✓
   Robot 1: x=  2.01, y= -5.67  ✓
   Robot 2: x=  0.88, y= -9.12  ✓

🎯 目标位置 (应在围墙内: X∈[-2.3, 3.5], Y∈[-11, -0.1]):
   Goal  0: x=  2.34, y= -2.89  ✓
   Goal  1: x= -0.67, y= -9.88  ✓
   Goal  2: x=  1.45, y= -4.56  ✓
```

### 2. 简化版训练脚本: `train_stage.sh`

由于自动课程学习较复杂，创建了手动分阶段训练脚本。

**使用方法**:
```bash
# 阶段1: 单机器人 - map1 开放空间
./train_stage.sh 1 500

# 阶段2: 双机器人 - map1 开放空间  
./train_stage.sh 2 500

# 阶段3: 三机器人 - 走廊对换
./train_stage.sh 3 500

# 阶段4: 四机器人 - 十字路口
./train_stage.sh 4 500

# 阶段5: 四机器人 - 仓库走廊
./train_stage.sh 5 500
```

**参数**:
- 第一个参数: 阶段编号 (1-5)
- 第二个参数: Episodes数量 (默认500)

**阶段配置**:
| 阶段 | 地图 | 机器人数 | 描述 |
|------|------|----------|------|
| 1 | map1 | 1 | 单机器人 - 开放空间基础导航 |
| 2 | map1 | 2 | 双机器人 - 学习会车避让 |
| 3 | corridor_swap | 3 | 三机器人 - 窄通道协调 |
| 4 | intersection | 4 | 四机器人 - 十字路口复杂交互 |
| 5 | warehouse_aisles | 4 | 四机器人 - 仓库环境实战 |

**特性**:
- ✅ 自动检测上一阶段模型
- ✅ 详细的训练日志
- ✅ 清晰的进度提示
- ✅ 支持 Ctrl+C 中断
- ✅ 自动清理 Gazebo 进程

## 验证步骤

### 1. 重新编译
```bash
colcon build --packages-select start_reinforcement_learning --symlink-install
source install/setup.bash
```

### 2. 测试Spawn区域
```bash
./test_spawn_bounds.sh
```

预期结果: 所有机器人和目标位置都显示 ✓

### 3. 运行训练
```bash
# 快速测试 (10 episodes)
./train_stage.sh 1 10
```

预期结果: 
- Gazebo 启动成功
- 机器人在围墙内生成
- 训练正常运行
- 日志保存到 `curriculum_logs/stage1_*.log`

## 文件清单

修改的文件:
- ✏️ [start_curriculum_training.sh](start_curriculum_training.sh) - 移除了不支持的参数
- ✏️ [spawn_presets.yaml](src/start_reinforcement_learning/config/spawn_presets.yaml) - 修正了map1和map2的区域

新增的文件:
- ✨ [test_spawn_bounds.sh](test_spawn_bounds.sh) - Spawn区域测试脚本
- ✨ [train_stage.sh](train_stage.sh) - 简化版分阶段训练脚本
- 📄 [FIX_SUMMARY.md](FIX_SUMMARY.md) - 本文档

## 后续优化建议

### 短期 (立即可做)
1. **测试所有地图**: 运行 `test_spawn_bounds.sh` 验证修复
2. **单阶段训练**: 使用 `train_stage.sh 1 500` 训练第一阶段
3. **检查碰撞率**: 确认机器人不再卡在墙里

### 中期 (1-2周)
1. **课程学习集成**: 修改 `matd3_main.py` 支持自动阶段切换
2. **模型迁移**: 实现阶段间模型参数迁移 (transfer learning)
3. **性能追踪**: 记录每个阶段的成功率、碰撞率等指标

### 长期 (未来工作)
1. **自适应阈值**: 根据训练表现自动调整晋级阈值
2. **动态难度**: 实时调整环境复杂度 (障碍物密度、机器人速度等)
3. **回退机制**: 性能下降时自动回退到上一阶段

## 常见问题

**Q: 为什么移除了 `--curriculum-learning` 参数？**

A: 因为 `matd3_main.py` 从未实现该参数的处理逻辑。未来可以添加，但当前使用 launch 参数更稳定。

**Q: 如何继续使用上一阶段的模型？**

A: 在运行下一阶段前，复制模型文件:
```bash
cp -r models/curriculum/stage1/* models/  # 使用stage1的模型继续训练stage2
./train_stage.sh 2 500
```

**Q: 训练中断了怎么办？**

A: 脚本会自动保存日志和模型checkpoint。再次运行相同命令会从最新checkpoint继续训练。

**Q: 如何可视化训练进度？**

A: 使用分析脚本:
```bash
python3 analyze_curriculum.py
```
会生成学习曲线图表 (PNG) 和文本报告。

## 相关文档

- [课程学习详细设计](CURRICULUM_LEARNING.md)
- [快速开始指南](CURRICULUM_QUICKSTART.md)
- [系统架构说明](SYSTEM_ARCHITECTURE.md)
- [文件导航](FILES_OVERVIEW.md)
