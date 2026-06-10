# 🔍 数据同步验证工具

确保强化学习的观测、奖励与模拟器实时场景完美对应！

## 🚀 快速开始（30秒）

```bash
cd ~/work/multi-robot-exploration-rl

# 运行快速诊断
./scripts/run_sync_validation.sh 3

# 选择: 1 (TF树监控 - 单次检查)
```

**看到所有✅ = 一切正常！** 

**看到❌ = 需要修复，继续阅读...**

---

## 📚 文档

- **[快速入门](QUICK_START_VALIDATION.md)** ⚡ - 2分钟诊断TF，5分钟完整验证
- **[完整指南](DATA_SYNC_VALIDATION_GUIDE.md)** 📖 - 详细使用说明和问题排查
- **[功能总结](SYNC_VALIDATION_SUMMARY.md)** 📋 - 所有新增功能一览

---

## 🛠️ 工具箱

### 1. TF树监控器
**功能**: 诊断TF变换问题
```bash
python3 src/start_reinforcement_learning/start_reinforcement_learning/env_logic/tf_monitor.py 3
```

### 2. 数据同步可视化器
**功能**: RViz中实时显示数据新鲜度
```bash
python3 src/start_reinforcement_learning/start_reinforcement_learning/env_logic/sync_visualizer.py 0 &
rviz2
```

### 3. 自动验证系统
**功能**: 训练时自动检查（已集成到logic.py）
- 每50步打印详细报告
- 实时监控数据新鲜度
- 自动诊断TF问题

---

## 🎯 解决的问题

### 问题1: TF变换失败
```
[INFO] Waiting for transform map -> odom: Invalid frame ID "odom"
```

**解决**: 
1. 运行 `tf_monitor.py` 查看实际frame名
2. 修改logic.py中的frame命名
3. 验证修复结果

### 问题2: 不确定数据同步
**担心**: "观测和奖励是否对应实时场景？"

**解决**: 
- 自动时间戳监控
- 每50步同步报告
- RViz可视化验证

---

## ✅ 验证清单

训练前检查：
- [ ] TF监控器无❌
- [ ] RViz圆环全绿🟢
- [ ] 同步报告无警告
- [ ] 数据年龄<200ms

**全部✅ = 可以开始训练！** 🎉

---

## 📊 示例输出

### 正常情况
```
📊 数据同步报告 (Episode 1, Step 50)
📍 里程计数据:
  Robot0: ✅ 年龄=12.3ms, 位置=(1.23, 4.56)
📡 激光雷达数据:
  Robot0: ✅ 年龄=5.2ms, 最近障碍=0.85m
```

### 异常情况
```
📍 里程计数据:
  Robot0: ❌ 年龄=350.0ms  ← 数据过时！
⚠️ 数据同步问题: Robot0 odom陈旧: 350.0ms
```

---

## 🔧 配置

### 调整验证频率
```python
# 在 logic.py 或训练脚本中
env.sync_validation_interval = 100  # 每100步报告（默认50）
env.max_data_age_ms = 150  # 更严格的阈值（默认200）
```

### 关闭验证（生产环境）
```python
env.enable_sync_validation = False
env.debug_obs_warnings = False
```

---

## 🎨 RViz颜色说明

机器人上方的圆环：
- 🟢 **绿色**: 数据新鲜（< 100ms）✅
- 🟡 **黄色**: 数据稍旧（100-300ms）⚠️
- 🔴 **红色**: 数据过时（> 300ms）❌

激光雷达点：
- 🔴 红点: 障碍很近（< 0.5m）
- 🟡 黄点: 障碍较近（0.5-1.0m）
- 🟢 绿点: 障碍安全（> 1.0m）

---

## 🐛 常见问题

### Q: TF监控器显示 "odom未找到"
**A**: Frame名称不匹配，运行 `ros2 run tf2_ros tf2_echo map my_bot0/odom` 查看实际名称

### Q: 数据年龄总是很大
**A**: 检查Gazebo是否暂停，或仿真时间是否启动

### Q: RViz不显示marker
**A**: 确认Fixed Frame设为`map`，并添加了MarkerArray话题

---

## 📞 需要帮助？

1. 查看 [完整指南](DATA_SYNC_VALIDATION_GUIDE.md)
2. 运行诊断工具
3. 检查终端输出和日志

---

## 🎉 总结

有了这套工具，你可以：
- ✅ 快速诊断TF问题
- ✅ 实时监控数据同步
- ✅ 可视化验证对应关系
- ✅ 确信训练数据准确

**现在可以放心训练强化学习模型了！** 🚀
