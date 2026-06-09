# 项目文件总览

## 📁 核心文件

### 训练相关
- `start_with_distance_field.sh` - 标准训练脚本（单一环境）
- `start_curriculum_training.sh` - **课程学习训练脚本**（推荐）
- `curriculum_manager.py` - 课程学习管理器
- `curriculum_config.yaml` - 课程配置文件

### 分析工具
- `view_interaction_logs.py` - 交互日志分析
- `analyze_curriculum.py` - 课程学习分析可视化
- `test_distance_field.py` - 距离场测试工具

### 工具脚本
- `kill_all_ros.sh` - 清理ROS进程
- `kill_gazebo.sh` - 清理Gazebo进程

## 📚 文档

### 入门文档
- `README.md` - 项目主文档
- `USAGE.md` - 快速使用指南
- `CURRICULUM_QUICKSTART.md` - **课程学习快速开始**（推荐阅读）

### 技术文档
- `SYSTEM_ARCHITECTURE.md` - 系统架构说明（顶部视觉系统）
- `CURRICULUM_LEARNING.md` - 课程学习详细设计
- `DISTANCE_FIELD_FIX.md` - 距离场问题修复记录

### 参考文档
- `相关工作对比_2024-2025.md` - 相关研究对比

## 🗂️ 目录结构

```
multi-robot-exploration-rl/
├── src/                          # 源代码
│   ├── start_rl_environment/     # 环境包（Gazebo世界、地图）
│   └── start_reinforcement_learning/  # 训练包（RL算法）
├── build/                        # 编译输出
├── install/                      # 安装文件
├── log/                          # 编译日志
├── interaction_logs/             # 训练交互日志
├── curriculum_logs/              # 课程学习日志
├── models/                       # 保存的模型
│   └── curriculum/               # 课程学习模型
└── images/                       # 图片资源
```

## 🎯 推荐工作流

### 新手入门
1. 阅读 `CURRICULUM_QUICKSTART.md`
2. 运行 `./start_curriculum_training.sh`
3. 使用 `python3 analyze_curriculum.py` 查看结果

### 研究使用
1. 阅读 `SYSTEM_ARCHITECTURE.md` 了解系统设计
2. 阅读 `CURRICULUM_LEARNING.md` 了解训练策略
3. 根据需求修改 `curriculum_config.yaml`
4. 运行训练并记录实验数据

### 调试问题
1. 查看 `DISTANCE_FIELD_FIX.md` 了解常见问题
2. 使用 `test_distance_field.py` 验证距离场
3. 使用 `view_interaction_logs.py` 分析训练日志

## 🔍 文件依赖关系

```
curriculum_config.yaml
    ↓
curriculum_manager.py
    ↓
start_curriculum_training.sh
    ↓
[训练过程]
    ↓
curriculum_logs/*.jsonl
    ↓
analyze_curriculum.py
    ↓
curriculum_analysis.png
stage_comparison.png
curriculum_report.txt
```

## 📊 日志文件说明

### 训练日志
- `curriculum_logs/training_*.log` - 终端输出日志
- `curriculum_logs/curriculum_log_*.jsonl` - 结构化训练数据
- `interaction_logs/interaction_log_*.jsonl` - 详细交互记录

### 分析输出
- `curriculum_analysis.png` - 学习曲线图
- `stage_comparison.png` - 阶段对比图
- `curriculum_report.txt` - 文本报告

## 🚀 快速命令参考

```bash
# 课程学习训练
./start_curriculum_training.sh

# 分析训练结果
python3 analyze_curriculum.py

# 查看最新日志
tail -f curriculum_logs/training_*.log

# 清理环境
./kill_all_ros.sh
./kill_gazebo.sh

# 重新编译
colcon build --packages-select start_reinforcement_learning
source install/setup.bash
```

---

**创建日期**: 2026年1月15日
