# Multi-Robot Exploration with Reinforcement Learning (DGTA)

基于图神经网络和多智能体强化学习的机器人导航训练系统。该项目实现了DGTA (Decentralized Graph-based multi-roboT plAnner)算法，支持多机器人协同导航和避障。

## 项目简介

本项目是一个完整的多智能体强化学习训练框架，集成了：
- **强化学习算法**: MAPPO (Multi-Agent Proximal Policy Optimization)
- **图神经网络**: 用于处理机器人间的通信和协作
- **仿真环境**: 基于ROS 2 Humble + Gazebo
- **多种训练场景**: 支持走廊、仓库、交叉路口等多种地图

## 系统要求

### 操作系统
- Ubuntu 22.04 LTS (推荐)

### 硬件要求
- CPU: 多核处理器 (推荐8核以上)
- 内存: 16GB+ RAM (多智能体训练建议32GB+)
- GPU: NVIDIA GPU (可选，用于加速神经网络训练)
- 存储: 20GB+ 可用空间

## 快速开始

### 1. 克隆项目

```bash
git clone git@github.com:Bluestone-work/TII.git
cd TII
```

### 2. 环境配置

详细安装步骤请参考 [INSTALL.md](INSTALL.md)

**快速安装（已有ROS 2 Humble和Conda）:**

```bash
# 创建conda环境
conda env create -f environment.yml
conda activate dgta

# 或使用pip安装
pip install -r requirements.txt

# 编译项目
colcon build --symlink-install
source install/setup.bash
```

### 3. 启动训练

```bash
# 2智能体训练
./start_gnn_mappo_training.sh

# 6智能体训练
NUM_AGENTS=6 ./start_gnn_mappo_training.sh
```

### 4. 监控训练

```bash
# TensorBoard
tensorboard --logdir=./log --port=6006
# 浏览器访问 http://localhost:6006
```

## 文档

- [INSTALL.md](INSTALL.md) - 详细安装指南（含自动化脚本）
- [environment.yml](environment.yml) - Conda环境配置
- [requirements.txt](requirements.txt) - Python依赖列表

## 项目结构

```
TII/
├── src/                              # ROS 2源代码包
│   ├── start_reinforcement_learning/ # 强化学习环境
│   ├── gnn_marl_training/            # GNN-MAPPO训练
│   ├── gnn_marl_training_DGTA_nobuffer/ # DGTA无缓冲版本
│   └── ...
├── README.md                         # 本文件
├── INSTALL.md                        # 详细安装指南
├── environment.yml                   # Conda环境配置
└── requirements.txt                  # Python依赖
```

## 常见问题

### 编译错误

```bash
# 缺少empy
pip install empy==3.3.4

# 确保ROS环境已加载
source /opt/ros/humble/setup.bash
```

### Gazebo问题

```bash
# 清理僵尸进程
killall -9 gzserver gzclient
```

### 无显示器环境

```bash
# 安装虚拟显示
sudo apt install xvfb
xvfb-run -a ./start_gnn_mappo_training.sh
```

更多问题请参考 [INSTALL.md](INSTALL.md)

## 引用

如果您在研究中使用了本项目，请引用相关论文。

## 许可证

MIT License

---

**最后更新**: 2026-07-03
