# SB3 RecurrentPPO 训练包

使用Stable-Baselines3的RecurrentPPO算法训练机器人导航，作为MATD3的对比实验。

## 特点

- **独立训练**: 每个机器人有自己的RecurrentPPO策略
- **简单架构**: 将其他机器人视为动态障碍物，不需要复杂的多智能体通信
- **LSTM记忆**: RecurrentPPO使用LSTM处理部分可观测性
- **成熟库**: 基于Stable-Baselines3，稳定可靠

## 安装依赖

```bash
pip install stable-baselines3 sb3-contrib gymnasium
```

## 快速开始

### 1. 启动Gazebo仿真环境

```bash
cd ~/work/multi-robot-exploration-rl
source install/setup.bash
ros2 launch start_rl_environment main.launch.py map_number:=3 robot_number:=1
```

### 2. 开始训练

```bash
# 在新终端
cd ~/work/multi-robot-exploration-rl
source install/setup.bash

# 使用launch文件
ros2 launch sb3_training train_ppo.launch.py map_number:=3 robot_number:=1 total_timesteps:=1000000

# 或者直接运行
ros2 run sb3_training train_ppo --robot_number 1 --map_number 3 --total_timesteps 1000000
```

## 训练参数

### 环境参数
- `--robot_number`: 机器人数量 (默认: 1)
- `--map_number`: 地图编号 (默认: 3)
- `--random_mode`: 使用随机起始位置
- `--max_steps`: 每个episode最大步数 (默认: 300)

### PPO参数
- `--total_timesteps`: 总训练步数 (默认: 1,000,000)
- `--learning_rate`: 学习率 (默认: 3e-4)
- `--n_steps`: 每次更新的步数 (默认: 2048)
- `--batch_size`: 批次大小 (默认: 64)
- `--n_epochs`: 优化轮数 (默认: 10)
- `--gamma`: 折扣因子 (默认: 0.99)
- `--clip_range`: PPO裁剪范围 (默认: 0.2)
- `--ent_coef`: 熵系数，控制探索 (默认: 0.01)

### 示例

```bash
# 单机器人训练，100万步
ros2 run sb3_training train_ppo \
    --robot_number 1 \
    --map_number 3 \
    --total_timesteps 1000000 \
    --learning_rate 3e-4 \
    --device cuda

# 多机器人训练（每个机器人独立策略）
ros2 run sb3_training train_ppo \
    --robot_number 2 \
    --map_number 3 \
    --total_timesteps 2000000

# 继续训练已有模型
ros2 run sb3_training train_ppo \
    --robot_number 1 \
    --map_number 3 \
    --load_path /path/to/model.zip \
    --total_timesteps 500000
```

## 监控训练

### TensorBoard

```bash
tensorboard --logdir ~/work/multi-robot-exploration-rl/sb3_logs
```

在浏览器打开 http://localhost:6006 查看训练曲线

### 实时日志

训练过程会输出：
- Episode奖励
- Episode长度
- 策略损失
- 价值函数损失
- 熵值

## 模型保存

- **自动保存**: 每10,000步保存一次检查点
- **最终模型**: 训练结束保存到 `sb3_models/*/final_model.zip`
- **中断保存**: Ctrl+C中断时自动保存当前模型

保存路径格式：
```
sb3_models/ppo_map3_robots1_20260125_153045/
├── ppo_model_10000_steps.zip
├── ppo_model_20000_steps.zip
├── ...
└── final_model.zip
```

## 与MATD3对比

| 特性 | RecurrentPPO | MATD3 |
|------|--------------|-------|
| **算法类型** | On-policy | Off-policy |
| **多智能体** | 独立训练 | 中心化训练 |
| **记忆** | LSTM | 无 |
| **探索** | 策略熵 | 噪声探索 |
| **样本效率** | 低 | 高 |
| **训练稳定性** | 高 | 中 |
| **实现复杂度** | 简单 | 复杂 |

## 故障排除

### 1. 导入错误
```bash
pip install stable-baselines3 sb3-contrib gymnasium
```

### 2. CUDA内存不足
```bash
# 使用CPU
--device cpu

# 或减小batch_size
--batch_size 32
```

### 3. 训练不稳定
```bash
# 降低学习率
--learning_rate 1e-4

# 增加熵系数（更多探索）
--ent_coef 0.02
```

## 性能调优建议

1. **学习率**: 从3e-4开始，如果不稳定降到1e-4
2. **熵系数**: 初期0.01，后期可降到0.001
3. **批次大小**: GPU内存允许的情况下越大越好
4. **GAE lambda**: 0.95是个好的默认值
5. **裁剪范围**: 0.2是标准值，不建议修改

## 下一步

1. **评估模型**: 查看trained模型在Gazebo中的表现
2. **对比MATD3**: 比较收敛速度、最终性能、稳定性
3. **调参优化**: 根据初步结果调整超参数
4. **扩展到多机器人**: 测试多个独立策略的协作效果
