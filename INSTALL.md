# 详细安装指南

完整的环境配置步骤，适用于从零开始在新服务器上部署DGTA项目。

## 系统要求

- Ubuntu 22.04 LTS
- 16GB+ RAM (推荐32GB用于多智能体训练)
- 20GB+ 磁盘空间

## 方法一：使用自动化脚本（推荐）

### 步骤1: 基础环境安装

```bash
# 下载并运行安装脚本
wget https://raw.githubusercontent.com/Bluestone-work/TII/main/install_environment.sh
chmod +x install_environment.sh
./install_environment.sh

# 重新加载环境
source ~/.bashrc
```

### 步骤2: Python依赖安装

```bash
# 创建并激活conda环境
conda create -n dgta python=3.10 -y
conda activate dgta

# 使用environment.yml
conda env create -f environment.yml

# 或使用requirements.txt
pip install -r requirements.txt
```

### 步骤3: 编译项目

```bash
cd ~/TII
colcon build --symlink-install
source install/setup.bash
```

## 方法二：手动安装

### 1. 安装ROS 2 Humble

```bash
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install ros-humble-desktop ros-humble-gazebo-ros-pkgs ros-humble-navigation2 ros-dev-tools
```

### 2. 安装Python依赖

```bash
conda create -n dgta python=3.10 -y
conda activate dgta

pip install numpy==1.24.3 scipy matplotlib pandas seaborn
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install gymnasium==0.29.1 stable-baselines3==2.1.0 tensorboard
pip install torch-geometric torch-scatter torch-sparse
pip install empy==3.3.4 colcon-common-extensions
pip install pyyaml opencv-python pillow
```

### 3. 配置环境变量

在 `~/.bashrc` 添加：

```bash
source /opt/ros/humble/setup.bash
export GAZEBO_MODEL_PATH=/usr/share/gazebo-11/models:$GAZEBO_MODEL_PATH
```

## 远程服务器配置

### 无显示器环境

```bash
sudo apt install xvfb
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 &
```

### 后台运行

```bash
# 使用tmux
tmux new -s training
./start_gnn_mappo_training.sh
# Ctrl+B, D 分离

# 使用nohup
nohup ./start_gnn_mappo_training.sh > training.log 2>&1 &
```

## 常见问题

### 1. 编译错误: Cannot find empy

```bash
pip install empy==3.3.4
```

### 2. ROS 2命令找不到

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
```

### 3. Gazebo启动失败

```bash
killall -9 gzserver gzclient
```

### 4. torch-scatter安装失败

```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### 5. GPU版本PyTorch

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## 验证安装

```bash
# 检查ROS包
ros2 pkg list | grep gnn_marl_training

# 检查Python包
python -c "import torch; print(torch.__version__)"
python -c "import gymnasium; print(gymnasium.__version__)"
python -c "import stable_baselines3; print('SB3 OK')"
```

## TensorBoard远程访问

```bash
# 在本地执行端口转发
ssh -L 6006:localhost:6006 user@server

# 在服务器启动TensorBoard
tensorboard --logdir=./log --port=6006

# 本地浏览器访问
# http://localhost:6006
```

---

**维护者**: wj  
**更新日期**: 2026-07-03
