# 障碍物随机初始化实现总结

## 修改的文件

### 1. gnn_marl_env.py (核心环境文件)

#### 构造函数修改 (Line ~1265)
```python
# 添加了两个新参数
num_static_obstacles=8,      # 静态障碍物数量
random_obstacles=True,       # 是否启用随机初始化
```

#### 成员变量添加 (Line ~1577)
```python
self.num_static_obstacles = max(0, min(int(num_static_obstacles), 8))
self.random_obstacles = bool(random_obstacles)
self.static_obstacle_names: list = [f'static_obs_{i}' for i in range(8)]
```

#### 新增方法: _spawn_random_obstacles() (Line ~3800)
- 功能：每次reset时随机spawn静态和动态障碍物
- 碰撞检测：确保障碍物之间、障碍物与机器人之间不重叠
- 支持地图：Map 8 (circle_swap_arena), Map 9 (warehouse_dynamic)
- 约60行代码

#### reset()方法修改 (Line ~3002)
```python
# 在机器人spawn之前先spawn障碍物（只有robot_id==0时执行）
if self.robot_id == 0:
    self._spawn_random_obstacles()
```

#### 配置字典更新 (Line ~168)
```python
'num_static_obstacles': config.get('num_static_obstacles', 8),
'random_obstacles': bool(config.get('random_obstacles', True)),
```

### 2. train_gnn_mappo_full.py (训练脚本)

#### 命令行参数添加 (Line ~1283)
```python
parser.add_argument("--num_static_obstacles", type=int, default=None,
                    help="可选：静态障碍物数量（默认8）")
parser.add_argument("--random_obstacles", action="store_true", default=False,
                    help="是否每次reset时随机初始化障碍物位置")
```

#### 参数处理 (Line ~1346)
```python
if args.num_static_obstacles is not None:
    stage_cfg["num_static_obstacles"] = int(args.num_static_obstacles)
else:
    stage_cfg["num_static_obstacles"] = 8
if args.random_obstacles:
    stage_cfg["random_obstacles"] = True
else:
    stage_cfg["random_obstacles"] = False
```

#### 环境配置传递 (Line ~1578, 两处)
```python
"num_static_obstacles": stage_cfg.get('num_static_obstacles', 8),
"random_obstacles": stage_cfg.get('random_obstacles', False),
```

### 3. run_curriculum.sh (课程学习脚本)

#### 变量定义 (Line ~67)
```bash
NUM_STATIC_OBSTACLES_OVERRIDE=""
RANDOM_OBSTACLES=0  # 0=关闭, 1=开启
```

#### 命令行解析 (Line ~318)
```bash
--num_static_obstacles) NUM_STATIC_OBSTACLES_OVERRIDE="$2"; shift 2 ;;
--random_obstacles) RANDOM_OBSTACLES=1; shift 1 ;;
```

#### 训练命令构建 (Line ~760)
```bash
if [[ -n "$NUM_STATIC_OBSTACLES_OVERRIDE" ]] && script_supports_arg "--num_static_obstacles"; then
    cmd+=(--num_static_obstacles "$NUM_STATIC_OBSTACLES_OVERRIDE")
fi
if (( RANDOM_OBSTACLES == 1 )) && script_supports_arg "--random_obstacles"; then
    cmd+=(--random_obstacles)
fi
```

## 核心实现逻辑

### 碰撞检测算法

```python
def is_position_valid(x, y, radius, min_sep):
    """检查位置是否有效（不与已有障碍物/机器人重叠）"""
    for ox, oy, osep in occupied_positions:
        dist = math.sqrt((x - ox)**2 + (y - oy)**2)
        if dist < (radius + osep):
            return False
    return True
```

### spawn流程

1. 确定spawn区域（根据地图编号）
2. 记录已占用位置（机器人当前位置）
3. 逐个spawn静态障碍物
   - 随机生成位置
   - 碰撞检测（最多尝试100次）
   - 调用 SetEntityState 服务
   - 记录到占用列表
4. 逐个spawn动态障碍物（同上）
5. 将未使用的障碍物移到地图外

### 安全距离参数

```python
STATIC_OBS_RADIUS = 0.20     # 静态障碍物半径
DYNAMIC_OBS_RADIUS = 0.22    # 动态障碍物半径
MIN_OBSTACLE_SEP = 0.6       # 障碍物之间最小间隔
MIN_ROBOT_SEP = 1.0          # 障碍物与机器人最小间隔
```

## 使用方法

### 完整示例

```bash
./run_curriculum.sh \
  --run_suffix "random_obs_test" \
  --start_stage 1 \
  --end_stage 4 \
  --num_obstacles 4 \
  --num_static_obstacles 5 \
  --random_obstacles \
  --gat_critic_mode gat \
  --graph_ablation dual_graph
```

### 关键点

1. 必须同时指定 `--random_obstacles` 才会启用随机spawn
2. 不指定数量时使用默认值（动态8个，静态8个）
3. 数量范围：0-8（超出会被clamp）
4. 仅在 Map 8 和 Map 9 上有效

## 验证清单

- [x] 环境代码修改完成
- [x] 训练脚本参数添加
- [x] 课程学习脚本更新
- [x] 碰撞检测逻辑实现
- [x] 支持地图配置
- [x] 文档编写完成

## 测试建议

1. **基础功能测试**
```bash
# 少量障碍物
./run_curriculum.sh --start_stage 1 --end_stage 1 \
  --num_obstacles 2 --num_static_obstacles 2 --random_obstacles
```

2. **极限测试**
```bash
# 最大数量障碍物
./run_curriculum.sh --start_stage 1 --end_stage 1 \
  --num_obstacles 8 --num_static_obstacles 8 --random_obstacles
```

3. **对比测试**
```bash
# 关闭随机spawn（使用固定位置）
./run_curriculum.sh --start_stage 1 --end_stage 1 \
  --num_obstacles 4 --num_static_obstacles 4
# （不加--random_obstacles）
```

## 已知限制

1. 仅支持 Map 8 和 Map 9
2. 最多各8个障碍物（world文件限制）
3. spawn失败100次后会跳过该障碍物
4. 多智能体环境下只有 robot_id==0 执行spawn

## 未来改进方向

1. 支持更多地图
2. 动态调整spawn区域大小
3. 增加障碍物类型（方形、椭圆等）
4. 支持障碍物大小随机化
5. 添加可视化spawn区域的工具
