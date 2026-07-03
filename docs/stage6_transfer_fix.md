# Stage6 环境迁移改进方案(circle_swap → intersection 质变场景)

> **问题**: Stage1-5 全在 circle_swap(map8),Stage6 突然切到 intersection(map4),
> 从 Stage5 续训(熵已塌缩到0.26)后,全程负回报 -161→-80,迁移完全失败。

---

## 一、问题诊断

### 1.1 Stage5→6 是质变,不是量变

| Stage | 场景 | 车数 | 几何结构 |
|---|---|---|---|
| 1-5 | circle_swap (map8) | 2/4/6/8/4 | 圆形对角换位,中心区域是通过瓶颈 |
| 6 | **intersection (map4)** | 4 | **十字路口,四向交汇,无中心圆** |

**Stage1-5 共 5 个阶段全在同一地图**,策略已**过拟合 circle_swap 的几何特征**(中心避让、
沿圆弧绕行)。Stage6 突然换地图,这些特征全失效 → 迁移失败。

### 1.2 Stage5 策略状态:熵塌缩 + 僵化

```
Stage5 末期(iter28-40): 回报 177~188, 熵 0.0005~0.70 (塌缩!)
Stage6 开局(iter1):     回报 -161,    熵 3.42 (重新reset?)
```

Stage5 策略已完全确定性化(熵触底0.0005),几乎无探索能力。即便 Stage6 熵重新到3.4,
**网络权重仍携带着 circle_swap 的过拟合知识**,在 intersection 里不适用 → 负迁移。

### 1.3 Stage6 训练配置:保守到无法适应

```python
# Stage6 续训时的超参(从 Stage5 继承)
lr: 1e-4 (已衰减到极低)
clip_param: 0.10 (极保守,防"破坏已学策略")
entropy_coeff: 0.0005 (极低,几乎不鼓励探索)
```

这套配置是为**保护 Stage5 的好策略**设计的,但在新环境里反而**锁死了适应能力**——
策略改不动,只能用 Stage5 的过拟合知识硬碰 intersection,当然失败。

---

## 二、改进方案(三层,按成本递增)

### 方案 A:立即可做(低成本,针对 Stage6 本身)

**核心思路**:Stage6 是新环境,应该**给策略更新自由度**,而不是"保护已学策略"。

#### A1. Stage6 起始时**提高 LR 和 Clip**(临时放宽更新约束)

```python
# 在 run_curriculum.sh 或 train_gnn_mappo_full.py 里,Stage6 专用配置
if stage == 6:
    lr = 3e-4           # 回到初始 LR(不继承 Stage5 的 1e-4)
    clip_param = 0.20   # 回到标准 Clip(不用 Stage5 的 0.10)
    entropy_coeff 初始 = 0.002  # 略高于默认,鼓励探索新环境
```

**实现位置**:`train_gnn_mappo_full.py` 在加载 Stage5 checkpoint 后、创建 trainer 前,
根据 `args.env_stage` 判断,若是 6 则覆盖 lr/clip_param。

#### A2. Stage6 前 10-20 iter **不衰减 LR**(给足适应窗口)

```python
# 延迟 LR schedule 起点
lr_schedule = [
    [0,          3e-4],
    [100_000,    3e-4],  # 前100k步(~20 iter)保持高LR
    [300_000,    1e-4],
    [train_steps, 1e-5],
]
```

让策略在新环境有足够时间"忘掉" circle_swap 的过拟合,学习 intersection 特征。

#### A3. **不从 Stage5 续训,从 Stage3(6车) 续训**

Stage5 熵已塌缩(0.26),策略僵化。**Stage3(6车)末期熵还在 1.84,保留探索性**,
且 6 车的协调能力足够迁移到 4 车 intersection。

```bash
# 修改 run_curriculum.sh 或手动指定
--init_checkpoint ray_results/GNN_MAPPO_Stage2_Cont_EnvStage3/best
```

---

### 方案 B:中期结构性改进(中成本,治本)

#### B1. Stage5.5 **混合场景过渡阶段**(新增一个 stage)

在 Stage5(circle_swap 4车)和 Stage6(intersection 4车)之间插入一个混合训练阶段:

```python
5.5: {
    "name": "Stage 3.75 · 混合场景(circle_swap + intersection)",
    "map_number": [8, 4],  # 每个 episode 随机选一个
    "sample_ratio": [0.5, 0.5],  # 各占 50%
    "max_episode_steps": 900,
    "description": "交替训练 circle_swap 和 intersection,平滑过渡",
}
```

**实现**: 在 `GNNMARLEnv.reset` 里根据 `map_number` 是列表时随机选。训练 100-200k 步
后再进入纯 Stage6,策略已部分适应 intersection。

#### B2. **奖励函数加场景无关项**(减少对地图几何的依赖)

当前奖励严重依赖 A* 引导(path_progress、subgoal),这些在换地图时失效。增加**场景通用奖励**:

- 接近目标的直线距离变化(不依赖 A*)
- 前方空隙宽度奖励(鼓励找通道,不管什么地图)
- 相对速度避让(动态避碰,不依赖静态地图)

#### B3. **观测加地图拓扑特征编码**(显式告知"换地图了")

当前观测没有"我在哪种地图"的显式信号,策略不知道自己进了新环境。可以加:
- one-hot map_id (8 维,circle_swap=1, intersection=0, ...)
- 或用简单的拓扑特征(连通度、障碍物密度)让策略感知环境类型

---

### 方案 C:根本性方案(高成本,架构级)

#### C1. **Domain Randomization**(从 Stage1 就混合多地图)

别再搞单地图 curriculum,从头就在多个地图随机训练:
- Stage1: 2车 × [circle_swap, intersection, hallway] 随机
- Stage2: 4车 × 同上
- ...

强制策略学**场景无关的避碰和导航**,而非记忆某个地图。代价:训练步数要翻倍+。

#### C2. **分层策略(高层规划 + 低层控制)**

- 高层:地图无关的全局规划(Dijkstra / learned value map)
- 低层:RL 策略只管局部避碰和跟随 waypoint

换地图时只需重新规划,低层避碰策略可复用。代价:架构复杂度++。

---

## 三、推荐执行顺序

| 优先级 | 方案 | 成本 | 预期效果 |
|---|---|---|---|
| 🔴 **立即做** | **A3: 从 Stage3 续训 + A1: 提高 LR/Clip** | 极低 | Stage6 有望到正回报 |
| 🟠 中期 | B1: 插入混合场景 Stage5.5 | 中 | 平滑过渡,后续 curriculum 鲁棒 |
| 🟡 长期 | C1: Domain Randomization | 高 | 根本解决迁移问题 |

**我的建议**:先做 A3+A1 验证 Stage6 能否跑通。能 → 说明续训源头和超参是瓶颈,不用动架构。
不能 → 再上 B1(混合场景)。C1/C2 留给论文下一版或真要部署时再考虑。

---

## 四、具体代码改动(方案 A3+A1)

### 改动 1:`run_curriculum.sh` Stage6 条件分支

```bash
# 在 stage 6 的配置块里加:
if [ "$STAGE" -eq 6 ]; then
    echo "  [Stage6] 环境迁移专用配置: 从 Stage3 续训 + 提高 LR/Clip"
    INIT_CHECKPOINT="ray_results/GNN_MAPPO_Stage2_Cont_EnvStage3/best"
    EXTRA_ARGS="--override_lr 3e-4 --override_clip_param 0.20 --override_entropy_coeff 0.002"
fi
```

### 改动 2:`train_gnn_mappo_full.py` 加 override 参数

```python
# 在 argparse 部分加:
parser.add_argument("--override_lr", type=float, default=None,
                    help="覆盖 lr(用于环境迁移时临时提高学习率)")
parser.add_argument("--override_clip_param", type=float, default=None)
parser.add_argument("--override_entropy_coeff", type=float, default=None)

# 在创建 trainer config 时:
if args.override_lr is not None:
    print(f"⚙️ 环境迁移: lr 覆盖为 {args.override_lr}")
    config["lr"] = args.override_lr
    config["lr_schedule"] = None  # 禁用 schedule,保持恒定
if args.override_clip_param is not None:
    print(f"⚙️ 环境迁移: clip_param 覆盖为 {args.override_clip_param}")
    config["clip_param"] = args.override_clip_param
if args.override_entropy_coeff is not None:
    config["entropy_coeff"] = args.override_entropy_coeff
```

---

## 五、验证指标

改完后重跑 Stage6,看这些指标:
- **回报**: 前 20 iter 能否从 -160 升到 -50 以内?(说明开始适应)
- **r_goal**: 能否从 0 升到 >5?(说明开始到达目标)
- **熵**: 保持在 2.0~3.0(有探索),不要又塌到 0.5(说明没锁死)

如果 20 iter 后回报还在 -100 以下,说明 A 方案不够,要上 B1(混合场景)。

---

- **创建日期**: 2026-06-23
- **关联**: [8agent_improvement_plan.md](8agent_improvement_plan.md) 第八章环境迁移部分
- **状态**: 待实施,优先 A3+A1
