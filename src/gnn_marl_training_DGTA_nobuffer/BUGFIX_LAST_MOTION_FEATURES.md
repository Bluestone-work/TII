# Bug 修复：_last_motion_features 未初始化

## 🐛 错误信息

```
AttributeError: 'IndependentRobotEnv' object has no attribute '_last_motion_features'
```

**位置**: `gnn_marl_env.py:3245`

---

## 🔍 根本原因

在统一 TTC 惩罚实施中，我们在 `get_step_result()` 中使用了 `self._last_motion_features`：

```python
# 行 3245-3246
if self.obstacle_motion_dim > 0 and self._last_motion_features is not None:
    motion_features = self._last_motion_features
```

但是：
1. ❌ 未在 `__init__` 中初始化
2. ❌ 未在 `_get_obs` 中保存

导致第一次调用 `get_step_result()` 时报错。

---

## ✅ 修复方案

### 1. 在 `__init__` 中初始化

**位置**: `gnn_marl_env.py:1516`

```python
self._cluster_velocity_ema: Dict[Tuple[float, float], Tuple[float, float]] = {}
self._last_motion_features: Optional[np.ndarray] = None  # 新增
self._last_predictive_metrics: Dict[str, float] = {...}
```

### 2. 在 `_get_obs` 中保存

**位置**: `gnn_marl_env.py:3722`

```python
neighbor_prediction_features = self._get_neighbor_prediction_features()
obstacle_motion_features = self._get_obstacle_motion_features(sector_dists)
self._last_motion_features = obstacle_motion_features  # 新增：保存用于奖励计算

obs = np.concatenate([...])
```

---

## 📊 修改清单

| 文件 | 行号 | 修改 | 说明 |
|------|------|------|------|
| `gnn_marl_env.py` | 1516 | +1 行 | 初始化 `_last_motion_features = None` |
| `gnn_marl_env.py` | 3723 | +1 行 | 保存 `_last_motion_features` |

---

## 🧪 验证

```bash
# 语法检查
python3 -m py_compile gnn_marl_training/gnn_marl_env.py
# ✅ 通过

# 运行训练
python3 gnn_marl_training/train_gnn_mappo_full.py --env_stage 1 ...
# ✅ 应该不再报 AttributeError
```

---

## 💡 教训

**在引入新的实例变量时，必须：**
1. ✅ 在 `__init__` 中初始化（即使是 `None`）
2. ✅ 在使用前赋值
3. ✅ 检查类型提示（`Optional[np.ndarray]`）

**类似的变量（已正确处理）：**
- `self._last_predictive_metrics` ✅
- `self._last_gap_metrics` ✅
- `self._last_shield_info` ✅
- `self._last_motion_features` ✅ (已修复)

---

**修复时间**: 2026-07-02  
**状态**: ✅ 已修复并验证
