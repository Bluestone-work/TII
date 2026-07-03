# 定量诊断流程 - 使用说明

## 准备工作已完成

已为你准备好完整的定量诊断工具链:

### 1. 静态分析 (无需训练,已运行)
- **`static_reward_analysis.py`** — 从代码逻辑推算奖励-感知失配
- **输出**: 已在终端打印,核心发现记录在 `DIAGNOSIS_REPORT.md`

### 2. 动态采集+可视化 (需要你的训练环境)
- **`collect_episode_diagnostics.py`** — 从运行中的 Gazebo 采集 episode 数据
- **`visualize_diagnostics.py`** — 生成图表和 HTML 报告
- **`run_diagnosis.sh`** — 一键运行上述两步

---

## 等你的训练跑完后,执行这个:

```bash
cd /home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training_DGTA_nobuffer

# 方式 1: 用随机策略采集 (最快,适合验证感知问题)
bash run_diagnosis.sh

# 方式 2: 用你的 checkpoint 采集 (如果想看训练后的策略表现)
bash run_diagnosis.sh /path/to/your/checkpoint

# 执行完会在 diagnosis_output_YYYYMMDD_HHMMSS/ 生成报告
```

**前提**: Gazebo 必须在运行 (你现在正在跑的那个就可以)

---

## 输出产物

执行完后会生成:

```
diagnosis_output_YYYYMMDD_HHMMSS/
├── diagnosis_report.html              ← 📊 主报告 (浏览器打开)
├── diagnosis_data.jsonl               ← 原始数据
├── reward_vs_distance_boxplot.png     ← 奖励-距离箱线图
├── reward_vs_distance_scatter.png     ← 奖励-距离散点图
├── dynamic_token_distance_hist.png    ← 动态识别分布
├── collision_analysis.png             ← 碰撞事件分析
└── reward_vs_velocity.png             ← 速度-奖励关系
```

**核心看 `diagnosis_report.html`**, 里面会有:
- 定量证据: "不动 vs 前进" 的奖励对比
- 碰撞时的 min_dist 分布 (是否和阈值一致)
- 动态识别率 (是否 <30%, 说明速度估计有问题)
- 中距离 (0.5-0.75m) 的奖励均值 (如果是负的,说明 static 太保守)

---

## 预期的关键发现

基于静态分析,我预计你会看到:

1. **⚠️ 中密度场景奖励为负** → 确诊奖励失衡
2. **⚠️ 不动比前进奖励更高** → 确诊策略学"不动"是理性选择
3. **⚠️ 动态识别率 <30%** → 确诊速度估计不稳定
4. **⚠️ 碰撞时 min_dist 中位数 >0.25m** → 感知精度不够

---

## 看完报告后的决策树

```
[查看 HTML 报告]
    │
    ├─ 如果发现 "不动 > 前进" 
    │   → 优先改 【奖励重平衡】 (见 DIAGNOSIS_REPORT.md 第四节方案1)
    │
    ├─ 如果碰撞率高但 min_dist 在合理范围
    │   → 优先改 【扇区距离拼回观测】 (见方案2)
    │
    ├─ 如果动态识别率 <30%
    │   → 优先改 【动态速度平滑】 (见方案3)
    │
    └─ 如果以上都有
        → 先改 1+2 (共3小时), 跑 Stage2 50 iters 验证
```

---

## 时间线

- **现在**: 静态分析完成 ✓
- **你的训练跑完后**: 执行 `bash run_diagnosis.sh` (~15 分钟采集+分析)
- **看完报告**: 告诉我发现了什么,我帮你改对应的代码
- **改完后**: 跑 Stage2 50-100 iters 验证

---

## 如果 run_diagnosis.sh 报错

常见问题:

1. **"gzserver 未运行"** → 确保 Gazebo 启动了
2. **"环境初始化失败"** → 检查 ROS2 环境变量 `source /opt/ros/.../setup.bash`
3. **"缺少依赖"** → `pip install pandas matplotlib seaborn`
4. **采集卡住** → 可能是环境 reset 失败,Ctrl+C 中断,检查 Gazebo 日志

---

## 备用方案 (如果一键脚本跑不通)

分步手动执行:

```bash
# 步骤 1: 采集数据
python3 collect_episode_diagnostics.py \
    --num_episodes 20 \
    --env_stage 2 \
    --num_agents 4 \
    --output diagnosis_data.jsonl

# 步骤 2: 可视化
python3 visualize_diagnostics.py \
    --input diagnosis_data.jsonl \
    --output_dir ./diagnosis_output

# 步骤 3: 打开报告
firefox ./diagnosis_output/diagnosis_report.html
```

---

**我现在在等你的训练跑完,然后你执行 `bash run_diagnosis.sh`,把 HTML 报告里的关键发现告诉我,我再帮你精确定位改哪里。**
