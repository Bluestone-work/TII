#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
obstacle_mover.py — 50 Hz 随机游走驱动 dyn_obs_0~7，支持 7 张地图

设计要点
────────
1. 使用 Wall Clock（不依赖 use_sim_time），避免 Gazebo 启动竞态导致定时器不触发
2. 随机游走：每 2~6 秒随机偏转方向；碰墙/互碰时反弹 + 随机偏转，绝不永久停止
3. 默认速度为 agent 上限的一半：0.11 m/s（更贴近行人/低速车流）
4. 每个障碍物独立维护 1 个在途 async 请求，彻底避免积压导致视觉瞬移

地图墙体结构（AABB 已含膨胀量 0.32 m = 障碍物半径 0.22 + 余量 0.10）
────────────
Map1/2 : 开阔空场，无内部障碍
Map3 (corridor_swap)   : 外墙 + 两段隔断墙 + 两根柱子
Map4 (intersection)    : 外墙 + 4 个象限大方块，中央十字通道开放
Map5 (warehouse_aisles): 外墙 + 4 排货架 + 2 个瓶颈块
Map6 (interaction_hub)      : 外墙 + 四象限块 + 四个错位窄门
Map7 (interaction_hub_mini) : 小尺度十字交汇 + 四个中央柱，交互密度更高
"""

import hashlib
import json
import math
import os
import random
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts
from gazebo_msgs.srv import SetEntityState
from std_srvs.srv import Trigger

# ─────────────────────────────────────────────────────────────────────────────
# 全局运动参数
# ─────────────────────────────────────────────────────────────────────────────
Z_ACTIVE       = 0.4     # 活跃障碍物 z 高度（圆柱体半高 0.4 m）
DEFAULT_TIMER_HZ = 15.0   # 默认定时器频率（Hz）；SetEntityState 高频异步写入会造成 GUI 中视觉跳变
UNUSED_PARK_REFRESH_S = 5.0  # 周期性停放未启用槽位，防止仿真重置后模型回到场内
UNUSED_PARK_X = 20.0
UNUSED_PARK_Y = 20.0
UNUSED_PARK_SPACING = 1.0
OBS_SPEED      = 0.22    # m/s
OBS_MIN_DIST   = 0.60    # 障碍物互碰距离（半径×2 + 余量 0.16 m）
DIR_CHANGE_MIN = 2.0     # 随机方向变更最小间隔（秒）
DIR_CHANGE_MAX = 6.0     # 随机方向变更最大间隔（秒）
MAX_BOUNCE     = 18      # 碰撞时最多尝试的反弹候选方向数（从12增加到18，减少卡死）

# ─────────────────────────────────────────────────────────────────────────────
# 每张地图的配置
#   bounds       : (xmin, xmax, ymin, ymax) 可行驶区域边界
#   aabbs        : [(bxmin,bxmax,bymin,bymax), ...] 静态障碍 AABB（已膨胀 0.32 m）
#   spawn_points : [(x,y), ...] 各障碍物安全初始位置
# ─────────────────────────────────────────────────────────────────────────────
MAP_CONFIGS = {
    # ── Map1: 原始仓库走廊（×0.6，X≈[-1.2,1.9], Y≈[-6.4,0]）──────────────────
    1: {
        'bounds': (-1.2, 1.9, -6.4, -0.3),
        'aabbs':  [],
        'spawn_points': [
            (0.4, -0.7), (0.4, -1.4), (0.4, -2.2), (0.4, -2.9),
            (0.4, -3.6), (0.4, -4.3), (0.4, -5.0), (0.4, -5.8),
        ],
    },

    # ── Map2: 原始仓库 L 型走廊（×0.6，左区 X≈[-1.1,1.7] Y≈[-6.7,0]，右区 X=[1.7,6.0]）──
    2: {
        'bounds': (-1.1, 6.0, -6.8, -0.2),
        'aabbs':  [(1.53, 1.80, -6.60, -2.22)],  # 竖向隔断墙 ×0.6
        'spawn_points': [
            ( 0.3, -0.9), ( 0.3, -2.4),   # 左区上部
            ( 0.3, -3.9), ( 0.3, -5.7),   # 左区下部
            ( 3.3, -0.9), ( 5.4, -1.2),   # 右区上部
            ( 3.3, -4.2), ( 5.4, -5.7),   # 右区下部
        ],
    },

    # ── Map3: corridor_swap（12×12m，walls ±6m）───────────────────────────────
    # 隔断墙在 x≈0，中央缺口 y∈(-0.93,0.93) 可穿越
    3: {
        'bounds': (-5.7, 5.7, -5.7, 5.7),
        'aabbs': [
            (-0.42,  0.42, -5.7, -0.93),   # divider_lower
            (-0.42,  0.42,  0.93,  5.7),   # divider_upper
            (-2.12, -0.88, -1.42, -0.18),  # pillar_left（不变）
            ( 1.18,  2.42,  0.28,  1.52),  # pillar_right（不变）
        ],
        'spawn_points': [
            (-4.5, -4.0), (-4.5,  4.0),
            (-2.0, -4.5), (-2.0,  4.5),
            ( 2.0, -4.5), ( 2.0,  4.5),
            ( 4.5, -4.0), ( 4.5,  4.0),
        ],
    },

    # ── Map4: intersection（×0.6，12×12m，十字走廊宽 1.8m）──────────────────────
    # quad blocks: center ±3.27m, size 4.74m → inner edge ±0.90m
    4: {
        'bounds': (-5.7, 5.7, -5.7, 5.7),
        'aabbs': [
            (-5.7, -0.58, -5.7, -0.58),   # quad_nn (SW)  内边 -0.90 膨胀-0.32→-0.58
            (-5.7, -0.58,  0.58,  5.7),   # quad_np (NW)
            ( 0.58,  5.7, -5.7, -0.58),   # quad_pn (SE)
            ( 0.58,  5.7,  0.58,  5.7),   # quad_pp (NE)
        ],
        'spawn_points': [
            (-3.0,  0.0), ( 3.0,  0.0),   # 左/右臂中线
            ( 0.0, -3.0), ( 0.0,  3.0),   # 南/北臂中线
            (-1.7,  0.0), ( 1.7,  0.0),   # 左/右臂靠中心
            ( 0.0, -1.7), ( 0.0,  1.7),   # 南/北臂靠中心
        ],
    },

    # ── Map5: warehouse_aisles（×0.6，12×12m，4 排货架 1.2×9.6m）────────────────
    # shelves: x=-3.9/-1.5/0.9/3.3；choke gap=0.4m（robot 0.21m 可通过）
    5: {
        'bounds': (-5.7, 5.7, -5.7, 5.7),
        'aabbs': [
            (-4.82, -2.98, -5.12,  5.12), # shelf_0 (x∈[-4.5,-3.3] +膨胀)
            (-2.42, -0.58, -5.12,  5.12), # shelf_1 (x∈[-2.1,-0.9])
            (-0.02,  1.82, -5.12,  5.12), # shelf_2 (x∈[0.3,1.5])
            ( 2.38,  4.22, -5.12,  5.12), # shelf_3 (x∈[2.7,3.9])
            (-0.96,  0.12, -0.80,  0.80), # choke_left  (gap 0.4m)
            (-0.12,  0.96, -0.80,  0.80), # choke_right
        ],
        'spawn_points': [
            (-2.7, -3.6), (-2.7,  3.6),   # 通道 1（shelf_0/shelf_1 之间）
            (-0.3, -3.3), (-0.3,  3.3),   # 中央通道（choke 两侧）
            ( 2.1, -3.6), ( 2.1,  3.0),   # 通道 3（shelf_2/shelf_3 之间）
            ( 4.5, -4.2), ( 4.5,  4.2),   # 东侧通道
        ],
    },

    # ── Map6: interaction_hub（12x12m，四臂交汇 + 错位窄门）──────────────────────
    6: {
        'bounds': (-5.7, 5.7, -5.7, 5.7),
        'aabbs': [
            (-5.7, -1.08, -5.7, -1.08),
            (-5.7, -1.08,  1.08,  5.7),
            ( 1.08,  5.7, -5.7, -1.08),
            ( 1.08,  5.7,  1.08,  5.7),
            (-4.42, -2.08, -0.02,  1.92),
            ( 2.08,  4.42, -1.92,  0.02),
            (-1.92,  0.02, -4.42, -2.08),
            (-0.02,  1.92,  2.08,  4.42),
        ],
        # 注意：旧版本把 spawn 放在四象限静态墙块内部（如 -5.0,-2.4），
        # 会导致 Gazebo 里看起来“没有动态障碍物”。这里改为四臂可通行走廊内。
        'spawn_points': [
            (-3.6, -0.60), (-2.8, -0.60),
            ( 2.8,  0.60), ( 3.6,  0.60),
            ( 0.60, -3.6), ( 0.60, -2.8),
            (-0.60,  2.8), (-0.60,  3.6),
        ],
    },

    # ── Map7: interaction_hub_mini（12x12m，小尺度十字交汇 + 中央四柱）────────
    7: {
        'bounds': (-5.7, 5.7, -5.7, 5.7),
        'aabbs': [
            (-5.7, -0.68, -5.7, -0.68),
            (-5.7, -0.68,  0.68,  5.7),
            ( 0.68,  5.7, -5.7, -0.68),
            ( 0.68,  5.7,  0.68,  5.7),
            (-1.77, -0.53, -1.77, -0.53),
            (-1.77, -0.53,  0.53,  1.77),
            ( 0.53,  1.77, -1.77, -0.53),
            ( 0.53,  1.77,  0.53,  1.77),
        ],
        'spawn_points': [
            (-4.3, -0.60), (-4.3,  0.60),
            ( 4.3, -0.60), ( 4.3,  0.60),
            (-0.60, -4.3), ( 0.60, -4.3),
            (-0.60,  4.3), ( 0.60,  4.3),
        ],
    },

    # ── Map8: circle_swap_arena (8x8m 空旷场地，中央动态障碍模拟非合作 agent)
    8: {
        'bounds': (-3.8, 3.8, -3.8, 3.8),
        'aabbs': [],  # 空旷场地无静态障碍
        'spawn_points': [
            ( 0.0,  1.5),  ( 1.0, -1.0),  (-1.5,  0.5),
            ( 0.5,  1.2),  (-0.8, -1.3),  ( 1.3,  0.0),
            (-0.5,  1.5),  ( 0.0, -1.5),
        ],
    },

    # ── Map9: warehouse_dynamic (8×8m 基于circle_swap，带静态+动态障碍物)
    # 动态障碍物在外围移动，静态障碍物由warehouse_obstacle_spawner.py生成
    9: {
        'bounds': (-3.8, 3.8, -3.8, 3.8),  # 与Map 8相同
        'aabbs': [],  # 静态障碍物由warehouse_obstacle_spawner管理
        'spawn_points': [
            # Opposite pairs keep every 2/4/6/8-obstacle prefix symmetric.
            ( 0.0,  3.0),  ( 0.0, -3.0),
            ( 3.0,  0.0),  (-3.0,  0.0),
            ( 2.5,  2.0),  (-2.5, -2.0),
            ( 2.5, -2.0),  (-2.5,  2.0),
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
def _in_wall(x: float, y: float,
             aabbs: list,
             xmin: float, xmax: float,
             ymin: float, ymax: float) -> bool:
    """判断点 (x,y) 是否在墙体/障碍物内（含膨胀裕量）。"""
    if x <= xmin or x >= xmax or y <= ymin or y >= ymax:
        return True
    for bxmin, bxmax, bymin, bymax in aabbs:
        if bxmin < x < bxmax and bymin < y < bymax:
            return True
    return False


OBS_MIN_DIST = 0.60  # 两个动态障碍物中心间最小安全距离（半径 0.22×2 + 余量 0.16）


def _obs_collides(x: float, y: float, others: list) -> bool:
    """检查 (x,y) 是否与其他障碍物位置重叠。others = [(ox,oy), ...]"""
    for ox, oy in others:
        if math.hypot(x - ox, y - oy) < OBS_MIN_DIST:
            return True
    return False


class _ObsState:
    """
    随机游走状态机。

    行为逻辑：
    - 以固定速度沿当前角度 θ 行进
    - 每隔 DIR_CHANGE_MIN ~ DIR_CHANGE_MAX 秒随机偏转 ±135°
    - 碰到墙体或其他障碍物时：以"正反方向"为基准，依次扩大偏转范围，
      最多尝试 MAX_BOUNCE 个候选角度；找到可行角度则执行，绝不永久停止
    - 极端情况（完全夹住）：随机重置方向，下帧继续尝试（等待 0.1 s）
    """

    def __init__(self, name: str, x0: float, y0: float, speed: float,
                 aabbs: list, xmin: float, xmax: float,
                 ymin: float, ymax: float, rng: random.Random):
        self.name    = name
        self.cur_x   = float(x0)
        self.cur_y   = float(y0)
        self._speed  = speed
        self._aabbs  = aabbs
        self._bounds = (xmin, xmax, ymin, ymax)
        self._rng    = rng
        self._inflight  = None
        self._angle     = rng.uniform(0.0, 2.0 * math.pi)
        self._dir_timer = rng.uniform(DIR_CHANGE_MIN, DIR_CHANGE_MAX)
        self._inflight_time = 0.0

    @property
    def is_ready(self) -> bool:
        """上一次 set_entity_state 请求已完成（或从未发出），可以发下一条。"""
        if self._inflight is None or self._inflight.done():
            return True
        return False

    def advance(self, dt: float, others: list) -> tuple:
        """
        推进一步。others = [(ox,oy), ...] 其他障碍物当前位置（用于互碰检测）。
        返回新的 (x, y)，绝不返回"停在原地超过 1 帧"的结果。
        """
        # ── 随机方向变更倒计时 ─────────────────────────────────────────────
        self._dir_timer -= dt
        if self._dir_timer <= 0.0:
            self._angle += self._rng.uniform(-math.pi * 0.75, math.pi * 0.75)
            self._angle %= 2.0 * math.pi
            self._dir_timer = self._rng.uniform(DIR_CHANGE_MIN, DIR_CHANGE_MAX)

        # ── 尝试按当前方向前进 ─────────────────────────────────────────────
        step = self._speed * dt
        nx = self.cur_x + step * math.cos(self._angle)
        ny = self.cur_y + step * math.sin(self._angle)

        if not (_in_wall(nx, ny, self._aabbs, *self._bounds) or
                _obs_collides(nx, ny, others)):
            self.cur_x, self.cur_y = nx, ny
            return nx, ny

        # ── 碰撞反弹：依次扩大偏转范围，找到可行方向 ──────────────────────
        base = self._angle + math.pi   # 以正反方向为基准
        for k in range(MAX_BOUNCE):
            # 从小偏转逐步扩大到整个半球（±π），总能找到出路
            spread = math.pi * (k + 1) / MAX_BOUNCE
            candidate = base + self._rng.uniform(-spread, spread)
            candidate %= 2.0 * math.pi
            bx = self.cur_x + step * math.cos(candidate)
            by = self.cur_y + step * math.sin(candidate)
            if not (_in_wall(bx, by, self._aabbs, *self._bounds) or
                    _obs_collides(bx, by, others)):
                self._angle = candidate
                self._dir_timer = self._rng.uniform(DIR_CHANGE_MIN, DIR_CHANGE_MAX)
                self.cur_x, self.cur_y = bx, by
                return bx, by

        # ── 极端情况（被完全夹住）：强制小步移动 + 随机方向 ───────────────
        self._angle = self._rng.uniform(0.0, 2.0 * math.pi)
        self._dir_timer = 0.05   # 缩短等待时间，更快重试

        # 尝试在当前位置附近随机小步移动（0.02m），即使碰撞也移动
        # 这样可以打破"多个障碍物互相卡住"的僵局
        tiny_step = 0.02  # 2cm 微移
        escape_x = self.cur_x + tiny_step * math.cos(self._angle)
        escape_y = self.cur_y + tiny_step * math.sin(self._angle)

        # 检查是否在边界内（允许与其他障碍物重叠，因为需要打破僵局）
        if not (escape_x <= self._bounds[0] or escape_x >= self._bounds[1] or
                escape_y <= self._bounds[2] or escape_y >= self._bounds[3]):
            self.cur_x, self.cur_y = escape_x, escape_y
            return escape_x, escape_y

        # 边界外则保持原地，但下一帧会用新的随机方向重试
        return self.cur_x, self.cur_y


# ─────────────────────────────────────────────────────────────────────────────
class ObstacleMover(Node):

    def __init__(self):
        super().__init__('obstacle_mover')

        # ── 参数声明 ───────────────────────────────────────────────────────
        self.declare_parameter('num_obstacles', 8)
        self.declare_parameter('map_number',    3)
        # speed_scale 统一声明为浮点，兼容 launch/CLI 传入 1.0
        self.declare_parameter('speed_scale',   1.0)
        self.declare_parameter('update_hz',     DEFAULT_TIMER_HZ)
        self.declare_parameter('seed',          0)
        self.declare_parameter('use_current_positions', False)  # 是否从当前位置开始
        self.declare_parameter('fixed_initial_states_file', '')

        n_obs   = min(8, max(0, int(self.get_parameter('num_obstacles').value)))
        scale   = float(self.get_parameter('speed_scale').value)
        map_num = int(self.get_parameter('map_number').value)
        update_hz = float(self.get_parameter('update_hz').value)
        update_hz = max(5.0, min(100.0, update_hz))
        self._dt = 1.0 / update_hz
        self._update_hz = update_hz

        # ── 选取地图配置（未知地图回退到 map1）──────────────────────────
        cfg    = MAP_CONFIGS.get(map_num, MAP_CONFIGS[1])
        aabbs  = cfg['aabbs']
        xmin, xmax, ymin, ymax = cfg['bounds']
        spawns = cfg['spawn_points']
        fixed_states_file = os.path.abspath(os.path.expanduser(
            str(self.get_parameter('fixed_initial_states_file').value or '').strip()
        )) if str(self.get_parameter('fixed_initial_states_file').value or '').strip() else ''
        self._fixed_initial_states = {}
        self._current_initial_state = {}
        if fixed_states_file:
            with open(fixed_states_file, 'r', encoding='utf-8') as handle:
                fixed_payload = json.load(handle)
            if int(fixed_payload.get('map_number', -1)) != map_num:
                raise RuntimeError(
                    f'fixed dynamic-state map mismatch: file={fixed_states_file} '
                    f'expected={map_num} found={fixed_payload.get("map_number")}'
                )
            expected_count = int(fixed_payload.get('expected_num_dynamic_obstacles', -1))
            if expected_count != n_obs:
                raise RuntimeError(
                    f'fixed dynamic-state obstacle count mismatch: configured={n_obs} '
                    f'file={expected_count}'
                )
            self._fixed_initial_states = {
                int(item['episode_seed']): dict(item)
                for item in fixed_payload.get('episodes', [])
            }
            if not self._fixed_initial_states:
                raise RuntimeError(f'fixed dynamic-state file has no episodes: {fixed_states_file}')
            config_file = os.path.join(
                os.path.dirname(os.path.dirname(fixed_states_file)),
                'config',
                'dynamic_obstacles.json',
            )
            with open(config_file, 'r', encoding='utf-8') as handle:
                fixed_config = json.load(handle)
            spawns = fixed_config['maps'][str(map_num)]['spawn_points']

        # ── 如果spawn_points为空，强制禁用动态障碍物 ──────────────────────
        if len(spawns) == 0:
            n_obs = 0
            self.get_logger().info(f'[Map {map_num}] spawn_points为空，动态障碍物已禁用')

        if n_obs > 0 and len(spawns) > 0:
            self._validate_spawn_distances(spawns, map_num)

        # ── 服务客户端 ─────────────────────────────────────────────────────
        self._client = self.create_client(SetEntityState, '/set_entity_state')

        # ── 构建随机游走状态机 ────────────────────────────────────────────
        self._base_seed = int(self.get_parameter('seed').value)
        self._speed = OBS_SPEED * scale
        self._aabbs = aabbs
        self._bounds = (xmin, xmax, ymin, ymax)
        self._spawn_points = tuple(spawns)
        self._n_active = n_obs
        self._states = self._make_states(self._base_seed)
        self._skipped_busy_ticks = 0

        # ── 等待服务上线（最多 20 s）──────────────────────────────────────
        if not self._client.wait_for_service(timeout_sec=20.0):
            self.get_logger().warn(
                '/set_entity_state 20 s 内未上线，将在首次 tick 时重试')
        # ── 未激活障碍物停放状态 ─────────────────────────────────────────────
        # Gazebo world 文件里始终有 8 个 dyn_obs 模型；num_obstacles < 8 时，
        # 未被驱动的模型需要主动移到地图外让它们不可见、不碰撞。
        # 不删除运行中的 Gazebo 模型：GUI 和 spawn_entity 并发启动时，
        # DeleteEntity 会让物理线程访问已释放的 Inertial 指针并崩溃。
        self._unused_parked = (n_obs >= 8)   # 全 8 个都激活则无需停放
        self._unused_park_futures = {}
        self._tick_count = 0
        self._unused_park_refresh_ticks = max(
            1, int(round(self._update_hz * UNUSED_PARK_REFRESH_S)))
        # ── 定时器使用 Simulation Clock（与Gazebo RTF加速同步，保持障碍物sim-time速率恒定）
        self._sim_timer = self.create_timer(
            self._dt, self._tick)
        self._paused = False
        self._paired_frames = []
        self._paired_frame_index = 0
        self._paired_trajectory_fingerprint = ''
        self._paired_trajectory_spec = {}
        self._paired_apply_futures = []
        self._reset_service = self.create_service(
            AddTwoInts, '/reset_dynamic_obstacles', self._reset_episode)
        self._start_service = self.create_service(
            Trigger, '/start_dynamic_obstacles', self._start_episode)
        self._advance_service = self.create_service(
            Trigger, '/advance_dynamic_obstacles', self._advance_paired_episode)
        self._sync_service = self.create_service(
            Trigger, '/sync_dynamic_obstacles', self._sync_paired_episode)
        self._snapshot_counter = 0
        self._snapshots = {}
        self._snapshot_service = self.create_service(
            AddTwoInts, '/snapshot_dynamic_obstacles', self._snapshot_episode)
        self._restore_service = self.create_service(
            AddTwoInts, '/restore_dynamic_obstacles', self._restore_snapshot)

        self.get_logger().info(
            f'[obstacle_mover] map={map_num}  {n_obs} 个障碍物  '
            f'速度={self._speed:.3f} m/s (scale×{scale})  {self._update_hz:.0f} Hz  随机游走模式')

    def _make_states(self, seed: int) -> list:
        """Create the exact seeded obstacle state used at an episode boundary."""
        if not self._spawn_points:
            return []
        rng = random.Random(int(seed))
        xmin, xmax, ymin, ymax = self._bounds
        states = []
        used_positions = []
        for i in range(self._n_active):
            x0, y0 = self._spawn_points[i % len(self._spawn_points)]
            for _attempt in range(20):
                test_x = x0 + rng.uniform(-0.10, 0.10)
                test_y = y0 + rng.uniform(-0.10, 0.10)
                if not _obs_collides(test_x, test_y, used_positions):
                    x0, y0 = test_x, test_y
                    break
            used_positions.append((x0, y0))
            states.append(
                _ObsState(
                    f'dyn_obs_{i}', x0, y0, self._speed,
                    self._aabbs, xmin, xmax, ymin, ymax, rng,
                )
            )
        if self._fixed_initial_states:
            expected = self._fixed_initial_states.get(int(seed))
            if expected is None:
                raise RuntimeError(
                    f'episode seed {seed} is absent from the fixed dynamic-state file'
                )
            actual_positions = {
                state.name: [float(state.cur_x), float(state.cur_y)]
                for state in states
            }
            if int(expected.get('count', -1)) != len(states):
                raise RuntimeError(
                    f'fixed dynamic-state count mismatch for seed {seed}: '
                    f'expected={expected.get("count")} actual={len(states)}'
                )
            if actual_positions != expected.get('positions'):
                raise RuntimeError(
                    f'fixed dynamic-state positions mismatch for seed {seed}: '
                    f'expected={expected.get("positions")} actual={actual_positions}'
                )
            self._current_initial_state = {
                'episode_index': int(expected['episode_index']),
                'seed': int(seed),
                'count': len(states),
                'positions': actual_positions,
                'state_hash': str(expected['state_hash']),
            }
        else:
            self._current_initial_state = {
                'seed': int(seed),
                'count': len(states),
                'positions': {
                    state.name: [float(state.cur_x), float(state.cur_y)]
                    for state in states
                },
            }
        return states

    @staticmethod
    def _frame(states: list) -> list:
        return [
            [state.name, round(float(state.cur_x), 9), round(float(state.cur_y), 9)]
            for state in states
        ]

    def _precompute_paired_trajectory(self, seed: int, max_steps: int) -> None:
        states = self._make_states(seed)
        frames = [self._frame(states)]
        control_dt = 0.1
        for _ in range(max_steps):
            current = [(state.cur_x, state.cur_y) for state in states]
            for index, state in enumerate(states):
                others = [position for other, position in enumerate(current) if other != index]
                state.advance(control_dt, others)
            frames.append(self._frame(states))
        spec = {
            'algorithm': 'seeded_random_walk_v3_step_synchronous',
            'episode_seed': int(seed),
            'map_number': int(self.get_parameter('map_number').value),
            'num_dynamic_obstacles': len(states),
            'obstacle_speed_mps': round(float(self._speed), 9),
            'control_dt': control_dt,
            'max_episode_steps': int(max_steps),
            'dynamic_initial_state': self._current_initial_state,
            'frames': frames,
        }
        payload = json.dumps(spec, sort_keys=True, separators=(',', ':')).encode('ascii')
        self._paired_frames = frames
        self._paired_frame_index = 0
        self._paired_trajectory_fingerprint = hashlib.sha256(payload).hexdigest()
        self._paired_trajectory_spec = {
            key: value for key, value in spec.items() if key != 'frames'
        }
        self._paired_trajectory_spec['frame_count'] = len(frames)

    def _reset_episode(self, request, response):
        """Reset and pause a deterministic trajectory for an explicit episode seed."""
        seed = int(request.a)
        max_steps = int(request.b)
        self._paused = True
        if max_steps > 0:
            self._precompute_paired_trajectory(seed, max_steps)
        else:
            self._paired_frames = []
            self._paired_frame_index = 0
            self._paired_trajectory_fingerprint = ''
            self._paired_trajectory_spec = {}
        self._states = self._make_states(seed)
        self._paired_apply_futures = []
        for state in self._states:
            state._inflight = self._send(state.name, state.cur_x, state.cur_y)
            self._paired_apply_futures.append(state._inflight)
        response.sum = seed
        self.get_logger().info(
            f'[obstacle_mover] paired trajectory reset seed={seed} obstacles={len(self._states)} '
            f'max_steps={max_steps}')
        return response

    def _start_episode(self, _request, response):
        """Start a paused trajectory with a fresh timer phase."""
        if self._paired_frames:
            self._paused = True
            response.message = json.dumps({
                'fingerprint': self._paired_trajectory_fingerprint,
                'spec': self._paired_trajectory_spec,
            }, sort_keys=True, separators=(',', ':'))
        else:
            self._sim_timer.reset()
            self._paused = False
            response.message = 'deterministic obstacle trajectory started'
        response.success = True
        return response

    def _advance_paired_episode(self, _request, response):
        """Apply exactly one precomputed obstacle frame per environment step."""
        next_index = self._paired_frame_index + 1
        if not self._paired_frames or next_index >= len(self._paired_frames):
            response.success = False
            response.message = 'paired trajectory exhausted or not initialized'
            return response
        frame = self._paired_frames[next_index]
        self._paired_apply_futures = []
        by_name = {state.name: state for state in self._states}
        for name, x, y in frame:
            state = by_name[name]
            state.cur_x = float(x)
            state.cur_y = float(y)
            state._inflight = self._send(name, x, y)
            self._paired_apply_futures.append(state._inflight)
        self._paired_frame_index = next_index
        response.success = True
        response.message = str(next_index)
        return response

    def _sync_paired_episode(self, _request, response):
        """Barrier ensuring the current paired frame reached Gazebo."""
        pending = [future for future in self._paired_apply_futures if not future.done()]
        failed = [
            future for future in self._paired_apply_futures
            if future.done() and (
                future.result() is None or not bool(future.result().success)
            )
        ]
        response.success = not pending and not failed
        response.message = (
            f'apply_failed:{len(failed)}' if failed else str(self._paired_frame_index)
        )
        return response

    def _export_snapshot(self) -> dict:
        rng_state = self._states[0]._rng.getstate() if self._states else None
        return {
            'rng_state': rng_state,
            'states': [
                {
                    'name': state.name,
                    'cur_x': float(state.cur_x),
                    'cur_y': float(state.cur_y),
                    'speed': float(state._speed),
                    'aabbs': state._aabbs,
                    'bounds': tuple(state._bounds),
                    'angle': float(state._angle),
                    'dir_timer': float(state._dir_timer),
                }
                for state in self._states
            ],
        }

    def _import_snapshot(self, payload: dict) -> None:
        rng = random.Random()
        if payload['rng_state'] is not None:
            rng.setstate(payload['rng_state'])
        restored = []
        for item in payload['states']:
            state = object.__new__(_ObsState)
            state.name = item['name']
            state.cur_x = float(item['cur_x'])
            state.cur_y = float(item['cur_y'])
            state._speed = float(item['speed'])
            state._aabbs = item['aabbs']
            state._bounds = tuple(item['bounds'])
            state._rng = rng
            state._angle = float(item['angle'])
            state._dir_timer = float(item['dir_timer'])
            state._inflight = None
            state._inflight_time = 0.0
            restored.append(state)
        self._states = restored
        for state in self._states:
            state._inflight = self._send(state.name, state.cur_x, state.cur_y)

    def _snapshot_episode(self, request, response):
        """Pause and retain a branch-restorable obstacle state-machine snapshot."""
        self._paused = True
        requested = int(request.a)
        if requested > 0:
            snapshot_id = requested
        else:
            self._snapshot_counter += 1
            snapshot_id = self._snapshot_counter
        self._snapshots[snapshot_id] = self._export_snapshot()
        response.sum = snapshot_id
        return response

    def _restore_snapshot(self, request, response):
        """Restore a prior snapshot and remain paused until start is requested."""
        snapshot_id = int(request.a)
        payload = self._snapshots.get(snapshot_id)
        if payload is None:
            response.sum = -1
            return response
        self._paused = True
        self._import_snapshot(payload)
        response.sum = snapshot_id
        return response

    def _validate_spawn_distances(self, spawn_points, map_num):
        """验证所有 spawn 点之间的距离，确保不会初始化太近"""
        if len(spawn_points) < 2:
            return

        min_dist = float('inf')
        min_pair = None

        for i in range(len(spawn_points)):
            for j in range(i + 1, len(spawn_points)):
                x1, y1 = spawn_points[i]
                x2, y2 = spawn_points[j]
                dist = math.hypot(x2 - x1, y2 - y1)

                if dist < min_dist:
                    min_dist = dist
                    min_pair = (i, j)

        if min_dist < OBS_MIN_DIST:
            self.get_logger().warn(
                f'  Map{map_num} spawn 点 {min_pair[0]} 和 {min_pair[1]} 距离过近: '
                f'{min_dist:.2f}m < {OBS_MIN_DIST}m，可能导致障碍物初始化冲突')
        else:
            self.get_logger().info(
                f' Map{map_num} spawn 点最小距离: {min_dist:.2f}m (OK)')

    # ── 禁用未激活障碍物 ──────────────────────────────────────────────────────
    def _park_unused(self):
        """将未启用模型移到地图外，避免 Gazebo 运行期删除实体的竞态。"""
        if self._n_active >= 8:
            self._unused_parked = True
            return
        for i in range(self._n_active, 8):
            previous = self._unused_park_futures.get(i)
            if previous is not None and not previous.done():
                continue
            req = SetEntityState.Request()
            req.state.name = f'dyn_obs_{i}'
            req.state.pose.position.x = UNUSED_PARK_X + i * UNUSED_PARK_SPACING
            req.state.pose.position.y = UNUSED_PARK_Y
            req.state.pose.position.z = Z_ACTIVE
            req.state.pose.orientation.w = 1.0
            self._unused_park_futures[i] = self._client.call_async(req)
        if not self._unused_parked:
            self._unused_parked = True
            self.get_logger().info(
                f'[obstacle_mover] dyn_obs_{self._n_active}~dyn_obs_7 已停放到地图外禁用'
                '（将周期性刷新以抵抗仿真重置）')

    # ── 定时回调 ──────────────────────────────────────────────────────────────
    def _tick(self):
        if not self._client.service_is_ready():
            return
        self._tick_count += 1
        # 首次服务就绪时立即停放，之后周期性刷新。Gazebo 的 world/reset
        # 可能恢复模型初始位姿，不能把“已停放”当作永久状态。
        if (not self._unused_parked or
                self._tick_count % self._unused_park_refresh_ticks == 0):
            self._park_unused()
        if self._paused:
            return
        # 先快照所有障碍物的当前位置（用于互碰检测，避免用刚更新的位置误判）
        cur_pos = [(s.cur_x, s.cur_y) for s in self._states]
        for i, s in enumerate(self._states):
            # 每个障碍物独立维护 1 个在途请求：
            #   - 上次请求仍在途 → 跳过（避免请求积压 → Gazebo 乱序回包 → 视觉瞬移）
            #   - 请求已完成    → 计算下一步位置并发送
            if not s.is_ready:
                self._skipped_busy_ticks += 1
                if self._skipped_busy_ticks % int(max(1.0, self._update_hz * 10.0)) == 0:
                    self.get_logger().warn(
                        '[obstacle_mover] set_entity_state service is slow; '
                        'skipping dynamic obstacle update to preserve continuous motion')
                continue
            others = [p for j, p in enumerate(cur_pos) if j != i]
            nx, ny = s.advance(self._dt, others)
            import time
            s._inflight_time = time.time()
            s._inflight = self._send(s.name, nx, ny)

    def _send(self, name: str, x: float, y: float, z: float = Z_ACTIVE):
        req = SetEntityState.Request()
        req.state.name               = name
        req.state.pose.position.x    = float(x)
        req.state.pose.position.y    = float(y)
        req.state.pose.position.z    = float(z)
        req.state.pose.orientation.w = 1.0
        return self._client.call_async(req)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleMover()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
