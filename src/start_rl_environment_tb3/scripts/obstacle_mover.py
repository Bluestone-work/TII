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
Map4 (intersection)    : 外墙 + 4 个象限大方块 + 2 个中央小方块
Map5 (warehouse_aisles): 外墙 + 4 排货架 + 2 个瓶颈块
Map6 (interaction_hub)      : 外墙 + 四象限块 + 四个错位窄门
Map7 (interaction_hub_mini) : 小尺度十字交汇 + 四个中央柱，交互密度更高
"""

import math
import random
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState

# ─────────────────────────────────────────────────────────────────────────────
# 全局运动参数
# ─────────────────────────────────────────────────────────────────────────────
Z_ACTIVE       = 0.4     # 活跃障碍物 z 高度（圆柱体半高 0.4 m）
DEFAULT_TIMER_HZ = 30.0   # 默认定时器频率（Hz）；并行多开时 100Hz 负载过高，易视觉跳变
OBS_SPEED      = 0.11    # m/s，默认约为 agent max_linear_vel(0.22) 的一半
OBS_MIN_DIST   = 0.60    # 障碍物互碰距离（半径×2 + 余量 0.16 m）
DIR_CHANGE_MIN = 2.0     # 随机方向变更最小间隔（秒）
DIR_CHANGE_MAX = 6.0     # 随机方向变更最大间隔（秒）
MAX_BOUNCE     = 12      # 碰撞时最多尝试的反弹候选方向数

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
    # center blocks: center ±0.36m, size 0.48m
    4: {
        'bounds': (-5.7, 5.7, -5.7, 5.7),
        'aabbs': [
            (-5.7, -0.58, -5.7, -0.58),   # quad_nn (SW)  内边 -0.90 膨胀-0.32→-0.58
            (-5.7, -0.58,  0.58,  5.7),   # quad_np (NW)
            ( 0.58,  5.7, -5.7, -0.58),   # quad_pn (SE)
            ( 0.58,  5.7,  0.58,  5.7),   # quad_pp (NE)
            (-0.92,  0.20, -0.92,  0.20), # center_block1 (膨胀 0.32)
            (-0.20,  0.92, -0.20,  0.92), # center_block2
        ],
        'spawn_points': [
            (-2.7,  0.5), (-2.7, -0.5),   # 左臂中段
            ( 2.7,  0.5), ( 2.7, -0.5),   # 右臂中段
            ( 0.5, -2.7), (-0.5, -2.7),   # 南臂中段
            ( 0.5,  2.7), (-0.5,  2.7),   # 北臂中段
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
        import time
        if self._inflight is None or self._inflight.done():
            return True
        # 加入 0.2 秒超时机制，避免高负载下丢失服务响应导致障碍物永久停止
        if (time.time() - self._inflight_time) > 0.2:
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

        # ── 极端情况（被完全夹住）：随机重置方向，原地等 1 帧 ───────────
        self._angle = self._rng.uniform(0.0, 2.0 * math.pi)
        self._dir_timer = 0.1   # 0.1 s 后立刻再次尝试，不永久停止
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

        # ── 服务客户端 ─────────────────────────────────────────────────────
        self._client = self.create_client(SetEntityState, '/set_entity_state')

        # ── 构建随机游走状态机 ────────────────────────────────────────────
        rng   = random.Random()   # 每次启动随机种子，行为不重复
        speed = OBS_SPEED * scale
        self._states: list = []
        for i in range(n_obs):
            x0, y0 = spawns[i % len(spawns)]
            # 在初始点附近随机抖动，防止多个障碍物完全重叠
            x0 += rng.uniform(-0.25, 0.25)
            y0 += rng.uniform(-0.25, 0.25)
            self._states.append(
                _ObsState(f'dyn_obs_{i}', x0, y0, speed,
                          aabbs, xmin, xmax, ymin, ymax, rng))

        # ── 等待服务上线（最多 20 s）──────────────────────────────────────
        if not self._client.wait_for_service(timeout_sec=20.0):
            self.get_logger().warn(
                '/set_entity_state 20 s 内未上线，将在首次 tick 时重试')
        # ── 标记未激活障碍物是否已下沉 ───────────────────────────────────────
        # Gazebo world 文件里始终有 8 个 dyn_obs 模型；num_obstacles < 8 时，
        # 未被驱动的模型需要主动移到 z=-10 让它们不可见、不碰撞。
        # 不使用延迟定时器，而是在 _tick 首次检测到服务就绪时立即执行，
        # 避免 3s 时服务仍未就绪导致静默跳过、障碍物永远不下沉的问题。
        self._n_active    = n_obs
        self._unused_sunk = (n_obs >= 8)   # 全 8 个都激活则无需下沉
        # ── 定时器使用 Wall Clock（彻底规避 use_sim_time 导致 /clock 未就绪时不触发）
        import rclpy.clock as _rclpy_clock
        self._wall_timer = self.create_timer(
            self._dt, self._tick,
            clock=_rclpy_clock.Clock())

        self.get_logger().info(
            f'[obstacle_mover] map={map_num}  {n_obs} 个障碍物  '
            f'速度={speed:.3f} m/s (scale×{scale})  {self._update_hz:.0f} Hz  随机游走模式')

    # ── 下沉未激活障碍物（仅执行一次，由 _tick 在服务就绪后调用）────────────
    def _sink_unused_once(self):
        """将 dyn_obs_{n_active}~dyn_obs_7 下沉至 z=-10.0，只执行一次。"""
        self._unused_sunk = True
        if self._n_active >= 8:
            return
        for i in range(self._n_active, 8):
            req = SetEntityState.Request()
            req.state.name = f'dyn_obs_{i}'
            req.state.pose.position.x = 0.0
            req.state.pose.position.y = 0.0
            req.state.pose.position.z = -10.0
            req.state.pose.orientation.w = 1.0
            self._client.call_async(req)
        self.get_logger().info(
            f'[obstacle_mover] ✅ dyn_obs_{self._n_active}~dyn_obs_7 已下沉至 z=-10.0')

    # ── 定时回调 ──────────────────────────────────────────────────────────────
    def _tick(self):
        if not self._client.service_is_ready():
            return
        # 服务首次就绪：立即下沉未激活的障碍物（比 3s 定时器更可靠）
        if not self._unused_sunk:
            self._sink_unused_once()
        # 先快照所有障碍物的当前位置（用于互碰检测，避免用刚更新的位置误判）
        cur_pos = [(s.cur_x, s.cur_y) for s in self._states]
        for i, s in enumerate(self._states):
            # 每个障碍物独立维护 1 个在途请求：
            #   - 上次请求仍在途 → 跳过（避免请求积压 → Gazebo 乱序回包 → 视觉瞬移）
            #   - 请求已完成    → 计算下一步位置并发送
            if not s.is_ready:
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
        rclpy.shutdown()


if __name__ == '__main__':
    main()
