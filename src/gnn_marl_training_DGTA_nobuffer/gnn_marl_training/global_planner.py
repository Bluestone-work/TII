"""
全局路径规划器 - 改进版
包含视线剪枝（Floyd算法简化）和更好的路径点提取
"""
import numpy as np
import heapq
import math
from typing import List, Optional, Tuple
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import maximum_filter



class PathTrackingUtils:
    """基于全局路径的连续前瞻追踪工具。"""

    @staticmethod
    def _segment_length(a, b):
        return float(math.hypot(b[0] - a[0], b[1] - a[1]))

    @staticmethod
    def project_to_segment(point, seg_start, seg_end):
        px, py = point
        ax, ay = seg_start
        bx, by = seg_end
        abx = bx - ax
        aby = by - ay
        ab2 = abx * abx + aby * aby
        if ab2 < 1e-8:
            return (float(ax), float(ay)), 0.0
        apx = px - ax
        apy = py - ay
        t = (apx * abx + apy * aby) / ab2
        t = max(0.0, min(1.0, t))
        return (float(ax + t * abx), float(ay + t * aby)), float(t)

    @classmethod
    def get_path_projection(
        cls,
        point,
        path_points,
        reference_progress=None,
        max_backtrack=0.40,
        max_forward_snap=2.00,
    ):
        if not path_points:
            return {
                'projection': tuple(point),
                'segment_index': 0,
                'segment_t': 0.0,
                'arc_progress': 0.0,
                'lateral_error': 0.0,
            }
        if len(path_points) == 1:
            proj = tuple(path_points[0])
            return {
                'projection': proj,
                'segment_index': 0,
                'segment_t': 1.0,
                'arc_progress': 0.0,
                'lateral_error': float(math.hypot(point[0] - proj[0], point[1] - proj[1])),
            }

        best = None
        best_global = None
        min_progress = None
        max_progress = None
        if reference_progress is not None and math.isfinite(float(reference_progress)):
            min_progress = max(0.0, float(reference_progress) - max(0.0, float(max_backtrack)))
            max_progress = float(reference_progress) + max(0.0, float(max_forward_snap))
        acc_len = 0.0
        for idx in range(len(path_points) - 1):
            a = path_points[idx]
            b = path_points[idx + 1]
            proj, t = cls.project_to_segment(point, a, b)
            seg_len = cls._segment_length(a, b)
            lateral = float(math.hypot(point[0] - proj[0], point[1] - proj[1]))
            cand = {
                'projection': proj,
                'segment_index': idx,
                'segment_t': t,
                'arc_progress': acc_len + t * seg_len,
                'lateral_error': lateral,
            }
            if best_global is None or cand['lateral_error'] < best_global['lateral_error']:
                best_global = cand
            in_progress_window = (
                min_progress is None
                or (min_progress - 1e-8 <= cand['arc_progress'] <= max_progress + 1e-8)
            )
            if in_progress_window and (best is None or cand['lateral_error'] < best['lateral_error']):
                best = cand
            acc_len += seg_len
        return best if best is not None else best_global

    @classmethod
    def get_rolling_subgoal(
        cls,
        point,
        path_points,
        lookahead_dist=0.8,
        reference_progress=None,
        max_backtrack=0.40,
        max_forward_snap=2.00,
    ):
        if not path_points:
            return {
                'subgoal': tuple(point),
                'projection': tuple(point),
                'segment_index': 0,
                'segment_t': 0.0,
                'arc_progress': 0.0,
                'lateral_error': 0.0,
                'path_heading': 0.0,
            }
        if len(path_points) == 1:
            proj_info = cls.get_path_projection(point, path_points)
            proj_info['subgoal'] = tuple(path_points[0])
            proj_info['path_heading'] = 0.0
            return proj_info

        proj_info = cls.get_path_projection(
            point,
            path_points,
            reference_progress=reference_progress,
            max_backtrack=max_backtrack,
            max_forward_snap=max_forward_snap,
        )
        seg_idx = proj_info['segment_index']
        seg_t = proj_info['segment_t']
        projection = proj_info['projection']
        remaining = max(0.0, float(lookahead_dist))

        i = seg_idx
        curr = projection
        while i < len(path_points) - 1:
            seg_end = path_points[i + 1]
            seg_len = cls._segment_length(curr, seg_end)
            if remaining <= seg_len + 1e-8:
                ratio = 0.0 if seg_len < 1e-8 else remaining / seg_len
                sx = curr[0] + ratio * (seg_end[0] - curr[0])
                sy = curr[1] + ratio * (seg_end[1] - curr[1])
                path_heading = float(math.atan2(seg_end[1] - curr[1], seg_end[0] - curr[0]))
                proj_info['subgoal'] = (float(sx), float(sy))
                proj_info['path_heading'] = path_heading
                return proj_info
            remaining -= seg_len
            i += 1
            curr = path_points[i]

        prev_idx = max(0, len(path_points) - 2)
        prev_pt = path_points[prev_idx]
        end_pt = path_points[-1]
        proj_info['subgoal'] = tuple(end_pt)
        proj_info['path_heading'] = float(math.atan2(end_pt[1] - prev_pt[1], end_pt[0] - prev_pt[0]))
        return proj_info


class AStarPlanner:
    """A*路径规划器"""
    
    def __init__(
        self,
        map_data,
        resolution=0.05,
        origin=(-10.0, -10.0),
        use_voronoi: bool = False,
        voronoi_min_clearance_m: float = 0.35,
        inflation_margin_m: float = 0.40,
        clearance_cost_weight: float = 0.0,
        preferred_clearance_m: float = 0.65,
        prune_min_clearance_m: float = 0.55,
    ):
        self.map_data = map_data
        self.resolution = resolution
        self.origin = origin
        self.height, self.width = map_data.shape
        self.obstacle_mask = (map_data > 50).astype(np.uint8)
        # 自由空间到最近障碍物的欧式距离（像素/米）
        free_mask = (self.obstacle_mask == 0).astype(np.uint8)
        self.clearance_map_px = distance_transform_edt(free_mask)
        self.clearance_map_m = self.clearance_map_px * self.resolution
        # 依据期望安全边界(米)换算膨胀半径，默认 0.40m
        inflation_radius_px = max(1, int(round(float(inflation_margin_m) / self.resolution)))
        self.inflated_map = self._inflate_obstacles(map_data, inflation_radius=inflation_radius_px)
        self.use_voronoi = bool(use_voronoi)
        self.voronoi_min_clearance_m = float(voronoi_min_clearance_m)
        self.clearance_cost_weight = max(0.0, float(clearance_cost_weight))
        self.preferred_clearance_m = max(0.0, float(preferred_clearance_m))
        self.prune_min_clearance_m = max(0.0, float(prune_min_clearance_m))
        self.voronoi_skeleton = self._build_voronoi_skeleton() if self.use_voronoi else None
        
    def _inflate_obstacles(self, map_data, inflation_radius=3):
        from scipy.ndimage import binary_dilation
        obstacle_mask = (map_data > 50).astype(np.uint8)
        inflated = binary_dilation(obstacle_mask, iterations=inflation_radius)
        return inflated.astype(np.uint8) * 100
    
    def world_to_grid(self, x, y):
        grid_x = int((x - self.origin[0]) / self.resolution)
        # The env passes map_data after np.flipud(...), so row 0 already
        # corresponds to the map origin's lower Y edge in world coordinates.
        grid_y = int((y - self.origin[1]) / self.resolution)
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):
        x = (grid_x + 0.5) * self.resolution + self.origin[0]
        y = (grid_y + 0.5) * self.resolution + self.origin[1]
        return x, y

    def get_clearance_m(self, world_x, world_y) -> float:
        gx, gy = self.world_to_grid(world_x, world_y)
        gx = int(np.clip(gx, 0, self.width - 1))
        gy = int(np.clip(gy, 0, self.height - 1))
        return float(self.clearance_map_m[gy, gx])

    def _build_dynamic_free_mask(
        self,
        blocked_world_points: List[Tuple[float, float]],
        block_radius_m: float,
        start: Optional[Tuple[int, int]] = None,
        goal: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        free_mask = (self.inflated_map <= 50).copy()
        if not blocked_world_points:
            return free_mask

        radius_px = max(1, int(math.ceil(float(block_radius_m) / max(self.resolution, 1e-6))))
        yy, xx = np.ogrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
        disk = (xx * xx + yy * yy) <= (radius_px * radius_px)

        for wx, wy in blocked_world_points:
            gx, gy = self.world_to_grid(float(wx), float(wy))
            x0 = max(0, gx - radius_px)
            x1 = min(self.width - 1, gx + radius_px)
            y0 = max(0, gy - radius_px)
            y1 = min(self.height - 1, gy + radius_px)
            if x0 > x1 or y0 > y1:
                continue

            mask_x0 = radius_px - (gx - x0)
            mask_x1 = radius_px + (x1 - gx)
            mask_y0 = radius_px - (gy - y0)
            mask_y1 = radius_px + (y1 - gy)
            free_mask[y0:y1 + 1, x0:x1 + 1] &= ~disk[mask_y0:mask_y1 + 1, mask_x0:mask_x1 + 1]

        for anchor in (start, goal):
            if anchor is None:
                continue
            ax, ay = anchor
            # 原因: 即使障碍物很近，也要确保起点/终点可通行
            radius = 2  # 从1增加到2，保护5x5区域
            x0 = max(0, ax - radius)
            x1 = min(self.width - 1, ax + radius)
            y0 = max(0, ay - radius)
            y1 = min(self.height - 1, ay + radius)
            free_mask[y0:y1 + 1, x0:x1 + 1] = (self.inflated_map[y0:y1 + 1, x0:x1 + 1] <= 50)

        return free_mask

    def _build_voronoi_skeleton(self) -> np.ndarray:
        """
        基于 clearance 局部极大值构造“近似 Voronoi 骨架”。
        用于把全局路径推向通道中线，降低贴墙概率。
        """
        free = self.inflated_map <= 50
        min_clear_px = max(1.0, self.voronoi_min_clearance_m / self.resolution)
        local_max = self.clearance_map_px >= maximum_filter(self.clearance_map_px, size=3)
        skeleton = local_max & free & (self.clearance_map_px >= min_clear_px)
        return skeleton

    def _nearest_skeleton_cell(self, node: Tuple[int, int], max_radius: int = 60):
        if self.voronoi_skeleton is None:
            return None
        x0, y0 = node
        if 0 <= x0 < self.width and 0 <= y0 < self.height and self.voronoi_skeleton[y0, x0]:
            return node

        for r in range(1, max_radius + 1):
            x_min = max(0, x0 - r)
            x_max = min(self.width - 1, x0 + r)
            y_min = max(0, y0 - r)
            y_max = min(self.height - 1, y0 + r)
            cand = np.argwhere(self.voronoi_skeleton[y_min:y_max + 1, x_min:x_max + 1])
            if cand.size == 0:
                continue
            best = None
            best_d2 = float('inf')
            for cy, cx in cand:
                gx = int(cx + x_min)
                gy = int(cy + y_min)
                d2 = (gx - x0) ** 2 + (gy - y0) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best = (gx, gy)
            if best is not None:
                return best
        return None

    def _search(self, start, goal, use_mask: np.ndarray = None):
        """通用 A* 搜索；use_mask 为 True 的网格可通行。"""
        if use_mask is not None:
            def node_ok(nx, ny):
                return (0 <= nx < self.width and 0 <= ny < self.height and use_mask[ny, nx])
        else:
            def node_ok(nx, ny):
                return self.is_valid(nx, ny)

        if not node_ok(*start) or not node_ok(*goal):
            return None

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0.0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return self._reconstruct_path(came_from, current)

            cx, cy = current
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                nx, ny = cx + dx, cy + dy
                if not node_ok(nx, ny):
                    continue
                if dx != 0 and dy != 0:
                    if not (node_ok(cx + dx, cy) and node_ok(cx, cy + dy)):
                        continue

                base = 1.414 if dx != 0 and dy != 0 else 1.0
                # 轻微偏好通道中线，避免 A* 在同等路径长度下贴墙/贴障碍。
                clearance_bonus = 0.0
                if self.clearance_cost_weight > 0.0:
                    clearance_m = float(self.clearance_map_m[ny, nx])
                    if clearance_m < self.preferred_clearance_m:
                        clearance_bonus = self.clearance_cost_weight * (
                            (self.preferred_clearance_m - clearance_m)
                            / max(self.preferred_clearance_m, 1e-6)
                        )
                tentative_g = g_score[current] + base + clearance_bonus

                nb = (nx, ny)
                if nb not in g_score or tentative_g < g_score[nb]:
                    came_from[nb] = current
                    g_score[nb] = tentative_g
                    f = tentative_g + self.heuristic(nb, goal)
                    heapq.heappush(open_set, (f, nb))

        return None

    def _plan_voronoi(self, start, goal):
        if self.voronoi_skeleton is None:
            return None

        s_v = self._nearest_skeleton_cell(start)
        g_v = self._nearest_skeleton_cell(goal)
        if s_v is None or g_v is None:
            return None

        p1 = self._search(start, s_v)
        p2 = self._search(s_v, g_v, use_mask=self.voronoi_skeleton)
        p3 = self._search(g_v, goal)
        if p1 is None or p2 is None or p3 is None:
            return None

        merged = p1[:-1] + p2[:-1] + p3
        dedup = [merged[0]]
        for node in merged[1:]:
            if node != dedup[-1]:
                dedup.append(node)
        return dedup
    
    def is_valid(self, grid_x, grid_y):
        if grid_x < 0 or grid_x >= self.width or grid_y < 0 or grid_y >= self.height:
            return False
        if self.inflated_map[grid_y, grid_x] > 50:
            return False
        return True
    
    def heuristic(self, a, b):
        # 使用对角距离启发式，比欧氏距离在栅格中更好
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)
    
    def get_neighbors(self, node):
        x, y = node
        neighbors = []
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
            nx, ny = x + dx, y + dy
            if dx != 0 and dy != 0:
                if not (self.is_valid(x + dx, y) and self.is_valid(x, y + dy)):
                    continue
            if self.is_valid(nx, ny):
                cost = 1.414 if dx != 0 and dy != 0 else 1.0
                neighbors.append(((nx, ny), cost))
        return neighbors

    def check_line_of_sight(
        self,
        start_grid,
        end_grid,
        free_mask: Optional[np.ndarray] = None,
        min_clearance_m: float = 0.0,
    ):
        """Conservative grid line check: every crossed cell must be safe."""
        if free_mask is not None:
            def cell_ok(cx, cy):
                return 0 <= cx < self.width and 0 <= cy < self.height and bool(free_mask[cy, cx])
        else:
            cell_ok = self.is_valid

        x0, y0 = start_grid
        x1, y1 = end_grid
        cells = list(self._bresenham_cells(x0, y0, x1, y1))

        prev_cell = None
        for idx, (cell_x, cell_y) in enumerate(cells):
            if not cell_ok(cell_x, cell_y):
                return False
            if (
                min_clearance_m > 0.0
                and idx not in (0, len(cells) - 1)
                and self.clearance_map_m[cell_y, cell_x] < min_clearance_m
            ):
                return False

            if prev_cell is not None:
                px, py = prev_cell
                if cell_x != px and cell_y != py:
                    if not (cell_ok(cell_x, py) and cell_ok(px, cell_y)):
                        return False
                    if min_clearance_m > 0.0:
                        if self.clearance_map_m[py, cell_x] < min_clearance_m:
                            return False
                        if self.clearance_map_m[cell_y, px] < min_clearance_m:
                            return False

            prev_cell = (cell_x, cell_y)

        return True

    @staticmethod
    def _bresenham_cells(x0: int, y0: int, x1: int, y1: int):
        """Yield cells on a Bresenham line between two grid coordinates."""
        x0 = int(x0)
        y0 = int(y0)
        x1 = int(x1)
        y1 = int(y1)
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x = x0
        y = y0
        while True:
            yield x, y
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    def validate_world_path(
        self,
        path: List[Tuple[float, float]],
        free_mask: Optional[np.ndarray] = None,
        min_clearance_m: float = 0.0,
    ) -> bool:
        if not path:
            return False
        if len(path) == 1:
            gx, gy = self.world_to_grid(*path[0])
            return self.check_line_of_sight((gx, gy), (gx, gy), free_mask=free_mask)

        for a, b in zip(path[:-1], path[1:]):
            ag = self.world_to_grid(*a)
            bg = self.world_to_grid(*b)
            if not self.check_line_of_sight(ag, bg, free_mask=free_mask, min_clearance_m=min_clearance_m):
                return False
        return True

    def path_min_clearance_m(self, path: List[Tuple[float, float]]) -> float:
        """Return the minimum static-map clearance along a world-coordinate path."""
        if not path:
            return float("nan")
        values = []
        for point in path:
            values.append(self.get_clearance_m(float(point[0]), float(point[1])))
        for a, b in zip(path[:-1], path[1:]):
            ag = self.world_to_grid(*a)
            bg = self.world_to_grid(*b)
            for gx, gy in self._bresenham_cells(ag[0], ag[1], bg[0], bg[1]):
                if 0 <= gx < self.width and 0 <= gy < self.height:
                    values.append(float(self.clearance_map_m[gy, gx]))
        return min(values) if values else float("nan")

    def validate_grid_path(
        self,
        path: List[Tuple[int, int]],
        free_mask: Optional[np.ndarray] = None,
        min_clearance_m: float = 0.0,
    ) -> bool:
        if not path:
            return False
        if len(path) == 1:
            x, y = path[0]
            if free_mask is not None:
                return 0 <= x < self.width and 0 <= y < self.height and bool(free_mask[y, x])
            return self.is_valid(x, y)
        for a, b in zip(path[:-1], path[1:]):
            if not self.check_line_of_sight(a, b, free_mask=free_mask, min_clearance_m=min_clearance_m):
                return False
        return True

    def plan(self, start_pos, goal_pos):
        start = self.world_to_grid(*start_pos)
        goal = self.world_to_grid(*goal_pos)

        if not self.is_valid(*goal):
            goal = self._find_nearest_valid_cell(goal)
            if goal is None:
                return None

        if not self.is_valid(*start):
            start = self._find_nearest_valid_cell(start)
            if start is None:
                return None

        raw_path = None
        if self.use_voronoi:
            raw_path = self._plan_voronoi(start, goal)

        if raw_path is None:
            raw_path = self._search(start, goal)
        if raw_path is None:
            return None

        # 关键步骤：路径平滑/剪枝。剪枝后必须再次做线段级校验；
        # 不安全则回退 raw A* 栅格路径，禁止发布穿墙/贴角直线。
        pruned_path = self._prune_path(raw_path)
        prune_clearance = self.prune_min_clearance_m if self.clearance_cost_weight > 0.0 else 0.0
        if not self.validate_grid_path(pruned_path, min_clearance_m=prune_clearance):
            pruned_path = raw_path
        if not self.validate_grid_path(pruned_path, min_clearance_m=0.0):
            return None
        return [self.grid_to_world(x, y) for x, y in pruned_path]

    def plan_with_dynamic_obstacles(
        self,
        start_pos,
        goal_pos,
        blocked_world_points: Optional[List[Tuple[float, float]]] = None,
        block_radius_m: float = 0.45,
    ):
        blocked_world_points = blocked_world_points or []

        # 边界检查和障碍物处理
        start = self.world_to_grid(*start_pos)
        goal = self.world_to_grid(*goal_pos)

        # 检查是否超出地图范围
        if not (0 <= start[0] < self.width and 0 <= start[1] < self.height):
            print(f" [AStarPlanner] 起点 {start_pos} → grid{start} 超出地图边界 "
                  f"(地图尺寸: {self.width}×{self.height})", flush=True)
            return None

        if not (0 <= goal[0] < self.width and 0 <= goal[1] < self.height):
            print(f" [AStarPlanner] 终点 {goal_pos} → grid{goal} 超出地图边界 "
                  f"(地图尺寸: {self.width}×{self.height})", flush=True)
            return None

        # 检查起点/终点是否在静态障碍物中（使用原始obstacle_mask，不是inflated）
        if self.obstacle_mask[start[1], start[0]] > 0:
            print(f"  [AStarPlanner] 起点在静态障碍物中！尝试就近找自由点...", flush=True)
            new_start = self._find_nearest_free_cell(start)
            if new_start is None:
                print(f" [AStarPlanner] 无法为起点找到附近的自由空间", flush=True)
                return None
            start = new_start

        if self.obstacle_mask[goal[1], goal[0]] > 0:
            print(f"  [AStarPlanner] 终点在静态障碍物中！尝试就近找自由点...", flush=True)
            new_goal = self._find_nearest_free_cell(goal)
            if new_goal is None:
                print(f" [AStarPlanner] 无法为终点找到附近的自由空间", flush=True)
                return None
            goal = new_goal

        if not blocked_world_points:
            return self.plan(start_pos, goal_pos)

        # A robot can be outside the inflated free mask while remaining outside
        # the raw obstacle cell. Project such endpoints to the nearest safe cell
        # so replanning can guide it away instead of repeatedly returning None.
        start_valid = self.is_valid(*start)
        goal_valid = self.is_valid(*goal)

        if not start_valid:
            projected_start = self._find_nearest_valid_cell(start)
            if projected_start is not None:
                print(
                    f"   ↪ 起点投影到膨胀地图自由点: {start} → {projected_start}",
                    flush=True,
                )
                start = projected_start
                start_valid = True
        if not goal_valid:
            projected_goal = self._find_nearest_valid_cell(goal)
            if projected_goal is not None:
                print(
                    f"   ↪ 终点投影到膨胀地图自由点: {goal} → {projected_goal}",
                    flush=True,
                )
                goal = projected_goal
                goal_valid = True

        if not start_valid or not goal_valid:
            print(f"  [AStarPlanner] 规划失败: start_pos={start_pos} → grid={start} valid={start_valid}, "
                  f"goal_pos={goal_pos} → grid={goal} valid={goal_valid}", flush=True)
            if not start_valid:
                print(f"    起点网格({start[0]}, {start[1]}) 越界或在障碍物中", flush=True)
            if not goal_valid:
                print(f"    终点网格({goal[0]}, {goal[1]}) 越界或在障碍物中", flush=True)
            return None

        free_mask = self._build_dynamic_free_mask(
            blocked_world_points=blocked_world_points,
            block_radius_m=block_radius_m,
            start=start,
            goal=goal,
        )
        raw_path = self._search(start, goal, use_mask=free_mask)
        if raw_path is None:
            print(f"  [AStarPlanner] A*搜索失败: 无法找到从 {start} 到 {goal} 的路径", flush=True)
            print(f"    可能原因: 障碍物完全阻挡了路径，或终点被膨胀障碍物覆盖", flush=True)
            print(f"    Start世界坐标: {start_pos}, Goal世界坐标: {goal_pos}", flush=True)
            print(f"    blocked_world_points示例: {blocked_world_points[:3] if len(blocked_world_points) > 0 else '空'}", flush=True)
            return None
        pruned_path = self._prune_path(raw_path, free_mask=free_mask)
        prune_clearance = self.prune_min_clearance_m if self.clearance_cost_weight > 0.0 else 0.0
        if not self.validate_grid_path(pruned_path, free_mask=free_mask, min_clearance_m=prune_clearance):
            pruned_path = raw_path
        if not self.validate_grid_path(pruned_path, free_mask=free_mask, min_clearance_m=0.0):
            print("  [AStarPlanner] 路径线段级校验失败，拒绝发布不安全路径", flush=True)
            return None
        return [self.grid_to_world(x, y) for x, y in pruned_path]

    def _find_nearest_valid_cell(
        self,
        cell: Tuple[int, int],
        max_radius: int = 12,
    ) -> Optional[Tuple[int, int]]:
        """Find the nearest cell outside the planner's inflated obstacle mask."""
        from collections import deque

        x0, y0 = cell
        if self.is_valid(x0, y0):
            return cell
        if not (0 <= x0 < self.width and 0 <= y0 < self.height):
            return None

        queue = deque([(x0, y0)])
        visited = {(x0, y0)}
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                node = (nx, ny)
                if node in visited or not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if abs(nx - x0) + abs(ny - y0) > max_radius:
                    continue
                visited.add(node)
                if self.is_valid(nx, ny):
                    return node
                queue.append(node)
        return None

    def _find_nearest_free_cell(self, cell: Tuple[int, int], max_radius: int = 10) -> Optional[Tuple[int, int]]:
        """
        在给定cell周围搜索最近的自由空间（BFS）

        用于处理起点/终点spawn在障碍物中的情况

        Args:
            cell: (grid_x, grid_y)
            max_radius: 最大搜索半径（像素）

        Returns:
            最近的自由cell，或None
        """
        x0, y0 = cell
        if 0 <= x0 < self.width and 0 <= y0 < self.height and self.obstacle_mask[y0, x0] == 0:
            return cell

        # BFS搜索
        from collections import deque
        queue = deque([cell])
        visited = {cell}

        while queue:
            x, y = queue.popleft()

            # 检查8邻域
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nx, ny = x + dx, y + dy

                if (nx, ny) in visited:
                    continue

                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue

                # 距离检查（曼哈顿距离）
                dist = abs(nx - x0) + abs(ny - y0)
                if dist > max_radius:
                    continue

                visited.add((nx, ny))

                # 找到自由空间
                if self.obstacle_mask[ny, nx] == 0:
                    print(f"    找到自由点: 从({x0},{y0}) → ({nx},{ny}), 距离={dist}px", flush=True)
                    return (nx, ny)

                queue.append((nx, ny))

        return None

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def _prune_path(self, path, free_mask: Optional[np.ndarray] = None):
        """
        使用视线检查（Line of Sight）去除多余的路径点
        原理：如果 path[i] 能直接看到 path[i+2]，则 path[i+1] 是多余的
        """
        if len(path) < 3: return path
        
        pruned = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            next_idx = current_idx + 1
            max_jump_cells = 6 if self.clearance_cost_weight > 0.0 else 12
            max_jump = min(len(path) - 1, current_idx + max_jump_cells)
            for i in range(max_jump, current_idx, -1):
                min_clearance = self.prune_min_clearance_m if self.clearance_cost_weight > 0.0 else 0.0
                if self.check_line_of_sight(
                    path[current_idx],
                    path[i],
                    free_mask=free_mask,
                    min_clearance_m=min_clearance,
                ):
                    next_idx = i
                    break
            
            pruned.append(path[next_idx])
            current_idx = next_idx
            
        return pruned

class WaypointExtractor:
    """
    Extract sparse waypoints from path and apply B-spline smoothing.
    """
    def __init__(self, distance_threshold=1.2, min_clearance_m=0.45, turn_keep_angle_deg=35.0,
                 smooth_path=False, spline_resolution=0.15):
        self.distance_threshold = float(distance_threshold)
        self.min_clearance_m = float(min_clearance_m)
        self.turn_keep_angle_deg = float(turn_keep_angle_deg)
        self.smooth_path = bool(smooth_path)
        self.spline_resolution = float(spline_resolution)  # meters between interpolated points

    @staticmethod
    def _turn_angle_deg(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        v1 = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float32)
        v2 = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float32)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        cos_t = float(np.dot(v1, v2) / (n1 * n2))
        cos_t = float(np.clip(cos_t, -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_t)))
    
    def extract(self, path: List[Tuple[float, float]], planner: AStarPlanner = None) -> List[Tuple[float, float]]:
        if not path or len(path) < 2:
            return path

        if planner is not None:
            # AStarPlanner already returns a line-of-sight-pruned path. A second
            # distance-based sparsification can reconnect non-adjacent safe
            # segments and draw a straight RViz line through an obstacle.
            return path

        # 1. Preserve start
        waypoints = [path[0]]

        # 2. Distance filter + turn angle preservation
        for i in range(1, len(path) - 1):
            curr = path[i]
            prev = waypoints[-1]
            dist = math.hypot(curr[0] - prev[0], curr[1] - prev[1])

            turn_keep = False
            if i < len(path) - 1:
                turn_angle = self._turn_angle_deg(path[i - 1], curr, path[i + 1])
                turn_keep = turn_angle >= self.turn_keep_angle_deg

            if dist >= self.distance_threshold or turn_keep:
                waypoints.append(curr)

        # 3. Handle goal
        goal = path[-1]
        last_added = waypoints[-1]
        dist_to_goal = math.hypot(goal[0] - last_added[0], goal[1] - last_added[1])

        if dist_to_goal < 0.5 and len(waypoints) > 1:
            waypoints.pop()
            waypoints.append(goal)
        else:
            waypoints.append(goal)

        # 4. B-spline smoothing (if enabled and waypoints >= 4)
        if self.smooth_path and len(waypoints) >= 4:
            smoothed = self._smooth_bspline(waypoints)
            if planner is None or planner.validate_world_path(smoothed):
                waypoints = smoothed

        return waypoints

    def _smooth_bspline(self, waypoints: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Apply cubic B-spline interpolation to smooth sharp turns."""
        try:
            from scipy.interpolate import splprep, splev
        except ImportError:
            return waypoints  # fallback if scipy not available

        if len(waypoints) < 4:
            return waypoints

        # Convert to numpy
        points = np.array(waypoints, dtype=np.float64)
        x = points[:, 0]
        y = points[:, 1]

        # Compute spline (k=3 for cubic, s=0 for exact interpolation)
        # s > 0 allows smoothing; s=0 passes through all points
        try:
            tck, u = splprep([x, y], s=0.05, k=min(3, len(waypoints) - 1))
        except Exception:
            return waypoints  # fallback on error

        # Compute total arc length for uniform sampling
        path_length = 0.0
        for i in range(len(waypoints) - 1):
            path_length += math.hypot(waypoints[i+1][0] - waypoints[i][0],
                                      waypoints[i+1][1] - waypoints[i][1])

        # Sample every spline_resolution meters
        num_samples = max(2, int(path_length / self.spline_resolution))
        u_new = np.linspace(0, 1, num_samples)
        x_new, y_new = splev(u_new, tck)

        smoothed = [(float(x_new[i]), float(y_new[i])) for i in range(len(x_new))]

        # Always preserve start and goal
        smoothed[0] = waypoints[0]
        smoothed[-1] = waypoints[-1]

        return smoothed
