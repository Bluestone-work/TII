from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

Route = Tuple[Tuple[float, float], Tuple[float, float]]
RoutePlan = Dict[str, Route]


@dataclass(frozen=True)
class FixedScenarioCase:
    name: str
    category: str
    description: str
    map_number: int
    num_agents: int
    route_plan: RoutePlan


def _shift_route_plan(
    route_plan: RoutePlan,
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    mirror_x: bool = False,
    mirror_y: bool = False,
) -> RoutePlan:
    out: RoutePlan = {}
    for aid, (start_xy, goal_xy) in route_plan.items():
        sx, sy = float(start_xy[0]), float(start_xy[1])
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        if mirror_x:
            sx, gx = -sx, -gx
        if mirror_y:
            sy, gy = -sy, -gy
        out[aid] = ((sx + dx, sy + dy), (gx + dx, gy + dy))
    return out


def _case(name: str, category: str, description: str, map_number: int, route_plan: RoutePlan) -> FixedScenarioCase:
    return FixedScenarioCase(
        name=name,
        category=category,
        description=description,
        map_number=map_number,
        num_agents=len(route_plan),
        route_plan=route_plan,
    )


MAP3_H = ((-4.8, 0.0), (4.8, 0.0))
MAP3_H_UP = ((4.8, 0.6), (-4.8, 0.6))
MAP3_H_DOWN = ((-4.8, -0.6), (4.8, -0.6))

MAP6_MAIN = ((-4.6, 0.0), (4.6, 0.0))
MAP6_MAIN_REV = ((4.6, 0.0), (-4.6, 0.0))
MAP6_VERT = ((0.0, -4.8), (0.0, 4.8))
MAP6_VERT_REV = ((0.0, 4.8), (0.0, -4.8))
MAP6_UP = ((-4.2, 1.4), (4.2, 1.4))
MAP6_UP_REV = ((4.2, 1.4), (-4.2, 1.4))
MAP6_DOWN = ((-4.2, -1.4), (4.2, -1.4))
MAP6_DOWN_REV = ((4.2, -1.4), (-4.2, -1.4))
MAP6_DIAG_A = ((-4.8, 0.8), (4.8, -0.8))
MAP6_DIAG_B = ((4.8, 0.8), (-4.8, -0.8))
MAP6_DIAG_C = ((-3.8, 2.4), (3.8, -2.4))
MAP6_DIAG_D = ((3.8, 2.4), (-3.8, -2.4))
MAP6_DIAG_E = ((-2.6, -3.8), (2.6, 3.8))
MAP6_DIAG_F = ((2.6, -3.8), (-2.6, 3.8))


def _plan(*routes: Route) -> RoutePlan:
    return {f"agent_{idx}": route for idx, route in enumerate(routes)}


_FIXED_SCENARIOS: List[FixedScenarioCase] = [
    _case("weak_follow_01", "weak_interaction", "单机主通道直行。", 6, _plan(MAP6_MAIN)),
    _case("weak_follow_02", "weak_interaction", "单机纵向穿行。", 6, _plan(MAP6_VERT)),
    _case("weak_follow_03", "weak_interaction", "单机右上对角。", 6, _plan(MAP6_DIAG_B)),
    _case("weak_follow_04", "weak_interaction", "单机左上对角。", 6, _plan(MAP6_DIAG_A)),
    _case("weak_follow_05", "weak_interaction", "单机纵向反向穿行。", 6, _plan(MAP6_VERT_REV)),
    _case("weak_follow_06", "weak_interaction", "单机主通道反向直行。", 6, _plan(MAP6_MAIN_REV)),
    _case("weak_follow_07", "weak_interaction", "单机右上远距斜穿。", 6, _plan(MAP6_DIAG_D)),
    _case("weak_follow_08", "weak_interaction", "单机左下远距斜穿。", 6, _plan(MAP6_DIAG_E)),
    _case("corner_l_01", "corner_turn", "对角切入主通道。", 6, _plan(MAP6_DIAG_A)),
    _case("corner_l_02", "corner_turn", "镜像对角切入主通道。", 6, _plan(MAP6_DIAG_B)),
    _case("corner_l_03", "corner_turn", "远距斜切穿越。", 6, _plan(MAP6_DIAG_D)),
    _case("corner_l_04", "corner_turn", "远距斜切镜像。", 6, _plan(MAP6_DIAG_E)),
    _case("corner_l_05", "corner_turn", "主通道切入纵向。", 6, _plan(MAP6_MAIN)),
    _case("corner_l_06", "corner_turn", "纵向切入主通道。", 6, _plan(MAP6_VERT)),
    _case("wall_close_01", "wall_margin", "窄通道水平直行。", 3, _plan(MAP3_H)),
    _case("wall_close_02", "wall_margin", "上侧贴墙直行。", 3, _plan(MAP3_H_UP)),
    _case("wall_close_03", "wall_margin", "下侧贴墙直行。", 3, _plan(MAP3_H_DOWN)),
    _case("wall_close_04", "wall_margin", "窄通道反向直行。", 3, _plan(((4.8, 0.0), (-4.8, 0.0)))),
    _case("wall_close_05", "wall_margin", "对角穿越。", 3, _plan(((-4.5, 1.8), (4.5, -1.8)))),
    _case("head_on_01", "strong_interaction", "双机正中会车。", 6, _plan(MAP6_MAIN, MAP6_MAIN_REV)),
    _case("head_on_02", "strong_interaction", "双机纵向会车。", 6, _plan(MAP6_VERT, MAP6_VERT_REV)),
    _case("head_on_03", "strong_interaction", "双机对角会车。", 6, _plan(MAP6_DIAG_A, MAP6_DIAG_B)),
    _case("head_on_04", "strong_interaction", "双机远距斜向会车。", 6, _plan(MAP6_DIAG_D, MAP6_DIAG_E)),
    _case("head_on_05", "strong_interaction", "主通道与纵向反向交错。", 6, _plan(MAP6_MAIN, MAP6_VERT_REV)),
    _case("cross_01", "crossing", "主通道与纵向穿越。", 6, _plan(MAP6_MAIN, MAP6_VERT)),
    _case("cross_02", "crossing", "反向主通道与反向纵向穿越。", 6, _plan(MAP6_MAIN_REV, MAP6_VERT_REV)),
    _case("cross_03", "crossing", "主通道与纵向反向穿越。", 6, _plan(MAP6_MAIN, MAP6_VERT_REV)),
    _case("cross_04", "crossing", "反向主通道与纵向穿越。", 6, _plan(MAP6_MAIN_REV, MAP6_VERT)),
    _case("cross_05", "crossing", "双斜线交叉。", 6, _plan(MAP6_DIAG_A, MAP6_DIAG_F)),
    _case("merge_01", "merge_conflict", "斜切并入主通道。", 6, _plan(MAP6_DIAG_A, MAP6_MAIN)),
    _case("merge_02", "merge_conflict", "纵向并入主通道。", 6, _plan(MAP6_MAIN, MAP6_VERT)),
    _case("merge_03", "merge_conflict", "反向主通道与纵向汇流。", 6, _plan(MAP6_MAIN_REV, MAP6_VERT)),
    _case("merge_04", "merge_conflict", "斜切并入反向主通道。", 6, _plan(MAP6_DIAG_F, MAP6_MAIN_REV)),
    _case("rear_pressure_01", "rear_pressure", "同向主通道跟进。", 6, _plan(MAP6_MAIN, MAP6_DIAG_A)),
    _case("rear_pressure_02", "rear_pressure", "同向纵向跟进。", 6, _plan(MAP6_VERT, MAP6_DIAG_E)),
    _case("rear_pressure_03", "rear_pressure", "反向主通道跟进压迫。", 6, _plan(MAP6_MAIN_REV, MAP6_DIAG_B)),
    _case("rear_pressure_04", "rear_pressure", "斜切后方压迫。", 6, _plan(MAP6_DIAG_A, MAP6_MAIN_REV)),
    _case("three_agent_01", "multi_agent", "主通道十字交汇。", 6, _plan(MAP6_MAIN, MAP6_VERT, MAP6_MAIN_REV)),
    _case("three_agent_02", "multi_agent", "主通道双纵向交汇。", 6, _plan(MAP6_MAIN, MAP6_VERT_REV, MAP6_VERT)),
    _case("three_agent_03", "multi_agent", "双机对冲加纵向穿越。", 6, _plan(MAP6_MAIN, MAP6_MAIN_REV, MAP6_VERT)),
    _case("three_agent_04", "multi_agent", "斜切汇入主通道。", 6, _plan(MAP6_DIAG_A, MAP6_MAIN, MAP6_VERT)),
    _case("three_agent_05", "multi_agent", "窄通道三机交汇。", 3, _plan(MAP3_H, MAP3_H_UP, MAP3_H_DOWN)),
    _case("three_agent_06", "multi_agent", "双斜线加主通道混合。", 6, _plan(MAP6_DIAG_A, MAP6_DIAG_F, MAP6_MAIN_REV)),
    _case("deadlock_01", "deadlock_like", "主通道中段对峙。", 6, _plan(MAP6_MAIN, MAP6_MAIN_REV)),
    _case("deadlock_02", "deadlock_like", "纵向通道对峙。", 6, _plan(MAP6_VERT, MAP6_VERT_REV)),
    _case("deadlock_03", "deadlock_like", "窄通道对峙。", 3, _plan(MAP3_H, ((4.8, 0.0), (-4.8, 0.0)))),
    _case("deadlock_04", "deadlock_like", "窄通道偏置对峙。", 3, _plan(((-4.8, 0.6), (4.8, 0.6)), ((4.8, -0.6), (-4.8, -0.6)))),
    _case("deadlock_05", "deadlock_like", "双对角近距对峙。", 6, _plan(MAP6_DIAG_A, MAP6_DIAG_F)),
    _case("deadlock_06", "deadlock_like", "双向加纵向僵持。", 6, _plan(MAP6_MAIN, MAP6_VERT_REV, MAP6_VERT)),
]


FIXED10_CLASSIC_SCENARIOS: List[FixedScenarioCase] = [
    _case("classic_weak_main", "weak_interaction", "单机主通道直行。", 6, _plan(MAP6_MAIN)),
    _case("classic_weak_diag", "weak_interaction", "单机远距斜穿。", 6, _plan(MAP6_DIAG_D)),
    _case("classic_wall_margin", "wall_margin", "窄通道贴墙直行。", 3, _plan(MAP3_H_UP)),
    _case("classic_head_on_main", "strong_interaction", "双机主通道会车。", 6, _plan(MAP6_MAIN, MAP6_MAIN_REV)),
    _case("classic_head_on_diag", "strong_interaction", "双机远距斜向会车。", 6, _plan(MAP6_DIAG_D, MAP6_DIAG_E)),
    _case("classic_cross", "crossing", "主通道与纵向穿越。", 6, _plan(MAP6_MAIN, MAP6_VERT)),
    _case("classic_merge", "merge_conflict", "斜切并入主通道。", 6, _plan(MAP6_DIAG_A, MAP6_MAIN)),
    _case("classic_rear_pressure", "rear_pressure", "同向主通道跟进。", 6, _plan(MAP6_MAIN, MAP6_DIAG_A)),
    _case("classic_three_agent", "multi_agent", "主通道十字交汇。", 6, _plan(MAP6_MAIN, MAP6_VERT, MAP6_MAIN_REV)),
    _case("classic_deadlock", "deadlock_like", "窄通道对峙。", 3, _plan(MAP3_H, ((4.8, 0.0), (-4.8, 0.0)))),
]


PAPER_CORE4_SCENARIOS: List[FixedScenarioCase] = [
    _case("paper_head_on_main", "head_on", "主通道双向会车。", 6, _plan(MAP6_MAIN, MAP6_MAIN_REV)),
    _case("paper_cross_main_vert", "crossing", "标准正交十字穿越。", 6, _plan(MAP6_MAIN, MAP6_VERT)),
    _case("paper_tmerge_vertical", "merge_tjunction", "纵向并入主通道。", 6, _plan(MAP6_MAIN_REV, MAP6_VERT)),
    _case("paper_four_agent_intersection", "four_agent", "四车十字路口同时竞争。", 6, _plan(MAP6_MAIN, MAP6_MAIN_REV, MAP6_VERT, MAP6_VERT_REV)),
]


PAPER_INTERACTION_SCENARIOS: List[FixedScenarioCase] = [
    _case("paper_head_on_main", "head_on", "主通道双向会车。", 6, _plan(MAP6_MAIN, MAP6_MAIN_REV)),
    _case("paper_head_on_vertical", "head_on", "纵向通道双向会车。", 6, _plan(MAP6_VERT, MAP6_VERT_REV)),
    _case("paper_cross_main_vert", "crossing", "标准正交十字穿越。", 6, _plan(MAP6_MAIN, MAP6_VERT)),
    _case("paper_cross_reverse_order", "crossing", "反向到达时序的十字穿越。", 6, _plan(MAP6_MAIN_REV, MAP6_VERT)),
    _case("paper_cross_diagonal", "crossing", "对角 crossing。", 6, _plan(MAP6_DIAG_A, MAP6_DIAG_F)),
    _case("paper_tmerge_diagonal", "merge_tjunction", "对角并入主通道。", 6, _plan(MAP6_DIAG_A, MAP6_MAIN)),
    _case("paper_tmerge_vertical", "merge_tjunction", "纵向并入主通道。", 6, _plan(MAP6_MAIN_REV, MAP6_VERT)),
    _case("paper_three_agent_cross_headon", "three_agent", "主通道对冲加纵向穿越。", 6, _plan(MAP6_MAIN, MAP6_VERT, MAP6_MAIN_REV)),
    _case("paper_three_agent_bivert", "three_agent", "水平主通道加双纵向压力。", 6, _plan(MAP6_MAIN, MAP6_VERT_REV, MAP6_VERT)),
    _case("paper_four_agent_intersection", "four_agent", "四车十字路口同时竞争。", 6, _plan(MAP6_MAIN, MAP6_MAIN_REV, MAP6_VERT, MAP6_VERT_REV)),
    _case("paper_deadlock_narrow", "deadlock_like", "窄通道双车对峙。", 3, _plan(MAP3_H, ((4.8, 0.0), (-4.8, 0.0)))),
    _case("paper_deadlock_threeway", "deadlock_like", "三车中心僵持。", 6, _plan(MAP6_MAIN, MAP6_VERT_REV, MAP6_VERT)),
]


CORNER_CURRICULUM_SCENARIOS: List[FixedScenarioCase] = [
    _case("corner_outer_slot_left", "corner_curriculum", "外侧槽位左转脱困。", 6, _plan(((-4.8, 0.8), (-0.8, -2.6)))),
    _case("corner_outer_slot_right", "corner_curriculum", "外侧槽位右转脱困。", 6, _plan(((4.8, 0.8), (0.8, -2.6)))),
    _case("corner_wall_exit_left", "corner_curriculum", "贴墙左转驶出窄口。", 6, _plan(((-4.6, 1.6), (0.6, -2.2)))),
    _case("corner_wall_exit_right", "corner_curriculum", "贴墙右转驶出窄口。", 6, _plan(((4.6, 1.6), (-0.6, -2.2)))),
    _case("corner_narrow_u_turn_left", "corner_curriculum", "窄口左向掉头回正。", 3, _plan(((-4.6, 0.6), (3.8, -0.6)))),
    _case("corner_narrow_u_turn_right", "corner_curriculum", "窄口右向掉头回正。", 3, _plan(((4.6, 0.6), (-3.8, -0.6)))),
    _case("corner_main_cut_left", "corner_curriculum", "主通道切角左转。", 6, _plan(((-4.8, -0.4), (1.2, 3.6)))),
    _case("corner_main_cut_right", "corner_curriculum", "主通道切角右转。", 6, _plan(((4.8, -0.4), (-1.2, 3.6)))),
    _case("corner_slot_two_agent", "corner_curriculum", "槽位转弯并伴随主通道压力。", 6, _plan(((-4.8, 0.8), (-0.8, -2.6)), MAP6_MAIN_REV)),
    _case("corner_cross_pressure", "corner_curriculum", "切角转弯伴随纵向穿越。", 6, _plan(((4.6, 1.6), (-0.6, -2.2)), MAP6_VERT)),
    _case("corner_three_agent_mix", "corner_curriculum", "拐角脱困叠加双机交汇。", 6, _plan(((-4.8, 0.8), (-0.8, -2.6)), MAP6_MAIN_REV, MAP6_VERT)),
    _case("corner_deadlock_release", "corner_curriculum", "窄口拐角对峙脱困。", 3, _plan(((-4.6, 0.6), (3.8, -0.6)), ((4.6, -0.6), (-3.8, 0.6)))),
]


FIXED_BENCHMARK_SCENARIOS: Dict[str, List[FixedScenarioCase]] = {
    "fixed50_v1": _FIXED_SCENARIOS,
    "fixed10_classic_v1": FIXED10_CLASSIC_SCENARIOS,
    "paper_interaction_v1": PAPER_INTERACTION_SCENARIOS,
    "paper_core4_v1": PAPER_CORE4_SCENARIOS,
    "corner_curriculum_v1": CORNER_CURRICULUM_SCENARIOS,
}


def list_fixed_benchmark_sets() -> List[str]:
    return sorted(FIXED_BENCHMARK_SCENARIOS.keys())


def get_fixed_benchmark_cases(name: str) -> List[FixedScenarioCase]:
    key = str(name).strip()
    if key not in FIXED_BENCHMARK_SCENARIOS:
        raise KeyError(f"Unknown fixed benchmark set: {name}")
    return list(FIXED_BENCHMARK_SCENARIOS[key])
