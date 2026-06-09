from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


RoutePlan = Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]


@dataclass(frozen=True)
class OptionTestScenario:
    name: str
    description: str
    map_number: int
    num_agents: int
    route_plan: RoutePlan
    ego_agent_id: str = "agent_0"
    background_option_by_agent: Dict[str, str] = field(default_factory=dict)
    start_yaws: Dict[str, float] = field(default_factory=dict)
    env_overrides: Dict[str, float | int | bool | str] = field(default_factory=dict)


SCENARIOS: Dict[str, OptionTestScenario] = {
    "single_follow": OptionTestScenario(
        name="single_follow",
        description="单机器人沿主通道跟随全局路径。",
        map_number=6,
        num_agents=1,
        route_plan={
            "agent_0": ((-4.6, 0.0), (4.6, 0.0)),
        },
    ),
    "l_corner": OptionTestScenario(
        name="l_corner",
        description="L 型转角，测试 nominal follow 与 detour 的起始可行性。",
        map_number=4,
        num_agents=1,
        route_plan={
            "agent_0": ((-3.2, 0.6), (0.0, -4.0)),
        },
    ),
    "narrow_corridor": OptionTestScenario(
        name="narrow_corridor",
        description="窄通道直行，测试左右 detour 是否本身就不可执行。",
        map_number=5,
        num_agents=1,
        route_plan={
            "agent_0": ((-5.0, -2.8), (5.0, -2.8)),
        },
    ),
    "static_blocker_on_path": OptionTestScenario(
        name="static_blocker_on_path",
        description="路径前方静态占位机器人，测试 stop/backoff/detour 可行性。",
        map_number=6,
        num_agents=2,
        route_plan={
            "agent_0": ((-4.6, 0.0), (4.6, 0.0)),
            "agent_1": ((0.8, 0.0), (1.2, 0.0)),
        },
        background_option_by_agent={
            "agent_1": "stop_wait",
        },
    ),
    "head_on_two_agents": OptionTestScenario(
        name="head_on_two_agents",
        description="双机器人迎面会车，测试让行、后退和左右绕行。",
        map_number=6,
        num_agents=2,
        route_plan={
            "agent_0": ((-4.8, 0.0), (4.8, 0.0)),
            "agent_1": ((4.8, 0.0), (-4.8, 0.0)),
        },
        background_option_by_agent={
            "agent_1": "follow_path",
        },
    ),
    "intersection_crossing": OptionTestScenario(
        name="intersection_crossing",
        description="十字路口交叉穿越，测试 follow/wait/detour 的局部交互。",
        map_number=4,
        num_agents=2,
        route_plan={
            "agent_0": ((-4.2, 0.0), (4.2, 0.0)),
            "agent_1": ((0.0, -4.2), (0.0, 4.2)),
        },
        background_option_by_agent={
            "agent_1": "follow_path",
        },
    ),
    "left_wall_close": OptionTestScenario(
        name="left_wall_close",
        description="左侧贴墙，测试 detour_left 是否应被 mask。",
        map_number=3,
        num_agents=1,
        route_plan={
            "agent_0": ((-4.8, 0.6), (4.8, 0.6)),
        },
    ),
    "right_wall_close": OptionTestScenario(
        name="right_wall_close",
        description="右侧贴墙，测试 detour_right 是否应被 mask。",
        map_number=3,
        num_agents=1,
        route_plan={
            "agent_0": ((-4.8, -0.6), (4.8, -0.6)),
        },
    ),
    "both_sides_narrow": OptionTestScenario(
        name="both_sides_narrow",
        description="双侧都窄，测试 detour 双侧与 backoff/replan 边界。",
        map_number=3,
        num_agents=1,
        route_plan={
            "agent_0": ((-4.8, 0.0), (4.8, 0.0)),
        },
    ),
    "rear_blocked": OptionTestScenario(
        name="rear_blocked",
        description="后方有占位机器人，测试 backoff feasibility 与失败归因。",
        map_number=6,
        num_agents=2,
        route_plan={
            "agent_0": ((-1.0, 0.0), (4.6, 0.0)),
            "agent_1": ((-2.0, 0.0), (2.0, 0.0)),
        },
        background_option_by_agent={
            "agent_1": "stop_wait",
        },
    ),
    "deadlock_like": OptionTestScenario(
        name="deadlock_like",
        description="近似僵局：对向通道占据且前方近距，测试 wait/backoff/replan。",
        map_number=6,
        num_agents=2,
        route_plan={
            "agent_0": ((-2.8, 0.0), (2.8, 0.0)),
            "agent_1": ((2.8, 0.0), (-2.8, 0.0)),
        },
        background_option_by_agent={
            "agent_1": "stop_wait",
        },
    ),
}


def get_scenario(name: str) -> OptionTestScenario:
    key = str(name).strip()
    if key not in SCENARIOS:
        raise KeyError(f"Unknown option test scenario: {name}")
    return SCENARIOS[key]


def list_scenarios() -> list[str]:
    return sorted(SCENARIOS.keys())
