"""Immutable scenario bank shared by all formal evaluation methods."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


Point = tuple[float, float]
Route = tuple[Point, Point]


@dataclass(frozen=True)
class FormalScenario:
    case_id: str
    category: str
    map_number: int
    routes: tuple[Route, ...]
    max_episode_steps: int
    num_dynamic_obstacles: int = 0
    obstacle_speed_mps: float = 0.075
    sensor_noise_std: float = 0.0
    route_mode: str = "fixed"

    @property
    def num_agents(self):
        return len(self.routes)

    def route_plan(self):
        return {
            f"agent_{index}": route
            for index, route in enumerate(self.routes)
        }

    def to_record(self):
        record = asdict(self)
        record["num_agents"] = self.num_agents
        record["route_plan"] = self.route_plan() if self.route_mode == "fixed" else {}
        return record


M6_H = ((-4.6, 0.0), (4.6, 0.0))
M6_H_REV = ((4.6, 0.0), (-4.6, 0.0))
M6_V = ((0.0, -4.8), (0.0, 4.8))
M6_V_REV = ((0.0, 4.8), (0.0, -4.8))
M6_D1 = ((-4.8, 0.8), (4.8, -0.8))
M3_H = ((-4.8, 0.0), (4.8, 0.0))
M3_H_REV = ((4.8, 0.0), (-4.8, 0.0))


PAPER_MICRO_V1 = (
    FormalScenario("head_on_main", "head_on", 6, (M6_H, M6_H_REV), 1600),
    FormalScenario("head_on_vertical", "head_on", 6, (M6_V, M6_V_REV), 1600),
    FormalScenario("merge_diagonal", "merge", 6, (M6_D1, M6_H), 1600),
    FormalScenario("merge_vertical", "merge", 6, (M6_H_REV, M6_V), 1600),
    FormalScenario("crossing_main", "crossing", 6, (M6_H, M6_V), 1600),
    FormalScenario("crossing_reverse", "crossing", 6, (M6_H_REV, M6_V_REV), 1600),
    FormalScenario(
        "bottleneck_offset",
        "bottleneck",
        3,
        (((-4.8, 0.55), (4.8, 0.55)), ((4.8, -0.55), (-4.8, -0.55))),
        2000,
    ),
    FormalScenario(
        "bottleneck_three_agent",
        "bottleneck",
        3,
        (M3_H, M3_H_REV, ((-4.2, 0.6), (4.2, 0.6))),
        2200,
    ),
    FormalScenario("deadlock_narrow", "deadlock", 3, (M3_H, M3_H_REV), 2000),
    FormalScenario(
        "deadlock_threeway",
        "deadlock",
        6,
        (M6_H, M6_V_REV, M6_V),
        2000,
    ),
)


PAIRED_NAV_V1 = (
    FormalScenario(
        "paired_map3", "paired_navigation", 3,
        (
            ((4.8, 0.2), (-4.8, 0.2)),
            ((-4.5, 1.8), (4.5, -1.8)),
            ((3.6, -2.6), (-3.6, -2.6)),
            ((3.8, 2.4), (-3.8, -2.4)),
        ),
        2000, 6, 0.055,
    ),
    FormalScenario(
        "paired_map4", "paired_navigation", 4,
        (
            ((-4.2, 0.0), (4.2, 0.0)),
            ((4.2, 0.0), (-4.2, 0.0)),
            ((0.0, -4.2), (0.0, 4.2)),
            ((0.0, 4.2), (0.0, -4.2)),
        ),
        2000, 2, 0.055,
    ),
    FormalScenario(
        "paired_map5", "paired_navigation", 5,
        (
            ((-2.7, -5.25), (-2.7, 5.25)),
            ((-2.7, 5.25), (-2.7, -5.25)),
            ((2.1, -5.25), (2.1, 5.25)),
            ((2.1, 5.25), (2.1, -5.25)),
        ),
        2000, 0, 0.055, route_mode="seeded_random",
    ),
    FormalScenario(
        "paired_map9", "paired_navigation", 9,
        (
            ((2.0, 0.0), (-2.0, 0.0)),
            ((-2.0, 0.0), (2.0, 0.0)),
            ((1.4142, 1.4142), (-1.4142, -1.4142)),
            ((-1.4142, -1.4142), (1.4142, 1.4142)),
        ),
        2000, 6, 0.055,
    ),
)


# P0 uses Map 4 and Map 9 as the shared training/evaluation domains and keeps
# Map 5 held out for structural robustness. Each case is repeated with the
# same 50 explicit episode seeds for every method and training seed.
P0_CORE_V1 = (
    PAIRED_NAV_V1[1],  # Map 4: medium interaction at the structured crossing.
    PAIRED_NAV_V1[3],  # Map 9: strong robot/obstacle interaction.
    PAIRED_NAV_V1[2],  # Map 5: held-out warehouse-aisle robustness.
)


# Low-cost P1 supplement: the Map 3 case is evaluated with the same episode
# seeds both with and without the safety filter. It is not a training domain.
P0_P1_MAP3_V1 = (
    PAIRED_NAV_V1[0],
)


SCENARIO_BANKS = {
    "paper_micro_v1": PAPER_MICRO_V1,
    "paired_nav_v1": PAIRED_NAV_V1,
    "p0_core_v1": P0_CORE_V1,
    "p0_p1_map3_v1": P0_P1_MAP3_V1,
}


def get_scenario_bank(name="paper_micro_v1"):
    try:
        return tuple(SCENARIO_BANKS[str(name)])
    except KeyError as exc:
        raise KeyError(f"unknown scenario bank {name!r}; choices={sorted(SCENARIO_BANKS)}") from exc


def get_scenario(case_id, bank="paper_micro_v1"):
    for scenario in get_scenario_bank(bank):
        if scenario.case_id == case_id:
            return scenario
    raise KeyError(f"unknown scenario case {case_id!r} in {bank!r}")


def write_scenario_manifest(path, bank="paper_micro_v1"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bank": bank,
        "cases": [scenario.to_record() for scenario in get_scenario_bank(bank)],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
