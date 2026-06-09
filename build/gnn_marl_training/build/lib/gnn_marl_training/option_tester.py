from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np

from gnn_marl_training.option_feasibility import (
    OPTION_NAMES,
    evaluate_option_feasibility,
    feasibility_to_row,
)
from gnn_marl_training.option_primitives import (
    apply_primitive_command,
    create_option_primitive,
)
from gnn_marl_training.option_test_scenarios import get_scenario, list_scenarios

if TYPE_CHECKING:
    from gnn_marl_training.gnn_marl_env import GNNMARLEnv


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def _build_env_config(spec, args) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "num_agents": int(spec.num_agents),
        "map_number": int(spec.map_number),
        "max_episode_steps": int(args.max_episode_steps),
        "communication_range": 3.5,
        "interaction_neighbor_perception_range": 3.5,
        "enable_neighbor_obs": True,
        "enable_local_map": False,
        "comm_mode": "decentralized",
        "comm_dropout_prob": 0.0,
        "comm_latency_steps": 1,
        "comm_jitter_steps": 0,
        "comm_noise_std": 0.0,
        "auto_reset_agents": False,
        "reset_on_collision_event": True,
        "collision_ends_episode": True,
        "high_conflict_mode": "off",
        "high_conflict_prob": 0.0,
        "action_mode": "interaction_mode",
        "rolling_lookahead_dist": float(args.rolling_lookahead_dist),
        "obstacle_filter_range": float(args.obstacle_filter_range),
        "obstacle_filter_fov_deg": 360.0,
        "obstacle_top_k": int(args.obstacle_top_k),
        "predictive_feature_enable": True,
        "predictive_horizon_sec": 1.2,
        "predictive_social_ttc_safe": 2.2,
        "predictive_front_ttc_safe": 1.2,
        "predictive_min_sep": 0.55,
        "predictive_social_range": 2.5,
        "predictive_social_penalty_scale": 0.17,
        "predictive_front_penalty_scale": 0.16,
        "social_proximity_risk_scale": 0.34,
        "gap_feature_enable": True,
        "yielding_enable": True,
        "yielding_soft_dist": 0.90,
        "yielding_stop_dist": 0.50,
        "yielding_hard_stop_dist": 0.30,
        "yielding_ttc": 2.4,
        "yielding_commit_steps": 5,
        "replan_on_deadlock": True,
        "replan_cooldown_steps": 25,
        "stall_global_replan_enable": True,
        "stall_global_replan_sec": 5.0,
        "dynamic_replan_neighbor_dist": 1.8,
        "dynamic_replan_ttc": 2.6,
        "dynamic_replan_block_radius": 0.55,
        "replan_fixed_cost": 0.03,
        "replan_freq_cost": 0.012,
        "replan_time_cost": 0.015,
        "replan_time_budget_sec": 0.08,
        "replan_window_steps": 80,
        "obstacle_motion_feature_enable": True,
        "obstacle_motion_top_k": 3,
        "progress_reward_scale": 0.6,
        "path_progress_reward_scale": 1.2,
        "goal_progress_reward_scale": 1.0,
        "goal_reward": 24.0,
        "collision_penalty": 20.0,
        "time_penalty": 0.0,
        "close_obstacle_penalty_scale": 0.30,
        "close_obstacle_dist": 0.55,
        "team_reward_lambda": 1.0,
    }
    config.update(dict(spec.env_overrides))
    return config


def _forced_reset(env: "GNNMARLEnv", route_plan: Mapping[str, Any], *, seed: int | None) -> Tuple[Dict, Dict]:
    original_builder = env._build_episode_route_plan
    env._build_episode_route_plan = lambda: (dict(route_plan), "option_test_forced", {"scenario_forced": True})
    try:
        return env.reset(seed=seed)
    finally:
        env._build_episode_route_plan = original_builder


def _align_agents_to_routes(env: "GNNMARLEnv", route_plan: Mapping[str, Any], start_yaws: Mapping[str, float]) -> None:
    # 第一步：强制覆写 goal_pos + 重规划 global path，再传送
    # 必须重规划，因为 agent.reset() 可能因 _is_valid_start_goal_pair 校验失败
    # 而回退到随机起终点，导致 global path 与场景 route_plan 完全不一致。
    for aid, route in route_plan.items():
        if aid not in env.agents:
            continue
        (start_xy, goal_xy) = route
        start_x, start_y = float(start_xy[0]), float(start_xy[1])
        goal_x, goal_y = float(goal_xy[0]), float(goal_xy[1])
        yaw = start_yaws.get(
            aid,
            math.atan2(goal_y - start_y, goal_x - start_x),
        )

        agent = env.agents[aid]
        agent.goal_pos = (goal_x, goal_y)
        agent.last_spawn_pos = (start_x, start_y)

        if agent.planner:
            path = agent.planner.plan((start_x, start_y), (goal_x, goal_y))
            if path:
                agent.global_waypoints = agent.waypoint_extractor.extract(path, planner=agent.planner)
                agent.current_waypoint_index = 0
                if hasattr(agent, "vis") and agent.vis:
                    agent.vis.publish_waypoints(
                        agent.global_waypoints,
                        robot_id=agent.robot_id,
                        namespace=agent.vis_namespace,
                    )
            else:
                agent.global_waypoints = [agent.goal_pos]
                agent.current_waypoint_index = 0

        agent._set_robot_pose(start_x, start_y, float(yaw))

    env._wait_and_spin_all(0.25)

    for aid, route in route_plan.items():
        if aid not in env.agents:
            continue
        agent = env.agents[aid]
        env.robot_positions[aid] = env._get_robot_position(agent)
        env.robot_velocities[aid] = env._get_robot_velocity(agent)
        current_target = agent._get_tracking_target()
        agent._publish_tracking_visuals(current_target)
        agent._obs_target_state = np.array(current_target, dtype=np.float32)
        agent.prev_target_point = tuple(current_target)
        agent.prev_dist_to_goal = math.hypot(
            float(agent.goal_pos[0]) - float(agent.current_pose["x"]),
            float(agent.goal_pos[1]) - float(agent.current_pose["y"]),
        )
        agent.prev_dist_to_target = math.hypot(
            float(current_target[0]) - float(agent.current_pose["x"]),
            float(current_target[1]) - float(agent.current_pose["y"]),
        )
        agent.prev_path_progress = float(getattr(agent, "path_progress", 0.0))
        agent.prev_abs_target_angle = abs(float(agent._get_target_angle(current_target)))
        agent._cached_step_tracking_target = None
        agent._cached_step_tracking_step = -1
    env._update_interaction_contexts(list(env.agent_ids))


def _collect_raw_step_results(env: "GNNMARLEnv", active_aids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    raw_step_results: Dict[str, Dict[str, Any]] = {}
    for aid in active_aids:
        obs, rew, done, truncated, info = env.agents[aid].get_step_result()
        pos = env._get_robot_position(env.agents[aid])
        vel = env._get_robot_velocity(env.agents[aid])
        env.robot_positions[aid] = pos
        env.robot_velocities[aid] = vel
        raw_step_results[aid] = {
            "obs": obs,
            "rew": rew,
            "done": bool(done),
            "truncated": bool(truncated),
            "info": info,
            "pos": pos,
            "vel": vel,
        }
    return raw_step_results


def _sync_pair_collisions(env: "GNNMARLEnv", raw_step_results: Dict[str, Dict[str, Any]]) -> None:
    active_aids = list(raw_step_results.keys())
    for i in range(len(active_aids)):
        for j in range(i + 1, len(active_aids)):
            ai = active_aids[i]
            aj = active_aids[j]
            info_i = raw_step_results[ai]["info"]
            info_j = raw_step_results[aj]["info"]
            i_collision = info_i.get("event") == "collision"
            j_collision = info_j.get("event") == "collision"
            if not (i_collision or j_collision):
                continue

            pair_dist = float(np.linalg.norm(raw_step_results[ai]["pos"] - raw_step_results[aj]["pos"]))
            hard_sync_dist = max(
                float(getattr(env.agents[ai], "collision_hard_dist", 0.18)),
                float(getattr(env.agents[aj], "collision_hard_dist", 0.18)),
            ) + 0.02
            if pair_dist > hard_sync_dist:
                continue

            if i_collision and not j_collision:
                info_j["event"] = "collision"
                info_j["collision_source"] = "pair_sync"
                info_j["synced_collision_with"] = ai
                raw_step_results[aj]["rew"] = float(raw_step_results[aj]["rew"]) - float(
                    getattr(env.agents[aj], "collision_penalty", 20.0)
                )
            elif j_collision and not i_collision:
                info_i["event"] = "collision"
                info_i["collision_source"] = "pair_sync"
                info_i["synced_collision_with"] = aj
                raw_step_results[ai]["rew"] = float(raw_step_results[ai]["rew"]) - float(
                    getattr(env.agents[ai], "collision_penalty", 20.0)
                )


def _stop_all_agents(env: "GNNMARLEnv") -> None:
    for agent in env.agents.values():
        agent._publish_vel(0.0, 0.0)
    env._wait_and_spin_all(0.10)


def _episode_record_from_status(
    *,
    scenario_name: str,
    option_name: str,
    episode_idx: int,
    status: Mapping[str, Any],
    initial_mask_allow: bool,
    final_event: str,
) -> Dict[str, Any]:
    row = {
        "scenario": scenario_name,
        "option_name": option_name,
        "episode": int(episode_idx),
        "initial_mask_allow": int(bool(initial_mask_allow)),
        "final_event": str(final_event),
    }
    row.update(dict(status))
    return row


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(dict(row), ensure_ascii=False, default=_json_default))
            fp.write("\n")


def _write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_npz(path: Path, rows: List[Mapping[str, Any]]) -> None:
    if not rows:
        np.savez_compressed(path, empty=np.array([], dtype=np.float32))
        return
    keys = sorted({key for row in rows for key in row.keys()})
    arrays = {}
    for key in keys:
        arrays[key] = np.array([row.get(key) for row in rows], dtype=object)
    np.savez_compressed(path, **arrays)


def _summarize_episode_rows(rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    success_flags = [bool(row.get("success", False)) for row in rows]
    failure_reasons = Counter(
        str(row.get("failure_reason", ""))
        for row in rows
        if str(row.get("failure_reason", ""))
    )
    summary = {
        "episodes": len(rows),
        "success_rate": float(np.mean(success_flags)) if rows else 0.0,
        "initial_feasible_rate": float(np.mean([bool(row.get("initial_feasible", False)) for row in rows])) if rows else 0.0,
        "mask_allow_on_start_rate": float(np.mean([bool(row.get("initial_mask_allow", False)) for row in rows])) if rows else 0.0,
        "collision_rate": float(np.mean([str(row.get("final_event", "")) == "collision" for row in rows])) if rows else 0.0,
        "mean_steps": float(np.mean([float(row.get("steps_executed", 0.0)) for row in rows])) if rows else 0.0,
        "mean_progress_gain": float(np.mean([float(row.get("progress_gain", 0.0)) for row in rows])) if rows else 0.0,
        "mean_front_clearance_gain": float(np.mean([float(row.get("front_clearance_gain", 0.0)) for row in rows])) if rows else 0.0,
        "mean_social_risk_drop": float(np.mean([float(row.get("social_risk_drop", 0.0)) for row in rows])) if rows else 0.0,
        "mean_ttc_gain": float(np.mean([float(row.get("ttc_gain", 0.0)) for row in rows])) if rows else 0.0,
        "mean_safety_override_count": float(np.mean([float(row.get("safety_override_count", 0.0)) for row in rows])) if rows else 0.0,
        "mean_emergency_override_count": float(np.mean([float(row.get("emergency_override_count", 0.0)) for row in rows])) if rows else 0.0,
        "failure_reason_counts": dict(failure_reasons),
    }
    return summary


def run_option_test(args: argparse.Namespace) -> Path:
    from gnn_marl_training.gnn_marl_env import GNNMARLEnv

    spec = get_scenario(args.scenario)
    output_root = Path(args.output_dir).expanduser().resolve()
    run_id = f"{spec.name}_{args.option}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = GNNMARLEnv(_build_env_config(spec, args))
    step_rows: List[Dict[str, Any]] = []
    episode_rows: List[Dict[str, Any]] = []

    try:
        for episode_idx in range(int(args.num_episodes)):
            seed = None if args.seed is None else int(args.seed) + episode_idx
            _forced_reset(env, spec.route_plan, seed=seed)
            _align_agents_to_routes(env, spec.route_plan, spec.start_yaws)

            terminated_agents: set[str] = set()
            ego_agent_id = str(spec.ego_agent_id)
            initial_feasibility = evaluate_option_feasibility(
                env.agents[ego_agent_id],
                option_state=args.option,
                include_replan=not args.disable_replan,
            )

            primitives = {}
            for aid in env.agent_ids:
                option_name = args.option if aid == ego_agent_id else spec.background_option_by_agent.get(
                    aid,
                    args.background_option,
                )
                primitives[aid] = create_option_primitive(
                    option_name,
                    max_steps=int(args.max_option_steps if aid == ego_agent_id else args.max_episode_steps),
                    terminate_on_success=(aid == ego_agent_id),
                    enable_safety_overrides=not args.disable_safety_overrides,
                )

            final_event = ""
            for _ in range(int(args.max_episode_steps)):
                active_aids = [aid for aid in env.agent_ids if aid not in terminated_agents]
                if not active_aids or ego_agent_id in terminated_agents:
                    break

                env.current_step_count += 1
                pre_feasibility: Dict[str, Any] = {}
                commands: Dict[str, Any] = {}

                for aid in active_aids:
                    feasibility = evaluate_option_feasibility(
                        env.agents[aid],
                        option_state=primitives[aid].option_name,
                        include_replan=not args.disable_replan,
                    )
                    pre_feasibility[aid] = feasibility
                    commands[aid] = primitives[aid].step(
                        env.agents[aid],
                        feasibility,
                        force_execute=(aid != ego_agent_id) or (not args.respect_action_mask),
                        enable_safety_overrides=not args.disable_safety_overrides,
                    )
                    apply_primitive_command(
                        env.agents[aid],
                        commands[aid],
                        global_step=int(env.current_step_count),
                    )

                for aid in env.agent_ids:
                    if aid not in active_aids:
                        env.agents[aid]._publish_vel(0.0, 0.0)

                env._wait_and_spin_all(0.10)
                for aid in env.agent_ids:
                    env.robot_positions[aid] = env._get_robot_position(env.agents[aid])
                    env.robot_velocities[aid] = env._get_robot_velocity(env.agents[aid])
                env._update_interaction_contexts(active_aids)

                raw_step_results = _collect_raw_step_results(env, active_aids)
                _sync_pair_collisions(env, raw_step_results)

                for aid in active_aids:
                    post_feasibility = evaluate_option_feasibility(
                        env.agents[aid],
                        option_state=primitives[aid].option_name,
                        include_replan=not args.disable_replan,
                    )
                    primitives[aid].observe_transition(
                        env.agents[aid],
                        raw_step_results[aid]["info"],
                        post_feasibility.local_metrics,
                        commands[aid],
                    )

                    if aid == ego_agent_id:
                        step_row: Dict[str, Any] = {
                            "scenario": spec.name,
                            "option_name": args.option,
                            "episode": int(episode_idx),
                            "global_step": int(env.current_step_count),
                            "agent_id": aid,
                            "reward": float(raw_step_results[aid]["rew"]),
                            "done": int(bool(raw_step_results[aid]["done"])),
                            "truncated": int(bool(raw_step_results[aid]["truncated"])),
                            "event": str(raw_step_results[aid]["info"].get("event", "")),
                            "option_phase": commands[aid].option_phase,
                            "policy_mode": commands[aid].policy_mode,
                            "executed_mode": commands[aid].executed_mode,
                            "cmd_linear": float(commands[aid].cmd_vel[0]),
                            "cmd_angular": float(commands[aid].cmd_vel[1]),
                            "tracking_target_x": float(commands[aid].tracking_target[0]),
                            "tracking_target_y": float(commands[aid].tracking_target[1]),
                            "nominal_target_x": float(commands[aid].nominal_target[0]),
                            "nominal_target_y": float(commands[aid].nominal_target[1]),
                            "action_mask_allow": int(bool(commands[aid].action_mask_allow)),
                            "safety_override": int(bool(commands[aid].safety_override)),
                            "emergency_override": int(bool(commands[aid].emergency_override)),
                            "option_done": int(bool(primitives[aid].done)),
                            "option_success": int(bool(primitives[aid].success)),
                            "option_failed": int(bool(primitives[aid].failed)),
                            "failure_reason": str(primitives[aid].failure_reason),
                            "progress_gain": float(primitives[aid].progress_gain),
                            "goal_distance_drop": float(primitives[aid].goal_distance_drop),
                            "front_clearance_gain": float(primitives[aid].front_clearance_gain),
                            "social_risk_drop": float(primitives[aid].social_risk_drop),
                            "ttc_gain": float(primitives[aid].ttc_gain),
                            "lateral_displacement": float(primitives[aid].lateral_displacement),
                            "backward_distance": float(primitives[aid].backward_distance),
                            "rolling_pullback_count": int(primitives[aid].rolling_pullback_count),
                            "safety_override_count": int(primitives[aid].safety_override_count),
                            "emergency_override_count": int(primitives[aid].emergency_override_count),
                            "near_miss_count": int(primitives[aid].near_miss_count),
                            "wall_scrape_count": int(primitives[aid].wall_scrape_count),
                        }
                        step_row.update(feasibility_to_row(pre_feasibility[aid], prefix="pre_"))
                        step_row.update(feasibility_to_row(post_feasibility, prefix="post_"))
                        info = raw_step_results[aid]["info"]
                        for key in (
                            "reward_total",
                            "path_tracking_reward",
                            "avoidance_reward",
                            "social_risk",
                            "social_risk_delta",
                            "clear_reward",
                            "stuck_score",
                            "replan_cost",
                            "replan_attempted",
                            "replan_success",
                            "high_level_nav_reward",
                            "high_level_interaction_reward",
                            "high_level_safety_reward",
                            "high_level_efficiency_penalty",
                            "high_level_policy_penalty",
                        ):
                            if key in info:
                                step_row[key] = info[key]
                        step_rows.append(step_row)

                    info = raw_step_results[aid]["info"]
                    event = str(info.get("event", ""))
                    if raw_step_results[aid]["done"] or raw_step_results[aid]["truncated"] or event in {"goal", "collision"}:
                        terminated_agents.add(aid)
                        if aid == ego_agent_id:
                            final_event = event or ("timeout" if raw_step_results[aid]["truncated"] else "")

                if primitives[ego_agent_id].done:
                    if not final_event:
                        final_event = (
                            primitives[ego_agent_id].success_reason
                            if primitives[ego_agent_id].success
                            else primitives[ego_agent_id].failure_reason
                        )
                    terminated_agents.add(ego_agent_id)

            _stop_all_agents(env)

            status = primitives[ego_agent_id].status()
            episode_rows.append(
                _episode_record_from_status(
                    scenario_name=spec.name,
                    option_name=args.option,
                    episode_idx=episode_idx,
                    status=status,
                    initial_mask_allow=initial_feasibility.is_feasible(args.option),
                    final_event=final_event,
                )
            )
    finally:
        _stop_all_agents(env)

    step_jsonl = run_dir / "step_records.jsonl"
    step_npz = run_dir / "step_records.npz"
    episode_jsonl = run_dir / "episode_records.jsonl"
    episode_csv = run_dir / "episode_records.csv"
    summary_json = run_dir / "summary.json"

    _write_jsonl(step_jsonl, step_rows)
    _write_npz(step_npz, step_rows)
    _write_jsonl(episode_jsonl, episode_rows)
    _write_csv(episode_csv, episode_rows)

    summary = _summarize_episode_rows(episode_rows)
    summary.update(
        {
            "scenario": spec.name,
            "option_name": args.option,
            "output_dir": str(run_dir),
        }
    )
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    print(
        f"[option_tester] scenario={spec.name} option={args.option} "
        f"episodes={summary['episodes']} success_rate={summary['success_rate']:.3f}"
    )
    print(f"[option_tester] summary={summary_json}")
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Atomic option tester / feasibility evaluator for interaction_mode branch.",
    )
    parser.add_argument("--scenario", type=str, default="single_follow", help="固定测试场景名称")
    parser.add_argument("--option", type=str, required=False, default="follow_path", choices=OPTION_NAMES, help="要测试的原子 option")
    parser.add_argument("--background-option", type=str, default="follow_path", choices=OPTION_NAMES, help="未在场景中显式指定时，背景机器人使用的 option")
    parser.add_argument("--num-episodes", type=int, default=3, help="每个场景/option 重复测试次数")
    parser.add_argument("--max-option-steps", type=int, default=14, help="ego option 的最大持续执行步数")
    parser.add_argument("--max-episode-steps", type=int, default=40, help="单次测试 episode 最大步数")
    parser.add_argument("--output-dir", type=str, default="option_test_results", help="输出目录")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--respect-action-mask", action="store_true", help="若当前 option 不可行，则不强制执行")
    parser.add_argument("--disable-safety-overrides", action="store_true", help="关闭 tester 内部 emergency/safety override，纯强制执行")
    parser.add_argument("--disable-replan", action="store_true", help="在 feasibility/action mask 中禁用 replan")
    parser.add_argument("--rolling-lookahead-dist", type=float, default=0.8, help="环境中的 rolling lookahead 距离")
    parser.add_argument("--obstacle-filter-range", type=float, default=1.5, help="局部障碍观测半径")
    parser.add_argument("--obstacle-top-k", type=int, default=9, help="Top-K 障碍编码数量")
    parser.add_argument("--list-scenarios", action="store_true", help="只列出可用场景")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_scenarios:
        for name in list_scenarios():
            print(name)
        return

    run_option_test(args)


if __name__ == "__main__":
    main()
