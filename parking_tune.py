"""Parking-only parameter tuner.

This runner starts each episode near the planner's 2-1 point instead of
replaying the full approach from the map start. It keeps the professor
simulator untouched and uses the existing student planner logic directly.

Example:
  python parking_tune.py --episodes 80 --workers 4 --targets T21
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib
import io
import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT if (ROOT / "self-parking-sim").exists() else ROOT.parent
SIM_DIR = WORKSPACE_ROOT / "self-parking-sim"
ALGO_DIR = ROOT if (ROOT / "student_planner.py").exists() else WORKSPACE_ROOT / "self-parking-user-algorithms"
RESULT_JSON = WORKSPACE_ROOT / "parking_tune_result.json"
RESULT_MD = WORKSPACE_ROOT / "parking_tune_result.md"
POLICY_JSON = WORKSPACE_ROOT / "parking_policy_candidates.json"

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["PARKING_DISABLE_TUNED_POLICY"] = "1"
sys.path.insert(0, str(SIM_DIR))
sys.path.insert(0, str(ALGO_DIR))

sim = importlib.import_module("demo_self_parking_sim")
student_planner = importlib.import_module("student_planner")


PARAM_SPACE: Dict[str, List[float]] = {
    "PARKING_MIN_ROLL_SPEED": [0.30, 0.38, 0.43, 0.50, 0.58],
    "SECOND_APPROACH_MIN_SPEED": [0.18, 0.25, 0.32, 0.40, 0.50],
    "REAR_REVERSE_SPEED": [0.24, 0.30, 0.34, 0.40, 0.48],
    "REAR_REVERSE_MIN_SPEED": [0.18, 0.24, 0.30, 0.36, 0.42],
    "PARKING_PREPARE_BRAKE": [0.20, 0.30, 0.40, 0.50],
    "PARKING_FINAL_STOP_IOU": [0.35, 0.45, 0.55, 0.65],
    "PARKING_FINAL_STOP_DISTANCE": [0.20, 0.30, 0.45, 0.65],
    "PARKING_CENTER_STOP_DISTANCE": [0.8, 1.2, 1.5, 1.8, 2.2],
}

POLICY_GROUPS = {
    "lower_front_in",
    "lower_rear_in",
    "upper_front_in",
    "upper_rear_in",
}
DEPLOY_MIN_POLICY_SCORE = 90.0


def quiet_call(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def load_map_entry(map_cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    map_cache: Dict[Any, Any] = {}
    old_cwd = Path.cwd()
    try:
        os.chdir(SIM_DIR)
        return sim.ensure_map_loaded(map_cfg, map_cache, seed=seed)
    finally:
        os.chdir(old_cwd)


def apply_policy(policy: Dict[str, float]) -> None:
    for name, value in policy.items():
        if hasattr(student_planner, name):
            setattr(student_planner, name, float(value))


def random_policy(rng: random.Random) -> Dict[str, float]:
    return {name: rng.choice(values) for name, values in PARAM_SPACE.items()}


def candidate_rank(item: Dict[str, Any]) -> Tuple[int, float, float]:
    result = str(item.get("result") or "").lower()
    score = float(item.get("score") or 0.0)
    reward = float(item.get("reward") or 0.0)
    return (1 if result == "success" else 0, score, reward)


def deployable_candidate(item: Dict[str, Any]) -> bool:
    result = str(item.get("result") or "").lower()
    try:
        score = float(item.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return result == "success" and score > DEPLOY_MIN_POLICY_SCORE


def merge_policy_candidates(policy_candidates: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    try:
        existing = json.loads(POLICY_JSON.read_text(encoding="utf-8"))
        raw_existing = existing.get("policy_candidates", {})
        if isinstance(raw_existing, dict):
            for group, item in raw_existing.items():
                if isinstance(item, dict) and deployable_candidate(item):
                    merged[str(group)] = item
    except OSError:
        pass
    except json.JSONDecodeError:
        pass

    for group, item in policy_candidates.items():
        if not deployable_candidate(item):
            continue
        old = merged.get(group)
        if old is None or candidate_rank(item) >= candidate_rank(old):
            merged[group] = item
    return dict(sorted(merged.items()))


def stage_maps(stage_filter: Optional[int]) -> List[Dict[str, Any]]:
    maps = list(sim.AVAILABLE_MAPS)
    if stage_filter is not None:
        maps = [cfg for cfg in maps if int(cfg.get("stage", 0)) == stage_filter]
    return maps


def parse_targets(raw: Optional[str]) -> Optional[List[int]]:
    if not raw:
        return None
    result: List[int] = []
    for token in raw.split(","):
        token = token.strip().upper()
        if not token:
            continue
        if token.startswith("T"):
            token = token[1:]
        result.append(int(token))
    return result


def parse_policy_groups(raw: Optional[str]) -> Optional[set[str]]:
    if not raw or raw.lower() == "all":
        return None
    groups = {token.strip().lower() for token in raw.split(",") if token.strip()}
    unknown = groups - POLICY_GROUPS
    if unknown:
        raise ValueError(f"Unknown policy group(s): {sorted(unknown)}")
    return groups


def eligible_target_indices(map_entry: Dict[str, Any]) -> List[int]:
    assets = map_entry["assets"]
    free_indices = [idx for idx, occ in enumerate(assets.occupied_idx) if not bool(occ)]
    return free_indices or list(range(len(assets.slots)))


def target_row_number(assets: Any, target_idx: int) -> int:
    centers = sorted({round(float(slot[2] + slot[3]) * 0.5, 2) for slot in assets.slots})
    if not centers:
        return 1
    target_slot = assets.slots[target_idx]
    target_y = round(float(target_slot[2] + target_slot[3]) * 0.5, 2)
    closest_idx = min(range(len(centers)), key=lambda idx: abs(centers[idx] - target_y))
    return closest_idx + 1


def row_group_for_target(assets: Any, target_idx: int) -> str:
    row_number = target_row_number(assets, target_idx)
    return "upper" if row_number == 3 else "lower"


def planner_maneuver_for_target(
    map_entry: Dict[str, Any],
    map_cfg: Dict[str, Any],
    target_idx: int,
) -> str:
    assets, payload = build_payload_for_target(map_entry, map_cfg, target_idx)
    target_slot = tuple(float(v) for v in assets.slots[target_idx].tolist())
    planner = make_planner(payload)
    target_pose = planner._target_pose(list(target_slot))
    return planner._select_parking_maneuver(list(target_slot), target_pose)


def policy_group_for_target(
    map_entry: Dict[str, Any],
    map_cfg: Dict[str, Any],
    target_idx: int,
) -> str:
    assets = map_entry["assets"]
    row_group = row_group_for_target(assets, target_idx)
    maneuver = planner_maneuver_for_target(map_entry, map_cfg, target_idx)
    maneuver_group = "rear_in" if maneuver.startswith("rear") else "front_in"
    return f"{row_group}_{maneuver_group}"


def build_payload_for_target(map_entry: Dict[str, Any], map_cfg: Dict[str, Any], target_idx: int):
    assets = copy.deepcopy(map_entry["assets"])
    if 0 <= target_idx < len(assets.occupied_idx):
        assets.occupied_idx[target_idx] = False
    payload = sim.build_map_payload(assets)
    payload["expected_orientation"] = map_cfg.get("expected_orientation")
    return assets, payload


def make_planner(payload: Dict[str, Any]) -> Any:
    planner = student_planner.PlannerSkeleton()
    student_planner.planner = planner
    quiet_call(planner.set_map, payload)
    return planner


def initial_parking_pose(
    planner: Any,
    target_slot: Tuple[float, float, float, float],
    target_pose: Tuple[float, float, float],
) -> Optional[Tuple[float, float, float]]:
    p1 = planner.rear_second_p1 or planner.debug_approach_point
    if p1 is None:
        return None
    sequence = planner._rear_y_parking_points((p1[0], p1[1], target_pose[2]), target_pose)
    if len(sequence) < 3:
        return None
    if planner._is_rear_parking_mode():
        next_point = sequence[1]
    else:
        next_point = sequence[2]
    yaw = math.atan2(next_point[1] - p1[1], next_point[0] - p1[0])
    return p1[0], p1[1], yaw


def prepare_parking_only_planner(
    payload: Dict[str, Any],
    target_slot: Tuple[float, float, float, float],
    start_state: Any,
    planned_p1: Optional[Tuple[float, float]] = None,
) -> Tuple[Any, Optional[str]]:
    planner = make_planner(payload)
    signature = tuple(round(float(v), 3) for v in target_slot)
    target_pose = planner._target_pose(list(target_slot))
    planner.target_signature = signature
    planner.parking_mode = planner._expected_parking_mode({"target_slot": target_slot})
    planner.parking_maneuver = planner._select_parking_maneuver(list(target_slot), target_pose)
    p1 = planned_p1 or (start_state.x, start_state.y)
    planner.debug_approach_point = (p1[0], p1[1])
    planner.rear_second_p1 = (p1[0], p1[1]) if planner._is_rear_parking_mode() else None
    planner.waypoints = [
        (start_state.x, start_state.y, start_state.yaw, "D"),
        (start_state.x, start_state.y, start_state.yaw, "D"),
    ]
    planner.waypoint_index = 0
    planner.parking_state = student_planner.PARKING_STATE_APPROACH
    planner.parking_segment_ready = False
    planner.approach_reached_latched = True
    planner.parking_prepare_full_stop_seen = False
    return planner, planner.parking_maneuver


def run_episode(case: Dict[str, Any]) -> Dict[str, Any]:
    rng = random.Random(int(case["episode_seed"]))
    map_cfg = case["map_cfg"]
    target_idx = int(case["target_idx"])
    policy = dict(case["policy"])
    apply_policy(policy)

    map_entry = load_map_entry(map_cfg, seed=int(case["map_seed"]))
    assets, payload = build_payload_for_target(map_entry, map_cfg, target_idx)
    target_slot = tuple(float(v) for v in assets.slots[target_idx].tolist())
    row_number = target_row_number(assets, target_idx)
    policy_group = case.get("policy_group") or policy_group_for_target(map_entry, map_cfg, target_idx)
    params = sim.Params()
    params.timeout = float(case["timeout"])

    probe_planner = make_planner(payload)
    xmin, _, ymin, _ = assets.extent
    normal_start = sim.State(xmin + 4.0, ymin + 6.0, math.radians(90.0), 0.0)
    normal_obs = sim.build_obs_payload(
        0.0,
        normal_start,
        target_slot,
        params,
        map_cfg.get("expected_orientation"),
    )
    quiet_call(probe_planner.compute_path, normal_obs)
    target_pose = probe_planner._target_pose(list(target_slot))
    base_pose = initial_parking_pose(probe_planner, target_slot, target_pose)
    if base_pose is None:
        return failed_result(case, "no_2_1_pose", policy)

    noise_xy = float(case["noise_xy"])
    noise_yaw = math.radians(float(case["noise_yaw_deg"]))
    start_state = sim.State(
        base_pose[0] + rng.uniform(-noise_xy, noise_xy),
        base_pose[1] + rng.uniform(-noise_xy, noise_xy),
        base_pose[2]
        + math.radians(float(policy.get("APPROACH_PREALIGN_YAW_OFFSET_DEG", 0.0)))
        + rng.uniform(-noise_yaw, noise_yaw),
        0.0,
    )
    initial_start_pose = (start_state.x, start_state.y, start_state.yaw)
    planner, maneuver = prepare_parking_only_planner(
        payload,
        target_slot,
        start_state,
        planned_p1=(base_pose[0], base_pose[1]),
    )

    control = sim.InputCmd()
    delta = 0.0
    state = start_state
    t = 0.0
    why = "timeout"
    collision_reason = None
    collision_count = 0
    move_dist = 0.0
    prev_x, prev_y = state.x, state.y
    stats = sim.RoundStats()
    stats.prev_gear = control.gear
    stats.min_abs_steer = abs(delta)
    stats.prev_delta_sign = 0

    line_rects = getattr(assets, "line_rects", None)
    if line_rects is None:
        line_rects = sim.compute_line_rects(assets)
        assets.line_rects = line_rects

    while t < params.timeout:
        obs = sim.build_obs_payload(
            t,
            state,
            target_slot,
            params,
            map_cfg.get("expected_orientation"),
        )
        cmd = quiet_call(planner.compute_control, obs)
        if not isinstance(cmd, dict):
            cmd = {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}
        control.delta_tgt = sim.clamp(float(cmd.get("steer", 0.0)), -params.maxSteer, params.maxSteer)
        control.accel = sim.clamp(float(cmd.get("accel", 0.0)), 0.0, 1.0)
        control.brake = sim.clamp(float(cmd.get("brake", 0.0)), 0.0, 1.0)
        control.gear = "R" if str(cmd.get("gear", "D")).upper().startswith("R") else "D"

        delta = sim.move_toward(delta, control.delta_tgt, params.steerRate * params.dt)
        if control.gear != stats.prev_gear:
            stats.gear_switches += 1
            stats.prev_gear = control.gear
        stats.min_abs_steer = min(stats.min_abs_steer, abs(delta))
        steer_sign = 0
        if abs(delta) >= sim.STEER_FLIP_DEADZONE:
            steer_sign = 1 if delta > 0 else -1
        if steer_sign != 0:
            prev_sign = stats.prev_delta_sign
            if prev_sign != 0 and steer_sign != prev_sign:
                stats.direction_flips += 1
            stats.prev_delta_sign = steer_sign

        gear_sign = 1.0 if control.gear == "D" else -1.0
        a_accel = gear_sign * params.maxAccel * control.accel
        v_sign = math.copysign(1.0, state.v) if abs(state.v) > 1e-6 else 0.0
        a_brake = -v_sign * params.maxBrake * control.brake
        a_coast = 0.0
        if control.accel < 1e-3 and control.brake < 1e-3 and abs(state.v) > 1e-3:
            a_coast = -v_sign * params.coastDecel
        state = sim.step_kinematic(state, delta, a_accel + a_brake + a_coast, params)

        move_dist += math.hypot(state.x - prev_x, state.y - prev_y)
        prev_x, prev_y = state.x, state.y
        stats.avg_speed_accum += abs(state.v)
        stats.speed_samples += 1

        car_poly = sim.car_polygon(state, params)
        slot_iou = sim.compute_slot_iou(car_poly, target_slot)
        slot_orientation = sim.determine_parking_orientation(state, target_slot)
        if slot_iou > stats.final_iou:
            stats.final_iou = slot_iou
            stats.final_orientation = slot_orientation
            stats.final_speed = abs(state.v)

        collided = False
        xmin, xmax, ymin, ymax = assets.extent
        if getattr(sim, "ENABLE_BOUNDARY_COLLISIONS", False):
            if any(not (xmin <= vx <= xmax and ymin <= vy <= ymax) for vx, vy in car_poly):
                collided = True
                collision_reason = "boundary"
        if not collided:
            for idx, rect in enumerate(assets.slots):
                if idx == target_idx or not assets.occupied_idx[idx]:
                    continue
                if sim.poly_intersects_rect(car_poly, tuple(rect)):
                    collided = True
                    collision_reason = "occupied_slot"
                    break
        if not collided:
            for rect in line_rects:
                if sim.poly_intersects_rect(car_poly, tuple(rect)):
                    collided = True
                    collision_reason = "line"
                    break
        if not collided and getattr(sim, "ENABLE_STATIONARY_COLLISIONS", True):
            collided, _ = sim.detect_stationary_collision(car_poly, assets, threshold=assets.FreeThr)
            if collided:
                collision_reason = "stationary"
        if sim.rect_contains_poly(target_slot, car_poly):
            collided = False
            collision_reason = None

        reached = (
            slot_iou >= sim.PARKING_SUCCESS_IOU
            and abs(state.v) <= 0.2
            and slot_orientation != "unknown"
        )
        if collided:
            collision_count += 1
            why = "collision"
            break
        if reached:
            stats.final_iou = slot_iou
            stats.final_orientation = slot_orientation
            stats.final_speed = abs(state.v)
            why = "success"
            break
        t += params.dt

    stats.elapsed = t
    stats.distance = move_dist
    stats.final_speed = max(stats.final_speed, abs(state.v))
    stage_idx, stage_profile = sim.get_stage_profile(map_cfg)
    score, details = sim.compute_round_score(stats, stage_profile, why, assets.extent)
    target_cx = 0.5 * (float(target_slot[0]) + float(target_slot[1]))
    target_cy = 0.5 * (float(target_slot[2]) + float(target_slot[3]))
    final_position_error = math.hypot(state.x - target_cx, state.y - target_cy)
    expected_orientation = str(map_cfg.get("expected_orientation") or "").lower()
    orientation_match = (
        expected_orientation
        and stats.final_orientation != "unknown"
        and stats.final_orientation == expected_orientation
    )
    reward = (
        float(score)
        + 70.0 * float(stats.final_iou)
        + (10.0 if orientation_match else 0.0)
        + (6.0 if stats.final_iou >= sim.PARKING_SUCCESS_IOU and stats.final_speed <= 0.2 else 0.0)
        + (20.0 if why == "success" else 0.0)
        - (35.0 if why == "collision" else 0.0)
        - (20.0 if why == "timeout" else 0.0)
        - 1.2 * final_position_error
        - 0.8 * stats.gear_switches
        - 0.4 * stats.direction_flips
    )

    return {
        "episode": int(case["episode"]),
        "map_key": map_cfg.get("key"),
        "map_name": map_cfg.get("name"),
        "stage": stage_idx,
        "target_idx": target_idx,
        "row_number": row_number,
        "policy_group": policy_group,
        "result": why,
        "collision_reason": collision_reason,
        "score": round(float(score), 3),
        "reward": round(reward, 3),
        "elapsed": round(stats.elapsed, 3),
        "parking_iou": round(float(stats.final_iou), 3),
        "parking_orientation": stats.final_orientation,
        "final_speed": round(float(stats.final_speed), 3),
        "final_position_error": round(final_position_error, 3),
        "orientation_match": bool(orientation_match),
        "gear_switches": int(stats.gear_switches),
        "steer_flips": int(stats.direction_flips),
        "start_pose": [
            round(initial_start_pose[0], 3),
            round(initial_start_pose[1], 3),
            round(math.degrees(initial_start_pose[2]), 3),
        ],
        "final_pose": [round(state.x, 3), round(state.y, 3), round(math.degrees(state.yaw), 3)],
        "base_2_1_pose": [round(base_pose[0], 3), round(base_pose[1], 3), round(math.degrees(base_pose[2]), 3)],
        "maneuver": maneuver,
        "policy": policy,
    }


def failed_result(case: Dict[str, Any], reason: str, policy: Dict[str, float]) -> Dict[str, Any]:
    return {
        "episode": int(case["episode"]),
        "map_key": case["map_cfg"].get("key"),
        "map_name": case["map_cfg"].get("name"),
        "stage": case["map_cfg"].get("stage"),
        "target_idx": int(case["target_idx"]),
        "row_number": case.get("row_number"),
        "policy_group": case.get("policy_group"),
        "result": "error",
        "collision_reason": reason,
        "score": 0.0,
        "reward": -50.0,
        "elapsed": 0.0,
        "parking_iou": 0.0,
        "parking_orientation": "unknown",
        "final_speed": 0.0,
        "gear_switches": 0,
        "steer_flips": 0,
        "start_pose": None,
        "base_2_1_pose": None,
        "maneuver": "unknown",
        "policy": policy,
    }


def build_cases(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rng = random.Random(args.seed)
    maps = stage_maps(args.stage)
    requested_targets = parse_targets(args.targets)
    requested_groups = parse_policy_groups(args.policy_group)
    base_targets: List[Tuple[Dict[str, Any], int, int, str]] = []
    for map_cfg in maps:
        map_entry = load_map_entry(map_cfg, seed=args.map_seed)
        if requested_targets is not None:
            target_indices = requested_targets
        elif map_cfg.get("variant") == "single_free_slot":
            target_indices = list(range(len(map_entry["assets"].slots)))
        else:
            target_indices = eligible_target_indices(map_entry)
        for target_idx in target_indices:
            if 0 <= target_idx < len(map_entry["assets"].slots):
                row_number = target_row_number(map_entry["assets"], int(target_idx))
                policy_group = policy_group_for_target(map_entry, map_cfg, int(target_idx))
                if requested_groups is not None and policy_group not in requested_groups:
                    continue
                base_targets.append((map_cfg, int(target_idx), row_number, policy_group))
    if not base_targets:
        raise RuntimeError("No target cases to tune.")

    cases = []
    for episode in range(args.episodes):
        map_cfg, target_idx, row_number, policy_group = rng.choice(base_targets)
        policy = random_policy(rng) if args.random_policy else {}
        cases.append(
            {
                "episode": episode,
                "episode_seed": rng.randint(0, 2**31 - 1),
                "map_seed": args.map_seed,
                "map_cfg": map_cfg,
                "target_idx": target_idx,
                "row_number": row_number,
                "policy_group": policy_group,
                "policy": policy,
                "timeout": args.timeout,
                "noise_xy": args.noise_xy,
                "noise_yaw_deg": args.noise_yaw_deg,
            }
        )
    return cases


def write_results(results: List[Dict[str, Any]]) -> None:
    results = sorted(results, key=lambda item: item["reward"], reverse=True)
    generated_at = datetime.now().isoformat(timespec="seconds")
    best = results[0] if results else None
    best_by_group: Dict[str, Dict[str, Any]] = {}
    for item in results:
        group = str(item.get("policy_group") or "unknown")
        if group not in best_by_group:
            best_by_group[group] = item
    policy_candidates = {
        group: {
            "policy": item.get("policy", {}),
            "reward": item.get("reward"),
            "score": item.get("score"),
            "target_idx": item.get("target_idx"),
            "row_number": item.get("row_number"),
            "result": item.get("result"),
        }
        for group, item in sorted(best_by_group.items())
    }
    merged_policy_candidates = merge_policy_candidates(policy_candidates)
    payload = {
        "generated_at": generated_at,
        "best": best,
        "best_by_group": best_by_group,
        "policy_candidates": policy_candidates,
        "merged_policy_candidates": merged_policy_candidates,
        "results": results,
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    POLICY_JSON.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "policy_candidates": merged_policy_candidates,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Parking Tune Result",
        "",
        f"- generated_at: `{generated_at}`",
        f"- episodes: `{len(results)}`",
    ]
    if best:
        lines.extend(
            [
                f"- best_reward: `{best['reward']}`",
                f"- best_score: `{best['score']}`",
                f"- best_policy: `{json.dumps(best['policy'], ensure_ascii=False)}`",
                "",
            ]
        )
    if policy_candidates:
        lines.extend(["## Best By Group", ""])
        for group, item in policy_candidates.items():
            lines.append(
                f"- `{group}`: reward `{item['reward']}`, score `{item['score']}`, "
                f"target `T{item['target_idx']}`, policy `{json.dumps(item['policy'], ensure_ascii=False)}`"
            )
        lines.append("")
    if merged_policy_candidates:
        lines.extend(["## Merged Policy Candidates", ""])
        for group, item in merged_policy_candidates.items():
            lines.append(
                f"- `{group}`: result `{item.get('result')}`, score `{item.get('score')}`, "
                f"reward `{item.get('reward')}`, target `T{item.get('target_idx')}`"
            )
        lines.append("")
    lines.extend(
        [
            "| rank | group | row | stage | map | target | result | score | reward | iou | orientation | gear | steer | policy |",
            "| ---: | --- | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for rank, item in enumerate(results[:30], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    str(item.get("policy_group")),
                    str(item.get("row_number")),
                    str(item.get("stage")),
                    str(item.get("map_name")),
                    str(item.get("target_idx")),
                    str(item.get("result")),
                    f"{item.get('score', 0.0):.1f}",
                    f"{item.get('reward', 0.0):.1f}",
                    f"{item.get('parking_iou', 0.0):.3f}",
                    str(item.get("parking_orientation")),
                    str(item.get("gear_switches")),
                    str(item.get("steer_flips")),
                    "`" + json.dumps(item.get("policy", {}), ensure_ascii=False) + "`",
                ]
            )
            + " |"
        )
    RESULT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parking-only tuning episodes from noisy 2-1 poses.")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--stage", type=int, default=None)
    parser.add_argument("--targets", type=str, default=None, help="Comma separated target ids, e.g. T21,T26")
    parser.add_argument(
        "--policy-group",
        type=str,
        default=None,
        help="Policy group filter: lower_front_in, lower_rear_in, upper_front_in, upper_rear_in, or comma list.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--map-seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--noise-xy", type=float, default=1.0)
    parser.add_argument("--noise-yaw-deg", type=float, default=40.0)
    parser.add_argument("--random-policy", action="store_true", help="Sample simple parking parameters per episode.")
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress and save partial results every N episodes.")
    return parser.parse_args()


def report_progress(results: List[Dict[str, Any]], done: int, total: int) -> None:
    best = max(results, key=lambda item: item["reward"]) if results else None
    success_count = sum(1 for item in results if item.get("result") == "success")
    if best is None:
        print(f"[parking_tune] progress: {done}/{total}", flush=True)
        return
    print(
        "[parking_tune] progress:"
        f" {done}/{total}"
        f" success={success_count}"
        f" best_reward={best['reward']:.1f}"
        f" best_score={best['score']:.1f}"
        f" best_target=T{best['target_idx']}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    cases = build_cases(args)
    total = len(cases)
    progress_every = max(1, int(args.progress_every))
    print(
        "[parking_tune] start:"
        f" episodes={total}"
        f" workers={max(1, int(args.workers))}"
        f" timeout={args.timeout}s"
        f" noise_xy=+/-{args.noise_xy}m"
        f" noise_yaw=+/-{args.noise_yaw_deg}deg"
        f" policy_group={args.policy_group or 'all'}",
        flush=True,
    )
    if args.workers <= 1:
        results = []
        for idx, case in enumerate(cases, start=1):
            results.append(run_episode(case))
            if idx % progress_every == 0 or idx == total:
                write_results(results)
                report_progress(results, idx, total)
    else:
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_episode, case) for case in cases]
            for idx, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if idx % progress_every == 0 or idx == total:
                    write_results(results)
                    report_progress(results, idx, total)
    write_results(results)
    best = max(results, key=lambda item: item["reward"])
    print(
        "[parking_tune] done:"
        f" episodes={len(results)}"
        f" best_reward={best['reward']:.1f}"
        f" best_score={best['score']:.1f}"
        f" result={best['result']}"
        f" target=T{best['target_idx']}"
        f" success={sum(1 for item in results if item.get('result') == 'success')}",
        flush=True,
    )
    print(f"[parking_tune] saved: {RESULT_JSON.name}, {RESULT_MD.name}", flush=True)
    print(f"[parking_tune] policy candidates: {POLICY_JSON.name}", flush=True)


if __name__ == "__main__":
    main()
