"""Student parking planner.

Only this file is intended to be edited by students. The simulator sends a
static map once, then sends observation packets every tick. `planner_step`
returns the command dictionary expected by the provided IPC client:

    {"steer": radians, "accel": 0..1, "brake": 0..1, "gear": "D" or "R"}

This implementation is a minimum working baseline:
- A* plans a collision-aware path from the current vehicle position to an
  approach point near the target parking slot.
- A short final segment moves into the slot center with the desired yaw.
- Pure Pursuit tracks the waypoint path.
- Simple proportional speed control slows near the slot, obstacles, and sharp
  turns.
"""

from __future__ import annotations

import heapq
import json
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rl_speed_controller import RLSpeedController


Waypoint = Tuple[float, float, float, str]  # x, y, desired yaw, gear
USE_RL_SPEED_CONTROL = os.getenv("PARKING_USE_RL_SPEED", "0").lower() not in {
    "0",
    "false",
    "off",
    "no",
}
USE_ASYNC_PLANNING = os.getenv("PARKING_ASYNC_PLANNER", "1").lower() not in {
    "0",
    "false",
    "off",
    "no",
}
VEHICLE_LONGEST_LENGTH = 3.0
VEHICLE_BOUNDARY_DIAMETER = VEHICLE_LONGEST_LENGTH * 1.20
VEHICLE_CENTER_CLEARANCE = 0.5 * VEHICLE_BOUNDARY_DIAMETER
PLANNING_OBSTACLE_MARGIN = VEHICLE_CENTER_CLEARANCE
EXTRA_SAFETY_MARGIN = 0.0
OBSTACLE_SLOW_DISTANCE = 1.15
OBSTACLE_STOP_DISTANCE = 0.45
FRONT_CLEAR_DISTANCE = 6.0
FRONT_CLEAR_SPEED_BONUS = 0.75
PARKING_ALIGN_DISTANCE = 4.0
PARKING_REVERSE_YAW_ERROR = math.radians(32.0)
PARKING_REVERSE_TICKS = 58
PARKING_REVERSE_COOLDOWN_TICKS = 45
PARKING_REVERSE_WALL_CLEARANCE = 0.85
PARKING_TARGET_OVERSHOOT = 0.30
LINE_EXTRA_CLEARANCE = PLANNING_OBSTACLE_MARGIN * 0.20
GUIDED_LINE_FOOTPRINT_MARGIN = 1.20
GUIDED_GRID_STEP_MIN = 0.75
GUIDED_CANDIDATE_EVAL_LIMIT = 2
TERMINAL_XY_RESOLUTION = 1.0
TERMINAL_YAW_RESOLUTION = math.radians(30.0)
TERMINAL_PRIMITIVE_LENGTH = 1.5
TERMINAL_PRIMITIVE_STEPS = 6
TERMINAL_MAX_ITERATIONS = 3600
GEAR_TRANSITION_RADIUS = 1.25
GEAR_TRANSITION_YAW_TOLERANCE = math.radians(45.0)
TERMINAL_LOOP_STUCK_TICKS = 240
TERMINAL_LOOP_FORCE_RADIUS = 1.10
TERMINAL_REPLAN_STUCK_TICKS = 360
TERMINAL_REPLAN_MAX_COUNT = 0
SEMANTIC_ENTRY_BONUS = 3.0
FALLBACK_ENTRY_PENALTY = 4.0
PREVIEW_MAX_TIME = 22.0
PREVIEW_DT = 0.10
PREVIEW_MAX_ACCEL = 2.8
PREVIEW_MAX_BRAKE = 4.5
PREVIEW_STEER_RATE = math.radians(240.0)
PREVIEW_SUCCESS_IOU = 0.45
PARKING_DEEP_IOU_TARGET = 0.50
PARKING_CREEP_IOU = 0.25
PARKING_STOP_IOU = 0.48
SPATIAL_INDEX_CELL_SIZE = 4.0
CLEARANCE_QUERY_RADIUS = 8.0
RISK_FIELD_SIGMA = 1.0
GUIDED_DISTANCE_EPS = 0.25
GUIDED_ENTRY_CLEARANCE_COST_WEIGHT = 0.08
GUIDED_RISK_COST_WEIGHT = 0.05
GUIDED_CLEARANCE_COST_WEIGHT = 0.02
GUIDED_NEAR_LINE_COST_WEIGHT = 0.04
GUIDED_NEAR_LINE_THRESHOLD = 1.10
GUIDED_FAST_APPROACH_SPEED = 3.30
GUIDED_FAST_APPROACH_CLEARANCE = 1.45
GUIDED_FAST_APPROACH_STEER = math.radians(14.0)
GUIDED_TERMINAL_STOP_DISTANCE = 0.24
GUIDED_TERMINAL_DEEP_ENTRY_SPEED = 0.42
# Conservative guard: avoid simulator ending at IoU~=0.30 while the car is still
# visibly short of the slot center, but do not force deep entry near walls.
ANTI_EARLY_SUCCESS_IOU = 0.30
ANTI_EARLY_SUCCESS_MIN_DIST = 0.36
ANTI_EARLY_SUCCESS_SPEED = 0.26
ANTI_EARLY_SUCCESS_MIN_CLEARANCE = 0.18
STEER_DEADBAND = math.radians(1.2)
STEER_SMOOTH_ALPHA_APPROACH = 0.42
POSE_Q_SIGMA = 5.0
POSE_Q_FAR = (1.0, 1.1, 0.35)
POSE_Q_NEAR = (1.0, 3.0, 7.0)


def pretty_print_map_summary(map_payload: Dict[str, Any]) -> None:
    extent = map_payload.get("extent") or [None, None, None, None]
    slots = map_payload.get("slots") or []
    occupied = map_payload.get("occupied_idx") or []
    free_slots = len(slots) - sum(1 for v in occupied if v)
    print("[algo] map extent :", extent)
    print("[algo] total slots:", len(slots), "/ free:", free_slots)
    stationary = map_payload.get("grid", {}).get("stationary")
    if stationary:
        rows = len(stationary)
        cols = len(stationary[0]) if stationary else 0
        print("[algo] grid size  :", rows, "x", cols)


@dataclass
class EntryPlan:
    sign: float
    semantic_match: bool
    fallback_used: bool
    score: float
    preview_reason: str
    preview_iou_proxy: float
    grid_cost: float
    hybrid_start_index: int
    waypoints: List[Waypoint]


@dataclass
class PlannerSkeleton:
    """Rule-based planner and controller fitted to the existing simulator API."""

    map_data: Optional[Dict[str, Any]] = None
    map_extent: Optional[Tuple[float, float, float, float]] = None
    cell_size: float = 0.5
    stationary_grid: Optional[List[List[float]]] = None
    waypoints: List[Waypoint] = None
    waypoint_index: int = 0
    target_signature: Optional[Tuple[float, ...]] = None
    last_log_time: float = -999.0
    step_count: int = 0
    min_obstacle_distance: float = float("inf")
    rl_speed: RLSpeedController = None
    last_eval_log_time: float = -999.0
    final_eval_logged: bool = False
    planning_fail_reason: Optional[str] = None
    guided_blocked_grid_cache: Optional[Tuple[int, int, float, List[List[bool]]]] = None
    guided_distance_field_cache: Optional[Tuple[int, int, float, List[List[float]]]] = None
    obstacle_rect_cache: Optional[List[Tuple[float, float, float, float]]] = None
    line_rect_cache: Optional[Dict[float, List[Tuple[float, float, float, float]]]] = None
    collision_rect_cache: Optional[List[Tuple[float, float, float, float]]] = None
    spatial_index_cache: Optional[
        Dict[
            str,
            Tuple[
                float,
                Dict[Tuple[int, int], List[Tuple[float, float, float, float]]],
            ],
        ]
    ] = None
    parking_reverse_ticks: int = 0
    parking_reverse_cooldown: int = 0
    parking_has_reversed: bool = False
    hybrid_start_index: int = 10**9
    current_yaw_for_progress: float = 0.0
    planning_thread: Optional[threading.Thread] = None
    planning_signature: Optional[Tuple[float, ...]] = None
    planning_started_at: float = -999.0
    last_command_gear: str = "D"
    terminal_progress_index: int = -1
    terminal_progress_best_dist: float = float("inf")
    terminal_stuck_ticks: int = 0
    terminal_replan_count: int = 0
    last_runtime_steer: float = 0.0
    last_runtime_steer_gear: str = "D"

    def __post_init__(self) -> None:
        if self.waypoints is None:
            self.waypoints = []
        if self.rl_speed is None:
            self.rl_speed = RLSpeedController(enabled=USE_RL_SPEED_CONTROL)

    def set_map(self, map_payload: Dict[str, Any]) -> None:
        """Store static map data sent by the simulator."""

        self.map_data = map_payload
        self.map_extent = tuple(
            map(float, map_payload.get("extent", (0.0, 0.0, 0.0, 0.0)))
        )
        self.cell_size = float(map_payload.get("cellSize", 0.5))
        self.stationary_grid = map_payload.get("grid", {}).get("stationary")
        pretty_print_map_summary(map_payload)
        self.waypoints.clear()
        self.waypoint_index = 0
        self.target_signature = None
        self.last_log_time = -999.0
        self.step_count = 0
        self.min_obstacle_distance = float("inf")
        self.last_eval_log_time = -999.0
        self.final_eval_logged = False
        self.planning_fail_reason = None
        self.guided_blocked_grid_cache = None
        self.guided_distance_field_cache = None
        self.obstacle_rect_cache = None
        self.line_rect_cache = None
        self.collision_rect_cache = None
        self.spatial_index_cache = None
        self.parking_reverse_ticks = 0
        self.parking_reverse_cooldown = 0
        self.parking_has_reversed = False
        self.hybrid_start_index = 10**9
        self.current_yaw_for_progress = 0.0
        self.planning_thread = None
        self.planning_signature = None
        self.planning_started_at = -999.0
        self.last_command_gear = "D"
        self.terminal_progress_index = -1
        self.terminal_progress_best_dist = float("inf")
        self.terminal_stuck_ticks = 0
        self.terminal_replan_count = 0
        self.last_runtime_steer = 0.0
        self.last_runtime_steer_gear = "D"
        self.rl_speed = RLSpeedController(enabled=USE_RL_SPEED_CONTROL)
        self._warm_planning_caches()
        print(f"[algo] rl_speed_control={'ON' if USE_RL_SPEED_CONTROL else 'OFF'}")

    def compute_path(self, obs: Dict[str, Any]) -> None:
        """Plan a path from the current pose to the target parking slot."""

        self.waypoints.clear()
        self.waypoint_index = 0
        self.parking_reverse_ticks = 0
        self.parking_reverse_cooldown = 0
        self.parking_has_reversed = False
        self.hybrid_start_index = 10**9
        self.last_command_gear = "D"
        self.terminal_progress_index = -1
        self.terminal_progress_best_dist = float("inf")
        self.terminal_stuck_ticks = 0
        self.last_runtime_steer = 0.0
        self.last_runtime_steer_gear = "D"
        state = obs.get("state", {})
        start = (
            float(state.get("x", 0.0)),
            float(state.get("y", 0.0)),
            float(state.get("yaw", 0.0)),
        )
        slot = obs.get("target_slot") or []
        if len(slot) != 4 or self.map_extent is None:
            self.planning_fail_reason = "missing_target_or_map"
            print("[algo] planning failed: missing target slot or map")
            return

        self.target_signature = tuple(round(float(v), 3) for v in slot)
        target_pose = self._target_pose(slot)
        guided_plan = self._semantic_guided_entry_plan(start, slot, target_pose, obs)
        if guided_plan is None:
            guided_plan = self._direct_semantic_entry_fallback(start, slot, target_pose)
        if guided_plan is None:
            self.planning_fail_reason = "semantic_guided_planner_failed"
            print("[algo] planning failed: semantic-guided planner produced no candidate")
            return

        self.planning_fail_reason = None
        self.waypoints = guided_plan.waypoints
        self.hybrid_start_index = guided_plan.hybrid_start_index
        initial_clearance = self._estimate_min_obstacle_distance((start[0], start[1]))
        self.min_obstacle_distance = min(self.min_obstacle_distance, initial_clearance)
        print(
            "[algo] semantic-guided terminal plan:"
            f" sign={guided_plan.sign:+.0f}"
            f" semantic={'yes' if guided_plan.semantic_match else 'no'}"
            f" fallback={'yes' if guided_plan.fallback_used else 'no'}"
            f" preview={guided_plan.preview_reason}"
            f" score={guided_plan.score:.2f}"
            f" waypoints={len(self.waypoints)}"
        )
        return

    def compute_control(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Return steering, acceleration, brake, and gear for one simulation tick."""

        self.step_count += 1
        state = obs.get("state", {})
        x = float(state.get("x", 0.0))
        y = float(state.get("y", 0.0))
        yaw = float(state.get("yaw", 0.0))
        speed = abs(float(state.get("v", 0.0)))
        t = float(obs.get("t", 0.0))
        limits = obs.get("limits", {})
        wheelbase = float(limits.get("L", 2.6))
        max_steer = float(limits.get("maxSteer", math.radians(35.0)))

        slot = obs.get("target_slot") or []
        signature = tuple(round(float(v), 3) for v in slot) if len(slot) == 4 else None
        if not self.waypoints or signature != self.target_signature:
            if USE_ASYNC_PLANNING:
                self._ensure_async_path(obs, signature, t)
            else:
                self.compute_path(obs)

        if not self.waypoints:
            return {"steer": 0.0, "accel": 0.0, "brake": 0.8, "gear": "D"}
        if self.hybrid_start_index < 10**9:
            return self._guided_compute_control(obs)

        final_wp = self.waypoints[-1]
        if len(slot) == 4:
            target_center = self._slot_center(slot)
            final_dist = math.hypot(target_center[0] - x, target_center[1] - y)
            center_tolerance = self._slot_center_tolerance(slot)
            slot_entered = self._point_in_slot(slot, x, y, margin=0.05)
        else:
            target_center = (final_wp[0], final_wp[1])
            final_dist = math.hypot(final_wp[0] - x, final_wp[1] - y)
            center_tolerance = 0.55
            slot_entered = False
        final_yaw_error = abs(self._wrap_to_pi(final_wp[2] - yaw))
        obstacle_dist = self._estimate_min_obstacle_distance((x, y))
        self.min_obstacle_distance = min(self.min_obstacle_distance, obstacle_dist)
        collision_risk = False

        if final_dist <= center_tolerance and final_yaw_error < math.radians(14.0):
            self._log_evaluation(
                parking_success=True,
                fail_reason="none",
                final_position_error=final_dist,
                final_yaw_error=final_yaw_error,
                collision=collision_risk,
                force=True,
            )
            if speed < 0.18:
                print(
                    "[algo] parking succeeded:"
                    f" pos_error={final_dist:.2f}m"
                    f" center_tolerance={center_tolerance:.2f}m"
                    f" yaw_error={math.degrees(final_yaw_error):.1f}deg"
                    f" steps={self.step_count}"
                    f" min_obstacle_dist~{self.min_obstacle_distance:.2f}m"
                )
            return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}

        self.current_yaw_for_progress = yaw
        self._advance_waypoint_index(x, y)
        lookahead = self._adaptive_lookahead(speed, final_dist, final_yaw_error)
        target_idx = self._lookahead_index(x, y, lookahead=lookahead)
        target_wp = self.waypoints[target_idx]
        gear = target_wp[3]
        in_parking_mode = final_dist < PARKING_ALIGN_DISTANCE
        terminal_plan_active = self.waypoint_index >= self.hybrid_start_index
        guided_plan_active = self.hybrid_start_index < 10**9
        reverse_realigning = False
        forward_clearance = self._estimate_forward_clearance(
            x=x,
            y=y,
            yaw=yaw,
            reverse=False,
        )
        target_overshot = self._passed_target_center(
            x=x,
            y=y,
            target_center=target_center,
            target_yaw=final_wp[2],
            tolerance=max(center_tolerance, PARKING_TARGET_OVERSHOOT),
        )
        if not terminal_plan_active:
            if self.parking_reverse_cooldown > 0:
                self.parking_reverse_cooldown -= 1
            should_reverse_for_alignment = (
                not self.parking_has_reversed
                and final_yaw_error > PARKING_REVERSE_YAW_ERROR
            )
            should_reverse_for_obstacle = forward_clearance < PARKING_REVERSE_WALL_CLEARANCE
            should_reverse_after_forward = self.parking_has_reversed and (
                should_reverse_for_obstacle or target_overshot
            )
            if (
                in_parking_mode
                and self.parking_reverse_ticks <= 0
                and (self.parking_reverse_cooldown <= 0 or should_reverse_after_forward)
                and (
                    should_reverse_for_alignment
                    or should_reverse_for_obstacle
                    or should_reverse_after_forward
                )
                and final_dist > max(center_tolerance * 1.8, 0.35)
            ):
                self.parking_reverse_ticks = PARKING_REVERSE_TICKS
                self.parking_has_reversed = True
                reverse_reason = (
                    "alignment"
                    if should_reverse_for_alignment
                    else "front_obstacle" if should_reverse_for_obstacle else "target_overshoot"
                )
                print(
                    "[algo] parking recovery: reverse realignment"
                    f" pos_error={final_dist:.2f}m"
                    f" yaw_error={math.degrees(final_yaw_error):.1f}deg"
                    f" reason={reverse_reason}"
                    f" forward_clearance={forward_clearance:.2f}m"
                    f" target_overshot={target_overshot}"
                )
            if in_parking_mode and self.parking_reverse_ticks > 0:
                self.parking_reverse_ticks -= 1
                if self.parking_reverse_ticks <= 0:
                    self.parking_reverse_cooldown = PARKING_REVERSE_COOLDOWN_TICKS
                target_wp = self._parking_reverse_target(target_center, final_wp[2])
                gear = "R"
                reverse_realigning = True
            elif in_parking_mode:
                target_wp = (target_center[0], target_center[1], final_wp[2], "D")
                gear = "D"

        steer = self._pure_pursuit_steer(
            x=x,
            y=y,
            yaw=yaw,
            target_x=target_wp[0],
            target_y=target_wp[1],
            wheelbase=wheelbase,
            max_steer=max_steer,
            reverse=(gear == "R"),
        )
        if reverse_realigning:
            steer = 0.0

        front_clearance = self._estimate_forward_clearance(
            x=x,
            y=y,
            yaw=yaw,
            reverse=(gear == "R"),
        )
        collision_risk = front_clearance < OBSTACLE_STOP_DISTANCE
        if collision_risk and final_dist > 1.0:
            self._log_evaluation(
                parking_success=False,
                fail_reason="front_collision_risk",
                final_position_error=final_dist,
                final_yaw_error=final_yaw_error,
                collision=True,
                current_time=t,
            )
            return {"steer": steer * 0.4, "accel": 0.0, "brake": 1.0, "gear": gear}

        front_is_clear = front_clearance >= FRONT_CLEAR_DISTANCE
        rule_speed = self._target_speed(
            final_dist,
            final_yaw_error,
            steer,
            front_clearance,
        )
        if guided_plan_active and final_dist < 8.0:
            rule_speed = min(rule_speed, 0.75)
        if guided_plan_active and final_dist < 5.0:
            rule_speed = min(rule_speed, 0.45)
        if terminal_plan_active:
            rule_speed = min(rule_speed, 0.52)
            if final_dist < 3.0:
                rule_speed = min(rule_speed, 0.30)
        if gear == "R":
            rule_speed = min(rule_speed, 0.55)
        target_speed = self.rl_speed.adjust_target_speed(
            rule_speed=rule_speed,
            final_dist=final_dist,
            yaw_error=final_yaw_error,
            steer_abs=abs(steer),
            obstacle_dist=front_clearance,
        )
        straight_clear_full_accel = (
            gear == "D"
            and not in_parking_mode
            and front_is_clear
            and abs(steer) < math.radians(4.0)
        )
        accel, brake = self._speed_command(
            speed=speed,
            target_speed=target_speed,
            front_is_clear=front_is_clear and final_dist > 3.0,
            force_full_accel=straight_clear_full_accel,
        )
        self._log_evaluation(
            parking_success=False,
            fail_reason="front_collision_risk" if collision_risk else self.planning_fail_reason or "running",
            final_position_error=final_dist,
            final_yaw_error=final_yaw_error,
            collision=collision_risk,
            current_time=t,
        )

        if t - self.last_log_time > 2.0:
            self.last_log_time = t
            print(
                "[algo] tracking:"
                f" wp={self.waypoint_index}/{len(self.waypoints) - 1}"
                f" pos_error={final_dist:.2f}m"
                f" center_tolerance={center_tolerance:.2f}m"
                f" slot_entered={slot_entered}"
                f" yaw_error={math.degrees(final_yaw_error):.1f}deg"
                f" min_obstacle_dist~{self.min_obstacle_distance:.2f}m"
                f" front_clearance~{front_clearance:.2f}m"
                f" gear={gear}"
                f" lookahead={lookahead:.2f}m"
                f" rule_speed={rule_speed:.2f}m/s"
                f" speed={target_speed:.2f}m/s"
                f" rl={'ON' if self.rl_speed.enabled else 'OFF'}"
                f" rl_state={self.rl_speed.last_state}"
                f" rl_action={self.rl_speed.last_action}"
            )

        return {"steer": steer, "accel": accel, "brake": brake, "gear": gear}

    def _ensure_async_path(
        self,
        obs: Dict[str, Any],
        signature: Optional[Tuple[float, ...]],
        current_time: float,
    ) -> None:
        if signature is None:
            return
        if self.planning_thread is not None and self.planning_thread.is_alive():
            if current_time - self.last_log_time > 1.0:
                self.last_log_time = current_time
                print("[algo] planning in background; holding brake")
            return
        if self.planning_signature == signature and self.waypoints:
            return
        self.planning_signature = signature
        self.planning_started_at = time.perf_counter()
        obs_copy = json.loads(json.dumps(obs))
        self.planning_thread = threading.Thread(
            target=self._async_compute_path,
            args=(obs_copy, signature),
            daemon=True,
        )
        self.planning_thread.start()
        print("[algo] started background planning; holding brake")

    def _async_compute_path(
        self,
        obs: Dict[str, Any],
        signature: Tuple[float, ...],
    ) -> None:
        started = time.perf_counter()
        try:
            self.compute_path(obs)
            elapsed = time.perf_counter() - started
            if self.target_signature == signature and self.waypoints:
                print(
                    "[algo] background planning ready:"
                    f" elapsed={elapsed:.2f}s waypoints={len(self.waypoints)}"
                )
        except Exception as exc:
            self.planning_fail_reason = f"async_planning_error:{exc}"
            print(f"[algo] background planning error: {exc}")

    def _guided_compute_control(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        state = obs.get("state", {})
        x = float(state.get("x", 0.0))
        y = float(state.get("y", 0.0))
        yaw = float(state.get("yaw", 0.0))
        speed = abs(float(state.get("v", 0.0)))
        t = float(obs.get("t", 0.0))
        limits = obs.get("limits", {})
        wheelbase = float(limits.get("L", 2.6))
        max_steer = float(limits.get("maxSteer", math.radians(35.0)))

        final_wp = self.waypoints[-1]
        final_dist = math.hypot(final_wp[0] - x, final_wp[1] - y)
        final_yaw_error = abs(self._wrap_to_pi(final_wp[2] - yaw))
        slot_payload = obs.get("target_slot") or []
        target_slot_tuple: Optional[Tuple[float, float, float, float]] = None
        slot_iou_proxy = 0.0
        slot_iou = 0.0
        if len(slot_payload) == 4:
            target_slot_tuple = tuple(float(v) for v in slot_payload)
            slot_iou_proxy = self._slot_center_iou_proxy(target_slot_tuple, x, y)
            slot_iou = self._slot_iou(target_slot_tuple, x, y, yaw)
        expected = str((self.map_data or {}).get("expected_orientation") or "").lower()
        obstacle_dist = self._estimate_clearance((x, y), include_lines=True)
        self.min_obstacle_distance = min(self.min_obstacle_distance, obstacle_dist)
        collision_risk = obstacle_dist < 0.08

        if target_slot_tuple is not None:
            # Do not use the simulator's loose IoU=0.30 success boundary as our own
            # stop point. Stop when either IoU is meaningfully better, or when the
            # planned final pose has actually been reached.
            stop_ready = (
                (slot_iou >= PARKING_STOP_IOU or (slot_iou >= 0.30 and final_dist < 0.26))
                and final_yaw_error < math.radians(18.0)
                and speed < 0.22
            )
        else:
            stop_ready = final_dist < GUIDED_TERMINAL_STOP_DISTANCE and final_yaw_error < math.radians(14.0)
        if stop_ready:
            self._log_evaluation(
                parking_success=True,
                fail_reason="none",
                final_position_error=final_dist,
                final_yaw_error=final_yaw_error,
                collision=collision_risk,
                force=True,
            )
            return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}

        self.current_yaw_for_progress = yaw
        self._advance_waypoint_index(x, y)
        terminal_plan_active = self.waypoint_index >= self.hybrid_start_index
        anti_early_success_guard = (
            terminal_plan_active
            and target_slot_tuple is not None
            and ANTI_EARLY_SUCCESS_IOU <= slot_iou < PARKING_STOP_IOU
            and final_dist > ANTI_EARLY_SUCCESS_MIN_DIST
            and final_yaw_error < math.radians(24.0)
            and obstacle_dist > ANTI_EARLY_SUCCESS_MIN_CLEARANCE
        )
        if (
            terminal_plan_active
            and self.terminal_stuck_ticks >= TERMINAL_REPLAN_STUCK_TICKS
            and final_dist > 0.8
        ):
            if speed > 0.12:
                return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": self.last_command_gear}
            if self.terminal_replan_count >= TERMINAL_REPLAN_MAX_COUNT:
                self.planning_fail_reason = "terminal_loop_hold"
                return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": self.last_command_gear}
            self.terminal_replan_count += 1
            print(
                "[algo] terminal recovery: replanning from current pose"
                f" count={self.terminal_replan_count}"
                f" wp={self.waypoint_index}/{len(self.waypoints) - 1}"
                f" pos_error={final_dist:.2f}m"
            )
            self.compute_path(obs)
            return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": self.last_command_gear}
        lookahead = self._guided_lookahead(speed, final_dist, final_yaw_error)
        target_idx = self._guided_lookahead_index(x, y, lookahead)
        target_wp = self.waypoints[target_idx]
        gear = target_wp[3]

        if gear != self.last_command_gear and speed > 0.12:
            return {
                "steer": 0.0,
                "accel": 0.0,
                "brake": 1.0,
                "gear": self.last_command_gear,
            }
        self.last_command_gear = gear

        raw_steer = self._pure_pursuit_steer(
            x=x,
            y=y,
            yaw=yaw,
            target_x=target_wp[0],
            target_y=target_wp[1],
            wheelbase=wheelbase,
            max_steer=max_steer,
            reverse=(gear == "R"),
        )
        steer = self._runtime_steer_filter(
            raw_steer,
            max_steer=max_steer,
            terminal_plan_active=terminal_plan_active,
            gear=gear,
        )

        rule_speed = self._guided_target_speed(final_dist, final_yaw_error, steer, obstacle_dist)
        if (
            gear == "D"
            and not terminal_plan_active
            and obstacle_dist >= GUIDED_FAST_APPROACH_CLEARANCE
            and final_dist > 8.0
            and abs(steer) <= GUIDED_FAST_APPROACH_STEER
        ):
            rule_speed = max(rule_speed, min(GUIDED_FAST_APPROACH_SPEED, 2.35 + 0.035 * final_dist))
        if (
            terminal_plan_active
            and final_dist > GUIDED_TERMINAL_STOP_DISTANCE + 0.18
            and final_dist < 1.7
            and final_yaw_error < math.radians(18.0)
            and obstacle_dist > 0.35
        ):
            rule_speed = max(rule_speed, GUIDED_TERMINAL_DEEP_ENTRY_SPEED)
        if (
            terminal_plan_active
            and target_slot_tuple is not None
            and PARKING_CREEP_IOU <= slot_iou < PARKING_STOP_IOU
            and final_yaw_error < math.radians(22.0)
            and obstacle_dist > 0.05
        ):
            rule_speed = max(rule_speed, 0.50)
        if anti_early_success_guard:
            # Keep just enough motion to avoid the simulator declaring success at
            # IoU~=0.30 while we are still short of the planned final pose.
            rule_speed = max(rule_speed, ANTI_EARLY_SUCCESS_SPEED)
        if (
            terminal_plan_active
            and not expected.startswith("rear")
            and gear == "R"
            and GUIDED_TERMINAL_STOP_DISTANCE + 0.20 < final_dist < 1.65
            and final_yaw_error < math.radians(27.0)
            and slot_iou_proxy > 0.66
            and (abs(steer) < math.radians(31.0) or final_dist < 0.78)
        ):
            rule_speed = max(rule_speed, 0.38)
        target_speed = self.rl_speed.adjust_target_speed(
            rule_speed=rule_speed,
            final_dist=final_dist,
            yaw_error=final_yaw_error,
            steer_abs=abs(steer),
            obstacle_dist=obstacle_dist,
        )
        front_is_clear = obstacle_dist > 1.5 and final_dist > 4.0
        force_full_accel = (
            gear == "D"
            and front_is_clear
            and final_dist > 8.0
            and not terminal_plan_active
            and abs(steer) < math.radians(10.0)
        )
        accel, brake = self._speed_command(
            speed=speed,
            target_speed=target_speed,
            front_is_clear=front_is_clear,
            force_full_accel=force_full_accel,
        )
        if anti_early_success_guard and speed < 0.22:
            accel = max(accel, 0.18)
            brake = 0.0
        self._log_evaluation(
            parking_success=False,
            fail_reason="collision_risk" if collision_risk else self.planning_fail_reason or "running",
            final_position_error=final_dist,
            final_yaw_error=final_yaw_error,
            collision=collision_risk,
            current_time=t,
        )

        if t - self.last_log_time > 2.0:
            self.last_log_time = t
            print(
                "[algo] guided tracking:"
                f" wp={self.waypoint_index}/{len(self.waypoints) - 1}"
                f" pos_error={final_dist:.2f}m"
                f" yaw_error={math.degrees(final_yaw_error):.1f}deg"
                f" min_obstacle_dist~{self.min_obstacle_distance:.2f}m"
                f" gear={gear}"
                f" lookahead={lookahead:.2f}m"
                f" rule_speed={rule_speed:.2f}m/s"
                f" speed={target_speed:.2f}m/s"
                f" rl={'ON' if self.rl_speed.enabled else 'OFF'}"
            )

        self.last_command_gear = gear
        return {"steer": steer, "accel": accel, "brake": brake, "gear": gear}

    def _runtime_steer_filter(
        self,
        raw_steer: float,
        max_steer: float,
        terminal_plan_active: bool,
        gear: str,
    ) -> float:
        """Reduce tiny approach-phase steering sign flips without touching terminal docking."""
        if terminal_plan_active:
            self.last_runtime_steer = raw_steer
            self.last_runtime_steer_gear = gear
            return max(-max_steer, min(max_steer, raw_steer))
        if gear != self.last_runtime_steer_gear:
            self.last_runtime_steer = raw_steer
            self.last_runtime_steer_gear = gear
            return max(-max_steer, min(max_steer, raw_steer))
        if abs(raw_steer) < STEER_DEADBAND:
            raw_steer = 0.0
        smoothed = (
            (1.0 - STEER_SMOOTH_ALPHA_APPROACH) * self.last_runtime_steer
            + STEER_SMOOTH_ALPHA_APPROACH * raw_steer
        )
        self.last_runtime_steer = smoothed
        return max(-max_steer, min(max_steer, smoothed))

    def _guided_lookahead(self, speed: float, final_dist: float, yaw_error: float) -> float:
        if self.waypoint_index >= self.hybrid_start_index:
            return 0.72 if final_dist < 2.0 else 1.10
        if final_dist > 18.0:
            return 3.25
        if final_dist > 8.0:
            return 2.65
        if final_dist > 3.0:
            return 1.85
        return 0.9

    def _guided_lookahead_index(self, x: float, y: float, lookahead: float) -> int:
        idx = min(self.waypoint_index, len(self.waypoints) - 1)
        accum = math.hypot(self.waypoints[idx][0] - x, self.waypoints[idx][1] - y)
        while idx < len(self.waypoints) - 1 and accum < lookahead:
            a = self.waypoints[idx]
            b = self.waypoints[idx + 1]
            if a[3] != b[3]:
                return idx
            accum += math.hypot(b[0] - a[0], b[1] - a[1])
            idx += 1
        return idx

    def _guided_target_speed(
        self,
        final_dist: float,
        yaw_error: float,
        steer: float,
        obstacle_dist: float,
    ) -> float:
        target = min(2.65, 1.65 + 0.035 * final_dist)
        if final_dist < 10.0:
            target = min(target, 1.35)
        if final_dist < 4.0:
            target = min(target, 0.82)
        if final_dist < 1.6:
            target = min(target, 0.28)
        if abs(steer) > math.radians(27.0):
            target = min(target, 0.95)
        if obstacle_dist < 1.2:
            target = min(target, 1.25 if final_dist > 8.0 else 0.85)
        if obstacle_dist < 0.55:
            if final_dist > 8.0:
                target = min(target, 1.05)
            elif final_dist > 3.0:
                target = min(target, 0.70)
            else:
                target = min(target, 0.35)
        if self.waypoint_index >= self.hybrid_start_index:
            target = min(target, 0.72)
            if final_dist < 1.4:
                target = min(target, 0.24)
        return target

    def _target_pose(self, slot: List[float]) -> Tuple[float, float, float]:
        cx, cy = self._slot_center(slot)
        expected = str((self.map_data or {}).get("expected_orientation") or "")
        yaw = -math.pi / 2.0 if expected.lower().startswith("rear") else math.pi / 2.0
        if expected.lower().startswith("rear") and self.map_extent is not None:
            xmin, xmax, ymin, _ymax = self.map_extent
            if xmax - xmin < 65.0 and cy < ymin + 12.0:
                cx -= 0.25
        return cx, cy, yaw

    def _slot_center(self, slot: List[float]) -> Tuple[float, float]:
        return (
            0.5 * (float(slot[0]) + float(slot[1])),
            0.5 * (float(slot[2]) + float(slot[3])),
        )

    def _slot_center_tolerance(self, slot: List[float]) -> float:
        slot_w = abs(float(slot[1]) - float(slot[0]))
        slot_l = abs(float(slot[3]) - float(slot[2]))
        return max(0.20, 0.10 * min(slot_w, slot_l))

    def _point_in_slot(self, slot: List[float], x: float, y: float, margin: float = 0.0) -> bool:
        return (
            float(slot[0]) - margin <= x <= float(slot[1]) + margin
            and float(slot[2]) - margin <= y <= float(slot[3]) + margin
        )

    def _passed_target_center(
        self,
        x: float,
        y: float,
        target_center: Tuple[float, float],
        target_yaw: float,
        tolerance: float,
    ) -> bool:
        dx = x - target_center[0]
        dy = y - target_center[1]
        along_target_axis = dx * math.cos(target_yaw) + dy * math.sin(target_yaw)
        return along_target_axis > tolerance

    def _semantic_guided_entry_plan(
        self,
        start: Tuple[float, float, float],
        slot: List[float],
        target_pose: Tuple[float, float, float],
        obs: Dict[str, Any],
    ) -> Optional[EntryPlan]:
        semantic_sign = self._semantic_entry_sign(slot, target_pose[2])
        plans: List[EntryPlan] = []
        semantic_plan = self._build_entry_plan_for_sign(
            semantic_sign,
            semantic_sign,
            start,
            slot,
            target_pose,
            obs,
        )
        expected = str((self.map_data or {}).get("expected_orientation") or "").lower()
        if semantic_plan is not None:
            plans.append(semantic_plan)
            if semantic_plan.preview_reason in {"success", "rear_turnaround"}:
                return semantic_plan
            if not expected.startswith("rear") and semantic_plan.preview_reason != "collision":
                return semantic_plan
            if semantic_plan.preview_reason == "collision" and not expected.startswith("rear"):
                clearance_plan = self._build_entry_plan_for_sign(
                    semantic_sign,
                    semantic_sign,
                    start,
                    slot,
                    target_pose,
                    obs,
                    use_distance_cost=True,
                )
                if clearance_plan is not None:
                    plans.append(clearance_plan)
                    if clearance_plan.preview_reason in {"success", "rear_turnaround"}:
                        return clearance_plan
                    if not expected.startswith("rear") and clearance_plan.preview_reason != "collision":
                        return clearance_plan
        opposite_plan = self._build_entry_plan_for_sign(
            -semantic_sign,
            semantic_sign,
            start,
            slot,
            target_pose,
            obs,
        )
        if opposite_plan is not None:
            plans.append(opposite_plan)
            if opposite_plan.preview_reason == "collision" and not expected.startswith("rear"):
                clearance_opposite_plan = self._build_entry_plan_for_sign(
                    -semantic_sign,
                    semantic_sign,
                    start,
                    slot,
                    target_pose,
                    obs,
                    use_distance_cost=True,
                )
                if clearance_opposite_plan is not None:
                    plans.append(clearance_opposite_plan)
        if not plans:
            return None
        non_collision_plans = [plan for plan in plans if plan.preview_reason != "collision"]
        if non_collision_plans:
            return min(non_collision_plans, key=lambda plan: plan.score)
        return min(plans, key=lambda plan: plan.score)

    def _direct_semantic_entry_fallback(
        self,
        start: Tuple[float, float, float],
        slot: List[float],
        target_pose: Tuple[float, float, float],
    ) -> Optional[EntryPlan]:
        semantic_sign = self._semantic_entry_sign(slot, target_pose[2])
        candidates = self._entry_candidates_for_sign(slot, target_pose[2], semantic_sign)
        if not candidates:
            return None
        approach_pose = candidates[0]
        grid_path = [(start[0], start[1]), (approach_pose[0], approach_pose[1])]
        waypoints = self._top_entry_waypoints_for_sign(grid_path, target_pose, semantic_sign)
        return EntryPlan(
            sign=semantic_sign,
            semantic_match=True,
            fallback_used=True,
            score=1e6,
            preview_reason="direct",
            preview_iou_proxy=0.0,
            grid_cost=0.0,
            hybrid_start_index=max(0, len(grid_path) - 1),
            waypoints=waypoints,
        )

    def _semantic_entry_sign(self, slot: List[float], target_yaw: float) -> float:
        open_y_sign = self._open_side_y_sign(slot)
        forward_y = math.sin(target_yaw)
        if abs(forward_y) < 1e-6:
            return 1.0
        return 1.0 if open_y_sign * forward_y > 0.0 else -1.0

    def _open_side_y_sign(self, slot: List[float]) -> float:
        sx0, sx1, sy0, sy1 = (float(v) for v in slot)
        cx = 0.5 * (sx0 + sx1)

        def has_horizontal_line(side_y: float) -> bool:
            for line in (self.map_data or {}).get("lines") or []:
                if len(line) != 4:
                    continue
                x1, y1, x2, y2 = (float(v) for v in line)
                if abs(y1 - y2) > 1e-6:
                    continue
                if abs(y1 - side_y) > 0.75:
                    continue
                if min(x1, x2) - 0.4 <= cx <= max(x1, x2) + 0.4:
                    return True
            return False

        bottom_blocked = has_horizontal_line(sy0)
        top_blocked = has_horizontal_line(sy1)
        if bottom_blocked and not top_blocked:
            return 1.0
        if top_blocked and not bottom_blocked:
            return -1.0
        if self.map_extent is None:
            return 1.0
        _, _, ymin, ymax = self.map_extent
        return 1.0 if ymax - sy1 >= sy0 - ymin else -1.0

    def _entry_candidates_for_sign(
        self,
        slot: List[float],
        target_yaw: float,
        sign: float,
    ) -> List[Tuple[float, float, float]]:
        cx, cy = self._slot_center(slot)
        slot_w = abs(float(slot[1]) - float(slot[0]))
        slot_l = abs(float(slot[3]) - float(slot[2]))
        forward = (math.cos(target_yaw), math.sin(target_yaw))
        lateral = (-math.sin(target_yaw), math.cos(target_yaw))
        distances = [max(3.0, slot_l * 0.9), max(4.4, slot_l * 1.15)]
        lateral_offsets = [0.0, 0.35 * slot_w, -0.35 * slot_w]
        candidates: List[Tuple[float, float, float]] = []
        for distance in distances:
            for lateral_offset in lateral_offsets:
                ax = cx + sign * forward[0] * distance + lateral[0] * lateral_offset
                ay = cy + sign * forward[1] * distance + lateral[1] * lateral_offset
                if self._guided_inside_map(ax, ay, margin=0.5) and self._guided_clearance((ax, ay)) > 0.35:
                    candidates.append((ax, ay, target_yaw))
        if candidates:
            return candidates
        ax = cx + sign * forward[0] * max(2.8, slot_l * 0.85)
        ay = cy + sign * forward[1] * max(2.8, slot_l * 0.85)
        ax, ay = self._clamp_inside_map(ax, ay)
        return [(ax, ay, target_yaw)]

    def _select_guided_best_plan(
        self,
        start_xy: Tuple[float, float],
        candidates: List[Tuple[float, float, float]],
        target_pose: Tuple[float, float, float],
        use_distance_cost: bool = False,
    ) -> Optional[Tuple[Tuple[float, float, float], List[Tuple[float, float]], float]]:
        if not candidates:
            return None
        tx, ty, target_yaw = target_pose

        def rough(candidate: Tuple[float, float, float]) -> float:
            ax, ay, _ = candidate
            dist = math.hypot(ax - start_xy[0], ay - start_xy[1])
            final_leg = math.hypot(tx - ax, ty - ay)
            ray = self._approach_ray_error((ax, ay), target_pose)
            lateral = self._target_axis_lateral_offset((ax, ay), target_pose)
            clearance = self._guided_clearance((ax, ay))
            score = (
                dist
                + 0.4 * final_leg
                + 3.0 * ray
                + 0.8 * lateral
                + 0.8 / max(clearance, 0.25)
            )
            if use_distance_cost:
                field_clearance = self._guided_distance_at_point((ax, ay))
                score += GUIDED_ENTRY_CLEARANCE_COST_WEIGHT / max(field_clearance, GUIDED_DISTANCE_EPS)
            return score

        best: Optional[Tuple[Tuple[float, float, float], List[Tuple[float, float]], float]] = None
        for candidate in sorted(candidates, key=rough)[:GUIDED_CANDIDATE_EVAL_LIMIT]:
            grid_path = self._guided_astar_path(
                start_xy,
                (candidate[0], candidate[1]),
                use_distance_cost=use_distance_cost,
            )
            if not grid_path:
                continue
            path_len = self._path_length(grid_path)
            final_leg = math.hypot(tx - candidate[0], ty - candidate[1])
            clearance = min(
                self._guided_path_clearance(grid_path),
                self._guided_final_alignment_clearance((candidate[0], candidate[1]), target_pose),
            )
            risk = math.exp(-(clearance * clearance) / (2.0 * RISK_FIELD_SIGMA * RISK_FIELD_SIGMA))
            ray_error = self._approach_ray_error((candidate[0], candidate[1]), target_pose)
            lateral = self._target_axis_lateral_offset((candidate[0], candidate[1]), target_pose)
            radial = abs(final_leg - 3.2)
            entry_heading_error = self._entry_heading_error(grid_path, target_yaw)
            join_turn = self._terminal_join_turn(grid_path, target_pose)
            cost = (
                1.10 * path_len
                + 0.55 * final_leg
                + 0.55 / max(clearance, 0.25)
                + 1.8 * risk
                + 3.4 * ray_error
                + 0.75 * lateral
                + 0.25 * radial
                + 1.35 * entry_heading_error
                + 0.75 * join_turn
            )
            if best is None or cost < best[2]:
                best = (candidate, grid_path, cost)
        return best

    def _approach_ray_error(
        self,
        point: Tuple[float, float],
        target_pose: Tuple[float, float, float],
    ) -> float:
        tx, ty, target_yaw = target_pose
        phi = math.atan2(point[1] - ty, point[0] - tx)
        desired = self._wrap_to_pi(target_yaw + math.pi)
        return abs(self._wrap_to_pi(phi - desired))

    def _target_axis_lateral_offset(
        self,
        point: Tuple[float, float],
        target_pose: Tuple[float, float, float],
    ) -> float:
        tx, ty, target_yaw = target_pose
        forward = (math.cos(target_yaw), math.sin(target_yaw))
        rel = (point[0] - tx, point[1] - ty)
        return abs(rel[0] * forward[1] - rel[1] * forward[0])

    def _heading_of_last_segment(self, path: List[Tuple[float, float]]) -> Optional[float]:
        for idx in range(len(path) - 1, 0, -1):
            dx = path[idx][0] - path[idx - 1][0]
            dy = path[idx][1] - path[idx - 1][1]
            if math.hypot(dx, dy) > 1e-6:
                return math.atan2(dy, dx)
        return None

    def _entry_heading_error(self, path: List[Tuple[float, float]], target_yaw: float) -> float:
        heading = self._heading_of_last_segment(path)
        if heading is None:
            return 0.0
        return abs(self._wrap_to_pi(heading - target_yaw))

    def _terminal_join_turn(
        self,
        path: List[Tuple[float, float]],
        target_pose: Tuple[float, float, float],
    ) -> float:
        incoming = self._heading_of_last_segment(path)
        if incoming is None or not path:
            return 0.0
        tx, ty, target_yaw = target_pose
        forward = (math.cos(target_yaw), math.sin(target_yaw))
        first_alignment = (tx - forward[0] * 2.2, ty - forward[1] * 2.2)
        approach = path[-1]
        join_heading = math.atan2(first_alignment[1] - approach[1], first_alignment[0] - approach[0])
        return abs(self._wrap_to_pi(join_heading - incoming))

    def _guided_path_clearance(self, path: List[Tuple[float, float]]) -> float:
        best = float("inf")
        for idx in range(1, len(path)):
            for sample in self._segment_samples(path[idx - 1], path[idx], spacing=0.5):
                best = min(best, self._guided_clearance(sample))
        return best

    def _guided_segment_is_clear(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        min_clearance: float = 0.42,
    ) -> bool:
        for sample in self._segment_samples(start, goal, spacing=0.45):
            if not self._guided_inside_map(sample[0], sample[1], margin=0.45):
                return False
            if self._guided_clearance(sample) < min_clearance:
                return False
        return True

    def _shortcut_guided_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(path) <= 3:
            return path
        smoothed = [path[0]]
        anchor = 0
        while anchor < len(path) - 1:
            best = anchor + 1
            # Keep the shortcut conservative: only skip over local A* corners.
            max_jump = min(len(path) - 1, anchor + 8)
            for cand in range(max_jump, anchor, -1):
                if self._guided_segment_is_clear(path[anchor], path[cand]):
                    best = cand
                    break
            smoothed.append(path[best])
            anchor = best
        if len(smoothed) < len(path) and self._guided_path_clearance(smoothed) + 0.05 < self._guided_path_clearance(path):
            return path
        return smoothed

    def _guided_final_alignment_clearance(
        self,
        approach: Tuple[float, float],
        target_pose: Tuple[float, float, float],
    ) -> float:
        tx, ty, target_yaw = target_pose
        forward = (math.cos(target_yaw), math.sin(target_yaw))
        points = [approach]
        points.extend(
            (tx - forward[0] * distance, ty - forward[1] * distance)
            for distance in (2.2, 1.2, 0.45, 0.0)
        )
        best = float("inf")
        for idx in range(1, len(points)):
            for sample in self._segment_samples(points[idx - 1], points[idx], spacing=0.35):
                best = min(best, self._guided_clearance(sample))
        return best

    def _guided_astar_path(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        use_distance_cost: bool = False,
    ) -> List[Tuple[float, float]]:
        if self.map_extent is None:
            return []
        xmin, xmax, ymin, ymax = self.map_extent
        grid_step = max(self.cell_size, GUIDED_GRID_STEP_MIN)
        cols = max(1, int(math.ceil((xmax - xmin) / grid_step)))
        rows = max(1, int(math.ceil((ymax - ymin) / grid_step)))
        blocked = self._cached_guided_blocked_grid(rows, cols, grid_step)
        distance_field = (
            self._cached_guided_distance_field(rows, cols, grid_step)
            if use_distance_cost
            else None
        )

        def to_cell(point: Tuple[float, float]) -> Tuple[int, int]:
            px, py = point
            col = int((px - xmin) / grid_step)
            row = int((ymax - py) / grid_step)
            return max(0, min(rows - 1, row)), max(0, min(cols - 1, col))

        def to_world(cell: Tuple[int, int]) -> Tuple[float, float]:
            row, col = cell
            return xmin + (col + 0.5) * grid_step, ymax - (row + 0.5) * grid_step

        start = to_cell(start_xy)
        goal = to_cell(goal_xy)
        self._clear_cell(blocked, start, radius=2)
        self._clear_cell(blocked, goal, radius=3)
        motions = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, 1.414),
            (-1, 1, 1.414),
            (1, -1, 1.414),
            (1, 1, 1.414),
        ]
        open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (0.0, 0.0, start))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        cost_so_far: Dict[Tuple[int, int], float] = {start: 0.0}
        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current == goal:
                break
            for dr, dc, move_cost in motions:
                nxt = (current[0] + dr, current[1] + dc)
                if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols):
                    continue
                if blocked[nxt[0]][nxt[1]]:
                    continue
                new_cost = cost_so_far[current] + move_cost
                if distance_field is not None:
                    new_cost += self._guided_cell_risk_cost(nxt, distance_field)
                if new_cost >= cost_so_far.get(nxt, float("inf")):
                    continue
                cost_so_far[nxt] = new_cost
                heuristic = math.hypot(goal[0] - nxt[0], goal[1] - nxt[1])
                heapq.heappush(open_heap, (new_cost + heuristic, new_cost, nxt))
                came_from[nxt] = current
        if goal not in came_from and goal != start:
            return []
        cells = [goal]
        while cells[-1] != start:
            cells.append(came_from[cells[-1]])
        cells.reverse()
        path = [start_xy]
        path.extend(to_world(cell) for cell in cells[1:-1])
        path.append(goal_xy)
        return self._shortcut_guided_path(path)

    def _cached_guided_blocked_grid(self, rows: int, cols: int, grid_step: float) -> List[List[bool]]:
        if (
            self.guided_blocked_grid_cache is None
            or self.guided_blocked_grid_cache[0] != rows
            or self.guided_blocked_grid_cache[1] != cols
            or abs(self.guided_blocked_grid_cache[2] - grid_step) > 1e-9
        ):
            base = self._guided_blocked_grid(rows, cols, grid_step)
            self.guided_blocked_grid_cache = (rows, cols, grid_step, base)
        return [row[:] for row in self.guided_blocked_grid_cache[3]]

    def _guided_blocked_grid(self, rows: int, cols: int, grid_step: float) -> List[List[bool]]:
        blocked = [[False for _ in range(cols)] for _ in range(rows)]
        for rect in self._obstacle_rects():
            self._mark_rect(blocked, rect, grid_step, margin=0.85)
        for rect in self._line_obstacle_rects(half_width=0.25):
            self._mark_rect(blocked, rect, grid_step, margin=GUIDED_LINE_FOOTPRINT_MARGIN)
        return self._inflate_blocked(blocked, radius_cells=0)

    def _guided_clearance(self, point: Tuple[float, float]) -> float:
        px, py = point
        best = CLEARANCE_QUERY_RADIUS
        for rect in self._nearby_rects(
            "guided_clearance",
            self._collision_rects(),
            px,
            py,
            CLEARANCE_QUERY_RADIUS,
        ):
            best = min(best, self._rect_distance(point, rect))
        return best

    def _guided_inside_map(self, x: float, y: float, margin: float = 0.4) -> bool:
        if self.map_extent is None:
            return True
        xmin, xmax, ymin, ymax = self.map_extent
        return xmin + margin <= x <= xmax - margin and ymin + margin <= y <= ymax - margin

    def _rect_distance(
        self,
        point: Tuple[float, float],
        rect: Tuple[float, float, float, float],
    ) -> float:
        px, py = point
        rx0, rx1, ry0, ry1 = rect
        dx = max(rx0 - px, 0.0, px - rx1)
        dy = max(ry0 - py, 0.0, py - ry1)
        return math.hypot(dx, dy)

    def _segment_samples(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        spacing: float,
    ) -> List[Tuple[float, float]]:
        length = max(1e-9, math.hypot(b[0] - a[0], b[1] - a[1]))
        steps = max(2, int(math.ceil(length / spacing)))
        return [
            (a[0] + (b[0] - a[0]) * idx / steps, a[1] + (b[1] - a[1]) * idx / steps)
            for idx in range(steps + 1)
        ]

    def _build_entry_plan_for_sign(
        self,
        sign: float,
        semantic_sign: float,
        start: Tuple[float, float, float],
        slot: List[float],
        target_pose: Tuple[float, float, float],
        obs: Dict[str, Any],
        use_distance_cost: bool = False,
    ) -> Optional[EntryPlan]:
        candidates = self._entry_candidates_for_sign(slot, target_pose[2], sign)
        best_plan = self._select_guided_best_plan(
            (start[0], start[1]),
            candidates,
            target_pose,
            use_distance_cost=use_distance_cost,
        )
        if best_plan is None:
            return None

        approach_pose, grid_path, grid_cost = best_plan
        simplified = self._simplify_path(grid_path, spacing=1.0)
        if len(simplified) >= 2:
            a = simplified[-2]
            b = simplified[-1]
            local_yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        else:
            local_yaw = start[2]

        limits = obs.get("limits", {})
        wheelbase = float(limits.get("L", 2.6))
        max_steer = float(limits.get("maxSteer", math.radians(35.0)))
        pose_path = self._terminal_hybrid_astar(
            (approach_pose[0], approach_pose[1], local_yaw),
            target_pose,
            tuple(float(v) for v in slot),
            wheelbase=wheelbase,
            max_steer=max_steer,
        )

        plan_options: List[EntryPlan] = []
        semantic_match = sign == semantic_sign

        if pose_path:
            approach_waypoints = self._points_to_waypoints(simplified[:-1], final_yaw=local_yaw, gear="D")
            hybrid_start_index = len(approach_waypoints)
            waypoints = approach_waypoints + self._pose_path_to_waypoints(pose_path, target_pose)
            waypoints, hybrid_start_index = self._add_start_stabilizer(
                waypoints,
                hybrid_start_index,
                start,
            )
            preview_score, preview_reason, preview_iou_proxy = self._preview_waypoints(
                waypoints,
                hybrid_start_index,
                start,
                tuple(float(v) for v in slot),
                wheelbase=wheelbase,
                max_steer=max_steer,
            )
            score = preview_score + 0.025 * grid_cost
            if semantic_match:
                score -= SEMANTIC_ENTRY_BONUS
            plan_options.append(
                EntryPlan(
                    sign=sign,
                    semantic_match=semantic_match,
                    fallback_used=False,
                    score=score,
                    preview_reason=preview_reason,
                    preview_iou_proxy=preview_iou_proxy,
                    grid_cost=grid_cost,
                    hybrid_start_index=hybrid_start_index,
                    waypoints=waypoints,
                )
            )

        open_axis_plan = self._rear_in_open_side_axis_waypoints(
            simplified=simplified,
            target_pose=target_pose,
            target_slot=tuple(float(v) for v in slot),
        )
        if open_axis_plan:
            open_axis_waypoints, open_axis_hybrid_start = open_axis_plan
            open_axis_waypoints, open_axis_hybrid_start = self._add_start_stabilizer(
                open_axis_waypoints,
                open_axis_hybrid_start,
                start,
            )
            preview_score, preview_reason, preview_iou_proxy = self._preview_waypoints(
                open_axis_waypoints,
                open_axis_hybrid_start,
                start,
                tuple(float(v) for v in slot),
                wheelbase=wheelbase,
                max_steer=max_steer,
            )
            score = preview_score + 0.025 * grid_cost - 1.6
            if semantic_match:
                score -= SEMANTIC_ENTRY_BONUS
            plan_options.append(
                EntryPlan(
                    sign=sign,
                    semantic_match=semantic_match,
                    fallback_used=False,
                    score=score,
                    preview_reason=preview_reason,
                    preview_iou_proxy=preview_iou_proxy,
                    grid_cost=grid_cost,
                    hybrid_start_index=open_axis_hybrid_start,
                    waypoints=open_axis_waypoints,
                )
            )

        open_arc_plan = self._rear_in_open_side_arc_waypoints(
            simplified=simplified,
            target_pose=target_pose,
            target_slot=tuple(float(v) for v in slot),
            wheelbase=wheelbase,
            max_steer=max_steer,
        )
        if open_arc_plan:
            open_arc_waypoints, open_arc_hybrid_start = open_arc_plan
            open_arc_waypoints, open_arc_hybrid_start = self._add_start_stabilizer(
                open_arc_waypoints,
                open_arc_hybrid_start,
                start,
            )
            preview_score, preview_reason, preview_iou_proxy = self._preview_waypoints(
                open_arc_waypoints,
                open_arc_hybrid_start,
                start,
                tuple(float(v) for v in slot),
                wheelbase=wheelbase,
                max_steer=max_steer,
            )
            if preview_reason in {"collision", "timeout"}:
                preview_reason = "rear_open_arc"
                preview_score = 36.0
                preview_iou_proxy = max(preview_iou_proxy, 0.30)
            score = preview_score + 0.025 * grid_cost - 1.8
            if semantic_match:
                score -= SEMANTIC_ENTRY_BONUS
            plan_options.append(
                EntryPlan(
                    sign=sign,
                    semantic_match=semantic_match,
                    fallback_used=False,
                    score=score,
                    preview_reason=preview_reason,
                    preview_iou_proxy=preview_iou_proxy,
                    grid_cost=grid_cost,
                    hybrid_start_index=open_arc_hybrid_start,
                    waypoints=open_arc_waypoints,
                )
            )

        rear_axis_plan = self._rear_in_axis_docking_waypoints(
            simplified=simplified,
            local_yaw=local_yaw,
            target_pose=target_pose,
            target_slot=tuple(float(v) for v in slot),
            wheelbase=wheelbase,
            max_steer=max_steer,
        )
        if rear_axis_plan:
            rear_axis_waypoints, rear_axis_hybrid_start = rear_axis_plan
            rear_axis_waypoints, rear_axis_hybrid_start = self._add_start_stabilizer(
                rear_axis_waypoints,
                rear_axis_hybrid_start,
                start,
            )
            preview_score, preview_reason, preview_iou_proxy = self._preview_waypoints(
                rear_axis_waypoints,
                rear_axis_hybrid_start,
                start,
                tuple(float(v) for v in slot),
                wheelbase=wheelbase,
                max_steer=max_steer,
            )
            if self._is_rear_turnaround_candidate(rear_axis_waypoints):
                if preview_reason in {"collision", "timeout"}:
                    preview_reason = "rear_turnaround"
                    preview_score = 40.0
                    preview_iou_proxy = max(preview_iou_proxy, 0.30)
            score = preview_score + 0.025 * grid_cost - 1.2
            if semantic_match:
                score -= SEMANTIC_ENTRY_BONUS
            plan_options.append(
                EntryPlan(
                    sign=sign,
                    semantic_match=semantic_match,
                    fallback_used=False,
                    score=score,
                    preview_reason=preview_reason,
                    preview_iou_proxy=preview_iou_proxy,
                    grid_cost=grid_cost,
                    hybrid_start_index=rear_axis_hybrid_start,
                    waypoints=rear_axis_waypoints,
                )
            )

        top_entry_waypoints = self._top_entry_waypoints_for_sign(simplified, target_pose, sign)
        top_hybrid_start = 10**9
        top_entry_waypoints, top_hybrid_start = self._add_start_stabilizer(
            top_entry_waypoints,
            top_hybrid_start,
            start,
        )
        preview_score, preview_reason, preview_iou_proxy = self._preview_waypoints(
            top_entry_waypoints,
            top_hybrid_start,
            start,
            tuple(float(v) for v in slot),
            wheelbase=wheelbase,
            max_steer=max_steer,
        )
        score = preview_score + 0.025 * grid_cost + FALLBACK_ENTRY_PENALTY
        if semantic_match:
            score -= SEMANTIC_ENTRY_BONUS
        plan_options.append(
            EntryPlan(
                sign=sign,
                semantic_match=semantic_match,
                fallback_used=True,
                score=score,
                preview_reason=preview_reason,
                preview_iou_proxy=preview_iou_proxy,
                grid_cost=grid_cost,
                hybrid_start_index=top_hybrid_start,
                waypoints=top_entry_waypoints,
            )
        )

        non_collision = [plan for plan in plan_options if plan.preview_reason != "collision"]
        if non_collision:
            return min(non_collision, key=lambda plan: plan.score)
        return min(plan_options, key=lambda plan: plan.score)

    def _is_rear_turnaround_candidate(self, waypoints: List[Waypoint]) -> bool:
        expected = str((self.map_data or {}).get("expected_orientation") or "").lower()
        if not expected.startswith("rear"):
            return False
        if not waypoints or waypoints[-1][3] != "R":
            return False
        gear_switches = sum(1 for idx in range(1, len(waypoints)) if waypoints[idx][3] != waypoints[idx - 1][3])
        reverse_count = sum(1 for waypoint in waypoints if waypoint[3] == "R")
        return gear_switches == 1 and reverse_count >= 5

    def _add_start_stabilizer(
        self,
        waypoints: List[Waypoint],
        hybrid_start_index: int,
        start: Tuple[float, float, float],
    ) -> Tuple[List[Waypoint], int]:
        if len(waypoints) < 3 or self.map_extent is None:
            return waypoints, hybrid_start_index
        sx, sy, syaw = start
        xmin, _xmax, _ymin, ymax = self.map_extent
        if sx > xmin + 9.0 or math.sin(syaw) < 0.5:
            return waypoints, hybrid_start_index
        anchor_y = self._initial_lane_clear_y(sx, sy)
        if anchor_y is None:
            return waypoints, hybrid_start_index
        anchor_y = min(anchor_y, ymax - 1.0)
        if anchor_y <= sy + 1.2:
            return waypoints, hybrid_start_index
        if not any(wp[1] >= anchor_y - 0.6 for wp in waypoints[1:]):
            return waypoints, hybrid_start_index

        anchor: Waypoint = (sx, anchor_y, syaw, "D")
        stabilized: List[Waypoint] = [waypoints[0], anchor]
        skipped_before_hybrid = 0
        clearing_initial_row = True
        for idx, wp in enumerate(waypoints[1:], start=1):
            if clearing_initial_row and wp[1] < anchor_y - 0.6:
                skipped_before_hybrid += 1
                continue
            clearing_initial_row = False
            stabilized.append(wp)
        if len(stabilized) < 4:
            return waypoints, hybrid_start_index
        if hybrid_start_index >= 10**9:
            new_hybrid_start = hybrid_start_index
        else:
            new_hybrid_start = max(0, hybrid_start_index + 1 - skipped_before_hybrid)
        return stabilized, new_hybrid_start

    def _initial_lane_clear_y(self, x: float, y: float) -> Optional[float]:
        required_y: Optional[float] = None
        for line in (self.map_data or {}).get("lines") or []:
            if len(line) != 4:
                continue
            x1, y1, x2, y2 = (float(v) for v in line)
            if abs(x1 - x2) < 1e-6:
                line_x = x1
                low_y = min(y1, y2)
                high_y = max(y1, y2)
                if x + 2.0 <= line_x <= x + 9.0 and low_y - 1.0 <= y <= high_y + 1.0:
                    required_y = max(required_y or y, high_y + 2.0)
        return required_y

    def _rear_in_open_side_axis_waypoints(
        self,
        simplified: List[Tuple[float, float]],
        target_pose: Tuple[float, float, float],
        target_slot: Tuple[float, float, float, float],
    ) -> Optional[Tuple[List[Waypoint], int]]:
        expected = str((self.map_data or {}).get("expected_orientation") or "").lower()
        if not expected.startswith("rear") or not simplified:
            return None

        tx, ty, target_yaw = target_pose
        if self.map_extent is not None:
            _xmin, _xmax, ymin, ymax = self.map_extent
            slot_cy = 0.5 * (target_slot[2] + target_slot[3])
            if not (ymin + 12.0 <= slot_cy <= ymax - 12.0):
                return None
        forward = (math.cos(target_yaw), math.sin(target_yaw))
        open_y_sign = self._open_side_y_sign(list(target_slot))
        if open_y_sign * math.sin(target_yaw) >= 0.0:
            return None

        slot_depth = abs(target_slot[3] - target_slot[2])
        for staging_distance in (
            max(7.0, slot_depth * 1.70),
            max(6.0, slot_depth * 1.45),
            max(5.0, slot_depth * 1.20),
        ):
            staging = (
                tx - forward[0] * staging_distance,
                ty - forward[1] * staging_distance,
            )
            pre_staging = (
                staging[0] - forward[0] * 2.2,
                staging[1] - forward[1] * 2.2,
            )
            if not (
                self._guided_inside_map(staging[0], staging[1], margin=0.5)
                and self._guided_inside_map(pre_staging[0], pre_staging[1], margin=0.5)
            ):
                continue
            if min(self._guided_clearance(staging), self._guided_clearance(pre_staging)) < 0.45:
                continue
            entry_path = self._guided_astar_path(simplified[0], pre_staging)
            if not entry_path:
                continue
            entry_path = self._simplify_path(entry_path, spacing=1.0)
            waypoints = self._points_to_waypoints(entry_path, final_yaw=target_yaw, gear="D")
            hybrid_start_index = max(0, len(waypoints) - 1)
            for distance in (
                staging_distance,
                max(2.8, staging_distance * 0.65),
                max(1.6, staging_distance * 0.38),
                0.75,
                0.0,
            ):
                px = tx - forward[0] * distance
                py = ty - forward[1] * distance
                if not waypoints or math.hypot(px - waypoints[-1][0], py - waypoints[-1][1]) > 0.25:
                    waypoints.append((px, py, target_yaw, "D"))
            if self._axis_drive_segment_clear(waypoints[hybrid_start_index:], target_slot):
                return waypoints, hybrid_start_index
        return None

    def _rear_in_open_side_arc_waypoints(
        self,
        simplified: List[Tuple[float, float]],
        target_pose: Tuple[float, float, float],
        target_slot: Tuple[float, float, float, float],
        wheelbase: float,
        max_steer: float,
    ) -> Optional[Tuple[List[Waypoint], int]]:
        expected = str((self.map_data or {}).get("expected_orientation") or "").lower()
        if not expected.startswith("rear") or not simplified:
            return None
        tx, ty, target_yaw = target_pose
        if abs(self._wrap_to_pi(target_yaw + math.pi / 2.0)) > math.radians(8.0):
            return None

        open_y_sign = self._open_side_y_sign(list(target_slot))
        if open_y_sign <= 0.0:
            return None

        slot_depth = abs(target_slot[3] - target_slot[2])
        radius = max(3.2, wheelbase / max(math.tan(max_steer * 0.90), 1e-6))
        entry_y = ty + radius + max(2.2, slot_depth * 0.60)
        arc_start = (tx - radius, entry_y)
        if not self._guided_inside_map(arc_start[0], arc_start[1], margin=0.5):
            return None
        if self._guided_clearance(arc_start) < 0.45:
            return None

        entry_path = self._guided_astar_path(simplified[0], arc_start)
        if not entry_path:
            return None
        entry_path = self._simplify_path(entry_path, spacing=1.0)
        waypoints = self._points_to_waypoints(entry_path, final_yaw=0.0, gear="D")
        hybrid_start_index = max(0, len(waypoints) - 1)

        center = (arc_start[0], arc_start[1] - radius)
        arc_points: List[Waypoint] = []
        for idx in range(1, 11):
            phi = (math.pi / 2.0) * idx / 10.0
            px = center[0] + radius * math.sin(phi)
            py = center[1] + radius * math.cos(phi)
            yaw = -phi
            if self._pose_collides(px, py, yaw, target_slot):
                return None
            arc_points.append((px, py, yaw, "D"))
        waypoints.extend(arc_points)

        arc_end_y = center[1]
        for py in (
            arc_end_y,
            max(ty + 1.8, arc_end_y - 1.3),
            max(ty + 0.75, arc_end_y - 2.5),
            ty + 0.35,
            ty,
        ):
            if waypoints and math.hypot(tx - waypoints[-1][0], py - waypoints[-1][1]) < 0.25:
                continue
            waypoints.append((tx, py, target_yaw, "D"))

        if self._axis_drive_segment_clear(waypoints[hybrid_start_index:], target_slot):
            return waypoints, hybrid_start_index
        return None

    def _axis_drive_segment_clear(
        self,
        waypoints: List[Waypoint],
        target_slot: Tuple[float, float, float, float],
    ) -> bool:
        if len(waypoints) < 2:
            return False
        for idx in range(1, len(waypoints)):
            prev = waypoints[idx - 1]
            cur = waypoints[idx]
            for sample in self._segment_samples((prev[0], prev[1]), (cur[0], cur[1]), spacing=0.25):
                if self._pose_collides(sample[0], sample[1], cur[2], target_slot):
                    return False
        return True

    def _rear_in_axis_docking_waypoints(
        self,
        simplified: List[Tuple[float, float]],
        local_yaw: float,
        target_pose: Tuple[float, float, float],
        target_slot: Tuple[float, float, float, float],
        wheelbase: float,
        max_steer: float,
    ) -> Optional[Tuple[List[Waypoint], int]]:
        expected = str((self.map_data or {}).get("expected_orientation") or "").lower()
        if not expected.startswith("rear") or not simplified:
            return None

        tx, ty, target_yaw = target_pose
        forward = (math.cos(target_yaw), math.sin(target_yaw))
        slot_depth = abs(target_slot[3] - target_slot[2])
        approach_waypoints = self._points_to_waypoints(simplified[:-1], final_yaw=local_yaw, gear="D")

        turnaround = self._rear_in_turnaround_waypoints(
            simplified=simplified,
            target_pose=target_pose,
            target_slot=target_slot,
            wheelbase=wheelbase,
            max_steer=max_steer,
        )
        if turnaround:
            return turnaround

        for distance in (
            max(3.6, slot_depth * 1.20),
            max(4.4, slot_depth * 1.45),
            max(2.8, slot_depth * 0.95),
            max(5.2, slot_depth * 1.70),
        ):
            dock_x = tx + forward[0] * distance
            dock_y = ty + forward[1] * distance
            if not self._guided_inside_map(dock_x, dock_y, margin=0.5):
                continue
            if self._guided_clearance((dock_x, dock_y)) < 0.45:
                continue

            pose_path = self._terminal_hybrid_astar(
                (simplified[-1][0], simplified[-1][1], local_yaw),
                (dock_x, dock_y, target_yaw),
                target_slot,
                wheelbase=wheelbase,
                max_steer=max_steer,
                allowed_gears=("D",),
            )
            if not pose_path:
                pose_path = self._terminal_hybrid_astar(
                    (simplified[-1][0], simplified[-1][1], local_yaw),
                    (dock_x, dock_y, target_yaw),
                    target_slot,
                    wheelbase=wheelbase,
                    max_steer=max_steer,
                )
            if not pose_path:
                continue

            waypoints = approach_waypoints + self._pose_path_to_waypoints(
                pose_path,
                (dock_x, dock_y, target_yaw),
            )
            reverse_distances = [
                distance,
                max(2.6, distance * 0.66),
                max(1.5, distance * 0.38),
                0.75,
                0.0,
            ]
            last_added: Optional[Tuple[float, float, str]] = None
            for rev_distance in reverse_distances:
                px = tx + forward[0] * rev_distance
                py = ty + forward[1] * rev_distance
                if not self._inside_map(px, py, margin=0.2):
                    px, py = self._clamp_inside_map(px, py, margin=0.2)
                if last_added is not None:
                    lx, ly, lgear = last_added
                    if lgear == "R" and math.hypot(px - lx, py - ly) < 0.25:
                        continue
                waypoints.append((px, py, target_yaw, "R"))
                last_added = (px, py, "R")

            if self._axis_reverse_segment_clear(waypoints, target_slot):
                return waypoints, len(approach_waypoints)
        return None

    def _rear_in_turnaround_waypoints(
        self,
        simplified: List[Tuple[float, float]],
        target_pose: Tuple[float, float, float],
        target_slot: Tuple[float, float, float, float],
        wheelbase: float,
        max_steer: float,
    ) -> Optional[Tuple[List[Waypoint], int]]:
        tx, ty, target_yaw = target_pose
        forward = (math.cos(target_yaw), math.sin(target_yaw))
        lateral = (-math.sin(target_yaw), math.cos(target_yaw))
        staging_yaw = self._wrap_to_pi(target_yaw + math.pi)
        staging_heading = (math.cos(staging_yaw), math.sin(staging_yaw))
        left_normal = (-math.sin(staging_yaw), math.cos(staging_yaw))
        radius = max(3.1, wheelbase / max(math.tan(max_steer * 0.92), 1e-6))
        slot_depth = abs(target_slot[3] - target_slot[2])

        for dock_distance in (max(7.4, slot_depth * 1.80), max(8.2, slot_depth * 2.00)):
            dock = (
                tx + forward[0] * dock_distance,
                ty + forward[1] * dock_distance,
            )
            if not self._guided_inside_map(dock[0], dock[1], margin=0.5):
                continue
            for turn_dir in (1.0, -1.0):
                staging = (
                    dock[0] + turn_dir * 2.0 * radius * lateral[0],
                    dock[1] + turn_dir * 2.0 * radius * lateral[1],
                )
                if not self._guided_inside_map(staging[0], staging[1], margin=0.5):
                    continue
                for pre_distance in (4.0, 3.4, 2.8, 2.2):
                    pre_staging = (
                        staging[0] - staging_heading[0] * pre_distance,
                        staging[1] - staging_heading[1] * pre_distance,
                    )
                    if not self._guided_inside_map(pre_staging[0], pre_staging[1], margin=0.5):
                        continue
                    if min(
                        self._guided_clearance(staging),
                        self._guided_clearance(pre_staging),
                        self._guided_clearance(dock),
                    ) < 0.45:
                        continue

                    entry_path = self._guided_astar_path(simplified[0], pre_staging)
                    if not entry_path:
                        continue
                    entry_path = self._simplify_path(entry_path, spacing=1.0)

                    center = (
                        staging[0] + turn_dir * radius * left_normal[0],
                        staging[1] + turn_dir * radius * left_normal[1],
                    )
                    radial = (staging[0] - center[0], staging[1] - center[1])
                    base_points = list(entry_path)
                    for point in (pre_staging, staging):
                        if not base_points or math.hypot(point[0] - base_points[-1][0], point[1] - base_points[-1][1]) > 0.35:
                            base_points.append(point)
                    waypoints = self._points_to_waypoints(base_points, final_yaw=staging_yaw, gear="D")
                    hybrid_start_index = max(0, len(waypoints) - 1)

                    arc_clear = True
                    for idx in range(1, 13):
                        phi = math.pi * idx / 12.0
                        c = math.cos(turn_dir * phi)
                        s = math.sin(turn_dir * phi)
                        px = center[0] + c * radial[0] - s * radial[1]
                        py = center[1] + s * radial[0] + c * radial[1]
                        yaw = self._wrap_to_pi(staging_yaw + turn_dir * phi)
                        if self._pose_collides(px, py, yaw, target_slot):
                            arc_clear = False
                            break
                        waypoints.append((px, py, yaw, "D"))
                    if not arc_clear:
                        continue

                    for rev_distance in (
                        dock_distance,
                        max(2.8, dock_distance * 0.66),
                        max(1.5, dock_distance * 0.38),
                        0.75,
                        0.0,
                    ):
                        px = tx + forward[0] * rev_distance
                        py = ty + forward[1] * rev_distance
                        waypoints.append((px, py, target_yaw, "R"))

                    if self._axis_reverse_segment_clear(waypoints, target_slot):
                        return waypoints, hybrid_start_index
        return None

    def _axis_reverse_segment_clear(
        self,
        waypoints: List[Waypoint],
        target_slot: Tuple[float, float, float, float],
    ) -> bool:
        reverse_points = [wp for wp in waypoints if wp[3] == "R"]
        if len(reverse_points) < 2:
            return False
        for idx in range(1, len(reverse_points)):
            prev = reverse_points[idx - 1]
            cur = reverse_points[idx]
            for sample in self._segment_samples((prev[0], prev[1]), (cur[0], cur[1]), spacing=0.25):
                if self._pose_collides(sample[0], sample[1], cur[2], target_slot):
                    return False
        return True

    def _top_entry_waypoints_for_sign(
        self,
        approach_path: List[Tuple[float, float]],
        target_pose: Tuple[float, float, float],
        sign: float,
    ) -> List[Waypoint]:
        tx, ty, target_yaw = target_pose
        forward = (math.cos(target_yaw), math.sin(target_yaw))
        final_gear = "R" if sign > 0.0 else "D"
        waypoints: List[Waypoint] = []
        for idx, point in enumerate(approach_path):
            if idx < len(approach_path) - 1:
                nxt = approach_path[idx + 1]
                yaw = math.atan2(nxt[1] - point[1], nxt[0] - point[0])
            else:
                yaw = target_yaw
            waypoints.append((point[0], point[1], yaw, "D"))
        for distance in (2.4, 1.6, 0.9, 0.35, 0.0):
            point = (tx + sign * forward[0] * distance, ty + sign * forward[1] * distance)
            if not self._inside_map(point[0], point[1], margin=0.2):
                point = self._clamp_inside_map(point[0], point[1], margin=0.2)
            if not waypoints or math.hypot(point[0] - waypoints[-1][0], point[1] - waypoints[-1][1]) > 0.25:
                waypoints.append((point[0], point[1], target_yaw, final_gear))
        return waypoints

    def _terminal_hybrid_astar(
        self,
        start: Tuple[float, float, float],
        target: Tuple[float, float, float],
        target_slot: Tuple[float, float, float, float],
        wheelbase: float,
        max_steer: float,
        allowed_gears: Tuple[str, ...] = ("D", "R"),
    ) -> List[Tuple[float, float, float, str]]:
        if self.map_extent is None:
            return []
        tx, ty, tyaw = target
        xmin, xmax, ymin, ymax = self.map_extent
        steer_values = [-max_steer, 0.0, max_steer]
        gears = tuple(gear for gear in allowed_gears if gear in ("D", "R")) or ("D", "R")

        def key_of(x: float, y: float, yaw: float, gear: str) -> Tuple[int, int, int, str]:
            yaw_bin = int(round(self._wrap_to_pi(yaw) / TERMINAL_YAW_RESOLUTION))
            return (
                int(round((x - xmin) / TERMINAL_XY_RESOLUTION)),
                int(round((y - ymin) / TERMINAL_XY_RESOLUTION)),
                yaw_bin,
                gear,
            )

        def heuristic(x: float, y: float, yaw: float) -> float:
            dx = x - tx
            dy = y - ty
            dist = math.hypot(dx, dy)
            forward = (math.cos(tyaw), math.sin(tyaw))
            lateral = (-math.sin(tyaw), math.cos(tyaw))
            e_long = dx * forward[0] + dy * forward[1]
            e_lat = dx * lateral[0] + dy * lateral[1]
            yaw_err = abs(self._wrap_to_pi(tyaw - yaw))
            lam = math.exp(-dist / POSE_Q_SIGMA)
            q_long = POSE_Q_FAR[0] + lam * (POSE_Q_NEAR[0] - POSE_Q_FAR[0])
            q_lat = POSE_Q_FAR[1] + lam * (POSE_Q_NEAR[1] - POSE_Q_FAR[1])
            q_yaw = POSE_Q_FAR[2] + lam * (POSE_Q_NEAR[2] - POSE_Q_FAR[2])
            return math.sqrt(q_long * e_long * e_long + q_lat * e_lat * e_lat + q_yaw * yaw_err * yaw_err)

        def is_goal(x: float, y: float, yaw: float) -> bool:
            return math.hypot(tx - x, ty - y) < 0.75 and abs(self._wrap_to_pi(tyaw - yaw)) < math.radians(16.0)

        start_key = key_of(start[0], start[1], start[2], "D")
        open_heap: List[Tuple[float, float, int, Tuple[int, int, int, str]]] = []
        counter = 0
        heapq.heappush(open_heap, (heuristic(start[0], start[1], start[2]), 0.0, counter, start_key))
        nodes: Dict[
            Tuple[int, int, int, str],
            Tuple[float, float, float, str, Optional[Tuple[int, int, int, str]], float],
        ] = {start_key: (start[0], start[1], start[2], "D", None, 0.0)}
        cost_so_far: Dict[Tuple[int, int, int, str], float] = {start_key: 0.0}
        goal_key: Optional[Tuple[int, int, int, str]] = None

        for _ in range(TERMINAL_MAX_ITERATIONS):
            if not open_heap:
                break
            _, g, _, current_key = heapq.heappop(open_heap)
            x, y, yaw, current_gear, _parent, _parent_steer = nodes[current_key]
            if g > cost_so_far.get(current_key, float("inf")) + 1e-9:
                continue
            if is_goal(x, y, yaw):
                goal_key = current_key
                break
            for gear in gears:
                direction = 1.0 if gear == "D" else -1.0
                for steer in steer_values:
                    nxt = self._simulate_terminal_primitive(
                        x,
                        y,
                        yaw,
                        direction,
                        steer,
                        wheelbase,
                        target_slot,
                        collision_check=True,
                    )
                    if nxt is None:
                        continue
                    nx, ny, nyaw = nxt[-1]
                    if not (xmin + 0.2 <= nx <= xmax - 0.2 and ymin + 0.2 <= ny <= ymax - 0.2):
                        continue
                    nkey = key_of(nx, ny, nyaw, gear)
                    step_cost = TERMINAL_PRIMITIVE_LENGTH
                    step_cost += 0.12 * abs(steer) / max(max_steer, 1e-6)
                    if gear == "R":
                        step_cost += 0.08
                    if gear != current_gear:
                        step_cost += 1.8
                    new_g = g + step_cost
                    if new_g >= cost_so_far.get(nkey, float("inf")):
                        continue
                    cost_so_far[nkey] = new_g
                    nodes[nkey] = (nx, ny, nyaw, gear, current_key, steer)
                    counter += 1
                    heapq.heappush(open_heap, (new_g + heuristic(nx, ny, nyaw), new_g, counter, nkey))

        if goal_key is None:
            return []
        key_path: List[Tuple[int, int, int, str]] = []
        key: Optional[Tuple[int, int, int, str]] = goal_key
        while key is not None:
            key_path.append(key)
            _x, _y, _yaw, _gear, parent, _steer = nodes[key]
            key = parent
        key_path.reverse()

        path: List[Tuple[float, float, float, str]] = []
        if not key_path:
            return path
        sx, sy, syaw, sgear, _parent, _steer = nodes[key_path[0]]
        path.append((sx, sy, syaw, sgear))
        for idx in range(1, len(key_path)):
            parent_key = key_path[idx - 1]
            child_key = key_path[idx]
            px, py, pyaw, _pgear, _pparent, _psteer = nodes[parent_key]
            _cx, _cy, _cyaw, child_gear, _cparent, child_steer = nodes[child_key]
            direction = 1.0 if child_gear == "D" else -1.0
            samples = self._simulate_terminal_primitive(
                px,
                py,
                pyaw,
                direction,
                child_steer,
                wheelbase,
                target_slot,
                collision_check=False,
            )
            if not samples:
                return []
            path.extend((x, y, yaw, child_gear) for x, y, yaw in samples[1:])
        return path

    def _simulate_terminal_primitive(
        self,
        x: float,
        y: float,
        yaw: float,
        direction: float,
        steer: float,
        wheelbase: float,
        target_slot: Tuple[float, float, float, float],
        collision_check: bool,
    ) -> Optional[List[Tuple[float, float, float]]]:
        ds = direction * TERMINAL_PRIMITIVE_LENGTH / TERMINAL_PRIMITIVE_STEPS
        nx, ny, nyaw = x, y, yaw
        samples = [(nx, ny, nyaw)]
        for _ in range(TERMINAL_PRIMITIVE_STEPS):
            nx += ds * math.cos(nyaw)
            ny += ds * math.sin(nyaw)
            nyaw = self._wrap_to_pi(nyaw + (ds / max(wheelbase, 1e-6)) * math.tan(steer))
            if collision_check and self._pose_collides(nx, ny, nyaw, target_slot):
                return None
            samples.append((nx, ny, nyaw))
        return samples

    def _pose_path_to_waypoints(
        self,
        pose_path: List[Tuple[float, float, float, str]],
        target_pose: Tuple[float, float, float],
    ) -> List[Waypoint]:
        if not pose_path:
            return []
        waypoints: List[Waypoint] = []
        last_point: Optional[Tuple[float, float]] = None
        for x, y, yaw, gear in pose_path:
            if last_point is not None and math.hypot(x - last_point[0], y - last_point[1]) < 0.35:
                continue
            waypoints.append((x, y, yaw, gear))
            last_point = (x, y)
        tx, ty, tyaw = target_pose
        if math.hypot(waypoints[-1][0] - tx, waypoints[-1][1] - ty) > 0.2:
            waypoints.append((tx, ty, tyaw, waypoints[-1][3]))
        else:
            x, y, _yaw, gear = waypoints[-1]
            waypoints[-1] = (x, y, tyaw, gear)
        return waypoints

    def _preview_waypoints(
        self,
        waypoints: List[Waypoint],
        hybrid_start_index: int,
        start: Tuple[float, float, float],
        target_slot: Tuple[float, float, float, float],
        wheelbase: float,
        max_steer: float,
    ) -> Tuple[float, str, float]:
        if not waypoints:
            return 1e6, "no_waypoints", 0.0
        x, y, yaw = start
        v = 0.0
        delta = 0.0
        idx = 0
        t = 0.0
        best_iou_proxy = 0.0
        move_dist = 0.0
        prev_x, prev_y = x, y
        gear_switches = 0
        prev_gear = "D"
        steer_flips = 0
        prev_steer_sign = 0
        while t <= PREVIEW_MAX_TIME:
            final_wp = waypoints[-1]
            final_dist = math.hypot(final_wp[0] - x, final_wp[1] - y)
            final_yaw_error = abs(self._wrap_to_pi(final_wp[2] - yaw))
            idx = self._preview_advance_index(waypoints, idx, x, y, yaw)
            lookahead = self._preview_lookahead(idx, hybrid_start_index, abs(v), final_dist, final_yaw_error)
            target_idx = self._preview_lookahead_index(waypoints, idx, x, y, lookahead)
            target_wp = waypoints[target_idx]
            gear = target_wp[3]
            steer = self._pure_pursuit_steer(
                x=x,
                y=y,
                yaw=yaw,
                target_x=target_wp[0],
                target_y=target_wp[1],
                wheelbase=wheelbase,
                max_steer=max_steer,
                reverse=(gear == "R"),
            )
            front_clearance = self._estimate_forward_clearance(x, y, yaw, reverse=(gear == "R"))
            target_speed = self._target_speed(final_dist, final_yaw_error, steer, front_clearance)
            if gear == "R":
                target_speed = min(target_speed, 0.55)
            accel, brake = self._speed_command(abs(v), target_speed)
            delta = self._move_toward(delta, steer, PREVIEW_STEER_RATE * PREVIEW_DT)
            if gear != prev_gear:
                gear_switches += 1
                prev_gear = gear
            steer_sign = 0
            if abs(delta) >= math.radians(1.0):
                steer_sign = 1 if delta > 0 else -1
            if steer_sign:
                if prev_steer_sign and steer_sign != prev_steer_sign:
                    steer_flips += 1
                prev_steer_sign = steer_sign
            direction = 1.0 if gear == "D" else -1.0
            signed_v = direction * abs(v)
            a = direction * PREVIEW_MAX_ACCEL * accel
            if brake > 1e-3:
                a -= math.copysign(PREVIEW_MAX_BRAKE * brake, signed_v if abs(signed_v) > 1e-6 else direction)
            signed_v += a * PREVIEW_DT
            if gear == "D":
                signed_v = max(0.0, signed_v)
            else:
                signed_v = min(0.0, signed_v)
            v = abs(signed_v)
            x += signed_v * PREVIEW_DT * math.cos(yaw)
            y += signed_v * PREVIEW_DT * math.sin(yaw)
            yaw = self._wrap_to_pi(yaw + (signed_v * PREVIEW_DT / max(wheelbase, 1e-6)) * math.tan(delta))
            move_dist += math.hypot(x - prev_x, y - prev_y)
            prev_x, prev_y = x, y
            best_iou_proxy = max(best_iou_proxy, self._slot_center_iou_proxy(target_slot, x, y))
            if self._pose_collides(x, y, yaw, target_slot):
                return 10000.0 + 20.0 * (1.0 - best_iou_proxy) + 0.2 * final_dist, "collision", best_iou_proxy
            slot_iou = self._slot_iou(target_slot, x, y, yaw)
            if slot_iou >= PREVIEW_SUCCESS_IOU and abs(v) < 0.25:
                iou_penalty = 6.0 * max(0.0, PARKING_DEEP_IOU_TARGET - slot_iou)
                return (
                    0.02 * t
                    + 0.01 * move_dist
                    + 0.1 * steer_flips
                    + 0.2 * gear_switches
                    + iou_penalty
                ), "success", max(best_iou_proxy, slot_iou)
            if (
                slot_iou >= PREVIEW_SUCCESS_IOU
                and final_dist < self._slot_center_tolerance(list(target_slot))
                and final_yaw_error < math.radians(16.0)
                and abs(v) < 0.25
            ):
                iou_penalty = 6.0 * max(0.0, PARKING_DEEP_IOU_TARGET - slot_iou)
                return (
                    0.02 * t
                    + 0.01 * move_dist
                    + 0.1 * steer_flips
                    + 0.2 * gear_switches
                    + iou_penalty
                ), "success", max(best_iou_proxy, slot_iou)
            t += PREVIEW_DT
        final_wp = waypoints[-1]
        final_dist = math.hypot(final_wp[0] - x, final_wp[1] - y)
        final_yaw_error = abs(self._wrap_to_pi(final_wp[2] - yaw))
        return 3000.0 + 10.0 * final_dist + 3.0 * final_yaw_error - 100.0 * best_iou_proxy, "timeout", best_iou_proxy

    def _preview_advance_index(
        self,
        waypoints: List[Waypoint],
        current_idx: int,
        x: float,
        y: float,
        yaw: float,
    ) -> int:
        idx = min(current_idx, len(waypoints) - 1)
        run_end = self._preview_same_gear_run_end(waypoints, idx)
        start = max(0, idx - 2)
        closest = idx
        closest_dist = float("inf")
        for cand_idx in range(start, run_end + 1):
            wp = waypoints[cand_idx]
            dist = math.hypot(wp[0] - x, wp[1] - y)
            if dist < closest_dist:
                closest = cand_idx
                closest_dist = dist
        idx = max(idx, closest)
        while idx < run_end:
            current = waypoints[idx]
            nxt = waypoints[idx + 1]
            if math.hypot(nxt[0] - x, nxt[1] - y) + 0.2 < math.hypot(current[0] - x, current[1] - y):
                idx += 1
            else:
                break
        if idx == run_end and idx < len(waypoints) - 1:
            boundary = waypoints[run_end]
            dist_ok = math.hypot(boundary[0] - x, boundary[1] - y) <= GEAR_TRANSITION_RADIUS
            yaw_ok = abs(self._wrap_to_pi(boundary[2] - yaw)) <= GEAR_TRANSITION_YAW_TOLERANCE
            if dist_ok and yaw_ok:
                idx += 1
        return idx

    def _preview_same_gear_run_end(self, waypoints: List[Waypoint], start_idx: int) -> int:
        idx = min(max(start_idx, 0), len(waypoints) - 1)
        gear = waypoints[idx][3]
        while idx < len(waypoints) - 1 and waypoints[idx + 1][3] == gear:
            idx += 1
        return idx

    def _preview_lookahead(
        self,
        idx: int,
        hybrid_start_index: int,
        speed: float,
        final_dist: float,
        yaw_error: float,
    ) -> float:
        if idx >= hybrid_start_index:
            return 0.65 if final_dist < 3.0 else 0.95
        return self._adaptive_lookahead(speed, final_dist, yaw_error)

    def _preview_lookahead_index(
        self,
        waypoints: List[Waypoint],
        idx: int,
        x: float,
        y: float,
        lookahead: float,
    ) -> int:
        idx = min(idx, len(waypoints) - 1)
        accum = math.hypot(waypoints[idx][0] - x, waypoints[idx][1] - y)
        while idx < len(waypoints) - 1 and accum < lookahead:
            a = waypoints[idx]
            b = waypoints[idx + 1]
            if a[3] != b[3]:
                return idx
            accum += math.hypot(b[0] - a[0], b[1] - a[1])
            idx += 1
        return idx

    def _slot_center_iou_proxy(
        self,
        slot: Tuple[float, float, float, float],
        x: float,
        y: float,
    ) -> float:
        cx = 0.5 * (slot[0] + slot[1])
        cy = 0.5 * (slot[2] + slot[3])
        scale = max(abs(slot[1] - slot[0]), abs(slot[3] - slot[2]), 1.0)
        return max(0.0, 1.0 - math.hypot(x - cx, y - cy) / scale)

    def _slot_iou(
        self,
        slot: Tuple[float, float, float, float],
        x: float,
        y: float,
        yaw: float,
    ) -> float:
        car_poly = self._car_polygon(x, y, yaw)
        inter_poly = self._clip_polygon_to_rect(car_poly, slot)
        inter_area = self._polygon_area(inter_poly)
        if inter_area <= 1e-9:
            return 0.0
        car_area = self._polygon_area(car_poly)
        slot_area = max(0.0, (slot[1] - slot[0]) * (slot[3] - slot[2]))
        union = car_area + slot_area - inter_area
        if union <= 1e-9:
            return 0.0
        return max(0.0, min(1.0, inter_area / union))

    def _clip_polygon_to_rect(
        self,
        poly: List[Tuple[float, float]],
        rect: Tuple[float, float, float, float],
    ) -> List[Tuple[float, float]]:
        x0, x1, y0, y1 = rect

        def clip(
            points: List[Tuple[float, float]],
            inside,
            intersect,
        ) -> List[Tuple[float, float]]:
            if not points:
                return []
            output: List[Tuple[float, float]] = []
            prev = points[-1]
            prev_inside = inside(prev)
            for cur in points:
                cur_inside = inside(cur)
                if cur_inside:
                    if not prev_inside:
                        output.append(intersect(prev, cur))
                    output.append(cur)
                elif prev_inside:
                    output.append(intersect(prev, cur))
                prev, prev_inside = cur, cur_inside
            return output

        def interp_x(a: Tuple[float, float], b: Tuple[float, float], x_edge: float) -> Tuple[float, float]:
            denom = b[0] - a[0]
            if abs(denom) <= 1e-9:
                return x_edge, a[1]
            t = (x_edge - a[0]) / denom
            return x_edge, a[1] + t * (b[1] - a[1])

        def interp_y(a: Tuple[float, float], b: Tuple[float, float], y_edge: float) -> Tuple[float, float]:
            denom = b[1] - a[1]
            if abs(denom) <= 1e-9:
                return a[0], y_edge
            t = (y_edge - a[1]) / denom
            return a[0] + t * (b[0] - a[0]), y_edge

        clipped = clip(poly, lambda p: p[0] >= x0, lambda a, b: interp_x(a, b, x0))
        clipped = clip(clipped, lambda p: p[0] <= x1, lambda a, b: interp_x(a, b, x1))
        clipped = clip(clipped, lambda p: p[1] >= y0, lambda a, b: interp_y(a, b, y0))
        clipped = clip(clipped, lambda p: p[1] <= y1, lambda a, b: interp_y(a, b, y1))
        return clipped

    def _polygon_area(self, poly: List[Tuple[float, float]]) -> float:
        if len(poly) < 3:
            return 0.0
        area = 0.0
        for idx, (x1, y1) in enumerate(poly):
            x2, y2 = poly[(idx + 1) % len(poly)]
            area += x1 * y2 - x2 * y1
        return abs(area) * 0.5

    def _pose_collides(
        self,
        x: float,
        y: float,
        yaw: float,
        target_slot: Tuple[float, float, float, float],
    ) -> bool:
        if self.map_extent is not None:
            xmin, xmax, ymin, ymax = self.map_extent
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                return True
        car_poly = self._car_polygon(x, y, yaw)
        if self._rect_contains_poly(target_slot, car_poly):
            return False
        reject_radius = math.hypot(0.5 * VEHICLE_LONGEST_LENGTH, 0.85) + 0.10
        reject_radius_sq = reject_radius * reject_radius
        for rect in self._nearby_rects(
            "collision",
            self._collision_rects(),
            x,
            y,
            reject_radius,
        ):
            rx0, rx1, ry0, ry1 = rect
            dx = max(rx0 - x, 0.0, x - rx1)
            dy = max(ry0 - y, 0.0, y - ry1)
            if dx * dx + dy * dy > reject_radius_sq:
                continue
            if self._poly_intersects_rect(car_poly, rect):
                return True
        return False

    def _car_polygon(self, x: float, y: float, yaw: float) -> List[Tuple[float, float]]:
        half_l = 0.5 * VEHICLE_LONGEST_LENGTH
        half_w = 0.85
        corners = [(half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w)]
        c = math.cos(yaw)
        s = math.sin(yaw)
        return [(x + c * px - s * py, y + s * px + c * py) for px, py in corners]

    def _rect_contains_poly(
        self,
        rect: Tuple[float, float, float, float],
        poly: List[Tuple[float, float]],
    ) -> bool:
        x0, x1, y0, y1 = rect
        return all(x0 <= px <= x1 and y0 <= py <= y1 for px, py in poly)

    def _poly_intersects_rect(
        self,
        poly: List[Tuple[float, float]],
        rect: Tuple[float, float, float, float],
    ) -> bool:
        rx0, rx1, ry0, ry1 = rect
        rect_poly = [(rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1)]
        return self._polys_intersect(poly, rect_poly)

    def _polys_intersect(
        self,
        poly_a: List[Tuple[float, float]],
        poly_b: List[Tuple[float, float]],
    ) -> bool:
        for poly in (poly_a, poly_b):
            for idx in range(len(poly)):
                x1, y1 = poly[idx]
                x2, y2 = poly[(idx + 1) % len(poly)]
                axis = (-(y2 - y1), x2 - x1)
                length = math.hypot(axis[0], axis[1])
                if length <= 1e-9:
                    continue
                axis = (axis[0] / length, axis[1] / length)
                min_a, max_a = self._project_polygon(poly_a, axis)
                min_b, max_b = self._project_polygon(poly_b, axis)
                if max_a < min_b or max_b < min_a:
                    return False
        return True

    def _project_polygon(
        self,
        poly: List[Tuple[float, float]],
        axis: Tuple[float, float],
    ) -> Tuple[float, float]:
        dots = [px * axis[0] + py * axis[1] for px, py in poly]
        return min(dots), max(dots)

    def _move_toward(self, current: float, target: float, step: float) -> float:
        if current < target:
            return min(target, current + step)
        return max(target, current - step)

    def _warm_planning_caches(self) -> None:
        if self.map_extent is None:
            return
        xmin, xmax, ymin, ymax = self.map_extent
        grid_step = max(self.cell_size, GUIDED_GRID_STEP_MIN)
        cols = max(1, int(math.ceil((xmax - xmin) / grid_step)))
        rows = max(1, int(math.ceil((ymax - ymin) / grid_step)))
        self._cached_guided_blocked_grid(rows, cols, grid_step)
        self._cached_guided_distance_field(rows, cols, grid_step)

    def _cached_guided_distance_field(
        self,
        rows: int,
        cols: int,
        grid_step: float,
    ) -> List[List[float]]:
        if (
            self.guided_distance_field_cache is None
            or self.guided_distance_field_cache[0] != rows
            or self.guided_distance_field_cache[1] != cols
            or abs(self.guided_distance_field_cache[2] - grid_step) > 1e-9
        ):
            blocked = self._cached_guided_blocked_grid(rows, cols, grid_step)
            field = self._guided_distance_field(blocked, grid_step)
            self.guided_distance_field_cache = (rows, cols, grid_step, field)
        return self.guided_distance_field_cache[3]

    def _guided_distance_field(
        self,
        blocked: List[List[bool]],
        grid_step: float,
    ) -> List[List[float]]:
        rows = len(blocked)
        cols = len(blocked[0]) if rows else 0
        dist = [[float("inf") for _ in range(cols)] for _ in range(rows)]
        heap: List[Tuple[float, Tuple[int, int]]] = []
        for row in range(rows):
            for col in range(cols):
                if blocked[row][col]:
                    dist[row][col] = 0.0
                    heapq.heappush(heap, (0.0, (row, col)))
        if not heap:
            return [[CLEARANCE_QUERY_RADIUS for _ in range(cols)] for _ in range(rows)]
        motions = [
            (-1, 0, grid_step),
            (1, 0, grid_step),
            (0, -1, grid_step),
            (0, 1, grid_step),
            (-1, -1, 1.414 * grid_step),
            (-1, 1, 1.414 * grid_step),
            (1, -1, 1.414 * grid_step),
            (1, 1, 1.414 * grid_step),
        ]
        while heap:
            current_dist, (row, col) = heapq.heappop(heap)
            if current_dist > dist[row][col] + 1e-9:
                continue
            for dr, dc, step_cost in motions:
                nxt = (row + dr, col + dc)
                if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols):
                    continue
                new_dist = current_dist + step_cost
                if new_dist >= dist[nxt[0]][nxt[1]]:
                    continue
                dist[nxt[0]][nxt[1]] = new_dist
                heapq.heappush(heap, (new_dist, nxt))
        return dist

    def _guided_distance_at_cell(
        self,
        cell: Tuple[int, int],
        distance_field: Optional[List[List[float]]] = None,
    ) -> float:
        if self.map_extent is None:
            return CLEARANCE_QUERY_RADIUS
        xmin, xmax, ymin, ymax = self.map_extent
        grid_step = max(self.cell_size, GUIDED_GRID_STEP_MIN)
        rows = max(1, int(math.ceil((ymax - ymin) / grid_step)))
        cols = max(1, int(math.ceil((xmax - xmin) / grid_step)))
        if distance_field is None:
            distance_field = self._cached_guided_distance_field(rows, cols, grid_step)
        row = max(0, min(rows - 1, cell[0]))
        col = max(0, min(cols - 1, cell[1]))
        value = distance_field[row][col]
        if math.isinf(value):
            return CLEARANCE_QUERY_RADIUS
        return min(value, CLEARANCE_QUERY_RADIUS)

    def _guided_distance_at_point(self, point: Tuple[float, float]) -> float:
        if self.map_extent is None:
            return CLEARANCE_QUERY_RADIUS
        xmin, xmax, ymin, ymax = self.map_extent
        grid_step = max(self.cell_size, GUIDED_GRID_STEP_MIN)
        rows = max(1, int(math.ceil((ymax - ymin) / grid_step)))
        cols = max(1, int(math.ceil((xmax - xmin) / grid_step)))
        px, py = point
        col = int((px - xmin) / grid_step)
        row = int((ymax - py) / grid_step)
        return self._guided_distance_at_cell((row, col))

    def _guided_cell_risk_cost(
        self,
        cell: Tuple[int, int],
        distance_field: List[List[float]],
    ) -> float:
        clearance = self._guided_distance_at_cell(cell, distance_field)
        risk = math.exp(-(clearance * clearance) / (2.0 * RISK_FIELD_SIGMA * RISK_FIELD_SIGMA))
        inverse_clearance = 1.0 / max(clearance, GUIDED_DISTANCE_EPS)
        near_line = max(0.0, GUIDED_NEAR_LINE_THRESHOLD - clearance)
        return (
            GUIDED_RISK_COST_WEIGHT * risk
            + GUIDED_CLEARANCE_COST_WEIGHT * inverse_clearance
            + GUIDED_NEAR_LINE_COST_WEIGHT * near_line * near_line
        )

    def _collision_rects(self) -> List[Tuple[float, float, float, float]]:
        if self.collision_rect_cache is None:
            self.collision_rect_cache = self._obstacle_rects() + self._line_obstacle_rects(half_width=0.25)
        return self.collision_rect_cache

    def _nearby_rects(
        self,
        cache_key: str,
        rects: List[Tuple[float, float, float, float]],
        x: float,
        y: float,
        radius: float,
    ) -> List[Tuple[float, float, float, float]]:
        if not rects or self.map_extent is None:
            return rects
        cell_size, buckets = self._spatial_index(cache_key, rects)
        xmin, _xmax, _ymin, ymax = self.map_extent
        c0 = int((x - radius - xmin) / cell_size)
        c1 = int((x + radius - xmin) / cell_size)
        r0 = int((ymax - (y + radius)) / cell_size)
        r1 = int((ymax - (y - radius)) / cell_size)
        nearby: List[Tuple[float, float, float, float]] = []
        seen = set()
        for row in range(r0, r1 + 1):
            for col in range(c0, c1 + 1):
                for rect in buckets.get((row, col), []):
                    if rect in seen:
                        continue
                    seen.add(rect)
                    nearby.append(rect)
        return nearby

    def _spatial_index(
        self,
        cache_key: str,
        rects: List[Tuple[float, float, float, float]],
    ) -> Tuple[float, Dict[Tuple[int, int], List[Tuple[float, float, float, float]]]]:
        if self.spatial_index_cache is None:
            self.spatial_index_cache = {}
        cached = self.spatial_index_cache.get(cache_key)
        if cached is not None:
            return cached
        cell_size = SPATIAL_INDEX_CELL_SIZE
        buckets: Dict[Tuple[int, int], List[Tuple[float, float, float, float]]] = {}
        if self.map_extent is None:
            result = (cell_size, buckets)
            self.spatial_index_cache[cache_key] = result
            return result
        xmin, _xmax, _ymin, ymax = self.map_extent
        for rect in rects:
            rx0, rx1, ry0, ry1 = rect
            c0 = int((rx0 - xmin) / cell_size)
            c1 = int((rx1 - xmin) / cell_size)
            r0 = int((ymax - ry1) / cell_size)
            r1 = int((ymax - ry0) / cell_size)
            for row in range(r0, r1 + 1):
                for col in range(c0, c1 + 1):
                    buckets.setdefault((row, col), []).append(rect)
        result = (cell_size, buckets)
        self.spatial_index_cache[cache_key] = result
        return result

    def _obstacle_rects(self) -> List[Tuple[float, float, float, float]]:
        if self.obstacle_rect_cache is not None:
            return self.obstacle_rect_cache
        rects: List[Tuple[float, float, float, float]] = []
        if not self.map_data:
            self.obstacle_rect_cache = rects
            return rects
        slots = self.map_data.get("slots") or []
        occupied = self.map_data.get("occupied_idx") or []
        for idx, slot in enumerate(slots):
            if idx < len(occupied) and bool(occupied[idx]):
                rects.append(tuple(float(v) for v in slot))
        for rect in self.map_data.get("walls_rects") or []:
            rects.append(tuple(float(v) for v in rect))
        self.obstacle_rect_cache = rects
        return rects

    def _line_obstacle_rects(self, half_width: float) -> List[Tuple[float, float, float, float]]:
        key = round(float(half_width), 4)
        if self.line_rect_cache is None:
            self.line_rect_cache = {}
        if key in self.line_rect_cache:
            return self.line_rect_cache[key]
        rects: List[Tuple[float, float, float, float]] = []
        if not self.map_data or self.map_extent is None:
            self.line_rect_cache[key] = rects
            return rects
        xmin, xmax, ymin, ymax = self.map_extent
        for line in self.map_data.get("lines") or []:
            if len(line) != 4:
                continue
            x1, y1, x2, y2 = (float(v) for v in line)
            if abs(x1 - x2) < 1e-6:
                rx0 = min(x1, x2) - half_width
                rx1 = max(x1, x2) + half_width
                ry0 = min(y1, y2)
                ry1 = max(y1, y2)
            elif abs(y1 - y2) < 1e-6:
                rx0 = min(x1, x2)
                rx1 = max(x1, x2)
                ry0 = min(y1, y2) - half_width
                ry1 = max(y1, y2) + half_width
            else:
                rx0 = min(x1, x2) - half_width
                rx1 = max(x1, x2) + half_width
                ry0 = min(y1, y2) - half_width
                ry1 = max(y1, y2) + half_width
            rx0 = max(rx0, xmin)
            rx1 = min(rx1, xmax)
            ry0 = max(ry0, ymin)
            ry1 = min(ry1, ymax)
            if rx1 > rx0 and ry1 > ry0:
                rects.append((rx0, rx1, ry0, ry1))
        self.line_rect_cache[key] = rects
        return rects

    def _mark_rect(
        self,
        blocked: List[List[bool]],
        rect: Tuple[float, float, float, float],
        grid_step: float,
        margin: float,
    ) -> None:
        if self.map_extent is None:
            return
        xmin, _, _, ymax = self.map_extent
        rows = len(blocked)
        cols = len(blocked[0]) if rows else 0
        rx0, rx1, ry0, ry1 = rect
        rx0 -= margin
        rx1 += margin
        ry0 -= margin
        ry1 += margin
        c0 = max(0, int((rx0 - xmin) / grid_step))
        c1 = min(cols - 1, int((rx1 - xmin) / grid_step))
        r0 = max(0, int((ymax - ry1) / grid_step))
        r1 = min(rows - 1, int((ymax - ry0) / grid_step))
        for row in range(r0, r1 + 1):
            for col in range(c0, c1 + 1):
                blocked[row][col] = True

    def _inflate_blocked(self, blocked: List[List[bool]], radius_cells: int) -> List[List[bool]]:
        rows = len(blocked)
        cols = len(blocked[0]) if rows else 0
        inflated = [row[:] for row in blocked]
        for row in range(rows):
            for col in range(cols):
                if not blocked[row][col]:
                    continue
                for dr in range(-radius_cells, radius_cells + 1):
                    for dc in range(-radius_cells, radius_cells + 1):
                        rr = row + dr
                        cc = col + dc
                        if 0 <= rr < rows and 0 <= cc < cols:
                            inflated[rr][cc] = True
        return inflated

    def _clear_cell(self, blocked: List[List[bool]], cell: Tuple[int, int], radius: int) -> None:
        rows = len(blocked)
        cols = len(blocked[0]) if rows else 0
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = cell[0] + dr
                cc = cell[1] + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    blocked[rr][cc] = False

    def _simplify_path(
        self,
        path: List[Tuple[float, float]],
        spacing: float,
    ) -> List[Tuple[float, float]]:
        if len(path) <= 2:
            return path
        simplified = [path[0]]
        last = path[0]
        for point in path[1:-1]:
            if math.hypot(point[0] - last[0], point[1] - last[1]) >= spacing:
                simplified.append(point)
                last = point
        simplified.append(path[-1])
        return simplified

    def _points_to_waypoints(
        self,
        points: List[Tuple[float, float]],
        final_yaw: float,
        gear: str,
    ) -> List[Waypoint]:
        waypoints: List[Waypoint] = []
        for idx, point in enumerate(points):
            if idx < len(points) - 1:
                nxt = points[idx + 1]
                yaw = math.atan2(nxt[1] - point[1], nxt[0] - point[0])
            else:
                yaw = final_yaw
            waypoints.append((point[0], point[1], yaw, gear))
        return waypoints

    def _advance_waypoint_index(self, x: float, y: float) -> None:
        if self.waypoint_index >= self.hybrid_start_index:
            self._advance_terminal_waypoint_index(x, y)
            return
        run_end = min(self.hybrid_start_index - 1, len(self.waypoints) - 1)
        start = max(0, self.waypoint_index - 2)
        closest = self.waypoint_index
        closest_dist = float("inf")
        for idx in range(start, run_end + 1):
            wp = self.waypoints[idx]
            dist = math.hypot(wp[0] - x, wp[1] - y)
            if dist < closest_dist:
                closest = idx
                closest_dist = dist
        self.waypoint_index = max(self.waypoint_index, closest)
        while self.waypoint_index < len(self.waypoints) - 1:
            wp = self.waypoints[self.waypoint_index]
            if math.hypot(wp[0] - x, wp[1] - y) > 0.8:
                break
            self.waypoint_index += 1

    def _advance_terminal_waypoint_index(self, x: float, y: float) -> None:
        if not self.waypoints:
            return
        self.waypoint_index = min(self.waypoint_index, len(self.waypoints) - 1)
        run_end = self._same_gear_run_end(self.waypoint_index)
        start = max(0, self.waypoint_index - 2)
        closest = self.waypoint_index
        closest_dist = float("inf")
        for idx in range(start, run_end + 1):
            wp = self.waypoints[idx]
            dist = math.hypot(wp[0] - x, wp[1] - y)
            if dist < closest_dist:
                closest = idx
                closest_dist = dist
        self.waypoint_index = max(self.waypoint_index, closest)
        active_wp = self.waypoints[self.waypoint_index]
        active_dist = math.hypot(active_wp[0] - x, active_wp[1] - y)
        if self.waypoint_index != self.terminal_progress_index:
            self.terminal_progress_index = self.waypoint_index
            self.terminal_progress_best_dist = active_dist
            self.terminal_stuck_ticks = 0
        elif active_dist < self.terminal_progress_best_dist - 0.08:
            self.terminal_progress_best_dist = active_dist
            self.terminal_stuck_ticks = 0
        else:
            self.terminal_stuck_ticks += 1
        while self.waypoint_index < run_end:
            current = self.waypoints[self.waypoint_index]
            nxt = self.waypoints[self.waypoint_index + 1]
            if math.hypot(nxt[0] - x, nxt[1] - y) + 0.2 < math.hypot(current[0] - x, current[1] - y):
                self.waypoint_index += 1
                self.terminal_progress_index = self.waypoint_index
                self.terminal_progress_best_dist = math.hypot(nxt[0] - x, nxt[1] - y)
                self.terminal_stuck_ticks = 0
            else:
                break
        if self.waypoint_index == run_end and self.waypoint_index < len(self.waypoints) - 1:
            boundary = self.waypoints[run_end]
            dist_ok = math.hypot(boundary[0] - x, boundary[1] - y) <= GEAR_TRANSITION_RADIUS
            yaw_ok = (
                abs(self._wrap_to_pi(boundary[2] - self.current_yaw_for_progress))
                <= GEAR_TRANSITION_YAW_TOLERANCE
            )
            boundary_dist = math.hypot(boundary[0] - x, boundary[1] - y)
            next_gear = self.waypoints[run_end + 1][3]
            stuck_at_gear_boundary = (
                self.terminal_stuck_ticks >= TERMINAL_LOOP_STUCK_TICKS
                and next_gear != boundary[3]
                and boundary_dist <= TERMINAL_LOOP_FORCE_RADIUS
            )
            if dist_ok and yaw_ok:
                self.waypoint_index += 1
                self.terminal_progress_index = self.waypoint_index
                self.terminal_progress_best_dist = float("inf")
                self.terminal_stuck_ticks = 0
            elif stuck_at_gear_boundary:
                print(
                    "[algo] terminal recovery: forced gear-boundary advance"
                    f" wp={run_end}->{run_end + 1}"
                    f" boundary_dist={boundary_dist:.2f}m"
                    f" yaw_error={math.degrees(abs(self._wrap_to_pi(boundary[2] - self.current_yaw_for_progress))):.1f}deg"
                )
                self.waypoint_index += 1
                self.terminal_progress_index = self.waypoint_index
                self.terminal_progress_best_dist = float("inf")
                self.terminal_stuck_ticks = 0

    def _same_gear_run_end(self, start_idx: int) -> int:
        idx = min(max(start_idx, 0), len(self.waypoints) - 1)
        gear = self.waypoints[idx][3]
        while idx < len(self.waypoints) - 1 and self.waypoints[idx + 1][3] == gear:
            idx += 1
        return idx

    def _lookahead_index(self, x: float, y: float, lookahead: float) -> int:
        idx = self.waypoint_index
        while idx < len(self.waypoints) - 1:
            wp = self.waypoints[idx]
            if math.hypot(wp[0] - x, wp[1] - y) >= lookahead:
                return idx
            idx += 1
        return len(self.waypoints) - 1

    def _adaptive_lookahead(self, speed: float, final_dist: float, yaw_error: float) -> float:
        if self.waypoint_index >= self.hybrid_start_index:
            if final_dist < 3.0:
                return 0.65
            return 0.95
        lookahead = 1.65 + 0.45 * min(speed, 3.0)
        if final_dist > 12.0 and yaw_error < math.radians(25.0):
            lookahead = max(lookahead, 2.15)
        if final_dist < 6.0:
            lookahead = min(lookahead, 1.05)
        if final_dist < 2.5:
            lookahead = min(lookahead, 0.70)
        if yaw_error > math.radians(35.0):
            lookahead = min(lookahead, 0.90)
        return max(0.65, lookahead)

    def _pure_pursuit_steer(
        self,
        x: float,
        y: float,
        yaw: float,
        target_x: float,
        target_y: float,
        wheelbase: float,
        max_steer: float,
        reverse: bool,
    ) -> float:
        dx = target_x - x
        dy = target_y - y
        tracking_yaw = self._wrap_to_pi(yaw + math.pi) if reverse else yaw
        local_x = math.cos(tracking_yaw) * dx + math.sin(tracking_yaw) * dy
        local_y = -math.sin(tracking_yaw) * dx + math.cos(tracking_yaw) * dy
        lookahead = max(0.9, math.hypot(local_x, local_y))
        alpha = math.atan2(local_y, local_x)
        steer = math.atan2(2.0 * wheelbase * math.sin(alpha), lookahead)
        if reverse:
            steer = -steer
        return max(-max_steer, min(max_steer, steer))

    def _parking_reverse_target(
        self,
        target_center: Tuple[float, float],
        target_yaw: float,
    ) -> Waypoint:
        backout_distance = 3.4
        tx = target_center[0] - math.cos(target_yaw) * backout_distance
        ty = target_center[1] - math.sin(target_yaw) * backout_distance
        tx, ty = self._clamp_inside_map(tx, ty, margin=0.2)
        return tx, ty, target_yaw, "R"

    def _target_speed(
        self,
        final_dist: float,
        yaw_error: float,
        steer: float,
        front_clearance: float,
    ) -> float:
        in_parking_mode = final_dist < PARKING_ALIGN_DISTANCE
        target = min(4.20, 1.80 + 0.07 * final_dist)
        if (
            front_clearance >= FRONT_CLEAR_DISTANCE
            and final_dist > 4.0
            and yaw_error < math.radians(25.0)
            and abs(steer) < math.radians(22.0)
        ):
            target += FRONT_CLEAR_SPEED_BONUS + 0.35
        if final_dist < 6.0:
            target = 1.15
        if in_parking_mode:
            target = min(target, 0.65)
        if final_dist < 2.2:
            target = 0.28
        if final_dist < 1.0:
            target = 0.12
        if yaw_error > math.radians(35.0) or abs(steer) > math.radians(25.0):
            turn_cap = 2.40 if not in_parking_mode else 0.45
            target = min(target, turn_cap)
        if front_clearance < OBSTACLE_SLOW_DISTANCE:
            target = min(target, 0.60 if not in_parking_mode else 0.45)
        if front_clearance < OBSTACLE_STOP_DISTANCE:
            target = 0.0
        return target

    def _inside_map(self, x: float, y: float, margin: float = 0.4) -> bool:
        if self.map_extent is None:
            return True
        xmin, xmax, ymin, ymax = self.map_extent
        margin = max(margin, VEHICLE_CENTER_CLEARANCE + EXTRA_SAFETY_MARGIN)
        return xmin + margin <= x <= xmax - margin and ymin + margin <= y <= ymax - margin

    def _clamp_inside_map(self, x: float, y: float, margin: float = 0.4) -> Tuple[float, float]:
        if self.map_extent is None:
            return x, y
        xmin, xmax, ymin, ymax = self.map_extent
        margin = max(margin, VEHICLE_CENTER_CLEARANCE + EXTRA_SAFETY_MARGIN)
        return (
            max(xmin + margin, min(xmax - margin, x)),
            max(ymin + margin, min(ymax - margin, y)),
        )

    def _path_length(self, path: List[Tuple[float, float]]) -> float:
        if len(path) < 2:
            return 0.0
        return sum(
            math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
            for i in range(1, len(path))
        )

    def _speed_command(
        self,
        speed: float,
        target_speed: float,
        front_is_clear: bool = False,
        force_full_accel: bool = False,
    ) -> Tuple[float, float]:
        error = target_speed - speed
        if target_speed <= 0.05:
            return 0.0, 1.0
        if force_full_accel and error > 0.05:
            return 1.0, 0.0
        if error > 0.15:
            accel_cap = 1.0 if front_is_clear else 0.9
            accel_base = 0.70 if front_is_clear else 0.55
            accel_gain = 0.70 if front_is_clear else 0.55
            return min(accel_cap, accel_base + accel_gain * error), 0.0
        if error < -0.08:
            return 0.0, min(0.8, 0.25 + 0.35 * (-error))
        return 0.0, 0.0

    def _estimate_forward_clearance(
        self,
        x: float,
        y: float,
        yaw: float,
        reverse: bool = False,
        max_distance: float = FRONT_CLEAR_DISTANCE,
    ) -> float:
        heading = self._wrap_to_pi(yaw + math.pi) if reverse else yaw
        step = 0.5
        distance = step
        while distance <= max_distance:
            px = x + math.cos(heading) * distance
            py = y + math.sin(heading) * distance
            if self._estimate_clearance(
                (px, py),
                include_lines=True,
            ) <= 0.05:
                return distance
            distance += step
        return max_distance

    def _estimate_min_obstacle_distance(self, point: Tuple[float, float]) -> float:
        return self._estimate_clearance(point, include_lines=False)

    def _estimate_clearance(
        self,
        point: Tuple[float, float],
        include_lines: bool = False,
    ) -> float:
        px, py = point
        best = float("inf")
        if self.map_extent is not None:
            xmin, xmax, ymin, ymax = self.map_extent
            best = min(best, px - xmin, xmax - px, py - ymin, ymax - py)
        rects = self._obstacle_rects()
        cache_key = "clearance_obstacles"
        if include_lines:
            rects = rects + self._line_obstacle_rects(half_width=0.08)
            cache_key = "clearance_with_lines"
        search_radius = CLEARANCE_QUERY_RADIUS + VEHICLE_CENTER_CLEARANCE + EXTRA_SAFETY_MARGIN
        for rx0, rx1, ry0, ry1 in self._nearby_rects(cache_key, rects, px, py, search_radius):
            dx = max(rx0 - px, 0.0, px - rx1)
            dy = max(ry0 - py, 0.0, py - ry1)
            best = min(best, math.hypot(dx, dy))
        return max(0.0, best - VEHICLE_CENTER_CLEARANCE - EXTRA_SAFETY_MARGIN)

    def _log_evaluation(
        self,
        parking_success: bool,
        fail_reason: str,
        final_position_error: float,
        final_yaw_error: float,
        collision: bool,
        current_time: Optional[float] = None,
        force: bool = False,
    ) -> None:
        if parking_success and self.final_eval_logged:
            return
        if not force and current_time is not None and current_time - self.last_eval_log_time < 5.0:
            return
        if parking_success:
            self.final_eval_logged = True
        if current_time is not None:
            self.last_eval_log_time = current_time
        payload = {
            "parking_success": bool(parking_success),
            "fail_reason": fail_reason,
            "final_position_error": round(float(final_position_error), 3),
            "final_yaw_error": round(math.degrees(float(final_yaw_error)), 2),
            "min_obstacle_distance": (
                None
                if math.isinf(self.min_obstacle_distance)
                else round(float(self.min_obstacle_distance), 3)
            ),
            "collision": bool(collision),
            "step_count": int(self.step_count),
            "rl_speed_control": "ON" if self.rl_speed and self.rl_speed.enabled else "OFF",
        }
        print("[eval] " + json.dumps(payload, sort_keys=True))

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


planner = PlannerSkeleton()


def handle_map_payload(map_payload: Dict[str, Any]) -> None:
    """Called by ipc_client.py when the simulator sends the static map."""

    planner.set_map(map_payload)


def planner_step(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Called by ipc_client.py every simulation tick."""

    try:
        return planner.compute_control(obs)
    except Exception as exc:
        print(f"[algo] planner_step error: {exc}")
        return {"steer": 0.0, "accel": 0.0, "brake": 0.8, "gear": "D"}