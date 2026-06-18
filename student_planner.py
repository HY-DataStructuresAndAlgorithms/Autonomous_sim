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
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rl_speed_controller import RLSpeedController


Waypoint = Tuple[float, float, float, str]  # x, y, desired yaw, gear
USE_RL_SPEED_CONTROL = False
VEHICLE_LENGTH = 3.0
VEHICLE_WIDTH = 1.6
VEHICLE_FRONT_LENGTH = 1.6
VEHICLE_REAR_LENGTH = 1.4
VEHICLE_HALF_WIDTH = 0.5 * VEHICLE_WIDTH
VEHICLE_RECT_MARGIN = 1.5
APPROACH_DRIVING_VEHICLE_MARGIN = 0.5
PARKING_VEHICLE_RECT_MARGIN = 0.05
PLANNING_OBSTACLE_MARGIN = VEHICLE_RECT_MARGIN
EXTRA_SAFETY_MARGIN = 0.0
OBSTACLE_SLOW_DISTANCE = 1.15
OBSTACLE_STOP_DISTANCE = 0.35
FRONT_CLEAR_DISTANCE = 6.0
FRONT_CLEAR_SPEED_BONUS = 0.75
PARKING_ALIGN_DISTANCE = 6.0
FRONT_APPROACH_DISTANCE = 5.0
FRONT_APPROACH_REACHED_TOLERANCE = 1.0
REAR_APPROACH_DISTANCE = 2.0
REAR_Y_TRIANGLE_HEIGHT = 7.0
REAR_Y_TRIANGLE_HALF_WIDTH = 4.2
REAR_Y_POINT2_EXTRA_WIDTH = 2.0
REAR_Y_POINT1_VERTICAL_PULL = 2.0
FIRST_APPROACH_EXTRA_Y_DISTANCE = 2.0
REAR_APPROACH_YAW_TOLERANCE = math.radians(30.0)
REAR_REVERSE_START_TOLERANCE = 1.0
REAR_POINT3_REACHED_TOLERANCE = 1.0
REAR_ENTRY_DISTANCE = 2.0
REAR_ALIGN_YAW_TOLERANCE = math.radians(8.0)
REAR_ALIGN_SPEED = 0.45
REAR_REVERSE_SPEED = 0.34
REAR_REVERSE_FINAL_SPEED = 0.22
REAR_REVERSE_MIN_SPEED = 0.30
REAR_STEER_RATE_LIMIT = math.radians(7.0)
PARKING_SEGMENT_TRIGGER_DISTANCE = 6.00
APPROACH_CRUISE_SPEED = 5.30
APPROACH_LOOKAHEAD_WINDOW = 18
APPROACH_CURVATURE_SLOW_ANGLE = math.radians(32.0)
APPROACH_CURVATURE_SLOW_SPEED = 1.35
APPROACH_RECOVERY_MIN_SPEED = 2.20
APPROACH_ON_PATH_MARGIN = 0.50
APPROACH_PATH_DEVIATION_SPEED_DISTANCE = 1.20
APPROACH_PREALIGN_DISTANCE = 4.0
APPROACH_PREALIGN_MAX_BLEND = 0.45
APPROACH_PREALIGN_YAW_OFFSET_DEG = 0.0
APPROACH_PREALIGN_YAW_BLEND = 0.35
APPROACH_STUCK_SECONDS = 3.0
APPROACH_STUCK_MIN_DISTANCE = 2.0
APPROACH_STUCK_FRONT_BLOCKED = 1.20
APPROACH_STUCK_REVERSE_DISTANCE = 1.80
APPROACH_STUCK_REVERSE_STOP_CLEARANCE = 0.45
APPROACH_STUCK_FORWARD_DISTANCE = 1.20
APPROACH_STUCK_FORWARD_STOP_CLEARANCE = 0.45
PARKING_PREPARE_FULL_STOP_SPEED = 0.03
PARKING_PREPARE_BRAKE = 0.45
PARKING_REVERSE_STOP_CLEARANCE = 0.45
PARKING_STATE_APPROACH = "APPROACH_MODE"
PARKING_STATE_PREPARE_STOP = "PREPARE_STOP"
PARKING_STATE_SECOND_APPROACH = "SECOND_APPROACH_MODE"
PARKING_STATE_ALIGN_CHECK = "ALIGN_MODE"
PARKING_STATE_REVERSE_PARKING = "REVERSE_PARKING_MODE"
PARKING_STATE_STOP = "STOP_MODE"
LINE_EXTRA_CLEARANCE = 0.20
LINE_COLLISION_HALF_WIDTH = 0.25
LINE_HARD_MARGIN = 0.05
A_STAR_GRID_STEP = 1.00
A_STAR_MAX_HEADING_STEP = 1  # 8-heading grid: one step means at most 45 degrees.
A_STAR_HEURISTIC_WEIGHT = 1.45
A_STAR_USE_POSE_GRID = False
START_ALIGNMENT_DISTANCE = 2.2
START_ALIGNMENT_MIN_SPEED = 0.35
START_STEER_LIMIT = math.radians(14.0)
PARKING_SUCCESS_IOU = 0.30
PARKING_FINAL_STOP_IOU = 0.50
PARKING_FINAL_STOP_DISTANCE = 0.30
PARKING_CENTER_STOP_DISTANCE = 1.5
PARKING_STOP_SPEED = 0.20
PARKING_FULLY_INSIDE_STOP_DELAY = 1.0
PARKING_MIN_ROLL_SPEED = 0.43
SECOND_APPROACH_MIN_SPEED = 0.25
PARKING_STRAIGHTEN_YAW_TOLERANCE = math.radians(1.5)
PARKING_STRAIGHTEN_MIN_IOU = 0.75
TUNABLE_POLICY_NAMES = {
    "PARKING_MIN_ROLL_SPEED",
    "SECOND_APPROACH_MIN_SPEED",
    "REAR_REVERSE_SPEED",
    "REAR_REVERSE_MIN_SPEED",
    "PARKING_PREPARE_BRAKE",
    "PARKING_FINAL_STOP_IOU",
    "PARKING_FINAL_STOP_DISTANCE",
    "PARKING_CENTER_STOP_DISTANCE",
}
PARKING_POLICY_CANDIDATES_CACHE: Optional[Dict[str, Dict[str, float]]] = None
MIN_TUNED_POLICY_SCORE = 90.0
PARKING_ORIENTATION_ALIGNMENT_THRESHOLD = math.cos(math.radians(48.0))
BOTTOM_ROW_PRETURN_LEFT_OFFSET = 1.7
BOTTOM_ROW_PRETURN_UP_OFFSET = 3.0
BOTTOM_ROW_PRETURN_CURVE_SAMPLES = 10
PRETURN_ROW_TARGETS = (
    (range(0, 6), 0),
    (range(11, 16), 11),
    (range(22, 27), 22),
)
FRONT_ROUTE_EARLY_X_OFFSET = 3.5
FRONT_ROUTE_GOAL_Y_TOLERANCE = 2.0
FRONT_ROUTE_LATE_PROGRESS = 0.75
FRONT_ROUTE_MAX_LENGTH_RATIO = 1.20


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
    blocked_grid_cache: Optional[Tuple[int, int, float, List[List[bool]]]] = None
    line_penalty_grid_cache: Optional[Tuple[int, int, float, List[List[float]]]] = None
    pose_collision_grid_cache: Optional[Tuple[int, int, float, List[List[List[bool]]]]] = None
    parking_segment_ready: bool = False
    parking_state: str = PARKING_STATE_APPROACH
    debug_approach_point: Optional[Tuple[float, float]] = None
    debug_preturn_point: Optional[Tuple[float, float]] = None
    debug_astar_path: Optional[List[Tuple[float, float]]] = None
    debug_parking_points: Optional[List[Tuple[float, float]]] = None
    rear_second_p1: Optional[Tuple[float, float]] = None
    parking_mode: str = "front_in"
    parking_maneuver: str = "front_in"
    rear_last_steer: float = 0.0
    last_command_steer: float = 0.0
    fully_inside_slot_since: Optional[float] = None
    parking_prepare_full_stop_seen: bool = False
    approach_reached_latched: bool = False
    approach_stuck_anchor: Optional[Tuple[float, float, float]] = None
    approach_recovery_start: Optional[Tuple[float, float]] = None
    approach_forward_recovery_start: Optional[Tuple[float, float]] = None

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
        self.blocked_grid_cache = None
        self.line_penalty_grid_cache = None
        self.pose_collision_grid_cache = None
        self.parking_segment_ready = False
        self.parking_state = PARKING_STATE_APPROACH
        self.debug_approach_point = None
        self.debug_preturn_point = None
        self.debug_astar_path = None
        self.debug_parking_points = None
        self.rear_second_p1 = None
        self.parking_mode = self._expected_parking_mode()
        self.parking_maneuver = "front_in"
        self.rear_last_steer = 0.0
        self.last_command_steer = 0.0
        self.fully_inside_slot_since = None
        self.parking_prepare_full_stop_seen = False
        self.approach_reached_latched = False
        self.approach_stuck_anchor = None
        self.approach_recovery_start = None
        self.approach_forward_recovery_start = None
        self.rl_speed = RLSpeedController(enabled=USE_RL_SPEED_CONTROL)
        self._warm_planning_caches(warm_pose_grid=False)
        print(f"[algo] parking_mode={self.parking_mode}")
        print(f"[algo] rl_speed_control={'ON' if USE_RL_SPEED_CONTROL else 'OFF'}")

    def compute_path(self, obs: Dict[str, Any]) -> None:
        """Plan a path from the current pose to the target parking slot."""

        self.waypoints.clear()
        self.waypoint_index = 0
        self.parking_segment_ready = False
        self.parking_state = PARKING_STATE_APPROACH
        self.debug_approach_point = None
        self.debug_preturn_point = None
        self.debug_astar_path = None
        self.debug_parking_points = None
        self.rear_second_p1 = None
        self.rear_last_steer = 0.0
        self.last_command_steer = 0.0
        self.fully_inside_slot_since = None
        self.parking_prepare_full_stop_seen = False
        self.approach_reached_latched = False
        self.approach_stuck_anchor = None
        self.approach_recovery_start = None
        self.approach_forward_recovery_start = None
        state = obs.get("state", {})
        self.parking_mode = self._expected_parking_mode(obs)
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
        self.parking_maneuver = self._select_parking_maneuver(slot, target_pose)
        self._apply_tuned_policy(slot)
        candidates = self._approach_candidates(slot, target_pose[2])
        best_plan = self._select_best_plan(
            (start[0], start[1]),
            candidates,
            target_pose,
            start_yaw=start[2],
        )

        if best_plan is None:
            self.planning_fail_reason = "all_approach_candidates_failed"
            print("[algo] planning failed: A* failed, no fallback path will be used")
            return
        else:
            self.planning_fail_reason = None
            approach_pose, grid_path, cost = best_plan
            self.debug_approach_point = (approach_pose[0], approach_pose[1])
            self.debug_astar_path = list(grid_path)
            self.rear_second_p1 = (
                (approach_pose[0], approach_pose[1])
                if self._is_rear_parking_mode()
                else None
            )
            print(
                "[algo] planning success:"
                f" candidates={len(candidates)}"
                f" selected=({approach_pose[0]:.2f}, {approach_pose[1]:.2f})"
                f" a_star_points={len(grid_path)} cost={cost:.2f}"
                f" parking_mode={self.parking_mode}"
                f" maneuver={self.parking_maneuver}"
            )

        grid_path = self._avoid_late_turnaround_path(
            start=start,
            approach_pose=approach_pose,
            original_path=grid_path,
        )
        simplified = self._simplify_path(grid_path, spacing=1.0)
        simplified = self._prepend_start_alignment(simplified, start)
        simplified = self._insert_row_preturn(
            points=simplified,
            start=start,
            slot=slot,
        )
        t_points = self._rear_y_parking_points(
            (approach_pose[0], approach_pose[1], target_pose[2]),
            target_pose,
        )
        t_points[0] = (approach_pose[0], approach_pose[1])
        self.debug_parking_points = self._rear_second_debug_points(t_points)
        points = simplified
        self.debug_astar_path = list(points)
        self.waypoints = self._points_to_waypoints(
            points,
            final_yaw=points[-1][2] if points and len(points[-1]) > 2 else target_pose[2],
            gear="D",
        )

        initial_clearance = self._estimate_min_obstacle_distance((start[0], start[1]), yaw=start[2])
        self.min_obstacle_distance = min(self.min_obstacle_distance, initial_clearance)
        print(
            "[algo] path ready:"
            f" waypoints={len(self.waypoints)}"
            f" stage=approach_only"
            f" parking_mode={self.parking_mode}"
            f" maneuver={self.parking_maneuver}"
            f" target=({target_pose[0]:.2f}, {target_pose[1]:.2f}, "
            f"yaw={math.degrees(target_pose[2]):.1f}deg)"
            f" min_obstacle_dist~{initial_clearance:.2f}m"
        )

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
        dt = float(limits.get("dt", 1.0 / 60.0))
        wheelbase = float(limits.get("L", 2.6))
        max_steer = float(limits.get("maxSteer", math.radians(35.0)))

        slot = obs.get("target_slot") or []
        signature = tuple(round(float(v), 3) for v in slot) if len(slot) == 4 else None
        if not self.waypoints or signature != self.target_signature:
            self.compute_path(obs)

        if not self.waypoints:
            return self._command(steer=0.0, accel=0.0, brake=0.8, gear="D")

        final_wp = self.waypoints[-1]
        if len(slot) == 4:
            target_center = self._slot_center(slot)
            target_pose = self._target_pose(slot)
            final_yaw = target_pose[2]
            final_dist = math.hypot(target_center[0] - x, target_center[1] - y)
            center_tolerance = self._slot_center_tolerance(slot)
            slot_entered = self._point_in_slot(slot, x, y, margin=0.05)
        else:
            target_center = (final_wp[0], final_wp[1])
            target_pose = (final_wp[0], final_wp[1], final_wp[2])
            final_yaw = final_wp[2]
            final_dist = math.hypot(final_wp[0] - x, final_wp[1] - y)
            center_tolerance = 0.55
            slot_entered = False
        final_yaw_error = abs(self._wrap_to_pi(final_yaw - yaw))
        obstacle_dist = self._estimate_min_obstacle_distance((x, y), yaw=yaw)
        self.min_obstacle_distance = min(self.min_obstacle_distance, obstacle_dist)
        collision_risk = False
        parking_iou = self._slot_iou(slot, x, y, yaw) if len(slot) == 4 else 0.0
        actual_orientation = (
            self._parking_orientation_label(slot, yaw) if len(slot) == 4 else "unknown"
        )
        orientation_ok = actual_orientation == self.parking_mode
        simulator_ready_to_stop = (
            len(slot) == 4
            and self.approach_reached_latched
            and parking_iou >= PARKING_FINAL_STOP_IOU
            and final_dist <= PARKING_FINAL_STOP_DISTANCE
            and orientation_ok
        )
        vehicle_fully_inside_slot = (
            len(slot) == 4
            and self._vehicle_fully_inside_slot(slot, x, y, yaw)
        )
        center_is_close_to_slot_center = (
            len(slot) == 4
            and final_dist <= PARKING_CENTER_STOP_DISTANCE
        )
        fully_inside_stop_ready = self._fully_inside_stop_ready(
            current_time=t,
            fully_inside=(
                self.approach_reached_latched
                and vehicle_fully_inside_slot
                and center_is_close_to_slot_center
            ),
        )
        planner_ready_to_stop = simulator_ready_to_stop or fully_inside_stop_ready

        if planner_ready_to_stop and speed <= PARKING_STOP_SPEED:
            self._log_evaluation(
                parking_success=True,
                fail_reason="none",
                final_position_error=final_dist,
                final_yaw_error=final_yaw_error,
                collision=collision_risk,
                force=True,
            )
            print(
                "[algo] parking succeeded:"
                f" pos_error={final_dist:.2f}m"
                f" slot_iou={parking_iou:.2f}"
                f" orientation={actual_orientation}/{self.parking_mode}"
                f" speed={speed:.2f}m/s"
                f" yaw_error={math.degrees(final_yaw_error):.1f}deg"
                f" steps={self.step_count}"
                f" min_obstacle_dist~{self.min_obstacle_distance:.2f}m"
            )
            self.parking_state = PARKING_STATE_STOP
            stop_gear = "R" if self._is_rear_parking_mode() else "D"
            return self._command(steer=0.0, accel=0.0, brake=1.0, gear=stop_gear)
        if planner_ready_to_stop:
            stop_gear = "R" if self._is_rear_parking_mode() else "D"
            return self._command(
                steer=self.last_command_steer,
                accel=0.0,
                brake=1.0,
                gear=stop_gear,
            )

        self._advance_waypoint_index(x, y)
        approach_remaining = self._approach_remaining(x, y)
        approach_reached_tolerance = (
            REAR_REVERSE_START_TOLERANCE
            if self._is_rear_parking_mode()
            else FRONT_APPROACH_REACHED_TOLERANCE
        )
        if len(slot) == 4:
            approach_target_reached = (
                self.debug_approach_point is not None
                and (
                    approach_remaining <= approach_reached_tolerance
                    or self._approach_target_passed(x, y)
                )
            )
            direct_target_reached = (
                self.debug_approach_point is None
                and final_dist <= PARKING_ALIGN_DISTANCE
            )
            if (
                not self.parking_segment_ready
                and self.parking_state == PARKING_STATE_APPROACH
                and (approach_target_reached or direct_target_reached)
            ):
                if not self.approach_reached_latched:
                    print(
                        "[algo] approach target reached: holding for parking segment"
                        f" approach_remaining={approach_remaining:.2f}m"
                        f" pos_error={final_dist:.2f}m"
                        f" speed={speed:.2f}m/s"
                    )
                self.approach_reached_latched = True
            if self.parking_state == PARKING_STATE_SECOND_APPROACH:
                parking_entry_triggered = (
                    not self.parking_segment_ready
                    and approach_remaining <= REAR_REVERSE_START_TOLERANCE
                )
            else:
                parking_entry_triggered = (
                    not self.parking_segment_ready
                    and self.approach_reached_latched
                )
            if parking_entry_triggered:
                if self.parking_state == PARKING_STATE_APPROACH:
                    self.parking_state = PARKING_STATE_PREPARE_STOP
                    self.parking_prepare_full_stop_seen = False
                    print(
                        "[algo] parking state: PREPARE_STOP"
                        f" mode={self.parking_mode}"
                        f" approach_remaining={approach_remaining:.2f}m"
                        f" pos_error={final_dist:.2f}m"
                        f" speed={speed:.2f}m/s"
                    )
                if not self.parking_prepare_full_stop_seen:
                    if speed > PARKING_PREPARE_FULL_STOP_SPEED:
                        return self._command(
                            steer=self.last_command_steer,
                            accel=0.0,
                            brake=PARKING_PREPARE_BRAKE,
                            gear="D",
                        )
                    self.parking_prepare_full_stop_seen = True
                    print(
                        "[algo] parking state: full stop confirmed"
                        f" speed={speed:.3f}m/s"
                    )
                    return self._command(
                        steer=self.last_command_steer,
                        accel=0.0,
                        brake=PARKING_PREPARE_BRAKE,
                        gear="D",
                    )
                segment_ready = self._ensure_parking_segment(
                    x=x,
                    y=y,
                    target_pose=target_pose,
                )
                if not segment_ready:
                    return self._command(
                        steer=self.last_command_steer,
                        accel=0.0,
                        brake=PARKING_PREPARE_BRAKE,
                        gear="D",
                    )
            if (
                self.parking_state == PARKING_STATE_PREPARE_STOP
                and not self.parking_segment_ready
            ):
                return self._command(
                    steer=self.last_command_steer,
                    accel=0.0,
                    brake=PARKING_PREPARE_BRAKE,
                    gear="D",
                )
            final_wp = self.waypoints[-1]
            approach_remaining = self._approach_remaining(x, y)
        approach_pending = (
            self.debug_approach_point is not None
            and not self.parking_segment_ready
            and not self.approach_reached_latched
            and self.parking_state == PARKING_STATE_APPROACH
            and approach_remaining > approach_reached_tolerance
        )
        approach_tracking = approach_pending and final_dist > 2.2
        in_parking_mode = (
            (final_dist < PARKING_ALIGN_DISTANCE and not approach_pending)
            or self.parking_segment_ready
            or self.parking_state != PARKING_STATE_APPROACH
        )
        second_approach_mode = self.parking_state == PARKING_STATE_SECOND_APPROACH
        if approach_tracking and not second_approach_mode:
            self._advance_approach_waypoint_index(x, y)
        tracking_reference_idx = (
            self.waypoint_index
            if second_approach_mode
            else min(self.waypoint_index + 1, len(self.waypoints) - 1)
        )
        tracking_reference = self.waypoints[tracking_reference_idx]
        tracking_yaw_error = self._tracking_yaw_error(
            x=x,
            y=y,
            yaw=yaw,
            target_wp=tracking_reference,
            reverse=False,
        )
        control_yaw_error = (
            tracking_yaw_error
            if second_approach_mode
            else final_yaw_error if in_parking_mode else tracking_yaw_error
        )
        lookahead = (
            0.0
            if second_approach_mode
            else self._adaptive_lookahead(speed, final_dist, control_yaw_error)
        )
        if second_approach_mode:
            target_idx = self.waypoint_index
        elif approach_tracking:
            target_idx = self._approach_lookahead_index(x, y, lookahead=lookahead)
        else:
            target_idx = self._lookahead_index(x, y, lookahead=lookahead)
        target_wp = self.waypoints[target_idx]
        gear = self._segment_gear_for_target(target_idx)
        approach_curvature = (
            self._approach_path_curvature(self.waypoint_index)
            if approach_tracking
            else 0.0
        )
        moving_to_approach = (
            approach_tracking
            and final_dist > 2.2
            and gear == "D"
        )
        clearance_vehicle_margin = (
            PARKING_VEHICLE_RECT_MARGIN
            if in_parking_mode
            else APPROACH_DRIVING_VEHICLE_MARGIN
            if moving_to_approach
            else VEHICLE_RECT_MARGIN
        )
        forward_clearance = self._estimate_forward_clearance(
            x=x,
            y=y,
            yaw=yaw,
            reverse=False,
            steer=0.0,
            wheelbase=wheelbase,
            vehicle_margin=clearance_vehicle_margin,
        )
        stuck_command = self._approach_stuck_recovery_command(
            obs=obs,
            x=x,
            y=y,
            yaw=yaw,
            t=t,
            wheelbase=wheelbase,
            moving_to_approach=moving_to_approach,
            front_clearance=forward_clearance,
            vehicle_margin=clearance_vehicle_margin,
        )
        if stuck_command is not None:
            return stuck_command

        if self._is_rear_parking_mode() and self.parking_segment_ready:
            return self._rear_parking_state_command(
                x=x,
                y=y,
                yaw=yaw,
                speed=speed,
                target_pose=target_pose,
                final_dist=final_dist,
                final_yaw_error=final_yaw_error,
                parking_iou=parking_iou,
                wheelbase=wheelbase,
                max_steer=max_steer,
                current_time=t,
                dt=dt,
            )

        if in_parking_mode and self.parking_state == PARKING_STATE_APPROACH:
            self.parking_state = PARKING_STATE_ALIGN_CHECK

        if in_parking_mode:
            target_wp = self._parking_entry_target(
                x=x,
                y=y,
                planned_target=target_wp,
                target_center=target_center,
                final_yaw=final_yaw,
            )
            gear = target_wp[3]
        if in_parking_mode:
            control_yaw_error = final_yaw_error
        else:
            control_yaw_error = self._tracking_yaw_error(
                x=x,
                y=y,
                yaw=yaw,
                target_wp=target_wp,
                reverse=(gear == "R"),
            )

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
        if approach_tracking:
            steer = self._blend_approach_prealign_steer(
                steer=steer,
                x=x,
                y=y,
                yaw=yaw,
                approach_remaining=approach_remaining,
                target_pose=target_pose,
                wheelbase=wheelbase,
                max_steer=max_steer,
            )
        if (
            in_parking_mode
            and final_yaw_error <= PARKING_STRAIGHTEN_YAW_TOLERANCE
            and self._parking_point3_reached(x, y, target_pose)
            and parking_iou >= PARKING_STRAIGHTEN_MIN_IOU
        ):
            steer = 0.0
        if (
            not in_parking_mode
            and gear == "D"
            and speed < START_ALIGNMENT_MIN_SPEED
        ):
            steer = max(-START_STEER_LIMIT, min(START_STEER_LIMIT, steer))

        front_clearance = self._estimate_forward_clearance(
            x=x,
            y=y,
            yaw=yaw,
            reverse=(gear == "R"),
            steer=steer,
            wheelbase=wheelbase,
            vehicle_margin=clearance_vehicle_margin,
        )
        collision_risk = front_clearance <= OBSTACLE_STOP_DISTANCE
        if collision_risk:
            self._log_evaluation(
                parking_success=False,
                fail_reason="front_collision_risk",
                final_position_error=final_dist,
                final_yaw_error=final_yaw_error,
                collision=True,
                current_time=t,
            )
            if moving_to_approach:
                return self._command(
                    steer=steer * 0.4,
                    accel=0.0,
                    brake=0.0,
                    gear=gear,
                )
            return self._command(
                steer=steer * 0.4,
                accel=0.0,
                brake=1.0,
                gear=gear,
            )

        front_is_clear = front_clearance >= FRONT_CLEAR_DISTANCE
        rule_speed = self._target_speed(
            final_dist,
            control_yaw_error,
            steer,
            front_clearance,
        )
        approach_deviation = 0.0
        approach_on_path = False
        approach_recovering_from_deviation = False
        if moving_to_approach:
            approach_deviation = self._approach_path_deviation(x, y)
            approach_on_path = approach_deviation <= APPROACH_ON_PATH_MARGIN
            rule_speed = self._approach_speed(
                base_speed=rule_speed,
                approach_remaining=approach_remaining,
                steer_abs=0.0 if approach_on_path else abs(steer),
                front_clearance=front_clearance,
            )
            if (
                not approach_on_path
                and approach_curvature >= APPROACH_CURVATURE_SLOW_ANGLE
            ):
                rule_speed = min(rule_speed, APPROACH_CURVATURE_SLOW_SPEED)
            if (
                approach_deviation >= APPROACH_PATH_DEVIATION_SPEED_DISTANCE
                and front_clearance >= OBSTACLE_SLOW_DISTANCE
            ):
                approach_recovering_from_deviation = True
                rule_speed = max(rule_speed, APPROACH_RECOVERY_MIN_SPEED)
        if gear == "R":
            rule_speed = min(rule_speed, 0.55)
        if (
            second_approach_mode
            and gear == "D"
            and front_clearance >= OBSTACLE_STOP_DISTANCE
        ):
            rule_speed = max(rule_speed, SECOND_APPROACH_MIN_SPEED)
            rule_speed = min(rule_speed, 3.0)
        if (
            in_parking_mode
            and not second_approach_mode
            and not planner_ready_to_stop
            and front_clearance >= OBSTACLE_STOP_DISTANCE
        ):
            rule_speed = max(rule_speed, PARKING_MIN_ROLL_SPEED)
        if (
            not in_parking_mode
            and gear == "D"
            and not approach_recovering_from_deviation
            and not approach_on_path
            and speed < START_ALIGNMENT_MIN_SPEED
            and abs(steer) > math.radians(8.0)
        ):
            rule_speed = min(rule_speed, 1.20)
        target_speed = self.rl_speed.adjust_target_speed(
            rule_speed=rule_speed,
            final_dist=final_dist,
            yaw_error=control_yaw_error,
            steer_abs=abs(steer),
            obstacle_dist=front_clearance,
        )
        straight_clear_full_accel = (
            gear == "D"
            and not in_parking_mode
            and speed >= START_ALIGNMENT_MIN_SPEED
            and front_is_clear
            and abs(steer) < math.radians(4.0)
        )
        approach_full_accel = (
            moving_to_approach
            and front_clearance >= 2.0
            and abs(steer) < math.radians(18.0)
        )
        approach_recovery_accel = (
            moving_to_approach
            and approach_recovering_from_deviation
            and front_clearance >= 2.0
        )
        accel, brake = self._speed_command(
            speed=speed,
            target_speed=target_speed,
            front_is_clear=front_is_clear and final_dist > 3.0,
            force_full_accel=straight_clear_full_accel or approach_full_accel or approach_recovery_accel,
            gentle_brake=in_parking_mode and not planner_ready_to_stop,
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
                f" approach_remaining={approach_remaining:.2f}m"
                f" center_tolerance={center_tolerance:.2f}m"
                f" slot_entered={slot_entered}"
                f" slot_iou={parking_iou:.2f}"
                f" orientation={actual_orientation}/{self.parking_mode}"
                f" parking_state={self.parking_state}"
                f" parking_mode={self.parking_mode}"
                f" maneuver={self.parking_maneuver}"
                f" tracking_yaw_error={math.degrees(tracking_yaw_error):.1f}deg"
                f" final_yaw_error={math.degrees(final_yaw_error):.1f}deg"
                f" min_obstacle_dist~{self.min_obstacle_distance:.2f}m"
                f" front_clearance~{front_clearance:.2f}m"
                f" vehicle_margin={clearance_vehicle_margin:.2f}m"
                f" gear={gear}"
                f" lookahead={lookahead:.2f}m"
                f" path_curvature={math.degrees(approach_curvature):.1f}deg"
                f" path_deviation={approach_deviation:.2f}m"
                f" on_path={approach_on_path}"
                f" recovery_accel={approach_recovery_accel}"
                f" rule_speed={rule_speed:.2f}m/s"
                f" speed={target_speed:.2f}m/s"
                f" rl={'ON' if self.rl_speed.enabled else 'OFF'}"
                f" rl_state={self.rl_speed.last_state}"
                f" rl_action={self.rl_speed.last_action}"
            )

        return self._command(steer=steer, accel=accel, brake=brake, gear=gear)

    def _command(self, steer: float, accel: float, brake: float, gear: str) -> Dict[str, Any]:
        self.last_command_steer = steer
        command: Dict[str, Any] = {
            "steer": steer,
            "accel": accel,
            "brake": brake,
            "gear": gear,
        }
        if self.debug_approach_point is not None:
            command["debug_approach_point"] = [
                float(self.debug_approach_point[0]),
                float(self.debug_approach_point[1]),
            ]
        if self.debug_preturn_point is not None:
            command["debug_preturn_point"] = [
                float(self.debug_preturn_point[0]),
                float(self.debug_preturn_point[1]),
            ]
        if self.debug_astar_path:
            debug_path = self._debug_path_with_parking_markers(
                self.debug_astar_path,
                self.debug_parking_points,
            )
            command["debug_astar_path"] = [
                [float(point[0]), float(point[1])]
                for point in debug_path
            ]
        if self.debug_parking_points:
            command["debug_parking_points"] = [
                [float(point[0]), float(point[1])]
                for point in self.debug_parking_points
            ]
        return command

    def _debug_path_with_parking_markers(
        self,
        path: List[Tuple[float, float]],
        markers: Optional[List[Tuple[float, float]]],
    ) -> List[Tuple[float, float]]:
        if not markers:
            return path
        result: List[Tuple[float, float]] = []
        marker_idx = 0
        marker_radius = 0.22
        for point in path:
            result.append(point)
            if marker_idx >= len(markers):
                continue
            marker = markers[marker_idx]
            if math.hypot(point[0] - marker[0], point[1] - marker[1]) > 0.35:
                continue
            mx, my = marker
            result.extend(
                [
                    (mx + marker_radius, my),
                    (mx, my + marker_radius),
                    (mx - marker_radius, my),
                    (mx, my - marker_radius),
                    (mx + marker_radius, my),
                    (mx, my),
                ]
            )
            marker_idx += 1
        return result

    def _approach_stuck_recovery_command(
        self,
        obs: Dict[str, Any],
        x: float,
        y: float,
        yaw: float,
        t: float,
        wheelbase: float,
        moving_to_approach: bool,
        front_clearance: float,
        vehicle_margin: float,
    ) -> Optional[Dict[str, Any]]:
        if self.approach_forward_recovery_start is not None:
            forward_distance = math.hypot(
                x - self.approach_forward_recovery_start[0],
                y - self.approach_forward_recovery_start[1],
            )
            forward_clearance = self._estimate_forward_clearance(
                x=x,
                y=y,
                yaw=yaw,
                reverse=False,
                steer=0.0,
                wheelbase=wheelbase,
                vehicle_margin=vehicle_margin,
            )
            if (
                forward_distance >= APPROACH_STUCK_FORWARD_DISTANCE
                or forward_clearance < APPROACH_STUCK_FORWARD_STOP_CLEARANCE
            ):
                print(
                    "[algo] approach recovery: forward complete, replanning"
                    f" forward_distance={forward_distance:.2f}m"
                    f" forward_clearance={forward_clearance:.2f}m"
                )
                self.approach_forward_recovery_start = None
                self.approach_stuck_anchor = None
                self.compute_path(obs)
                return self._command(steer=0.0, accel=0.0, brake=0.0, gear="D")
            return self._command(steer=0.0, accel=0.45, brake=0.0, gear="D")

        if self.approach_recovery_start is not None:
            reverse_distance = math.hypot(
                x - self.approach_recovery_start[0],
                y - self.approach_recovery_start[1],
            )
            reverse_clearance = self._estimate_forward_clearance(
                x=x,
                y=y,
                yaw=yaw,
                reverse=True,
                steer=0.0,
                wheelbase=wheelbase,
                vehicle_margin=vehicle_margin,
            )
            if reverse_clearance < APPROACH_STUCK_REVERSE_STOP_CLEARANCE:
                if front_clearance > APPROACH_STUCK_FORWARD_STOP_CLEARANCE:
                    print(
                        "[algo] approach recovery: rear blocked, moving forward"
                        f" reverse_distance={reverse_distance:.2f}m"
                        f" rear_clearance={reverse_clearance:.2f}m"
                        f" front_clearance={front_clearance:.2f}m"
                    )
                    self.approach_recovery_start = None
                    self.approach_forward_recovery_start = (x, y)
                    return self._command(steer=0.0, accel=0.45, brake=0.0, gear="D")
                print(
                    "[algo] approach recovery: front and rear blocked, replanning"
                    f" rear_clearance={reverse_clearance:.2f}m"
                    f" front_clearance={front_clearance:.2f}m"
                )
                self.approach_recovery_start = None
                self.approach_stuck_anchor = None
                self.compute_path(obs)
                return self._command(steer=0.0, accel=0.0, brake=0.8, gear="D")
            if reverse_distance >= APPROACH_STUCK_REVERSE_DISTANCE:
                print(
                    "[algo] approach recovery: reverse complete, replanning"
                    f" reverse_distance={reverse_distance:.2f}m"
                    f" reverse_clearance={reverse_clearance:.2f}m"
                )
                self.approach_recovery_start = None
                self.approach_stuck_anchor = None
                self.compute_path(obs)
                return self._command(steer=0.0, accel=0.0, brake=0.0, gear="D")
            return self._command(steer=0.0, accel=0.45, brake=0.0, gear="R")

        if not moving_to_approach:
            self.approach_stuck_anchor = None
            return None

        if self.approach_stuck_anchor is None:
            self.approach_stuck_anchor = (t, x, y)
            return None

        anchor_t, anchor_x, anchor_y = self.approach_stuck_anchor
        elapsed = t - anchor_t
        moved = math.hypot(x - anchor_x, y - anchor_y)
        if moved >= APPROACH_STUCK_MIN_DISTANCE:
            self.approach_stuck_anchor = (t, x, y)
            return None
        if elapsed < APPROACH_STUCK_SECONDS:
            return None

        self.approach_stuck_anchor = None
        if front_clearance < APPROACH_STUCK_FRONT_BLOCKED:
            rear_clearance = self._estimate_forward_clearance(
                x=x,
                y=y,
                yaw=yaw,
                reverse=True,
                steer=0.0,
                wheelbase=wheelbase,
                vehicle_margin=vehicle_margin,
            )
            if rear_clearance < APPROACH_STUCK_REVERSE_STOP_CLEARANCE:
                self.approach_forward_recovery_start = (x, y)
                print(
                    "[algo] approach stuck: rear blocked, moving forward before replan"
                    f" moved={moved:.2f}m/{APPROACH_STUCK_SECONDS:.1f}s"
                    f" rear_clearance={rear_clearance:.2f}m"
                    f" front_clearance={front_clearance:.2f}m"
                )
                return self._command(steer=0.0, accel=0.45, brake=0.0, gear="D")
            # Temporarily disable reverse recovery during approach driving.
            # It can create confusing backward motion while debugging path
            # tracking; replan from the current pose instead.
            print(
                "[algo] approach stuck: front blocked, reverse recovery disabled; replanning"
                f" moved={moved:.2f}m/{APPROACH_STUCK_SECONDS:.1f}s"
                f" front_clearance={front_clearance:.2f}m"
            )
            self.compute_path(obs)
            return self._command(steer=0.0, accel=0.0, brake=0.0, gear="D")

        print(
            "[algo] approach stuck: replanning from current pose"
            f" moved={moved:.2f}m/{APPROACH_STUCK_SECONDS:.1f}s"
            f" front_clearance={front_clearance:.2f}m"
        )
        self.compute_path(obs)
        return self._command(steer=0.0, accel=0.0, brake=0.0, gear="D")

    def _rear_parking_state_command(
        self,
        x: float,
        y: float,
        yaw: float,
        speed: float,
        target_pose: Tuple[float, float, float],
        final_dist: float,
        final_yaw_error: float,
        parking_iou: float,
        wheelbase: float,
        max_steer: float,
        current_time: float,
        dt: float,
    ) -> Dict[str, Any]:
        target_yaw = target_pose[2]

        if self.parking_state == PARKING_STATE_ALIGN_CHECK:
            if speed > 0.45:
                return self._command(steer=0.0, accel=0.0, brake=0.35, gear="D")
            if final_yaw_error <= REAR_ALIGN_YAW_TOLERANCE:
                self.parking_state = PARKING_STATE_REVERSE_PARKING
                self.waypoint_index = 0
                self.rear_last_steer = 0.0
                print(
                    "[algo] rear parking state: REVERSE_PARKING_MODE"
                    f" yaw_error={math.degrees(final_yaw_error):.1f}deg"
                    f" pos_error={final_dist:.2f}m"
                )
            else:
                align_target_x = x + math.cos(target_yaw) * 3.0
                align_target_y = y + math.sin(target_yaw) * 3.0
                steer = self._pure_pursuit_steer(
                    x=x,
                    y=y,
                    yaw=yaw,
                    target_x=align_target_x,
                    target_y=align_target_y,
                    wheelbase=wheelbase,
                    max_steer=max_steer,
                    reverse=False,
                )
                steer = self._limit_rear_steer(steer, dt, max_steer)
                target_speed = REAR_ALIGN_SPEED if final_yaw_error > math.radians(18.0) else 0.25
                accel, brake = self._speed_command(
                    speed=speed,
                    target_speed=target_speed,
                    front_is_clear=False,
                    gentle_brake=True,
                )
                return self._command(steer=steer, accel=accel, brake=brake, gear="D")

        if self.parking_state == PARKING_STATE_STOP:
            return self._command(steer=0.0, accel=0.0, brake=1.0, gear="R")

        self.parking_state = PARKING_STATE_REVERSE_PARKING
        # Final stop is handled globally after the vehicle rectangle is fully
        # inside the slot for PARKING_FULLY_INSIDE_STOP_DELAY seconds.

        self._advance_waypoint_index(x, y)
        point3_x, point3_y, _ = self._reverse_start_pose(target_pose)
        point3_reached = math.hypot(point3_x - x, point3_y - y) <= REAR_POINT3_REACHED_TOLERANCE
        if point3_reached and len(self.waypoints) >= 2:
            self.waypoint_index = len(self.waypoints) - 1
        target_idx = self._lookahead_index(x, y, lookahead=0.70)
        target_wp = self.waypoints[target_idx]
        steer = self._pure_pursuit_steer(
            x=x,
            y=y,
            yaw=yaw,
            target_x=target_wp[0],
            target_y=target_wp[1],
            wheelbase=wheelbase,
            max_steer=max_steer,
            reverse=True,
        )
        steer = self._limit_rear_steer(steer, dt, max_steer)
        if (
            point3_reached
            and final_yaw_error <= PARKING_STRAIGHTEN_YAW_TOLERANCE
            and parking_iou >= PARKING_STRAIGHTEN_MIN_IOU
        ):
            steer = 0.0
            self.rear_last_steer = 0.0
        reverse_clearance = self._estimate_forward_clearance(
            x=x,
            y=y,
            yaw=yaw,
            reverse=True,
            steer=steer,
            wheelbase=wheelbase,
            vehicle_margin=PARKING_VEHICLE_RECT_MARGIN,
        )
        target_speed = REAR_REVERSE_FINAL_SPEED if final_dist < 1.4 else REAR_REVERSE_SPEED
        target_speed = max(target_speed, REAR_REVERSE_MIN_SPEED, PARKING_MIN_ROLL_SPEED)
        if reverse_clearance < PARKING_REVERSE_STOP_CLEARANCE:
            target_speed = 0.0
        accel, brake = self._speed_command(
            speed=speed,
            target_speed=target_speed,
            front_is_clear=False,
            gentle_brake=True,
            accel_deadband=0.03,
        )
        if current_time - self.last_log_time > 2.0:
            self.last_log_time = current_time
            print(
                "[algo] rear parking:"
                f" state={self.parking_state}"
                f" wp={self.waypoint_index}/{len(self.waypoints) - 1}"
                f" pos_error={final_dist:.2f}m"
                f" yaw_error={math.degrees(final_yaw_error):.1f}deg"
                f" reverse_clearance={reverse_clearance:.2f}m"
                f" point3_reached={point3_reached}"
                f" speed={target_speed:.2f}m/s"
            )
        return self._command(steer=steer, accel=accel, brake=brake, gear="R")

    def _parking_point3_reached(
        self,
        x: float,
        y: float,
        target_pose: Tuple[float, float, float],
    ) -> bool:
        if not self.parking_segment_ready:
            return False
        if self._is_rear_parking_mode():
            point3_x, point3_y, _ = self._reverse_start_pose(target_pose)
            return (
                math.hypot(point3_x - x, point3_y - y)
                <= REAR_POINT3_REACHED_TOLERANCE
            )
        if len(self.waypoints) <= 2:
            # Front parking segment can omit point 3 when the car is already
            # at that pose when the segment is created.
            return True
        point3_x, point3_y = self.waypoints[-2][0], self.waypoints[-2][1]
        return (
            self.waypoint_index >= len(self.waypoints) - 2
            or math.hypot(point3_x - x, point3_y - y)
            <= REAR_POINT3_REACHED_TOLERANCE
        )

    def _limit_rear_steer(self, steer: float, dt: float, max_steer: float) -> float:
        max_delta = min(REAR_STEER_RATE_LIMIT, math.radians(180.0) * max(dt, 1e-3))
        delta = max(-max_delta, min(max_delta, steer - self.rear_last_steer))
        limited = self.rear_last_steer + delta
        limited = max(-max_steer, min(max_steer, limited))
        self.rear_last_steer = limited
        return limited

    def _apply_tuned_policy(self, slot: List[float]) -> None:
        group = self._policy_group_for_slot(slot)
        policies = self._load_tuned_policy_candidates()
        policy = policies.get(group)
        if not policy:
            return
        applied: Dict[str, float] = {}
        for name, value in policy.items():
            if name not in TUNABLE_POLICY_NAMES:
                continue
            try:
                globals()[name] = float(value)
                applied[name] = float(value)
            except (TypeError, ValueError):
                continue
        if applied:
            print(
                "[algo] tuned parking policy applied:"
                f" group={group}"
                f" policy={json.dumps(applied, sort_keys=True)}"
            )

    def _policy_group_for_slot(self, slot: List[float]) -> str:
        row = self._target_row_number(slot)
        row_group = "upper" if row == 3 else "lower"
        maneuver_group = "rear_in" if self.parking_maneuver.lower().startswith("rear") else "front_in"
        return f"{row_group}_{maneuver_group}"

    def _target_row_number(self, slot: List[float]) -> int:
        if len(slot) != 4 or not self.map_data:
            return 1
        slots = self.map_data.get("slots") or []
        if not slots:
            return 1
        centers = sorted(
            {
                round(0.5 * (float(candidate[2]) + float(candidate[3])), 2)
                for candidate in slots
                if len(candidate) == 4
            }
        )
        if not centers:
            return 1
        target_y = round(0.5 * (float(slot[2]) + float(slot[3])), 2)
        closest_idx = min(range(len(centers)), key=lambda idx: abs(centers[idx] - target_y))
        return closest_idx + 1

    def _load_tuned_policy_candidates(self) -> Dict[str, Dict[str, float]]:
        global PARKING_POLICY_CANDIDATES_CACHE
        if os.environ.get("PARKING_DISABLE_TUNED_POLICY") == "1":
            return {}
        if PARKING_POLICY_CANDIDATES_CACHE is not None:
            return PARKING_POLICY_CANDIDATES_CACHE
        candidates: Dict[str, Dict[str, float]] = {}
        search_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "parking_policy_candidates.json"),
            os.path.join(os.getcwd(), "parking_policy_candidates.json"),
        ]
        for path in search_paths:
            try:
                with open(path, "r", encoding="utf-8") as fp:
                    payload = json.load(fp)
            except OSError:
                continue
            raw_candidates = payload.get("policy_candidates", {})
            if not isinstance(raw_candidates, dict):
                continue
            for group, item in raw_candidates.items():
                if not isinstance(item, dict):
                    continue
                result = str(item.get("result") or "").lower()
                try:
                    score = float(item.get("score") or 0.0)
                except (TypeError, ValueError):
                    score = 0.0
                if result != "success" or score <= MIN_TUNED_POLICY_SCORE:
                    continue
                policy = item.get("policy", item)
                if not isinstance(policy, dict):
                    continue
                filtered: Dict[str, float] = {}
                for name, value in policy.items():
                    if name not in TUNABLE_POLICY_NAMES:
                        continue
                    try:
                        filtered[name] = float(value)
                    except (TypeError, ValueError):
                        continue
                if filtered:
                    candidates[str(group)] = filtered
            if candidates:
                break
        PARKING_POLICY_CANDIDATES_CACHE = candidates
        return candidates

    def _target_pose(self, slot: List[float]) -> Tuple[float, float, float]:
        cx, cy = self._slot_center(slot)
        yaw = -math.pi / 2.0 if self.parking_mode.lower().startswith("rear") else math.pi / 2.0
        return cx, cy, yaw

    def _expected_parking_mode(self, obs: Optional[Dict[str, Any]] = None) -> str:
        expected = self._find_parking_mode_text(obs)
        if "rear" in expected:
            return "rear_in"
        if "front" in expected:
            return "front_in"
        return "front_in"

    def _find_parking_mode_text(self, obs: Optional[Dict[str, Any]] = None) -> str:
        keys = (
            "expected_orientation",
            "parking_orientation",
            "required_orientation",
            "orientation",
            "requirement",
            "stage",
            "stage_name",
            "map_name",
            "map_id",
        )
        sources: List[Dict[str, Any]] = []
        if obs is not None:
            sources.append(obs)
        if self.map_data:
            sources.append(self.map_data)
        for source in sources:
            for key in keys:
                value = source.get(key)
                if value is None:
                    continue
                text = str(value).lower()
                if "rear" in text or "front" in text:
                    return text
        return ""

    def _is_rear_parking_mode(self) -> bool:
        return self.parking_maneuver.lower().startswith("rear")

    def _select_parking_maneuver(
        self,
        slot: List[float],
        target_pose: Tuple[float, float, float],
    ) -> str:
        open_sign = self._rear_open_side_sign(target_pose, slot=slot)
        desired_front_sign = 1.0 if self.parking_mode.lower().startswith("front") else -1.0
        # Forward parking enters from the open side toward the slot center, so
        # the final front direction is opposite to the opening direction.
        if desired_front_sign * open_sign < 0.0:
            return "front_in"
        return "rear_in"

    def _slot_center(self, slot: List[float]) -> Tuple[float, float]:
        return (
            0.5 * (float(slot[0]) + float(slot[1])),
            0.5 * (float(slot[2]) + float(slot[3])),
        )

    def _slot_center_tolerance(self, slot: List[float]) -> float:
        slot_w = abs(float(slot[1]) - float(slot[0]))
        slot_l = abs(float(slot[3]) - float(slot[2]))
        return max(0.15, 0.05 * min(slot_w, slot_l))

    def _avoid_late_turnaround_path(
        self,
        start: Tuple[float, float, float],
        approach_pose: Tuple[float, float, float],
        original_path: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        if not self.parking_mode.lower().startswith("front") or self.map_extent is None:
            return original_path
        sx, sy, syaw = start
        ax, ay, _ = approach_pose
        dx = ax - sx
        dy = ay - sy
        if abs(dx) < 12.0 or abs(dy) < 8.0 or len(original_path) < 5:
            return original_path

        first_goal_y_idx: Optional[int] = None
        for idx, point in enumerate(original_path):
            if abs(point[1] - ay) <= FRONT_ROUTE_GOAL_Y_TOLERANCE:
                first_goal_y_idx = idx
                break
        if first_goal_y_idx is None:
            return original_path

        progress_when_y_reached = abs(original_path[first_goal_y_idx][0] - sx) / max(abs(dx), 1e-6)
        path_fraction_when_y_reached = first_goal_y_idx / max(len(original_path) - 1, 1)
        if (
            progress_when_y_reached < FRONT_ROUTE_LATE_PROGRESS
            and path_fraction_when_y_reached < FRONT_ROUTE_LATE_PROGRESS
        ):
            return original_path

        direction_x = 1.0 if dx >= 0.0 else -1.0
        via = self._clamp_inside_map(
            sx + direction_x * FRONT_ROUTE_EARLY_X_OFFSET,
            ay,
            margin=0.2,
            vehicle_margin=PARKING_VEHICLE_RECT_MARGIN,
        )
        if math.hypot(via[0] - sx, via[1] - sy) < 1.0:
            return original_path

        first_leg = self._astar_path(
            (sx, sy),
            via,
            start_yaw=syaw,
            goal_yaw=None,
        )
        if not first_leg:
            return original_path
        second_leg = self._astar_path(
            via,
            (ax, ay),
            start_yaw=None,
            goal_yaw=None,
        )
        if not second_leg:
            return original_path

        combined = first_leg[:-1] + second_leg
        if len(combined) < 3:
            return original_path
        if self._path_length(combined) > self._path_length(original_path) * FRONT_ROUTE_MAX_LENGTH_RATIO:
            return original_path
        print(
            "[algo] late-turn route adjusted:"
            f" via=({via[0]:.2f}, {via[1]:.2f})"
            f" original_points={len(original_path)}"
            f" routed_points={len(combined)}"
        )
        return combined

    def _insert_row_preturn(
        self,
        points: List[Tuple[float, float]],
        start: Tuple[float, float, float],
        slot: List[float],
    ) -> List[Tuple[float, float]]:
        if len(points) < 2:
            return points
        sx, sy, syaw = start
        target_idx = self._target_slot_index(slot)
        anchor_idx = self._preturn_anchor_index(target_idx)
        if anchor_idx is None:
            return points
        preturn = self._row_preturn_point(anchor_idx, start)
        if preturn is None:
            return points

        insert_idx = self._preturn_insert_index(points, start)
        before = points[insert_idx - 1]
        after = points[insert_idx]
        px, py = preturn
        if (
            math.hypot(px - before[0], py - before[1]) < 0.8
            or math.hypot(px - after[0], py - after[1]) < 0.8
        ):
            return points
        print(
            "[algo] row preturn inserted:"
            f" target=T{target_idx}"
            f" anchor=T{anchor_idx}"
            f" point=({px:.2f}, {py:.2f})"
            f" insert_idx={insert_idx}"
            f" start_yaw={math.degrees(syaw):.1f}deg"
        )
        self.debug_preturn_point = (px, py)
        curve_points = self._preturn_curve_points(
            points=points,
            insert_idx=insert_idx,
            preturn=(px, py),
        )
        return points[:insert_idx] + curve_points

    def _target_slot_index(self, slot: List[float]) -> Optional[int]:
        if len(slot) != 4 or not self.map_data:
            return None
        target = [float(v) for v in slot]
        for idx, candidate in enumerate(self.map_data.get("slots") or []):
            if len(candidate) != 4:
                continue
            if all(abs(float(candidate[i]) - target[i]) <= 1e-3 for i in range(4)):
                return idx
        return None

    def _preturn_anchor_index(self, target_idx: Optional[int]) -> Optional[int]:
        if target_idx is None:
            return None
        for target_range, anchor_idx in PRETURN_ROW_TARGETS:
            if target_idx in target_range:
                return anchor_idx
        return None

    def _row_preturn_point(
        self,
        anchor_idx: int,
        start: Tuple[float, float, float],
    ) -> Optional[Tuple[float, float]]:
        if not self.map_data:
            return None
        slots = self.map_data.get("slots") or []
        if not (0 <= anchor_idx < len(slots)):
            return None
        t0_approach = self._approach_pose_for_slot_index(0)
        anchor_approach = self._approach_pose_for_slot_index(anchor_idx)
        if t0_approach is None or anchor_approach is None:
            return None
        sx, sy, _ = start
        t0_preturn = self._clamp_inside_map(
            sx - BOTTOM_ROW_PRETURN_LEFT_OFFSET,
            sy + BOTTOM_ROW_PRETURN_UP_OFFSET,
            margin=0.2,
            vehicle_margin=0.0,
        )
        offset_x = t0_preturn[0] - t0_approach[0]
        offset_y = t0_preturn[1] - t0_approach[1]
        return self._clamp_inside_map(
            anchor_approach[0] + offset_x,
            anchor_approach[1] + offset_y,
            margin=0.2,
            vehicle_margin=0.0,
        )

    def _approach_pose_for_slot_index(
        self,
        slot_idx: int,
    ) -> Optional[Tuple[float, float, float]]:
        if not self.map_data:
            return None
        slots = self.map_data.get("slots") or []
        if not (0 <= slot_idx < len(slots)):
            return None
        slot = [float(v) for v in slots[slot_idx]]
        target_pose = self._target_pose(slot)
        old_maneuver = self.parking_maneuver
        try:
            self.parking_maneuver = self._select_parking_maneuver(slot, target_pose)
            candidates = self._approach_candidates(slot, target_pose[2])
            return candidates[0] if candidates else None
        finally:
            self.parking_maneuver = old_maneuver

    def _preturn_curve_points(
        self,
        points: List[Tuple[float, float]],
        insert_idx: int,
        preturn: Tuple[float, float],
    ) -> List[Tuple[float, float]]:
        before = points[insert_idx - 1]
        goal = points[-1]
        samples = max(3, BOTTOM_ROW_PRETURN_CURVE_SAMPLES)
        curve: List[Tuple[float, float]] = []
        first_control = (preturn[0], before[1])
        second_control = (preturn[0], goal[1])
        curve.extend(
            self._quadratic_bezier_points(
                before,
                first_control,
                preturn,
                samples=samples,
                include_start=False,
            )
        )
        curve.extend(
            self._quadratic_bezier_points(
                preturn,
                second_control,
                goal,
                samples=samples,
                include_start=False,
            )
        )
        return curve

    def _quadratic_bezier_points(
        self,
        start: Tuple[float, float],
        control: Tuple[float, float],
        end: Tuple[float, float],
        samples: int,
        include_start: bool,
    ) -> List[Tuple[float, float]]:
        first_step = 0 if include_start else 1
        result: List[Tuple[float, float]] = []
        for step in range(first_step, samples + 1):
            t = step / samples
            one_minus = 1.0 - t
            x = (
                one_minus * one_minus * start[0]
                + 2.0 * one_minus * t * control[0]
                + t * t * end[0]
            )
            y = (
                one_minus * one_minus * start[1]
                + 2.0 * one_minus * t * control[1]
                + t * t * end[1]
            )
            result.append((x, y))
        return result

    def _preturn_insert_index(
        self,
        points: List[Tuple[float, float]],
        start: Tuple[float, float, float],
    ) -> int:
        if len(points) <= 2:
            return 1
        sx, sy, syaw = start
        nx, ny = points[1][0], points[1][1]
        dx = nx - sx
        dy = ny - sy
        along = dx * math.cos(syaw) + dy * math.sin(syaw)
        lateral = abs(-dx * math.sin(syaw) + dy * math.cos(syaw))
        if 0.4 <= along <= START_ALIGNMENT_DISTANCE + 0.8 and lateral <= 0.6:
            return 2
        return 1

    def _slot_iou(self, slot: List[float], x: float, y: float, yaw: float) -> float:
        if len(slot) != 4:
            return 0.0
        slot_rect = tuple(float(v) for v in slot)
        car_poly = self._vehicle_polygon(x, y, yaw, margin=0.0)
        intersection = self._clip_polygon_with_rect(car_poly, slot_rect)
        intersection_area = self._polygon_area(intersection)
        if intersection_area <= 0.0:
            return 0.0
        car_area = self._polygon_area(car_poly)
        slot_area = max(0.0, (slot_rect[1] - slot_rect[0]) * (slot_rect[3] - slot_rect[2]))
        union_area = max(car_area + slot_area - intersection_area, 1e-9)
        return intersection_area / union_area

    def _vehicle_fully_inside_slot(
        self,
        slot: List[float],
        x: float,
        y: float,
        yaw: float,
    ) -> bool:
        if len(slot) != 4:
            return False
        x0, x1, y0, y1 = (float(v) for v in slot)
        for px, py in self._vehicle_polygon(x, y, yaw, margin=0.0):
            if not (x0 <= px <= x1 and y0 <= py <= y1):
                return False
        return True

    def _fully_inside_stop_ready(
        self,
        current_time: float,
        fully_inside: bool,
    ) -> bool:
        if not fully_inside:
            self.fully_inside_slot_since = None
            return False
        if self.fully_inside_slot_since is None:
            self.fully_inside_slot_since = current_time
            print("[algo] vehicle fully inside slot: stop timer started")
            return False
        return (
            current_time - self.fully_inside_slot_since
            >= PARKING_FULLY_INSIDE_STOP_DELAY
        )

    def _parking_orientation_label(self, slot: List[float], yaw: float) -> str:
        if len(slot) != 4:
            return "unknown"
        slot_rect = tuple(float(v) for v in slot)
        width = slot_rect[1] - slot_rect[0]
        height = slot_rect[3] - slot_rect[2]
        forward_x = math.cos(yaw)
        forward_y = math.sin(yaw)
        axis_value = forward_y if height >= width else forward_x
        if abs(axis_value) < PARKING_ORIENTATION_ALIGNMENT_THRESHOLD:
            return "unknown"
        return "front_in" if axis_value >= 0.0 else "rear_in"

    def _polygon_area(self, poly: List[Tuple[float, float]]) -> float:
        if len(poly) < 3:
            return 0.0
        area = 0.0
        for idx, (x1, y1) in enumerate(poly):
            x2, y2 = poly[(idx + 1) % len(poly)]
            area += x1 * y2 - x2 * y1
        return abs(area) * 0.5

    def _clip_polygon_with_rect(
        self,
        poly: List[Tuple[float, float]],
        rect: Tuple[float, float, float, float],
    ) -> List[Tuple[float, float]]:
        xmin, xmax, ymin, ymax = rect

        def clip_edge(points, inside_fn, intersect_fn):
            if not points:
                return []
            clipped = []
            start = points[-1]
            start_inside = inside_fn(start)
            for end in points:
                end_inside = inside_fn(end)
                if end_inside:
                    if not start_inside:
                        clipped.append(intersect_fn(start, end))
                    clipped.append(end)
                elif start_inside:
                    clipped.append(intersect_fn(start, end))
                start = end
                start_inside = end_inside
            return clipped

        def intersect_vertical(start, end, x_bound):
            sx, sy = start
            ex, ey = end
            if abs(ex - sx) < 1e-9:
                return x_bound, sy
            ratio = (x_bound - sx) / (ex - sx)
            return x_bound, sy + ratio * (ey - sy)

        def intersect_horizontal(start, end, y_bound):
            sx, sy = start
            ex, ey = end
            if abs(ey - sy) < 1e-9:
                return sx, y_bound
            ratio = (y_bound - sy) / (ey - sy)
            return sx + ratio * (ex - sx), y_bound

        points = poly
        points = clip_edge(points, lambda point: point[0] >= xmin, lambda s, e: intersect_vertical(s, e, xmin))
        points = clip_edge(points, lambda point: point[0] <= xmax, lambda s, e: intersect_vertical(s, e, xmax))
        points = clip_edge(points, lambda point: point[1] >= ymin, lambda s, e: intersect_horizontal(s, e, ymin))
        points = clip_edge(points, lambda point: point[1] <= ymax, lambda s, e: intersect_horizontal(s, e, ymax))
        return points

    def _point_in_slot(self, slot: List[float], x: float, y: float, margin: float = 0.0) -> bool:
        return (
            float(slot[0]) - margin <= x <= float(slot[1]) + margin
            and float(slot[2]) - margin <= y <= float(slot[3]) + margin
        )

    def _approach_candidates(
        self,
        slot: List[float],
        target_yaw: float,
    ) -> List[Tuple[float, float, float]]:
        cx, cy = self._slot_center(slot)
        if self._is_rear_parking_mode():
            return self._rear_approach_candidates((cx, cy, target_yaw), slot)
        return self._front_approach_candidates((cx, cy, target_yaw), slot)

    def _front_approach_candidates(
        self,
        target_pose: Tuple[float, float, float],
        slot: Optional[List[float]] = None,
    ) -> List[Tuple[float, float, float]]:
        target_yaw = target_pose[2]
        candidates: List[Tuple[float, float, float]] = []
        for side in (-1.0, 1.0):
            p1, _, _ = self._rear_y_formula_points(target_pose, lateral_side=side, slot=slot)
            candidates.append((p1[0], p1[1], target_yaw))
        return candidates

    def _reverse_start_pose(
        self,
        target_pose: Tuple[float, float, float],
        distance: float = REAR_APPROACH_DISTANCE,
    ) -> Tuple[float, float, float]:
        tx, ty, target_yaw = target_pose
        open_sign = self._rear_open_side_sign(target_pose)
        start_x = tx
        start_y = ty + open_sign * distance
        start_x, start_y = self._clamp_inside_map(
            start_x,
            start_y,
            margin=0.2,
            vehicle_margin=PARKING_VEHICLE_RECT_MARGIN,
        )
        return start_x, start_y, target_yaw

    def _rear_approach_candidates(
        self,
        target_pose: Tuple[float, float, float],
        slot: Optional[List[float]] = None,
    ) -> List[Tuple[float, float, float]]:
        target_yaw = target_pose[2]
        candidates: List[Tuple[float, float, float]] = []
        for side in (-1.0, 1.0):
            p1, _, _ = self._rear_y_formula_points(target_pose, lateral_side=side, slot=slot)
            candidates.append((p1[0], p1[1], target_yaw))
        return candidates

    def _rear_open_side_sign(
        self,
        target_pose: Tuple[float, float, float],
        slot: Optional[List[float]] = None,
    ) -> float:
        tx, ty, _ = target_pose
        slot_open_sign = self._slot_open_side_from_lines(slot)
        if slot_open_sign is not None:
            return slot_open_sign
        open_signs = self._open_vertical_approach_signs(tx, ty)
        return self._best_open_side_by_clearance(tx, ty, open_signs)

    def _best_open_side_by_clearance(
        self,
        tx: float,
        ty: float,
        open_signs: List[float],
    ) -> float:
        signs = open_signs if open_signs else [1.0, -1.0]
        scored = [
            (self._open_side_clearance_score(tx, ty, sign), sign)
            for sign in signs
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def _open_side_clearance_score(self, tx: float, ty: float, sign: float) -> float:
        if self.map_extent is None:
            return 0.0
        _, _, ymin, ymax = self.map_extent
        mid_y = 0.5 * (ymin + ymax)
        yaw = math.pi / 2.0 if sign > 0.0 else -math.pi / 2.0
        p3 = (tx, ty + sign * REAR_APPROACH_DISTANCE)
        base_y = p3[1] + sign * REAR_Y_TRIANGLE_HEIGHT
        raw_points = [
            p3,
            (tx - REAR_Y_TRIANGLE_HALF_WIDTH, base_y - sign * REAR_Y_POINT1_VERTICAL_PULL),
            (tx + REAR_Y_TRIANGLE_HALF_WIDTH, base_y - sign * REAR_Y_POINT1_VERTICAL_PULL),
        ]

        score = 0.0
        if self._vertical_approach_line_hits_wall(tx, ty, sign):
            score -= 40.0
        if ty < mid_y and sign > 0.0:
            score += 4.0
        elif ty > mid_y and sign < 0.0:
            score += 4.0

        for raw_x, raw_y in raw_points:
            clamped_x, clamped_y = self._clamp_inside_map(
                raw_x,
                raw_y,
                margin=0.2,
                vehicle_margin=PARKING_VEHICLE_RECT_MARGIN,
            )
            clamp_error = math.hypot(clamped_x - raw_x, clamped_y - raw_y)
            if clamp_error > 1e-6:
                score -= 30.0 + 15.0 * clamp_error
            if not self._inside_map(
                clamped_x,
                clamped_y,
                margin=0.2,
                vehicle_margin=PARKING_VEHICLE_RECT_MARGIN,
            ):
                score -= 25.0
            clearance = self._estimate_pose_clearance(
                clamped_x,
                clamped_y,
                yaw,
                include_lines=True,
                vehicle_margin=PARKING_VEHICLE_RECT_MARGIN,
            )
            if clearance <= 0.0:
                score -= 50.0
            else:
                score += min(clearance, 4.0)
        return score

    def _slot_open_side_from_lines(self, slot: Optional[List[float]]) -> Optional[float]:
        if slot is None:
            slot = self._current_target_slot()
        if slot is None or not self.map_data:
            return None
        x0, x1, y0, y1 = (float(v) for v in slot)
        bottom_score = self._slot_horizontal_edge_line_score(x0, x1, y0)
        top_score = self._slot_horizontal_edge_line_score(x0, x1, y1)
        min_edge_score = 0.35 * max(0.1, x1 - x0)
        bottom_closed = bottom_score >= min_edge_score
        top_closed = top_score >= min_edge_score
        if bottom_closed and not top_closed:
            return 1.0
        if top_closed and not bottom_closed:
            return -1.0
        return None

    def _slot_horizontal_edge_line_score(self, x0: float, x1: float, edge_y: float) -> float:
        if not self.map_data:
            return 0.0
        score = 0.0
        tolerance = 0.35
        for line in self.map_data.get("lines") or []:
            if len(line) != 4:
                continue
            lx0, ly0, lx1, ly1 = (float(v) for v in line)
            if abs(ly0 - ly1) > 1e-6:
                continue
            if abs(ly0 - edge_y) > tolerance:
                continue
            overlap = min(max(lx0, lx1), x1) - max(min(lx0, lx1), x0)
            if overlap > 0.0:
                score += overlap
        return score

    def _current_target_slot(self) -> Optional[List[float]]:
        if self.target_signature is None or len(self.target_signature) != 4:
            return None
        return [float(v) for v in self.target_signature]

    def _rear_y_formula_points(
        self,
        target_pose: Tuple[float, float, float],
        lateral_side: float = -1.0,
        slot: Optional[List[float]] = None,
    ) -> List[Tuple[float, float]]:
        tx, ty, _ = target_pose
        open_sign = self._rear_open_side_sign(target_pose, slot=slot)
        p3 = (tx, ty + open_sign * REAR_APPROACH_DISTANCE)
        base_y = p3[1] + open_sign * REAR_Y_TRIANGLE_HEIGHT
        p1 = (
            tx + lateral_side * REAR_Y_TRIANGLE_HALF_WIDTH,
            base_y
            - open_sign * REAR_Y_POINT1_VERTICAL_PULL
            + open_sign * FIRST_APPROACH_EXTRA_Y_DISTANCE,
        )
        p2 = (
            tx - lateral_side * (REAR_Y_TRIANGLE_HALF_WIDTH + REAR_Y_POINT2_EXTRA_WIDTH),
            base_y,
        )
        return [
            self._clamp_formula_point(p1),
            self._clamp_formula_point(p2),
            self._clamp_formula_point(p3),
        ]

    def _clamp_formula_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        return self._clamp_inside_map(
            point[0],
            point[1],
            margin=0.2,
            vehicle_margin=PARKING_VEHICLE_RECT_MARGIN,
        )

    def _rear_y_parking_points(
        self,
        approach_pose: Tuple[float, float, float],
        target_pose: Tuple[float, float, float],
        p3: Optional[Tuple[float, float, float]] = None,
        side: Optional[float] = None,
        lateral_width: Optional[float] = None,
    ) -> List[Tuple[float, float]]:
        tx, ty, target_yaw = target_pose
        if side is None:
            ax, ay, _ = approach_pose
            side = 1.0 if ax >= tx else -1.0
        points = self._rear_y_formula_points(target_pose, lateral_side=side)
        if p3 is not None:
            points[-1] = (p3[0], p3[1])
        return points

    def _sampled_path_is_clear(
        self,
        points: List[Tuple[float, float]],
        vehicle_margin: float,
        step_size: float = 0.45,
    ) -> bool:
        if len(points) < 2:
            return False
        rects = self._collision_rects(include_lines=True)
        for idx in range(len(points) - 1):
            x0, y0 = points[idx]
            x1, y1 = points[idx + 1]
            seg_len = math.hypot(x1 - x0, y1 - y0)
            if seg_len < 1e-6:
                continue
            yaw = math.atan2(y1 - y0, x1 - x0)
            steps = max(1, int(math.ceil(seg_len / step_size)))
            for step in range(steps + 1):
                ratio = step / steps
                px = x0 + (x1 - x0) * ratio
                py = y0 + (y1 - y0) * ratio
                if self._pose_collides_with_rects(
                    px,
                    py,
                    yaw,
                    rects,
                    vehicle_margin=vehicle_margin,
                ):
                    return False
        return True

    def _fast_approach_path(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
    ) -> List[Tuple[float, float]]:
        sx, sy = start_xy
        gx, gy = goal_xy
        candidates = [
            [start_xy, goal_xy],
            [start_xy, (sx, gy), goal_xy],
            [start_xy, (gx, sy), goal_xy],
        ]
        for points in candidates:
            if self._sampled_path_is_clear(
                points,
                vehicle_margin=VEHICLE_RECT_MARGIN,
                step_size=2.0,
            ):
                return points
        return []

    def _safe_approach_candidates(
        self,
        desired: Tuple[Tuple[float, float, float], ...] | List[Tuple[float, float, float]],
        yaw: float,
        vehicle_margin: float,
        max_candidates: int,
    ) -> List[Tuple[float, float, float]]:
        candidates: List[Tuple[float, float, float]] = []
        seen = set()
        offsets = (
            (0.0, 0.0),
            (0.6, 0.0),
            (-0.6, 0.0),
            (0.0, 0.6),
            (0.0, -0.6),
            (1.2, 0.0),
            (-1.2, 0.0),
            (0.0, 1.2),
            (0.0, -1.2),
            (1.2, 1.2),
            (1.2, -1.2),
            (-1.2, 1.2),
            (-1.2, -1.2),
            (2.0, 0.0),
            (-2.0, 0.0),
            (0.0, 2.0),
            (0.0, -2.0),
        )
        for base_x, base_y, _ in desired:
            for ox, oy in offsets:
                px, py = self._clamp_inside_map(
                    base_x + ox,
                    base_y + oy,
                    margin=0.2,
                    vehicle_margin=vehicle_margin,
                )
                key = (round(px, 2), round(py, 2))
                if key in seen:
                    continue
                seen.add(key)
                if self._pose_is_collision_free(
                    px,
                    py,
                    yaw,
                    include_lines=True,
                    vehicle_margin=vehicle_margin,
                ):
                    candidates.append((px, py, yaw))
                    if len(candidates) >= max_candidates:
                        return candidates
        return candidates

    def _open_vertical_approach_signs(self, x: float, y: float) -> List[float]:
        signs: List[float] = []
        for sign in (1.0, -1.0):
            if not self._vertical_approach_line_hits_wall(x, y, sign):
                signs.append(sign)
        if signs:
            return signs
        return [1.0, -1.0]

    def _vertical_approach_line_hits_wall(self, x: float, y: float, sign: float) -> bool:
        if self.map_extent is None or not self.map_data:
            return False
        _, _, ymin, ymax = self.map_extent
        end_y = ymax if sign > 0.0 else ymin
        x0 = x
        x1 = x
        y0 = min(y, end_y)
        y1 = max(y, end_y)
        wall_margin = VEHICLE_HALF_WIDTH + VEHICLE_RECT_MARGIN
        for rect in self.map_data.get("walls_rects") or []:
            rx0, rx1, ry0, ry1 = (float(v) for v in rect)
            rx0 -= wall_margin
            rx1 += wall_margin
            if rx0 <= x0 <= rx1 and not (y1 < ry0 or ry1 < y0):
                return True
        return False

    def _select_best_plan(
        self,
        start_xy: Tuple[float, float],
        candidates: List[Tuple[float, float, float]],
        target_pose: Tuple[float, float, float],
        start_yaw: float,
    ) -> Optional[Tuple[Tuple[float, float, float], List[Tuple[float, float]], float]]:
        best: Optional[Tuple[Tuple[float, float, float], List[Tuple[float, float]], float]] = None
        started_at = time.perf_counter()
        for candidate in candidates:
            goal_xy = (candidate[0], candidate[1])
            grid_path = self._astar_path(
                start_xy,
                goal_xy,
                start_yaw=start_yaw,
                goal_yaw=None,
            )
            if not grid_path:
                continue
            path_len = self._path_length(grid_path)
            clearance = self._estimate_min_obstacle_distance(
                (candidate[0], candidate[1]),
                yaw=candidate[2],
            )
            final_leg = math.hypot(target_pose[0] - candidate[0], target_pose[1] - candidate[1])
            clearance_penalty = 8.0 / max(clearance, 0.20)
            lateral_error = abs(
                (candidate[0] - target_pose[0]) * math.sin(target_pose[2])
                - (candidate[1] - target_pose[1]) * math.cos(target_pose[2])
            )
            cost = (
                path_len
                + 0.65 * final_leg
                + 2.5 * lateral_error
                + clearance_penalty
            )
            if best is None or cost < best[2]:
                best = (candidate, grid_path, cost)
            if best is not None and time.perf_counter() - started_at > 0.08:
                break
        return best

    def _prepend_start_alignment(
        self,
        path: List[Tuple[float, float]],
        start: Tuple[float, float, float],
    ) -> List[Tuple[float, float]]:
        if len(path) < 2:
            return path
        sx, sy, syaw = start
        goal_along_start_yaw = (
            (path[-1][0] - sx) * math.cos(syaw)
            + (path[-1][1] - sy) * math.sin(syaw)
        )
        if goal_along_start_yaw <= START_ALIGNMENT_DISTANCE + 0.35:
            return path
        forward_point = (
            sx + math.cos(syaw) * START_ALIGNMENT_DISTANCE,
            sy + math.sin(syaw) * START_ALIGNMENT_DISTANCE,
        )
        if not self._pose_is_collision_free(
            forward_point[0],
            forward_point[1],
            syaw,
            include_lines=True,
        ):
            return path

        corrected = [(sx, sy), forward_point]
        for point in path[1:]:
            along_start_yaw = (
                (point[0] - sx) * math.cos(syaw)
                + (point[1] - sy) * math.sin(syaw)
            )
            if along_start_yaw <= START_ALIGNMENT_DISTANCE + 0.35:
                continue
            if math.hypot(point[0] - forward_point[0], point[1] - forward_point[1]) < 0.75:
                continue
            corrected.append(point)
        return corrected if len(corrected) >= 2 else path

    def _ensure_parking_segment(
        self,
        x: float,
        y: float,
        target_pose: Tuple[float, float, float],
    ) -> bool:
        if self.parking_segment_ready or not self.waypoints:
            return self.parking_segment_ready
        approach_wp = self.waypoints[-1]
        approach_remaining = math.hypot(approach_wp[0] - x, approach_wp[1] - y)
        target_remaining = math.hypot(target_pose[0] - x, target_pose[1] - y)
        near_approach = approach_remaining <= PARKING_SEGMENT_TRIGGER_DISTANCE
        near_target = target_remaining <= PARKING_ALIGN_DISTANCE
        if not near_approach and not near_target:
            return False

        if self._is_rear_parking_mode():
            if self.parking_state != PARKING_STATE_SECOND_APPROACH:
                selected = self._rear_second_approach_segment(x, y, target_pose)
                if selected is None:
                    return False
                points, segment_cost = selected
                self.waypoints = self._points_to_waypoints(
                    points,
                    final_yaw=target_pose[2],
                    gear="D",
                )
                self.waypoint_index = 0
                self.parking_state = PARKING_STATE_SECOND_APPROACH
                full_points = self._rear_y_parking_points(
                    (points[0][0], points[0][1], target_pose[2]),
                    target_pose,
                )
                full_points[0] = points[0]
                self.debug_parking_points = self._rear_second_debug_points(full_points)
                self.debug_approach_point = points[-1]
                self.debug_astar_path = list(full_points)
                print(
                    "[algo] rear second approach ready:"
                    f" points={[(round(p[0], 2), round(p[1], 2)) for p in points]}"
                    f" reverse_start=({full_points[-1][0]:.2f}, {full_points[-1][1]:.2f})"
                    f" cost={segment_cost:.2f}"
                )
                return True
            selected = self._rear_reverse_parking_segment(x, y, target_pose)
        else:
            selected = self._front_t_parking_segment(x, y, target_pose)
        if selected is None:
            # Parking entry candidates would collide or leave the map; keep
            # following the A* approach path and retry from a better pose.
            return False
        points, segment_cost = selected

        if self._is_rear_parking_mode():
            self.waypoints = self._points_to_waypoints(
                points,
                final_yaw=target_pose[2],
                gear="R",
            )
            self.parking_state = PARKING_STATE_REVERSE_PARKING
            self.rear_last_steer = 0.0
        else:
            self.waypoints = self._points_to_waypoints(
                points,
                final_yaw=target_pose[2],
                gear="D",
                target_pose=target_pose,
                parking_start_index=0,
            )
            self.parking_state = PARKING_STATE_ALIGN_CHECK
        self.waypoint_index = 0
        self.parking_segment_ready = True
        print(
            "[algo] parking segment ready:"
            f" mode={self.parking_mode}"
            f" maneuver={self.parking_maneuver}"
            f" waypoints={len(self.waypoints)}"
            f" cost={segment_cost:.2f}"
            f" target=({target_pose[0]:.2f}, {target_pose[1]:.2f}, "
            f"yaw={math.degrees(target_pose[2]):.1f}deg)"
        )
        return True

    def _rear_second_debug_points(
        self,
        points: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        if len(points) >= 4:
            return [(points[1][0], points[1][1]), (points[-2][0], points[-2][1]), (points[-1][0], points[-1][1])]
        return [(point[0], point[1]) for point in points]

    def _rear_second_approach_segment(
        self,
        x: float,
        y: float,
        target_pose: Tuple[float, float, float],
    ) -> Optional[Tuple[List[Tuple[float, float]], float]]:
        target_yaw = target_pose[2]
        p1 = self.rear_second_p1 if self.rear_second_p1 is not None else (x, y)
        sequence = self._rear_y_parking_points((p1[0], p1[1], target_yaw), target_pose)
        if len(sequence) != 3:
            return None
        sequence[0] = p1
        # The vehicle may stop slightly past the precomputed 2-1 point.
        # Track the second leg from the actual stopped pose, not from stale p1.
        points: List[Tuple[float, float]] = [(x, y), sequence[1]]
        if not self._sampled_path_is_clear(
            points,
            vehicle_margin=PARKING_VEHICLE_RECT_MARGIN,
        ):
            return None
        return points, self._path_length(points)

    def _rear_reverse_parking_segment(
        self,
        x: float,
        y: float,
        target_pose: Tuple[float, float, float],
    ) -> Tuple[List[Tuple[float, float]], float]:
        tx, ty, target_yaw = target_pose
        start_x, start_y, _ = self._reverse_start_pose(target_pose)
        entry = (
            tx + math.cos(target_yaw) * REAR_ENTRY_DISTANCE,
            ty + math.sin(target_yaw) * REAR_ENTRY_DISTANCE,
        )
        points = [(x, y)]
        if math.hypot(start_x - x, start_y - y) > 0.45:
            points.append((start_x, start_y))
        if math.hypot(entry[0] - points[-1][0], entry[1] - points[-1][1]) > 0.25:
            points.append(entry)
        points.append((tx, ty))
        return points, self._path_length(points)

    def _front_t_parking_segment(
        self,
        x: float,
        y: float,
        target_pose: Tuple[float, float, float],
    ) -> Tuple[List[Tuple[float, float]], float]:
        tx, ty, _ = target_pose
        sequence = self._rear_y_parking_points((x, y, target_pose[2]), target_pose)
        point3 = sequence[2]
        points: List[Tuple[float, float]] = [(x, y)]
        if math.hypot(point3[0] - x, point3[1] - y) > 0.45:
            points.append(point3)
        points.append((tx, ty))
        return points, self._path_length(points)

    def _blend_approach_prealign_steer(
        self,
        steer: float,
        x: float,
        y: float,
        yaw: float,
        approach_remaining: float,
        target_pose: Tuple[float, float, float],
        wheelbase: float,
        max_steer: float,
    ) -> float:
        if (
            approach_remaining > APPROACH_PREALIGN_DISTANCE
            or self.debug_approach_point is None
        ):
            return steer
        target = self._approach_prealign_target(target_pose)
        if target is None:
            return steer
        prealign_steer = self._pure_pursuit_steer(
            x=x,
            y=y,
            yaw=yaw,
            target_x=target[0],
            target_y=target[1],
            wheelbase=wheelbase,
            max_steer=max_steer,
            reverse=False,
        )
        target_yaw = math.atan2(target[1] - y, target[0] - x)
        target_yaw += math.radians(APPROACH_PREALIGN_YAW_OFFSET_DEG)
        yaw_error = self._wrap_to_pi(target_yaw - yaw)
        yaw_steer = max(-max_steer, min(max_steer, 0.75 * yaw_error))
        yaw_blend = max(0.0, min(1.0, APPROACH_PREALIGN_YAW_BLEND))
        prealign_steer = (1.0 - yaw_blend) * prealign_steer + yaw_blend * yaw_steer
        progress = 1.0 - max(0.0, approach_remaining) / max(
            APPROACH_PREALIGN_DISTANCE,
            1e-6,
        )
        blend = max(0.0, min(APPROACH_PREALIGN_MAX_BLEND, progress * APPROACH_PREALIGN_MAX_BLEND))
        mixed = (1.0 - blend) * steer + blend * prealign_steer
        return max(-max_steer, min(max_steer, mixed))

    def _approach_prealign_target(
        self,
        target_pose: Tuple[float, float, float],
    ) -> Optional[Tuple[float, float]]:
        if self.debug_approach_point is None:
            return None
        ax, ay = self.debug_approach_point
        tx, _, target_yaw = target_pose
        side = 1.0 if ax >= tx else -1.0
        sequence = self._rear_y_parking_points(
            (ax, ay, target_yaw),
            target_pose,
            side=side,
        )
        if len(sequence) != 3:
            return None
        if self._is_rear_parking_mode():
            return sequence[1]
        return sequence[2]

    def _astar_path(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        start_yaw: Optional[float] = None,
        goal_yaw: Optional[float] = None,
        goal_yaw_tolerance: float = math.pi / 4.0,
    ) -> List[Tuple[float, float]]:
        if self.map_extent is None:
            return []
        xmin, xmax, ymin, ymax = self.map_extent
        grid_step = max(self.cell_size, A_STAR_GRID_STEP)
        cols = max(1, int(math.ceil((xmax - xmin) / grid_step)))
        rows = max(1, int(math.ceil((ymax - ymin) / grid_step)))
        blocked = self._cached_blocked_grid(rows, cols, grid_step)
        line_penalties = self._cached_line_penalty_grid(rows, cols, grid_step)

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
        self._clear_cell(blocked, start, radius=0, grid_step=grid_step)
        self._clear_cell(blocked, goal, radius=0, grid_step=grid_step)

        heading_moves = [
            (0, 1, 0.0),       # east
            (-1, 1, math.pi / 4.0),
            (-1, 0, math.pi / 2.0),
            (-1, -1, 3.0 * math.pi / 4.0),
            (0, -1, math.pi),
            (1, -1, -3.0 * math.pi / 4.0),
            (1, 0, -math.pi / 2.0),
            (1, 1, -math.pi / 4.0),
        ]
        pose_collisions = (
            self._cached_pose_collision_grid(rows, cols, grid_step)
            if A_STAR_USE_POSE_GRID
            else None
        )
        start_heading = self._heading_index(start_yaw if start_yaw is not None else 0.0)
        goal_heading = self._heading_index(goal_yaw) if goal_yaw is not None else None
        start_state = (start[0], start[1], start_heading)

        open_heap: List[Tuple[float, float, Tuple[int, int, int]]] = []
        heapq.heappush(open_heap, (0.0, 0.0, start_state))
        came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        cost_so_far: Dict[Tuple[int, int, int], float] = {start_state: 0.0}
        goal_state: Optional[Tuple[int, int, int]] = None
        expansions = 0
        max_expansions = max(2500, rows * cols * 4)

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            expansions += 1
            if (current[0], current[1]) == goal and self._goal_heading_ok(
                current[2],
                goal_heading,
                tolerance=goal_yaw_tolerance,
            ):
                goal_state = current
                break
            if expansions > max_expansions:
                break
            for turn in range(-A_STAR_MAX_HEADING_STEP, A_STAR_MAX_HEADING_STEP + 1):
                next_heading = (current[2] + turn) % len(heading_moves)
                dr, dc, _ = heading_moves[next_heading]
                nxt = (current[0] + dr, current[1] + dc, next_heading)
                if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols):
                    continue
                if blocked[nxt[0]][nxt[1]]:
                    continue
                if pose_collisions is not None and pose_collisions[next_heading][nxt[0]][nxt[1]]:
                    continue
                move_cost = math.hypot(dr, dc)
                turn_penalty = 0.55 * abs(turn)
                heading_penalty = 0.0
                if goal_heading is not None:
                    heading_penalty = 0.05 * self._heading_index_distance(next_heading, goal_heading)
                clearance_penalty = line_penalties[nxt[0]][nxt[1]]
                new_cost = (
                    cost_so_far[current]
                    + move_cost
                    + turn_penalty
                    + heading_penalty
                    + clearance_penalty
                )
                if new_cost >= cost_so_far.get(nxt, float("inf")):
                    continue
                cost_so_far[nxt] = new_cost
                heuristic = math.hypot(goal[0] - nxt[0], goal[1] - nxt[1])
                heapq.heappush(
                    open_heap,
                    (new_cost + A_STAR_HEURISTIC_WEIGHT * heuristic, new_cost, nxt),
                )
                came_from[nxt] = current

        if goal_state is None:
            return []

        states = [goal_state]
        while states[-1] != start_state:
            states.append(came_from[states[-1]])
        states.reverse()
        path = [start_xy]
        path.extend(to_world((state[0], state[1])) for state in states[1:-1])
        path.append(goal_xy)
        return path

    def _heading_index(self, yaw: float) -> int:
        return int(round(self._wrap_to_pi(yaw) / (math.pi / 4.0))) % 8

    def _heading_index_distance(self, a: int, b: int) -> int:
        diff = abs(a - b) % 8
        return min(diff, 8 - diff)

    def _goal_heading_ok(
        self,
        heading: int,
        goal_heading: Optional[int],
        tolerance: float = math.pi / 4.0,
    ) -> bool:
        if goal_heading is None:
            return True
        heading_error = self._heading_index_distance(heading, goal_heading) * (math.pi / 4.0)
        return heading_error <= tolerance + 1e-9

    def _warm_planning_caches(self, warm_pose_grid: bool = True) -> None:
        if self.map_extent is None:
            return
        xmin, xmax, ymin, ymax = self.map_extent
        grid_step = max(self.cell_size, A_STAR_GRID_STEP)
        cols = max(1, int(math.ceil((xmax - xmin) / grid_step)))
        rows = max(1, int(math.ceil((ymax - ymin) / grid_step)))
        self._cached_blocked_grid(rows, cols, grid_step)
        self._cached_line_penalty_grid(rows, cols, grid_step)
        if warm_pose_grid:
            self._cached_pose_collision_grid(rows, cols, grid_step)

    def _cached_blocked_grid(self, rows: int, cols: int, grid_step: float) -> List[List[bool]]:
        if (
            self.blocked_grid_cache is None
            or self.blocked_grid_cache[0] != rows
            or self.blocked_grid_cache[1] != cols
            or abs(self.blocked_grid_cache[2] - grid_step) > 1e-9
        ):
            base = self._build_blocked_grid(rows, cols, grid_step)
            self.blocked_grid_cache = (rows, cols, grid_step, base)
        return [row[:] for row in self.blocked_grid_cache[3]]

    def _cached_line_penalty_grid(self, rows: int, cols: int, grid_step: float) -> List[List[float]]:
        if (
            self.line_penalty_grid_cache is None
            or self.line_penalty_grid_cache[0] != rows
            or self.line_penalty_grid_cache[1] != cols
            or abs(self.line_penalty_grid_cache[2] - grid_step) > 1e-9
        ):
            penalties = self._build_line_penalty_grid(rows, cols, grid_step)
            self.line_penalty_grid_cache = (rows, cols, grid_step, penalties)
        return self.line_penalty_grid_cache[3]

    def _cached_pose_collision_grid(
        self,
        rows: int,
        cols: int,
        grid_step: float,
    ) -> List[List[List[bool]]]:
        if (
            self.pose_collision_grid_cache is None
            or self.pose_collision_grid_cache[0] != rows
            or self.pose_collision_grid_cache[1] != cols
            or abs(self.pose_collision_grid_cache[2] - grid_step) > 1e-9
        ):
            grids = self._build_pose_collision_grid(rows, cols, grid_step)
            self.pose_collision_grid_cache = (rows, cols, grid_step, grids)
        return self.pose_collision_grid_cache[3]

    def _build_pose_collision_grid(
        self,
        rows: int,
        cols: int,
        grid_step: float,
    ) -> List[List[List[bool]]]:
        headings = [
            0.0,
            math.pi / 4.0,
            math.pi / 2.0,
            3.0 * math.pi / 4.0,
            math.pi,
            -3.0 * math.pi / 4.0,
            -math.pi / 2.0,
            -math.pi / 4.0,
        ]
        grids = [[[False for _ in range(cols)] for _ in range(rows)] for _ in headings]
        if self.map_extent is None:
            return grids
        xmin, _, _, ymax = self.map_extent
        rects = self._collision_rects(include_lines=True)
        for heading_idx, heading in enumerate(headings):
            grid = grids[heading_idx]
            for row in range(rows):
                y = ymax - (row + 0.5) * grid_step
                for col in range(cols):
                    x = xmin + (col + 0.5) * grid_step
                    grid[row][col] = self._pose_collides_with_rects(x, y, heading, rects)
        return grids

    def _build_line_penalty_grid(self, rows: int, cols: int, grid_step: float) -> List[List[float]]:
        penalties = [[0.0 for _ in range(cols)] for _ in range(rows)]
        if self.map_extent is None:
            return penalties
        line_rects = self._line_obstacle_rects(half_width=LINE_COLLISION_HALF_WIDTH)
        if not line_rects:
            return penalties
        margin = PLANNING_OBSTACLE_MARGIN + LINE_EXTRA_CLEARANCE
        for rect in line_rects:
            self._mark_penalty_rect(penalties, rect, grid_step, margin=margin, penalty=2.0)
        return penalties

    def _mark_penalty_rect(
        self,
        penalties: List[List[float]],
        rect: Tuple[float, float, float, float],
        grid_step: float,
        margin: float,
        penalty: float,
    ) -> None:
        if self.map_extent is None:
            return
        xmin, _, _, ymax = self.map_extent
        rows = len(penalties)
        cols = len(penalties[0]) if rows else 0
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
                penalties[row][col] = max(penalties[row][col], penalty)

    def _build_blocked_grid(self, rows: int, cols: int, grid_step: float) -> List[List[bool]]:
        blocked = [[False for _ in range(cols)] for _ in range(rows)]
        if self.map_extent is None:
            return blocked
        xmin, _, _, ymax = self.map_extent

        for rect in self._obstacle_rects():
            self._mark_rect(blocked, rect, grid_step, margin=PLANNING_OBSTACLE_MARGIN)
        for rect in self._line_obstacle_rects(half_width=LINE_COLLISION_HALF_WIDTH):
            self._mark_rect(blocked, rect, grid_step, margin=LINE_HARD_MARGIN)
        # Lines are simulator collision objects, so the initial point-grid
        # planner also treats them as hard obstacles.
        return self._inflate_blocked(blocked, radius_cells=0)

    def _obstacle_rects(self) -> List[Tuple[float, float, float, float]]:
        rects: List[Tuple[float, float, float, float]] = []
        if not self.map_data:
            return rects
        slots = self.map_data.get("slots") or []
        occupied = self.map_data.get("occupied_idx") or []
        for idx, slot in enumerate(slots):
            if idx < len(occupied) and bool(occupied[idx]):
                rects.append(tuple(float(v) for v in slot))
        for rect in self.map_data.get("walls_rects") or []:
            rects.append(tuple(float(v) for v in rect))
        return rects

    def _line_obstacle_rects(self, half_width: float) -> List[Tuple[float, float, float, float]]:
        rects: List[Tuple[float, float, float, float]] = []
        if not self.map_data or self.map_extent is None:
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

    def _clear_cell(
        self,
        blocked: List[List[bool]],
        cell: Tuple[int, int],
        radius: int,
        grid_step: float,
    ) -> None:
        rows = len(blocked)
        cols = len(blocked[0]) if rows else 0
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = cell[0] + dr
                cc = cell[1] + dc
                if (
                    0 <= rr < rows
                    and 0 <= cc < cols
                    and not self._cell_hits_protected_obstacle(rr, cc, grid_step)
                ):
                    blocked[rr][cc] = False

    def _cell_hits_protected_obstacle(self, row: int, col: int, grid_step: float) -> bool:
        if self.map_extent is None or not self.map_data:
            return False
        xmin, _, _, ymax = self.map_extent
        x = xmin + (col + 0.5) * grid_step
        y = ymax - (row + 0.5) * grid_step
        for rect in self.map_data.get("walls_rects") or []:
            rx0, rx1, ry0, ry1 = (float(v) for v in rect)
            if rx0 <= x <= rx1 and ry0 <= y <= ry1:
                return True
        line_margin = LINE_HARD_MARGIN
        for rx0, rx1, ry0, ry1 in self._line_obstacle_rects(half_width=LINE_COLLISION_HALF_WIDTH):
            rx0 -= line_margin
            rx1 += line_margin
            ry0 -= line_margin
            ry1 += line_margin
            if rx0 <= x <= rx1 and ry0 <= y <= ry1:
                return True
        return False

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
        target_pose: Optional[Tuple[float, float, float]] = None,
        parking_start_index: Optional[int] = None,
    ) -> List[Waypoint]:
        waypoints: List[Waypoint] = []
        for idx, point in enumerate(points):
            segment_gear = gear
            if idx < len(points) - 1:
                nxt = points[idx + 1]
                yaw = math.atan2(nxt[1] - point[1], nxt[0] - point[0])
                if target_pose is not None and (
                    parking_start_index is None or idx >= parking_start_index
                ):
                    tx, ty, target_yaw = target_pose
                    near_target = (
                        math.hypot(point[0] - tx, point[1] - ty) <= PARKING_ALIGN_DISTANCE + 0.5
                        and math.hypot(nxt[0] - tx, nxt[1] - ty) <= PARKING_ALIGN_DISTANCE + 0.5
                    )
                    if near_target:
                        travel_dot = math.cos(self._wrap_to_pi(yaw - target_yaw))
                        yaw = target_yaw
                        if self._is_rear_parking_mode():
                            segment_gear = "D" if travel_dot >= 0.0 else "R"
                        else:
                            segment_gear = "D"
            else:
                yaw = final_yaw
                if waypoints:
                    segment_gear = waypoints[-1][3]
            waypoints.append((point[0], point[1], yaw, segment_gear))
        return waypoints

    def _advance_waypoint_index(self, x: float, y: float) -> None:
        while self.waypoint_index < len(self.waypoints) - 1:
            wp = self.waypoints[self.waypoint_index]
            if math.hypot(wp[0] - x, wp[1] - y) > 0.8:
                break
            self.waypoint_index += 1

    def _advance_approach_waypoint_index(self, x: float, y: float) -> None:
        nearest_idx = self._nearest_future_path_index(x, y)
        if nearest_idx > self.waypoint_index:
            self.waypoint_index = nearest_idx

    def _lookahead_index(self, x: float, y: float, lookahead: float) -> int:
        idx = self.waypoint_index
        while idx < len(self.waypoints) - 1:
            wp = self.waypoints[idx]
            if math.hypot(wp[0] - x, wp[1] - y) >= lookahead:
                return idx
            idx += 1
        return len(self.waypoints) - 1

    def _approach_lookahead_index(self, x: float, y: float, lookahead: float) -> int:
        reference_idx = self._nearest_future_path_index(x, y)
        reference_idx = max(self.waypoint_index, reference_idx)
        distance_accum = 0.0
        idx = reference_idx
        while idx < len(self.waypoints) - 1:
            current = self.waypoints[idx]
            nxt = self.waypoints[idx + 1]
            distance_accum += math.hypot(nxt[0] - current[0], nxt[1] - current[1])
            idx += 1
            if distance_accum >= lookahead:
                return idx
        return len(self.waypoints) - 1

    def _nearest_future_path_index(self, x: float, y: float) -> int:
        if not self.waypoints:
            return 0
        start_idx = max(0, min(self.waypoint_index, len(self.waypoints) - 1))
        end_idx = min(len(self.waypoints) - 1, start_idx + APPROACH_LOOKAHEAD_WINDOW)
        if start_idx >= end_idx:
            return start_idx
        best_idx = start_idx
        best_dist = float("inf")
        for idx in range(start_idx, end_idx):
            ax, ay = self.waypoints[idx][0], self.waypoints[idx][1]
            bx, by = self.waypoints[idx + 1][0], self.waypoints[idx + 1][1]
            seg_dist, seg_t = self._approach_point_to_segment_distance(x, y, ax, ay, bx, by)
            if seg_dist < best_dist:
                best_dist = seg_dist
                best_idx = idx + 1 if seg_t > 0.65 else idx
        return max(start_idx, min(best_idx, len(self.waypoints) - 1))

    def _approach_path_deviation(self, x: float, y: float) -> float:
        if not self.waypoints:
            return 0.0
        start_idx = max(0, min(self.waypoint_index, len(self.waypoints) - 1))
        end_idx = min(len(self.waypoints) - 1, start_idx + APPROACH_LOOKAHEAD_WINDOW)
        if start_idx >= end_idx:
            wp = self.waypoints[start_idx]
            return math.hypot(wp[0] - x, wp[1] - y)
        best_dist = float("inf")
        for idx in range(start_idx, end_idx):
            ax, ay = self.waypoints[idx][0], self.waypoints[idx][1]
            bx, by = self.waypoints[idx + 1][0], self.waypoints[idx + 1][1]
            seg_dist, _ = self._approach_point_to_segment_distance(x, y, ax, ay, bx, by)
            best_dist = min(best_dist, seg_dist)
        return best_dist

    def _approach_point_to_segment_distance(
        self,
        px: float,
        py: float,
        ax: float,
        ay: float,
        bx: float,
        by: float,
    ) -> Tuple[float, float]:
        dx = bx - ax
        dy = by - ay
        denom = dx * dx + dy * dy
        if denom <= 1e-9:
            return math.hypot(px - ax, py - ay), 0.0
        t = ((px - ax) * dx + (py - ay) * dy) / denom
        t = max(0.0, min(1.0, t))
        closest_x = ax + t * dx
        closest_y = ay + t * dy
        return math.hypot(px - closest_x, py - closest_y), t

    def _approach_path_curvature(self, center_idx: int) -> float:
        if len(self.waypoints) < 3:
            return 0.0
        start_idx = max(1, min(center_idx, len(self.waypoints) - 2))
        end_idx = min(len(self.waypoints) - 2, start_idx + 3)
        max_turn = 0.0
        for idx in range(start_idx, end_idx + 1):
            prev = self.waypoints[idx - 1]
            curr = self.waypoints[idx]
            nxt = self.waypoints[idx + 1]
            yaw_in = math.atan2(curr[1] - prev[1], curr[0] - prev[0])
            yaw_out = math.atan2(nxt[1] - curr[1], nxt[0] - curr[0])
            max_turn = max(max_turn, abs(self._wrap_to_pi(yaw_out - yaw_in)))
        return max_turn

    def _segment_gear_for_target(self, target_idx: int) -> str:
        if not self.waypoints:
            return "D"
        gear_idx = target_idx
        if target_idx > self.waypoint_index:
            gear_idx = max(self.waypoint_index, target_idx - 1)
        gear_idx = max(0, min(gear_idx, len(self.waypoints) - 1))
        return self.waypoints[gear_idx][3]

    def _adaptive_lookahead(self, speed: float, final_dist: float, yaw_error: float) -> float:
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

    def _tracking_yaw_error(
        self,
        x: float,
        y: float,
        yaw: float,
        target_wp: Waypoint,
        reverse: bool,
    ) -> float:
        target_heading = math.atan2(target_wp[1] - y, target_wp[0] - x)
        tracking_yaw = self._wrap_to_pi(yaw + math.pi) if reverse else yaw
        return abs(self._wrap_to_pi(target_heading - tracking_yaw))

    def _parking_entry_target(
        self,
        x: float,
        y: float,
        planned_target: Waypoint,
        target_center: Tuple[float, float],
        final_yaw: float,
    ) -> Waypoint:
        final_dist = math.hypot(target_center[0] - x, target_center[1] - y)
        if final_dist < 1.0:
            return target_center[0], target_center[1], final_yaw, planned_target[3]
        return planned_target[0], planned_target[1], final_yaw, planned_target[3]

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
            target = min(target, 0.90)
        if final_dist < 2.2:
            target = 0.38
        if final_dist < 1.0:
            target = PARKING_MIN_ROLL_SPEED
        if yaw_error > math.radians(35.0) or abs(steer) > math.radians(25.0):
            turn_cap = 2.40 if not in_parking_mode else 0.60
            target = min(target, turn_cap)
        if front_clearance < OBSTACLE_SLOW_DISTANCE:
            target = min(target, 0.60 if not in_parking_mode else 0.55)
        if front_clearance < OBSTACLE_STOP_DISTANCE:
            target = 0.0
        return target

    def _approach_remaining(self, x: float, y: float) -> float:
        if self.debug_approach_point is None:
            return float("inf")
        return math.hypot(
            self.debug_approach_point[0] - x,
            self.debug_approach_point[1] - y,
        )

    def _approach_target_passed(self, x: float, y: float) -> bool:
        if self.debug_approach_point is None or len(self.waypoints) < 2:
            return False
        target_x, target_y = self.debug_approach_point
        prev_wp = self.waypoints[-2]
        vx = target_x - prev_wp[0]
        vy = target_y - prev_wp[1]
        seg_len_sq = vx * vx + vy * vy
        if seg_len_sq < 1e-6:
            return False
        wx = x - target_x
        wy = y - target_y
        passed_along = wx * vx + wy * vy
        lateral_error = abs(wx * vy - wy * vx) / math.sqrt(seg_len_sq)
        return passed_along > 0.0 and lateral_error <= 2.0

    def _approach_speed(
        self,
        base_speed: float,
        approach_remaining: float,
        steer_abs: float,
        front_clearance: float,
    ) -> float:
        if front_clearance < OBSTACLE_SLOW_DISTANCE:
            return base_speed
        if steer_abs < math.radians(10.0):
            target = min(APPROACH_CRUISE_SPEED, 2.60 + 0.45 * approach_remaining)
        elif steer_abs < math.radians(22.0):
            target = min(3.80, 1.80 + 0.30 * approach_remaining)
        else:
            target = min(2.20, 1.20 + 0.18 * approach_remaining)
        return max(base_speed, target)

    def _inside_map(
        self,
        x: float,
        y: float,
        margin: float = 0.4,
        vehicle_margin: float = VEHICLE_RECT_MARGIN,
    ) -> bool:
        if self.map_extent is None:
            return True
        xmin, xmax, ymin, ymax = self.map_extent
        margin = max(margin, VEHICLE_HALF_WIDTH + vehicle_margin + EXTRA_SAFETY_MARGIN)
        return xmin + margin <= x <= xmax - margin and ymin + margin <= y <= ymax - margin

    def _clamp_inside_map(
        self,
        x: float,
        y: float,
        margin: float = 0.4,
        vehicle_margin: float = VEHICLE_RECT_MARGIN,
    ) -> Tuple[float, float]:
        if self.map_extent is None:
            return x, y
        xmin, xmax, ymin, ymax = self.map_extent
        margin = max(margin, VEHICLE_HALF_WIDTH + vehicle_margin + EXTRA_SAFETY_MARGIN)
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

    def _line_margin_penalty(self, point: Tuple[float, float]) -> float:
        clearance = self._estimate_clearance(point, include_lines=True)
        if clearance >= LINE_EXTRA_CLEARANCE:
            return 0.0
        shortage = LINE_EXTRA_CLEARANCE - clearance
        return 4.0 * shortage / max(LINE_EXTRA_CLEARANCE, 1e-6)

    def _speed_command(
        self,
        speed: float,
        target_speed: float,
        front_is_clear: bool = False,
        force_full_accel: bool = False,
        gentle_brake: bool = False,
        accel_deadband: float = 0.15,
    ) -> Tuple[float, float]:
        error = target_speed - speed
        if target_speed <= 0.05:
            return 0.0, 1.0
        if force_full_accel and error > 0.05:
            return 1.0, 0.0
        if error > accel_deadband:
            accel_cap = 1.0 if front_is_clear else 0.9
            accel_base = 0.70 if front_is_clear else 0.55
            accel_gain = 0.70 if front_is_clear else 0.55
            return min(accel_cap, accel_base + accel_gain * error), 0.0
        if error < -0.08:
            if gentle_brake:
                return 0.0, min(0.28, 0.08 + 0.18 * (-error))
            return 0.0, min(0.8, 0.25 + 0.35 * (-error))
        return 0.0, 0.0

    def _estimate_forward_clearance(
        self,
        x: float,
        y: float,
        yaw: float,
        reverse: bool = False,
        steer: float = 0.0,
        wheelbase: float = 2.6,
        max_distance: float = FRONT_CLEAR_DISTANCE,
        vehicle_margin: float = VEHICLE_RECT_MARGIN,
    ) -> float:
        step = 0.35
        distance = 0.0
        direction = -1.0 if reverse else 1.0
        px, py, pyaw = x, y, yaw
        curvature = math.tan(steer) / max(wheelbase, 1e-6)

        while distance < max_distance:
            pyaw = self._wrap_to_pi(pyaw + direction * curvature * step)
            px += direction * math.cos(pyaw) * step
            py += direction * math.sin(pyaw) * step
            distance += step
            if self._estimate_pose_clearance(
                px,
                py,
                pyaw,
                include_lines=True,
                vehicle_margin=vehicle_margin,
            ) <= 0.05:
                return distance
        return max_distance

    def _estimate_min_obstacle_distance(
        self,
        point: Tuple[float, float],
        yaw: Optional[float] = None,
        vehicle_margin: float = VEHICLE_RECT_MARGIN,
    ) -> float:
        if yaw is not None:
            return self._estimate_pose_clearance(
                point[0],
                point[1],
                yaw,
                include_lines=False,
                vehicle_margin=vehicle_margin,
            )
        return self._estimate_clearance(point, include_lines=False, vehicle_margin=vehicle_margin)

    def _estimate_clearance(
        self,
        point: Tuple[float, float],
        include_lines: bool = False,
        vehicle_margin: float = VEHICLE_RECT_MARGIN,
    ) -> float:
        px, py = point
        best = float("inf")
        if self.map_extent is not None:
            xmin, xmax, ymin, ymax = self.map_extent
            best = min(best, px - xmin, xmax - px, py - ymin, ymax - py)
        rects = self._obstacle_rects()
        if include_lines:
            rects = rects + self._line_obstacle_rects(half_width=LINE_COLLISION_HALF_WIDTH)
        for rx0, rx1, ry0, ry1 in rects:
            dx = max(rx0 - px, 0.0, px - rx1)
            dy = max(ry0 - py, 0.0, py - ry1)
            best = min(best, math.hypot(dx, dy))
        return max(0.0, best - VEHICLE_HALF_WIDTH - vehicle_margin - EXTRA_SAFETY_MARGIN)

    def _estimate_pose_clearance(
        self,
        x: float,
        y: float,
        yaw: float,
        include_lines: bool = False,
        vehicle_margin: float = VEHICLE_RECT_MARGIN,
    ) -> float:
        vehicle_poly = self._vehicle_polygon(x, y, yaw, margin=vehicle_margin)
        if self._polygon_outside_map(vehicle_poly):
            return 0.0
        rects = self._obstacle_rects()
        if include_lines:
            rects = rects + self._line_obstacle_rects(half_width=LINE_COLLISION_HALF_WIDTH)
        best = self._polygon_map_clearance(vehicle_poly)
        for rect in rects:
            rect_poly = self._rect_polygon(rect)
            if self._polygons_intersect(vehicle_poly, rect_poly):
                return 0.0
            best = min(best, self._polygon_distance(vehicle_poly, rect_poly))
        return max(0.0, best - EXTRA_SAFETY_MARGIN)

    def _pose_is_collision_free(
        self,
        x: float,
        y: float,
        yaw: float,
        include_lines: bool,
        vehicle_margin: float = VEHICLE_RECT_MARGIN,
    ) -> bool:
        return self._estimate_pose_clearance(
            x,
            y,
            yaw,
            include_lines=include_lines,
            vehicle_margin=vehicle_margin,
        ) > 0.0

    def _collision_rects(self, include_lines: bool) -> List[Tuple[float, float, float, float]]:
        rects = self._obstacle_rects()
        if include_lines:
            rects = rects + self._line_obstacle_rects(half_width=LINE_COLLISION_HALF_WIDTH)
        return rects

    def _pose_collides_with_rects(
        self,
        x: float,
        y: float,
        yaw: float,
        rects: List[Tuple[float, float, float, float]],
        vehicle_margin: float = VEHICLE_RECT_MARGIN,
    ) -> bool:
        vehicle_poly = self._vehicle_polygon(x, y, yaw, margin=vehicle_margin)
        if self._polygon_outside_map(vehicle_poly):
            return True
        vx0 = min(point[0] for point in vehicle_poly)
        vx1 = max(point[0] for point in vehicle_poly)
        vy0 = min(point[1] for point in vehicle_poly)
        vy1 = max(point[1] for point in vehicle_poly)
        for rect in rects:
            rx0, rx1, ry0, ry1 = rect
            if vx1 < rx0 or rx1 < vx0 or vy1 < ry0 or ry1 < vy0:
                continue
            if self._polygons_intersect(vehicle_poly, self._rect_polygon(rect)):
                return True
        return False

    def _vehicle_polygon(
        self,
        x: float,
        y: float,
        yaw: float,
        margin: float = 0.0,
    ) -> List[Tuple[float, float]]:
        front = VEHICLE_FRONT_LENGTH + margin
        rear = VEHICLE_REAR_LENGTH + margin
        half_width = VEHICLE_HALF_WIDTH + margin
        local_points = [
            (front, half_width),
            (front, -half_width),
            (-rear, -half_width),
            (-rear, half_width),
        ]
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        return [
            (
                x + lx * cos_yaw - ly * sin_yaw,
                y + lx * sin_yaw + ly * cos_yaw,
            )
            for lx, ly in local_points
        ]

    def _rect_polygon(self, rect: Tuple[float, float, float, float]) -> List[Tuple[float, float]]:
        rx0, rx1, ry0, ry1 = rect
        return [(rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1)]

    def _polygon_outside_map(self, poly: List[Tuple[float, float]]) -> bool:
        if self.map_extent is None:
            return False
        xmin, xmax, ymin, ymax = self.map_extent
        return any(x < xmin or x > xmax or y < ymin or y > ymax for x, y in poly)

    def _polygon_map_clearance(self, poly: List[Tuple[float, float]]) -> float:
        if self.map_extent is None:
            return float("inf")
        xmin, xmax, ymin, ymax = self.map_extent
        return min(min(x - xmin, xmax - x, y - ymin, ymax - y) for x, y in poly)

    def _polygons_intersect(
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
        values = [x * axis[0] + y * axis[1] for x, y in poly]
        return min(values), max(values)

    def _polygon_distance(
        self,
        poly_a: List[Tuple[float, float]],
        poly_b: List[Tuple[float, float]],
    ) -> float:
        best = float("inf")
        for point in poly_a:
            best = min(best, self._point_to_polygon_distance(point, poly_b))
        for point in poly_b:
            best = min(best, self._point_to_polygon_distance(point, poly_a))
        return best

    def _point_to_polygon_distance(
        self,
        point: Tuple[float, float],
        poly: List[Tuple[float, float]],
    ) -> float:
        return min(
            self._point_to_segment_distance(point, poly[idx], poly[(idx + 1) % len(poly)])
            for idx in range(len(poly))
        )

    def _point_to_segment_distance(
        self,
        point: Tuple[float, float],
        start: Tuple[float, float],
        end: Tuple[float, float],
    ) -> float:
        px, py = point
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-12:
            return math.hypot(px - x1, py - y1)
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.hypot(px - proj_x, py - proj_y)

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
