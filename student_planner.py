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
VEHICLE_RECT_MARGIN = 0.05
PLANNING_OBSTACLE_MARGIN = VEHICLE_RECT_MARGIN
EXTRA_SAFETY_MARGIN = 0.0
OBSTACLE_SLOW_DISTANCE = 1.15
OBSTACLE_STOP_DISTANCE = 0.45
FRONT_CLEAR_DISTANCE = 6.0
FRONT_CLEAR_SPEED_BONUS = 0.75
PARKING_ALIGN_DISTANCE = 4.0
FINAL_ALIGNMENT_DISTANCES = (3.4, 2.4, 1.45, 0.75, 0.35, 0.0)
FINAL_ALIGN_CLEARANCE_MIN = 0.10
PARKING_SEGMENT_TRIGGER_DISTANCE = 1.25
PARKING_REVERSE_YAW_ERROR = math.radians(32.0)
PARKING_REVERSE_TICKS = 58
PARKING_REVERSE_COOLDOWN_TICKS = 45
PARKING_REVERSE_WALL_CLEARANCE = 0.85
PARKING_REVERSE_REARM_CLEARANCE = 0.65
PARKING_REVERSE_REARM_DISTANCE = 0.90
PARKING_TARGET_OVERSHOOT = 0.30
LINE_EXTRA_CLEARANCE = 0.20
LINE_COLLISION_HALF_WIDTH = 0.25
LINE_HARD_MARGIN = 0.05
A_STAR_GRID_STEP = 2.00
A_STAR_MAX_HEADING_STEP = 1  # 8-heading grid: one step means at most 45 degrees.


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
    parking_reverse_ticks: int = 0
    parking_reverse_cooldown: int = 0
    parking_has_reversed: bool = False
    parking_reverse_completed: bool = False
    parking_last_reverse_end: Optional[Tuple[float, float]] = None
    parking_segment_ready: bool = False

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
        self.parking_reverse_ticks = 0
        self.parking_reverse_cooldown = 0
        self.parking_has_reversed = False
        self.parking_reverse_completed = False
        self.parking_last_reverse_end = None
        self.parking_segment_ready = False
        self.rl_speed = RLSpeedController(enabled=USE_RL_SPEED_CONTROL)
        self._warm_planning_caches(warm_pose_grid=False)
        print(f"[algo] rl_speed_control={'ON' if USE_RL_SPEED_CONTROL else 'OFF'}")

    def compute_path(self, obs: Dict[str, Any]) -> None:
        """Plan a path from the current pose to the target parking slot."""

        self.waypoints.clear()
        self.waypoint_index = 0
        self.parking_reverse_ticks = 0
        self.parking_reverse_cooldown = 0
        self.parking_has_reversed = False
        self.parking_reverse_completed = False
        self.parking_last_reverse_end = None
        self.parking_segment_ready = False
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
            print(
                "[algo] planning success:"
                f" candidates={len(candidates)}"
                f" selected=({approach_pose[0]:.2f}, {approach_pose[1]:.2f})"
                f" a_star_points={len(grid_path)} cost={cost:.2f}"
            )

        simplified = self._simplify_path(grid_path, spacing=1.0)
        points = simplified
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
        wheelbase = float(limits.get("L", 2.6))
        max_steer = float(limits.get("maxSteer", math.radians(35.0)))

        slot = obs.get("target_slot") or []
        signature = tuple(round(float(v), 3) for v in slot) if len(slot) == 4 else None
        if not self.waypoints or signature != self.target_signature:
            self.compute_path(obs)

        if not self.waypoints:
            return {"steer": 0.0, "accel": 0.0, "brake": 0.8, "gear": "D"}

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

        self._advance_waypoint_index(x, y)
        if len(slot) == 4:
            self._ensure_parking_segment(
                x=x,
                y=y,
                target_pose=target_pose,
            )
            final_wp = self.waypoints[-1]
        in_parking_mode = final_dist < PARKING_ALIGN_DISTANCE
        tracking_reference = self.waypoints[min(self.waypoint_index + 1, len(self.waypoints) - 1)]
        tracking_yaw_error = self._tracking_yaw_error(
            x=x,
            y=y,
            yaw=yaw,
            target_wp=tracking_reference,
            reverse=False,
        )
        control_yaw_error = final_yaw_error if in_parking_mode else tracking_yaw_error
        lookahead = self._adaptive_lookahead(speed, final_dist, control_yaw_error)
        target_idx = self._lookahead_index(x, y, lookahead=lookahead)
        target_wp = self.waypoints[target_idx]
        gear = target_wp[3]
        reverse_realigning = False
        forward_clearance = self._estimate_forward_clearance(
            x=x,
            y=y,
            yaw=yaw,
            reverse=False,
            steer=0.0,
            wheelbase=wheelbase,
        )
        target_overshot = self._passed_target_center(
            x=x,
            y=y,
            target_center=target_center,
            target_yaw=final_yaw,
            tolerance=max(center_tolerance, PARKING_TARGET_OVERSHOOT),
        )
        if self.parking_reverse_cooldown > 0:
            self.parking_reverse_cooldown -= 1
        should_reverse_for_alignment = (
            not self.parking_has_reversed
            and target_wp[3] != "R"
            and final_yaw_error > PARKING_REVERSE_YAW_ERROR
        )
        forward_after_reverse = 0.0
        if self.parking_last_reverse_end is not None:
            forward_after_reverse = math.hypot(
                x - self.parking_last_reverse_end[0],
                y - self.parking_last_reverse_end[1],
            )
        should_reverse_for_obstacle = (
            not self.parking_has_reversed
            and forward_clearance < PARKING_REVERSE_WALL_CLEARANCE
        )
        should_reverse_after_forward = (
            self.parking_reverse_completed
            and forward_after_reverse >= PARKING_REVERSE_REARM_DISTANCE
            and forward_clearance < PARKING_REVERSE_REARM_CLEARANCE
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
                else "front_obstacle" if should_reverse_for_obstacle else "forward_until_collision_risk"
            )
            print(
                "[algo] parking recovery: reverse realignment"
                f" pos_error={final_dist:.2f}m"
                f" yaw_error={math.degrees(final_yaw_error):.1f}deg"
                f" reason={reverse_reason}"
                f" forward_clearance={forward_clearance:.2f}m"
                f" forward_after_reverse={forward_after_reverse:.2f}m"
                f" target_overshot={target_overshot}"
            )
        if in_parking_mode and self.parking_reverse_ticks > 0:
            self.parking_reverse_ticks -= 1
            if self.parking_reverse_ticks <= 0:
                self.parking_reverse_cooldown = PARKING_REVERSE_COOLDOWN_TICKS
                self.parking_reverse_completed = True
                self.parking_last_reverse_end = (x, y)
            target_wp = self._parking_reverse_target(target_center, final_yaw)
            gear = "R"
            reverse_realigning = True
        elif in_parking_mode:
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
        if reverse_realigning:
            steer = 0.0

        front_clearance = self._estimate_forward_clearance(
            x=x,
            y=y,
            yaw=yaw,
            reverse=(gear == "R"),
            steer=steer,
            wheelbase=wheelbase,
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
            control_yaw_error,
            steer,
            front_clearance,
        )
        if gear == "R":
            rule_speed = min(rule_speed, 0.55)
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
                f" tracking_yaw_error={math.degrees(tracking_yaw_error):.1f}deg"
                f" final_yaw_error={math.degrees(final_yaw_error):.1f}deg"
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

    def _target_pose(self, slot: List[float]) -> Tuple[float, float, float]:
        cx, cy = self._slot_center(slot)
        expected = str((self.map_data or {}).get("expected_orientation") or "")
        yaw = -math.pi / 2.0 if expected.lower().startswith("rear") else math.pi / 2.0
        return cx, cy, yaw

    def _slot_center(self, slot: List[float]) -> Tuple[float, float]:
        return (
            0.5 * (float(slot[0]) + float(slot[1])),
            0.5 * (float(slot[2]) + float(slot[3])),
        )

    def _slot_center_tolerance(self, slot: List[float]) -> float:
        slot_w = abs(float(slot[1]) - float(slot[0]))
        slot_l = abs(float(slot[3]) - float(slot[2]))
        return max(0.15, 0.05 * min(slot_w, slot_l))

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

    def _approach_pose(self, slot: List[float], target_yaw: float) -> Tuple[float, float, float]:
        cx, cy = self._slot_center(slot)
        above = (cx, cy + 3.0, target_yaw)
        if self._is_valid_approach_point(above[0], above[1], target_yaw):
            return above
        return cx, cy - 3.0, target_yaw

    def _approach_candidates(
        self,
        slot: List[float],
        target_yaw: float,
    ) -> List[Tuple[float, float, float]]:
        cx, cy = self._slot_center(slot)
        above = (cx, cy + 3.0, target_yaw)
        below = (cx, cy - 3.0, target_yaw)
        candidates: List[Tuple[float, float, float]] = []
        if self._is_valid_approach_point(above[0], above[1], target_yaw):
            candidates.append(above)
        elif self._is_valid_approach_point(below[0], below[1], target_yaw):
            candidates.append(below)
        if candidates:
            return candidates
        fallback_x, fallback_y = self._clamp_inside_map(below[0], below[1])
        return [(fallback_x, fallback_y, target_yaw)]

    def _is_valid_approach_point(self, x: float, y: float, yaw: float) -> bool:
        return self._pose_is_collision_free(x, y, yaw, include_lines=True)

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
            grid_path = self._astar_path(
                start_xy,
                (candidate[0], candidate[1]),
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
            yaw_align = abs(self._wrap_to_pi(candidate[2] - target_pose[2]))
            clearance_penalty = 8.0 / max(clearance, 0.20)
            lateral_error = abs(
                (candidate[0] - target_pose[0]) * math.sin(target_pose[2])
                - (candidate[1] - target_pose[1]) * math.cos(target_pose[2])
            )
            cost = (
                path_len
                + 0.65 * final_leg
                + 2.0 * yaw_align
                + 2.5 * lateral_error
                + clearance_penalty
            )
            if best is None or cost < best[2]:
                best = (candidate, grid_path, cost)
            if best is not None and time.perf_counter() - started_at > 0.08:
                break
        return best

    def _append_final_alignment(
        self,
        approach_path: List[Tuple[float, float]],
        target_pose: Tuple[float, float, float],
    ) -> List[Tuple[float, float]]:
        if not approach_path:
            return [(target_pose[0], target_pose[1])]
        points = list(approach_path)
        tx, ty, target_yaw = target_pose
        axis_x = math.cos(target_yaw)
        axis_y = math.sin(target_yaw)
        lateral_x = -axis_y
        lateral_y = axis_x
        last_x, last_y = points[-1]
        rel_x = last_x - tx
        rel_y = last_y - ty
        along = rel_x * axis_x + rel_y * axis_y
        lateral = rel_x * lateral_x + rel_y * lateral_y
        approach_side = 1.0 if along >= 0.0 else -1.0

        # First pull the final leg onto the slot centerline. This prevents the
        # parking controller from entering the slot diagonally after A* reaches
        # the 1st approach point.
        if abs(lateral) > 0.20:
            centerline_distance = max(2.8, min(abs(along), FINAL_ALIGNMENT_DISTANCES[0]))
            centerline_point = (
                tx + approach_side * axis_x * centerline_distance,
                ty + approach_side * axis_y * centerline_distance,
            )
            centerline_point = self._safe_alignment_point(centerline_point)
            if math.hypot(
                centerline_point[0] - points[-1][0],
                centerline_point[1] - points[-1][1],
            ) > 0.25:
                points.append(centerline_point)

        current_entry_distance = abs(
            (points[-1][0] - tx) * axis_x + (points[-1][1] - ty) * axis_y
        )
        for distance in FINAL_ALIGNMENT_DISTANCES:
            if distance == 0.0:
                point = (tx, ty)
                if math.hypot(point[0] - points[-1][0], point[1] - points[-1][1]) > 0.25:
                    points.append(point)
                continue
            if distance >= current_entry_distance - 0.20:
                continue
            point = (
                tx + approach_side * axis_x * distance,
                ty + approach_side * axis_y * distance,
            )
            point = self._safe_alignment_point(point)
            if math.hypot(point[0] - points[-1][0], point[1] - points[-1][1]) > 0.25:
                points.append(point)
                current_entry_distance = distance
        return points

    def _ensure_parking_segment(
        self,
        x: float,
        y: float,
        target_pose: Tuple[float, float, float],
    ) -> None:
        if self.parking_segment_ready or not self.waypoints:
            return
        approach_wp = self.waypoints[-1]
        approach_remaining = math.hypot(approach_wp[0] - x, approach_wp[1] - y)
        target_remaining = math.hypot(target_pose[0] - x, target_pose[1] - y)
        if (
            approach_remaining > PARKING_SEGMENT_TRIGGER_DISTANCE
            and target_remaining > PARKING_ALIGN_DISTANCE + 0.8
        ):
            return

        points = self._append_final_alignment([(x, y)], target_pose)
        self.waypoints = self._points_to_waypoints(
            points,
            final_yaw=target_pose[2],
            gear="D",
            target_pose=target_pose,
            parking_start_index=0,
        )
        self.waypoint_index = 0
        self.parking_segment_ready = True
        self.parking_reverse_ticks = 0
        self.parking_reverse_cooldown = 0
        self.parking_has_reversed = False
        self.parking_reverse_completed = False
        self.parking_last_reverse_end = None
        print(
            "[algo] parking segment ready:"
            f" waypoints={len(self.waypoints)}"
            f" target=({target_pose[0]:.2f}, {target_pose[1]:.2f}, "
            f"yaw={math.degrees(target_pose[2]):.1f}deg)"
        )

    def _safe_alignment_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        x, y = point
        if not self._inside_map(x, y, margin=0.2):
            x, y = self._clamp_inside_map(x, y, margin=0.2)
        if self._estimate_min_obstacle_distance((x, y)) >= FINAL_ALIGN_CLEARANCE_MIN:
            return x, y
        return point

    def _astar_path(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        start_yaw: Optional[float] = None,
        goal_yaw: Optional[float] = None,
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
        pose_collisions = self._cached_pose_collision_grid(rows, cols, grid_step)
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
            if (current[0], current[1]) == goal and self._goal_heading_ok(current[2], goal_heading):
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
                if pose_collisions[next_heading][nxt[0]][nxt[1]]:
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
                heapq.heappush(open_heap, (new_cost + heuristic, new_cost, nxt))
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

    def _goal_heading_ok(self, heading: int, goal_heading: Optional[int]) -> bool:
        if goal_heading is None:
            return True
        return self._heading_index_distance(heading, goal_heading) <= 1

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
        # Lines are handled as hard obstacles by the orientation-aware vehicle
        # footprint grid. Keeping them out of this point grid avoids blocking
        # entire parking aisles with coarse cells.
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
                        segment_gear = "D" if travel_dot >= 0.0 else "R"
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

    def _lookahead_index(self, x: float, y: float, lookahead: float) -> int:
        idx = self.waypoint_index
        while idx < len(self.waypoints) - 1:
            wp = self.waypoints[idx]
            if math.hypot(wp[0] - x, wp[1] - y) >= lookahead:
                return idx
            idx += 1
        return len(self.waypoints) - 1

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
        margin = max(margin, VEHICLE_HALF_WIDTH + VEHICLE_RECT_MARGIN + EXTRA_SAFETY_MARGIN)
        return xmin + margin <= x <= xmax - margin and ymin + margin <= y <= ymax - margin

    def _clamp_inside_map(self, x: float, y: float, margin: float = 0.4) -> Tuple[float, float]:
        if self.map_extent is None:
            return x, y
        xmin, xmax, ymin, ymax = self.map_extent
        margin = max(margin, VEHICLE_HALF_WIDTH + VEHICLE_RECT_MARGIN + EXTRA_SAFETY_MARGIN)
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
        steer: float = 0.0,
        wheelbase: float = 2.6,
        max_distance: float = FRONT_CLEAR_DISTANCE,
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
            if self._estimate_pose_clearance(px, py, pyaw, include_lines=True) <= 0.05:
                return distance
        return max_distance

    def _estimate_min_obstacle_distance(
        self,
        point: Tuple[float, float],
        yaw: Optional[float] = None,
    ) -> float:
        if yaw is not None:
            return self._estimate_pose_clearance(point[0], point[1], yaw, include_lines=False)
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
        if include_lines:
            rects = rects + self._line_obstacle_rects(half_width=LINE_COLLISION_HALF_WIDTH)
        for rx0, rx1, ry0, ry1 in rects:
            dx = max(rx0 - px, 0.0, px - rx1)
            dy = max(ry0 - py, 0.0, py - ry1)
            best = min(best, math.hypot(dx, dy))
        return max(0.0, best - VEHICLE_HALF_WIDTH - VEHICLE_RECT_MARGIN - EXTRA_SAFETY_MARGIN)

    def _estimate_pose_clearance(
        self,
        x: float,
        y: float,
        yaw: float,
        include_lines: bool = False,
    ) -> float:
        vehicle_poly = self._vehicle_polygon(x, y, yaw, margin=VEHICLE_RECT_MARGIN)
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
    ) -> bool:
        return self._estimate_pose_clearance(x, y, yaw, include_lines=include_lines) > 0.0

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
    ) -> bool:
        vehicle_poly = self._vehicle_polygon(x, y, yaw, margin=VEHICLE_RECT_MARGIN)
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
