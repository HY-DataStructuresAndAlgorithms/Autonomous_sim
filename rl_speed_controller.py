"""Small optional RL extension for parking speed control.

The planner and Pure Pursuit controller remain rule based. This module only
adjusts the rule-based target speed with a tiny tabular policy. It is designed
to be easy to explain in a short presentation:

state = (distance bin, yaw-error bin, steering bin, obstacle-clearance bin)
action = speed multiplier

The table is initialized with a safe policy. Optional Q-learning hooks are kept
for future work, but online training is disabled by default so the baseline
stays deterministic and runnable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Tuple


StateKey = Tuple[str, str, str, str]


@dataclass
class RLSpeedController:
    """Tabular RL policy that gently scales the rule-based target speed."""

    enabled: bool = True
    training: bool = False
    epsilon: float = 0.0
    alpha: float = 0.15
    gamma: float = 0.90
    actions: Tuple[float, ...] = (0.75, 0.90, 1.00, 1.08)
    q_table: Dict[StateKey, Dict[float, float]] = field(default_factory=dict)
    last_state: StateKey | None = None
    last_action: float | None = None

    def __post_init__(self) -> None:
        self._seed_safe_policy()

    def adjust_target_speed(
        self,
        rule_speed: float,
        final_dist: float,
        yaw_error: float,
        steer_abs: float,
        obstacle_dist: float,
    ) -> float:
        """Return a safe RL-adjusted target speed."""

        if not self.enabled:
            return rule_speed

        state = self._state_key(final_dist, yaw_error, steer_abs, obstacle_dist)
        action = self._select_action(state)
        self.last_state = state
        self.last_action = action

        adjusted = rule_speed * action

        # Safety shield: RL may slow down freely, but cannot speed up in risky
        # near-target, high-yaw, high-curvature, or close-obstacle states.
        if (
            final_dist < 2.5
            or yaw_error > 0.55
            or steer_abs > 0.45
            or obstacle_dist < 1.5
        ):
            adjusted = min(adjusted, rule_speed)

        return max(0.15, min(1.35, adjusted))

    def update_from_reward(self, reward: float, next_state: StateKey | None = None) -> None:
        """Optional Q-learning update for later experiments.

        Current baseline does not call this with simulator rewards. It is here
        so future work can tune the speed policy without changing planner APIs.
        """

        if not self.training or self.last_state is None or self.last_action is None:
            return
        table = self.q_table.setdefault(
            self.last_state,
            {action: 0.0 for action in self.actions},
        )
        current = table[self.last_action]
        future = 0.0
        if next_state is not None:
            future_table = self.q_table.setdefault(
                next_state,
                {action: 0.0 for action in self.actions},
            )
            future = max(future_table.values())
        table[self.last_action] = current + self.alpha * (
            reward + self.gamma * future - current
        )

    def _select_action(self, state: StateKey) -> float:
        table = self.q_table.setdefault(state, {action: 0.0 for action in self.actions})
        if self.training and random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(table, key=table.get)

    def _state_key(
        self,
        final_dist: float,
        yaw_error: float,
        steer_abs: float,
        obstacle_dist: float,
    ) -> StateKey:
        dist_bin = "near" if final_dist < 2.5 else "mid" if final_dist < 7.0 else "far"
        yaw_bin = "yaw_high" if yaw_error > 0.55 else "yaw_low"
        steer_bin = "turn_high" if steer_abs > 0.45 else "turn_low"
        obs_bin = "obs_close" if obstacle_dist < 1.5 else "obs_clear"
        return dist_bin, yaw_bin, steer_bin, obs_bin

    def _seed_safe_policy(self) -> None:
        for dist_bin in ("near", "mid", "far"):
            for yaw_bin in ("yaw_low", "yaw_high"):
                for steer_bin in ("turn_low", "turn_high"):
                    for obs_bin in ("obs_clear", "obs_close"):
                        state = (dist_bin, yaw_bin, steer_bin, obs_bin)
                        values = {action: 0.0 for action in self.actions}
                        best = 1.00
                        if dist_bin == "far" and yaw_bin == "yaw_low" and steer_bin == "turn_low":
                            best = 1.08
                        if dist_bin == "near":
                            best = 0.75
                        if yaw_bin == "yaw_high" or steer_bin == "turn_high":
                            best = min(best, 0.90)
                        if obs_bin == "obs_close":
                            best = 0.75
                        values[best] = 1.0
                        self.q_table[state] = values
