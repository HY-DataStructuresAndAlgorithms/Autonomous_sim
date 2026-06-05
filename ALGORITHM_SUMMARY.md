# Autonomous Parking Algorithm Summary

## Current Baseline

- A* based path planning
- Multiple candidate approach poses around the target parking slot
- Planning attempt for each approach candidate
- Best path selection using a cost score
- Final alignment waypoints based on target yaw instead of directly connecting to the slot center
- Pure Pursuit controller
- Adaptive lookahead distance
- Rule-based speed control
- Optional RL speed correction

## Design Principle

The professor simulator code is preserved. The student algorithm keeps the
existing IPC interface and returns the simulator command format:

```json
{
  "steer": 0.05,
  "accel": 0.2,
  "brake": 0.0,
  "gear": "D"
}
```

The planner and controller remain rule based for reliability. Reinforcement
learning is only used as an optional speed correction module, so the baseline
can still run without RL.
