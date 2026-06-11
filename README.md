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

## 문제 발견과 해결 과정
- 차량을 점 처럼 계획함 -> 차량을 최대 길이 + 20%의 마진을 지름삼아 원형으로 바운더리 설정
- 조향 가능 곡률을 A*가 고려하지 않음 -> 곡률 고려한 경로 설정
