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
- 시뮬레이션 경로 흔적이 오래 지나면 사라짐 -> `self-parking-sim/demo_self_parking_sim.py`에서 주행 궤적을 계속 남기도록 수정
- 차량의 영역으로 인해 벽 넘어로 주차 구역에 닿으려고 시도 -> 최종 성공 기준을 주차구역 정중앙의 10% 영역에 차량 중앙이 닿아야 성공으로 변경 [해결 실패] -> 첫 목표를 3m앞 지점으로 계획후 접근하는 방식 채택[거의 성공]
- 주차 구역 접근까지 좋았으나 주차에 성공하지 못함 -> 주차 로직 필요
- 주차로직 생성했으나 주차 구역에 일부만 들어와도 정지하고 종료되버리는 상황 발생 -> 시뮬레이션의 종료 기준으로 주차 모드에 들어가면 주차 구역에서의 방향과 중심위치가 맞을때까지 조향과 위치 보정을 하면서 성공 조건에 들어올때까지 정지 하지 않게 수정
- 타임아웃 문제 속도가 너무 느림 -> 주차 구역과 멀더라도 yaw error가 크면 속도를 낮춰버려서 멀리 있어도 속도가 낮아짐 이 제한을 주차 구간에서만 강하게 적용 -> 목표와의 거리가 멀수록 엑셀을 더 밟게 수정 -> 직선 구간 accel 100% 사용 기능 추가
- 절대 후진을 하지 않음 (점수 일부만 받음) -> 목표지점과 가까워졌지만 거리가 있고 yaw가 차이가 많이 날경우 재정렬 후 진입
- 후진 시에도 핸들 각도가 그대로라서 왔다갔다만 반복함 -> 후진시에는 핸들 각도 0도로 고정 후진 지속 시간 증가
- 장애물과 라인에 너무 가까운 경로로 생성해 가끔 벽에 부딪힘 -> 경로 생성할때 최대한 꺽임수를 줄이는 경로 채택[폐기] -> line 마진을 생성해 경로 생성당시 line에 닿지 않는 경로 생성 -> 마진 생성시 주차 라인 진입 불가 A* 패널티로 만들어 라인과 가까운 경로를 싫어하게 생성
- 주차 후진문제