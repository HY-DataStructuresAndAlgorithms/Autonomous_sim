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

## 팀원 설정 방법
1. `self-parking-sim` 폴더에서 가상환경을 만들고 패키지를 설치한다.

```powershell
cd J:\work\Self-Parking\self-parking-sim
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. 첫 번째 PowerShell에서 시뮬레이터를 실행한다.

```powershell
cd J:\work\Self-Parking\self-parking-sim
.\.venv\Scripts\Activate.ps1
python demo_self_parking_sim.py
```

3. 두 번째 PowerShell에서 학생 알고리즘을 실행한다.

```powershell
cd J:\work\Self-Parking\self-parking-user-algorithms
..\self-parking-sim\.venv\Scripts\Activate.ps1
python my_agent.py --host 127.0.0.1 --port 55556
```

4. RL speed correction을 끄고 baseline만 비교하려면 아래처럼 실행한다.

```powershell
$env:PARKING_USE_RL_SPEED="0"
python my_agent.py --host 127.0.0.1 --port 55556
```
