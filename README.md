# 자율주차 학생 알고리즘

교수님이 제공한 self-parking simulator의 학생 알고리즘 구현 파일이다. 시뮬레이터 코드는 그대로 두고, 학생 알고리즘 쪽에서 경로 계획, 경로 추종, 주차 모드, 강화학습 기반 파라미터 튜닝을 구현했다.

## 실행 방법

시뮬레이터를 먼저 실행한 뒤, 다른 PowerShell에서 학생 알고리즘을 실행한다.

```powershell
cd J:\work\Self-Parking\self-parking-user-algorithms
..\self-parking-sim\.venv\Scripts\python.exe my_agent.py --host 127.0.0.1 --port 55556
```

시뮬레이터와 학생 알고리즘은 IPC로 통신한다. 학생 알고리즘은 매 tick마다 아래 형식의 명령을 반환한다.

```json
{
  "steer": 0.05,
  "accel": 0.2,
  "brake": 0.0,
  "gear": "D"
}
```

## 주요 파일

- `student_planner.py`: 실제 주차 알고리즘 구현
- `my_agent.py`: simulator와 IPC 연결을 담당하는 실행 파일
- `ipc_client.py`: IPC 통신 helper
- `rl_speed_controller.py`: 초기 실험용 RL 속도 보정 모듈
- `parking_tune.py`: 주차 후반부 파라미터 튜닝 스크립트
- `render_replay_path.py`: replay를 이미지로 렌더링하는 보조 도구

## 학생 알고리즘 구조

전체 구조는 rule-based baseline을 중심으로 구성했다. 강화학습은 baseline을 대체하지 않고, 일부 파라미터를 조정하는 보조 모듈로만 사용한다.

### 1. 입력 정보 사용

학생 알고리즘은 simulator에서 전달하는 정보를 사용한다.

- 차량 상태: `x`, `y`, `yaw`, `velocity`
- 목표 주차 슬롯: `(xmin, xmax, ymin, ymax)`
- 맵 정보: 벽, 주차 라인, 점유 슬롯, 빈 슬롯
- stage별 요구 orientation

차량은 점이 아니라 직사각형으로 보고 충돌과 clearance를 판단한다. 차량 중심만 기준으로 계획하면 벽이나 주차 라인에 너무 가까워지는 문제가 있었기 때문에, 차량 크기와 margin을 고려하도록 했다.

### 2. 경로 계획

경로 계획은 A* 기반으로 구현했다.

1. 목표 주차 슬롯 주변에 1차 approach point를 만든다.
2. 현재 차량 위치에서 approach point까지 A* 경로를 생성한다.
3. 경로 후보가 여러 개일 경우 cost가 낮은 경로를 선택한다.
4. A* 경로 생성 후 불필요한 중간점을 줄이고, 차량이 따라가기 쉬운 waypoint 형태로 후처리한다.
5. approach point에 도달하면 주차 모드로 전환한다.

A* cost는 단순 최단거리만 보지 않는다. 벽, 점유 슬롯, 주차 라인과의 충돌 가능성을 피하고, 너무 위험한 경로에는 높은 비용을 주도록 했다. 이 방식으로 경로가 짧더라도 벽이나 라인에 너무 붙는 경우를 줄였다.

### 3. 주차 모드

approach point에 도달하면 일반 주행 경로 추종을 멈추고 주차 모드로 전환한다. 주차 모드는 일반적인 T자형 주차 동작에서 아이디어를 얻어, Y자형 waypoint sequence로 구성했다.

- 2-1: 주차를 시작하기 위한 기준 위치
- 2-2: 슬롯 중심선에 맞추기 위한 보조 위치
- 2-3: 실제 슬롯 진입 직전 위치

전면 주차와 후면 주차 모두 같은 구조를 사용하되, 최종 진입 방향과 gear 사용만 다르게 처리한다. 후면 주차에서는 2-2 이후 R gear를 고정하고, waypoint마다 D/R을 다시 판단하지 않는다. 이렇게 해야 gear switching이 줄고, 후진 주차 동작이 더 안정적으로 유지된다.

### 4. 제어기

경로 추종에는 Pure Pursuit controller를 사용했다.

- lookahead point를 기준으로 steering angle 계산
- approach 구간에서는 상대적으로 높은 속도 허용
- 주차 구간에서는 낮은 속도와 steering rate limit 적용
- 목표 슬롯에 충분히 들어가면 감속 후 정지

초기에는 공격적인 가속, 후진 복구, 장애물 회피, 특수 케이스 보정 등을 많이 추가했다. 하지만 기능이 많아질수록 특정 상황에는 좋아지고 다른 상황에서는 실패하는 문제가 생겼다. 최종적으로는 예외 처리를 줄이고, 경로 생성과 주차 진입이라는 핵심 동작에 집중하도록 정리했다.

## 강화학습 및 파라미터 튜닝

강화학습은 전체 주차 정책을 새로 학습하는 방식으로 사용하지 않았다. A* planner와 Pure Pursuit controller는 그대로 유지하고, 주차 후반부의 일부 파라미터만 튜닝하는 방식으로 사용했다.

### 튜닝 대상

`parking_tune.py`는 2-1 위치 근처에서 episode를 시작해 주차 구간만 반복 실험한다. 시작 위치와 yaw에는 작은 random noise를 주어, 한 위치에만 과적합되지 않도록 했다.

현재 튜닝 대상은 다음과 같다.

- 주차 구간 최소 속도
- 2차 접근 구간 최소 속도
- 후진 주차 속도
- 후진 주차 최소 속도
- 주차 준비 단계 brake 강도
- 최종 정지 IoU 기준
- 최종 정지 거리 기준
- 슬롯 중심 근처 정지 기준

A* approach path 생성 파라미터는 강화학습에서 제외했다. 해당 값을 학습으로 바꾸면 일부 케이스에서는 좋아졌지만, 다른 케이스에서 A* 경로 생성이 실패하는 문제가 있었다. 따라서 1차 경로 계획은 rule-based로 고정하고, 학습은 주차 후반부 속도와 정지 기준에만 사용한다.

### 보상 기준

보상은 simulator score를 기본으로 사용하고, 주차 품질을 반영하는 항목을 추가했다.

- simulator score가 높을수록 보상 증가
- parking IoU가 높을수록 보상 증가
- 최종 orientation이 요구 방향과 맞으면 보상
- 주차 성공 시 추가 보상
- 충돌 시 큰 패널티
- timeout 시 패널티
- 최종 위치 오차가 클수록 패널티
- gear switch가 많을수록 패널티
- steering reversal이 많을수록 패널티

학습 결과가 항상 좋아지는 것은 아니었기 때문에, 안전장치를 두었다. 학습된 policy는 성공했고 score가 90점을 초과한 경우에만 학생 알고리즘에 적용된다. 그 이하의 결과는 저장되어 있어도 실제 주행에서는 무시된다.

### 실행 예시

```powershell
cd J:\work\Self-Parking\self-parking-user-algorithms
..\self-parking-sim\.venv\Scripts\python.exe parking_tune.py --episodes 200 --workers 4 --policy-group lower_front_in --timeout 25 --random-policy
```

policy group은 다음 네 가지로 나누었다.

- `lower_front_in`
- `lower_rear_in`
- `upper_front_in`
- `upper_rear_in`

아래 행과 중간 행은 비슷한 구조로 보고 `lower`로 묶고, 맨 위 행은 `upper`로 구분했다.

## 시뮬레이터 분석에서 확인한 점

구현 중 simulator의 판정 방식이 일반적인 주차 개념과 조금 다른 부분을 확인했다.

1. `front_in` / `rear_in`은 실제 진입 방향보다 최종 차량 yaw에 더 크게 의존한다.
2. Stage 1, 2는 `front_in`, Stage 3는 `rear_in`을 요구한다.
3. 주차 성공에는 IoU뿐 아니라 낮은 최종 속도가 필요하다.
4. Stage 3에서는 일부 슬롯 크기가 약간 다르다. 반복 학습 중 특정 케이스가 자주 실패해 원인을 추적하다가 확인했다.

이 때문에 단순히 “앞으로 들어가기” 또는 “뒤로 들어가기”만으로는 충분하지 않았다. 목표 슬롯의 열린 방향, stage가 요구하는 orientation, 최종 yaw 기준을 함께 고려해야 했다.

## `front_in` / `rear_in` 판단 방법

주차 방향은 단순히 stage 이름만 보고 정하지 않았다. 먼저 simulator가 전달하는 `expected_orientation` 또는 map payload 안의 orientation 관련 값을 읽어 `front_in` 또는 `rear_in` 요구를 확인한다. 값이 명확하지 않은 경우에는 기본값을 `front_in`으로 둔다.

그 다음 목표 주차 슬롯의 열린 방향을 계산한다. 슬롯의 아래쪽 경계선과 위쪽 경계선에 주차 라인이 얼마나 있는지 비교하여, 위쪽이 열려 있는지 아래쪽이 열려 있는지 판단한다. 아래쪽이 막혀 있고 위쪽이 열려 있으면 위쪽에서 접근 가능한 슬롯으로 보고, 반대의 경우에는 아래쪽에서 접근 가능한 슬롯으로 본다.

최종 maneuver는 이 두 정보를 함께 사용해 정한다.

1. simulator가 요구하는 최종 orientation을 확인한다.
2. 목표 슬롯의 열린 방향을 확인한다.
3. 해당 슬롯에 진입했을 때 최종 차량 yaw가 simulator의 `front_in` / `rear_in` 판정 기준과 맞는 maneuver를 선택한다.

즉 실제로 차량이 앞부터 들어가는지, 뒤부터 들어가는지만 보고 판단하지 않는다. simulator는 최종 차량 yaw를 기준으로 orientation을 판정하기 때문에, 알고리즘도 최종 yaw가 요구 방향과 맞도록 `front_in` 또는 `rear_in` maneuver를 선택한다.

## 현재 결과

최근 전체 테스트 기준 결과는 다음과 같다.

- 전체 테스트: 71개 target case
- 성공: 69 / 71
- 평균 점수: 85.38
- 최고 점수: Stage 3 Target 21, 94.1점
- Stage 1: 24 / 25 성공, 평균 80.33점
- Stage 2: 13 / 13 성공, 평균 86.25점
- Stage 3: 32 / 33 성공, 평균 88.87점

Stage 1은 평균 속도 비중이 비교적 커서 빠른 주행이 유리하다. 반면 현재 알고리즘은 속도보다 정확한 주차, orientation, steering reversal 감소에 더 초점을 맞추고 있다. 그 결과 Stage 2와 Stage 3에서 더 높은 점수가 나왔다.

강화학습 적용 후 반복적으로 실패하던 Stage 3의 한 케이스가 성공으로 바뀌었다. 특히 line collision으로 실패하던 케이스가 score 90점 이상으로 성공한 점이 의미 있었다.

## 한계점

현재 planner는 full Hybrid A*가 아니다. A* 기반 경로와 waypoint sequence를 사용하기 때문에, 차량의 연속적인 회전 반경을 완벽하게 반영하지 못한다. 일부 케이스에서는 차량이 실제로 따라가기 어려운 경로가 생성될 수 있다.

또한 학생 알고리즘을 simulator에서 불러올 때 제한 시간이 있다. 더 많은 후보 경로를 만들거나 복잡한 trajectory optimization을 적용하고 싶었지만, 초기 계산이 길어지면 timeout이나 연결 실패가 발생할 수 있었다. 그래서 계산량이 큰 방법 대신, 가볍고 설명 가능한 A* 기반 접근과 단순한 주차 sequence를 사용했다.
