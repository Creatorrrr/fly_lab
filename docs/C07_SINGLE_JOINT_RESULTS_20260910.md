# C0.7 후속 결과: BANC와 단일 관절 Hill 근육 연결

**새 `flylab.single-joint.v1` 엔진을 구현하고 실제 MPS 신경 계산·MuJoCo 물리로 실행했다.** BANC v888의 158,706개 선언 뉴런을 계속 계산하면서 왼쪽 앞다리 Fe–Ti 관절의 두 빠른 운동뉴런을 Hill 근육에 연결한다. 관절 움직임은 106개 FeCO 감각 뉴런의 입력으로 돌아간다.

**기술 실행은 통과했지만, 감각 피드백에 의한 운동·안정화는 실패했다.** 기본 입력과 4배 입력 모두 자유롭게 움직이는 관절에서 두 운동뉴런을 모집하지 못했다. 이 결과는 완전한 가상 초파리나 M2 생리 교정 완료가 아니다. 기존 브라우저의 42관절 목표각 모델은 계속 사용되며, 새 엔진은 별도 Python/CLI 실험 장치다.

![실제 기록으로 그린 단일 관절 결과](../verification/c07_followup_20260910/figures/single_joint.png)

## 1. 개발한 내용

| 구성 | 구현과 검사 |
|---|---|
| 기하학 진단 | 원본 14자유도와 파생 1자유도를 401개 각도에서 비교. site 경로 해석 미분·길이 유한차분·MuJoCo 모멘트암 대조 |
| 새 신경–근육 엔진 | 전체 BANC 계산 → 실제 빠른 운동뉴런 두 개의 발화율 → 활성 → Hill 힘 → 힘줄 모멘트암 → 관절 운동 |
| 감각 환류 | 관절각·속도 → 필터/지연 수용체 proxy → 실제 BANC ID에 held mV 입력. claw/hook 방향과 club 고역 성분 구분 |
| 개입 | 실제 ID 자극·발화 억제·신경 출력 차단, 감각 피드백 차단/극성 교환/추가 지연, 운동 출력 차단, 관절 외력 |
| 오류 감시 | 매 10µs 물리 step에 부호 반전/특이점 감시. 보정 힘 없이 즉시 중단하고 부분 시계와 상태 보존 |
| 복원 | 신경·물리·필터/지연 시계, 출력, 개입, 최근 관측의 일치 검사. 새 객체로 검증 후 원자적으로 교체. 실패 상태는 증빙으로만 저장 |
| 검증 도구 | 실제 대조 캠페인, 고정 자세 신경 probe, 반전 중단 실험, 저장 기록 재평가, PNG/SVG 내보내기 |

핵심 소스: [single_joint.py](../flylab/c/single_joint.py), [muscle_calibration.py](../flylab/c/muscle_calibration.py), [muscles.py](../flylab/c/muscles.py), [receptors.py](../flylab/c/receptors.py).

## 2. 모멘트암 부호 반전의 원인

굴근/신근은 두 개의 고정 femur site를 거쳐 tibia의 움직이는 site로 이어진다. 신근의 마지막 선분과 회전축의 관계가 바뀌면서 **q = 0.61343969465rad**에서 힘줄 길이 미분이 0이 되고 그 아래에서 부호가 반전된다. MuJoCo의 tensile force는 음수이며 `관절 토크 = dL/dq × 근육 힘`이다. 힘의 부호를 임의로 뒤집을 문제가 아니다.

| 대조 | 최대 절대 오차 |
|---|---:|
| 해석 기하와 MuJoCo 모멘트암 | 3.00e-16mm |
| 길이 유한차분과 MuJoCo 모멘트암 | 2.05e-10mm |
| 원본/파생 모멘트암 | 2.98e-16mm |
| 원본/파생 위치 | 8.88e-16mm |

원본 XML과 파생 장치에서 동일하므로 **C0.7의 관절 고정 변환이 만든 오류는 아니다.** 기존 장치의 모델 hash `d7ee1e33…4e0810`도 변하지 않았다. 제공된 0002 동작 clip의 q 범위는 0.647705~2.226199rad여서 이번 반전점은 그 clip 밖에 있다. 하지만 원본 허용 범위 0.4789~2.502rad 안이다. 단일 clip만으로 생리적 허용 범위를 좁히지 않았다. 원본 전체 외형/접촉 모델을 검증했다는 뜻도 아니다.

원 모델의 공식 설명은 LF 다리만 근육으로 구동하는 실험 모델임을 명시한다. 원 논문은 여러 표본의 구조와 운동학으로 구축·최적화한 모델이며 Fe–Ti에서는 빠른 굴근·신근 MTU를 사용한다. 이는 모든 운동단위의 직접 측정 파라미터가 확보됐다는 의미가 아니다. [FlyGym 공식 문서](https://neuromechfly.org/tutorials/6_muscle_imitation/), [FlyMimic 원 논문](https://arxiv.org/html/2509.06426v1)

### 실제 중단 반례

동일한 신근 흥분 0.35를 준 두 실행을 보존했다. 감시를 끄면 1ms 관측 10개에서 부호가 반전됐고, 마지막 상태에서는 다시 정상 부호로 돌아왔다. 따라서 종료 시점만 검사하면 문제를 놓친다.

새 감시는 **82.25ms, q = 0.613422rad**에서 최초 반전을 감지해 중단했다. 중단 전 83개 공통 관측의 각도 차이는 정확히 0이었다. 새 감시는 이전 궤적을 교정하지 않는다. 원본 경로에 대한 생리적 교정 필요 상태도 유지한다. [원자료](../verification/c07_followup_20260910/guard_probe/report.json)

## 3. 실제 뉴런 대응과 모델 가정

| BANC v888 cell type | 실제 root ID | Hill actuator |
|---|---|---|
| tibia_flexor_Fast | 720575941481179066 | LFTibia_flex_93434 |
| tibia_extensor_FETi | 720575941639281525 | LFTibia_extensor_93932 |

이 둘은 BANC의 left/front_leg/peripheral_target_type annotation으로 검증한다. actuator 이름 끝 숫자는 뉴런 ID가 아니다. SETi·accessory flexor와 다른 근육을 이 두 MTU에 뭉뚱그려 연결하지 않았다. 해당 뉴런들은 여전히 전체 신경 계산에 포함된다.

감각은 claw 27개, 방향 가정이 있는 hook 24개, club 55개로 총 106개다. `SNpp50 → q 양의 방향`, `SNpp51 → q 음의 방향`, `SNpp41/39 → 양/음 속도`는 기존 v2의 cell type 전이 가설이다. 이번에 BANC 실측 tuning을 확보한 것은 아니다. 전용 접촉 장치가 없으므로 load/touch 입력은 추가하지 않았다. 신경망의 NT 예측값·중앙 연결 가중치·미확정 가중치 0 정책도 변경하지 않았다. 관찰된 뉴런 신원과 미측정 NMJ/모집 모델을 구분한다.

모집식 `u = rate / (100Hz + rate)`는 초기 공학 가정이며 50/200Hz 대조도 기록했다. 신경 0.1ms, 물리 0.01ms, 제어/감각 1ms다. 시각 t에서 관절을 관측하고 **t의 운동 발화율**을 다음 물리 구간에 유지한다. 신경을 t+1ms까지 먼저 계산하더라도 그 미래 출력으로 이미 진행 중인 물리 구간을 구동하지 않는다.

50/200Hz 대조에서는 운동뉴런 자체가 발화하지 않았으므로, 이 두 결과로 모집 곡선의 타당성이나 안정성을 판단하지 않는다.

최소 흥분 0.0001과 활성 감쇠·수동 힘은 원본대로다. 운동 차단은 이 하한을 초과하는 추가 명령을 차단한다. 모든 힘/토크는 **원본 모델 단위**로 기록하며 SI 변환은 `null`이다. 접촉, 6다리 근육 구동, 생리적 최대 힘·모집 곡선은 미검증이다.

## 4. 실제 검증 결과

| 검증 | 실행/결과 | 해석 |
|---|---|---|
| 전체 회귀 | **253/253 PASS**, skip 0 | 기존 245개 + 새 단일 관절/판정 8개. 실제 MPS·MuJoCo 검사 포함 |
| 기본 감각 이득 18mV | 22회 × 0.15초 = 3.3 모델초 | 모든 조건 완료, 원래 13개 기술 gate PASS |
| 별도 감각 이득 72mV | 22회 × 0.15초 = 3.3 모델초 | 모든 조건 완료, 동일 기술 gate PASS |
| 복원 후 계산 | 각 44회에서 1ms 진행→복원→같은 1ms 재진행 | 전체 신경·물리·감각 상태 정확히 일치; 추가 계산 합 0.088 모델초 |
| 고정 자세 신경 probe | 3각도 × 3이득 × 출력 차단 2 = 18회, 각각 0.3초 | 실제 MPS 신경 계산 5.4 모델초. **물리 실험은 아님** |
| 반전/감시 대조 | 0.1초 반례 + 0.08225초 중단 | 기술 검출 4/4 PASS; 원본 경로 교정 완료는 아님 |
| 원자료 재평가 | 두 캠페인 각각 일치 | 힘 투영·역학 잔차 독립 재계산, 체크포인트 파일 hash 확인. 차단/억제와 무구동의 전체 물리 state도 정확히 일치 |

총 44회 물리 캠페인은 두 이득 설정에서 같은 22조건을 반복한 것이다. 44개의 서로 다른 행동 과제를 통과했다는 의미가 아니다. 원자료 재평가를 새 물리 실행으로 더하지 않는다. 전체 회귀 뒤 추가된 plotting/원자료 감사 도구는 실제 파일로 실행했다.

### 직접 운동 자극은 연결됐다

10~50ms 동안 20mV로 직접 자극하면 각 운동뉴런이 3회 발화한다. 150ms의 관절각은 무구동 1.623259rad, 굴근 자극 1.682404rad, 신근 자극 0.870349rad다. 작은 q가 더 큰 실제 관절 내각에 대응하는 것도 기하학 검사로 확인했다. 자극 종료 뒤 남는 rate·활성도·힘도 기록한다. 발화 억제와 운동 차단 대조에서는 해당 추가 운동이 사라졌다.

### 감각 피드백에 의한 운동은 실패했다

기본 18mV에서 감각 활동과 중앙 신경 전파는 생겼지만 두 빠른 운동뉴런의 발화는 0이었다. 72mV, 방향 교환, 5ms 지연, 다른 초기각, 모집 이득 50/200Hz에서도 자유 관절의 피드백과 피드백 차단 궤적 차이가 0이었다. `sensorimotor_status = FAIL_NO_FEEDBACK_MOTOR_EFFECT`로 판정한다.

외력 후 편차는 약 0.00543rad에서 최종 약 0.00066~0.00069rad로 줄었다. 그러나 피드백을 끈 비교에서도 정확히 같았다. **이 회복은 기준 역학과 최소 활성의 결과이며 신경 피드백의 안정화 성공으로 인정하지 않는다.** 첫 기록의 `PASSIVE_ONLY_IN_THIS_EXPERIMENT` 표기는 최종 감사에서 `BASELINE_MECHANICS_ONLY`로 명확히 했으며, 원본 파일은 보존했다.

## 5. 발화하지 않는 원인 추적

고정 자세에서는 감각 이득 18/36mV일 때 두 운동뉴런 모두 발화하지 않았다. 72mV에서 q=1.862와 2.2rad의 신근만 각각 2회/3회 발화했고 감각 출력 synapse를 차단하면 사라졌다. q=1.2rad에서는 둘 다 발화하지 않았다.

고정 q=1.862, 18mV의 최종 신경 전류 상태 h는 굴근 −3.388mV, 신근 +1.602mV였다. 72mV에서는 −25.897mV와 +7.661mV였다. 입력 증가가 양쪽을 균형 있게 활성화하지 않았으며, 굴근 쪽은 억제 방향으로 더 크게 변했다. 이것은 현재 부호·가중치·LIF·감각 가정 아래의 결과다. 실제 생물학에서 그 운동뉴런이 동일하게 억제된다는 결론은 아니다.

고정 자세의 신근 첫 발화는 1ms 관측 경계에서 101ms/145ms에 확인됐다. 움직이는 장치는 그 전에 q가 내려가므로 동일한 고정 자극을 받지 않는다. **고정 자세의 반응을 자유 관절의 폐루프 성공으로 전용할 수 없다.** 기본 감각 이득을 72mV로 바꾸지 않았다.

전압은 1ms 경계의 샘플이고 발화는 0.1ms 신경 step으로 계산한다. 최초 probe 원자료의 `motor_peak_voltage_mV`는 경계에서 관측한 최댓값이며, 발화 직전 실제 최고 전압이 아니다. 새 도구는 `motor_sampled_peak_voltage_mV`와 별도 해상도·발화 시각을 명시한다.

## 6. 실행 방법

현재 프로젝트의 `.venv`와 기존 검증된 BANC bundle을 사용한다. 모든 `--out`은 새 경로여야 한다. 같은 경로에 결과를 덮어쓰지 않는다.

```bash
.venv/bin/python tools/verify_c_muscle_geometry.py --out verification/my_joint_geometry
.venv/bin/python tools/verify_c_muscle_guard.py --out verification/my_joint_guard
.venv/bin/python tools/verify_c_single_joint.py --out verification/my_joint_native
.venv/bin/python tools/verify_c_single_joint.py --require-feedback-effect --out verification/my_joint_feedback_gate
.venv/bin/python tools/verify_c_single_joint.py --sensory-gain 72 --out verification/my_joint_gain72
.venv/bin/python tools/probe_c_joint_feedback.py --out verification/my_joint_fixed
.venv/bin/python tools/reevaluate_c_joint.py --source verification/my_joint_native --out verification/my_joint_audit
.venv/bin/python tools/run_c_checks.py --tier regression --out verification/my_joint_regression
```

`SingleJointLoop.step(stimulation={실제_ID: mV}, suppress_ids=[...], mute_ids=[...], feedback=True, motor_connected=True, torque=...)`는 한 제어 구간을 실행한다. `frame()`에 선택 뉴런 전압/발화율/누적 발화, 실제 근육 힘·활성·길이·토크, 감각, 부분 시계/오류가 포함된다. 개입은 해당 구간에 적용되며 지속 자극은 구간마다 전달한다. `snapshot()/restore()`는 NPY 기반 StateStore와 연결한다. fault checkpoint는 분석용이며 제어 재개는 건강한 checkpoint에서 한다.

신경은 MPS, 단일 세계의 MuJoCo 물리는 CPU다. 이 작업에서 새로운 브라우저 실험 화면은 추가하지 않았다. 8772~8775의 기존 서버는 모두 HTTP 200을 확인했으며 재시작/실험 초기화를 하지 않았다. 이는 새 브라우저 회귀나 전신 근육 보행 검증은 아니다.

기본 CLI 종료 코드는 기술 실행 기준이며 실패한 감각 효과도 별도 출력한다. `--require-feedback-effect`를 지정하면 기술 검사가 통과해도 감각에 의한 실제 운동 효과가 없을 때 종료 코드 1을 반환한다. 현재 두 감각 이득의 기록은 이 추가 기준에 실패한다.

## 7. 다음 개발의 구체적 경계

1. **양방향 감각→운동 모집 교정:** 고정 자세/속도별 claw/hook 입력과 각 motor의 막전위·입력 부호·활성 presynaptic 경로를 분리한다. NT 예측의 낮은 신뢰도, 빠른/느린 운동단위의 서로 다른 임계값, 하행 구동 누락을 각각 별도 가설로 검사한다. 가중치를 한꺼번에 키우거나 운동 출력에 임의 보정각을 더하지 않는다.
2. **국소 안정화 조건:** 교정 후보별로 같은 외력에 대해 피드백 정상/차단/출력 차단, 양 방향과 미사용 자세를 비교한다. 현재 수동 회복과 구분되는 효과가 없으면 한 다리·6다리로 확장하지 않는다.
3. **힘·경로·감각의 측정 근거:** SI 단위, Fmax/수축 속도/모집 자료, FeCO tuning, 0.61344rad 근처의 실제 경로·사용 범위를 확보해야 한다. 원본 OpenSim 개발 저장소 링크는 이번 확인에서 404였으며, 힘줄을 임의로 재배치하지 않았다.
4. **한 다리 이후:** 실제 힘 모델을 가진 한 다리 접촉부터 다관절·양측·6다리로 확대하고 기존 지속 보행/고착/외란 기준을 다시 적용한다. 아직 구현하지 않은 범위다.

## 8. 증빙 묶음

- 원자료: `verification/c07_followup_20260910/` — 실행별 profile, 전체 최종 신경/물리/필터 checkpoint, 1ms trace, 고정 자세 probe, 실패 반례, regression log, 감사 결과, PNG/SVG.
- 검토 ZIP: `artifacts/releases/FLY_LAB_C07_JOINT_REVIEW_20260910.zip` — 최종 소스와 주요 결과/그림.
- 전체 증거 ZIP: `artifacts/releases/FLY_LAB_C07_JOINT_EVIDENCE_20260910.zip` — 이번 실행 원자료와 최종 소스. 전체 BANC 원본 bundle·설치 환경·기존 C0.6/C0.7 전체 증거 ZIP은 중복 포함하지 않는다. 재계산에는 명시된 graph hash의 기존 bundle과 런타임이 필요하다.
- 각 ZIP 옆 `.index.json`과 내부 `EVIDENCE_MANIFEST.json`에 SHA256을 기록한다. 기존 C0.7 최초 결과·소스 ZIP은 변경하지 않았다.
