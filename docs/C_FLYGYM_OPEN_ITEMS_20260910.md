# FlyGym/CUDA 미완료 항목과 오류 점검 — 2026-09-10

점검 기준은 `c1daedb6b5709f5ba7e22b3d7410f491530ef91c`이다. 구현 코드와 기존 결과를 읽고, 격리된 작은 신경 회로와 실제 CUDA/MuJoCo-Warp 몸으로 배치 경계 조건을 실행했다. 구현 코드는 수정하지 않았다. 기존 서버의 실험에는 명령을 보내지 않고 HTTP 상태만 확인했다.

현재 서버는 응답하며 신경 백엔드는 `exp_lif_cuda`, 장치는 RTX 4080이다. 신경 및 연구 모델의 자동 장치 선택도 CUDA로 확인했다. 기본 물리는 CPU이며 Warp는 선택하는 실험 경로다. Windows에서 MPS 사용 불가로 표시되는 것은 정상적인 장치 판정이다. [현재 실행 환경 확인](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flygym-audit-20260910/runtime-readiness.json).

**이번에 재현한 코드 오류는 2건이다.** 첫 번째는 의도적으로 준비 단계 예외를 발생시켜 확인한 오류 처리 결함이며, 정상 입력에서 저절로 같은 예외가 발생했다는 뜻은 아니다.

| ID / 우선도 | 문제 | 재현 결과와 영향 | 수정이 필요한 부분 |
|---|---|---|---|
| E1 / 높음 | 실패한 CUDA 배치가 추가 실행을 받음 | 2세계가 5 ms까지 진행한 뒤 세계 1의 준비 단계에 예외를 주입했다. 두 세계가 `FAILED`가 되어도 `active_worlds`에 남았다. 다시 실행하면 두 몸과 신경이 10 ms까지 진행한 후 `Encoder/control clock mismatch`가 발생했다. 세계 1의 논리 시각은 여전히 5 ms여서 물리 시각과도 달라졌다. | 실패 후 추가 실행 차단, 활성 목록 정리, 일부 세계만 갱신된 준비 상태의 처리, 화면의 실패 상태 갱신. |
| E2 / 중간 | 전 세계 일시정지 상태의 복원에서 배치 이력 번호 감소 | 1세계를 5 ms에서 일시정지해 저장·복원하면 `epoch: 1 → 0`으로 바뀐다. 이 검사에서 세계 상태와 논리 시각은 유지됐다. 저장 전후 이력 메타데이터는 일치하지 않는다. | 활성 세계가 없는 복원에서도 저장된 epoch를 그대로 보존. |

E1의 [실행 진입과 예외 처리](C:/Users/ckthd/Dev/Projects/fly_lab/flylab/c/batch.py:60)는 활성 세계 목록만 검사하고 실패 상태를 차단하지 않는다. 앞 세계의 준비가 끝난 뒤 뒤 세계에서 예외가 나면 일부 encoder 상태도 이미 바뀐다. 단일 엔진의 [기존 정지 검사](C:/Users/ckthd/Dev/Projects/fly_lab/flylab/c/engine.py:278)와 동작이 다르다. 화면의 [배치 버튼 상태 처리](C:/Users/ckthd/Dev/Projects/fly_lab/src/c/app.js:207)도 배치 존재 여부만으로 실행 버튼을 활성화한다.

E2는 [복원 시 epoch에서 1을 빼는 코드](C:/Users/ckthd/Dev/Projects/fly_lab/flylab/c/batch.py:132)와 [활성 세계가 없으면 바로 반환하는 코드](C:/Users/ckthd/Dev/Projects/fly_lab/flylab/c/batch.py:39)의 조합이다. 기존 배치 복원 검사는 실행 중인 세계가 있는 경우여서 이 조건을 다루지 않았다.

두 결과는 [배치 재현 보고서](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flygym-audit-20260910/batch-edge-report.json)에 있다. [재현 스크립트](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flygym-audit-20260910/reproduce_batch_edges.py)는 6뉴런 시험 회로를 사용한다. 전뇌 장시간 실행에서의 오류 발생 빈도를 측정한 것은 아니다.

**구현은 있지만 검증에 실패했거나 성능 목표에 도달하지 못한 항목은 다음과 같다.** 아래 수치는 기존 원자료를 이번에 다시 확인한 값이며, 이번 점검에서 해당 장시간 실험을 다시 실행하지는 않았다.

| ID | 항목 | 확인된 상태 | 남은 일 |
|---|---|---|---|
| V1 | CPU↔Warp 물리 일치와 미래 상태 복원 | 5시드 모두 기존 CPU↔Warp 허용값을 초과했다. 같은 Warp 체크포인트에서 재개한 미래 qpos의 완전 일치도 5시드 모두 실패했다. | 접촉·적분·저장 상태 차이 원인 규명과 허용 범위 검증. 채택 기준을 통과하기 전 기본 물리 전환은 미완료. |
| V2 | CUDA 물리의 단일 세계 성능 | 새 resident 경로는 이전 Warp보다 빨라졌으나, 20 ms 몸·제어 구간의 wall 중앙값은 CPU 0.03789 s, resident 0.17935 s로 약 4.73배 느리다. | 전체 신경·감각·기록·화면을 포함한 병목 개선과 60모델초 실시간 기준 검증. |
| V3 | BANC 회로에 의한 보행 | 8조건 모두 실행은 완료했지만 보행 실패·고착 검출. 10초 동안 최대 순이동 약 0.438 mm로 기존 5 mm 기준에 못 미쳤다. 판정은 `REJECTED_ALL_CANDIDATES`. | 감각→운동뉴런 모집, 운동단위 합산, 접착·발 들림의 효과를 분리해 원인 규명. |
| V4 | 단일 관절 감각 피드백과 외력 회복 | 해당 두 근육의 정상 신경 모집 활성도가 0이었다. 정상 감각·차단·수동 조건의 외력 반응이 같았다. 기계 교사의 외력 후 누적 오차도 수동 대비 약 1.56–1.70배였다. | 감각 입력이 실제 운동 출력을 바꾸는지 검증하고 양방향 외력 회복 기준 통과. |

V1에서 유지한 허용값은 qpos 최대 원소 차이 0.001, qvel 0.1, 발 힘 0.15 BW다. qpos/qvel에는 서로 다른 종류의 일반화 좌표가 섞이므로 모든 값을 rad 또는 rad/s로 해석할 수 없다. **CUDA 제어 커널 자체의 대조는 통과**했으며, 신경 CUDA 지원 실패로 분류하지 않는다. 또한 V2는 5시드·warmup 이후 3반복의 짧은 몸·제어 측정이다. 전뇌 전체 처리율과 구분한다. [Warp 검증·성능 원자료](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flygym-remaining-20260910/resident-verify-01/report.json).

V3와 V4는 별도 BANC 연구 실험의 결과다. CPG/hybrid 보행 제어기나 PPO 정책의 실행 성공으로 대체할 수 없다. [BANC 보행 결과](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flygym-implementation-20260910/banc-gait/compact-report.json), [같은 관절의 교사·신경 대조 결과](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flygym-implementation-20260910/joint-teacher/report.json).

**아직 구현·연결·교정이 완료되지 않은 확장 항목도 있다.**

| ID | 항목 | 현재 구현 범위 | 남은 범위 |
|---|---|---|---|
| M1 | 전신 근육 제어 | 고정된 왼쪽 앞다리의 공식 15 MTU 모방학습 도구와 별도 BANC 실험 | 지면 접촉을 포함한 근육 제어, 양측·여섯 다리 확장, 검증된 신경 회로와 근육 정책 연결. |
| M2 | 복안의 낱눈별 신경 연결 | 공식 Retina와 100/200 Hz 입력, 눈 전체 평균 명암을 R1–6 집단에 전달 | 검토된 낱눈↔실제 뉴런 retinotopy 자료, motion/loom의 검토된 신경 포트 연결. 현재 motion/loom은 계산만 하며 기본 프로파일의 별도 신경 입력으로 연결되지 않는다. |
| M3 | 후각의 해부학·화학 교정 | 네 위치 food/hazard 농도와 반응, plume·적응, 기본 palp food 신경 포트 | palp 실제 위치 교정, 수용기별 화학 반응·이득 검증, palp hazard 포트. generic hazard를 geosmin으로 취급하는 가정도 미검증. |
| M4 | 배치 캠페인 스케줄러와 메모리 부족 대응 | 같은 몸·지형·물리 설정의 1–32세계, 세계별 정지·재개·취소와 저장 | 모델 해시별 여러 작업을 묶는 캠페인 큐, VRAM 부족 시 배치 크기를 줄여 재시도하는 경로. 현재 생성 실패는 예외로 끝나며 자동 축소는 없다. 실제 OOM을 발생시켜 검사하지는 않았다. |
| M5 | FlyBody의 후속 동작 확장 | legs-only 42 DOF 보행 어댑터 | 자유 비행·이착륙. 초기 개선안의 후순위 확장에 해당한다. |

M1의 CUDA PPO 실행은 128 step 후 64 step 이어학습, 합계 192 step까지 확인했다. 학습·평가 모두 같은 `0002` clip을 사용했고 정책 수렴, 독립 clip 성능, 외력 회복은 입증되지 않았다. [근육 실험 구현](C:/Users/ckthd/Dev/Projects/fly_lab/flylab/c/muscle_imitation.py:7), [학습·평가 실행 도구](C:/Users/ckthd/Dev/Projects/fly_lab/tools/train_c_muscle_imitation.py:20).

M2/M3는 외부 retinotopy 및 site calibration을 받는 인터페이스까지는 구현됐다. 필요한 근거 데이터가 실제로 반영된 상태와 구분해야 한다. 현재 기본 프로파일의 [미연결 관측과 교정 상태](C:/Users/ckthd/Dev/Projects/fly_lab/tools/build_c_multimodal_profile.py:57)에 이 한계가 기록되어 있다. M4의 [배치 구성](C:/Users/ckthd/Dev/Projects/fly_lab/flylab/c/batch.py:16)은 한 구성으로 모든 세계를 만든다. [Warp 묶음](C:/Users/ckthd/Dev/Projects/fly_lab/flylab/warp_batch.py:43)은 동일한 형상·설정만 허용한다. M5는 [FlyBody 모델 생성](C:/Users/ckthd/Dev/Projects/fly_lab/flylab/body_options.py:43)에서 확인했다.

**이미 있는 기능도 검증 범위는 더 넓혀야 한다.**

- 전뇌 32세계는 세계당 10 ms와 복원 후 5 ms 추가 실행까지 확인했다. 이 검사에서는 512채널 기록을 껐다. 전뇌 2세계에서의 512채널 검사와 32세계 장시간 기록·화면 동시 실행을 구분해야 한다.
- 지형·FlyBody 제어기 비교는 주로 50–100 ms다. 장시간 장애물 통과, 다양한 시드와 지형 조합의 안정적인 보행은 미검증이다.
- 운동학의 발끝 오차는 실제 root를 공유한 조건부 FK 값이다. 측정된 전신 이동 궤적과의 일치 검증이 남아 있다. gain 교정도 같은 clip의 별도 시간 구간만 사용했다.
- 연속 관측의 3/10 FPS는 설정 상한이다. 60모델초 동안 신경·물리·512채널 기록·화면을 포함해 sim/wall ≥ 1을 유지하는 완료 기준은 아직 통과하지 않았다.
- 최근 전체 회귀 기록은 300개 중 272 통과·28 생략이다. 별도 native/CUDA 7개는 모두 통과했으며 3개는 일반 회귀와 겹친다. 이번 E1/E2 경계 조건은 이 기존 검사에 포함되지 않았다. MPS 실제 장치 검증도 Windows에서는 수행하지 않았다.

이 검증 범위와 수치는 [후속 구현 보고서](C:/Users/ckthd/Dev/Projects/fly_lab/docs/C_FLYGYM_REMAINING_20260910.md)에 기록되어 있다. 연결된 원자료 22개의 SHA-256을 이번 점검에서 모두 다시 계산해 일치함을 확인했다. [점검 근거 목록](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flygym-audit-20260910/audit-evidence.json).

**과거 오류 중 현재 미해결 목록에서 제외한 항목:** FlyBody의 `l_funiculus` 센서 이름 오류는 모델별 실제 분절 이름으로 수정됐고, 별도 실제 감각 실행으로 확인됐다. 10번 이상 세계의 JSON 문자열 정렬 때문에 나던 `Active world order mismatch`도 숫자 순서 복원으로 수정돼 32세계 복원·추가 실행이 완료됐다. 이 문제는 E2의 전 세계 일시정지 epoch 오류와 다르다.

자동 Retina, plume, 15 MTU 실험 도구, GPU 내부 보행 제어, 전뇌 CUDA 배치, 복잡 지형, FlyBody 보행, 연속 관측 화면은 이미 반영되어 있다. 이전 단계 문서의 해당 미구현 표시는 당시 조사 기록이다. 현재 미완료 항목은 위의 오류 수정, 신경·근육·감각 연결, 교정 및 검증 조건이다.

수정 순서는 E1의 실패 후 진행 차단과 E2의 복원 이력 보존을 먼저 처리하고, Warp 물리·복원과 BANC 감각→운동 모집을 각각 조사하는 것이 적절하다. retinotopy·후각 교정과 여섯 다리 근육 확장은 필요한 근거 및 독립 검증 자료를 갖춰 진행해야 한다.
