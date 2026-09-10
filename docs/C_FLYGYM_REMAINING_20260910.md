# FlyGym 잔여 확장 적용과 검증 (2026-09-10)

남아 있던 기능을 실행 가능한 모듈·명령·화면으로 추가했다. 기본은 **신경 CUDA 자동 선택 + CPU MuJoCo 물리**다. Warp와 새로운 감각·운동학·근육·대체 몸 실험은 명시적으로 선택한다. 기능 구현 완료와 생물학적 성공, 물리 백엔드 채택은 별도의 판정이다.

## 반영 항목

| 항목 | 구현과 사용 | 확인 범위 |
|---|---|---|
| 기록 운동학 재생·교정 | `kinematics.py`, `locomotion_experiments.py`, `run_c_flygym_experiments.py`, `calibrate_c_kinematics.py` | 공식 Spotlight 42관절 재생, 각도·속도·조건부 FK 발끝 오차·실제 actuator 힘·receptor 출력. 초기화 뒤에는 물리 적분으로만 상태가 바뀐다. |
| rule / CPG / hybrid 비교 | 같은 물리 몸·시드에서 개별 실행, phase·접촉·지지·미끄럼 원자료 NPZ | 모델 과제 성공과 기계적 실행 완료를 분리한다. |
| GPU 내부 보행·반사·접촉 | `hybrid_cuda.cu`, `warp_control.py`, `body_warp_resident.py`; `--physics-backend warp --physics-control cuda` | 10 kHz 제어·물리를 CUDA Graph에 묶고 보통 200 Hz에 CPU 관측을 갱신. CuPy/Warp가 같은 버퍼와 스트림을 사용한다. |
| 실제 GPU 타임라인·메모리 | `profile_c_gpu.py`, `gpu_profile.py` | Nsight kernel node, 복사, 동기화, NVTX 구간, process CUDA allocation 최고값과 device 전체 표본 최고값을 구별한다. |
| 자동 복안 신경 입력 | `retinal_input.py`, `bindings-multimodal-v2.json` | 공식 Retina 2눈 × 721낱눈 × 2채널. 100/200 Hz 논리 시계, 명암·시간변화·어두운 면적 증가 proxy, 필터와 이전 영상 복원. 샘플 tick 생략을 거절한다. |
| 후각 네 위치·시간변화 | `FourSiteOdor`, `CSensorAdapter`, `build_c_multimodal_profile.py` | 좌우 antenna/palp의 음식·위험 수치 채널, Gaussian/역제곱/주기적으로 이류하는 plume, 감각 적응·복원·차단. |
| 연속 실제 메시·감각 화면 | 화면의 연속 관측, 영상 종류, 최대 3/10 FPS | 재생을 멈추지 않고 별도 모델에서 관측한다. 표시 FPS와 100/200 Hz 신경 샘플링은 독립이다. |
| 15 MTU 모방학습 | `c/muscle_imitation.py`, `train_c_muscle_imitation.py`, `requirements-imitation.txt` | 공식 고정된 왼쪽 앞다리 모델, 제한된 역동역학 교사, PPO 학습·평가·이어학습. 가능한 경우 CUDA를 선택한다. |
| 복잡 지형 | `BodyOptions`, 화면의 지형 선택 | 평면·틈·블록·혼합·경사. 실제 지지 접촉 법선과 접선 방향 미끄럼을 기록한다. |
| FlyBody 보행 어댑터 | `BodyOptions(model="flybody")`, 화면의 새 실험 몸 | 공식 legs-only 42 DOF와 FlyBody 보행 궤적, 접착 해제, 다중 geom 접촉 집계, 모델별 복안·후각 위치. 비행은 포함하지 않는다. |
| 전뇌 GPU 배치 통합 | `c/neural_batch.py`, `warp_batch.py`, `c/batch.py`, `run_c_cuda_batch.py`, 화면의 CUDA 배치 | topology 공유, 세계별 weight/LIF/queue/센서/몸/개입/기록 소유. 세계별 중지·재개·취소, 선택 세계 GPU 렌더링, 전체 체크포인트 저장·복원. |

현재 로컬 새 감각 프로파일은 `data/acquisitions/local-windows-20260910/bindings-multimodal-v2.json`이다. 사용자 기존 프로파일과 진행 상태를 이 프로파일로 바꾸지 않는다. 배치는 현재 단일 실험과 독립이다. 오른쪽 신경 패널의 집단·자극 설정을 선택 세계 기록·개입에 사용할 수 있다. 현재 배치를 저장하고 닫은 뒤 다른 배치를 준비하거나 복원한다. 기록 파일은 닫을 때 완료 처리되며, 복원 뒤 기록을 다시 시작하면 새 구간을 만든다.

## 측정 결과

- Spotlight 50 ms 재생: 관절 RMSE **0.0404795 rad**, 속도 **10.3297 rad/s**, 실제 root를 공유한 FK 발끝 **0.0990364 mm**. 원래 기록의 전신 이동을 맞춘 결과는 아니다.
- servo gain 후보 0.5/1/2 중 2를 앞 50 ms에서 선택했다. 같은 clip의 별도 0.5–0.55 s 구간에서 gain 1의 **0.0299874 rad**가 gain 2의 **0.0190096 rad**로 감소했다. 기본 gain은 바꾸지 않았고 독립 개체 검증으로 표시하지 않는다.
- 공식 LF 15 MTU 0.1 s: 수동/교사 관절 RMSE **0.392514 / 0.381903 rad**. CUDA PPO **128 학습 step + 50 평가 step** 실행과 정책 저장을 확인했다. 같은 clip을 사용했으며 수렴·외력 회복·독립 검증의 증거가 아니다.
- FlyBody에서 hybrid/CPG/rule 각각 50 ms가 실제 물리로 완료됐다. 복안 `(2,721,2)` 및 후각 네 위치도 실제 FlyBody에서 확인했다. NeuroMechFly의 블록은 세 제어기 × 50 ms, 평면/틈/혼합/경사는 hybrid × 100 ms 실행이다. 장시간 지형 통과나 BANC 보행 성공 판정은 하지 않았다.
- 실제 FAFB **139,255 뉴런 × 2세계**에서 자동 Retina, plume 후각, 512채널 기록, 선택 GPU 렌더링과 전체 체크포인트를 포함해 세계당 50 ms 실행했다. 해당 전뇌 체크포인트를 다시 읽어 세계당 70 ms까지 진행했다.
- 실제 FAFB **139,255 뉴런 × 32세계**에서도 세계당 10 ms, 자동 감각, GPU 렌더링, 507,873,283-byte 전체 체크포인트 저장을 실행했다. 세계별 512채널 기록은 이 32세계 검사에서는 껐다. 복원 후 모두 15 ms까지 진행했다. 10번 이상 세계의 JSON 문자열 정렬로 인한 복원 순서 오류를 이 검사에서 찾아 수정했다. 최고 VRAM 표본은 다른 프로세스를 포함해 **9,551,872,000 bytes**였다. 짧은 기능·용량 검사이며 처리율 채택 결과는 아니다.
- GPU 보행 kernel 대조: 5시드 × 전진/후진/정지에서 controller state 최대 차이 **0**, action 최대 차이 **1.19e-7** 이하였다. 접착 상태도 일치했다. 잘못된 제어 시계는 상태 변경 전에 거절한다.
- 5시드의 20 ms 물리 구간, graph warmup 뒤 3반복 중앙값을 비교했다. 시드별 중앙값들의 중앙값은 기본 CPU **0.0379 s**, 조건을 맞춘 CPU **0.0364 s**, 기존 Warp+CPU제어 **1.5563 s**, 새 Warp+CUDA제어 **0.1794 s**다. 기존 Warp 경로 대비 약 **8.6배**, 기본 CPU 대비는 느리다. 이는 몸·제어만의 짧은 측정이며 전뇌/UI 처리율은 아니다.
- 기존 채택 허용값 `qpos 0.001`, `qvel 0.1`, 발 힘 `0.15 BW`를 유지했다. **5시드 모두 CPU↔Warp 물리 비교는 실패**했다. 같은 Warp 체크포인트 이후 미래 qpos의 완전 일치도 실패했다. 제어 kernel의 일치가 물리 적분의 일치를 뜻하지 않으므로 기본값으로 채택하지 않았다.
- Nsight node trace: kernel **62,456개**, 메모리 복사 **3,341개**, synchronization **1,090개**를 수집했다. 계측 process의 CUDA allocation 최고값 **103,132,877 bytes**는 pinned host/driver 등의 메모리를 제외한다. 2세계 실행의 device 전체 표본 최고값 **2,785,583,104 bytes**와 측정 범위가 다르다. NVTX 시간 겹침 합은 배타적인 단계 비용 또는 wall latency로 해석하지 않는다.

원자료는 `verification/flygym-remaining-20260910/` 아래에 있다. `replay-01`, `calibration-01`, `muscle-01`, `imitation-train-01`, `blocks-01`, `terrain-and-senses-01`, `flybody-01`, `flybody-senses-02`, `full-batch-01`, `full-batch-restore-01`, `full-batch-32-01` 및 `restore-02.json`, `resident-verify-01`, `nsight-nodes-01`을 사용했다. 초기 실패 파일도 보존했다. 큰 데이터·NPZ·trace·정책 파일은 Git에 넣지 않는다.

## 아직 외부 근거가 필요한 부분

기본 visual 포트는 각 눈의 R1–6 실제 ID 집단에 눈 전체 명암 평균을 전달하는 **집단 proxy**다. 낱눈과 개별 뉴런의 retinotopy를 발명하지 않았다. `--retinotopy`의 graph hash, 실제 ID, eye/index, 근거를 명시한 포트를 제공하면 개별 낱눈 연결로 교체한다. motion/loom은 계산되지만 검토된 별도 포트를 명시하기 전에는 임의 시각 뉴런에 연결하지 않는다.

palp 위치와 broad food 반응, generic hazard를 geosmin으로 보는 화학 가정은 교정되지 않았다. 네 위치의 food/hazard 값은 계산하지만 기본 팔프 신경 포트는 MxLbN의 좌우 olfactory ORN에 broad food proxy만 전달한다. aversive palp receptor를 임의로 지정하지 않는다. `--site-calibration`으로 네 분절·offset·evidence·uncertainty를 제공할 수 있다. FlyBody는 실제 `l_antenna/r_antenna` 원점과 rostrum 기준·thorax 축의 명시적 팔프 가정 위치를 사용한다.

LF 15 MTU 모방학습은 고정된 한 다리 실험이다. 검증되지 않은 BANC 모집을 전체 여섯 다리의 근육 제어로 확대하지 않았다. 기존 BANC 보행 실패와 외력 회복 실패 판정도 그대로 유지한다. Warp 세계를 pause/resume/cancel로 재편성할 때는 integration state에서 새 batch epoch를 구성하므로, 재편성 없는 실행과 bitwise 미래 궤적 일치를 보장하지 않는다.

## 실행 예시

프로젝트 루트의 PowerShell에서 실행한다. `--out`은 기존 결과와 겹치지 않는 새 경로를 지정한다.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-imitation.txt
.venv\Scripts\python.exe -X utf8 tools/run_c_flygym_experiments.py --task replay --seconds .1 --out verification/my-replay
.venv\Scripts\python.exe -X utf8 tools/calibrate_c_kinematics.py --out verification/my-calibration
.venv\Scripts\python.exe -X utf8 tools/run_c_flygym_experiments.py --task controllers --terrain gaps --seconds .1 --out verification/my-terrain
.venv\Scripts\python.exe -X utf8 tools/run_c_flygym_experiments.py --task controllers --model flybody --seconds .1 --out verification/my-flybody
.venv\Scripts\python.exe -X utf8 tools/train_c_muscle_imitation.py --steps 2048 --eval-steps 100 --out verification/my-imitation
.venv\Scripts\python.exe -X utf8 tools/verify_c_resident.py --out verification/my-resident-check
.venv\Scripts\python.exe -X utf8 tools/profile_c_gpu.py --graph data/acquisitions/local-windows-20260910/bundle --bindings data/acquisitions/local-windows-20260910/bindings-multimodal-v2.json --physics resident --out verification/my-nsight
.venv\Scripts\python.exe -X utf8 tools/run_c_cuda_batch.py --graph data/acquisitions/local-windows-20260910/bundle --bindings data/acquisitions/local-windows-20260910/bindings-multimodal-v2.json --worlds 2 --seconds .05 --out verification/my-batch
```

모방학습 `--resume <policy.zip>`는 optimizer를 포함한 PPO 정책 학습을 이어 간다. 환경의 동일 순간을 재개하는 기능과는 다르다. `--data`, `--train-clip`, `--eval-clip`으로 외부 clip을 지정할 수 있으며 내용 hash를 남긴다. 다른 clip 이름만으로 생물학적 held-out 검증을 주장하지 않는다.

기준 구현: [FlyGym 저장소](https://github.com/NeLy-EPFL/flygym), [공식 GPU 시뮬레이션](https://neuromechfly.org/tutorials/3_gpu_accelerated_simulation/), [FlyBody 사용](https://neuromechfly.org/tutorials/5b_using_flybody_model/), [근육 모방학습](https://neuromechfly.org/tutorials/6_muscle_imitation/), [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/).

## 화면과 회귀 검사

화면에서 실제 FAFB multimodal 2세계 배치를 만들고, 세계 0의 8채널 기록·신경 개입·GPU 메시 표시를 실행했다. 세계 1을 10 ms에서 정지시킨 동안 세계 0은 15 ms로 진행했다. 재개·취소 후 저장한 `batch-d16024093c0d`를 화면에서 복원했고 세계 0은 20 ms, 취소된 세계 1은 10 ms를 유지했다. 기록은 한 chunk, 누락 0으로 완료했다. 전뇌 2세계 512채널 검증은 별도 CLI 결과다.

연속 실제 메시 표시를 켜고 재생하여 모델 시각이 진행하면서 영상 시각이 갱신되는 것을 확인했다. 관측 FPS는 상한이며 3/10 FPS 실시간 성능을 보장하지 않는다. 검증 뒤에는 원래 `C_STRICT` / `bilateral-geosmin-v1` / 신경 CUDA / 14.750 s 체크포인트 `checkpoint-02658835728c`로 복원했다.

전체 회귀: **300개 중 272 통과, 28 환경 조건 검사 생략**. 별도로 새 감각·근육·CUDA 배치의 **7개 검사 모두 통과**했다. 이 중 일반 검사 3개는 전체 회귀에도 포함된다. FlyBody 분절 수정 후 관련 감각 9검사는 6통과/3 native 조건 생략이며, 실제 FlyBody 감각은 별도 native 실행으로 확인했다. `build_c.py`, JavaScript 구문 검사와 변경 diff 검사를 통과했다. 큰 원자료의 파일 hash와 최종 UI 상태 비교는 동반 JSON 증거 목록에 기록한다.

최종 저장은 `checkpoint-b29cb0a60ed1`이다. 이전 저장과의 차이는 새 body 옵션 메타데이터, CPU control backend 명시, 화면 재구독 epoch뿐이며 신경·몸·감각·필터 상태는 같다. CUDA PPO 정책은 64 step 추가 학습하여 총 192 step까지 이어학습했다. [증거 목록](C_FLYGYM_REMAINING_EVIDENCE_20260910.json)에 버전, 보고서 SHA-256, 검사 수, UI 확인 결과를 기록했다.
