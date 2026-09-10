# FlyGym 개선안 구현과 Windows 실행 결과

2026-09-10 · Windows / NVIDIA RTX 4080 16 GB · 기존 CUDA 변경 위에 적용

**물리 백엔드 분리, 상세 계측, 같은 관절의 기계 교사, 실제 복안·4지점 후각 관측과 검증 도구를 구현했다. 기본 실행은 CUDA 신경 + CPU 물리다.** Warp 후보와 BANC 행동 후보는 실제 실행했지만 채택 기준을 통과하지 못했다. 해당 실패를 해소하지 않은 상태에서 전신 근육이나 전뇌 배치로 확장하지 않았다.

기준 계획은 [FlyGym 공식 구현 기반 개선안](C:/Users/ckthd/Dev/Projects/fly_lab/docs/C_FLYGYM_IMPROVEMENT_PLAN_WINDOWS_20260910.md)이다. 계획 문서는 당시 조사 이력을 보존했다. 이 문서의 수치는 이번 구현 후 실제 실행에서 얻었다. 원자료는 `C:/Users/ckthd/Dev/Projects/fly_lab/verification/flygym-implementation-20260910/`, 파일 해시와 요약은 [검증 근거 JSON](C:/Users/ckthd/Dev/Projects/fly_lab/docs/C_FLYGYM_IMPLEMENTATION_EVIDENCE_20260910.json)에 있다.

| 계획 항목 | 이번에 적용·실행한 범위 | 채택 및 남은 조건 |
|---|---|---|
| 1. CPU 몸 계약·프로파일러 | 물리 프로파일, step/forward 경계, 상태 출처·옵션 기록, 준비/정상 실행/상세 계측/입력 재생 분리 | 기본 CPU 결과 보존 확인. 장시간·5시드 종합 성능 검증은 별도 |
| 2. Warp 1세계 | 실제 CUDA 물리 어댑터, CPU 관측 동기화, 환경 변경, 장치 상태 저장·복원, A/B/C 비교 | 첫 정합성 검사 실패. 실험 경로로만 제공 |
| 3. 기계·신경 대조 | 공식 42 DOF 운동학 로더, 같은 2 MTU/1 DOF의 교사, 실제 전체 BANC 24조건 | 목표각 유지 가능. 신경 피드백의 외력 회복 개선은 실패 |
| 4. 합산·접착 대조 | 4구성 × 감각/직접 자극, 각 10모델초; 실제 발 높이·접촉·미끄럼 추가 기록 | 8조건 모두 보행 기준 실패. 15 MTU/한 다리/전신 확장 보류 |
| 5. 감각 | 공식 Retina 관측, 더듬이 2곳·palp 가정 위치 2곳, 별도 후각 프로파일, UI | 더듬이 입력·복원·차단 확인. Retina/palp 신경 ID 매핑과 행동 검증은 미완료 |
| 6. 배치·메시 | 실제 메시 관측 UI, Warp 물리만 1/8/16/32세계 평가 | 전뇌 배치 스케줄러·캠페인 취소/재개·공유 topology는 미구현 |

사용한 FlyGym은 기존 2.1.0 pin을 유지했다. 현재 설치본과 공식 소스의 구현 일치 확인은 앞선 계획에 기록되어 있다. 추가 설치는 `warp-lang==1.14.0`, `mujoco-warp==3.9.0`, BANC 취득용 `pyarrow==23.0.1`이다. MuJoCo 3.9.0, CuPy 14.2.0, PyTorch 2.14.0+cu130을 교체하지 않았다. 실측 드라이버는 591.86이며 설치된 Warp wheel은 CUDA Toolkit 12.9 / 드라이버 API 13.1을 보고했다.

**물리 계약과 계측.** [physics.py](C:/Users/ckthd/Dev/Projects/fly_lab/flylab/physics.py)는 신경 `--backend`와 별도로 `--physics-backend cpu|warp`, noslip, MULTICCD, 버퍼 용량과 CUDA Graph 설정을 관리한다. 기본 프로파일의 기존 모델 해시·체크포인트 호환을 보존하고, 다른 물리 설정은 별도 정체성으로 기록한다. [body.py](C:/Users/ckthd/Dev/Projects/fly_lab/flylab/body.py)는 물리 적분/forward와 관측의 경계를 제공한다.

변경 전 소스 155개를 별도 보존한 뒤 같은 CUDA 체크포인트에서 20제어 구간, 100 ms를 비교했다. 최종 코드에서도 tick·몸·물리·감각·명령·운동 발화율 trace가 정확히 일치했다. 두 trace의 SHA-256은 `959fdb32509bcd7c8c4af0ba5b2130d1780a8d7d94faffdcd9cfec01c1ed7336`이다. 이는 해당 구간의 회귀 확인이며 장시간 행동 정확도 증거는 아니다.

[profile_c_runtime.py](C:/Users/ckthd/Dev/Projects/fly_lab/tools/profile_c_runtime.py)는 구성/settling/JIT/warm-up과 정상 실행을 분리한다. 상세 실행에는 cProfile, MuJoCo timer, CUDA stream event와 NVTX 제출 범위를 기록한다. 별도로 기록된 actuator/외력 입력과 신경 입력을 재생해 구성요소를 측정한다. Nsight 전체 커널·복사 타임라인 수집이나 모든 작업의 비용 분해를 완료했다는 뜻은 아니다.

50 ms 정상 실행 × 3회, 별도 20 ms warm-up에서 전체 wall 중앙값은 0.116709 s, sim/wall은 0.4284였다. 분리한 물리 입력 재생 0.051016 s와 신경 입력 재생 0.071789 s는 모두 목표 상태와 정확히 일치했다. 두 시간을 더해 전체 비용으로 계산하지 않는다. 상세 계측의 추가 비용은 정상 처리율과 분리되어 있으며, 이번 변경의 전체 속도 향상이나 60모델초 실시간 달성을 주장하지 않는다. [프로파일 원자료](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flygym-implementation-20260910/profile-01/report.json).

**Warp 평가와 기본값 유지 이유.** [body_warp.py](C:/Users/ckthd/Dev/Projects/fly_lab/flylab/body_warp.py)는 기존 0.1 ms 반사 제어를 CPU에서 유지하고 매 물리 tick마다 Warp 상태를 CPU 관측에 동기화하는 최초 후보다. 초기 neutral settling만 조건별 CPU 옵션으로 수행한다. 따라서 아직 CPU 왕복을 없앤 최적화된 GPU 제어기는 아니다. 구조는 [공식 GPU 예제](https://neuromechfly.org/tutorials/3_gpu_accelerated_simulation/)와 [MuJoCo Warp API](https://mujoco.readthedocs.io/en/stable/mjwarp/api.html)를 참고했다.

이 arena의 contact margin과 MULTICCD 조합은 Warp 3.9에서 거절되었다. margin을 그대로 유지하고 B/C에서 MULTICCD를 함께 끄는 명시적 비교로 수정했다. **A↔B 차이는 noslip과 MULTICCD의 복합 변경 효과이며 noslip 단독 효과가 아니다.**

| 조건 | 물리 | noslip / MULTICCD | 20 ms 실행 × 3회 wall 중앙값 |
|---|---|---|---:|
| A | 기존 CPU | 5 / 켬 | 0.037750 s |
| B | 비교용 CPU | 0 / 끔 | 0.033932 s |
| C | Warp 동기화 어댑터 | 0 / 끔 | 1.110646 s |

seed 42의 첫 정합성 비교에서 사전 기준 `qpos 0.001`, `qvel 0.1`, 발 힘 `0.15 BW`를 넘었다. B↔C 최대 차이는 각각 0.030388, 5.618988, 0.168241 BW였다. qpos/qvel은 모델 일반화 좌표의 최대 원소 차이이므로 단일 rad/mm 단위 오차로 해석하지 않는다. 같은 Warp 백엔드에서 저장 후 미래 궤적의 엄격한 일치도 실패했다. GPU 반복 자체에서도 차이가 관측되었다. 접촉 API 일치, 논리 시계, 환경 변경 후 step, 잘못된 상태 거절 검사는 통과했다.

MuJoCo 3.9의 `contact.geom`과 별도 `geom1/geom2` 필드 간 복사 누락을 어댑터에서 보완했다. 접촉·제약 버퍼 넘침은 upstream 복사가 자르기 전에 검사한다. 모든 device 배열을 저장해도 미래 궤적의 동일성이 보장되지 않는 현재 한계를 기록했다. 기본 CPU 물리의 옵션을 바꾸지 않았고, 5시드·전체 폐루프·기록·UI를 포함한 채택 성능 실험으로 확대하지 않았다. [정합성 실패 원자료](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flygym-implementation-20260910/warp-verify-01/report.json).

**같은 단일 관절에서의 기계 교사와 BANC.** [joint_teacher.py](C:/Users/ckthd/Dev/Projects/fly_lab/flylab/c/joint_teacher.py)는 같은 두 Hill 근육에 제한된 활성도를 출력하는 목표각 추적 제어기다. 기존 10 µs 물리 dt와 모멘트암 반전 감시를 유지한다. 공식 `MotionSnippet`의 42관절 이름·좌우 좌표 변환은 [kinematics.py](C:/Users/ckthd/Dev/Projects/fly_lab/flylab/kinematics.py)에 연결했다. 이 로더가 기록 운동학을 전신 근육으로 모방 실행한 것은 아니다. clip은 교정/학습 자료로 표시한다.

공개 BANC 888 v2 원자료를 기존 release manifest의 URL·SHA-256으로 확인해 취득하고, 뉴런 158,706개·쌍 연결 11,584,852개·해부학적 시냅스 35,457,154개의 묶음을 재구성했다. 선언된 roster 범위의 계산이며 전체 CNS 완전성이나 생물학적 검증을 뜻하지 않는다. 기존 live FAFB 데이터를 교체하지 않았다.

[compare_c_joint_teacher.py](C:/Users/ckthd/Dev/Projects/fly_lab/tools/compare_c_joint_teacher.py)로 초기각 1.4/2.0 rad × 외력 0/±0.01 모델 토크 단위 × 수동/교사/정상 신경/감각 차단을 실행했다. 조건당 0.3모델초, 외력은 50–80 ms, 회복 평가는 80–300 ms다. 24조건 모두 오류 없이 완료했다.

| 초기각 | 무외력 수동 최종각 | 무외력 교사 최종각 |
|---:|---:|---:|
| 1.4 rad | 1.610194 rad | 1.400213 rad |
| 2.0 rad | 1.611571 rad | 1.998406 rad |

교사가 목표각을 유지하는 것은 확인했지만, 같은 무외력 궤적을 뺀 외력 후 IAE는 수동 대비 약 1.56–1.70배로 더 컸다. **교사의 외력 회복 개선도 입증하지 못했다.** 정상 신경 조건에서는 감각 입력이 최대 10.297 mV였으나 해당 두 근육의 요청 활성도는 모두 0이었다. 정상/차단/수동의 외력 편차와 IAE가 같아 피드백 IAE 20% 감소 기준을 통과하지 못했다. 원인 조사 범위는 이 설정에서의 감각→운동단위 모집과 기계 교사의 외력 반응이며, 몸의 원리적 제어 불가능성을 뜻하지 않는다. 이번 자세를 이후 최종 확인용 미사용 자료로 재사용하지 않는다. [24조건 보고서](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flygym-implementation-20260910/joint-teacher/report.json).

**보행 가설과 실제 발 측정.** 기존 v2, 새 운동단위 합산, 발 들림에 따른 접착 해제, 두 변경의 조합을 감각 입력/양측 DNg100 직접 자극으로 대조했다. 기존 10초·순이동 5 mm·부호 있는 전방 이동 5 mm·고착 없음 기준을 그대로 사용했다. CPG를 대신 조정하지 않았다. 실제 tip 좌표/속도, 바닥 접촉과 5 ms 표본의 접촉 지속시간·미끄럼 적분을 추가했다. 연속 접촉의 완전한 기록이나 Jacobian 예측 들림과 동일한 값으로 해석하지 않는다.

| 구성 | 감각: 순이동 / 전방 mm | 직접 자극: 순이동 / 전방 mm |
|---|---:|---:|
| 기존 v2 | 0.0678 / 0.0669 | 0.4383 / 0.4174 |
| 운동단위 합산 | 0.0813 / 0.0785 | 0.3299 / 0.2540 |
| 접착 해제 | 0.0678 / 0.0669 | 0.4370 / 0.3881 |
| 조합 | 0.0813 / 0.0785 | 0.3418 / 0.3058 |

8조건, 총 80모델초를 완료했지만 모두 고착이 검출되었고 보행 기준은 실패했다. 결과는 `REJECTED_ALL_CANDIDATES`다. 기존 후보 v3의 `EXPERIMENTAL_NOT_VALIDATED` 상태와 기본 바인딩을 유지했다. 따라서 틈·블록·경사면, 한 다리 근육 접촉 실험과 전신 확장은 보류했다. [간결한 보행 결과](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flygym-implementation-20260910/banc-gait/compact-report.json), [전체 원자료 보고서](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flygym-implementation-20260910/banc-gait/report.json).

**새 감각과 사용자 화면.** [flygym_senses.py](C:/Users/ckthd/Dev/Projects/fly_lab/flylab/flygym_senses.py)는 공식 vision.yaml의 눈 위치·방향·화각과 `Retina.correct_fisheye`/`raw_image_to_hex_pxls`를 사용한다. 눈당 512×450 RGB와 721개 낱눈 × 2채널을 계산한다. 눈 렌더링은 private model/data를 사용해 원래 ray 제외 정책과 눈의 자기 가림을 분리한다. 자동 Retina 신경 입력이나 100/200 Hz 영상 입력 프로파일은 아직 구현하지 않았다.

후각은 좌우 antenna body 원점과 rostrum 상대 palp 가정 위치에서 음식/위험 성분을 각각 읽는다. Gaussian과 regularized inverse-square를 선택할 수 있다. 원농도와 압축 반응을 별도 저장하고 몸 회전, foodOff, 감각 필터 복원을 검사했다. palp 전용 해부학적 분절이 없으므로 그 두 위치는 교정 전 공학적 가정이다. 시간변화 plume은 미구현이다.

새 `bindings-four-site-odor-v1.json`은 기존 실제 ID·gain을 유지하고 두 더듬이 반응을 기존 후각 포트에 전달한다. 로컬 기본 바인딩이 평균 냄새 하나이므로 이 프로파일도 두 더듬이의 평균을 같은 68개 ORN에 연결한다. 네 지점을 네 신경 채널에 연결한 것으로 해석하면 안 된다. Retina와 palp는 관측 전용이다. 실제 FAFB/CUDA 139,255뉴런 실행에서 체크포인트 복원 후 50 ms의 신경·몸·감각 상태가 정확히 일치하고, `sensor_off` 시 포트 입력이 0이 되는 것을 확인했다. [프로파일 통합 검사](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flygym-implementation-20260910/four-site-integration/report.json).

화면에 **실제 메시 / 복안 관측 / 후각 4지점** 버튼과 물리 장치 표시를 추가했다. 관측은 재생을 멈추고 해당 모델 시각을 보여준다. 서버 재실행 후 실제 세 버튼을 눌렀으며, 2.295모델초에서 관측 전·후의 몸/신경/감각/필터/명령/예약/시계가 저장 파일 비교로 정확히 일치했다. 50 ms 한 단계 실행도 2.345초까지 정상 진행했다. 최종 체크포인트는 `checkpoint-aaf2c76ff2c5`다. 앱은 [로컬 FLY LAB](http://127.0.0.1:8766/)에 열려 있고, 현재 C_SHADOW의 기존 실험을 유지한다. [UI 상태 보존 검사](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flygym-implementation-20260910/ui-observation-state.json).

**물리 배치 한정 평가.** [benchmark_c_warp_batch.py](C:/Users/ckthd/Dev/Projects/fly_lab/tools/benchmark_c_warp_batch.py)는 같은 모델과 고정 중립 actuator 입력으로 물리 세계만 배치한다. 전뇌·감각 폐루프·기록·렌더링이 포함되지 않는다. 커널 초기화와 첫 CUDA Graph 실행을 측정에서 분리한 50 step × 3회 짧은 연속 구간에서 아래 값을 얻었다.

| 세계 수 | 전체 world-step/s 중앙값 |
|---:|---:|
| 1 | 1,762 |
| 8 | 12,759 |
| 16 | 23,178 |
| 32 | 40,892 |

8/16/32세계에서 한 세계에만 외력을 주는 검사에서 건드리지 않은 세계의 qpos 최대 차이는 1.20e-7 이하였다. 외력을 준 세계는 1.02e-4 변했다. 이는 허용치 1e-5의 짧은 물리 독립성 검사이며, 세계 간 간섭이 모든 조건에서 0이라는 증명은 아니다. 32세계의 전체 처리율은 세계당 실시간이나 32개 전뇌 실행 성능을 뜻하지 않는다. VRAM은 driver 할당량 차이만 기록했고 최고값은 측정하지 않았다. 최초 측정과 graph warm-up을 분리한 재측정 모두 보존했다. [재측정 원자료](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flygym-implementation-20260910/warp-batch-steady/report.json).

**검증 결과와 재현.** 전체 회귀는 293개 중 269개 통과, 24개 환경 조건 검사 건너뜀이다. 실제 CUDA 테스트는 포함하여 통과했다. 별도 native 감각/교사 포함 검사 6개도 통과했으며 그중 일반 검사 3개는 전체 회귀와 겹친다. `build_c.py`, `git diff --check`, 변경 전/최종 CPU trace 비교, 실제 UI 버튼·체크포인트 검사를 완료했다. Warp/행동 평가의 실패는 테스트 실행 성공과 별개로 그대로 보존했다.

첫 회귀에서는 Warp 초기화 로그가 doctor JSON stdout에 섞였고, `-X utf8`만 준 부모에서 실행한 하위 프로세스에 UTF-8 모드가 상속되지 않아 4건의 인코딩 실패가 발생했다. Warp 진단을 stderr로 보내도록 수정했다. 최종 전체 검사는 `PYTHONUTF8=1`을 상속하여 통과했다. MPS 장치 검사는 이 Windows PC에서 실행하지 않았다.

추가 경계 검사로 사용 불가능한 Warp를 명시하면 `--doctor`가 `BLOCKED`를 반환하는지 확인했다. Warp 버전 pin이 다른 경우도 사용 가능으로 표시하지 않는다. 후각 관측 RPC는 현재 감각 프로파일의 Gaussian/역제곱 설정을 사용하며, 두 필드에 대한 회귀 검사를 포함한다.

아래 명령은 프로젝트 루트의 PowerShell 기준이다. `verification/.../rerun-*` 출력 경로는 기존 결과가 없는 새 이름으로 지정한다. 실패한 검증의 기준을 바꾸거나 결과를 덮어쓰지 않는다.

```powershell
$env:PYTHONUTF8='1'
.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
$env:FLYLAB_NATIVE_TESTS='1'
.venv\Scripts\python.exe -X utf8 -m unittest tests.test_flygym_improvements -v
Remove-Item Env:FLYLAB_NATIVE_TESTS
.venv\Scripts\python.exe -X utf8 run_c.py --doctor
.venv\Scripts\python.exe -X utf8 tools/profile_c_runtime.py --checkpoint artifacts/c/checkpoints/checkpoint-d18e8cacfdad --seconds .05 --warmup-seconds .02 --out verification/flygym-implementation-20260910/rerun-profile
.venv\Scripts\python.exe -X utf8 tools/verify_c_warp.py --out verification/flygym-implementation-20260910/rerun-warp
.venv\Scripts\python.exe -X utf8 tools/benchmark_c_warp_batch.py --out verification/flygym-implementation-20260910/rerun-batch
.venv\Scripts\python.exe -X utf8 tools/compare_c_joint_teacher.py --graph data/acquisitions/banc888-windows-20260910/bundle --backend exp_lif_cuda --out verification/flygym-implementation-20260910/rerun-joint
.venv\Scripts\python.exe -X utf8 tools/compare_c_banc_gait.py --graph data/acquisitions/banc888-windows-20260910/bundle --baseline data/acquisitions/banc888-windows-20260910/bindings-neuromuscular-v2.json --protocol verification/flygym-implementation-20260910/gait-protocol.json --backend exp_lif_cuda --out verification/flygym-implementation-20260910/rerun-gait
```

현재 프로파일을 새 데이터 묶음에 만들 때는 `tools/build_c_flygym_sensor_profile.py --graph <bundle> --base <bindings.json> --out <new-bindings.json>`을 사용한다. 지원하는 `--field`는 `gaussian`과 `inverse-square`다. 출력은 새 파일만 생성한다. 선택 Warp 의존성 설치는 `.venv\Scripts\python.exe -m pip install -r requirements-warp.txt`, 명시적 후보 실행은 `launch_c.bat --physics-backend warp`다. 기존 CPU 체크포인트를 Warp 상태로 암묵 변환하지 않는다.

다음 확장에 필요한 것은 Warp의 반복·복원 차이 해소와 GPU 안의 반사/접촉 처리, 같은 관절에서의 감각 입력→운동단위 모집 검증, 교사의 외력 회복 개선, Retina/palp의 검토된 실제 ID 매핑이다. 이를 통과하기 전에는 15 MTU·전신 근육, 복잡 지형, 전뇌 배치와 장시간 실시간 성능을 완료로 표시하지 않는다.
