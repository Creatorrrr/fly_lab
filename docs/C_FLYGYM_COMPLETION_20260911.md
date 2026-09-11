# FlyGym 후속 반영과 실제 검증 — 2026-09-11

대상은 `C_FLYGYM_OPEN_ITEMS_20260910.md`의 E1/E2, V1–V4, M1–M5 및 검증 범위다. 코드 오류, 자료 도입, 실행 도구와 검증을 추가했다. **Warp의 정밀도·실시간성, BANC의 자발 보행, 전신 근육의 생물학적 재현이 해결됐다는 뜻은 아니다.** 통과하지 못한 기준은 유지했다.

기본 실행은 CUDA 신경 + CPU 물리다. 원래 실행 중이던 C_STRICT 실험은 14.750초에서 보존했다. 업데이트 전후 체크포인트의 신경·몸·encoder·sensor·세계·설정이 정확히 같았다. 새 감각 프로파일은 새 실험의 선택 목록에 추가했다.

## 반영 상태

| 항목 | 이번 반영 | 확인 결과 / 남은 한계 |
|---|---|---|
| E1 실패한 배치 재실행 | 실패 즉시 활성 세계와 실행 자원을 분리하고 추가 계산을 차단. 준비 단계에서 모든 세계의 감각·encoder·근육 상태를 준비한 뒤 일괄 확정하며, 준비 실패 시 원복. 화면도 실패 상태를 재조회. | 예외 주입 후 재시도에서 몸·신경 시각이 진행하지 않는 회귀 검사 통과. |
| E2 일시정지 배치 복원 | 재구성 시 epoch 증가 여부를 분리해 빈 활성 목록에서도 저장한 이력 번호 보존. | 전 세계 일시정지 상태를 반복 저장·복원해 epoch 유지 확인. |
| V1 Warp 수치·복원 | 접촉 레코드 정렬·실제 솔버 옵션 대조 진단, CUDA 제어 입력의 원자적 검증, 복원 시 고착된 오류 상태 초기화, GPU 실행 예외의 몸 fault 반영. GPU가 바꾼 실제 옵션을 실행 메타데이터에 반영. | 접촉 정렬만으로 완전 재현성을 얻지 못했다. 기본 물리 전환 보류. qpos 0.001, qvel 0.1, 발 힘 0.15 BW 기준을 완화하지 않았다. |
| V2 처리율·화면 | 복안 변환 CUDA 커널, 감각 포트 일괄 계산, 연속 관측의 요청 굶주림 수정, 관측 중복 판정에 정수 제어 시각 사용. | 구성 요소 최적화 및 60모델초/512채널/영상 검증. 전체 실시간 기준은 미달. |
| V3 BANC 보행 | 전뇌를 유지한 추가 감각→운동 모집 진단과 기계 제어기·지형 비교 경로 확장. | 기존 8조건 BANC 보행 실패를 성공으로 재분류하지 않았다. 추가 신경 진단에서도 해당 두 운동뉴런의 발화는 0. |
| V4 외력 회복 | 같은 2-MTU 관절 기계 교사의 이득을 훈련 조건에서 선택하고 다른 각도·외력에서 대조. | 교사의 회복 오차는 개선. BANC 정상·감각 차단 조건의 회복 차이는 여전히 기준 미달. |
| M1 근육 제어 | 공식 15-MTU LF 모델에 작은 접촉 플랫폼 추가. 지면 접촉·힘·복원 검사, 시간 구간 분리 PPO 학습·평가 및 수동/교사/미학습/학습 정책의 같은 조건 비교. | 접촉과 학습 실행은 확인. 6다리의 근육·부착점·관절 매핑 자료, 독립 개체/전신 이동 검증과 BANC 연결은 미완료. |
| M2 복안 연결 | FAFB v783 열 주석과 연결 수로 R1–6의 열을 추정하여 1,040개 낱눈 포트 생성. 공식 광학 변환의 CPU/CUDA 일치와 미래 상태 복원 검증. | 6,678개 R1–6에 연결. 모호하거나 매핑할 수 없는 1,778개는 외부 입력에서 제외하되 신경 계산에서는 유지. 796개 해부학적 열↔721개 모델 낱눈의 광학 등록은 미교정. |
| M3 화학 후각 | DoOR 실측 정규화 반응·SFR를 사용한 41수용기/82양측 포트. 물질 InChIKey를 환경 편집·저장·복원에 연결. | 더듬이 36종, palp 5종. 물질 미지정 냄새원은 수용기 자극에서 제외하고 화면에 표시. 절대 농도·혼합물 상호작용·신경 이득·palp 좌표는 미교정. |
| M4 CUDA 캠페인 | 모델 구성별 작업 큐, 실제 컴파일 모델 해시 일치 검사, 저장된 진행 저널, 중지·재개·세계 취소, 메모리 부족 시 세계 수를 절반으로 줄여 재시도. | 16 KiB 한도의 독립 CuPy 풀에서 실제 할당 실패→2세계에서 1세계로 축소→모든 작업 완료 확인. 장치 전체 VRAM을 소진한 검사는 아니다. 기존 단일 실험 비교 캠페인 API와 별도 모듈/CLI로 제공한다. |
| M5 비행·이착륙 | 고정 커밋의 공식 FlyBody `Flying`·날개 공력·다리·지면 접촉을 이용한 세 작업. 유효한 행동 범위, 시간 종료, 자세·접촉 기반 성공 판정 추가. | 3작업 각각 0.3초/1,500제어 실행 완료. 초기 날갯짓은 공식 prototype이며 세 작업의 성공 판정은 모두 false. 학습된 비행 정책이나 BANC 조종은 제공된 것으로 간주하지 않는다. |

## 오류 처리와 추가로 찾은 결함

- 4×MSAA 복안 영상은 같은 몸 상태를 반복 렌더링할 때 일부 픽셀이 1단계 달라졌다. 이 차이가 실제 전뇌의 미래 상태 불일치를 만들었다. 복안 전용 모델을 단일 샘플로 바꾸고 광학 계약을 `official-fisheye-single-sample-v2`로 명시했다. 공식 fisheye 변환과 낱눈 집계 방식은 유지했다.
- 자동 복안 체크포인트는 `flylab.retinal-input.v2`다. 이전 v1 광학 체크포인트는 명시적으로 거부한다. 자동 복안이 없는 기존 사용자 실험은 복원 가능하며 실제로 확인했다.
- 연속 관측이 계산 요청의 짧은 유휴 구간만 기다리던 조건을 제거했다. 관측 요청은 한 번에 하나이며 서버의 직렬 실행 경계에서 처리된다. 긴 실행 후 누적 부동소수점 시간 오차로 같은 화면을 다시 요청하던 조건도 정수 `control_tick` 비교로 교체했다.
- 냄새 물질을 되돌렸는데 편집 선택란이 이전 물질을 유지하던 표시 오류를 수정했다. 실제 UI에서 지정→적용→되돌리기를 확인했다.
- Windows 기본 cp949 환경에서 기본 그래프와 HTML 빌드가 실패하던 읽기/쓰기를 UTF-8로 명시했다.
- FlyBody의 기본 `WalkerPose.qpos=None`을 숫자로 대입하면 NaN이 되던 초기화 경로를 피했다. 부동소수점 시간 때문에 마지막 제어에서 종료 신호가 늦어지는 문제도 수정했다. 마지막 정수 제어 경계에서 종료하도록 Composer 시간 한계를 마지막 두 경계 사이에 둔다.
- 경사면을 내려갈 때 정상적인 몸도 세계 좌표의 z가 음수가 될 수 있다. 이를 낙하로 오인하던 판정을 지면 법선에 대한 높이·몸의 기울기로 수정했다. CPU, Warp, resident, 배치, 기계 대조와 CNS 관절 구동이 같은 판정을 사용한다. 경사면 모델 해시에 `slope-plane-v1`을 포함해 이전 판정의 경사면 체크포인트와 구분한다. 평면 모델의 기존 해시는 유지한다.
- Warp upstream은 float32 계산을 위해 tolerance를 최소 1e-6으로 바꾸고 ISLAND를 비활성화한다. 기존 메타데이터는 CPU 거울 모델의 1e-8과 플래그를 출력했다. 이제 실제 GPU 옵션을 저장해 표시하고, 요청된 CPU 값은 `cpu_mirror_options`로 남긴다. 매 프레임 GPU에서 값을 다시 읽지 않는다.

## 수치와 검증 해석

**감각 계산:** 1,122개 포트의 held-drive 계산을 기존 순서대로 한 번에 합산한다. 겹치는 뉴런의 float32 덧셈 순서와 float64 필터 계산을 유지한다. 실제 그래프 대상과 합성 감각 값으로 필터·지연·차단·시간 간격을 바꿔 대조한 결과가 정확히 같았다. 포트 계산 중앙값은 6.493 ms → 1.483 ms, 약 **4.38배**였다. 렌더링·신경·물리·기록을 제외한 수치다.

복안의 공식 광학 변환+낱눈 집계는 양쪽 눈당 4.615 ms → CUDA 0.295 ms, 약 **15.65배**였다. 입력/출력 전송을 포함하며 네이티브 영상 렌더링은 제외한다. CPU에서는 더 느렸던 NumPy gather 대신 공식 컴파일 변환으로 돌아간다. CPU fallback은 같은 측정에서 4.853 ms였다.

**최종 감각 프로파일:** 139,255개 뉴런을 모두 계산하는 `exp_lif_cuda`에서 영상 입력, 몸, 신경, encoder의 체크포인트 이후 미래가 정확히 같았다. 감각 입력 전체 차단 시 외부 drive 합계는 0이었다. 이는 생물학적 시각·후각 행동의 검증과 다르다.

DoOR의 Or49a/Or85f와 Or33c/Or85e는 각각 같은 주석 뉴런 집단으로 맵핑됐다. 이를 별도 세포처럼 합산하면 자발 발화와 공통 전달이 중복된다. 공동 반응 교정 자료가 없어 네 수용기의 포트를 제외하고 사유를 기록했다. 최종 프로파일은 `bindings-evidence-sensory-v4.json`이며 이전 생성본 v3는 검증 폴더로 옮겼다. 관측값이 없는 8개 수용기-물질 조합은 보간해 실측값으로 취급하지 않는다.

**CUDA 전뇌 배치:** 32세계 × 139,255뉴런 × 512기록채널로 세계당 0.5모델초를 완료했다. 44.664실제초, 제어 구간 중앙값 0.436초였다. 전체 장치의 표본 VRAM 최대는 5.410 GB이며 다른 프로그램의 메모리도 포함한다. 저장 후 32세계 모두 0.05모델초를 추가 실행하여 tick 5500에 도달했다. 이 재개 검사는 무중단 Warp 궤적과 완전 일치한다는 판정이 아니다.

솔버 옵션을 맞춘 시드 42의 20 ms 추가 대조에서도 CPU↔resident 최대 차이는 qpos 0.02907, qvel 5.6201, 발 힘 0.16824 BW로 기존 기준을 넘었다. CPU의 tolerance·ISLAND 옵션을 GPU와 맞춰도 이 최대값이 바뀌지 않았다. 같은 Warp 상태의 미래 완전 일치도 접촉 정렬 여부 모두 실패했다. 따라서 설정 메타데이터 오류는 수정했지만 수치 불일치의 원인 전체가 해결된 것은 아니다 (`warp-effective-02/report.json`).

**기계 교사:** 훈련 각도 1.3/2.1 rad, 외력 ±0.008에서 검토한 뒤 kp=3600, kd=120을 선택했다. 기존 값은 900/60이었다. 다른 각도 1.4/2.0 rad, 외력 ±0.01의 네 검증 조건에서 외력 후 누적 오차가 수동 대비 **64–68% 감소**했다. 같은 24조건 대조에서 BANC 감각 피드백의 회복 기준은 통과하지 못했다. 이 교사 변경은 일반 보행 엔진의 신경 모델을 바꾸지 않는다.

추가 8조건 BANC 진단에서는 전압 사건 적분 가설, 두 각도, 이득 18/36, 감각 출력 차단 여부를 대조했다. 감각을 켜면 신전 운동뉴런에 역치 이하 반응이 전달됐고 차단 시 사라졌다. 두 운동뉴런의 발화 수는 모든 조건에서 0이었다. 말초 감각 신호가 도달하는 것과 운동 출력을 모집하는 것은 별개의 결과다.

**15-MTU 학습·접촉:** 공식 225프레임 clip에서 train [0,135), validation [180,225)를 사용했다. 20% 시간 간격을 두었지만 같은 개체·trial이며 독립 생물학적 표본이 아니다. CUDA PPO 32,768 step과 44제어의 평가를 완료했다. 같은 평가 도구의 CPU 정책 추론 대조에서 평균 보상은 수동 0.1621, 미학습 0.1605, 학습 0.1806이었다. 그러나 평균 관절 RMSE는 수동 0.1455 rad, 학습 0.1984 rad로 나빠졌다. 수렴이나 더 정확한 움직임을 입증하지 못했다.

LF 접촉 장치는 측정 clip의 첫 발 위치 아래에 작은 고정 플랫폼을 추가한다. 몸·관절·MTU 부착점을 바꾸지 않는다. 224제어의 수동/교사 실행에서 각각 20/44회의 실제 접촉을 관측했다. 고정된 한 다리의 접촉 실험이며 6다리 지면 보행이 아니다. 여기의 15-MTU 모방 교사와 위의 개선된 2-MTU 외력 회복 교사는 별개다.

**지형 보행:** 두 몸 × 다섯 지형 × hybrid/CPG/rule × 시드 42/43의 60조건을 각 3모델초까지 검사했다. 수정 전 경사면 12조건이 모두 정지했고, 해당 12조건을 다시 실행하자 모두 3초를 완료했다. 나머지 48조건은 판정식이 같으므로 기존 원자료와 합쳐 **47완료 / 13정지**다. 정지는 NeuroMechFly의 평면 3·틈 4·블록 1·혼합 3조건과 FlyBody의 틈 1·혼합 1조건이다. 멈춘 조건을 정상 보행으로 재분류하지 않았다. 모든 지형에서의 안정 보행이나 장시간 성공을 입증하지 못했다.

CUDA resident와 1세계 배치도 각각 0.7모델초 경사 보행을 완료했다. 세계 z의 최솟값은 -0.562/-0.586 mm였지만 최종 지면 높이는 1.312/1.311 mm였고 fault가 없었다. 실제 GPU 물리 경로에 수정한 판정이 적용되는지 확인한 결과다.

## 재현 명령

Windows 저장소 루트에서 실행한다. 기존 결과 디렉터리가 있으면 새 이름을 사용한다. CUDA 신경과 Retina에는 `requirements-cuda.txt`, Warp 배치에는 `requirements-warp.txt`, 근육 학습에는 기존 연구 의존성이 필요하다.

```powershell
.venv\Scripts\python.exe tools/acquire_c_sensory_data.py --out data/acquisitions/sensory-20260910
.venv\Scripts\python.exe tools/build_c_evidence_sensory_profile.py --graph data/acquisitions/local-windows-20260910/bundle --base data/acquisitions/local-windows-20260910/bindings-multimodal-v2.json --assets data/acquisitions/sensory-20260910 --columns --out data/acquisitions/local-windows-20260910/bindings-evidence-sensory-v4.json
.venv\Scripts\python.exe tools/verify_c_sensory_evidence.py --graph data/acquisitions/local-windows-20260910/bundle --bindings data/acquisitions/local-windows-20260910/bindings-evidence-sensory-v4.json --out verification/sensory-recheck
.venv\Scripts\python.exe tools/run_c_cuda_campaign.py --help
.venv\Scripts\python.exe tools/evaluate_c_muscle_policy.py --contact-platform --out verification/contact-recheck
.venv\Scripts\python.exe tools/run_c_locomotion_matrix.py --seconds 3 --seeds 42 43 --workers 4 --out verification/terrain-recheck
.venv\Scripts\python.exe tools/diagnose_c_warp_options.py --compare-effective --out verification/warp-solver-recheck
.venv\Scripts\python.exe -m pip install -r requirements-flight.txt
.venv\Scripts\python.exe tools/acquire_c_flight_source.py --out data/acquisitions/flybody-tasks
.venv\Scripts\python.exe tools/run_c_flight_tasks.py --source data/acquisitions/flybody-tasks --seconds .3 --out verification/flight-recheck
```

CUDA 캠페인 spec 예:

```json
{"schema":"flylab.cuda-campaign.v1","seconds":0.05,"max_worlds":4,"record_channels":512,"jobs":[{"id":"flat-42","seed":42,"mode":"C_SHADOW"},{"id":"flat-43","seed":43,"mode":"C_SHADOW"},{"id":"slope-42","seed":42,"mode":"C_SHADOW","body_options":{"terrain":"slope"}}]}
```

`run_c_cuda_campaign.py`의 `--spec`에 위 파일을 지정한다. `--max-controls`로 정상 경계에서 저장하고 `--resume`으로 이어갈 수 있다. 출력 디렉터리의 `cancel.json`에는 취소할 job id 배열을 둔다. 외부 API 예외는 OOM으로 재분류하지 않으며 실패한 캠페인을 정상 재개로 위장하지 않는다.

## 근거와 미충족 조건

원자료는 `verification/flygym-completion-20260910/`에 있다. 주요 결과는 `evidence-sensory-v4-05/report.json`, `encoder-vector-01/report.json`, `retina-compute-02/report.json`, `batch32-record512-01/report.json`, `batch32-restore512-02/report.json`, `joint-holdout-01/report.json`, `joint-voltage-events-01/report.json`, `imitation-matched-01/report.json`, `muscle-contact-01/report.json`, `flight-tasks-01/report.json`이다. 최신 브라우저·회귀·지형 결과 및 파일 해시는 아래 실행 결과 절에 기록한다.

자료 출처는 [FlyGym](https://github.com/NeLy-EPFL/flygym), [공식 근육 모델 범위에 관한 이슈](https://github.com/NeLy-EPFL/flygym/issues/276), [DoOR.data](https://github.com/ropensci/DoOR.data), [OpticLobe.jl](https://github.com/hsseung/OpticLobe.jl), [FAFB v783 column assignment](https://storage.googleapis.com/flywire-data/codex/data/fafb/783/column_assignment.csv.gz), [FlyBody](https://github.com/TuragaLab/flybody)다. 11개 감각 자료의 URL·커밋·SHA-256은 `C_SENSORY_ASSETS_20260910.json`에 고정했다. FlyBody 코드는 `d015e9bfe441bd90ae431bac24c55cb74bdbce26`을 사용한다.

남은 한계는 (1) Warp 물리의 채택 정밀도·미래 궤적 재현성, (2) 전체 실시간 처리율, (3) BANC 감각→운동 모집과 자발 보행, (4) 6다리 근육·전신 이동 실측 자료, (5) retinotopy 광학 등록·palp 위치·화학 이득 및 혼합물 교정, (6) motion/loom 특징을 별도 직접 주입할 검토된 매핑, (7) 학습된 비행·이착륙 정책이다. Windows 장치에서는 실제 MPS 검증을 할 수 없다. 누락된 자료를 임의 생성하거나 기계 교사/CPG의 성공을 BANC의 성공으로 바꾸지 않았다.

## 최종 실행 결과

전체 native/CUDA/비행 환경 회귀: **317개 중 296 통과, 21 생략**, 82.859초 (`regression-native-08.log`). 이후 GPU 옵션 메타데이터 변경과 관련된 native/CUDA/캠페인 21개 검사를 모두 통과했다 (`warp-metadata-09.log`). 브라우저 playback 단위 검사도 6개 모두 통과했다. MPS 등 생략 사유는 로그에 있으며, 생략을 통과로 집계하지 않았다.

최종 v4 프로파일의 실제 브라우저 실행은 **60모델초 / 544.831실제초 = 0.1101×**였다. 139,255뉴런 CUDA, CPU 물리, CUDA 복안 100 Hz, 512채널 관측·기록, 네이티브 영상 최대 3 FPS를 함께 켰다. 시작 0.050초에서 종료 60.050초까지 12,000개 제어 시각을 진행하고 영상 1,367회를 갱신했다. 마지막 표시 빈도는 2.7 FPS, 서버가 계산 구간만 집계한 SIM/WALL은 0.178×다. **전체 경과 시간 기준 실시간 목표 ≥1은 미달**이다. 이전 v3 실행과는 프로파일·시작 상태가 달라 전체 성능의 A/B 배수로 비교하지 않는다.

512채널 기록은 6,000행으로 정상 종료했다. 브라우저 콘솔 오류·경고는 0개였다. 정지 후 연속 관측을 켜고 2.5초 관찰했을 때 이미지 추가 갱신은 0회이고 모델 시각도 그대로였다. 배치 최초/재개와 UI 기록을 합친 **67개 기록 / 14,976행**에 대해 파일 해시, 512개 고유 채널, 유한 신호, 일정한 tick 간격, 정상 종료와 유실 0을 확인했다 (`recording-integrity.json`).

최종 실행·원자료 연결은 `endurance-v4.json`, `locomotion-final.json`, `slope-cuda-01.json`, `warp-effective-02/report.json`, `live-final-03.json`에 남겼다. 기존 실험은 여전히 C_STRICT, CUDA, 14.750초다. 재시작 전후 신경·몸·encoder·감각·세계·설정을 포함한 상태가 같으며, 화면 재접속에 따른 `subscription_epoch`만 달라졌다. 최종 자료와 변경 파일의 SHA-256 목록은 `completion-evidence.json`에 있다.
