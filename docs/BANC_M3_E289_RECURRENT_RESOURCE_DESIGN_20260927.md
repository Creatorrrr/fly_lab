# E289/E290 — 기존 premotor 재귀회로의 활동 의존적 전달 자원

2026-09-27 root 설계. 자연 보행 M3의 F1–F8과 사용자 지정 개발 절차를 이어간다. E287의 유효 기각과 E288 미실행을 보존한다. 사용자 전체 시간/토큰 한도는 없고 원본 누적7869.382029599514/7925초, 예약0이다. 본 문서는 구현·생성검증 설계이며 원본 GPU/신체 실행 예약이 아니다.

## 방향과 반증 가능한 가설

E287은 고정 E277 source 응답과 강화한 startup/envelope 아래 전체 monotone decoder를 한 번 적합했으나 후보를 만들지 못했다. 그 저장 dual은 이 한정된 affine 가족의 제약 충돌을 지지한다. E193/E196의 감각 전달 변화에도 여섯 다리의 eligible 완전주기가0이었고, E178은 강한 지속 억제를 보였으나 한 억제항 제외만으로는 충분하지 않았다. 단순 근육 전달·정적 입력 보정으로 보행 협응이 회복된다는 근거는 없다.

현재 r/z 모델의 subtractive postsynaptic adaptation과 E267의 비음수 선형 source 이력 혼합은 내부 premotor 재귀 간선의 전달 자원을 바꾸지 않는다. 새 가설은 실제 presynaptic 활동에 따른 재귀 전달의 소모·회복이 지속 입력에 대한 회로 반응을 바꾸고 감각 피드백 아래 반복적인 상태 전환을 가능하게 한다는 것이다. 대안은 이 동역학이 신호를 약화하거나 균형만 해치고 협응을 만들지 못한다는 것이다. **STP를 BANC에서 측정하거나 복원했다는 주장이 아닌 공학적 동역학 가설**이다. 생물학적 정확성과 실제 보행 개선은 별도 미검증이다.

기준은 E150 원래 MotorTimescaleNetwork+monotone body이며 E277 기각 source/decoder를 중첩하지 않는다. 기존 source CSR 간선·부호·세포/감각/운동 전달/신체는 보존한다. `tools/banc_m3_recurrent_mask.py`의 정적 선정 규칙으로 직접 motor presynaptic VNC intrinsic population과 그 사이 실제 nonzero mutual edges만 고른다. 기존 E29 선택은 population7136/edges725849였지만 E150 실제 weights에서 같은 규칙을 다시 검증하기 전 현재 roster로 확정하지 않는다. 활동값/평가결과로 node·edge를 선별하지 않는다. 새 간선, 외부 CPG/위상/clock 파형, 교사 런타임 주입, root force는 없다.

## E289 신경 구현 계약

선택 간선의 unique source p마다 **하나의 공유 resource R_p**를 둔다. 이는 선택된 recurrent terminals가 한 presynaptic pool을 공유한다는 명시적인 모델링 가정이다. 간선별 자원을 만들거나 모든 다른 outgoing edge에 암묵 적용하지 않는다.

`dR_p/dt = (1-R_p)/tau_p - u_p*r_p*R_p`

선택한 e=(p,j)의 전류는 `W_e * (r_p*mask_p) * R_p`이며, 미선택 간선과 `u_p==0` 간선은 원래 packed CSR 식을 그대로 실행한다. `r_p`는 해당 RK 단계의 실제 rate이며 resource 변화는 실제 rate에만 의존한다. Mask는 전달 전류를 즉시0으로 만들지만 원래 rate의 resource 상태 갱신은 계속된다. Rate·adaptation·resource는 네 RK4 단계에서 공동 적분하며 rate 작업 버퍼를 덮어쓰기 전에 resource 미분을 계산한다. Endpoint 저역통과, one-control 지연, source history 필터 중첩은 없다.

- `u`: sorted unique source 수의 native float64, [0,.5]. `tau`: 같은 shape float64, [.02,.5]초. 원래 cap/유효 dt를 사용하며 `dt <= min(tau)/20`, `dt*(1/min(tau)+max(u)*max(cap))<=.1`을 요구한다. Rate 단위에 대한 실측 추정을 추가하지 않으며 이 식의 u는 현재 모델의 rate 단위에 대한 depletion coefficient다.
- R 및 u/tau 계산은 float64, RK-stage r은 원래 float32를 float64로 확장한다. 전류에서만 R을 float32로 바꾼다. 원래 masked-rate 곱을 보존한 뒤 R의 곱과 CSR 합을 원래 float32 FTZ/reduction 순서로 수행한다. FMA를 끄고 scalar oracle과 정밀도 계약을 명시한다.
- `u=0` 전체 neutral은 신경 r/z/drive/mask/tick와 원래 CUDA captured advance가 bit exact여야 한다. 이 경우 R은 reset부터1이며 기존 current식에 추가 곱을 하지 않는다. 혼합 u에서0인 source의 간선도 같은 원래 식을 사용한다.
- 기존 canonical CSR의 모든 original weight bytes/positions를 보존한다. **선택된 간선만 nonzero**여야 하며 선택하지 않은 원래 explicit-zero 항목도 그대로 유지한다. 선택 간선 pre/post 모두 제공한 premotor population 내부, actual native motor roster와 population disjoint를 엄격히 확인한다. 정렬/중복/범위/dtype/부호/metadata 오류는 reject하며 topology를 조용히 정렬·수정하지 않는다.
- 생성·reset 때 R=1, neural 원래 reset 의미 유지. CUDA warmup/capture는 scientific state가 아니므로 종료 후 resource도1로 돌린다. 모든 capture 주소는 안정적으로 유지한다.
- snapshot은 원래 metadata/state와 `recurrent_resource` metadata, `synaptic_resource` float64 배열을 포함한다. 새 identity는 원래 neural identity와 모든 selection/u/tau/precision bytes를 포함한다. arrays는 owned copies다.
- restore는 전체 metadata/state dtype/shape/finite/range를 먼저 검사하고 staging을 마친 뒤 기존 pointer에 복사한다. R∈[0,1], zero tick은 R=1, u0 source는 R=1을 요구한다. 잘못된 metadata/state가 기존 live 모델을 부분 변경하면 안 된다. 일반 restore는 저장 R을 정확히 복원한다.
- `import_base_snapshot(saved)`는 명시적인 원래 neural identity 이식이다. 처음 적용하는 파라미터는 tick0/rate0/adaptation0/drive0/mask1인 원래 neutral에서만 허용하고 R=1로 초기화한다. 일반 restore에서 identity를 무시하지 않는다.

모듈 API는 `RecurrentResourceNetwork(MotorTimescaleNetwork)`이며 `edge_ids`, `premotor_ids`, `utilization`, `recovery_tau` keyword를 받는다. Public `selection` owned copy, `resource` tensor, `source_timescale_identity`, scalar edge/current oracle, `checked_selection`, `resource_derivative`, `resource_cuda_source`, reset/snapshot/restore/import API를 제공한다. Metadata source key는 `source_model_hash`다. `resource`는 shape(unique sources,), `recurrent_resource`는 metadata key, `synaptic_resource`는 state key다. Wrapper가 같은 계약을 사용한다.

E289 owned: `tools/banc_m3_recurrent_resource_synapses.py`, `tests/test_c_banc_recurrent_resource_synapses.py`. 실제 gpt-6-sol/xhigh 구현→root FIRST→필요시 같은 Sol 첫수정→root SECOND→root 잔여수정. 기존 모듈은 읽기 전용이다.

## E290 whole-session 연결

새 `RecurrentResourceTrial`은 exact open `MonotoneRecruitmentTrial`의 neural layer만 교체한다. 네트워크를 만들고 neutral 이식·전체 초기 상태 대조에 성공한 뒤 body ownership을 한 번 이전한다. 실패하면 기존 base를 임의로 닫거나 상태를 바꾸지 않는다. 기존 source-filter layer를 중첩하지 않는다. Adapter/bank/신체/held request/physical clock/controller/sensory state는 exact 보존한다.

Patch key `recurrent_resource_trial_patch`, network 위 metadata/state key와 새 trial/patch schema를 사용한다. 원래 graph/neural/adapter identity와 전체 original metadata/selection/original monotone patch 및 patch hash를 보존한다. 명시적 `remove_resource_layer`는 검증한 patch에 따라 원래 state/metadata를 복원한다. Nonzero whole from_checkpoint/restore는 실제 기존 body 복원 경로를 사용하며 neural-only shortcut이나 whole adapter 제외는 없다. Invalid metadata는 새 body 생성 전 reject하고, 실패/cleanup 중 최초 예외와 추가 notes를 보존한다. Whole restore가 새 세션을 commit한 뒤 old body close에 실패하면 commit 사실을 명확히 남긴다.

E290 owned: `tools/banc_m3_recurrent_resource_trial.py`, `tests/test_c_banc_recurrent_resource_trial.py`. 실제 gpt-6-sol/high가 E289 API에 맞춰 구현하고 같은 FIRST/첫수정/SECOND 절차를 따른다. E289가 아직 제출되지 않았을 때 fake dependency/metadata 시험만 허용하며 기존 원본/신체를 생성하지 않는다. 기존 E271 ownership/restore 의미를 참고하되 새 schema/필드를 명확히 한다.

## 검증과 실제 도착점

E289 생성 CPU≤120초, 생성 신경≤64cells/≤256edges, solver/body/original/CUDA0. 서로 연결된 작은 recurrent network에서 analytic constant-rate resource equilibrium/recovery, 독립 고정단계 coupled RK4, 비중립 실제 state 변화, 부호/원래 간선/현재 mask, neutral exact, mixed neutral, stable pointers, reset/nonzero disk future, invalid restore unchanged를 검사한다. 이후 root 별도 reservation에서만 tiny CUDA 검사를 수행한다. CPU/CUDA 오차 한도는 테스트 전에 모형 크기/기간과 함께 고정하고 결과를 본 뒤 완화하지 않는다. 생리/보행 성공 판정과 분리한다.

E290 생성 CPU≤90초, fake metadata/ownership/whole restore/save/close 실패만 검사하고 원본/CUDA/body0. 실제 통합·GPU·neutral exact·cut/disk future는 소스 승인 후 root의 별도 원본 설계/예약에서 한다. 비용은 실패 포함 실제 시간과 모델/step/setup을 기록한다.

첫 실제 행동 판별은 새 원본 roster에서 **고정 u=.05, tau=.150초 한 후보**를 사전에 선택한다. 이는 탐색으로 얻은 최적값이나 실측치가 아니다. Neutral u0의 E150 .005초 whole/trace exact를 통과한 뒤, 후보를 원래 neutral에서 .5초까지 실제 BANC–신체로 실행하고 기존 .04/.018 q·.025/.009 LTM/건강성 기준을 유지한다. 통과하면 동일 session1.5초에서 E288의 같은 전진/slip/eligible bout/여섯 다리 complete-cycle 기준을 적용한다. 이 후보 실행 전 상세 source/input/package freeze와 비용 상한을 별도 작성한다. 임의 tau/U grid, 다른 seed, 같은 후보 반복 fit은 하지 않는다. 양성일 때만 같은 후보10초/F2/F6/F7/F8로 확장한다. 어떤 신경 toy oscillation도 자연 보행 완료가 아니다.

## 원본 실행 전 실제 CSR 호환성 FIRST 정정

Root passive actual-CSR 검사에서 E150의 저장 CSR에는 전체11,584,852개 항목 중 explicit-zero1,499,070개가 있다는 사실을 확인했다. 초기 설계의 '기존 nonzero CSR' 표현을 전체 storage에 적용한 FIRST 구현은 이 정상 원본을 거절했다. 선택한725,849개 recurrent 항목은 모두 nonzero이며 roster를 바꿀 이유는 없다. 전체 zero를 제거하면 기존 edge ID와 identity가 바뀌므로 제거·재정렬·재인덱싱하지 않는다. 같은 Sol 첫 수정은 원래 전체 CSR의 finite/canonical/storage 검증을 유지하되 nonzero 검증 범위를 선택 간선에 한정한다. Root 생성 CUDA PASS는 이전 source의 tiny 범위 결과로 보존하고, 소스 변경 후 별도 검증을 거쳐 새로운 source-bound qualification을 발행한다. Passive0.2506595초/process3.2583739초, 원본/신체/CUDA/solver0. 보행·startup기준/후보u,tau/F1–F8 불변이다.
