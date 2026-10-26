# E254 저장 감각 입력으로 동일 신경망 재생 및 전체 incoming 관측 — 2026-09-25

E253에서 초기 LH femur_reductor 출력 증가는 MN22787/6373에 분산됐고, MN6373의 관측 current는 감소했는데 근육 기여는 증가했다. 그 MN incoming 가중치의96.38%는 활동이 관측되지 않았다. 이번에는 기존 신체 실험 E251을 정확히 신경만 재생하여 입력과 내부 상태의 누락을 채운다. 새로운 보행 후보나 개입을 만들지 않는다.

## 고정 입력과 비교

baseline/candidate 각각의 E251 observed-start 전체 network snapshot, 원래 CSR, E150 membrane/cell configuration, 각300 control의 sensory-input.old_filtered 및 포트 매핑을 사용한다. 두 arm은 원래 E251처럼 서로 다른 저장된 감각 이력을 받는다. 이는 E251 결과를 관찰하는 재생이며 새로운 폐루프 보행/미노출 검증이 아니다. F1–F8과 E251 기각/E253 집중 가설 기각은 유지한다.

state의 실제 키는 parameters이며 p가 아니다. cuts 네 가지 모두 false, parameters.descending_drive=0, navigation.cut=true, forward_drive=5000, turn0/brakingfalse를 고정 확인한다. graph source의 20개 GABA 포트와2개 DNg100을 metadata로 확인하고 encoder identity를 맞춘다. old_filtered는 float64 상태 그대로 읽어 원래 float32 sensory scatter(+0 포함, 중복 ID는 원순서 np.add.at)를 수행한다. 그 뒤 descending_drive를 더하고, −5000/전진+5000 navigation vector를 더한다. 중복·겹친 포트의 원래 연산 순서를 지킨다. E251의 returned_source_drive23도 합산 전 sensory 벡터와 exact 비교한다. frozen cut/config/clock가 다르면 실행 전 invalid다.

E250의 검증된 body-free _constructor를 각 arm에 사용하되 E250의 tick40000 전용 _load_inputs는 사용하지 않는다. 새 E251 네트워크 두 개는 observed-start network를 그대로 복원한다. 모델 identity, 실제 float32 CSR 및 packed CUDA rowptr/column/weight lookup, tau/gain, 전체 초기 snapshot이 exact여야 한다. 초기 동적 상태의 hash 필드도 교체하지 않는다. 정해진 원래20 tick/control을300회 재생한다.

## 유효성 gate와 고정 질문

모든301 endpoint에서 원래 MN391 rate와 activity82의 rate/drive/adaptation이 bitwise exact여야 한다. 마지막 control300에서는 저장된 E251 final network snapshot의 모든 필드가 exact여야 한다. 시작/끝 관측 전후 전체 snapshot도 같아야 한다. 전체망 중간 궤적의 독립 참조는 E251에 없으므로 MN/선택 cohort 일치와 마지막 전체 일치를 구분한다. 어느 하나라도 실패하면 H254 판정은 unavailable이며 replay를 더 진행하지 않고 이미 끝난 작업과 partial raw를 보존한다.

관측 대상은 E253과 같은 전체5MN [6373,22787,52774,121544,139521]이다. 실제 CSR의 이5행 모든 entry와 모든 presynaptic ID, output mask를 저장한다. source23+bridge4+MN391을 포함하는 기존 observed418과 모든 incoming presynaptic ID의 합집합을 정렬해 매 endpoint의 rate/drive/adaptation을 기록한다. 5MN에 대한 실제 packed CUDA _current 관측도 기록한다. 이 호출은 state를 변경하거나 neural tick을 전진시키지 않는다. 그 행을 부모의 독립 float32 lane32 누적/warp tree로 재계산하여 actual current와 bitwise 비교한다. endpoint current는 RK substep의 causal latency 증거가 아니다.

별도로 각 edge의 float64 W*(rate*mask)와 arm 차이를 보존해 observed418/unobserved aggregate, signed/positive/negative 변화를 재구성한다. float64 수학 합과 실제 float32 CUDA 합의 반올림 차이도 기록하며 서로 동일한 산술이라고 부르지 않는다. 고정 threshold/coefficient/cap/membrane tau 및 MN별 adaptation gain/tau를 저장한다. endpoint별 delta current/drive/−adaptation과 그 합 delta margin을 분리해, 입력 이외의 이력 설명도 남긴다. bank를 업데이트하거나 근육/몸체를 생성하지 않는다.

기술적 재생·관측 gate를 통과할 때만 H254_unobserved_compensation을 판정한다. 대상은 사전 특정한 MN6373, startup endpoint1..99이다. O=sum delta observed current, U=sum delta unobserved current로 놓고, O<0, U>0, U>−O가 모두 성립하면 지지한다. 이는 미관측 재귀 입력이 관측된 감소를 상쇄하는지에 대한 signed endpoint 분해이며 생물학적 인과 증명이 아니다. window별 O/U/−adaptation/drive/margin을 모두 보고하고 입력 상쇄가 없으면 adaptation/history 쪽을 다음 설계에서 검토한다. 결과를 보고 기준이나 대상 MN을 바꾸지 않는다.

startup1..99, steady100..299, full0..300 및 control64/84의5MN 전체를 보고한다. signed edge 증가/감소는 모든 raw를 보존하고 각 MN의 양/음 상위10개를 graph ID/CSR edge ID로 표기한다(동률 pre ID, edge ID순). 순위는 설명용이며 다음 간선 변경 승인이 아니다. 지지/기각 어느 쪽도 새 gain sweep·다리 확장·신체 실행으로 자동 연결하지 않는다.

## 구현, 검토, 자원

실제 gpt-6-sol/high가 tools/replay_c_banc_m3_lh_full_inputs.py, tests/test_c_banc_lh_full_inputs.py 두 파일을 구현한다. 부모 FIRST→필요시 같은Sol 첫수정→부모 SECOND→남은 수정은 부모가 수행한다. 합성 검사에서는 감각 scatter의 중복/겹친 포트와 부호0, 초기/마지막 clock 및 stale-reference 차단, input row와 mask, 원본 작업 카운트/부분 실패, 늦은 publication의 성공 철회를 검증한다. 원본 모델은 부모 source 승인 뒤 한 번만 실행한다.

원본1/retry0, GPU1/CPU1, 네트워크 생성2, setup 실행120tick(그중 capture40), 과학12000tick, endpoint current probe602회, 몸체/physics/fit/solver0. 추가 replay, 새 신체, 배열 fitting은 없다. CUDA tensor identity 및 actual packed backend를 확인하고 torch/threadpool 설정을 복구한다. 요청/진입/실제완료/실패시unknown partial을 구분하여 setup/capture와 science를 모두 보존한다. raw는 observation 완료 즉시 저장해 뒤의 exact 검사나 storage 오류로 소실되지 않게 한다.

internal55초(마감6)/external60/supervisor65초. host4GiB, global GPU additional3GiB(프로세스 전용치가 아님을 명시), 최소GPUfree3GiB, output256MiB/minimumdisk1GiB. 출력은 verification/banc-natural-gait-m3-20260919/e254-lh-full-input-replay/에 원본 write-once 생성한다. 초기/마지막 network StateStore, 모든 raw/ID/CSR/관측availability/clock/parameters, generated drive의 sparse nonzero port mapping와300행/원래filter, summary/errors/ledger/hash manifest를 남긴다. finalization이나 cleanup 실패 후 stale success를 남기지 않는다.

E253 감독 누적6855.228979599637/6933초·잔여77.771020400363초에서65초 예약, 배정증액0. 준비/생성검사/부모감사는 별도로 기록하고 미측정 개발/자문은null이다. E150/M2/default/UI와 F6 미노출/과거 실패/F1–F8 미완료를 보존한다. 정확한 신경 재생은 제품 성능 개선이나 자연스러운 보행 성공을 대신하지 않는다.
