# E256 기존 시간상수 집합에 따른 감각 경로 분할 대조 — 2026-09-25

E254/E255는25간선 복원이 내부 회로와 MN의 초기 출력을 바꾸며, 그 변화가 기존 감각 이력에서도 유지됨을 확인했다. E250의 네 목적지20156/63026/111970/152280 중 이미 고정된 intrinsic_timescale38에는20156/63026/152280이 들어간다. 111970만 밖이다. original24에는63026/152280만 있고 네 목적지 모두 core_adaptation에 속한다. 따라서 'core와 non-core' 또는 생물학적 리듬/비리듬의 분리라고 부르지 않는다.

## 한 가지 후보와 비교

같은 E150 baseline CSR에서 source23→111970으로 향하는 기존6간선만 원래 rate 단위 sign(count)*.03 값으로 복원한다. 나머지19개 감각→bridge 간선은 E150의 .001배 값을 유지하고, 모든 비대상 data/indptr/indices는 exact 보존한다. 6개 선정 규칙은 E250 고정25간선의 postsynaptic row가 E245 intrinsic_timescale_node_rows 밖인 경우이며, 결과를 보고 대상을 바꾸지 않는다. 준비 파일에 실제6개ID/count/old/new와 보존19개를 고정했다. 노드111970은 GABA 예측 IN13A001/좌측 single_leg_neuromere다. 담당 다리·생물학적 리듬 기능은 주석으로 확정할 수 없다.

E250와 같은 E245 final tick40000 동적 상태와 E248의 원래 low/high 감각, 나머지 held drive를 사용한다. baseline 두 arm의 전체41행 rate와 마지막 전체 snapshot을 E248에 먼저 exact 비교한다. 새6간선 후보의6 arms(intact/bridge4 outgoing cut/sensory23 outgoing cut × low/high)를 실행하고, 별도의 세 번째 모델에서 저장 후보 시작을 읽어 intact_low 미래41행/전체final exact를 확인한다. 모델들은 순차 소유한다. 이200ms 작동점은 E251 startup이나 신체 폐루프가 아니므로 성공을 실제 초기 자세 개선으로 해석하지 않는다.

E250 constructor/identity transplant/packed CSR 검증을 재사용한다. 새 CSR은 새 identity를 갖되 identity chain 필드 외 모든 동적 상태·모수는 같아야 한다. 원래 두 cut 비교에서 candidate sensory_cut은 E248 전체rate/동적final exact이고, bridge_cut은4bridge 외 모든rate/해당final exact여야 한다. source-cut low/high의 비source 전체rate exact도 요구한다. 모든형태/시계/finite/실제mask/held drive를 확인하고 요청/진입/완료/부분 미상 작업을 따로 기록한다. 새 weight identity를 기존 것으로 위장하지 않는다.

## 고정 H256와 결정

E250와 같은56개 free LH MN의 controls1..40에서 rate/cap200으로 R=RMS(intact_high−intact_low), D=RMS((intact_high−intact_low)−(bridge_cut_high−bridge_cut_low))를 계산한다. R≥1e−4, D≥1e−4, D≥.2R은 그대로 유지한다.

추가 보호 질문은 기존 original24의 출력 변화 크기다. low/high 각각의 후보−baseline rate를 모두 모아 controls1..40/original24/cap200의 RMS를 Q6라 한다. 저장 E250 전체25간선 후보의 같은 두 arm과 baseline으로 같은 Q25를 계산한다. Q25>0이고 Q6≤.2Q25인 경우에만 '기존24 출력 변화80% 이상 감소' 조건을 지지한다. 이는 고정200ms의 rate 차이 기준이며 위상 안정성·생물학적 동일성·보행 기준이 아니다. baselineR/D, 6개R/D/Q6, 기존25개R/D/Q25, 변화의 양·음·signed 통계를 함께 남긴다.

`H256_supported`는 신경반응3조건과 Q보호2조건을 모두 충족해야 한다. 참조를 바꾸거나 core 집합·시간창·수치 문턱을 조정하지 않는다. 지지하면 같은6간선 후보의 짧은 신체 폐루프를 별도 설계하며, 초기 자세·전진·slip·연속전진 기준은 E251과 같다. 미달이면 이 고정 구조 분할은 후속 신체 후보로 기각하고, 반응 소실/기존24 변화 잔존을 구분한다. 같은6간선 배율을 수치 탐색하지 않는다. 모든 기술 gate 실패는 unavailable이며 과학적 기각과 구분한다.

## 구현·검토·자원

부모 설계 뒤 실제 gpt-6-sol/high가 새 `tools/probe_c_banc_m3_lh_timescale_partition.py`, `tests/test_c_banc_lh_timescale_partition.py`를 구현한다. 원래 E250 파일은 고정한다. 부모FIRST→필요시 같은Sol 첫수정→부모SECOND→잔여부모수정. 신규 신경 수식은 없고 CSR6개 범위, 새 identity, 원래 대조 재현·cut·disk, 고정 Q통계 및 작업/실패 게시를 연결하므로 high다. 생성fixture에서6/19/비대상 보존과 membership 기반선택, sign/count, R/D/Q와 경계, 주요 오류·부분 state·nested 성공 철회를 확인한다.

원본1/retry0/CPU1/GPU1, constructor3/setup180실행 중capture60, 과학7200tick(9×40×20), body/physics/fit/solver0. internal34초(마감5)/external35/supervisor37초, host6GiB/장치추가GPU6GiB/최소GPUfree7GiB/출력512MiB/최소disk2GiB. E250 같은3모델·9arm은18.51초였으므로 같은 작업량과 작은 saved Q계산에 이 한도를 둔다. 이전 누적6895.812142299637/6933초·잔여37.187857700363초에서37초예약, 배정 증가0. 결과와 비용을 지우는 자동 재시도는 없다.

출력 `verification/banc-natural-gait-m3-20260919/e256-lh-timescale-partition/`에 actual candidate CSR/6edge patch/count/선택roster/원래·새identity, baseline/candidate/disk 전체rate41행 및 모든 끝상태/drive/mask/시계, partial-final/오류/ledger/metrics/manifest/result를 보존한다. 생성자 setup 실패는 완료0과 부분 미상을 구분하고 snapshot 저장이나 cleanup/publication 실패 후 모든 성공 판정을 철회한다. 부모는 전체CSR6개변경, 실제조건, R/D/Q와disk/cut을 저장raw에서 독립 검산한다. 준비·생성·검산은 원본과 별도, 미측정 개발/자문은null. E251기각/F1–F8미완료/E150/M2/UI/F6/기존실패를 보존한다.
