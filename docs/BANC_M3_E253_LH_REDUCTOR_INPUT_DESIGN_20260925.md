# E253 LH femur reductor의 저장 MN·기존 입력 분해 — 2026-09-25

E252의 첫·최대 startup 위반은 물리 관절17 LH femur_roll이며, 그 명령 차이는 muscle41 femur_reductor 항에서 나온다. 이번에는 그 근육 전체5MN의 고정 decoder 기여와4bridge의 실제 기존 입력을 분리한다. 세포를 반응 크기에 따라 미리 고르지 않는다. 해부학 MN columns[13,44,128,296,343]/graph IDs[6373,22787,52774,121544,139521]를 모두 사용한다.

## 고정 질문과 판정

경쟁 설명은 한 MN·한 직접 bridge 경로에 집중된 출력 증가와 여러 경로/미관측 입력을 포함한 분산 변화다. H253_single_observed_route는 startup endpoint1..99에서 아래 두 집중도 조건의 conjunction이다.

1. 각5MN의 양의 근육 activation 기여 변화 합 P_m=sum_t max(delta A_m,0)를 구한다. 전체 P가 양수이고 최대 MN의 비중이80% 이상이다.
2. 그 MN으로4bridge가 보내는 양의 직접 입력 변화 P_j=sum_t max(W[m,j]*(rate_C[j]-rate_B[j])*mask[j],0)의 전체가 양수이고 최대 bridge 비중이80% 이상이다.

동률은 graph row ID가 작은 순서다. 두 조건 각각과 conjunction을 보고하고 분모0은 unavailable/미지지다. 음의·signed 기여도 전체 보고한다. 이80%는 제한된 진단의 집중도 기준이며 생물학적 기준이 아니다. 두 조건이 참이어도 전체158706뉴런의 원인 지배나 보행 개선을 증명하지 않는다.

지지되면 선택 경로의 부호·실제 source→bridge 복원 간선·미관측 입력 범위를 확인한 뒤, 그 경로를 분리하는 한 번의 신경 대조를 별도로 설계한다. 미지지면 여러 MN/bridge 또는 미관측 기존 입력을 관찰하는 쪽을 우선한다. 어느 경우에도 이 분석에서 새 계수, gain sweep, bridge 차단, 신체 재실행, 나머지 다리 복원을 실행하지 않는다.

## 고정 계산과 입력

E251301행의 baseline/candidate bank391×3/activation96/activity82/controlMN391과 실제 전체 CSR, 초기 output mask 및 동일 decoder parameters를 사용한다. E150 calibration·muscle provenance, E250 exact25복원 간선, E252 결과/선택 근거와 비용을 동결한다. 두 팔의 source/bridge/MN ID, clock, 데이터 중복, bank·activation availability, 초기 mask/decoder params 및 대상5MN의 모든 CSR incoming 행을 검증한다. 대상 MN의 incoming weights는 E251 개입 대상이 아니므로 두 팔의5행이 exact여야 한다. 모든 numeric observed 값은 finite여야 한다.

고정 monotone decoder의 같은 kappa/slopes/tau를 읽고, 대상 근육이 variable group인지 확인한다. 각 MN의 3개 bank별 기여는 low*bank+(high-low)*max(bank-kappa,0)이며 합은 A_m이다. 5MN 합을 저장 activation41과 모든301행에 대해 atol1e−10으로 확인한다. row0에서 decoder 활성과 held command가 다를 수 있으므로 여기서는 activation을 대조한다. delta A_m과 .02*moment[17,41]*delta A_m을 보존해 총 근육·명령 차이와 검산한다. 신규 bank update/decoder 실행/fit은 없다. 각 MN·bank의 원래 low/high slope 및 hinge region도 저장한다.

관측 신호는 source23+bridge4+전체MN391의 합집합418개다. activity82의 reachable55는 전체391의 해당 ID와 exact 대응해야 한다. arbitrary graph ID 순서에 의존하지 않고 대응하며 mask를 적용한다. 대상5MN으로 들어오는 모든 실제 CSR entry의 edge index/pre/post/weight를 저장하고, 관측418에 속하는 entry에는301행의 baseline/candidate/차이 current를 계산한다. 나머지 entry는 관측 안 됨으로 표시하며0 current로 해석하지 않는다.

category는 bridge4를 각각, sensory23 합, observedMN391 합으로 중복 없이 나눈다. 이들의 합은 observed subtotal일 뿐 full recurrent current가 아니다. 각 MN마다 전체/observed/unobserved의 간선 수·nonzero 수·absolute weight 합·비율을 보고한다. topology weight coverage는 actual current coverage가 아니다. 실제 수치 입력이 없는 unobserved contribution을 추정하거나0으로 채우지 않는다. 양/음/signed current 변화와4bridge별 원래 간선 부호, source→해당 bridge의 E250 복원 간선 IDs를 함께 기록한다.

startup1..99, steady100..299, full0..300의 세 고정 구간을 보고하고, E252가 이미 정한 control64/84의5MN·4bridge·signed muscle 기여 표를 작성한다. 이 점/창들은 E253 결과를 보고 선택하지 않는다. 시계는 endpoint t*.005s이며 native physics는 실행하지 않는다. 같은 endpoint의 상관으로 인과 지연이나 고유 neural mechanism을 단정하지 않는다.

## 구현과 검토·자원

부모 설계 후 실제 gpt-6-sol/medium가 `tools/analyze_c_banc_m3_lh_reductor_input.py`와 `tests/test_c_banc_lh_reductor_input.py`만 구현한다. 부모FIRST→필요시같은Sol첫수정→부모SECOND→남은수정은부모다. 생성 fixture는 임의 ID 순열, 양/음 slope 차이의 positive hinge 식, mask/미관측 entry 구분,0분모/동률, 전체5MN·4bridge 합산, late publication 실패를 검증한다. prior source의 runtime heavy imports 없이 numpy/threadpoolctl로 계산한다.

출력 `verification/banc-natural-gait-m3-20260919/e253-lh-reductor-input/`에는 실제 mapping/CSR incoming entry, observed418 raw rates/selected current, 5MN×3bank 기여와 hinge, window/concentration/table 요약, lineage/hash·output manifest·errors/ledger/terminal을 저장한다. 원본은 source hash/부모 승인 후1회/retry0이며 storage/clock/ID 오류는 invalid다. 기술적 집중도 기각은 valid일 수 있다. partial output과 실패 비용을 보존하며 성공 결과가 늦은 실패 뒤 남지 않게 한다.

CPU1/GPU0, model/neural/physics/fit/solver0, internal30(마감5)/external40/supervisor45초, host2GiB/output64MiB/free disk512MiB. 기존 누적6854.113728099637/6933초·잔여78.886271900363초에서45초를 예약하고 배정 증액0이다. 원본당300초 한도, 별도 준비/생성검사/부모감사 비용, 미측정 개발/자문null을 유지한다. F1–F8 미완료/E251 기각/E150 best partial/M2/UI/F6 미노출/과거 실패는 그대로다.
