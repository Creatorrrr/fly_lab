# E179: 두 무발화 운동뉴런과 근육 전체 출력을 구분

E178은 저장된 CUDA 입력 전류 56개를 bit exact로 검산했고, 두 RM tibia-flexor 운동뉴런의 큰 음의 입력을 설명했다. 그러나 두 세포의 무발화는 같은 근육의 전체 출력 부족을 뜻하지 않는다. E161/E164/E167은 근육 합계·명령 또는 한 팔의 전달 검산을 다뤘으며, E177 두 팔의 다섯 운동뉴런과 세 필터별 실제 기여를 분해하지 않았다. 이번 저장 자료 진단으로 그 차이를 확인한다. 새 신경·물리·fit 실행은 없다.

## 질문과 판정

1. 원래 391개 운동뉴런의 endpoint rate → 세 필터 → 96개 근육 활성 → 42개 관절 명령이 저장 자료와 일치하는가? 불일치는 자료/구현 계약 문제이며 생물학적 실패로 판정하지 않는다.
2. RM `tibia_flexor_muscle`(근육 행74, columns24/69/263/355/380)의 실제 출력은 어느 세포와 필터가 담당하는가? E178의 node108052/153939는 이 집합의 일원이어야 한다. 나머지 세포가 전 구간 활성 상태라고 미리 가정하지 않는다. node index는 저장 motor_ids, root ID는 저장 roster에서 각각 확인한다.
3. 근육 활성의 존재와 지지 해제의 관측을 구분한다. nonzero 출력이 확인되면 '두 무발화 세포 때문에 근육 전체가 꺼졌다'는 설명만 배제할 수 있다. 정상 보행 reference/개입 대조 없이 '발을 들기에 충분하다'고 판정하지 않는다. 출력이 없거나 매우 작아도 임계치를 사후 만들어 부족을 확정하지 않는다.

분석이 유효하고 다른 세포가 출력을 담당하면, 다음 설계는 근육 전체의 신경 발화를 임의로 올리는 대신 이미 관측된 다관절 명령/전달의 한계를 함께 다룬다. 안정적인 지지 해제가 없으면 조건부 전환 분석은 관측 불충분으로 남긴다. 출력 복구나 인과 주장을 위한 실험은 별도 사전 설계가 필요하다. E175가 배제한 동일 제한 양수 history-readout 가족을 다시 fit하지 않는다. M3 전체 F1–F8은 그대로 유지한다.

## 입력과 시계

E177 baseline/candidate 각각의 `start/state.json`과 manifest가 참조하는 최소 adapter 배열, `native-recruitment-state.npz`, `native-contact.npz`, `trace.json`, `muscle-provenance.json`, output manifest를 pin한다. 모델 복원과 전체 checkpoint 로드는 금지한다. 부모 freeze는 JSON 메타데이터, 파일 hash와 공개 NPY header만 읽고 원본 배열 값은 읽지 않는다. 사용 source와 E178 결과를 hash로 연결하며 기존 보호 증거를 승계한다.

- control 0..2000, neural tick=20*control, dt=.005초, native contact 100000행/10초/dt=.0001초. rate와 bank/activation은 기록된 float64, tick은 int64, availability는 모두 true여야 한다. 시계 오차 한도1e-8초.
- 각 control row t는 (t-1,t]에 적용되어 완료된 command를 담는다. t=0은 복원 anchor여서 target 식 검산·시간 평균에서 제외한다. native index i의 명령 row는 `i//50+1`이며 시각 `(i+1)*.0001`을 확인한다. row t의 neural endpoint rate가 해당 row의 decoder 입력이다. 한 tick 뒤로 옮기거나 보간하지 않는다.
- trace control_tick/time_s/command_interval, joint_targets 42열, endpoint motor_rates 391열, 몸체 speed/upright/feet load를 검산한다. trace의 rate는 기록 계약을 확인해 같음을 검산한다. unavailable/비유한 값을0으로 채우지 않는다.

## 전달 검산과 분해

calibration의 motor_ids/pool/moment/rest/ltm, parameters의 tau/kappa/slopes/membership/fixed_muscles를 참조 그대로 로드한다. 두 팔의 배열 및 named groups/roster는 동일해야 한다. pool의 각 열은 정확히 한 근육에 속하고 group columns/roster/provenance와 일치해야 한다. fixed는 LTM 6개와 singleton 근육의 합집합이다. fixed 근육은 저장 original_activation을 사용하며 monotone bank 기여로 잘못 분해하지 않는다. variable 근육만 학습된 monotone 전달이다.

각 인접 저장 행에서 `bank[t-1] + (-expm1(-.005/tau))*(rate[t,:,None]-bank[t-1])`를 계산해 bank[t]와 대조한다. 새로운 동역학 rollout을 하지 않는다. tau는 [.005,.01,.04]다. 96개 activation은 실행 코드와 동일하게 fixed를 원래 활성에서 복사하고, variable은 아래 두 합으로 복원한다.

`sum(low*bank) + sum((high-low)*maximum(bank-kappa,0))`

각 다섯 MN·세 bank의 low 항, signed hinge 항, 합을 raw로 보존한다. 음의 hinge 보정은 양수 high slope가 더 작다는 뜻이며 억제성 해부학 연결이 아니다. 개별 기여의 총합과 저장 근육 활성도 대조한다. bank/activation/개별합/target의 최대 절대오차 한도는1e-10이며 사후 완화하지 않는다. target은 행1..2000에서 `rest+.02*(moment@activation)`를 대조한다. 순간 original pool 입력 `pool*rate`는 필요 시 별도 이름으로 보고하며 필터링된 original_activation과 혼동하지 않는다.

raw에는 두 팔의 control/time, 해당 다섯 IDs/columns/roster, endpoint rate, bank, low/hinge/합 기여, 실제/재구성 RM activation, fixed 여부, 42개 재구성 command residual, RM 7개 실제/요청 관절, RM 근육별 7관절 command 기여, native support를50개씩 모은 비율/해제 횟수를 담는다. RM 부호별 명령 기여는 위치 서보 요청의 대수적 분해이며 물리적인 근육 힘이 아니다. 전체 검산 잔차의 최대값과 위치를 JSON에 남긴다.

요약은 t>0 전체(0,10], startup(0,1], 후기(1,10]의 정해진 세 구간이다. 다섯 rate와 MN/filter 기여의 mean/max/정확한 zero 행수, 시간 적분 기여와 그 비율, RM 근육 활성 mean/min/max, 원래 활성과의 차이, 두 팔 차이 첫 행/최대값을 보고한다. 분모0 비율은 null이다. 작은 양수를0으로 반올림해 silent로 만들지 않는다.

## native contact와 전환

RM native normal_load_bw>=.02를 지지로 정의한다. 모든 downcrossing을 보존하고 E161의 고정 조건을 그대로 사용한다: 전10ms 연속 지지, 후10ms 연속 비지지, ±20ms 온전한 창. native index i가 첫 비지지 sample이면 pre=[i-100,i), post=[i,i+100), 창=[i-200,i+200]이다. 배제 이유는 경계→pre불안정→post불안정의 순서로 집계한다. 기준 완화·평활화·새 임계치 탐색을 하지 않는다.

각 이벤트의 native 시각, control row, 선택·배제 이유, 해당 근육 활성과 5개 MN 기여를 남긴다. 선택 이벤트가0개면 조건부 요약은 unavailable이고 모든 downcrossing은 단순 기술 통계다. native contact의 비지지 구간 길이 최대값/지지 비율을 함께 보고한다. 이러한 관측은 보행 성공·회복 인과·신경 기전 검증을 대신하지 않는다.

## 구현, 자원, 출력

부모 설계 → actual gpt-6-sol/high 최초 구현 → 부모 FIRST → 같은 Sol 첫 수정 → 부모 SECOND → 부모 잔여 수정 순서를 따른다. Sol 소유는 `tools/analyze_c_banc_m3_muscle_contribution.py`, `tests/test_c_banc_muscle_contribution.py` 두 파일이다. 부모는 독립 생성 oracle과 리뷰를 병행한다. 소스 gate와 hash 검사를 통과하기 전 원본 배열 열람은0이다. 생성 자료로 fixed/variable 구분, 시간축, zero 분모, 비정상 입력/실패 publication을 확인한다.

원본 저장 자료 분석 CPU1/process1/retry0, internal90/external105/supervisor120초, output/raw128MiB/free1GiB. 이전 보수 누적3420.470999800018초, 동일 배정3656초 안에서 수행한다. GPU/모델구성/신경 step/물리 step/fit은0. 부모 검산 비용도 별도 기록한다. 입력을 단일 분석 process에서 읽고 두 팔을 순차 처리하여 메모리를 제한한다.

출력 디렉터리는 `verification/banc-natural-gait-m3-20260919/e179-muscle-contribution`이며 `contributions.npz`, `summary.json`, `events.json`, `input-receipt.json`, `result.json`, `output-manifest.json`을 만든다. manifest에 실제 파일 hash/bytes를 담고 실패 시 성공 표기를 제거한다. immutable input/source는 실행 전후 확인한다. 부모가 raw를 독립 검산한 뒤에만 해석한다. 활성 UI/M2/F6는 변경하지 않으며 후보 채택이나 목표 완료를 선언하지 않는다.
