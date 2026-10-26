# E152 전진 중단·지지 운동 저장자료 진단

## 목적과 판단 경계

E150의 넓어진 다리 운동폭과 F3 전진이 .3초 이상 연속 전진으로 이어지지 않는 이유를 좁힌다. 동일 FlyBody의 E142, E150 실제 BANC 주행과 E85 외부 기준 주행을 각각 기술한다. E85는 가능한 신체 운동의 비교 자료이며 BANC 성공 증거가 아니다. 관측된 연관성을 원인이나 개선 효과로 주장하지 않는다. F1–F8, Gaussian 20ms, 속도 .5 mm/s 및 최소 구간 .3초는 바꾸지 않는다. 새 후보·gain fitting·neural/CUDA/물리·F6·UI 실행은 없다.

부모의 사전 schema inventory는 E142/E150 각 trace 및 세 NPZ의 형태·키, E85 trace와 contact 형태를 읽었다(각 1.3339233초, .5029970초). 여기서는 새 시뮬레이션이나 통계 탐색을 하지 않았다. E152는 아래 식과 출력 범위를 고정한 별도 saved-data 분석이다.

## 고정 데이터와 수식

원본은 세 recording의 `trace.json` 및 `native-contact.npz` 각 하나뿐이다. 모두 t=0..10초 2001 boundary rows와 t=.0001..10초 100000 physics rows, 관절42/다리6이다. original body hash와 출발 qpos/qvel을 세 기록 사이에서 검증한다. trace의 초기 행은 기하 미분에만 쓰고 native 통계의 endpoint는 1..2000이다. 부동소수 시간 오차 허용은 절대 2e-8초, 결측을 보간하거나 이어 붙이지 않는다. bool과 실제 slip validity를 검증하고, E85의 빠진 slip_valid만 isfinite(slip)에서 유도했다는 provenance를 적는다. 지원하중 .02 BW 이상인데 slip이 없으면 invalid다.

기존 `analyze_c_banc_biological_gait.load_model/analyze`를 그대로 사용해 고정 window [.5,10), 물리 Thorax 미분·forward축 투영의 Gaussian sigma .02초 전진 속도, 기존 적격 구간·F4/F5 원자료를 산출한다. 본 분석이 별도 F4/F5 통과를 만들지 않는다. trace physical position/basis/foot coordinates, 관절 actual/target, boundary load/contact와 native finite/nonnegative loads를 검증한다. 동일 body hash·same zero pose는 요구하지만 외부 기준의 신경 model hash 일치는 요구하지 않는다.

각 trace에서 각 발의 tarsus5 origin을 Thorax 기준 좌표로 투영한다. `ego_AP = dot(foot_world - thorax, forward_basis)` (mm), `ego_AP_velocity = gaussian_filter1d(gradient(ego_AP,.005), sigma=4, mode='nearest')` (mm/s). 양수는 몸 기준 앞으로 움직임이며 실제 접촉점 추진력·힘·일과 동일하지 않다. 별도로 world foot 속도 `gradient(foot_world,.005)`를 해당 시각 forward축에 투영한 후 동일 smoothing을 적용해 저장한다. 관절 tracking은 각 다리 순서7개의 target-actual RMS다. load>=.02를 boundary 지원으로 검증한다.

Native (t-.005,t] 50개 샘플을 endpoint j에 일대일 대응시킨다. 각 block마다 다리별 평균 normal load, supported duty, supported load sum, supported load*slip sum, 전체 support count 평균, count<3 비율, count==0 비율을 저장한다. slip은 supported & valid인 샘플만 곱하며 NaN을0으로 치환해 지지 중 미끄럼 누락을 숨기지 않는다. 후속 집계는 분자 합/분모 합으로 계산하고 block 비율의 평균을 하중 가중 평균이라 부르지 않는다. force includes adhesion, slip은 실제 MuJoCo 접촉점 크기, 발 proxy와 구분한다.

고정 window를 speed>.5인 forward와 나머지 interrupted 두 집합으로 나눈다. 각 집합과 전체에 대해 속도·yaw rate·tracking·다리 AP velocity의 p05/median/p95, native support count/duty/load/하중 가중 slip, boundary supported 상태에서 AP velocity>0 비율을 각 다리 및 전체로 출력한다. 빈 집합/분모0은 null이며 실패나 0값으로 숨기지 않는다. 여섯 support bit로 정의한 모든64 boundary 패턴의 샘플 수와 속도 중앙값을 빠짐없이 출력한다. 조합 순서는 bit0=LF,1=LM,2=LH,3=RF,4=RM,5=RH다.

중단 사건은 window 내부에서 직전 speed>.5이고 현재 speed<=.5인 하향 crossing이다. 각 crossing 앞20/뒤20 boundary(±.1초)의 정확한41행 창이 모두 window에 들어갈 때만 event stack에 포함한다. 모든 crossing, 포함/제외 사유와 index를 기록한다. 겹친 사건을 독립 표본이라 하지 않는다. event stack의 평균/중앙값 곡선과 사건별 pre[-.1,-.005], post[0,.095] 값은 서술값이다. 추론 p-value나 인과 판정은 만들지 않는다.

추가로 기존 phase_and_events에 `moving=window`만 주어 all-window foot 주기·좌우 phase를 진단한다. 이 출력에는 backward/stationary가 포함됨을 명시하고 `exploratory_only=true`, `eligible_gait_pass=false`로 고정한다. 기존 정식 eligible 결과를 항상 같이 저장한다. 이 진단은 주기 자체 소실과 지지 타이밍 불일치를 구분하는 참고값이다.

## 산출물과 소스 계약

신규 파일 두 개만 실제 Sol/high가 FIRST 구현한다:
`tools/analyze_c_banc_m3_interrupted_support.py`, `tests/test_c_banc_interrupted_support.py`.
부모는 설계·freeze·독립 수식 검산·리뷰를 맡는다. 같은 Sol FIRST correction 후 부모 SECOND review에서 남는 문제는 부모가 고친다. 난이도는 고정된 수식의 다중 자료 정렬·수치 집계이므로 high이며 CUDA/state architecture 변경이 없다. 기존 agent 재사용 시 실제 설정 high를 그대로 명시하고 followup이 effort를 바꾼다고 주장하지 않는다.

CLI는 --freeze, --freeze-sha256, --source-review, --source-review-sha256, --output. 모든 사전 SHA/부모 source gate 확인 후에만 원본을 읽고 output은 새 빈 디렉터리여야 한다. gate `accepts_single_saved_diagnostic=true` 및 본 두 source SHA가 필요하다. inputs/source/preservation을 전후 모두 검증한다. 기존 helper는 read-only다. GPU/model을 생성하는 import/call 없이 NumPy/SciPy와 순수 분석 helper만 사용한다. 원본 실행 전 부모 review에 사용할 순수 함수 API를 제공한다.

각 recording의 features.npz(원본 t/window/speed/foot_velocity/관절tracking/boundary support와 모든 native block 분자·분모), event-stacks.npz(공통 event index/offset/각41샘플 feature), summary.json, official-quality.json, exploratory-all-window.json을 저장한다. root result.json, timing.json, ledger.json, error.json(해당시), output-manifest.json 및 progress.jsonl을 만든다. 세 원본 각각의 raw features를 요약 assertion보다 먼저 저장하고 오류가 나도 이미 만든 파일을 남긴다. accepted는 분석 무결성만 의미하며 candidate_adopted/F6_exposed/whole_goal_complete=false, E150_original_still_incomplete=true를 필수 저장한다. 실제 read 횟수와 각 source의 block/sample 수를 ledger에 기록한다. plot은 부모가 저장 feature만으로 별도 제작한다.

## 한도와 검증

원본 분석 프로세스1, 자동재시도0, CPU thread1/GPU0, fit/model/physics0. 내부35초/외부45초/감독55초, raw512MiB/free12GiB. 이전 원본 누적2442.1740688초를 보존하고 ceil2443+55=2498초로 기존 전체2514초 내에서 배정한다. 원본 시간에는 raw·요약·manifest·마지막 결과 저장이 포함되고 초과 또는 마지막 publication 실패는 accepted=false다. source review는 위 새 한도와 입력 봉인을 확인한 뒤 한 번만 원본 실행을 허용한다. 별도 부모 saved-output 검산은 내부30/외부40초, 모델0으로 수행하고 비용을 분리 보고한다.

생성 테스트는 native50:1 경계·변형된 clock·지원 중 NaN·비지원 NaN·가중 slip 분자/분모·회전 좌표·집합 공집합·양끝 사건 제외·어긋난 input/gate·늦은 저장/실패를 검사한다. 부모는 독립 NumPy 수식으로 original features·각 mask/count·집계·event alignment를 비교한다. 결과가 미끄럼/앞방향 지지 동작/지원 공백과 관련된다면, 그 이후 개입은 별도 설계하며 이 분석만으로 자동 fitting이나 native rerun을 시작하지 않는다.
