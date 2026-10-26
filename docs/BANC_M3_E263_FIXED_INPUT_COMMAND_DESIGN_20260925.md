# E263 저장된 고정 감각 입력의 q19 명령 재구성 — 2026-09-25

E262의 짧은 물리 반환은 양수지만 작았다. E260에서 같은 U0 감각을 고정해도 연결 W6의 muscle44 변화가 유지됐다는 결과는 이미 관측됐다. E263은 이를 새 독립 발견으로 포장하지 않고, 기존 신경 효과를 실제 초기 q19 오류와 연결하는 저장 자료 재분석이다. 근육44 하나만 보면 놓칠 수 있는 길항근 상쇄를 포함한다.

고정 비교는 E260의 B=(W0,U0), C=(W6,U6), X=(W6,U0)이며 입력은 원래101개 endpoint다. E257 baseline/candidate control0..100, E258 초기 error/96근육 투영과 연결한다. calibration의 q19 비영 항 전체는 muscles42/43/44이며 각각 moment −1.8088900095777785/−1.8088900095777785/+1.8088900095777785다. 이들 세 근육이 original adapter의 fixed 집합 밖임을 직접 확인한다. 기존 snapshot fixed 집합은19개이며 이를 long-tendon6개와 혼동하지 않는다.

각 팔의 원래 motor-rate391과 baseline의 동일 초기 bank391×3, tau[.005,.01,.04], kappa/slopes/membership으로 bank를 산술 재생한다. control0에서는 주어진 bank를 쓰며 control1부터 `bank += -expm1(-.005/tau)*(rate-bank)`다. 세 근육만 기존 monotone 식으로 계산하고 q19=rest19+.02*moment19·activation을 구한다. 모델 생성, neuron 또는 physics 진행, fit, solver는 없다. 이 산술 bank 갱신은 신경 재실행과 구분해3팔×100회로 기록한다.

유효성은 B/C의 모든101행 MN391 원본 일치, bank 전체 재구성1e−10, 세 activation 및 control1..100 q19의 E257 일치1e−10, 세 팔 muscle44/E260 bank2MN 일치1e−10, E258 세 근육 투영·전체q19차·startup error와의 일치1e−10를 요구한다. control0의 held 명령과 decoder 산술값 차이는 별도 보고하고 첫 행을 임의로 교체하지 않는다. 모두 유한·비음의 rate/bank/activation, exact 시계·목록·고정 입력 hash도 검증한다. E260 prior audit의 결과는 계승하되 원래 파일 검증과 구분한다.

T=qC−qB, C0=qX−qB, S=qC−qX를 startup controls1..99 및 고정 cases56/70에 정렬한다. 각 근육의 signed/positive/negative 기여, 세 근육 합, baseline/candidate error와 `virtual_X_error=baseline_error+C0`를 저장한다. X error는 가상 명령이며 그 신체 궤적을 관측했다고 쓰지 않는다. 새 위반/최대오차도 q19에서만 보고한다. H263은 startup L1잔차 sum|T−C0|/sum|T|≤.2, 두 case 각각 |T−C0|/|T|≤.2 및 C0가 candidate error와 같은 방향인 조건의 conjunction이다. 분모≤1e−12는 판정 불가다. 이20%는 E260과 같은 조건부 보존 문턱이며 자연보행 기준이 아니다. 결과 일부를 이미 본 후 정한 재분석 기준임을 명시한다.

지지하면 다음 선택은 양의 피드백 부호 교정에서 연결 복원에 따른 tonic 동작점과 기존 decoder의 양립성으로 옮긴다. 미지지하면 길항근 상쇄/전체 명령 경로를 우선한다. 어느 결과도 연결 삭제·감각 부호 반전·무작위 gain sweep·UI 채택·새 신체 실행을 자동 승인하지 않는다. 구조 후보를 별도 설계하고 기존 실패 이력을 검토한다.

실제 gpt-6-sol/medium는 tools/analyze_c_banc_m3_fixed_input_command.py와 tests/test_c_banc_fixed_input_command.py만 구현한다. 부모 FIRST→필요시 같은 Sol 첫수정→부모 SECOND→부모 잔여수정 순서다. 검사는 실제 위험인 행0/1 정렬, 길항근 부호와 상쇄, 감소하는 high slope, 0분모, 불완전 자료·최종 게시 실패 판정 철회를 다룬다. 부모는 다른 scalar min/max 식으로 저장 원자료를 독립 검산한다.

출력은 verification/banc-natural-gait-m3-20260919/e263-fixed-input-command/에 raw.npz/result.json/ledger.json/errors.json/output-manifest.json을 보존한다. 입력과 source hash를 실행 전에 고정하고 기존 출력 덮어쓰기는 금지한다. 원본 분석1/retry0/CPU1/GPU0, internal10(마감2)/external12/supervisor15초, host2GiB/output32MiB/min free512MiB다. 누적7082.177781699607/7101초·잔여18.822218300393초에서15초 예약하며 증액0. 준비·검토·독립 감사는 별도 기록, 미측정 개발null. E261무효/E257기각/E150부분/M2/UI/F6미노출/F1–F8미완료/기존 비용 보존.
