# E222: 고정 인과반응의 전체 BANC·신체 10초 검증

E218 조건부 공유계수와 E221 실제 CUDA 신경 클래스가 검산됐다. 이제 원래 E150 중립 초기상태에서 전체 BANC와 동일 FlyBody를 한 번 연결해10초 평지 전진을 검증한다. 이 결과가 나오기 전에 보행 개선·생물학적 기전·목표 도달을 주장하거나 UI/M2를 교체하지 않는다.

## 변경 범위와 불변 조건

기존 실제 gpt-6-sol/high가 부모 설계에 따라 새 trial wrapper/runner와 각 테스트를 작성한다. 소유 파일은 tools/banc_m3_native_grouped_trial.py, tools/verify_c_banc_m3_native_grouped_body.py, tests/test_c_banc_native_grouped_trial.py, tests/test_c_banc_native_grouped_body.py다. 기존 E221·이전 소스는 변경하지 않는다. 부모 FIRST→필요시 같은 Sol 첫 수정→SECOND→부모 잔여 수정 절차를 적용한다. Sol은 작은 mock/CPU 테스트와 Ruff까지만 실행하고 원본·실제CUDA·body는 부모 승인 source gate 전 실행하지 않는다.

MotorTimescaleTrial 기반 연구 wrapper를 만든다. E150 start의 monotone wrapper를 정식으로 해제해 원래 AnatomicalRecruitmentAdapter(calibration/activation/original 상태)를 복원하고, E221 NativeGroupedResponseNetwork로 network만 교체한다. 이를 E150 원래 신경+monotone와 혼동하지 않는다. E220에서 미리 고정한 새 신경+canonical10ms 전달 조합이다. 기존 canonical adapter의 pool·moment·rest·activation은 float64인지 확인하고, live motor float32 rate를 float64로 전달한다. gain .02/10ms/tendon .2*tanh(a/10),42관절/96근육/391MN/6LTM/두 abdomen hold .2는 원래 그대로다.

E217 parameters/prototypes와 E218 candidate의 beta만 모델 파라미터로 읽는다. free343/fixed48의 기존 motor ID와 original membership을 사용하며96×64 beta의 번호를 유지한다. scale/center/sigma/slow/tau/cap 일치, pool의 단일 소유권, fixed6 zero beta, 그래프158706 및원래 graph hash/CSR와신경 파라미터 identity를 검증한다. 새 body, solver, 근육 sign/gain/limit 또는 신경 연결 weight 변경은 없다. 모든 런타임 E/I/drive는 live graph에서 오며 B/D/기록 rate/목표 관절을 동작 입력으로 사용하지 않는다.

wrapper identity/checkpoint에는 고정 새 파라미터의 해부학적 provenance와 기존 timescale network identity, canonical adapter identity를 넣는다. from_checkpoint는 own patch를 해제한 정확한 MotorTimescaleTrial 상태를 만든 뒤 wrapper를 재구성하고 새 r64/fast/slow 포함 전체 상태를 복원한다. 중립 최초 전환은 원래 control/neural tick0·영 rate/adaptation/새 기억을 확인한다. 임의 초기화·시간 이동·불일치 상태를 숨기지 않는다. deepcopy로 원본 snapshot을 보존하고 공유 body 소유권/예외 close도 기존 wrapper 패턴에 맞춘다.

## 실제 측정

한 원본 감독 프로세스 안에서 main 최대2000control, control당20neural(.00025초)/50physics(.0001초), 합10초다. 원래 E113의 AdvanceLedger/install_native_observer/native_arrays/body_quality/supported_slip와 기존 biological gait analyzer·capture_control을 재사용한다. 코드 clone/global override로 기존 평가 기준을 바꾸지 않는다. 매 advance 전/후 clock·실패를 기록하고 raw trace JSONL을 즉시 보존한다. 신체 수치 오류·nonfoot contact·upright<.9 등 기존 native health guard가 발동하면 즉시 중단해 측정 실패로 남긴다. 자원 중단/기술오류/실제 건강 실패를 구분한다.

1초(200control)에 전체 checkpoint를 디스크 저장·로드해 새 wrapper clone을 만들고0.1초(20control) 진행한다. main1.1초(220control) 전체 상태와 기존 exact 규약(비과학적 wall_s 제외)으로 비교한다. 각 native contact의 정확한 physics sample 수와 실제 neural/physical elapsed를 확인한다. source graph/CSR는 원래 일치해야 한다. 새64상태/기억·motor rate·96activation·42command·6tendon을 control마다 raw로 남겨 부모가 canonical 전달을 독립 재구성할 수 있게 한다. 저장 파라미터 및 fixed MN은 실제 live로 진행되며 원래 rate로 덮어쓰지 않는다.

main10초를 완료하면 그 시점 final snapshot과 보행 raw를 먼저 봉인한다. 이후 같은 main에서 motor cut4control/재시작16control을 추가0.1초만 실행해 별도 cut raw·clock에 남긴다. 차단 동안96activation/42neutral command/6motor tendon0, 두 abdomen hold .2 유지, 재시작 양의activation/비중립command를 검사한다. 신경 기억 자체를 강제로 reset하지 않는다. cut 실패는 main 측정과 구분해 보존하며 F2/F7 완료를 허위 선언하지 않는다. main 건강 중단 시 이 후속 cut과 새 pose/target 시험은 생략한다.

main·clone·cut 총최대2040control=10.2 neural/physical초,40800neural tick/102000physics step이다. 소스 construction warmup/capture는 별도 실제 비용으로 기록하며 이 시간에 넣어 숨기지 않는다. 원본 start·1s·main1p1·clone1p1·final 및 원시 rate/history/activation/command/native-contact, clock/상태 비교/모델 구성·SHA를 남긴다. 실패 때도 얻은 raw·부분 clock·정확한 오류와 정리 결과를 먼저 보존한다.

F3/F4/F5는 원래10초/.5초제외/전진10mm·net/path.5·upright.9·foot접촉·.3초bout·다리당10cycle·LR반주기±.15/R.5·관절width25/30/40/35도·periodCV.3의 고정 기준이다. 원래99 startup q RMS/max .018/.04 및 LTM .009/.025 진단도 E197 loaded_startup_target 배열과 동일하게 계산한다. E150 같은 window의 quality/slip와 비교하고 criteria를 재정의하지 않는다. 목표 도달·F6·전체F7·F8영상 검증은 이 단일 assay의 성공으로 대체하지 않는다.

## 예산과 실행 경계

사전 hash 검증/로드/구성/측정/분석/저장까지 내부175초(마지막 저장·정리20초 reserve), 외부185초, 감독200초. CPU1/GPU1, RSS8GiB, 새Torch/CuPy할당4GiB, 출력2GiB/free disk4GiB, 원본process1/retry0/fit0/solver0이다. 누적5795.139958599688/6004초·잔여208.86004140031218초에서200초 예약, 총배정 증액0. 부모 생성검사 비용과 E221 생성CUDA3.4858618초는 별도 보존하며 이 시도를0비용으로 표현하지 않는다. 시간 부족이나 측정 실패에 자동재시도하지 않는다.

완료되는 것은 실제 full-BANC body 진단뿐이다. 기본 선택 모델이나 기존 M2, 활성 UI, 미사용F6초기조건은 보존한다. F1–F8 미완료/goal active를 유지하며 결과에 따라 잔여 자원 안에서 다음 단계를 별도로 정한다.
