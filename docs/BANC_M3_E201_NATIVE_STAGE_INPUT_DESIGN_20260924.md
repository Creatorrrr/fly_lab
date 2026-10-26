# E201: E150 신경 내부 응답 학습을 위한 정확한 단계 입력

## 결정과 목적

E200에서 비활성12근육 보완의 저장 명령 개선 상한이6.56%였다. 따라서 다음 후보는 기존 운동뉴런의 입력→발화 비선형 응답 자체를 학습하는 국소 모델로 정한다. BANC의158706노드/원래 연결,391 MN→96근육 대응 및 양수 출력을 유지한다. 교사 명령은 오프라인 손실에만 사용하고 런타임 입력은 감각과 신경 상태뿐이다. 이는 미측정 세포 모델 가정이며 생물학적 동일성 주장이 아니다.

우선 기존 r/z 상태와 적분기를 유지하면서 입력 비선형식에 영계수에서 항등인 유계 보정을 넣는 작은 후보를 설계한다. gain/시간상수만 바꾼 E123/E128/E134와 달리 입력 크기에 따른 발화 형태를 바꿀 수 있다. 새 이력 상태를 추가하는 더 큰 모델은 이 후보의 한계가 확인되기 전에 만들지 않는다. 구체적 기저·학습 범위·손실·한도는 다음 적합 실행 전에 별도 고정한다.

그 선행 조건은 **현재 최선 E150의 실제 RK4 네 단계 입력**이다. E150 경계 자료는5ms 간격이고 E124의 경계 보간은 이미 부정확했다. E126의 단계 자료는 다른 E110/E113/E121 문맥이므로 E150로 바꿔 부르지 않는다. E201은 새 파라미터·보행 후보를 시험하지 않고, 동일 E150의10초 전체 단계를 한 번 수집한다. 최선 E150·F1–F8 실패·F6 미사용·M2/UI는 그대로다.

## 구현 범위

actual gpt-6-sol/high가 다음4개 파일만 구현한다. 부모 FIRST, 필요시 같은Sol 첫 수정, 부모 SECOND와 잔여수정 순서를 지킨다.

- `tools/banc_m3_timescale_stage_input.py`: MotorTimescaleNetwork의 수동 관측 하위 클래스 및 기존 MonotoneRecruitmentTrial의 network 교체 함수.
- `tools/observe_c_banc_m3_timescale_stage_input.py`: E201 단일 실행기.
- `tests/test_c_banc_timescale_stage_input.py`: 합성 CUDA parity/lifecycle 시험.
- `tests/test_c_banc_timescale_stage_capture.py`: 입력 gate/부분 보존/결과 의미 시험.

기존 MotorTimescaleNetwork의 gain/T/원래CSR/packed kernel을 유지한다. `csr_input` lane0의 `drive + recurrent current` 직후, threshold/adaptation을 빼기 전에391 MN 전류만 저장한다. 한 신경 step당 네 stage,40000 step의 총160000×391 float32이며 payload250240000bytes다. drive를 포함한 총 입력과 threshold를 뺀 입력을 혼동하지 않는다. 기존 `_timescale_source`가 만든 stage/finish를 그대로 유지하고 CSR의 계산 순서·부동소수점 옵션을 변경하지 않는다.

관측 버퍼·ID map·counter·armed·overflow는 과학 모델 identity와 checkpoint 밖의 수동 기록 상태다. 사전 GPU용량 검사를 하고 constructor stream에서 warmup/capture 완료 후 한 번만 arm한다. 관측 시작 후 restore와 두 번째 arm은 거부한다. 정지·동기화 이후 readback, overflow를 포함한 counter/NaN padding을 보존한다. 신경 tick마다 CPU 동기화나 추가 CSR 계산은 넣지 않는다. 기존 runtime의5ms readout은 유지한다.

실행기는 원래 E150 `start`를 MonotoneRecruitmentTrial.from_checkpoint로 복원한다. 원래 session 전체와 network identity/state가 같은 것을 확인한 뒤 수동 관측망으로 바꾼다. E36/Core-only wrapper로 대체하지 않는다. 모든 nested motor gain/logT·monotone bank·adapter 상태가 보존되어야 한다. full158706 graph와 실제 packed CUDA를 필수로 확인하고 fallback은 허용하지 않는다.

## 고정 비교와 판정

1. 같은 start에서2000 control/40000 neural/100000 physics/10초 실행한다. neural dt .00025, physics dt .0001, control dt .005다. observer의 capacity160000, 각MN counter160000, overflow0, 종료armed0, stage 모든값 finite여야 한다.
2. 원래 E150의1.1초 `main-1p1` 및10초 `final` 전체 snapshot이 동일해야 한다. 비교에서 기존 측정 wall_s만 제외하며 모델·body·network·adapter의 필드를 제외하지 않는다. 처음 전체snapshot도 같은 방식으로 비교한다.
3. 원래2001 control/body trace,2001×391 MN rate,2001×96 activation,2001×391×3 recruitment bank,2001×42 명령을 모두 대조한다. 원래 trace에 존재하는 complete/health/clocks도 유지한다. target q의 초기 held row0을 새 adapter 생성 명령으로 검증하는 E200 오류를 반복하지 않는다.
4. stage입력과함께 실제초기MN rate/z, tau/a/threshold/cap/gain/T/ID,391MN 소유권/원래 근육 파라미터의 provenance를 저장한다. 실제 원본 배열에서 읽으며 임의 고정값으로 대신하지 않는다. 후속 조건부 RK4 재구성은 별도 사전 설계 후에 수행하며 이번 수집 성공에 포함하지 않는다.
5. 유효성은 `valid_exact_native_stage_dataset` 또는 실패/미결로 판정한다. parity 불일치가 있으면 자료를 적합에 사용하지 않는다. 모든 raw와부분오류를 보존한 후 비교한다. 계산만 완료된 것과 parity 통과를 분리한다. 새 보행 개선/학습/자연보행 성공을 주장하지 않는다.

## 실행 전제와 자원

실행 전 부모가 freeze/input/source hash를 확인한다. 이전자료 전체를 덮어쓰거나hash기준을 갱신하지 않는다. 정해진 source gate와 SHA를 통과한 코드만 부모가 실행한다. GraphStore/StateStore의 manifest 참조파일, 비교 raw, 재사용 소스를 동결한다. 테스트는 작은 생성그래프만 사용하며 전체BANC/F6/원본물리 실행은 하지 않는다. 합성 CUDA 생성·warmup/capture/main 비용을 원본과 별도 보고한다.

원본은 **CPU1/GPU1/process1/자동retry0**, 내부270초(모델 실행/수집245초까지,마감25초), 외부285초, 감독300초, 출력2GiB, 관측GPU최대512MiB, 사전free disk12GiB다. 기존 E150 감독188.283초와 full10초 재구성을 근거로 예약했다. 새 단계300초를 기존 상한5144초에 사전 추가하여 **5444초**, 누적5032.9505163997055초는 보존한다. 새잔여411.0494836002945초 중300초 한 번을 예약한다. 사용자 총예산이 아니며 학습/다음 후보의 자동예약도 아니다.

부모는 실행 직전에 GPU 가용성·메모리·경합 프로세스와 디스크를 읽기 확인한다. 활성앱이나 사용자 작업을 종료하지 않는다. 원본 실행 중 자체PID만 감독한다. 타임아웃/오류시 가능한부분raw를 보존하고 자동반복하지 않는다. 미계측개발/리뷰시간을0으로 기록하지 않는다.

M2 config SHA256 `d3664dedd79d8dd97ee0355156ec4c5a31347c6112883aa60daef7fc08830b66`, F6미노출, UI변경0, whole_goal_complete=false, goal=active는 불변이다. 자료가 유효할 때 국소 신경 반응식의 항등 복원→조건부학습→전체BANC와물리→원래F1–F8 순으로 진행한다.
