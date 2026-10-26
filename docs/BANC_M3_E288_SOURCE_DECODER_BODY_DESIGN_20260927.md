# E288 — 동일 source+decoder 후보의 실제 startup와 짧은 보행 판별

2026-09-27 root 사전 설계. E287 결과를 보기 전에 실제 도착점을 고정한다. E287 개발과 병행하여 actual gpt-6-sol/high가 새 runner/test를 준비하지만, 조건부 selection/원자료 root 검토와 이 runner source 승인 전 원본 실행0이다. Source와 decoder를 함께 평가하는 한 후보이며, E277 단독 source 후보의 과거 기각은 유지한다.

## 후보와 초기 상태

후보는 E287에서 단 한 번 fit한 float64 monotone slopes와 동일 E277 float32 source coefficients, 원래 kappa/tau/membership/모든 anatomical calibration이다. E287 selection이 원래 train20%/development≤1/다리별1.05와 사전 내부 startup 여유 및 slope 물리 범위를 모두 통과해야 한다. E287 final candidate/predictions/result/root audit SHA를 freeze한다. 현재 후보 파일은 아직 없으므로 placeholder를 실제 input으로 취급하지 않는다.

E279/E272의 graph/E150 neutral 전체 checkpoint/원래 physical model·physics·sensory/controller·native contact observer를 재사용한다. Control tick0, neural tick0, rate/adaptation/drive=0, output_mask=1, decoder bank/activation/original_activation=0을 엄격히 확인한다. Neutral arm은 원래 decoder와 neutral source 계수로 생성하여 .005초 trace2/native50/recruitment2 및 source layer 제외 전체 checkpoint를 exact 대조한다.

Candidate arm은 원래 checkpoint의 MotorTimescale 기반 신체·신경·anatomical adapter를 복원한 후 기존 `MonotoneRecruitmentTrial`에 새 slopes와 원래 tau/kappa/membership를 넣고 `SourceFilterTrial`에 고정 source 계수를 넣는다. Monotone adapter를 다른 monotone adapter 안에 중첩하지 않는다. 기존 restore에서 identity 검사를 제거하거나 old hash를 새 값으로 속이지 않는다. 새 adapter identity·patch를 정상 생성하고 모든 실제 상태를 비교한다. Candidate 초기 whole snapshot에서는 승인한 parameter/identity/해당 patch-derived metadata만 달라야 하며 모든 body/network/원래adapter state/controlheld/zero bank·activation은 원래와 exact다. 전체 adapter나 patch를 통째로 비교에서 제외해 상태 차이를 숨기지 않는다.

각 neutral/candidate의 neural identity, 새 adapter identity 및 두 factor SHA는 실행 전에 pure metadata로 계산·고정한다. 원래 source provenance/CSR/부호와 E150 calibration은 유지한다. 디스크 저장 representation은 E279의 tuple/list 정규화만 재사용하고 배열 dtype/bytes 또는 수치 tolerances는 완화하지 않는다.

## 연속 실제 평가와 판정

기존 E279 `run_arm`, native observer/ledger/raw publication/partial-error 처리/health 검사 및 analyzers는 읽기 전용 재사용할 수 있다. 새 dependency 구현과 gate/명시적 종합 판정은 신규 runner에 작성하며 기존 모듈 전역 monkeypatch는 하지 않는다.

1. Neutral .005초 exact 실패이면 candidate 생성0. 실제 CUDA packed 전체158706 BANC와 FlyBody를 사용한다.
2. Candidate 원래 시작에서0.5초까지 실행하여 native 매100µs finite/fault/upright≥.9/비발접촉0을 확인한다. Control1..99 실제 qRMS≤.018/max≤.04, LTM RMS≤.009/max≤.025를 그대로 적용한다. E287 조건부 q/LTM101행과 actual의 차이도 원자료로 저장한다. 전달 차이는 경험적 내부 여유의 보장 여부를 평가하는 관찰값이다.
3. 실제 startup 실패 또는 건강성 실패면 완료 prefix/신경·물리 비용/상태를 보존하고 중단한다. 통과할 때 동일 session을1.5초(control300)까지 계속한다. Reset/restart/seed교체/새 fit은 없다.
4. 완성된0.5≤t<1.5초 구간에서 기존 E257 `short_health`, `short_forward`, `short_window_slip`, 기존 biological gait analyzer의 eligible bout와 완전주기를 사용한다. Baseline 같은 구간의 전진1.6790085779712631mm 이상, supported slip 비악화, longest eligible bout≥.3초 및 baseline보다 짧지 않음, 여섯 다리 각각 baseline의0보다 많은 완전주기를 요구한다. Missing/incomplete metric은 pass로 간주하지 않는다. 기존 metric의 실제 key/구간은 저장 baseline과 대조한다.
5. 위의 동일 후보 조건을 모두 통과한 경우에만 다음 별도 예약의10초 F3–F5/F2·미노출F6/F7/F8로 진행한다. 1.5초 주기 증가를 최종 다리별10주기·phase/CV 판정으로 대체하지 않는다. 음성이면 실제 관측을 기록하고 RF startup 반복으로 되돌아가지 않는다.

최종 result는 `run_valid`, `neutral_exact`, `actual_startup_pass`, `short_gait_pass`, 각 metric/check/완료범위, next stage eligibility를 명시한다. Natural gait/F1–F8 complete/제품 채택/F6 노출은 이번 실행에서 항상false다. 무관한 과거 E278/E279 schema/명칭으로 이 결합 후보의 provenance를 덮어쓰지 않는다. 재사용 arm receipt가 기존 schema를 가지면 재사용 경로임을 별도 기록한다.

## 구현·검증·자원

신규 소유 파일은 `tools/verify_c_banc_m3_source_decoder_body.py`, `tests/test_c_banc_source_decoder_body.py`다. Actual gpt-6-sol/high→root FIRST→필요시 같은Sol 첫수정→SECOND/root잔여수정. 생성CPU90초, 원본/CUDA/physics0. Fake dependencies 및 pure metadata 검산으로 새 slope identity/원래 상태 불변, fixed LTM/singleton, 초기 실패 시 candidate0, startup실패 시100controls정지, 통과 시같은session300controls, 실제6leg complete-cycle/slip/bout gate, missing metric 거절, partial/clock/save/cleanup primary 실패·비용 보존을 확인한다. 기존 의존 소스는 변경하지 않는다.

원본은 E287 선택 및 root 양쪽 source review 후 **단 한 번**, science85초/finalization30초/internal115초/supervisor125초, RAM6GiB/GPU추가4GiB/output2GiB/free disk4GiB다. Body2개, science neural≤6020ticks/physics≤15050steps; warmup/capture/settle은 기존 ledger에서 별도 실측한다. 원래 caller process와 descendants를 감독하고 종료를 확인한다. 전체 프로그램 원본 누적7864.040608399493/7905초는 본 설계에서 변경하지 않으며 현재예약0이다. E287의 실제 비용까지 이월하여 필요한 배정·예약을 실행 전에 별도 명시한다.
