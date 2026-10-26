# E278 — 실패한 source 학습 후보의 실제 전달 진단

2026-09-27 root 사전 설계. E277은 유효한 조건부 기각(train 개선 5.392% < 20%, LM development 미달)이며 이를 보존한다. E272의 승격 gate를 변경하거나 E277이 통과했다고 간주하지 않는다. 기존 goal-plan의 실패 뒤 제한된 진단 원칙에 따라, 아래 한 후보의 조건부 예측이 실제 재귀망·감각·신체에서 유지되는지를 구분한다. 최종 F1–F8과 M2/UI/F6는 그대로 둔다.

## 질문과 고정 비교

주장: source 계수의 조건부 운동 명령 개선만으로는 실제 폐루프 개선을 예상하기 어렵다. 후보는 E277 마지막 `candidate.npz`의 **coefficients32를 float64로 정확히 확장한 동일 1660×3 계수** 하나다. E275와 선택 경쟁하지 않으며 재학습·혼합·추가 반올림·gain 변경은 없다. E277의 여섯 `final-*-startup.npz`가 조건부 q/LTM 예측이다. 목표는 E103의 원래 99 startup 표본이며 기준 신체 기록은 E150/E257의 같은 neutral 전체 checkpoint와 원래 trace/native/recruitment다.

실제 변경은 E267 source-filter neural layer뿐이다. 원래 BANC CSR·노드·부호·생리·E150 decoder·감각·신체·adhesion·tendon·control 5ms·neural 0.25ms·physics 0.1ms를 고정한다. 실제 source rate로 RK4 history를 갱신한다. 교사·기록 stage/trajectory를 실제 실행에 주입하지 않는다. 고정 입력 예측과 폐루프 관측의 차이는 **전체 전달 불일치**이며, 단독 감각/재귀 경로의 인과 효과로 주장하지 않는다.

## 하나의 단계적 실행

1. E271 wrapper + neutral source 계수 `[1,0,0]`로 기존 전체 checkpoint를 복원한다. 초기 전체 snapshot은 source layer/schema/patch 이외 exact여야 한다. 원래 E150 control0..1 두 행, native 첫50 표본(정당한 NaN slip의 비트 포함), recruitment 두 행을 exact 비교한다. 어느 항목이 다르면 후보 생성·실행 0으로 종료한다. 신경/물리 생성 중 warmup/capture/settle 비용은 과학 tick과 따로 기록한다.
2. 기준이 exact일 때 새로 복원한 동일 전체 neutral 상태에서 E277 한 후보를 실행한다. 초기 source history=0, zero neural/control clocks를 확인한다. native observer는 별도 MjData에서 매100µs finite/upright≥.9/nonfoot0/body fault 없음 조건을 검사하며 첫 위반에서 중단한다. 모든 control은 preentry/advance/readout journal과 실제 clock을 남긴다.
3. candidate .5초(control0..100)에 원래 `strict_startup`을 적용한다. 원래 control1..99 qRMS≤.018/qmax≤.04, LTMRMS≤.009/LTMmax≤.025를 바꾸지 않는다. 불완전/실패면 더 실행하지 않는다. 동시에 동일 control1..99에서 실제 q/LTM 대 E277 조건부 예측의 RMS/max, 다리별 및 .1초 구간별 차이를 기록한다. 시작 표본0 또는 마지막100으로 원래99개를 대체하지 않는다.
4. strict startup과 health가 모두 통과한 경우에만 **같은 연속 세션**을 1.5초(control300/native15000)까지 실행한다. E257의 `short_health`, `short_forward`, `short_window_slip`, 원래 gait analyzer를 그대로 사용해 E257 baseline의 같은 .5≤t<1.5 구간과 비교한다. 전진·하중 slip·최장 적격 bout·다리별 완전 주기 수를 각각 보여 준다. 이 비교는 설명용이며 후보 채택·10초/F6/회귀 실행을 열지 않는다.

## 관측 뒤 결정과 종료

- Neutral 불일치 또는 기술/자원 오류: source wrapper/runner 또는 실제 실행 차이를 먼저 진단하며 생물학적 반증으로 세지 않는다. 자동 원본 재시도는 없다.
- 완전한 .5초에서 실제 대 조건부 q 차이 RMS>.0018 또는 max>.004, LTM 차이 RMS>.0009 또는 max>.0025이면 `material_prediction_mismatch=true`로 기록한다. 원래 startup 예산의 10%를 사용하는 **다음 설계 선택용** 기준이며 성공/최종 보행 gate가 아니다. 이 경우 다음 수정은 실제 피드백 문맥을 반영한 학습/검증 구조를 우선 검토한다. 작은 차이는 실제 보행 성공을 뜻하지 않는다.
- 예측 차이는 작으나 원래 startup이 실패하면 제약 경계와 실제 수치 전달을 검토한다. 기준을 느슨하게 하거나 후보를 줄여 재시도하지 않는다.
- .5초가 건강하고 strict startup이 통과하더라도 1.5초 품질이 좋아지지 않으면 점별 교사 명령 목적함수와 실제 지지·진행·협응 사이의 연결을 다음 설계 대상으로 삼는다. E89 조건부 위상 정렬 뒤 E90 실제 실패를 함께 검토하며 또 다른 정렬/solver 반복으로 대체하지 않는다.
- 모든 분기에서 E277의 조건부 기각과 E272 미실행은 유지한다. 본 진단은 F2 저장 미래/차단, F3–F8 통과, 생물학적 기전 확인을 주장하지 않는다.

## 구현·원자료·자원

Root 설계 → 기존 실제 `gpt-6-sol/high` 에이전트 `e265_bridge_current_impl` 구현 → root FIRST → 필요 시 같은 Sol 첫 수정 → root SECOND → 남은 수정 root 순서다. 기존 neural/physical 수학을 고치지 않는 통합·계측 작업이므로 high를 쓴다. 새 runner와 생성 테스트만 소유하며 과거 소스·raw·계획을 수정하지 않는다. 서브에이전트는 fake CPU 생성 검산만 허용한다. 원본/CUDA/body는 root source review/freeze 뒤 한 번만 실행한다.

초기/최종 전체 checkpoint, trace/control 명령, 실제391MN+51source의 rate/adaptation/drive, source history, E150 bank/activation, native 접촉/슬립, partial/error/ownership cleanup, 원본/설치 소스 SHA와 비용을 보존한다. SourceFilter의 nonneutral `_current()`는 금지되어 있으므로 관찰 시 이를 호출하지 않는다. 관찰 불가/불완전 배열을 0이나 pass로 바꾸지 않는다. 검산에는 neutral mismatch→candidate0, startup fail→no continuation, measured health stop, readout/constructor/storage/close failure와 transferred ownership 정리를 포함한다.

현재 원본 보수 누적7761.5885750995985초/배정7905초/잔여143.41142490040147초 안에서 **원본 한 호출 최대140벽초**, 내부 과학 작업110초와 최종 정리 여유30초를 동결한다. candidate 최대1.5초 + neutral .005초, 과학 neural6020ticks/physics15050steps, body 최대2개. 실제 생성 setup 추가 비용은 별도 계측한다. RSS6GiB/추가 CUDA4GiB/output2GiB/최소 free disk4GiB, thread1, GPU 단독이다. source review 이전 실제 예약/원본 실행은 0이다. 실측 supervisor wall을 기존 누적에 더하며 실패 비용도 남긴다.
