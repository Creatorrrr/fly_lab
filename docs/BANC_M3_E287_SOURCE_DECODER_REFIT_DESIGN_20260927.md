# E287 — 변경된 source 동역학과 전체 운동 전달의 한 번의 재보정

2026-09-27 root 사전 설계. E286 이후 RF 국소 startup 실험을 종료한다. 현재 세션이 설계하고 실제 gpt-6-sol/xhigh가 구현한다. FIRST→같은Sol 첫수정(필요시)→SECOND→root 잔여수정 절차를 따른다. 구현 전 원본 예약0이며 과거 기각·비용/F1–F8/M2/E150/UI/F6는 유지한다.

## 큰 목표와 이번 변경

E267은 기존 간선의 합산 전 source 동역학을 바꿨지만 E270–E286에서는 E150에서 정한 운동뉴런→근육 전달을 고정했다. E277의 source 후보는 train 개선5.392%/LM 개발 기준에 미달했고, 실제 .5초 E279는 건강했으나 startup peak .04086948545>.04였다. E285/E286의 별도 RF 상태에서는 양의 방향 근육의 높은 전달 분기가 사용되지 않았다. 이 자료들은 고정된 전달과 바뀐 신호의 조합을 재검토할 근거이며, startup 오차가 F4/F5 실패의 원인이라는 증거는 아니다.

유일한 구조적 개발 변경은 **source 동역학을 바꾼 뒤에도 원래 값으로 고정했던 monotone 운동 전달을 함께 보정하는 것**이다. E277의 최종 `coefficients32`를 float64로 정확히 확장한1660×3 source 계수는 그대로 고정하고, 기존391MN/96근육의 비LTM·비singleton 전달 low/high slopes만 전체 train 자료에 맞춰 한 번 재적합한다. 새 source 계수·kappa·tau·신경 생리·CSR·부호·감각·신체·힘줄·외부 파형을 추가하거나 바꾸지 않는다. 시간/위상/목표 궤적 입력, additive 출력 bias, root 힘은 없다.

H287은 이 결합 후보가 원래 train20%/개발 비악화/다리별1.05 기준과 실제 startup에 필요한 여유를 함께 만족한다는 것이다. 경쟁 설명은 전달 보정으로 해결되지 않는 신경 시간 구조의 부족이다. 양성이면 같은 후보의 실제 BANC–body 판별로 즉시 진행하고, 음성이면 이 가족의 반복 fit·RF 여유 조정 없이 회로/신호 구조를 재검토한다. E277/E285 등의 과거 기각은 소급 승격하지 않는다.

## 원자료와 고정 후보

E270 원래 input freeze/원래 leg ownership/target, E277 freeze·source review·root 원자료 감사·candidate SHA 및 원래 leg별 final startup/teacher q/LTM를 고정한다. E269/E273 CUDA engine을 수정 없이 사용하여 고정 E277 source 계수로 startup101행과 teacher2001행을 각각 한 번 계산한다. 매 상태에 새 engine 하나, 한 번에 실제 engine1개, derivative=False/host=False, forward 총42000ticks다. bank/rate/original_activation/q/LTM를 명시적으로 host_copy하고 response를 해제한 후 engine을 해제·cleanup한다. 원래 E150 slopes의 q/LTM가 E277 저장 값과 abs1e-10로 일치해야 fit을 허용한다. 초기 bank/activation/original_activation=0과 명시적391MN/96근육/42관절/6LTM ID축도 확인한다.

고정 source 계수로 계산된 teacher 자료를 저장하되 optimizer에는 controls1..1200 및 startup1..99만 넘긴다. Development1201..2000은 최종 slope 후보를 저장·SHA 고정한 후 단 한 번 평가한다. 결과를 보고 다시 fit/선택하지 않는다. 미노출 F6는 전혀 사용하지 않는다. 비교 baseline은 E277에서 검증한 원래 neutral의 같은 q/target이고, E277 source+old decoder도 별도로 보고한다.

## 내부 여유의 근거와 불변 실제 기준

E281–E286의 .036/.0162를 새 fit의 조건으로 재사용하지 않는다. E279에서 관측된 actual−conditional의 q max=.0015168425596647772 및 RMS=.00013667128182327153에 고정 추가 여유.0001rad를 둔다. 따라서 새 내부 q peak는 `.04−.0015168425596647772−.0001 = .03838315744033522`, RMS는 `.018−.00013667128182327153−.0001 = .01776332871817673`다. 이는 한 기존 후보에서 얻은 경험적 여유이며 새 decoder의 전달 오차 상한 보장은 아니다. 실제 .5초 startup q .018/.04와 LTM .009/.025 및 F1–F8은 그대로다. LTM transfer/source가 고정이므로 LTM은 기존 내부 .009/.025로 별도 확인한다.

이 변경은 RF 정체값에 맞춘 기준 설정이 아니라 관측된 전달 오차에 근거한 prospective 방법 변경이다. E281–E286 판정은 그대로 유지하며 새 후보가 실제 startup에서 실패하면 추가 보행은 중단한다.

## 한 번의 전체 train 적합

기존 nonnegative continuous monotone 식 `low*min(bank,kappa)+high*max(bank−kappa,0)`를 유지한다. kappa/5·10·40ms bank/고정 해부학 membership/moment/rest는 원래 값이다. Slopes의 순서는 explicit free motor ×3filter×[low,high]이며 근육 바깥 연결은0이다. LTM와 singleton은 기존 canonical slopes와 original_activation을 exact 유지한다.

각 free motor/filter pair에 lower/upper 보조 변수를 둔다. `0≤lower≤low,high≤upper≤2`, 근육별 sum(lower)≥.500001, sum(upper)≤1.999999, motor별 sum(lower)≥1e-6/해당근육MN수다. 이로써 기존 .5..2 근육 slope budget과 per-MN minimum을 보존한다. 실제 slopes로 계산한 min/max도 최종 다시 검증한다. 기존 bounds를 잘라 맞추거나 해를 repair하지 않는다.

Startup99×42 affine q error의 모든 peak 행과 전역 RMS SOC 하나에 위 새 내부 여유를 적용한다. 다리별 임의 quota는 없다. Fixed LTM 결과는 원래 한도로 검사한다. 목적함수는 `mean(train_q_error**2)+1e-8*sum((x−x0)**2)`이며 x에는 slope와 lower/upper 모두 포함하고 x0은 기존 slopes/min/max다. Mean은1200×42 모든 원래 행을 사용한다. Sparse joint QP/SOC를 만들고 수학적으로 동일한 per-leg Gram/QR 압축은 허용하되 원래 dense-small 문제와 비교하고 실제 원자료 오차로 최종 판정한다.

Clarabel0.11.1 단 한 번, CPUthreads1/max_iter100/time_limit20s/tol1e-8. Retry/fallback/penalty sweep/source 재학습0. 원래 P/q/A/b, solver x/s/z/status/iterations/실제 시간 및 실제 계수·오차·물리 slope budget 검사를 보존한다. Solved/AlmostSolved 및 원래 primal residual≤1e-8를 요구한다. 최적성/가족 불가능성은 독립 dual 증명 없이 주장하지 않는다. 시간 종료·미인증해를 성공으로 처리하지 않는다. 과학 실행이 유효한 bounded rejection과 실행 invalid/incomplete를 분리한다.

원래 train RMS ratio≤.8, development≤1, 모든 다리의 train/development≤1.05를 모두 확인한다. 기존 startup 실제 기준과 새 방법 여유도 별도 보고한다. 출력은 fixed source f32 및 새 float64 slopes/kappa/membership/identity/hash, 조건부 startup/full teacher 예측과 명확한 selection이다. 후보 통합/자연 보행 판정은 아직0이다.

## 실제 보행으로 돌아가는 고정 경로

E287 조건부 선택을 통과한 단 하나의 source+decoder 후보는 별도 E288 runner에서 .005초 neutral exact 확인→원래 전체 checkpoint에서 실제 .5초 startup→같은 세션1.5초로 연속 진행한다. E272의 실제 건강성·forward/slip 비악화·eligible bout≥.3초 및 각 다리 완전주기 증가를 그대로 판정한다. 내부 q RMS 감소만으로 다음 단계를 대신하지 않는다. 실제 .5초 또는1.5초 음성이면 관측된 행동을 기준으로 구조를 재검토한다. 양성일 때 같은 후보10초 F3–F5/F2·미노출F6/F7/F8로 진행한다. E288 구현은 E287 개발과 병행해 준비할 수 있으나 원본 실행은 E287 선택/소스 검토 뒤에만 예약한다.

## 구현과 비용 경계

Sol 소유 파일은 `tools/banc_m3_source_decoder_fit.py`, `tools/fit_c_banc_m3_source_decoder.py`, `tests/test_c_banc_source_decoder_fit.py` 세 개다. 기존 소스는 읽기 전용, 본문 설계/기록은 root 소유다. 소스/입력/패키지/provenance freeze와 root source review를 요구한다. 생성 CPU120초/tiny solver20회·변수60이하, 원본/CUDA/fullsize0. 생성 affine 직접 검산, fixed 행·ID·train/development 격리, 실제 gate, partial failure/cost/engine 수명, 단회 solver 상한을 확인한다.

원본 source 승인 후 한 번 science50+cleanup10/supervisor60초, solver1/20초, engine2/forward42000/derivative0/physics0, RAM6GiB/GPU추가 추정4GiB/출력512MiB/free disk2GiB를 사전 예약한다. 현재 누적7864.040608399493/7905초·잔여40.959391600507초는 유지하며 이 문서는 배정 증액이나 실행 예약이 아니다. 필요한 다음 단계 배정은 기존 회계를 이월하여 실행 전에 명시한다. 사용자 전체 시간/토큰 예산은 지정되지 않았다. E288은 원래115초 내부/별도 감독 한도를 참고해 실제 소스/원본 계약을 준비하고, 새 예약은 아직0이다.
