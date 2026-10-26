# E270 — 보행 후보 한 개를 위한 제한된 source 계수 적합

2026-09-27. E269 엔진 개발과 독립적으로 root가 학습 계약을 설계한다. E268의 실제 단계 자료를 사용한다. 이 문서는 원본 fit 실행 승인이 아니라 구현 사양이며, 최종 소스·입력·한도 SHA는 root 리뷰 후 첫 실행 전에 고정한다. 과거 실패/F1–F8/M2/E150/UI/F6와 원본누적7187.140763999607초를 보존한다.

## 사전 관측과 변경 이유

root는 저장391MN rate에 **실제 두 adapter 클래스**의 비차단decode만 적용했다(신경/신체생성0,fit0,2100control decode). 기존 E150 decoder identity가exact다. startup99의qRMS .017727092685/max .039457797987,LTMRMS .001363757230/max .013190800279로기존한계안이다. 다리별qRMS는 [.022730,.010844,.020237,.019455,.012216,.017720]이므로 `.018`을각다리에별도로강제하면원래전역계약을바꾸게된다. **전역RMS를보존한다.**

E150관절moment는다리별로분리되고각MN은한근육만소유한다. source간선수는LF/LM/LH/RF/RM/RH=[300,282,254,261,268,295],모수수=[900,846,762,783,804,885]다. 전체4980차원의거대한SLSQP를피하고,이조건부분리구조에서각다리를고정순서로한번씩적합해한후보를합성한다. 재귀전체망에서다리가분리된다는주장은아니다.

기준E150전달의교사명령오차는train1..1200 RMS .161324729569rad,이미관찰한development1201..2000 RMS .097982675918rad다. 개발구간은후보선택/gradient/반복정지에사용하지않고최종후보한번의검사에만사용한다. source식·E150전달과gain/τ/신체/감각은E269설계대로고정한다.

## 원본 fit 전 필수 gate

1. 생성CUDA: E269조건부RK4/원래E150전달/3계수analytic B와q/LTM Jacobian이CPU oracle·중심차분에맞고경계/부분완료/메모리검증을통과해야한다. 실제 Sol FIRST→필요한첫Sol수정→rootSECOND→root잔여수정.
2. 원본한번의초기계산에서중립c=[1,0,0] startup100/teacher2000의f64응답을E268nativeendpoint와비교한다. rate와adaptation최대절대오차각≤.002,실제E150기록rate decode참조에대한q≤.00005rad,LTM≤.00001,activation≤.005. recorded rate를직접decode한E269출력은같은actualadapter참조와q/LTM≤1e-11,activation/bank≤1e-10. 허용치를결과뒤완화하지않는다.
3. 입력순서·source/edge/motor/근육·실제inverseτ/decoderSHA·모든E268카운터/valid/rawSHA를검사한다. 중립startup의원래전역조건을통과해야한다. 한gate라도실패하면optimizer0회로오류/부적합을보존한다.

## 단 한 번의 최적화

LF→LM→LH→RF→RM→RH 순서,다리마다scipy SLSQP한호출만한다. 초기값은그다리의neutraltriple이다. maxiter60,ftol1e-9,unique coefficient 평가최대80회/다리,공유fit벽시간180초/다리최대30초. fit시간은엔진준비·forward·bank·Jacobian·CPUoptimizer·저장까지포함한다. 비용상한이면그때까지의최선유효후보를보존하고다른초기값으로재시작하지않는다. 캐시는동일f64모수bytes의같은응답을재사용하며새평가/실패/미완료/host전송도구분해기록한다. 원본실행전체는추후동결하는≤290벽초(개별300이하)에포함한다.

각다리목적: `mean((q_teacher[1:1201,leg]-target)^2)/baseline_leg_MSE + 1e-4*mean((c-neutral)^2)`.

기준leg_MSE는E269중립응답에서한번계산하고고정한다(저장actualdecoder값과차이를기록). gradient는압축B와actualE150moment를통과한것이다. 임의phase/사인파/선행teacher/수동joint값은신경런타임입력으로넣지않는다. teacher는이offline학습의목표일뿐이다.

범위: c≥0,各triple합 `.125*(1+1e-6)` 이상 `8*(1-1e-6)` 이하. 최종float32변환에서원래[.125,8]을보장할내부여유며실제범위는바꾸지않는다. SLSQP범위는各c [0,8]이고triple합의명시적선형constraints와analyticJacobian을사용한다.

시작 보호 controls1..99는 E103의 원래42q/6LTM targets다. 현재 다리 밖의 이미 선택한 후보와 아직 처리하지 않은 중립 계수의 오차를 고정하고, 전체42q RMS≤.018 및 전체6LTM RMS≤.009를 그대로 제약으로 사용한다. 구체적으로 `sum(current_leg_error²)+fixed_other_legs_SSE <= total_element_count*limit²`다. 전체q max≤.04와 LTM max≤.025도 유지한다. 각joint의 max제약은99점 절대값의 최대치7개, LTM은1개로 표현할 수 있다. 동률이면 첫 argmax의 미분을 일관되게 선택한다. 기존 전역 한도를 각 다리의 독립 RMS 한도로 바꾸지 않는다. 반올림/근사용 selector여유는 상대1e-4이며 RMS/max를 `limit*(1-1e-4)` 안에 둔다. 수치 조건이 나빠도 다른 계수 범위나 τ를 탐색하지 않는다.

다리별완료상태와무관하게그다리의全유효평가중시작보호·범위조건을만족하고중립보다학습MSE가나쁘지않은것만선택한다. 그중목적이가장작은첫평가를선택한다. 새후보가없으면그다리neutral을보존한다. 다리선택직후다른다리에고정할startupSSE를갱신한다. 경계값을거짓pass로반올림하지않는다. 직전다리의선택이future다리결과를보고바뀌지않는다.

## 최종선택·가설 판정

합친f64c를**한번**float32로변환하고다시f64로올린실제후보coefficients를E267범위validator로검사한다. 반올림후예측을한번재평가하며이결과로refit하지않는다. 최종 모든q/LTM의 최대오차와 전역RMS가 원래startup strict조건을 통과해야한다. 학습전체qRMS≤중립의.8,개발전체qRMS≤중립,각다리학습/개발qRMS≤중립의1.05를조건부지지기준으로동결한다. 학습만성공하고개발/초기보호가실패하면미달로보존하며body로자동승격하지않는다. optimizer상한종료는전역불가능성이나수렴증거가아니다.

조건부지지가있으면한같은계수후보를기존zero/neutral신경상태에서전체BANC와감각·E150monotone·실제신체에연결한다. .5초startup 통과→1.5초 전진/slip비악화,bout≥.3초·주기개선→지지시10초/F1–F8로연속진행한다. 해당native패치·checkpoint/미래·motor cut과물리조건의구현은별도root설계/검토와사전한도를거친다. 외부phase/teacher/CPG/root수정은최종런타임에없다.

## 소유와 실행 경계

실제gpt-6-sol/high가 `tools/fit_c_banc_m3_source_coefficients.py`, `tests/test_c_banc_source_coefficients_fit.py` 두파일만구현한다. 엔진파일은다른Sol이소유하므로수정하지않는다. root FIRST→같은Sol 첫수정→SECOND→root 잔여수정 절차다. CPU의작은가짜engine/analyticquadratic만사용한helper·selection·constraint·budget검사는허용한다. scipy실행이있으면generatedsolver호출/iteration/wall을명시한다. CUDA/원본E268/전체BANC/신체/원본fit실행·다른에이전트는금지한다.

runner는명시적freeze·source-review·input SHA 없이는original을거부한다. finalcandidate/매leg최선raw/checks/cost를성공단언전저장하며부분실패를보존한다. root가원본source gate와예약을완료할때까지원본실행은0이다. 현재새원본예약0,현재남은37.859236000393초에전체fit/body가들어간다고가정하지않는다.
