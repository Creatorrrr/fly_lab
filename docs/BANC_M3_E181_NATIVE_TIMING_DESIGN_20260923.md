# E181 실제 보행 결과로 모집 시간응답 최적화

상태: 원본 실행 전 설계. 기존 finite-goal/F1–F8을 유지한다. 직전 E180은 유효한 실제 비교와 평가기 재현을 완료한 progress이며, 1% canonical 혼합 후보는 slip 증가와 전진 지속성 미달로 기각됐다. 같은 alpha sweep을 하지 않는다. 현재 세션이 설계하고 실제 gpt-6-sol/high가 구현·첫 수정을 하며, 부모 FIRST/SECOND 리뷰 후 잔여 수정은 부모가 맡는다.

## 가설과 범위

H181: 상수 MN 입력에 대한 모집 이득을 유지하면서 기존 5/10/40ms filter 성분의 시간 분포를 다리별로 조절하고, 실제 전체 BANC–FlyBody 전진 속도 부족을 목적함수로 선택하면 전진 끊김을 줄일 수 있다. 경쟁 설명은 시간응답 교정만으로는 회로·신체의 지지 전환 문제를 해소하지 못하거나 초기 보호 조건 때문에 유용한 변화가 불가능하다는 것이다.

E146/E148/E169/E175는 저장 신호의 조건부 명령 적합이다. E180은 실제 물리에서 한 고정 방향을 비교했지만 최적화는 하지 않았다. 부모의 소스 검토와 gpt-6-sol/medium의 독립 감사에서 제안한 가족의 실제 물리 목적함수 최적화 선례는 확인되지 않았다. 새로운 생물학적 법칙이라는 뜻은 아니다. 최종 runtime은 기존 신경 상태와 감각만 사용한다.

## 변경과 불변 조건

원래 E150 중립 checkpoint를 모든 팔의 출발점으로 쓴다. 각 해부학적 MN–근육 쌍의 세 bank low/high 계수 L,H에 c=min(L,H), a=L-c, b=H-c를 정의한다. 성분 x에 R(x,theta)=x*w*sum(x)/sum(x*w), w=(tau/.01)**theta_leg를 적용한다. 합이 0이면 0을 유지한다. 새 low=R(c)+R(a), high=R(c)+R(b)다. theta가 0인 다리는 산술 없이 원본을 복사한다.

각 MN low/high 합과 상수 입력 극한의 출력은 유지된다. min 합은 감소하지 않고 max 합은 증가하지 않아 원래 근육 [0.5,2] 예산·MN 하한을 보존한다. 곱셈 재가중은 원래 0인 성분의 support도 보존한다. 부동소수점 오차는 별도로 검사하고 기존 adapter 생성 검증을 통과해야 한다. fixed19는 bit-exact 복사한다. 신경 노드·연결·동역학, MN 소유, tau/kappa, moment/rest/pool, 몸체와 입력은 고정한다. 움직이는 상태에서 계수를 바꾸지 않으며 외부 위상·궤적·CPG·root/힘 보정을 추가하지 않는다.

## 실제 목적함수와 제한 최적화

개발 평가 창은 [0.5,3)초다. 기존 biological gait evaluator의 실제 Thorax 위치 미분/몸체 전방 투영/20ms Gaussian 속도 v를 사용해 J=mean(max(0,0.5-v)) mm/s를 최소화한다. 임의 단위의 가중합은 만들지 않는다. 기존 초기99 명령 RMS<=.02rad/max<=.04rad 및 LTM 보호, native 건강성, 초기 E150 대비 전진·net/path 각각95% 이상, 하중 가중 slip 증가 없음이 선택의 필수 제약이다. 결측/불완전/NaN은 부적격이며 0점으로 채우지 않는다.

고정 순서 lf,lm,lh,rf,rm,rh의 한 번의 좌표 순회를 한다. 각 축 시작의 동일한 현재 최선 theta에서 +/- 방향을 둘 다 제안하며, 초기 step=.125, box |theta|<=.125다. 원본 저장 startup 조건의 실현 가능성 확인에만 step .125,.0625,.03125,.015625의 최대4회 backtracking을 쓴다. 첫 feasible 비중복 제안만 실제3초 실행한다. 한쪽 결과로 반대쪽 제안을 바꾸지 않는다. 두 결과 중 제약을 만족하며 현재 최선 J보다 max(1e-6,0.01*J) 이상 감소한 결과만 다음 축의 중심으로 선택한다. 동률이면 J, theta의 L2 norm, plus/minus 고정 순서로 판정한다. 선택 자체는 탐색이며 최종 채택이 아니다. 실제 startup이 실패하면 해당 방향 부적격으로 남기고 다른 사전 방향만 계속한다. 기술 오류가 발생하면 캠페인을 중단하며 자동 재시도는 없다.

비교군 theta=0을 새 runner로 먼저 실행하고 E180 baseline의601 control/MN/bank/activation 및30000 native prefix를 직접 재현해야 후보를 실행한다. 한 축당 최대2개의 새 native 후보, 총 baseline1+후보12=13회다. 같은 좌표의 추가 sweep/반복 순회/사후 step 추가는 이번 계약에 없다. 최종 최선이 개발 진입 조건(전진·net/path95%, slip 비증가, longest bout>=.3s, 여섯 다리 적격 완전 주기>=1)을 모두 통과할 때만 별도 사전 기록 후 동일 후보10초 검증을 설계한다. 미달이면 작은 J 감소를 자연 보행 성공이나 F3–F5 통과라고 하지 않는다.

## 실행과 검증

새 도구 tools/optimize_c_banc_m3_native_timing.py와 tests/test_c_banc_native_timing.py만 Sol이 소유한다. E180 및 기존 runtime/평가기/제품 파일은 읽기 전용이다. E180의 검증된 native 계측·startup·상태·시계·평가 helper를 재사용하되 E180 gate를 위조하거나 monkeypatch하여 다른 팔을 실행하지 않는다. 새 restore/runner 계약과 E181 provenance를 사용한다.

도구는 순수 tilt/사전 startup/목적함수/선택 함수를 제공한다. CLI baseline은 한 팔, CLI axis는 해당 축의 최대 두 팔을 각 별도 자식 process에서 순차 실행한다. 축0에는 부모 baseline 재현 승인, 축1–5에는 직전 축 결과의 정확한 hash와 원래 baseline 승인/hash가 필요하다. 사용자 지정 theta나 임의 팔을 실행하는 CLI는 제공하지 않는다. 부모는 각 축 사이 결과·실패·누적 자원을 확인한다. 입력/소스/원본 보호 해시와 새 source review가 없으면 원본 배열을 읽거나 모델을 만들지 않는다. 출력은 create-only이고 의도·요청·성공·실패/시각/중단/비용을 남긴다.

생성 자료 검증은 zero byte identity·상수응답/합/support/fixed 보존·기울기 예산·잘못된 입력, 실제 속도 부족 산식, 제약 우선 선택/비교 불가능 결과 거부, 동일 중심 +/- 제안, startup backtracking·중복·순서·횟수, partial/실패 raw 보존과 CLI import 경계를 포함한다. 부모는 별도 scalar 방식으로 성분 재분배와 raw neural→bank→activation→명령 및 native metric을 검산한다. 작성자 성공 보고만으로 source gate를 만들지 않는다.

## 자원와 원래 목표

기존 원본 보수 누적3533.5252058999613초와 단계 배정3656초를 기록 그대로 승계한다. 사용자 지정 전체 시간/토큰 예산은 없으며 이번 새 단계의 최대13*110초 native 감독과30초 저장 계산을 prospectively 배정한다. 누적 배정은 ceil(3533.5252058999613)+1460=4994초다. 이전 사용량을 초기화하지 않는다. 새 원본값 노출/모델/fit은 현재0이다.

native 자식별 내부100/외부105/감독110초, 3모델초/12000neural ticks/30000physics steps, 자동 재시도0. 축 driver는 최대270초, CPU1/GPU1 순차, 동시 owned GPU job1개다. 출력 arm별1GiB/전체14GiB 이하, 시작 free20GiB. 실제 측정 비용·setup/capture·settling·부모 생성/해시 검산 비용은 분리한다. 소스/기술 오류나 한도 초과 시 affected campaign은 중단하여 증거를 보존한다. 활성 UI/서버, M2, 과거 실패, F6 미사용 초기 상태는 보존한다. 이번3초 탐색으로 F1–F8을 대체하거나 목표를 완료 처리하지 않는다.
