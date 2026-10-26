# E209: 부호별 입력 기억을 쓰는 국소 운동뉴런 반응 기저

F1–F8 finite-goal은 active·미완료다. E208은 원래 E150 전체 BANC·신체10초와 native current를 exact 보존하여4입력 채널을 확보했다. 기존 국소 반응 가족에서 닫힌25MN 전부에 양의 recurrent 입력이 있으나 음의 입력 RMS가3.95–1461배 컸다. E203–E207의 같은 rectifier 보정, solver/초기화 반복은 새 후보를 만들지 못했다. 따라서 순전류에 작은 bounded correction을 추가하는 대신, **각 MN이 자신의 실제 부호별 입력과 인과적 입력 기억에서 반응을 생성하는 모델**을 설계한다.

부모 설계와 실제 gpt-6-sol/medium의 읽기 감사를 병행했다. E146은 발화율 뒤의 근육 필터, E170/E175는 저장 rate의 양수 근육 전달 가족, E97의 softplus는 학습용 근사였다. 검토 범위에서는 아래 E/I 입력기억→positive basis→순간 발화 구조를 이미 구현·실패한 사례를 찾지 못했다. 이는 유리함의 증명은 아니다. E206의 SOC 구성/Clarabel 계상 일부는 재사용 가능하지만 기존 fitter 전체는 차원과 가족이 달라 재사용하지 않는다.

## 변경 가정과 지켜야 할 경계

변경 대상은 비LTM 자유343MN의 국소 순간 반응식이다. 기존 BANC 노드·edge·부호·근육 소유권을 보존하며 각 MN의 입력은 자신의 실제 signed recurrent 합과drive뿐이다. 외부 time/phase/teacher/target은 실행 입력에 없다. 교사는 뒤의 학습 목적함수에만 사용한다. 반응은 국소 입력 값에 대한 학습된 비선형 함수이므로 synaptic 부호 원자료가 같다는 사실을 단조로운 생리 효과가 보존됐다는 주장으로 확대하지 않는다.

새 phi에는 기존 adaptation z가 들어가지 않는다. 따라서 이343MN의 기존 z feedback을 입력기억 모델로 **대체하는 가설**이다. z 배열을 계산·저장하더라도 기존 적응 기전이 유지됐다고 말하지 않는다. 나머지 신경과48LTM MN은 원래 식을 쓴다. 전체 신경 피드백이 바뀌므로 조건부 입력에서의 성과는 실제 폐회로 성과가 아니다.

기저 계수가 주어졌을 때 원래 membrane `dr=(phi-r)/tau`를 유지한다. 이후 constrained fit에서는 원래 anatomical pool+10ms activation을 우선 검토하여 조건부 목표식의 선형성을 확보한다. 이는 E150 monotone decoder 제거와 신경 반응 변경의 **결합 변화**다. 같은 두 신경 출력(원래/새 반응)을 두 decoder(monotone/canonical)로 계산하는2×2 readout 대조를 이후 fit의 필수 산출물로 한다. 결과를 신경 단독 효과라고 부르지 않는다. 이 단계 E209는 basis 계산과 원래 rate의 canonical readout 대조까지만 하며 적합/채택은 하지 않는다.

## 고정 국소 모델

신경dt=.00025s, RK4 네 단계의 실제 입력을 E208에서 그대로 읽는다. E=positive recurrent>=0, I=negative magnitude>=0, d=실제drive다. 각391MN에 대해 원본 native160000행의 최대값으로 고정 scale을 한 번 계산한다: `sE=max(1,log1p(max E))`, `sI=max(1,log1p(max I))`, `sD=max(1,max abs(asinh(d)))`. `x=[log1p(E)/sE, log1p(I)/sI, asinh(d)/sD]`다. 입력을 사후 clip하거나 미래 확인 상태로 scale을 다시 맞추지 않는다.

세 인과 기억 `dh/dt=(x-h)/.016`은 원래 RK4와 같은 네 단계 입력으로 적분하며 초기값은 모두0이다. descriptor `f=[xE,xI,xD,hE,hI,hD]`는 각 RHS의 그 단계 상태에서 계산한다. 각 control endpoint descriptor는 마지막RK4입력의 x와 완료된h를 쓴다. row0 descriptor는 첫 native입력의 x와h=0이다. 이 endpoint 정의는 prototype 선택용이며 실행 중에는 항상 실제 해당 RK stage의descriptor를 쓴다.

각MN의 prototype16개는 원본2001 endpoint descriptor로 deterministic farthest-point sampling한다. 첫 prototype은row0, 이후 이미 선택한 prototype까지의 최소 제곱거리가 가장 큰 행을 선택하며 동률은 가장 작은행이다. 모든 거리가0이면 row0을 반복한다. `sigma2=max(1e-4, median(각 prototype의 엄격히 양수인 최근접 제곱거리))`; 양수 거리가 하나도 없으면1이다. 중심/폭/scale은 이 사전규칙으로 고정하며 fit 변수나 실행 중 적응값이 아니다. 이 기하학적 전처리는 데이터를 사용하는 모델 설정임을 계상하되 원본 optimizer 호출로 세지 않는다.

`l_j=−||f−center_j||²/(2*sigma2)`, `p_j=exp(l_j−max l)/sum exp(l−max l)`로 **partition of unity**를 구성한다. 원점에서도 양수인 Gaussian만으로 무입력 발화를 만들지 않도록 `g=−expm1(−8*(xE+abs(xD)+max(hE,0)+abs(hD)))`를 별도 적용한다. E와 drive 및 그 기억이 0이면 g=0 exact다. 순수 억제 입력만 있고 과거 흥분/drive도 없을 때는 발화하지 않는다. g는 0..1이다. 동결 전 부모 수학 검토에서 RK 중간 단계의 hE는 불연속 입력 아래 음수가 될 수 있음을 확인했다(초기 h=0, x1>0, x2=0이면 h3=−.25*(dt/tau_h)^2*x1). 따라서 위 `max(hE,0)`를 반응식의 명시적 일부로 정한다. 저장 h나 완료 상태를 사후 clip하지 않으며, 완료 hE 비음수와 기저 경계는 별도로 검사한다.

새 순간 반응은 `phi=cap*g*sum(beta_j*p_j)`, 각beta∈[0,1]이다. p의 합이1이므로 phi는0..cap이다. 단순히16개의 가중치를 각각cap 이하로 제한하는 식과 다르다. 초기r은 원래 E150값이며 임의로 바꾸지 않는다. runtime feature에시간축 인덱스나prototype의원래행 인덱스를 입력하지 않는다.

## 두 번의 조건부 CUDA 계산과 정확한 선형성

첫 CUDA kernel은 고정 입력/scale에서 h와2001 descriptor를 만든다. CPU가 고정규칙으로 중심/폭을 만든 뒤 두 번째 kernel이 같은h를 재생하면서 각기저의 `dB_j/dt=(g*p_j−B_j)/tau`, 초기B=0을 같은RK4로 적분한다. 원래 초기r의 homogeneous 부분 H는 `dH/dt=−H/tau`로 적분한다. 따라서 임의의 고정beta에 대해 `r=H+cap*sum(beta_j*B_j)`다. 고정 입력과 중심/폭/scale/gate일 때만 이 선형성이 성립하며 실제 BANC 피드백 전체가 선형이라는 뜻이 아니다.

두 번째kernel은 결정적 검산용beta_j=(j+1)/17을 사용한 scalar rate도 독립으로 함께 적분한다. 이는 학습된 후보가 아니다. 완료된2001 rate와 B/H 조합의 차이가atol2e-10/rtol1e-11 이내여야 한다. 첫/두번째kernel의 endpoint h/descriptor는같은수학/연산순서로bytes exact여야 한다. B>=−1e-12, sum B<=1−H_factor+1e-10, H_factor>=0, 모든state finite를 검사하고 실패값을clip해고치지않는다. 실제 r의cap 경계도 generated 검증에서 확인한다. native f32 CUDA 전체 integration과의 일치는 나중에 실제 후보를 구현할 때별도로 측정한다.

원래391MN의모든입력 basis를 계산해mapping을 보존하되, 이후학습은E203의동결free343/fixed48mask만사용한다. conditional fixed48은E201의실제원본rate를 쓰고, 최종 runtime에서는고정원본rate파일을재생하는것이아니라현재실제입력에원래native식을적용해야한다.

canonical decoder의 affine 준비자료도 생성한다. `rate_intercept[:,free]=H[:,free]`, fixed48열은E201원래rate다. 원래pool·rest·moment·initial original activation과10ms EMA로 `q_intercept`를 계산한다. 각MN 기저D는 `cap*pool[own_muscle,mn]*EMA10ms(B_j, initial0)`이며 `q=q_intercept+.02*moment @ (소유근육별 sum(D*beta))`다. row0 q는원래held joint_angles를유지한다. D는2001×391×16이지만fixed48열은0으로저장해학습불가를명시한다. 각MN는원래한근육만소유해야하며다른근육연결을만들지않는다. fixed LTM/tendon은conditional comparator와exact 유지한다.

원래E150 rates를canonicalpool로만읽은readout-only대조의command/startup/각다리metrics도저장한다. 별도fit이아니며보행향상으로판정하지않는다. probe beta의affine q와직접canonical decoder를독립경로로비교하며 atol=2e-12/rtol=2e-12를 적용한다. 고정 LTM 비교는 같은 원본 canonical 계산의 결과와 bytes exact여야 한다. 원본 native trace와의 오차도 별도 보고하여 계산 경로의 반올림 차이를 숨기지 않는다. 새neural+monotone대조와실제적합/후보판정은다음사전설계에서수행한다.

## 실행유효성·후속·소유권·자원

H209는 이입력기억모델의조건부표현을native stage입력에서비음수·cap·무입력경계와정확한affine readout으로구현할수있다는구현가능성검사다. 위수치검산,원래마스크·입력/생리/근육정체성,원자료/호출계수가모두맞아야valid다. 통과하면같은basis를변경하지않고별도명시적box/SOC 적합을설계한다. 기존20%command 개선·다리별q/velocity비악화·출발qRMS.018/max.04·tendonRMS.009/max.025는뒤의proxy후보기준으로유지한다. 이proxy는최종보행의필요조건으로격상하지않는다. 미달/자원실패는그대로보존하고임의반복이나F1–F8완료를선언하지않는다.

부모가설계하고실제gpt-6-sol/high가새 `tools/banc_m3_causal_neural_basis.py`, `tools/prepare_c_banc_m3_causal_neural_basis.py`, `tests/test_c_banc_causal_neural_basis.py`만구현한다. 새역전파없이고정입력기억/RK4기저를계산하는한정CUDA모듈로범위를좁혀high를사용한다. 부모FIRST→필요시같은Sol첫수정→부모SECOND/잔여수정→원본한번순서다. 이전소스/게이트를수정하거나monkeypatch하지않는다. 이전helper의경로검사/hash/state load와canonical decoder math만읽기재사용가능하다.

작은생성NumPy기준은각RK단계/시간축·정규화합1·원점gate0·prototype중복/동률·혼합선형성·초기상태를검사한다. 새kernel의작은실제CUDA대조는부모FIRST후한테스트에서동일생성입력의history/basis두launch만,generated neural64tick이내,신체0으로한번실행한다. 모의실패에서는raw가finite검사전에보존되고첫kernel실패시두번째launch가차단되는지확인한다. 실제original호출은부모source gate후다음한번뿐이다.

원본CPU1/GPU1/process1,history40000tick+basis40000tick(각160000stage),launch2/optimizer0/전체BANC·물리0/retry0/warmup0다. 입력3채널은각각한번device upload하며고정buffer를재사용한다. 내부85/model65/외부95/감독100벽초, GPU1.5GiB/RSS6GiB/출력512MiB/free disk4GiB다. E208까지누적5502.980811599704/5624초·잔여121.019188400296초에서100초를예약하고이번배정증가는0이다. 준비/생성/부모검산은별도,미측정은0아님. 실패시가능한history/basis/H/probe·parameters/prototypes·affine readout/원오류/비용을남기며자동원본재시도는없다. M2·E150최선·UI·F6·과거실패는보존한다.
