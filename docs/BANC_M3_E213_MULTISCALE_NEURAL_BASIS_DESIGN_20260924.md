# E213: 출력 상한을 보존하는 두 시간척도의 신경 입력 기억

## 결정과 근거

E211의 전신 출발 제약을 포함한 고정 반응 기저는 수치 infeasible였고, E212에서 동일 출력 상한·근육/관절 결합의 시점별 독립 출력은 main RMS .001291991과 startup max3.33e-16으로 목표 자세를 만들 수 있었다. 따라서 같은 힘의 상한을 늘리기 전에 반응식의 시간 표현을 바꾸는 가설을 한 번 시험한다. 현재 E209의16ms 기억은 기존 모델의 자유MN 적응 시간상수 중앙 약300ms보다 짧다. 느린 입력 기억이 짧은 기억에서 겹치는 문맥을 구별할 수 있다는 가설이다. 이것은 기존 rate-dependent adaptation 복원도, 알려진 생물학 기전도 아니다.

별도 실제 gpt-6-sol/medium의 저장 자료 감사에서 prototype의57.1%가 startup0..99행에서 왔지만 main feature와 basis는 실제로 변하고, main 최근접 center가 startup인 비율은21.3%였다. 단순한 초기 과대표집 또는 후기 외삽만을 원인으로 단정하지 않는다. 6→9차원에 같은16개 중심을 쓰면 표현 범위가 부족할 수 있다는 반론도 남긴다. E213은 전처리/기저 검증이며 성능 판정은 이후 같은 전신 제약 적합에서 한다. 실패한 E211을 다시 푸는 시도가 아니다.

## 단 하나의 신경 반응 변경

E208 실제 E/I/drive 160000×391 f32, E209의 frozen scale, cap/r0/tau, E201의 actual inverse_adaptation_tau f32를 사용한다. x=[log1p(E)/sE, log1p(I)/sI, asinh(d)/sD]와 빠른 상태 h_fast'=(x-h_fast)/.016는 그대로다. 새 느린 상태 h_slow'=(x-h_slow)*float64(actual inverse_adaptation_tau)이며 각 채널 초기0이다. 기존 E201 adaptation_tau의 재계산 reciprocal로 대체하지 않는다. stage RK4 dt=.00025,20tick/control,입력4stage를 유지한다.

새 descriptor는 [x3,h_fast3,h_slow3]의9차원이다. 빠른/느린 상태는 각 RHS의 stage 상태를 사용하며 저장 endpoint descriptor는 마지막 stage x와 완료 상태를 연결한다. row0은 첫 입력 x와 두 상태0이다. 흥분 gate는 E209의 g=-expm1(-8*(xE+abs(xD)+max(h_fastE,0)+abs(h_fastD)))를 그대로 쓰며 느린 기억은 gate에 들어가지 않는다. h_slow나 입력을 사후 clip하지 않는다. 시간/phase/목표/과거행index는 runtime feature에 들어가지 않는다.

16 prototype은 전체2001 저장 descriptor에서 원래 row0-first/farthest/first-tie 규칙을9차원에 동일 적용한다. sigma2도 기존 strict-positive nearest distance median, 최소1e-4/퇴화시1이다. scale/중심/폭은 실행 중 변하지 않고 새 fit변수가 아니다. 입력기억 추가와 그에 따른 중심/폭 재계산이 변경 묶음이며 다른 변화의 효과라고 분해 주장하지 않는다.

p는 동일 normalized Gaussian partition-of-unity, phi=cap*g*sum(beta_j*p_j), beta0..1, r'=(phi-r)/tau다. B_j'=(g*p_j-B_j)/tau,B0=0;H'=-H/tau,H0=r0를 유지한다. 따라서 sum B와 canonical D의 합으로 정한 순간별 출력 envelope는 E209와 수치 허용범위에서 동일해야 한다. free343/fixed48를 그대로 두고 fixed48는 conditional 원본 rate를 사용한다. 최종 native 통합에서는 fixed48도 실제 입력에 원래 식으로 반응해야 하며 파일 재생이 허용되지 않는다.

## 원자료와 검증

새 CUDA history40000tick→고정 prototype→basis40000tick의2launch만 허용한다. E/I/drive를1회씩 upload하고 버퍼를 재사용한다. h_fast,h_slow,descriptor,B,H,H_factor,probe(beta=(j+1)/17),parameters,prototypes,canonical affine D/q_intercept/probe 직접/affine과 원시 오류/ledger/manifest를 저장한다. 기존 E209 canonical_readout/affine_readout/validate_affine_readout 및 원래 경로/hash/load helper는 읽기 재사용할 수 있다. 기존 소스 monkeypatch/수정은 없다.

첫/두번째 history는 fast/slow/descriptor 각각 bytes exact여야 하고, 기존 E209의 h_fast/H/H_factor/q_intercept/fixedLTM/tendon은 bytes exact 보존을 확인한다. E209 sum B와 sum D의 비교는 atol2e-10/rtol2e-11이며 이 기준을 사후 완화하지 않는다. B>=-1e-12,sum B<=1-H_factor+1e-10,finite,probe와 H+cap*Bbeta의 atol2e-10/rtol1e-11,직접/affine q atol2e-12/rtol2e-12를 확인한다. 새 h_slow는 부모가 native 원시입력에서 RK4 선형 재귀의 별도 계산으로 검산한다. 입력/상태/출력의 실패 raw는 검증 전에 저장하고 첫 launch 실패 시 두번째를 실행하지 않는다.

작은 생성 검사는 최대64tick의 NumPy RK4와 실제 CUDA 두 launch를 하나의 테스트에서 비교한다. 실제 CUDA는 부모 FIRST/SECOND 후1회만, raw를 assertion 전에 저장한다. pure/mock에서는 stage순서·두 기억·무입력 gate0·기저합1·tie/퇴화 prototype·출력 envelope·불완전 저장을 확인한다. 원본 CUDA 또는 solver를 mock 이름으로 숨겨 실행하지 않는다. 실제 원본은 부모 source gate 후1회,optimizer0/전체BANC0/신체0다.

후속에는 같은 beta5488개와 원래 startup 전신 norm, q/velocity/per-leg 제약,20% 개선 proxy를 적용한 하나의 CPU 전신 fit을 별도 동결해 평가한다. 표현 구축만으로 개선 채택하지 않는다. proxy 미달이면 기록하고 대안을 선택하며 F1–F8을 바꾸지 않는다. 유효 후보가 나온 뒤에만 신경식/decoder2×2,현재 입력의 full BANC native 통합,저장/차단/10초 gait,미사용F6/기존목표F7을 순서대로 검증한다.

## 소유권과 자원

부모 설계→실제 gpt-6-sol/high의 한정 CUDA 구현→부모 FIRST→동일Sol 첫수정→부모 SECOND/잔여수정 순서다. 새 파일은 tools/banc_m3_multiscale_neural_basis.py, tools/prepare_c_banc_m3_multiscale_neural_basis.py, tests/test_c_banc_multiscale_neural_basis.py뿐이다. 과거 실험 코드/결과와 production/UI는 수정하지 않는다. 추가 위임은 없다.

원본 CPU1/GPU1/process1,launch2/조건부총80000tick/solver0/전체신경0/물리0/retry0/warmup0. model65/내부85/외부95/감독100벽초, GPU1.5GiB/RSS6GiB/output512MiB/free4GiB. 기존 원본 누적5642.788942499708/배정6004초·잔여361.211057500292초에서100초를 예약하고 배정 증가는0이다. 준비/생성/독립검산은 별도,미측정은0 아님. 원본 계수 fit·native 통합은 이 실행 범위에 포함되지 않는다. M2/최선E150/F6미노출/F1–F8 active·미완료와 전체 실패·비용 이력을 보존한다.
