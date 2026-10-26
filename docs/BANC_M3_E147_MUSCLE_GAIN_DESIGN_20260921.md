# E147 — 근육별 활성-관절 전달 크기의 제한 보정

2026-09-21. finite M3의 다음 제한 실험이다. E146은 valid/refuted이며, 직전 목표 턴은 구현·실행·독립 검산으로 다음 결정을 바꾼 progress다. F1–F8, E142 실제 신체의 F3 통과/F4–F5 실패, 원래 초기99, 과거 실패, M2/UI/F6 미노출을 유지한다. 현재 소스와 E146 최종 결과를 다시 확인했다.

현재 adapter는 모든 근육 활성에 고정 .02rad/rate와 calibrated moment를 적용한다. 이 값은 공학적 읽기 변환이며, `prepare_c_banc_m3_front_moment.py`에서도 앞다리 일부 전달 크기를 양의 scalar로 보정한 이력이 있다. E145의 정규화된 정적 혼합과 E146의 세 시간 응답 혼합은 같은 복합 제안 조건에 모두 미달했다. 필터 시간을 더 늘리는 방법은 비교 가능한 대안이지만, 이번에는 DC 전달 크기를 고정1로 둔 가정 자체를 제한적으로 시험한다. 실측 근육 힘 모델을 구현했다고 주장하지 않는다.

H147: E146과 동일한 5/10/40ms basis에서 근육별 총 전달 gain을0.5–2로 허용하면, 고정 초기99 조건과 후기 command RMS20% 감소 및 전체/각 다리 command/velocity 비악화를 함께 만족하는 조건부 후보가 존재한다. 경쟁 설명은 이 범위의 전달 크기를 허용해도 neural signal/moment 방향/고정 rest의 표현력이 부족하다는 것이다. 이 공학적 제안 조건은 자연 보행의 필요조건이 아니다.

## 고정 입력과 변경 범위

E146 원본 `causal-filter-raw.npz`의 filter_bank[2000,391,3]을 재사용한다. 새 EMA rollout0이다. E143의 원래 b/pool/membership/moment/rest/ltm 및 E103 초기 target은 같은 hash로 고정한다. bank의 shape/유한성, motor-major order, tau와 middle10ms exact를 검증한다. 새로운 신경/CUDA/물리/LP 호출0, 새 원본 fit1회다. 뉴런·연결 provenance, 양의 같은 근육 내 membership, moment의 방향/부호, rest, 표적/위상/시간창을 변경하지 않는다.

후보는 effective W[96,391,3]로 저장한다. LTM6와 원래 singleton 행은 E146과 동일하게 원래 pool의 middle branch만 유지한다. 가변 근육은 원래 소속 모든 MN의 세 branch coefficient를 직접 변수로 둔다. 변수 순서는 muscle ID 오름차순, 그 안은 motor-major flat3*i+k 오름차순이다. anchor 제거: q의 affine constant는 rest와 고정 근육 활성만 포함하며, 각 변수 열은 .02*bank[:,i,k]*moment[:,m]이다. actual activation a_m=sum(W_mik*b_ik), command q=rest+.02*M*a다. 고정 근육 activation은 E144의 같은391-term reduction을 직접 사용해 bit-exact 보존한다.

gain_m=sum(i,k)W_mik이다. 원래 canonical baseline은 middle branch의 pool, gain1이다. lifted baseline은 각 원래 MN weight의1e-6씩을 fast/slow로 옮기고 middle에는1-2e-6를 둔다. 원래 균등 pool을 검증한다. solver는 모든 가변 branch>=1e-6/원래 해당 근육 MN수, <=2와 근육별 gain∈[.500001,1.999999]를 사용한다. 작은 내부 guard는 사전에 정한 수치 여유이며 실제 판정은 gain∈[.5,2] atol1e-12, branch>=0, 원래 MN별 branch 합>=1e-6/n atol1e-12다. 고정 행은 정확히 원래값이고, 다른 근육으로의 연결0이다. 사후 clipping, 정규화, gain 보정 또는 중간 후보 조합은 금지한다. effective W와 gain을 그대로 저장하며, 기존 normalized-pool runtime과 즉시 호환된다고 주장하지 않는다.

## 적합·판정·검증

E146과 같은1827 endpoint, 실제 인접1826 velocity pair, 초기99, 전체/다리별15 SOC 및 초기99*42의 +/-sample 제약을 유지한다. 목적 max t, -1<=t<=1, 각 물리 norm/한도<=1-t다. 전체 command 한도 baseline RMS*.8, 전체 velocity 및 다리별 command/velocity는 각각 baseline, startup RMS .018/max .04, fixed tendon RMS .009/max .025이다. thin QR 전체 R을 보존하고 truncation하지 않는다. anatomical provenance에서 실제 다리 근육을 읽고 원래 column 목록·cross-leg moment0·전체96 partition을 확인한다. canonical/lifted 직접 norm과 QR norm, lifted t=-.251 feasibility, canonical 대비 lifted command delta를 기록한다.

Clarabel0.11.1/CPU1/max_threads1/qdldl/max_iter100/time_limit30초, tol_gap_abs/rel/feas/infeas_abs/rel1e-10와 나머지 defaults를 유지한다. 단일 solver 반환 벡터만 물리식으로 다시 평가한다. 저장은 pre-solve bank/variable mapping/row-sum matrix/A/b/q/cones/QR, 실제 settings/history/x/s/z/status/잔차, candidate effective W/gain/출력/지표 순서다. 이미 만들어진 caller-owned raw는 계산·판정·저장·자원 오류에서도 보존하고, actual solve 진입 횟수와 첫/secondary 오류를 구분한다. raw를 먼저 저장하고 제안/자원 판정을 한다.

E145의 별도 dual 수치 진단을 재사용한다. self-dual cone 투영, SOC 내부 pad1e-12*(1+tailnorm), r=q+A.T*z, L=-b.T*z+sum(min(r*l,r*u))와 추가1e-8*(1+abs(b.T*z)+sumabs(boxterms))를 사용한다. 이번 변수 box는 coefficient∈[0,2], t∈[-1,1]이다. 음의 수치 t 상한은 위의 guarded bounded-gain3filter family에 한정한다. 이는 exact rational 증명이나 자연 보행 불가능성의 증명이 아니다.

valid run에서 실제 복합 조건을 통과하면 H147 supported/조건부 proposal, 물리 미달이고 충분한 음의 수치 상한이면 해당 family refuted, 그 외 inconclusive다. 어느 경우든 이번 실험만으로 보행 채택은 하지 않는다. 통과하면 후속 새 adapter/trial identity와 bank state/motor-cut/saved-future를 갖춘 native 시험을 먼저 설계한다. 같은 최종 후보의 실제 F1–F8을 모두 요구한다. 현재 기존 live UI/M2를 바꾸지 않는다.

## 실행·소유권·자원

부모 설계→실제 Sol/high FIRST 구현→부모 FIRST review→같은 Sol FIRST correction(필요 시)→부모 SECOND review→부모 잔여 수정이다. Sol/high는 저장 신호의 양의 선형 전달·희소 conic 적합에 적절하며, 후속 native/CUDA/state 복원은 xhigh가 필요하다. E146을 추가 수정하는 작업이 아니라 새 E147 두 파일 구현이다: `tools/fit_c_banc_m3_muscle_gain.py`, `tests/test_c_banc_muscle_gain.py`. 검증된 E144/E145/E146 pure helper·safe solver를 재사용하고 원본 코드/봉인 자료는 수정하지 않는다. 부모는 freeze/독립 검산/감독/결과 기록을 소유한다. FIRST 구현·수정에서는 생성자료만 실행한다.

원본1프로세스/재시도0/fit1, CPU1/GPU0, filter/neural/CUDA/physics/LP0, fit30/internal60/external65/supervisor70초, raw128MiB/free1GiB다. E146 보수 누적2199.860127400025를 보존하고 prior2200+70=2270으로 사전 배정해 기존2271 안에 둔다. 생성자료와 부모 검산 비용은 별도 기록한다. 부모 최종 source gate 전에는 원본 적합을 실행하지 않는다.
