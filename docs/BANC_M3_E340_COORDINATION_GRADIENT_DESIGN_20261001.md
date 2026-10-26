# E340: 기존 successor·coordination 항의 출력 기울기 배분

ROOT 설계에 실제 GPT-6 Astra/max의 M1–M6 리뷰를 반영한 실행 설계다. 같은 자연 보행 finite-goal의 진단 단계이며 목표·F1–F8·BANC provenance·neural-only runtime·M2·UI·미사용 F6·과거 실패·비용을 보존한다. E339 저장 명령 분석과 독립 원자료 감사가 종료되었고, 현재 원본 누적9567.165820199525/9660초, 잔여92.83417980047489초, 예약0이다. 아직 E340 원본 실행을 예약하지 않는다.

## 질문과 선택 근거

E339는 reference/live coverage가 모두 통과한 상태에서15쌍 중11쌍의 큰 encoded 위상 이탈을 확인했다. 좌우 세쌍은49%,32.5%,66%다. 기존 live coordination0.8140994661 중 weighted angular0.7905267966이 대부분이다. 기존 목적에는 coordination 계수1이 이미 있다. 전체 successor6.8843837496와 coordination0.8140994661의 손실 값만으로 학습 영향의 크기를 판단할 수 없다.

이번 질문은 **현재 합산 목적의 출력 공간 하강 방향이 협응 항을 줄이는가, successor 항이 그 방향과 크기를 지배하는가**다. 출력 기울기를 분리해 다음 실제 neural parameter 설계에서 단순 gain 변경을 검토할 근거가 있는지, 경계·목적 구성이나 parameter 전달을 우선해야 하는지 결정한다. 손실을 중복 추가하거나 실제 gain sweep/optimizer/theta/신경/신체 실행을 하지 않는다. 작은 출력 공간 진단은 BANC에서 그 방향을 구현할 수 있다는 증거가 아니다.

## 고정 자료 및 유한한 provenance

E339의 같은 reference5필드 및 neutral/held/live 각각 q/z/R/coordination_pair_loss4필드만 사용한다. 모든 자료는 `verification/banc-natural-gait-m3-20260919/` 아래다.

| 역할 | literal 상대 경로 | SHA256 | bytes |
|---|---|---|---:|
| reference | e332-smooth-temporal-saved/reference.npz | 140e23c865d7e110cd93699b41612c1ffb02eb88dc100ac435c4dcba5acf6c01 | 124353 |
| neutral | e336-saved-gait-transfer/neutral-smooth.npz | 8b75c4b1d2cd7d4efc0fe2db0ad49ad50b7dbf2d9ca25b6f38359c9488603d4c | 1644664 |
| held | e336-saved-gait-transfer/held-smooth.npz | 0ea9afacfb31fcb7595b1c37d235dfbfb39979f366c1a0866f0ec65b3cab1ee6 | 1644664 |
| live | e336-saved-gait-transfer/live-smooth.npz | 4d2ef1a8a22243b905adea9420ef7230e93e51d19d01e73615c37e84b9953ef4 | 1644664 |

Schema/dtype/finite 조건, leg 순서 LF LM LH RF RM RH, Gaussian 계수, reference offsets, 주파수/dt/normalization/temperature와201행 대응은 E339의 고정 계약 그대로다. 모든 member의 안전한 header를 관측하고 선택 값만 실제로 읽으며 pre/post decode SHA와 bytes를 대조한다. E339의 qualified reader/Encoder를 고정 source의 library로 사용할 수 있지만 기존 CLI/worker를 재실행하지 않는다. Loader mock 성공은 reader qualification이 아니다.

직접 closure에는 새로운 owned2 `tools/analyze_c_banc_m3_coordination_gradient.py`, `tests/test_c_banc_coordination_gradient.py`, 실제 사용하는 E339 reader/scanner와 E332 smooth objective, 의미를 확인한 E330 window_loss/E333 perform source, E339 final decision/audit/supervisors/result/manifest/freeze/source review/current proof, 본 설계의 사전 snapshot·실제 Astra review/reflection·실제 Sol dispatch·최종 current source 생성 자격을 명시한다. 기존 E339 evidence를 재귀적으로 모두 다시 검증했다고 주장하지 않는다. 종료된 producer의 상태와 명시한 직접 pin edge 및 사용할 입력4개의 실제 pins를 확인한다. 추가 원본·state/gradient/weight/theta/전체 inherited leaf/F6 자료를 열지 않는다.

## 정확한 미분 및 기존 창 경계

기존 E333은 시작 control100,104,…,296의50개 창을 사용했다. 각 창에서 고정 boundary q[c]는 detach하고 live 출력 q[c+1..c+4]4개를 미분한다. `window_loss`는4쌍 목적을 계산한 뒤50으로 나눈다. E340의 primary는 같은 **출력 공간의 경계 처리**이며 새 신경 window를 실행하거나 과거 무효 E333 전체를 qualify하지 않는다. Live/held 자료의 동일 창 분할은 출력 진단이며 실제 neural reset 재현을 뜻하지 않는다.

T=200, q[k]는 c100+k 경계다. 각 pair k=0..199에서 previous=q[k], next=q[k+1]다. 창 시작 k%4==0의 previous는 미분하지 않고, 다른 previous는 같은 창의 앞선 live 출력이다. 결과는 `g_successor`, `g_coordination`, `g_total` 각각[50,4,42]이며 q[c101..c300]의 고정 순서로 보관한다. q0의 gradient를 넣거나 창 시작의 previous gradient를 앞 창의 마지막 출력에 더하지 않는다.

각 leg의 w_s=softmax(-sum_j((q_j-C_sj)^2)/tau_l), successor target h_k=sum_s w_s A_sk, moment z=sum_s w_s b_s다. C/A/b/tau는 고정 reference constants다. Weight derivative는 `dw_s/dq_j=(-2/tau_l)*w_s*((q_j-C_sj)-sum_r w_r*(q_j-C_rj))`다. `dh_k/dq_j=sum_s A_sk*dw_s/dq_j`, `dz_axis/dq_j=sum_s b_s,axis*dw_s/dq_j`로 target 경로를 포함한다.

Successor S는 `sum_(t,l,j)((q_next-h_prev)^2/n_l)/(T*42)`다. Next gradient는 `2*(q_next-h_prev)/(T*42*n_l)`, previous gradient는 `-J_h^T*g_next`다. 내부3개 previous 경로를 해당 live 행에 누적하고 창 시작 경로만 제거한다. Coordination C는 `sum_(t,a<b)(1-z_a·z_b)/(T*15)`, `dC/dz_l=-sum_(m!=l)z_m/(T*15)`이며 next의 moment Jacobian을 통과한다. S/C의 계수는 각각1이고 g_total=g_S+g_C다. 전체15쌍과 좌우3쌍을 혼동하지 않는다.

현재 고정 E332 `SmoothTemporalObjective.evaluate_pairs`의 CPU float64 autograd를 primary 계산 경로로, NumPy analytic chain rule을 교차검산으로 사용하며 **동일 actual q**에서 비교한다. 원본에서도 loss/z/R/saved pair loss와 analytic↔autograd g_S/g_C/g_total을 검증한다. gq tolerance는 rtol1e-8/atol2e-10, constants1e-12/1e-15, 저장 z/R 및 loss1e-10/1e-10을 사전에 고정한다. 허용하지 않는 detach/1/2 Gaussian/분모/창 경계 차이를 tolerance로 덮지 않는다.

`SmoothTemporalObjective` 생성자의 startup_q[99,42]/startup_tendon[99,6]는 evaluate_pairs에서 쓰이지 않는다. 명시적으로 표시한 zero float64 placeholder로 signature를 채울 수 있으며 evaluate/evaluate startup guard는 호출하지 않는다. 이 placeholder는 원본 startup 상태나 제약 통과의 근거가 아니다. E330/E333는 source text 의미만 확인하고 모듈 import 또는 네트워크 loader를 호출하지 않는다. Torch CPU tensor/command autodiff는 허용하되 BANC parameter backward, neural tick, CUDA context/allocation/kernel은0이다.

## 사전에 정하는 판정과 다음 분기

Primary는 live50창의 g_S/g_C/g_total을[200,42]로 펼친 동일 출력 공간이다. Aggregate/per-window/per-leg L2 norm, dot, cosine, norm ratio 및 `D_C=-dot(g_C,g_total)`를 보고한다. D_C는 작은 정규화하지 않은 `-g_total` 변화에 대한 coordination의 first-order 변화이며 유한 step 효과나 neural parameter 효과가 아니다.

- n_C==0 또는 n_total==0이면 `output_gradient_inconclusive`, 이유 exact_zero다. 또한 primary N=8400에서 η(g)=sqrt(N)*2e-10+1e-8*||g||2를 계산하여 n_C<=10η(g_C) 또는 n_total<=10η(g_total)이면 같은 inconclusive, 이유 below_declared_precision이다. Raw norm을0으로 치환하지 않는다. 이 정보량 규칙은 비교 tolerance에 기반한 공학적 사전 규칙이며 실제 오차의 수학적 상계나 생물학적 기준이 아니다. 정의 불능/null과 precision 부족/null을 구분하고 epsilon 분모를 사용하지 않는다.
- nonzero gradient에서 cosine(g_C,g_total)<-1e-10이면 `combined_output_descent_opposes_coordination`이다.
- 위 조건이 아니고 n_S/n_C>=10이면 `successor_output_gradient_dominates`다. 비율10은 이번 방향 선택의 공학적 진단 문턱이며 생물학적 기준이 아니다. D_C/cosine의 부호와 크기도 함께 표시해 실제 감소와 중립을 구분한다.
- 나머지는 `no_fixed_output_gradient_obstacle`다. 두 계산 경로가 gradient tolerance를 통과해도 고정 분기가 다르면 `output_gradient_inconclusive`, 이유 branch_unstable로 끝내며 경로를 사후 선택하지 않는다. 이는 parameter 전달, 짧은 창 효과, 실제 신체 추진이나 encoder 편향 문제가 없다는 증명이 아니다.
- Neutral/held는 같은 개발 자료의 기술 대조이며 독립 반복이나 primary 선택 근거로 사용하지 않는다. 계산/reader/provenance/clock/publication/ROOT 감사 실패는 technical_invalid로, science는 not_evaluated로 끝낸다. 이전 원본을 반복하지 않는다.

Opposition/dominance이면 다음 실제 neural 설계에서 기존 두 항의 전달·배분을 분리해 확인하는 방향을 우선한다. 자동 gain sweep이나 새 목적 채택은 아니다. No obstacle이면 출력 공간만으로 설명하지 못했으므로 현재 parameter로의 Jacobian/짧은 window와 closed-loop 관계를 우선한다. Inconclusive이면 정적 encoder/기울기 제한을 기록하고 별도 근거 있는 방법을 설계한다. 어떤 분기도 F4나 실제 신경 개선의 증명으로 대신하지 않는다.

## 구현·검증·저장·자원 계약

ROOT 설계 → 실제 GPT-6 Astra/max 읽기 전용 리뷰 → ROOT 반영 → 실제 GPT-6.1 Sol/high 구현 → ROOT FIRST → 필요하면 같은 Sol ONE R1 → ROOT SECOND → 잔여 ROOT 수정이다. 연쇄 미분, detach 경계, analytic/실제 objective 일치와 실패 게시를 다루므로 high를 선택한다. Sol 재위임0, owned2와 `e340-sol-*` 생성 증거만 편집한다. E339 및 과거 소스/결과를 수정하지 않는다.

생성 자격은 analytic↔현재 CPU autograd, 아래에 고정한 조건부 목적의 두 방향 FD, previous target derivative 제거·50 scaling·15 pair 분모·boundary overlap fault, 같은 global phase/half phase의 angular 성질·대칭 중심의 작은 기울기·정확한 zero 입력의 분기, precision/null/cosine/norm ratio10 및 branch instability, 실제 생성 reader의 selected decode와 posthash, unsafe extra/mutated/stale current proof, publication/rename/readback/late science deadline→exit1를 검증한다. Unnormalized moment에서는 같은/half phase라도 concentration의 radial derivative가 남으므로 전체 gq=0이라고 가정하지 않는다. ROOT 독립 audit의 실제 CLI가 아래의 생성 경로에서 wrong window detach/scale/row/pair/gradient/result를 거절해야 한다. Original gq를 보고 난 뒤 threshold를 바꾸지 않는다.

## 실제 Astra M1–M6 반영의 세부 계약

M1: b[w]=copy(q[4w]), x[w,i]=q[4w+i+1]로 분리하며 b[50,42]를 immutable 상수로, x[50,4,42]만 변수로 만든다. F_b(x)=sum_w L4(cat(b_w,x_w,:3),x_w)/50다. FD에서는 b를 original copy로 유지하고 x만 perturb한다. q201를 perturb한 뒤 b를 다시 추출하지 않는다. Direction1=sin(.37*(arange(x.size)+1)), Direction2=cos(.53*(arange(x.size)+1))를 float64로 만든 뒤 각각 L2 norm1로 정규화하여 x shape으로 reshape한다. Central FD steps1e-6/5e-7 두 개를 모두 사용하고 scalar directional derivative tolerance는 rtol1e-5/atol1e-7로 고정한다. 생성 자료는 boundary와 live를 반드시 분리해 두며 앞 창 마지막 output을 바꿔도 다음 boundary가 그대로임을 실제로 검증한다. Live/held의 b는 해당 저장 trace의 값이고 E333의 fixed original boundary 값 자체를 재현했다는 주장을 하지 않는다.

M2/M3: primary CPU autograd와 analytic 양쪽에서 위 precision 규칙 및 같은 분기를 계산한다. 예상 밖 gradient/norm/dot/cosine/ratio overflow/NaN은 technical_invalid다. 정의 불능을 정해진 null로 표현하는 경우만 예외다. precision 부족이면 ratio/cosine의 권한 없는 분모 보완을 하지 않고 null의 이유와 raw norms/η를 함께 저장한다. n_S가0이면 S/C cosine은 undefined다. Dominance는 크기 우세이며 해로운 희석과 동치가 아니다. Dot(g_C,g_S)>0인 도움·반대·중립과 D_C를 같이 보고한다. O1도 precision이 충분한 경우 dot(g_C,g_total)/n_C²로 기술 보고하며 새 채택 문턱은 만들지 않는다.

M4: `tools/analyze_c_banc_m3_temporal_objective.py`와 그 top-level import인 `tools/banc_m3_phase_free_temporal.py`를 실행되는 한정 local import source로 추가한다. 후자는 math/torch만 import한다. 기존 E329 objective/CLI를 호출하지 않고 constructor의 numpy_interpolate만 사용한다. 해당 파일과 실제 실행되는 한정 import edges를 source inventory에 pin하며 neural/body loader를 import하지 않는다. Read-only source로 확인한 E330/E333 의미 pins와 실제 실행 source를 분리해 표시한다.

M5: ROOT 독립 audit executable은 생성과 원본에서 동일 파일·SHA를 사용한다. 생성 namespace는 BASE의 fresh `parent-e340-*` 아래이며 literal 원본 root/path/hash를 거절한다. 실제 CLI를 감독 하에 실행하여 작은 생성4개 archive와 생성 production output을 읽고 정상 사례를 통과시킨다. Wrong detach/scale/row/pair/gradient/result6종은 그 같은 CLI가 거절하고 exit1/실패 보고서를 남겨야 한다. Loader mock, helper만 실행한 PASS나 production gradient/classifier expected 호출은 대체 자격이 아니다. 생성 namespace를 production CLI override로 노출하지 않는다. 현재 ROOT source와 생성 계약/감독/logs/qualification 및 actual audit executable source pin이 최종 freeze에 들어간 뒤에만 원본 분석을 실행한다. Audit의 독립 covariance J_h=(2/tau)*(E_w[AC^T]-h E_w[C]^T), J_z=(2/tau)*(E_w[bC^T]-z E_w[C]^T)와 별도 누적/분기를 사용한다.

M6: 각 neutral/held/live `*-gradient.npz`의 고정 배열은 primary g_successor/g_coordination/g_total과 analytic_g_successor/analytic_g_coordination/analytic_g_total, successor_next_path/successor_previous_internal_path8개 [50,4,42], all201_z[201,6,2], all201_R[201,6], pair_losses[200,15], successor_terms[200,6]다. 모두 finite `<f8`; boundary previous_internal_path[:,0]는 구조적으로0이며 omitted Jacobian을0으로 위장한 값이 아니라 실제 gradient에 누적하지 않은 경로라는 표시다. 전체 Jacobian을 추가로 저장하지 않고 원본에서 다시 계산한다. 각 archive 실제 serialized bytes<=640KiB다. 출력 증거 quota는 아래와 같이 고정한다.

| 파일 분류 | 개별 상한 | 개수 |
|---|---:|---:|
| gradient NPZ | 640KiB | 3 |
| selected-read ledger / manifest | 32KiB /16KiB | 각1 |
| original result /receipt /stdout | 256KiB | 각1 |
| original/audit stderr | 32KiB | 각1 |
| original/audit supervisor | 16KiB | 각1 |
| original/audit owned-PID record | 8KiB | 각1 |
| original/audit external receipt | 64KiB | 각1 |
| independent audit report/stdout | 256KiB | 각1 |
| authoritative final decision | 64KiB | 1 |
| retained failure/revocation headroom | 512KiB | 전체공유 |

고정 quotas 합계4064KiB<4096KiB이며 나머지32KiB는 미지정 overhead다. Audit checks는 배열/역할/창 단위로 묶어 기록하고 수만 늘리는 scalar별 중복문서를 만들지 않는다. Same role의 `.partial`도 해당 archive quota 및 전체 current-used에 포함한다. Fresh target과 stage를 동시에 남기는 복제는 허용하지 않는다. 매 write/compression/rename/print 전에 serialized payload/현재실제bytes/미발행 필수기록 예약/512KiB failure 여유를 확인하고 초과 전에 fail-closed한다. 외부 로그/감독/receipt/final decision도 정해진 source의 bounded serialization과 reserved quotas를 적용한다. Staging 중복·총량 초과·rename/readback/late deadline·감사 실패의 생성 사례를 actual publication/CLI로 검증한다. 기존 성공 파일을 덮어쓰지 않고 authoritative final decision이 worker 또는 이후 receipt/audit failure를 반드시 거절한다.

사전 snapshot/반영/실제 dispatch·현재 owned2 소스·Sol/ROOT 생성 계약/감독/보고서/로그/source pins를 유한한 closed proof로 연결한다. 두 branch는 zero original calls 및 현재 source2/exit0/감독 성공과 byte pins를 모두 확인한다. 최종 freeze와 ROOT source review를 CLI literal로 검증한 뒤에만 원본 reader를 호출한다. Mutable progress/cost ledger를 실행 source pin으로 쓰지 않는다.

원본 출력은 fresh literal `verification/banc-natural-gait-m3-20260919/e340-coordination-gradient/`다. 분석1회/retry0, science12초/internal16초/finalization4초/supervisor20초, CPU thread1/CUDA hidden/host2GiB다. s0는 NumPy/Torch/scanner import와 gate 전이다. ROOT 독립 원자료 감사1회/internal8초/supervisor10초로 analytic gradient와 분기/원본 대응을 재검산한다. 실행 전20+10초가 잔여 내인지 확인하고 prospective reservation을 작성한다. 현재92.8341798초를 reset/증가하지 않는다. 실패 비용도 각 supervisor 실제 elapsed를 한 번씩 계상한다. source/generated 공유 상한은 ROOT/Sol 각60초·각32MiB다.

작은 gradient3개와 필요 Jacobian/분리 누적 경로, scalar/cosine/norm summaries를 저장하며 neural state나weights를 출력하지 않는다. Numeric→actual selected-read ledger→provisional terminal/동일receipt→비자기참조 manifest→byte/hash readback→동일 terminal stdout 순서다. 모든 오류는 성공 상태를 취소하고 exit1/별도 revocation을 보존한다. 출력/staging/stdout/stderr/감독/외부receipt/감사 전체를 합해4MiB 이하인지 종료 후 확인한다. ROOT의 independent audit 및 external receipt가 모두 PASS할 때만 별도 final decision에서 고정 diagnostic을 채택한다. 저장 성공 terminal을 사후 덮어쓰지 않는다.

최종 candidate/F1–F8/neural improvement/자연성/생물학적 기전은 false다. 실제 BANC/CUDA/10초 신체/미사용2개 초기상태/6개 목표와 정지·재시작/영상/M2 복원은 계속 미달 항목이다. 진단 통과를 전체 목표 완료로 기록하지 않는다.
