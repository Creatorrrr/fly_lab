# E331: 저장 자료로 시간 목적의 방향 오차 분해

2026-10-01 현재 세션 설계. Actual Astra6/max의 `parent-e331-astra-design-review.md` 리뷰를 반영했으며 actual GPT6.1Sol/high에게 구현을 맡긴다. E330은 유효한 실제 실행이었지만 예측 상대오차38.4139%로 사전5% 기준을 기각했다. minus의 국소 손실 감소와 native CSR 전달 가능성은 별도 PASS다. 이에 따라 예정했던 조건부 신체 실험은 미실행이며, 이번 E331은 새로운 저장 자료 진단이다. E330의 기각/5%/handoff 부적격/F1–F8을 바꾸지 않는다.

## 질문과 고정 범위

H331: E330의 실제 관측 차이와 efficacy VJP 예측 차이를, 저장된 관절 명령과 그 명령에 대한 objective 미분을 사용해 objective의 유한 변화와 신경·decoder의 유한 응답으로 분해할 수 있다. Reference nearest segment의 변경을 고정하는 수학적 반사실도 계산하여 objective 오차에 어느 정도 기여하는지 확인한다. 이는 신경 미분 오류·생물학적 원인·보행 성공을 곧바로 확정하는 실험이 아니다.

E330 원자료의50개 창100,104,...,296 및 neutral/plus/minus 모두를 사용한다. Control140/208이 큰 오차를 보였고 상위5개가 절대 residual98.79%를 차지하지만 일부 창만 선택하지 않는다. 새 신경/신체/감각 실행, 전체8547 효능 gradient 재계산, CUDA, optimizer, theta 변경, step-size search, 원본 retry는0이다. 미사용 F6/M2/UI는 비노출이다. 단4×42 관절 명령에 대한 CPU objective 미분만 새로 계산한다.

## 사전 입력 계약

E330 완료 supervisor의 소유 PID 종료와 root932-check raw audit PASS/run_valid:true/refuted를 확인한다. Current source/input producer manifest, original freeze/source review, result/advance-ledger/output-manifest/receipts, 모든 필요한 raw NPZ, E329 승인 objective·qualification·reference/정상화 bytes, E318 raw 경계/clock, E330 direction/각 창 gradient와 actual f32 theta를 source·입력 SHA 계약으로 고정한다. 모든 gate는 NPZ 숫자나 Torch 미분 전에 수행한다. fresh output만 허용한다. 분석 전후 고정 bytes를 확인하고 결과가 참조하는 원본 결과/ledger SHA를 남긴다.

Reference curves[6,128,7]f64와 정상화(6,)f64는 E329 승인 bytes/leg order/frequency2.382760007990865/dt.005/all15 coefficient1을 그대로 쓴다. CPU constructor의 정상화 값을 먼저 허용오차로 검증하고 승인된 f64 vector를 실제 loss tensor로 옮겨 bytes를 확인한다. E330 생성 CUDA 자격을 CPU 실행 증거로 대체하지 않는다. 단위는 실제 관절 rad/phase proxy cycle로 구분한다.

## 전체50개 창 계산

1. 시작 c의 고정 원래 q_boundary[c]와 각 arm의4개 q를 검사한다. previous=concat(boundary,q[:3]),next=q이며 경계만 고정한다. 실제 four scalar losses는 E329 approved evaluate_pairs로 재현하고 원래 saved loss_terms/E330 result와1e-10으로 일치해야 한다. Source-independent NumPy 또는 생성 수치미분 자격을 제공한다.
2. Neutral q[4,42]를 미분 변수로 두고 `L_neutral/50`의 g_q[4,42]를 계산한다. q_previous 내부3개와 successor target/coordination 연결을 모두 유지한다. 고정 경계는 미분하지 않는다. CPU f64·finite shape를 요구하며 exact0 g_q도 정상 결과다. g_q와 arm q/고정 boundary/중간 진단을 새 raw에 보존한다.
3. 실제 명령 차이 δQ=Q_plus−Q_minus, 실제 rounded efficacy δθ=theta_plus−theta_minus를 사용한다. 재현 CPU losses에서 창별 O=(L_plus−L_minus)/50, Q=g_q·δQ, T=g_theta_window·δθ로 둔다. 원래 E330의 GPU 관측 O도 별도 기록하고 재현 차이를 보고한다. 원래 g_theta는이미1/50이며 중복 나누지 않는다. `O−T=(O−Q)+(Q−T)`를 별도 계산해1e-12 절대오차로 확인한다. Q−T에는 신경/decoder 비선형 유한 응답, surrogate/반올림 VJP 차이와 CPU/GPU objective cotangent·branch·산술 차이 가능성이 함께 들어간다. E330은 당시 GPU g_q와 선택 m을 저장하지 않았으므로 이번 CPU 자료만으로 동일성을 확인했다고 쓰지 않고 순수 신경 잔차나 adjoint bug로 이름 붙이지 않는다. 개별 g_theta 파일은 direction.window_gradients의 control 순서와 정확히 일치해야 한다.
4. Neutral/plus/minus의 previous·next 각각에 nearest segment m, raw_u/clamped u, proxy phase, distance와 비국소 branch gap을 기록한다. Neutral 대비 cyclic segment거리>1 AND circular phase차>2/128을 endpoint 사이의 비국소 선택 변경으로, clamp regime 변경을 별도로 기록한다. strict below0/above1, exact0/exact1, interior를 구분한다. Adjacent segment 변화나 phase wrap을 비국소 불연속으로 세지 않고 endpoint 사이의 선택 변경만으로 연속 명령 경로의 불연속을 입증했다고 쓰지 않는다. tie gap<=1e-12/target knot도 기록하고 competitor가 없으면null을 사용한다. 정확한 tie/endpoint/knot의 g_q는 선택된 program cotangent이며 유일한 도함수나 중앙차분 일치를 주장하지 않는다. 진단을 무조건 성공 판정으로 바꾸지 않는다.
5. Neutral의 previous/next nearest segment index만 고정하고, plus/minus q에서 그 segment의 투영비율·clamp·phase·successor target interpolation과 all15coordination을 다시 계산하는 NumPy 반사실 loss B_plus/B_minus를 만든다. 새 runtime controller나 phase 고정 명령은 없다. Neutral에서 이 branch-fixed 함수는 실제 neutral loss와1e-10으로 일치해야 한다. 고정된 m 안에서는 실제 live q를 사용하고 clamp·target curve interpolation은 그대로 둔다. 따라서 이것이 모든 비선형성/vertex/clamp를 없앤 함수라고 주장하지 않는다.
6. F=(B_plus−B_minus)/50로 두어 세 항 `(O−F)+(F−Q)+(Q−T)=O−T`를1e-12로 검증한다. O−F는 모든 nearest-index 선택을 neutral index에 고정한 효과이며 정상 인접 선분 이동도 포함한다. 비국소 불연속만의 기여가 아니다. F−Q에는 선택 branch 내 clamp/target interpolation knot/cosine/선택 derivative convention을 포함한 objective의 유한 비선형성이 남는다. Q−T는3항에서 설명한 CPU/GPU objective 차이도 포함할 수 있는 잔차다. 신체·신경의 실제 개입 인과성으로 과장하지 않는다.

전체50개에 signed sum, sum(abs(window contribution)), 절대 residual 비중과 최고값/clock을 모두 보고한다. 다리 보고는 Q_l=sum(g_q*δQ)의 해당7개 명령 좌표 기여, successor의 다리별 O/F, coordination15pair별 O/F와 geometry로 제한한다. Coordination의 cross-leg 의존성이 Q_l에 들어 있으며 생물학적 책임이 아니다. 저장 총 g_theta만으로 다리별 T/(Q−T)를 유일하게 복원할 수 없으므로 임의 그룹 배정/균등 배분을 하지 않는다. T와 Q−T는 창별/전체로 보고한다. Denominator가0이면 비중은null로 보고한다. 항별 절대합이 총 residual 절대합으로 더해지는 것은 아니므로 비중 합100%도 주장하지 않는다. Cancellation 때문에 signed total의 임의 비율을 근거로 단일 원인을 확정하지 않는다. 상위 창/전체 분포와 branch 변경 표를 제공하되 별도80% 등 사후 채택 threshold를 만들지 않는다. 결과는 계산 유효성/오차 분해/원인 가설만 내며 E330 refuted, candidate_adopted/F1–F8/biological_gait는false로 유지한다. 다음 변경 설계는 이 근거를 검토한 뒤 별도로 작성/리뷰한다.

## 자격·실패·소유·자원

새 `tools/analyze_c_banc_m3_temporal_residual.py`, `tests/test_c_banc_temporal_residual.py` 두 파일만 Sol 소유다. E329/E330/과거 backend는 불변이다. 생성 tests에는 일반7D polyline의 branch-fixed neutral equality, stable branch에서 fixed-index neutral 미분과 g_q의 독립 중앙차분 대조, adjacent/wrap/nonlocal/clamp 구분과 인접 vertex 이동에서 O−F≠0이지만 nonlocal=false인 반례, live previous+1/50 Q 미분 독립 수치미분, 고정 boundary, exact0 residual/finite 거부, 두·세 항 폐합, SHA gate가 payload 전에 거부되는 경우, 부분 계산/게시 실패 보존을 포함한다. 대수적 폐합만으로 Q/F 계산이 옳다고 쓰지 않는다. Tests가 곧 실제 분석 결과는 아니다. 필요한 q/boundary/loss/gradient/reference key만 allow_pickle=False로 읽으며 neural loader/build를 호출하거나 불필요한 큰 rate/adaptation/full CSR를 로드하지 않는다.

Root FIRST→필요시 같은 Sol R1→root SECOND→잔여수정root를 지킨다. Sol/root generated CPU각90초, saved CPU supervisor120/internal110/finalization10초, host2GiB/output128MiB/free disk1GiB, CPU thread1/CUDA 미노출로 제한한다. 실패와 최초 오류/부분 raw/시간을 보존한다. 계산 결과를 supervisor 종료 후 root가 NumPy/scalar/실제 raw와 독립 대조한다. 원본0으로 기록하며 새 original allocation/reservation은 없다. 현재 원본 누적9200.673308899522/9360초, 예약0을 그대로 유지한다. 전체 목표는 active다.

## Actual Astra 리뷰 반영

원래 초안SHA e852edee8a1a8fc3428cf8645b4d9706d76eebaedf39896076e8e217d0129567과 리뷰를 불변 보존했다. 필수3개(CPU/GPU cotangent 차이, 전체 nearest-index 고정 의미, 복원 가능한 leg 항 범위)를 반영했으며 stable-branch 값·미분/인접 이동 반례/endpoint convention/null/대수적 폐합 한계/가벼운 key별 loader도 자격에 반영했다. 준비 시간을 포함한 internal 시작부터100초까지 계산과 사후 hash를 마치고 finalization10초를 실제 남긴다. Source/input 사후 변조와 게시 실패는 분석 valid=false로 기록한다. 구현은이 반영본을 따라야 하며 새로운 CUDA/원본 실험은 포함하지 않는다.
