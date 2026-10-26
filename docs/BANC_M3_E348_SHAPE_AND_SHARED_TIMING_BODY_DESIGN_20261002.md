# E348 현재 신체의 운동 형태와 공통 타이밍 대조

ROOT가 E347 실제 독립 결과를 바탕으로 설계한다. 전체 M3 finite-goal/F1–F8, M2/UI/F6, 기존 실패·원본 누적을 유지한다. E347의 standalone NVML/저장 진단은 끝났지만 M3 보행 모델은 채택하지 않았다.

## 근거와 선택

E343 네 조건부 창의 C/S 감소는 E346 전체 신체의 startup 통과로 이어지지 않았다. E347은 command-target 최대오차0.04052003713554253rad를 tick56/column19에서 독립 확인했고 8개 column이0.04를 넘었다. 이것은 관측 관절 위치나 단일 신경의 원인 증명이 아니다. 동일 theta의 재생 또는 특정 관절 margin 교정을 반복하지 않는다.

E335는 유효한1.5초 실행이지만 forward1.7214386mm/slip2.0461588mm/s/bout.21초/eligible cycles 여섯 다리0으로 기각됐다. E336에서 command의 위상 진행2.0174–2.4811회와 실제 관절 추종RMS.001096–.001979rad를 확인했다. E339에서 reference와 비교한15쌍 중11쌍이 고정 encoded 협응 이탈 기준을 넘었다. 운동 출력이 전혀 없거나 위치 서보가 명령을 크게 놓친다는 설명과 맞지 않는다. C/S 국소 gradient의 추가 조합만으로 실제 접촉 개선을 가정하지 않는다.

E337의 현재 파형 교체 물리 대조는 전체/선택 archive schema 혼동으로 constructor 이전 technical_invalid였다. E338도 historical selected6을 complete7로 오해해 거부한 뒤 소스만 교정했고 실제 신체 가설은 미판정이다. 과거 실패를 없애거나 자동 재시도로 바꾸지 않는다. 이 새 대조는 필요한 body JSON·control/native/reference만 읽고 별도 정확한 선택 reader를 사용한다. 수천 input/whole graph/drive chunk를 재귀 검증하지 않는다.

H348은 현재 신체에서 reference 형태와 저장 BANC의 공통 타이밍 목표가 추진/접촉 품질을 개선할 수 있다는 가설이다. 대안은 현재 상태 재설정 없는 neural 학습의 반복 또는 contact/구동 가정 변경이다. 현재 물리 대조가 유효하게 확인되지 않았으므로 먼저 이 작은 판별 실험을 선택한다. 이 실험의 외부 저장 명령은 진단·학습 목표 선택용이고 최종 BANC 경로가 아니다.

## 고정 세 arm과 해석

각 arm은 같은 E335 observed-start body를 독립적으로 복원한다. neural/network/adapter 복원·실행0, CPG advance0, body timestep100us/control5ms/interpolateTrue/300controls/15000native를 유지한다. 고정 body.configure/profile/options/model hash를 유지한다. E335 historical compiled physics identity와 joint-name 순서의 직접 비교 필드는 not_recorded로 보존한다. actual B/I/C 사이 identity/order 동등성과 B 전체 재현으로 원래 실행과 연결한다. actual constructor/import/API를 생성 자격에서 먼저 확인한다. 완전한 literal body JSON을 선택하고 network나 adapter references를 decode하지 않는다.

1. **B 저장 명령 재현:** E335 command/adhesion/tendon/stiffness 등 모든 action을 같은 rowc로 전달한다. row0은 restore 직후 held-action anchor이고 c1..300만 실행한다. NPZ command[0]과 restored last_action의 동등성을 새로 가정하지 않는다. command 계획 prefix와 actual anchor를 각각 확인한다. 301경계의 qpos/qvel/ctrl/physics clock 및15000 native time/load/slip/slip_valid/nonfoot/upright를 기존 증거와 finite bytes exact/NaN 위치 동일로 대조한다. NaN payload bit 동일성을 증명했다고 주장하지 않는다. 재현 실패는 scientific refutation이 아닌 technical_invalid이고 I/C는 실행하지 않는다. 기존 saved gait 요약을 새 arm의 측정값으로 복사하지 않는다.
2. **I 다리별 reference 형태:** c100..300의 source command를 기존 승인 encoder의 각 다리 z/R로 좌표화한다. finite/R>=.5/201행/6다리/order가 필요하다. 각 phi_l=atan2(z_y,z_x)/(2pi) mod1에서 승인 curves[6,128,7]를 periodic linear interpolation한다. reference에 해부학 offset이 이미 있으므로 추가 offset0이다. c<=100은 source bytes를 직접 사용하고 c101..119에서 기존 smoothstep((c*.005-.5)/.1), c>=120은 reference shape다. 나머지 source action은 B와 같다.
3. **C 저장 LF 공통 타이밍:** 동일한 curves와 이행을 사용하지만 여섯 다리 모두 같은 source LF global-reference phi를 사용한다. source phi를 이후 물리 출력으로 갱신하거나 새 clock/CPG/unwrap/phase 보정으로 대체하지 않는다. LF 선택은 사전에 고정하며 성과를 보고 다른 다리를 선택하지 않는다. 변화에는 다리 사이 관계와 LF를 제외한 다리의 순간 진행 속도/시간 순서가 함께 포함된다. 결과를 순수한 inter-leg coupling 인과로 부르지 않고 공통 타이밍 패키지 대조로 보고한다.

I/C는 c101 실행 전에 prefix0..100와 첫5000native의 exact 비교를 실제로 통과해야 한다. prefix 이전 health-stop 또는 불일치는 technical_invalid이고 남은 arm을 중단한다. 개입 뒤 q가 바뀌어 encoded phase 또는 실제 발의 위상이 그대로라는 주장은 하지 않는다. root/pose/중간 neural/body reset·외력 보정·adhesion 변경0이다. native health nonfoot0/upright>=.9/fault0/finite를 유지하고 prefix 통과 뒤 고정 health-stop는 scientific partial로 보존하며 해당 arm만 중단한다. 다음 사전 고정 arm은 원래 observed-start를 fresh restore해 실행한다. 기술/자원 실패만 남은 arm을 중단한다. 실제 settling/native requested/returned/unreturned를 보존하며 없는 full metrics는 null/not_tested, H348 full 비교는 not_evaluated다. 관측 geometry는 실제8점(coxa,trochanterfemur,tibia,tarsus1..5) 전체 native 좌표/순서로 수집한다. 옛 source와 좌표가 다르면 gait evaluator 수정/baseline 재평가를 구분하고 사후 수치 기준을 낮추지 않는다.

## 판정과 후속 경로

유효성은 B의 exact 재현, actual action schema/API, independent body restore, 모두300/15000 실제 요청·반환·확인, per-arm cleanup과 전체 종료 자격으로 닫는다. body-only이므로 whole BANC/CUDA neural 자격을 주장하지 않는다. 물리 관측은 기존 shadow mj_forward/contact 정의와8점 전체 trace로 유지한다. 지원 slip은 실제 .5<=t<1.5/load>=.02 mask이고 없는 값은0으로 채우지 않는다.

모든 arm에서 직접 forward, supported load-weighted slip, longest forward bout, 여섯 eligible complete cycles, health 및 shape/encoded coordination를 같은 evaluator로 측정한다. I-B와 C-I 차이를 함께 보고한다. H348 supported는 C가 I보다 forward>1e-9mm 더 크고 slip<−1e-9mm/s 더 작으며, C가 forward>=1.6790085779712631/slip<=1.8026436968249955/bout>=.3/각6cycles>0/health를 모두 만족하는 경우다. 둘 중 일부 개선은 partial/inconclusive로 보존하며 전체 가설 통과나 보행 모델 채택이 아니다. 개입 coverage/collector/order가 없으면 technical_invalid다.

H348 supported이면 최종 실행에 C를 직접 넣지 않고 해부학적 BANC의 공통 timing 학습 목표와 startup 제약을 함께 설계한다. I만 도움되면 근육별 출력 형태 교정이 우선이라는 근거로 사용한다. I/C가 유효하게 실패하면 같은 phase/blend/LF sweep을 하지 않고 contact/support와 학습 목표의 관계를 다시 결정한다. 어떤 분기도 F1–F8/생물학적 기전/자연 보행 완료가 아니다. 최종 실행의 외부 위상·재생·CPG 금지는 그대로다.

## 구현·자원·원본 접근 경계

ROOT 설계→actual GPT6Astra/max 리뷰→ROOT 반영→actual GPT6.1Sol/high FIRST→ROOT FIRST→필요시 SAME ONE R1→ROOT SECOND→잔여 ROOT로 진행한다. 새 tools/probe_c_banc_m3_shape_shared_timing_body.py와 tests/test_c_banc_shape_shared_timing_body.py 및 own e348-sol-*만 맡긴다. 기존 E337/E338/source3/E343/E347 및 원본 증거는 보존한다. ROOT가 exact selective descriptor를 metadata로 고정한다. E337 native static helpers(body_class/body_arguments/configure/install/boundary/applied_action 등)만 재사용하고 Dependencies.load/selected/gate/evaluate 및 H337 classifier는 호출하지 않는다. 새 selected loader와 literal body924, 고정 독립 metric encode/shape/physical 및 gait.load_model/analyze를 연결한다. startup/whole NPY 추가 읽기0이고 NPZ extras는 열지 않는다. actual gait dialect는 window.longest_forward_bout_s와 events[leg].complete_cycles다. caller 숫자만 바꾼 전체 inherited gate는 사용하지 않는다.

Sol FIRST+ONE R1 failed-inclusive generated35초/32MiB, ROOT generated35초/32MiB다. 원본 없이 합성3arm/science failure/extra unselected NPZ/서명 sentinel과 실제 same FlyBody snapshot→fresh restore→동일1control(50native)의 연속상태/native exact를 확인한다. 이 예정 검사 안에서 import/load, ctor/settling/restore, control50+shadow contact/boundary, cleanup/output 시간을 분리 기록한다. 3ctor/6000settling/45000science/900controls/903boundaries와 plan/evaluate/read/hash/IO를 포함한70science/85internal/95outer 및audit8 전망과 여유를 ROOT가 판단한다. 1control 외삽은 보장이 아니고 여유가 부족하면 원본예약0을 유지한다. 별도 timing campaign0이다. 실행 clock은 NumPy/library import와 input hash/read/restore 이전에 시작하며 final publication/file close까지 포함한다. 이는 whole BANC/원본 startup 자격이 아니다. 실제 세-arm 비교는 source/input freeze와 현재 자격 후 once original95/internal85(science70/final15)/audit8/retry0, CPU1/host2GiB/local1GPU·새 neuralGPU jobs0, model4.5초(+명시 constructor settling)/per-run<=31.5 모델·300wall 조건 안이다. raw16MiB/arm, stage64MiB, failure4MiB, stdout64KiB, free disk>=512MiB다. 현재 cumulative9780.954023399501/9900, 예약0/jobs0, 잔여119.04597660049876초이고 미래95+8은 그 안이다. 이전 실패 비용/미실행 예약은 초기화하지 않는다. 몸체/reference/control/native의 실제 허용 파일·선택 member 및 exact SHA/bytes·원자료 audit는 구현 전 확정해야 한다. 이 문서만으로 원본 읽기나 constructor 예약을 시작하지 않는다.


## actual Astra max 최종 리뷰의 ROOT 반영

2026-10-02 actual GPT-6 Astra/max 최종 M1–M3(리뷰 MD SHA2ad5415283338fd1bdbed7bc61659b3005790aa3bf4969f0e08d5e583e91fd16/JSON d31ca34d3aabaf6e054a5c81b2f0d420742c63fd6a31b89980701270ec906f52)을 위 계약에 반영했다. 원래 검토 설계는 parent-e348-reviewed-design-v1.md로 보존했다. 새 selected registration v2는 같은 과학3NPZ/finite JSON을 유지한다. 구현 전 원본 NPZ/NPY의 새 hash/header/value 접근, NN/body/CUDA 실행, 원본 예약은 모두0이다. 이 반영은 구현 자격이나 원본 실행 완료가 아니다. 다음 actual GPT-6.1 Sol/high FIRST 구현과 ROOT FIRST/SAME ONE R1/SECOND/잔여 ROOT 순서를 따른다.
