# E311: 고정 근육 경로와 가변 모집 전달의 제약 분리

Root 사전 설계, 2026-09-28. E310의 실제 paired continuation은 자원 후보의 지속 보행 이익을 보이지 않았다. 해당 u/tau branch는 확대하지 않는다. 같은 근육의 여러 운동뉴런 신호를 독립 합산하는 현재 전달에 상호작용을 추가할 가능성을 검토하되, 과거 E287 실패를 잘못 일반화하지 않는 작은 선행 판별만 수행한다.

E287은 **바뀐 E277 source의 고정 응답**에서 startup+계수 envelope가 충돌했다. Teacher는 목적함수에 있으며 feasibility 조건이 아니다. E150의 원래 응답과 SSP/u0는 시작 조건을 통과했으므로, E287의 불가능성을 이들에 옮길 수 없다. E308의 LM fixed muscle19 관측도 다른 source/관절의 원인으로 일반화하지 않는다. E175의 개별 MN 이력 계수와 E192 singleton 확대의 기각도 유지한다.

## 질문과 결정

H311: E287의 시작 제약 충돌에 기존 variable77 근육의 읽기 방식을 바꿀 자유도가 실제로 있는가. 특히 fixed19 근육만 연결된 관절이나, 가변 근육의 실제 입력이 모두0인 시점의 충돌인지 판별한다. 이는 회로/근육의 생물학적 원인 추정이 아니라 기존 공학적 출력 계열의 한계를 구분하는 질문이다.

- 고정 경로의 직접 위반 또는 아래의 더 넓은 활성 box 배제가 확인되면, E277 source를 그대로 둔 nonlinear 모집 fit을 시작하지 않는다. 시작을 통과했던 E150/neutral source에서의 상태 생성·근육 결합 설계는 별도 질문으로 남긴다.
- 배제되지 않으면 feasible 또는 새 후보의 개선으로 부르지 않는다. 실제 variable multi-MN 지원 범위를 근거로 후속 구조 하나를 별도 설계할 수 있을 뿐이다.
- 원래 자료 재구성/계보/증명 수치가 불일치하면 진단 무효로 남긴다. 자동 원본 재시도, 새 solver, fit, 신경/신체 실행은 없다.
- 어떤 결과도 F1–F8, E287 기각, E310 기각, startup/short 판정, M2/UI/F6를 바꾸지 않는다.

## 동결 입력과 범위

Base는 verification/banc-natural-gait-m3-20260919다. E287 result/decoder-solve.json/decoder-problem.npz/decoder-solve.npz/source-startup.npz/source-startup.json/cost.json, e287-source-decoder-fit-freeze.json 및 source review, parent-e287-original-final-review.json, parent-e287-saved-ray-audit.json, parent-e287-output-manifest.json을 pin한다. 정적 mapping은 e269-decoder-reference/engine-physiology-decoder.npz, leg axes는 e270-prepared-inputs/axes.npz로서 기존 E287 freeze의 참조 SHA와 일치해야 한다. E310 root audit와 결과 보고서는 분기 근거로만 pin한다.

필요한 원본 mapping 키만 읽는다: motor_ids, membership, pool, low/high/kappa, fixed, moment/rest/ltm, leg_motor/leg_muscle, target_startup_command. Full native stage, teacher 및 development 수치 배열은 읽거나 분석하지 않는다. E287 source-startup의 101개 bank/rate/original_activation/q/ltm와 정확히 control1..99만 사용한다. 같은 391MN/96근육/42관절/6다리, variable77/freeMN330/fixed19와 기존 membership·moment·fixed 정의를 검증한다. 0행은 원래 초기 상태이며 보호 target99행과 섞지 않는다.

## 계산

1. Saved sparse A/b/cone_dimensions와 raw solver_y로 원래 유한 정밀도 certificate를 재구성한다. Nonnegative/SOC dual margins, b·y, Aᵀy, [0,2] box upper 및 원래 1e−10 scale allowance가 parent-e287-saved-ray-audit와 일치해야 한다. 새 최적화는 없다. 3960변수/99×42행/원래 envelope 배치를 검증한다.
2. 현재 근육식으로 fixed offset o[t,q]=rest[q]+.02 M[q,fixed]·original_activation[t,fixed]−target[t,q]와 free-MN hinge design을 저장 startup_error_offset/startup_design에 대조한다. 순서/dtype/shape/finite/nonnegative 검증 뒤 max residual≤1e−13을 요구하고 실제 residual을 저장한다. 원래 low/high로 q를 다시 만들어 source-startup q와 대조한다. 기존 지원 0/nonzero 판정은 저장된 정확한0으로 하며 작은 값을 임의 삭제하지 않는다.
3. 모든 joint/time 행에 anatomical variable muscle 지원, 실제 nonzero additive basis 지원, fixed 근육 목록과 원래 MN membership을 기록한다. Dual peak 질량은 y+ + y−이며 정렬은 내림차순 후 row index 오름차순이다. 이는 상대적 constraint 관측이며 원인 ranking이 아니다. 전체99×42 raw 질량/offset/support 배열, 상위20행과 모든 직접 위반행의 provenance를 저장한다. 근육 이름을 임의로 추정하지 않고 원래 index/leg/MN ID로 표시한다.
4. 각 variable muscle의 시점별 자유 activation을 0≤a[t,m]≤U[t,m]=2 max(bank[t,해당MN,filter])로 완화한다. 이는 E287 양수 기여/총 slope≤2를 포함하는 **명시적 공학적 완화**이며 생리 범위가 아니다. 각 시간/근육을 독립 선택하고 MN floor/근육 lower/공유 coefficient 조건은 제거하므로 원래 가족을 포함한다. Fixed19 activation/LTM와 source bank는 그대로다. U를 초과하는 새로운 활성 모델/다른 source까지 배제하지 않는다.
5. .02 M_variable과 [0,U]로 각 q의 가능한 error interval을 계산한다. interval이 원래 ±Q_PEAK와 만나지 않는 행, anatomical variable지원0 행, 현재U0 때문에 조정 불가능한 행을 구분한다. 각 좌표의 최소절대잔차로 만든 낙관적 RMS lower bound도 기록한다. Q_PEAK=.03838315744033522/Q_RMS=.017763328718176728은 E287의 원래 내부 여유 기준이며 최종 native .04/.018과 혼동하지 않도록 둘 모두에 대한 직접 위반 여부를 별도 기술한다. 판별 allowance는 1e−10*(1+행의 절대 offset 및 U×moment 합)이다.
6. 원래 y의 startup peak/SOC 부분으로 새로운 box의 분리 상한을 계산한다. w[t,q]=y+[t,q]−y−[t,q]−ySOCvector[t,q], c[t,m]=.02 Σq w[t,q] M[q,m], B=Q_PEAK Σ(y++y−)+sqrt(99*42)*Q_RMS*ySOCscalar−Σ w*o. 가능한 weighted slack 상한은 B+Σ U*max(−c,0)이다. 원래 nonnegative/SOC dual 조건이 성립하고 이 상한에 1e−10*(1+절대 RHS 항 합+Σ U*abs(c))를 더해도 음수일 때만 이 명시적 box 배제를 지지한다. 상한이 음수가 아니면 inconclusive이며 feasibility 증명이 아니다. 모든 중간배열/항/원래 envelope를 버렸다는 설명을 저장한다. 새 cone projection/solver/dual 보정은 없다.

행별/전체 box 배제는 같은 source와 fixed19, U 범위에서만 유효하다. Teacher 정확도, 신경 정보 유무, 자연 보행의 불가능성, 특정 근육의 인과 효과는 이 진단의 결론이 아니다. 특히 독립 시점의 activation 완화는 실행 가능한 controller가 아니다.

## 구현·검사·비용

Root 설계 → 실제 gpt-6-sol/xhigh 구현 → root FIRST → 필요 시 같은 Sol R1 → root SECOND → root 잔여 수정. 기존 certificate 부호/행 배치와 더 넓은 가족의 포함성, source 구분에 대한 검증 난이도로 xhigh를 사용한다. 새 tools/analyze_c_banc_m3_decoder_obstruction.py 및 tests/test_c_banc_decoder_obstruction.py만 소유한다. 기존 실행기/평가기/원본은 읽기 전용이다.

순수 generated 검사는 작은 exact fixture로 fixed-only 위반, mixed variable support, zero-input U0, signed moment/box 구간, shared-coefficient 제약을 없앤 relaxation, dual 부호와 SOC scalar, 작은 negative upper와 allowance, 양수 upper의 inconclusive, 잘못된 row/dtype/input hash/status, 저장 실패의 원래 오류 보존을 검증한다. 새 실제/생성 solver 모두0이며 fake 검사는 모든 시도 합30초/CPU1/host1.5GiB다. 구현자는 원본 수치 분석을 수행하지 않는다.

CLI는 --freeze --freeze-sha256 --source-review --source-review-sha256 --output으로 고정한다. Freeze schema flylab.m3.e311.decoder-obstruction.freeze.v1, review/result 대응 v1이다. Fresh output e311-decoder-obstruction. Source/input/design/package/version/preservation pins, 원래 producer semantic 상태를 검사하고 분석 시작/끝/실패/원자료/한계와 결과·manifest를 남긴다. 원래 첫 오류와 후속 게시 오류를 보존하고 실패 시 success seal을 발행하지 않는다. Source gate 전 원본 예약0이다.

현재 원본 누적8416.23501749954/8560초·잔여143.76498250045915초. Root source review 이후 저장자료 원본 분석1회/재시도0에 supervisor20초/internal15초+finalization5초, CPU1/host1.5GiB/output32MiB/free256MiB를 현재 잔여에서 예약한다. 새 model/neural/body/physics/CUDA/fit/solver/QR0이다. 알려진 지원·생성 비용과 unknown은 별도 보존하고 이전 비용을 초기화하지 않는다. 이 단계만 끝나도 자연 보행 목표는 계속 active다.
