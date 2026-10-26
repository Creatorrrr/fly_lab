**ROOT가 Astra 리뷰를 반영한 후속 설계: 기존 R 방향의 유한 척도를 startup 전체 구간에서 검사하고, 배포 가능한 후보 하나를 실제 보행 평가로 넘긴다. 현재는 소스 검토이며 실행 권한이나 검증 결과가 아니다.**

[최초 설계](BANC_M3_STARTUP_CONSTRAINED_RESEARCH_DESIGN_20261004.md)는 리뷰 입력으로 보존한다. 실제 GPT-6 Astra/max의 [전체 리뷰](../verification/banc-natural-gait-m3-20260919/parent-e360-startup-research-astra-review.json)23,713B/SHAf21f98497e05a73b2630c633955d4a6a76d350732f3d3b409e47d96279f20e3d와 [읽기용 리뷰](../verification/banc-natural-gait-m3-20260919/parent-e360-startup-research-astra-review.md)6,684B/SHA859a549c0eb9058c5f172cf026ea52d42913af9401f6163049a5293f049ee27b를 ROOT가 전체 재독하고 descriptor를 확인했다. SR-01..06을 아래에 반영했다. 리뷰가 실행이나 Sol 구현 권한을 추가하지는 않는다.

현재 승인된 P10 한 번은 첫 Ruff exit1/B023 14건으로 닫혔다. 실제0.885초를 한 번 차감했고 비용83개와 원래 invalid capture proof·독립 owned closure를 보존했다. [P10 결과](BANC_M3_E360_P10_RESULT_20261004.md)와 [리뷰된 P11 제안](BANC_M3_E360_P11_REVIEWED_PROPOSAL_20261004.md)은 유지한다. P11의 직접 범위 승인은 아직 없고, 이 연구는 E360 생성 검증 복구를 대체하지 않는다. 새 배정·P11 통합·추가 runtime·원본 실험은 이 문서에서 승인하지 않는다.

첫 과학 질문은 **startup 조건과 기존 R 개선 조건을 만족하는 유한 척도의 효능 후보가 실제1.5초 보행 평가에서 적격인가**다. 같은 θ=1과 후보 하나의 비교는 제한된 실행 가능성을 판단하며 기존 선택법보다 성공 확률을 높였다는 인과 주장을 입증하지 않는다. 해부학적 roster/graph/topology/부호·8,547개 효능 그룹/725,849개 선택 edge·신체·마찰·decoder·네 R 창100/164/228/292 및 기존16×42 정규화는 유지한다. 시작 이후0.5초의 저장 창에서 학습하면서 후보를 처음부터 적용하는 현재 차이를 다음 가설의 근거로 삼되 확정된 보행 실패 원인으로 표현하지 않는다. 시간상수·감각 필터·외부 리듬·다리 협응 모델을 동시에 바꾸지 않는다.

첫 설계의 다중 제약 QP는 후순위로 둔다. `banc_m3_checkpointed_rollout.py:286–298`은 한 rollout에 대한 반복 backward와 higher-order 미분을 명시적으로 거부한다. RMS 두 개와 여러 max 성분의 VJP를 구하려면 별도 forward/recompute/backward 또는 새 연산자 구현과 검증이 필요하다. 그 비용을 기존 그래프 재사용으로 작다고 가정하지 않는다. 현재는 기존 네 R 창의 독립 backward만 유지한다.

최소 변경의 계산 순서는 다음과 같다.

1. 동일 neutral SSP checkpoint·body/adapter/graph와 시작 tick0을 고정한다. 기존 E318의 시작 상태·full drive·endpoint가 이 경로에 재사용 가능한지 먼저 확인한다. baseline θ=1의 전체 startup99 재현이 적격이어야 후보 선택을 시작한다.
2. 기존 합산 R gradient에서 `d=-g_R/max(abs(g_R))`를 한 번 구한다. 실행 전 등록한 양의 유한 척도 목록을 **내림차순의 고정 순서**로 검사하고 `θ_requested = 1 + α d`를 제안한다. 요청 척도는 `0<α≤0.001`이다. 목록 길이·검사 수·중단 규칙·R 개선 판정·정량 대응 오차 허용값은 실행 중 고르거나 바꾸지 않는다. 작은 척도일수록 안전하다고 가정하거나 이분 탐색하지 않는다.
3. 각 제안은 양의 최종 float32 θ를 float64로 정확히 승격한 값과 실제 배포 float32 선택 가중치로 평가한다. **ROOT의 후속 등록 권고는 요청 bound와 실제 bound를 각각 검사해 `max(abs(θ32→64-1))≤0.001`도 요구하는 것이다.** 반올림 때문에 이 실제 bound를 넘으면 해당 제안을 제외하고 θ를 재클리핑하지 않는다. 요청0.001을 실제 float32 bound로 착각하거나 숨은 허용오차를 더하지 않는다. 요청값/실제 δ를 모두 기록한다. 반올림 후 같은 θ 또는 같은 배포 가중치가 되는 중복 제안과 실제 가중치 무변화, nonfinite·부호·unselected edge 변경·기존16,384 palette 제한 위반도 등록한 규칙에 따라 제외한다. 정수/float 타입이나 원래 곱셈·반올림 계약을 몰래 바꾸지 않는다. 이 정책은 후속 계약에 명시해야 하며 과거 실행의 판정에 소급 적용하지 않는다.
4. 고정된 시작 상태와 기준 감각 drive를 사용해 **전체99 controls의 forward**를 수행한다. max_controls=4인 기존 R 창을 몰래99로 확장하지 않고, startup용 명시적 trajectory를 별도로 연결한다. 새 constraint VJP나 신체 미분은 도입하지 않는다. 네 기존 R 창의 실제 rounded forward 손실과 startup 네 제한을 함께 검사한다.
5. 등록한 선택 규칙에 따라 적격 제안 하나만 선정한다. 저장된 입력에서 선정 가능한 제안이 없으면 **등록한 척도 목록에서 후보를 찾지 못함**으로 기록한다. 방향을 회전하지 않는 이 검사는 모든 가능한 효능 방향의 부재나 신경 리듬 부재를 증명하지 않는다. 평가 후 목록을 연장하거나 실제 신체를 여러 번 돌려 좋은 후보를 고르지 않는다.
6. 선정된 동일 후보를 θ=1 대조군과 실제 폐루프에서 비교한다. **neutral과 candidate의 전달 증명을 구분한다.** 후보 전용 `candidate_body_handoff.py:1154–1166`은 바뀐 network identity/CSR를 요구하므로 θ=1에는 그대로 사용할 수 없다. 기존 `startup_baseline.py:784–838`의 neutral source equality와 기존 후보 distinct-identity 검사를 각각 보존한다. 각 인스턴스를 처음부터99 controls까지 진행하고 startup 적격이면 같은 인스턴스를300 controls/15000 native steps까지 이어가는 범위를 새 계약에 등록한다. 기존 neutral `run_prefix`의 정확히99 lifecycle을 조용히 연장하거나 후보 검사를 느슨하게 만들거나 바뀐 neutral identity를 꾸미지 않는다. 실제 감각 입력이 달라지는 경로는 이 단계의 결과가 판정한다.

유한 척도 목록의 내부 forward 제안과 실제 신체 후보1개를 구분해 비용과 시도 횟수를 등록한다. 유한 목록의 성공을 global feasibility나 실제 보행 개선으로 표현하지 않는다. 실제 신체에서 실패한 뒤 척도·방향·decoder·마찰을 바꾸면 새 후보/modified retry이며 이전 자격과 실행 권한을 승계하지 않는다. finite ray에서 효과가 없으면 QP·회전한 방향·연속 학습은 각각 별도 가설과 자원 범위로 제안한다.

기존 startup 조건은 q RMS≤0.018/q max≤0.04/tendon RMS≤0.009/tendon max≤0.025다. 이것은 **관절 명령의 오차**이며 기하학적 무릎 운동폭이 아니다. 원래 startup99처럼 boundary0를 제외한 명령1..99를 원래99행 목표와 비교하며 모든 구간에서 네 조건을 계산한다. baseline/candidate의 surrogate 오차와 actual 오차를 구분한다. VJP는 unrounded 방정식의 미분이며 실제 float32/FTZ/kink·감각·물리 폐루프 전체 Jacobian이 아니다.

E318은 재사용 검토의 구체적 후보가 있다. 현재 text/metadata에서 `e318-full-drive-capture-freeze.json`의 initial whole 경로와 network/adapter/graph 식별자가 E307 neutral 및 E345 입력 inventory와 연결된다. `result.json`은 controls300·12 chunks·chunk_saved_controls300·run_valid/learning_input_approved를 기록하고, 기존 raw-audit의 result SHA도 현재 result JSON과 일치한다. capture 소스에는 control0 시작 상태와1..300 full drive·endpoint 저장 경로가 있다. 이는 과거 선언·감사의 연결 확인이고 이번 원본 배열의 값·header·SHA·연속성 재검증이 아니다. 따라서 새 baseline capture부터 필요하다고 가정하지 않는다.

E307 freeze의 neutral_utilization=0.0과 first-control 소스의 `zeros_like` neutral selection 설정, E318 소스의 control0 resource=1 검사는 fixed-resource 조건의 근거다. 일반 SSP wrapper가 모든 자산에 u0를 강제하는 것은 아니므로, 후속 승인된 원본 검토에서 실제 resolved selection·resource 초기 상태·중간 step 및 그래프 경로를 확인한다. 검증된 초기 rate/adaptation/bank/original_activation·mask/설정과 순서가 맞는 drive가 필요하며 모든 boundary 상태의 별도 저장을 필수라고 가정하지 않는다. 그 전에는 resource 누락을 확정된 수치 버그라고 부르거나 baseline 동등성을 완료했다고 기록하지 않는다.

E358/E359는 technical_invalid/unqualified로 보존한다. E359 producer qmax0.0396360246919627을 적격 θ=1 대조군이나 robust margin으로 쓰지 않는다. 저장 입력의 대응·소스/감사/clock/deadline 실패는 보행 가설 inconclusive인 technical-invalid다. 유한 목록 내 미선정, 유효한 startup 기각, 유효한 보행 기각, short-screen 적격을 각각 구분하고 원래 실패를 새 source review로 소급 유효화하지 않는다.

실제1.5초 screen은 기존 전진≥1.6790085779712631mm/slip≤1.8026436968249955mm/s/bout≥0.3초/여섯 다리 각각 적격 주기>0/native health/startup 조건을 유지한다. AEP→PEP→AEP 주기는0.3초 이상 전진 bout 안에서 정의된 기존 지표다. 적격 주기0을 모든 신경 리듬·움직임의 부재로 해석하지 않는다. 상세 위상·기하학적 무릎·CV를 보존하되 단기 관측으로 F4/F5 장기 기준을 통과했다고 주장하지 않는다.

단기 적격 이후에는 같은 θ·checkpoint·body·decoder로10초/F3–F5→미사용 초기 상태2개/F6→기존6목표·정지/재시작/F7→F2 저장 상태의 미래 재현·신경 출력 차단 및 F8 raw/provenance/영상/산출물을 검증한다. F3 순전진10mm/directness0.5/upright0.9/nonfoot0, F4 다리별10주기·좌우0.5±0.15cycle·R≥0.5, F5 LF25°/RF30°/LH40°/RH35°·CV≤0.3를 유지한다. M2/UI와 해부학적 provenance 및 과거 후보·실패는 보존한다. 교사·소스·lint·loss·startup·1.5초만으로 자연 보행이나 생물학적 기전의 완료를 선언하지 않는다.

자원은 기존 원장83/shared1400·local1289·잔여896.0529844/896.010초를 유지한다. 원본 cumulative10020.115960899504/10200·stage126.361/300·E360 local0/20·worker 최대1/auto0도 그대로다. 기존99-control prefix46.640/48.295초는20초 지역 원본 cap보다 크며, E318 저장 비용을 현재 계산/신체 비용으로 대체하지 않는다. 유한 ray full99 forward·paired body1.5초·저장/close/감사/최종 반환 비용은 미측정이다. 현재 잔여에 들어간다고 주장하지 않으며, 사전 등록된 입출력·후보/modified-retry·실제 full-return once debit 범위가 충족되기 전 계산하지 않는다. 이 연구 검토에서 Python/import/AST/compile/Ruff/test/probe·원본 NPZ/NPY/checkpoint/array header/hash/value·CUDA/물리·UI/서비스 실행은0이다.

후속 독립 변경은 실제 startup/단기 결과를 보고 결정한다. 지속성이 부족하면 상태를 이어가는 학습과 후보가 방문한 상태를 포함하는 방향을 우선 검토한다. [DAgger 원문](https://proceedings.mlr.press/v15/ross11a.html)의 관측 분포 문제는 그 근거이며 현재 cadence 교사가 모든 학생 상태에 유효 행동을 주는 expert라는 증거는 아니다. [Pugliese 공식 저장소](https://github.com/smpuglie/Pugliese_2026)는 VNC 동역학/JAX 구현의 자료로 사용한다. 그 저장소를 현재 신체 보행 성공의 증거로 쓰거나 CUDA 경로를 곧바로 교체하지 않는다. Sapkal 원문은403으로 독립 확인하지 못했으므로 기전·뉴런 ID·새 기준을 직접 채택하지 않는다.

설계 리뷰 SR-01의 첫 방법 축소, SR-02의 E318 재사용, SR-03의 조건부 u0, SR-04의 neutral/candidate 증명 분리, SR-05의 배포·전체 구간·고정 선택 규칙, SR-06의 제한된 인과/판정을 반영했다. 게시 후 전달된 반올림 bound 사항도 위의 실제0.001 검사 정책으로 명시했다. 최초 설계와 리뷰는 수정하지 않는다.

다음 구현 순서는 실행 복구·자산·범위가 준비된 후 ROOT의 반영 설계→적절한 effort의 실제 GPT6.1Sol FIRST→ROOT FIRST→필요하면 같은 ONE R1→ROOT SECOND→ROOT 잔여다. 새 리뷰나 전달 메시지를 실행 권한으로 쓰지 않는다. 현재는 새 연구 근거와 실제 리뷰를 반영한 독립 진행이 있으므로 전체 M3 goal은 active/incomplete다. P11의 pending 범위 승인과 F1–F8 미완료 상태를 유지한다.
