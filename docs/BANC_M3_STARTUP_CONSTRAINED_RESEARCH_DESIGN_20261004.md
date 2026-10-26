**ROOT 설계: 다음 과학 변경은 기존 신경 효능의 방향 선정에 startup 제약을 넣는 것으로 한정한다. 현재는 연구 검토·제안 단계이며, P11이나 새 원본 실험을 승인하거나 실행하지 않는다.**

현재 등록된 P10은 첫 Ruff exit1/B02314건으로 닫혔고 결과·전체 로그·실제0.885초의 once debit·원래 invalid capture proof·독립 owned closure를 보존했다. [P10 결과](BANC_M3_E360_P10_RESULT_20261004.md)와 원장83은 그대로다. P11은 아직 proposal이며 직접 승인/새 literal/activation/request/실행이 없다. E360의 생성 검증 및 원본 진입 조건을 마무리하는 경로를 이 연구로 교체하지 않는다. 본문은 그 다음 과학 단계의 설계 초안이다.

전체 finite-goal은 기존 활성 M3/F1–F8다. 동일 후보의 시작→실제1.5초→10초/F3–F5→미사용 초기 상태2개/F6→기존6개 목표·정지·재시작/F7→신경 출력 차단·저장 상태 미래 재현/F2·산출물/F8을 유지한다. BANC 해부학적 roster·연결·provenance·M2/UI·과거 실패·자원 회계를 보존한다. 교사·저장 손실·startup·단기 적격·생성 검증을 자연 보행이나 생물학적 기전의 완료로 승격하지 않는다.

2026-10-04 전달된 [개발 적용 검토](researches/BANC_WALKING_RESEARCH_20261004.md)와 [ZIP 첨부 후 전체 답변](researches/BANC_WALKING_RESEARCH_20261004_GPT_PRO.md)의 A–E·최종 권고를 읽었다. [근거집](researches/BANC_RESEARCH_EVIDENCE_20261004.md)과 [출처](researches/BANC_WALKING_RESEARCH_20261004_SOURCE.json), [첨부 무결성 기록](researches/BANC_ATTACHMENT_INTEGRITY_20261004.json)은 입력 자료로 보존한다. 원문 뒤쪽은 브라우저 본문에서 Markdown으로 옮긴 자료이며, 130개 첨부 일치 주장은 첨부 내부 검사다. 이번 ROOT 검사는 현재 로컬8개 소스의 SHA가 기록된 스냅샷과 같다는 확인이다. ZIP 전체를 다시 읽거나 과거 실행을 소급 유효화하지 않았다.

현재 코드로 확인한 판단은 다음과 같다.

| 확인 | 현재 근거 | 다음 결정 |
|---|---|---|
| 방향 선정은 합산 R gradient 정규화이고 startup 입력이 없다. | reference_parameter_direction.py:327–341,914–924 | startup을 먼저 검사하는 기존 평가기에 더해 방향 제안 때도 제약을 고려한다. |
| 네 창100/164/228/292는 저장된 rate/adaptation/bank/activation과 감각 drive에서 각각20ms를 진행한다. | 같은 파일:43,786–800; coordination_parameter_direction.py:1217–1224 | 기존 R 목적의 네 창·16×42 정규화는 첫 비교에서 고정한다. startup 제약용 구간을 별도로 구분한다. |
| 실제 후보는 처음부터 적용되고 control99에서 startup을 판단한 뒤 같은 인스턴스를300까지 이어간다. | candidate_body_handoff.py:1251–1281 | 0.5초에 재설정하거나 교사 warm-start로 검사하지 않는다. |
| 적격 주기는0.3초 이상 전진 bout 안의 AEP→PEP→AEP다. | analyze_c_banc_biological_gait.py:180–240 | 주기0을 모든 신경 리듬 부재로 해석하지 않고 기존 평가 정의를 유지한다. |
| rounded forward의 VJP는 별도의 실수 방정식 미분이다. | grouped_current_adjoint.py:1–5,82–114; ssp_adjoint.py:1–5 | float32/FTZ/정류 kink·물리·감각 폐루프 전체의 정확한 Jacobian으로 주장하지 않는다. |
| make_candidate는 원래 SSP 설정과 동적 상태를 전달하며, patch_csr는 원래 topology·미선택 edge·부호를 보존한다. | ssp_efficacy_trial.py:34–114 | 최종 배포 theta와 선택 가중치의 정확한 바이트를 기준으로 평가한다. palette≤16384를 유지한다. |
| 현재 evaluate는 이미 계산한 gait 결과의 summary 일부만 반환한다. | common_descent_body.py:913–983 | 후속 구현에서는 이미 계산된 위상·기하학적 무릎·주기CV를 함께 보존할 수 있다. 판정 기준은 바꾸지 않는다. |

ZIP의 E-1에 없던9개 SSP/adjoint/decoder/rollout/평가 소스는 현재 로컬에 모두 있다. 존재·텍스트 SHA를 확인했고 주요 호출과 수식을 읽었다. 정확한 resolved checkpoint·가중치·startup raw 값은 이번에 읽지 않았다. 현재 u0라는 trial 문서 설명만으로 실제 자산의 모든 utilization=0/resource=1을 확정하지 않는다. actual SSP에는 resource 상태가 있고 adjoint rollout의 전달 상태는 rate/adaptation/bank/activation이므로, 해당 고정 조건과 영향은 승인된 자산 검토에서 먼저 확인해야 한다. 기본 M2 adapter의 필터·시간상수도 현재 SSP 경로의 실제 설정으로 자동 대입하지 않는다.

E358/E359는 모두 technical_invalid/unqualified를 유지한다. E358 producer qmax0.04220501637004681과 E359 producer qmax0.0396360246919627은 startup 문제의 단서다. 두 실행을 적격 신체 비교나 유효한 theta=1 대조군으로 사용하지 않는다. 특히 E359의 여유0.0003639753080373rad를 새 후보의 허용 변화량이나 검증된 robust margin으로 채택하지 않는다. 최대 오차의 시간·관절은 raw를 읽지 않아 미확인이다.

첫 변경의 가설은 “같은 신경·그래프·decoder·신체에서 startup 제약을 고려한 효능 방향은 기존 R 개선을 유지하면서 실제1.5초 판정까지 도달할 가능성을 높인다”다. 유일한 주요 조작은 방향 선정 규칙이다. 노드/edge/8,547개 그룹/725,849개 선택 edge/목표·신체·마찰·decoder·현재 네 R 창을 고정한다. baseline은 같은 SSP 체크포인트의 theta=1이고 후보는 양의 float32 theta 하나다. 이 단계에서 시간상수·감각 필터·글루탐산 부호·리듬 모델·GPU 물리를 동시에 바꾸지 않는다.

방향 제안은 기존 R gradient g_R에 대해 box 범위 ||delta||_infinity≤0.001 안에서 g_R·delta+(lambda/2)||delta||²를 줄이고 startup 네 제한의 국소 근사를 만족시키는 작은 제약 문제를 사용한다. 기존99-control 전체의 q RMS≤0.018/q max≤0.04/tendon RMS≤0.009/tendon max≤0.025를 유지한다. 이 q는 관절 명령 오차이며 F5의 기하학적 무릎 운동폭이 아니다.

거대한 전체 Jacobian을 새로 구축하지 않는다. 처음에는 두 RMS 제약과 baseline 최대 오차에 가까운 양·음 성분의 유한 active set에 대한 VJP를 이용해 방향을 제안한다. 최대 성분만 안전하다고 판단하면 다른 시간·관절이 새 최대가 될 수 있으므로, 최종 theta를 float32로 반올림한 뒤 startup99 전체 명령/tendon의 surrogate 재평가로 네 제한을 확인한다. 자산·제약·예산이 갖춰질 때 bounded proposal 내부에서만 후보를 선정하며 실제 신체 후보는 하나다. lambda·active-set 규모·근사/실제 오차 허용 정책·부동소수점 no-change·palette 실패는 실행 전 등록에서 고정해야 한다. 이번 문서는 이 숫자를 이미 검증된 값으로 만들지 않는다.

startup 근사는 실제 시작 상태와0→99의 상태 연속성·기준 drive가 확보된 경우에만 구성한다. 기존 assemble(max_controls=4)를 몰래99로 바꾸지 않고, 같은 모델 수식을 쓰는 별도 명시적 startup-constraint trajectory로 연결한다. 네 R 창과 손실 정규화는 그대로다. 신체를 미분하지 않는 저장 drive 근사이므로 실제 감각 피드백으로 달라지는 경로는 최종 신체 검사에서 확인한다. 근사가 actual native forward와 맞지 않거나 유의미한 배포 가능한 방향이 없으면 hypothesis inconclusive/no feasible direction으로 기록하고 threshold 완화·외부 시동·추가 후보·숨은 warm-up을 넣지 않는다.

필요한 다음 증거는 등록된 same-source/같은 neutral SSP checkpoint, startup99 목표 명령·tendon·0→99 기준 감각·상태의 확보 여부, fixed resource 조건, actual/adjoint의 finite perturbation 일치, 최종 f32 theta/배포 선택 가중치·palette, 같은 후보를 넘기는 body_data/initial_body binding이다. 모든 원본 배열·header/hash/value IO와 계산은 별도 실제 승인·ROOT admission 및 actual full-return once debit이 필요하다. 이번에는 text/hash/metadata만 읽었고 원본 NPZ/NPY/checkpoint·CUDA/물리·Python/import/AST/compile/Ruff/test/probe를 실행하지 않았다.

실제 paired body 검사는 theta=1과 후보 하나 각각을 처음부터99 controls까지 진행한 뒤 startup 적격이면 같은 인스턴스를300 controls/15000 native step까지 이어간다. 현재 screen() 그대로 전진≥1.6790085779712631mm/slip≤1.8026436968249955mm/s/bout≥0.3초/여섯 다리 주기 각각>0/native health/startup 네 조건 모두를 확인한다. 상세 위상·무릎·CV 보존은 관측을 보완하는 것이며1.5초의 적은 주기로 F4/F5의 장기 기준을 충족했다고 표현하지 않는다. technical_invalid, qualified startup rejection, qualified gait rejection, short-screen qualified를 구분한다.

단기 적격이면 동일 theta·checkpoint·신체·decoder로10초/F3–F5를 확인하고 미사용2초기 상태와 기존6목표·정지/재시작, F2/F8을 이어간다. 10초에서 실패한 뒤 theta/decoder/마찰을 바꾸면 새 후보이며 이전1.5초 자격을 승계하지 않는다. F3 순전진10mm·directness0.5/upright0.9/nonfoot0, F4 다리별10주기·좌우0.5±0.15cycle·R≥0.5, F5 LF25°/RF30°/LH40°/RH35°·CV≤0.3를 그대로 유지한다.

실행 자원은 현재 P11 범위와 분리한다. 기존 generated shared1400/local1289와 잔여896.0529844/896.010, full83 및 원본 cumulative10020.115960899504/10200·stage126.361/300·E360 local0/20을 유지한다. 현재 원본 worker 최대1·준비5/4.5·worker12/11/10/9·seal2/1.5는 변경하지 않는다. 실제99-control prefix46.640/48.295초가 기존20초 지역 원본 한도보다 크고 paired1.5초·startup 민감도 비용은 미측정이다. 따라서 이 과학 비교가 현재 잔여나 cap에 들어간다고 주장하지 않는다. 후속 실행 전 해당 경로의 준비→계산→신체→저장·close→감사→최종 반환을 포함한 구체적 범위·자원·원본 IO·후보/modified-retry 계약을 등록하고 필요한 추가 agreement만 요청한다. 지금 새 배정이나 실험을 요청하거나 실행하지 않는다.

그 다음 독립 변경의 우선순위는 startup과 실제 단기 결과를 보고 결정한다. 지속성이 부족하면 상태를 재설정하지 않는 연속 학습 및 후보가 방문한 상태를 포함한다. 그 이후 길항 모집→관절 방향·고유감각/하중·다리 간 경로·하행 명령을 순서대로 점검한다. [DAgger 원문](https://proceedings.mlr.press/v15/ross11a.html)은 행동에 따라 바뀌는 관측 분포에서의 모방학습 문제를 다룬다. 현재 고정 cadence 교사를 임의 학생 상태에서 유효 행동을 주는 expert로 취급하지 않는다. [Pugliese 공식 저장소](https://github.com/smpuglie/Pugliese_2026)는 VNC 신경 동역학/JAX 구현을 제공하며, 이를 현재 신체 평가의 성공 증거로 쓰거나 현재 CUDA rate 경로를 JAX로 교체하지 않는다. Sapkal 원문 직접 접근은403으로 실패했고 이번에 기전 세부 주장을 독립 확인하지 못했다. 버전별 뉴런 ID·새 생물학 수치·외부 코드 라이선스/환경/체크포인트를 즉시 채택하지 않는다.

다음은 실제 GPT-6 Astra/max의 이 설계·현재 소스 리뷰다. ROOT가 반영한 뒤, 실행 복구·자산·과학 범위가 갖춰지면 GPT-6.1 Sol에 적절한 effort로 구현을 맡기고 ROOT FIRST→같은 ONE R1→ROOT SECOND→ROOT 잔여 순서를 지킨다. 소스 준비·논문·리뷰·자동 goal continuation은 새 실험 authority가 아니다. 현재 새로운 연구 자료로 판단을 보완할 독립 작업이 생겼으므로 blocked를 지금 선언하지 않으며 전체 goal을 active/incomplete로 유지한다. F1–F8의 실제 효과가 확인돼야 complete이고, 동일 blocker가 세 연속 회차에 반복돼도 의미 있는 독립 진행이 없을 때만 blocked로 판단한다.
