# E330: 새 시간 목적의 실제 BANC 효능 방향 검증

2026-10-01 현재 세션 설계. `parent-e330-astra-design-review.md`의 actual gpt-6-astra/max 리뷰를 아래 계약에 반영했다. E329의 actual GPT-6.1-Sol/xhigh 구현·root 자격 절차가 진행 중이므로, E329 자격이 완료되기 전에는 E330 원본 실행이나 구현을 시작하지 않는다. E327의 실제 신체 기각과 F1–F8 전체는 불변이다.

## 질문과 범위

H330: 현재 SSPRK33/u0 전체158706 BANC에서 E329의 successor+all-15-pair 목적에 대한 50개20ms 조건부창 미분이 실제 float32 크기비례 효능 perturbation의 국소 방향을 설명하며, 같은 방향의 reset 없는1.5초 신경 실행이 새 손실을 줄이고 기존 시작 네 제약을 지킨다.

이는 현재 rounded native forward에 붙인 real-equation surrogate VJP가 실제 f32 유한 방향을 설명하는지 검증하는 단계다. f32 양자화·FTZ 자체를 정확히 미분했다고 주장하지 않는다. 저장 감각 drive 조건의 신경 실행이므로 신체의 지속 보행·감각 폐루프·F3–F8의 성공으로 해석하지 않는다. 성공하면 같은 theta의 실제 신체 E331로 진행한다. 실패하면 실패값을 보존하고 원인을 분리한 다음 근거 있는 다음 설계를 한다. E323의 full-horizon BPTT를 반복하지 않는다.

## 입력과 불변 조건

- E318의 원래 control0, 원래 300개 held sensory drives/rates/adaptation/motor rows, 원래 100/300 whole checkpoints를 사용한다. E315/E318의 현재 SSP 전체 신경 identity, E29 8547개 P→P efficacy groups와725849 selected CSR edges, E150 decoder, physics/body mapping을 변경하지 않는다.
- E329의 fixed E85 M128 closed curves, frequency2.382760007990865Hz, dt.005, successor 정상화와 all15pair 계수1을 사용한다. runtime에 목적함수/교사/phase를 넣지 않는다.
- E329 source/root CPU+CUDA/실제 reference+saved characterization이 승인되어야 한다. 실제 reference phase 재현 또는 저장 입력 비국소 동률이 자격을 실패하면 gate에서 원본 실행을 거부한다. E327보다 점수가 낮아야 한다는 사후 기준을 두지 않는다.
- 모든 과거 source와 결과는 읽기 전용이다. 새 runner와 생성 tests만 구현하고 현재 checkpointed rollout·SSP adjoint·monotone adjoint·old loss를 수정하지 않는다. M2/UI/F6 및 실제 신체는 이 단계에서 비노출이다.

## 실행 순서와 판정

1. exact input/source/qualification/review hash, fresh output, limits와 고정 paths를 pre-read gate에서 검사한다. 완료된 E329의 objective/CLI/tests, root 최종 review·generated CPU/CUDA 자격·saved result/contract와 기존 E324/E323 source/input closure를 연결한다. 각 승인과 결과가 현재 source/E85/E103/E318 SHA를 실제로 가리키는지 검증한다. E329/E330의 최종 curves[6,128,7]f64 bytes hash, leg/joint order, normalization, frequency/dt/all15 계수/steady 범위도 고정한다. original child process 안에서 passive gate 후 loader/CuPy 전에 torch CUDA 초기화를 수행한다. 부모의 초기화로 이를 대신하지 않는다.

   E329의 CPU참조 정상화 f64 vector는 JSON에서도 exact round-trip되며 그 bytes SHA를 확인한다. E330의 GPU constructor에서 재계산한 vector가 이 값과 rtol1e-12/atol1e-15로 일치하는지 검사하고, rollout 이전에 고정된 E329 vector를 GPU tensor로 옮겨 사용한다. CPU/GPU reduction의 작은 반올림 차이를 새 학습/다른 reference bytes라고 숨기지 않는다. 이 값은 학습된 변수가 아니라 승인된 고정 s_l이다. 같은 곡선/정상화 source 및 실제 GPU 사용 bytes를 결과에 기록한다.
2. 기존 E324 loader/build 경로를 읽기 전용으로 재사용하되 새 목적만 추가한다. 고정 decoder를 원래 motor rows로300control 재생하여301개의 bank/original 경계를 구성하고 q/tendon은 E318과≤1e-11rad/command, bank/original≤1e-10으로 재현해야 한다. decoder replay 자체는 SSP를 실행하지 않는다. SSP rate/adaptation byte/dtype exact 확인은 다음 중립 창의 endpoint에 적용한다. 중립0→300연속 SSP 재현은 E318/기존 승인 증거를 재사용하며 이번에 새로 실행했다고 쓰지 않는다.
3. 시작 c=100,104,...,296, 각4control을 저장 r/z/bank/original로 시작해 원래 drives[c:c+4]로 실행한다. neutral q/tendon/endpoints는 원자료 및 재생 bank와 위 허용오차로 재현한다. q_previous=concat(raw.command[c],output.q[:3]),q_next=output.q다. 경계만 고정하고 내부 previous/target 미분을 유지한다. 각 손실/50에 backward를 걸어50개8547 gradient를 합산한다. 각 창 원자료·receipt·항별 loss·gradient를 저장한다.
4. neutral 50창의 training/successor/coordination/LR 각각 평균값은 E329 저장 neutral의 full200pairs 항과≤1e-10으로 일치해야 한다. 이 동일성은 값에만 적용한다. 조건부 gradient를 full-horizon 미분이라고 부르지 않는다. finite exact0 gradient는 invalid가 아니라 inconclusive 무신호 결과로 보존하고 perturbation/continuation을 미실행으로 기록한다. 그 외 finite/nonzero g에서 d=g/max(abs(g))로 고정하고 theta_plus=float32(1+.001*d),theta_minus=float32(1-.001*d)를float64로 보관한다. theta64==float64(theta32), finite/positive를 요구한다. tiny gradient의 sign만으로 모든 그룹을 같은 크기로 바꾸지 않는다.
5. theta 생성 직후 실제 selected weight 변화와 full native palette를 계산한다. 원래 CSR를 복사하여 미선택 bytes를 보존하고 selected에 f64(original32)*theta64를 한 번f32로 반올림해 적용하며 theta1의 bytes와 finite/nonzero/sign을 유지한다. 전체 data32.view(uint32)의 bit-pattern unique≤16384, node≤2**18/edge<2**31의 기존 native 표현 한도를 기록한다. selected만 세거나 ±0를 합치지 않는다. GroupedCurrent의 원래palette+동적효능 forward 가능성과 patched native CSR 표현 가능성은 다르다. palette가 부적격이면 예정한 ± 국소 판정까지는 기록하되 연속 신경/신체 실행을 생략한다.
6. 같은50창과 경계에서 plus/minus를 각각 한 번씩 실행한다. observed=mean(L_plus)-mean(L_minus),predicted=g·(theta_plus-theta_minus)를 실제 반올림된 theta로 계산한다. 모든 scalar의finite를 먼저 검사한다. |observed|>1e-9,predicted>1e-9, observed와 같은 부호, relative_error≤.05를 local_direction=supported로 사전 고정한다. predicted<0은 방향/계산 계약 위반invalid다. 모든 연산/receipt가 정상인데 의미 있는 불일치면 refuted, 너무 작으면 inconclusive, 손상된 입력/원본/clock/수치는 invalid다. local_descent=(neutral_loss-minus_loss≥1e-9)는 별도 결과로 기록한다.
7. local_direction=supported AND local_descent=true AND theta_minus native_eligibility=true인 경우에만 동일 theta_minus를 원래 control0에서 reset 없이300control,6000tick1.5초 연속 실행한다. 원래 saved drives와 bank/original을 사용하며 중간 저장 경계로 덮어쓰지 않는다. new training_loss가 E329 neutral보다≥1e-9 작고 startup q RMS≤.018/max≤.04,tendon RMS≤.009/max≤.025이며 모든 상태/수치가 유효해야 neural continuation supported다. old shape와 각 successor/coord/LR 항도 보고하되 old shape 개선을 새 별도 합격 기준으로 추가하지 않는다. 선행 조건이 실패하면 continuation=not_tested와 정확한 사유를 남긴다.
8. E331 인계 묶음에 theta_minus NPZ/hash, source CSR/positions/groups hash, selected actual32/full patched CSR bytes hash, source control0/decoder identity, startup 및 native eligibility를 고정한다. 후보 identity는 source identity를 출처로 기록하되 효능 변경에 따라 달라져야 한다. E330 끝 상태를 신체 시작으로 이식하지 않고 E331은 원래 전체control0으로 시작한다. 신경 방향·local descent·continuous개선/startup·native전달·최종보행 판정을 합치지 않는다.

학습 optimizer나 step-size search, 새로운 decoder fitting, 신체 실행, 자동 원본 retry는0이다. 연속 후보가 실패하더라도 좋은20ms 값을 실제 보행 개선으로 대체하지 않는다. 같은 방향을 임의 다른 방식으로 다시 실행하려면 새로운 근거·구분된 단계·사전 계약이 필요하다.

## 자원과 소유

예상 원본 상한은 forward18000ticks(50neutral×80 + 100perturbed×80 + continuous6000),recompute4000ticks,backward4000blockticks/200blocks,decoder replay300이다. 부가 참조 qualification은E329에서 끝내며 이 원본 runner에서 새 phase 탐색을 하지 않는다. original 감독process1회, 최대rollout forward151회/backward50회,body0,optimizer0,retry0이다. 정상 미실행 분기에서는 요청한 roster만 expected로 사용한다. 미완료 request/host return/device confirm와 부분 실패·저장 실패 비용은 그대로 기록한다.

제안 배정은 supervisor220초/internal200초/science180초/finalization20초,host6GiB/additionalGPU8GiB/output2GiB/free-disk4GiB다. science 절대 deadline=min(science_start+180,internal_start+200-20)으로 준비 뒤에도 finalization20초를 남긴다. supervisor의 추가20초는 종료/게시 보호이며science연장이 아니다. E324의 동일 work가161.1942초였고 추가 phase 목적 비용을 고려한다. 원본 직전 실제 load/resource를 확인한다. 새 유한 단계 배정300초로 기존 단계 상한9060→9360초를 사전에 확장하는 제안이며, 현재 누적9034.148870999521초와 과거 실패는 그대로 합산한다. 사용자 전체 예산을 새로 만들지 않는다. 아직 실제 예약이나 배정 변경은 없다. 남은 배정은 E331같은 구분된 실제 신체 계약에만 사용한다.

actual gpt-6-astra/max가 본 설계를 read-only 검토하고 root가 반영한 뒤 actual gpt-6.1-sol/xhigh가 새 runner `tools/verify_c_banc_m3_temporal_direction.py`와 의미 있는 생성 tests `tests/test_c_banc_temporal_direction.py`를 구현한다. Root FIRST→필요시 동일Sol R1→root SECOND→root 잔여수정 및 생성 qualification 후 source/freeze/review를 고정한다. Sol/root 생성CPU각120초,root generatedCUDA40초의 임시 자격 배정도 별도 원본0으로 기록한다. 이 E330 배정은 Astra 리뷰에서 확정/수정한다.

원본 결과·advance ledger는 supervisor 종료 후에만 root가 읽고 비용을 한 번 합산한다. 같은 supervisor가 소유한 PID만 한도 초과 시 종료한다. root는 저장 q/tendon/rawθ/gradient/CSR/receipt를 독립 재계산해 판정하고 F1–F8/candidate_adopted/생물학적 동등성은false로 유지한다. 목표는 계속 active다.

## Astra 리뷰 반영

현재 세션은 actual Astra6/max의 리뷰 후 local descent 선행조건,0/비유한 gradient 분기,surrogate VJP 의미,실제f32/full-CSR bit palette,동일θ 인계,E329 byte 승인연결,decoder/SSP 재생 구별,conditional roster 및finalization deadline을 반영했다. 원래 초안과 리뷰는 `parent-e330-pre-astra-design.md` 및 `parent-e330-astra-design-review.md`로 보존한다. 다음은 E329 자격 완료 후 actual GPT-6.1-Sol/xhigh 구현이다.
