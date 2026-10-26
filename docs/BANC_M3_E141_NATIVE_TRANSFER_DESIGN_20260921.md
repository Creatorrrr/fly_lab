# E141 근육 전달 개선의 실제 전체 BANC·신체 진단

2026-09-21 · M3/H141 · 부모 설계 → 실제 Sol/xhigh 구현·첫 수정 → 부모 재리뷰·잔여 수정

## 방향 결정과 보존할 미달

직전 E140은 progress: RM이 원래 초기 자세와1%손실 개선을 함께 충족했지만 LH/RH는 미달했다. H140 refuted와 여섯 다리 조건부 패키지 미채택을 변경하지 않는다. 저장 원자료의 독립 재계산에서 여섯 다리 모두 원래 startup selector를 통과한다. LH/RH의 손실 개선은 각각0.9948242081%,0.9998616098%로 원래1%보다 부족하다. 이를 반올림하여 통과시키지 않는다.

E128 이후 근육 단위 적합만 반복했고, 가장 최근 실제 전체 BANC·신체 E121은 F3만 통과했으며 전진 적격 완전 주기가 없었다. 미세한 조건부 손실 차이를 더 줄이는 접근과, 이미 달라진 근육 출력의 실제 폐루프 전달을 먼저 확인하는 접근을 비교한다. 후자가 지속 전진·교대·운동폭이라는 원래 목표에 더 직접적인 정보를 준다. E141은 별도의 **개발용 전달 진단**으로 같은 파라미터를 전체 신경망·감각·실제 FlyBody에서 실행한다. 조건부 적합 성공이나 제품 후보 승격을 선언하는 실험이 아니다.

H141: 고정 여섯 다리 전달 모수의 조합이 원래 zero-state 전체 BANC·신체에서10초 건강성을 유지하고 F3/F4/F5 및 원래 초기99 제약을 함께 충족한다. 경쟁 설명은 기록된 입력의 조건부 출력 개선이 전체망 피드백·float32 계산·기계적 상호작용에서 유지되지 않거나, source 협응 자체의 결함이 남는다는 것이다. 동일 후보의 실제 신경·근육·접촉·발 운동을 기록해 다음 수정 위치를 결정한다.

## 고정 파라미터와 개입

| 다리 | 고정 개발 원자료 | 원래 조건부 판정 |
|---|---|---|
| LF | E137 global8 | 통과 보존 |
| LM | E139 global7 | 통과 보존 |
| LH | E140 global18 | startup 통과,1%손실 guard 미달 보존 |
| RF | E139 global36 | 통과 보존 |
| RM | E140 global29 | 통과 보존 |
| RH | E140 global47 | startup 통과,1%손실 guard 미달 보존 |

LH/RH는 해당 E140 전체 유효 응답 중 복구 잔차가 가장 낮은 고정 응답이다. 새로운 결과를 보고 선택을 바꾸지 않는다. E137/E139/E140의 누적 선택 노출과 실패는 보존한다. 기존 E121의872 target-port 연결/64,38 source 막 시정수,source core gain/연결,96근육 adapter,391 MN/158706전체망,body/world/초기 자세/감각·고정 전진 입력은 유지한다.

새 모수는 기존1572 anatomical fit edge의 delta,96근육별 gain과ell뿐이다. 실제 float32 연결은 `float32(E121_base_fit_weight * exp(delta))`, 근육 gain은 기존 E123 규칙으로 float32 변환하여 membership에 따라391MN에 매핑한다. 적응 시간상수는 E138의 검증된 `.3*exp(ell)` float64→tau32/inverse32 규칙을 그대로 사용한다. edge의 존재·전후 노드·부호, 원래 per-edge 범위와 gain[0,8],ell[-log3,log10]를 재확인한다. calibration의 source_values는 유지하고 values와 individual_edge_efficacies만 일관되게 갱신한다. 모든 비대상 CSR 요소와 막 매개변수는 exact다.

## 실행 순서와 직접 증거

1. 부모가 원자료 SHA/해부학적 매핑/합성 float32 값/원래 실패 판정을 독립 계산하고 동결한다. 과거 E140의 accepted package를 생성하거나 수정하지 않고, E141 경로에 진단 전용 모수를 저장한다.
2. 같은 E1283문맥의 전체391 MN 조건부 CUDA forward를 한 번 수행하여, 같은 모수의 여섯 저장 응답을 합친 rate/adaptation/activation/command/tendon과atol1e-8,rtol0으로 비교한다. derivative bank는 실행하지 않는다. 이 단계는 조립 일관성 검사이며 원래 loss guard 실패를 없애지 않는다. 먼저 raw와 실제 요청·완료 비용을 저장한다.
3. 기존 E121 zero-state에서 Efficacy/Recruitment/Intrinsic→MotorAdaptation→MotorTimescale 연구 wrapper를 적용하고, 원래 body/world·0 neural/physics tick·감각·navigation 설정과 같은 근육 adapter를 확인한다. source·CSR·근육 모수 provenance를 저장한다. 실제 전체158706 BANC/CUDA를20신경tick/5ms control,50물리substep/control로10초 실행한다. 외부CPG/teacher/위상/궤적 재생/root 보정은 없다.
4. native100us 접촉/건강성과5ms control trace,source/MN rate·adaptation·수신 전류·근육 활성/목표·실제 관절을 저장한다.1초 디스크 checkpoint의 clone을0.1초 실행하고 main1.1초 전체 snapshot과exact로 비교한다. 복원은 MotorTimescaleTrial 전체 계층을 사용한다. 실제 비용과 부분 실패를 보존하고 모든 세션/observer를 닫는다.
5. 저장 원자료를 먼저 보존한 뒤 기존 교정된 `analyze_c_banc_biological_gait.py`, E113 body_quality/native 접촉/slip, E121 original_startup로 평가한다. 부모는 raw/시계/모수/전체 snapshot/수치와 비교 기준을 재검토한다. E121의 동일 [.5,10) 결과와 비교하되 변화의 각 구성요소별 인과 효과로 분리하지 않는다.

조건부 조립이나 native mapping/저장·시계·원자료 무결성이 실패하면 dependent body 실행을 하지 않거나 중단하며 validity=invalid, H141=not_tested다.10초가 완주하고 무결성을 충족하지만 F3/F4/F5/초기99 중 하나라도 미달하면 H141 refuted로 기록한다. 건강성 중단은 관측된 신체 실패와 나머지 미완성 항목을 구분한다. 늦은 분석·저장 오류는 완료로 둔갑시키지 않는다.

## 기준과 다음 행동

F3는 원래 순전진≥10mm,순변위/경로≥.5,upright≥.9,비지원 신체 충돌0이다. F4는 전진 적격 구간 내 여섯 다리 각각≥10완전 주기,LR3쌍 반주기 거리≤.15/R≥.5다. F5는 같은 기하학적 무릎95–5% 운동폭 LF25/RF30/LH40/RH35도 이상과 모든 다리 주기CV≤.3이다. F3/F4/F5를 같은 실행에서 요구한다. 원래 초기99 qRMS.018/qmax.04/LTM RMS.009/max.025도 그대로 기술·판정한다. 독립 표본이나 생물학적 생리 검증으로 해석하지 않는다.

성공해도 E140 조건부 미달은 남으며 M3/F1–F8 완료나 제품 adoption은false다. 실제 보행 결과가 유망하면 동일한 고정 후보의 F1/F2 통합·운동 차단,미사용 F6 두 자세,기존 F7 목표6개·정지/재시작·M2 slip 비교,F8영상/증거/복원을 계속한다. 실패하면 기록된 실제 source→MN→근육→발/접촉의 어긋남으로 다음 개입을 정한다. F6는 이번에0노출,활성 UI와M2는 변경하지 않는다.

## 소유권과 사전 자원

새 runner `tools/verify_c_banc_m3_native_timescale_transfer.py`와 `tests/test_c_banc_native_timescale_transfer.py`를 실제 Sol/xhigh가 구현한다. 새 CUDA 방정식을 만들지 않지만 전체망 연결·적응 계층·신체 소유권·디스크 미래가 함께 연결되므로 xhigh를 사용한다. 부모가 설계·원본 실행·리뷰를 담당하며 첫 수정은 같은 Sol,두 번째 리뷰 뒤 잔여 수정은 부모가 맡는다. 기존 E121/E123/E128/E134/E138/E140 및 제품 소스는 보존한다.

기존2101초 배정과 실제 보수적 누적1980.3578631000128초를 보존한다. 새 전체 신체 진단에는1981+290=2271초의 사전 누적 상한을 배정한다. 원본1프로세스/재시도0,준비45초,신체+분석180초,내부260초/종료reserve12초,외부275초,감독·최종manifest290초다. 기존1프로세스300벽초 상한 안이다. CPU1/GPU1,raw4GiB/시작여유12GiB다.

원본 조건부 forward1회는3×40000tick=120000 context tick/30조건부초/6000 decoder endpoint/derivative bank0이다. native main2000+clone20control은40400신경tick/10.1신경초/101000물리step/10.1물리초다. 신경 모델의 CUDA 준비 작업은 과학적 tick과 별도로 실제 생성 횟수·가능한 정확한 준비량을 기록하고, 모르면0으로 두지 않는다. 생성 CPU 구현·검산과 부모 저장 자료 검산 비용은 별도 보존한다. 이번 실험에서 fit/LP/추가 target/추가 posture/제품UI 실행은0이다.
