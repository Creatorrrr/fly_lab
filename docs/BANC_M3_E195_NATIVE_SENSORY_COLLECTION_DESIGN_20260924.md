# E195: 지난 물리 구간의 속도 감각 수집

2026-09-24 원본 실행 전 설계. E194는 같은 저장 궤적에서 여섯 무릎 모두 경계 표집의 정보 손실을 확인했다. E193의 hook/club 정지와 결합하면 감각 전달을 수정할 근거가 생겼지만, 자연 보행 실패의 단일 원인으로 확정하지 않는다. 전체 F1–F8과 M2/UI, 미사용 F6를 유지한다. 현재 원본 누적4768.139314499720/4994초, 잔여225.860685500280초다.

## 질문과 단일 변경

직전5ms의 실제100µs 운동으로 hook/club 비선형 특징을 계산해 전달하면 감각 신경 반응 및 지속 전진이 회복되는가? 경쟁 설명은 감각 표집을 복구해도 현재 수용기 스케일·신경 동역학·운동 전달의 다른 한계가 남는다는 것이다. 이득이나 가중치를 함께 바꾸지 않는다.

물리 단계가 완료될 때 별도 shadow MjData에 actor의 완전한 integration state를 복사하고 mj_forward 후 여섯 무릎의 기하 속도를 읽는다. actor에 추가 forward/step을 하지 않는다. 신체 생성 settling 뒤에만 설치하며 1control당50표본이다. 원래 body step과 모터 전달은 그대로다. E156의 중간 운동 명령이나 E22의 연속 목표각, E65의 지연 제거를 재시험하지 않는다.

각 표본에서 원래 `typed_receptor_feature`와 같은 hook SNpp41/39/기타 및 club 식을 계산한다. 30ms 속도 저역 통과 상태는 해당 구간 동안 원래 경계 값으로 유지한다. 비선형 특징을 먼저 계산한 뒤 실제 표본 간격으로 가중 평균한다. 평균 속도·각도 차분·미래 상태로 대체하지 않는다. 구간(t[k-1],t[k]]의 특징은 encode(t[k])의 request에 들어간다. 원래 drive는 이전 filtered를 먼저 반환하므로 새 request의 효과는 다음 control drive부터 나타난다. 이 인과 지연을 명시적으로 유지한다.

tick0에는 지난 구간이 없으므로 원래 encode와 exact다. 이후의 후보 encode는 원래 RateBodyAdapter.encode를 한 번 호출하고, **hook/club 포트만** 저장한 old_filtered와 구간 특징으로 같은 `old + (1-exp(-.005/.02))*(18*feature-old)`를 적용해 교체한다. 해당 last_features도 일치시킨다. 이미 반환된 drive, 나머지 포트, 원래30ms velocity_lowpass 갱신은 손대지 않는다. 이는 특징을 필터 갱신 전에 주입하는 것과 수학적으로 같고 기존 인코더 소스의 복제를 피한다. sensory_cut에서도 원래대로 drive만 차단하고 필터는 갱신한다.

## 구조와 복원

새 연구용 `NativeVelocityAdapter`는 기존 MonotoneRecruitmentAdapter를 `_motor`로 보관하고 decode/운동 속성은 그대로 위임한다. 기존 e150 관측의 `.original.activation`은 원래 RateBodyAdapter 활성이라는 의미를 유지해야 한다. encode만 후보가 담당한다. 별도 identity/schema에 원래 adapter identity·알고리즘·시계 계약을 포함한다. production RateBodyAdapter, 신체, 기존 자료는 수정하지 않는다.

새 Trial은 기존 E150 MonotoneRecruitmentTrial의 전체 원형을 검증한 뒤 후보 adapter와 body 관측을 설치한다. snapshot은 원래 전체 신경/신체/운동 상태와 함께 누적 feature 합·기간·표본 수·구간 시각·encode 수·마지막 특징을 저장한다. 완료된 마지막 구간은 다음 encode 전까지 pending이므로 버리지 않는다. 복원은 해당 스키마/identity/shape/범위/50표본과clock을 검증하고 원래 trial로 풀어 복원한 뒤 후보 상태를 되붙인다. 오류 시 연결한 method slot을 복구하고 소유 body만 닫는다. 원본0.1초 전체 미래 복원은 이번 단일3초 대조의 완료 조건으로 추가하지 않고 F2 후속 검증으로 남긴다. 이번에는 실제 start/final checkpoint disk roundtrip 및 합성 미래 일치 시험을 수행한다.

## 비교와 기록

기존 E193/E185의0–3초 baseline을 재사용하고 같은 E150 candidate-neutral에서 후보 하나만 시작한다. whole graph158706/CSR/신경 identity, 운동3bank 파라미터, body hash `1b2a89084311ebd2ccbccbe2c57f2ec29fa937e115bb6cd4c7f6cd16ba7c2e6f`, gain18/load1/front hair plate false를 고정한다. 초기 underlying 전체 checkpoint는 exact여야 한다. E181 verified_inputs와 기존100초 arm·startup/health/clock 가드를 그대로 사용하되 E195의 별도 gate와 명시적 candidate restore factory를 추가한다. legacy E181 출력의 controller 명칭만으로 후보를 baseline이라고 주장하지 않고 E195 provenance에 변경을 기록한다.

E193 Collector의 신경/인코더 raw 관측을 재사용하되 후보의 encode 호출을 관측하고 runtime expected adapter identity를 후보의 결정적 identity와 대조한다. 원래 특징 검산은 hook/club에서 의도적으로 달라지므로 전용 검사로 대체한다. native30000행의 시각·control/substep·각도/속도·LP·속도 특징, encoder600행, 신경601행, 기존 native 접촉·391MN/96근육/42관절·trace·start/final checkpoint를 보존한다. 입력/신경/근육/움직임의 변화가 어디서 끊기는지 분리해 보고한다. 샘플/예외/정리 실패를 버리지 않는다.

유효성: 전제 hash/전체 시작 상태, 실제 CUDA packed, 신경12000/물리30000/control600, 매 구간50개 관측, 완료 구간만 소비, 감각 수식·필터·drive와 비대상 포트의 독립 재구성, actor 비간섭, 기존 startup/health 가드, 부분 기록과 자원 제한을 만족해야 한다. 지연 계약상 encode tick0/1의 반환 drive는 기존과 같아야 하므로 처음 두 control의 신경·운동 출력 및 trace 경계0/1/2, 처음100개 native 접촉도 exact로 대조한다. 변경된 drive는 tick2부터 허용한다. 관측 입력만 바뀌어도 baseline 궤적의 exact 유지가 필수인 E193과는 다른 인과 실험이다.

전달 가설은 평가0.5–3초에서 hook/club request·filtered drive·rates의 차이와 exact-zero/포화 수를 보고한다. 신경 변화는 모델 단위 RMS 차이>1e-6 여부를 함께 기록하며 임상/생물학적 효과 크기로 해석하지 않는다. 보행 후속 진행은 원래 strict startup/health 통과에 더해 longest forward bout≥.3초, baseline 대비 forward≥90%, net/path 감소≤.05, native 하중 가중 slip≤baseline, 여섯 무릎95–5폭≥baseline90%를 **모두** 만족할 때만 고려한다. 실패한 항목은 다른 개선으로 상쇄하지 않는다. 짧은 대조는 F3/F4/F5 또는 전체 채택을 통과시키지 않는다. 효과가 없으면 정보 손실 진단과 후보 기각을 함께 보존한다.

## 작업 순서와 자원

부모 설계→실제 gpt-6-sol/high 최초 구현→부모 FIRST→같은 Sol 첫 수정→부모 SECOND/잔여 수정이다. Sol 소유는 새 `tools/banc_m3_native_sensory.py`, `tools/verify_c_banc_m3_native_sensory.py`, 대응 테스트2개다. 부모가 metadata/freeze/source approval, 감독과 독립 검산을 담당한다. 기존 파일과 UI는 수정하지 않는다. 과학적 원본 실행은 승인된 소스 이후다.

원본1process/1arm/전체 BANC1·body1/자동retry0/CPU·BLAS1/GPU1/내부140초/외부145초+cleanup5초=150초를 같은 잔여에서 예약한다. E181 arm100초 제한은 불변이며 resource guard면 불완전이다. 시작 settling2000을 따로 계수하고 full scientific3초만 실행한다. 출력2GiB·arm1GiB/free20GiB, 기존 queue/GPU readiness와 모니터를 적용한다. 원본감독 wall을 현재 누적에 더하며 준비·합성·부모 검산·미집계 비용은 별도로 보존한다. F6 노출0, 후보미채택, 자연 보행/F1–F8 active·미완료다.
