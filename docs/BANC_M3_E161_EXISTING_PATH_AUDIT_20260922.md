# E161 기존 RM 감각·운동 경로 감사

실제 gpt-5.6-sol/medium인 /root/e158_sol_rm_mapping_audit가 저장 자료와 소스만 읽었다. 파일 변경·모델·StateStore·신경·물리·fit 실행은 0이다. 부모는 rate_body.encode, typed_receptor_feature, MotorTimescaleNetwork와 MonotoneRecruitmentAdapter의 적용 경로를 직접 확인했다. 세부 edge 수치는 감사 에이전트가 원래 bundle CSR과 E150 actual-global-csr, recruitment-metadata, adapter-provenance 및 E07b topology/muscle provenance를 교차 조회한 결과다. 이 문서는 보행 개선 또는 생물학적 기전 검증 결과가 아니다.

## 실제 입력과 기존 직접 연결

E150 bindings v2의 RM sensory cohort는 claw13, hook22, club52, load2, touch484개다. 실제 encode는 측정 knee angle/velocity, support_load_bw와 non_support_load_bw를 사용한다. 현재 호출은 직전 filtered 값을 drive에 넣고 현재 관측으로 20ms filter를 갱신하므로 한 제어 단계의 지연도 존재한다(flylab/c/rate_body.py:489–556).

load SNpp53 두 세포는 graph30981/root720575941471495543과 graph107685/root720575941573309691이다. 검사한 55 recorded upstream source로 향하는 직접 해부 연결은 없다. RM 운동뉴런의 직접 표적은 group74 tibia flexor의 graph12036/32830/108052/153939와 group75 accessory flexor의 graph117483이다. 원래 .275/.55 가중치는 현재 3e−5/6e−5이며 비율은 .0001090909다. group68–72,76,77,78,79로 향하는 직접 load edge는 없다. 이는 검사한 직접 연결의 결과이며 미관측 중간뉴런을 통과하는 경로 부재를 뜻하지 않는다.

Hook은 recorded IN16B016(graph122032/root720575941595919217)에 직접 두 edge, 현재 합+.15/최대+.09를 가진다. 다만 그 source에서 대상 RM 운동뉴런으로 가는 현재 가중치는 그룹별 최대 절대값 약 .00006–.00393이다. claw/hook의 RM 운동뉴런 직접 가중치도 대부분 3e−5–1e−3이다. 이것은 하중 입력과 별도의 관절 상태 입력 경로다.

typed_receptor_feature의 polarity는 MANC 연결과 cell-type 이름을 전이한 negative-feedback 가정이다. BANC 실측 tuning이 아니다. Club도 실제 진동 주파수 모델 대신 knee-velocity high-pass를 받는다(flylab/c/neuromuscular.py:118–123,153–165).

## 현재 RM 출력의 소유 위치

E150 adapter-provenance의 moment는 group68/69를 femur-pitch 음,70/71/72를 양,74/75를 tibia-pitch 음,76을 양,77을 tarsus-pitch 음,79를 양으로 보낸다. group78 LTM은 일반 관절 moment가0이며 별도 rm_tarsus tendon 경로를 쓴다. moment 부호나 근육의 extensor/levator 이름만으로 현재 자세에서 발이 올라간다고 판단하지 않는다. 이것을 E161의 world-Z Jacobian 분해로 확인한다.

주요 현재 source는 RM E1 IN17A001 graph97727/root720575941560438643, RM E2 INXXX466 graph43618/root720575941487873488, relay IN03A010 graph54014/root720575941504444354, 억제성 IN13A001 graph144453/root720575941645048993, IN09A002 graph131125/root720575941613448668이다. recruitment-metadata의 source_to_motor는 원래 접촉 수 자체가 아니라 현재 교정된 유효 가중치다.

* group76: E1→SETi35767 +1.3668, relay→FETi11037/SETi35767 +.8566/+.5499, IN13A001→SETi35767 −.9786.
* group79: E1→38466 +.5538, INXXX464 graph100866→12864 +.3292, IN13A001→12864/38466 −.6/−.2995.
* group78: E1→84551 +8.6064, IN09A002→84551 −4.68. tendon 경로이며 adhesion 명령이 아니다.
* group74/75: IN13A001의 강한 음 가중치(그룹별 최대 절대14.6361/11.5938)와 E1/E2 양 가중치가 공존한다.

group71/78의 actual_motor_gain32≈1e−10은 **adaptation 계수**다. MotorTimescaleNetwork는 (cell_gain×rate−adaptation)/cell_tau를 계산하므로 이 수치를 근육 출력 권한이0이라는 근거로 쓰지 않는다(tools/banc_m3_motor_timescale.py:83–97,132–136). 감사 도중의 반대 추론은 철회됐다.

Adhesion은 현재 adapter가 항상 false6개를 반환하고 monotone wrapper가 그대로 전달한다. E150 저장 trace도 모두 false다. 따라서 접지 해제 실패를 켜진 adhesion 탓으로 설명하거나 adhesion을 끄는 후보를 만들지 않는다.

## 기존 실패와 다음 판단

E05의 전역 afferent 변환은 전진10.7812→9.5418mm, net/path .3434→.3281로 악화했다. E95 relay는 실제 .4428mm/적격주기0, E96/E97 source-to-MN fit은 전체 명령 기준 실패, E102는 .085초 upright 종료였다. E109는 adaptation τ 배율1.0을 유지하고 shape 기준에 실패했다. E146/E147 filter/gain과 E158–E160 singleton40ms도 시작 기준과 명령 품질을 함께 만족하지 못했다. 따라서 무차별적인 감각 증폭이나 τ 탐색을 재개하지 않는다.

E1/IN13A001에서 group76/79로 가는 기존 opponent edge와 hook→122032→MN 경로는 검사할 수 있는 실제 소유 위치다. 하지만 전자의 실제 수직 방향과 이벤트 타이밍은 아직 미확인이고, 후자는 하중이 아니라 knee-velocity proxy이며 후단 가중치가 작다. 직접 load 복원은 알려진 flexor 표적에 도달하므로 이를 lift command라고 부를 근거가 없다. 부모가 E161 저장자료 결과와 함께 다음 개입을 정하며, 현재 후보 파라미터 변경·채택은 없다. F1–F8 미완료, M2/F6/UI 보존.

## 후속 감사: 약화의 정확한 소유 단계와 단위

추가 Sol/medium 읽기 전용 감사와 부모 소스 확인으로 약화의 소유 단계를 찾았다. rate_weights는 GraphStore의 원래 .275/contact를 그대로 쓰지 않고 sign×contact_count×.03의 rate-network 단위로 재구성한다(flylab/c/research_annotations.py:72–80). E43의 fit_c_banc_m3_rhythmic_recruitment.py:219–238은 391개 운동뉴런 수신 행에서 선택한 source와 navigation port20개를 제외한 기존 비영 연결에 .001을 곱한다. 즉 현재 rate 모델 내부에서의 억제는 정확히1/1000이며, 원래 GraphStore 대비 .00010909에는 단위 변환 .03/.275도 포함된다. 이 비율 전체를 같은 단위의 기능적 억제 크기로 해석하지 않는다.

이는 신경망 전체 또는 감각뉴런 전체를 줄인 것이 아니다. **MN으로 들어가는 비선택 직접 입력**에 적용되며 감각→비MN의 간접 경로는 이 mask에서 제외된다. 해당 하중 입력5개는 E150 assembled-calibration의 아래 기존 CSR 항목이다.

| CSR edge | presynaptic→postsynaptic | rate source→current |
|---|---|---|
|844054|107685→12036|.03→.00003|
|2400285|30981→32830|.06→.00006|
|7908675|107685→108052|.03→.00003|
|8578714|30981→117483|.06→.00006|
|11240394|30981→153939|.03→.00003|

이5개는 E96/E101 이후 개별 fit 대상1572개에 포함되지 않았다. E123 adaptation gain과 E138 timescale은 이 synaptic 값을 변경하지 않았다. patch_weights는 기존 CSR topology와 부호 및 비선택 값을 보존한다(tools/banc_m3_synaptic_efficacy.py:12–29). 현재 생성 경로는 rate_weights→navigation inhibitory 보정→동결 calibration patch다(tools/verify_c_banc_m3_native_timescale_transfer.py:1086–1093).

weight는 presynaptic rate를 current로 전달하는 계수다. cap200이면 .00003/.00006 edge 자체의 최대 직접 항은 .006/.012 current unit이지만, 전체 recurrent current·threshold·간접 경로를 검사하지 않고 기능적 무반응을 선언할 수 없다. E05는 외부 afferent drive 변환 변경이고, E18은 motor incoming 전체107,221개 변경으로 sensory-only 복원과 다르다. 집중한 과거 기록 조회에서 이5개 측정된 기존 sensory→MN edge만 복원한 시도는 찾지 못했다. 따라서 이 위치는 이전 실패와 구별되는 최소 개입면이지만, 수직 운동 분석과 인과 시험 전에는 채택 또는 lift 효과를 주장하지 않는다.

부모도 저장 calibration과 실제 CSR을 직접 대조해5개 edge의 주소·양끝·source/current 값과 개별 fit 비선택을 확인했다(parent-e161-feedback-provenance-v2.json). E43처럼 float64 factor와 곱한 뒤 float32로 저장하면 다섯 값 모두 비트 단위로 맞는다. 첫 부모 checker의 float32 scalar 곱은 NumPy의 scalar promotion 때문에1ULP 차이를 내어 엄격 비교에 실패했으며 원본 checker를 보존하고 실제 생성 산술을 적용한 v2에서 확인했다. 후보 생성이나 신경/물리 실행은 없다.
