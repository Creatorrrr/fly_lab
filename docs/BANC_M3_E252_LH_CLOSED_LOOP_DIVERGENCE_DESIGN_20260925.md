# E252 저장 폐루프 전달과 초기 오차 위치 진단 — 2026-09-25

E251은 유효하게 끝났지만 H251은 기각됐다. 기존25간선 복원 뒤 초기 q 최대 오차가 .05086635rad로 한도.04를 넘었고, 전진과 slip이 악화했다. 이번 분석은 그 저장 자료의 전달 차이와 초기 명령 오차 위치를 확인한다. 새 보행 후보나 단일 간선 인과 효과를 평가하지 않는다.

## 질문과 고정 판정

경쟁 설명은 LH에 한정된 초기 motor/명령 편향과 다른 다리까지 퍼진 전신 응답이다. 각각의 source/bridge/MN 발화, bank, 근육 activation, 관절 명령과 실제 관절을 별도로 본다. 신경 rate 단위는 모델 단위이며 측정 Hz로 부르지 않는다. 같은 endpoint의 여러 차이는 20 neural RK4 substeps의 선후관계를 증명하지 않는다.

H252_local_startup은 아래 세 조건의 conjunction으로만 기술적 국소화 지지 여부를 판정한다. 이는 기전의 인과 증명이나 개선 판정이 아니다.

1. startup endpoint1..99에서 baseline |q-target|≤.04이면서 candidate |q-target|>.04인 새 위반이 하나 이상 있다.
2. 그런 새 위반의 모든 관절이 LH에 속한다.
3. 같은99행에서 양의 오차제곱 증가 max((C-T)^2-(B-T)^2,0)의 LH 합계가 전체의50% 이상이다. 분모0이면 unavailable/미지지다.

음의 기여도 숨기지 않도록 각 다리의 signed SSE 변화와 양의/음의 기여를 모두 보고한다. q RMS/최대, LTM RMS/최대는 기존 한도로 재계산하되 E251의 기각을 바꾸지 않는다. 다른 다리의 작은 변화도 모두 보고하고, 50% 기준을 생물학적 기준으로 부르지 않는다.

지지되면 최초 위반·최대 오차 관절에서 LH의 어느 근육/모터 출력 변화가 명령 오차를 늘렸는지에 근거해 다음 최소 개입을 별도 설계한다. 미지지면 전 다리별 signed 기여와 최초 변화에 따라 분산된 경로를 우선 조사한다. 어느 경우에도 gain sweep/fit/전체 다리 복원/새 신체 실험은 이번 분석의 자동 후속이 아니다. 정적 연관만으로 병목의 단일 원인을 확정하지 않는다.

## 고정 입력과 계산

원자료는 E251 baseline/candidate의 완전한301 endpoint,300 encode,15000 native 표본이다. startup1..99, steady100..299, full0..300 세 구간을 고정하며 창을 사후 선택하지 않는다. native는 모든15000행과 [.5,1.5) 구간을 구분한다. trace/원래 E103 target99/E150 calibration·muscle provenance/E250 exact25간선 및 E251 감사·manifest·초기 mask를 동결한다. E251 source23/bridge4/reachableMN55는 graph row ID이고 전체 MN391는 calibration.motor_ids와 별도 대응한다. 총 LH MN56 중 reachable55만 저장 activity에 있다는 차이를 유지한다.

E251 sensory metadata의 실제 joint_columns(6×7)와 원래 LEGS/DOFS/joint_suffixes 순서를 사용해42열의 물리 관절 이름을 매긴다. 특히 coxa pitch/roll과 femur pitch/roll은 body 열 순서와 다르므로 slice 안의 suffix 순서를 가정하지 않는다. muscle-provenance의96 groups·columns와 calibration/parameters의391 membership·pool support를 대조한다. 실제 calibration 파일에도 membership이 있으므로 입력에서 존재 여부를 확인하고 parameters와 exact 비교한다.

encode의 control_tick_before=t는 endpoint t+1에 적용된다. 반환 source drive는 old_filtered에서 계산하며 새 filter는 다음 control에서 사용된다. 직접 배열 비교는 각자의 명시된 clock을 유지한다. stage별 첫 exact 차이와 아래 절대 tolerance 초과 첫 차이, RMS/max를 함께 저장한다. floating tolerance는 기록상의 가시 크기일 뿐 통과 생리 기준이 아니다: feature/filter/source drive1e−6, neural rate/adaptation/bank1e−5, activation1e−6, command/actual q1e−7rad, body position1e−7(저장 좌표 단위), native load/slip/upright1e−7. 첫 차이가 없으면 null이고 row0을0초 그대로 기록한다. 같은5ms endpoint 안의 stage 순서나 두 run이 달라진 뒤의 feedback을 단일 간선 causal latency로 해석하지 않는다.

E250 원래/복원된25 edge의 post bridge4별 직접 기여를 source endpoint rates와 실제 output_mask로 float64 합산한다. baseline I_B=W_B r_B, candidate I_C=W_C r_C, intervention=(W_C-W_B)r_B, feedback=W_C(r_C-r_B)를 모든301행에서 구한다. I_C-I_B=intervention+feedback을 검증한다. 이는 선택25간선의 endpoint 산술이며 전체 재귀 전류나 CUDA packed RK4 재실행이 아니다. 새 bridge/source 단순 포화 판정을 추가하지 않는다.

근육→명령은 원래 rest+.02*moment@activation을 각 arm row1..300에서 검산한다. row0은 아직 decode 전이므로 별도 보고하며 강제 일치를 요구하지 않는다. signed 근육별 delta command=.02*moment[j,m]*(A_C-A_B)[m]를 전체301행에 저장하고42관절 합산과 실제 C-B를 row1..300에서 대조한다. max joint contributor 순위만으로 근육의 생리학적 원인을 주장하지 않는다. 초기 target 오차 분해는 (C-T)^2-(B-T)^2=2*(B-T)*(C-B)+(C-B)^2이며 개별 관절·다리·전체 수준 모두 보존한다.

모든 availability/shape/dtype/ID uniqueness/clock/finite 값과 입력 hash/버전을 검증한다. source drive의 endpoint t+1 대응을 raw drive와 exact 대조한다. 누락·불완전·중복·clock 불일치·수치 오류는 invalid다. 부정적인 국소화 판정은 valid일 수 있다. H251 실패/F1–F8 미완료/F6 미노출은 유지한다.

## 구현, 검토, 결과와 자원

부모가 설계하고 실제 gpt-6-sol/medium가 `tools/analyze_c_banc_m3_lh_closed_loop_divergence.py`와 `tests/test_c_banc_lh_closed_loop_divergence.py`만 구현한다. 부모 FIRST→필요시 같은 Sol 첫수정→부모 SECOND→남은 수정은 부모 순서다. 생성 fixture는 임의 graph ID 순열·동일 시작/첫 row 차이·one-control 시차·SSE signed cancellation/0분모·25간선 decomposition·근육 명령 합산·late publication 실패를 검증한다. 실제 saved 자료로 실행하는 원본은 source freeze/부모 승인 후1회다.

출력 `verification/banc-natural-gait-m3-20260919/e252-lh-closed-loop-divergence/`에 정렬된 raw differences/25간선 기여/근육 signed 기여/초기 error decomposition NPZ, mapping·stage/per-leg/위반 표·판정 JSON, 입력/출력 manifest, progress/resource/error/terminal result를 보존한다. 실패 시 성공 결과를 남기지 않고 원본1/retry0로 종료한다. 부모는 독립적으로 raw 지표·신경/근육 ID·clock·분해·판정을 검산한다.

신규 모델 생성/신경 tick/physics/CUDA/fit/solver는 모두0이다. CPU1/GPU0, internal30초(마감5 포함)/external40/supervisor45, host2GiB/output64MiB/free disk512MiB, 원본당300초 한도를 유지한다. 기존 누적6853.242210999630/6933초와 잔여79.757789000370초에서45초만 예약한다. 사전 배정 증액0이며 준비·생성검사·부모감사 비용은 별도, 미측정 개발/자문 비용은null이다. E251 결과/기존 실패/E150 best partial/M2/default/UI를 보존한다. Goal API paused 상태를 바꾸는 기능은 없으며 사용자의 계속 요청에 따른 승인 범위 작업을 이어간다.
