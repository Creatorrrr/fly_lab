# E187 MN144018의 저장 입력 전류와 사건 시계

2026-09-24 · 사전 설계 v1 · M3 finite-goal의 저장 자료 진단. 이전 E186은 RM coxa-roll의 명령 출처를 group67/단일 MN144018로 좁혔으므로 progress였다. E160의 같은 근육 40ms 기각, E95의 다중 변경과 물리 가설 실패, E185 원본 감독 rejection과 별도 저장 원자료 승인, F1–F8·F6 미노출·M2/UI 보존을 유지한다.

## 질문과 다음 결정

MN144018(root720575941644474312)의 현재 incoming 연결 중 어떤 입력이 제동 사건의 회수 명령과 함께 증가하는가? E1=97727, E2=43618, relay=54014, IN13A001=144453도 고정 비교 대상으로 둔다. E150의 기존 446-neuron 기록에 다섯 노드 모두 있으며 새로운 full-BANC 실행 없이 비교할 수 있다. 단, E185 3초 실행과 동일한 경로임을 먼저 검증한다.

예측 A: 현재 연결에서 특정 양의 입력이 negative-force pre보다 lead/post에서 크고 같은 post의 positive-force 조건보다 커서 제한적인 기존 edge 효능 대조를 설계할 근거가 된다. 대안 B: 알려진 relay가 작은 항이거나 입력이 추진에도 같거나 더 크게 필요하며, 관측되지 않은 source의 잔여 전류 또는 adaptation이 설명을 남긴다. 수치 전부와 개별 사건 반례를 보고한다. A를 인과 증명으로 판정하지 않으며 임의 lag/사건/threshold 탐색, 자동 edge 선정·감쇠, 새 모델/학습/물리/신경 실행은 하지 않는다. 결과에 따라 다음 대조 1개를 별도 설계하며 불충분하면 필요한 관측만 명시한다.

## 입력과 재사용 검증

부모 metadata가 모든 입력/설계 SHA, NPZ/NPY header, 상태 디렉터리, 보존 파일, 재사용 소스 SHA를 고정한다. E150-native-monotone-body의 actual-global-csr, assembled-calibration/parameters, recruitment-signals/metadata, start-observation, native-recruitment-state, native-contact, trace, clocks/journal, network-layer-identities/csr-provenance를 사용한다. E185 baseline의 start/final 및 601/30000 원자료, 승인된 E186 aligned/analysis/terminal/final-review를 연결한다. 원래 graph bundle의 nodes/counts/weights/indices/indptr/identity/manifest를 사용한다. E178의 순수 NumPy CUDA arithmetic helper를 변경 없이 재사용한다.

1. E150와 E185의 graph/model/body/adapter identity, 초기 network 전체 상태·patch·cuts·parameters/efficacy/초기 위치/seed/world 및 adapter 상태를 파일명과 무관하게 배열 의미로 비교한다. wall-clock과 실행 기록 provenance는 동일성 요구 대상이 아니다. 의미상 차이는 숨기지 말고 fail closed 한다. E185의 theta=0 wrapper 메타데이터는 명시적으로 검사하며 결과를 기록한다.
2. E150 trace samples[:601]와 E185 samples 전체의 물리/신경/제어 값 및 native-recruitment의 모든 601행을 비교한다. native-contact의 모든 30000행도 비교한다. 시간은 i*.005/k*.0001, 허용 2e-8; 비교 실행 사이의 실측 배열은 exact equality를 요구한다. NaN 가용성은 동일 마스크일 때만 equal_nan으로 처리한다. 나머지601행 신경 metadata를 임의로 제외하지 않는다. 신경 signal motor391을 roster로 정렬해 native motor_rate[1:601]과 정확 비교한다. muscle_activation도 실제96과 일치해야 한다.
3. E150 신호는 anchor 없이 control1..2000의 endpoint다. control0은 start-observation에서 가져오며 사건 계산은1..600만 쓴다. current는 종료 rate에 `_current(rate)`를 평가한 관측이며 다음 neural step의 실제 적분 입력 전체를 뜻하지 않는다. 명령 row i는 (i-1,i]의 물리 구간에 적용된다. source 실행에서 main mask 변경이 없고 cuts=false임을 확인하며 전체 mask가 start/1s/1.1s/10s/E1853s에서 같은지 검증한다. 관측되지 않은 중간 전신 상태를 창작하지 않는다.
4. 경계 전체 상태는 E150 start/1s/1.1s/10s와 E185 final3s, 총5개다. 전체 network dt=.00025, ticks=control*20, n=158706, packed CUDA 경로를 검증한다. 모든446 observed rate/drive/adaptation이 해당 endpoint signal과 bit-exact 일치해야 한다. 3초 재사용은 위 prefix 검증 통과 뒤에만 허용한다.

## 계산식

5. 현재 CSR topology를 원본 graph와 비교한다. 다섯 postsynaptic 행의 모든 incoming edge를 원래 순서대로 저장한다. edge ID/pre ID/root/type/contacts/graph mV weight/rate baseline(sign*contacts*.03)/navigation scaling/assembled calibration/individual-fit/현재 float32 weight를 구분한다. 기존20 navigation root에 해당하는 motor-bound 음의 baseline에는 원래×64를 재현하고 calibration replacement 및 individual mapping을 검증한다. 최종 실제 row와 bit-exact 같아야 한다. 과거 .00069를 현재값으로 가정하지 않는다.
6. 경계에서 전체 incoming rates/masks로 E178 `warp_current`를 계산하고 다섯 post의 저장 recurrent_current와 float32 bits가 같아야 한다. packed pre-mask는 Torch float32/subnormal 보존, CSR multiply/add는 CUDA FTZ·fmad=false·32lane fixed reduction이다. float64 edge 합은 설명용이며 CUDA total과 동일시하지 않는다. 5경계×5post의 모든 edge term과 양/음 합을 저장한다.
7. 연속601행에는 incoming pre가 observed446에 실제 포함된 연결만 `float64(weight)*float32(rate*mask)`로 계산한다. 각 연결의 CUDA 단일 product도 필요하면 별도 저장하되 partial sum을 정확한 전체 CUDA current라고 부르지 않는다. 알려진 양·음/전체 descriptive sum, 저장 actual current, `current - known_sum` 잔여를 저장한다. 잔여에는 미관측 입력과 부동소수 합산차가 함께 들어가며 결측 source를0으로 대치하지 않는다. actual current+drive-threshold-adaptation의 endpoint net input은 float64 진단치이며 적분기 미분 재현/인과성 주장이 아니다. 다섯 노드의 rate/drive/current/adaptation/net input 및 시간 변화도 보존한다.
8. 사건은 E186의 전체11개, 선택8개[131,219,308,391,433,480,521,564] 그대로다. bands pre[-20,-10),lead[-10,0),post[0,10),recover[10,21), whole1..600/evaluation100..599/startup1..99, selected8 pooled와 모든 개별11사건을 보존한다. E186의 RM native_count[:,4,:]와 negative force magnitude weight를 사용한다. negative/zero/positive를 평균 전 native에서 구분한 조건이며 새 event/lag를 선택하지 않는다. E186 window labels/condition counts와 일치해야 한다.
9. 각 신경 신호 및 모든 observed incoming edge의 조건평균은 sum(native_count*x)/sum(native_count)다. 유한치만 그 유효 가중치에 포함하고 분모0은 NPZ NaN/JSON null, count0으로 보존한다. 시작 anchor에 native를 만들지 않는다. negative magnitude 가중 평균을 별도로 제공한다. 일반 시간 평균도 같은 고정 bands에 함께 제공해 제동 조건의 선택 효과를 드러낸다. 양/음 입력 합에 대한 known contribution coverage는 경계 전체 입력의 절대 합을 분모로 보고하고, 순전류 near-zero 나눗셈으로 과장하지 않는다. 통계적 독립 표본·유의성·실제 근육 힘으로 해석하지 않는다.

## 구현, 실행, 판정

새 도구 `tools/analyze_c_banc_m3_mn_current.py`와 테스트 `tests/test_c_banc_mn_current.py`만 실제 gpt-6-sol/high가 구현한다. 부모 설계→Sol 구현→부모FIRST→필요시 같은Sol첫수정→부모SECOND→부모 잔여수정. high 이유는 시계/CSR calibration provenance/부분 관측·부동소수 경계의 결합이며 CUDA 수식 자체는 재사용한다. 원본 입력 분석은 소스 리뷰 승인·hash gate 뒤1회만. 합성 tests는 stale hash, prefix 불일치, 0가중치/결측, 행·roster 순서, subnormal CUDA 누적과 일반 합의 차이, 전체/부분 incoming 구분, event 반례 및 원본 출력 충돌을 검증한다. 원본 데이터 미리 계산해 테스트 기대값을 만들지 않는다.

출력 `e187-mn-current/`의 analysis.json, aligned.npz, terminal.json에 전체 입력/current inventory, 경계 및 모든 bands/사건, 검증 항등식/한계를 담는다. 기존 출력 덮어쓰기는 금지다. 부모가 독립 식으로 주요 identity·조건 평균·전류 항을 검산한다. 원본1회 CPU1/BLAS1, internal40초, 부모 실행43초+cleanup2초=45초, 재시도0, 출력256MiB, 여유disk2GiB, 새모델/신경/물리/GPU/fit0. 누적4346.283222499653/4994초(잔여647.7167775003472초)에서 실제 감독 벽시간을 실패 포함 승계한다. 준비·리뷰 비용은 별도 기록한다.

유효성, 관측이 다음 선택에 주는 근거, 후보 채택 없음, 전체 F1–F8 미완료를 별도로 판정한다. 불일치로 분석이 멈추면 실패 영수증과 소비 시간을 보존하고 숨겨 재실행하지 않는다. 현재 설계 시점 E187 원본 실행0, F6/UI 변경0이다.
