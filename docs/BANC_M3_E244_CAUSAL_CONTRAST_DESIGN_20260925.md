# E244: 자기 신경 입력의 변화량 표현 대조

E243은 관찰된 뒤4초의 큰 범위 이탈을 지지하지 않았고 LM/LH/RM에서 가까운 입력과 다른 목표 명령의 조합을 확인했다. E242는 여전히 기각한다. H244는 절대 E/I 입력과 두 저역 기억의 차이를 사용하고 각 채널의 학습 변동량으로 정규화하면 운동 상태를 더 잘 구별할 수 있다는 가설이다. slow 성분이 원인이라는 확정이나 전체 신경망 정보 한계 주장은 아니다. 이 단계는 새 표현 한 개의 저장 자료 대조이며 학습·정책·보행 후보가 아니다.

## 사전에 고정하는 단일 표현

E241의 원래 descriptor=(x3,fast3,slow3)를 읽는다. 각 MN 자신의 z6=(x-fast,fast-slow)만 사용한다. 현재값과 자기 인과 기억의 차이며 다른 MN/몸/목표/외부시각/phase/시계를 추가하지 않는다. 원래 h_fast 16ms, h_slow 실제 inverse_adaptation_tau, 입력 scale/gate/cap/tau, anatomy는 바꾸지 않는다. 지연 차분이나 미래 표본은 사용하지 않는다.

각 MN·좌표의 표준화 s=max(std(z[control1..1200], ddof=0),1e-4)를 float64로 계산한다. 모든391 MN의 s를 기록하되 원래 free343만 이웃 특징에 사용한다. 중심 빼기는 거리 차이를 바꾸지 않으므로 하지 않는다. training 밖 표본과 teacher q는 s 계산에 들어가지 않는다. 이는 표현 변환과 채널 표준화를 함께 바꾸는 하나의 사전 가족이며 결과에서 두 요소의 기여를 따로 식별했다고 주장하지 않는다.

E243과 동일한 원래 민감도 w=.02*cap*pool*norm(moment_leg7)를 다리별 합1로 정규화한다. 다리별 특징은 원래 MN/좌표 순서로 concat(sqrt(w_m)*z_m,a/s_m,a)다. 유클리드 거리, TRAIN reference control1..1200, query1..2000, 절대control차20이하 양쪽 제외, 동률은 최초 학습control이다. query0는 index -1/distance0/q차이0로 제외한다. 한 다리·128query 이하 batch로 계산하고 2001×391×6 전체 특징이나 거리행렬을 파일로 복제하지 않는다.

기준 이웃 index/distance/q차이는 이미 검산된 E243 raw를 사용하고 새 기준 탐색은 실행하지 않는다. 기준 index 범위/100ms 제외/teacher q 차이 exact를 확인하고 기존 report의 TRAIN/steady/뒤구간 RMS를 재현한다. 원래 anatomy w를 다시 계산하여 E243 w와 일치해야 한다. 새 raw에는 scale391×6, free원본순서, sensitivity6×343, 새 nearest index/distance2001×6, 새/기준 teacher q차이2001×42, 실제 batch 완료 수/행 prefix를 저장한다. float64 유한성·완료 범위·부분 raw 보존을 확인한다.

## 판정

전체TRAIN1..1200, steady100..1200, 관찰된 뒤1201..2000의 q차이 RMS/최대와 거리분포를 전체/6다리로 보고한다. 1초 10구간도 같은 쌍 q차이 RMS/최대를 기록한다. 이웃으로 고른 명령은 실행하지 않으며 명령 차분 속도를 보행 성능으로 주장하지 않는다.

H244 지지는 steady와 관찰된 뒤구간 각각에서 전체 q차이 RMS가 E243의80%이하이고 모든6다리 RMS가 각 기준 이하여야 한다. 기준 그대로 비교하고 epsilon으로 통과시키지 않는다. 수치 비교 허용치는 atol2e-10/rtol2e-12, 정렬/ID/원본은 exact다. 실패 조건을 모두 남기며 통과를 보행 성공 또는 독립 일반화 검증으로 승격하지 않는다. TRAIN 입력이 아닌 뒤4초는 E242부터 노출된 개발 자료다. F6의 두 초기자세는 미노출을 유지한다.

통과하면 해당 자기입력 반응 가족의 새 prototype/basis 및 단일 계수 적합을 별도로 설계하고, 다음 독립 검증 입력을 새로 확보한다. 실패하면 단순한 자기입력 변화량 표현은 지지하지 않는 것으로 기록하며 bandwidth·역치·표준화 sweep을 자동 실행하지 않는다. 그때는 원래 감각→신경 경로의 운동 상태 구분을 검토한다. E150 최선/M2/UI/기존 모든 기각/F1–F8 기준을 유지한다.

## 개발과 실행 범위

실제 gpt-6-sol/medium이 tools/analyze_c_banc_m3_causal_contrast.py와 tests/test_c_banc_causal_contrast.py만 구현한다. 저장 배열의 명시적 변환/이웃 비교이며 새 solver·CUDA가 없어 medium을 선택한다. 부모 설계→Sol 구현→부모 FIRST→필요시 같은Sol 첫수정→부모 SECOND→잔여 부모 수정. E243의 검산된 순수 helper를 읽기 재사용할 수 있지만 frozen 파일은 수정하지 않는다. 부모는 표준화·해부학 가중치와150개 이웃 탐색을 독립 검산한다.

생성 검사는 자기입력 공통상수 불변성, 여섯 좌표 순서, TRAIN 전용 std와 floor, 미래 표본 변경에 대한 scale 불변성, 비영 거리/제외/동률, 기준 q provenance/판정 실패, partial raw 및 최초/마무리 오류와 최신 manifest를 포함한다. 생성 solver/CUDA/모델 호출0이다.

원본1/재시도0/fit0/solver0/CUDA0/fullBANC0/body0/physics0, CPU1. 내부60초(마무리10초), 외부75초, 감독85초, RSS3GiB/출력128MiB/여유2GiB로 E243 감독 규모를 유지한다. 원본 누적6591.914057899619/6755초, 잔여163.08594210038063초에서85초 예약, 배정증가0. 모든 frozen SHA/패키지/부모review를 수치 load 전 확인하며 반환 raw·실제 계산진입/완료·threadpool/오류·프로세스 트리·시간을 보존한다. 분석/생성/부모감사 비용은 별도로 기록한다.
