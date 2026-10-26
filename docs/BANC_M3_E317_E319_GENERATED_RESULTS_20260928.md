# E317/E319 생성 CPU·CUDA 검증

E317의 checkpoint 재계산 full BPTT 및 E319의 위상 독립 곡선 거리 구현은 생성 입력 검산을 통과했다. actual gpt-6-sol이 최초 구현과 첫 수정을 하고 root가 두 차례 리뷰와 독립 CUDA 검산을 수행했다. 실제 BANC 긴 구간이나 자연 보행은 아직 검증하지 않았다.

E317 FIRST에서 기본20×float32 timestep과 .005초의 이중 반올림 불일치를 검출하고 같은 Sol이 native timestep의 반ULP 규칙으로 고쳤다. SECOND43개 통과 후 생성 N397/MN391/group9,5control×20tick의 block1/2(불균등 마지막 구간 포함)를 검증했다. 출력6필드와 최초4상태 미분은 전체 unroll과 정확히 같고 효능 미분차≤6.662e-16이다. 엔진을 호출하지 않는 별도 dense f64 신경·decoder 수식과 미분 relativeL2 최대2.138e-6으로 고정5e-6 이내다. 각 receipt는 CUDA 동기화 후 forward100/recompute100/backward100tick을 확인했다. 생성 fixture peak allocated는 naive22,636,032B, block1 20,230,656B, block2 20,900,352B로 실제 BANC 성능 주장은 하지 않는다. 같은 retained graph의 반복 backward는 지원하지 않는다.

E319 FIRST에서 영점 RMS의 NaN 미분, 추상 cuda 장치 비교와 극대 입력 overflow를 검출했다. 같은 Sol R1 후 SECOND6개 통과, CUDA 생성300control/M17에서 CPU와 shape loss차0, per-leg차≤2.221e-16, q미분차≤1.085e-19다. 영점 RMS 미분은 유한하고 hot evaluate/backward의 CPU·NumPy·item·tolist·bool 추출 호출0이다. 곡선 거리는 각 다리 운동 형태를 평가하며 주파수·진행 방향·좌우 협응·균형·전진을 보장하지 않는다.

원자료는 verification/banc-natural-gait-m3-20260919의 parent-e317-independent-cuda.json, parent-e319-cuda.json, 각 first/second review 및 supervisor에 있다. 승인한 정확한 source/evidence SHA는 parent-e317-generated-qualification.json, parent-e319-generated-qualification.json에 고정했다. 최초 실패도 보존했다. root CPU/CUDA 비용은 각각 E31711.9340367/5.7781907초, E3198.5603329/2.0997373초다. 원본/신체/fit0이며 기존 원본 누적8506.463176899651/8560초는 변하지 않았다.

다음은 E318로 현재 full drive를 수집한 뒤 E320으로 전체1.5초 실제 방향 미분을 판정하는 것이다. E315의1ms 성공과 E24/E25의 장기 미분 실패를 함께 보존하고 원본 재현/방향 검증을 통과하기 전 학습 근거로 쓰지 않는다. 자연 보행 F1–F8는 미달이며 M2/UI/F6는 그대로 보존한다.
