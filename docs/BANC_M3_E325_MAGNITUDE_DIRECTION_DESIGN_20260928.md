# E325: 검증된 국소 미분의 크기를 보존한 단일 방향

E324는 현재 전체 BANC의 50×20ms 조건부 미분이 실제 변화와0.048604% 오차로 맞음을 보였지만, 8028개 그룹을 같은 ±0.1% 폭으로 바꾼 sign 방향의 연속 실행은 shape/startup 모두 실패했다. 저장 gradient 제곱합98.4286%가10개 그룹에 집중됐다. 유효 미분을 같은 폭으로 이산화해 약한 그룹에도 큰 변화를 주는 방식이 불필요한 상태 변화를 만들 수 있다는 가설을 검사한다. 이 설명은 아직 가설이다. 기존 기각과 비용은 보존한다.

E325는 E324의 검증된 gradient g를 그대로 불러와 d=g/max(abs(g)) 하나만 사용한다. 최대 효능 변화 .001은 그대로이며 상대적인 변화 방향을 sign에서 gradient 크기 비례로 바꾼다. theta±=float32(1±.001*d)를 float64로 승격한다. 파라미터 탐색/step-size sweep/재미분/optimizer/재시도0이다. 이 방향은 유클리드 거리에서의 gradient 방향이며 해부학/효능 양수/원래8547그룹·725849edge를 유지한다. gradient가 작아 f32에서 변하지 않는 그룹을 명시적으로 계수한다. 저장 자료로 예상된 pair 변화1.724590272e-6은 E324 sign 예측의53.3926%, minus theta L2변화0.00191872는 이전0.08960062보다 작다. 이 수치는 선형 예측이며 실제 개선 증거가 아니다.

원본 계약은 E324 전체 신경망158706, 현재 CSR11584852, E318 control0/300drive, 고정 E150 decoder와 E319 M128 닫힌곡선을 그대로 사용한다. E324 source/input/output/감사V2를 현재 hash로 검증한 gate 후 한 번 로드한다. E324의 중립50개 원자료와 gradient 및 감사결과는 입력으로 재사용하며 중립 신경/BPTT를 다시 돌리지 않는다. g가50개 저장 gradient의 정확한 합인지 확인한다. E324의 판정·전체 상태 재현·continuation 실패와 출처를 확인하고 진행한다.

E318의 원래 raw391MN으로 decoder를 control0부터300번 재생하여 bank/original 경계를 구성한다. q/LTM300개≤1e-11, bank/original100/300≤1e-10을 다시 대조한다. 각 c=100,104,...296에서 원래 r/z[c-1], 재생 bank/original[c], 원래 mask와 drive[c:c+4]로 plus/minus 두 번만 실행한다. loss는 E324.short_shape의 같은4×6×7 최근접 선분 평균이다. 모든50개 창을 사용하고 일부 창 선택0. 창별 raw q/LTM/끝상태/receipt와 loss, predicted contribution을 저장한다.

전체 J±는50창 평균, observed=J+−J−, predicted=sum(g*(실제theta+−theta−))다. 양쪽finite·절대값>1e-9, 같은 부호, 상대오차≤5%만 supported다. Supported이고 Jminus<E324 중립일 때에만 같은 theta_minus로 control0부터300control을 리셋 없이 한 번 실행한다. 연속 E319 손실이 중립보다 낮고 E103 startup의 .018/.04/.009/.025를 모두 통과해야 continuation_supported다. 결과 뒤에 기준을 바꾸지 않는다. 조건부 신경 진단이며 신체 후보 채택·F 성공은 항상0이다.

최대 신경 forward14000tick(2×50×4×20 + 조건부6000), backward/recompute0, decoder replay300회, 신체/물리/optimizer0이다. 중립 재사용은 실행 비용으로 세지 않고 출처를 표시한다. Requested/host-returned/device-confirmed/오류/부분 실행 계수와 raw 보존은 E324의 수정된 패턴을 따른다. 최초 오류를 게시 오류로 덮지 않고, 요청을 완료로 쓰지 않는다. 원본 supervisor140/internal120/science100/finalization20초, CPU1/host6GiB/추가GPU8GiB/output2GiB/free disk4GiB다. 누적8896.9861794996/9060초에서 잔여163.0138205004초 중140초는 생성 자격 통과 뒤에만 예약한다. 원본 상한 증가0, 활성 원장은 감독 종료 전 읽지 않는다.

Actual gpt-6-sol/xhigh가 새 tools/verify_c_banc_m3_magnitude_direction.py 및 tests/test_c_banc_magnitude_direction.py만 구현한다. 기존E324/E323/adjoint/decoder source를 수정하지 않는다. 수식은 단순하지만 전체망 원자료 재사용·출처·실패 계수·조건부 실행의 정확성이 필요해 기존 Sol/xhigh를 재사용한다. Root FIRST→같은 Sol R1(필요하면)→root SECOND→root 잔여 수정이다. Sol/root 생성CPU각60초, root 생성CUDA30초/host2GiB/추가GPU1GiB. 생성 작은 모델에서 실제 코드의 공유 방향/중립 재사용/미분 실행0/독립 NumPy손실/조건부 분기와 실패 경계를 검증한다. 원본은 그 후 단회다. F1–F8/M2/UI/F6미노출은 유지한다.
