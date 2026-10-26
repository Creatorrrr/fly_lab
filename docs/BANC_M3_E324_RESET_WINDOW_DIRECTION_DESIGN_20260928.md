# E324: 전체 steady 구간을 20ms 창으로 나눈 조건부 미분

E323은 중립 실행을 정확 재현하고6000tick 미분을 완료했지만, 고정 ±0.1% sign 방향의 실제 손실 차이가 예측과 반대였다. 감소 후보는 형태 손실과 시작 q 제한에 모두 실패했다. E24/E25 장기1ULP 실패, E315 국소1ms 통과, E323 source/원자료/155.438904초 비용을 보존한다. 큰 gradient가 장기 민감도·양자화·VJP 차이 중 무엇 때문인지 아직 단정하지 않는다.

E324의 변경 변수는 미분을 연결하는 시간 길이다. control101–300 전체200개를 시작경계100,104,...296의50개 비중첩 창으로 나누며 각 창은4control=20ms다. 일부 좋은 표본만 고르지 않는다. 모든 창에서 동일한8547개 P→P 효능 theta를 공유하고, 창 간 상태와 미분만 끊는다. 원래 현재 CSR의725849개 선택 연결과 해부학/SSP/decoder/감각drive/shape곡선은 유지한다. E29의 old source_weights는 사용하지 않는다. 런타임 보행에 reset이나 교사 명령을 넣는 개발은 아니다.

창 c의 시작 r/z는 E318 원래 control c의 기록, mask는 원래 control0 mask, 입력은 E318 drive[c:c+4]다. bank/original_activation은 control0의 원래 저장값에서 E318 raw391MN 기록과 고정 E150 decoder의 실제5ms 업데이트를300번 재생해 얻는다. 이 재생은 학습·신경 advance·신체0이며 decoder replay300회로 별도 계수한다. 전체300개 q/LTM을 원래 기록과 최대오차1e-11 이내, control100/300의 bank/original은 저장 whole과1e-10 이내로 확인한다. 각 창의 theta=1 출력 q/LTM≤1e-11, 종료 r/z dtype·bytes exact, bank/original≤1e-10이 통과해야 그 창의 gradient를 계산한다. 재현 실패는 실험 전체 기술 실패로 중단한다.

손실은 E319의 M128 닫힌곡선 최근접 선분 거리와 같은 수식이다. 각 창의4×6×7 q값에서 다리별7차원 제곱거리를7로 나누고 최근접 선분을 취한 뒤 시간·다리 평균한다. 50개 창 손실의 평균 J_reset이 목적이며 중립 J_reset은 E323 중립 손실과1e-12 이내여야 한다. 가짜 시작100행을 붙이거나 조건부 창의 startup 통과를 보고하지 않는다. 원래 startup 판정은 아래 연속 실행에서만 한다.

각 창의 gradient를50으로 나누어 합친 g8547를 한 번 계산하고, d=sign(g)를 사용한다. 같은 실제 f32(1±.001d)를 f64로 올린 theta 쌍만 평가한다. 각 창은 세 조건 모두 같은 원래 시작상태에서 실행한다. 전체 observed=J_plus−J_minus와 predicted=sum_f64(g*(theta_plus−theta_minus))가 finite, 양쪽 절대값>1e-9, 동부호, 상대오차≤5%일 때만 supported다. 기준을 결과 뒤에 변경하지 않는다. 창별 gradient, 손실, observed/predicted, q/LTM/종료상태/receipt를 모두 저장하여 상쇄·이상창을 드러낸다. 별도 창별 성공 게이트를 사후 추가하지 않는다.

이 국소 가설이 supported이고 J_minus<J_neutral인 경우에만, 동일 theta_minus 하나로 원래 control0에서300control을 reset 없이 연속 실행한다. 새 gradient/optimizer/후보 선택은 없다. 연속 결과는 원래 E319 전체shape손실이 E323 중립보다 낮고 E103 startup99의 .018/.04/.009/.025 네 제한을 모두 통과해야 continuation_supported다. 실제 관절·감각·몸체가 아직 닫힌 루프로 계산된 것은 아니므로 통과해도 신체 후보 채택은0이며 다음 신체 검증을 따로 설계한다. 조건부 입력 진단과 F1–F8 성공을 구분한다.

원본 호출은 단회, 최대 forward18000tick(50×4×20×3 + 선택적연속6000), 재계산4000tick, backward4000tick/200block, decoder replay300회, optimizer/body/physics0이다. 초기 조건 재사용50×3은 명시적인 surrogate 정의이며 실행 실패 재시도는0이다. 전체감각 dataset/source를 gate 뒤 한 번 로드하고 순차 창 처리하며 작은 같은고정model을50번 재생성하지 않는다. 모든 요청/host반환/device확인/실패와 가능한 raw를 원장에 저장한다. 구현 exception과 NaN은 과학 반증과 구분한다.

원본 감독240초/내부220초/과학190초/게시30초, CPU1/host6GiB/추가GPU8GiB/output2GiB/free disk4GiB로 제한한다. 누적8735.791969599613/9060초에서 남은324.208030400387초 중240초를 source 검토와 생성 검산 통과 뒤에만 예약한다. 누적 상한을 늘리지 않는다. 감독 종료 전 활성 원장 읽기는0, 종료 뒤 실제시간을 한 번 합산한다.

actual gpt-6-sol/xhigh가 tools/verify_c_banc_m3_reset_window_direction.py와 tests/test_c_banc_reset_window_direction.py 두 파일만 개발한다. exact 내부상태·전체BPTT·조건부실행·receipt와 실패보존이 결합된 난도를 반영한다. Root FIRST→동일Sol R1→root SECOND→root 잔여수정 순서를 지킨다. 기존 E314/E316/E317/E319/E323 source는 수정하지 않는다. Gate 전 heavy import/CUDA/원본 payload를 금지하고, source/input closure·E318/E323 감사와 입력조건을 현재 pin으로 검증한다. 새 fresh E324 output과 명시적 freeze/review를 쓴다.

생성 검산은 Sol CPU90초/root CPU90초, root 생성 CUDA60초, CPU1/host2GiB/추가GPU1GiB다. 원본 크기 데이터를 숨겨 생성 검산으로 실행하지 않는다. 작은 생성 신경망에서 reset-state와 모든창 공유 theta 미분의 독립 비교, objective의 NumPy 선분거리 대조, 중립 검증 전 backward 방지, 불완전 창/exception/확인없는 완료/연속 실행 분기와 실제 누적계수의 실패 경계를 확인한다. 코드/실험 비용을 모두 보존한다.

관련 연구는 접근을 뒷받침할 뿐 BANC 결과를 대신하지 않는다. [Multiple shooting for neural differential equations](https://arxiv.org/abs/2109.06786)는 구간 분할과 경계 제약을, [Multistep penalty NODE](https://arxiv.org/abs/2407.00568)는 불안정한 동역학에서 짧은 창과 경계 손실을 사용한다. E324는 trainable 경계나 경계 최적화가 없는 고정 관측상태 조건부 검산이므로 이 논문들의 전체 방법 구현으로 부르지 않는다. F1–F8/M2/UI/F6미노출과 모든 이전 실패는 그대로다.
