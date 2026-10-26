# E326: 시작 제약으로 정한 단일 효능 보폭

E325는 연속 형태 손실을0.8116% 낮췄지만 시작 q 최대0.04350538>0.04로 기각됐다. RMS와 tendon 제약은 통과했다. 저장 자료에서21/4158스칼라가 위반했고, 다섯 관절 열의 양방향 오차 폭이0.08보다 커 일정 offset은 불가능하다. E325 결과·기각·한도는 그대로 남긴다. 새 질문은 실제 관측된 시작 제약으로 정한 더 작은 단일 효능 변화가 형태 개선과 시작 안정성을 함께 만족하는지다. 임의 step-size grid나 반복 backtracking은 하지 않는다.

현재 모델의 E323 중립300출력(q0,t0)과 E325 실제 연속minus(q1,t1)를 사용한다. 시작99행에 대해 affine 예측 q(a)=q0+a(q1−q0), t(a)=t0+a(t1−t0)를 정의하고 a∈[0,1]에서 기존 E103 RMS/최대 .018/.04/.009/.025 네 제약을 모두 만족하는 구간의 상한 a_max를 해석적으로 계산한다. 최대 제약은 모든 스칼라 선형 부등식, RMS는 제곱평균의 볼록 이차 부등식이다. a=0의 엄격한 기존 통과를 확인하고, 상한과 어느 제약이 제한하는지 전부 기록한다. 한 경계에 붙지 않도록 허용구간의 중점 a*=a_max/2를 단 하나 선택한다. 수치 여유를 이유로 원래 기준을 완화하지 않는다. 관측된 q최대 부등식만의 상한은0.1537563862였지만 구현에서는 네 제약 전체를 계산해 결정한다.

theta*=float32(1+a*×(E325 실제theta_minus−1))를 float64로 승격한다. 이는 이미 유효하게 관측된 효능 방향의 보폭을 출력 제약으로 정하는 별도 실험이며 E325의 판정을 소급 변경하지 않는다. 각 그룹의 실제 f32 변화·원래 현재 CSR의 변화·선형 예측과 실제 결과 차이를 저장한다. E325 saved gradient를 다시 최적화하거나 미분하지 않으며, 반올림된 theta와 이상적 보간 간 차이도 공개한다. Affine 신경 궤적은 가설이지 보장이 아니며 새 원본 실행으로만 판정한다.

E325 source/input/output/audit와 E323 중립, E103 시작 기준의 현재 hash를 gate에서 검증한 뒤 원래 전체BANC158706·현재CSR·20tick/control·E150 decoder·E319 목적을 그대로 구성한다. E325 Dependencies의 입력 검증을 재사용할 수 있다. 원래 control0 r/z/bank/original/mask에서 고정된 E318 drive300행과 새 theta 하나로300control을 연속 실행한다. 중간 reset0, 별도 중립 신경 재실행0, 별도 decoder 경계 재생0, backward/recompute/optimizer/body/physics0이다. 실제 forward6000tick과 포함된 decoder300회의 요청/반환/장치확인, 상태·q/LTM·receipt를 보존한다.

실제 모든300행이 유한하고 receipt가 완전하며, 전체 E319 shape가 E323중립보다 낮고 E103 startup 네 제약을 모두 통과할 때만 constrained_continuation_supported다. 이것만으로 물리 보행 후보를 채택하거나 F1–F8 통과를 주장하지 않는다. 실패하면 최초 오류·사용 비용·raw를 남기며 동일 계약 재시도0, 제약 재설정0, 원본 재탐색0이다. 기준이 통과한 뒤에만 별도의 실제 신체 검증 설계를 검토한다.

원본 supervisor60/internal50/science35/finalization15초, CPU1/host6GiB/추가GPU8GiB/output1GiB/free disk4GiB다. 현재 누적8955.373115399565/9060초, 잔여104.626884600435초 중60초는 생성 자격이 통과한 뒤에만 예약한다. 원본 한도 증가0. 감독이 종료되기 전 활성 결과/원장을 읽지 않고 종료 뒤 실제 시간을 한번 합산한다.

Actual gpt-6-sol/high가 새 tools/verify_c_banc_m3_constrained_secant.py와 tests/test_c_banc_constrained_secant.py만 구현한다. Root가 설계·FIRST검토, 필요하면 같은 Sol이 R1, root SECOND 후 잔여수정은 root가 맡는다. 기존 모든 source는 불변이다. Sol/root CPU각45초, root 생성CUDA20초/host2GiB/추가GPU1GiB. 작은 배열에서 네 부등식의 해석해를 독립 계산과 비교하고, 영변화·비유한·불가능한 중립·극소 계수·f32 무변화 및 실패 계수/게시를 검사한다. 생성 작은 네트워크에서 실제 연속 경로의 새 backward0/중간reset0/receipt/독립 NumPy 목적을 확인한 다음에 원본 단회다. F1–F8/M2/UI/F6미노출을 유지한다.
