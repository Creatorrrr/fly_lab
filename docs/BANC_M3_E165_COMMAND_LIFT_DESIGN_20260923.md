# E165: 저장 자세에서 여섯 다리 명령의 수직·전방 방향 진단

부모 설계, 2026-09-23. E164는 유효한 단일 적합으로 종료했고 고정된 onset 함수족을 기각했다. 기존 신경 리듬을 유지한 E107 shape-only 진단은 성공했지만, 현재 BANC의 어느 명령이 지지 발 회수와 충돌하는지는 아직 분리되지 않았다. 이 실험은 새로운 보행 후보·fit이 아닌 정적 기구학 진단이다. F1–F8, 원래 시작99, M2, 미사용 F6 두 자세, 활성 UI와 모든 실패를 보존한다.

## 경쟁 설명과 기존 실험의 차이

1. 현재 요청 자체가 지지 중인 발의 각 부분을 아래/앞으로 향하게 한다. 이 경우 MN→근육 전달 또는 상류 입력을 수정할 근거가 되며, 서보의 단순 지연만으로 설명하기 어렵다.
2. 실제 서보에 유효한 요청은 위쪽인데 지지가 남는다. 이 경우 접촉·힘줄·부착·구동의 동역학을 우선 조사할 근거다. 국소 방향이 위쪽이라는 사실만으로 실제 발이 떠야 한다고 주장하지 않는다.
3. raw 요청과 actuator에 유효한 요청의 방향이 다르거나 다섯 tarsal segment의 부호가 섞인다. 이 경우 포화 또는 기하학적 모호성을 별도로 보고하고 앞의 두 설명 중 하나로 강제 분류하지 않는다.

E152는 추적 오차·발 운동·지지를, E161은 RM 1–2초의 명령 변화율을 접촉 Jacobian에 투영했고, E163은 신경 위상·활성·명령과 초기 힘을 대조했다. 이번 수량은 **같은 현재 자세에서 target−actual이 암시하는 변위**이며 명령 변화율이 아니다. E155/E157에서 실제 RM 접촉은 tarsus2/3였으므로 tarsus5만 보지 않고 여섯 다리의 tarsus1–5 body 원점30개를 모두 계산한다. 이 원점들은 실제 접촉점과 같지 않다. 결과를 접촉점 힘·미래 clearance·인과 기여율로 해석하지 않는다.

새 실제 gpt-6-sol/high 읽기 전용 감사는 일반 비대칭 활성화와 하중별 반응속도가 M2-E64/E54에서 이미 기각됐음을 확인했다. 단순 반복을 새 기전으로 취급하지 않는다. 사용량 의존 말초 효능과 기존 감각→MN 전달은 후속 가설이며 이번 진단만으로 서로 구별할 수 없다. E150의 neural_request가 null이므로 저장자료만으로 감각 drive→MN 전파를 복원했다고 주장하지 않는다.

## 입력·시계·기하 계약

원래 E150 start StateStore, E150 trace.json, E152 e150/features.npz 및 event-stacks.npz, E154 context.json, E155 layout.json과 부모 최종 검토를 사용한다. E164 최종 결과·부모68검산·고정 source gate와 이전 보존 해시도 연결한다. 설계와 freeze는 첫 새 방향 계산 전에 고정한다.

E150의2001행/control0..2000, control_dt=.005, neural20tick/control, qpos109/qvel108/ctrl82/각도42를 확인한다. control0은 초기 관측이므로 기하 진단은 **control1..2000의2000개 endpoint**다. shape/finite/clock/body identity, 요청−실제각도와 저장 tracking error, features의 load와 trace의 load, time 및 기존 [.5,10) window, event46/offset−20..20을 검사한다. 시계 atol2e−8, 같은 저장 신체·각도·좌표 비교 atol1e−12를 사용하며 오류 앞 raw를 보존한다.

기존 `analyze_c_banc_observation.new_body`와 E154의 `construct_body_safely`로 동일 FlyBody1개를 구성한다. 생성 settling2000 step과 복원은 준비 비용으로 별도 계수한다. StateStore1회, GraphStore/network0. 원래 actor는 복원 후 읽기 전용이고 별도 shadow MjData1개만 사용한다. E155 `validate_layout`의 joint/actuator/hinge/root 매핑 검사를 재사용할 수 있다. seed/world/params/body model hash와 기존 E154 body 설정·발 geom mapping·weight0를 대조한다. flat terrain,42개 unit-gear position actuator, control clamping 활성화를 확인한다. 지원하지 않는 설정은 추측하지 않고 거부한다.

각 저장 endpoint에서 shadow qpos/qvel/ctrl/time을 설정하여 mj_forward를 정확히1회 호출한다. scientific physics integration0이다. Thorax rotation 및30개 tarsal body 원점의 world→UI 변환 좌표가 원기록과 일치해야 한다. 순서는 LF/LM/LH/RF/RM/RH, 각 tarsus1..5이며 body name/ID와 실제 joint qpos/dof/actuator 주소를 저장한다. XML 이름이나 index를 추측하지 말고 body.body_indices/body_ids와 MuJoCo names를 함께 대조한다.

각 원점에 mj_jacBody를1회 적용한다(총60,000회). full Jacobian 중 같은 다리7관절을 J7로 저장한다. 다른 다리 actuator 열은 atol1e−12에서0임을 확인한다. floating-root와 수동 관절은 명령 변위에 넣지 않는다. 세 요청을 나란히 계산한다:

- requested: trace.joint_targets_rad.
- delivered: trace.ctrl[실제42 actuator IDs]. 요청과의 차이를 저장하며 같다고 가정하지 않는다.
- effective: delivered에 **해당 모델의 actuator_ctrllimited/ctrlrange만** 적용한 위치 요청. 실제 control limit 동작을 관측하기 위한 계산이며 모델·후보를 수정하는 clip이 아니다. 제한이 없는 actuator는 그대로 둔다. 제어 clamp disableflag가 설정돼 있으면 이 설계는 거부한다.

각 요청 r와 현재 실제 각도 q에 대해 d=J7*(r−q)를 계산한다. 단위 mm, z=world-Z 수직, ap=수평 정규화 Thorax +x와 내적이다. 3D변위, z/AP 각 관절 기여7개 및 합을 보존한다. 다섯 segment의 min/max/평균, `all_positive`(모두>1e−12mm), `all_nonpositive`(모두<=1e−12mm), mixed를 계산한다. 1e−12는 수치 영점대이며 생물학적 들림 높이 기준이 아니다. 유한 변위 적분·부착 해제·근육 힘으로 확대하지 않는다.

## 고정 집계·사건

집계 구간은 기존 E152 [.5,10) window다. all, forward(speed>.5mm/s), interrupted(speed<=.5), 각 다리 supported(load>=.02BW), supported_interrupted, supported_interrupted_recovery(추가 ego_ap_velocity>0)의 고정6집합을 다리별로 보고한다. 각 수량의 표본수·산술평균·분위수[.05,.5,.95]와 부호 분율을 저장한다. supported 집합에는 추가로 원래 boundary normal load를 사용한 분자합/분모합 평균을 보고하고 분모0은 null로 유지한다. 빈 집합·NaN을0이나 PASS로 채우지 않는다. raw 세 요청의 차이와 control clipping 표본수도 함께 제시한다.

기존46개 속도 하향 사건과 −20..20control/±.1초 창을 그대로 가져온다. 새 peak/lag/창 검색은 없다. 이 진단의 control1..2000 범위를 벗어난 표본은 unavailable/NaN이며 실제 validity mask와 온전한 사건 수를 표시한다. 46개를 독립 생물학 표본으로 취급하지 않는다. raw event stack은 요청별 z/AP 및5segment 범위·load·기존 actual ego/world AP velocity를 갖는다. scalar summary 전에 raw를 저장한다. 자동 후보 선택 또는 보행 PASS는 없다.

## 구현·검증·자원

소유 파일은 `tools/analyze_c_banc_m3_command_lift.py`, `tests/test_c_banc_command_lift.py` 두 개다. 부모 설계→실제 gpt-6-sol/high 첫 구현→부모 FIRST 리뷰→같은 Sol 첫 수정→부모 SECOND 리뷰/잔여 직접 수정이다. high 선택 이유는 신체 기하·좌표계·actuator 제한·오류 보존을 함께 다루며 새 신경/CUDA 구조는 없기 때문이다. 기존 body·M2·평가기·UI·과거 도구는 읽기 전용이다.

CLI --freeze/--freeze-sha256/--source-review/--source-review-sha256/--out. exact freeze/design/두 owned-source gate, 기존 artifact 전후 hash와 실행 snapshot 동일성을 확인한다. hash 중복은 병합하여 전후 각1회 읽는다. 새 출력만 허용하며 원본1회/자동재시도0이다. 아직 원본 발송 승인은 없다.

생성 테스트는 회전된 root/다른다리 열 분리, J×delta 독립 scalar, raw/delivered/effective 차이와 제한 없는 actuator, 혼합 segment 부호, 빈/NaN 집합, 같은양수/음수/0 하중 가중, clock/mapping 오류, 명령0 기준, 저장 실패와 마지막 저장 시간·용량 초과의 성공 취소를 포함한다. 작은 생성 MuJoCo chain에서 Jacobian을 중심차분 h=1e−6rad와 비교(atol1e−7mm/rad)하며 원본 body를 만들지 않는다. full2001 입력/2000기하 결과의 generated fake-body pipeline과 첫/중간/마지막 오류의 partial 보존을 테스트한다. 부모는 별도 scalar 및 생성 기구학 검산을 수행한다.

데이터 shape/구조 검사 후 배열은 사전 할당하고 available/completed mask를 둔다. 최초 예외를 유지하며 미완성·partial·atomic 임시 파일은 남긴다. 숫자 판정은 해당 raw 저장 뒤에 한다. body/shadow cleanup, 입력/source snapshot 재검사, 마지막 파일 저장까지 성공 조건에 포함한다. 실패한 최종 저장·정리·자원 검사는 성공 receipt를 취소한다. bare 실행 횟수는 성공한 호출 직후 증가시켜 관측/저장 오류와 분리한다.

원본 process1, CPU1/GPU0, body1/shadow1/StateStore1, settling2000, scientific physics0/neural0/fit0, shadow forward2000/Jacobian60000. 내부90/외부110/감독120초, raw192MiB/free2GiB. 이전 감독 보수누적2817.1626464000437초를 올림한2818에120을 더한 사전2938초다. 새로운구성 또는 다음 forward 진입 전에 저장·정리10초 여유를 확인한다. 반복 fit·τ/gain 탐색·F6 노출·실제 보행 승격은 이 설계에 없다. 유효한 진단 뒤 근거에 따라 다음 단일 개입을 별도 설계한다.
