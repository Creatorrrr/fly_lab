# E185: 원래 보행을 그대로 재현하며 지면 반력과 질량 중심 관측

E184는 완료된 원자료 분석과 부모 315개 검산으로 다음 결정을 바꿨으므로 이전 goal turn은 progress다. 두 기록 모두 전진 하향 통과 11개/최장 전진 .215초다. 전반적인 MN 출력과 평균 약 2.5개 다리 지지가 남은 사건 직후에 slip이 증가했지만, 어느 다리가 추진·제동하는지는 기록되지 않았다. F1–F8과 자연 보행 목표는 그대로 미완료다.

## 질문·비교·변경 범위

E182 theta0 baseline을 같은 3초/전체 BANC 158706/CUDA/초기 상태/모든 파라미터로 한 번 재현한다. 변경은 별도 shadow MjData에서 읽는 계측뿐이다. 제어, 신경, 가중치, 근육 전달, adhesion, 몸체, 원래 100µs 건강·slip 관측은 그대로다. 외부 CPG/위상/궤적/root 교정과 새 연결은 추가하지 않는다. 제품·M2·활성 UI·F6에는 반영하지 않는다.

경쟁 설명은 A 다리별 지면 반력이 상쇄됨, B 전체 지면의 전진 반력이 감소함, C 몸통 왕복 운동과 전체 질량 중심의 이동이 다름이다. 지면 힘만으로 모든 외력이나 원인을 설명했다고 주장하지 않는다. 결과에 따라 기존 신경 경로/근육·신체 전달의 한 가지 인과 개입을 고르는 관측이며 새로운 보행 후보가 아니다.

## 기록 정의

기존 E113 `install_native_observer`의 한 step, integration-state 복사, shadow `mj_forward`, feet·upright·nonfoot·fault 검사와 원래 tuple을 보존하는 별도 observer를 작성한다. live MjData에는 어떤 관측용 forward/상태 쓰기도 하지 않는다. 새 observer 설치 시 기존 hook을 보관하고 모든 종료 경로에서 복원한다. 새 함수는 explicit E185 실행 구간에만 E181 protocol의 observer hook으로 연결하고 예외에도 module hook을 복구한다. 원래 소스 파일과 gate를 수정하지 않는다.

추가 표본은 30000개의 같은 native 시각에서 다음을 보존한다.

1. `_contact_forces('feet', floor_only=True)`와 같은 geom1/geom2 부호·contact frame 전치·exclude/지면 필터를 사용한 30개 발마디 world 힘 및 6개 다리 합. model force 단위와 BW 단위를 함께 둔다. floor는 support_geom_ids다. 접촉이 없으면 힘은 0이며, slip의 결측과 구별한다.
2. shadow thorax 위치/회전, 물리 forward 단위벡터, XY 평면에 정규화한 heading, 각 다리 force의 두 방향 투영. y-up UI와 native z-up을 섞지 않는다. native 좌표를 그대로 기록하고 좌표 의미를 metadata에 둔다.
3. 검증된 fly body_ids, body_mass와 shadow xipos로 계산한 전체 질량 중심 위치. ID 중복/범위/누락, mass 유한·비음수·총합을 body.mass와 확인하고, 양의 질량을 가진 동적 fly body가 누락되면 거부한다. 실제 구성에서 이 검사가 성립하는지 원본 실행 전에 소스/합성 구조로 검토한다.
4. body.mass, weight0, gravity, IDs/질량, 관측 시점/모델·물리 identity를 metadata로 보존한다. force 원자료에는 시각·유한성·완전성 및 부분 실패를 드러내는 필드가 있어야 한다. append가 실패하거나 건강 중단이면 이미 수집한 원자료도 finally에 저장한다.

이 힘은 **적분 뒤 상태에서 shadow forward로 재계산한 순간 반력**이다. dt=.0001을 곱해 합한 것은 그 표본의 적분 근사다. 직전 solver의 실제 충격량이라고 부르지 않는다. normal force에 adhesion 반응이 포함되므로 이를 빼지 않은 net weight-support 주장도 하지 않는다. 합성 contact로 두 geom 순서, 제외된/비지면 접촉, 회전 frame, 6×5 합계/투영/질량 중심을 확인한다. 신체 움직임이나 별도 toy MuJoCo 실행은 합성 테스트에 쓰지 않는다.

## 실행·재현 유효성

새 `tools/observe_c_banc_m3_ground_reaction.py`, `tests/test_c_banc_ground_reaction_observer.py`만 실제 `gpt-6-sol/high`가 개발한다. 이유는 기존 CUDA/native 실행과 물리 좌표, 원본 hook 복원·실패 비용을 함께 다루기 때문이다. 부모 FIRST → 같은 Sol 첫 수정 → 부모 SECOND → 잔여 부모 수정 후 해시 source gate를 발행한다. Sol은 합성 테스트만, 원본 실행은 부모만 한다.

E185 freeze/source-review는 이 설계, 두 소스, 부모 metadata, E184 최종 비용·원자료 검토, E182 baseline 전체 inventory, 원래 E181 freeze/source-review의 실 해시를 묶는다. E181 `verified_inputs(..., mode='baseline')`가 반환한 gate로만 원래 `run_arm`을 호출한다. 임의 gate 객체나 과거 판정 조작은 금지한다. 실행 출력은 `verification/banc-natural-gait-m3-20260919/e185-ground-reaction/` 한 곳의 create-only이며 실패도 재실행하지 않는다. receipt를 모델 접근 전에 만들어 import/gate/실패 비용을 남긴다.

원래 기록은 계속 같은 포맷으로 `baseline/`에 남긴다. 600 control/12000 neural/30000 physical 시계와 startup99/건강 판정을 유지한다. E181 `compare_baseline_prefix`를 E182 기준에 그대로 적용한다. 신경 bank/MN/activation, joint targets/qpos/qvel/ctrl, body position/basis/contact 등은 **정확히 같아야** 하며, 기존 시간 허용오차2e-8도 바꾸지 않는다. 추가로 actual joint/body legs와 native force의 시계/개수/finite/질량을 확인한다. 비교 실패·불완전·건강 중단은 계측 유효성 실패로 보존하고 force 원인 판정에 사용하지 않는다. J/slip/순전진/.215초와 같은 결과를 재현해야 한다.

유효할 때만 부모가 E184 기준에서 고정한 사건 index `[131,219,308,391,433,480,521,564]`를 사용한다. 모든11개 사건 및 제외 이유는 기존 기록 그대로 보존한다. ±100ms/41제어 표본, pre[-100,-50), lead[-50,0), post[0,50), recover[50,100]ms를 유지하며 native50표본을 오른쪽 닫힌5ms 구간에 놓는다. 여섯 다리 양/음 force와 합, 적분 근사, 질량 중심/몸통 속도를 모든 사건별로 보고한다. 선택된 한 다리·lag만으로 결론 내리지 않는다. Gaussian20ms 및 중앙차분은 비인과적이므로 선행 시점은 원인 증거가 아니다. F4는 기존 thorax 기준을 유지한다.

## 자원·종료·비용

이전 누적 **4236.676628199639/4994초**, 잔여757.323371800361초를 승계한다. E185 전체 원본 단계는150벽초 이내(원본 준비/gate/readiness/실패/저장/분석 포함), main child는100초 내부/105초 외부+5초 정리다. native팔1회, solver/최적화0, 재실행0, 단일 CPU worker/단일 GPU worker. free20GiB, baseline raw1GiB, 전체새출력2GiB. 부모 합성·소스 검토는 별도 기록한다. 모든 원본 분석 시간은 기존 누적에 더하며 부분 실패도 0으로 취급하지 않는다.

모델 전 기존 E182 queue/client readiness를 두 번 2초 간격으로 읽고 모두 ready일 때만 시작한다. native 중 queue/GPU 자원도 기존 reader로 2초 간격 기록한다. 다른 client의 중단·설정 변경 권한은 사용하지 않는다. unknown/비정상 관측은 기록하고 계측 해석을 보류한다. readiness 실패는 native0의 비용 영수증으로 종료한다.

model construction, CUDA setup/capture, settling, main neural/physics 횟수와 partial 실패는 원래 E181 ledger를 보존하고 직접 센 것과 source-derived 추정치를 구별한다. 완료 후 부모가 hook/source/raw 보존, child 종료, 정확 재현, 힘 변환·질량 중심·event 집계를 리뷰한다. 어떤 결과도 자체로 F1–F8 완료/후보 채택이 아니다. A/B가 구분되면 기존 연결/전달의 한정 개입을 설계하고, C가 강하면 몸통-질량중심 운동을 유발하는 관절·접촉 조합을 진단한다. 실패 시 원본 재실행 없이 실제 실패 단계와 새 근거가 있는 다음 방법을 선택한다.


## 원본 전 v2: 발 이외 다리 접촉도 분리 기록

FIRST source review에서 FlyBody의 TIBIA_TARSUS_ONLY 및 `nonfoot_contact()`(flylab/body.py:630)가 경골-바닥 접촉을 정상 허용함을 확인했다. 따라서 30개 발마디 합만으로 전체 지면 반력을 부르면 누락이 생긴다. 원본 실행·force 결과 열람 전 같은 shadow contact pass에 48개 다리 마디(6×coxa, trochanterfemur, tibia, tarsus1–5) 및6개 전체 다리 합, 검증된 fly body_ids/geom_bodyid 기반 전체 fly-지면 합, 전체 fly에서 다리 합을 뺀 잔차를 추가한다. 기존30개 발마디 필드는 유지하고 전체 다리/전체 fly에 model force·BW·forward/XY heading 투영을 둔다. 발이 닿지 않고 경골만 닿는 합성 접촉으로 누락 방지를 검증한다. 모든 관측은 같은 적분 뒤 shadow 상태에서만 이뤄지고 원래 tuple·힘/제어·실행1회 예산은 바뀌지 않는다.

부모 원자료 분석은 전체 fly와 전체 다리의 XY heading 반력을 주 비교로 삼고, 발마디와 나머지의 기여를 구별한다. 양/음 힘은 각 native 표본에서 먼저 나눈 뒤 평균·적분한다. 상쇄 비율은 (P+N−|P−N|)/(P+N), P=여섯 다리 양의 힘 합, N=음의 힘 크기 합으로 정의하고 P+N=0은 결측으로 둔다. 방향이 섞인 후 절댓값을 취하거나 0을 채우지 않는다. 원래5ms thorax 속도/F4와 추가100µs shadow thorax/COM 속도는 구별한다. 추가 속도도 중앙차분 후 Gaussian20ms이며 회전 몸통 forward와 XY heading을 각각 투영한다. 표본 위치·필터 차이는 인과 해석의 한계로 보고한다.

부모 외부감독은 driver125초+정리5초 이내, 후속 원자료 분석은20초 이내로 전체150초 상한을 지킨다. 원본 준비/readiness/import/native/save/추가분석 실패 비용을 모두 단계에 합산한다. 설계v1과 최초metadata는 보존하고 v2 metadata 해시를 source freeze에 사용한다.
