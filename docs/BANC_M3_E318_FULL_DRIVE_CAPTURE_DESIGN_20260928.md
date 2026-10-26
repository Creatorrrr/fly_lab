# E318: 현재 SSP 중립 모델의 전체 감각 입력 수집

E307/E310의 activity는 158706개 중 7527개 뉴런의 drive만 저장했고 whole snapshot은 몇몇 시점만 보존했다. 따라서 현재 감각·descending·navigation 입력의 전체 제어 시퀀스를 기존 파일에서 정확히 복원할 수 없다. E317 이후의 전체 신경 조건부 학습에 필요한 입력을 현재 모델에서 한 번 수집한다. 기존 E20/E29/RK4 drive나 외부 teacher 입력으로 대체하지 않는다.

H318은 관측만 추가한 동일 SSP/u0 세션이 기존 E307의0–.5초 및 E310 neutral의.5–1.5초와 동일한 신경·decoder·물리 결과를 내고, 매 제어에서 실제 full drive를 저장할 수 있다는 것이다. 경쟁 설명은 입력 관측 시점 또는 복원 상태가 달라져 후속 궤적이 변하는 경우다. 이 실험에는 신경/운동 파라미터 변경·fit·새 보행 후보가 없고 E310의 실패 판정을 변경하지 않는다.

## 소스와 소유

신규 `tools/verify_c_banc_m3_full_drive_capture.py`, `tests/test_c_banc_full_drive_capture.py`만 actual gpt-6-sol/xhigh가 구현한다. Root FIRST → 필요한 같은 Sol R1 → root SECOND → root 잔여 수정을 지킨다. 기존 프로파일·native observer·advance journal·raw capture/save 구현을 재사용하되 E310의 두 arm 실행과 전체 옛 계보 gate를 복제하지 않는다. 모델 및 제품 소스는 변경하지 않는다. 구현 단계는 생성 CPU만 허용한다.

고정 복원점은 `verification/banc-natural-gait-m3-20260919/e307-ssp-neutral-body/neutral/observed-start`의 control0 whole이다. `SSPRecoveryResourceTrial.from_checkpoint`로 그대로 복원한다. 모델 identity는 `f9ea25308cf5e074047bf16afba55de9fac89411ca48127f2c37cc4e6e34c7be`, adapter는 `1fd7b86939a6846e01c4837fb5c2fbe24ce8c776e3331c1f75921a376c53b067`다. GraphStore 경로·hash는 이 producer의 freeze에서 가져와 고정한다. u0, resource1, 실제 CUDA packed/20tick을 검사한다.

## 관측과 비교

세션 소유 network의 `advance_readout` 한 곳을 일시적으로 감싼다. 호출 직전의 full drive는 adapter.encode·descending·navigation 합산을 마친 실제 입력이다. 정확한 float32 (158706,) 원본 값을 제어 번호와 함께 소유 복사한다. 다른 값으로 변환하지 않는다. expected motor IDs391/steps20/현재 tick을 확인하고 원래 메서드를 원래 인자로 한 번 호출한다. 반환값을 그대로 반환한다. 반환 직후 전체 rate/adaptation, 반환 MN391과 실제 network tick을 저장하고, 해당 host copy가 끝난 prefix만 장치 확인 완료라고 기록한다. Motor readout은 현재 native `_readout_packet`과 같은 raw `rate[motor_ids]`여야 한다. output_mask는 presynaptic current에만 적용되므로 readout에 다시 곱하지 않는다. Mask 자체는 whole에서 그대로 보존·검증한다. 요청·반환·장치확인·디스크저장 수와 실패한 부분 레코드를 구분한다.

첫 whole과 control100/300 whole을 저장한다. 실제 neural drive/r/z/MN은25 또는50control의 제한된 chunk NPZ로 순서대로 저장하며 앞 chunk를 덮어쓰지 않는다. 기존 raw collector로 control0..300의 bank/original activation/activation/q/LTM, 선택 activity, trace와 native contact15000samples도 저장한다. Native health guard는 기존 upright≥.9/nonfoot0을 유지한다. 건강·기술 실패에는 진행을 중단하고 이미 얻은 raw와 가능한 partial state를 보존한다. 예외 후 observer를 원래 instance/class 상태로 정확히 되돌리고 세션을 close한다. 첫 실패와 후속 저장/cleanup 실패를 별도로 보존한다.

Whole 초기 복원은 wall까지 정확해야 한다. control100/300 whole은 각각 E307 neutral/final 및 E310 neutral/final과 비교하며 진행 시간 계측인 최상위 wall_s만 명시적으로 제외한다. 다른 필드를 임의로 지우거나 tolerance를 확대하지 않는다. 순수 저장 표현의 tuple/scalar 차이는 기존 require_saved_whole_equal 규칙만 허용한다. Raw는 E310 neutral/combined의 전체 control0..300와 native15000을 채널별 dtype·shape·bytes로 비교한다. 비교에서 관측된 새 full drive의 subset이 기존 activity.drive와 정확히 같은지도 검사한다. 미래 replay·F6 또는 생물학적 타당성을 주장하지 않는다.

원본은 완전한 full-drive300, endpoint300, controlraw301, native15000, exact 기존 비교, 오류0 및 source/input/output pin이 모두 통과해야 학습 입력으로 승인한다. 불일치는 유효 비교 실패 또는 기술 원인으로 구분하고 학습을 시작하지 않는다. Partial records는 진단 근거만 된다. 수집 성공은 보행 개선이 아니다.

## 실행과 자원

Fake dependencies로 정확한 호출 순서·drive 직전/MN 직후·입력 불변·detach 복원·partial 실패와 저장/cleanup 실패·off-by-one·ref mismatch·chunk 경계·원본 경로 gate를 검사한다. Sol CPU 전체120초/root120초, CPU1/host2GiB, 원본0이다. Pass 이후 root가 source와 StateStore 모든 파일, graph bundle, E307/E310 비교 raw 및 producer 결과/manifest/audit를 pin하고 별도 review/qualification을 기록한다. 모듈 import나 passive gate에서 모델·GraphStore·StateStore·CUDA를 생성하지 않는다.

미래 원본은 단회/재시도0, scientific neural6000tick/physics15000step/완료control300, body1이다. Constructor의 neutral settling·CUDA capture/setup은 기존 construction_with_body_count로 별도 실제 계측하고 unknown 비용을0으로 간주하지 않는다. Supervisor150초/internal135초/science105초/finalization30초, CPU1/host6GiB/additionalGPU4GiB/output2GiB/free4GiB다. 현재 누적8506.463176899651/8560초·예약0이며 아직 원본 실행하지 않는다. 소스 검토 뒤 150초 예약 시 단계 배정을8560→8660초로 사전 증액한다. 이는 이전 비용을 보존한 현 finite-goal 내부 단계 배정이며 사용자 전체 예산을 새로 설정하지 않는다.

Root가 원자료와 반환된 감독 원장을 별도로 감사한 후 한 번만 실제 wall을 합산한다. 다음은 저장된 full drive를 사용한 E317 전체 신경 rollout 재현 및 실제 학습 구간의 유한 크기 방향 검증이다. 조건부 drive 학습과 closed-loop 보행의 차이를 유지하고, 동일 후보의 기존 startup→short→10초 및 F1–F8을 모두 통과해야 승격한다. M2·활성 UI·미사용 F6·모든 기존 실패와 비용은 유지한다.
