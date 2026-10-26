# E154 E153 기록 형식 교정 후 별도 검증

E153 원본 시도1은 exit1로 종료됐다. 몸체1개/settling2000step을 수행했고 anchor qpos/qvel/ctrl은 원본과 같았으나 첫 scientific command 이전에 `neural_request=None`을 거부했다. scientific/neural step0이며 힘 측정은 없다. 감독14.1772279초, 누적2480.1772279초였다. E153의 실패·raw9파일·실행 소스71f81c0c…를 보존한다. 이는 보행 실패의 새 측정이나 유효한 힘 결과가 아니다.

같은 Sol의 FIRST correction과 부모 SECOND review 이후 발견한 잔여이므로 부모가 직접 수정한다. 부모 SECOND review는 생성 request dict만 검증하여 실제 E150의 null request 계약을 놓쳤다. E150 trace의 모든2001행은 neural_request가 명시적 null이며, joint_targets/adhesion/tendon_inputs 및 command_interval은 존재한다. 체크포인트의 네 cuts는 false, push_left=0, push_force=[0,0,0]이다. E150 metadata는 unchanged E141 native body protocol을 명시하고, 원래 main loop는 외부 push/cut 이벤트 없이 고정 drive에서 advance/capture_control한다.

## 제한된 교정

기존 explicit neural_request는 그대로 검증한다. null fallback은 정확히 `flylab.m3.e150.native-monotone-body-trace.v1` + E150 numeric_protocol=`unchanged E141 native body protocol` + 모든 row의 명시적 null + checkpoint 네 cuts false/외력0일 때만 허용한다. 이 조건에서는 원래 main protocol의 no-cut/no-push를 명시적 요청으로 해석한다. 추정 provenance를 context에 기록하고 원본 trace를 수정하지 않는다. 미등록 schema/부분 누락/혼합 request/활성 cut 또는 외력이면 생성 전에 거부한다. 200개 command를 모두 신체 생성 전에 검증해 이번 형식 오류를 조기에 잡는다. 최종 수용은 여전히 E153 설계의 매 control qpos/qvel/ctrl와 10000개 native time/load/slip/valid/nonfoot/upright exact 비교에 달린다.

E153 힘 수식, [1,2]초 구간, mask, 모든 F1–F8, CPU body/CUDA0, 1body+1shadow, settling2000+scientific10000, 목표/모델/파라미터는 그대로다. 변경 파일은 동일 tool/test 두 개이며 원래 실행 source는 archive로 보존한다. 부모의 생성 schema regression, 전체 mock replay, force scalar 검산 및 CLI 검토 뒤 별도 실행한다. 새 body integration 성공이 BANC 새 보행 후보의 성공은 아니다.

## 추가 시도의 비용과 한도

이것은 같은 진단을 완성하기 위한 **두 번째 전체 시도**이며 E153 실패를 새 이름으로 성공 처리하지 않는다. 수정 소스에 대한 원본 process1을 별도 사전 배정하고 자동재시도0이다. 이전2514초 배정과 실제 사용2480.1772279초를 보존한다. ceil2481+감독48=2529초로 누적 상한을15초 추가한다. 이유는 scientific step 이전의 기록 형식 결함을 수정한 후 한 번 검증하기 위해서이며, 가설 적합·반복수·성공 기준을 완화하지 않는다. 내부35/외부40/감독48초, CPU1/GPU0, raw512MiB/free12GiB는 유지한다. 이 문서·입력·원래 E153 결과·수정 source gate를 실행 전에 동결한다. 추가 실패 시 원본을 반복 실행하지 않고 raw와 미해결 원인을 보고한다.

F6/UI/M2 변경0, E150 original incomplete 유지, 후보 채택/goal complete=false. 독립 saved review는 CPU1/내부30/외부40초, 모델/적분0이다. signed contact 결과 이후 제어기 개입은 별도 설계한다.
