# E177 초안: 같은 모델에서 하중 피드백 복원 대조

E176 상태 wrapper 리뷰와 생성 검증 뒤 사용할 부모 설계 초안이다. 아직 동결 계약, 실제 입력 전체 해시, 원본 source gate, 새 원본 실행은 없다. 이 문서는 원본 실행 허가를 대신하지 않는다.

## 변경과 비교 대상

E150의 동일한 `candidate-neutral`에서 두 팔을 독립적으로 복원한다. 양쪽 모두 E176 `LoadFeedbackTrial` 경로로 재구축한다. baseline은 같은 다섯 기존 CSR entry에 원래 값과 같은 target을 지정한 identity patch, candidate는 고정 `.03/.06` 복원 patch다. 두 실제 신경망의 모든 비선택 data, CSR topology, tau/a/threshold/cap, gain/log-tau, 감각 encoder, readout/근육 mapping, 세계/신체 상태, 제어·물리·신경 dt는 같아야 한다. 새로운 연결, 외부 phase/teacher/root 교정, fit은 없다.

baseline도 같은 wrapper·신경 재구축 경로를 거치게 하여 wrapper 자체와 효능 개입을 구분한다. 원래 E150 main은 완전한 원시자료가 있고 마지막 cut 판정이 기술 실패였다는 이력을 유지한다. 이를 전체 원본 성공으로 바꾸지 않는다.

## 실행 전에 고정할 사항

1. 실제 입력은 E150 neutral checkpoint의 모든 구성 파일, 같은 GraphStore bundle, E150 고정 parameter/calibration/muscle provenance, 원래 startup99 표본, 기존 observation IDs, 부모가 검산한 E150 main 비교 원시자료로 한정한다. 새로운 자세나 F6 자료를 사용하지 않는다. 전체 해시와 필요한 NPZ key/shape/dtype만 먼저 확인한다.
2. E176 최종 source review와 그 실제 두 소스 hash를 pin한다. 기존 E141/E150 native helper를 재사용할 경우 정확한 함수와 isolated override 허용 목록을 고정한다. 기존 source를 전역 수정하지 않는다.
3. baseline을 먼저 10초 실행한다. 완료한 원래 E150 main과 MN rate, 활성 bank/근육 활성 및 관절·tendon 명령의 동일 표본을 비교한다. 단순히 설정 이름이 같은 것을 동일 비교로 간주하지 않는다. 허용 오차/bit-exact 대상은 원본을 다시 실행하기 전에 결정하고 동결한다. 불일치 시 candidate는 시작하지 않고 원인을 별도로 검토한다.
4. candidate는 같은 고정 10초를 실행한다. 양쪽에서 원래 1초 disk checkpoint→독립20 control 미래를 비교하고, 별도4 control motor cut/16 control restart를 기록한다. cut 검사는 F7 자율 정지·재시작을 대신하지 않는다. 전체 BANC CUDA 158706 neurons, 391 MN, 96 muscles, dt .00025×20/.005/.0001×50을 유지한다.
5. 직접 피드백 효과를 해석할 수 있도록 두 sensory cell과 다섯 MN의 실제 rate/drive/adaptation/전체 recurrent current, 선택된 다섯 edge의 `weight × masked presynaptic rate`, RM 실제 하중 및 다리 운동을 동일 시계로 보존한다. 이 항은 직접 current의 진단이며 indirect current나 단독 물리적 근육 힘으로 해석하지 않는다.

## 판정과 비용의 경계

건강 중지/실행 실패/자료 누락과 유효한 보행 실패를 구분한다. raw를 먼저 저장하고 부모는 요약 판정보다 raw와 실제 CSR 대조를 먼저 검토한다. strict startup qRMS .018/qmax .04, LTM RMS .009/max .025, native F3–F5와 기존 미끄러짐을 그대로 평가한다. 전진 거리만 늘거나 신경 발화만 변해도 자연 보행 후보로 채택하지 않는다. 본 시험이 F3–F5를 만족해도 F1/F6/F7/F8은 같은 후보의 후속 검증이 필요하다.

예상 scientific 요청은 2×(main10 + future.1 + cut/restart.1)=20.4초, neural81600 ticks/physics204000 steps다. 모든 source restore·CUDA warmup/capture·body 구성은 별도로 실제 진입 회계에 포함한다. CPU1/GPU1, 자동 재시도0, arm당 준비60/body180초, 전체 내부560/외부600/감독615초, raw8GiB/free20GiB를 검토한다. 사전 동결할 경우 기존 누적3085.011408499908초를 유지하고 ceil3086+615=3701초로 누적 원본 한도를 명시적으로 갱신한다. **현재 한도3202초는 아직 갱신하지 않았다.** 생성 테스트와 저장 검산 시간은 별도이며 알려지지 않은 과거 비용을0으로 간주하지 않는다.

먼저 E176을 완료하고 실제 실행기의 설계를 확정한다. 구현은 실제 gpt-6-sol/high, 부모 FIRST 리뷰와 같은 Sol 첫 수정, 부모 SECOND 리뷰 및 잔여 수정을 거친다. 단순히 code/test PASS가 나왔다는 이유로 이 초안을 원본 실행 허가로 전환하지 않는다.
