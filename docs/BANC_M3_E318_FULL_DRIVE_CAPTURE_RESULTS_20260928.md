# E318 전체 신경 입력 수집 결과

현재 SSP/u0 모델에서 전체158706 뉴런의 실제 입력과 제어 경계 rate/adaptation, raw MN391을1.5초/300control 동안 수집했다. root 독립 원자료 감사는 수집을 승인했다. 보행 파라미터 변화, 최적화나 새 보행 후보는 없다.

원래 E307/E310과 control0 whole은 wall까지 같고, control100/300 whole은 계측 wall_s만 제외하고 모든 상태가 정확히 일치했다. 전체301개 control/recruitment/activity와15000개 native contact 및 trace는 채널별 dtype/shape/bytes가 같았다. 새 full drive의7527개 기존 관측부분은 이전 drive와 같고 모든300개 raw MN은 해당 full rate의391개 순서와 정확히 일치했다.12개25control chunk,395개출력파일377,411,788B를 저장했다. Source97/input402개 pin 및 출력manifest/상태파일도 다시 확인했다.

Root 독립 audit32,507개 check가 통과했다(parent-e318-original-raw-audit.json,3.394245초,신규 모델/CUDA/물리0). 원본은 단회61.9564337초, neural6000tick/physical15000step/body1 및 별도 계측 construction/setup/settling이다. device-total 추가 메모리 peak522,190,848B, 감독 treeRSS3,709,788,160B로 고정 범위 이내다. 전체 감독비용을 한 번만 합산하여8568.419610599642/8660초, 잔여91.5803894003584초, 예약0이다. 원본 재시도는0이다.

개발은 actual gpt-6-sol/xhigh 최초 구현→root FIRST→같은Sol R1→root SECOND→root 잔여수정 순서였다. FIRST의 provenance/비용/device총메모리 결함을 Sol이 고쳤고, SECOND에서 마지막manifest 저장 실패가 성공으로 반환되는 결함을 root가 고쳤다. 생성15개/Ruff와 독립 주입실패가 통과했다. Windows 파일교체 오류 최초probe도 보존했고 owned 파일교체만 제한재시도한다. 실제 수집에는 오류가 없었다.

사전 freeze는 e318-full-drive-capture-freeze.json, source-review는 e318-full-drive-capture-source-review.json이다. 원자료/result/cost는 verification/banc-natural-gait-m3-20260919/e318-full-drive-capture에 있으며 parent-e318-original-supervisor.json이 원본비용의 기준이다.

H318은 이 고정된현재모델/1.5초 관측재현 범위에서 지지됐다. 신경 가중치를 바꾸었을 때 고정입력 replay의 개선이 실제 feedback 보행으로 이어지는지는 미검증이다. 다음 E320에서 전체1.5초의 방향미분 예측을 먼저 검사한다. F1–F8는 미달이며 M2/UI/미사용F6/이전실패는 보존했다.
