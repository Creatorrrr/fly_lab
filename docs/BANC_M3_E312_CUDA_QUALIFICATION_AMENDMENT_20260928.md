# E312 CUDA qualification 기술 중단 후 잔여 검사 사전 계약

원래 설계의 모델, lambda0/1, 오차 허용치와 F1–F8은 유지한다. Root SECOND source review 이후 최초 생성 CUDA pytest 묶음이 25.103922499984037초에서 자식 wall25초 상한에 도달하여 감독자가 종료했다. stdout의 두 완료 표시만 남았으며 전체 receipt/정확한 신경 tick은 없으므로 전체 PASS로 해석하지 않는다. stderr에는 수치 예외가 기록되지 않았다. 이 사실만으로 모델의 수치 성공이나 실패를 단정하지 않는다. 실패한 감독 receipt 및 부분 로그를 보존한다.

중단한 100-tick CPU/CUDA 비교 묶음은 재실행하지 않는다. 원래 준비했던 root 독립 39-cell CUDA 검산은 앞선 감독자가 중단하여 아직 시작하지 않았다. 이 미실행 검산을 최대35초로 수행한다. 이어 아직 실행되지 않은 CUDA 음수 subnormal/sticky fault/pointer 교체 검사 한 건을 최대10초로 수행한다. 동일한 수치 기준, 직접 실행 대 graph 실행, lambda0 SSP 정확성, lambda1 독립 계산, nonzero disk future를 모두 요구한다. 전체 pytest 묶음이 통과했다는 주장은 하지 않는다.

최초 실패 비용을 포함한 생성 CUDA 단계의 사전 상한을60→80초로 늘린다. 이번 잔여 감독 요청은 최대45초이고 누적 최대70.104초다. 실패한 요청의 자동 반복은 없으며, 이 문서는 root가 부분 출력과 종료 원인을 확인한 뒤 작성한 새 사전 계약이다. CPU1/host2GiB/additionalGPU512MiB/output16MiB, 원본0/실제 신체0/solver0/fit0을 유지한다. 원래 설계 파일은 SHA 고정을 유지하고 이 추가 계약을 qualification의 근거로 함께 pin한다.

전체 BANC 원본 누적8416.895937599536/8560초와 E313 미예약 상태는 변하지 않는다. 생성 CUDA 통과는 실제 보행이나 생물학적 검증 완료가 아니다.
