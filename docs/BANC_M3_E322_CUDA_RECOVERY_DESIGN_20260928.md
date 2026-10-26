# E322: 초기화 순서를 고친 full-horizon 기술 회복

E321의 네 생성 CUDA process 중 CuPy-first만 같은 CUBLAS_STATUS_INVALID_VALUE를 재현했다. 첫 GEMV는 통과했으나 gradient 평가의 다음 GEMV 및 뒤의 두 크기에서 실패했다. Torch-only/Torch-first/Torch-CUDA-first는 모든 순방향·gradient 비교를 통과했다. 실패 조건에만 NVIDIA cu13 경로와 Torch 경로의 cublasLt64_13.dll이 동시에 관찰됐다. 초기화 순서 원인은 지지되지만 DLL 내부 결합 기전까지 입증한 것은 아니다.

E320 FIRST→같은 gpt-6-sol/xhigh R1→root SECOND가 끝났으므로 남은 기술 수정은 사용자 지정 절차대로 root가 한다. 실패 당시 최종 source/test는 parent-e320-failed-final-sources에 원래 freeze hash와 대조해 보존했다. 기존 E320 결과·원본 8.6558974초 비용·H320 not_tested 판정은 유지한다.

수정은 resource meter가 CuPy를 import하지 않고 Torch의 mem_get_info를 쓰게 하는 것이다. 이 호출이 Torch의 CUDA 초기화를 먼저 수행한다. 신경·decoder·loss·direction 수식과 E318 원본 입력, 모든 과학 판정·자원 상한은 그대로다. 명시적인 새 E322 schema/출력 디렉터리/동결 계약을 사용하며 E320 출력은 덮어쓰지 않는다. 환경·DLL 설치·앱·M2는 변경하지 않는다.

원본 전 검사는 생성 CPU 40초, 생성 CUDA 30초로 한정한다. CPU는 기존 orchestration 18검사와 정적 검사다. CUDA는 새 process에서 실제 수정 meter를 가장 먼저 호출하고, CuPy가 import되지 않았음과 Torch CUDA 초기화를 확인한 뒤 이미 검증된 E317 생성 full-BPTT fixture를 실행한다. 생성 CUDA host2GiB/additionalGPU1GiB, CPU1, 원본/신체/fit0이다. 과거 생성 검사를 성공했다는 이유만으로 실제 입력 검사를 생략하지 않는다.

생성 검사가 통과하면 E322 원본을 단회 예약한다. 남은 원본 482.9244920003657초 중 감독480초, 내부450초/과학410초/게시40초를 사용하며 누적 상한9060초를 늘리지 않는다. 원본 누적 출발8577.075507999634초, retry0. 원본 시작 전에 현재 source/전체 input/review/실패·회복 증거를 고정한다. 감독 종료 전 활성 원장 내용은 읽지 않는다. 종료 후 실제 감독시간을 한 번만 합산한다.

E320의 중립300control 정확 재현, 6000tick 전체 backward, 고정 sign 방향 한 쌍, 관측/예측 방향 오차≤5%, 기존 startup 네 한도를 그대로 적용한다. 불일치/비유한값/기술 실패/시간·메모리 초과면 실패 원인과 부분 실행량을 보존한다. 성공해도 조건부 감각 입력의 미분 검증이며 자연 보행이나 F1–F8 완료로 선언하지 않는다. F6 초기상태는 노출하지 않는다.
