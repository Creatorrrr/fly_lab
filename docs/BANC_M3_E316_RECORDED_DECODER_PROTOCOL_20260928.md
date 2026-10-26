# E316 저장된 전체 근육 변환 재생 계약

E316 생성 CPU/CUDA qualification 뒤 실행할 단회 공학적 검증이다. 새 신경 계산·몸체·물리·학습은 0이다. 이미 저장된 E150의 MN 입력을 신규 functional decoder에 전달하여 기록된 내부 상태와 관절·힘줄 명령을 비교한다. 기존 E150의 보행 실패와 F1–F8 미완료는 그대로 유지한다.

현재 SSP/u0 whole(E307)의 adapter identity와 E150 start의 identity가 모두 `1fd7b86939a6846e01c4837fb5c2fbe24ce8c776e3331c1f75921a376c53b067`임을 metadata로 확인했다. E150의 원래 전체 실험은 cut 검사 실패로 incomplete이며, 완료된 main 기록의 독립 감사만 입력 근거로 쓴다. 다른 신경 solver 간 동역학 일치 검사가 아니다.

입력은 `e150-native-monotone-body/start/`, `native-recruitment-state.npz`, `trace.json`이며, E307 final state.json과 기존 E150 output manifest/parent saved-main audit를 계보에 묶는다. 정확한 source/input/생성 qualification SHA를 실행 전 freeze에 저장한다. motor rate의 float64 기록이 native float32 값의 정확한 변환인지 검사한 뒤 float32로 되돌린다. control0 bank와 원래 activation을 출발 상태로 사용한다.

control1–2000(원래 기록 10초)을 순서대로 .005초 timestep으로 CUDA에서 한 번 재생한다. 목표 명령을 입력으로 전달하지 않는다. 모든 bank/original activation/activation의 최대 절대차 ≤1e-10, 42 joint q와 6 tendon의 최대 절대차 ≤1e-11을 요구한다. 2000행 모두 유한하고 입력·출력 누락이 없어야 valid/supported다. control0 q와 원래 start body action을 neutral 값의 계보로 묶지만 실제 몸체의 cut/restart 검증으로 부르지 않는다.

원자료는 500control씩 네 chunk에 저장한다. CPU copy가 완료된 prefix에 대해서만 device-confirmed control 수를 올리고 저장 완료와 구분한다. 실패 시 완료 prefix/첫 오류/후속 저장 오류 및 비용을 보존하며 자동 재시도하지 않는다. root는 저장된 네 chunk에서 차이를 독립 계산하고 해시·계보·원장을 재검산한다.

원본 1회/재시도0, science16초/최종화3초/감독20초, CPU1/host2GiB/GPU추가512MiB/출력100MiB/free512MiB다. 현재 누적8501.121034199608/8560초에서 생성 검토 승인 후20초만 예약하며 배정 증액은 없다. 실제 감독 wall을 단1회 합산한다. 생성 CPU/CUDA 및 단순 hash 검사는 별도 비용이다. 결과가 통과해도 현재 페이지·M2·보행 후보를 변경하지 않고 후속 시작 제약/주기내 운동 형태 학습 설계에만 사용한다.
