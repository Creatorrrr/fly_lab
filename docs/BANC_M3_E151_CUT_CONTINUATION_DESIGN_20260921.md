# E151 동일 후보의 운동 차단·재시작 확인

E150 부모 잔여 검증기 수정의 연속 확인이다. E150의 10초 main/clone 원자료와 전체 incomplete는 그대로 보존한다. 수정된 검증기는 여섯 다리 LTM의 0과 원래 두 복부 자세 유지값 0.2를 각각 엄격히 검사한다. 자세·신경망·모집 계수·CUDA 동역학을 바꾸지 않으며 F3–F5를 재실행하거나 재분류하지 않는다.

부모가 E150 두 번째 리뷰 이후의 잔여 수정과 이 좁은 검증을 맡는다. 새로운 Sol 구현 또는 effort 변경을 주장하지 않는다. 실제 Sol/high FIRST 구현 → 부모 FIRST review → 같은 Sol FIRST correction → 부모 SECOND review → 부모 잔여 수정의 순서를 따른다.

동결된 E150 checkpoint-1s를 원래 전체 158706 BANC 그래프로 한 번 복원하고 검토된 `run_cut_restart`를 사용한다. 정확히 4 cut + 16 restart, 400 neural ticks, 1000 physics steps다. source 및 restore warmup/capture는 별도로 기록한다. 첫 control의 원자료를 E150에서 이미 기록된 control과 비교한다. 나머지 19개 control은 새 측정으로 명시한다. 새 전체 main·fit·solver·LP·외부 CPG·F6 노출·UI 변경은 없다.

원본1/재시도0, CPU1/GPU1, 내부45/외부55/감독65초, raw512MiB/free12GiB다. 이전 실제 누적2417.283482900006초를 보존하고 ceil2418+65=2483초를 배정한다. 기존 전체 배정2514초 이내다. raw/ledger/결과/manifest 저장까지 포함하고 최초 오류와 cleanup 오류를 보존한다. 최초 실행 전 동결 입력/보존/코드 SHA 및 부모 source gate가 필요하다. 후속 부모 검산은 저장자료만 사용하며, 성공하더라도 F2의 다리 운동 전달 증거일 뿐 F7의 자율 정지·목표 재시작이나 M3 완료가 아니다.
