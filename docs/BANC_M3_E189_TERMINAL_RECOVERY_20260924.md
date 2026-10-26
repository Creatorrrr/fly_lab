# E189 종료 보고 복구 — 원래 실패 보존

low 원본은 600 control, 12000 neural, 30000 physics와 시작/종료 checkpoint 및 반력·근육·관절 원자료를 저장했다. 감독 wall은102.825174799997초이며 새 누적4555.749688599717/4994초다. terminal은 실패이고 이 판정을 바꾸지 않는다. 최종 보고서에서 이미 닫힌 session.network.tick을 읽어 AttributeError가 발생했으며 몸체 close가 network를 None으로 해제하는 실제 동작을 합성 owner가 재현하지 못했다.

부모가 두 번째 리뷰 이후 잔여 수정으로 계산 횟수를 close 전에 보존하고, 동일한 객체 해제를 합성 검사에 넣는다. 기존 v1 동결·소스·감독·terminal·모든 원본 출력은 보존한다. 제어·가중치·초기 상태·물리·관측·startup·시간창·자원 한도는 바꾸지 않는다. low를 자동 또는 수동 재실행하지 않는다.

우선 부모는 원본 v1의 전체 source/input SHA와 저장 checkpoint·ledger·601 control/근육·30000 native/force, 초기 단일 CSR/CUDA 변경, startup 및 고정 비교창을 독립 검산한다. 종료 보고 처리만 실패했고 complete native 자료의 의미가 원래 계약과 같은 것이 확인될 때에만 별도 saved-result sidecar를 만들고 이를 **원본 실행 성공과 구별된 저장 자료의 수용**으로 기록한다. 원본 result/terminal을 덮어쓰거나 실패를 제거하지 않는다. 검산이 실패하면 low의 물리 효과를 유효한 비교로 쓰지 않는다.

high는 수정된 종료 처리와 별도 v2 동결·부모 리뷰·감독으로 원래 1.02배/zero-state/3초를 단1회 실행한다. low 실패 원본과 같은 신경·물리 경로임을 v1→v2 변경 검토로 확인한다. 이전 원본 비용을 승계하며 high 한도150초, 재시도0, F6노출0, M2/UI 보존이다. F1–F8과 자연보행은 미완료이고 어느 팔도 최종 후보로 채택하지 않는다.
