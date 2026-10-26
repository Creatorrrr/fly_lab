# E207 결과: 작은 문턱 통과 초기화 미채택

실제 gpt-6-sol/high 구현 → 부모 FIRST 정적 검토 → 부모 SECOND 생성/모의 시험2개 및 Ruff를 통과했다. 수정이 필요한 결함은 없어 형식적인 첫수정은 만들지 않았다. 테스트는 모의 장치 호출만 했고, 원본은 이후 한 프로세스에서 CUDA 두 번을 실행했다.

기준 E205 선택43의 네 신경 배열은 bit exact 복원됐다. 사전 고정한178개에만 bias 0.075241666280를 적용하여 첫 RK 단계의 순간 목표 발화 .01Hz를 확인했다. 원래 초기 rate/adaptation과 나머지213MN의 모든 신경 배열,48LTM 및 tendon은 유지됐다. 이 순간 목표는 초기 rate를 덮어쓴 값이 아니다.

늦은 구간에서 rate RMS>=1e-4Hz와 max h_data>=1e-6을 함께 충족한 뉴런은 **61/178**개였다. 사전 기준89개에 못 미쳐 H207은 `bounded_threshold_initialization_not_supported`다. 기준 수치를 바꾸지 않으며 작은 문턱 통과만으로 충분한 학습 범위를 확보했다고 보지 않는다.

실제 조건부 명령 RMS는 0.058634534502rad, 원래 E150 대비 개선율은 -3.436714%로 오히려 악화됐다. 출발 RMS 0.023855876826rad/max 0.120812474292rad는 .018/.04 한도를 초과했다. 여섯 다리 명령/속도 오차도 악화됐다. 이 초기화를 후보로 채택하거나 물리에 적용하지 않았다.

부모는 bias 역식/마스크/상태 불변/decoder를 독립 검산했고, 명령 민감도 D 최대 차이는 1.137e-13였다. 기존20% 개선·출발·각 다리 보호 판정은 모집 신호 선별 기준과 별개로 유지했다. 추가 신경 계산 없이 저장 raw만 검산했다.

원본 감독 14.007838999998초, 누적 5426.123305699704/5564초, 잔여 137.876694300296초다. CUDA2/조건부tick80000/stage320000, 전류 업로드1/생리8/계수2, GPU buffer350432968bytes, actual runner peak RSS 1646166016bytes다. 감독은 이번에 Windows launcher와 실제 owned descendants까지 추적했다. 원본1/retry0/optimizer0/warmup0/전체BANC·물리0, 잔류 owned process0이다. 준비·모의/부모 검산은 별도다.

다음은117개 미달이 신경 반응 부족인지 decoder 전달 제한인지 원자료에서 구분하여 내부 신경 상태/모집 설계를 판단한다. 같은 초기화의 크기나 판정 문턱을 바꿔 재시도하지 않는다. M2·최선E150·UI·실패와 비용을 보존했고 F6 미노출/F1–F8 active·미완료다.

근거: `verification/banc-natural-gait-m3-20260919/e207-threshold-initialization/`, `parent-e207-source-review.json`, `parent-e207-output-review.json`, `parent-e207-supervisor.json`.
