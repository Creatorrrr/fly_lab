# E207: 영 기울기 운동뉴런의 작은 문턱 통과 초기화

부모 사전 설계이며 F1–F8 자연 보행 목표는 active·미완료다. E203–E205의 벌점 학습, E206의 명시적 제약 접선은 20% 명령 개선에 미달했다. 별도 저장 감사에서 비LTM 자유 MN343개 중178개가 rate/adaptation/두 민감도·7계수 모두0으로 남았고, 이들 중 허용된 응답식 범위에서 초기 단계 반응을 만들 수 없는 세포는 없었다. E206의 선형 해는 이 영 기울기를 넘지 못하고 일부 음수 rate도 예측했다. 국소 solver/반복 수만 바꾸는 후속 대신 같은 신경 반응식의 초기 학습점을 바꿔 본다.

H207은 초기 순간 목표 .01Hz의 작은 문턱 통과가 이178개 중 적어도 절반에 늦은 구간의 측정 가능한 발화와 명령 민감도를 만든다는 것이다. 경쟁 설명은 초기 과도응답 뒤 다시 억제되거나, 민감도는 생겨도 명령/출발 품질을 악화시킨다는 것이다. 이 공학적 초기화 선별 조건은 자연 보행 또는 생물학적 발화 기준이 아니다. 참이면 새로 열린 자유도를 제한된 적합에서 검토하고, 거짓이면 초기 문턱만 넘기는 접근을 충분한 학습 범위로 부르지 않는다. 이번에 fit·신체 시험·채택은 없다.

## 변경 및 고정

기준은 E205 선택43의 391×7 weights다. E201의 실제160000 native RK 단계 전류와 생리, 매 평가의 원래 E150 초기 r/z, E202 반응식/float64 조건부 RK4 및 해석적 민감도, E150 decoder, 목표·출발99·시간축·343자유/48LTM 마스크를 유지한다. 앞선 E206의 선형 예측이나 candidate 계수는 쓰지 않는다. F6/UI/M2/최선E150와 모든 실패·비용을 보존한다.

178개 마스크는 자유 MN 중 저장된 E205의 rate/adaptation/rate_sensitivity/adaptation_sensitivity 전 행·기저와 weights가 모두 정확히0인 교집합으로 재구성한다. 고정 snapshot/free/motor ID와 부모 감사의178개 열이 exact여야 한다. E205 preconditioner의 해당 D 및 initial_weights도0임을 확인한다.

각 해당 MN의 첫 실제 RK 입력과 원래 초기 상태로 `s0=coefficient*(I_stage0-threshold-initial_adaptation)`를 계산한다. `rho=.01Hz`, `v=2*(atanh(rho/cap)-s0)`, `w_seed_bias=atanh(v)`다. |v|<1, |w_seed_bias|<=1, cap>rho, 모든 값 유한을 확인하고 실패 시 자르지 않고 종료한다. 기존0인 일곱 계수 중 상수 기저 하나만 이 값으로 바꾸며 나머지6개는0, 나머지213MN의 모든 계수는 exact 보존한다. 이로써 첫 단계의 순간 `phi=cap*max(tanh(s0+.5*tanh(w_seed_bias)),0)`가 .01Hz임을 atol1e-10/rtol1e-10로 확인한다. 이는 실제 초기 rate를 .01로 덮어쓰는 조작이 아니다. 실제 r/z는 원래 E150 값으로 시작하고 동일 RK4로 진행한다.

## 두 번의 실제 조건부 CUDA 계산

검증된 E203 `PersistentLocalResponse`, `validate_response`, `DecoderAdjoint`, `load_targets`, `direct_metrics`, `feasible` 및 E202 loader/IO를 읽기 전용 재사용한다. 새 커널/역전파/외부 신호/해부학 decoder 변경은 없다. 디바이스 입력/생리는 한 번만 올리고 1) 기존 E205 weights, 2) 정확히 한 seed weights를 순서대로 계산한다. 초기 r/z/민감도는 매 호출 원래 초기값/0이다. 추가 warmup/fit/선 탐색/원본 retry는0이다.

첫 호출의 네 response 배열은 E205 선택과 dtype/shape/bytes exact, decoder는 atol2e-12/rtol1e-12여야 한다. 실패하면 seed 호출0으로 끝낸다. 두 호출 모두 네 response 배열을 finite/dtype/shape 검증 전에 가능한 raw 그대로 저장한다. 바뀌지 않은213MN의 네 신경 배열은 seed에서도 기준과 bit exact이고48LTM 및 tendon 전체도 exact여야 한다. 호출 요청·동기화·완료불명·전송·부분 오류 비용을 원래 PersistentLocalResponse stats로 기록한다.

선택 원자료/첫 단계 식/마스크/seed weights, 두 response와 decoder, 초기화 후 D/h/scale, 비교 metrics와 판정을 저장한다. `command_diagonal_scale`로 seed 상태의 D/h/scale을 계산한다. h는 기존 helper의 regularization `2e-4/2401`을 포함하므로 이를 뺀 `h_data`를 별도로 저장한다. E206의 선형 command를 새로운 실제 신경 결과로 바꾸어 쓰지 않는다.

178개 각각의 늦은 target1824행에서 rate RMS와 max_k(h_data)를 계산한다. **rate RMS>=1e-4Hz AND max_k(h_data)>=1e-6**인 세포를 지속 학습 신호가 열린 것으로 세고89개 이상이면 H207 supported, 아니면 bounded test not supported다. 이 수치는 subnormal 잔류/초기 한 표본의 양성만 세는 것을 피하기 위해 사전에 정한 공학적 선별 기준이다. rate RMS/최대값과 h_data의 전체 값·분위수, 기준의0 여부도 공개한다. 순간 식 검증, 이 모집 판정, 기존20%/다리별 비악화/출발 조건 판정, 자연 보행 미완료를 서로 구분한다. 기존 후보 기준을 이번 선별 기준으로 교체하지 않는다. 실제 command가 악화되면 그대로 기록하며 초기화 성공을 보행 개선이라고 부르지 않는다.

## 구현·리뷰·자원

실제 gpt-6-sol/high가 새 `tools/verify_c_banc_m3_threshold_initialization.py`, `tests/test_c_banc_threshold_initialization.py`만 구현한다. 동일 커널을 재사용하지만 두 GPU 경로의 기준 복원/불변 배열/원자료·부분 실패 처리가 결합되어 high를 사용한다. 부모 FIRST→필요시 같은Sol 첫수정→부모 SECOND/잔여수정 뒤 원본1회다. 기존 소스/gate를 바꾸거나 monkeypatch하지 않는다. CLI와 부모 source review는 E205의 독립 frozen gate 형식을 쓴다.

생성 검사는 bias 역식·범위 오류·수정 마스크/초기 상태 불변, 기준 복원 실패 시 두 번째 호출 차단, 비유한 raw 보존, seed 불변 MN 위반 거부와 실제 호출 한도2를 mock으로 확인한다. 변경 없는 실제 CUDA 커널 테스트는 반복하지 않는다. 부모는 원본 raw로 식·마스크·213MN/LTM 불변·decoder·민감도 선별·실제 metrics/비용을 독립 재계산하며 추가 신경 계산은 하지 않는다.

원본 process1/CPU1/GPU1/실제 조건부 호출2(각40,000tick/160,000stage)/fit0/retry0/warmup0다. 정상 총80,000조건부tick/320,000stage/20조건부초이며 전체158706 BANC/신체0이다. 내부35/외부40/감독45초, GPU buffer768MiB/출력256MiB/실행 프로세스 및 owned child peak RSS4GiB/free disk4GiB다. 이전 누적5412.115466699706/5564초의 잔여151.884533300294초에서45초를 예약하며 배정을 늘리지 않는다. Windows venv launcher와 실제 owned child 메모리/종료를 함께 추적한다. 준비·생성·부모 검산 비용은 별도이며 미측정은0이 아니다. 오류/시간·메모리 한도/출력 실패는 incomplete 또는invalid이고 원 오류·raw·비용을 남기며 자동 재시도하지 않는다.
