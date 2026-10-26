# E204 결과: 탐색 진행 회복, 명령 형상 개선은 미달

변수 단위와 선 탐색 한도를 조정한 실행은 40 iteration/실제 CUDA 56평가를 완료했다. 원래 E203의 iteration0 정체는 해소했지만 최대 iteration에 도달했고 수렴을 입증하지 못했다. 명령 RMS는 0.056686385350 → 0.056582081411rad, 개선 0.184002%<20%다. 실행 `valid_bounded_search_no_proposal`, 후보 미채택이다.

선택 평가55의 전체 velocity RMS 1.944316128169, 각 다리 command/velocity는 모두 비악화다. 초기 command RMS/max 0.017827011514/0.039844276216는 .018/.04 안이며 tendon RMS/max .001363757198/.013190797934도 통과했다. 48 LTM MN은 E202와 exact다. 이 조건들이 명령 개선 미달을 상쇄하지 않는다.

부모 FIRST에는 결함이 없어 Sol 첫 수정이 필요하지 않았다. 부모 SECOND의 독립 loop-decoder chain-rule 오차≤1.92e-11, 생성 시험3개/Ruff를 통과했고 원본 초기 E202 네 배열 exact/E203 목적함수·기울기 복원, 56평가의 독립 직접 지표·scale/gradient·선택·decoder 검산도 통과했다. 실제 전체BANC/물리0, 조건부2240000tick/8960000stage, 원본process1/optimizer1/retry0이다.

GPU 버퍼350432968bytes, current 전송1, kernel 126.419262500초, readback 0.691851100초다. 감독 141.688625800001초, 누적 5285.855097999705/5444초, 잔여 158.144902000295초다. 준비/생성 검증은 별도이며 최선 E150/M2/UI/F6를 유지한다. F1–F8은 active·미완료다.

## 다음 선택의 근거

전체 merit 감소 7.080193570 중 실제 command 항 감소는 0.005748299(0.081188%)에 불과했다. 나머지는 대부분 원래 판정 한도보다 .5% 엄격한 내부 여유 벌점을 줄인 것이다. 최종 gradient 최대 4307.595740341, 계수 최대 0.002438243, 범위끝 계수0이며 optimizer는 수렴 종료가 아니다. 이를 반응 모델의 전역 불가능성으로 쓰지 않는다.

기존 E89의 84위상 비음수 해부학 inverse RMS .001289183rad 증거를 재사용한다. 해당 moment/rest/pool/membership는 현재 E201과 byte-equivalent 값이다. 분포가 달라 현재의 조건부 합격 증거는 아니지만, 동일 행렬의 자유 활성 용량을 다시 NNLS로 검사하기보다 실제 신경 생성/적합을 다룰 근거다. 이전 E200의 침묵12근육만 추가하는 가족과도 다르다.

다음 후보 설계는 (1) 학습 벌점을 실제 고정 판정 한도에 맞추고, (2) 저장된 실제 국소 미분과 decoder의 command 민감도로 계수별 단위를 정하는 제한된 계속 학습이다. 보호 판정 기준과 과거 실패를 완화하거나 성공으로 바꾸지 않는다. 새 세포 상태나 더 큰 기능군으로 바꾸는 구조 대안은 현재 최적화가 수렴하지 않은 사실과 비교해 다음 결정에 남긴다. 실제 보행·생물학적 유효성은 여전히 미검증이다.

근거는 `e204-scaled-local-response-fit/`, `parent-e204-source-review.json`, `parent-e204-output-review.json`, `parent-e204-direction-review.json`, `parent-e204-supervisor.json`이다.
