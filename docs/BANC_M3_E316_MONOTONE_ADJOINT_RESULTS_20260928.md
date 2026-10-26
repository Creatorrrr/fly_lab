# E316: 현재 근육·관절 변환의 CUDA 미분 연결 검증

현재 391 운동뉴런→96 근육→42 관절 변환을 functional PyTorch 계산으로 구현했다. 기존 세 filter bank, 고정 LTM/singleton 경로와 별도의 original activation을 보존한다. 실제 E150의 저장된 10초·2,000control 전체에서 신호와 명령을 재현했다. 새 학습·보행 후보·신체 실행은 없으며 자연 보행 F1–F8은 계속 미완료다.

구현은 [engine](../tools/banc_m3_monotone_adjoint.py)와 [tests](../tests/test_c_banc_monotone_adjoint.py)다. Root 설계 → actual gpt-6-sol/xhigh 구현 → root FIRST → 같은 Sol 첫 수정 → root SECOND를 거쳤다. SECOND 이후 root 잔여 source 수정은 필요하지 않았다. 최종 SHA는 각각 `cf2faee028ba58724a81d504d4fcb11b842c02dc9177784311a86ef3611283c0`, `1abf571f5f532efdc7d668a02ca21d314bcdba31f877e916381b10c393f2d3e6`이다.

첫 검토에서 정상 범위의 수식과 gradient는 통과했다. 다만 유효한 큰 유한 상태에서 motor cut의 합산이 overflow하여 모든 출력에 NaN이 생기는 문제를 재현했다. 같은 Sol의 R1에서 안전한 영 미분 연결, 출력 유효성 검사, 고정 행렬과 벡터의 곱을 적용했다. 두 개의 큰 broadcast 중간 배열은 없앴다. 과거 실패 소스와 NaN 결과, R1의 overflow fixture timestep 오류도 보존했다.

## 검증

- CPU 16검사와 실제 기존 두 adapter의 생성 입력 대조, 독립 scalar/유한차분을 통과했다. CPU1차·2차 감독 합39.8251507초/90초다.
- CUDA 생성 검사와 신경 20tick→decoder→q의 독립 float64 미분 대조를 통과했다. 합성 미분 상대 L2 오차는 `4.81e-7`, decoder 자체 directional FD 상대차는 최대 `5.26e-9`다. 새 SSP/current VJP를 reference 계산에 재사용하지 않았다.
- CUDA hot step에서 tensor CPU/scalar 추출을 금지한 검사도 통과했다. 생성 CUDA 전체10.2644567초/45초다. 생성 500control .5887초는 실행 한도 확인용이며 이전 구현 대비 가속 비율을 주장하지 않는다.

현재 E307 SSP/u0의 adapter identity와 E150의 identity는 모두 `1fd7b86939a6846e01c4837fb5c2fbe24ce8c776e3331c1f75921a376c53b067`이다. E150 main 기록의 완료 부분만 사용했다. E150 전체 실험의 cut 검사 실패·incomplete 판정은 바꾸지 않는다.

## 저장된 실제 10초 재생

MN rate를 입력으로, control0의 실제 bank/original activation에서 시작해 .005초씩2,000회를 계산했다. 정답 관절 명령을 모델 입력에 넣지 않았다. CUDA 출력은500control씩 저장하고, 실제 장치 완료가 확인된 prefix와 파일 저장 완료를 구분했다.

| 항목 | 최대 절대차 | 사전 기준 |
|---|---:|---:|
| 세 filter bank | 2.8421709430e-14 | ≤1e-10 |
| 원래 activation | 1.4210854715e-14 | ≤1e-10 |
| 최종 근육 activation | 8.5265128291e-14 | ≤1e-10 |
| 42 관절 명령 | 4.2743586448e-15 rad | ≤1e-11 |
| 6 힘줄 명령 | 5.5511151231e-17 | ≤1e-11 |

Valid/supported다. Root는 네 원자료 chunk에서 오차를 다시 계산하고 source/input/output SHA, 입력 계보, 시간과 자원을 **183검사**로 독립 확인했다. 이 검증은 기존 decoder의 재현과 생성 입력의 gradient 정합성에 한정된다. 실제 전체 BANC→body의 장기 gradient나 자연 보행 개선을 증명하지 않는다.

실행 장치는 RTX4080, PyTorch2.14.0+cu130/CUDA13.0이었다. 원본 저장 재생1회/재시도0, 새 neural tick/physics/body/fit0이다. 감독 wall5.342142700043041초, 내부wall3.36877초, 추가 GPU peak reserved30MiB, 감독 peak host약1.05GiB로 한도 이내다. 원자료 감사는 .78946초(외부1.19043초), 모델 추가 실행0이다. Sol 계측 CPU19.9738061초/90초이며 agent peak RSS와 일부 읽기 overhead는 계측하지 않았다.

원본 감독 시간을 한 번 합산한 누적은 **8506.463176899651/8560초**, 잔여53.536823100348556초, 예약0, 소유 실행0이다. M2·활성 UI·미사용 F6 및 모든 이전 실패·비용을 보존했다.

## 다음 단계와 한계

[E314/E315 결과](BANC_M3_E314_E315_NATIVE_ADJOINT_RESULTS_20260928.md)의 실제 신경 국소 검증과 이번 decoder 검증을 토대로, 시작 조건을 지키는 한 주기 내 근육 동작 학습을 설계한다. 현재 E150에는 반복이 있으나 전진이 끊기므로 추가 리듬 생성 자체를 목표로 삼지 않는다. 고정 교사 절대 위상·외부 CPG·몸체 보정을 최종 제어에 넣지 않는다.

후속 학습 전에 실제 사용할 시간 구간에서 유한 크기 연결 변화와 미분 예측을 대조해야 한다. E24/E25의 장기 미분 실패와 E29의 조건부 학습 실패는 그대로 남는다. 통과한 동일 후보만 원래 startup .5초→short1.5초→10초/F1–F8 검증에 진입한다. 현재 페이지에는 새 보행 후보를 적용하지 않았다.

사전 문서: [구현 설계](BANC_M3_E316_MONOTONE_ADJOINT_DESIGN_20260928.md), [저장 재생 계약](BANC_M3_E316_RECORDED_DECODER_PROTOCOL_20260928.md). 근거는 `verification/banc-natural-gait-m3-20260919/parent-e316-generated-qualification.json`, `parent-e316-source-review.json`, `e316-recorded-decoder/`, `parent-e316-original-supervisor.json`, `parent-e316-original-raw-audit.json`에 있다.
