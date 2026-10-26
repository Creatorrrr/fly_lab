# E332 연속 시간 학습 목적의 저장 자료 검증 결과

2026-10-01. 새 Gaussian successor·다리 협응 목적은 고정한 reference 식별성, 정적 반례 검사 및 저장 명령 공간의 유한 변화 검사를 통과했다. **BANC 가중치 방향·연속 신경 출력·실제 신체 보행의 검증은 아직 아니다.** 전체 M3 목표는 active이며 같은 후보의 F1–F8, 자연 보행 및 생물학적 기전 입증은 미완료다. E330의 실제 신경 방향 오차38.4139% 기각은 그대로다.

## 확인한 결과

Hard nearest 선택을 모든128 reference anchor의 Gaussian 가중 평균으로 대체했다. Previous command에서 만든 successor target과 여섯 다리15쌍의 원형 좌표가 모두 미분된다. Reference와 위상 표현은 오프라인 학습 목적에만 사용하며 최종 신경 실행에는 외부 CPG·위상·궤적을 넣지 않는다. 고정 온도는 기존 승인 정상화의7배이며 새 온도 탐색은 없었다.

| 저장50창의 명령 공간 비교 | 실제 변화 O | 명령 미분 Q | 상대오차 |
|---|---:|---:|---:|
| Signed sum | .004399494599588624 | .004399470547000598 | .0005467125% |
| 창별 절대합 | .005539686001153416 | .005539722281521492 | .0014611959% |

두 상대오차 모두 사전5% 기준을 통과했다. 둘째 비율의 분자는 창별 `abs(O−Q)` 합이며 두 절대합의 차이가 아니다. 이는 E330에서 저장한 동일한 명령 변화에 **새 목적**이 일관되게 반응했다는 결과다. 이전 목적의 E330 효능 gradient를 새 목적의 gradient로 재사용하지 않았다. 신경망 adjoint가 수정되었다거나 검증되었다고 해석하지 않는다.

| 고정 reference 조건 | 새 training loss |
|---|---:|
| 전진 | .04367615165340654 |
| 정지 | .9957473866882238 |
| 역순 | 3.878336250305545 |
| 다리별 고정 위상 불일치 | 161.11865889873548 |

여섯 다리 각각의 정지·역순 successor margin 및 협응 구별을 모두 통과했다. 미리 정한129개 정적 자세의 **최저** 손실은 .293675670274974, 전진 평균과의 margin은 .24999951862156744로 사전1e−8 기준을 넘었다. 다른 정적 고정점이나 전역 정적 최적해가 없다는 증명은 아니다. Gaussian blur 때문에 전진 reference의 손실도0이 아니며 위상 신뢰도R이 높다는 것만으로 실제 위상이 식별됐다고 보지 않는다.

E318 중립의 새 손실은7.77211086209774, E327의 새 손실은7.709795474203607였다. E327의 실제 신체 기각은 유지한다. 손실 감소를 보행 품질 개선으로 승격하지 않는다.

## 실제 모델과 독립 검증

현재 세션 설계→actual `gpt-6-astra/max` 리뷰·필수3개 반영→actual `gpt-6.1-sol/xhigh` 구현→root FIRST를 완료했다. Sol Ruff/21tests PASS. Root FIRST blocking defect0으로 동일Sol R1은 불필요했고 root production 수정0이다. Root 생성 CPU33/CUDA41/publication4 검사와 별도 독립 수식6검사가 통과했다. Actual RTX4080의 CPU/CUDA 상수 bytes·값·미분을 대조했으며 CUDA를 CPU 테스트로 대체하지 않았다.

Passive gate는178source/1542input bytes를 확인했다. 승인된 곡선·정상화·온도·future anchors·원형 basis의 실제 bytes를 고정하고 CPU saved 분석을 단회 수행했다. 감독 종료 후 root가 production objective/analyzer를 호출하지 않는 순수 NumPy Gaussian·covariance 미분으로 실제54개 raw를 대조했다. **2,946검사 PASS**, 독립 loss 최대오차1.24345e−14, 전체 명령 미분 최대오차1.04361e−14, 성분 미분 최대오차1.04639e−14다. Reference129개 최저점과50창의 각 successor/coordination 성분 및 signed/L1 판정도 독립 재계산했다.

## 비용과 다음 단계

- Sol 생성 CPU 실패 포함6.8403672/90초. Root 생성 CPU4.5711982/90초 및 CUDA2.2816718/60초. Passive freeze5.2880549초.
- Saved CPU 감독12.0014285/120초, 계산·사후 hash11.4645541/100초, tree peak622759936bytes/2GiB, output43980473bytes/128MiB.
- Root saved raw audit 감독2.9237510/120초, peak100577280bytes. Torch·CUDA·신경·신체·optimizer·이전 gtheta 읽기0.
- 생성 CUDA Torch peak17999872bytes. 초기 context를 포함한 전역10ms 표본 증가에1MiB 양자화 여유를 더한 값320864256bytes/1GiB다. 다른 process의 영향이나 표본 사이의 숨은 peak를 배제하는 증명은 아니다. 관측 helper 및 감독의 소유 PID는 모두 종료됐다.
- 원본 신경/신체 실행0, 누적9200.673308899522/9360초·잔여159.326691100478초·예약0을 유지한다. 사용자 총예산을 새로 만들지 않았다. M2/UI/F6·해부학적 provenance 및 기존 실패는 보존한다.

다음 E333은 이 목적의 **새 효능 gradient**를 실제 BANC에서 계산하고 동일한 f32 가중치 변화의 방향·연속 출력·기존 시작 제약을 검증하는 별도 단계다. E332 자격만으로 신체 후보를 넘기지 않는다. 해당 설계도 actual Astra6/max 검토 및 root 반영을 거쳐 actual GPT6.1Sol이 구현한다.

증거는 `verification/banc-natural-gait-m3-20260919/parent-e332-first-source-review.json`, `parent-e332-first-{cpu,cuda,publication}.json`, `e332-smooth-temporal-contract.json`, `e332-smooth-temporal-saved/`, `parent-e332-saved-supervision.json`, `parent-e332-saved-raw-audit.json`, `parent-e332-audit-supervision.json` 및 실제 Astra 리뷰에 있다.
