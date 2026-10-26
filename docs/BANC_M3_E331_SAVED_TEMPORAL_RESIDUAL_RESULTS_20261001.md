# E331 저장 시간 목적 오차 분해 결과

2026-10-01. 전체 BANC 자연 보행 M3 목표는 **active·미완료**다. E330의 실제 방향 오차38.4139%/사전5% 기준 기각, 연속·신체 미실행 및 F1–F8을 그대로 유지한다. 이번 단계는 저장된 50창의 CPU 분석이며 신경·신체·CUDA·optimizer·새 후보 채택은 모두0이다.

## 확인된 결과

50창100,104,...,296 전체에서 원래 neutral/plus/minus 관절 명령, 경계,8547효능 gradient와 실제 f32 효능 차이를 재사용했다. CPU 손실 값은 원래 GPU 관측과 최대7.10695e-17 차이였다. 이는 **손실 값 재현**이며 원래 GPU cotangent/선택 branch의 동일성을 입증하지 않는다.

| 항 | signed sum | 창별 절대합 | 전체 residual 절대합 대비 |
|---|---:|---:|---:|
| 실제 변화 O | .00271382624395613 | .00643020176439716 | — |
| 명령 미분 Q | .00440948988336027 | .00568769943475432 | — |
| 원래 효능 미분 T | .00440655879426144 | .00568942846891910 | — |
| nearest index 고정 반사실 F | .00440985970109189 | .00568806371512594 | — |
| O−T | −.00169273255030531 | .00179856513108818 | 100% |
| O−F | −.00169603345713575 | .00178865064706589 | 99.4488% |
| F−Q | .000000369817731621 | .000000509353823490 | .02832% |
| Q−T | .00000293108909883 | .0000109536895392 | .60902% |

각 항의 절대합은 상쇄 방식이 달라 비중을100%로 합산할 수 없다. O−F는 **neutral의 nearest index를 고정한 효과 전체**이며 정상 인접 선택 이동도 포함한다. 신체·신경 개입의 인과 기여나 비국소 불연속만의 비중이 아니다. Q−T에는 신경/decoder 유한 변화·surrogate·반올림과 아직 대조되지 않은 CPU/GPU objective cotangent/branch/산술 차이가 함께 남는다.

| control | 다리/arm | nearest index | phase 차이 | 명령 차이 L2(rad) | O−F |
|---|---|---|---:|---:|---:|
| 140 / .700s | RF plus | 112→109 | .0241024 | .000119228 | −.000994784589 |
| 208 / 1.040s | LM plus | 74→75 | .00582031 | .000379479 | −.000680117391 |
| 284 / 1.420s | LH plus | 96→97 | .00635855 | .000108393 | .0000447452473 |

최댓값140은 비국소 endpoint 선택 변경이고208/284는 인접 변경이다. 따라서 비국소 변경만 제거하는 조치로 전체 문제 해결을 기대할 근거는 부족하다. previous와next는 내부 명령을 공유하므로 위140/208의 동일 사건이 양쪽에 기록된다. 전체 표기 횟수는 nonlocal2/adjacent17/clamp7/wrap0이며 **독립 신체 사건 수**가 아니다. endpoint 비교로 연속 명령 경로의 불연속 자체를 증명하지 않는다.

관측상 O−F가 우세하고 Q와T는 가까웠다. 다음 설계는 gradient clipping/신경망 재배선보다, 오프라인 학습용 시간 목적의 hard nearest 선택을 연속 표현으로 바꾸고 식별성·실제 방향 검증을 다시 자격화하는 것이 타당하다. 이는 새로운 objective 가설이며 E330 기각을 통과로 변경하는 조치가 아니다. 최종 runtime에 외부 위상·CPG·trajectory를 넣지 않는다.

## 구현·리뷰·독립 검증

현재 세션 설계→actual `gpt-6-astra/max` 리뷰와 필수3개 반영→actual `gpt-6.1-sol/high` 구현을 진행했다. Root FIRST 수학17검사는 통과했지만 terminal disk/return 상태 불일치와 비상 저장 실패의 최초 오류 소실2개를 직접 재현했다. 같은Sol R1은 게시/예외 처리만 수정했다. Root SECOND 독립 수학17/publication3검사가 모두 통과했고 잔여 production 수정은0이었다. 최종 Sol Ruff/18tests PASS. 모델 이름은 실제 실행 모델이며 이전 gpt-6-sol 실행을 소급해 바꾸지 않았다.

사전 passive freeze는175source/1470input bytes를 검증했다. NPZ 숫자 읽기/Torch import0이었다. 고정 SHA 계약으로 CPU saved 분석을 단회 수행하고 감독 종료 후 root가 production analyzer를 호출하지 않는 NumPy 손실·반사실·선택 geometry와 닫힌식 미분을 대조했다. **3374검사 PASS**, 독립 손실 최대오차7.10543e-15, gq 닫힌식 최대오차1.60982e-15이다.8400명령 좌표 중7728개는 ordinary stable derivative 조건을 확인했다. Neutral previous/next의 any-tie 표기193개 및 endpoint/knot 근처는 선택된 program cotangent 비교이며 유일한 수학 도함수로 표현하지 않는다. 최초 root raw-audit helper는 Python 예약어 사용으로 숫자 읽기 전에 실패했다(.1205818초). 실패source/log/supervisor를 보존한 root 수정 v2에서 통과했으며 신경/신체 원본 재실행은0이었다.

## 비용과 보존

- Sol 생성 CPU 실패포함5.5006982/90초. Root 생성5.2263020/90초. Passive freeze2.7329451초.
- Saved CPU 감독7.6409862/120초, 내부 계산·사후 hash7.0939117/100초. Tree peak576843776bytes(상한2GiB), output7494637bytes(상한128MiB).
- Root saved audit 감독은 실패.1205818+통과2.1749895초. Torch/CUDA/model/body/optimizer0.
- 원본 누적9200.673308899522/9360초, 잔여159.326691100478초, 예약/활성원본0을 유지한다. 기존단계9034.148870999521/9060 및 E330 사전배정+300의 이력도 그대로다. 사용자 전체 예산을 추가하지 않았다.
- M2/UI/F6·해부학적 provenance·신경 단독 runtime·동일후보 F1–F8 및 모든 기존 실패를 보존한다. 후보/자연보행/생물학적 기전 입증은false다.

증거는 `verification/banc-natural-gait-m3-20260919/` 아래 `parent-e331-{first-source-review,second-source-review,second-cpu,second-publication-probe,passive-freeze,saved-supervision,saved-raw-audit}.json`, `e331-saved-temporal-residual-contract.json`, `e331-saved-temporal-residual/`의 raw/result/ledger/manifest와 실제 Astra 리뷰에 있다. E331 최종 runner SHA는084ea7dea42e818dc5a43152149cee336e8c83dc80d58339c37878e30e053a52, tests SHA는f244fef882118d8de311aac2f614a429f86cdc8544103fd325ca5c2db52637ed다.
