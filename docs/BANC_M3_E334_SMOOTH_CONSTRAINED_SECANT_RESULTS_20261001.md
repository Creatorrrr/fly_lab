# E334: 시작 제한을 통과한 단일 신경 출력 후보

고정 감각 입력에서 새 후보가 시작 구간의 네 제한을 모두 통과하고, 지속 출력의 smooth 시간 목적을 **1.413613% 개선**했다. 단회 원본 실행은 유효하게 끝났고, 별도 NumPy 원자료 감사 **5,094개 검사**가 통과했다. 실제 신체·접촉·감각 되먹임에서 걷는지는 아직 검증하지 않았다. 전체 자연보행 목표와 동일 후보 F1–F8은 미완료다.

| 실제 원자료 측정 | 결과 | 사전 기준 |
| --- | ---: | ---: |
| 시작 관절 명령 RMS, rad | 0.017730250061 | ≤0.018 통과 |
| 시작 관절 명령 최대, rad | 0.039732036327 | ≤0.04 통과 |
| 시작 tendon RMS | 0.001363757098 | ≤0.009 통과 |
| 시작 tendon 최대 | 0.013190716271 | ≤0.025 통과 |
| 중립 steady200 training loss | 7.772110862098 | 고정 비교값 |
| 후보 steady200 training loss | 7.662243283945 | ≥1e−9 감소 통과 |
| 감소량 | 0.109867578153 | 1.413613% |

사전 secant 중점 α=0.16794657931590495를 한 번만 사용했다. 실제 f32 효능은 8,547개 중 629개 그룹이 바뀌었고, 선택 간선 205,195개의 가중치 비트가 달라졌다. 전체 158,706개 뉴런·11,584,852개 간선의 토폴로지, 비선택·효능1 간선 비트와 선택 부호를 보존했다. 전체 uint32 가중치 palette는 8,040개로 고정 상한 16,384 이내다. 원래 CSR에 효능을 한 번 적용한 후보이며, 후보 CSR에 효능을 다시 적용하지 않는다.

실제 control0에서 기존 decoder와 ordered held drives를 사용해 300 controls/6,000 neural ticks를 연속 실행했다. 외부 상태 재주입·replay·backward·optimizer·신체 실행은 0이며 장치 완료 확인은 1회다. 실제 사용한 r/z/bank/activation/full mask/drives/motor IDs 및 신경·decoder·목적 상수 내용 검증과 원본 입력의 종료 후 SHA 검사를 통과했다. 선형 secant 예측과 실제 신경 출력은 구분한다. 실제 관절 명령 최대 오차는 예측값보다 약 4.71e−6 rad 높았고, 실제 제한으로 판정했다.

Root 설계 → 실제 Astra6/max 리뷰 M1–M7 반영 → 실제 GPT6.1Sol/high 구현 → root FIRST → 같은 Sol R1 → root SECOND 순서로 진행했다. FIRST 독립 CPU99 검사에서 같은 바이트의 잘못된 drives 자료형을 허용하는 결함 1개를 발견했다. Sol이 실제 tensor와 25control host chunk의 dtype·shape 검증을 수정했고, SECOND CPU99/RTX4080 CUDA99 검사와 Sol25 tests/Ruff가 통과했다. 추가 root 생산 수정은 0이다. 생성 신경 fixture는 full BANC/native SSP의 검증을 대신하지 않는다. 원본은 별도의 전체 BANC 실행이다.

Root 감사 도구의 첫 시도는 승인 상수 메타데이터에 `numpy_dtype`와 `layout`이 포함된 형식을 잘못 비교해 5개 검사에서 실패했다. 첫 소스·보고서·로그·5.2131958초 비용을 보존했고, 형식 비교만 수정한 v2가 통과했다. 측정값·허용 오차·guard·후보·원본 실행은 바꾸지 않았다. 감사 재실행은 저장 자료 CPU만 사용했다.

원본 감독은 **25.8276879초 / 60초**, 최대 owned tree RSS 2,775,666,688 bytes였고 두 owned PID의 종료를 확인했다. 원본 비용을 한 번 합산한 누적은 **9,431.526588899522 / 9,660초**, 잔여 **228.4734111004782초**, 예약·활성 실행 0이다. 단계 한도 증가·원본 재시도는 없다. 생성 CPU 비용은 Sol11.4567338/90초, root11.3475781/90초(상속 closure·FIRST 실패·SECOND·passive 포함), root CUDA7.1068003/60초다. CUDA 생성 실험의 Torch peak reserved는 48MiB, context를 포함한 sampled additional은 약333MiB이며, 미포착 peak나 다른 GPU 작업을 제외했다는 주장은 하지 않는다. 저장 감사는 실패 포함10.4332357/30초다.

E333의 deadline 무효와 E330 방향·E327 신체 기각은 유지한다. E334는 단일 forward와 원자료 7개로 작업 구성을 줄였으므로 E333의 205초와 같은 작업량의 CUDA 처리량 비교로 해석하지 않는다. UI/M2/F6 비노출·신경 단독/해부학 계약은 그대로다. 다음은 같은 실제 효능과 CSR을 새 실제 신체 control0에 적용하는 E335 폐루프 검증의 별도 설계·Astra 리뷰다. E334 terminal state는 신체 초기화에 사용하지 않는다.

증거는 `verification/banc-natural-gait-m3-20260919/`의 `e334-smooth-constrained-secant-freeze.json`, `parent-e334-original-source-review.json`, `parent-e334-original-supervision.json`, `e334-smooth-constrained-secant/`, `parent-e334-original-raw-audit-v2.json`, `parent-e334-audit-v2-supervision.json`, `parent-e334-final-decision.json`에 연결된다.
