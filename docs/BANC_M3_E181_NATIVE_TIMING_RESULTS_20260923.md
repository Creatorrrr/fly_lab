# E181 실제 보행 목적 최적화: 구현 검증, 원본 탐색은 자원 중단

**최적화 도구와 기준 재현은 확인했으나, 후보의 보행 개선 여부는 판정하지 못했다.** 같은 모델의 시간응답만 바꾸는 bounded coordinate optimizer를 구현했다. 원본 탐색은 왼쪽 중간다리 첫 후보가 시간 한도에 걸려 중단됐고 후보는 채택하지 않았다. 자연 보행 및 원래 F1–F8은 미완료다.

## 실제로 확인한 내용

각 MN–근육의 low/high 모집 계수 합을 보존하면서 기존5/10/40ms 성분을 다리별로 재분배한다. 원래 BANC 노드·연결·동역학과 몸체·moment/rest·MN membership·fixed19는 유지한다. 실제 Thorax 위치로 계산한 전진 속도 부족 J를 최소화하고, 원래 baseline 대비 전진/경로 효율95% 이상·slip 비증가·초기 보호를 선택의 필수 제약으로 구현했다.

부모가 설계하고 actual gpt-6-sol/high가 최초 구현과 첫 수정을 수행했다. FIRST에서 실제 Windows venv redirector 때문에 정상 child도 거부하는 문제를 모델 없이 재현했다. 같은 Sol이 조상 process 확인·소유 후손 종료·로그 및 불명 비용 회계를 수정했다. 부모 SECOND에서 독립 scalar/선택/목적함수46개, 합성 pytest19개, Ruff와 format check가 통과했다. 형식적인 추가 수정은 하지 않았다.

| 실행 | 확인된 결과 | 판정 |
| --- | --- | --- |
| theta=0 새 runner, 전체158706 BANC/CUDA + FlyBody3초 | 601 control와30000 native를 기존 E180 baseline과 비교한 부모7240검사 통과. J=0.1991308666mm/s | 유효한 기준 재현 |
| LF의 +/- 및 사전4단계 startup backtracking | 8개 제안 모두 기존 max명령오차0.04rad 초과. 가장 작은 음의 step도0.0402326456rad | 이 사전 step 집합에서 native 후보0개. LF 시간응답 가족 전체 기각은 아님 |
| LM plus theta=.125 | 저장/실제startup 통과. 283control=1.415모델초 후 resource guard | 완전3초 결과 없음. J/보행 개선 판정 불가 |
| LM minus theta=-.0625 | 저장startup 통과 | 앞선 기술 중단으로 미실행 |
| LH/RF/RM/RH | 원본 탐색0 | 미실행 |

부모의 별도 양수 branch 계산으로 기록된 startup 제안11개와 zero 기준을 검산해12개가 통과했다. 중단된 LM plus의43개 원자료 검산에서는 시작 전체 상태·실제 기울기 변화·신경→bank→96활성→42명령·시계가 일치했다. bank 오차0, 활성8.53e-14, 관절 명령4.88e-15rad였다. 기록된 구간은 수치/몸통 접촉 건강성이 유지됐지만, 이를3초 또는10초 보행 통과로 해석하지 않는다.

## 시간 초과와 공유 자원 관측

E181 LM driver는14:37:07.685961 UTC에 시작했고, 같은 로컬 GPU를 쓰는 별도 이미지 생성 queue client는14:37:08.417602 UTC에 시작했다.14:41:07 조회에서도 그 client와 queue running1개가 확인됐다. 해당 사용자 작업에는 개입하지 않았다. 기준 구성11.328초와 달리 후보 구성은31.176초였으며, 후보는 process91.719초/1.415모델초에서 내부 guard를 지켰다.

동시 작업과 구성 지연은 확인된 사실이다. 자원 경합이 느려진 원인이라는 해석은 추정이며 실행 중 GPU 이용률을 수집하지 않아 확정할 수 없다. 후보의 과학적 가설을 반증하는 결과도 아니다. 원본 한도와 재시도0 규칙에 따라 해당 campaign을 종료했고, 시간 한도를 늘려 실패를 지우거나 이후 축을 실행하지 않았다. native/driver process는 종료됐으며 소유 후손은 남아 있지 않았다.

## 자원와 증거

원본 모델 process2회(완료baseline1, 불완전후보1), 자동재시도0. 합계 scientific4.415초,17660neural ticks,44150physics steps다. 별도 setup480실행 steps/240capture 기록 steps, 두 완료된 몸체 구성의 초기 settling각2000단계는 기존 소스 근거로 구분했다.

실제 native 감독 시간은 baseline53.514335초, 후보95.737312초다. 실패 후보 비용을110초로 보수 예약한 native 누적163.514335초와 startup0.502714초를 기존3533.525206초에 더한 보수 원본 누적은3697.542255/4994초다. 원래 사용량·실패·보수 여유를 유지하며 부모 생성/해시/검산 및 driver 전체 시간은 별도 기록한다. 입력92/source92/보호3305/E180baseline172/소유source2의 보존 확인은 parent-e181-final-review.json에 기록한다.

구현은 tools/optimize_c_banc_m3_native_timing.py와 tests/test_c_banc_native_timing.py다. 설계·freeze·실제 모델/effort·두 소스 스냅샷·FIRST/SECOND 리뷰·원본 로그·중단 checkpoint는 verification/banc-natural-gait-m3-20260919/에 있다. 핵심 결과는 parent-e181-baseline-review.json, e181-native-timing/axis-lf/parent-axis-review.json, parent-e181-startup-independent.json, parent-e181-partial-review.json, parent-e181-resource-observation.json이다.

다음 방향은 docs/BANC_M3_AFTER_E181_RESOURCE_RECOVERY_DRAFT_20260923.md다. 동일 후보와 비교 조건을 보존한 자원 준비 확인과 별도 사전 계약이 필요하다. 이 초안 자체가 재실행은 아니다. M2/UI/F6를 보존했고 goal은 active다.
