# E297 — 초기 명령 오차와 관측된 부분 전달의 잔여 변화

2026-09-28. 저장 분석과 root 독립 감사는 통과했다. **E296 후보의 시작 조건 기각과 F1–F8 미완료는 그대로다.** 새로운 신경·신체·CUDA·solver 실행이나 후보 선택은 없다.

## 확인한 차이

원래 E150의 startup99행 관절 오차는 RMS **.01772709268**, peak **.03945779799**로 이미 고정 .018/.04에 가깝다. E296는 .01893939994/.07604739934로 두 기준을 넘었다. 원래42열 중17열에서 peak .04를 넘으며, 후보의 오차가 가장 큰 것은 원래8번 열의 control99이다. 이름이 확인되지 않은 관절을 추정해 붙이지 않았다.

기준→후보 변화는 별도 controls1..100에서 관절 RMS .00630283331/최대 .04778024434였다. 이것은 reference-target 오차와 다른 정의다. 모든99×42 원자료에서 `candidate_error=baseline_error+candidate_delta`, `SSE(candidate)=SSE(baseline)+SSE(delta)+2*sum(baseline_error*candidate_delta)`를 검산했다. 일부 관절의 평균 오차는 줄어도 특정 시점 peak는 한도를 넘는다. 기준을 완화하거나 한 관절만 제외하지 않았다.

| 동일100행 기준→후보 신호 | RMS 변화 | 최대 절대 변화 |
| --- | ---: | ---: |
| core24 rate | .076922515 | .494018555 |
| MN391 rate | .190480632 | 3.171562195 |
| MN391 adaptation | .005468986 | .093294144 |
| 근육96 activation | .147745537 | 1.277923105 |
| 관절42 명령 | .006302833 | .047780244 |
| 힘줄6 명령 | .000006286 | .000056380 |

Core24의 drive/adaptation과 MN391 drive는 byteexact였다. 직접 drive가 같다는 사실로 간접 감각 경로를 배제하지 않는다. 변화 필드의 첫 exact차이는5ms 격자 첫 관측점이며 substep 발생 순서나 인과 선행을 뜻하지 않는다. Activation과 rate는 저장된 모델 단위다.

## 실제 관측 범위와 전달

E150 metadata55source 중 E296 시간 기록에는53개만 있다. 두 DN51845/136232의 계수는 각각117/122개 MN에서 양수 nonzero이므로 빠진 rate를0 또는 baseline으로 채우지 않았다. 실제 시계열 분석은 core24+other29의 공통53개이며, 전체55개는 full checkpoint가 존재하는 control100=.5초에서만 비교했다. 그 endpoint의 DN 둘은 기준/후보 모두199.9993896484375로 exact였고 입력 차이는0이었다. 중간 구간도 같다고 주장하지 않는다.

원래 `source_to_motor` float32(391,55)는 E257의 원래 calibrated CSR 부분행렬과 byteexact다. 원시 해부학 synapse 수치라는 뜻은 아니며 기존 보정 효능을 포함한다. 새 회귀나 계수 학습을 수행하지 않았다.

| 관측된 부분입력 변화 | controls1..100 RMS / max | control100 full55 RMS / max |
| --- | ---: | ---: |
| core24 signed 항 | .13183501 / 3.03029028 | .13362287 / .89295820 |
| other29 trajectory / other31 endpoint signed 항 | 1.18552205 / 29.98245964 | .53463554 / 5.56662318 |
| 공통53 trajectory / 전체55 endpoint signed 항 | 1.19374034 / 29.97981330 | .55494919 / 5.56710041 |

Other29 변화는 음수계수 항 RMS1.18417418에서 컸고 양수계수 항은.08640893이었다. 그러나 음수항 변화는 최솟값−29.98314/최댓값12.68963으로 양방향이며 평균−.06210420이다. 이를 일괄적인 억제 완화라고 부르지 않는다. 원래 core 입력 계수를 보존해도 다른 경로의 활동과 운동 출력은 달라질 수 있다.

이 값은 동일 시점 rate와 고정 계수의 float64 수학합이다. 관측된 부분 전달의 변화이며 전체 수신 전류, CUDA 누산 순서 exact, RK 단계 전류, 실제 반사실 또는 인과 기여율이 아니다. Metadata55 밖의 afferent와 기록하지 않은 원래 premotor 활동도 남는다. 선택된 자원 연결은 premotor→premotor여서 MN 전달 M에 resource를 다시 곱하지 않았다.

## 구현·검토와 비용

Root 설계→실제 gpt-6-sol/high FIRST→root 검토 과정에서 생산자가 result를 output manifest 뒤에 쓰는 기존 순서를 잘못 가정한 호환성 오류를 발견했다. 첫 source·freeze·실패영수증을 보존하고 같은 Sol이 R1 수정했다. 결과 SHA/literal 경로/schema/E296 identity/비용/startup을 별도로 묶으며 raw/final/ledger 검증은 유지했다. Root SECOND의30tests/Ruff/독립 tiny 손계산과 실제 passive gate가 통과했다. Root 잔여 소스 수정은 없었다.

실제 저장 분석 감독5.579648399958387초/peak1011318784bytes, root 독립 raw/CSR/sparse 재계산·startup SSE 감사1.3951799999922514초/process2.1285334초다. 독립 전달 재결합 최대차1.1368683772161603e−13, 모든 셀의44649개 통계를 검산했다. 첫 gate 실패3.9899763초, Sol FIRST+R1 생성10.9905916초, root FIRST/SECOND 생성 감독6.416545399988536/6.3611527000321075초도 별도 보존했다. 모두 원본0이며 원본 누적 **8078.0529733995245/8150초**, 예약0은 변하지 않았다.

R1 freeze SHA `15b7192a629b05a52af0a3936eaad4334e4aa9c180ae21c35e739f66ef82a00d`; source-review SHA `803816a0190dc5c221b4df57c7178512fd14bb040452d1e3c4554b24b4a556d0`이다. 증거는 `verification/banc-natural-gait-m3-20260919/e297-observed-transfer-residual/`, `parent-e297-saved-audit.json`, `parent-e297-signed-saved-summary.json`에 있다. 미사용 F6/M2/UI와 모든 실패를 유지한다.

다음 설계에서는 핵심 경로를 더 많이 제외하기만 하는 방법과, 자원 변화가 정상상태 전달까지 낮추는 구조를 바꾸는 방법을 구분한다. 원래 startup/F1–F8, u/tau/seed를 조정해 이전 후보를 통과시키지 않는다. 다음 실제 후보는 별도 사전 설계와 검토를 거쳐야 한다.
