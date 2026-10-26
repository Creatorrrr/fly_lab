# E340 저장 보행의 협응 기울기 진단 결과

실제 저장 보행 명령에 대한 조건부 출력 기울기 진단과 독립 원자료 감사를 완료했다. Live의 고정 판정은 `successor_output_gradient_dominates`다. 다음 명령 예측 기울기 크기는 협응 기울기의 **22.62184배**이며, 두 항을 계수1로 합친 하강 방향은 협응 손실을 줄이는 방향을 유지한다. 자연스러운 보행, 신경 파라미터 방향의 실현 가능성, 동일 후보 F1–F8은 아직 입증하지 않았다.

## 결과와 해석

각 궤적의 c100 경계부터 c101..300 출력200개를 사용했다. 50개의 4출력 창에서 경계는 별도 immutable copy로 고정하고 창 내부 previous 경로만 미분했다. 기존 E332의 successor S와 모든15쌍 coordination C를 계수1로 사용했다. 새로운 loss나 gain을 추가하거나 최적화하지 않았다.

| 저장 궤적 | S/C 기울기 크기 비율 | C와 S의 cosine | C와 합산 기울기의 cosine | C 단독 대비 1차 협응 변화 비율 | 고정 판정 |
|---|---:|---:|---:|---:|---|
| Neutral | 33.20817 | 0.019320 | 0.049382 | 1.641586 | S 기울기 크기 우세 |
| Held | 22.81292 | −0.084893 | −0.041172 | −0.936652 | 합산 하강이 협응에 반대 |
| Live | 22.62184 | −0.023969 | 0.020238 | 0.457778 | S 기울기 크기 우세 |

Live의 독립 재계산 값은 `||gS||=6.026145507535317`, `||gC||=0.26638618396582636`, `||gS+gC||=6.0256482956446415`다. `dot(gC,gS)=-0.038476947095393524`, `dot(gC,gS+gC)=0.03248465191248157`이므로 현재 출력 공간에서 작은 `−(gS+gC)` 변화는 C를 감소시키는 방향이다. `D_C=-0.03248465191248157`이며, 표의 마지막 수치 열은 `dot(gC,gS+gC)/||gC||²`다. 동일 학습률의 출력 기울기 변화에 대한 국소 수치이고, 동일 길이의 정규화된 step이나 Adam/실제 신경 가중치 변화의 효과와 같지 않다.

Live S/C/합산 loss는 각각 `6.884383749606792 / 0.8140994660882208 / 7.698483215695013`다. 이 loss 크기만으로 기울기 우세를 판정하지 않았다. Primary CPU float64 autograd와 별도 analytic의 수치·고정 판정은 일치했다. Exact-zero 및 사전 precision 부족 분기는 발생하지 않았다.

Held/live는 입력과 이후 상태가 함께 다른 궤적이다. 두 판정의 차이는 감각 피드백의 단독 인과 효과를 입증하지 않는다. Gaussian global-reference 좌표와 조건부 출력 미분은 실제 발걸음 위상, full-horizon 신경 미분, quantized native 프로그램의 미분, 신체 접촉·힘 또는 생물학적 기전의 증거가 아니다. E333의 기존 deadline 무효와 E335 body 기각도 유지한다.

다음 질문은 **현재 해부학적 효능 파라미터를 통해 협응 방향을 실제로 전달할 수 있는가**다. 출력 기울기의 크기 우세만으로 학습 계수를 올리거나 gain 탐색을 시작하지 않는다. 기존 truncated neural VJP의 범위와 실제 f32 유한 변화의 일치 여부를 구분한 별도 제한 진단을 먼저 설계한다.

## 구현과 독립 검증

현재 세션 설계 → 실제 `gpt-6-astra/max`의 M1–M6 리뷰 → 현재 세션 반영 → 실제 `gpt-6.1-sol/xhigh` FIRST 구현 → ROOT FIRST → 같은 Sol ONE R1 → ROOT SECOND를 수행했다. FIRST에서 실제 실행 상한 및 PID 종료 기록의 모순을 받아들이는 B1을 synthetic metadata로 재현해 거절했다. 실제 자원 초과나 가짜 프로세스를 실행하지 않았다. R1은 실행 시간/RSS/PID evidence 검사만 강화했으며 과학 함수15개·상수9개의 AST는 FIRST와 동일하다.

현재 소스는 110개 관련 생성 테스트/Ruff PASS다. 13개는 역사적 대형 worker/IO11개와 predecessor metadata2개로 명시적으로 제외했고 skip0이다. 이를 현재 전체 대형 I/O 재실행으로 표현하지 않는다. ROOT는 현재 소스의 실제 생성 production worker를 실행한 뒤 원자료 감사와 동일 SHA의 독립 감사 CLI를 직접 감독했다. 정상1개 PASS, wrong detach/scale/row/pair/gradient/result6개는 해당 수치 또는 최종 분류의 독립 mismatch로 exit1이었다. FD/scalar/resource208개와 B1 metadata19사례도 PASS다. 생성 정상 감사는 939개 보고 그룹/3789개 실제 검사이며, ROOT가 production gradient/classifier를 expected로 사용하지 않았다.

ROOT SECOND 이후 product 소스의 추가 수정 필요0이다. ROOT의 준비 코드에서는 종료된 PID 번호가 새 프로세스에 재사용된 실제 사례를 확인해 직후 종료 관측 및 새 process create time으로 구분했다. 최초 metadata 등록 실패와 그에 따른 missing-review 준비 실패도 보존했다. 해당 두 행정 작업에서는 원자료/예약0이며, 비소유 프로세스 종료0이다.

원자료 분석1회 및 같은 독립 감사 CLI의 원자료 감사1회가 성공했다. 실제 선택값은 reference5개 + neutral/held/live 각4개인 **17개**다. Typed12배열/역할, 원본 대응, immutable boundary, 내부 previous 누적, loss·기울기·norm·dot·precision·분기·창/다리 통계, actual selected-read ledger, pre/post SHA 및 게시물 정합성을 대조했다. 독립 원자료 감사 **938개 보고 그룹/3788개 실제 검사 PASS**다. Provisional result/receipt/stdout/manifest와 외부 두 completion receipt가 모두 유효할 때만 별도 ROOT final decision에서 이 진단을 채택했다.

## 비용·증거·목표 상태

Sol FIRST+R1 실패 포함 `44.460656299997936/60초`, 보존 `33,015,623/33,554,432B`다. ROOT 감독 생성 검증은 실패 포함 `23.820202099996095/60초`이며 그 직후 보존 `20,342,019/33,554,432B`다. Nested 감사 CLI 비용은 ROOT 외부 감독 시간에 포함했으며 중복 계상하지 않았다. 행정 및 도구 wall은 별도 비영 시간으로 기록했다.

원자료 분석 실제 감독 `4.562363599994569/20초`, owned peak `624,173,056B/2GiB`; 원자료 감사 `2.079177899999195/10초`, peak `48,168,960B/2GiB`다. 각 실제 비용을 한 번만 합산한 누적은 **`9573.807361699519/9660초`**, 잔여 **`86.19263830048112초`**, 예약0이다. Original job1/audit1/retry0이고 추가 신경·신체·CUDA·parameter VJP·optimizer0이다. 최종 original 출력·로그·감독·외부 receipts·감사·decision 실제 합산은 **2,223,425B/4MiB**다. 기존 원자료·실패·M2·UI·F6 미사용 상태를 보존했다.

현재 소스 SHA256:

- `tools/analyze_c_banc_m3_coordination_gradient.py`: `25bcc4bd0e6e877e9a398cd02996ae75101b79cf97c1254e09428e60e338d4fa`
- `tests/test_c_banc_coordination_gradient.py`: `bf7613eae9c0c85126ce70f39721c13b91c15c585cd2e1b9392c9b3d5c86de62`
- 동일 ROOT 감사 executable: `a17c6f1df8f0a46ae418756132c47b113b9263101c87c7762c4e1237f94e8155`
- Freeze: `e2f839a4329fd237aeb006e14484340e9fa4c13990209d9378f284369e581b08`
- 원자료 전 ROOT review: `25758a99df7eb016643eb1aec7ee8011fa21115a26c56fe4745916785dc5b168`

실행 근거는 `verification/banc-natural-gait-m3-20260919/parent-e340-final-decision.json`, `parent-e340-audit-report.json`, `parent-e340-reviewed-submission.json`, `parent-e340-current-generated-qualification.json`, `e340-sol-r1-submission.md`, 두 generated 비용 JSON 및 `parent-e340-administrative-failures.json`에 있다. 신경 방향/신체 보행 개선의 후속 설계는 실제 Astra/max 리뷰를 거친다. **후보 채택=false, 동일후보 F1–F8 완료=false, 생물학적 자연 보행 검증=false, 전체 목표=active**다.
