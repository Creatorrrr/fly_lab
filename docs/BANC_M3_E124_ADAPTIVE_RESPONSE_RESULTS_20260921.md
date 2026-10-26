# E124 결과: CUDA 조건부 응답 구현, 적합 진입 기준 미달

2026-09-21 KST. 전체 M3 목표는 active/unmet이며 F1–F8을 유지한다.

부모 설계 → Sol/xhigh 구현 → 부모 첫 리뷰 → 같은 Sol 첫 수정 → 부모 두 번째 리뷰 → 부모 잔여 직접 수정 순서로 진행했다. 신규 solver/runner/tests만 개발했고 기존 M2와 실제 E121 보행 후보는 변경하지 않았다. 새 Sol 생성은 thread limit으로 거부되어 기존 Sol/xhigh 작업자를 독립 E124 개발에 재사용했다.

## 구현과 검증

고정 전류·초기 상태·생리·근육 매핑 13개 배열, 19,147,024 bytes를 GPU에 한 번 전송한다. 매 평가에서391 MN gain만3,128 bytes 전송한다. dedicated CUDA stream에서 MN r/z와 gain 감도를 함께 RK4 적분하고, pool/근육 EMA/관절 요청/힘줄도 CUDA에서 계산한다. CPU fallback이나 실제 발화율 잔차를 더한 응답 보정은 없다.

첫 리뷰는 입력 재전송/CPU decoder, 동기화 후 전송 실패의 누락된 완료 횟수, ledger 저장 실패의 raw 손실, 입력 alias/음수 초기 상태, 소스 snapshot 누락을 찾았다. 같은 Sol이 수정했다. 두 번째 리뷰에서 부모는 decoder 중간 출력 손실, 준비 계산 후 오류의 완료 횟수 손실, null stream 동기화, 출력 shape/dtype 검사 누락을 재현하고 직접 수정했다. 모든 실패 자료는 보존했다.

최종 합성26 tests PASS, Ruff/문법 검사 PASS. 테스트 후에는 로컬 import 순서만 Ruff에 맞춰 정리했다. 수치/runner 소스는 테스트 snapshot과 동일하다. 부모 최종 테스트: CPU12회/1080 context ticks(.270초), CUDA 정상9회/880ticks(.220초), 완료 후 실패3회/60ticks(.015초), 동기화 실패1회의20요청ticks는 완료 미상, 별도 준비2회/40ticks(.010초). 독립 DOP853 9회/.185 neuron초/18528 함수 평가. 실제 BANC·물리·원본 응답과 구분한다. 부모 첫/두 번째 독립 실패 주입의 CUDA20/40ticks도 별도 기록했다. Sol은 실패를 포함한11개 pytest invocation을 receipt에 남겼다.

## 실제 기록 실험

PID42572, 원본 프로세스1회, retry0, timeout0. E110/E113/E121 실제 전류 각각10초를 gain0/1/1±1e-4로 평가했다. 총4 batch/12조건부응답/120조건부초/480000 context RK4 ticks, GPU decoder24000 context 끝점 갱신이다. 별도 소형 준비20ticks/1decoder update가 있다. 전체 BANC/신체 전진, 적합, LP, F6 노출은0이다.

각 batch의 조건부 solver+decoder+전송은 약 .55–.57초, 합계2.220328초다. 파일 검증·준비·저장 등을 포함한 외부 프로세스는10.319162초, supervisor/manifest까지10.594930초다. 이전 보수적940초를 포함하면950.594930/1065초다. 이 속도는391 MN의 고정 전류 계산에 대한 것이며158706 전체 BANC나 실제 보행의 속도 향상 수치가 아니다.

gain0의 관절 요청 재현은 다음과 같다. 사전 기준은 RMS .001rad/max .01rad다.

| 실제 문맥 | RMS rad | 최대 rad | 기준 |
| --- | ---: | ---: | --- |
| E110 | .000256980 | .018674187 | 최대 오차 미달 |
| E113 | .000179823 | .007589699 | 통과 |
| E121 | .000127725 | .007589699 | 통과 |

세 문맥의 rate/activation/LTM 재현 기준은 통과했다. 하지만 E110 한 문맥의 실패도 남겨야 하므로 전체 fidelity는 **false**다. gain0 adaptation은 모두 실제0과 일치했다. E110 최대 관절 오차는1.075초의 관절2에서 발생했다.

gain1 감도의 중앙차분 검증도 **false**다. rate 최대 차이 .20918613(허용 .02077880), activation .03068741(허용 .00684356), q .000711553(허용 .000187671), tendon .000191033(허용 .0000626542)가 기준을 넘었다. adaptation 감도는 통과했다. 정류 분기 경계와 고정 유한차분 간격의 영향은 가능한 원인이지만 이번 결과만으로 원인을 확정하지 않는다.

uniform gain1은 원래99 시작 구간을 보존하지 않는다. E121의 시작 q RMS .08063862/max .45819820rad, LTM RMS .01125015/max .07220433으로 원래 기준을 넘었다. 이는 진단용 gain1을 실제 보행에 바로 적용할 근거가 없음을 보여준다. 선택된 보행 후보는 없다.

## 독립 리뷰와 다음 단계

부모는 봉인19개 원본 파일과 입력·소스·기존 결과 해시, CPU RK4 첫10구간, SciPy IIR 재표현, 유한차분 통계를 검산했다.144개 검토 항목이 통과했고 **유효한 실패 진단**으로 수용했다. 원본 exit1은 사전 fidelity/FD 기준 미달에 의한 종료이며 구현 예외나 timeout으로 숨기지 않았다. 새 CPU 참조는3문맥/600ticks/.150조건부초, SciPy IIR8batch/48000 context 표본 갱신, 실제 BANC/물리0이다. 부모 리뷰 외부13.278541초(내부 수치·파일검산3.589378초)를 별도로 기록한다.

이 상태로 gain 최적화를 시작하지 않는다. 다음 설계는 두 문제를 분리한다. 먼저 감도 최대 불일치의 MN과 정류 분기를 작은 고정 전류 문제로 확인해, 구현 미분과 유한 크기 perturbation 차이를 구분한다. 동시에 E110의1.075초 오차 주변에서 native CUDA가 계산하는 더 조밀한 MN 전류를 관측할 방법을 설계한다. 기존5ms 관측을 정답 rate 잔차로 보정하거나 E110을 제외하지 않는다. 본 E124 결과를 덮어쓰거나 한도를 완화해 재실행하지 않는다.

그 이후에도 원래99·기존 형태20%와 같은 최종 후보의 F1–F8 전체를 검증해야 한다. E121은 여전히 실제 F3만 통과한 개발 후보이며 자연스러운 보행은 미완료다.

자료 위치: `verification/banc-natural-gait-m3-20260919/e124-adaptive-response/parent-final-review.json`, `original-output-manifest.json`, `metrics.json`, `parent-decoder-and-sensitivity-raw.npz`. 입력/설계는 `e124-frozen-design.json`, 소스 리뷰는 `parent-e124-final-source-review.json`, 비용은 `e124-original-process-wall.json`, `e124-supervisor-final-wall.json`, `parent-e124-final-test-counts.json`, `e124-sol-synthetic-receipt.json`에 있다.
