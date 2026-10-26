# E216 결과: 여섯 다리 조건부 동적 용량과 전역 proxy 통과

E215 LM 해를 그대로 재사용하고 나머지5다리를 한 번씩 계산했다. 5개 sparse QP가 모두 Solved였고, 여섯 다리 합산은 **기존 strict 전역 proxy를 모두 통과**했다. 원래 시간상수, cap/gate, 10ms 근육 전달과 해부학적 소유권을 바꾸지 않았다. 모든 forcing은 작은 epsilon 확대에 의존하지 않고 실제 [0,S] 안에 있다.

| 조건부 저장 명령 지표 | 기준 E150 | 동적 용량 witness |
|---|---:|---:|
| main 관절 명령 RMS (rad) | .05668638535 | .00207540896 |
| main 명령 속도 RMS (rad/s) | 1.94664326116 | .15208068973 |
| startup 전역 q RMS / 한도 | — | .00842657350 / .018 |
| startup q 최대 / 한도 | — | .03999866771 / .04 |
| startup tendon RMS / 한도 | — | .00136375723 / .009 |
| startup tendon 최대 / 한도 | — | .01319080028 / .025 |

main 명령 RMS 감소는96.3387876%다. 각 다리 q/v도 모두 비악화 조건을 만족했다. LF/LM/LH/RF/RM/RH 명령 RMS는 .00116899447/.00305869197/.00258510431/.00118439743/.00084687043/.00251376976rad다. 합산된 frame은0..1923이며 이후 구간을 채워 넣지 않았다. row0는 기존 held command이고 tendon은 고정 E213 intercept다.

**이 수치는 자유로운 시간별 forcing을 허용한 고정 입력 진단 결과다.** 실제 신경 반응을 학습했거나 신체가 이렇게 움직였다는 결과가 아니다. 본 결과와 E214 고정16기저의 불가능 결과를 함께 보면, 같은 관측 입력/전달 가설에서 출력 상한이나 필터의 대역폭만으로 실패를 설명하기 어렵고 반응 표현을 개선할 근거가 생긴다. 새로운 반응 가족이 실제 BANC feedback에서 안정적일지와 자연 보행은 미검증이다. F1–F8 active/미완료, 후보 채택0/F6미노출, 최선E150/M2/UI 보존.

## 구현·독립 검산

부모 설계→실제 gpt-6-sol/high 구현→부모 FIRST 차단 결함 없음→부모 순수/모의7개 및 Ruff 통과 후 실행했다. 불필요한 수정 회전은 없었다. E215의 builder/solver/검증된 실제 생성 예제를 SHA로 재사용하여 새로운 실제 생성 solver는0회다. 공유 elapsed 시계로 150초 내부 예산/90초 남은 시간 조건을 각 호출 전후에 적용했으며 LM 새 호출0, 나머지5개 각각1회, retry0이다.

부모는 다리별 원자료와 원래 moment/pool/B에서 sparse 문제를 다시 만들고 KKT·두 단계 필터·전역 지표를 검산했다. 물리 map 최대차4.54747e-13, 새5다리 KKT stationarity최대5.07320e-13/equality최대1.53735e-11, primal/dual cone 위반0이다. 합산 q 차8.88178e-16rad. 기존 LM도 고정된 독립검산 증거와 raw로 연결했다. source/manifest/M2 SHA와 소유 프로세스 종료를 확인했다.

근거: verification/banc-natural-gait-m3-20260919/e216-six-leg-dynamic-capacity/의 leg-* 원자료, combined-conditional.npz, global-metrics.json 및 parent-e216-output-review.json. 구현은 tools/analyze_c_banc_m3_six_leg_capacity.py와 tests/test_c_banc_six_leg_capacity.py다.

## 자원과 후속 근거

원본 프로세스1회7.480250199994771초, 새 solver5회 합2.878723699999682초/constructor합.4691734000080032초다. 누적**5696.177954499701/6004초**, 잔여307.8220455002993초. peak process930590720bytes/tree942866432bytes, 출력77416899bytes, 모든 소유 프로세스 종료. 준비.04074630000104662초, 부모 cheap wrapper1.4301742000025115초, raw 독립검산9.356992799999716초는 별도다. Sol 순수검사.85초도 별도 보고값이며 미측정 설계/agent 비용은0으로 세지 않는다.

별도 저장 자료 분석 parent-e215-fixed-basis-contrast-audit.json은 LM witness 한 개의 인접 forcing 변화를 기존 고정 beta 기저의 가능한 범위와 비교했다. startup(end2..99)2건, 경계end100 9건, 실제 main쌍(end101..1923)1004건이 범위를 벗어났다. 가장 큰 .03485 위반은 end101이었다. 범위 계산의 S 재구성 오차3.33e-16, 부모 저장분석.2257553초/새solver0이다. 이 witness가 유일한 해는 아니므로 모든 feasible 해가 같은 위반을 필요로 한다는 증거로 쓰지 않는다.

다음 개선은 출력 상한/시간상수를 키우는 대신, 실제 신경 입력과 인과적 기억에 대한 반응 해상도를 높이는 별도 가족을 설계하는 것이다. 학습용 기준과 F1–F8는 그대로이며 임의 forcing을 runtime 제어기로 채택하지 않는다.
