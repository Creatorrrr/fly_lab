# E336 저장 명령·신체 전달 진단 결과

저장 자료 진단과 독립 원자료 감사는 유효하게 완료했다. 이 결과로 E335 후보를 채택하거나 자연스러운 생물학적 보행을 입증하지 않는다. E335의 고정 body 판정 rejected와 F1–F8 미완료를 유지한다.

## 확인한 결과와 후속 개발 방향

같은 theta의 실제 감각 폐루프(E335)에서 여섯 다리의 명령 위상은 c100..300 동안 각각 2.0174–2.4811회 전진했다. 모든 다리에서 200 edge가 R≥.5로 유효하며, 다리별 역방향 edge는 5–42개다. 고정 reference 예상 진행은 2.3827600회다. 명령 리듬이 완전히 정지했다는 설명은 이 구간의 저장 값과 맞지 않는다. 위상은 soft global-reference 좌표이며 해부학적 발걸음 수나 자연 보행의 통과 증거가 아니다.

실제 관절의 zero-lag 추적 RMS는 0.001096–0.001979rad이고, 정의된 nonzero 명령 폭 대비 실제 폭 비율은 약0.9770–1.0324다. 이 구간의 관절은 명령을 가깝게 따라간다. 그러나 위치 서보의 추적 성공만으로 근육·힘·접촉이 생물학적으로 타당하다고 결론 내릴 수 없다.

실제 전진은 1.7214386mm, native 하중 가중 slip은2.0461588mm/s, 최장 전진 bout은.21s이며 공식 eligible cycle은 여섯 다리 모두0이다. Neutral은 전진1.6857973mm/slip1.9427288mm/s/bout.21s다. 따라서 다음 설계는 명령의 추가 시간적 진행만을 최적화하기보다 발의 접촉·adhesion·지지 구간 및 몸통 추진과 협응의 관계를 좁혀 검토한다. 이 권고는 관측값에 따른 후속 가설이며 원인을 단독으로 입증한 결론이 아니다.

|다리|명령 위상 진행|역방향 edge|관절 추적 RMS(rad)|무릎 95−5(deg)|native slip(mm/s)|native 지지 비율|경계 adhesion 비율|
|---|---:|---:|---:|---:|---:|---:|---:|
|LF|2.1167|13|0.001723|61.96|1.9850|0.3434|0.0000|
|LM|2.3662|17|0.001221|18.17|1.7188|0.3469|0.0000|
|LH|2.2793|21|0.001831|42.59|2.0844|0.5171|0.0000|
|RF|2.4811|42|0.001096|56.76|2.3998|0.2543|0.0000|
|RM|2.2272|38|0.001684|18.70|1.1982|0.5873|0.0000|
|RH|2.0174|5|0.001979|48.12|3.4979|0.3410|0.0000|

위 표의 명령 위상은201 경계/200 edge, tracking·무릎·adhesion은 c100..299의200경계다. Native 지지는 실제 저장 time mask와 load≥.02BW를 사용한다. Contact transition은 걸음 수가 아니며 adhesion/receptor/native support는 다른 관측량이다.

Held(E334)는 body 없이 같은 theta로 E318 입력을 공급한 신경 명령 대조다. Held/live 명령 RMS 차이는 다리별0.002951–0.005807rad이고, 신경 입력 차이 median은624개 column/전체158706, RMS.0045761이다. 이 차이에는 입력과 이후 상태 변화가 함께 포함되므로 loop gain이나 감각 피드백 실패의 인과 효과로 해석하지 않는다. Tonic은 cut-cue 후에도 기존20포트−5000 및 bilateral DNg100+5000을 유지한다.

두 body의 native 실제 mask는5001..15000/10000표본이며 nominal5000..14999와 끝 두 tick이 다르다. 허용된 floating clock 경계와 기존 evaluator의 저장 실제 time>=.5 &time<1.5 수식을 그대로 재현했다. 표본을 옮기거나 retime하지 않았다.

## 실행·검토 증거와 한계

Root 설계→actual Astra6/max M1–M8 반영→actual GPT6.1Sol/xhigh FIRST→root FIRST(외부 dtype 표기2건 기각)→같은 Sol ONE R1→root SECOND(155검사)→남은 original clock/sink API guard를 root가 수정했다. 최종 exact source Ruff/105 생성 테스트/159 독립 검사 PASS다. Sol 실패 포함12.9055462/90s, root48.5908326/90s이며 실패한 style 검사·root 경로 fixture·프로파일 감시 timeout도 보존했다. Historical 대형 I/O PASS를 재사용했고 최종 전부 재실행했다고 주장하지 않는다.

Original saved analysis1은20.7515529s/40s, owned peak289165312B/2GiB로 성공했다. Independent saved raw audit7141개 모두PASS,9.6821256s/20s, owned peak315822080B다. 새 신경/CUDA/body 실행0이며 original 비용은9513.922589299525/9660에 한 번만 합산했다. 잔여146.0774107004745/예약0이다. 원자료·실패·M2/UI·F6 미사용 상태와 동일후보 전체 F1–F8 계약을 보존했다.

세 trajectory는 독립 생물학적 반복이 아니다. R은 원형 soft-weight 집중도이고 통계적 신뢰도/곡선 근접도가 아니다. Static residual로 fixed-point 안정성을 선언하지 않으며 신체 Jacobian/feedback 인과 효과/자연성은 아직 입증하지 않았다. 다음 원본 실험 전에 현재 세션의 새 설계와 actual Astra 리뷰가 필요하다.

증거: verification/banc-natural-gait-m3-20260919/e336-saved-gait-transfer/, parent-e336-saved-raw-audit.json, parent-e336-original-accounting.json, parent-e336-descriptive-summary.json.
