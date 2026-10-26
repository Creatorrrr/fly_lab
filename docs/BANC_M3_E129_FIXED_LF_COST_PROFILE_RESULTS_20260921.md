# E129: 고정 LF 평가 비용 계측 결과

2026-09-21 · M3/F1–F8 active/unmet

E128의 공개된 LF 첫 입력(delta/gain=0) 한 점을 기존 계산 그대로 한 번 평가했다. 원본 응답·미분·loss 비교에서 최대 차이는0이며 branch/ID/mapping도 일치했다. 부모의 독립104검사와1633해시 검증이 통과했다. 원본 프로세스PID23704는9.780871초, supervisor9.816794초, 내부7.602772초로 종료했다. 재시도·시간초과는 없다.

| 측정 구간 | 시간 | 의미 |
| --- | ---: | --- |
| evaluate_leg 전체 | 2.225543초 | CPU wall, 초기 geometry 호출과 원자료 저장 포함 |
| engine.solve | 1.997102초 | 위 전체에 포함된 CPU wall |
| joint_forward | 1.774796초 | 해당 CUDA 스트림의 event 구간 |
| edge_bank | 0.209658초 | 해당 CUDA 스트림의 event 구간 |
| muscle_ema | 0.000311초 | decoder 전체가 아닌 EMA 구간 |
| geometry.leg 3회 | 0.029214 / 0.001502 / 0.001559초 | Torch 현재 스트림 구간, 첫 호출 비용 포함 |
| Torch 역전파 1회 | 0.082650초 | Torch 현재 스트림 구간 |
| raw 저장 | 0.017489초 | CPU wall |

중첩 구간을 합산하지 않는다. 전진+edge-bank 구간은1.984454초로 전체 평가의89.167%였다. 전체/원래 LF unique 평균2.123997초 비율은1.047809이며, 사전0.65 기준을 넘었다. H129는 **이 한 점에서 평가 파이프라인과 신경 CUDA 계산이 상당한 비용을 차지한다**는 범위에서 지지된다. 전체 최적화의 정확한 비용 분해나 성능 향상 증명은 아니다. decoder GEMM/elementwise, scalar 검사, event 계측 오버헤드 및 미측정 잔여의 원인별 비용은 확정하지 않았다.

원본은1 LF batch/3 conditional responses/30조건부초/120000 context ticks/8280000 scalar MN ticks/131040000 edge-stage ticks/1 bank/6000 decoder endpoints다. 기하 forward와 backward는 각각5473행이며 알려지지 않은 완료량은0이다. 별도의 synthetic tiny1MN/20ticks/.005조건부초/1bank를 준비에 사용했다. 원본 적합·전체158706 BANC·물리·F6는0이다. 보수적 공유 누적은1404.816794/1530초다.

부모 설계→Sol/high 구현→부모 첫 리뷰→같은 Sol 첫 수정→부모 두 번째 리뷰/마지막 부분 직접 수정 순서를 완료했다. 최종20개의 생성·가짜 입력 테스트와 Ruff가 통과했다. 첫 구현·첫 수정·실패한 테스트 비용과 소스 스냅샷도 보존했다. 계측 도구만 연구용으로 채택하며, E128의 feasible0/reject 및 E121의 F4/F5 미달은 그대로다.

다음 E130은 `joint_forward` 내부의 고정 `expm1(delta)` 반복 계산을 평가당 한 번으로 옮기는 제한된 CUDA 변경이다. 동역학·누적 순서·가중치·감도·목표를 보존하고, 이미 노출된 비영점 LF 입력에서 원본과 수치·시간을 직접 비교하는 사전 설계를 따로 고정한다. 새 보행 후보나 목표 완료를 선언하지 않는다.

원본: `verification/banc-natural-gait-m3-20260919/e129-fixed-lf-profile/`; supervision: `e129-fixed-lf-profile-supervision/`; 부모 검토: `parent-e129-native-output-review.json`.
