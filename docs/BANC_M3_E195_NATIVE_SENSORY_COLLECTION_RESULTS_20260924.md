# E195: native 속도 감각 전달 대조 결과

2026-09-24. 후보는 **기각**한다. 관측 시점의 정보 손실은 수정됐지만, 지속 전진과 미끄러짐 기준은 개선되지 않았다. 전체 BANC/CUDA 3초 원본 1회는 완료됐고 이후 보고서 JSON 직렬화가 실패했다. 실패 로그·잘린 analysis.json·동결 소스를 보존했으며, 같은 원시 자료로 보고서를 복원하고 독립 검산했다. 원본 재실행은 없다.

## 실행과 결과

부모 설계 → 실제 gpt-6-sol/high 구현 → 부모 FIRST → 같은 Sol 첫 수정 → 부모 SECOND → 부모 잔여 수정 순서다. 최종 합성 시험17개/Ruff/CLI가 통과했다. 직전 5ms의 실제100µs hook/club 비선형 특징만 평균하며 gain18, 20ms 필터, 30ms 경계 LP, 지연, 운동3bank, 신체, 전체 CSR은 고정했다. 포트71개/속도 감각451세포가 대상이다.

전체 graph158706, 실제 CUDA packed, control600/neural12000/physics30000이 완료됐다. 시작 자세·native health·매 구간50개 관측, actor 비간섭, delayed drive 및 첫 두control 신경·운동·접촉 exact, 실제 memory→disk start/final 검사가 통과했다. 부모 검산 11726개와 원본 동결소스114개가 통과했다. 변경 후 전체 모델 미래 복원은 아직 수행하지 않았으며 합성 복원시험과 구별한다.

| 평가0.5–3초 | 기존 E193 | E195 후보 | 판정 |
|---|---:|---:|---|
| 최초 방향 전진(mm) | 4.452523366 | 4.305689357 | -3.298%, 기존90%는 유지 |
| net/path | 0.630837801 | 0.619136849 | 감소0.05 이내 |
| 최장 지속 전진(s) | 0.215 | 0.215 | 0.3초 미달 |
| native 하중 가중 slip(mm/s) | 1.953976190 | 1.978475429 | +1.254%, 악화 |
| 구간 내내 rate=0인 hook/club 세포 | 445/451 | 444/451 | 1개만 감소 |

여섯 무릎95–5폭은 기존90% 이상이며 eligible complete cycles는 모든 다리0이다. 감각 request 차이 RMS1.6046105716, 지연drive 차이 RMS1.6525452220, rate 차이 RMS0.0609731680은 모델 단위다. 신경 차이가 존재하지만 보행 후속 screen은 6개 중 지속 전진·slip 2개 실패다. F3/F4/F5 또는 자연 보행 달성으로 승격하지 않는다.

저장된 경계 자료에서 hook/club drive의 세포별 최대값 중앙값은 각각4.3311/3.1802이며 신경 threshold7.5보다 낮다. hook132/137, club312/314 세포는 평가 경계의 `drive + current - threshold - adaptation`도 한 번도 양수가 아니었다. 이는 다음 전달 계약 검토의 근거이며, 저장하지 않은 신경 substep의 정확한 반응이나 생물학적 역치 측정으로 해석하지 않는다. 신호 이득이나 역치를 보행 성적에 맞춰 즉시 바꾸지 않는다.

## 보고 실패와 보존

원본 runner의 screen 비교에 남은 NumPy bool이 strict JSON 저장에서 TypeError를 일으켰다. 실제600control 결과와 validity/raw는 먼저 저장돼 있었다. 원래 함수에 NumPy scalar를 넣어 오류를 모델 없이 재현했고, 복원 보고서는 잘린 원래 파일의 전체 prefix와 exact다. parent-e195-output-check-v2.py는 별도 scalar schema를 행배열로 취급해 실패했으며 이 기록도 보존했다. v3는 scalar를 전체 exact 비교하도록 고쳤고 수치·판정 조건은 그대로다.

원본4개 소스는 e195-frozen-source/에 당시 해시로 보존했다. 이후 부모가 screen 값을 builtin bool로 변환하고 NumPy scalar 회귀시험을 추가했다. 원본 실패를 성공으로 바꾸지 않으며, 현재 소스 변경은 보고 직렬화 수정이다. 전체 신경/물리 재실행0이다.

## 자원과 다음 위치

원본 감독 114.391253700000초, child90.055976200001초, 재시도0·소유잔류프로세스0. 누적 4882.530568199720/4994초, 잔여 111.469431800280초다. 준비·합성·부모 검산은 별도 기록하며 미계측 비용은0으로 간주하지 않는다. F6 노출0/후보미채택/M2·UI 보존/전체 F1–F8 active·미완료다.

다음 검토는 입력의 표집만이 아니라 수용기 특징→모델 입력 단위→세포 발화 범위의 계약이다. 기존 감각 gain·역치 변경 실험과 생물학적 자료를 먼저 확인하고, 보행 점수와 독립적인 calibration 근거가 있을 때만 새 후보를 설계한다.

증거: [부모 독립 검산](../verification/banc-natural-gait-m3-20260919/parent-e195-output-review-v3.json), [복원 보고](../verification/banc-natural-gait-m3-20260919/parent-e195-analysis-recovered.json), [실패/복원 영수증](../verification/banc-natural-gait-m3-20260919/parent-e195-report-recovery.json), [감독](../verification/banc-natural-gait-m3-20260919/parent-e195-supervisor.json), [경계 역치 검토](../verification/banc-natural-gait-m3-20260919/parent-e195-saved-threshold-review.json).
