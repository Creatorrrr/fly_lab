# E199: 96근육 중12개는 저장된 보행 내내 무출력

유효한 저장 진단이다. E150의391 MN 중196개가 무발화였으며, 이를 실제 소유 근육과 연결하면 **12/96근육의 activation이 전체2000경계에서 정확히0**이다. 나머지84근육은 한 번 이상 출력이 있다. 이 발견은 무출력 근육의 존재를 확인하며, 해당 근육을 활성화하면 지속 보행이 좋아진다는 인과 증거는 아니다.

| 창 | 소속 MN 전부 rate0인 근육 | 일부만 rate0 | rate0 MN 없음 | activation 항상0 |
| --- | ---: | ---: | ---: | ---: |
| 전체 ticks1–2000 | 12 | 63 | 21 | 12 |
| 시작 ticks1–99 | 12 | 65 | 19 | 12 |
| 보행 ticks100–2000 | 12 | 63 | 21 | 12 |

세 창의 무출력 근육 명단은 같다. 이12근육의 소속28 MN은 실제 rate와 순간 phi가 모두0이다. 초기 bank/rate/activation은0이고, 모든 소속 rate가0인데 activation은 양수인 모순은 없었다. 지원되는 근육의 출력이 보행에 충분하거나 적절한 위상이라는 주장은 하지 않는다.

| 다리 | 무출력 근육 | 전체96순서의 index |
| --- | --- | --- |
| LF | sternotrochanter extensor | 8 |
| LM | sternal anterior rotator; accessory trochanter flexor | 17,21 |
| LH | pleural remotor and abductor; accessory trochanter flexor; tibia flexor; tarsus levator | 34,37,42,47 |
| RF | sternal adductor | 52 |
| RM | sternal anterior rotator; sternal posterior rotator; sternotrochanter extensor | 65,66,71 |
| RH | pleural remotor and abductor | 82 |

근육 이름과 소유권은 기존 provenance다. `moment`가 표시하는 joint row는 현재 위치 명령 변환의 지원 범위이며 실제 힘 측정이 아니다. LTM6개는 이 무출력 명단에 없다.

## 검토와 실행

actual gpt-6-sol/medium이 진단 도구와5개 합성 시험을 구현했다. 부모 FIRST는 시작 창을 과거 필터 잔류로 잘못 표시하고 후기 창을 초기0 모순으로 표시할 수 있는 설명 오류를 발견했다. 동일 Sol의 첫 수정 후 부모 SECOND,5개 시험/Ruff,별도 비상수 합성25검사를 통과했다. 부모의 잔여 소스 수정은 없다.

원본은 한 번 실행했다. 도구는 검산된 E198 요약을 사용했고, 부모는 E150의 실제 motor_rate에서 각 창의 무발화 소속을 직접 재계산했다. Python 정렬·선형 보간과 math.fsum으로96근육×3창의 통계를 독립 대조해302검사를 통과했다. 감독 PID35228은 exit0으로 종료했고 소유 실행 프로세스는 없다. 이후 요약 출력 명령 한 건의 Python 괄호 오류를 고쳤으며 원본/검산 재실행은 없었다.

- 결과: `verification/banc-natural-gait-m3-20260919/e199-muscle-support/result.json`, SHA `a44bd77e2447af3cef0e1428ac134ad7b8c9f8a954bfbd0537fda7279cc0970e`.
- 검산: `parent-e199-output-review.json`, 구현 고정: `parent-e199-source-review.json` 및 `e199-source-snapshot/`.
- 원본 감독0.3153725000011036초. 누적5031.485320299715/5144초, 남은112.51467970028534초.
- 별도 준비0.02664070000173524초, 부모 합성0.03929819999757456초, 원자료 검산0.22291360000235727초, 기존 요약 해석0.016236100003879983초. 나머지 개발·문헌·리뷰 비용은 미계측이며0이 아니다.

## 해석과 다음 결정

E198의 전류 분위수를 이12근육에 연결한 `parent-e199-silent-support-interpretation.json`에서는 LH tibia flexor5개의 입력 중앙값이 모두 강한 음수다. 현재 기록에서 임의로 모든 MN의 양수 slope를 키우거나 비음수 threshold만 낮춰 필요한 위상 출력을 얻는다는 근거는 없다. 정적 분위수는 새로운 신경 궤적 예측이 아니다.

다음에는12개 무출력 근육이 현재 부족한 명령 방향을 실제로 제공할 수 있는지부터 확인한다. 양수 출력만 더해도 기존 개발용 참조 오차를 줄일 여지가 없다면 이들을 무조건 모집하는 개입은 선택하지 않는다. 여지가 있다면 해당 근육의 기존 입력과 신경 반응식을 대상으로 제한된 학습을 설계한다. [학습 동역학 구조 검토](BANC_M3_FLYGM_STRUCTURAL_DIRECTION_REVIEW_20260924.md)는 그 대안이며 FlyGM 결과를 로컬 BANC 성과로 가져오지 않는다.

새 신경/물리/fit/F6 실행0, M2/UI 변경0, 후보 채택0이다. E150의 최선 성과와 모든 기존 실패, F1–F8 active·미완료 상태를 유지한다.
