# E252 저장 폐루프 초기 오차 위치 결과 — 2026-09-25

E251의 초기 회귀는 왼쪽 뒷다리 명령에 집중된다. 고정된 기술적 국소화 H252는 지지됐다. 새 q 최대오차 위반37개가 모두 LH이고, 양의 초기 오차제곱 증가 중98.8844970898%가 LH였다. 다른 다리에도 변화가 있으며 특히 .5초 이후에는 전신으로 커진다. LH에만 영향이 있다고 해석하지 않는다.

첫 새 위반은 control64(0.320s)의 물리 관절17, LH femur_roll이다. 기준 오차−.037885089124rad에서 후보−.040726392503rad로 커졌다. 같은 관절의 최대 오차는 control84(0.420s)에서−.035533913084→−.050866351515rad였다. 두 지점의 후보−기준 명령 차이−.002841303379/−.015332438431rad는 모두 muscle41 `femur_reductor_muscle`의 signed moment 항에서 나온다. 최대 지점 activation은13.931177681328→14.697799602873이다. 이는 고정 decoder의 정확한 산술적 기여이며, 한 신경 연결의 고유 인과 효과를 증명하지 않는다.

startup LH 명령 차이 RMS .004860401240rad이고 다른 다리들은.000006716330–.000029883782rad다. LH의 signed SSE 변화+.007622464627과 양의+.038702369133/음의−.031079904507을 모두 보존했다. 개선 항을 숨기거나 양의 항만 전체 변화처럼 보고하지 않았다. 기존 strict startup/F1–F8/H251 실패는 그대로다.

선택25간선의 직접 입력 복원은 startup RMS 기준 .000712984235→.707862511636 모델 단위다. 그 차이의 intervention 항 RMS .712271267689와 feedback 항 .010328289236을 별도로 저장했다. 이는 source endpoint와25간선만의 재구성이며 전체 재귀 입력/실측 생리 단위가 아니다. source/bridge/MN·근육·명령·몸의 첫 endpoint 차이는 control6(30ms), 반환 감각 drive 차이는 이후 적용 endpoint8(40ms)이다. control6 내부 첫 native 차이는25.1ms다. encode→neural→decode→50physics 순서상 endpoint보다 먼저인 native 시간은 오류가 아니며 같은 control 안의 neural 원인 순서를 추론하지 않는다.

실제 gpt-6-sol/medium가 구현하고 FIRST에서 선택 JSON wrapper, 관측/적용 시점 구분, SSE cross/제곱 항 보존, native floating 경계 창 선택을 첫 수정했다. 부모 SECOND12생성검사/Ruff 및 독립 결함회귀4개가 통과했다. 부모 제품 잔여 수정은0이다. 고정 소스로 원본1회/retry0을 실행했고, 독립 원자료·ID·clock·해부학·분해·판정739검사가 통과했다. native slip의 unavailable NaN과 양쪽 유효 비교 분모를 유지했다.

원본 CPU 감독.871517100008초, 누적6854.113728099637/6933초·잔여78.886271900363초·증액0. 출력4019452B, peak tree136675328B/최대 프로세스123912192B. 원본 소유 프로세스0, model/CUDA/neural/physics/fit/solver0. 부모 감사.893072699997초(프로세스1.088392099991초)는 별도다. 부모 생성 수학검사의 NumPy bool JSON 영수증 기록 실패와 복구도 보존했고 원본 실행과 구분했다.

원자료 `verification/banc-natural-gait-m3-20260919/e252-lh-closed-loop-divergence/`, 판정 `parent-e252-final-review.json`, 독립 검산 `parent-e252-output-review.json`, 감독 `parent-e252-supervisor.json`에 있다. 다음은 LH femur_reductor41의 실제 MN과4bridge 기존 간선 기여를 동결하여 조사하는 E253이다. 이 결과만으로 gain sweep, 전 다리 복원, 새 신체 실행을 하지 않는다. E150 best partial/M2/default/UI/F6 미노출/전체 목표 미완료를 보존한다.
