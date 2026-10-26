# E258 저장6간선 후보의 초기 명령 오차 분해 — 2026-09-25

E257은 전진 증가에도 strict startup/slip/bout을 위반했다. 새 모델 실행 전에 저장된 초기 오차가 이전25간선 실패의 LH femur_reductor(고정 muscle41)와 같은 명령 경로에서 나타나는지 확인한다. 신경·신체 인과 실험이나 후보 탐색이 아니라 저장 endpoint의 산술 분해다. E257 기각은 유지한다.

## 고정 질문과 판정

startup controls1..99에서 baseline |command−target|≤.04이고 candidate >.04인 새 위반을 모두 찾는다. 최초는 control 우선·joint column 우선, 최대는 candidate 절대 오차의 control/joint 순서 argmax로 고정한다. 물리 joint_columns/DOFS 대응은 E257 sensory metadata를 사용한다.

원래 rest+.02*moment@activation이 각 팔의 controls1..300 명령을1e−12rad 이내로 재구성해야 한다. row0은 decode 전이므로 잔차를 별도로 남긴다. 96근육별 signed delta command=.02*moment[j,m]*(A_C−A_B)[m]를301×42×96으로 보존하고, controls1..300의 근육 합계가 실제 명령 차이와 일치해야 한다. startup signed SSE 변화는 양/음·linear/quadratic 항과 다리별 합계로 모두 보고한다.

H258_same_muscle은 아래 조건을 모두 만족할 때만 지지한다.

1. 새 위반이 존재하고 모두 LH이며 LH의 양의 SSE 증가 비중이50% 이상이다. E252와 같은 국소화 문턱이다.
2. 최초 새 위반과 최대 오차의 두 고정 endpoint에서 각각 candidate error의 부호 방향으로 투영한 근육별 delta command의 양수 합이0보다 크다.
3. 각 endpoint의 그 양수 합에서 고정 muscle41(LH femur_reductor)의 양수 기여 비중이80% 이상이다. 기존 E253의 단일출력 우세 문턱80%를 같은 국소화 질문에 적용하며 대상을 결과에 따라 바꾸지 않는다.

지지되면 이전 실패와 같은 muscle 출력의 방향·시간 구간을 다음 기전 검토 대상으로 삼는다. 미지지면 실제 양/음 기여와 고정 endpoint의 다른 named muscle을 보고하고 기존41 가정을 기각한다. 단일 근육의 생물학적 원인/신경 인과성으로 부르지 않는다. 어느 결과도 gain sweep이나 새 모델 실행을 자동 허용하지 않는다.

## 추가 기록과 유효성

E257 original24의 rate/drive/adaptation 차이를 full0..300, startup1..99, steady100..299, 초기200ms1..40에 보고한다. 각 창에서 RMS/max/양·음·signed 합계와 첫 exact/절대1e−5 초과 endpoint를 저장한다. core rate는 cap200으로 나눈 RMS도 제공하고 E256의 정적 Q6 숫자를 나란히 기록하되, tick0의 실제 감각 폐루프와 tick40000 held input은 다른 조건이므로 비율로 인과·보호 성공을 판정하지 않는다. 기존82 activity에서 source23/bridge4/reachable55와 실제 source-drive one-control 지연, muscle41 activation 및 전체391 motor 차이도 같은 창/명시된 clock으로 기록한다. temporal 순서를5ms 내부 신경 단계의 인과 latency로 해석하지 않는다.

고정 입력은 E257 두 완전한301 endpoint의 control/activity/core/sensory/recruitment/metadata 및 감사·manifest·result, 원래 E150 calibration/muscle provenance/parameters/E103 target, original24 roster와 E256 Q receipt다. 전체 신경 궤적을 새로 생성하지 않는다. 실제 motor391/muscle96/joint42/핵심24/활동82 ID·shape/dtype/clock/finite/availability, 중복 저장된 MN/activation, source old_filter→endpoint t+1 적용을 확인한다. E257 baseline/source result는 유효·완전·부모 감사 통과이되 H257은false여야 한다. 누락·오류는 invalid이며 음의 H258은 valid일 수 있다.

## 구현과 자원

실제 gpt-6-sol/medium가 `tools/analyze_c_banc_m3_lh_partition_startup.py`와 `tests/test_c_banc_lh_partition_startup.py`만 구현한다. 새로운 신경/물리 수식이나 모델 실행 없이 검증된 E252 stage/SSE·기존 decoder의 저장 배열 산술을 적용하므로 medium다. 부모 FIRST→필요시 같은Sol 첫수정→SECOND→부모 잔여수정 순서를 유지한다. 고정96근육 순서와 임의 joint mapping, signed cancellation/0분모, 최초·최대 동률 결정, row0 제외, source-drive 시차, core형식, 늦은 게시 실패 후 nested H258 철회를 생성 fixture로 검증한다.

출력 `verification/banc-natural-gait-m3-20260919/e258-lh-partition-startup/`에 전체 muscle contribution/command reconstruction/startup SSE/core·stage 차이 및 명시적 시계를 NPZ로 저장하고, 고정 endpoint의96기여·named mapping·창별 통계·H258 및 inputs/output manifests/ledger/errors/result를 남긴다. 부모가 기존 산술 helper와 별도 NumPy/scalar 계산으로 원자료·판정·오류·자원을 독립 검산한다.

원본1/retry0/CPU1/GPU0, 모델 생성·신경tick·physics·CUDA·fit·solver0. internal30(마감5)/external40/supervisor45초, host2GiB/output64MiB/free disk512MiB다. 누적6995.528808599633/7074초·잔여78.471191400367초에서45초를 사전 예약하고 배정 증액0이다. 원본당300초 한도를 유지한다. 준비·생성·부모 검산은 원본과 별도, 미측정 개발·자문은null. F1–F8미완료/E251·E257기각/E150/M2/UI/F6 미노출/과거 비용과 실패를 유지한다.
