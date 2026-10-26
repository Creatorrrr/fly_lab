# E257 고정6간선 복원 후보의 신체 폐루프 대조 — 2026-09-25

E256에서 E250 기존25간선 중 목적지가 기존 intrinsic_timescale38 밖인111970으로 향하는6개만 복원하면, 200ms 감각 운동 반응 R/D가 고정 문턱을 통과하면서 original24 출력 변화는 전체25개 복원의 .03814%였다. 시작은 교사 끝 tick40000이었고 신체가 없었으므로 초기 자세나 보행 개선은 검증되지 않았다. E257은 동일6간선 후보를 E150의 노출된 control0 전체 신경·신체 상태로 되돌려, 원래25개 후보가 실패한 E251의 고정 신체 기준을 적용한다. E251 기각은 보존한다.

## 하나의 후보, 두 팔, 변경하지 않는 판정

baseline과 candidate를 순서대로 각각 최대300controls/1.5초 실행한다. 동일 E150 `candidate-neutral` 전체 상태를 사용하고, baseline은 원래 CSR, candidate는 E256에서 검증한6개 기존 data 값만 복원한 CSR이다. 후보의 실제 전체CSR과 model identity는 E256과 일치해야 한다. 나머지19개 E250 대상 및 모든 비대상 CSR, tau/gain, 신체·adapter·navigation·tonic·seed·초기 동적 상태를 유지한다. 새로운 간선/세포/교사/CPG/위상/root 보정/fit/solver/배율 탐색은 없다. F6 상태를 사용하지 않는다.

E251의 검증된 전체 상태 전달·소유권·부분 실행 계측과 관측/분석 방식을 재사용한다. 새 trial wrapper는 명시적 E257 schema/6간선 provenance를 가지며, candidate는 새 생성자가 계산한 네 network identity 항만 바꾼다. baseline은 identity까지 exact, 그 외 두 팔의 전체 신경·신체 시작 상태는 E150과 exact여야 한다. 원래 동결 E251/E256 소스를 수정하거나 글로벌 함수를 monkeypatch해 의미를 바꾸지 않는다.

전체158706뉴런의 실제 CUDA packed/capture20/dt.00025, 원래 MuJoCo 신체를 사용한다. 매 control 실제 감각 encode→20neural ticks→decode→50physics steps를 유지한다. source23/bridge4/reachable55의 기존82개 관측과 MN391·근육96·명령42·tendon6·native 접촉/슬립을 기록한다. 추가로 original24의 rate/drive/adaptation을 별도 `core-activity.npz`에 수동 관측한다. 이 기록은 drive/mask/상태를 바꾸지 않으며 rate/drive/adaptation readout 외 신경 계산을 추가하지 않는다. baseline의301행 control/recruitment와15000 native는 원래 E150과 exact여야 하며, 이 비교가 실패하면 후보를 실행하지 않는다.

H257은 E251의 아래 다섯 조건을 그대로 모두 요구한다.

1. 두 팔 모두301 control행/15000 native를 완료하고 upright≥.9, 비발 접촉0, 수치·body fault0, 정확한 신경·물리 시계를 만족한다.
2. 후보는 원래 E103 target의 strict startup99에서 q RMS≤.018rad/max≤.04rad, LTM RMS≤.009/max≤.025를 만족한다.
3. 원래 초기 heading에 대한 [.5,1.5)초 전진 거리가 baseline 이상이다.
4. 같은 구간에서 load≥.02BW인 유효 native 하중 가중 slip이 baseline 이하이다. 선택된 접촉에서 누락·비유한 slip은 통과시키지 않는다.
5. 변경하지 않은 gait evaluator의 longest_forward_bout_s가 .3초 이상이며 baseline 이상이다.

여섯 다리 운동폭·cycle/phase/period와 기존24 출력 변화를 기술하되 추가 관측으로 판정을 대체하거나 사후 문턱을 조정하지 않는다. H257이 참일 때만 동일 후보의10초 및 저장 복원/차단 검증을 다음으로 설계한다. 미달이면 이6간선 구조 분할은 보행 후보로 기각한다. baseline 재현/저장/계측 오류는 unavailable, 실제 신체 건강성 또는 동역학 미달은 유효 부정 결과로 구분한다. 짧은 통과도 F2/F3–F5/F7이나 자연·생물학적 보행 성공을 뜻하지 않는다.

## 구현, 검토, 원자료

부모 설계 후 실제 gpt-6-sol/high가 `tools/verify_c_banc_m3_lh_partition_body.py`와 `tests/test_c_banc_lh_partition_body.py`를 구현한다. E251의 검증된 판정/관측/계측을 재사용하되 명시적6간선 load/restore와 새 wrapper/runner가 필요하므로 high다. 부모 FIRST→필요시 같은Sol 첫수정→부모 SECOND→잔여 부모수정 순서를 유지한다. 생성 fixture로 실제6개만 변경/19보존·새 identity·원래 whole 상태/소유권·부분 신경/물리/생성자 계측·cleanup 및 늦은 게시 실패 철회·H257 고정 문턱·core readout 비변경을 확인한다.

출력은 `verification/banc-natural-gait-m3-20260919/e257-lh-partition-body/`다. E251 호환 이름의 두 whole start/final, control/native/recruitment/sensory/activity/적용명령/clock/control/construction/cleanup ledger, actualCSR·packed/identity 증거와 별도 core-activity301행을 저장한다. 기술 실패 때 부분 원자료·최초/부차 오류·실제 비용을 유지하고 성공 판정을 철회한다. 부모는 저장된 전체 CSR6개 변경/whole state/prefix/실제 sense→command·RMS/전진/slip/bout/계측·manifest를 독립 검산한다. 후보의 whole disk future는 실행하지 않으며 F2 미완료를 유지한다.

## 자원 배정

원본1/retry0, CPU1/GPU1, whole restore2/underlying network constructor14, setup 실행560neural ticks와 capture 기록280ticks, body2/settling4000physics를 별도 계측한다. 과학600controls/12000neural ticks/30000physics steps(3.0neural·physical모델초), fit·solver0다. 실패한 부분 계산과 미상 실행량도 기록한다.

E251의 동일 paired1.5초 원본 감독은80.976708초였다. 입력검증·생성·snapshot·원자료 및 새24개 readout 여유를 포함해 동일 internal140(마감15)/external150/supervisor160초를 사전 배정한다. 누적6913.656375099632초와 과거 비용을 유지하고, 미래 allocation을6933→7074초(+141)로 늘린다. 잔여160.343624900368초에서160초를 예약한다. 이는 승인된 로컬 개발의 단계 배정이며 사용자 전체 시간·토큰 예산을 새로 설정하지 않는다. 원본당300초 제한도 유지한다.

host6GiB/장치추가GPU6GiB/최소GPUfree7GiB/output768MiB/free disk2GiB다. 준비·생성·부모 검산은 원본과 별도, 미측정 개발·자문은null이다. 자동 재시도나 실패 비용 삭제는 없다. E150 best partial/M2/default/UI/F6 미노출과 모든 원본·실패를 보존한다. F1–F8은 같은 최종 후보가 충족해야 하며 현재 미완료다. goal API의 paused 상태를 해제했다고 기록하지 않고 사용자의 계속 요청에 따른 일반 승인 작업을 진행한다.
