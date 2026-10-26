# E313: postsynaptic load의 전체 상태 연결과 제한 보행 비교

Root 사전 설계, 2026-09-28. [E312](BANC_M3_E312_POSTSYNAPTIC_LOAD_DESIGN_20260928.md)의 단일 lambda1 기전을 원래 E150 whole control0에 연결한다. E312 생성 CUDA qualification과 root 소스 검토를 모두 통과하기 전에는 실제 BANC/신체를 실행하지 않는다. E312와 독립적으로 wrapper/runner의 fake 구현·검토만 병행할 수 있다.

## 변경과 고정 비교

실제 graph158706/11584852, 원래 E289 P7136→P7136 간선725849 전체,391MN/96근육/42관절, E150 단조 모집과 기존 신체/감각/상수 descending 입력을 유지한다. 선택 간선은 원래 E289 selection에서 읽어 u0/R1으로 만들며 나머지 roster/CSR bytes는 고정한다. 적분기는 E303 SSPRK33, source resource는 전부u0이다. E296 보호 subset이나 E309 자원 활성과 결합하지 않는다. Neutral lambda0와 단일 후보 lambda1 외의 파라미터를 탐색하지 않는다.

신규 PostsynapticLoadTrial은 원래 MonotoneRecruitmentTrial control0/neural0의 전체 상태를 검증한 뒤에만 신경층을 교체한다. E312 PostsynapticLoadNetwork의 public API와 metadata를 사용한다. 새 trial/patch schema와 identity는 원래 graph·adapter·source neural metadata·monotone patch·선택·lambda·load metadata를 포함한다. 새로운 h는 f64 selected-post-ID 순서0이며 기존 R은1이다. 신체 ownership은 모든 검증 후 이전하고, 실패 시 caller를 보존한다.

Whole snapshot/save/from_checkpoint/restore는 nonzero control에서도 신경 r/z/h/R뿐 아니라 bank·held action·sensory memory·body warmstart·clock·wall 및 원래 모든 상태를 정확히 복원한다. 원래 shape로 된 임시 표현은 body 복원용일 뿐 original RK4 미래로 실행하지 않는다. 이전 SSP/resource-only schema와 lambda/selection이 다른 snapshot은 먼저 거절한다. Commit 후 old-body close 실패는 이미 새 whole이 반영됐음을 명시한다. 한정된 tuple/list 저장 표현만 기존 checkpoint_representation으로 처리하며 배열 dtype/bytes/모든 수치와 wall을 면제하지 않는다. 기존 모듈이나 schema를 monkeypatch하지 않는다.

## 원본 프로토콜

1. 모든 소스·입력·설계·package·보존 pin과 producer 의미를 검사한다. E312 root 검토/생성 CPU·CUDA qualification 및 이 wrapper의 실제 tiny neural+fake whole integration을 gate로 요구한다. 원래 E150 whole control0, E289 selection, E307 startup 및 E310 neutral continuation, 원래 startup target/short evaluator를 검증하고 비교 범위를 고정한다.
2. Lambda0 한 마리를 control0에서 생성하여 단1control/20neural/50native만 실행한다. 초기 original fields, 첫 native 50표본과 명령/모터/bank 및 첫 신경 rate/adaptation/drive/mask/tick이 저장 E307 neutral의 같은 구간과 exact여야 한다. Model identity·schema·새 h는 별도로 검증하며 원래 RK4와 exact를 요구하지 않는다. 새로운 whole을 memory→disk로 비교하고 종료한다. Neutral 본1.5초를 재실행하지 않는다.
3. Lambda1 한 마리를 같은 원래 whole control0에서 생성한다. 초기 원래 전체 상태와 신경 r/z/mask/drive/tick은 exact, h0/R1 및 새 metadata를 검사한다. 기전은 첫 SSP stage부터 동작하므로 후보 첫20tick이 neutral과 같다는 잘못된 조건은 넣지 않는다. .5초/100control을 한 번 실행해 원래 control1..99 startup q RMS≤.018/max≤.04, LTM RMS≤.009/max≤.025, upright≥.9/비발접촉0·시계·finite를 검사한다.
4. Startup을 통과했을 때만 동일 session을 1.5초/300control까지 이어 간다. 기존 short [.5,1.5) forward≥1.6790085779712631mm, slip≤1.8026436968249955mm/s, 최장 연속bout≥.3초, 여섯다리 각각 완전주기>0 및 원래 건강성을 모두 요구한다. 비교 heading은 원래 row0이다. 별도로 E307+E310 neutral의 같은 창에 대한 delta를 계산하며 absolute 기준을 matched delta로 대체하지 않는다. 두 자료를 runtime prefix로 재생하지 않는다.
5. Startup 실패면 기각하고 control101을 실행하지 않는다. Short 미달이면 이 고정 후보를 기각한다. Technical/nonfinite/resource 오류는 첫 원인·실제 완료량·부분 raw를 보존하고 자동 재시도하지 않는다. 성능/기술 실패를 구분한다. 통과해도 adoption/F1–F8/F2/F6/10초/natural/biological=false이며 별도 같은 후보10초 설계의 자격만 true다.

각 arm의 initial/final whole, 최초 control neural, trace, native contacts, MN/bank/activation/q/tendon, h와 g의 매 control 실제 배열·min/max, 모든 fault와 setup/capture/settling/science counters, 폐기/cleanup/미계측 비용 및 manifest를 저장한다. U는 매 단계 전체를 대량 저장하지 않고 실제 마지막 단계 gross diagnostic을 한 control 단위로 보존하며 stage time이라는 의미를 명시한다. Runtime device-to-host 전송은 이 작은 진단 배열로 한정하고 매control full network snapshot을 만들지 않는다. h endpoint와 마지막 gross는 같은 시점이라고 주장하지 않는다.

## 구현·검사·자원

실제 gpt-6-sol/xhigh가 `tools/banc_m3_postsynaptic_load_trial.py`, `tests/test_c_banc_postsynaptic_load_trial.py`, `tools/verify_c_banc_m3_postsynaptic_load_body.py`, `tests/test_c_banc_postsynaptic_load_body.py`만 소유한다. Root FIRST→필요 시 같은 Sol R1→root SECOND→root 잔여 수정 순서다. 기존 E304/E306/E307/E310 helpers를 재사용하되 이미 알려진 schema/NumPy scalar·영/비영복원/partial raw/최종화 오류를 다시 도입하지 않는다. Test 실행은 fake/생성 CPU 전체120초, CPU1/host2GiB. 구현 agent는 실제 CUDA/GraphStore/신체/solver/fit/원본0이다.

Generated tests는 전형적인 요청/실패를 미러링하는 수만 늘리지 말고, 실제 nonzero whole/future, 처음 caller 소유권·손상복원·기존/새 identity 경계, lambda0 only exact와 lambda1 early divergence, strict startup 경계·실패 후 noadvance, 이어가기 clock/원래 heading/short 양성 및 기각, 첫 과학 오류+게시/close 오류와 raw 보존, NumPy scalar delta를 검증한다. Root는 E312 실제 tiny neural을 연결한 fake whole 복원을 별도로 확인한다.

CLI는 --freeze --freeze-sha256 --source-review --source-review-sha256 --output. Fresh output은 verification/banc-natural-gait-m3-20260919/e313-postsynaptic-load-body다. Gate 이전 원본 예약0. 실제 원본은 root review와 passive gate 뒤 한 번/재시도0에 supervisor130초/internal115초+최종화15초, CPU1/host6GiB/additionalGPU4GiB/output2GiB/free4GiB를 예약할 계획이다. 현재 잔여143.1040624004636초 안이며 부족 시 실제 시작 전에 단계 배정을 명시한다. 신체2/과학neural6020/native15050 이하, setup·capture·settling 별도실측. 전체원본 누적8416.895937599536/8560초·예약0을 유지한다. M2/UI/F6/고정F1–F8/과거기각·비용은 불변이다.
