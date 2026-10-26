# E176: 기존 하중 감각 연결 복원 후보의 상태 구현

E175의 같은 저장 전달 가족을 다시 적합하지 않는다. 다음 변경 위치는 신경망 안의 기존 감각→운동신경 효능이다. E161에서 실제 E150 CSR와 대조한 RM load SNpp53→tibia flexor의 다섯 연결에 한정하여, E43의 비선택 MN 입력 0.001 배율을 제거한다. 현재 값에 1000을 곱해 반올림 오차를 만들지 않고, 원래 rate 모델의 float32 `.03/.06` 값을 직접 복원한다. 이는 해부학적 연결을 유지한 공학적 효능 대조이며 실측 생리 상수 복원이 아니다.

| CSR data index | pre graph index | post graph index | 현재 float32 값 | 복원 float32 값 |
| --- | --- | --- | --- | --- |
| 844054 | 107685 | 12036 | .00003 | .03 |
| 2400285 | 30981 | 32830 | .00006 | .06 |
| 7908675 | 107685 | 108052 | .00003 | .03 |
| 8578714 | 30981 | 117483 | .00006 | .06 |
| 11240394 | 30981 | 153939 | .00003 | .03 |

독립 gpt-6-sol/medium의 소스·이력 감사에서 이 다섯 연결만 복원한 과거 시험은 찾지 못했다. E05 전역 감각 gain, E18 전체 MN incoming, E162 group70 근육 효능 변경과 구분한다. 검색 범위 밖의 시험 부재까지 증명한 것은 아니다. E161에서 group74/75의 명령 변화는 해당 제동 창에서 아래쪽으로 투영됐다. 따라서 복원을 발 들기나 보행 개선으로 예측하지 않는다. 간접 회로, 자세, 하중의 폐루프 반응을 실제 대조해야 한다.

## 이번 구현 범위와 소유 파일

부모가 설계하고 실제 `gpt-6-sol/xhigh`가 첫 구현한다. 상태 저장·복원 계약이 새로 추가되므로 xhigh를 사용한다. 부모 FIRST 리뷰 → 같은 Sol의 첫 수정 → 부모 SECOND 리뷰 → 부모 잔여 수정 순서를 지킨다.

- `tools/banc_m3_load_feedback.py`
- `tests/test_c_banc_load_feedback.py`

기존 실행 소스, M2, UI, body, 감각 encoder, MN→근육 전달, 시간상수, fit 결과를 수정하지 않는다. 모듈은 `MonotoneRecruitmentTrial`을 받아 같은 `MotorTimescaleNetwork` 클래스와 CUDA 구현으로 효능만 바꾸는 연구용 wrapper를 제공한다. 새로운 CUDA kernel, 외부 CPG/위상/시각 gate, 추가 연결, 최적화는 없다.

실제 E176 파라미터는 위 다섯 행 하나뿐이다. 작은 생성 신경망 검증을 위해 동일한 엄격한 patch schema를 지원하지만, 이를 실제 후보 탐색 권한으로 사용하지 않는다. 후속 실제 실행기는 위 고정 행과 원본 graph/checkpoint/source hash를 따로 검증해야 한다.

## API와 상태 계약

1. `fixed_rm_load_patch()`는 위 표의 독립 복사된 int64 index와 float32 source/target 배열을 반환한다. schema, graph index와 CSR index를 모두 보존한다. 사전에 검증한 `.03/.06`의 출처는 `parent-e161-feedback-provenance-v2.json`이며 GraphStore의 `.275/contact` 단위와 혼합하지 않는다.
2. `patch_feedback_weights(source, patch)`는 square float32 CSR, 같은 길이의 비어 있지 않은 유일 int64 edge IDs, 유효 pre/post index, 정확한 실제 CSR 양끝, finite nonzero same-sign source/target을 검사한다. 각 선택 값은 bit-exact source와 일치해야 한다. 원본을 변경하지 않고 선택한 값만 바꾼 CSR를 반환한다. 비선택 data/indices/indptr 및 차원을 bit-exact 유지한다. alias, 입력 배열의 사후 변경, 중복·잘못된 edge, NaN/inf/부호 반전은 거부한다.
3. `LoadFeedbackTrial(base, patch)`는 E150 monotone trial과 정확한 MotorTimescaleNetwork를 사용한다. 같은 tau/a/threshold/cap, adaptation gain/tau, core IDs/gain, motor IDs/gain/log-tau, dt/device/cuda implementation/capture steps로 신경망을 새로 만든다. 새로운 network identity와 core/motor/timescale metadata만 새 구조에서 얻고, rate/adaptation/drive/output_mask/tick과 나머지 기존 dynamic state는 bit-exact 이식한다. 기존 CUDA graph의 tensor 포인터를 in-place 교체하거나 전역 모듈을 수정하지 않는다.
4. wrapper는 새로운 schema와 `load_feedback_patch` provenance를 저장한다. provenance에 고정 patch와 원래 network의 model/core/motor/timescale identity metadata를 포함한다. 원래 adapter 및 기존 patch 구조를 재작성하지 않는다. network identity는 실제 변경된 weight로 계산되어야 하며 기존 hash를 억지로 유지하지 않는다.
5. `from_checkpoint(graph, saved)`는 provenance 검증 후 새 network metadata를 원래 metadata로 되돌린 source checkpoint를 만든다. 원래 MonotoneRecruitmentTrial 경로로 source를 복원하고 같은 고정 효능 patch를 한 번 적용한 뒤 전체 saved snapshot과 정확히 비교한다. rate/adaptation/감각 filter/activation/신체 상태를 초기화하지 않는다. stale source identity, patch 값, endpoint, schema, snapshot 추가/누락/형 변경을 조용히 받아들이지 않는다.
6. wrapper 생성은 base의 body 소유권을 인수한다. 성공 후 이중 close를 방지하며, 실패 시 생성된 body owner를 정리하고 원래 예외와 정리 실패를 보존한다. 네트워크 구축·restore의 실패를 정상 결과로 표시하지 않는다. 저장·복원에서 반환하는 배열은 원본 state와 독립적이어야 한다.

## 생성 검증과 부모 리뷰

Sol은 원본 NPZ/GraphStore/checkpoint/CUDA/MuJoCo를 로드하거나 실행하지 않는다. 생성 CSR와 fake trial, 필요하면 작은 생성 CPU MotorTimescaleNetwork로 검증한다. 실제 원본이 없는 테스트를 실제 신경/보행 증거로 보고하지 않는다.

필수 검증은 정확히 선택한 다섯 값의 표, generic 생성 CSR의 선택 값/비선택 값/양끝/부호 보존, 잘못된 source·index·dtype·schema 실패, 입력/반환/저장 alias 방지, nonzero rate/adaptation/drive/output_mask/tick의 이식, 같은 source/target일 때 동일 identity·상태, 같은 모수의 작은 CPU 신경망 미래 일치, wrapper snapshot/restore/미래 일치, 상태 또는 metadata 변조 거부, 실패 시 body 소유권/정리 오류 보존이다. 테스트의 kernel 경로와 original scientific 실행량을 구분한다. 부모는 별도 scalar patch oracle와 상태 이식 계산을 수행한다.

## 실제 보행 대조로 넘어가는 조건

이번 단계에는 원본 값 로드·신경/물리 실행·fit·F6 노출이 없다. 독립 리뷰와 생성 검증이 통과한 뒤에만 부모가 별도 실제 실행 계약을 고정한다. 그 계약은 같은 E150 초기 상태와 같은 instantiated 모델에서 baseline/candidate의 변경된 CSR 다섯 값만 대조하고, 실제 CUDA, 저장 미래, 원래 strict startup, 10초 native F3–F5 및 미끄러짐을 검증한다. 이 후보의 의미 있는 전진 여부는 실제 자료에서 판단하며, wrapper 완성이나 작은 생성 시험 통과로 승격하지 않는다.

원래 strict startup qRMS .018/qmax .04, LTM RMS .009/max .025와 F1–F8은 유지한다. 미래 실제 대조의 입력·실행 수·자원·판정은 실행 전에 별도로 고정한다. 이번 단계의 원본 누적은 3085.011408499908/3202초 그대로이며, 생성·리뷰 비용은 따로 기록한다. E150 부분 성과, E175와 모든 과거 실패를 유지하고 candidate_adopted/whole_goal_complete는 false다.
