# E297 — 보호 후 남은 관측 전달과 시작 명령 오차의 저장 분석

Root 설계, 2026-09-28. E296는 .5초 건강성과 core향6172 선택 연결의 계수1을 유지했지만 qRMS .0189394/max .0760474로 원래 strict startup을 실패했다. 저장 탐색에서 core24 rate 변화 RMS .0769225/max .494019, drive/adaptation exact였다. 선택725849연결은 전부 premotor→premotor이며 직접 motor-target은0이다. 후보를 채택하거나 u/tau를 바꾸어 재시도하지 않는다.

질문은 **두 기록에 공통으로 관측된 core24와 나머지29source의 rate 변화가 고정된53source→391motor 전달에서 얼마나 나타나는지, 각 motor/근육/관절의 차이 및 원래 startup99행 오차가 어디에 남는지**다. 관측하지 않은 source rate를0으로 채우거나, 이 부분 합을 전체 수신 전류 또는 단일 인과 원인으로 부르지 않는다. 신경·신체·CUDA·solver·fit·새 후보 선택은 모두0이다.

구현 전 Sol/xhigh의 읽기 검토에서 E150 metadata55 중 candidate 관측에51845/136232 두source가 빠졌음을 확인했다. 두 열은 각각117/122개의 nonzero 양수 motor계수를 가져 생략해도 같은55합이라고 할 수 없다. 초기 설계 원문은 `verification/banc-natural-gait-m3-20260919/e297-design-before-observed-coverage-correction.md`에 보존한다. 새 trajectory 범위는 결과순위가 아닌 관측 가능성의 교집합53개다. 후보 whole final은 전체 neural state를 포함하므로 control100의 full55 endpoint만 별도로 확인한다.

## 입력과 provenance

E296 protected roster, candidate activity/native-contact/native-recruitment-state/trace와 final whole checkpoint의 모든 파일, result/ledger/root raw audit/source manifest/output manifest/freeze/source review와 원래 E150 recruitment-signals/recruitment-metadata/native-recruitment-state/trace/output manifest, E257 baseline core-activity/baseline-csr/output manifest, 원래 E103 startup-inputs를 사용한다. 사용 파일의 SHA와 실제 생산자 manifest의 bytes/SHA, E296 원본 source-review/실행 manifest를 동결한다. E296 root audit pass/run_valid/healthy100control/원래startup미달/채택false를 요구한다. 과거 producer 소스는 편집하지 않는다.

E150 observed_ids446의 순서는 정렬되지 않았다. 명시적인 unique ID map을 사용하며 E296 관측에는 공통53source와391motor가 전부 있어야 한다. 원래24core는 E257 고정 ID 전부이며 공통53source의 부분집합이어야 한다. 누락두source의 ID/계수/관측 불가를 출력한다. 후보와 기준의controls1..100/physical5ms를 원래2e-8초 허용치로 정렬하고 관측하지 않은 baseline0행은 만들지 않는다. 원래99 startuptargets는 controls1..99이며100행 paired-change와 섞어 보고하지 않는다.

## 고정 계산

승인된 E295의 `aligned_signals`/`first_and_windows`, E293 typed/clock validation을 읽기 전용으로 재사용할 수 있다. E296 입력을 명시적으로 전달하되 E295의 원본 gate/provenance를 변경하거나 전역 monkeypatch로 E294처럼 가장하지 않는다. 새 gate/provenance가 E296 실제 생산자를 검증한다. Core24/MN391의 rate/drive/adaptation,96근육activation,42관절/6힘줄 명령을100행 원래 기준과 대응하고 모든 셀/열의 fixed windows1..10/11..20/21..100 요약과 첫 exact차이를 보존한다.

순수 `observed_transfer(metadata, baseline_signals, candidate_activity, core_ids, csr)`를 노출한다. 원래 source_to_motor float32(391,55)는 E257 원래CSR에서 motor행/source열을 직접 뽑은 행렬과 byteexact여야 한다. 불일치 시 임의 계수를 대체하지 않고 분석을 중단한다. Metadata의55source 순서를 유지한 공통 관측53개와 누락2개를 계산한다. 같은 고정 float32 계수를 float64 수학합으로 적용하여 controls1..100의 baseline/candidate/delta를 core24source, 나머지 관측29source, 전체 공통관측53source별로 저장한다. signed/positive-weight/negative-weight 항을 각각 제공한다. 그룹 합과 signed=positive+negative를 작은 부동소수점 재결합 오차 안에서 검증한다. 이 값은 모델 입력 단위이며 CUDA 순서 exact current·전체 current·생리 암페어가 아니다. Resource를 직접 MN향 입력에 다시 곱하지 않는다. E296 resource가 걸린 선택 연결에는 MN target이 없다.

후보 final checkpoint는 StateStore 등 승인된 passive reader로만 읽고 control_tick100/network.tick2000/물리.5초와 기존 audit/SHA를 확인한다. 전체 rate에서 source55/motor391을 뽑되, 관측된53source/MN391은 candidate activity100행과 byteexact여야 한다. 이 endpoint에서만 missing2의 baseline/candidate rate와391motor 전달항 및55전체합을 제공한다. 기존E150100행의55source rate와 짝을 맞추며100 이전 누락rate를 보간·복원하지 않는다. 재생/construct/CUDA는0이다.

해부학적 고정 분할만 사용하며 모든391motor와 관측 가능한 source 원자료/통계를 남긴다. 가장 큰 셀만 골라 새 후보를 만들지 않는다. 한 번의 짧은 이행 궤적을 정상상태 tonic/phasic 또는 반복 주기 증명으로 사용하지 않는다. Metadata55 밖의 afferent, candidate 두source의 중간rate, E150에서 기록하지 않은 premotor rate의 baseline 변화는 미관측으로 명시한다. Direct drive exact 여부와 간접 감각 영향을 구분한다. 첫 차이는5ms 관찰 시점이며 substep onset/인과 선행이 아니다.

`startup_error_decomposition(baseline_q, candidate_q, original_targets)`는99×42 원래 정의에서 baseline_error=baseline-target, candidate_delta=candidate-baseline, candidate_error=candidate-target를 저장한다. 모든42열에 RMS/max/peak control을 기록하고 SSE(candidate_error)=SSE(baseline_error)+SSE(candidate_delta)+2*sum(baseline_error*candidate_delta)를 독립적으로 확인한다. 원래 기준과 후보의 같은99행전체RMS/최대값을 제시한다. 새 보행 PASS나 원래 시작 문턱 변경은 없다. joint 이름을 데이터로 확인하지 못하면 원래42열 index로 표기한다.

## 실행과 검토

실제 gpt-6-sol/high가 신규 `tools/analyze_c_banc_m3_observed_transfer_residual.py`, `tests/test_c_banc_observed_transfer_residual.py`만 구현한다. Root FIRST→필요시같은Sol 첫수정→root SECOND→root 잔여수정이다. 다른 소스·raw·docs·CP는 읽기 전용이다. 작은 생성 fixture는 unsorted ID/정확한CSR행렬/부호·빈그룹/공유source/전달합/99·100행구분/누락·중복·dtype·clock·SHA오류와 실패영수증을 검사한다. 기존에 충분히 검증된 함수의 구현을 그대로 복제해 테스트 수를 늘리지 않는다.

ROOT/OWNED/SOURCE_FILES/INPUT_NAMES/LIMITS/FREEZE_SCHEMA/REVIEW_SCHEMA 및 gate/run CLI를 노출한다. 새 freeze/source-review SHA와 fresh fixed output `verification/banc-natural-gait-m3-20260919/e297-observed-transfer-residual`을 요구한다. 실제 실행 source/input/output manifest와 실패·비용 영수증을 보존하고 모든 adoption/F1–F8/F6/causal/candidate-selection flag는 false다. 모델 생성 없이 NPZ/JSON만 읽는다.

생성 검증45초/1GiB, 실제 저장 분석1회45초/2GiB/output64MiB/CPU1이다. Root는 별도의 tiny 손계산, 실제 sparse 재계산, startup99행 SSE와 모든열 요약을 검산한다. 원본 누적8078.0529733995245/8150초와 모든 실패를 이월하고 이번 원본 예약0이다. 결과가 core rate 보존만으로 남은 전달 오차를 설명하기 어렵다면 관측된 다른source 또는 동역학의 DC/과도 응답을 구조적으로 검토하며, 같은 u/tau/seed 격자나 startup/F1–F8 완화로 전환하지 않는다. 새 원본 후보와 자원은 별도 root 설계·검토 후에만 정한다.
