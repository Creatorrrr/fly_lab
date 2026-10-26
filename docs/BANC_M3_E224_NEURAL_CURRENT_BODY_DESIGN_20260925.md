# E224: 같은 E150 출발 상태에서 신경 전류 기반 근육 변환 비교

## 근거와 판정 범위

E222의 MN rate 교체 모델은 실제 몸체에서 0.2244초에 upright .9 아래로 떨어졌다. E223은 재귀 신경 상태를 유지하는 별도 effector의 생성 CUDA 계산만 검증했다. 이번 실험은 재귀율 교체를 피하는 것이 몸체 안정성을 개선하는지 판별하는 두 개의 사전 고정 비교다. 새 변환은 기존 MN firing 자체가 아니며, 단순 명칭 변경으로 F2를 통과시키지 않는다. 원래 해부학·지속 감각 루프를 보존하는 모델 가설로만 취급한다.

## 고정 비교

1. `legacy_canonical`: E150의 원래 MotorTimescaleNetwork + canonical float64 근육 계산. E150 monotone 해독기는 제거하되, 신경 파라미터·몸체·initial action·감각 인코딩·자세는 동일하다.
2. `effector_canonical`: 같은 출발 상태 + E223 NeuralCurrentEffectorNetwork + 같은 canonical 계산. 전체 neuronal rate/adaptation/CSR는 기존 식을 사용한다. 자유343 MN의 별도 effector만 근육 입력에 사용하고, 고정48 MN은 원래 rate를 사용한다.

두 arm은 순서대로 각각 최대600 controls=3.0초를 한 번 실행한다. 한 arm의 건강 중단은 저장 후 다음 arm으로 진행한다. 기술 오류·자원 초과는 즉시 실험 종료하며 재시도하지 않는다. body 상태나 입력을 비교 결과에 맞춰 수정하지 않는다. E150·E218·E222의 저장 prefix는 사후 대조만 가능하고 런타임 궤적 입력으로 사용하지 않는다. 목표·phase·time·teacher·CPG·root 보정은 입력에 넣지 않는다.

## 구현 계약

- 새 파일 `tools/banc_m3_neural_current_trial.py`, `tools/verify_c_banc_m3_neural_current_body.py`, 해당 두 테스트만 Sol/high가 수정한다. E223 및 과거 파일은 동결한다.
- 부모가 설계·freeze를 작성하고 실제 gpt-6-sol/high가 구현한다. 부모 FIRST → 문제 있을 때 같은 Sol 첫 수정 → 부모 SECOND → 남은 문제는 부모가 수정한다.
- `unwrap_e150_checkpoint`, `checked_grouped_parameters`, `CanonicalRate64Adapter`의 동결 계약을 재사용한다. source neural identity, 391/343/48 IDs, 원래96 muscle membership, 모든 해부학 계수 및 E217/E218 bytes를 검사한다.
- session.rate와 일반 readout은 원래 neuronal rate를 유지한다. effector arm의 decode 경계에서만 별도 effector를 받아 canonical decoder로 전달한다. motor cut은 원래 neutral/activation0 경로를 유지한다. 이 시험에서 실제 motor-cut/clone 실행은 하지 않고 `not_executed`로 기록한다.
- effector bridge는 네트워크 포인터를 저장 직렬화하지 않는다. checkpoint에는 원래 신경 state와 별도 effector/history state, wrapper 모드/identity/provenance가 모두 포함되어야 한다. 원래 layer로 unwrap한 뒤 재바인딩하여 복원한다. 서로 다른 arm의 checkpoint를 혼동하지 않는다.
- 동일 E150 start를 두 번 독립 복원하며 시작 body state·initial action·감각·clock 동일성을 체크한다. 초기 canonical activation과 원래 신경 rate/adaptation/drive는0이어야 한다.
- control raw는 원래 motor_rate32[391], 실제 decoder_input32[391], free effector64[343], fast/slow[343,3]을 구분한다. legacy arm에는 effector 배열을 가짜0으로 채우지 말고 해당 key가 없고 mode가 명확한 raw를 사용한다. 두 arm 모두 activation96, command42, tendon6, control/neural/physics clocks를 저장한다.
- 100us 물리 native contact/pose/health observer, control journal, 부분 실패 AdvanceLedger를 유지한다. upright<.9, 비발접촉, numeric fault는 기존 즉시 중단 기준을 그대로 적용한다. 건강 중단을 기술 실패나 성공으로 덮어쓰지 않는다.
- 시작/끝 체크포인트(건강할 때만), trace.jsonl, raw npz, native-contact, per-arm result/ledger, 전체 result/source snapshot/output manifest를 보존한다. 실패 중간 상태는 정상 checkpoint로 저장하지 않는다.

## 사전 판정과 자원

각 arm에 complete_3s, 실제 건강 중단 시간·upright·nonfoot·clock, 몸체 이동, raw 유효성을 보고한다. 3초 건강 유지가 다음 단계 검토 조건이다. 이것은 F3–F5의 고정 10초 기준을 대체하지 않는다. 둘 다 실패하면 effector 후보를 거절하고 원래 불충분한 동역학/근육 변환 가설을 재검토한다. 한쪽이 개선되어도 전체 goal·F2·F6·F7 통과로 간주하지 않는다.

원본 누적 `5818.539935299679s`, 한도 `6004s`, 잔여 `185.4600647003208s`. 추가 할당 0. 이번 예약160s 이내: internal145s(최종저장20s 예약), external150s, supervisor160s. original process1, arm2, retries0, fit/solver0, CPU1, GPU1, RSS6GiB, 추가GPU2GiB, output512MiB, free disk2GiB. 과학 계산 최대24000 neural ticks/60000 physics steps, constructor warmup/capture는 별도 실측한다. 실사용 supervisor 전체를 원본 누적에 반영하고 미사용 예약은 반환한다.

실행 전 freeze/input/source review/hash/package/M2를 검사한다. 모든 owned process와 descendant의 종료를 추적한다. 독립 부모 raw 감사 후만 결과를 확정한다. UI/default/F6/기존 M2 변경 및 goal 완료 선언은 금지한다.
