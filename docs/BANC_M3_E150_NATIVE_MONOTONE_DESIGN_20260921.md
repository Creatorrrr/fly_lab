# E150 고정 단조 모집 후보의 실제 BANC/FlyBody 검증

2026-09-21. E148 동일 계수의 수정 후 저장자료 검산과 E149 adapter의 부모 두 번째 리뷰가 통과했다. 저장자료의 21.2189% 명령 오차 개선이 실제 닫힌 고리 보행으로 전달되는지를 한 번의 개발 시작 상태에서 검증한다. F6 두 미사용 자세와 M2·활성 UI는 사용하지 않는다. 현재 최신 실제 E142는 F3 통과/F4·F5·strict startup 미달이며 이를 그대로 보존한다.

## 구현과 입력

부모가 설계·동결·독립 검토를 맡고 기존 Sol/high가 아래 신규 두 파일을 FIRST 구현한다. 새 xhigh 생성은 세션 agent thread limit으로 거절되었다. 이번 하위 작업은 검토된 CUDA 신경망/physics/회계 helper의 재사용과 명시적 factory 연결로 범위를 제한한다. 새로운 CUDA kernel, 신경 동역학이나 adapter 상태 구현은 없다. 해당 연결과 실패 처리 코드는 부모가 집중 검토한다. 같은 Sol 첫 수정 → 부모 두 번째 리뷰 → 부모 잔여 수정 순서는 유지한다.

- tools/verify_c_banc_m3_native_monotone_body.py
- tests/test_c_banc_native_monotone_body.py

입력은 E141/E142와 동일한 전체158706 BANC/원래 CSR/neutral checkpoint, E142 calibration·muscle provenance와 E148의 한 번 반환한 동일 slope/kappa/tau/membership이다. 부모가 파라미터 bundle을 복사하고 원본과 bit-exact를 확인해 SHA를 동결한다. 재적합/후보 재선택/곡선 보정/신규 신경 연결·CPG·위상·teacher·root 보정은 없다. 신경 CUDA .00025초×20, 제어 .005초, 신체 .0001초×50을 유지한다. runtime은 E149 MonotoneRecruitmentTrial을 사용한다.

CLI는 --freeze/--freeze-sha256/--source-review/--source-review-sha256/--output이다. 입력/재사용소스/보존파일/신규두소스/부모실행허용키와 해시를 모두 검사한 뒤에만 GraphStore/StateStore/실제 모델에 진입한다. source gate의 accepts_single_native_monotone=true가 필요하다. 생성 테스트는 fake/stub만 사용한다. 자식 에이전트는 원본 자료/GraphStore/StateStore/CUDA/MuJoCo/실험을 실행하지 않는다.

## 준비와 동일 프로토콜의 재사용

검토된 E142 load_cached_candidate/verify_zero 또는 동등한 명시적 복사 경로로 기존 원본 zero와 그래프/파라미터를 확인한다. 기존 MotorTimescaleTrial.from_checkpoint를 profiled_construction으로 한 번 복원한 후 E149 wrapper를 적용한다. network 객체/identity/전체 source CSR와 기존 motor timescale metadata가 동일하고 control_tick/network.tick0, 물리시간0, filter bank/activation0임을 확인한다. 신규 candidate-neutral을 저장하고 기존 zero의 body/world/cuts/navigation/parameters/seed/cpg/initial heading을 보존한다. body 소유자를 닫고 실패 시 최초 예외와 정리 오류를 함께 보존한다.

기존 verify_c_banc_m3_native_timescale_transfer._native_body의 검토된 code를 재사용할 수 있다. 전역 모듈은 수정하지 않는다. types.FunctionType으로 code/default/closure를 그대로 둔 별도 globals 사본을 만들고 **MotorTimescaleTrial factory, _observe의 추가 bank 기록, _save_body_raw의 추가 자료 저장, INTERNAL_LIMIT_S(250)**만 교체한다. BODY_LIMIT_S180/RESERVE_S12와 그 밖의 evaluator/advance/회계/guard는 바꾸지 않는다. 허용한 네 override 이외의 값이 같음을 생성 테스트와 부모 리뷰에서 검증한다. 원래 helper 함수와 전역 bindings는 그대로 유지한다. 이 허용 목록을 벗어나는 동적 변경은 금지한다.

_observe는 원래 함수 호출과 동일 readout을 유지하고 실제 bank/activation/original activation/391 MN rate/time/control tick을 추가로 caller-owned 기록에 남긴다. 시작과 매 main control을 포함한다. 읽기 실패 전에 접근 가능한 추가 상태를 partial에 게시한다. _save_body_raw는 원래 raw 저장을 유지하고 추가 native-recruitment-state.npz를 분석보다 먼저 저장한다. 원래 trace의 내용/수치는 바꾸지 않고 schema/controller/실험 provenance만 E150으로 명시한다. 이전 helper의 E121/E141 호환 필드명이 남는 보고서는 재사용한 동일 수치 프로토콜임을 별도 명시하고, E142-E150 비교 sidecar를 새로 작성한다.

## 원본 실행과 검증

main2000 controls=10초를 실제 감각→BANC→새 adapter→신체로 실행한다. 1초 checkpoint를 저장/재로드한 독립 clone20 controls=.1초와 main1.1초의 **전체 저장 미래 bit-exact**를 요구한다. main과 clone을 동시에 advance하지 않는다. 원본신체 health/clock guard, 실패 전 실제 진입·반환·readout·부분 자료 보존은 기존 프로토콜과 동일하다.

main이 기술적으로 완료되고 checkpoint-1s가 저장됐으면 같은 이미 노출된 checkpoint에서 별도 cut/restart clone을 복원한다. set_cuts({'motor':True}) 후4 controls=.02초 동안 매 control bank/신규activation/원래activation0, tendon0, neutral target을 확인한다. set_cuts({'motor':False}) 후16 controls=.08초를 진행해 유한 출력과 실제 양의 MN/activation/비neutral 명령 재개를 확인한다. 이 검사는 F2의 운동 전달 차단에 관한 증거이며 F7 목표 접근·자율 정지/재시작을 대신하지 않는다. 신체 fault/clock 부족을 성공으로 간주하지 않는다. cut/restart에도 별도 AdvanceLedger/ControlJournal, raw/control/native data와 실패 시 정리/부분 저장을 적용한다. 원래 main 통계에는 합치지 않는다.

요청 scientific 합은 main10+future.1+cut/restart.1=10.2초, 40800신경tick/102000물리step이다. 준비 warmup/capture/각 restore의 실제 진입 비용은 profiled_construction 원래 helper로 별도 계수한다. 모듈/프로세스 종료를0작업으로 간주하지 않는다. 그래프나 신체를 감추어 재실행하지 않는다.

F3–F5는 기존 body_quality/analyze와 [0.5,10) 창, 전방속도 .5mm/s·Gaussian20ms·bout .3초 기준을 그대로 쓴다. F3 전진10mm/net-path.5/upright.9/nonfoot0, F4 여섯 다리10주기와 좌우 반주기 거리.15/결집.5, F5 LF25/RF30/LH40/RH35도 및 모든 periodCV.3이다. strict 초기99 qRMS .018/qmax.04, tendonRMS.009/max.025를 요구한다. 실제 새 닫힌 고리의 실패는 저장자료 성공으로 덮어쓰지 않는다. M3 F1–F8은 이 단계 성공만으로 완료하지 않는다.

## 실패·출력·자원

원본1프로세스/재시도0, CPU1/GPU1, 준비60/기존 body180/내부250/외부270/감독285초, raw4GiB/free12GiB다. 원래 누적2228.8917264초와 이전2271초 배정을 보존하고 **이번 실행 전에 ceil2229+285=2514초**로 누적 원본프로세스 한도를 배정한다. 기존 실패·기준을 완화하지 않으며 전체 프로젝트 생애 비용과 혼동하지 않는다. solver/LP/조건부 응답0, candidate_adopted/F6_exposed/active_UI_changed/whole_goal_complete 모두false다. 실제 모형 실행은 부모 최종 gate 뒤 한 번만 한다.

run은 첫 예외/secondary 오류를 보존하고 결과·실행ledger·manifest·필수 저장 포함 자원 검사를 수행한다. prepare/model 생성/restore/storage와 original forward counter는 실제 경계에서 갱신한다. body partial raw를 먼저 저장하며 분석 실패와 건강상 중지를 구분한다. 원본 출력 디렉터리는 새 경로만 허용한다. 부모는 실제 입력/모델/provenance·bank recurrence/직접 명령/저장미래/실제 clock·F3–F5/startup·cut/restart를 원자료로 독립 검산한다.
