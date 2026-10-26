# E141 실행 전 설계 정정

2026-09-21 · 원본·조건부·신경·신체 실행 전 · v1 보존, 실행 계약 v2

Sol/xhigh의 읽기 전용 통합 조사와 현재 소스 확인으로 초기99 기준의 출처를 정정한다. E121 `original_startup()`의 실제 신체 허용치는 qRMS.02/qmax.05/LTM RMS.01/max.03이다. E141 본문에 기재한 .018/.04/.009/.025는 E128–E140 조건부 적합 기준이다. 이를 E121의 원래 신체 기준이라고 부른 표현은 잘못됐다.

E141의 사전 수치 .018/.04/.009/.025는 유지한다. 기존 E121 함수의 raw 계산과 기존 판정도 별도 기록하되, E141 공동 screen에는 명시적인 더 엄격한 네 숫자를 사용한다. 이는 새 원본 결과에 따른 기준 변경이 아니며, 아직 원본 실행은0이다. 부모는 첫 설계·freeze·준비 기록을 보존하고 이 정정과 수치 필드를 포함한 v2 freeze로 실행을 봉인한다.

모수 bounds는 동결된 원래 delta/gain/ell에 적용한다. 실제 연결값은 `float32(E121_float32_base_weight * exp(delta64))`이며, calibration의 derived efficacy는 대응하는 반올림 전 target64를 원래 source_values로 나눈 값이다. 이 파생 기록의 범위는 [0.0009999999615300753,10.000000486358948]로, 기저 float32 반올림 때문에 원래 추상 efficacy의 [.001,10] 경계를 조금 벗어난다. delta bounds를 완화하거나 weight를 clamp하지 않는다. 부모 준비 원자료와 실제 float32 값을 exact로 비교하고, 해부학적 ID/CSR/부호/비대상 연결 불변을 검사한다. 모델의 추상 모수와 실제 float32 표현을 혼동하지 않는다.

실제 setup 계측은 각 trial construction 범위에 한해 Python profiler로 기존 `_interval` 함수의 진입·반환을 관찰하고, 별도 CUDA synchronize 완료를 기록한다. 보호된 신경 모듈이나 커널·파라미터를 monkeypatch하지 않는다. 준비 단계의 실제 작업을 원본40,400 scientific tick과 분리하고 오류 시 미확정 작업을0으로 표기하지 않는다. profiler는 원래 값을 보존하고 범위 종료 때 복원한다.

E140 조건부 guard의 LH/RH 미달과 전체 패키지 미채택, M3 F1–F8, 활성 UI·M2·F6 미노출 및 나머지 E141 설계·자원은 그대로 유지한다.
