# E298 — 일정 입력을 보존하는 회복 계수의 저장 궤도 검사

Root 설계, 2026-09-28. E296와 E297의 유효 기각 및 F1–F8 미완료를 이월한다. E150 startup도 이미 원래 오차 한도에 가까우므로 같은 u/tau 격자, 보호 source 추가 선택, 닫힌 E287 재적합을 반복하지 않는다. 다음 공학 가설은 자원 상태 R 자체를 전류에 곱하는 대신 회복 중에만 감쇠하는 전달식이다. 아직 신규 신경 모델이나 신체 후보가 아니며 이 단계는 저장 데이터의 조건부 대수 검사다.

## 식과 한계

E296의 고정 source roster/u=.05/tau=.150과 면제 정책을 그대로 유지한다. 자원 미분은 `Rdot=(1-R)/tau-u*r*R`이다. 같은 상태에서 순서대로 float64 `a=R*(1+(tau*u)*r)`, `g64=min(1,a)`, `g32=float32(g64)`, 기존 계수 `old32=float32(R)`를 계산한다. 미래 커널이라면 같은 coupled RK 단계의 unmasked r/R가 필요하며 mask는 전류에만 적용한다. u0/R1 또는 g32가 정확히1인 전류는 원래 packed CSR 식을 직접 실행해 추가 곱과 FTZ 순서 변경을 피해야 한다. 이 단계는 그 커널을 구현하지 않는다.

수학적으로 uncapped a=1−tau*Rdot, capped g=1−tau*max(Rdot,0)이다. 유효 r≥0, 0≤R≤1에서 R≤g≤1이고 g<1은 R가 순간 평형 1/(1+tau*u*r)보다 작은 회복 상태라는 뜻이다. 단순히 rate가 하강했다는 뜻은 아니다. R1에서 일정 rate를 주면 해석해에서 g1이 유지되지만 임의의 낮은 저장 R에서는 일정 rate에서도 억제될 수 있다. 정적 직류 전달을 보존한다는 수학 성질과 유한정밀도 bit exact를 구분한다.

또한 uncapped 전달 r*g의 국소 기울기 R*(1+2*tau*u*r)는1보다 클 수 있다. g≤1이나 기존 scalar resource dt 제한만으로 coupled 재귀 회로의 안정성·보행을 보장하지 않는다. 후속 구현이 필요하면 별도 설계와 tiny CPU/CUDA RK 검증을 거친다. 생물학적으로 측정된 시냅스식이라는 주장은 하지 않는다. E267/E277의 선형 motor source-history 필터와는 다른 내부 재귀 비선형 가설이다.

## 입력과 고정 계산

승인된 E297 R1의22개 입력 집합과 producer provenance 검사를 재사용해 E296의 실제 저장 활동/roster/원래CSR/core24/원본 감사 및 source bytes를 검증한다. 입력 집합 전체를 SHA로 동결하되 baseline 중간값을 새 식의 상태로 섞지 않는다. 모델 construct/advance/CUDA/body/solver/fit0이다. E296 원본 출력 manifest에는 result가 없다는 저장 순서 계약과 별도 result 영수증 검증을 유지한다.

E296 control0..100/neural20ticks-per-control/physical5ms의 실제 관측 source7136 rate와 R를 unique ID map으로 정렬한다. 모든 값의 dtype/shape/유한성과 r≥0, R범위를 엄격 검증한다. 음수 rate를 clamp하거나 보간하지 않고 실패 영수증을 남긴다. u0 source5623의 R/g/old 계수1, 활성 source1513 및 edge117973, 원래 선택edge725849/7136source/전체core24 ID와 모든core향6172 계수1, 음수 선택 연결의u0를 검증한다. 원래CSR에 대한 selected pre/post/weight byteexact와 source→slot 대응을 검사한다. 기존 생산자 파일은 변경하지 않는다.

순수 `recovery_factor(rate, resource, utilization, tau)`는 작은 생성 fixture에도 사용 가능하며 float32 rate, float64 R/u/tau와 명시적인 호환 shape를 요구한다. 위 a/g64/g32/old32/Rdot를 저장한다. 순수 계산에서 u0는 R1이어야 한다. 식의 capped 항등식과 R≤g≤1을 float64의 고정 작은 연산 오차 안에서 확인하고 float32 g32≥old32/≤1을 별도로 확인한다. 같은 실제 rate에 대한 source `r*(1-old32)`, `r*(1-g32)`, 제거된 억제량을 모두 기록한다.

101행×7136 target에 대해 선택 양수 연결의 같은 저장 rate를 사용한 기존/새 식 손실을 float64 합으로 저장한다. 원래 weights32를 float64로 승격한 후 rate64*(1-factor32)와 곱한다. 모든 타깃을 기록하며 core/other 고정 분할만 요약한다. 선택 음수 계수는 원래 식 그대로임을 확인한다. 이 값은 중간 mask가 저장되지 않은 **unmasked same-state proxy**이며 실제 CUDA 전류나 전류 변화에 대한 인과·새 궤적 예측이 아니다. 중간 masks=1로 측정했다고 주장하지 않는다.

초기행0을 별도 제공하고 control1..100 전체 및 고정 구간1..10/11..20/21..100에서 활성1513만의 g32 unity 비율/억제 endpoint 수/한 번이라도 비자명한 source 수/1−g와g−R/r가중 손실을 요약한다. 비활성 source를 넣어 unity 비율을 부풀리지 않는다. 모든 source와 target별 원자료 및 순서 ID를 보존하되 큰 값 순위로 후보를 고르지 않는다. 저장된 E296 r/R에서 새 식이 작동하는지는 새 모델에서의 비퇴화 증명이나 startup/gait 개선 증거가 아니다.

## 검증·실행·판정

실제 gpt-6-sol/high가 `tools/analyze_c_banc_m3_recovery_factor_saved.py`, `tests/test_c_banc_recovery_factor_saved.py` 두 파일만 구현한다. Root FIRST→필요하면 같은 Sol R1→root SECOND→root 잔여 수정의 순서를 지킨다. E297의 safe/pin/provenance/load를 읽기 전용으로 재사용하고 E298 자체 gate/identity/output을 둔다. ROOT/OWNED/SOURCE_FILES/INPUT_NAMES/LIMITS/FREEZE_SCHEMA/REVIEW_SCHEMA와 gate/run CLI를 제공한다. 원본 gate/전역값을 monkeypatch하지 않는다.

생성 검증은45초/1GiB/CPU1, 저장 실행1회45초/2GiB/출력128MiB/CPU1이다. 순수 손계산, float32 unity rounding/음수 및 nonfinite 거부, u0/R1, unsorted IDs/중복/누락/clock/dtype/해시, 공유 source와 부호, zero rate에서 계수는 억제돼도 전류 손실0인 사례를 검사한다. tau=.15/u=.05/R=.4에서 r=100/200/300은 각각 g=.7/1/1, Rdot=2/0/−2이며 r0은g.4이나 전류0이다. 일정 r의 해석 R(t)≥Rinf이면g1인 수학 검증과 부동소수점 경계 차이를 명시한다. 실패·자원·출력 manifest를 남긴다.

Root는 별도 순수 손계산과 실제 저장값의 sparse 재계산, 각 고정 구간 집계 및 입력/output SHA를 감사한다. 모든 활성 endpoint에서 g32=1이면 관측 궤도의 퇴화로 기록하고 이 진단을 근거로 바로 신체 실험을 하지 않는다. 비unity라도 해당 rate가 모두0이어서 새 source 손실과 양수 target 손실이 모두0인 경우 전류에 대한 비자명성이 없는 것으로 구분한다. finite/bounded 및 실제 같은-state 양수 전류 손실의 비자명 조건을 만족해도 후속 별도 neural-engine 설계/tiny 자격 확인으로만 진전할 수 있다. 작은 효과 하나만으로 유망한 보행 후보라고 부르지 않는다. 후보 채택/보행 개선/생물학/원인/F1–F8/F6 노출 flags는 모두false이다.

원본 누적8078.0529733995245/8150초와 모든 과거 실패를 보존하며 이번 원본 예약0이다. M2/E150/UI/미노출 F6와 사용자 전체 예산 미지정 상태도 그대로다. 다음 원본 실행은 자동으로 따라오지 않고 새로운 모델과 자원 설계를 먼저 동결한다.
