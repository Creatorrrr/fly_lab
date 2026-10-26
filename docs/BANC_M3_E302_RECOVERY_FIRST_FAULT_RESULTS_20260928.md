# E302 최초 RK 위반 진단 결과

2026-09-28 root 판정: **진단 유효, E301 후보 보행은 여전히 미검증**. gpt-6-sol/xhigh가 구현하고 root FIRST에서 65 fake tests/Ruff 및 독립 masked-source 중단 검사를 통과했다. FIRST 이후 수정 요청은 없었다. 실제 저장 자료 로더와 SHA gate도 통과했다.

같은 E301 후보를 신체 없이 한 번 실행했다. 9 neural ticks가 완료됐고, 10번째 tick의 stage index 3에서 source `96198`의 rate가 `-3.3061957083191373e-7`이 되어 최초 실제 flag **4(active rate)**를 기록했다. 원래 factor kernel까지만 실행하고 그 단계의 CSR 전에 멈췄다. 원래 E301의 누적 flag29와 구별한다.

앞선 39개 factor stage에서 7,136개 계수는 전부 정확히 1이었다. 최초 위반 때 R은 전부 유효한 [0,1]이고 u=0인 source는 R=1이었다. 따라서 **회복 억제의 전달 변화가 시작되기 전에 원래 rate RK4 중간 상태의 음수가 엄격한 resource 입력 조건과 충돌했다**. 이것은 생물학적 발화 억제 실패를 보여주는 실험이 아니다.

Root가 저장된 full rate/adaptation/drive/mask와 원래 CSR 한 행을 원래 warp 순서로 독립 재계산했다. Source96198의 stage1 current는 `7.504231929779053`, target은 `0.004231929779052734`; stage2 current는 `7.497054576873779`, target은 0이었다. 원래 tau=.02, dt=.00025에서 zero-base RK stage2 rate `2.6449562938068993e-5`와 stage3 음수는 각각 저장 float32와 byteexact였다. Zero base는 이 관계에서 재구성한 값이며 tick9 전체 endpoint를 직접 저장했다고 주장하지 않는다. 오류는 subnormal보다 훨씬 큰 정상 float32 값이므로 FTZ 경계 문제가 아니다.

E302 held drive는 실제 neutral/final에서 읽었다. 원래 whole 초기 상태가 같고 첫 control의 생산 코드가 동일하며 candidate partial adapter와 neutral final adapter의 encode 상태도 같다는 근거로 후보 첫 입력과의 동등성을 추론했다. 실패 후보에서 직접 저장한 drive라고 표기하지 않는다. E302 그래프 replay는 해당 인스턴스에서만 잠시 해제했고 원래 커널/순서/dt를 유지했다.

원본 supervisor `10.817045499978121`초를 전부 이월했다. 누적 **8144.108421799498 / 8210초**, 잔여 **65.89157820050241초**, 예약0이다. 직접 신경 생성자1, science9완료+1partial, factor40, 실행 setup40/capture기록20, unknown0, body/physics/solver0이다. 실제 내부 wall6.337초, tree RSS1,521,991,680bytes; self peak1,593,167,872bytes, Torch peak reserved260,046,848bytes, sampled additional GPU314,572,800bytes. 전체 GPU peak의 연속 측정은 아니다. 모든 소유 PID 종료를 확인했다.

Root raw audit는 108 source pins/802 input pins, 실제 raw6파일 및 별도result hash, 비용/단계/초기 상태·형식·factor/원래CSR 재계산을 확인했다. `verification/banc-natural-gait-m3-20260919/parent-e302-original-raw-audit.json` PASS는 이 진단의 유효성을 뜻한다. E301의 `run_valid=false`, startup·short unavailable과 F1–F8/F6 미완료·후보 미채택을 유지한다. Root audit는 saved-only .770초/추가 원본0이다.

Agent의 모든 생성검사 비용22.0085391초 및 이전 ownership 회귀 실패6사례, 별도 saved-only6.2355994초를 보존한다. Root FIRST 감독6.169421초, 실제 input gate 및 준비 비용은 별도 receipt에 기록했다. 모든 frozen E299/E300/E301/E302 source와 raw는 유지한다.

다음 변경은 음수를 무시하거나 0으로 잘라 통과시키지 않고 **중간 단계의 물리적 범위를 보존하는 적분법**을 별도 identity로 구현하는 것이다. 기존 근육 보정·기준·유효성 검사를 유지하고, 수치 적분법 변경의 오차와 실제 보행 효과를 별도로 검증한다. 자연 보행 달성 주장은 없다.
