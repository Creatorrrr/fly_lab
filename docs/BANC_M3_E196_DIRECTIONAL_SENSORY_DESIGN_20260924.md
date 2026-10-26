# E196: hook 움직임 방향 표현의 단일 대조

2026-09-24 원본 실행 전 설계. 기존 finite-goal F1–F8, M2/UI, 미사용 F6, 과거 실패를 유지한다. E195는 표집 복구 후에도 longest bout .215초, slip +1.254%로 기각됐다. 현재 원본 감독 누적은 4882.530568199720/4994초다. 이번 문서는 다음 후보를 사전 정의하며 자연 보행 달성을 뜻하지 않는다.

## 근거와 반증

현재 hook 특징은 `min(1, max(0, sign * velocity) / 20)`이다. E195의 실제100µs 특징을 전달했을 때도 hook132/137, club312/314 세포가 평가 구간 내내 rate0이었다. 저장된 경계의 drive/current/adaptation 검토는 낮은 구동 여유를 보였지만, 미관측 신경 substep이나 생물학적 역치를 측정한 결과는 아니다.

과거 M2 E14 감각 gain18→72는 slip +18.43%, 속도79.245%로 기각됐다. M3 E05의 고립세포 역함수 calibration도 전진10.781→9.542mm, net/path .343→.328로 실패했다. 이득 증가·역치 감소·재귀 억제 취소를 다시 시험하지 않는다. 현재 rate는 모델 단위이며 mV 또는 실측 spikes/s로 해석하지 않는다.

[Dallmann 등의 최종 논문](https://faculty.washington.edu/tuthill/docs/Dallmann_et_al-2025-Nature.pdf)의 방법은 hook 활성에서 속도의 방향을 이진 함수로 표현하고 이를 calcium kernel로 변환한다. 이전 수동 운동 자료에는 5°/s, 잡음이 더 큰 새 추적 자료에는 50°/s 문턱을 사용했다. 이 수치는 기록 잡음에 따른 모델 선택이며 생물학적 신경 역치나 절대 발화율 calibration이 아니다. 같은 연구는 자발적 운동 중 hook 억제를 보고하므로, 모든 감각세포의 활성 증가를 성공 조건으로 삼지 않는다. club은 양방향 운동/진동과 관련되어 이번에 바꾸지 않는다.

SNpp41 양의 속도/SNpp39 음의 속도라는 현재 대응은 MANC 연결에서 추정해 BANC cell type에 옮긴 음성 피드백 가설이다. BANC 자체의 직접 측정된 방향 선호는 아직 없다. 이번에는 그 가설을 고정하고 속도 크기 표현과 방향 표현만 구분한다.

## 후보와 비교

E195와 같은 native 구간 수집을 사용한다. 각100µs 표본에서 hook SNpp41은 `velocity > 5*pi/180`, SNpp39는 `velocity < -5*pi/180`이면1, 나머지는0이다. 정확히 문턱인 경우0이다. 미지 hook type은0을 유지한다. 각 구간 특징은 실제 dt 가중 평균이다. 5°/s는 낮은 잡음의 원래 자료에 사용한 값을 빌린 고정 공학 가정이며 보행 성적으로 조정하지 않는다. 단계별 문턱 sweep이나 실패 후50°/s 자동 재시험은 없다.

club 특징, 다른 수용기, gain18, 20ms 필터, 30ms 경계 LP, 원래 지연 drive, 운동3bank, 신경 역치·적응·재귀 연결, 전체158706 graph/CSR, body는 고정한다. 외부 위상/CPG/궤적/시간표를 추가하지 않는다. 같은 E150 neutral에서 시작한다. 유일한 E195 대비 인과 변경은 hook 특징 함수다. E193에 비해서는 native 수집까지 두 변경이므로 E193 비교만으로 hook 효과라고 부르지 않는다.

기존 E195의 완전 raw/start/final/validity와 복원 보고를 인과 비교로 재사용한다. 원본 종료의 JSON 실패와 당시 source archive를 보존한다. 효과 보고는 E195 대비와 E193 대비를 분리한다. **후속 진행 screen은 더 좋은 원래 E193 기준을 유지**한다: strict startup/health, 최장 지속 전진≥.3초, 전진≥E193의90%, net/path≥E193-.05, native 하중 가중 slip≤E193, 여섯 무릎95–5폭≥E193의90%. 전부 만족해야 후속10초 검증을 고려한다. 실패한 E195를 기준으로 문턱을 낮추지 않는다. 발화 증가만으로 채택하지 않는다.

## 실행 전 확인과 기록

부모는 E195 저장 궤적에서 이 방향 특징을 계산하여 실제 변경 크기와 조건부 drive 변화를 확인한다. 새로운 신경/물리 적분, fit, parameter search는 없다. 이 계산이 유한하고 대상 외 특징이 exact이며 hook 입력 차이가 RMS>1e-6일 때만 실제 후보를 실행한다. 이것은 효과가 전달될 수 있는지의 구현 전제이며 닫힌고리 결과 예측 또는 보행 합격 조건이 아니다.

기존 E195의 raw 관측·완전 checkpoint·실제 disk roundtrip·partial/error 보존을 재사용한다. 새로운 adapter/trial/patch identity에 방향 함수, radians 단위, 고정 deadband와 원래 motor identity를 포함한다. E195 checkpoint는 E196로 묵시적 복원하지 못해야 한다. pending50표본/시간/필터 상태가 완전히 보존되어야 한다. 첫 두control의 반환 drive·신경·모터·trace 및 처음100개 native 접촉은 E193/E195와 exact여야 한다. 독립 검산은 구현 함수를 재호출하지 않고 방향 식을 직접 계산한다. 합성 미래 복원과 실제 전체 모델 미래 복원은 구별하며 후자는 F2 후속이다.

E196 전용 helper/runner/tests2개를 새 파일로 구현한다. 기존 E195 소스·원본은 바꾸지 않는다. schema/path가 명확한 연구 runner 재사용은 허용하지만 전역 monkeypatch로 다른 실험의 gate를 우회하지 않는다. 보고 NumPy bool 직렬화 수정도 유지한다. 감각 request/drive/rate, 근육/명령, 보행을 분리 보고하며 신경 변화는 모델 단위다.

## 자원과 소유권

기존4994초 배정과 누적4882.530568199720초를 보존하고, 이미 승인된 유한 목표의 이 단일 가설에 **다음 지역 단계150초를 사전 추가 배정**하여 상한5144초로 명시한다. 원래 총 사용자 시간 제한을 새로 해석하거나 누적을 초기화하지 않는다. 잔여261.469431800280초 중150초를 예약한다. 준비/합성/문헌/검산 및 과거 미계측 비용은 별도로 남기며0으로 간주하지 않는다. 이번 배정이 이후 반복 탐색의 자동 예산이 되지 않는다.

원본1process/1arm/전체 BANC1·body1/자동retry0, CPU·BLAS1/GPU1, scientific3초/control600/neural12000/physics30000, settling2000 별도. 기존 E181 arm100초와 startup/health/resource 가드는 불변. 전체 내부140초, 외부145초+cleanup5초=150초, 출력2GiB/arm1GiB/free20GiB, 시작 직전2회 readiness 및 PID/create_time 소유 정리를 적용한다. 부모 source approval/freeze 후에만 원본을 실행한다.

부모 설계 → 실제 gpt-6-sol/high 최초 구현 → 부모 FIRST → 같은 Sol 첫 수정 → 부모 SECOND → 부모 잔여 수정 순서다. Sol은 새 helper/runner/tests2개, 부모는 계획/metadata/gate/감독/독립 검산을 소유한다. 실패 시 후보를 기각하고 경쟁 설명과 실제 정보를 기록한다. 짧은 screen은 F1–F8 또는 자연 보행 완료가 아니다. F6 노출0, M2/UI 보존, 전체 목표 active·미완료다.
