# E267 — 기존 premotor→MN 간선의 source별 인과적 여과

2026-09-27. 같은 M3 finite-goal의 구현 단계다. F1–F8, E150 최선 부분 결과, M2/활성 UI, 미노출 F6와 과거 실패를 보존한다. 런타임 구현 검증을 보행 성공으로 집계하지 않는다.

## 방향 결정

E240은 같은 신체와 원래 힘줄에서 교사 보행의 수행 가능성을 보였다. E246의 내부 51개 상태는 교사 명령을 구별하지만 MN 발화와 합산 E/I/drive 표현은 고정 기준에 미달했다. 표현 차원 차이가 있으므로 인과적 정보 손실의 증명은 아니다. E247에서 기존 내부→free MN 경로는 330/343개 MN, 모든 90개 대상 근육군에 도달한다.

E128의 정적 간선 효능+MN 적응 공동 적합, E168–175의 출력 이력, E218–244의 합산 입력 반응, E251/E257의 감각 간선 복원은 각각의 기각을 유지한다. E265/E266은 국소 전류 진단이며 지속 보행 개선이 아니다. 따라서 그 국소 감각 진단을 계속 세분화하기보다, 합산 **전**의 source별 입력 동역학을 실제 재귀망에 구현한다. 실제 gpt-6-sol/high의 읽기 전용 방향 검토를 참고했으며 선택과 설계 책임은 현재 세션에 있다.

이는 생리 실측의 복원이 아니라 허용된 세포/시냅스 동역학에 대한 공학적 가설이다. 새로운 BANC 간선, 외부 위상, 명령 궤적, root 힘, 교사 런타임 주입은 없다.

## 구현할 유일한 변경

기준은 `MotorTimescaleNetwork`의 원래 packed CUDA RK4와 E150의 canonical 근육 전달이다. 선택한 기존 nonzero CSR 간선 e=(pre, post)에서만 다음 입력을 사용한다.

`I_e = W_e * mask_pre * (c_e0*r_pre + c_e1*h_pre,fast + c_e2*h_pre,slow)`

`dh_pre,k/dt = (r_pre - h_pre,k)/tau_k`, `tau=[0.005, 0.040] s`.

- 상태는 선택 간선들의 unique source별 2개다. 동일 source의 여러 축삭 끝은 이 상태를 공유하되 간선마다 세 계수가 다르다. 미선택 source에 상태를 추가하지 않는다.
- 계수는 float64 입력을 검증하여 한 번 float32로 바꾼다. 세 계수 각각 ≥0, 합은 [0.125, 8]이다. 변환 후도 같은 범위를 만족해야 한다. 이 범위는 첫 구현의 공학적 제한이며 측정값이 아니다. 원래 W의 부호·CSR topology는 보존한다.
- `c=[1,0,0]`은 모든 신경 rate/adaptation/drive/mask/tick에서 원래 실행과 bit exact여야 한다. 이를 위해 이 계수의 간선은 원래 packed 연산식을 그대로 실행한다. 새 이력 상태는 수동 관측만 하며 원래 상태에 영향이 없어야 한다.
- 다른 간선은 원래 CSR 순서와 원래 식을 그대로 사용한다. 선택 간선의 세 항 계산 순서와 float32 cast, FTZ를 precision contract로 명시한다. 동역학의 source 이력은 float64이며 원래 float32 RK 단계 rate를 입력으로 사용한다.
- 신경 rate/adaptation과 이력 상태를 **같은 네 RK4 단계**에서 적분한다. 단계 rate가 다음 단계로 덮어써지기 전에 해당 source 이력 미분을 계산한다. control endpoint만 여과하거나 한 control 늦은 값을 사용하는 것은 이 설계가 아니다.
- 원래 mask는 instantaneous 및 두 history 항 모두에 현재 시점에 적용한다. source 차단 시 남은 history가 출력을 누출하면 안 된다. history 자체는 실제 rate를 기억한다. MN 출력 차단의 기존 의미도 유지한다.
- 선택 간선의 post는 제공한 free MN roster에 속해야 한다. source와 post는 실제 CSR ID여야 하고 원래 nonzero weight bytes를 확인한다. 정렬·중복·범위·비유한 값·잘못된 dtype을 묵인하지 않는다.
- 파라미터는 인스턴스가 소유하며 alias mutation을 방지한다. topology/계수/두 tau/원래 모델 identity/정밀도 계약이 새 identity와 checkpoint에 포함된다. native CUDA graph는 안정된 pointer를 사용한다.
- reset, snapshot, restore, 저장 미래를 지원한다. restore는 모든 inherited metadata와 state 및 새 state를 검사한 뒤에만 live tensor를 변경한다. 잘못된 restore가 부분 상태를 남기면 안 된다. 원래 checkpoint의 이식은 별도 명시적 함수로 하고 일반 restore에서 모형 identity를 무시하지 않는다.
- 명시적 원래 checkpoint 이식에서는 두 이력을 저장된 해당 source rate의 float64 복사로 초기화한다. 이는 일정한 과거 입력을 가정하는 초기 조건이며 미래/교사 입력을 읽지 않는다. reset은 원래 rate와 두 이력을 0으로, 일반 restore는 저장된 이력을 그대로 복원한다. 최초 정적 준비 뒤 이 초기 조건을 명확히 했고 변경 전 설계를 verification의 `e267-design-before-initialization-clarification.md`에 보존했다. 원본 실행은 아직 0이다.

구현 파일은 `tools/banc_m3_source_filtered_synapses.py`, 테스트는 `tests/test_c_banc_source_filtered_synapses.py`다. 기존 제품/실험 모듈을 수정하지 않는다. root가 선택 자료와 원본 실행기/판정 기록을 소유한다.

## 검증과 다음 실제 보행 판별

1. 실제 gpt-6-sol/high가 구현한다. 현재 세션 FIRST에서 소스와 독립 NumPy 계산을 검토한다. 필요한 첫 수정은 같은 Sol, SECOND 후 잔여 수정은 현재 세션에서 한다.
2. 생성된 작은 연결망으로 source 구별, 동일 source의 다중 간선, 양/음 부호, 네 단계 이력 결합, 중립 bit exact, mask 즉시 차단, native captured advance, reset/비영 상태 restore/손상 checkpoint 무변경을 확인한다. 이 검사는 실제 BANC나 보행 성과가 아니다.
3. 실제 E150 CSR와 E245의 고정 51개 source에서 free MN으로 향하는 기존 간선만 선택한다. 전체 신경의 기존 저장 시작을 명시적으로 이식하고, 중립 계수 아래 기존 B 신경 재생 0.5초를 비교한다. 시작·끝 원래 상태 및 매 control의 MN/core/기존 관측과 drive를 검사한다. 이력은 별도 보존한다. 새 neural model과 계수가 없는 기존 모델을 혼동하지 않는다. 원본 실행 전 source/input freeze와 구체적 실행·자원 계약을 추가한다.
4. 이후 학습은 중립 대비 source별 계수의 역할을 비교하며 교사 자료를 학습에만 사용한다. 조건부 적합을 채택 근거로 삼지 않고, 실제 재귀망/신체에서 얻은 입력으로 다시 판별한다. 같은 family의 무한 계수·tau 탐색을 하지 않는다. 정확한 학습 목적·자료·횟수·중단·비용은 첫 학습 전 별도 고정한다.
5. 첫 후보는 E150 control0 같은 시작의 실제 신체 0.5초 strict startup을 확인하고, 통과하면 같은 실행을 1.5초까지 진행한다. 전진/slip 비악화와 bout≥0.3초 및 실제 주기 개선을 요구한다. 명령 RMS만 좋아진 후보는 확대하지 않는다. 통과 후보의 원래 10초 F3–F5, F2/확인 F6/기능 F7/통합·영상 F8까지 같은 후보로 이어간다.

현재 단계의 성공은 새 동역학이 실제 CUDA 재귀 경로에서 정확하고 복원 가능하다는 것뿐이다. 양성 보행 후보가 없으면 M3는 계속 미완료다.

## 자원

원본 보수 누적 `7126.338238699598 / 7138 s`, 잔여 `11.66176130040185 s`를 보존한다. 현재 문서 작성과 구현은 원본 모델 실행 허가로 취급하지 않는다. 생성 CPU 검사는 별도 실제 시간을 기록한다. 생성 CUDA 검사는 root의 개별 실행 예약 뒤 단일 GPU 프로세스로만 실행한다. 실제 원본 재생 및 fit/body는 소스 검토 후 필요한 작업과 상한을 명시하고 기존 누적을 이월한다. 사용자 전체 시간/토큰 예산을 새로 설정하거나 누적을 0으로 초기화하지 않는다.
