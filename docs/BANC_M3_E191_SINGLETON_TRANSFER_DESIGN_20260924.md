# E191: 단일 운동뉴런 근육의 선형 전달 가정 해제

부모 설계, 원본 수치 실행 전. 전체 finite-goal과 F1–F8, M2/UI, 미사용 F6, 과거 실패와 누적 비용을 유지한다. 직전 E189/E190은 실제 원자료와 독립 검산을 얻은 progress지만 보행 개선 후보는 없었다.

## 선택 근거와 질문

E107은 같은 신체에서 다리 내부 명령 파형을 바꾸면 연속 전진과 적격 주기를 회복할 수 있음을 보인 외부 교사 진단이다. E163의 강한 신경 위상 결합, E181–183의 시점 변경 실패 및 E190의 비고유적 하강/제동 관계를 고려해 작은 이득·지연 순회를 반복하지 않는다. E150 `MonotoneRecruitmentAdapter`는 96근육 중 LTM6과 단일 MN 근육13의 합집합19를 선형10ms로 고정한다. E147/E169/E175의 보호된 양수 전달 가족도 이19개를 바꾸지 않았다. 단일 MN 소유권이 선형 근육 전달을 요구하는 것은 아니다. 그러나 이것이 현재 실패 원인이라는 증거도 아직 없다.

H191: LTM6은 그대로 두고 단일 MN13개도 기존 나머지 근육과 같은 비음수 단일-knot 전달을 사용하면, 기존 해부학/초기 자세/근육 및 MN 기여 보호 안에서 같은 파형 오차20% 감소가 가능한가? 경쟁 설명은 이 추가 자유도가 있어도 현재 신경 신호로 필요한 파형을 표현할 수 없다는 것이다. 이 단계는 저장 신경 출력의 조건부 적합이며 보행·생물학 검증이 아니다.

독립 actual gpt-6-sol/high 소스 감사도 이 범위 차이를 확인했다. E158–E160은 RM singleton67/MN144018 하나의10→40ms 변경을 시험했고 startup max.047299>.04로 신체 전에 기각됐다. 이를 보존하며 singleton 변경이 전부 미시험이었다고 주장하지 않는다. 이번의 nonlinear90-muscle family는 별도이고 전체 F1–F8에 singleton 선형 고정 요구는 없다.

직접 Hill/힘 구동 전환은 지금 선택하지 않는다. 현재 moment는 상대 방향이고 여섯 다리의 길이 경로·힘 단위가 교정되지 않았다. E24/E26 활성 강성, E58 활성 감쇠, E64 비대칭 활성의 실패를 보존한다. [2025 musculoskeletal 연구](https://arxiv.org/html/2509.06426v1)는 길이/모멘트암과 물리 파라미터를 별도로 추정·적합했고 환경 접촉 부재를 한계로 밝힌다. LF 자산의 수치를 여섯 다리에 그대로 이식할 근거로 삼지 않는다.

## 범위와 고정 식

새 원본 CPU 프로세스1, Clarabel solve1 이하, 자동 재시도0. 새 신경·신체·상태 생성/적분, CUDA, history/EMA 갱신, F6, 교사 생성, native 후보 수출은0이다. 소유 파일은 `tools/fit_c_banc_m3_singleton_transfer.py`, `tests/test_c_banc_singleton_transfer.py`다. 구현은 actual gpt-6-sol/high, 부모 FIRST → 같은 Sol 첫 수정 → 부모 SECOND 및 잔여 수정 순서다. 기존 코드와 자료는 읽기 전용이다.

입력은 E169 `pre-solve-raw.npz`의 원형 bank/activation/original_activation/MN/clock/availability, E150 adapter/parameter/state provenance, 기존1824 target/1823 velocity 및99 startup이다. E169 전체 실행은 native velocity 재구성 오차로 invalid였음을 유지한다. 새 로더는 그 solver/candidate/성공 표지를 사용하지 않고 저장된 입력의 출처, 별칭 일치, 완전성, 시간, 소유권, 현재 E150 출력 재구성을 독립 검사한다. 참조는 이미 노출된 개발 자료이며 미사용 확인으로 세지 않는다. 부모 freeze는 실제 입력/소스/설계 SHA를 지정하고 source-review SHA gate를 별도로 요구한다.

입력의 정렬된 motor_ids는 실제 BANC graph index391개이며0..390으로 다시 번호를 매기지 않는다. 누적 물리 시간 비교는 원래 E164/E169 freeze의 clock_atol2e−8초를 유지한다. 이는 새로운 command/velocity 수치 허용과 별개의 원래 기록 시계 계약이다.

해부학391 MN/96근육/6다리17,15,16,17,15,16과 moment/rest/.02를 유지한다. 고정은 원래 LTM6뿐이다. 가변은 나머지90근육/343MN이며 기존330MN와 새 singleton13MN을 구분하여 원자료에 저장한다. MN의 실제 graph-column 오름차순, filter0/1/2, low/high 순으로1029pair/2058coeff다. 입력 bank5/10/40ms와 kappa는 고정한다. 각 MN/filter basis는 `min(bank,kappa)`와 `max(bank-kappa,0)`다. w0는 기존 E150 slopes에서 같은 위치를 추출한다. singleton w0는 filter1의 low/high=1이고 나머지0이다. 새로운 knot/tau/gain 선택은 없다.

각 근육의 후보 활성은 native 순서 `sum(low*bank) + sum((high-low)*max(bank-kappa,0))`로 계산한다. LTM6은 원래 activation exact. candidate command는 각 tick에서 `rest + .02*(moment @ activation)`이다. t0는 decode 전 원래 관측이므로 exact 유지한다. 데이터 적합은 `q_recorded + .02*M*sum((w-w0)*basis)`의 affine 표현을 사용하며 w0에서 exact baseline이다. 기존 fixed singleton의 original10ms와 bank10ms의 산술 차이를 포함해 baseline과 후보의 native 활성/명령/속도 대응을 별도로 저장한다. 실행 가능한 후보 판정은 실제 native 식에서 다시 계산한 모든 원래 기준을 요구한다.

## 비교·보호·수치 계약

명령1824tick100..1923, 인접 velocity1823, startup1..99와 목표 배열을 그대로 사용한다. 전체 command RMS≤원래E150의.8, 전체 및 각 다리 velocity RMS 비증가, 각 다리 command RMS 비증가, startup q RMS≤.018/max≤.04rad, LTM RMS≤.009/max≤.025를 모두 요구한다. 전체 F1–F8 기준은 변경하지 않는다.

실제 w의 범위[0,2] exact, 각 MN sum_filter(min(low,high))≥1e−6/n_owned, 근육 sum min≥.5 및 sum max≤2(물리 보호 atol1e−12). 새 singleton에도 같은 보호를 적용한다. support 외0과 LTM 기울기/활성 불변, finite/nonnegative, 단조 양수 전달을 확인한다. 작은 음수 solver 결과를 clipping하지 않는다.

SOCP 변수는 w2058, ell1029, u1029, t1=4117. box, 각 pair ell≤low/high≤u,343 MN floor,90 lower/upper budgets,8316 startup ±행, 기존15 SOC(전체2+각다리12+startup1)를 구성한다. margin t∈[−1,1]을 최대화하고 기존 기준은 t=0이다. 실제 물리 family는 위 원래 조건이며, solver에는 사전 inner guard g=1e−8을 사용해 w∈[g,2−g], ell/u∈[0,2], epigraph RHS=−g, floor/근육 budget RHS에−g를 적용한다. 이는 경계의 작은 음수 반올림을 사후 clipping하지 않기 위한 수치 보호이고 조건 완화가 아니다. baseline 재구성은 항상 원래 w0이다. 가능점 witness는 각 MN의 근육 소유수 n_m에 대해 v=.999*w0+.001/(3*n_m), ell=min(v)−g, u=max(v)+g, t=−.251로 고정한다. raw/slack을 저장한 뒤 검사하며 실패하면 witness를 바꿔 재실행하지 않는다. inner 문제의 실패를 전체 물리 family 배제로 주장하지 않는다.

다리별 command/velocity/startup18개 thin QR, active 열/offset/직접 norm을 보존한다. 검산 벡터는 w0,zero,.5+.5*cos(arange(2058)*.173). QR norm 허용은1e−12+1e−10*direct norm이다. native 활성/명령 재구성 atol1e−12, velocity 대응 atol5e−10rad/s를 사전 사용한다. 후자는5ms 차분의 float64 합산 오차를 분리하기 위한 새로운 수치 계약이다. E169의 기존1e−12 실패를 성공으로 고치지 않으며 실제 보행/파형/시작 문턱은 완화하지 않는다. native 실제 명령으로 직접 평가하므로 affine 통과만으로 승격할 수 없다.

Clarabel0.11.1, CPU BLAS1/solver max_threads1, qdldl, max_iter100/time_limit20s, gap_abs/rel/feas/infeas_abs/rel1e−10, 나머지 defaults. solver 생성과 solve를 합친 실제 wall도20초 이하여야 한다. solve1회만, 초기/최종 직접 검산은 fitting 아님. primal/slack/dual과 원래 행렬을 보존한다. 원본 실행 전 부모 SECOND 검토에서 수치 판정을 고정한다: 반환 Solved/AlmostSolved 여부와 별개로 primal/dual 최대 잔차1e−7 이하, 선형 slack≥−1e−8, SOC 위반≤1e−7을 모두 요구한다. 유한 후보와 원형 반환을 먼저 저장한 뒤 시간/수치 기준을 검사한다. 생성시험을 통과시키기 위해 기준을 완화하지 않는다. 이 단계에서 엄밀한 전체-family 불가능성은 주장하지 않는다. 모든 gate 통과는 conditional proposal; 미달은 valid_rejection_inconclusive이며 반환 margin과 잔차를 보고한다. 데이터/수치/저장/자원 오류는 invalid_not_tested다.

## 자원·기록·다음 결정

기존 원본 누적4655.7939677997165/4994초, 잔여338.2060322002835초를 유지하고 이번 감독95초를 예약한다. 내부80/외부90/감독95초, solve20초, CPU1/GPU0, 출력256MiB/free2GiB. 실패 비용도 누적하며 한도를 새로 시작하지 않는다. 소스 작성·생성시험·부모 검산은 별도 실제 시간으로 기록한다. 초기 metadata 조회에서 NumPy private header API 오류1회가 있었고 payload/solve/모델 실행은0, 정확한 내부 경과는 미계측이다. 이어 public header metadata 조회는.023372700001345947초였다.

원형 입력 필수 subset, mapping/basis/w0/baseline 재구성,18QR/sparse problem/witness는 solve 전에 저장한다. 반환 solver 원형은 후속 gate 전에 저장하고 실제 candidate 활성/명령/metrics/gates 및 오류를 보존한다. 입력/소스 해시를 전후 확인한다. 결과 최종 게시 실패는 성공으로 남기지 않는다. 감독은 자체 wall/PID/종료/디스크 원자료를 저장하며 제한 초과 시 해당 자식만 종료한다.

조건부 제안이 통과하면 부모가 같은 계수의 실제 adapter·zero/cut/snapshot 복원과3초 wholeBANC 비교를 설계한다. 이는 자동 실행 권한이 아니며 해당 source review와 남은 동일 자원을 확인한다. 기각이면 실제 원래/확장 표현의 잔차를 근거로 다음 신경 또는 전달 구조를 선택한다. 같은 singleton 계수·knot/tau 순회는 이번 결과만으로 반복하지 않는다. 후보 미채택, 전체 목표 active/incomplete다.
