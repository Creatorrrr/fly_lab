# E352 완전 보행 주기 누락 원인 분리 설계

ROOT design v1. 실제 Astra 6/max 설계 리뷰 → ROOT 반영 → 실제 GPT-6.1 Sol/high 구현 → ROOT FIRST → 필요하면 같은 Sol의 ONE R1 → ROOT SECOND → 잔여 ROOT 수정 → 다음 단계. 새로운 구현 단계이며 E351의 사용한 R1을 다시 사용하지 않는다. 활성 자연 보행 목표와 F1–F8은 그대로다.

## 확인된 근거와 다음 결정

E351 별도 독립 감사가 실제 종료까지 통과했다. 원본 E350 worker와 과거 auditor의 technical-invalid는 유지했다. B/I/C 자료의 현재 source/action/geometry/native/dual-health/force/phase-plan 및 계산값 연결을 검증했다. E351 보고서 source2 a9a5804b/18e7c934, final decision a95841d5; 원본 신경/신체/GPU 추가 실행0. 누적9845.069767399496/9900초, 남은54.93023260050359초, 예약·실행0.

|대조|전진 mm|하중 가중 slip mm/s|최장 연속 전진 s|인정된 완전 주기|
|---|---:|---:|---:|---|
|B 원래 신경 출력 재생|1.721438635|2.046158820|.210|6다리 모두0|
|I 각 다리 형상 + 각자 원래 위상|1.552692778|2.018704367|.295|6다리 모두0|
|C 각 다리 형상 + 공유 LF 원래 위상|1.814082247|1.672598822|.360|RM1, 나머지5다리0|

C의 인정된 연속 전진 구간은0.5–0.86초와1.18–1.495초다. H348의 forward/slip/bout/health 기준은 통과했지만 all-six-cycle 기준이 실패했다. 어떤 원인인지 아직 확인하지 않았다. 짧은 연속 구간, 위상 정체/역행, 실제 발 왕복 운동 부족, 접촉과 미끄럼은 후보 설명일 뿐이다. E352는 새 learning이나 teacher 수치 탐색 전에 이를 분리하는 고정 read-only 진단이다. 진단을 위해 측정 기준을 바꾸거나 실제 보행 성공으로 재분류하지 않는다.

## 유한 입력과 원본 접근 경계

새 소스2만 소유한다: `tools/analyze_c_banc_m3_cycle_coverage.py`, `tests/test_c_banc_cycle_coverage.py`. 실행한 E350/E351/E349/core/gait/supervisors와 기존 raw/report/review/등록을 수정하지 않는다. Sol은 원본/기존 실제 NPZ의 해시·헤더·수치 읽기, 보존된 과학 효과 계산, body/model/GraphStore/NN/physics/CPG/CUDA/NVML 실행0. 구현 동안 PY/고정 유한 JSON/파일 stat만 허용하며 합성 fixture로 검증한다. ROOT가 리뷰를 반영한 뒤 별도 E352 등록을 만든다.

원본 데이터 읽기는 현재 source/freeze/qualification/ROOT review/reservation/정확한 sole CLI argv/신규 입력 등록을 모두 통과한 ONE E352 job 안에서만 허용한다. source2/연관 helper의 SHA도 시작 전에 검증한다. E351의 source-routed private gate를 우회하거나 `generated=True`로 실제 raw를 읽지 않는다.

허용 NPZ는 유한6개다: E351 등록의 original control/reference2개를 같은 selected-member/header/dtype/CRC/descriptor 방식으로 읽고 original native는 열지 않는다. 기존 E350 shape-plan1개와 B/I/C native3개는 E351의 qualified recovery-time SHA/bytes로 고정한다. 이것을 acquisition-time digest로 바꾸지 않는다. 새 native 전원6필드/15000표본/shape/dtype/finite/NaN mask/유효 slip/클록을 검증한다. 다른 NPZ/NPY/재귀 상태 파일 읽기는 허용하지 않는다. 각 파일 byte/hash/header/payload는 once cache로 재사용한다.

유한 JSON dossier: E351 final decision/report/원래 등록/reflection/current-source qualification와 감독 종료 증거, E350 B/I/C trace/receipt/evaluation/source-pose metadata 및 source policy를 기존 provenance로 고정한다. 이번 진단에서 모든 옛 자료를 다시 독립 감사한 것처럼 말하지 않는다. E351의 자격 범위에 의존하는 부분과 E352에서 직접 재계산한 부분을 구분한다. 실제 trace301rows/control5ms/모델1b2a8908/다리6x8/finite geometry/원래 action 및 source identity를 확인한다. 미승인 임의 경로와 Inf/NaN/빈 자료는 fail closed. 원본 숫자는 설계·생성 검증에서 읽지 않는다.

## 계산과 판정

1. 고정 gait analyzer `bcceb5d1...`의 실제 load-model arithmetic과 `phase_and_events`/`analyze`를 현재 E351 cached helper로 사용한다. ROOT가 source closure를 pin한다. 세 대조의 실제301row trace에서 data를 만들고 표준 bouts/eligible cycles/기존 gait report가 E351 independent metrics와 일치하는지 확인한다. `moving=window & speed>.5`, .3초 연속 구간, Gaussian20ms, Savitzky-Golay<=25ms, peak spacing40ms/prominence.02, time .5<=t<1.5를 그대로 유지한다. official cycles와 H348을 새로 완화하거나 합치지 않는다.
2. B/I/C의 모든 연속 moving run(짧은 것 포함)을 동일 time indices로 목록화한다. 각 run의 실제 duration/eligible 여부, 각 다리 AP 범위와 endpoint 값, 같은 peak filter/spacing/prominence로 얻은 local peaks/사이 최소점, 정규 complete-cycle count를 보고한다. eligible run의 count는 실제 `phase_and_events` 결과와 정확히 연결한다. 짧은 run의 local geometric events는 설명용으로만 별도 표기하며 성공 count에 더하지 않는다. 구간 경계에서 peaks가 생략될 수 있음을 기록한다.
3. 별도 설명용 fixed .5–1.5 window의 geometry에서 같은 peak filter를 한 번 적용하고 peak-to-peak pairs를 명시적으로 열거한다. pair가 하나의 eligible run 내부에 있는지, 짧은 run/비이동/속도 하락 구간을 가로지르는지 분류한다. whole-window filter context와 bout-local filter context가 달라질 수 있다. 이것은 causal exclusion count가 아니며 standard events와 동일 event라고 자동 대응하지 않는다. gap을 연결한 결과를 인정 주기로 재사용하지 않는다. 관찰 창 앞·뒤 partial-cycle은 complete로 세지 않는다.
4. original control/reference로 source phase encoder와 fixed shape-plan을 독립 재구성한다(E351 frozen math.fsum arithmetic). plan B/I/C/z/R/phi와 실제 trace command를 검증한다. 전 다리 source phi와 C 공유 LF phi의 principal-step signed net progress/positive and negative variation/max step/circular wrap 위치를 control time으로 보고한다. control tick100..300, reliability>=.5를 유지한다. c101..119 blend 구간과 c>=120 fully-shaped 구간을 분리한다. principal unwrapping은 multi-turn alias를 보증하지 않는다. 이것은 source-output embedding phase이며 신경·실제 발의 위상 측정이 아니다.
5. 각 C eligible run 및 전체 fully-shaped 구간의 phase span/역행/피크 위치/geometry-cycle coverage를 같은 타임라인에 놓는다. phase가 충분히 진행해도 발 왕복이 없는지, run 사이에서 주기가 잘리는지, source phase의 정체/역행이 있는지를 근거와 미확정 부분으로 구분한다. 확인되지 않은 임계값이나 후보 자동 선택은 만들지 않는다.
6. 고정±20ms 주변을 포함해 C speed-threshold crossing times와 native contact를 연결한다. native load>=.02의 leg별 support/0..6 supporting-leg distribution, 하중 가중 slip과 missing count, upright/nonfoot를 보고한다. native15000 clock와 control301 clock을 실제 time으로 정렬하며 마지막 boundary 밖 자료를 만들지 않는다. adhesion을 포함하는 load나 contact transitions를 biological stance/step count로 부르지 않는다. 관찰상의 동시성은 인과나 힘 제어 성공이 아니다.

진단 산출물은 fresh `verification/banc-natural-gait-m3-20260919/e352-cycle-coverage/report.json`이다. JSON에는 actual source/reader/inputs/selected-member provenance, once reads, 고정 표준 결과, 설명용 분류, phase/geometry/contact 시간 자료와 unavailable/first/secondary errors, 잠정적 다음 의사결정 자료를 포함한다. 복구된 E351 H348은 partial 그대로 보존하고 candidate_adopted/F1_F8_complete/biology_proven=false, new_body_neural_physics_CPG_GPU_calls0. success/error 모두 strict JSON/caps/deadline/final PID를 따르며 late failure는 provisional success보다 우선한다. 원본 E350 실패는 변경하지 않는다.

## 자원과 생성 검증

Sol 생성 전체 실패 포함20초/8MiB, ROOT FIRST/R1 이후 생성20초/8MiB. CPU1/host2GiB. 실제 PID/create-time 종료 증거와 정확한 current pre-execution SHA/argv를 남긴다. 설명용 peak attribution은 실제 frozen analyzer가 gap을 연결하지 않는 합성 양성·음성 검사를 포함한다. 알려진 periodic6-leg trace의 eligible-cycle count 일치, 짧은 run 미인정, gap-crossing 설명용 이벤트가 official count에 더해지지 않음, phase wrap/역행/blend alignment, native/control 마지막 time alignment/유효하지 않은 slip과 zero support, strict serializer/Inf/NaN 및 private gate 전 원본 read0, 임의 경로 rejection을 검증한다. 동작을 그대로 복사한 기대식만 검사하지 않는다.

ROOT가 current qualification/source freeze/review 후 one original6초/internal5초를 prospectively 예약한다. 누적9845.069767399496/9900초, original_evidence_calls1/retry0/body-neural-physics-CPG-GPU0. supervisor E267 `1d868250...`와 creation-time observer `cf7b0473...`를 그대로 사용한다. output<=4MiB/stage<=64MiB/cache<=64MiB. ENTRY에서 import/read/hash/calc/flush/fsync/close까지5초; source metadata 실패와 timeout/late failure를 보존한다. 원본 실패 시 동일 job cached failure만 허용하고 두 번째 read/audit/body retry는 없다. 실제 qualified observer inclusive 비용을 once charge하며 nested 시간을 중복하지 않는다. unknown closure는 보수적으로6초 차감한다. 9900초를 초기화/연장하거나 수치 기준/캡을 완화하지 않는다.

## 다음 단계

ROOT가 실제 diagnostic report를 검토한 뒤 개선 후보를 설계한다. 가능한 비교는 fixed shared cadence가 source phase 불규칙성과 연속 bout/foot-cycle coverage를 개선하는지 확인하는 동일 신체 teacher 진단이다. 아직 cadence 값·body 실행·신경 학습을 선택하거나 승인하지 않는다. 실제 몸이 인정 주기를 만들지 못하는 상태에서 BANC 신경 조정으로 책임을 넘기지 않는다. 다음 개발도 Astra 설계 리뷰와 반영, Sol 구현과 ROOT 리뷰 순서를 따른다. 자연 보행 완료는 동일 BANC 후보의 실제 F1–F8 전체 증거가 필요하다.

## ROOT의 실제 Astra 6/max 리뷰 반영

실제 리뷰 MD `2e8b3063a1ad57d4fc0eb1bf176124ef8c639b928f9239cc8c308e991f5df87c`, JSON `d9ea28be5a529e82ecf1ee9c9170a8c72aa109c7608df590146434f19742f3d9`. 보완 필요 M1–M3를 모두 반영한다. 원 v1 `d2a55866...`는 별도 snapshot으로 보존한다. 리뷰는 유한8파일/static만 사용했으며 원본/trace 숫자 계산과 source/test/body/NN/GPU 실행0이다.

M1: canonical run은 [lo,hi), duration=(hi-lo-1)*dt, eligibility>=.3-1e-9다. 표준 `phase_and_events` 결과가 official이다. 5ms의 고정 SG window5를 1–4표본 설명용 run에 적용할 수 없으므로 AP raw endpoints/range는 보고하되 filtered peaks/pairs는 None+reason/filter_unavailable_mask로 남긴다. 작은 window/padding/0 대체를 사용하지 않는다. Hilbert amplitude/trim은 phase valid mask이며 현재 event count의 추가 필터가 아니다. 예상 masked phase NaN/invalid slip은 None+mask/count, unexpected NaN/Inf·mask 모순은 technical. whole-window pairs는 standard cycles와 자동 동일시하거나 합산하지 않는다.

M2: standard .5<=t<1.5와 phase c100..300의 마지막1.5초 boundary를 구분한다. 포함된 run indices만 coverage에 쓰며 native/phase 마지막 샘플로 eligible interval을 추가하지 않는다. phi는 cycles 단위의 source-output embedding이다. 각 동일 연속 구간에서 d=((phi[k]-phi[k-1]+.5)%1)-.5, net=sum(d), positive=sum(max(d,0)), negative=sum(max(-d,0)), u=cumulative principal increments, span=max(u)-min(u)를 사용한다. gap을 건너 합산하지 않는다. 반회전 tie/multi-turn alias는 direction_ambiguous로 기록한다. c101..119 blend와 c>=120 fully-shaped 내부 증분 및 그 경계를 가로지르는 증분을 별도 명시한다.

Crossing은 분석 창 안의 인접 actual speed>.5 상태가 달라진 sampled bracket [t[k-1],t[k]]만 세며 anchor=t[k]로 고정한다. 관찰 창 진입·종료/censoring을 speed crossing에 더하지 않는다. native context는 [anchor-.02,anchor+.02), 실제 관찰 domain0<t<=1.5와 분석 domain[.5,1.5)의 교집합, native time을 사용한다. 고정 시계 round(t/1e-4) 및2e-8 clock tolerance를 검증해 inclusive boundary의 부동소수 drift가 frame count를 바꾸지 않게 한다. trace/native 시간이 단순 정수 index처럼 같다고 가정하지 않는다. context별 count와 clipped endpoints/원래 bracket를 보고하고 전역 context union에는 중복 표본을 더하지 않는다. occupancy 분모는 해당 context native 표본 수, support>=.02 leg count0..6. weighted slip=sum(load*slip)/sum(load), support missing이 하나라도 있거나 support0이면 None+reason. contact observations와 인과/생물학적 stance는 구분한다.

M3: frozen E351을 수정하지 않는다. E352 own6/5 gate 뒤 exact six-path reader adapter만 활성화한다. old E351 gate/full recover/global mutation/generated=True actual read는 금지한다. old register의 native/shape-plan null SHA 대신 같은 path/bytes에 qualified E351 report recovery-time SHA를 결합한 E352 registration을 ROOT가 만든다. acquisition digest라는 명칭을 붙이지 않는다. E351 report의 external_completion_qualified=false/body_evidence_qualified=false는 유지하며 final decision의 report/identity pin과 실제 qualified 종료로 의존 범위를 상속한다. 읽지 않는 forwarded-orientation raw 건강을 E352 직접 검산으로 주장하지 않는다. E352가 직접 재계산한 geometry/phase/native와 E351에서 상속한 부분을 명시한다. success/error caps/ENTRY-through-close와 exit/PID/late-error 우선은 작은 generated 허용에 포함한다. 별도 감독기나 추가 campaign은 없다.
