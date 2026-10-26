# BANC M3 실행 계약과 시도 데이터 분리 — ROOT 설계

실제 목표는 동일 BANC 후보를 control0→99→300의 같은 FlyBody 인스턴스에서 감각 폐루프로 평가하는 것이다. F1–F8 전체, 해부학 provenance, M2/UI, 실패 후보, 미사용 상태 노출, 교사·외부 CPG·위상·궤적·root 교정 금지는 유지한다. 이 변경은 실행 복구이며 자연 보행의 증거가 아니다.

## 현재 증거와 방향 선택

P30 Ruff는 실제 exit0/1.772초로 끝났다. scanner는 exit1/2.01초로 끝났고 비용은 한 번 처리됐다. 현재 폐쇄111 원장에는 61 calls/27 passes가 있으며 reservation/pending/hold가 없다. 생성 shared680.0860156/1700초, E360 local569.129/1589초다. 원본 cumulative10020.115960899504/10200초, stage126.361/300초, E360 local0/20초는 유지한다.

직접 원인은 `_CURRENT_P30_PINS[32,41]`의 정상 빈 P27/P29 stderr를 `lib.read_spec()`에 넘긴 것이다. 그 함수는 bytes>0를 요구한다. 정확한 완료 로그에는 이미 `historical_p12_stream_bytes()`의 bytes≥0, exact descriptor, bounded read/reread, UTF8, root, SHA, historical ownership 검증이 있다. 일반 입력의 bytes>0 규칙을 완화하지 말고 두 로그와 이후 동종 소비자를 그 기존 경로로 연결한다. P30 source/freeze/collector/실패 비용을 보존한다.

별도의 문제는 새로운 실행마다 숫자별 정책, 활성화, 과거 비용 구간, 소스·검사·Native caller를 여러 곳에 복제하는 구조다. 최근 변경은 actual104/108/109, default28, 빈 로그 등에서 연달아 실패했다. P31 번호를 붙여 같은 구조를 복제하는 대신 고정된 안전 계약과 현재 실행 manifest/상태 데이터를 분리한다. 현재 복잡한 구현 전체를 다시 만드는 범용 프로젝트는 범위가 아니다.

## 고정 계약과 데이터

1. 새 소형 실행 adapter는 정적인 규칙만 구현한다. contained path/no-link drift, typed descriptor, text/binary 경계, bounded resources, 단일 소유 작업, 처음 FINAL까지 실제 관측 시계, once cost, 실패·불확실 소유 시 다음 실행 차단, durable raw output/readback이다. 승인 요청이나 통과 결과를 합성하지 않는다.
2. 시도 manifest에는 고정 source snapshots, 실제 root design/review/reflection, 고정 science configuration/criteria, 실행 argv/role/timeout/output, 필요한 입력 descriptors, 실제 폐쇄 predecessor descriptor와 현재 cumulative baseline, reservation/order가 들어간다. 어떤 숫자·pass namespace도 코드에 신규 복제하지 않는다. 동일 call identity를 다시 실행하지 않는다. 변경된 소스는 새 data identity를 갖고 원래 실패와 비용은 보존한다.
3. 기존 P30 및 이전 policy/activation/cost 기록은 불변의 감사 입력이다. 새 adapter는 감사된 폐쇄111 snapshot과 실제 live의 값이 같고 모든 기존 비용·grant·Original counters·ownership이 보존됐는지 확인한다. 과거 모듈은 실행하지 않는다. manifest가 다른 predecessor나 cumulative를 제시하면 실패한다. 새 실험 비용은 같은 cumulative에 더한다. 기존 실패 pass를 수정하거나 P30를 성공으로 바꾸지 않는다.
4. 권한의 기준은 현재 사용자의 '앞으로 내 승인 받지 말고 스스로 판단해서 진행'이다. 이전 per-attempt 승인 규칙은 새 reversible 개발 및 필요한 테스트를 차단하지 않는다. 자동 무제한 재시도·예산 초기화로 해석하지 않는다. 현재 잔여 내에서 한 작업씩 명시적으로 등록하고 실행한다. 새 사용자 승인 질문은 하지 않는다.

## 이번 변경의 최소 구현과 검증

먼저 Sol은 Source-only로 두 부분만 제안한다: (a) 정확한 완료 로그 read/enrollment 경로 수정과 해당 기존 의미 있는 검사 재사용, (b) ROOT가 실제 명령에 사용할 고정 adapter와 manifest 형식. 안전 경계/감사 semantics/negative cases를 생략하는 새 실행 경로는 채택하지 않는다. source-closure, synthetic fixture, 현재 suite 및 qualification이 각각 어떤 필수 조건을 증명하는지 기존 production 코드를 조사해 mapping을 제시한다. 기존 증명을 재사용할 수 없는 부분만 adapter에 구현하고, 동일 의미의 필수 검사는 변경된 source에서 실제로 수행한다. 작은 adapter에 과거 숫자별 proof chain을 복제하지 않는다.

현재 84 ordered names/252 phases, fixed18 reader/독립2MiB, 역사 exact membership, ownership-before-membership, poison/clock/fences, Native15+same PROGRAM16 의미와 부정 사례는 유지한다. adapter가 기존 paid scanner/fixture/suite/Q 진입점을 어떻게 안전하게 실행할지 실제 호출 수준으로 제안한다. 8초 PROGRAM/37 extraction 등 기존 검사의 시간 또는 의미를 임의로 약화하지 않는다. 해당 경로를 동등하게 이어갈 수 없다면 통과시키지 말고 ROOT가 판단할 구체적 결함을 제출한다. 검사가 자체 과거 정책을 강제해 새로운 사용자 위임에 맞는 데이터 실행을 거부하는 경우, 해당 현재 selector만 prospective semantics로 바꾸고 역사 reader의 결과·규칙은 유지한다.

검증 단계는 fixed manifest → 실제 생성 scanner/fixture/current suite/qualification → 실질적 원본 계산과 body 순이다. 필수 generated 전체와 별도 closure 후처리를 현재 생성 잔여 약1019.87초 안에 사전 배분한다. 실패는 실제 비용으로 누적하고, 같은 실패 명령을 그대로 반복하지 않는다. 실제 계측 전 완료 시간은 추정으로 표시한다. 결과가 없거나 ownership uncertain이면 동작을 차단한다.

## 실제 몸체 실험으로의 연결

generated가 닫히면 ROOT는 기존 startup 보존 설계를 재사용한다. E318 saved initial/fullDrive300의 source/graph/adapter/u0/resource1과 baseline 대응을 실제 원본 단계에서 확인한다. 기존 네 R창 gradient 방향과 유한 척도, 실제 float32 δ≤0.001, palette/selected edges, 전체 startup99 조건, neutral/candidate identity 분리를 유지한다. 후보 하나를 control0에서 적용하고 99 startup가 적격이면 같은 인스턴스를300 controls까지 잇는다. neutral/candidate 원본 실행 허용량은 이전 actual46–48초 startup 비용과 저장·감사 비용을 포함해 별도 실제 계획으로 충분히 배정한다. 현재 local20초를 몰래 늘리거나 generated 초와 상계하지 않는다. 예산이 부족하면 같은 사용자 위임 안에서 필요량·누적 보존·구체적 용도를 반영한 amendment를 ROOT가 작성한다. 이 설계 자체는 body/CUDA 실행이나 보행 PASS가 아니다.

## 소유자와 채택 기준

ROOT 설계 → 실제 GPT-6 Astra/max review → ROOT reflection → 실제 GPT-6.1 Sol/xhigh FIRST → ROOT FIRST → 필요하면 같은 Sol sole ONE_R1 → ROOT SECOND → ROOT 잔여 수정 → 실제 적용/freeze/실행. 높은 effort는 Python/Native/caller/ledger의 실제 의미를 연결해야 하는 난도에 따른다. P30 sole R1은 이미 사용했으므로 기존 P30 residual 자체는 ROOT가 맡는다. 이번 새 구조의 FIRST/R1은 별도이며 이전 파일을 덮어쓰지 않는다.

Sol은 proposal만 수정하고 canonical/live/active plan/history/old handoff/GOAL을 수정하지 않는다. ROOT는 실제 review와 시행 경로를 확인한 뒤 적용한다. Source-only 단계에는 대상 Python import/AST/compile/test/PROGRAM/private probe, 원본 archive/array/header/checkpoint/hash, CUDA/body/UI를 실행하지 않는다. 메타데이터·텍스트·SHA·Native ParseFile만 사용한다. 상위 goal은 ACTIVE/INCOMPLETE다. 실험적으로 유효한 기각 후 다음 보행 방법을 선택하며 F1–F8 전부가 실제로 충족될 때만 완료한다.
