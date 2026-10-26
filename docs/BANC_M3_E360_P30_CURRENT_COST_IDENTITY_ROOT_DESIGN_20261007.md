# P30 현재 비용 식별과 실행 재개 — ROOT 설계

2026-10-07. 실제 보행 실험을 막은 비용 이력 분류 오류를 수정한다. 성공한 검사와 실패한 실행 준비를 구분하며, 기존 결과·비용·보행 기준은 유지한다. 이 설계는 전체 M3 목표를 대체하지 않는다.

## 확인된 결과와 문제

P29의 실제 활성화는 성공했고 Ruff도 exit0/전체 반환2.057초/cap3으로 통과했다. 원본 capture/collector가 검증됐고 비용은 정확히 한 번 기록됐다. 실제 비용3348B/SHAf0975a24…, closed109 원장212957B/SHA050f8836…,109 debit/59 call/26 paid pass, next_index1, failed_cost=null, hold/pending/reservation0이다. Ruff의 내부 완료·collector·capture 증명은 모두 true다.

Ruff admin 바깥 ROOT 호출은 LASTEXITCODE 조건으로 exit1을 냈다. 실제 account의 admin 경로는 원장 기록과 JSON 출력으로 끝나며 native exit를 명시하지 않는다. 실제 LASTEXITCODE 값은 확인할 수 없고 재구성하지 않는다. 원래 오류/FINAL은 보존한다. 비용 기록 성공이나 Ruff 통과를 이 바깥 오류와 혼동하지 않는다. ROOT는 새 후속 admin caller에서 실제 script 예외를 보존하고 반환된 budget·한 번 기록된 cost·원장 상태를 확인하도록 수정했다. 기존 실패 caller·P29 freeze·cost는 바꾸지 않았다.

이후 scanner 준비는 실제 한 번 실행되어 exit1로 끝났다. 원래 ErrorRecord는 Cost-Authority-Spec4173 → Assert-Cost-Authority4178 → main6109이다. 첫 현재 비용 행108의 실제 pass는 P29인데 Native의 `$version=28`과 Python cost_authority_spec의 기본값28이 남아 있다. Native 실제 오류와 Python Source 결함을 구분한다. scanner 요청2208B/SHAccc5e55b…는 생성됐으나 call 등록·scanner 실행·비용은 없으며 원장은 정확히 closed109 그대로다. 실패2731B/SHAb10df607…와 genuine return·first ErrorRecord·as-executed Source를 보존한다.

P29 검토가 현재 숫자 기본값을 놓쳤다는 점도 보존한다. namespace rename과 JSON tuple 확인만으로 현재 유료 행의 실제 분류 경로를 검증할 수 없었다. 같은 실패를 변경 없이 반복하지 않는다.

## 바꿀 방식과 실제 검사

1. **현재 비용에 대한 숫자 기본값을 제거한다.** Python과 Native는 완료된 역사 범위를 명시적으로 찾고, 그 밖의 현재 행은 compiled current 정책의 pass ID와 descriptor에 정확히 연결한다. 현재 신뢰 기준을 namespace 숫자 추측이나 mutable Describe로 대체하지 않는다. typed index/pass 검사와 unknown/future 거부는 유지한다. Native switch는 역사 분류만 담당하고 현재 tuple을 숫자28/29/30 default에 걸어 두지 않는다.

   새로운 현재P30의 비용 시작 index는109다. 역사 범위108..109는 실제 P29이며 원래 logical policy12804/SHA24bd4b5b…와 frozen physical policy를 사용한다. 105/106=P25,107=P27도 그대로다. compiled current pass ID와 정책 tuple을 한곳에서 관리하고 실제 cost classifier·Call-Policy·bootstrap·event·fresh pass·freeze가 같은 값인지 확인한다. 역사 Source와 정책을 전역 치환하지 않는다.

2. **첫 현재 유료 행을 실제로 검사한다.** 기존84 named tests/252 phases와 동일 PROGRAM16 안에서, 새 P30 Ruff가 실제로 기록된 뒤 그 원래 first current cost/actual record를 읽어 Python 및 Native의 실제 cost authority 함수를 positive로 호출한다. fabricated cost나 JSON mirror validator로 대체하지 않는다. 현재 시작 index109/실제 current pass/현재 descriptor가 실제 함수로 수락됨을 확인한다. 같은 함수의 wrong index/pass/descriptor/future/type negatives도 유지한다. 기존 whole historical109/full cost/paid105–108/정책/roster END와 bool activation true/false 검사를 보존한다.

   기존 전체 역사 positive가 첫 현재 행108을 포함하지 않아 이번 오류를 잡지 못했다. 전체 역사 검사와 첫 현재 행 positive를 구분한다. scratch-root isolation/cleanup,37 slots/Native15+same PROGRAM16/one process8초/full11/Source14/late fourth/nonown Q를 유지한다. 실제8초 적합성은 실행으로 판단한다.

3. **성공한 지불 전임자와 실패한 Source 준비를 분리한다.** P30의 Source 전임자는 실제 P29 freeze/account이고 지불 전임자는 실제 성공한 P29 Ruff다. P29를 지불된 실패로 바꾸거나 failed_cost를 채우지 않는다. 현재 Source 수정과 실제 unpaid Native scanner admission failure가 후속 시도를 요구한 근거다.

   새 현재 admission은 exact closed109/ownership0/성공한 P29 paid Ruff(0/2.057/cap3/charged_once/internal_valid/collector_valid)/원래 unpaid Native failure+first ErrorRecord+genuine FINAL/실제 Source correction을 독립적으로 검증한다. 역사 Concrete-Failed-Predecessor 규칙은 그대로 두고 이 현재 전환만 명시적으로 검증한다. 새 request/pass에 paid-success descriptor와 별도 unpaid-source-failure provenance를 구분해 기록한다. 기존 필드를 실제 실패라고 거짓 이름 붙여 재사용하지 않는다. 새 activation pair2/grant0, counts109/59/26 → Ruff 등록109/60/27 → 첫 debit109/count110이다. namespace30과 paid pass count27은 다르다. future31과 unpaid24/26/28은 거부·보존한다.

## 유한한 이력과 자원

P29의 새로운 역사 물리 파일은 실제로 필요한 direct read만 목록과 bytes/SHA를 파생한다. old H1160/64980982B와 fixed18 readers/scope를 그대로 보존한 확장 table을 ROOT가 작성한다. 성공한 P29 cost·actual·request·trace·capture descriptor JSON·manifest·stdout/stderr, closed109, freeze/정책/Source4/Native4/collector, 필요하면 원래 unpaid admission failure 입력을 역사/current ownership으로 명시적으로 구분한다. wrapper failure·review·nested provenance는 읽기 필요가 없는 한 metadata만 유지한다. 불필요한 재귀 forest를 추가하지 않는다.

H table 확장이 필요한 정확한 direct set/개수/aggregate는 구현 전에 실제로 파생한다. D512/maxleaf2MiB/table262144B/fixed-reader2MiB aggregate와 두 journals/fences/poison/root/ownership-before-membership/table-miss fallback 금지는 유지한다. unknown empty historical 입력도 거부한다. Source pin 확인은 runtime graph 인증이 아니다.

실제 generated shared676.3040156000001/local565.347, limits1700/1589, shared remaining1023.6959844/local1023.653, primary768+fallback120=888, local margin135.653이다. 기존 caps3/30/30/60/20/40/520/25/40, 모든705/suffix/MAX825/705/future125/65/fixture60/55/51/42/suite520/480/450/360,host2GiB/output16MiB를 유지한다. new grant0/resetfalse이며 이전 비용을 지우지 않는다. Original cumulative10020.115960899504/10200,stage126.361/300,E35948.181/85,E3600/20도 유지한다. 현재20초는 startup46–48초에 부족하므로 필수 검사 후 실제 allowance를 맞춘다.

P29 실제 정책12804/24bd4b5b…, freeze10168/8ee8d82b…, account714552/9162e88f…와 Source4/Native4/collector·caller/원래 오류/성공 비용을 불변으로 보존한다. 새로운P30 제안만 작성하며 canonical/live ledger/GOAL은 ROOT 적용 전까지 유지한다. 기존 ROOT admin-caller residual3878B/SHA44adb30d…는 Source-only이며 아직 실행하지 않았다. 새로운 실제 역할에 정확히 연결하고, wrapper 오류를 숨기거나 account/admin/Ruff를 재실행하지 않는다.

## 진행 절차와 보행 결과

ROOT 설계 → 실제 GPT-6 Astra/max 리뷰 → ROOT 반영 → 같은 실제 GPT-6.1 Sol의 적절한 effort FIRST → ROOT FIRST → 필요 시 같은 Sol sole ONE_R1 → ROOT SECOND → 남은 수정은 ROOT → 실제 apply/freeze/activation/ordered gates 순서로 진행한다. 새 승인은 묻지 않는다. Source 단계는 Native UTF8/text/JSON/SHA/regex/ParseFile metadata만 사용한다. Python/import/AST/compile/Ruff/tests/PROGRAM/account/helper/extracted function/ScriptBlock/private probe/network/Original arrays·headers·checkpoints/GPU/body/UI는 아직 실행하지 않는다.

필수 검사 후에는 같은 후보0→99→300/1.5초 실제 신체 screen을 수행한다. startup4/contact/slip/displacement/qualified six-leg cycles/bout/native health를 원 기준대로 평가하고 전체 F1–F8로 이어간다. 외부 CPG/교사/위상/궤적/root 교정 없이 BANC 해부학 provenance를 유지하며 M2/UI·기존 실패·미사용 초기상태 노출을 보존한다. 현재 자연스러운 보행은 아직 검증하지 못했다. 실행 인프라 개선을 보행이나 생물학적 기전의 성공으로 보고하지 않는다.
