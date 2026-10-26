# ROOT 잔여 수정: 실제 inventory 설명자와 신체 실행 — 2026-10-07

목표는 동일하게 감사된 BANC 후보의 실제 신체 결과를 얻는 것이다. F1–F8, startup99/short 기준, 생물학적 미검증, M2/UI/미사용 F6와 기존 자원 한도를 보존한다. 사용자의 최신 자율 진행 지시에 따라 추가 승인 없이 판단한다. 기존 body-retry의 Sol FIRST/sole R1와 ROOT FIRST/SECOND는 완료됐으므로 이 단계의 잔여 구현은 ROOT가 맡고, 구현 전에 실제 GPT-6 Astra/max 설계 리뷰를 반영한다. 새 Sol 수정 라운드를 만들지 않는다.

## 확인한 증거와 가설

Same-s 정적 검사 exit0, 수학11 PASS, 실행24 PASS로 qualification을 실제 등록했다. 이후 fixed b-body는 genuine FINAL 5.446초/exit1, timeout=false/known closed/once charged다. contact-body-report의 run_valid=false, first_error는 new-candidate-input/ValueError/descriptor fields, secondary_errors=[]이다. 실제 보행 결과가 없으며 constructor0 또는 neural0 영수증을 만들지 않는다. b-body-audit는 미실행이다.

고정 Original binder의 JSON metadata를 읽으면 135개 inventory.files 중131개는 bytes/sha256/role/producer_pin_inherited_from, 4개는 bytes/sha256/kind/selected_archive_role다. 현재 body_plan의 dict(spec,path=name)는 provenance 키를 strict three-field descriptor에 그대로 전달한다. 첫 whole_npy_descriptor의 descriptor fields 거부와 부합한다. 기존 generated fixture는 provenance 키가 없는 synthetic record여서 이 오류를 놓쳤다. 원래 payload/header를 ROOT에서 읽거나 hash하지 않았다.

가설: inventory의 provenance는 고정 binder에서 그대로 검증하고, 파일 접근 설명자는 path/bytes/sha256 세 필드로 명시적으로 투영하면 이 거부를 해결한다. 경쟁 설명: 투영 이후 header/assembly/lifecycle 또는 실제 startup/short가 여전히 실패할 수 있다. 수정 성공과 실제 gait 성공을 구분한다.

## 최소 구현과 의미 있는 회귀

body_plan에서 inventory record를 바꾸지 않고 whole={path:name,bytes:spec[bytes],sha256:spec[sha256]}를 만든다. descriptor()의 정확한 필드 집합/자료형/hash/cap 검사를 완화하지 않는다. whole160MiB/header4096/version1·2/C-order/exactsize/digest/stamp/posthash, NPZmember64MiB/global archive bounds/selected512MiB/3*selected+96MiB<=host24GiB는 유지한다. source134/old audited 제외/new exact candidate 및 Saved.arrays semantics도 유지한다.

기존 header-only math fixture를 실제 inventory의 두 provenance record 형태로 바꾼다. immutable admitted inventory가 변하지 않고 digest/posthash에 three-field 설명자가 전달됨을 검증한다. Original payload를 생성 fixture에 읽지 않는다. 기존 수학11 및 gate24 모두 유지하고, Native/Python 일치·history·per-call120·300/380/460/600/900·단회 비용/overrun/hold·owner9 회귀를 그대로 실행한다. 테스트 추가 개수보다 실제 놓친 입력 경로를 검증한다.

## 유한한 c 실행 계약과 역사 보존

기존 b descriptor의 maximum_retries=1과 고정 b call/cost/결과/미사용 b audit를 수정하거나 재실행하지 않는다. 새 immutable descriptor-c는 별도의 단일 기술 보정 단계이며 maximum_retries=1, automatic_repeats=0, newgrant/reset/resource increment=0이다. 오직 contact-learning-20261007-c-body 및 c-body-audit를 허용한다. direction 및 candidate audit 재학습/재평가 없이 기존 k candidate 전체 identity를 그대로 사용한다.

c admission의 필수 predecessor는 실제 closed262/144, 과거 b descriptor, actual b cost/manifest/report/outer/trace, qualified-s, source-s 및 as-executed-s12다. b의 exit1/run_valid=false/first_error 정확한 descriptor fields tuple, known-close/once-cost와 source-s를 엄격히 확인한다. generic failure 예외는 만들지 않는다. 과거237→241→245→249→252→255→258 prefix와 l/m/n/o/p/q/r21 costs/caps/timeout6·10, 그 뒤 s45/46/47 세 actual PASS 및 b 기술 실패를 그대로 검증한다. 기존 generated transfer 네 이벤트도 그대로 유지한다.

현재 Source-t는 k candidate provenance와 분리한다. fresh lint48/math49/gate50만 current-t qualification으로 사용한다. c-body admission은 정확히265 debits/147 charged calls/Original22이며 c-body-audit는266/148/Original23이다. 모든 prefix 및 cost provenance를 검증하고 unrelated/extra call을 거부한다. 동일 candidate·동일 thresholds·동일 limits이며 b 실패를 gait 또는 proxy의 과학적 반증으로 취급하지 않는다.

Source-t closure에는 기존 frozen-s 전체 inherited closure, 실행된 s12 archive, s qualification/세 costs/JUnit, b actual evidence/immutable descriptor, closed262 및 이 설계/actual review/ROOT reflection을 포함한다. 과거 Source를 현재로 재표시하거나 mutable ledger/self-freeze backreference를 추가하지 않는다. 모든 request/output/trace는 CreateNew, suffix는 metadata 파일명만, fixed c IDs는 한 번만 실행한다.

## 고정 자원과 실행 결정

Actual closed262: generated593.228/900, shared remaining378.5599844000001; Original11994.244960899505/16200, remaining4205.755039100494; science1974.1290000000001/6000, all owners/pending/holds/reservations0. Fresh lint8/4/4, math12/8/4, gate64/56/8: outer84+close20=104 <= local remainder306.772/shared remainder. 새 생성 재배정이나 grant는 없다. Body300/285/270/15/output96MiB/host24GiB/GPUdelta8GiB; CPU audit120/100/90/20/output16MiB/host2GiB/thread1, outer420+close180 <= 기존 Original 남은 한도다. 실제 모든 지출은 genuine FINAL clock으로 단회 기록한다.

전체 same-t qualification 뒤 한 번만 c-body를 실행한다. exit0/run_valid=true인 경우에만 independent CPU body audit를 실행한다. Healthy startup99 rejection은 과학적 미달이므로 audit할 수 있지만 성공으로 채택하지 않는다. 기술 실패면 실패 비용/결과를 보존하고 c audit와 자동 재실행은 하지 않는다. Short PASS도 F1–F8 완료 또는 생물학적 자연 보행의 증거가 아니다. 실제 결과가 나온 뒤 학습 접근의 다음 판단을 한다.

## 핵심 immutable pins

| 증거 | bytes | SHA256 |
|---|---:|---|
| verification/banc-natural-gait-m3-20260919/parent-contact-learning-source-freeze-20261007-s.json | 122306 | d301dcca02617bc53f243e274b41bf3a8739929734f505426ec585506a0e881d |
| verification/banc-natural-gait-m3-20260919/parent-contact-learning-qualified-20261007-s.json | 4982 | c8cc15ed3c7e4099a52a7e1083993835720bfbdab5f9bd2f185e261b582160e2 |
| verification/banc-natural-gait-m3-20260919/parent-contact-learning-as-executed-s/index.json | 4159 | 588e9542bc82b20accbc749678a516fa2ca3e4acd57078176fb04ce3c6f7ab8d |
| verification/banc-natural-gait-m3-20260919/parent-contact-body-retry-closed262-20261007-t.json | 436252 | 422a6544b33cf59c5bebc8f07fdd0ded668a64588b3b01fd5c7db1f6fda0ac2c |
| verification/banc-natural-gait-m3-20260919/parent-contact-learning-ROOT-actual-result-20261007-b.json | 2560 | cd3ba1d335cae691e08548ef7fd84eee2e021dc1e8035f10d2e1386700a6e998 |
| verification/banc-natural-gait-m3-20260919/parent-contact-learning-20261007-b-body-full-return.json | 2908 | 909e0fff5f9495ae358c10dbb86c85b538e8de7af32d322c5002d13a334bdb3a |
| verification/banc-natural-gait-m3-20260919/parent-contact-learning-20261007-b-body/result/contact-body-report.json | 1164 | dbbb508bd607f96491ad23dadba9280e9485cc9e0ef874f719f2bc2a854eb8af |

ROOT GOAL_PLAN.md151552B/SHA16898e461fede082206f9584c60b3fb11d994a7ab708cdfb19f950e35126eace is unchanged. Active plan/progress are outside Source-s and may receive honest current status. F1–F8 ACTIVE/INCOMPLETE, candidate unadopted, biology UNPROVEN, exact native VJP UNVERIFIED remain.
