# E339: 저장 명령의 다리 간 협응 이탈 확인

현재 live 저장 명령에서 사전에 고정한 encoded angular discrepancy 기준을 넘는 다리 쌍이 **15개 중 11개**다. Reference calibration과 live coverage가 통과했고, ROOT가 원본 q와 reference5필드에서 독립적으로 재계산한 원자료 감사 **5,591개**도 통과했다. H339는 고정된 encoded rule에서 supported다. 이는 기존 협응 손실의 효과·기울기·배분을 먼저 검토할 근거다. 새 손실을 중복 추가하거나 gain sweep을 승인한 결과가 아니다.

## 실제 결과와 해석 범위

Reference는 모든 다리에 같은 global phase를 넣은 128개 합성 명령이다. 곡선에는 해부학적 offset이 이미 포함되어 있어 다시 더하지 않았다. Reference의 모든15쌍과 모든6다리 valid count는128/128, live는200/200이다. Reference pair resultant 최소0.9985363771, reference deviation 최대0.020303974cycles로 calibration 기준을 통과했다. Live의 단위 moment coherence 평균은0.5470894066으로 reference0.9996634475보다 낮다.

| 좌우 다리 쌍 | Reference deviation 최대 (cycles) | 이탈 문턱 (cycles) | 이탈 행/유효 행 | 이탈 비율 |
|---|---:|---:|---:|---:|
| LF–RF | 0.015599034 | 0.165599034 | 98/200 | 49% |
| LM–RM | 0.020303974 | 0.170303974 | 65/200 | 32.5% |
| LH–RH | 0.012506263 | 0.162506263 | 132/200 | 66% |

이탈은 각 행의 reference bias와의 absolute wrapped difference가 `reference max + .15cycles`를 엄격하게 넘는 경우다. 이탈 비율은 pair별 유효 행을 분모로 삼았다. 세 좌우 쌍 모두 사전 문턱25%를 넘는다. 전체15쌍의 세부 결과는 불변 result와 independent audit에 남겼다.

기존 협응 손실 평균0.8140994661은 집중도 성분0.0235726695와 weighted angular 성분0.7905267966으로 분해된다. 값 기준으로는 angular 성분이 대부분이지만, 이 비율이 학습의 기울기 영향이나 신경 parameter에 도달하는 효과를 뜻하지 않는다. E335의 전체 smooth loss에서 successor6.8843837496와 coordination0.8140994661이 합쳐졌다는 사실만으로도 어느 기울기가 우세한지 단정할 수 없다.

이 결과는 고정 격자에서 encoded 명령 관계를 측정했다. 실제 발 접촉의 위상, F4의 eligible cycles, 자연스러운 생물학적 보행, 협응 이탈이 미끄러짐을 일으켰다는 인과관계, 연속 위상·곡선 밖 명령에서 encoder의 정확성을 입증하지 않는다. R>=.5는 집중도 조건이다. 기존 E335의 보행 기각, F1–F8 미달, candidate_adopted=false를 유지한다.

## 구현 및 리뷰

ROOT 설계 → 실제 GPT-6 Astra/max 리뷰와 M1–M7 반영 → 실제 GPT-6.1 Sol/high FIRST 구현 → ROOT FIRST → 같은 Sol ONE R1 → ROOT SECOND를 수행했다. FIRST 수치 검산은 맞았지만, 계산 직후 science deadline 오류가 성공 상태를 남기는 결함과 근거 없는 generated proof를 freeze가 수락하는 결함을 생성 자료로 재현했다. 두 결함을 같은 Sol이 첫 수정으로 고쳤다.

R1에서는 science 오류가 status/analysis/publication/suggested 진단을 취소하고 CLI가 exit1을 반환한다. 현재 Sol·ROOT의 계약/감독/보고서/로그/source pins를 묶은 유한한 proof를 검증하고 freeze에도 직접 byte pins를 포함한다. 현재 source에서 Ruff와 생성59시험이 통과했다. ROOT SECOND의 독립 생성 검산5,297항목 및 잘못된 행/순서/단위/분모/손실/결과/입력·출력 변조 등13개 거절 사례가 통과했다. ROOT의 생산 소스 잔여 수정은 필요하지 않았다.

최종 tool SHA는 `bc186b98fa62e67946be1dea75bcf480cfcbd01ababa0eb43a8358372481d940`, tests SHA는 `92d24f7d119f25da07bf0fcbbf3d91a874247b3caeb36c7a3523809b57ff9e1a`다. 생성59시험과5,297항목은 실제 보행의 성공 횟수가 아니다.

## 단회 실행과 자원

최종 source/proof/freeze/review를 고정한 뒤 저장 분석1회와 독립 원자료 감사1회만 실행했다. 분석 감독0.4403256초/peak RAM45,961,216B, 감사 감독1.5096314초/46,546,944B로 두 절대 wall cap15초/10초 안에서 종료했다. 원본17개 선택 배열을 각각 pre/post hash와 bytes로 확인하고, 모든 archive member의 안전한 header를 관측했다. Extra 배열의 값은 해석하지 않았다. 동일 result/receipt/stdout, manifest byte hashes와 readback, 외부receipt, 독립 감사 및 4MiB 보관 상한도 통과했다. 재시도0, body/neural/CUDA/UI/F6 호출0이다.

실패 포함 생성 비용은 ROOT5.8361957/60초, Sol FIRST+ONE R1 6.1567349/60초다. 기존 모든 실패와 비용은 보존했다. 원본 감독1.949957초를 한 번씩만 합산해 누적**9567.165820199525/9660초**, 잔여**92.83417980047489초**, 예약0, 원본 실행 중0이다. 과거 E337/E338 실패를 재시도하거나 재분류하지 않았다.

## 다음 판단

기존 coordination 항이 없는 상황이 아니다. 다음 설계는 같은 저장 명령에서 successor와 coordination의 **출력 공간 기울기**를 분리하고, 현재 합산 하강 방향이 협응을 줄이는지, 어느 항이 크기와 방향을 지배하는지 확인한다. 먼저 정확한 연쇄 미분과 201개 경계/200쌍의 겹침을 검증한다. 이 진단은 BANC parameter gradient나 fresh neural/body의 효과를 대체하지 않는다. 실제 신경 변화는 별도 설계, 실제 Astra/max 리뷰, 적정 effort의 GPT-6.1 Sol 구현과 ROOT 리뷰 이후에만 진행한다.

증거: `verification/banc-natural-gait-m3-20260919/parent-e339-final-decision.json`, `parent-e339-saved-raw-audit.json`, `parent-e339-original-external-receipt.json`, `parent-e339-audit-external-receipt.json`, `e339-saved-coordination/result.json`, `e339-saved-coordination/output-manifest.json`, 최종 freeze/source review/current-generated qualification 및 실제 reviewer/implementation dispatch 기록.
