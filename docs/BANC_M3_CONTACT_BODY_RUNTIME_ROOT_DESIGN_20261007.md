# 실제 신체 실행 시간 진단과 개선 — ROOT 설계, 2026-10-07

목표는 고정된 BANC 후보를 실제 신체에서 끝까지 실행하고 startup/short 결과를 얻는 것이다. 자연 보행 F1–F8, 전체 BANC 노드·연결 provenance, 외부 CPG/교사/궤적/root 보정 금지, M2/UI/미사용 F6 및 기존 모든 실패·비용은 유지한다. 추가 승인 없이 판단하되 새 자원 grant/reset은 만들지 않는다. 이번 단계는 새 runtime 작업이므로 ROOT 설계→실제 GPT-6 Astra/max 리뷰 반영→GPT-6.1 Sol의 적절한 effort FIRST→ROOT FIRST→필요한 same Sol sole R1→ROOT SECOND→ROOT 잔여 순서로 진행한다.

## 실제 결과와 해결할 불확실성

Same-x actual lint60 exit0/1.703초, math61 11 PASS/4.908초, gate62 full24 PASS/57.642초와 Native qualification4999B/SHA268f6efe3f4e230106c1d6dc156a2f63c8f02c397adcc93dba70c188734cd41e 등록을 확인했다. Source-x169449B/SHA0ea95b308496add5983cc0570bd545b818b4bd28ae9fb734aaba342eb3bec083다.

실제 fixed c-body는268.84초/exit1/known closed/once cost/outer timeout=false, owner interval270.0479967999272초다. Terminal stderr의 첫 오류는 existing-continuous-candidate-body/TimeoutError/contact same owner ENTRY internal deadline이다. Posthash와 publication에서도 같은 deadline이 뒤이어 실패했고 durable result bytes=0, work=null이다. Constructor/neural/physics0 영수증을 만들지 않는다. Broad stage는 ray.arm_data 또는 continuous_arm을 포함하므로 정확한 병목·실제 work는 미확정이다. c audit와 c 재실행은 하지 않는다.

Body worker는 input_plan/선택 후보 로딩과 digest 확인 뒤 stage를 existing-continuous-candidate-body로 바꾼다. Source에서 body_plan/arm_data가 파일별로 fence(pre_array)를 반복하고 fence가 전체 current Source/과거 cost/archived metadata/history를 검증하는 것을 확인했다. 반복 검증이 주요 시간 요인이라는 가설이 있으나 아직 측정으로 입증하지 않았다. 경쟁 설명은 native constructor/assets/CUDA warmup 또는 실제 control이다. 근거 없이 CUDA나 constructor를 원인으로 단정하지 않는다.

## 첫 구현: 측정과 실패 증거를 먼저 남긴다

최소 bounded metadata phase journal을 기존 result-supervision/output cap 안에 추가한다. body-plan 시작/종료, candidate header/load/hash, arm_data 시작/종료, pre_construct, construct, initial transfer, control, posthash/publication와 cold fence 호출 수/누적 wall time을 기록한다. 실제 시작/반환을 구분하고 미반환 작업은 UNKNOWN/NOT_RETURNED로 남긴다. 큰 array·state·그래프·neural output을 journal에 넣지 않는다. Max events 및 JSON bytes를 사전에 고정한다.

엄격한 ENTRY science deadline270/child285/outer300/cleanup15와 모든 Original payload 접근·Source/owner/lease/hash 검증을 유지한다. Deadline 후에는 새로운 Original reads/hash/neural/physics/constructor work를 하지 않는다. 이미 취득한 작은 metadata journal/first error/secondary errors만 남길 수 있도록 실제 owner의 남은 cleanup 안의 제한된 failure publication을 설계한다. 이것은 run_valid=true 또는 posthash success를 만들지 않으며 종료/EOF/once-cost를 그대로 요구한다. Metadata publication도 거부되면 bounded stderr가 최종 증거다.

검사는 실제 놓친 경로를 다룬다. Deadline으로 science와 posthash가 실패해도 acquired phase receipt와 first error가 바뀌지 않고 bounded journal이 남는지, 시간·count·cap이 실제 값인지, owner/Source drift가 거부되는지 확인한다. 기존 math11/gate24와 실제 owner9/metadata3/이전 모든 assertions를 보존한다. Root에서 Original payload/header/hash를 직접 읽지 않는다.

## 이후 최적화의 결정 조건

Phase journal이 반복 cold metadata/history 검증을 병목으로 확인할 때만 worker-scoped proof reuse를 검토한다. 변경 전후 같은 candidate·초기화·control·caps·evaluator로 비교하고 Source/binder/Original input pin, critical owner/lease/shared-lock, pre-construct/publication cold checks와 Original data pre/post digest를 유지해야 한다. Cache가 mutable data를 신뢰하거나 Source/metadata 변조를 놓치면 거부한다. 안전한 reuse가 안 되면 병목이 아닌 다른 경로의 변경을 시도하지 않고 실제 측정에 맞춰 constructor/load/CUDA의 다음 방법을 선택한다.

이번 새 d 단계는 동일 audited k 후보의 한 번의 body와 genuine exit0/run_valid=true일 때만 CPU audit를 허용하는 별도 immutable prospective 계약이 필요하다. 기존 b/c 계약·소모된 call·실패는 수정하지 않는다. Journal만 얻고 기술적으로 실패하면 gait 성공/채택/전체 완료로 해석하지 않는다. Diagnostic progress와 final F1–F8를 분리한다.

## 자원과 다음 분기

Actual closed276/158448154B/SHA1ddc9e0c8065914918e9bcbdd0617932359bc5ae1b313ead35ed888df025e351: generated822.9329999999999/900/shared remaining148.85498440000015, Original12263.084960899505/16200/stage2369.33/6300/science2242.969/6000/remaining3936.9150391004946, owners/pending/holds/reservations0다. Original 추가600의 예약 여유는 기존 한도 안이다.

Fresh full generated8+12+80+close20=120은 현재 local remainder77.06700000000012를 초과한다. 따라서 source implementation/실제 검사는 추가 conserved allocation을 검증하기 전까지 시작하지 않는다. 새 monetary/resource grant 없이 existing shared148.85498440000015 안의60을 distinct prospective event로 local900→960에 재배정하면 전체 unused137.06700000000012가 shared 잔여 안이며120을 수용한다. 네 이전 allocation/초기 policy/모든 actual usage/prefix와 source를 보존하고 원장의 event pointer/limit 두 필드만 단회 변경한다. 이것은 설계 제안이며 아직 적용하지 않았다. Reviewer는 더 작은 적절한 검증 범위 또는 안전한 지출 방식이 있으면 권고할 수 있다. Local/global/per-call120/closing reserves와 ordered Float64 accounting은 완화하지 않는다.

핵심 실제 증거는 parent-contact-learning-ROOT-actual-result-20261007-c.json3069B/SHAa1b1a8f894c086f0db0ac784cc54d933d174de80eb212d3f81d79c909ec522d2와 c-body cost2909B/SHAfd9fb439a051f4dcb86bd0893f48630a30e3b1477d180943238830c0723357f5다. 현재 원장과 실행 Source-x는 변경하지 않고 metadata/source 리뷰만 수행한다. Goal ACTIVE/INCOMPLETE, candidate unadopted/biology UNPROVEN/exact native VJP UNVERIFIED는 유지한다.
