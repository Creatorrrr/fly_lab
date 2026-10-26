# 실제 Astra 검토 반영과 저장 진단 구현 계약

ROOT가 actual GPT-6 Astra/max의 `parent-after-startup-ray-astra-review.md`5387B/SHA646f79cdff0418d54da5df479b54b5d4c024f3da6df63fd2a7afccbe8065ea8e와 JSON18244B/SHA6b79f7fed5c9e4154213443fa3a6270d2ca7fa23c450ad94f8c714dad3ce46ce를 읽고 A1–A4를 반영한다. 검토는 실행 결과가 아니다. 구현자는 GPT-6.1 Sol/high; 먼저 Source-only proposal, ROOT FIRST, 같은 Sol sole R1이 필요하면 시행, ROOT SECOND와 잔여 수정 후 실제 실행한다.

새 파일은 작은 `tools/analyze_c_banc_m3_startup_ray_transfer.py`와 핵심 numerical/invalid-input 테스트 한 파일이다. 기존 production scientific modules/body/evaluator/threshold/registry/UI를 변경하지 않는다. 새 모듈은 stdlib top-level이며 numeric imports는 ROOT가 명시적으로 호출하는 saved-Original CPU 역할 안에서만 수행한다. 과거 E336/E360 인증이나 constructor를 실행하지 않는다.

입력은 `parent-after-startup-ray-saved-roster.json`4377B/SHA07ad6d8d52f6e8b63b7aac016efbc62a0fbc4ad757e6f48781f9dfb9a5e360ef. 선택한 members만 읽는다. 입력 file cap96MiB/member16MiB/선택 expanded 총64MiB/host cap2GiB/output16MiB. ray raw67,264,638B는 file cap 안에 있고 전체 확장하지 않는다. cadence 원본 command301×42/tick/time와 normalization6을 실제 shape/dtype/finite/hash/clock 검증한다. before/after whole hashes, 재사용 helpers Source pins, input clocks, 정확한 label 창 인덱스와 실제 receipt counts를 기록한다.

A1 관측 경계: tracking42/live drive/state는 UNOBSERVED. geometric knee degrees를 명령42 radians와 차감하지 않는다. qualifiedcycles0은 eligible forward bout 없음에 따른 결과; contact transitions/명령 진동/발 AP peaks는 steps가 아니다. 결과는 관측된 association/mixed/unidentified로 반환하며 H1/H2 인과를 강제 선택하지 않는다. 시간행은 독립 반복 표본이 아니다.

A2 비교는 leg order lf/lm/lh/rf/rm/rh, command order leg-major7과 control100..299/.5≤t<1.5/200행으로 고정한다. full trace에서 기존20ms speed smoothing 후 official mask 적용. AP/height는 recorded body basis로 투영하고 leg length로 정규화하며 height world 값은 별도 기술. interval Δfoot AP는 두 endpoints의 같은 load status일 때만 loaded/unloaded로 구분하고 transition은 별도 표시한다. loaded native slip은 load≥.02BW/valid과 load-weighted 분모를 유지; unavailable=null과 counts를 보고한다. native world slip·body-frame AP·propulsive force는 다른 지표다. simultaneous support, speed sign/percentile/bout fragmentation, geometric knee95–5 span과 command change를 arm별/delta로 보고한다.

live R는 원 label controls[c+1:c+5]와 정확히 같은 live command controls를 WINDOWS100/164/228/292에서 읽고16×42 denominator/반복6norm을 유지한다. ray q_neutral/alpha4.q_candidate도 같은 저장 raw에서 확인하고 per-window R/O 및 총합을 비교한다. 값 차이는 neural replay 증거가 아니다. 원 SmoothTemporalObjective200pairs100→101..299→300 점수를 body report와 대응시키고 successor/coordination/per-leg/confidence로 분해한다. 두끝 모두 창내199pairs는 별도 diagnostic 표시. temporal fixed point 한계와 command-space surrogate 경계를 출력한다. CPU/no_grad, torch.cuda/neural forward/backward/bodyconstructor0.

A3 actual 실행은 ROOT가 같은 shared Original lock/Job supervisor와 Native once writer를 사용해 saved-Original 비용 lane에서 outer120/child100/cleanup20, worker1으로 제한한다. 기존 +6000과 모든 비용을 유지, 이 diagnosis의 resource increment0. ROOT는 새 finite CPU 요청만 등록하고 기존8역할 roster를 변형하지 않는다. 모델/물리/원본whole loader0임을 실제 import/call 제한과 출력에 구분한다. output은 concise report JSON/Markdown와 작은 그림 최대1개이며 input raw를 보고서에 복사하지 않는다. 입력누락/nonfinite/clock drift/timeout는 technical-invalid/미완료; false 성공을 만들지 않는다. infrastructure는 작은 기존 supervisor role extension으로 한정하고 과거 인증을 반복하지 않는다.

A4 saved-only 결과로 다음 변경을 식별하지 못하면 별도 최소 계측 설계를 작성한다. 고정neutral/이미기각alpha.0001 후보 각최대1회, 같은instance0→99→300. appliedcommand42와 actualjoint42의 pre/post time을 정렬하고 네R창 boundary state/drive는 필요할 때만 요청한다. cap을 먼저 고정하고 같은startup/nativehealth·sourcebodydecoder를 유지한다. 계측성공은 새 evidence 수집이며 reject후보를채택하거나 과거결과를고치는일이 아니다. 본진단은 그실험을실행하지 않는다.

F1–F8 INCOMPLETE/goal ACTIVE/M2/UI/provenance/failed records/비용 보존. 다음 실제 short 개선 후보만10초/F6/F7/F8로 진전한다.
