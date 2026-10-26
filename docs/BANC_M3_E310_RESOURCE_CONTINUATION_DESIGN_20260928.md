# E310: 기각된 자원 후보의 지속 보행 효과 판별

Root 사전 설계, 2026-09-28. 목표는 기존 자연 보행 M3의 F1–F8이며 이번 진단의 완료를 그 목표 완료로 삼지 않는다. E309 u=.005는 유효한 실행에서 startup q max .04078679924683386 > .04로 기각됐다. E306 u=.05도 기각이며 E307 같은 SSP/u0는 .0396360246919627로 시작 기준을 통과했다. 모두 .5초에서 끝났으므로 이 SSP resource 방식의 이후 협응 효과는 관측되지 않았다. E309의 minR .8823834098465781/minG .9167576432228088은 교란이 있었음을 보여줄 뿐 보행의 이익은 아니다.

소모율·tau를 더 조절하는 방법은 선택하지 않는다. 기존 전달 계수/감각 부호 변경의 E175/E177/E196/E277/E287 기각, E262의 작은 국소 반환, E308의 부분 incoming 관측을 함께 검토했다. Terminal별 resource와 core-afferent 보호를 분리하는 구조는 현재 source 공유 방식과 다르지만, 실제 stance/swing 경로를 식별할 자료가 부족하고 core edge unity가 core rate를 보존하지도 않는다. 지금 임의로 경로를 골라 새 후보를 만들 근거는 부족하다. 실제 gpt-6-sol/xhigh 읽기 전용 자문도 이 간극을 지적했다. 최종 설계 결정은 root가 한다.

## 질문과 고정 비교

**H310:** 고정된 .005 resource 후보가 같은 SSP/u0 대조군보다 [.5,1.5) 구간의 지속 전진과 여섯 다리 반복을 개선하는가. 반대 설명은 시작 자세를 바꾸기만 하고 협응의 이익을 만들지 않는다는 것이다. 두 저장 whole 상태에서 각각 단 한 번 이어 실행한다. 이는 과거 .5초 경로 차이를 포함한 동일 원래 초기 조건의 폐루프 대비이지, 현재 순간에서 자원만 바꾼 독립 인과 실험이 아니다.

- Neutral 입력: `verification/banc-natural-gait-m3-20260919/e307-ssp-neutral-body/neutral/final`, 원래 u0/R1/g1, tick2000/control100/physics .5초.
- Candidate 입력: 같은 base의 `e309-reduced-resource-body/candidate/final`, 고정 u=.005와 저장된 실제 비영 resource/신경/근육 bank/감각 필터/물리·접촉·warmstart/RNG/held 상태를 유지한다.
- E307/E309 source/result/manifest/원자료/root audit와 actual whole state 파일을 pin한다. E309의 startup 기각과 E307 통과는 원본 결과 그대로 읽고 새 startup 판정으로 대체하지 않는다.
- 새 파라미터/roster/weights/부호/적분기/decoder/물리/감각/초기 자세/seed/외부 요청은 없다. Prefix 재실행, original neutral state에서 후보 재구성, R=1 reset, 교사·phase·clock·root·명령 주입, u/tau grid/fit은 금지한다.

## 실행과 검증

`SSPRecoveryResourceTrial.from_checkpoint`의 실제 nonzero whole 복원 경로를 사용한다. 기존 zero-state E291 `run_arm`을 arm 이름이나 global monkeypatch로 속이지 않고 작은 별도 continuation 루프를 작성한다. E303/E304 엔진·wrapper와 기존 native observer/advance journal/recorder/short 평가·비용 계측은 재사용한다. 기존 source/raw/evaluator는 읽기 전용이다.

순서는 neutral→candidate, 신체는 한 번에 하나다. 두 arm의 validity와 실패를 독립 기록한다. Neutral이 기술 오류 또는 건강성 위반으로 완전한 비교 기록을 만들지 못하면 candidate는 실행하지 않는다. 완전하지만 short 기준을 못 맞춘 neutral은 유효한 대조군이다. Candidate의 원래 startup 실패는 이번 diagnostics-only 실행을 막는 새 gate가 아니며, 이를 startup 통과 또는 후보 승격으로 기록할 수 없다.

복원 직후 새 관측 snapshot을 실제 저장 whole과 모든 배열·scalar dtype/bytes로 비교한다. Body/sensory/held/adapter/resource를 제외하지 않는다. 원래 wall_s는 복원 직후 그대로여야 하며 실행 후 새 벽시간은 별도 기록한다. Clock/identity/selection/u/R/원래 graph/body/adapter를 확인한 뒤에만 한 control을 실행한다. 복원 snapshot이나 source를 참조값으로 덮어쓰지 않는다. 첫 새 control101은 neural2020/physics .505, 정확히20 신경 tick과50 native step이어야 하고 정상 decoder 및 held control 경로를 거친 기록을 남긴다. 이것을 독립 미래 replay exact 증명으로 과장하지 않는다. F2 disk future는 별도 미검증이다.

각 arm은 control101..300의 **추가200 control**만 실행한다. Native100µs finite/upright≥.9/비발 접촉0/신체 및 신경 fault 검사를 유지하며 첫 실제 건강성 위반에서 정지한다. Arm별 추가 신경4000/물리10000, 두 arm 합 **신경8000/물리20000**이다. 초기0.5초의 저장된 tick은 새 실행 비용으로 세지 않는다. Constructor setup/capture/settling은 실제 계측해 별도로 모두 기록한다. Step/constructor 진입 실패·부분 실행·복원 실패·close/storage failure의 primary/secondary 오류와 불명 비용을 보존한다.

새 raw는 control100..300의201 endpoint와 새 native10000개를 별도 저장한다. Initial row100/recruitment/activity는 해당 원본의 마지막 기록과 과학 값으로 비교한다. 원래 evaluator 입력에는 각자의 frozen prefix trace0..99 + 실제 새 trace100..300을 결합하고, native는 원래 prefix5000 + 실제 새10000을 결합하되 겹침/누락/전역 clock을 검증한다. 이는 평가 자료의 연결이며 신체/신경 런타임 입력 재생이 아니다. `short_forward` 등의 최초 기준 방향이 .5초 방향으로 바뀌지 않도록 원래 control0의 pose/basis/initial heading을 보존한다. 합친 자료와 독립 continuation raw를 모두 저장하고 provenance를 명시한다. Reference parameter/trace를 controller에 전달하지 않는다.

두 arm에 원래 E257/E288 `short_health`, `short_forward`, `short_window_slip`, gait analyzer를 같은 [.5,1.5) 및 원래 .3초 eligible bout 정의로 적용한다. 원래 absolute 기준 forward≥1.6790085779712631mm, slip≤1.8026436968249955, bout≥.3초, 여섯 complete cycles 각각>0을 그대로 보고한다. E150 원래 baseline short와 비교한 판정과 새 matchedSSP/u0 대비 raw 차이를 모두 기록한다. 새로운 상대 합격으로 absolute 실패를 덮지 않는다. 단순 총 이동 거리나 resource 활성은 협응 성공의 대체 지표가 아니다.

정량 결과에는 각 arm의 forward/slip/net-path(기존 evaluator 제공 범위)/최장 연속 전진 구간/eligible bout/다리별 complete cycles/신체 건강/원래 startup 판정/endpoint R,g 요약과 matched 차이를 포함한다. 부재·불완전 항목은 unavailable이며0/pass가 아니다. Whole initial restoration, actual interval clocks, resource metadata/current source, output manifests와 costs를 root가 독립 검산한다.

## 결정

- 복원/clock/기술 오류: 진단 무효 또는 불완전으로 기록하고 해석을 중단한다. 과학적 기각으로 세지 않으며 자동 원본 재시도는 없다.
- Candidate가 건강성을 잃거나, 지속 bout/여섯 complete cycles의 개선 없이 끝나거나, 고정 forward/slip에 미달하면 이 고정 resource branch를 보행 개선 후보로 더 확대하지 않는다. 신호 약화 강도만 바꾸는 추가 탐색 대신 신경 협응/운동 전달의 다른 구조를 사전 설계한다.
- Candidate가 건강하고 absolute short 기준을 모두 통과하고 matched neutral 대비 지속 bout와 완전 주기를 개선하면, 해당 구간에서만 resource 방식의 이익이 있다는 근거로 기록한다. E309의 startup 기각은 그대로이며 candidate/adoption/F1–F8/F2/F6/10초 eligibility는 **항상 false**다. 시작 보존을 만족하는 구조를 별도 근거로 설계하기 전에는 10초/미사용 자세/목표 도달을 실행하지 않는다.
- 두 arm 모두 좋아지면 solver/기존 경로의 효과와 resource의 추가 이익을 구분하며, 같은 후보에서 시작과 보행을 모두 통과했다고 혼합하지 않는다. 진단 결과만으로 생물학적 기전 복원을 주장하지 않는다.

## 소유·비용·산출물

Root 설계 → 기존 실제 gpt-6-sol/xhigh `/root/e306_ssp_body_impl` 구현 → root FIRST → 필요 시 같은 Sol R1 → root SECOND → root 잔여 수정. 새로운 nonzero whole 복원/clock offset/동일 evaluator 결합 및 오류 계측의 난이도로 xhigh를 유지한다. 새 `tools/verify_c_banc_m3_resource_continuation.py`, `tests/test_c_banc_resource_continuation.py` 두 파일만 구현 소유한다. 생성 fake/순수 metadata 검사 전 시도 합90초/host1.5GiB/CPU1, 원본/model/neural/body/physics/CUDA/fit0. Root source review 전 실제 저장 데이터의 수치 분석이나 실제 원본 실행은 하지 않는다. 구조와 schema 읽기는 별도 측정한다. 테스트는 nonzero state 보존/초기 mismatch 차단/정확한 추가200control/첫101interval/원본prefix와 새raw의 경계/neutral 실패 후 candidate0/원래 startupfalse와 절대승격금지/health 및 부분비용/출력 무효화를 다룬다.

현재 누적8328.606086199525 / 배정8400초, 예약0이다. **지금은 구현과 생성 검사만 허용한다.** Root review/passive saved-input gate 이후 단회 paired original/retry0에 supervisor220초/internal205초/science175초/finalization30초를 사전 배정·예약한다. Host6GiB/추가GPU4GiB/output2GiB/free4GiB/CPU1, GPU 단독, body최대2/science neural8000/physics20000. 생성·capture·settling 비용은 이 scientific 한도 밖에서 실제 계측하고 전체 시간/메모리 한도 안에 포함한다. 기존 누적비용은 이월한다. 총 시간은 사용자 상한이 아닌 작업 단계 배정이며, 실패를 삭제하거나 누적을 초기화하지 않는다.

Fresh output `verification/banc-natural-gait-m3-20260919/e310-resource-continuation`. Freeze/review/result schema는 `flylab.m3.e310.resource-continuation.{freeze,review,result}.v1`. 원본별 observed-start/final 또는 partial state, continuation+combined trace/native/activity/recruitment, endpoints R/g, evaluator raw, advance/constructor ledger, source/input/package/preservation pins, first/secondary errors, 결과 hash/manifest 및 late failure invalidation을 저장한다. 기존 E307/E309 결과/기각·startup/F1–F8/M2/UI/F6 미노출과 모든 과거 실패·비용은 변하지 않는다.
