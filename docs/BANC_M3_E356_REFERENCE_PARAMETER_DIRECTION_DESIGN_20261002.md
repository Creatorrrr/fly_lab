# E356: 검증된 기준 보행을 향하는 BANC 파라미터 방향

ROOT 반영 설계 v2, 2026-10-02. 실제 Astra6/max 리뷰의 M1/M2/M3를 반영했다. 리뷰 당시 v1은 `parent-e356-reviewed-v1-design.md`로 byte-exact 보존한다. 기존 M3 finite-goal의 방법 변경이다. 사용자 지시의 실제 실행 순서인 ROOT 설계 → GPT-6 Astra/max 설계 리뷰 → ROOT 반영 → GPT-6.1 Sol 구현 → ROOT FIRST → SAME Sol ONE R1(필요 시) → ROOT SECOND → ROOT 잔여 수정 → 다음 단계를 적용한다. 모델 설정은 실제 에이전트 도구에서 확인한다. 이 문서 저장은 원자료 실행 허가 또는 성공 판정이 아니다.

## 판단할 질문과 보존할 사실

H356: 전체 BANC의 현재 native 신경 출력에서, 검증된 E353의 느린 기준 보행 명령을 향하는 새 R 파라미터 VJP를 계산하면, 단 한 번의 작은 native 가중치 조정이 같은 네 고정 창에서 실제 R 손실을 줄이는가?

기존 위상 자유 S/C 개선이 몸체 보행으로 충분히 전달되지 않은 것이 이번 방법 변경의 이유다. E353은 같은 FlyBody의 1.5초 기준 명령에서 1.9749839267771232 mm 전진, slip 1.399688502351197 mm/s, 0.995초 bout와 여섯 다리 1~2개 완성 주기를 실제 확인했다. `parent-e353-final-decision.json`의 독립 감사와 실제 worker/audit 종료가 닫혀 있어 이후 학습의 목표 명령으로 사용 가능하다. 이것은 지속적인 자연 보행 또는 신경 자율 보행의 증명이 아니다.

E354의 실제 outer exit1, worker exit unknown, H354 미평가를 보존한다. E355 원자료 감사도 계산 후 보고서 튜플 직렬화에서 실제 실패했으며 H355 미평가다. E355 ROOT 잔여 수정 뒤의 생성 데이터 전체16 case/48 phase, Ruff, passive freeze 통과는 구현 자격만 증명한다. E354 raw의 q 기울기, 기존 R 스칼라 또는 E355의 저장되지 않은 계산값을 이번 신경 파라미터 기울기의 증거로 재사용하지 않는다. E354/E355 실행, raw 또는 감사를 재시도하지 않는다.

원래 F1–F8, M2, UI8770, F6와 생물학적 기전 미검증을 유지한다. 최종 구동에 모델 밖 CPG, 시간 위상, 교사 명령, root 힘, 몸체 보정 또는 해부학적 연결 우회를 넣지 않는다. 이 실험의 고정 감각 drive와 교사 label은 학습 진단에만 쓰이며 final runtime 기능으로 연결하지 않는다. 전체 BANC 신경망의 노드·연결 provenance를 유지한다.

## 바꾸는 것과 고정하는 것

새 owned product는 `tools/verify_c_banc_m3_reference_parameter_direction.py`와 `tests/test_c_banc_reference_parameter_direction.py` 두 파일이다. E342의 pinned `assemble`, `validate_packet`, `metadata_packet`, `load_selected/load_packet`, `patch_weights`, `receipt_dict/work_counts`, `reconciliation`, native guard 및 GPU sampler의 검토된 primitive를 필요한 범위에서 재사용할 수 있다. E342의 `gate`, `run`, `worker`, `collect`, CLI, 과거 자격 또는 65/60/48초·6 GiB/8 GiB 제한을 새 authority로 호출하거나 상속하지 않는다. 과거 runner/test, reader, 신경/decoder/rollout 소스, supervisor, 원자료, 과거 실패 보고서를 수정하지 않는다.

고정 모델은 158706 nodes, 11584852 edges, original 725849 selected anatomical edges, 8547 efficacy groups, 7136 premotor population, 391 motor neurons, 96 muscles와 42 joint outputs다. original coefficient, threshold, tau, gain, inverse adaptation tau, output mask, calibrated decoder, original neutral과 native rounding/FTZ 규칙을 그대로 쓴다. 시작 상태·감각 drive·decoder boundary는 네 창 모두 E342/E333 original packet 그대로다. 창 시작 c=(100,164,228,292), 각4 control, control_dt=.005, 20 neural ticks/control, neural_dt=.00025를 고정한다. 초기 state와 drive는 gradient에서 분리한다. native float32 forward에 대한 backward는 기존 real-equation surrogate VJP다. float32 양자화 자체의 미분이나 완전한 native Jacobian을 입증했다는 표현은 금지한다.

E353 label은 E335 몸체 경로의 c0..100 prefix에서 만든 slower cadence(2.382760007990865 Hz) 명령이다. 반면 이번 neural 시작 상태와 감각 drive는 E318/E333 neutral provenance다. 이를 같은 physical trajectory 또는 동일 몸체 초기조건의 paired replay로 부르지 않는다. 목적은 이 조건부 신경 출력이 유용한 목표 명령 쪽으로 변경 가능한지를 확인하는 것이다. open-loop 고정 drive와 짧은 reset 창의 개선이 감각 closed-loop 안정성, startup, 장기 bout 또는 F4/F5를 증명하지 못한다는 경쟁 설명을 기록한다.

## 유한 input scope와 gate

ROOT의 별도 `parent-e356-reference-direction-registration.json`에 E342 freeze의 정확한23 role bindings/selected_schema와 source/metadata pins를 상속한다. 이것은 원자료를 새로 읽어 확인한 inventory가 아니다. phase/source/decoder identity는 pinned E342/E333/E318 metadata를 통해 확인한다. E353의 실제 independent audit, final decision, closed worker seal, run/audit full-return JSON을 source-pinned metadata로 명시한다.

추가 원자료는 E353 `cadence-plan.npz` 하나의 `command` float64[301,42], `control_tick` int64[301], `control_time_s` float64[301] 세 selected members뿐이다. E332 normalization은 기존 reference role에서 읽은 같은6 양수 상수를 사용한다. 실제 `control_time_s` key는 E353/E355의 닫힌 등록 schema에서 상속해 literal로 고정했다. tick0..300, time=tick*.005, finite/C-order/dtype/shape, exact source/size/hash와 label c+1..c+4 정렬을 검증한다. E335/E354 raw, body arrays, extra NPZ payload, 일반 glob, gzip 또는 historical Dependencies/StateStore/GraphStore를 읽지 않는다. 기존23 role의 header-only extras는 pinned E342 scanner의 원래 bounded header 관찰 범위만 허용하며 value 읽기는 listed schema로 제한한다. 추가 label은 selected header/value만 열고 ZIP directory extras는 payload/header를 열지 않는다.

passive import/argparse/design/reflection/source/JSON inspection은 Torch/NumPy science, CUDA context, 원자료 hash/header/value, native 모델, body, optimizer를 호출하지 않는다. 작은 현재 source/JSON authority만 읽는다. 원자료/실제 Torch/CUDA는 새 gate 뒤 sole owned worker에서만 허용한다. 새 gate는 현재 owned2와 보호 source pins, ROOT 반영된 설계/Astra review, 현재 complete-case 자격, literal source/metadata/array scope, fresh output, 실제 original 잔여와 최종 supervisor contract를 검증한다. pre-spawn과 pre-array에서 같은 최종 contract의 필수 하위 필드를 검사한다. 누락/unknown/형 변환/bool-as-int/상충 pin/alias/traversal/중복 JSON/NaN/overflow는 실패 처리한다.

원자료 binding을 사전 fresh hash/header/value로 확인하는 숨은 dry run을 두지 않는다. worker가 각 selected value와 실제 read count를 수집한다. protected legacy loader의 원래 pre/post hash ledger는 변경하지 않고, label reader는 once-cached hash/header/value로 읽는다. worker 종료 후 결과 audit가 새 raw만 별도로 읽으며 inherited original inputs를 다시 열지 않는다.

## 새 손실과 계산 순서

첫 gate 이후 frozen `assemble`로 새 CUDA 모델을 만든다. 새 native neutral θ=ones float64[8547] leaf와 원래 states/drive/output mask로 네 창을 실제 계산한다. q/tendon/bank/activation의 기존 tolerance, rate/adaptation의 exact bytes, original boundary/clock/shape/S/C/total/LR를 E333 endpoint와 검증한다. 네 창 모두 neutral fidelity를 통과하기 전에 어느 창의 neural backward도 요청하지 않는다. q와 end state를 실제 획득한 시점에 raw/counters를 기록하고 모든 실패에서도 acquired partial evidence를 보존한다.

타깃 t_w는 E353 명령의 `[c+1:c+5]` literal slice다. q_w와 t_w shape=[4,42]. n_j=normalization[floor(j/7)]다. label/norm은 단 한 번 float64 CUDA tensor로 clone/detach하고 target parameter/gradient를 만들지 않는다. hot loss에서 host 변환, `.item()`, index search, teacher/cadence runtime query를 하지 않는다.

각 창의 기여는 `R_w = sum((q_w-t_w)^2/n_j)/(16*42)`, 전체는 `R=sum_w R_w`다. 따라서 `gq_w=2*(q_w-t_w)/(16*42*n_j)`이며 window mean4에 또 mean4를 곱하거나 full300 분모를 쓰지 않는다. 16개 output row의 mean이라는 의미를 고정한다.

각 neutral graph에서 실제 `autograd.grad(R_w,q_w,retain_graph=True)`로 새 direct-q cotangent를 획득하고, 닫힌 식의 독립 NumPy 계산과 비교한다. 이 q-target 요청은 neural 작업이 추가되지 않았음을 actual rollout counters로 확인한다. 이어 `R_w.backward()`를 정확히 한 번 호출해 새 θ leaf의 gradient를 얻는다. 총 네 실제 R neural backwards, 네 gtheta_w[8547], 합 `g_R`를 저장한다. q-only leaf selfcheck 또는 예전 C/total/R/qgradient를 gtheta의 대체물로 인정하지 않는다. 매 요청과 return/recompute/control/tick/block을 실제 finally receipt에 기록한다. backward error를 not-requested로 바꾸지 않는다.

θ 후보는 다른 optimizer/line search 없이 `theta_req=1-.001*g_R/max(abs(g_R))`, `theta_actual=float32(theta_req)`를 정확히 float64로 promote한 하나뿐이다. nonzero finite gradient와 실제 representable change를 요구한다. positive theta, selected sign 보존, all unselected weights unchanged, original anatomical group mapping, native f32 multiplication과 전체 unique palette<=16384를 검사한다. palette 계산은 이 one candidate의 선택된 weights와 full weight identity를 검사하는 primitive만 호출한다. failing native deployability면 proposal neural forward를 하지 않는다. flat/zero quantized direction은 valid-inconclusive terminal로 보고하고 sweep/epsilon 변경을 하지 않는다.

그 하나의 candidate를 같은 네 original start/drive에서 실제 forward한다. graph/backward는 만들지 않는다. 모든 forward와 acquired q/tendon/end-state/실제 selected weight SHA를 기록한다. neutral+candidate 총32 controls/640 forward ticks가 최대다. 네 R backwards는16 checkpoint blocks/16 recomputed controls/320 recompute ticks가 최대다. 실제 counters가 예정보다 작거나 다르면 완료 판정을 하지 않는다. body/CPG/teacher stepping/fits/optimizer loops/full sequence/CPU fallback/두 번째 sign arm/추가 epsilon/재시도는0이다.

## 사전 판정식

별도 ROOT/owned independent analysis는 새 raw의 float64 NumPy만 사용한다. window identity, actual gq의 닫힌 식, g_R=sum gtheta_w, actual f32-promoted theta, R/O/Q/Q_exact/T의 산술과 raw-대-manifest/ledger/work/source/closure 기록의 정합성을 독립 확인한다. full original parameters/anatomy/sign/unselected weights/full palette는 original packet을 읽은 pinned worker의 검사이며 auditor가 원본을 독립 재계산한 것으로 부르지 않는다. `independent_original_deployability_recomputed=false`와 `full_parameter_Jacobian_independently_proven=false`를 유지한다. target/norm/window slice의 C-order digest는 worker의 실제 selected read/slice ledger에 연결한다.

`O_w=R_candidate_w-R_neutral_w`, `Q_w=sum(gq_w*(q_candidate_w-q_neutral_w))`, `Q_exact_w=Q_w+sum((q_candidate_w-q_neutral_w)^2/n_j)/(16*42)`, `T_w=dot(gtheta_w,theta_actual-1)`로 고정한다. Q_exact와 실제 O는 quadratic identity tolerance rtol1e-10/atol1e-12로 같아야 한다. 이것은 loss의 대수적 consistency이며 neural Jacobian 증명과 다르다. 전체 dot(g_R,delta_theta)=sum T도 같은 tolerance로 확인한다.

H356의 제한된 조건부 지지는 `I_R=-sum O>1e-9`, `-sum T>1e-9`, 기존 E342 reconciliation의 same-sign/aggregate relative error<=.05/window L1 relative error<=.05가 O/Q와 O/T 모두 통과할 때만 가능하다. predicted와 actual을 실제 f32 delta로 비교하며 requested double delta를 대신 쓰지 않는다. S/C/total/LR의 actual neutral/candidate 변화도 별도 보고한다. cadence label과 위상 자유 proxy는 다른 목적이므로 S 또는 C 증가 자체를 R의 조건부 지지로 덮거나 R 실패를 proxy 성공으로 바꾸지 않는다. 이 실험은 candidate를 runtime에 채택하지 않는다.

유효 completed run에서 R non-decrease 또는 VJP/output response 불일치는 H356 refuted 또는 explicit inconclusive(meaningful signal 없음)다. 잘못된 neutral replay, unknown work, resource/deadline/source/publication/closure 실패는 technical_invalid/H356 not_evaluated다. surrogate가 짧은 고정 창의 개선을 예측했다는 것까지만 보고한다. full parameter Jacobian, natural gait, BANC autonomy, F1–F8, biology와 milestone completion은 false다.

## 실제 자원·종료·출력

현재 original cumulative=9872.876823299503/9900초, 잔여27.123176700497424초, reserved/jobs0이다. 9900을 reset/extend/waive하지 않는다. E356의 prospectively reserved allowance는 worker actual outer<=22초와 independent audit actual outer<=3초, 합25초 한 번이다. worker sole E267 child<=21초, import entry부터 internal<=20초/science<=18초; audit child<=2.5/internal<=2초다. 출판/hash/flush/fsync/close/readback/sampler close/stdout/실제 process tree 자연 종료까지 포함한다. ROOT outer witness/actual tool full-return 중 넓은 실제 비용을 한 번만 charge하고 reserve를 해제한다. 좁은 E267 elapsed만 charge하지 않는다. 실패 비용도 포함하며 worker/audit 각각 retry0이다. worker fail이면 dependent audit를 실행하지 않는다.

E342의 실제 original CUDA run은 48 forward controls+16 recompute controls에서22.1362958초, audit1.7491206초였다. sampled tree RSS2114306048 bytes, CUDA allocated553692672 bytes, context-included additional GPU peak902823936 bytes였다. 이번32 forward+16 recompute의 제한은 이 근거로 선택한 bounded feasibility test이며 시간/메모리 성공을 보장하지 않는다. native 원래 소스/CUDA cache가 변경되지 않은 현재 경로만 쓴다.

original host tree RSS cap=2 GiB, CPU threads1, additional GPU sampled peak+2 MiB quantization allowance<=1 GiB다. GPU0의 pre-context total-device baseline과 owned sampler/process/thread termination을 기록한다. total-device samples는 외부 작업의 완전한 배제 또는 unsampled peak 보장이 아니다. arbitrary old8GiB cap을 상속하지 않고 새로운 cap을 전후/마지막 resource check에서 검사한다. sampler를 닫은 다음 살아있는 owned thread는 completion을 무효화한다. 원래 PID-create-time identity supervisor와 ROOT outer witness를 재사용하고 unowned 프로세스를 죽이지 않는다.

worker output `verification/banc-natural-gait-m3-20260919/e356-reference-parameter-direction/`는 fresh다. raw.npz<=16 MiB, result/receipts 각256 KiB, selected-ledger512 KiB, manifest32 KiB, sampler completion32 KiB, stdout256 KiB, stderr32 KiB, emergency failure128 KiB와 ROOT/audit/closure metadata를 포함하는 recursive cap24 MiB를 고정한다. raw에 input 전체 CSR/teacher archive를 복사하지 않는다. publication before-first-write quota/exclusive create/flush/fsync/close/readback/deadline, primary-first-error 보존과 bounded emergency record를 구현한다. partial raw는 acquired numeric evidence만 저장한다. tuple/tensor/path/unsupported JSON은 명시적 report representation으로 변환하며 strict JSON serializer를 약화하지 않는다. success report 뒤 late close/resource error가 생기면 external success=false, scientific qualification=false가 되는 별도 revocation/failure record를 남긴다.

Sol FIRST+ONE R1 generated whole cost<=40초/retained<=16 MiB, ROOT FIRST/SECOND/residual/current qualification 합<=40초/retained<=16 MiB로 제한한다. 실패·Ruff·private pytest·native small CUDA·normalizer/passive를 포함한다. original과 generated 사용량을 혼합하지 않는다. original은 ROOT만 실행한다. Sol은 현재 source/JSON와 작고 explicit generated arrays만 사용한다. 순차 sole CUDA owner다.

## 구현 자격·리뷰 handoff

owned TEST의 private entrypoint가 실제 `pytest.main`을 한 번 호출한다. pinned empty ini, `--noconftest`, fresh outside-root basetemp, 전체 owned file와 ROOT가 등록한 whole extra files를 명시한다. inherited selection/plugin env를 실제 지운다. AST roster와 actual raw nodeid/item.path/정확한 current source hashes/모든 setup-call-teardown/실제 collection·run/typed requested=returned=1/exit0를 같은 outer full-return에 연결한다. `-k/-m/collect-only`, skip/deselect/collection error/fixture-only receipt, selfdeclared whole-case count 또는 pretend argv로 전체 실행을 인정하지 않는다. E355 R1의 ini-relative nodeid와 bool count 수정 원칙을 그대로 유지한다.

전체 mandatory cases는 아래16 함수로 고정하고 FIRST에서 더해진 case는 ROOT 승인 없이 제거/파라미터화/rename하지 않는다. ROOT extra는 actual path/INI 상대 경로에 맞게 별도로 등록한다.

1. `test_strict_metadata_and_finite_registration`
2. `test_passive_import_and_pre_array_gate`
3. `test_current_whole_case_qualification`
4. `test_final_supervisor_contract_required_fields`
5. `test_teacher_clock_and_four_window_alignment`
6. `test_independent_R_loss_and_direct_q_derivative`
7. `test_all_neutral_fidelity_precedes_R_backward`
8. `test_fresh_native_R_parameter_VJP_cpu`
9. `test_actual_f32_direction_and_anatomical_deployability`
10. `test_output_quadratic_and_parameter_response_analysis`
11. `test_actual_requested_returned_work_and_partial_failure`
12. `test_no_signal_and_palette_rejection_no_proposal`
13. `test_full_generated_publication_and_late_revocation`
14. `test_resources_absolute_deadline_and_sampler_cleanup`
15. `test_private_pytest_selection_nodeids_and_typed_counts`
16. `test_small_native_cuda_R_direction_end_to_end`

필요한 실제 작은 native CUDA case는 같은 native20 ticks/control을 사용하고 explicit TestDimensions만 허용한다. 임의 mock Torch/VJP/decoder의 성공을 native case로 등록하지 않는다. no-gradient/palette/neutral/late-error는 생성 데이터에서 실제 해당 branch를 완결하고 차단된 neural work count를 검증한다. root independent normalized MSE/derivative/quadratic response 계산이 구현식을 그대로 복제하는 selfcheck에 그치지 않게 closed-form fixture를 사용한다. source2의 current Ruff check+format --check와 actual whole-cases/current freeze를 닫는다.

ROOT는 새 유한 등록 및 설계를 실제 Astra6/max에 source/JSON-only review로 넘긴다. reviewer는 original/gzip/newraw fresh hash/header/value, Torch/NumPy science, CUDA/body/teacher/tests/외부 연구/추가 agent/writes를 호출하지 않는다. 지정한 review MD/JSON만 쓴다. 실제로 읽은 source/metadata와 SHA를 보고하고 inherited pins 전체를 reviewed라고 주장하지 않는다. ROOT가 받은 verdict/수정 사항을 reflection과 versioned 설계에 반영한 뒤에만 Sol 구현을 시작한다.

Sol6.1은 high effort(새 손실의 native VJP, 엄격한 gate, CUDA receipt와 deadline/publication의 결합 때문에)를 사용한다. writes는 owned2와 등록한 e356-sol-* 생성 증거뿐이다. FIRST API/source hashes, actual case/function/phase/소스 실행 proof, 모든 generated 실패 비용, retained bytes를 반환하고 original을 실행하지 않는다. ROOT FIRST에서 defects가 있으면 같은 Sol의 ONE R1으로 수정한다. ROOT SECOND 이후 남은 것은 현재 세션이 수정한다. 원자료 worker/audit는 이 순서와 current qualification이 닫힌 뒤 새 final contract/예약을 ROOT가 작성하고 실제 pre-spawn validator가 통과한 경우에만 실행한다.

## 다음 결정과 공통 원칙

실제 H356 조건부 지지가 나오면 동일 candidate를 보존하되 runtime 채택은 보류하고, 누적 비용·남은 allowance 내에서만 startup/bout/cycles/slip과 전체 closed-loop 검증의 다음 설계를 작성한다. 이번 after-window label loss 감소를 자연 보행의 개선이라고 발표하지 않는다. actual response가 실패하면 neutral fidelity, output algebra, parameter VJP/rounding 또는 native deployability 중 실제 발견된 원인에 따라 다음 in-scope 방법을 정한다. 무효 실행은 scientific refutation으로 세지 않고 같은 original attempt를 반복하지 않는다.

요청 outcome/기준·authorization·9900초 한도를 유지하고 기록을 지시로 해석하지 않는다. 근거 있는 개발은 계속하되 unknown usage를0으로 두지 않는다. 원자료와 과거 실패/최선 artifact를 보존하고, 주어진 scope 밖 실행·비용·조건 변경은 새로 승인된 것으로 간주하지 않는다. 기존 active plan/progress/checkpoint/verification 위치에 단계의 관측·해석·resource·남은 claim을 간결하게 연결한다. setup/review/test/report 통과로 unmet goal을 complete 처리하지 않는다. skill의 plan-and-execute workflow는 이미 사용자에게 승인되어 있으며 이번 reversible in-scope 구현 때문에 별도 확인을 요구하지 않는다.


## v2 actual Astra 반영의 구체적인 연결

M1: 반영 등록의 `raw_contract`가 complete normal26, no-signal15, palette-rejected18 member의 literal 이름/dtype/shape를 고정한다. quantized-zero는 candidate requested/actual theta를 획득했으면17 fields, 만들기 전의 zero-gradient는15 fields다. 부분 실패는 이 closed26 이름 안의 실제 획득 subset만 허용하고 window field 첫 축은 실제 prefix0..4, 고정 metadata/parameter 축은 literal이다. archive16MiB/member(header 포함)4MiB 및 finite/C-order를 검사한다. proposal 없는 branch의 O/Q/Q_exact는 not_evaluated/null이며 예측 T를 관측 응답으로 부르지 않는다. no-signal/quantized-zero/palette-rejected도 실제 완료된 네 R backwards와16 recompute를 기록한다. strict proof가 없는 partial을 completed branch로 승격하지 않는다.

M2: 새 ROOT E356 adapter와 이후 실제 SHA를 반영 source closure/최종 contract/current proof에 포함한다. `parent-e356-root-exit-witness.py`는 original_calls=1일 때 같은 ENTRY에서 child Popen 전에 owned `validate_supervisor_contract(contract_name, phase='pre_spawn', root=ROOT, timely=outer_check)`를 실제 호출한다. 함수는 passive source/JSON-only이고 passed=true, 현재 source2, 실제 contract_sha256를 반환해야 한다. witness는 같은 final C digest/current source2/phase/원자료 접근0를 bounded fresh `prefix+'-pre-spawn.json'`에 flush/fsync/close하고 검증한 C bytes가 그대로인지 다시 확인한다. C가 literal pre_spawn_receipt_output을 정하며 C self-hash 순환 pin을 만들지 않는다. worker/audit pre-array gate는 이 same digest/current source receipt가 있어야만 arrays를 허가한다. pre_spawn fresh와 pre_array에서 이미 열린 live observer/worker logs를 구분하고, audit baseline은 actual worker once charge 뒤 값이다. adapter hook은 최초 original 이전에 ROOT current generated review에서 source/API 및 실제 generated pre-spawn surrogate branch로 확인한다. 원자료 dry run은 없다.

generated fixture는 controlled outside-root basetemp/explicit TestDimensions를 사용한다. 그 안의 E342 prefix는 보호 synthetic_fixture의 요구 그대로 유지하며 E356 original namespace/CLI와 혼합하지 않는다. 기존 E351 generated workspace prefix 또는 original dimension 권한을 바꾸지 않는다.

M3: protected GPU sampler를 interval_s=.5로 생성한다. internal20 안에 pre-context baseline 포함 sample/owned sampler PIDs의 conservative cap42, 오류0, 실제 post-close thread 종료, 전체 E346 identity JSON<=8192 bytes를 확인한다. original host RSS의 비교 scope는 기존 E267이 실제 관측하는 supervised worker와 descendants이며 metadata wrapper PID/종료는 별도 기록한다. 전체 앱 RAM/unsampled GPU peaks/external workload를 증명했다는 표현은 하지 않는다. 모든 PID/error를 보존하며 초과를 잘라 PASS로 만들지 않는다. pre-spawn receipt/sampler/outer/audit/normalizer 및 emergency/revocation도 recursive24MiB의 advance reserve 안에 포함한다. cap/deadline/late-close 위반은 provisional science보다 우선한다. 추가 timing campaign은 없다.

literal API/CLI handoff: owned `validate_supervisor_contract(contract_name, *, phase, root=ROOT, timely=...) -> {passed: true, source_sha256: owned2, contract_sha256: current_digest, ...}`를 제공한다. passive freeze CLI는 `--freeze <reflected_registration> --qualification <Q>`; worker는 `--run --freeze <F> --review <R> --output <fixed_worker_dir>`; NumPy-only audit는 `--audit --freeze <F> --review <R_audit> --output <fixed_audit_dir>`다. original CLI에 dimension override는 없고 TEST private whole entrypoint는 기존 계획대로 별도다. 각 actual argv/phase/한도/경로는 최종 source-pinned C와 같아야 한다. freeze 작성도 actual current qualification 뒤 ROOT가 실행하며 original 예약으로 해석하지 않는다.
