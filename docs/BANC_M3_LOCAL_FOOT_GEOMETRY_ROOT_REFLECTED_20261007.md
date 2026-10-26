# Astra geometry 검토 반영과 NEW Sol FIRST 범위

ROOT가 실제 GPT-6 Astra/max review JSON13073B/SHAc850b30b96099725427b9f3a569b88d3355b44b759699f4ef70ec8957d869d4e와 MD6741B/SHAa7a340c187caff7a986cf4dfa80ce407041620eaaabe826827f435592f2b13fe를 읽고 ACCEPT_AFTER_SMALL_CORRECTIONS와 세 교정을 수용했다. Source-only 검토이며 actual saved-output correspondence 외 새 body/CPU 실행은 NOT_RUN이다. 원형 ROOT design9870B/SHA69783b9849afcd461a1290eee884796a10c581e8c6dff288f39d914fadd817a6는 보존한다.

## 확정한 작은 수정

FK 불변 필드는 qpos0, body_pos/body_quat/body_parentid/body_mocapid/body_jntadr/body_jntnum/body_dofadr/body_dofnum, jnt_pos/jnt_axis/jnt_type/jnt_bodyid/jnt_qposadr/jnt_dofadr, dof_bodyid/dof_jntid/dof_parentid와 nq/nv/njnt/nbody/nmocap다. Active frontend42 ID/name와 actuator transmission/gear/ctrl clamp flags/ranges는 별도 바인딩한다. Runtime gain/bias 등 dynamic model 변화는 geometry invariance와 구분한다. MJB byte hash와 FK field hashes를 모두 보존하고 full-nv native/CPU J를 비교한 뒤 active dof42를 선택한다. Perturbation은 hinge qpos42를 사용한다.

AP는 start-thorax anterior axis와 world vector의 내적, vertical은 world+Z다. Start-thorax z를 추가하면 body-dorsal이라는 별도 열이다. 두 boundary 각각 자기 thorax frame의 relative AP를 또 별도로 유지한다. Delta residual은 **미귀속 운동학 잔차**이며 contact/servo/힘·root 원인으로 분류하지 않는다. Neutral-only local geometry를 candidate geometry로 적용했다고 주장하지 않는다.

네 c→c+1의 실제 적용 raw control과 compiled clamp를 사용하고 neutral의 tau0/held 경로를 확인한다. Interpolation이 있으면 final endpoint를 interval 대표값으로 승격하지 않는다. Actual two-boundary normal_load_bw>=.02 BW masks와 recorded boundary load bool를 별도 명시하고 같은 source 정의로 masks/분모를 만든다. Empty category는 UNOBSERVED다. 이 stage의4local transitions를200control/full stride로 확대하지 않는다.

Supervisor가 geometry_analyze를 subprocess 생성 전 OPENBLAS_NUM_THREADS/OMP_NUM_THREADS/MKL_NUM_THREADS/NUMEXPR_NUM_THREADS=1 기존 branch에 추가한다. CPU 분석기 import 뒤 env를 설정하는 것으로 대신하지 않는다. mj_sizeModel 결과를 배정 전에 typed positive/bounded size로 확인하고 buffer/copy/temporary/durable를 host/output cap에 포함한다. 정확한 binding save/load는 작은 generated synthetic roundtrip에서 검증한다. Native Model save는1개, CPU Model/MjData 각1개다.

별도 CPU forward는8 baseline+4anchors×42hinges×2perturb=344회이며 선택적 anchor restore4를 쓰면348회다. 각 perturbation마다6tip을 한꺼번에 읽는다. Read-only live anchor full-nv mj_jacBody는24회이며 extra live forward/step/neural0이다. Position/frame-point abs1e-9mm/J native-CPU-FD abs1e-8mm/rad/h1e-6rad/rtol0는 변경하지 않는다. Root/passive/mocap/model/start frame을 고정하고 FD에 ctrl clamp0, saved quaternion 사후 normalization0이다.

## FIRST 소유권과 고정 실행 계약

실제 GPT-6.1 Sol/xhigh에 NEW geometry FIRST를 맡긴다. Full physical context/kinematic mapping, source-lazy two-role admission, actual existing owner/cost 연결을 함께 다루는 난이도 때문에 xhigh를 유지한다. 이전 live replay sole R1은 종료했으며 이번 단계 FIRST→ROOT FIRST→필요시 same Sol sole R1→ROOT SECOND→ROOT residual 순서다. Sol은 새로운 parent-local-foot-geometry-sol-FIRST-* proposal/handoff만 쓰고 canonical/import/AST/Ruff/tests/Original arrays/CUDA/body/ledger/grants/추가 subagents0이다. 실제 검증은 ROOT가 한다.

기존 observer/continuous arm은 tracker factory/작은 hook을 넣어 재사용한다. 새로운 geometry capture/analyzer와 작은 admission/owner/prepare routing, 의미 있는 작은 generated fixtures만 제안한다. Broad infrastructure, whole old suite, replay repeat0. Shared supervisor/account 변경은 기존9개 실제 Job checks를 같은 bounded qualification에서 확인한다. 검증은 필요한 small synthetic model roundtrip/FK-J-FD/post-forward wrapper/partial restoration/pre-import thread1/lineage/once-refusal에 한정한다.

같은 J grant2227B/SHA7245e218f5c9abae27a849f880f37bc6013df87ae5e514947887183b4a2dba2f/offset165/inputs210093B/SHA4bdd075fc9980a275319fc0a26210c15335050ff7e96bc2fd29eb0ec5aeda151를 잇는다. NEW roster a7416B/SHAffb13299890e16fa50b928ea5e4b24932c9ca44e1743053fdf674b68c3947349와 historical replay archive index2126B/SHA354376566bdc119a92f338c0853a0c2911b2c1bddaf173362d1dc823c3c886e6가 완료한 actual replay b/현재 closed200 cost lineage/neutral recording/Source6 archive를 묶는다. Tracking c archive8과 이전 Source/rejected candidates/failed costs는 보존하고 현재 canonical을 역사 freeze로 다시 검사하지 않는다.

geometry_capture300/285/270/15초/host24GiB/output96MiB/GPUdelta+allowance8GiB/sample3000; geometry_analyze120/100/90/20초/host2GiB/output16MiB/CPUthread1. 결과에서 supervision3MiB를 예약하고 combined cap에 포함한다. Existing neutral same instance0→99→300의 정상6000neural/15000physics와 constructor 작업을 따로 기록하며 추가livecalls0, analyzer mj_step/body/neural/CUDA/gradient/newcandidate0이다. 정상 control-boundary full pose8/MJB1과 anchorJ4를 기록한다. Common trajectory exact/drift0를 실제 기존 neutral과 확인하며 mismatch를 historical correspondence PASS로 분류하지 않는다.

Current200debits/82directcalls/global11325.160960899506/16200/stage1431.4060000000002/6300/science1305.045/6000/generated155.67600000000004/300/sharedgenerated883.8880156000001/1700/allclosed0에서 새grant-reset0이다. Planned new Original caps420/worker1/restart0, generated total outer caps≤110(close reserve20포함)이며 actual genuine FINAL만 once 청구한다. Pre-spawn refusal/partial failed worker도 비용과 first error/valid bitmap/receipt/known close를 보존한다. No automatic rerun/threshold relaxation. Original 자료 분석은 CPU여도 Original 비용이다.

결과가 통과하면 ROOT가 실제4×6leg local 표를 읽고 contact-conditioned neural objective를 설계한다. 충분하지 않으면 미식별을 유지하고 최초 차이를 조사한다. F1–F8 ACTIVE/INCOMPLETE/M2/UI/provenance/unused F6/biology UNPROVEN, 기존 candidate REJECT_NOT_ADOPTED, root GOAL_PLAN unchanged를 보존하며 Source/테스트/국소 기하 성공은 보행 성공이 아니다. 최신 사용자 지시에 따라 추가 승인 대기 없이 진행한다.

## FIRST 중 metadata 계보 교정

Sol이 geometry roster a의 historical_closed_200_ledger descriptor가 mutable live path를 가리켜 향후 generated 비용 추가 뒤 그 hash로 다시 읽을 수 없음을 지적했다. ROOT는 아직 정확히 closed200인382156B/SHA9350e7a26124a77be6c2497995050477d945c8bab966ab2bbc7bd4bafccf4642 bytes를 `parent-local-foot-geometry-closed-200-ledger.json`으로 byte-exact 복사했다. 현재 원장이나 cost를 바꾸지 않았다. Roster b8114B/SHA42ac6c9363c648b4c819e68031fac6052bc61ada5519f4bc30d6b12ebf0cdaa3는 이 immutable copy를 historical_closed_200_ledger로 바인딩하며 a와 원래 live-path descriptor를 as-of 기록으로 보존한다. FIRST의 실제 roster pin은 b다. Historical closed200 bytes/costs와 current fresh before-ledger/actual lease를 구분해 검사한다. 이 metadata 교정은 새 실험·criteria/resource/Source 변경이 아니고 imports/numerical/Original/비용 증분0이다.
