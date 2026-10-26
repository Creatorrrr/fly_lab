# 실제 추종 계측 ROOT FIRST 검토

현재 actual GPT-6.1 Sol/xhigh의 science FIRST 안정판 Source를 읽고 확인한 초기 지적이다. observer32740B/SHA c0c4b1f0fcf884e7cdf7077fc829c2e437139c65f00b1e5f77e005836236cee6, analyzer31671B/SHA dbe0d0bfed7726503f29d7e3da342716c4baa82faa72c95050568233fd1e92f0, tests9546B/SHA35ab6355d279262eb445e3ce6f81d23d981d2d16a56ea642e047c0337daf339f. infra FIRST까지 대조한 뒤 같은 Sol에게 통합 sole R1을 요청한다. 실행·fixtures·actual capture는 아직 NOT_RUN이다.

- R1-1: analyzer의 uppercase LEGS와 `startswith('LF_')` joint name 추론은 실제 body/common의 lowercase leg key, namespace와 `c_thorax-lf_coxa` 같은 compiled 이름에 맞지 않는다. Source의 실제 adapter.joints(6,7) 또는 검증된 equivalent index map을 metadata에 저장하고, 여섯 leg/42 unique indices와 actual joint names의 leg binding을 확인한다. trace/AP/internal leg order는 lf/lm/lh/rf/rm/rh를 사용한다. fixture에 production 같은 namespace/hip 이름을 넣어 실패를 재현한다.
- R1-2: 기존 native 계약의 slip_mm_s는 invalid일 때 NaN을 허용한다. selected loader의 일괄 finite 검증과 compare_values의 NaN 통계는 실제 정상 입력을 거부하거나 nonfinite JSON을 만든다. 다른 numeric fields는 finite를 유지하고 slip만 validity-aware 처리한다. baseline comparison은 finite/invalid mask와 NaN 위치를 비교하고 finite values의 통계·invalid counts를 보고한다. nonfinite를0으로 대체하지 않는다. valid slip의 nonfinite는 technical-invalid다.
- R1-3: 현재 `temporal200` 이름의 pair는99→100..298→299이며 기존 temporal canonical200인100→101..299→300과 다르다. 여기서는 official post-boundary physical200_end100..299와 both-in-window199_end101..299로 정확히 명명하고 원래 temporal200이라고 부르지 않는다. native 창은 원본 native.time_s에 .5≤t<1.5를 적용하고 support threshold는 >=.02로 유지한다. control interval과 native post-time 창의 차이를 명시한다. 경계 fixture로 counts·threshold equality를 확인한다.
- R1-4: physics API의 interpolate default는 True다. continuous_joint_activation이 False일 때 kwargs가 생략되는 경로에서도 실제 default를 metadata에 표시하고 tau0이어서 held인지 구분한다. 요청 rad와 native ctrl unit/gear verification도 유지한다. 새 inference/physics 호출 없이 실제 source branch와 observed trajectory를 비교한다.
- R1-5: output cap의 범위는 result와 opaque supervision을 합친 논리적 durable bytes다. 새 tracking 세 역할에서만 결과 Publisher에3MiB를 예약하고 stdout/stderr를 각각1MiB로 제한하면 outer metadata 여유를 포함해 role96/96/16MiB, 전체208MiB를 유지할 수 있다. 기존 ray/saved/generated role의 cap 의미는 변경하지 않는다. serialization의 임시 host buffer도 기존 estimate와 host cap에 포함하고 실제 durable 합계를 기록한다. 검사만 선언해 무제한 로그가 별도208MiB 밖에 생기게 하지 않는다.

one-call/read-only copies, 네 pre-state/16 drives, initialized validity buffers, nested ownership 복원과 예외 보존 방향은 수용한다. actual old/new Source 계약 연결, 원장과 known close, publication/cap의 동작은 infra FIRST에서 따로 검토한다. FIRST snapshots는 고치지 않으며 runtime 실패가 남으면 sole R1 이후 ROOT가 잔여를 마무리한다. F1–F8와 기각 후보·M2/UI·누적 비용 보존.

## infra FIRST 통합 검토

ROOT가 helper22429B/SHA55843698622edad04e79991a8a1256cdefd01f407268a42b461533cbd32a67b9, supervisor28508B/SHA30e3eeb98ea10a51480d64528220436500a81c2d0677bebb5c0ea11a7c682f25, account44204B/SHA9a366a25cb2acac748133a8fd512b8bf53a9aee0ddada3084a6c1f426ae575a8, prepare11570B/SHA585ac088b16603c7967b8f85813bcd1d63ac6f9f2678e3f3f5a1940a322229bd, tests8513B/SHA0ecc93c613297edf53b6dc28ed807205cb75870b17abf7de2b5caf03bf31d4b6과 두 FIRST handoff를 읽었다. 같은 J11 history/generated tail·fresh before·ordered predecessor·작은 sibling role 연결은 수용한다. 기존 Source/read-only 제안은 실행 검증이 아니다.

- R1-6: 새 역할에서 pre-spawn Admission이 거부되면 현재 supervisor는 try/Job/output 이전에 예외가 나서 outer.json 없이 끝난다. Native witness가 과거 세 call만 다루므로 새 거부의 비용이 자동 출판·청구되지 못한다. 새 tracking 역할의 stdlib pre-spawn 거부만 실제 supervisor가 finite outer/streams metadata로 기록해 worker-not-started/Job-not-created를 명시하고 원본 배열0과 일반 worker 실패를 구분한다. 기존 역할을 재해석하지 않는다. Native commit은 이 실제 known-close 기록을 사용해 failure duration도 once 청구할 수 있어야 한다. Source-only/generation fixture에서 before-array refusal과 duplicate commit을 확인하고 runtime PASS라고 부르지 않는다.

Source closure는 실제 original body/native의 기존 finite names를 current pin으로 이어받으므로 현재246+추가 Sources를 다루되 과거 freeze를 덮어쓰지 않는다. 추가 서명/인증체계나 과거 Q/suite 재검증은 요청하지 않는다. generated 검증은 새 두 tests와 기존9 Job tests, 실제 lint로 제한한다. prepare canonical 명은 FIRST의 `parent-tracking-capture-prepare-native.ps1`를 채택한다. JSON float/type·Native creationticks 보존과 출력combined cap을 유지한다.

같은 running GPT-6.1 Sol/xhigh에 위 R1-1..6을 한 번의 sole R1으로 요청한다. 그 뒤에는 ROOT가 SECOND와 잔여 수정을 수행한다. 전체 목표 F1–F8은 ACTIVE/INCOMPLETE다.
