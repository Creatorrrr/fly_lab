# E273 — 조건부 CUDA forward의 수학을 보존한 병렬화

2026-09-27 root 설계. E270 단회원본은유효하지만고정gate미달이고body승격0이다. 조건부forward+decoder의CUDA호출/대기149.914초,derivative21.135초,hostcopy.298초를기록했다. 현재forward는다리별약60–70MN을32thread/block으로실행하여2–3block안에서긴시간loop와매stage간선합을계산한다. 병목은기록된사실이며아래병렬화의속도효과는미검증가설이다. 전체BANC속도나생물학적모델을바꾸지않는다.

실제gpt-6-sol/xhigh가 `tools/banc_m3_source_coefficient_engine.py`와 `tests/test_c_banc_source_coefficient_engine.py` 두파일만수정한다. E269승인본은 `verification/banc-natural-gait-m3-20260919/e270-original-reviewed-sources/`에원본SHA로보존되었다. root FIRST→같은Sol 첫수정→rootSECOND→root잔여수정. 기존E269모든API/validator/오류·receipt/소유권계약을보존한다. 다른원본/신체/학습/계획파일을수정하지않는다. 에이전트는CPU생성검사/Ruff만실행하며CUDA는root예약후root가실행한다.

## 정확성 경계

원래f64조건부식·3계수·stage순서·E150decoder·tangent·chunk최대1000step·정밀도/마스크·입력/state는그대로다. 기존kernel을검증reference로남기고명시적constructor옵션 `forward_implementation='precomputed'|'reference'`를제공한다. 새기본값precomputed는root비교통과전원본실행하지않는다. reference는현재원본코드의32thread forward경로다. E270 frozen성과기준은변경하지않는다.

각(stage,MN)에대한조건부current는주어진c와고정source입력만으로결정되므로timeloop앞에서병렬로계산할수있다. f64배열[S,N]을준비하고각element에서원래edge_order대로동일한덧셈/곱셈/괄호/--fmad=false 식을순차수행한다. GEMM,다른합순서,낮은정밀도,근사tanh는쓰지않는다. current kernel은최대65536element씩나누어bounded launch한다.

precomputed forward는같은RK4/phi/branch식으로prepared current를읽는다. 매MN을하나의thread block(1thread)으로분리해다른MN의긴시간loop가GPU에분산되도록한다. 이것은순수실행배치변경이다. bank/activation/tangent는먼저기존경로를유지해불필요한동시변경을피한다. 동일입력/계수의모든rate,z,phi,branch,bank,activation,q,LTM,B와sensitivity는기존reference와**bit exact**여야한다. 부호0/NaN미완료영역은완료범위를기준으로구분한다.

추가current배열/실행temporary를기존4GiB사전할당한도에포함한다. `current` receipt bucket을lifetime/current응답에추가해요청/완료element·byte·wall을기록한다. 실패시완료current범위와부분current가보존되며neural완료범위는실제동기화된chunk만표시한다. 자동fallback/retry/autotune없다. reference옵션에서도명시적으로current-precompute0을기록한다. 기존host=False선택복사와productownership계약은변경하지않는다.

## 사전검증

기존생성CPU/CUDA검사를유지하고reference/precomputed의중립·비중립·binarymask·halfstage·서로다른inverseτ·fixed/variable/hingle·1000step경계를bit exact 비교한다. analytic미분은기존독립중심차분허용치를유지한다. current 첫/중간launch실패,neural첫launch실패,크기한도,정확한요청·완료receipt를검사한다. 큰GPU시험은명시적opt-in이다.

root matched 생성benchmark는고정seed의66MN/51source/300existingedges/16muscles/7q/1LTM/1200controls를사용한다. 원본자료는없다. 같은계수/입력·derivative=False로각경로warmup1회+측정3회,순서reference/precomputed를교대로한다. constructor/JIT시간과timed solve wall을구분하고반복별결과를모두남긴다. 같은hardware/software,output bit exact가필수다. median solve wall≥1.5배개선을통과기준으로미리고정한다. 이를전체BANC또는실제보행의속도개선으로부르지않는다. 미달이면최적화기본채택을보류하고원래경로를유지한다.

현재새원본예약0. 이후한같은후보의feasible-step학습은별도root설계/원본freeze로진행하며F1–F8/시작보호/개발비악화는그대로다.
