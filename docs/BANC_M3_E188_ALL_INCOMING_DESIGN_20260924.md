# E188: MN144018 전체195 incoming의 동일3초 native 관측

2026-09-24 · 원본 실행 전 설계 v1. 이전 E187은601 control/30000 native prefix·25경계 CUDA current를 검증하고 post 조건차이의73.16%가 미관측 잔여임을 확인했으므로 progress다. F1–F8·M2/UI 보존·F6 미노출·과거 실패를 유지한다. E187 원본4.8017342초를 포함한 누적4351.0849566997495/4994초, 잔여642.9150433002505초를 승계한다.

## 결정과 예측

큰 알려진 relay54014→MN144018를 일괄 감쇠하지 않는다. E187의 current 음−양 차이 .677326444682091 중 known positive .0598619273496794, known negative .121915196429412, residual .495549320903003이었다. 이 진단 차이의73.16%를 실제185개 미관측 입력과 float32 누적차로 나누는 것이 이번 단일 결정이다. A는 특정 기존 입력의 시간 변화가 잔여 차이를 설명하는 경우, B는 입력이 서로 상쇄되거나 모든 사건의 일관성이 약해 한 연결을 고르기 어려운 경우다. A에서도 해당 입력이 물리적 제동 원인이라는 증명이나 개입 성공은 아니다.

E150의2000×446 및 E185의601×391 관측 범위와 E187 observed-pre mask를 확인했으며 대상195개 중185개의 연속 기록이 없다. E03/E39는 다른 모델·입력 조건이고 E178은0/1/1.1/10초 경계뿐이므로 현재 사건의 시간열로 사용할 수 없다. 이번 관측은 같은 model/initial-state/CUDA/600controls/30,000native를1회 추가한다. 새로운 파라미터·fit·효능/필터/위상 선택은0이다.

## 구현 경로와 동결

새 `tools/observe_c_banc_m3_mn_inputs.py`, `tests/test_c_banc_mn_inputs.py`만 실제 gpt-6-sol/high가 구현한다. high 이유는 현재 실행에 대한 비침습 observer, 정확한 CUDA current 및 시계·실패보존 검증이며 새 CUDA 수식은 구현하지 않는다. 부모 설계→Sol 구현→부모FIRST→필요시 같은Sol첫수정→부모SECOND→부모잔여수정이다. 부모가 metadata/freeze/source gate/감독/독립 검산/결과를 소유한다.

부모 metadata는 E187의 검증된195 edge/pre/root/type/contact/실제float32 weight 순서를 고정한다. 기존 E181 freeze/review, E185 승인된 baseline 및 E186 force-conditioned bins/E187 incoming outputs를 묶는다. E181의 `verified_inputs(...mode='baseline')`를 실제로 통과한 gate를 사용하며 검증 객체를 위조하거나 원래 source gate를 우회하지 않는다. 그 뒤 기존 graph와 E150 candidate-neutral을 로드하고 `e181.run_arm(theta=(0,)*6, wall_limit_s=100, process_began=entry_start)`을 정확히1회 호출한다. E181/E150/shared runtime 파일을 수정하지 않는다.

관측 연결점은 `e181.e150.make_observer`다. 원래 factory와 반환 callback을 보존하고, callback이 원래 관측을 성공적으로 반환한 직후 별도 neural readout을 기록한다. 이 callback은 시작 anchor와600번 control 종료에서 호출된다. 모든 경로에서 factory를 finally 복원하며 기존 callback의 예외·반환값·부분 native 기록을 숨기지 않는다. 다른 session/보행 client를 생성하지 않는다. 기존 native health/startup99/clock guards는 그대로 유지한다. E185 force observer를 다시 설치하지 않으며 native body 결과가 일치한 뒤에만 기존 E186 force bins를 재사용한다.

처음 callback에서 실제 graph.n158706/deviceCUDA/packed implementation/clock0, network·adapter·body identity를 검사한다. 실제 runtime `network.weights` CSR의 해당 행 시작/끝·pre·float32 values를 동결195행과 비교한다. packed column/weight table의 해석도 기존 packed 구현과 일치하는지 확인한다(전체행 원래 edge 순서 유지, 행의 일부만 계산하지 않음). 이 검사는 CPU 또는 reduced network로 대체하지 않는다. 출력 `runtime-provenance.json`에 결과와 실제 장치/시계를 남긴다.

## 관측·분석 식

1. control0..600, time i*.005, neural tick i*20, physical time i*.005를 읽는다.195개 pre의 실제float32 rate와 output_mask, MN144018의 rate/drive/adaptation 및 `_current(network.rate)[144018]`를 기록한다. `_current`는 저장 endpoint rate를 기존 packed CUDA kernel에 넣어 읽는 관측이다. 적분기의 모든 RK stage 전류를 기록했다고 주장하지 않는다. shape/dtype/finite/row order와 callback 순서·개수601을 검사한다.
2. 시작/종료 clock 및 state가 읽기 전후 달라지지 않아야 한다. `_current` 자체는 rate/drive/adaptation/mask/weights를 수정하지 않으며 readout tensor를 copy해 저장한다. 전체상태 추가 계산/관측 전용state복원/추가neural step은0이다. 실제 sensor encoding/근육/controls/신체물성은 변경하지 않는다. rate나 mask를 모델에 다시 쓰지 않는다.
3. 관측은 `incoming-signals.npz`(control_tick,network_tick,time_s,pre_ids,edge_ids,weights,pre_rate,pre_mask,post_rate,post_drive,post_adaptation,post_current,complete)로 anchor포함601행을 남긴다. 실패해도 수집한 부분 rows/complete/error와 원래 baseline raw를 보존하며 실패를 재실행 성공으로 바꾸지 않는다. 이전 출력을 덮어쓰지 않는다.
4. baseline trace 전체601samples, native-recruitment 모든601행, native-contact 모든30000행, 초기/최종 상태의 신경/adapter/body 값은 E185와 정확히 비교한다. 상태는 배열 이름과 wall_s만 무관하며 실제 배열 의미가 같아야 한다. E181 gate bookkeeping/프로파일시간은 물리값과 분리한다. 목표관절/tendon/startup99를 새 comparator로 재정의하지 않는다. `valid_3s_diagnostic`·startup pass·3초/12000neural/30000physics·fault 없음·저장 성공을 모두 요구한다. 한 항이라도 달라지면 force bins를 붙이지 않고 실패로 판정한다.
5. E178의 변경 없는 `warp_current`로195 incoming의 float32 CUDA 누적을601시점 모두 재현해 실제 GPU current와 bit-exact 비교한다. packed Torch pre-mask subnormal 보존과 CUDA FTZ/fmad=false/32lane 순서를 지킨다. 설명용 항은 float64(weight)*float32(rate*mask)다. 모든195개 항의 합, 이전10개 known합, 이전185개 unknown합, `GPUtotal-descriptive_full_sum` rounding residual을 각각 저장한다. E187의195 edge와 이전10개 contribution/current/rate/drive/adaptation의3초 기록이 동일해야 한다. unknown을0으로 채우지 않는다.
6. `E187residual = new_unknown_sum + rounding_residual`을 검증한다. 항합/조건평균 보존오차 절대1e-9/상대1e-12 이내, CUDA total은 tolerance없이bits로 비교한다. 모든195항을 edge ID/root/type/contacts/현재weight·provenance와 함께 출력하며 큰 항과 차이의 정렬은 설명용이다. selective discard나새label/lag/slope/threshold탐색은0이다.
7. E186의11사건, 고정8선택[131,219,308,391,433,480,521,564],52window, RM native_count와negative magnitude 가중치를 그대로 적용한다. bands pre[-20,-10),lead[-10,0),post[0,10),recover[10,21); whole1..600/evaluation100..599/startup99/selected8/개별11사건을 전부 보존한다. 조건 평균은각항의유효count 가중치로 계산하고 분모0은NPZ NaN/JSON null·count0이다. magnitude 및 일반 시간평균도 별도 기록한다. event433 RM 음표본0과521동일control의양/음겹침을 반례로 유지한다.
8. 최종 산출물은 baseline/ 원자료, incoming-signals.npz, runtime-provenance.json, validity.json, analysis.json, aligned.npz, terminal.json 및 외부 감독 영수증이다. 어떤 새입력이 평균차이를 설명하는지 보고하되 한trajectory의 조건부 차이를 독립표본·생물학적인과·근육힘으로 부르지 않는다. 새후보효능/10초보행은 별도사전설계에서만 선택한다.

## 자원, 실패와 검증

원본 native1회/3초(12,000neural·30,000physics), GPU1, CPU worker1/BLAS1, automatic retry0이다. 부모 supervisor는새도구1process와소유한descendants만 관리한다. 실행145초+cleanup5초의총150초를모든hash/import/readiness/모델/쓰기/분석에적용한다. run_arm의내부100초시계(10초정리reserve)를완화하지않는다. 새도구내부총140초, 신규출력2GiB/기존baseline raw1GiB/여유disk20GiB상한이다. 기준실행과원본budget은새로시작하지않는다. 원본 actual supervisor wall을실패포함4351.0849566997495에가산한다. 부모순수검산·준비·합성tests비용은별도기록한다.

GPU 준비확인은기존E182 `_readiness()`의2회queue/client검사를사용하고 실행중자원모니터를보존한다. 기존확인에현재GPU telemetry를함께남긴다. 살아있는다른작업을임의종료하지않는다. supervisor는PID와생성시간으로owned만구분하고 이미Popen.wait로reap된child를살아있다고판정하지않는다(E185후처리race를반복하지않음). monitor오류/실제자원겹침/timeout/원본체인예외/미완료raw를별도로보존하고성공으로바꾸지않는다.

source/input/design/metadata SHA와parent approved review를CLI가원본모델생성전에요구한다. 합성tests는observer반환/예외/후크복원·clock변경/누락·CSRpre순서/weight변조·packed decoding·CUDA lane/FTZ·195known/unknown/rounding분해·zero-force조건·stale-source/hash·기존출력충돌을검증한다. 부모는별도수식으로실제601current와조건별기여/반례를검산한다. 관측기능테스트와자료유효성은보행개선이나M3완료가아니다. 현재설계시점원본E188=0, F6/UI변경0, 후보미채택, 목표active·미완료다.
