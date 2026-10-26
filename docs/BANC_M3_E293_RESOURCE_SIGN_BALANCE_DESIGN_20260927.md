# E293 — 저장된 재귀 전달 변화의 부호별 원인 분해

E292 고정 후보(u=.05, tau=.150, 기존 VNC premotor7136 사이725849연결)는 실제CUDA/신체에서 neutral exact를 통과했지만0.3124초 upright=.89993517<.9로 중단돼 기각됐다. 신경1280/물리3174, setup560/capture280/settle4000, supervisor49.2856196초. 누적7967.464174899509/8045초, 예약0. 원본 output/SHA/clock/neutral whole exact와 unhealthy partial component snapshots는 root audit로 확인했다. Partial 시점은 신경.315초, 신체.3124초이며 서로 같다고 보고하지 않는다.

기술적 오류가 없는 E292 실패를 바탕으로 다음 구조 변경에 앞서, 저장된 신경 activity와 원래 선택 CSR만으로 흥분/억제 전달의 변화량을 분해한다. 모든부호를 같은 resource dynamics로 바꾸는 것의 영향과 단순 전체감소를 구분하는 것이 목적이다. 초기 탐색에서 명령차이는 .05–.10초 최대.2425099rad, .20–.31초 최대2.2688762rad였다. 어떤 파라미터나 후보도 여기서 fit/선택/실행하지 않는다.

## 고정 질문과 계산

E292 candidate activity의63개 completed control0..62와 E150 동일control motor391/recruitment/trace, E292 native3124를 사용한다. control0..62 끝은.31초이며 .3124초의 partial neural state와 혼합하지 않는다. 원래 roster의 weights32/pre_ids/post_ids/source_ids/edge_source_slots로 해당 순간의 선택 재귀 전달을 계산한다. E292 observed-start network output_mask=1, partial-final output_mask=1을 확인하고 frozen 현재경로에는 mute가 없음을 source audit로 확인한다.

각 저장시점과 각 post neuron에 대해 float64 수학적 합으로 `J_exc=sum(w>0 w*r)`, `J_inh=sum(w<0 w*r)`, `J_exc_R=sum(w>0 w*r*R)`, `J_inh_R=sum(w<0 w*r*R)`를 구한다. 흥분 손실은 J_exc−J_exc_R≥0, 억제 완화는 J_inh_R−J_inh≥0, 순변화는 후자−전자다. 이는 **같은 저장된 rate에서의 순간 항 분해**이며 원래 baseline 실행의 실제 current 또는 모델 재실행/미래반사실/인과보행개선 증거가 아니다. CUDA warp/FTZ/RK4 exact oracle도 아니다. 원래 topology/current dtype/IDs/bytes는 바꾸지 않는다.

all63×7136 post 배열을 저장하고, 고정 t=0/.05/.10/.20/.31초마다 signed 총합, loss magnitudes, 순양/순음 post수, 각 성분 분포를 보고한다. net 최대양/최대음 각10 post와 실제 ID를 기술적으로 보고하되 이 순위를 runtime 대상선택에 쓰지 않는다. candidate motor391의 동일시점 baseline delta와 q42 command 차이, upright/native load, resource분포를 함께 보고한다. .5초startup가 관측되지 않았으므로 축약prefix오차를 startup PASS/FAIL로 부르지 않는다. native guard의 처음 upright<.9 또는nonfoot/비유한(각field 실제기록)을 확인하고 미측정구간의 oscillation/cycle/보행성공을 추정하지 않는다.

## 소유권·검증·한도

Root 설계→기존 actual gpt-6-sol/high가 신규 `tools/analyze_c_banc_m3_resource_sign_balance.py`와 `tests/test_c_banc_resource_sign_balance.py`만 작성한다. 기존 frozen/source/data/doc은 읽기전용. Root FIRST→필요시 같은Sol R1→root SECOND→root 잔여수정 순서다. API는 `decompose(selection, activity)`, `analyze(inputs, output)` 및 `--freeze/--freeze-sha256/--source-review/--source-review-sha256/--output` CLI. Frozen input별SHA/output/source review (approved/sourceSHA/emptyblocking) 검증 후 실행한다. 저장 분석 결과는 candidate_adopted/F1_F8_complete/F6_exposed/natural_gait_complete 모두false.

검증은 작은 손계산 회로에서부호/공유source/빈sign/neutralR1, ID행순서변조·dtype·shape·중복·nonfinite·outofbounds·resource범위·시계오류를 거절하고 입력불변성을 확인한다. Fullgraph/model/CUDA/body/solver/fit원본0. Agent 생성검사max60초1thread/1GiB, root별도검토와实际saved分析max30초/2GiB/output128MiB. 실제분석은승인된inputsource freeze뒤1회로 하고 전체원본비용변경없다. Input파일: E289roster, E292candidateactivity/native/recruitment/trace, E150trace/native-recruitment, E292 observed-start 및partial-final/network manifest/state/arrays, E292root-audit/result/ledger. 데이터미완료·잘못된ID/시계/핀이면fail-closed한다. GraphStore 또는 원래신경/신체constructor 불필요하다.

Root가 결과를 검토해 억제연결까지 일괄 약화한 구조의 문제인지, 흥분 감소·출력decoder·다른 구조가 주원인인지 다음 설계를 정한다. 어떤 결과도 기존F1–F8/startup/health한계를 완화하거나 기존 후보기각을 되돌리지 않는다.
