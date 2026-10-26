# E303 양수 범위를 보존하는 회복 신경 적분기

Root 설계, 2026-09-28. E302 실제 첫 오류는 tick10/stage3/source96198의 r=-3.3061957083191373e-7, flag4였다. 앞선39 stage 모든 g=1, R는정상이다. 원래RK4 중간 상태와 resource r>=0 계약의 충돌이므로 세포 파라미터·회로·u/tau를 조정할 근거가 아니다. E301/E302 결과와 비용은 동결한다.

이번 결과물은 새로운 **SSPRK(3,3) 공동 신경/resource 엔진과 tiny 자격 검사**다. 아직 실제 BANC/body 원본 실행을 예약하지 않는다. 현재 원본누적8144.108421799498/8210초·예약0, F1–F8/M2/UI/F6미노출 불변이다. Root 설계→실제gpt-6-sol/xhigh 구현→root FIRST→필요하면같은Sol R1→root SECOND→root잔여수정이다. 신규 `tools/banc_m3_ssp_recovery_synapses.py`, `tests/test_c_banc_ssp_recovery_synapses.py` 두 파일만 agent가 소유한다.

문헌의 SSP convex Euler 분해를 참조한다. [Gottlieb/Ketcheson/Shu 논문, section1](https://www.davidketcheson.info/assets/papers/sspreview.pdf), [저자 Numipedia SSPRK33 계수](https://ketch.github.io/numipedia/methods/SSPRK33.html). 아래 ODE의 범위 보존 조건은 root의 이 모델에 대한 적용이며 문헌이 BANC 생물학이나 보행을 검증한 것은 아니다.

연속식은 그대로 둔다. `T=max(0,cap*tanh((a/cap)*(drive+W*(mask*r)-threshold-z)))`, `rdot=(T-r)/tau_mem`, `zdot=(gain*r-z)/tau_adapt`, `Rdot=(1-R)/tau_R-u*r*R`; source 단계에서 `g32=float32(min(1,R*(1+tau_R*u*r)))`. 기존 선택 간선에만 g를 적용하고 u0 또는 g1은 원래 `weight*masked_rate` 연산을 그대로 쓴다. Premotor/source/motor 집합과 CSR의 explicit zero·byte·warp합 순서, 각 세포의tau/gain, 자원u/tau와 신규source영역은 변경하지 않는다.

변경은 시간 적분법이다. 전체 상태 y=(r,z,R)에 같은 stage를 적용한다. Euler map E(y,h)에 대해 `y1=E(y0,h)`, `y2=3/4*y0+1/4*E(y1,h)`, `y_next=1/3*y0+2/3*E(y2,h)`. 매 stage factor와CSR을 먼저 계산하고 r/z/R를 같은 단계 입력으로 갱신한다. R 출력이 r 갱신 전의 같은 source rate를 사용하도록 순서를 보장한다. Base y0를 마지막 갱신까지 보존한다. CUDA Graph에 캡처 가능한 고정3stage이며 Python stage별hostcopy를 production에 넣지 않는다.

Euler neuron map은 음수항의 뺄셈 cancellation을 피하는 동치형 `rFE=(1-h/tau_mem)*r+(h/tau_mem)*T`, `zFE=(1-h/tau_adapt)*z+(h/tau_adapt)*(gain*r)`를 사용한다. 자원은 `RFE=(1-h*(1/tau_R+u*r))*R+h/tau_R`, u0는 정확한R1을 유지한다. 이는 clamp나 음수 무시가 아니다. 원래 T의 정류 외에 r/z/R를 clip하지 않는다. 새 생성자는 finite 양수tau/cap, gain>=0, `h<=min(tau_mem)/2`와 기존 adaptation/resource timestep 제한을 엄격하게 검사한다. 원래 자원 조건 `h*(1/min(tau_R)+max(u)*max(cap))<=.1`을 유지한다. 이 조건에서 Euler map과 SSP convex 조합은 exact arithmetic에서 r,z>=0와0<=R<=1을 보존한다. 실제 float32/64 경계는 CPU/CUDA 검사로 확인하고 위반 시 실패로 남긴다. Source rate 범위 검사를 완화하지 않는다.

양수뿐 아니라 u0 neutral도 **동일 SSPRK3**를 사용한다. u0에서만 원래 RK4로 되돌리는 분기, 기존 identity 사칭, 원래 E150 neutral과의 bitexact를 약속하는 것은 금지한다. 새로운 수치방법은 같은 ODE의 다른 이산화이므로 neutral trajectory도 달라질 수 있다. 향후 body runner는 이 차이를 명시적으로 측정하고 원래 startup/short/F1–F8 기준을 그대로 적용해야 한다. 기존 E301 neutral exact gate를 그대로 재사용해 성공이라 부르지 않는다. 기존 calibration을 새solver에 맞게 fit하지 않는다.

새 schema/precision/identity `flylab.m3-ssprk3-recovery-resource.v1`에 3stage계수, Euler 동치식/연산순서, float32신경/float64자원, no-FMA/pointwise noFTZ/원래CSR FTZ를 명시한다. 원래 source_timescale_identity를 별도로 보존한다. 기존 E299의 pure selection·메타데이터 validation, CSR factor경로, constructor resource/provenance/pointer/restore 검사 등은 재사용할 수 있지만 모듈 global/과거소스는 수정하지 않는다. Subclass를 쓴다면 부모 constructor 중 동적dispatch/warmup/capture까지 새 적분기를 사용하고 초기화가 안전함을 검증한다. CPU neutral 분기가 RK4로 빠지지 않게 한다.

API는 `SSPRecoveryResourceNetwork`(기존 MotorTimescale 생성자+edge_ids/premotor_ids/utilization/recovery_tau), 해당 `resource_metadata/checked_resource_metadata/resource_identity/SCHEMA/PRECISION/SELECTION_KEYS/checked_selection`을 제공한다. 신경·resource snapshot/restore/reset/import_base_snapshot와 readout/advance_readout를 유지한다. 새 scratch/pointer mutation도 public advance에서 거부한다. Snapshot/restore는 새 identity만 받고 base import는 원래tick0/rate0/z0/drive0/mask1만 허용한다. 일반 restore는 fresh whole wrapper에서 사용하는 경로이며 신체를 생성하지 않는다.

기존 resource fault1/2/4/8/16을 유지하고 모든신경의 invalid/negative r,z를 검출하는 별도 bit32를 추가할 수 있다. 원래 오류를 NaN/clamp로 지우거나 정상 성공으로 바꾸지 않는다. 실패 플래그는device sticky이며 reset/유효restore만 해제한다. 성공 advance_readout는 원래 fusedpacket 한 번의 host copy를 유지한다. 소수점 양수/음수subnormal과 signedzero는 E299R1의 PTX conversion·비트 guard를 재사용한다. 실제 경계초과는 과거처럼 실패로 보존한다.

Agent 단계는 원본/fullBANC/CUDA/body/solver/fit0, 생성CPU toy≤64 cells/10000 logical ticks/60초 누적 외부실행/1.5GiB host/CPU1이다. 실패시도와 구성/setup/capture/step/벽시간을 전부 기록한다. Raw original E302를읽어 추가 fulltrajectory를 계산하지 않는다. 테스트는 작은 다중간선/마스크/비활성source/음양zero/subnormal 경계, 같은 단계r/R, 원래CSR byte/방향, snapshot/disk restore후미래exact, pointer/parameter변경거부, 잘못된상태·dt거부, sticky fault 및 singlepacket, CPU/CUDA 자격용 case를 포함한다. 생성 test 실제CUDA는 root 승인전skip이다.

수치정확도는 고정된 작은 smooth case와 사전 정의된 h,h/2,h/4의 수렴 검사로 분리한다. 이것은 후보 파라미터검색이 아니라 동일 방정식의 적분오차 검증이며 source중립/active를 모두 검사한다. CPU high-precision 독립oracle와 3stage식을 비교하고, 문턱을 지나가는 fixed toy에서 원래RK4 내부 음수가 발생하더라도 SSP의 모든 중간단계는 유효한지를 확인한다. 3stage/3CSR 횟수를 직접 확인한다. 허용오차는 구현 제출 전에 고정한다: 동일방법CPU/CUDA rate/z maxabs<=2e-5, R<=2e-8, rtol0; neutral u0 factor/R는 bitexact1; 한장치captured/uncaptured 및diskfuture는 bitexact. Convergence test는 반올림 floor에 걸리지 않는 작은 smooth **float64 독립oracle**에서 3차global order를 확인하며 production float32에서 3차 수렴을 억지로 주장하지 않는다.

Root FIRST 이후 실제 tinyCUDA는 별도 사전 계약으로60초/host2GiB/추가GPU1GiB/CPU1/≤3000logical ticks/≤64cells, root1회/실패면재시도0으로 제한한다. CPU/CUDA범위·계수·diskfuture 통과 후에만 실제 BANC의 동일 첫drive prefix 및 neutral/startup/body의 새 설계를 정한다. E303 tiny 성공은 fullBANC·보행·생물학·F2완료가 아니다. 여기서 수치범위/동등성/자원이 실패하면 원자료를남기고 root가 다음방향을재검토한다.
