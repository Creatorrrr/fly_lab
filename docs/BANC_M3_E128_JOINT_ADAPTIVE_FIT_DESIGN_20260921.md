# E128: 기존 연결 효능과 근육 적응의 공동 제약 적합

2026-09-21 · 부모 설계 · 원본 실행 전 계약

E127에서 전체391 MN의 native 단계 전류를 쓰는 CUDA 적응 응답·gain 감도·독립 참조가 통과했다. 그러나 큰 gain만 적용한 중심 사례는 초기99 q/LTM을 위반했다. E119/E120은 정적1572 edge 조정만으로 초기 자세와 후기 형태를 함께 개선하기 어려운 국소 한계를 확인했다. E128은 같은 초기 제약 아래 기존 edge 효능과 활동 의존적 MN 적응을 함께 조정하면 추가 개선이 가능한지를 시험한다. 고정 입력에서의 조건부 적합이며 실제 BANC 폐루프 개선을 가정하지 않는다.

## 입력과 같은 후보의 정의

E126의 E110/E113/E121 각각10초, 네 native RK4 단계의391 MN current와69 masked source rate를 사용한다. 초기 신경/적응/근육,391/96/42/6 순서,dt=.00025,20tick/control,EMA tau=.01,adaptation tau=.3,원래 tau/a/threshold/cap을 유지한다. E127 기준 응답과 E124 실패도 보존한다. 새 위상/CPG/시간 gate/신체 보정을 실제 운동에 넣지 않는다.

E121의 전체 calibration을 후보 기준으로 둔다. 문맥별로 값이 다른 기존 edge와 원래1572 fit edge의 합집합2444개 및69 source의 **명시적 ID**를 사용한다. 각 문맥 current를 `I_ctx + S_ctx @ (W121 - Wctx).T`로 옮긴 뒤, 선택1572개에 `W121_edge*(exp(delta)-1)`을 더한다. W는 네 단계의 실제 masked source에 곱한다. 이 식은 기록된 다른 presynaptic 입력을 그대로 둔 조건부 이식이며 E110/E113가 실제 E121 폐루프를 겪었다는 뜻이 아니다. E121 문맥의 기준 이식 항은 정확히0이어야 한다. 같은 중간 시각인 stage2/3를 합치거나5ms 보간으로 대체하지 않는다.

부모가 `e128-inputs.npz`에 E127 작은 생리/decoder 입력,2444 union/1572 fit ID 대응,세 context W,기존 signed 원가중치/효능/bounds/다리별 MN·근육·edge 소유권,기존 두 문맥 target/mask와 E121 own-phase target/mask를 동결한다. E121 phase는 기존 `quadratic_peaks(prominence=20)`/`phase_trace`/`own_leg_targets`를 같은6 E1 ID와 E85 기준·offset으로 사용하여 저장 rate에서 한 번 만든다. 이는 offline label이며 실제 제어 입력이 아니다. 첫 두 문맥의 원래 common1823/fixed1828/own mask를 줄이지 않는다. E121는 자기 own mask를 별도로 사용하고 두 원래 문맥의 판정을 대체하지 않는다. 고정 E111 geometry와 원래 q/foot/velocity scale을 유지한다.

## 변수와 판정

여섯 다리는 원래 서로 분리된 MN/pool/moment 소유권으로 계산한다. 한 다리의 변수는 해당 edge delta,해당 muscle gain 및 epigraph t다. delta bounds는 E115의 원래 ±log(1.1)과 efficacy [.001,10]의 교집합 그대로다. 기준 효능/선택1572 값이 E113와 E121에서 같은지도 부모가 확인한다. gain은 근육별[0,8],초기 delta=0/gain=0 하나만 쓴다. 원래 source/sign/edge/node/효능 상한/신체/decoder를 바꾸지 않는다.

각 문맥 loss는 기존과 같은 네 등가 가중치의 정규화 q/foot/adjacent velocity/LTM 오차다. 첫 두 문맥은 기존 common1823 own targets, E121는 자기 own target의 연속 mask를 쓴다. 목표 LTM=.2와 기존 scale을 유지한다. 목적은 `t + 6e-5*sum(delta^2)/1572 + 1e-5*sum(gain^2)/96`이다. 시작 t는 세 loss의 최댓값+.01이다. SLSQP는 정확한 목적/제약 Jacobian,ftol1e-9,maxiter100으로 다리마다 한 번만 실행한다.

초기99 signed q/LTM 모든 표본 및 RMS를 유지한다. 내부 caps [.018,.04,.009,.025],optimizer cap=(1-2e-4)^2,selector cap=(1-1e-4)^2를 그대로 쓴다. 부등식은 세 t-loss,3×99×7 q 양·음4158행,3×99 LTM 양·음594행,q RMS3행,LTM RMS3행의 합4761행이며 모두>=0이다. 원래 [.02,.05,.01,.03] 판정도 별도로 보존한다. 현재 peak만 골라 미분하거나 시작 sample을 제외하지 않는다.

같은 delta/gain의 float64 bytes는 캐시한다. t만 변경되거나 기존 점으로 돌아오면 응답을 다시 실행하지 않는다. 실제 모든 초기 내부 제약을 만족하는 기록 중 worst loss+prior가 최소인 점을 선택한다. solver success나 선형 예측만으로 채택하지 않는다. 한 다리라도 feasible 기록이 없으면 joint 후보를 만들지 않는다. 실패·한도 종료와 모든 노출을 보존한다.

## CUDA 전달과 미분

새 `tools/banc_m3_joint_adaptive_recruitment.py`는 E127의 검증된 네 단계 r/z/v/w 식을 재사용하며 current를 float64로 처리한다. native input은 float32에서 정확히 확장하고 작은 delta 전류를 float32로 다시 양자화하지 않는다. source/current/생리/decoder/edge mapping은 각 엔진에 한 번 전송하고 이후 delta/gain만 바꾼다. 실제 CUDA,private stream,소유 복사 및 requested/enqueued/synchronized/unknown 전송 기록을 유지한다. 전체/소형/다리 subset을 명시적 index로 지원하고 최대3문맥/160000stage/391MN/69source/1572 fit edge/96muscle를 할당 전에 검사한다.

기준 forward에서 각 native RK4 단계의 `phi_prime`를 저장한다. 같은 단계의 실제 r/z 상태로 계산한 값이어야 한다. MN의 근육 gain 감도는 E127의 v/w 식을 그대로 적분한다. edge e의 delta 감도 p=dr/d(delta_e),q=dz/d(delta_e)는

`p' = (phi_prime * (Wcandidate_e*S_pre_e - q) - p) / tau_post`

`q' = (gain_muscle*p - q) / .3`

를 같은 네 RK4 단계에서 적분한다. 단계별 source와 phi_prime를 각각 사용한다. 초기 p=q=0이다. source는 고정 기록이므로 재귀 입력의 미분은 포함하지 않는다. gain 감도는 근육별 pool을 통해 합쳐진 E127 activation_sensitivity를 사용한다.

신경 history를40,000-step Torch autograd graph로 만들지 않는다. CUDA에서 선형 edge sensitivity를 순차 적분하고20 tick마다 같은 EMA를 적용한다. 각 edge의 B는 해당 근육 activation에 대한 delta 미분이며 pool 계수를 포함한다. gain 열의 B는 해당 근육 activation_sensitivity다. 명시적인 parameter→muscle mapping을 사용하여 `Jq=.02*moment[j,m]*B`, `JLTM=.02*(1-tanh(activation_LTM/10)^2)*1[m=LTM]*B`로 조립한다.

고정 E111 geometry와 정규화 loss는 Torch CUDA에서 계산한다. activation을 독립 leaf로 한 loss gradient는 문맥 간에 섞지 않는다. 문맥 loss 합의 activation cotangent를 얻으면 각 문맥이 독립이므로 문맥별 parameter gradient는 `sum_time(cotangent[t,m]*B[t,param])`다. CUDA sensitivity kernel은 이 합과 초기99 B를 반환한다. 모든 시간의 거대한 parameter Jacobian을 CPU에 복사할 필요는 없다. CuPy/Torch 경계에는 실제 stream 동기화 또는 검증된 이벤트 인계를 사용한다. 작은 host 복사가 필요하면 크기와 비용을 그대로 기록한다.

각 unique 평가에서 full forward 한 번과 edge derivative bank 한 번,geometry/loss backward 한 번만 호출한다. 초기99 B,세 q/LTM errors,세 loss/네 parts/모든 ratios,loss gradients,delta/gain/IDs/parameter-muscle mapping/q coefficient/LTM state coefficient를 raw로 저장한다. 첫 평가와 기술 오류에는 full response도 남긴다. 계산된 raw를 수치 assertion 전에 보존하고 저장 실패·미확정 CUDA 실행량·원래 예외를 숨기지 않는다.

## 기존 기준과 실제 float32 후보 확인

원본 fit 시작 전 새 full engine의 native-current 모드에서 gain0 및 E127 center를 각 한 batch 실행하여 E127 원본의 여덟 field/branch를 비교한다(max1e-8/branch exact). 이 모드는 검증 전용이며 fit은 항상 같은 W121 후보 이식 모드를 사용한다. source×weight 이식은 작은 독립 NumPy/모든 edge ID 대응 합성 테스트로 확인한다. E121 delta0/gain0의 candidate 모드도 E127 gain0와 일치해야 한다.

그 뒤 같은 W121/gain0의 세 문맥 baseline을 full engine으로 한 번 계산한다. E121 자체 baseline의 own mask q/foot/velocity를 보존한다. 이 baseline을 얻기 위해 fit을 숨겨 반복하지 않는다. 모든 다리 feasible 선택 후 기존 `selected_actual_float32`의 ID 기반 calibration export를 재사용한다. 반환되는 legacy49열 dense 행렬을 새69열 current로 오인하지 않고 실제 updated calibration의 선택 edge 값을 사용한다. 모든 다른 calibration 필드와 선택하지 않은 값을 보존한다. gain96은 실제 native float32로 변환한 값을 다시 float64로 확장해 최종 conditional 평가에 사용한다. solver값/실제 float32 차이와 정류 경계를 보고한다.

최종 전체391/세 문맥 응답은 한 번만 계산한다. 첫 두 문맥은 기존 `frozen_output_metrics`의 E114 같은 문맥 W113 대비 q악화<=5%,foot/velocity 각각>=20% 개선,원래99/내부 per-leg 기준과 fixed/own/legacy 보고를 그대로 사용한다. E121도 자기 W121/gain0 baseline 대비 같은 q/foot/velocity 비율 및 초기 조건을 통과해야 joint screen 통과다. 각 legacy 절대/상대 판정과 실제 전진 여부는 구분한다. 새 source 계산기·기존 baseline 차이도 남기며 기존 수치를 새로운 baseline만으로 대체하지 않는다.

부모는 final candidate raw가 있을 때 한 번의 독립 NumPy 네 단계 응답(세 문맥30조건부초,120000 context tick)을 계산하여 최종 q/LTM/activation/rate/z 최대차1e-8 및 원래 판정을 검산한다. 모든 screen과 독립 참조가 통과한 경우에만 이후 별도10초 native full BANC+body 후보로 진행한다. 여기서는 실제 보행 채택/F6/제품 실행은0이다.

## 자원과 분업

한 원본 프로세스만 실행하며 재시도0이다. Sol은 새 위 engine,`tools/fit_c_banc_m3_joint_adaptive_recruitment.py`,`tests/test_c_banc_joint_adaptive_recruitment.py`만 구현한다. 결합된 CUDA 동역학·edge 감도·전체 초기 제약 적합이므로 gpt-5.6-sol/xhigh다. 부모는 데이터 준비·원본 supervision·독립 참조·리뷰를 소유한다. 첫 리뷰 뒤 같은 Sol 첫 수정,재리뷰 뒤 잔여는 부모가 직접 수정한다. 합성 시험에는 작은 native4-stage/정류 경계/억제·흥분 edge/서로 다른 gain·tau·mapping/직접 유한차분 Jacobian/4761 제약 일반식/캐시·선택·float32 export·부분 실패 기록을 포함한다. 원본 데이터로 숨은 테스트/fit은 금지한다.

원본은 최대100 unique(delta,gain)/다리,합600,다리30초(마지막3초 보존 여유),fit185초,프로세스 내부215초/외부230초/supervisor235초다. full response는 native gain0/center 두 개와 transported baseline 한 개,최종 후보 최대 한 개로 최대4 batch/120조건부초/480000contexttick이다. leg response 최대600batch/1800 leg-context response/18000 leg조건부초/72000000 leg-contexttick과 derivative bank 최대600회는 별도 계상한다. scalar MN/edge tick도 실제 subset 수로 기록한다. tiny CUDA setup 한 번은 별도다.

부모 final reference는 내부35초/외부45초/supervisor50초다. E127까지 누적1211.366730초를1212로 올려235+50을 배정하여1497<=기존 shared1530초다. 기존 원본/준비/합성/부모 비용과 실패를 지우지 않는다. 기술 오류면 남은 fit을 시작하지 않고 partial을 보존한다. 다리 soft cap은 이미 평가한 기록을 선택한 뒤 다음 다리로 진행하되 전체 한도는 넘지 않는다. 원본 입출력/source/설계 SHA와 외부 stdout/stderr/PID/exit/timeout/manifest를 보존한다.

이 단계의 통과는 조건부 joint fit의 지원 여부이며 전체 목표 완료가 아니다. 같은 최종 후보의 실제 F1–F8,원래 초기99/shape20%,E121 F4/F5 실패,F6 두 미사용 자세,기존6목표/정지·재시작/slip/M2 복원·영상의 검증 범위는 유지한다.
