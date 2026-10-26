# E126: 보행 적합을 위한 native 단계 입력 확보

2026-09-21 KST · 부모 설계 · 원본 실행 전 계약

M3 finite-goal의 F1–F8은 그대로다. E125의 실제 전체 BANC1.2초는 관측 투명성을 확인했고 단계 전류로 q 재구성 오차를99.97% 줄였다. 선택11 MN에서는 작은 변화량의 미분이 정류 분기와 일치했다. 아직 전체10초·세 문맥의 조건부 입력과 전체391 감도는 확보되지 않았다. 이 단계는 기존 노출 문맥의 입력을 확보해 근육 적응·기존 synapse 효능 적합을 가능하게 한다. 적합·보행 후보 채택·F6 노출은0이다.

## 필요한 입력과 보존 범위

기존 E110/E113/E121 calibration의130943 edge ID 순서는 같다. E121과 다른 값은 E110에서2329개, E113에서872개이며, 기존 적합 변수1572 edge와 합치면2444개의 기존 edge다. 수신자는389개의 기존 MN이고 presynaptic source는69개(기존49+target port20)다. 83877을 포함한다. 해부학적 노드/연결을 추가하지 않는다. 부모가 graph indices/indptr와 실제 calibration 값에서 이 명단, row/column, context별 값,1572 fit mask를 확정하고 SHA로 고정한다.

각 문맥에서391 MN 총 전류와69 source의 `masked[row]`를 실제 RK4 단계마다 기록한다. 후자는 그 단계의 발화율×출력 mask이며 CSR 계산이 실제 사용한 값이다. 공통 후보 전류를 조건부로 계산할 때는 기존 전류에 선택 edge의 가중치 차이×이 값을 더할 수 있다. 이는 각 기존 문맥의 고정된 presynaptic 상태 아래 비교이며, 바뀐 전체 재귀·감각·신체를 실행한 결과로 부르지 않는다.

E125 소스/원본, M2 소스263개, E110/E113/E121 및 기존 실패를 수정하지 않는다. 새 `StageInputNetwork`는 별도 연구 모듈에서 기존 Core와 E125의 검증된 관측 lifecycle을 재사용한다. E125의20000단계 상한은 변경하지 않는다. 새 모듈의 최대160000단계와 관측 버퍼512MiB 상한을 생성 전에 검사한다. 첫 arm만 허용하고 생성 stream을 소유한다. 관측 시작 후 restore를 거부하며 종료tick을 고정한다.

native CSR의 기존 drive add 직후 lane0에서 읽기만 추가한다. MN current와 source masked rate는 별도 row→column map·열별 int64 counter·float32 버퍼를 사용한다. 겹치는 명단도 허용하되 각각 독립적으로 한 번 기록한다. 원래 CSR 합산/순서/float32 연산과 RK4 stage/finish/신경 identity는 유지한다. 추가 CSR, tick별 CPU 동기화, 주기·신체 보정은 없다. warmup/capture는 GPU armed=0이다. 오버플로는 감싸 쓰지 않고 완료/요청 수와 오류를 보존한다.

## 단일 원본 수집 프로토콜

부모 최종 소스 리뷰 뒤 E110→E113→E121 순서로 문맥별 새 프로세스 하나씩만 실행한다. 각 문맥의 원래 `start`로 전체158706 회로와 같은 신체를 복원하고 같은 가중치/생리의 관측 네트워크로 교체한다. 기존 decoder·sense/input·world·초기 자세를 유지한다.

각 문맥은2000 control/10초,40000 neural ticks,100000 physics ticks,160000 RK4 stages다. 원래 neural/physics timestep은 .00025/.0001초다. 원래 start와 observed start 전체를 비교한다. 1.1초와10초 전체 checkpoint,2001 control/body행,2000 motor rate/activation을 원래 기록과 비교한다. E110 처음19200 current단계는 E125 원본과도 정확히 일치해야 한다.

source masked rate의 control 시작 단계는 직전 끝점의 실제 source rate×mask와 bit exact여야 한다. 이를 위해 초기 및 각 control 끝점69 source의 masked rate를 별도 저장한다. 기존 실행에는 control 중 mask 변경이 없음을 소스와 실제 상태로 확인한다. 내부 RK4 중간 단계의 기록은 독립 소형 float32 회로 검증으로 확인한다.

raw current `[160000,391]`과 source masked rate `[160000,69]`는 별도 `.npy`로 무압축 저장한다. 각 CUDA 배열의 host 복사와 파일 저장을 독립적으로 시도해 한 실패가 다른 원자료를 지우지 않게 한다. metadata는 IDs/counter/완료 범위/시작·종료tick/overflow/생성 소스 SHA를 포함한다. initial/full final neural 상태, 실제 생리·gain·pool/moment/rest/LTM, endpoint r/z/activation/q/tendon/source masked rate, 전체 trace와 checkpoint, clock/setup/오류/스레드·GPU·프로세스 wall·source snapshot을 새 디렉터리에 보존한다. 부분 실행도 요청/완료/미확정 CUDA 비용을 구분한다. raw 저장 후 판정한다.

두 관측 배열은 문맥당294400000 bytes다. 각 프로세스는 하나의 문맥만 소유하며 종료 뒤 다음 문맥을 시작한다. 메모리 부족/오류/투명성 실패면 남은 문맥은 실행하지 않고 원인을 판정한다. 자동 재시도는 없다.

## 판정 및 부모 독립 재구성

가설은 확장 관측도 기존 전체 dynamics를 보존하고 모든 문맥의10초 응답을 기존 전류 보간보다 정확하게 재구성한다는 것이다. 경쟁 설명은 장시간 수치/기록 순서·소스 mapping 오류다. 각 문맥의 완전한 상태/trace/끝점 비교와 stage cardinality/finite/dtype/ID/first-stage source 검증을 모두 통과해야 해당 기록이 유효하다. 실패한 문맥을 유효한 문맥으로 대체하지 않는다.

세 수집이 모두 유효할 때만 부모가 저장된 단계 current를 이용해 gain0의391 MN을 각 문맥10초씩 독립 NumPy float64 RK4로 한 번 재구성한다. 총3 conditional context 응답/30 conditional초/120000 context ticks/6000 muscle EMA 끝점이다. 실제 BANC나 신체는 다시 진행하지 않는다. 원래 E124 gain0 PWL 결과를 기존 파일에서 읽어 비교하며 재실행하지 않는다. 모든 문맥에서 rate RMS/max .5/5Hz, activation .25/2.5Hz, q .001/.01rad, LTM .0005/.005를 그대로 적용한다. 기존 초기99 목표 대비 오차와 실패도 별도 유지한다. 현재 gain0 재현이 adaptive gain/gradient 적합 성공을 뜻하지 않는다.

## 분업·자원·다음 결정

부모가 설계와 실제 edge/source 매핑·원본 provenance를 준비한다. Sol/gpt-5.6-sol/xhigh가 새 `tools/banc_m3_stage_dataset.py`, `tools/verify_c_banc_m3_stage_dataset.py`, `tests/test_c_banc_stage_dataset.py`를 구현한다. CUDA capture와 두 종류 stage 관측/stream/보존을 함께 다루므로 xhigh를 선택한다. 부모 첫 리뷰 → 같은 Sol 첫 수정 → 부모 재리뷰 → 잔여 직접 수정 순서다. Sol은 소형 synthetic만 실행하며 원본 graph/body 수집은 부모가 소유한다.

각 원본 프로세스 내부150초(마지막20초 보존), 외부180초, supervisor185초. 이전 보수적 누적974.220554초를 보존하고 이전975초+3×185초=1530초로 기록한다. E125의1076초 계약은 변경하지 않는다. 단일GPU,CPU/BLAS/Torch thread1. 총 actual main neural/physics 각30초,모델 생성5개×3문맥의 warmup/capture 총900 native ticks를 별도 계수한다. 실제 실패/완료 수를 보고하고 미확정을0으로 바꾸지 않는다. 원본과 별개인 부모 재구성 외부30초, 준비/검토/소형시험 비용도 각각 기록한다. 전체 goal은 무토큰예산이며 새로운 구매/서비스/UI 조작은 없다.

입력과 재사용 source·본 문서·부모 검토·버전을 수집 전에 SHA로 고정한다. 수집은 parameter 학습이 아니며 새 운동 효과를 주장하지 않는다. 이 자료가 유효하면 다음으로 dense CUDA adaptive response와 전체391 감도를 검증하고, 기존 초기99/형태 기준 아래 공동 가중치·근육 적응 제약적합을 설계한다. 실제 전체 BANC와 신체의 F3–F8 검증을 마지막에 반드시 수행한다. 전체 F1–F8은 active/unmet이다.
