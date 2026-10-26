# E125: native CUDA 전류 관측과 정류 경계 감도 분리

2026-09-21 KST · 부모 설계 · 새 실행 전 고정 계약

M3 finite-goal의 F1–F8과 기존 실패는 유지한다. E124는 유효한 조건부 진단이지만 E110 q 최대 재구성 오차 .018674rad와 gain1 감도 검사를 통과하지 못했다. 두 오차를 구분해야 근육 적응의 제약 적합을 진행할 수 있다. 본 단계에는 적합, 새 보행 후보, F6 초기자세 노출이 없다.

## A. 신경 계산을 바꾸지 않는 관측

가설 A는 5ms 간격 전류의 선형 보간이 native .25ms RK4 단계 전류를 놓쳐 E110의 일시적인 명령 오차를 만든다는 것이다. 경쟁 설명은 조건부 방정식·근육 복호화 또는 native 수치 연산의 불일치다. 이미 관측한 E110 시작 상태에서 동일 회로·감각·신체로 1.2초만 재현하고 실제 각 RK4 단계의 총 전류를 저장한다. 1.075초의 기존 최대 오차를 포함하며 이후 구간이나 초기자세를 추가하지 않는다.

새 연구용 `ObservedCoreAdaptationNetwork`는 기존 CoreAdaptationNetwork를 상속한다. 기존 파일은 수정하지 않는다. 기존 CUDA CSR kernel의 `value=add_exact(drive[row],value)` 직후, threshold와 adaptation을 빼기 전에 선택한391 MN의 float32 값을 별도 버퍼에 복사한다. CSR 합산·순서·float32 연산·stage/finish·신경 identity는 그대로다. 추가 CSR 연산과 tick별 CPU 복사/동기화는 없다.

GPU에 native row→관측 열 매핑, 열별 int64 카운터, armed 상태, overflow 상태, `[capacity_stages,391]` float32 버퍼를 둔다. 각 행의 lane0만 자기 열 카운터와 버퍼를 쓰므로 공유 카운터 경쟁을 피한다. armed=0으로 생성·warmup·CUDA graph capture를 마친다. graph 안에서도 GPU armed 값을 읽는다. main 실행 직전 상태를 동기화한 뒤 카운터·버퍼를 지우고 arm한다. 정상1.2초는4800 native ticks,19200 stages, 모든 열 카운터19200이다. 용량 초과는 데이터 덮어쓰기 없이 명시적으로 검출하고, 실제 요청 카운터를 보존한다. 값이 없는 버퍼 부분은 NaN이며 완료 영역과 구분한다. 실험 후 host로 복사한다.

관측 상태는 신경 상태나 생리 파라미터가 아니다. 기존 snapshot/restore의 모델 identity와 상태 계약은 그대로이며 별도 observation raw에 시작 native tick·카운터·완료 영역·source hash를 저장한다. 체크포인트를 관측 네트워크에 복원해도 정확한 미래를 유지해야 한다. 읽기·arm/reset/disable은 동기화와 명확한 단일 stream 소유권을 지킨다. CPU 또는 다른 neural 구현의 조용한 fallback은 금지한다.

새 wrapper는 E110 `start`를 기존 IntrinsicRecruitmentTrial로 복원한 뒤 동일 source_matrix·tau·계수·cap·적응·core IDs의 관측 네트워크로 교체하고 whole snapshot 동일성을 확인한다. CUDA 모델 생성별 warmup/capture(각3×capture_steps)는 본 실행과 분리해 계수한다. 실제 runner는 원래240 control을 한 번 수행한다. 첫240 command/activation/관측 rate 및 가능한 body trace를 E110 원자료와 비교하고,1.1초 whole snapshot을 기존 E110 `main-1p1`과 비교한다. 차이가 있으면 관측 도구의 유효성은 실패이며 전류 가설은 판정하지 않는다. 새 clone 미래나 추가 body trial은 실행하지 않는다.

원자료를 먼저 저장한다. 부모 독립 계산은 저장된 단계 전류를 float64로 확장해 gain0의391 MN을4800 RK4 ticks와240 muscle EMA 끝점만 적분한다. 이를 기존 E124 gain0의1.2초 prefix 및 원래 E110과 비교한다. 전체 .005..1.2초에서 기존 E124 RMS/max 한계(rate .5/5Hz, activation .25/2.5Hz, q .001/.01rad, LTM .0005/.005)를 모두 충족하고 q 최대 오차가 기존 prefix보다90% 이상 줄면 A를 지지한다. 관측 투명성이 유효한데 개선되지 않으면 원인 분리는 불충분하다. E124 자체의 실패 판정은 변경하지 않는다. 이 관측 구현의 채택은 수치 투명성과 기록의 정확성이 확인된 연구 경로에 한하며 제품 보행 채택과 구분한다.

## B. 작은 고정 집합의 감도 진단

부모는 Sol 구현과 독립적으로 기존 E124 입력만 사용해 다음11 MN을 분석한다: E113 열125, E110 열100/302(근육61), E110 열12/25/57/102/126/136/169/171(LF LTM). 선택은 기존 최대 오차 위치에서 고정했다. 전체391 MN 또는 원래4 batch를 다시 실행하지 않는다.

독립 NumPy float64 RK4로0..10초의 동일 PWL 입력을 gain1 및1±h, h={1e-4,1e-5,1e-6,1e-7}에서 한 vector batch로 계산한다. 총99 scalar MN 응답,990 scalar MN 모형초,3,960,000 scalar MN RK4 ticks이며 actual BANC/physics/fit은0이다. baseline과±1e-4 응답/감도를 원래 E124 raw에서 선택한 값과 비교한다(절대오차1e-8 이하). 각 단계의 양수 정류 분기, baseline 경계 여유, 끝점 r/z/v/w를 원자료로 남긴다. 근육61 및 LF LTM은 원래 pool 가중치와 EMA로 합산한다.

가설 B는 ±1e-4 차분의 분기 교차가 국소 tangent와의 차이를 만든다는 것이다. h 감소에 따라 분기 차이와 오차를 모두 보고한다. h=1e-7에서 고정 E124 감도 한계 `2e-5+2e-4*max(abs(FD))`를 만족하고 h=1e-4 대비 최대 오차가10배 이상 줄면 해당 선택 집합에서 B를 지지한다. 그렇지 않으면 불충분으로 유지한다. 모든 field와 모든 h를 보고하며 유리한 h만 선택하거나 정류 함수를 평활화하지 않는다. 선택 집합의 결과로 전체391 감도가 검증되었다고 주장하지 않는다.

## 분업·자료·자원

부모 설계 → Sol/gpt-5.6-sol/xhigh 구현 → 부모 첫 리뷰 → 같은 Sol 첫 수정 → 부모 재리뷰 → 남은 문제는 부모가 직접 수정한다. 새 agent 제한이 있어 기존 `/root/m3_sol_e123_motor_adaptation`를 새로운 E125 구현에 재사용한다. Sol은 새 observer/runner/test 세 파일만 소유하고 실제 E110 원본 실행은 하지 않는다. 부모는 fixture/provenance와 B의 독립 수치 분석을 수행한다. 기존 E123/E124 소스·모든 과거 원자료·M2·UI는 보존한다.

A 원본은 부모 최종 소스 리뷰 뒤 단일 프로세스로만 실행한다. GPU1개, CPU/BLAS/Torch thread1. 내부90초(마지막15초 보존), 외부120초, supervisor 포함125초, 자동 재시도0. 기존 누적950.5949305초를 보존하며 보수적 이전951초 +125초 =1076초로 기록한다(E124의1065초 계약을 재작성하지 않는다). 실제 main neural/physics 각1.2초, native ticks4800/physics ticks12000/control240. 모델 생성별 setup은 따로 계수한다. A 부모 재구성은 별도 단일 CPU 분석,4800 context RK4 ticks/1.2 조건부 초/240 decode, 외부30초 한도다.

B는 별도 단일 CPU 프로세스, 내부45초(마지막5초 보존)/외부60초, 자동 재시도0다. 소형 테스트·fixture 준비·컴파일 및 부모 oracle 비용은 실제 main 실행과 분리해 모두 보고한다. 새 계산 전 입력·소스·본 문서를 SHA256으로 고정하고 신규 디렉터리에 raw,부분 결과,오류,ledger,wall,source snapshot을 보존한다. 실패 후 계약 변경은 원본을 유지한 새 설계로만 한다.

두 진단의 결과에 따라 다음 적합의 전류 표현과 미분 검증 방법을 설계한다. gain1은 선택된 보행 후보가 아니며 초기99 실패를 유지한다. 같은 최종 후보의 F1–F8을 전부 충족하기 전까지 전체 목표는 active/unmet이다.
