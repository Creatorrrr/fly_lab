# E221: 실제 packed BANC 재귀 경로의 새 반응 구현과 작은 CUDA 검산

E218/E220 조건부 후보를 실제 신경 재귀 경로에 넣는 기반을 구현한다. 이번 단계는 신경 클래스와 생성된 작은 재귀 그래프의 CUDA 검증까지다. 원래 BANC 전체 그래프·기록 입력·몸체·F6는 실행하지 않는다. 원래 후보의 성능 판정이나 학습을 다시 하지 않는다.

부모가 설계하고 기존 실제 gpt-6-sol/high가 구현한다. 네 RK stage·혼합 정밀도·CUDA graph·checkpoint가 얽힌 작업이므로 high를 사용한다. 부모 FIRST→필요시 같은 Sol 첫 수정→부모 SECOND→남은 부모 수정 절차다. 소유 파일은 tools/banc_m3_native_grouped_response.py와 tests/test_c_banc_native_grouped_response.py 두 개다. 기존 성공/실패 소스, 앱, UI를 변경하지 않는다.

## 방정식과 정밀도

MotorTimescaleNetwork를 기반으로 기존 packed CSR 총전류 및 나머지 뉴런 방정식을 그대로 둔다. 각 stage에서 같은 CSR의 부호별 곱 합 E/I를 별도 float32 레지스터에 계산하며 drive도 그 stage의 float32 값이다. 총전류의 기존 덧셈 순서를 바꾸지 않는다. free MN만 입력 E/I/drive를 float64로 변환한다.

반응은 E217/E218과 동일하다. x=[log1p(E)/sE,log1p(I)/sI,asinh(d)/sD], fast'=(x-fast)/.016, slow'=(x-slow)*actual_inverse_adaptation_tau. 9차원 자기 x/fast/slow와 고정 자기64개 center/sigma²로 안정화 Gaussian softmax를 구한다. gate=-expm1(-8*(xE+abs(xD)+max(fastE,0)+abs(fastD))). phi=cap*gate*sum(beta_owner*p), r'=(phi-r)/tau. beta는 기존 근육 소유권으로 공유하며 실행 중 적합·center 이동·clip·time/phase/teacher 입력은 없다.

free MN의 r/fast/slow와 모든 반응 RK 연산은 float64다. dt=.00025를 float64로 사용한다. tau/cap과 slow reciprocal는 실제 원래 float32 파라미터 값을 float64로 올린 값과 정확히 일치해야 한다. 각 stage의 r을 float32로 반올림해 기존 전체 rate 및 masked rate에 기록한 뒤 다음 CSR을 계산한다. 기존 mask 곱의 pointwise float32 의미를 유지한다. 원래 adaptation z는 free MN에서도 기존 float32 식으로 계속 적분하지만 새 반응식은 z를 입력으로 쓰지 않는다. fixed48과 나머지 뉴런은 실제 변경된 CSR 입력 아래 원래 방정식으로 움직이며 저장 rate를 대입하지 않는다.

기존 public motor readout/snapshot rate는 float32를 유지하고 free r64를 별도 신경 상태로 보존한다. 후속 근육 전달에서는 rate를 float64로 올리고 원래 pool·moment·rest·10ms activation·tendon을 계산한다. E220의 모든-readout-float32 실패를 숨기지 않는다. 이 단계에서 adapter/body 통합은 하지 않는다.

## 연결과 상태

기존 네 stage 사이에 추가 free-response kernel을 넣어 work_rate/work_masked를 갱신하거나 동등한 정확한 방식을 사용한다. stage0은 base 상태, stage1/2/3은 각 RK 중간 상태다. finish에서 free r64/기억을 RK 가중합으로 갱신하고 최종 full rate의 free 위치도 반올림한다. stage 인덱스는 적분 순서일 뿐 반응의 시간 입력이 아니다. CUDA graph가 정확한 순서를 포착해야 하고 persistent 포인터는 교체하지 않는다.

새 상태와 scratch/파라미터는 graph capture 전에 할당한다. 부모 생성자의 warmup/capture가 끝나면 free r64/fast/slow를 과학적 초기 0으로 되돌리고 full rate/adaptation/drive와 일치함을 확인한다. reset도 원래 상태와 새 상태를 함께 초기화한다. 그래프 재생 안에 매회 reset을 삽입하면 안 된다. 원래 네트워크 상위 identity와 precision 규약, ordered motor/free IDs, membership, scale/center/sigma/beta/slow/tau/cap의 바이트를 새 identity에 포함한다.

snapshot은 원래 상태와 새 파라미터·r64/fast/slow를 모두 담는다. restore는 누락·wrong dtype/shape·ID 순서·비유한값·음의 rate·범위 밖 beta·identity·float32(rate64)와 full rate의 불일치를 상태 변경 전에 거부한다. 부호 있는 drive history는 허용하고 임의 clip하지 않는다. 원래 restore의 검사를 재사용하되 새 상태 검증을 먼저 한다. 다른 포인터나 metadata를 가진 checkpoint를 조용히 받아들이지 않는다. 생성 작은 모델에서도 같은 클래스·packed CUDA 경로를 사용하도록 motor 수와 group 수는 검증 가능한 일반 shape로 지원하되 원래 후보 적용에서는391/343/48/96/64를 별도 확인한다.

## 검증과 한도

Sol은 작은 CPU 생성 파라미터의 검증·식·상태 검사만 실행한다. 실제 CUDA 실행은 부모 소스 검토 뒤 별도 감독 프로세스 하나로 한다. 8개 뉴런·4개 motor(3 free/1 fixed),2개 근육 그룹·각64 center의 고정 생성 fixture를 사용한다. 양/음 부호 재귀 edge, 같은 근육2개 MN, fixed motor와 비motor feedback을 포함한다. 원래 모델·원본 파라미터 파일은 읽지 않는다.

부모가 독립 CPU stage oracle로 각 궤적 최대32 neural tick, 전체 생성 검증의 과학적 advance 합계 최대128 tick의 drive/출력mask 변화에 따른 whole rate/z 및 새 r64/fast/slow를 비교한다. 생성자 warmup/capture는 별도 기록하며 과학적 궤적으로 세지 않는다. full float32 rate/z atol5e-5/rtol2e-5, 새 float64 상태 atol2e-6/rtol2e-5를 사전 한계로 둔다. near-zero·무입력·부호입력, free-mask 차단의 하류 변화, full graph replay 대 직접 stage 실행, 초기화, 디스크 저장→restore→동일 미래 입력의 byte-exact 상태, 잘못된 checkpoint 거부 후 원상태를 확인한다. 부모 oracle와 대조하지 않은 테스트 PASS는 actual CUDA 정확성으로 부르지 않는다. 자유 반응을 끄거나 기록 rate를 고정하는 우회는 없다.

CPU1/GPU1, 생성 감독 프로세스1/재시도0, 내부45/외부50/감독60초, RSS4GiB/추가GPU2GiB/출력64MiB. 원본 프로세스0/solver0/physics0이며 원본 잔여208.86004140031218/6004초를 소모하지 않는 별도 측정 생성 검증이다. 이 생성 비용도 따로 명시해 총 비용0으로 숨기지 않는다. 실패하면 raw·오류·자원 기록을 보존하고 새 실행은 별도 사전 계약으로만 한다. 실제 BANC·body 후속은 새 freeze와 원본 잔여 예산으로 설계한다.

현재 후보 채택0, 최선E150·M2·UI보존, F6미노출, F1–F8 active·미완료, 생물학적 기전 미검증을 유지한다.
