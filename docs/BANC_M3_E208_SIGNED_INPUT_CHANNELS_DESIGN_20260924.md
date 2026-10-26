# E208: 기존 BANC 입력을 부호별로 분리하는 수동 CUDA 계측

F1–F8 자연 보행 finite-goal은 active·미완료다. E203–E205의 같은 국소 반응식 학습, E206의 명시적 제약 접선, E207의 작은 문턱 초기화가 후보 기준에 미달했다. E207 부모 독립 감사에서 미달117개는 rate/h 동시미달111, rate만미달1, h만미달5였다. 정확한 control별80 RK stage를 사용하면25개는 `c*(max I−threshold)+.5<=0`여서 기록된 후기 입력과 비음수 적응 아래 현재 보정 범위로 순간 발화를 만들 수 없다. 이 판단은 원본 입력이 고정된 가족에 한정된다. 초기 Sol 감사의20-stage 색인 오류와 철회된 수치는 부모 감사에 보존했다.

이 근거는 같은 작은 bias의 반복보다 신경이 입력을 통합하는 방식을 검토할 이유다. 현재는 모든 signed synaptic input을 한 스칼라로 합산한다. 다음 구조적 대안은 같은 해부학적 입력의 흥분/억제 채널과 인과적 내부 상태를 분리하는 모델이다. 아직 그 모델을 생물학적으로 검증된 것으로 보거나 바로 채택하지 않는다. 기존 단일 연결 gain, 같은 합산 전류의7기저 계수 학습과 구분한다. E208은 이 대안에 필요한 입력을 **기존 E150 전체 BANC·신체 경로를 그대로 재현하면서** 수집한다.

## 질문과 후속 결정

H208은 현재 가족에서 닫혀 있는25MN 중 적어도13개가 후기 구간에 양의 recurrent 입력을 받으면서 더 큰 음의 입력으로 상쇄된다는 것이다. 경쟁 설명은 해당 MN에 이용할 양의 입력 자체가 없거나, 입력 통합보다 출력 전달/상태가 주요 제한이라는 것이다. 각 MN의 positive recurrent RMS가1e-3 이상이고 negative recurrent magnitude RMS가 positive RMS보다 크면 상쇄 입력 있음으로 센다. 1e-3은 부동소수 잔류를 배제하는 공학적 관측 문턱이지 생물학적 발화 기준이 아니다. 같은25개·동일1824 control에서 각 채널 RMS/표준편차/범위, drive, net 전류도 함께 공개한다.

13개 이상이면 분리 채널/내부 상태의 조건부 학습을 다음 사전 설계로 검토한다. 미달하면 넓은 양의 입력 상쇄 설명을 지지하지 않고, 기존 net 입력 이력/출력 전달 대안을 검토한다. 어느 결과도 자연 보행 향상이나 변경 모델의 도달 가능성 증명이 아니다. 이번에 신경 동역학 변경·fit·새 후보·F6 노출·UI 변경은 없다. 실제 후속 모델 식/학습/채택은 별도 사전 설계 및 검증을 거친다.

## 계측 불변 조건

E150 start/main-1p1/final 및2001개 전체 제어 기록,391MN·96근육과158706노드/11584852선언연결을 유지한다. 전체 rate/adaptation/drive/output mask와 신경 identity, 감각·신체 및 decoder 설정은 동일하다. E207 weights/seed는 사용하지 않는다. 현재 source의 음수 부호를 억제성 모델 채널이라고 부르며 실측 receptor 해석으로 확대하지 않는다. zero/masked edge 정책도 그대로다.

실제 packed CUDA CSR의 원래 `value += lookup[...] * masked[...]` 순서, warp reduction, drive 더하기, threshold/adaptation 빼기, stage/finish 식과 capture를 변경하지 않는다. 관측 대상391MN에서만 별도의 두 합계를 같은32-lane 순서로 계산한다. `weight>0`의 product는 positive 채널, `weight<0`의 minus-product는 negative magnitude 채널이다. 원래 net reduction은 그대로 두며 새 합계를 원래 계산에 되먹이지 않는다. 세 번째 추가 채널은 같은 실제 RK 단계의 drive다. 원래 total-current와 함께 **4개의160000×391 float32 배열**을 기록한다. 각 열당 공유된 한 counter로 같은 position에4채널을 저장하고, overflow 시 out-of-bounds 쓰기를 하지 않는다. 관측 버퍼는 신경 identity/snapshot 밖에 둔다.

E201 observer의 one-stream/one-arm/capture 후 arm/restore 금지/disable 후 readback을 유지한다. 모든 배열을 constructor/capture 전에 할당하며 NaN으로 초기화한다. 원래 current buffer와 추가3채널/map/counter/flag를 합쳐1GiB 이내로 제한한다. 초기화/capture의 기록은 armed=0이어서 원본 자료에 섞이지 않는다. 1 control=20 neural ticks×4RK=80 stage, 총2000control/40000tick/160000stage다. 늦은 target 행t에 대응하는 단계는 정확히 `[(t−1)*80:t*80]`다.

## 실제 재현과 원자료 검증

E201의 고정 소스는 편집하거나 gate를 monkeypatch하지 않는다. read-only helpers `endpoint`, `save_endpoints`, `save_motor_parameters`, `compare_signal_adaptation`, `validate_raw`, `output_manifest`, `AdvanceLedger`, exact/state loader 등을 재사용할 수 있다. 새 runner는 자체 freeze/source review를 검사하고 기존 E150를 복원해 새 observer로만 치환한다. 기존7회 모델 구성/14 interval warmup/7 interval capture와 본 주행40000tick/100000물리step을 그대로 계수하며 setup 실패의 미확정 완료를0으로 쓰지 않는다.

원본 process 한 번에서10모델초를 실행한다. start/1.1s/final 전체 snapshot과2001 trace/native rate/activation/bank/adaptation은 E150와 exact여야 한다. 새 total-current는 E201 저장 배열과 dtype/shape/bytes exact다. 네 관측 배열을 검증 전에 각각 독립 저장하여 한 hostcopy 실패가 다른 채널/카운터/부분 raw를 지우지 않게 한다. finite/nonnegative(channel E/I)/shape/counter160000/overflow0/armed0을 검사한다. 전체391MN의 실제 초기 생리/소유권도 기존 E201 parameter 배열과 exact 대조한다.

새 분리합의 double 검산 `drive+positive−negative`와 원래 total은 float32 합산 순서가 다를 수 있으므로 bit exact를 요구하지 않는다. 각 표본에서 `abs(residual)<=1e-5+64*eps32*(abs(drive)+positive+negative)`를 적용하고, 실제 최대 절대/정규화 오차를 공개한다. 이 허용은 현재 주행 exact 조건에는 적용하지 않는다. 원래 CSR net 자체와 기존 궤적은 계속 exact다. 또한 stage0의 positive/negative는 원래 snapshot의 masked rate와 실제 관측 MN CSR row를 float64로 합산한 부모 독립 검산과 같은 오차 범위로 비교할 수 있게 native MN CSR row 값/indices/indptr를 별도 저장한다. graph/실제 가중치를 바꾸지 않는다.

판정은 실행 유효성, H208 입력 상쇄 관측, 새로운 모델 미시험, 최종 F1–F8 미완료를 분리한다. 실패에도 원 오류·가능한4채널·snapshot/trace/카운터/비용을 보존한다. 자동 원본 재시도는 없다.

## 소유권·검사·자원

현재 세션이 설계하고 실제 gpt-6-sol/high가 `tools/banc_m3_signed_input_channels.py`, `tools/observe_c_banc_m3_signed_input_channels.py`, `tests/test_c_banc_signed_input_channels.py`만 구현한다. 새 신경 동역학/미분은 없고 검증된 observer를 확장하는 CUDA 계측/실패 보존 통합이므로 high를 선택한다. 부모 FIRST→필요시 같은Sol 첫수정→부모 SECOND/잔여수정→원본 한 번 순서다.

필수 생성 검사는 원래 stage/finish·net reduction 보존, partial NaN/overflow/한 채널 hostcopy 실패 시 형제 raw 보존, source gate와80-stage 색인이다. 새 계측 커널에는 **작은 실제 CUDA 대조 한 번**을 수행한다. 양/음/zero 연결과 서로 다른 RK 중간 입력을 가진 생성 그래프에서 기본/관측 모델 두 개만 만들고, 원래 snapshot exact와 독립 수동 channel 합을 확인한다. 생성 main은 base3tick/observer3tick 및 disable 후2tick 이내, warmup8tick/capture4tick을 별도로 기록한다. 기존 E201 실제 CUDA 시험 전체를 다시 돌리지 않는다. 모의 검사는 실제 GPU 계산으로 세지 않는다.

원본 CPU1/GPU1/process1,10모델초,fit0/retry0다. 내부135/model115/외부150/감독155벽초, 관측GPU1GiB/출력2GiB/owned process tree RSS12GiB/최소free disk12GiB다. 추가 GPU실험은 동시에 실행하지 않는다. 기존 원본 누적5426.123305699704/5564초를 보존하고,4채널 raw와 전체 BANC 관측을 위해 단계 배정을60초 늘린5624초에서155초를 사전 예약한다. 이는 사용자 전체 예산이 아니라 단계 회계이며 모든 실제비용은 계속 누적한다. 준비·생성·부모 검산은 별도다. 무측정은0이 아니다. M2·최선E150·모든 실패/F6/UI 보존은 변하지 않는다.
