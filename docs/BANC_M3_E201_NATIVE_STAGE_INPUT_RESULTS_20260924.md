# E201: 현재 E150의 전체 신경 단계 입력 확보

기존 E150와 동일한 전체 BANC158706/CUDA/신체10초 실행에서391 MN의160000 RK4 단계 입력을 확보했다. 전체 시작·1.1초·10초 snapshot,2001개 신체 trace, MN 발화·적응·근육 활성·모집 bank가 원본과 정확히 같다. 비교에서 측정 wall_s만 제외했다. 관측기를 추가해도 원래 모델의 동작이 변하지 않았음을 확인했다. **새 학습이나 보행 개선 결과는 아니다.**

## 구현과 검증

`tools/banc_m3_timescale_stage_input.py`는 기존 MotorTimescaleNetwork의 CSR에 수동 기록을 추가한다. drive+recurrent current를 threshold/adaptation 차감 전에 저장하고 원래 stage/finish와 identity를 유지한다. GPU 기록은 한 번 arm하며 stream·복원·용량·overflow를 검사한다. 실행기 `tools/observe_c_banc_m3_timescale_stage_input.py`는 원본과 비교하기 전에 부분 raw를 저장한다.

부모 설계→actual gpt-6-sol/high 구현→부모 FIRST→같은Sol 첫 수정→부모 SECOND/잔여 수정 순서를 적용했다. FIRST에서2000 signal행/2001 native행 혼동, 중첩모델 생성·진행량 집계 오류, 결과 저장 실패를 성공으로 끝낼 수 있는 경로를 수정했다. 부모 SECOND에서는 실제 재사용 소스32개를 추가로 hash 고정·검사하도록 보강했다. 원래 동결27개 재사용 소스와250입력·M2 해시는 그대로다.

10개 합성 시험과 Ruff, 부모 독립 전류 계산을 통과했다. 독립 시험의 첫 두 시도는 부모 fixture의 tau 해상도와 이진 output-mask 제약 오류였다. 두 실패·소스·비용을 보존하고 기존 제약에 맞춘 생성자료만 수정했다. 실제 BANC 원본 재실행은0이다. 최종 독립 생성 시험은 영연결망 및 이진차단 연결망에서 순서를 바꾼 관측 ID·음수 drive·전체상태 exact를 확인했다.

## 원본 결과

- PID2388, exit0, 원본1회, 자동retry0, 잔류 프로세스0.
- 실제 main:2000 control/40000 neural tick/100000 physics step,10초.
- 복원 경로의6개 모델과 관측망1개, 총7개 constructor. warmup280/capture140 tick은 main과 분리했다.
- float32 current160000×391=62560000개, payload250240000bytes, counter각160000/overflow0/armed0.
- 실제 초기 상태·cell계수·gain/T·근육 소유권을 함께 보존했다. 경계 입력 보간이나 다른 E126문맥으로 대체하지 않았다.
- 부모의 별도 원자료 검산은 전체 snapshot/trace/endpoints/파라미터에서2175680개 leaf 비교와 행렬 유한값을 확인했다. 이 수를 소프트웨어 시험 수로 세지 않는다.

원본 감독72.28309830000217초, 보수 누적 **5105.233614699708/5444초**, 잔여 **338.7663853002923초**다. 준비0.4162309초, 부모 원본 검산3.3298356초는 별도다. Sol 합성3모델/12warmup/6capture/10main tick, 부모 같은 시험3모델/12/6/10, 부모 독립 시험의 실패 및 성공 합계8완료모델/32/16/20도 별도다. 부모 첫 실패는 constructor 완료0이었다. 미계측 개발·리뷰 비용을0으로 취급하지 않는다.

자료의 상태는 `valid_exact_native_stage_dataset`이다. F6 미노출·후보 미채택·M2/UI와최선E150보존, 원래 F1–F8과 자연 보행은 여전히 미완료다. 다음은 이 실제 단계 입력으로 국소 신경 반응을 계산하는 학습용 모델의 원본 재현·기울기·계산비용 검증이다. 고정 입력 적합이 실제 전체 신경망 되먹임과 신체 보행을 대체하지 않는다.

근거: `verification/banc-natural-gait-m3-20260919/e201-native-stage-input/`, `parent-e201-output-review.json`, `parent-e201-supervisor.json`. result SHA256 `853c1b012f335dc957baa064acca5fa418d1d7519ec1216e85c87c40fbce7f13`, current SHA256 `d91f17ca144a5ca80f3f84d5e19185317728c461cfbbc3b52d86112e9838f304`.
