# E129: 동결한 LF 한 평가의 계산 비용 계측

2026-09-21 · 부모 사전 설계 · M3/F1–F8 미완료

E128은 유효한 원본 적합이었지만169.505초/93 unique에서 다리별 wall cap에 도달했고, feasible 기록은0이었다. LF는27.6119616초/13 unique였다. 평가 내부와 SLSQP의 시간 분해가 원본에 남아 있지 않으므로 CUDA가 병목이라고 가정하지 않는다. E128을 다시 적합하거나 실패를 대체하지 않고, 이미 공개된 LF 첫 평가 하나의 비용을 계측한다.

**Experiment ID** E129 / **Hypothesis ID** H129 / **Milestone ID** M3.

가설은 고정 응답·미분·기하/loss·저장을 포함한 평가 파이프라인이 LF의 unique당 평균 비용 중 큰 부분이라는 것이다. 경쟁 설명은 SLSQP의4761 제약 처리, 그 밖의 CPU 작업, 초기화 및 점마다 다른 비용이 더 크다는 것이다. 한 점으로 전체 최적화 비용을 확정하지 않으며 관측하지 않은 잔여 시간을 전부 SLSQP 시간으로 부르지 않는다.

## 동결 조건과 실행

E128 입력/freeze/최종 검토 소스, 원본 출력112개 hash를 보존한다. 새 입력은 만들지 않는다. 같은391/96/69/2444/1572 배열에서 원래 LF69 MN/17 muscle/273 edge subset을 명시적 ID로 사용한다. 원본 `leg-0-evaluation-001-raw.npz`의 delta/gain가 모두0인지 확인하고 그대로 사용한다. 같은3 contexts/native160000 stages/2001 endpoints, 초기99/mask/target/scales 및 TorchFootGeometry hash를 유지한다.

새 `tools/profile_c_banc_m3_joint_fit_cost.py`는 동결된 E128 engine/fitter를 재사용한다. 새 kernel·ODE·미분식·loss·optimizer·가중치·efficacy·gain 변경은 없다. 원본 프로세스 한 번, 원본 LF 평가 한 번만 허용한다. 이 평가에는 조건부3×10초/120000 leg-context ticks,69×120000 scalar MN ticks,273×4×120000 edge-stage ticks,1 derivative bank,기하/loss backward1회가 포함된다. 실제158706 BANC/신체/StateStore/GraphStore/F6/공동 후보/최적화는0이다.

E128의 명시적인 synthetic tiny setup 한 번(1MN/20ticks/.005조건부초/1bank)을 별도로 계상한다. 기하/신경 원본점 추가 warmup은 하지 않는다. 고정 배열·geometry 업로드와 초깃값 준비 비용을 별도로 기록한다. 원본 평가의 첫 Torch geometry 호출 비용을 숨기지 않는다.

## 계측 범위와 정합

호스트 구간은 perf_counter로 engine 생성/고정 업로드, evaluate_leg 전체, engine.solve, geometry.leg 각 문맥, torch.autograd.grad, activation cotangent 합산, constraint 조립, np.savez를 기록한다. 중첩 구간을 단순 합하여 total로 만들지 않는다. geometry/autograd의 CPU 반환 시간과 동기화된 CUDA timeline 시간을 구분한다.

E128 모듈의 kernel callable에 프로세스 내부 관찰 래퍼를 씌워 `joint_forward`, `muscle_ema`, `edge_bank`의 같은 private stream에 CuPy CUDA events를 기록한다. 계산 인자·출력과 호출 횟수를 바꾸지 않는다. setup과 원본을 다른 phase로 기록한다. 모든 래퍼는 finally에서 복원한다. 반환 전 오류·동기화 오류는 requested/returned/enqueued/unknown을 구별하고 원래 예외 및 후속 계측 오류를 보존한다.

CUDA event 시간은 해당 스트림 구간의 경과 시간이며 CUPTI 커널 전수 계측이 아니다. decoder GEMM/elementwise, 라이브러리 내부 scalar 검사 등 미계측 구간과 payload 계상 범위를 명시한다. 기존 engine의 전송/geometry/forward/bank 메타데이터도 그대로 저장한다. 원본 actor나 제품 서버의 전역 상태를 건드리지 않는다.

원본 raw를 수치 assertion 전에 저장한다. 원본 E128 LF 첫 평가와 여덟 response field 및 B_initial99/q·LTM errors/loss gradient 최대차1e-8,loss/parts/ratios 최대차1e-10,branch bits exact,delta/gain/IDs/mapping exact를 확인한다. 원래 raw와 새 raw/hash/비교 수치를 모두 남긴다. 파이프라인 비교가 실패하면 timing은 진단만으로 남기고 병목 가설은 판정하지 않는다. 일부 계측 실패를0시간으로 처리하지 않는다.

## 판단과 한도

원래 LF 평균은27.61196159996325/13초로 고정한다. 유효한 원본 replay의 evaluate_leg wall/이 평균이>=.65면 파이프라인이 상당한 비용을 차지한다는 한 점 근거다. <=.35면 그 파이프라인만으로 원래 평균을 설명하기 어렵다. 그 사이는 불확정이다. 측정된 forward+edge-bank CUDA 구간 합이 evaluate_leg wall의>=.5일 때만 신경 CUDA kernel을 우선 개선할 근거로 삼는다. 한 점의 원래 평균 비교는 탐색적이며 속도 향상, 전체 optimizer 병목 또는 보행 성공의 증명이 아니다. 결과에 따라 후속 측정/개선안을 별도로 설계한다.

내부30초/외부40초/supervisor45초,one CPU thread,실제 RTX4080 CUDA,원본 한 프로세스/한 평가/retry0이다. 이전 보수적 공유1394.0015740999952초를1395초로 올려45초 배정,1440<=1530초다. 초기화와 실패 비용을 누락하거나 기준을 넓히지 않는다. E128의 닫힌 결과와 평가 한도는 그대로 남긴다.

부모는 설계/freeze/원본 supervision/리뷰를 소유한다. Sol은 새 profiler와 `tests/test_c_banc_joint_fit_cost_profile.py`만 구현하며, 기존 검증식/engine/fit 소스는 수정하지 않는다. 계산 변경 없는 여러 구성요소의 계측 작업이므로 **gpt-5.6-sol/high**를 사용한다. 부모 첫 리뷰→같은 Sol 첫 수정→부모 두 번째 리뷰→잔여는 부모 직접 수정 순서를 유지한다. 합성 테스트는 호출/phase 분리, 래퍼 복원, 예외·부분 계측, raw-before-assert,원본 hash/한 평가 guard를 검증한다. 원본 실행은 부모 최종 소스 리뷰가 승인된 뒤에만 한다.
