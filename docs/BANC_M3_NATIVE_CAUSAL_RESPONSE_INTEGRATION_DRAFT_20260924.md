# 인과 반응의 실제 BANC 통합 작업 메모

상태: E211 조건부 적합 결과 전의 부모 설계 메모다. 구현/원본 신경 실행/신체/F6 노출을 시작하는 freeze가 아니다. E211 결과를 확인하여 필요한 범위와 자원을 사전에 확정한다. 현재 새 반응은 저장된 입력의 조건부 계산이며 자연 보행 모델로 채택되지 않았다.

## 소스에서 확인한 연결 지점

- `tools/banc_m3_signed_input_channels.py::_signed_timescale_source`는 현재 packed CSR 합의 연산 순서를 유지하면서 native 부호별 E/I/drive를 얻는다. E208 관찰 배열은 신경 반응으로 돌아가지 않는다.
- `flylab/c/adaptive_fused.py::ExactRKMixin._step_fused`는 RK4 네 stage마다 masked 현재 rate의 CSR 전류를 다시 계산한다. 새 입력 기억도 같은 stage에서 적분해야 하며 5ms마다 끝점 입력을 고정하는 대용으로 바꾸면 안 된다.
- `tools/banc_m3_motor_timescale.py`는 원래 개별 tau/신경 identity를 보존하는 기존 교체 기반이다. `banc_walking.py`에서 5ms control당20 neural tick, tick당4 stage다.
- 새 자유343 MN만 E209의 반응식을 사용하고 fixed48 LTM MN와 나머지158315 뉴런은 기존 식을 유지한다. 여기서 fixed는 반응식 고정이지 시간별 rate 고정이 아니다. 바뀐 신경 출력의 재귀 영향으로 이들의 실제 rate가 달라질 수 있으며 기록된 E150 rate로 덮어쓰면 안 된다.
- sensory/descending/circuit cut은 drive 또는 output mask 상류에 적용된다. motor cut은 신경 계산 뒤 adapter의 activation/명령에 적용되므로, 운동 차단이 신경 기억을 초기화한다고 가정하지 않는다.

## 아직 결정·검증해야 할 계약

E209의 h와 rate 적분은 float64이고 실제 packed 네트워크의 rate/CSR 입력은 float32다. 이 정밀도 차이를 숨긴 채 조건부 기저와 full runtime이 exact라고 주장하면 안 된다. 자유 MN의 내부 rate/h 정밀도, CSR로 내보내는 변환 위치, 실제 motor readout 정밀도를 명시하고 작은 생성 recurrent network와 독립 stage 적분으로 먼저 확인한다. 모델 identity에는 선택된 정밀도와 변환 규약도 포함한다.

새 scale(391×3), centers(391×16×6), sigma2, beta, 자유/고정 MN ID·소유 근육, 원래 tau/cap, 입력 기억 h를 포함한 완전한 상태 계약이 필요하다. scale/center는 실행 중 재학습하지 않는다. 외부 시각/단계 번호/교사 명령/목표 궤적을 반응 입력으로 넣지 않는다. 음의 input-history 중간값을 임의 clip하지 않고, gate의 max(hE,0)만 E209 정의대로 사용한다.

현재 StateStore v3와 E150 adapter checkpoint는 새 기억을 저장하지 않는다. network snapshot/restore와 model hash를 확장하고 잘못된 ID 순서·누락된 상태·shape·비유한값을 상태 변경 전에 거부해야 한다. 디스크 재시작 뒤 동일한 future 입력에서 전체 신경 상태/adapter/물리 궤적을 비교한다. CUDA graph capture/warmup에서 h/r가 과학적 초기상태를 몰래 전진시키지 않는지도 검사한다.

canonical decoder는 E150 monotone decoder와 별개인 변경이다. 신경 반응 교체와 decoder 교체를 분리해 설명하고 저장된 2×2 대조를 보존한다. 현재 몸체의 해부학적 pool/moment/rest/6 LTM, 기존 실제 motor cut과 neutral command를 유지한다. runtime rate/history를 인위적 phase 또는 reference command로 대체하지 않는다.

## 실행 단계의 구분

후속 freeze는 생성 작은 recurrent CUDA 검산, 기존 기준 동작 보존 대조, 신경/adapter 저장·재시작 및 차단, 실제 동일 초기상태10초 BANC–신체 측정을 구분하고 각각 자원을 사전 배정해야 한다. 조건부 오차가 작다는 사실은 실제 피드백에서 지속 보행한다는 증거가 아니다. 새 전체 모델의 F3/F4/F5와 slip을 확인한 후에만 F6 미사용 초기상태 및 F7 목표 회귀로 진행한다. UI/최선E150/M2는 검증된 후보 채택 전 보존한다.

F1–F8 미완료 상태와 생물학적 기전 미검증을 유지한다. 본 메모에는 원본 실행 승인이나 새 실험 자원 배정이 없다.

## E217 검산 후 E218 진행 중인 갱신 (2026-09-25)

위 본문은 E211 이전 설계의 기록이다. E214 적합도 음성이었고, E215/E216은 원래 반응속도·cap/gate·근육필터를 지킨 임의 시간별 forcing의 조건부 전신 용량을 확인했다. 현재 실제 구현 후보를 찾는 단계는 E218이며, 아직 원본 적합 결과나 native 후보는 없다. 이 갱신도 실행 freeze가 아니다.

E217이 검산한 현재 가족의 파라미터는 own-MN centers391×64×9, scale391×3, sigma²391, actual inverse_adaptation_tau391, 원래 tau/cap/r0, membership391, shared beta96×64다. beta 중 free90근육만 조정하고 fixed6근육은 무시한다. runtime 상태는 두 기억 h_fast/h_slow 각391×3과 새 반응 rate다. h_fast16ms/h_slow 실제 adaptation reciprocal를 사용하고 반응 gate는 fast에만 의존한다. E217의 shared group anchor 선택은 offline이며 다른MN의 실시간 상태는 입력에 없다.

native 통합으로 넘어가면 B기저2001×391×64나 D_group 시간배열을 재생하지 않는다. 현재 자기 x/h_fast/h_slow에서64확률을 계산하고 고정 beta_owner를 내적해 자기 phi를 얻어 rate를 적분한다. 해부학적 연결의 실제 전류는 매 RK stage에서 바뀐 전체 rate로 재계산해야 한다. fixed48과 나머지 뉴런도 실제 재귀 입력에 반응해야 하며, E217/E218 조건부 검산에서 사용한 기록 rate 대입은 native에 허용되지 않는다.

float64 조건부 검산과 실제 packed float32 CSR 사이 계약은 여전히 미결정이다. native 정밀도 규약을 따로 고정하고 작은 재귀 생성 모델에서 stage 단위 검산을 먼저 해야 한다. snapshot에는 두 기억·정밀도·공유계수·소유권·center의 provenance가 모두 포함되어야 한다. 원래 adaptive 상태를 보존할지 여부도 명시하고 warmup/capture/restore가 기억을 몰래 전진시키지 않게 한다. 조건부 fit의 proxy 통과가 선행되어야 하며, 아직 신경·신체/F6 실행 자원은 배정하지 않았다.
