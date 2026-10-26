# E178: 저장된 전체 신경 상태의 입력 전류 분해

E177은 같은 초기 상태의 두 CUDA 실행을 완료했지만 보행 협응과 반복성이 개선되지 않았다. 특히 운동뉴런108052/153939는 두 팔 모두 발화율0이며 경계 recurrent current가 음수였다. 이번 질문은 그 입력을 만드는 실제 source와 가중치 변환 이력이다. 음의 합을 생물학적 원인으로 단정하거나 억제 연결을 삭제하지 않는다.

## 고정 범위

- E177 baseline/candidate 두 팔의 start, checkpoint-1s, main-1p1, final만 읽는다. 메타데이터로 확인한 control tick은0/200/220/2000, network tick은0/4000/4400/40000, neural dt는.00025초다. 주 실행 경계8개이며 clone/cut 상태는 사용하지 않는다.
- 각 상태의 전체158706개 rate, drive, output_mask, adaptation을 원래 JSON 참조로 읽는다. 전체 저장소나 모델을 복원하지 않는다. 실제 arm별 actual-global-csr, feedback-boundary-state, recruitment-metadata, assembled-calibration/parameters와 원본 graph CSR/count/weight/nodes를 pin한다.
- 분석 post IDs는 기존 E177 관측 순서30981,107685,12036,32830,108052,117483,153939로 고정한다. 56개 완전 row를 분석한다. 기록되지 않은 값을0으로 대체하거나446cell 로그로 전체 presynaptic 상태를 추정하지 않는다.
- node index와 namespaced root ID를 구분한다. 원본 GraphStore의.275/contact와 rate-model의.03/contact를 혼용하지 않는다. E150은 optional annotation reference를 사용하지 않았다. glutamate inhibitory 및 transmitter별 부호는 모델 가정이다.

## 계산과 검산

원본 값을 읽기 전에 입력 hash 및 부모 source gate를 검사한다. state manifest에 있는 선택4배열 참조와 state.json hash, 차원·dtype·유한성·경계tick·feedback availability를 검사한다. 실제 CSR topology를 graph topology와 대조하고 calibration 및 individual-fit entry를 실제 CSR와 비교하되 E177의 고정5개 차이를 명시적으로 적용한다. 두 팔 CSR 차이는 바로 그5개여야 한다.

전류 기준은 실제 CUDA boundary observer의 `weight * float32(rate * mask)`다. `--fmad=false`의32lane 순서(각lane edge+=32 누적, shuffle16/8/4/2/1)를 NumPy float32 곱·덧셈으로 재현한다. 7개 관측 셀의 rate/drive/mask/adaptation과 tick을 경계 로그에 exact 대조하고, 재구성 current도 float32 bit exact이어야 한다. 허용오차를 사후 늘리지 않는다. 누락·비유한·합산 불일치면 해당 진단은 기술적으로 미결이다.

해석용 각 edge 기여는 float32 masked rate와 실제 float32 weight를 float64로 곱해 math.fsum으로 더한다. 양·음·전체합, masked/unmasked 차이, graph/rate 기본값과 실제값 비율·부호 변화, 접촉 수, source ID/NT/세포형/상위분류/검증 annotation, 고정feedback/individual-fit/기타calibration/navigation-port 소속을 저장한다. exclusive group 우선순위는 feedback, navigation, individual fit, other calibration, base rate이며 겹치는 bool도 별도 보존한다. 모든 row의 edge별 raw를 저장하고 양·음 기여 상위10개와 그룹 합을 요약한다. double합은 CUDA 실행값을 대체하지 않는다.

recruitment-metadata에 post가 존재하면 tau/a/threshold/cap과 held drive/adaptation으로 경계의 유효 입력 `drive + current - threshold - adaptation`을 보고한다. 해당 post가 metadata cohort에 없으면 명시적인 unavailable로 남기며 대체 상수를 넣지 않는다. 순간 target-rate 계산은 경계 진단일 뿐 RK4 interval 전체의 원인 증명이 아니다.

## 사전 결정

1. 배열/현재합/부호·변환 이력이 코드 계약과 불일치하면 구현 또는 자료 계약 오류로 분류하고 좁은 수정부터 검토한다.
2. 계산이 일치하고 특정 모델 입력이 지배하면 source의 해부학 annotation과 공학적 efficacy를 구분해 다음 수정 위치를 선택한다. 음의 입력이라는 이유만으로 연결을 제거하거나 부호를 뒤집지 않는다.
3. 감각 활동/전환 시점이 이4시점에 충분히 포함되지 않으면 관측 범위가 부족하다고 기록한다. 추가 신경 실행은 별도 설계 없이는 하지 않는다.

어떤 경우에도 이 저장 진단으로 보행 성공, F6/F7 통과, 생물학적 검증을 선언하지 않는다. 기존 F1–F8과 활성 목표는 유지한다.

## 구현·자원

부모가 설계와 독립 oracle 및 리뷰를 맡고 실제gpt-6-sol/high가 최초 구현을 맡는다. 첫 수정은 같은Sol, 두 번째 리뷰 이후 잔여 수정은 부모가 맡는다. 별도gpt-6-sol/medium 읽기 전용 감사는 source 부호/단위/합산 규칙을 확인했다.

소유 파일은 tools/analyze_c_banc_m3_saved_current.py와 tests/test_c_banc_saved_current.py 두 개다. 기존 코드·M2·UI·과거 증거·F6를 변경하지 않는다. 생성 자료 검사 후 부모 gate를 생성하고 원본 저장자료 분석은 한 process, retry0으로 실행한다. 신경/물리/model construction/GPU/fit은 모두0이다. CPU1, internal90/external105/supervisor120초, raw128MiB/free1GiB를 고정한다. 기존 원본 보수 누적3417.939522700035/3656초 안에 배정하며 새 과학 신경·물리 비용은0으로 별도 기록한다. 저장 분석·부모 검산 wall 비용은 숨기지 않는다.

출력은 고정 e178-saved-current 디렉터리의 edge-contributions.npz, row-summary.json, source-nodes.json, input-receipt.json, result.json 및 output-manifest.json이다. manifest는 실제 파일 목록/hash/bytes를 포함하고 원본 mismatch 또는 실패를 성공으로 저장하지 않는다. 원본 재실행은 자동으로 하지 않는다.
