# E283 — 실제 중간점에서 Jacobian을 갱신하는 시작 보정

2026-09-27 root 사전 설계. M3 finite-goal/F1–F8과 E277 학습·E279 native·E281/E282 초기화 기각을 유지한다. E282의 RF-only 저장 dual은 원래 neutral±.05의 고정 중립 선형화에서 q peak 하한 .036805003121을 지지한다. Global norm/LTM/다른 다리/1e-6 여유 제거만으로 최종 내부 peak .0359964에 도달할 수 없다. 이 증거는 비선형 경로 전체의 불가능성을 뜻하지 않는다.

H283은 완만한 중간 목표에서 실제 응답과 fresh J를 얻으면 동일 neutral 영역 안에서 고정 최종 목표에 도달할 수 있다는 것이다. 경쟁 설명은 갱신한 국소 응답에서도 제약 충돌이 남거나 네 번의 보정으로 실제 비선형 오차를 줄이지 못한다는 것이다. E281 네 성공 다리만 합치면 RF는 여전히 중립이므로 RF 선형 불가능성을 해결하지 못한다. 그래서 RF를 포함한 전체를 중간 목표로 이동시키는 continuation을 선택한다. 최종 목표 완화나 같은 중립 문제 재시도가 아니다.

## 고정 두 단계

동일 E281 startup101 cases/context/engine/targets/51 source/1,660 edges/4,980 계수와 LF/LM/LH/RF/RM/RH 순서를 사용한다. 전체 중립 q/LTM/both J를 E281 저장값 abs1e-10/1e-9로 재현한다. 모든 실제 q/LTM 원자료는 변환하지 않고 저장한다. E282 inward-f32 coefficient interval, coefficient/interior/원래 중립±.05/한 보정±.05, exact pre-evaluation interval 검사와 sparse 인증은 유지한다.

| 단계 | 실제 q peak 한도 | 실제 global q SSE 한도 | q 제약용 scale | 최대 보정 |
|---|---|---|---:|---:|
| 0 중간 목표 | `.038*(1-1e-4)` | `4158*(.0171*(1-1e-4))**2` | `18/19` | 2 |
| 1 최종 목표 | `.036*(1-1e-4)` | `4158*(.0162*(1-1e-4))**2` | `1` | 2 |

중간값은 기존 .04/.018과 최종 .036/.0162의 산술 중간이다. 결과에 맞춘 다른 중간값 sweep은 없다. 두 단계 모두 LTM peak `.025*(1-1e-4)`와 각 다리 SSE `99*(.009*(1-1e-4))**2` 그대로다. 실제 native .04/.018/F1–F8 기준은 바뀌지 않는다.

제약 계산에만 qerr와 q Jacobian을 해당 scale로 함께 곱하면 E282의 동일 global QP/SOC와 certificate를 재사용할 수 있다. LTM과 계수는 scale하지 않는다. 원래 q/J는 별도 raw 파일에 그대로 남긴다. QP metadata에 stage/scale/물리 한도를 명시하고 root가 저장 원문제가 `scale*actual qerr/J`인지 검산한다. 중간 단계의 1e-6 선형 여유는 제약 단위 기준이라 실제 q 단위에서는 19/18배다. 이 변환은 중간 문제의 단위변환이며 실제 신경 출력이나 최종 점수를 바꾸는 경로가 아니다.

각 보정 후 여섯 다리를 새로운 startup engine에서 순차 재평가한다. 평가는 항상 원래 target과의 실제 qerr를 확인한 뒤 제약 view를 만든다. 단계 수용은 원래 단위의 위 peak/SSE/LTM/계수 조건과 normalized strict 제약을 **모두 tolerance0**으로 만족해야 한다. 중간 단계에 성공하면 마지막 실제 응답과 fresh J를 다음 단계로 넘기며 neutral trust 기준을 재설정하지 않는다. 단계 전환 자체의 추가 평가·초기화·상태 blending은 없다. 최종 단계가 이미 만족되면 보정 없이 다음 확인으로 간다.

각 단계 최대2회/합계4회이며, 단계 실패 또는 uncertified QP는 즉시 전체 기각이다. 중간 실패를 넘어 최종 단계로 가거나 세 번째 보정/line search/clip/새 scale/retry를 실행하지 않는다. 원시 해는 수정하지 않는다. 모든 f64 최종 조건이 만족된 경우에만 계수를 한 번 f32→f64로 바꾸고 여섯 실제 startup을 derivative=False로 재평가한다. 최종 normalized scale1/원래 단위 기준 모두 만족해야 `initialization_supported`다. 중간 성공은 최종 성공으로 기록하지 않는다.

## 실행 의미·판정과 자원

Source 모델이나 decoder/시각/감각/신체를 바꾸지 않고 최적화의 접근 경로만 바꾼다. Teacher/train/development/whole BANC/body/physics/F6/제품 후보 생성·채택·fullfit은0이다. 성공은 다음 fullfit 설계를 위한 조건부 초기화의 근거이며 자연 보행 완료가 아니다. 실패는 고정 단계·중립 영역·네 보정 한도 내의 결과이고 모델 전체의 불가능성이 아니다. 실행/저장/deadline 오류는 invalid/incomplete 및 부분 원자료/실제 비용을 보존한다.

Actual gpt-6-sol/high가 신규 `tools/verify_c_banc_m3_startup_continuation.py`, `tests/test_c_banc_startup_continuation.py` 두 파일만 구현한다. 새 runner 안에 stage view/strict stage/driver를 두고 E282의 문제 구성·인증·engine factory/meter/round evaluator를 그대로 import하여 재사용한다. 기존 frozen source를 수정하거나 production module globals를 monkeypatch하지 않는다. Root FIRST→같은Sol 첫수정→root SECOND→root 잔여수정 순서. 생성 CPU wall120초/tiny≤20회/n≤6, 원본/GPU0에서 FIRST 제출한다. Root 비용은 별도 기록한다.

E282 freeze/review/root raw audit/RF-only proof를 SHA로 고정하고 inherited24 source, 기존 모든 input/cases/roster/packages 및 E280 resource qualification을 계승한다. Production six-leg/4,980변수를 E282 validator로 강제한다. Root 승인·입력/소스 검증 전에는 원본예약0이다.

원본은1회, science40초/cleanup15초/supervisor55초, RAM6GiB/engine GPU추가 추정4GiB/출력2GiB/free disk4GiB. 각 평가마다새engine, 실제 live1, engine/evaluation≤36, derivative평가≤30, forward≤72000/derivative≤60000ticks, QP≤4(max4초/100iter/thread1/Clarabel0.11.1/tol1e-8)다. Stage·round·QP·engine 및 오류/cleanup/partial 비용을 모두 보존한다. 원본 누적7845.051662199552/7905초, 잔여59.948337800447916초 내에서55초를 배정하며 증액은 없다. 현재 GPU 등은 root 원본 직전에 재확인한다. M2/E150/UI/F6 미노출을 보존하고 실제 결과에 따라 다음 결정을 기록한다.
