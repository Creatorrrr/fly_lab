# E138 전체 BANC 근육별 적응 시간상수 결과

2026-09-21 KST · H138 supported · 유효성 valid · M3 F1–F8 미완료

**전체 BANC의 연구용 CUDA 시간상수 기능 검증 통과**. 원래158706개 신경과391개 운동신경의 CUDA 계산에 근육별 적응 시간상수를 연결했다. 기본값은 기존 전체 상태·운동 출력과 정확히 일치하는지, 비균일값은 실제 출력에 영향을 주는지, 저장 복원 뒤 미래가 정확히 재현되는지를 따로 검증했다. 이는 고정 감각 입력을 사용한 신경 기능 검증이며 자연스러운 보행이나 생물학적 기전 검증 결과는 아니다.

## 구현과 리뷰

부모가 설계하고 실제 Sol/xhigh가 구현했다. 새 CUDA 포인터, float32 연산 순서와 상태 복원이 함께 바뀌어 xhigh를 선택했다. 부모 첫 리뷰에서 생성 CUDA 준비 비용의 잘못된 기대값과 실행 전 기록 저장 실패의 상태 구분 문제를 발견했다. 같은 Sol/xhigh가 첫 수정을 맡았고 부모가 두 번째 리뷰를 수행했다. 원래 실패 기록은 보존했다. 최종 소스와 리뷰 해시는 `parent-e138-final-review.json`에 있다.

부모의 별도6셀 검산 54개 항목은 CUDA·CPU·독립 DOP853, 입력 변경, 신경 출력 차단, 저장 미래와 손상된 상태 거부를 확인했다. 원본 결과는 저장된 전체 신경 raw, 실제 장치 시간상수, CSR·기존 생리 매개변수 및 독립 decoder 계산으로 다시 검산했다. 원본 독립 검사 360개, 해시 217개다. 부모의 첫 저장 자료 검산에서 CSR 구성요소 해시 묶음을 파일 SHA와 비교하는 타입 오류가 발견됐다. 부모가 해당 비교만 수정하고 같은 raw를 다시 검산했다. 원래 실패 보고서와 비교 기준을 보존했으며 원본 신경 실행은 반복하지 않았다.

## 원본 실행과 범위

원본1프로세스, 모델 4개, 신경 640tick/0.160초, decoder 32회다. CUDA 준비 240tick은 별도로 계수했다. 실제 신체·물리·적합·LP·F6 노출은0회다. 외부 15.300031초, 감독 15.342523초, 보수적 누적 1716.342523/1742초이며 재시도는 없다.

모든 arm의 운동신경 gain은1로 고정했다. E137에서 선택된 LF17근육의 시간상수만 적용한 기능 실험이며 E137 gain/연결 보정을 보행 후보로 적용한 것이 아니다. M2 263파일과 활성 UI·제품 기본값을 보존했다. 남은 다섯 다리의 적합, 동일한 전체 후보의 실제 폐루프 보행, 미사용 초기상태와 목표 이동 회귀 및 제품 복원 통합이 남아 있다.

다음 설계: Native timescale capability now has whole-BANC CUDA evidence. Freeze a distinct remaining-five-leg fit only after using all93 saved E128 responses to choose fixed starts, identical startup/guard contracts and a prospective resource allocation. Keep E137 LF selected8 separate until a single 96-muscle/391-MN candidate passes combined conditional and actual closed-loop body checks. Do not combine successful metrics from different candidates. F6 remains untouched; product restoration/integration and all F1-F8 remain pending.

근거: `verification/banc-natural-gait-m3-20260919/parent-e138-final-review.json`, `parent-e138-native-output-review-corrected.json`, `e138-native-timescale/`.
