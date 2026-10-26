# E200: 무출력12근육만 모집하는 개입의 명령 보완 가능성

2026-09-24 E199 이후 부모 사전 설계. E199에서12/96근육·28MN이 전체 저장 보행에서 무출력이었다. 활동 중인84근육은 그대로 두고 이12개를 새로 모집하는 것이 현재 명령 차이를 메울 만한 방법인지, 비용이 큰 새 신경 동역학 구현 전에 판별한다. E178/E179의 RM 일부 MN 침묵과 달리 이번에는 근육 전체가 무출력이다.

H200: 이12근육에 자유로운 비음수 activation을 허용하면 기존 개발용 참조에 대한 E150의42관절 명령 RMS를20% 이상 줄일 수 있다. 경쟁 설명은 부족한 방향이 다른 활성 근육의 크기·형태·위상에 있고12근육 추가만으로는 충분하지 않다는 것이다. 이20%는 이전 저장 전달 시험과 연결되는 **방법 선택 기준**이며 최종 F 기준이나 보행 개선의 필요조건이 아니다. 더 작은 개선도 실제 보행에 도움이 될 가능성은 남긴다.

## 고정 입력과 수학

E197의 이미 독립 검산된 `pre-solve-raw.npz`에서 기존 target_indices1824개, target1824×42, recorded_command2001×42를 재사용한다. E197의 prefilter 후보는 사용하지 않는다. E150 adapter의42×96 moment/rest와 native activation, E199 결과의 전체 창 무출력 명단을 맞춘다. 원본 manifest와 부모 최종 검토, 모든 입력 SHA를 고정한다. E85 참조를 새로 정렬하거나 목표 mask를 고르지 않는다.

S=[8,17,21,34,37,42,47,52,65,66,71,82], B=.02*moment[:,S], Y=target-recorded_command[target_indices]. 각 행에서 `min_{a>=0} ||B a - Y_t||²`를 SciPy NNLS로 계산한다. 자유 a는 상한·활성 시간상수·신경 입력·시작 상태·속도 제약을 모두 생략한 **낙관적 오프라인 완화**다. 미래 runtime controller로 저장하거나 재생하지 않는다. NNLS 후보와 별도로 B의 SVD 열공간에 대한 signed projection도 계산한다. 후자는 비음수 조건까지 생략한 더 넓은 상한 비교다.

정합: E199의 무출력 index가 S와 exact, 모든 native activation[:,S]==0, original native clock/completion/availability 정상, E197 loaded moment/rest와 E150 adapter exact. rest+.02*activation@moment.T와 recorded_command 차이≤1e-12. target_indices는 유일·증가 int64의1824개/1..2000, target/command는 유한. 원본 기록과 mask를 변경하지 않는다.

NNLS에는 maxiter=10*12를 고정하고 자동 재시도하지 않는다. a≥0, KKT gradient B.T@(B a-Y)≥-1e-9, complementary max|a*gradient|≤1e-9를 검사하며 실패는 inconclusive다. SVD rank 문턱은 max(B.shape)*eps*s_max의 통상 float64 수치 rank로 고정하고 singular values/rank/orthogonality와 잔차를 남긴다. baseline/NNLS/signed-span의 전체 및6다리 RMS, 제곱 오차 감소량,12근육별 activation 분위수와 정확0 비율을 모두 보고한다. 여러 근육이 같은 관절을 길항 구동하면 a는 비유일할 수 있으므로 개별 분해를 유일한 생물학적 해로 부르지 않는다.

NNLS 오차 비율이≤.8−1e-9이면 'saved nonnegative command headroom'이며 해당 신경 입력/반응식 개입 설계로 연결한다. signed-span의 오차 비율이>.8+1e-9이면 이12근육만으로20%를 달성한다는 가설을 해당 저장 과제에서 기각한다. 그 사이에는 signed relaxation 통과/비음수 미달 또는 문턱 근처로 분리하며 신경·물리 가능성으로 승격하지 않는다. 모든 경우 startup/velocity/실제 폐회로와 F1–F8은 미검증 상태다.

## 실행 계약

actual gpt-6-sol/medium이 `tools/analyze_c_banc_m3_silent_muscle_capacity.py`, `tests/test_c_banc_silent_muscle_capacity.py`만 구현한다. 작은 NNLS/선형대수와 정합 작업이므로 medium이다. 부모 FIRST → 필요한 첫 수정은 동일 Sol → SECOND/잔여 수정은 부모. 부모는 별도 SciPy bounded least squares와 투영 검산으로 비교하며 원본 전체 신경/신체를 실행하지 않는다.

원본1/retry0, CPU1/GPU0, 내부15/외부20/감독30초, 출력8MiB. 누적5031.485320299715/5144초에서30초를 예약하며 상한 증액0이다. 준비/합성/검산 비용은 별도 기록하고 미계측을0으로 간주하지 않는다. 결과·raw 배열·소스 SHA/원본 실패를 보존한다. M2/UI/F6 변경0, 최선E150/모든실패/F1–F8 active·미완료 유지.
