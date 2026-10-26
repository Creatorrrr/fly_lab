# E180 실제 폐루프 보행을 목적함수로 연결하는 제한 실험

상태: 실행 전 설계. E179 이후 제안했던 여섯 다리 Hill 전용은 채택하지 않는다. 설치된 FlyGym musculoskeletal 자산은 별도 FlyMimic 몸체의 왼쪽 앞다리 15개 근육만 포함한다. 프로젝트 fixture는 그중 경골 길항근 두 개만 사용하며, SI 힘 교정과 나머지 다리 경로가 없다. 또한 기존 M2-E24/E26 강성 증가와 E58 활성 의존 감쇠의 유효 실패를 보존한다. 이 한계를 임의 경로 복제나 상수 적합으로 숨기지 않는다.

## 방향과 가능한 결론

부모와 실제 gpt-6-sol/medium의 범위를 제한한 소스 감사에서, E150 계열 모집 적합은 저장 신경 신호에 대한 조건부 명령 적합이었다. E169의 '폐루프 입력'도 이미 저장된 bank를 사용했으며 새로운 물리 실행은 0이다. 저장 명령 오차 가족의 기각은 실제 폐루프 보행 목적함수의 기각이 아니다. 다음 방법은 실제 전체 BANC–FlyBody 실행을 평가 함수로 연결하는 것이다. 이번 한 쌍의 비교는 최적화 완료나 자연 보행 성공을 뜻하지 않는다.

가설 H180: 기존 학습된 근육 모집 전달을 원래 해부학적 평균 전달 쪽으로 아주 조금 정규화했을 때, 조건부 교사 오차가 아니라 실제 감각·신경·신체 폐루프에서 전진 지속성의 유용한 변화가 관측되는가? 경쟁 설명은 기존 보정이 필수적이어서 변화가 없거나 오히려 악화한다는 것이다. 이 방향 선택은 신경 전달 구조와 실현 가능한 비교에서 나온 공학적 가설이며 생물학적 사실이 아니다.

## 고정 비교와 소스 구현

- 비교군: E150 원래 중립 checkpoint, 기존 MonotoneRecruitmentTrial 전체 신경 및 신체. 최신 E177 baseline도 동일한 E150 미래를 재현했다.
- 후보: `S(alpha) = (1-alpha)*S150 + alpha*S_anatomical`, `alpha=0.01` 한 값. `S_anatomical`은 기존 pool을 10ms bank의 두 branch에 넣는 원래 canonical 전달이다. 나머지 bank의 canonical 기울기는 0이다. fixed 19개(6 LTM와 singleton)는 바이트 단위로 원래 값을 유지한다. 해부학적 membership, 부호, pool, moment, rest, kappa, tau, 신경 연결·동역학, 몸체와 입력은 유지한다.
- 양수 slope, 각 근육 min/max slope 합 [0.5,2], per-MN 양수 하한은 기존 MonotoneRecruitmentAdapter가 실제 생성 값으로 다시 검증한다. 기존 가족의 볼록 결합은 이 제약을 보존하지만 부동소수점 허용치는 원래 구현을 따른다. 새 Hill 모델이나 임의 힘 상수, 목표/시간/위상 입력은 없다.
- alpha=0은 연산 없이 원래 배열을 복사하여 byte-exact baseline을 보존한다. 후보는 중립(신경/몸체 시각0, bank/activation0)에서만 조립한다. 움직이는 checkpoint의 파라미터를 제자리에서 바꾸지 않는다. 실제 기존 trial schema/identity/snapshot을 사용하고 후보 파라미터를 provenance와 함께 저장한다.

## 단계와 실행 경계

1. 부모 설계 후 실제 gpt-6-sol/high가 `tools/probe_c_banc_m3_native_objective.py`와 대응 테스트를 처음 구현한다. 부모 FIRST 리뷰, 필요한 경우 같은 Sol 첫 수정, 부모 SECOND 리뷰/잔여 수정 순서다. 기존 production·M2·UI·평가기 파일은 편집하지 않는다.
2. 생성 자료로 alpha=0 exact, canonical endpoint, fixed 19/원래 계수 불변, 잘못된 입력 거부, native 평가의 불완전·건강성 종료/분모와 clock 처리를 확인한다. 본 모듈 import 자체가 모델을 만들거나 원본 자료를 읽어서는 안 된다.
3. 부모가 입력 checkpoint/graph/startup 원자료 및 재사용 소스 해시를 별도 freeze하고 소스 리뷰를 통과한 뒤에만 원본을 읽거나 모델을 실행한다. 원래 시작99 q RMS<=.02rad/max<=.04rad 및 LTM 동일 문턱을 유지한다. 저장 신호의 사전 대조를 통과하더라도 실제 후보 startup을 다시 검사한다.
4. 최초 실제 paired 진단은 중립에서 각 3초(600 control), neural .25ms*20/control, physics .1ms*50/control, CPU1/GPU1 순차 실행이다. 실제 CUDA 장치, 전체 snapshot, 원시 control·native contact, 종료 원인, 실제 시각과 비용을 남긴다. 비교군이 보존 E150 3초 prefix를 재현해야 유효한 비교다. 기존 evaluator로 품질을 계산하고 [0.5,3] 같은 창을 비교한다. 3초 결과를 10초 F3–F5 통과라고 표시하지 않는다.
5. 개발 진입 조건은 기존 E162와 일치하는 방향으로, 실제 전진과 net/path가 비교군의 95% 이상, native slip 증가 없음, 건강성 유지, longest forward bout>=.3초, 모든 다리에 적격 주기>=1을 요구한다. 짧은 창의 이 조건은 최종 F1–F8을 대체하지 않는다. 긍정적이면 같은 alpha를 새 fit 없이 원래 10초 계약으로 검증하고, 부정적이면 이 한 방향의 실패로 기록한다. alpha sweep이나 사후 문턱 완화는 하지 않는다.

## 자원와 기록

E179 이후 원본 실행 보수 누적3423.9201528000413초/기존 배정3656초를 승계한다. 이번 소스 구현·생성 검산은 원본 실행0으로 구분한다. 실제 paired 진단은 사전 freeze에서 내부/외부 한도와 남은 배정을 확인하며, 자동 재시도0·각 process300초 이하·한 GPU·출력1GiB 이하·여유4GiB를 적용한다. 아직 freeze/source gate/원본 읽기/모델·신경·물리 실행은0이다. 실제 비용이 확인되기 전 예산을 사용한 것으로 기록하지 않는다.

이 설계는 기존 finite-goal의 방법 변경이다. 최종 동일 후보의 F1–F8, F6 미노출, M2/활성 UI 보존, 과거 유효 실패·invalid 실행 및 누적 자원은 모두 유지한다. 전체 목표는 active이고 후보 채택은 없다.
