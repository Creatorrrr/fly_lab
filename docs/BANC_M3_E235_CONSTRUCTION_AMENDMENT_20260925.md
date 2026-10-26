# E235 실행 전 초기화 집계 정정

E235 원래 설계의 `trial복원내부2+교체1`은 부모의 잘못된 추정이었다. 원본 E235는 아직 실행하지 않았으며 기존 설계와 freeze v1은 보존한다. 이 정정은 입력·가설·출력·과학적 99 controls/1980 neural ticks/4950 physics steps·시간/메모리/출력 상한을 변경하지 않는다.

실제 복원 경로는 BancWalkingSession의 ResearchRateNetwork 1, CoreTrial 1, EfficacyTrial 1, IntrinsicRecruitmentTrial 1, MotorAdaptationTrial 1, MotorTimescaleTrial 1, NeuralCurrentTrial 1로 총7개이며 KneeTrial/RecruitmentTrial은 decoder만 감싼다. E234 observer 교체1을 더하면 총8개다. 이전 E224 effector construction receipt에도 capture interval7/실행 interval14, setup실행280/capture140 ticks가 기록되어 있다. E235에서는 기존 `profiled_construction`의 실제 진입/반환/동기화 계측을 유지하며 restore7+observer1 capture, setup실행320/capture160 ticks를 과학적1980 ticks와 분리해 검증한다. 실패하거나 동기화가 확인되지 않은 작업을 완료 또는0으로 처리하지 않는다.

`e235-effector-startup-stages-freeze-v2.json`만 실행에 사용하고 이전 freeze SHA를 보존한다. 원본 누적6090.0136165996555/6154초, 잔여63.986383400344494초 중60초예약 및 재시도0 그대로다. E235 FIRST의 neural_tick 표시가 stage0을tick0으로 보고하던 오류도 수정한다. stage index는0-based이며 해당 실행 tick은 floor(stage/4)+1, control은 floor(stage/80)+1이다. 건강 중단은 과학적 중단으로 보존하며 CLI 요약이 누락된 정상완료 필드를 조회해 기술오류로 바꾸지 않도록 한다.
