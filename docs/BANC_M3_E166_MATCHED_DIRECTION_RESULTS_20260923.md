# E166 결과: 전진·중단 대응 비교는 판별 불충분

E166은 유효하게 완료한 저장자료 진단이다. 여섯 다리 모두 사전에 고정한 공통 표본 기준에 미달해 판정은 `inconclusive`다. 자연 보행 후보를 만들거나 채택하지 않았고 M3 F1–F8 전체는 미완료다.

E150의 기존 공통1824시점에서 같은 다리·시간창·비슷한 E1 위상/하중/AP 속도의 전진 순간을 중단 순간에 대응했다. 접지 load≥.02BW, 실제 ego AP>0의 조건이다. speed>.5mm/s는 순간 분류이며 지속적인 정상 보행 대조군이 아니다. 결과 확인 후 허용폭이나50% coverage 기준을 바꾸지 않았다.

|다리|적격 중단 표본|대응 표본|coverage|고유 전진 표본|대응 전진의 다섯 원점 모두 비양수|
|---|---:|---:|---:|---:|---:|
|LF|52|2|3.85%|2|50%|
|LM|111|33|29.73%|23|100%|
|LH|208|38|18.27%|17|100%|
|RF|4|0|0%|0|자료 없음|
|RM|224|71|31.70%|25|100%|
|RH|232|52|22.41%|19|100%|

대응 표본은 control 중복을 허용한다.71 RM pair는 독립71회 실험이 아니며 고유 전진 시점은25개다. 해당 RM 표본의 중단−전진 평균 Z 차이는−0.00005048mm다. LM−0.00008054mm, LH−0.00061778mm, RH−0.00105230mm였으나 작은 공통 하위집합 밖으로 일반화하지 않는다. LF는2 pair뿐이고 RF는 대응이 없어 방향 차이를 해석하지 않는다. 세 고정 시간창의 결과와 대응 전/후·고유 control 통계는 원본 summary.json에 모두 남겼다.

이 결과는 E165의 아래쪽 부호를 중단 특이 결함으로 단정하기 어렵다는 제한을 추가한다. 특히 대응된 중·뒷다리의 전진 순간에도 같은 부호가 나타났다. 동시에 공통 표본이 부족하므로 전체 상태에서 차이가 없다는 결론도 내리지 않는다. 위상·하중·AP 속도를 대응해도 몸 자세와 접촉 이력 등 차이가 남아 인과 효과가 아니다. 정적 body-origin 방향은 힘·실제 접촉점·발 clearance가 아니다.

따라서 E165 부호를 단독 근거로 말초 고갈 모델을 적합·도입하는 경로는 보류한다. 다음 설계는 별도로 검증된 E107의 within-leg shape 개선과 기존 제한된 출력 함수족의 실패를 기준으로 판단해야 한다. 임의 caliper 확대, 고갈 시간상수 sweep, E166의 불충분 판정을 통과로 변경하는 일은 하지 않는다. 감각/신경/근육 전달의 원인을 확정한 결과가 아니다.

실제 gpt-6-sol/high FIRST 구현 → 부모 FIRST 리뷰 → 같은 Sol FIRST correction → 부모 SECOND 리뷰/잔여 직접 수정을 수행했다. 첫 리뷰의 분수 tick/NaN mask 손실, 추가 caliper 허용폭, 후속 오류에서 pair 유실을 수정했고, 두 번째 리뷰의 관측 가용 범위와 고정 mask 대조 누락은 부모가 직접 수정했다. 원래 잘못된 입력과 실패 재현은 보존했다.

최종25 tests, Ruff/format/CLI 및 독립 cwd/PYTHONPATH 제거 생성 실행이 통과했다. 독립 생성 매칭4761 검사를 수행했고 부모 잔여 수정 전후 matching/summary 함수의 AST 동일성을 확인했다. 원본 결과의 별도 scalar 검산6277개도 통과했다. 테스트 수는 독립 보행 반복 수가 아니다.

원본 process1/자동재시도0, 신체/StateStore/GraphStore/신경/물리/Jacobian/fit0. 원본14파일, 감독17.451293초, 보수누적2880.451293/사전2953초다. 독립 저장 검산6.286652초는 별도 기록했다.3356개 고정 입력·과거 파일과 승인 소스/원본 출력도 보존됐다. M2/F6/UI는 그대로이며 전체 목표는 active다.

원본: [e166-matched-direction](C:/Users/ckthd/Dev/Projects/fly_lab/verification/banc-natural-gait-m3-20260919/e166-matched-direction/summary.json). 독립 검산: [result.json](C:/Users/ckthd/Dev/Projects/fly_lab/verification/banc-natural-gait-m3-20260919/e166-parent-original-verification/result.json). 계약: [E166 설계](C:/Users/ckthd/Dev/Projects/fly_lab/docs/BANC_M3_E166_MATCHED_DIRECTION_DESIGN_20260923.md).
