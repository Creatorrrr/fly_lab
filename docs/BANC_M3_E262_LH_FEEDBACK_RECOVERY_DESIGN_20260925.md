# E262 전체 힘줄 입력을 보존한 폐루프 자극 복구

E261은 첫 sham에서 held tendon 입력의 복부2개 항목(각0.2)이 빠져 무효였다. 원래 snapshot에는 이 세션 dictionary가 직접 포함되지 않는다. 부모 감사는 prefix0..55 전체와 sham의 기존 신경/몸/native 배열이 exact이며 trace 차이가 tendon key뿐임을 확인했다. E261 실패·비용·H261 미시험을 보존한다. 유효한 control55 fork를 재사용해 prefix55controls는 반복하지 않는다.

E262의 과학 질문과 대조는 E261과 같다. E257 W6 후보의 동일 control55 whole에서 sham0/plus/minus/sham1 각20controls를 실행한다. 첫control56의 body 호출에만 q19 목표±0.0013821410102253795rad를 더한다. joint 및 실제 actuator 제어 범위를 모두 확인하며 다른41목표·adhesion·tendon·kwargs·신경·감각·decoder는 유지한다. 이후 원래 decoder 출력만 사용한다. 이 진단 pulse는 최종 제어기가 아니다.

시작은 검증된 E261/fork다. 그래프·신체 identity와 전체 saved state는 그대로 구성하고 E257 control55의 모든8개 tendon_inputs를 별도 sidecar로 고정한다. 복원 함수는 전체8개 dict를 명시적으로 복사하고 adapter의6개 값과 일치하는지 검증한다. 단순 update로 이전 branch의 여분 key가 남지 않으며, adapter 외 복부2개도 빠지지 않아야 한다. 알려진 모델의 tendon_control.names와 값/키를 대조하고 시작·끝 held 입력 sidecar 및 trace를 저장한다. 오염된 dict·비기본 복부값·여분 key를 포함한 생성 fixture에서 정확히 원래 전체 입력으로 돌아오는지 시험한다. 기존 E261 소스·결과·승인 해시는 수정하지 않는다.

4회 제자리 복원 전체 state exact, 각 anchor control55 trace exact, sham0/sham1의55..75 전체 원래 E257 trace/MN391/82/core24/bank/감각/native1000행 exact 및 두 sham 최종whole·held 입력 exact를 요구한다. wall_s만 비교에서 제외한다. 실제 첫control 신경·원래 decoder는 sham과 같고 오직 전달된q19만 pulse로 다름을 확인한다. endpoint21행/encode20행/native1000행, 원래/전달명령·물리굴곡/Jacobian·입력해시를 그대로 보존한다. 실제 finite·시계·경고/몸통접촉·upright≥.9와 pulse clipping 조건을 유지하며, 이미 알려진 E257 startup 추적오차를 새 기술 실패 기준으로 추가하지 않는다.

검출은 plus−minus의 control56 실제q19>1e−8rad다. controls58..75 원래 decoder q19의 paired 평균D>1e−8이면 supported_positive_return, D<−1e−8이면 refuted_compensatory_return, 나머지는 inconclusive_below_resolution이다. 검출 미달과 기술/건강성 실패는 별도 inconclusive/invalid다. 감각/bridge111970/MN47329/bank/muscle44/전체q19 signed·positive·negative 차이를 기술하고 전체 회로의 상쇄 가능성을 유지한다. 특정 연결의 유일한 매개나 자연 보행 향상으로 확대하지 않는다.

실제 gpt-6-sol/high가 새 tools/probe_c_banc_m3_lh_feedback_recovery.py 및 tests/test_c_banc_lh_feedback_recovery.py만 구현한다. 부모 FIRST→필요시 같은 Sol 첫수정→부모 SECOND→부모 잔여수정. 이미 검토된 E261 관측/분기 함수의 명시적 인자 재사용은 허용하나 gate·schema·전역상수·원본출력 경로를 monkeypatch하지 않는다. 부모는 동일 raw에서 독립 검산한다. 오류·cleanup·publication 실패는 지지 해석을 철회하고 원자료/비용을 보존한다.

원본1/retry0/CPU1/GPU1/whole1/network7/setup280실행중capture140/body1/settle2000, 네 분기80controls=1600neural ticks/4000physics steps, 제자리복원4. 추가 prefix/current probe/fit/solver0. E261 관측29.222초와 prefix 생략을 근거로 internal45(마감8)/external48/supervisor50초를 사전 예약한다. 기존 누적7050.637872599607초는 유지하고 미래 단계 배정7074→7101초(+27), 잔여50.362127400393초다. 사용자 지정 전체 시간 예산 신설이나 실패비용 초기화가 아니다. 호스트6GiB/추가GPU4GiB/최소GPUfree4GiB/출력384MiB/최소disk1GiB, 준비·생성·부모감사 별도/개발 미측정null.

새 write-once 출력은 verification/banc-natural-gait-m3-20260919/e262-lh-feedback-recovery/다. E150 best partial·M2/UI·F6 미노출·같은 최종 후보의 F1–F8 미완료를 보존한다. 결과가 positive면 해당 자세의 감각·모집 변환 교정 대조를 설계하고, compensatory면 단순 감각 부호 반전은 배제한 채 동작점/다른 근육 상쇄를 우선 조사한다. 불검출은 더 큰 임의 자극으로 바로 반복하지 않고 관측 해상도와 실제 전달을 먼저 판정한다.
