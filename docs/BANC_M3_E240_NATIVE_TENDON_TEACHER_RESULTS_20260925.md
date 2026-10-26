# E240 결과: 관절 교사와 실제 BANC 힘줄 출력의 양립

원래 BANC decoder의 같은 control 힘줄 출력을 유지한 10초 관절 교사 실행은 고정 F3–F5 수치 검사를 통과했다. 전진 17.3305439146mm, net/path .7329958371, 전체 native 최소 upright .9905250215, nonfoot 0, 수치 오류 0이다. E85의 고정 힘줄 기준 전진 17.3491093916mm와 근접한다. 이것은 다음 학습에 사용할 수 있는 교사 자료이며 자율 보행 또는 생물학적 타당성 성공은 아니다.

고정 .5≤t<10초 구간에서 다리별 complete cycles는 LF/LM/LH/RF/RM/RH=22/22/22/22/21/21, period CV=.00248–.00350이다. 좌우 평균 위상은 .500497/.499819/.496800, circular R=.889531/.856767/.868525다. LF/RF/LH/RH 무릎 95–5폭은 71.265585/71.314461/46.867509/46.929827도다. 원래 평가기를 그대로 사용하고 E85 비교 결과도 재현했으며, 창과 기준을 조정하지 않았다.

실제 gpt-6-sol/high 첫 구현에 대해 부모 FIRST에서 후기 정리 실패가 arm 성공 기록을 남길 수 있음을 지적했다. 같은 Sol의 첫 수정 후 부모 SECOND와 생성 테스트 6개/Ruff/compile을 통과했다. 원본 1회/retry 0 후 부모는 21,741개 저장 결과 검사를 수행했다. 초기 whole state와 첫 control 신경 입력/출력 exact, 2001 control/40000 neural ticks/100000 physics steps/160000 RK stages, 391 MN 순서와 finite/counter/overflow/시간축, 실제 MuJoCo 힘줄 actuator ctrl=동일 control decoder 출력 및 원래 pool/EMA 재계산 오차 2.78e−17을 확인했다. 전체 신경 및 신체 최종 상태를 저장했다.

감독 129.1397226초, 최대 프로세스 트리 RSS 2,920,439,808B, 추가 GPU 1,478,492,160B, 출력 817,823,509B였다. 알려진 setup 280 실행/140 capture는 과학 ticks와 별개다. 소유 PID 23972/45624/42000은 모두 정상 종료했고 강제 종료·남은 프로세스는 없다. 보수적 누적 원본 시간은 6537.21311039964/6605초, 잔여 67.78688960036015초다. 부모 저장 검산은 별도 프로세스 6.158339초(검산 내부 5.153419초)이며 모델을 만들지 않았다.

근거는 `verification/banc-natural-gait-m3-20260919/parent-e240-source-review.json`, `parent-e240-supervisor.json`, `parent-e240-output-review.json`, `parent-e240-final-review.json`과 `e240-native-tendon-teacher/`의 원시 자료다. 다음에는 앞 6초에서만 입력 스케일과 반응 중심을 구성하고 뒤 4초를 분리하여, 이 조건의 관절 명령을 학습할 기반을 만든다. E197 과거 startup proxy와 E150 위상 재배치 목표를 새 교사의 목표로 혼용하지 않는다. F1–F8 미완료, F6 미노출, E150/M2/UI 보존 상태를 유지한다.
