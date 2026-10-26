# E301 — 실제 신체 연결에서 첫 신경 구간 유효성 실패

고정 E296 보호 roster와 u=.05/tau=.150에 E299/E300 회복 억제식을 적용했다. Neutral .005초는 기존 E150 control/native contact/recruitment 및 모든 whole scientific field와 정확히 일치했다. 독립 wall 시간만 별도 측정값으로 보존했다.

Candidate는 첫 control의 신경20 ticks(.005초)를 호출한 뒤 `Sticky recovery resource fault: 29`로 중단했다. Candidate 완료 control0, 실제 native physics0이며 몸체에는 실패한 신경 출력이 전달되지 않았다. Flag29는 resource 상태1·활성 source rate4·factor8·최종resource16의 누적 OR이다. 누적 flag만으로 어느 값이 먼저 잘못되었는지 확정할 수 없다. 초기 endpoint의 R/g가1인 것은 실패 이후 상태나 RK 내부 단계가 정상이라는 뜻이 아니다.

Candidate whole final과 network 부분 checkpoint는 유효성 오류 때문에 저장되지 않았고, 그 불가 사유와 adapter/body 부분 자료는 보존했다. 원래 오류, 시계, 실제 진행량, close를 root saved-byte audit로 확인했다. Root audit PASS는 이 실패 증거의 일관성 판정이며, 원본 실험의 `run_valid=false`를 바꾸지 않는다. Startup과 short gait 지표는 unavailable이고 보행 성능은 판정하지 않았다. F1–F8·F2 완료, 후보 채택, F6 노출,10초 실행 모두 false다.

원본 supervisor55.238402900초를 모두 이월해 누적 `8133.2913762995195 / 8210 s`, 잔여 `76.7086237004805 s`다. 실제 science neural40 ticks(대조군20+후보20), physics50 steps(대조군만), 구성2회, 실행 setup560 ticks/capture280 ticks, settling4000 steps이다. Host tree peak3,233,841,152 bytes, 소유 PID와 세션은 모두 종료했고 남은 원본 예약0이다. Root raw audit1.762642100초(프로세스2.1436267초), model/body/원본 추가0이다.

다음은 별도 E302 진단이다. 저장된 대조군 첫 control drive와 동일한 원래 신경 초기 상태로, 실제 후보 엔진의 RK 단계에서 처음 위반한 source/rate/R을 관찰한다. Strict guard·dt·u/tau·연결을 바꾸거나 실패 출력을 clamp하지 않는다. 전체 신체 재실행 없이 신경 한 개/최대20 ticks로 한정하며, 첫 잘못된 단계 뒤 CSR을 진행하기 전에 멈춘다. 첫 입력 동등성은 동일 whole 초기 상태와 driver 코드로 뒷받침하되 후보 실패 시의 drive를 직접 저장한 측정값과 구분한다.

근거는 `verification/banc-natural-gait-m3-20260919/e301-recovery-resource-body/`, `parent-e301-original-supervisor.json`, `parent-e301-original-raw-audit.json`, `parent-e301-live-preflight.json`이다. E301 freeze SHA `3a7dcc2897f8aaf5547c2ac623a177c6641e89e37fecf676b7ccb087c299d742`, source review SHA `978b0f0eb8b5151ef9341606cd06b47f13936a74784fcc33f371b3535034cd32`이다.
