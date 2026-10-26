# E241 결과: 같은 보행 입력의 학습 기저 준비 완료

E240의 실제 BANC 입력으로 앞6초만 사용한 scale/64개 반응 중심과 전체10초의 history/B/H/affine readout을 만들었다. 부모의 저장 검산806개가 통과했다. 최대 anchor row는1200이며 뒤4초가 scale/중심 결정에 들어가지 않았다. 명령은 같은 control의 E85 적용 q와 exact이고, 학습 제외48개는 해당 궤적에서 실제로 관측한 MN 출력을 유지했다.

원래 decoder 관절 재구성 오차는2.00e−15rad, 독립적으로 pool 후 필터링한 D_group 오차는5.68e−14, RK4 닫힌식 decay 오차는9.44e−16이다. 원래 decoder의 교사 대비 q RMS는 학습.1573768543/검증.0987525110rad, 속도 RMS는3.497391287/2.657878534rad/s다. startup1..99는q RMS.225166718/max1.026368680rad다. 이 수치는 개선의 비교군이며 새 물리 성공 허용치가 아니다. 아직 계수를 fit하지 않았다.

실제 gpt-6-sol/high 구현에 대한 부모 FIRST에서 저장 inverse의 float64 승격 가정, 후기 저장 실패, GPU 동기화 뒤 readback 실패 회계, 생성 GPU 정리/기한, 테스트와 실제 구간 분리 경로를 지적했다. 같은 Sol 첫 수정 뒤 부모 SECOND의 남은 manifest 순서 문제를 부모가 고쳤다. CPU10검사/Ruff/compile, 별도 생성 CUDA1회와 저장 배열19검사(최대8.88e−16)가 통과했다.

원본 저장 분석은1회/retry0, 기존 history/basis CUDA 각1회(조건부40000ticks씩), 전체 BANC/신체/물리/solver/fit0이다. 감독36.9326022초, 최대트리RSS3,200,102,400B, 최대추가GPU1,658,847,232B, 출력734,803,782B, 소유PID3532/46728/45896모두정상종료다. 원본 누적6574.145712599628/6680초·잔여105.85428740037241초다. 생성GPU 감독3.3521431초와 부모 저장 검산은 별도다. 부모 검산의 첫 직접 launcher는 import 경로 오류로 .1647975초만에 종료했으며 원시 검사는0이었다. 동결 소스를 바꾸지 않고 저장소 root에서 runpy로 실행한 검산은5.2379628초(내부4.2272101초)에 완료했다. 두 기록을 모두 보존한다.

근거는 `verification/banc-natural-gait-m3-20260919/e241-matched-input-basis/`, `parent-e241-source-review.json`, `parent-e241-supervisor.json`, `parent-e241-output-review.json`, `parent-e241-final-review.json`이다. 다음은 앞6초의 q와 속도를 함께 맞추는 양수 근육 공유 계수의 단일 fit이다. 이후 시간 분리·실제 float32 readout·동일 신체·교사 없는 폐루프를 차례로 확인한다. 현재 F1–F8미완료/F6미노출/E150·M2·UI보존이다.
