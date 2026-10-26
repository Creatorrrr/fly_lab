# Startup ray: ROOT SECOND와 ROOT 잔여 수정

같은 실제 GPT-6.1 Sol/xhigh가 ONE_R1에서 작성한 science75396 bytes/e65b33212e24b6d37f222d16f6275112cda71a659662a228f94942c1a007b752, supervisor26778/c4f968c8579c7a11710ce18dae1ce877915f125edecaddd822a02aea7d41ab95, numerical tests5499/1d42986ebe9dcbefb2a4dfc69b56cb246953130dde9e16f13c0bbfecc5c7995c, execution tests13439/9813b16b50ef3230849fe9d52e53d036b69580c8d1fb7525083bf0d6a3a718c4 및 Native CLI3818/56e6e945ecd6e957e18313b4a61a2dacfb97fdcad7f337759e35188b9c2635c6을 읽었다. 최종 인계 서류 작성은 ROOT가 중단했고, 이는 Sol의 stable handoff 출판이라고 주장하지 않는다. ROOT 수령 기록은 `parent-startup-ray-ROOT-received-Sol-R1.json`이다. 추가 R1은 없으며 잔여는 ROOT가 수정한다.

FIRST에서 찾은 path normalization, 실제 기존 shared lock, 정확한 PID/Windows 생성 ticks, suspended worker→Job 소속→resume, same ENTRY, 예약·pending·actual EOF, 새 dependency loader와 독립 forward roster 검사가 구현됐다. 생성 테스트는 production gate의 정상/거부/late drift/pre-array와 실제 정상/nonzero/timeout-descendant·공유 lock 경쟁을 관찰하도록 작성됐다. 이것은 Source 리뷰 결과이며 runtime 인증이 아니다.

ROOT 잔여 범위:

- 실제 존재하는 E318 whole/chunks/control/recruitment에서0..99를 추출하는 `banc_m3_startup_ray_inputs.py`를 연결한다. 제안의 미존재 startup4 별도 archive를 실제 입력으로 가정하지 않는다. 전체 neutral99 재현은 실제 ray에서 먼저 검사한다.
- explicit cold pre-array 검사와 원본 파일별 hash/timeliness를 유지하면서 digest의 매 block마다 전체 Source를 재검사하는 중복을 제거한다. pre-spawn/pre-array/pre-construct/pre-publication/post-readback fence는 유지한다.
- Native admin/collector의 실제 Source·request·기존118 costs·issued offset·once record·Original6000을 연결하고, 기존 미통과 E360을 통과라고 표시하지 않는다.
- 실제 lint/정확한 수치6/유한 gate/process·중복 once 검사를 시행하고 필요한 잔여 수정은 ROOT가 마무리한다. 실제검증 전에 Source freeze를 만든다. 구현이 바뀌면 새 Source에 대해 다시 확인한다.

현재 Python/Original/CUDA/body/수치 검증은 미실행이며 F1–F8, short screen/startup99, anatomical provenance/no external correction 및 M2/UI 기준은 그대로다. 과학 결과가 실패해도 기준을 낮추거나 body 결과로 alpha를 재선정하지 않는다.
