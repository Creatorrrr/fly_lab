# E221 결과: 실제 CUDA 재귀 신경 경로의 생성 모델 검산 통과

새 free-MN 반응을 packed BANC 신경 클래스에 연결했다. 네 RK stage마다 실제 재귀 입력의 부호별 전류를 사용하고, 자기 반응·두 기억을 float64로 적분한 뒤 다음 CSR 입력으로 나가는 rate는 원래 float32로 변환한다. fixed MN와 나머지 뉴런 및 adaptation은 원래 식을 유지한다. 별도 운동 위상·교사·기록 rate 재생은 없다.

부모 설계→기존 실제 gpt-6-sol/high 구현→FIRST 2결함→같은 Sol 첫 수정→SECOND 순서로 진행했다. 첫 검토에서 실제96근육 중 fixed-only6개 때문에 후보 입력을 거부하는 문제와, GPU 소유권 mapping 값 변화가 checkpoint identity에 드러나지 않는 문제를 고쳤다. 원래 근육 번호, 명시적 정밀도 계약, 생성 시 파라미터 fingerprint 및 실제 device mapping 검사가 포함됐다. 부모 CPU 테스트5개/Ruff 및 원래 후보의 metadata 검사가 통과했다.

RTX4080 실제 CUDA 생성 모델1회를 독립 CPU RK 계산과 대조했다. 8뉴런/4motor/3free/1fixed, 부호 있는 재귀 연결과2근육 그룹을 사용했다. 원본 BANC와 몸체는 실행하지 않았다.

- 전체 rate의 최대 차이7.450580596923828e-9, adaptation5.684341886080802e-14, 새 rate64 9.658496225029012e-12, fast기억3.677779337584591e-11, slow기억2.0375746852567198e-12로 모두 사전 허용오차 안이다.
- CUDA graph와 직접 stage 실행이 byte-exact했고 디스크 checkpoint를 새 인스턴스에 복원한 미래 상태도 byte-exact했다.
- 출력mask 차단은 다른 뉴런 rate를0.008290089666843414만큼 변화시켰다. 무입력 영 상태는 영 상태로 유지됐다.
- 누락기억·dtype/shape·NaN·음의rate·ID순서·beta·identity·full-rate 불일치·원래mask 오류의10개 checkpoint를 상태 변경 없이 거부했다.

생성 CUDA 과학적66tick + 생성자 warmup/capture12tick, 별도 CPU참조42tick이다. 생성 감독3.4858617999998387초, 내부2.858903200001805초, 부모 raw/hash 검산0.18007140001282096초, CPU참조준비0.05573210000875406초, 부모 순수검사/metadata1.3240799000050174초. 최대 단일 RSS1008693248bytes/관찰 합산1020219392bytes, Torch peak allocated81920bytes/reserved6291456bytes, 관찰 GPUcontext 증가56623104bytes, 출력45391bytes. 재시도·미종료 소유 프로세스·원본 실행·solver·physics는0이다. 준비·설계의 미측정 비용은 별도이며0으로 간주하지 않는다.

원본 누적5795.139958599688/6004초·잔여208.86004140031218초는 유지된다. 이 결과는 작은 재귀 모델에서 구현 정확성을 확인한 것이며, 전체 BANC의 자연 보행 성공을 뜻하지 않는다. 실제 몸체 검증 전 후보 채택0, E150최선/M2/UI보존, F6미노출, F1–F8 active·미완료다.
