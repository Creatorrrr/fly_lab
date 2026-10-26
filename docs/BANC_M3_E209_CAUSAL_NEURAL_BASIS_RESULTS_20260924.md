# E209 결과: 인과적 신경 반응 기저의 CUDA 계산 검증

E208에서 저장한 실제 E/I/drive 160000×391 입력으로 16ms 입력 기억과 정규화된 국소 반응 기저 16개를 계산했다. history와 basis 두 CUDA launch만 실행했으며, 학습·전체 BANC·신체 실행은 0이다. 기저 계수는 아직 학습하거나 채택하지 않았다.

원래 초기 발화율과 membrane tau/cap를 유지하고, 반응은 실제 국소 입력과 기억으로만 구성했다. 자유343MN의 adaptation feedback 대체 가설과 canonical 근육 전달 변경을 명시했다. 이전 반응식이나 생물학적 적응 기전을 보존했다는 주장이 아니다.

원본 계산에서 두 kernel의 기억/descriptor bytes가 같았고 B 비음수·mass 상한이 통과했다. 독립 적분한 probe와 기저 혼합의 최대 발화 차이는 1.98951966012828e-13, affine 근육 명령과 직접 decoder의 최대 차이는 4.44089209850063e-15였다. 부모는 별도의 RK4 다항식과 선형 필터로 사전 고정한6MN의 전체160000 stage를 계산했다. 발화 최대 오차는3.553e-13 이내, 모든 MN의 D 재구성 오차는 5.6843418860808e-14였다. 모든 prototype/폭/scale/소유권/마스크를 별도 재검산했다.

고정48MN과6LTM의 원래/새 canonical activation 및 tendon은 exact였다. 원래 native LTM activation·tendon과도 최대 차이가0이었다. 입력 채널3개는 각각 한 번 device upload했고 전체 조건부 neural tick은80000, source stage 소비는320000이다. 전체 BANC tick이나 물리 step으로 세지 않는다.

전달 방식만 바꾼 대조에서 원래 E150 신경 출력의 command RMS는0.07162502844595357이었다. 원래 monotone 기준0.056686385349786386보다 크며 startup max0.040311391073225344도 기존 .04를 넘는다. 따라서 canonical 변경 자체를 개선으로 채택하지 않는다. 이후 학습에서는 신경 출력/decoder 2×2 대조와 원래 proxy 기준을 유지한다.

실제 gpt-6-sol/high 구현→부모 FIRST→같은 Sol의 R1–R4 수정→부모 SECOND 순서로 진행했다. 부모 잔여 수정은 생성 CUDA의 sampling interval을1로 맞추고 계획된 device 할당과 실행 간격을 고정한 것이다. 순수/모의10개, 작은 실제 CUDA1개(2launch·8tick·warmup0), Ruff가 통과했다. 실패 raw 보존과 계산 후 시간 제한도 생성 시험으로 확인했다.

원본 감독 13.440542800003초, 누적 5516.421354399707/5624초, 잔여 107.578645600293초다. CUDA history/basis 실행과 readback은 각각 1.131236600/6.317169100초, 실제 추가 CuPy pool 최고 926301184bytes, runner peak RSS 2637840384bytes, 출력 352251255bytes다. 원본1/retry0/fit0, 잔류 owned process0. 준비·생성·부모 검산 비용은 별도 기록했다.

다음 단계는 동결된 기저의 beta 5488개를 조건부 제약 아래 학습하는 것이다. 검증 성공은 계산 구현에 한정되며 새 보행 후보나 자연스러운 보행의 성공이 아니다. F1–F8 active·미완료, 최선E150/M2/UI/F6와 이전 실패를 보존한다.

근거: `verification/banc-natural-gait-m3-20260919/e209-causal-neural-basis/`, `parent-e209-source-review.json`, `parent-e209-output-review.json`, `parent-e209-generated-receipt.json`, `parent-e209-supervisor.json`.
