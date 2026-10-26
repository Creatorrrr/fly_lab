# E223 사전 설계: 신경 내부 rate를 보존하는 전류 기반 운동 전달

E222는 0.2244초 upright 기준에서 기각됐다. 조건부 적합과 live 명령 차이는 0.3295rad까지 커졌고 전달 산술 검산은 통과했다. 다음 판별에서는 원래 신경간 발화·CSR 동역학을 유지하고, 각 MN의 live 흥분/억제/외부 입력과 인과 기억에서 근육 전달만 계산하는 별도 가설을 시험한다. 부모와 읽기 전용 gpt-6-sol/medium의 방향 검토는 `parent-e222-direction-review.json`에 기록했다.

이 출력은 원래 MN 발화율이 아닌 **병렬 전류 기반 effector 신호**다. E150의 원래 MN rate에 대한 monotone decoder와도 다르다. BANC 해부학적 노드·연결을 보존한 운동 변환 가설로 검토할 수 있지만, 생물학적 발화 기전이나 F2 통과를 이름 변경으로 주장하지 않는다. 최종 신경 구동 적격성·실제 건강/보행·감각 의존성·차단·저장 미래는 별도 실제 검증이 필요하다.

## 이번 변경과 한계

기존 실제 gpt-6-sol/high가 두 새 파일 `tools/banc_m3_neural_current_effector.py`, `tests/test_c_banc_neural_current_effector.py`만 작성한다. 부모 FIRST→같은 Sol 첫 수정→SECOND→부모 잔여 수정 순서를 유지한다. 새로운 방정식 적합이 아니라 고정 CUDA 반응의 출력 위치와 checkpoint를 바꾸는 통합 작업이므로 high를 유지한다. 기존 E221/E222 소스와 후보 계수는 수정하지 않는다.

새 `NeuralCurrentEffectorNetwork`는 E221의 수학·stage별 signed current·float64 반응 상태를 재사용하되, 원래 전체 float32 rate/adaptation와 CSR masked stage에 free 출력을 덮어쓰지 않는다. E221 native stage/finish가 기록하던 full-rate 및 masked destination을 별도의 고정 CUDA scratch로 전달하는 최소 변경을 우선 사용한다. 기존 `_step_fused`나 전역 kernel/source를 바꾸지 않는다. scratch는 파라미터/상태를 바꾸는 입력이 아니며 매번 계산 결과만 받는다. CUDA graph가 사용하는 주소는 고정한다.

일반 `readout`/`advance_readout`은 원래 신경 rate를 그대로 반환한다. 별도 명시적인 `effector_readout()`만 원래 MN 순서의 float32 벡터를 반환하며 free MN 자리는 병렬 float64 신호를 float32로 반올림하고 fixed MN 자리는 원래 rate를 사용한다. 이를 같은 canonical float64 근육 전달에 연결하는 body wrapper는 이번 단계에 포함하지 않는다. runtime 시간·phase·teacher·저장 출력 입력은 없다. 원래 막/적응·가중치·양방향 감각 경로는 보존한다.

새 schema, precision/routing 계약, identity에 원래 timescale identity와 계수/지도 provenance를 포함한다. checkpoint에는 전체 원래 신경 상태와 별도 effector/fast/slow 상태를 명시한다. effector와 원래 free MN rate가 다를 수 있으며 둘을 같은 상태인 것처럼 검증하지 않는다. E221 checkpoint를 이 모델로 조용히 재해석하거나 rate를 조작해 기존 restore 검사를 우회하지 않는다. 완전한 사전 검증 후 각 상태를 복원하고 잘못된 schema/계수/지도/shape/dtype/비유한 값은 live 상태 변경 전에 거부한다. reset은 모든 동역학 상태를 초기화한다.

## 판별과 검증

부모 source review 이후 **생성 실제 CUDA 프로세스 하나**만 실행한다. 첫 fixture는 E221의 고정 8뉴런 입력/가중치/계수다. 부모의 독립 CPU RK 계산은 free effector 값을 전체 neuronal rate에 주입하지 않는 별도 방정식으로 만든다. 같은 held drive와 output mask에서 원래 `MotorTimescaleNetwork`와 새 모델의 전체 rate/adaptation/readout이 bitwise 일치해야 한다. 동시에 병렬 effector는 원래 MN rate와 실제로 달라야 한다. 동일 입력에 원래 CSR를 보존했다는 검증과 effector 계산 검증을 구별한다.

추가 39뉴런 생성 fixture는 행당 32개를 넘는 연결로 warp 축약 경계를 검증한다. 원래 네트워크와 새 네트워크의 전체 상태가 같은지를 비교한다. 생성 데이터는 고정 seed로 사전에 저장하고 사용 파라미터/drive/mask를 해시한다. 8뉴런 effector 독립 오차 한계는 rate64/fast/slow 각 max 2e-6, 원래 float32 rate/adaptation CPU 비교 max 2e-6이며 GPU 원래 모델 대비는 byte-exact다. graph/direct, 새 인스턴스에 디스크 복원 후 미래 재현은 byte-exact여야 한다. 잘못된 체크포인트 거부, 영 입력의 영 출력과 차단에 대한 신호 의존성도 확인한다. 노드 수를 바꾼 생성 검증은 원래 BANC 보행이 아니다.

CPU/mock 테스트는 필요한 checkpoint/routing 의미만 검증한다. 실제 CUDA 전 source를 고정하며 Sol은 실제 CUDA·원본 graph·body·solver를 실행하지 않는다. 생성 검증 후 원래 E217/E218 파라미터의 정적 모양/provenance를 다시 확인할 수 있지만 모델 진행은 하지 않는다.

## 비용과 다음 결정

이번 원본 신경/신체/solver/fit 실행은 0, 생성 CUDA process 1/retry 0, 실제 scientific ticks 최대 200과 구성 warmup/capture 별도 계측이다. CPU1/GPU1, 내부45/외부50/감독60벽초, RSS2GiB/추가GPU1GiB/출력100MiB/free1GiB로 제한한다. 현재 원본 누적5818.539935299679/6004초·잔여185.460064700321초를 보존한다. 생성/개발/부모 검산 비용을 별도 계측하며 0으로 취급하지 않는다. 원본 배정을 증액하지 않는다.

검증 실패 시 원래 신경 동역학이 보존됐다고 주장하지 않는다. 통과하면 같은 E150 시작에서 `원래 rate+canonical`과 `별도 effector+canonical`의 최대3초 body 대조를 새로운 사전 설계로 정한다. E222의 실패와 최선 E150를 지우거나, 이 생성 검증을 F1–F8 완료로 대체하지 않는다. F6 미사용 자세·M2·활성 UI는 보존한다.
