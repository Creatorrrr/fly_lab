# E327: E326 단일 효능의 실제 신체·감각 되먹임 검증

E326은 고정 E318 감각 입력에서 연속1.5초 형태 손실을.163321% 낮추면서 기존 시작 네 제약을 통과했다. 이제 동일한 f32 theta 하나가 실제 감각·접촉·신체 폐루프에서도 시작 안정성과 기존 짧은 보행 기준을 만족하는지 단회 확인한다. 재학습·새 theta·grid·backtracking·기준완화는 없다. F1–F8/M2/UI/F6는 그대로 유지한다.

원본은 E318이 재현한 E307 전체 control0 체크포인트, 현재158706-neuron SSPRK33/u0 회로와 E150 monotone decoder, 같은 MuJoCo 신체다. E326 constrained-step.npz의 실제 theta 및 E29 mask를 현재 CSR에 f32(w0*f64(theta[group]))로 적용한다. 원본 CSR의 indptr/indices/모든 비선택 data와 선택 부호는 정확히 유지한다. 원래 SSPRecoveryResourceNetwork 클래스로 변경된 CSR의 새 네트워크를 생성하고, 모든 physiology/core/motor/resource u0 설정과 r/z/drive/mask/resource/tick을 유지한다. 예전 RK4 EfficacyTrial을 재사용하지 않는다. 변경된 가중치를 원래 model_hash로 표시하거나 기존 CUDA graph pointer의 값을 몰래 바꾸지 않는다.

새 SSPEfficacyTrial 계층은 원래 SSP trial과 효능 계층을 명확히 구분하는 별도 schema/identity를 둔다. 원래 체크포인트 metadata와 변경 네트워크 metadata, 선택 위치/원본값/실제값/그룹/theta/source identity를 보존한다. whole snapshot에는 신체·adapter·감각·rng·clock까지 전부 포함한다. source 원상표현에서 부모의 whole 복원 후 새 네트워크로 다시 구성하는 from_checkpoint를 제공해 생성 테스트로 roundtrip/변조 거부/실패 소유권을 확인하되, 이번 원본에서는 restore를 위해 두 번째 신체를 만들지 않는다. 원본 시작의 신경 metadata와 새 효능 계층을 제외한 모든 동적 값과 신체/adapter/감각 초기상태는 정확히 동일해야 한다. 다음 정식 검증에 whole restore/cut이 필요하므로 이번 성공만으로 F2/F8을 선언하지 않는다.

신체1개만 생성하고 control1..300/신경6000tick/물리15000step(1.5초)을 중간 reset 없이 실행한다. E318/E310의 검증된 native contact 관찰·clock·construction ledger·advance/readout·raw 저장·short evaluator를 그대로 재사용하되, 새 신체 결과를 중립 E318 저장 자료와 비교한다. 감각 입력은 새 실제 body observation에서 생성하며 E318 drive를 주입하지 않는다. 원래 ledger/통제계수에 드러나지 않는 setup step/미반환 interval이 있으면 무효다. 근육/관절 명령을 보간하거나 출력에서 직접 보정하지 않는다.

각 구간의 requested/returned/confirmed 신경·물리 비용, control-journal, native-contact, control-raw301, activity/recruitment, trace, observed-start/final whole, 실제 patched CSR 또는 그 재현 가능 roster/hash, source/input/output manifest를 보존한다. 부분 실패/health stop도 raw 및 최초 오류를 보존한다. upright<.9 또는 nonfoot 발생 시 기존 health stop으로 종료하며 성공으로 바꾸지 않는다. CUDA meter는 Torch CUDA를 먼저 초기화하는 E323 경로로 유지한다.

최종 분류에는 (1)공학적으로 유효한 원본 실행, (2)E103 startup q RMS≤.018/max≤.04 및 tendon RMS≤.009/max≤.025, (3)실제1.5초 shape와 저장 중립의 차이, (4)기존 [.5,1.5) forward≥1.6790085779712631mm, slip≤1.8026436968249955, bout≥.3초, 여섯 다리 eligible cycles>0을 각각 기록한다. 개선 또는 채택은 기준 전체가 통과한 경우에만 별도 후보 검토로 남기며 생물학적 보행/F1–F8 완료는 false다. 기존 중립의 동일 evaluator 결과도 저장자료만으로 재계산해 evaluator 불일치를 차단한다. 기준 일부만 좋아져도 전체 성공으로 섞지 않는다.

현재 원본 누적8976.486234799555/9060초, 잔여83.513765200445초다. 자격 통과 뒤 supervisor80/internal70/science55/finalization15초 단회만 예약한다. CPU1/host6GiB/추가GPU4GiB/output1GiB/free disk4GiB. 원본 재시도0, 한도 증가0. 감독 종료 후 실제 사용량만 한번 합산한다. 초기 구성에 너무 오래 걸리는 등55초 science 안에 완료할 수 없으면 부분 실패로 보존한다.

Actual gpt-6-sol/xhigh가 새 `tools/banc_m3_ssp_efficacy_trial.py`, `tools/verify_c_banc_m3_constrained_secant_body.py`와 대응 tests 두 파일만 구현한다. Root FIRST→필요시 같은 Sol R1→root SECOND→잔여수정root 순서다. 기존 E318/E326 및 모든 source는 불변이다. Sol/root 생성 CPU각90초, root 생성 CUDA30초/host2GiB/추가GPU1GiB. 생성 작은 SSP 네트워크에서 neutral theta 정확성, f32 CSR·부호/비선택 유지, 동적 상태 전달·snapshot/restore, 기존 grouped-current adjoint의 forward와 변경 theta 실제 native one-control20tick 일치를 확인한다. 신체 fake 경로로 clock/부분 실패/최초 오류/후반 게시 실패를 검증한다. 원본 payload·CUDA·신체는 root 자격 완료 이전에 실행하지 않는다.
