# 다음 판별 질문 초안: 다리 사이의 명령 협응

2026-10-01 ROOT 독립 설계 초안. 아직 새 실험이나 구현을 시작하지 않았다. 현재 E338 R1과 ROOT SECOND/잔여 수정이 끝난 뒤, 실제 Astra/max 리뷰와 ROOT 반영을 거쳐 다음 단계로 확정할 후보 질문이다. 현재 finite-goal/F1–F8과 E337/E338의 기술적 실패, 비용, 원본 재시도0, M2/UI/F6 미사용 상태를 보존한다.

E336의 저장 자료에서는 여섯 다리 모두1초 동안 global-reference 명령 위상이2.0174–2.4811회 진행하고, 실제 관절은 명령을 RMS.001096–.001979rad로 추적한다. 그런데 E335의 최장 전진 bout은.21초이며 엄격 eligible cycle은0이다. 위상이 전혀 진행하지 않거나 서보가 명령을 전달하지 못한다는 설명만으로는 이 결과를 설명하기 어렵다. E336은 다리별 위상 진행과 tracking을 검산했으나, global-reference 좌표에서 다리 사이의 관계가 reference와 얼마나 다른지 직접 요약하지 않았다. 이 누락은 협응 학습을 설계하기 전에 판별할 가치가 있다.

현재 확인한 FlyGym demo의 `cpg_controller.py:calculate_ddt`는 각 다리 oscillator의 위상 차이와 phase bias로 coupling을 계산한다. `hybrid_controller.py`는 tarsus/thorax 높이와 contact force를 감각 관측으로 사용한다. 이는 다리 사이의 협응과 감각 보정이 보행 제어에서 별도 역할을 한다는 구현상의 참고다. 이 CPG나 preprogrammed curve를 최종 BANC 런타임에 연결하는 제안은 아니다. 최종 보행은 실제 BANC 신경 출력만 사용해야 한다.

다음 후보 질문은 **현재 neural-derived 관절 명령의 global-reference 위상 관계가, 같은 reference encoding을 거친 협응된 곡선의 관계에서 벗어나는가**다. 기존 E337의 per-leg shape body counterfactual을 재실행하는 질문이 아니며, E338의 실패한5 archive metadata CLI를 새 이름으로 재시도하지 않는다. 새 질문은 이미 저장된 명령의 다리 사이 관계를 판별하여 다음 신경 학습에 협응 항을 추가할 근거가 있는지 결정하는 저장 진단이다. 실행 오류는 협응 가설의 기각이 아니다.

입력 후보는 E336의 `neutral-smooth.npz`, `held-smooth.npz`, `live-smooth.npz`에 있는 q/z/R와 E332의 고정 curves/temperature/phase_basis다. 각 파일의 producer manifest/file SHA, header의 실제 전체 관측과 선택 필드 정책을 새 사전 계약에 동결한다. E338 R1의 완전 검사/명시적 historical-selected 의미를 활용할 수 있지만, 생성 전용 decoder를 원본 읽기 API로 바꾸지 않는다. 원본 값 선택 reader가 필요하면 새 도구와 별도 명시적 계약에만 작성한다. 신경 raw drive/StateStore/GraphStore/전체 신경 상태, body/physics/CUDA/F6 읽기와 실행은 필요하지 않다.

진단은 각 행/다리에 대해 기존 z를 독립 NumPy soft encoder로 재계산해 저장 z/R와 먼저 대조한다. `R>=.5`인 값만 유효하며 epsilon/phase 채움/역진 삭제/임의 offset 보정을 하지 않는다. 유효한 unit circular moment `u=z/|z|`로 세 좌우 쌍과 전체15쌍의 원형 평균 bias/집중도, 시간별6다리 coherence를 기록한다. C는 anatomical offsets가 이미 들어간 global-reference 곡선이므로 offset을 두 번 더하지 않는다. 같은 encoder를 `q_l=C_l(phi)`라는 모든 다리의 공통 phi128개 grid에 적용한 합성 reference를 비교하여 encoder 자체의 모호성과 leg별 형상 편향을 분리한다. 이 합성 reference는 원본 교사 rollout이나 실제 신체 보행으로 부르지 않는다.

확정 전 Astra 리뷰에서 유효 coverage, 원형 평균이 정의되지 않을 때의 null 규칙, 선택 필드, 수치 tolerance, 제한된 비교와 다음 결정 기준을 구체화해야 한다. 좌우 관계의 차이를 최종 발걸음 F4 수치로 환산하지 않으며, 상관/위상 proxy를 인과 관계나 생물학적 기전으로 해석하지 않는다. 세 저장 trajectory는 독립 개체/반복이 아니다. 비교가 큰 불일치를 보이면 다음 후보는 reference-relative inter-leg loss를 학습에만 사용하는 실제 BANC 효능 교정이다. 불일치가 작거나 proxy가 모호하면 접촉·stance 추진/명령 형상 방향을 우선 검토하고, 단순히 위상 항의 gain만 반복하지 않는다.

자원 후보는 새 saved analysis 한 번 supervisor15초/internal12초, 생성검사 ROOT/Sol 각 기존 기록과 구분한 사전 상한, CPU1/GPUhidden/host2GiB/선택 값만 decode다. 실제 원본 numerical cumulative9565.21586319952/9660초의 잔여94.78413680047925초 안에서 예약할 수 있는지 원본 실행 직전에 재확인하고 실패를 포함하여 한 번 계상한다. 이 초안은 예약0이며 어떤 비용·기준도 변경하지 않는다. 고정 가설/대조/validity/decision/bounds가 확정되지 않아 현재는 실행 권한이 없다.

근거: `docs/BANC_M3_E336_SAVED_GAIT_TRANSFER_RESULTS_20261001.md`, `tools/analyze_c_banc_m3_saved_gait_transfer.py:compute`(C의 global-reference 의미), `tools/banc_m3_smooth_temporal.py`(동일 soft encoder의 상수), 로컬 FlyGym demo source `C:/Users/ckthd/Dev/Projects/flygym/src/flygym_demo/complex_terrain/cpg_controller.py`와 `hybrid_controller.py`. 현재 질문은 생물학적 자연 보행의 통과가 아니라 다음 neural-only 구현을 정하는 좁은 진단이다.
