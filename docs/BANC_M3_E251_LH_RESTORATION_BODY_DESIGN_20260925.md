# E251 LH 연결 복원의 신체 폐루프 비교 — 2026-09-25

M3 finite-goal의 고정 F1–F8을 향한 최소 신체 진단이다. E250에서 25개 기존 LH 감각→bridge 간선 복원은 200ms 신경 반응 기준을 통과했다. 그 시작점은 교사 궤적 끝의 신경 상태였고 신체는 없었다. 실제 감각 입력을 되먹임하는 신체에서 도움이 되는지는 별도 문제다. 이번에는 E150의 노출된 control0 전체 상태를 사용한다. F6 초기 자세는 사용하지 않는다.

## 고정 대조와 실행

baseline과 candidate를 순서대로 각각 최대300controls, 1.5초 실행한다. 둘 다 `e150-native-monotone-body/candidate-neutral`의 동일한 전체 상태에서 시작한다. baseline은 원래 전체 CSR, candidate는 E250의 기존25개 data entry만 원래 rate 값으로 복원한 CSR이다. E250 동결 간선/원래 접촉수·부호·float32값을 그대로 사용한다. 추가 간선·세포·gain 탐색·fit·solver·교사·위상·root 보정은 없다.

원래 MonotoneRecruitmentTrial의 body/adapter/parameters/navigation/tonic/seed/모든 동적 상태를 보존한다. 같은 소유권 이전 방식의 LH 전용 wrapper를 두 팔 모두에 사용한다. baseline은 identity patch이며 모든 network identity도 동일해야 한다. candidate는 새 생성자의 model_hash 및 core_adaptation.base_model_hash, motor_adaptation.source_core_identity, motor_timescale.source_motor_identity 네 항만 바꾼다. 그 외 전체 상태는 exact다. 기존 RM-load wrapper의 이름이나 schema를 LH로 위장하지 않는다. E177의 패치 검사는 generic same-sign이며 5개 고정 제약은 없다. 재사용 가능한 것은 상태 이전 원리다.

전체158706뉴런, 실제 CUDA packed CSR/capture20/dt.00025 및 CPU MuJoCo를 유지한다. actual packed rowptr/encoded indices/decoded weights를 전체 CSR에 대조한다. 매 control 원래 encode→20 neural ticks→decode→50 physics steps를 실행한다. 감각 입력을 기록할 뿐 대체하지 않는다. navigation.cut이 있어도 원래 tonic drive가 남는다는 사실을 보존한다.

baseline의301개 control trace와 recruitment,15000개 native contact를 E150의 같은 prefix에 exact 대조한다. baseline 재현 실패는 비교 invalid이며 candidate를 실행하지 않는다. candidate에서 native 건강성 중단이 발생하면 실제 완료 prefix/실패 control/부분 신경·물리 비용을 저장한다. 측정된 물리적 실패는 유효한 부정 결과일 수 있지만 저장·계측·시계·구성 오류는 invalid/not_tested다. baseline과 candidate 이외의 새 whole restore/arm/retry는 없다. 후보의 whole disk future는 이번에 실행하지 않으며 F2 통과로 표시하지 않는다.

## H251와 다음 결정

경쟁 설명은 작은 국소 신경 반응 개선이 초기 전신 동역학에 충분하지 않거나 기존 균형·협응을 해치는 것이다. H251은 아래 조건을 모두 충족할 때만 짧은 폐루프 개선으로 지지한다.

1. 두 팔 모두301개 control/15000개 native를 완료하고 native upright≥.9, 비발 접촉0, 수치·body fault0, 시계 정합성을 만족한다.
2. 후보는 기존 strict E141 startup99를 통과한다. 원래 E103 target에 대해 q RMS≤.018rad/max≤.04rad, LTM RMS≤.009/max≤.025이다. 기준 변경은 없다.
3. 동일 [.5,1.5)초에서 원래 초기 heading에 대한 후보 전진 거리가 baseline 이상이다.
4. 동일 [.5,1.5)초에서 load≥.02 BW인 유효 native 접촉의 하중 가중 미끄러짐이 baseline 이하이다. 선택된 하중에서 missing/nonfinite slip이 있으면 측정 불가로 두며 통과시키지 않는다.
5. 변경하지 않은 biological gait evaluator의 후보 longest_forward_bout_s가 .3초 이상이고 baseline 이상이다.

전진 거리는 기존 F3와 같은 평면 좌표/초기 heading/마지막 포함 표본의 차이이며 구간만 위의1초로 명시한다. 미끄러짐은 기존 supported_slip의 가중식과 같은 정의로 계산하되 `short_window_slip`으로 구분한다. 원래10초 evaluator/body_quality를 수정하거나 짧은 결과를 F3–F5/F7 PASS로 표시하지 않는다. 여섯 다리 관절 운동폭, eligible cycle/phase/period, 감각23→bridge4→MN391 변화도 원자료와 함께 기술하되 사후에 H251 기준을 바꾸지 않는다.

유효한 H251 지지일 때만 같은 후보의 전체10초 보행 및 저장 복원·차단 검증을 다음으로 설계한다. 건강성/strict startup/전진/미끄러짐/연속 전진 중 어느 하나가 미달하면 이번25개 일괄 복원은 보행 후보로 보류하고, 기록된 실제 감각→내부→MN/근육 응답과 최초 신체 차이를 먼저 분석한다. 배율 sweep이나 나머지 다리 일괄 복원은 자동 후속이 아니다. invalid면 원본과 비용을 보존하고 해당 구현·계측 오류만 구분하여 복구 설계를 한다.

## 구현과 원자료

부모 설계 후 실제 gpt-6-sol/high가 `tools/verify_c_banc_m3_lh_restoration_body.py`와 `tests/test_c_banc_lh_restoration_body.py`만 작성한다. 부모 FIRST→필요시 같은Sol 첫수정→부모SECOND→남은 수정은 부모 순서다. 기존 StateStore/MonotoneRecruitmentTrial/advance_control/ControlJournal/AdvanceLedger/profiled_construction/native observer/capture_control/strict_startup/gait 분석을 활용한다. 2000controls와 clone이 고정된 원래 full-body runner는 축약·변조하지 않는다.

출력은 `verification/banc-natural-gait-m3-20260919/e251-lh-restoration-body/`다. 각 팔의 start/final whole checkpoint, control trace, native-contact, recruitment state, 실제 sensor 특징/필터 및 반환 drive/선택 source·bridge 발화, applied joint/tendon/adhesion, clock/control/construction/cleanup ledger, CSR 및 identity 증거를 보존한다. 관측 부분 실패도 availability와 오류를 남긴다. 최종 판정은 raw 저장/manifest/cleanup 뒤에만 확정하며 늦은 실패 후 성공 결과가 디스크에 남지 않게 한다.

생성 fixture로 단일 CSR 변경/identity baseline·candidate, 전체 상태 전달·소유권, 부분 physics/constructor 계측, 짧은 구간 판정, native 건강성 실패와 저장 실패 구분을 확인한다. 원본 전 source/input/package/hash freeze 및 부모 검토가 필요하다. 부모는 원자료에서 독립적으로 CSR/state/prefix/parity/metrics/실제 실행량을 검산한다. 재사용 소스와 frozen 과거 파일은 수정하지 않는다.

## 자원과 보존

단일 원본1/retry0, CPU1/GPU1, 과학600controls/12000neural ticks/30000physics steps, 각3.0모델초 상한이다. whole restore2에서 예상 underlying network constructor14, 실행 setup560ticks와 capture기록280ticks를 별도로 실제 계측한다. FlyGymBody 생성자마다 중립 settling2000physics가 있으므로 setup physics4000도 별도 계측하며 실패한 일부 호출을0으로 만들지 않는다. 생성/restore 중에는 scientific advance를 하지 않는다.

E193의3초 관측87.7611348초, E195의3초 신체 진단114.3912537초 감독 비용을 참고한다. 두 팔의 생성·snapshot·저장 비용을 포함하여 internal140초(마감15초 포함)/external150/supervisor160초를 배정한다. 기존 누적6772.265502999628초는 보존하고 이번 미래 배정만6800→6933초(+133)로 늘린다. 잔여160.734497000372초에서160초를 예약한다. 이는 승인된 개발의 단계별 로컬 실행 배정이며 사용자 전체 시간·토큰 예산을 새로 설정하지 않는다.

host6GiB/추가GPU6GiB/output768MiB/free disk2GiB/free GPU7GiB, 원본당300초 제한을 유지한다. 준비/생성검사/부모검산 비용은 별도, 측정하지 않은 개발·자문 비용은null이다. E250까지의 실패/비용/원자료, E150 best partial, M2/default/UI를 보존한다. F1–F8은 동일 최종 후보로 충족해야 하며 현재 미완료다. goal API는 현재 paused를 반환하지만 사용자의 계속 요청에 따라 승인 범위의 작업을 이어가고, 없는 resume 기능을 성공했다고 기록하지 않는다.
