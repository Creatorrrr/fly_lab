# E329 시간 목적 구현·자격 결과

2026-10-01. **새 오프라인 시간 successor+여섯 다리 참조 관계 목적의 구현과 수치 자격은 통과했다. 실제 신경 가중치 갱신과 신체 개선은 아직 검증하지 않았다.** 전체 M3/F1–F8 목표는 active·미완료다.

현재 세션 설계 → actual gpt-6-astra/max 설계 리뷰 → root 반영 → actual gpt-6.1-sol/xhigh 구현 → root FIRST → 동일 Sol R1 → root SECOND를 실행했다. FIRST는 아주 작은 음수 phase가 modulo 계산에서1로 반올림되어 index128을 만드는 오류를 독립 검사로 확인했다. 동일 Sol이 Torch와 NumPy 경계를 수정했으며 SECOND에서 잔여 production 수정은 없었다. 최초 source·실패 기록을 보존했다.

## 구현과 독립 검증

- 새 `tools/banc_m3_phase_free_temporal.py`는 이전 명령에서 다음5ms 참조 자세를 구하는 successor 및 q_next의15개 다리쌍 관계를 학습한다. 중간 명령의 previous/next 양쪽 미분을 유지하며, 마지막 명령의 coordination 미분도 적용한다. 참조만 고정한다. 최종 controller에는 참조·phase·손실을 연결하지 않았다.
- 새 `tools/analyze_c_banc_m3_temporal_objective.py`는 원본·source hash를 모두 검사한 후 저장 command/control clock만 분석한다. 참조 위상은 명령 공간 proxy이며 생물학적 위상 관측이 아니다. 교사 보행 선호가 다른 좋은 보행을 배제할 수 있다.
- Sol 생성CPU34 tests/Ruff4files PASS. Root의 다른 다차원 곡선 및 독립 phasor 식/중앙차분에서14개 CPU 검사 PASS. 연결된4control 미분 최대오차7.2993e-11, 분리 previous2.6518e-11/next9.3671e-11이었다. all-previous detach 반례를 구별했다.
- RTX4080의 작은 생성 f64 값·미분 CPU/CUDA 비교17개 PASS, 추가 GPU39845888bytes(약38MiB), Torch peak reserved2097152bytes. 이것은 새 손실의 CUDA 자격이며 전체 BANC 성능이나 보행 성공의 증거가 아니다.

## 실제 참조와 저장 명령 분석

E85의 실제 기준곡선260표본에서 phase round-trip 최대오차5.5511e-17cycle, successor 최대오차4.4409e-16rad였다. 참조 순환 손실0,128anchor 정지 평균1, 역순3.93307, 비순환 순열160.99056이었다. 한 좌우쌍 전체를.125cycle 이동하면 LR 손실0이지만 all-pair손실.1562097이므로 앞·가운데·뒤 쌍 사이의 어긋남도 구별한다. 임의 공통 시작phase의 정상 순환도 통과했다. 각 phase에서 정지 손실이 모두1이라는 뜻은 아니며, 손실의 양수가 항상 비영 미분을 뜻하지도 않는다.

| 저장 명령 | 전체 새 손실 | successor | all-pair 관계 | LR 진단 |
|---|---:|---:|---:|---:|
| E318 중립 | 8.10350250 | 7.28178915 | .82171335 | .60934429 |
| E327 기각 후보 | 8.04273108 | 7.23297701 | .80975407 | .61479305 |

**신체에서 실패한 E327도 새 손실은 더 낮았다.** 따라서 새 손실을 신체 품질의 대리 합격 판정으로 사용하지 않는다. E327의 전진−2.82584%/slip+3.17646%/적격주기0 기각은 그대로다. 시간 순서와 관계를 구별하는 손실이 구현됐다는 사실과 그 손실을 줄이는 실제 BANC 방향이 신체를 개선한다는 사실은 별개다.

두 저장 입력의 loss에 쓰이는 control100..300에서 비국소 동률은 모두0이었다. 가장 작은 competing squared-distance gap은 중립 RM7.15865e-6rad²로 고정 동률한도1e-12보다 컸다. 다만 off-curve phase 전환은 있으며 중립 RM의 최대 원형 jump는.37375cycle다. 이 proxy를 실제 관절 phase로 승격하지 않는다. startup과 기존 shape도 원래 값으로 재현했고50창 평균은 전체200쌍 값과 일치했다. 독립 NumPy 원자료 재계산49검사 PASS로 수치·곡선 bytes·clock·phase/jump/비국소 동률·출처를 확인했다.

## 출처와 비용

최종 source hashes는 `parent-e329-final-source-review.json` 및 `e329-sol-r1-implementation.md`, 정확한 source/input 계약은 `e329-temporal-saved-contract.json`, 결과는 `e329-temporal-saved-characterization/result.json`, 독립 감사는 `parent-e329-saved-raw-audit.json`에 있다. 곡선 C-order SHA256는7439463c70f6b9f32f599baafda946a80e8160c7e13d2861b1f134531f96b505, 정상화 vector SHA256는c5324b972080cdaffc3e7e011e190fed78a94cb26107984e3e2934e18e0609d3이다.

Sol 생성CPU(실패포함)20.5628651/120초. Root FIRST 실패6.638300초와 SECOND6.650424초 및 저장 raw 감사.547373초를 보존한다. Root 생성CUDA2.163854/30초, saved characterization2.276338/20초·peak648376320bytes로 사전한도 안에 끝났다. 원본 신경/신체 실행은0이며 원본 누적9034.148870999521초와 기존9060초 단계상한은 변하지 않았다.

다음 E330 설계는 actual Astra6/max 리뷰를 반영했다. 자격된 목적의50×20ms surrogate VJP와 실제float32 크기비례 효능 방향을 비교하고, local descent/native 전달/연속1.5초/startup을 나누어 판단한다. E330가 성공하더라도 같은 theta의 실제 신체 E331 및10초 전체F1–F8 검증이 남는다. 후보채택·자연보행·생물학적동등성·F1–F8 PASS는 모두false다.
