# C 0.4.0 검증 자료 인덱스

기계 판독용 [JSON 인덱스](C_EVIDENCE_INDEX.json)는 각 진입 파일의 크기·SHA256·판정 범위를 기록한다. 상세 설명은 [개발 결과](C_IMPLEMENTATION_20260910.md)에 있다. 전체 파일 무결성은 ZIP의 `EVIDENCE_MANIFEST.json`과 외부 `.zip.index.json`으로 확인한다.

| 자료 | 판정 | 범위 |
|---|---|---|
| [최종 추출 소스 회귀](../verification/c_full_development_20260909/clean_checkout_tests_v3.log) | PASS | 205 tests, 0 skipped. Actual MPS tests included; not 205 physical campaigns. |
| [새 환경·최종 소스 재현](../verification/c_full_development_20260909/final_source_validation.json) | PASS | Clean extracted source, full snapshot compute, two actual compact campaign cases and resume. |
| [검사한 소스와 최종 비문서 파일](../verification/c_full_development_20260909/source_reproduction_identity.json) | PASS | 163 non-document source/dependency/data files matched; reports/checksums finalized afterwards. |
| [전뇌–몸 연결·복원](../verification/c_full_development_20260909/native_c1/report.json) | COMPLETE_WITH_LIMITATIONS | Eight short actual MPS/MuJoCo cases and direct motor contribution; no natural walking approval. |
| [오류 주입·복구](../verification/c_full_development_20260909/recovery_native_v2/report.json) | PASS | Actual MPS/MuJoCo recovery and diagnostic/recording boundary checks. |
| [추출 소스의 전체 스냅샷 판정](../verification/c_full_development_20260909/clean_checkout_full_snapshot_v3/report.json) | COMPLETE_WITH_LIMITATIONS | 139255 loaded nodes, independent data reference and actual MPS compute; physicalExecuted false. |
| [후각 전파·출력 진단](../verification/c_full_development_20260909/pathways_v2_attempt2/report.json) | DIAGNOSTIC | Sixteen fixed-input neural cases; natural odor forward output zero, not a physical trial. |
| [Poisson 입력 가설 진단](../verification/c_full_development_20260909/pathways_poisson_v1/report.json) | DIAGNOSTIC | Sixteen neural cases; h-current pulses differ from the cited voltage-input model; natural forward output zero. |
| [좌우 시각 입력](../verification/c_full_development_20260909/sensor_geometry_v1/report.json) | PASS | Actual independent MuJoCo geometries, input delay and disabled input; not retinal or avoidance validation. |
| [머리 접촉 입력](../verification/c_full_development_20260909/head_contact_v1.json) | PASS | Prescribed collision overlap only; no locomotion or bristle mechanics approval. |
| [몸 단독 5동작](../verification/c_full_development_20260909/body_motor_v3_evaluation.json) | PASS | Drive 1: forward/backward/yaw/stop-release; drive 3.3 failures retained separately. |
| [54조건 과제 해석](../verification/c_full_development_20260909/pilot_interpretation.json) | TECHNICAL_PASS_TASKS_NOT_APPROVED | 540 model seconds, frozen source. C_STRICT: 3 FAIL, 6 INCOMPLETE, 6 NOT_EVALUATED, 3 NOT_APPLICABLE. |
| [최종 캠페인 실행·재개](../verification/c_full_development_20260909/compact_native_v1/campaign.json) | COMPLETE | Two actual cases, 0.1 total model seconds. Detail-file hashes checked on completed-campaign resume. |
| [진행 저장량](../verification/c_full_development_20260909/campaign_progress_storage_v1/report.json) | PASS | Same 53 completed cases: 9897019 to 31292 bytes; no trace loss and no end-to-end speed claim. |
| [CPU/MPS·시간 간격 비교](../verification/c_full_development_20260909/numerics_v1/report.json) | BACKEND_PASS_DT_NOT_APPROVED | Full graph CPU/MPS fixed-input parity; finer dt changed selected spike counts. |
| [40회 비교 해석](../verification/c_full_development_20260909/performance_interpretation.json) | NO_CLEAR_THROUGHPUT_GAIN | Exact states/traces/recordings; median speed ratios 0.975 to 1.010 with overlapping distributions. |
| [MJX 실제 모델 적재](../verification/c_full_development_20260909/mjx_probe_v2/report.json) | BLOCKED | METAL JIT ran; unchanged model failed plane-mesh margin/gap support; no GPU physics executed. |
| [BANC→선택 tibia 운동](../verification/c_full_development_20260909/cns_tibia_v2/report.json) | PASS_LIMITED_SCOPE | 158706 declared neurons and six tibia DOFs; CPG bypass/actual actuation/restore; remaining DOFs and proprioception unbound. |
| [날개 물리·경로 독립 복원](../verification/c_full_development_20260909/flight_rig_v2/report.json) | PASS_LIMITED_SCOPE | Real ellipsoid fluid force and relocated assets restore. Stable/neural flight and takeoff/landing not validated. |
| [실제 화면·독립 작업](../verification/c_full_development_20260909/browser_validation_v1.json) | PASS | Observed Playwright actions and server readback; not an automated UI test count. |
| [사용자 실험 최종 보존](../verification/c_full_development_20260909/user_final_state_comparison.json) | PASS | 40.230 seconds; all checkpoint values exactly identical between post-upgrade and final delivery. |
| [실패/성공 종료 코드](../verification/c_full_development_20260909/motor_evaluator_exit_codes.json) | PASS | Re-scored retained real FAIL/PASS motor traces with exits 1/0. Initial wrong-argument probe retained. |

PASS가 붙은 진단이라도 자연 행동·생물학적 재현을 함께 승인하지 않는다. C_STRICT의 자연 전진은 실패했고 고정 180조건·120초 내구성 평가는 선행 과제 미통과로 실행하지 않았다. X1–X4는 설명된 연구 범위까지만 구현·검증됐다.

이전 테스트·외부 리뷰의 숫자는 이번 실행 횟수에 합산하지 않는다. 최종 회귀는 205개 한 차례의 결과이며, 실제 54조건 파일럿과 최종 코드의 두 짧은 조건은 서로 다른 실험이다.
