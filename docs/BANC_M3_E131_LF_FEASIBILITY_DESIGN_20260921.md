# E131: LF 초기 자세 제약 복원

2026-09-21 · H131/M3 · 부모 사전 설계 · F1–F8 미완료

E128은 원래 보행 형태 목적과 초기99 제약을 함께 최적화했지만 여섯 다리 모두 제한 시간 내 feasible 기록이 없었다. LF best-objective 기록13은 최악 squared inner ratio1.0874722로 여섯 best-objective 중 가장 컸다. E130은 같은 비영점 점의 응답/미분을 정확히 유지하면서 전체 평가 중앙1.896배를 확보했다. H131은 이 점에서 **초기 제약 위반을 우선 줄이는 국소 적합**으로 원래 LF selector 기준에 도달할 수 있다는 제한된 주장이다. 경쟁 설명은 고정 delta/gain bounds 내 제약 충족이 어렵거나 국소 곡률/정류 경계와 남은 평가 수가 방해한다는 것이다. 실패는 전역 불가능성의 증거가 아니다.

## 고정 입력과 개입

입력은 E128 LF 기록13과 E130에서 검증한 동일 fixture/기하/69MN/17muscle/273edge/3contexts다. 기존 E128/E129/E130 소스와 결과를 수정하지 않는다. 기존 부모/같은 Sol 리뷰 절차를 완료한 E130 `precomputed_expm1` adapter와 E128 `evaluate_leg`, `constraint_values`, `selector_feasible`를 재사용한다. 새 CUDA 커널·ODE·decoder·연결·gain bounds·초기99·INNER/ORIGINAL·selector/optimizer cap 변경은 없다.

delta/gain 시작점은 LF 기록13의 273+17 float64 값이며 영점으로 되돌리지 않는다. optimizer Hessian/state는 저장되지 않았으므로 기존 optimizer의 완전한 재개가 아니라 **기록13에서의 후속 국소 적합**이다. delta bounds는 고정 fixture, gain은[0,8]을 그대로 쓴다. E128 best-objective를 본 개발용 선택이며 독립 확인 데이터가 아니다.

매 원본 평가에서 고정 `evaluate_leg(...leg=0,index=1)`을 별도 `evaluation-001/` 등의 폴더로 실행하여 여덟 응답·branch·초기99/B/오차·loss/미분/기하 원자료를 먼저 저장한다. outer evaluation index와 원래 raw index1을 구분한다. 첫 새 원자료는 E128 기록13의 공통 필드 및 E130 첫 candidate 전체 응답/분기와 E130 허용오차로 비교한다. 두 비교가 통과하지 않으면 최적화를 시작하지 않는다. 새 입력 warmup은0, 별도 E128 synthetic tiny1MN/20ticks/.005조건부초/1bank만 한 번 허용한다.

## 제약 복원 목적과 종료

E128의 정확한4761 inequality/Jacobian을 계산하되 보행 loss epigraph인 첫3행은 복원 목적에서 제외한다. 나머지4758 초기99 제약의 값 c와 Jacobian J를 그대로 사용한다. v=min(c,0), 복원 목적 L=.5 sum(v²), 미분 J[:, :-1]^T v다. 마지막 epigraph 열은 최적화 변수에서 제외한다. 원래 optimizer_cap=(1-2e-4)²을 사용하여 selector_cap=(1-1e-4)²보다 엄격한 내부 여유를 보존한다. L-BFGS-B, maxiter100, maxls20, ftol1e-12, gtol1e-10을 사전 고정한다. 동일 delta/gain byte key는 캐시만 조회하고 새 응답으로 세지 않는다.

첫 시작 평가도 unique 수와 비용에 포함한다. 원자료·요청/완료·해시·진단을 기록한 다음 `selector_feasible(ratios)`가 처음 참이 되면 즉시 중단한다. 목적값만 작거나 scipy.success만 참인 것을 통과로 간주하지 않는다. 제약을 통과한 기록의 delta/gain는 후속 후보 제안으로만 저장한다. 이 실험에서 전체6다리 결합/형태20%/실제BANC/신체/F6/제품 반영은 하지 않는다. 원래 전체 채택 기준을 생략하거나 대체하지 않는다.

원래 LF 기록13 대비 새 feasible 기록이 존재하면 H131 supported와 LF 복원 방법의 제한적 연구 채택으로 기록한다. 기록이 없으면 유효한 bounded failure/inconclusive, gait adoption reject다. 미완료/오류/입력 불일치는 invalid 또는 completed subtest에만 한정한다. 모든 평가와 실패를 보존하며 최선 위반값 감소만으로 natural gait나 기존 M3 통과를 주장하지 않는다.

## 실행·자원·보존

실제 RTX4080/CUDA, CPU thread1, LF engine 및 Torch geometry 각 한 번 생성. E129 관찰 래퍼는 각 평가의 정확한 phase, solve/transfer/geometry 완료를 기록한다. E130 adapter는 평가별 새 scope를 사용하여 준비/전진 events와 임시273 double 비용을 구분한다. 원본 unique 최대16, 조건부480초/1920000contextticks/132480000scalar MNticks/2096640000edge-stage ticks/16banks/96000decoder끝점/기하 forward·backward 각각87568행이 상한이다. 실제 완료·불명확한 완료는 별도 계상한다.

fit wall20초, 새 원본 호출 전 fit/내부 deadline에2초 여유를 둔다. 전체 내부27초/외부32초/supervisor35초, 원본 프로세스1회/retry0이다. 전체 deadline이 먼저 도달하면 중단하며 남은 unique를 채우지 않는다. 이전 실제 보수적누적1427.7245423000422초를1428로 올려35초를 배정한다. 배정 종료1463<=공유1530초이며 기존 테스트·실패·원본 비용을 초기화하지 않는다. 부모 독립 원자료 검토는 신경/기하를 추가 실행하지 않는다.

모든 raw·ledger·source/environment·partial/error·manifest를 보존한다. 실패한 구성 업로드, solve/기하 실패와 cleanup의 2차 오류를 기록하되 원래 예외를 유지한다. 요청·enqueue·sync·unknown의 계상은 기존 검증된 관찰 도구를 재사용한다. finalization/manifest/벽시계 한도는 성공 반환 전에 확인한다.

## 소유권과 검증

부모 설계→**gpt-5.6-sol/high** 구현→부모 첫 리뷰→같은 Sol 첫 수정→부모 두 번째 리뷰→잔여 부모 직접 수정이다. 새 신경 수학이나 CUDA 미분을 추가하지 않는 범위의 최적화/계상 코드이므로 high로 설정한다. Sol은 `tools/fit_c_banc_m3_lf_feasibility.py`, `tests/test_c_banc_lf_feasibility.py`만 구현한다. 부모는 freeze·supervisor·독립 원자료 검토를 담당한다.

생성 입력에서 활성 signed/RMS 제약과 음/양 방향의 복원 미분을 유한차분으로 확인하고, known-feasible 복원·bounds·cache·최초 feasible 중단·deadline/unique cap·예외와 partial 보존을 검증한다. 테스트는 기존 원본/전체 BANC/신체/F6를 실행하지 않는다. 실제 synthetic neural 호출이 필요하면 사전 설명하고 별도 비용을 남긴다; CPU 생성 제약 테스트를 기본으로 한다.

CLI는 `--freeze --freeze-sha --source-review --source-review-sha --out`이다. 부모의 정확한 두 소스 해시와 `accepted_for_single_original_lf_feasibility=true`, `blocking_defects=[]`가 필요하다. freeze의 E130 source review를 통해 기존 입력/보존 해시를 재검증하며 최신 live source와 일치해야 한다. 첫 부모 최종 source 검토 전 원본 실행은0이다.
