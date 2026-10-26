# E132: 초기 제약을 충족한 LF에서 보행 목적 재최적화

2026-09-21 · H132/M3 · 원본 실행 전 부모 설계 · F1–F8 미완료

E130의 exact CUDA 가속과 E131의 LF 초기99 복원이 유효했다. 그러나 복원 후 최악 보행fit loss는1.9383% 증가했고, E128 영점 LF 대비 저장된 normalized foot/velocity RMS 성분은 약1%/.3%만 감소했다. 이 normalized 성분 비교는 전체 shape20% screen을 대체하지 않는다. 다른 다리로 제약 복원만 확대하기 전에, 제약을 만족한 LF 점에서 원래 보행 목적을 더 낮출 수 있는지 검사한다.

H132는 **이미 feasible한 LF에서 시작하고 검증된 빠른 CUDA를 사용하면, 고정 초기99를 유지하면서 원래 최악 loss+prior를1% 이상 줄일 수 있다**는 국소 주장이다. 경쟁 설명은 원래 delta/gain bounds와 초기 제약이 목적 개선을 제한하거나, 남은 예산에서 SLSQP가 충분히 진행하지 못한다는 것이다. 유효한 미달은 bounded inconclusive이며 전역 불가능성을 뜻하지 않는다.

## 입력과 변경

시작점은 E131 evaluation011/selected-lf-proposal의273delta+17gain float64 값으로 고정한다. E128의 기존391MN/96근육/69source/2444union/1572fit 연결 전체 구조, LF69MN/17근육/273edge, 세 문맥/160000native stages/2001끝점, 고정 기하/target/fit masks/초기99/bounds와 기존 cap을 유지한다. E128/E129/E130/E131의 소스 및 결과는 수정하지 않는다. optimizer Hessian을 복원하는 것이 아니라 저장된 feasible vector에서 새 국소 SLSQP를 시작한다.

E128의 `EvaluationCache`, `objective_and_jac`, `constraint_values`, `solve_one_leg`, `choose_records`를 재사용한다. delta/gain뿐 아니라 epigraph t가 있는 기존 변수 정의와4761 부등식, 정확한 미분, 원래 regularizer를 그대로 사용한다. 초기 t=max(losses)+.01이다. E131의 제약 복원 목적을 추가 최적화하거나 cap을 완화하지 않는다. E130 adapter와 E129 계측을 평가별로 사용한다. 새 신경 수학·CUDA 미분·CPG·time gate·교사 입력·신체 보정을 추가하지 않는다.

첫 원본 호출부터 unique 수에 포함한다. 모든 평가를 별도 evaluation-NNN 폴더에서 `evaluate_leg(...leg=0,index=1)`로 실행하여8응답/branch/초기99/B/loss/오차/미분 원자료를 보존한다. 최초 응답은 E131 선택 raw의 전체 저장 필드와 응답/미분1e-8,loss/parts/ratios1e-10,branch/ID/delta/gain exact로 비교한다. 최초 실제 selector feasibility도 다시 확인한다. 어느 쪽이든 실패하면 optimizer를 요청하지 않는다. 별도 synthetic tiny1MN/20ticks/.005조건부초/1bank만 한 번 실행하며 원본 warmup은 없다.

## 판정

E131과 달리 첫 feasible에서 멈추지 않는다. 정상 SciPy 종료 또는 사전 unique/wall 한도까지 원래 목적을 최적화한다. solver.success와 무관하게 실제 초기99 selector cap을 만족한 저장 기록 가운데 true worst objective(max(losses)+기존prior)가 가장 작은 기록을 선택하고 동률이면 더 이른 index를 선택한다. 첫 start도 후보에 포함하므로 새 점이 나쁘면 시작점을 유지한다. 모든 실패/비feasible/line search 기록을 보존한다.

새 LF proposal의 연구 채택 조건은 유효한 최초 재현,모든 요청·완료·원자료·비용 검증,원래 selector feasibility,그리고 선택 true worst objective <=.99*start true worst objective다. 유효하지만1%에 못 미치면 이 후속 proposal을 채택하지 않는다. 이는 국소 진전 판단이며 원래 전체 shape20%/q5%/초기99 및 M3 F1–F8을 대체하지 않는다. E128 영점·E131 시작·선택점의3문맥별 q/foot/velocity/LTM normalized loss 성분을 모두 기록하고 악화도 보고한다.

선택 proposal은 float64 solver 값이다. 여섯 다리 결합,실제float32 calibration/gain 반영과 전체 conditional screen,독립 NumPy 참조,실제 BANC/body/F6/목표도달 검증은 아직 수행하지 않는다. 전체 보행 후보 및 제품 채택은 항상 false다.

## 실행 한도와 계상

실제 RTX4080/CUDA, CPU thread1, LF engine/기하 각 한 번 생성한다. 최대40 unique(최초 재현 포함),기존 SLSQP maxiter100/ftol1e-9,fit55초,전체 내부64초/외부69초/supervisor73초,원본 프로세스1회/retry0이다. 기존 EvaluationCache의 leg reserve3초/process reserve5초를 유지하며 deadline은 fit 시작+55와 전체 시작+64 중 먼저 도달하는 쪽이다. 각 cache hit는 원본 응답으로 세지 않는다.

원본 상한40LFbatch/120조건부응답/1200조건부초/4800000contextticks/331200000scalar MNticks/5241600000edge-stage ticks/40banks/240000decoder끝점/기하 forward·backward 각각218920행이다. tiny 비용,고정/동적 전송,준비/전진/미분/기하의 요청·반환·동기화·unknown,실제 optimizer 요청/반환/의도한 한도 중단/오류,부분실행과 wall을 별도로 남긴다. 예외가 생겨도 elapsed 및 counter를 finally에서 보존하고 cleanup의 부가 오류가 원래 오류를 덮지 않게 한다.

E131까지 실제 보수적누적1449.3127684000065초를1450으로 올려73초를 배정한다. 종료1523<=기존 공유1530초다. 실패·테스트·이전 원본 비용을 초기화하지 않는다. 최종 manifest와 시간 확인이 성공 반환보다 먼저다. 부모 독립 검토는 저장 배열만 사용하며 신경/기하/fit을 더 실행하지 않는다.

## 개발 소유권

부모 설계→**gpt-5.6-sol/high** 구현→부모 첫 리뷰→같은 Sol 첫 수정→부모 두 번째 리뷰→남은 수정은 부모 직접 수행한다. 기존 수학·CUDA·SLSQP를 재사용하고 warm start/선택/계상 계약을 연결하는 범위이므로 high를 실제 호출에 설정한다.

Sol은 새 `tools/fit_c_banc_m3_lf_constrained_continuation.py`, `tests/test_c_banc_lf_constrained_continuation.py`만 구현한다. 부모는 freeze·원본 supervisor·독립 배열 검토를 소유한다. 기존 검증 helper를 재사용하되 E131의 최초 feasible 종료·16회 상한·LF 복원 판정을 E132의 판정으로 잘못 재사용하지 않는다. 버전/스키마/에러 이름은 E132로 명시한다.

테스트는 생성 CPU 입력으로 feasible warm start,고정 SLSQP 목적/Jacobian과 epigraph cache,초기 불일치 시 optimizer0,실제 최선 feasible 선택,1%기준,비feasible·추가평가·cache/시각 한도 및 오류 계상을 확인한다. 원본/생성 신경망·신체/F6는 실행하지 않는다. 모든 테스트 invocation과 실패를 append-only로 보존한다.

CLI `--freeze --freeze-sha --source-review --source-review-sha --out`와 정확한 새 두 소스 해시를 가진 부모 receipt의 `accepted_for_single_original_lf_constrained_continuation=true`, `blocking_defects=[]`를 요구한다. E131 부모검토를 통해 E130/E129/E128 입력/소스/보존 해시를 재검증하고 E131 original19파일의 manifest도 확인한다. 최종 부모 source review 전 원본 실행은0이다.
