# E192: singleton 전달 적합의 입력 출처 검증 복구

부모 사전 설계. E191은 입력 별칭 오류로 fit0/invalid이며 이를 성공으로 바꾸지 않는다. 그 원형·코드·freeze·비용을 그대로 보존한다. E191 이후 root 출처 감사가 각 원본 해시와 배열의 정확한 일치를 확인했고, 두 잘못된 별칭 외 기존 입력 조건은 통과했다. 이번은 같은 가설 H191의 아직 실행되지 않은 단일 적합을 별도 실행으로 복구하는 작업이다. 최종 F1–F8 및 모든 실제 과학적 문턱은 불변이다.

## 변경 범위

새 소유 파일은 `tools/fit_c_banc_m3_singleton_transfer_recovery.py` 및 `tests/test_c_banc_singleton_transfer_recovery.py`다. E191 frozen 소스는 읽기 전용이며 standalone 새 사본의 diff를 검토한다. actual gpt-6-sol/high 최초 구현 → 부모 FIRST → 필요 시 같은 Sol 첫 수정 → 부모 SECOND 및 잔여 직접 수정. 부모는 별도 감독·독립 결과 검산을 담당한다.

E191의 `loaded_trace_recorded_command == loaded_startup_recorded_command`와 `loaded_trace_tendon[1:] == loaded_startup_recorded_tendon` 두 조건만 출처별 검사로 바꾼다. 어느 배열도 덮어쓰거나 다른 배열로 치환하지 않는다. trace는 E150 actual이며 startup은 E103에 보존된 E100 기준이다. E103 전체 명령과 E150 전체 명령이 달라도 출처가 일치하면 올바른 입력이다.

E169 `pre-solve-raw.npz`의 같은 원형을 사용한다. 추가로 해시가 고정된 E103 `e103-startup-preserving-recruitment/startup-inputs.npz`와 E150 `e150-native-monotone-body/trace.json`를 각 원래 출처로 읽는다. freeze `source_origins.startup/trace`에 workspace relative path/SHA256을 넣고 실행 전후 검증한다. E169 원래 freeze의 input_sha256과 부모 동결에서 일치를 확인한다.

필수 loaded_startup_mask(2001,bool)를 추가하고 1..99를 검증한다. FIRST metadata 감사에서 E169는 loaded_startup_time_s를 저장하지 않았음을 확인했다. 따라서 그 별칭을 만들지 않고 실제 E103 origin_startup_time_s(2001,float64)와 .005초 그리드를 exact 검증한다. E103 원래 mask/target_command/target_tendon/recorded_command/recorded_tendon과 대응 loaded 필드는 모두 exact여야 한다. `target_command == recorded_command[mask]`도 유지한다. E103 target_tendon은 당시 activation에서 계산된 값이므로 recorded_tendon과 새로운 exact 동일성을 발명하지 않는다. E150 trace samples2001행의 필수 joint_targets_rad 및 tendon_inputs 6개를 원래 LEGS 순서로 읽어 loaded_trace와 exact 검증한다. 누락 키를0으로 채우지 않는다. 이 출처 원형과 기존 loaded 원형은 모든 검증 전에 pre-solve raw에 함께 저장한다. 로드 도중 오류에도 이미 수집한 원형과 primary/secondary 오류를 보존한다. 부분 행의 shape 오류가 최초 누락 키 오류를 덮지 않아야 하며 pickle 없이 원형을 보존한다.

나머지 E191의 입력 소유권, 90근육/343MN/1029pair/2058weights/4117변수/21189linear/15SOC/18QR, LTM6 exact, inner guard1e−8, witness t−.251, native/affine 표현, 물리 및 MN/근육 합산 보호, 1824command/1823velocity, 99startup, 원래20%·각다리·시작 자세 기준은 그대로다. E191 설계의 solver Clarabel0.11.1/CPU1/qdldl/max_iter100/time_limit20 및 생성+solve포함20초, 수치 기준(primal/dual/SOC1e−7, linear1e−8, native command/activation1e−12, velocity5e−10), 유한 candidate 선저장과 실패 후 success 취소도 그대로다. 결과는 proposal/valid_rejection_inconclusive/invalid_not_tested로 구분하고 family 불가능성을 주장하지 않는다.

## 시험과 자원

새 생성시험은 E103 기준 기록과 E150 실제 기록이 실제로 다르고 각자의 출처와는 일치하는 fixture를 쓴다. 다른 출처가 정상 통과하고 어느 한 원형 출처/target/mask/hash 손상은 적합 전에 기각되는지 검증한다. 기존 다양한 신호의 mapping/native/QR와 해가 알려진 zero-drive real SOCP(t=−.25), 시간 초과와 늦은 게시 실패 시험을 보존한다. 원본 numeric input/모델/fit는 구현·시험 중0이다.

기존 누적4658.458035699718/4994초, 잔여335.541964300282초에서 감독95초만 예약한다. 내부80/외부90/감독95초, 원본 프로세스1/fit1 이하/자동 재시도0, CPU1/GPU0, 출력256MiB/free2GiB. 새 신경·신체·state·CUDA·F6·native export0. 소스 준비·생성시험·부모 검산 비용은 별도 측정하며 원본 비용은 기존 누적에 추가한다. 더 넓은 예산이나 새 한도를 만들지 않는다.

부모 SECOND 승인과 frozen source/hash gate 후 원본을 한 번 실행한다. proposal이면 별도 native/cut/snapshot/3초 비교를 설계하고 남은 동일 예산과 source review를 확인한다. invalid는 보존하고 이 실행을 자동 반복하지 않는다. 유효 미달은 실제 출력의 한계를 다음 결정에 사용한다. 어느 결과도 자연 보행 완료나 후보 채택 자체가 아니다.
