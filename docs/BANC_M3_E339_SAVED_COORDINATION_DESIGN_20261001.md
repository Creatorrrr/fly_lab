# E339 저장된 다리 간 명령 협응 판별 설계

2026-10-01 ROOT 설계와 실제 Astra 6/max의 M1–M7 리뷰 반영본이다. 같은 자연 보행 finite-goal의 다음 진단 단계이며 새 목표가 아니다. F1–F8, 신경 단독 런타임, BANC 해부학적 provenance, 과거 실패·raw·M2·UI·미사용 F6와 기존 원본 비용 한도를 보존한다. 아직 원본 실행을 예약하지 않았다.

## 목적과 현재 근거

E336의 독립 원자료 검산에서 E335 live 명령은 다리별 1초 동안 global-reference 위상이2.0174–2.4811회 진행하고 실제 관절은 RMS.001096–.001979rad로 추적했다. 그러나 최장 전진 bout은.21초이고 공식 eligible cycle은0이다. 이번 질문은 각 다리의 진행 여부보다 **다리 사이의 global-reference 명령 위상 관계가 협응된 reference에서 얼마나 벗어나는가**다. 결과로 신경 학습의 기존 협응 항을 먼저 교정할지, 접촉·stance 추진/명령 형상 문제를 우선할지 결정한다.

현재 E332 `SmoothTemporalObjective.evaluate_pairs`에는 이미 모든15쌍에 대한 `mean(1 - z_a·z_b)`가 계수1로 들어 있다. 협응 항이 없다는 가정이나 같은 항을 다시 추가하는 제안은 하지 않는다. 이 항은 원형 moment 크기와 각도 차이를 함께 벌점으로 삼으므로, 기존 손실의 두 성분을 분해하고 단위 moment의 관계를 따로 확인한다. 새 gain, loss나 neural 학습은 이번 단계에 포함되지 않는다.

확인한 로컬 FlyGym demo CPG는 위상 차이와 phase bias로 oscillator coupling을 계산하며 hybrid controller는 tarsus/thorax 높이와 접촉 힘을 사용한다. 이는 협응과 감각 보정의 구분에 관한 구현 참고다. 최종 BANC 런타임에 CPG, 기준 위상, 궤적이나 root 교정을 넣지 않는다.

## 판별 질문, 대조와 변경 범위

H339: 현재 live trajectory에 고정128개 reference 격자에서 구한 envelope보다 큰 encoded angular discrepancy가 지정된 크기와 coverage에서 존재한다. 경쟁 설명은 다리 간 encoded 관계에 해당 rule의 큰 차이가 검출되지 않지만 발 접촉/지지/추진이나 관절 명령 형상이 신체 보행에 불충분하다는 것이다. Reference encoding 자체가 모호하거나 coverage가 부족하면 proxy_limited로 판정하며 어느 설명도 채택하지 않는다. 실행 오류는 H339의 반증이 아니다. Positive도 연속 위상이나 reference 밖 명령의 encoder 편향을 배제하지 않는다. R>=.5는 weight 집중도이며 기준 곡선 근접도나 위상 정확도의 보증이 아니다.

대조는 같은 encoder에 모든 다리의 공통 global phase128개를 넣은 `q_l=C_l(phi)`다. C에는 해부학적 offsets가 이미 들어 있다. offsets를 다시 더하지 않는다. 이 대조는 합성 reference 명령이며 신체 교사 rollout, 접촉 위상, F4 발걸음이나 생물학적 반복으로 부르지 않는다. Neutral/held/live는 같은 개발 자료의 서로 다른 trajectory이며 독립 개체나 confirmatory 반복이 아니다. 일차 판정은 live와 합성 reference만 사용하고 neutral/held는 기술 통계만 보고한다.

새 owned source는 `tools/analyze_c_banc_m3_saved_coordination.py`와 `tests/test_c_banc_saved_coordination.py` 두 개뿐이다. E336/E337/E338와 neural/runtime/UI 소스를 수정하지 않는다. E338의 현재 `scan_archive`를 헤더 관측용으로만 재사용할 수 있다. 생성 전용 `decode_generated`, `gate`, `metadata_worker` 또는 실패한 실제 E338 CLI를 호출하거나 변경하지 않는다. 원본 값을 읽는 API는 새 E339 도구 안의 명시적인 고정 입력 reader뿐이다.

## 입력과 provenance

각 경로는 `verification/banc-natural-gait-m3-20260919/` 아래다. 파일 bytes/SHA는 producer manifest의 저장 기록이며 이번 리뷰 전 array 값이나 헤더를 새로 읽은 결과가 아니다.

|입력|저장 파일 SHA256|bytes|해석할 값|
|---|---|---:|---|
|`e332-smooth-temporal-saved/reference.npz`|`140e23c865d7e110cd93699b41612c1ffb02eb88dc100ac435c4dcba5acf6c01`|124353|curves, normalization, temperature, future_anchors, phase_basis|
|`e336-saved-gait-transfer/neutral-smooth.npz`|`8b75c4b1d2cd7d4efc0fe2db0ad49ad50b7dbf2d9ca25b6f38359c9488603d4c`|1644664|q, z, R, coordination_pair_loss|
|`e336-saved-gait-transfer/held-smooth.npz`|`0ea9afacfb31fcb7595b1c37d235dfbfb39979f366c1a0866f0ec65b3cab1ee6`|1644664|q, z, R, coordination_pair_loss|
|`e336-saved-gait-transfer/live-smooth.npz`|`4d2ef1a8a22243b905adea9420ef7230e93e51d19d01e73615c37e84b9953ef4`|1644664|q, z, R, coordination_pair_loss|

Reference schema는 curves[6,128,7], normalization[6], temperature[6], future_anchors[6,128,7], phase_basis[128,2]이며 모두 `<f8`이다. Trajectory는 q[201,42], z[201,6,2], R[201,6], coordination_pair_loss[15]의 `<f8`이다. 모든 선택 값은 finite여야 한다. 다리 순서는 LF, LM, LH, RF, RM, RH로 고정한다.

구현 전에 ROOT는 직접 사용하는 source/JSON의 pins를 만들고, 리뷰 후 설계와 reflection, 실제 model dispatch, 생성 자격과 최종 owned source를 freeze에 추가한다. `e336-saved-gait-transfer/output-manifest.json`의 SHA는 `e2fcc128f0c0cce3f4f13ba4865382445a96a7621c0f521895e2ae1bc8958f21`, E332 manifest는 `acfe3cde1dccd8aa6b2e2dc0b35697b91c245ca82229cf47606ac5029f5f16f0`다. E336 final decision과 7141항목 감사, 고정 reference producer, 사용할 E338 scanner 및 본 설계를 연결한다.

이번 작은 저장 진단은 직접 의존하는 파일의 bytes와 과거에 종료된 producer 증거를 검증한다. 이미 종료된 약3000개 inherited leaf를 모두 다시 hash하는 실험과 구분하며, 그 재검증을 했다고 주장하지 않는다. F2의 최종 후보에 필요한 전체 BANC closure를 면제하지 않고 runtime/candidate adoption은0이다. 원본4개 archive는 실행 시 pre/post SHA·bytes 일치, 모든 member의 안전한 header 관측, 선택 필드의 dtype/shape를 모두 확인한 뒤 선택 값만 allow_pickle=False로 읽는다. 추가 배열도 header 안전성을 검증하되 값은 해석하지 않는다. E338의 historical-selected 기록을 새 complete roster로 간주하지 않는다.

## 固定計算とvalidity

E332의 frequency=2.382760007990865Hz, dt=.005s, 128개 anchor의 wrapped linear future interpolation으로 normalization n, tau=7n, cos/sin basis를 독립 NumPy로 재계산해 저장 constants와 rtol1e-12/atol1e-15로 비교한다. n/tau는 positive finite여야 한다. Torch나 기존 encoder를 import/호출하지 않는다.

각 q를 Gaussian softmax로 다시 encoding해 저장 z/R와 rtol1e-10/atol1e-10로 비교한다. 201개 경계의 모든 행을 비교하고 통계는 c101..300에 대응하는 마지막200행을 사용한다. 저장 coordination_pair_loss는 모든15쌍의 `mean(1 - z_a·z_b)`로 독립 재계산해 같은 tolerance로 비교한다. 저장 mask/R만 그대로 신뢰해 계산하지 않는다.

R>=.5일 때만 단위 moment u=z/R를 정의한다. 낮은 R에 epsilon을 더하거나 위상 채움/unwrap/역진 삭제/시각 변경/다리별 offset 보정을 하지 않는다. Pair는 양쪽 다리가 valid인 행에서만 angle δ=arg(u_a conjugate(u_b))/(2π), mean circular resultant, coverage count, mean bias를 계산한다. Pair mean magnitude<=1e-12면 bias=null이다. 6다리 coherence는 모든 다리가 valid인 행의 `|mean_l u_l|`를 보고하고 다른 행은 보완하지 않는다.

같은 행의 기존 loss를 `1-RaRb`와 `RaRb*(1-cos(2πδ))`로 분해한다. δ의 단위는 cycles다. 낮은 R도 포함한 원래 z로 대수적으로 계산하며 R=0에서도 나누지 않는다. 실제 angular weighted 성분은 `RaRb - dot(z_a,z_b)`로 구하고, 두 성분의 합과 기존 loss가 일치하는지 검증한다. 음의 반올림 오차는 숨기지 않고 저장한 뒤 tolerance 이내인지 판정한다. 유효한 normalized angular 통계와 원래 unmasked loss의 모집단은 구분해 표시한다.

합성 reference도 같은 encoder로 z/R/u를 만든다. 15쌍마다 reference mean bias, resultant, valid coverage, mean에서의 absolute wrapped deviation 최대값을 구한다. Reference sampling과 live의 위상 체류 분포 차이를 고려해 평균끼리만 비교하지 않고, reference의 모든128개 anchor가 만드는 deviation envelope와 live의 각 행을 비교한다.

## 事前に固定する判定

아래 문턱값은 F4나 생물학적 기준을 변경하지 않는다. 진단에서 큰 angular discrepancy를 찾기 위한 공학적 decision rule이다.

- Reference calibration: 모든15쌍의 valid coverage>=.95, bias가 null이 아니며, 각 pair의 reference deviation max<=.05cycles다. 또한 모든6다리의 공통 valid coverage>=.95여야 한다.
- Live coverage: 모든15쌍의 valid coverage>=.90, 모든6다리의 공통 valid coverage>=.90이다.
- 각 live pair에서 reference bias와의 absolute wrapped deviation이 `reference max deviation + .15cycles`를 엄격하게 초과하는 행의 비율을 구한다. 분모는 그 pair의 유효 행이다.
- Calibration/coverage가 성립하고, 15쌍 중 하나라도 이탈 비율>=.25면 `angular_misalignment_prominent`다. 전체15쌍과 좌우3쌍을 고정 순서로 보고한다. 모든 pair가 미달이면 `no_prominent_angular_misalignment_at_fixed_rule`다. Calibration/coverage가 성립하지 않으면 `proxy_limited`다.
- technical_invalid, proxy_limited, positive/negative diagnostic은 구분한다. 실행 또는 저장·감독·ROOT 원자료 검산이 실패하면 과학 판정을 채택하지 않는다.

Count 비교는 정수로 한다. Reference pair/all6 coverage는122/128 이상, live는180/200 이상이며 이탈 비율 문턱은 `4*exceed_count >= valid_count`다. 이탈의 엄격한 `>`와 calibration의 `<=.05`에 tolerance를 덧붙이지 않는다. Pair 순서는 `(a,b), a<b`의 사전식 순서, 좌우3쌍의 0-based index는 `[2,7,11]`이다. Reference bias null은 proxy 제한이지만 live mean bias null만으로 proxy_limited를 추가하지 않는다. 판정에는 live의 유효한 각 행별 편차를 사용한다. 빈 집합의 mean/max는 null이며 위상을0으로 채우지 않는다.

Positive면 기존 협응 loss의 효과/기울기/배분을 살펴보는 다음 설계를 선택한다. 단순 중복 추가나 gain sweep을 승인하는 결과는 아니다. Negative는 고정 rule에서 큰 discrepancy가 검출되지 않았다는 뜻이며 접촉 원인의 입증이나 작고 짧은 협응 문제의 배제가 아니다. 이 경우 접촉·stance 추진/형상 방향을 우선하고 phase proxy 판정을 보행 협응의 증명으로 대체하지 않는다. Proxy_limited면 이번 encoder로 두 방향을 결정하지 못했다고 기록하고 같은 원본 실행을 반복하지 않는다.

## Astra 필수 보완의 구체적 실행 계약

M2의 합성 reference는 정확히 `q_ref[s,7*l+j]=curves[l,s,j]`다. Encoder는 `softmax(-sum_j((q-C)**2)/temperature)`이며 임의 Gaussian1/2 계수를 넣지 않는다. 저장 z/R는201행 모두 비교하지만 저장 coordination_pair_loss와 모든 판정 통계는 encoded[1:]의200행으로 계산한다. 기존 전체 협응 항은15개 pair loss 평균에 계수1이다. 낮은 neutral/held coverage는 기술 통계이며 live 판정 gate를 바꾸지 않는다.

M3의 scanner 호출은 `scan_archive(...,descriptor=None,descriptor_scope="observation")`로 고정한다. E338의 qualify_role/SPECS는 새5필드/4필드 선택 schema를 대신하지 않는다. Scanner의 posthash는 헤더 관측 이후의 증거다. 새 reader는 값 읽기와 np.load context 닫기 이후 SHA와 bytes를 다시 확인해야 한다. Decode 실패가 먼저 발생하면 그 오류와 이후 posthash 오류/미완료 사유를 함께 남긴다. 생성 namespace에서는 같은 binding 검사 → scanner → 새 선택 schema → 실제 selected decode 경로를 통과한다. 허용된 extra numeric 배열 성공, unsafe extra, 잘못된 필드, decode 이후 파일 변경, 선택 필드 외 값 접근 차단을 시험한다. 원본 binding의 역할별 literal 경로/SHA/bytes/fields는 production CLI에서 교체할 수 없고 loader를 mock한 성공은 이 자격을 대신하지 않는다.

M4의 직접 의존성은 유한한 고정 목록으로 freeze에 둔다. Source는 최종 owned2, 현재 E338 scanner, 검토한 E336 analyzer와 E332 smooth objective다. JSON/문서는 이 설계, 실제 Astra review/reflection/dispatch와 생성 자격, 설계 등록의 사전 snapshot, E336/E332의 output manifests와 아래 종료 증거다. Mutable progress/cost ledger를 실행 pins로 쓰지 않는다. 여기 적은 text evidence의 직접 pins와 명시한 edge만 확인하며 재귀 inherited leaf hashing을 호출하지 않는다.

|종료 증거|고정 상태와 연결|
|---|---|
|`parent-e336-final-decision.json`|analysis_valid=true, root_independent_raw_qualification=true, raw_checks=7141; evidence map의 manifest/audit 연결|
|`parent-e336-saved-raw-audit.json`|passed=true, first_error=null, checks7141개 모두true; final decision의 audit pin|
|`parent-e336-original-supervision.json`|success=true와 final decision/audit의 저장 supervisor pin 연결|
|`parent-e332-final-decision.json`|analysis_valid=true, input_qualification_pass=true, root_raw_audit_checks=2946; reference producer의 result/audit/manifest pins|
|`parent-e332-saved-raw-audit.json`|passed=true, checks2946개 모두true와 final decision의 audit pin|
|`parent-e332-saved-supervision.json`, `parent-e332-audit-supervision.json`|success=true와 final decision의 감독 pins|
|`e332-smooth-temporal-saved/result.json`|analysis_valid=true, input_qualification_pass=true, first_error=null 및 E332 final decision의 result pin|

이 table의 경로는 기존 verification base 아래다. JSON 자체의 상태와 연결된 pin을 검증하며 전체 upstream source 재검증으로 부르지 않는다. Runtime/F2 후보 자격과 분리한다. ROOT 잔여 source 수정이 발생하면 변경된 SHA에 관한 해당 검사를 실행/보존한 뒤 현재 생성 자격을 다시 작성한다. 이전 hash의 PASS를 새 source PASS로 승계하지 않는다.

M5의 CLI worker s0는 NumPy/scanner import와 gate 전에 잡는다. CPU thread1과 CUDA 숨김은 NumPy import 전에 적용한다. Science deadline=s0+8, internal=s0+12이며 finalization deadline=min(s0+12,final_begin+4)다. 단계별 전체 clock 재시작은 없다. ROOT audit은 별도 시작으로 internal8/supervisor10이다. Supervisor는 interpreter/worker 시작을 포함한 owned tree elapsed/RSS/종료를 측정한다.4MiB 출력 한도는 original output directory의 모든 파일/retained staging, 원본 stdout/stderr/감독/외부receipt 및 감사 output/stdout/stderr/receipt를 포함한다. 실패 기록도 포함하며 source/design/generated 자격 파일은 별도 생성32MiB 한도다. Fresh output이 있으면 기존 파일을 수정하지 않고 stdout/외부receipt로 실패를 보존한다.

M6의 저장 순서는 작은 numeric archives와 selected-read ledger → provisional terminal result/동일receipt → 해당 고정 파일들의 byte hashes와 bytes를 담은 manifest → 모든 파일/manifest readback과 hash 검증 → 동일 terminal JSON stdout이다. Manifest는 자신이나 stdout을 자기참조 hash하지 않는다. 계산의 suggested_diagnostic은 provisional이고 science_adopted=false다. 저장/rename/readback/clock 실패는 technical_invalid로 반환하고 exit1을 내며, 남은 성공 terminal이 있다면 invalid revocation을 시도하고 실패도 보존한다. Supervisor 종료 후 ROOT가 parse한 stdout/result/receipt 일치와 cap/종료를 별도의 외부receipt로 확인한다. 그 외부receipt와 ROOT 원자료 감사가 PASS한 뒤에만 별도 ROOT final decision에서 과학 판정을 채택한다. 완료된 원본 terminal/manifest를 사후 덮어써 성공으로 바꾸지 않는다.

M7의 ROOT audit은 E339 numerical/classification helper나 verdict를 정답으로 호출하지 않는다. Pinned 원본 q와 reference5필드부터 독립 수식으로 constants/encoder/201행 대비/200행 선택/15쌍 loss와 분해/원형 통계/reference envelope/이탈count/6다리 coherence/최종분기를 다시 계산한다. 자체 선택 읽기 ledger와 output hash 확인을 남긴다. 감사의 생성 qualification은 row selection, pair 순서, cycles/radians, denominator 및 결과 파일을 의도적으로 바꾼 사례를 거절해야 한다. 원본 감사 실패 시 retry 없이 science 미채택으로 종료한다.

## 実装、検証と実行境界

ROOT 설계 → 실제 GPT-6 Astra/max 읽기 전용 리뷰 → ROOT 반영 → 실제 GPT-6.1 Sol/high의 새2개 source 구현 → ROOT FIRST → 필요하면 같은 Sol ONE R1 → ROOT SECOND → 잔여 ROOT 수정 순서다. 독립 수치 재현, 원형 통계, 엄격한 reader와 실패 저장을 포함하므로 effort는 high다. Sol은 재위임하지 않는다. Source2 외에는 Sol의 생성 시험 증거만 쓸 수 있다. 모든 원본 값/header, 전체 leaf hashing, body/neural/CUDA/F6/UI/기존 CLI job을 구현 시험에서 사용하지 않는다.

생성 시험은 literal scalar 독립 encoder와 vector encoder의 일치, global phase/anatomical offset 이중 적용 검출, 같은 위상/반주기/대칭 낮은R/null/circular wrap, coverage/deviation 경계, loss 분해, 잘못된 schema/변경 SHA/unsafe extra/선택 값만 읽기/고정 원본 binding, 과학적 결과와 기술적 오류 구분, deadline/저장 readback 실패를 다룬다. 수만 늘리는 구현 복제 시험을 작성하지 않는다. ROOT는 독립 생성 자료와 source를 review하며 원본 reference 값을 본 뒤 판정을 바꾸지 않는다.

원본 저장 분석은 자격 후에만1회, retry0다. supervisor15초, science8초, finalization 최대4초, internal12초, CPU1/GPUhidden, host2GiB, output 상한4MiB다. s0는 gate 전이다. 출력은 fresh literal `verification/banc-natural-gait-m3-20260919/e339-saved-coordination/`다. Result/receipt/stdout과 manifest의 일치·readback·hash를 마지막으로 확인하고 publication 전에 성공을 채택하지 않는다. 원본4개 archive의 선택 값 읽기를 실제 ledger에 저장한다. Weights나 neural/state/body 자료는 출력하지 않는다. 3개 trajectory와 reference의 z/R/pair mask/angle/decomposition/이탈 수를 작은 npz에 저장해 ROOT의 독립 재계산을 가능하게 한다.

ROOT 원자료 감사는 별도1회 supervisor10/internal8초다. 저장 output과 필요한 선택 source만 독립 재계산하며, 과학 판정·선택 값 대응·계산/decomposition/masks·publication을 확인한다. 감독 elapsed는 실패 포함 한 번만 계상한다. 실제 비용이 cap을 넘으면 초과분도 기록한다.

기존 원본 누적9565.21586319952/9660초, 잔여94.78413680047925초다. 현재 예약0이며 자격을 갖춘 두 실행 전 최대15+10초가 잔여 내인지 재확인하고 새 reservation을 사전에 기록한다. 기존 budget을 reset/증가하지 않는다. ROOT와 Sol의 새 단계 생성 감독 상한은 각60초, retained 각32MiB며 이전 단계 비용은 모두 별도로 보존한다. GUI/body/neural/CUDA/F6 호출0이다. 최종 후보 채택/자연성/F1–F8 완료는 이번 결과로 성립하지 않는다.

## 根拠と未達事項

`docs/BANC_M3_E336_SAVED_GAIT_TRANSFER_RESULTS_20261001.md`, `tools/analyze_c_banc_m3_saved_gait_transfer.py:269-337,401-439,1539-1593`, `tools/banc_m3_smooth_temporal.py:170-207`, producer manifests와 E338 R1 소스 결과가 현재의 근거다. 다음 neural-only 후보와 실제 CUDA/폐루프/10초 body/미사용2개 초기 상태/6개 목표·정지·재시작/원자료·영상/M2 복원은 미달인 상태로 계속 진행한다.
