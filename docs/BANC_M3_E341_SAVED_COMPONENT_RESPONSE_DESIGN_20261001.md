# E341 기존 신경 효능 변화의 손실별 실제 반응 분리

2026-10-01 actual `gpt-6-astra/max` 리뷰 **불충분**을 ROOT가 반영했다. **아래 원본203 archive 분석 및 구현 계획은 미실행 종료한다.** 원래 초안은 `parent-e341-pre-astra-design.md`에 보존한다. 아래 계산계획은 실행 승인·현재 채택 설계가 아니다. 동일 BANC M3 finite-goal/F1–F8, 과거 실패, M2/UI/F6 비노출과 비용을 유지한다.

## 실제 리뷰 반영과 종료

M1: 이미 감사된 E333 metadata에 local50창 평균 coordination neutral0.8226893699603092 → minus0.8226535248837067, successor neutral6.94942149213743 → minus6.94730426361345가 있다. 실제 minus 개선 C=0.00003584507660248093, S=0.00211722852398033다. 이는 기존 감사의 재사용이며 mean 차이에 다시/50하지 않는다. H341의 C 감소 여부는 이미 알려졌다. Component Q 재계산만으로 다음 개입을 달리 고를 충분한 근거가 없어 새 원본 분석/감사를 수행하지 않는다.

M2: `observed_minus_C_response=reduced_in_saved_local_prefix`, `component_Q_reconciliation=not_evaluated`, `status=closed_unexecuted_redundant_original_design`다. 새 numerical PASS/technical failure/구현 완료가 아니다. E341 original/header/value/hash/NN/body/CUDA0, 예약0, 원본 누적9573.807361699519/9660초 및 잔여86.19263830048112초를 유지한다. E333 whole invalid_deadline/effective qualifications=false/startup/body failure와 후보/F1–F8/biology false도 유지한다.

다음은 별도 C 전용 actual parameter VJP와 실제 f32 효능 변화 전달 설계다. 이번 Astra 리뷰가 다음 설계를 승인한 것은 아니며, 새 ROOT 설계 → actual Astra6/max → ROOT 반영 → actual GPT6.1 Sol 구현/ROOT FIRST/SAME Sol ONE R1/ROOT SECOND/잔여ROOT를 별도로 적용한다. 근거: `verification/banc-natural-gait-m3-20260919/parent-e341-astra-design-review.md`, `parent-e341-prior-observed-component-evidence.json`.

## 보존된 미실행 계산계획

## 질문과 선택 근거

E339에서 live 명령의 encoded angular misalignment가 고정15쌍 중11쌍에 두드러졌다. E340 독립 원자료 감사938그룹/3788검사는 live 출력의 `||gS||/||gC||=22.6218395`, `cos(gC,gS+gC)=0.020237774`, `dot(gC,gS+gC)/||gC||²=0.457777902`를 확인했다. 크기 우세만으로 실제 neural parameter 방향을 설명할 수 없다.

E333은 전체 deadline 무효이지만, 독립 원자료 감사3414개가 완료된 50개 neutral/plus/minus local prefix의 실제 f32 효능 변화와 출력·O/Q/T를 검증했다. 기존 총손실 방향의 primary 오차0.09418499%, window-L1 오차0.20920826%는 그 저장 prefix의 진단이다. 전체 E333/continuation 자격을 회복하지 않는다. 이미 150개의 실제 국소 신경 출력이 있으므로 새로운 parameter VJP나 신경 시뮬레이션 전에 이를 손실별로 분리하는 것이 가장 작은 추가 검증이다.

H341: **이 기존 실제 신경 효능 minus 변화는 local prefix에서 다음 명령 예측 S뿐 아니라 협응 C도 줄였는가?** 기존 출력 공간의 S/C gradient를 그 실제 `delta_q=q_plus−q_minus`에 투영하면 손실별 반응과 일치하는가? 결과로 실제 가중치 변화에서 협응 신호가 약해지는지/반대인지/이미 전달되는지 구분한다. 이는 한 기존 방향의 효과이며 full `gtheta_C`의 크기, 전체 parameter Jacobian, 새 C 전용 방향의 실현 가능성이나 실제 보행 개선을 증명하지 않는다.

## 고정 자료와 범위

새 source owned2는 `tools/analyze_c_banc_m3_saved_component_response.py`와 `tests/test_c_banc_saved_component_response.py`다. 구현은 actual GPT6.1Sol/xhigh, ROOT FIRST → 필요시 같은 Sol ONE R1 → ROOT SECOND → 남은 ROOT 수정 순서다. 과거 source/raw/audit/분류의 수정0이다.

원본 입력은 다음 **203개 literal archive의 선택값508개**다. Gate 준비 시 기존 manifest/final/audit metadata에서 이 목록의 SHA/bytes를 얻어 유한 inventory로 고정하고, 원본 NPZ의 current hash/header/value는 current source 생성 자격·ROOT source review·사전 예약 뒤에만 읽는다.

1. `e332-smooth-temporal-saved/reference.npz`의 curves[6,128,7], normalization[6], temperature[6], future_anchors[6,128,7], phase_basis[128,2]: 5개 finite `<f8`.
2. `e333-smooth-efficacy-direction/window-{c:03d}-{neutral,plus,minus}.npz`, c=100,104,...,296의150개. 각 q[4,42], fixed_original_q_boundary[42], loss_terms[4]만 선택:450개 finite `<f8`. Loss 순서는 training/successor/coordination/LR이며 LR 값은 재계산 정합성 확인만 한다.
3. `e333-smooth-efficacy-direction/window-{c:03d}-neutral-gq.npz`의 gq[4,42],50개 finite `<f8`. 이는 기존 전체 S+C 출력 기울기이며 component neural VJP로 이름을 바꾸지 않는다.
4. `e333-smooth-efficacy-direction/direction.npz`의 theta_plus/theta_minus[8547],2개 finite `<f8`, 양수 exact f32-promoted. 실제 delta는 두 저장값의 차이이며 `.002*d`로 대체하지 않는다. Direction의 full window parameter gradients 등 다른 값은 선택하지 않는다.
5. `e336-saved-gait-transfer/neutral-smooth.npz`의 q[201,42],1개 finite `<f8`. 원래 E318 neutral과 local prefix의 boundary/neutral 출력 대응을 확인하는 직접 연결 자료이며 추후 mismatch를 재인덱싱이나 다른 궤적으로 바꾸지 않는다.

150개의 큰 archive에 들어 있는 rate/adaptation/bank/original_activation 등의 값은 읽지 않는다. 안전 scanner로 전체 header를 관측하되 선택값과 header-only 범위를 구분한다. Current E338 scanner, E339 fixed selected-reader의 공통 안전 primitives를 재사용할 수 있지만 그 고정 role schema를 monkeypatch하지 않는다. 새 typed selected reader를 명시적 인자로 만든다. 새 neural/decoder/body loader, Torch CUDA/CuPy, parameter VJP, optimizer, CSR patch 및 새로운 효능 후보0이다.

Text/JSON direct closure는 E340 final/audit/외부 two receipts/현재 source-review/freeze/생성 proof/정확 source2/결과 보고서, E333 final/raw audit/audit-source/source-manifest/result/advance-ledger/receipts/output-manifest/원래 supervisor와 contract, E332 canonical proof, 현재 안전 scanner와 실제 계산 import source만이다. E333 independent audit `passed=true`, checks3414 all true, saved prefix direction 기록을 확인하면서 E333 `run_valid=false`, formal invalid 및 effective qualifications=false도 동시에 요구한다. E340 diagnostic/source/current generated producer가 일치해야 한다. 역사적 NPZ·source/input closure 전체를 다시 읽거나 생체 모델을 재구성하지 않는다.

선택203 input pins는 E333의 immutable manifest와 ROOT final/raw-audit에서 고정한 원자료 descriptors에 연결한다. Archive roster/정렬/shape/dtype/첫·마지막 c/label/source producer 및150 local receipt의 completed4control/80ticks를 직접 대조한다. 실패 continuation receipt를 local prefix 자격으로 사용하지 않는다. 원본 단계에서 모든 선택 archive pre/post SHA/bytes를 확인하고 selected arrays의 C-order SHA를 기록한다.

## 고정 계산

Reference Gaussian, τ=7n, successor anchors와 anatomy phase offsets는 E332/E340 그대로다. 각 창의 별도 `b_w`는 세 arm의 저장 boundary가 exact 동일해야 하며 copied immutable이다. `x_arm[w,i]`는 각 arm의 실제 저장 q다. Neutral b와 q는 frozen neutral-smooth q의 q[4w] 및 q[4w+1..4w+4]에 rtol0/atol1e−11로 대조한다. 경계 overlap은 고정 경계와 live 변수의 별도 copy로 유지한다. 실패하면 technical_invalid이고 다른 row/arm/window subset0이다.

각 창의 **4행 평균** 손실 `L_{j,w}^{arm}`를 S/C/total에 대해 재계산한다. `ell=L/50`이며 **50개의 전체 prefix를 모두** 사용한다. Primary는 현재 qualified E340 `numpy_gradients(reference,b,x_neutral)`의 공개 numerical API를 사용한다. 그 공개 API 내부 numerical helper 호출은 허용하지만 새 구현이 private helper를 직접 호출하거나 old worker/gate를 재사용하지 않는다. Plus/minus value equation은 새 구현에서 명시적으로 구현한다. E340 numerical API의 lazy CPU Torch import는 실행 clock/실제 import source에 포함하되 CPU-only이며 CUDA query/context0이다. 기존 E333 gq는 neutral의 `gS+gC`와 rtol1e−8/atol2e−10로 대조한다. ROOT expected는 별도 covariance oracle이며 product gradient/classifier를 호출하지 않는다.

`gq_{j,w}=d(L_{j,w}/50)/dx_neutral,w`, `delta_q_w=x_plus,w−x_minus,w`다. 아래를 모두 저장한다.

- `O_{j,w}=(L_{j,w}^plus−L_{j,w}^minus)/50`.
- `Q_{j,w}=sum(gq_{j,w} * delta_q_w)`.
- `I_{j,w}=(L_{j,w}^neutral−L_{j,w}^minus)/50`: **실제 minus 개선**이며 centered O와 구분한다.
- Aggregate `O_j=sum_w O_jw`, `Q_j=sum_w Q_jw`, `I_j=sum_w I_jw`; arm별 mean loss, window별 O/Q/I 및 per-leg successor/LR 진단.
- `O_total=O_S+O_C`, `Q_total=Q_S+Q_C`, `I_total=I_S+I_C`와 기존 total gq의 Q를 수치 대조한다. 또 다시/50 또는2h로 나누지 않는다. Plus/minus actual theta delta를 parameter-space gradient로 부르지 않는다.

Primary signed response 비교는 의미있는 양쪽 `abs(O_j)>1e−9` AND `abs(Q_j)>1e−9`에서 같은 부호 및 `abs(O_j−Q_j)/abs(Q_j)≤.05`다. Negative Q/O는 합법적인 component 증가 방향이며 technical error가 아니다. Window cancellation guard는 `sum abs(O_jw−Q_jw)/max(sum abs(O_jw),sum abs(Q_jw))≤.05`다. 양 분모0 또는 의미없는 aggregate는 null/inconclusive로 남기고 epsilon/zero 치환/유리한 branch 선택0이다. Existing total-prefix verdict와 component verdict는 분리한다. 예상 밖 loss/gradient/norm/dot/ratio nonfinite는 technical_invalid다.

Primary 판정은 C에 대해 다음 사전 순서를 사용한다. Numeric/publication/외부 completion/독립 audit 실패 → `technical_invalid`; signed와window-L1 중 하나가 유효하게 불일치 → `component_output_prediction_not_reconciled`; 무신호/분모0 → `component_response_inconclusive`; reconciliation이 유효하고 `I_C>=1e−9` → `existing_minus_reduces_coordination`; `I_C<=−1e−9` → `existing_minus_increases_coordination`; 그 사이 → `component_response_inconclusive`. S와 total의 판정/수치는 보조로 그대로 보고한다. Centered O의 부호로 minus 효과 I를 대신하지 않는다. I_C의 부호를 전체 parameter-space gradient의 norm/cosine으로 확대하지 않는다.

## 결과에 따른 다음 결정

`existing_minus_reduces_coordination`이면 기존 변화의 협응 전달이 끊겼다고 주장하지 않는다. C/S 개선의 실제 크기와 E339/E335의 남은 협응·접촉 문제를 함께 보고, C 전용 parameter VJP 또는 긴 시간 범위의 전달을 구분한 새 설계를 선택한다. `existing_minus_increases_coordination`이면 기존 총손실 방향의 component compromise를 근거로 별도 C 전용 VJP/제약 방향을 우선한다. Prediction mismatch 또는 inconclusive면 기존 real-equation surrogate/finite-response/precision 범위를 좁히며 이번 동일 저장-run 재시도0이다. 모두 신경 개선/후보 채택/F1–F8 완료가 아니다. 답변이 어떤 것이든 기존 실패를 지우거나 현재 objective weight를 사후 변경하지 않는다.

## 자격·자원·게시 계약

원본 전 current source2를 사용하는 생성 CPU tests/Ruff와 ROOT 독립 covariance/FD/component-signed/cancellation/null/actualf32delta/shape·row·boundary·scale/old-gq/unsafe extra/selected posthash/clock/publication 실패 검증을 완료한다. Fixed50창과4행 API를 쓰고, 생성 archive는 fresh `e341-sol-*`/`parent-e341-*` namespace 아래다. 원본 CLI에 input/scale/window/limit override0, CPUthreads1/CUDA hidden이다. Metadata proof는 source2/helper argv/output/감독 실제 elapsed≤declared cap/actual positive RSS≤host cap/정확 PID completion와 logs를 finite union으로 닫는다. 현재 valid producer와 legacy 출력 PASS를 구분한다.

ROOT independent audit executable은 생성과 원본에서 동일 source/SHA로 실제 CLI를 감독한다. 정상 production 결과1개를 통과하고 wrong boundary/scale/row/gradient/component-sign/result6종을 숫자/판정 mismatch로 거절해야 한다. 이를 freeze에 연결한 뒤에만 원본 분석1회/retry0다. ROOT expected는 같은 Gaussian의 별도 covariance Jacobian, 별도 dot/mean/분류이다. Metadata만 검사한 verifier를 원자료 수치 감사로 부르지 않는다.

사전 한도는 **원본 분석25초 + 독립 원자료 감사20초 =45초**이며 현재 `9573.807361699519/9660초`, 잔여 `86.19263830048112초` 안에서만 qualify 뒤 예약한다. 원본 분석의 clock은 numerical imports/gate/모든 input read/hash 전 s0, science16초/internal21초/finalization5초/supervisor25초다. 감사는 clock-before-import/internal16초/supervisor20초다. CPU1/host2GiB/CUDA hidden/original output4MiB이며 neural/body/GPU work0이다. 초과는 technical_invalid·실제 비용 한 번 합산·재시도0이며 기존 누적이나 limit 증가0이다. 생성 Sol FIRST+optionalONE R1 공유60초/retained32MiB, ROOT FIRST+SECOND+잔여 공유60초/retained32MiB이며 E340의 비용과 raw를 그대로 보존하고 새 단계 allowance임을 구분한다.

원본 output은 literal `verification/banc-natural-gait-m3-20260919/e341-saved-component-response/`다. Normal raw `component-response.npz`는 reference5배열, b[50,42], q_neutral/q_plus/q_minus/delta_q/gS/gC/gTotal/old_total_gq[50,4,42]8개, theta_plus/theta_minus[8547], losses[50,3,4] 및 O/Q/I[50,3]만 저장한다. Neural endpoint/가중치/전체 state/parameter gradient arrays를 출력하지 않는다. 모두 finite `<f8`, raw serialized≤1MiB다. Selected-read ledger≤512KiB, result/receipt/stdout 각각256KiB, independent audit report/stdout 각각256KiB, manifest16KiB, 두 stderr 각32KiB/감독 각16KiB/PID 각8KiB/external receipts 각64KiB, final decision64KiB, 전체 공유failure/revocation512KiB다. 고정 maxima 합계3648KiB<4096KiB이며 every write/rename/print 전에 실제 staging 포함 bytes + 미게시 required quotas + failure headroom을 확인한다. Serialized raw는 once-write/staging rename이며 duplicate stage+target0이다.

Numeric/raw → 실제 selected ledger → provisional identical result/receipt/stdout → nonself manifest/hash/readback → 두 외부 completion receipts → 독립 원자료 감사 → 별도 authoritative final decision 순서다. Error/late deadline/rename/readback/외부 감사 실패는 성공을 무효화하고 first/secondary errors 및 기존 성공 bytes를 보존한다. 성공 파일 overwrite로 실패를 지우지 않는다. 실제 모든 원자료/로그/감독/receipt/감사/decision 합계4MiB를 확인한다.

E333 whole invalid, startup qmax .04124934>.04, E335 body rejection, E337/E338 technical failures, F1–F8 같은후보 및 M2/UI/F6 상태는 불변이다. 현재 단계의 valid saved response report는 자연스러운 생물학적 보행이나 전체 목표 성공이 아니다.
