# E139 나머지 다섯 다리의 시간상수 기반 초기 자세 복구

2026-09-21 KST · M3/H139 · 부모 설계 · 실제 Sol/high 구현·첫 수정 · 부모 리뷰·잔여 수정

## 현재 근거와 선택

직전 E138은 progress다. 전체158706 BANC/CUDA의 새 근육별 시간상수는 기본값 whole-state/decoder exact, 비균일값의 실제 효과, 저장 미래 exact를 통과했다. 실제 신체는 실행하지 않았으며 F1–F8은 계속 미완료다. E138까지 원래 프로세스의 보수적 누적은1716.3425233999733초다.

E138의 읽기 전용 교차 다리 감사는 E128의93개 raw와 SHA를 검토했다. LM/LH/RF/RM/RH에서 원래 목적함수가 가장 낮은 기록은 각각15/16/16/17/15번이다. 모두 각 E1281번 대비1% 이상 목적함수가 낮지만 초기 selector는 실패했다. LF에서 E137의 같은 복구 방법은 원래 개선을 유지하면서 초기 selector를 통과했다. 이제 동일한 복구 원리를 다른 다리의 해부학적 차원으로 일반화한다. 새로운 신경 방정식이나 CUDA kernel을 추가하는 단계가 아니다.

H139: 아래 고정 시작점에서 기존 edge/gain/log-tau bounds 안의 제약 복구가, **다섯 다리 각각**의 초기 selector를 충족하고 각 원래 E1281번 대비 목적함수1% 이상 개선을 유지한다. 가장 강한 경쟁 설명은 다른 다리에서는 지속 파형과 초기 자세의 비선형 상충 때문에 LF의 성공이 일반화되지 않는다는 것이다. 모든 다리에서 단일 시간상수/gain 격자를 반복하는 방법 대신, 이미 존재하는 개선점을 고정하고 민감도에 근거해 복구한다.

| 다리 | 시작 E128 raw | 기준 목적함수(E1281) | 시작 목적함수 | MN / 근육 / edge / 새 변수 |
|---|---:|---:|---:|---|
| LM | 15 | 0.5411264888902965 | 0.5270208864229483 | 63 / 15 / 281 / 311 |
| LH | 16 | 1.3384424326419444 | 1.3247807673715613 | 63 / 16 / 240 / 272 |
| RF | 16 | 0.7363699081575308 | 0.7270290438136352 | 70 / 17 / 238 / 272 |
| RM | 17 | 0.8475814708172877 | 0.8381755115338686 | 63 / 15 / 255 / 285 |
| RH | 15 | 1.095506272045288 | 1.0838524600879194 | 63 / 16 / 285 / 317 |

시작점은93개 기존 기록 전체에서 선택했음을 명시한다. 실행 결과를 보고 시작점이나 기준 분모를 교체하지 않는다. LF는 E1378번을 보존하며 이 실험에서 재적합하지 않는다.

## 동역학·목적·인덱스 계약

E128의 동일한 E110/E113/E121 세 문맥, 160000개 four-stage 입력/40000 neural tick/2001 sample, 391MN/96근육/69 source/1572 fit edge와 원래 geometry를 사용한다. E134 `CudaAdaptationTimescaleEngine` 및 기존 frozen mapping을 재사용한다. T=.3*exp(ell), ell∈[-log3,log10], gain∈[0,8], delta는 원래 per-edge bounds다. gain/시간상수는 실측 생리가 아닌 연구용 가설이다.

변수 x=[delta_E, gain_M, ell_M]이며 크기는 E+2M이다. 근육·운동신경·edge를 dummy padding하지 않는다. 해당 다리의 motor/muscle/edge mask와 원래 global IDs를 검증하고, geometry leg, joint slice[7*leg:7*leg+7], 목표/scale/ltm leg index를 함께 적용한다. 서로 다른 다리의 근육 수15/16/17과 edge 수를 LF의17/273으로 고정하지 않는다. CUDA B의 열은 기존 edge_muscle, gain 근육, log-tau 근육 순서다.

원래 목적함수는 세 문맥의 max(loss)+prior다. prior=6e-5*sum(delta²)/1572+1e-5*sum(gain²)/96이며 ell prior는0이다. loss의 관절/발/속도/LTM 항, scale, fit-time mask, 초기99 및 평가기는 변경하지 않는다.

원래4758개의 startup signed/RMS residual과 Jacobian에, G=.99*해당 다리 E1281 목적함수인3개 guard residual (G-loss_i-prior)/G를 붙인다. R=.5*sum(min(c_aug,0)²), gradient=J_aug.T@min(c_aug,0)다. 기존 `fit.constraint_values`/`selector_feasible`를 재사용한다. optimizer cap .99960004, selector cap .9998000100000001은 그대로다. selector가 엄격 optimizer residual의0을 요구하지 않는다는 차이도 보존한다.

L-BFGS-B maxiter100/maxls20/ftol1e-12/gtol1e-10을 유지한다. optimizer는 다리당 한 번만 요청한다. delta/gain/ell bytes별 완료된 응답만 캐시하며 최초 공동 충족 raw와 완료 receipt 저장 직후 해당 다리를 정상 중단한다. 한계/성공/오류로 끝난 캐시는 다시 열지 않는다. 최초 충족 뒤 추가 평가나 실패 기록 제외로 결과를 바꾸지 않는다.

## 원래 응답을 확인하는 순서

다리 순서는 LM→LH→RF→RM→RH다. 다리마다 최대24개의 실제 응답에 아래 두 확인을 포함한다.

1. E1281번의 delta/gain과 ell=0으로 baseline replay를 한다. 원래 raw에 있는8응답 및 branch, identity, B_initial99의 기존 E+M 열, loss/parts/초기 오차/계수/cotangent/ratios/loss_jac 기존 열을 비교한다. branch/IDs/요청 값은 exact, 신경 응답과 B는atol1e-8, loss와 loss_jac/기타 수치는atol1e-10, rtol0이다.
2. 표의 고정 최저 목적함수점의 delta/gain과 ell=0으로 start replay를 한다. 해당 E128 raw에는 전체 신경 응답이 저장되지 않았으므로 있다고 주장하지 않는다. 실제로 있는 IDs/값/old parameter map prefix, B_initial99 prefix, loss/parts/초기 오차/계수/cotangent/ratios/loss_jac prefix를 같은 기준으로 비교한다. 전체 새 응답과 B는 이번부터 저장한다. 이 시작점은 selector 실패임을 재확인한다.
3. 두 비교가 완료된 뒤 동일 시작점에서 복구 optimizer를 한 번 요청한다. 시작 응답의 캐시 재사용은 새 응답으로 세지 않는다. baseline은 최적화 후보 캐시에 포함하지 않는다. 최초 공동 충족은 전체 local evaluation3번 이후여야 한다.

원본 raw에는 각 응답의 모든11개 E134 field, branch_bits, full B 및 initial99 B, index/요청값/geometry feet, 손실과 cotangent·미분·초기 계수를 저장한다. 원시 관측은 수치 판정보다 먼저 보존한다. 원래 baseline/start raw를 덮어쓰지 않는다. 주기적인 FD 추가 sweep이나 원래 적합의 재시작은 하지 않는다.

개별 다리의 정상 한계 종료는 그 범위에서 refuted로 기록하고 다음 고정 다리를 진행한다. 기술 오류/불완전한 응답/미확정 작업/late-save 오류는 전체 실행을 invalid/not_tested로 중단하고 모든 완료·부분 raw와 첫 예외를 남긴다. 서로 다른 MN 수에 LF69를 곱하지 않으며 response_progress의 동기화·요청·완료량으로 실제 일을 집계한다.

## 구현·검증·소유권

새 `tools/fit_c_banc_m3_cross_leg_timescale_restoration.py`와 `tests/test_c_banc_cross_leg_timescale_restoration.py`만 실제 Sol/high가 구현한다. 검증된 수학의 가변 차원·다리 인덱스·optimizer·비용 기록 통합이므로 high를 선택한다. 새로운 CUDA 수학은 도입하지 않는다. 기존 E128/E134/E136/E137/E138 및 제품·M2 소스는 수정하지 않는다.

E136의 generic `register_intent`, `evaluate_with_receipts`, geometry/transfer helpers와 E137의 검증된 복구·캐시 수명 절차를 재사용할 수 있다. 단, LF 전용 평가/shape와 `WORK_PER_EVALUATION`·69MN 계수를 그대로 재사용하거나 전역 monkeypatch하지 않는다. 새 코드에서 필요한 dimension-dependent 부분을 명시한다. shared fixture/geometry는 한 번 준비하고 각 engine의 고정 전송과 해제를 실제로 기록한다.

Sol은 생성 CPU 배열/작은 fixture로만 개발한다. 원본/CUDA/신체/LP/적합 실험을 실행하지 않는다. 다리 인덱스와15/16/17 근육 차원, 비LF target/scale/ltm, 미분(독립 차분/complex-step의 적절한 구간), first replay prefix, 다리별 상이한 작업량, first qualifier/terminal cache, pre-entry·post-return·final-save 오류를 시험한다. 생성 neural 모델은8cell 이하이며 원래 입력을 사용하지 않는다. 부모가 별도 source·수치·저장 raw 검산을 담당한다. 첫 수정은 같은 Sol, 두 번째 리뷰 뒤 잔여 수정은 부모가 한다.

## 자원과 판정

E138의1742초 배정 및 실제1716.3425233999733초 기록은 바꾸지 않는다. 기존 finite-goal 안의 새 다섯 다리 단계에 사전 추가 배정을 기록한다. 보수적 시작1717+새 감독260=1977초를 새로운 누적 상한으로 둔다. 이는 이전 실패나 사용량의 초기화가 아니다. 상위 개별 로컬 실행300벽초 한도 안이다.

원본1프로세스/재시도0, 다리5개/optimizer 최대5회, baseline/start를 포함해 다리당 최대24 unique/전체120 응답이다. 각 다리 engine/확인/fit40초, leg reserve3초, 전체 fit block210초, 내부230초/외부240초/감독 및 최종 manifest260초, process reserve10초다. 저장 공간8GiB/실행 전 여유12GiB 이상, CPU thread1/GPU1이다. 부모 생성 시험·읽기 검산 비용은 따로 기록한다. 기록하지 않은 비용을0으로 취급하지 않는다.

생성 tiny CUDA20tick/.005조건부초/1bank setup을 원본 프로세스에서 한 번만 별도 계수한다. 최대 원본 작업량은3600조건부초/14400000 context tick/927360000 scalar MN tick/14964480000 edge-stage tick/120bank/720000decoder endpoint/656760 geometry forward 및 같은 backward row다. 실제 반환/동기화/미확정·부분 비용과 모든 보존 raw의 크기를 함께 기록한다. 전체391 conditional batch/full158706 BANC/실제 신체/LP/F6/UI 실행은0이다.

H139 supported는 다섯 다리 모두의 유효한 공동 충족 응답을 요구한다. 유효하게 다섯 bounded fit을 마쳤지만 하나라도 없으면 refuted다. 개별 성공은 보존하되 전체 성공으로 합치지 않는다. 모두 성공하면 원래 global edge IDs→1572 배열과 muscle indices→96 gain/ell,391MN membership를 검증하여 E1378번 LF와 한 연구용 파라미터 묶음을 만든다. 어느 다리라도 없으면 완전한 묶음을 만들지 않는다. 이 묶음도 `candidate_adopted=false`이며 최종 M3가 아니다.

다음은 같은 파라미터 묶음의 전체 조건부 일관성과 실제 BANC·감각·신체 폐루프10초를 새 계약으로 검증하는 것이다. 전체 F1–F8, 원래 M2·실패 판정·F6 미노출과 활성 UI는 유지한다. 추후 생물학적 동등성이나 자연 보행 판정을 이번 조건부 적합으로 대체하지 않는다.
