# E140 남은 세 다리의 잔차 기반 제약 복구

2026-09-21 · M3/H140 · 부모 설계 → 실제 Sol/high 구현·첫 수정 → 부모 재리뷰·잔여 수정

E139는 유효한 진전이지만 다섯 다리 공동 복구 가설은 refuted다. LM/RF는 통과했고 LH/RM/RH는 사전40초 한도에서 미달했다. 전체 후보와 M3는 채택하지 않았다. 이 실험은 같은 finite-goal의 남은 전달 병목을 다루며 F1–F8을 대체하지 않는다.

## 선택 근거와 고정 시작점

E139의 각 다리 local3/5 응답에서는 단위 초기 역곡률을 사용하는 L-BFGS-B의 이동이 복구 목적값을 크게 악화시켰다. LH는 초기 selector를 복구한 뒤 손실비0.990055026433으로 원래0.99 guard를 조금 넘겼고 이후 진행이 느렸다. RH/RM는 초기 제약도 남았다. 전체 이력의 반복 복사·JSON 변환 비용은56행에서 저장1회 중앙값0.294729초였으며 같은 내용의 단일 변환 compact 저장은0.047426초였다. 이 저장 측정은 CUDA 배속이나 보행 개선 증거가 아니다.

H140: 이미 저장된 다음 세 시작점에서, 원래 잔차의 정확한 Jacobian과 변수별 크기 조절을 사용하는 reflective trust-region 최소제곱법이 원래 손실 guard와 초기 selector를 세 다리 모두에서 복구한다. 경쟁 설명은 입력 문맥과 모델 bounds 안에서 초기 자세·지속 파형의 상충이 남거나 국소 선형 근사가 복구를 지원하지 않는다는 것이다. 단순히 이전 시간을 늘리는 방법 대신 이동의 생성 방식과 기록 비용을 바꾼다. 결합 변경이므로 각각의 인과 기여나 순수 optimizer 배속을 주장하지 않는다.

| 다리 | 고정 시작 E139 global/local | MN/근육/edge/변수 | E1281 기준 손실 | 시작 손실비 | 시작 최대 startup 비율 |
|---|---|---|---:|---:|---:|
| LH | 25/18 | 63/16/240/272 | 1.3384424326419444 | 0.9900550264327855 | 0.9996034671336481 |
| RM | 47/11 | 63/15/255/285 | 0.8475814708172877 | 0.9889407962109295 | 1.0085201945091316 |
| RH | 56/9 | 63/16/285/317 | 1.095506272045288 | 0.9899949062640104 | 1.0184088900098733 |

각 시작점은 해당 E139 전체 유효 응답 중 복구 목적값이 가장 낮은 응답이다. 같은 값이면 먼저 나온 기록을 고른다. E139의 총56개 선택 노출을 보존하며 새 응답을 보고 시작점을 바꾸지 않는다. LF E1378, LM E1397, RF E13936의 성공은 다시 적합하지 않고 그대로 보존한다.

## 원래 수학과 새로운 풀이

E128의 동일3문맥/160000 four-stage 입력/40000tick/2001sample과 E134 `CudaAdaptationTimescaleEngine`, E139 `evaluate_timescale_leg`를 변경 없이 재사용한다. 기존 해부학적391MN/96근육/1572 fit edge, global IDs, geometry, 관절·발·속도·LTM 손실, 시간 mask와 초기99는 동일하다. x=[delta_E,gain_M,ell_M], gain∈[0,8], ell∈[-log3,log10], T=.3exp(ell), delta는 원래 per-edge bounds다. 이 세 다리의 모든 bounds는 엄격한 lower<upper임을 확인했다.

원래4758 startup residual에 G=.99*각 E1281 손실인3 guard residual `(G-loss_i-prior)/G`를 붙인다. prior=6e-5*sum(delta²)/1572+1e-5*sum(gain²)/96, ell prior0이다. E139와 같은 c_aug, J_aug를 만든다. 새 solver에는 r=min(c_aug,0)와 **c_aug<0인 행의 J_aug만 남긴 CSR Jacobian**을 제공한다. c_aug=0 행의 미분은0으로 정한다. 비용은 원래와 같은 .5*r.T*r이고 gradient=J_active.T*r다. 잔차 가중치·단위·허용범위를 바꾸지 않는다.

설치된 SciPy1.18.1의 `least_squares`를 다리당 한 번 사용한다. method='trf', tr_solver='lsmr', x_scale='jac', loss='linear', ftol=xtol=gtol=1e-12, max_nfev=18, verbose=0이다. tr_options는 regularize=True, atol=btol=1e-12, conlim=1e12, maxiter=300으로 고정한다. Jacobian은 analytic CSR이고 numerical finite difference/추가 neural sweep을 사용하지 않는다. 정규화는 신경 prior가 아니라 내부 선형계 풀이 방식이다. 함수·Jacobian callback은 같은 완료 응답을 캐시한다. SciPy가 bounds의 초기값을 내부로 미세 이동하면 이를 별도 실제 응답으로 계수한다.

선택 조건은 원래 `selector_feasible(ratios)` 및 원래 목적함수 max(loss)+prior ≤G를 동시에 충족하는 것이다. optimizer cap0.99960004와 selector cap0.9998000100000001의 차이를 그대로 유지한다. 첫 충족 raw와 완료 receipt 저장 직후 해당 다리를 중단한다. 새 첫 replay는 선택하지 않으며 최초 새 충족은 local2 이상이다. 정상 반환·한계·기술 오류 후 cached/new 호출과 optimizer 재호출을 모두 차단한다. E139에서 고친 요청·반환·종료 수명주기 및 첫 예외 보존을 유지한다.

## 비교·기록·유효성

순서는 LH→RM→RH다. 각 다리에서 고정 E139 시작점을 먼저 한 번 재현한다. delta/gain/ell, IDs, parameter map, branch는 exact; 모든11 response, full B/initial99는atol1e-8; loss/parts/cotangent/초기 오차·계수·Jacobian/ratios는atol1e-10, rtol0이다. 원래 E128 분모는 저장된 raw와 SHA로 확인하며 다시 신경 실행하지 않는다. 세 시작점은 새 비교에서도 공동 충족하지 않아야 한다. 비교 완료 뒤에만 solver를 요청한다.

모든 실제 응답은 E139와 같은 전체 raw를 먼저 저장한다. r, sparse Jacobian의 data/indices/indptr/shape 및 원래 c_aug는 별도 같은 응답의 잔차 raw로 저장하여 부모가 미분과 비용을 재구성할 수 있게 한다. 성공·미달·큰 이동·부분 실패를 남긴다. 전역/다리 local index와 각 raw hash, 원래 요청·실제 반환·완료·미확정 작업 및 변수별 MN/edge 비용을 기록한다.

새 runner의 저장 함수는 `fit.jsonable`를 한 번 적용한 compact JSON을 사용한다. 기존 모듈의 전역 함수나 writer를 monkeypatch하지 않는다. 전체 정보와 각 durable intent/return/raw/complete 전환을 유지하며 기록을 생략하거나 실패를 덮어 성공으로 만들지 않는다. 새 파일 경로와 새 hash로 기록한다. 실제 저장 호출 수와 시간을 별도로 계수한다. 이미 완료한 E139 기록은 변경하지 않는다.

정상 자원 한도나 정상 solver 종료에서 미충족인 다리는 해당 범위에서 refuted로 기록하고 다음 고정 다리를 진행한다. 기술 오류·불완전/미확정 응답·늦은 저장 오류는 전체를 invalid/not_tested로 종료한다. H140 supported는 세 다리의 유효한 공동 충족을 모두 요구한다. 모두 성공할 때에만 보존한 LF/LM/RF와 원래 edge 위치·muscle 위치·391 membership으로 하나의 여섯 다리 연구 묶음을 만든다. 이 묶음도 candidate_adopted=false이며 실제 전체 BANC·신체·F1–F8 검증이 필요하다.

## 구현 소유권·자원

실제 Sol/high는 새 `tools/fit_c_banc_m3_residual_trust_restoration.py`와 `tests/test_c_banc_residual_trust_restoration.py`만 구현하고 생성 CPU 시험만 한다. 새로운 CUDA 수학이 아닌 검증된 수학·sparse solver·다리 상태·기록 통합이므로 high를 선택한다. 부모는 독립 수치/수명주기/저장 raw 검산과 원본을 실행한다. 첫 수정은 같은 Sol, 부모 두 번째 리뷰 뒤 잔여 수정은 부모가 한다. 기존 E128/E134/E136/E137/E138/E139·제품·M2·UI 소스는 보존한다.

기존1977초 배정과 실제1890.4416891초를 보존한다. 새 세 다리 방법에1891+210=2101초의 사전 누적 상한을 추가 배정한다. 원본1프로세스/재시도0, optimizer 최대3회, 다리당 replay 포함18 unique/전체54 응답이다. 다리55초/reserve3초, fit block165초, 내부180초/reserve10초, 외부190초, 감독·최종 manifest210초다. 개별 실행300벽초라는 기존 상한 안이다. Raw4GiB, 시작 여유12GiB, CPU1/GPU1이다. 생성 tiny CUDA20tick/.005조건부초/1bank를 한 번 별도 계수한다.

최대 원본 작업은1620조건부초/6480000 context tick/408240000 scalar MN tick/6739200000 edge-stage tick/54bank/324000decoder endpoint/295542 geometry forward 및 같은 backward row다. 부모·Sol의 생성 CPU 및 저장 자료 검산 비용은 별도 기록한다. 이전 실패나 사용량을0으로 초기화하지 않는다. full391 batch/full158706 BANC/실제 신체/LP/F6/UI는 이번에0이다.

다음은 같은 완성 묶음의 전체 조건부 일관성과 실제 BANC·감각·신체10초 폐루프 확인이다. 목표는 여전히 원래 자연 보행 M3이며 이 잔차 적합만으로 성공 선언하지 않는다.
