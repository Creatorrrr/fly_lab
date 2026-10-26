# E137: 목적함수 개선을 보존하는 초기 자세 복구

2026-09-21 KST · 부모 설계 · 원본 실행 전 동결 · M3 pending

이전 E136 턴은 progress다. 새 CUDA 적합 45개를 독립 검산했고, 목적함수가 감소한 새 지점은 모두 초기 자세 제약을 위반함을 확인했다. H136 refuted와 기준을 유지한다. 가장 낮은 목적함수의45번은 1.107858426943197로 기준 1.137195003910538보다2.579731% 낮지만, 초기 비율 1.000470916414981로 selector 0.999800010000000를 넘는다. 엄격 startup 제약 위반은8개이며, 이미 원래1% 개선선 1.125823053871432보다 좋은 손실을 확보했다. 따라서 목적함수를 더 낮추는 E136을 재개하지 않고, 초기 제약 위반의 복구를 별도 검증한다.

## H137과 판정

H137: 고정 시작점 E13645에서 기존 307변수 동역학과 bounds를 유지한 startup 복구가, E1361 기준의 원래 worst objective+prior를1% 이상 낮춘 상태에서 초기99 selector 제약을 만족한다. 가장 강한 대안 설명은 작은 수치상의 위반이 초기 상태/지속 파형 사이의 실제 비선형 상충이어서, 복구하면 그 개선 여유도 사라진다는 것이다.

시작점은 지금 저장된45번 하나이며 결과를 보고35번이나 다른 초기점으로 바꾸지 않는다. 첫 실제 replay의 전체11응답/branch/307열 B/ID/시작오차/손실/미분이 E13645와 일치해야 한다. 응답/B 허용차1e-8, 손실/parts/ratios/loss_jac1e-10, ID/branch/파라미터 exact다. 이 검사를 통과하기 전에 optimizer를 요청하지 않는다. E1361의 목적함수를 분모로 유지하며 복구 시작점45의 목적함수로 분모를 바꾸지 않는다.

원래4758 startup signed/RMS 제약 c_start 및 Jacobian을 사용한다. 추가3개 행은 c_guard_i=(G-loss_i-prior)/G, G=.99*E1361_original_objective=1.1258230538714322로 고정한다. 원래 prior=6e-5*sum(delta²)/1572+1e-5*sum(gain²)/96이며 ell prior는0이다. 복구 목적은 R=.5*sum(min(concat(c_start,c_guard),0)²), 미분은 J_aug.T@min(c_aug,0)다. guard 미분은 -(loss_jac_i+prior_grad)/G다. guard는 원래 목적함수 개선을 보존하며 목적함수 자체와 기준을 바꾸지 않는다.

L-BFGS-B의 maxiter100/maxls20/ftol1e-12/gtol1e-10을 E131 복구 방식과 같이 사용한다. 직접 delta273/gain17/ell17 좌표이며 bounds는 기존 delta, gain[0,8], ell[-log3,log10]이다. 새 prior/정규화/파형/연결은 추가하지 않는다. E136 평가기/receipt/계산량/최종 오류 처리와 E134 CUDA 계산기를 재사용하며 기존 전역 함수를 monkeypatch하지 않는다. E136의 시작점 feasible 강제·SLSQP·result selector는 복구 문제와 다르므로 그대로 호출하지 않는다.

Primary는 모든 완전 기록 중 **초기 selector 통과 및 objective/baseline<=.99를 동시에 만족하는 첫 원본 index**다. 그 raw와 loss/ratio가 저장되고 cache 완료가 기록된 직후 정상 bounded stop한다. startup만 통과해도 guard가 미달이면 계속한다. 조건을 충족하는 기록이 없고 모든 실행/비용/참조가 유효하면 H137 refuted, 있으면 supported 및 LF 조건부 proposal만 채택한다. 기술 오류/누락/미확정/늦은 저장 실패는 invalid/not_tested이고 실패를 제외한 채 성공을 선언하지 않는다. 전체 M3/F1–F8 판정은 별도이며 완료로 승격하지 않는다.

## 범위와 자원

새 `tools/fit_c_banc_m3_timescale_restoration.py`와 `tests/test_c_banc_timescale_restoration.py`를 실제 Sol/high가 구현한다. 부모가 설계·독립 수치/원본 검토를 하고, 첫 리뷰의 수정은 같은 Sol/high, 재리뷰의 잔여 수정은 부모가 한다. 이미 검증된 동역학을 재사용하는 307변수 제약 복구 통합이므로 high를 선택한다. Sol은 작은 generated CPU fixture만 시험하며 원본/LP/신경/CUDA/신체를 실행하지 않는다.

원본1프로세스/복구 fit1회/최대18 unique(첫 replay 포함)/재시도0이다. 고정 공유 입력은 한 번 올리고 매 평가의 full B/11응답/branch/cotangent/손실/초기감도/fit feet/raw를 E136 형식으로 보존한다. 평가별 restoration_value/gradient_norm/guard ratios/선택 자격도 기록한다. 원본 raw 한도1GiB. 초기20tick/.005초/1bank tiny CUDA는 원본 프로세스 내부의 별도 setup 한 번만 허용한다. 추가 FD sweep은 없다.

fit40초, leg reserve3초/process reserve5초, 내부48초/외부52초/supervisor56초다. E136까지 보수적 누적1682.1102980999858초를 보존하고 새 사전 기준1683+56=1739<=1742초로 배정한다. 이전1530/1742 계약 및 E136의 실제 실패를 변경하지 않는다. 최대18평가는540조건부초/2160000 context tick/149040000 scalarMN tick/2358720000 edge-stage tick/18bank/108000decoder endpoint/98514 geometry forward 및 같은 backward row다. 요청·진입·반환·동기화·raw·partial·cache·optimizer·오류·최종 manifest를 구분하고 모든 실제/미확정 비용을 남긴다. 준비/생성시험/독립 NumPy 검산 비용은 별도다.

전체391/158706 BANC/신체/F6/UI는0이다. M2 263파일과 미노출 확인 자세를 보존한다. 성공하면 다른 다리의 동일 적합 가능성과 동일 후보의 native/body F1–F8로 진행한다. 미달하면 같은 fit의 재시작/평가 추가/기준 완화를 하지 않고 저장된 복구 변화에서 다음 설계를 판단한다.
