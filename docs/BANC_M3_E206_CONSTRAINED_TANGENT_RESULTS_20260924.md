# E206 결과: 명시적 제약의 국소 예측도 후보 미달

실제 gpt-6-sol/high 구현 → 부모 FIRST의 정규화/비유한 반환 검증 지적 → 같은 Sol 첫 수정 → 부모 SECOND를 완료했다. 부모 독립 좌표·부호·SOC 검산과 생성 math/mock 시험 4개/Ruff가 통과했다. 실제 생성 Clarabel 풀이 한 번은 해석해 .5와 일치했으며 반복하지 않았다.

원본 CPU Clarabel 한 번이 35 iteration, solver 5.340861300초에 `Solved`로 종료했다. 원자료 기반 출발/다리별 비악화/계수/신뢰영역 조건은 통과하지만 전체 명령 RMS는 0.054830025377rad, 원래 대비 개선 3.274790%로 20% 기준에 미달한다. 결과는 `bounded_linearized_search_no_proposal`이다. 선택 u의 norm 0.999998999997와 startup RMS 0.017999982000/max 0.039999960000가 각각 사전 한도에 가깝다.

부모는 저장된 신경 접선에서 D를 독립 재구성했고 최대 차이 1.137e-13, 14 SOC의 실제 residual norm 및 QP 목적함수/기울기/변수 변환을 독립 확인했다. 최대 stationarity 오차 3.469e-16, primal 방정식 절대 오차 4.14e-08, solver 보고 정규화 primal 잔차 2.75e-10다. 수렴 표기와 직접 후보 판정은 구별했다.

선형 rate 예측은 8989개 제어 표본에서 음수이며 최소 -5.947855594Hz다. 이는 사후 추가 합격 기준이 아니며, 이 근사가 실제 비선형 반응/생물학적 발화가 아님을 보여주는 진단이다. 이 해를 신경망이나 신체에 적용하지 않았다. 단위 trust 안의 국소 접선 결과를 전체 비선형 가족의 불가능성으로 일반화하지 않는다.

별도 읽기 감사에서 자유 MN 343개 중178개는 선택된 rate/adaptation/모든 민감도/7계수가 정확히0이었다. 같은 응답식의 허용 계수 w0=1을 초기 단계에 대입하면 h=0.343297078816, 순간 목표 발화율 약66.08Hz로 양수가 된다. 따라서 현재 영 기울기는 전체 허용 범위에서 모집 불가능하다는 뜻이 아니다. 이 증거는 정적 대입이며 이후 궤적이나 보행은 검증하지 않았다. 후속은 이 큰 값을 그대로 적용하는 것이 아니라 초기 순간 목표 .01Hz의 작은 문턱 통과를 사전 고정해 실제 CUDA에서 지속적 학습 신호 여부를 한 번 비교하는 설계다. 다른 뉴런/모델/목표/최종 기준은 유지한다.

원본 감독 10.760998199999초, 누적 5412.115466699706/5564초, 잔여 151.884533300294초다. actual runner의 Windows peak_wset는 769372160bytes로4GiB 이내다. 감독기가 기록한 4521984bytes는 venv launcher만의 값이라 전체 실험 메모리로 사용하지 않는다. 향후 감독기는 owned child도 측정한다. 원본 retry0, 잔류 owned launcher/자식0, 새CUDA/조건부신경/전체BANC/물리0이다. 부모와 Sol 생성·정적 검산 비용은 별도다.

근거: `verification/banc-natural-gait-m3-20260919/e206-constrained-tangent/`, `parent-e206-source-review.json`, `parent-e206-output-review.json`, `parent-e206-resource-review.json`, `parent-e206-zero-gradient-audit.json`. M2·E150·UI·원본 실패 보존, F6 미노출, 미채택, F1–F8 active·미완료다.
