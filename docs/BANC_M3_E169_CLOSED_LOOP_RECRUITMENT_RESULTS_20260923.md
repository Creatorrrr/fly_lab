# E169 현재 폐루프 입력의 전달 계수 재적합 결과

E169 원본은 **incomplete / invalid_not_tested**이며 후보를 채택하지 않았다. 한 번의 적합 뒤 native velocity 재구성 오차가 고정 허용치 `1e-12`를 넘어 `1.3820056210533949e-12`로 나왔다. 최초 오류·exit1·원시 자료를 보존하고 재적합·자동 재실행·계수 clipping·허용치 변경을 하지 않았다.

저장된 후보 자체도 목표에 미달한다. 아래 값은 독립 저장자료 계산이며 실제 보행 개선량이 아니다.

| 조건부 지표 | 기존 E150 | E169 반환 후보 | 기준 |
| --- | ---: | ---: | --- |
| 전체 명령 RMS | .05668638535 | .05500941318 | 기존보다 20% 이상 감소 |
| 전체 velocity RMS | 1.94664326116 | 1.91050114695 | 비악화 |
| 초기 자세 RMS | .01772709268 | .02183437502 | ≤ .018 |
| 초기 자세 최대 오차 | .03945779799 | .04852083341 | ≤ .04 |

명령 오차 감소는 2.9583332%다. 여섯 다리별 command/velocity는 비악화지만 초기 자세는 실패한다. 최소 계수 `-3.0432e-11`, 근육 lower 최솟값 `.49999999999569333`, upper 최댓값 `2.0000000001859224`도 고정 물리 허용 범위 검사를 통과하지 않는다. 이를 근접 성공으로 처리하지 않는다. Clarabel 상태는 AlmostSolved이며 최적 수렴을 주장하지 않는다.

## 완료된 저장 행렬 하위 계산

오류 뒤 별도 E169-D1 진단을 사전 명시하고, 이미 저장된 affine SOCP의 인증만 독립 검산했다. 부모 scalar/행렬/지표/dual 79개 검산 및 실제 gpt-6-sol/high의 별도 감사가 통과했다. 실제 계수 box와 물리 허용오차를 포함한 relaxed max-margin 상한은 부모 계산 `-.21302082063077124`, 별도 감사 `-.21302082063076697`다.

이 값은 **해당 E150 입력에 고정한 anchored affine SOCP의 동일 조건부 목표가 배제된다**는 완료된 수학적 하위결론이다. E169 원본의 invalid 판정을 바꾸지 않으며, native 전달 family 전체·다른 신경 입력·실제 자율 보행 또는 생물학적 가능성의 기각이 아니다. 같은 입력과 같은 함수에서 계수만 반복 선택하는 것은 다음 방법으로 삼지 않는다.

저장 dual의 큰 SOC 가중치는 overall command와 startup에 있었다. 이는 스케일된 해당 인증의 가중치이며, 초기 자세 보호가 생물학적 실패 원인이라거나 이 기준을 완화하면 성공한다는 증거는 아니다.

## 구현·비용·보존

부모 설계 → 실제 gpt-6-sol/high FIRST 구현 → 부모 FIRST 리뷰 → 같은 Sol FIRST 수정 → 부모 SECOND 리뷰를 완료했다. 부모가 최초 solver/candidate raw 저장 실패 시 완료 배열 소실을 재현했고, 같은 Sol이 fallback 보존과 secondary 오류 기록을 수정했다. 생성 시험13개, 독립 생성171개 검산, 부모 오류 재현2개, Ruff check/format이 통과했다. 10개 수치 함수 AST는 FIRST부터 동일하다.

원본 process1/fit1/반환1/자동재시도0, solver 구간2.029005초, 감독33.394461초다. 보수 누적2961.394461/사전 배정3023초이며 이전 비용을 초기화하지 않는다. 원본 보존 해시3565개를 부모가 확인했다. 저장자료 진단은 새 fit 없이12.093655초로 별도 기록했다. 새 neural/physics/EMA/resource/body/StateStore/GraphStore는0이다. M2·활성 시뮬레이션·F6 미노출 상태를 유지했고 전체 F1–F8 목표는 미완료다.

원본 결과: `verification/banc-natural-gait-m3-20260919/e169-closed-loop-recruitment/result.json` SHA `a8b1865ee0f15ccd7d284830a3fb177e7b6dfbe2fd5e9c9b91f10c04bcf5d954`.

최종 소스 SHA `a23ee497834e0064b91d05e5e36b7fd2714e5fca2cedaa0b8748b8e10d85a256`, 시험 SHA `665d10f210998108e5ff8e4ffa6d130980ad22639d1fe19fb2977ff18d2344b0`. 부모 source gate는 `parent-e169-final-source-review.json`, 오류 전후 증거는 FIRST/SECOND review와 각 생성 디렉터리에 있다. D1 설계는 `parent-e169-completed-subtest-design.md`, 독립 계산은 `e169-parent-completed-certificate-verification/`, 별도 감사는 `parent-e169-completed-certificate-audit.json`이다.

다음에는 같은 고정 전달 계수의 재시도 대신, 초기 출력과 활동 이력이 쌓인 뒤의 출력을 구분하는 인과적 전달 구조를 설계한다. E168의 이미 검산된 유계 상태를 사용할 수 있는지 먼저 수학적으로 검토하며, 추가 원본 실행 전에 별도 가설·보호 범위·자원을 동결한다. 이것은 성체 다리의 생물학적 기전이 검증됐다는 주장이 아니다.
