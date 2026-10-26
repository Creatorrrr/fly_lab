# E174 이후: 시작 응답과 지속 응답을 구별하는 양수 이력 전달

E173은 고정 양수 계수로 저장 command 오차를 줄일 수 있음을 보였지만 startup가 크게 악화됐다. E174는 startup 보호만으로도 이 static 가족의 20% 개선 목표가 배제됨을 확인했다. 근육 보호 비교는 미결이므로 근육 예산이 문제가 없다고 단정하지 않는다. 고정 static 계수를 다시 적합하거나 startup 한도를 낮추는 방법은 반복하지 않는다. 같은 전체 BANC 후보의 F1–F8 완료가 여전히 목표다.

다음 부모 설계는 기존 신경 활동에서 계산하는 이력 상태 F를 유지하면서, 실제 전달 계수가 양수인 두 endpoint 사이에서 증가와 감소를 모두 허용하는 구조를 검토한다. 외부 시간표·위상·CPG·teacher를 실행 입력에 추가하지 않는다.

`c(F) = (1−F) a + F b`, `a,b∈[0,2]`, `F=2(1−R)∈[0,1]`로 표현한다. E170 좌표로는 `alpha=a`, `beta=b−a`다. beta가 음수일 수 있어도 모든 F에서 실제 전달 계수 c(F)는 양수 범위에 남는다. 이는 음수 근육 힘을 도입하는 것이 아니다. 초기/후기 endpoint를 같은 신경 활동의 이력으로 구별하는 공학적 전달 가설이며 성체 다리의 측정된 기전이라고 주장하지 않는다.

읽기 전용 gpt-6-sol/medium 감사와 부모 소스 확인에서 기존 비교와의 차이를 확인했다. E168은 `w0[1−lambda(1−R)]`의 제한된 비례 감쇠이며, E170 좌표로는 `alpha=w0`, `beta=−lambda*w0/2`다. factor 2를 생략하지 않는다. E170은 자유로운 alpha/beta를 적합했지만 두 값 모두 비음수로 제한했으므로 감소 endpoint를 포함하지 않았다. 두 이전 기각 결과는 새 endpoint 전체 가족을 배제하지 않는다.

E170에 저장된 QR에는 alpha와 F×beta 열이 이미 있다. 따라서 endpoint 좌표의 행렬은 `(M_alpha−M_beta) a + M_beta b + offset`으로 표현할 수 있으며 새 특징/QR/신경 실행 없이 먼저 저장 가족의 가능성을 검토할 수 있다. 원래 MN floor와 근육 예산을 유지하려면 pair별 모든 a/b low/high endpoint의 최소값과 최대값을 보호해야 한다. E170의 min(alpha), max(alpha+beta)를 음수 beta에도 그대로 사용하는 것은 잘못이다.

후속 E175의 정확한 변수/행/inner guard/true dual 복원/입력 hash/자원/판정 규칙은 원본 수치 계산 전에 별도 설계로 고정한다. 모든 원래 startup RMS/max, 전체와 다리별 command/velocity 보호를 함께 복원한다. 미세 제약 위반을 사후 허용하지 않고, 필요하면 원래 true 영역을 보존하는 수치용 inner 보호를 사전에 정의한다. 성공하더라도 먼저 native 전달 일치와 실제 전체 신경·신체 폐루프 검증이 필요하다. 저장 목표가 다시 배제되면 해당 endpoint 가족의 계수 탐색을 멈추고 신경 모집 또는 활성 동역학 변경으로 방향을 전환한다.

현재 E175는 방향 검토 단계이며 새 원본/fit은 0이다. 구현은 실제 gpt-6-sol이 맡고 부모 설계→Sol 첫 구현→부모 FIRST 리뷰→같은 Sol 첫 수정→부모 SECOND 리뷰 및 잔여 수정 순서를 유지한다. E174 gate는 소비됐으며 재시도하지 않는다. M2, 활성 UI, F6 미사용 초기상태와 기존 실패를 보존한다.

소스 근거: `tools/fit_c_banc_m3_resource_attenuation.py:629`, `tools/banc_m3_history_recruitment.py:92`, `tools/fit_c_banc_m3_history_recruitment.py:532`, `:620`, `:863`. 이 검토는 소스에 정의된 모델 범위에 대한 것이며 새로운 보행 효과의 관측은 아니다.
