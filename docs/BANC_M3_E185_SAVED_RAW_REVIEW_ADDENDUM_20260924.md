# E185 종료 판정과 저장된 관측의 분리 검토

원본 실행1회는 600 control/12000 neural/30000 physics 및 force30000 표본을 모두 저장했다. E181의 원래 비교와 전체 control 필드 비교는 차이0, startup·clock·health·finite 모두 통과했다. 그러나 E185 driver는 child exit0 직후 `unexpected_live_descendant`를 기록해 **실패로 종료했다**. 이 판정·모든 원자료·원본 소스 해시는 바꾸지 않는다. 재실행0을 유지한다.

실패 위치는 종료 감독이다. `supervise_child`는 `process.wait()` 뒤 한 번의 `_live_owned` 양성으로 상태를 영구적으로 바꾸며, 이어지는 최종 정리에서 실제 남은 PID가 없어도 상태를 회복하지 않는다. 이번 최종 정리 기록은 terminated/alive/errors가 모두 빈 목록이다. 부모의 새 PID+생성시각 확인도 전부 absent다. 이는 종료 과정 중 순간적인 관측 차이와 부합하지만, 당시 첫 검사에 어떤 PID가 잡혔는지는 저장하지 않았으므로 그 PID나 정확한 OS 원인은 단정하지 않는다.

새 힘 값의 분석 전, 부모가 **저장된 자료의 검토**를 다음 조건 모두에 한정한다. 이는 원래 driver를 PASS로 바꾸거나 후보를 채택하는 절차가 아니다.

1. 원래 driver 실패 원인은 `unexpected_live_descendant` 하나이고 child exit0, supervisor error/cap reason 없음. 최종 terminated/alive/errors가 모두 빈 목록이며, 새 PID+생성시각 감사도 모두 종료를 확인한다.
2. child 완전한 3초/초기·건강·시계 통과, 실제 전체 BANC CUDA, 원본 표본 누락0/원자료 쓰기 오류0, 양쪽 exact 비교 차이0.
3. E182의 두 readiness와39개 원본 resource 관측에 queue중복/unknown/오류 없음. child 전체88.91초<100, outer98.56초<130이며 실행·해석 비용 누적은 승계한다.
4. E185 원본 파일 전체와 원래 freeze/review/source/기준 E182 해시를 새 review gate가 묶는다. 원본 event/분석 공식·질량/좌표 검증·F1–F8 및 원래 observer의 force 의미는 바꾸지 않는다. 새 모델/신경/물리0, 최적화0, 원본 재실행0. 저장된 자료 분석1회만20초 내 허용하며 실패해도 반복하지 않는다.

부모의 독립 raw checker는 원래 E185 사전 분석 코드를 보존한 별도 파일에서 종료 영수증의 위 조건만 추가 판정한다. 그 뒤 같은8사건/41표본·모든 다리·전체 fly/COM 집계를 실행한다. 승인 대상은 기록된 관측의 사용뿐이다. 과거 감독 실패는 계속 보고하고 자연 보행 성과나 생물학적 증거로 승격하지 않는다. 합성 영수증에서 live/실제강제종료/errors/child실패를 각각 거부하는지 원본 분석 전에 확인한다.

종료 검사의 일반적인 코드 개선은 새로운 원본 실행 없이 검토할 수 있으나, 이번 실패 영수증을 수정해서 재사용하지 않는다. 저장된 자료로 과학적 질문을 풀 수 없으면 그 한계 그대로 기록한다.
