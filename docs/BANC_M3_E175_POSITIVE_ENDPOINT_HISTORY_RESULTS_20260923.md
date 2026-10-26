# E175: 양수 이력 전달도 저장 목표와 전체 보호 조건을 함께 충족하지 못함

단일 원본 실행은 유효한 저장 신호 진단으로 완료됐지만 후보는 채택하지 않았다. 같은 저장 신호에서 양수 계수를 활동 이력에 따라 증가하거나 감소하게 해도, 원래 command 오차 20% 감소와 모든 보호 조건을 함께 만족하는 가족은 유한 정밀도 dual 검사에서 배제됐다. 실제 전체 BANC–신체 보행 또는 생물학적 불가능성의 증거로 확대하지 않는다. F1–F8과 전체 목표는 미완료다.

| 항목 | 같은 반환점의 결과 | 원래 기준 |
|---|---:|---:|
| command RMS | .051359655849, baseline .056686385350 | baseline의 .8 이하 |
| command 오차 감소 | 9.396841% | 20% 이상 |
| velocity RMS | 1.784932690788, baseline 1.946643261158 | baseline 이하 |
| startup RMS | .020385710775 | .018 이하 |
| startup 최대 절대값 | .045301579501 | .04 이하 |
| 실제 MN/근육 lower/upper 위반 | 0/0/0 | 원래 물리 허용치 1e−12 |
| true 영역 projected dual upper | −.132539330356 | 음수면 t≥0 배제 |

새 좌표는 c(F)=(1−F)a+Fb이며 a,b는[0,2]다. 실제로 b<a인 branch가566개로 감소 방향이 사용됐다. 반환 계수 범위는 약1.0e−7~1.99999901이며, 신경 노드/해부학적 연결이나 새로운 외부 위상 입력을 추가하지 않았다. 모든 다리별 command와 velocity 오차가 baseline보다 작았지만 startup 및 전체 command 목표가 실패했으므로 일부 개선을 보행 후보 통과로 합치지 않는다.

원래 E170 QR와 sparse 행을 좌표 변환하고 모든15 SOC, startup pointwise, 네 endpoint min/max의 MN floor와 근육 예산을 유지했다. Inner guard는 수치용이며 true dual에서는 box g, envelope eta, 물리 delta+원래1e−12를 복원했다. 물리 실제 최소 slack은 MN1.29985e−7/lower1.60011e−7/upper1.59979e−7이다. 원래 기준의 위반을 사후 허용한 결과가 아니다.

Clarabel AlmostSolved/47iteration, t=−.132539487479가 반환됐다. Inner/true linear 최소 slack은 startup 행28383의 약−4.14e−11이며 보존했다. 모든15 primal SOC slack은 양수다. 이 반환점 자체를 완전 가능점으로 인증하지 않는다. Cone으로 각각 투영한 dual과 true box support를 사용한 음수 상한이 배제의 근거이며, 엄밀한 구간 산술 증명은 아니다. 직접 startup norm과 QR norm의 차이는 반환점 약2e−15, baseline 약2.4e−15로 동결 허용치 안이다.

부모 설계 후 실제 gpt-6-sol/high가 첫 구현과 첫 수정을 수행했다. 부모 FIRST 리뷰에서 CLI import, direct/QR 불일치의 validity 처리, 작은 행렬 원소의 허용치, 실제 endpoint와 보조변수의 판정 구분, 실패 자료 보존을 교정하도록 요청했다. SAME Sol 교정 후 부모 SECOND 리뷰를 완료했고 남은 서식 수정은 부모가 수행했다. AST가 동일함을 확인했다. 생성 테스트16개, 부모의 FIRST/SECOND 수치·전체 실행 검사445개, 원본 저장 검산177개가 통과했다. 별도 gpt-6-sol/high의 원시 자료 우선 감사에서도 true/inner upper와 물리 판정이 일치했다. 최초 CLI 실패와 생성 검산 기록도 남겼다.

원본은 process1/fit1/retry0, 감독25.118984599947초/32,045,717bytes로 사전 한도 안에 끝났다. 이전 원본비용을 합친 누적은3085.011408499908/3202초다. 부모 저장 검산2.800511999987초와 생성 검산 비용은 원본 적합 비용과 별도로 기록한다. 새 feature/QR/SVD/history update/신경/물리 실행은0이며 M2·활성 UI·미사용 F6는 유지했다. 이 gate는 소비됐고 같은 실행을 재시도하지 않는다.

원시 자료와 최종 판정: `verification/banc-natural-gait-m3-20260919/e175-endpoint-history/`, `e175-parent-saved-verification/result.json`, `parent-e175-saved-certificate-audit.json`, `parent-e175-final-review.json`. 다음 방향은 `docs/BANC_M3_AFTER_E175_DIRECTION_REVIEW_20260923.md`에 기록한다.
