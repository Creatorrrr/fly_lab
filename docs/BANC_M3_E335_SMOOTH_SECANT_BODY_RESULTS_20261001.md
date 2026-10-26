# E335: 동일 smooth secant 후보의 실제 신체 검증 — 기각

E334의 단일 theta를 fresh control0에서 한 번 적용해 실제 감각 폐루프 보행을 실행했다. 원본은 유효하게 종료했고 독립 원자료 감사 7,082개 검사도 모두 통과했다. 그러나 같은 후보의 고정 보행 판정은 **rejected**다. 미끄러짐, 연속 보행 구간, 여섯 다리의 유효 주기, 기존 shape 기준에 미달해 채택하지 않는다. 전체 M3 목표는 active이며 자연스러운 생물학적 보행은 입증되지 않았다.

| 동일 자료·고정 구간 지표 | 저장 neutral E318 | E335 실제 감각 폐루프 |
|---|---:|---:|
| smooth training loss | 7.77211086209774 | 7.698483215695015 |
| successor loss | 6.949421492137431 | 6.884383749606794 |
| all15 coordination loss | 0.8226893699603091 | 0.8140994660882208 |
| 좌우 coordination loss | 0.617544157339298 | 0.6262950390085055 |
| legacy shape loss | 0.002180594640846569 | 0.002185159582891293 |
| [.5,1.5) 고정 초기방향 전진 mm | 1.6857973223001235 | 1.7214386350178361 |
| 하중가중 발 미끄러짐 mm/s | 1.9427288085211483 | 2.046158820307313 |
| 최장 연속 전진 구간 s | 0.21 | 0.21 |
| 여섯 다리 complete eligible cycles | 각 0 | 각 0 |

Smooth loss는 0.947331% 개선했지만 물리적인 보행 품질 개선으로 이어지지 않았다. 별도의 held-input E334 loss 7.662243283944549와 live E335 loss 차이는 조건 변화에 따른 관찰이며, 이 차이만으로 감각 피드백의 인과적 실패를 단정하지 않는다. 공통 위상에 정의된 각 다리 reference curve에는 다리별 위상 차가 이미 들어 있다. 따라서 encoded global phase agreement를 여섯 다리가 같은 해부학적 위상으로 움직였다는 증거로 해석하지 않는다.

시작 guard q RMS/max는 0.01773033821234147/0.03973044015521421, tendon RMS/max는 0.0013637570817416666/0.013190716271414396으로 기존 네 기준을 통과했다. 최소 native upright 0.9253928135991221, 비발 접촉 0으로 건강 기준도 통과했다. 고정 forward 기준은 통과했으나 slip≤1.8026436968249955mm/s, bout≥.3s, 각 다리 eligible cycles>0 및 legacy shape 감소 기준은 실패했다. 임계값·평가 구간·후보 theta를 사후 변경하지 않았다.

원본은 body1, 300controls, 6000neural ticks, 15000physics steps를 확인했다. 구성 작업의 320 neural ticks(그 중 capture160)와 settling2000 physics steps는 별도 기록이다. 원본 재시도·추가 gradient·optimizer·theta 탐색·외부 CPG/위상/궤적/root 보정은 0이다. Theta64 SHA d7c8f19833c7364d8f345a5ae4dec2444a4be20606d5b23daf267e15525a7e40, once-patched CSR SHA fabd71c6b7b1153cd060901f88edbb7964403598623c913cee2dc3005de5c249를 유지했다.

실제 Astra6/max 설계 리뷰를 반영했고 GPT6.1Sol/xhigh 구현을 root FIRST에서 검토했다. Producer 연결·pre-context sampler의 결함은 같은 Sol R1에서 수정했으며 root SECOND가 통과했다. 남은 생산 소스 root 수정은 0이다. 생성 CPU144/nativeCUDA38/decoder40 검사를 별도로 통과했으며 원본 신체 결과와 구분한다. 실패 포함 Sol CPU42.849877/90초, root CPU11.7750713/90초·CUDA7.4922571/30초다.

독립 saved 감사의 최초 잘못된 reference 경로 예외와 v2의 12개 입력 불일치를 보존했다. v2는 target cue cut이면 navigation drive도 전부 0이라고 잘못 가정했다. 고정 target encoder는 cue cut에서도 기존 20포트 -5000 및 bilateral DNg100 +5000 tonic drive를 유지한다. Root의 fresh v3 감사는 고정 소스 AST·고정 nodes.jsonl UID에서 이 값을 복원해 7,082개 모두 통과했다. 기존 감각·신경·입력을 변경하거나 원본을 재실행하지 않았다. 이 tonic drive는 기존 공학적 입력 가설이며 생물학적 입력 경로 복원의 증거가 아니다.

원본 supervisor61.64444750000257초, preparation/science56.61143919999449초, 마지막 internal59.34503249999398초다. 소유 process tree peak RAM3,850,399,744B, recursive output834,992,917B이며 GPU pre-context baseline982,515,712B, sampled peak1,814,036,480B, context 포함 추가+양자화832,569,344B다. Device-wide sampling이라 외부 GPU 작업과 샘플 사이 peak는 배제할 수 없다. Sampler와 원본·감사 소유 프로세스의 당시 종료 증거를 보존한다. 현재 PID 재사용으로 이 역사적 종료 증거를 대체하지 않는다.

실패 포함 saved 감사21.988272500006133/30초, 원본 추가0이다. 원본 비용61.64444750000257초는 이미 한 번만 합산해 누적9493.171036399524/9660초, 잔여166.82896360047562초·예약0이다. 이 문서 작성으로 중복 합산하지 않는다. E333 invalid/E330 방향 반증/E327 신체 기각, F1–F8·M2·UI·F6·해부학 provenance 및 신경 단독 계약은 그대로다.

다음 단계는 저장된 E318/E334/E335 command, 실제 관절·접촉·감각 입력의 전달과 위상 진행을 분석하는 별도 E336 설계다. Root 설계→실제 Astra6/max 리뷰→반영→GPT6.1Sol 적정 effort 구현→root FIRST→필요시 같은 Sol R1→root SECOND/남은 root 수정을 따른다. 원본·신체 재실행과 새 후보 예약은 아직 없다.

증거: verification/banc-natural-gait-m3-20260919/parent-e335-original-raw-audit-v3.json, parent-e335-audit-v3-supervision.json, parent-e335-external-completion.json, parent-e335-final-decision.json 및 e335-smooth-secant-body/의 불변 terminal/raw.
