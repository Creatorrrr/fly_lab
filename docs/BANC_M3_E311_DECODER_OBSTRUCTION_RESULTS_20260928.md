# E311 — 고정 LM 경로의 내부 여유 충돌 확인

2026-09-28. E287의 저장된 변경 E277 응답과 제약을 재구성한 진단은 유효하다. 시점별 가변 근육 활성을 독립 선택하는 더 넓은 box에서도 **고정 근육 19 / MN92104만 연결된 LM joint8의 6행**은 원래 내부 peak 기준을 만족할 수 없다. 이 source를 그대로 둔 variable77 비선형 모집 적합은 진행하지 않는다. 실제 보행이나 다른 source의 불가능성을 뜻하지 않는다.

Control36–41의 고정 오차는 −.03866016253541499, −.03951792145025396, −.039981954442789736, −.03999598786656486, −.03955469572680134, −.03869190668869857 rad다. 해당 관절의 가변 근육 해부학적 지원은0이며 다른 가변 MN 읽기로 바꿀 수 없다. 모두 E287 내부 peak .03838315744033522를 넘지만 **최종 native .04를 직접 넘는 행은0**이다. 내부 여유는 E279의 조건부→실제 최대 차이 .0015168425596647772와 .0001을 뺀 값이었다. E279의 실제 startup 실패 및 E287 기각을 다시 통과로 바꾸지 않는다.

원래 affine 문제의 dual cone, sparse A/b, 원래 activation/명령을 검산했다. 기존 weighted-slack 상한 −.005164623588984897에 보수적 allowance2.139700585478089e−6을 더해도 음수다. 반면 envelope를 버린 더 넓은 box에 투영한 dual 상한은 **+250.4981729658599**로 불충분하며, 이 값으로 실행 가능성을 주장하지 않는다. 낙관적 raw RMS 하한 .005329916310487876 / 보수적 하한 .005329916286475793도 RMS 배제 근거가 아니다. Box 배제 판정은 위6개 고정 행만으로 성립한다.

Root는 production compute를 호출하지 않고 별도 scalar `math.fsum`/구간·support 공식으로 모든 raw 배열과 provenance, dual 항, native/internal 판정을 검산했다. 구간 최대 차이1.7763568394002505e−15, dual coefficient 차이6.938893903907228e−18이며 source/input/output pin, 완결 seal, 원래 비용 및 종료 PID를 확인했다. E150/SSP-u0, 다른 source, 상한 U보다 큰 activation, 일반 비선형 동역학, 생물학적 인과, 자연 보행에는 이 배제를 옮기지 않는다.

구현은 실제 gpt-6-sol/xhigh, FIRST는 root, 부분 raw 보존과 최종화 시간을 분리하는 첫 수정은 같은 Sol, SECOND는 root가 수행했다. Root 잔여 소스 수정0. Sol과 root 각각43개 생성검사/Ruff가 통과했다. Sol 생성 subprocess 합3.8167363초(최초 Ruff5건 실패 포함), 구조 읽기 등 지원3.2277102초, root SECOND 감독1.5416776000056416초다. Sol 초기 lint CPU/RSS는 미계측이며 생성 mock은 실제 분석에 포함하지 않는다.

원본 저장 분석1회/재시도0, supervisor **.6609200999955647초**, tree RSS68120576bytes, runner 자체 peak72663040bytes, 출력478449bytes다. 두 RSS는 서로 다른 샘플러의 관측이며 모두 상한 이내다. Gate/load/compute 각각1회이며 model/neural/body/physics/CUDA/fit/solver/QR0이다. Root 독립 검산 .32677639997564256초는 별도 수동 검산 비용이며 새 원본0이다. 누적 **8416.895937599536 / 8560초**, 잔여143.1040624004636초, 예약0, 실행 중 원본0이다.

E175 개별 MN 이력, E192 singleton 확대, E277 fit/development, E279 native startup, E287 재적합과 E310 resource branch의 기각을 유지한다. E279–E286에서 밝혀진 내부 여유와 실제 startup의 차이도 보존한다. 다음 설계는 이 고정 E277 source의 decoder 재시도에 기대지 않으며, 시작을 통과했던 기준 source의 지속 상태 생성과 근육 협응을 대상으로 정한다. 후보 채택0, F1–F8·자연스러운 생물학적 보행은 미완료다.

근거: [사전 설계](BANC_M3_E311_DECODER_OBSTRUCTION_DESIGN_20260928.md), [root 독립 검산](../verification/banc-natural-gait-m3-20260919/parent-e311-original-raw-audit.json), [소스 검토](../verification/banc-natural-gait-m3-20260919/parent-e311-source-review.json), [원본 비용](../verification/banc-natural-gait-m3-20260919/parent-e311-original-supervisor.json), [원자료](../verification/banc-natural-gait-m3-20260919/e311-decoder-obstruction/result.json).
