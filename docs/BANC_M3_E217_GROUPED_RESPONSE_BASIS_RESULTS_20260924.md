# E217 결과:64상태·근육별 공유 반응 기저 검산 통과

391MN의 자기 신경입력과 두 인과기억을 사용하는64기저와,90자유근육의 공유계수5760개를 위한 조건부 선형 전달을 구현했다. 원본 기록 전체에서 CUDA/독립 계산이 일치했다. 계수 적합·전체 BANC 피드백·물리보행은 아직 수행하지 않았다. E150이 최선 원본 보행이며 F1–F8은 active·미완료다.

부모설계→실제gpt-6-sol/high→부모FIRST에서 중복센터 sigma 규칙1결함→같은Sol 첫수정→부모SECOND 및 순수검사6/Ruff→생성CUDA1→원본CUDA1→부모독립검산 순서를 지켰다. 중복센터 검사는 nearest-other 거리0을 먼저 얻은 뒤 양수값만 median에 넣는 사전규칙을 확인한다. 모든96근육에서64개 고유 anchor/own center/sigma/weight가 부모의 다른 계산식과 일치했다.

기존 h_fast/h_slow/descriptor/H/H_factor는 byte exact다. B 합의 이전 기저와 최대차1.22125e-15, pooled group envelope차1.70530e-13, 직접/affine 관절명령차3.55271e-15였다. 부모는7개 MN(비표준 tau의92104 포함)에 대해160000 stage 전체를 닫힌 RK 다항식과 lfilter로 독립 재계산했고 허용오차 안에서 일치했다. 고정48MN과6LTM activation/tendon도 exact다. gate가1이 아니므로 B합을1-H_factor와 같다고 가정하지 않았다.

원본은3입력 upload 각각1회·새 history0·basis CUDA1·optimizer0·전체신경0·physics0·retry0였다. RTX4080 실행 중 GPU사용률100%를 한 번 관찰했다. kernel27.6586276초/readback포함27.7289708초, 감독35.8939399초다. CuPy소유 peak1,265,798,656bytes, 단일프로세스 peak3,485,966,336bytes, 출력631,650,046bytes로 사전한도 안이다. 종료후 소유프로세스가 남지 않았다.

원본누적5732.071894399702/6004초, 잔여271.9281056002983초로 배정증가0이다. 별도 측정:geometry준비1.2438958초/SECOND geometry1.5564735초/부모순수검사1.4135822초/생성CUDAwrapper2.2282546초/생성독립검산.7036701초/원본독립검산6.5996259초. 최초준비 및 FIRST 비용은 해당JSON에 보존했고 agent설계·구현비용 미측정을0으로 치환하지 않았다.

자료는 `verification/banc-natural-gait-m3-20260919/e217-grouped-response-basis/`의 raw/ledger/manifest와 `parent-e217-{first-review,second-review,source-review,generated-independent,output-review,supervisor,final-review}.json`이다. E214/E215/E216 실패·진단/가정은 보존했다. F6 미노출, 후보채택0, M2와UI 변경0이다. 다음 단계는 이 검산된 기저의 실제 공유계수 적합을 별도로 사전설계한다.
