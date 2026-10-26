# 실제 명령과 관절 추종을 구분하는 최소 계측 설계

기존 유한 목표 F1–F8은 ACTIVE/INCOMPLETE다. 사용자 최신 지시에 따라 범위 안에서 승인 요청 없이 진행한다. 해부학적 노드·연결 provenance, 신경 단독 구동, 고정 startup/short gate, M2/UI, 실패 후보와 누적 비용은 그대로 유지한다. 이 단계는 기각 후보의 원인을 식별하는 진단이며 보행 채택 시험이 아니다.

## 근거와 질문

실제 startup-ray J의 중립과 alpha .0001 후보는 각각 한 인스턴스로 control0→99→300을 실행했다. 후보의 startup/health는 통과했으나 forward1.6058769432mm/slip1.9605032380mm/s/bout.21s/qualifiedcycles0으로 기각됐다. 중립도 slip/bout/cycles 기준을 통과하지 못했다. 저장 CPU 진단 `parent-startup-ray-20261007-j-saved-transfer-03/result/report.json`은 성공했으며 실제 명령 R 합계는 후보에서 .041373049 감소했지만 전진은 .0799203791mm 감소했다. R만 줄이는 방향은 보행 개선을 보장하지 않는다.

저장 native/trace에서 RH의 지지 중 body-frame AP 이동 합계가 다른 다섯 다리와 반대였고 하중 가중 native slip도 가장 높았다. 이는 시간행의 연관이며 원인이나 추진력을 뜻하지 않는다. 실제 42관절 각도와 live drive/state가 없어서 명령 방향·추종 오차·폐루프 입력 변화 중 어느 경로를 수정해야 하는지 아직 구분할 수 없다.

## 사전 가설과 결정 규칙

H1: RH의 적용 명령 자체가 지지 중 발의 전진 방향을 만든다. H2: 명령은 후방 발 이동을 요구하지만 실제 관절 추종과 접촉 제약이 반대 방향 이동을 만든다. H3: 둘이 섞이거나 새 계측이 기존 동작을 충분히 재현하지 않아 식별되지 않는다. H1/H2는 운동학적 설명 후보이며, 하중·힘·작용점 원인이 계측되지 않으면 인과 결론을 내리지 않는다.

실제 command/actual의 같은 MuJoCo 관절 순서·rad·시간 경계를 먼저 확인한다. control t의 적용 command는 interval(t−1,t]에 속한다. q_pre=actual[t−1], q_post=actual[t], e_post=command[t]−q_post. control0은 초기 상태이며 적용 interval 비교에서 제외한다. official 창은 control100..299의 후경계200개이고 두 끝 모두 창 안인199쌍과 원래 temporal200쌍은 따로 표기한다. 시간행은 독립 반복 표본이 아니다.

낮은 추종 오차와 동일 부호의 명령/실측 관절·발 이동이 관측되면 다음 설계는 신경 출력의 하중·추진 타이밍을 우선 다룬다. 큰 추종 오차 또는 반대 부호·lag가 관측되면 다음 설계는 단위·순서·decoder/body 전달 경로를 우선 다룬다. 이 기준은 상대 비교와 원래 관절 운동 범위로 기술하고 임의 cutoff로 가설 PASS를 만들지 않는다. 양쪽 설명이 남으면 mixed/unidentified로 기록하고 식별 가능한 작은 다음 개입을 설계한다.

## 고정한 실제 작업

두 arm을 순차 실행한다. 각각 최대 한 번, 현재 검증된 neutral initializer와 이미 기각된 alpha .0001 실제 float32 selected weights/θ를 사용한다. 같은 실제 session/body/network/adapter 인스턴스로 control0→99→300, 최대1.5초만 실행한다. startup99 실패나 health 실패에서 멈추며, 실패 후보를10초/F6/F7로 승격하지 않는다. 새 alpha·학습·CPG·teacher·위상·궤적·root 보정은 없다.

기존 `continuous_arm`/OriginalFactory/LiveObserver/native arrays/evaluator를 재사용한다. observer는 기존 실제 `advance_readout` 호출을 바깥에서 감싸며 호출을 정확히 한 번 전달한다. adapter.encode를 다시 부르지 않는다. physics advance를 추가하지 않는다. `b.capture()`의 기존 full row에는 joint_angles_rad와 joint_targets_rad가 있으므로 이 값만 추가 저장한다. applied command42, actual42, q_pre/post 경계, control/network/physics clocks, original native/lean trace를 보존한다. production session/decoder/body는 변경하지 않는다.

앞으로의 조건부 신경 방향 비교에 필요한 최소 자료도 네 R창에서만 저장한다. 창 시작100/164/228/292의 pre-advance rate/adaptation, decoder bank/original_activation, SSP resource와 source IDs/zero-utilization invariants를 읽고, 실제 control101..104/165..168/229..232/293..296에 전달된 drive16×158706 float32를 저장한다. 필요 field는 실제 deployed class의 instance attributes에서 검증하며 whole snapshot/11.6M weights를 계측 목적으로 복사하지 않는다. resource·decoder state가 필요 없다는 근거가 있으면 그 field를 생략한 조건부 범위를 명시한다. drive를 저장해도 다른 sensory history를 동일 개입으로 해석하지 않는다. 이 단계에서 neural replay/forward/backward는 없다.

목표 관절과 실제 관절의 RMS/max/95–5span, loaded/unloaded/transition 구분, lag0..4controls(최대20ms)의 설명적 오차를 같은 denominator로 보고한다. lag 선택은 원인/성공 판정이 아니다. 발 AP는 body basis를 이용하며 command→foot 운동학은 기존 geometry helper를 동일 관절 정의로 검증한 경우에만 비교한다. geometric knee degrees와 command radians를 직접 차감하지 않는다. 힘·토크·slip·body-frame AP는 서로 다른 지표다.

## 실행 연결과 자원

현재187 debit, Original global11030.977960899505/16200초, stage1137.223/6300초, science-local1010.862/6000초, generated126.22/300초이다. 추가 grant0이며 기존 비용·실패·J grant offset165를 보존한다. owner worker1/automatic restart0, 기존 Windows Job/PID+creationticks/lock/EOF/once-writer를 재사용한다. 새 계측 역할은 기존 J8 역할을 변형하지 않는 별도 finite request/role이다. fresh current before-ledger와 J 입력·후보 identity를 묶고, 역사적 J source freeze를 현재 소스로 덮어쓰지 않는다. 기존 saved CPU 역할의 실패 한도나 roster도 바꾸지 않는다.

arm별 outer300/child280/internal250/cleanup20초, 둘 합계 outer600초; saved 분석 outer120/child100/internal80/cleanup20초. host24GiB, GPU 기존 monitor 한도와 sample cap을 유지하고 output arm별96MiB/diagnosis16MiB, 전체 새 durable output208MiB. drive10,157,184B와 four boundary state 약5MiB에 기존 native/control/trace를 더해 cap 내에 둔다. 실제 사용은 genuine full return으로 모두 Original에 한 번 청구한다. 생성 검증은 기존 잔여173.78초에서 lint/필요한 numerical observer fixtures와 기존 실제 Job tests만 수행한다. 완료·실패·timeout 모두 정리와 비용 기록을 남긴다. 재실행은 동일 실패 반복이 아니라 새 원인과 수정이 있을 때만 별도 prospective request로 판단한다.

## 구현과 리뷰

ROOT 설계 → actual GPT-6 Astra/max 검토 → ROOT 반영 → actual GPT-6.1 Sol/xhigh 구현 → ROOT FIRST → 필요하면 같은 Sol sole R1 → ROOT SECOND → ROOT 잔여 수정 순서다. 구현은 새 계측 도구·필요한 작은 admission/owner 역할 연결·meaningful 테스트에 한정한다. 기존 8 과학 역할, saved 진단, F1–F8 evaluator/threshold, production/UI를 변경하지 않는다. 출판·리뷰·fixture PASS는 실제 보행이나 biological mechanism 증거가 아니다.

계측 완료 판정은 실제 work/counters/known close, clocks·단위·관절 순서·J weights provenance, finite 자료, pre/post hash와 기존 baseline의 비교 가능한 재현에 따른다. 재현 여부는 raw command/native/trace 비교의 max/RMS와 실제 native drift를 보고하며 임의 epsilon을 사후 선택하지 않는다. exact replay가 아니라면 새로운 관측 trajectory임을 유지한다. 최종 진단은 run validity/hypothesis verdict/adoption/전체 목표 상태를 구분하고, 다음 개발할 경로 하나를 근거와 한계로 선택한다.
