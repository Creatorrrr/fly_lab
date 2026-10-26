# E337 현재 파형 교체 대조: technical invalid

원본 비교는 신체 생성 전에 자료 스키마 검증에서 중단됐다. H337의 supported/not-supported 판정은 없다. BANC 자율 보행·생물학적 보행·F1–F8은 미완료이고 후보 채택·UI 변경·F6 노출은0이다. E335 기각과 E336의 유효한 저장 진단은 그대로 유지한다.

ROOT 설계 → actual GPT6 Astra/max 리뷰 → 반영 → actual GPT6.1 Sol/high 구현 → ROOT FIRST의3 blocker → 동일 Sol ONE R1 → ROOT SECOND → ROOT 잔여 alias/path 핀 수정 순서를 실행했다. 현재 생성 검증54 PASS와 passive 검증 PASS는 원본 loader의 전체 자료 경로를 검증한 성공이 아니었다. 생성 fixture가 실제 frozen archive descriptor의 중첩 포맷을 포함하지 않아 다음 오류를 놓쳤다.

원인: `Dependencies.selected()`가 `freeze["input_members"][name]` 전체를 NPY member map으로 취급했다. 실제 동결 descriptor는 `{"sha256", "bytes", "members"}`이고, 그 안의 `members`가 배열 목록이다. 따라서 실제 command archive의 배열 키와 컨테이너 키를 비교해 `full frozen archive key roster differs`를 발생시켰다. E336의 올바른 선택 reader는 이미 `descriptor["members"]`를 사용하고 있었다. 오류를 회피하기 위해 비교를 제거하거나 원본 descriptor를 평탄화하지 않는다.

원본 job1/retry0. Actual supervisor 51.293273900/80초, tree RSS53,334,016B, exit1/kill0. 비용은 종료 후 한 번만 합산하여 9565.21586319952/9660초이며 잔여 94.78413680047925초다. 두 owned PID가 종료된 뒤 결과를 읽었고 예약은0으로 해제했다. Result와 advance ledger는 오류/invalid로 일치하며 publication 자체는 완료됐다. 이 publication 성공은 과학적 성공이 아니다.

Selected numerical reads0, constructor requests0, arms empty, neural/CUDA0. 모델 전진0은 생성 요청 자체가 없었음에서 확인하며 unknown 전진을0으로 채우지 않았다. 원본 자료의 metadata/trace와 ZIP headers 검사는 수행됐다. 시간51.293초의 내부 단계별 분해는 기록되지 않아 해시/trace/header별 정확한 병목은 미확인이다.

독립 saved audit 3.379032500/30초는 현재207 source/2951 input 핀, 저장 manifest와 invalid 보존 등8검사를 통과했다. 유효성 실패 분기의 감사로, 파형·물리·보행 metric을 수치 재계산했다는 주장은 하지 않는다. 감사 numeric reads0, 신체/신경/CUDA0이며 두 audit owned PID도 종료됐다.

다음 단계는 원본 보행 비교의 자동 재시도가 아니라 **전체 archive descriptor와 selected payload의 계약을 구현·검증하는 자료 사전 검사**다. ROOT 새 설계 → actual Astra6/max 리뷰 → GPT6.1 Sol 구현/필요시 동일 Sol ONE R1 → ROOT 잔여 수정 순서를 유지한다. E337 frozen source와 실패 산출물은 보존하고, 원본 비교 가설은 미판정으로 둔다. 생성 회귀에는 실제 중첩 descriptor/선택하지 않는 neuron 배열/추가·누락 키/dtype/endian/order/위험 NPY 검증을 포함하여 같은 입력 스키마 오류가 물리 실행 단계까지 도달하지 않게 한다.
