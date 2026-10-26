# E268 native 자료 수집 사전 계약

2026-09-27. E268 관측기 FIRST에서 생성 CUDA 3개 테스트와 부모 독립 8검사가 통과했다. 전체 신경 상태와 단계별 전류/이력의 독립 비교 오차는 0이다. 수정이 필요하지 않아 첫 수정·SECOND·부모 제품 수정은 없다. CPU 15개 통과는 Sol의 생성 검사이며 원본 학습이나 보행 증거가 아니다.

원본은 한 번, 두 arm을 순서대로 실행한다. E267 중립1660×3계수, 원래158706노드/11584852간선,391MN/51source다. startup100controls/2000ticks/.5초와 teacher2000controls/40000ticks/10초, 총42000과학ticks/168000stage를 기록한다. 두 모델 생성의 warmup 예상80ticks와 capture기록40ticks는 별도로 실측한다. 신체/physics/fit/solver/자동retry는0이다.

가설은 관측기가 전체 재귀 계산을 바꾸지 않고 필요한 네 RK4 단계 입력을 정확히 수집한다는 것이다. 반대 설명은 store 삽입·이력 단계·drive 재구성 오류다. 판정은 시작/끝 전체 원래 신경 상태 exact, startup101endpoint의 기존rate/adaptation/drive와 E267이력 exact, teacher2001endpoint4046rate exact, 모든2100 full-drive SHA exact, motor/source counts가 각각8000/160000stage이고 overflow0, 모든수치 finite다. 부모는 원자료부터 단계 이력·카운터·상태·입력 SHA를 독립 검토한다. 구현 오류/원본 오류는 보행 가설의 반증으로 세지 않는다.

준비에서 기존교사 control1..2000의 old_filtered를 실제 scatter 순서로 처리하고,20 navigation의−5000 및 두 descending의+5000을 더한 전체drive를 복원했다. 저장sensory/effective_drive와 모두exact,3626개 지원집합 밖은0이다. endpoint를 보간하지 않는다. 두arm의초기전체신경state도exact임을 확인했다.

원본 감독90초, 내부80초 이후 추가advance금지, 정리10초, 프로세스RSS6GiB, GPU관측512MiB/model, 전체출력768MiB, 사용가능디스크2GiB다. 한GPU작업만 실행하고실패원자료를보존한다. E267의.5초advance+동기화1.9388초에 따르면10.5초는약40.7초이며,두생성/관측/readback/검증여유를포함해90초로사전예약한다. 속도개선주장은하지않는다.

원본누적7134.341214599615초를보존하고배정7182→7225초(+43)로조정한다. 사전잔여90.658785400385초 중90초를예약하며실제감독wall로누적한다. 생성검토6.5170458초,Sol CPU/Ruff두번,입력준비4.8305382초와도구process5.2789535초는별도기록한다. fit와body는이예약에포함하지않으며실행전별도한도를고정한다.

성공하면 이 자료로 기존 E150 monotone 전달을 보존하는4980계수 제한 적합을 설계·구현한다. 학습성공과 실제 신체 성공은 별도다. F1–F8/M2/E150/UI/F6와과거실패·누적을보존한다.
