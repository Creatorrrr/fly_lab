# E188 재개: 종료된 외부 서비스의 준비 검사

2026-09-24, 원본E188=0 상태의 실행 전 보완이다. 사용자의 재개 요청 후 blocked 감사는 새로 시작한다. 현재8188연결은 connection refused, TCP목록에 해당 listener없음, 관측 프로세스 이외의Python없음, RTX4080은GPU0%/842MiB다. 이는 외부 작업이 성공 완료했다는 증거는 아니며, 현재 서버와 작업 프로세스가 실행 중이지 않다는 증거다.

기존 E188 resource readiness는 HTTP queue가200을 반환해야만 진행하므로 종료된 서버도 오류로 막는다. 서비스를 임의로 재시작하거나 허위 빈 queue 응답을 만들지 않고, 아래 별도 `service_stopped` 상태를 관측한다. 신경망·몸체·입력·195 incoming·601 endpoints·E185 exact 비교·F1–F8·원본1회/재시도0·CPU1/BLAS1/GPU1·E181100초/E188140초/부모150초·누적4994초는 같다.

## 준비 및 모니터 조건

새 순수 자원 helper는 기존 E182 `_sample_resources()` 및 `_client_status()`를 호출한다. E182 파일은 변경하지 않는다. E188의 두 번 준비 검사(2초 간격)와 실행 중 모니터만 새 helper를 사용한다. 모든 raw queue/GPU/client 자료를 남기고, queue error를 지우지 않는다.

- `queue_empty`: 실제 HTTP200 결과가 running0/pending0이며 queue error가 없고, 기존 clients가 모두absent/reused이고, GPU telemetry가 유효한 경우다. 실행 중에도 동일하다.
- `service_stopped`: queue 요청이 실패한 경우에만 별도로 재확인한다. 127.0.0.1:8188 TCP 연결이 timeout이나일반오류가 아닌 명시적 connection refused여야 하고, TCP listener목록에도8188이없어야 한다. 현재 관측 process와 그 조상/자손을 제외한 Python/PyPy process가 없어야 하고, 모든 관련 process조회·port조회가 성공해야 한다. 기존 clients도absent/reused여야 한다. 준비 단계에서는 실제 GPU telemetry의모든GPU가사용률≤5%,used memory≤2048MiB를추가요구한다. 실행 중에는 자기BANC의GPU사용을포함하므로 이두 idle조건은적용하지않고 service/process/client조건과유효telemetry를매번확인한다.
- queue사용중,부분조회,timeout,접근오류,invalid telemetry,unknown client,listener 존재,외부Python존재는 준비실패/monitor실패다. 포트가다시열렸거나서비스가돌아오면실제queue를검사한다. 실행중실제외부작업겹침은실패로남긴다.

service_stopped는 이전 실패를 성공으로 재분류하는 것이 아니다. 아직 원본E188이없으며, 이번 새준비관측에만 적용한다. 모니터의 raw error, 별도검증결과,구분된mode와resource_clear를보존한다. fail-closed 오류처리 및원래모델원본실패보존은유지한다.

## 구현·리뷰·동결

실제 gpt-6-sol/medium이 `tools/banc_m3_resource_readiness.py`, `tests/test_c_banc_resource_readiness.py`를 구현하고 E188 tool/test의자원검사호출만연결한다. 난이도는새모델연산이없는포트·프로세스·예외상태분류이므로medium이다. 부모설계→Sol구현→부모FIRST→같은Sol첫수정→부모SECOND→부모잔여수정을지킨다. 부모는이보완문서,현재동결의보존,새source gate/감독연결/독립검산기변경을담당한다.

합성검증은정상empty/busy queue,명시적refused+listener없음+외부process없음,timeout,접근거부,listener충돌,외부Python,unknown client,invalid GPU자료,준비idle검사와실행중자기GPU사용구분,monitor raw error보존을포함한다. 모델생성없는실제현재자원검사2회를검토한후에만원본을실행한다. 이전부모검증전달동결과승인기록은별도보존하고새helper소스및이문서를동결에포함한다.

원본누적4351.0849566997495/4994초·잔여642.9150433002505초를승계한다. 자연보행전체미완료/F6노출0/후보미채택이며기존M2와사용자UI를변경하지않는다.


### Windows 조상 프로세스 경계 명확화 (FIRST 리뷰 중)

현재 실제 chain은 Python/venv→PowerShell→Codex→ChatGPT→sihost→svchost→services→wininit(pid1676)이며 wininit의기록된부모1248은현재PID목록에없다. 종료된Windows초기화부모가계속살아있어야한다는조건을추가하지않는다. 완전한PID목록에없는부모는 별도 `psutil.pid_exists(parent)==False`로부재를확인하면 live ancestor chain의끝으로기록한다. 그missing PID는허용목록에넣지않으며 `ancestor_terminal`에missing PID/마지막자식/verified_absent를남긴다. 조회오류나목록밖인데live인경우는fail-closed다. pid0root도별도로기록한다. 현재자기자손과개별확인된살아있는조상만제외하며, 형제·사촌·조상의다른자손은외부Python으로본다. PID부모생성시간역전이나cycle은거부한다.


### 부모 SECOND 이후 실제 Windows 비용 보완

합성41tests 후 실제두관측은TCP1초timeout으로차단됐다(원본0). 별도연결관측에서3/5초제한은각각2.0599/2.0408초에명시적WinError10061을반환했다. 기존실패는보존하고probe timeout만3초로두며 timeout자체는계속거부한다. 모든process이름을조회하되ppid/birth는Python과현재실행family에대해서만조회하여관련없는시스템process의반복부모목록조회비용을줄인다. 조회오류·PID변경·형제/사촌제외금지는같다.

두번준비관측(간격2초)은동일150초부모감독안에서전체input검증이끝난뒤실행한다. 부모검증영수증에원본관측2개와완료monotonic시각을함께넣고그SHA를자식에전달한다. 자식은기존live parent/정확한launcher/원본파일stat·receiptSHA검증후 모델import전에관측완료후10초이내인지확인하고두관측을자기readiness.json에그대로보존한다. 런타임monitor는자식에서계속실제자원을조회한다. 자식100초/내부140초/부모150초·원본1회/재시도0는유지하며준비비용도부모150초에포함한다. 부모SECOND이후잔여수정이므로현재세션이수정한다. 이는기존모델검증gate나관측성공판정의우회가아니다.
