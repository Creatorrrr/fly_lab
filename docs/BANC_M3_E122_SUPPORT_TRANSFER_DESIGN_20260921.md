# E122: 기록된 신경 출력과 실제 지지 동작의 전달 진단

2026-09-21 KST · 부모 설계 · Sol/high 구현·첫 수정 · 부모 리뷰·잔여 수정

## 질문과 현재 근거

E121은 기존20개 목표 포트에서 운동신경으로 향하는872개 연결의64배 증폭만 제거한 실제 전체 BANC/CUDA10초 개입이다. 원래99 초기 명령과 저장 미래를 유지하면서 전진10.516212mm/net-path.584337로 F3를 통과했다. 여섯 신경 리듬은21–22주기/.419–.428초로 존재하지만 실제 최장 전진bout은.3초/적격 완전주기0이고 F4/F5는 실패다. parent-final-review SHA1119a26a4c32d65dd11980e174538c773cea145a9a9dbbf058b6a08bf5121dff와 실패 기준을 유지한다.

다음 구현 선택을 위해 요청한 동작이 연속 전진에 맞지 않는 부분과, 실제 신체가 요청을 추적·지지 접촉으로 전달하지 못한 부분을 구분한다. 저장된 E85 외부 교사, E113, E121 세 기록은 모두2001개0–10초/5ms 신체 sample과100000개100us native 접촉을 갖는다. 신체 hash는 모두1b2a89084311ebd2ccbccbe2c57f2ec29fa937e115bb6cd4c7f6cd16ba7c2e6f이며 t0의qpos/qvel/42관절각이 exact임을 부모가 구조 점검했다. 교사의 성공은 BANC 성공이 아니다.

## 입력·불변 조건

`e122-frozen-design.json`에 실제 읽는 세 trace/native/result, E113/E121 calibration·MN/근육 기록·anatomy/start manifest, E121의 독립 최종검토·원본manifest, 원래99 기준, 비교기·adapter·기존 body/trace source SHA를 동결한다. E113/E121 원본 hash 목록도 검사하고 original start는 StateStore manifest/array SHA를 검증하여 읽는다. decoder 재표현의 신호는 각 자신의 실제 rate와 실제 calibration만 사용한다. E110/E119/E120 입력을 E121의 새 입력으로 간주하지 않는다.

새 도구 `tools/analyze_c_banc_m3_support_transfer.py`와 `tests/test_c_banc_support_transfer.py` 두 파일만 Sol이 구현한다. 분석 전용으로 GraphStore/신경망/Body 생성, CUDA 계산, 물리 전진, solver/fit/LP, 다른 initial state/pose, F6, 제품/UI/서버 변경은 금지한다. 정적 좌표 변환과 decoder 재표현은 아래 명시한 횟수만 수행한다. 최종 CPG/phase/trajectory/root 금지와 원래 F1–F8은 변경하지 않는다.

## 정해진 계산

### 1. 기록 검증과 정확한 decoder 재표현

세 trace의time_s=arange(2001)*.005, control_tick=0..2000, 요청/측정42관절, 실제기하학 좌표·basis, native time(1..100000)*.0001(절대오차2e-8이하), 최초 물리상태/공통body hash를 확인한다. 원래 모양/축이 다르거나 비유한 필수 값이면 raw와 오류를 보존하고 불완전으로 종료한다. E85 native 파일에 slip_valid가 없다는 스키마 차이는 명시하고, E85에는 저장 slip의 isfinite를 가용성으로 도출한다. E113/E121의 제공 slip_valid는 자체값과 일치해야 한다. 하중>=.02BW인 slip은 반드시 유한해야 하며 무하중NaN을0으로 치환해 평균에 넣지 않는다.

E113과 E121 각각 정확히 한 번, 기록2000×446 신호에서 자기391MN ID 순서로rate를 추출한다. 관측/metadata/calibration/anatomy의391motor/96muscle/membership과 각근육columns를 교차 확인한다. pool은 비음수이고 각행합1, 각MN이정확히한근육에기여하며membership과일치해야한다. 시작rate와96activation은원래저장0이다. 순서는LEGS=(lf,lm,lh,rf,rm,rh), 각7관절이며muscle의leg·name·columns출처를보존한다.

고정 dt=.005, alpha=-expm1(-dt/.01)에서 각391MN의 선형 filtered-rate를 f[t]=f[t-1]+alpha*(rate[t]-f[t-1]), f[0]=0으로 한 번 계산한다. 실제MN발화를변경하거나신경모델을다시적분하지않는다. pool@f로96activation, rest+.02*moment@activation으로42요청, .2*tanh(activation[ltm]/10)으로6tendon을재표현한다. 실제저장activation/요청/tendon과 비교하여 최대절대오차1e-10 이내여야한다. 추가로 저장activation→요청의순수행렬대수도 확인한다. raw를 먼저 저장한 뒤 판단한다.

재표현은 총2회/각2000sample/총4000 decoder step이며 이를 실제neural/physical초로표기하지않는다. 부모의독립NumPy/SciPy 검산2회는별도비용/횟수로기록한다. E85는MN신호가없는외부교사이므로decoder를재표현하지않는다.

### 2. 요청·추적·지지·전진 비교

각기록을원래.5<=t<10(1900개control상태)창에서비교하고초기99는별도요약한다. 몸통위치미분을 실제전방basis에투영하고Gaussian sigma20ms를적용하는 기존정의를그대로사용한다. 분석층의분류는 speed>.5, 0<=speed<=.5, speed<0 세그룹이다. 이는기술적구분이며>.5단일sample을F4적격완전주기로부르지않는다. 기존analyze/load_model 결과와원래F판정은그대로보존한다.

42개관절의raw요청-측정오차RMS/최대값/95%값과각7관절leg별요약, 요청및측정의동적RMS(자기창평균제거), 그비율을계산한다. 분모0/비유한값은 unavailable/null로남긴다. raw요청의실제제한범위를이기록만으로추정하지않으며, 요청추적차에actuator/joint limit 영향이포함될수있다. 이것만으로서보의결함이나인과지배를판정하지않는다.

실제기록의각다리tarsus5원점을몸통basis로변환해전후위치/높이와전후속도(중앙미분+sigma20msGaussian)를계산한다. 원래load_bearing/.02BW boundary와연결하여양수전후속도(몸통기준앞으로가는발)의지지비율, 하중가중평균속도, 실제전진그룹별지지/미끄럼/추적차를비교한다. 발원점대용의운동은실제접촉점의마찰힘이나추력의부호가아니다.

native는정확히50step씩control구간(t-1,t]로묶어하중지지fraction과하중가중slip을계산하고,해당구간끝의몸통속도분류와연결한다. control-boundary 접촉과native구간평균은별도필드로유지한다. 임의보간·시계이동·누락구간연결은하지않는다. 각velocity group의native비용이원래1900×50=.5초포함창과일치하는지명시한다: 상태창t=.5..9.995의구간은native(.495,9.995]다. 원래F7 slip창native[.5,10)은별도로보고하며서로같은창인것처럼표기하지않는다.

### 3. 근육·MN 기여와 그림

C=.02*moment@pool(42×391)과각MNfiltered-rate(2000×391)를factorized raw로보존한다. 불필요한2000×42×391tensor는만들지않는다. 각창/전진그룹의선형요청기여의평균과자기평균을뺀RMS를42×391및42×96으로요약하고,원래anatomy이름으로큰항들을표시한다. 선형기여의크기를실패책임·인과기여율로바꾸지않으며rest항과부호/상쇄를함께보고한다.

정적PNG는원래.5–10초전체를보여주고세기록의전진속도/관절추적오차/하중지지중전방발운동을같은축으로비교한다. 저장record의그림이며실제body영상이나F8완료로세지않는다. 데이터와summary가그림보다먼저저장되고그림오류도오류로남는다. UI나브라우저는건드리지않는다.

## 판단과 자원

decoder정확성/입력/시계/원래F평가기보존이검증돼야유효한전달진단이다. 주요요청파형차·상보모집과추적·접촉차를기술하고,부모가그근거와기존실패에따라다음한정개입을설계한다. 임의성공임계값으로생리변경을자동선택하지않는다. 이분석의candidate_adopted/F6_exposed/whole_goal_complete는항상false다.

CPU thread1(OMP/MKL/OPENBLAS), 원래분석한프로세스/재실행0, 내부25/외부40/child35초,새dataset/재표현시작은내부마감3초전까지다. 첫dataset raw부터점진저장하고예외·저장실패·cleanup에서주예외와secondary notes를보존한다. 최종저장/plot종료를자원확인에포함한다. E121 누적917.794741초를918로올려기존shared988초안에서실행한다(새외부40초를모두써도958초). 과거예산/실패는바꾸지않고부모검산비용은별도기록한다.

출력e122-support-transfer가있으면거부한다. 최종부모source리뷰후에만원래2회decoder/3기록분석을한번실행한다. Sol/high 구현→부모첫리뷰→같은Sol첫수정→부모재리뷰·잔여직접수정순서다. 분석성공은자연보행성공이아니며,전체목표는실제최종M3의F1–F8검증까지active다.
