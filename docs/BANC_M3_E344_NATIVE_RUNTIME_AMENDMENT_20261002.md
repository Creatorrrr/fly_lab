# E344 실제 runtime 및 lossless trace 보정 — 원본 실행 전

동일 E343 `theta_minus`, 초기 전체 상태, live sensory→BANC→decoder→native physics, startup99 및 모든 원래 보행 기준을 유지한다. ROOT SECOND 뒤의 잔여 수정이며 원본 예약·실행은 0이다. 같은 Sol ONE R1은 이미 종료했으므로 잔여 수정은 ROOT가 수행한다.

## 직접 관측한 차이

- `parent-e344-native-01-report.json`: 실제 packed SSP 신경20회와 MuJoCo physics50회가 requested/host-returned/device-confirmed로 완료했다. 2000회 settling과 neural setup40/capture20도 구분해 기록했다. 프로세스 및 sampler 종료를 확인했다.
- 같은 실행은 `measured landmarks dtype/shape`로 게시에 실패했다. 현재 `walking_trace.capture_control`은 `flylab.body.LINKS` 순서인 coxa, trochanterfemur, tibia, tarsus1, tarsus2, tarsus3, tarsus4, tarsus5의 **다리당8개** 실제 body origin을 기록한다. 구현의5개 가정과 최초 설계의5개 표현이 틀렸다.
- 실제 FlyGym 정의는 `.venv/Lib/site-packages/flygym`과 `flygym_demo`에서 로드되었다. 최초 descriptor는 인접 `flygym/src`만 등록하여 실제 로드된41개 정의를 선언하지 못했다. 이것은 신체/보행 자격이 아니며 당시 실행과 실패·비용을 보존한다.

## 최소 수정

1. Native trace에서8개 좌표를 **삭제·축약·추가·재배열 없이** 저장한다. 기존 gait evaluator의 다리 길이는8개 점의 segment 길이 합으로 계산하므로5개로 줄이면 평가값이 변한다. 기존 evaluator와 neutral 비교를 그대로 유지한다.
2. Synthetic fake fixture의5개 점은 명시된 synthetic scope로만 허용한다. Actual native smoke와 original은8개 점을 확인한다. Trace top metadata에 실제 점 개수와 순서를 기록하고 ROOT audit에서도 native/fake 범위를 구분한다. 바이트 상한2MiB는 유지하며 넘으면 실패시킨다.
3. `parent-e344-native-dependency-descriptor-v2.json`에 실제 설치본 PY47개와 native constructor asset14개를 추가한다.14개 asset의 바이트는 최초 local descriptor에 포함된 대응 asset과 모두 같았다. 최초 descriptor 및 registration은 보존하고 v2 registration에서 새 descriptor를 직접 고정한다. 기존 cached meshes85개와 네트워크 다운로드 금지, 전체 source/config/XML/mesh/asset byte pin을 유지한다.
4. Workspace 내부의 FlyGym 정의 경로는 canonical project-relative key로 기록하고, 외부의 등록된 정의는 absolute key로 기록한다. Source alias/collision을 거부하고 현재 실제 실행에서 관측한 source closure를 다시 선언한다. 패키지 설치·업데이트·모델 변경은 없다.
5. ROOT의 현재 fake301을 다시 게시한다. 같은 audit CLI의 정상/6개 coherent fault에는 공유하는 immutable 정상 파일과 fault마다 바뀐 파일만 사용해 이전 실패 artifact와32MiB 합집합을 보존한다. 과거 source SHA를 현재 실행으로 세탁하지 않는다.

## 고정 자원 및 판정

ROOT 실패 포함 generated 누적19.928994600020815/60초, Sol FIRST+ONE R1 누적59.13786539998546/60초이며 재설정하지 않는다. ROOT 잔여40.071005399979185초 안에서 현재 fake/actual native/same CLI 자격을 닫아야 한다. 실제 standalone native1은 전체 BANC constructor·sensory의 motor causality·보행·F1–F8 증거가 아니다.

원본은 자격 완료 뒤에만 동일한 body80초(science60/internal72/final8)와 audit12초(internal10)를 예약한다. 총 사용자 예산은 지정되지 않았고9660은 ROOT가 이전 단계에 배정한 값이다. 사전 자격 뒤 prospective9780으로 조정하는 기존 권한 해석을 유지한다. 개별 원래 cap31.5model초/300wall초, GPU1, 자동 retry0, F6/UI/M2 보존을 바꾸지 않는다.

실험이 실행되면 원래1.5초 고정 startup4/smooth/legacy shape/forward/slip/bout/all-six cycles/health를 같은 기준으로 판정한다. 실행 유효성, 가설, 후보 채택, 자연스러운 보행 및 생물학적 기전을 각각 구분한다. 일부 physics 반환, generated 검사, geometry 보정 또는 작은 native 연결은 목표 완료가 아니다.
