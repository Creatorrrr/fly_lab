> 로컬 반영: fix_2를 현재 작업 폴더에 적용했습니다. 이 환경에서 새로 실행한 검증 결과는 [FIX2_LOCAL_VALIDATION.md](FIX2_LOCAL_VALIDATION.md)에 별도로 기록합니다. 아래 첨부본의 설명·수치는 당시 기록입니다.

> **fix_2 추가 검증:** 새 회귀 95개 PASS, 첨부 실제 물리 원자료 재분석. 상세 내용과 새/과거 결과의 구분은 [REASSESSMENT_FIX2.md](REASSESSMENT_FIX2.md)에 있습니다. 아래는 기존 이력입니다.

# 현재 로컬 검증

실제 물리·벽 회피·다조건·브라우저 검증은 [LOCAL_VALIDATION.md](LOCAL_VALIDATION.md)를 참고합니다.

아래는 **v0.2.0 제작 당시의 기록**이며 현재 수정본의 실행 결과와 구분합니다.

# FLY LAB B 검증 보고서

## 결론

**통합 코드/관찰 도구는 작성되었지만 실제 FlyGym/MuJoCo 물리 실행은 이 제작 환경에서 검증하지 못했습니다.** B를 검증 완료 디지털 트윈이나 안정 보행 보장판으로 제시하지 않습니다. 생성된 테스트 화면은 모두 `UI TEST FIXTURE`가 붙은 합성 대역 화면입니다.

## 실제 수행한 검사

| 검사 | 결과 | 무엇을 입증하는가 |
|---|---|---|
| Python 순수 계산/대역 엔진 33개 | PASS | 데이터 범위·sparse 정규화·좌표 변환·설정 거부·개입·상태/리플레이·구독/CSV/비교 로직 |
| 실제 Python loopback 서버 8개 | PASS | Host/Origin·토큰·단일 연결·WebSocket 요청/응답·미지원 경로와 스텝 제한. 몸은 명시적 대역 |
| Chromium UI 27개 | PASS | 화면 구성·선택·개입·다운로드·상태 복원·모바일 레이아웃. 명시적 대역 RPC |
| A JavaScript / B Python 수치 대조 | PASS | 같은 입력과 dt의 750스텝. 활성도 최대 절대 차이 9.992007221626409e-16 |
| A 설정 cold-start migration 추가 점검 | PASS | A 형식 설정을 B 보행 설정으로 변환하고 대역 엔진에서 초기화. 물리 상태 이전은 하지 않음 |
| Native physics acceptance gate | **BLOCKED** | FlyGym/MuJoCo 미설치·패키지 취득 차단. checks=[] 이며 PASS가 아님 |
| 실제 브라우저→실제 물리 전체 연결 | **미검증** | 두 경로를 합쳐 실행하지 못함 |
| WebGL 하드웨어 / native offscreen camera | **미검증** | UI는 Canvas 원근 3D 경로로 점검 |

순수 계산/서버 테스트는 총 **41개**입니다. 같은 테스트를 반복 실행한 횟수를 합쳐 늘리지 않았습니다. 별도 Python/JS 수치 대조는 이 41개에도 포함되며 추가 1개로 중복 합산하지 않습니다. fixture 기반 상태 복원의 성공을 MuJoCo 적분 상태 복원 성공으로 바꾸어 해석하지 않습니다.

## UI 검사 방법의 제한

`agent-browser` CLI는 설치되어 있지 않았습니다. Playwright/Chromium을 사용했습니다. 정상 localhost 접속을 시도했지만 `net::ERR_BLOCKED_BY_ADMINISTRATOR`로 차단되었습니다. 따라서 같은 배포 HTML을 `page.set_content`로 넣고 **테스트에 한하여** 전송 계층을 명시적 Python fixture RPC로 대체했습니다.

즉 실제 UI 코드는 실행했지만 production 브라우저 WebSocket 연결/기동 경로를 통과했다고 주장하지 않습니다. 서버 WebSocket의 네트워크 검사는 별도의 aiohttp 클라이언트로 loopback에서 수행했습니다. 다운로드 파일은 확장자뿐 아니라 JSON 파싱, CSV 열 수, PNG 시그니처도 검사했습니다.

`tests/UI_FIXTURE_desktop.png`, `UI_FIXTURE_mobile.png`는 UI 검사용 상태입니다. 실제 FlyGym/MuJoCo로 움직이는 장면의 증거가 아닙니다. production launcher는 이 대역을 제공하지 않고 의존성이 없으면 중단합니다.

## 미검증 사항

- FlyGym 설치 후 실제 모델 compile/초기 정착, 전체 몸 접촉 preset의 안정성.
- 상위 신경 구동을 42관절 CPG/반사로 연결했을 때의 지속 보행·조향 정확성.
- 정적 장애물 재컴파일 뒤 native 상태 보존 및 충돌 broadphase 정상 동작.
- MuJoCo mjSTATE_INTEGRATION + CPG/reflex 체크포인트 연속성.
- 실제 감각 광선/가림·접촉력 단위·CPU 성능·장시간 여러 seed 실험.
- 실제 브라우저 WebSocket과 native offscreen camera.
- 비행은 미검증 이전에 **미구현**. C 전뇌/GPU backend도 **미구현**.

## 재현

```sh
python -m unittest tests.test_core tests.test_server -v
python tests/browser_smoke.py
python tools/verify_physics.py --seconds 3 --output physics_verification.json
```

마지막 명령은 실제 패키지가 없으면 exit 2/BLOCKED입니다. 설치 후 PASS를 얻는 것은 최소 동작 gate일 뿐 생물학적 정확성/장시간 안정성의 승인이 아닙니다. 후속 B 안정화와 C 확장 전에 복수 seed·서로 다른 감각 조건·신경 출력 차단 대조를 실제 몸에서 검사해야 합니다.

원시 로그: `tests/all_console.txt`, `tests/parity_result.json`, `tests/browser_result.json`, `tests/physics_verification.json`.
