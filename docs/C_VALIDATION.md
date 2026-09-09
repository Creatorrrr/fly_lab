# FLY LAB C 0.3.0 — 로컬 검증 결과

실행일: 2026-09-09. 작업 경로: `/Users/chasoik/Projects/FLY_LAB`.

**C의 전뇌 계산·실제 몸 연결·개입·기록·복원 경로를 구현하고 실행했습니다. 자연 입력에서의 안정적 탐색, 실제 후진 보행, GPU 수치 일치, 생물학적 타당성은 검증 완료로 보지 않습니다.**

설계 기준은 [조회한 최신 대화 전체](C_DESIGN_REFERENCE.md)입니다. 대화에 언급된 별도 21장 설계서 첨부는 조회 도구에서 제공되지 않았습니다. 구현 범위·명령·가정은 [C_ARCHITECTURE.md](C_ARCHITECTURE.md)에 있습니다.

## 실행

```sh
cd /Users/chasoik/Projects/FLY_LAB
.venv/bin/python run_c.py --port 8766
```

[C 화면](http://127.0.0.1:8766)을 열면 기본 C_SHADOW 모드에서 초기화 후 일시정지합니다. C_STRICT 선택 후 **새 실험**을 누르면 C 출력만 몸에 적용됩니다. **한 단계**는 50모델ms입니다. 이 Mac의 CPU에서는 모델 시간보다 느리게 계산됩니다. **DNp09 전진 출력 직접 자극**은 인위적 개입이며 자연 감각에 의한 보행 성공을 의미하지 않습니다.

## 실제 확인한 결과

| 검증 층 | 결과 | 범위와 근거 |
|---|---|---|
| 전체 회귀검사 | **136/136 통과** | 기존 B 95개 + C 41개. [로그](../verification/c_local_20260909/all_tests.log) |
| B 실제 물리 기준선 | **14/14 통과** | FlyGym/MuJoCo, 0.5모델초, 수평 순변위 2.493864mm, 복원 후 적분 상태 오차 0. [원자료](../verification/c_local_20260909/b_physics.json) |
| 전체 스냅샷 계산 | **통과** | 모든 139,255개 뉴런 계산. 희소 연결 3,732,460쌍, 유효 가중치 연결 3,421,226쌍. [C 보고서](../verification/c_local_20260909/acceptance/report.json) |
| 포트 | **공학적 검토 완료** | 실제 ID와 해시 확인. ORN_DM1 68개 자연 입력, DNa02/DNp09/MDN 출력. 생물학적으로 승인된 감각·운동 복원이라는 뜻은 아님. [binding](../data/fafb783/bindings.json) |
| 신경 출력의 실제 몸 기여 | **확인** | DNp09 직접 자극 중 연결 유지/차단 대조. 같은 초기 상태에서 개입 전 궤적 일치, 두 궤적 평균 3D 거리 차이 0.664642mm. [대조군·개입군 기록](../verification/c_local_20260909/acceptance/motor_contribution/comparison.json) |
| 실제 체크포인트 연속성 | **통과** | 활성 지연 큐가 있는 상태를 복원하고 50ms 계속 계산. 전뇌 전압 최대 차이 0mV, 큐 동일, 물리 적분 상태 차이 0. [C 보고서](../verification/c_local_20260909/acceptance/report.json) |
| 짧은 C 물리 조건 | **8/8 통과** | 각 0.3모델초의 유한 상태·시계·물리 오류 검사. 장기 보행 성공 기준이 아님. 아래 표 참고 |
| 원본/재배선/감각 차단 | **반응 차이 확인** | 동일 합성 입력의 50ms 신경 계산. 195/136/0 스파이크. 실제 몸 없는 비교이며 행동 우월성 주장은 하지 않음. [원자료](../verification/c_local_20260909/topology/comparison.json) |
| 브라우저와 서버 저장 | **확인** | 실제 C_STRICT 자극·시간 진행·8개 원시 신호·기록 저장·체크포인트 복원, 데스크톱/모바일 가로 넘침 없음. [브라우저 증거](../verification/c_local_20260909/browser_verification.json) |
| 실제 기록 재생 | **최종 코드에서 일치** | 브라우저의 0.15초 기록을 API로 재생한 결과와 0.1초 체크포인트에서 0.05초 계속 계산한 결과가 동일. 신경·물리·인코더·감각·활성 개입 상태를 값 단위로 비교. [API 읽기 검증](../verification/c_local_20260909/final_api_readback.json) |
| CUDA 실장치 | **BLOCKED** | Apple M1 Max에서 NVIDIA CUDA/CuPy 실행 불가. 코드 경로는 있으며 실제 CPU–GPU 비교는 미실행 |
| 생물학적 검증 | **미완료** | 모델 가중치, 전달물질 부호, 감각 인코딩과 운동 디코더에 공학적 가정이 있음 |

최종 회귀검사는 Python 3.12.7, NumPy 2.5.3, SciPy 1.18.1, FlyGym 2.1.0, MuJoCo 3.9.0에서 수행했습니다. FlyGym 소스는 B에서 고정한 `ca65a510c2afe6ac61c51df4f274c8d190c2f95f`와 일치합니다. 테스트용 작은 그래프와 가짜 몸은 단위·서버 테스트에서만 사용하고, `acceptance/`와 브라우저 실험에는 실제 다운로드한 전체 그래프와 FlyGym을 사용했습니다.

최종 화면 검토에서 체크포인트 복원 후 모드 선택 상자가 이전 값을 유지하는 오류를 발견해 수정했습니다. 복원한 모드·시드·마찰·메타데이터와 영역 요약을 함께 갱신합니다. 이 표시 수정과 seed telemetry 추가 뒤 [서버·기록·복원 검사 6개](../verification/c_local_20260909/final_delivery_tests.log)를 다시 통과했고 실제 브라우저에서도 복원을 재확인했습니다. 신경과 몸의 적분 결과에는 영향을 주지 않는 변경입니다.

## 물리 실험의 관찰 범위

| 조건 | 모드 | 실제 명령 출처 | 수평 순변위 | yaw 변화 |
|---|---|---|---:|---:|
| seed7, 마찰 0.5, DNp09 자극 | C_STRICT | connectome_lif | 2.4104mm | +0.6608rad |
| seed19, 마찰 1.5, DNp09 자극 | C_STRICT | connectome_lif | 1.8266mm | +0.5977rad |
| seed42, MDN 자극 | C_STRICT | connectome_lif | 0.1948mm | −0.0800rad |
| seed42, DNa02 좌 자극 | C_STRICT | connectome_lif | 1.1248mm | −0.3085rad |
| seed42, DNa02 우 자극 | C_STRICT | connectome_lif | 1.1604mm | +0.2984rad |
| seed7, 장애물, DNp09 자극 | C_ASSISTED | **recovery_supervisor** | 1.3672mm | +0.7133rad |
| seed19, 감각 차단 | C_STRICT | connectome_lif | 약 0mm | 약 0rad |
| seed42, 자연 입력 | C_SHADOW | **legacy_b_rate** | 1.9281mm | −0.1394rad |

좌·우 자극의 몸 조향 부호는 의도한 방향과 일치했습니다. MDN 자극은 음수 전진 명령을 생성했지만 이번 0.3초 궤적의 순이동은 후방이 아니므로 **실제 후진 보행을 검증했다고 보고하지 않습니다**. 장애물 조건은 보조 제어가 구동했으며 이를 C_STRICT의 회피 성공으로 합산하지 않습니다. 세 시드와 마찰 두 값은 짧은 통합 검증이며 통계적인 강건성 증거가 아닙니다.

추가로 C_STRICT, seed42, 자연 스칼라 냄새 입력을 **3.2모델초** 실행했습니다. 기존 fix_2의 3초·1mm 고착 감시는 위치 범위 4.9972mm, 관측 누락 없음으로 통과했습니다. 하지만 **32개 표본 모두 전진 명령 0, 지속적인 yaw 요청**이었고 보행 재개도 관찰하지 못했습니다. 따라서 원시 보고서의 `strict_navigation.status=PASS`는 해당 고착 감시만의 결과입니다. **안정적인 자연 탐색을 승인하는 PASS가 아닙니다.**

원시 보고서를 그대로 보존하고 [별도 해석 JSON](../verification/c_local_20260909/evidence_interpretation.json)에 이 제한을 명시했습니다. 이후 `verify_c.py` 출력에는 `stable_exploration_validated: false`와 전진 명령 표본 수를 직접 포함하도록 보강했습니다. 이 보고용 변경 뒤 장시간 물리 캠페인을 다시 실행하지 않았으며 신경·몸의 수치 적분 코드는 변경하지 않았습니다.

## 데이터와 성능

공식 Codex FAFB v783의 master roster와 연결 자료를 사용했습니다. 원본 5,342,446개 영역별 연결 행을 뉴런 쌍 3,732,460개로 집계했으며 해부학적 접촉 합계는 50,666,648개입니다. Codex에서 이미 적용한 pair-total 5개 이상 기준이 있으므로 모든 미가공 접촉 자료와 같은 범위는 아닙니다. 뉴런 제외는 0개입니다.

미확정 전달물질 뉴런 19,658개와 신경조절성 뉴런 1,677개도 상태 계산에 포함했습니다. 이들의 전달 효력을 명시적으로 0으로 마스킹한 연결은 311,234개입니다. 자연 신경 입력은 냄새 평균만 연결했고, 시각·근접·접촉 등의 자연 신경 포트와 stop 집단은 미완성으로 공개합니다. 이는 이후 생물학적 감각 바인딩 연구가 필요한 부분입니다.

- 그래프 해시: `56fb288462f35781af1df7c66b6b0d36e1a2a1205ace3b155092b225bedfb6d0`
- 포트 해시: `6350020a1d5fdbd1fee25b95a47f3e02ab13c32bb4d571545b8f97a82387dbbc`
- [원본 다운로드·해시](../data/fafb783/raw/download_manifest.json), [bundle manifest](../data/fafb783/bundle/manifest.json).
- M1 Max CPU, 전뇌 단독 50모델ms 계산: **2.6329실제초, 실시간의 0.0190배**. 한 짧은 실행에서 측정한 값이며 장시간 성능 보장은 아닙니다.
- 희소 배열 60,276,384바이트, 신경 상태 14,761,030바이트, 측정 프로세스 peak RSS 552,402,944바이트. 실제 몸 연결은 별도 비용이 추가됩니다.

## 보존과 재현

기존 B의 뇌·몸·감각 모델, B 데이터와 빌드 화면 및 이전 검증 원자료를 보존했습니다. C와 연결하기 위해 B Engine의 readout 경계와 Server의 응답 hook을 수정했고 95개 기존 회귀검사와 실제 물리 기준선을 다시 통과했습니다. 변경 전 기준선은 [해시 목록](../verification/c_local_20260909/b_baseline_hashes.json)과 `b_baseline.tar.gz`에 있습니다. Git 저장소가 아니므로 commit이나 push는 생성하지 않았습니다.

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python tools/verify_c.py --physics --cuda --seconds .3 --navigation-seconds 3.2 --out verification/new_c_run
.venv/bin/python tools/c_topology_control.py --out verification/new_topology_run
```

각 재실행은 새 출력 디렉터리를 사용합니다. 이번 결과는 [verification/c_local_20260909](../verification/c_local_20260909/)에, UI에서 만든 고정 집단 기록은 [run-6363ba0110a4](../artifacts/c/runs/run-6363ba0110a4/manifest.json)에 남아 있습니다. 이 기록은 15개 신호 표본/8개 뉴런, 종료 tick 1500, 누락 0이며 전체 뉴런 전압을 저장한 기록은 아닙니다.
