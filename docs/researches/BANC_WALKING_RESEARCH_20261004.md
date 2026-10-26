# BANC 보행 제어 리서치 — 개발 적용 검토

- 정리일: 2026-10-04
- 출처: ChatGPT 대화 [보행 제어 연구 제안](chatgpt-conversation://6ac1c4d7-77a0-83ee-8896-72b3cbd3c43d)
- 채택한 답변: ZIP 첨부 후의 최종 답변, 메시지 `399526be-540c-475d-bb7f-2f96c7fb5b71`.
- 입력 ZIP: `BANC_GPT_PRO_RESEARCH_20261004-135304.zip`
- ZIP SHA256: `3bd3fca9b529ea906ac3f517d7e4afc28cf17cff2e89140e4157e074d1e8dbe3`
- 입력 스냅샷: `2026-10-04T13:53:04.940390+09:00`

## 먼저 읽을 내용

**현재 진행 중인 수정 단계를 결과·실패 원인·자원 사용과 함께 마무리한 다음, 이 리서치를 다음 단계 계획에 반영한다.**

핵심 제안은 **startup을 보존하도록 신경 효능의 업데이트 방향을 선정하고, 같은 후보를 실제 1.5초 폐루프까지 평가하는 것**이다. 이미 구현된 rate 동역학을 새로 도입하자는 초기 답변은 ZIP을 검토한 최종 답변에서 수정되었다.

현재 단계가 마무리되면 먼저 최신 소스·결과와 아래 확인 사항을 대조한다. 관련 가설이 여전히 유효하면 한 가지 주요 변경을 대상으로 다음 실험을 설계한다. 기존 M3 목표, F1–F8, M2/UI, provenance, 실패 이력과 자원 회계를 유지한다.

## 원문 확보 범위와 검증 상태

- [첨부 후 최종 답변 전체](/C:/Users/ckthd/Dev/Projects/fly_lab/docs/researches/BANC_WALKING_RESEARCH_20261004_GPT_PRO.md): 결론부터 A–E 절과 최종 권고까지 저장했다.
- 처음 대화 읽기 도구는 앞 20,000자를 반환했고 `truncated=true`였다. 이후 사용자가 열어둔 로그인 Chrome에서 누락된 C 절 뒷부분·D·E 절·최종 권고를 읽어 보완했다.
- 저장 방식은 앞 20,000자 원문과 브라우저 본문에서 옮긴 뒷부분의 결합이다. 뒷부분의 표·수식·코드 목록은 Markdown으로 변환했으므로 원래 직렬화 형식과 차이가 있을 수 있다.
- 별도 첨부도 다운로드해 바이트를 그대로 저장했다: [K1–K8 코드·실험 근거집](/C:/Users/ckthd/Dev/Projects/fly_lab/docs/researches/BANC_RESEARCH_EVIDENCE_20261004.md), [ZIP 무결성 기록](/C:/Users/ckthd/Dev/Projects/fly_lab/docs/researches/BANC_ATTACHMENT_INTEGRITY_20261004.json).
- 원문의 `:chatgpt-content-reference{index=...}`는 ChatGPT 내부 인용 표식이다. 본문에 명시된 논문·저장소 URL과 위 로컬 첨부를 사용한다.
- 이 문서의 **로컬 확인**은 아래에 나열한 소스의 정적 검사에 한정한다. 논문 원문·외부 저장소·체크포인트를 이번 작업에서 재조사하거나 실행하지 않았다. 외부 근거집의 [코드 확인]·[논문 보고]와 이번 로컬 확인을 구분한다.
- [출처 및 파일 해시 기록](/C:/Users/ckthd/Dev/Projects/fly_lab/docs/researches/BANC_WALKING_RESEARCH_20261004_SOURCE.json)을 함께 보존한다.

## 1. 현재 코드에서 다시 확인한 핵심

아래 파일은 정리 시점에 첨부 스냅샷과 바이트 및 SHA256이 같았다. 최근 M3의 다른 소스·데이터 전체가 동일하다는 뜻은 아니다.

| 확인 사항 | 정적 근거 | 해석 범위 |
|---|---|---|
| 방향 선정은 R 손실의 합산 gradient를 정규화해 θ를 변경한다. `make_direction()`에는 startup 제약 입력이 없다. | `tools/verify_c_banc_m3_reference_parameter_direction.py:327–341, 914–924` | 함수와 호출에서 확인. 이 불일치가 보행 실패의 유일한 원인이라는 것은 미검증 가설이다. |
| 학습 창은 `100, 164, 228, 292`, control dt는 5ms이고 rollout의 `max_controls=4`이다. 각 창에서 저장된 rate·adaptation·bank·activation과 감각 drive를 사용한다. | 같은 reference 파일 `:43, 786–800`; `tools/verify_c_banc_m3_coordination_parameter_direction.py:60, 98, 1224` | 각 창은 20ms. 후보가 만드는 실제 감각·신체 상태의 연속 변화와 동일한 학습 범위라고 볼 수 없다. |
| 후보를 구성한 뒤 control 1부터 300까지 진행하고, control 99에서 startup을 검사한다. | `tools/verify_c_banc_m3_common_descent_body.py:1442–1457`; `tools/verify_c_banc_m3_candidate_body_handoff.py:1251–1279` | 0.495초의 검사에 도달하기 전부터 후보가 적용되는 호출 경로가 존재한다. 이번에 신체 실행은 하지 않았다. |
| 완전 주기는 0.3초 이상 전진 bout 안에서 발의 AEP→PEP→AEP로 센다. | `tools/analyze_c_banc_biological_gait.py:180–240` | 적격 주기 0은 이 정의의 미충족이다. 모든 다리 운동 또는 신경 진동의 부재를 단독으로 입증하지 않는다. |
| rate 모델과 적응 상태의 적분이 이미 있다. | `flylab/c/research_rate.py:3, 146–165`; `flylab/c/adaptive_rate.py:224–255` | “rate 모델을 처음 도입한다”는 제안은 현재 코드에 맞지 않는다. 실제 최근 M3 설정은 해당 경로의 설정·체크포인트로 확인한다. |
| 직접 관절 목표 입력 함수가 있다. | `flylab/body.py:425`부터의 `step_joint_targets()` | 함수의 직접 입력 경로와 CPG 객체 존재를 구분한다. 실제 최종 실행의 우회 제어 여부는 실행 경로별로 검증한다. |

제품 기본 adapter의 시간상수·필터를 최근 M3의 `SSPRecoveryResourceTrial` / `SSPEfficacyTrial` / monotone decoder 경로에 자동으로 대입하지 않는다.

## 2. 리서치가 제안한 우선순위

### 선행 조건 — 현재 실행 복구 단계 종료

문서 정리 중에는 E360/P10의 Ruff B023 및 결과·테스트의 정수 타입 수정안을 검토하고 있었다. 전달 직전의 최신 상태는 P11 승인 대기이며, 세션은 소스와 원장을 확인하면서 추가 검증 실행을 보류한다고 보고했다.

이 기록은 문서 작성 시점의 세션 상태이다. 단계 종료 시점의 최신 결과를 기준으로 다음 계획을 세운다. 생성 검증 성공과 원본 BANC 신체 실행 적격성을 따로 기록하고, 기존 source closure·시간 제한·결과 저장·독립 감사 경로가 실제 후보 평가를 뒷받침하는지 확인한다.

### 1순위 — startup을 보존하는 효능 업데이트

**외부 리서치 제안·미실행:** 현재 그룹별 효능 overlay에서, 저장 입력 명령 손실을 줄이면서 기존 startup 제한을 보존하는 방향을 고른다. 최종 float32 반올림과 backend palette 제약을 확인한다.

국소 gradient 또는 Jacobian 근사는 후보 제안에 쓰는 근사이다. 실제 물리 폐루프 전체의 정확한 민감도나 실제 startup 통과를 보장하지 않는다. 방향이 없으면 해당 파라미터 집합·국소 근사의 한계로 기록한다.

### 2순위 — 연속 학습 구간과 후보가 방문한 상태

**외부 리서치 제안·미실행:** startup 문제가 해결된 뒤 한 주기 이상의 연속 구간과 후보의 실제 신체·감각 상태를 학습 범위에 포함한다.

현재 네 창·4 controls·손실 정규화·감사 계약이 고정되어 있으므로, 단순히 숫자만 늘리지 말고 별도 실험 버전으로 정의한다. DAgger를 사용한다면 임의의 학생 상태에서 유효 행동을 내는 교사 인터페이스가 필요하다. 고정 cadence 궤적을 재생하는 교사의 역할을 먼저 구분한다.

### 3순위 — 감각·운동 인터페이스 교정

**외부 리서치 제안·미실행:** 길항 운동 집단→관절 방향, 고유감각·접촉·하중 입력, 다리 사이 경로, 하행 명령을 순서대로 점검한다.

기본 adapter의 필터·지연이 최근 연구 경로에 실제 적용되는지 확인한 뒤 변경을 선택한다. 다른 BANC 버전의 뉴런 ID는 cell type·좌우·분절·매핑 신뢰도와 함께 대응시킨다.

## 3. 다음 실험을 작성할 때 참고할 구체안

**미실행 제안:** 동일 SSP 체크포인트의 θ=1을 기준군으로 하고, startup 제약으로 선택한 θ 하나를 후보군으로 비교한다. 첫 변경은 방향 선정 규칙으로 한정하고, 그래프·그룹·선택 edge·decoder·신체·학습 창·평가기를 고정하는 안을 우선 검토한다.

이 제안은 현재 단계 종료 후 최신 코드와 실패 이력을 검토하여 구체화한다. 새 실험의 실행 범위와 자원은 현행 개발 workflow 및 기존 승인 범위에 따라 정한다.

### 기존 startup 조건

다음은 현재 `startup99()`에서 정적으로 확인한 값이다.

| 지표 | 한도 |
|---|---:|
| 관절 명령 RMS 오차 | ≤0.018 |
| 관절 명령 최대 오차 | ≤0.04 |
| tendon RMS 오차 | ≤0.009 |
| tendon 최대 오차 | ≤0.025 |

근거: `tools/verify_c_banc_m3_common_descent_body.py:343–369`.

### 기존 1.5초 선별 조건

다음은 현재 `screen()`의 값이며 F1–F8 전체 완료 기준은 기존 목표 계획을 따른다.

| 지표 | 한도 |
|---|---:|
| 전진 | ≥1.6790085779712631 mm |
| native 하중 접촉 slip | ≤1.8026436968249955 mm/s |
| 최장 전진 bout | ≥0.3초 |
| 여섯 다리 적격 완전 주기 | 각각 >0 |
| native health | 통과 |
| startup 네 조건 | 모두 통과 |

근거: `tools/verify_c_banc_m3_candidate_body_handoff.py:1213–1222`.

control 1에서 시작한 같은 후보·인스턴스를 startup 99 controls에서 300 controls까지 이어서 평가한다. 0.5초에 상태를 다시 설정하거나 외부 교사로 시작시켜 선별하지 않는다.

결과는 기술적 무효, 적격 실행의 startup 기각, 적격 실행의 보행 기각, 단기 후보 적격으로 구분한다. 단기 적격 후보는 같은 후보로 10초·미사용 초기 상태·목표 접근/정지 검증을 진행한다. 이 개발 순서는 최종 답변 D 절과 기존 M3 목표에 따른다. F6의 미사용 초기 상태 2개, F7의 기존 6개 목표 조건, F2·F8의 미래 재현·운동 출력 차단·산출물 검증을 그대로 유지한다.

원문은 제약 방향 선정 및 통합을 실행 복구 이후 약 2–4 인일로 추정한다. 이는 외부 계획 추정이며 개발 완료나 벽시계 실행 시간의 보장은 아니다. 기존 prefix 실행의 비용을 고려해 준비·실행·저장·감사를 포함한 실측 비용을 다음 계획에서 산정한다.

## 4. 참고할 논문·코드

아래는 **외부 답변이 다룬 자료와 제안 역할**을 정리한 목록이다. 세부 버전·라이선스·공개 자산 상태와 논문의 결과는 이 작업에서 독립 재검증하지 않았다. 실제 코드를 가져오기 전 원문·공식 저장소·커밋·라이선스·현재 의존성을 확인한다. 자세한 비교표는 원문 확보분 A 절에 있다.

| 자료 | 외부 답변이 제시한 참고 역할 |
|---|---|
| [BANC 논문](https://www.nature.com/articles/s41586-026-10735-w), [BANC-project](https://github.com/htem/BANC-project) | 주석·감각운동 경로·provenance 및 데이터 버전 대응 |
| [Pugliese 논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC13142387/), [공식 코드](https://github.com/smpuglie/Pugliese_2026) | 이미 사용하는 rate·적응 동역학의 대조 기준과 회로 가설 |
| [Sapkal 등 원문](https://www.biorxiv.org/content/10.64898/2026.04.29.721658v2.full) | 관절 내 협응과 다리 사이 결합, 접촉·하중·고유감각의 역할 |
| [FlyGM v3](https://arxiv.org/html/2602.17997v3), [공식 페이지](https://lnsgroup.cc/research/FlyGM/) | 고정 그래프 정책과 물리 신체 학습 방법. 코드 접근·참조 입력의 적합성은 확인 필요 |
| [FlyGym](https://github.com/NeLy-EPFL/flygym), [NeuroMechFly v2 논문](https://www.nature.com/articles/s41592-024-02497-y) | 현재 신체·관측·교사 자산의 재사용 |
| [Flybody 논문](https://www.nature.com/articles/s41586-025-09029-4), [코드](https://github.com/TuragaLab/flybody) | 운동 데이터·교사 비교. 현재 FlyGym 환경과의 의존성 확인 |
| [Flyvis 논문](https://www.nature.com/articles/s41586-024-07939-3), [코드](https://github.com/TuragaLab/flyvis) | 구조를 유지하는 파라미터 공유·학습 제약 |
| [DigitalSphinx2026](https://github.com/Brunton-Lab/DigitalSphinx2026) | 학습 기반 참고와 decoder의 자유도·생물학적 해석 점검 |
| [Guan 등 사전논문](https://arxiv.org/abs/2609.38665) | 리듬 및 길항 협응 가설. 외부 답변도 초록 수준 확인으로 한정 |
| [Shiu 등 논문](https://www.nature.com/articles/s41586-024-07763-9) | 뇌 모델의 감각운동 예측 비교 기준 |

## 5. 개발 세션에서 반영하는 절차

1. 지금 등록된 단계의 수정·검토·결과 기록을 마친다.
2. 이 문서와 최종 답변 전체을 읽고, 최신 소스·결과에서 여전히 유효한 지적을 확인한다.
3. 현행 workflow에 따라 다음 단계의 가설·한 가지 주요 변경·비교군·고정 평가 기준·자원·실패 시 종료 조건을 작성한다.
4. 먼저 같은 후보가 실제 1.5초 평가까지 도달하도록 계획한다. 저장 입력 손실이나 startup margin의 개선은 각각의 진단 결과로 기록한다.
5. M3 장기 목표의 F1–F8, M2/UI, 실패 이력, provenance와 회계를 이어서 유지한다.
6. 원문 E 절에 열거된 활성 SSP·adjoint·decoder 소스, 후보 가중치·checkpoint, startup·교사·보행 raw를 로컬에서 대조하고 결과를 기록한다. 해당 파일들이 연구 ZIP에 없었다는 사실과 현재 로컬에 존재하는지 여부를 구분한다.

## 6. 다음 검토에 필요한 추가 자료

원문 E 절은 연구 ZIP에 포함되지 않은 최근 SSP·adjoint·decoder 구현 9개, 실제 후보·baseline 배열과 체크포인트, startup 및 교사·보행 raw, 설치 환경·신체 자산을 열거한다. 진행 중인 개발 세션은 현재 로컬 자료를 이용해 이 부분을 검토한다. 특히 startup 최대 오차의 시간·관절과 실제/adjoint 연산 차이를 확인하여 제약 방향 선정의 근거를 만든다.

본 정리에서 변경한 것은 리서치 문서뿐이다. 실행 코드·가중치·신체 설정·목표 계획을 수정하거나 신경·GPU·물리 실험을 수행하지 않았다.

