# Startup ray 이후 실제 폐루프 보행 개선 설계

ROOT 설계. 사용자 최신 지시에 따라 범위 안의 가역적 개발·실험·자원 배정은 재승인 없이 진행한다. 다음 변경은 실제 GPT-6 Astra/max 설계 검토 → ROOT 반영 → GPT-6.1 Sol 적절한 effort 구현 → ROOT FIRST → 같은 Sol sole R1 → ROOT SECOND/잔여 수정 순서를 따른다.

## 실제 확인된 상태

`parent-startup-ray-20261007-j`의 input/ray/candidate audit/neutral body/candidate body/두 body audit/close 8개 실제 역할은 모두 exit0, 종료·once 비용 확인을 마쳤다. 최신 Source freeze15 SHA687cafab7bc16130f275e0c6c21f8e0fb6bb8079a98bff1cc8667b6461da4a4f. 독립 raw audit와 body audit는 통과했으나 보행 short screen은 두 arm 모두 실패했다.

R 네 창과 independent backward, neutral full99 재현, actual float32 배포 대응은 실제 CUDA에서 확인됐다. alpha[.0009,.00075,.0005,.00025,.0001] 중 처음 적격은 .0001이다. R 개선 .0008977462253523666; candidate startup q RMS .01773407450114629/max .03988620046095437, tendon RMS .0013637288790340596/max .013190496149836584. 규칙과 임계값은 변경하지 않았다.

실제1.5초/300controls/6000 neural/15000physics 비교: neutral forward1.6857973223001235mm/slip1.9427288085211483mm/s/bout.21s/minupright.9255629161268596, candidate forward1.605876943242406mm/slip1.9605032379998013mm/s/bout.21s/minupright.9256346073321408. 둘 다 nonfoot0, startup4/health pass. candidate는 forward/slip/bout/6leg qualified cycle gate 실패로 reject이며 배포하지 않는다.

주의: qualified cycle0은 .3초 eligible forward bout가 없는 분석 결과다. 다리 무운동 증거가 아니다. 저장 trace .5–1.5초의 모든 다리는 load 전환5–11회, 발끝 높이 .03–.78mm 범위로 움직였다. 접촉 전환은 보행 주기가 아니다. 기준을 낮춰 주기 성공으로 바꾸지 않는다.

## 다음 가설과 분기

H1: saved sensory drive에서 선택한 R-only 가중치 descent는 실제 감각·접촉 폐루프의 추진/지지 타이밍을 최적화하지 않아서 .0001 후보도 short gait를 악화시킨다. 대안 설명 H2는 명령과 실제 관절/발 위치의 추종 오차 또는 지지된 접촉에서의 미끄러짐이다. 이 둘을 saved 실제 trace/command/native만으로 먼저 구분한다. 새 alpha sweep, 큰 학습, 무조건10초 연장은 하지 않는다.

actual Astra의 중간 관찰: .0001의 R 총합 개선 안에도 창100/164는 각각 악화하고228/292가 개선한다. 창별 signed 값과 startup margin을 모두 보존한다. H1/H2는 서로 배타적이지 않으므로 saved 진단으로 원인을 강제로 선택하지 않는다. 결과 분기는 관측된 연관/혼합/미식별이며 인과 판정이 필요한 경우에만 최소 추가 관측으로 진행한다.

1. 기존 두 arm의 freeze된 control/native/trace/evaluation과 기존 evaluator만 재사용해 command 변화, 다리별 unloaded/loaded 발 전후 이동, native slip, forward speed sign/지속성, contact 동시 지지 수, 기하학적 knee95–5폭을 같은 공식 .5≤t<1.5초 창으로 계산한다. 진동 자체는 descriptive만; 기존 qualified cycle/bout 판정 불변. timestamp/몸 좌표/leg order/단위를 명시하고 raw hash를 묶는다. 수치 진단은 모델/physics/Original constructor를 실행하지 않는다.
2. 저장된 live command의 R 네 창 오차를 saved-input R과 비교하되 neural state/drive가 저장되지 않아 동일 신경 rollout 재현이나 차이의 원인은 입증할 수 없다. lean trace는 measured42 joint angle도 버렸으므로 command-vs-measured tracking은 UNOBSERVED다. geometric knee는 measured joint42와 같은 좌표가 아니며 근거 없이 직접 차감하지 않는다. 필요한 관측이 빠진 경우 이를 H1/H2 지지로 오인하지 않고 고정 중립/후보 최소 관측을 포함한 bounded actual closed-loop 진단을 다음 구체 단계로 설계한다. 목표점을 향한 보행이나 생물학적 기전으로 해석하지 않는다.
3. H1 우세 시 다음 적격 후보는 실제 closed-loop 감각을 기록한 창의 temporal successor/coordination objective와 startup 제약을 같이 사용한다. 기존 SmoothTemporalObjective를 재사용하고 새 external runtime phase/CPG/teacher/trajectory/root 보정은 추가하지 않는다. H2 우세 시 고정된 source decoder/body에서 추종·접촉 병목을 먼저 해결하는 구체적 가역 변경을 설계한다. 현 단계에서 어느 분기든 학습 성공을 미리 선언하지 않는다.

## 자원과 완료 경계

173 실제 debits 보존. direct generated98.09000000000002/300, shared826.3020156000001/1700; Original11023.894960899506/global16200, 새과학 local1003.779/6000. jobs/pending/holds0. 기존 +6000은 한 번만 발급됐고 모든 rebind resource_increment0이다. 남은 local4996.221을 다음 과학 판단에 유지한다. saved 진단은 outer120/child100/cleanup20, 한 worker, output16MiB, CPU only, GPU/새body work0으로 한정하고 실제 full-return을 once 기록한다. 추가 quota 필요 시 ROOT가 근거와 증분을 기록해 배정한다. 리셋이나 과거 비용 삭제는 금지다.

진단 완료는 원인 분기를 결정하는 좁은 결과다. 전체 F1–F8, 자연 보행, F6 미사용 상태, F7 목표점/stop/restart, 영상/UI/생물학 기전은 미완료로 유지한다. M2/UI·해부학 provenance·E150 partial/과거실패·root GOAL_PLAN.md를 보존한다. 이후 실제 short screen을 통과한 동일 후보만10초 F3–F5, F6/F7/F8로 승격한다.
