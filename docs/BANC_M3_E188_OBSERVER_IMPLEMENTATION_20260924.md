# E188 전체 입력 관측기: 구현 및 실행 전 검토

2026-09-24. [사전 설계](BANC_M3_E188_ALL_INCOMING_DESIGN_20260924.md)의 관측기 구현과 소스 검토를 완료했다. 이 문서는 보행 실행 결과가 아니다. 원본 E188 실행은 아직 0회이고, 185개 미관측 입력의 설명력 및 자연 보행 개선 여부는 미검증이다.

## 구현과 검토

실제 `/root/e188_gpt6sol_all_incoming`을 `gpt-6-sol/high`로 호출했다. 부모 설계 → Sol 구현 → 부모 FIRST → 같은 Sol 첫 수정 → 부모 SECOND → 부모 잔여 수정 순서를 지켰다. 변경 파일은 `tools/observe_c_banc_m3_mn_inputs.py`, `tests/test_c_banc_mn_inputs.py`다. 부모는 별도로 감독기·동결·독립 검산기를 작성했다.

관측기는 기존 E181의 실제 검증 절차와 theta0 baseline을 사용한다. 원래 관측 callback이 성공한 뒤 MN144018의 전체195 incoming rate/mask와 MN rate/drive/adaptation/current를 읽는다. CUDA packed CSR의 행·pre·float32 weight를 검사하고, 신경 및 신체 시계를 진행시키지 않는다. 실패 시 부분 행과 가용성·원래 예외·정리/저장 오류를 각각 남긴다. E185의601 control/30000 native 및 시작·종료 상태가 정확히 일치해야 E186 반력 조건을 붙인다. 기존 E178 계산을 재사용해 601개 endpoint CUDA current를 검산하고, 이전 잔여를185개 새 입력의 합과 부동소수점 누적차로 나눈다.

FIRST에서7개 테스트는 통과했지만, 실제 E187 review의 상태 문자열과 다른 값을 요구해 실행이 차단될 오류를 발견했다. 부분 관측 실패 기록과 예외 보존, 모듈 진입 시각을 기준으로 한 시간 제한도 같은 Sol이 수정했다. SECOND에서10개 테스트가 통과했고, 부모는 terminal 저장 예외를 성공으로 삼킬 수 있는 잔여 오류 및 마지막 monitor 정리 예외를 수정했다. 최종12개 테스트·Ruff·직접 파일 CLI `--help`가 통과했다. 독립 검산기의 합성17행은 zero/mask/subnormal을 포함해 기존 CUDA 수식의 결과와 bit-exact였다.

## 1차 동결의 실행 전 증거

- metadata: `verification/banc-natural-gait-m3-20260919/parent-e188-frozen-metadata.json`, SHA `4f8be3b06b7ea5ab34f45804bdd2065090c6c1400ca8a325bf4ea638776fbd38`.
- freeze: `e188-mn-inputs-freeze.json`, SHA `b37dab29dbb92be290dc8dc16ffb403cd7a4e4e6303c1c259e1ae7bc1a4e8a28`.
- parent source review: `parent-e188-source-review.json`, SHA `23fe166abec836f4db23f72b351f3f51de3a269ffd6946278bc17201c786d012`.
- tool SHA `d90900318feab9a975d443dfd9df4dc3fe308a4feb95ffba46d048d1bd0f7d46`; test SHA `5633ae773591be362ec326d17bba34513de69c56901ae3510e9bad86fa9aa3ee`.
- 실제3,988개 입력과103개 소스의 검증 절차를 모델 생성 없이 통과했다. 실제 runtime CSR·601 current·baseline 동일성은 실행 후 검증 사항이다.

CPU1 사전 검증27.8944초, 같은 절차의 비용 진단27.9708초였다. 비용 진단에서 SHA 확인4,098회20.7701초, 경로 검사4,101회7.0753초가 관측됐다. 당시 외부 queue가 실행 중이었으므로 이를 자원이 비어 있을 때의 실행 시간으로 단정하지 않는다. 원래 E181100초/도구140초/부모150초와 누적4994초 상한은 변경하지 않았다.

## 남은 실행

외부 ComfyUI queue가 running1/pending0으로 관측돼 아직 원본 실행을 시작하지 않았다. 다른 작업을 종료하지 않았다. 실행 전 새 readiness2회가 모두 clear일 때 감독기로 단 한 번 실행하며, 실행 중 queue/GPU monitor와 원본 실패 상태를 보존한다. 입력·소스 검증과 준비 비용은 별도로 기록하고 원본 누적4351.0849566997495/4994초, 잔여642.9150433002505초는 유지한다.

관측기 구현은 보행 후보 채택이나 생물학적 타당성 증거가 아니다. M2/UI/F6 및 F1–F8 계약은 그대로이며 전체 목표는 active·미완료다. 실행 및 독립 검산 후에만 특정 입력의 변화가 잔여 차이를 설명하는지 결론 내린다.

## 실행 시간 보완과 현재 동결

이후 [부모 검증 전달 보완](BANC_M3_E188_PARENT_GATE_AMENDMENT_20260924.md)을 원본 실행 전에 등록했다. 기존3초 baseline의84.3400초에 새 검증27.8944초를 더하면 자식100초 제한에 걸릴 위험이 있어, E188 전체 파일 검증을 같은150초 부모 실행 안으로 옮겼다. 자식은 부모의 검증 영수증 SHA·PID/생성시간·실행 명령·유효 시간·파일 정체성과 stat을 확인한다. 원래 E181 `verified_inputs`와 실제 ValidatedGate는 자식에서 그대로 사용하며, 신경·물리 조건과 기존100/140/150초 제한은 같다. 부모 검증 비용도 원본 누적 wall에 포함한다.

Windows 실제 합성 프로세스 검사에서 venv 실행기가 추가되어 직접 부모 PID 비교가 실패했다. 실패 기록을 보존하고, 정확히 한 개의 같은 executable/arguments를 가진 venv 실행기만 검증하도록 부모가 수정했다. 최종17개 합성 테스트·Ruff·직접 CLI와 실제 Windows 부모/자식 영수증 검증이 통과했다. 새 신경/물리 원본 실행은0회다. 과거 1차 동결·리뷰와 당시 부모 소스는 `verification/banc-natural-gait-m3-20260919/e188-pre-parent-gate/`에 보존했고 원래 SHA가 일치하는지 확인했다.

현재 freeze SHA는 `cbba65213280c21201c150fb35f9d90d6cc265601ce78a1d1f0dcb81255f1e9e`, source review SHA는 `b3f40c61e8bd2d6513439f63be833f6217e3ca1d9d75959ad0db564cc15a25d6`이다. tool SHA는 `c338c108455ea328051733c15566ff6a133176256b3d475cac09c53281c7f852`, test SHA는 `de019cf6882735eb00a9fa87773722e21c1b1e99ef60219a6692dc3aecca43c7`이다.

외부 queue의57번 작업 `cd931e31-e158-41fd-9c24-abe497a49ee0`을 현재 queue 및 해당 작업 history로 다시 확인했다. 같은 작업이 running이고 완료 history는 아직 없었다. 작업이 멈췄다고 가정하거나 종료하지 않았다. 원본 E188 실행은 여전히 준비 완료·자원 대기 상태다.
