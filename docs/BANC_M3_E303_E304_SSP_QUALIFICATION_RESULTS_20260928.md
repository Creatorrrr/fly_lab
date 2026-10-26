# E303/E304 SSP 적분기 및 연결 자격 결과

2026-09-28 root FIRST 검토. 새 SSPRecoveryResourceNetwork와 whole wrapper는 **작은 생성 신경망의 CPU/CUDA·저장복원 자격을 통과**했다. 실제 전체 BANC와 신체 보행은 아직 실행하지 않았다. E301 실패, E302 최초 음수 중간 상태, 과거 비용과 고정 F1–F8를 보존한다.

Root 설계 → 실제 gpt-6-sol/xhigh E303 및 gpt-6-sol/high E304 구현 → root FIRST 소스 검토/검증 순서다. 제출 후 수정할 결함이 발견되지 않아 R1을 요구하지 않았다. E303 두 SHA는 `63de1e5df6d7b3e02de2a54a3d8861b63ba8e28fd4098d8b8fd8931f2852ee3c`, `41f2b443d7de9b7c0376d2ee01e656d234cb8e8b1c2b0ffb340bb52f1fca5021`; E304는 `9a782914409fd48c6ada7126fce04bf50ac8c391207d97f7c604b898bcb49e62`, `90eb8149fe402fc8a24be00c03777d9e09650bf24f11feb9571472bee4521fbc`이다. 이전 소스를 수정하지 않았다.

같은 신경/resource ODE에 공동 SSPRK33 3단계를 사용하며 neutral도 같은 적분기를 쓴다. 원래 CSR·파라미터·선택 간선과 u/tau는 유지했다. 모든 신경의 음수/비유한 상태 검사와 resource 범위 검사를 유지하고, 잘못된 상태를 clipping하지 않는다. 새 identity는 기존 RK4와의 차이를 구별한다. 원래 neutral과 미래 상태가 같다는 주장은 하지 않는다. 방법의 조건부 양수 보존 도출과 문헌은 [설계](BANC_M3_E303_SSP_RECOVERY_ENGINE_DESIGN_20260928.md)에 있다.

| 검증 | 관측 결과 |
|---|---|
| Root CPU / Ruff | 32 PASS, CUDA 3 skip / PASS |
| 실제 RTX 4080 CUDA | 3 PASS: neutral 및 active CPU 대비, graph/disk 미래 동일, native32·포인터 guard |
| 독립 41cell/117edge/7tick 계산 | CPU/CUDA 모두 rate 최대차 2.384185791015625e-7, adaptation 0, resource 1.4774848011711583e-12 |
| 다중 lane CSR / 마스크 source | 현재 계산 exact, 출력 mask0 source도 resource 갱신, 비활성 R1 exact |
| E304 whole wrapper | fake scaffold 67검사 PASS; 실제 4cell neural + fake body/adapter의 nonzero disk 복원 후 8tick 미래 exact |
| 독립 float64 수렴 | 고정 smooth toy 오차비 8.1940375/8.0969127; production float32 수렴차수 증거는 아님 |

Root CPU supervisor 8.283775899966713초/RSS626741248B, CUDA supervisor13.171024799987208초/RSS1029758976B로 사전 45초/1.5GiB 및60초/2GiB 범위 내다. CPU suite 실제 tiny26개/67tick과 독립 tiny1개/7tick, oracle2160tick을 계측했다. CUDA 묶음은 suite5개(실제GPU3/CPU2)/440science tick, 독립GPU1개/7tick, wrapper실제GPU4개/23tick이다. 세 묶음의 Python setup/capture step 호출은45/15/60이며 capture 호출을 과학적 전진에 더하지 않는다. 독립 profiler는 setup 실행10/capture기록5/unknown0을 구분했다. Suite Torch 최대 reserved6291456B, 독립4194304B, wrapper 최대 allocated77312B이며 이 수치를 전체 GPU의 연속 peak라고 주장하지 않는다. 모든 supervisor 소유 PID가 종료되었다.

E303 Sol의 모든 사전 시도도 남겼다: pytest3회, 첫26PASS/3FAIL/2skip, 다음29PASS/2skip, 마지막32PASS/3skip. 첫 실패는 smooth fixture의 정류 경계 통과2건과 상속 base gain 계약 위반 fixture1건으로, 허용오차 완화 없이 fixture를 수정했다. Ruff 첫 import style finding도 포함한다. 총 외부14.4173903초, fixture3.946538799966초/CPU3.8125초/RSS615747584B; tiny CPU생성71개, science200tick, float64 oracle6480tick이다. E304 Sol 생성검사 총8.8505665초 및 root4.035349299956579초는 별도 영수증에 보존했다.

핵심 영수증은 `verification/banc-natural-gait-m3-20260919/parent-e303-source-review.json` SHA `021605fe96750e1382a47a407cec791dea88dc4481bd27fcd927677cb41fe530` 및 `parent-e303-tiny-qualification.json` SHA `be348d4ab660b63469a434dedecfb51a96c633af661ddad206dba6faa1426f20`이다. 후자는 원시 stdout/각 검사/계약/supervisor/독립 계산 증거 SHA를 포함한다. E304 실제 연결은 `parent-e304-tiny-integration.json`에 별도 기록했다.

원본 누적은 **8144.108421799498/8210초**, 예약0이다. 이번 실제 fullBANC/body/physics/fit 호출은0, F2/F1–F8/F6/adoption/생물학 주장은 모두 false다. 다음 E305는 동일 실제 첫 입력을 새 SSP neutral/candidate에 각각20tick 적용하는 body 없는 자격 진단이며, runner root 검토와 저장 입력 gate를 통과한 뒤에만 별도 예약한다.
