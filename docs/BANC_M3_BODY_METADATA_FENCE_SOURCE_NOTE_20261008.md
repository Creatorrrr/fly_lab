# 신체 입력 단계의 반복 검증 경로 — ROOT Source 관찰, 2026-10-08

실제 c-body는 268.84초에 내부 deadline으로 실패했고 durable 신체 결과가 없었다. Constructor, 신경 계산, 물리 계산 각각의 실제 작업량과 시간은 UNKNOWN이다. 아래는 원래 입력을 읽지 않고 현재 Source와 두 metadata descriptor만 읽어 얻은 별도의 관찰이다. 실행 시간 측정이나 병목 확정이 아니다.

현재 `tools/verify_c_banc_m3_startup_ray.py:826`의 `arm_data()`에서 후보 입력 경로가 끝까지 진행한다면 명시적으로 호출하는 `admission.fence("pre_array")`는 382회다. 계산은 최초 1회 + admitted Original 파일 134개 + plain NPY header 126개 + native constructor asset 115개 + metadata/graph/state 경계 3회 + selected 역할 3회다. Descriptor의 경로만 세었으며 Original payload/header/hash는 읽지 않았다. Nested helper의 추가 fence는 이 수에 포함하지 않았다. 조기 종료한 c-body가 실제로 382회를 실행했다고 주장하지 않는다.

`tools/banc_m3_contact_learning_admission.py:1037`은 이 경계를 `_cold_fence()`로 보낸다. 각 호출은 request/grant/inputs/roster/contract/Source freeze/before ledger 7종 metadata의 fresh exact read, 전체 Source closure의 현재 bytes/hash, inherited retry/history Source metadata, live ledger의 모든 역사 비용 및 owner 상태, `validate_history()`의 실제 비용과 archive read를 다시 검증한다. 따라서 수백 개 입력마다 전체 Source 및 과거 검증을 반복하는 비용은 신체 시험의 별도 병목 가설이다. Count conformance나 Native C# 비교 helper 개선은 이 Python 경로를 변경하지 않는다.

| 읽은 Source/metadata | bytes | SHA256 |
|---|---:|---|
| tools/verify_c_banc_m3_startup_ray.py | 77875 | a89bf6053cdc6774db6efe30640bf5a5e6c0f0a42eb7253a09f718c3e8003a14 |
| tools/banc_m3_contact_learning_admission.py | 104010 | 2855307ab64a87e5824989e8d525a989d5024d83c67220f899b7ae4c8bc58a69 |
| verification/banc-natural-gait-m3-20260919/parent-e345-complete-input-inventory.json | 110837 | 0a3008d05b41e9a307bb7e251e312efa9b6be544de2e03a93aa78646429bb9b8 |
| verification/banc-natural-gait-m3-20260919/parent-e344-native-dependency-descriptor-v2.json | 62058 | e37cf17bd381412cadd4de55926c34be154d4c2b30a92aa185b055a040dc15e1 |

다음 성능 판단은 기존 phase/fence timing이 실제 측정값을 출판한 뒤에 한다. 반복 검증이 시간을 지배하면 worker 범위에서 immutable Source/history 검증과 fresh deadline/live owner/resource guards를 나누는 구조를 검토한다. Constructor/load가 지배하면 입력 및 body 구성 경로를, 실제 신경/물리 계산이 지배하면 해당 CUDA 경로를 검토한다. 어느 경우든 기존 Source 변경 탐지·정확한 owner·이력과 비용·검증 qualification을 약화하거나 검증하지 않은 proof cache를 도입하지 않는다. 구조 변경은 별도의 ROOT 설계 → 실제 Astra/max 리뷰 → ROOT 반영 → Sol 구현 → ROOT 리뷰 절차를 거친다.

현재 새 신체 실행·canonical 변경·qualification·속도 개선·보행 성공은 없다. 이 Source 관찰은 Count 진단의 범위를 신체 성능 증거로 확대하지 않고 이후 실제 측정 및 최적화 방향을 구분하는 근거다. F1–F8, M2/UI, BANC provenance, unused F6, 모든 기존 실패·비용·자원 한도는 유지한다.
