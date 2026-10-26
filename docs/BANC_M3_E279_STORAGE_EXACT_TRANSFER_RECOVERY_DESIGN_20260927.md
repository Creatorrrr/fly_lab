# E279 — 저장 표현을 기준으로 한 E278 기술 복구

2026-09-27 root 사전 설계. E278은 중립 초기 전체상태의 메모리 표현 비교에서 중단된 `invalid_not_tested`다. 과학 신경/물리0, 후보 생성0이며 setup 비용과23.010962100001052벽초를 보존한다. 남긴 중립 final 전체checkpoint를 원본과 비교하면 허용된 source layer 이외 값·배열·자료형 차이는0이다(`parent-e278-neutral-state-diff.json`). 원래 StateStore는 tuple을 list로, NumPy scalar를 Python scalar로 저장한다. 메모리 표현을 저장된 표현과 타입까지 비교하면 같은 저장 상태를 거부할 수 있다.

수정은 **디스크 경계를 지나는 초기 상태 비교**에만 StateStore의 정확한 표현 규칙을 적용하는 것이다. ndarray는 dtype·shape·byte를 그대로 보존하고 float의 부호/모든 bit, 원래 wall_s와 모든 필드를 비교한다. 수치 허용오차·필드 제외·다른 초기 상태는 허용하지 않는다. 변환한 tuple/NumPy scalar 경로·자료형을 receipt에 남겨 실제 차이를 식별한다. 기존 E271의 순수 메모리 소유권 비교, 신경 수학, E277 계수, 모든 startup·전달·건강·최종 F 기준은 바꾸지 않는다. 원래 E278 source/archive/실패 receipt는 불변이다.

사용자가 지정한 root FIRST → 같은 Sol 첫 수정 → root SECOND까지 E278에서 완료했다. 이 후속 잔여 오류는 root가 runner와 해당 테스트에서 수정한다. 생성 검사로 저장표현 동등성은 허용하지만 array dtype·1ULP·signed zero·wall 값 변경은 거부함을 확인한다. 원본 재실행 없이 이미 남긴 전체 상태를 먼저 대조한다.

그 뒤 새 source/input freeze와 출력 디렉터리를 가진 E279 단회 복구 실행만 허용한다. **E278의 실패한 원본을 성공으로 덮거나 자동 retry하지 않는다.** 과학 질문과 단계는 E278과 동일하다: neutral .005초 exact → 최신 E277 한 후보 .5초 → 원래 strict startup 통과일 때만 연속1.5초. 이번에도 후보 채택/10초/F6/최종 보행 주장은 없다. 디스크 표현 비교가 실패하면 후보 생성0으로 다시 기술 진단한다. 실제 기준 미달이면 그 값을 기록하고 그대로 중단한다.

현재 원본 누적7784.5995371996/7905초, 잔여120.40046280040042초. E279 감독 한도 **115초**, 내부 과학85초+정리여유30초로 이전보다 줄인다. 기존배정을 늘리지 않는다. RSS6GiB/추가CUDA4GiB/output2GiB/free disk4GiB/thread1/GPU단독, body≤2/scientificneural≤6020ticks/physics≤15050steps와 실제setup 별도계측을 유지한다. 이전실패 비용·M2/E150/UI/미노출F6와 F1–F8은 그대로 보존한다.
