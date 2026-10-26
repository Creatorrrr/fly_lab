# E171 V2: 저장 필드 계약 정정

기본 수치/판정/보존 계약은 `BANC_M3_E171_SAVED_QR_CAPACITY_DESIGN_20260923.md`(SHA852a16b9f2f20fd7fdc00c2e63619820d81f7da3e052fbabd0fbb68bfd2e91c7)를 그대로 따른다. 이 문서는 V1 최초 원본 실패 후, 새 적합을 시작하기 전에 작성한 prospective 수정이다. V1 출력·소스·freeze·gate는 별도 그대로 보존한다. 원본 process1/fit0, 감독0.5126641999231651초, 누적3046.003456999897초다. 원인은 부모 설계가 메모리 내부 키 pair_start/pair_end를 archive 저장 키로 잘못 취급한 것이다. 가설 판정은 invalid/not_tested이며 숫자 미달이나 신경 표현력의 반증이 아니다.

ZIP central directory만 확인한 결과 두 archive 모두 실제 필드는 `affine_rows_velocity_start`, `affine_rows_velocity_end`였다. V2는 내부 canonical `pair_start`/`pair_end`를 이 저장 필드로 명시적으로 연결한다. `allowed_keys`는 내부 canonical 목록이며 `archive_key_map`에 위의 두 exact alias를 동결한다. 나머지 저장 키는 동일하다. 원래 tick100..1922/101..1923을 그대로 확인하고 재생성·대체하지 않는다. metadata preflight는 모든43개 whitelist 키를 alias 적용 후 ZIP `getinfo`로 존재 확인하고, 그 뒤에만 np.load의 선택된 키 값 접근을 허용한다. 전체 원본 feature/bank/rate/state 접근은 계속0이며 zip metadata 검사는 수치 자료 열람이 아니다.

고정12회 SVD, float64 cutoff, signed 계수, 동일 QR/행수/threshold/해상도와 F1–F8·M2/UI/F6 계약은 변경하지 않는다. V1 allowance는 소비됐으며 자동 retry는0이다. 원인이 확인된 이 V2에만 별도 source gate 후 process1/fit12를 새로 배정한다. CPU1/internal90/external105/supervisor120초/raw128MiB를 유지하고, 이전 누적을 초기화하지 않고 bound를3166에서3167초(ceil3046.003456999897+120)로 정정한다. 새 출력은 `e171-saved-qr-capacity-v2/`다. V1 및 V2 비용/실행 수는 별도로 합산한다.

부모 SECOND 리뷰 이후 남은 수정이라는 사용자 절차에 따라 부모가 이 schema 정정과 alias/missing-member 생성 시험을 수행한다. 새 실험 원본을 열기 전에 변경된 소스·시험·V2 freeze·gate를 hash로 고정한다. 서브에이전트의 실제 모델 gpt-6-sol/high 및 이전 두 리뷰·수정 증거는 그대로다. 새로운 후보 채택은 없으며 전체 자연 보행 목표는 active/pending이다.
