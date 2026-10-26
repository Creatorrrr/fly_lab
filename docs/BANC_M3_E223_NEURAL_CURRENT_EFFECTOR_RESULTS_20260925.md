# E223: 원래 신경 상태를 보존하는 병렬 근육 변환의 CUDA 검증

E223은 생성한 8개·39개 신경 fixture에서 통과했다. 원본 BANC와 물리 몸체는 실행하지 않았다. 새 effector는 원래 운동 신경의 발화율이 아니라, 해당 신경의 E/I/drive와 인과적 기억으로부터 계산하는 별도 근육 변환 가설이다. F2 및 자연 보행 완료를 주장하지 않는다.

- 같은 외부 입력·mask에서 기존 GPU 모델의 전체 rate/adaptation과 바이트 단위 일치. 39개 fixture는 한 CSR row의 38개 nonzero로 warp32 경계를 포함한다.
- 독립 CPU 계산 최대 오차: neuronal rate `1.4901161193847656e-8`, effector `1.280420214300193e-12`, fast memory `7.662797349775041e-11`.
- CUDA graph/direct 실행 및 디스크 복원 후 미래 상태가 바이트 단위 일치. 잘못된 checkpoint 10개를 상태 변경 없이 거절했다.
- 영 입력·영 상태 출력은 0. 입력 차단에 effector가 변했고, effector와 원래 MN rate가 실제로 다른 값임을 확인했다.
- Sol/high 구현 → 부모 FIRST에서 control마다 전체 상태·계수 전송 지적 → 같은 Sol 첫 수정 → 부모 SECOND 10 tests/Ruff 통과. 부모 잔여 코드 수정 0회.

생성 GPU 과학 계산 104 ticks, 실행 warmup 20 ticks, capture 10 ticks를 분리 기록했다. supervisor `15.061590600002091s`, 내부 `14.094818900004611s`, tree peak RSS `1,023,279,104B`, CUDA 관측 추가 `60,817,408B`, 출력 `65,913B`; 소유 PID 모두 종료, 재시도 0. 원본 누적은 `5818.539935299679/6004s`, 잔여 `185.4600647003208s` 그대로다. 부모 저장 배열 감사 내부 `.0362248s`, process `.3838368s`는 별도 비용이다.

증거: `verification/banc-natural-gait-m3-20260919/parent-e223-source-review.json`, `parent-e223-output-review.json`, `parent-e223-supervisor.json`, `e223-neural-current-effector-generated/`. 원본 신경망·몸체 성능은 다음 E224에서 분리 검증한다. E150 best, M2, UI, F1–F8 판정, 미사용 F6 상태를 보존했다.
