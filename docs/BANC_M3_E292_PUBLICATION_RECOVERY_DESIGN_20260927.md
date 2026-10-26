# E292 — 기록 충돌 수정 후 동일 고정 후보의 신체 판별

E291은 neutral .005초의 trace/native/recruitment와 전체 과학 endpoint exact를 통과했지만 candidate control9에서 `advance-ledger.json.tmp → advance-ledger.json` 교체가 WinError5로 실패했다. 실제 과학 실행은 신경200ticks/물리500steps, setup560+capture280/settle4000, 감독48.796525700018초이며 신체2개 모두 종료했다. 후보 자원은 최소.80540568로 변했으나 보행 판정 구간에 도달하지 않았다. 이는 기술적 미완료이고 생물학적 후보의 기각·성공이 아니다. root 독립 raw/SHA/clock/whole audit로 확인했고 원본누적7918.178555299532/7995초, 예약0으로 이월한다.

Root가 실행 중 PowerShell Get-Content로 같은 ledger를 읽은 시점과 일치해 Windows 읽기 handle의 replace 방해가 가능한 원인이다. 정확한 오류 당시 handle 주체는 확인되지 않았다. 원본 파일을 보존하고 실제 Windows 읽기 잠금으로 원래 writer의 실패 및 수정 writer의 회복을 작은 임시 파일에서 먼저 재현한다.

E291 구현은 이미 같은 Sol의 첫 수정과 root SECOND를 거쳤으므로, 사용자 지정 순서에 따라 이번 잔여 수정은 root가 수행한다. E289/E290/기존 과학·평가 경로는 변경하지 않는다. E291 runner의 NativeDependencies.write만 bounded atomic receipt writer를 사용한다. WinError5/32/33 PermissionError에 한해 동일 값의 publication을 최대6회 시도하고 .005/.01/.02/.04/.08초만 기다린다(최대.155초). 과학 control은 재실행하지 않는다. 각 오류·재시도·대기는 별도 append 기록으로 보존하며 기록 실패/영구 권한 오류/다른 오류는 fail-closed한다. 실제 감독시간은 전부 비용에 포함한다. 실행 중 교체되는 ledger를 열어 모니터링하지 않는다.

생성 검증은 기존67CPU fake tests/Ruff와 실제 Windows 작은 파일 충돌, 주입 transient/permanent/다른오류, bytes/값 보존을 포함한다. 원본/CUDA/body/solver0, 감독60초/RSS2GiB. E291 실패 source/freeze/raw/cost는 이미 별도 보존되어 있다. 수정 source 승인 후 별도 E292 freeze/output과 prospectively125초 원본1/retry0를 배정한다. stage배정의 필요한 증가를 실행 전 기록하며 누적비용은 초기화하지 않는다.

과학 후보는 E291과 완전히 동일한 roster/u=.05/tau=.150/original E150 neutral/decoder/graph/body다. E291 원본에서는 startup이나1.5초 판정을 하지 못했으므로 E292에서 처음 해당 고정 판정을 수행한다. Neutral .005 exact→actual startup .5 strict→통과 시 같은session1.5를 [E291 설계](BANC_M3_E291_RECURRENT_RESOURCE_BODY_DESIGN_20260927.md) 그대로 적용한다. whole endpoint의 독립실측wall 비용만 별도 보존하고 모든 과학 필드 exact, F1–F8/M2/UI/F6미노출/보행 기준/신경수치/CUDA/tolerance는 불변이다. 실패 데이터에 맞춘 u/tau/seed 변경·grid·fit은 없다. E292가 유효한 보행 판정을 내리면 그 결과에 따라 다음 설계를 선택한다.
