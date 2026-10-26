# E230: 원본 1회 기술 실패, 효과 판정 불가

E230의 대조군은 E229와 같은 0.2초/2000 native steps를 완료했다. 경골 한 관절 고정군은 control1620의 첫 변경을 완료한 뒤 `physical.write(...)`에서 `AttributeError: 'tuple' object has no attribute 'write'`로 중단됐다. 파일 스트림 변수 `physical`을 첫 물리 차이 계산용 tuple로 덮어쓴 것이 원인이다. 부모 FIRST 소스 리뷰와 11개 합성 테스트는 이 실제 분기를 놓쳤다. 소스·원본·실패 전표를 수정하거나 성공으로 바꾸지 않는다.

부모 저장자료 검산 317개는 대조군의 전체 raw 및 start/final 일치, 변경군의 600-step lead 일치, 남은 650 native 상태·접촉·Jacobian, joint26 외 명령 불변을 확인했다. 변경군의 명령 및 native 상태는 control1620까지 있으나 `physical.jsonl`은 control1619까지만 저장됐다. 전체 비교/변경군 metrics는 생성되지 않았다. 따라서 가설은 **완전 창에서 미시험**, 후보 채택은 false다. 부분 건강이나 첫 차이를 보행 개선으로 해석하지 않는다.

원본1/retry0/body1/shadow1, 실제 과학2650steps/53controls/0.265초, 접촉6815/Jac13170. 신경/CUDA/fit/solver0. 감독11.377564099995652초, 내부8.162306300000637초, 구성2.895544900005916초, 프로세스 peak1,295,900,672B. 모든 소유 프로세스 종료, 강제종료0. 누적6062.944038699661/6094초·잔여31.055961300338822초다. 부모 저장 검산 내부1.0071845000056783초/명령1.4193583초는 별도다.

증거는 `verification/banc-natural-gait-m3-20260919/e230-rf-tibia-hold/`, `parent-e230-supervisor.json`, `parent-e230-failure-review.json`이다. 같은 Sol의 첫 수정은 별도 E231 기술 복구로 제한한다. E230 재시도·원본 덮어쓰기 없이 고정 가설/창/기준을 보존한다. E225 startup·10초·적격주기 미달, E150 최선, M2/UI/F6미노출, F1–F8 및 전체 목표 미완료를 유지한다.
