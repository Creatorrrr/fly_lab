# E165 실행 경로 오류의 단일 복구 보충 계약

기존 설계와 freeze는 수정하지 않는다. 첫 원본 발송의 ModuleNotFoundError와 모든 출력·감독 기록을 보존한다. Python이 tools/ 파일을 직접 실행할 때 저장소 root를 import 경로에 넣지 못한 실행 결함이다. --help와 in-process 생성 pytest는 이 결함을 검출하지 못했다. 부모 검증의 누락이며 과학적 반증이 아니다.

첫 발송은 process1/재시도0, 감독17.22746449999977초, StateStore0/신체0/shadow0/forward0/Jacobian0/신경0/물리0/fit0이었다. 입력 계산 전 실패했다. 원래 실패 디렉터리를 재사용하지 않는다. 전체 E165 프로세스 시도 수를2로 명시적으로 확장하되 추가1회만 허용하며 자동 재시도는 여전히0이다. 사용자 승인된 같은 유한 목표 안의 기술 복구이며 새로운 후보나 수치 탐색이 아니다.

부모 SECOND 리뷰의 잔여 직접 수정으로 script bootstrap에 실제 ROOT를 sys.path에 넣었다. 별도 프로세스를 무관한 cwd/PYTHONPATH 없음에서 실행해 같은 script bootstrap 후 flylab.c.storage import를 확인한다. 기존 고정 샘플·해시·신체 설정·수식·집계·영점대·원래 F1–F8는 그대로다. 첫 결과를 새로운 이름으로 지우거나 자원 한도를 초기화하지 않는다.

첫 감독 이후 보수 누적2835.2274644999998초를2836으로 올림한다. 기존 누적 상한2938초 안에서 복구 감독102초/외부92초를 할당하고 child 내부90초/새 forward 전 저장 여유10초를 유지한다. CPU1/GPU0, StateStore1/신체1/shadow1/settling2000/forward2000/Jacobian60000, 신경·과학적 물리·fit0, raw192MiB/free2GiB다. child의 기존 freeze/timing에는 원래 발송의 한도 문맥이 보존되며, 이번 감독 기록이 더 엄격한 외부 한도와 합산 비용을 판정한다.

원본3293 고정 파일 전후 검사와 별도로 첫 실패/감독/gate 파일의 bytes 해시를 보충 freeze에 고정하고 복구 전후 확인한다. 새 v3 source gate는 두 현재 소스 SHA, 이 보충 계약 SHA 및 보충 freeze SHA를 갖는다. 새 출력 e165-command-lift-recovery, 새 감독 e165-command-lift-recovery-supervision은 기존에 없어야 한다. 정확히1회 발송하고 timeout/error/수치 실패 여부와 무관하게 같은 복구를 반복하지 않는다. 유효한 수치 진단일 때만 저장 raw의 독립 검산과 과학 해석을 수행한다. 전체 자연 보행 목표는 미완료다.
