# E254 실행 인자 오류와 단일 복구 실행 사전 기록 — 2026-09-25

첫 부모 launcher는 E251의 --source-review/--source-review-sha256/--output을 남겨 두었다. E254 실제 CLI는 --review/--review-sha256/--out이므로 argparse가 exit2로 종료했다. 이는 부모 구현 오류다. gate, 원본 입력 로딩, 모델 생성, CUDA 신경 계산, 결과 생성은 시작되지 않았다. 감독 비용 .29089390000444837초와 stdout/stderr/argv/소스/리뷰를 보존한다. H254의 지지·기각 결과는 없다.

부모 SECOND 후 남은 수정은 사용자가 정한 순서대로 부모가 수행한다. Sol의 분석 도구와 테스트/고정 입력/질문은 그대로 두고, 새 launcher-v1에 정확한 옵션 이름을 적용한다. 새 source-review-v1은 이전 launcher와 실패 기록, 이번 복구 기록 및 새 launcher/부모 auditor를 모두 hash로 연결한다. 이전 파일을 덮어쓰지 않는다.

이번 기록은 E254의 OS 프로세스 launch retry0 계약에 대한 실행 전 명시적 수정이다. 전체 launch 시도는 최대2회, 그중 시작 전 실패1회와 추가 복구1회(launch retry1)이다. 과학 engine 실행은 최대1회, 네트워크 생성2/setup120중capture40/과학12000tick/원래고정602관측이며 새로운 신체/fit/solver는0이다. 이미 계산된 실험을 반복하거나 가설·기준을 바꾸는 복구가 아니다.

복구 직전 누적6855.519873499641/6933초, 잔여77.480126500359초. 한 번의65초 감독 예약, 배정증액0. internal55/external60/supervisor65와 나머지 메모리/출력 제한은 E254 그대로다. 이전 비용과 새 비용을 모두 누적하고, 생성 CLI parse 검사/준비/부모감사는 별도로 기록한다. F1–F8/E150/M2/UI/F6/기존 실패를 보존한다.
