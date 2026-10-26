# E188 실행 전 자원 보완: 부모 입력 검증 전달

2026-09-24. 원본 E188=0인 상태에서 등록한다. 기본 [E188 설계](BANC_M3_E188_ALL_INCOMING_DESIGN_20260924.md)의 생물학적 질문, theta0, 전체158706-neuron CUDA,195 incoming/601 endpoints/3초, E185 raw exact 비교, E18652창, F1–F8, M2/UI/F6 보존은 모두 같다.

## 근거와 결정

E185 동일 baseline의 `process_wall_s`는84.3400455초, `body_wall_s`는57.7228189초였다. 새 E188 input/source gate를 CPU1로 측정하면27.8944213초이고, 반복 비용 진단27.9708151초 중 SHA4,098회20.7700918초·경로4,101회7.0753355초였다. 이는 아직 E188의 원본 모델 실행 시간이 아니며 당시 외부 queue가 실행 중이었다. 그러나 기존 실행에 이 검증 시간을 그대로 더하면100초 제한을 초과할 위험이 구체적이다.

원본 감독기의 총150초(실행145+정리5), 도구 내부140초, 실제 E181 child100초와10초 정리 reserve, CPU1/BLAS1/GPU1/원본1회/재시도0, 누적4994초를 늘리지 않는다. **E188 전체 입력 SHA 검증만 부모 감독기의 같은 원본 실행 안으로 옮긴다.** 부모 검증의 모든 비용은150초와 원본 누적 wall에 포함한다. 자식의100초 시계는 자식 모듈 진입부터이며, 기존 E181 `verified_inputs`는 자식에서 그대로 실제 호출한다. 기존 E181 ValidatedGate를 직렬화하거나 위조하지 않는다.

## 검증 전달 계약

부모는 출력을 만들기 전 기존 E188 `gate`로3988입력/103소스(최종 소스 목록 기준)를 모두 실제 SHA 검증한다. 입력별 device/inode/size/mtime_ns를 검증 전후 비교하여 중간 변경도 거부한다. 동결된 감독기 자신과 소스 리뷰 SHA를 확인하고 CPU1 affinity를 적용한 뒤 이 절차를 수행한다.

통과 후 create-only `parent-e188-input-validation.json`에 freeze/review/metadata SHA, 원래 input SHA map, 입력별 stat, 부모 PID·생성시간·검증 완료 monotonic 시각을 남긴다. 부모는 이 파일의 SHA를 자식 CLI `--parent-validation-sha256`으로 전달한다. 자식은 정해진 파일의 SHA와 schema·freeze/review/metadata·input map을 검사하고, 부모 PID가 실제 os.getppid()이며 생성시간이 같고 살아 있는지 확인한다. 발급 후145초 이내여야 한다. 입력별 stat이 검증 직후와 달라졌으면 거부한다. 소스·metadata·설계·보완 문서 SHA는 자식에서도 실제 확인한다. 같은 실행의 부모 검증 기록만 받아들이며, 단독 CLI 실행에 이 인자가 없으면 시작하지 않는다.

실제 Windows 합성 프로세스 검사에서 venv `python.exe`가 base Python을 한 단계 더 실행함을 확인했다. 이 환경에서는 실제 부모 PID 자체가 아니라 **정확히 한 개의 검증된 venv 실행기**를 사이에 둘 수 있다. 실행기가 살아 있고, 그 부모가 영수증 PID이며, 실행 파일이 현재 `sys.executable`이고 실행 인자 전체가 현재 자식과 같아야 한다. 영수증 PID의 생성시간·실제 감독기 명령 검증은 그대로다. 임의 조상 검색이나 다른 실행기의 허용은 없다. 최초 합성 프로세스 실패는 보존하며 신경/물리 원본 실행으로 계산하지 않는다.

부모 검증 완료를 나타내는 새 객체를 E181에 전달하지 않는다. 자식은 원래 E181 전체 SHA 및 baseline 계약 검사를 수행해 얻은 실제 ValidatedGate만 `run_arm`에 전달한다. 이후 원래 readiness2회와 resource monitor, model restore, observer, clock/startup/health/후처리를 그대로 수행한다. 입력 추가 검증의 실행 위치만 바뀌며 모델 계산·자료·관측·과학적 판정은 같다.

## 리뷰 및 보존

Sol 첫 구현·첫수정 및 부모SECOND 이후에 발견한 실행 시간 위험에 대한 부모 잔여 수정이다. 원본 실험 전에 영수증 SHA/다른 부모/오래된 영수증/변경된 input stat의 거부와 기존 검증·관측12개 회귀를 합성 자료로 확인한다. 부모 독립 출력 검산은 부모 영수증과 감독 wall 포함 여부도 확인한다.

이전 freeze/source review/SECOND 기록은 `e188-pre-parent-gate/`에 보존하고 새 소스와 이 문서를 묶어 다시 동결한다. 이전 원본 실행·실패·결과를 덮어쓰는 작업은 없다. 원본 E188=0, 누적4351.0849566997495/4994초·잔여642.9150433002505초, 후보미채택/F6노출0/전체목표active·미완료다.
