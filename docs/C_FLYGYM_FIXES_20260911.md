# CUDA 캠페인·배치 오류 수정 — 2026-09-11

직전 점검에서 재현한 E3–E5와 전체 회귀 중 확인한 Windows 진단 출력 오류를 수정했다. 기존 실시간성·Warp 정밀도·BANC 보행·전신 근육/감각 교정의 미충족 판정은 이번 수정의 완료 범위에 포함하지 않는다.

## 수정 내용

| 항목 | 수정 후 동작 | 검증 |
|---|---|---|
| **E3: 복원 중 OOM** | 저장된 세계의 원래 번호와 체크포인트를 유지하면서 배치를 절반씩 분할한다. 남은 분할 작업도 저널에 저장한다. 한 세계도 할당할 수 없으면 `PAUSED`와 `waiting_for_memory`로 종료하여 `--resume`으로 다시 시도할 수 있다. 계산 중 오류는 계속 `FAILED`로 처리한다. | 실제 CuPy 할당 실패, 반복 중지·재개, 분할 후 재시작, 세계 취소, 기록 채널 보존 검사를 통과했다. |
| **E4: 거짓 완료 저널** | 작업 목록·취소 상태·진행 tick·현재/대기/완료 청크를 교차 검사한다. 체크포인트 파일 해시, 그래프·바인딩·시드·모드·몸·세계 신원, 신경·encoder·몸·GPU 제어 시각을 대조한다. 완료 증거가 없거나 서로 모순되면 입력을 변경하기 전에 거부한다. | tick 0의 `COMPLETE`, 목표 tick만 꾸민 `COMPLETE`, 누락·손상·잘못 매핑한 체크포인트, 중복 청크를 거부했다. 정상 완료 재개와 실행 전 취소는 허용했다. |
| **E5: 종료 후 진행** | 닫은 배치의 step/pause/resume/cancel/checkpoint를 실행 전에 차단한다. 배치의 활성 목록과 실행 참조를 정리하고, 일부 자원 정리가 실패해도 나머지 자원을 닫는다. 닫은 엔진의 진행·기록 시작도 차단한다. | 종료 후 재호출에서 신경·encoder·몸·엔진 시각과 신경 배열이 그대로였다. 파일도 생성되지 않았다. 중복 종료와 정리 실패 검사도 통과했다. |
| **Windows 진단 출력** | JSON 파일은 UTF-8로 읽고 쓴다. 콘솔 JSON은 ASCII 이스케이프를 사용하고, 내부 Python 자식 프로세스는 UTF-8 입출력으로 실행한다. | cp949와 UTF-8 비활성화를 강제로 적용한 검사에서 한글·대시 문자를 보존했다. 의존성이 없으면 JSON과 종료 코드 2를 남기며 `BLOCKED`를 보고했다. |

캠페인 저널은 임시 파일을 flush/fsync한 뒤 교체한다. 새 체크포인트의 해시를 저널에 보관하고, 중단 전에 만들어진 체크포인트 디렉터리나 기록 세그먼트와 이름이 충돌하지 않도록 했다. 유효한 기존 v1 저널은 새 필드가 없어도 내부 파일 해시와 진행 증거를 검증하여 읽는다.

**분할 복원의 물리 의미:** 원래 체크포인트에서 선택한 세계의 신경·감각·제어기와 MuJoCo 적분 상태를 복원한다. 접촉·제약 배열은 공유 인덱스를 가지므로 단순히 잘라 쓰지 않고 새 물리 epoch에서 다시 구성한다. 원래 세계 번호와 이 사실을 체크포인트 및 기록 이벤트의 `regroup_origin`에 남긴다. 이후 궤적의 비트 단위 일치나 CPU 물리와의 일치를 보장하는 기능은 아니다.

## 실행 결과

- **최종 전체 회귀: 327개 중 306 통과 / 21 생략, 실패 0**, 107.111초. 생략은 실제 MPS 장치가 필요한 20개와 별도 CUDA 검증 도구를 지정한 1개다. [실행 로그](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flygym-fixes-20260911/regression-native-06.log).
- 관련 배치·캠페인 검사 **25개 통과**. 새 경계 조건 검사 10개를 포함하며 전체 회귀와 중복되므로 수를 합산하지 않는다. [관련 검사](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flygym-fixes-20260911/fixes-targeted-01.log).
- 첫 전체 회귀에서 기존 진단 도구의 인코딩 실패 4개를 발견했다. 수정 후 강제 cp949 조건의 진단 검사 **13개 통과**, 최종 전체 회귀에서도 통과했다. [최초 실패](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flygym-fixes-20260911/regression-native-02.log), [인코딩 회귀](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flygym-fixes-20260911/encoding-04.log).
- 수정한 Python 파일 11개의 Ruff `F,E9` 검사 통과. 새 캠페인 모듈·검사·CLI 파일 4개는 포맷 검사도 통과했다. [정적 검사](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flygym-fixes-20260911/static-05.log).

### 전뇌 2세계의 실제 CUDA 복구

139,255뉴런·3,732,460연결을 각 세계에 유지하고, C_STRICT / 낱눈·DoOR v4 감각 / 세계당 512개 기록 채널로 별도 검증했다. 실행 중인 브라우저 실험에 명령을 보내지 않은 독립 프로세스였다.

1. 두 세계를 각각 tick 50(5 ms)에 저장했다.
2. 독립 CuPy 풀을 8 KiB로 제한해 2세계 복원과 축소된 1세계 복원 모두 실제 `OutOfMemoryError`가 나게 했다. 재호출도 `PAUSED`로 끝났고 두 작업은 tick 50과 원본 체크포인트를 유지했다.
3. 풀 제한을 16 KiB로 늘린 뒤 두 세계를 하나씩 복원했다. 신경 전체 상태, encoder, 감각, 개입, 기록 집단 등 15개 상태 항목의 해시가 각 원본 세계와 같았다. 두 작업 모두 목표 tick 150(15 ms)까지 완료했다.
4. 네 기록 세그먼트 모두 512채널·누락 0이며, 이벤트·몸·신호 파일의 해시를 확인했다. 정상 완료 저널을 다시 열어도 완료 상태를 유지했다.
5. 같은 전뇌 그래프로 거짓 완료 두 경우를 거부하고, 종료한 배치가 tick 150에서 더 진행하지 않는 것도 확인했다.

장치 전체의 VRAM을 소진한 시험은 아니다. 검증 범위는 2세계의 짧은 복구 경로이며 장시간 처리율을 측정한 결과가 아니다. [전뇌 검증 결과](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flygym-fixes-20260911/full-brain-03/report.json), [재현 스크립트](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flygym-fixes-20260911/verify_full_recovery.py).

## 실행 중인 실험 적용

서버를 갱신하기 직전 화면의 **3.050초 / C_STRICT / CUDA 신경 / CPU 물리 / 일시정지 / 기록 꺼짐** 상태를 `checkpoint-a7c38f007bce`에 저장했다. 수정된 서버를 이 상태로 실행하고 브라우저를 다시 연결했다.

갱신 후 `checkpoint-9f5286086690`과 대조하여 신경·몸·감각·encoder·세계·설정·기록 집단·개입 등 모든 저장 필드가 같음을 확인했다. 연결을 다시 만들 때 증가하는 `subscription_epoch`만 1→2였다. 실험을 진행시키지 않았으며 서버는 HTTP 200, stderr 예외 없음이다. [상태 보존 대조](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flygym-fixes-20260911/live-preservation.json), [서버 확인](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flygym-fixes-20260911/live-bootstrap.json).

핵심 구현은 [캠페인](C:/Users/ckthd/Dev/Projects/fly_lab/flylab/c/cuda_campaign.py), [저널 검증](C:/Users/ckthd/Dev/Projects/fly_lab/flylab/c/cuda_campaign_state.py), [배치 생명주기와 선택 복원](C:/Users/ckthd/Dev/Projects/fly_lab/flylab/c/batch.py), [물리 적분 상태 재구성](C:/Users/ckthd/Dev/Projects/fly_lab/flylab/warp_batch.py)에 있다. 기존 단일 실험 캠페인 모듈은 변경하지 않았다.
