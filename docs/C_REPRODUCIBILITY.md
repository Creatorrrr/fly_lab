# C 실행과 독립 재현

이 문서는 2026-09-09에 시작해 2026-09-10에 검증한 개발본의 설치·데이터·증빙 경로다. 과거 배수나 짧은 물리 검사로 자연 행동을 승인하지 않는다. 현재 결과는 [개발 결과](C_IMPLEMENTATION_20260910.md)를 확인한다.

## macOS Apple silicon

Python 3.12로 별도 환경을 만든다. 이 Mac에서 Python 3.12.7, Torch 2.14.0, FlyGym 2.1.0, MuJoCo 3.9.0을 실제 사용했다. 아래 제약 파일은 macOS arm64에서 새 가상환경 설치를 확인한 목록이며, Linux CUDA 검증을 대신하지 않는다.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-mps.txt -c constraints-macos-arm64.txt
.venv/bin/python run_c.py --doctor
```

CPU 환경은 `requirements-c.txt`를 사용한다. CUDA는 여기에 해당 호스트 CUDA와 맞는 CuPy 패키지가 필요하며 이 Mac에서는 승인하지 않았다. CUDA용 잠금 환경을 검증했다고 주장하지 않는다.

## 제공된 데이터로 실행

소스 Git만으로 대용량 데이터가 포함되지는 않는다. 배포 ZIP에 포함된 `data/fafb783/raw/`, `bundle/`과 기존 binding을 함께 풀면 기본 경로로 실행할 수 있다. 체크포인트는 정확히 같은 graph/binding/model/runtime 신원을 요구한다.

```sh
.venv/bin/python tools/verify_c.py --backend exp_lif_mps --out verification/my-data-check
.venv/bin/python run_c.py --backend exp_lif_mps
```

`verify_c.py` 기본 scope는 독립 고정 reference와 비교하는 `full-snapshot`이다. 작은 테스트 회로는 명시적 `--scope loaded-graph`에서 계산만 검증할 수 있다. 필수 FAIL은 종료 1, 미확보·불완전은 종료 2이며, 물리를 요청하지 않은 성공은 신경 계산 검증이다.

## 새 자료 획득과 재빌드

기존 binding 파일을 덮어쓰지 않는다. 비어 있는 새로운 획득 경로를 사용한다.

```sh
.venv/bin/python tools/prepare_c.py --destination data/acquisitions/my-fafb-build
.venv/bin/python run_c.py --backend exp_lif_mps \
  --graph data/acquisitions/my-fafb-build/bundle \
  --bindings data/acquisitions/my-fafb-build/bindings.json
```

공식 서버에서 자료를 얻을 수 없다면 실패가 보존된다. 이미 고정 원자료가 있으면 `--raw-source data/fafb783/raw`를 추가한다. 빌더는 평균 후각, 양측 후각 v1/v2, 제한된 시각·머리 접촉 프로파일을 함께 만든다. 추가 입력 가설은 다음처럼 별도 생성한다.

```sh
.venv/bin/python tools/build_c_poisson_profile.py --graph data/acquisitions/my-fafb-build/bundle \
  --base data/acquisitions/my-fafb-build/bindings-bilateral-geosmin-v2.json \
  --out data/acquisitions/my-fafb-build/bindings-odor-poisson-current-hypothesis-v1.json
```

새 `content-v2` 신원은 획득 시각·경로와 내용 신원을 분리한다. 이번 재빌드의 graph hash는 `c914dc5ea54fc1744ed7bf03b56ec190ae26b0d418722c057a70f4f7e75d8a8f`, 기존 고정 bundle은 `56fb288462f35781af1df7c66b6b0d36e1a2a1205ace3b155092b225bedfb6d0`이다. roster·주석·해부 연결·모델 가중치 신원은 모두 같았다. 이것은 구형 체크포인트의 hash를 바꾸어도 된다는 뜻이 아니다.

0.6.0은 검증된 v3 형식의 0.3.0·0.4.0·0.5.0 체크포인트를 읽을 수 있다. 새 저장은 0.6.0으로 표시해 구형 실행기가 수용체 방향 필터·접촉 이력을 조용히 버리지 못하게 한다. 새 프로파일로 기존 신경 상태를 자동 변환하지 않는다. 서버 시작 전에 복원하려면 `run_c.py --restore-checkpoint <체크포인트 이름>`을 사용한다. 구형 코드와의 실제 후속 계산 비교, 새 프로파일과 행동 한계는 [0.6 결과](C06_REPAIR_RESULTS_20260910.md)에 기록했다.

원자료가 독립 reference와 달라지면 중단하고 차이를 검토한다. 자동으로 새로운 자료를 정답 reference로 승인하지 않는다. 형제 `bindings*.json`은 최대 16개를 발견하며, 선택하지 않은 손상/비호환 프로파일은 이유와 함께 사용 불가로 표시한다.

## 장시간 실험

```sh
.venv/bin/python tools/run_c_campaign.py --out verification/my-pilot
# 미리 만든 spec 사용
.venv/bin/python tools/run_c_campaign.py --spec my-spec.json --out verification/my-campaign \
  --cancel-file verification/cancel-my-campaign
# 취소를 요청할 때 이 파일을 생성한다.
touch verification/cancel-my-campaign
# 취소 파일을 제거한 후 같은 명령에 --resume을 붙인다.
```

재개에는 동일 소스·데이터·binding·backend·runtime·spec이 필요하다. 서로 다른 코드로 이어서 실행하려면 새 실험을 시작한다. worker가 살아 있는 동안 OS 파일 잠금이 중복 시작을 막는다. 창을 닫아도 독립 worker는 계산하며, 서버의 정상 종료는 소유 worker에 취소를 요청한다. 일반 화면 재생은 여전히 요청이 있을 때만 진행한다.

진행 파일의 `cases`는 요약이다. `result_file`과 `result_sha256`이 각 조건의 상세 판정·고착 창·실제 궤적 지표를 가리킨다. 재개할 때 결과와 청크의 해시를 검사한다. 상세 결과를 화면 갱신마다 다시 저장하지 않으며 원시 자료를 삭제하거나 다운샘플링하지 않는다.

기본 파일럿은 3개 seed × 6장면 × 3모드 × 10모델초다. 실행 완료와 과제 통과는 별도 필드다. 먹이 접근은 30초 과제이므로 10초에 도달하지 않은 조건을 실패 또는 성공으로 단정하지 않는다. 위험 회피는 대응 대조군이 없으면 NOT_EVALUATED다.

## 검증 패키지

```sh
.venv/bin/python tools/pack_c_evidence.py --out artifacts/releases/my-evidence.zip \
  --include data/fafb783/raw --include data/fafb783/bundle \
  --include verification/my-campaign
.venv/bin/python tools/pack_c_evidence.py --verify artifacts/releases/my-evidence.zip
```

ZIP의 EVIDENCE_MANIFEST는 모든 구성 파일의 크기·SHA256을 기록한다. 자기 자신을 포함하는 순환 해시는 만들지 않는다. 인덱스는 ZIP 자체의 SHA256을 기록한다. 구성 파일이 빠지거나 바뀌면 검증이 실패한다. 파일 해시 PASS는 개별 행동 과제의 PASS를 뜻하지 않는다. 외부 업로드나 GitHub 릴리스 발행은 이 도구가 수행하지 않는다.
