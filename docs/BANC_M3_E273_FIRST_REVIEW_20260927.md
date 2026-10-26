# E273 FIRST 리뷰와 메모리 조건 수정

2026-09-27 root 리뷰. FIRST source `92ddb786ac6a4dce7bf37e0765484cff7848fd18f60551f7990c18328aeae23f`, tests `48e613a8709a2a29964e501bc3dd0e2ba65c98077b147194487275b36e92be06`.

생성 검사 68개와 독립 수식/중심차분 검사는 통과했다. 8회 대표 solve의 36개 shared-array 비교도 모두 bit exact다. 그러나 측정 reference median .411067초, precomputed median 1.553071초, speedup .264680으로 사전 1.5배 기준은 실패했다. 정확성 지원과 성능 실패를 구분하며 새 기본 경로는 아직 채택하지 않는다.

원본/optimizer/BANC/body 실행은 0이다. 생성 supervisor 21.5156387초, RSS 1,625,882,624 bytes. 상세 결과는 `parent-e273-first-benchmark.json`, 생성 계약/로그/supervisor는 같은 verification 폴더에 남는다. 이 결과는 삭제하거나 재시도로 대체하지 않는다.

비교 종료 후 읽기 전용 GPU 조회에서 RTX 4080 사용 메모리 15,863 MiB, GPU 사용률 0%였다. LM Studio의 llama-server와 ComfyUI 프로세스 등이 살아 있다. 개별 프로세스의 점유량은 WDDM 조회에서 N/A이므로 특정 앱을 원인으로 단정하지 않는다. reference의 세 측정은 .411/.419/.400초, precomputed는 1.716/.146/1.553초로 큰 변동이 있으며 current 준비는 .005–.008초, 차이는 forward 호출/대기에 집중된다. 메모리 압력에 의한 eviction/paging은 이 관찰과 일치하는 **미검증 설명**이다.

## 같은 Sol의 첫 수정 요청

실험 조건을 prospective v2로 바꾸어, 대표 입력을 중복 상주시킨 두 engine 대신 **한 engine**에서 `forward_implementation`만 reference/precomputed로 바꾸며 같은 8-solve 순서를 실행한다. 두 경로 모두 매 solve 완전 초기 상태에서 시작하는 기존 계약을 따른다. 입력 device buffer 포인터가 모든 실행에서 같음을 확인한다. 실제 fitting도 같은 입력을 중복 업로드할 필요가 없으므로 이번 변경은 시험 자체의 불필요한 메모리 중복을 제거한다.

각 solve timer는 반환 직후 끝낸다. shared scientific 배열 9개를 그 뒤 `host_copy`로 CPU에 복사하고 GPU 결과의 지역 참조를 제거한다. 이후 solve의 `_start_response`에서 이전 응답 참조가 해제되게 한다. CPU 배열에서 dtype/shape/raw bytes를 비교한다. 두 경로의 큰 device 출력들을 비교를 위해 동시에 유지하지 않는다. current literal 비교와 derivative 정확성은 이미 별도 작은 생성 검사에 있다.

constructor 1회, warmup reference/precomputed 각 1회, timed reference/precomputed 각 3회, 총 solve 8회는 고정한다. 재시도, 추가 warmup, 가장 좋은 값만 선택, 수학/정밀도/크기/계수/1.5배 기준 변경은 없다. constructor/JIT, solve, 타이머 밖 host copy 비용과 memory-pool used/total 및 `memGetInfo`를 기록한다. 외부 앱이나 활성 UI를 종료하지 않는다. v2가 실패하면 E273 성능 채택은 계속 보류한다.

이 첫 수정은 benchmark helper/test/docstring과 필요한 생성 검증만 수정한다. kernel/engine 제품 수학은 변경하지 않는다. 에이전트는 CPU/Ruff만 실행하고, root가 새 source를 SECOND 리뷰한 뒤 새로운 생성 CUDA 계약을 예약한다. benchmark 실패를 숨기지 않고 E274 원본 실행은 E273 최종 gate 이후에만 진행한다.
