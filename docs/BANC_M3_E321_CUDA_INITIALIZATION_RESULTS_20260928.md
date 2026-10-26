# E321 생성 CUDA 진단 결과

네 개의 독립 프로세스에서 같은 float64 행렬·벡터를 생성했다. 원본 BANC 입력, 모델, 신체, 최적화는 사용하지 않았다. actual gpt-6-sol/medium 구현과 같은 Sol의 첫 수정 후 root가 재검토했다.

| 초기화 순서 | 첫 96×1173 GEMV | 미분 검사 안의 다음 GEMV | 뒤의 96×391·42×96 GEMV |
| --- | --- | --- | --- |
| Torch만 사용 | 통과 | 통과 | 통과 |
| CuPy import → Torch import → CuPy context → Torch CUDA | 통과 | CUBLAS_STATUS_INVALID_VALUE | 같은 오류 |
| Torch import → CuPy import → CuPy context → Torch CUDA | 통과 | 통과 | 통과 |
| Torch CUDA 초기화 → CuPy import/context | 통과 | 통과 | 통과 |

통과 조건은 동기화 뒤 NumPy 결과와 rtol=atol=1e-12로 대조했다. 최대 순방향 절대 오차는 6.039613253960852e-14, 벡터 미분 오차는 1.4210854715202004e-14이며 행렬 미분 오차는 0이다. 실패 조건의 미분은 backward에 진입하기 전 두 번째 GEMV에서 중단됐다. 따라서 이 오류를 신경망 미분식의 실패로 해석하지 않는다.

환경은 Torch 2.14.0+cu130, CuPy 14.2.0, CuPy runtime 13020, RTX 4080이었다. 실패 조건에서만 torch/lib와 nvidia/cu13/bin/x86_64 두 경로의 cublasLt64_13.dll이 함께 관찰됐다. 나머지 세 조건에서는 Torch 경로의 cuBLAS 계열만 관찰됐다. 초기화 순서에 따른 오류 재현은 확인됐지만, DLL 내부의 어떤 결합이 오류를 일으켰는지는 아직 규명하지 않았다.

생성 CUDA 감독은 합계 8.742947200022172초, host tree 최대 871223296 bytes였다. Torch peak reserved 최대 24MiB, 초기화 뒤 관측한 device 사용 증가 최대 78MiB다. 외부 nvidia-smi 시작·종료 관측은 같았다. Context를 포함한 순간 최대 사용량을 연속 측정한 것은 아니며, WDDM 외부 메모리 관측과 CUDA memGetInfo를 같은 지표로 뺄셈하지 않았다. 원본 누적 비용은 이 진단으로 늘지 않았다.

이에 따라 full-horizon 도구의 resource meter에서 CuPy 의존을 제거하고 torch.cuda.mem_get_info로 Torch CUDA를 먼저 초기화했다. 환경이나 설치 파일은 바꾸지 않았다. 실제 수정 meter를 먼저 호출한 후 E317 생성 신경망을 실행한 E322 검증은 6종 순방향 상태가 byte-exact였고, 독립 dense 식과의 미분 상대 L2 오차 최대 2.1378049347405584e-6로 통과했다. 생성 CUDA 감독 5.132241초, 추가 device 관측 80MiB였다.

E322의 원본 실행은 별도의 root CLI 경로 오류로 원본 payload를 열기 전에 중단됐다. 이 실패와 3.277557599998545초는 보존·합산했고, E323에서 기존 -m 모듈 실행 방식으로 수정했다. E323의 실제 원본 검증 결과는 별도로 판정한다. E321/E322 생성 검사의 통과는 자연 보행 검증을 뜻하지 않는다.

원시 증거: verification/banc-natural-gait-m3-20260919/parent-e321-generated-audit.json 및 parent-e321-{torch_only,cupy_first,torch_first,torch_cuda_first}.json. 각 기록은 순서, 예외, 동기화, shape/dtype/stride, DLL 경로, 자원 관측과 source hash를 보존한다.
