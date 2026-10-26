# E178 원본 실행 전 수치 계약 보완

원본 배열값을 아직 읽지 않은 FIRST 리뷰에서 설치 CuPy의 `cupy/cuda/compiler.py:667`이 RawKernel 옵션에 `-ftz=true`를 덧붙이는 것을 확인했다. `flylab/c/adaptive_fused.py:16`도 이 규칙을 명시한다. 따라서 단순한 NumPy float32 계산은 subnormal이 나타나면 실제 CSR 커널과 다르다. E178 V1 freeze/첫 구현/생성 검사 실패는 보존하며, V2는 같은 입력·56row·bit exact 기준·비용·보행 조건을 유지하고 이 실제 연산 규칙만 명시한다. 원본 실행 재시도가 아니다.

CUDA CSR 곱·덧셈은 subnormal 입력과 출력(절대값이 float32 최소 normal보다 작고0이 아닌 값)을 부호를 유지한0으로 처리한다.32lane 합산 순서와 `--fmad=false`는 그대로다. packed 경로의 사전 `rate*mask`는 Torch pointwise이므로 subnormal을 보존하며 다음 CSR 곱에서 FTZ가 적용된다. reference 경로는 `rate*mask`부터 CSR 안에서 FTZ가 적용된다.

E177의 저장 motor_adaptation_patch.source_cuda_implementation은 두 팔 모두 packed다. 후속 네트워크 생성은 이 요청을 계승하며, 실제 CSR의 노드수≤2**18·고유weight bit수≤2**14·edge수<2**31일 때 packed, 아니면 reference로 구현되어 있다. 분석기는 원본 저장 CSR로 이 조건을 확인하고 사용한 경로를 기록한다. 신경/커널 실행 없이 source와 저장값만 사용한다.

`warp_current`는 이 경로를 구분해 bit exact 결과를 생성한다. 해석용 float64 edge 기여는 V1 정의를 유지하며 실제 CSR 곱의 float32 값·FTZ 영향 개수/합도 함께 기록하여 이상적 곱과 하드웨어에서 소실된 값을 혼동하지 않게 한다. direct feedback 전류는 E177 관측기가 Torch pointwise에서 별도 산출하므로 FTZ 없이 원래 식과 비교한다.

V2는 설치 compiler.py와 이 보완 문서를 hash pin한다. 생성 subnormal 입력·중간곱·합산 상쇄·양/음 부호 검산을 추가하고 부모 source gate는 V2 hash에 묶는다. 허용오차 증가나 기준 변경은 없다.
