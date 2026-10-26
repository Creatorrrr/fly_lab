# E321: E320 cuBLAS 초기화 실패의 생성 입력 분리

E320 원본은 첫20 neural tick 뒤 E316 decoder의96×1173 float64 GEMV에서 CUBLAS_STATUS_INVALID_VALUE로 중단됐다. 전체 control완료0, backward/효능방향/신체/optimizer0이며 H320은 not_tested다. 단회 감독8.655897399992682초를 이미 합산하여 원본 누적8577.075507999634/9060초다. E320 원본 retry0과 실패 source/output을 보존한다.

E316/E317 생성 CUDA 및 E316 원래391MN replay는 통과했으나 E320의 resource meter는 CuPy를 먼저 import하고 memGetInfo로 context를 연다. H321은 이 초기화 순서 또는 Windows CUDA DLL 선택이 GEMV 실패를 설명한다는 것이다. 경쟁 설명은 실제 model buffer/stride 또는 full-network 연산 후에만 생기는 오류다. 원본 데이터 없이 같은 GEMV 크기를 만들어 구분한다. 아직 원인으로 단정하지 않는다.

actual gpt-6-sol/medium이 신규 tools/probe_c_banc_m3_cuda_initialization.py 한 파일만 구현한다. Root FIRST→필요시 같은Sol R1→root SECOND→root 잔여수정 순서를 지킨다. 생성 CPU import/CLI 검사는 Sol20초/root20초, CPU1/host1GiB. Sol은 CUDA/원본/모델/fit/신체를 실행하지 않는다. 단순 실험 CLI의 구현을 복제하는 단위테스트는 추가하지 않고 root가 실제 생성 CUDA 비교로 검증한다.

각 조건은 별도 새 Python process다. 모듈 전역에서 torch/cupy를 import하지 않는다. NumPy와 psutil은 공통으로 쓸 수 있다. 조건은 torch_only, cupy_first(import CuPy→import Torch→CuPy memGetInfo→Torch CUDA), torch_first(import Torch→import CuPy→CuPy memGetInfo→Torch CUDA), torch_cuda_first(import Torch→Torch CUDA init→import CuPy→CuPy memGetInfo) 4개로 고정한다. 순서를 명시적으로 기록하며 별도 파일에 결과를 저장한다. 기존 앱 환경변수·DLL·설치·GPU설정은 바꾸지 않는다.

값은 고정된 NumPy float64 생성 행렬/벡터로 만든다. 순서마다96×1173,96×391,42×96의 GEMV와 첫 GEMV의 gradient를 검사한다. Host NumPy의 같은 계산과 finite/allclose를 비교하며 tolerance는 relative1e-12/absolute1e-12다. 예외 시 첫 오류·traceback·시도/host반환/device확인과 가능한 후속 오류를 보존한다. GPU 동기화 이전에 완료를 주장하지 않는다. 결과에는 실제 module/torch/CUDA/CuPy version, 장치, dtype/shape/stride, torch peak memory, 가능한 현재 프로세스의 CUDA/cuBLAS DLL 경로를 기록한다. DLL 열람 불가도 명시하고 그것만으로 실패원인으로 판정하지 않는다.

Root source review 뒤 생성 CUDA 총60초, 각 process≤12초, host2GiB, torch추가256MiB/context포함추가GPU1GiB 범위에서 한 번씩 실행한다. 실제 감독 wall은 생성 비용으로 별도 기록하며 원본 누적은 변하지 않는다. 먼저 실패한 조건도 다음 독립 조건을 막지 않도록 실험 자체의 오류는 결과에 기록하고 CLI는 유효한 진단 기록 게시를 반환한다. 원본계산0/GraphStore0/StateStore0/신체0/fit0이다.

CuPy-first만 재현되고 순서변경에서 사라지면 E320 meter의 초기화 순서만 수정하는 기술회복을 별도 원본 계약으로 검토한다. 모든 조건이 통과하면 이 가설은 지지되지 않으며 실제 matrix layout 또는 kernel 상호작용을 생성 fixture로 분리한다. 모두 실패하면 현재 환경의 GEMV 오류로 범위를 좁힌다. 어느 결과도 긴 미분이나 자연 보행의 반증으로 해석하지 않는다. 기준을 낮추거나 자동 반복하지 않고 새로운 근거와 기술수정 후에만 다음 단회 원본을 계획한다. F1–F8/M2/UI/F6미노출/모든 비용은 보존한다.
