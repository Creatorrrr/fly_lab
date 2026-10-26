# E197 이후 신경 생성 경로와 생리 자료 검토

전체 F1–F8은 아직 미완료다. E197은 고정 입력의 prefilter 근육 전달 가족을 기각했으며 새 자연 보행 후보를 얻지 못했다. 최선 E150의 10초 전진과 미달한 F4/F5를 그대로 보존한다.

## 실제 경로와 기존 변경

`flylab/c/adaptive_rate.py:224–255`와 packed CUDA 식은 signed recurrent current를 rectified half-tanh로 변환하고 발화율 r 및 적응 z를 RK4로 적분한다. E150은 tau=.02, a=1, threshold=7.5, cap=200의 기본 반응식 위에 core adaptation 예외와 MN별 gain/시간상수, downstream monotone 모집층을 사용한다. BANC 해부학적 연결과 이 가정된 생리 계산을 구별한다.

E123/E128/E134/E138의 gain·적응 시간상수 적합은 이미 같은 r/z 방정식을 사용했다. E176의 기존 감각-MN 연결 복구와 E189의 단일 excitatory relay 약화도 실제 출력을 바꿨지만 지속 전진을 개선하지 못했다. E187/E188/E190의 사건별 방향은 일관된 단일 MN/힘 원인을 지지하지 않는다. 따라서 같은 gain/tau/edge sweep을 다음 방법으로 선택하지 않는다. MN별 a/threshold는 검토한 M3 가족에 없었지만 **미시험이라는 이유만으로 유리한 방향이 정해지는 것은 아니다**. 이것은 새 상태 방정식도 아니다.

## 형태 면적과 생리 유형의 증거 경계

actual gpt-6-sol/high 읽기 감사와 부모 소스 확인을 수행했다. E6 `rate-banc-02/size-alignment.npz`는 면적/측정 mask를, E17 `morphology-size-01/alignment-and-predictions.npz`는 root/region과 형태 기반 면적 예측·지원 범위를 보관한다. `flylab/c/research_morphology.py`는 `electrical_size_validation=False`를 명시한다. 예측 면적의 검증은 전기적 역치 검증이 아니다.

`tools/build_c_banc.py:27–44`는 원래 cell_type 및 source annotation을 보존하고 `flylab/c/neuromuscular.py`는 몸쪽/다리/근육으로 MN을 대응시킨다. 기존 `verification/body-control-20260911/motor-unit-source/current-morphology-inventory.json`의 LF tibia19개에는 `tibia_flexor_Fast`, FETi/SETi 이름이 있다. 그러나 같은 폴더의 `alignment.json`은 `individual_slow_id_resolved=false`, `neural_control_validated=false`다. 검토한 자료에는 전체391개에 대한 검증된 slow/intermediate/fast 대응표가 없다. 이 사실을 모든 개별 생리 표지의 부재라고 확대하지 않는다.

[Azevedo 등(2020)의 성체 경골 굴근 연구](https://pmc.ncbi.nlm.nih.gov/articles/PMC7347388/)는 유형별 흥분성·힘 생성·감각 응답 차이와 모집 순서를 지지하지만, 단일 면적→BANC391 역치 변환식을 제공하지 않는다. 느린 세포의 자발 발화가 nicotinic 차단에 감소한 관측도 있으므로 임의의 내재성 tonic bias를 그 논문으로 정당화하지 않는다. 해당 연구 결과를 다른 다리·모든 근육·현재 신체 서보의 실측 교정으로 간주하지 않는다.

E36은 기존 면적 가설의 a/s,threshold×s를 **core/relay188개**에 적용한 실험이었다. MN391개 실험으로 기록하지 않는다. 또한 `CellSizeTrial`은 core schema를 요구하므로 E150 checkpoint를 직접 받지 않는다. 억지 schema 변경 후 이전 생성자를 쓰면 이후 MN gain/tau 및 모집 wrapper를 보존하지 못한다. 현재 실패를 일으킨 제품 버그라고 주장하지 않으며 후속 코드에서 이 오래된 wrapper를 재사용하지 않는다.

## 선택한 다음 단계

면적만으로 MN 생리를 일괄 배정하지 않는다. [E198 사전 설계](BANC_M3_E198_MOTOR_OPERATING_RANGE_DESIGN_20260924.md)의 기존 E150 순간 전류·적응·발화 진단을 먼저 수행한다. 이미 기록된2000 경계의446개 관측 가운데391 MN을 정확히 대조하여 입력 부족/적응 억제/포화의 분포와 정적 반응 미분을 확인한다. 5ms 표본을 새로운 신경 적분이나 미래 궤적 감도로 해석하지 않는다. 변경 방향과 크기의 식별에는 별도 조건부 적합/실제 폐회로 검증이 필요하다.

이 검토 자체의 새 원본 fit/neural/physics/F6는0이다. E198은 별도 설계와 소스 검토 뒤 한 번 실행하며 이전 E197까지의 누적5030.738693899719/5144초를 보존한다. 연구·구현·검토 비용은 별도이며 미계측을0으로 처리하지 않는다. M2와 활성 UI는 변경하지 않는다.
