# E198: 실제 운동신경 작동 범위의 저장 자료 진단

2026-09-24, 부모 사전 설계. 보행 후보나 신경 생리 적합이 아니다.

E197의 양수 prefilter 전달 가족은 고정 E150 입력과 기존 보호 조건에서 기각됐다. 다음 변경 위치를 정하기 위해 **E150 실제 391 MN의 출력 제한이 원래 입력, 적응, 포화 중 어디에 나타나는지** 확인한다. 같은 전달계수 적합을 반복하지 않는다. 감각/운동신경의 반응 차이만으로 보행 원인이나 개선을 주장하지 않는다.

## 근거와 범위

실제 신경은 `phi=max(cap*tanh(a*(drive+J-threshold-z)/cap),0)`, `dr/dt=(phi-r)/tau`, `dz/dt=(g*r-z)/T`를 사용한다. E123/E134/E138은 기존 적응 gain/시간상수를, E176/E189는 기존 연결 효능을 바꿨다. E36의 a/threshold 변경 대상은 core/relay188개였으며 MN391개가 아니었다. a/threshold 변경은 새로운 상태 방정식이 아닌 기존 반응식의 모수 변경이다.

[Azevedo 등(2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7347388/)은 성체 경골 굴근 운동신경의 모집·흥분성·힘·감각 응답 차이를 보고한다. 그러나 해당 연구를 BANC391개에 그대로 대응할 수 없다. 기존 `motor-unit-source/alignment.json`은 개별 slow identity 미해결이며, E17은 형태 면적 예측이지 전기적 생리 검증이 아니다. 일부 Fast/FETi/SETi 이름만으로 전체 세포의 생리 유형을 지정하지 않는다. 느린 운동신경의 자발 발화에도 흥분성 시냅스 입력이 관여하므로 임의의 내재성 tonic bias를 도입할 근거로 쓰지 않는다. 이번 진단에서 면적을 역치로 환산하지 않는다.

## 입력·계산·정합

입력은 E150 원래 `recruitment-signals.npz`, `recruitment-metadata.npz`, `native-recruitment-state.npz` 세 파일뿐이다. 부모가 현재 원본 manifest에 기록된 SHA와 파일을 대조하여 동결한다. 메타데이터와 원본 배열 header만 앞서 읽었고 이 새 작동 범위 계산은 아직 하지 않았다. E85 교사, 미사용 F6, 별도 원본 및 현재 UI를 읽거나 실행하지 않는다.

metadata의 observed446와 motor391은 int64 unique index이며 motor391이 observed의 정확한 부분집합이어야 한다. 신호는2000행, observed446열; native는2001행이며 motor391열이다. 모든 completion/rate availability가 참이어야 하며, 신호의 motor rate가 native[1:]의 동일391순서와 정확히 같아야 한다. control ticks0..2000, network ticks0..40000(step20), 두 시간축 및 예상5ms grid가 절대오차1e-9 이하여야 한다. time0은 계산에 새로 만들지 않는다. 입력 유한성/양수 a,tau,cap, 허용 음수 없는 rate/adaptation을 확인한다.

계산은 float64에서 I=drive+J, u=I-threshold, m=u-z, x=a*m/cap, h=tanh(x), phi=max(cap*h,0)이다. phi는 **경계 시점의 순간 목표 발화율**이며 실제 발화율 r 또는 다음5ms 발화 예측이 아니다. RK4의 내부20step/4stage를 이 표본에서 재현하거나 수치 적분하지 않는다.

세 창을 고정한다: 전체 ticks1..2000, startup1..99, gait100..2000(0.5..10초). 각 MN/창마다 다음을 저장한다.

- u/m/r/phi의 min,p05,median,p95,max. NumPy quantile의 linear 방식을 명시한다.
- m>0인 비율; m<=0인 비율을 **u<=0인 입력 부족**과 **u>0,m<=0인 적응 억제**로 서로 겹치지 않게 나눈다. 세 비율 합은1이다.
- 활성 표본 중 상대기울기 1-h²<=.1인 포화 표본의 전체 창 대비 비율. .1은 기술통계용 사전 구분이며 생물학적 경계나 채택 기준이 아니다.
- `dphi/da=m*(1-h²)`, `dphi/dthreshold=-a*(1-h²)`의 창 RMS. m<=0에서는 모두0으로 정의한다. m=0의 미분 불연속에서0 선택을 명시하며 미래 궤적 감도와 구분한다.
- 원래 MN a/threshold/tau/cap/gain/적응 log tau를 보존한다. 새 값을 선택하지 않는다.

창별로 target 항상0/일부 활성/항상 활성인 MN수, 실제 r가 항상0인 MN수, 입력 부족만 관측된 MN수, 적응 억제가 한 번이라도 관측된 MN수, 포화가 한 번이라도 관측된 MN수를 요약한다. 활성 여부는 엄격한0 비교이며 임의 발화 문턱을 추가하지 않는다. 비율의 MN평균은 단순 균등 평균으로 표시하고 근육 힘/출력 기여로 해석하지 않는다. 결과는 모든391개를 보존하고 유리한 세포를 선별하지 않는다.

## 판정과 다음 결정

기술 정합 실패는 inconclusive이고 자동 원본 재실행하지 않는다. 정합 통과는 유효한 작동 범위 관측이다. 항상 입력 부족인 세포는 양수 slope 변경만으로 해당 저장 상태에서 모집되지 않는다. 적응 억제는 이미 수행한 g/T 실험과 대조해야 하며 같은 sweep을 정당화하지 않는다. 포화는 slope 증가가 직접 반응을 늘리는 근거가 아니다. 서로 다른 제한이 함께 보이면 전체 MN에 동일 면적 규칙을 적용하지 않는다.

결과로 다음 변경을 고르되, 이 관측만으로 유리한 a/threshold 방향·크기를 선택하거나 보행 가설을 통과시키지 않는다. 기존 g/T/W/전달식과 구별되는 후속 조건부 모수 식별 또는 폐회로 개입이 필요하면 별도 사전 설계를 만든다. 원래 F1–F8, E150 최선 산출물 및 실패, M2/UI를 보존한다.

## 구현·리뷰·자원

소유 파일은 `tools/analyze_c_banc_m3_motor_operating_range.py`, `tests/test_c_banc_motor_operating_range.py`다. actual gpt-6-sol/medium에 기존 배열의 정렬·기술통계·실패 보존 구현을 위임한다. 부모가 설계와 독립 oracle를 작성하고 FIRST → 같은 Sol 첫 수정 → 부모 SECOND/잔여 수정 순서를 지킨다. Sol은 합성 자료만 시험하며 원본 분석은 부모 최종 source gate 뒤 한 번 실행한다. 결과 metadata에 동결/소스 SHA, 원본 계산1회, fit/neural/physics0 및 경계를 기록한다. 출력은 결과 JSON과 per-MN NPZ 및 오류 기록으로 충분하며 새 검증 프레임워크를 만들지 않는다.

입력 SHA 검증을 수치 열람 전에 수행하고 새 출력 디렉터리만 사용한다. 예외는 진단0/partial 여부와 함께 보존하고 다시 던진다. CPU1, GPU0, 내부15초/외부20초/감독30초, 출력8MiB, 원본1회/retry0. 기존 누적5030.738693899719/5144초에서30초를 예약하며 상한은 늘리지 않는다. 준비·합성·부모 재계산은 별도 기록하고 미계측 비용은0이 아니다. 새 model/GraphStore/StateStore/CUDA/신경/물리/F6는0이다.
