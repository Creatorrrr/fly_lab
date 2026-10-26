# E324: 국소 미분 검증 통과, 연속 실행 후보는 기각

전체 BANC의 고정된 steady200개 control을 50개의 20ms 창으로 나눈 미분은 실제 ±0.1% 변화와 일치했다. 관측 손실 차이3.22844954e-6, 예측3.23001946e-6, 상대오차0.00048604(0.048604%)로 사전 5% 기준을 통과했다. 중립 손실0.002180594640846569는 E323과 일치하며, 모든 창 끝의 rate/adaptation은 E318 원본과 dtype·bytes exact다. q 최대오차3.5527e-15rad다.

국소 minus 손실은0.002178982843862891로 낮아졌지만, 동일 가중치로 원래 시작점에서 리셋 없이1.5초 실행하면0.0021813364998730695로 중립보다 악화됐다. 시작 q RMS0.01821472>0.018, 최대0.05569402>0.04도 실패했다. tendon RMS0.00136386/max0.01319187은 제한 이내다. 조건부 국소 미분의 유효성과 연속 보행 개선을 구분하며 후보는 채택하지 않는다. 신체/optimizer/F6 노출0, F1–F8 미완료다.

Actual gpt-6-sol/xhigh FIRST→root FIRST→같은 Sol R1→root SECOND를 수행했다. Root가 발견한 부분 decoder 횟수 누락, 연속 실행 요청 오기록, 최초 계산 오류를 후속 기록 오류가 덮는 문제를 수정했다. 12개 CPU 검사, Ruff, module 진입점과 lazy import가 통과했다. Root CPU9.349457초, Sol 실패포함 약21.6초다. 생성 CUDA25.627252초에서 독립 dense 미분 상대L2 4.4234e-7, NumPy 손실오차5.55e-17, q 유한차분 상대오차1.71e-10을 확인했다.

원본 감독161.19420989998616초, 신경 forward18000tick/recompute4000/backward4000tick(200block), decoder replay300회다. Host peak 약2.76GB, 추가GPU1,354,760,192B, Torch peak1,298,137,088B였다. 원본 누적8896.9861794996/9060초, 잔여163.0138205004초이며 예약/원본실행중0이다. 이전 기술 실패와 E323 반증의 비용을 그대로 보존한다.

Root 독립 NumPy 감사3613개가 source/input/output, 매 창의 손실·원본 상태, decoder 재생, gradient 합산·실제 f32효능·현재 CSR 변화·조건부 연속분기·시작 제한·계수를 확인했다. 최초 감사는 계산 후 NumPy bool의 JSON 저장에서 실패했으므로 원래 source와 불완전 JSON·감독3.275837초를 보존했다. V2에서 Python bool 변환을 수정하고 상태/가중치 검사도 추가했으며 감독3.280602초에 통과했다. 감사는 원본 신경/CUDA/신체를 재실행하지 않았다.

저장 자료 분석에서 국소 gradient L1=0.00161498/L2=0.000449411, 긴 E323 gradient와 cosine0.04150이었다. 8028그룹의 sign 변경 중 gradient 제곱합98.4286%가 상위10그룹,99.9861%가 상위100그룹에 집중됐다. 이는 동일 폭 sign 변경 대신 크기에 비례하는 방향을 검증할 근거다. 원인을 확정하는 인과 증거는 아니다. 시작 경계에서 조용한36그룹/42edge만으로 예상되는 쌍 손실차이는4.74e-12로 현재 유의성 기준1e-9보다 작다. 경계의 무활동은 SSP 내부 무활동을 입증하지 않으며 이 경로는 이번에 원본 실험으로 진행하지 않는다.

근거: `verification/banc-natural-gait-m3-20260919/parent-e324-original-raw-audit-v2.json`, `parent-e324-original-supervisor.json`, `parent-e324-generated-qualification.json`, `parent-e324-saved-analysis.json`, `e324-reset-window-direction/`.
