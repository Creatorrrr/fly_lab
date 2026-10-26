# E138 전체 BANC의 근육별 적응 시간상수와 상태 복원

2026-09-21 KST · M3/H138 · 부모 설계 · 실제 Sol/xhigh 구현·첫 수정 · 부모 리뷰·잔여 수정

## 목적과 근거

E137은 LF의 초기 selector를 만족하면서 원래 목적함수2.5730257% 개선을 보존한 조건부 제안을 얻었다. E137의8번 raw가 유일한 LF 시간상수 출처이며 최종 보행 후보라는 뜻은 아니다. 전체 BANC의 E123 경로는 세포별 gain을 지원하지만 적응 시간상수는 단일0.3초다. 조건부 E134 계산기를 전체 신경망으로 옮기는 데 필요한 실행·저장·복원 기능을 구현한다.

H138: 기존 전체 BANC의 CSR/RK4 경로에 근육별 시간상수를 연결해도, ell=0이면 기존 양수 gain 모델의 실제 전체 상태와 출력이 정확히 같고, 비균일 ell은 실제 MN 적응·출력에 영향을 주며 저장한 상태에서 미래를 정확히 재현한다. 경쟁 설명은 GPU capture 이전에 포인터가 고정되지 않거나, 근육/신경 순서 또는 연산 정밀도·복원 identity가 달라져 조건부 매개변수가 실제 모델에 적용되지 않는 것이다.

이 실험은 전체158706 신경망의 고정된 입력 진단이다. 신체 구성·물리 전진·새 감각 입력·모집 적합·F6 노출은0이며, 보행 품질이나 전체 F1–F8 성공으로 판정하지 않는다. 제품 선택·서버·활성 UI·기존 M2는 보존한다. 성공 뒤에도 다른5다리의 동일 후보 적합과 실제 폐루프/목표 이동 검증이 남는다.

## 구현 계약

새 `tools/banc_m3_motor_timescale.py`, `tools/verify_c_banc_m3_motor_timescale.py`, `tests/test_c_banc_motor_timescale.py`만 Sol이 구현한다. 기존 E123/E134·제품·평가기와 과거 결과는 수정하지 않는다. E123의 입력 재구성, 원시 관측, 신경/decoder 계측, StateStore를 재사용하고 전역 함수를 monkeypatch하지 않는다.

`MotorTimescaleNetwork`는 기존 `MotorAdaptationNetwork`를 확장한다. ordered motor_ids와 그 순서의 float64 motor_log_tau를 검증한다. 기본 적응 시간상수는 기존0.3초로 고정한다. ell은[-log(3),log(10)]이며 유한값이어야 한다. membrane tau와 모든 CSR/부호/weights/gain/core/threshold/cap/coefficient/drive/mask/기존 RK4 순서를 유지한다. 각 selected MN의 canonical T64=0.3*exp(ell64), T32=float32(T64), inverse32=float32(1.0/T64)로 동결한다. non-MN은 T64=0.3, T32=float32(0.3), inverse32=float32(1.0/0.3)이다. ell0이면 기존 시간상수와 같은 bytes다. 임의 clip이나 새 정규화는 허용하지 않는다.

CPU/reference 경로는 `(cell_gain*rate-adaptation)/cell_T32`로 계산한다. 기존 CPU의 scalar0.3 나눗셈과 float32 inverse 곱셈은 생성128개 값 중35개에서 달랐으므로 기존 CPU 나눗셈을 유지한다. packed CUDA는 기존 `mul_exact(sub_exact(mul_exact(cell_gain[i],rate[i]),adaptation[i]),cell_inverse_tau[i])`를 stage/finish 모두 사용한다. 기존 tanh/FTZ/CSR reduction과 연산 순서·fmad=false를 유지한다. 모든 시간상수 tensor와 gain tensor는 warmup/capture 전에 생성되고 그 뒤 포인터를 교체하지 않는다. 소스 문자열을 변환한다면 기대한 치환 수를 검사하여 소스 변화 시 명시적으로 거부한다.

새 identity는 source motor identity, schema, ordered IDs, 요청 ell64와 실제 T64/T32/inverse32 bytes를 포함한다. snapshot은 이 전체 정보를 저장하며 restore는 dtype/shape/order/bytes/hash/schema와 부모의 rate/adaptation/drive/mask/tick 검증을 모두 마친 뒤 상태를 바꾼다. 손상된 timescale metadata를 기존 신경 상태에 일부 적용하면 안 된다. outgoing cut과 신경 자체 적응의 의미를 유지한다.

`MotorTimescaleTrial`은 이미 구성된 `MotorAdaptationTrial`을 확장한다. 전역96근육의 ell96을 원래 pool/membership/named columns/root roster로391MN에 대응한다. 몸통·adapter·calibration을 보존하고 새 신경 identity로 상태를 옮긴다. 전체 snapshot에 source motor identity, source adapter identity,96근육 요청값과391MN 대응·실제값을 저장한다. from_checkpoint는 timescale layer를 벗겨 E123→Intrinsic→Recruitment→Efficacy→Core를 복원한 뒤 동일한 timescale layer를 적용한다. 기존 E123 생성자는 old network의 type을 CoreAdaptationNetwork로 제한하므로 이미 만든 MotorAdaptationNetwork를 E123 생성자에 재투입하지 않는다. 실패 시 소유한 body를 닫고 최초 예외를 보존한다. E138은 wrapper의 실제 몸통을 만들지 않으며 작은 생성 stub으로 소유권/복원 검증을 수행한다. 실제 몸통 미래는 후속 후보 검증에서 확인한다.

## 원래 입력과 고정 변환

E123의 전체 graph bundle, E121 calibration/anatomy/1초 StateStore, core identity `bfc89ead7088892f8c7fb479d25010c35a9c590825e777b8e8c9b9a3d01674d7`, tick4000, dt0.00025, 저장된 drive/mask/96activation을 그대로 사용한다. graph.n=158706, 원래391MN 및96근육 순서를 검사한다. E123의 원래 입력/소스/보존 해시 계약도 유지한다.

부모가 생성하는 `e138-timescale-inputs.npz`는 E1378번 `muscle_indices`가 E128 `leg_muscle_mask[0]`의 전역 인덱스와 같은지, E128/E121 membership/pool/391MN/96근육 이름이 동일한지 검증한 뒤 ell96을0으로 두고 LF17칸만 원래 ell17로 채운다. motor_log_tau=ell96[membership]이다. 동결 파일에는 요청과 T64/T32/inverse32 배열을 함께 둔다. 신경 계산·적합은 수행하지 않는다.

실제 모든 arm의391MN gain은1로 고정한다. 이는 시간상수 경로의 실제 효과를 확인하기 위한 양수 진단값이며 E137의 gain/edge 후보를 전체에 적용하거나 보행용으로 선택하는 것이 아니다. E121 CSR를 유지하고 E137 delta를 적용하지 않는다. E137의 conditional float64 결과와 full-BANC float32 폐루프의 동등성을 주장하지 않는다.

## 생성 시험과 한 번의 실제 CUDA 실행

Sol은 작고 생성된 CPU fixture로만 개발한다. 부모 소스 리뷰 후 생성 CPU/CUDA 시험을 실행한다. 작은 혼합 부호6셀·양수 gain·비영 r/z·outgoing cut·drive 변화와 두 개 이상 비균일 시간상수를 포함한다. ell0은 동일 device의 기존 MotorAdaptationNetwork와 전체 r/z/drive/mask/tick가 bit exact여야 한다. 비균일 CPU와 packed CUDA는 양 상태 rtol/atol2e-5, 독립 DOP853도 rtol/atol2e-5다. native time conversion의 소수 차이는 이 별도 허용차 안에서 판단하며 단일 기준을 사후 바꾸지 않는다. malformed ID/log/dtype/hash 복원 거부 전후 상태와 새 identity·capture 포인터·disk future를 검사한다. 생성 신경은 모델마다 최대8셀, 원본 graph/fixture에서 실행하지 않으며 CPU/CUDA/참조 적분·setup·실제 tick 비용을 따로 계수한다. 생성 CUDA 시험은 부모만 실행하며 동시 GPU 사용은1명이다.

최종 source review 후 원본 프로세스1개에서 다음을 한 번씩 실행한다. 출력 `e138-native-timescale`이 이미 있으면 거부하고 재실행하지 않는다.

1. **original_motor_gain_one:** E123 MotorAdaptationNetwork의391MN gain1, 기존 T0.3. E1211초 상태와 저장 drive에서10×20 step=.05초.
2. **zero_log_tau:** 새 모델,391MN gain1/ell0. 같은 .05초. 매5ms 전체158706 rate/adaptation/drive/mask/tick,391MN,96activation,42request,6tendon이 첫 arm과 bit exact여야 한다.
3. **heterogeneous_log_tau:** 새 모델,동일 gain1, 동결 LF ell→391MN 적용. 같은 .05초. 전체 상태가 유한하고 LF 중 적어도 하나의 MN rate/adaptation과 전체 decoder request에 실제 차이가 있어야 한다. 나머지 neuron의 상태는 재귀 신경 전파로 달라질 수 있으며 비선택 physiology/시간상수/CSR는 동일해야 한다.
4. 세 번째 arm 종료에서 full neural/96activation/calibration/held drive를 StateStore로 저장한다. 원본과 같은 설정의 disk clone이 각각20step=.005초 더 진행한 전체 신경·decoder 미래를 bit exact로 비교한다. clone은 저장된 metadata와 모든 실제 timescale arrays를 검증하여 복원한다.

원본 총 neural advance=.16초/640tick, decoder32회, full model 생성4회, 각 CUDA warmup/capture60step 총240setup step이다. setup을 scientific640tick에 숨겨 합치지 않는다. 모든 모델은 restore 뒤 tick4000 또는 저장한4200에서 시작함을 확인한다. body/fit/LP/E134조건부 재실행은0이다.

매 모델/advance 전에 요청을 ledger에 저장하고 반환과 실제 tick delta를 구분한다. advance 이후 관측/저장 실패에서도 이미 완료한 신경 계산을 남긴다. 모든 완전/부분 raw와 configuration/state/provenance를 수치 판정 전에 저장한다. 원래 예외·이후 저장/정리 오류·미확정 setup/advance를 보존한다. 요청·constructor/capture·advance·decoder·StateStore·final manifest 비용을 구분한다. 정상 종료만으로 성공 판정하지 않으며 늦은 저장 오류가 성공을 남기면 안 된다.

## 자원·판정·다음 단계

기존 E137까지 실제 보수적1700.9802867999533초, 새 기준1701초를 보존한다. 원본 내부30초/외부35초/supervisor40초로 배정하여1701+40=1741<=1742를 유지한다. 모델/advance/clone 시작에는6초 저장·정리 reserve, 마지막 manifest에도 시간을 남긴다. CPU thread1/GPU1, raw1GiB, process1/retry0다. 부모 원본 감독·최종 manifest·종료 확인까지 포함한다. 생성 시험/부모 read-only NumPy 검산 비용은 별도이며 기록하지 않은 비용을0으로 보지 않는다. source review 이전에 원본 실행하지 않는다.

H138 supported는 올바른 원래 재구성·실제 neutral whole-state/decoder exact·비균일 actual effect·disk future exact·생성 수치/상태 시험·완전한 기록·자원·원본 보존을 모두 요구한다. 적절히 완료된 유효 비교에서 기능이 틀리면 refuted, 기술 오류/누락/미확정은 invalid/not_tested다. 지원되면 새 연구용 native timescale 기능만 허용한다. `candidate_adopted=false`, `F6_exposed=false`, `whole_goal_complete=false`를 유지한다.

부모가 설계·독립 정적 검산과 실제 결과를 리뷰한다. 새 CUDA 포인터·연산 정밀도·신경 state identity/복원이 연결되므로 실제 Sol/xhigh로 구현한다. 부모 첫 리뷰에서 수정이 필요하면 같은 Sol/xhigh가 첫 수정하고, 부모 두 번째 리뷰 이후 잔여 수정은 부모가 한다. 형식적으로 수정 사항을 만들지 않는다. 다음에는 남은5다리의 저장된 적합 자료와 native 적용 가능성을 함께 검토해 동일 후보와 새 자원 계획을 사전에 정하고, 전체 신경·신체 F1–F8로 이어간다.
