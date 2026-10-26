# E216: 나머지 다리와 전역 동적 용량 proxy

E215의 LM은 원래 막/근육 시간상수와 cap/gate를 유지하면서 명령 RMS88.48% 감소, startup RMS.00615/max.03138, velocity.17666을 달성했다. 이는 임의 시간별 forcing의 조건부 witness이며 고정된 신경 반응 함수나 실제 보행은 아니다. LM 결과를 다시 풀지 않고, 나머지 LF/LH/RF/RM/RH에 동일한 동적 필요조건 문제를 한 번씩 적용해 기존 여섯 다리 전역 proxy를 합산한다. E214 고정 기저 적합의 재시도는 아니다.

## 고정 수식과 입력

E215 동결 입력에 E215 source/test/source-review/generated-independent/final-review, LM pre-solve/physical-inputs/solver-raw/info/diagnostic/result/manifest를 추가로 고정한다. E215 LM solved 상태·raw x와 물리 재구성을 읽어 재사용하며 LM 새 solver0이다. 원본 E201391MN/free343/fixed48/96근육과 E213 기저·q0, E197목표 및 E150 trace는 그대로다.

E215의 그룹별 G/F/S/L과 x=timestep-major normalized 추가 activation, T/alpha 소거식, 0<=x<=1, forcing[-2e-12,S+2e-12]를 그대로 쓴다. 나머지 다리는 모두 tau=.019999999552965164s이며 서로 다른 tau를 평균내지 않는다. 다리별 자유 MN/근육 수는 LF61/16, LH56/15, RF62/16, RM55/14, RH55/15다. LM54/14는 E215 고정 해다. pool의 단일 근육 소유와 다리 간 moment0, 초기 free rate/B0=0, 근육 내 같은 tau를 각 다리에서 확인한다.

각 다리는 frame1..1923, startup1..99, main100..1923, 기존 main1823 velocity쌍을 사용한다. q=q0+.02*moment*G*x, 동일 main normalized MSE 목적/ridge0/PAD1과 startup pointwise .04, 해당 다리 main q/v RMS 비악화 제약이다. 전역 startup RMS 또는 다리별 .018을 추가 학습 제약으로 넣지 않는다. `tools/analyze_c_banc_m3_dynamic_capacity.py`의 검산된 sparse builder/save_pre_solve/solve_once/forward_reconstruction/infeasible_diagnostic을 그대로 재사용하고 수정하지 않는다. 다리별 보고는 실제 해당 다리 baseline을 쓴다(E215 LM 전용 index1 진단함수를 다른 다리에 그대로 적용하지 않는다).

5다리를 LF,LH,RF,RM,RH 순서로 한 번씩 수행하고 각 원자료를 leg-* 폴더에 즉시 보존한다. 프레임별 재최적화/재시도/새 hyperparameter/목표 이동은 없다. Solved 및 원래 tolerance1e-8/forward2e-10에 대한 직접 수치 적합성을 확인한다. 한 다리가 실패하거나 시간 부족이면 이미 계산한 raw와 비용을 보존하며 전체 해를0이나 원래 궤적으로 보충하지 않는다. 진단 실패는 bounded unresolved/conditional infeasible로 구분하고 그 뒤 새 원본 호출은 중단한다.

## 합산 판정

6다리 모두 유효한 경우에만 LM의 기존 q_linear와 새5다리 raw에서 순방향 재구성한 q를 실제 leg순서 LF/LM/LH/RF/RM/RH로 합친다. 저장 command는 **row0의 기존 held command와 rows1..1923만**, frame_indices0..1923을 명시한다. 나머지 시간 구간을 만들지 않는다. tendon은 고정된 E213 tendon_intercept[:1924] 그대로다. 기존 direct_metrics/feasible 함수가 실제 필요로 하는 동일 표본에서 main전역명령20%감소, 전역·각 다리q/v비악화, startup전역qRMS.018/max.04 및 tendonRMS.009/max.025를 변경 없이 계산한다. 다리별 성공이 전역 성공을 보장한다고 가정하지 않는다. 평가는 strict 원래 proxy이고 개별 solver 진단 tolerance는 별도 보고한다.

전역 proxy PASS이면 고정 기록 입력·원래 필터에서 6다리 조건부 궤적의 용량이 있다는 근거다. 고정 RBF 반응식의 표현력/학습을 다음 대상으로 삼을 수 있으나, 원래 몸체로 실행되거나 감각 feedback에 반응하는 신경 후보는 아직 아니다. PASS하지 않으면 어느 합산 조건이 남았는지 기록하며 자동 재적합하지 않는다. F1–F8 판정을 바꾸거나 현재 UI에 자유 forcing을 연결하지 않는다.

## 검토·실행·자원

부모 설계→기존 실제 gpt-6-sol/high 구현→부모 FIRST→필요시 동일Sol 첫수정→부모 SECOND/잔여수정. 소유 파일은 tools/analyze_c_banc_m3_six_leg_capacity.py와 tests/test_c_banc_six_leg_capacity.py다. 근육 소유권의 일반화, LM raw 재사용, 다리별 실패/회계, 전역 판정 분리 때문에 high를 사용한다. 기존 E215와 모든 이전 증거는 수정하지 않는다.

생성 순수/모의 검사는 leg별 선택·baseline·LM 재호출 금지·부분 결과 합산 금지·회계와 저장을 확인한다. 실제 생성 solver는 **0회 추가**, 부모가 재검산한 E215의 동일 sparse builder와 동일 solve_once 실제 생성1 증거를 재사용한다. 핵심 수식/solver source는 SHA로 고정하며 새 모의 조합 검사를 수행한다.

원본 프로세스1/새 solver최대5/각 다리1/LM0/retry0, CPU1/GPU0/새신경0/물리0. 내부150s/각 solver90s soft/외부180s/감독200s, RSS8GiB/출력256MiB/free4GiB. **모든 solve_once의 elapsed에는 동일 프로세스 시작 시계**를 전달한다. 각 constructor 전후 내부150초에서 온전한90초가 남는지 확인하므로 개별90초를5회 합산한 새450초 예산이 아니다. 조건을 충족하지 못하면 나머지 호출을 하지 않는다. 각 다리 ledger와 aggregate 실제 호출 수를 보존하고 이미 완료한 다리를 재풀이하지 않는다.

누적5688.697704299706/배정6004초·잔여315.30229570029405초 중200초 예약, 추가 배정0. 준비/생성모의/부모검산/agent비용은 별도이며 미측정을0으로 세지 않는다. F6미노출/최선E150/M2/UI 보존, F1–F8 active·미완료. 전역 결과 뒤 신경 반응 표현 변경은 별도 사전 설계한다.
