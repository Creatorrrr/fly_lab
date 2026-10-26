# E186 저장 반력과 근육별 명령의 정합 진단

2026-09-24 · M3 finite-goal · diagnosis · 사전 설계 v1. F1–F8, M2/UI, BANC 해부학적 provenance, F6 미노출을 유지한다. 이전 E185 단계는 실제 반력 관측으로 다음 개입 결정을 바꾼 progress다. E185 감독 실패는 그대로 보존하고 별도 승인된 저장 원자료만 사용한다.

## 질문과 결정

E185의 고정 전진 중단 8사건에서 앞으로 미는 힘이 사라지기보다 다른 다리의 반대 방향 힘과 겹쳤다. A: 지지 중인 다리의 전방 회수 움직임과 그것을 따라가는 목표 관절각 변화가 반대 반력과 함께 나타난다. B: 그 일치가 약하고 추적 오차·몸통 자세·접촉 전달의 설명이 남는다. 이 저장 관측은 인과적 A/B 판정을 하지 않는다. A와 일치하는 다리/근육/사건이 있으면 그 정확한 전달 경로를 후속 제한 개입 대상으로 검토한다. 없다면 동일 gain/전역 timescale 탐색으로 돌아가지 않고 실제 상태의 관절 Jacobian/접촉 전달 진단이 먼저다. 모든 다리와 사건의 반례도 보존한다. 원래 E177, E181/E183 실패와 E180의 임의 Hill 계수 배제를 유지한다.

양적 진단 산출물은 391 MN→96근육→42목표 관절각→실제 관절/발 움직임→native 반력의 같은 시간축이다. 목표각 기여는 물리 근육 힘·토크·관절 power가 아니다. 각속도 부호로 전방 발 회수 명령을 단정하지 않는다. tarsus5 원점 변위는 진짜 접촉점 속도가 아니다. Gaussian 사건 선정은 미래 표본을 쓰는 분석이며 제어 법칙이 아니다. 전방 회수 proxy와 joint alignment는 A와의 일치 정도만 나타낸다.

## 고정 입력과 시계

- 유일한 원본은 `verification/banc-natural-gait-m3-20260919/e185-ground-reaction/`의 theta=0 저장 실행이다. 새 baseline/candidate/신경/물리 실행은 0이다. E185 final review의 191개 output SHA와 실제 파일을 묶고, 실행 소스 snapshot 및 실행 후 감독 수정의 서로 다른 해시를 유지한다.
- control 601행(0 anchor), native 30000행, MN391/근육96/관절42/다리6. control 시각 i×.005, native k×.0001, 허용 시계오차2e-8. command row i는 `(i-1,i]`의 50 native 표본에 적용된 요청이다. anchor0에 native를 만들지 않는다.
- E184의 모든 11사건과 선택된 8인덱스 `[131,219,308,391,433,480,521,564]`를 유지한다. 원래 속도 정의로 사건이 같음을 재계산한다. event offsets -20..20, bands pre[-100,-50), lead[-50,0), post[0,50), recover[50,100]ms = 10/10/10/11행. 누락·겹침 제외를 다시 선택하지 않는다. startup은 기존 control1..99, 전체 관측은 control1..600, 개발 평가구간은100..599이다.
- 다리 순서 lf,lm,lh,rf,rm,rh. E184 joint-order.json의 전체42행을 사용한다. zero moment row30은 RM coxa pitch이며 관절 제외 이유가 아니다. muscle groups/roster/membership/moment/pool과 state manifest/patch 복제 참조를 검증한다. 복제 배열 이름이 달라도 bytes가 같은 정상 상태를 허용한다.

## 계산식과 결측 처리

1. E179의 순수 NumPy bank recurrence와 monotone reconstruction을 재사용한다. bank[i]=bank[i-1]+(-expm1(-.005/tau))*(rate[i]-bank[i-1]); variable77 근육의 각 MN 기여는 sum_filter(low*bank+(high-low)*max(bank-kappa,0)). fixed19는 실제 original adapter의 pool[m,n]*bank[n,tau=.01] 기여이며 원래 activation recurrence도 검증한다. 초기 bank/activation/original_activation 0, tau [.005,.01,.04], fixed19의 canonical slope와 pool/one-owner mapping을 확인한다. 각 MN391 기여를 저장하고 membership으로 더해 전체96 activation과 일치해야 한다. low/high 모두 비음수여도 hinge correction 자체는 음수일 수 있다.
2. 모든 기여 `Q[i,m,j]=.02*activation[i,m]*moment[j,m]`. rest+sum_m Q를 실제 targets와 i=1..600에서 절대1e-10 이내로 비교한다. bank/activation/MN합 residual도1e-10. E185 관측/원래 startup99 통과를 그대로 확인하고 startup actual_command를 trace1..99 및 actual_LTM을 tendon_inputs와 비교한다. startup original comparator를 새 기준으로 재정의하지 않는다.
3. measured joint interval velocity=(actual[i]-actual[i-1])/.005; target change rate=(target[i]-target[i-1])/.005; contribution change는 같은 backward difference. target anchor는 decoder 산출물이 아니므로 target/contribution rate는 i>=2만 정의하고0,1은 NaN. 실제 velocity는i>=1만 정의. 추적오차는 target[i]-actual[i], 별도로 interval start discrepancy=target[i]-actual[i-1]. 각다리7관절의 target/measured velocity cosine는 두 norm>1e-12일 때만 정의한다. 분모0은 결측이다.
4. UI 좌표의 body basis에서 forward를 얻고 y 성분을0으로 하여 평면 heading을 정규화한다(native z-up→UI y-up). tarsus5 world interval velocity=diff(position)/dt를 양끝 heading 정규화 평균에 투영한다. body-relative fore/aft coordinate=(foot-bodyposition) dot heading; 그 backward difference도 별도로 저장한다. 회전하는 heading이 relative-coordinate velocity에 기여한다는 한계를 표시한다. heading 소실/비유한/basis 비직교는 기각한다. native force heading은 E185의 native XY heading을 그대로 쓰고 force/normal clock 정합을 확인한다. force vectors/BW/다리합/전체fly residual/heading 직교 검증을 보존한다.
5. native 단계에서 support=normal_load_bw>=.02; 각다리 force `all_leg_heading_xy_force_bw`의 positive>0, negative<0, exact_zero==0를 조건으로 삼는다. support∧sign 마스크를 먼저 만든 뒤 control마다 count(0..50), signed force sum, normal-load sum을 저장한다. 지원되지 않은 표본을0값 관측으로 섞지 않는다. signsplit을 bin 평균 후 수행하지 않는다. supported slip은 valid/finite이어야 한다.
6. native sign 조건별 control 관측은 50표본 내 같은 interval command/motion proxy로 묶는다. 조건부 표본 수 가중평균=sum(count_i*x_i)/sum(count_i)이며 NaN은 관측에서 제외하고 그 유효 count를 별도 기록한다. 음의 반력 magnitude 가중평균도 sum((-force_native)_sum_i*x_i)/sum((-force_native)_sum_i)로 별도 표기한다. 분모0은 JSON null/NPZ NaN이며 count0을 남긴다. 힘0을 음/양으로 임의 분류하지 않는다. 신경/근육/관절 배열의 다리별 소유권을 적용하고 양/음/0 support 조건의 모든 다리·사건·band 결과를 보존한다.
7. 각조건의 forward-relative-foot fraction(v_rel>0), joint-alignment fraction(cos>0), 동시 fraction을 각 유효 count와 함께 제공한다. 이 fraction에 임의 합격 임계값을 만들지 않는다. whole, evaluation, startup99, pooled selected8 및 개별8/모든11 사건(겹친3은 탐색 기록)별 집계를 구분한다. 96근육과391MN의 평균/변화 및42관절 기여·변화는 원래 순서 그대로 저장한다. 크기 정렬은 설명용이고 새 다리/lag/cutoff를 선택하거나 유리한 표본만 제시하지 않는다.

## 유효성·출력·자원

새 도구/테스트만 실제 `gpt-6-sol/high`가 소유한다. 부모는 설계·metadata·동결·source gate·독립 리뷰·결과 기록을 소유한다. 부모FIRST→같은Sol첫수정→부모SECOND→부모잔여수정 순서이며 필요 없는 수정을 만들지 않는다. 원본 분석은 source gate 후1회, CPU1, process tree 벽시간45초(시작/해시/쓰기 포함), retry0. BLAS thread1, 새MuJoCo/BANC/신경/물리/CUDA0. 합성 테스트는 원본 계산으로 세지 않으며 부모 준비·리뷰 비용을 따로 기록한다.

입력 SHA, frozen design/metadata/source SHA, 부모 approved review를 CLI가 요구한다. 보호 파일과 출력 충돌을 기각하고 기존 출력은 덮어쓰지 않는다. 실행 시도·시간·오류 영수증을 남긴다. 부모 감독은 전체 process tree timeout/종료를 책임진다. 유효 출력은 analysis.json(수치와 한계/가설판정), aligned.npz(전체배열·조건부 집계), terminal.json이다. native_sign conditioned count 보존, MN/근육/목표각 재구성, 올바른 interval/anchor, 결측/zero norm, six-leg row30, startup99, 해시변조를 합성 반례로 검증한다. 부모는 별도 식으로 원본의 대표/전체 보존 항등식을 검토한다. 테스트 통과를 자연 보행으로 세지 않는다.

누적4339.688065899652/4994초에서 승계(남음654.3119341003476초), 이 단계 최대45초. 실제 감독 벽시간만큼 누적 증가하며 실패도 포함한다. 기존 E185 감독 rejection을 completed로 바꾸지 않는다. 실행 유효성, A/B와의 일치 관측, 후보 채택 없음, M3 전체 incomplete를 각각 기록한다. F1–F8 전체 목표는 active다.
