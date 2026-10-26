# E218:64상태·근육별 공유 반응계수의 전신 적합

E217은 원본 CUDA1회 및 부모 독립 검산을 통과했다. h/H·고정6LTM은 exact, group envelope차1.7053e-13, q직접/affine차3.5527e-15다. E218은 이 기저로 고정된 인과 반응계수를 찾는 실험이다. E216의 임의 시간별 forcing 성공을 자율보행으로 오인하지 않는다. 이64기저 공유계수 가족은 기존16기저 독립 MN계수 가족의 상위집합이 아니다.

## 동결된 수학 문제

free90원래근육×64변수=5760, 각 beta∈[0,1]. LF/LM/LH/RF/RM/RH의 free근육은16/14/15/16/14/15, 변수1024/896/960/1024/896/960이다. 원래391MN의 owner/pool/moment/ltm 및6다리 관절 블록을 검사하고 cross-leg moment가0임을 확인한다. 근육에는 원래 소속 MN의 자기 입력·자기 기억만 쓰는 반응계수를 공유한다. 새로운 runtime feature/edge/phase/clock/외부CPG를 도입하지 않는다.

`J[t,k,g,j]=.02*moment[k,g]*D_group[t,g,j]`, `q=q0+J*beta`. 고정6LTM/48MN의 기저0, q0/tendon/초기 activation/held row0는 E217과 같다. E197의 원래100..1923 command1824행,101..1923 velocity1823쌍,1..99 startup99행을 사용한다. 다리별 목적은 command residual의 기준RMS·표본수 정규화 제곱합과 `1e-8*mean(beta_leg²)`이며6개를 합한다.

각 다리 command/velocity norm은 원래 다리별 기준이하, startup 모든값 absolute≤.04*PAD, 전신 startup RMS≤.018*PAD이고 PAD=1-1e-6이다. beta상자[0,1]는 padding/clip없이 실제좌표다. 여섯 startup RMS를 별도로 제한하지 않는다. 동일한 augmented-Gram SOC압축은 양의 eigenvalue를 모두 보존하고 상대 Frobenius≤1e-10/최소 eigen≥-1e-10*max(1,maxeig)를 검사한다. 원래 q/v의 물리단위에서 결과를 다시 평가한다. 직접 행렬과 압축규모는 eigenvalue수에 무관하며 다리별width+1행 factor를 보존한다. 전체 cones19개(다리마다 NN/qSOC/vSOC, 마지막 globalstartupSOC)다.

전역 command RMS≤.8*.056686385349786386, 전역/각다리velocity비악화·각다리command비악화, startup q RMS/max .018/.04 및 tendon .009/.025의 **기존 엄격한 proxy를 결과에서 별도 판정**한다. 이 proxy는 자연보행의 필요조건이라고 주장하지 않는다. 성공조건은 status Solved, 유한 raw, beta가 실제[0,1], equality/cone residual≤1e-6, objective 직접오차≤1e-6, direct 원래pool/10ms EMA와affine q atol2e-10/rtol2e-12, 원래strictproxy통과다. raw를자르거나실패한beta를후보로만들지않는다.

## 호출·저장·검토

기존 E210 quadratic_objective/compressed_soc/box_and_startup, E211 전역startup조립/assessment, E215 solve_once(90초soft/150초내잔여확보)는 shape가 호환되는 부분만 재사용한다. group축을16-MN축으로 가장하지 않는 명시적group builder/저장형식을 새로 만든다. mutable global patch나 기존소스 수정은 없다. standalonegroupmatrix,원래group/column순서,각localP/A/b/offset/압축factor/metadata와globalP/A/b/cone/startup 및source identity를 첫원본solve전에 저장한다. local solver0/global solver1/retry0, fullBANC/CUDA/물리0이다.

해가나오면 원본B/H와beta_owner로391rate를 재구성하고 fixed48은원래savedrate로 대입해 canonicalreadout을직접 계산한다. D_group q와대조하고 기존strictproxy를검사한다. 이 단계는 여전히 원래기록입력에조건부이며 실제피드백정확성·신체보행은다음단계다. candidate계수는모든위검사통과때만저장하며model family/geometry/scale/slowinverse provenance를동봉한다. UI나기존M2에는채택하지않는다.

PrimalInfeasible 등해없는결과도 raw x/s/z와info/ledger를저장한다. dualcone가실제로0위반인경우에만 `sum min(A.T*z,0) - b*z`의상자분리진단을확정한다. 작은음수cone를허용하여잘못된증명을만들지않는다. PAD를제거한 b도따로검산하며scope는이조건부가족뿐이다. 한도/수치실패는불가능성으로확대하지않는다. solver후기록실패나constructor 실패도ledger와오류를보존한다.

부모설계→기존실제gpt-6-sol/high구현→부모FIRST→필요시같은Sol첫수정→부모SECOND/잔여수정. 소유파일 tools/fit_c_banc_m3_grouped_response.py 및 tests/test_c_banc_grouped_response_fit.py다. 새실제생성solver0: E215의동일solve_once실제analytic생성1증거를재사용하고 새group행렬/전역연결/strictbeta/proxy/실패회계는순수·mock검사,부모는original저장행렬을다른구성으로검산한다. 새생성solver를자동추가하지않는다.

## 자원·종료

CPU1/GPU0/원본process1/solver1/새생성solver0. Clarabel maxiter150, maxthreads1,time_limit90soft,tol_gap_abs/rel/tol_feas1e-8. 내부150/외부180/감독200초,RSS8GiB,출력768MiB,free disk4GiB. hash·행렬조립도내부시계에포함하고constructor전후90초solver여유가없으면호출하지않는다. softlimit초과도별도실패회계다. 패키지/입력/재사용source/검토SHA는별도freeze와실행직전확인한다.

누적5732.071894399702/6004초·잔여271.9281056002983초에서200초를예약하며추가배정0이다. 준비/순수/부모검산/agent비용은별도로기록,미측정을0으로간주하지않는다. F1–F8active·미완료,최선E150·M2·과거실패·활성UI보존,F6미노출이다. 본실험이proxy를통과해도2×2·실제CUDA신경반응/native저장·모터차단/10초신체F3–F5/F6/F7는완료가아니고각별도실행설계가필요하다.
