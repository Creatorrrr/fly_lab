# E236 저장 입력의 동결 반응식 전체 재생

H236: E235 실제 E/I 변화가 동결 E217/E218 반응식 안에서 E233의 startup free-effector/명령 차이를 설명한다. 경쟁설명은 RK 상태전달·연산구현 오류다. E235 전체BANC startup은 E224와정확히같으며, drive전부같고 E/I는약18ms부터다르다. 부모745검산에서 기억stage전체와선택effector stage가수식과일치했다. 이번은 상태를 stage마다정답으로리셋하지않고 처음부터끝까지 독립재생한다. 특정감각연결인과/물리적보행개선은검증하지않는다.

## 고정 비교

CPU NumPy float64로두입력arm을순서대로계산한다. actual은 E235의7920×343 E/I/drive, training은 E208 첫7920×391의동일free_columns이다. 두arm은같은E235 control0 free_effector64/h_fast/h_slow에서시작하며원래parameters.initial_rate/free와E217기억0의일치를검증한다. beta96×64, centers391×64×9/free343, sigma2, scale, tau/cap/inverse는E217/E218 그대로다. 실제모델생성·CUDA·물리·fit·solver·추가원본수집0.

각tick의4개외생입력을원래순서대로받으며 x=[log1p(E)/scale0,log1p(I)/scale1,asinh(drive)/scale2], descriptor=[x,h_fast,h_slow]다. 원래64중심의안정화softmax RBF, gate=-expm1(-8*(xE+abs(xD)+max(hfastE,0)+abs(hfastD))), target=cap*gate*sum(softmax*beta[group])를사용한다. dr=(target-r)/tau, dhfast=(x-hfast)/.016, dhslow=(x-hslow)*inverse_adaptation_tau를동일RK4/DT.00025로적분한다. target clipping/새평활/새계수/phase/timefeature0. 신경망 CSR이나신체를진행하지않는다. memory-RK가r와독립임을이용한연산분할/유한chunk vectorization은허용하되수학동일성은생성scalar예제로검증한다.

actual의모든stage를E235관측과비교하고100control경계상태도검증한다. training은E217 h_fast/h_slow 첫100행/free와E218 rate 첫100행/free를검증한다. 자유343 외fixed48은각각저장motor_rate32 또는E218 rate를사용하며서로같음을확인한다. actual readout은r를float32로변환해고정391MN순서에넣고원래pool96×391/10msactivation/42joint moment및6LTM tendon을재생한다. training은같은float32 readout과원래E218 float64 conditional readout을둘다보존한다. control0 command는저장held값,activation0는저장값이며r0로다시decode하지않는다. LTM순서[15,30,46,63,78,94]는고정lf/lm/lh/rf/rm/rh로검증한다.

수치재현유효조건은모든원시형상/순서/유한값과초기동일성, actual全stage와경계free64오차≤2e-6/기억≤2e-10, training境界같은상한, actual명령·tendon 및training float64 conditional명령·tendon≤1e-7/activation≤5e-5다. float32 readout byte exact 여부와최대오차를별도기록하고명령검증에숨기지않는다. 이수치상한은원래startup/보행기준을변경하지않는다. 원래E197 loaded_startup_target_command/loaded_startup_target_tendon에대한99행 RMS/max .018/.04 및.009/.025를actual/replay/training두readout 각각보고하며threshold padding없다.

둘다재현되면같은반응식의입력변화로출력차이가재현되었다고판정한다. actual−trainingfloat32의명령차이를E235실제−trainingfloat32와대조하고차이가남으면원인미해결로남긴다. 이조건부반응재생으로감각연결인과·자율주기·10초·실제보행개선을주장하지않는다. 실패시상태/수식/정밀도경로를수정하고새학습으로우회하지않는다. 통과시후속한후보는E85교사신체아래BANC의입력/기억과목표를함께수집하는matched-data설계이며E03b/E16–E29의과거실패를보존한다.

## 실행·자원

실제 gpt-6-sol/high가 tools/replay_c_banc_m3_saved_effector_inputs.py 와 tests/test_c_banc_saved_effector_inputs.py만구현한다. 부모FIRST→필요시같은Sol첫수정→부모SECOND→부모잔여수정. 부모만원본저장분석1회실행,재시도0. 생성검사는소형CPU만허용하고실제배열의수치재생은승인된1회로제한한다. 각arm control경계raw 및오차원시행렬/분석/진입·반환원장/manifest를검증전에보존한다. 부분실패도현재arm의완성tick을기록한다. 출력은 e236-saved-effector-input-replay 신규디렉터리다.

CPU1/GPU0/신경·신체생성0/物理0/fit0/solver0,2arm×1980RK ticks는저장反응수식계산이다. 내부75초(마감10초예약)/외부85/감독95초,RSS4GiB/출력256MiB/free2GiB。원본보수누적6118.154093699664/6154초와잔여35.845906300336초를유지하고실행前단계배정+60→6214초、잔여95.845906300336초에서95초예약한다.全局사용자시간상한은없다。E235원본28.1404771초와모든실패비용유지。F1–F8미완료/F6미노출/최선E150/M2/UI/원래freeze·증거보존。
