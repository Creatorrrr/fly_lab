# E155 RM 접촉점 속도의 관절별 기여

E154 원래1–2초의 실제 force에서는 중단 중 RM의 평균 제동이 −.09464BW다. 부모의 추가 saved-data 분해(.0283143초/모델·physics0, parent-e155-rm-contact-segment-design-observation.json)는 RM 접촉이 tarsus2/3에만 있음을 확인했다. 따라서 tarsus5 원점 대신 **실제 접촉점**의 세계 상대속도를 root, RM의7개 구동 관절, 나머지 DOF로 분해하여 제한된 개입의 대상을 정한다. 기여 분해는 운동학적 등식이며 근육 또는 신경의 인과 기여율이 아니다. 원래 실패·F1–F8은 유지한다.

## 고정된 입력과 계산

E150 checkpoint1s/trace, E154 contacts.npz/native-forces.npz/context.json, E152 E150 features.npz만 사용한다. E154의 native index49,99,...,9999(원래tick201..400)와 E150 동일control의 qpos/qvel/ctrl을 대응한다. n=200개 boundary이며 시간은1.005..2초, native 속도 통계와 전진 mask는 E152 동일endpoint 값 그대로다. 원래200개의 모든 명령·pose를 먼저 검증하고 모델을 생성한다. 추가 선택·phase shift·lag search·gain fitting은 없다.

기존 new_body(saved)로 같은 모델1개를 구성한다. 생성자 settling2000 step은 실제 계수하고 scientific integration0과 분리한다. 부모가 검토한 E154 construct_body_safely를 재사용할 수 있다. 복원된 actor 데이터는 유지하고 별도 shadow mjData1개에만 각 저장 qpos/qvel/ctrl/time을 설정하여 mj_forward한다. shadow의 Thorax rotation은 원래 E154 endpoint rotation과 atol1e-12에서 일치해야 한다. 이 forward는 관측 기하 계산일 뿐 새 주행이나 힘 계산 결과를 사용하지 않는다.

각 endpoint의 E154 raw contact 중 RM(leg4), wrench[0]>0인 contact를 원래 순서로 선택한다. geom 순서로 foot/support를 확인하고 segment20..24 및 raw sign과 맞는지 검증한다. 그 원래 world contact position에서 `mj_jac`로 Jfoot, Jsupport(3×nv)를 얻는다. `J=Jfoot-Jsupport`, normal n=원래frame[0], 접선 Jacobian `T=(I-n*n.T)J`, `v=T*qvel`, 수평전방 기여는 각 DOF별 `a_k=dot(h,T[:,k])*qvel[k]`다. h는 E154 native heading. 전체AP=sum(a_k), root=자유관절6DOF의합, 각RM구동관절=해당DOF기여, residual=나머지전부합이다. root와 residual의 생물학적 의미를 추측하지 않고 실제 joint/dof mapping을 저장한다.

42개 leg actuator의 joint transmission과 hinge type/고유 dof address를 모델에서 검증한다. RM은 원래42관절의28..34번째다. free joint는 정확히1개/6DOF이고 RM7과 겹치지 않아야 한다. 각RM joint의 name/joint_id/dof_address와 모든 DOF의 joint name을 저장한다. 원래 contact geom/body mapping, normal/frame/position/wrench, qvel, Jfoot, Jsupport, projected columns, velocity/분해 항을 raw에 저장하여 모델 없이 부모가 독립 검산할 수 있게 한다. NaN·shape·mapping 오류는 즉시 실패하며 assertion 직전까지 수집한 원시 행은 보존한다.

각endpoint RM contact의 max(wrench[0],0)/weight0를 합친 normal load와 load 가중 `norm(v)`는 E154 원래 RM normal load/slip에 atol1e-11, rtol1e-12에서 일치해야 한다. support>=.02 BW인 표본만 slip 비교하며 나머지 slip은 NaN/valid=false를 유지한다. 이 일치가 실패하면 관절별 기여로 설명하지 않는다. frame tangent projection/합 등 scalar 수식은1e-11에서 검산한다. 원래physics를 다시 실행하지 않는다.

## 산출과 해석

contact-kinematics.npz는 control offsets201, 각contact의 source native index/raw index/geom/segment, 원래pos/frame/wrench, qvel(nv), Jfoot/Jsite support(3×nv), DOF별 AP 기여(nv), tangent velocity3을 보관한다. control-kinematics.npz는200개의 tick/time, 원래RM force/load/slip/valid, 재구성load/slip, 하중 가중AP/root/RM7/residual의 분자·분모 및 검증표시를 보관한다. layout.json은 model/body identity와 모든joint/DOF/actuator/geom 이름·매핑, resourceledger, result/summary/timing/error/manifest를 저장한다. summary보다 raw를 먼저 저장하고 실패 raw를 지우지 않는다.

집계는 all, forward(speed>.5), interrupted, supported_RM_braking(load>=.02 & force<0), supported_RM_propulsion(load>=.02 & force>0)의 고정5집합이다. 집합별 표본수/지지수/normal-load-weighted 각AP기여의 합을 분자합/분모합으로 계산한다. 분모0은null. per-DOF 각기여는 부호를 유지한다. 비율을 causal percentage라 하지 않는다. 특정관절의양의AP가커도개입효과를확정하지않고, 해당근육·관절요청을짧은맞춤대조로시험할단서로만쓴다. 새로운후보채택·F4/F5판정없음이다.

## 개발과 자원

부모설계→실제 Sol/high FIRST 구현(`tools/analyze_c_banc_m3_rm_contact_jacobian.py`, `tests/test_c_banc_rm_contact_jacobian.py`)→부모FIRST review→같은Sol FIRST correction→부모SECOND review/잔여직접수정. 고정행렬분해와기존신체기하API통합으로 high를선택하며새신경/CUDA구조는없다. Sol은생성자료test만실행하고원본 StateStore/모델/physics실행은하지않는다. CLI --freeze --freeze-sha256 --source-review --source-review-sha256 --output, exactfreeze/sourcegate/원본1/owned2SHA및전후보존해시를확인한다.

원본프로세스1/자동재시도0, CPU1/GPU0, 몸체모델1/shadow1/생성settling2000/kinematicforward200/scientificphysics0/neural0. 내부35/외부45/감독55초, raw512MiB/free12GiB. 이전보수누적2511.2355973초와2529초배정을보존하며 ceil2512+55=2567초를새기하진단에사전배정한다. 마지막저장까지한도포함, provisional accepted=false→모두완료때만true. failed/incomplete는그대로기록한다. F6/UI/M2변경0, E150/E153의기존실패유지, candidate_adopted/whole_goal_complete=false.

생성검증은 회전된contact frame, 움직이는 support Jacobian, 자유DOF/7RM/residual분리, weighted slip과scalar방향 차이, missing-supported-slip, offset/clock/mapping틀림, emptygroup, partial failure보존을포함한다. 부모는단순생성MuJoCo chain의Jacobian을기하finite difference로검산하고, 원본saved J를독립scalar로재집계한다. 독립saved검산은모델/적분0, CPU1/내부30/외부40초다.
