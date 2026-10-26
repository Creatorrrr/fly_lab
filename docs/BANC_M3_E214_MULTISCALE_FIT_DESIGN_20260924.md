# E214: 느린 기억 기저의 원래 전신 제약 적합

E213 실제 CUDA 원본과 부모 독립 RK 검산이 완료됐고 기존 출력 envelope가 보존됐다. 이 문서는 그 고정 기저를 사용하는 다음 한정 적합의 사전 설계다. 아직 새 적합이나 보행 성공의 증거는 없다.

부모 검산된 E213의 새9D descriptor로 만들어진 B/H/D를 동결해 동일 beta5488개의 전신 제약 문제를1회 푼다. E211과 달라지는 것은 학습 행렬을 생성하는 반응 기저다. E150/E197 기준·원자료·16개 중심/뉴런·free343/fixed48·최종proxy/F1–F8은 유지한다.

E210의 검산된 build_leg_problem으로 원래 다리별 normalized q 목적,1e-8 mean(beta²) regularizer,box[0,1],점별 startup.04*PAD,q/v nonworse norm을 만든다. 여섯 문제는 저장하지만 local solve는0이다. 기존 load_local_problem으로 읽어 E211 assemble_global(strict=False)의 첫3개cone×6 +전신 startup.018*PAD SOC를 결합한다. 추가 각다리 startup RMS cone은 제거한다. 첫3개의NN/q/v와 global startup,총19cone/5488변수/4158startup행/모든motor columns coverage를 별도 검증한다. 새 기저의 압축rank/nnz는 달라질 수 있으므로 E211의 옛 raw nnz나총행수를 요구하지 않는다. 이는 제약을 제거하지 않고 새 데이터의 희소 표현을 허용하는 차이다. 모든 norm factor와augmentedGram 오차를 보존하고 positive eigen pruning은 없다.

목적·사전PAD1-1e-6·Clarabel1회(max_threads1,max_iter150,tol1e-8,time_limit180)·정확한box·direct물리proxy는 E211과 동일하다. 전체denseqJacobian/globaldenseGram은 만들지 않는다. 새 local A/P를순차생성해디스크에보존하고필요없는densejq/jv를 해제한 후전신 결합한다. 준비가내부240초에서180초남기지못하면실제solve없이한정미해결로끝낸다. 결과 raw와설정·status·residual·dual은판정전에저장하고clip/원본재시도/seed탐색은없다.

Solved+strict box의완성계수는조건부rate를복원하고새/원래신경×monotone/canonical의2×2대조를저장한다. canonical direct와affine,고정LTM/tendon보존을검산하고원래proxy(전체q20%향상,각다리q/v·전체v비악화,startupRMS.018/max.04,tendon.009/.025)를그대로적용한다. 실패하는대조도보존한다. 모델native/물리호출은0이며이적합이성공해야별도native통합설계를고정한다. infeasibility는이조건부가족만제약한다.

구현은실제gpt-6-sol/high가새tools/fit_c_banc_m3_multiscale_response.py와tests/test_c_banc_multiscale_response_fit.py를소유,부모FIRST/같은Sol첫수정/부모SECOND·잔여수정순서다. 검산된기존build/save/load/assemble/solve/assess/readout/proxy helper를읽기재사용할수있으며모듈전역변수monkeypatch는금지다. 새로운실제생성solver는중복하지않고E211의수정후분석해/KKT검산을근거로재사용하되,새로운연결·ownership·rank/shape·실행횟수·raw저장을mock검사한다. 라이브러리속성은설치된소스에서확인한다.

자원은 원본 CPU1/GPU0/process1/solver1, 내부240/solver180soft/외부270/감독300초, RSS8GiB/output512MiB/free4GiB다. E213 원본까지 누적5657.905989399707/6004초·잔여346.094010600293초에서300초를 사전 예약한다. 단계 배정 증가0이며 준비/모의/부모 검산은 별도 계상한다. 현재 실제새solver0이며 최선E150/M2/F6미노출/UI/실패비용이력을 보존한다. F1–F8 active·미완료다.


부모는 실제 모듈 API를 확인했다. build_leg_problem의 cone_dimensions/cone_types/startup_matrix는 저장 후 load_local_problem의 dims/kinds/M으로 바뀐다. local solve는 하지 않으며 assemble_global(strict=False) 사용 뒤 변수폭·19cone·마지막4159차원·motor coverage를 명시 확인한다. 그 함수의 strict=False는 옛 자료의행수/nnz검사만 제거한다. 기존 solver helper의240초 내부/180초 soft 계약은 이번 자원과 같아 그대로 재사용할 수 있다. 부모 source gate와원본시도1회만허용하는out디렉터리를필수로유지한다.
