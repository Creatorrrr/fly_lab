# E214 결과: 느린 기억 추가만으로 전신 제약을 만족하지 못함

E213의 두 시간척도 반응을 동일한 전신 출발·q·velocity 조건으로1회 적합한 결과는 PrimalInfeasible다. 부모가 원래근육/D에서 여섯local 물리행렬·목적·norm factor를 다시 계산하고 전신결합행렬 및dual을 독립검산했다. 후보계수·2×2·native·실제보행0으로 미채택이며, 느린입력기억추가만으로이조건부가족의문제를해결한다는가설은지지되지않았다. 자연스러운보행전체가불가능하다는뜻은아니다.

전신A34451×5488/19cones/A nnz10944239/P nnz617784,원본global solve1/local solve0. 새기저라A nnz는E211과다르며343MN/5488변수/4158startup/4159마지막cone/소유권은명시검증했다. strict=False가생략하는카디널리티검사를외부검사로복원했고옛고정nnz만요구하지않는다.

부모행렬 A/P/startup 차이0,여섯norm Gram 상대오차최대3.746e-14다. dualcone위반0, max|A^Tz|7.404e-11, b^Tz=-16.0079204441. 기존작은PAD를제거한원래물리경계에도b^Tz=-15.9990745241,box내(A^Tz)^Tbeta최소=-1.325e-9로수치불가능분리가남는다. 실패를단순solver중단이나부가startup제약때문이라고처리하지않는다.

E212에서동일상한의시점별자세는가능했다. 그러나E212는rate/activation의시간연속성까지풀었으므로현재실패가반응함수표현만의문제라고결론내릴수없다. 다음은실제MN시간상수와dual의시간/다리기여를저장자료에서검토해출력반응속도와함수표현을구분하는유한진단을설계한다. 동일적합재시도나기준완화는하지않는다.

부모설계→실제gpt-6-sol/high→부모FIRST차단결함없음→mock7/Ruff→원본1→부모검산순서다. 기존검산된E211 생성해를재사용하여새생성solver0이고과거실패는보존한다.

원본감독26.730293699999초,solver call벽시간13.451134700001초,원본누적5684.636283099706/6004초·잔여319.363716900294초. peak tree2546561024byte/individual2534281216byte,출력374554128byte,잔존프로세스0. 준비.266337초/cheap wrapper1.4000917초/부모원자료검산7.351306199998초는별도다. M2·최선E150·UI·F6미노출과F1–F8 active·미완료를유지한다.

원자료 `verification/banc-natural-gait-m3-20260919/e214-multiscale-response-fit/`, `parent-e214-output-review.json`, `parent-e214-final-review.json`.
