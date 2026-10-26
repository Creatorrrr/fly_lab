# E213 결과: 두 시간척도의 조건부 CUDA 반응 검산 완료

기존16ms 입력 기억에 원래 actual inverse_adaptation_tau를 이용한 느린 기억3채널을 추가한 계산 경로를 구현했다. 실제CUDA history/basis2launch와 부모의 별도 RK4 다항식·lfilter 검산이 일치한다. 아직 학습계수나 전체BANC·실제 보행 후보는 없다.

기존 h_fast/H/H_factor/초기q/fixedLTM·tendon은 bytes exact다. 모든391MN의 출력 envelope 합 차이는 B1.22125e-15,D1.70530e-13으로 사전허용치 안이다. 새9D prototype 선택행·중심·폭을 부모가 전체 재계산했고 일치했다. 사전 고정한6MN의 전체160000 stage에 대해 별도 계산한 h_slow 최대차이는7.472e-14,B9.426e-14,probe rate7.433e-12이다. 전체391MN canonical D/affine/direct q도 별도검산했고 maxD오차5.684e-14,고정tendon exact다.

기억만으로 성능이 향상됐다는 판정은 하지 않는다. 기존 readout-only 대조의 RMS .071625와 startup max.040311391은 이전과 동일한 미달이며, 다음 E214에서 고정된 새 기저를 같은 전신 제약에 적합해야 한다. native feedback에 대한 검증도 남아 있다.

부모 설계→실제 gpt-6-sol/high 구현→부모 FIRST 차단결함없음→cheap/mock8 및Ruff→생성CUDA1테스트(8conditional ticks/2launch)→원본1회→부모독립검산 순서였다. 수정이 필요하지 않아 인위적 첫수정을 만들지 않았다. CUDA원본은80000conditional ticks,fit0/fullBANC0/physics0/retry0/F6미노출이다.

원본 감독15.117046899999초,원본 누적5657.905989399707/6004초,잔여346.094010600293초. 출력427518595byte,peak individual Wset2768670720byte,ownedCuPy pool964009472byte,잔존프로세스0. 준비 .2227026초/cheap wrapper1.3214607초/생성wrapper4.4189197초/부모검산4.422616099997초는별도다. 최선E150/M2/UI/실패이력과F1–F8 active·미완료를 보존한다.

원자료 `verification/banc-natural-gait-m3-20260919/e213-multiscale-neural-basis/`, `parent-e213-output-review.json`, `parent-e213-final-review.json`.
