# E187 결과: MN144018 입력의 대부분과 사건 간 차이는 다른 문제

2026-09-24. 저장 자료 분석은 유효하다. E150의 첫601 control/30000 native 표본과 E185 반력 실행은 동일했고, 다섯 시점×다섯 뉴런의 CUDA recurrent current 25개가 float32 bit 단위로 재현됐다. 그러나 현재 자료는 relay 연결의 일괄 감쇠를 정당화하지 않는다. **보행 개선 후보는 채택하지 않았으며 자연 보행/F1–F8은 여전히 미완료다.**

## 무엇을 확인했는가

[사전 설계](BANC_M3_E187_MN_CURRENT_DESIGN_20260924.md)에 따라 E150의446-neuron 연속 기록을 사용했다. 초기 전체 상태는 배열 파일명과 무관하게 의미상 동일했으며, trace·391MN/96근육 상태·native contact의3초 prefix가 exact equality를 통과했다. 전체 CSR도 graph→navigation→calibration→individual fit으로 재구성해 실제 저장값과 일치했다. 새 BANC/MuJoCo/신경 적분/물리/GPU 실행은0이다.

MN144018(root720575941644474312, RM sternal adductor group67)은195개의 incoming edge를 가진다. 선행뉴런10개가 기존 연속 기록에 있고185개는 없다. 기록된 입력은 1/1.1/3/10초 경계에서 전체 절대 전류의96.98–98.73%를 설명했다. 이는 네 경계의 결과이며 전 구간의 coverage라고 확대하지 않는다. 0초는 입력0으로 분모가 없어 비율을 null로 남겼다.

가장 큰 기록된 흥분 입력은 **IN03A010(graph54014)→MN144018**이다. 현재 edge10511590은23 anatomical contacts, rate baseline+.69, individual-fit 현재값+.2087680697441101이다. E95 당시의+.00069를 재사용하지 않았다. 선택8사건 post에서 이 edge는 알려진 흥분 입력의99.742%를 차지한다. 그러나 지지·음반력 조건의 기여15.5081과 지지·양반력 조건의15.4484는 거의 같고, 사건별 우세 방향도 뒤집힌다. 사건131/308은 음<양,391/480은 음>양,521은 동률이며433에는 RM 음반력 표본이 없다. 모든11사건과 결측을 원자료에 보존했다.

## 남은 차이의 위치

E186과 동일한 selected8 post band, RM native 조건 수 가중평균이다. 전류 단위는 이 rate model의 내부 척도다. graph mV weight와 혼동하지 않는다.

| 항 | 음반력 조건 | 양반력 조건 | 음−양 | 전체 전류 차이의 분해 |
|---|---:|---:|---:|---:|
| 저장 recurrent current |14.745046|14.067719|0.677326|100%|
| 기록된 양의 입력 합 |15.548219|15.488357|0.059862|8.84%|
| 기록된 음의 입력 합 |−0.532678|−0.654593|0.121915|18.00%|
| 나머지 전류 |−0.270495|−0.766044|0.495549|73.16%|

나머지 전류에는 미관측185개 입력과 float32 누적 차이가 함께 들어간다. **73.16%는 이 조건별 전류 평균 차이의 산술적 분해이며, 제동 원인의 비율이나 물리적 인과 효과가 아니다.** 경계의 절대 입력을 대부분 설명해도 작은 조건 차이는 설명하지 못할 수 있다. 미관측 입력을0으로 채웠다면 이 공백을 놓쳤을 것이다.

기록된 억제 입력은 주로 IN16B016(graph122032)→MN144018, edge10511676이다.120 contacts의 rate baseline−3.6이 현재−.0036으로 조정돼 있다. post의 억제는 음반력 조건−.53268보다 양반력 조건−.65459에서 더 크다. E161의 이 뉴런 관련 hook은 knee-velocity proxy였으며 실제 load sensor라고 부를 근거가 없다. MN 자체의 post 평균 rate도 음2.87885/양3.36864여서 instantaneous current 평균만으로 운동 출력 차이를 설명하지 않는다.

![MN144018 입력 비교](C:/Users/ckthd/Dev/Projects/fly_lab/verification/banc-natural-gait-m3-20260919/e187-parent-figures/mn144018-inputs.png)

## 개발 및 검증

실제 `/root/e187_gpt6sol_mn_current`를 `gpt-6-sol/high`로 호출했다. 부모 FIRST에서601행을10초 control2000으로 접근하는 coverage 오류와 직접 CLI import 오류를 재현했다. 같은 Sol이 첫 수정을 수행했고 부모 SECOND에서 해결을 확인했다. 부모는 남은 Ruff format과 `*_change`가 초당 변화율이 아닌5ms당 backward difference라는 표기를 정리했다. 최종18 tests, Ruff, 직접 CLI가 통과했다. 초기 실패 스냅샷·리뷰는 삭제하지 않았다.

부모 검산 코드는 구현을 import하지 않고 별도 NumPy 수식으로511 입력 보존, raw prefix, 모든 기록된 incoming 항/잔여 항,25개의 CUDA 누적,52개 window의 조건 평균 등 **5023 검사**를 통과했다. 별도의 `gpt-6-sol/medium` 읽기 전용 감사도 같은 자료로 단일 효능 개입이 아직 정당화되지 않는다고 판단했다. E95/E160/E161 및 E03/E39/E178의 다른 실행·관측 범위를 검토했으며 새로운 모델이나 수치를 재실험하지 않았다.

원본 CPU 분석1회 감독 벽시간4.801734200097초(내부4.349678초), 제한45초, retry0, CPU1, 종료코드0, timeout/남은 owned process0이다. 출력은 analysis.json/aligned.npz/terminal.json,144배열,약2.826MB다. 원본 누적은 **4351.08495669975/4994초**, 잔여642.91504330025초다. 부모 metadata3.1818551초, 최초/두번째 tests2.0606241/2.0931158초, 최종tests2.0216147초, 독립 검산1.6145561초, 결과·도표.8802839초는 별도 기록했다. 두 metadata setup 오류와 기타 준비/소스 리뷰 시간도 원본0으로 구분했으며 준비 비용 전체가 합산됐다고 주장하지 않는다.

## 다음 결정

현재 relay의 일괄 감쇠/증폭이나 이미 기각된 group67 40ms를 반복하지 않는다. [다음 관측 초안](BANC_M3_AFTER_E187_ALL_INCOMING_DRAFT_20260924.md)은 E185와 같은3초 baseline에서 MN144018의195개 선행 rate/mask를 동기 기록하는1회 관측으로 좁힌다. 기존 E03/E39는 조건이 다른 실행이고 E178은 경계 상태여서 현재8사건의 미관측 입력을 대신하지 못한다. 전체 기여의 CUDA 일치·사건별 반례를 확인한 뒤 기존 해부 연결의 제한 대조를 고른다. F6/UI/M2는 유지한다.

증거: `verification/banc-natural-gait-m3-20260919/`의 `e187-mn-current-freeze.json`, `parent-e187-source-review.json`, `parent-e187-first-review.json`, `parent-e187-second-review.json`, `parent-e187-supervisor.json`, `parent-e187-output-review.json`, `parent-e187-observations.json`, `parent-e187-final-review.json`; 원본은 `e187-mn-current/`다. 실행 유효성은통과, 인과 가설은미확정, 보행 후보는미채택, 전체 M3는active/미완료다.
