# E259 LH 무릎 신전근의 저장 입력·변환 분해 — 2026-09-25

E258에서 최초control56/최대70 초기 관절19(tibia_pitch) 오차의 양의 방향 기여는 muscle44 tibia_extensor_muscle이100%였다. 기존muscle41 설명은 기각한다. 고정muscle44의 전체2MN columns[68,107]/graph rows[31172,47329]를 사용해 감소한 근육 출력과 관측된 기존 억제 입력을 분리한다. 반응 순위로 MN을 미리 선택하지 않는다.

## 고정 질문과 분기

H259_single_inhibitory_route는 startup1..99에서 다음 두 조건의 conjunction이다.

1. 각2MN의 근육 기여 변화 delta A_m을 음의 방향으로 투영한 P_m=sum_t max(−delta A_m,0)의 전체가 양수이고, 최대 MN의 비중이80% 이상이다. 동률은 작은 graph row ID 순서다.
2. 그 MN에 대한 기존4bridge의 직접 입력 변화 delta I_j=W[m,j]*(rate_C[j]−rate_B[j])*mask[j]에서 P_j=sum_t max(−delta I_j,0)를 구한다. 전체가 양수이며 **사전 지정 bridge111970**의 비중이80% 이상이다. 개입6간선의 유일한 목적지가111970이므로 결과에 따라 다른 bridge로 바꾸지 않는다.

분모0은 unavailable/미지지다. 양·음·signed·투영 항을 함께 보존한다. 이는 기존E253과 같은80% 진단 문턱이며 부분 관측 입력의 집중도일 뿐 전체 신경 원인이나 생물학적 기준이 아니다. 지지 시 이 관측 경로를 분리하는 신경 대조를 별도로 설계하고, 미지지 시 분산된 MN·관측/미관측 입력을 우선 검토한다. 어느 분기도 gain sweep, 새 신체 실행, 근육 계수 수정, 채택을 자동 허용하지 않는다.

## 고정 산술과 입력

E257 두301 endpoint의 bank391×3/activation96/control391/activity82/core24 및 초기 decoder params/output mask/실제 CSR 두 팔을 고정한다. E258 유효결과·부모검산·최종검토 및 E150 calibration/근육명과 E2566간선 provenance를 연결한다. 두 MN의 모든 incoming CSR 행은 개입 대상이 아니므로 exact 동일해야 한다. 신경·decoder·bank update·fit은 수행하지 않는다.

동일 tau/kappa/low/high slopes에서 각MN×3bank의 activation=low*min(bank,kappa)+high*max(bank−kappa,0)를 계산한다. 두MN 합이 모든301행 저장 activation44와1e−10 이내로 일치해야 한다. .02*moment[19,44]*delta A_m 합은 E258 고정muscle44 명령 기여와 일치해야 한다. 이 moment 계수는 저장calibration의1.8088900095777785이며 단위1로 가정하지 않는다.

각bank의 delta와 low/high/kappa/양 팔 hinge region 및 실제 secant slope delta contribution/delta bank를 저장한다. delta bank=0이면 secant는NaN/available=false이며 기여 변화0을 검증한다. secant는[minimum(low,high),maximum(low,high)] 범위여야 하고, |delta activation|≤sum(max(low,high)*|delta bank|)를1e−10 이내로 확인한다. 이는 기존 변환의 민감도 검산이며 새 계수·counterfactual fit이나 과도한 증폭 판정 문턱을 만들지 않는다.

관측rate 합집합은 source23+bridge4+MN391+original24 중 중복을 제거한440개다. 기존418개 밖의core22를 별도 category로 추가한다. 중복82/391/core는 값·ID·clock이 일치해야 하며 중복 합산하지 않는다. category는 bridge4 각각/source23/MN391/추가core22의7개다. 모든 incoming entry의 edge ID/pre/post/weight를 보존하고, 관측 entry만 실제 mask를 적용한301행 current를 계산한다. 미관측 entry는NaN/available=false로 보존하며0으로 간주하지 않는다. 모든 관측 값은finite여야 한다.

전체/관측/미관측 간선수·nonzero수·absolute weight 합과 비율을 MN별로 보고한다. 이는 topology weight coverage이며 실제 current coverage가 아니다. startup[1,100), steady[100,300), full[0,301), initial_200ms[1,41)의2MN·4bridge·관측 category 차이, 고정controls56/70 사례를 모두 보고한다. 모든rate 차이는 피연산자를f64로 승격한 후 계산한다. clock endpoint t*.005는5ms 내부 인과 순서가 아니다.

## 구현·검토와 자원

실제 gpt-6-sol/medium가 `tools/analyze_c_banc_m3_lh_extensor_input.py` 및 `tests/test_c_banc_lh_extensor_input.py`만 구현한다. 기존E253 산술·게시 코드를 명시적으로 적용하되2MN/6간선/7category/440관측/음의 집중도/고정bridge111970로 바꾸며 기존E253 파일은 변경하지 않는다. 부모FIRST→필요시같은Sol첫수정→SECOND→부모 잔여수정 순서다. 생성 검사는 임의 ID/기여 부호/음의 slope 차이/hinge 교차·0bank 변화/분모0/동률/미관측NaN/중복category/늦은 게시 실패 판정 철회를 다룬다.

출력 `verification/banc-natural-gait-m3-20260919/e259-lh-extensor-input/`에2MN×3bank 기여·민감도, 전체incoming 및 부분 관측current와availability, fixed mapping/window/case/H259, provenance/manifest/errors/ledger를 저장한다. 부모는 독립 positive hinge/scalar sum/CSR 합산으로 검산한다.

원본1/retry0/CPU1/GPU0, 모델·neural·physics·CUDA·fit·solver0. internal30(마감5)/external40/supervisor45초, host2GiB/output64MiB/free disk512MiB다. 누적6996.279667499624/7074초·잔여77.720332500376초에서45초 예약, 증액0. 원본당300초·과거 실패와 비용·별도 준비/생성/검산·미측정 개발null을 보존한다. E257기각/E150 best partial/M2/UI/F6 미노출/F1–F8미완료를 유지한다.
