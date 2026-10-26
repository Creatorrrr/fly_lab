# E265: LH 중계111970의 전체 수신 전류 재현 — 2026-09-26

M3 finite-goal/F1–F8은 그대로다. E264의 상수 보정은 유효하게 기각됐으며 후보를 채택하거나 신체 시험으로 승격하지 않는다. 이번 진단은 **남은 중계 입력 차이가 복원한 여섯 감각 항의 시간 변화로 설명되는지, 다른428개 회로 입력 변화가 중요한지**를 구분한다. 기존 관측은 수신434개 중271개만 포함하므로 일부 자료를 전체 전류처럼 해석하지 않는다.

## 고정 비교와 실제 실행

새 모수나 새 후보는 없다. B는 기존 W0/U0, X는 기존 W6/U0, Z는 기존 W6/U0와 drive[111970]의 고정 bias32=−0.19308185577392578이다. B→X→Z 순서로 각100controls/.5초를 재현한다. B는 E260 B, X는 E264 X_U0, Z는 E264 Z_U0를 기준으로 한다. B용W0와 X/Z공유W6를 각각 한 번 생성하며, 각 팔은 원래 전체 control0 상태에서 복원한다. 입력·출력 mask·원래 core·MN·운동 변환·파라미터는 바꾸지 않는다. E264의 상수 보정 도출도 재적합하지 않는다.

각111970 수신434개의 presynaptic 발화와 mask, 간선별 float64 `W*r*mask`, 실제 장치의 packed float32 수신 합, 해당 뉴런의 rate/drive/adaptation을101개 endpoint에서 기록한다. 기존 union1239에 누락된 presynaptic163을 더한 관측을 사용해 기존 모든 MN391/관측82/core24/endpoint1239도 정확히 비교한다. 전체 신경 시작·끝 상태와 full drive SHA가 각각 기준과 exact여야 한다. q19/bank/세근육의 기존 산술은 재생산해1e−10 이내의 기존값과 비교한다. B에는 기존 bank44 두 열과 E263의 q19/activation 참조를, X/Z에는 E264의 full bank를 사용한다.

`network._current` 관측303회는 endpoint당1회이며, 관측 직전·직후 네트워크 전체 snapshot이 exact여야 한다. packed float32 합은 독립적인32lane 합산 순서로 검증하고, float64 간선 합과의 차이는 반올림 정보로 따로 보존한다. float64 합이 장치 비트와 같다고 요구하지 않는다. 원래 RK4의 중간 stage 전류를 관측했다는 주장은 하지 않는다. 관측으로 신경을 한 tick이라도 진행하지 않는다.

## H265와 판정

각 팔의 float64 간선 전류를 `I_arm,e(t)`라 한다. 대상6개는 E256의 기존 간선 `[8191121,8191220,8191247,8191292,8191340,8191396]`이다. 모든434개의 부호 있는 합을 보존한다.

Z 대 B의 유효 입력 차이 `D=sum(I_Z−I_B)+(drive_Z−drive_B)−(z_Z−z_B)`를 다음처럼 나눈다.

- `S=sum_6(I_Z−I_B)+(drive_Z−drive_B)`
- `O=sum_other428(I_Z−I_B)−(z_Z−z_B)`
- 모든 endpoint에서 `D=S+O`를1e−10 이내로 확인한다. 같은 분해를 X 대 B, Z 대 X에도 제공한다.

H265는 **Z의 남은 유효 입력 차이는 위 S 항으로 충분히 설명된다**는 제한된 대수적 가설이다. startup1..99의 `RMS(O)/RMS(D)≤.2`, 미리 고른 controls43/56/70 각각 `abs(O)/abs(D)≤.2`와 `S*D>0`, startup S의 시간 표준편차>1e−12를 동시에 요구한다. RMS(D) 또는 세 고정시점 abs(D)가1e−12 이하이면 `inconclusive`; 모두 정의됐고 조건미달이면 `refuted`. 실제 입력/복원/관측/자원/게시 검증이 실패하면 `not_tested`이며 false와 구분한다.

각 팔과 대조의 rate, margin, adaptation, 전체·6개·나머지 전류의 mean/min/max/RMS 및 부호별 합을 보고한다. 434개 중 과거 관측에 있던271개와 빠진163개를 별도로 표시한다. H265는 원래 뉴런 rate·q19 오차의 전체 원인, 경로의 고유 인과 기여율, 생물학적 회로 기능 또는 보행 개선을 증명하지 않는다.

H265 지지이면 감각 입력의 정적·시간 변화 성분을 함께 다루는 인터페이스/기존 효능 가정을 검토하고, 미지지이면 다른428개 입력의 실제 기여부터 검토한다. 둘 다 새 후보는 별도 사전 설계해야 한다. 기존 적응gain0만으로 적응을 켜거나 동적 중심화를 채택하지 않는다. bias·평균 창·τ의 격자 탐색, 동일 decoder 재적합, 신체승격은 이번 범위에 없다.

## 구현·검증·자원

현재 세션이 설계와 독립 합산/상태 감사를 맡는다. 실제 gpt-6-sol/high가 `tools/probe_c_banc_m3_lh_bridge_current.py`와 `tests/test_c_banc_lh_bridge_current.py`만 구현한다. 실제CUDA 관측/복원과 원자료 대조를 함께 다루므로 high다. 부모 FIRST→필요 시 같은 Sol의 첫 수정→부모 SECOND→남은 수정은 부모 순서를 따른다. 기존 E264/E260의 준비·순수 연산·constructor를 명시적으로 재사용할 수 있지만 기존 파일 변경이나 전역 monkeypatch는 하지 않는다.

원본1회/retry0, CPU1/GPU1, 신경 생성2/setup120실행+capture40기록, 과학300controls=6000ticks/current probes303, 신체·물리·fit·solver0. internal28초(마감6초 확보)/external32초/supervisor35초, host6GiB/추가GPU3GiB/출력256MiB/min disk1GiB/min GPU3GiB다. E264의2생성/8000tick 실행18.9163초와 E260의4생성/10000tick/505probe 실행24.0103초를 근거로35초를 예약한다. 누적7101.567570399613/7138초, 잔여36.432429600387초 안에서 배정하며 전체 예산을 새로 만들지 않는다. 준비·생성·감사 및 실패 비용은 별도로 기록하고 미측정 개발 시간은null이다.

출력 `verification/banc-natural-gait-m3-20260919/e265-lh-bridge-current/`에 시작·끝/부분 상태, 모든 원자료·드라이브·bank·파라미터·CSR/backend·시간축·입력 hash·호출 장부·오류·판정·manifest를 남긴다. 일부 실패 뒤에 완료한 것처럼 기록하지 않는다. E150부분/E257기각/E261무효/E264기각, 기존M2/UI, F6미노출, 동일 최종 후보의F1–F8미완료를 유지한다.
