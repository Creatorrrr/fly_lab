# E296 — 핵심 회로 입력의 자원 계수를 보존한 실제 신체 결과

2026-09-28. **유효한 실험이지만 시작 조건 미달로 후보를 기각했다.** 기존 F1–F8은 미완료이며 제품 M3/UI에는 적용하지 않았다. Neutral .005초의 전체 과학 상태와 신체·운동 기록은 원래 결과와 exact였다. 각 실행의 실측 wall 시간은 별도로 보존했다.

## 실제 변경과 결과

E294의 positive-only 자원 소모에서, 선택된 outgoing 중 원래 core24로 향하는 연결이 하나라도 있는 source를 u0으로 면제했다. 원래7136 source/725849 연결과 CSR·해부학적 provenance는 유지했다. 1513 source/117973 연결만 u=.05, tau=.150을 사용했다. 이들은 premotor→premotor 연결이며 motor를 직접 향하는 선택 연결은0개다. 면제 source의 다른 outgoing도 함께 계수1이므로 core 보호만의 단독 인과 효과를 주장하지 않는다.

| 실제 0.5초 판별 | E294 | E296 | 고정 기준 |
| --- | ---: | ---: | ---: |
| 원래99행 관절 명령 RMS 오차 | 0.2280028963 | 0.01893939994 | ≤0.018 |
| 최대 절대 관절 명령 오차 | 1.3178885184 | 0.07604739934 | ≤0.04 |
| 힘줄 명령 RMS 오차 | 0.00139207825 | 0.00136387460 | ≤0.009 |
| 힘줄 명령 최대 절대 오차 | 0.01317739868 | 0.01319157226 | ≤0.025 |
| native upright 최저 | 0.9241210984 | 0.9228102840 | ≥0.9 |
| native 비발 접촉 수 | 0 | 0 | 0 |

E296는100control/5000native sample을 정상적으로 완료했다. 관절 명령 오차가 크게 줄었으나 두 기준을 넘었으므로1.5초 continuation과10초 검증은 수행하지 않았다. 생물학적 보행, 완전한 주기, 목표 도착을 확인한 결과가 아니다. u/tau·seed·문턱을 변경해 같은 후보를 다시 시도하지 않는다.

## 구현·독립 검증

실제 gpt-6-sol/high가 신규 runner/test를 구현했다. Root FIRST에서33pytest/Ruff와 실제 roster의 별도 집합 계산, 작은 혼합 부호·공유 source fixture, inherited source manifest를 검산하여 승인했다. 수정 결함이 없어서 Sol 첫수정이나 root 잔여수정은 발생하지 않았다. 기존 E289/E290 CUDA kernel/trial과 E291 실행·cleanup을 그대로 재사용했다.

원본 종료 후 root는 입력/source/output SHA, 원래 전체 시작 상태, neutral .005의 전체 과학 endpoint/trace/native/recruitment,100control·2000neural tick·5000physics step, 건강성, 원래 startup99행 오차를 별도로 재계산했다. 모든5623 비활성 source의 R이 기록 전체와 최종 체크포인트에서 exact1이며, core향6172 선택 연결도 그 집합에 들어감을 확인했다. 활성 source의 R 최저는.40832297628161457이다. 모든 원본 process 종료와 신체 close도 확인했다.

고정 원본 neutral identity는 유지하고 새 후보 identity `81b7ba38e8b45af66a253b0dd63416e378e2743459d6a1f2dbc4a34ca6c52369`를 별도로 검증했다. Freeze SHA `7c82276aec2049b68764e26b098ea333073711988f8311dfe3faeab7fa5d4936`, source-review SHA `f7cfec5a3fd458664fd17897ba03ccb64287b79e6cc30b0eabf6b14d7a14a098`이다.

Root의 후속 저장 탐색에서 E150/E257와 E296의 core24 drive/adaptation은101행 exact지만 rate는 동일하지 않았다. Controls1..100 rate RMS 변화 .07692251472/max .49401855469다. 이는 E294의45.7697 RMS보다 작지만, 계수 보존이 전체 미래 신경 상태 보존을 뜻하지 않음을 보여준다. 남은 관절 오차의 단일 원인을 확정하지 않는다. 다음은 원래 관측된55source와391motor의 입력·출력 및 모든42관절의 저장 변화 위치를 비교하는 단계다.

## 비용과 보존

원본1/retry0. 감독55.94453789998079초, peak host3201204224 bytes, 실제 신체 구성2회/close2회다. 과학2020neural ticks/5050physics steps 외에 알려진 setup560ticks, capture setup280ticks, settling4000physics steps를 별도 기록했다. 원본 누적8022.108435499544→**8078.0529733995245/8150초**, 잔여71.9470266004755초다. 단계 배정8095→8150은 원본 실행 전에 기록했고 과거 비용을 초기화하지 않았다.

Sol 생성 검증9.6554283초에는 Ruff I001 실패 .2903755초를 포함한다. Root 생성 검증 감독7.654103500011843초/peak746926080bytes, 처음 supervisor CLI 인자 오류 .4219182초(검사·원본 실행0)를 보존했다. 실제 입력 passive gate6.894550000026356초, 원본 후 root 감사1.8640178000205196초/process2.2465142초이며 모두 원본 추가 실행0이다. 현재 원본 예약0, 미노출 F6·기존 M2/E150/UI·모든 실패는 보존됐다.

증거는 `verification/banc-natural-gait-m3-20260919/`의 `parent-e296-source-review.json`, `parent-e296-passive-gate.json`, `parent-e296-original-supervisor.json`, `parent-e296-original-raw-audit.json`, `parent-e296-exploratory-core-and-topology.json`과 `e296-core-afferent-resource-body/` 원자료다.

![원래0.5초 기록의 명령 오차·몸통 자세·core24 활동 변화 비교](../verification/banc-natural-gait-m3-20260919/e296-saved-comparison.png)

그림은 저장 관측 비교이며 보행 영상이 아니다. 관절 오차는 log, core RMS는 symmetric log 축을 사용했다. 별도 plotting passive1.0510467999847606초/process1.7357794초, 원본0이며 root가 렌더를 확인했다.
