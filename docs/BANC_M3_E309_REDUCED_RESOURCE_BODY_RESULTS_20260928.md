# E309 소모율 축소 후보: 시작 최대 오차 기준 미달

실제 CUDA 신경망과 신체를 0.5초 실행했다. 활성 source 1513개의 utilization만 .05에서 .005로 줄였고, 기존 연결·회복 시간·적분기·근육 변환·감각·초기 신체 상태는 유지했다. 실행과 원자료 검산은 유효하지만, 원래 startup q max가 **.04078679924683386 rad**로 고정 기준 .04를 넘었다. 후보는 채택하지 않으며 1.5초 short, 10초, F6은 실행하지 않았다. 자연 보행이나 보행 품질 개선은 아직 입증하지 못했다.

| 같은 SSP 조건 | q RMS (rad) | q max (rad) | 원래 시작 판정 |
|---|---:|---:|---|
| E307, u=0 | .017728691633713805 | .0396360246919627 | 통과 |
| E306, u=.05 | .017894977272018938 | .06620980268835475 | 미달 |
| E309, u=.005 | .01773341164402887 | .04078679924683386 | 미달 |

E309 최대 오차는 control95 / .475초 / joint18에서 발생했다. 실제 명령 .08027085475947528, 원래 참조 .12105765400630913 rad이며, .04를 처음 넘은 control도95다. E306의 최대였던 joint8과 다르므로 한 근육의 오차만으로 전체 시작 실패를 설명하지 않는다. E307 대비 전체 .5초 q 차이의 최대는 .004252314982940847 rad, RMS .00023113341588325246이며 최대 위치는 control100/joint18이다. E306 대비 최대 차이는 .0367290763140965 rad, control98/joint8이다. Root audit의 `E306_u005` 키는 E306의 **u=.05** 기록을 가리킨다.

LTM RMS .0013637403603647155 / max .013190561316255553, 최소 upright .9255510569476302, 비발 접촉0으로 나머지 시작 기준은 통과했다. 신경 resource fault0이다. 기존 whole 초기 상태는 새 resource 계층을 제거하면 byteexact이고, 첫20 신경 tick의 rate/adaptation/drive/mask/tick 및 첫 trace2행/native50도 E307과 exact다. 새 resource·metadata·identity는 별도로 검증했으며 옛 whole network와 exact라고 주장하지 않는다.

끝점 resource 최소 .8823834098465781, g32 최소 .9167576432228088, g<1은48600 표본/활성1105 source에서 관측됐다. 따라서 u0와 동일한 무효 개입은 아니다. 다만 내부 stage의 전체 이력을 측정한 값은 아니며 신경 교란의 존재가 협응 개선을 뜻하지 않는다. 7136 source 중 비활성5623개 R은 계속1이고, utilization 이외 모든 선택 배열은 원본과 exact다. 변경 전후 metadata와 새 identity `b451037f14cb789d0447aca39db7c043eceedf8852e9822b05d2785f4e70fafa`를 독립 재구성했다.

실제 gpt-6-sol/xhigh FIRST 구현 뒤 root 소스 검토와 생성39개/Ruff를 통과했다. 검토 결함이 없어 R1은 요청하지 않았다. Agent 생성검사77 cases/13.929738초, root 생성검사39 cases/6.437890900007915초이며 실제 원본 실행과 구분한다. Agent 지원 읽기·편집의 전체 시간은 한 출력 누락 때문에 불명이다. 저장 입력 passive gate13.354751500010025초, root 원자료 audit2.9037275000009686초도 새 모델 호출0이다.

원본은 후보 신체1 /100control /science neural2000tick /physics5000step /setup280tick /capture140tick /settling2000step, 재시도0이다. Supervisor58.20830920001026초, runner40.60637529997621초, tree RSS peak3734638592byte, 산출물300621600byte다. 소유 PID11184/17352는 종료했다. Supervisor 시간을 한 번만 합산한 누적 **8328.606086199525 / 배정8400초**, 잔여71.39391380047528초, 예약0이다.

근거는 `verification/banc-natural-gait-m3-20260919/e309-reduced-resource-body/`의 raw 및 `parent-e309-original-raw-audit.json`이다. Freeze `10a8ae10150e6de006d7705346272e977ce84b99b4833e1cf60d4f8205a17343`, review `6585540f59985d3fabe6bb9431ada58e10342cc4545412b0a3b0d3369bfb1e9c`, result `5c0c7332f391ba4a5d15c7b930c922b059b2d3146d40c755dbca6f9055d21574`, output manifest `f9f19871106ce50600f1784c8bb345376899d93800ace08c85e5e1ba886c935e`로 고정했다. Root가 재사용 소스122개/입력1963개, 출력 manifest, 원래 시작 계산, resource factor와 metadata identity, 실제 clock 및 비용을 검산했다.

단일 소모율 축소는 시작 보존 가설을 충족하지 못했다. 자동 u/tau 탐색으로 이어가지 않는다. 감각·재귀 회로·근육 전달의 기존 실패와 반례를 비교하여 구조적 대안을 검토한다. Resource 방식의 뒤 시점 협응 효과는 이번 실험에서 **미검증**이며, 이를 판단할 필요가 있으면 채택과 분리한 별도 유한 진단을 사전 설계해야 한다. 원래 F1–F8/M2/UI/F6 미노출, 모든 과거 실패·비용은 유지하고 목표는 계속 진행한다.
