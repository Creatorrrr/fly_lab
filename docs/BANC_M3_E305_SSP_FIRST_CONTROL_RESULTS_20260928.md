# E305 전체 BANC의 SSP 첫 입력 결과

2026-09-28. 실제 전체 BANC에서 같은 첫 held drive를 SSP neutral과 보호 roster 후보에 각각20tick 적용했고 **양쪽 모두 유효하게 완료**했다. E302의 tick10/stage3 음수 문제를 겪었던 동일 초기 입력 구간을 새 적분기로 통과했다. 실제 신체 보행·장기 안정성·생물학적 보행은 이 결과로 검증되지 않았다.

Root 설계 → 실제 gpt-6-sol/high 구현 → root FIRST 전체 소스 검토/32fake테스트/Ruff → 저장 입력 gate → 한 번의 실제 실행 → root 원자료 검산 순서다. FIRST 후 수정은 필요하지 않았다. 구현 SHA `d3d3c206f08f347016c095ae7241265422ff0ffed18cd4121ec68988c6988731`, 테스트 `c22ec9beaee32200074964a7daff6cf4f72319ed967f4b0197188dc31e65a51e`를 동결했다. Agent의 모든 시도19.119061초에는 최초 fixture 실패1건/Ruff3건을 포함한다. Root 생성 supervisor4.735469100007322초/RSS637681664B, 저장입력 passivegate9.129520799964666초이며 이 단계 모델/CUDA/body생성은0이다.

158706개 세포, 원래 CSR/명시적0/파라미터/원래tick0 상태를 검증했다. Drive는 E301 neutral/final의 실측 fullfloat32이며, 실패했던 candidate의 미저장 drive와 같다는 것은 기존 초기 상태 및 생산 코드의 동일성에 근거한 추론이다. 새 코드가 candidate의 옛 drive를 직접 측정했다고 주장하지 않는다.

| 항목 | 실측/저장 검산 |
|---|---|
| 실제 advance | neutral1회20tick, candidate1회20tick; 실패/재시도0 |
| 신경 상태 | 모든 rate/adaptation finite 및 비음수, rate 최대44.23986053466797 |
| 후보 resource | 최소0.9991328143485322, 최대1; 39source가1미만, 비활성source는1 |
| 끝점 factor | 두 arm 모두7136개 g=1; 중간 stage history 실측은 아님 |
| 후보 대 SSP neutral | rate/adaptation/drive/mask/g 끝점 byteexact |
| SSP neutral 대 기존 RK4 rate | RMS7.839338491322014e-6, 최대차0.0006058216094970703 |
| SSP neutral 대 기존 RK4 adaptation | RMS6.843821496448349e-7, 최대차0.00010120868682861328 |

새 방법과 기존 RK4의 차이는 그대로 보고한다. 과거 neutral exact gate를 통과했다고 하지 않고, 기존 startup/short/F1–F8 기준도 변경하지 않는다. 새 neutral identity는 `f9ea25308cf5e074047bf16afba55de9fac89411ca48127f2c37cc4e6e34c7be`, 후보는 `6a745fd026eb8d3555147c7e61009e8f1b42826ceffc19893e07457e5f98fbc5`다.

실제 신경생성2, science40tick, 실행된 setup80tick/capture기록40tick/unknown0, body/physics/fit0이다. 실제 supervisor15.898630400013644초/RSS1852678144B, runner11.850193299993407초였다. 완료 상태를 저장한105개 manifest 파일과 별도 result/sha의 총 출력122518206B를 검증했다. 모든 소유 PID는 종료되었다. Source112개/input829개/모든 출력 byte와 수치 차이를 root가 독립 재계산했다.

원자료 `verification/banc-natural-gait-m3-20260919/e305-ssp-first-control/`, root검산 `parent-e305-original-raw-audit.json`이다. Result SHA `cf1a7ca75e32210f34e4e80aee2ff417e76d07eea2b20136e6a77699b8605315`, output manifest SHA `86ce5e49ed0b176559c9601cde7ba99321f870874ff5952ed30e8262e59e27c2`. 독립 검산은1.8931404000031762초로 원본추가실행0이다.

원본비용15.898630400013644초를 모두 이월하여 누적 **8160.007052199511/8210초**, 잔여49.992947800488764초, 예약0이다. Run_valid는 첫 신경입력 자격의 성공이며, F2/F1–F8/F6/adoption/생물학 검증은 미완료다. 다음 단계는 새 solver와 동일 후보를 실제 body에 연결해 첫 입력의 신경 일치 및 원래 .5초 startup gate를 확인하고, 통과할 때만 같은 session에서1.5초까지 진행하는 것이다.

추가 저장자료 분해(원본실행0,0.048079초):391 motor_adaptation.ids에서 SSP 대 RK4 rate 최대차0.00010776519775390625/RMS7.025673390366757e-6, adaptation최대차3.965396899729967e-9이다. 이는 첫5ms만의차이이며 장기보행영향이나새합격기준이아니다. `parent-e305-saved-difference-localization.json`에기록했다.
