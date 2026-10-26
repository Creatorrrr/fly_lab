# E308 저장된 근육 전달 분해 결과

E306의 자원 감쇠 후보와 E307의 같은 SSP 대조군 사이 관절 명령 차이는 저장된 근육 활성 변화로 재구성된다. Root 독립 검산에서 합산 residual 최대는 1.4855738167396382e-15 rad였다. 이는 전달 계산의 확인이며 후보의 원래 시작 기준 실패, 자연 보행 미달을 바꾸지 않는다.

가장 큰 active−neutral 차이는 control98 / .490초 / 왼쪽 가운데 다리 coxa roll(index8)의 −.0347909692497127 rad다. 그 관절의 유일한 비영 moment는 muscle19 `lm/pleural_remotor_and_abductor_muscle`의 −4이며, 활성 변화 +.43488711562140825가 명령 차이를 설명한다. 이 근육은 **fixed original_activation 경로**를 사용하고 운동뉴런92104 하나에 연결된다. 따라서 이 최대 차이를 새 variable-muscle hinge 계수의 오동작으로 해석하지 않는다. 기존 근육 경로에서도 변한 신경 활동이 명령으로 전달됐다.

해당 시점 MN92104 rate는 active5.500031471252441 / neutral5.0511369705200195, 직접 drive는 양쪽0, adaptation은 .12969271838665009 / .12557324767112732다. 실제 CSR 입력212개 중185개(87.2642%)의 시간 기록만 있다. 관측된 양수 연결의 입력 변화 합은 +.13483282660528847, 음수 연결 합은 +.20190594031104747로, 부분합에서는 흥분 증가와 억제 약화가 함께 나타난다. 특히 억제 연결113364/145794의 rate 감소가 양수 차이를 만들지만 반대 방향 항도 있다. 이 source들의 endpoint R/g는1이다. R/g 보호가 그 source의 rate 불변을 뜻하지 않는다.

나머지27개 입력은 unavailable/NaN으로 보존했다. 전체 입력을 재구성하거나 두 source의 인과 효과를 입증했다고 하지 않는다. 값은 float64 `weight32 * delta_rate32` 관측 끝점 합이며 실제 packed CUDA stage current나 전체 유효 drive가 아니다. 마스크 근거도 초기·최종 상태와 고정 실행 경로에 한정한다.

별도 원래99개 startup 참조 최대 오차는 control99 / .495초 / joint8의 .06620980268835475 rad다. 두 실행의 최대 차이 시점과 혼동하지 않는다. 모든42관절×96근육×101관측, 고정·가변 근육 활성, major MN rate/drive/adaptation, 입력 부호별 항을 `verification/banc-natural-gait-m3-20260919/e308-resource-muscle-contrast/`에 보존했다. 절대 q 재계산은 control1..100에서 root와 원자료가 정확히 같고, 활성 hinge 재계산 residual은 최대8.526512829121202e-14다. Control0은 decode 전 anchor로 별도 남겼다.

실제 gpt-6-sol/xhigh 구현, root FIRST 전체 소스 검토 및53개 생성 검사/Ruff를 통과했다. 구현 중 root가 실제 body.modelHash, 원래99개 참조 시간축, E306 biological_claim 필드 부재를 확인해 Sol이 반영했다. FIRST 이후 R1이 필요한 결함은 없었다. 출력 실패와 잠긴 seal의 명시 무효화도 검사했다. Agent 전체 생성 검사139회 PASS, 생성명령5.9357422초/구조 읽기5.4787274초; root 생성 감독1.2313079999876209초, passive pin 검증1.4061660000006668초, 독립 원자료 감사1.0763538000173867초다. 초기 lint/조회/patch-context 실패도 source-review에 보존했다.

원본 저장 분석1회/재시도0은 supervisor2.6204322000266984초, 내부2.3270041000214405초(계산.22546739998506382초), supervisor peak282529792B/자체 peak286031872B였다. 새 model/neural/body/physics/CUDA/fit/solver 호출은0이다. 원본 누적 **8270.397776999514 / 배정8290초**, 잔여19.602223000485537초, 예약0·소유 프로세스 종료 확인. 결과 SHA `b9312bb4e7947acbfd6bcdf3ef11aacc3e577267cdd73028a0afce321c4af670`, output manifest SHA `d69111a9c26a04cb00abb6a22f3757d8d00d29b205ce15429db746a57f2fb8ca`; root 독립 근거는 `parent-e308-original-raw-audit.json`이다.

현재 판단은 자원 동역학의 영향 크기를 직접 검사할 근거가 생겼다는 것이다. 특정 운동뉴런을 골라 보정하거나 decoder/원래 startup 문턱을 바꾸지 않는다. F1–F8, M2/UI/F6 미노출, 이전 실패와 누적 비용은 유지하며 전체 목표는 진행 중이다.
