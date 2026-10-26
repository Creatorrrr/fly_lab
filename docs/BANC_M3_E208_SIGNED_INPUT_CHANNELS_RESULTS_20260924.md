# E208 결과: 합산 전류 뒤에 남아 있는 흥분·억제 입력

기존 E150의 전체158706 BANC→실제 신체10초를 한 번 재현하여391MN의 native RK 입력4채널을 기록했다. 각 채널160000×391 float32이며 total/positive recurrent/negative magnitude/drive를 같은 counter로 정렬했다. 새 신경 동역학이나 후보 계수는 적용하지 않았다.

현재 국소 반응 가족에서 후기 활성화가 막힌25MN 모두 positive RMS>=1e-3이고 negative RMS가 더 컸다. 사전 H208 기준13/25를 충족했다. 이 결과는 **양의 입력이 없어서만 침묵하는 것이 아니라, 현재 부호 모델의 더 큰 음의 입력과 합산되어 억제되는 사례가 존재함**을 보여준다. 새로운 내부 상태 모델이 보행을 개선한다는 증거나 receptor의 생물학적 실측 검증은 아니다.

전체 start/1.1s/final의 과학적 snapshot과2001개 trace, 모든 E201 생리·소유권 및 total current bytes가 원본과 exact였다. 실행 wall_s metadata만 기존 frozen comparator와 동일하게 비교에서 제외했고 값은 부모 검산 기록에 보존했다. 독립 검산의 signed 합 최대 절대 차이 0.000978469849, 사전 반올림 허용 대비 최대비 0.063681037416로 통과했다. 합산 순서 차이에 대한 이 허용은 원래 궤적 exact 조건에 적용하지 않았다.

부모는 실제 motor CSR rows와 저장된 전체 신경 rate/output mask를 사용하여 초기 stage0 및1.1초 다음 control221 첫 stage17600을 별도 float64 합산했다. 후기 snapshot의 E/I 최대 오차는 각각 0.000081943288/0.000168785876로 사전 오차 범위 안이었다. counter160000/overflow0/armed0,4채널 finite 및 E/I 비음수, 원자료 manifest를 확인했다.

실제 gpt-6-sol/high 구현의 첫 부모 리뷰에서 생성 시험 capture 계수와 RK 중간 단계 검사가 부족한 점을 지적했고 같은 Sol이 수정했다. 부모 SECOND에서 math/mock5개, Ruff, 작은 실제 CUDA 대조1회가 통과했다. 생성 모델2/warmup8/capture4/main8tick/신체0이며 원본 비용과 분리했다.

원본 감독 76.857505900000초, 누적 5502.980811599704/5624초, 잔여 121.019188400296초다. 원본 모델 구성7/warmup280/capture140/main40000신경tick·100000물리step, 출력 1115963216bytes, 실제 runner peak RSS 4227784704bytes다. 원본1/retry0/fit0, 잔류 owned process0. parent 처음 비교의 wall_s 제외 누락을 저장 자료 검산기에서 수정했으며 원본을 재실행하지 않았다.

후속은 같은 해부학적 입력의 두 부호와 인과적 기억을 구분하는 신경 내부 반응 모델을 설계하는 것이다. 현재7기저 bounded rectifier의 작은 bias 반복을 계속하지 않는다. 모든 입력은 실행 시 실제 연결/신경 상태에서 생성되어야 하며 외부 위상·궤적은 학습 목표로만 허용된다. 전달층과 신경 반응을 함께 바꿀 경우 결합 변화임을 밝히고 각 효과를 분리할 대조를 사전에 정한다. 아직 새 모델 fit·폐회로·F6·채택은0이고 F1–F8 active·미완료, M2·최선E150·UI·실패와 비용은 보존한다.

근거: `verification/banc-natural-gait-m3-20260919/e208-signed-input-channels/`, `parent-e208-source-review.json`, `parent-e208-generated-receipt.json`, `parent-e208-output-review.json`, `parent-e208-supervisor.json`.
