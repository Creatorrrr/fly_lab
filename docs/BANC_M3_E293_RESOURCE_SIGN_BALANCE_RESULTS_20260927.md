# E293 저장된 전달 변화의 부호별 분해

E292에서는 resource를 흥분·억제 연결에 함께 적용하면 동일한 저장 발화율에서 **억제성 전달 감소가 흥분성 전달 감소보다 컸다.** 초기0.05–0.10초의 비율은약1.9배였고, 대상7136개신경 중약75%이상에서 모델 입력의 순변화가양수였다. 이 결과는 양·음 연결을 같은 규칙으로 약화한 구조를 구별해 시험할 근거이며, 보행 실패의 인과관계를 확정한 결과는아니다.

| 시각(s) | 억제완화/흥분손실 크기비 | 순입력 증가 post비율 | E150 대비 motor rate RMS차이 | 최대관절명령차이(rad) |
|---:|---:|---:|---:|---:|
| .05 | 1.9078 | 77.13% | .2412 | .01460 |
| .10 | 1.8969 | 75.08% | 2.9974 | .24251 |
| .20 | 1.7928 | 75.57% | 12.4175 | 1.15231 |
| .31 | 1.3178 | 69.97% | 22.1430 | 2.26888 |

계산은 원래선택725849개edge의부호·weight와 E292의각순간rate/R를사용했다. `J_exc/J_inh`와같은rate에resource를곱한`J_exc_R/J_inh_R`를각post별float64합으로분리했다. 단위는 모델입력단위이며 전류의물리단위(ampere)가아니다. 원래E150의실제신경current를재구성하거나, resource가없었을미래신경상태를시뮬레이션한것도아니다. GPU의float32warp/FTZ/RK4 exact 결과라는주장도하지않는다.

완성된63개control 관측은0..0.31초다. Native health검사는0.3124초upright=.89993517에서종료됐고신경partial state는0.315초다. 마지막신경state를0.31초rate와혼합하지않았다. 시작/partial끝output_mask는모두1을직접확인했고,중간mask불변은동결된실행경로에서mute호출이없다는코드검토근거다. 매tickmask를직접기록한것으로표시하지않았다.

actualgpt-6-sol/high가분석도구를구현했다. Root FIRST에서54tests/Ruff와별도6cell/5edge/3시점손계산을통과했다. 실제저장분석을별도동결한뒤감독3.9806917초에완료했고,root는다른연산순서의signed sparse matrix곱으로전체63×7136성분을다시계산했다. 최대절대차이는1.3643e−12였다. 원본신경·CUDA·신체·물리·solver·fit은0회이며현재원본누적7967.464174899509초는변하지않았다.

다음[E294설계](BANC_M3_E294_POSITIVE_SOURCE_RESOURCE_DESIGN_20260927.md)는활동이나순위가아닌원래연결부호만으로양수전용source3194개에u=.05를유지하고음수edge를가진source3942개는u=0으로한다. 모든source/edge/weight/tau/.5초startup/실제health/1.5초gait와F1–F8기준은유지한다. 같은rate에대한음수edge의전달가중치는보존되지만, 이후신경rate까지기존모델과같다고가정하지않는다. E294는검증전후보이고채택하지않았다.

전체배열·기술통계: `verification/banc-natural-gait-m3-20260919/e293-resource-sign-balance/`. Root독립검증: `parent-e293-saved-audit.json`. 소스: `tools/analyze_c_banc_m3_resource_sign_balance.py`.
