# E184: 기존 원자료로 전진 중단의 시차와 접촉을 판별

이전 E183은 실제5팔과 독립 검산으로 LH+.125의 J/slip 중간 개선을 찾았으나 최장 전진 .215초/각 다리 적격 주기0을 바꾸지 못했다. 이전 goal turn은 progress이며 실행 process는 종료됐음을 현재 다시 확인했다. 자연 보행 진입과 F1–F8은 미완료다. 같은 theta grid를 다시 평가하지 않고 기존 신호에서 다음 개입의 대상을 고른다.

입력은 검증된 `e182-native-timing-recovery/baseline/`과 `e183-remaining-native-timing/axis-lh/plus/`의3초 raw 두 개뿐이다. E15010초나 다른 후보는 이번 비교에 추가하지 않는다. 각 trace/native-contact/native-recruitment-state/timing-parameters/start state 및 필요한 start 배열, result와 기존 parent raw review의 해시/전체 raw inventory를 freeze에 고정한다. 단일 CPU, 실제 신경/물리/모델 구성/GPU/solver0이다. 새 파일은 `tools/analyze_c_banc_m3_forward_break.py`, `tests/test_c_banc_forward_break.py` 두 개다. 부모가 설계하고 actual gpt-6-sol/high가 최초 구현한다. 시간 정렬·좌표·조건부 결측·신경 그룹 집계를 함께 다루므로 high를 선택한다. 부모 FIRST→같은 Sol 첫 수정→부모 SECOND→잔여 부모 수정 절차를 유지한다.

원래 전진 정의를 유지한다. t=0..3, 601개의5ms 제어 표본, native30000개의100us 표본을 검증하고 끊긴 표본을 보간하지 않는다. `analyze_c_banc_biological_gait.position_forward_speed`의 몸통 위치 미분·물리 forward basis 투영·Gaussian20ms를 재사용한다. [.5,3)에서 speed[i-1]>.5, speed[i]<=.5인 모든 하향 통과를 기록한다. 이벤트 창은[-100,+100]ms의41개 제어 표본이며 창 전체가[.5,3)에 있는 것만 summary에 적격이다. 사건 창이 겹치면 시간순 첫 사건을 유지하고 이전 선택 창의 마지막 표본보다 다음 창의 첫 표본이 뒤인 사건만 집계한다. 제외 사유와 전체 사건을 모두 보존한다. 동일 한 궤적에서 나온 반복 사건이므로 독립 생물학적 반복이나 통계적 인과 검증으로 취급하지 않는다.

native 시각은 (1..30000)*.0001과 허용오차1e-8로 대조한 뒤 제어 tick1..600의50개씩 오른쪽 닫힌 (t-.005,t] 구간에 대응시킨다. row0에는 native 관측이 없으며0으로 채우지 않는다. 매 구간의 다리별 평균 normal_load_bw, load>=.02인 표본 비율, 지지 수 평균/0개 지지 비율을 계산한다. slip은 load>=.02인 실제 native 표본의 load 가중 평균이며 미지지 다리의 slip은 결측이다. 지지 표본에 slip_valid=false 또는 비유한 값이 있으면 거부한다. 평균들의 비가 아니라 같은 원표본의 분자/분모 합을 사용한다. 하중은 기존 adhesion 포함 측정임을 유지한다.

위치의 UI 변환은 flylab/common.py의 S와 walking_trace.capture_control을 따른다. body.legs의 마지막 점은 tarsus5 body origin이고 실제 접촉점이 아니다. mm/y-up 기록에서 이 proxy의 world 속도를 현재 body forward에 투영한 값과, body 상대 forward 위치의 미분을 각각 계산한다. 후자는 몸체 회전에 따른 좌표 변화도 포함한다. 두 값을 native 접촉 slip/추진력으로 혼동하지 않는다. 실제 ground tangential force/fore-aft impulse는 현 raw에 없다는 제한을 명시한다.

motor_rate391→membership96→groups.leg의 다리 순서lf,lm,lh,rf,rm,rh를 실제 metadata에서 검증한다. 각 MN/근육을 누락 또는 중복하지 않고 다리별 MN 평균, 근육활성 평균 및 차분 RMS를 계산하되 원래391/96배열도 출력 raw에 남긴다. 관절 명령/실제/오차42열은 start calibration moment의 비제로 근육 열과 group leg로 row의 다리를 검증해 매핑한다(모든 row가 정확히 한 다리, 다리당7row). 이 조건이 아니면 추측하지 않고 거부한다. 다리별 실제 관절 RMS 추종오차·명령 변화 RMS와 전체 명령/실제를 보존한다. 기록 row i의 command가 이미 지난(t[i-1],t[i]]를 구동했다는 정의를 유지해 5ms보다 작은 선후관계를 주장하지 않는다.

경쟁 설명은 (A) 지지 손실이 전진 중단과 겹침, (B) 지지가 남은 상태의 발 이동/slip 및 여러 다리 명령 조합, (C) 전진 중단에 앞선 운동 출력/활성 변화다. 관측 정리는 이벤트 전체 곡선과 고정 pre[-100,-50), lead[-50,0), post[0,50), recover[50,100]ms의 구간 통계다. 다리별 결과와 표본 수/결측을 그대로 제공하고, 사후 특정 유리한 사건/다리/lag만 선택하지 않는다. 지원 이벤트별 수치, 팔별 평균 곡선, 전체3초 개요와 이벤트 평균 PNG, JSON/NPZ를 남긴다. 히트맵/plot은 같은 raw 배열에서 그리고 임계선·event 시간·단위/후보를 표시한다. 자동 인과 판정이나 p-value/생물학적 동등성 주장은 없다. J/bout의 원 결과 재현은 정렬 검사이며 성공 기준 완화가 아니다.

해시와 source gate가 검증된 뒤 create-only 출력 `verification/banc-natural-gait-m3-20260919/e184-forward-break/`에 한 번 실행한다. 전체 공식 분석60벽초(cap), 내부50초/부모 외부55초+정리5초, output256MiB, free디스크1GiB, 재실행0. 이전 누적4233.282374799717/4994초를 승계하며 모든 본 분석 import/입력검증/실패/저장/plot 시간을 기록한다. 부모 합성/메타 준비/검토 시간은 별도로 기록하며 원본 분석 실패를 지우지 않는다. 원본 실행은 부모가 맡고 Sol은 합성 테스트만 실행한다. import/--help는 original 파일을 읽지 않으며 GraphStore/StateStore/BancWalkingSession/MuJoCo/Torch를 불러오거나 instantiate하지 않는다. 필요한 start npy는 JSON의 안전한 상대경로를 검증해 np.load(allow_pickle=False)로만 읽는다.

결과 후 부모가 scalar clock/binning·이벤트/overlap 선택·집계 및 실제 그림을 검산한다. 실제 출력 변화가 선행하는지와 지지/추종 변화가 선행하는지 관측하고 제한된 다음 개입을 설계한다. 위상만 맞고 인과가 불명확하면 부족한 접선 힘/추진 충격량의 최소 native 관측을 별도 사전 설계한다. 분석 도구/PNG/테스트는 최종 보행 구현의 대체물이 아니다. M2/UI/F6 노출0 및 모든 실패, 최종 신경-only·해부학적 provenance·F1–F8은 유지한다.

## 원자료 분석 전 메타데이터 정정 v2

첫 metadata preflight는 moment의 0-based row 30이 모두 0이어서 중단됐다. 실패 스크립트·v1 설계·오류·실행 비용은 `parent-e184-prepare-v1-failed.py`, `e184-forward-break-design-v1.md`, `parent-e184-mapping-preflight-failure.json`에 보존한다. 아직 이벤트 분석은 실행하지 않았다. moment만으로 모든 관절의 다리를 알 수 있다는 가정을 다음 근거로 대체한다. 관측 창·성공 기준·원본 팔·횟수·자원 한도는 바꾸지 않는다.

설치된 FlyGym의 pinned commit ca65a510c2afe6ac61c51df4f274c8d190c2f95f 및 해당 anatomy/flybody/base_fly/tree/simulation 소스와 FlyLab body/body_options를 읽어 실제 순서를 추적했다. LEGS 순서는 lf,lm,lh,rf,rm,rh이며 anatomical edge 삽입 순서를 유지하는 DFS와 YAW_ROLL_PITCH 순서를 쓴다. 생물학적 관절 축 제한과 LEGS_ACTIVE_ONLY를 적용하면 다리마다 coxa yaw/roll/pitch, trochanterfemur roll/pitch, tibia pitch, tarsus1 pitch의 7개가 남는다. actuator dict 삽입 순서와 Simulation 내부 ID 순서가 이를 유지하며 whole_body에서도 FlyLab의 leg_action_indices가 같은 42개 순서를 고른다. 이 추적은 모델 생성이나 FlyGym import 없이 수행하며 설치 파일과 이웃 소스의 해시 일치도 확인한다.

이렇게 도출한 이름/다리/row 42개의 provenance JSON과 원본 소스 스냅샷을 freeze에 고정한다. 모든 비제로 moment row가 이 다리 매핑과 일치해야 한다. 0인 row는 정확히 [30]만 허용하며 이는 rm coxa pitch이다. 이 관절도 RM의 7관절 추종/명령 RMS에 포함한다. 0 계수는 다리 정체성의 부재가 아니며, 원인이나 정상 생리라는 뜻도 아니다. 이 단계에서는 moment나 관절 명령을 수정하지 않는다.
