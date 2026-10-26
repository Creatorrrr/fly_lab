# E245: 같은 교사 보행에서 감각→신경 전달의 수동 관측

M3 finite-goal의 진단 자료 수집이다. E242의 뒤구간 적합 실패, E243의 큰 입력범위 이탈 비지지, E244 변화량 표현 기각을 보존한다. 같은 계수 가족의 재적합 대신 실제 수용기 특징과 감각/내부 뉴런 반응을 구별할 자료를 얻는다. 이번 수집 자체로 정보 손실·리듬 비동기를 확정하거나 자율 보행 후보를 채택하지 않는다. F1–F8, 최선 E150, M2, 미노출 F6와 활성 UI는 그대로다.

## 대상과 주장

부모는 기존 `build_spec(graph, 2)`를 메타데이터에만 적용하고, 실제 gpt-6-sol/high가 snapshot ID의 출처를 별도 감사했다. `verification/banc-natural-gait-m3-20260919/e245-passive-signal-roster.json`(SHA `f108afb5a709eda202137d29ce6b84c924da42aa6a12211d234b97cb612ec86b`)에 원래 순서의 241 ports, 3,604 sensory IDs, 53 중복제거 receptor keys와 그래프 주석을 저장했다. `core_adaptation.ids` 442개는 원래 설정 내부 24개와 추가 적응 제거 대상 418개의 합집합이다. 이 중 391개는 MN, 49개는 VNC intrinsic, 2개는 descending이다. 생물학적 CPG 목록이 아니다. intrinsic-timescale 대상 38개도 이 합집합에 포함된다. 관측 발화 열은 sensory와 442개 합집합의 정렬된 실제 node row 4,046개이며, 포트 순서와 MN391 순서로 되돌리는 열 매핑을 함께 고정한다. 모델 구성 시 roster와 실제 adapter/network를 대조한다.

가설 H245는 이 관측을 추가해도 기존 E240의 같은 교사 명령/원래 신경 동작/실제 신체가 정확하게 보존된다는 것이다. 경쟁 설명은 관측이 입력 시점·메서드 호출·상태·CUDA 스트림 순서를 바꾼다는 것이다. 대상은 E240과 같은 10초 2,000 controls, 원래 BANC 158,706 nodes 및 네이티브 tendon이다. 직접 신경 입력, efficacy, kinetics, decoder, 해부학 연결을 변경하지 않는다. 교사 q는 E85의 기존 2,000×42 float64 명령이며 native tendon은 해당 control의 원래 decoder에서 온다.

## 기록 시점과 구현 경계

1. c=1..2000의 encode 진입은 c−1 종료 신체를 읽는다. 원래 호출이 반환하는 `leg_observation`과 `knee_kinematics(include_direction=False)`를 한 번씩 가로채 복사한다. 계측 때문에 이 계산을 다시 호출하거나 인코더를 복제하지 않는다. 원래 dict/배열/반환 drive를 바꾸지 않는다. 각 encode의 raw q42/velocity42/support6/non_support6/knee flexion6/knee velocity6, 원래 filtered241와 velocity_lowpass42, encode 후 last_features241/new filtered241/new velocity_lowpass42를 기록한다.
2. 원래 encode 반환 float32 drive의 sensory3,604 원소와 실제 `network.advance_readout`에 전달된 float32 drive의 selected4,046 원소를 복사한다. 전자는 이전 filtered로 생성된다. 새 filter는 다음 control에 사용된다. 원래 descending/navigation 덧셈을 보존하며, 네트워크에 전달된 값을 직접 기록한다.
3. 원래 `advance_readout`을 같은 drive/391MN IDs/20 ticks로 한 번 호출하고, 완료 뒤 selected4,046의 현재 rate를 별도로 읽기만 한다. CUDA graph/커널/원래 MN 반환값을 변경하거나 상태를 대입하지 않는다. row0은 시작 state의 실제 rate/drive/filters이며 encode 관측 유효 mask는 false이다. c>=1 rows는 post-neural/pre-physics rate다. rate[c]와 body trace[c]는 같은 시각 표기를 갖지만 body는 해당 명령의 50 native steps 이후임을 metadata에 명시한다.
4. E240의 기존 391MN 네 단계 전류·control 기록·trace·native contact·초기/최종 whole snapshot을 그대로 보존한다. 추가 시계열은 별도 `signal-raw.npz`와 metadata에 저장한다. 실행 중 record는 관측 메모리에만 추가하며 snapshot schema에 포함하지 않는다. 설치된 instance/module 메서드는 기존 instance binding 여부까지 finally 복원한다. 새 body/network constructor는 없다. 기존 E240 복원 6개 및 signed observer 1개 neural construction, body 1개(생성 중 2,000 settle physics 포함)를 그대로 계상한다.

구현은 새 `tools/observe_c_banc_m3_passive_signals.py`와 `tests/test_c_banc_passive_signals.py`만 소유한다. E240 및 기존 수집/상태/원래 신경 소스를 재사용하고 변경하지 않는다. 기존 `run_arm`을 scoped callback으로 감싸는 경우 callback 복원과 실패 경로를 테스트한다. source gate는 새 freeze와 부모 review의 SHA, 모든 실제 import/입력 경로를 검증하며 새 output만 허용한다. gate 우회나 실패를 성공으로 바꾸는 monkeypatch는 금지한다. 저장 raw부터 보존하고 최초 오류 및 후속 cleanup/storage 오류를 남긴다. 부분 rows와 completed flags를 구분하고 late publication/deadline failure는 최종 valid 판정을 취소한다.

## 유효성 및 결과별 다음 행동

- 설치 전후 initial whole scientific state는 E240 observed-start와 byte exact. final whole도 E240 final과 byte exact이며, 기존 `exact`처럼 측정 wall_s만 제외한다. 과학 state 필드를 추가로 제외하지 않는다.
- 전체 2,001행 MN391, q42, tendon8, adhesion, counter/time, body trace 전 행과 native-contact 배열은 E240과 exact. 기존 stage 전류 첫 160행(두 controls), ID 및 counter를 exact 비교한다. 전체 stage 파일의 해시/shape/finite/count도 남긴다.
- 추가 raw는 2,001행이며 control0 invalid, c>=1 encode/body/neural 관측 각 한 번, network tick=20c, 원래 physics 간격/시점 일치, 포트/선택 ID 일치, 정상 범위 및 유한성을 확인한다. MN 열 subset은 기존 MN 매 control 값과 byte exact. 마지막 필터/발화/drive는 최종 whole와 exact. 최초 raw/필터는 초기 whole와 exact. 원래 encode 호출을 통해 새 filtered를 기록하고 이전 filtered와 다음 row의 연결도 검증한다.
- 최초 두 controls 또는 도중 MN/명령/신체 불일치, 수치·접촉·전도 건강성 오류 시 즉시 중단하고 partial evidence를 보존한다. 전체 exact 실패는 H245 not_tested/자료 invalid, 해당 원인만 수정하는 별도 사전계획 없이 자동 재실행하지 않는다.
- 유효 자료면 H245 supported(관측 비침습성의 이 고정 실행 범위). 신호의 주기·지연·구별성 분석은 별도 사전 계약으로 진행한다. 자율/자연 보행, F2/F6/전체 목표는 미완료이고 후보 채택은 false다.

## 자원과 담당

부모 설계 → 실제 gpt-6-sol/high 구현 → 부모 FIRST → 필요하면 같은 Sol 첫 수정 → 부모 SECOND → 남은 수정은 부모 순서다. high 선택 이유는 GPU 판독/인코더 호출 순서/실패 복원/whole 재현을 함께 다루기 때문이다. 생성 fixture만으로 단일 호출·row 정렬·old filter 주입·반환 불변·bindings 복원·부분 실패·늦은 저장 실패를 확인한다. 테스트가 실제 원본 보행을 호출하면 안 된다.

E240 원본 감독 129.1397226초, output817,823,509B였다. 추가 raw 약 90MiB와 roster5.6MB, 전체 비교 비용에 여유를 두어 이번 단일 원본 internal170초(마무리15초 포함), external185초, supervisor200초, CPU1/GPU1, RSS6GiB, 추가 GPU4GiB, output2GiB, 여유 disk3GiB를 동결한다. fit/solver/retry0, 원본 process1, 10 model초이며 기존 개별300벽초 이내다. 구현/검토 후 source SHA가 확정되기 전 원본을 실행하지 않는다.

기존 보수 원본 누적 6596.09548459962 /6755초를 유지하고 이번 미래 배정만 +45초, 상한6800초로 정한다. 실행 전 잔여203.90451540037975초에서 supervisor200초를 예약한다. 사용자 지정 전체 예산이 아니라 기존 승인된 로컬 단계 회계이며 과거 실패·비용은 지우지 않는다. 원본 비용은 실제 감독 후 더한다. 메타데이터 조회 1.2677531초 및 roster 구성1.5817304초는 원본 모델 실행이 아닌 별도 정적 준비비이고, Sol advisory/development 미측정 비용은 null이다. GPU/UI 경합 상태는 실행 직전 확인한다. 결과에 따라 비용을 소급 늘리지 않는다.
