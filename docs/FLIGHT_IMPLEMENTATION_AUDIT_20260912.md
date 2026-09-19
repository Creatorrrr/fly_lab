**FlyLab 날개·비행 구현 조사 — 2026-09-12**

날개 관절 제어와 별도 공기역학 실험은 구현되어 있다. 일반 FlyLab 화면의 비행 기능, 안정적인 비행·이착륙 정책, BANC 신경회로에 의한 비행은 완성되어 있지 않다. 아래 결론은 현재 코드, 실제 컴파일된 모델, 이번에 재실행한 물리 실험, FlyGym/FlyBody 공식 자료를 구분해 정리했다.

조사 환경은 Windows의 FlyLab 자체 `.venv`이며 FlyGym 2.1.0 (`ca65a510c2afe6ac61c51df4f274c8d190c2f95f`), MuJoCo 3.9.0, dm-control 1.0.39다. 별도 FlyBody 소스는 `verification/flygym-completion-20260910/flybody-source`의 깨끗한 고정 커밋 `d015e9bfe441bd90ae431bac24c55cb74bdbce26`을 사용했다. 이 조사에서는 기능 구현 코드를 변경하거나 학습된 정책을 설치하지 않았다.

| 현재 실행 경로 | 날개 운동 | 비행 처리 |
|---|---|---|
| 웹의 도식적 몸 렌더러 | 고정된 다각형 좌표로 날개 표시 | 날개 관절값을 반영하는 애니메이션 없음 |
| 기본 NeuroMechFly 보행 몸 | 구성된 모델의 날개 관절·구동기 0개 | 보행 전용 |
| FlyBody 다리 전용 몸 | 구성된 모델의 날개 관절·구동기 0개 | 보행 전용 |
| FlyBody 전신 몸 | 좌우 각각 yaw/roll/pitch, 총 6개 위치 서보 | 관절 목표 제어 가능. 비행 조종기 없음 |
| `FlightRig` | 218 Hz 기본 패턴, 진폭·좌우 비대칭 명령 | 원본 FlyBody XML의 날개와 MuJoCo ellipsoid 공기력. 지면 접촉을 끈 단기 실험 |
| `make_flight_env` | 원본 WingBeatPatternGenerator와 호출자가 주는 보정 행동 | flight/takeoff/landing 자체 과제, 지면 접촉·다리·보상·성공 조건. 기본 실행에는 학습된 정책 없음 |

근거 코드:

- [도식적 날개 렌더링](C:/Users/ckthd/Dev/Projects/fly_lab/src/ui/world-view.js:50)
- [전신 관절과 구동기 구성](C:/Users/ckthd/Dev/Projects/fly_lab/flylab/body_options.py:79)
- [전신 직접 관절 명령 API](C:/Users/ckthd/Dev/Projects/fly_lab/flylab/body.py:467)
- [별도 FlightRig](C:/Users/ckthd/Dev/Projects/fly_lab/flylab/c/flight.py:13)
- [원본 Flying을 상속한 자체 FlightTask](C:/Users/ckthd/Dev/Projects/fly_lab/flylab/c/flybody_flight.py:42)
- [C 서버의 flight=False](C:/Users/ckthd/Dev/Projects/fly_lab/flylab/c/server.py:104)

`flylab/brain.py`와 `src/core/brain.js`에는 과거 `mode == 'flight'`에 대한 수직 속도 출력 분기가 남아 있다. 그러나 B의 `config_values()`는 walk 이외의 모드를 거부하며, 실제 비행 물리에 연결되지 않는다. 이 문자열만으로 비행 지원으로 분류할 수 없다. 별도 비행 모듈은 일반 C 엔진·서버·화면에서 호출되지 않고 CLI 실험 도구에서 사용한다.

이번 전신 관절 실험에서는 날개 6축 모두 현재 각도에서 관절 범위 안쪽으로 약 0.1 rad 이동하도록 명령했다. 50 ms 뒤 최대 절대 오차는 0.000957 rad였다. 이는 작은 목표 변화의 위치 추종 근거이며 218 Hz 날갯짓이나 양력 발생을 검증한 결과는 아니다. 최초 탐침의 범위 바깥 명령은 API에서 거부되었고, 유효 범위 안으로 탐침 명령을 수정해 측정했다.

**FlyGym과 원본 FlyBody의 차이**

FlyGym의 실험적 `FlyBody` 클래스는 원본 몸을 FlyGym의 장면 구성·관절·서보 API에 맞게 변환한다. 공식 문서는 좌우 날개 3축을 제공한다고 설명하지만, 이것이 비행 제어기 제공을 의미하지는 않는다. 공개 튜토리얼의 FlyBody 예제는 기록 재생과 보행이다. [FlyGym FlyBody API](https://neuromechfly.org/api_reference/flygym/compose/fly/flybody/), [공식 튜토리얼 목록](https://neuromechfly.org/)

변환 코드 `_is_excluded_geom()`은 이름에 `fluid`가 들어간 원본 형상을 제외한다. 호출부에는 `We do not care about flight`라는 설명도 있다. 설치된 모델의 날개에는 시각 메시가 있지만 원본 `wing_left_fluid`, `wing_right_fluid`가 없다. 실제 컴파일 결과에서도 전신·다리 전용 FlyBody 모두 `geom_fluid[:,0]`의 활성 항목이 0개였다. [공식 변환 소스](https://github.com/NeLy-EPFL/flygym/blob/ca65a510c2afe6ac61c51df4f274c8d190c2f95f/src/flygym/flybody/parse_flybody.py#L151)

다만 FlyGym FlyBody의 전역 density=0.00128, viscosity=0.000185는 남아 있다. 따라서 ‘유체 관련 효과가 전혀 없다’고 단정하는 것은 부정확하다. 확인한 누락은 원본 비행용 날개 ellipsoid 형상과 이를 사용하는 비행 과제·제어 경로다. `pose/flight/yaw_roll_pitch.yaml`도 정적 관절각 파일이며 시간에 따른 날갯짓 생성기가 아니다. `_correct_wing_default_pose()`는 위치 서보를 위한 날개 휴지 자세 보정이다.

원본 TuragaLab FlyBody의 `Flying`은 날개 ellipsoid 형상을 활성화하고 공기력 계수, 날개 관절 강성·감쇠, 구동기 게인을 설정한다. 기본 비행의 물리 간격은 50 µs, 제어 간격은 200 µs이며 기본 날갯짓은 218 Hz다. 원본은 cm, g, s를 사용하고 FlyGym은 길이를 mm로 변환하므로, 물리·관절 파라미터와 축 정의를 검토 없이 그대로 옮기면 안 된다. [Flying 구현](https://github.com/TuragaLab/flybody/blob/d015e9bfe441bd90ae431bac24c55cb74bdbce26/flybody/tasks/base.py#L269), [상수](https://github.com/TuragaLab/flybody/blob/d015e9bfe441bd90ae431bac24c55cb74bdbce26/flybody/tasks/constants.py)

원본 연구의 비행 제어는 기본 날갯짓 패턴에 학습된 MLP의 조종 출력을 더하는 구조다. 날개 운동으로 생긴 힘과 토크를 MuJoCo가 적분하여 몸이 이동한다. 유체 모델은 타원체를 이용한 근사 모델이며, 날개 주변 전체 유동장을 계산하는 CFD는 아니다. 실측 비행 궤적 추종, 급회전, 회피 기동, 수평 비행 사례가 공개되어 있다. 이 정책은 강화학습 신경망이며 FlyWire/BANC 신경회로 자체의 비행을 입증한 것은 아니다. [2025 Nature 원논문 및 보충 영상](https://www.nature.com/articles/s41586-025-09029-4)

**이번 물리 재실행 결과**

| 검사 | 결과 | 해석 |
|---|---|---|
| FlightRig 유체 off/on 대조 | 최종 위치 차이 0.281321 mm, 복원·자산 위치 변경 후 복원 동일 | 짧은 날개 운동에서 유체 효과가 실제 계산됨 |
| flight | 0.3초, 1,500 제어, 예외 없음, 종료 success=false | 안정적인 목표 고도 유지 실패 |
| takeoff | 0.3초, 1,500 제어, 예외 없음, 종료 success=false | 일시적 성공 조건 충족 후 자세·상태 유지 실패 |
| landing | 0.3초, 1,500 제어, 예외 없음, 종료 success=false | 제어된 착륙 성공 근거 없음 |

FlightRig 대조는 20 ms 기록 이후 복원 검사용 1 ms 추가 적분을 포함한 최종 위치를 비교한다. 그 결과는 지속 비행이나 이착륙 검증이 아니다.

takeoff의 순간별 success는 약 0.1472~0.1556초에서 true였지만 마지막에는 false였고, 종료 시 upright는 -0.399였다. 따라서 ‘전 구간에서 한 번도 이륙 조건을 만족하지 않았다’고 해석하면 안 된다. flight와 landing은 초기 상태와 0 보정 입력이 같아 동일한 물리 궤적을 보였다. 이름과 보상 목표만으로 다른 조종 행동이 생기지 않는다는 점을 보여준다. 이 세 과제는 공식 `Flying` 기반으로 FlyLab이 작성한 실험이며, 공식적으로 학습·검증된 이착륙 정책을 가져온 것이 아니다.

재현 명령은 PowerShell에서 프로젝트 루트를 작업 디렉터리로 사용한다. 출력 경로는 아직 존재하지 않는 새 경로를 지정한다.

```powershell
.\.venv\Scripts\python.exe tools\run_c_flight_tasks.py --source verification\flygym-completion-20260910\flybody-source --seconds .3 --out verification\flight-audit-repeat\tasks
.\.venv\Scripts\python.exe tools\verify_c_flight.py --assets verification\flygym-completion-20260910\flybody-source\flybody\fruitfly\assets --out verification\flight-audit-repeat\fluid
```

이번 원자료:

- [컴파일 모델과 날개 목표 추종](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flight-audit-20260912/model-probe.json)
- [공기력 대조 및 복원 결과](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flight-audit-20260912/fluid/report.json)
- [세 과제 실행 보고](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flight-audit-20260912/tasks/report.json)
- [궤적·순간 성공 판정 요약](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flight-audit-20260912/trace-summary.json)
- [공식 환경과 자체 과제의 행동 차원 비교](C:/Users/ckthd/Dev/Projects/fly_lab/verification/flight-audit-20260912/action-compatibility.json)

**재사용할 수 있는 공개 비행 예제**

| 예제 | 제공되는 행동 | 코드 |
|---|---|---|
| Flight imitation | 실제 비행의 무게중심 위치·몸 자세 궤적 추종 | [fly-env-examples.ipynb](https://github.com/TuragaLab/flybody/blob/main/docs/fly-env-examples.ipynb)의 `flight_imitation()` |
| Vision flight — bumps | 눈의 영상으로 울퉁불퉁한 지형 위의 높이를 조절 | 같은 노트북의 `vision_guided_flight(..., bumps_or_trench='bumps')` |
| Vision flight — trench | 굽은 통로를 벽에 충돌하지 않고 통과 | 같은 노트북의 `vision_guided_flight(..., bumps_or_trench='trench')` |
| 비행 제어기 재사용 | 학습된 하위 비행 제어기를 고정하고 상위 시각 조종기를 결합 | [controller-reuse-vision-flight.ipynb](https://github.com/TuragaLab/flybody/blob/main/docs/controller-reuse-vision-flight.ipynb) |

공식 Figshare의 파일 메타데이터도 조회했다. `trained-fly-policies.zip` 약 6.54 MB, `datasets_flight-imitation.zip` 약 12.88 MB, `flight-controller-reuse-checkpoints.zip` 약 31.98 MB가 공개되어 있다. 비행 노트북은 실측 기본 패턴 `wing_pattern_fmech.npy`와 TensorFlow 학습 정책을 로드한다. 이번 조사는 파일 공개와 예제 코드를 확인한 범위이며 학습 정책의 로컬 실행은 수행하지 않았다. [공식 데이터 배포처](https://doi.org/10.25378/janelia.25309105), [조회한 공개 메타데이터 API](https://api.figshare.com/v2/articles/25309105)

후속 구현은 별도 원본 FlyBody 환경에서 공식 패턴·정책으로 비행을 재현하는 것이 우선이다. 현재 자체 과제는 다리까지 활성화하여 행동 차원이 66이고, 공식 기본 `flight_imitation()`은 12다. 관측 배열, 입력 범위 정규화, 관절 필터, 좌표·시간 단위를 맞추지 않고 학습 정책을 그대로 연결할 수 없다. 재현 후 상위 목표·조향 제어와 화면 렌더링을 연결하고, 지면 접촉·다리 동작을 포함한 이착륙은 별도의 정책과 성공 기준으로 검증해야 한다. BANC에 연결하는 단계에서도 학습된 비행 제어기를 쓰는 것과 BANC가 직접 날개를 제어하는 것은 구분해야 한다.
