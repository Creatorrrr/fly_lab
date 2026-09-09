# C 0.4 연구 확장 실행법과 경계

다음 확장은 기본 관찰 화면의 신경 모델이나 보행 제어기를 자동 교체하지 않는다. 실제 데이터 또는 실제 물리로 실행한 단계와 아직 연결·보정되지 않은 단계를 구별한다. 현재 기본 화면은 FAFB 뇌 LIF → 하행 출력 해석 → 공학적 CPG → 42개 관절 물리다.

## X1: 세포별 매개변수, 시냅스 지연·효과, 전기 결합, 가소성

`flylab/c/research_neural.py`는 CPU와 MPS에서 별도 실행하는 ResearchLIF다. 그래프의 모든 포함 뉴런을 계산하고, 명시적 physiology profile의 대상에만 다음 변경을 적용한다.

- 세포별 rest/reset/threshold/tau_m/tau_syn.
- 실제로 존재하는 화학 연결에만 연결별 지연과 수용체 효과로 해석한 mV 가중치 override.
- 근거 또는 가정이 기록된 두 뉴런 사이의 양방향 전압차 결합.
- 선택한 기존 비영 가중치에 대한 reward-gated STDP. 부호를 뒤집지 않고 크기 상한을 지키며 보상 0에서는 가중치를 갱신하지 않는다.
- 지연 큐·전압·발화율·불응기·가중치·학습 trace를 포함하는 독립 체크포인트.

상위 profile에는 `schema=flylab.physiology.v1`, 정확한 `graph_hash`, `evidence`, `uncertainty`가 필수다. 선택적 `cells`, `synapses`, `gap_junctions`, `plasticity` 항목도 근거를 요구한다. 같은 이름만으로 다른 표본의 뉴런을 합치지 않는다. 미확정/신경조절성 연결을 자동 흥분성으로 바꾸지 않는다.

```sh
.venv/bin/python tools/run_c_research_neural.py \
  --graph data/fafb783/bundle --profile my-physiology.json \
  --inputs my-input-sequence.json --device mps --out verification/my-physiology-trial
```

입력 JSON은 `capture_ids`와 `segments`로 구성한다. 각 segment에는 `ticks`, `stimuli=[{ids, amplitude_mV}]`, `reward`를 둔다. 보상은 -1..1의 스칼라이며 위치·목표 방향 명령을 출력하지 않는다. 이 도구는 신경 실험이며 몸이나 먹이의 정답으로 보상을 자동 생성하지 않는다.

현재 검증은 작은 합성 회로의 CPU/MPS 비교, 정확한 상태 복원, 기본 LIF와의 발화 비교, reward=1/0에서 선택 가중치만 달라지는 검사다. 실제 전뇌의 수용체·gap junction·세포 매개변수 보정 자료를 모두 확보한 상태가 아니다. 다중 구획 형태학 모델, 학습된 자연 행동, 생물학적 기억 재현을 완료한 것으로 표시하지 않는다. 연구용 edge별 Torch 경로는 기본 Metal 희소 엔진과 다른 구현이며 전뇌 장시간 성능을 승인하지 않았다.

입력 가설 프로파일 `bindings-odor-poisson-current-hypothesis-v1.json`도 별도 제공한다. 150 Hz/unit과 68.75 mV의 Poisson **synaptic-drive h** 펄스를 사용한다. [Shiu 연구 코드](https://github.com/philshiu/Drosophila_brain_model/blob/main/model.py)는 직접 전압 v에 Poisson 입력을 넣고 대상 불응기를 제거하므로 이 프로젝트와 정확히 같은 방정식이 아니다. 그 차이를 프로파일에 기록했고 원 논문의 재현이라고 부르지 않는다. 이번 가설 검사에서도 자연 후각에 따른 전진 출력은 0이었다.

## X2: 같은 표본의 BANC 뇌·VNC와 선택 운동뉴런

[2026 BANC 원 논문](https://www.nature.com/articles/s41586-026-10735-w)이 공개한 v888 자료를 이용했다. annotation은 갱신될 수 있으므로 공개 GCS 객체의 generation과 SHA256을 고정했다. 메타데이터의 현재 `root_id`가 아닌 스냅샷의 **root_888**을 사용한다.

```sh
.venv/bin/python -m pip install -r requirements-data.txt
.venv/bin/python tools/acquire_c_research_data.py banc --destination data/acquisitions/my-banc/raw
.venv/bin/python tools/build_c_banc.py --raw data/acquisitions/my-banc/raw --out data/acquisitions/my-banc/bundle
```

이번 자료 경로는 `data/acquisitions/banc888-v2-20260909/`다. `bundle/`에는 158,706개 선언된 뉴런, 11,584,852개 쌍 연결, 35,457,154개 접촉을 사용했다. 세포 분류 또는 proofread/roughly proofread 근거가 있는 neuronal roster를 선택하고 명시적인 glia/non-neuronal/trachea를 제외했다. 분류 불확실한 객체를 조용히 뉴런으로 채우지 않았다. 따라서 scope는 `declared_neuronal_roster`이고 **fullBrain=false**다. FAFB와 합치지 않는다. BANC도 lamina·ocellar ganglion 등을 완전히 포함하지 않으며 모든 말초 신경·근육의 복원은 아니다.

`flylab/c/cns.py`는 실제 `tibia_flexor_muscle`/`tibia_extensor_muscle` 운동뉴런 주석을 여섯 다리의 tibia pitch에 연결한다. 해당 실험에서는 CPG를 우회하고 실제 MuJoCo position actuator를 사용한다. 나머지 36개 다리 자유도는 중립 목표로 유지한다. rate → 각도, gain, 필터와 상한은 공학적 변환이며 근육 힘-길이/속도 곡선을 재현하지 않는다.

```sh
.venv/bin/python tools/verify_c_cns.py \
  --graph data/acquisitions/banc888-v2-20260909/bundle \
  --out verification/my-cns-test
```

새 테스트는 무자극, 왼앞다리 flexor/extensor 자극, 운동 연결 차단, BANC DNp09 직접 자극을 실행한다. 각 조건은 모든 158,706개 상태를 계산한다. 왼앞다리의 실제 세 분절로 계산한 무릎 내부 각도가 flexor에서 감소, extensor에서 증가했고, 운동 연결 차단은 무자극과 같았다. neural/body/actuator 체크포인트에서 20ms 이어 계산한 상태도 일치했다. DNp09 자극은 일부 tibia 목표에 전달됐으나 자율 보행 판정은 하지 않는다.

아직 미연결인 부분: 다른 관절·근육, 다리 고유수용성 감각의 검토된 부호/좌우/지연, 신경 회로로 생성하는 보행 주기, 지지·조향·외력·장애물의 폐루프. 이 도구가 몸의 일부를 구동한다고 해서 CPG 전체가 대체된 것은 아니다.

## X3: 가상 섭취와 내부 에너지

화면에서 **새 실험에 가상 섭취·에너지 적용**을 선택하면 해당 실험에 모델을 붙인다. 기본값은 끔이다. 식품 냄새원의 위치가 입 대용점 반경 0.5mm 안이고 이동 속도가 충분히 작을 때 가상 먹이량을 소비한다. 에너지 용량, 초기 에너지, 기초 소비, 이동 소비, 섭취 속도는 `MetabolicParameters`의 명시적 공학 단위다.

섭취 ledger, 누적 소비·섭취, energy/hunger, clock을 저장·복원하고 에너지 보존식과 먹이 수량 일치를 검사한다. 기록에 섭취 이벤트와 매개변수 hash를 남긴다. 같은 초기 신경/몸 상태에서 이 모델을 켜거나 꺼도 현재 운동 출력은 같아야 한다.

입·인두 근육이나 실제 섭식 동작을 재현하지 않았고 hunger를 임의의 뉴런에 넣지 않았다. 따라서 화면은 **신경 조절 미연결**로 표시한다. ResearchLIF의 가소성 계산 기능과 이 에너지 기능은 아직 학습 행동을 검증한 하나의 폐루프로 통합되지 않았다.

## X4: 실제 날개 관절과 공기력 시험 장치

[공개 flybody](https://github.com/TuragaLab/flybody)의 commit을 고정하고 원본 모델·날개 mesh·fluid ellipsoid 설정을 이용하는 `FlightRig`를 추가했다. 출처와 commit은 `data/releases/flybody-source.json`에 있다.

```sh
.venv/bin/python tools/acquire_c_research_data.py flybody --destination data/acquisitions/my-flybody
.venv/bin/python tools/verify_c_flight.py \
  --assets data/acquisitions/my-flybody/flybody/fruitfly/assets \
  --out verification/my-flight-rig
```

이 rig는 실제 날개 joint actuator와 MuJoCo 유체력을 계산한다. 몸통 위치를 매 tick 덮어쓰지 않는다. 길이/질량/시간 단위는 상류 모델의 cm/g/s를 따르고 화면·보고서 위치는 mm로 변환한다. 218Hz의 주기적 시범 입력은 신경망 출력이나 학습된 제어가 아니다. 충돌은 끈 공중 시험이며 이륙·착륙은 구현하지 않는다.

유체력 켬/끔의 20ms 실험에서 몸 위치가 달라졌고 MuJoCo integration state의 복원 후 결과가 정확히 일치했다. 그러나 두 조건 모두 초기 높이를 유지하지 못했다. **안정 비행 PASS가 아니며 기본 앱의 flight 지원 표시는 계속 false**다. 비행 운동 회로·감각/평형 피드백·이륙·착륙·지속 비행은 남은 연구 단계다.

최종 `flight-model.v2` 신원에는 파일 경로 대신 XML 설정과 mesh 내용 해시를 사용한다. 실제 자산을 다른 디렉터리에 복사한 뒤 체크포인트를 복원해 1ms 이어 계산한 적분 상태가 동일했다. 초기 경로 의존 모델의 실험은 `flight_rig_v1`에 별도로 보존했다.
