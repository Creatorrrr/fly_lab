# E267 결과 — source별 시냅스 여과 런타임 구현·CUDA 검증

기존 premotor→MN 입력을 합산하기 전에 신경원별 활동 이력을 적용하는 `SourceFilteredSynapseNetwork`를 구현했다. 이 구현은 실제 전체 BANC/CUDA에 이식 가능하고, 중립 계수에서 원래 신경 계산과 저장 미래가 정확히 유지된다. **계수 학습과 보행 후보 검증은 아직 하지 않았다.** F1–F8과 자연스러운 보행은 미완료다.

고정 내부51개에서 free343MN으로 향하는 기존 nonzero1660간선(양수1091/음수569)을 사용하며336MN에 도달한다. E247의 동적 두 단계 경로330MN과는 집계 정의가 다르다. 새 연결·부호 반전·외부 위상·교사 명령은 넣지 않았다. 5/40ms source 이력과 간선별 비음수 세 계수는 공학적 동역학 가설이다.

실제 gpt-6-sol/high 구현 뒤 부모 FIRST에서 CUDA inference tensor 할당 오류와 free MN roster 검증 누락을 찾았다. 같은 Sol의 첫 수정 후 부모 SECOND에서 생성 CUDA2테스트, 별도6뉴런 재귀망의 독립 coupled RK4 30검사, 상태 차단·손상 복원 무변경·저장 미래를 통과했다. 최종 Sol CPU23테스트/Ruff도 통과했다. 부모 제품 잔여 수정0이다. 첫 CUDA 실패는 scientific advance 전 발생했으며 원본 BANC 실행이 아니었다.

실제 RTX4080, Torch2.14.0+cu130/CUDA13.0 packed graph에서 전체158706뉴런/11584852간선의 원본을 한 번 실행했다. E265 B와 같은 시작·drive를 사용한0.5초/101endpoint에서 기존1402개 rate/adaptation/drive 및 전체 drive SHA가 exact였고, 원래 전체 신경 시작·끝 상태도 exact였다. 새51×2 이력은 실제로 변했고 checkpoint에 포함된다. Disk복원 후 같은.005초 미래를 반복한 전체 상태도 exact였다. 원본520검사에 더해 부모는 작성자 판정보다 먼저 raw·전체 상태·실제 소스·manifest를 읽어198검사를 통과했다.

모형 identity는 `9d4db90c1442186f2a499c56c9205cd628a3970096c0606a8a4faf5e71001e45`, 원래 모델은 `a31f899cd948a4297a73db66569e5edc2210ac9aa3601aa83e96cb9ff68ce244`다. 이 검증은 중립 런타임 검증이며 비중립 후보의 실제 보행이나 최종 F2의 신체 포함 저장/차단을 대체하지 않는다.

원본 supervisor8.002975900초/내부7.043056초, neural2040ticks(기준2000+두 미래20씩), constructor1, 별도 warmup40ticks/capture기록20ticks였다. 실제 관측한 setup 세 interval을 원자료에 남겼다. body/physics/fit/solver/retry0, 소유PID 종료, 최대 tree RSS1492033536B, Torch peak allocated251426304B다. 100회 advance+동기화 합1.9388015초는 관측·초기화를 제외하며 별도 대조군과 맞춘 속도 향상 수치가 아니다.

원래 누적7126.338238699598초를 유지한 채 사전에 단계 배정7138→7182초(+44)를 기록했고, 실제 사용 후 누적7134.341214599615/7182초·잔여47.658785400385초다. 부모 raw 감사.2650405초(프로세스.7800276초), FIRST 생성CUDA 실패3.2094806초, SECOND 생성CUDA 두 프로세스 합감독6.773873초는 별도다. 생성 CPU·정적 준비·미측정 개발 시간은 별도 영수증/미측정으로 유지하며0으로 간주하지 않는다.

다음은 기존 native stage 기록과 학습기의 재사용 가능 부분을 확인하여 source별 계수의 학습을 고정한다. 조건부 교사 적합만으로 후보를 채택하지 않고, 실제 재귀망과 신체의.5초 startup→1.5초 전진·slip·연속 bout/주기→원래10초 F3–F5를 연결한다. E150/M2/UI/F6와 모든 이전 실패를 보존한다.

근거: `verification/banc-natural-gait-m3-20260919/e267-native-source-filter-neutral/`, `parent-e267-second-review.json`, `parent-e267-output-review.json`, `parent-e267-native-supervisor.json`. 구현: `tools/banc_m3_source_filtered_synapses.py`, 테스트: `tests/test_c_banc_source_filtered_synapses.py`.
