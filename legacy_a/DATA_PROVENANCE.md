# 데이터 출처와 재현

## 고정 원자료

- 저장소: https://github.com/aplbrain/seismic
- 프로젝트: State Estimation/Integration of Sensors from a Miniature Insect Connectome (SEISMIC)
- 커밋: `1a789eefc2ca03c1f79e283069e41d20d12ed91f`
- 파일: `neuroaiengines/networks/hemibrain_conn_df_both.csv`
- 원본 Git blob SHA-1: `9067c6cf66677c6f3bdda69daaf39385209ce620`
- 고정 파일 주소: https://github.com/aplbrain/seismic/blob/1a789eefc2ca03c1f79e283069e41d20d12ed91f/neuroaiengines/networks/hemibrain_conn_df_both.csv
- 원본 스크립트 `neuroaiengines/networks/pull_data.py`의 데이터셋 표기: `hemibrain:v1.2.1`.

저장소의 README는 시냅스 수준 hemibrain 자료로 방향 추정 RNN을 구성·최적화하는 연구라고 설명한다. A는 그 **공개 연결 표 일부만 사용**하고 학습 코드·체크포인트·훈련 가중치는 사용하지 않는다.

## 이번 발췌

원본 파일을 GitHub 읽기 도구로 확인하고 다음 1-based 파일 행 구간을 가져왔다. 헤더는 1행이고 데이터 행만 아래에 포함된다.

```
2–65
1000–1063
1500–1563
```

총 192개 ROI 행. 원본의 이름 없는 첫 열은 `source_record`로 명명했고 수치·세포 주석은 유지했다. 이것은 회로 기능에 따라 선정한 완전한 표본이 아니라, 실행 가능한 데이터 기반 프로토타입을 위한 **편의상 연속 행 발췌**다.

`source_record`는 원본 CSV의 dataframe 인덱스이고 파일 행 번호와 같지 않다. 데이터를 검증할 때 이 둘을 혼동하지 않는다.

같은 `(bodyId_pre,bodyId_post)`의 서로 다른 ROI 행은 합산한다. 방향을 유지하고 자기 연결도 보존한다. 노드 ID는 `hemibrain:<bodyId>` 문자열로 보관한다. 원본 bodyId, type, instance, hemisphere, index, index_fix 주석도 유지한다.

계수는 노드 98, 뉴런 쌍 177, count 합 3747이다. 원본 ROI는 EB/PB다. 필터링된 원표의 모든 행이 count>=4라고 가정하지 않는다. 일부 ROI 행에는 1~3이 들어 있다.

## 파일

`data/hemibrain_excerpt.csv`의 SHA-256:

```
5da036a89b66fdfa5bb56695075fc3d3c95b71d19bc5a635f9bdfb491e895be8
```

`tools/prepare_data.py`는 원본 주석의 중복 일관성·행/노드/연결 수를 검증하고 `data/circuit.json`과 `data/provenance.json`을 만든다. JSON의 각 연결에는 원본 source_record 목록과 ROI별 값이 남는다.

제작 환경에서는 원본 전체 CSV 바이트 다운로드가 제한되어, **원본 전체 Git blob 해시를 로컬에서 재검증한 것은 아니다.** 고정된 GitHub 파일의 지정 구간을 읽어 사용했고, 로컬 발췌의 해시·내부 집계는 확인했다. 이미 받은 원본 전체 CSV가 있다면 다음 명령으로 추가 대조할 수 있다.

```sh
python3 tools/verify_source.py --input /path/to/hemibrain_conn_df_both.csv
```

이 도구는 원본 Git blob 해시와 선택 행 값을 모두 대조하며 네트워크를 사용하지 않는다. 전체 원본 입력을 사용한 성공 경로는 이번 제작 환경에서 검증하지 못했으며, 도구의 내부 비교 로직과 로컬 해시는 단위 검사한다.

## 해석 금지

발췌에 없는 연결이 실제 뇌에도 없다고 말하지 않는다. 98개 ID의 전체 유도 그래프라고 말하지 않는다. 일부 실제 연결이 있다는 이유로 모든 추가 모델 노드를 실제 뉴런으로 표시하지 않는다. 모든 연결을 흥분성으로 둔 것은 A의 가정이며 원자료가 제공한 생리학적 부호가 아니다.

실제 전체 뇌 데이터가 필요하면 B/C 작업에서 별도로 공개 데이터 버전·개체·영역·필터·신호 모델을 고정해야 한다. A의 편향 발췌를 전뇌 표준 모델로 확대해석하지 않는다.

## 관련 논문

방향과 목표를 조향으로 연결하는 생물학적 연구:
https://www.nature.com/articles/s41586-024-07039-2

A의 PFL3-L/R 노드는 이 계산 아이디어를 참고한 **설계 readout**이다. 논문의 특정 생물학적 회로나 피팅된 PFL3 모델을 그대로 복제했다고 주장하지 않는다.

상류 라이선스와 저작권 표시는 `THIRD_PARTY_NOTICES.txt`에 있다. 이 앱은 SEISMIC, Janelia, FlyWire, NeuroMechFly의 공식 배포물이 아니다.
