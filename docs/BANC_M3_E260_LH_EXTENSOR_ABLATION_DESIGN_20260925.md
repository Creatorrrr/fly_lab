# E260 고정 감각 아래 무릎 신전근 억제 연결 대조

E259에서 muscle44 감소의 96.7876%는 MN47329, 이 MN의 관측 bridge 억제 증가 99.99557%는 111970에 집중됐다. 연결 하나를 없앴을 때의 조건부 신경 반응을 조사한다. 이 연결은 실제 CSR entry3473458, pre111970→post47329, weight float32(−0.8800315856933594)다. 대상은 E259 결과에 따른 이번 설계에서 고정하며 실행 중 바꾸지 않는다.

## 비교와 재현 조건

W0는 E257 baseline CSR, W6는 동일 실험의 여섯 감각→111970 간선 복원 CSR이다. U0/U6는 각 실험의 저장 old_filtered에서 원래 순서로 만든 감각·descending·navigation drive다. 같은 control0 신경 상태와 muscle44의 저장 bank에서 0.5초(100controls×20tick)를 진행한다.

| arm | CSR | 저장 감각 | 역할 |
| --- | --- | --- | --- |
| B | W0 | U0 | baseline 101행 재현 |
| C | W6 | U6 | candidate 101행 재현 |
| X | W6 | U0 | 같은 입력에서 연결 복원 효과 |
| Bcut | W0의 entry3473458만 +0.0f32 | U0 | baseline 연결 제거 |
| Xcut | W6의 같은 entry만 +0.0f32 | U0 | 복원망 연결 제거 |

모델 순서는 W0(B), W6(C→자체 시작 상태 exact reset→X), W0cut(Bcut), W6cut(Xcut)이다. 모델4개는 순차 생성·해제하며 동시에 한 모델만 유지한다. 원래 모델은 원래 identity 그대로 복원한다. cut 모델은 E250의 검증된 metadata transplant만 허용하고 모든 동적 배열/dt/tick/mask/비대상 신경 파라미터를 원래 시작과 exact로 유지한다. 전체 CSR 구조와 비대상 weight는 exact여야 한다. W6−W0의 여섯 차이도 그대로다. 제거한 slot을 CSR에서 삭제하지 않으며 새 간선·gain·tau·output mask를 추가하지 않는다.

B/C 각각의 rate391, 기존 관측82 rate/drive/adaptation, core24 rate/drive/adaptation은 E257 처음101행과 bitexact여야 한다. 생성 drive는 저장 old_filtered→float32 scatter→descending→navigation 순서 및 returned_source_drive와 exact여야 한다. 두 diagonal 재현이 모두 통과해야 가설을 판정한다. 재생 마지막 control100 전체망의 과거 reference는 없으므로 전체망 마지막 exact를 주장하지 않는다. 모든 arm의 전체 시작·끝 신경 snapshot과 실제 model identity/packed CUDA CSR 검증을 저장한다. 관측은 상태를 변경하지 않아야 한다.

## 신호와 가설

전체 incoming은 muscle44의 두 MN31172/47329에 대해 기록한다. 2MN incoming의 pre 합집합과 기존440 관측의 합집합을 endpoint마다 읽는다. 전체 incoming current64, 실제 packed CUDA current32, rate/drive/adaptation 및 target threshold를 보존한다. bridge111970뿐 아니라 미관측이었던 입력도 포함된다. bank는 저장 control0의 두 MN×3 bank에서 시작하며 alpha=1−exp(−.005/tau) 및 bank+=alpha*(endpoint_rate−bank)의 원래 갱신 순서를 따른다. muscle44 activation은 기존 low*bank+(high−low)*max(bank−kappa,0) 합이다. B/C 101행 bank는 exact, activation44는 1e−10 이내로 E257 재현해야 한다. row0은 갱신하지 않는다. 근육 변환 계수는 수정하지 않는다.

Y는 muscle44 활성이다. startup[1,100)에서 T=sum(Y_C−Y_B), C0=sum(Y_X−Y_B), Ccut=sum(Y_Xcut−Y_Bcut)를 사용한다. R=C0−Ccut와 각 arm 대비 차이, signed/positive/negative 합 및 controls56/70 사례를 보존한다. 원래 입력에서 연결 효과가 유지되는 조건은 T<0, C0<0, abs(C0−T)≤.2*abs(T)다. 제거 후 감쇠 조건은 abs(Ccut)≤.2*abs(C0)다.

`H260_existing_edge_attenuation`은 위 두 조건과 두 cut arm의 MN47329 startup rate가 각각 완전히 상수이지 않음(ptp>0)을 모두 만족할 때만 지지한다. 분모0은 unavailable/미지지이며 부호 반전도 숨기지 않는다. 20%는 E255에서 사용한 개발용 차이 보존 범위이며 생물학적 기준이 아니다. MN47329 및 다른 MN31172의 각 arm rate min/max, 정확한 cap200/zero 비율, range, 평균과 cut 전후 tonic 변화도 보고한다. ptp=0/상한·바닥 고정에 의해 사라진 차이는 지지로 채택하지 않는다. 부분적인 clipping 비율이 커진 경우도 보고서에서 해석을 제한한다.

같은 분해를 MN2 rate/drive/−adaptation/actual current/current64, 개별 incoming current64 및 bank2×3에 저장한다. 기준망에 연결을 제거하면 동작점 자체가 달라지므로 C0−Ccut를 유일한 생물학적 매개율로 해석하지 않는다. 이것은 고정 입력과 인위적 연결 제거 조건에서의 신경 진단이다. 신체 폐루프, 보행 개선, 연결 삭제의 생물학적 정당성 또는 새 후보 채택을 검증하지 않는다. 지지 여부에 따라 전체 입력/포화/감각 인터페이스 교정을 검토하되 임의 gain 탐색이나 body 실행은 별도 설계한다.

## 구현·검토·자원

실제 gpt-6-sol/high가 `tools/probe_c_banc_m3_lh_extensor_ablation.py`와 `tests/test_c_banc_lh_extensor_ablation.py`를 구현한다. E250 constructor/actual packed CSR 검증, E254의 파라미터화된 drive/incoming/edge-current helper, E255의 상태 reset/관측/부분 결과 보존을 재사용한다. E254 prepare의 E251/5MN/300controls나 E255의 4arm/5MN 전역 상수를 원본에 monkeypatch하지 않는다. E257 schema/6개 변경/2MN/5arm/101행으로 명시적으로 준비한다. 부모 FIRST→필요시 같은 Sol 첫 수정→SECOND→부모 잔여 수정과 독립 검산 후 한 번 실행한다.

원본1/retry0/CPU1/GPU1, network4/setup실행240 중capture80/과학10000tick/currentprobe505, body·physics·fit·solver0. 이전 E255는2모델/8000tick에18.17s였으므로 추가 두 생성과101회 관측을 포함해 internal55s(마감8)/external60s/supervisor65s를 예약한다. host4GiB/추가GPU3GiB/최소GPUfree3GiB, output256MiB/최소disk1GiB. 누적6997.40523829962/7074s, 잔여76.59476170038033s에서65s 예약, 증액0이다. 원본당300s 상한을 유지한다. 최종화/cleanup 실패는 성공 판정을 철회하고 모든 부분 raw와 최초·추가 오류 및 실행 비용을 남긴다.

출력은 `verification/banc-natural-gait-m3-20260919/e260-lh-extensor-ablation/`에 write-once로 보존한다. constructor/arm/관측의 requested·entered·completed·unknown partial을 구분한다. 부모는 source CSR와 단일cut, 원래 drive, 32lane actual current 재구성, 기존 bank 재구성, diagonal 및 모든 차이/가설을 독립 계산한다. 준비/생성/부모검산 비용은 원본과 별도, 미측정 개발은 null이다. E257 기각, E150 best partial, M2/UI/F6 미노출, F1–F8 고정 계약과 모든 과거 실패를 유지한다.
