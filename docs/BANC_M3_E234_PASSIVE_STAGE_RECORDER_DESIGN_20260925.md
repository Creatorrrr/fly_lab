# E234: 별도 신경 전달 경로의 수동 RK 입력 기록기

E233 정정된 저장분해는 실제와 조건부 기억이 첫5ms에는 같지만 control5부터 달라지고, startup 실패가 free effector 동역학 차이에 있으며 readout 양자화만 제거해서는 복구되지 않음을 보였다. 아직 실제9D feature의 현재 E/I/drive가 저장되지 않아 그 차이가 실제 입력 변화인지 수집·구현 문제인지 직접 확인하지 못했다. H234는 E223 계산을 바꾸지 않고 각 RK stage의 실제 입력과 effector 상태를 기록할 수 있다는 **관측 도구 가설**이다. 보행 개선/후보 채택 가설이 아니다.

## 구현

실제 gpt-6-sol/high가 `tools/banc_m3_effector_stage_observer.py`, `tests/test_c_banc_effector_stage_observer.py`만 구현한다. 부모 FIRST→필요시 같은Sol 첫수정→SECOND→부모 잔여수정. 기존 동결 파일은 수정하지 않는다. `EffectorStageObserverNetwork`는 `NeuralCurrentEffectorNetwork`를 상속하고 별도 out-of-checkpoint CUDA 관측 버퍼를 생성자 graph capture 전에 할당한다. identity와 snapshot은 부모와 같다.

기존 CSR/native CUDA 소스나 목적지를 바꾸지 않는다. `_launch_native`에서 원래 호출 직전에 **별도의 쓰기 전용 관측 kernel**을 같은 stream에 실행한 뒤 super 원래 호출을 그대로 한다. 실제 인자 구조 native_stage27/native_finish22를 검증하며 양쪽의 처음12개 common 인자와 stage state 인덱스15,16,17을 사용한다. E/I는 args0/1의 free순서 float32, drive는 args2[args3] float32, effector/fast/slow는 args15/16/17의 float64다. stage 시점 상태는 base12..14가 아니다. 기록이 계산용 원래 버퍼에 쓰지 않음을 생성 검사와 GPU 대조로 확인한다.

각 free MN별 counter와 공통 armed/overflow를 사용해 record[position,MN]에 E,I,drive,r 및 fast/slow3좌표를 저장한다. 한 native호출에서 MN당1행 증가, 4stage/neural tick. 전체나 fixed MN firing은 원래 경로 그대로이며 향후 body기록의 control rows에서 얻는다. 최대capacity7920(99controls×20ticks×4), 최대343 free, 관측buffer≤256MiB. 생성8뉴런에서는 더 작은 free수와capacity를 허용한다. float32 E/I/drive, float64 r/fast/slow 배열, counters int64, armed/overflow int32. 미사용 행은 NaN, 범위초과 쓰기는 막고 overflow1과 실제 counters를 보존한다.

초기화/capture 동안 armed0; arm은 capture 완료 후 한 번만, 시작tick과stream 고정, stop 전 hostcopy 금지. restore/reset은 arm 이후 금지해 수집 epoch 혼합을 막는다. stop 후 반복 stop은 안전하고 이후 advance는 기존 신경 계산을 계속하되 관측raw/counters가 변하지 않아야 한다. 원래 snapshot에 관측 포인터/버퍼/metadata를 추가하지 않는다. 새 객체는 원래 E223 snapshot을 정확히 restore할 수 있어야 한다. 생성자 CUDA/device/dt/shape/capacity/메모리 validation은 할당 전에 한다. kernel layout 또는 stream이 바뀌면 닫힌 실패다.

## 부모 생성 CUDA 비교

기존 동결 `parent-e223-fixture8.npz`의8뉴런/3free/1fixed/2muscle/8interval×2tick만 사용한다. 원본BANC bundle·신체/실제보행data는 읽지 않는다. 원래 E223과 새 관측자의 각 interval snapshot(rate/adaptation/effector/fast/slow/identity/clock) 및 readout은 byte exact여야 한다. 같은 관측자를 graph실행과 direct실행으로 비교하고 관측raw도exact로 비교한다. 4interval후 디스크 snapshot을 새 관측자로 restore해 나머지4interval의 상태와 관측raw를 정확히 비교한다.

CPU 독립 E223 scalar 참조를 이용해 각stage E/I/drive/r/fast/slow를 같은 인자 순서로 검사한다. 기존 CPU 최대절대오차2e−6을 유지하고 rawdrive는exact, counter=64, overflow0, 기록행shape64×3를 요구한다. capacity4에서2tick(8stages)을 실행하는 별도 overflow관측자도 원래E223 첫2tick 상태와exact, counter8/overflow1/first4rawexact여야 한다. stop 후1interval(2tick)을 양쪽에 진행해 신경 상태 exact/관측buffer불변을 확인한다. 생성 science tick합계62, 생성자5개(원래E223/graph/direct/restore/overflow), setup warmup/capture별도기록. 생성테스트 통과가 원본BANC 또는 보행 성공이 아니다.

## 판정과 자원

부모 source review와 input/source해시를 실행 전에 고정하고 generated CUDA단일job1회/자동retry0. 불일치·shape/NaN/overflow오류·자원초과 시 전체관측가설 미통과, raw/단계별cost와부분상태를 보존한다. 통과 시에만 다음 E235 실제startup.495초 수집을 별도 설계한다. 이 단계에서 fitting/physics/전체BANC0이다.

CPU1/GPU1, 내부45초/외부50/감독60초, RSS2GiB/추가GPU1GiB/출력100MiB/free1GiB. 환경numpy2.5.3/scipy1.18.1/torch2.14.0+cu130/cupy14.2.0, 기존생성fixture 사용. 생성CUDA 비용은 원본누적6090.0136165996555/6119초(잔여28.986383400344494초)와 별도계측, 원본추가배정0. F1–F8미완료/F6미노출, 기존실패/E150최선/M2/UI보존.
