# E268 — E267 계수 학습을 위한 실제 RK4 입력 기록

2026-09-27. E267 중립 전체 BANC/CUDA 검증을 통과한 같은 동역학을 사용한다. 최종 목표와 F1–F8, 기존 모든 실패, M2/UI/F6를 보존한다. 다음 작업은 **단계 자료 한 번 수집 → source별 계수 한 번 적합 → 실제 재귀·신체 후보 시험**이다. 또 다른 국소 감각 진단으로 대체하지 않는다.

## 학습 경로와 변경 범위

E128 conditional CUDA forward/간선 tangent/activation cotangent 구조를 재사용할 수 있지만1572간선+96gain 하드코딩을 그대로 쓰지 않는다. E267의1660×3=4980개 비음수 계수만 맞추며, 세포 gain/τ와 E150의 **실제 기존 monotone 근육 adapter**를 고정한다. 앞선 E267 문서의 'canonical E150 전달'은 이 기존 전달을 보존한다는 의미로 명확히 한다. `unwrap_e150_checkpoint`로 monotone patch를 제거한 lower adapter는 이번 후보가 아니다. 신경 자료의 teacher 문맥과 근육 decoder 선택을 혼동하지 않는다.

native 단계에서 `total = drive + CSR_sum`을 기록한다. 조건부 후보는 `total + sum(W*mask*((c0-1)*r+c1*hfast+c2*hslow))`를 사용한다. 이는 고정 source 입력 아래의 float64 증분 근사이며 실제 packed float32 재귀망의 대체가 아니다. 중립 MN/decoder 재현·해석적 gradient 검증 후 한 후보를 학습하고, 같은 계수의 실제 신체에서 판정한다. 목적·계수 선택·반복/시간 한도와 수치 허용치는 첫 fit 전에 별도 동결한다.

strict startup은 E150 실제 문맥의 controls1..99에서 원래 `loaded_startup_target_command/tendon`과 비교하는 q RMS≤.018/max≤.04, LTM RMS≤.009/max≤.025다. 교사 명령이나 baseline의 자기 명령을 보호 목표로 바꾸지 않는다. 교사1..1200은학습,1201..2000은 이미 관찰한 개발 비교다. F6는 여전히 미노출이다.

## 필요한 자료

E245의 선택4046개 발화는 control endpoint이며 RK4 중간 단계가 없다. E126의69 source 자료는 다른 회로 문맥의 masked rate이므로 E267의 raw-rate 이력 입력으로 사용할 수 없다. 실제 E240/E245 교사 폴더에서 확인한 stage 파일은 positive_recurrent/negative_magnitude/drive다. total_current 파일이 있다고 가정하지 않고 이번에 정확한 `drive + CSR_sum`을 기록한다.

두 문맥을 신체 재실행 없이, 저장 drive로 전체 신경만 재생한다.

- **startup:** E265 B/E267과 같은 전체 control0 신경 상태, 기존100controls/0.5초. controls1..99가 보호 구간이며100은 끝 일치 확인에도 사용한다.
- **teacher:** E245/E240 교사와 같은 전체 시작 상태, 기존2000controls/10초. 실제 sensory scatter 후 원래20 navigation ports의−5000 배경 및 두 descending의+5000을 원래float32 순서로 더한다. 선택4046개 바깥을 모두0으로 가정하거나 endpoint를 보간하지 않는다.

같은1660간선/51source/391MN, 모든계수[1,0,0]. 매 neural step의 네 RK4 단계(0, half1, half2, end)를 각각 기록한다. 두 half-stage를 합치지 않는다. 단계 source raw rate/mask 및 두 실제 source history와 MN의 post-drive total을 저장한다. 한 모델초에4000neural step/16000stage다. 학습 입력에 교사 q나 위상 값을 추가하지 않는다.

## 관측기 구현

소유 파일은 `tools/banc_m3_source_stage_observer.py`, `tests/test_c_banc_source_stage_observer.py`다. 실제gpt-6-sol/high가 새 하위 작업으로 구현하고 부모 FIRST→필요시 같은Sol 첫수정→SECOND→부모 잔여 수정 절차를 적용한다. 기존 E267/제품 모듈을 변경하지 않는다.

`SourceFilteredSynapseNetwork`를 상속한다. E267의 실제 `source_filtered_cuda_source` CSR에서 유일한 `value = add_exact(drive[row], value);` 직후 lane0 수동 store만 추가한다. source값은 해당 단계의 `rate[row]`, 현재 `mask[row]`, 같은 단계 `history[2*source_column:...]`를 읽는다. `masked[row]`를 source raw로 기록하면 안 된다. 원래 합산·history 미분·stage/finish 순서와 모든 신경 식은 유지한다.

관측 배열은 모형 identity/checkpoint 밖에 둔다. armed=0으로 생성/warmup/capture를 수행하고, 전체 신경 초기 상태 이식 뒤 한 번 arm한다. 관측기는 생성 stream을 고정하고 포인터를 바꾸지 않는다. CPU fallback을 허용하지 않는다. 생성/복원 중 inference tensor 문제를 재발시키지 않도록 일반 텐서를 capture 전에 할당한다.

- motor IDs는 원래391 roster 순서를 보존하고 실제 native motor roster와 exact여야 한다. source IDs는 E267 선택의 sorted unique51과 exact여야 한다.
- 최대160000stage, GPU 관측 메모리512MiB. 각 field는 `total_input`:float32[S,M], `source_rate`:float32[S,P], `source_mask`:float32[S,P], `source_history`:float64[S,P,2]. 두 mapping 배열, count/armed/overflow와 모든 버퍼 비용을 합산해 한도를 확인한다.
- motor/source별 count가 모두 예상 stage수와 같고 overflow0이어야 완전하다. capacity를 넘을 때 out-of-bounds store를 하지 말고 overflow와 partial raw를 남긴다. arm 뒤 restore/reset을 금지하고 다른 stream을 거부한다. disable은 안전하게 관측 종료 tick을 기록한다.
- export는 complete 여부와 actual valid counts를 포함한다. 오류 뒤에도 안전한 공통 prefix와 count/overflow를 읽을 수 있어야 한다. 파일 저장은 root runner가 담당한다. source buffers와 source/target ID mapping을 함께 저장한다.
- 파라미터/경계 검사용 pure CPU helpers를 제공한다. 생성 CUDA 테스트는 부모 예약 후에만 실행한다. 같은 작은 신경망에서 관측 on/off의 전체 과학 상태 exact, 첫 단계 직접합, raw와mask 구별, 두half 단계 분리, 계수 비중립 경로, overflow/partial, 재arm·복원·stream 거부를 검증한다.

원본 실행기는 root 소유다. 시작·끝 원래 전체 신경 상태 및 매control의 기존신호 exact가 필요하다. startup은 E267의 새source history endpoint까지 exact여야 한다. teacher는 선택4046개 endpoint 및 전체끝을 E245와 비교한다. 원본입력/drive/소스와 실행 상한은 준비 뒤 별도 freeze한다. 원본 실패·부분배열을 지우거나 자동으로 반복하지 않는다.

## 자원·판정

현재 원본 누적7134.341214599615/7182초, 잔여47.658785400385초다. 이번 문서/구현은 아직 원본 실행 예약이 아니다. 자료 수집은 총10.5모델초, 신경만/physics0이며 기존 개별31.5모델초 한도 안이다. E267 실제 advance+.5초 동기화1.9388초 및 모형 전체8.003초를 비용 근거로 사용하되 관측과10초 비용은 아직 미측정이다. 구현·생성 검토 후 원본 수집 상한과 필요한 미래 배정을 명시한다. fit와 실제 body의 비용을 현재 잔여에 모두 넣는다고 가정하지 않는다.

관측 자료와 적합 성공은 자율 보행 채택이 아니다. 한 후보가 실제.5초 startup을 통과하면 같은 실행을1.5초까지 이어 전진/slip 비악화, bout≥.3초와 주기 개선을 판정하고, 지지할 때만 원래10초/F1–F8로 진행한다. 미달 시 문턱 완화나 계수/τ sweep으로 결과를 바꾸지 않는다.
