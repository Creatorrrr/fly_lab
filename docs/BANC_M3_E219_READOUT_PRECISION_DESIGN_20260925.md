# E219: E218 후보의 전달방식 대조와 정밀도 민감도

E218은 고정 반응계수로 원래 조건부 proxy를 통과했고, 부모 독립 검산을 받았다. q오차는57.057365%감소했지만 startup RMS/max여유가각3.95e-8/4.01e-8rad다. 실제native로옮기기전에 신경 반응 교체와 근육 전달 교체의 영향을 분리하고, 출력값 반올림의 영향을 확인한다. 본실험은 저장 rate의 전달 계산만 하며 신경 적분·physics·CUDA·학습을 실행하지 않는다.

## 동결된6가지 계산

원래E201의2001×391rate와E218 candidate의동일shape rate, E150 start의원래adapter/calibration/parameters/bank/activation/heldq를 사용한다. 2×2 네 경우는 original/new rate × canonical10ms/원래monotone5·10·40ms hinge전달이다. canonical은검산된E209 canonical_readout, monotone은E203 DecoderAdjoint.forward를재사용한다. E218의통합대상은사전에canonical로정해져있으며이결과로다른decoder를재선택하지않는다. row0는실제heldcommand와실제초기activation을따른다.

정밀도추가2경우는new rate에만적용한다. `rate32_readout64`는모든391rate를np.float32로한번반올림한뒤float64 canonical에넣는다. `readout32`는rate/pool/moment/rest/초기activation을np.float32로만들고 alpha32=np.float32(-expm1(-.005/.01)),gain32=np.float32(.02)로매frame `a+=alpha32*(pool32@rate32-a)`, `q=rest32+gain32*(moment32@a)`를계산한다. tendon도float32 .2/tanh(a[ltm]/float32(10))를쓰고원래6다리순서를유지한다. 결과는float64배열에보존하되 계산중dtype는float32임을검사한다. row0 heldq는원래float64값을보존하고초기activation은각계산정밀도에맞는값을저장한다.

이두profile은끝점rate/decoder 반올림의민감도진단뿐이다. 실제native RK중간rate/h/CSR 재귀입력의정밀도검증이나native성공으로표현하지않는다. 원래계수·신경파라미터·상한·목표·기준을변경하거나재학습하지않는다.

## 비교와 증거

각6경우의command/activation/tendon(원래monotone은bank도),정확한dtype·ID·초기값·sourceSHA·계산순서·strictmetrics/proxy와startup한계여유를저장한다. raw를판정전에기록한다. original_monotone은기록E150 command와atol2e-10/rtol2e-12에서일치해야하고original_canonical은E217기존canonical기준과같은오차안에서일치해야한다. new_canonical은E218 candidate의q/activation/tendon과같은오차안에서일치하고원래strictproxy통과를재확인해야한다. 이기준복원실패는실험invalid이며조건을완화하지않는다.

다른3경우의proxy성공/실패는있는그대로보고한다. 모든경우의q/velocity/각다리/startup/tendon 지표와E197원래기준을사용한다. 조건부오차개선·물리보행·생물학적타당성을구분하고6profile중선택검색을하지않는다. 부모는반복식을다르게구성하여기준2×2와정밀도dtype/오차/한계여유를독립검산한다. 기존raw/hash/failures를보존한다.

## 구현·한도

이계산은이미검증된readout재사용과2정밀도profile만필요하므로 기존실제gpt-6-sol/medium에게구현을맡긴다. 부모설계→Sol구현→부모FIRST→필요시같은Sol첫수정→부모SECOND/잔여수정. 소유파일 tools/analyze_c_banc_m3_readout_precision.py, tests/test_c_banc_readout_precision.py. StateStore.load는저장배열만읽고신경·bodymodel을생성하지않는다. 순수작은생성입력의dtype/초기값/원래행순서검사만하며새실제생성solver/CUDA0이다.

원본process1/6고정readout/CPU1/GPU0/fit0/neural0/physics0/retry0, 내부20/외부30/감독40초, RSS4GiB,출력64MiB,free disk1GiB. hash검증·입력로드를내부시간에포함한다. 누적5789.9107909997/6004초·잔여214.0892090002999초에서40초예약,증액0. 준비/순수/부모검산/agent의미측정비용을0으로두지않는다. 실제fullBANC/물리후보채택0/F6미노출/최선E150·M2·UI보존/F1–F8active·미완료. native구현·검증은이결과이후별도구체계약과예산으로진행한다.
