# D19: 실제 검증 시간을 줄이고 보행 평가까지 연결

현재 M3 목표는 ACTIVE/INCOMPLETE다. 사용자는 일반 승인을 다시 요청하지 말고 판단·진행하도록 지시했다. ROOT가 설계하고 같은 Astra/max 검토를 반영한 뒤 구현한다. 이전 D18 Sol FIRST/soleR1/ROOT SECOND와 residual03 봉인은 불변이다. D18의 빠뜨린 기대값 수정은 ROOT가 private candidate로 완료했고, 이 설계는 그 이후 운영 회복과 성능 검증의 별도 단계다.

## 현재 사실과 다음 성공 기준

실제 r4는 FF 두 표시 후 owned timeout으로 끝났고 JUnit은 없다. private metadata actor 실제 f808ac/session70332 → f46f0c/exit0은 helper controls44와 세 실제 fee proof 호출을 통과해525.537초를 한 번 정산했다. ROOT 독립9cc15f/exit0은323 debits/205 calls,500202-byte canonical6b486f2c, 한 cost1544/250ca9ea, whole forward/inverse의형·순서·binary64,backup/lockrelease/Source871/current13불변을 확인했다. Original은불변이다. 원래운영validator는이actualwrapper이력에여전히legacy suffix거부를한다. 비용정산은일반runtime PASS가아니다.

첫실패는Native독립원래115+finite6목록에서새세경로를누락한것이다. 두번째는actualsavedfixture후첫sequence비교의r3/321/180/160기대값이다. 후자는정적508b21의근거이며복구된pytesttraceback이아니다. ROOT private candidate261128bytes는세곳을수정했고d40b39/exit0으로전체inverse,원래목록의순서·중복,기존finite6,나머지6역할과전체negative/나머지본문보존을확인했다. 운영채택이나projecttest실행은아직없다.

성공기준은새로운정확한Source에서동일한고정4가실제완주하고,그후실제690 qualification을통과하며,같은후보의실제Originalbody와audit에서F3/F4/F5를확인할수있는상태다. 메타데이터회복이나교사proxy에서멈추지않는다. 실제신경폐루프보행이실패하면그측정값을다음모델수정의원인으로사용한다.

## 구현 범위

1. 현재 Source/검토/운영13과r4request/trace/outer/capture/actor/paidcost/before/terminal323을고정된별도원문으로보존한다. 새epoch가r4이력을검증할때기존legacy branch와동일exit branch는보존하고이번한건의완전한wrapperproof만별도통로로연결한다. 세fee호출에서통과한실제actor 원문·launch·firstFINAL·controls·현재paidfold를인증한다. 비용의기존124/1/true필드와actualcommand를유지한다. guard를전역완화하거나collector명령을꾸미지않는다. paid323/HOLD를새Source에서한번회복할때과거322의모든값·형·순서·binary64와한번의r4fee를독립재구성한다. 과거HOLD의제거는검토된다른Source로회복하는명시적단계이며성공기록으로바꾸지않는다.

2. 새 helper Source key를 추가하지 않는다. 현재 Source는262142bytes이고 JSON 문자열 밖 공백을 모두 제거해도262141bytes다. 절감은 마지막 개행1byte뿐이다. ROOT 실제2922ab/exit0과 독립 Astra 관측이 공백 축소 전제를 반박했다. 기존871members와 Source cap262144를 유지하고, 기존 mutable Source member인 `tools/banc_m3_conserved_resource.py`(188126/262144; 여유74018)와 `parent-direct-science-account.ps1`(328408/393216; 여유64808)에 이름이 구분되는 D19 역사검증 함수를 넣는 zero-addition 안을 사용한다. 기존 함수의 raw 본문은 보존하고 새 epoch의 좁은 entry point만 연결한다. Source에는 기존 정확한 replacement pin만 바꾸고 새로운 보고서는 별도 supplemental provenance로 인증한다. 기존 Sol 봉인 보고서를 새 구현의 보고서로 재사용하지 않는다. 구현 후 실제 파일 크기·Source 크기·전체 inverse·음성 검사를 확인한다. 한도에 맞지 않으면 이 안의 채택을 거부하며 cap을 올리거나 history를 삭제하지 않는다.

3. 과거 r4 helper의 logical current Source/review/current13은 새 epoch가 바뀌면 같은 경로에서 같은 bytes/hash를 갖지 않는다. 따라서 과거 논리 descriptor는 그대로 두고 이번에 보존한 immutable archive와의 정확한 연결을 새 역사검증 함수에서 확인한다. 과거 paid actor는 raw 증거로만 읽고 실행하거나 unpaid322/cost-absent/ENTRY/OwnedBefore 절차를 재사용하지 않는다. actual r4 command/firstFINAL/525.537fee와 paid323의 whole typed/order/binary64 fold를 읽기 전용으로 인증한다. 신규 metadata 연결은 Source fence를 우회하거나 cache/managed acceptance 상태를 임의 설정하지 않는다. Native130909/131072에는 작은 hook만 넣는다. 필요하면 명시적인 문자열 밖 comment/빈줄만 정리하고 inverse를 기록한다. C# template, backend hash, 타입·캐시·compiler·lookup·negative 검사와 기존 sentinel은 유지한다. metadata 회복 중에는 NPZ/Original payload/worker/교사/보행을 실행하지 않는다.

4. 445.1363초는 부모 ENTRY부터 첫 Native 관측까지의 누적 시간이다. Native PID 생성은 부모 소유자 생성442.1806249초 뒤이며, Native 자체 실행시간은 현재 자료로 알 수 없다. 기존 temp 저널도 과거 실행에서 복사된 자료라 현재 병목 증거가 아니다. fixture 준비, Source/provenance 검증, Native spawn 직전/직후를 현재 PID/FILETIME·실제 command/root·같은 부모 ENTRY와 각 단계의 별도 단조 start/end로 관측한다. cap 경계검사의 큰 PowerShell byte[] slice를 Array.Copy로 바꾸는 것은 아직 입증되지 않은 의미 보존 후보일 뿐이다. Array.Copy를 주요 병목 해결책이나540초 내 완주 근거로 사용하지 않는다. 적용할 경우 검사개수·cap+1거부·정확한cap전체 bytes·strictUTF8/0/1/buffer boundary·실제 reader와 cleanup을 전부 유지한다. 관측 전에는 속도 개선을 주장하지 않는다.

5. 고정4노드및원래control을유지한새Source·새call을실제한번실행한다. r4자체와같은Source를다른이름으로반복하지않는다. 초기화와기대값정상검사는큰Native검사보다먼저실패위치를알수있도록배치할수있으나검사를삭제하거나결과를합성하지않는다. 실제완주/무skip/JUnit/ownedclosure/oncefee가있어야다음qualification의prerequisite다. 호환성실패·관측불완전·timeout은그상태로기록하며690이나Original을자동진행하지않는다.

## 기존 자원 안의 선택

정산후실제generatedlocal763.2486916000007/shared771.0366760000006,Original1106.9150391004946이다. 기존고정4의540과qualification690을그대로보존하면generated가부족하다. 새globalgrant나refund/reset을사용하지않는다. 추천안은**기존Original미사용분500초를generated에한번보존이전**하는것이다. actualbefore고정,현재emptyownership/soleknownr4HOLD,ownPID/FILETIME/CreateNew/atomicreplace/readback과전체계산을검증하는metadata단계를설계한다. generatedlocal5794→6294/global6530→7030,Originalglobal13370→12870이고합19900은변하지않는다. offsets118/165,이미소비된비용,Originalscience-local6000/stage6300은변하지않는다. 이전후generatedlocal약1263.25/shared1271.04,Original606.92다.540+690과조건부Original600을모두보존하며gen약33.25/original약6.92의여유가남는다. 이수치는신규할당의가정이며현재canonical은아직변경하지않았다. 이전을검증하지못하면현재HOLD와cap을유지하고science를실행하지않는다.

Source회복/previouspaid검증/보존이전/첫admission의순서와ledgercounts를Astra가검토한다. 예상새bootstrappaid324/calls206,qualificationpaid328/calls210,body329/211,audit330/212를독립typedfold로검증한다. 아직실제신규hash/JUnit/cost/미래성공을작성하지않는다. 원래코드·과거callnames·role조건·실험history와M2/UI를보존한다.

## Decision fork 및 종료 조건

- 기존 host의 metadata연결/Source871·cap/wholepaid323/보존이전이검증됨 → 새고정4를최대540초envelope안에서한번실행하고현재단계별관측을수집한다.
- bytecopy변경이정상/거부결과보존을증명하지못함 → 채택하지않고원래검사본문을보존한다. 공정한실제같은조건비교가없는속도향상주장은하지않는다.
- 고정4실제PASS/knownclosure/fee/JUnit → 같은후보에40/35/170/420및25close를포함한690을진행한다.
- 고정4또는690의실제실패 → 해당관측으로실패원인을확정하고비용/HOLD를유지한다. 동일candidate자동반복이나이름교체재시도를하지않는다. metadata작업만계속늘리는대신다음근거있는다른가설/자원내방향을선택한다.
- qualification실제PASS → 최소실제finalpath를조건부Original300body+120audit+180close에서평가한다. F3/F4/F5를같은후보에서측정한다. proxy손실감소를보행통과로표시하지않는다.
- 실제bodyvalid이지만보행F3/F4/F5미달 → 실제발접촉·운동폭·좌우phase/속도/안정성에따라다음신경모델가설을설계한다. candidate미채택을유지하며F1–F8을완료로표시하지않는다.
- unknown/uncertainty/소유권미확인 → 원문·비용책임·reservation/exclusion을보존하고자동재시도하지않는다.

고정F1–F8,eight frozen8352/fulltheta/alpha0.0001,최종외부teacher/CPG/위상/궤적/rootcorrection금지와BANCnode/connectivityprovenance를유지한다. 코드/테스트/도구회복성공,실제BANC신경단독폐루프보행,생물학적기전은별도로판정한다. E150partial/candidate4unadopted/NativeVJPunverified/biologyunproven을지우지않는다.
