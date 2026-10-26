# E296 — 기존24core로 투사하는 source의 자원 소모를 제외한 후보

Root 설계, 2026-09-28. E295 저장분석/독립검산은 core24 drive/adaptation과391MNdrive가 E150과 exact인데 rate와 명령은 달라졌음을 확인했다. Core향 선택 양수 손실의 core-source 비중은 .05/.10/.20/.50초56.32/67.24/72.03/61.36%이고 나머지source기여도 남는다. E294의 .5초건강성 유지·startup qRMS.228/max1.318 기각은 그대로다. 다음은 동일 동역학에서 적용 위치를 바꾸는 한 개 구조 후보이며 새로운 생리 기전으로 부르지 않는다.

## 고정 정책과 해석 범위

E294의 positive-only u=.05/negative-present u0, tau=.150을 유지하되 **원래 선택 outgoing 중 기존24core를 향하는 것이 하나라도 있는 source는 u0**로 한다. Core24 roster는 기존 E257 baseline/core-activity.npz의 core_ids 전부이며 결과·활동·순위·seed로 고르지 않는다. 따라서24core를 향하는 선택6172간선의 resource계수는 정확히1이다. 신경 rate나 전체 미래 입력이 원래대로라는 보장은 없다. 비선택 afferent와 간접 네트워크 경로는 남아 있다.

정적 조사에서 positive-only3194source 중1681개가 면제되고1513개/117973개선택간선에만 자원동역학이 남는다. 원래 전체source7136/선택edge725849/core24/MN391 및 모든CSR bytes/위치를 유지한다. Source별 R이 공유되므로 면제source의 다른 outgoing도 함께 원래 전달식으로 돌아간다. 개선되더라도 core향 coefficient 보호와 전체resource영향 감소를 인과적으로 분리하지 않는다. 24개를 전체생물학적 core라고 확대하지 않는다. All-zero퇴화 후보가 아니다.

관절 decoder·근육·신체·world·감각·신경초기값·u/tau 크기·source core 모수는 바꾸지 않는다. Neutral은 전source u0이다. 외부위상/CPG/교사런타임/rootforce/새간선/시간스위치가 없다. 모든 기존 실패·비용·M2/UI·F6미노출과F1–F8을 유지한다. 기존실측parameter를복원했다고 주장하지 않는다.

## 판별과 자원

원래 E150 neutral .005 whole scientific/trace/native/recruitment exact(독립실측wall비용은별도보존)→실제 .5초 strict startup(qRMS≤.018/max≤.04,LTM≤.009/.025,nativeupright≥.9/비발접촉0)→통과한같은session만1.5초 short로이어간다. 원래short forward/slip/eligiblebout/6leg완전주기 기준을그대로쓴다. 실패후parameter/seed/기준을바꿔재시도하지않는다. 통과시에만같은후보10초와나머지F기준을별도동결한다.

원본1/retry0,science85+final30/internal115/supervisor125초,body≤2/live1,scientificneural≤6020/physics≤15050,host6GiB/addGPU4GiB/output2GiB/freeDisk4GiB/CPUthread1. 현재원본누적8022.108435499544/8095초를모두이월한다. 구현·검토·tiny기존승인SHA 확인뒤실행전에필요한단계배정및125초예약을별도기록하며현재원본예약0이다. 전체사용자시간/토큰한도는없다.

## 구현·검토 계약

실제 gpt-6-sol/high가 신규 `tools/verify_c_banc_m3_core_afferent_resource_body.py`와 `tests/test_c_banc_core_afferent_resource_body.py`만 구현한다. Root FIRST→필요시같은Sol 첫수정→root SECOND→root잔여수정이다. 기존 E289/E290/E291/E294/E295 소스·raw·docs·CP는읽기전용이다. 새GPU커널은없으며 현재 source/tiny qualification SHA를상속한다.

순수 `protected_source_selection(original_uniform_roster, core_ids)`는 승인 E294.positive_source_selection으로 원래 E289 uniform roster를 검증/positive-policy로변환한다. core_ids는unique int64,실제원본premotor IDs안이어야하며 입력값을변경하지않는다. 선택 post가core인source의u만0으로변경해반환한다. Empty core/unknown/duplicate/type위반을거절하며generic작은fixture는허용하되 actual loader에서는동결24개를요구한다. 선택edge/Rsource 배열구조와나머지typedmetadata는변하지않는다.

`NativeDependencies`는승인E294클래스를상속한다. `load`는literal E294 freeze로 E294.NativeDependencies.load를호출하여정확한baseline provenance와실제입력을먼저검증한다. 원래E289uniform roster(기존 selection alias)를읽어E294 positive-policy가로딩된baseline roster와exact임을확인한뒤,동결 core-reference의24개로새roster를정적으로계산한다. 별도동결 `protected_selection`과exact비교하고새candidateidentity/원래neutralidentity를계산해검증한다. data의selection/roster/freeze만명시적으로교체한다. 기존construct/initial/capture/neutral/run_arm/execute는수정없이읽기전용재사용한다. Monkeypatch나moduleglobal변경은금지한다.

신규 gate는 E296 freeze/review schema,OWNED/source/input/package/design/output/limits/criteria를검증한다. Root는 `base_freeze/base_review/base_original_audit`로 E294원본 freeze/source-review/root raw audit를핀하고, E294→E292→E279 lineage와기존E289/E290/tiny source승인을보존한다. 같은baseline input aliases/bytes와원래source/preservationpins가그대로여야한다. E294의source review가새candidate승인인것처럼처리하지않는다. `core_reference`는 E257 baseline/core-activity.npz이며E295 frozen inputs/producers에묶인SHA와같아야한다. `e295_freeze/e295_root_audit`도핀하여판별근거를보존한다. 검증을위해이미존재하는E294output의fresh-output gate를다시호출하지않으며,그외기존literal provenance/load 검증을재사용한다.

출력은새 `verification/banc-natural-gait-m3-20260919/e296-core-afferent-resource-body`로고정한다. 실제source manifest는새E296 OWNED와실제상속sourcepins전체를보존해야한다. E291 result schema재사용가능하나diagnostic_id E296,모든natural/F/adoptionflag는false이며strictstartup와short가실제통과한경우만별도10초 eligible이다. E294와같은CPU/GPU환경·cleanup·bounded receipt publication을유지한다. 추가baseline/body실행이나관측을숨겨넣지않는다.

ROOT/OWNED/SOURCE_FILES/BASE_KEYS/PATHS/LIMITS/STARTUP_LIMITS/FREEZE_SCHEMA/REVIEW_SCHEMA, gate/execute/run CLI를노출하고root에게exactfreeze필드와alias를보고한다. 새입력은기존E294모든입력+core_reference+protected_selection이며원래positive roster도유지한다. 생성60초/2GiB·원본/model/body/CUDA/solver0에서작은source-shared면제/간접영향범위/남은active/중립identity/입력불변/오류거절/실제constructvector전달/manifest/기존startupfail후continuation0를검사한다. Root독립정책계산과원본raw/hash/clock/startup/면제Rexact1 감사뒤에만후보를판정한다.
