# BANC M3 검증 실행 오류 정산 결과 — 2026-10-08

사용자의 위임에 따라 추가 승인 질문 없이 진행한다. 정산한 검사는 보행 성공이 아니다. F 단계의 첫 lint 호출은 1.755초 뒤 exit1로 종료됐고, Ruff와 물리 신체 실행까지 도달하지 못했다.

실제 반환 전체와 실행 당시 소스 네 개를 보존했다. 문제의 보관 파일은 269자 경로에서 실제 존재하고, 읽기 전용으로 확인한 LongPathsEnabled 값은0이다. Python의 긴 경로 처리 제한이 유력하다. 별도 미정산 Python 재현 검사는 하지 않았다. 참고: https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=powershell 및 https://docs.python.org/3.12/using/windows.html#removing-the-max-path-limitation .

ROOT 설계 → 실제 GPT6 Astra/max 리뷰 → ROOT 반영 → 실제 GPT6.1Sol/high 구현 → ROOT FIRST/SECOND를 수행했다. 수정이 필요한 항목이 없어 SolR1은 요청하지 않았다. 원본 account200156바이트와 commit 전체가 그대로 보존된 별도 복구 사본을 실제 실행해 기록을 생성했다. 그 기록은 ROOT_ADMINISTRATIVE_RECONSTRUCTION/NOT_QUALIFIED이며 실제 supervisor의 성공 기록으로 꾸미지 않았다. 생성 시에는 ROOT의 admission 예약15가 아직 남아 있다고 명시했다.

그 다음 원본 Native commit을 실제 실행했다. exit0 반환과 독립 전체 파일 확인에서 실패 비용1.755초가 정확히 한 번 반영됐고, 이전283개 비용과165개 호출 기록이 유지됐다. 이제284개 비용/166개 종료된 호출이며, 모든 예약·worker·pending·hold가 비어 있고 물리 잠금도 해제됐다. 실행 한도2300/1564/15600과 기존 F 재배분 포인터, Original 사용량은 변경되지 않았다.

실제 증거:
- ROOT 정산 전체: verification/banc-natural-gait-m3-20260919/parent-f-lint-settlement-ROOT-actual-result-20261008.json5902/e19b00791d3a30630446c6b7d0a123a3594c93c2d574efe3a2d8d1c2fb50abdd.
- 실제 비용: parent-direct-science-20261008-contact-f-lint-full-return.json1290/3088f456c7a2fa69f1ae6cb814f0148fa3ebc08a08d72d9ed1b2969c190acdb1.
- 전체 종료 snapshot: parent-f-lint-closed284-20261008.json455909/cdfd6e6653c2b6f3a08e0cb9ae60c5947fd0bc0d2a21bdf21b628e5ca64c86d8.
- 실제 복구 기록: parent-direct-science-20261008-contact-f-lint/outer.json24603/a3ea07580402be63b0bbd8eb0ec68fbe9d989f286f937af2e431c6c0c7d94864.
앞의 짧은 파일명은 모두 같은 verification/banc-natural-gait-m3-20260919 폴더를 기준으로 한다.

현재 남은 과학 실행 한도는 direct620.7550000000002초/shared628.5429844초/Original3336.9150391004946초다. 추가 자원 배정이나 환급은 없다. 실패한 F를 재실행하거나 F math/gate/body로 진행하지 않는다.

다음 G 수정은 경로 처리와 종료/실패 정산을 분리해 같은 문제가 다시 정산을 막지 않도록 한다. ROOT 설계를 실제 Astra/max에게 리뷰받고, 요구 A1-A7을 docs/BANC_M3_G_QUALIFICATION_REPAIR_ROOT_REFLECTED_20261008.md14189/66924e4d50c435bdf36e9712357e0e8e78b2e0b20ad63fa660ee69195baa8ed5에 반영했다. 실제 GPT6.1Sol/max가 별도 FIRST 소스 제안을 구현 중이다. 기존 재배분·한도·모든 비용을 유지하면서 G 포인터 하나만 추가하고, 전체19 math와21+32 gate를 통과하기 전에는 Original 신체를 차단한다. 긴 경로의 텍스트 I/O를 수정하며 크기·해시·전체 readback·경로 포함·reparse 검사도 유지한다. G 적용/등록/검사/신체는 아직 NOT_RUN이다.

F1-F8 목표는 ACTIVE/INCOMPLETE다. candidate-k 미채택, Native VJP 미검증, 생물학적 자연 보행 미검증이며, M2/UI·미사용F6·BANC 신경 해부학적 provenance·root GOAL은 유지했다. 다음 실제 신체 실행에서는 이미 추가한8개 준비 구간의 시간과 startup/주기/slip/균형을 측정해 판단한다.
