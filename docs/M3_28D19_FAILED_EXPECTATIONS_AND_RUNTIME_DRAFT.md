> ROOT 정정2026-10-11: Source는 이미 minified여서 공백 축소 절감은1byte다. 새 helper key 추가안은 거부됐다. 현재 설계는 기존 mutable Python/account Source 파일에 좁은 역사검증 함수를 추가하는871-member zero-addition 안이다. 이 초안보다 `M3_28D19_RUNTIME_RECOVERY_AND_BOUNDED_GAIT_PATH.md`의 수정 설계가 우선한다.

# r4 실패 기대값과 실행시간 개선 설계 초안

사용자의 일반 승인 없이 판단·진행하라는 지시에 따라 ROOT가 다음 단계를 설계한다. 이 문서는 설계 초안이다. r4 정산 검토와 별도 범위이며, 아직 Astra 검토·운영 채택·새 프로젝트 실행을 뜻하지 않는다.

실제 r4는 두 FF를 출력한 후 소유된 시간 초과로 종료됐다. JUnit과 최종 pytest traceback은 없다. 첫 Native 하위 관측은 독립 기대 배열의 `original115 plus finite6` 길이 불일치를 기록했다. 실제 코드에는 새 세 ledger 경로가 이미 추가되어 finite9다. 두 번째 테스트는 실제 저장 fixture 생성 후 첫 전체 sequence 비교에서 아직 r3/321/180/160 기대값을 사용한다. 현재 Source에 묶인 sequence는 r4/322/540/520이다. 두 번째 원인은 `residual03/r4-second-static-diagnosis.json`의 실제508b21/exit0 정적 비교로 뒷받침되며 복구된 traceback으로 표시하지 않는다.

수정은 테스트의 독립 Native ledger 기대 배열에 정확한 새 세 경로를 순서대로 추가하고, 고정7개 역할 비교의 첫 호출 및 전체 sequence 기대의 첫 항목만 현재 설계값으로 바꾸는 것이다. original115의 순서·중복과 기존finite6은 그대로 둔다. 나머지6개 역할, 전체 비교, bool/int·순서·추가키 negatives, 타입/캐시/컴파일러 검사, 실패 history와 caps를 유지한다. 운영 코드의 맞는 값을 다시 기대값 생성기로 복사하거나 검사를 제거하지 않는다. 기존r4 Source/검토/실제 반환과 이미 소비된 실행은 보존하며 새 구현은 private candidate로 먼저 검토한다.

정정: 약445.1363초는 부모 ENTRY부터 첫 Native 관측까지의 누적 시간이며 Native 하위 프로세스의 실행시간이 아니다. Native PID는 부모 소유자 생성442.1806249초 뒤에 생성됐다. Native 전용 시간은 미확인이다. 실제 전체 first FINAL 비용525.537초는 변하지 않는다. pytest-1183 안의 기존s19 및s28 저널도 이전 소유자·이전 실행에서 복사된 자료이므로 이번 단계별 성능 근거로 사용하지 않는다. 다음 실제 검증에는 현재 Generated wrapper의 PID/생성시간, 실제 명령 경로, 단조 시각을 포함한 최소 단계별 관측을 남긴다. 고정 검사 수를 줄이거나 역사 증거 읽기를 생략하기 전에 실제 시간과 같은 실패/정상 결과가 보존되는지 확인해야 한다.

정산 후 generated local 예상 잔여는763.248691600001초, shared는771.0366760000006초다. 기존690 qualification과 close 예산을 유지한 채540초 고정4를 자동 반복하면 부족하다. r4를 새 이름으로 반복하지 않는다. 새 실행의 허용 조건은 오류 수정·다른 Source·읽기 전용 검토 완료·필요한 정확한 과거 paid/HOLD 보존·실제 잔여 내 고정 envelope·각 결과의 다음 행동이다. 기존 canonical failure HOLD를 실제 성공처럼 제거하거나 비용/한도를 리셋하지 않는다.

Decision fork: 정적 기대값 수정과 역변환이 맞으면 actual saved 입력에서 도출한 별도 candidate를 리뷰한다. 타입·순서·history 또는 Source 차이가 늘어나면 채택하지 않고 원인을 수정한다. 현재 자료로 실제 실행시간을 줄였다는 근거가 없으면 속도 향상을 주장하지 않고 새 실제 관측의 범위를 먼저 설계한다. 어떤 결과에서도 F1–F8은 ACTIVE/INCOMPLETE다. actual fixed4 PASS와 같은 후보의 qualification이 없으면690 이후 Original/body를 실행하지 않는다. 메타데이터 정산·테스트 수정·proxy 결과는 BANC 실제 신경 폐루프 보행이나 생물학적 자연성의 증거가 아니다.
