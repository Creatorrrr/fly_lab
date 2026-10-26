# Native scalar 의미 검토 — ROOT, 2026-10-08

이는 Native 비용 설계와 독립적으로 ROOT가 확인한 primary 문서와 Source 검토다. 보호 실행, Add-Type compile, Python import 또는 scalar scratch 검사는 하지 않았다. local equivalence와 속도 개선은 NOT_RUN이다. 설계 원형7858 bytes/SHA b61b977199c5844ea8802270c3735c51399c326539c24b870e60d93745c833f1와 실제 z 결과는 수정하지 않는다.

PowerShell 공식 문서는 string 비교가 case-sensitive 연산자에서도 InvariantCulture를 사용한다고 설명한다. 따라서 값의 -ceq를 단순 Ordinal byte 문자열 비교로 교체하는 것은 이번 변경의 전제가 될 수 없다. Scalar runtime type equality를 먼저 확인한 뒤 PowerShell의 비교 의미를 그대로 사용하는 방법을 검토해야 한다. [PowerShell comparison operators](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_comparison_operators?view=powershell-7.6), [LanguagePrimitives.Equals API](https://learn.microsoft.com/en-us/dotnet/api/system.management.automation.languageprimitives.equals?view=powershellsdk-7.4.0).

현재 Native account Same의 dictionary 의미는 `$a.Contains($key)`이며 Prepare Contact-SameTyped는 별도의 Ordinal key 집합을 만든다. 두 함수를 같은 의미라고 가정해 합치지 않는다. 우선 account Same의 정확한 의미만 다루고 다른 comparator를 변경하지 않는 편이 scope가 작다. Key 순서를 정규화한 JSON serialization 비교는 scalar runtime type과 array order 또는 IDictionary comparer 의미를 잘못 대체할 수 있으므로 사용하지 않는다.

지원 타입 밖의 PSObject wrapper, 사용자 정의 객체, non-string dictionary key, multi-dimensional/nonzero-bound array, cycle/depth/node 한도 및 compiler failure는 success로 취급하지 않는다. 최적화가 적용되지 않은 baseline fallback과 compiler가 실제로 작동한 fast path를 구분해야 한다. Semantic test만 통과하고 계속 fallback을 사용한 경우 성능 개선 증거가 없다. 이미 로드된 임의의 type 이름을 새 Source와 동일한 compiled helper라고 재사용하지 않는다.

기존 scalar -ceq와 동치 여부가 아직 확인되지 않은 NaN/Infinity/negative zero/Unicode normalization 및 embedded null에는 Source 검토만으로 true를 만들지 않는다. 지원 범위를 엄격하게 제한하고 원래 경로로 돌아가거나, actual owned scratch에서 원래 comparator와 동일한 accepted/rejected 결과를 확인해야 한다. Json metadata finite 규칙을 comparator가 대체하지 않는다. 실제 counts/budget의 Int64/Double/Bool 규칙은 원래 guard가 그대로 적용한다.

ROOT가 요구할 Source 리뷰 항목은 완전한 노드 방문, scalar 타입 선행 검사, null/empty distinction, array 순서, dictionary key comparer 보존, bounded depth/node 방문, 오류의 원래 진단 복원, fallback 동작, 실제 fast-path witness, lazy compile 비용 및 기존 Source extraction compatibility다. 새 파일 I/O, history proof cache, Original payload 접근, native grant/reset, 후보 또는 scientific body 변경은 금지한다.

실제 Source-x JUnit의18.313초/22.381초는 전체 Native group 시간이며 comparator만의 시간이 아니다. Compiled 대조가 비용을 줄일 것이라는 판단은 가설이다. Read-Pin/hash와 중복 budget validation이 우세하면 이 변경만으로 full14/30 실행 계약을 회복할 수 없다. 원래 전체 gate68 timeout을 숨기거나 완료로 바꾸지 않는다. 자원 부족 때문에 검증되지 않은 proposal은 Source-only/UNQUALIFIED로 유지한다.
