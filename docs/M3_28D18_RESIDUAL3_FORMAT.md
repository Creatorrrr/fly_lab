# D18 잔여 수정 03: current 계약 JSON의 명시적 공백 축소 설계

실제 Astra f93d66 검토와 ROOT 반영 e2601ac8을 보존한다. ROOT 첫 private c5 후보4fd118/exit0은135909 bytes로131072 cap을4837 bytes 넘었다. 이는 실패한 fit이며 운영·프로젝트 실행은 없었다. 첫 private 후보와 작성 소스/결과는 보존하고 채택하지 않는다.

새 제안은 c5의 마지막 top-level `actual320` value JSON만 공백 없는 separators=(',',':')로 직렬화하는 것이다. 그 값 밖의 bytes는 첫 private 후보와 완전히 같다. 이 값 안의 모든 필드/배열/문자열/순서/자료형과 숫자 bits를 유지한다. 어떤 필드나 검증 규칙도 제거하지 않고 Source/c5 cap을 높이지 않는다. 과거 원문·Sources·reviews·이전 계약은 그대로 보존한다. 은밀한 일반 압축이나 임의 Source 정리는 없다.

ROOT own stdlib 측정5b0622/exit0의 `residual03/contract-whitespace-design.json`은 원문135909/11f2a1fd, pretty value27769→compact19226의8543 bytes 감소, 제안127366/5bdaf900d68b8f0eae6e4653ac15062369fab59644c25fe37f8e387ebc28f355, headroom3706을 확인했다. 제안 전체의 JSON 파싱 결과가 첫 후보와 모든 키 순서·유형·배열·binary64에서 동일하고, value 밖의 prefix/suffix 및 전체 역변환이 원문과 동일했다. 아직 compact candidate를 쓰거나 실제 parser/Native를 실행하지 않았다. own stdlib JSON 검사는 실제 production reader PASS를 대신하지 않는다.

새 리뷰 범위는 이 명시적인 한 value의 JSON 공백 변경뿐이다. 실제 Source-bound c5 pin의 모든 독립 compiled anchors·review/index를 최종 완성 hash/bytes에 재결합해야 한다. 일반 semantic comparator, Metadata prefix/caps, Source before guard, cache/compiler/backend, 기존 같은133 reason 및 모든 negatives는 유지한다. complete11 fit·typed/raw forward/inverse·actual SourceTRUE/FALSE closure가 선행한다. Runtime 추가 의존 파일이나 새 Source member를 만들지 않는다.

새 review/ROOT 반영은 `residual03/contract-format/`의 별도 설계 증거로 보존하고 기존 f93d66 봉인을 변경하지 않는다. 실제 final 보고서와 최종 독립 closure가 이 보완 pin을 명시 인증한다. 현재 residual_provenance의 기본4는 그대로이고 supplementary raw evidence는 future hash나 fifth residual member로 역기입하지 않는다. 실제 채택/Native/r4/보행은 모두 미실행이다. 일반 승인 요청 없이 기존 budget/범위 내에서 진행한다.
