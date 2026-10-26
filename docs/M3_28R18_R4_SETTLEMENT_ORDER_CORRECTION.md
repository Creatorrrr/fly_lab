# r4 정산 helper의 기존 요청 descriptor 연결 수정

실제 candidate-v2 metadata invocation은2df3d1/exit1,1.469395초에 `R4 metadata proof: dictionary order`로 진입 guard에서 거부됐다. actual-refusal.json에 원문 반환을 보존했다. controls stdout과 공통 transaction 진입 전이었다. 후속 stdlib 실제e3e9df/exit0은 canonical288c 불변,cost absent,lock absent를 확인했다. 과학 worker나r4를 다시 실행하지 않았다. 이번 metadata invocation의 실패를 성공으로 바꾸거나 숨기지 않는다.

실제 canonical admitted/pending request와 stored outer request의 키 순서는bytes/path/sha256이고, genuine trace와 capsule은path/bytes/sha256이다. 값과 유형은 같다. Astra R4S-C3가 요구한 기존 원문 보존 및 typed semantic 연결에 따라 이 세 연결만 R4-Whole의 ordered=false로 비교한다. 각 원문 전체와 전달 outer 자체의 trusted raw 비교는 여전히 원래 키 순서까지 검사한다. request의 exact3 fields,bytes의int/bool 경계,alias/hash/size,배열 및binary64 guards는 유지한다. 재정 원장의 전체 before/after/inverse 순서 보존은 변경하지 않는다.

ROOT만 새 private candidate-v3를 작성하고 전체 세 구역 inverse 및Source871/current13을 다시 리뷰한다. 새 메모리 controls와metadata commit으로 한정하며, 과거 operating Source와review,actualr4request/trace/outer/wrapper,실패candidate-v2는 불변이다. known no-write 거부를 확인한 뒤의 helper 수정 실행이며 unknown/uncertainty나r4 worker의 자동 재시도가 아니다. 비용525.537/실패HOLD/no-successor/690 및Original NOT_RUN/F1–F8 ACTIVE/INCOMPLETE는 그대로 유지한다.
