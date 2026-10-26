# E190 실행 전 보충: 서로 다른 힘 집계 순서의 정확한 재현

원본 값·원본 분석을 읽기 전에 부모가 소스를 대조해 원래 설계의 force exact 비교 전제를 정정했다. 원래 설계와 v1 freeze를 보존한다. E157은 `analyze_c_banc_m3_signed_contact.py`에서 각 contact의 `sign*(frame.T@wrench[:3])/weight0`를 먼저 계산하고 leg에 더한다. E185는 `observe_c_banc_m3_ground_reaction.py`에서 물리 단위 contact force를 segment에 더한 다음 다섯 segment를 합하고 마지막에 weight0로 나눈다. 같은 기계 상태여도 이 float64 집계 순서는 비트 차이를 낼 수 있다. 원본 관측 차이를 본 뒤 허용치를 넓히는 수정이 아니다.

E157의 `observed-contacts.npz`(24836개 전체 관측 contact, offsets10001)를 v2 입력에 추가한다. native_index/offsets/원래 contact_id 순서·leg/segment/sign/frame/wrench/weight0를 검증하고 두 집계 순서를 각각 그대로 재구성한다. E157 방식 결과는 원래 E157 force_world_bw와 exact, E185 방식 결과는 E185 segment_force의 공통10000표본 및 leg_force_bw와 exact여야 한다. normal이0인 관측 contact도 생략하지 않는다. 분모 weight0는 E157 layout의 양수 유한 값을 사용하고 E185 ground-reaction-metadata.json에 기록된 weight0와 exact 대조한다. 이 검사가 실패하면 원래 방침대로 invalid/not-tested다.

두 원래 force_world 배열의 불일치 개수·최대 절대 차이를 별도로 남겨 단순 exact equality라고 보고하지 않는다. 공통 native 시간·회전·heading·하중·slip·mask·비발 접촉·upright와 E161/E186 command 비교의 exact 요구는 유지한다. E185 force.available/finite/error와 E157 completion flags도 모두 정상이어야 하며 두 body hash를 대조한다.

새 분석의 canonical AP는 E185 `leg_heading_xy_force_bw[10000:20000,4]`로 고정한다. E157 원래 planar AP도 원자료에 보존한다. pitch가 포함된 `leg_forward_force_bw`로 대체하지 않는다. 이렇게 같은 E185 사건·반력 정의를 유지하며 E161 수직 분해와 결합한다.

E190 최초 구현 source/test snapshot 뒤 실제 같은 gpt-6-sol/medium의 첫 수정에 이 부모 설계 정정을 포함한다. shape/time/force-order 및 실제 저장 실패와 무접촉 bin 변화 회귀를 합성 자료로 검증한다. v2 freeze/source review 후 원본 CPU 분석1회만 실행한다. 원본 runs0/retry0, 이전 비용·25/35/40초·CPU1/GPU0·32MiB·F1–F8·M2/UI/F6 조건은 그대로다.
