# E178 이후: 단일 무발화 세포와 전체 근육 전달을 구분

E178에서 계산·저장·단위 오류는 발견하지 못했고 두 무발화 RM tibia-flexor MN의 큰 억제는 해부학적 IN13A001 source와 기존 individual fit 값으로 설명된다. 같은 경계에서 그 항을 제외해도 나머지 입력은 threshold보다 작다. 따라서 이 두 MN의 억제를 임의로 지우거나 같은5개 감각 연결을 더 키우는 반복은 우선하지 않는다.

다음 E179의 판별 질문은 **두 MN의 silence가 실제 tibia-flexor pool와 지지→스윙 전환의 부족을 의미하는가, 아니면 이미 다른 MN이 출력을 담당하는가**다. 이 질문이 해결되어야 흥분성/억제성 전달의 새로운 수정 위치와 필요한 효과를 정할 수 있다. E175의 제한된 양수 history readout 가족 실패도 유지하며 같은 가족을 다시 fit하지 않는다.

우선 기존 E177 저장 full391 MN rate, bank, muscle activation/원래 activation, adapter의 정확한 muscle membership/pooling/monotone weights와 native foot contact·관절 명령을 같은 control tick으로 연결하는 범위를 부모가 정한다. RM tibia-flexor5개 MN을 이름·root ID로 확인하고 정상 출력3개 및 무출력2개의 실제 pool 기여, 필터·근육 활성과 foot unload 전환을 분리한다. 필요하면 이미 저장된6다리 전환 진단과 대조하되 없는 신경 상태·힘을0이나 추정값으로 채우지 않는다. 이전 source가 이미 같은 질문에 답했는지 먼저 확인해 중복 분석을 피한다.

고정 판정은 다음과 같다. whole-muscle 출력이 충분한데 foot가 들리지 않으면 신경 발화 회복을 목표로 삼지 않고 해당 순간의 다관절 전달을 검토한다. whole-muscle 출력이 실제 부족하고 upstream 여유가 확인되면 해부학적 부호·노드·연결을 보존하는 새로운 모집/입력 모델의 한정된 가설을 설계한다. 저장 관측만으로 전환 인과를 구분할 수 없으면 필요한 가장 작은 조건부 관측을 별도 사전 계약으로 정한다. source 억제 제거, 새로운 gain sweep, 신경/물리 실험은 이 문서만으로 실행하지 않는다.

E179는 방향 초안이며 freeze/source gate/새 원본 분석/신경·물리·fit은 아직0이다. 다음 구현도 부모 설계→actual gpt-6-sol(난이도별 effort)→부모FIRST→같은Sol첫수정→부모SECOND/잔여수정이다. 전체 목표 active, F1–F8 미완료, M2/UI/F6 보존, 누적3420.470999800018/3656초를 승계한다.
