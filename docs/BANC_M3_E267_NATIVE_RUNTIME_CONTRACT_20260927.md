# E267 전체 BANC 중립 재현 계약

새 source 여과 런타임의 CUDA 생성 검사와 부모 독립 coupled RK4 계산은 통과했다. FIRST의 inference tensor 할당 결함과 free MN roster 검증 누락은 같은 gpt-6-sol/high가 수정했다. SECOND는 생성 CUDA 2테스트와 독립30검사 통과이며 부모 제품 잔여 수정은 없다.

이번 원본은 보행 후보 탐색이 아니라, 새 구현을 실제 158,706개 신경/11,584,852개 간선에 이식할 수 있는지 확인하는 단일 신경 재현이다. 기존 E265 B의 W0와 control0 신경 상태, 100control 감각/하행 drive를 고정한다. 선택한 기존1660간선 계수를 모두[1,0,0]으로 하고 source별 두 이력만 수동으로 적분한다. 실제 CUDA packed graph를 사용한다.

100controls/0.5초의 기존1402개 rate/adaptation/drive와 전체 drive SHA, 원래 전체 시작·끝 상태가 bit exact여야 한다. 실제 packed CSR와 원래 세포 모수도 검사한다. 새 51×2 이력은 매 endpoint와 전체 checkpoint에 저장한다. 끝상태를 disk에 저장하고 같은 인스턴스에서 held final drive .005초를 진행한 뒤 disk복원→같은.005초의 모든 상태가 exact여야 한다. 이는 새 인스턴스/신체를 포함한 최종 F2의 대체가 아니다.

과학2040 neural ticks = 기준2000 + 두 미래20씩. fullBANC constructor1, 생성 setup은 별도 `_interval` 호출별 실제 단계수와 capture 여부를 기록한다. body/physics/solver/fit/F6/활성 UI 변경0, 후보 채택0. 불일치·비유한 상태·resource limit이면 원자료와 최초 실패를 보존하고 중단한다. 원본 자동 재시도0이다.

E265의 신경2모형/3팔0.5초는24.153732초였다. 새 구현은 한 모형이지만 CUDA 초기화와 새 이력/스냅샷 비용을 포함해 internal50/외부child55초, 감독 정상 종료 비용을 포함한 예약55초로 한정한다. 원본 누적7126.338238699598초는 유지하고 미래 단계 배정을7138→7182초(+44)로 갱신한다. 잔여55.661761300402초에서55초를 예약한다. 사용자 전체 예산 신설이나 누적 초기화가 아니다. 실제 감독 비용을 기존 누적에 더하며, 생성 CPU/CUDA·부모 검산 비용은 별도 영수증을 유지한다. RAM tree6GiB/output128MiB를 넘으면 중단한다.

원본 실행 전 현재 소스·선택·CSR·신경 초기/끝·drive·비교 배열과 M2보존 hash를 `parent-e267-native-contract.json`에 고정하고 부모 SECOND 판정과 함께 확인한다. 원본을 통과해도 보행 학습 및 실제 신체 .5→1.5→10초와 같은 후보의 나머지 F1–F8은 미완료다.
