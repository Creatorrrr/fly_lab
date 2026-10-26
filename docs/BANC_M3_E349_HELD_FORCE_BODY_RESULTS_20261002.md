# E349 실제 기준 재현 결과

기준 B는300회 제어·15000 native 물리·constructor2000 settling을 끝냈으나 upright 바이트 불일치로 기술적으로 거부되었다. I/C는 실행되지 않았으며 H348·보행 개선 가설은 판정하지 않았다.

독립 once 감사에서301개 qpos/qvel/ctrl/관절 명령과 나머지5개 native 필드(time/load/slip/slip_valid/nonfoot)는 유한 바이트 및 NaN 위치가 정확히 일치했다. Upright15000개 모두 다르고 최대오차0.0007827991786664157이지만, 원본의 다음 샘플은 재현의 이전 샘플과14999개 바이트가 정확히 같다. 최소upright는 양쪽0.9253928135991221. E335의 live-bound pose 인스턴스 wrapper를 shadow copy가 그대로 보유한다는 소스 추론과 일치한다. 이는 기존 fidelity 조건을 완화할 허가가 아니다.

외력300회 전후 실제 관측은 모두0이며 원본 neural_request null은 보존된다. 실패한 완전B 출력7781744B는 실패4MiB 상한을 넘는 추가 기술 오류도 발생했다. 실패 자료와 한도 위반을 보존했고 통과로 간주하지 않는다.

실제 inclusive body16.11542470000859초와 audit0.5870140000042738초를 각각 한 번 계상했다. 누적9798.25293289951/9900초, 예약·실행job0, retry0. E350에서 관측 의미를 원본과 맞추고 decoded JSON 값은 그대로인 compact trace 출력을 설계한다. F1-F8·자연/생물학적 보행·제품 채택은 미완료다.
