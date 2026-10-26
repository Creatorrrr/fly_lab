# E323: 패키지 실행 명령 회복

E322는 root가 명령을 python script.py로 구성해 flylab package import가 실패했다. 이전 E320은 python -m tools.verify_c_banc_m3_full_horizon_direction으로 실행됐다. 이 실행 명령 오류를 root가 인정하며 실패 source/output와 감독3.277557599998545초를 모두 보존한다. 원본누적8580.353065599633/9060초, 신경tick/신체/optimizer0. CUDA 초기화 회복과 full-horizon 과학 가설의 원본 결과는 아직 미판정이다.

E323은 E322의 신경/decoder/loss/direction/meter 구현을 그대로 쓰고 명시적인 실험 ID와 fresh output만 바꾼다. 이 치환을 역으로 수행하면 E322 source와 byte-text가 일치하는지 검증한다. 실질적 기술 수정은 부모 실행 명령을 이미 사용했던 -m 모듈 호출로 되돌리는 것이다. 환경 또는 설치 변경은 없다. 사용자 지정 SECOND 이후 잔여 수정으로 root가 처리한다.

생성 CPU 10초 안에서 정확한 -m CLI의 help와 local package 검색 경로를 확인한다. E322의 18검사 및 실제 meter부터 시작한 생성 CUDA full-BPTT 통과는 동일 기능 source 증명과 함께 이전한다. 실험 이름만 바뀐 부분에 대한 CUDA 반복은 하지 않는다. 새 passive freeze/review는 모든 source/input hash와 증거를 검증한다.

원본 단회 감독465초를 남은479.646934400367초에서 예약한다. 기존 코드의 상한480초보다 좁고, 누적9060초를 늘리지 않는다. 내부450초/과학410초/게시40초, 전체감각300control와6000tick미분, 고정방향±0.1%, 실제방향오차≤5%, 모든 startup/중립재현 기준은 E320/E322와 동일하다. 기존 실패를 숨기거나 같은출력을 덮어쓰지 않는다. 감독 종료 후에만 결과/원장을 읽고 실제 감독시간을 한 번 합산한다. F1–F8, M2, F6미노출은 그대로다.
