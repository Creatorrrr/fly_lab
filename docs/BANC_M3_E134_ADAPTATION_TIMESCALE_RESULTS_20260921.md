# E134 결과: 근육별 적응 시간상수의 CUDA 계산 검증

2026-09-21 KST · H134 supported · 조건부 계산기 채택 · M3 F1–F8 미완료

전체391 운동신경/96근육에서 근육별 적응 시간상수와 그 감도를 계산하는 새 연구 모듈을 구현했다. 고정된 세 입력 문맥에서 CUDA와 독립 NumPy의11출력 최대 차이는 5.684342e-14, 정류 branch 차이는0이었다. 이 결과는 다음 제약 적합에 사용할 계산기의 검증이다. 실제 자연 보행이나 생물학적 기전의 검증으로 해석하지 않는다.

## 변경과 검증

기존 0.3초 고정 적응에 ell=log(T/0.3)을 추가했다. 근육별 T의 공학적 탐색 범위는[0.1,3]초다. 기존 membrane tau/기록 네 단계 전류/기존 연결 provenance/초기상태와 decoder를 유지하고, edge1572+gain96+logtau96의1764개 조건부 감도 열을 제공한다. 기존 모듈 전역 패치는 사용하지 않는다.

실제 Sol/gpt-5.6-sol/xhigh가 새 신경 동역학과 CUDA 감도를 구현했다. 부모 설계→Sol 구현→부모 첫 리뷰→같은 Sol 첫 수정→부모 두 번째 리뷰 순서로 완료했다. 두 번째 리뷰에서 잔여 소스 수정은 필요 없었다. 부모 합성11개/Ruff, 독립 fake IO/bounds probe, 작은 행렬지수/Frechet 참조가 통과했다. 초기 합성 실패와 lint 실패를 포함한 모든 시험·소스·비용은 보존했다.

첫 리뷰에서는 반환 후 raw 저장 오류가 알려진 진행 기록을 없애는 문제, log(.1/.3)와 -log(3)의 표현 차이로 원래 freeze를 거절하는 문제, stage별 분기 진단 누락을 수정했다. 실험의 고정 경계값이나 수치 기준은 바꾸지 않았다.

| 검사 | 결과 |
|---|---|
| ell=0 기존8출력 재현 | 최대 차0 |
| 기존 B1668열 | 최대 차8.881784e-16 |
| E132 LF 기록과 rate/activation/B 초기99/branch | 차0 |
| 최종 h=1e-8 시간상수/gain/edge 감도 | 모든 대상 field 통과 |
| 독립 NumPy11출력 | 최대 차5.684342e-14, branch exact |

큰 h=1e-5의 tau FD는 tendon, rate에서 실패했다. +/-의 분기 변화 neuron-tick은 E11016/E11311/E12110이었다. h=1e-6은1/0/0, h=1e-7과 사전 고정 최종 h=1e-8은0/0/0이다. 큰 h 실패를 지우거나 다른 h로 판정을 사후 교체하지 않았다. 모든 원본 시각·세 문맥·391신경·96근육을 포함했다.

## 실행·보존

RTX4080/CuPy14.2.0 실제 CUDA에서15 batch/45 context response/450조건부초를 한 번 실행했다. 외부시간 35.568212초, supervisor 36.969014초. 독립 NumPy는30조건부초 한 번, 외부 19.458499초였다. CPU/BLAS/Torch thread1, 원본 재시도0, 요청15/반환15/미확정0/반환후오류0이다. 별도 tiny2batch/40tick을 기록했다. 실제 full158706 BANC/신체/geometry/fit/F6노출은0이다.

Windows venv 실행기와 실제 Python worker PID를 구분했다. CUDA 실행기14720, 참조 실행기32432/참조 worker47424를 관측했다. 별도 작은 프로세스 probe에서 같은 venv의 launcher→worker 관계를 확인했으며, 과거 worker 부모 PID를 새로 관측했다고 주장하지 않는다.

CUDA 원본21파일, 참조2파일을 봉인했다. 부모 CUDA review505검사/58해시와 최종 검토가 통과했고 M2의263파일이 보존됐다. 기존1516.4284439000185초 누적과 종료된1530초 allocation은 그대로다. 새 H134 계약의 보수적 누적은 1573.549034/1742초다. 준비·합성·리뷰 비용 및 실패도 별도 기록에 남겼다.

## 다음 개발

초기99 제약 아래 실제 목적함수를 더 줄일 수 있는지 먼저 확인한다. E134 ell0의 LF 감도와 E132의 동일한 응답/손실 cotangent를 사용하면 새 신경 재실행 없이 국소 개선 여지를 평가할 수 있다. 이 검사는 새 계약을 동결한 뒤 수행하며 아직 E135 원본을 시작하지 않았다. 유효한 방향이면 시간상수를 포함한 제약 적합, 전체391과 native158706 상태 검증, 같은 후보의 신체/F1–F8 검증으로 이어간다.

근거는 `verification/banc-natural-gait-m3-20260919/parent-e134-final-review.json`, `parent-e134-cuda-completion-review.json`, `e134-center-parent/result.json`, `e134-adaptation-timescale/`와 두 supervision manifest다. 새 소스는 `tools/banc_m3_adaptation_timescale.py`, `tools/verify_c_banc_m3_adaptation_timescale.py`, `tests/test_c_banc_adaptation_timescale.py`다.
