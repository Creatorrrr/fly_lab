# E358 후보 계산과 동일 후보 신체 실행 결과

2026-10-03. 전체 M3/F1–F8은 미완료다. 후보의 저장 입력에 대한 신경 응답은 조건부 개선했지만, 동일 후보의 실제 신체 실행은 독립 자격을 얻지 못했다. M2와 UI에 채택하지 않았다.

후보 계산은 실제 전체 반환23.289초, 별도 raw 감사2.372초로 실행 당시 source snapshot 아래 유효했다. 실제 f32 계수 변경, native palette 한도, 신경 work 및 저장·감사 조건을 확인했다. R 개선0.00887903942174223, O/Q 상대 오차0.0009343804209771942, O/T0.0009435436453516087이었다. 이는 f64 대리 모델과 실제 f32 응답의 조건부 일치이며, native f32 전체 Jacobian이나 실제 보행 개선을 증명하지 않는다.

신체 실행은48.295초/exit1이었다. 실제99 controls,1980 neural ticks,4950 physics steps와100 boundary rows까지 반환했다. 생산자 startup 보고는 q RMS0.017792953582825483, q max0.04220501637004681, tendon RMS0.0013636523143322314, tendon max0.013189925135820152이었다. 고정 q max0.04를 초과해0.495 모델초에서 중단했다. 이후 imported-source closure 검사가 실패해 manifest publication을 마치지 못했으므로 **technical_invalid/unqualified**이다. 이 수치는 독립 body 감사 결과가 아니며, full1.5초·10초·F1–F8을 통과한 결과로 사용하지 않는다. 관측된 모든 소유 프로세스는 종료했고 예약을 해제했다.

ROOT는 등록된 installed-source absolute path와 같은 파일의 ROOT-relative import key 사이47개 alias 누락을 소스에서 찾았다. 같은 등록 SHA의 동일 파일 alias만 인정하고 collision을 거부하도록 수정했다. 이후 실패에서는 actual imports와 mismatch를 먼저 남긴다. 이 수정 후19개/57phase 회귀와 별도 실제 supervisor→저장·fsync·종료→reread→실제 반환 fixture가 닫혔다. 실패한 원본 body를 반복하거나 소급 유효화하지 않았다. 실행 당시 adapter/test source snapshot을 별도로 보존했다.

기존 원본9893.754960899503초에 E358 실제78.18초를 더한9971.934960899503/10200초, 공동stage78.18/300초를 사용했다. 공유 생성 누적99.7460156/150초이며 모든 실패와 helper 비용을 포함한다. 변경 후보 retry는 사용하지 않았다. 다음 단계는 남은 같은 예산에서 theta=1 기준 모델의 startup99 가능성을 별도로 사전 등록해 확인하는 [E359 계획](BANC_M3_E359_PAIRED_STARTUP_BASELINE_PLAN_20261003.md)이다.

기계 판독 결과와 실행 당시 provenance: [ROOT 결과](../verification/banc-natural-gait-m3-20260919/parent-e358-root-results-before-e359.json), [실행 원장](../verification/banc-natural-gait-m3-20260919/parent-e358-execution-ledger.json). 실제 goal API 상태 blocked와 최신 사용자 요청에 따른 현재 작업 재개를 구분한다.
