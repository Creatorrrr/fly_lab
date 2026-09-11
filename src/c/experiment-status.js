/* Read-only interpretation of the current frame; never changes neural input. */
(function(F){
'use strict';
F.describeStrictExperiment=function(f,playing){
 if(!f||f.mode!=='C_STRICT')return {state:'other_mode',message:'',warnings:[]};
 const warnings=[],ports=f.sensory_ports||[],command=f.command||{};
 if(f.binding?.chemical_odorants?.length){
  const unnamed=(f.world?.sources||[]).filter(s=>s.strength>0&&!s.odorant&&(s.kind!=='food'||f.world.foodOn));
  if(unnamed.length)warnings.push(`물질 미지정 냄새원 ${unnamed.length}개: ${unnamed.map(s=>s.id).join(', ')}. DoOR 후각 입력에 반영되지 않습니다. 환경 편집에서 냄새 물질을 지정하세요.`);
 }
 const odor=f.sensor_diagnostics?.four_site_odor;
 if(odor?.concentration?.length&&f.world?.sources?.some(s=>s.strength>0&&(s.kind!=='food'||f.world.foodOn))&&
    odor.concentration.every(row=>row.every(v=>v===0))){
  warnings.push('현재 후각 지점의 냄새 농도가 0입니다. 냄새원 위치와 바람 방향을 확인하세요.');
 }
 if(f.physics?.attachment==='tethered'||f.physics?.wholeBody?.root_fixed)warnings.push('몸통 고정 실험입니다. 이동 거리 대신 관절·힘 반응을 확인하세요.');
 let state,message;
 if(f.fault){state='fault';message='실험 오류로 중단: '+f.fault;}
 else if(f.stopped){state='stopped';message='실험 중지 상태입니다.';}
 else if(!f.simTime){state='ready';message='아직 계산 전입니다. 재생 또는 한 단계로 시작하세요.';}
 else if(command.motor_coupled===false){state='disconnected';message='신경–몸 연결이 차단되어 운동 출력이 몸에 적용되지 않습니다.';}
 else if(f.neuromuscular?.enabled){state='joint_output';message='신경→관절 구동 경로를 사용합니다. 관절 출력과 실제 운동을 함께 확인하세요.';}
 else if(Object.values(command.u_final||{}).some(v=>Math.abs(v)>1e-7)){
  state='motor_output';message='C 신경 운동 명령이 발생하고 있습니다. 실제 이동·방향과 과제 성공 여부는 별도 확인이 필요합니다.';
 }else if(Object.values(f.motor_rates_Hz||{}).some(v=>v>1e-7)){
  state='zero_command';message='운동 출력 뉴런은 활성화됐지만 적용 명령은 0입니다. 상쇄·정지 게이트와 출력 변환을 확인하세요.';
 }else if(ports.some(p=>p.enabled&&Math.abs(p.value)>0)){
  state='silent_motor';message='감각 입력이 들어오지만 현재 운동 출력은 0입니다. 감각→보행 반응이 발생하지 않은 상태입니다.';
 }else{
  state='zero_input';message='현재 활성 감각 입력과 운동 출력이 모두 0입니다. 자극 배치·물질 지정·감각 차단 상태를 확인하세요.';
 }
 if(!playing&&f.simTime&&!f.fault&&!f.stopped)message='일시정지 · '+message;
 return {state,message,warnings};
};
})(globalThis.Fly=globalThis.Fly||{});
