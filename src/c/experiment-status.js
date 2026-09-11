/* Read-only interpretation of the current frame; never changes neural input. */
(function(F){
'use strict';
// Display thresholds discard the subnormal tail of the activation filter.
const EPS=1e-7;
F.neuromuscularActivity=function(f){
 const loop=f?.neuromuscular;
 if(!loop?.enabled)return null;
 const muscleRates=(loop.muscles||[]).flatMap(row=>Object.values(row.muscle_rates_Hz||{}));
 const tendonRates=(loop.tendons?.last||[]).flatMap(row=>(row.sources||[]).map(source=>{
  const rates=source.rates_Hz||[];
  return rates.length?rates.reduce((sum,v)=>sum+v,0)/rates.length:0;
 }));
 const maxAbs=values=>values.length?Math.max(...values.map(v=>Math.abs(v))):0;
 const tendons=f.physics?.tendons;
 const modes=tendons?.neural_controls?.modes||loop.tendons?.modes||{};
 const neuralInputs=[],manualInputs=[];
 for(const [i,name] of (tendons?.names||[]).entries()){
  if(modes[name]==='neural')neuralInputs.push(tendons.inputs?.[i]);
  if(modes[name]==='manual')manualInputs.push(tendons.inputs?.[i]);
 }
 const available=muscleRates.length>0&&Array.isArray(loop.offset_rad)&&loop.offset_rad.length>0&&
  (!loop.tendons||!!tendons)&&(!tendons||(tendons.names||[]).every((name,i)=>
   ['neural','manual'].includes(modes[name])&&Number.isFinite(tendons.inputs?.[i])))&&
  [...muscleRates,...tendonRates,...(loop.offset_rad||[])].every(Number.isFinite);
 const motorRateMaxHz=maxAbs([...muscleRates,...tendonRates]);
 const jointOffsetMaxRad=maxAbs(loop.offset_rad||[]);
 const neuralTendonMaxInput=maxAbs(neuralInputs),manualTendonMaxInput=maxAbs(manualInputs);
 const metrics=`운동 집단 발화율 최대 ${motorRateMaxHz.toFixed(3)} Hz · 신경 관절 목표 변화 최대 ${jointOffsetMaxRad.toFixed(4)} rad · 신경 힘줄 입력 최대 ${neuralTendonMaxInput.toFixed(4)} (모델 단위)`;
 return {available,motorRateMaxHz,jointOffsetMaxRad,neuralTendonMaxInput,manualTendonMaxInput,
  hasNeuralInput:jointOffsetMaxRad>EPS||neuralTendonMaxInput>EPS,
  hasMotorSpikes:motorRateMaxHz>EPS,hasManualInput:manualTendonMaxInput>EPS,
  message:available?metrics:'운동 활성 측정값을 기다리는 중입니다.'};
};
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
 if(f.physics?.attachment==='tethered'||f.physics?.wholeBody?.root_fixed)warnings.push('현재 몸통이 고정되어 공간 이동이 불가능합니다. 자유 이동은 몸통 설정을 바꾸고 새 실험을 만들 때 적용됩니다.');
 const activity=F.neuromuscularActivity(f);
 if(activity?.hasManualInput)warnings.push('수동 힘줄 입력이 적용되어 있습니다. 신경 출력과 별도로 몸을 구동합니다.');
 let state,message;
 if(f.fault){state='fault';message='실험 오류로 중단: '+f.fault;}
 else if(f.stopped){state='stopped';message='실험 중지 상태입니다.';}
 else if(!f.simTime){state='ready';message='아직 계산 전입니다. 재생 또는 한 단계로 시작하세요.';}
 else if(command.motor_coupled===false){state='disconnected';message='신경–몸 연결이 차단되어 운동 출력이 몸에 적용되지 않습니다.';}
 else if(activity){
  if(!activity.available){state='joint_pending';message='신경→관절 경로는 연결됐습니다. 현재 운동 활성 측정값은 아직 없습니다.';}
  else if(activity.hasNeuralInput){state='joint_output';message='신경 관절 목표 변화 또는 힘줄 입력이 적용 중입니다. 자세 유지·국소 반응과 보행은 구분해서 확인하세요.';}
  else if(activity.hasMotorSpikes){state='zero_joint_command';message='운동 집단은 발화하지만 신경 관절 목표 변화와 힘줄 입력은 0입니다. 상쇄·부착 제어와 출력 변환을 확인하세요.';}
  else {state='silent_joint_motor';message='신경 경로는 연결됐지만 현재 운동 집단 발화와 신경 구동 입력은 0입니다. 재생은 계산을 진행하며, 자동으로 보행 자극을 만들지 않습니다.';}
 }
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
