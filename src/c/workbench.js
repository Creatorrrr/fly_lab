/* Environment editor and independent observation / intervention / record cohorts. */
(function(F){
'use strict';
F.createWorkbench=function({state,rpc,view,applyFrame,tell,button,command,updatePlay,regions,renderSelection}){
 const $=id=>document.getElementById(id),colors=['#76d9c4','#eac684','#87b4e8','#c9a1ed','#ed9994','#a6d883','#88d9e5','#dba3c6'];
 const wb={intervention:[],record:[],chart:[],events:[],presets:[],pick:false,preview:false,gaps:0,
  trimmed:0,lastEnd:null,lastReceive:null,connectedAt:performance.now(),lastRegions:0,regionsPending:false,listOffset:0,extraItems:[]};
 function put(id,text){if($(id).textContent!==text)$(id).textContent=text;}
 function pause(){state.playing=false;updatePlay();}
 function label(id){return state.names.get(id)||id.replace('flywire:fafb:783:','');}
 function idsText(ids){return ids.map(id=>label(id)+'\n'+id).join('\n\n')||'지정된 뉴런이 없습니다.';}
 function cohortDisplay(){
  put('observe-ids',idsText(state.selected));
  put('intervention-count',wb.intervention.length+'개');put('intervention-ids',idsText(wb.intervention));
  put('record-count',wb.record.length+'개');put('record-ids',idsText(wb.record));
  $('observe-to-record').disabled=state.recording;$('preset-record').disabled=state.recording;
 }
 async function lookup(ids){
  const unknown=[...new Set(ids)].filter(id=>!state.names.has(id));if(!unknown.length)return;
  const rows=await rpc.request('lookup',{ids:unknown.slice(0,512)});
  rows.forEach(n=>state.names.set(n.id,(n.cell_type||'유형 미지정')+' · '+n.root_id.slice(-5)));
  cohortDisplay();chartPicker();renderSelection();
 }
 function chartPicker(){
  const key=state.selected.join('|')+';'+wb.chart.join('|')+';'+state.selected.map(label).join('|');
  if($('chart-targets').dataset.key===key)return;$('chart-targets').dataset.key=key;$('chart-targets').replaceChildren();
  state.selected.forEach(id=>{const row=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.checked=wb.chart.includes(id);input.disabled=!input.checked&&wb.chart.length>=8;row.title=id;row.style.color=input.checked?colors[wb.chart.indexOf(id)]:'#8ca5ae';row.append(input,document.createTextNode(label(id)));input.onchange=()=>{wb.chart=input.checked?[...wb.chart,id]:wb.chart.filter(x=>x!==id);chartPicker();drawSignals();renderSelection();};$('chart-targets').append(row);});
 }
 function resetSignals(){wb.events=[];wb.gaps=0;wb.trimmed=0;wb.lastEnd=null;wb.lastReceive=null;state.traces.clear();state.signals.clear();state.spikes=0;}
 function subscriptionChanged(){
  resetSignals();wb.chart=wb.chart.filter(id=>state.selected.includes(id));if(!wb.chart.length)wb.chart=state.selected.slice(0,8);
  chartPicker();lookup(state.selected).catch(e=>tell(e.message,true));
 }
 function acceptSignals(buffer){
  const {header:h,values,events}=F.decodeCSignals(buffer);
  if(h.subscription_epoch!==state.epoch||h.sequence<=state.sequence)return;
  if(state.frame?.mode!=='B_COMPAT'&&(h.ids.length!==state.selected.length||h.ids.some((id,i)=>id!==state.selected[i])))throw Error('구독 대상과 신호 대상이 다릅니다.');
  if(wb.lastEnd!==null&&h.start_tick>wb.lastEnd)wb.gaps++;
  if(state.sequence&&h.sequence!==state.sequence+1)wb.gaps++;
  state.sequence=h.sequence;wb.lastEnd=h.end_tick;wb.lastReceive=performance.now();
  const t=h.end_tick*h.neural_dt;
  h.ids.forEach((id,i)=>{const value=values[i];state.signals.set(id,value);let trace=state.traces.get(id)||[];
   if(!trace.length||trace.at(-1)[0]!==t)trace.push([t,value.voltage,value.rate]);
   const earliest=trace.findIndex(p=>p[0]>=t-15);if(earliest>0)trace.splice(0,earliest);
   if(trace.length>1000)trace.splice(0,trace.length-1000);state.traces.set(id,trace);
  });
  for(const e of events)wb.events.push({...e,t:e.tick*h.neural_dt});
  wb.events=wb.events.filter(e=>e.t>=t-15);if(wb.events.length>20000){wb.trimmed+=wb.events.length-20000;wb.events=wb.events.slice(-20000);}
  state.spikes+=h.spike_count;$('spike-count').textContent='원시 발화 이벤트 '+state.spikes.toLocaleString();
  drawSignals();timing();
 }
 function timing(){
  const now=performance.now(),elapsed=(now-wb.connectedAt)/1000,age=wb.lastReceive===null?null:(now-wb.lastReceive)/1000;
  $('signal-timing').textContent=`모델 ${state.frame?.simTime.toFixed(3)||'0.000'} s · 접속 경과 ${elapsed.toFixed(1)} 실제초 · 최신 tick ${wb.lastEnd??'—'} · 마지막 수신 ${age===null?'—':age.toFixed(1)} 실제초 전 · 수신 공백 ${wb.gaps}회`+(wb.trimmed?` · 화면 이벤트 버퍼 ${wb.trimmed}개 순환 (기록과 별개)`:'');
 }
 function chartContext(id){const c=$(id),r=c.getBoundingClientRect(),dpr=Math.min(2,devicePixelRatio||1);c.width=Math.max(1,r.width*dpr);c.height=Math.max(1,r.height*dpr);const ctx=c.getContext('2d');ctx.scale(dpr,dpr);ctx.font='9px ui-monospace';return {c,ctx,w:r.width,h:r.height};}
 function drawSignals(){
  const end=state.frame?.simTime||0,start=Math.max(0,end-15),span=Math.max(1,end-start),ids=wb.chart;
  for(const [canvas,column,unit] of [['trace',1,'mV'],['rate-trace',2,'Hz'],['raster',null,'발화 시점']]){
   const {c,ctx,w,h}=chartContext(canvas),left=36,right=w-8,top=18,bottom=h-23;
   if(w<=0)continue;
   let lo=column===1?-75:0,hi=column===1?-35:Math.max(10,...ids.flatMap(id=>(state.traces.get(id)||[]).map(p=>p[2])));
   if(column===1){const vals=ids.flatMap(id=>(state.traces.get(id)||[]).map(p=>p[1]));lo=Math.min(lo,...vals);hi=Math.max(hi,...vals);}
   const x=t=>left+(t-start)/span*(right-left),y=v=>bottom-(v-lo)/(hi-lo)*(bottom-top);
   ctx.fillStyle='#8ca5ae';ctx.fillText(unit,4,11);ctx.fillText(start.toFixed(2)+'s',left,h-6);ctx.fillText(end.toFixed(2)+'s',Math.max(left,right-44),h-6);
   const items=[...(state.frame?.interventions.items||[]),...wb.extraItems];
   ctx.save();ctx.beginPath();ctx.rect(left,top,Math.max(0,right-left),Math.max(0,bottom-top));ctx.clip();
   for(const e of items){if(e.status==='pending'||!e.ids?.some(id=>ids.includes(id)))continue;const dt=state.frame.neuralDt,a=e.at_tick*dt,b=Math.min(end,(e.ended_tick??e.expires_tick)*dt);if(b<=a||b<start||a>end)continue;ctx.fillStyle='#eac68418';ctx.fillRect(x(a),top,x(b)-x(a),bottom-top);}
   if(column!==null){
    ctx.strokeStyle='#23353d';for(let i=0;i<4;i++){const v=lo+i*(hi-lo)/3;ctx.beginPath();ctx.moveTo(left,y(v));ctx.lineTo(right,y(v));ctx.stroke();}
    ids.forEach((id,i)=>{ctx.strokeStyle=colors[i];ctx.lineWidth=1.2;ctx.beginPath();(state.traces.get(id)||[]).forEach((p,j)=>j?ctx.lineTo(x(p[0]),y(p[column])):ctx.moveTo(x(p[0]),y(p[column])));ctx.stroke();});
   }else{
    wb.events.forEach(e=>{const i=ids.indexOf(e.id);if(i<0)return;ctx.fillStyle=colors[i];const row=(bottom-top)/Math.max(1,ids.length);ctx.fillRect(x(e.t),top+i*row+2,1.4,Math.max(2,row-4));});
   }ctx.restore();
   ctx.fillStyle='#8ca5ae';if(column!==null){ctx.fillText(hi.toFixed(0),2,top+4);ctx.fillText(lo.toFixed(0),2,bottom);}else ids.forEach((id,i)=>ctx.fillText(String(i+1),12,top+(i+.6)*(bottom-top)/ids.length));
   c.dataset.seriesIds=JSON.stringify(ids);c.dataset.eventCount=String(canvas==='raster'?wb.events.filter(e=>ids.includes(e.id)).length:0);c.dataset.latestTick=String(wb.lastEnd??'');
  }
 }
 function kindLabel(kind){return {food:'먹이 냄새원',hazard:'위험 냄새원',obstacle:'물리 장애물'}[kind];}
 function objects(){return state.frame?[...state.frame.world.sources,...state.frame.world.obstacles.map(o=>({...o,kind:'obstacle'}))]:[];}
 function renderWorld(){
  const items=objects(),key=JSON.stringify(items),select=$('environment-object');
  if(select.dataset.key!==key){const previous=select.value;select.replaceChildren(new Option('새 물체',''));items.forEach(o=>select.add(new Option(`${o.id} · ${kindLabel(o.kind)} · (${o.p[0]}, ${o.p[2]})`,o.id)));select.value=items.some(o=>o.id===previous)?previous:'';select.dataset.key=key;}
  const f=state.frame;$('environment-count').textContent=`냄새원 ${f.world.sources.length}/24 · 장애물 ${f.world.obstacles.length}/12`;
  const odorants=f.binding.chemical_odorants||[],odorSelect=$('environment-odorant'),odorKey=JSON.stringify(odorants);
  if(odorSelect.dataset.key!==odorKey){const selected=odorSelect.value;odorSelect.replaceChildren(new Option('미지정',''),...odorants.map(o=>new Option(o.name,o.inchikey)));odorSelect.value=selected;odorSelect.dataset.key=odorKey;}
  $('odorant-field').hidden=!odorants.length||$('environment-kind').value==='obstacle';
  const unnamed=f.world.sources.filter(o=>!o.odorant).length;
  if(odorants.length&&unnamed)$('environment-count').textContent+=` · 물질 미지정 ${unnamed}개 (수용기 입력 없음)`;
  $('environment-undo').disabled=!f.environment?.undo_depth;$('environment-delete').disabled=!select.value;
  $('environment-kind').disabled=!!select.value;$('environment-apply').textContent=select.value?'변경 적용':'배치 적용';
  $('environment-sensors').textContent=`감각 tick ${f.sensorTick} · 먹이 좌/우 ${f.sensors.odor.map(v=>v.toFixed(4)).join(' / ')} · 위험 ${f.sensors.danger.toFixed(4)} · 전방 ${f.sensors.nearRanges[4].toFixed(2)} mm`+(f.environment?.sensor_refresh_pending?' · 다음 계산 경계에서 감각 갱신':'')+(f.sensors.odor.every(v=>v>=.999)?' · 양쪽 먹이 센서가 상한에 도달했습니다. 강도를 낮추면 좌우 차이를 확인할 수 있습니다.':'');
 }
 function environmentFields(){const obstacle=$('environment-kind').value==='obstacle';$('strength-field').hidden=obstacle;$('odorant-field').hidden=obstacle||!state.frame?.binding.chemical_odorants?.length;$('radius-field').hidden=!obstacle;$('environment-y').disabled=obstacle;if(obstacle)$('environment-y').value=$('environment-radius').value;wb.preview=true;drawPreview();}
 function environmentPayload(){
  const kind=$('environment-kind').value,position=['x','y','z'].map(c=>Number($('environment-'+c).value));
  return {position,...(kind==='obstacle'?{radius:Number($('environment-radius').value)}:{strength:Number($('environment-strength').value),...(state.frame?.binding.chemical_odorants?.length?{odorant:$('environment-odorant').value||null}:{})})};
 }
 function drawPreview(){
  const c=$('world-preview'),ctx=c.getContext('2d');c.width=view.overlay.width;c.height=view.overlay.height;ctx.setTransform(view.dpr,0,0,view.dpr,0,0);ctx.clearRect(0,0,view.width,view.height);
  if(!wb.preview||!state.frame||!view.vp)return;
  const data=environmentPayload(),kind=$('environment-kind').value,p=data.position,r=kind==='obstacle'?data.radius:.65;
  if(!p.every(Number.isFinite)||!Number.isFinite(r))return;
  ctx.strokeStyle=kind==='hazard'?'#ed9994':kind==='obstacle'?'#eac684':'#76d9c4';ctx.lineWidth=2;ctx.setLineDash([5,3]);ctx.beginPath();
  for(let i=0;i<=40;i++){const a=i*Math.PI/20,q=view.project([p[0]+r*Math.cos(a),p[1],p[2]+r*Math.sin(a)]);if(q.visible)(i?ctx.lineTo(q.x,q.y):ctx.moveTo(q.x,q.y));}ctx.stroke();
  const q=view.project(p);if(q.visible){ctx.font='11px sans-serif';ctx.fillStyle=ctx.strokeStyle;ctx.fillText('미리보기 · 적용 전',q.x+10,q.y-10);}
  c.dataset.position=JSON.stringify(p);$('environment-preview-note').textContent=`미리보기 (${p.map(n=>n.toFixed(2)).join(', ')}) mm · 서버 검증 후 적용`;
 }
 function chooseObject(){const o=objects().find(o=>o.id===$('environment-object').value);if(o){$('environment-kind').value=o.kind;['x','y','z'].forEach((c,i)=>$('environment-'+c).value=o.p[i]);if(o.kind==='obstacle')$('environment-radius').value=o.r;else {$('environment-strength').value=o.strength;$('environment-odorant').value=o.odorant||'';}}renderWorld();environmentFields();}
 function endPreview(){wb.pick=false;wb.preview=false;$('environment-pick').setAttribute('aria-pressed','false');view.canvas.classList.remove('placing');drawPreview();}
 async function environmentCommand(type,payload,op='command'){
  pause();if(wb.editing)throw Error('환경 편집을 적용하는 중입니다.');wb.editing=true;
  const controls=[...document.querySelectorAll('.environment-editor button,.environment-editor input,.environment-editor select')];
  controls.forEach(c=>c.disabled=true);put('environment-preview-note','서버에서 환경을 적용하고 물리 상태를 확인합니다…');
  try{const result=op==='command'?await command(type,payload):await rpc.request(op,payload);if(op!=='command')applyFrame(result);if(type==='undo_environment'||op==='environment_load')chooseObject();endPreview();put('environment-preview-note',`서버 적용 완료 · tick ${result.tick}. 되돌리기는 현재 시각에서 환경만 바꿉니다.`);return result;}
  catch(e){put('environment-preview-note','환경이 적용되지 않았습니다: '+e.message);throw e;}
  finally{wb.editing=false;controls.forEach(c=>c.disabled=false);renderWorld();$('environment-y').disabled=$('environment-kind').value==='obstacle';}
 }
 async function loadEnvironmentList(){const rows=await rpc.request('environments');$('environments').replaceChildren(new Option('저장한 환경',''));rows.forEach(r=>$('environments').add(new Option(r.name,r.name)));}
 function renderInterventions(){
  const root=$('intervention-list'),items=[...(state.frame?.interventions.items||[]),...wb.extraItems],unique=[...new Map(items.map(i=>[i.serial,i])).values()];
  const key=JSON.stringify(unique);if(root.dataset.key===key)return;root.dataset.key=key;root.replaceChildren();
  unique.forEach(e=>{const row=document.createElement('div');row.className='intervention-item';row.dataset.serial=e.serial;row.dataset.status=e.status;const label=document.createElement('div');
   label.textContent=`#${e.serial} · ${{pending:'예약',active:'활성',expired:'만료',cancelled:'취소'}[e.status]} · ${e.kind} · ${e.ids?.length??e.edge_count??0}개${e.amplitude_mV!==undefined?' · '+e.amplitude_mV+'mV':''}\ntick ${e.at_tick} → ${e.ended_tick??e.expires_tick} · 잔여 ${(Math.max(0,(e.expires_tick-(state.frame?.tick||0)))*(state.frame?.neuralDt||.0001)*(e.status==='pending'||e.status==='active'?1:0)).toFixed(3)} 모델초`;
   row.append(label);const details=document.createElement('details'),summary=document.createElement('summary'),ids=document.createElement('pre');summary.textContent='대상 보기';ids.textContent=(e.ids||e.channels||[]).join('\n');details.append(summary,ids);row.append(details);
   if(['pending','active'].includes(e.status)){const cancel=document.createElement('button');cancel.textContent='취소';cancel.setAttribute('aria-label',`개입 ${e.serial} 취소`);cancel.onclick=()=>{pause();command('cancel_intervention',{serial:e.serial}).then(()=>{wb.extraItems=[];renderInterventions();}).catch(err=>tell(err.message,true));};row.append(cancel);}root.append(row);
  });
  $('interventions-more').hidden=items.length>=(state.frame?.interventions.total||0);
 }
 async function loadPresets(){wb.presets=await rpc.request('presets');$('cohort-preset').replaceChildren();wb.presets.forEach((p,i)=>$('cohort-preset').add(new Option(p.name+' · '+p.ids.length+'개',String(i))));}
 function preset(){return [...(wb.presets[Number($('cohort-preset').value)]?.ids||[])];}
 button('preset-observe',async()=>applyFrame(await rpc.request('subscribe',{ids:preset()})));
 button('preset-intervene',async()=>{wb.intervention=preset();await lookup(wb.intervention);cohortDisplay();});
 button('preset-record',async()=>{if(state.recording)throw Error('기록 중에는 대상을 바꿀 수 없습니다.');wb.record=preset();await lookup(wb.record);cohortDisplay();});
 button('observe-to-intervene',()=>{wb.intervention=[...state.selected];cohortDisplay();});
 button('observe-to-record',()=>{if(state.recording)throw Error('기록 중에는 대상을 바꿀 수 없습니다.');wb.record=[...state.selected];cohortDisplay();});
 button('cohort-save',async()=>{await rpc.request('cohort_save',{name:$('cohort-name').value,ids:state.selected});await loadPresets();tell('관측 집단 프리셋을 저장했습니다.');});
 button('environment-new',()=>{$('environment-object').value='';chooseObject();});
 $('environment-object').addEventListener('change',chooseObject);
 for(const id of ['environment-kind','environment-x','environment-y','environment-z','environment-strength','environment-radius'])$(id).addEventListener('input',environmentFields);
 button('environment-pick',()=>{pause();wb.pick=!wb.pick;$('environment-pick').setAttribute('aria-pressed',String(wb.pick));view.canvas.classList.toggle('placing',wb.pick);if(wb.pick)view.top();});
 let drag=null;
 view.canvas.addEventListener('pointerdown',e=>{if(!state.frame)return;if(wb.pick){const r=view.canvas.getBoundingClientRect(),p=view.pointOnPlane(e.clientX-r.left,e.clientY-r.top,Number($('environment-y').value));if(p){['x','y','z'].forEach((c,i)=>$('environment-'+c).value=p[i].toFixed(2));environmentFields();}return;}drag=[e.clientX,e.clientY];view.canvas.setPointerCapture(e.pointerId);});
 view.canvas.addEventListener('pointermove',e=>{if(!drag)return;view.firstPerson=false;view.follow=false;view.theta-=(e.clientX-drag[0])*.006;view.phi=Math.max(.035,Math.min(1.5,view.phi+(e.clientY-drag[1])*.005));drag=[e.clientX,e.clientY];});
 for(const event of ['pointerup','pointercancel','lostpointercapture'])view.canvas.addEventListener(event,()=>drag=null);
 view.canvas.addEventListener('wheel',e=>{e.preventDefault();view.distance=Math.max(5,Math.min(90,view.distance*Math.exp(e.deltaY*.001)));},{passive:false});
 button('environment-apply',async()=>{const id=$('environment-object').value,payload=environmentPayload();const before=new Set(objects().map(o=>o.id));await environmentCommand(id?'update_object':'place',id?{id,...payload}:{kind:$('environment-kind').value,...payload});if(!id){const added=objects().find(o=>!before.has(o.id));if(added)$('environment-object').value=added.id;}renderWorld();});
 button('environment-delete',()=>environmentCommand('delete_object',{id:$('environment-object').value}));
 button('environment-clear',()=>environmentCommand('clear_added',{}));
 button('environment-undo',()=>environmentCommand('undo_environment',{}));
 button('environment-save',async()=>{pause();const name=$('environment-name').value,r=await rpc.request('environment_save',name?{name}:{});await loadEnvironmentList();$('environments').value=r.name;tell('환경 저장: '+r.name);});
 button('environment-load',async()=>{await environmentCommand(null,{name:$('environments').value},'environment_load');tell('현재 시각에서 환경을 불러왔습니다.');});
 button('interventions-more',async()=>{const offset=(state.frame?.interventions.items.length||0)+wb.extraItems.length,r=await rpc.request('interventions',{offset,limit:50});wb.extraItems.push(...r.items);renderInterventions();});
 $('pair-origin').addEventListener('change',()=>{$('pair-seed').disabled=$('pair-origin').value!=='fresh';});
 function interventionSpec(){
  if(!wb.intervention.length)throw Error('개입 집단을 지정하세요. 관측 집단 복사 또는 프리셋을 사용할 수 있습니다.');
  const onset=Number($('intervention-onset').value),duration=Number($('duration').value);
  if(onset<0||onset>3600||duration<=0||Math.abs(onset/.005-Math.round(onset/.005))>1e-8||Math.abs(duration/.005-Math.round(duration/.005))>1e-8)throw Error('시점과 기간은 0.005 모델초 단위로 입력하세요.');
  return {kind:$('intervention-kind').value,ids:[...wb.intervention],amplitude_mV:Number($('amplitude').value),duration_controls:Math.round(duration/.005),at_tick:state.frame.tick+Math.round(onset/state.frame.neuralDt)};
 }
 function frame(f){
  $('metabolism-status').textContent=f.metabolism?.enabled?`가상 에너지 ${f.metabolism.energy.toFixed(3)} · 배고픔 ${(f.metabolism.hunger*100).toFixed(0)}% · 누적 섭취 ${f.metabolism.intake.toFixed(4)} · ${f.metabolism.feeding?'섭취 중':'섭취 없음'} · 신경 조절 미연결`:'섭취·에너지 모델 꺼짐';
  const clipped=(f.sensory_ports||[]).filter(p=>p.clipped).map(p=>p.name);$('saturation-status').textContent='입력 상한 적용: '+(clipped.join(', ')||'없음')+' · 운동 출력 상한: '+Object.entries(f.motor_diagnostics?.clipped||{}).filter(([k,v])=>v).map(([k])=>k).join(', ');

  if(state.recording)wb.record=[...f.recording.cohort_ids];
  $('mode-explanation').textContent=`현재 실행: ${f.mode} · `+(f.neuromuscular?.enabled?'BANC 운동뉴런이 관절을 직접 구동하고 다리 감각을 신경망에 되돌립니다.':({C_SHADOW:'B가 몸을 구동합니다. C 신경 개입은 관측·회로 검사에 적용됩니다.',C_STRICT:'C 신경 출력을 몸에 적용합니다.',C_ASSISTED:'C 신경 출력에 회피 보조가 개입할 수 있습니다.',B_COMPAT:'기존 B 회로가 몸을 구동합니다.'}[f.mode]))+` 현재 명령: ${f.command.command_source} · 위 모드 설정은 새 실험에만 적용됩니다.`;
  cohortDisplay();renderWorld();renderInterventions();timing();
 }
 async function ready(){
  resetSignals();wb.extraItems=[];wb.record=[...state.frame.recording.cohort_ids];wb.intervention=[...state.selected];
  await campaigns();await loadPresets();await lookup([...new Set([...state.selected,...wb.record])]);await loadEnvironmentList();cohortDisplay();chartPicker();
  applyFrame(await rpc.request('frame'));
 }
 let campaignPoll=0,campaignBusy=false;
 function showCampaign(r){$('campaign-status').textContent=JSON.stringify(r,null,2);}
 async function campaigns(){const rows=await rpc.request('campaign_list'),selected=$('campaign-list').value;$('campaign-list').replaceChildren();for(const row of rows){const o=document.createElement('option');o.value=row.id;o.textContent=row.id+' · '+row.status;$('campaign-list').append(o);}if(rows.some(r=>r.id===selected))$('campaign-list').value=selected;if(rows.length)showCampaign(rows.find(r=>r.id===$('campaign-list').value)||rows.at(-1));}
 button('campaign-refresh',campaigns);
 button('campaign-start',async()=>{pause();const f=state.frame,r=await rpc.request('campaign_start',{spec:{schema:'flylab.campaign.v1',cases:[{name:'interactive',mode:f.mode,seed:f.seed,scene:$('campaign-scene').value,seconds:Number($('campaign-seconds').value)}]}});await campaigns();$('campaign-list').value=r.id;showCampaign(r);});
 button('campaign-pilot',async()=>{pause();const r=await rpc.request('campaign_start');await campaigns();$('campaign-list').value=r.id;showCampaign(r);});
 button('campaign-cancel',async()=>showCampaign(await rpc.request('campaign_cancel',{id:$('campaign-list').value})));
 button('campaign-resume',async()=>{pause();showCampaign(await rpc.request('campaign_resume',{id:$('campaign-list').value}));});
 $('campaign-list').addEventListener('change',()=>rpc.request('campaign_status',{id:$('campaign-list').value}).then(showCampaign).catch(e=>tell(e.message,true)));
 function animate(now,drawn){if(now-campaignPoll>3000&&!campaignBusy&&$('campaign-list').value){campaignPoll=now;campaignBusy=true;rpc.request('campaign_status',{id:$('campaign-list').value}).then(showCampaign).catch(e=>tell(e.message,true)).finally(()=>campaignBusy=false);}if(drawn)drawPreview();if(now-wb.lastRegions>=5000){wb.lastRegions=now;timing();if($('regions-auto').checked&&!state.busy&&!wb.regionsPending&&!document.hidden){wb.regionsPending=true;regions().catch(e=>tell(e.message,true)).finally(()=>wb.regionsPending=false);}}}
 function showComparison(report){
  const first=report.comparisons?.[0];if(!first?.preview)return;$('comparison-chart').hidden=false;
  const {ctx,w,h}=chartContext('comparison-chart');ctx.fillStyle='#8ca5ae';ctx.fillText('첫 비교 · 청록: 대조군 / 노랑: 개입군 · 동일 모델 시간축',8,12);
  const series=first.preview,seconds=Math.max(.005,...series.flatMap(s=>s.points.map(p=>p.model_s)));
  for(const [i,key,title] of [[0,'signed_forward_mm','실제 전진 변위 mm'],[1,'command_mm_s','적용 명령 mm/s'],[2,'forward_rate_Hz','전진 출력 집단 Hz']]){
   const top=30+i*(h-48)/3,bottom=top+(h-48)/3-18,values=series.flatMap(s=>s.points.map(p=>p[key])),lo=Math.min(0,...values),hi=Math.max(lo+.01,...values);
   ctx.fillStyle='#8ca5ae';ctx.fillText(title,8,top-3);
   series.forEach((s,j)=>{ctx.strokeStyle=colors[j];ctx.beginPath();s.points.forEach((p,k)=>{const x=40+p.model_s/seconds*(w-50),y=bottom-(p[key]-lo)/(hi-lo)*(bottom-top);k?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();});
   ctx.fillText(hi.toFixed(2),3,top+10);ctx.fillText(lo.toFixed(2),3,bottom);
  }ctx.fillStyle='#8ca5ae';ctx.fillText('0s',40,h-3);ctx.fillText(seconds.toFixed(3)+'s',w-60,h-3);
 }
 return {...wb, state:wb, frame, ready, subscriptionChanged, acceptSignals, drawSignals, drawPreview, interventionSpec, animate, pause,
  recordIds:()=>[...wb.record],interventionIds:()=>[...wb.intervention],resetSignals,showComparison};
};
})(globalThis.Fly);
