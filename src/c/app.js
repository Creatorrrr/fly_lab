/* C viewer: bounded subscriptions, raw units, explicit causality and source. */
(function(F){
'use strict';
const PROTOCOL='flylab.protocol.v3', $=id=>document.getElementById(id);
const view=new F.WorldView($('world'),$('world-overlay'));
const state={frame:null,playing:false,busy:false,selected:[],catalog:[],offset:0,total:0,epoch:0,sequence:0,
  signals:new Map(),traces:new Map(),spikes:0,closed:false,names:new Map(),recording:false,backend:'exp_lif_cpu_reference',dataset:'연결망'};
const colors=['#76d9c4','#eac684','#87b4e8','#c9a1ed','#ed9994','#a6d883','#88d9e5','#dba3c6'];
function setText(node,value){if(node.textContent!==value)node.textContent=value;}
function tell(message,error=false){$('alert').hidden=!message;$('alert').textContent=message||'';if(error)console.error(message);}
function button(id,handler){$(id).addEventListener('click',()=>Promise.resolve().then(handler).catch(e=>{state.playing=false;updatePlay();tell(e.message,true);}));}
class Transport{
 constructor(){this.id=1;this.pending=new Map();this.tail=Promise.resolve();}
 async connect(){
  const r=await fetch('/api/bootstrap',{cache:'no-store'});if(!r.ok)throw Error('서버 연결 오류 '+r.status);
  const config=await r.json();if(config.protocol!==PROTOCOL)throw Error('C 서버와 화면의 프로토콜이 다릅니다.');
  if(!config.graphAvailable||!config.bindingsAvailable)throw Error('전뇌 bundle 또는 포트 파일이 없습니다. tools/build_c_graph.py와 tools/build_c_bindings.py를 실행하세요.');
  if(!config.dependencies.ready&&!config.testDouble)throw Error('FlyGym/MuJoCo 의존성을 확인하세요. python run_c.py --doctor');
  if(config.testDouble)tell('TEST FIXTURE · 이 연결은 실제 물리 검증이 아닙니다.');
  $('mode').value=config.defaultMode;
  await new Promise((resolve,reject)=>{
   const ws=this.ws=new WebSocket('ws://'+location.host+'/ws');ws.binaryType='arraybuffer';
   const timer=setTimeout(()=>{ws.close();reject(Error('WebSocket 연결 시간 초과'));},15000);
   ws.onopen=()=>ws.send(JSON.stringify({protocol:PROTOCOL,token:config.token}));
   ws.onmessage=event=>{
    if(event.data instanceof ArrayBuffer){try{receiveSignals(event.data);}catch(e){tell(e.message,true);}return;}
    let m;try{m=JSON.parse(event.data);}catch(e){reject(e);return;}
    if(m.authenticated){clearTimeout(timer);resolve();return;}
    const p=this.pending.get(m.requestId);if(!p)return;this.pending.delete(m.requestId);clearTimeout(p.timer);
    if(m.ok)p.resolve(m.result);else p.reject(Error(m.error||'서버 오류'));
   };
   ws.onerror=()=>{clearTimeout(timer);reject(Error('WebSocket 연결 실패. 다른 C 관찰 창이 열려 있는지 확인하세요.'));};
   ws.onclose=()=>{clearTimeout(timer);state.closed=true;state.playing=false;updatePlay();$('status-dot').className='';$('connection-status').textContent='연결 종료';
    for(const p of this.pending.values()){clearTimeout(p.timer);p.reject(Error('서버 연결이 종료됐습니다.'));}this.pending.clear();reject(Error('서버 연결 종료'));};
  });return config;
 }
 request(op,payload={}){
  const run=()=>new Promise((resolve,reject)=>{
   if(this.ws?.readyState!==WebSocket.OPEN){reject(Error('서버 미연결'));return;}
   const requestId=this.id++,timer=setTimeout(()=>{this.ws.close();reject(Error('계산 응답 시간 초과'));},op==='paired'?3600000:600000);
   this.pending.set(requestId,{resolve,reject,timer});this.ws.send(JSON.stringify({protocol:PROTOCOL,requestId,op,payload}));
  });const result=this.tail.then(run);this.tail=result.catch(()=>{});return result;
 }
}
const rpc=new Transport();
let playback;
const workbench=F.createWorkbench({state,rpc,view,applyFrame,tell,button,command,updatePlay,regions,renderSelection});
function updatePlay(){$('play').textContent=state.playing?'Ⅱ 일시정지':'▶ 재생';$('play').disabled=state.closed||!state.frame;$('step').disabled=state.closed||!state.frame||state.playing;playback?.sync();const status=F.describeStrictExperiment(state.frame,state.playing),element=$('experiment-status');element.hidden=status.state==='other_mode';element.dataset.state=status.state;setText(element,[status.message,...status.warnings].join(' '));}
function commandText(c){return c?`${c.forwardSpeed.toFixed(2)} mm/s · ${c.yawRate.toFixed(2)} rad/s`:'사용 안 함';}
function renderJoints(f){const whole=f.physics.wholeBody;if($('joints').closest('details').open){const body=$('joints');body.replaceChildren();const names=whole?.names||f.physics.jointNames,angles=whole?.angles_rad||f.physics.jointAngles,targets=whole?.targets_rad||f.physics.jointTargets,forces=whole?.actuator_force||f.physics.actuatorForces,mappings=new Map((f.neuromuscular?.actuation?.axes||[]).map(row=>[row.name,row]));names.forEach((name,i)=>{const tr=document.createElement('tr'),mapping=mappings.get(name),label=!mapping?'매핑 정보 없음':mapping.status==='unmapped'?'신경 출력 미연결':`공학적 매핑 · 운동 ${mapping.motor_neurons}개 (+${mapping.positive_motor_neurons}/−${mapping.negative_motor_neurons})`;[name,angles[i].toFixed(4),targets[i].toFixed(4),forces[i].toFixed(4),label].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td);});body.append(tr);});}}
$('joints').closest('details').addEventListener('toggle',()=>{if(state.frame)renderJoints(state.frame);});
function applyFrame(f){
 if(!f||f.schema!=='flylab.frame.v3')return;
 const changed=state.epoch!==f.subscription.epoch||state.selected.join('|')!==f.subscription.ids.join('|')||(state.frame&&f.tick<state.frame.tick);
 state.frame=f;state.epoch=f.subscription.epoch;state.selected=f.subscription.ids;view.setFrame(f);
 if(changed)workbench.subscriptionChanged();
 $('waiting').hidden=true;$('mode-tag').textContent=f.mode;$('clock').textContent=f.simTime.toFixed(3)+' s';
 $('scope').textContent=f.mode==='B_COMPAT'?'B_COMPAT · 기존 부분 신경 모델':`${state.dataset} · ${f.scope.simulatedNodes.toLocaleString()} 뉴런 · ${f.scope.pairEdges.toLocaleString()} 연결`;
 $('neural-scope-note').textContent=f.mode==='B_COMPAT'?'현재 B 부분 신경 모델을 사용하며 C 전뇌 계산은 꺼져 있습니다.':`${f.scope.simulatedNodes.toLocaleString()}개 모델 뉴런을 계산하며 선택한 신호만 전송합니다. 전체 스냅샷 판정: ${f.scope.fullBrain?'확인':'미승인'}`;
 $('distance').textContent=f.body.travel.toFixed(2)+' mm';$('speed').textContent=f.body.speed.toFixed(2)+' mm/s';
 $('physics-device').textContent=f.physics.backend.includes('warp')?'CUDA 물리 · 검증 후보':'CPU 물리';
 $('mean-rate').textContent=f.neural?f.neural.mean_rate_Hz.toFixed(3)+' Hz':'B rate';$('performance').textContent=f.performance.sim_wall_ratio.toFixed(3)+'×';
 $('command-source').textContent=f.neuromuscular?.enabled?'BANC 운동뉴런 → 관절':f.command.command_source;$('neural-command').textContent=commandText(f.command.u_neural)+(f.neuromuscular?.enabled?' (관측값)':'');
 const activity=F.neuromuscularActivity(f);
 $('motor-activity').hidden=!activity;$('motor-activity').textContent=activity?.message||'';
 $('neural-command').previousElementSibling.textContent=activity?'참고 · 하행뉴런 출력':'신경 출력';
 if(activity)$('neural-command').textContent=commandText(f.command.u_neural)+' (현재 관절 제어에 사용 안 함)';
 $('assist-command').textContent=commandText(f.command.u_assist);$('final-command').textContent=activity?(activity.available?`신경 관절 변화 최대 ${activity.jointOffsetMaxRad.toFixed(4)} rad · 신경 힘줄 입력 최대 ${activity.neuralTendonMaxInput.toFixed(4)}`:'신경 구동 측정 대기'):commandText(f.command.u_final);
 const dt=f.neuralDt||.0001;$('causal-time').textContent=`감각 ${(f.command.sensor_tick*dt).toFixed(4)}s → 신경 출력 ${(f.command.neural_readout_tick*dt).toFixed(4)}s → 몸 구동 [${(f.command.interval_start_tick*dt).toFixed(4)}, ${(f.command.interval_end_tick*dt).toFixed(4)})s`+(f.command.assist_reason?' · '+f.command.assist_reason:'');
 const channelTotals=new Map();for(const p of f.binding.sensory){const row=channelTotals.get(p.channel)||{ports:0,targets:0};row.ports++;row.targets+=p.targets;channelTotals.set(p.channel,row);}
 $('port-note').textContent='현재 C 입력: '+[...channelTotals].map(([channel,row])=>`${channel} · ${row.ports}포트 / ${row.targets}대상 연결`).join(', ')+` · 미연결: ${f.binding.unused_observations.join(', ')} · 위험 입력: ${f.binding.chemical_odorants?.length?'지정한 물질의 DoOR 반응 (농도·이득 미교정)':f.binding.hazard_semantics==='unbound'?'미연결':'geosmin 가정 (실험)'} · 미확정 전달물질 ${f.scope.unknownNeurotransmitters.toLocaleString()} 뉴런 / 효력 0인 연결 ${f.scope.maskedEdges.toLocaleString()}개`;
 $('port-values').textContent='현재 입력 세기: '+(f.sensory_ports.slice(0,12).map(p=>`${p.name} ${p.value.toFixed(3)} ${p.unit}${p.enabled?'':' (차단)'}`).join(' · ')||'아직 계산 전')+(f.sensory_ports.length>12?` · 총 ${f.sensory_ports.length}포트 중 12개 표시`:'')+' · 출력 집단: '+Object.entries(f.motor_rates_Hz).map(([name,rate])=>`${name} ${rate.toFixed(2)} Hz`).join(' · ');
 const legLoop=f.neuromuscular;
 const actuation=legLoop?.actuation;
 $('actuation-status').textContent=actuation?`현재 신체의 위치 구동 ${actuation.active_axes}축 중 ${actuation.mapped_axes}축에 신경 출력 매핑 · 미연결 ${actuation.unmapped_axes}축${actuation.passive_joints===null?'':` · 수동 관절 ${actuation.passive_joints}개`} · 힘줄 신경 연결 ${actuation.mapped_tendon_controls||0}/${actuation.tendon_controls?.length||0}개 · 전체 운동뉴런 ${actuation.graph_motor_neurons}개 중 ${actuation.mapped_motor_neurons}개 사용, ${actuation.unmapped_motor_neurons}개 미연결. 매핑은 구동 가능 경로를 뜻하며, 실제 활성·행동 성공은 별도로 확인합니다.`:'현재 실행의 축별 신경 매핑 정보가 없습니다.';
 $('neuromuscular-status').textContent=legLoop?.enabled?`BANC 다리 폐루프 · 감각 ${legLoop.sensory_neurons.toLocaleString()}개 → 전체 신경망 → 운동 ${legLoop.motor_neurons}개 · 관절 ${legLoop.covered_dofs}/${legLoop.total_leg_dofs} 연결 · 사전 보행 궤적 CPG 꺼짐 · 미연결: ${legLoop.unbound_dofs.join(', ')||'없음'} · 근육 힘과 감각 세부 반응은 검증되지 않은 공학적 변환입니다.`:(['B_COMPAT','C_SHADOW'].includes(f.mode)?'현재 몸 구동: 기존 B 제어기와 보행 CPG를 사용합니다. C 전뇌 출력으로 구동하는 모드는 아닙니다.':'현재 몸 구동: 하행뉴런의 전진·회전 출력을 기존 CPG 보행기에 전달합니다.');
 $('leg-feedback-off').disabled=!legLoop?.enabled;
 const contact=legLoop?.contact?.support_load_bw?legLoop.contact:null;
 $('leg-contact-diagnostics').textContent=contact?['LF','LM','LH','RF','RM','RH'].map((leg,i)=>`${leg}: 지지 ${contact.support_load_bw[i].toFixed(3)} BW · 부착력 ${contact.adhesion_force_bw[i].toFixed(3)} BW · 지면 외 접촉 ${contact.non_support_load_bw[i].toFixed(3)} BW · 부착 ${f.physics.adhesion[i]?'켜짐':'꺼짐'}`).join('\n'):'하중 보정 진단은 BANC v2 프로파일에서 표시됩니다.';
 $('leg-contact-diagnostics').hidden=!legLoop?.enabled;
 $('leg-receptor-diagnostics').textContent=legLoop?.schema==='flylab.neuromuscular.v2'?`방향 미확정 수용체 ${legLoop.unresolved_polarity_targets}개: 외부 각도·방향 자극 보류 · 출력 제한 범위를 넘은 축 ${(legLoop.muscles||[]).reduce((sum,m)=>sum+(m.bounded_axes||[]).filter(Boolean).length,0)}/42 · 보행 성공 여부는 별도 행동 검사로 판정합니다.`:'';
 $('motor').checked=f.config.motorCoupled;$('cue').checked=f.world.cueOn;$('food').checked=f.world.foodOn;
 const whole=f.physics.wholeBody;
 renderBodyControls(f);
 $('body-status').textContent=f.physics.testDouble?'TEST DOUBLE':whole?`${whole.names.length} 위치 서보 · ${f.physics.tendons?.names.length||0} 힘줄 · ${whole.passive.names.length} 수동 관절 · ${whole.root_fixed?'몸통 고정':'자유 이동'}`:`${f.physics.jointNames.length} 관절 · ${f.physics.backend}`;
 const feet=$('feet');
 if(!feet.children.length)['LF','LM','LH','RF','RM','RH'].forEach(leg=>{const row=document.createElement('div');row.className='foot';const label=document.createElement('span');label.textContent=leg;row.append(label,document.createElement('b'),document.createElement('i'));feet.append(row);});
 [...feet.children].forEach((row,i)=>{setText(row.children[1],f.physics.contactsBW[i].toFixed(2)+' BW');row.children[2].style.width=Math.min(100,f.physics.contactsBW[i]*100)+'%';});
 renderJoints(f);
 $('events').replaceChildren(...f.events.slice(-8).reverse().map(e=>{const div=document.createElement('div');div.textContent=`${(e.tick*dt).toFixed(3)} s · ${e.kind}`;return div;}));
 state.recording=f.recording.active;$('record-status').textContent=f.recording.active?`기록 중 · ${(f.recording.bytes/1024).toFixed(1)} KB`:'기록 꺼짐';$('record').textContent=f.recording.active?'기록 종료':'기록 시작';
 $('selected-count').textContent=`${state.selected.length} / 512`;
 if(f.fault){state.playing=false;tell('실험 중단: '+f.fault,true);}updatePlay();renderSelection();workbench.frame(f);
}
function receiveSignals(buffer){workbench.acceptSignals(buffer);renderSelection();}

function renderSelection(){
 const root=$('selection'),key=state.selected.join('|');
 if(root.dataset.subscriptionKey!==key){root.replaceChildren();root.dataset.subscriptionKey=key;
  for(const id of state.selected){const row=document.createElement('div');row.className='signal-row';row.append(document.createElement('span'),document.createElement('span'),document.createElement('span'));row.children[1].className='voltage';row.children[2].className='rate';root.append(row);}
  if(!state.selected.length)root.textContent='검색 결과에서 모니터링할 뉴런을 선택하세요.';
 }
 for(const [i,id] of state.selected.entries()){
  const value=state.signals.get(id),row=root.children[i],name=row.children[0];
  setText(name,state.names.get(id)||id.replace('flywire:fafb:783:',''));name.title=id;name.style.color=workbench.state.chart.includes(id)?colors[workbench.state.chart.indexOf(id)]:'#8ca5ae';
  setText(row.children[1],value?value.voltage.toFixed(2)+' mV':'— mV');
  setText(row.children[2],value?value.rate.toFixed(2)+' Hz':'— Hz');
 }
 for(const row of $('catalog').children)row.classList.toggle('selected',state.selected.includes(row.dataset.id));
}
function drawTrace(){workbench.drawSignals();}

async function search(reset=true){
 if(reset)state.offset=0;const result=await rpc.request('catalog',{query:$('search').value,offset:state.offset,limit:50});
 state.catalog=result.items;state.total=result.total;const list=$('catalog');list.replaceChildren();
 result.items.forEach(n=>{
  state.names.set(n.id,(n.cell_type||'유형 미지정')+' · '+n.root_id.slice(-5));
  const row=document.createElement('button');row.className='neuron-row';row.dataset.id=n.id;row.setAttribute('role','listitem');
  const group=document.createElement('div'),name=document.createElement('b'),id=document.createElement('code'),nt=document.createElement('span');name.textContent=n.cell_type||'유형 미지정';id.textContent=n.root_id+' · '+n.soma_side;nt.textContent=n.nt_type||'미확정';group.append(name,id);row.append(group,nt);
  row.onclick=()=>toggle(n.id).catch(e=>tell(e.message,true));list.append(row);
 });$('catalog-count').textContent=`${result.total.toLocaleString()}개 · ${result.offset+1}–${Math.min(result.offset+50,result.total)}`;$('previous').disabled=state.offset===0;$('next').disabled=state.offset+50>=state.total;renderSelection();
}
async function toggle(id){const ids=state.selected.includes(id)?state.selected.filter(x=>x!==id):[...state.selected,id];applyFrame(await rpc.request('subscribe',{ids}));}
async function regions(){const rows=await rpc.request('regions');$('regions').replaceChildren(...rows.map(r=>{const box=document.createElement('div');box.className='region';const name=document.createElement('span');name.textContent=r.region;const value=document.createElement('strong');value.textContent=r.mean_rate_Hz.toFixed(2)+' Hz';const count=document.createElement('small');count.textContent=r.neurons.toLocaleString()+' 뉴런';box.append(name,value,count);return box;}));}
async function checkpoints(){const rows=await rpc.request('checkpoints');$('checkpoints').replaceChildren();const blank=document.createElement('option');blank.value='';blank.textContent='저장한 체크포인트';$('checkpoints').append(blank);rows.forEach(r=>{const o=document.createElement('option');o.value=o.textContent=r.name;$('checkpoints').append(o);});}
function applyProfileModes(){
 const profile=state.profiles?.find(p=>p.name===$('binding-profile').value);
 const modes=profile?.execution?.modes||state.supportedModes;
 if(!modes)return;
 for(const option of $('mode').options)option.disabled=!modes.includes(option.value);
 if(!modes.includes($('mode').value))$('mode').value=modes[0];
}
$('binding-profile').addEventListener('change',applyProfileModes);
function applyReady(result){state.profiles=result.profiles;state.supportedModes=result.capabilities.modes;state.dataset=result.manifest.dataset_id+' v'+result.manifest.snapshot_id;$('dataset-label').textContent=state.dataset+' · '+(result.frame.physics.bodyModel==='flybody'?'FlyBody':'NeuroMechFly');
 $('physical-observation').hidden=true;observationLastTick=-1;
 const labels={'bindings.json':'기존 평균 냄새','bindings-bilateral-geosmin-v1.json':'좌우 먹이·geosmin v1 (실험)','bindings-bilateral-geosmin-v2.json':'좌우 먹이·geosmin v2 · 농도 압축 (실험)','bindings-four-site-odor-v1.json':'4지점 후각 · 더듬이 입력 (실험)','bindings-visual-head-contact-research-v1.json':'좌우 냄새·물체 시야·머리 접촉 (연구)','bindings-odor-poisson-current-hypothesis-v1.json':'후각 Poisson 전류 입력 (검증 가설)'};
 labels['bindings-walking-population-v1.json']='DNg100·DNg97 보행 출력 (실험)';
 labels['bindings-walking-reset-current-v2.json']='보행 출력·발화 후 전류 초기화 (가설)';
 labels['bindings-walking-poisson-reset-current-v3.json']='약한 냄새 Poisson 입력·보행 출력 (가설)';
 labels['bindings-walking-voltage-events-v4.json']='Poisson 전압 사건·보행 출력 (가설)';
 labels['bindings-evidence-sensory-v4.json']='낱눈 열 연결·DoOR 41수용기 후각 v4 (연구)';
 labels['bindings-multimodal-v2.json']='자동 복안·4지점 후각·plume v2 (연구)';
 labels['bindings-neuromuscular-v1.json']='BANC 뇌·VNC·다리 폐루프 (연구)';
 labels['bindings-neuromuscular-v2.json']='BANC 하중 보정·방향 감각 v2 (보행 미검증)';
 labels['bindings-walking-visual-contact-v5.json']='보행·시각·머리 접촉 v5 (연구)';
 labels['bindings-walking-visual-contact-walk-off-v6.json']='보행·시각·접촉·Bluebell 정지 v6 (연구)';
 applyFrame(result.frame);$('binding-profile').replaceChildren(...(result.profiles||[]).map(p=>{const label=labels[p.name]||p.profile||p.name;const option=new Option(label+(p.available===false?' · 사용 불가':''),p.name,p.current,p.current);option.disabled=p.available===false;option.title=p.reason||'';return option;}));$('mode').value=result.frame.mode;$('seed').value=result.frame.seed;$('body-model').value=result.frame.physics.wholeBody?'flybody_whole':(result.frame.physics.bodyModel||'neuromechfly');$('body-terrain').value=result.frame.physics.terrain||'flat';$('body-attachment').value=result.frame.physics.attachment||'free';const tn=result.frame.physics.tendons?.names||[];$('body-tendons').value=tn.length===8?'all':tn.length===6?'tarsi':tn.length===2?'abdomen':'none';
 applyProfileModes();
 state.backend=result.capabilities.backend;$('backend').value=state.backend==='legacy_b_rate'?'exp_lif_cpu_reference':state.backend;$('backend').disabled=state.backend==='legacy_b_rate';
 const friction=String(result.frame.config.friction);
 if(![...$('friction').options].some(o=>o.value===friction))$('friction').add(new Option(friction+'×',friction));
 $('friction').value=friction;
 $('manifest').textContent=JSON.stringify({capabilities:result.capabilities,manifest:result.manifest,binding:result.binding},null,2);
 const device=state.backend==='exp_lif_mps'?'MPS':state.backend==='exp_lif_cuda'?'CUDA':'CPU';
 $('connection-status').textContent='LOCAL · '+device+' / '+(result.capabilities.physical?'PHYSICS':'TEST FIXTURE');
}
async function refreshExperiment(result,message){
 state.sequence=0;state.signals.clear();state.traces.clear();state.spikes=0;applyReady(result);
 $('search').value='DNa02';await search();await regions();await checkpoints();await workbench.ready();tell(message);
}
function renderBodyControls(f){
 const supported=typeof f.physics.attachment==='string';for(const id of ['body-attachment','body-tendons','shared-init'])$(id).disabled=!supported;const select=$('body-command-name'),whole=supported?f.physics.wholeBody:null,t=f.physics.tendons,entries=[];
 if(whole)whole.names.forEach((name,i)=>{if(!['lf','lm','lh','rf','rm','rh'].includes(whole.regions[i]))entries.push(['joint:'+name,name+' · rad',whole.limits_rad[i]]);});
 if(t)t.names.forEach((name,i)=>entries.push(['tendon:'+name,name+' · 힘줄 모터',t.limits[i]]));
 const signature=entries.map(e=>e[0]).join('|');if(select.dataset.signature!==signature){select.replaceChildren(...entries.map(([value,label,limits])=>new Option(label+' ['+limits.join(', ')+']',value)));select.dataset.signature=signature;}
 $('body-command-apply').disabled=!entries.length;
 const mapping=new Map((t?.neural_controls?.controls||[]).map(row=>[row.name,row]));
 const [kind,selected]=select.value.split(':');$('body-command-neural').disabled=kind!=='tendon'||!mapping.has(selected)||mapping.get(selected).status==='unmapped';
 $('tendon-status').textContent=t?t.names.map((name,i)=>{const row=mapping.get(name),status=!row||row.status==='unmapped'?'신경 미연결':row.status==='population_hypothesis'?'신경 연결 · 복부 집단 가설':'신경 연결 · 장힘줄 근육 매핑',mode=t.neural_controls?.modes[name]==='neural'?'신경 구동':'수동 구동';return `${name}: ${status} · ${mode}\n입력 ${t.inputs[i].toFixed(3)}, 힘 ${t.actuator_forces[i].toFixed(4)}, 관절각 ${t.angles_rad[i].map(v=>v.toFixed(3)).join(', ')} rad`;}).join('\n')+(t.neural_controls?.controls.some(row=>row.status==='population_hypothesis')?'\n복부는 MNad26의 좌우 평균 → 양의 pitch, 좌우 차이 → yaw를 사용하는 실험용 가설입니다. 목표 근육·생리적 방향은 미확정입니다.':''):'힘줄 구동을 사용하지 않습니다.';
}
 $('body-command-name').addEventListener('change',()=>{if(state.frame)renderBodyControls(state.frame);});
button('body-command-apply',async()=>{workbench.pause();const [kind,name]=$('body-command-name').value.split(':');const payload=kind==='joint'?{targets:{[name]:Number($('body-command-value').value)}}:{tendon_inputs:{[name]:Number($('body-command-value').value)}};applyFrame(await rpc.request('command',{type:'body_actuation',payload}));tell('직접 구동 명령을 적용했습니다. 한 단계 또는 재생으로 진행합니다.');});
button('body-command-neural',async()=>{workbench.pause();const [kind,name]=$('body-command-name').value.split(':');if(kind!=='tendon')return;applyFrame(await rpc.request('command',{type:'body_actuation',payload:{tendon_modes:{[name]:'neural'}}}));tell('선택한 힘줄을 신경 구동으로 되돌렸습니다.');});
let sharedCheckpoint=null;
function showShared(result){const obs=result.observation||result;const rows=Object.entries(obs.flies||{});$('shared-status').textContent=rows.length?`${obs.time_s.toFixed(3)} s · ${rows.length}개체 · 개체 간 접촉 ${obs.interfly_contacts.length}개\n`+rows.map(([name,f])=>`${name}: 위치 ${f.position_mm.map(v=>v.toFixed(2)).join(', ')} mm · 구동 ${f.drive.join(', ')}`).join('\n'):'공유 공간이 없습니다.';$('shared-status').dataset.time=String(obs.time_s||0);if(result.render_error){$('shared-status').textContent+='\n영상 오류: '+result.render_error;$('shared-image').hidden=true;}if(result.image){$('shared-image').src='data:image/png;base64,'+result.image;$('shared-image').hidden=false;}const names=Object.keys(obs.flies||{});if(names.length){const select=$('shared-fly');if([...select.options].map(o=>o.value).join()!==names.join())select.replaceChildren(...names.map(n=>new Option(n,n)));}for(const id of ['shared-step','shared-save','shared-close','shared-drive'])$(id).disabled=!names.length;}
button('shared-init',async()=>{workbench.pause();showShared(await rpc.request('shared_init',{count:Number($('shared-count').value),seed:Number($('seed').value)}));sharedCheckpoint=null;$('shared-restore').disabled=true;});
button('shared-step',async()=>{workbench.pause();showShared(await rpc.request('shared_advance',{steps:20}));});
button('shared-drive',async()=>showShared(await rpc.request('shared_drive',{drives:{[$('shared-fly').value]:[Number($('shared-left').value),Number($('shared-right').value)]}})));
button('shared-save',async()=>{const r=await rpc.request('shared_save',{});sharedCheckpoint=r.name;$('shared-restore').disabled=false;tell('공유 공간 상태를 저장했습니다.');});
button('shared-restore',async()=>showShared(await rpc.request('shared_restore',{name:sharedCheckpoint})));
button('shared-close',async()=>{await rpc.request('shared_close',{});showShared({closed:true});$('shared-image').hidden=true;$('shared-restore').disabled=true;sharedCheckpoint=null;});
function selectedBodyOptions(){const full=$('body-model').value==='flybody_whole';return {model:full?'flybody':$('body-model').value,terrain:$('body-terrain').value,...($('body-attachment').value==='tethered'?{attachment:'tethered'}:{}),...($('body-tendons').value!=='none'?{tendons:$('body-tendons').value}:{}),...(full?{actuation:'whole_body',servo_profile:'tracking_all'}:{})};}
async function init(){
 state.playing=false;updatePlay();$('waiting').hidden=false;
 const payload={mode:$('mode').value,seed:Number($('seed').value),metabolism:$('metabolism-enabled').checked?{}:null};
 payload.body_options=selectedBodyOptions();if(payload.body_options.attachment==='tethered')payload.motion_expected=false;
 // Before attach/init there is no catalog: honor the server's --bindings.
 if(state.frame)payload.profile=$('binding-profile').value;
 let result;try{result=await rpc.request('init',payload);}catch(e){$('waiting').hidden=!!state.frame;throw e;}
 state.signals.clear();state.traces.clear();state.spikes=0;state.sequence=0;applyReady(result);
 const recovery=result.recovery?.kind==='fault_reset'?(result.recovery.diagnostic_saved?' 오류 진단 저장: '+result.recovery.diagnostic_id:' 오류 진단은 메모리에 보존했습니다. 파일 저장 실패: '+result.recovery.diagnostic_save_error):'';
 $('search').value='DNa02';await search();await regions();await checkpoints();await workbench.ready();tell('실험을 준비했습니다.'+(result.source_checkpoint?' 이전 상태 저장: '+result.source_checkpoint:'')+recovery+(result.cleanup_error?' 이전 실험 정리 오류: '+result.cleanup_error:'')+' 재생 또는 한 단계로 진행하세요.');
}
async function command(type,payload={}){const r=await rpc.request('command',{type,payload});if(r.schema==='flylab.frame.v3')applyFrame(r);return r;}
button('play',()=>{state.playing=!state.playing;updatePlay();});
button('leg-feedback-off',async()=>{workbench.pause();await command('intervene',{kind:'sensor_off',channels:['leg_feedback'],duration_controls:200});applyFrame(await rpc.request('frame'));tell('다리 감각만 1모델초 차단하도록 예약했습니다. 후각 입력은 유지됩니다.');});
button('step',async()=>{state.playing=false;updatePlay();applyFrame(await rpc.request('advance',{steps:10}));});
button('init',init);button('track',()=>view.track());button('wide',()=>view.home());button('top',()=>view.top());button('eye',()=>{view.follow=true;view.firstPerson=true;});
$('backend').addEventListener('change',async()=>{
 const backend=$('backend').value;state.playing=false;updatePlay();$('backend').disabled=true;
 try{const r=await rpc.request('backend',{backend});await refreshExperiment(r,'계산 장치를 전환했습니다. 모델 시간과 상태를 유지했고 이전 상태도 저장했습니다: '+r.source_checkpoint);}
 catch(e){$('backend').value=state.backend;tell(e.message,true);}
 finally{$('backend').disabled=state.backend==='legacy_b_rate';}
});
button('search-button',()=>search());$('search').addEventListener('keydown',e=>{if(e.key==='Enter')search().catch(e=>tell(e.message,true));});
button('previous',()=>{state.offset=Math.max(0,state.offset-50);return search(false);});button('next',()=>{state.offset+=50;return search(false);});
button('clear-selection',async()=>{applyFrame(await rpc.request('subscribe',{ids:[]}));state.traces.clear();drawTrace();});
button('intervene',async()=>{workbench.pause();const result=await command('intervene',workbench.interventionSpec());applyFrame(await rpc.request('frame'));tell(`개입 #${result.serial} 예약: tick ${result.at_tick}. 재생 또는 한 단계로 진행하세요.`);});
button('forward-stim',async()=>{const presets=await rpc.request('presets');const ids=presets.find(p=>p.name==='Forward output population')?.ids||[];if(!ids.length)throw Error('현재 프로파일에 전진 출력 집단이 없습니다.');workbench.pause();workbench.state.intervention=ids;await command('intervene',{kind:'stimulate',ids,amplitude_mV:20.,duration_controls:100});applyFrame(await rpc.request('frame'));tell('현재 전진 출력 집단 '+ids.length+'개에 20mV / 0.5모델초 직접 자극을 예약했습니다. 재생 또는 한 단계로 진행하세요.');});
button('release',()=>command('release_all'));button('push',()=>command('push',{bw:.5,duration:.05}));
for(const [id,type,key] of [['motor','configure','motorCoupled'],['cue','cue','enabled'],['food','food','enabled']])$(id).addEventListener('change',()=>command(type,{[key]:$(id).checked}).catch(e=>tell(e.message,true)));
 $('friction').addEventListener('change',()=>command('configure',{friction:Number($('friction').value)}).catch(e=>tell(e.message,true)));
button('regions-refresh',regions);
let observationBusy=false,observationLastWall=0,observationLastTick=-1;
async function observePhysical(kind,pause=true){
 if(observationBusy)return;
 if(pause)workbench.pause();
 observationBusy=true;
 const ids=['native','eyes','odor'].map(k=>'observe-'+k);ids.forEach(id=>$(id).disabled=true);
 try{
  const result=await rpc.request('physical_observation',{kind});
  $('physical-observation').hidden=false;
  $('physical-observation-time').textContent=`모델 시각 ${result.time_s.toFixed(4)} s · `+(kind==='eyes'?'왼쪽·오른쪽 복안, 눈당 721개 낱눈':kind==='native'?'실제 MuJoCo 메시':'더듬이 2곳·palp 2곳의 가상 냄새 반응');
  const img=$('physical-observation-image');img.hidden=!result.image;if(result.image)img.src=result.image;
  const values=$('physical-observation-values');values.hidden=kind!=='odor';
  if(kind==='odor')values.textContent=result.sites.map((site,i)=>`${site}: 먹이 ${result.response[i][0].toFixed(4)} · 위험 ${result.response[i][1].toFixed(4)}`).join('\n')+'\n'+(result.model.site_calibration?'외부 위치 교정 자료: '+JSON.stringify(result.model.site_calibration.evidence):'palp 위치는 해부학적 교정 전의 가정입니다.');
  const now=performance.now();$('observe-rate').textContent=observationLastWall?`표시 ${(1000/(now-observationLastWall)).toFixed(1)} FPS`:'';
  observationLastWall=now;observationLastTick=result.control_tick??Math.round(result.time_s/.005);
 }finally{observationBusy=false;ids.forEach(id=>$(id).disabled=false);}
}
for(const kind of ['native','eyes','odor'])button('observe-'+kind,()=>observePhysical(kind));
setInterval(()=>{
 // The server serializes observation and integration. Waiting for the narrow
 // playback idle gap can starve previews during continuous calculation.
 if(!$('observe-live').checked||document.hidden||state.closed||!state.frame||observationBusy)return;
 if(performance.now()-observationLastWall<1000/Number($('observe-fps').value))return;
 if(observationLastTick===state.frame.controlTick)return;
 observePhysical($('observe-kind').value,false).catch(e=>{$('observe-live').checked=false;tell(e.message,true);});
},50);
$('observe-kind').addEventListener('change',()=>{observationLastTick=-1;});
let batchState=null;
const batchButtons=['step','save','close','observe','pause','resume','cancel','record','intervene','release'];
function batchControls(){
 const w=batchState?.worlds[$('batch-world').value];
 for(const id of batchButtons)$('batch-'+id).disabled=!batchState;
 $('batch-init').disabled=!!batchState;$('batch-restore').disabled=!!batchState;
 const failed=Object.values(batchState?.worlds||{}).some(world=>world.status==='FAILED');
 $('batch-step').disabled=!batchState?.active_worlds?.length||failed;
 $('batch-save').disabled=!batchState||failed;
 if(!w)return;
 $('batch-pause').disabled=w.status!=='RUNNING';$('batch-resume').disabled=w.status!=='PAUSED';
 for(const id of ['cancel','record','intervene','release'])$('batch-'+id).disabled=!['RUNNING','PAUSED'].includes(w.status);
 $('batch-record').textContent=w.recording?'선택 세계 기록 종료':'선택 세계 기록';
}
function showBatch(result){
 batchState=result;const selected=$('batch-world').value;$('batch-world').replaceChildren();
 for(const [id,world] of Object.entries(result.worlds)){const option=document.createElement('option');option.value=id;option.textContent=`세계 ${id} · ${world.status}`;$('batch-world').append(option);}
 if(result.worlds[selected])$('batch-world').value=selected;
 $('batch-status').textContent=Object.entries(result.worlds).map(([id,w])=>`세계 ${id}: ${w.model_seconds.toFixed(3)} s · ${w.status}${w.recording?' · 기록 중':''}${w.fault?' · '+w.fault:''}`).join('\n');
 batchControls();
}
async function batchCheckpoints(){const rows=await rpc.request('batch_checkpoints');const selected=$('batch-checkpoints').value;$('batch-checkpoints').replaceChildren(new Option('저장한 CUDA 배치',''),...rows.map(r=>new Option(r.name,r.name)));if(rows.some(r=>r.name===selected))$('batch-checkpoints').value=selected;}
async function batchRequest(op,payload){
 try{return await rpc.request(op,payload);}catch(error){
  try{showBatch(await rpc.request('batch_summary'));}catch{}
  throw error;
 }
}
$('batch-world').addEventListener('change',()=>{batchControls();$('batch-image').hidden=true;$('batch-image-time').textContent='';});
button('batch-init',async()=>{if(selectedBodyOptions().actuation==='whole_body'||selectedBodyOptions().attachment==='tethered'||selectedBodyOptions().tendons!=='none')throw new Error('전신·몸통 고정·힘줄은 CPU 물리의 단일 실험에서 지원합니다.');workbench.pause();$('batch-init').disabled=true;try{showBatch(await rpc.request('batch_init',{worlds:Number($('batch-count').value),seed:Number($('seed').value),profile:$('binding-profile').value,mode:$('mode').value==='B_COMPAT'?'C_SHADOW':$('mode').value,body_options:selectedBodyOptions()}));tell('별도 CUDA 배치를 준비했습니다.');}finally{batchControls();}});
button('batch-step',async()=>showBatch(await batchRequest('batch_advance',{steps:1})));
for(const action of ['pause','resume','cancel'])button('batch-'+action,async()=>showBatch(await batchRequest('batch_control',{action,world:$('batch-world').value})));
button('batch-save',async()=>{const r=await rpc.request('batch_checkpoint');await batchCheckpoints();$('batch-checkpoints').value=r.name;tell('전체 배치 저장: '+r.name);});
button('batch-refresh',batchCheckpoints);
button('batch-restore',async()=>{const name=$('batch-checkpoints').value;if(!name)throw Error('복원할 CUDA 배치를 선택하세요.');workbench.pause();showBatch(await rpc.request('batch_restore',{name}));tell('배치 복원 완료: '+name);});
button('batch-record',async()=>{const world=$('batch-world').value,action=batchState.worlds[world].recording?'stop':'start';showBatch(await rpc.request('batch_record',{world,action,ids:workbench.recordIds()}));});
button('batch-intervene',async()=>showBatch(await rpc.request('batch_intervention',{world:$('batch-world').value,type:'intervene',payload:workbench.interventionSpec()})));
button('batch-release',async()=>showBatch(await rpc.request('batch_intervention',{world:$('batch-world').value,type:'release_all',payload:{}})));
button('batch-observe',async()=>{const r=await rpc.request('batch_observation',{world:$('batch-world').value});$('batch-image').src=r.image;$('batch-image').hidden=false;$('batch-image-time').textContent=`세계 ${r.world} · ${r.time_s.toFixed(3)} s · ${r.renderer}`;tell($('batch-image-time').textContent);});
button('batch-close',async()=>{await rpc.request('batch_close');batchState=null;batchControls();$('batch-status').textContent='배치를 닫았습니다.';$('batch-image').hidden=true;});
button('record',async()=>{if(state.recording){applyFrame(await rpc.request('record_stop'));tell('기록을 저장했습니다.');}else{workbench.pause();const r=await rpc.request('record_start',{ids:workbench.recordIds()});applyFrame(r.frame);tell('기록 시작: '+r.name);}});
button('checkpoint',async()=>{state.playing=false;updatePlay();const r=await rpc.request('checkpoint');tell(`체크포인트 저장: ${r.name} · ${(r.bytes/1048576).toFixed(1)} MB`);await checkpoints();$('checkpoints').value=r.name;});
button('restore',async()=>{const name=$('checkpoints').value;if(!name)throw Error('복원할 체크포인트를 선택하세요.');state.playing=false;updatePlay();const r=await rpc.request('restore',{name});state.sequence=0;state.spikes=0;state.signals.clear();state.traces.clear();applyReady(r);await regions();await workbench.ready();tell('복원 완료: '+name);});
button('paired',async()=>{workbench.pause();const kind=$('pair-kind').value,intervention={kind};if(kind==='sensor_off')intervention.channels=['*'];if(['suppress_spiking','stimulate','mute_outgoing'].includes(kind)){intervention.ids=workbench.interventionIds();if(!intervention.ids.length)throw Error('개입 집단을 지정하세요.');}if(kind==='stimulate')intervention.amplitude_mV=Number($('amplitude').value);
 const payload={intervention,seconds:Number($('pair-seconds').value),onset:Number($('pair-onset').value),ids:workbench.recordIds(),repeats:Number($('pair-repeats').value),origin:$('pair-origin').value};if(payload.origin==='fresh')payload.seed=Number($('pair-seed').value);
 $('paired').disabled=true;$('experiment-result').textContent='같은 상태의 대조군·개입군을 계산합니다. 현재 실험은 보존됩니다…';
 try{const r=await rpc.request('paired',payload);workbench.showComparison(r);$('experiment-result').textContent=JSON.stringify(r,(key,value)=>key==='preview'?undefined:value,2);tell('대조 비교 원자료를 저장했습니다. 기술 실행과 과제 행동 판정은 결과에서 별도로 표시합니다.');}finally{$('paired').disabled=false;}
});
document.addEventListener('visibilitychange',()=>{if(document.hidden){state.playing=false;updatePlay();}});
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(document.activeElement?.tagName))return;if(e.code==='Space'){e.preventDefault();state.playing=!state.playing;updatePlay();}if(e.key==='f')view.track();});
window.addEventListener('resize',drawTrace);
playback=F.createPlaybackPump({
 shouldRun:()=>state.playing&&!state.closed&&!!state.frame&&!document.hidden,
 advance:async steps=>applyFrame(await rpc.request('advance',{steps})),
 onBusy:busy=>{state.busy=busy;},
 onError:e=>{state.playing=false;updatePlay();tell(e.message,true);}
});
let lastDraw=0;
function animate(now){requestAnimationFrame(animate);workbench.animate(now,false);if(!document.hidden&&now-lastDraw>=33&&view.needsDraw()){view.draw();workbench.drawPreview();lastDraw=now;}}
requestAnimationFrame(animate);
rpc.connect().then(async(config)=>{
 $('status-dot').className='ready';
 for(const option of $('backend').options){const info=config.backendAvailability?.[option.value];option.disabled=info?.available===false;option.title=info?.reason||info?.device||'';}
 await batchCheckpoints();if(config.hasBatch)showBatch(await rpc.request('batch_summary'));
 if(config.hasExperiment){await refreshExperiment(await rpc.request('attach'),'저장된 현재 실험에 연결했습니다. 재생으로 계속 진행하세요.');}else await init();
}).catch(e=>{$('waiting').textContent='초기화 실패';tell(e.message,true);});
})(globalThis.Fly);
