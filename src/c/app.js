/* C viewer: bounded subscriptions, raw units, explicit causality and source. */
(function(F){
'use strict';
const PROTOCOL='flylab.protocol.v3', $=id=>document.getElementById(id);
const view=new F.WorldView($('world'),$('world-overlay'));
const state={frame:null,playing:false,busy:false,selected:[],catalog:[],offset:0,total:0,epoch:0,sequence:0,
  signals:new Map(),traces:new Map(),spikes:0,closed:false,names:new Map(),recording:false,backend:'exp_lif_cpu_reference'};
const colors=['#76d9c4','#eac684','#87b4e8','#c9a1ed','#ed9994','#a6d883','#88d9e5','#dba3c6'];
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
   const requestId=this.id++,timer=setTimeout(()=>{this.ws.close();reject(Error('계산 응답 시간 초과'));},600000);
   this.pending.set(requestId,{resolve,reject,timer});this.ws.send(JSON.stringify({protocol:PROTOCOL,requestId,op,payload}));
  });const result=this.tail.then(run);this.tail=result.catch(()=>{});return result;
 }
}
const rpc=new Transport();
function updatePlay(){$('play').textContent=state.playing?'Ⅱ 일시정지':'▶ 재생';$('play').disabled=state.closed||!state.frame;$('step').disabled=state.closed||!state.frame||state.playing;}
function commandText(c){return c?`${c.forwardSpeed.toFixed(2)} mm/s · ${c.yawRate.toFixed(2)} rad/s`:'사용 안 함';}
function applyFrame(f){
 if(!f||f.schema!=='flylab.frame.v3')return;
 if(state.frame&&f.tick<state.frame.tick){state.traces.clear();state.spikes=0;}
 state.frame=f;state.epoch=f.subscription.epoch;state.selected=f.subscription.ids;view.setFrame(f);
 $('waiting').hidden=true;$('mode-tag').textContent=f.mode;$('clock').textContent=f.simTime.toFixed(3)+' s';
 $('scope').textContent=f.mode==='B_COMPAT'?'B_COMPAT · 기존 부분 신경 모델':`FAFB v783 · ${f.scope.simulatedNodes.toLocaleString()} 뉴런 · ${f.scope.pairEdges.toLocaleString()} 연결`;
 $('distance').textContent=f.body.travel.toFixed(2)+' mm';$('speed').textContent=f.body.speed.toFixed(2)+' mm/s';
 $('mean-rate').textContent=f.neural?f.neural.mean_rate_Hz.toFixed(3)+' Hz':'B rate';$('performance').textContent=f.performance.sim_wall_ratio.toFixed(3)+'×';
 $('command-source').textContent=f.command.command_source;$('neural-command').textContent=commandText(f.command.u_neural);
 $('assist-command').textContent=commandText(f.command.u_assist);$('final-command').textContent=commandText(f.command.u_final);
 const dt=f.neuralDt||.0001;$('causal-time').textContent=`감각 ${(f.command.sensor_tick*dt).toFixed(4)}s → 신경 출력 ${(f.command.neural_readout_tick*dt).toFixed(4)}s → 몸 구동 [${(f.command.interval_start_tick*dt).toFixed(4)}, ${(f.command.interval_end_tick*dt).toFixed(4)})s`+(f.command.assist_reason?' · '+f.command.assist_reason:'');
 $('port-note').textContent='현재 입력: '+f.binding.sensory.map(p=>`${p.channel} → ${p.targets} 뉴런 (${p.method})`).join(', ')+` · 미확정 전달물질 ${f.scope.unknownNeurotransmitters.toLocaleString()} 뉴런 / 효력 0인 연결 ${f.scope.maskedEdges.toLocaleString()}개`;
 $('motor').checked=f.config.motorCoupled;$('cue').checked=f.world.cueOn;$('food').checked=f.world.foodOn;
 $('body-status').textContent=f.physics.testDouble?'TEST DOUBLE':`${f.physics.jointNames.length} 관절 · ${f.physics.backend}`;
 const feet=$('feet');feet.replaceChildren();['LF','LM','LH','RF','RM','RH'].forEach((leg,i)=>{
  const row=document.createElement('div');row.className='foot';const label=document.createElement('span');label.textContent=leg;const number=document.createElement('b');number.textContent=f.physics.contactsBW[i].toFixed(2)+' BW';const bar=document.createElement('i');bar.style.width=Math.min(100,f.physics.contactsBW[i]*100)+'%';row.append(label,number,bar);feet.append(row);
 });
 if($('joints').closest('details').open){const body=$('joints');body.replaceChildren();f.physics.jointNames.forEach((name,i)=>{const tr=document.createElement('tr');[name,f.physics.jointAngles[i].toFixed(4),f.physics.jointTargets[i].toFixed(4),f.physics.actuatorForces[i].toFixed(4)].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td);});body.append(tr);});}
 $('events').replaceChildren(...f.events.slice(-8).reverse().map(e=>{const div=document.createElement('div');div.textContent=`${(e.tick*dt).toFixed(3)} s · ${e.kind}`;return div;}));
 state.recording=f.recording.active;$('record-status').textContent=f.recording.active?`기록 중 · ${(f.recording.bytes/1024).toFixed(1)} KB`:'기록 꺼짐';$('record').textContent=f.recording.active?'기록 종료':'기록 시작';
 $('selected-count').textContent=`${state.selected.length} / 512`;
 if(f.fault){state.playing=false;tell('실험 중단: '+f.fault,true);}updatePlay();renderSelection();
}
function receiveSignals(buffer){
 const view=new DataView(buffer);if(buffer.byteLength<8||String.fromCharCode(...new Uint8Array(buffer,0,4))!=='FLC3')throw Error('신호 프레임 형식 오류');
 const len=view.getUint32(4,true);if(len>1000000||8+len>buffer.byteLength)throw Error('신호 헤더 길이 오류');
 const h=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,8,len))),base=8+len,n=h.channel_count;
 if(h.schema!=='flylab.signals.v3'||h.dtype!=='<f4'||h.unit_descriptor?.join(',')!=='mV,Hz'||!Number.isInteger(n)||n<0||n>512||h.ids.length!==n||h.ids.some(i=>typeof i!=='string')||h.payload_bytes!==buffer.byteLength-base||h.payload_bytes!==n*8+h.spike_count*12)throw Error('신호 프레임의 단위·크기가 다릅니다.');
 if(h.subscription_epoch!==state.epoch||h.sequence<=state.sequence)return;
 state.sequence=h.sequence;
 for(let i=0;i<n;i++){
  const voltage=view.getFloat32(base+i*4,true),rate=view.getFloat32(base+n*4+i*4,true);
  if(!Number.isFinite(voltage)||!Number.isFinite(rate))throw Error('유한하지 않은 신경 신호');
  state.signals.set(h.ids[i],{voltage,rate});let trace=state.traces.get(h.ids[i])||[];
  const t=h.end_tick*h.neural_dt;
  if(!trace.length||trace[trace.length-1][0]!==t)trace.push([t,voltage]);
  while(trace.length&&trace[0][0]<t-15)trace.shift();if(trace.length>1000)trace=trace.slice(-1000);
  state.traces.set(h.ids[i],trace);
 }
 state.spikes+=h.spike_count;$('spike-count').textContent='발화 이벤트 '+state.spikes.toLocaleString();renderSelection();drawTrace();
}
function renderSelection(){
 const root=$('selection');root.replaceChildren();
 for(const [i,id] of state.selected.entries()){
  const value=state.signals.get(id),row=document.createElement('div');row.className='signal-row';
  const name=document.createElement('span');name.textContent=state.names.get(id)||id.replace('flywire:fafb:783:','');name.title=id;name.style.color=colors[i%colors.length];
  const voltage=document.createElement('span');voltage.className='voltage';voltage.textContent=value?value.voltage.toFixed(2)+' mV':'— mV';
  const rate=document.createElement('span');rate.className='rate';rate.textContent=value?value.rate.toFixed(2)+' Hz':'— Hz';row.append(name,voltage,rate);root.append(row);
 }
 if(!state.selected.length)root.textContent='검색 결과에서 모니터링할 뉴런을 선택하세요.';
 for(const row of $('catalog').children)row.classList.toggle('selected',state.selected.includes(row.dataset.id));
}
function drawTrace(){
 const c=$('trace'),r=c.getBoundingClientRect(),dpr=Math.min(2,devicePixelRatio||1);c.width=r.width*dpr;c.height=r.height*dpr;
 const ctx=c.getContext('2d');ctx.scale(dpr,dpr);const w=r.width,h=r.height;ctx.clearRect(0,0,w,h);
 const end=state.frame?.simTime||0,start=Math.max(0,end-15),span=Math.max(1,end-start);ctx.font='9px ui-monospace';
 for(const value of [-70,-60,-50,-40]){const y=12+(-35-value)/45*(h-28);ctx.strokeStyle='#23353d';ctx.beginPath();ctx.moveTo(29,y);ctx.lineTo(w,y);ctx.stroke();ctx.fillStyle='#7c959e';ctx.fillText(value,2,y+3);}
 state.selected.slice(0,8).forEach((id,i)=>{const points=state.traces.get(id)||[];ctx.strokeStyle=colors[i];ctx.lineWidth=1.3;ctx.beginPath();points.forEach(([t,v],j)=>{const x=30+(t-start)/span*(w-35),y=12+(-35-v)/45*(h-28);j?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();});
}
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
function applyReady(result){
 applyFrame(result.frame);$('mode').value=result.frame.mode;$('seed').value=result.frame.seed;
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
 $('search').value='DNa02';await search();await regions();await checkpoints();tell(message);
}
async function init(){
 state.playing=false;updatePlay();$('waiting').hidden=false;state.signals.clear();state.traces.clear();state.spikes=0;state.sequence=0;
 const result=await rpc.request('init',{mode:$('mode').value,seed:Number($('seed').value)});applyReady(result);
 $('search').value='DNa02';await search();await regions();await checkpoints();tell('실험을 준비했습니다. 재생하거나 직접 자극으로 신경–몸 연결을 검사하세요.');
}
async function command(type,payload={}){const r=await rpc.request('command',{type,payload});if(r.schema==='flylab.frame.v3')applyFrame(r);return r;}
button('play',()=>{state.playing=!state.playing;updatePlay();});
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
button('intervene',async()=>{if(!state.selected.length)throw Error('개입할 뉴런을 선택하세요.');const result=await command('intervene',{kind:$('intervention-kind').value,ids:state.selected,amplitude_mV:Number($('amplitude').value),duration_controls:Math.round(Number($('duration').value)/.005)});tell(`개입 예약: ${result.kind} · tick ${result.at_tick}. 다음 계산 경계에서 적용됩니다.`);});
button('forward-stim',async()=>{const r=await rpc.request('catalog',{query:'DNp09',limit:50});const ids=r.items.filter(n=>n.cell_type==='DNp09').map(n=>n.id);if(ids.length!==2)throw Error('검토된 DNp09 쌍을 찾을 수 없습니다.');await command('intervene',{kind:'stimulate',ids,amplitude_mV:20.,duration_controls:100});tell('DNp09에 20mV / 0.5모델초 직접 자극을 예약했습니다. 재생 또는 한 단계로 진행하세요.');});
button('release',()=>command('release_all'));button('push',()=>command('push',{bw:.5,duration:.05}));
for(const [id,type,key] of [['motor','configure','motorCoupled'],['cue','cue','enabled'],['food','food','enabled']])$(id).addEventListener('change',()=>command(type,{[key]:$(id).checked}).catch(e=>tell(e.message,true)));
 $('friction').addEventListener('change',()=>command('configure',{friction:Number($('friction').value)}).catch(e=>tell(e.message,true)));
button('regions-refresh',regions);
button('record',async()=>{if(state.recording){applyFrame(await rpc.request('record_stop'));tell('기록을 저장했습니다.');}else{const r=await rpc.request('record_start');applyFrame(r.frame);tell('기록 시작: '+r.name);}});
button('checkpoint',async()=>{state.playing=false;updatePlay();const r=await rpc.request('checkpoint');tell(`체크포인트 저장: ${r.name} · ${(r.bytes/1048576).toFixed(1)} MB`);await checkpoints();$('checkpoints').value=r.name;});
button('restore',async()=>{const name=$('checkpoints').value;if(!name)throw Error('복원할 체크포인트를 선택하세요.');state.playing=false;updatePlay();const r=await rpc.request('restore',{name});state.sequence=0;state.spikes=0;state.signals.clear();state.traces.clear();applyReady(r);await regions();tell('복원 완료: '+name);});
button('paired',async()=>{state.playing=false;updatePlay();const kind=$('pair-kind').value,intervention={kind};if(kind==='sensor_off')intervention.channels=['*'];if(kind==='suppress_spiking'){if(!state.selected.length)throw Error('억제할 뉴런을 선택하세요.');intervention.ids=state.selected;}$('experiment-result').textContent='동일한 초기 상태에서 두 조건을 순서대로 계산합니다…';const r=await rpc.request('paired',{intervention,seconds:.5,onset:.2});$('experiment-result').textContent=JSON.stringify(r,null,2);tell('대조 비교 원자료를 저장했습니다. 결과는 모델 반응이며 생물학적 검증이 아닙니다.');});
document.addEventListener('visibilitychange',()=>{if(document.hidden){state.playing=false;updatePlay();}});
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(document.activeElement?.tagName))return;if(e.code==='Space'){e.preventDefault();state.playing=!state.playing;updatePlay();}if(e.key==='f')view.track();});
window.addEventListener('resize',drawTrace);
let lastAdvance=0,lastDraw=0;
function animate(now){requestAnimationFrame(animate);if(!document.hidden&&now-lastDraw>=33){view.draw();lastDraw=now;}if(state.playing&&!state.busy&&!document.hidden&&now-lastAdvance>=16){state.busy=true;lastAdvance=now;rpc.request('advance',{steps:state.backend==='exp_lif_mps'?2:10}).then(applyFrame).catch(e=>{state.playing=false;tell(e.message,true);}).finally(()=>{state.busy=false;updatePlay();});}}
requestAnimationFrame(animate);
rpc.connect().then(async(config)=>{$('status-dot').className='ready';if(config.hasExperiment){await refreshExperiment(await rpc.request('attach'),'저장된 현재 실험에 연결했습니다. 재생으로 계속 진행하세요.');}else await init();}).catch(e=>{$('waiting').textContent='초기화 실패';tell(e.message,true);});
})(globalThis.Fly);
