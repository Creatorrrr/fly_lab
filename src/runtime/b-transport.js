/* Versioned, same-origin, one-request-at-a-time RPC. No local/mock fallback. */
(function(F){
'use strict';F.VERSION='0.2.0';F.DT=.005;F.PROTOCOL='flylab.protocol.v2';
class BTransport{
 constructor(){this.name='PYTHON · MuJoCo';this.next=1;this.pending=new Map();this.tail=Promise.resolve();this.closed=false;}
 async connect(){
  if(!['http:','https:'].includes(location.protocol)||!['127.0.0.1','localhost'].includes(location.hostname))throw Error('B는 Python 서버가 필요합니다. ZIP을 풀고 python run.py 실행 후 http://127.0.0.1:8765 에 접속하세요. HTML 더블클릭만으로 물리 계산을 실행하지 않습니다.');
  const r=await fetch('/api/bootstrap',{cache:'no-store'});if(!r.ok)throw Error('로컬 서버 응답 오류 '+r.status);const config=await r.json();
  if(config.protocol!==F.PROTOCOL)throw Error('서버/화면 프로토콜 버전이 다릅니다.');
  if(!config.dependencies.ready&&!config.testDouble)throw Error(config.dependencies.hint+' 현재: '+JSON.stringify(config.dependencies.versions));
  if(config.testDouble){document.body.classList.add('test-fixture');document.getElementById('verification-banner').textContent='UI TEST FIXTURE · 합성 테스트 상태 · 실제 물리 실행 화면 아님';this.name='TEST DOUBLE · NOT PHYSICS';}
  await new Promise((resolve,reject)=>{
   const ws=this.ws=new WebSocket((location.protocol==='https:'?'wss:':'ws:')+'//'+location.host+'/ws');
   const timer=setTimeout(()=>{ws.close();reject(Error('서버 연결 시간 초과'));},15000);
   ws.onopen=()=>ws.send(JSON.stringify({protocol:F.PROTOCOL,token:config.token}));
   ws.onmessage=ev=>{let m;try{m=JSON.parse(ev.data);}catch{return;}
    if(m.authenticated){clearTimeout(timer);resolve();return;}
    const p=this.pending.get(m.requestId);if(!p)return;this.pending.delete(m.requestId);clearTimeout(p.timer);
    if(m.protocol!==F.PROTOCOL)p.reject(Error('프로토콜 불일치'));else if(m.ok)p.resolve(m.result);else p.reject(Error(m.error||'Server error'));
   };
   ws.onerror=()=>{clearTimeout(timer);reject(Error('WebSocket 연결 실패. 다른 관찰 창이 이미 연결되어 있는지 확인하세요.'));};
   ws.onclose=()=>{clearTimeout(timer);this.closed=true;for(const p of this.pending.values()){clearTimeout(p.timer);p.reject(Error('서버 연결이 끊겼습니다. 자동으로 A 계산기로 전환하지 않습니다.'));}this.pending.clear();reject(Error('서버 연결 종료'));};
  });return config;
 }
 request(op,payload={}){const work=()=>new Promise((resolve,reject)=>{
  if(this.closed||this.ws?.readyState!==WebSocket.OPEN){reject(Error('서버 미연결. 페이지를 다시 여세요.'));return;}
  const id=this.next++;const timer=setTimeout(()=>{this.close();reject(Error('계산 응답 시간 초과. 서버 로그를 확인하세요.'));},300000);
  this.pending.set(id,{resolve,reject,timer});this.ws.send(JSON.stringify({protocol:F.PROTOCOL,requestId:id,op,payload}));
 });const p=this.tail.then(work);this.tail=p.catch(()=>{});return p;}
 close(){this.closed=true;this.ws?.close();}
}
F.BTransport=BTransport;
})(globalThis.Fly);
