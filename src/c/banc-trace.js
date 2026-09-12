/* Measured 5ms poses only. Display/replay never steps or edits the simulator. */
(function(F){
'use strict';
class BancTimeline {
 constructor(capacity=6000){this.capacity=capacity;this.reset(null);}
 reset(epoch){this.epoch=epoch;this.samples=[];this.evicted=0;this.gaps=0;}
 ingest(packet){
  if(!packet||packet.schema!=='flylab.banc-trace.v1')throw Error('BANC 기록 형식이 다릅니다.');
  if(packet.epoch!==this.epoch)this.reset(packet.epoch);
  for(const row of packet.samples){
   if(!Number.isInteger(row.control_tick)||!Number.isFinite(row.time_s)||Math.abs(row.time_s-row.control_tick*.005)>1e-10)throw Error('BANC 기록 시각 오류');
   const last=this.samples.at(-1);
   if(last&&row.control_tick<=last.control_tick)continue;
   if(last)this.gaps+=Math.max(0,row.control_tick-last.control_tick-1);
   this.samples.push(row);
  }
  const excess=Math.max(0,this.samples.length-this.capacity);
  if(excess){this.samples.splice(0,excess);this.evicted+=excess;}
 }
 at(time){
  if(!this.samples.length)return null;
  let lo=0,hi=this.samples.length;
  while(lo<hi){const mid=(lo+hi)>>1;if(this.samples[mid].time_s<=time)lo=mid+1;else hi=mid;}
  // Zero-order hold, not synthetic contact/pose interpolation or extrapolation.
  return this.samples[Math.max(0,lo-1)];
 }
}
F.BancTimeline=BancTimeline;
F.createBancTraceView=function({canvas,overlay,status,diagnostic,slider}){
 const timeline=new BancTimeline(),view=new F.WorldView(canvas,overlay);
 view.distance=10;view.showTrail=false;
 let world=null,mode='live',speed=.25,cursor=0,lastWall=null,lastReceived=null,ratio=.18,lastDrawn=null;
 function ingest(result){
  if(result.trace_epoch&&result.trace_epoch!==timeline.epoch){timeline.reset(result.trace_epoch);lastDrawn=null;cursor=0;lastWall=null;lastReceived=null;mode='live';}
  if(!result.trace)return;
  const old=timeline.samples.at(-1),now=performance.now();
  timeline.ingest(result.trace);world=result.world||world;
  if(lastReceived!==null&&old){const delta=timeline.samples.at(-1).time_s-old.time_s;const wall=(now-lastReceived)/1000;if(delta>0&&wall>0)ratio=Math.min(10,Math.max(.001,.7*ratio+.3*delta/wall));}
  lastReceived=now;
  if(!lastDrawn&&timeline.samples.length){cursor=timeline.samples[0].time_s;view.target=[...timeline.samples[0].body.position];}
 }
 function replay(value=1){lastDrawn=null;if(!timeline.samples.length)return;mode='replay';speed=value;cursor=timeline.samples[0].time_s;lastWall=null;}
 function live(){lastDrawn=null;mode='live';cursor=timeline.samples.at(-1)?.time_s||0;lastWall=null;}
 function seek(fraction){lastDrawn=null;if(!timeline.samples.length)return;mode='paused';const first=timeline.samples[0].time_s,last=timeline.samples.at(-1).time_s;cursor=first+(last-first)*Math.max(0,Math.min(1,fraction));lastWall=null;}
 function draw(now){
  const dt=lastWall===null?0:Math.min(.1,(now-lastWall)/1000);lastWall=now;
  const first=timeline.samples[0],latest=timeline.samples.at(-1);
  if(!first||!world)return;
  if(mode==='live'){
   cursor=Math.max(cursor,first.time_s);
   // Bound visual delay, but never invent a future physical state.
   if(latest.time_s-cursor>.15)cursor=latest.time_s-.025;
   cursor=Math.min(latest.time_s,cursor+dt*ratio);
  }else if(mode==='replay'){
   cursor=Math.min(latest.time_s,cursor+dt*speed);
   if(cursor>=latest.time_s)mode='paused';
  }
  const sample=timeline.at(cursor);
  if(sample!==lastDrawn){
   view.setFrame({tick:sample.control_tick,simTime:sample.time_s,body:sample.body,world,config:{mode:'BANC'}});
   const label=mode==='live'?'지연 표시 · 실제 기록':mode==='replay'?`기록 재생 ${speed}×`:'기록 정지';
   status.textContent=`${label} | 표시 ${sample.time_s.toFixed(3)} 모델초 / 계산 ${latest.time_s.toFixed(3)} 모델초 | 5ms 측정 자세, 보간 없음 | 기록 ${timeline.samples.length}개${timeline.gaps?' | 누락 '+timeline.gaps+'구간':''}${timeline.evicted?' | 오래된 기록 '+timeline.evicted+'개 제외':''}`;
   diagnostic.textContent=sample.feet.legs.map((leg,i)=>`${leg.toUpperCase()}: 하중 ${sample.feet.normal_load_bw[i].toFixed(3)} BW · ${sample.feet.load_bearing[i]?'지지':'비지지'} · 접촉점 미끄러짐 ${sample.feet.slip_mm_s[i]===null?'측정 대상 아님':sample.feet.slip_mm_s[i].toFixed(3)+' mm/s'} · 부착 ${sample.adhesion[i]?'켜짐':'꺼짐'}`).join('\n')+'\n하중은 부착 반력을 포함합니다. 5ms 표본의 진단값이며 보행 성공 판정이 아닙니다.';
   const span=latest.time_s-first.time_s;slider.value=span>0?String((sample.time_s-first.time_s)/span):'0';
   lastDrawn=sample;
  }
  if(view.needsDraw())view.draw();
 }
 function reset(){timeline.reset(null);mode='live';cursor=0;lastDrawn=null;lastReceived=null;lastWall=null;view.frame=null;status.textContent='새 BANC 상태 기록을 기다립니다.';diagnostic.textContent='접촉 진단 기록이 없습니다.';}
 return {ingest,replay,live,seek,draw,reset,view,timeline};
};
})(globalThis.Fly);
