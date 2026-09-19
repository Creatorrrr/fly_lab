const assert=require('node:assert/strict');
const {test}=require('node:test');
globalThis.Fly={WorldView:class{
 constructor(canvas){this.canvas=canvas;this.trail={clear(){}};this.frame=null;}
 setFrame(value){this.frame=value;}
 needsDraw(){return false;}
}};
require('../src/c/banc-trace.js');
const T=Fly.BancTimeline,row=i=>({control_tick:i,time_s:i*.005,body:{position:[i,0,0]},
 feet:{legs:['lf'],normal_load_bw:[.03],load_bearing:[true],slip_mm_s:[i]},adhesion:[true]});
const packet=(epoch,ids,extra={})=>({schema:'flylab.banc-trace.v1',epoch,samples:ids.map(row),...extra});
function harness(){
 const h={canvas:{style:{}},overlay:{style:{}},status:{},diagnostic:{},slider:{}};
 h.trace=Fly.createBancTraceView(h);
 h.ingest=(epoch,ids,extra={})=>h.trace.ingest({trace_epoch:epoch,trace:packet(epoch,ids,extra),world:{}});
 return h;
}

test('timeline retains exact measured poses and accounts for bounded history',()=>{
 const t=new T(3);
 t.ingest(packet('a',[0,1,2]));assert.equal(t.at(.007).control_tick,1);assert.equal(t.at(10).control_tick,2);
 t.ingest(packet('a',[2,3]));assert.equal(t.samples.length,3);assert.equal(t.evicted,1);assert.equal(t.gaps,0);
 t.ingest(packet('a',[6]));assert.equal(t.gaps,2);assert.equal(t.at(.02).control_tick,3);
 assert.equal(t.gapAt(.02),true);assert.equal(t.gapAt(.03),false);
 t.ingest(packet('b',[0]));assert.equal(t.samples.length,1);assert.equal(t.epoch,'b');assert.equal(t.gaps,0);assert.equal(t.evicted,0);
 assert.throws(()=>t.ingest({schema:'wrong',epoch:'b',samples:[]}));
 assert.throws(()=>t.ingest({schema:'flylab.banc-trace.v1',epoch:'b',samples:[{control_tick:1,time_s:2}]}));
});

test('first packet reports server eviction independently of the browser buffer',()=>{
 const h=harness();
 h.ingest('a',[100,101],{evicted_samples:100,gap_before_page:true});h.trace.draw(0);
 assert.equal(h.trace.timeline.serverEvicted,100);assert.equal(h.trace.timeline.evicted,0);
 assert.match(h.status.textContent,/서버 버퍼 100개 제외/);assert.match(h.status.textContent,/첫 표본 이전 기록 누락/);
 h.ingest('a',[101],{evicted_samples:105});h.trace.draw(1);
 assert.equal(h.trace.timeline.serverEvicted,105);assert.match(h.status.textContent,/서버 버퍼 105개 제외/);
 h.ingest('b',[200]);h.trace.draw(2);
 assert.equal(h.trace.timeline.serverEvicted,0);assert.equal(h.trace.timeline.gapBeforeFirst,false);
});

test('a new epoch clears the prior displayed future before its first trace arrives',()=>{
 const h=harness();h.ingest('a',[0,1]);h.trace.draw(0);
 assert.equal(h.trace.view.frame.tick,0);assert.equal(h.canvas.style.visibility,'');
 h.trace.ingest({trace_epoch:'b'});h.trace.draw(1);
 assert.equal(h.trace.view.frame,null);assert.equal(h.canvas.style.visibility,'hidden');
 assert.equal(h.overlay.style.visibility,'hidden');assert.match(h.status.textContent,/기다립니다/);
 assert.match(h.diagnostic.textContent,/기록이 없습니다/);assert.equal(h.slider.value,'0');
 h.ingest('b',[20,21]);h.trace.draw(2);
 assert.equal(h.trace.view.frame.tick,20);assert.equal(h.canvas.style.visibility,'');
 h.trace.reset();assert.equal(h.trace.view.frame,null);assert.equal(h.canvas.style.visibility,'hidden');
});

test('conflicting response and trace epochs are rejected before replacing the display',()=>{
 const h=harness();h.ingest('a',[0]);h.trace.draw(0);const before=h.trace.view.frame;
 assert.throws(()=>h.trace.ingest({trace_epoch:'b',trace:packet('a',[1]),world:{}}),/실험이 다릅니다/);
 assert.equal(h.trace.timeline.epoch,'a');assert.equal(h.trace.view.frame,before);
 h.trace.ingest({trace:packet('b',[30]),world:{}});h.trace.draw(1);
 assert.equal(h.trace.timeline.epoch,'b');assert.equal(h.trace.view.frame.tick,30);
});

test('seeking a missing interval marks the gap without inventing contact measurements',()=>{
 const h=harness();h.ingest('a',[0,1,6]);h.trace.seek(.5);h.trace.draw(0);
 assert.equal(h.trace.view.frame.tick,1);assert.match(h.status.textContent,/기록 공백/);
 assert.match(h.diagnostic.textContent,/측정되지 않았습니다/);assert.equal(Number(h.slider.value),.5);
 h.trace.seek(1);h.trace.draw(1);assert.equal(h.trace.view.frame.tick,6);
 assert.doesNotMatch(h.status.textContent,/기록 공백/);assert.match(h.diagnostic.textContent,/6.000 mm\/s/);
});

test('the replay clock and seeking only select existing samples',()=>{
 const h=harness();h.ingest('a',[0,1,2,3,4,5]);const before=JSON.stringify(h.trace.timeline.samples);
 h.trace.replay(1);h.trace.draw(100);h.trace.draw(110);assert.equal(h.trace.view.frame.tick,2);
 h.trace.draw(125);assert.equal(h.trace.view.frame.tick,5);h.trace.draw(2000);assert.equal(h.trace.view.frame.tick,5);
 h.trace.replay(.25);h.trace.draw(2100);h.trace.draw(2120);assert.equal(h.trace.view.frame.tick,1);
 h.trace.seek(.8);h.trace.draw(2200);assert.equal(h.trace.view.frame.tick,4);
 assert.equal(JSON.stringify(h.trace.timeline.samples),before);
});

test('sparse browser drawing preserves playback speed and hidden time is paused',()=>{
 const h=harness();h.ingest('a',Array.from({length:1001},(_,i)=>i));
 h.trace.replay(1);h.trace.draw(0);h.trace.draw(1000);
 assert.equal(h.trace.view.frame.tick,200);
 h.trace.suspend();h.trace.draw(10000);assert.equal(h.trace.view.frame.tick,200);
 h.trace.replay(.25);h.trace.draw(11000);h.trace.draw(13000);
 assert.equal(h.trace.view.frame.tick,100);
});
