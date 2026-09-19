const assert=require('node:assert/strict');
const {test}=require('node:test');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const app=fs.readFileSync(path.join(__dirname,'../src/c/app.js'),'utf8');
const pump=fs.readFileSync(path.join(__dirname,'../src/c/playback.js'),'utf8');
function harness(){
 const h={time:0,frames:[],handlers:{},nodes:new Map(),timers:new Map(),steps:[],mode:'live'};
 let serial=0;
 const context=vm.createContext({Fly:{},state:{closed:false},Promise,
  $:id=>{if(!h.nodes.has(id))h.nodes.set(id,{value:'.5',addEventListener:(event,fn)=>{h.handlers[id+':'+event]=fn;}});return h.nodes.get(id);},
  button:(id,fn)=>{h.handlers[id]=fn;},
  bancTraceView:{replay:s=>{h.mode='replay';h.frames.push(['replay',s,h.time]);},seek:s=>{h.mode='seek';h.frames.push(['seek',s,h.time]);},live:()=>{h.mode='live';}},
  rpc:{tail:Promise.resolve()},setTimeout,clearTimeout,
 });
 vm.runInContext(pump,context);
 h.context=context;
 h.pump=context.Fly.createPlaybackPump({shouldRun:()=>vm.runInContext('bancPlaying',context),now:()=>h.time,
  schedule:fn=>{h.timers.set(++serial,fn);return serial;},cancel:id=>h.timers.delete(id),onError:()=>{},
  advance:()=>new Promise(resolve=>h.steps.push(()=>{h.time+=.01;resolve();}))});
 context.bancPlayback=h.pump;
 vm.runInContext('let bancPlaying=true,bancAction=0;'+app.slice(app.indexOf('function pauseBanc(){'),app.indexOf('function showBancNavigation(')),context);
 vm.runInContext(app.slice(app.indexOf("button('banc-view-live'"),app.indexOf("button('banc-view-track'")),context);
 h.start=()=>{h.pump.sync();const [id,fn]=h.timers.entries().next().value;h.timers.delete(id);return fn();};
 return h;
}

test('real replay and seek handlers wait for an in-flight automatic advance',async()=>{
 for(const [name,mode] of [['banc-replay','replay'],['banc-replay-slow','replay'],['banc-replay-seek:input','seek']]){
  const h=harness();const advance=h.start(),action=h.handlers[name]();
  await Promise.resolve();await Promise.resolve();assert.equal(h.mode,'live');assert.equal(h.frames.length,0);
  h.steps[0]();await advance;await action;
  assert.equal(h.mode,mode);assert.equal(h.frames[0][2],.01);assert.equal(h.timers.size,0);
 }
});

test('replay also waits for a manual step in the serialized transport',async()=>{
 const h=harness();let finish;
 h.context.rpc.tail=new Promise(resolve=>finish=()=>{h.time=.05;resolve();});
 const action=h.handlers['banc-replay']();await Promise.resolve();assert.equal(h.frames.length,0);
 finish();await action;assert.deepEqual(h.frames,[['replay',1,.05]]);assert.equal(h.steps.length,0);
});

test('seek retains the user position when the last live frame updates the slider',async()=>{
 const h=harness();const advance=h.start();
 h.nodes.get('banc-replay-seek').value='.3';
 const seek=h.handlers['banc-replay-seek:input']();
 h.nodes.get('banc-replay-seek').value='1';
 h.steps[0]();await advance;await seek;
 assert.deepEqual(h.frames,[['seek',.3,.01]]);
});

test('a later seek or live selection supersedes a waiting replay',async()=>{
 const h=harness();const advance=h.start(),replay=h.handlers['banc-replay']();
 const seek=h.handlers['banc-replay-seek:input']();h.steps[0]();await advance;await Promise.all([replay,seek]);
 assert.deepEqual(h.frames,[['seek',.5,.01]]);
 const k=harness();const pending=k.start(),old=k.handlers['banc-replay']();k.handlers['banc-view-live']();
 k.steps[0]();await pending;await old;assert.equal(k.frames.length,0);assert.equal(k.mode,'live');
});
