const assert=require('node:assert/strict');
const {test}=require('node:test');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const context={Fly:{}};
vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../src/c/playback.js'),'utf8'),context);

function harness(){
 const h={running:false,time:0,timers:new Map(),requests:[],busy:[],errors:[]};let serial=0;
 h.pump=context.Fly.createPlaybackPump({
  shouldRun:()=>h.running,now:()=>h.time,
  schedule:fn=>{h.timers.set(++serial,fn);return serial;},cancel:id=>h.timers.delete(id),
  advance:steps=>new Promise((resolve,reject)=>h.requests.push({steps,resolve,reject})),
  onBusy:value=>h.busy.push(value),onError:error=>{h.errors.push(error);h.running=false;}
 });
 h.fire=()=>{const [id,fn]=h.timers.entries().next().value;h.timers.delete(id);return fn();};
 h.start=()=>{h.running=true;h.pump.sync();};
 h.finish=(ms)=>{h.time+=ms;h.requests.at(-1).resolve();};
 return h;
}

test('requests continue after completion without an animation frame',async()=>{
 const h=harness();h.start();const first=h.fire();
 assert.equal(h.requests[0].steps,2);h.finish(2);await first;
 assert.equal(h.timers.size,1);const second=h.fire();
 assert.equal(h.requests[1].steps,10);
 h.running=false;h.finish(2);await second;
 assert.equal(h.timers.size,0);
});

test('adaptive batches stay bounded and shrink on a slow engine',async()=>{
 const h=harness();h.start();let run=h.fire();h.finish(2);await run;
 run=h.fire();assert.equal(h.requests.at(-1).steps,10);h.finish(1000);await run;
 run=h.fire();assert.equal(h.requests.at(-1).steps,1);
 h.running=false;h.finish(100);await run;
});

test('repeated play and pause cannot overlap calculations',async()=>{
 const h=harness();h.start();h.pump.sync();assert.equal(h.timers.size,1);
 const run=h.fire();h.running=false;h.pump.sync();h.start();h.pump.sync();
 assert.equal(h.requests.length,1);assert.equal(h.timers.size,0);
 h.finish(30);await run;assert.equal(h.timers.size,1);
 h.running=false;h.pump.sync();assert.equal(h.timers.size,0);
 assert.deepEqual(h.busy,[true,false]);
});

test('pause cancels queued work and an in-flight reply never restarts it',async()=>{
 const h=harness();h.start();h.running=false;h.pump.sync();
 assert.equal(h.timers.size,0);assert.equal(h.requests.length,0);
 h.start();const run=h.fire();h.running=false;h.pump.sync();h.finish(20);await run;
 assert.equal(h.requests.length,1);assert.equal(h.timers.size,0);
});

test('hidden or disconnected state is checked again before sending',async()=>{
 const h=harness();h.start();h.running=false;await h.fire();
 assert.equal(h.requests.length,0);assert.equal(h.timers.size,0);
});

test('a failed response releases busy state and stops scheduling',async()=>{
 const h=harness();h.start();const run=h.fire();
 h.requests[0].reject(Error('physical fault'));await run;
 assert.equal(h.errors[0].message,'physical fault');assert.equal(h.timers.size,0);
 assert.deepEqual(h.busy,[true,false]);
});
