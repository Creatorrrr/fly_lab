const assert=require('node:assert/strict');
const {test}=require('node:test');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const app=fs.readFileSync(path.join(__dirname,'../src/c/app.js'),'utf8');
function harness(pages){
 const h={messages:[],exports:[],requests:[]};
 vm.runInNewContext(app.slice(app.indexOf("button('banc-trace-export'"),app.indexOf('async function bancCheckpoints(')),{
  button:(_,fn)=>{h.run=fn;},pauseBancAndWait:async()=>true,
  rpc:{request:async(op,payload)=>{h.requests.push([op,payload]);return pages.shift();}},
  tell:message=>h.messages.push(message),
  Blob:class{constructor(parts){h.exports.push(JSON.parse(parts.join('')));}},
  URL:{createObjectURL:()=>'',revokeObjectURL:()=>{}},
  document:{createElement:()=>({click(){}})},setTimeout:fn=>fn(),
 });
 return h;
}
const page=(samples,extra={})=>({epoch:'a',samples,first_available_tick:0,last_available_tick:1,
 evicted_samples:0,next_after:samples.at(-1)?.control_tick,has_more:false,...extra});

test('export retains the matching full initial physics state while assembling pages',async()=>{
 const initial={control_tick:0,time_s:0,body:{state:[0,1],push_left:.003,push_force:[1,2,3]}};
 const h=harness([
  page([{control_tick:0}],{has_more:true,initial_physics:initial,command_replay:{eligible:true,verified:false}}),
  page([{control_tick:1}],{command_replay:{eligible:true,verified:false}}),
 ]);
 await h.run();assert.equal(h.exports.length,1);assert.equal(h.exports[0].samples.length,2);
 assert.deepEqual(h.exports[0].initial_physics,initial);
 assert.match(h.messages[0],/수치 일치는 별도 검증/);
});

test('an evicted anchor is exported as diagnostic data with an explicit replay limitation',async()=>{
 const h=harness([page([{control_tick:100}],{first_available_tick:100,last_available_tick:100,
  evicted_samples:100,initial_physics:null,
  command_replay:{eligible:false,verified:false,reasons:['INITIAL_SAMPLE_EVICTED_OR_UNAVAILABLE']}})]);
 await h.run();assert.equal(h.exports[0].initial_physics,null);
 assert.equal(h.exports[0].command_replay.eligible,false);
 assert.match(h.messages[0],/불완전/);assert.match(h.messages[0],/INITIAL_SAMPLE_EVICTED_OR_UNAVAILABLE/);
});

test('export refuses pages from a changed epoch or moving retained range',async()=>{
 for(const changed of [{epoch:'b'},{first_available_tick:1}]){
  const h=harness([page([{control_tick:0}],{has_more:true}),page([{control_tick:1}],changed)]);
  await assert.rejects(h.run(),/변경됐습니다/);assert.equal(h.exports.length,0);
 }
});
