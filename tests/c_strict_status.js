/* UI counterexamples using recorded frame fields, with no browser or physics mocks. */
const assert=require('node:assert/strict');
require('../src/c/experiment-status.js');
const f={mode:'C_STRICT',simTime:1,world:{foodOn:true,sources:[{id:'food-1',kind:'food',strength:1.2}]},
 binding:{chemical_odorants:[{name:'ethyl acetate'}]},physics:{},sensory_ports:[{enabled:true,value:2}],
 motor_rates_Hz:{forward:0,yaw_left:0},command:{motor_coupled:true,u_final:{forwardSpeed:0,yawRate:0,verticalSpeed:0}}};
const status=Fly.describeStrictExperiment;
assert.equal(status(f,true).state,'silent_motor');
assert.equal(status(f,true).warnings.length,1);
assert.match(status(f,false).message,/일시정지/);
assert.equal(status({...f,simTime:0},false).state,'ready');
assert.equal(status({...f,fault:'bad physics'},false).state,'fault');
assert.equal(status({...f,sensory_ports:[]},true).state,'zero_input');
assert.equal(status({...f,motor_rates_Hz:{forward:10}},true).state,'zero_command');
assert.equal(status({...f,command:{...f.command,u_final:{forwardSpeed:1}}},true).state,'motor_output');
assert.equal(status({...f,command:{...f.command,motor_coupled:false}},true).state,'disconnected');
assert.equal(status({...f,neuromuscular:{enabled:true}},true).state,'joint_output');
assert.match(status({...f,physics:{attachment:'tethered'}},true).warnings[1],/몸통 고정/);
assert.equal(status({...f,mode:'C_SHADOW'},true).state,'other_mode');
assert.equal(status({...f,world:{...f.world,foodOn:false}},true).warnings.length,0);
assert.equal(status({...f,world:{...f.world,sources:[{...f.world.sources[0],odorant:'named'}]}},true).warnings.length,0);
const emptyPlume={...f,sensor_diagnostics:{four_site_odor:{concentration:[[0,0],[0,0],[0,0],[0,0]]}}};
assert.match(status(emptyPlume,true).warnings[1],/냄새 농도가 0/);
assert.equal(status({...emptyPlume,world:{...f.world,sources:[]}},true).warnings.length,0);
assert.equal(status({...emptyPlume,sensor_diagnostics:{four_site_odor:{concentration:[[0.2,0],[0,0],[0,0],[0,0]]}}},true).warnings.length,1);
console.log('C_STRICT display: 17 counterexamples passed');
