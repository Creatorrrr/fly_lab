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
assert.equal(status({...f,neuromuscular:{enabled:true}},true).state,'joint_pending');
assert.match(status({...f,physics:{attachment:'tethered'}},true).warnings[1],/몸통이 고정/);
assert.equal(status({...f,mode:'C_SHADOW'},true).state,'other_mode');
assert.equal(status({...f,world:{...f.world,foodOn:false}},true).warnings.length,0);
assert.equal(status({...f,world:{...f.world,sources:[{...f.world.sources[0],odorant:'named'}]}},true).warnings.length,0);
const emptyPlume={...f,sensor_diagnostics:{four_site_odor:{concentration:[[0,0],[0,0],[0,0],[0,0]]}}};
assert.match(status(emptyPlume,true).warnings[1],/냄새 농도가 0/);
assert.equal(status({...emptyPlume,world:{...f.world,sources:[]}},true).warnings.length,0);
assert.equal(status({...emptyPlume,sensor_diagnostics:{four_site_odor:{concentration:[[0.2,0],[0,0],[0,0],[0,0]]}}},true).warnings.length,1);
const quiet={...f,neural:{mean_rate_Hz:50},neuromuscular:{enabled:true,
 muscles:[{leg:'lf',muscle_rates_Hz:{long_tendon_muscle:0,tarsus_depressor_muscle:0}}],offset_rad:[0,0]},
 physics:{jointTargets:[1,2],actuatorForces:[15,30]}};
// Network activity, neutral position servos and their support force do not
// establish neural motor activity. This is the paused-page regression.
assert.equal(status(quiet,true).state,'silent_joint_motor');
assert.match(status(quiet,true).message,/운동 집단 발화와 신경 구동 입력은 0/);
assert.equal(Fly.neuromuscularActivity(quiet).motorRateMaxHz,0);
assert.equal(status({...quiet,motor_rates_Hz:{forward:20}},true).state,'silent_joint_motor');
const firing={...quiet,neuromuscular:{...quiet.neuromuscular,
 muscles:[{muscle_rates_Hz:{agonist:10,antagonist:10}}]}};
assert.equal(status(firing,true).state,'zero_joint_command');
assert.equal(status({...quiet,neuromuscular:{...quiet.neuromuscular,offset_rad:[-.05,0]}},true).state,'joint_output');
const tendons={names:['lf_tarsus'],inputs:[.068],neural_controls:{modes:{lf_tarsus:'neural'}}};
const tendonFrame={...quiet,physics:{tendons},neuromuscular:{...quiet.neuromuscular,
 tendons:{modes:{lf_tarsus:'neural'},last:[{sources:[{rates_Hz:[80,100]}]}]}}};
assert.equal(status(tendonFrame,true).state,'joint_output');
assert.equal(Fly.neuromuscularActivity(tendonFrame).motorRateMaxHz,90);
assert.equal(Fly.neuromuscularActivity(tendonFrame).neuralTendonMaxInput,.068);
const manual={...quiet,physics:{tendons:{...tendons,neural_controls:{modes:{lf_tarsus:'manual'}}}}};
assert.equal(status(manual,true).state,'silent_joint_motor');
assert.match(status(manual,true).warnings.at(-1),/수동 힘줄/);
assert.equal(Fly.neuromuscularActivity(manual).neuralTendonMaxInput,0);
assert.equal(Fly.neuromuscularActivity(manual).manualTendonMaxInput,.068);
assert.equal(status({...tendonFrame,command:{motor_coupled:false}},true).state,'disconnected');
// A filtered input remains active after a stimulus ends, but its denormal tail
// must not leave the page claiming perpetual neural drive.
assert.equal(status({...quiet,physics:{tendons}},true).state,'joint_output');
assert.equal(status({...quiet,physics:{tendons:{...tendons,inputs:[1.5e-323]}}},true).state,'silent_joint_motor');
assert.equal(status({...tendonFrame,physics:{}},true).state,'joint_pending');
assert.equal(status({...quiet,neuromuscular:{...quiet.neuromuscular,offset_rad:[NaN]}},true).state,'joint_pending');
const before=JSON.stringify(tendonFrame);status(tendonFrame,false);Fly.neuromuscularActivity(tendonFrame);
assert.equal(JSON.stringify(tendonFrame),before);
console.log('C_STRICT display: baseline and neuromuscular counterexamples passed');
