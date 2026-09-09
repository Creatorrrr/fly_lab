// B controller reference, including engineered avoidance memory. Compare the Python port on identical input.
const fs=require('fs'),path=require('path'),vm=require('vm');const root=path.join(__dirname,'..');
for(const file of ['math','connectome','brain'])vm.runInThisContext(fs.readFileSync(path.join(root,'src/core/'+file+'.js'),'utf8'));
const x=JSON.parse(fs.readFileSync(0,'utf8'));const b=new Fly.NeuralController(new Fly.Circuit(x.graph),x.seed);
for(const step of x.steps){if(step.event)b.intervene(step.event.ids,step.event.kind,step.event.amplitude,step.event.duration);b.step(step.sensor,x.dt,x.config);}
console.log(JSON.stringify(b.snapshot()));
