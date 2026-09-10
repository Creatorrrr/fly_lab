(function(F){
'use strict';

// One bounded calculation at a time. Rendering never gates the next request.
F.createPlaybackPump=function({shouldRun,advance,onBusy=()=>{},onError,
 now=()=>performance.now(),schedule=fn=>setTimeout(fn,0),cancel=id=>clearTimeout(id),
 targetMs=40,maxSteps=10}){
 let timer=null,busy=false,msPerStep=null;
 function sync(){
  if(!shouldRun()){
   if(timer!==null){cancel(timer);timer=null;}
   return;
  }
  if(!busy&&timer===null)timer=schedule(run);
 }
 async function run(){
  timer=null;
  if(busy||!shouldRun())return;
  const steps=Math.max(1,Math.min(maxSteps,msPerStep===null?2:Math.floor(targetMs/msPerStep)));
  busy=true;onBusy(true);
  const start=now();
  try{
   await advance(steps);
   const measured=Math.max(.01,(now()-start)/steps);
   msPerStep=msPerStep===null?measured:.7*msPerStep+.3*measured;
  }catch(error){onError(error);}
  finally{busy=false;onBusy(false);sync();}
 }
 return {sync};
};
})(globalThis.Fly);
