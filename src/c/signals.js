/* FLC3 decoder shared by the viewer and raw-event contract tests. */
(function(F){
'use strict';
F.decodeCSignals=function(buffer){
 const v=new DataView(buffer);
 if(buffer.byteLength<8||String.fromCharCode(...new Uint8Array(buffer,0,4))!=='FLC3')throw Error('신호 프레임 형식 오류');
 const len=v.getUint32(4,true);
 if(len>1000000||8+len>buffer.byteLength)throw Error('신호 헤더 길이 오류');
 const h=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,8,len))),base=8+len,n=h.channel_count;
 if(h.schema!=='flylab.signals.v3'||h.dtype!=='<f4'||h.unit_descriptor?.join(',')!=='mV,Hz'||
    !Number.isInteger(n)||n<0||n>512||!Array.isArray(h.ids)||h.ids.length!==n||new Set(h.ids).size!==n||h.ids.some(i=>typeof i!=='string')||
    !Number.isSafeInteger(h.spike_count)||h.spike_count<0||h.payload_bytes!==buffer.byteLength-base||h.payload_bytes!==n*8+h.spike_count*12||
    h.voltage_offset!==0||h.rate_offset!==n*4||h.spike_offset!==n*8||h.spike_dtype!=='uint64_tick,uint32_channel'||
    !Number.isSafeInteger(h.sequence)||h.sequence<1||!Number.isSafeInteger(h.subscription_epoch)||h.subscription_epoch<1||
    !Number.isSafeInteger(h.start_tick)||h.start_tick<0||!Number.isSafeInteger(h.end_tick)||h.end_tick<h.start_tick||
    !Number.isFinite(h.neural_dt)||h.neural_dt<=0)throw Error('신호 프레임의 단위·크기·시각이 다릅니다.');
 const values=[],events=[];
 for(let i=0;i<n;i++){
  const voltage=v.getFloat32(base+i*4,true),rate=v.getFloat32(base+n*4+i*4,true);
  if(!Number.isFinite(voltage)||!Number.isFinite(rate))throw Error('유한하지 않은 신경 신호');
  values.push({voltage,rate});
 }
 for(let i=0;i<h.spike_count;i++){
  const offset=base+n*8+i*12,tick=Number(v.getBigUint64(offset,true)),channel=v.getUint32(offset+8,true);
  if(!Number.isSafeInteger(tick)||tick<h.start_tick||tick>h.end_tick||channel>=n)throw Error('발화 이벤트의 tick·대상이 잘못됐습니다.');
  events.push({tick,id:h.ids[channel]});
 }
 return {header:h,values,events};
};
})(globalThis.Fly=globalThis.Fly||{});
