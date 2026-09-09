#!/usr/bin/env python3
"""Save a live C checkpoint via the authenticated localhost protocol, without stepping."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import aiohttp
from flylab.c.integrity import checked_name,write_json

async def run(port,name,out):
    out=Path(out)
    if out.exists():raise ValueError('Session readback output must be new')
    checked_name(name);url=f'http://127.0.0.1:{port}'
    async with aiohttp.ClientSession() as session:
        async with session.get(url+'/api/bootstrap') as response:response.raise_for_status();config=await response.json()
        async with session.ws_connect(url+'/ws',origin=url) as socket:
            protocol=config['protocol'];await socket.send_json(dict(protocol=protocol,token=config['token']))
            if not (await socket.receive_json()).get('authenticated'):raise RuntimeError('Session authentication failed')
            rid=0
            async def rpc(op,payload):
                nonlocal rid
                rid+=1;await socket.send_json(dict(protocol=protocol,requestId=rid,op=op,payload=payload))
                message=await socket.receive_json(timeout=120)
                if not message.get('ok'):raise RuntimeError(message.get('error'))
                if op=='frame':
                    binary=await socket.receive(timeout=120)
                    if binary.type!=aiohttp.WSMsgType.BINARY or binary.data[:4]!=b'FLC3':raise RuntimeError('Binary signal response missing')
                return message['result']
            # A legacy server's profile discovery may reject newer sibling
            # profiles. Frame/checkpoint do not enumerate those files.
            frame=await rpc('frame',{})
            if frame['recording']['active']:raise ValueError('Active recording requires an explicit segmented recording migration')
            saved=await rpc('checkpoint',dict(name=name))
            capabilities=dict(backend=config.get('backend') if frame.get('neural') else 'legacy_b_rate',physical=not frame['physics']['testDouble'])
            result=dict(status='PASS',port=port,checkpoint=saved,frame=frame,capabilities=capabilities,
                        stepped=False,token_recorded=False)
            write_json(out,result)
            print(dict(status='PASS',checkpoint=name,model_seconds=frame['simTime'],mode=frame['mode'],backend=capabilities['backend']))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8766);p.add_argument('--name',required=True);p.add_argument('--out',required=True);a=p.parse_args();asyncio.run(run(a.port,a.name,a.out))
