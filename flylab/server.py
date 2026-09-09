"""Loopback-only WebSocket server. One controlling client, one physical thread.
No autonomous stepping, no arbitrary file execution, no pickle loading.
"""
from __future__ import annotations
import asyncio, json, secrets, hmac, logging, io, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit
from aiohttp import web, WSMsgType
from .engine import Dispatcher
from .body import dependency_report,FlyGymBody
from . import PROTOCOL

ROOT=Path(__file__).resolve().parents[1]
LOG=logging.getLogger('flylab')

class Server:
    def __init__(self,port=8765,body_factory=FlyGymBody):
        self.port=port;self.token=secrets.token_urlsafe(32);self.dispatcher=Dispatcher(body_factory)
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='flylab-physics')
        self.active=None;self.busy=False;self.test_double=body_factory is not FlyGymBody
        self.protocol=PROTOCOL
    def valid_host(self,request):
        return request.host in {f'127.0.0.1:{self.port}',f'localhost:{self.port}'}
    def valid_origin(self,request,require=False):
        origin=request.headers.get('Origin')
        if origin is None:return not require
        return origin in {f'http://127.0.0.1:{self.port}',f'http://localhost:{self.port}'}
    async def index(self,request):
        return web.FileResponse(ROOT/'FLY_LAB_B.html',headers={'Cache-Control':'no-store'})
    async def bootstrap(self,request):
        if not self.valid_origin(request):raise web.HTTPForbidden(text='Origin rejected')
        return web.json_response(dict(protocol=PROTOCOL,token=self.token,dependencies=dependency_report(),testDouble=self.test_double),headers={'Cache-Control':'no-store'})
    async def doctor(self,request):return web.json_response(dependency_report())
    async def ws(self,request):
        if not self.valid_origin(request,True):raise web.HTTPForbidden(text='Same-origin localhost WebSocket required')
        if self.active is not None:raise web.HTTPConflict(text='이미 연결된 관찰 창이 있습니다. 그 창을 닫고 연결하세요.')
        socket=web.WebSocketResponse(max_msg_size=10*1024*1024,heartbeat=30)
        self.active=socket
        try:
            await socket.prepare(request)
            # Token stays out of URL/logs. The first message is the handshake.
            raw=await asyncio.wait_for(socket.receive(),15)
            if raw.type!=WSMsgType.TEXT:await socket.close(code=1008);return socket
            try:hello=json.loads(raw.data)
            except ValueError:await socket.close(code=1008);return socket
            if not isinstance(hello,dict) or hello.get('protocol')!=self.protocol or not hmac.compare_digest(str(hello.get('token','')),self.token):
                await socket.close(code=1008,message=b'Invalid session token');return socket
            await socket.send_json(dict(protocol=self.protocol,authenticated=True))
            loop=asyncio.get_running_loop()
            async for msg in socket:
                if msg.type!=WSMsgType.TEXT:break
                rid=None
                try:
                    req=json.loads(msg.data,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Non-finite JSON')))
                    rid=req.get('requestId') if isinstance(req,dict) else None
                    result=await loop.run_in_executor(self.executor,self.dispatcher.handle,req)
                    await self.send_reply(socket,result)
                except Exception as e:
                    LOG.warning('request=%r: %s',rid,str(e))
                    await socket.send_json(dict(protocol=self.protocol,requestId=rid,ok=False,error=str(e)))
            return socket
        finally:
            self.active=None
            # Controller state remains in RAM until next init/restore or process exit.
            # No simulation steps occur without explicit advance requests.
    async def send_reply(self,socket,result):
        await socket.send_str(json.dumps(result,ensure_ascii=False,allow_nan=False))
    async def cleanup(self,app):
        if self.active is not None:await self.active.close(code=1001,message=b'Server stopping')
        await asyncio.get_running_loop().run_in_executor(self.executor,self.dispatcher.close)
        self.executor.shutdown(wait=True,cancel_futures=True)
    def app(self):
        @web.middleware
        async def guard(request,handler):
            if not self.valid_host(request) or not self.valid_origin(request):raise web.HTTPForbidden(text='Loopback host/origin only')
            response=await handler(request)
            response.headers['X-Content-Type-Options']='nosniff';response.headers['Cross-Origin-Resource-Policy']='same-origin'
            response.headers['Referrer-Policy']='no-referrer';response.headers['X-Frame-Options']='DENY'
            return response
        app=web.Application(middlewares=[guard],client_max_size=10*1024*1024)
        app.router.add_get('/',self.index);app.router.add_get('/FLY_LAB_B.html',self.index)
        app.router.add_get('/api/bootstrap',self.bootstrap);app.router.add_get('/api/doctor',self.doctor);app.router.add_get('/ws',self.ws)
        app.on_cleanup.append(self.cleanup)
        return app

def main():
    import argparse
    parser=argparse.ArgumentParser(description='FLY LAB B: local NeuroMechFly observation server')
    parser.add_argument('--port',type=int,default=8765);parser.add_argument('--doctor',action='store_true');args=parser.parse_args()
    if args.doctor:
        report=dependency_report();print(json.dumps(report,ensure_ascii=False,indent=2));return 0 if report['ready'] else 2
    if not 1024<=args.port<=65535:parser.error('Port must be 1024..65535')
    if not (ROOT/'FLY_LAB_B.html').exists():raise SystemExit('Run python build.py first')
    print('FLY LAB B — http://127.0.0.1:%s\n정지: Ctrl+C. 브라우저를 닫으면 계산 요청이 중단됩니다.'%args.port,flush=True)
    web.run_app(Server(args.port).app(),host='127.0.0.1',port=args.port,access_log=None)
    return 0
