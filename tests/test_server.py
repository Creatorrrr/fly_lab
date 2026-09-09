"""Loopback security and serialization tests against an explicit body double."""
import unittest,json
from aiohttp import web,ClientSession,WSServerHandshakeError
from flylab.server import Server
from tests.fixture_body import FixtureBody
from flylab import PROTOCOL

class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.server=Server(8767,FixtureBody);self.runner=web.AppRunner(self.server.app());await self.runner.setup()
        self.site=web.TCPSite(self.runner,'127.0.0.1',8767);await self.site.start();self.http=ClientSession();self.base='http://127.0.0.1:8767'
    async def asyncTearDown(self):await self.http.close();await self.runner.cleanup()
    async def test_01_foreign_host_rejected(self):
        async with self.http.get(self.base+'/api/bootstrap',headers={'Host':'attacker.example'}) as r:self.assertEqual(r.status,403)
    async def test_02_foreign_origin_rejected(self):
        async with self.http.get(self.base+'/api/bootstrap',headers={'Origin':'https://attacker.example'}) as r:self.assertEqual(r.status,403)
    async def test_03_bootstrap_not_cached(self):
        async with self.http.get(self.base+'/api/bootstrap') as r:
            self.assertEqual(r.headers['Cache-Control'],'no-store');self.assertEqual((await r.json())['protocol'],PROTOCOL)
    async def test_04_ws_requires_origin(self):
        with self.assertRaises(WSServerHandshakeError):await self.http.ws_connect(self.base+'/ws')
    async def test_05_invalid_token(self):
        async with self.http.ws_connect(self.base+'/ws',headers={'Origin':self.base}) as ws:
            await ws.send_json(dict(protocol=PROTOCOL,token='bad'));m=await ws.receive();self.assertEqual(m.data,1008)
    async def test_06_rpc_roundtrip_and_no_idle_steps(self):
        async with self.http.get(self.base+'/api/bootstrap') as r:token=(await r.json())['token']
        async with self.http.ws_connect(self.base+'/ws',headers={'Origin':self.base}) as ws:
            await ws.send_json(dict(protocol=PROTOCOL,token=token));self.assertTrue((await ws.receive_json())['authenticated'])
            for i,op,p in [(1,'init',{}),(2,'advance',{'steps':10}),(3,'frame',{}),(4,'advance',{'steps':999})]:
                await ws.send_json(dict(protocol=PROTOCOL,requestId=i,op=op,payload=p));r=await ws.receive_json()
                if i==1:self.assertFalse(r['result']['capabilities']['physics'])
                elif i in (2,3):self.assertEqual(r['result']['tick'],10)
                else:self.assertFalse(r['ok'])
    async def test_07_single_client(self):
        async with self.http.ws_connect(self.base+'/ws',headers={'Origin':self.base}) as one:
            with self.assertRaises(WSServerHandshakeError):await self.http.ws_connect(self.base+'/ws',headers={'Origin':self.base})
            await one.close()
    async def test_08_static_source_not_served(self):
        async with self.http.get(self.base+'/flylab/server.py') as r:self.assertEqual(r.status,404)

if __name__=='__main__':unittest.main(verbosity=2)
