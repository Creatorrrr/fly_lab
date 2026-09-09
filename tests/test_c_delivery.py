import copy
import json
from pathlib import Path
import tempfile
import unittest
from aiohttp import web, ClientSession, WSMsgType
from flylab.c import PROTOCOL
from flylab.c.engine import CEngine
from flylab.c.experiments import paired, replay_recording
from flylab.c.integrity import write_json
from flylab.c.protocol import decode_signals
from flylab.c.server import CServer, CDispatcher
from tests.c_fixtures import graph_fixture, bindings_fixture
from tests.fixture_body import FixtureBody
from tests.test_c import same_state


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.path = Path(self.temp.name)
        self.graph = graph_fixture(); self.bindings = bindings_fixture(self.graph)
        self.engine = CEngine(self.graph, self.bindings, mode='C_STRICT', body_factory=FixtureBody)

    def tearDown(self):
        self.engine.close(); self.temp.cleanup()

    def test_recorded_commands_replay_with_exact_neural_and_body_state(self):
        e = self.engine
        e.start_recording(self.path/'run')
        e.schedule(dict(kind='stimulate', ids=[self.graph.nodes[1]['id']], amplitude_mV=30., at_tick=100))
        e.step(5); e.command('push', {'bw':.2, 'duration':.05}); e.step(10)
        e.command('configure', {'motorCoupled':False}); e.step(7); e.stop_recording()
        restored = replay_recording(self.graph, self.bindings, self.path/'run', FixtureBody)
        try:
            same_state(e.neural.snapshot(), restored.neural.snapshot())
            same_state(e.encoder.snapshot(), restored.encoder.snapshot())
            same_state(e.body.snapshot(), restored.body.snapshot())
        finally:restored.close()

    def test_pair_preserves_original_and_proves_motor_effect(self):
        e = self.engine
        e.schedule(dict(kind='stimulate', ids=[self.graph.nodes[1]['id']], amplitude_mV=30.))
        before = e.checkpoint()
        result = paired(e, self.path/'pair', {'kind':'motor_disconnect'}, seconds=.2, onset=.1)
        same_state(before, e.checkpoint())
        self.assertTrue(result['pre_intervention_matched'])
        self.assertGreater(result['mean_path_separation_mm'], 0)
        self.assertFalse(result['physicalExecuted'])
        self.assertFalse(result['biologicalValidation'])

    def test_state_names_cannot_escape_artifact_directory(self):
        self.graph.save(self.path/'graph'); write_json(self.path/'bindings.json', self.bindings.spec)
        d = CDispatcher(self.path/'graph', self.path/'bindings.json', self.path/'artifacts', FixtureBody)
        try:
            d.handle(dict(protocol=PROTOCOL, requestId=1, op='init', payload={'mode':'C_STRICT'}))
            for op in ('checkpoint','restore','replay'):
                with self.assertRaises(ValueError):
                    d.handle(dict(protocol=PROTOCOL, requestId=2, op=op, payload={'name':'../../escape'}))
        finally:d.close()

    def test_failed_dispatch_restore_preserves_live_instance(self):
        self.graph.save(self.path/'graph'); write_json(self.path/'bindings.json', self.bindings.spec)
        d = CDispatcher(self.path/'graph', self.path/'bindings.json', self.path/'artifacts', FixtureBody)
        try:
            d.handle(dict(protocol=PROTOCOL, requestId=1, op='init', payload={'mode':'C_STRICT'}))
            before = d.engine; checkpoint = before.checkpoint()
            with self.assertRaises(FileNotFoundError):
                d.handle(dict(protocol=PROTOCOL, requestId=2, op='restore', payload={'name':'missing'}))
            self.assertIs(d.engine,before); same_state(checkpoint,d.engine.checkpoint())
        finally:d.close()

    def test_backend_transfer_preserves_current_clock_and_recording_blocks_it(self):
        self.graph.save(self.path/'graph');write_json(self.path/'bindings.json',self.bindings.spec)
        d=CDispatcher(self.path/'graph',self.path/'bindings.json',self.path/'artifacts',FixtureBody)
        def call(op,payload=None):return d.handle(dict(protocol=PROTOCOL,requestId=1,op=op,payload=payload or {}))
        try:
            call('init',{'mode':'C_STRICT'});call('advance',{'steps':10})
            call('record_start',{'name':'fixed-backend'})
            before=d.engine
            with self.assertRaises(ValueError):call('backend',{'backend':'exp_lif_cpu_reference'})
            self.assertIs(d.engine,before)
            call('record_stop')
            result=call('backend',{'backend':'exp_lif_cpu_reference'})['result']
            self.assertEqual(result['frame']['tick'],500)
            self.assertTrue((self.path/'artifacts/checkpoints'/result['source_checkpoint']/'manifest.json').is_file())
            self.assertEqual(result['frame']['performance']['measured_model_s'],0)
        finally:d.close()


class CServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory(); p=Path(self.temp.name)
        g=graph_fixture();g.save(p/'graph');write_json(p/'bindings.json',bindings_fixture(g).spec)
        self.server=CServer(0,p/'graph',p/'bindings.json',p/'artifacts',FixtureBody)
        self.runner=web.AppRunner(self.server.app());await self.runner.setup()
        site=web.TCPSite(self.runner,'127.0.0.1',0);await site.start()
        self.server.port=site._server.sockets[0].getsockname()[1]
        self.base=f'http://127.0.0.1:{self.server.port}';self.http=ClientSession()

    async def asyncTearDown(self):
        await self.http.close();await self.runner.cleanup();self.temp.cleanup()

    async def test_real_websocket_json_then_binary_and_no_idle_advancement(self):
        async with self.http.get(self.base+'/api/bootstrap') as response:
            bootstrap=await response.json();self.assertEqual(bootstrap['protocol'],PROTOCOL)
        async with self.http.ws_connect(self.base+'/ws',headers={'Origin':self.base}) as ws:
            await ws.send_json(dict(protocol=PROTOCOL,token=bootstrap['token']))
            self.assertTrue((await ws.receive_json())['authenticated'])
            for rid,op,payload in [(1,'init',{'mode':'C_STRICT'}),(2,'advance',{'steps':10}),(3,'frame',{})]:
                await ws.send_json(dict(protocol=PROTOCOL,requestId=rid,op=op,payload=payload))
                reply=await ws.receive_json();self.assertTrue(reply['ok'])
                message=await ws.receive();self.assertEqual(message.type,WSMsgType.BINARY)
                h,voltage,rate,spikes=decode_signals(message.data)
                self.assertEqual(h['end_tick'],0 if rid==1 else 500)
                self.assertLess(voltage.max(),0)
            await ws.send_json(dict(protocol=PROTOCOL,requestId=4,op='advance',payload={'steps':11}))
            self.assertFalse((await ws.receive_json())['ok'])

    async def test_foreign_origin_stays_rejected_in_v3(self):
        async with self.http.get(self.base+'/api/bootstrap',headers={'Origin':'https://example.com'}) as response:
            self.assertEqual(response.status,403)

    async def test_attach_and_bootstrap_keep_existing_experiment(self):
        bootstrap=await (await self.http.get(self.base+'/api/bootstrap')).json()
        self.assertFalse(bootstrap['hasExperiment'])
        async with self.http.ws_connect(self.base+'/ws',headers={'Origin':self.base}) as ws:
            await ws.send_json(dict(protocol=PROTOCOL,token=bootstrap['token']))
            await ws.receive_json()
            for rid,op,payload in [(1,'init',{'mode':'C_STRICT'}),(2,'advance',{'steps':2}),(3,'attach',{})]:
                await ws.send_json(dict(protocol=PROTOCOL,requestId=rid,op=op,payload=payload))
                reply=await ws.receive_json();self.assertTrue(reply['ok']);await ws.receive()
                if op=='attach':self.assertEqual(reply['result']['frame']['tick'],100)
            current=await (await self.http.get(self.base+'/api/bootstrap')).json()
            self.assertTrue(current['hasExperiment'])


if __name__=='__main__':unittest.main()
