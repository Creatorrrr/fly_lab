import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
from flylab.c import PROTOCOL
from flylab.c.engine import CEngine
from flylab.c.experiments import replay_recording, paired_campaign
from flylab.c.behavior import behavior_metrics, trace_sample
from flylab.c.integrity import write_json
from flylab.c.protocol import signal_frame, decode_signals
from flylab.c.server import CDispatcher
from tests.c_fixtures import graph_fixture, bindings_fixture
from tests.fixture_body import FixtureBody
from tests.test_c import same_state
from tests.test_c_mps import mps_available


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)
        self.g=graph_fixture();self.b=bindings_fixture(self.g)
        self.e=CEngine(self.g,self.b,mode='C_STRICT',body_factory=FixtureBody)
        self.ids=[n['id'] for n in self.g.nodes]

    def tearDown(self):
        self.e.close();self.temp.cleanup()

    def test_environment_crud_strength_sensor_and_undo_preserve_time_and_neural_state(self):
        e=self.e;e.step(3);before=e.checkpoint();body=e.body
        e.command('place',dict(kind='food',position=[1,.7,1],strength=3.))
        self.assertIs(body,e.body);self.assertEqual(e.tick,150)
        same_state(before['neural'],e.neural.snapshot());same_state(before['body'],e.body.snapshot())
        obj=e.world['sources'][-1];self.assertEqual(obj['strength'],3.)
        e.command('update_object',dict(id=obj['id'],position=[2,.7,1],strength=4.))
        self.assertEqual(e.world['sources'][-1]['strength'],4.)
        e.command('delete_object',{'id':obj['id']});self.assertNotIn(obj['id'],[o['id'] for o in e.world['sources']])
        e.command('undo_environment',{});self.assertEqual(e.world['sources'][-1]['strength'],4.)
        e.command('undo_environment',{});self.assertEqual(e.world['sources'][-1]['strength'],3.)
        e.step(1);self.assertGreater(e.last_sensors['odor'][0],before['last_sensors']['odor'][0])
        self.assertFalse(e.frame()['environment']['sensor_refresh_pending'])

    def test_wall_body_overlap_and_invalid_values_roll_back_entire_checkpoint(self):
        e=self.e
        for payload in [dict(kind='obstacle',position=[23,1,0],radius=2),
                        dict(kind='obstacle',position=[0,1,0]),
                        dict(kind='obstacle',position=[7,1,-3]),
                        dict(kind='hazard',position=[7,.7,4],strength=-1),
                        dict(kind='food',position=[7,float('nan'),4]),
                        dict(kind='food',position=[7,.7,4],radius=1)]:
            before=e.checkpoint()
            with self.assertRaises(ValueError):e.command('place',payload)
            same_state(before,e.checkpoint())

    def test_failed_model_restore_preserves_live_body_world_and_history(self):
        e=self.e;e.step(2);original=e.body;before=e.checkpoint();candidates=[]
        class Broken(FixtureBody):
            def restore(self,state):candidates.append(self);raise RuntimeError('injected restore failure')
        e.body_factory=Broken
        with self.assertRaisesRegex(RuntimeError,'injected'):e.command('place',dict(kind='obstacle',position=[15,1,10]))
        self.assertIs(e.body,original);self.assertFalse(original.closed);self.assertTrue(candidates[0].closed)
        same_state(before,e.checkpoint())

    def test_obstacle_transfer_and_undo_preserve_physical_integration_state(self):
        e=self.e;e.step(4);before=e.body.snapshot();old=e.body
        e.command('place',dict(kind='obstacle',position=[15,2,10],radius=1.3))
        self.assertTrue(old.closed);self.assertEqual(e.world['obstacles'][-1]['p'][1],1.3)
        same_state(before,e.body.snapshot());e.command('undo_environment',{});same_state(before,e.body.snapshot())

    def test_undo_at_current_tick_rejects_body_intrusion_without_popping_history(self):
        e=self.e;e.command('delete_object',{'id':'rock-1'})
        e.body.p=np.array([7.,3.,.9]);before=e.checkpoint()
        with self.assertRaises(ValueError):e.command('undo_environment',{})
        same_state(before,e.checkpoint())

    def test_capacity_and_imported_object_id_collision(self):
        e=self.e;world=copy.deepcopy(e.world);world['sources'][0]['id']='user-1'
        e.command('load_environment',{'world':world});e.command('place',dict(kind='food',position=[0,.7,0]))
        self.assertEqual(e.world['sources'][-1]['id'],'user-2')
        for _ in range(20):e.command('place',dict(kind='food',position=[0,.7,0]))
        before=e.checkpoint()
        with self.assertRaises(ValueError):e.command('place',dict(kind='food',position=[0,.7,0]))
        same_state(before,e.checkpoint())

    def test_cancel_does_not_erase_arrivals_or_other_active_masks(self):
        e=self.e
        a=e.schedule(dict(kind='suppress_spiking',ids=[self.ids[1]],duration_controls=10))
        b=e.schedule(dict(kind='suppress_spiking',ids=[self.ids[2]],duration_controls=10))
        e.step(1);e.neural.queue[0,3]=7.;queue=e.neural.queue.copy();tick=e.tick
        e.command('cancel_intervention',{'serial':a['serial']})
        self.assertFalse(e.neural.suppress[1]);self.assertTrue(e.neural.suppress[2]);self.assertEqual(tick,e.tick)
        np.testing.assert_array_equal(queue,e.neural.queue)
        status={r['serial']:r['status'] for r in e.frame()['interventions']['items']}
        self.assertEqual(status,{a['serial']:'cancelled',b['serial']:'active'})

    def test_pending_cancel_and_expiry_survive_checkpoint(self):
        e=self.e
        a=e.schedule(dict(kind='stimulate',ids=[self.ids[1]],at_tick=500))
        b=e.schedule(dict(kind='mute_outgoing',ids=[self.ids[2]],duration_controls=1))
        e.command('cancel_intervention',{'serial':a['serial']});e.step(2)
        restored=CEngine.from_checkpoint(self.g,self.b,e.checkpoint(),FixtureBody)
        try:
            self.assertEqual(restored.frame()['interventions']['items'],e.frame()['interventions']['items'])
            with self.assertRaises(ValueError):restored.command('cancel_intervention',{'serial':b['serial']})
        finally:restored.close()

    def test_new_metadata_validation_and_old_checkpoint_compatibility(self):
        e=self.e;e.command('place',dict(kind='food',position=[3,.7,3]));s=e.checkpoint()
        bad=copy.deepcopy(s);bad['environment_updated_tick']=1
        with self.assertRaises(ValueError):CEngine.from_checkpoint(self.g,self.b,bad,FixtureBody)
        for k in ('environment_updated_tick','environment_history','intervention_history','record_cohort_ids'):s.pop(k)
        old=CEngine.from_checkpoint(self.g,self.b,s,FixtureBody)
        try:self.assertEqual(old.tick,e.tick);self.assertEqual(old.environment_history,[])
        finally:old.close()

    def test_fixed_cohort_edit_cancel_undo_and_loaded_world_replay_exactly(self):
        e=self.e;cohort=[self.ids[0],self.ids[1]];e.start_recording(self.path/'run',cohort)
        a=e.schedule(dict(kind='stimulate',ids=[self.ids[1]],amplitude_mV=30))
        e.command('place',dict(kind='hazard',position=[3,.7,4],strength=2));e.step(7)
        e.subscribe([self.ids[4]]);e.command('update_object',dict(id='user-1',strength=4));e.step(5)
        e.command('cancel_intervention',{'serial':a['serial']});e.command('undo_environment',{});e.step(8)
        e.command('load_environment',{'world':copy.deepcopy(e.world)});e.stop_recording()
        m=json.loads((self.path/'run/manifest.json').read_text());self.assertEqual(m['cohort_ids'],cohort)
        self.assertEqual(sum(c['rows'] for c in m['chunks']),10)
        replay=replay_recording(self.g,self.b,self.path/'run',FixtureBody)
        try:
            for k in ('neural','body','encoder','sensors','world','environment_history','intervention_history','record_cohort_ids'):
                same_state(e.checkpoint()[k],replay.checkpoint()[k])
        finally:replay.close()

    def test_id_lookup_saved_environment_cohort_and_auto_checkpoint(self):
        self.g.save(self.path/'graph');write_json(self.path/'bindings.json',self.b.spec)
        d=CDispatcher(self.path/'graph',self.path/'bindings.json',self.path/'artifacts',FixtureBody)
        def call(op,**payload):return d.handle(dict(protocol=PROTOCOL,requestId=1,op=op,payload=payload))['result']
        try:
            call('init',mode='C_STRICT');call('advance',steps=3)
            self.assertEqual(call('lookup',ids=[self.ids[5]])[0]['id'],self.ids[5])
            call('cohort_save',name='mine',ids=[self.ids[3]]);self.assertEqual(call('presets')[-1]['ids'],[self.ids[3]])
            call('environment_save',name='scene');before=copy.deepcopy(d.engine.world)
            call('command',type='delete_object',payload={'id':'food-1'})
            call('environment_load',name='scene');self.assertEqual(d.engine.world,before)
            r=call('init',mode='C_SHADOW');self.assertTrue(r['source_checkpoint'])
            call('restore',name=r['source_checkpoint']);self.assertEqual(d.engine.tick,150)
            call('record_start',name='active',ids=[self.ids[1]])
            for op in ('init','restore','replay'):
                with self.assertRaises(ValueError):call(op,name='scene')
            for op in ('cohort_save','environment_save','environment_load'):
                with self.assertRaises(ValueError):call(op,name='../escape',ids=[self.ids[1]])
        finally:d.close()

    def test_comparison_seeds_fixed_cohort_and_original_preservation(self):
        before=self.e.checkpoint()
        r=paired_campaign(self.e,self.path/'campaign',dict(kind='stimulate',ids=[self.ids[1]],amplitude_mV=30),
                          seconds=.03,onset=.01,ids=[self.ids[1]],origin='fresh',seed=7,repeats=2)
        same_state(before,self.e.checkpoint());self.assertEqual(r['status'],'COMPLETE')
        self.assertEqual([c['seed'] for c in r['comparisons']],[7,8])
        self.assertTrue(all(c['pre_intervention_matched'] for c in r['comparisons']))
        self.assertEqual(r['task_status'],'NOT_EVALUATED')

    def test_distinct_binding_profiles_restore_only_the_matching_hash(self):
        self.g.save(self.path/'graph');write_json(self.path/'bindings.json',self.b.spec)
        other=copy.deepcopy(self.b.spec);other['profile']='alternate';other['sensory'][0]['channel']='odor_left'
        write_json(self.path/'bindings-alternate.json',other)
        d=CDispatcher(self.path/'graph',self.path/'bindings.json',self.path/'artifacts',FixtureBody)
        def call(op,**payload):return d.handle(dict(protocol=PROTOCOL,requestId=1,op=op,payload=payload))['result']
        try:
            call('init',mode='C_STRICT');before=call('checkpoint',name='base')
            changed=call('init',mode='C_STRICT',profile='bindings-alternate.json')
            self.assertNotEqual(changed['binding']['hash'],self.b.hash)
            self.assertEqual(call('restore',name=before['name'])['binding']['hash'],self.b.hash)
            bad=d.engine.checkpoint();bad['binding_hash']='unknown'
            with self.assertRaises(ValueError):d.checkpoint_bindings(bad)
        finally:d.close()

    def test_profile_builder_preserves_motor_mapping_and_requires_annotated_afferents(self):
        from tools.build_c_odor_profile import build
        # Independent tiny annotation roster; the actual archived IDs are checked by the native diagnostic.
        g=graph_fixture();g.nodes[0].update(cell_type='ORN_DM1',soma_side='left',nerve='AN')
        g.nodes[4].update(cell_type='ORN_DM1',soma_side='right',nerve='AN',super_class='sensory',flow='afferent')
        g.nodes[5].update(cell_type='ORN_DA2',soma_side='unknown',nerve='AN',super_class='sensory',flow='afferent')
        base=copy.deepcopy(self.b.spec)
        base['motor']['yaw_left']['ids']=[self.ids[1]];base['motor']['yaw_right']['ids']=[self.ids[2]]
        base['unused_observations']=['danger','contact'];base['profile']='test'
        built=build(g,base);same_state(base['motor'],built['motor'])
        b=__import__('flylab.c.ports',fromlist=['PortBindings']).PortBindings(g,built)
        from flylab.c.ports import SensoryEncoder
        p=copy.deepcopy(self.e.last_sensors);p['odor']=[.8,.1];p['danger']=.5
        for s in built['sensory']:s['gain']=18
        b=__import__('flylab.c.ports',fromlist=['PortBindings']).PortBindings(g,built)
        drive,_,_=SensoryEncoder(b,42).encode(p,.005,.0001)
        self.assertGreater(drive[0],drive[4]);self.assertGreater(drive[5],0)
        off,_,_=SensoryEncoder(b,42).encode(p,.005,.0001,['*']);self.assertFalse(np.any(off))
        g.nodes[5]['flow']='efferent'
        with self.assertRaises(ValueError):build(g,base)

    def test_signed_metrics_do_not_call_reverse_or_rotation_forward(self):
        f=self.e.frame();a=trace_sample(f);b=copy.deepcopy(a);b.update(simTime=.005,tick=50,position=[-1,.9,0])
        m=behavior_metrics([a,b],self.e.world)
        self.assertEqual(m['forward_mm'],0);self.assertEqual(m['backward_mm'],1)
        self.assertEqual(m['stuck_status'],'INCOMPLETE_WINDOW');self.assertEqual(m['observation_gaps'],0)

    def test_javascript_decoder_matches_binary_spike_ticks_and_rejects_corruption(self):
        e=self.e;e.subscribe([self.ids[1]]);e.schedule(dict(kind='stimulate',ids=[self.ids[1]],amplitude_mV=100));e.step(10)
        binary=signal_frame(e);h,v,r,spikes=decode_signals(binary);self.assertGreater(len(spikes),0)
        (self.path/'frame.bin').write_bytes(binary)
        script="""const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
vm.runInThisContext(fs.readFileSync('src/c/signals.js','utf8'));
const b=fs.readFileSync(process.argv[1]),ab=b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),decoded=Fly.decodeCSignals(ab);
console.log(JSON.stringify(decoded));const dv=new DataView(ab);dv.setUint32(8+dv.getUint32(4,true)+8+8,999,true);assert.throws(()=>Fly.decodeCSignals(ab));"""
        output=subprocess.check_output(['node','-e',script,str(self.path/'frame.bin')],text=True)
        decoded=json.loads(output);self.assertEqual([x['tick'] for x in decoded['events']],spikes['tick'].tolist())
        self.assertEqual(decoded['values'][0],dict(voltage=float(v[0]),rate=float(r[0])))


@unittest.skipUnless(mps_available(), 'Actual MPS required')
class AdditionalMotorReadoutTests(unittest.TestCase):
    def test_full_subscription_plus_motor_readout_preserves_events_and_uses_one_transfer(self):
        from flylab.c.neural import create_backend
        from types import SimpleNamespace
        from unittest.mock import Mock
        g=graph_fixture(n=1032);cpu=create_backend(g);mps=create_backend(g,backend='exp_lif_mps')
        capture=np.arange(1024,dtype=np.int32);motor=np.arange(1024,1032,dtype=np.int32)
        mps.set_readout_cohort(motor)
        original=mps.observation_library;read=Mock(wraps=original.read_state)
        mps.observation_library=SimpleNamespace(read_state=read,summarize=original.summarize)
        drive=np.full(g.n,30.,np.float32);cpu.advance(drive,100,capture);mps.advance(drive,100,capture)
        before=read.call_count;values=mps.readout(motor)
        self.assertEqual(before,read.call_count)
        np.testing.assert_array_equal(values['rate_Hz'],cpu.readout(motor)['rate_Hz'])
        np.testing.assert_allclose(values['voltage_mV'],cpu.readout(motor)['voltage_mV'],rtol=0,atol=1e-4)
        self.assertEqual(cpu.last_events,mps.last_events)
        self.assertTrue(all(e['index']<1024 for e in mps.last_events))


if __name__=='__main__':unittest.main()
