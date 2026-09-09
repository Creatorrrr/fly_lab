import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from flylab.c import PROTOCOL
from flylab.c.engine import CEngine
from flylab.c.inputs import InputRejected
from flylab.c.server import CDispatcher
from flylab.c.integrity import write_json
from flylab.c.ports import PortBindings
from tests.c_fixtures import graph_fixture, bindings_fixture
from tests.fixture_body import FixtureBody
from tests.test_c import same_state


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.g = graph_fixture(); self.b = bindings_fixture(self.g)
        self.g.save(self.path/'graph'); write_json(self.path/'bindings.json', self.b.spec)
        self.d = CDispatcher(self.path/'graph', self.path/'bindings.json', self.path/'artifacts', FixtureBody)
        self.call('init', mode='C_STRICT')

    def tearDown(self):
        self.d.close(); self.temp.cleanup()

    def call(self, op, **payload):
        return self.d.handle(dict(protocol=PROTOCOL, requestId=1, op=op, payload=payload))['result']

    def stimulus(self, **changes):
        return dict(kind='stimulate', ids=[self.g.nodes[1]['id']], amplitude_mV=100., **changes)

    def test_overlap_rejected_without_mutation_and_half_open_endpoints(self):
        e = self.d.engine
        for _ in range(10): e.schedule(self.stimulus(duration_controls=1))
        before = e.checkpoint()
        with self.assertRaises(InputRejected): e.schedule(self.stimulus(duration_controls=1))
        same_state(before, e.checkpoint())
        e.schedule(self.stimulus(at_tick=50, duration_controls=1))
        e.step(2); self.assertFalse(e.fault); self.assertEqual(e.tick, 100)

    def test_03_checkpoint_migrates_without_changing_baseline_dynamics(self):
        e=self.d.engine;e.step(3);saved=e.checkpoint();saved['app_version']='0.3.0'
        saved.pop('metabolism');saved['sensors'].pop('transduction');saved['sensors'].pop('diagnostics');saved['sensors'].pop('previous_silhouette')
        restored=CEngine.from_checkpoint(self.g,self.b,saved,FixtureBody)
        try:
            e.step(4);restored.step(4)
            for field in ('neural','body','encoder','sensors'):same_state(e.checkpoint()[field],restored.checkpoint()[field])
            self.assertEqual(restored.checkpoint()['app_version'],'0.4.0')
        finally:restored.close()

    def test_negative_overlap_and_cancel(self):
        e = self.d.engine
        events = [e.schedule(dict(self.stimulus(), amplitude_mV=-100.)) for _ in range(10)]
        with self.assertRaises(InputRejected): e.schedule(dict(self.stimulus(), amplitude_mV=-1.))
        e.command('cancel_intervention', dict(serial=events[0]['serial']))
        e.schedule(dict(self.stimulus(), amplitude_mV=-100.)); e.step(1)

    def test_restore_rejects_combined_schedule_and_cancel_cannot_unmask_overload(self):
        e=self.d.engine
        negative=e.schedule(dict(self.stimulus(),amplitude_mV=-100.))
        for _ in range(11):e.schedule(self.stimulus())
        before=e.checkpoint()
        with self.assertRaises(InputRejected):e.command('cancel_intervention',dict(serial=negative['serial']))
        same_state(before,e.checkpoint())
        bad=copy.deepcopy(before);bad['pending']=bad['pending'][1:]
        with self.assertRaises(InputRejected):CEngine.from_checkpoint(self.g,self.b,bad,FixtureBody)

    def test_combined_sensory_drive_rejected_before_all_clocks_and_journals(self):
        spec = copy.deepcopy(self.b.spec)
        spec['sensory'][0].update(ids=[self.g.nodes[1]['id']], input_kind='direct_injection', baseline=10., gain=0.)
        e = CEngine(self.g, PortBindings(self.g, spec), mode='C_STRICT', body_factory=FixtureBody)
        try:
            for _ in range(10): e.schedule(self.stimulus())
            before = e.checkpoint()
            with self.assertRaises(InputRejected): e.step(1)
            same_state(before, e.checkpoint()); self.assertEqual(e.body.time, 0.); self.assertIsNone(e.fault)
        finally: e.close()

    def test_faulted_init_uses_diagnostic_and_allows_next_step(self):
        old = self.d.engine; old.fault = 'injected neural fault'; old.neural.v[0] = np.nan
        reply = self.call('init', mode='C_STRICT')
        self.assertTrue(old.body.closed); self.assertIsNot(old, self.d.engine)
        self.assertEqual(reply['recovery']['kind'], 'fault_reset')
        self.assertTrue(reply['recovery']['diagnostic_saved']); self.assertIsNone(reply['source_checkpoint'])
        path = self.path/'artifacts/diagnostics'/(reply['recovery']['diagnostic_id']+'.json')
        report = json.loads(path.read_text()); self.assertFalse(report['restorable'])
        self.assertEqual(report['neural']['v']['nonfinite'], 1)
        self.assertEqual(report['control_tick'],0);self.assertEqual(report['model_tick'],0)
        self.call('advance', steps=1); self.assertEqual(self.d.engine.tick, 50)

    def test_fault_dump_disk_failure_does_not_prevent_reset(self):
        self.d.engine.body.fault = 'body fault'
        with patch('flylab.c.server.write_json', side_effect=OSError('disk full')):
            reply = self.call('init', mode='C_STRICT')
        self.assertFalse(reply['recovery']['diagnostic_saved'])
        self.assertIn('disk full', reply['recovery']['diagnostic_save_error'])
        self.assertIsNotNone(self.d.last_fault_report)

    def test_failed_new_constructor_keeps_old_live(self):
        old = self.d.engine; old.fault = 'injected'
        with patch('flylab.c.server.CEngine', side_effect=RuntimeError('creation failed')):
            with self.assertRaisesRegex(RuntimeError, 'creation failed'): self.call('init', mode='C_STRICT')
        self.assertIs(old, self.d.engine); self.assertFalse(old.body.closed)

    def test_normal_checkpoint_failure_preserves_old(self):
        old = self.d.engine
        with patch('flylab.c.server.StateStore.save', side_effect=OSError('disk full')):
            with self.assertRaises(OSError): self.call('init', mode='C_STRICT')
        self.assertIs(old, self.d.engine); self.assertFalse(old.body.closed)

    def test_nonselected_incompatible_profile_is_visible_and_isolated(self):
        write_json(self.path/'bindings-incompatible.json', dict(self.b.spec, graph_hash='different'))
        reply = self.call('init', mode='C_STRICT', profile='bindings.json')
        unavailable = [p for p in reply['profiles'] if p['name']=='bindings-incompatible.json'][0]
        self.assertFalse(unavailable['available']); old = self.d.engine
        with self.assertRaisesRegex(ValueError, 'Profile unavailable'):
            self.call('init', mode='C_STRICT', profile='bindings-incompatible.json')
        self.assertIs(old, self.d.engine)

    def test_body_fault_marks_recording_failed_and_can_restart(self):
        e = self.d.engine; e.start_recording(self.path/'run')
        original_step = e.body.step
        def fault_after_step(*args):
            original_step(*args); e.body.fault = 'injected body failure'
        e.body.step = fault_after_step
        e.step(1)
        self.assertIsNone(e.recorder)
        self.assertEqual(json.loads((self.path/'run/manifest.json').read_text())['status'], 'FAILED')
        self.call('init', mode='C_STRICT'); self.call('advance', steps=1)


if __name__ == '__main__': unittest.main()
