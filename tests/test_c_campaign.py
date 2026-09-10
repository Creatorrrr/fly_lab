import copy
import json
import tempfile
import unittest
from pathlib import Path
from flylab.c.campaign import run_campaign, pilot_spec, validate_spec
from flylab.c.tasks import evaluate
from flylab.c.behavior import trace_sample
from flylab.c.engine import CEngine
from flylab.c.sensors import CSensorAdapter
from flylab.c.pathways import reachability, propagate
from flylab.c.ports import PortBindings
from flylab.sensors import default_world
from flylab.engine import config_values
from tests.fixture_body import FixtureBody
from tests.c_fixtures import graph_fixture, bindings_fixture
from tests.test_c import same_state

class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.g=graph_fixture();self.b=bindings_fixture(self.g)
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)
        self.spec=pilot_spec(.03,seeds=(7,),modes=('C_STRICT',));self.spec['cases']=self.spec['cases'][:1]
    def tearDown(self):self.temp.cleanup()
    def run_case(self,out,**kwargs):
        return run_campaign(self.g,self.b,self.spec,self.path/out,backend='exp_lif_cpu_reference',body_factory=FixtureBody,checkpoint_controls=2,**kwargs)
    def test_cancel_resume_preserves_every_tick_and_final_state(self):
        calls=[0]
        def cancel():calls[0]+=1;return calls[0]>4
        partial=self.run_case('resumed',cancelled=cancel)
        self.assertEqual(partial['status'],'CANCELLED')
        done=self.run_case('resumed',resume=True);self.run_case('reference')
        self.assertEqual(done['status'],'COMPLETE');self.assertFalse(done['physicalExecuted'])
        states=[]
        from flylab.c.storage import StateStore
        for name in ('resumed','reference'):
            d=self.path/name/self.spec['cases'][0]['name'];p=json.loads((d/'progress.json').read_text())
            states.append(StateStore.load(d/p['checkpoint']))
            rows=[json.loads(line) for c in p['chunks'] for line in (d/c['file']).read_text().splitlines()]
            self.assertEqual([r['tick'] for r in rows],list(range(0,301,50)))
        for key in ('neural','encoder','sensors','body'):same_state(states[0][key],states[1][key])
    def test_changed_identity_rejects_resume(self):
        self.run_case('run');other=copy.deepcopy(self.b.spec);other['profile']='changed'
        with self.assertRaisesRegex(ValueError,'Cannot resume'):
            run_campaign(self.g,PortBindings(self.g,other),self.spec,self.path/'run',resume=True,backend='exp_lif_cpu_reference',body_factory=FixtureBody)
    def test_compact_progress_preserves_and_verifies_detailed_result(self):
        report=self.run_case('compact');row=report['cases'][0]
        self.assertNotIn('metrics',row)
        path=self.path/'compact'/row['result_file'];detail=json.loads(path.read_text())
        self.assertIn('metrics',detail);self.assertEqual(detail['task_status'],row['task_status'])
        self.assertEqual(self.run_case('compact',resume=True)['status'],'COMPLETE')
        detail['task_status']='changed';path.write_text(json.dumps(detail))
        with self.assertRaisesRegex(ValueError,'Case result hash mismatch'):
            self.run_case('compact',resume=True)
    def test_multiple_interventions_and_committed_evidence_loader(self):
        from flylab.c.sensorimotor_campaign import load_case
        case=self.spec['cases'][0]
        ids=[self.g.nodes[1]['id']]
        case['interventions']=[dict(kind='stimulate',ids=ids,amplitude_mV=20.,duration_controls=6),
                               dict(kind='suppress_spiking',ids=ids,at_tick=50,duration_controls=2)]
        self.run_case('multiphase')
        root=self.path/'multiphase';loaded=load_case(root,case['name'])
        self.assertEqual(loaded['interventions'],case['interventions'])
        self.assertEqual(len(loaded['trace']),7)
        progress=json.loads((root/case['name']/'progress.json').read_text())
        trace=root/case['name']/progress['chunks'][0]['file'];trace.write_text(trace.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'Trace chunk hash mismatch'):load_case(root,case['name'])
        case['intervention']=case['interventions'][0]
        with self.assertRaisesRegex(ValueError,'one intervention'):validate_spec(self.spec)
    def test_bad_spec_is_rejected_before_output_created(self):
        self.spec['cases'][0]['seconds']=.007
        with self.assertRaises(ValueError):self.run_case('bad')
        self.assertFalse((self.path/'bad').exists())
    def test_disabled_motion_and_gaps_never_pass_stationary_check(self):
        e=CEngine(self.g,self.b,body_factory=FixtureBody)
        try:a=trace_sample(e.frame())
        finally:e.close()
        rows=[dict(copy.deepcopy(a),simTime=i*.005,tick=i*50,motion_enabled=False) for i in range(601)]
        r=evaluate(rows,default_world(),'walking',required_seconds=3)
        self.assertEqual(r['metrics']['stuck_status'],'INCOMPLETE_WINDOW')
        self.assertNotEqual(r['task_status'],'PASS')
        rows=[dict(copy.deepcopy(a),simTime=i*.005,tick=i*50,position=[i*.01,.9,0.]) for i in range(2001)]
        self.assertEqual(evaluate(rows,default_world(),'walking')['task_status'],'PASS')
        rows.pop(1000)
        self.assertEqual(evaluate(rows,default_world(),'walking')['technical_status'],'INCOMPLETE')
    def test_compressive_sensor_preserves_bilateral_difference_and_checkpoint(self):
        world=default_world();world['sources']=[dict(id='food',kind='food',p=[2.,.7,-2.],strength=5.)]
        body=FixtureBody(42,world,config_values())
        old=CSensorAdapter(42);new=CSensorAdapter(42,dict(kind='compressive-odor-v2',half_concentration=1.))
        self.assertEqual(old.observe(body,world,.005,config_values())['odor'],[1.,1.])
        packet=new.observe(body,world,.005,config_values());self.assertNotEqual(*packet['odor'])
        state=new.snapshot();expected=new.observe(body,world,.005,config_values());new.restore(state)
        self.assertEqual(expected,new.observe(body,world,.005,config_values()))
        self.assertTrue(all(v<1. for v in packet['odor']))
    def test_pathways_separate_reachability_and_propagation(self):
        r=reachability(self.g,self.b)
        self.assertEqual(r['paths']['anatomical']['test_odor']['motor']['forward'][0]['hops'],1)
        result=propagate(self.g,self.b,dict(name='zero'),seconds=.01)
        self.assertEqual(result['active_nodes'],0);self.assertFalse(result['physicalExecuted'])

    def test_sensor_bad_history_cannot_partially_restore(self):
        sensor=CSensorAdapter(42);before=sensor.snapshot();bad=copy.deepcopy(before)
        bad['rng']=100;bad['previous_silhouette']=[0.,2.]
        with self.assertRaises(ValueError):sensor.restore(bad)
        same_state(before,sensor.snapshot())

    def test_food_entry_after_deadline_fails(self):
        e=CEngine(self.g,self.b,body_factory=FixtureBody)
        try:a=trace_sample(e.frame())
        finally:e.close()
        world=default_world();world['sources']=[dict(id='food',kind='food',p=[10.,.7,0.],strength=1.)]
        rows=[dict(copy.deepcopy(a),simTime=i*.005,tick=i*50,position=[10. if i>=6100 else 0.,.9,0.]) for i in range(6201)]
        self.assertEqual(evaluate(rows,world,'food',required_seconds=31.)['task_status'],'FAIL')

    def test_worker_lease_survives_manager_recreation_and_release(self):
        from flylab.c.jobs import CampaignJobs
        from flylab.c.locking import acquire
        from flylab.c.integrity import write_json
        import os
        jobs=CampaignJobs(self.path/'jobs',self.path);directory=jobs.root/'campaign-abc';directory.mkdir(parents=True)
        write_json(directory/'job.json',dict(id='campaign-abc'))
        fd=acquire(jobs.root/'.worker.lock')
        try:
            write_json(jobs.root/'.worker-owner.json',dict(id='campaign-abc'))
            recreated=CampaignJobs(jobs.root,self.path)
            self.assertTrue(recreated.active());self.assertTrue(recreated.status('campaign-abc')['running'])
            with self.assertRaises(ValueError):recreated.start(self.b,'exp_lif_cpu_reference',self.spec)
        finally:os.close(fd)
        self.assertFalse(jobs.active());self.assertEqual(jobs.status('campaign-abc')['status'],'INTERRUPTED')

if __name__=='__main__':unittest.main()
