"""Single independent campaign worker; JSON status survives UI disconnection."""
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from .campaign import validate_spec, pilot_spec
from .integrity import read_json, write_json, checked_name
from .locking import acquire, locked

class CampaignJobs:
    def __init__(self, root, graph_path):
        self.root=Path(root);self.graph_path=Path(graph_path).resolve();self.processes={}

    def active(self):
        return locked(self.root/'.worker.lock')

    def start(self, bindings, backend, spec=None):
        spec=validate_spec(pilot_spec(bindings=bindings) if spec is None else spec,bindings,backend)
        if self.active():raise ValueError('A campaign worker is already active; cancel or finish it first')
        name='campaign-'+secrets.token_hex(6);directory=self.root/name;directory.mkdir(parents=True)
        write_json(directory/'spec.json',spec);write_json(directory/'bindings.json',bindings.spec)
        write_json(directory/'job.json',dict(id=name,created=time.time(),backend=backend,graph=str(self.graph_path)))
        self._launch(name,False)
        return self.status(name)

    def _launch(self,name,resume):
        directory=self.root/checked_name(name);job=read_json(directory/'job.json')
        fd=acquire(self.root/'.worker.lock')
        args=[sys.executable,str(Path(__file__).resolve().parents[2]/'tools/run_c_campaign.py'),
              '--graph',job['graph'],'--bindings',str((directory/'bindings.json').resolve()),
              '--backend',job['backend'],'--spec',str((directory/'spec.json').resolve()),
              '--out',str((directory/'result').resolve()),'--cancel-file',str((directory/'cancel').resolve()),
              '--worker-lock-fd',str(fd)]
        if resume:args.append('--resume')
        try:
            with (directory/'worker.log').open('ab') as log:
                p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,cwd=Path(__file__).resolve().parents[2],pass_fds=(fd,))
            write_json(self.root/'.worker-owner.json',dict(id=name,pid=p.pid,started=time.time()))
        finally:
            # Do not LOCK_UN: the child retains the inherited file description.
            os.close(fd)
        self.processes[name]=p
        write_json(directory/'worker.json',dict(pid=p.pid,started=time.time()))

    def status(self,name):
        directory=self.root/checked_name(name)
        if not (directory/'job.json').exists():raise ValueError('Unknown campaign job')
        p=self.processes.get(name);report=read_json(directory/'result/campaign.json') if (directory/'result/campaign.json').exists() else {}
        owner=read_json(self.root/'.worker-owner.json') if (self.root/'.worker-owner.json').exists() else {}
        running=self.active() and owner.get('id')==name
        status=report.get('status','STARTING')
        if not running and status in ('RUNNING','STARTING'):status='INTERRUPTED' if p is None else 'FAILED'
        return dict(id=name,status=status,running=running,cancel_requested=(directory/'cancel').exists(),
                    current_case=report.get('current_case'),current_model_s=report.get('current_model_s',0.),
                    completed_model_s=report.get('completed_model_s',0.),total_model_s=report.get('total_model_s'),
                    completed_cases=len(report.get('cases',[])),error=report.get('error'),
                    task_results=[dict(name=c['name'],task=c['task_status'],technical=c['technical_status']) for c in report.get('cases',[])],
                    output=str(directory.resolve()),biological_validation=False)

    def cancel(self,name):
        self.status(name);(self.root/name/'cancel').touch(exist_ok=True)
        return self.status(name)

    def resume(self,name):
        if self.active():raise ValueError('A campaign worker is already active')
        state=self.status(name)
        if state['status'] not in ('CANCELLED','INTERRUPTED','FAILED'):raise ValueError('Job is not resumable')
        cancel=self.root/name/'cancel'
        if cancel.exists():cancel.unlink()
        self._launch(name,True);return self.status(name)

    def list(self):
        return [self.status(p.name) for p in sorted(self.root.glob('campaign-*')) if (p/'job.json').is_file()]

    def close(self):
        for name,p in self.processes.items():
            if p.poll() is None:self.cancel(name)
