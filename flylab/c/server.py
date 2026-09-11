"""Local v3 observation server with bounded catalogs and binary subscriptions."""
from pathlib import Path
from dataclasses import asdict
import json
import secrets
from aiohttp import web
from . import VERSION, PROTOCOL, MODES
from .engine import CEngine
from .graph import GraphStore
from .ports import PortBindings
from .neural import LIFParameters, NEURAL_BACKENDS, BACKEND_CHOICES, create_backend
from .backend_selection import backend_availability, resolve_backend
from .model_config import execution_capabilities, validate_execution, model_identity
from .protocol import signal_frame
from .storage import StateStore
from .integrity import read_json, write_json, digest, checked_name, bounded_int, file_hash
from .diagnostics import fault_report
from .jobs import CampaignJobs
from .workbench import checked_world, intervention_rows
from ..server import Server as BServer, ROOT
from ..body import FlyGymBody, dependency_report


class CDispatcher:
    def __init__(self, graph_path, binding_path, artifacts, body_factory=FlyGymBody,
                 backend='exp_lif_cpu_reference', default_mode='C_SHADOW'):
        self.graph_path, self.binding_path = Path(graph_path), Path(binding_path)
        self.artifacts = Path(artifacts).resolve()
        self.factory = body_factory
        self.backend, self.default_mode = resolve_backend(backend), default_mode
        self.graph = self.bindings = self.engine = None
        self.profile_cache = {}
        self.profile_entries = {}
        self.profile_fingerprints = {}
        self.last_fault_report = None
        self.jobs = CampaignJobs(self.artifacts/'campaigns', self.graph_path)
        self.batch=None
        self.shared=None

    def load(self):
        if self.graph is None:
            graph = GraphStore.load(self.graph_path)
            bindings = PortBindings(graph, read_json(self.binding_path))
            self.graph, self.bindings = graph, bindings
            self.profile_cache[self.binding_path.name] = bindings

    def profiles(self):
        self.load()
        paths = [self.binding_path]+[p for p in sorted(self.binding_path.parent.glob('bindings*.json')) if p!=self.binding_path]
        available = {p.name for p in paths[:16]}
        self.profile_entries = {}
        for path in paths[:16]:
            try:
                fingerprint = file_hash(path)
                if self.profile_fingerprints.get(path.name) != fingerprint:
                    binding = PortBindings(self.graph, read_json(path))
                    self.profile_cache[path.name] = binding
                    self.profile_fingerprints[path.name] = fingerprint
                binding = self.profile_cache[path.name]
                self.profile_entries[path.name] = dict(name=path.name, profile=binding.spec.get('profile', path.name),
                    hash=binding.hash, available=True, status='AVAILABLE', execution=execution_capabilities(binding))
            except Exception as exc:
                self.profile_cache.pop(path.name, None)
                self.profile_fingerprints.pop(path.name, None)
                self.profile_entries[path.name] = dict(name=path.name, profile=path.stem,
                    available=False, status='UNAVAILABLE', reason=str(exc))
        return {name: value for name, value in self.profile_cache.items() if name in available}

    def checkpoint_bindings(self, state):
        for binding in self.profiles().values():
            if binding.hash == state.get('binding_hash'): return binding
        raise ValueError('Checkpoint binding profile is unavailable; no implicit port conversion')

    def need(self):
        if self.engine is None: raise ValueError('Initialize an experiment first')
        return self.engine

    def replace(self, new):
        try:
            result = self.ready(new)
            json.dumps(result, allow_nan=False)
            result['_prepared_binary'] = signal_frame(new)
            from .protocol import decode_signals
            decode_signals(result['_prepared_binary'])
        except Exception:
            new.close()
            raise
        old, self.engine = self.engine, new
        self.bindings = new.bindings
        if new.neural: self.backend = new.neural.backend
        if old:
            try: old.close()
            except Exception as exc: result['cleanup_error'] = str(exc)
        return result

    def ready(self, engine=None):
        e = engine if engine is not None else self.need()
        self.profiles()
        support=execution_capabilities(e.bindings)
        return dict(capabilities=dict(version=VERSION, protocol=PROTOCOL, modes=support['modes'], execution=support,
                    backend=e.neural.backend if e.neural else 'legacy_b_rate', neuralBackends=support['neural_backends'],
                    physical=not e.body.test_double, fullBrain=e.neural is not None and e.graph.full_brain,
                    maxSubscription=512, maxAdvance=10, flight=False, biologicalValidation=False,
                    checkpoint=True, replay=True, selectedBinarySignals=True),
                    manifest=e.graph.manifest, binding=e.bindings.summary(), frame=e.frame(),
                    profiles=[dict(entry, current=entry.get('hash')==e.bindings.hash)
                              for entry in self.profile_entries.values()])

    def handle(self, message):
        if not isinstance(message, dict) or message.get('protocol') != PROTOCOL:
            raise ValueError('flylab.protocol.v3 required')
        rid = bounded_int(message.get('requestId'), 'requestId', 1)
        payload = message.get('payload', {})
        if not isinstance(payload, dict): raise ValueError('Payload object required')
        op = message.get('op')
        if op == 'init':
            if set(payload)-{'mode', 'seed', 'config', 'motion_expected', 'profile','metabolism','initial_pose','body_options'}: raise ValueError('Unknown initialization field')
            self.load()
            profile = payload.get('profile', self.binding_path.name)
            bindings = self.profiles().get(profile)
            if bindings is None:
                reason = self.profile_entries.get(profile, {}).get('reason', 'Unknown binding profile')
                raise ValueError('Profile unavailable: ' + reason)
            validate_execution(bindings, payload.get('mode', self.default_mode), self.backend)
            current = self.engine
            faulted = current is not None and bool(current.fault or current.body.fault)
            if current and current.recorder and not faulted: raise ValueError('Finish recording before starting a new experiment')
            saved = None
            recovery = dict(kind='new', restorable_source=False)
            if faulted:
                report = fault_report(current)
                self.last_fault_report = report
                name = 'fault-'+secrets.token_hex(6)
                recovery.update(kind='fault_reset', diagnostic_id=name, diagnostic=report)
                try:
                    (self.artifacts/'diagnostics').mkdir(parents=True, exist_ok=True)
                    write_json(self.artifacts/'diagnostics'/(name+'.json'), report)
                    recovery['diagnostic_saved'] = True
                except Exception as exc:
                    recovery.update(diagnostic_saved=False, diagnostic_save_error=str(exc))
            elif current:
                saved = 'before-init-'+secrets.token_hex(6)
                StateStore.save(self.artifacts/'checkpoints'/saved, current.checkpoint())
                recovery.update(kind='checkpoint_reset', restorable_source=True)
            new = CEngine(self.graph, bindings, mode=payload.get('mode', self.default_mode),
                          seed=payload.get('seed', 42), config=payload.get('config'), backend=self.backend,
                          body_factory=self.factory, motion_expected=payload.get('motion_expected', True),metabolism=payload.get('metabolism'),
                          initial_pose=payload.get('initial_pose'),body_options=payload.get('body_options'))
            result = self.replace(new)
            result['source_checkpoint'] = saved
            result['recovery'] = recovery
        elif op == 'attach': result = self.ready()
        elif op == 'frame': result = self.need().frame()
        elif op == 'shared_init':
            from ..shared_arena import SharedFlyArena
            if set(payload)-{'count','seed','spacing_mm','collisions'}:raise ValueError('Unknown shared arena field')
            candidate=SharedFlyArena(**payload)
            candidate.set_drives({n:[.6,.6] for n in candidate.names})
            old=self.shared;self.shared=candidate
            if old:old.close()
            result=self.shared_result()
        elif isinstance(op,str) and op.startswith('shared_'):
            if self.shared is None:raise ValueError('Create a shared arena first')
            if op=='shared_advance':
                if set(payload)-{'steps'}:raise ValueError('Unknown shared advance field')
                self.shared.advance(bounded_int(payload.get('steps',20),'steps',1,20))
                result=self.shared_result()
            elif op=='shared_drive':
                if set(payload)!={'drives'}:raise ValueError('Named drives required')
                self.shared.set_drives(payload['drives']);result=self.shared_result()
            elif op=='shared_save':
                name='shared-'+secrets.token_hex(6)
                directory=self.artifacts/'shared-checkpoints';directory.mkdir(parents=True,exist_ok=True)
                write_json(directory/(name+'.json'),self.shared.snapshot());result=dict(name=name)
            elif op=='shared_restore':
                from ..shared_arena import SharedFlyArena
                name=checked_name(payload.get('name'))
                saved=read_json(self.artifacts/'shared-checkpoints'/(name+'.json'))
                candidate=SharedFlyArena(**saved['spec'])
                try:candidate.restore(saved)
                except Exception:
                    candidate.close();raise
                self.shared.close();self.shared=candidate;result=self.shared_result()
            elif op=='shared_close':
                self.shared.close();self.shared=None;result=dict(closed=True)
            else:raise ValueError('Unknown shared arena operation')
        elif op == 'backend':
            target = payload.get('backend')
            if target not in NEURAL_BACKENDS: raise ValueError('Unknown neural backend')
            current = self.need()
            if current.recorder: raise ValueError('Finish the current recording before changing its backend')
            checkpoint = current.checkpoint()
            # Save the source before moving the state; a failed transfer keeps it live.
            name = 'before-backend-'+secrets.token_hex(6)
            StateStore.save(self.artifacts/'checkpoints'/name, checkpoint)
            new = CEngine.from_checkpoint(self.graph, self.bindings, checkpoint, self.factory,
                                          backend_override=target)
            result = self.replace(new)
            result['source_checkpoint'] = name
        elif op == 'advance':
            if self.jobs.active(): raise ValueError('Campaign worker is active; live playback is paused to isolate computation')
            result = self.need().step(bounded_int(payload.get('steps', 10), 'steps', 0, 10))
        elif op == 'campaign_start':
            e=self.need()
            if e.recorder:raise ValueError('Finish live recording before launching a campaign')
            result=self.jobs.start(e.bindings,self.backend,payload.get('spec'))
        elif op == 'campaign_status': result=self.jobs.status(payload.get('id'))
        elif op == 'campaign_cancel': result=self.jobs.cancel(payload.get('id'))
        elif op == 'campaign_resume': result=self.jobs.resume(payload.get('id'))
        elif op == 'campaign_list': result=self.jobs.list()
        elif op == 'subscribe':
            self.need().subscribe(payload.get('ids')); result = self.need().frame()
        elif op == 'catalog':
            self.load(); result = self.graph.catalog(payload.get('query', ''), payload.get('offset', 0), payload.get('limit', 100))
        elif op == 'neighbors':
            self.load(); result = self.graph.neighbors(payload.get('id'), payload.get('limit', 100))
        elif op == 'lookup':
            self.load(); result = [self.graph.nodes[int(i)] for i in self.graph.resolve(payload.get('ids'))]
        elif op == 'presets':
            e = self.need()
            result = [dict(name=name, ids=port['ids'], binding_hash=e.bindings.hash)
                      for name, port in [('Forward output population', e.bindings.spec['motor']['forward']),
                                         ('MDN', e.bindings.spec['motor']['backward']),
                                         ('DNa02 left', e.bindings.spec['motor']['yaw_left']),
                                         ('DNa02 right', e.bindings.spec['motor']['yaw_right'])] if port['ids']]
            result.append(dict(name='Motor outputs', ids=[e.graph.nodes[int(i)]['id'] for i in e.bindings.motor_indices], binding_hash=e.bindings.hash))
            result.extend(dict(name=p['name'], ids=p['ids'], binding_hash=e.bindings.hash)
                          for p,_ in e.bindings.sensory if len(p['ids'])<=512)
            result.extend(dict(name=p['name'],ids=p['ids'],binding_hash=e.bindings.hash,
                               review_status=p['review_status'],uncertainty=p['uncertainty'])
                          for p,_ in e.bindings.research_cohorts)
            if e.neuromuscular:
                result.append(dict(name='BANC connected motor neurons',
                    ids=[e.graph.nodes[int(i)]['id'] for i in e.neuromuscular.motor_indices], binding_hash=e.bindings.hash))
                if e.neuromuscular.tendon_adapter:
                    for row in e.neuromuscular.tendon_adapter.rows:
                        for group in row['groups']:
                            name=f"Tendon {row['name']} {group.get('side',group.get('muscle',''))}"
                            result.append(dict(name=name, ids=group['ids'], binding_hash=e.bindings.hash,
                                               review_status='engineering_reviewed', uncertainty=row['hypothesis']))
                for row in e.neuromuscular.spec['rows']:
                    result.extend(dict(name=f"{row['leg']} {p['kind']}"+(' '+p['cell_type'] if p.get('cell_type') else ''), ids=p['ids'], binding_hash=e.bindings.hash)
                                  for p in row['sensory'] if len(p['ids'])<=512)
            directory = self.artifacts/'cohorts'
            if directory.exists():
                for p in sorted(directory.glob('*.json')):
                    saved = read_json(p)
                    if saved.get('graph_hash') == e.graph.hash:
                        e.graph.resolve(saved['ids']); result.append(dict(name=p.stem, ids=saved['ids'], custom=True))
        elif op == 'cohort_save':
            e = self.need(); name = checked_name(payload.get('name')); ids = payload.get('ids')
            if not len(e.graph.resolve(ids)): raise ValueError('A nonempty cohort is required')
            directory = self.artifacts/'cohorts'; directory.mkdir(parents=True, exist_ok=True)
            path = directory/(name+'.json')
            if path.exists(): raise ValueError('Cohort name already exists')
            write_json(path, dict(schema='flylab.cohort.v1', graph_hash=e.graph.hash, ids=ids))
            result = dict(name=name, ids=ids)
        elif op == 'interventions':
            result = intervention_rows(self.need(), payload.get('offset', 0), payload.get('limit', 50))
        elif op == 'environment_save':
            e = self.need(); name = checked_name(payload.get('name', 'environment-'+secrets.token_hex(6)))
            checked_world(e.world)
            directory = self.artifacts/'environments'; directory.mkdir(parents=True, exist_ok=True)
            path = directory/(name+'.json')
            if path.exists(): raise ValueError('Environment name already exists')
            write_json(path, dict(schema='flylab.environment.v1', world=e.world, environment_hash=digest(e.world)))
            result = dict(name=name, environment_hash=digest(e.world))
        elif op == 'environments':
            directory = self.artifacts/'environments'
            result = [dict(name=p.stem) for p in sorted(directory.glob('*.json'))] if directory.exists() else []
        elif op == 'environment_load':
            name = checked_name(payload.get('name')); saved = read_json(self.artifacts/'environments'/(name+'.json'))
            if saved.get('schema') != 'flylab.environment.v1' or saved.get('environment_hash') != digest(saved.get('world')):
                raise ValueError('Environment identity mismatch')
            result = self.need().command('load_environment', {'world': saved['world']})
        elif op == 'regions': result = self.need().region_summary()
        elif op == 'batch_init':
            if set(payload)-{'worlds','seed','profile','mode','body_options'}:raise ValueError('Unknown batch initialization option')
            if self.batch is not None:raise ValueError('Save and close the current batch before preparing another')
            self.load()
            count=bounded_int(payload.get('worlds',2),'batch worlds',1,32)
            seed=bounded_int(payload.get('seed',42),'batch seed',0,2**32-count)
            bindings=self.profiles().get(payload.get('profile',self.binding_path.name))
            if bindings is None:raise ValueError('Unknown batch binding profile')
            from .batch import BatchSession
            candidate=BatchSession(self.graph,bindings,[seed+i for i in range(count)],mode=payload.get('mode','C_SHADOW'),body_options=payload.get('body_options'))
            previous=self.batch;self.batch=candidate
            if previous:previous.close()
            result=self.batch.summary()
        elif op == 'batch_checkpoints':
            if payload:raise ValueError('Batch checkpoint catalog takes no options')
            directory=self.artifacts/'batch-checkpoints'
            result=[dict(name=p.name) for p in sorted(directory.glob('batch-*')) if (p/'manifest.json').is_file()]
        elif op == 'batch_restore':
            if set(payload)!={'name'}:raise ValueError('Select a batch checkpoint')
            if self.batch is not None:raise ValueError('Save and close the current batch before restoring another')
            self.load();path=self.artifacts/'batch-checkpoints'/checked_name(payload['name'])
            saved=StateStore.load(path,max_files=4096);binding=self.checkpoint_bindings(saved)
            from .batch import BatchSession
            self.batch=BatchSession.restore(self.graph,binding,path);result=self.batch.summary()
        elif op in ('batch_advance','batch_control','batch_summary','batch_checkpoint','batch_observation','batch_close','batch_record','batch_intervention'):
            if self.batch is None:raise ValueError('Initialize a CUDA batch first')
            if op=='batch_advance':
                if set(payload)-{'steps'}:raise ValueError('Unknown batch advance option')
                result=self.batch.step(bounded_int(payload.get('steps',1),'batch steps',1,10))
            elif op=='batch_control':
                if set(payload)!={'action','world'} or payload['action'] not in ('pause','resume','cancel'):raise ValueError('Invalid batch control')
                getattr(self.batch,payload['action'])(str(payload['world']));result=self.batch.summary()
            elif op in ('batch_record','batch_intervention'):
                if str(payload.get('world')) not in self.batch.engines:raise ValueError('Unknown batch world')
                key=str(payload['world']);e=self.batch.engines[key]
                if self.batch.status[key] not in ('RUNNING','PAUSED'):raise ValueError('World is no longer active')
                if op=='batch_record':
                    if set(payload)-{'world','action','ids'} or payload.get('action') not in ('start','stop'):raise ValueError('Invalid batch recording request')
                    if payload['action']=='start':
                        name='batch-world-'+key+'-'+secrets.token_hex(6)
                        e.start_recording(self.artifacts/'runs'/name,payload.get('ids'))
                    else:e.stop_recording()
                else:
                    if set(payload)!={'world','type','payload'} or payload['type'] not in ('intervene','release_all'):raise ValueError('Invalid batch intervention')
                    e.command(payload['type'],payload['payload'])
                result=self.batch.summary()
            elif op=='batch_checkpoint':
                if payload:raise ValueError('Batch checkpoint takes no options')
                result=self.batch.checkpoint(self.artifacts/'batch-checkpoints'/('batch-'+secrets.token_hex(6)))
            elif op=='batch_observation':
                if set(payload)!={'world'}:raise ValueError('Select a batch world')
                key=str(payload['world'])
                if key not in self.batch.engines:raise ValueError('Unknown batch world')
                import base64,io
                from PIL import Image
                if key in self.batch.active:
                    pixels=self.batch.physics.render(self.batch.active.index(key));renderer='GPU batch'
                else:pixels=self.batch.engines[key].body.preview();renderer='CPU paused-world view'
                buffer=io.BytesIO();Image.fromarray(pixels).save(buffer,format='PNG')
                result=dict(world=key,time_s=self.batch.engines[key].control_tick*.005,renderer=renderer,
                    image='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii'))
            elif op=='batch_close':self.batch.close();self.batch=None;result=dict(closed=True)
            else:result=self.batch.summary()
        elif op == 'physical_observation':
            if set(payload)-{'kind'}: raise ValueError('Unknown observation option')
            e=self.need(); kind=payload.get('kind')
            if e.body.test_double: raise ValueError('Native physical observations require the real body')
            from ..flygym_senses import FourSiteOdor
            if kind=='odor':
                sensor=e.sensors.four_site_odor or FourSiteOdor()
                result=sensor.observe(e.body,e.world)
            elif kind in ('eyes','native'):
                import base64, io
                import numpy as np
                from PIL import Image
                if kind=='eyes':
                    observation=e.body.compound_eye_observation()
                    pixels=np.concatenate(list(observation['raw_rgb']),axis=1)
                    result=dict(metadata=observation['metadata'],time_s=observation['time_s'],
                        ommatidia=observation['ommatidia'].tolist())
                else:
                    pixels=e.body.preview()
                    result=dict(time_s=e.body.physics_time(),metadata=dict(observation_only=True,
                        body_model_hash=e.body.model_hash,kind='native MuJoCo mesh'))
                buffer=io.BytesIO();Image.fromarray(pixels).save(buffer,format='PNG')
                result['image']='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')
            else:raise ValueError('Observation must be eyes, native or odor')
            result['control_tick']=e.control_tick
        elif op == 'command': result = self.need().command(payload.get('type'), payload.get('payload', {}))
        elif op == 'checkpoint':
            name = checked_name(payload.get('name', 'checkpoint-'+secrets.token_hex(6)))
            result = StateStore.save(self.artifacts/'checkpoints'/name, self.need().checkpoint())
        elif op == 'restore':
            if self.engine and self.engine.recorder: raise ValueError('Finish recording before restoring an experiment')
            self.load(); name = checked_name(payload.get('name'))
            state = StateStore.load(self.artifacts/'checkpoints'/name)
            result = self.replace(CEngine.from_checkpoint(self.graph, self.checkpoint_bindings(state), state, self.factory))
        elif op == 'checkpoints':
            directory = self.artifacts/'checkpoints'
            result = [{'name': p.name} for p in sorted(directory.iterdir()) if p.is_dir() and (p/'manifest.json').is_file()] if directory.exists() else []
        elif op == 'diagnostics':
            directory = self.artifacts/'diagnostics'
            result = dict(reports=[dict(name=p.stem, restorable=False) for p in sorted(directory.glob('*.json'))],
                          latest=self.last_fault_report)
        elif op == 'record_start':
            name = checked_name(payload.get('name', 'run-'+secrets.token_hex(6)))
            self.need().start_recording(self.artifacts/'runs'/name, payload.get('ids'))
            result = dict(name=name, frame=self.need().frame())
        elif op == 'record_stop': self.need().stop_recording(); result = self.need().frame()
        elif op == 'paired':
            from .experiments import paired, paired_campaign
            name = checked_name(payload.get('name', 'pair-'+secrets.token_hex(6)))
            options = dict(seconds=payload.get('seconds', .5), onset=payload.get('onset', .2), ids=payload.get('ids'))
            extended = bool(set(payload)&{'repeats','origin','seed'})
            if extended: options.update(repeats=payload.get('repeats',1),origin=payload.get('origin','current'),seed=payload.get('seed'))
            result = (paired_campaign if extended else paired)(self.need(),self.artifacts/'experiments'/name,
                        payload.get('intervention',{'kind':'motor_disconnect'}),**options)
        elif op == 'replay':
            if self.engine and self.engine.recorder: raise ValueError('Finish recording before replay')
            from .experiments import replay_recording
            name = checked_name(payload.get('name'))
            self.load()
            path = self.artifacts/'runs'/name
            binding = self.checkpoint_bindings(StateStore.load(path/'initial_checkpoint'))
            replayed = replay_recording(self.graph, binding, path, self.factory)
            result = self.replace(replayed)
        else: raise ValueError('Unsupported C operation: ' + str(op))
        response = dict(protocol=PROTOCOL, requestId=rid, ok=True, result=result)
        if self.engine is not None and op in ('init', 'attach', 'backend', 'frame', 'advance', 'subscribe', 'restore', 'replay'):
            response['_binary'] = result.pop('_prepared_binary') if isinstance(result,dict) and '_prepared_binary' in result else signal_frame(self.engine)
            self.engine.selected_events = []
        return response

    def shared_result(self):
        import base64
        import io
        from PIL import Image
        result=dict(observation=self.shared.observation())
        try:
            stream=io.BytesIO();Image.fromarray(self.shared.preview()).save(stream,format='PNG')
            result['image']=base64.b64encode(stream.getvalue()).decode('ascii')
        except Exception as exc:result['render_error']=str(exc)
        return result

    def close(self):
        self.jobs.close()
        if self.batch:self.batch.close();self.batch=None
        if self.shared:self.shared.close();self.shared=None
        if self.engine: self.engine.close(); self.engine = None


class CServer(BServer):
    def __init__(self, port=8766, graph_path=ROOT/'data/fafb783/bundle',
                 binding_path=ROOT/'data/fafb783/bindings.json', artifacts=ROOT/'artifacts/c',
                 body_factory=FlyGymBody, backend='exp_lif_cpu_reference', default_mode='C_SHADOW'):
        super().__init__(port, body_factory)
        self.protocol = PROTOCOL
        self.dispatcher = CDispatcher(graph_path, binding_path, artifacts, body_factory, backend, default_mode)

    def app(self):
        app = super().app()
        app.router.add_get('/FLY_LAB_C.html', self.index)
        return app

    async def index(self, request):
        return web.FileResponse(ROOT/'FLY_LAB_C.html', headers={'Cache-Control':'no-store'})

    async def bootstrap(self, request):
        d = self.dispatcher
        return web.json_response(dict(protocol=PROTOCOL, token=self.token, dependencies=dependency_report(),
                                      testDouble=self.test_double, backend=d.backend, defaultMode=d.default_mode,
                                      hasExperiment=d.engine is not None,hasBatch=d.batch is not None,graphAvailable=(d.graph_path/'manifest.json').is_file(),
                                      bindingsAvailable=d.binding_path.is_file(),
                                      backendAvailability=backend_availability()), headers={'Cache-Control':'no-store'})

    async def send_reply(self, socket, result):
        binary = result.pop('_binary', None)
        await socket.send_str(json.dumps(result,ensure_ascii=False,allow_nan=False,separators=(',',':')))
        if binary is not None: await socket.send_bytes(binary)


def main():
    import argparse
    local_path=ROOT/'flylab.local.json'
    defaults=json.loads(local_path.read_text(encoding='utf-8')) if local_path.is_file() else {}
    if not isinstance(defaults,dict) or set(defaults)-{'graph','bindings','backend'} or any(not isinstance(v,str) for v in defaults.values()):
        raise ValueError('flylab.local.json supports string graph, bindings and backend fields only')
    parser = argparse.ArgumentParser(description='FLY LAB C: FAFB v783 connectome / NeuroMechFly')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--graph', type=Path, default=ROOT/defaults.get('graph','data/fafb783/bundle'))
    parser.add_argument('--bindings', type=Path, default=ROOT/defaults.get('bindings','data/fafb783/bindings.json'))
    parser.add_argument('--artifacts', type=Path, default=ROOT/'artifacts/c')
    parser.add_argument('--backend', choices=BACKEND_CHOICES, default=defaults.get('backend','auto'))
    parser.add_argument('--physics-backend', choices=('cpu','warp'), default='cpu',
                        help='Independent physics backend; warp is an explicitly selected candidate')
    parser.add_argument('--noslip-iterations', type=int, default=None)
    parser.add_argument('--physics-control',choices=('cpu','cuda'),default='cpu',help='CUDA hybrid/reflex control requires --physics-backend warp')
    parser.add_argument('--multiccd', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--mode', choices=MODES, default='C_SHADOW')
    parser.add_argument('--doctor', action='store_true')
    parser.add_argument('--restore-checkpoint', help='Restore an artifact checkpoint before accepting browser connections')
    args = parser.parse_args()
    requested_backend = args.backend
    args.backend = resolve_backend(args.backend)
    from ..physics import PhysicsProfile, body_factory as select_body, physics_availability
    physics_profile=PhysicsProfile(backend=args.physics_backend,
        noslip_iterations=args.noslip_iterations,multiccd=args.multiccd,control_backend=args.physics_control)
    if not 1024 <= args.port <= 65535: parser.error('Port must be 1024..65535')
    if args.doctor:
        report = dict(physical=dependency_report(), graph=args.graph.is_dir(), bindings=args.bindings.is_file(),
                      numericalValidation='NOT_RUN_BY_DOCTOR', biologicalValidation=False,
                      backend_selection=requested_backend, backend_availability=backend_availability(),
                      physics_profile=asdict(physics_profile), physics_availability=physics_availability())
        try:
            selected_physics=report['physics_availability'][physics_profile.backend]
            if not selected_physics['available']:
                raise RuntimeError('Selected physics backend unavailable: '+str(selected_physics.get('reason','No CUDA device')))
            graph = GraphStore.load(args.graph); bindings = PortBindings(graph, read_json(args.bindings))
            parameters = validate_execution(bindings, args.mode, args.backend)
            report.update(graph=graph.manifest, bindings=bindings.summary(), status='READY', requested_backend=args.backend,
                          requested_mode=args.mode, model=model_identity(parameters))
            if args.backend!='exp_lif_cpu_reference':
                backend=create_backend(graph, parameters, args.backend)
                report['requested_backend_available']=True
                del backend
        except Exception as e: report.update(status='BLOCKED', error=str(e))
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report['status']=='READY' and report['physical']['ready'] else 2
    if not (ROOT/'FLY_LAB_C.html').is_file(): raise SystemExit('Run python build_c.py first')
    server = CServer(args.port, args.graph, args.bindings, args.artifacts, backend=args.backend, default_mode=args.mode,
                     body_factory=select_body(physics_profile))
    print(f'FLY LAB C {VERSION} - http://127.0.0.1:{args.port} [{args.mode}]', flush=True)
    app=server.app()
    if args.restore_checkpoint:
        name=checked_name(args.restore_checkpoint)
        async def restore_before_listen(app):
            import asyncio
            await asyncio.get_running_loop().run_in_executor(server.executor,server.dispatcher.handle,
                dict(protocol=PROTOCOL,requestId=1,op='restore',payload=dict(name=name)))
        app.on_startup.append(restore_before_listen)
    web.run_app(app, host='127.0.0.1', port=args.port, access_log=None)
    return 0
