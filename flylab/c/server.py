"""Local v3 observation server with bounded catalogs and binary subscriptions."""
from pathlib import Path
import json
import secrets
from aiohttp import web
from . import VERSION, PROTOCOL, MODES
from .engine import CEngine
from .graph import GraphStore
from .ports import PortBindings
from .neural import LIFParameters, NEURAL_BACKENDS, create_backend
from .protocol import signal_frame
from .storage import StateStore
from .integrity import read_json, write_json, digest, checked_name, bounded_int
from .workbench import checked_world, intervention_rows
from ..server import Server as BServer, ROOT
from ..body import FlyGymBody, dependency_report


class CDispatcher:
    def __init__(self, graph_path, binding_path, artifacts, body_factory=FlyGymBody,
                 backend='exp_lif_cpu_reference', default_mode='C_SHADOW'):
        self.graph_path, self.binding_path = Path(graph_path), Path(binding_path)
        self.artifacts = Path(artifacts).resolve()
        self.factory = body_factory
        self.backend, self.default_mode = backend, default_mode
        self.graph = self.bindings = self.engine = None
        self.profile_cache = {}

    def load(self):
        if self.graph is None:
            graph = GraphStore.load(self.graph_path)
            bindings = PortBindings(graph, read_json(self.binding_path))
            self.graph, self.bindings = graph, bindings
            self.profile_cache[self.binding_path.name] = bindings

    def profiles(self):
        self.load()
        paths = [self.binding_path]+[p for p in sorted(self.binding_path.parent.glob('bindings-*.json')) if p!=self.binding_path]
        for path in paths[:16]:
            if path.name not in self.profile_cache:
                self.profile_cache[path.name] = PortBindings(self.graph, read_json(path))
        return self.profile_cache

    def checkpoint_bindings(self, state):
        for binding in self.profiles().values():
            if binding.hash == state.get('binding_hash'): return binding
        raise ValueError('Checkpoint binding profile is unavailable; no implicit port conversion')

    def need(self):
        if self.engine is None: raise ValueError('Initialize an experiment first')
        return self.engine

    def replace(self, new):
        old, self.engine = self.engine, new
        self.bindings = new.bindings
        if new.neural: self.backend = new.neural.backend
        if old: old.close()
        return self.ready()

    def ready(self):
        e = self.need()
        return dict(capabilities=dict(version=VERSION, protocol=PROTOCOL, modes=list(MODES),
                    backend=e.neural.backend if e.neural else 'legacy_b_rate', neuralBackends=list(NEURAL_BACKENDS),
                    physical=not e.body.test_double, fullBrain=e.neural is not None and e.graph.full_brain,
                    maxSubscription=512, maxAdvance=10, flight=False, biologicalValidation=False,
                    checkpoint=True, replay=True, selectedBinarySignals=True),
                    manifest=e.graph.manifest, binding=e.bindings.summary(), frame=e.frame(),
                    profiles=[dict(name=name, profile=b.spec.get('profile',name), hash=b.hash,
                                   current=b.hash==e.bindings.hash) for name,b in self.profiles().items()])

    def handle(self, message):
        if not isinstance(message, dict) or message.get('protocol') != PROTOCOL:
            raise ValueError('flylab.protocol.v3 required')
        rid = bounded_int(message.get('requestId'), 'requestId', 1)
        payload = message.get('payload', {})
        if not isinstance(payload, dict): raise ValueError('Payload object required')
        op = message.get('op')
        if op == 'init':
            if set(payload)-{'mode', 'seed', 'config', 'motion_expected', 'profile'}: raise ValueError('Unknown initialization field')
            self.load()
            profile = payload.get('profile', self.binding_path.name)
            bindings = self.profiles().get(profile)
            if bindings is None: raise ValueError('Unknown binding profile')
            current = self.engine
            if current and current.recorder: raise ValueError('Finish recording before starting a new experiment')
            saved = None
            if current:
                saved = 'before-init-'+secrets.token_hex(6)
                StateStore.save(self.artifacts/'checkpoints'/saved, current.checkpoint())
            new = CEngine(self.graph, bindings, mode=payload.get('mode', self.default_mode),
                          seed=payload.get('seed', 42), config=payload.get('config'), backend=self.backend,
                          body_factory=self.factory, motion_expected=payload.get('motion_expected', True))
            result = self.replace(new)
            result['source_checkpoint'] = saved
        elif op == 'attach': result = self.ready()
        elif op == 'frame': result = self.need().frame()
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
            self.backend = target
            result = self.replace(new)
            result['source_checkpoint'] = name
        elif op == 'advance': result = self.need().step(bounded_int(payload.get('steps', 10), 'steps', 0, 10))
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
                      for name, port in [('DNp09', e.bindings.spec['motor']['forward']),
                                         ('MDN', e.bindings.spec['motor']['backward']),
                                         ('DNa02 left', e.bindings.spec['motor']['yaw_left']),
                                         ('DNa02 right', e.bindings.spec['motor']['yaw_right'])] if port['ids']]
            result.append(dict(name='Motor outputs', ids=[e.graph.nodes[int(i)]['id'] for i in e.bindings.motor_indices], binding_hash=e.bindings.hash))
            result.extend(dict(name=p['name'], ids=p['ids'], binding_hash=e.bindings.hash)
                          for p,_ in e.bindings.sensory if len(p['ids'])<=512)
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
            response['_binary'] = signal_frame(self.engine)
            self.engine.selected_events = []
        return response

    def close(self):
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
                                      hasExperiment=d.engine is not None, graphAvailable=(d.graph_path/'manifest.json').is_file(),
                                      bindingsAvailable=d.binding_path.is_file()), headers={'Cache-Control':'no-store'})

    async def send_reply(self, socket, result):
        binary = result.pop('_binary', None)
        await socket.send_str(json.dumps(result,ensure_ascii=False,allow_nan=False,separators=(',',':')))
        if binary is not None: await socket.send_bytes(binary)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='FLY LAB C: FAFB v783 connectome / NeuroMechFly')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--graph', type=Path, default=ROOT/'data/fafb783/bundle')
    parser.add_argument('--bindings', type=Path, default=ROOT/'data/fafb783/bindings.json')
    parser.add_argument('--artifacts', type=Path, default=ROOT/'artifacts/c')
    parser.add_argument('--backend', choices=NEURAL_BACKENDS, default='exp_lif_cpu_reference')
    parser.add_argument('--mode', choices=MODES, default='C_SHADOW')
    parser.add_argument('--doctor', action='store_true')
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535: parser.error('Port must be 1024..65535')
    if args.doctor:
        report = dict(physical=dependency_report(), graph=args.graph.is_dir(), bindings=args.bindings.is_file(),
                      cudaValidated=False, biologicalValidation=False)
        try:
            graph = GraphStore.load(args.graph); bindings = PortBindings(graph, read_json(args.bindings))
            report.update(graph=graph.manifest, bindings=bindings.summary(), status='READY', requested_backend=args.backend)
            if args.backend!='exp_lif_cpu_reference':
                backend=create_backend(graph, backend=args.backend)
                report['requested_backend_available']=True
                del backend
        except Exception as e: report.update(status='BLOCKED', error=str(e))
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report['status']=='READY' and report['physical']['ready'] else 2
    if not (ROOT/'FLY_LAB_C.html').is_file(): raise SystemExit('Run python build_c.py first')
    server = CServer(args.port, args.graph, args.bindings, args.artifacts, backend=args.backend, default_mode=args.mode)
    print(f'FLY LAB C {VERSION} — http://127.0.0.1:{args.port} · {args.mode}', flush=True)
    web.run_app(server.app(), host='127.0.0.1', port=args.port, access_log=None)
    return 0
