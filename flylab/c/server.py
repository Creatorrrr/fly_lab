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
from .integrity import read_json, checked_name, bounded_int
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

    def load(self):
        if self.graph is None:
            graph = GraphStore.load(self.graph_path)
            bindings = PortBindings(graph, read_json(self.binding_path))
            self.graph, self.bindings = graph, bindings

    def need(self):
        if self.engine is None: raise ValueError('Initialize an experiment first')
        return self.engine

    def replace(self, new):
        old, self.engine = self.engine, new
        if old: old.close()
        return self.ready()

    def ready(self):
        e = self.need()
        return dict(capabilities=dict(version=VERSION, protocol=PROTOCOL, modes=list(MODES),
                    backend=e.neural.backend if e.neural else 'legacy_b_rate', neuralBackends=list(NEURAL_BACKENDS),
                    physical=not e.body.test_double, fullBrain=e.neural is not None and e.graph.full_brain,
                    maxSubscription=512, maxAdvance=10, flight=False, biologicalValidation=False,
                    checkpoint=True, replay=True, selectedBinarySignals=True),
                    manifest=e.graph.manifest, binding=e.bindings.summary(), frame=e.frame())

    def handle(self, message):
        if not isinstance(message, dict) or message.get('protocol') != PROTOCOL:
            raise ValueError('flylab.protocol.v3 required')
        rid = bounded_int(message.get('requestId'), 'requestId', 1)
        payload = message.get('payload', {})
        if not isinstance(payload, dict): raise ValueError('Payload object required')
        op = message.get('op')
        if op == 'init':
            if set(payload)-{'mode', 'seed', 'config', 'motion_expected'}: raise ValueError('Unknown initialization field')
            self.load()
            new = CEngine(self.graph, self.bindings, mode=payload.get('mode', self.default_mode),
                          seed=payload.get('seed', 42), config=payload.get('config'), backend=self.backend,
                          body_factory=self.factory, motion_expected=payload.get('motion_expected', True))
            result = self.replace(new)
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
        elif op == 'regions': result = self.need().region_summary()
        elif op == 'command': result = self.need().command(payload.get('type'), payload.get('payload', {}))
        elif op == 'checkpoint':
            name = checked_name(payload.get('name', 'checkpoint-'+secrets.token_hex(6)))
            result = StateStore.save(self.artifacts/'checkpoints'/name, self.need().checkpoint())
        elif op == 'restore':
            self.load(); name = checked_name(payload.get('name'))
            state = StateStore.load(self.artifacts/'checkpoints'/name)
            result = self.replace(CEngine.from_checkpoint(self.graph, self.bindings, state, self.factory))
        elif op == 'checkpoints':
            directory = self.artifacts/'checkpoints'
            result = [{'name': p.name} for p in sorted(directory.iterdir()) if p.is_dir() and (p/'manifest.json').is_file()] if directory.exists() else []
        elif op == 'record_start':
            name = checked_name(payload.get('name', 'run-'+secrets.token_hex(6)))
            self.need().start_recording(self.artifacts/'runs'/name, payload.get('ids'))
            result = dict(name=name, frame=self.need().frame())
        elif op == 'record_stop': self.need().stop_recording(); result = self.need().frame()
        elif op == 'paired':
            from .experiments import paired
            name = checked_name(payload.get('name', 'pair-'+secrets.token_hex(6)))
            result = paired(self.need(), self.artifacts/'experiments'/name,
                            payload.get('intervention', {'kind':'motor_disconnect'}),
                            seconds=payload.get('seconds', .5), onset=payload.get('onset', .2))
        elif op == 'replay':
            from .experiments import replay_recording
            name = checked_name(payload.get('name'))
            self.load()
            replayed = replay_recording(self.graph, self.bindings, self.artifacts/'runs'/name, self.factory)
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
