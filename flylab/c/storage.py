"""Chunked recordings and atomic checkpoints. NPY only, never pickle."""
from pathlib import Path
import copy
import importlib.metadata
import os
import shutil
import tempfile
import time
import numpy as np
from .integrity import canonical, digest, file_hash, read_json, write_json, checked_name


def runtime_versions():
    import platform
    versions = {'python': platform.python_version(), 'numpy': np.__version__}
    for name in ('scipy', 'mujoco', 'flygym', 'cupy-cuda12x', 'cupy-cuda13x'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    return versions


class StateStore:
    @staticmethod
    def save(path, state):
        path = Path(path)
        if path.exists():
            raise FileExistsError('Checkpoint already exists')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix='.' + path.name + '-', dir=path.parent))
        files = {}
        def pack(value):
            if isinstance(value, np.ndarray):
                if value.dtype.hasobject:
                    raise ValueError('Object arrays are prohibited')
                name = f'array_{len(files):05}.npy'
                np.save(temporary / name, value, allow_pickle=False)
                files[name] = file_hash(temporary / name)
                return {'__npy__': name}
            if isinstance(value, dict): return {k: pack(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)): return [pack(v) for v in value]
            if isinstance(value, np.generic): return value.item()
            return value
        try:
            encoded = pack(state)
            write_json(temporary / 'state.json', encoded)
            files['state.json'] = file_hash(temporary / 'state.json')
            write_json(temporary / 'manifest.json', {'schema': 'flylab.state_store.v3', 'files': files,
                                                    'versions': runtime_versions(), 'state_hash': digest(files)})
            temporary.rename(path)
        except Exception:
            shutil.rmtree(temporary)
            raise
        return dict(name=path.name, state_hash=digest(files), bytes=sum(p.stat().st_size for p in path.iterdir()))

    @staticmethod
    def load(path):
        path = Path(path)
        m = read_json(path / 'manifest.json')
        if m.get('schema') != 'flylab.state_store.v3' or not isinstance(m.get('files'), dict) or len(m['files']) > 1000:
            raise ValueError('Invalid checkpoint manifest')
        if m.get('versions') != runtime_versions():
            raise ValueError('Checkpoint runtime versions differ; cross-version continuation is not validated')
        if m.get('state_hash') != digest(m['files']) or 'state.json' not in m['files']:
            raise ValueError('Checkpoint identity mismatch')
        for name, sha in m['files'].items():
            checked_name(name)
            if not (name == 'state.json' or name.startswith('array_') and name.endswith('.npy')):
                raise ValueError('Unknown checkpoint file')
            if (path / name).is_symlink() or file_hash(path / name) != sha:
                raise ValueError('Checkpoint file hash mismatch: ' + name)
        def unpack(value):
            if isinstance(value, dict):
                if '__npy__' in value:
                    if set(value) != {'__npy__'} or value['__npy__'] not in m['files'] or not value['__npy__'].endswith('.npy'):
                        raise ValueError('Invalid checkpoint array reference')
                    return np.load(path / value['__npy__'], allow_pickle=False)
                return {k: unpack(v) for k, v in value.items()}
            if isinstance(value, list): return [unpack(v) for v in value]
            return value
        return unpack(read_json(path / 'state.json'))


class Recorder:
    """Fixed recording cohort independent of display subscription, bounded RAM."""
    def __init__(self, path, provenance, ids, max_bytes=2*1024**3, chunk_rows=100):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.ids = list(ids)
        self.max_bytes = max_bytes
        self.chunk_rows = chunk_rows
        self.rows = []
        self.bytes = 0
        self.closed = False
        self.manifest = dict(schema='flylab.recording.v3', provenance=copy.deepcopy(provenance),
                             cohort_ids=self.ids, signal_columns=['voltage_mV', 'rate_Hz'],
                             neural_dt=provenance['neural_dt'], signals_hz=100, body_hz=10,
                             spike_scope='fixed selected cohort', full_voltage_recording=False,
                             chunks=[], status='OPEN', dropped_records=0, versions=runtime_versions())
        self.events = (self.path / 'events.jsonl').open('wb')
        self.bodies = (self.path / 'body.jsonl').open('wb')
        write_json(self.path / 'manifest.json', self.manifest)

    def _budget(self, size):
        if self.closed: raise RuntimeError('Recorder closed')
        if self.bytes + size > self.max_bytes:
            raise RuntimeError('FAULT_RECORDING_CAPACITY: recording stopped, no silent data loss')
        self.bytes += size

    def event(self, record):
        data = canonical(record) + b'\n'
        self._budget(len(data)); self.events.write(data); self.events.flush()

    def body(self, record):
        data = canonical(record) + b'\n'
        self._budget(len(data)); self.bodies.write(data); self.bodies.flush()

    def signal(self, tick, voltage, rate):
        row = np.stack([voltage, rate], axis=-1).astype(np.float32)
        if row.shape != (len(self.ids), 2) or not np.isfinite(row).all():
            raise ValueError('Invalid recorded signals')
        self._budget(row.nbytes + 8)
        self.rows.append((int(tick), row))
        if len(self.rows) >= self.chunk_rows: self.flush()

    def flush(self):
        if not self.rows: return
        serial = len(self.manifest['chunks'])
        name = f'signals_{serial:06}.npy'
        ticks_name = f'ticks_{serial:06}.npy'
        values = np.stack([r[1] for r in self.rows])
        ticks = np.asarray([r[0] for r in self.rows], dtype=np.int64)
        for filename, array in ((name, values), (ticks_name, ticks)):
            temp = self.path / (filename + '.tmp')
            with temp.open('wb') as f:
                np.save(f, array, allow_pickle=False); f.flush(); os.fsync(f.fileno())
            temp.replace(self.path / filename)
        self.manifest['chunks'].append(dict(file=name, ticks_file=ticks_name, rows=len(ticks),
            start_tick=int(ticks[0]), end_tick=int(ticks[-1]), sha256=file_hash(self.path / name),
            ticks_sha256=file_hash(self.path / ticks_name)))
        self.rows.clear()
        write_json(self.path / 'manifest.json', self.manifest)

    def close(self, status='COMPLETE', reason=None):
        if self.closed: return
        try:
            self.flush()
            self.events.flush(); self.bodies.flush()
            self.manifest.update(status=status, reason=reason, bytes=self.bytes,
                                 events_sha256=file_hash(self.path / 'events.jsonl'),
                                 body_sha256=file_hash(self.path / 'body.jsonl'))
            write_json(self.path / 'manifest.json', self.manifest)
        finally:
            self.events.close(); self.bodies.close(); self.closed = True
