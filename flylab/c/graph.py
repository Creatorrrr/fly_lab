"""Immutable master roster + postsynaptic-major CSR: W[post, pre].

Counts are anatomical contacts; signed mV weights are a separate model choice.
No node is inferred from edge endpoints and no isolated node is discarded.
"""
from pathlib import Path
import copy
import json
import re
import shutil
import tempfile
from functools import cached_property
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from .integrity import canonical, digest, file_hash, read_json, write_json, finite

NAMESPACE = 'flywire:fafb:783:'
SIGN = {'ACH': 1, 'GABA': -1, 'GLUT': -1}
MODULATORY = {'DA', 'SER', 'OCT'}
ARRAYS = ('indptr', 'indices', 'counts', 'weights')


def external_id(root, dataset='fafb', snapshot='783'):
    if not isinstance(root, str) or not re.fullmatch(r'[1-9][0-9]{0,19}', root):
        raise ValueError('FAFB root IDs must be decimal strings, never JSON numbers')
    if dataset not in ('fafb', 'banc') or not isinstance(snapshot, str) or not snapshot.isdecimal():
        raise ValueError('Unsupported dataset namespace')
    return f'flywire:{dataset}:{snapshot}:' + root


def normalized_nt(value):
    v = str(value or '').strip().upper()
    return {'ACETYLCHOLINE': 'ACH', 'GLUTAMATE': 'GLUT', 'DOPAMINE': 'DA',
            'SEROTONIN': 'SER', 'OCTOPAMINE': 'OCT'}.get(v, v)


class GraphStore:
    def __init__(self, nodes, indptr, indices, counts, weights, manifest):
        self.nodes = nodes
        self.n = len(nodes)
        self.index = {n['id']: i for i, n in enumerate(nodes)}
        self.indptr, self.indices, self.counts, self.weights = indptr, indices, counts, weights
        self.manifest = copy.deepcopy(manifest)
        self.validate()
        self.matrix = csr_matrix((self.weights, self.indices, self.indptr), shape=(self.n, self.n), copy=False)
        self.groups = {}
        for i, node in enumerate(nodes):
            for region in node.get('regions', []):
                self.groups.setdefault(region, []).append(i)
        self.groups = {k: np.asarray(v, dtype=np.int32) for k, v in sorted(self.groups.items())}
        self.search_text = [' '.join(str(n.get(k, '')) for k in
                            ('id', 'cell_type', 'class', 'super_class', 'soma_side',
                             'output_side', 'nt_type', 'regions')).lower() for n in nodes]
        self.path = None

    def validate(self):
        if not 1 <= self.n <= 2_000_000 or len(self.index) != self.n:
            raise ValueError('Empty, oversized or duplicate master roster')
        dataset = self.manifest.get('dataset_id', 'flywire_fafb').removeprefix('flywire_')
        for node in self.nodes:
            if node.get('id') != external_id(node.get('root_id'), dataset, self.manifest.get('snapshot_id','783')):
                raise ValueError('Invalid namespaced root ID')
            if node.get('soma_side', 'unknown') not in ('left', 'right', 'center', 'unknown'):
                raise ValueError('Invalid biological soma side')
        m = len(self.indices)
        if self.indptr.shape != (self.n + 1,) or self.counts.shape != (m,) or self.weights.shape != (m,):
            raise ValueError('CSR shape mismatch')
        if self.indptr.dtype.kind not in 'iu' or self.indices.dtype.kind not in 'iu' or self.counts.dtype.kind not in 'iu':
            raise ValueError('CSR indices and anatomical counts must be integers')
        if self.indptr[0] != 0 or self.indptr[-1] != m or np.any(np.diff(self.indptr) < 0):
            raise ValueError('Invalid CSR offsets')
        if m and (self.indices.min() < 0 or self.indices.max() >= self.n or self.counts.min() < 1):
            raise ValueError('Invalid endpoint or synapse count')
        if not np.isfinite(self.weights).all():
            raise ValueError('Non-finite synaptic weight')
        for i in range(self.n):
            a = self.indices[self.indptr[i]:self.indptr[i+1]]
            if len(a) > 1 and np.any(np.diff(a.astype(np.int64)) <= 0):
                raise ValueError('CSR rows must have sorted unique presynaptic indices')
        for key, actual in [('simulated_node_count', self.n), ('pair_edge_count', m),
                            ('anatomical_synapse_count', int(self.counts.sum())),
                            ('effective_edge_count', int(np.count_nonzero(self.weights)))]:
            if self.manifest.get(key) != actual:
                raise ValueError('Manifest count mismatch: ' + key)
        if self.manifest.get('matrix_orientation') != 'W[post,pre]':
            raise ValueError('Matrix orientation must be W[post,pre]')
        for a in (self.indptr, self.indices, self.counts, self.weights):
            a.flags.writeable = False

    @property
    def hash(self):
        return self.manifest['graph_hash']

    @cached_property
    def full_brain(self):
        from .data_identity import verify_snapshot
        return verify_snapshot(self)['status'] == 'PASS'

    @classmethod
    def from_edges(cls, nodes, pre, post, counts, *, metadata=None, unit_weight=.275,
                   unknown_policy='block', modulatory_policy='mask_zero'):
        finite(unit_weight, 'unit_weight', 0, 10)
        if unknown_policy not in ('block', 'mask_zero') or modulatory_policy != 'mask_zero':
            raise ValueError('Explicit supported neurotransmitter policies required')
        nodes = copy.deepcopy(nodes)
        n = len(nodes)
        index = {node['id']: i for i, node in enumerate(nodes)}
        if len(index) != n:
            raise ValueError('Duplicate master ID')
        pre, post, counts = np.asarray(pre), np.asarray(post), np.asarray(counts)
        if pre.ndim != 1 or pre.shape != post.shape or pre.shape != counts.shape:
            raise ValueError('Edge column shape mismatch')
        if any(a.size and a.dtype.kind not in 'iu' for a in (pre, post, counts)):
            raise ValueError('Integer endpoint indices and contact counts required')
        if len(pre) and (pre.min() < 0 or post.min() < 0 or pre.max() >= n or post.max() >= n or counts.min() < 1):
            raise ValueError('Unknown endpoint or nonpositive count')
        c = coo_matrix((counts.astype(np.int64), (post.astype(np.int32), pre.astype(np.int32))), shape=(n, n)).tocsr()
        c.sum_duplicates(); c.sort_indices()
        nt = [normalized_nt(node.get('nt_type')) for node in nodes]
        unknown = [i for i, t in enumerate(nt) if t not in SIGN and t not in MODULATORY]
        if unknown and unknown_policy == 'block':
            raise ValueError(f'BLOCKED_NEUROTRANSMITTER: {len(unknown)} unresolved neurons')
        sign = np.asarray([SIGN.get(t, 0) for t in nt], dtype=np.float32)
        weights = c.data.astype(np.float32) * np.float32(unit_weight) * sign[c.indices]
        m = dict(schema='flylab.graph.v3', loader_version='1', dataset_id='flywire_fafb',
                 specimen_id='FAFB', sex='female', snapshot_id='783', annotation_release='unspecified',
                 scope='fixture', source_node_count=n, excluded_node_count=0,
                 excluded_node_ids=[], raw_file_hashes={}, upstream_filters=[], applied_filters=[],
                 side_conventions=dict(annotation='biological', image='FAFB original, mirrored historically',
                                       brain_display='none; no invented anatomy', body='UI x right, y up, z; positive yaw clockwise'))
        m.update(metadata or {})
        m.update(simulated_node_count=n, source_connection_rows=len(pre), pair_edge_count=len(c.data),
                 anatomical_synapse_count=int(c.data.sum()), effective_edge_count=int(np.count_nonzero(weights)),
                 masked_edge_count=int(np.count_nonzero(weights == 0)),
                 unknown_neurotransmitter_count=len(unknown),
                 modulatory_neuron_count=sum(t in MODULATORY for t in nt),
                 matrix_orientation='W[post,pre]',
                 weight_model=dict(unit_weight_mV=unit_weight, signs=SIGN, unknown_policy=unknown_policy,
                                   modulatory_policy=modulatory_policy, receptor_model=False))
        import hashlib
        if m.get('graph_hash_schema') == 'content-v2':
            fields = ('dataset_id','specimen_id','sex','snapshot_id','scope','excluded_node_ids',
                      'upstream_filters','applied_filters','weight_model','matrix_orientation')
            h = hashlib.sha256(canonical(dict(nodes=nodes, model={k:m.get(k) for k in fields})))
            for a, dtype in ((c.indptr,'<i8'),(c.indices,'<i8'),(c.data,'<i8'),(weights,'<f4')):
                h.update(np.asarray(a,dtype=dtype).tobytes())
        else:
            h = hashlib.sha256(canonical(dict(nodes=nodes, metadata=m)))
            for a in (c.indptr, c.indices, c.data, weights):
                h.update(str(a.dtype).encode()); h.update(a.tobytes())
        m['graph_hash'] = h.hexdigest()
        return cls(nodes, c.indptr, c.indices, c.data, weights, m)

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError('Graph bundles are immutable: ' + str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix='.' + path.name + '-', dir=path.parent))
        try:
            with (temporary / 'nodes.jsonl').open('wb') as f:
                for n in self.nodes:
                    f.write(canonical(n) + b'\n')
            for k in ARRAYS:
                np.save(temporary / (k + '.npy'), getattr(self, k), allow_pickle=False)
            m = copy.deepcopy(self.manifest)
            m['bundle_files'] = {p.name: file_hash(p) for p in sorted(temporary.iterdir())}
            write_json(temporary / 'manifest.json', m)
            write_json(temporary / 'identity.json', {'manifest_sha256': file_hash(temporary / 'manifest.json')})
            temporary.rename(path)
        except Exception:
            shutil.rmtree(temporary)
            raise
        self.path = path.resolve()

    @classmethod
    def load(cls, path):
        path = Path(path)
        manifest = read_json(path / 'manifest.json')
        if read_json(path / 'identity.json').get('manifest_sha256') != file_hash(path / 'manifest.json'):
            raise ValueError('Graph manifest hash mismatch')
        expected = {'nodes.jsonl'} | {k + '.npy' for k in ARRAYS}
        if manifest.get('schema') != 'flylab.graph.v3' or set(manifest.get('bundle_files', {})) != expected:
            raise ValueError('Invalid graph bundle schema/files')
        for name, sha in manifest['bundle_files'].items():
            if file_hash(path / name) != sha:
                raise ValueError('Graph file hash mismatch: ' + name)
        with (path / 'nodes.jsonl').open() as f:
            nodes = [json.loads(line) for line in f]
        arrays = [np.load(path / (k + '.npy'), mmap_mode='r', allow_pickle=False) for k in ARRAYS]
        graph = cls(nodes, *arrays, manifest)
        graph.path = path.resolve()
        return graph

    def resolve(self, ids, maximum=512):
        if not isinstance(ids, list) or len(ids) > maximum or any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('Unique string neuron IDs required')
        try:
            return np.asarray([self.index[i] for i in ids], dtype=np.int32)
        except KeyError as e:
            raise ValueError('BLOCKED_PORT_BINDING: unresolved ID ' + str(e)) from e

    def catalog(self, query='', offset=0, limit=100):
        if not isinstance(query, str) or len(query) > 200:
            raise ValueError('Catalog query must be at most 200 characters')
        from .integrity import bounded_int
        bounded_int(offset, 'offset', 0, self.n); bounded_int(limit, 'limit', 1, 512)
        found = [i for i, s in enumerate(self.search_text) if query.lower() in s]
        return dict(total=len(found), offset=offset, items=[self.nodes[i] for i in found[offset:offset+limit]])

    def neighbors(self, neuron_id, limit=100):
        i = int(self.resolve([neuron_id])[0])
        from .integrity import bounded_int
        bounded_int(limit, 'limit', 1, 512)
        start, end = self.indptr[i:i+2]
        order = np.argsort(-self.counts[start:end], kind='stable')[:limit] + start
        return dict(direction='incoming', total=int(end-start), items=[dict(
            pre=self.nodes[int(self.indices[j])]['id'], post=neuron_id,
            count=int(self.counts[j]), weight_mV=float(self.weights[j]), edge_index=int(j)) for j in order])
