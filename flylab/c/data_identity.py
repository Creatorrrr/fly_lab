"""Portable content identities and independent pinned snapshot acceptance."""
from pathlib import Path
import hashlib
import numpy as np
from .integrity import canonical, digest, read_json


def identities(graph):
    roster = hashlib.sha256(canonical([n['id'] for n in graph.nodes])).hexdigest()
    h = hashlib.sha256(roster.encode())
    for values in (graph.indptr, graph.indices, graph.counts):
        h.update(np.asarray(values, dtype='<i8').tobytes())
    anatomy = h.hexdigest()
    h.update(np.asarray(graph.weights, dtype='<f4').tobytes())
    h.update(canonical(graph.manifest['weight_model']))
    return dict(roster_sha256=roster, anatomy_sha256=anatomy, model_sha256=h.hexdigest(),
                annotations_sha256=digest(graph.nodes))


def reference_for(graph):
    return dict(schema='flylab.snapshot_reference.v1',
                dataset_id=graph.manifest['dataset_id'], snapshot_id=graph.manifest['snapshot_id'],
                specimen_id=graph.manifest['specimen_id'], source_node_count=graph.n,
                identities=identities(graph), raw_file_hashes=graph.manifest.get('raw_file_hashes', {}),
                upstream_filters=graph.manifest.get('upstream_filters', []),
                sources=graph.manifest.get('sources', []), graph_hash=graph.hash)


def verify_snapshot(graph, reference_path=None):
    """Self-declared full_snapshot metadata is necessary, never sufficient."""
    m = graph.manifest
    if m.get('scope') != 'full_snapshot' or m.get('excluded_node_count') != 0 or m.get('excluded_node_ids'):
        return dict(status='FAIL', reason='A complete unmodified source snapshot is required', scope=m.get('scope'))
    if reference_path is None:
        reference_path = Path(__file__).resolve().parents[2]/'data/releases'/f"{m['dataset_id']}-{m['snapshot_id']}.json"
    reference_path = Path(reference_path)
    if not reference_path.is_file():
        return dict(status='BLOCKED', reason='Independent snapshot reference unavailable', reference=str(reference_path))
    r = read_json(reference_path)
    checks = dict(schema=r.get('schema') == 'flylab.snapshot_reference.v1',
                  source_count=m.get('source_node_count') == graph.n == r.get('source_node_count'),
                  raw_hashes=bool(r.get('raw_file_hashes')) and m.get('raw_file_hashes') == r.get('raw_file_hashes'),
                  upstream_filters=m.get('upstream_filters') == r.get('upstream_filters'))
    for k in ('dataset_id', 'snapshot_id', 'specimen_id'):
        checks[k] = m.get(k) == r.get(k)
    actual = identities(graph)
    for k in ('roster_sha256', 'anatomy_sha256'):
        checks[k] = actual[k] == r.get('identities', {}).get(k)
    return dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
                reference=str(reference_path), reference_sha256=digest(r), identities=actual,
                source_nodes=graph.n, model_matches_reference=actual['model_sha256']==r.get('identities', {}).get('model_sha256'),
                scope='Pinned source roster and anatomical graph; physiology is a separate model')


def validation_status(gates, required, cases=()):
    failures = [name for name in required if gates.get(name, {}).get('status') in ('FAIL', 'FAILED')]
    failures.extend(c.get('name', 'physical_case') for c in cases if c.get('status') in ('FAIL', 'FAILED'))
    incomplete = [name for name in required if gates.get(name, {}).get('status') != 'PASS' and name not in failures]
    if failures: return dict(status='FAIL', exit_code=1, failed=failures, incomplete=incomplete)
    if incomplete: return dict(status='BLOCKED', exit_code=2, failed=[], incomplete=incomplete)
    return dict(status='COMPLETE_WITH_LIMITATIONS', exit_code=0, failed=[], incomplete=[])
