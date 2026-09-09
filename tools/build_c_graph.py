#!/usr/bin/env python3
"""Download the public FAFB snapshot products and build a hashed C bundle.

The official master table includes disconnected neurons. Connection rows are
neuropil partitions, never combined with a second pair-total table. Downloads
are pinned locally by bytes, SHA-256 and the declared annotation acquisition.
"""
from pathlib import Path
import argparse
import array
import csv
import gzip
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore, external_id
from flylab.c.integrity import file_hash, read_json, write_json
import numpy as np

PRODUCTS = ('neurons', 'classification', 'consolidated_cell_types', 'connections_princeton')
SOURCE = 'https://codex.flywire.ai/api/download_resource?data_product={}&dataset=fafb'


def rows(path):
    with gzip.open(path, 'rt', newline='', encoding='utf-8-sig') as f:
        yield from csv.DictReader(f)


def download(raw):
    raw.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in PRODUCTS:
        path = raw / (name + '.csv.gz')
        url = SOURCE.format(name)
        if not path.exists():
            temporary = path.with_suffix('.part')
            subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location',
                            '--max-time', '300', url, '-o', str(temporary)], check=True)
            next(rows(temporary))  # refuse login HTML or malformed/empty gzip
            temporary.rename(path)
        manifest.append(dict(product=name, url=url, bytes=path.stat().st_size,
                             sha256=file_hash(path)))
    write_json(raw / 'download_manifest.json', manifest)


def build(raw, output, unknown_policy, *, acquisition=None):
    began = time.perf_counter()
    sources = read_json(raw / 'download_manifest.json')
    for p in sources:
        if p['product'] in PRODUCTS and file_hash(raw / (p['product'] + '.csv.gz')) != p['sha256']:
            raise ValueError('Raw source checksum mismatch: ' + p['product'])
    if {r['product'] for r in sources} != set(PRODUCTS):
        raise ValueError('All four hashed official products are required')
    nodes = []
    for r in rows(raw / 'neurons.csv.gz'):
        nodes.append(dict(id=external_id(r['root_id']), root_id=r['root_id'],
                          cell_type='', super_class='', soma_side='unknown',
                          output_side='unreviewed', nt_type=r['nt_type'],
                          nt_confidence=float(r['nt_type_score'] or 0),
                          source_group=r.get('group', ''), regions=[]))
    nodes.sort(key=lambda n: int(n['root_id']))
    index = {n['root_id']: i for i, n in enumerate(nodes)}
    if len(index) != len(nodes):
        raise ValueError('Duplicate master roster ID')
    for product in ('classification', 'consolidated_cell_types'):
        seen = set()
        for r in rows(raw / (product + '.csv.gz')):
            root = r['root_id']
            if root not in index or root in seen:
                raise ValueError('Duplicate or unrecognized annotation ID: ' + root)
            seen.add(root)
            n = nodes[index[root]]
            if product == 'classification':
                n.update({k: r.get(k, '') for k in ('class', 'super_class', 'sub_class', 'flow', 'nerve', 'hemilineage')})
                n['soma_side'] = r.get('side') or 'unknown'
            else:
                n['cell_type'] = r.get('primary_type', '')
                n['additional_types'] = r.get('additional_type(s)', '')
    pre, post, counts = array.array('i'), array.array('i'), array.array('q')
    regions = [set() for _ in nodes]
    nt_mismatch_rows = 0
    # Duplicate per-neuropil entries are rejected with a compact uint64 key
    # after reading. Pair aggregation occurs only after this invariant holds.
    roi_codes, roi_values = {}, array.array('H')
    for r in rows(raw / 'connections_princeton.csv.gz'):
        try:
            a, b = index[r['pre_root_id']], index[r['post_root_id']]
        except KeyError as e:
            raise ValueError('Connection endpoint absent from master roster') from e
        region = r['neuropil']
        if not region or region.lower() in ('all', 'total'):
            raise ValueError('Expected neuropil partitions; total rows would double-count')
        value = r['syn_count']
        if not value.isdecimal() or not 1 <= int(value) <= 1_000_000_000:
            raise ValueError('Positive integer synapse count required')
        code = roi_codes.setdefault(region, len(roi_codes))
        pre.append(a); post.append(b); counts.append(int(value)); roi_values.append(code)
        regions[a].add(region); regions[b].add(region)
        if r.get('nt_type') and nodes[a]['nt_type'] and r['nt_type'] != nodes[a]['nt_type']:
            nt_mismatch_rows += 1
    pre, post, counts = np.asarray(pre, dtype=np.int32), np.asarray(post, dtype=np.int32), np.asarray(counts, dtype=np.int64)
    codes = (post.astype(np.uint64) * len(nodes) + pre.astype(np.uint64)) * max(1, len(roi_codes)) + np.asarray(roi_values, dtype=np.uint64)
    codes.sort()
    if len(codes) > 1 and np.any(codes[1:] == codes[:-1]):
        raise ValueError('Duplicate pair/neuropil rows; refuse ambiguous aggregation')
    del codes
    for n, region in zip(nodes, regions):
        n['regions'] = sorted(region)
    graph = GraphStore.from_edges(nodes, pre, post, counts, unknown_policy=unknown_policy,
        metadata=dict(scope='full_snapshot', graph_hash_schema='content-v2',
                      annotation_release=acquisition or datetime.now(timezone.utc).isoformat(),
                      raw_file_hashes={p['product'] + '.csv.gz': p['sha256'] for p in sources},
                      sources=[p['url'] for p in sources],
                      master_roster_product='neurons', edge_product='connections_princeton',
                      upstream_filters=['FAFB v783 proofread neurons; Codex pair total >=5 synapses',
                                        'connection rows partitioned by neuropil'],
                      applied_filters=['no neuron exclusions; no additional edge threshold'],
                      connection_nt_disagreement_rows=nt_mismatch_rows,
                      nt_precedence='master neuron prediction; disagreements counted',
                      source_documentation='https://codex.flywire.ai/faq'))
    graph.save(output)
    report = dict(status='PASS', seconds=time.perf_counter()-began, manifest=graph.manifest,
                  bundle=str(Path(output).resolve()), physicalExecuted=False)
    print(json.dumps(report, indent=2))
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('data/fafb783/raw'))
    parser.add_argument('--out', type=Path, default=Path('data/fafb783/bundle'))
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--unknown-policy', choices=('block', 'mask_zero'), default='block')
    args = parser.parse_args()
    if args.download:
        download(args.raw)
    build(args.raw, args.out, args.unknown_policy)


if __name__ == '__main__':
    main()
