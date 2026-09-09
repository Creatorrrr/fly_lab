#!/usr/bin/env python3
"""Resolve a conservative engineering profile against the downloaded snapshot.

Only idealized scalar odor is bound naturally. Vision/contact/proprioception
remain explicitly unbound until reviewed retinotopy and sensory boundaries are
available. DN stimulation is an intervention, not a natural sensory success.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore, external_id
from flylab.c.ports import PortBindings
from flylab.c.integrity import write_json

CODEX = 'https://codex.flywire.ai/faq'
STEER = 'https://www.nature.com/articles/s41586-024-07039-2'
WALK = 'https://www.nature.com/articles/s41586-024-07523-9'


def build(graph):
    def resolve_type(name):
        found = [n['id'] for n in graph.nodes if n.get('cell_type') == name]
        if not found: raise ValueError('BLOCKED_PORT_BINDING: missing cell type ' + name)
        return found
    review = dict(review_status='engineering_reviewed', evidence=[CODEX],
                  uncertainty='Scalar odor concentration to ORN drive is an engineering assumption; receptor tuning and gain are not biologically calibrated.')
    sensory = [dict(name='odor_DM1', channel='odor_mean', ids=resolve_type('ORN_DM1'),
                    input_kind='sensory', boundary='annotated olfactory receptor neurons, bilateral scalar input',
                    method='drive_mV', gain=18., baseline=0., cap=24., tau_s=.020,
                    delay_controls=0, offset=0., scale=1., **review)]
    def port(ids, gain, source, uncertainty, **extra):
        return dict(ids=ids, gain=gain, review_status='engineering_reviewed',
                    evidence=[CODEX, source], uncertainty=uncertainty, **extra)
    # Reviewed exact IDs from this FAFB annotation snapshot; do not route by a
    # suffix, display position, hemisphere letter, or another specimen's ID.
    left = external_id('720575940629327659')
    right = external_id('720575940604737708')
    for neuron_id, side in ((left, 'left'), (right, 'right')):
        node = graph.nodes[int(graph.resolve([neuron_id])[0])]
        if node['cell_type'] != 'DNa02' or node['soma_side'] != side or node['super_class'] != 'descending':
            raise ValueError('BLOCKED_PORT_BINDING: DNa02 review no longer matches annotation')
    motor = dict(
        forward=port(resolve_type('DNp09'), .08, WALK,
                     'DNp09 is a context-dependent command-neuron candidate; an engineering rate-to-CPG decoder does not reconstruct its VNC network.'),
        backward=port(resolve_type('MDN'), .08, WALK,
                      'MDN backward-walking role informs this engineering readout; motor gain and leg circuitry are not biological reconstructions.'),
        stop=dict(ids=[], gain=0., reason='No stop population reviewed; no implicit tonic forward drive.'),
        yaw_left=port([left], .08, STEER,
                      'Ipsiversive output direction inferred from DNa02 literature plus corrected soma annotations; exact axon projection is not independently reconstructed.',
                      output_side='left', side_basis='explicit ID review and literature ipsiversive steering'),
        yaw_right=port([right], .08, STEER,
                       'Ipsiversive output direction inferred from DNa02 literature plus corrected soma annotations; exact axon projection is not independently reconstructed.',
                       output_side='right', side_basis='explicit ID review and literature ipsiversive steering'))
    spec = dict(schema='flylab.bindings.v3', graph_hash=graph.hash, profile='fafb783-odor-dn-engineering-v1',
                sensory=sensory, motor=motor, unused_observations=['panorama', 'nearRanges', 'contact', 'angularVelocity',
                                                               'odorChange', 'danger', 'forwardSpeed', 'clearanceDown', 'clearanceUp'],
                biological_validation=False, output_convention='positive yaw is clockwise; right DNa02 minus left DNa02',
                no_tonic_drive=True, side_convention='corrected biological annotations; no image-derived lateralization')
    PortBindings(graph, spec)
    return spec


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph', type=Path, default=Path('data/fafb783/bundle'))
    p.add_argument('--out', type=Path, default=Path('data/fafb783/bindings.json'))
    a = p.parse_args()
    if a.out.exists(): raise SystemExit('Binding profiles are versioned; choose a new output path')
    spec = build(GraphStore.load(a.graph))
    write_json(a.out, spec)
    print(a.out, 'sensory targets:', len(spec['sensory'][0]['ids']))
