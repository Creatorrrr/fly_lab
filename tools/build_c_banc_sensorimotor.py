#!/usr/bin/env python3
"""Build same-specimen BANC ports and an explicit leg actuator hypothesis."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.ports import PortBindings
from flylab.c.neuromuscular import build_spec, SOURCE, FECO
from flylab.c.integrity import write_json


def build(graph,version=1):
    cns = build_spec(graph,version)
    def ids(types, side=None):
        values = [n['id'] for n in graph.nodes if n['cell_type'] in types and
                  (side is None or n['soma_side'] == side)]
        if not values: raise ValueError('Missing BANC cell type: ' + str(types))
        return values
    review = dict(review_status='engineering_reviewed', evidence=[SOURCE],
                  uncertainty='BANC annotation membership; input gain and readout are engineering hypotheses, not physiological calibration.')
    sensory = []
    for channel, types, side in [('odor_left', ('ORN_DM1',), 'left'),
                                  ('odor_right', ('ORN_DM1',), 'right'),
                                  ('danger', ('ORN_DA2',), None)]:
        sensory.append(dict(name=channel, channel=channel, ids=ids(types, side), input_kind='sensory',
                            method='drive_mV', gain=18., baseline=0., cap=24., tau_s=.02,
                            delay_controls=1, offset=0., scale=1., **review))
    motor = {}
    for role, types, side, gain in [('forward', ('DNg100', 'DNg97'), None, .008),
                                   ('backward', ('MDN',), None, .015),
                                   ('yaw_left', ('DNa02',), 'left', .004),
                                   ('yaw_right', ('DNa02',), 'right', .004)]:
        motor[role] = dict(ids=ids(types, side), gain=gain, output_side=side, **review)
    motor['stop'] = dict(ids=[], gain=0.)
    spec = dict(schema='flylab.bindings.v3', graph_hash=graph.hash,
                profile=f'banc888-neuromuscular-v{version}', profile_version=version,
                sensory=sensory, motor=motor, neuromuscular=cns, no_tonic_drive=True,
                sensor_model=dict(kind='compressive-odor-v2', half_concentration=1.),
                motor_decoder=dict(kind='bounded-opponent-v1', speed_limit=1., yaw_limit=1.5, stop_half_drive=1.),
                hazard_semantics='geosmin olfactory proxy', biological_validation=False,
                unused_observations=['retinal image', 'vestibular input', 'unresolved leg receptor tuning'],
                motor_readout_only=True,
                interpretation='Descending speed/yaw are telemetry; BANC motor neuron rates actuate joints directly. No predefined gait CPG.')
    PortBindings(graph, spec)
    return spec


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph', default='data/acquisitions/banc888-v2-20260909/bundle')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--version',type=int,choices=(1,2,3),default=1,
                   help='Adapter version; v3 is an unvalidated actuator experiment, not a walking preset')
    a = p.parse_args()
    if a.out.exists(): raise SystemExit('Choose a new profile path')
    s = build(GraphStore.load(a.graph),a.version)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    write_json(a.out, s)
    print({k:s['neuromuscular'][k] for k in ('motor_neurons', 'sensory_neurons', 'covered_dofs')})
