"""Small synthetic circuit, explicitly test-only and never a production default."""
import numpy as np
from flylab.c.graph import GraphStore, external_id
from flylab.c.ports import PortBindings


def graph_fixture(pre=(0, 1, 2), post=(1, 2, 1), counts=(4, 5, 2), n=6):
    nodes = [dict(id=external_id(str(100+i)), root_id=str(100+i), cell_type='TEST_ONLY',
                  nt_type='GABA' if i == 2 else 'ACH', soma_side='unknown',
                  super_class='sensory' if i == 0 else 'descending', flow='afferent' if i == 0 else 'efferent',
                  regions=['TEST_ROI']) for i in range(n)]
    return GraphStore.from_edges(nodes, np.asarray(pre, dtype=np.int32), np.asarray(post, dtype=np.int32),
                                 np.asarray(counts, dtype=np.int64), metadata={'scope': 'fixture'})


def bindings_fixture(graph, method='drive_mV'):
    names = [n['id'] for n in graph.nodes]
    review = dict(review_status='engineering_reviewed', evidence=['SYNTHETIC TEST ONLY'], uncertainty='Synthetic test mapping, no biology.')
    sensory = [dict(name='test_odor', channel='odor_mean', ids=[names[0]], input_kind='sensory',
                    method=method, gain=0., baseline=0., cap=100., tau_s=0., delay_controls=0,
                    offset=0., scale=1., pulse_mV=2., **review)]
    motor = {}
    for role, index, gain in [('forward', 1, .08), ('backward', 2, .08), ('stop', 3, .08),
                              ('yaw_left', 4, .08), ('yaw_right', 5, .08)]:
        motor[role] = dict(ids=[names[index]], gain=gain, output_side=role[4:] if role.startswith('yaw_') else None, **review)
    return PortBindings(graph, dict(schema='flylab.bindings.v3', graph_hash=graph.hash, sensory=sensory, motor=motor))
