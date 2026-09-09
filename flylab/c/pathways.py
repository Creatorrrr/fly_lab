"""Anatomical reachability and measured propagation are distinct reports."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from .neural import create_backend
from .ports import SensoryEncoder, MotorDecoder


def reachability(graph, bindings):
    result = {}
    for kind, mask in (('anatomical', np.ones(len(graph.weights), dtype=bool)),
                       ('effective', graph.weights != 0)):
        edges = csr_matrix((mask.astype(np.float32), graph.indices.copy(), graph.indptr.copy()), shape=(graph.n, graph.n))
        edges.eliminate_zeros()
        forward = edges.T.tocsr()
        ports = {}
        for port, indices in bindings.sensory:
            distance = dijkstra(forward, directed=True, indices=indices, min_only=True, unweighted=True)
            ports[port['name']] = dict(reachable_nodes=int(np.isfinite(distance).sum()),
                motor={role:[dict(id=graph.nodes[int(i)]['id'], hops=int(distance[i]) if np.isfinite(distance[i]) else None)
                             for i in ids] for role, (_, ids) in bindings.motor.items()})
        result[kind] = ports
    return dict(graph_hash=graph.hash, binding_hash=bindings.hash, paths=result,
                interpretation='Static reachability does not imply spiking, signed excitation, or successful behavior')


def packet(odor=(0.,0.), danger=0.):
    return dict(schema='flylab.sensors.v2', odor=list(odor), danger=danger, panorama=[0.]*64,
                nearRanges=[10.]*9, contact=0, angularVelocity=0., forwardSpeed=0.,
                clearanceDown=.7, clearanceUp=16., odorChange=0.)


def propagate(graph, bindings, case, *, backend='exp_lif_cpu_reference', seconds=.2, seed=42):
    controls=round(seconds/.005)
    if not 1<=controls<=24000 or abs(controls*.005-seconds)>1e-9: raise ValueError('Integral 5 ms duration up to 120 s required')
    neural=create_backend(graph,backend=backend); encoder=SensoryEncoder(bindings,seed); decoder=MotorDecoder(bindings)
    capture=np.asarray(sorted({int(i) for _, ids in bindings.sensory for i in ids}|set(bindings.motor_indices)),dtype=np.int32)
    if len(capture)>512: raise ValueError('Choose a diagnostic cohort <=512 cells')
    rows=[]
    for _ in range(controls):
        observation=packet(case.get('odor',(0.,0.)),case.get('danger',0.))
        drive,pulses,ports=encoder.encode(observation,.005,neural.p.dt,case.get('disabled',()),
                    case.get('features',dict(loom_left=0.,loom_right=0.,head_contact=0.)))
        if role:=case.get('direct_motor'):
            drive[bindings.motor[role][1]]+=case.get('amplitude_mV',20.)
        neural.advance(drive,round(.005/neural.p.dt),capture,pulses)
        command,rates=decoder.decode(neural)
        readout=neural.readout(capture)
        rows.append(dict(tick=neural.tick,ports=ports,command=command,motor_rates_Hz=rates,
                         decoder=decoder.diagnostics,spikes=neural.last_events,
                         **{k:v.tolist() for k,v in readout.items()}))
    snapshot=neural.snapshot()
    return dict(name=case['name'],case=case,seconds=seconds,backend=neural.backend,
                graph_hash=graph.hash,binding_hash=bindings.hash,seed=seed,
                simulated_nodes=graph.n,active_nodes=int(np.count_nonzero(snapshot['spike_count'])),
                cumulative_spikes=int(snapshot['spike_count'].sum()),
                capture_ids=[graph.nodes[int(i)]['id'] for i in capture],trace=rows,
                physicalExecuted=False,behavior_validation='NOT_EVALUATED')


def diagnostic_cases():
    cases=[dict(name='none'),dict(name='symmetric',odor=[.5,.5]),dict(name='odor_off',odor=[.5,.5],disabled=['*'])]
    for intensity in (.1,.5,.95):
        for side in (0,1):
            odor=[0.,0.];odor[side]=intensity
            cases.append(dict(name=f'odor_{side}_{intensity}',odor=odor))
        cases.append(dict(name=f'geosmin_{intensity}',danger=intensity))
    cases.extend(dict(name='direct_'+role,direct_motor=role) for role in ('forward','backward','yaw_left','yaw_right'))
    return cases
