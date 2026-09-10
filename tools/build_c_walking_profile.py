#!/usr/bin/env python3
"""Separate, source-bound walking readout hypothesis. No tonic motor drive."""
import argparse
import copy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.ports import PortBindings
from flylab.c.integrity import read_json, write_json

SOURCE = 'https://www.nature.com/articles/s41586-024-07854-7'
CPG = 'https://github.com/smpuglie/Pugliese_2026/tree/5626b7312ffbbe768793a148cec82f93d7530fce'
FAFB_FORWARD = {
    'DNg100': {'720575940640978048', '720575940647228468'},
    'DNg97': {'720575940620300308', '720575940626730883'},
}


def build(graph, base, reset_current=False, poisson=False, voltage_events=False):
    if voltage_events:
        spec = build(graph, base, poisson=True)
        spec.update(profile='fafb783-walking-voltage-events-v4', profile_version=4,
                    neural_parameters=dict(integration='exact-exponential-voltage-events-v1'),
                    neural_model_note='Reference-inspired voltage events, no refractoriness on Poisson targets, reset synaptic current after spikes and freeze it during refractory intervals. Exact held-input integration and tick scheduling remain this simulator conventions, not bitwise Brian2 reproduction.')
        for port, original in zip(spec['sensory'], base['sensory']):
            port['uncertainty'] = original['uncertainty'] + ' Poisson voltage-event input, 150 Hz/unit and 68.75 mV/event; receptor response is uncalibrated. Uses explicit voltage-events model, not synaptic-current pulses.'
        PortBindings(graph, spec)
        return spec
    if poisson:
        from tools.build_c_poisson_profile import build as poisson_profile
        spec = poisson_profile(graph, build(graph, base, reset_current=True))
        spec.update(profile='fafb783-walking-poisson-reset-current-v3', profile_version=3)
        spec['motor']['forward']['gain'] = .08
        PortBindings(graph, spec)
        return spec
    original = PortBindings(graph, base)
    if graph.manifest['dataset_id'] != 'flywire_fafb' or graph.manifest['snapshot_id'] != '783':
        raise ValueError('Pinned FAFB v783 profile required')
    selected = []
    for cell_type, expected in FAFB_FORWARD.items():
        rows = [n for n in graph.nodes if n['cell_type'] == cell_type]
        if {n['root_id'] for n in rows} != expected or any(n['super_class'] != 'descending' for n in rows):
            raise ValueError('Walking cell identity differs from reviewed snapshot')
        selected.extend(n['id'] for n in rows)
    spec = copy.deepcopy(base)
    spec.update(profile='fafb783-walking-population-v1', profile_version=1,
                parent_binding_hash=original.hash, biological_validation=False,
                motor_decoder=dict(kind='bounded-opponent-v1', speed_limit=1., yaw_limit=1.5, stop_half_drive=1.),
                no_tonic_drive=True)
    spec['motor']['forward'].update(ids=sorted(selected), gain=.008, evidence=[SOURCE, CPG],
        uncertainty='DNg100/BDN2 and DNg97/oDN1 walking population. Mean-rate gain and speed ceiling are engineering choices, not measured speed calibration.')
    for role in ('yaw_left', 'yaw_right'):
        spec['motor'][role]['gain'] = .004
        spec['motor'][role]['uncertainty'] += ' Reduced engineering gain limits excessive rate-to-yaw saturation; no target bearing is used.'
    spec['motor']['backward']['gain'] = .015
    if reset_current:
        spec.update(profile='fafb783-walking-reset-current-v2', profile_version=2,
                    neural_parameters=dict(integration='exact-exponential-reset-current-v1'),
                    neural_model_note='Explicit reset-current hypothesis: synaptic current is cleared on a spike. Other LIF integration and refractory rules are unchanged; not a reproduction of the Shiu/Brian2 model.')
    # Halting is expected through the network's walking inhibition. No invented
    # descending stop ID: an empty stop pool remains explicitly unbound.
    PortBindings(graph, spec)
    return spec


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph', default='data/fafb783/bundle')
    p.add_argument('--base', default='data/fafb783/bindings-bilateral-geosmin-v2.json')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reset-current', action='store_true')
    p.add_argument('--poisson', action='store_true', help='Poisson-current sensory input with explicit reset-current dynamics')
    p.add_argument('--voltage-events', action='store_true', help='Reference-inspired Poisson voltage events with explicit refractory semantics')
    a = p.parse_args()
    if a.out.exists(): raise SystemExit('Choose a new profile path')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    write_json(a.out, build(GraphStore.load(a.graph), read_json(a.base), a.reset_current, a.poisson, a.voltage_events))
