"""BANC v888 sensory–motor loop, with an explicit engineering muscle adapter.

Only the neuron/side/body-part/muscle identities are anatomical observations.
Moment arms, receptor tuning and activation-to-position are model hypotheses.
No imposed rhythm, gait trajectory, target bearing or body position is used.
"""
import copy
import math
import numpy as np
from .integrity import digest, finite, bounded_int

SOURCE = 'https://www.nature.com/articles/s41586-026-10735-w'
FECO = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC8665017/'
FECO_CIRCUIT = 'https://elifesciences.org/reviewed-preprints/97766v1'
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
PARTS = {'f': 'front_leg', 'm': 'middle_leg', 'h': 'hind_leg'}
DOFS = ('coxa_yaw', 'coxa_pitch', 'coxa_roll', 'femur_pitch', 'femur_roll', 'tibia_pitch', 'tarsus_pitch')
# Signed relative joint targets, not measured muscle moment arms. Multi-axis
# muscles remain explicitly multi-axis. All coefficients are in the profile.
MUSCLE_VECTORS = {
    'sternal_anterior_rotator_muscle': {'coxa_yaw': 1.},
    'sternal_posterior_rotator_muscle': {'coxa_yaw': -1.},
    'tergopleural_promotor_muscle': {'coxa_pitch': 1.},
    'pleural_remotor_and_abductor_muscle': {'coxa_pitch': -1., 'coxa_roll': -1.},
    'sternal_adductor_muscle': {'coxa_roll': 1.},
    'trochanter_flexor_muscle': {'femur_pitch': -1.},
    'accessory_trochanter_flexor_muscle': {'femur_pitch': -1.},
    'trochanter_extensor_muscle': {'femur_pitch': 1.},
    'sternotrochanter_extensor_muscle': {'femur_pitch': 1.},
    'tergotrochanter_extensor_muscle': {'femur_pitch': 1.},
    'femur_reductor_muscle': {'femur_roll': -1.},
    'tibia_flexor_muscle': {'tibia_pitch': 1.},
    'accessory_tibia_flexor_muscle': {'tibia_pitch': 1.},
    'tibia_extensor_muscle': {'tibia_pitch': -1.},
    'tarsus_depressor_muscle': {'tarsus_pitch': -1.},
    'long_tendon_muscle': {'tarsus_pitch': -1.},
    'tarsus_levator_muscle': {'tarsus_pitch': 1.},
}


def joint_suffixes(leg):
    return (f'/c_thorax-{leg}_coxa-yaw', f'/c_thorax-{leg}_coxa-pitch',
            f'/c_thorax-{leg}_coxa-roll', f'/{leg}_coxa-{leg}_trochanterfemur-pitch',
            f'/{leg}_coxa-{leg}_trochanterfemur-roll',
            f'/{leg}_trochanterfemur-{leg}_tibia-pitch', f'/{leg}_tibia-{leg}_tarsus1-pitch')


def sensory_kind(node):
    a = node.get('source_annotations', {})
    if node.get('super_class') != 'sensory': return None
    sub, target = a.get('cell_sub_class') or '', a.get('peripheral_target_type')
    if node.get('class') == 'chordotonal_organ_neuron' and target == 'chordotonal_organ':
        for kind in ('claw', 'hook', 'club'):
            if f'_{kind}_' in sub: return kind
    if node.get('class') == 'campaniform_sensillum_neuron' and target == 'campaniform_sensillum': return 'load'
    if node.get('class') == 'bristle_neuron' and target == 'bristle' and a.get('cell_function') == 'tactile': return 'touch'
    return None


def build_spec(graph, version=1):
    if version not in (1, 2, 3): raise ValueError('Unsupported neuromuscular version')
    if (graph.manifest['dataset_id'], graph.manifest['snapshot_id']) != ('flywire_banc', '888'):
        raise ValueError('BANC v888 required; no cross-specimen neural IDs')
    motors, sensory, exclusions = {}, {}, []
    for n in graph.nodes:
        a = n.get('source_annotations', {})
        part = a.get('body_part_effector') if n['super_class'] == 'motor' else a.get('body_part_sensory')
        if part not in PARTS.values(): continue
        side = n['soma_side']
        if n['super_class'] == 'motor':
            muscle = a.get('peripheral_target_type')
            if side in ('left', 'right') and muscle in MUSCLE_VECTORS:
                motors.setdefault((side, part, muscle), []).append(n['id'])
            else: exclusions.append(dict(id=n['id'], reason='Motor side or muscle unresolved'))
        elif n['super_class'] == 'sensory':
            kind = sensory_kind(n)
            if side in ('left', 'right') and kind:
                sensory.setdefault((side, part, kind), []).append(n['id'])
            else: exclusions.append(dict(id=n['id'], reason='Sensory side, organ, subtype, or tuning unresolved'))
    rows = []
    for leg in LEGS:
        side, part = ('left' if leg[0] == 'l' else 'right'), PARTS[leg[1]]
        groups = [dict(muscle=m, ids=motors[(side, part, m)], vector=MUSCLE_VECTORS[m])
                  for m in MUSCLE_VECTORS if (side, part, m) in motors]
        ports = [dict(kind=k, ids=sensory[(side, part, k)])
                 for k in ('claw', 'hook', 'club', 'load', 'touch') if (side, part, k) in sensory]
        if not groups or not any(p['kind'] == 'claw' for p in ports):
            raise ValueError('Missing leg motor or FeCO position cohort: ' + leg)
        rows.append(dict(leg=leg, side=side, body_part=part, muscles=groups, sensory=ports))
    covered = sum(len({d for g in row['muscles'] for d in g['vector']}) for row in rows)
    spec = dict(schema='flylab.neuromuscular.v1', graph_hash=graph.hash, rows=rows,
                sensory_gain_mV=18., sensory_cap_mV=24., sensory_tau_s=.02,
                sensory_delay_controls=1, motor_gain_rad_per_Hz=.003,
                activation_tau_s=.03, maximum_offset_rad=.5,
                adhesion_policy='neural-tarsus-balance-v1', covered_dofs=covered, total_leg_dofs=42,
                motor_neurons=sum(len(g['ids']) for r in rows for g in r['muscles']),
                sensory_neurons=sum(len(p['ids']) for r in rows for p in r['sensory']),
                excluded=exclusions, evidence=[SOURCE, FECO], biological_validation=False,
                tuning='Claw: normalized tibia angle; hook/club: speed magnitude. Subtype polarity, preferred angle and vibration response are unresolved; these are population proxies.',
                muscle_model='Signed rate-to-position map with fixed position actuators. No measured force-length curves, moment arms, motor-unit recruitment or reconstructed muscle tissue.',
                imposed_cpg=False)
    if version >= 2:
        for row in rows:
            ports=[]
            for p in row['sensory']:
                groups={}
                for index in graph.resolve(p['ids'], maximum=10000):
                    node=graph.nodes[index]
                    groups.setdefault(node['cell_type'] or 'untyped',[]).append(node['id'])
                for cell_type, ids in sorted(groups.items()):
                    ports.append(dict(kind=p['kind'],cell_type=cell_type,ids=ids))
            row['sensory']=ports
        spec.update(schema='flylab.neuromuscular.v2',
            adhesion_policy='neural-tarsus-active-v2',adhesion_threshold_Hz=1.,
            load_half_bw=.5,touch_half_bw=.05,velocity_filter_tau_s=.03,
            motor_response='bounded-tanh-position-v2',
            receptor_model='typed-negative-feedback-hypothesis-v2',
            tuning=('Claw SNpp50/51 and hook SNpp41/39 receive opposing features. Polarity is a negative-feedback '
                    'hypothesis inferred from MANC downstream connectivity, transferred by cell-type name to BANC, '
                    'not measured BANC tuning. Other claw/hook types get zero external drive until resolved. '
                    'Club gets high-pass knee velocity (not a resolved vibration-frequency model). '
                    'Load uses flat-floor normal force minus commanded pad adhesion; tactile bristles get '
                    'non-floor contact only, not every foot support. Per-bristle receptive fields remain unresolved.'),
            evidence=[SOURCE,FECO,FECO_CIRCUIT])
    if version == 3:
        for row in rows:
            for muscle in row['muscles']:
                muscle['units'] = [dict(id=graph.nodes[i]['id'], cell_type=graph.nodes[i]['cell_type'])
                                   for i in graph.resolve(muscle['ids'])]
        spec.update(schema='flylab.neuromuscular.v3',
            motor_pooling='sum-saturating-units-v3', rate_half_Hz=100.,
            adhesion_policy='neural-tarsus-lift-release-v3', lift_release_mm=.02,
            muscle_model=('Additive bounded per-neuron rate-to-position proxy; equal unit capacities are an '
                          'engineering hypothesis, not measured slow/fast muscle forces. Cell types remain '
                          'individually inspectable. Position actuators and v2 signed vectors are retained.'),
            adhesion_model=('Neural tarsal balance requests adhesion. A local joint-Jacobian estimate vetoes '
                            'attachment if the neural joint target lifts the pad along the flat-floor normal. '
                            'This is an explicit peripheral actuator rule, without imposed gait timing.'),
            adoption_status='EXPERIMENTAL_NOT_VALIDATED')
    return spec


def motor_unit_response(rates, rate_half_Hz):
    """Additive equal-capacity unit proxy, in equivalent Hz, not muscle force.

    Saturating each unit before summing avoids both population-size dilution
    and a single high-rate unit dominating without bound. Unknown cell types
    are not relabeled as slow, intermediate or fast units.
    """
    return rate_half_Hz * (rates / (rate_half_Hz + rates))


def typed_receptor_feature(kind, cell_type, angle, velocity, velocity_lowpass, load, touch, spec):
    """Declared v2 receptor hypothesis. Positive tibia q means flexion."""
    position=float(np.clip(angle/math.pi,0.,1.))
    if kind=='claw':
        return (position if cell_type=='SNpp50' else 1-position if cell_type=='SNpp51' else 0.)
    if kind=='hook':
        direction={'SNpp41':1.,'SNpp39':-1.}.get(cell_type,0.)
        return min(1.,max(0.,velocity*direction)/20.)
    if kind=='club':return min(1.,abs(velocity-velocity_lowpass)/50.)
    if kind=='load':return load/(load+spec['load_half_bw'])
    if kind=='touch':return touch/(touch+spec['touch_half_bw'])
    raise ValueError('Unsupported receptor kind')


class NeuromuscularLoop:
    def __init__(self, graph, specification, body):
        self.spec = copy.deepcopy(specification)
        s = self.spec
        self.version={'flylab.neuromuscular.v1':1,'flylab.neuromuscular.v2':2,
                      'flylab.neuromuscular.v3':3}.get(s.get('schema'))
        if self.version is None or s.get('graph_hash') != graph.hash:
            raise ValueError('Neuromuscular graph identity mismatch')
        # Verify all anatomical memberships independently of the supplied IDs.
        expected = build_spec(graph,self.version)
        if s.get('rows') != expected['rows']:
            raise ValueError('Neuromuscular identity, side, muscle, or model vectors differ')
        for key in ('covered_dofs', 'total_leg_dofs', 'motor_neurons', 'sensory_neurons', 'excluded'):
            if s.get(key) != expected[key]: raise ValueError('Incorrect anatomical coverage: ' + key)
        if s.get('biological_validation') is not False:
            raise ValueError('This experimental muscle map is not biologically validated')
        for key, lo, hi in (('sensory_gain_mV', 0., 100.), ('sensory_cap_mV', .001, 100.),
                            ('sensory_tau_s', .001, 1.), ('motor_gain_rad_per_Hz', 0., .02),
                            ('activation_tau_s', .001, 1.), ('maximum_offset_rad', .001, 1.)):
            finite(s.get(key), key, lo, hi)
        policies = ({'neural-tarsus-active-v2', 'neural-tarsus-lift-release-v3'}
                    if self.version == 3 else {expected['adhesion_policy']})
        if s.get('sensory_delay_controls') != 1 or s.get('adhesion_policy') not in policies:
            raise ValueError('Unsupported sensory latency or adhesion policy')
        if self.version>=2:
            for key,lo,hi in (('adhesion_threshold_Hz',0.,100.),('load_half_bw',.01,10.),
                               ('touch_half_bw',.001,10.),('velocity_filter_tau_s',.001,1.)):
                finite(s.get(key),key,lo,hi)
            for key in ('motor_response','receptor_model'):
                if s.get(key)!=expected[key]:raise ValueError('Unsupported '+key)
        if self.version == 3:
            finite(s.get('rate_half_Hz'), 'unit half-rate', 1., 1000.)
            finite(s.get('lift_release_mm'), 'pad release lift', .001, 1.)
            if s.get('motor_pooling') not in ('mean-linear-v2', 'sum-saturating-units-v3'):
                raise ValueError('Unsupported motor unit pooling')
            if s.get('adoption_status') != 'EXPERIMENTAL_NOT_VALIDATED':
                raise ValueError('This candidate is not validated for adoption')
            if not callable(getattr(body, 'foot_kinematics', None)):
                raise ValueError('v3 requires physical foot kinematics')
        if s.get('imposed_cpg') is not False: raise ValueError('No imposed CPG allowed in this adapter')
        self.graph, self.body, self.hash = graph, body, digest(s)
        self.joints, self.muscles, self.ports = [], [], []
        self.port_types=[]
        for row in s['rows']:
            indices = []
            for suffix in joint_suffixes(row['leg']):
                matches = [i for i, name in enumerate(body.joint_names) if name.endswith(suffix)]
                if len(matches) != 1: raise ValueError('Physical joint mapping not unique: ' + suffix)
                indices.append(matches[0])
            self.joints.append(indices)
            self.muscles.append([(g['muscle'], graph.resolve(g['ids']),
                                  np.array([g['vector'].get(d, 0.) for d in DOFS])) for g in row['muscles']])
            for p in row['sensory']:
                self.ports.append((row['leg'], p['kind'], graph.resolve(p['ids'], maximum=10000)))
                self.port_types.append(p.get('cell_type'))
        self.joints = np.array(self.joints, np.int32)
        if len(np.unique(self.joints)) != 42: raise ValueError('42 unique physical DOFs required')
        self.motor_indices = np.array(sorted({int(i) for groups in self.muscles for _, ids, _ in groups for i in ids}), np.int32)
        self.tendon_adapter = None
        self.tendon_inputs = {}
        if getattr(body, 'tendon_control', None) is not None:
            from .tendon_neural import NeuralTendonAdapter
            self.tendon_adapter = NeuralTendonAdapter(graph, s, body)
            # LTM drives the distal tendon instead of the first-tarsal-joint
            # position proxy. Depressor/levator continue to control that joint;
            # LTM's existing explicit adhesion request is retained.
            for leg, groups in zip(LEGS, self.muscles):
                if leg in self.tendon_adapter.rerouted_legs:
                    for name, _, vector in groups:
                        if name == 'long_tendon_muscle': vector[:] = 0.
            self.motor_indices = np.union1d(self.motor_indices, self.tendon_adapter.motor_indices)
            self.hash = digest(dict(leg_adapter=s, tendon_adapter=self.tendon_adapter.spec,
                                    body_model_hash=getattr(body, 'model_hash', None)))
        from .actuation_map import describe_actuation
        self._actuation_map = describe_actuation(self)
        self.channels = {'leg_feedback'} | {f'leg_{leg}' for leg in LEGS} | {f'leg_{leg}_{kind}' for leg, kind, _ in self.ports}
        self.port_names=[f'leg_{leg}_{kind}'+('_'+self.port_types[i] if self.version>=2 else '')
                         for i,(leg,kind,_) in enumerate(self.ports)]
        self.channels.update(self.port_names)
        self._port_legs = [LEGS.index(leg) for leg, _, _ in self.ports]
        self._port_channels = [frozenset(('*', 'leg_feedback', f'leg_{leg}',
                                         f'leg_{leg}_{kind}', self.port_names[j]))
                               for j, (leg, kind, _) in enumerate(self.ports)]
        self.filtered = np.zeros(len(self.ports))
        self.delayed = np.zeros(len(self.ports))
        self.offset = np.zeros(42)
        self.sensor_tick = self.motor_tick = 0
        self.last_sensory, self.last_motor = [], []
        self.velocity_lowpass=np.zeros(42)
        self.last_contact={}

    def encode(self, disabled=(), dt=.005):
        finite(dt, 'feedback dt', .0001, .05)
        observation = self.body.leg_observation()
        q, v, load = [np.asarray(observation[k], dtype=float) for k in ('angles_rad', 'velocities_rad_s', 'load_bw')]
        if q.shape != (42,) or v.shape != (42,) or load.shape != (6,) or not all(np.isfinite(x).all() for x in (q, v, load)) or np.any(load < 0):
            raise ValueError('Invalid physical leg observation')
        if self.version>=2:
            contact={key:np.asarray(observation.get(key),float) for key in
                     ('load_bw','support_load_bw','non_support_load_bw','adhesion_force_bw','floor_normal_load_bw')}
            if any(x.shape!=(6,) or not np.isfinite(x).all() or np.any(x<0) for x in contact.values()):
                raise ValueError('v2 requires compensated support and non-floor contact observations')
            support,touch=contact['support_load_bw'],contact['non_support_load_bw']
        features = {}
        for i, leg in enumerate(LEGS):
            knee = self.joints[i, 5]
            features[leg] = dict(claw=float(np.clip(q[knee]/math.pi, 0., 1.)),
                                 hook=min(1., abs(float(v[knee]))/20.),
                                 club=min(1., abs(float(v[knee]))/50.),
                                 load=min(1., float(load[i])/.5), touch=float(load[i] > .01))
        port_features=[]
        for j,(leg,kind,_) in enumerate(self.ports):
            i=self._port_legs[j];knee=self.joints[i,5]
            value=features[leg][kind] if self.version==1 else typed_receptor_feature(
                kind,self.port_types[j],q[knee],v[knee],self.velocity_lowpass[knee],support[i],touch[i],self.spec)
            port_features.append(value)
        values=np.minimum(self.spec['sensory_cap_mV'],self.spec['sensory_gain_mV']*np.asarray(port_features))
        filtered = self.filtered + (1-math.exp(-dt/self.spec['sensory_tau_s']))*(values-self.filtered)
        drive = np.zeros(self.graph.n, np.float32)
        diagnostics = []
        disabled = set(disabled)
        for j, (leg, kind, ids) in enumerate(self.ports):
            name = self.port_names[j]
            enabled = self._port_channels[j].isdisjoint(disabled)
            value = float(self.delayed[j]) if enabled else 0.
            np.add.at(drive, ids, value)
            diagnostics.append(dict(name=name, feature=port_features[j], value=value, unit='mV', enabled=enabled, targets=len(ids)))
            if self.version>=2:
                diagnostics[-1].update(cell_type=self.port_types[j],receptor_kind=kind,
                    polarity_unresolved=kind in ('claw','hook') and self.port_types[j] not in ('SNpp50','SNpp51','SNpp39','SNpp41'))
        self.filtered[:], self.delayed[:] = filtered, filtered
        self.sensor_tick += 1
        self.last_sensory = diagnostics
        if self.version>=2:
            self.velocity_lowpass+=(1-math.exp(-dt/self.spec['velocity_filter_tau_s']))*(v-self.velocity_lowpass)
            self.last_contact={key:value.tolist() for key,value in contact.items()}
        return drive, diagnostics

    def decode(self, neural, disconnected=False, dt=.005):
        finite(dt, 'muscle dt', .0001, .05)
        rates = neural.readout(self.motor_indices)['rate_Hz']
        if rates.shape != self.motor_indices.shape or not np.isfinite(rates).all() or np.any(rates < 0):
            raise ValueError('Invalid muscle neural rates')
        table = dict(zip(self.motor_indices, rates))
        requested = np.zeros(42)
        leg_outputs = {}
        diagnostics, adhesion = [], np.ones(6, bool)
        for i, groups in enumerate(self.muscles):
            vector = np.zeros(7)
            values, outputs, units = {}, {}, []
            for name, ids, moment in groups:
                unit_rates = np.array([table[int(j)] for j in ids])
                rate = float(np.mean(unit_rates))
                values[name] = rate
                output = rate
                if self.version == 3:
                    contributions = (motor_unit_response(unit_rates, self.spec['rate_half_Hz'])
                                     if self.spec['motor_pooling']=='sum-saturating-units-v3'
                                     else unit_rates / len(ids))
                    if self.spec['motor_pooling'] == 'sum-saturating-units-v3':
                        output = float(contributions.sum())
                    units.extend(dict(id=self.graph.nodes[int(j)]['id'],
                                      cell_type=self.graph.nodes[int(j)]['cell_type'], muscle=name,
                                      rate_Hz=float(r), equivalent_output_Hz=float(c))
                                 for j, r, c in zip(ids, unit_rates, contributions))
                outputs[name] = output
                vector += output * moment * self.spec['motor_gain_rad_per_Hz']
            leg_outputs[LEGS[i]] = outputs
            limit=self.spec['maximum_offset_rad']
            requested[self.joints[i]] = (np.clip(vector,-limit,limit) if self.version==1 else limit*np.tanh(vector/limit))
            # A pad is released by the relative tarsal elevator drive. This is
            # an explicit actuator assumption; it does not create gait timing.
            close = outputs.get('tarsus_depressor_muscle', 0.) + outputs.get('long_tendon_muscle', 0.)
            opened=outputs.get('tarsus_levator_muscle',0.)
            adhesion[i] = (opened<=close+1. if self.version==1 else close>opened+self.spec['adhesion_threshold_Hz'])
            diagnostics.append(dict(leg=LEGS[i], muscle_rates_Hz=values, requested_offset_rad=requested[self.joints[i]].tolist()))
            if self.version>=2:
                diagnostics[-1].update(raw_offset_rad=vector.tolist(),bounded_axes=(np.abs(vector)>limit).tolist(),
                                       adhesion_requested=bool(adhesion[i]))
            if self.version == 3:
                diagnostics[-1].update(motor_units=units, muscle_output_equivalent_Hz=outputs,
                                       motor_disconnected=bool(disconnected))
        if disconnected:
            # Immediate zero neural contribution, retaining neutral support.
            next_offset = np.zeros_like(self.offset)
            adhesion[:] = self.version==1
        else:
            next_offset = self.offset + (1-math.exp(-dt/self.spec['activation_tau_s']))*(requested-self.offset)
        targets = self.body.neutral + next_offset
        if self.version == 3:
            feet = self.body.foot_kinematics(targets)
            lift = np.asarray(feet['target_lift_mm'])
            velocity = np.asarray(feet['velocity_mm_s'])
            if (lift.shape != (6,) or velocity.shape != (6, 3) or
                    not np.isfinite(lift).all() or not np.isfinite(velocity).all()):
                raise ValueError('Invalid pad kinematics')
            release = ((lift > self.spec['lift_release_mm']) & adhesion &
                       (self.spec['adhesion_policy'] == 'neural-tarsus-lift-release-v3'))
            adhesion[release] = False
            for i, row in enumerate(diagnostics):
                row.update(adhesion_applied=bool(adhesion[i]), lift_release=bool(release[i]),
                           target_lift_mm=float(lift[i]), foot_velocity_mm_s=velocity[i].tolist())
        self.offset[:] = next_offset
        if self.tendon_adapter:
            self.tendon_inputs = self.tendon_adapter.decode(table, leg_outputs, disconnected, dt)
        self.motor_tick += 1
        self.last_motor = diagnostics
        return targets, adhesion

    def summary(self):
        result = dict(enabled=True, hash=self.hash, covered_dofs=self._actuation_map['mapped_axes'], total_leg_dofs=42,
                    motor_neurons=len(self.motor_indices), sensory_neurons=self.spec['sensory_neurons'],
                    sensor_tick=self.sensor_tick, motor_tick=self.motor_tick, imposed_cpg=False,
                    biological_validation=False, sensory=copy.deepcopy(self.last_sensory),
                    muscles=copy.deepcopy(self.last_motor), offset_rad=self.offset.tolist(),
                    unbound_dofs=[f"{leg}_{d}" for leg, groups in zip(LEGS, self.muscles) for axis,d in enumerate(DOFS)
                                  if not any(vector[axis] != 0 for _,_,vector in groups)],
                    tuning=self.spec['tuning'], muscle_model=self.spec['muscle_model'])
        result['actuation'] = copy.deepcopy(self._actuation_map)
        if self.tendon_adapter:
            result['tendons'] = self.tendon_adapter.summary()
        if self.version>=2:
            result.update(schema=self.spec['schema'],contact=copy.deepcopy(self.last_contact),
                unresolved_polarity_targets=sum(len(ids) for j,(_,kind,ids) in enumerate(self.ports)
                    if kind in ('claw','hook') and self.port_types[j] not in ('SNpp50','SNpp51','SNpp39','SNpp41')),
                adhesion_policy=self.spec['adhesion_policy'],motor_response=self.spec['motor_response'])
        if self.version == 3:
            result.update(motor_pooling=self.spec['motor_pooling'],
                          adoption_status=self.spec['adoption_status'],
                          adhesion_model=self.spec['adhesion_model'])
        return result

    def snapshot(self, *, copy_diagnostics=True):
        # encode/decode replace diagnostic lists and contact dictionaries; they
        # never edit an earlier diagnostic record. A control-boundary rollback
        # can retain those records, while exported checkpoints own deep copies.
        clone = copy.deepcopy if copy_diagnostics else lambda value: value
        result = dict(hash=self.hash, sensor_tick=self.sensor_tick, motor_tick=self.motor_tick,
                    filtered=self.filtered.copy(), delayed=self.delayed.copy(), offset=self.offset.copy(),
                    last_sensory=clone(self.last_sensory), last_motor=clone(self.last_motor))
        if self.version>=2:result.update(velocity_lowpass=self.velocity_lowpass.copy(),last_contact=clone(self.last_contact))
        if self.tendon_adapter:
            result['tendons'] = self.tendon_adapter.snapshot()
        return result

    def restore(self, state):
        if state.get('hash') != self.hash: raise ValueError('Neuromuscular checkpoint identity mismatch')
        if self.tendon_adapter:
            self.tendon_adapter.validate_state(state.get('tendons'))
        elif state.get('tendons') is not None:
            raise ValueError('Checkpoint requires neural tendons')
        st = bounded_int(state.get('sensor_tick'), 'leg sensory tick')
        mt = bounded_int(state.get('motor_tick'), 'leg motor tick')
        arrays = {}
        for key, low, high in (('filtered', 0., self.spec['sensory_cap_mV']),
                               ('delayed', 0., self.spec['sensory_cap_mV']),
                               ('offset', -self.spec['maximum_offset_rad'], self.spec['maximum_offset_rad'])):
            a = np.asarray(state.get(key))
            if a.dtype != getattr(self, key).dtype or a.shape != getattr(self, key).shape or not np.isfinite(a).all() or np.any((a < low) | (a > high)):
                raise ValueError('Neuromuscular state out of range: ' + key)
            arrays[key] = a.copy()
        if self.version>=2:
            a=np.asarray(state.get('velocity_lowpass'))
            if a.dtype!=self.velocity_lowpass.dtype or a.shape!=(42,) or not np.isfinite(a).all():
                raise ValueError('Invalid receptor velocity history')
            contact=copy.deepcopy(state.get('last_contact'))
            allowed={'load_bw','support_load_bw','non_support_load_bw','adhesion_force_bw','floor_normal_load_bw'}
            if not isinstance(contact,dict) or set(contact)!=(allowed if st else set()):
                raise ValueError('Invalid receptor contact history')
            for value in contact.values():
                acontact=np.asarray(value)
                if acontact.shape!=(6,) or not np.isfinite(acontact).all() or np.any(acontact<0):raise ValueError('Invalid receptor contact values')
            arrays['velocity_lowpass']=a.copy()
        for key, a in arrays.items(): getattr(self, key)[:] = a
        self.sensor_tick, self.motor_tick = st, mt
        self.last_sensory = copy.deepcopy(state.get('last_sensory', []))
        self.last_motor = copy.deepcopy(state.get('last_motor', []))
        if self.version>=2:self.last_contact=contact
        if self.tendon_adapter:
            self.tendon_adapter.restore(state['tendons'])
            self.tendon_inputs = {name: float(self.tendon_adapter.activation[i])
                for i, name in enumerate(self.tendon_adapter.control.names)
                if self.tendon_adapter.modes[name] == 'neural'}
