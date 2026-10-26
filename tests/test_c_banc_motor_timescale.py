"""Generated six-cell E138 checks; use ``-k cpu`` before parent CUDA review."""

from __future__ import annotations

import copy
import json
import math
import time

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix

from flylab.c.adaptive_rate import AdaptiveRateNetwork
from flylab.c.research_rate import ResearchRateNetwork
from flylab.c.storage import StateStore
from tools import verify_c_banc_m3_motor_timescale as runner
from tools.banc_m3_core_adaptation import CoreAdaptationNetwork
from tools.banc_m3_motor_adaptation import MotorAdaptationNetwork, MotorAdaptationTrial
from tools.banc_m3_motor_timescale import (
    LOG_TAU_LOWER,
    LOG_TAU_UPPER,
    MotorTimescaleNetwork,
    MotorTimescaleTrial,
    _timescale_source,
    canonical_timescales,
    checked_motor_log_tau,
    map_muscle_log_tau,
)


@pytest.fixture(autouse=True)
def meter(monkeypatch, request):
    counts = {'test': request.node.name, 'model_attempts': 0,
        'completed_models': 0, 'cuda_setup_steps': 0, 'logical_steps': 0,
        'logical_neural_s': 0., 'dop853_rhs_calls': 0, 'dop853_wall_s': 0.}
    native_init = AdaptiveRateNetwork.__init__
    native_advance = ResearchRateNetwork._advance

    def counted_init(self, *args, **kwargs):
        counts['model_attempts'] += 1
        native_init(self, *args, **kwargs)
        counts['completed_models'] += 1
        if self.device == 'cuda':
            counts['cuda_setup_steps'] += 3*self.interval_steps

    def counted_advance(self, drive, steps, **kwargs):
        before = self.tick
        try:
            return native_advance(self, drive, steps, **kwargs)
        finally:
            delta = self.tick-before
            counts['logical_steps'] += delta
            counts['logical_neural_s'] += delta*self.dt

    monkeypatch.setattr(AdaptiveRateNetwork, '__init__', counted_init)
    monkeypatch.setattr(ResearchRateNetwork, '_advance', counted_advance)
    began = time.perf_counter()
    yield counts
    counts['wall_s'] = time.perf_counter()-began
    print('E138_GENERATED_COST '+json.dumps(counts, sort_keys=True))


def generated_model(device, ell, implementation='packed'):
    weights = csr_matrix(np.array([
        [0., .15, 0., -.10, 0., 0.],
        [.08, 0., .06, 0., 0., 0.],
        [0., -.10, 0., .12, 0., 0.],
        [0., 0., .09, 0., -.08, 0.],
        [0., 0., 0., .11, 0., 0.],
        [0., 0., 0., 0., .07, 0.],
    ], dtype=np.float32))
    parameters = (np.array([.025, .03, .028, .024, .027, .026], np.float32),
        np.ones(6, np.float32), np.full(6, 1.5, np.float32),
        np.full(6, 40., np.float32))
    common = {'adaptation_gain': 8., 'adaptation_tau_s': .3, 'dt': .00025,
        'device': device, 'capture_steps': 20, 'cuda_implementation': implementation,
        'core_ids': np.array([1, 3, 4], np.int64), 'core_gain': 0.,
        'motor_ids': np.array([3, 4], np.int64),
        'motor_gains': np.array([1.1, 2.3], np.float32)}
    old = MotorAdaptationNetwork(weights, *parameters, **common)
    new = MotorTimescaleNetwork(weights, *parameters, **common,
                               motor_log_tau=np.asarray(ell, np.float64))
    saved = old.snapshot()
    saved['rate'] = np.array([4., 3., 2., 5., 1., 6.], np.float32)
    saved['adaptation'] = np.array([.2, .1, .3, .4, .15, .25], np.float32)
    saved['drive'] = np.array([2.8, 2.5, 2.7, 3.1, 2.9, 3.0], np.float32)
    saved['tick'] = 17
    old.restore(saved)
    patched = copy.deepcopy(saved)
    patched['model_hash'] = new.identity
    patched['motor_timescale'] = new.snapshot()['motor_timescale']
    new.restore(patched)
    return old, new, weights, parameters


def test_cpu_neutral_exact_full_state_and_pointers(meter):
    old, new, _, _ = generated_model('cpu', [0., 0.])
    assert old.identity == new.source_motor_identity
    assert old.identity != new.identity
    ptrs = (new._cell_adaptation_tau.data_ptr(), new._cell_inverse_tau.data_ptr())
    for model in (old, new):
        model.set_muted([0, 3])
    drive = old.drive.cpu().numpy().copy()
    for control in range(3):
        if control == 1:
            drive[4] += .21
        old.advance(drive, steps=20)
        new.advance(drive, steps=20)
        for key in ('rate', 'adaptation', 'drive', 'output_mask', 'tick'):
            left, right = old.snapshot()[key], new.snapshot()[key]
            if isinstance(left, np.ndarray):
                assert left.tobytes() == right.tobytes()
            else:
                assert left == right
    assert ptrs == (new._cell_adaptation_tau.data_ptr(),
                    new._cell_inverse_tau.data_ptr())
    assert meter['logical_steps'] == 120
    assert meter['cuda_setup_steps'] == 0


def test_cpu_heterogeneous_dop853_cut_drive_and_disk_future(tmp_path, meter):
    old, new, weights, parameters = generated_model('cpu', [math.log(.5),
                                                             math.log(3.)])
    new.set_muted([3])
    old.set_muted([3])
    drive = new.drive.cpu().numpy().copy()
    rate0 = new.rate.cpu().numpy().copy()
    z0 = new.adaptation.cpu().numpy().copy()
    mask = new.output_mask.cpu().numpy().copy()
    gain = new._cell_gain.cpu().numpy().copy()
    tau = new._cell_adaptation_tau.cpu().numpy().copy()
    assert tau[3] != tau[4] and tau[0] == np.float32(.3)

    def derivative(_, value):
        meter['dop853_rhs_calls'] += 1
        rate, adaptation = value[:6], value[6:]
        current = weights @ (rate*mask)
        target = np.maximum(parameters[3]*np.tanh(
            (drive+current-parameters[2]-adaptation)
            * (parameters[1]/parameters[3])), 0)
        return np.r_[(target-rate)/parameters[0],
                     (gain*rate-adaptation)/tau]

    began = time.perf_counter()
    reference = solve_ivp(derivative, (0., .005), np.r_[rate0, z0],
                          method='DOP853', rtol=1e-11, atol=1e-12)
    meter['dop853_wall_s'] = time.perf_counter()-began
    assert reference.success
    new.advance(drive, steps=20)
    old.advance(drive, steps=20)
    np.testing.assert_allclose(new.snapshot()['rate'], reference.y[:6, -1],
                               rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(new.snapshot()['adaptation'], reference.y[6:, -1],
                               rtol=2e-5, atol=2e-5)
    assert not np.array_equal(new.snapshot()['adaptation'],
                              old.snapshot()['adaptation'])
    assert new.snapshot()['output_mask'][3] == 0
    saved = new.snapshot()
    StateStore.save(tmp_path/'small-checkpoint', saved)
    disk = StateStore.load(tmp_path/'small-checkpoint')
    clone = generated_model('cpu', [math.log(.5), math.log(3.)])[1]
    clone.restore(disk)
    drive[4] += .2
    new.advance(drive, steps=20)
    clone.advance(drive, steps=20)
    for key in ('rate', 'adaptation', 'drive', 'output_mask', 'tick'):
        left, right = new.snapshot()[key], clone.snapshot()[key]
        if isinstance(left, np.ndarray):
            assert left.tobytes() == right.tobytes()
        else:
            assert left == right
    assert meter['logical_steps'] == 80


def test_cpu_invalid_input_and_restore_are_atomic(meter):
    ids = np.array([3, 4], np.int64)
    for wrong_ids, wrong_ell in (
        (np.array([3, 3], np.int64), np.zeros(2, np.float64)),
        (np.array([3, 6], np.int64), np.zeros(2, np.float64)),
        (ids.astype(np.int32), np.zeros(2, np.float64)),
        (ids, np.zeros(2, np.float32)),
        (ids, np.array([np.nan, 0.], np.float64)),
        (ids, np.array([np.nextafter(LOG_TAU_LOWER, -np.inf), 0.])),
        (ids, np.array([0., np.nextafter(LOG_TAU_UPPER, np.inf)])),
    ):
        with pytest.raises(ValueError):
            checked_motor_log_tau(wrong_ids, wrong_ell, 6)
    checked_motor_log_tau(ids, np.array([LOG_TAU_LOWER, LOG_TAU_UPPER]), 6)
    old, new, _, _ = generated_model('cpu', [0., math.log(2.)])
    before = new.snapshot()
    for change in (
        lambda x: x['motor_timescale']['ids'].__setitem__(0, 4),
        lambda x: x['motor_timescale'].__setitem__(
            'log_tau', x['motor_timescale']['log_tau'].astype(np.float32)),
        lambda x: x['motor_timescale']['inverse32'].__setitem__(4, 1.),
        lambda x: x['motor_timescale']['tau64'].__setitem__(3, .9),
        lambda x: x.__setitem__('model_hash', 'bad'),
        lambda x: x['rate'].__setitem__(0, np.nan),
        lambda x: x['output_mask'].__setitem__(0, .5),
    ):
        damaged = copy.deepcopy(before)
        change(damaged)
        with pytest.raises(ValueError):
            new.restore(damaged)
        for key in ('rate', 'adaptation', 'drive', 'output_mask', 'tick'):
            left, right = new.snapshot()[key], before[key]
            if isinstance(left, np.ndarray):
                assert left.tobytes() == right.tobytes()
            else:
                assert left == right
    assert old.identity == new.source_motor_identity
    assert meter['logical_steps'] == 0


def test_cpu_anatomical_mapping_and_kernel_source_guard(meter):
    membership = np.arange(391, dtype=np.int64) % 96
    pool = np.zeros((96, 391), np.float64)
    pool[membership, np.arange(391)] = 1.
    muscles = [{'leg': 'lf', 'muscle': 'm'+str(i),
        'columns': np.flatnonzero(membership == i).tolist()} for i in range(96)]
    ell96 = np.zeros(96, np.float64)
    ell96[3] = math.log(.5)
    ell96[75] = math.log(3.)
    ids, motor_ell, returned = map_muscle_log_tau(np.arange(391, dtype=np.int64),
        pool, membership, muscles, ell96)
    np.testing.assert_array_equal(ids, np.arange(391))
    np.testing.assert_array_equal(motor_ell, ell96[membership])
    assert returned.tobytes() == ell96.tobytes()
    tau64, tau32, inverse32 = canonical_timescales(ids, motor_ell, 391)
    assert tau64.dtype == np.float64 and tau32.dtype == inverse32.dtype == np.float32
    assert tau32[3] == np.float32(.3*math.exp(ell96[3]))
    assert inverse32[3] == np.float32(1/tau64[3])
    assert _timescale_source(True).count('cell_inverse_tau[i]') == 2
    broken = copy.deepcopy(muscles)
    broken[0]['columns'] = []
    with pytest.raises(ValueError):
        map_muscle_log_tau(ids, pool, membership, broken, ell96)
    assert meter['completed_models'] == 0


def test_cpu_actual_timescale_configuration_raw_precedes_rejection(tmp_path, meter):
    _, network, _, _ = generated_model('cpu', [0., math.log(2.)])
    destination = tmp_path/'actual-timescale.npz'
    network._cell_inverse_tau[3] = 9.
    with pytest.raises(ValueError, match='Saved actual native timescale'):
        runner._save_timescale_configuration(destination, network,
            network.motor_ids, network.motor_log_tau)
    with np.load(destination, allow_pickle=False) as raw:
        assert raw['full_inverse32'][3] == np.float32(9.)
        assert raw['motor_ids'].tolist() == [3, 4]
        assert raw['motor_log_tau'].dtype == np.float64
    assert meter['logical_steps'] == 0


def test_cpu_wrapper_failure_closes_shared_body_once(meter):
    class Body:
        closes = 0
        def close(self):
            self.closes += 1

    body = Body()
    base = MotorAdaptationTrial.__new__(MotorAdaptationTrial)
    base.closed = False
    base.body = body
    base.network = object()
    with pytest.raises(ValueError, match='Unmodified E123'):
        MotorTimescaleTrial(base, np.zeros(96, np.float64), [], np.zeros(391))
    assert body.closes == 1
    assert meter['completed_models'] == 0


def test_cpu_wrapper_six_cell_layer_restore_without_body_advance(monkeypatch, meter):
    """Exercise the layer order with a generated six-cell body stub."""
    old = generated_model('cpu', [0., 0.])[0]
    ids = old.motor_ids.copy()
    ell96 = np.zeros(96, np.float64)
    ell96[3] = math.log(2.)
    muscles = [{'leg': 'lf', 'muscle': 'm'+str(i), 'columns': []}
               for i in range(96)]

    class Body:
        def __init__(self):
            self.closes = 0
        def close(self):
            self.closes += 1
        def snapshot(self):
            return {'generated_body': 1}

    graph = type('Graph', (), {'nodes': [{'id': 'generated:'+str(i)}
                                        for i in range(6)]})()
    bodies = []

    def base_layer(saved=None):
        model = generated_model('cpu', [0., 0.])[0]
        if saved is not None:
            model.restore(saved)
        body = Body()
        bodies.append(body)
        base = MotorAdaptationTrial.__new__(MotorAdaptationTrial)
        base.graph = graph
        base.body = body
        base.closed = False
        base.network = model
        base.adapter = type('Adapter', (), {})()
        base.adapter.identity = 'generated-adapter'
        base.adapter.groups = muscles
        base.adapter.calibration = {'motor_ids': ids.copy(),
                                    'pool': np.zeros((96, 2))}
        base.motor_adaptation_patch = {'generated': True}
        return base

    def small_mapping(motor_ids, _pool, _membership, _muscles, values):
        assert np.array_equal(motor_ids, ids)
        return ids.copy(), values[[3, 75]].copy(), values.copy()

    def small_snapshot(self):
        return {'schema': self.schema, 'network': self.network.snapshot(),
                'body': self.body.snapshot(),
                'motor_adaptation_patch': {'generated': True}}

    def small_from_checkpoint(cls, graph_arg, saved):
        assert graph_arg is graph
        assert saved['schema'] == MotorAdaptationTrial.schema
        return base_layer(saved['network'])

    monkeypatch.setattr('tools.banc_m3_motor_timescale.map_muscle_log_tau',
                        small_mapping)
    monkeypatch.setattr(MotorAdaptationTrial, 'snapshot', small_snapshot)
    monkeypatch.setattr(MotorAdaptationTrial, 'from_checkpoint',
                        classmethod(small_from_checkpoint))
    base = base_layer()
    trial = MotorTimescaleTrial(base, ell96, muscles, np.zeros(2, np.int64))
    assert trial.network.source_motor_identity == old.identity
    assert trial.motor_timescale_patch['motor_roster'] == [
        'generated:3', 'generated:4']
    saved = trial.snapshot()
    clone = MotorTimescaleTrial.from_checkpoint(graph, saved)
    assert clone.snapshot()['network']['model_hash'] == saved['network']['model_hash']
    assert meter['logical_steps'] == 0 and meter['cuda_setup_steps'] == 0
    trial.close()
    clone.close()
    assert bodies[-2].closes == bodies[-1].closes == 1


def test_cpu_verifier_preflight_and_reserve(tmp_path, meter):
    assert runner.ARMS == (('original_motor_gain_one', 10),
        ('zero_log_tau', 10), ('heterogeneous_log_tau', 10),
        ('heterogeneous_future', 1), ('clone_future', 1))
    with pytest.raises(ValueError, match='freeze SHA'):
        runner.reviewed_inputs(tmp_path/'missing', 'bad', tmp_path/'missing', 'bad')
    with pytest.raises(TimeoutError, match='6 s'):
        runner.require_room(time.perf_counter()-24, 'synthetic')
    assert meter['completed_models'] == 0


@pytest.mark.parametrize('label', ['state_store_save', 'state_store_load', 'manifest'])
@pytest.mark.parametrize('fail_at', [1, 2])
def test_cpu_storage_manifest_preentry_receipt_failure_is_known_zero(
        label, fail_at, meter):
    row = {'stage': 'intent', 'requested': False, 'entered': False,
           'returned': False, 'known_zero_work': False,
           'unknown_partial_work': False}
    secondary = []
    attempts = {'count': 0}
    durable_rows = []

    def receipt():
        attempts['count'] += 1
        if attempts['count'] == fail_at:
            raise OSError('injected pre-entry ledger failure')
        durable_rows.append(copy.deepcopy(row))

    with pytest.raises(OSError, match='injected pre-entry'):
        runner.durable_preentry(receipt, row, label,
                                'unknown_partial_work', secondary,
                                time.perf_counter())
    assert attempts['count'] == fail_at+1
    assert not secondary
    assert row['requested'] is row['entered'] is row['returned'] is False
    assert row['known_zero_work'] is True
    assert row['unknown_partial_work'] is False
    assert row['stage'] == label+('_intent' if fail_at == 1 else '_entry') \
        +'_receipt_failed'
    assert durable_rows[-1] == row
    assert meter['completed_models'] == meter['logical_steps'] == 0


@pytest.mark.parametrize('failure', [
    'clone_constructor', 'raw_save', 'readback',
    'model_intent_receipt', 'model_entry_receipt', 'model_return_receipt',
    'advance_intent_receipt', 'advance_entry_receipt', 'advance_return_receipt',
    'store_save_intent_receipt', 'store_save_entry_receipt',
    'store_save_return_receipt', 'store_load_intent_receipt',
    'store_load_entry_receipt', 'store_load_return_receipt'])
def test_cpu_verifier_fault_keeps_raw_and_work(
        failure, tmp_path, monkeypatch, meter):
    """Generated six-cell runner fault; no original graph/model/GPU is opened."""
    weights = csr_matrix(np.zeros((6, 6), np.float32))
    parameters = (np.full(6, .02, np.float32), np.ones(6, np.float32),
                  np.full(6, 1.5, np.float32), np.full(6, 40., np.float32))
    core_ids = np.array([1, 3, 4], np.int64)
    source = CoreAdaptationNetwork(weights, *parameters,
        adaptation_gain=8., adaptation_tau_s=.3, dt=.00025,
        device='cpu', core_ids=core_ids, core_gain=0.)
    initial = source.snapshot()
    initial['tick'] = 4000
    initial['rate'] = np.array([4., 3., 2., 5., 1., 6.], np.float32)
    initial['adaptation'] = np.array([.2, .1, .3, .4, .15, .25], np.float32)
    initial['drive'] = np.full(6, 3., np.float32)
    source.restore(initial)
    initial = source.snapshot()
    pool = np.zeros((96, 2), np.float64)
    pool[3, 0] = pool[75, 1] = 1.
    cal = {'motor_ids': np.array([3, 4], np.int64), 'pool': pool,
           'membership': np.array([3, 75], np.int64),
           'rest': np.zeros(42), 'moment': np.ones((42, 96)),
           'ltm': np.arange(6)}
    original = {'graph_hash': 'generated', 'network': initial,
                'adapter': {'activation': np.ones(96)}}
    graph = type('Graph', (), {'n': 6, 'hash': 'generated'})()
    relative = tmp_path.relative_to(runner.ROOT)
    np.savez(tmp_path/'calibration.npz', **cal)
    freeze = {'source_e121': str(relative)}

    class ThreadLimits:
        def restore_original_limits(self):
            pass

    def small_baseline(_weights, _parameters, saved_original, motor_ids,
                       gain, setup_record):
        model = MotorAdaptationNetwork(weights, *parameters,
            adaptation_gain=8., adaptation_tau_s=.3, dt=.00025,
            device='cpu', core_ids=core_ids, core_gain=0.,
            motor_ids=motor_ids, motor_gains=np.full(len(motor_ids), gain, np.float32))
        state = copy.deepcopy(saved_original['network'])
        state['model_hash'] = model.identity
        state['motor_adaptation'] = model.snapshot()['motor_adaptation']
        model.restore(state)
        setup_record['stage'] = 'restored_original_tick4000'
        return model

    real_class = MotorTimescaleNetwork
    attempts = {'count': 0}

    class SmallCpuTimescale(real_class):
        def __init__(self, *args, **kwargs):
            attempts['count'] += 1
            if failure == 'clone_constructor' and attempts['count'] == 3:
                raise RuntimeError('injected clone constructor failure')
            kwargs['device'] = 'cpu'
            kwargs['cuda_implementation'] = 'reference'
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(runner, 'reviewed_inputs', lambda *_: freeze)
    monkeypatch.setattr(runner.e123, 'configure_one_cpu_thread',
                        lambda: (ThreadLimits(), {'generated_cpu_threads': 1}))
    monkeypatch.setattr(runner.e123, 'native_graph_and_inputs',
                        lambda _: (graph, cal, None, {}, original))
    monkeypatch.setattr(runner.e123, 'reconstruct_weights', lambda *_: weights)
    monkeypatch.setattr(runner.e123, 'native_parameters', lambda *_: parameters)
    monkeypatch.setattr(runner.e123, 'make_network', small_baseline)
    monkeypatch.setattr(runner.e123, 'CORE_IDENTITY', source.identity)
    monkeypatch.setattr(runner, 'checked_timescale_inputs', lambda *_: {
        'motor_log_tau': np.array([math.log(.5), math.log(3.)], np.float64),
        'lf_motor_columns': np.array([0, 1], np.int64)})
    monkeypatch.setattr(runner, 'MotorTimescaleNetwork', SmallCpuTimescale)
    if failure == 'raw_save':
        old_save = runner.e123.save_arm_raw

        def fail_canonical(path, *args, **kwargs):
            if path.name == 'original_motor_gain_one-raw.npz':
                raise OSError('injected canonical raw save failure')
            return old_save(path, *args, **kwargs)

        monkeypatch.setattr(runner.e123, 'save_arm_raw', fail_canonical)
    elif failure == 'readback':
        def fail_readback(_network):
            raise RuntimeError('injected raw readback failure')

        monkeypatch.setattr(runner.e123, 'raw_neural', fail_readback)
    if failure.endswith('_receipt'):
        original_write = runner.e123.write_json
        hit = {'count': 0}
        target = {
            'model_intent_receipt': ('model_attempts', 'intent'),
            'model_entry_receipt': ('model_attempts', 'constructor_requested'),
            'model_return_receipt': ('model_attempts', 'restored_original_tick4000'),
            'advance_intent_receipt': ('advance_requests', 'intent'),
            'advance_entry_receipt': ('advance_requests', 'advance_requested'),
            'advance_return_receipt': ('advance_requests', 'advance_entered'),
            'store_save_intent_receipt': ('state_store_requests', 'intent'),
            'store_save_entry_receipt': ('state_store_requests',
                                         'state_store_save_requested'),
            'store_save_return_receipt': ('state_store_requests',
                                          'state_store_save_returned'),
            'store_load_intent_receipt': ('state_store_requests', 'intent'),
            'store_load_entry_receipt': ('state_store_requests',
                                         'state_store_load_requested'),
            'store_load_return_receipt': ('state_store_requests',
                                          'state_store_load_returned')}[failure]

        def fail_one_receipt(path, value):
            if path.name == 'advance-ledger.json' and hit['count'] == 0:
                rows = value[target[0]]
                owner_matches = (not failure.startswith('store_')
                    or rows and rows[-1]['owner'] == (
                        'state_store_load' if failure.startswith('store_load')
                        else 'state_store_save'))
                if rows and owner_matches and rows[-1]['stage'] == target[1]:
                    hit['count'] += 1
                    raise OSError('injected '+failure+' ledger receipt failure')
            return original_write(path, value)

        monkeypatch.setattr(runner.e123, 'write_json', fail_one_receipt)
    output = tmp_path/'generated-failed-run'
    expected = {'clone_constructor': 'injected clone constructor failure',
                'raw_save': 'injected canonical raw save failure',
                'readback': 'injected raw readback failure'}.get(
                    failure, 'injected '+failure+' ledger receipt failure')
    with pytest.raises((RuntimeError, OSError), match=expected):
        runner.run(None, None, None, None, output)
    if failure.endswith('_receipt'):
        assert hit['count'] == 1
        ledger = json.loads((output/'advance-ledger.json').read_text())
        error = json.loads((output/'execution-error.json').read_text())
        assert error['primary']['message'] == expected
        assert not (output/'result.json').exists()
        row = ledger[target[0]][-1]
        preentry = failure.endswith(('intent_receipt', 'entry_receipt'))
        if preentry:
            assert row['requested'] is row['entered'] is row['returned'] is False
            assert row['known_zero_work'] is True
            assert row.get('unknown_partial_setup', False) is False
            assert row.get('unknown_partial_advance', False) is False
            assert row.get('unknown_partial_work', False) is False
            assert row['stage'].endswith('_receipt_failed')
        else:
            assert row['requested'] is row['entered'] is row['returned'] is True
            assert row['known_zero_work'] is False
        expected_steps = (600 if failure.startswith('store_') else 20
                          if failure == 'advance_return_receipt' else 0)
        assert ledger['logical_steps'] == meter['logical_steps'] == expected_steps
        assert ledger['decoder_controls'] == expected_steps//20
        if failure == 'advance_return_receipt':
            assert (output/'original_motor_gain_one-raw.npz').exists()
        if failure == 'store_save_return_receipt':
            assert (output/'heterogeneous-checkpoint').is_dir()
        return
    name = ('heterogeneous_future-raw.npz' if failure == 'clone_constructor'
            else 'original_motor_gain_one-raw-emergency.npz' if failure == 'raw_save'
            else 'original_motor_gain_one-raw.npz')
    with np.load(output/name, allow_pickle=False) as raw:
        assert raw['ticks'][-1] == (4220 if failure == 'clone_constructor'
                                   else 4200 if failure == 'raw_save' else 4020)
        assert bool(raw['decoded'][-1]) is (failure != 'readback')
        assert raw['rate'].shape[1:] == (6,)
    ledger = json.loads((output/'advance-ledger.json').read_text())
    expected_steps = {'clone_constructor': 620, 'raw_save': 200,
                      'readback': 20}[failure]
    expected_controls = {'clone_constructor': 31, 'raw_save': 10,
                         'readback': 0}[failure]
    assert ledger['logical_steps'] == meter['logical_steps'] == expected_steps
    assert ledger['decoder_controls'] == expected_controls
    assert ledger['cuda_setup_kernel_steps'] == meter['cuda_setup_steps'] == 0
    if failure == 'clone_constructor':
        assert len(ledger['model_attempts']) == 4
        assert ledger['model_attempts'][-1]['requested'] is True
        assert ledger['model_attempts'][-1]['returned'] is False
        assert ledger['model_attempts'][-1]['unknown_partial_setup'] is True
    assert len(ledger['advance_requests']) == expected_steps//20
    if failure == 'readback':
        assert ledger['advance_requests'][-1]['unknown_partial_advance'] is True
        assert ledger['advance_requests'][-1]['returned'] is False
    else:
        assert all(row['requested'] and row['returned']
                   for row in ledger['advance_requests'])
    assert not (output/'result.json').exists()
    error = json.loads((output/'execution-error.json').read_text())
    assert error['primary']['message'] == expected


def test_cuda_neutral_exact_and_heterogeneous_reference(meter):
    """Parent-only generated CUDA check; never run in first implementation."""
    import torch
    assert torch.cuda.is_available(), 'Actual CUDA required; no skip/fallback'
    old, zero, _, _ = generated_model('cuda', [0., 0.])
    cpu = generated_model('cpu', [math.log(.5), math.log(3.)])[1]
    gpu = generated_model('cuda', [math.log(.5), math.log(3.)])[1]
    zero_ptrs = (zero._cell_adaptation_tau.data_ptr(),
                 zero._cell_inverse_tau.data_ptr())
    gpu_ptrs = (gpu._cell_adaptation_tau.data_ptr(),
                gpu._cell_inverse_tau.data_ptr())
    drive = old.drive.cpu().numpy().copy()
    for model in (old, zero, cpu, gpu):
        model.set_muted([3])
        model.advance(drive, steps=20)
    for key in ('rate', 'adaptation', 'drive', 'output_mask', 'tick'):
        left, right = old.snapshot()[key], zero.snapshot()[key]
        if isinstance(left, np.ndarray):
            assert left.tobytes() == right.tobytes()
        else:
            assert left == right
    for key in ('rate', 'adaptation'):
        np.testing.assert_allclose(cpu.snapshot()[key], gpu.snapshot()[key],
                                   rtol=2e-5, atol=2e-5)
    assert zero_ptrs == (zero._cell_adaptation_tau.data_ptr(),
                         zero._cell_inverse_tau.data_ptr())
    assert gpu_ptrs == (gpu._cell_adaptation_tau.data_ptr(),
                        gpu._cell_inverse_tau.data_ptr())
    # Three generated_model calls construct four CUDA and two CPU models.
    assert meter['cuda_setup_steps'] == 4*3*20 == 240
