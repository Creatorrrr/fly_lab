"""Small synthetic checks only; the original 158706-cell run belongs to review."""

import copy
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix

from flylab.c.adaptive_rate import AdaptiveRateNetwork
from flylab.c.research_rate import ResearchRateNetwork
from tools import verify_c_banc_m3_motor_adaptation as runner
from tools.banc_m3_core_adaptation import CoreAdaptationNetwork
from tools.banc_m3_intrinsic_recruitment import IntrinsicRecruitmentTrial
from tools.banc_m3_motor_adaptation import (
    MotorAdaptationNetwork,
    MotorAdaptationTrial,
    map_muscle_gains,
)
from tools.verify_c_banc_m3_motor_adaptation import (
    accounted_control,
    require_room,
    save_arm_raw,
    save_model_configuration,
)


@pytest.fixture(autouse=True)
def meter(monkeypatch, request):
    """Count completed constructions and real tick deltas, not estimates."""
    counts = {'test': request.node.name, 'model_attempts': 0,
              'completed_models': 0, 'cuda_setup_steps': 0,
              'logical_steps': 0, 'model_s': 0., 'wall_s': 0.}
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
            counts['model_s'] += delta*self.dt

    monkeypatch.setattr(AdaptiveRateNetwork, '__init__', counted_init)
    monkeypatch.setattr(ResearchRateNetwork, '_advance', counted_advance)
    began = time.perf_counter()
    yield counts
    counts['wall_s'] = time.perf_counter()-began
    print('E123_SYNTHETIC_COUNT', counts)


def model(device, gains=(0., 1.5), implementation='packed'):
    # Mixed signs, recurrent paths, isolated drive, and an outgoing cut.
    weights = csr_matrix(np.array([
        [0., .15, 0., -.10, 0., 0.],
        [.08, 0., .06, 0., 0., 0.],
        [0., -.10, 0., .12, 0., 0.],
        [0., 0., .09, 0., -.08, 0.],
        [0., 0., 0., .11, 0., 0.],
        [0., 0., 0., 0., .07, 0.],
    ], dtype=np.float32))
    parameters = (
        np.array([.025, .03, .028, .024, .027, .026], np.float32),
        np.ones(6, np.float32),
        np.full(6, 1.5, np.float32),
        np.full(6, 40., np.float32),
    )
    common = {'adaptation_gain': 8., 'adaptation_tau_s': .3, 'dt': .00025,
              'device': device, 'capture_steps': 20,
              'cuda_implementation': implementation,
              'core_ids': np.array([1, 3, 4], np.int64), 'core_gain': 0.}
    source = CoreAdaptationNetwork(weights, *parameters, **common)
    new = MotorAdaptationNetwork(weights, *parameters, **common,
        motor_ids=np.array([3, 4], np.int64),
        motor_gains=np.asarray(gains, np.float32))
    saved = source.snapshot()
    saved['rate'] = np.array([4., 3., 2., 5., 1., 6.], np.float32)
    saved['adaptation'] = np.array([.2, .1, .3, .4, .15, .25], np.float32)
    saved['drive'] = np.array([2.8, 2.5, 2.7, 3.1, 2.9, 3.0], np.float32)
    saved['tick'] = 17
    source.restore(saved)
    candidate = copy.deepcopy(saved)
    candidate['model_hash'] = new.identity
    candidate['motor_adaptation'] = new.snapshot()['motor_adaptation']
    new.restore(candidate)
    return source, new, weights, parameters


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_zero_gain_full_state_parity_and_checkpoint(device, meter):
    import torch
    if device == 'cuda':
        assert torch.cuda.is_available(), 'Actual CUDA is required; no skip/fallback'
    source, candidate, _, _ = model(device, gains=(0., 0.))
    np.testing.assert_array_equal(source._cell_gain.cpu().numpy(),
                                  candidate._cell_gain.cpu().numpy())
    assert candidate.identity != source.identity
    drive = source.drive.cpu().numpy().copy()
    source.set_muted([0, 3])
    candidate.set_muted([0, 3])
    for control in range(3):
        source.advance(drive, steps=20)
        candidate.advance(drive, steps=20)
        for key in ('rate', 'adaptation', 'drive', 'output_mask', 'tick'):
            np.testing.assert_array_equal(source.snapshot()[key], candidate.snapshot()[key])
    before = candidate.snapshot()
    for corruption in (
        lambda x: x['motor_adaptation']['gains'].__setitem__(0, 1.),
        lambda x: x['motor_adaptation'].__setitem__(
            'gains', x['motor_adaptation']['gains'].astype(np.float64)),
        lambda x: x['motor_adaptation']['ids'].__setitem__(0, 2),
        lambda x: x.__setitem__('model_hash', 'bad'),
        lambda x: x['core_adaptation'].__setitem__('gain', 4.),
        lambda x: x['rate'].__setitem__(0, np.nan),
        lambda x: x['output_mask'].__setitem__(0, .5),
    ):
        damaged = copy.deepcopy(before)
        corruption(damaged)
        with pytest.raises(ValueError):
            candidate.restore(damaged)
        for key in ('rate', 'adaptation', 'drive', 'output_mask', 'tick'):
            np.testing.assert_array_equal(candidate.snapshot()[key], before[key])
    assert meter['logical_steps'] == 120
    assert meter['model_s'] == pytest.approx(.03)
    assert meter['cuda_setup_steps'] == (120 if device == 'cuda' else 0)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_heterogeneous_gain_cut_drive_and_dop853(device, meter):
    import torch
    if device == 'cuda':
        assert torch.cuda.is_available(), 'Actual CUDA is required; no skip/fallback'
    _, network, weights, parameters = model(device, gains=(.5, 2.25))
    drive = network.drive.cpu().numpy().copy()
    rate0 = network.rate.cpu().numpy().copy()
    z0 = network.adaptation.cpu().numpy().copy()
    gain = network._cell_gain.cpu().numpy().copy()
    expected_gain = np.array([8., 0., 8., .5, 2.25, 8.], np.float32)
    np.testing.assert_array_equal(gain, expected_gain)
    network.set_muted([3])
    mask = network.output_mask.cpu().numpy().copy()

    def derivative(_, value):
        rate, adaptation = value[:6], value[6:]
        current = weights @ (rate*mask)
        target = np.maximum(parameters[3]*np.tanh(
            (drive+current-parameters[2]-adaptation)
            * (parameters[1]/parameters[3])), 0)
        return np.r_[(target-rate)/parameters[0],
                     (gain*rate-adaptation)/.3]

    reference = solve_ivp(derivative, (0., .005), np.r_[rate0, z0],
                          method='DOP853', rtol=1e-11, atol=1e-12)
    assert reference.success
    network.advance(drive, steps=20)
    actual = network.snapshot()
    np.testing.assert_allclose(actual['rate'], reference.y[:6, -1],
                               rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual['adaptation'], reference.y[6:, -1],
                               rtol=2e-5, atol=2e-5)
    assert actual['adaptation'][3] != z0[3]  # muted output still adapts
    assert actual['output_mask'][3] == 0
    clone = model(device, gains=(.5, 2.25))[1]
    clone.restore(actual)
    new_drive = drive.copy()
    new_drive[3] += .2
    network.advance(new_drive, steps=20)
    clone.advance(new_drive, steps=20)
    for key in ('rate', 'adaptation', 'drive', 'output_mask', 'tick'):
        np.testing.assert_array_equal(network.snapshot()[key], clone.snapshot()[key])
    assert meter['logical_steps'] == 60
    assert meter['model_s'] == pytest.approx(.015)
    assert meter['cuda_setup_steps'] == (240 if device == 'cuda' else 0)


def test_gain_input_and_anatomical_membership_validation():
    for gains, ids in (([np.nan], [3]), ([9], [3]), ([-.1], [3]),
                       ([1, 2], [3, 3]), ([1], [9])):
        with pytest.raises(ValueError):
            MotorAdaptationNetwork(csr_matrix((6, 6), dtype=np.float32),
                *[np.ones(6, np.float32) for _ in range(4)],
                adaptation_gain=8., core_ids=np.array([1]), core_gain=0.,
                motor_ids=np.asarray(ids), motor_gains=np.asarray(gains))
    membership = np.arange(391) % 96
    pool = np.zeros((96, 391), np.float64)
    pool[membership, np.arange(391)] = 1.
    muscles = [{'leg': 'lf', 'muscle': 'm'+str(i),
                'columns': np.flatnonzero(membership == i).tolist()}
               for i in range(96)]
    muscle_gains = np.linspace(0., 8., 96, dtype=np.float32)
    ids, values, compact = map_muscle_gains(
        np.arange(391), pool, membership, muscles, muscle_gains)
    np.testing.assert_array_equal(ids, np.arange(391))
    np.testing.assert_array_equal(values, muscle_gains[membership])
    np.testing.assert_array_equal(compact, muscle_gains)
    broken = copy.deepcopy(muscles)
    broken[0]['columns'] = []
    with pytest.raises(ValueError):
        map_muscle_gains(ids, pool, membership, broken, muscle_gains)


def test_cpu_cuda_heterogeneous_agreement(meter):
    import torch
    assert torch.cuda.is_available(), 'Actual CUDA is required; no skip/fallback'
    cpu = model('cpu', gains=(.5, 2.25))[1]
    cuda = model('cuda', gains=(.5, 2.25))[1]
    drive = cpu.drive.cpu().numpy().copy()
    cpu.set_muted([0, 3])
    cuda.set_muted([0, 3])
    cpu.advance(drive, steps=20)
    cuda.advance(drive, steps=20)
    for key in ('rate', 'adaptation'):
        np.testing.assert_allclose(cpu.snapshot()[key], cuda.snapshot()[key],
                                   rtol=2e-5, atol=2e-5)
    assert meter['logical_steps'] == 40
    assert meter['model_s'] == pytest.approx(.01)
    assert meter['cuda_setup_steps'] == 120


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_silent_motor_state_and_outgoing_cut(device, meter):
    import torch
    if device == 'cuda':
        assert torch.cuda.is_available(), 'Actual CUDA is required'
    source, candidate, _, _ = model(device, gains=(1., 2.))
    for network in (source, candidate):
        saved = network.snapshot()
        saved['rate'].fill(0)
        saved['drive'].fill(0)
        saved['adaptation'][:] = np.array([.2, .1, .3, .4, .15, .25], np.float32)
        network.restore(saved)
        network.set_muted([3, 4])
        network.advance(np.zeros(6, np.float32), steps=20)
    for key in ('rate', 'adaptation', 'drive', 'output_mask', 'tick'):
        np.testing.assert_array_equal(source.snapshot()[key], candidate.snapshot()[key])
    assert candidate.snapshot()['rate'][3] == 0
    assert candidate.snapshot()['adaptation'][3] < .4
    assert meter['logical_steps'] == 40
    assert meter['cuda_setup_steps'] == (120 if device == 'cuda' else 0)


@pytest.mark.parametrize('failure', [
    'post_tick', 'nonfinite', 'nonfinite_adaptation',
    'observation', 'combined_observation', 'decoder'])
def test_accounted_failure_retains_latest_raw(failure, tmp_path, meter, monkeypatch):
    source, network, _, _ = model('cpu', gains=(1., 1.))
    cal = {'motor_ids': np.array([3, 4]),
           'pool': np.zeros((96, 2)), 'rest': np.zeros(42),
           'moment': np.zeros((42, 96)), 'ltm': np.arange(6)}
    activation = np.zeros(96)
    ledger = {'logical_steps': 0, 'logical_neural_s': 0.,
              'arms': {}, 'decoder_controls': 0}
    rows = []
    if failure in ('post_tick', 'combined_observation'):
        native = network.advance

        def post_tick(drive, steps):
            native(drive, steps=steps)
            raise ValueError('injected post-tick validation')

        network.advance = post_tick
        if failure == 'combined_observation':
            monkeypatch.setattr(runner, 'raw_neural', lambda _: (_ for _ in ()).throw(
                RuntimeError('injected secondary raw observation failure')))
    elif failure == 'nonfinite':
        native = network.advance

        def corrupt_after_tick(drive, steps):
            native(drive, steps=steps)
            network.rate[3] = float('nan')
            raise ValueError('injected nonfinite post-tick state')

        network.advance = corrupt_after_tick
    elif failure == 'nonfinite_adaptation':
        native = network.advance

        def corrupt_adaptation(drive, steps):
            native(drive, steps=steps)
            network.adaptation[3] = float('nan')

        network.advance = corrupt_adaptation
    elif failure == 'observation':
        network.snapshot = lambda: (_ for _ in ()).throw(
            ValueError('injected observation failure'))
    else:
        cal['pool'] = np.zeros((95, 2))
    with pytest.raises((ValueError, RuntimeError)) as raised:
        accounted_control(network, source.drive.cpu().numpy().copy(), 20,
                          cal, activation, ledger, 'fault', rows)
    if failure == 'combined_observation':
        assert str(raised.value) == 'injected post-tick validation'
        assert any('secondary raw observation' in note for note in raised.value.__notes__)
        assert len(ledger['observation_errors']) == 1
    assert meter['logical_steps'] == ledger['logical_steps'] == 20
    assert ledger['logical_neural_s'] == pytest.approx(.005)
    assert ledger['arms']['fault'] == 20
    assert ledger['decoder_controls'] == 0
    assert len(rows) == 1 and rows[0]['tick'] == 37
    assert rows[0]['decoded'] is False
    original = {'graph_hash': 'synthetic', 'network': source.snapshot(),
                'adapter': {'activation': activation}}
    destination = tmp_path/'fault-raw.npz'
    save_arm_raw(destination, 'fault', rows, network, cal,
                 source.drive.cpu().numpy(), original)
    with np.load(destination, allow_pickle=False) as raw:
        assert raw['ticks'].tolist() == [37]
        np.testing.assert_array_equal(raw['rate'][0], rows[0]['rate'])
        if failure == 'nonfinite':
            assert np.isnan(raw['rate'][0, 3])
        if failure == 'nonfinite_adaptation':
            assert np.isnan(raw['adaptation'][0, 3])
        assert not raw['decoded'][0]


def test_configuration_raw_precedes_full_gain_rejection(tmp_path):
    source, network, weights, parameters = model('cpu', gains=(1., 1.))
    expected = source._cell_gain.cpu().numpy().copy()
    raw = tmp_path/'configuration.npz'
    save_model_configuration(raw, network, weights, parameters, expected,
                             np.array([3, 4]), 1.)
    with np.load(raw, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['_cell_gain'],
                                      network._cell_gain.cpu().numpy())
        assert len(str(saved['values_sha256'])) == 64
    network._cell_gain[5] = 0
    damaged = tmp_path/'damaged-configuration.npz'
    with pytest.raises(ValueError, match='per-cell'):
        save_model_configuration(damaged, network, weights, parameters,
                                 expected, np.array([3, 4]), 1.)
    assert damaged.exists()  # Actual corrupt runtime tensor is retained.


def test_deadline_reserves_eight_seconds():
    require_room(100., 'synthetic', now=136.999)
    with pytest.raises(TimeoutError, match='8 s'):
        require_room(100., 'synthetic', now=137.)


def test_runner_persists_main_future_before_clone_init_failure(
        tmp_path, monkeypatch, meter):
    """Synthetic 391-cell runner fault; never loads the original BANC graph."""
    n = 391
    weights = csr_matrix((n, n), dtype=np.float32)
    parameters = (np.full(n, .02, np.float32),
                  np.ones(n, np.float32), np.full(n, 7.5, np.float32),
                  np.full(n, 200., np.float32))
    core_ids = np.arange(n, dtype=np.int64)
    source = CoreAdaptationNetwork(weights, *parameters,
        adaptation_gain=8., adaptation_tau_s=.3, dt=.00025,
        device='cpu', core_ids=core_ids, core_gain=0.)
    initial = source.snapshot()
    initial['tick'] = 4000
    initial['rate'] = np.full(n, 10., np.float32)
    initial['adaptation'] = np.zeros(n, np.float32)
    initial['drive'] = np.full(n, 15., np.float32)
    source.restore(initial)
    initial = source.snapshot()
    membership = np.arange(n) % 96
    pool = np.zeros((96, n))
    pool[membership, np.arange(n)] = 1./np.bincount(membership)[membership]
    cal = {'motor_ids': np.arange(n), 'pool': pool,
           'rest': np.zeros(42), 'moment': np.ones((42, 96)),
           'ltm': np.arange(6), 'membership': membership}
    original = {'graph_hash': 'synthetic', 'network': initial,
                'adapter': {'activation': np.ones(96)}}
    graph = SimpleNamespace(n=n, hash='synthetic')
    monkeypatch.setattr(runner, 'reviewed_inputs', lambda _: {
        'source_e121': 'verification/banc-natural-gait-m3-20260919/e121-port-interference'})
    monkeypatch.setattr(runner, 'native_graph_and_inputs',
                        lambda _: (graph, cal, None, None, original))
    monkeypatch.setattr(runner, 'reconstruct_weights', lambda *_: weights)
    monkeypatch.setattr(runner, 'native_parameters', lambda *_: parameters)

    class ThreadLimits:
        def restore_original_limits(self):
            pass

    monkeypatch.setattr(runner, 'configure_one_cpu_thread',
                        lambda: (ThreadLimits(), {'synthetic_cpu_threads': 1}))
    attempts = {'count': 0}

    def build_small(_, __, saved_original, motor_ids=None, gain=None,
                    setup_record=None):
        attempts['count'] += 1
        if attempts['count'] == 4:
            raise ValueError('injected clone initialization failure')
        common = {'adaptation_gain': 8., 'adaptation_tau_s': .3,
                  'dt': .00025, 'device': 'cpu', 'core_ids': core_ids,
                  'core_gain': 0.}
        if motor_ids is None:
            network = CoreAdaptationNetwork(weights, *parameters, **common)
        else:
            network = MotorAdaptationNetwork(weights, *parameters,
                **common, motor_ids=motor_ids,
                motor_gains=np.full(len(motor_ids), gain, np.float32))
        state = copy.deepcopy(saved_original['network'])
        state['model_hash'] = network.identity
        if motor_ids is not None:
            state['motor_adaptation'] = network.snapshot()['motor_adaptation']
        network.restore(state)
        setup_record.update(stage='restored_original_tick4000',
                            cuda_setup_kernel_steps_completed=0)
        return network

    monkeypatch.setattr(runner, 'make_network', build_small)
    output = tmp_path/'synthetic-failed-run'
    with pytest.raises(ValueError, match='clone initialization failure'):
        runner.run(None, output)
    with np.load(output/'positive-main-future-raw.npz', allow_pickle=False) as raw:
        assert raw['ticks'].tolist() == [4220]
        assert raw['decoded'].tolist() == [True]
        assert raw['rate'].shape == (1, n)
    ledger = json.loads((output/'advance-ledger.json').read_text())
    assert ledger['logical_steps'] == meter['logical_steps'] == 620
    assert ledger['decoder_controls'] == 31
    assert ledger['cuda_setup_kernel_steps'] == meter['cuda_setup_steps'] == 0
    assert len(ledger['model_attempts']) == 4
    assert ledger['model_attempts'][-1]['stage'] == 'attempted'
    assert not (output/'result.json').exists()
    error = json.loads((output/'execution-error.json').read_text())
    assert error['primary']['message'] == 'injected clone initialization failure'


def test_wrapper_closes_owned_body_on_mapping_failure():
    class Body:
        closed = False
        def close(self):
            self.closed = True

    class Adapter:
        def __init__(self):
            self.calibration = {'motor_ids': np.arange(391),
                                'pool': np.zeros((96, 391))}

    base = IntrinsicRecruitmentTrial.__new__(IntrinsicRecruitmentTrial)
    base.body = Body()
    base.adapter = Adapter()
    base.network = object()
    with pytest.raises(ValueError):
        MotorAdaptationTrial(base, np.zeros(96), [], np.zeros(391))
    assert base.body.closed


def test_cpu_wrapper_mapping_snapshot_restore_and_single_close(monkeypatch, meter):
    """Synthetic lower layer checks state ownership; no body future is claimed."""
    class Graph:
        def __init__(self):
            self.nodes = [{'id': 'synthetic:'+str(i)} for i in range(391)]

    class Body:
        def __init__(self):
            self.closes = 0

        def close(self):
            self.closes += 1

        def snapshot(self):
            return {'synthetic_body_state': 1}

    graph = Graph()
    membership = np.arange(391) % 96
    pool = np.zeros((96, 391))
    pool[membership, np.arange(391)] = 1.
    muscles = [{'leg': 'lf', 'muscle': 'm'+str(i),
                'columns': np.flatnonzero(membership == i).tolist()}
               for i in range(96)]
    gains = np.linspace(0, 8, 96, dtype=np.float32)
    created_bodies = []

    def fake_base(invalid=False):
        base = IntrinsicRecruitmentTrial.__new__(IntrinsicRecruitmentTrial)
        base.graph = graph
        base.closed = False
        base.body = Body()
        created_bodies.append(base.body)
        base.adapter = type('Adapter', (), {})()
        base.adapter.calibration = {'motor_ids': np.arange(391), 'pool': pool}
        base.adapter.motor_ids = np.arange(391)
        base.adapter.groups = muscles
        base.adapter.identity = 'synthetic-adapter'
        if invalid:
            base.network = object()
        else:
            values = np.full(391, .02, np.float32)
            base.network = CoreAdaptationNetwork(
                csr_matrix((391, 391), dtype=np.float32),
                values, np.ones(391, np.float32),
                np.full(391, 7.5, np.float32),
                np.full(391, 200., np.float32),
                adaptation_gain=8., adaptation_tau_s=.3,
                dt=.00025, device='cpu', core_ids=np.arange(391),
                core_gain=0.)
        return base

    def fake_snapshot(self):
        return {'schema': self.schema, 'network': self.network.snapshot(),
                'body': self.body.snapshot(), 'adapter_hash': self.adapter.identity}

    fail_restore = {'value': False}

    def fake_from_checkpoint(cls, graph_arg, saved):
        assert graph_arg is graph
        base = fake_base(invalid=fail_restore['value'])
        if not fail_restore['value']:
            base.network.restore(saved['network'])
        return base

    monkeypatch.setattr(IntrinsicRecruitmentTrial, 'snapshot', fake_snapshot)
    monkeypatch.setattr(IntrinsicRecruitmentTrial, 'from_checkpoint',
                        classmethod(fake_from_checkpoint))
    base = fake_base()
    trial = MotorAdaptationTrial(base, gains, muscles, membership)
    assert trial.network.device == 'cpu'
    assert trial.network.cuda_implementation == 'torch-cpu-csr'
    np.testing.assert_array_equal(trial.network.motor_gains, gains[membership])
    state = trial.snapshot()
    clone = MotorAdaptationTrial.from_checkpoint(graph, state)
    np.testing.assert_array_equal(clone.network.snapshot()['rate'],
                                  trial.network.snapshot()['rate'])
    assert clone.snapshot()['motor_adaptation_patch']['motor_roster'][3] == 'synthetic:3'
    assert meter['logical_steps'] == meter['cuda_setup_steps'] == 0
    fail_restore['value'] = True
    with pytest.raises(ValueError, match='Unmodified intrinsic'):
        MotorAdaptationTrial.from_checkpoint(graph, state)
    assert created_bodies[-1].closes == 1
    trial.close()
    clone.close()


def test_partial_raw_is_saved_and_existing_file_rejected(tmp_path):
    source, network, _, _ = model('cpu', gains=(0., 1.))
    initial = source.snapshot()
    original = {'graph_hash': 'synthetic', 'network': initial,
                'adapter': {'activation': np.zeros(96)}}
    cal = {'motor_ids': np.array([3, 4])}
    row = {'tick': network.tick, 'rate': network.readout(),
           'adaptation': network.adaptation.cpu().numpy().copy(),
           'drive': network.drive.cpu().numpy().copy(),
           'output_mask': network.output_mask.cpu().numpy().copy(),
           'motor_rate': network.readout()[cal['motor_ids']],
           'activation': np.zeros(96), 'request': np.zeros(42),
           'tendon': np.zeros(6), 'decoded': True, 'read_errors': []}
    destination = tmp_path/'partial.npz'
    save_arm_raw(destination, 'synthetic', [row], network, cal,
                 initial['drive'], original)
    with np.load(destination, allow_pickle=False) as raw:
        np.testing.assert_array_equal(raw['rate'][0], row['rate'])
        np.testing.assert_array_equal(raw['initial_adaptation'], initial['adaptation'])
        assert raw['ticks'].tolist() == [17]
    with pytest.raises(FileExistsError):
        save_arm_raw(destination, 'synthetic', [row], network, cal,
                     initial['drive'], original)
