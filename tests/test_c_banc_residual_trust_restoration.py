"""Generated CPU-only checks for E140 sparse residual trust restoration."""

# ruff: noqa: B023  # each loop executes its closures before the next iteration

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from tools import fit_c_banc_m3_residual_trust_restoration as e140


def data(edges=3, muscles=1, shift=0.):
    p = edges + 2 * muscles
    q = np.zeros((3, 99, 7))
    q[0, 0, 0] = .8 + .2 * shift
    jq = np.zeros((3, 99, 7, p))
    jq[0, 0, 0, 0] = .2
    ltm = np.zeros((3, 99))
    jltm = np.zeros((3, 99, p))
    loss_jac = np.zeros((3, p))
    loss_jac[:, 0] = [.3, -.1, .2]
    return {'delta': np.r_[np.array([shift]), np.zeros(edges - 1)],
        'gain': np.zeros(muscles), 'log_tau': np.zeros(muscles),
        'losses': np.array([1.1 + .3 * shift, 1. - .1 * shift,
                            1.02 + .2 * shift]),
        'loss_jac': loss_jac, 'q_error': q, 'ltm_error': ltm,
        'q_jac': jq, 'ltm_jac': jltm}


@pytest.mark.parametrize(('edges', 'muscles', 'parameters'),
                         [(240, 16, 272), (255, 15, 285), (285, 16, 317)])
def test_sparse_residual_matches_original_math(edges, muscles, parameters):
    value = e140.augmented_residual_jac(
        data(edges, muscles), 1.05, 1.05 / .99, edges, muscles)
    assert value['jacobian'].shape == (4761, parameters)
    assert value['jacobian'].format == 'csr'
    assert value['gradient'].shape == (parameters,)
    assert value['value'] == pytest.approx(
        .5 * value['residual'] @ value['residual'], abs=1e-14)
    assert np.all(value['residual'] <= 0)


def test_sparse_gradient_generated_directional_difference():
    center = e140.augmented_residual_jac(data(), 1.05, 1.05 / .99, 3, 1)
    step = 1e-6
    plus = e140.augmented_residual_jac(
        data(shift=step), 1.05, 1.05 / .99, 3, 1)
    minus = e140.augmented_residual_jac(
        data(shift=-step), 1.05, 1.05 / .99, 3, 1)
    assert center['gradient'][0] == pytest.approx(
        (plus['value'] - minus['value']) / (2 * step), abs=1e-7)


def test_zero_residual_row_has_zero_jacobian(monkeypatch):
    p = 5
    c = np.ones(4761)
    c[3] = -2.
    c[4] = 0.
    jac = np.ones((4761, p + 1))

    def constraints(*_args):
        return c, jac, np.zeros((3, 4))

    monkeypatch.setattr(e140.fit, 'constraint_values', constraints)
    answer = e140.augmented_residual_jac(data(), 2., 2. / .99, 3, 1)
    assert answer['residual'][0] == -2
    assert answer['residual'][1] == 0
    assert answer['jacobian'].getrow(0).nnz == p
    assert answer['jacobian'].getrow(1).nnz == 0


def test_large_multirow_gradient_accepts_float64_accumulation_order(monkeypatch):
    p = 5
    generator = np.random.default_rng(140)
    c = np.r_[np.ones(3), -np.linspace(.25, 250., 4758)]
    jac = generator.normal(size=(4761, p + 1)) * 1e6

    def constraints(*_args):
        return c, jac, np.zeros((3, 4))

    monkeypatch.setattr(e140.fit, 'constraint_values', constraints)
    answer = e140.augmented_residual_jac(data(), 2., 2. / .99, 3, 1)
    dense = e140.e139.restoration_value_jac(data(), 2., 2. / .99, 3, 1)
    difference = np.max(np.abs(answer['gradient'] - dense['gradient']))
    assert answer['active_count'] == 4758
    assert np.linalg.norm(answer['gradient']) > 1e9
    assert difference > 1e-13


def cache(evaluate, start=None, bounds=None, on_start=lambda *_: None,
          on_complete=lambda *_: None, maximum=18, clock=lambda: 0.):
    start = np.zeros(2) if start is None else np.asarray(start, np.float64)
    bounds = (np.tile([-2., 2.], (len(start), 1)) if bounds is None else
              np.asarray(bounds, np.float64))
    return e140.ResidualCache(evaluate, start, bounds, on_start, on_complete,
        max_unique=maximum, leg_deadline=55, block_deadline=165,
        internal_deadline=180, clock=clock)


def payload(residual, jacobian, qualifies=False):
    return {'residual': np.asarray(residual, np.float64),
        'jacobian': csr_matrix(np.asarray(jacobian, np.float64)),
        'qualifies': qualifies}


def test_fun_jac_share_cache_and_return_defensive_copies():
    calls = []
    subject = cache(lambda x, i: (calls.append((i, x.copy())) or
        payload(x - 1, np.eye(2))))
    x = np.array([.2, .3])
    fun = subject.fun(x)
    jac = subject.jac(x)
    fun[:] = 99
    jac.data[:] = 99
    assert np.array_equal(subject.fun(x), x - 1)
    assert np.array_equal(subject.jac(x).toarray(), np.eye(2))
    assert len(calls) == 1 and subject.completed == 1
    assert subject.hits == 3


def test_first_qualifier_completion_then_terminal():
    completed = []
    subject = cache(lambda _x, i: payload([1., 0.], np.eye(2), i == 2),
        on_complete=lambda _c, _d, i: completed.append(i))
    subject.fun(np.zeros(2))
    point = np.array([.1, 0.])
    with pytest.raises(e140.BoundedStop,
                       match='first_guarded_selector_feasible') as stopped:
        subject.fun(point)
    for vector in (point, np.ones(2)):
        with pytest.raises(e140.BoundedStop) as again:
            subject.jac(vector)
        assert again.value is stopped.value
    assert completed == [1, 2] and subject.selected_index == 2


def test_completion_failure_removes_payload_and_latches():
    error = OSError('completion receipt')
    subject = cache(lambda _x, _i: payload([1., 0.], np.eye(2)),
        on_complete=lambda *_: (_ for _ in ()).throw(error))
    with pytest.raises(OSError) as failure:
        subject.fun(np.zeros(2))
    assert failure.value is error
    assert subject.cache == {} and subject.completed == 0
    with pytest.raises(OSError) as repeated:
        subject.fun(np.ones(2))
    assert repeated.value is error


def test_real_least_squares_smoke_counts_bounds_move_and_latches():
    entries = []

    def evaluate(x, index):
        entries.append((index, x.copy()))
        return payload(x - .5, np.eye(2))

    subject = cache(evaluate, start=np.zeros(2),
                    bounds=np.tile([0., 2.], (2, 1)))
    leg = {'optimizer': {'status': 'not_requested', 'intent_persisted': False,
        'requested': 0, 'returned': 0, 'bounded_stop': 0,
        'failed_unknown': 0}}
    result, reason = e140.run_optimizer(subject, leg, lambda: None)
    assert ((result is not None and result.status >= 0) or
            (result is None and reason == 'leg_unique_cap'))
    assert len(entries) >= 2
    assert np.array_equal(entries[0][1], np.zeros(2))
    assert not np.array_equal(entries[1][1], entries[0][1])
    assert leg['optimizer']['requested'] == 1
    assert leg['optimizer']['returned'] == int(result is not None)
    assert leg['optimizer']['bounded_stop'] == int(result is None)
    terminal = subject.terminal_error
    with pytest.raises(e140.BoundedStop) as repeated:
        e140.run_optimizer(subject, leg, lambda: None)
    assert repeated.value is terminal


def test_optimizer_request_and_return_failures_terminal():
    for stage in ('intent', 'request', 'returned'):
        subject = cache(lambda x, _i: payload(x - 1, np.eye(2)))
        leg = {'optimizer': {'status': 'not_requested',
            'intent_persisted': False, 'requested': 0, 'returned': 0,
            'bounded_stop': 0, 'failed_unknown': 0}}
        error = OSError(stage)

        def persist():
            status = leg['optimizer']['status']
            if ((stage == 'intent' and status == 'intent') or
                    (stage == 'request' and status == 'in_flight_unknown') or
                    (stage == 'returned' and status == 'returned')):
                raise error

        fake = lambda _fun, x, **_kw: SimpleNamespace(success=True, status=1,
            message='ok', nfev=1, njev=1, cost=1., optimality=1., x=x)
        with pytest.raises(OSError) as failure:
            e140.run_optimizer(subject, leg, persist, solver=fake)
        assert failure.value is error
        for operation in (lambda: subject.fun(np.zeros(2)),
                lambda: subject.jac(np.ones(2)),
                lambda: e140.run_optimizer(subject, leg, persist, solver=fake)):
            with pytest.raises(OSError) as repeated:
                operation()
            assert repeated.value is error
        if stage == 'returned':
            assert leg['optimizer']['requested'] == leg['optimizer']['returned'] == 1


def test_solver_error_terminal_preserves_identity():
    subject = cache(lambda x, _i: payload(x - 1, np.eye(2)))
    leg = {'optimizer': {'status': 'not_requested', 'intent_persisted': False,
        'requested': 0, 'returned': 0, 'bounded_stop': 0,
        'failed_unknown': 0}}
    error = ArithmeticError('solver')
    entries = []

    def solver(*_args, **_kwargs):
        entries.append(1)
        raise error

    with pytest.raises(ArithmeticError) as failure:
        e140.run_optimizer(subject, leg, lambda: None, solver=solver)
    assert failure.value is error
    with pytest.raises(ArithmeticError) as repeated:
        e140.run_optimizer(subject, leg, lambda: None, solver=solver)
    assert repeated.value is error and entries == [1]


def test_residual_raw_round_trip_and_byte_size(tmp_path):
    answer = e140.augmented_residual_jac(data(), 1.05, 1.05 / .99, 3, 1)
    path = tmp_path / 'residual.npz'
    digest, size = e140.save_residual_raw(path, answer)
    assert digest == e140.fit.sha256(path) and size == path.stat().st_size
    with np.load(path, allow_pickle=False) as raw:
        rebuilt = csr_matrix((raw['csr_data'], raw['csr_indices'],
            raw['csr_indptr']), shape=tuple(raw['csr_shape']))
        assert np.array_equal(rebuilt.toarray(), answer['jacobian'].toarray())
        assert np.array_equal(raw['residual'], answer['residual'])


def test_dual_raw_byte_accounting_preserves_over_cap_file(tmp_path):
    answer = e140.augmented_residual_jac(data(), 1.05, 1.05 / .99, 3, 1)
    path = tmp_path / 'residual.npz'
    ledger = {'raw_bytes': 7}
    row = {}
    with pytest.raises(ValueError, match='dual-raw cap'):
        e140.account_residual_raw(ledger, row, path, answer, 8)
    assert path.exists() and row['residual_raw_saved'] is True
    assert row['residual_raw_sha256'] == e140.fit.sha256(path)
    assert ledger['raw_bytes'] == 7 + path.stat().st_size


def test_completion_receipt_failure_marks_post_return_and_keeps_dual_raw(
        tmp_path):
    losses = np.array([1., .9, .8])
    ratios = np.zeros((3, 4))
    parts = np.zeros((3, 4))
    raw_path = tmp_path / 'raw.npz'
    residual_path = tmp_path / 'residual.npz'
    np.savez(raw_path, losses=losses, ratios=ratios, parts=parts)
    np.savez(residual_path, residual=np.zeros(4761))
    row = {'index': 1, 'status': 'raw_saved', 'raw_saved': True,
        'solve_requested': True, 'solve_returned': True,
        'post_return_error': False, 'raw_path': str(raw_path),
        'raw_sha256': e140.fit.sha256(raw_path), 'losses': losses.tolist(),
        'ratios': ratios.tolist(), 'parts': parts.tolist(), 'prior': 0.,
        'true_worst_objective': 1., 'selector_feasible': True,
        'residual_raw_saved': True, 'residual_raw_path': str(residual_path),
        'residual_raw_sha256': e140.fit.sha256(residual_path)}
    ledger = {'original_evaluations': [row]}
    leg = {}
    subject = SimpleNamespace(started=1, completed=1, hits=0,
        selected_index=None, fun_calls=1, jac_calls=1)
    primary = OSError('completion ledger')
    attempts = []

    def persist():
        attempts.append(row['status'])
        if len(attempts) == 1:
            raise primary

    with pytest.raises(OSError) as failure:
        e140.complete_receipt(ledger, leg, subject, row, persist)
    assert failure.value is primary
    assert attempts == ['raw_saved', 'failed']
    assert row['post_return_error'] is True
    assert row['raw_saved'] is row['residual_raw_saved'] is True


def test_compact_writer_calls_jsonable_once_and_accounts(tmp_path, monkeypatch):
    calls = []
    original = e140.fit.jsonable

    def counted(value):
        calls.append(1)
        monkeypatch.setattr(e140.fit, 'jsonable', original)
        return original(value)

    monkeypatch.setattr(e140.fit, 'jsonable', counted)
    writer = e140.CompactWriter()
    path = tmp_path / 'value.json'
    writer.save(path, {'array': np.arange(3)})
    assert calls == [1]
    assert json.loads(path.read_text()) == {'array': [0, 1, 2]}
    assert writer.metadata()['calls'] == writer.metadata()['jsonable_calls'] == 1
    assert writer.metadata()['bytes'] == path.stat().st_size


def test_compact_writer_failed_write_accounts_attempt_conversion_and_time(
        tmp_path):
    ticks = iter((10., 10.25))
    writer = e140.CompactWriter(clock=lambda: next(ticks))
    with pytest.raises(FileNotFoundError) as failure:
        writer.save(tmp_path / 'missing' / 'value.json', {'value': 1})
    metadata = writer.metadata()
    assert failure.value.filename.endswith('value.json')
    assert metadata == {'attempts': 1, 'calls': 0, 'failures': 1,
        'wall_s': .25, 'bytes': 0, 'jsonable_calls': 1,
        'conversions_completed': 1, 'format': 'compact_sorted_utf8'}


@pytest.mark.parametrize('metadata_failure', [False, True])
def test_tiny_failure_keeps_engine_after_and_primary(tmp_path, monkeypatch,
                                                     metadata_failure):
    primary = RuntimeError('tiny solve')
    primary.response_progress = {'kernel_entries': 1, 'complete': False}
    secondary = OSError('metadata')

    class Engine:
        def __init__(self, _problem):
            self.metadata_calls = 0

        def metadata(self):
            self.metadata_calls += 1
            if metadata_failure and self.metadata_calls == 2:
                raise secondary
            return {'entries': self.metadata_calls - 1}

        def solve(self, *_args, **_kwargs):
            raise primary

    monkeypatch.setattr(e140.fit, 'synthetic_tiny_setup_problem', dict)
    monkeypatch.setattr(e140.tau_module, 'CudaAdaptationTimescaleEngine',
                        Engine)
    ledger = {}
    with pytest.raises(RuntimeError) as failure:
        e140.run_tiny_setup(tmp_path, ledger, e140.CompactWriter())
    assert failure.value is primary
    row = ledger['synthetic_tiny']
    assert row['response_progress'] == primary.response_progress
    if metadata_failure:
        assert row['engine_after_error'] == repr(secondary)
        assert repr(secondary) in primary.__notes__[0]
    else:
        assert row['engine_after'] == {'entries': 1}


def test_preflight_rejects_bad_sha_before_file_read(tmp_path):
    with pytest.raises(ValueError, match='Exact E140 freeze SHA'):
        e140.reviewed_inputs(tmp_path / 'missing', '0' * 64,
                             tmp_path / 'review', '0' * 64)
