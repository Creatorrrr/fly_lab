"""Focused E108 contract tests; no shadow body or neural model is built."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import OptimizeResult

from tools.fit_c_banc_m3_block_constrained_recruitment import (
    BlockLimit,
    NativeCache,
    accepted_review,
    check_static_cross_jacobian,
    choose_records,
    close_shadow_body,
    json_safe,
    leg_partition,
    main,
    optimizer_record,
    resource_screen,
    shadow_body_lifetime,
    solve_block,
    startup_ratios,
    strict_feasible,
    verify_metadata_weights,
)


class BlockConstrainedRecruitmentTests(unittest.TestCase):
    def test_actual_metadata_float32_widening_is_exact_and_rejects_changes(self):
        native = np.zeros((391, 49), dtype=np.float32)
        native[0, 0] = -1.234375
        self.assertTrue(verify_metadata_weights(native.astype(np.float64), native))
        changed = native.copy()
        changed[0, 0] = np.nextafter(changed[0, 0], np.float32(0))
        with self.assertRaises(AssertionError):
            verify_metadata_weights(native.astype(np.float64), changed)
        with self.assertRaises(ValueError):
            verify_metadata_weights(native.astype(np.float64), native.astype(np.float64))
        bad = native.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            verify_metadata_weights(native.astype(np.float64), bad)

    def test_reviews_require_explicit_acceptance_and_empty_defects(self):
        accepted_review({'accepted_as_valid_failed_body_diagnostic': True,
                         'blocking_defects': []},
                        'accepted_as_valid_failed_body_diagnostic')
        for review in ({'accepted_as_valid_failed_body_diagnostic': 1,
                        'blocking_defects': []},
                       {'accepted_as_valid_failed_body_diagnostic': True,
                        'blocking_defects': ['problem']},
                       {'accepted_as_valid_failed_body_diagnostic': True}):
            with self.assertRaises(ValueError):
                accepted_review(review, 'accepted_as_valid_failed_body_diagnostic')

    def test_partition_rejects_one_shared_motor_and_edge(self):
        counts = (69, 63, 63, 70, 63, 63)
        edge_counts = (273, 281, 240, 238, 255, 285)
        moment = np.zeros((42, 96))
        pool = np.zeros((96, 391))
        ltm = np.arange(6)
        edge_rows = []
        begin = 0
        for leg, (mn_count, edge_count) in enumerate(zip(counts, edge_counts)):
            rows = np.arange(begin, begin+mn_count)
            moment[7*leg, leg] = 1
            pool[leg, rows] = 1
            edge_rows.extend(np.resize(rows, edge_count))
            begin += mn_count
        edge_rows = np.asarray(edge_rows)
        motor_mask, edge_mask = leg_partition(moment, pool, ltm, edge_rows)
        self.assertEqual(motor_mask.sum(), 391)
        self.assertEqual(edge_mask.sum(), 1572)
        self.assertTrue(np.all(edge_mask.sum(axis=0) == 1))
        leg_partition(moment, pool, ltm, edge_rows,
                      expected_motor=motor_mask, expected_edge=edge_mask)
        with self.assertRaises(ValueError):
            leg_partition(moment, pool, ltm, edge_rows,
                          expected_motor=np.roll(motor_mask, 1, axis=0))
        pool[1, 0] = 1
        with self.assertRaisesRegex(ValueError, 'sharing'):
            leg_partition(moment, pool, ltm, edge_rows)

    def test_static_foot_cross_leg_jacobian_must_be_zero(self):
        jac = np.zeros((2, 6, 3, 42))
        for leg in range(6):
            jac[:, leg, :, 7*leg] = 1
        self.assertEqual(check_static_cross_jacobian(jac), 0.)
        jac[0, 0, 0, 7] = 1e-4
        with self.assertRaisesRegex(ValueError, 'cross-leg'):
            check_static_cross_jacobian(jac)

    def test_original_four_inequalities_and_strict_margin(self):
        q = np.zeros((99, 7))
        tendon = np.zeros(99)
        q[0, 0] = .06
        ratios = startup_ratios(q, tendon, np.zeros_like(q), np.zeros_like(tendon))
        self.assertLess(ratios[0], 1)
        self.assertGreater(ratios[1], 1)
        self.assertFalse(strict_feasible(ratios))
        edge = np.array([(1-1e-4)**2]*4)
        self.assertTrue(strict_feasible(edge))
        self.assertFalse(strict_feasible(np.array([1., 0., 0., 0.])))
        self.assertFalse(strict_feasible(np.array([np.nan, 0., 0., 0.])))
        with self.assertRaises(ValueError):
            startup_ratios(q[:98], tendon, np.zeros_like(q[:98]), np.zeros_like(tendon))

    def test_cache_is_bit_exact_and_counts_before_native_call(self):
        starts = []
        seen = []
        def evaluate(x, index):
            seen.append((index, x.copy()))
            return {'objective': float(x[0])}
        cache = NativeCache(evaluate, 1, max_calls=2,
            deadline=time.perf_counter()+10, total_deadline=time.perf_counter()+10,
            on_start=lambda n, x: starts.append((n, x.copy())))
        self.assertEqual(cache.get(np.array([0.]))['objective'], 0.)
        cache.get(np.array([0.]))
        cache.get(np.array([-0.]))  # numerically equal, different float64 bits
        self.assertEqual(cache.calls, 2)
        self.assertEqual(cache.hits, 1)
        self.assertEqual([n for n, _ in starts], [1, 2])
        self.assertEqual([n for n, _ in seen], [1, 2])
        with self.assertRaises(BlockLimit):
            cache.get(np.array([1.]))
        self.assertEqual(cache.calls, 2)

    def test_native_selection_ignores_training_score_and_requires_feasibility(self):
        records = [
            {'objective': 2., 'ratios': [.1]*4, 'smooth_loss': -100.},
            {'objective': 1., 'ratios': [1.1, .1, .1, .1], 'smooth_loss': -200.},
            {'objective': 1.5, 'ratios': [.2]*4, 'smooth_loss': 200.},
        ]
        best_any, best_feasible = choose_records(records)
        self.assertIs(best_any, records[1])
        self.assertIs(best_feasible, records[2])
        self.assertIsNone(choose_records([records[1]])[1])
        self.assertEqual(choose_records([]), (None, None))

    def test_nonfinite_optimizer_evidence_has_null_and_counts(self):
        failed = OptimizeResult(success=False, status=2, message='ABNORMAL',
            nit=1, nfev=3, njev=2, fun=np.nan, jac=np.array([np.nan, np.inf]))
        record = optimizer_record(failed, 'scipy_failure')
        self.assertEqual(record['stop'], 'scipy_nonfinite_result')
        self.assertIsNone(record['objective'])
        self.assertIsNone(record['gradient_max_abs'])
        self.assertEqual(record['gradient_nonfinite_counts']['nan'], 1)
        self.assertEqual(record['gradient_nonfinite_counts']['positive_infinity'], 1)
        self.assertEqual(record['nfev'], 3)
        self.assertIsNone(json_safe({'a': np.nan})['a'])
        json.dumps(json_safe(record), allow_nan=False)

    def test_actual_scipy_callsite_persists_nonfinite_return_before_raise(self):
        failed = OptimizeResult(success=False, status=2, message='BAD RETURN',
            nit=4, nfev=7, njev=5, fun=np.nan, jac=np.array([np.inf]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'optimizer.json'
            with (patch('tools.fit_c_banc_m3_block_constrained_recruitment.minimize',
                        return_value=failed),
                  self.assertRaisesRegex(ValueError, 'preserved record')):
                solve_block(None, np.zeros(1), 1, path)
            saved = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(saved['message'], 'BAD RETURN')
            self.assertEqual(saved['nfev'], 7)
            self.assertIsNone(saved['objective'])
            self.assertEqual(saved['stop'], 'scipy_nonfinite_result')

    def test_resource_unknown_started_cost_and_prior_shared_budget(self):
        ledger = {'prior_shared_wall_s': 8., 'whole_wall_s': 10.,
                  'fit_start_elapsed_s': 1., 'fit_wall_s': None,
                  'leg_started_evaluations': {'lf': 1},
                  'leg_wall_s': {'lf': None}}
        self.assertEqual(resource_screen(ledger), 'limit_exceeded_or_unverifiable')
        ledger['fit_start_elapsed_s'] = None
        self.assertEqual(resource_screen(ledger), 'limit_exceeded_or_unverifiable')
        ledger['fit_start_elapsed_s'] = 1.
        ledger['fit_wall_s'] = 8.
        self.assertEqual(resource_screen(ledger), 'limit_exceeded_or_unverifiable')
        ledger['leg_wall_s']['lf'] = 7.
        self.assertEqual(resource_screen(ledger), 'within_limits')
        ledger['whole_wall_s'] = 217.
        self.assertEqual(resource_screen(ledger), 'within_limits')
        ledger['whole_wall_s'] = 217.01
        self.assertEqual(resource_screen(ledger), 'limit_exceeded')
        ledger['whole_wall_s'] = 10.
        ledger['leg_wall_s']['lf'] = 24.01
        self.assertEqual(resource_screen(ledger), 'limit_exceeded')

    def test_actual_error_finally_records_fit_leg_time_and_prior_charge(self):
        def fail_after_start(_root, _graph, _out, started, ledger):
            ledger['fit_start_elapsed_s'] = time.perf_counter()-started
            ledger['active_leg'] = 'lf'
            ledger['active_leg_start_elapsed_s'] = time.perf_counter()-started
            ledger['leg_started_evaluations']['lf'] = 1
            raise RuntimeError('mocked native failure')
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'v2'
            with self.assertRaisesRegex(ValueError, 'exactly 8s'):
                main(Path(folder), Path(folder), out, prior_shared_wall_s=0.)
            self.assertFalse(out.exists())
            with (patch('tools.fit_c_banc_m3_block_constrained_recruitment._run',
                        side_effect=fail_after_start),
                  self.assertRaisesRegex(RuntimeError, 'mocked native failure')):
                main(Path(folder), Path(folder), out, prior_shared_wall_s=8.)
            ledger = json.loads((out/'execution-ledger.json').read_text(encoding='utf-8'))
            error = json.loads((out/'execution-error.json').read_text(encoding='utf-8'))
            self.assertGreaterEqual(ledger['fit_wall_s'], 0.)
            self.assertGreaterEqual(ledger['leg_wall_s']['lf'], 0.)
            self.assertEqual(ledger['prior_shared_wall_s'], 8.)
            self.assertEqual(error['message'], 'mocked native failure')
            self.assertEqual(error['resource_screen'], ledger['resource_screen'])

    def test_actual_shadow_cleanup_closes_even_when_exact_raises(self):
        class FakeBody:
            closed = False
            def snapshot(self):
                return {'clock': 1}
            def close(self):
                self.closed = True
        with tempfile.TemporaryDirectory() as folder:
            body = FakeBody()
            def reject(_before, _after, _label):
                raise AssertionError('changed clock')
            with self.assertRaisesRegex(AssertionError, 'changed clock'):
                close_shadow_body(body, {'clock': 0}, Path(folder), exact_check=reject)
            self.assertTrue(body.closed)
            saved = json.loads((Path(folder)/'shadow-body-final.json').read_text(encoding='utf-8'))
            self.assertFalse(saved['snapshot_exact'])
            self.assertTrue(saved['closed'])
            pending = RuntimeError('original technical failure')
            body2 = FakeBody()
            close_shadow_body(body2, {'clock': 0}, Path(folder),
                              pending_error=pending, exact_check=reject)
            self.assertTrue(body2.closed)
            self.assertIn('changed clock', pending.__notes__[0])

    def test_actual_shadow_lifetime_closes_on_initial_snapshot_failure(self):
        class FakeBody:
            closed = False

            def snapshot(self):
                raise RuntimeError('initial snapshot failed')

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as folder:
            body = FakeBody()
            with (
                self.assertRaisesRegex(RuntimeError, 'initial snapshot failed'),
                shadow_body_lifetime(None, Path(folder), body_factory=lambda _: body),
            ):
                self.fail('The body must not be exposed before its initial snapshot')
            self.assertTrue(body.closed)
            saved = json.loads((Path(folder)/'shadow-body-final.json').read_text(encoding='utf-8'))
            self.assertIsNone(saved['snapshot_exact'])
            self.assertTrue(saved['closed'])


if __name__ == '__main__':
    unittest.main()
