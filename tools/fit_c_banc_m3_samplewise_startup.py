"""E119 bounded two-context fit with every signed startup sample constrained."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.analyze_c_banc_m3_samplewise_startup import (
    derivative_bank,
)
from tools.analyze_c_banc_m3_samplewise_startup import (
    reviewed_inputs as e118_inputs,
)
from tools.banc_m3_context_response import ContextResponse
from tools.banc_m3_torch_foot_geometry import TorchFootGeometry
from tools.fit_c_banc_m3_context_trust import (
    GEOMETRY_HASH,
    INNER,
    LEGS,
    OPTIMIZER_CAP,
    ORIGINAL,
    SELECTOR_CAP,
    LegStage,
    candidate_weights,
    choose_records,
    delta_key,
    efficacy_boundary_roundoff,
    frozen_output_metrics,
    native_rounding_comparison,
)
from tools.fit_c_banc_m3_dense_foot_recruitment import selected_actual_float32
from tools.verify_c_banc_m3_e117_nonlinear_steps import (
    load_npz,
    require,
    rescue_raw,
    save_json,
    sha256,
)

DESIGN_SHA = 'e3e32a18a03725bc9f855921dd96861c3f854991883dd6e36ce12e693429f5de'
FREEZE_SHA = '04ac16b0d67124b82c4bcea4dce478d16364bb8ab6f308bfc1220d492f01ac82'
CONSTRAINT_ROWS = 3174
UNIQUE_LIMIT = 100
TOTAL_UNIQUE_LIMIT = 600
LEG_WALL_S = 24.
LEG_RESERVE_S = 3.
FIT_WALL_S = 150.
PROCESS_WALL_S = 175.
PRIOR_PROCESS_S = 717.
SHARED_PROCESS_S = 917.


class LegSoftStop(RuntimeError):
    """A recorded per-leg cap; continue independent legs within hard limits."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def reviewed_inputs(root):
    """Pinned read-only preflight without geometry, response or derivative."""
    root = Path(root)
    require(sha256(ROOT/'docs/BANC_M3_E119_SAMPLEWISE_FIT_DESIGN_20260920.md')
            == DESIGN_SHA and sha256(root/'e119-frozen-design.json') == FREEZE_SHA,
            'E119 design/freeze changed')
    freeze = json.loads((root/'e119-frozen-design.json').read_text(
        encoding='utf-8'))
    require(freeze['actual_execution_authorized'] is False and
            freeze['constraints_per_leg'] == CONSTRAINT_ROWS and
            freeze['leg_unique_delta_limit'] == UNIQUE_LIMIT and
            freeze['total_unique_delta_limit'] == TOTAL_UNIQUE_LIMIT and
            freeze['solver'] == 'SLSQP' and
            freeze['solver_maxiter'] == 100 and
            freeze['solver_ftol'] == 1e-9 and
            freeze['leg_wall_limit_s'] == LEG_WALL_S and
            freeze['leg_finalize_reserve_s'] == LEG_RESERVE_S and
            freeze['fit_wall_limit_s'] == FIT_WALL_S and
            freeze['internal_wall_limit_s'] == PROCESS_WALL_S and
            freeze['prior_conservative_process_s'] == PRIOR_PROCESS_S and
            freeze['shared_process_limit_s'] == SHARED_PROCESS_S and
            freeze['warm_loss_gradient_parity_max_abs'] == 1e-8 and
            freeze['warm_startup_jacobian_parity_max_abs'] == 1e-9 and
            freeze['optimizer_cap'] == OPTIMIZER_CAP and
            freeze['selector_cap'] == SELECTOR_CAP and
            freeze['inner_startup_caps'] == INNER.tolist() and
            freeze['original_startup_caps'] == ORIGINAL.tolist(),
            'E119 frozen scope/resources changed')
    # The freeze separates per-leg and final conditional pair limits.
    require(freeze['leg_conditional_pair_limit'] == 1200 and
            freeze['leg_conditional_seconds_limit'] == 12000 and
            freeze['derivative_bank_limit'] == TOTAL_UNIQUE_LIMIT and
            freeze['derivative_linear_filter_limit'] == TOTAL_UNIQUE_LIMIT and
            freeze['derivative_endpoint_filter_limit'] == TOTAL_UNIQUE_LIMIT and
            freeze['final_full_conditional_pairs_limit'] == 2 and
            freeze['final_full_conditional_seconds_limit'] == 20 and
            freeze['actual_neural_s'] == freeze['actual_physical_s'] == 0 and
            freeze['F6_exposed'] is False,
            'E119 original-resource boundary changed')
    for relative, expected in freeze['input_sha256'].items():
        require(sha256(root/relative) == expected,
                f'E119 frozen input changed: {relative}')
    for relative, expected in freeze['source_sha256'].items():
        require(sha256(ROOT/relative) == expected,
                f'E119 reused source changed: {relative}')
    review = json.loads((root/'e118-samplewise-startup/parent-final-review.json')
                        .read_text(encoding='utf-8'))
    require(review['accepted_as_valid_derivative_diagnostic'] is True and
            review['blocking_defects'] == [],
            'E118 independent derivative validation missing')
    data = e118_inputs(root)
    selected = load_npz(root/'e115-context-trust-fit/selected-delta.npz')
    require(selected['delta'].shape == (1572,) and
            np.array_equal(selected['edge_ids'], data['fixture']['edge_ids']) and
            np.array_equal(selected['edge_rows'], data['fixture']['edge_rows'])
            and np.array_equal(selected['edge_cols'], data['fixture']['edge_cols'])
            and np.isfinite(selected['delta']).all() and
            np.all(selected['delta'] >= data['bounds'][:, 0]) and
            np.all(selected['delta'] <= data['bounds'][:, 1]),
            'E115 warm start/bounds/edge mapping changed')
    jacobians = {name: load_npz(root/f'e118-samplewise-startup/'
                              f'jacobian-{name}.npz') for name in LEGS}
    for leg, name in enumerate(LEGS):
        actual = jacobians[name]
        positions = np.flatnonzero(data['edge_mask'][leg])
        require(np.array_equal(actual['edge_positions'], positions) and
                np.array_equal(actual['edge_ids'], data['fixture'][
                    'edge_ids'][positions]) and
                actual['q_jacobian'].shape == (2, 99, 7, len(positions)) and
                actual['tendon_jacobian'].shape == (2, 99, len(positions)),
                f'E118 original leg derivative axes changed: {name}')
    data.update(root=root, e119_freeze=freeze, selected=selected,
                e118_jacobians=jacobians)
    return data


def factor_jacobians(cached):
    """Reconstruct one leg's signed sample Jacobians from compact factors."""
    v = np.asarray(cached['v'], dtype=np.float64)
    qco = np.asarray(cached['q_coefficient'], dtype=np.float64)
    zco = np.asarray(cached['ltm_coefficient'], dtype=np.float64)
    gain = np.asarray(cached['ltm_gain'], dtype=np.float64)
    if v.ndim != 3 or v.shape[:2] != (2, 99) or \
            qco.shape != (7, v.shape[-1]) or zco.shape != (v.shape[-1],) or \
            gain.shape != (2, 99):
        raise ValueError('Aligned compact E119 startup factors required')
    return (v[:, :, None, :]*qco[None, None],
            v*gain[:, :, None]*zco[None, None])


def samplewise_terms(delta, t, cached):
    """Exact 3174 signed-sample/RMS/epigraph inequalities and Jacobian."""
    delta = np.asarray(delta, dtype=np.float64)
    losses = np.asarray(cached['losses'], dtype=np.float64)
    loss_jac = np.asarray(cached['loss_gradient'], dtype=np.float64)
    q = np.asarray(cached['q_error'], dtype=np.float64)
    z = np.asarray(cached['tendon_error'], dtype=np.float64)
    n = len(delta)
    if delta.ndim != 1 or n == 0 or not np.isfinite(delta).all() or \
            not np.isfinite(t) or losses.shape != (2,) or \
            loss_jac.shape != (2, n) or q.shape != (2, 99, 7) or \
            z.shape != (2, 99) or not all(np.isfinite(x).all() for x in
                (losses, loss_jac, q, z)):
        raise ValueError('Finite E119 loss/startup/epigraph data required')
    jq, jz = factor_jacobians(cached)
    if jq.shape != (2, 99, 7, n) or jz.shape != (2, 99, n):
        raise ValueError('E119 factor edge axis mismatch')
    root_cap = math.sqrt(OPTIMIZER_CAP)
    prior = 6e-5*float(np.sum(delta*delta))/1572.
    objective_jac = np.r_[1.2e-4*delta/1572., 1.]
    q_rms = np.mean(q*q, axis=(1, 2))/INNER[0]**2
    z_rms = np.mean(z*z, axis=1)/INNER[2]**2
    inequalities = np.r_[t-losses,
        (root_cap-q/INNER[1]).ravel(order='C'),
        (root_cap+q/INNER[1]).ravel(order='C'),
        (root_cap-z/INNER[3]).ravel(order='C'),
        (root_cap+z/INNER[3]).ravel(order='C'),
        OPTIMIZER_CAP-q_rms, OPTIMIZER_CAP-z_rms]
    jac = np.zeros((CONSTRAINT_ROWS, n+1), dtype=np.float64)
    jac[:2, :n] = -loss_jac
    jac[:2, n] = 1.
    offset = 2
    for block in (-jq/INNER[1], jq/INNER[1],
                  -jz/INNER[3], jz/INNER[3]):
        count = int(np.prod(block.shape[:-1]))
        jac[offset:offset+count, :n] = block.reshape(count, n, order='C')
        offset += count
    jac[offset:offset+2, :n] = -2*np.einsum('ctj,ctje->ce', q, jq)/(
        99*7*INNER[0]**2)
    offset += 2
    jac[offset:offset+2, :n] = -2*np.einsum('ct,cte->ce', z, jz)/(
        99*INNER[2]**2)
    offset += 2
    require(offset == CONSTRAINT_ROWS and inequalities.shape ==
            (CONSTRAINT_ROWS,), 'E119 signed constraint row count changed')
    return {'objective': float(t+prior), 'objective_jac': objective_jac,
        'inequalities': inequalities, 'inequality_jac': jac,
        'prior': prior,
        'true_worst_objective': float(np.max(losses)+prior),
        'epigraph_gaps': t-losses}


def selector_record(index, name, delta, cached):
    delta = np.asarray(delta, dtype=np.float64)
    losses = np.asarray(cached['losses'], dtype=np.float64)
    ratios = np.asarray(cached['ratios'], dtype=np.float64)
    prior = 6e-5*float(np.sum(delta*delta))/1572.
    return {'index': index, 'leg': name, 'delta': delta.tolist(),
        'losses': losses.tolist(), 'parts': cached['parts'].tolist(),
        'ratios': ratios.tolist(), 'prior': prior,
        'true_worst_objective': float(np.max(losses)+prior),
        'selector_feasible': bool(np.all(ratios <= SELECTOR_CAP)),
        'optimizer_startup_feasible': bool(np.all(ratios <= OPTIMIZER_CAP))}


def no_joint_result(selected, ledger):
    return {'status': 'no_joint_feasible_candidate',
        'robust_local_supported': False, 'candidate_adopted': False,
        'native_gait_tested': False, 'no_full_candidate_composed': True,
        'feasible_legs': [LEGS[i] for i, item in enumerate(selected)
                          if item is not None],
        'unfeasible_legs': [LEGS[i] for i, item in enumerate(selected)
                            if item is None], 'ledger': ledger}


def detach_available(tensors, raw, primary=None):
    for key, value in tensors.items():
        if key in raw or not isinstance(value, torch.Tensor):
            continue
        try:
            raw[key] = value.detach().cpu().numpy().copy()
        except Exception as secondary:
            if primary is None:
                raise
            primary.add_note(f'E119 secondary {key} materialization: {secondary}')


def materialize_feet(tensor, fixed_count):
    """Copy one already computed fixed-geometry tensor, never rerun geometry."""
    return tensor.detach().cpu().numpy().copy().reshape(2, fixed_count, 6, 3)


def rescue_available_feet(tensor, fixed_count, raw, primary, secondary_errors):
    if tensor is None or fixed_count is None or 'feet_fixed' in raw:
        return
    try:
        raw['feet_fixed'] = materialize_feet(tensor, fixed_count)
    except Exception as secondary:  # noqa: BLE001
        secondary_errors.append({'at': 'feet_materialization',
            'type': type(secondary).__name__, 'message': str(secondary)})
        primary.add_note(f'E119 secondary feet materialization: {secondary}')


class DeltaCache:
    """Only delta bytes identify one forward+bank; t is constraint algebra."""

    def __init__(self, evaluator, on_hit=lambda: None):
        self.evaluator = evaluator
        self.on_hit = on_hit
        self.values = {}

    def get_delta(self, delta):
        delta = np.asarray(delta, dtype=np.float64)
        key = delta_key(delta)
        if key not in self.values:
            self.values[key] = self.evaluator(delta.copy())
        else:
            self.on_hit()
        return self.values[key]

    def get(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1 or len(x) < 2:
            raise ValueError('E119 delta plus epigraph t required')
        return samplewise_terms(x[:-1], x[-1], self.get_delta(x[:-1]))


class LegFit:
    """One bounded leg: forward once, derivative bank once per unique delta."""

    def __init__(self, stage, data, out, ledger, persist, deadlines):
        self.stage, self.data, self.out = stage, data, out
        self.ledger, self.persist, self.deadlines = ledger, persist, deadlines
        self.records = []
        self.cache = DeltaCache(self.evaluate_delta, self.cache_hit)

    def cache_hit(self):
        name = self.stage.name
        self.ledger['cache_hits'][name] += 1
        self.deadlines(name, soft=False)
        self.persist()

    def response_materialized(self, index):
        name = self.stage.name
        self.ledger['leg_materialized'][name] = index
        self.ledger['leg_pairs_materialized'][name] = 2*index
        self.ledger['leg_conditional_s_materialized'][name] = 20.*index

    def evaluate_delta(self, delta):
        stage, data, ledger = self.stage, self.data, self.ledger
        name = stage.name
        self.deadlines(name)
        index = ledger['leg_started'][name]+1
        if index > UNIQUE_LIMIT:
            raise LegSoftStop('leg_unique_delta_limit')
        if sum(ledger['leg_started'].values()) >= TOTAL_UNIQUE_LIMIT:
            raise TimeoutError('E119 total600 unique-delta limit')
        bounds = data['bounds'][stage.edges]
        if delta.shape != (len(stage.edges),) or not np.isfinite(delta).all() or \
                np.any(delta < bounds[:, 0]) or np.any(delta > bounds[:, 1]):
            raise ValueError('Delta outside exact frozen trust bounds')
        # Complete the only host->GPU delta transfer before charging a call.
        parameter = torch.tensor(delta, dtype=torch.float64,
            device=stage.device, requires_grad=True)
        torch.cuda.synchronize() if stage.device.type == 'cuda' else None
        base = data['fixture']['weights'][1, stage.motor].copy()
        rows, cols = stage.rows.detach().cpu().numpy(), \
                     stage.cols.detach().cpu().numpy()
        local_w = base.copy()
        local_w[rows, cols] = base[rows, cols]*np.exp(delta)
        raw = {'delta': delta.copy(), 'bounds': bounds.copy(),
            'edge_positions': stage.edges.copy(),
            'edge_ids': data['fixture']['edge_ids'][stage.edges].copy(),
            'local_rows': rows.copy(), 'source_cols': cols.copy(),
            'local_candidate_weights': local_w,
            'time_s': data['fixture']['time_s'].copy(),
            'context_names': data['fixture']['context_names'].copy()}
        tensors = {}
        output = None
        ledger.update(active_leg=name, active_delta=delta.tolist(),
            active_delta_index=index, stage='leg_response')
        ledger['leg_started'][name] = index
        ledger['leg_pairs_started'][name] = 2*index
        ledger['leg_conditional_s_started'][name] = 20.*index
        try:
            self.persist()
        except BaseException as primary:
            ledger['leg_started'][name] = index-1
            ledger['leg_pairs_started'][name] = 2*(index-1)
            ledger['leg_conditional_s_started'][name] = 20.*(index-1)
            ledger['response_start_receipt_failed'] = {'leg': name,
                'index': index}
            rescue_raw(self.out/f'error-input-{name}-{index:03d}', raw,
                       primary, 'leg input')
            raise
        try:
            stage.last_response = None
            ledger['response_call_entered'][name] += 1
            output = stage.tensors(parameter)
            tensors.update({key: value for key, value in output.items()
                            if isinstance(value, torch.Tensor)})
            tensors.update({f'response_{key}': value for key, value in
                            output['response'].items()})
            if stage.device.type == 'cuda':
                torch.cuda.synchronize()
            self.response_materialized(index)
            ledger['stage'] = 'leg_response_materialized'
            self.persist()
            metric = output['losses']
            loss_grad = torch.autograd.grad(metric, parameter,
                grad_outputs=torch.eye(2, dtype=torch.float64,
                                       device=stage.device),
                is_grads_batched=True)[0]
            tensors['loss_gradient'] = loss_grad
            for key in ('parts', 'losses', 'ratios'):
                raw[key] = output[key].detach().cpu().numpy().copy()
            raw['loss_gradient'] = loss_grad.detach().cpu().numpy().copy()
            response = output['response']
            raw['q_error'] = (response['command'][:, 1:100]-stage.start_q).detach(
                ).cpu().numpy().copy()
            raw['tendon_error'] = (response['tendon'][:, 1:100, 0]-
                stage.start_tendon).detach().cpu().numpy().copy()
            candidate = candidate_weights(stage.base_w, stage.rows,
                stage.cols, parameter.detach())
            staged = {'source': stage.source[:, :100],
                'current': response['current'][:, :100].detach(),
                'weights': candidate.detach(),
                'rows': stage.rows, 'cols': stage.cols,
                'tau': stage.response.tau, 'a': stage.response.a,
                'threshold': stage.response.threshold,
                'cap': stage.response.cap}
            ledger['bank_started'][name] = index
            try:
                self.persist()
            except BaseException:
                ledger['bank_started'][name] = index-1
                ledger['bank_start_receipt_failed'] = {'leg': name,
                    'index': index}
                raise

            def record(key, value):
                tensors[f'bank_{key}'] = value

            def on_filter(filter_name, state):
                ledger[f'{filter_name}_filter_{state}'][name] += 1

            ledger['bank_call_entered'][name] += 1
            with torch.no_grad():
                bank = derivative_bank(staged, record=record,
                                       on_filter=on_filter)
            tensors.update({f'bank_{key}': value for key, value in bank.items()})
            v = bank['activation_derivative'][:, 1:100].detach().cpu().numpy(
                ).copy()
            q_coeff = (.02*(stage.response.moment @ stage.response.pool)[
                :, stage.rows]).detach().cpu().numpy().copy()
            ltm_index = int(stage.response.ltm[0])
            z_coeff = (.02*stage.response.pool[ltm_index, stage.rows]).detach(
                ).cpu().numpy().copy()
            activation = response['muscle_activation'][:, 1:100,
                                                       ltm_index].detach()
            gain = (1-torch.tanh(activation/10)**2).cpu().numpy().copy()
            torch.cuda.synchronize() if stage.device.type == 'cuda' else None
            raw.update(v=v, q_coefficient=q_coeff, ltm_coefficient=z_coeff,
                       ltm_gain=gain)
            ledger['bank_materialized'][name] = index
            ledger['linear_filter_completed'][name] = \
                ledger['linear_filter_returned'][name]
            ledger['endpoint_filter_completed'][name] = \
                ledger['endpoint_filter_returned'][name]
            self.persist()
            # Non-compressed factored raw is the scientific per-eval evidence.
            np.savez(self.out/f'eval-{name}-{index:03d}.npz', **raw)
            ledger['leg_raw_saved'][name] = index
            ledger['bank_raw_saved'][name] = index
            ledger['leg_pairs_raw_saved'][name] = 2*index
            ledger['leg_conditional_s_raw_saved'][name] = 20.*index
            self.persist()
            if index == 1:
                trace = {key: value.detach().cpu().numpy().copy()
                         for key, value in response.items()}
                trace.update({key: value.detach().cpu().numpy().copy()
                              for key, value in bank.items()})
                trace.update(delta=delta.copy(), edge_ids=raw['edge_ids'],
                             time_s=raw['time_s'],
                             context_names=raw['context_names'])
                np.savez_compressed(self.out/f'warm-trace-{name}.npz', **trace)
                ledger['warm_trace_saved'][name] = True
                self.persist()
                self.warm_parity(raw, name)
            if not all(np.isfinite(raw[key]).all() for key in
                       ('parts', 'losses', 'ratios', 'loss_gradient', 'v',
                        'q_error', 'tendon_error', 'q_coefficient',
                        'ltm_coefficient', 'ltm_gain')):
                raise ValueError(f'Nonfinite E119 response/derivative {name}')
            # Selection uses the true saved nonlinear ratios, never an affine
            # or a returned-SLSQP-success shortcut.
            record = selector_record(index, name, delta, raw)
            with (self.out/f'history-{name}.jsonl').open('a',
                    encoding='utf-8') as stream:
                stream.write(json.dumps(record, allow_nan=False)+'\n')
            self.records.append(record)
            return {key: raw[key] for key in ('parts', 'losses', 'ratios',
                'loss_gradient', 'q_error', 'tendon_error', 'v',
                'q_coefficient', 'ltm_coefficient', 'ltm_gain')}
        except BaseException as primary:
            response = getattr(stage, 'last_response', None)
            if isinstance(response, dict):
                tensors.update({f'response_{key}': value for key, value in
                                response.items()})
            detach_available(tensors, raw, primary)
            if ledger['leg_materialized'][name] < index and all(
                    f'response_{key}' in raw for key in ('current', 'forcing',
                    'motor_rate', 'muscle_activation', 'command', 'tendon')):
                self.response_materialized(index)
            rescue_raw(self.out/f'error-raw-{name}-{index:03d}', raw,
                       primary, 'leg evaluation')
            raise

    def warm_parity(self, raw, name):
        ref = self.data['baseline'][name]
        jacref = self.data['e118_jacobians'][name]
        jq, jz = factor_jacobians(raw)
        differences = {
            'parts': float(np.max(np.abs(raw['parts']-ref['parts']))),
            'losses': float(np.max(np.abs(raw['losses']-ref['losses']))),
            'ratios': float(np.max(np.abs(raw['ratios']-ref['ratios']))),
            'loss_gradient': float(np.max(np.abs(raw['loss_gradient']-
                                                ref['jacobian'][:2]))),
            'q_error': float(np.max(np.abs(raw['q_error']-
                                              jacref['q_error']))),
            'tendon_error': float(np.max(np.abs(raw['tendon_error']-
                                                   jacref['tendon_error']))),
            'q_jacobian': float(np.max(np.abs(jq-
                                                 jacref['q_jacobian']))),
            'tendon_jacobian': float(np.max(np.abs(jz-
                                                 jacref['tendon_jacobian'])))}
        passed = bool(all(np.isfinite(x) for x in differences.values()) and
            all(differences[key] <= 1e-8 for key in
                ('parts', 'losses', 'ratios', 'loss_gradient')) and
            all(differences[key] <= 1e-9 for key in
                ('q_error', 'tendon_error', 'q_jacobian', 'tendon_jacobian')))
        save_json(self.out/f'warm-parity-{name}.json', {
            'differences': differences, 'loss_limit': 1e-8,
            'startup_jacobian_limit': 1e-9, 'passed': passed,
            'unique_delta_index': 1, 'raw_saved_before_assertions': True})
        require(passed, f'E119 E115/E118 warm parity failed: {name}')


def solve_leg(fit, bounds, warm_delta):
    """Exactly one SLSQP start; the charged warm delta is already cached."""
    initial = fit.cache.get_delta(warm_delta)
    x0 = np.r_[warm_delta, float(np.max(initial['losses'])+.01)]
    constraints = [{'type': 'ineq',
        'fun': lambda x: fit.cache.get(x)['inequalities'],
        'jac': lambda x: fit.cache.get(x)['inequality_jac']}]
    return minimize(lambda x: fit.cache.get(x)['objective'], x0,
        method='SLSQP', jac=lambda x: fit.cache.get(x)['objective_jac'],
        bounds=[tuple(row) for row in bounds]+[(0., None)],
        constraints=constraints,
        options={'maxiter': 100, 'ftol': 1e-9, 'disp': False})


def solver_receipt(solution):
    if solution is None:
        return None
    return {'success': bool(getattr(solution, 'success', False)),
        'status': int(getattr(solution, 'status', -1)),
        'message': str(getattr(solution, 'message', 'unavailable')),
        'nit': int(getattr(solution, 'nit', -1)),
        'nfev': int(getattr(solution, 'nfev', -1)),
        'njev': int(getattr(solution, 'njev', -1)),
        'x': np.asarray(getattr(solution, 'x', []), dtype=np.float64),
        'fun': float(getattr(solution, 'fun', math.nan))}


def run_once(root, out):
    """One later-authorized CUDA fit with durable, exact resource accounting."""
    root, out = Path(root), Path(out)
    process_started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    names = LEGS
    ledger = {'experiment': 'E119-samplewise-startup-fit',
        'stage': 'created', 'prior_conservative_process_s': PRIOR_PROCESS_S,
        'shared_process_limit_s': SHARED_PROCESS_S,
        'process_wall_s': None, 'process_wall_status': 'not_yet_read',
        'fit_calls': 0, 'fit_wall_s': None,
        'leg_wall_s': {name: None for name in names},
        'active_leg': None, 'active_delta': None, 'active_delta_index': None,
        'leg_started': {name: 0 for name in names},
        'leg_materialized': {name: 0 for name in names},
        'leg_raw_saved': {name: 0 for name in names},
        'response_call_entered': {name: 0 for name in names},
        'leg_pairs_started': {name: 0 for name in names},
        'leg_pairs_materialized': {name: 0 for name in names},
        'leg_pairs_raw_saved': {name: 0 for name in names},
        'leg_conditional_s_started': {name: 0. for name in names},
        'leg_conditional_s_materialized': {name: 0. for name in names},
        'leg_conditional_s_raw_saved': {name: 0. for name in names},
        'bank_started': {name: 0 for name in names},
        'bank_call_entered': {name: 0 for name in names},
        'bank_materialized': {name: 0 for name in names},
        'bank_raw_saved': {name: 0 for name in names},
        'linear_filter_started': {name: 0 for name in names},
        'linear_filter_returned': {name: 0 for name in names},
        'linear_filter_completed': {name: 0 for name in names},
        'endpoint_filter_started': {name: 0 for name in names},
        'endpoint_filter_returned': {name: 0 for name in names},
        'endpoint_filter_completed': {name: 0 for name in names},
        'warm_trace_saved': {name: False for name in names},
        'cache_hits': {name: 0 for name in names},
        'solver_calls': {name: 0 for name in names},
        'leg_termination_reason': {name: None for name in names},
        'final_batch_started': 0, 'final_batch_materialized': 0,
        'final_batch_raw_saved': 0, 'final_pairs_started': 0,
        'final_pairs_materialized': 0, 'final_pairs_raw_saved': 0,
        'final_conditional_s_started': 0.,
        'final_conditional_s_materialized': 0.,
        'final_conditional_s_raw_saved': 0.,
        'final_response_call_entered': False,
        'actual_neural_s': 0., 'actual_physical_s': 0.,
        'native_network_calls': 0, 'body_calls': 0,
        'F6_exposed': False, 'candidate_adopted': False,
        'secondary_errors': []}
    fit_started = None
    leg_started = {}

    def refresh_clock(primary=None):
        try:
            elapsed = time.monotonic()-process_started
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError('Unverifiable E119 monotonic wall time')
            ledger['process_wall_s'] = elapsed
            ledger['process_wall_status'] = 'available'
            if fit_started is not None:
                ledger['fit_wall_s'] = time.monotonic()-fit_started
            for name, stamp in leg_started.items():
                ledger['leg_wall_s'][name] = time.monotonic()-stamp
        except Exception as secondary:
            ledger['process_wall_s'] = None
            ledger['process_wall_status'] = 'unavailable'
            if primary is None:
                raise
            ledger['secondary_errors'].append({'at': 'clock',
                'type': type(secondary).__name__, 'message': str(secondary)})
            primary.add_note(f'E119 secondary clock failure: {secondary}')

    def persist():
        refresh_clock()
        save_json(out/'execution-ledger.json', ledger)

    def deadlines(name=None, *, soft=True):
        elapsed = time.monotonic()-process_started
        if elapsed > PROCESS_WALL_S or PRIOR_PROCESS_S+elapsed > \
                SHARED_PROCESS_S:
            raise TimeoutError('E119 internal/shared process wall limit')
        if fit_started is not None and time.monotonic()-fit_started > FIT_WALL_S:
            raise TimeoutError('E119 fit150 wall limit')
        if name is not None and name in leg_started:
            elapsed_leg = time.monotonic()-leg_started[name]
            if elapsed_leg > LEG_WALL_S:
                raise TimeoutError(f'E119 leg24 wall limit: {name}')
            if soft and elapsed_leg >= LEG_WALL_S-LEG_RESERVE_S:
                raise LegSoftStop('leg_wall_finalize_reserve')

    def failure_save(filename, value, primary):
        try:
            save_json(out/filename, value)
        except Exception as secondary:  # noqa: BLE001
            ledger['secondary_errors'].append({'at': filename,
                'type': type(secondary).__name__, 'message': str(secondary)})
            primary.add_note(f'E119 secondary {filename} save: {secondary}')

    primary = None
    full_output = None
    full_raw = {}
    feet_tensor = None
    feet_fixed = None
    foot_raw = {}
    fixed_count = None
    metrics = None
    try:
        for original, name in (
            (ROOT/'tools/fit_c_banc_m3_samplewise_startup.py', 'runner.py'),
            (ROOT/'tests/test_c_banc_samplewise_fit.py',
             'test_c_banc_samplewise_fit.py'),
            (ROOT/'docs/BANC_M3_E119_SAMPLEWISE_FIT_DESIGN_20260920.md',
             'frozen-design.md'),
            (root/'e119-frozen-design.json', 'e119-frozen-design.json')):
            shutil.copy2(original, out/name)
        ledger['stage'] = 'preflight'
        persist()
        data = reviewed_inputs(root)
        deadlines()
        require(torch.cuda.is_available() and
                'RTX 4080' in torch.cuda.get_device_name(0) and
                all(os.environ.get(key) == '1' for key in
                    ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')),
                'Reviewed RTX4080/single-thread CUDA environment required')
        torch.set_num_threads(1)
        require(torch.get_num_threads() == 1,
                'Torch CPU thread count changed')
        ledger['environment'] = {'device': torch.cuda.get_device_name(0),
            'torch_version': torch.__version__, 'torch_cuda':
            torch.version.cuda, 'torch_threads': torch.get_num_threads(),
            'omp': os.environ['OMP_NUM_THREADS'],
            'mkl': os.environ['MKL_NUM_THREADS'],
            'openblas': os.environ['OPENBLAS_NUM_THREADS']}
        geometry = TorchFootGeometry.load(
            root/'e111-cuda-foot-geometry/geometry.npz', device='cuda:0')
        require(geometry.geometry_hash == GEOMETRY_HASH,
                'E111 fixed E32 geometry identity changed')
        fit_started = time.monotonic()
        ledger['fit_calls'] = 1
        ledger['stage'] = 'fit'
        try:
            persist()
        except BaseException:
            ledger['fit_calls'] = 0
            ledger['fit_start_receipt_failed'] = True
            raise
        selected = []
        for leg, name in enumerate(LEGS):
            deadlines()
            leg_started[name] = time.monotonic()
            stage = LegStage(data, leg, geometry, device='cuda:0')
            fit = LegFit(stage, data, out, ledger, persist, deadlines)
            warm = data['selected']['delta'][stage.edges].copy()
            solution = None
            stop_reason = None
            try:
                fit.cache.get_delta(warm)
                deadlines(name, soft=False)
                ledger['solver_calls'][name] = 1
                try:
                    persist()
                except BaseException:
                    ledger['solver_calls'][name] = 0
                    ledger['solver_start_receipt_failed'] = name
                    raise
                solution = solve_leg(fit, data['bounds'][stage.edges], warm)
            except LegSoftStop as stopped:
                stop_reason = stopped.reason
            best_any, best_feasible = choose_records(fit.records)
            report = {'leg': name, 'solver': solver_receipt(solution),
                'termination_reason': stop_reason if stop_reason is not None
                    else 'scipy_returned',
                'best_any': best_any, 'best_feasible': best_feasible,
                'unique_deltas': len(fit.records),
                'cache_hits': ledger['cache_hits'][name]}
            save_json(out/f'selection-{name}.json', report)
            selected.append(best_feasible)
            ledger['leg_termination_reason'][name] = report[
                'termination_reason']
            ledger['stage'] = f'leg_{name}_finished'
            ledger['active_leg'] = None
            ledger['active_delta'] = None
            ledger['active_delta_index'] = None
            persist()
            deadlines(name, soft=False)
            leg_started.pop(name)
        ledger['fit_wall_s'] = time.monotonic()-fit_started
        # Include final selection/persistence within the fit deadline.
        deadlines()
        fit_started = None
        deadlines()
        if any(record is None for record in selected):
            ledger['stage'] = 'no_joint_feasible_candidate'
            persist()
            result = no_joint_result(selected, ledger)
            save_json(out/'result.json', result)
            return result
        delta = np.zeros(1572, dtype=np.float64)
        for leg, record in enumerate(selected):
            delta[np.flatnonzero(data['edge_mask'][leg])] = record['delta']
        boundary = efficacy_boundary_roundoff(data['efficacy'], delta,
                                              data['bounds'])
        np.savez_compressed(out/'efficacy-boundary-raw.npz',
            delta=delta, bounds=data['bounds'], base_efficacy=data['efficacy'],
            pre=boundary['pre'], post=boundary['post'],
            correction=boundary['correction'], tolerance=boundary['tolerance'],
            trust_valid=boundary['trust_valid'],
            original_valid=boundary['original_valid'], valid=boundary['valid'])
        require(np.all(boundary['valid']),
                'Selected efficacy exceeds original or frozen trust bounds')
        f = data['fixture']
        rows, cols = f['edge_rows'], f['edge_cols']
        efficacy, solver_edge, native_edge, edge_rounding = \
            native_rounding_comparison(f['original_signed_edge_weights'],
                data['efficacy'], f['weights'][1, rows, cols], delta,
                efficacy_override=boundary['post'])
        np.savez_compressed(out/'selected-delta.npz', delta=delta,
            efficacy=efficacy, edge_ids=f['edge_ids'], edge_rows=rows,
            edge_cols=cols, efficacy_pre_roundoff=boundary['pre'],
            efficacy_roundoff_correction=boundary['correction'],
            selected_solver_W=solver_edge,
            expected_native_float32_W=native_edge,
            expected_native_minus_solver=edge_rounding)
        train_final = dict(data['train'])
        train_final['current_w'] = f['weights'][1]
        native_w, cal = selected_actual_float32(train_final,
            data['calibration'], efficacy)
        solver_w = f['weights'][1].copy()
        solver_w[rows, cols] = solver_edge
        np.savez_compressed(out/'rounding-raw.npz',
            solver_float64_W=solver_w, native_float32_W=native_w,
            selected_solver_W=solver_edge, selected_native_W=native_edge,
            selected_native_minus_solver=edge_rounding,
            signed_source_times_efficacy=f['original_signed_edge_weights']*
                efficacy, absolute_difference=np.abs(solver_w-native_w))
        np.savez_compressed(out/'calibration.npz', **cal)
        require(np.array_equal(native_w[rows, cols], native_edge),
                'Native selected float32 calibration/W rounding mismatch')
        deadlines()
        response = ContextResponse(time_s=f['time_s'], tau=f['tau'], a=f['a'],
            threshold=f['threshold'], cap=f['cap'], pool=f['pool'],
            moment=f['moment'], rest=f['rest'], ltm=f['ltm'], device='cuda:0')
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float64,
                                           device='cuda:0')
        source = tensor(f['source_rate'])
        background = tensor(f['background'])
        candidate = tensor(native_w[None])
        contexts = torch.tensor([0, 1], dtype=torch.long, device='cuda:0')
        weights = torch.tensor([0, 0], dtype=torch.long, device='cuda:0')
        torch.cuda.synchronize()
        full_raw = {'time_s': f['time_s'].copy(),
            'actual_float32_source_weights': native_w.copy(),
            'pair_context_indices': np.array([0, 1], dtype=np.int64),
            'pair_weight_indices': np.array([0, 0], dtype=np.int64)}
        deadlines()
        ledger.update(stage='final_response', final_batch_started=1,
            final_pairs_started=2, final_conditional_s_started=20.)
        try:
            persist()
        except BaseException as primary_start:
            ledger.update(final_batch_started=0, final_pairs_started=0,
                final_conditional_s_started=0., final_start_receipt_failed=True)
            rescue_raw(out/'error-final-input', full_raw, primary_start,
                       'final input')
            raise
        ledger['final_response_call_entered'] = True
        full_output = response.evaluate(source, background, candidate,
                                        contexts, weights)
        detach_available(full_output, full_raw)
        torch.cuda.synchronize()
        ledger.update(stage='final_materialized', final_batch_materialized=1,
            final_pairs_materialized=2, final_conditional_s_materialized=20.)
        persist()
        np.savez_compressed(out/'prediction.npz', **full_raw)
        ledger.update(stage='final_raw_saved', final_batch_raw_saved=1,
            final_pairs_raw_saved=2, final_conditional_s_raw_saved=20.)
        persist()
        deadlines()
        fixed = np.flatnonzero(f['fixed_mask'])
        fixed_count = len(fixed)
        foot_raw = {'fixed_mask': f['fixed_mask'].copy(),
            'common_mask': f['common_mask'].copy(),
            'own_mask': f['own_mask'].copy()}
        feet_tensor = geometry.full(tensor(full_raw['command'][:, fixed].reshape(
            -1, 42)))
        feet_fixed = materialize_feet(feet_tensor, fixed_count)
        torch.cuda.synchronize()
        foot_raw['feet_fixed'] = feet_fixed
        np.savez_compressed(out/'foot-proxy-raw.npz', **foot_raw)
        metrics = frozen_output_metrics(full_raw, feet_fixed, data)
        save_json(out/'metrics.json', metrics)
        save_json(out/'provenance.json', {'design_sha256': DESIGN_SHA,
            'freeze_sha256': FREEZE_SHA,
            'input_sha256': data['e119_freeze']['input_sha256'],
            'source_sha256': data['e119_freeze']['source_sha256'],
            'runner_sha256': sha256(ROOT/
                'tools/fit_c_banc_m3_samplewise_startup.py'),
            'test_sha256': sha256(ROOT/'tests/test_c_banc_samplewise_fit.py'),
            'geometry_hash': geometry.geometry_hash,
            'candidate_adopted': False, 'native_gait_tested': False,
            'F6_exposed': False})
        deadlines()
        ledger['stage'] = 'complete'
        persist()
        result = {'status': 'completed_samplewise_conditional_fit',
            'robust_local_supported': metrics['robust_local_supported'],
            'candidate_adopted': False, 'native_gait_tested': False,
            'F6_exposed': False, 'ledger': ledger}
        save_json(out/'result.json', result)
        return result
    except BaseException as error:
        primary = error
        ledger['stage'] = 'error'
        ledger['primary_error'] = {'type': type(error).__name__,
            'message': str(error), 'traceback': traceback.format_exc()}
        refresh_clock(error)
        failure_save('error.json', ledger['primary_error'], error)
        if full_output is not None:
            detach_available(full_output, full_raw, error)
        if full_raw and not ledger['final_batch_raw_saved']:
            rescue_raw(out/'error-available/final', full_raw, error,
                       'final response')
        rescue_available_feet(feet_tensor, fixed_count, foot_raw, error,
                              ledger['secondary_errors'])
        if foot_raw:
            rescue_raw(out/'error-available/feet', foot_raw, error, 'feet')
        if metrics is not None:
            failure_save('error-available-metrics.json', metrics, error)
        raise
    finally:
        if primary is not None:
            refresh_clock(primary)
            failure_save('execution-ledger.json', ledger, primary)
            failure_save('error.json', {'primary_error': ledger['primary_error'],
                'process_wall_s': ledger['process_wall_s'],
                'process_wall_status': ledger['process_wall_status']}, primary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/
        'verification/banc-natural-gait-m3-20260919')
    parser.add_argument('--out', type=Path, default=ROOT/
        'verification/banc-natural-gait-m3-20260919/e119-samplewise-fit')
    args = parser.parse_args()
    run_once(args.root, args.out)


if __name__ == '__main__':
    main()
