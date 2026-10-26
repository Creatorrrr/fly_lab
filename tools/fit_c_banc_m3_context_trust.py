"""E115 bounded two-context conditional efficacy fit; never advances BANC/body.

Actual CUDA fitting is reserved for one separately parent-approved process.
Only the two recorded E110/E113 inputs are used; the response is an endpoint
approximation, and its E111 foot geometry is a command-derived proxy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from tools.banc_m3_context_response import ContextResponse, id_positions
from tools.banc_m3_torch_foot_geometry import TorchFootGeometry
from tools.fit_c_banc_m3_block_constrained_recruitment import (
    absolute_shape_screen,
    leg_partition,
)
from tools.fit_c_banc_m3_constrained_startup import native_shape_screen, startup_screen
from tools.fit_c_banc_m3_dense_foot_recruitment import (
    scalar_metrics,
    selected_actual_float32,
)
from tools.verify_c_banc_m3_e114_contexts import error_stats
from tools.verify_c_banc_m3_e114_contexts import reviewed_inputs as e114_inputs

DESIGN_SHA = 'c1522921e5e5c8ac1e02129576d1b7605cc80af5d46a5d8491b5875b104d8ca2'
FROZEN_SHA = '8a8a7939a5948a1d9600493696fe3af718d64eb61fc0abc7466018214554ae74'
GEOMETRY_HASH = '36a47d2f0f0d4e83d5ff8ecf1af95b9ce63269d5158d98642b4f89e0bc230953'
PARENT_DATA_RAW_SHA = '2cfced9562b14727ca27e925ea69f500e776a9bdf6fffb843990f51848ec20a4'
LEGACY_ADDENDUM_SHA = '12df4db93c88e41fdcb2a2ffce510ec759745c81ff148c6c930a5b71de72a759'
DENSE_HELPER_SHA = '5eb2d0c2d1ec8f3d0db43192610748d8c2d27e765b9c42a0aff6912f7ea54066'
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
INNER = np.array([.018, .04, .009, .025], dtype=np.float64)
ORIGINAL = np.array([.02, .05, .01, .03], dtype=np.float64)
OPTIMIZER_CAP = (1-2e-4)**2
SELECTOR_CAP = (1-1e-4)**2
DELTA_LIMIT = math.log(1.1)
LEG_LIMIT_S = 28.
LEG_FINALIZE_RESERVE_S = 2.
LEG_UNIQUE_LIMIT = 500
FIT_LIMIT_S = 180.
PROCESS_LIMIT_S = 210.
PRIOR_PROCESS_S = 598.
SHARED_PROCESS_LIMIT_S = 838.


class LegSoftStop(RuntimeError):
    """Expected per-leg search exhaustion; existing records remain selectable."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), ensure_ascii=False,
                                    allow_nan=False, indent=2), encoding='utf-8')


def trust_bounds(efficacy):
    efficacy = np.asarray(efficacy, dtype=np.float64)
    if efficacy.ndim != 1 or not np.isfinite(efficacy).all() or \
            np.any(efficacy <= 0):
        raise ValueError('Finite positive E113 efficacy required')
    lower = np.maximum(-DELTA_LIMIT, np.log(.001/efficacy))
    upper = np.minimum(+DELTA_LIMIT, np.log(10./efficacy))
    if np.any(lower > upper) or np.any(lower > 0) or np.any(upper < 0):
        raise ValueError('Empty original-efficacy/trust intersection')
    return np.stack((lower, upper), axis=1)


def delta_key(delta):
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 1 or not np.isfinite(delta).all():
        raise ValueError('Finite one-dimensional delta required')
    return delta.tobytes()


def selected_edge_values(cal_edge_ids, cal_values, selected_edge_ids):
    """The full calibration edge IDs are not sorted."""
    positions = id_positions(cal_edge_ids, selected_edge_ids)
    values = np.asarray(cal_values)
    if values.ndim != 1 or len(values) != len(cal_edge_ids):
        raise ValueError('One calibration value per explicit edge ID required')
    return values[positions]


def epigraph_terms(delta, t, cached):
    """Only t algebra changes on a delta cache hit; no new model response."""
    delta = np.asarray(delta, dtype=np.float64)
    losses = np.asarray(cached['losses'], dtype=np.float64)
    parts = np.asarray(cached['parts'], dtype=np.float64)
    ratios = np.asarray(cached['ratios'], dtype=np.float64)
    jac = np.asarray(cached['jacobian'], dtype=np.float64)
    n = len(delta)
    if n == 0 or losses.shape != (2,) or parts.shape != (2, 4) or \
            ratios.shape != (2, 4) or jac.shape != (10, n) or \
            not np.isfinite(delta).all() or not np.isfinite(t) or \
            not all(np.isfinite(x).all() for x in (losses, parts, ratios, jac)):
        raise ValueError('Finite two-context loss/startup/gradient record required')
    coefficient = 6e-5*(n/1572.)
    prior = coefficient*np.mean(delta*delta)
    grad = np.r_[2*coefficient*delta/n, 1.]
    inequalities = np.r_[t-losses, OPTIMIZER_CAP-ratios.ravel()]
    inequality_jac = np.zeros((10, n+1), dtype=np.float64)
    inequality_jac[:2, :n] = -jac[:2]
    inequality_jac[:2, n] = 1.
    inequality_jac[2:, :n] = -jac[2:]
    return {'objective': float(t+prior), 'objective_jac': grad,
            'inequalities': inequalities, 'inequality_jac': inequality_jac,
            'true_worst_objective': float(np.max(losses)+prior),
            'prior': float(prior), 'losses': losses.copy(),
            'ratios': ratios.copy(), 'parts': parts.copy(),
            'epigraph_gaps': (t-losses).copy()}


def choose_records(records):
    """Select by true worst loss and all eight strict startup ratios."""
    finite = [record for record in records
              if np.isfinite(record['true_worst_objective']) and
              np.isfinite(record['ratios']).all()]
    best_any = min(finite, key=lambda row: row['true_worst_objective']) \
        if finite else None
    feasible = [row for row in finite if np.all(record_ratios(row) <= SELECTOR_CAP)]
    best_feasible = min(feasible, key=lambda row: row['true_worst_objective']) \
        if feasible else None
    return best_any, best_feasible


def record_ratios(record):
    return np.asarray(record['ratios'], dtype=np.float64)


def candidate_weights(base_w, rows, cols, delta):
    """Solver starts at *actual float32 E113 W*, widened to float64."""
    if not isinstance(base_w, torch.Tensor) or base_w.dtype != torch.float64 or \
            delta.dtype != torch.float64 or base_w.device != delta.device:
        raise ValueError('Same-device float64 baseline W and delta required')
    result = base_w.clone()
    result[rows, cols] = base_w[rows, cols]*torch.exp(delta)
    return result


def native_rounding_comparison(original_signed, base_efficacy,
                               actual_base_w, delta, efficacy_override=None):
    """Keep solver's W113 origin separate from native source*eff float32."""
    original_signed = np.asarray(original_signed, dtype=np.float32)
    base_efficacy = np.asarray(base_efficacy, dtype=np.float64)
    actual_base_w = np.asarray(actual_base_w, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    if original_signed.shape != base_efficacy.shape or \
            actual_base_w.shape != base_efficacy.shape or \
            delta.shape != base_efficacy.shape or \
            not all(np.isfinite(x).all() for x in
                    (original_signed, base_efficacy, actual_base_w, delta)):
        raise ValueError('Aligned finite selected-edge rounding operands required')
    efficacy = base_efficacy*np.exp(delta) if efficacy_override is None else \
        np.asarray(efficacy_override, dtype=np.float64)
    if efficacy.shape != base_efficacy.shape or not np.isfinite(efficacy).all():
        raise ValueError('Finite aligned native efficacy required')
    solver = actual_base_w*np.exp(delta)
    native = (original_signed*efficacy).astype(np.float32).astype(np.float64)
    return efficacy, solver, native, native-solver


def efficacy_boundary_roundoff(base_efficacy, delta, bounds):
    """Expose an exact-boundary float64 product before narrowly clipping ULPs."""
    base_efficacy = np.asarray(base_efficacy, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    bounds = np.asarray(bounds, dtype=np.float64)
    if base_efficacy.ndim != 1 or delta.shape != base_efficacy.shape or \
            bounds.shape != (len(delta), 2) or \
            not all(np.isfinite(x).all() for x in (base_efficacy, delta, bounds)):
        raise ValueError('Finite aligned efficacy, delta and trust bounds required')
    pre = base_efficacy*np.exp(delta)
    post = np.clip(pre, .001, 10.)
    tolerance = 8*np.spacing(np.maximum(np.abs(pre), .001))
    trust_valid = (delta >= bounds[:, 0]) & (delta <= bounds[:, 1])
    original_valid = (pre >= .001-tolerance) & (pre <= 10.+tolerance)
    valid = trust_valid & original_valid & np.isfinite(pre)
    return {'pre': pre, 'post': post, 'correction': post-pre,
            'tolerance': tolerance, 'trust_valid': trust_valid,
            'original_valid': original_valid, 'valid': valid}


def reviewed_inputs(root):
    """Read-only frozen data preflight; no CUDA/geometry/model construction."""
    root = Path(root)
    require(sha256(ROOT/'docs/BANC_M3_E115_CONTEXT_TRUST_FIT_DESIGN_20260920.md')
            == DESIGN_SHA and sha256(root/'e115-frozen-design.json') == FROZEN_SHA,
            'E115 design/freeze changed')
    spec = json.loads((root/'e115-frozen-design.json').read_text(encoding='utf-8'))
    require(spec['actual_execution_authorized'] is False and
            spec['leg_unique_deltas'] == LEG_UNIQUE_LIMIT and
            spec['leg_wall_s'] == LEG_LIMIT_S and
            spec['fit_wall_s'] == FIT_LIMIT_S and
            spec['internal_wall_s'] == PROCESS_LIMIT_S and
            spec['prior_conservative_process_s'] == PRIOR_PROCESS_S and
            spec['shared_process_limit_s'] == SHARED_PROCESS_LIMIT_S,
            'Frozen E115 scope/resource contract changed')
    for relative, digest in spec['input_sha256'].items():
        require(sha256(root/relative) == digest, f'Frozen input changed: {relative}')
    for relative, digest in spec['source_sha256'].items():
        require(sha256(ROOT/relative) == digest, f'Reused source changed: {relative}')
    require(sha256(ROOT/'tools/fit_c_banc_m3_dense_foot_recruitment.py')
            == DENSE_HELPER_SHA, 'Reused dense calibration/metrics helper changed')
    require(sha256(root/'e115-legacy-reference-inputs.json') == LEGACY_ADDENDUM_SHA,
            'E115 legacy reference addendum changed')
    legacy = json.loads((root/'e115-legacy-reference-inputs.json').read_text(
        encoding='utf-8'))
    require(legacy['actual_execution_authorized'] is False,
            'Legacy reference addendum scope changed')
    for relative, digest in legacy['inputs_sha256'].items():
        require(sha256(root/relative) == digest,
                f'Legacy E112/E101 reference changed: {relative}')
    review = json.loads((root/'e114-context-validation/parent-final-review.json')
                        .read_text(encoding='utf-8'))
    require(review['accepted_as_valid_conditional_tool'] is True and
            review['blocking_defects'] == [] and review['parent_conditional_responses'] == 1,
            'E114 final independent validation missing')
    fixture, _ = e114_inputs(root)
    cal = load_npz(root/'e112-dense-foot-recruitment-v2/calibration.npz')
    train = load_npz(root/'parent-e112-training-inputs.npz')
    e114 = load_npz(root/'e114-context-validation/response-raw.npz')
    feet114 = load_npz(root/'e114-context-validation/foot-proxy-raw.npz')
    require(np.array_equal(e114['pair_context_indices'], [0, 0, 1, 1]) and
            np.array_equal(e114['pair_weight_indices'], [0, 1, 0, 1]),
            'E114 pair order changed')
    require(np.array_equal(fixture['edge_ids'], cal['individual_edge_ids']) and
            np.array_equal(fixture['source_ids'], cal['individual_source_basis_ids']),
            'E113 selected edge/source identity changed')
    require(np.array_equal(selected_edge_values(cal['edge_ids'], cal['values'],
            fixture['edge_ids']).astype(np.float64),
            fixture['weights'][1, fixture['edge_rows'], fixture['edge_cols']]),
            'E113 unsorted calibration values differ from actual W')
    require(np.array_equal(fixture['edge_ids'], train['edge_ids']) and
            np.array_equal(fixture['original_signed_edge_weights'],
                           train['original_signed_edge_weights']) and
            np.array_equal(fixture['weights'][0], train['current_w']),
            'Original signed/edge mapping changed')
    efficacy = cal['individual_edge_efficacies'].astype(np.float64)
    require(efficacy.shape == (1572,) and np.isfinite(efficacy).all(),
            'E113 selected efficacy missing')
    bounds = trust_bounds(efficacy)
    motor_mask, edge_mask = leg_partition(fixture['moment'], fixture['pool'],
        fixture['ltm'], fixture['edge_rows'],
        expected_motor=train['leg_motor_mask'], expected_edge=train['leg_edge_mask'])
    require(sha256(root/'parent-e115-data-review-raw.npz') == PARENT_DATA_RAW_SHA,
            'Parent independent baseline raw changed')
    independent = load_npz(root/'parent-e115-data-review-raw.npz')
    require(np.array_equal(independent['initial_efficacy'], efficacy) and
            np.array_equal(independent['delta_lower'], bounds[:, 0]) and
            np.array_equal(independent['delta_upper'], bounds[:, 1]) and
            np.array_equal(independent['leg_motor_mask'], motor_mask) and
            np.array_equal(independent['leg_edge_mask'], edge_mask) and
            independent['leg_muscle_mask'].shape == (6, 96) and
            independent['initial_startup_ratios'].shape == (6, 2, 4) and
            independent['initial_common_loss_parts'].shape == (6, 2, 4) and
            independent['common_target_feet'].shape == (2, 1823, 6, 3),
            'Independent baseline axes/partition/bounds changed')
    for leg in range(6):
        owned = independent['leg_muscle_mask'][leg]
        motors = motor_mask[leg]
        require(int(owned.sum()) in (15, 16, 17) and
                np.all(fixture['pool'][~owned][:, motors] == 0) and
                np.all(fixture['moment'][7*leg:7*leg+7, ~owned] == 0) and
                owned[fixture['ltm'][leg]],
                f'Leg {leg} has ambiguous muscle ownership')
    require(np.array_equal(fixture['own_mask'][0], fixture['fixed_mask']) and
            np.array_equal(np.flatnonzero(fixture['common_mask']),
                           np.arange(100, 1923)),
            'Common 1823 time support changed')
    require(e114['command'].shape == (4, 2001, 42) and
            feet114['feet_fixed'].shape == (4, 1828, 6, 3),
            'E114 baseline response/geometry shape changed')
    return {'fixture': fixture, 'calibration': cal, 'train': train,
            'e114': e114, 'e114_feet': feet114, 'efficacy': efficacy,
            'bounds': bounds, 'motor_mask': motor_mask,
            'edge_mask': edge_mask, 'spec': spec,
            'independent': independent, 'legacy': legacy}


def leg_losses_and_startup(command, tendon, feet, *, common, startup,
                           target_q, target_feet, target_start_q,
                           target_start_tendon, q_scale, foot_scale,
                           velocity_scale, dt=.005):
    """Two separate contexts, four equal loss parts and eight startup ratios."""
    if command.ndim != 3 or tendon.ndim != 3 or feet.ndim != 3 or \
            command.shape[0] != 2 or tendon.shape[0] != 2 or feet.shape[0] != 2:
        raise ValueError('Two context outputs required')
    q_delta = command[:, common] - target_q
    foot_delta = feet-target_feet
    velocity_delta = (feet[:, 1:]-feet[:, :-1])/dt - \
        (target_feet[:, 1:]-target_feet[:, :-1])/dt
    ltm_delta = tendon[:, common, 0]-.2
    parts = torch.stack((
        (q_delta/q_scale).square().mean(dim=(1, 2)),
        (foot_delta/foot_scale).square().mean(dim=(1, 2)),
        (velocity_delta/velocity_scale).square().mean(dim=(1, 2)),
        (ltm_delta/.2).square().mean(dim=1)), dim=1)
    startup_q = command[:, startup] - target_start_q
    startup_ltm = tendon[:, startup, 0] - target_start_tendon
    ratios = torch.stack((
        startup_q.square().mean(dim=(1, 2))/INNER[0]**2,
        startup_q.square().amax(dim=(1, 2))/INNER[1]**2,
        startup_ltm.square().mean(dim=1)/INNER[2]**2,
        startup_ltm.square().amax(dim=1)/INNER[3]**2), dim=1)
    return parts, ratios


class LegStage:
    """One owned MN/muscle/leg block, two fixed recorded input contexts."""

    def __init__(self, data, leg, geometry, *, device='cuda:0'):
        f = data['fixture']
        self.leg = leg
        self.name = LEGS[leg]
        self.motor = np.flatnonzero(data['motor_mask'][leg])
        self.edges = np.flatnonzero(data['edge_mask'][leg])
        self.muscle = np.flatnonzero(data['independent']['leg_muscle_mask'][leg])
        local_rows = np.searchsorted(self.motor, f['edge_rows'][self.edges])
        require(np.array_equal(self.motor[local_rows], f['edge_rows'][self.edges]),
                'Selected edge outside its one MN leg block')
        self.device = geometry.device
        require(self.device == torch.device(device), 'E111 geometry device changed')
        self.tensor = lambda value: torch.as_tensor(value, dtype=torch.float64,
                                                     device=self.device)
        self.rows = torch.as_tensor(local_rows, dtype=torch.long, device=self.device)
        self.cols = torch.as_tensor(f['edge_cols'][self.edges], dtype=torch.long,
                                    device=self.device)
        self.base_w = self.tensor(f['weights'][1, self.motor])
        self.source = self.tensor(f['source_rate'])
        self.background = self.tensor(f['background'][:, :, self.motor])
        self.common = torch.as_tensor(np.flatnonzero(f['common_mask']),
                                      dtype=torch.long, device=self.device)
        self.startup = torch.as_tensor(np.flatnonzero(f['startup_mask']),
                                       dtype=torch.long, device=self.device)
        self.target_q = self.tensor(f['own_target'][:, f['common_mask'],
                                                   7*leg:7*leg+7])
        self.target_foot = self.tensor(data['independent'][
            'common_target_feet'][:, :, leg])
        self.start_q = self.tensor(f['startup_target_command'][:,
                                     7*leg:7*leg+7])
        self.start_tendon = self.tensor(f['startup_target_tendon'][:, leg])
        self.q_scale = self.tensor(f['joint_scale'][7*leg:7*leg+7])
        self.foot_scale = self.tensor(f['foot_scale'][leg])
        self.velocity_scale = self.tensor(f['velocity_scale'][leg])
        self.geometry = geometry
        local_ltm = np.flatnonzero(self.muscle == f['ltm'][leg])
        require(len(local_ltm) == 1, 'Leg LTM muscle not owned exactly once')
        self.response = ContextResponse(time_s=f['time_s'],
            tau=f['tau'][self.motor], a=f['a'][self.motor],
            threshold=f['threshold'][self.motor], cap=f['cap'][self.motor],
            pool=f['pool'][self.muscle][:, self.motor],
            moment=f['moment'][7*leg:7*leg+7][:, self.muscle],
            rest=f['rest'][7*leg:7*leg+7], ltm=local_ltm, device=self.device)
        self.context_indices = torch.tensor([0, 1], dtype=torch.long,
                                            device=self.device)
        self.weight_indices = torch.tensor([0, 0], dtype=torch.long,
                                           device=self.device)

    def tensors(self, delta):
        candidate = candidate_weights(self.base_w, self.rows, self.cols, delta)
        response = self.response.evaluate(self.source, self.background,
            candidate.unsqueeze(0), self.context_indices, self.weight_indices)
        # The response exists before geometry/loss/gradient; retain it for a
        # failure-only raw rescue without issuing another conditional call.
        self.last_response = response
        command = response['command']
        selected = command[:, self.common]
        feet = self.geometry.leg(selected.reshape(-1, 7), self.leg).reshape(
            2, len(self.common), 3)
        parts, ratios = leg_losses_and_startup(command, response['tendon'],
            feet, common=self.common, startup=self.startup,
            target_q=self.target_q, target_feet=self.target_foot,
            target_start_q=self.start_q, target_start_tendon=self.start_tendon,
            q_scale=self.q_scale, foot_scale=self.foot_scale,
            velocity_scale=self.velocity_scale)
        return {'response': response, 'feet': feet,
                'parts': parts, 'losses': parts.sum(dim=1), 'ratios': ratios}


def detach_numpy(value):
    return value.detach().cpu().numpy().copy()


class DeltaCache:
    """Only the delta bit pattern identifies a conditional model evaluation."""

    def __init__(self, evaluator, on_hit=lambda: None):
        self.evaluator = evaluator
        self.on_hit = on_hit
        self.values = {}

    def get_delta(self, delta):
        key = delta_key(delta)
        if key not in self.values:
            self.values[key] = self.evaluator(np.asarray(delta,
                                                         dtype=np.float64).copy())
        else:
            self.on_hit()
        return self.values[key]

    def get(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1 or len(x) < 2:
            raise ValueError('SLSQP delta plus epigraph t required')
        return epigraph_terms(x[:-1], x[-1], self.get_delta(x[:-1]))


class LegFit:
    """One SLSQP leg with durable per-unique-delta response accounting."""

    def __init__(self, stage, data, out, ledger, persist, deadlines):
        self.stage = stage
        self.data = data
        self.out = out
        self.ledger = ledger
        self.persist = persist
        self.deadlines = deadlines
        self.records = []
        self.cache = DeltaCache(self.evaluate_delta, self.cache_hit)

    def cache_hit(self):
        self.ledger['leg_cache_hits'][self.stage.name] += 1
        self.deadlines(self.stage.name)
        self.persist()

    def baseline_parity(self, output, parts, ratios, index):
        stage, data = self.stage, self.data
        baseline = data['e114']
        pair = np.array([1, 3])
        np.savez_compressed(self.out/f'baseline-{stage.name}.npz',
            delta=np.zeros(len(stage.edges)),
            current=detach_numpy(output['response']['current']),
            motor_rate=detach_numpy(output['response']['motor_rate']),
            muscle_activation=detach_numpy(output['response']['muscle_activation']),
            command=detach_numpy(output['response']['command']),
            tendon=detach_numpy(output['response']['tendon']),
            feet_common=detach_numpy(output['feet']),
            loss_parts=parts, startup_ratios=ratios)
        saved = load_npz(self.out/f'baseline-{stage.name}.npz')
        comparisons = {
            'current': (saved['current'], baseline['current'][pair][:, :, stage.motor], 1e-8),
            'motor_rate': (saved['motor_rate'], baseline['motor_rate'][pair][:, :, stage.motor], 1e-9),
            'muscle_activation': (saved['muscle_activation'],
                baseline['muscle_activation'][pair][:, :, stage.muscle], 1e-9),
            'command': (saved['command'], baseline['command'][pair][:, :,
                7*stage.leg:7*stage.leg+7], 1e-9),
            'tendon': (saved['tendon'], baseline['tendon'][pair][:, :,
                stage.leg:stage.leg+1], 1e-9),
            'startup_ratios': (ratios, data['independent'][
                'initial_startup_ratios'][stage.leg], 1e-8),
            'common_loss_parts': (parts, data['independent'][
                'initial_common_loss_parts'][stage.leg], 1e-8)}
        report = {key: {'max_difference': float(np.max(np.abs(actual-target))),
                        'limit': limit, 'passed': bool(np.isfinite(actual).all()
                            and actual.shape == target.shape and
                            np.max(np.abs(actual-target)) <= limit)}
                  for key, (actual, target, limit) in comparisons.items()}
        save_json(self.out/f'baseline-parity-{stage.name}.json', {
            'unique_delta_index': index, 'raw_saved_before_assertions': True,
            'fields': report, 'passed': all(v['passed'] for v in report.values())})
        require(all(v['passed'] for v in report.values()),
                f'Original E114/parent initial baseline mismatch: {stage.name}')

    def evaluate_delta(self, delta):
        stage = self.stage
        name = stage.name
        self.deadlines(name)
        if self.ledger['leg_started_evaluations'][name] >= LEG_UNIQUE_LIMIT:
            raise LegSoftStop('leg_unique_delta_limit')
        if sum(self.ledger['leg_started_evaluations'].values()) >= 3000:
            raise TimeoutError('E115 total3000 unique-delta response limit')
        bounds = self.data['bounds'][stage.edges]
        if delta.shape != (len(stage.edges),) or not np.isfinite(delta).all() or \
                np.any(delta < bounds[:, 0]-1e-12) or \
                np.any(delta > bounds[:, 1]+1e-12):
            raise ValueError('Delta outside frozen per-edge trust intersection')
        # Complete all input transfer before recording a conditional call.
        parameter = torch.tensor(delta, dtype=torch.float64,
            device=stage.device, requires_grad=True)
        index = self.ledger['leg_started_evaluations'][name]+1
        self.ledger.update(active_leg=name, active_delta=delta.tolist(),
                           active_delta_index=index, stage='leg_response')
        self.ledger['leg_started_evaluations'][name] = index
        self.ledger['leg_context_pairs_started'][name] = 2*index
        self.ledger['leg_conditional_model_s_started'][name] = 20.*index
        try:
            self.persist()
        except BaseException as error:
            self.ledger['leg_started_evaluations'][name] = index-1
            self.ledger['leg_context_pairs_started'][name] = 2*(index-1)
            self.ledger['leg_conditional_model_s_started'][name] = 20.*(index-1)
            self.ledger['start_receipt_failed'] = {'leg': name, 'index': index}
            try:
                np.savez_compressed(self.out/f'error-input-{name}-{index:04d}.npz',
                                    delta=delta, bounds=bounds)
            except Exception as secondary:  # noqa: BLE001
                error.add_note(f'E115 secondary pre-response input save: {secondary}')
            raise
        self.ledger['response_call_entered'][name] += 1
        output = None
        raw = {'delta': delta.copy()}
        try:
            stage.last_response = None
            output = stage.tensors(parameter)
            metric = torch.cat((output['losses'], output['ratios'].reshape(-1)))
            jacobian = torch.autograd.grad(metric, parameter,
                grad_outputs=torch.eye(10, dtype=torch.float64,
                                       device=stage.device),
                is_grads_batched=True)[0]
            raw['parts'] = detach_numpy(output['parts'])
            raw['losses'] = detach_numpy(output['losses'])
            raw['ratios'] = detach_numpy(output['ratios'])
            raw['jacobian'] = detach_numpy(jacobian)
            self.ledger['leg_completed_evaluations'][name] = index
            self.ledger['leg_context_pairs_completed'][name] = 2*index
            self.ledger['leg_conditional_model_s_completed'][name] = 20.*index
            # The raw is the response's four loss parts, eight constraints,
            # and all exact gradients. Initial time traces get a separate raw.
            np.savez_compressed(self.out/f'eval-{name}-{index:04d}.npz', **raw)
            self.ledger['leg_raw_saved_evaluations'][name] = index
            self.ledger['leg_context_pairs_raw_saved'][name] = 2*index
            self.ledger['leg_conditional_model_s_raw_saved'][name] = 20.*index
            self.persist()
            if index == 1:
                self.baseline_parity(output, raw['parts'], raw['ratios'], index)
            if not all(np.isfinite(raw[key]).all() for key in
                       ('parts', 'losses', 'ratios', 'jacobian')):
                raise ValueError(f'Nonfinite E115 response or gradient {name}')
            prior = (6e-5*len(delta)/1572.)*float(np.mean(delta*delta))
            record = {'index': index, 'leg': name, 'delta': delta.tolist(),
                      'losses': raw['losses'].tolist(),
                      'parts': raw['parts'].tolist(),
                      'ratios': raw['ratios'].tolist(),
                      'true_worst_objective': float(np.max(raw['losses'])+prior),
                      'prior': prior,
                      'selector_feasible': bool(np.all(raw['ratios'] <= SELECTOR_CAP)),
                      'optimizer_startup_feasible': bool(
                          np.all(raw['ratios'] <= OPTIMIZER_CAP))}
            with (self.out/f'history-{name}.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, allow_nan=False)+'\n')
            self.records.append(record)
            return {'losses': raw['losses'], 'parts': raw['parts'],
                    'ratios': raw['ratios'], 'jacobian': raw['jacobian']}
        except BaseException as error:
            # No second response: salvage whichever CUDA tensors can transfer.
            response = getattr(stage, 'last_response', None)
            if output is not None or response is not None:
                available = [(key, value) for key, value in (output or {}).items()
                             if isinstance(value, torch.Tensor)]
                if output is not None:
                    response = output.get('response', response)
                if isinstance(response, dict):
                    available.extend((f'response_{key}', value)
                                     for key, value in response.items()
                                     if isinstance(value, torch.Tensor))
                for key, tensor in available:
                    if key not in raw:
                        try:
                            raw[key] = detach_numpy(tensor)
                        except Exception as secondary:  # noqa: BLE001
                            error.add_note(f'E115 secondary {key} transfer: {secondary}')
            destination = self.out/f'error-raw-{name}-{index:04d}'
            try:
                destination.mkdir(exist_ok=True)
                for key, value in raw.items():
                    try:
                        np.save(destination/f'{key}.npy', value, allow_pickle=False)
                    except Exception as secondary:  # noqa: BLE001
                        error.add_note(f'E115 secondary {key} raw save: {secondary}')
            except Exception as secondary:  # noqa: BLE001
                error.add_note(f'E115 secondary error raw directory: {secondary}')
            raise


def solve_leg(fit, bounds):
    """Exactly one SLSQP start, with a t-independent delta cache."""
    n = len(bounds)
    zero = np.zeros(n, dtype=np.float64)
    initial = fit.cache.get_delta(zero)  # Charged unique response 1.
    x0 = np.r_[zero, float(np.max(initial['losses'])+.01)]
    constraints = [{'type': 'ineq',
        'fun': lambda x: fit.cache.get(x)['inequalities'],
        'jac': lambda x: fit.cache.get(x)['inequality_jac']}]
    result = minimize(lambda x: fit.cache.get(x)['objective'], x0,
        method='SLSQP', jac=lambda x: fit.cache.get(x)['objective_jac'],
        bounds=[tuple(row) for row in bounds]+[(0., None)],
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-9, 'disp': False})
    return result


def common_scores(q_values, foot_values, target_q, target_foot):
    dq = q_values-target_q
    df = foot_values-target_foot
    dv = (np.diff(foot_values, axis=0)-np.diff(target_foot, axis=0))/.005
    return {'q_rms': float(np.sqrt(np.mean(dq*dq))),
        'foot_rms': float(np.sqrt(np.mean(df*df))),
        'velocity_rms': float(np.sqrt(np.mean(dv*dv))),
        'LF_row6_rms': float(np.sqrt(np.mean(dq[:, 6]**2))),
        'per_leg_q_rms': [float(np.sqrt(np.mean(dq[:, 7*j:7*j+7]**2)))
                          for j in range(6)],
        'per_leg_foot_rms': [float(np.sqrt(np.mean(df[:, j]**2)))
                             for j in range(6)],
        'per_leg_velocity_rms': [float(np.sqrt(np.mean(dv[:, j]**2)))
                                 for j in range(6)]}


def frozen_output_metrics(raw, feet_fixed, data):
    """Report separate original99, fixed E110 and per-context common screens."""
    f = data['fixture']
    selected = np.flatnonzero(f['fixed_mask'])
    common = np.flatnonzero(f['common_mask'])
    within_fixed = f['common_mask'][selected]
    target_feet_common = data['independent']['common_target_feet']
    baseline_raw = data['e114']
    baseline_feet = data['e114_feet']['feet_fixed']
    pair = (1, 3)
    result = {'foot_semantics': 'fixed E32 geometry from command; no actual foot path',
              'common_count': len(common), 'fixed_count': len(selected),
              'contexts': []}
    fixed_input = {'selected': selected, 'target': f['fixed_target'],
        'target_feet': f['fixed_target_feet'],
        'target_velocity': f['fixed_target_velocity'],
        'joint_scale': f['joint_scale'], 'foot_scale': f['foot_scale'],
        'velocity_scale': f['velocity_scale']}
    for ci in range(2):
        command, tendon = raw['command'][ci], raw['tendon'][ci]
        foot = feet_fixed[ci]
        velocity = np.diff(foot, axis=0)/.005
        fixed = scalar_metrics(command, foot, velocity, tendon, fixed_input)
        startup = startup_screen(command, tendon,
            f['startup_target_command'], f['startup_target_tendon'],
            f['time_s'], f['startup_mask'])
        startup_q = command[f['startup_mask']]-f['startup_target_command']
        startup_ltm = tendon[f['startup_mask']]-f['startup_target_tendon']
        startup_per_leg = []
        for leg in range(6):
            q = startup_q[:, 7*leg:7*leg+7]
            ltm = startup_ltm[:, leg]
            values = np.array([np.sqrt(np.mean(q*q)), np.max(np.abs(q)),
                               np.sqrt(np.mean(ltm*ltm)), np.max(np.abs(ltm))])
            startup_per_leg.append({'values': values.tolist(),
                'inner_ratios': ((values/INNER)**2).tolist(),
                'inner_pass': bool(np.all((values/INNER)**2 <= SELECTOR_CAP)),
                'original_pass': bool(np.all(values <= ORIGINAL))})
        own = np.flatnonzero(f['own_mask'][ci])
        own_foot = foot[f['own_mask'][ci, selected]]
        # All common rows are in the fixed E110 set; the E113 own set is
        # exactly the common set and the E110 own set extends five more rows.
        own_target_foot = target_feet_common[ci] if ci == 1 else \
            f['fixed_target_feet']
        own_q = error_stats(command[own], f['own_target'][ci, own])
        own_foot_stats = error_stats(own_foot, own_target_foot)
        own_velocity = error_stats(np.diff(own_foot, axis=0)/.005,
                                   np.diff(own_target_foot, axis=0)/.005)
        own_dq = command[own]-f['own_target'][ci, own]
        own_df = own_foot-own_target_foot
        own_dv = np.diff(own_df, axis=0)/.005
        own_results = {'count': len(own), 'q': own_q,
            'foot_proxy': own_foot_stats, 'velocity_proxy': own_velocity,
            'LF_row6_rms': float(np.sqrt(np.mean(own_dq[:, 6]**2))),
            'per_leg_q_rms': [float(np.sqrt(np.mean(
                own_dq[:, 7*j:7*j+7]**2))) for j in range(6)],
            'per_leg_foot_rms': [float(np.sqrt(np.mean(own_df[:, j]**2)))
                                 for j in range(6)],
            'per_leg_velocity_rms': [float(np.sqrt(np.mean(own_dv[:, j]**2)))
                                     for j in range(6)]}
        final_common_foot = foot[within_fixed]
        base_common_foot = baseline_feet[pair[ci], within_fixed]
        common_target_q = f['own_target'][ci, common]
        common_target_foot = target_feet_common[ci]
        final_common = common_scores(command[common], final_common_foot,
                                     common_target_q, common_target_foot)
        baseline_common = common_scores(baseline_raw['command'][pair[ci], common],
                                        base_common_foot, common_target_q,
                                        common_target_foot)
        ratios = {key: final_common[key]/baseline_common[key]
                  for key in ('q_rms', 'foot_rms', 'velocity_rms')}
        robust = bool(all(item['inner_pass'] for item in startup_per_leg) and
            ratios['q_rms'] <= 1.05 and ratios['foot_rms'] <= .8 and
            ratios['velocity_rms'] <= .8)
        result['contexts'].append({'name': str(f['context_names'][ci]),
            'startup_original': startup, 'startup_per_leg': startup_per_leg,
            'fixed_E110_1828': fixed, 'own_phase': own_results,
            'common_1823': {'final': final_common,
                            'E114_same_context_W113': baseline_common,
                            'ratios': ratios, 'robust_screen_pass': robust}})
    actual_summary = json.loads((Path(data['root'])/
        'e112-dense-foot-recruitment-v2/baseline-summary.json').read_text(
            encoding='utf-8'))
    actual_metrics = actual_summary['actual_metrics']
    result['legacy_E112_fixed_relative'] = native_shape_screen(
        result['contexts'][0]['fixed_E110_1828'], actual_metrics)
    result['legacy_E101_absolute'] = absolute_shape_screen(
        result['contexts'][0]['fixed_E110_1828'])
    result['robust_local_supported'] = bool(all(
        context['common_1823']['robust_screen_pass']
        for context in result['contexts']))
    return result


def run_once(root, out):
    """Single reviewed CUDA fitting process; preserve attempts and raw faults."""
    root, out = Path(root), Path(out)
    process_started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    ledger = {'experiment': 'E115-context-trust-fit', 'stage': 'created',
        'prior_conservative_process_s': PRIOR_PROCESS_S,
        'shared_process_limit_s': SHARED_PROCESS_LIMIT_S,
        'process_wall_s': None, 'process_wall_status': 'not_yet_read',
        'fit_calls': 0, 'fit_wall_s': None,
        'active_leg': None, 'active_delta': None, 'active_delta_index': None,
        'leg_started_evaluations': {leg: 0 for leg in LEGS},
        'leg_completed_evaluations': {leg: 0 for leg in LEGS},
        'leg_raw_saved_evaluations': {leg: 0 for leg in LEGS},
        'leg_context_horizon_s': 10.,
        'leg_contexts_per_unique_delta': 2,
        'leg_context_pairs_started': {leg: 0 for leg in LEGS},
        'leg_context_pairs_completed': {leg: 0 for leg in LEGS},
        'leg_context_pairs_raw_saved': {leg: 0 for leg in LEGS},
        'leg_conditional_model_s_started': {leg: 0. for leg in LEGS},
        'leg_conditional_model_s_completed': {leg: 0. for leg in LEGS},
        'leg_conditional_model_s_raw_saved': {leg: 0. for leg in LEGS},
        'leg_cache_hits': {leg: 0 for leg in LEGS},
        'leg_solver_calls': {leg: 0 for leg in LEGS},
        'leg_wall_s': {leg: None for leg in LEGS},
        'response_call_entered': {leg: 0 for leg in LEGS},
        'final_batch_started': 0, 'final_batch_materialized': 0,
        'final_batch_raw_saved': 0, 'final_pairs_started': 0,
        'final_pairs_materialized': 0, 'final_pairs_raw_saved': 0,
        'final_response_call_entered': False,
        'final_conditional_model_s_started': 0.,
        'final_conditional_model_s_completed': 0.,
        'final_conditional_model_s_raw_saved': 0.,
        'actual_neural_s': 0., 'actual_physical_s': 0.,
        'body_calls': 0, 'secondary_errors': []}
    fit_started = None
    leg_started = {}

    def refresh_clock(primary=None):
        try:
            elapsed = time.monotonic()-process_started
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError('Unverifiable E115 monotonic wall time')
            ledger['process_wall_s'] = elapsed
            ledger['process_wall_status'] = 'available'
            if fit_started is not None:
                ledger['fit_wall_s'] = time.monotonic()-fit_started
            for name, stamp in leg_started.items():
                ledger['leg_wall_s'][name] = time.monotonic()-stamp
        except Exception as secondary:  # retain original failure
            ledger['process_wall_s'] = None
            ledger['process_wall_status'] = 'unavailable'
            if primary is None:
                raise
            ledger['secondary_errors'].append({'at': 'clock',
                'type': type(secondary).__name__, 'message': str(secondary)})
            primary.add_note(f'E115 secondary clock failure: {secondary}')

    def persist():
        refresh_clock()
        save_json(out/'execution-ledger.json', ledger)

    def deadlines(name=None, *, soft=True):
        elapsed = time.monotonic()-process_started
        if elapsed > PROCESS_LIMIT_S or PRIOR_PROCESS_S+elapsed > \
                SHARED_PROCESS_LIMIT_S:
            raise TimeoutError('E115 internal/shared process wall limit')
        if fit_started is not None and time.monotonic()-fit_started > FIT_LIMIT_S:
            raise TimeoutError('E115 fit180 wall limit')
        if name is not None and name in leg_started:
            leg_elapsed = time.monotonic()-leg_started[name]
            if leg_elapsed > LEG_LIMIT_S:
                raise TimeoutError(f'E115 leg28 wall limit: {name}')
            if soft and leg_elapsed >= LEG_LIMIT_S-LEG_FINALIZE_RESERVE_S:
                raise LegSoftStop('leg_wall_finalize_reserve')

    def failure_save(filename, value, primary):
        try:
            save_json(out/filename, value)
        except Exception as secondary:  # noqa: BLE001
            ledger['secondary_errors'].append({'at': filename,
                'type': type(secondary).__name__, 'message': str(secondary)})
            primary.add_note(f'E115 secondary {filename} save: {secondary}')

    primary = None
    full_output = None
    full_raw = {}
    try:
        for original, name in ((ROOT/'tools/fit_c_banc_m3_context_trust.py',
                                'runner.py'),
                               (ROOT/'tests/test_c_banc_context_trust.py',
                                'test_c_banc_context_trust.py'),
                               (ROOT/'docs/BANC_M3_E115_CONTEXT_TRUST_FIT_DESIGN_20260920.md',
                                'frozen-design.md'),
                               (root/'e115-frozen-design.json', 'e115-frozen-design.json'),
                               (root/'e115-legacy-reference-inputs.json',
                                'e115-legacy-reference-inputs.json')):
            shutil.copy2(original, out/name)
        ledger['stage'] = 'preflight'
        persist()
        data = reviewed_inputs(root)
        data['root'] = root
        deadlines()
        import os
        require(all(os.environ.get(key) == '1' for key in
                    ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')),
                'Single-thread CPU environment required')
        torch.set_num_threads(1)
        require(torch.get_num_threads() == 1 and torch.cuda.is_available() and
                'RTX 4080' in torch.cuda.get_device_name(0),
                'Reviewed RTX4080/cuda:0 environment required')
        ledger['environment'] = {'device': torch.cuda.get_device_name(0),
            'torch_version': torch.__version__, 'torch_cuda': torch.version.cuda,
            'torch_threads': torch.get_num_threads(),
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
            solution = None
            stop_reason = None
            try:
                # Initial delta zero is a charged response; no hidden baseline run.
                fit.cache.get_delta(np.zeros(len(stage.edges), dtype=np.float64))
                deadlines(name)
                ledger['leg_solver_calls'][name] = 1
                try:
                    persist()
                except BaseException:
                    ledger['leg_solver_calls'][name] = 0
                    ledger['solver_start_receipt_failed'] = name
                    raise
                solution = solve_leg(fit, data['bounds'][stage.edges])
            except LegSoftStop as stopped:
                stop_reason = stopped.reason
            best_any, best_feasible = choose_records(fit.records)
            solver_report = None if solution is None else {
                'success': bool(solution.success),
                'status': int(solution.status), 'message': str(solution.message),
                'nit': int(solution.nit), 'nfev': int(solution.nfev),
                'njev': int(solution.njev),
                'x': np.asarray(solution.x).tolist(),
                'fun': float(solution.fun)}
            report = {'leg': name, 'solver': solver_report,
                'termination_reason': stop_reason if stop_reason is not None else
                    'scipy_returned',
                'best_any': best_any, 'best_feasible': best_feasible,
                'unique_deltas': len(fit.records),
                'cache_hits': ledger['leg_cache_hits'][name]}
            save_json(out/f'selection-{name}.json', report)
            selected.append(best_feasible)
            ledger.setdefault('leg_termination_reason', {})[name] = \
                report['termination_reason']
            ledger['stage'] = f'leg_{name}_finished'
            ledger['active_leg'] = None
            ledger['active_delta'] = None
            ledger['active_delta_index'] = None
            persist()
            final_leg_wall = time.monotonic()-leg_started[name]
            ledger['leg_wall_s'][name] = final_leg_wall
            if final_leg_wall > LEG_LIMIT_S:
                raise TimeoutError(f'E115 leg28 wall limit: {name}')
            leg_started.pop(name)
        ledger['fit_wall_s'] = time.monotonic()-fit_started
        # Keep the fit clock active until the last selection and persistence
        # have also passed the shared fit deadline.
        deadlines()
        fit_started = None
        deadlines()
        if any(record is None for record in selected):
            ledger['stage'] = 'no_joint_feasible_candidate'
            persist()
            result = {'status': 'no_joint_feasible_candidate',
                'robust_local_supported': False,
                'no_full_candidate_composed': True,
                'feasible_legs': [LEGS[i] for i, record in enumerate(selected)
                                  if record is not None],
                'unfeasible_legs': [LEGS[i] for i, record in enumerate(selected)
                                    if record is None],
                'ledger': ledger}
            save_json(out/'result.json', result)
            return result
        delta = np.zeros(1572, dtype=np.float64)
        for leg, record in enumerate(selected):
            delta[np.flatnonzero(data['edge_mask'][leg])] = record['delta']
        boundary = efficacy_boundary_roundoff(data['efficacy'], delta,
                                              data['bounds'])
        np.savez_compressed(out/'efficacy-boundary-raw.npz',
            delta=delta, bounds=data['bounds'],
            base_efficacy=data['efficacy'], pre=boundary['pre'],
            post=boundary['post'], correction=boundary['correction'],
            tolerance=boundary['tolerance'],
            trust_valid=boundary['trust_valid'],
            original_valid=boundary['original_valid'], valid=boundary['valid'])
        require(np.all(boundary['valid']),
                'Selected efficacy exceeds original or frozen trust bounds')
        efficacy, solver_edge, native_edge, edge_rounding = \
            native_rounding_comparison(data['fixture'][
                'original_signed_edge_weights'], data['efficacy'],
                data['fixture']['weights'][1][data['fixture']['edge_rows'],
                                              data['fixture']['edge_cols']], delta,
                efficacy_override=boundary['post'])
        rows, cols = data['fixture']['edge_rows'], data['fixture']['edge_cols']
        np.savez_compressed(out/'selected-delta.npz', delta=delta,
            efficacy=efficacy, edge_ids=data['fixture']['edge_ids'],
            edge_rows=rows, edge_cols=cols,
            efficacy_pre_roundoff=boundary['pre'],
            efficacy_roundoff_correction=boundary['correction'],
            selected_solver_W=solver_edge,
            expected_native_float32_W=native_edge,
            expected_native_minus_solver=edge_rounding)
        train_final = dict(data['train'])
        train_final['current_w'] = data['fixture']['weights'][1]
        native_w, cal = selected_actual_float32(train_final,
            data['calibration'], efficacy)
        solver_w = data['fixture']['weights'][1].copy()
        solver_w[rows, cols] = solver_edge
        np.savez_compressed(out/'rounding-raw.npz',
            solver_float64_W=solver_w, native_float32_W=native_w,
            selected_solver_W=solver_edge, selected_native_W=native_edge,
            selected_native_minus_solver=edge_rounding,
            signed_source_times_efficacy=data['fixture'][
                'original_signed_edge_weights']*efficacy,
            absolute_difference=np.abs(solver_w-native_w))
        np.savez_compressed(out/'calibration.npz', **cal)
        require(np.array_equal(native_w[rows, cols], native_edge),
                'Native selected float32 calibration/W rounding mismatch')
        deadlines()
        f = data['fixture']
        staged = ContextResponse(time_s=f['time_s'], tau=f['tau'], a=f['a'],
            threshold=f['threshold'], cap=f['cap'], pool=f['pool'],
            moment=f['moment'], rest=f['rest'], ltm=f['ltm'], device='cuda:0')
        tensor = lambda value: torch.as_tensor(value, dtype=torch.float64,
                                                device='cuda:0')
        # Stage all five inputs before charging the actual full batch.
        source = tensor(f['source_rate'])
        background = tensor(f['background'])
        candidate = tensor(native_w[None])
        contexts = torch.tensor([0, 1], dtype=torch.long, device='cuda:0')
        weights = torch.tensor([0, 0], dtype=torch.long, device='cuda:0')
        deadlines()
        ledger.update(stage='final_response', final_batch_started=1,
                      final_pairs_started=2,
                      final_conditional_model_s_started=20.)
        try:
            persist()
        except BaseException:
            ledger.update(final_batch_started=0, final_pairs_started=0,
                          final_conditional_model_s_started=0.,
                          final_start_receipt_failed=True)
            raise
        ledger['final_response_call_entered'] = True
        full_output = staged.evaluate(source, background, candidate,
                                      contexts, weights)
        for key, value in full_output.items():
            full_raw[key] = detach_numpy(value)
        torch.cuda.synchronize()
        ledger.update(stage='final_materialized', final_batch_materialized=1,
                      final_pairs_materialized=2,
                      final_conditional_model_s_completed=20.)
        persist()
        full_raw.update(time_s=f['time_s'], actual_float32_source_weights=native_w,
            pair_context_indices=np.array([0, 1]),
            pair_weight_indices=np.array([0, 0]))
        np.savez_compressed(out/'prediction.npz', **full_raw)
        ledger.update(stage='final_raw_saved', final_batch_raw_saved=1,
                      final_pairs_raw_saved=2,
                      final_conditional_model_s_raw_saved=20.)
        persist()
        deadlines()
        fixed = np.flatnonzero(f['fixed_mask'])
        feet = geometry.full(tensor(full_raw['command'][:, fixed].reshape(-1, 42)))
        feet_fixed = detach_numpy(feet).reshape(2, len(fixed), 6, 3)
        torch.cuda.synchronize()
        np.savez_compressed(out/'foot-proxy-raw.npz', feet_fixed=feet_fixed,
            fixed_mask=f['fixed_mask'], common_mask=f['common_mask'],
            own_mask=f['own_mask'])
        metrics = frozen_output_metrics(full_raw, feet_fixed, data)
        save_json(out/'metrics.json', metrics)
        save_json(out/'provenance.json', {'source_sha256': {
            'runner': sha256(ROOT/'tools/fit_c_banc_m3_context_trust.py'),
            'tests': sha256(ROOT/'tests/test_c_banc_context_trust.py')},
            'reused_source_sha256': {**data['spec']['source_sha256'],
                'tools/fit_c_banc_m3_dense_foot_recruitment.py': DENSE_HELPER_SHA},
            'input_sha256': data['spec']['input_sha256'],
            'legacy_addendum_sha256': LEGACY_ADDENDUM_SHA,
            'independent_baseline_raw_sha256': PARENT_DATA_RAW_SHA,
            'geometry_hash': geometry.geometry_hash,
            'no_native_or_physical_run': True})
        deadline_status = 'within_internal_limits_external_receipt_required'
        deadlines()
        ledger['stage'] = 'complete'
        persist()
        result = {'status': 'completed_conditional_fit',
            'robust_local_supported': metrics['robust_local_supported'],
            'resource_status': deadline_status,
            'fit_or_native_gait_improvement_claim': False,
            'ledger': ledger}
        save_json(out/'result.json', result)
        return result
    except BaseException as error:
        primary = error
        ledger['stage'] = 'error'
        ledger['primary_error'] = {'type': type(error).__name__,
            'message': str(error), 'traceback': traceback.format_exc()}
        refresh_clock(error)
        failure_save('error.json', ledger['primary_error'], error)
        if full_output is not None and ledger['final_batch_raw_saved'] == 0:
            destination = out/'error-final-raw'
            try:
                destination.mkdir(exist_ok=True)
                for key, value in full_output.items():
                    if key not in full_raw:
                        try:
                            full_raw[key] = detach_numpy(value)
                        except Exception as secondary:  # noqa: BLE001
                            error.add_note(f'E115 secondary final {key} transfer: {secondary}')
                for key, value in full_raw.items():
                    try:
                        np.save(destination/f'{key}.npy', value, allow_pickle=False)
                    except Exception as secondary:  # noqa: BLE001
                        error.add_note(f'E115 secondary final {key} save: {secondary}')
            except Exception as secondary:  # noqa: BLE001
                error.add_note(f'E115 secondary final rescue: {secondary}')
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
        'verification/banc-natural-gait-m3-20260919/e115-context-trust-fit')
    args = parser.parse_args()
    run_once(args.root, args.out)


if __name__ == '__main__':
    main()
