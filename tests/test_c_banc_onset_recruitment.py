from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from tools import fit_c_banc_m3_onset_recruitment as e164

FIXED = np.array(
    [4, 15, 16, 19, 30, 31, 34, 35, 46, 47, 52, 63, 64, 67, 78, 82, 83, 94, 95]
)
VARIABLE = np.setdiff1d(np.arange(96), FIXED)


def generated_mapping():
    membership = np.empty(391, dtype=np.int64)
    membership[:330] = np.resize(VARIABLE, 330)
    ltm = np.array([15, 30, 46, 63, 78, 94])
    singleton = np.setdiff1d(FIXED, ltm)
    membership[330:] = np.r_[singleton, np.repeat(ltm, 8)]
    slopes = np.zeros((96, 391, 3, 2), dtype=np.float64)
    for motor, muscle in enumerate(membership[:330]):
        slopes[muscle, motor, :, :] = 0.05
    counts = np.bincount(membership, minlength=96)
    for motor in range(330, 391):
        muscle = membership[motor]
        slopes[muscle, motor, 1, :] = 1.0 / counts[muscle]
    return membership, slopes


def generated_layout():
    membership, slopes = generated_mapping()
    return e164.build_parameter_layout(membership, slopes, FIXED)


def test_onset_feature_rectifies_fast_minus_slow_only():
    bank = np.zeros((2, 3, 3))
    bank[:, :, 0] = [[3, 1, 2], [0, 4, 5]]
    bank[:, :, 2] = [[1, 2, 2], [1, 1, 2]]
    np.testing.assert_array_equal(e164.onset_feature(bank), [[2, 0, 0], [0, 3, 3]])


def test_steady_bank_has_zero_onset_and_fast_slow_exchange_changes_sign():
    rng = np.random.default_rng(164)
    values = rng.random((4, 7))
    bank = np.repeat(values[:, :, None], 3, axis=2)
    assert np.count_nonzero(e164.onset_feature(bank)) == 0
    bank[:, :, [0, 2]] = bank[:, :, [2, 0]]
    assert np.count_nonzero(e164.onset_feature(bank)) == 0
    bank[:, :, 0] += 0.3
    np.testing.assert_allclose(e164.onset_feature(bank), 0.3)


def test_reconstruct_activation_preserves_fixed_and_matches_explicit_formula():
    membership, slopes = generated_mapping()
    rng = np.random.default_rng(1)
    bank = rng.random((5, 391, 3))
    kappa = np.full(391, 0.4)
    original = rng.random((5, 96))
    actual = e164.reconstruct_activation(
        bank, original, slopes, kappa, membership, FIXED
    )
    np.testing.assert_array_equal(actual[:, FIXED], original[:, FIXED])
    muscle = VARIABLE[0]
    members = np.flatnonzero(membership == muscle)
    low = slopes[muscle, members, :, 0]
    delta = slopes[muscle, members, :, 1] - low
    expected = np.sum(low[None] * bank[:, members], axis=(1, 2)) + np.sum(
        delta[None] * np.maximum(bank[:, members] - kappa[members][None, :, None], 0),
        axis=(1, 2),
    )
    np.testing.assert_allclose(actual[:, muscle], expected, atol=0, rtol=0)


def test_parameter_layout_exact_dimensions_support_and_budgets():
    layout = generated_layout()
    assert layout.variable_muscles.size == 77
    assert layout.variable_motors.size == 330
    assert layout.parameter_count == 407
    assert np.array_equal(layout.variable_motors, np.arange(330))
    assert np.all(layout.tonic_lower >= 0.5)
    assert np.all(layout.tonic_upper <= 2)
    assert np.array_equal(layout.lower[:77], np.full(77, 0.5))
    assert np.array_equal(layout.upper[77:], np.full(330, 2.0))


def test_wrong_frozen_mapping_is_rejected():
    membership, slopes = generated_mapping()
    wrong = np.arange(330)
    wrong[-1] = 390
    with pytest.raises(ValueError, match="variable motors"):
        e164.build_parameter_layout(membership, slopes, FIXED, VARIABLE, wrong)


def test_source_review_requires_concrete_approval_and_exact_one_process():
    review = {
        "approved": True,
        "blocking_defects": [],
        "original_processes_allowed": 1,
        "freeze_sha256": e164.FREEZE_SHA,
        "source_sha256": {name: "generated" for name in e164.OWNED},
    }
    e164.validate_source_review_contract(review)
    for key, value in (("approved", False), ("original_processes_allowed", 0)):
        changed = dict(review)
        changed[key] = value
        with pytest.raises(ValueError, match="two-source review"):
            e164.validate_source_review_contract(changed)
    changed = dict(review)
    changed["accepts_single_onset_recruitment_fit"] = True
    changed["approved"] = False
    with pytest.raises(ValueError, match="two-source review"):
        e164.validate_source_review_contract(changed)


def test_apply_transfer_baseline_and_fixed_outputs_are_exact():
    layout = generated_layout()
    rng = np.random.default_rng(2)
    activation = rng.random((8, 96))
    onset = rng.random((8, 391))
    baseline = np.r_[np.ones(77), np.zeros(330)]
    actual = e164.apply_onset_transfer(activation, onset, layout, baseline)
    np.testing.assert_array_equal(actual, activation)
    vector = baseline.copy()
    vector[0] = 0.75
    vector[77] = 0.2
    changed = e164.apply_onset_transfer(activation, onset, layout, vector)
    np.testing.assert_array_equal(changed[:, FIXED], activation[:, FIXED])


def generated_affine_saved(rows=110):
    layout = generated_layout()
    rng = np.random.default_rng(3)
    activation = 0.2 + rng.random((rows, 96))
    onset = rng.random((rows, 391)) * 0.1
    moment = np.zeros((42, 96))
    group_legs = np.repeat(np.arange(6), [17, 15, 16, 17, 15, 16])
    for muscle, leg in enumerate(group_legs):
        moment[7 * leg + muscle % 7, muscle] = (-1) ** muscle * 0.2
    rest = np.linspace(-0.1, 0.1, 42)
    vector = np.r_[np.ones(77), np.zeros(330)]
    baseline_activation = e164.apply_onset_transfer(activation, onset, layout, vector)
    q = rest + 0.02 * (baseline_activation @ moment.T)
    target_indices = np.arange(2, 10)
    pair_start, pair_end = target_indices[:-1], target_indices[1:]
    target = q[target_indices] + 0.01
    target_velocity = (q[pair_end] - q[pair_start]) / 0.005 + 0.1
    startup_target = q[1:100] + 0.005
    saved = {
        "activation": activation,
        "onset": onset,
        "layout": layout,
        "moment": moment,
        "rest": rest,
        "target_indices": target_indices,
        "pair_start": pair_start,
        "pair_end": pair_end,
        "target": target,
        "target_velocity": target_velocity,
        "startup_target": startup_target,
        "startup_tendon": np.zeros((99, 6)),
        "tendon": np.zeros((rows, 6)),
        "startup_ltm_RMS": 0.0,
        "startup_ltm_max_abs": 0.0,
    }
    metrics, _ = e164.direct_metrics(saved, vector)
    saved["baseline_metrics"] = metrics
    return saved


def test_sparse_affine_matches_direct_candidate_for_nontrivial_vector():
    saved = generated_affine_saved()
    layout = saved["layout"]
    vector = np.r_[np.linspace(0.6, 1, 77), np.linspace(0, 0.01, 330)]
    candidate = e164.apply_onset_transfer(
        saved["activation"], saved["onset"], layout, vector
    )
    direct = saved["rest"] + 0.02 * (candidate @ saved["moment"].T)
    for leg in range(6):
        matrix, offset, _ = e164._leg_affine(
            saved["activation"],
            saved["onset"],
            saved["moment"],
            saved["rest"],
            layout,
            leg,
            saved["target_indices"],
            0.02,
        )
        affine = (matrix @ vector + offset).reshape(-1, 7)
        np.testing.assert_allclose(
            affine,
            direct[saved["target_indices"], 7 * leg : 7 * (leg + 1)],
            atol=1e-15,
            rtol=0,
        )


def test_thin_qr_preserves_multiple_generated_norms():
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(120, 8))
    offset = rng.normal(size=120)
    r = e164.thin_qr(matrix, offset)
    rm, ro = e164.e145.compressed_affine(r)
    for vector in (np.zeros(8), np.ones(8), rng.normal(size=8)):
        assert np.linalg.norm(matrix @ vector + offset) == pytest.approx(
            np.linalg.norm(rm @ vector + ro), abs=1e-12, rel=1e-12
        )


def problem_freeze():
    return {
        "contract": {
            "g_bounds": [0.5, 1.0],
            "beta_bounds": [0.0, 2.0],
            "margin_bounds": [-1.0, 1.0],
            "tonic_lower_budget": 0.5,
            "total_upper_budget": 2.0,
            "motor_floor_fraction": 1e-6,
            "physical_atol": 1e-12,
            "baseline_witness_margin": -0.251,
            "proposal_command_RMS_ratio": 0.8,
            "proposal_velocity_RMS_ratio": 1.0,
            "startup_limits": {
                "q_RMS": 0.018,
                "q_max_abs": 0.04,
                "LTM_RMS": 0.009,
                "LTM_max_abs": 0.025,
            },
            "QR_absolute_atol": 1e-12,
            "QR_relative_atol": 1e-10,
            "soc_count": 15,
        }
    }


def test_problem_has_15_soc_exact_bounds_and_feasible_baseline_witness():
    saved = generated_affine_saved(110)
    saved["target_indices"] = np.arange(10, 100)
    saved["pair_start"] = np.arange(10, 99)
    saved["pair_end"] = np.arange(11, 100)
    baseline_vector = np.r_[np.ones(77), np.zeros(330)]
    activation = e164.apply_onset_transfer(
        saved["activation"], saved["onset"], saved["layout"], baseline_vector
    )
    q = saved["rest"] + 0.02 * (activation @ saved["moment"].T)
    saved["target"] = q[saved["target_indices"]] + 0.01
    saved["target_velocity"] = (
        q[saved["pair_end"]] - q[saved["pair_start"]]
    ) / 0.005 + 0.1
    saved["startup_target"] = q[1:100] + 0.005
    saved["baseline_metrics"] = e164.direct_metrics(saved, baseline_vector)[0]
    problem, raw = e164.build_conic_problem(saved, problem_freeze())
    assert len(problem.criteria) == 15
    assert len(problem.cones) == 16
    assert problem.q.shape == (408,)
    assert problem.q[-1] == -1
    assert raw["baseline_witness"].shape == (408,)
    assert np.min(raw["baseline_witness_slack"][: problem.cone_dims[0]]) >= -1e-12


def test_soc_sign_is_norm_over_limit_le_one_minus_margin():
    matrix = sparse.csr_matrix([[2.0, -1.0], [0.5, 3.0]])
    rows, rhs = e164.e145.make_soc_rows(matrix, [0.25, -0.75], 2.5, 2)
    point = np.array([0.2, -0.1, 0.4])
    slack = rhs - rows @ point
    assert slack[0] == pytest.approx(0.6)
    np.testing.assert_allclose(
        slack[1:], (matrix @ point[:2] + np.array([0.25, -0.75])) / 2.5
    )


def generated_full_inputs():
    membership, slopes = generated_mapping()
    n = 2001
    rate = np.ones((n, 391))
    bank = np.ones((n, 391, 3))
    original = np.full((n, 96), 0.2)
    kappa = np.full(391, 0.5)
    activation = e164.reconstruct_activation(
        bank, original, slopes, kappa, membership, FIXED
    )
    moment = np.zeros((42, 96))
    group_legs = np.repeat(np.arange(6), [17, 15, 16, 17, 15, 16])
    for muscle, leg in enumerate(group_legs):
        moment[7 * leg + muscle % 7, muscle] = 0.1
    rest = np.zeros(42)
    command = rest + 0.02 * (activation @ moment.T)
    tendon = 0.2 * np.tanh(activation[:, np.array([15, 30, 46, 63, 78, 94])] / 10)
    counts = np.bincount(membership, minlength=96)
    pool = np.zeros((96, 391))
    pool[membership, np.arange(391)] = 1.0 / counts[membership]
    rows = [
        {
            "control_tick": index,
            "time_s": index * 0.005,
            "joint_targets_rad": command[index].tolist(),
            "motor_rates": rate[index].tolist(),
            "tendon_inputs": {
                leg + "_tarsus": float(tendon[index, j])
                for j, leg in enumerate(e164.LEGS)
            },
        }
        for index in range(n)
    ]
    offsets = np.array([0, 0.5, 0, 0.5, 0, 0.5])
    phase = np.mod(np.arange(n)[:, None] / 100 + offsets, 1.0)
    ref_time = np.arange(2000) * 0.005
    ref_phase = (
        2 * np.pi * np.mod((np.arange(2000) % 200)[:, None] / 200 + offsets, 1.0)
    )
    ref_command = np.tile(np.sin(ref_phase[:, :1]), (1, 42)) * 0.01
    mask = np.zeros(n, bool)
    mask[100:1924] = True
    indices = np.flatnonzero(mask)
    return {
        "recruitment": {
            "bank": bank,
            "activation": activation,
            "original_activation": original,
            "motor_rate": rate,
            "control_tick": np.arange(n),
            "network_tick": np.arange(n) * 20,
            "physics_time_s": np.arange(n) * 0.005,
            **{
                key: np.ones(n, bool)
                for key in (
                    "complete",
                    "bank_available",
                    "activation_available",
                    "original_activation_available",
                    "motor_rate_available",
                )
            },
        },
        "adapter": {
            "membership": membership,
            "motor_ids": np.arange(391),
            "moment": moment,
            "rest": rest,
            "ltm": np.array([15, 30, 46, 63, 78, 94]),
            "pool": pool,
        },
        "parameters": {
            "membership": membership,
            "motor_ids": np.arange(391),
            "slopes": slopes,
            "kappa": kappa,
            "tau": np.array([0.005, 0.01, 0.04]),
        },
        "calibration": {
            "membership": membership.copy(),
            "motor_ids": np.arange(391),
            "moment": moment.copy(),
            "rest": rest.copy(),
            "ltm": np.array([15, 30, 46, 63, 78, 94]),
            "pool": pool.copy(),
        },
        "trace": {
            "body_hash": "generated-body",
            "adapter_hash": "generated-adapter",
            "graph_hash": "generated-graph",
            "samples": rows,
        },
        "state": {
            "graph_hash": "generated-graph",
            "monotone_recruitment_patch": {
                "motor_roster": [f"generated-root-{index}" for index in range(391)]
            },
            "adapter": {
                "hash": "generated-adapter",
                "bank": bank[0].tolist(),
                "activation": activation[0].tolist(),
                "groups": [
                    {
                        "leg": e164.LEGS[leg],
                        "columns": np.flatnonzero(membership == muscle).tolist(),
                    }
                    for muscle, leg in enumerate(group_legs)
                ],
                "parameters": {
                    "motor_roster": [f"generated-root-{index}" for index in range(391)]
                },
            },
        },
        "phase_diagnostic": {
            "control_time_s": np.arange(n) * 0.005,
            "e1_phase": phase,
            "e1_phase_available": np.ones((n, 6), bool),
        },
        "reference": {
            "time_s": ref_time,
            "phase": ref_phase,
            "command": ref_command,
        },
        "target_mask": {
            "control_time_s": np.arange(n) * 0.005,
            "mask": mask,
            "target_indices": indices,
            "pair_start": indices[:-1],
            "pair_end": indices[1:],
        },
        "startup": {
            "mask": np.r_[False, np.ones(99, bool), np.zeros(1901, bool)],
            "target_command": command[1:100],
            "target_tendon": tendon[1:100],
            "recorded_command": command,
            "recorded_tendon": tendon[1:],
        },
    }


def generated_full_freeze(data):
    membership = data["adapter"]["membership"]
    return {
        "sample_contract": {
            "fixed_muscles": FIXED.tolist(),
            "variable_muscles": VARIABLE.tolist(),
            "variable_motors": np.flatnonzero(np.isin(membership, VARIABLE)).tolist(),
            "clock_atol": 2e-8,
            "reconstruction_atol": 1e-12,
            "target_count": 1824,
            "target_first_tick": 100,
            "target_last_tick": 1923,
            "offsets": [0, 0.5, 0, 0.5, 0, 0.5],
            "lf_column": 0,
            "command_gain": 0.02,
            "tendon_max": 0.2,
            "tendon_rate_scale": 10.0,
            "tau_s": [0.005, 0.01, 0.04],
            "group_legs": np.repeat(np.arange(6), [17, 15, 16, 17, 15, 16]).tolist(),
        },
        "body_hash": "generated-body",
        "contract": {"reference_holdout_RMS_limit": 0.002},
    }


def test_generated_full_shape_saved_path_zero_onset_and_recurrence_raw():
    data = generated_full_inputs()
    raw = {}
    saved = e164.prepare_saved(data, generated_full_freeze(data), raw)
    assert saved["layout"].parameter_count == 407
    assert raw["recurrence_residual"].shape == (2000, 391, 3)
    assert np.count_nonzero(raw["recurrence_residual"]) == 0
    assert np.count_nonzero(raw["onset_feature"]) == 0
    np.testing.assert_array_equal(
        saved["activation"][:, FIXED],
        data["recruitment"]["original_activation"][:, FIXED],
    )


def test_generated_full_shape_invalid_clock_is_rejected_after_raw_derivation():
    data = generated_full_inputs()
    data["recruitment"]["physics_time_s"][100] += 3e-8
    with pytest.raises(ValueError, match="clock"):
        e164.prepare_saved(data, generated_full_freeze(data), {})


def test_checkpoint_npy_reference_is_frozen_and_loaded_without_state_store(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(e164, "ROOT", tmp_path)
    state_path = tmp_path / "state.json"
    state_path.write_text("{}", encoding="utf-8")
    array_path = tmp_path / "array_00037.npy"
    expected = np.arange(12, dtype=np.float64).reshape(3, 4)
    np.save(array_path, expected)
    relative = array_path.relative_to(e164.ROOT).as_posix()
    hashes = {relative: e164.sha256(array_path)}
    actual = e164.load_frozen_npy_reference(
        state_path, {"__npy__": "array_00037.npy"}, hashes
    )
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="not frozen"):
        e164.load_frozen_npy_reference(state_path, {"__npy__": "array_00037.npy"}, {})


def test_e103_historical_recording_is_not_used_and_recorded_baseline_is_exact():
    data = generated_full_inputs()
    data["startup"]["recorded_command"][:] = 3.0
    data["startup"]["recorded_tendon"][:] = 0.19
    data["trace"]["samples"][500]["joint_targets_rad"][0] += 5e-13
    raw = {}
    saved = e164.prepare_saved(data, generated_full_freeze(data), raw)
    recorded = np.asarray(
        [row["joint_targets_rad"] for row in data["trace"]["samples"]]
    )
    expected, _ = e164.metrics_from_recorded(saved, recorded)
    assert saved["baseline_metrics"] == expected
    assert raw["target_residual"][500, 0] != 0.0


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda d: d["trace"].update(body_hash="wrong"), "trace clock/ticks"),
        (
            lambda d: d["trace"].update(adapter_hash="wrong"),
            "graph or adapter identity",
        ),
        (lambda d: d["trace"].update(graph_hash="wrong"), "graph or adapter identity"),
        (
            lambda d: d["state"]["adapter"]["parameters"]["motor_roster"].reverse(),
            "root motor roster",
        ),
        (
            lambda d: d["calibration"]["moment"].__setitem__((0, 0), 9.0),
            "calibration/adapter",
        ),
        (
            lambda d: d["state"]["adapter"]["groups"][0].update(columns=[390]),
            "group leg ownership",
        ),
        (
            lambda d: d["reference"]["phase"].__setitem__((slice(None), 1), 0.123),
            "reference interpolation",
        ),
    ],
)
def test_full_generated_provenance_mutations_are_rejected(mutation, match):
    data = generated_full_inputs()
    mutation(data)
    with pytest.raises(ValueError, match=match):
        e164.prepare_saved(data, generated_full_freeze(data), {})


def test_full_1824_by_408_by_15soc_pipeline_and_incremental_qr_raw():
    data = generated_full_inputs()
    freeze = generated_full_freeze(data)
    freeze["contract"].update(problem_freeze()["contract"])
    raw = {}
    saved = e164.prepare_saved(data, freeze, raw)
    problem, returned = e164.build_conic_problem(saved, freeze, raw)
    assert returned is raw
    assert problem.A.shape[1] == 408
    assert len(problem.criteria) == 15
    assert sum(kind == "SecondOrderConeT" for kind in problem.cone_kinds) == 15
    assert "qr_overall_command_0_direct_norms" in raw
    assert raw["qr_overall_command_0_direct_norms"].shape == (3,)
    assert "baseline_witness_slack" in raw


def test_witness_failure_keeps_affine_qr_and_slack_in_caller_raw():
    data = generated_full_inputs()
    freeze = generated_full_freeze(data)
    freeze["contract"].update(problem_freeze()["contract"])
    freeze["contract"]["baseline_witness_margin"] = 1.0
    raw = {}
    saved = e164.prepare_saved(data, freeze, raw)
    with pytest.raises(ValueError, match="witness"):
        e164.build_conic_problem(saved, freeze, raw)
    assert "command_lf_data" in raw
    assert "qr_overall_command_0" in raw
    assert "baseline_witness_slack" in raw


def minimal_run_freeze(tmp_path):
    value = problem_freeze()
    value.update(
        {
            "sample_contract": {"fixed_muscles": FIXED.tolist()},
            "resource_limits": {
                "free_bytes": 0,
                "internal_s": 80,
                "raw_bytes": 10_000_000,
                "fit_s": 20,
            },
            "prior_actual_original_s": 2789.0,
            "prior_conservative_s": 2790,
            "dual_interior_pad": 1e-12,
            "dual_upper_bound_pad": 1e-8,
            "solver_settings": {
                "max_threads": 1,
                "direct_solve_method": "qdldl",
                "max_iter": 100,
                "time_limit": 20.0,
                "verbose": False,
                "tol_gap_abs": 1e-10,
                "tol_gap_rel": 1e-10,
                "tol_feas": 1e-10,
                "tol_infeas_abs": 1e-10,
                "tol_infeas_rel": 1e-10,
            },
        }
    )
    return value


def patch_run_gate(monkeypatch, tmp_path, freeze):
    freeze_file = tmp_path / "freeze.json"
    review_file = tmp_path / "review.json"
    freeze_file.write_text("{}")
    review_file.write_text("{}")
    freeze["design"] = "docs/BANC_M3_E164_ONSET_RECRUITMENT_DESIGN_20260922.md"
    monkeypatch.setattr(
        e164,
        "preflight_contract",
        lambda *args: (freeze, {"source_sha256": {}}, {}),
    )
    return freeze_file, review_file


def dummy_problem():
    return e164.ConicProblem(
        sparse.csc_matrix((408, 408)),
        np.r_[np.zeros(407), -1.0],
        sparse.csc_matrix((1, 408)),
        np.ones(1),
        (e164.clarabel.NonnegativeConeT(1),),
        ("NonnegativeConeT",),
        np.array([1]),
        np.r_[np.zeros(407), -1.0],
        np.r_[np.ones(407) * 2, 1.0],
        (),
        sparse.csr_matrix((0, 407)),
        np.empty(0),
    )


def patch_small_run_math(monkeypatch):
    saved = generated_affine_saved()
    problem = dummy_problem()
    monkeypatch.setattr(e164, "prepare_saved", lambda *_: saved)
    monkeypatch.setattr(
        e164, "build_conic_problem", lambda *_: (problem, {"x": np.ones(1)})
    )
    return saved, problem


def fake_solution(problem):
    x = np.r_[np.ones(77), np.zeros(330), -0.2]
    return SimpleNamespace(
        x=x,
        s=np.asarray(problem.b - problem.A @ x),
        z=np.zeros(problem.b.size),
        status="Solved",
        iterations=2,
        solve_time=0.01,
        obj_val=0.2,
        obj_val_dual=0.2,
        r_prim=0.0,
        r_dual=0.0,
    )


class SuccessfulSolver:
    problem = None

    def __init__(self, *_args):
        self.callback = None

    def set_termination_callback(self, callback):
        self.callback = callback

    def solve(self):
        return fake_solution(self.problem)

    def get_settings(self):
        return e164.clarabel.DefaultSettings()


def test_pre_solve_raw_failure_prevents_solver_construction(monkeypatch, tmp_path):
    freeze = minimal_run_freeze(tmp_path)
    freeze_file, review_file = patch_run_gate(monkeypatch, tmp_path, freeze)
    monkeypatch.setattr(e164, "prepare_saved", lambda *_: {})
    monkeypatch.setattr(
        e164, "build_conic_problem", lambda *_: (dummy_problem(), {"x": np.ones(1)})
    )
    original = e164.atomic_npz

    def fail_pre(path, arrays):
        if Path(path).name == "pre-solve-raw.npz":
            raise OSError("generated pre-save failure")
        return original(path, arrays)

    monkeypatch.setattr(e164, "atomic_npz", fail_pre)
    constructed = False

    def solver(*args):
        nonlocal constructed
        constructed = True

    with pytest.raises(OSError, match="pre-save"):
        e164.run(
            freeze_file,
            "h",
            review_file,
            "s",
            tmp_path / "out",
            input_loader=lambda _: {},
            solver_factory=solver,
        )
    assert constructed is False
    stale = list((tmp_path / "out").glob("result.json.invalid-stale-*.json"))
    if (tmp_path / "out/result.json").exists():
        assert (
            json.loads((tmp_path / "out/result.json").read_text())["validity"]
            == "incomplete"
        )
    assert stale or (tmp_path / "out/result.json").exists()


def test_generated_success_path_records_constructor_callback_solve_and_return(
    monkeypatch, tmp_path
):
    freeze = minimal_run_freeze(tmp_path)
    freeze_file, review_file = patch_run_gate(monkeypatch, tmp_path, freeze)
    _, problem = patch_small_run_math(monkeypatch)
    SuccessfulSolver.problem = problem
    result = e164.run(
        freeze_file,
        "h",
        review_file,
        "s",
        tmp_path / "out",
        input_loader=lambda _: {},
        solver_factory=SuccessfulSolver,
    )
    assert result["validity"] == "valid"
    assert result["conditional_proposal"] is False
    assert result["counts"]["solver_constructor_attempts"] == 1
    assert result["counts"]["callback_registration_completed"] == 1
    assert result["counts"]["fit_calls"] == 1
    assert result["counts"]["solver_returns"] == 1
    assert (tmp_path / "out/solver-return-raw.npz").exists()


def test_returned_solver_arrays_survive_post_return_ledger_failure(
    monkeypatch, tmp_path
):
    freeze = minimal_run_freeze(tmp_path)
    freeze_file, review_file = patch_run_gate(monkeypatch, tmp_path, freeze)
    _, problem = patch_small_run_math(monkeypatch)
    SuccessfulSolver.problem = problem
    expected = fake_solution(problem)
    atomic = e164.atomic_json

    def fail_return_ledger(path, value):
        if path.name == "ledger.json" and value.get("status") == "solve_returned":
            raise OSError("generated post-return ledger failure")
        return atomic(path, value)

    monkeypatch.setattr(e164, "atomic_json", fail_return_ledger)
    out = tmp_path / "out"
    with pytest.raises(OSError, match="post-return ledger"):
        e164.run(
            freeze_file,
            "h",
            review_file,
            "s",
            out,
            input_loader=lambda _: {},
            solver_factory=SuccessfulSolver,
        )
    with np.load(out / "solver-partial.npz", allow_pickle=False) as raw:
        np.testing.assert_array_equal(raw["solver_x"], expected.x)
        np.testing.assert_array_equal(raw["solver_s"], expected.s)
        np.testing.assert_array_equal(raw["solver_z_raw"], expected.z)
        assert "solver_settings_json" in raw and "history_json" in raw
    result = json.loads((out / "result.json").read_text())
    assert result["validity"] == "incomplete"
    assert result["counts"]["solver_returns"] == 1
    assert result["counts"]["fit_calls"] == 1
    assert not (out / "input-partial-raw.npz").exists()


@pytest.mark.parametrize("failure", ["reconstruction", "QR", "witness"])
def test_numerical_gate_evidence_is_on_disk_before_rejection(
    monkeypatch, tmp_path, failure
):
    data = generated_full_inputs()
    freeze = minimal_run_freeze(tmp_path)
    full = generated_full_freeze(data)
    freeze["sample_contract"] = full["sample_contract"]
    freeze["body_hash"] = full["body_hash"]
    freeze["contract"].update(full["contract"])
    freeze["resource_limits"]["raw_bytes"] = 192 * 1024**2
    freeze_file, review_file = patch_run_gate(monkeypatch, tmp_path, freeze)
    if failure == "reconstruction":
        data["trace"]["samples"][120]["joint_targets_rad"][0] += 0.001
    elif failure == "QR":
        qr = e164.thin_qr
        monkeypatch.setattr(e164, "thin_qr", lambda *args: 1.01 * qr(*args))
    else:
        freeze["contract"]["baseline_witness_margin"] = 1.0
    out = tmp_path / "out"
    observed = []
    enforce = e164.enforce_checks

    def verify_saved_before_gate(pending):
        if any(not ok for ok, _ in pending):
            with np.load(out / "pre-solve-raw.npz", allow_pickle=False) as raw:
                stage = "input" if failure == "reconstruction" else "problem"
                np.testing.assert_array_equal(
                    raw[f"{stage}_check_passed"], [ok for ok, _ in pending]
                )
                assert "target_residual" in raw
                if stage == "problem":
                    assert "qr_overall_command_0_direct_norms" in raw
                    assert "baseline_witness_slack" in raw
                    assert "A_data" in raw
            observed.append(stage)
        enforce(pending)

    monkeypatch.setattr(e164, "enforce_checks", verify_saved_before_gate)

    def forbidden_solver(*_args):
        pytest.fail("Numerical gate failure must prevent solver construction")

    with pytest.raises(ValueError, match=failure):
        e164.run(
            freeze_file,
            "h",
            review_file,
            "s",
            out,
            input_loader=lambda _: data,
            solver_factory=forbidden_solver,
        )
    assert len(observed) == 1
    assert not (out / "input-partial-raw.npz").exists()
    ledger = json.loads((out / "ledger.json").read_text())
    assert ledger["fit_calls"] == ledger["solver_constructor_attempts"] == 0


def test_almost_solved_same_finite_physical_vector_is_not_status_rejected(
    monkeypatch, tmp_path
):
    freeze = minimal_run_freeze(tmp_path)
    freeze_file, review_file = patch_run_gate(monkeypatch, tmp_path, freeze)
    saved, problem = patch_small_run_math(monkeypatch)
    for kind in ("overall_command_RMS", "overall_velocity_RMS"):
        saved["baseline_metrics"][kind] = 1e6
    for leg in e164.LEGS:
        saved["baseline_metrics"]["by_leg"][leg]["command_RMS"] = 1e6
        saved["baseline_metrics"]["by_leg"][leg]["velocity_RMS"] = 1e6
    monkeypatch.setattr(e164, "prepare_saved", lambda *_: saved)

    class AlmostSolved(SuccessfulSolver):
        def solve(self):
            answer = fake_solution(self.problem)
            answer.status = "AlmostSolved"
            return answer

    AlmostSolved.problem = problem
    result = e164.run(
        freeze_file,
        "h",
        review_file,
        "s",
        tmp_path / "out",
        input_loader=lambda _: {},
        solver_factory=AlmostSolved,
    )
    assert result["solver_status"] == "AlmostSolved"
    assert result["solver_status_is_solved"] is False
    assert "solver_solved" not in result["gates"]
    assert result["conditional_proposal"] is True


def test_constructor_failure_is_not_counted_as_fit_call(monkeypatch, tmp_path):
    freeze = minimal_run_freeze(tmp_path)
    freeze_file, review_file = patch_run_gate(monkeypatch, tmp_path, freeze)
    patch_small_run_math(monkeypatch)

    def failed_constructor(*args):
        raise RuntimeError("generated constructor failure")

    with pytest.raises(RuntimeError, match="constructor failure"):
        e164.run(
            freeze_file,
            "h",
            review_file,
            "s",
            tmp_path / "out",
            input_loader=lambda _: {},
            solver_factory=failed_constructor,
        )
    ledger = json.loads((tmp_path / "out/ledger.json").read_text())
    assert ledger["solver_constructor_attempts"] == 1
    assert ledger["solver_constructor_completed"] == 0
    assert ledger["fit_calls"] == 0
    assert (tmp_path / "out/solver-partial.npz").exists()


def test_callback_registration_failure_preserves_constructed_count(
    monkeypatch, tmp_path
):
    freeze = minimal_run_freeze(tmp_path)
    freeze_file, review_file = patch_run_gate(monkeypatch, tmp_path, freeze)
    patch_small_run_math(monkeypatch)

    class CallbackFailure:
        def __init__(self, *args):
            pass

        def set_termination_callback(self, callback):
            raise RuntimeError("generated callback failure")

    with pytest.raises(RuntimeError, match="callback failure"):
        e164.run(
            freeze_file,
            "h",
            review_file,
            "s",
            tmp_path / "out",
            input_loader=lambda _: {},
            solver_factory=CallbackFailure,
        )
    ledger = json.loads((tmp_path / "out/ledger.json").read_text())
    assert ledger["solver_constructor_completed"] == 1
    assert ledger["callback_registration_attempts"] == 1
    assert ledger["callback_registration_completed"] == 0
    assert ledger["fit_calls"] == 0


def test_solver_primary_survives_partial_raw_save_failure(monkeypatch, tmp_path):
    freeze = minimal_run_freeze(tmp_path)
    freeze_file, review_file = patch_run_gate(monkeypatch, tmp_path, freeze)
    patch_small_run_math(monkeypatch)

    def fail_solve(*_args, partial, **_kwargs):
        partial["returned_x"] = np.arange(3.0)
        raise RuntimeError("generated solver primary")

    monkeypatch.setattr(e164.e145, "solve_conic_problem", fail_solve)
    original = e164.atomic_npz

    def fail_partial(path, arrays):
        if Path(path).name == "solver-partial.npz":
            raise OSError("generated raw secondary")
        return original(path, arrays)

    monkeypatch.setattr(e164, "atomic_npz", fail_partial)
    with pytest.raises(RuntimeError, match="solver primary"):
        e164.run(
            freeze_file,
            "h",
            review_file,
            "s",
            tmp_path / "out",
            input_loader=lambda _: {},
        )
    result = json.loads((tmp_path / "out/result.json").read_text())
    ledger = json.loads((tmp_path / "out/ledger.json").read_text())
    assert result["primary_error"]["type"] == "RuntimeError"
    assert any(
        item["stage"] == "solver partial raw" for item in ledger["secondary_errors"]
    )


def test_atomic_json_replace_failure_preserves_unique_temp(monkeypatch, tmp_path):
    target = tmp_path / "receipt.json"
    original = Path.replace

    def fail_replace(self, destination):
        if Path(destination) == target:
            raise OSError("generated rename failure")
        return original(self, destination)

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="rename failure"):
        e164.atomic_json(target, {"validity": "incomplete"})
    temporary = list(tmp_path.glob("receipt.json.*.tmp"))
    assert len(temporary) == 1
    assert json.loads(temporary[0].read_text())["validity"] == "incomplete"


@pytest.mark.parametrize("receipt", ["result.json", "output-manifest.json"])
def test_persistent_terminal_publication_failure_removes_canonical_success(
    monkeypatch, tmp_path, receipt
):
    freeze = minimal_run_freeze(tmp_path)
    freeze_file, review_file = patch_run_gate(monkeypatch, tmp_path, freeze)
    _, problem = patch_small_run_math(monkeypatch)
    SuccessfulSolver.problem = problem
    original = e164.atomic_json
    armed = False

    def fail_late(path, value):
        nonlocal armed
        name = Path(path).name
        if name == "result.json" and value.get("validity") == "valid":
            armed = True
        if name == "output-manifest.json":
            armed = True
        if armed and name == receipt:
            raise OSError("generated persistent " + receipt)
        return original(path, value)

    monkeypatch.setattr(e164, "atomic_json", fail_late)
    with pytest.raises(OSError, match="generated persistent"):
        e164.run(
            freeze_file,
            "h",
            review_file,
            "s",
            tmp_path / "out",
            input_loader=lambda _: {},
            solver_factory=SuccessfulSolver,
        )
    result_path = tmp_path / "out/result.json"
    if result_path.exists():
        receipt_result = json.loads(result_path.read_text())
        assert receipt_result["validity"] == "incomplete"
        assert receipt_result["conditional_proposal"] is False
        assert "primary_error" in receipt_result
    assert not (tmp_path / "out/output-manifest.json").exists()


def test_terminal_resource_overrun_revokes_valid_result(monkeypatch, tmp_path):
    freeze = minimal_run_freeze(tmp_path)
    freeze["resource_limits"]["raw_bytes"] = 1
    freeze_file, review_file = patch_run_gate(monkeypatch, tmp_path, freeze)
    _, problem = patch_small_run_math(monkeypatch)
    SuccessfulSolver.problem = problem
    with pytest.raises(ValueError, match="byte limit"):
        e164.run(
            freeze_file,
            "h",
            review_file,
            "s",
            tmp_path / "out",
            input_loader=lambda _: {},
            solver_factory=SuccessfulSolver,
        )
    receipt = json.loads((tmp_path / "out/result.json").read_text())
    assert receipt["validity"] == "incomplete"
    assert receipt["conditional_proposal"] is False
    assert receipt["primary_error"]["type"] == "ValueError"


def test_numpy_json_conversion_and_nonfinite_marker():
    value = e164.jsonable(
        {
            "flag": np.bool_(True),
            "count": np.int64(3),
            "array": np.arange(2),
            "x": np.inf,
        }
    )
    assert value == {
        "flag": True,
        "count": 3,
        "array": [0, 1],
        "x": {"nonfinite": "inf"},
    }


def test_cli_requires_all_gate_arguments(monkeypatch):
    monkeypatch.setattr("sys.argv", ["e164"])
    with pytest.raises(SystemExit):
        e164.main()
