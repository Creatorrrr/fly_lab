"""E216 generated anatomy and mock composition; no actual solver call."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tools import analyze_c_banc_m3_dynamic_capacity as e215
from tools import analyze_c_banc_m3_six_leg_capacity as e216
from tools.verify_c_banc_m3_local_response import LEGS


def synthetic_anatomy():
    counts = e216.LEG_MOTORS
    muscle_counts = e216.LEG_MUSCLES
    membership = []
    groups = []
    for leg, n_motors, n_muscles in zip(LEGS, counts, muscle_counts):
        first = len(groups)
        groups.extend({"leg": leg} for _ in range(n_muscles))
        membership.extend([first] * (n_motors - n_muscles + 1))
        membership.extend(range(first + 1, first + n_muscles))
    assert len(membership) == 343 and len(groups) == 90
    for leg in LEGS:
        groups.append({"leg": leg})
        membership.extend([len(groups) - 1] * 8)
    membership = np.asarray(membership, np.int64)
    assert membership.shape == (391,)
    pool = np.zeros((96, 391))
    pool[membership, np.arange(391)] = 1.
    moment = np.zeros((42, 96))
    for muscle, group in enumerate(groups):
        moment[7 * LEGS.index(group["leg"]), muscle] = 1.
    B = np.zeros((2001, 391, 16))
    B[1:1924, :343, 0] = .01
    motor_ids = np.arange(391, dtype=np.int64) + 1000
    params = {"membership": membership, "pool": pool, "moment": moment,
        "cap": np.ones(391), "tau": np.full(391, .019999999552965164),
        "initial_rate": np.zeros(391), "motor_ids": motor_ids,
        "muscle_parameters_json": np.asarray(json.dumps(groups))}
    mask = {"free_columns": np.arange(343),
        "fixed_columns": np.arange(343, 391),
        "membership": membership.copy(), "motor_ids": motor_ids.copy()}
    return params, mask, {"B": B}, {"q_intercept": np.zeros((2001, 42))}


def test_five_new_legs_use_original_owner_counts_and_specific_baselines():
    params, mask, basis, affine = synthetic_anatomy()
    lm_muscles = np.asarray([g for g in np.unique(params["membership"][:343])
        if json.loads(str(params["muscle_parameters_json"]))[int(g)]["leg"] == "lm"])
    lm_motors = np.asarray([m for g in lm_muscles for m in range(343)
                            if params["membership"][m] == g])
    ownership = e216.verify_all_ownership(params, mask,
        {"motors": lm_motors, "muscles": lm_muscles})
    assert sum(len(row["motors"]) for row in ownership.values()) == 343
    with pytest.raises(ValueError, match="reused LM"):
        e216.verify_all_ownership(params, mask,
            {"motors": lm_motors[:-1], "muscles": lm_muscles})
    selected = []
    for leg in e216.NEW_LEGS:
        data = e216.derive_leg_inputs(params, mask, basis, affine, leg)
        index = LEGS.index(leg)
        assert len(data["motors"]) == e216.LEG_MOTORS[index]
        assert len(data["muscles"]) == e216.LEG_MUSCLES[index]
        assert data["S"].shape == (1923, e216.LEG_MUSCLES[index])
        assert np.min(data["S"]) >= 0
        assert np.all(data["F"] == e215.rk4_decay(.019999999552965164))
        selected.extend(data["motors"].tolist())
    assert len(set(selected)) == 343 - e216.LEG_MOTORS[1]
    with pytest.raises(ValueError, match="five new legs"):
        e216.derive_leg_inputs(params, mask, basis, affine, "lm")


def test_local_target_selects_own_seven_columns_and_baseline_index():
    target = {"target": np.tile(np.arange(42), (1824, 1)),
        "startup_command": np.tile(np.arange(42), (99, 1)),
        "velocity": np.tile(np.arange(42), (1823, 1)),
        "baseline": {"leg_command_rms": [1., 2., 3., 4., 5., 6.]}}
    for index, leg in enumerate(LEGS):
        if leg == "lm":
            continue
        local = e216.local_target(target, index)
        np.testing.assert_array_equal(local["target"][0],
                                      np.arange(7*index, 7*(index+1)))
        assert local["baseline"]["leg_command_rms"][index] == index + 1


def test_combination_requires_six_complete_legs_and_preserves_row0():
    blocks = {leg: np.full((1923, 7), i + 1., np.float64)
              for i, leg in enumerate(LEGS)}
    held = np.arange(42, dtype=np.float64)
    with pytest.raises(ValueError, match="all six"):
        e216.combine_six_leg_command(held, {k: v for k, v in blocks.items()
                                            if k != "rh"})
    combined = e216.combine_six_leg_command(held, blocks)
    assert combined.shape == (1924, 42)
    np.testing.assert_array_equal(combined[0], held)
    for i in range(6):
        np.testing.assert_array_equal(combined[1:, 7*i:7*(i+1)], i+1.)
    with pytest.raises(ValueError, match="1923x7"):
        e216.combine_six_leg_command(held, dict(blocks, rh=np.zeros((1922, 7))))


def test_global_proxy_is_only_computed_on_full_array_and_is_strict():
    q = np.zeros((1924, 42))
    tendon = np.zeros((1924, 6))
    target = {"indices": np.arange(100, 1924),
        "pair_start": np.arange(100, 1923),
        "pair_end": np.arange(101, 1924),
        "target": np.zeros((1824, 42)),
        "velocity": np.zeros((1823, 42)),
        "startup_command": np.zeros((99, 42)),
        "startup_tendon": np.zeros((99, 6)),
        "baseline": {"command_rms": 1., "velocity_rms": 1.,
            "leg_command_rms": [1.] * 6,
            "leg_velocity_rms": [1.] * 6}}
    assert e216.evaluate_global_proxy(q, tendon, target)[
        "strict_original_proxy_pass"] is True
    q[100:, :7] = 1.1
    report = e216.evaluate_global_proxy(q, tendon, target)
    assert report["strict_original_proxy_pass"] is False
    assert report["metrics"]["leg_command_rms"][0] > 1.


def test_call_accounting_excludes_lm_and_blocks_repeats():
    assert e216.count_new_calls({"lf": {"solver_calls": 1},
                                 "rf": {"solver_calls": 0}}) == 1
    with pytest.raises(ValueError, match="reused or unknown"):
        e216.count_new_calls({"lm": {"solver_calls": 1}})
    with pytest.raises(ValueError, match="cap"):
        e216.count_new_calls({"lf": {"solver_calls": 2}})
    with pytest.raises(e215.SolverAllowanceUnavailable):
        e215.ensure_solve_budget(lambda: 61.)
    e215.ensure_solve_budget(lambda: 60.)


def test_leg_solved_diagnostic_uses_actual_index_not_lm(tmp_path):
    problem, data, _generated_target, x = e215.generated_analytic_case()
    baseline = {"leg_command_rms": [.1, 9., 3., 4., 5., 6.]}
    target = {"baseline": baseline}
    raw = {"solver_x": x, "solver_s": np.asarray(problem["b"] - problem["A"] @ x),
           "solver_z": np.zeros(problem["A"].shape[0])}
    metrics, q = e216.assess_leg_solved(problem, data, target, raw,
        {"status": "Solved"}, tmp_path, 0)
    assert metrics["main_command_baseline_rms"] == .1
    assert metrics["leg"] == "lf"
    assert q.shape == (2, 1)
    assert (tmp_path / "diagnostic.npz").is_file()


def test_bad_gate_fails_closed_without_any_leg_call(tmp_path):
    out = tmp_path / "attempt"
    with pytest.raises(ValueError, match="frozen SHA"):
        e216.run(tmp_path / "missing-freeze", "wrong",
                 tmp_path / "missing-review", "wrong", out)
    result = json.loads((out / "result.json").read_text())
    ledger = json.loads((out / "ledger.json").read_text())
    assert result["validity"] == "incomplete_or_invalid"
    assert result["global_proxy_evaluated"] is False
    assert ledger["new_solver_calls"] == 0
    assert ledger["reused_lm_new_solver_calls"] == 0
