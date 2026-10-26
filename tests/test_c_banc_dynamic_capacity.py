"""Generated E215 algebra and mock returns; no real solver is called here."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from tools import analyze_c_banc_m3_dynamic_capacity as e215


def test_rk4_homogeneous_and_independent_stage_weights():
    tau = np.array([.019999999552965164, .014046886935830116])
    h = e215.DT / tau
    F = e215.rk4_decay(tau)
    expected = (1 - h + h*h/2 - h*h*h/6 + h*h*h*h/24) ** 20
    np.testing.assert_array_equal(F, expected)
    coefficients = e215.rk4_stage_input_coefficients(tau)
    assert coefficients.shape == (2, 4)
    assert np.min(coefficients) > 0
    # Independent four-stage scalar recursion, not bare positive RK weights.
    for row, single_tau in enumerate(tau):
        for stage in range(4):
            u = np.eye(4)[stage]
            k1 = u[0]
            k2 = -(e215.DT/2 * k1) / single_tau + u[1]
            k3 = -(e215.DT/2 * k2) / single_tau + u[2]
            k4 = -(e215.DT * k3) / single_tau + u[3]
            derived = e215.DT / 6 * (k1 + 2*k2 + 2*k3 + k4)
            assert abs(coefficients[row, stage] - derived) < 1e-18


def test_time_major_dynamic_elimination_and_forward_identity():
    problem, data, _target, expected = e215.generated_analytic_case()
    force = np.asarray(problem["forcing_map"] @ expected).reshape(2, 1)
    assert np.all(force >= 0)
    assert np.all(force <= data["S"])
    reconstruction = e215.forward_reconstruction(problem, data, expected)
    np.testing.assert_allclose(reconstruction["activation"],
                               reconstruction["forward_activation"], atol=1e-15)
    np.testing.assert_allclose(reconstruction["q_linear"],
                               reconstruction["q_forward"], atol=1e-15)
    assert problem["forcing_map"].shape == (2, 2)
    np.testing.assert_allclose(problem["forcing_map"].toarray()[1, 0],
                               -(e215.DECAY + .5) / e215.ALPHA)


def test_sparse_cone_order_objective_and_analytic_binding():
    problem, data, target, expected = e215.generated_analytic_case()
    assert problem["cone_types"] == ["NonnegativeConeT"] + ["SecondOrderConeT"] * 2
    n = 2
    nn = problem["A"][:problem["cone_dimensions"][0]].toarray()
    T = problem["forcing_map"].toarray()
    np.testing.assert_array_equal(nn[:n], -T)
    np.testing.assert_array_equal(nn[n:2*n], T)
    np.testing.assert_array_equal(nn[2*n:3*n], -np.eye(n))
    np.testing.assert_array_equal(nn[3*n:4*n], np.eye(n))
    np.testing.assert_array_equal(nn[4*n:4*n+1], [[1., 0.]])
    np.testing.assert_array_equal(nn[4*n+1:4*n+2], [[-1., 0.]])
    assert abs((problem["startup_map"] @ expected + problem["startup_offset"])[0] - .04) < 1e-15
    full_p = (problem["P"].toarray() + problem["P"].T.toarray()
              - np.diag(problem["P"].diagonal()))
    objective = (.5 * expected @ full_p @ expected
        + problem["linear"] @ expected + problem["objective_constant"])
    residual = (problem["command_map"] @ expected + problem["command_offset"])
    assert abs(objective - np.mean(np.square(residual))) < 1e-14
    assert abs(objective - .02205) < 1e-14
    assert np.max(problem["A"][:problem["cone_dimensions"][0]] @ expected
                  - problem["b"][:problem["cone_dimensions"][0]]) <= 1e-14
    # The normal-cone derivative points toward increasing x1 at its upper
    # startup boundary, while x2 reaches its unconstrained main optimum.
    assert full_p[0, 0] * expected[0] + problem["linear"][0] < 0
    assert abs((full_p @ expected + problem["linear"])[1]) < 1e-14
    assert data["S"].shape == (2, 1) and target["target"].shape == (2, 1)


def test_negative_envelope_rejected_without_clipping():
    _problem, data, target, _expected = e215.generated_analytic_case()
    data["S"][0, 0] = -e215.EPS * 2
    with pytest.raises(ValueError, match="materially negative"):
        e215.build_capacity_problem(data, target, [1, 2], [1], [1], [2], 1., 100.)


def test_pre_solve_and_physical_raw_contains_all_sparse_maps(tmp_path):
    problem, data, _target, _expected = e215.generated_analytic_case()
    data = dict(data, muscles=np.array([1]), motors=np.array([2]),
        motor_ids=np.array([3]), membership=np.array([1]),
        pool_owned=np.array([1.]), cap=np.array([2.]), tau=np.array([.02]),
        mass=np.zeros((3, 1)), J=np.zeros((2, 1)),
        G=np.array([2.]), free=np.array([2]), fixed=np.array([0]))
    e215.save_physical_inputs(tmp_path, data)
    e215.save_pre_solve(tmp_path, problem, "synthetic")
    with np.load(tmp_path / "pre-solve.npz", allow_pickle=False) as saved:
        for name in ("P", "A", "forcing_map", "command_map",
                     "velocity_map", "startup_map"):
            assert name + "_data" in saved.files
            assert name + "_indices" in saved.files
            assert name + "_indptr" in saved.files
            assert name + "_shape" in saved.files
        assert str(saved["source_basis_sha256"]) == "synthetic"
    with np.load(tmp_path / "physical-inputs.npz", allow_pickle=False) as saved:
        assert saved["S"].shape == (2, 1)
        assert saved["J"].shape == (2, 1)


def test_mock_return_raw_before_verdict_and_one_call_guard(tmp_path):
    problem, data, target, expected = e215.generated_analytic_case()
    settings_seen = []

    class FakeSolver:
        def __init__(self, _P, _q, _A, _b, _cones, settings):
            settings_seen.append(settings.max_threads)

        def solve(self):
            return SimpleNamespace(x=expected, s=problem["b"] - problem["A"] @ expected,
                z=np.zeros(len(problem["b"])), status="Solved", iterations=2,
                obj_val=-.2, obj_val_dual=-.2, r_prim=0., r_dual=0.,
                solve_time=.001)

    ledger = {"solve_requested": False, "constructor_entered": False,
        "constructor_completed": False, "solve_called": False,
        "solver_returned": False, "solver_calls": 0}
    raw, info = e215.solve_once(problem, tmp_path, ledger,
        elapsed=lambda: 0., solver_factory=FakeSolver)
    assert settings_seen == [1]
    assert ledger["solver_calls"] == 1 and ledger["solver_returned"]
    assert (tmp_path / "solver-raw.npz").is_file()
    assert (tmp_path / "solver-info.json").is_file()
    with pytest.raises(RuntimeError, match="already used"):
        e215.solve_once(problem, tmp_path, ledger,
            elapsed=lambda: 0., solver_factory=FakeSolver)
    diagnosis = e215.solved_diagnostic(problem, data, target, raw, info, tmp_path)
    assert diagnosis["feasible_lm_only"]
    assert diagnosis["main_command_rms"] > 0
    assert (tmp_path / "diagnostic.npz").is_file()


def test_constructor_depletes_budget_without_solve(tmp_path):
    problem, _data, _target, _expected = e215.generated_analytic_case()
    clock = [0.]

    class SlowConstructor:
        def __init__(self, *_args):
            clock[0] = 61.

        def solve(self):
            raise AssertionError("must not solve")

    ledger = {"solve_requested": False, "constructor_entered": False,
        "constructor_completed": False, "solve_called": False,
        "solver_returned": False, "solver_calls": 0}
    with pytest.raises(e215.SolverAllowanceUnavailable):
        e215.solve_once(problem, tmp_path, ledger,
            elapsed=lambda: clock[0], solver_factory=SlowConstructor)
    assert ledger["solve_requested"] and ledger["constructor_completed"]
    assert not ledger["solve_called"] and ledger["solver_calls"] == 0
    assert json.loads((tmp_path / "ledger.json").read_text())["constructor_completed"]


def test_bad_gate_fails_closed_without_solver(tmp_path):
    out = tmp_path / "attempt"
    with pytest.raises(ValueError, match="frozen SHA"):
        e215.run(tmp_path / "missing-freeze", "wrong",
                 tmp_path / "missing-review", "wrong", out)
    result = json.loads((out / "result.json").read_text())
    assert result["validity"] == "incomplete_or_invalid"
    assert result["candidate_adopted"] is False
    assert json.loads((out / "ledger.json").read_text())["solver_calls"] == 0


def test_near_cone_dual_cannot_falsely_certify_feasible_problem(tmp_path):
    # x=0 is feasible, even though 12 tiny negative dual entries together
    # make the uncorrected box separation exceed the numerical margin.
    problem = {"A": sparse.csc_matrix((12, 1)), "b": np.ones(12),
        "cone_dimensions": np.array([12]),
        "cone_types": ["NonnegativeConeT"]}
    raw = {"solver_z": np.full(12, -5e-9)}
    info = {"status": "PrimalInfeasible", "primal_residual": 0.,
            "dual_residual": 0.}
    report = e215.infeasible_diagnostic(problem, raw, info, tmp_path)
    assert report["box_separation_margin"] > e215.PHYSICAL_ATOL
    assert report["dual_cone_violation_max"] == 5e-9
    assert report["certificate_verified"] is False
    assert json.loads((tmp_path / "diagnostic.json").read_text()) == report


def test_exact_dual_cone_member_certifies_box_separation(tmp_path):
    # 0 <= -1 has no feasible x. A=0, z=1 is an exact NN dual member;
    # b'z=-1 is strictly below the box minimum 0.
    problem = {"A": sparse.csc_matrix((1, 1)), "b": np.array([-1.]),
        "cone_dimensions": np.array([1]),
        "cone_types": ["NonnegativeConeT"]}
    raw = {"solver_z": np.array([1.])}
    info = {"status": "PrimalInfeasible", "primal_residual": 0.,
            "dual_residual": 0.}
    report = e215.infeasible_diagnostic(problem, raw, info, tmp_path)
    assert report["dual_cone_violation_max"] == 0.
    assert report["box_separation_margin"] == 1.
    assert report["certificate_verified"] is True
