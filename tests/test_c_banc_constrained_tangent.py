"""Generated-only E206 sparse conic mapping and one tiny known solve."""

import json
from types import SimpleNamespace

import clarabel
import numpy as np
import pytest
from scipy import sparse

from tools import fit_c_banc_m3_constrained_tangent as fit


def synthetic_case():
    membership = np.arange(391) % 96
    groups = [{"leg": fit.LEGS[g // 16]} for g in range(96)]
    moment = np.zeros((42, 96))
    for muscle in range(96):
        moment[7 * (muscle // 16):7 * (muscle // 16) + 7, muscle] = .1
    start = {"adapter": {"parameters": {"membership": membership},
                         "calibration": {"moment": moment}, "groups": groups}}
    d = np.zeros((2001, 391, 7))
    d[1:, 0, 0] = np.linspace(.1, .2, 2000)
    d[1:, 96, 0] = .3  # A second MN on the same muscle.
    d[1:, 16, 1] = -.1  # A different leg.
    scale = np.ones((343, 7))
    weights = np.zeros((391, 7))
    q0 = np.zeros((2001, 42))
    tendon = np.zeros((2001, 6))
    decoded = {"command": q0, "tendon": tendon}
    indices = np.arange(100, 1924)
    target = {"indices": indices, "pair_start": indices[:-1],
        "pair_end": indices[1:], "target": np.full((1824, 42), .01),
        "velocity": np.zeros((1823, 42)),
        "startup_command": np.zeros((99, 42)),
        "startup_tendon": np.zeros((99, 6)),
        "baseline": {"command_rms": .1, "velocity_rms": .1,
            "leg_command_rms": [.1] * 6,
            "leg_velocity_rms": [.1] * 6}}
    free = np.arange(343)
    return start, d, scale, weights, decoded, target, free


def test_augmented_gram_compression_preserves_raw_norms():
    rng = np.random.default_rng(206)
    matrix = rng.normal(size=(80, 6))
    matrix[:, 5] = matrix[:, 1]  # Rank deficiency must not be manually truncated.
    offset = rng.normal(size=80)
    compressed, compressed_offset, note = fit.compress_norm(matrix, offset)
    assert note["relative_frobenius_error"] <= 1e-10
    for _ in range(5):
        u = rng.normal(size=6)
        np.testing.assert_allclose(np.linalg.norm(matrix @ u + offset),
            np.linalg.norm(compressed @ u + compressed_offset),
            atol=1e-10, rtol=1e-10)
    scaled = matrix * 1e-4
    small_factor, small_offset, small_note = fit.compress_norm(scaled, offset * 1e-4)
    small_augmented = np.column_stack((scaled, offset * 1e-4))
    small_gram = small_augmented.T @ small_augmented
    small_compressed = np.column_stack((small_factor, small_offset))
    actual_relative = (np.linalg.norm(small_compressed.T @ small_compressed - small_gram)
                       / np.linalg.norm(small_gram))
    np.testing.assert_allclose(small_note["relative_frobenius_error"], actual_relative,
                               atol=0, rtol=1e-12)
    zeros, zero_offset, zero_note = fit.compress_norm(np.zeros((4, 2)), np.zeros(4))
    assert zero_note["relative_frobenius_error"] == 0
    assert not np.any(zeros) and not np.any(zero_offset)


def test_full_generated_problem_sign_anatomy_objective_and_startup():
    start, d, scale, weights, decoded, target, free = synthetic_case()
    problem = fit.build_problem(start, d, scale, weights, decoded, target, free)
    assert problem["P"].shape == (2401, 2401)
    assert problem["A"].shape[1] == 2401
    assert len(problem["cones"]) == 15
    assert problem["cone_types"].count("SecondOrderConeT") == 14
    assert np.array_equal(problem["D"][0], np.zeros((391, 7)))
    assert 0 in problem["leg_columns"]["lf"]
    assert 96 * 7 in problem["leg_columns"]["lf"]
    assert 16 * 7 + 1 in problem["leg_columns"]["lm"]
    assert 16 * 7 + 1 not in problem["leg_columns"]["lf"]
    u = np.zeros(2401)
    u[0], u[96 * 7], u[16 * 7 + 1] = .1, -.15, .2
    command, delta = fit.predicted_command(problem, start, decoded["command"], u)
    np.testing.assert_array_equal(command[0], decoded["command"][0])
    assert np.all(command[:, 14:] == 0)
    assert np.array_equal(delta[343:], np.zeros((48, 7)))
    residual = command[target["indices"]] - target["target"]
    direct = (np.mean(residual**2) / (.8 * .1)**2 + 1e-8 * np.sum(u**2))
    symmetric_p = problem["P"] + problem["P"].T - sparse.diags(
        problem["P"].diagonal())
    quadratic = (problem["objective_constant"] + problem["linear"] @ u
                 + .5 * u @ (symmetric_p @ u))
    np.testing.assert_allclose(quadratic, direct, atol=1e-12, rtol=1e-12)
    for leg_index, leg in enumerate(fit.LEGS):
        velocity = ((command[target["pair_end"]] - command[target["pair_start"]])
                    / .005 - target["velocity"])
        for kind, source in (("command", residual), ("velocity", velocity)):
            record = problem["norm_records"][leg + "_" + kind]
            local_u = u[record["columns"]]
            compressed_norm = np.linalg.norm(
                record["factor"] @ local_u + record["offset"])
            raw = source[:, 7 * leg_index:7 * leg_index + 7]
            np.testing.assert_allclose(compressed_norm,
                np.sqrt(np.mean(raw**2)) / record["limit"], atol=1e-10, rtol=1e-10)
            assert record["normalizer"] == np.sqrt(raw.size) * record["limit"]
    assert problem["b"][problem["cone_dimensions"][0]] == fit.PAD
    # Clarabel s=b-Au reproduces the literal signed startup residual.
    cone_first = int(problem["cone_dimensions"][0])
    nonnegative_slack = problem["b"][:cone_first] - problem["A"][:cone_first] @ u
    offset = 2 * 2401
    predicted_startup = command[1:100] - target["startup_command"]
    np.testing.assert_allclose(nonnegative_slack[offset:offset + 4158],
        .04 * fit.PAD - predicted_startup.reshape(-1), atol=1e-12)


def test_fake_solver_failure_keeps_primary_and_history(tmp_path, monkeypatch):
    class FakeSolver:
        def __init__(self, *args):
            pass

        def set_termination_callback(self, callback):
            pass

        def solve(self):
            raise RuntimeError("generated sole solve failure")

    monkeypatch.setattr(fit.clarabel, "DefaultSolver", FakeSolver)
    problem = {"P": sparse.csc_matrix((1, 1)), "linear": np.zeros(1),
        "A": sparse.csc_matrix((1, 1)), "b": np.ones(1),
        "cones": [clarabel.NonnegativeConeT(1)]}
    ledger = {"solver_calls": 0, "errors": []}
    with pytest.raises(RuntimeError, match="sole solve failure"):
        fit.solve_once(problem, {"max_iter": 2}, tmp_path, ledger)
    assert ledger["solver_calls"] == 1
    assert (tmp_path / "solver-history.json").is_file()


def test_fake_solved_nonfinite_or_malformed_raw_is_retained_and_rejected(
        tmp_path, monkeypatch):
    class FakeSolved:
        def __init__(self, *args):
            pass

        def set_termination_callback(self, callback):
            pass

        def solve(self):
            return SimpleNamespace(x=[0.], s=[np.nan], z=[1.],
                status="Solved", iterations=1, obj_val=0., obj_val_dual=0.,
                r_prim=0., r_dual=0., solve_time=0.1)

    monkeypatch.setattr(fit.clarabel, "DefaultSolver", FakeSolved)
    problem = {"P": sparse.csc_matrix((1, 1)), "linear": np.zeros(1),
        "A": sparse.csc_matrix((1, 1)), "b": np.ones(1),
        "cones": [clarabel.NonnegativeConeT(1)]}
    ledger = {"solver_calls": 0, "errors": []}
    raw, info = fit.solve_once(problem, {"max_iter": 2}, tmp_path, ledger)
    assert ledger["solver_calls"] == 1
    with np.load(tmp_path / "solver-raw.npz") as saved:
        assert np.isnan(saved["solver_s"][0])
    with pytest.raises(ValueError, match="solver_s"):
        fit.validate_solver_return(raw, info, 1, variables=1)
    raw["solver_s"] = np.zeros(1)
    raw["solver_z"] = np.zeros(2)
    with pytest.raises(ValueError, match="solver_z"):
        fit.validate_solver_return(raw, info, 1, variables=1)
    raw["solver_z"] = np.zeros(1)
    info["objective_primal"] = np.nan
    with pytest.raises(ValueError, match="diagnostics"):
        fit.validate_solver_return(raw, info, 1, variables=1)


def test_generated_single_clarabel_binding_startup(tmp_path):
    """Run this test once only: the sole generated real Clarabel invocation."""
    p = sparse.csc_matrix([[2.]])
    q = np.array([-4.])  # Unconstrained u=2.
    A = sparse.csc_matrix([[1.], [-1.]])
    b = np.array([.5, 1.])  # Startup cap u<=.5 is binding.
    soc_a, soc_b = fit.soc_rows([[1.]], [0.], 1., width=1)
    A = sparse.vstack((A, soc_a), format="csc")
    b = np.r_[b, soc_b]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_threads = 1
    settings.direct_solve_method = "qdldl"
    solver = clarabel.DefaultSolver(p, q, A, b,
        [clarabel.NonnegativeConeT(2), clarabel.SecondOrderConeT(2)], settings)
    solution = solver.solve()
    assert str(solution.status) == "Solved"
    assert abs(solution.x[0] - .5) < 1e-6
    assert abs((b - A @ solution.x)[0]) < 1e-6
    receipt = {"generated_real_solver_calls": 1, "status": str(solution.status),
        "x": float(solution.x[0]), "binding_startup_limit": .5,
        "startup_slack": float((b - A @ solution.x)[0]),
        "objective_without_constant": float(solution.obj_val)}
    (tmp_path / "e206-generated-solve-receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8")
    print("E206_GENERATED_SOLVE", receipt)
