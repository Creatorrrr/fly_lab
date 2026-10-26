"""Generated E210 physical-beta conic checks; original fit never run here."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from tools import fit_c_banc_m3_causal_neural_basis as fit


def test_physical_quadratic_sign_constant_and_unpadded_box():
    C = np.array([[1., 2.], [-.5, 1.], [0., -2.]])
    c = np.array([.3, -.4, 1.2])
    P, linear, constant = fit.quadratic_objective(C, c)
    symmetric = P + P.T - sparse.diags(P.diagonal())
    for x in (np.zeros(2), np.array([.4, .9])):
        solver_value = .5 * x @ symmetric @ x + linear @ x + constant
        direct = np.linalg.norm(C @ x + c)**2 + fit.RIDGE * np.mean(x * x)
        assert abs(solver_value - direct) < 1e-13
    A, b = fit.box_and_startup(np.array([[1., -2.]]), np.array([.01]), 2)
    assert A[:2].toarray().tolist() == [[-1., 0.], [0., -1.]]
    assert A[2:4].toarray().tolist() == [[1., 0.], [0., 1.]]
    assert b[:4].tolist() == [0., 0., 1., 1.]
    assert b[4] == pytest.approx(.04 * fit.PAD - .01)


def test_normalized_gram_soc_sign_and_no_weak_column_pruning():
    matrix = np.array([[1., .01], [-.3, 0.], [.2, -.02]])
    offset = np.array([.1, -.3, .4])
    rows, rhs, factor, shifted, note = fit.compressed_soc(
        matrix, offset, .5, 2)
    assert note["normalizer"] == pytest.approx(np.sqrt(3) * .5)
    assert note["relative_frobenius_error"] <= 1e-10
    np.testing.assert_allclose(rows[1:].toarray(), -factor, atol=0, rtol=0)
    np.testing.assert_allclose(rhs[1:], shifted, atol=0, rtol=0)
    for x in (np.zeros(2), np.array([.5, .25])):
        assert np.linalg.norm(factor @ x + shifted) == pytest.approx(
            np.linalg.norm(matrix @ x + offset) / (np.sqrt(3) * .5), abs=1e-12)


def synthetic_anatomy():
    counts = [61, 54, 56, 62, 55, 55]
    free_leg = np.concatenate([np.full(count, k) for k, count in enumerate(counts)])
    membership = np.empty(391, np.int64)
    membership[:343] = free_leg * 16
    membership[343:] = np.repeat(np.arange(6) * 16 + 1, 8)
    moment = np.zeros((42, 96))
    for k in range(6):
        moment[7 * k, 16 * k] = 2
        moment[7 * k, 16 * k + 1] = 1
    groups = [{"leg": fit.LEGS[m // 16]} for m in range(96)]
    start = {"adapter": {"parameters": {"membership": membership},
        "calibration": {"moment": moment}, "groups": groups}}
    return start, np.arange(343), membership, moment


def test_anatomical_mapping_leg_order_and_reject_cross_leg():
    start, free, membership, moment = synthetic_anatomy()
    owners, stored_moment, legs = fit.leg_anatomy(start, free)
    assert [len(v) * 16 for v in legs] == list(fit.LEG_WIDTHS)
    np.testing.assert_array_equal(owners, membership)
    np.testing.assert_array_equal(stored_moment, moment)
    D = np.zeros((3, 391, 16))
    D[1, 0, 0] = 3.
    joint = fit.physical_joint_matrix(D, membership, moment, legs[0], 0)
    assert joint.shape == (3, 7, 976)
    assert joint[0].sum() == 0
    assert joint[1, 0, 0] == pytest.approx(.12)
    assert np.count_nonzero(joint[1]) == 1
    moment[7, 0] = .01
    with pytest.raises(ValueError, match="crosses"):
        fit.leg_anatomy(start, free)


def test_review_gate_rejects_before_original_inputs_or_solver(tmp_path):
    freeze = Path("verification/banc-natural-gait-m3-20260919/"
                  "e210-causal-neural-basis-fit-freeze.json")
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"schema": fit.REVIEW_SCHEMA,
        "approved": False, "blocking_defects": ["synthetic reject"],
        "freeze_sha256": fit.FREEZE_SHA, "source_sha256": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="source review"):
        fit.verify_gate(freeze, fit.FREEZE_SHA, review, fit.file_sha(review))
    with pytest.raises(ValueError, match="frozen SHA"):
        fit.verify_gate(freeze, "0" * 64, review, fit.file_sha(review))


def test_fake_solver_nonfinite_slack_raw_survives(tmp_path, monkeypatch):
    class FakeSolution:
        def __init__(self):
            self.x = [0.5, 0.2]
            self.s = [float("nan"), 1.]
            self.z = [0., 0.]
            self.status = "Solved"
            self.iterations = 3
            self.obj_val = 0.
            self.obj_val_dual = 0.
            self.r_prim = 0.
            self.r_dual = 0.
            self.solve_time = .001

    class FakeSolver:
        def __init__(self, *_args):
            pass

        def solve(self):
            return FakeSolution()

    monkeypatch.setattr(fit.clarabel, "DefaultSolver", FakeSolver)
    leg_dir = tmp_path / "leg-lf"
    leg_dir.mkdir()
    problem = {"P": sparse.eye(2, format="csc"), "linear": np.zeros(2),
        "A": sparse.eye(2, format="csc"), "b": np.ones(2),
        "cones": [fit.clarabel.NonnegativeConeT(2)], "width": 2,
        "cone_dimensions": np.array([2]), "cone_types": ["NonnegativeConeT"],
        "q_matrix": np.zeros((1, 2)), "q_offset": np.zeros(1),
        "velocity_matrix": np.zeros((1, 2)), "velocity_offset": np.zeros(1),
        "startup_matrix": sparse.csr_matrix((1, 2)),
        "startup_offset": np.zeros(1), "command_limit": 1.,
        "velocity_limit": 1., "normalizer": 1., "objective_constant": 0.}
    ledger = {"solver_calls": 0, "solver_wall_s": 0., "legs": {}}
    raw, info = fit.solve_leg_once(problem, leg_dir, ledger)
    report = fit.validate_solver_return(problem, raw, info)
    assert report["eligible_beta"] and not report["numerical_pass"]
    assert ledger["solver_calls"] == 1
    with np.load(leg_dir / "solver-raw.npz") as saved:
        assert np.isnan(saved["solver_s"][0])
    assert (leg_dir / "solver-info.json").is_file()
    assert (tmp_path / "ledger.json").is_file()


def test_constructor_deadline_blocks_solve_and_tracks_request(tmp_path, monkeypatch):
    calls = []

    class FakeSolver:
        def __init__(self, *_args):
            calls.append("construct")

        def solve(self):
            calls.append("solve")

    monkeypatch.setattr(fit.clarabel, "DefaultSolver", FakeSolver)
    problem = {"P": sparse.eye(2, format="csc"), "linear": np.zeros(2),
        "A": sparse.eye(2, format="csc"), "b": np.ones(2),
        "cones": [fit.clarabel.NonnegativeConeT(2)]}
    leg = tmp_path / "leg-lf"
    leg.mkdir()
    ledger = {"solver_calls": 0, "solver_wall_s": 0., "legs": {}}
    entry = {"solve_requested": True, "solve_called": False,
             "solver_returned": False}
    with pytest.raises(TimeoutError, match="allowance"):
        fit.solve_leg_once(problem, leg, ledger, entry=entry,
            remaining_guard=lambda: fit.ensure_leg_budget(lambda: 161.))
    assert calls == ["construct"]
    assert ledger["solver_calls"] == 0
    assert entry["constructor_completed"] and not entry["solve_called"]
    assert (tmp_path / "ledger.json").is_file()


def test_solve_exception_preserves_called_not_returned(tmp_path, monkeypatch):
    class FakeSolver:
        def __init__(self, *_args):
            pass

        def solve(self):
            raise RuntimeError("synthetic solver failure")

    monkeypatch.setattr(fit.clarabel, "DefaultSolver", FakeSolver)
    problem = {"P": sparse.eye(2, format="csc"), "linear": np.zeros(2),
        "A": sparse.eye(2, format="csc"), "b": np.ones(2),
        "cones": [fit.clarabel.NonnegativeConeT(2)]}
    leg = tmp_path / "leg-lf"
    leg.mkdir()
    ledger = {"solver_calls": 0, "solver_wall_s": 0., "legs": {}}
    entry = {"solve_requested": True, "solve_called": False,
             "solver_returned": False}
    with pytest.raises(RuntimeError, match="synthetic solver failure"):
        fit.solve_leg_once(problem, leg, ledger, entry=entry,
            remaining_guard=lambda: None)
    assert ledger["solver_calls"] == 1
    assert entry["constructor_completed"] and entry["solve_called"]
    assert not entry["solver_returned"]
    assert (tmp_path / "ledger.json").is_file()


def test_finite_bounded_solved_beta_survives_direct_training_failure():
    x = np.array([.5, .2])
    problem = {"width": 2, "P": sparse.eye(2, format="csc"),
        "linear": np.zeros(2), "A": sparse.csc_matrix((2, 2)),
        "b": np.ones(2), "cones": [fit.clarabel.NonnegativeConeT(2)],
        "cone_types": ["NonnegativeConeT"], "cone_dimensions": np.array([2]),
        "q_matrix": np.zeros((1, 2)), "q_offset": np.array([2.]),
        "velocity_matrix": np.zeros((1, 2)), "velocity_offset": np.zeros(1),
        "startup_matrix": sparse.csr_matrix((1, 2)),
        "startup_offset": np.zeros(1), "command_limit": 1.,
        "velocity_limit": 1., "normalizer": 1., "objective_constant": 4.}
    raw = {"solver_x": x, "solver_s": np.ones(2), "solver_z": np.zeros(2)}
    info = {"status": "Solved", "iterations": 1,
        "objective_primal": fit.RIDGE * np.mean(x*x), "objective_dual": 0.,
        "primal_residual": 0., "dual_residual": 0., "solve_time_s": .001}
    report = fit.validate_solver_return(problem, raw, info)
    assert report["eligible_beta"]
    assert not report["direct_training_pass"]
    assert report["direct"]["q_rms"] == 2.
    assert np.array_equal(raw["solver_x"], x)
    assert fit.solver_settings().max_threads == 1


def test_pre_call_ledger_failure_does_not_count_an_unmade_solve(tmp_path, monkeypatch):
    calls = []

    class FakeSolver:
        def __init__(self, *_args):
            pass

        def solve(self):
            calls.append("solve")

    original_save = fit.save_json

    def fail_before_solve(path, value):
        if Path(path).name == "ledger.json" and value.get("solver_calls") == 1:
            raise OSError("synthetic ledger failure")
        original_save(path, value)

    monkeypatch.setattr(fit.clarabel, "DefaultSolver", FakeSolver)
    monkeypatch.setattr(fit, "save_json", fail_before_solve)
    problem = {"P": sparse.eye(2, format="csc"), "linear": np.zeros(2),
        "A": sparse.eye(2, format="csc"), "b": np.ones(2),
        "cones": [fit.clarabel.NonnegativeConeT(2)]}
    leg = tmp_path / "leg-lf"
    leg.mkdir()
    ledger = {"solver_calls": 0, "solver_wall_s": 0., "legs": {}}
    entry = {"solve_requested": True, "solve_called": False,
             "solver_returned": False}
    with pytest.raises(OSError, match="synthetic ledger failure"):
        fit.solve_leg_once(problem, leg, ledger, entry=entry)
    assert not calls and ledger["solver_calls"] == 0
    assert entry["solve_requested"] and not entry["solve_called"]


def test_generated_clarabel_single_analytic_boundary(tmp_path):
    """DEFERRED: exactly one small real solve, beta=[.8,0] binding SOC/box."""
    target = np.array([1.5, .5])
    P = sparse.csc_matrix(2 * np.eye(2))
    linear = -2 * target
    box, box_rhs = fit.box_and_startup(sparse.csr_matrix((0, 2)),
                                      np.empty(0), 2)
    soc, soc_rhs = fit.soc_rows(np.array([[1., 1.]]), np.array([0.]), .8,
                                width=2)
    A = sparse.vstack((box, soc), format="csc")
    b = np.r_[box_rhs, soc_rhs]
    problem = {"P": P, "linear": linear, "A": A, "b": b,
        "cones": [fit.clarabel.NonnegativeConeT(len(box_rhs)),
                  fit.clarabel.SecondOrderConeT(len(soc_rhs))],
        "width": 2, "cone_dimensions": np.array([len(box_rhs), len(soc_rhs)]),
        "cone_types": ["NonnegativeConeT", "SecondOrderConeT"]}
    leg_dir = tmp_path / "leg-lf"
    leg_dir.mkdir()
    ledger = {"solver_calls": 0, "solver_wall_s": 0., "legs": {}}
    raw, info = fit.solve_leg_once(problem, leg_dir, ledger)
    assert info["status"] == "Solved"
    np.testing.assert_allclose(raw["solver_x"], [.8, 0.], atol=1e-6, rtol=0)
    assert ledger["solver_calls"] == 1
    print("E210_GENERATED_SOLVER", {"calls": 1,
        "status": info["status"], "x": raw["solver_x"].tolist(),
        "solve_time_s": info["solve_time_s"], "cuda_calls": 0,
        "full_banc_calls": 0, "physics_steps": 0})
