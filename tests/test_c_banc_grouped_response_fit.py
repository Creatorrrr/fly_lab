"""E218 physical group-axis and fail-closed mock tests; no actual solve."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import sparse

from tools import fit_c_banc_m3_grouped_response as grouped_fit
from tools.analyze_c_banc_m3_dynamic_capacity import solve_once
from tools.fit_c_banc_m3_global_startup import assemble_global
from tools.fit_c_banc_m3_grouped_response import (
    LEG_GROUPS,
    LEG_WIDTHS,
    PAD,
    bounded_infeasible_diagnostic,
    direct_group_readout,
    group_anatomy,
    group_joint_matrix,
    prepare_local_problems,
    run,
    unpadded_rhs,
    verify_gate,
    verify_group_global_structure,
)
from tools.verify_c_banc_m3_local_response import LEGS


def _anatomy():
    membership = np.empty(391, np.int64)
    membership[:90] = np.arange(90)
    membership[90:343] = np.arange(253) % 90
    membership[343:] = 90 + np.arange(48) % 6
    pool = np.zeros((96, 391))
    pool[membership, np.arange(391)] = 1.
    moment = np.zeros((42, 96))
    groups = []
    for i in range(6):
        for _ in range(LEG_GROUPS[i]):
            groups.append({"leg": LEGS[i]})
    groups.extend({"leg": leg} for leg in LEGS)
    for g, record in enumerate(groups):
        moment[7*LEGS.index(record["leg"]), g] = .1
    return membership, pool, moment, np.arange(90, 96), groups


def test_group_owner_leg_width_and_cross_leg_guard():
    membership, pool, moment, ltm, groups = _anatomy()
    by_leg = group_anatomy(membership, pool, moment, ltm, groups,
                           np.arange(343), np.arange(343, 391))
    assert tuple(len(v) for v in by_leg) == LEG_GROUPS
    assert tuple(len(v)*64 for v in by_leg) == LEG_WIDTHS
    assert np.array_equal(np.sort(np.concatenate(by_leg)), np.arange(90))
    moment[8, 0] = .1
    with pytest.raises(ValueError, match="cross-leg"):
        group_anatomy(membership, pool, moment, ltm, groups,
                      np.arange(343), np.arange(343, 391))


def test_group_physical_jacobian_column_order():
    D = np.zeros((2, 96, 64))
    D[1, 0, 0] = 2
    D[1, 1, 63] = 3
    moment = np.zeros((42, 96))
    moment[0, 0] = .4
    moment[1, 1] = -.5
    J = group_joint_matrix(D, moment, np.array([0, 1]), 0)
    assert J.shape == (2, 7, 128)
    assert J[1, 0, 0] == pytest.approx(.016)
    assert J[1, 1, 127] == pytest.approx(-.03)
    assert np.count_nonzero(J) == 2


def test_group_direct_rate_pool_ema_and_fixed_mn():
    membership, pool, moment, ltm, groups = _anatomy()
    frames = 3
    B = np.zeros((frames, 391, 64))
    B[1:, 0, 0] = [.2, .3]
    D = np.zeros((frames, 96, 64))
    alpha = -np.expm1(-.5)
    D[1, 0, 0] = alpha*.2
    D[2, 0, 0] = D[1, 0, 0] + alpha*(.3-D[1, 0, 0])
    H = np.zeros((frames, 391))
    original = H.copy()
    original[:, 343:] = .5
    activation0 = np.zeros(96)
    q0 = np.zeros((frames, 42))
    old_active = activation0.copy()
    for t in range(1, frames):
        old_active += alpha*(pool @ original[t]-old_active)
        q0[t] = .02*(moment @ old_active)
    arrays = {"D_group": D, "B": B, "H": H,
        "cap": np.ones(391), "membership": membership,
        "fixed": np.arange(343, 391), "original_rates": original,
        "q0": q0, "pool": pool, "moment": moment,
        "rest": np.zeros(42), "ltm": ltm, "groups": groups,
        "activation0": activation0}
    beta = np.zeros((96, 64))
    beta[0, 0] = .25
    _, rate, direct, affine = direct_group_readout(beta, arrays)
    assert np.array_equal(rate[:, 343:], original[:, 343:])
    assert rate[1, 0] == pytest.approx(.05)
    assert np.allclose(direct["command"], affine, atol=1e-16)
    assert np.array_equal(direct["command"][0], q0[0])


def test_unpadded_diagnostic_bounds_keep_solver_padding():
    locals_ = []
    for _ in range(6):
        locals_.append({"P": sparse.eye(2),
            "dims": np.array([4+1386, 3, 3, 694])})
    padded_rows = sum(int(np.sum(v["dims"][:3])) for v in locals_) + 4159
    problem = {"b": np.zeros(padded_rows),
               "cone_dimensions": np.r_[np.tile([1390, 3, 3], 6), 4159]}
    physical = unpadded_rhs(problem, locals_)
    assert physical[4] == pytest.approx(.04*(1-PAD))
    assert physical[1390] == pytest.approx(1-PAD)
    assert physical[1393] == pytest.approx(1-PAD)
    assert physical[-4159] == pytest.approx(.018*(1-PAD))
    assert np.array_equal(problem["b"], np.zeros_like(problem["b"]))


def test_dual_near_cone_never_certifies_feasible_box():
    problem = {"A": sparse.csc_matrix((12, 1)), "b": np.ones(12),
        "cone_types": ["NonnegativeConeT"],
        "cone_dimensions": np.array([12])}
    info = {"status": "PrimalInfeasible", "dual_residual": 0.}
    near = bounded_infeasible_diagnostic(problem,
        {"solver_z": np.full(12, -5e-9)}, info)
    assert near["box_separation_margin"] > 1e-8
    assert not near["certificate_verified"]
    problem["b"] = -np.ones(12)
    strict = bounded_infeasible_diagnostic(problem,
        {"solver_z": np.ones(12)}, info)
    assert strict["certificate_verified"]
    assert "conditional" in strict["scope"]


def test_sequential_group_local_save_without_solver(tmp_path):
    by_leg = []
    cursor = 0
    for count in LEG_GROUPS:
        by_leg.append(np.arange(cursor, cursor+count))
        cursor += count
    invoked = []

    def fake_builder(_D, _q0, _target, _moment, groups, index):
        invoked.append(index)
        width = len(groups[index])*64
        return {"leg": LEGS[index], "width": width,
            "group_columns": groups[index], "P": sparse.eye(width),
            "A": sparse.csc_matrix((1, width))}

    def fake_saver(path, problem, _sha):
        np.savez(path / "pre-solve.npz",
                 group_columns=problem["group_columns"])

    records = prepare_local_problems(tmp_path, None, None, None, None,
        by_leg, "fake", builder=fake_builder, saver=fake_saver)
    assert invoked == list(range(6))
    assert [r["variables"] for r in records] == list(LEG_WIDTHS)
    assert all(r["local_solver_calls"] == 0 for r in records)
    assert all((tmp_path / ("leg-"+leg) / "pre-solve.npz").is_file()
               for leg in LEGS)


def test_generic_global_assembly_restores_group_contract():
    by_leg, cursor, locals_ = [], 0, []
    for index, count in enumerate(LEG_GROUPS):
        columns = np.arange(cursor, cursor+count)
        by_leg.append(columns)
        cursor += count
        width = count*64
        dims = np.array([2*width+1386, 2, 2, 694])
        locals_.append({"leg": LEGS[index], "group_columns": columns,
            "motor_columns": columns, "P": sparse.eye(width, format="csc"),
            "A": sparse.csc_matrix((int(dims.sum()), width)),
            "M": sparse.csc_matrix((693, width)),
            "b": np.zeros(int(dims.sum())), "linear": np.zeros(width),
            "objective_constant": 0., "startup_offset": np.zeros(693),
            "dims": dims, "kinds": ["NonnegativeConeT"]
                + ["SecondOrderConeT"]*3})
    global_problem = assemble_global(locals_, strict=False)
    metrics = verify_group_global_structure(global_problem, locals_,
                                            by_leg, np.arange(90))
    assert metrics["variables"] == 5760 and metrics["cones"] == 19
    assert global_problem["startup_matrix"].shape == (4158, 5760)
    broken = dict(global_problem, motor_columns=np.r_[
        global_problem["motor_columns"][:-1], 88])
    with pytest.raises(ValueError, match="structure"):
        verify_group_global_structure(broken, locals_, by_leg, np.arange(90))


def test_mock_constructor_failure_preserves_request_accounting(tmp_path):
    problem = {"P": sparse.eye(1, format="csc"),
        "linear": np.zeros(1), "A": sparse.csc_matrix((1, 1)),
        "b": np.ones(1), "cones": []}
    ledger = {"solve_requested": False, "constructor_entered": False,
        "constructor_completed": False, "solve_called": False,
        "solver_returned": False, "solver_calls": 0}

    def failing_constructor(*_args):
        raise RuntimeError("mock constructor failure")

    with pytest.raises(RuntimeError, match="mock constructor failure"):
        solve_once(problem, tmp_path, ledger, elapsed=lambda: 0.,
                   solver_factory=failing_constructor)
    saved = json.loads((tmp_path / "ledger.json").read_text())
    assert saved["solve_requested"] and saved["constructor_entered"]
    assert not saved["solve_called"] and saved["solver_calls"] == 0
    assert not (tmp_path / "solver-raw.npz").exists()


def test_nonfinite_direct_raw_saved_before_rejection(tmp_path, monkeypatch):
    by_leg, cursor = [], 0
    for count in LEG_GROUPS:
        by_leg.append(np.arange(cursor, cursor+count))
        cursor += count
    malformed = np.full((2, 42), np.nan)

    def fake_direct(beta, _arrays):
        return (beta, np.zeros((2, 391)),
                {"command": malformed, "activation": np.zeros((2, 96)),
                 "tendon": np.zeros((2, 6))}, malformed)

    monkeypatch.setattr(grouped_fit, "direct_group_readout", fake_direct)
    with pytest.raises(ValueError, match="nonfinite"):
        grouped_fit.publish_reconstruction(tmp_path, {}, {}, {}, by_leg,
            [], {}, {"solver_x": np.zeros(5760)}, {}, {})
    with np.load(tmp_path / "reconstruction-raw.npz", allow_pickle=False) as raw:
        assert np.isnan(raw["q_direct"]).all()
        assert raw["beta"].shape == (96, 64)


def test_gate_failure_retains_invalid_result(tmp_path):
    with pytest.raises(ValueError):
        verify_gate(tmp_path / "missing", "bad", tmp_path / "review", "bad")
    out = tmp_path / "failed"
    with pytest.raises(ValueError):
        run(tmp_path / "missing", "bad", tmp_path / "review", "bad", out)
    result = json.loads((out / "result.json").read_text())
    assert result["validity"] == "incomplete_or_invalid"
    assert result["solver_calls"] == 0
