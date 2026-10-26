"""Generated E214 connection, ownership, and bounded failure checks.

No test in this module constructs Clarabel or loads the original response.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from tools import fit_c_banc_m3_multiscale_response as e214
from tools.fit_c_banc_m3_global_startup import assemble_global, assess_solution
from tools.verify_c_banc_m3_local_response import LEGS


def generated_locals():
    locals_ = []
    motors = []
    offset = 0
    for leg, width in zip(LEGS, e214.LEG_WIDTHS):
        count = width // 16
        columns = np.arange(offset, offset + count, dtype=np.int64)
        offset += count
        motors.append(columns)
        # The local first three cones have variable-specific NN and two
        # compressed norms. Their ranks are deliberately unlike E211.
        dims = np.array([2 * width + 2 * 693, 2, 3, 694], np.int64)
        rows = int(np.sum(dims))
        local = {"leg": leg, "P": sparse.eye(width, format="csc"),
            "A": sparse.csc_matrix((rows, width)),
            "M": sparse.csc_matrix((693, width)),
            "linear": np.zeros(width), "b": np.zeros(rows),
            "dims": dims, "kinds": ["NonnegativeConeT"]
                + ["SecondOrderConeT"] * 3,
            "motor_columns": columns, "startup_offset": np.zeros(693),
            "objective_constant": 0., "command_limit": 1.,
            "velocity_limit": 1.}
        locals_.append(local)
    return locals_, motors


def test_global_connection_accepts_new_rank_but_pins_owner_and_cones():
    locals_, motors = generated_locals()
    problem = assemble_global(locals_, strict=False)
    free = np.concatenate(motors)
    note = e214.verify_global_structure(problem, locals_, motors, free)
    assert note["variables"] == 5488
    assert note["cones"] == 19
    assert problem["cone_dimensions"][-1] == 4159
    assert problem["startup_matrix"].shape == (4158, 5488)
    assert problem["A"].nnz != 10767527  # No stale E211 nnz rule.

    wrong = [*motors]
    wrong[1] = wrong[1].copy()
    wrong[1][0] = wrong[0][0]
    with pytest.raises(ValueError, match="ownership"):
        e214.verify_global_structure(problem, locals_, wrong, free)
    problem["cone_dimensions"][-1] = 4158
    with pytest.raises(ValueError, match="contract"):
        e214.verify_global_structure(problem, locals_, motors, free)


def test_leg_preparation_is_sequential_and_never_calls_solver(tmp_path):
    motors = [np.arange(width // 16, dtype=np.int64)
              for width in e214.LEG_WIDTHS]
    events = []

    def builder(_start, _D, _q0, _target, ownership, index):
        assert ownership is motors
        assert len(events) == 2 * index
        events.append(("build", index))
        width = e214.LEG_WIDTHS[index]
        return {"leg": LEGS[index], "motor_columns": motors[index],
            "P": sparse.eye(width, format="csc"),
            "A": sparse.csc_matrix((2, width))}

    def saver(path, problem, sha):
        index = LEGS.index(problem["leg"])
        assert path == tmp_path / ("leg-" + LEGS[index])
        assert sha == "frozen-affine"
        events.append(("save", index))

    records = e214.prepare_local_problems(tmp_path, None, None, None, None,
        motors, "frozen-affine", builder=builder, saver=saver)
    assert events == [event for i in range(6)
                      for event in (("build", i), ("save", i))]
    assert len(records) == 6
    assert all(row["local_solver_calls"] == 0 for row in records)
    assert len(json.loads((tmp_path / "local-preparation.json").read_text())) == 6


def test_partial_leg_raw_survives_later_builder_error(tmp_path):
    motors = [np.arange(width // 16, dtype=np.int64)
              for width in e214.LEG_WIDTHS]

    def builder(_start, _D, _q0, _target, ownership, index):
        if index == 1:
            raise RuntimeError("generated second-leg failure")
        width = e214.LEG_WIDTHS[index]
        return {"leg": LEGS[index], "motor_columns": ownership[index],
            "P": sparse.eye(width, format="csc"),
            "A": sparse.csc_matrix((2, width))}

    def saver(path, _problem, _sha):
        (path / "pre-solve.npz").write_bytes(b"first-leg-raw")

    with pytest.raises(RuntimeError, match="second-leg"):
        e214.prepare_local_problems(tmp_path, None, None, None, None, motors,
            "affine", builder=builder, saver=saver)
    assert (tmp_path / ("leg-" + LEGS[0]) / "pre-solve.npz").read_bytes() == b"first-leg-raw"
    assert len(json.loads((tmp_path / "local-preparation.json").read_text())) == 1


def test_bounded_budget_and_raw_assessment_do_not_construct_solver():
    with pytest.raises(e214.SolverAllowanceUnavailable):
        e214.ensure_solve_budget(lambda: 60.00001)
    e214.ensure_solve_budget(lambda: 60.)
    locals_, _ = generated_locals()
    problem = assemble_global(locals_, strict=False)
    raw = {"solver_x": np.zeros(5488),
           "solver_s": np.zeros(problem["A"].shape[0]),
           "solver_z": np.zeros(problem["A"].shape[0])}
    info = {"status": "PrimalInfeasible"}
    report = assess_solution(problem, raw, info, locals_)
    assert not report["eligible_beta"]
    assert report["status"] == "PrimalInfeasible"


def test_bad_gate_emits_invalid_partial_without_solver(tmp_path):
    out = tmp_path / "attempt"
    with pytest.raises(ValueError, match="frozen SHA"):
        e214.run(tmp_path / "missing-freeze", "wrong",
                 tmp_path / "missing-review", "wrong", out)
    result = json.loads((out / "result.json").read_text())
    ledger = json.loads((out / "ledger.json").read_text())
    assert result["validity"] == "incomplete_or_invalid"
    assert result["candidate_adopted"] is False
    assert ledger["solver_calls"] == ledger["local_solver_calls"] == 0
    assert (out / "execution-errors.json").is_file()
    assert (out / "output-manifest.json").is_file()


def test_output_manifest_counts_nested_raw(tmp_path):
    leg = tmp_path / "leg-LF"
    leg.mkdir()
    (leg / "pre-solve.npz").write_bytes(b"raw")
    result = e214.output_manifest(tmp_path)
    assert result["total_bytes"] == 3
    assert result["files"]["leg-LF/pre-solve.npz"]["bytes"] == 3


def test_raw_shape_failure_distinguished_from_unresolved(tmp_path):
    # E211's reused helper must retain a returned malformed raw file before
    # rejecting it as a schema failure. This is a fake return, not a solve.
    from tools.fit_c_banc_m3_global_startup import validate_raw_shapes

    problem = {"A": sparse.csc_matrix((3, 2))}
    raw = {"solver_x": np.zeros(2), "solver_s": np.zeros(3),
           "solver_z": np.zeros(2)}
    np.savez(Path(tmp_path) / "solver-raw.npz", **raw)
    with pytest.raises(ValueError, match="solver_z"):
        validate_raw_shapes(problem, raw)
    with np.load(tmp_path / "solver-raw.npz") as saved:
        assert saved["solver_z"].shape == (2,)
