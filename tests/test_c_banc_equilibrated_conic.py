"""Generated <=6-variable CPU conic cases. No original/native/CUDA/body work."""

import copy
import time
from types import SimpleNamespace

import clarabel
import numpy as np
import pytest

from tools import banc_m3_equilibrated_conic as fit

COST = []


def counted_factory(*args):
    started = time.perf_counter()
    solver = clarabel.DefaultSolver(*args)
    row = {
        "constructor_calls": 1,
        "constructor_wall_s": time.perf_counter() - started,
        "solve_calls": 0,
        "iterations": 0,
        "solve_wall_s": 0.0,
    }
    COST.append(row)

    class Counted:
        def solve(self):
            row["solve_calls"] += 1
            began = time.perf_counter()
            try:
                result = solver.solve()
                row["iterations"] += int(result.iterations)
                return result
            finally:
                row["solve_wall_s"] += time.perf_counter() - began

        def get_solution(self):
            return solver.get_solution()

    return Counted()


@pytest.fixture(scope="module", autouse=True)
def report_tiny_solver_cost():
    yield
    print(
        "generated_e276_tiny_clarabel_cost",
        {
            key: sum(row[key] for row in COST)
            for key in (
                "constructor_calls",
                "constructor_wall_s",
                "solve_calls",
                "iterations",
                "solve_wall_s",
            )
        },
    )


def generated_problem(edges=1, jacobian_scale=1.0):
    assert edges <= 2
    n = edges * 3
    jq = np.zeros((99, 7, n), np.float64)
    jl = np.zeros((99, 1, n), np.float64)
    jq[0, 0, 1] = jacobian_scale * 0.2
    jq[98, 3, 1] = jacobian_scale * -0.3
    jl[0, 0, 1] = jacobian_scale * 0.05
    if edges == 2:
        jq[44, 2, 4] = jacobian_scale * 0.1
    current = {
        "coefficients": fit.e275.base.neutral(edges),
        "gradient": np.tile([0.0, -1.0, 0.2], edges),
        "qerr": np.full((99, 7), 0.002, np.float64),
        "lerr": np.full((99, 1), 0.001, np.float64),
        "startup_q_jacobian": jq,
        "startup_ltm_jacobian": jl,
    }
    return fit.e275.conic_problem(current, (0.03, 0.001), 0.05)


@pytest.mark.parametrize("edges", [1, 2])
def test_tiny_actual_solution_recovery_and_original_optimum(edges):
    original = generated_problem(edges)
    reference = fit.e275.solve_direction(original, 2.0, solver_factory=counted_factory)
    solved = fit.solve_equilibrated(original, 9.0, solver_factory=counted_factory)
    assert reference["valid"] and solved["valid"]
    assert solved["receipt"]["status"] == "Solved"
    assert solved["residuals"]["minimum"] >= -1e-8
    assert solved["receipt"]["settings"] == {
        "verbose": False,
        "max_iter": 100,
        "max_threads": 1,
        "tol_gap_abs": 1e-8,
        "tol_gap_rel": 1e-8,
        "tol_feas": 1e-8,
        "time_limit": 2.0,
    }
    expected = (
        0.5 * reference["direction"] @ (original["P"] @ reference["direction"])
        + original["q"] @ reference["direction"]
    )
    assert solved["receipt"]["physical_objective"] == pytest.approx(expected, abs=1e-7)
    assert solved["receipt"]["physical_objective"] == pytest.approx(
        solved["receipt"]["transformed_objective_times_radius"], abs=1e-12
    )
    np.testing.assert_allclose(
        original["A"] @ solved["direction"] + solved["s"], original["b"], atol=1e-8
    )
    assert solved["receipt"]["physical_kkt_inf_norm"] < 1e-7
    assert solved["receipt"]["constructor"]["completed"] == 1
    assert solved["receipt"]["solve"]["completed"] == 1


@pytest.mark.parametrize("magnitude", [1e-10, 1.0, 1e10])
def test_affine_dual_kkt_recovery_soc_single_scale_input_immutable(magnitude):
    original = generated_problem(jacobian_scale=magnitude)
    before = copy.deepcopy(
        {key: value for key, value in original.items() if key != "cones"}
    )
    transformed = fit.equilibrate_problem(original)
    r, d = transformed["variable_scale"], transformed["row_scale"]
    np.testing.assert_array_equal(transformed["q"], original["q"])
    np.testing.assert_array_equal(
        transformed["P"].toarray(), r * original["P"].toarray()
    )
    np.testing.assert_allclose(
        transformed["A"].toarray(), d[:, None] * original["A"].toarray() * r
    )
    np.testing.assert_array_equal(transformed["b"], d * original["b"])
    offset = original["nonnegative_rows"]
    for cone in original["cones"][1:]:
        assert np.all(d[offset : offset + cone.dim] == d[offset])
        offset += cone.dim
    ar = original["A"] * r
    maxima = np.asarray(abs(ar).max(axis=1).toarray()).ravel()
    maxima = np.maximum(np.maximum(maxima, np.abs(original["b"])), 1e-12)
    np.testing.assert_allclose(
        d[: original["nonnegative_rows"]], 1 / maxima[: original["nonnegative_rows"]]
    )
    u = np.array([0.1, 0.2, -0.05])
    spr = transformed["b"] - transformed["A"] @ u
    zpr = np.linspace(-0.001, 0.002, len(spr))
    raw = {"transformed_x": u, "transformed_s": spr, "transformed_z": zpr}
    restored = fit._restore(raw, transformed)
    np.testing.assert_allclose(
        original["A"] @ restored["direction"] + restored["s"],
        original["b"],
        rtol=1e-10,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        original["P"] @ restored["direction"]
        + original["q"]
        + original["A"].T @ restored["z"],
        transformed["P"] @ u + transformed["q"] + transformed["A"].T @ zpr,
        rtol=1e-12,
        atol=1e-8,
    )
    assert (
        restored["direction"][2] < 0
    )  # This helper never silently clips a raw direction.
    for key, value in original.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, before[key])
        elif key in ("P", "A"):
            for field in ("data", "indices", "indptr"):
                np.testing.assert_array_equal(
                    getattr(value, field), getattr(before[key], field)
                )
    assert transformed["q"] is not original["q"]


def fake_factory(original, *, status="Solved", direction=None, nonfinite=False):
    transformed = fit.equilibrate_problem(original)
    d = np.zeros_like(original["q"]) if direction is None else direction
    u = d / transformed["variable_scale"]
    spr = transformed["b"] - transformed["A"] @ u
    if nonfinite:
        u[0] = np.nan
    result = SimpleNamespace(
        status=status,
        iterations=3,
        x=u,
        s=spr,
        z=np.zeros_like(spr),
        obj_val=0.0,
        obj_val_dual=0.0,
        solve_time=0.001,
        r_prim=1e-10,
        r_dual=1e-10,
    )
    return lambda *_: SimpleNamespace(solve=lambda: result)


def test_strict_status_and_original_residual_not_weakened():
    original = generated_problem()
    almost = fit.solve_equilibrated(
        original, 0.2, solver_factory=fake_factory(original, status="AlmostSolved")
    )
    assert almost["residuals"]["minimum"] >= -1e-8 and not almost["valid"]
    bad = fit.solve_equilibrated(
        original,
        0.2,
        solver_factory=fake_factory(original, direction=np.array([0.0, 0.06, 0.0])),
    )
    assert (
        bad["receipt"]["status"] == "Solved"
        and bad["residuals"]["minimum"] < -1e-8
        and not bad["valid"]
    )
    nonfinite = fit.solve_equilibrated(
        original, 0.2, solver_factory=fake_factory(original, nonfinite=True)
    )
    assert not nonfinite["valid"] and nonfinite["residuals"] is None
    incomplete = lambda *_: SimpleNamespace(
        solve=lambda: SimpleNamespace(
            status="Solved", iterations=0, x=[0.0], s=[], z=[]
        )
    )
    result = fit.solve_equilibrated(original, 0.2, solver_factory=incomplete)
    assert not result["valid"] and not result["receipt"]["finite_solution"]


@pytest.mark.parametrize(
    "fault", ["dtype", "shape", "finite", "radius", "cone_count", "cone_order"]
)
def test_invalid_input_rejected_before_constructor(fault):
    original = generated_problem()
    if fault == "dtype":
        original["q"] = original["q"].astype(np.float32)
    elif fault == "shape":
        original["b"] = original["b"][:-1]
    elif fault == "finite":
        original["A"].data[0] = np.nan
    elif fault == "radius":
        original["radius"] = 0.0
    elif fault == "cone_count":
        original["cones"][1] = clarabel.SecondOrderConeT(693)
    elif fault == "cone_order":
        original["cones"][0], original["cones"][1] = (
            original["cones"][1],
            original["cones"][0],
        )

    def forbidden(*_):
        pytest.fail("invalid inputs must not construct solver")

    with pytest.raises(ValueError) as caught:
        fit.solve_equilibrated(original, 2.0, solver_factory=forbidden)
    receipt = caught.value.conic_receipt
    assert receipt["constructor"]["requested"] == receipt["solve"]["requested"] == 0
    assert receipt["equilibration"]["failed"] == 1


def test_constructor_error_preserves_same_primary_and_cost():
    original = generated_problem()
    primary = RuntimeError("generated constructor failure")

    def fail(*_):
        raise primary

    with pytest.raises(RuntimeError) as caught:
        fit.solve_equilibrated(original, 2.0, solver_factory=fail)
    assert caught.value is primary
    assert primary.conic_receipt["constructor"]["failed"] == 1
    assert primary.conic_receipt["constructor"]["wall_s"] >= 0
    assert primary.conic_receipt["solve"]["requested"] == 0


def test_solve_error_retrieves_possible_partial_without_retry():
    original = generated_problem()
    transformed = fit.equilibrate_problem(original)
    primary = RuntimeError("generated solve failure")
    calls = []
    partial = SimpleNamespace(
        status="MaxIterations",
        iterations=7,
        x=np.zeros(3),
        s=transformed["b"],
        z=np.zeros(len(original["b"])),
    )

    def solve():
        calls.append("solve")
        raise primary

    def retrieve():
        calls.append("partial")
        return partial

    factory = lambda *_: SimpleNamespace(solve=solve, get_solution=retrieve)
    with pytest.raises(RuntimeError) as caught:
        fit.solve_equilibrated(original, 2.0, solver_factory=factory)
    assert caught.value is primary and calls == ["solve", "partial"]
    assert primary.partial_conic_solution is partial
    np.testing.assert_array_equal(
        primary.partial_equilibrated_solution["direction"], np.zeros(3)
    )
    np.testing.assert_allclose(
        primary.partial_equilibrated_solution["s"], original["b"]
    )
    assert (
        primary.conic_receipt["iterations"] == 7
        and primary.conic_receipt["solve"]["failed"] == 1
    )


def test_failed_partial_retrieval_never_replaces_primary():
    original = generated_problem()
    primary = RuntimeError("primary solve error")

    def solve():
        raise primary

    def retrieve():
        raise OSError("partial storage error")

    with pytest.raises(RuntimeError) as caught:
        fit.solve_equilibrated(
            original,
            2.0,
            solver_factory=lambda *_: SimpleNamespace(
                solve=solve, get_solution=retrieve
            ),
        )
    assert caught.value is primary
    assert any("partial storage error" in note for note in primary.__notes__)
    assert primary.conic_receipt["partial_retrieval"]["failed"] == 1
    assert (
        primary.conic_receipt["iterations"] is None
        and not primary.conic_receipt["iterations_known"]
    )


def test_raw_storage_error_preserves_native_and_known_partial_arrays():
    original = generated_problem()
    transformed = fit.equilibrate_problem(original)
    primary = OSError("generated raw storage failure")
    reads = []

    class BadArray:
        def __array__(self, dtype=None, copy=None):
            reads.append(1)
            raise primary

    native = SimpleNamespace(
        status="Solved",
        iterations=2,
        x=np.zeros(3),
        s=BadArray(),
        z=np.zeros(len(original["b"])),
    )
    with pytest.raises(OSError) as caught:
        fit.solve_equilibrated(
            original,
            2.0,
            solver_factory=lambda *_: SimpleNamespace(solve=lambda: native),
        )
    assert caught.value is primary and len(reads) == 1
    assert primary.partial_conic_solution is native
    np.testing.assert_array_equal(
        primary.partial_equilibrated_solution["transformed_x"], np.zeros(3)
    )
    assert (
        primary.partial_equilibrated_solution["variable_scale"]
        == transformed["variable_scale"]
    )
    assert primary.conic_receipt["constructor"]["completed"] == 1
    assert primary.conic_receipt["solve"]["completed"] == 1
    assert primary.conic_receipt["raw_materialization"]["failed"] == 1


def test_zero_startup_columns_tiny_bounds_and_zero_ltm_are_not_changed():
    problem = generated_problem()
    current = {
        "coefficients": fit.e275.base.neutral(1),
        "gradient": problem["gradient"].copy(),
        "qerr": problem["qerr"].copy(),
        "lerr": problem["lerr"].copy(),
        "startup_q_jacobian": problem["startup_q_jacobian"].copy(),
        "startup_ltm_jacobian": np.zeros((99, 1, 3), np.float64),
    }
    current["coefficients"][0, 2] = 1e-13
    original = fit.e275.conic_problem(current, (0.03, 0.001), 0.05)
    transformed = fit.equilibrate_problem(original)
    assert original["bounds"][2, 0] == -1e-13
    assert np.all(original["startup_q_jacobian"][:, :, [0, 2]] == 0)
    assert np.all(original["startup_ltm_jacobian"] == 0)
    assert np.isfinite(transformed["row_scale"]).all() and np.all(
        transformed["row_scale"] > 0
    )
    # The lower-box row is scaled by the same explicit formula, without clipping.
    row = len(original["q"]) + 2
    assert transformed["b"][row] == pytest.approx(
        original["b"][row] * transformed["row_scale"][row], abs=0
    )
    assert transformed["b"][row] > 0
    start = original["nonnegative_rows"] + 694
    assert np.all(transformed["row_scale"][start:] == transformed["row_scale"][start])


def test_partial_accessor_storage_failure_preserves_primary():
    original = generated_problem()
    primary = RuntimeError("primary solve failure")

    class Solver:
        def solve(self):
            raise primary

        @property
        def get_solution(self):
            raise OSError("partial accessor storage failure")

    with pytest.raises(RuntimeError) as caught:
        fit.solve_equilibrated(original, 2.0, solver_factory=lambda *_: Solver())
    assert caught.value is primary
    assert any("partial accessor storage failure" in note for note in primary.__notes__)
    assert primary.conic_receipt["solve"]["failed"] == 1
    assert primary.conic_receipt["partial_retrieval"]["requested"] == 0
