"""E276 positive unit transforms for an unchanged E275 conic direction problem.

Independent helper only: import never solves, reads original data or runs neural/body
code. There is one solver construction and one solve, with no retry or clipping.
"""

from __future__ import annotations

# Solver and raw-materialization failures carry the identical primary exception.
# ruff: noqa: BLE001
import copy
import numbers
import time

import clarabel
import numpy as np
from scipy import sparse

from tools import fit_c_banc_m3_conic_source as e275


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(mapping, key, shape):
    value = mapping[key]
    _require(
        isinstance(value, np.ndarray)
        and value.dtype == np.float64
        and value.shape == shape
        and np.isfinite(value).all(),
        "Finite float64 shape required: " + key,
    )
    return value


def _validate(original):
    _require(isinstance(original, dict), "E275 conic problem mapping required")
    q = original["q"]
    _require(
        isinstance(q, np.ndarray)
        and q.dtype == np.float64
        and q.ndim == 1
        and q.size > 0
        and q.size % 3 == 0
        and np.isfinite(q).all(),
        "Finite float64 source-triple objective required",
    )
    n = q.size
    p, a = original["P"], original["A"]
    for key, value in (("P", p), ("A", a)):
        _require(
            isinstance(value, sparse.csc_matrix)
            and value.dtype == np.float64
            and value.has_canonical_format
            and value.has_sorted_indices
            and np.isfinite(value.data).all(),
            "Canonical finite float64 CSC required: " + key,
        )
        value.copy().check_format(full_check=True)
    _require(
        p.shape == (n, n) and a.ndim == 2 and a.shape[1] == n,
        "Conic matrix dimensions differ",
    )
    b = _array(original, "b", (a.shape[0],))
    radius = original["radius"]
    _require(
        isinstance(radius, numbers.Real)
        and not isinstance(radius, (bool, np.bool_))
        and np.isfinite(radius)
        and radius > 0,
        "Positive finite trust radius required",
    )
    cones = original["cones"]
    _require(
        isinstance(cones, (tuple, list))
        and len(cones) == 3
        and isinstance(cones[0], clarabel.NonnegativeConeT)
        and all(isinstance(cone, clarabel.SecondOrderConeT) for cone in cones[1:]),
        "E275 Nonnegative/SOC/SOC cone order required",
    )
    dims = [cone.dim for cone in cones]
    _require(
        all(isinstance(dim, int) and dim > 0 for dim in dims) and sum(dims) == len(b),
        "Conic row counts differ",
    )
    _require(
        original["nonnegative_rows"] == dims[0] and dims[1:] == [694, 100],
        "E275 startup cone row dimensions differ",
    )
    for key, shape in (
        ("coefficients", (n // 3, 3)),
        ("gradient", (n,)),
        ("bounds", (n, 2)),
        ("qerr", (99, 7)),
        ("lerr", (99, 1)),
        ("startup_q_jacobian", (99, 7, n)),
        ("startup_ltm_jacobian", (99, 1, n)),
        ("peaks", (2,)),
        ("radii", (2,)),
    ):
        _array(original, key, shape)
    _require(
        np.all(original["bounds"][:, 0] <= original["bounds"][:, 1])
        and np.all(original["peaks"] > 0)
        and np.all(original["radii"] >= 0),
        "Invalid original bounds/peak/SOC radii",
    )
    # Preserve all provided original array fields, including accounting metadata.
    for key, value in original.items():
        if isinstance(value, np.ndarray):
            _require(
                value.dtype == np.float64 and np.isfinite(value).all(),
                "Finite original float64 array required: " + key,
            )
    return float(radius), dims


def equilibrate_problem(original):
    """Return independent transformed arrays plus r and positive cone-safe D."""
    radius, dims = _validate(original)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ar = original["A"] * radius
        p = original["P"] * radius
        _require(
            np.isfinite(ar.data).all() and np.isfinite(p.data).all(),
            "Variable scaling is nonfinite",
        )
        row_max = np.asarray(abs(ar).max(axis=1).toarray()).ravel()
        row_max = np.maximum(row_max, np.abs(original["b"]))
        row_scale = np.empty_like(original["b"])
        start = 0
        for index, dim in enumerate(dims):
            stop = start + dim
            if index == 0:
                row_scale[start:stop] = 1 / np.maximum(row_max[start:stop], 1e-12)
            else:
                row_scale[start:stop] = 1 / max(
                    float(np.max(row_max[start:stop])), 1e-12
                )
            start = stop
        _require(
            np.isfinite(row_scale).all() and np.all(row_scale > 0),
            "Finite positive row scales required",
        )
        a = (sparse.diags(row_scale, format="csc") @ ar).tocsc()
        b = row_scale * original["b"]
    _require(
        np.isfinite(a.data).all() and np.isfinite(b).all(), "Row scaling is nonfinite"
    )
    a.sort_indices()
    p.sort_indices()
    return {
        "P": p,
        "q": original["q"].copy(),
        "A": a,
        "b": b,
        "cones": [
            clarabel.NonnegativeConeT(dims[0]),
            clarabel.SecondOrderConeT(dims[1]),
            clarabel.SecondOrderConeT(dims[2]),
        ],
        "variable_scale": radius,
        "row_scale": row_scale,
    }


def _phase():
    return {
        "requested": 0,
        "completed": 0,
        "failed": 0,
        "wall_s": 0.0,
        "wall_known": True,
    }


def _preserve(error, label, callback):
    try:
        return callback()
    except Exception as secondary:
        error.add_note(label + ": " + repr(secondary))
        return None


def _call(receipt, name, callback, clock):
    phase = receipt[name]
    phase.update(requested=1, wall_s=None, wall_known=False)
    began, primary = None, None
    try:
        began = clock()
        value = callback()
        phase["completed"] = 1
        return value
    except Exception as error:
        primary = error
        phase["failed"] = 1
        raise
    finally:
        if began is not None:

            def elapsed():
                phase.update(wall_s=clock() - began, wall_known=True)

            if primary is None:
                elapsed()
            else:
                _preserve(primary, "Conic phase cost recording failed", elapsed)


def _scalar(value):
    return float(value) if value is not None and np.isfinite(value) else None


def _raw(result):
    raw = {}
    for key, attribute in (
        ("transformed_x", "x"),
        ("transformed_s", "s"),
        ("transformed_z", "z"),
    ):
        try:
            raw[key] = np.array(
                getattr(result, attribute, []), dtype=np.float64, copy=True
            )
        except Exception as error:
            _preserve(
                error,
                "Partial transformed array attachment failed",
                lambda target=error, value=raw: setattr(
                    target, "partial_transformed_raw", value.copy()
                ),
            )
            raise
    return raw


def _restore(raw, transformed):
    radius, scale = transformed["variable_scale"], transformed["row_scale"]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return {
            "direction": radius * raw["transformed_x"],
            "s": raw["transformed_s"] / scale,
            "z": radius * scale * raw["transformed_z"],
        }


def _solution_info(original, result, raw, transformed, receipt):
    n, m = original["q"].size, original["b"].size
    finite = (
        raw["transformed_x"].shape == (n,)
        and raw["transformed_s"].shape == (m,)
        and raw["transformed_z"].shape == (m,)
        and all(np.isfinite(v).all() for v in raw.values())
    )
    restored = (
        _restore(raw, transformed)
        if finite
        else {
            "direction": np.array([], np.float64),
            "s": np.array([], np.float64),
            "z": np.array([], np.float64),
        }
    )
    finite = finite and all(np.isfinite(v).all() for v in restored.values())
    residuals = (
        e275.conic_residuals(original, restored["direction"]) if finite else None
    )
    receipt.update(
        status=str(getattr(result, "status", "incomplete_solution")),
        finite_solution=bool(finite),
        minimum_residual=None if residuals is None else residuals["minimum"],
    )
    iterations = getattr(result, "iterations", None)
    if (
        isinstance(iterations, numbers.Integral)
        and not isinstance(iterations, bool)
        and iterations >= 0
    ):
        receipt.update(iterations=int(iterations), iterations_known=True)
    receipt["info"] = {
        key: _scalar(getattr(result, key, None))
        for key in ("solve_time", "obj_val", "obj_val_dual", "r_prim", "r_dual")
    }
    if finite:
        d, s, z = restored["direction"], restored["s"], restored["z"]
        with np.errstate(over="ignore", invalid="ignore"):
            objective = float(0.5 * d @ (original["P"] @ d) + original["q"] @ d)
            kkt = original["P"] @ d + original["q"] + original["A"].T @ z
            affine = original["A"] @ d + s - original["b"]
        receipt.update(
            physical_objective=_scalar(objective),
            transformed_objective_times_radius=_scalar(
                transformed["variable_scale"] * getattr(result, "obj_val", np.nan)
            ),
            physical_dual_objective=_scalar(
                transformed["variable_scale"] * getattr(result, "obj_val_dual", np.nan)
            ),
            physical_kkt_inf_norm=_scalar(np.max(np.abs(kkt))),
            physical_affine_inf_norm=_scalar(np.max(np.abs(affine))),
        )
    valid = (
        receipt["status"] == "Solved"
        and finite
        and np.isfinite(residuals["minimum"])
        and residuals["minimum"] >= -1e-8
    )
    receipt["solution_valid"] = bool(valid)
    return {
        **restored,
        **raw,
        "valid": bool(valid),
        "residuals": residuals,
        "receipt": receipt,
        "variable_scale": transformed["variable_scale"],
        "row_scale": transformed["row_scale"].copy(),
    }


def solve_equilibrated(
    original,
    time_limit,
    *,
    solver_factory=clarabel.DefaultSolver,
    wall_clock=time.perf_counter,
):
    """Solve once; return original and transformed raw units with strict E275 gates."""
    receipt = {
        "solver_calls_kind": "actual equilibrated Clarabel SOCP calls",
        "clarabel_version": clarabel.__version__,
        "settings": None,
        "status": "requested",
        "iterations": None,
        "iterations_known": False,
        **{
            name: _phase()
            for name in (
                "equilibration",
                "constructor",
                "solve",
                "raw_materialization",
                "partial_retrieval",
            )
        },
    }
    solver, result, transformed, raw = None, None, None, None
    try:
        _require(clarabel.__version__ == "0.11.1", "Exact Clarabel 0.11.1 required")
        _require(
            isinstance(time_limit, numbers.Real)
            and not isinstance(time_limit, (bool, np.bool_))
            and np.isfinite(time_limit)
            and time_limit > 0,
            "Positive remaining solver time required",
        )
        settings, values = e275.solver_settings(min(2.0, float(time_limit)))
        receipt["settings"] = values
        transformed = _call(
            receipt, "equilibration", lambda: equilibrate_problem(original), wall_clock
        )
        solver = _call(
            receipt,
            "constructor",
            lambda: solver_factory(
                transformed["P"],
                transformed["q"],
                transformed["A"],
                transformed["b"],
                transformed["cones"],
                settings,
            ),
            wall_clock,
        )
        result = _call(receipt, "solve", solver.solve, wall_clock)
        raw = _call(receipt, "raw_materialization", lambda: _raw(result), wall_clock)
        return _solution_info(original, result, raw, transformed, receipt)
    except Exception as error:
        receipt["status"] = "execution_failure"
        retrieve = (
            _preserve(
                error,
                "Partial solution accessor failed",
                lambda: getattr(solver, "get_solution", None),
            )
            if result is None and solver is not None
            else None
        )
        if callable(retrieve):
            result = _preserve(
                error,
                "Partial Clarabel solution retrieval failed",
                lambda: _call(receipt, "partial_retrieval", retrieve, wall_clock),
            )
        partial = None
        if result is not None:
            partial = (
                raw
                if raw is not None
                else getattr(error, "partial_transformed_raw", None)
            )
            if partial is None:
                partial = _preserve(
                    error,
                    "Partial conic raw materialization failed",
                    lambda: _raw(result),
                )
            if partial is not None and transformed is not None:
                n, m = original["q"].size, original["b"].size
                if (
                    all(
                        k in partial
                        for k in ("transformed_x", "transformed_s", "transformed_z")
                    )
                    and partial["transformed_x"].shape == (n,)
                    and partial["transformed_s"].shape == (m,)
                    and partial["transformed_z"].shape == (m,)
                ):
                    restored = _preserve(
                        error,
                        "Partial conic unit restoration failed",
                        lambda: _restore(partial, transformed),
                    )
                    if restored is not None:
                        partial.update(restored)
                _preserve(
                    error,
                    "Partial unit metadata retention failed",
                    lambda: partial.update(
                        variable_scale=transformed["variable_scale"],
                        row_scale=transformed["row_scale"].copy(),
                    ),
                )
            iterations = _preserve(
                error,
                "Partial iteration receipt failed",
                lambda: getattr(result, "iterations", None),
            )
            if (
                isinstance(iterations, numbers.Integral)
                and not isinstance(iterations, bool)
                and iterations >= 0
            ):
                receipt.update(iterations=int(iterations), iterations_known=True)
        _preserve(
            error,
            "Conic failure receipt attachment failed",
            lambda target=error: setattr(
                target, "conic_receipt", copy.deepcopy(receipt)
            ),
        )
        _preserve(
            error,
            "Partial solution attachment failed",
            lambda target=error: setattr(target, "partial_conic_solution", result),
        )
        _preserve(
            error,
            "Partial restored raw attachment failed",
            lambda target=error: setattr(
                target, "partial_equilibrated_solution", partial
            ),
        )
        _preserve(
            error,
            "Transformed problem attachment failed",
            lambda target=error: setattr(target, "equilibrated_problem", transformed),
        )
        raise
