"""E172: two frozen, bounded saved-QR conic diagnostics.

Only selected E169/E170 QR members are opened, after the parent source gate.
No original feature bank, body, or fitted state is reconstructed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import clarabel
import numpy as np
import scipy
from scipy import sparse
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "tools/diagnose_c_banc_m3_bounded_qr.py"
TESTS = "tests/test_c_banc_bounded_qr.py"
KINDS = ("command", "velocity", "startup")
DENOMINATORS = {"command": 1824 * 42, "velocity": 1823 * 42, "startup": 99 * 42}
LEG_DENOMINATORS = {"command": 1824 * 7, "velocity": 1823 * 7, "startup": 99 * 7}
EPS = np.finfo(np.float64).eps


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def helper_module() -> Any:
    """Load only the reviewed E171 selected loader, validator and diagnostics."""
    spec = importlib.util.spec_from_file_location(
        "e171_saved_qr_helper", ROOT / "tools/diagnose_c_banc_m3_saved_qr_capacity.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight(freeze_path: Path, gate_path: Path, root: Path = ROOT) -> dict[str, Any]:
    """All source, version, CPU and ZIP directory checks precede np.load."""
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    require(
        freeze["schema"] == "flylab.m3.e172.bounded-qr-capacity.v1", "freeze schema"
    )
    require(freeze["family_order"] == ["static", "history"], "family order")
    require(sha256(freeze_path) == gate.get("freeze_sha256"), "gate freeze hash")
    require(
        gate.get("schema") == "flylab.m3.e172.parent-source-review.v1", "gate schema"
    )
    require(
        gate.get("accepts_single_bounded_qr_diagnostic") is True, "parent acceptance"
    )
    require(
        freeze["design_sha256"]
        == gate.get("design_sha256")
        == sha256(root / freeze["design"]),
        "design hash",
    )
    require(
        sha256(root / freeze["base_freeze"]) == freeze["base_freeze_sha256"],
        "base freeze hash",
    )
    require(set(gate.get("source_sha256", {})) == {SOURCE, TESTS}, "gate source names")
    for name in (SOURCE, TESTS):
        require(
            sha256(root / name) == gate["source_sha256"][name],
            f"gate source hash: {name}",
        )
    require(
        freeze["reused_helper_sha256"]
        == gate.get("reused_helper_sha256")
        == sha256(root / freeze["reused_helper"]),
        "reviewed helper hash",
    )
    require(
        freeze["archive_key_map"]
        == {
            "pair_start": "affine_rows_velocity_start",
            "pair_end": "affine_rows_velocity_end",
        },
        "archive aliases",
    )
    require(
        len(freeze["allowed_keys"]) == 43 and len(set(freeze["allowed_keys"])) == 43,
        "43 selected keys",
    )
    require(
        freeze["shared_input_hash_count"] == len(freeze["shared_input_hashes"]),
        "shared input count",
    )
    for name, expected in freeze["input_sha256"].items():
        require(sha256(root / name) == expected, f"input hash: {name}")
    prior = [
        json.loads((root / name).read_text(encoding="utf-8"))
        for name in (
            "verification/banc-natural-gait-m3-20260919/e169-closed-loop-recruitment-freeze.json",
            "verification/banc-natural-gait-m3-20260919/e170-history-recruitment-freeze-v2.json",
        )
    ]
    for item in prior:
        require(item["sample_contract"] == freeze["sample_contract"], "sample contract")
        for name, expected in freeze["shared_input_hashes"].items():
            require(item["input_sha256"].get(name) == expected, f"shared hash: {name}")
    helper = helper_module()
    require(
        set(freeze["allowed_keys"])
        == {
            "pair_motors",
            "pair_filters",
            "pair_muscles",
            "w0",
            "target_indices",
            "pair_start",
            "pair_end",
        }
        | {
            f"qr_{kind}_{leg}{suffix}"
            for kind in KINDS
            for leg in freeze["sample_contract"]["legs"]
            for suffix in ("", "_active")
        },
        "selected key set",
    )
    for family in freeze["family_order"]:
        helper.validate_archive_members(
            root / freeze["families"][family]["raw"],
            freeze["allowed_keys"],
            freeze["archive_key_map"],
        )
    require(np.__version__ == freeze["versions"]["numpy"], "NumPy version")
    require(scipy.__version__ == freeze["versions"]["scipy"], "SciPy version")
    require(clarabel.__version__ == freeze["versions"]["clarabel"], "Clarabel version")
    pools = threadpool_info()
    require(
        all(pool.get("num_threads") == 1 for pool in pools),
        f"CPU/BLAS threads: {pools}",
    )
    require(freeze["resource_limits"]["cpu_threads"] == 1, "CPU thread contract")
    return {"freeze": freeze, "helper": helper, "threadpool_info": pools}


def baseline_vector(saved: dict[str, np.ndarray], family: str) -> np.ndarray:
    w0 = np.asarray(saved["w0"], dtype=np.float64)
    return w0.copy() if family == "static" else np.r_[w0, np.zeros_like(w0)]


def local_residual(
    saved: dict[str, np.ndarray], family: str, kind: str, c: np.ndarray, legs: list[str]
) -> dict[str, np.ndarray]:
    result = {}
    for leg in legs:
        key = f"qr_{kind}_{leg}"
        qr = saved[key]
        result[leg] = qr[:, :-1] @ c[saved[f"{key}_active"]] + qr[:, -1]
    return result


def global_residual(problem: dict[str, Any], c: np.ndarray) -> np.ndarray:
    return np.asarray(problem["M"] @ c + problem["offset"])


def compare_local_global(
    saved: dict[str, np.ndarray],
    family: str,
    problem: dict[str, Any],
    c: np.ndarray,
    freeze: dict[str, Any],
    label: str,
) -> dict[str, float]:
    direct = np.concatenate(
        list(
            local_residual(
                saved, family, "command", c, freeze["sample_contract"]["legs"]
            ).values()
        )
    )
    global_value = global_residual(problem, c)
    atol = freeze["contract"]["QR_absolute_atol"]
    rtol = freeze["contract"]["QR_relative_atol"]
    require(
        np.allclose(global_value, direct, atol=atol, rtol=rtol),
        f"global/local QR pointwise {label}",
    )
    require(
        abs(np.linalg.norm(global_value) - np.linalg.norm(direct))
        <= atol + rtol * np.linalg.norm(direct),
        f"global/local QR norm {label}",
    )
    return {
        "global_norm": float(np.linalg.norm(global_value)),
        "direct_norm": float(np.linalg.norm(direct)),
        "max_abs_gap": float(np.max(np.abs(global_value - direct))),
    }


def build_problem(
    saved: dict[str, np.ndarray], family: str, freeze: dict[str, Any]
) -> dict[str, Any]:
    """Pure sparse scatter and fixed SOCP construction; supports generated small cones."""
    require(family in ("static", "history"), "family")
    ncoef = int(freeze["families"][family]["coefficient_count"])
    legs = freeze["sample_contract"]["legs"]
    rows, cols, values, offsets = [], [], [], []
    row_start = 0
    for leg in legs:
        key = f"qr_command_{leg}"
        qr = np.asarray(saved[key], dtype=np.float64)
        active = np.asarray(saved[f"{key}_active"], dtype=np.int64)
        require(qr.ndim == 2 and qr.shape[1] == active.size + 1, f"QR shape {leg}")
        require(
            np.all((active >= 0) & (active < ncoef))
            and np.unique(active).size == active.size,
            f"active {leg}",
        )
        loc = sparse.coo_matrix(qr[:, :-1])
        rows.extend((loc.row + row_start).tolist())
        cols.extend(active[loc.col].tolist())
        values.extend(loc.data.tolist())
        offsets.append(qr[:, -1].copy())
        row_start += qr.shape[0]
    M = sparse.csc_matrix((values, (rows, cols)), shape=(row_start, ncoef))
    offset = np.concatenate(offsets)
    baseline = baseline_vector(saved, family)
    require(baseline.size == ncoef, "baseline dimension")
    baseline_norm = float(np.linalg.norm(M @ baseline + offset))
    require(
        np.isfinite(baseline_norm) and baseline_norm > 0, "positive finite baseline"
    )
    target = float(freeze["contract"]["command_ratio"] * baseline_norm)
    guard = float(freeze["contract"]["inner_guard"])
    bound = float(freeze["contract"]["coefficient_bound"])
    n = ncoef + 1
    eye = sparse.eye(n, format="csc")
    linear_blocks = [-eye, eye]
    rhs = np.r_[np.full(ncoef, bound - guard), 1.0]
    linear_rhs = [rhs.copy(), rhs.copy()]
    if family == "history":
        require(ncoef % 2 == 0, "history halves")
        half = ncoef // 2
        endpoint = sparse.hstack(
            (
                sparse.eye(half, format="csc"),
                sparse.eye(half, format="csc"),
                sparse.csc_matrix((half, 1)),
            ),
            format="csc",
        )
        linear_blocks.extend((-endpoint, endpoint))
        linear_rhs.extend((np.full(half, bound - guard), np.full(half, bound - guard)))
    A_linear = sparse.vstack(linear_blocks, format="csc")
    b_linear = np.concatenate(linear_rhs)
    soc_head = sparse.csc_matrix(([target], ([0], [ncoef])), shape=(1, n))
    soc_tail = sparse.hstack((-M, sparse.csc_matrix((row_start, 1))), format="csc")
    A = sparse.vstack((A_linear, soc_head, soc_tail), format="csc")
    b = np.r_[b_linear, target, offset]
    q = np.zeros(n)
    q[-1] = -1.0
    lower = np.r_[np.full(ncoef, -bound), -1.0]
    upper = -lower
    inner_lower = np.r_[np.full(ncoef, -bound + guard), -1.0]
    inner_upper = -inner_lower
    true_b = b.copy()
    true_b[:ncoef] += guard
    true_b[n : n + ncoef] += guard
    if family == "history":
        true_b[2 * n : A_linear.shape[0]] += guard
    result = {
        "M": M,
        "offset": offset,
        "A": A,
        "b": b,
        "true_b": true_b,
        "q": q,
        "P": sparse.csc_matrix((n, n)),
        "cones": [
            clarabel.NonnegativeConeT(A_linear.shape[0]),
            clarabel.SecondOrderConeT(row_start + 1),
        ],
        "linear_rows": A_linear.shape[0],
        "soc_rows": row_start + 1,
        "baseline": baseline,
        "baseline_norm": baseline_norm,
        "target": target,
        "lower": lower,
        "upper": upper,
        "inner_lower": inner_lower,
        "inner_upper": inner_upper,
    }
    shape = freeze.get("problem_shapes", {}).get(family)
    if shape is not None:
        require(
            list(M.shape) == shape["M_shape"] and list(A.shape) == shape["A_shape"],
            "frozen matrix shape",
        )
        require(
            (n, A_linear.shape[0], row_start + 1)
            == (shape["variable_count"], shape["linear_rows"], shape["soc_rows"]),
            "frozen cone shape",
        )
    return result


def inner_witness(problem: dict[str, Any], freeze: dict[str, Any]) -> np.ndarray:
    guard = float(freeze["contract"]["inner_guard"])
    x = np.r_[
        (1 - guard) * problem["baseline"], freeze["contract"]["inner_witness_margin"]
    ]
    require(bool(np.isfinite(x).all()), "inner witness finite")
    require(
        np.all(
            problem["b"][: problem["linear_rows"]]
            - problem["A"][: problem["linear_rows"]] @ x
            >= -1e-12
        ),
        "inner witness linear",
    )
    soc = (
        problem["b"][problem["linear_rows"] :]
        - problem["A"][problem["linear_rows"] :] @ x
    )
    require(soc[0] >= np.linalg.norm(soc[1:]) - 1e-12, "inner witness SOC")
    return x


def project_dual(z: np.ndarray, linear_rows: int, pad: float) -> np.ndarray:
    value = np.asarray(z, dtype=np.float64)
    require(
        value.ndim == 1 and value.size > linear_rows and np.isfinite(value).all(),
        "dual finite/dimension",
    )
    linear = np.maximum(value[:linear_rows], 0)
    cone = value[linear_rows:]
    head, tail = float(cone[0]), cone[1:]
    length = float(np.linalg.norm(tail))
    if length <= head:
        projected = cone.copy()
    elif length <= -head:
        projected = np.zeros_like(cone)
    else:
        projected = np.r_[0.5 * (length + head), 0.5 * (1 + head / length) * tail]
    projected[0] = max(projected[0], np.linalg.norm(projected[1:])) + pad * (
        1 + np.linalg.norm(projected[1:])
    )
    result = np.r_[linear, projected]
    require(
        np.isfinite(result).all()
        and np.all(result[:linear_rows] >= 0)
        and result[linear_rows] >= np.linalg.norm(result[linear_rows + 1 :]),
        "projected dual cone",
    )
    return result


def dual_bound(
    problem: dict[str, Any], z: np.ndarray, true_family: bool, pad: float
) -> dict[str, Any]:
    """Upper bound on maximal t after minimizing stationarity over true boxes."""
    b = problem["true_b"] if true_family else problem["b"]
    h = np.asarray(problem["q"] + problem["A"].T @ z)
    lower = problem["lower"] if true_family else problem["inner_lower"]
    upper_box = problem["upper"] if true_family else problem["inner_upper"]
    terms = np.minimum(h * lower, h * upper_box)
    b_dot = float(b @ z)
    correction = float(pad * (1 + abs(b_dot) + np.sum(np.abs(terms))))
    upper = float(b_dot - np.sum(terms) + correction)
    require(np.isfinite(upper) and np.isfinite(h).all(), "dual bound finite")
    return {
        "upper": upper,
        "b_dot_z": b_dot,
        "box_min_sum": float(np.sum(terms)),
        "pad": correction,
        "stationarity": h,
        "box_terms": terms,
    }


def qr_metrics(
    saved: dict[str, np.ndarray], family: str, c: np.ndarray, freeze: dict[str, Any]
) -> tuple[dict[str, Any], list[float]]:
    legs = freeze["sample_contract"]["legs"]
    baseline = baseline_vector(saved, family)
    metrics = {}
    resolutions = []
    for kind in KINDS:
        actual, original = (
            local_residual(saved, family, kind, c, legs),
            local_residual(saved, family, kind, baseline, legs),
        )
        per_leg = {
            leg: {
                "norm": float(np.linalg.norm(actual[leg])),
                "baseline_norm": float(np.linalg.norm(original[leg])),
                "rms": float(
                    np.linalg.norm(actual[leg]) / math.sqrt(LEG_DENOMINATORS[kind])
                ),
                "baseline_rms": float(
                    np.linalg.norm(original[leg]) / math.sqrt(LEG_DENOMINATORS[kind])
                ),
            }
            for leg in legs
        }
        total = math.sqrt(sum(v["norm"] ** 2 for v in per_leg.values()))
        base = math.sqrt(sum(v["baseline_norm"] ** 2 for v in per_leg.values()))
        metrics[kind] = {
            "norm": total,
            "baseline_norm": base,
            "rms": total / math.sqrt(DENOMINATORS[kind]),
            "baseline_rms": base / math.sqrt(DENOMINATORS[kind]),
            "legs": per_leg,
        }
    for leg in legs:
        key = f"qr_command_{leg}"
        qr = saved[key]
        active = saved[f"{key}_active"]
        matrix, offset = qr[:, :-1], qr[:, -1]
        resolutions.append(
            float(
                freeze["contract"]["resolution_factor"]
                * EPS
                * max(matrix.shape)
                * (
                    np.linalg.norm(matrix) * np.linalg.norm(c[active])
                    + np.linalg.norm(offset)
                )
                + freeze["contract"]["resolution_absolute_pad"]
            )
        )
    return metrics, resolutions


def returned_witness(
    problem: dict[str, Any],
    x: np.ndarray,
    metrics: dict[str, Any],
    resolutions: list[float],
    family: str,
    freeze: dict[str, Any],
) -> dict[str, Any]:
    c = x[:-1]
    bound = float(freeze["contract"]["coefficient_bound"])
    finite = bool(np.isfinite(x).all())
    box = finite and bool(np.all(np.abs(c) <= bound))
    endpoint = finite and (
        family != "history"
        or bool(np.all(np.abs(c[: c.size // 2] + c[c.size // 2 :]) <= bound))
    )
    t_box = finite and bool(-1 <= x[-1] <= 1)
    command = metrics["command"]
    indicator = math.sqrt(sum(v * v for v in resolutions)) / math.sqrt(
        DENOMINATORS["command"]
    )
    baseline = command["baseline_rms"]
    target = freeze["contract"]["command_ratio"] * baseline
    resolution_ok = (
        indicator <= freeze["contract"]["resolution_baseline_fraction"] * baseline
    )
    direct_ok = command["rms"] <= target and command["rms"] + indicator <= target
    return {
        "finite": finite,
        "coefficient_box": box,
        "history_endpoint_box": endpoint,
        "margin_box": t_box,
        "direct_ratio": command["rms"] / baseline if baseline else None,
        "resolution_indicator_rms": indicator,
        "resolution_ok": resolution_ok,
        "direct_20_percent": direct_ok,
        "valid": bool(
            finite and box and endpoint and t_box and resolution_ok and direct_ok
        ),
    }


def settings_record(settings: Any) -> dict[str, Any]:
    def safe(value: Any, label: str) -> Any:
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            return str(value) if not math.isfinite(value) else value
        if isinstance(value, (list, tuple)):
            return [safe(item, f"{label}[{index}]") for index, item in enumerate(value)]
        if isinstance(value, dict):
            require(
                all(isinstance(key, str) for key in value), f"settings key: {label}"
            )
            return {key: safe(item, f"{label}.{key}") for key, item in value.items()}
        raise TypeError(f"unsupported solver setting {label}: {type(value).__name__}")

    result = {}
    for key in dir(settings):
        if key.startswith("_"):
            continue
        value = getattr(settings, key)
        if callable(value):
            continue
        result[key] = safe(value, key)
    return result


def bare_return(solution: Any) -> dict[str, np.ndarray]:
    return {
        "x": np.asarray(solution.x, dtype=np.float64).copy(),
        "s": np.asarray(solution.s, dtype=np.float64).copy(),
        "z": np.asarray(solution.z, dtype=np.float64).copy(),
        "status": np.asarray(str(solution.status)),
        "iterations": np.asarray(solution.iterations),
        "solve_time": np.asarray(solution.solve_time),
    }


def save_raw_first(
    out: Path, family: str, raw: dict[str, np.ndarray], report: dict[str, Any]
) -> Path:
    path = out / f"{family}-solver-raw.npz"
    try:
        np.savez_compressed(path, **raw)
        return path
    except Exception as exc:
        report["secondary_publication_errors"].append(
            {"path": str(path), "error": repr(exc)}
        )
        fallback = out / f"{family}-solver-raw.fallback.npz"
        try:
            np.savez_compressed(fallback, **raw)
            report["families"].setdefault(family, {})["raw_fallback"] = fallback.name
        except Exception as next_exc:  # noqa: BLE001 - retain secondary publication error
            report["secondary_publication_errors"].append(
                {"path": str(fallback), "error": repr(next_exc)}
            )
        raise


def resource_check(out: Path, started: float, limits: dict[str, Any]) -> None:
    require(
        time.monotonic() - started <= limits["internal_wall_s"], "internal wall limit"
    )
    require(
        sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        <= limits["max_output_bytes"],
        "raw output size limit",
    )


def baseline_comparison(
    selected: dict[str, dict[str, np.ndarray]], freeze: dict[str, Any]
) -> dict[str, Any]:
    result = {}
    for kind in KINDS:
        norms = []
        per_leg = {}
        for family in freeze["family_order"]:
            saved = selected[family]
            residuals = local_residual(
                saved,
                family,
                kind,
                baseline_vector(saved, family),
                freeze["sample_contract"]["legs"],
            )
            per_leg[family] = {
                leg: float(np.linalg.norm(residual))
                for leg, residual in residuals.items()
            }
            norms.append(math.sqrt(sum(v * v for v in per_leg[family].values())))
        tol = (
            freeze["contract"]["QR_absolute_atol"]
            + freeze["contract"]["QR_relative_atol"] * norms[0]
        )
        require(abs(norms[0] - norms[1]) <= tol, f"baseline global {kind}")
        for leg in freeze["sample_contract"]["legs"]:
            a, b = per_leg["static"][leg], per_leg["history"][leg]
            require(
                abs(a - b)
                <= freeze["contract"]["QR_absolute_atol"]
                + freeze["contract"]["QR_relative_atol"] * a,
                f"baseline leg {leg}/{kind}",
            )
        result[kind] = {
            "static_norm": norms[0],
            "history_norm": norms[1],
            "legs": per_leg,
        }
    return result


def run(
    freeze_path: Path,
    gate_path: Path,
    out: Path,
    root: Path = ROOT,
    solver_factory: Any = clarabel.DefaultSolver,
) -> dict[str, Any]:
    require(not out.exists(), "output already exists")
    started = time.monotonic()
    pre = preflight(freeze_path, gate_path, root)
    freeze, helper = pre["freeze"], pre["helper"]
    require(
        shutil.disk_usage(out.parent).free
        >= freeze["resource_limits"]["min_free_bytes"],
        "free disk",
    )
    out.mkdir(parents=False, exist_ok=False)
    report: dict[str, Any] = {
        "schema": "flylab.m3.e172.bounded-qr-capacity-result.v1",
        "status": "invalid_not_tested",
        "conclusion": "invalid_not_tested",
        "adoption": "not_applicable",
        "candidate_adopted": False,
        "whole_goal_complete": False,
        "F1_F8": "pending",
        "new_conic_fits": 0,
        "new_svd_fits": 0,
        "new_qr_factorizations": 0,
        "retries": 0,
        "startup_pointwise_max": "not_evaluated",
        "activation": "not_evaluated",
        "native_gate": "not_evaluated",
        "clipping": "not_evaluated",
        "threadpool_info": pre["threadpool_info"],
        "families": {},
        "primary_error": None,
        "secondary_publication_errors": [],
        "secondary_resource_errors": [],
        "provenance": {
            "freeze_sha256": sha256(freeze_path),
            "design_sha256": freeze["design_sha256"],
            "gate_sha256": sha256(gate_path),
            "source_sha256": {name: sha256(root / name) for name in (SOURCE, TESTS)},
            "reused_helper_sha256": freeze["reused_helper_sha256"],
            "selected_input_sha256": {
                freeze["families"][family]["raw"]: freeze["input_sha256"].get(
                    freeze["families"][family]["raw"]
                )
                for family in freeze["family_order"]
            },
        },
    }
    fit_total = 0.0
    try:
        for name in (SOURCE, TESTS):
            shutil.copyfile(root / name, out / ("source-" + Path(name).name))
        selected = {
            family: helper.read_selected(
                root / freeze["families"][family]["raw"],
                freeze["allowed_keys"],
                freeze["archive_key_map"],
            )
            for family in freeze["family_order"]
        }
        validation = helper.validate_saved(
            selected["static"], selected["history"], freeze
        )
        report["validation"] = validation
        report["baseline_comparison"] = baseline_comparison(selected, freeze)
        for family in freeze["family_order"]:
            resource_check(out, started, freeze["resource_limits"])
            saved = selected[family]
            problem = build_problem(saved, family, freeze)
            baseline = problem["baseline"]
            witness = inner_witness(problem, freeze)
            probes = {
                "baseline": baseline,
                "inner_witness": witness[:-1],
                "sine": np.sin(np.arange(baseline.size) * 0.173),
            }
            comparisons = {
                name: compare_local_global(saved, family, problem, c, freeze, name)
                for name, c in probes.items()
            }
            sparse.save_npz(out / f"{family}-M.npz", problem["M"])
            sparse.save_npz(out / f"{family}-A.npz", problem["A"])
            np.savez_compressed(
                out / f"{family}-problem.npz",
                b=problem["b"],
                true_b=problem["true_b"],
                q=problem["q"],
                offset=problem["offset"],
                baseline=baseline,
                inner_witness=witness,
                sine=probes["sine"],
                lower=problem["lower"],
                upper=problem["upper"],
                inner_lower=problem["inner_lower"],
                inner_upper=problem["inner_upper"],
            )
            report["families"][family] = {
                "status": "prepared",
                "baseline_norm": problem["baseline_norm"],
                "target_norm": problem["target"],
                "shape": {
                    "M": list(problem["M"].shape),
                    "A": list(problem["A"].shape),
                    "linear_rows": problem["linear_rows"],
                    "soc_rows": problem["soc_rows"],
                },
                "probe_comparisons": comparisons,
            }
            resource_check(out, started, freeze["resource_limits"])
            settings = clarabel.DefaultSettings()
            for key, value in freeze["solver_settings"].items():
                setattr(settings, key, value)
            require(report["new_conic_fits"] < 2, "max two conic fits")
            report["new_conic_fits"] += 1
            fit_start = time.monotonic()
            solver = solver_factory(
                problem["P"],
                problem["q"],
                problem["A"],
                problem["b"],
                problem["cones"],
                settings,
            )
            solution = solver.solve()
            fit_wall = time.monotonic() - fit_start
            fit_total += fit_wall
            raw = bare_return(solution)
            raw_path = save_raw_first(out, family, raw, report)
            report["families"][family]["raw_file"] = raw_path.name
            report["families"][family]["raw_sha256"] = sha256(raw_path)
            require(
                fit_wall <= freeze["resource_limits"]["family_fit_wall_s"]
                and fit_total <= freeze["resource_limits"]["total_fit_wall_s"],
                "fit wall limit",
            )
            require(
                raw["x"].shape == problem["q"].shape
                and raw["s"].shape == problem["b"].shape
                and raw["z"].shape == problem["b"].shape,
                "return shape",
            )
            require(
                all(
                    np.isfinite(raw[key]).all()
                    for key in ("x", "s", "z", "iterations", "solve_time")
                ),
                "return finite",
            )
            require(
                str(raw["status"]) in freeze["allowed_solver_statuses"], "solver status"
            )
            settings_actual = settings_record(solver.get_settings())
            for key, expected in freeze["solver_settings"].items():
                require(
                    settings_actual[key] == expected, f"actual solver setting {key}"
                )
            x = raw["x"]
            c = x[:-1]
            returned_comparison = compare_local_global(
                saved, family, problem, c, freeze, "returned"
            )
            metrics, resolutions = qr_metrics(saved, family, c, freeze)
            witness_result = returned_witness(
                problem, x, metrics, resolutions, family, freeze
            )
            projected = project_dual(
                raw["z"],
                problem["linear_rows"],
                freeze["contract"]["dual_projection_pad"],
            )
            duals = {
                name: dual_bound(
                    problem,
                    projected if name.startswith("projected") else raw["z"],
                    name.endswith("true"),
                    freeze["contract"]["dual_upper_bound_pad"],
                )
                for name in (
                    "raw_inner",
                    "raw_true",
                    "projected_inner",
                    "projected_true",
                )
            }
            np.savez_compressed(
                out / f"{family}-derived.npz",
                z_projected=projected,
                **{
                    f"stationarity_{key}": value["stationarity"]
                    for key, value in duals.items()
                },
                **{
                    f"box_terms_{key}": value["box_terms"]
                    for key, value in duals.items()
                },
                returned_coefficients=c,
            )
            upper = duals["projected_true"]["upper"]
            if witness_result["valid"] and upper < 0:
                conclusion = "invalid_numerical_contradiction"
            elif witness_result["valid"]:
                conclusion = "supported_returned_bounded_affine_witness"
            elif upper < 0:
                conclusion = "refuted_bounded_signed_saved_affine_family"
            else:
                conclusion = "inconclusive"
            report["families"][family].update(
                {
                    "status": "valid"
                    if conclusion != "invalid_numerical_contradiction"
                    else "invalid",
                    "conclusion": conclusion,
                    "fit_wall_s": fit_wall,
                    "solve_time_s": float(raw["solve_time"]),
                    "iterations": int(raw["iterations"]),
                    "solver_status": str(raw["status"]),
                    "actual_solver_settings": settings_actual,
                    "returned_comparison": returned_comparison,
                    "metrics": metrics,
                    "witness": witness_result,
                    "duals": {
                        name: {
                            k: v
                            for k, v in value.items()
                            if k not in ("stationarity", "box_terms")
                        }
                        for name, value in duals.items()
                    },
                    "physical": helper.coefficient_diagnostics(
                        c,
                        saved["pair_muscles"],
                        validation["variable_muscles"],
                        family,
                        freeze["contract"],
                        saved["pair_motors"],
                    ),
                }
            )
            resource_check(out, started, freeze["resource_limits"])
            require(
                conclusion != "invalid_numerical_contradiction",
                "numerical contradiction",
            )
        conclusions = [
            report["families"][name]["conclusion"] for name in freeze["family_order"]
        ]
        report["status"] = "valid_diagnostic"
        report["conclusion"] = (
            "supported_returned_bounded_affine_witness"
            if "supported_returned_bounded_affine_witness" in conclusions
            else "refuted_two_bounded_signed_saved_affine_families"
            if all(
                v == "refuted_bounded_signed_saved_affine_family" for v in conclusions
            )
            else "inconclusive"
        )
    except Exception as exc:  # noqa: BLE001 - preserve first numerical or return failure
        report["status"] = "invalid_not_tested"
        report["conclusion"] = "invalid_not_tested"
        report["primary_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    report["fit_wall_total_s"] = fit_total
    report["elapsed_s"] = time.monotonic() - started
    try:
        resource_check(out, started, freeze["resource_limits"])
    except Exception as exc:  # noqa: BLE001 - resource failure must not erase primary
        item = {"type": type(exc).__name__, "message": str(exc)}
        if report["primary_error"] is None:
            report["primary_error"] = item
        else:
            report["secondary_resource_errors"].append(item)
        report["status"] = report["conclusion"] = "invalid_not_tested"
    try:
        json_write(out / "result.json", report)
    except Exception as exc:  # noqa: BLE001 - preserve publication failure
        report["secondary_publication_errors"].append(
            {"path": str(out / "result.json"), "error": repr(exc)}
        )
        if report["primary_error"] is None:
            report["primary_error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["status"] = report["conclusion"] = "invalid_not_tested"
        try:
            json_write(out / "result.fallback.json", report)
        except Exception as next_exc:  # noqa: BLE001 - primary error remains authoritative
            report["secondary_publication_errors"].append(
                {"path": str(out / "result.fallback.json"), "error": repr(next_exc)}
            )
    # The result itself counts toward the frozen limit.
    try:
        resource_check(out, started, freeze["resource_limits"])
    except Exception as exc:  # noqa: BLE001 - final resource check after publication
        item = {"type": type(exc).__name__, "message": str(exc)}
        if report["primary_error"] is None:
            report["primary_error"] = item
        else:
            report["secondary_resource_errors"].append(item)
        report["status"] = report["conclusion"] = "invalid_not_tested"
        try:
            json_write(out / "result.json", report)
        except Exception as next_exc:  # noqa: BLE001 - distinct fallback after resource failure
            report["secondary_publication_errors"].append(
                {"path": str(out / "result.json"), "error": repr(next_exc)}
            )
            try:
                json_write(out / "result.fallback.json", report)
            except Exception as fallback_exc:  # noqa: BLE001 - retain original error
                report["secondary_publication_errors"].append(
                    {
                        "path": str(out / "result.fallback.json"),
                        "error": repr(fallback_exc),
                    }
                )
    try:
        manifest = {
            p.name: sha256(p)
            for p in out.iterdir()
            if p.is_file() and p.name != "output-manifest.json"
        }
        json_write(
            out / "output-manifest.json",
            {"files": manifest, "schema": "flylab.m3.e172.output-manifest.v1"},
        )
        resource_check(out, started, freeze["resource_limits"])
    except Exception as exc:  # noqa: BLE001 - preserve first failure after final save
        item = {"type": type(exc).__name__, "message": str(exc)}
        if report["primary_error"] is None:
            report["primary_error"] = item
        else:
            report["secondary_publication_errors"].append(item)
        report["status"] = report["conclusion"] = "invalid_not_tested"
        try:
            json_write(out / "result.json", report)
        except Exception as next_exc:  # noqa: BLE001 - distinct final fallback
            report["secondary_publication_errors"].append(
                {"path": str(out / "result.json"), "error": repr(next_exc)}
            )
            try:
                json_write(out / "result.fallback.json", report)
            except Exception as fallback_exc:  # noqa: BLE001 - retain first failure
                report["secondary_publication_errors"].append(
                    {
                        "path": str(out / "result.fallback.json"),
                        "error": repr(fallback_exc),
                    }
                )
        try:
            manifest = {
                p.name: sha256(p)
                for p in out.iterdir()
                if p.is_file() and p.name != "output-manifest.json"
            }
            json_write(
                out / "output-manifest.json",
                {"files": manifest, "schema": "flylab.m3.e172.output-manifest.v1"},
            )
        except Exception as manifest_exc:  # noqa: BLE001 - no refit on publication failure
            report["secondary_publication_errors"].append(
                {"path": str(out / "output-manifest.json"), "error": repr(manifest_exc)}
            )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--source-gate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.freeze, args.source_gate, args.out)
    print(
        json.dumps(
            {
                "status": result["status"],
                "conclusion": result["conclusion"],
                "new_conic_fits": result["new_conic_fits"],
            }
        )
    )
    return 0 if result["status"] == "valid_diagnostic" else 1


if __name__ == "__main__":
    sys.exit(main())
