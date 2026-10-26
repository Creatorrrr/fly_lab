"""E340 fixed CPU output-gradient diagnosis; no BANC parameter or body work."""

from __future__ import annotations

# The entry point takes its one execution clock before lazy numerical imports.
import argparse
import copy
import hashlib
import io
import json
import math
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "verification/banc-natural-gait-m3-20260919"
OUTPUT = BASE + "/e340-coordination-gradient"
FREEZE = BASE + "/e340-coordination-gradient-freeze.json"
REVIEW = BASE + "/parent-e340-original-source-review.json"
DEPS = BASE + "/parent-e340-reflected-direct-dependencies.json"
DESIGN = "docs/BANC_M3_E340_COORDINATION_GRADIENT_DESIGN_20261001.md"
REFLECTION = BASE + "/parent-e340-astra-reflection.json"
ASTRA = BASE + "/parent-e340-astra-design-review.md"
REGISTRATION = BASE + "/parent-e340-design-registration.json"
ASTRA_DISPATCH = BASE + "/parent-e340-actual-astra-dispatch.json"
SOL_DISPATCH = BASE + "/parent-e340-sol-dispatch.json"
EFFORT = BASE + "/parent-e340-actual-effort-note.json"
SNAPSHOT = BASE + "/parent-e340-pre-astra-design.md"
SUPERVISOR = BASE + "/parent-e267-supervise.py"
SUPERVISOR_SHA = "1d8682505f8060d3d8bcb50508811b9c4da531bdbf07c96200065bcc3324df8d"
OWNED = (
    "tools/analyze_c_banc_m3_coordination_gradient.py",
    "tests/test_c_banc_coordination_gradient.py",
)
SCHEMA = "flylab.m3.e340.coordination-gradient.freeze.v1"
PROOF_SCHEMA = "flylab.m3.e340.current-generated-proof.v1"
RESULT_SCHEMA = "flylab.m3.e340.coordination-gradient.result.v1"
LIMITS = {
    "science_wall_s": 12,
    "internal_wall_s": 16,
    "finalization_s": 4,
    "supervisor_wall_s": 20,
    "host_bytes": 2 << 30,
    "output_bytes": 4 << 20,
    "cpu_threads": 1,
    "retry_count": 0,
    "gpu_jobs": 0,
}
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
ROLES = ("neutral", "held", "live")
LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
PAIRS = tuple((a, b) for a in range(6) for b in range(a + 1, 6))
REF_SPEC = {
    "curves": (6, 128, 7),
    "normalization": (6,),
    "temperature": (6,),
    "future_anchors": (6, 128, 7),
    "phase_basis": (128, 2),
}
TRAJ_SPEC = {
    "q": (201, 42),
    "z": (201, 6, 2),
    "R": (201, 6),
    "coordination_pair_loss": (15,),
}
ARRAY_SPEC = {
    **{
        k: (50, 4, 42)
        for k in (
            "g_successor",
            "g_coordination",
            "g_total",
            "analytic_g_successor",
            "analytic_g_coordination",
            "analytic_g_total",
            "successor_next_path",
            "successor_previous_internal_path",
        )
    },
    "all201_z": (201, 6, 2),
    "all201_R": (201, 6),
    "pair_losses": (200, 15),
    "successor_terms": (200, 6),
}
G_RTOL, G_ATOL = 1e-8, 2e-10
FIXED = {
    DESIGN: "b16efd80c17fb1f60fcd875d921c1f03e9def6e7cd0c74ae67063734e07d8f3b",
    DEPS: "f5e5a14480aa67db3c07f369afb094b33641d72c01ba1b1e684652e9f13ffe67",
    ASTRA: "7f5d03d18c91aedbf54cd923b3a7dcf7f3060d4e71c8d9bd89cbb57dbe51859a",
    REFLECTION: "0724c23519ddeb7e98588f1a3aeaaf2c55dda205e598cca4549cee3992fb3808",
    ASTRA_DISPATCH: "27ba879f6f2f5891f8d4c86005e70cd5a0128c134258ca0f5851b1976e80beda",
    REGISTRATION: "e8ff36a0fcc253940000c020019c861449199c14dcf5174718cdcbdb74b17155",
    SNAPSHOT: "34cf850b76b3e0805db2bc1737e5c465f4d8f2383b657a27d23f4b57c21a3c89",
    EFFORT: "a854f5f82a30e77f3fcc039937510825c6f74c9e7350baf313e51ebca1bff49a",
    SOL_DISPATCH: "4bb088c74d35302af436e78e9a4ad29ac729a3f7e23943ca57c04433818d2214",
}


def environment():
    for key in THREAD_ENV:
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def runtime():
    """Lazy CPU-only imports. No CUDA availability/memory/context queries."""
    global np, torch, reader, SmoothTemporalObjective
    if "np" not in globals():
        environment()
        import numpy
        import torch as cpu_torch

        from tools import analyze_c_banc_m3_saved_coordination as saved
        from tools.banc_m3_smooth_temporal import SmoothTemporalObjective as objective

        np, torch, reader, SmoothTemporalObjective = numpy, cpu_torch, saved, objective
        torch.set_num_threads(1)


def safe(name, root=ROOT):
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or Path(name).is_absolute()
        or any(x in ("", ".", "..") for x in name.split("/"))
    ):
        raise ValueError("canonical relative path required")
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("path escapes root")
    return path


def digest(path, timely=lambda: None):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            timely()
            block = stream.read(1 << 20)
            if not block:
                break
            h.update(block)
    timely()
    return h.hexdigest()


def pin(path, timely=lambda: None):
    size = Path(path).stat().st_size
    value = {"sha256": digest(path, timely), "bytes": size}
    if Path(path).stat().st_size != size:
        raise ValueError("bytes changed during hash")
    return value


def json_bytes(value):
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        + "\n"
    ).encode("utf8")


def read_json(path):
    def reject(value):
        raise ValueError("nonfinite JSON: " + value)

    return json.loads(Path(path).read_text("utf8"), parse_constant=reject)


def typed(a, shape, label):
    runtime()
    if (
        not isinstance(a, np.ndarray)
        or a.dtype.str != "<f8"
        or a.shape != tuple(shape)
        or not np.isfinite(a).all()
    ):
        raise ValueError("finite <f8 shape required: " + label)
    return a


def close(a, b, label, *, rtol=1e-10, atol=1e-10):
    if (
        not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or not np.allclose(a, b, rtol=rtol, atol=atol)
    ):
        raise ValueError("numeric reconciliation differs: " + label)


def split_windows(q):
    typed(q, (201, 42), "q c100..300")
    boundary = q[np.arange(0, 200, 4)].copy(order="C")
    boundary.setflags(write=False)
    live = q[1:].reshape(50, 4, 42).copy(order="C")
    return boundary, live


def _previous(boundary, live):
    typed(boundary, (50, 42), "boundary")
    typed(live, (50, 4, 42), "live outputs")
    return np.concatenate((boundary[:, None], live[:, :3]), axis=1).reshape(200, 42)


def _encode(reference, q, *, derivative=False):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        delta = q.reshape(-1, 6, 1, 7) - reference["curves"][None]
        logits = (
            -np.sum(delta * delta, axis=-1) / reference["temperature"][None, :, None]
        )
        w = np.exp(logits - logits.max(axis=-1, keepdims=True))
        w /= w.sum(axis=-1, keepdims=True)
        target = np.einsum("tls,lsk->tlk", w, reference["future_anchors"])
        z = np.einsum("tls,sa->tla", w, reference["phase_basis"])
        out = {"target": target, "z": z, "R": np.linalg.norm(z, axis=-1)}
        if derivative:
            centered = delta - np.einsum("tls,tlsj->tlj", w, delta)[:, :, None]
            dw = (
                -2
                * w[..., None]
                * centered
                / reference["temperature"][None, :, None, None]
            )
            out["J_target"] = np.einsum(
                "tlsj,lsk->tlkj", dw, reference["future_anchors"]
            )
            out["J_z"] = np.einsum("tlsj,sa->tlaj", dw, reference["phase_basis"])
        if any(not np.isfinite(a).all() for a in out.values()):
            raise FloatingPointError("nonfinite encoder intermediate")
    return out


def _values(reference, previous, live):
    p, n = _encode(reference, previous), _encode(reference, live.reshape(200, 42))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        residual = live.reshape(200, 6, 7) - p["target"]
        successor_terms = (
            np.mean(residual * residual, axis=-1) / reference["normalization"][None]
        )
        pair = np.column_stack(
            [1 - np.sum(n["z"][:, a] * n["z"][:, b], axis=-1) for a, b in PAIRS]
        )
        losses = {
            "successor_loss": float(successor_terms.mean()),
            "coordination_loss": float(pair.mean()),
        }
        losses["training_loss"] = losses["successor_loss"] + losses["coordination_loss"]
    if not all(np.isfinite(v) for v in losses.values()):
        raise FloatingPointError("nonfinite conditional loss")
    return losses, residual, successor_terms, pair


def conditional_value(reference, boundary, live):
    """F_b(x): copied b stays fixed; each L4/50 is equivalent to mean200."""
    return _values(reference, _previous(boundary, live), live)[0]


def numpy_gradients(reference, boundary, live):
    previous = _previous(boundary, live)
    p = _encode(reference, previous, derivative=True)
    n = _encode(reference, live.reshape(200, 42), derivative=True)
    losses, residual, terms, pairs = _values(reference, previous, live)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        next_path = (
            2 * residual / (200 * 42 * reference["normalization"][None, :, None])
        )
        raw_previous = -np.einsum("tlkj,tlk->tlj", p["J_target"], next_path).reshape(
            50, 4, 42
        )
        internal = raw_previous.copy()
        internal[:, 0] = 0  # Pair0 refers to a detached boundary, not a live variable.
        gs = next_path.reshape(50, 4, 42).copy()
        gs[:, :3] += internal[:, 1:]
        dz = -(n["z"].sum(axis=1, keepdims=True) - n["z"]) / (200 * 15)
        gc = np.einsum("tlaj,tla->tlj", n["J_z"], dz).reshape(50, 4, 42)
        gt = gs + gc
    for name, array in (("S", gs), ("C", gc), ("total", gt)):
        typed(array, (50, 4, 42), "analytic " + name)
    return {
        "g_successor": gs,
        "g_coordination": gc,
        "g_total": gt,
        "successor_next_path": next_path.reshape(50, 4, 42),
        "successor_previous_internal_path": internal,
        "losses": losses,
        "successor_terms": terms,
        "pair_losses": pairs,
    }


def make_objective(reference):
    runtime()
    reader.Encoder(reference)  # Fixed canonical validation; no old CLI/worker.
    objective = SmoothTemporalObjective(
        reference["curves"],
        np.zeros((99, 42), dtype="<f8"),
        np.zeros((99, 6), dtype="<f8"),
        device="cpu",
        normalization=reference["normalization"],
    )
    for name in REF_SPEC:
        close(
            objective.canonical_constants[name],
            reference[name],
            "CPU actual " + name,
            rtol=1e-12,
            atol=1e-15,
        )
    return objective


def cpu_gradients(reference, boundary, live, *, objective=None):
    runtime()
    _previous(boundary, live)
    objective = make_objective(reference) if objective is None else objective
    b = torch.tensor(np.array(boundary, copy=True), dtype=torch.float64, device="cpu")
    x = torch.tensor(live, dtype=torch.float64, device="cpu", requires_grad=True)
    previous = torch.cat((b[:, None], x[:, :3]), dim=1).reshape(200, 42)
    terms = objective.evaluate_pairs(previous, x.reshape(200, 42))
    output = {}
    for index, (name, term) in enumerate(
        (
            ("g_successor", "successor_loss"),
            ("g_coordination", "coordination_loss"),
            ("g_total", "training_loss"),
        )
    ):
        g = torch.autograd.grad(terms[term], x, retain_graph=index < 2)[0]
        output[name] = typed(g.detach().numpy().copy(), (50, 4, 42), name)
    output["losses"] = {
        name: float(terms[name].detach())
        for name in ("training_loss", "successor_loss", "coordination_loss")
    }
    output["successor_terms"] = terms["successor_terms"].detach().numpy().copy()
    output["pair_losses"] = terms["coordination_terms"].detach().numpy().copy()
    return output


def fixed_fd_directions():
    runtime()
    index = np.arange(8400, dtype=np.float64) + 1
    return tuple(
        (a / np.linalg.norm(a)).reshape(50, 4, 42)
        for a in (np.sin(0.37 * index), np.cos(0.53 * index))
    )


def finite_difference_checks(reference, boundary, live, gradients):
    """Generated qualification only; central FD never re-extracts boundary."""
    records = []
    saved = boundary.copy()
    for di, direction in enumerate(fixed_fd_directions(), 1):
        for step in (1e-6, 5e-7):
            plus = conditional_value(reference, boundary, live + step * direction)
            minus = conditional_value(reference, boundary, live - step * direction)
            for g, term in (
                ("g_successor", "successor_loss"),
                ("g_coordination", "coordination_loss"),
                ("g_total", "training_loss"),
            ):
                fd = (plus[term] - minus[term]) / (2 * step)
                dot = float(np.sum(gradients[g] * direction))
                close(fd, dot, "conditional FD", rtol=1e-5, atol=1e-7)
                records.append(
                    {
                        "direction": di,
                        "step": step,
                        "term": term,
                        "finite_difference": fd,
                        "gradient_dot_direction": dot,
                    }
                )
    if not np.array_equal(boundary, saved):
        raise ValueError("FD changed immutable boundary")
    return records


def gradient_statistics(gs, gc):
    runtime()
    if gs.shape != gc.shape or gs.dtype.str != "<f8" or gc.dtype.str != "<f8":
        raise ValueError("same float64 gradient axes required")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        gt = gs + gc
        norms = {k: stable_norm(a) for k, a in (("S", gs), ("C", gc), ("total", gt))}
        dots = {"C_S": float(np.sum(gc * gs)), "C_total": float(np.sum(gc * gt))}
        eta = {
            k: float(np.sqrt(gc.size) * G_ATOL + G_RTOL * v) for k, v in norms.items()
        }
    if not all(
        np.isfinite(v) for v in (*norms.values(), *dots.values(), *eta.values())
    ):
        raise FloatingPointError("nonfinite gradient statistic")
    reason = (
        "exact_zero"
        if norms["C"] == 0 or norms["total"] == 0
        else "below_declared_precision"
        if norms["C"] <= 10 * eta["C"] or norms["total"] <= 10 * eta["total"]
        else None
    )
    row = {
        "N": gc.size,
        "norms": norms,
        "eta": eta,
        "dots": dots,
        "D_C": -dots["C_total"],
        "information_reason": reason,
        "cosine_C_total": None,
        "cosine_C_S": None,
        "norm_ratio_S_over_C": None,
        "relative_coordination_progress": None,
        "cosine_C_S_undefined_reason": "S_exact_zero" if norms["S"] == 0 else reason,
    }
    if reason is None:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            row["cosine_C_total"] = float(
                np.float64(dots["C_total"]) / (np.float64(norms["C"]) * norms["total"])
            )
            row["norm_ratio_S_over_C"] = float(np.float64(norms["S"]) / norms["C"])
            row["relative_coordination_progress"] = float(
                np.float64(dots["C_total"]) / (np.float64(norms["C"]) * norms["C"])
            )
            if norms["S"] != 0:
                row["cosine_C_S"] = float(
                    np.float64(dots["C_S"]) / (np.float64(norms["C"]) * norms["S"])
                )
        if any(
            v is not None and not np.isfinite(v)
            for v in (
                row[k]
                for k in (
                    "cosine_C_total",
                    "cosine_C_S",
                    "norm_ratio_S_over_C",
                    "relative_coordination_progress",
                )
            )
        ):
            raise FloatingPointError("nonfinite gradient ratio/cosine")
    return row


def stable_norm(array):
    """L2 with a scale factor; do not turn tiny nonzero data into exact zero."""
    scale = float(np.max(np.abs(array)))
    if scale == 0:
        return 0.0
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        value = float(np.float64(scale) * np.linalg.norm(array.ravel() / scale))
    if not np.isfinite(value):
        raise FloatingPointError("nonfinite gradient norm")
    return value


def classify_gradients(gs, gc):
    typed(gs, (50, 4, 42), "classify S")
    typed(gc, (50, 4, 42), "classify C")
    row = gradient_statistics(gs, gc)
    reason = row["information_reason"]
    branch = (
        "output_gradient_inconclusive"
        if reason is not None
        else "combined_output_descent_opposes_coordination"
        if row["cosine_C_total"] < -1e-10
        else "successor_output_gradient_dominates"
        if row["norm_ratio_S_over_C"] >= 10
        else "no_fixed_output_gradient_obstacle"
    )
    return {"classification": branch, "reason": reason, "aggregate": row}


def compare_branches(primary, analytic):
    if primary["classification"] != analytic["classification"]:
        return {
            "classification": "output_gradient_inconclusive",
            "reason": "branch_unstable",
        }
    return {"classification": primary["classification"], "reason": primary["reason"]}


def analyze_trajectory(reference, trajectory, *, objective=None, timely=lambda: None):
    for name, shape in TRAJ_SPEC.items():
        typed(trajectory[name], shape, name)
    b, x = split_windows(trajectory["q"])
    obj = make_objective(reference) if objective is None else objective
    primary = cpu_gradients(reference, b, x, objective=obj)
    timely()
    analytic = numpy_gradients(reference, b, x)
    with torch.no_grad():
        encoded = obj.encode(
            torch.tensor(trajectory["q"], dtype=torch.float64, device="cpu")
        )
    z, R = encoded["z"].numpy().copy(), encoded["R"].numpy().copy()
    arrays = {k: primary[k] for k in ("g_successor", "g_coordination", "g_total")}
    arrays.update(
        {
            "analytic_" + k: analytic[k]
            for k in ("g_successor", "g_coordination", "g_total")
        }
    )
    arrays.update(
        {
            k: analytic[k]
            for k in ("successor_next_path", "successor_previous_internal_path")
        }
    )
    arrays.update(
        all201_z=z,
        all201_R=R,
        pair_losses=primary["pair_losses"],
        successor_terms=primary["successor_terms"],
    )
    for name, shape in ARRAY_SPEC.items():
        typed(arrays[name], shape, name)
    try:
        for name in ("g_successor", "g_coordination", "g_total"):
            close(primary[name], analytic[name], name, rtol=G_RTOL, atol=G_ATOL)
        for name in primary["losses"]:
            close(primary["losses"][name], analytic["losses"][name], name)
        for name in ("pair_losses", "successor_terms"):
            close(primary[name], analytic[name], name)
        close(z, trajectory["z"], "all201 saved z")
        close(R, trajectory["R"], "all201 saved R")
        close(
            primary["pair_losses"].mean(axis=0),
            trajectory["coordination_pair_loss"],
            "saved15pair losses",
        )
        p = classify_gradients(primary["g_successor"], primary["g_coordination"])
        a = classify_gradients(analytic["g_successor"], analytic["g_coordination"])
        report = {
            "losses": primary["losses"],
            "primary": p,
            "analytic": a,
            "decision": compare_branches(p, a),
            "per_window": [
                gradient_statistics(
                    primary["g_successor"][i], primary["g_coordination"][i]
                )
                for i in range(50)
            ],
            "per_leg": {
                leg: gradient_statistics(
                    primary["g_successor"].reshape(50, 4, 6, 7)[:, :, i],
                    primary["g_coordination"].reshape(50, 4, 6, 7)[:, :, i],
                )
                for i, leg in enumerate(LEGS)
            },
            "gradient_reconciled": True,
            "window_count": 50,
            "live_controls": [101, 300],
            "boundary_pair_indices": list(range(0, 200, 4)),
            "coefficients": {"S": 1, "C": 1},
            "previous_internal_path_mapping": "pair0 detached; pair1..3 accumulate into live0..2",
            "placeholder_startup_unused": True,
            "startup_guard_called": False,
            "scope": "Conditional saved output space; not parameter VJP, neural improvement or body proof",
        }
        timely()
    except Exception as error:
        failure = NumericalFailure(str(error))
        failure.partial_arrays = arrays
        raise failure from error
    return arrays, report


class NumericalFailure(ValueError):
    """Attach finite acquired arrays so later scalar failures retain raw work."""


IDENTITIES = {
    "reference": (
        BASE + "/e332-smooth-temporal-saved/reference.npz",
        "140e23c865d7e110cd93699b41612c1ffb02eb88dc100ac435c4dcba5acf6c01",
        124353,
    ),
    "neutral": (
        BASE + "/e336-saved-gait-transfer/neutral-smooth.npz",
        "8b75c4b1d2cd7d4efc0fe2db0ad49ad50b7dbf2d9ca25b6f38359c9488603d4c",
        1644664,
    ),
    "held": (
        BASE + "/e336-saved-gait-transfer/held-smooth.npz",
        "0ea9afacfb31fcb7595b1c37d235dfbfb39979f366c1a0866f0ec65b3cab1ee6",
        1644664,
    ),
    "live": (
        BASE + "/e336-saved-gait-transfer/live-smooth.npz",
        "4d2ef1a8a22243b905adea9420ef7230e93e51d19d01e73615c37e84b9953ef4",
        1644664,
    ),
}
AUDIT_CASES = (
    "valid",
    "wrong_detach",
    "wrong_scale",
    "wrong_row",
    "wrong_pair",
    "wrong_gradient",
    "wrong_result",
)
AUDIT_SOURCE = BASE + "/parent-e340-gradient-audit.py"
EXECUTED_LOCAL = (
    "tools/banc_m3_smooth_temporal.py",
    "tools/analyze_c_banc_m3_temporal_objective.py",
    "tools/banc_m3_phase_free_temporal.py",
    "tools/analyze_c_banc_m3_saved_coordination.py",
    "tools/qualify_c_banc_m3_body_packet.py",
)
TEXT_ONLY = (
    "tools/verify_c_banc_m3_temporal_direction.py",
    "tools/verify_c_banc_m3_smooth_efficacy_direction.py",
)


def original_bindings():
    return {
        role: {
            "path": name,
            "sha256": sha,
            "bytes": size,
            "selected_schema": {
                k: {"shape": list(s), "dtype": "<f8", "finite": True}
                for k, s in (REF_SPEC if role == "reference" else TRAJ_SPEC).items()
            },
            "header_observation_scope": "observation",
            "descriptor": None,
            "original_archive_current_header_hash_or_values_observed_at_reflection": False,
        }
        for role, (name, sha, size) in IDENTITIES.items()
    }


def generated_root(root):
    root = Path(root).resolve()
    base = (ROOT / BASE).resolve()
    if root == base or not root.is_relative_to(base):
        raise ValueError("isolated generated descendant required")
    first = root.relative_to(base).parts[0]
    if not first.startswith(("e340-sol-", "parent-e340-")):
        raise ValueError("fresh E340 generated namespace required")
    return root


def generated_bindings(root, bindings):
    root = generated_root(root)
    if set(bindings) != {"reference", *ROLES}:
        raise ValueError("exact four generated bindings required")
    originals = {v[1] for v in IDENTITIES.values()}
    for role, value in bindings.items():
        if value.get("sha256") in originals:
            raise ValueError("original SHA forbidden before generated reader")
        safe(value["path"], root)
        expected = original_bindings()[role]
        if any(
            value.get(k) != expected[k]
            for k in ("selected_schema", "header_observation_scope", "descriptor")
        ):
            raise ValueError("fixed generated role schema differs")
    return root


def descriptor(value):
    return (
        isinstance(value, dict)
        and set(value) == {"sha256", "bytes"}
        and isinstance(value["sha256"], str)
        and re.fullmatch("[0-9a-f]{64}", value["sha256"]) is not None
        and type(value["bytes"]) is int
        and value["bytes"] >= 0
    )


def merge_pins(target, incoming):
    for name, value in incoming.items():
        safe(name)
        if not descriptor(value):
            raise ValueError("typed byte pin required: " + name)
        if name in target and target[name] != value:
            raise ValueError("pin collision: " + name)
        target[name] = copy.deepcopy(value)


def source_map(root=ROOT):
    return {name: digest(safe(name, root)) for name in OWNED}


def argv_path(value, root):
    if not isinstance(value, str) or not value:
        raise ValueError("argv path string required")
    path = Path(value.replace("\\", "/"))
    path = (path if path.is_absolute() else Path(root) / path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("argv path escapes root")
    return path.relative_to(Path(root).resolve()).as_posix()


def option(argv, key):
    if argv.count(key) != 1 or argv.index(key) + 1 >= len(argv):
        raise ValueError("sole explicit argv option required: " + key)
    return argv[argv.index(key) + 1]


def python_helper(argv, root):
    if (
        not isinstance(argv, list)
        or len(argv) < 5
        or argv[1:4] != ["-X", "utf8", "-B"]
        or argv_path(argv[0], root) != ".venv/Scripts/python.exe"
    ):
        raise ValueError("explicit Python UTF8 no-bytecode helper command required")
    return argv_path(argv[4], root)


def positive_finite_time(value):
    # Bounded before isfinite so an arbitrarily large JSON integer is rejected
    # without an integer-to-float overflow; bool is never a measured duration.
    return type(value) in (int, float) and 0 < value <= 60 and math.isfinite(value)


def positive_unique_pids(value):
    return (
        isinstance(value, list)
        and bool(value)
        and all(type(pid) is int and pid > 0 for pid in value)
        and len(set(value)) == len(value)
    )


def closed_execution(root, branch, evidence, current, *, expected_exit=0):
    """Bind actual completed argv/helper/report/logs/PIDs to its exact contract."""
    needed = {branch[k] for k in ("contract", "supervision", "report", "pids")}
    if not needed.issubset(evidence):
        raise ValueError("execution evidence omitted")
    contract = read_json(safe(branch["contract"], root))
    supervision = read_json(safe(branch["supervision"], root))
    report = read_json(safe(branch["report"], root))
    pids = read_json(safe(branch["pids"], root))
    commands, attempts = contract.get("commands"), supervision.get("attempts")
    if (
        not isinstance(commands, list)
        or len(commands) != 1
        or not isinstance(attempts, list)
        or len(attempts) != 1
        or attempts[0].get("argv") != commands[0].get("argv")
        or attempts[0].get("exit_code") != expected_exit
        or attempts[0].get("killed") is not False
        or supervision.get("success") is not (expected_exit == 0)
        or supervision.get("contract_sha256") != evidence[branch["contract"]]["sha256"]
        or contract.get("supervisor_output") != branch["supervision"]
        or contract.get("original_calls") != 0
        or supervision.get("original_calls") != 0
        or type(contract.get("max_tree_rss_bytes")) is not int
        or not 0 < contract["max_tree_rss_bytes"] <= 2 << 30
        or not positive_finite_time(commands[0].get("wall_s"))
        or not positive_finite_time(supervision.get("elapsed_s"))
        or not positive_finite_time(attempts[0].get("wall_s"))
        or not attempts[0]["wall_s"]
        <= supervision["elapsed_s"]
        <= commands[0]["wall_s"]
        or type(supervision.get("max_tree_rss_bytes")) is not int
        or not 0 < supervision["max_tree_rss_bytes"] <= contract["max_tree_rss_bytes"]
    ):
        raise ValueError("completed sole supervised command differs")
    needed.update(commands[0][k] for k in ("stdout", "stderr"))
    sources = contract.get("source_sha256", {})
    if (
        sources.get(SUPERVISOR) != SUPERVISOR_SHA
        or any(sources.get(k) != v for k, v in current.items())
        or report.get("source_sha256") != current
        or report.get("original_calls") != 0
        or report.get("passed") is not (expected_exit == 0)
    ):
        raise ValueError("current source/zero-original report differs")
    helper = python_helper(commands[0]["argv"], root)
    if (
        report.get("audit_source_path") != helper
        or sources.get(helper) != report.get("audit_source_sha256")
        or helper not in sources
        or argv_path(option(commands[0]["argv"], "--output"), root) != branch["report"]
    ):
        raise ValueError("executed helper/report output is unbound")
    if expected_exit == 0:
        checks = report.get("checks")
        if (
            report.get("first_error") is not None
            or not isinstance(checks, dict)
            or not checks
            or any(v is not True for v in checks.values())
        ):
            raise ValueError("report has failed/omitted current checks")
    elif report.get("first_error") is None:
        raise ValueError("expected CLI rejection must preserve first error")
    for name, sha in sources.items():
        if name not in evidence or evidence[name]["sha256"] != sha:
            raise ValueError("execution source pin omitted/differs: " + name)
    needed.update(sources)
    if (
        pids.get("all_owned_pids_ended") is not True
        or not positive_unique_pids(supervision.get("owned_pids"))
        or not positive_unique_pids(pids.get("owned_pids"))
        or not positive_unique_pids(attempts[0].get("pids"))
        or pids.get("owned_pids") != supervision.get("owned_pids")
        or attempts[0]["pids"] != supervision["owned_pids"]
        or not isinstance(pids.get("present_after_return"), dict)
        or set(pids["present_after_return"])
        != {str(pid) for pid in supervision["owned_pids"]}
        or any(value is not False for value in pids["present_after_return"].values())
    ):
        raise ValueError("owned completion evidence differs")
    return needed, contract, report


def verify_current_proof(root, proof, current, *, timely=lambda: None):
    """Finite proof: two current branches and same actual audit CLI seven cases."""
    if (
        proof.get("schema") != PROOF_SCHEMA
        or proof.get("passed") is not True
        or proof.get("original_calls") != 0
        or proof.get("source_sha256") != current
        or set(proof.get("branches", {})) != {"sol", "root"}
    ):
        raise ValueError("current generated proof header differs")
    evidence = proof.get("evidence", {})
    if not isinstance(evidence, dict) or not 1 <= len(evidence) <= 256:
        raise ValueError("bounded finite proof evidence required")
    originals = {v[1] for v in IDENTITIES.values()}
    for name, value in evidence.items():
        if (
            not descriptor(value)
            or value["bytes"] > 1 << 20
            or value["sha256"] in originals
        ):
            raise ValueError("generated proof unsafe/oversized pin")
        path = safe(name, root)
        if path.suffix.lower() == ".npz":
            generated_root(path.parent)
        if name in {v[0] for v in IDENTITIES.values()}:
            raise ValueError("original forbidden in generated proof")
        if pin(path, timely) != value:
            raise ValueError("generated evidence bytes differ: " + name)
    needed = set()
    for branch in proof["branches"].values():
        bound, _, _ = closed_execution(root, branch, evidence, current)
        needed.update(bound)
    same = proof.get("same_audit_cli", {})
    if set(same) != {"source", "qualification"} or same["source"] != AUDIT_SOURCE:
        raise ValueError("same actual audit executable descriptor required")
    if same["qualification"] not in evidence or same["source"] not in evidence:
        raise ValueError("same audit CLI pins missing")
    qualification = read_json(safe(same["qualification"], root))
    if (
        qualification.get("schema") != "flylab.m3.e340.audit-cli-qualification.v1"
        or qualification.get("passed") is not True
        or qualification.get("original_calls") != 0
        or qualification.get("source_sha256") != current
        or qualification.get("audit_source_sha256") != evidence[AUDIT_SOURCE]["sha256"]
        or set(qualification.get("cases", {})) != set(AUDIT_CASES)
    ):
        raise ValueError("current same CLI seven-case qualification differs")
    needed.update((same["source"], same["qualification"]))
    for case, branch in qualification["cases"].items():
        bound, contract, _ = closed_execution(
            root, branch, evidence, current, expected_exit=0 if case == "valid" else 1
        )
        argv = contract["commands"][0]["argv"]
        if (
            python_helper(argv, root) != AUDIT_SOURCE
            or option(argv, "--generated-case") != case
        ):
            raise ValueError("qualified audit helper/generated case differs")
        needed.update(bound)
    if set(evidence) != needed:
        raise ValueError("finite proof contains unrelated or omitted edges")
    return {
        "schema": proof["schema"],
        "source_sha256": current,
        "branches_closed": ["sol", "root"],
        "same_audit_cli": same,
        "audit_cli_sha256": evidence[AUDIT_SOURCE]["sha256"],
        "cases_closed": list(AUDIT_CASES),
    }


def verify_design_states(root, pins):
    reflection = read_json(safe(REFLECTION, root))
    dispatch = read_json(safe(SOL_DISPATCH, root))
    effort = read_json(safe(EFFORT, root))
    astra = read_json(safe(ASTRA_DISPATCH, root))
    registration = read_json(safe(REGISTRATION, root))
    if (
        reflection.get("all_required_items_reflected") is not True
        or reflection.get("required_items") != ["M" + str(i) for i in range(1, 7)]
        or reflection.get("reflected_design_sha256") != pins[DESIGN]["sha256"]
        or reflection.get("review_sha256") != pins[ASTRA]["sha256"]
        or reflection.get("actual_dispatch_sha256") != pins[ASTRA_DISPATCH]["sha256"]
        or reflection.get("pre_review_snapshot_sha256") != pins[SNAPSHOT]["sha256"]
        or reflection.get("reflected_direct_dependencies_sha256")
        != pins[DEPS]["sha256"]
        or registration.get("pre_review_snapshot_pin") != pins[SNAPSHOT]
        or astra.get("accepted_by_spawn_agent") is not True
        or dispatch.get("accepted_by_followup_task") is not True
        or dispatch.get("actual_model") != "gpt-6.1-sol"
        or dispatch.get("actual_effort") != "xhigh"
        or effort.get("actual_effort") != "xhigh"
        or dispatch.get("reflection_sha256") != pins[REFLECTION]["sha256"]
        or dispatch.get("actual_effort_note_sha256") != pins[EFFORT]["sha256"]
        or dispatch.get("design_sha256") != pins[DESIGN]["sha256"]
    ):
        raise ValueError("actual design/reflection/dispatch edge differs")


def verify_predecessor_states(root, pins, *, timely=lambda: None):
    """Closed E339 directly named states/edges; no recursive NPZ/state rehash."""
    prefix = BASE + "/parent-e339-"
    final = read_json(safe(prefix + "final-decision.json", root))
    audit = read_json(safe(prefix + "saved-raw-audit.json", root))
    review = read_json(safe(prefix + "original-source-review.json", root))
    generated = read_json(safe(prefix + "current-generated-qualification.json", root))
    frozen = read_json(safe(BASE + "/e339-saved-coordination-freeze.json", root))
    result = read_json(safe(BASE + "/e339-saved-coordination/result.json", root))
    manifest = read_json(
        safe(BASE + "/e339-saved-coordination/output-manifest.json", root)
    )
    if (
        final.get("schema") != "flylab.m3.e339.root-final-decision.v1"
        or final.get("analysis_valid") is not True
        or final.get("root_independent_raw_qualification") is not True
        or final.get("science_adopted") is not True
        or final.get("candidate_adopted") is not False
        or final.get("F1_F8_complete") is not False
        or final.get("original_cumulative_s") != 9567.165820199525
        or final.get("original_limit_s") != 9660
        or final.get("raw_checks") != 5591
        or audit.get("passed") is not True
        or len(audit.get("checks", {})) != 5591
        or any(v is not True for v in audit["checks"].values())
        or review.get("passed") is not True
        or review.get("ROOT_SECOND_completed") is not True
        or review.get("remaining_ROOT_source_fix_required") is not False
        or review.get("freeze_sha256")
        != pins[BASE + "/e339-saved-coordination-freeze.json"]["sha256"]
        or review.get("generated_qualification_sha256")
        != pins[prefix + "current-generated-qualification.json"]["sha256"]
        or generated.get("passed") is not True
        or generated.get("source_sha256") != review.get("source_sha256")
        or any(
            frozen.get("source_sha256", {}).get(k) != v
            for k, v in review.get("source_sha256", {}).items()
        )
        or result.get("status") != "valid_provisional_diagnostic"
        or result.get("analysis_valid") is not True
        or result.get("publication_valid") is not True
        or result.get("first_error") is not None
        or result.get("secondary_errors") != []
        or result.get("freeze_sha256") != review["freeze_sha256"]
        or result.get("review_sha256")
        != pins[prefix + "original-source-review.json"]["sha256"]
        or frozen.get("original_bindings") != original_bindings()
    ):
        raise ValueError("E339 terminal/current producer state differs")
    # All selected direct edges of final/result/manifest are checked as metadata;
    # predecessor output NPZ bytes are deliberately not opened by this gate.
    for name, value in manifest.get("files", {}).items():
        if (
            final.get("evidence", {}).get(BASE + "/e339-saved-coordination/" + name)
            != value
        ):
            raise ValueError("E339 manifest/final edge differs")
    for role in ("original", "audit"):
        receipt = read_json(safe(prefix + role + "-external-receipt.json", root))
        sup_name, contract_name = (
            prefix + role + "-supervision.json",
            prefix + role + "-contract.json",
        )
        if receipt.get("passed") is not True or receipt.get("first_error") is not None:
            raise ValueError("E339 external receipt not completed")
        for name, value in receipt.get("evidence", {}).items():
            if final.get("evidence", {}).get(name) != value:
                raise ValueError("E339 external/final descriptor differs")
        # Only this finite set of completion text edges is added, never inherited leaves.
        selected = [
            contract_name,
            prefix + role + "-pids.json",
            prefix + role + ".stdout.txt",
            prefix + role + ".stderr.txt",
        ]
        for name in selected:
            value = final["evidence"][name]
            if (
                receipt["evidence"].get(name) != value
                or pin(safe(name, root), timely) != value
            ):
                raise ValueError("E339 completed text producer edge differs")
            merge_pins(pins, {name: value})
        contract = read_json(safe(contract_name, root))
        sup = read_json(safe(sup_name, root))
        ended = read_json(safe(prefix + role + "-pids.json", root))
        command, attempts = contract.get("commands", []), sup.get("attempts", [])
        if (
            len(command) != 1
            or len(attempts) != 1
            or attempts[0].get("argv") != command[0].get("argv")
            or attempts[0].get("exit_code") != 0
            or attempts[0].get("killed") is not False
            or sup.get("success") is not True
            or sup.get("contract_sha256") != pins[contract_name]["sha256"]
            or receipt["evidence"].get(sup_name) != pins[sup_name]
            or contract.get("original_calls") != 1
            or sup.get("original_calls") != 1
            or ended.get("all_owned_pids_absent") is not True
            or set(ended.get("present_after_return", {}))
            != {str(pid) for pid in sup.get("owned_pids", [])}
            or any(v is not False for v in ended["present_after_return"].values())
        ):
            raise ValueError("E339 completed sole command differs")
    return {
        "closed_stage": "E339",
        "direct_raw_checks": 5591,
        "recursive_inherited_rehash": False,
    }


def make_freeze(*, proof, timely=lambda: None):
    """ROOT metadata preparation only; no original archive open/hash/header."""
    pins = {}
    for name, sha in FIXED.items():
        value = pin(safe(name), timely)
        if value["sha256"] != sha:
            raise ValueError("fixed reflected design dependency differs")
        merge_pins(pins, {name: value})
    deps = read_json(safe(DEPS))
    if deps.get("original_bindings") != original_bindings():
        raise ValueError("literal four archive identities differ")
    merge_pins(pins, deps["direct_text_pins"])
    merge_pins(pins, deps["local_source_pins"])
    for name in (
        REFLECTION,
        ASTRA_DISPATCH,
        REGISTRATION,
        SNAPSHOT,
        SOL_DISPATCH,
        EFFORT,
        SUPERVISOR,
        *OWNED,
    ):
        merge_pins(pins, {name: pin(safe(name), timely)})
    if pins[SUPERVISOR]["sha256"] != SUPERVISOR_SHA:
        raise ValueError("immutable supervisor differs")
    for name, value in pins.items():
        if pin(safe(name), timely) != value:
            raise ValueError("direct pin differs: " + name)
    verify_design_states(ROOT, pins)
    predecessor = verify_predecessor_states(ROOT, pins, timely=timely)
    current = {name: pins[name]["sha256"] for name in OWNED}
    proof_value = read_json(safe(proof))
    qualification = verify_current_proof(ROOT, proof_value, current, timely=timely)
    merge_pins(pins, proof_value["evidence"])
    merge_pins(pins, {proof: pin(safe(proof), timely)})
    return {
        "schema": SCHEMA,
        "source_sha256": current,
        "direct_pins": pins,
        "proof": proof,
        "qualification": qualification,
        "predecessor": predecessor,
        "original_bindings": original_bindings(),
        "output": OUTPUT,
        "limits": LIMITS,
        "executed_local_import_sources": list(EXECUTED_LOCAL),
        "source_text_only_never_imported": list(TEXT_ONLY),
        "original_cumulative_s": 9567.165820199525,
        "original_limit_s": 9660,
        "original_reserved_s": 0,
        "science_adopted": False,
        "candidate_adopted": False,
        "F1_F8_complete": False,
    }


def gate(freeze_sha256, review_sha256, *, timely=lambda: None):
    """No numerical imports or original archive I/O before this complete gate."""
    if (
        digest(safe(FREEZE), timely) != freeze_sha256
        or digest(safe(REVIEW), timely) != review_sha256
    ):
        raise ValueError("literal freeze/source review pin differs")
    frozen, review = read_json(safe(FREEZE)), read_json(safe(REVIEW))
    current = source_map()
    if (
        review.get("schema") != "flylab.m3.e340.root-second-source-review.v1"
        or review.get("passed") is not True
        or review.get("source_sha256") != current
        or review.get("freeze_sha256") != freeze_sha256
        or review.get("ROOT_SECOND_completed") is not True
        or review.get("remaining_ROOT_source_fix_required") is not False
        or frozen != make_freeze(proof=frozen["proof"], timely=timely)
    ):
        raise ValueError("current source review/frozen closure differs")
    timely()
    return frozen


class Clock:
    def __init__(self, s0, *, clock=time.perf_counter):
        self.s0, self.clock = s0, clock
        self.science_deadline, self.internal_deadline = s0 + 12, s0 + 16
        self.final_deadline = None

    def science(self):
        if self.clock() > self.science_deadline:
            raise TimeoutError("absolute science deadline exceeded")

    def begin_final(self):
        if self.final_deadline is None:
            self.final_deadline = min(self.internal_deadline, self.clock() + 4)

    def final(self):
        if self.final_deadline is None or self.clock() > self.final_deadline:
            raise TimeoutError("absolute finalization deadline exceeded")


def memory_usage():
    # CPU process only. No CUDA query or allocation, including for bookkeeping.
    import psutil

    process = psutil.Process()
    rss = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    if rss > LIMITS["host_bytes"]:
        raise MemoryError("owned host resource exceeded")
    return rss


class Budget:
    """Actual file+stage+stdout accounting, full missing quotas and failure reserve."""

    FAILURE_CAP, OVERHEAD_CAP = 512 << 10, 32 << 10

    def __init__(self, output, *, external=None):
        self.output = Path(output).resolve()
        self.caps = {
            self.output / (role + "-gradient.npz"): 640 << 10 for role in ROLES
        }
        self.caps.update(
            {
                self.output / "selected-read-ledger.json": 32 << 10,
                self.output / "output-manifest.json": 16 << 10,
                self.output / "result.json": 256 << 10,
                self.output / "receipt.json": 256 << 10,
            }
        )
        self.external = external or {
            "original.stdout.txt": self.output / "external" / "original.stdout.txt",
            "original.stderr.txt": self.output / "external" / "original.stderr.txt",
            "audit.stderr.txt": self.output / "external" / "audit.stderr.txt",
            "original-supervision.json": self.output
            / "external"
            / "original-supervision.json",
            "audit-supervision.json": self.output
            / "external"
            / "audit-supervision.json",
            "original-pids.json": self.output / "external" / "original-pids.json",
            "audit-pids.json": self.output / "external" / "audit-pids.json",
            "original-external-receipt.json": self.output
            / "external"
            / "original-external-receipt.json",
            "audit-external-receipt.json": self.output
            / "external"
            / "audit-external-receipt.json",
            "audit-report.json": self.output / "external" / "audit-report.json",
            "audit.stdout.txt": self.output / "external" / "audit.stdout.txt",
            "final-decision.json": self.output / "external" / "final-decision.json",
        }
        caps = {
            "original.stdout.txt": 256,
            "original.stderr.txt": 32,
            "audit.stderr.txt": 32,
            "original-supervision.json": 16,
            "audit-supervision.json": 16,
            "original-pids.json": 8,
            "audit-pids.json": 8,
            "original-external-receipt.json": 64,
            "audit-external-receipt.json": 64,
            "audit-report.json": 256,
            "audit.stdout.txt": 256,
            "final-decision.json": 64,
        }
        if set(self.external) != set(caps):
            raise ValueError("exact external reserved roster required")
        self.external = {k: Path(v).resolve() for k, v in self.external.items()}
        if len(set(self.external.values()) | set(self.caps)) != len(caps) + len(
            self.caps
        ):
            raise ValueError("output/external quota path collision")
        self.caps.update({self.external[k]: v << 10 for k, v in caps.items()})
        self.stdout_bytes = 0

    def observe(self):
        files = {
            p.resolve(): p.stat().st_size for p in self.output.rglob("*") if p.is_file()
        }
        for path in self.caps:
            if path.is_file():
                files[path] = path.stat().st_size
            stage = path.with_name(path.name + ".partial")
            if stage.is_file():
                files[stage] = stage.stat().st_size
        return files

    def check(self, *, proposed=None, failure=False):
        files = self.observe()
        if proposed is not None:
            path, size = proposed
            path = Path(path).resolve()
            if path in files:
                raise FileExistsError("fresh write would overwrite existing evidence")
            files[path] = size
        stdout = self.external["original.stdout.txt"]
        files[stdout] = max(files.get(stdout, 0), self.stdout_bytes)
        normal, reserve = 0, 0
        consumed = set()
        for path, cap in self.caps.items():
            stage = path.with_name(path.name + ".partial")
            used = files.get(path, 0) + files.get(stage, 0)
            if used > cap and not failure:
                raise ValueError("per-file stage/target quota exceeded: " + path.name)
            normal += used
            reserve += cap if used == 0 else 0
            consumed.update((path, stage))
        unknown = {p: size for p, size in files.items() if p not in consumed}
        failure_used = sum(
            size
            for p, size in unknown.items()
            if p.name in ("failure.json", "revocation.json", "emergency-failure.json")
            or p.name
            in (
                "failure.json.partial",
                "revocation.json.partial",
                "emergency-failure.json.partial",
            )
        )
        overhead = sum(unknown.values()) - failure_used
        if failure_used > self.FAILURE_CAP or (
            overhead > self.OVERHEAD_CAP and not failure
        ):
            raise ValueError("failure/overhead shared quota exceeded")
        total = normal + sum(unknown.values())
        reserved = (
            0
            if failure
            else (
                reserve
                + max(0, self.FAILURE_CAP - failure_used)
                + max(0, self.OVERHEAD_CAP - overhead)
            )
        )
        if total > LIMITS["output_bytes"] or total + reserved > LIMITS["output_bytes"]:
            raise ValueError(
                "aggregate actual/missing/failure-reserved output quota exceeded"
            )
        return {
            "actual_bytes": total,
            "missing_and_headroom_reserved_bytes": reserved,
            "cap_bytes": LIMITS["output_bytes"],
            "per_file_quota_enforced": True,
        }

    def before_stdout(self, payload):
        if self.stdout_bytes:
            raise ValueError("sole terminal stdout publication required")
        if len(payload) > self.caps[self.external["original.stdout.txt"]]:
            raise ValueError("terminal stdout quota exceeded")
        self.stdout_bytes = len(payload)
        self.check()


def original_external():
    mapping = {}
    for name in (
        "original.stdout.txt",
        "original.stderr.txt",
        "audit.stderr.txt",
        "original-supervision.json",
        "audit-supervision.json",
        "original-pids.json",
        "audit-pids.json",
        "original-external-receipt.json",
        "audit-external-receipt.json",
        "audit-report.json",
        "audit.stdout.txt",
        "final-decision.json",
    ):
        mapping[name] = ROOT / BASE / ("parent-e340-" + name)
    return mapping


def write_bytes(
    path,
    payload,
    budget,
    *,
    timely=lambda: None,
    fault=lambda stage: None,
    failure=False,
):
    """Exclusive stage, no existing target overwritten; actual serialized bytes checked."""
    path = Path(path)
    stage = path.with_name(path.name + ".partial")
    if path.exists() or stage.exists():
        raise FileExistsError("fresh target/stage already exists")
    timely()
    fault("before_write:" + path.name)
    budget.check(proposed=(stage, len(payload)), failure=failure)
    with stage.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    timely()
    budget.check(failure=failure)
    fault("before_rename:" + path.name)
    if path.exists():
        raise FileExistsError("target appeared before rename")
    stage.rename(path)
    timely()
    budget.check(failure=failure)
    return pin(path, timely)


def array_bytes(arrays):
    runtime()
    if set(arrays) != set(ARRAY_SPEC):
        raise ValueError("exact twelve raw array roster required")
    for name, shape in ARRAY_SPEC.items():
        typed(arrays[name], shape, name)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    payload = buffer.getvalue()
    if len(payload) > 640 << 10:
        raise ValueError("serialized gradient archive quota exceeded")
    return payload


def compact_read(detail):
    observed = detail.get("header_observation", {})
    return {
        k: copy.deepcopy(v)
        for k, v in detail.items()
        if k not in ("header_observation", "scanner_failure")
    } | {
        "all_header_observation_sha256": hashlib.sha256(
            json_bytes(observed)
        ).hexdigest(),
        "observed_member_count": len(observed.get("descriptor", {}).get("members", {})),
        "all_headers_observed_not_all_values_read": True,
        "scanner_first_error": detail.get("scanner_failure", {}).get("first_error"),
    }


def error_record(error, stage):
    return {"stage": stage, "type": type(error).__name__, "message": str(error)[:2048]}


def stdout_bytes(payload):
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def verify_output(output, manifest, *, timely=lambda: None):
    runtime()
    output = Path(output)
    for name, value in manifest["files"].items():
        path = safe(name, output)
        if pin(path, timely) != value:
            raise ValueError("output byte readback differs: " + name)
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != set(ARRAY_SPEC):
                    raise ValueError("readback raw roster differs")
                for key, shape in ARRAY_SPEC.items():
                    timely()
                    typed(archive[key], shape, "readback " + key)
    result = (output / "result.json").read_bytes()
    if result != (output / "receipt.json").read_bytes():
        raise ValueError("receipt/terminal bytes differ")
    if read_json(output / "output-manifest.json") != manifest:
        raise ValueError("manifest readback differs")
    return result


def invalid_terminal(first, secondary, *, s0, mode, analysis=None):
    return {
        "schema": RESULT_SCHEMA,
        "status": "technical_invalid",
        "analysis_valid": False,
        "publication_valid": False,
        "provisional": True,
        "suggested_diagnostic": "not_evaluated",
        "science_adopted": False,
        "candidate_adopted": False,
        "F1_F8_complete": False,
        "neural_improvement": False,
        "biology_proven": False,
        "first_error": first,
        "secondary_errors": secondary,
        "clock_s0": s0,
        "mode": mode,
        "analysis": analysis or {},
        "neural_body_CUDA_parameter_VJP_calls": 0,
    }


def preserve_failure(output, terminal, budget):
    """Never modify a provisional success; preserve first and emergency failures."""
    output = Path(output)
    path = output / (
        "revocation.json" if (output / "result.json").exists() else "failure.json"
    )
    try:
        write_bytes(path, json_bytes(terminal), budget, failure=True)
    except Exception as error:  # noqa: BLE001 -- Emergency publication failure is itself retained.
        terminal["secondary_errors"].append(error_record(error, "failure_publication"))
        emergency = output / "emergency-failure.json"
        try:
            write_bytes(emergency, json_bytes(terminal), budget, failure=True)
        except Exception as later:  # noqa: BLE001 -- Keep both failures in returned invalid terminal.
            terminal["secondary_errors"].append(
                error_record(later, "emergency_failure_publication")
            )
    return terminal


def _worker(
    root,
    bindings,
    output,
    *,
    mode,
    clock,
    s0,
    printer=stdout_bytes,
    fault=lambda stage: None,
    meter=memory_usage,
    freeze_sha256=None,
    review_sha256=None,
):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("fresh output required")
    output.mkdir(parents=True, exist_ok=False)
    budget = Budget(
        output, external=original_external() if mode == "original" else None
    )
    watch = Clock(s0, clock=clock)
    reads, analysis, artifacts = {}, {}, {}
    first, secondary, stage = None, [], "numerical_import"
    terminal = None

    def science():
        watch.science()
        meter()

    def final():
        watch.final()
        meter()

    try:
        watch.science()
        runtime()
        science()
        stage = "selected_reference"
        ref, detail = reader.fixed_reader(
            root, "reference", bindings["reference"], namespace=mode, timely=science
        )
        reads["reference"] = compact_read(detail)
        stage = "canonical_CPU_constructor"
        objective = make_objective(ref)
        science()
        for role in ROLES:
            stage = "selected_" + role
            trajectory, detail = reader.fixed_reader(
                root, role, bindings[role], namespace=mode, timely=science
            )
            reads[role] = compact_read(detail)
            stage = "calculate_" + role
            try:
                arrays, report = analyze_trajectory(
                    ref, trajectory, objective=objective, timely=science
                )
            except NumericalFailure as error:
                if hasattr(error, "partial_arrays"):
                    # Qualification is already invalid; preserve finite acquired raw
                    # using only the remaining absolute internal/finalization time.
                    watch.begin_final()
                    try:
                        artifacts[role + "-gradient.npz"] = write_bytes(
                            output / (role + "-gradient.npz"),
                            array_bytes(error.partial_arrays),
                            budget,
                            timely=final,
                            fault=fault,
                        )
                    except Exception as save_error:  # noqa: BLE001 -- Preserve first numerical error when saving finite partial raw fails.
                        secondary.append(
                            error_record(save_error, "partial_raw_publication")
                        )
                raise
            stage = "raw_" + role
            artifacts[role + "-gradient.npz"] = write_bytes(
                output / (role + "-gradient.npz"),
                array_bytes(arrays),
                budget,
                timely=science,
                fault=fault,
            )
            analysis[role] = report
            fault("after_raw:" + role)
            science()
        stage = "final_input_posthash"
        for role, binding in bindings.items():
            after = pin(safe(binding["path"], root), science)
            reads[role]["final_postscience_pin"] = after
            if after != {k: binding[k] for k in ("sha256", "bytes")}:
                raise ValueError("input changed after selected calculation")
        stage = "selected_read_ledger"
        ledger = {
            "schema": "flylab.m3.e340.selected-read-ledger.v1",
            "mode": mode,
            "reads": reads,
            "full_state_neural_weights_read": False,
        }
        artifacts["selected-read-ledger.json"] = write_bytes(
            output / "selected-read-ledger.json",
            json_bytes(ledger),
            budget,
            timely=science,
            fault=fault,
        )
        science()
        watch.begin_final()
        stage = "terminal_publication"
        terminal = {
            "schema": RESULT_SCHEMA,
            "status": "valid_provisional_diagnostic",
            "analysis_valid": True,
            "publication_valid": True,
            "provisional": True,
            "suggested_diagnostic": analysis["live"]["decision"]["classification"],
            "science_adopted": False,
            "candidate_adopted": False,
            "F1_F8_complete": False,
            "neural_improvement": False,
            "biology_proven": False,
            "first_error": None,
            "secondary_errors": [],
            "mode": mode,
            "freeze_sha256": freeze_sha256,
            "review_sha256": review_sha256,
            "clock_s0": s0,
            "science_deadline": watch.science_deadline,
            "internal_deadline": watch.internal_deadline,
            "final_deadline": watch.final_deadline,
            "limits": LIMITS,
            "analysis": analysis,
            "raw_artifacts": artifacts.copy(),
            "neural_body_CUDA_parameter_VJP_calls": 0,
            "interpretation": "Saved conditional output gradient; not an attainable parameter direction or gait proof",
        }
        payload = json_bytes(terminal)
        for name in ("result.json", "receipt.json"):
            artifacts[name] = write_bytes(
                output / name, payload, budget, timely=final, fault=fault
            )
        stage = "manifest_publication"
        manifest = {
            "schema": "flylab.m3.e340.output-manifest.v1",
            "files": artifacts.copy(),
            "self_included": False,
            "late_failure_requires_revocation_check": True,
        }
        write_bytes(
            output / "output-manifest.json",
            json_bytes(manifest),
            budget,
            timely=final,
            fault=fault,
        )
        stage = "byte_readback"
        fault("before_readback")
        readback = verify_output(output, manifest, timely=final)
        if readback != payload:
            raise ValueError("provisional bytes differ from returned terminal")
        final()
        budget.check()
        stage = "terminal_stdout"
        fault("before_stdout")
        final()
        budget.before_stdout(payload)
        printer(payload)
        stage = "postpublication"
        fault("after_stdout")
        final()
        budget.check()
        return terminal, 0
    except Exception as error:  # noqa: BLE001 -- Every failure invalidates and preserves completed raw work.
        first = error_record(error, stage)
        if hasattr(error, "detail"):
            reads[stage] = compact_read(error.detail)
            first = error.detail.get("first_error") or first
            secondary.extend(error.detail.get("secondary_errors", []))
        watch.begin_final()
        terminal = invalid_terminal(
            first, secondary, s0=s0, mode=mode, analysis=analysis
        )
        terminal.update(
            raw_artifacts=artifacts.copy(),
            partial_selected_reads=reads,
            science_deadline=watch.science_deadline,
            internal_deadline=watch.internal_deadline,
            final_deadline=watch.final_deadline,
        )
        if (output / "result.json").exists():
            terminal["revokes_provisional_result_sha256"] = digest(
                output / "result.json"
            )
        preserve_failure(output, terminal, budget)
        return terminal, 1


def generated_worker(
    root,
    bindings,
    *,
    output="diagnosis",
    clock=time.perf_counter,
    printer=stdout_bytes,
    fault=lambda stage: None,
    meter=memory_usage,
):
    """Actual full-schema CPU/selected-reader worker in explicit isolated namespace."""
    s0 = clock()
    environment()
    root = generated_bindings(root, bindings)
    target = safe(output, root)
    return _worker(
        root,
        bindings,
        target,
        mode="generated",
        clock=clock,
        s0=s0,
        printer=printer,
        fault=fault,
        meter=meter,
    )


def _run(s0, freeze_sha256, review_sha256):
    environment()
    watch = Clock(s0)
    try:
        frozen = gate(freeze_sha256, review_sha256, timely=watch.science)
    except Exception as error:
        target = safe(OUTPUT)
        if target.exists():
            raise
        target.mkdir(parents=True, exist_ok=False)
        terminal = invalid_terminal(
            error_record(error, "preimport_gate"), [], s0=s0, mode="original"
        )
        preserve_failure(target, terminal, Budget(target, external=original_external()))
        return terminal, 1
    return _worker(
        ROOT,
        frozen["original_bindings"],
        safe(OUTPUT),
        mode="original",
        clock=time.perf_counter,
        s0=s0,
        freeze_sha256=freeze_sha256,
        review_sha256=review_sha256,
    )


def run(freeze_sha256, review_sha256):
    return _run(time.perf_counter(), freeze_sha256, review_sha256)


def main():
    s0 = time.perf_counter()
    environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review-sha256", required=True)
    arg = parser.parse_args()
    _, code = _run(s0, arg.freeze_sha256, arg.review_sha256)
    return code


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())
