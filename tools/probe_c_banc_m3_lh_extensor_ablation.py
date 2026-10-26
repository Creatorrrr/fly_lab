"""E260: bounded body-free ablation of one existing LH extensor input."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import probe_c_banc_m3_lh_edge_restoration as e250
from tools import probe_c_banc_m3_lh_neural_response as e248
from tools import replay_c_banc_m3_lh_full_inputs as e254

BASE = Path("verification/banc-natural-gait-m3-20260919")
FREEZE_SHA = "c2984fdfb0cf4a1c82a631bbdf4ddd1de6a2a4bf375c045672ee7cdf06459e82"
FREEZE_SCHEMA = "flylab.m3.e260.lh-extensor-ablation.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e260.lh-extensor-ablation.review.v1"
OWNED = ("tools/probe_c_banc_m3_lh_extensor_ablation.py",
         "tests/test_c_banc_lh_extensor_ablation.py")
VALIDITY = "valid_fixed_input_neural_edge_ablation"
ORDER = ("B", "C", "X", "Bcut", "Xcut")
MODELS = ("W0", "W6", "W0cut", "W6cut")
BINDINGS = {"B": ("W0", "baseline"), "C": ("W6", "candidate"),
            "X": ("W6", "baseline"), "Bcut": ("W0cut", "baseline"),
            "Xcut": ("W6cut", "baseline")}
CONTROLS = 100
STEPS = 20
N = e254.N
TARGETS = np.asarray([31172, 47329], np.int64)
CUT_EDGE = 3473458
EDGE_PRE = 111970
EDGE_POST = 47329


def pinned(name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Slash-separated frozen path required")
    relative = Path(name)
    path = (ROOT / relative).resolve()
    if (relative.is_absolute() or ".." in relative.parts
            or not path.is_relative_to(ROOT) or not path.is_file()):
        raise ValueError("Invalid frozen path: " + name)
    return path


def gate(freeze_path: Path, freeze_sha: str, review_path: Path,
         review_sha: str, out: Path) -> dict:
    """Verify frozen bytes and packages before any original array is loaded."""
    if (freeze_sha != FREEZE_SHA or e254.sha256(freeze_path) != FREEZE_SHA
            or e254.sha256(review_path) != review_sha):
        raise ValueError("E260 freeze/review SHA differs")
    freeze, review = e248.read_json(freeze_path), e248.read_json(review_path)
    limits = freeze.get("resource_limits", {})
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("graph_hash") !=
            "9bd8468f5e5e40c3e6daaffde9f444466825ba8e69189f1e347568bff85920ff"
            or freeze.get("controls") != CONTROLS
            or freeze.get("steps_per_control") != STEPS
            or freeze.get("n") != N
            or freeze.get("declared_edges") != 11584852
            or freeze.get("neural_dt_s") != .00025
            or freeze.get("control_dt_s") != .005
            or freeze.get("model_order") != list(MODELS)
            or freeze.get("arm_order") != list(ORDER)
            or freeze.get("arm_bindings") != {k: {"weights": w, "input": u}
                                                 for k, (w, u) in BINDINGS.items()}
            or freeze.get("diagonal_references") != {"B": "baseline",
                                                        "C": "candidate"}
            or freeze.get("startup") != [1, 100]
            or freeze.get("case_controls") != [56, 70]
            or freeze.get("hypothesis") != "H260_existing_edge_attenuation"
            or freeze.get("cut_edge") != {"edge_id": CUT_EDGE,
                "pre_id": EDGE_PRE, "post_id": EDGE_POST,
                "source_value": -0.8800315856933594, "target_value": 0.0}
            or freeze.get("target_ids") != TARGETS.tolist()
            or freeze.get("motor_columns") != [68, 107]
            or freeze.get("muscle_index") != 44
            or freeze.get("joint_column") != 19
            or freeze.get("bridge_ids") != [20156, 63026, 111970, 152280]
            or freeze.get("observed_count") != 440
            or freeze.get("relative_tolerance") != .2
            or freeze.get("activation_atol") != 1e-10
            or freeze.get("require_cut_target_rate_ptp_positive") is not True
            or any(limits.get(key) != value for key, value in {
                "cpu_threads": 1, "gpu_processes": 1, "original_attempts": 1,
                "retries": 0, "neural_constructors": 4, "setup_executed_ticks": 240,
                "setup_capture_ticks": 80, "scientific_neural_ticks": 10000,
                "current_probe_calls": 505, "body_constructors": 0,
                "physics_steps": 0, "fit_calls": 0, "solver_calls": 0,
                "internal_s": 55, "finalization_reserve_s": 8,
                "external_s": 60, "supervisor_wall_s": 65,
                "process_memory_bytes": 4294967296,
                "additional_gpu_memory_bytes": 3221225472,
                "minimum_free_gpu_bytes": 3221225472,
                "output_bytes": 268435456,
                "minimum_free_disk_bytes": 1073741824}.items())):
        raise ValueError("E260 fixed science/resource contract differs")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("E260 source review not approved")
    pins: dict[str, str] = {}
    for mapping in (freeze["input_sha256"], freeze["reused_source_sha256"],
                    freeze["preservation_sha256"], review["source_sha256"],
                    review.get("additional_input_sha256", {}),
                    review.get("additional_reused_source_sha256", {}),
                    {freeze["design"]: freeze["design_sha256"]}):
        if not isinstance(mapping, dict):
            raise TypeError("E260 SHA mapping required")
        for name, digest in mapping.items():
            if name in pins and pins[name] != digest:
                raise ValueError("Conflicting E260 SHA pin: " + name)
            pins[name] = digest
    for name, digest in pins.items():
        if e254.sha256(pinned(name)) != digest:
            raise ValueError("E260 frozen byte differs: " + name)
    for name, version in freeze["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("E260 package version differs: " + name)
    if out.resolve() != (ROOT / freeze["output"]).resolve():
        raise ValueError("E260 output path differs")
    if out.exists() or not out.parent.is_dir():
        raise FileExistsError("E260 output must be new")
    if shutil.disk_usage(out.parent).free < limits["minimum_free_disk_bytes"]:
        raise OSError("E260 minimum free disk unavailable")
    return freeze


def cut_one_existing_edge(csr, edge: dict):
    """Keep CSR addresses and every other entry byte-exact."""
    from scipy.sparse import csr_matrix

    i = edge["edge_id"]
    if (csr.data.dtype != np.float32 or i != CUT_EDGE
            or int(csr.indices[i]) != EDGE_PRE
            or int(np.searchsorted(csr.indptr, i, side="right") - 1) != EDGE_POST
            or csr.data[i].tobytes() != np.float32(edge["source_value"]).tobytes()
            or np.float32(edge["target_value"]).tobytes() != np.float32(0).tobytes()):
        raise ValueError("E260 fixed existing-edge binding differs")
    data = csr.data.copy()
    data[i] = np.float32(0)
    result = csr_matrix((data, csr.indices.copy(), csr.indptr.copy()),
                        shape=csr.shape, dtype=np.float32)
    if (not e254.same_bits(result.indices, csr.indices)
            or not e254.same_bits(result.indptr, csr.indptr)
            or not e254.same_bits(result.data[:i], csr.data[:i])
            or not e254.same_bits(result.data[i + 1:], csr.data[i + 1:])):
        raise AssertionError("E260 non-target CSR bytes changed")
    return result


def exact_float32_patch_values(values, *, count=6) -> np.ndarray:
    """Accept stored float64 provenance only when each value is exactly f32."""
    source = np.asarray(values)
    if (source.shape != (count,) or source.dtype != np.float64
            or not np.isfinite(source).all()):
        raise ValueError("E260 finite float64 six-edge provenance required")
    narrowed = source.astype(np.float32)
    if (not np.isfinite(narrowed).all()
            or not np.array_equal(narrowed.astype(np.float64).view(np.uint64),
                                  source.view(np.uint64))):
        raise ValueError("E260 lossy six-edge float32 provenance")
    return narrowed


def incoming_two(csr, target_ids, observed_ids, output_mask):
    """Original CSR entries for the two fixed MN, in target and slot order."""
    targets = np.asarray(target_ids, np.int64)
    observed = np.asarray(observed_ids, np.int64)
    mask = np.asarray(output_mask)
    if (not np.array_equal(targets, TARGETS) or observed.ndim != 1
            or len(observed) != 440 or not np.array_equal(observed, np.unique(observed))
            or mask.shape != (N,) or mask.dtype != np.float32
            or not np.isin(mask, (0, 1)).all()):
        raise ValueError("E260 target/observed/output-mask mapping differs")
    counts = np.diff(csr.indptr)[targets].astype(np.int64)
    edges = np.concatenate([np.arange(csr.indptr[t], csr.indptr[t + 1],
                                      dtype=np.int64) for t in targets])
    pre = np.asarray(csr.indices[edges], np.int64)
    weights = np.asarray(csr.data[edges], np.float32)
    if not len(edges) or not np.isfinite(weights).all():
        raise ValueError("E260 finite complete incoming rows required")
    return {"target_ids": targets, "edge_ids": edges,
            "target_offsets": np.r_[0, np.cumsum(counts)].astype(np.int64),
            "csr_row_starts": csr.indptr[targets].astype(np.int64),
            "post_ids": np.repeat(targets, counts), "pre_ids": pre,
            "weights32": weights, "observed440_ids": observed,
            "observed_edge_mask": np.isin(pre, observed),
            "union_ids": np.union1d(observed, pre).astype(np.int64),
            "output_mask_start": mask.copy()}


def signed_window_parts(delta, lo: int, hi: int) -> dict[str, float]:
    """Separate signs at each endpoint, then sum; retain signed cancellation."""
    values = np.asarray(delta, np.float64)[lo:hi]
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Nonempty finite endpoint window required")
    return {"signed": float(values.sum(dtype=np.float64)),
            "positive": float(np.maximum(values, 0).sum(dtype=np.float64)),
            "negative": float(np.minimum(values, 0).sum(dtype=np.float64))}


def dynamic_start_equal(first: dict, second: dict, exact=None) -> None:
    """Allow only the documented model identity/patch metadata to differ."""
    if exact is None:
        from tools.verify_c_banc_observation import exact
    exclude = set(e250.METADATA)
    if set(first) != set(second):
        raise AssertionError("Crossed model snapshot fields differ")
    exact({k: v for k, v in first.items() if k not in exclude},
          {k: v for k, v in second.items() if k not in exclude},
          "E260 common unmodified neural dynamic start")


def diagonal_exact(raw: dict, reference: dict, row: int,
                   *, same=e254.same_bits) -> None:
    """Check original E257 saved motor/cohort/bank trajectory bytes."""
    for key in ("motor-rate.npy", "observed-rate.npy", "observed-drive.npy",
                "observed-adaptation.npy", "core-rate.npy", "core-drive.npy",
                "core-adaptation.npy", "bank.npy"):
        if not same(np.asarray(raw[key][row]), np.asarray(reference[key][row])):
            raise AssertionError(f"E260 diagonal {key} differs at {row}")
    if not np.allclose(raw["activation44.npy"][row],
                       reference["activation44.npy"][row], atol=1e-10, rtol=0):
        raise AssertionError(f"E260 diagonal muscle44 activation differs at {row}")


def reference_arrays(folder: Path) -> dict:
    with np.load(folder / "control-raw.npz", allow_pickle=False) as c:
        motor = c["motor_rate32"][:101].copy()
    with np.load(folder / "activity.npz", allow_pickle=False) as a:
        observed = {"observed-" + k + ".npy": a[k][:101].copy()
                    for k in ("rate", "drive", "adaptation")}
        observed["observed_ids"] = a["observed_ids"].copy()
    with np.load(folder / "core-activity.npz", allow_pickle=False) as a:
        core = {"core-" + k + ".npy": a[k][:101].copy()
                for k in ("rate", "drive", "adaptation")}
        core["core_ids"] = a["core_ids"].copy()
    with np.load(folder / "native-recruitment-state.npz", allow_pickle=False) as a:
        bank = a["bank"][:101].copy()
        activation = a["activation"][:101, 44].copy()
    return {"motor-rate.npy": motor, "bank.npy": bank[:, [68, 107], :],
            "activation44.npy": activation, **observed, **core}


def raw_arrays(folder: Path, incoming: dict, drive_count: int) -> dict:
    specs = {"endpoint-rate.npy": (CONTROLS + 1, len(incoming["union_ids"])),
             "endpoint-drive.npy": (CONTROLS + 1, len(incoming["union_ids"])),
             "endpoint-adaptation.npy": (CONTROLS + 1, len(incoming["union_ids"])),
             "actual-current.npy": (CONTROLS + 1, len(TARGETS)),
             "edge-current64.npy": (CONTROLS + 1, len(incoming["edge_ids"])),
             "drive-values.npy": (CONTROLS, drive_count),
             "motor-rate.npy": (CONTROLS + 1, 391),
             "observed-rate.npy": (CONTROLS + 1, 82),
             "observed-drive.npy": (CONTROLS + 1, 82),
             "observed-adaptation.npy": (CONTROLS + 1, 82),
             "core-rate.npy": (CONTROLS + 1, 24),
             "core-drive.npy": (CONTROLS + 1, 24),
             "core-adaptation.npy": (CONTROLS + 1, 24),
             "bank.npy": (CONTROLS + 1, 2, 3),
             "activation44.npy": (CONTROLS + 1,)}
    arrays = {}
    for key, shape in specs.items():
        dtype = (np.float64 if key in ("edge-current64.npy", "bank.npy",
                                       "activation44.npy") else np.float32)
        arrays[key] = np.lib.format.open_memmap(folder / key, mode="w+",
                                               dtype=dtype, shape=shape)
        arrays[key][:] = np.nan
        arrays[key].flush()
    return arrays


def bank_activation_step(previous_bank, rate, tau, kappa, slopes, membership,
                         *, muscle=44):
    """The original monotone decoder's -expm1 update and muscle hinge."""
    bank = np.asarray(previous_bank, np.float64).copy()
    rates = np.asarray(rate, np.float32)
    if (bank.shape != (2, 3) or rates.shape != (2,)
            or np.asarray(tau).shape != (3,) or np.asarray(kappa).shape != (2,)
            or np.asarray(slopes).shape != (2, 3, 2)
            or not np.array_equal(membership, [muscle, muscle])):
        raise ValueError("E260 two-MN muscle44 decoder binding differs")
    alpha = -np.expm1(-.005 / np.asarray(tau, np.float64))
    bank += alpha[None, :] * (rates[:, None] - bank)
    low = slopes[:, :, 0]
    delta = slopes[:, :, 1] - low
    hinge = np.maximum(bank - np.asarray(kappa)[:, None], 0.0)
    activation = np.sum(low * bank) + np.sum(delta * hinge)
    if not np.isfinite(bank).all() or not np.isfinite(activation):
        raise ValueError("E260 nonfinite decoder recurrence")
    return bank, float(activation)


def observe_endpoint(network, row: int, incoming: dict, raw: dict,
                     entry: dict, context: dict, *, probe=e254.observe_network) -> None:
    from tools.verify_c_banc_observation import exact

    before = network.snapshot() if row in (0, CONTROLS) else None
    probe(network, row, incoming["union_ids"], incoming, raw, entry)
    at = context["union_positions"]
    for key, ids in (("motor", context["motor_ids"]),
                     ("observed", context["observed_ids"]),
                     ("core", context["core_ids"])):
        indices = at[key]
        if not np.array_equal(incoming["union_ids"][indices], ids):
            raise AssertionError("E260 selected readout order changed")
        fields = ("rate",) if key == "motor" else ("rate", "drive", "adaptation")
        for field in fields:
            source = "endpoint-" + field + ".npy"
            destination = ("motor-rate.npy" if key == "motor" else
                           key + "-" + field + ".npy")
            raw[destination][row] = raw[source][row, indices]
            raw[destination].flush()
    if row:
        previous = raw["bank.npy"][row - 1]
        rate = raw["motor-rate.npy"][row, [68, 107]]
        bank, activation = bank_activation_step(
            previous, rate, context["tau"], context["kappa"],
            context["slopes"], context["membership"])
        raw["bank.npy"][row] = bank
        raw["activation44.npy"][row] = activation
        raw["bank.npy"].flush()
        raw["activation44.npy"].flush()
    if before is not None:
        exact(network.snapshot(), before, "E260 read-only neural endpoint probe")


def run_arm(network, saved: dict, input_data: dict, incoming: dict,
                    drive_ids: np.ndarray, navigation_ports: np.ndarray,
                    descending_ids: np.ndarray, raw: dict, entry: dict, context: dict,
                    persist, budget, reference: dict | None = None,
                    *, observe=observe_endpoint, controls=CONTROLS,
                    n=N, steps=STEPS,
                    reconstruct=e254.reconstruct_drive) -> None:
    """Restore exactly, then replay one original input sequence for 100 controls."""
    from tools.verify_c_banc_observation import exact

    network.restore(saved)
    exact(network.snapshot(), saved, "E260 arm restored dynamic start")
    entry["reset_exact"] = True
    persist()
    observe(network, 0, incoming, raw, entry, context)
    persist()
    if reference is not None:
        diagonal_exact(raw, reference, 0)
    for control in range(1, controls + 1):
        budget()
        sensory, drive = reconstruct(
            input_data["sensory"]["old_filtered"][control - 1],
            input_data["sensory_ids"], input_data["sensory_ports"], n,
            descending_ids, 0., navigation_ports, 5000.)
        sources = input_data["sensory"]["source_ids"]
        if not e254.same_bits(sensory[sources],
                              input_data["sensory"]["returned_source_drive"][control - 1]):
            raise AssertionError(f"E260 frozen sensory drive differs at {control}")
        raw["drive-values.npy"][control - 1] = drive[drive_ids]
        raw["drive-values.npy"].flush()
        entry["entered_controls"] = control
        entry["unknown_neural_partial"] = True
        persist()
        network.advance(drive, steps)
        entry["completed_controls"] = control
        entry["unknown_neural_partial"] = False
        persist()
        observe(network, control, incoming, raw, entry, context)
        persist()
        if reference is not None:
            diagonal_exact(raw, reference, control)
    entry["status"] = "science_complete"


def cut_edge_position(incoming: dict) -> int:
    at = np.flatnonzero(incoming["edge_ids"] == CUT_EDGE)
    if (len(at) != 1 or incoming["pre_ids"][at[0]] != EDGE_PRE
            or incoming["post_ids"][at[0]] != EDGE_POST):
        raise ValueError("E260 fixed cut entry absent from target incoming row")
    return int(at[0])


def verify_target_parameters(network, expected: dict) -> None:
    for key, source in (("threshold", network.threshold),
                        ("coefficient", network.coefficient),
                        ("cap", network.cap),
                        ("membrane_tau", network.tau),
                        ("cell_gain", network._cell_gain),
                        ("cell_adaptation_tau", network._cell_adaptation_tau),
                        ("inverse_adaptation_tau", network._cell_inverse_tau)):
        if not e254.same_bits(expected[key],
                              source[TARGETS].detach().cpu().numpy()):
            raise AssertionError("E260 target physiological parameter changed: " + key)


def ablation_contrasts(values: dict) -> dict[str, np.ndarray]:
    if set(values) != set(ORDER):
        raise ValueError("All five E260 arms required")
    arrays = {k: np.asarray(v, np.float64) for k, v in values.items()}
    if len({x.shape for x in arrays.values()}) != 1 or not all(
            np.isfinite(x).all() for x in arrays.values()):
        raise ValueError("Complete finite E260 values required")
    b, c, x, bc, xc = (arrays[k] for k in ORDER)
    return {"T": c - b, "C0": x - b, "Ccut": xc - bc,
            "R": (x - b) - (xc - bc), "Bcut_minus_B": bc - b,
            "Xcut_minus_X": xc - x, "C_minus_X": c - x}


def factors(raws: dict, incoming: dict, threshold: np.ndarray) -> tuple[dict, dict]:
    """Retain endpoint contrasts and descriptive window parts; fixed H260 only."""
    union = incoming["union_ids"]
    at = np.searchsorted(union, TARGETS)
    if np.any(at >= len(union)) or not np.array_equal(union[at], TARGETS):
        raise ValueError("E260 target readout alignment differs")
    offsets = incoming["target_offsets"]
    if (threshold.shape != (2,) or threshold.dtype != np.float32
            or not np.isfinite(threshold).all()):
        raise ValueError("E260 original target threshold required")
    observed: dict[str, dict[str, np.ndarray]] = {}
    for arm in ORDER:
        raw = raws[arm]
        current64 = np.stack([
            np.asarray(raw["edge-current64.npy"][:, offsets[j]:offsets[j + 1]],
                       np.float64).sum(axis=1, dtype=np.float64)
            for j in range(2)], axis=1)
        observed[arm] = {
            "rate": np.asarray(raw["motor-rate.npy"][:, [68, 107]], np.float64),
            "drive": np.asarray(raw["endpoint-drive.npy"][:, at], np.float64),
            "negative_adaptation": -np.asarray(
                raw["endpoint-adaptation.npy"][:, at], np.float64),
            "actual_current": np.asarray(raw["actual-current.npy"], np.float64),
            "current64": current64,
            "margin64": current64 + np.asarray(raw["endpoint-drive.npy"][:, at],
                                                  np.float64) - threshold.astype(np.float64)
            - np.asarray(raw["endpoint-adaptation.npy"][:, at], np.float64),
            "edge_current64": np.asarray(raw["edge-current64.npy"], np.float64),
            "bank": np.asarray(raw["bank.npy"], np.float64).reshape(101, 6),
            "activation44": np.asarray(raw["activation44.npy"], np.float64)[:, None],
        }
    output = {"target_ids": TARGETS, "edge_ids": incoming["edge_ids"],
              "edge_pre_ids": incoming["pre_ids"],
              "edge_post_ids": incoming["post_ids"]}
    report = {"schema": "flylab.m3.e260.factor-windows.v1", "windows": {},
              "cut_edge_id": CUT_EDGE, "control_cases": [56, 70],
              "rate_diagnostics": {}}
    windows = {"startup": (1, 100), "case56": (56, 57), "case70": (70, 71)}
    for variable in observed["B"]:
        for arm in ORDER:
            output[f"{variable}_{arm}"] = observed[arm][variable]
        effects = ablation_contrasts({a: observed[a][variable] for a in ORDER})
        for kind, values in effects.items():
            output[f"{variable}_{kind}"] = values
        report["windows"][variable] = {
            name: {kind: [signed_window_parts(values[:, j], lo, hi)
                          for j in range(values.shape[1])]
                   for kind, values in effects.items()}
            for name, (lo, hi) in windows.items()}
    for arm in ORDER:
        rate = observed[arm]["rate"][1:100]
        report["rate_diagnostics"][arm] = [{
            "motor_id": int(TARGETS[j]), "min": float(np.min(rate[:, j])),
            "max": float(np.max(rate[:, j])), "mean": float(np.mean(rate[:, j])),
            "range": float(np.ptp(rate[:, j])),
            "exact_zero_fraction": float(np.mean(rate[:, j] == 0)),
            "exact_cap200_fraction": float(np.mean(rate[:, j] == 200))}
            for j in range(2)]
    activations = {a: observed[a]["activation44"][:, 0] for a in ORDER}
    effects = ablation_contrasts(activations)
    sums = {k: signed_window_parts(v, 1, 100) for k, v in effects.items()}
    t, c0, ccut = (sums[k]["signed"] for k in ("T", "C0", "Ccut"))
    cut_dynamic = all(report["rate_diagnostics"][a][1]["range"] > 0
                      for a in ("Bcut", "Xcut"))
    supported = bool(t < 0 and c0 < 0 and abs(c0 - t) <= .2 * abs(t)
                     and abs(ccut) <= .2 * abs(c0) and cut_dynamic)
    report["H260"] = {"effects": sums, "T": t, "C0": c0, "Ccut": ccut,
                      "R": sums["R"]["signed"],
                      "cut_target_rate_dynamic": cut_dynamic,
                      "relative_tolerance": .2,
                      "H260_existing_edge_attenuation": supported,
                      "scope": "fixed-input neural lesion; no body or candidate adoption"}
    return output, report


def prepare(freeze: dict) -> dict:
    """Read only pinned original files after gate; construct no model or body."""
    from scipy.sparse import csr_matrix

    from flylab.c.storage import StateStore
    from tools.banc_m3_intrinsic_recruitment import patch_tau

    path = lambda key: pinned(freeze["paths"][key])
    config = e254.read_npz(path("configuration"),
                           ("membrane_tau", "cell_gain", "motor_ids"))
    cal = e254.read_npz(path("calibration"), ("tau_ids", "tau_values"))
    tau = patch_tau(np.full(N, .02, np.float32), cal["tau_ids"], cal["tau_values"])
    if (not e254.same_bits(tau, config["membrane_tau"])
            or config["motor_ids"].shape != (391,)
            or not np.array_equal(config["motor_ids"][[68, 107]], TARGETS)):
        raise ValueError("E260 exact original neural/motor physiology differs")
    csr = {}
    for model, key in (("W0", "baseline_csr"), ("W6", "candidate_csr")):
        values = e254.read_npz(path(key), ("indptr", "indices", "data"))
        if (values["data"].dtype != np.float32
                or len(values["data"]) != freeze["declared_edges"]
                or values["indptr"].shape != (N + 1,)
                or values["indices"].shape != values["data"].shape):
            raise ValueError("E260 original CSR precision/shape differs")
        csr[model] = csr_matrix((values["data"], values["indices"],
                                 values["indptr"]), shape=(N, N), dtype=np.float32)
        csr[model].check_format(full_check=True)
    if (not e254.same_bits(csr["W0"].indices, csr["W6"].indices)
            or not e254.same_bits(csr["W0"].indptr, csr["W6"].indptr)
            or np.count_nonzero(csr["W0"].data.view(np.uint32) !=
                                csr["W6"].data.view(np.uint32)) != 6):
        raise ValueError("E260 W6 must preserve structure and change six weights")
    patch = e254.read_npz(path("edge_patch"), ("edge_ids", "pre_ids", "post_ids",
                                               "source_values", "target_values"))
    selected = patch["edge_ids"]
    old32 = exact_float32_patch_values(patch["source_values"])
    new32 = exact_float32_patch_values(patch["target_values"])
    if (len(selected) != 6 or not np.array_equal(np.sort(selected),
                np.flatnonzero(csr["W0"].data.view(np.uint32) !=
                               csr["W6"].data.view(np.uint32)))
            or not e254.same_bits(csr["W0"].data[selected], old32)
            or not e254.same_bits(csr["W6"].data[selected], new32)):
        raise ValueError("E260 frozen six-edge patch differs")
    csr["W0cut"] = cut_one_existing_edge(csr["W0"], freeze["cut_edge"])
    csr["W6cut"] = cut_one_existing_edge(csr["W6"], freeze["cut_edge"])
    if not e254.same_bits(csr["W0cut"].data[CUT_EDGE],
                          csr["W6cut"].data[CUT_EDGE]):
        raise AssertionError("E260 cut values differ")
    ports = e248.read_json(path("drive_ports"))
    if (ports.get("graph_hash") != freeze["graph_hash"] or ports.get("n") != N
            or ports.get("navigation_cut") is not True
            or ports.get("navigation_forward_drive") != 5000.
            or ports.get("p_descending_drive") != 0.):
        raise ValueError("E260 original tonic/descending provenance differs")
    descending = np.asarray(ports["descending"], np.int64)
    navigation = np.asarray(ports["ports"], np.int64)
    if (descending.shape != (2,) or navigation.shape != (20,)
            or np.unique(np.r_[descending, navigation]).size != 22):
        raise ValueError("E260 20+2 frozen navigation rows differ")
    roster = e248.read_json(path("observation_roster"))
    core = np.asarray(roster["original24"], np.int64)
    additional = np.asarray(roster["additional_core_ids"], np.int64)
    if (roster["observed_union_count"] != 440 or core.shape != (24,)
            or additional.shape != (22,) or np.unique(core).size != 24):
        raise ValueError("E260 original observation cohort differs")
    arms = {}
    for name, folder in freeze["arms"].items():
        source = ROOT / folder
        whole = StateStore.load(source / "observed-start")
        if (whole["schema"] != "flylab.m3.e257.lh-partition-trial.v1"
                or whole["control_tick"] != 0 or whole["network"]["tick"] != 0
                or whole["network"]["device"] != "cuda"
                or whole["graph_hash"] != freeze["graph_hash"]
                or whole["cuts"] != {key: False for key in
                    ("sensory", "descending", "circuit", "motor")}
                or whole["parameters"]["descending_drive"] != 0.
                or whole["parameters"]["neural_dt_s"] != .00025
                or whole["navigation"]["cut"] is not True
                or whole["navigation"]["config"]["forward_drive"] != 5000.
                or whole["navigation"]["encoder"]["hash"] != ports["identity"]
                or whole["navigation"]["encoder"]["turn"] != 0
                or whole["navigation"]["encoder"]["braking"] is not False
                or not e254.same_bits(whole["network"]["output_mask"],
                                      np.ones(N, np.float32))):
            raise ValueError("E260 exact E257 neural/whole start required")
        sensory = e254.read_npz(source / "sensory-input.npz",
                                ("control_tick_before", "complete", "old_filtered",
                                 "source_ids", "returned_source_drive"))
        metadata = e248.read_json(source / "sensory-metadata.json")
        if (sensory["old_filtered"].shape[0] < CONTROLS
                or sensory["old_filtered"].shape[1:] != (241,)
                or sensory["old_filtered"].dtype != np.float64
                or not sensory["complete"][:CONTROLS].all()
                or not np.array_equal(sensory["control_tick_before"][:CONTROLS],
                                      np.arange(CONTROLS))
                or sensory["returned_source_drive"].dtype != np.float32
                or sensory["returned_source_drive"].shape[1:] != (23,)
                or not e254.same_bits(sensory["source_ids"],
                                      np.asarray(metadata["source_rows"], np.int64))):
            raise ValueError("E260 saved sensory timing/shape differs")
        sensory_ids = np.asarray(metadata["sensory_ids"], np.int64)
        sensory_ports = np.asarray(metadata["sensory_ports"], np.int64)
        if (sensory_ids.shape != sensory_ports.shape
                or np.any(sensory_ports < 0) or np.any(sensory_ports >= 241)):
            raise ValueError("E260 saved sensory binding differs")
        references = reference_arrays(source)
        arms[name] = {"start": whole["network"], "whole": whole,
                      "sensory": sensory, "sensory_ids": sensory_ids,
                      "sensory_ports": sensory_ports,
                      "reference": references}
    if set(arms) != {"baseline", "candidate"}:
        raise ValueError("E260 paired E257 arms missing")
    dynamic_start_equal(arms["baseline"]["start"], arms["candidate"]["start"])
    if (not e254.same_bits(arms["baseline"]["sensory"]["source_ids"],
                           arms["candidate"]["sensory"]["source_ids"])
            or not e254.same_bits(arms["baseline"]["sensory_ids"],
                                   arms["candidate"]["sensory_ids"])
            or not e254.same_bits(arms["baseline"]["sensory_ports"],
                                   arms["candidate"]["sensory_ports"])):
        raise ValueError("E260 paired encoder mapping differs")
    observed = np.unique(np.r_[arms["baseline"]["sensory"]["source_ids"],
                               freeze["bridge_ids"], config["motor_ids"], additional])
    if len(observed) != 440:
        raise ValueError("E260 original observed440 union differs")
    incoming = {key: incoming_two(value, TARGETS, observed,
                                  arms["baseline"]["start"]["output_mask"])
                for key, value in csr.items()}
    keys = ("target_ids", "edge_ids", "target_offsets", "csr_row_starts",
            "post_ids", "pre_ids", "union_ids", "output_mask_start")
    if any(not e254.same_bits(incoming["W0"][field], incoming[model][field])
           for model in MODELS for field in keys):
        raise ValueError("E260 two target incoming topology/mask differs")
    decoder = arms["baseline"]["whole"]["adapter"]
    other = arms["candidate"]["whole"]["adapter"]
    if not e254.same_bits(decoder["bank"], other["bank"]):
        raise ValueError("E260 original decoder bank start differs")
    params = decoder["parameters"]
    for field in ("tau", "kappa", "slopes", "membership"):
        if not e254.same_bits(params[field], other["parameters"][field]):
            raise ValueError("E260 anatomical decoder parameter differs: " + field)
    if (decoder["bank"].shape != (391, 3)
            or decoder["activation"].shape != (96,)
            or not np.array_equal(np.flatnonzero(params["membership"] == 44),
                                  [68, 107])):
        raise ValueError("E260 fixed muscle44 ownership differs")
    drive_ids = np.unique(np.r_[arms["baseline"]["sensory_ids"],
                                navigation, descending]).astype(np.int64)
    return {"config": config, "tau": tau, "csr": csr, "ports": ports,
            "navigation_ports": navigation, "descending_ids": descending,
            "arms": arms, "incoming": incoming, "drive_ids": drive_ids,
            "observed_ids": observed, "core_ids": core,
            "decoder": params, "bank0": decoder["bank"][[68, 107]].copy(),
            "activation0": float(decoder["activation"][44])}


def begin_arm(ledger: dict, arm: str) -> dict:
    entry = {"weights": BINDINGS[arm][0], "input": BINDINGS[arm][1],
             "requested_controls": CONTROLS, "reset_requested": True,
             "reset_exact": False, "entered_controls": 0, "completed_controls": 0,
             "unknown_neural_partial": False, "current_probe_requested": 0,
             "current_probe_entered": 0, "current_probe_completed": 0,
             "observed_endpoint_count": 0, "initial_saved": False,
             "final_saved": False, "diagonal_exact": None, "status": "requested"}
    ledger["arms"][arm] = entry
    return entry


def resource_check(freeze: dict, began: float, out: Path, ledger: dict,
                   *, snapshot=None) -> None:
    import psutil
    import torch

    limits = freeze["resource_limits"]
    if snapshot is None:
        memory = psutil.Process().memory_info()
        snapshot = {"elapsed_s": time.perf_counter() - began,
                    "resident_bytes": getattr(memory, "peak_wset", memory.rss),
                    "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "gpu_free_bytes": torch.cuda.mem_get_info()[0],
                    "disk_free_bytes": shutil.disk_usage(out).free}
    if len(ledger["resources"]) < 20:
        ledger["resources"].append(snapshot)
    if (snapshot["elapsed_s"] >= limits["internal_s"] - limits["finalization_reserve_s"]
            or snapshot["resident_bytes"] > limits["process_memory_bytes"]
            or snapshot["gpu_peak_allocated_bytes"] >
            limits["additional_gpu_memory_bytes"]
            or snapshot["gpu_free_bytes"] < limits["minimum_free_gpu_bytes"]
            or snapshot["disk_free_bytes"] < limits["minimum_free_disk_bytes"]):
        raise TimeoutError("E260 original science/resource cap reached")


def output_manifest(out: Path, limit: int) -> dict:
    files, total = {}, 0
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path.name in ("result.json", "output-manifest.json"):
            continue
        size = path.stat().st_size
        total += size
        if total > limit:
            raise OSError("E260 output bytes exceeded")
        files[path.relative_to(out).as_posix()] = {
            "sha256": e254.sha256(path), "bytes": size}
    return {"schema": "flylab.m3.e260.output-manifest.v1", "bytes": total,
            "files": files}


def revoke_factor_report(out: Path, *, persist=e248.write_json) -> list[dict]:
    """Keep numeric windows, but never a canonical true H260 after failure."""
    path = out / "factor-windows.json"
    if not path.exists():
        return []
    secondary = []
    try:
        report = e248.read_json(path)
        if not isinstance(report.get("H260"), dict):
            raise TypeError("E260 factor report H260 object missing")
        report["H260"]["H260_existing_edge_attenuation"] = False
        report["H260"]["interpretation_status"] = "unvalidated_incomplete"
        report["interpretation_status"] = "unvalidated_incomplete"
        persist(path, report)
        return secondary
    except Exception as error:  # noqa: BLE001
        secondary.append({"stage": "factor report invalidation",
                          "type": type(error).__name__, "message": str(error)})
    # If the write itself is unavailable, retain the exact raw numeric report
    # under an unmistakably unvalidated filename, never under the canonical one.
    try:
        path.replace(out / "factor-windows-unvalidated-raw.json")
    except Exception as error:  # noqa: BLE001
        secondary.append({"stage": "factor report fallback rename",
                          "type": type(error).__name__, "message": str(error)})
        try:
            path.unlink(missing_ok=True)
        except Exception as unlink_error:  # noqa: BLE001
            secondary.append({"stage": "factor report stale removal",
                              "type": type(unlink_error).__name__,
                              "message": str(unlink_error)})
    return secondary


def finish_receipts(out: Path, freeze: dict, ledger: dict, result: dict,
                    errors: list, primary, began: float,
                    *, persist=e248.write_json):
    """Publish only complete consistent receipts; revoke on every late fault."""
    limit = freeze["resource_limits"]["internal_s"]

    def fail(error, stage):
        nonlocal primary
        if primary is None:
            primary = error
        errors.append({"stage": stage, "type": type(error).__name__,
                       "message": str(error)})
        result.update(validity="resource_limited" if isinstance(primary, TimeoutError)
                      else "invalid_or_incomplete",
                      H260_existing_edge_attenuation=False,
                      factor_report_available=False,
                      finalization_pending=True,
                      first_error={"type": type(primary).__name__,
                                   "message": str(primary)})
        if isinstance(result.get("H260"), dict):
            # Retain measured contrasts, never a support verdict from an
            # incomplete science/cleanup/publication path.
            result["H260"]["H260_existing_edge_attenuation"] = False
            result["H260"]["interpretation_status"] = "unvalidated_incomplete"
        errors.extend(revoke_factor_report(out, persist=persist))

    if primary is not None:
        fail(primary, "prior science/cleanup")
    result["work"] = {
        "constructors_requested": sum(x["requested"] for x in ledger["models"].values()),
        "constructors_entered": sum(x["entered"] for x in ledger["models"].values()),
        "constructors_completed": sum(x["completed"] for x in ledger["models"].values()),
        "setup_executed_ticks_known": sum(x["setup_executed_ticks"] or 0
                                          for x in ledger["models"].values()),
        "setup_capture_ticks_known": sum(x["setup_capture_ticks"] or 0
                                         for x in ledger["models"].values()),
        "scientific_ticks_completed": STEPS * sum(x["completed_controls"]
                                                    for x in ledger["arms"].values()),
        "current_probe_calls_completed": sum(x["current_probe_completed"]
                                              for x in ledger["arms"].values()),
        "unknown_partial_models": [k for k, x in ledger["models"].items()
                                   if x["setup_unknown_partial"]],
        "unknown_partial_arms": [k for k, x in ledger["arms"].items()
                                 if x["unknown_neural_partial"]]}
    for name, value in (("advance-ledger.json", ledger),
                        ("errors.json", {"errors": errors})):
        try:
            persist(out / name, value)
        except Exception as error:  # noqa: BLE001
            fail(error, name)
    if time.perf_counter() - began >= limit:
        fail(TimeoutError("E260 finalization deadline"), "deadline")
    try:
        persist(out / "errors.json", {"errors": errors})
        persist(out / "output-manifest.json",
                output_manifest(out, freeze["resource_limits"]["output_bytes"]))
        result["elapsed_s"] = time.perf_counter() - began
        if result["elapsed_s"] >= limit:
            raise TimeoutError("E260 publication deadline")
        result["finalization_pending"] = False
        persist(out / "result.json", result)
        if time.perf_counter() - began >= limit:
            raise TimeoutError("E260 late result deadline")
    except Exception as error:  # noqa: BLE001
        fail(error, "publication")
        (out / "result.json").unlink(missing_ok=True)
        (out / "output-manifest.json").unlink(missing_ok=True)
        # Computed numbers remain inspectable; the support interpretation is
        # explicitly revoked above, even if later publication is incomplete.
        for name, value in (("advance-ledger.json", ledger),
                            ("result.json", result),
                            ("errors.json", {"errors": errors})):
            try:
                persist(out / name, value)
            except Exception as secondary:  # noqa: BLE001
                fail(secondary, "retry " + name)
                (out / name).unlink(missing_ok=True)
    return primary


def execute(freeze: dict, out: Path) -> None:
    from flylab.c.storage import StateStore
    from tools.verify_c_banc_observation import exact

    began = time.perf_counter()
    out.mkdir(parents=False, exist_ok=False)
    ledger = {"schema": "flylab.m3.e260.ledger.v1", "status": "entered",
              "models": {}, "arms": {}, "resources": [],
              "original_attempts": 1, "retries": 0,
              "body_constructors": 0, "physics_steps": 0,
              "fit_calls": 0, "solver_calls": 0}
    result = {"schema": "flylab.m3.e260.result.v1", "validity": "incomplete",
              "H260_existing_edge_attenuation": False,
              "factor_report_available": False, "candidate_adopted": False,
              "whole_goal_pass": False, "F6_exposed": False,
              "active_UI_changed": False, "finalization_pending": True}
    errors, primary = [], None
    network, active_dir, active_arm = None, None, None
    raws, limiter, old_torch_threads, old_environment = {}, None, None, {}
    try:
        import torch
        from threadpoolctl import threadpool_info, threadpool_limits

        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            old_environment[name] = os.environ.get(name)
            os.environ[name] = "1"
        old_torch_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        limiter = threadpool_limits(limits=1)
        if (torch.get_num_threads() != 1
                or any(x["num_threads"] != 1 for x in threadpool_info())):
            raise RuntimeError("E260 effective CPU thread count differs")
        e248.write_json(out / "threading.json", {"torch_threads": 1,
                         "threadpools": threadpool_info()})
        if torch.cuda.mem_get_info()[0] < freeze["resource_limits"]["minimum_free_gpu_bytes"]:
            raise MemoryError("E260 minimum free GPU memory unavailable")
        torch.cuda.reset_peak_memory_stats()
        inputs = prepare(freeze)
        drive_ids = inputs["drive_ids"]
        e248.write_json(out / "csr-fingerprints.json", {
            model: {field: hashlib.sha256(np.ascontiguousarray(
                getattr(inputs["csr"][model], field)).tobytes()).hexdigest()
                for field in ("indptr", "indices", "data")}
            for model in MODELS})
        for model in MODELS:
            np.savez(out / f"incoming-{model}.npz", **inputs["incoming"][model])
        np.savez(out / "drive-mapping.npz",
                 sensory_ids=inputs["arms"]["baseline"]["sensory_ids"],
                 sensory_ports=inputs["arms"]["baseline"]["sensory_ports"],
                 navigation_ports=inputs["navigation_ports"],
                 descending_ids=inputs["descending_ids"], drive_ids=drive_ids)
        np.savez(out / "decoder-parameters.npz", target_ids=TARGETS,
                 motor_ids=inputs["config"]["motor_ids"],
                 muscle_index=np.int64(44), motor_columns=np.asarray([68, 107], np.int64),
                 bank0=inputs["bank0"], activation0=inputs["activation0"],
                 tau=inputs["decoder"]["tau"],
                 kappa=inputs["decoder"]["kappa"][[68, 107]],
                 slopes=inputs["decoder"]["slopes"][44, [68, 107]],
                 membership=inputs["decoder"]["membership"][[68, 107]])
        observed_ids = inputs["arms"]["baseline"]["reference"]["observed_ids"]
        core_ids = inputs["arms"]["baseline"]["reference"]["core_ids"]
        if (observed_ids.shape != (82,) or core_ids.shape != (24,)
                or not e254.same_bits(observed_ids,
                    inputs["arms"]["candidate"]["reference"]["observed_ids"])
                or not e254.same_bits(core_ids,
                    inputs["arms"]["candidate"]["reference"]["core_ids"])
                or not e254.same_bits(core_ids, inputs["core_ids"])):
            raise ValueError("E260 original82/core24 roster differs")
        target_parameters = None
        for model in MODELS:
            model_entry = {"requested": True, "entered": False,
                           "completed": False, "setup_unknown_partial": False,
                           "setup_executed_ticks": None,
                           "setup_capture_ticks": None,
                           "status": "requested"}
            ledger["models"][model] = model_entry
            e248.write_json(out / "advance-ledger.json", ledger)
            resource_check(freeze, began, out, ledger)
            model_entry.update(entered=True, setup_unknown_partial=True,
                               status="constructing")
            e248.write_json(out / "advance-ledger.json", ledger)
            source_name = "baseline" if model.startswith("W0") else "candidate"
            source = inputs["arms"][source_name]
            network, restored, identity_fields, backend = e250._constructor(
                inputs["csr"][model], inputs["tau"], inputs["config"],
                source["start"], expected_identity=(
                    source["start"]["model_hash"] if model in ("W0", "W6") else None),
                restore_saved=(source["start"] if model in ("W0", "W6") else None))
            if bool(identity_fields) != model.endswith("cut"):
                raise AssertionError("E260 identity transplant scope differs")
            model_entry.update(completed=True, setup_unknown_partial=False,
                               setup_executed_ticks=60, setup_capture_ticks=20,
                               backend=backend, status="ready")
            if model in ("W0", "W6"):
                exact(restored, source["start"], "E260 original model start")
            else:
                dynamic_start_equal(restored, source["start"])
            if target_parameters is None:
                target_parameters = {"target_ids": TARGETS.copy()}
                for key, tensor in (("threshold", network.threshold),
                    ("coefficient", network.coefficient), ("cap", network.cap),
                    ("membrane_tau", network.tau), ("cell_gain", network._cell_gain),
                    ("cell_adaptation_tau", network._cell_adaptation_tau),
                    ("inverse_adaptation_tau", network._cell_inverse_tau)):
                    target_parameters[key] = tensor[TARGETS].detach().cpu().numpy().copy()
                np.savez(out / "target-parameters.npz", **target_parameters)
            verify_target_parameters(network, target_parameters)
            model_entry["identity_fields"] = identity_fields
            model_entry["model_hash"] = network.identity
            model_entry["source_model_hash"] = source["start"]["model_hash"]
            e248.write_json(out / f"{model}-binding.json", model_entry)
            e248.write_json(out / "advance-ledger.json", ledger)
            for arm in (a for a in ORDER if BINDINGS[a][0] == model):
                arm_dir = out / arm
                arm_dir.mkdir(exist_ok=False)
                entry = begin_arm(ledger, arm)
                active_dir, active_arm = arm_dir, entry
                e248.write_json(out / "advance-ledger.json", ledger)
                incoming = inputs["incoming"][model]
                raw = raw_arrays(arm_dir, incoming, len(drive_ids))
                raws[arm] = raw
                reference = source["reference"] if arm in ("B", "C") else None
                union = incoming["union_ids"]
                ids = {"motor": inputs["config"]["motor_ids"],
                       "observed": observed_ids, "core": core_ids}
                positions = {k: np.searchsorted(union, v) for k, v in ids.items()}
                if any(np.any(p >= len(union)) or not np.array_equal(union[p], ids[k])
                       for k, p in positions.items()):
                    raise ValueError("E260 full motor/cohort readout union incomplete")
                context = {"motor_ids": ids["motor"], "observed_ids": ids["observed"],
                           "core_ids": ids["core"], "union_positions": positions,
                           "tau": inputs["decoder"]["tau"],
                           "kappa": inputs["decoder"]["kappa"][[68, 107]],
                           "slopes": inputs["decoder"]["slopes"][44, [68, 107]],
                           "membership": inputs["decoder"]["membership"][[68, 107]]}
                raw["bank.npy"][0] = inputs["bank0"]
                raw["activation44.npy"][0] = inputs["activation0"]
                raw["bank.npy"].flush()
                raw["activation44.npy"].flush()

                def persist():
                    e248.write_json(out / "advance-ledger.json", ledger)

                def budget():
                    resource_check(freeze, began, out, ledger)

                # The observed start is saved after the arm-local exact restore.
                network.restore(restored)
                exact(network.snapshot(), restored, "E260 pre-arm exact reset")
                StateStore.save(arm_dir / "observed-start", network.snapshot())
                entry["initial_saved"] = True
                persist()
                run_arm(network, restored,
                                inputs["arms"][BINDINGS[arm][1]], incoming,
                                drive_ids, inputs["navigation_ports"],
                                inputs["descending_ids"], raw, entry, context,
                                persist, budget, reference)
                # Save the computed whole state before any end/diagonal verdict.
                StateStore.save(arm_dir / "final", network.snapshot())
                entry["final_saved"] = True
                if network.tick != CONTROLS * STEPS:
                    raise AssertionError("E260 final neural tick differs")
                entry["diagonal_exact"] = True if reference is not None else None
                entry["status"] = "valid_exact_diagonal" if reference is not None else "complete_ablation"
                persist()
                e254.flush_raw(raw)
                active_dir, active_arm = None, None
            network = None
            gc.collect()
            torch.cuda.empty_cache()
        if (sum(x["setup_executed_ticks"] for x in ledger["models"].values()) != 240
                or sum(x["setup_capture_ticks"] for x in ledger["models"].values()) != 80
                or sum(x["completed_controls"] for x in ledger["arms"].values()) != 500
                or sum(x["current_probe_completed"] for x in ledger["arms"].values()) != 505
                or not all(ledger["arms"][arm]["diagonal_exact"] for arm in ("B", "C"))):
            raise AssertionError("E260 exact diagonal/work accounting differs")
        effect_arrays, report = factors(raws, inputs["incoming"]["W0"],
                                         target_parameters["threshold"])
        np.savez(out / "factor-effects.npz", **effect_arrays)
        e248.write_json(out / "factor-windows.json", report)
        result.update(validity=VALIDITY, factor_report_available=True,
                      H260_existing_edge_attenuation=report["H260"][
                          "H260_existing_edge_attenuation"],
                      H260=report["H260"], exact_diagonals=True,
                      interpretation="fixed saved-input neural-only lesion; no body")
    except Exception as error:  # noqa: BLE001 - preserve partial scientific work
        primary = error
        errors.append({"stage": "science", "type": type(error).__name__,
                       "message": str(error), "traceback": traceback.format_exc()})
        if (network is not None and active_dir is not None
                and active_arm is not None and not (active_dir / "final").exists()):
            primary = e254.preserve_partial_state(
                network, active_dir / "partial-final", active_arm, primary)
    finally:
        for raw in raws.values():
            try:
                e254.flush_raw(raw)
            except Exception as error:  # noqa: BLE001
                if primary is None:
                    primary = error
                errors.append({"stage": "raw flush", "error": repr(error)})
        raws.clear()
        network = None
        try:
            gc.collect()
        except Exception as error:  # noqa: BLE001
            if primary is None:
                primary = error
            errors.append({"stage": "garbage collection", "error": repr(error)})
        if limiter is not None:
            try:
                limiter.restore_original_limits()
            except Exception as error:  # noqa: BLE001
                if primary is None:
                    primary = error
                errors.append({"stage": "threadpool restore", "error": repr(error)})
        if old_torch_threads is not None:
            try:
                import torch
                torch.set_num_threads(old_torch_threads)
            except Exception as error:  # noqa: BLE001
                if primary is None:
                    primary = error
                errors.append({"stage": "torch thread restore", "error": repr(error)})
        for name, old in old_environment.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old
        primary = finish_receipts(out, freeze, ledger, result, errors, primary, began)
        if primary is not None:
            raise RuntimeError("E260 invalid: " + str(primary)) from primary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    freeze = gate(args.freeze, args.freeze_sha256,
                  args.review, args.review_sha256, args.out)
    execute(freeze, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
