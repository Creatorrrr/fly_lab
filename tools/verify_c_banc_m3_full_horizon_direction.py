"""E323: one audited, conditional-input, 300-control neural direction check.

The command accepts only a root-approved, file-pinned freeze. Import and gate do
not construct a model, open an NPZ payload, or initialize CUDA. No producer run,
body, optimizer, or retry is part of this diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = "verification/banc-natural-gait-m3-20260919"
SCHEMA = "flylab.m3.e323.full-horizon-direction.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e323.full-horizon-direction.review.v1"
RESULT_SCHEMA = "flylab.m3.e323.full-horizon-direction.result.v1"
OUTPUT = BASE + "/e323-full-horizon-direction"
E318 = BASE + "/e318-full-drive-capture"
N, EDGES, SELECTED, GROUPS, CONTROLS = 158706, 11584852, 725849, 8547, 300
LIMITS = {
    "supervisor_wall_s": 480, "internal_wall_s": 450, "science_wall_s": 410,
    "finalization_reserve_s": 40, "host_bytes": 6 << 30,
    "additional_gpu_bytes": 8 << 30, "output_bytes": 2 << 30,
    "minimum_free_disk_bytes": 4 << 30, "forward_neural_ticks": 18000,
    "recompute_neural_ticks": 6000, "backward_block_ticks": 6000,
    "body_calls": 0, "optimizer_steps": 0, "retry_count": 0,
}
OWNED = ("tools/verify_c_banc_m3_full_horizon_direction.py",
         "tests/test_c_banc_full_horizon_direction.py")
REQUIRED_SOURCES = (*OWNED, "tools/verify_c_banc_m3_native_adjoint.py",
                    "tools/banc_m3_grouped_current_adjoint.py",
                    "tools/banc_m3_ssp_adjoint.py",
                    "tools/banc_m3_checkpointed_rollout.py",
                    "tools/banc_m3_monotone_adjoint.py",
                    "tools/banc_m3_phase_free_shape.py",
                    "tools/fit_c_banc_m3_native_all_leg.py",
                    "tools/verify_c_banc_m3_full_drive_capture.py")
REQUIRED_PATHS = {
    "e315_freeze": BASE + "/e315-native-adjoint-probe-freeze.json",
    "e318_freeze": BASE + "/e318-full-drive-capture-freeze.json",
    "e318_review": BASE + "/e318-full-drive-capture-source-review.json",
    "e318_result": E318 + "/result.json",
    "e318_audit": BASE + "/parent-e318-original-raw-audit.json",
    "e318_source_manifest": E318 + "/source-manifest.json",
    "e318_output_manifest": E318 + "/output-manifest.json",
    "e317_review": BASE + "/parent-e317-second-review.json",
    "e319_review": BASE + "/parent-e319-second-review.json",
    "e317_qualification": BASE + "/parent-e317-generated-qualification.json",
    "e319_qualification": BASE + "/parent-e319-generated-qualification.json",
    "e317_cuda": BASE + "/parent-e317-independent-cuda.json",
    "e319_cuda": BASE + "/parent-e319-cuda.json",
    "e85_reference": BASE + "/e85-space-compatible-cadence-reference/commands.npz",
    "e103_startup": BASE + "/e103-startup-preserving-recruitment/startup-inputs.npz",
    "e150_calibration": BASE + "/e150-native-monotone-body/assembled-calibration.npz",
    "e150_parameters": BASE + "/e150-fixed-monotone-parameters.npz",
    "e318_start": E318 + "/observed-start",
    "e318_final": E318 + "/whole-300",
    "e318_raw": E318 + "/control-raw.npz",
    "e318_chunks": E318 + "/full-drive-chunks",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(relative: str, root: Path = ROOT) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Repo-relative path required")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path escapes repository")
    return path


def _file_roster(paths: dict, root: Path) -> set[str]:
    files = set()
    for name in paths.values():
        path = safe(name, root)
        if path.is_dir():
            contents = [p for p in path.rglob("*") if p.is_file()]
            if not contents:
                raise ValueError("Empty pinned input directory: " + name)
        elif path.is_file():
            contents = [path]
        else:
            raise ValueError("Missing pinned input: " + name)
        files.update(p.relative_to(root).as_posix() for p in contents)
    return files


def gate(freeze_path: Path, freeze_sha256: str, review_path: Path,
         review_sha256: str, output: Path, *, root: Path = ROOT) -> dict:
    """Read only JSON and file hashes; all heavy dependencies stay unimported."""
    if output.exists() or output.resolve() != safe(OUTPUT, root):
        raise ValueError("Fresh literal E323 output required")
    if sha256(freeze_path) != freeze_sha256 or sha256(review_path) != review_sha256:
        raise ValueError("E323 freeze/review hash differs")
    freeze = json.loads(freeze_path.read_text("utf8"))
    review = json.loads(review_path.read_text("utf8"))
    if (freeze.get("schema") != SCHEMA or freeze.get("paths") != REQUIRED_PATHS
            or freeze.get("limits") != LIMITS or freeze.get("output") != OUTPUT
            or freeze.get("candidate_adopted") is not False):
        raise ValueError("Exact E323 frozen diagnostic contract required")
    pins, sources = freeze.get("input_sha256"), freeze.get("source_sha256")
    if not isinstance(pins, dict) or not _file_roster(REQUIRED_PATHS, root).issubset(pins):
        raise ValueError("Complete E323 input pin roster required")
    if not isinstance(sources, dict) or not set(REQUIRED_SOURCES).issubset(sources):
        raise ValueError("Complete E323 source closure required")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha256
            or review.get("source_sha256") != sources):
        raise ValueError("Matching root-approved E323 review required")
    if any(name in pins and pins[name] != digest for name, digest in sources.items()):
        raise ValueError("Contradictory source/input pin")

    def qualified_digest(name):
        return sources[name] if name in sources else pins.get(name)

    for name, digest in {**pins, **sources}.items():
        if not isinstance(digest, str) or sha256(safe(name, root)) != digest:
            raise ValueError("Frozen source/input differs: " + name)
    # The E318 producer result and its independent root audit must both exist.
    result = json.loads(safe(REQUIRED_PATHS["e318_result"], root).read_text("utf8"))
    audit = json.loads(safe(REQUIRED_PATHS["e318_audit"], root).read_text("utf8"))
    manifest = json.loads(safe(REQUIRED_PATHS["e318_output_manifest"], root).read_text("utf8"))
    producer = json.loads(safe(REQUIRED_PATHS["e318_source_manifest"], root).read_text("utf8"))
    e318_freeze = json.loads(safe(REQUIRED_PATHS["e318_freeze"], root).read_text("utf8"))
    e318_review = json.loads(safe(REQUIRED_PATHS["e318_review"], root).read_text("utf8"))
    e315_freeze = json.loads(safe(REQUIRED_PATHS["e315_freeze"], root).read_text("utf8"))
    if (result.get("schema") != "flylab.m3.e318.full-drive-capture.result.v1"
            or result.get("run_valid") is not True
            or result.get("learning_input_approved") is not True
            or audit.get("pass") is not True
            or audit.get("diagnostic_run_valid") is not True
            or audit.get("learning_input_approved") is not True
            or audit.get("result_sha256") != pins[REQUIRED_PATHS["e318_result"]]
            or audit.get("output_manifest_sha256") != pins[REQUIRED_PATHS["e318_output_manifest"]]
            or manifest.get("schema") != "flylab.m3.e318.output-manifest.v1"
            or e318_freeze.get("schema") != "flylab.m3.e318.full-drive-capture.freeze.v1"
            or e318_review.get("schema") != "flylab.m3.e318.full-drive-capture.review.v1"
            or e318_review.get("approved") is not True
            or e318_review.get("blocking_defects") != []
            or e318_review.get("freeze_sha256") != pins[REQUIRED_PATHS["e318_freeze"]]
            or e318_review.get("source_sha256") != producer.get("source_sha256")
            or producer.get("freeze_sha256") != pins[REQUIRED_PATHS["e318_freeze"]]
            or producer.get("review_sha256") != pins[REQUIRED_PATHS["e318_review"]]
            or e318_freeze.get("source_sha256") != producer.get("source_sha256")
            or e315_freeze.get("schema") != "flylab.m3.e315.native-adjoint-probe.freeze.v1"
            or producer.get("source_sha256") !=
            {key: sources.get(key) for key in producer.get("source_sha256", {})}):
        raise ValueError("Root-audited E318 complete dataset required")
    for name in pins:
        if name.startswith(E318 + "/") and name not in (REQUIRED_PATHS["e318_result"],
                REQUIRED_PATHS["e318_output_manifest"]):
            relative = name.removeprefix(E318 + "/")
            if manifest.get("files", {}).get(relative) != {
                "sha256": pins[name], "bytes": safe(name, root).stat().st_size}:
                raise ValueError("E318 producer manifest differs: " + name)
    for alias, schema in (("e317_review", "flylab.m3.e317.root-second-review.v1"),
                          ("e319_review", "flylab.m3.e319.root-second-review.v1")):
        qualification = json.loads(safe(REQUIRED_PATHS[alias], root).read_text("utf8"))
        if (qualification.get("schema") != schema or qualification.get("approved") is not True
                or qualification.get("blocking_defects") != []
                or not set(qualification.get("source_sha256", {})).issubset(sources)
                or any(qualified_digest(name) != digest for name, digest in
                       qualification.get("source_sha256", {}).items())):
            raise ValueError("Qualified E317/E319 generated sources required: " + alias)
    for alias in ("e317_cuda", "e319_cuda"):
        qualification = json.loads(safe(REQUIRED_PATHS[alias], root).read_text("utf8"))
        if qualification.get("passed") is not True:
            raise ValueError("Generated CUDA qualification required: " + alias)
    for alias, schema in (("e317_qualification", "flylab.m3.e317.generated-qualification.v1"),
                          ("e319_qualification", "flylab.m3.e319.generated-qualification.v1")):
        qualification = json.loads(safe(REQUIRED_PATHS[alias], root).read_text("utf8"))
        if (qualification.get("schema") != schema or qualification.get("approved") is not True
                or qualification.get("original_calls") != 0
                or qualification.get("blocking_defects") != []
                or any(qualified_digest(name) != digest for name, digest in
                       qualification.get("source_sha256", {}).items())
                or any(qualified_digest(name) != digest for name, digest in
                       qualification.get("evidence_sha256", {}).items())):
            raise ValueError("Approved generated qualification required: " + alias)
    qualifications = [json.loads(safe(REQUIRED_PATHS[key], root).read_text("utf8"))
                      for key in ("e317_qualification", "e319_qualification")]
    required = _file_roster(REQUIRED_PATHS, root)
    for qualification in qualifications:
        required.update(name for name in qualification.get("source_sha256", {})
                        if name not in sources)
        required.update(name for name in qualification.get("evidence_sha256", {})
                        if name not in sources)
    for name, digest in e318_freeze.get("input_sha256", {}).items():
        if pins.get(name) != digest:
            raise ValueError("E318 frozen producer input differs: " + name)
        required.add(name)
    for name, digest in e315_freeze.get("source_sha256", {}).items():
        if sources.get(name) != digest:
            raise ValueError("E315 frozen source differs: " + name)
    qualification = e315_freeze.get("qualification", {})
    if (not isinstance(qualification, dict) or not isinstance(qualification.get("path"), str)
            or pins.get(qualification["path"]) != qualification.get("sha256")):
        raise ValueError("Pinned E315 generated qualification required")
    required.add(qualification["path"])
    e315_qualification = json.loads(safe(qualification["path"], root).read_text("utf8"))
    if (e315_qualification.get("schema") !=
            "flylab.m3.e314.native-adjoint-generated-qualification.v1"
            or e315_qualification.get("approved") is not True
            or any(qualified_digest(name) != digest for name, digest in
                   e315_qualification.get("evidence_sha256", {}).items())):
        raise ValueError("Approved E314/E315 generated evidence required")
    required.update(name for name in e315_qualification.get("evidence_sha256", {})
                    if name not in sources)
    for descriptor in e315_freeze.get("producer_reviews", []):
        if pins.get(descriptor.get("path")) != descriptor.get("sha256"):
            raise ValueError("E315 producer review pin differs")
        required.add(descriptor["path"])
    for descriptor in e315_freeze.get("inputs", {}).values():
        if "sha256" in descriptor:
            if pins.get(descriptor["path"]) != descriptor["sha256"]:
                raise ValueError("E315 frozen input differs")
            required.add(descriptor["path"])
        else:
            folder = safe(descriptor["path"], root)
            for name in ("manifest.json", "state.json"):
                relative = (folder / name).relative_to(root).as_posix()
                if pins.get(relative) != descriptor[name.removesuffix(".json") + "_sha256"]:
                    raise ValueError("E315 state metadata differs")
            required.update(_file_roster({"state": descriptor["path"]}, root))
    if set(pins) != required:
        raise ValueError("Exact qualification/input pin roster required")
    if shutil.disk_usage(output.parent).free < LIMITS["minimum_free_disk_bytes"]:
        raise OSError("E323 free disk reserve unavailable")
    return freeze


def exact(left, right) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def curve128(reference: dict) -> np.ndarray:
    """The fixed E107 periodic LF column-0 interpolation, sampled at M=128."""
    if (reference["time_s"].dtype != np.float64 or reference["command"].dtype != np.float64
            or reference["phase"].dtype != np.float64
            or reference["command"].shape != (2000, 42)
            or reference["phase"].shape != (2000, 6)
            or reference["time_s"].shape != (2000,)):
        raise ValueError("Exact E85 42-joint phase reference required")
    late = reference["time_s"] >= 5
    if not late.any():
        raise ValueError("E85 late reference is empty")
    coordinates = np.mod(reference["phase"][late, 0] / (2 * np.pi), 1)
    values = reference["command"][late]
    order = np.argsort(coordinates)
    x, y = coordinates[order], values[order]
    extended_x = np.r_[x[-1] - 1, x, x[0] + 1]
    extended_y = np.vstack((y[-1], y, y[0]))
    query = np.arange(128, dtype=np.float64) / 128
    rows = np.column_stack([np.interp(query, extended_x, extended_y[:, j])
                            for j in range(42)])
    return np.ascontiguousarray(rows.reshape(128, 6, 7).transpose(1, 0, 2))


def classify(observed: float, predicted: float) -> dict:
    finite = bool(np.isfinite(observed) and np.isfinite(predicted))
    meaningful = finite and abs(observed) > 1e-9 and abs(predicted) > 1e-9
    relative = abs(observed - predicted) / abs(predicted) if meaningful else None
    status = ("supported" if meaningful and np.sign(observed) == np.sign(predicted)
              and relative <= .05 else "refuted" if meaningful else "inconclusive")
    return {"classification": status, "observed": observed if np.isfinite(observed) else None,
            "predicted": predicted if np.isfinite(predicted) else None,
            "relative_error": relative}


def usage() -> dict:
    import psutil
    import torch

    # E321 reproduces Windows cuBLAS failures when CuPy is imported first.
    # This meter needs only Torch; initialize its CUDA runtime before loaders.
    memory = psutil.Process().memory_info()
    free, total = torch.cuda.mem_get_info()
    return {"host_rss_bytes": memory.rss,
            "host_peak_bytes": getattr(memory, "peak_wset", memory.rss),
            "device_free_bytes": free, "device_total_bytes": total,
            "torch_reserved_bytes": torch.cuda.memory_reserved(),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved()}


class Dependencies:
    """Actual loaders and model constructors, reached only after gate()."""

    def load(self, freeze: dict, root: Path = ROOT) -> dict:
        from flylab.c.storage import StateStore
        from tools.verify_c_banc_m3_native_adjoint import load_inputs, load_npz

        paths = {key: safe(value, root) for key, value in freeze["paths"].items()}
        e315_freeze = json.loads(paths["e315_freeze"].read_text("utf8"))
        native = load_inputs(e315_freeze, root=root)
        start, final = StateStore.load(paths["e318_start"]), StateStore.load(paths["e318_final"])
        if (start["control_tick"] != 0 or start["network"]["tick"] != 0
                or final["control_tick"] != CONTROLS
                or final["network"]["tick"] != CONTROLS * 20
                or start["network"]["model_hash"] != native["network_identity"]
                or final["network"]["model_hash"] != native["network_identity"]
                or start["adapter"]["hash"] != final["adapter"]["hash"]
                or not exact(start["network"]["rate"], native["states"]["initial_neural"]["rate"])
                or not exact(start["network"]["adaptation"], native["states"]["initial_neural"]["adaptation"])
                or not exact(start["network"]["output_mask"],
                             native["states"]["initial_neural"]["output_mask"])):
            raise ValueError("E318 control0/E315 initial or control300 identity differs")
        chunks = sorted(paths["e318_chunks"].glob("full-drive-*.npz"))
        if len(chunks) != 12:
            raise ValueError("Twelve original E318 full-drive chunks required")
        arrays = [load_npz(path) for path in chunks]
        ticks = np.concatenate([row["control_tick"] for row in arrays])
        network_ticks = np.concatenate([row["network_tick"] for row in arrays])
        drives = np.concatenate([row["drive"] for row in arrays])
        rates = np.concatenate([row["rate"] for row in arrays])
        adaptations = np.concatenate([row["adaptation"] for row in arrays])
        motor_rates = np.concatenate([row["motor_rate"] for row in arrays])
        masks = np.concatenate([row["motor_mask"] for row in arrays])
        motor_ids = load_npz(paths["e150_calibration"])["motor_ids"]
        if (not np.array_equal(ticks, np.arange(1, CONTROLS + 1))
                or not np.array_equal(network_ticks, ticks * 20)
                or drives.dtype != np.float32 or drives.shape != (CONTROLS, N)
                or rates.dtype != np.float32 or rates.shape != (CONTROLS, N)
                or adaptations.dtype != np.float32 or adaptations.shape != (CONTROLS, N)
                or motor_rates.shape != (CONTROLS, 391)
                or masks.shape != (CONTROLS, 391)
                or not exact(motor_rates, rates[:, motor_ids])
                or not exact(masks, np.broadcast_to(start["network"]["output_mask"][motor_ids], masks.shape))
                or not np.isfinite(drives).all()):
            raise ValueError("E318 full held-drive/endpoints/control order invalid")
        if (not exact(rates[-1], final["network"]["rate"])
                or not exact(adaptations[-1], final["network"]["adaptation"])):
            raise ValueError("E318 final neural endpoint differs from final whole")
        raw = load_npz(paths["e318_raw"])
        if (not exact(raw["control_tick"], np.arange(CONTROLS + 1, dtype=np.int64))
                or not exact(raw["network_tick"], np.arange(CONTROLS + 1, dtype=np.int64) * 20)
                or raw["command"].shape != (CONTROLS + 1, 42)
                or raw["tendon"].shape != (CONTROLS + 1, 6)):
            raise ValueError("E318 raw command/LTM clock invalid")
        reference = load_npz(paths["e85_reference"])
        startup = load_npz(paths["e103_startup"])
        if (startup["target_command"].dtype != np.float64
                or startup["target_command"].shape != (99, 42)
                or startup["target_tendon"].dtype != np.float64
                or startup["target_tendon"].shape != (99, 6)):
            raise ValueError("Exact E103 startup99 targets required")
        return {"native": native, "start": start, "final": final, "drives": drives,
                "raw": raw, "curve": curve128(reference), "startup": startup,
                "calibration": load_npz(paths["e150_calibration"]),
                "parameters": load_npz(paths["e150_parameters"])}

    def build(self, data: dict, device: str):
        import torch
        from scipy.sparse import csr_matrix

        from tools.banc_m3_checkpointed_rollout import CheckpointedControlRollout
        from tools.banc_m3_grouped_current_adjoint import GroupedCurrent
        from tools.banc_m3_monotone_adjoint import MonotoneDecoder
        from tools.banc_m3_phase_free_shape import PhaseFreeShapeObjective
        from tools.banc_m3_ssp_adjoint import SSPAdjointModel

        native, start = data["native"], data["start"]
        a, m, p = native["csr"], native["mask"], native["physiology"]
        csr = csr_matrix((a["data"], a["indices"], a["indptr"]), shape=(N, N))
        operator = GroupedCurrent(csr, m["positions"], m["edge_parameters"], GROUPS, device)
        neural = SSPAdjointModel(operator, p["tau"], p["coefficient"], p["threshold"],
                                 p["cap"], p["gain"], p["inverse"], .00025)
        adapter = start["adapter"]
        calibration = {key: data["calibration"][key] for key in
                       ("moment", "rest", "pool", "motor_ids", "ltm")}
        parameters = data["parameters"]
        # StateStore preserves BodySnapshot action vectors as JSON lists.
        neutral = np.asarray(start["body"]["action"]["joint_angles"])
        if (adapter["hash"] != "1fd7b86939a6846e01c4837fb5c2fbe24ce8c776e3331c1f75921a376c53b067"
                or not all(exact(adapter["calibration"][key], calibration[key])
                           for key in calibration)
                or not all(exact(adapter["parameters"][key], parameters[key])
                           for key in ("tau", "kappa", "slopes", "membership"))
                or not isinstance(neutral, np.ndarray) or neutral.dtype != np.float64
                or neutral.shape != (42,) or not np.isfinite(neutral).all()):
            raise ValueError("E318 start adapter differs from fixed E150 decoder")
        decoder = MonotoneDecoder(calibration, parameters["tau"], parameters["kappa"],
                                  parameters["slopes"], parameters["membership"],
                                  neutral, device=device)
        rollout = CheckpointedControlRollout(neural, decoder, calibration["motor_ids"],
                                             steps_per_control=20, control_dt=.005,
                                             checkpoint_controls=1, max_controls=CONTROLS)
        objective = PhaseFreeShapeObjective(data["curve"],
                                            data["startup"]["target_command"],
                                            data["startup"]["target_tendon"], device=device)
        tensor = lambda value: torch.as_tensor(np.ascontiguousarray(value), device=device)
        initial = (tensor(start["network"]["rate"]), tensor(start["network"]["adaptation"]),
                   tensor(adapter["bank"]), tensor(adapter["original"]["activation"]),
                   tensor(data["drives"]), tensor(start["network"]["output_mask"]))
        return rollout, objective, initial


def _host(value):
    return value.detach().cpu().numpy().copy()


def _receipt(value) -> dict:
    keys = ("status", "forward_controls_requested", "forward_controls_returned",
            "forward_neural_ticks_requested", "forward_neural_ticks_returned",
            "recompute_controls_requested", "recompute_controls_returned",
            "recompute_neural_ticks_requested", "recompute_neural_ticks_returned",
            "backward_blocks_requested", "backward_blocks_returned",
            "backward_block_ticks_returned", "confirm_requested", "confirm_returned",
            "confirmed")
    return {key: getattr(value, key) for key in keys}


def _metrics(value: dict) -> dict:
    return {key: float(_host(item)) if key != "startup_pass" else bool(_host(item))
            for key, item in value.items() if key != "per_leg_shape_loss"} | {
            "per_leg_shape_loss": _host(value["per_leg_shape_loss"]).tolist()}


def _compare_neutral(output: dict, data: dict) -> dict:
    q_value, tendon_value = (_host(output[key]) for key in ("q", "tendon"))
    raw = data["raw"]
    rate, adaptation = _host(output["rate"]), _host(output["adaptation"])
    bank, original = _host(output["bank"]), _host(output["original_activation"])
    final = data["final"]
    specifications = (
        (q_value, (CONTROLS, 42), np.float64, "q"),
        (tendon_value, (CONTROLS, 6), np.float64, "tendon"),
        (raw["command"], (CONTROLS + 1, 42), np.float64, "E318 q"),
        (raw["tendon"], (CONTROLS + 1, 6), np.float64, "E318 tendon"),
        (rate, (N,), np.float32, "rate"),
        (adaptation, (N,), np.float32, "adaptation"),
        (final["network"]["rate"], (N,), np.float32, "E318 rate"),
        (final["network"]["adaptation"], (N,), np.float32, "E318 adaptation"),
        (bank, (391, 3), np.float64, "bank"),
        (original, (96,), np.float64, "original"),
        (final["adapter"]["bank"], (391, 3), np.float64, "E318 bank"),
        (final["adapter"]["original"]["activation"], (96,), np.float64,
         "E318 original"),
    )
    for value, shape, dtype, label in specifications:
        if (not isinstance(value, np.ndarray) or value.shape != shape
                or value.dtype != dtype or not np.isfinite(value).all()):
            raise ValueError("Finite exact neutral output/reference required: " + label)
    q_error = float(np.max(np.abs(q_value - raw["command"][1:])))
    tendon_error = float(np.max(np.abs(tendon_value - raw["tendon"][1:])))
    bank_error = float(np.max(np.abs(bank - final["adapter"]["bank"])))
    original_error = float(np.max(np.abs(original - final["adapter"]["original"]["activation"])))
    endpoint_exact = exact(rate, final["network"]["rate"]) and exact(
        adaptation, final["network"]["adaptation"])
    result = {"q_max_abs_rad": q_error, "tendon_max_abs_command": tendon_error,
              "final_rate_adaptation_dtype_bytes_exact": endpoint_exact,
              "final_bank_max_abs": bank_error,
              "final_original_max_abs": original_error,
              "control300_tick": final["control_tick"],
              "network_tick": final["network"]["tick"]}
    if (q_error > 1e-11 or tendon_error > 1e-11 or not endpoint_exact
            or bank_error > 1e-10 or original_error > 1e-10):
        raise ValueError("E323 neutral full-trace/native endpoint reproduction failed")
    return result


def compute(data: dict, *, dependencies=None, device="cuda", checkpoint=None,
            deadline=None, on_receipt=None, on_progress=None,
            on_boundary=None) -> tuple[dict, dict]:
    """Run one neutral backward and exactly one plus/minus pair from control0."""
    import torch

    dependencies = Dependencies() if dependencies is None else dependencies
    save = (lambda _name, _arrays: None) if checkpoint is None else checkpoint
    timely = (lambda: None) if deadline is None else deadline
    report_receipt = (lambda _name, _receipt: None) if on_receipt is None else on_receipt
    report_progress = (lambda _value: None) if on_progress is None else on_progress
    boundary = (lambda _name, _event: None) if on_boundary is None else on_boundary
    timely()
    boundary("construct", "requested")
    try:
        rollout, objective, initial = dependencies.build(data, device)
    except BaseException as error:
        try:
            boundary("construct", "failed")
        except Exception as secondary:  # noqa: BLE001 - preserve the constructor failure
            error.add_note("Construction journal failed: " + repr(secondary))
        raise
    boundary("construct", "host_returned")
    r, z, bank, original, drives, mask = initial
    theta = torch.ones(GROUPS, dtype=torch.float64, device=rollout.device,
                       requires_grad=True)
    result = {"conditional_fixed_neural_input": True, "full_controls": CONTROLS,
              "steps_per_control": 20, "checkpoint_controls": 1,
              "direction_count": 1, "body_calls": 0, "optimizer_steps": 0,
              "runs": {}}
    receipts = {}

    def perform(name, efficacy, backward=False):
        timely()
        receipt = None
        active = name + "_forward"
        try:
            boundary(active, "requested")
            output = rollout.run(r, z, bank, original, drives, efficacy, mask)
            receipt = output["receipt"]
            boundary(active, "host_returned")
            receipts[name] = _receipt(receipt)
            report_receipt(name, receipts[name])
            # Persist all forward raw and its host-returned receipt before loss/backward.
            arrays = {key: _host(output[key]) for key in
                      ("q", "tendon", "rate", "adaptation", "bank", "original_activation")}
            save(name, arrays)
            receipt.confirm()
            receipts[name] = _receipt(receipt)
            report_receipt(name, receipts[name])
            boundary(active, "device_confirmed")
            active = None
            scores = objective.evaluate(output["q"], output["tendon"])
            if backward:
                reproduction = _compare_neutral(output, data)
                active = name + "_backward"
                timely()
                boundary(active, "requested")
                scores["shape_loss"].backward()
                boundary(active, "host_returned")
                receipt.confirm()
                receipts[name] = _receipt(receipt)
                report_receipt(name, receipts[name])
                boundary(active, "device_confirmed")
                active = None
                gradient = _host(efficacy.grad)
                save("gradient", {"gradient": gradient})
            else:
                gradient = None
                reproduction = None
            result["runs"][name] = {"metrics": _metrics(scores),
                                    "receipt": _receipt(receipt),
                                    "raw_file": name + ".npz"}
            if reproduction is not None:
                result["neutral_reproduction"] = reproduction
            receipts[name] = _receipt(receipt)
            report_receipt(name, receipts[name])
            report_progress(result.copy())
            timely()
            return result["runs"][name]["metrics"], gradient
        except BaseException as error:
            if active is not None:
                try:
                    boundary(active, "failed")
                except Exception as secondary:  # noqa: BLE001 - keep original error
                    error.add_note("Phase journal failed: " + repr(secondary))
            if receipt is None:
                receipt = getattr(error, "receipt", None)
            if receipt is not None:
                try:
                    receipt.confirm()
                except Exception as secondary:  # noqa: BLE001 - retain original failure
                    error.add_note("Receipt confirmation failed: " + repr(secondary))
                finally:
                    receipts[name] = _receipt(receipt)
                    report_receipt(name, receipts[name])
            raise

    neutral, gradient = perform("neutral", theta, backward=True)
    if gradient.shape != (GROUPS,) or not np.isfinite(gradient).all() or not np.any(gradient):
        raise ValueError("E323 gradient zero/nonfinite: direction cannot be tested")
    direction = np.sign(gradient)
    minus = np.float32(1 - .001 * direction).astype(np.float64)
    plus = np.float32(1 + .001 * direction).astype(np.float64)
    if not np.any(minus != 1) or not np.any(plus != 1):
        raise ValueError("E323 f32 efficacy perturbation vanished")
    save("direction", {"gradient": gradient, "direction": direction,
                       "theta_plus": plus, "theta_minus": minus})
    selected = data["native"]["csr"]["data"][data["native"]["mask"]["positions"]]
    groups = data["native"]["mask"]["edge_parameters"]
    plus_weights = np.float32(selected.astype(np.float64) * plus[groups])
    minus_weights = np.float32(selected.astype(np.float64) * minus[groups])
    plus_delta = plus_weights.astype(np.float64) - selected.astype(np.float64)
    minus_delta = minus_weights.astype(np.float64) - selected.astype(np.float64)
    result["direction"] = {"nonzero_gradient_groups": int(np.count_nonzero(direction)),
                           "gradient_min": float(gradient.min()),
                           "gradient_max": float(gradient.max()),
                           "theta_plus_min": float(plus.min()),
                           "theta_plus_max": float(plus.max()),
                           "theta_minus_min": float(minus.min()),
                           "theta_minus_max": float(minus.max()),
                           "selected_weight_changed_plus_count": int(np.count_nonzero(plus_delta)),
                           "selected_weight_changed_minus_count": int(np.count_nonzero(minus_delta)),
                           "selected_weight_plus_delta_min": float(plus_delta.min()),
                           "selected_weight_plus_delta_max": float(plus_delta.max()),
                           "selected_weight_minus_delta_min": float(minus_delta.min()),
                           "selected_weight_minus_delta_max": float(minus_delta.max())}
    report_progress(result.copy())
    plus_metrics, _ = perform("plus", torch.as_tensor(plus.copy(), device=rollout.device))
    minus_metrics, _ = perform("minus", torch.as_tensor(minus.copy(), device=rollout.device))
    observed = plus_metrics["shape_loss"] - minus_metrics["shape_loss"]
    predicted = float(np.sum(gradient * (plus - minus), dtype=np.float64))
    result["comparison"] = classify(observed, predicted)
    result["minus_improves_neutral"] = minus_metrics["shape_loss"] < neutral["shape_loss"]
    result["minus_startup_pass"] = minus_metrics["startup_pass"]
    if (receipts["neutral"]["confirmed"] != {
            "forward_controls": 300, "forward_neural_ticks": 6000,
            "recompute_controls": 300, "recompute_neural_ticks": 6000,
            "backward_blocks": 300, "backward_block_ticks": 6000}
            or any(receipts[name]["confirmed"] != {
                "forward_controls": 300, "forward_neural_ticks": 6000,
                "recompute_controls": 0, "recompute_neural_ticks": 0,
                "backward_blocks": 0, "backward_block_ticks": 0}
                for name in ("plus", "minus"))):
        raise ValueError("E323 confirmed neural-only tick/block accounting differs")
    return result, receipts


def put_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def run(freeze_path: Path, freeze_sha256: str, review_path: Path, review_sha256: str,
        output: Path, *, root: Path = ROOT, gatekeeper=gate, dependencies=None,
        clock=time.perf_counter, writer=put_json, saver=np.savez_compressed,
        meter=usage, device="cuda") -> dict:
    """Passively gate first; thereafter journal every attempt and partial output."""
    began = clock()
    freeze = gatekeeper(freeze_path, freeze_sha256, review_path, review_sha256,
                        output, root=root)
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema": RESULT_SCHEMA, "diagnostic_id": "E323", "run_valid": False,
              "candidate_adopted": False, "biological_gait_proven": False,
              "F1_F8_complete": False, "F6_exposed": False, "M2_modified": False,
              "classification": "not_tested", "conditional_fixed_neural_input": True,
              "provenance": {"freeze_sha256": freeze_sha256,
                             "review_sha256": review_sha256,
                             "source_manifest": "source-manifest.json"},
              "publication": {"measured_boundary": "before_terminal_metadata_commit",
                              "external_audit_pins_required": ["result.json",
                                                               "advance-ledger.json"],
                              "metadata_excluded_from_manifest": True},
              "errors": []}
    ledger = {"limits": LIMITS.copy(), "retry_count": 0, "body_calls": 0,
              "optimizer_steps": 0, "attempts": [], "phases": {},
              "receipts": {}, "errors": []}
    first_error = None

    def failure(stage, error):
        nonlocal first_error
        entry = {"stage": stage, "type": type(error).__name__, "message": str(error)[:500]}
        ledger["errors"].append(entry)
        if first_error is None:
            first_error = entry
        return entry

    def persist():
        ledger["elapsed_s"] = clock() - began
        writer(output / "advance-ledger.json", ledger)

    def check_resource(value):
        if value["device_total_bytes"] != baseline["device_total_bytes"]:
            raise RuntimeError("E323 device identity changed")
        ledger["resource_latest"] = value
        ledger["additional_gpu_peak_bytes"] = max(
            ledger.get("additional_gpu_peak_bytes", 0),
            baseline["device_free_bytes"] - value["device_free_bytes"])
        ledger["additional_torch_peak_reserved_bytes"] = max(
            ledger.get("additional_torch_peak_reserved_bytes", 0),
            value["torch_peak_reserved_bytes"] - baseline["torch_reserved_bytes"])
        if (value["host_peak_bytes"] > LIMITS["host_bytes"]
                or ledger["additional_gpu_peak_bytes"] > LIMITS["additional_gpu_bytes"]
                or ledger["additional_torch_peak_reserved_bytes"]
                > LIMITS["additional_gpu_bytes"]):
            raise MemoryError("E323 host/device memory limit exceeded")

    def deadline():
        if clock() - began >= LIMITS["internal_wall_s"]:
            raise TimeoutError("E323 internal deadline exceeded")
        if clock() - science_began >= LIMITS["science_wall_s"]:
            raise TimeoutError("E323 science deadline exceeded")

    def checkpoint(name, arrays):
        path = output / (name + ".npz")
        ledger["attempts"].append({"stage": "save_" + name, "requested": True,
                                   "returned": False})
        persist()
        saver(path, **arrays)
        ledger["attempts"][-1]["returned"] = True
        ledger["attempts"][-1]["bytes"] = path.stat().st_size
        persist()

    science_began = clock()
    try:
        persist()
        baseline = meter()
        ledger["resource_baseline"] = baseline
        writer(output / "source-manifest.json", {
            "schema": "flylab.m3.e323.source-manifest.v1",
            "freeze_sha256": freeze_sha256, "review_sha256": review_sha256,
            "source_sha256": freeze["source_sha256"],
            "input_sha256": freeze["input_sha256"]})
        deps = Dependencies() if dependencies is None else dependencies
        data = deps.load(freeze, root=root)
        deadline()
        check_resource(meter())
        def on_receipt(name, value):
            ledger["receipts"][name] = value
            persist()

        def on_boundary(name, event):
            phases = ledger["phases"]
            if event == "requested":
                if name in phases:
                    raise ValueError("E323 phase was requested twice: " + name)
                phases[name] = {"requested": True, "host_returned": None,
                                "device_confirmed": None, "failed": False,
                                "requested_elapsed_s": clock() - began}
            elif event == "host_returned":
                phases[name]["host_returned"] = True
            elif event == "device_confirmed":
                phases[name]["device_confirmed"] = True
            elif event == "failed":
                phases[name]["failed"] = True
            else:
                raise ValueError("Unknown E323 phase event")
            phases[name]["last_event_elapsed_s"] = clock() - began
            persist()
            if event == "host_returned" and name == "construct" or event == "device_confirmed":
                check_resource(meter())
                deadline()

        def on_progress(value):
            result["partial"] = value
            persist()

        computed, receipts = compute(data, dependencies=deps, checkpoint=checkpoint,
                                     deadline=deadline, on_receipt=on_receipt,
                                     on_progress=on_progress,
                                     on_boundary=on_boundary, device=device)
        deadline()
        ledger["receipts"] = receipts
        result.update(computed)
        result["classification"] = computed["comparison"]["classification"]
        check_resource(meter())
    except Exception as error:  # noqa: BLE001 - diagnostic must retain partial cost
        failure("science", error)
        result["science_error"] = traceback.format_exc(limit=8)[-3000:]
    finally:
        ledger["science_wall_s"] = clock() - science_began
        finalization_began = clock()
        try:
            check_resource(meter())
        except Exception as error:  # noqa: BLE001
            failure("terminal_resources", error)
        result["first_error"] = first_error
        result["errors"] = ledger["errors"].copy()
        result["cost"] = ledger.copy()
        for name, value in (("advance-ledger.json", ledger), ("result.json", result)):
            try:
                writer(output / name, value)
            except Exception as error:  # noqa: BLE001
                failure("publish_" + name, error)
        try:
            total_bytes = sum(path.stat().st_size for path in output.rglob("*")
                              if path.is_file() and path.name != "output-manifest.json")
            files = {path.relative_to(output).as_posix(): {
                "sha256": sha256(path), "bytes": path.stat().st_size}
                for path in output.rglob("*") if path.is_file()
                and path.name not in ("output-manifest.json", "advance-ledger.json",
                                      "result.json", "publication-failure.json")}
            if total_bytes > LIMITS["output_bytes"]:
                raise OSError("E323 output size limit exceeded")
            writer(output / "output-manifest.json", {
                "schema": "flylab.m3.e323.output-manifest.v1", "files": files,
                "mutable_metadata_excluded": ["result.json", "advance-ledger.json",
                                              "publication-failure.json"],
                "external_audit_pins_required": True})
            if not (output / "output-manifest.json").is_file():
                raise OSError("E323 output manifest was not published")
        except Exception as error:  # noqa: BLE001
            failure("output_manifest", error)
        # Include all preliminary publication and manifest work in the bounded
        # measured interval. The final metadata write follows this explicit
        # boundary to avoid a self-referential timestamp/hash publication loop.
        ledger["publication_boundary"] = "before_terminal_metadata_commit"
        ledger["finalization_wall_s"] = clock() - finalization_began
        ledger["elapsed_s"] = clock() - began
        if ledger["finalization_wall_s"] > LIMITS["finalization_reserve_s"]:
            failure("finalization_deadline", TimeoutError("E323 finalization reserve exceeded"))
        if ledger["elapsed_s"] >= LIMITS["internal_wall_s"]:
            failure("internal_deadline", TimeoutError("E323 internal wall deadline exceeded"))
        result["first_error"] = first_error
        result["errors"] = ledger["errors"].copy()
        result["cost"] = ledger.copy()
        if first_error is None:
            result["run_valid"] = True
        else:
            result["run_valid"] = False
        try:
            writer(output / "advance-ledger.json", ledger)
            writer(output / "result.json", result)
            if (json.loads((output / "advance-ledger.json").read_text("utf8")) != ledger
                    or json.loads((output / "result.json").read_text("utf8")) != result):
                raise OSError("E323 terminal metadata differs from return value")
        except Exception as error:  # noqa: BLE001
            failure("terminal_publication", error)
            result.update(run_valid=False, first_error=first_error,
                          errors=ledger["errors"].copy(), cost=ledger.copy())
            try:
                writer(output / "publication-failure.json", result)
            except Exception as secondary:  # noqa: BLE001 - retain first failure
                failure("publication_failure_journal", secondary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("freeze-sha256", "review-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = run(args.freeze, args.freeze_sha256, args.review,
                 args.review_sha256, args.output)
    print(json.dumps(result, allow_nan=False), flush=True)
    if not result["run_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
