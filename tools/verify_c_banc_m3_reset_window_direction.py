"""E324: full steady-200 fixed-input direction test in fifty 20 ms resets.

Only an approved E324 freeze reaches the original payload/model. No training,
body, physics, branch search, source mutation, or retry occurs here.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np

from tools import verify_c_banc_m3_full_horizon_direction as e323

ROOT, BASE, E318 = e323.ROOT, e323.BASE, e323.E318
N, GROUPS, CONTROLS = e323.N, e323.GROUPS, e323.CONTROLS
SCHEMA = "flylab.m3.e324.reset-window-direction.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e324.reset-window-direction.review.v1"
RESULT_SCHEMA = "flylab.m3.e324.reset-window-direction.result.v1"
OUTPUT = BASE + "/e324-reset-window-direction"
E323 = BASE + "/e323-full-horizon-direction"
STARTS = tuple(range(100, 300, 4))
LIMITS = {"supervisor_wall_s": 240, "internal_wall_s": 220,
          "science_wall_s": 190, "finalization_reserve_s": 30,
          "host_bytes": 6 << 30, "additional_gpu_bytes": 8 << 30,
          "output_bytes": 2 << 30, "minimum_free_disk_bytes": 4 << 30,
          "forward_neural_ticks": 18000, "recompute_neural_ticks": 4000,
          "backward_block_ticks": 4000, "backward_blocks": 200,
          "decoder_replay_controls": 300, "body_calls": 0,
          "optimizer_steps": 0, "retry_count": 0}
OWNED = ("tools/verify_c_banc_m3_reset_window_direction.py",
         "tests/test_c_banc_reset_window_direction.py")
PATHS = {**e323.REQUIRED_PATHS,
         "e318_whole100": E318 + "/whole-100",
         "e323_freeze": BASE + "/e323-full-horizon-direction-freeze.json",
         "e323_review": BASE + "/e323-full-horizon-direction-source-review.json",
         "e323_qualification": BASE + "/parent-e323-generated-qualification.json",
         "e323_result": E323 + "/result.json",
         "e323_audit": BASE + "/parent-e323-original-raw-audit.json",
         "e323_source_manifest": E323 + "/source-manifest.json",
         "e323_output_manifest": E323 + "/output-manifest.json",
         "e323_ledger": E323 + "/advance-ledger.json"}


def gate(freeze_path: Path, freeze_sha256: str, review_path: Path,
         review_sha256: str, output: Path, *, root: Path = ROOT) -> dict:
    """Passive exact file/source/audit gate; no NPZ payload or CUDA import."""
    if output.exists() or output.resolve() != e323.safe(OUTPUT, root):
        raise ValueError("Fresh literal E324 output required")
    if (e323.sha256(freeze_path) != freeze_sha256
            or e323.sha256(review_path) != review_sha256):
        raise ValueError("E324 freeze/review hash differs")
    freeze = json.loads(freeze_path.read_text("utf8"))
    review = json.loads(review_path.read_text("utf8"))
    if (freeze.get("schema") != SCHEMA or freeze.get("paths") != PATHS
            or freeze.get("limits") != LIMITS or freeze.get("output") != OUTPUT
            or freeze.get("candidate_adopted") is not False):
        raise ValueError("Exact E324 diagnostic freeze required")
    pins, sources = freeze.get("input_sha256"), freeze.get("source_sha256")
    if (not isinstance(pins, dict) or not isinstance(sources, dict)
            or not set(OWNED).issubset(sources)
            or not e323._file_roster(PATHS, root).issubset(pins)
            or any(name in pins and pins[name] != digest for name, digest in sources.items())
            or review.get("schema") != REVIEW_SCHEMA
            or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha256
            or review.get("source_sha256") != sources):
        raise ValueError("Approved complete E324 source/input pins required")
    prior_freeze = json.loads(e323.safe(PATHS["e323_freeze"], root).read_text("utf8"))
    prior_review = json.loads(e323.safe(PATHS["e323_review"], root).read_text("utf8"))
    prior_qual = json.loads(e323.safe(PATHS["e323_qualification"], root).read_text("utf8"))
    prior_result = json.loads(e323.safe(PATHS["e323_result"], root).read_text("utf8"))
    prior_audit = json.loads(e323.safe(PATHS["e323_audit"], root).read_text("utf8"))
    prior_source = json.loads(e323.safe(PATHS["e323_source_manifest"], root).read_text("utf8"))
    prior_manifest = json.loads(e323.safe(PATHS["e323_output_manifest"], root).read_text("utf8"))
    e318_manifest = json.loads(e323.safe(PATHS["e318_output_manifest"], root).read_text("utf8"))
    if (prior_freeze.get("schema") != e323.SCHEMA
            or prior_freeze.get("paths") != e323.REQUIRED_PATHS
            or prior_freeze.get("source_sha256") != prior_source.get("source_sha256")
            or prior_review.get("schema") != e323.REVIEW_SCHEMA
            or prior_review.get("approved") is not True
            or prior_review.get("blocking_defects") != []
            or prior_review.get("freeze_sha256") != pins[PATHS["e323_freeze"]]
            or prior_review.get("source_sha256") != prior_freeze["source_sha256"]
            or prior_source.get("freeze_sha256") != pins[PATHS["e323_freeze"]]
            or prior_source.get("review_sha256") != pins[PATHS["e323_review"]]
            or prior_source.get("input_sha256") != prior_freeze.get("input_sha256")
            or prior_qual.get("schema") != "flylab.m3.e323.generated-qualification-transfer.v1"
            or prior_qual.get("approved") is not True
            or prior_qual.get("blocking_defects") != []
            or prior_qual.get("original_calls") != 0
            or prior_qual.get("current_source_sha256") != prior_freeze[
                "source_sha256"]["tools/verify_c_banc_m3_full_horizon_direction.py"]
            or prior_result.get("schema") != e323.RESULT_SCHEMA
            or prior_result.get("run_valid") is not True
            or prior_result.get("classification") != "refuted"
            or prior_audit.get("schema") != "flylab.m3.e323.root-original-raw-audit.v1"
            or prior_audit.get("pass") is not True
            or prior_audit.get("diagnostic_run_valid") is not True
            or prior_audit.get("verdict") != "refuted"
            or prior_audit.get("result_sha256") != pins[PATHS["e323_result"]]
            or prior_audit.get("output_manifest_sha256") != pins[PATHS["e323_output_manifest"]]
            or prior_audit.get("advance_ledger_sha256") != pins[PATHS["e323_ledger"]]
            or prior_manifest.get("schema") != "flylab.m3.e323.output-manifest.v1"
            or e318_manifest.get("schema") != "flylab.m3.e318.output-manifest.v1"):
        raise ValueError("Audited E323/E318 producer proof required")
    if not set(prior_freeze["source_sha256"]).issubset(sources):
        raise ValueError("E323 complete source closure required")
    for name, digest in prior_freeze["source_sha256"].items():
        if sources.get(name) != digest:
            raise ValueError("E323 source differs: " + name)
    for name, digest in prior_freeze["input_sha256"].items():
        if pins.get(name) != digest:
            raise ValueError("E323 input differs: " + name)
    for name, digest in {**pins, **sources}.items():
        if not isinstance(digest, str) or e323.sha256(e323.safe(name, root)) != digest:
            raise ValueError("E324 source/input differs: " + name)
    # E323 excludes terminal metadata from its manifest; independent audit pins it.
    for alias in ("e323_source_manifest",):
        path = e323.safe(PATHS[alias], root)
        name = path.relative_to(e323.safe(E323, root)).as_posix()
        if prior_manifest.get("files", {}).get(name) != {
                "sha256": pins[PATHS[alias]], "bytes": path.stat().st_size}:
            raise ValueError("E323 producer file differs")
    for path in e323.safe(PATHS["e318_whole100"], root).rglob("*"):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            name = path.relative_to(e323.safe(E318, root)).as_posix()
            if e318_manifest.get("files", {}).get(name) != {
                    "sha256": pins[relative], "bytes": path.stat().st_size}:
                raise ValueError("E318 whole100 producer file differs")
    required = e323._file_roster(PATHS, root) | set(prior_freeze["input_sha256"])
    if set(pins) != required:
        raise ValueError("Exact E324 input pin roster required")
    if shutil.disk_usage(output.parent).free < LIMITS["minimum_free_disk_bytes"]:
        raise OSError("E324 free disk reserve unavailable")
    return freeze


class Dependencies(e323.Dependencies):
    """Load the audited E318 chunks once, retaining every control boundary."""

    def load(self, freeze: dict, root: Path = ROOT) -> dict:
        from flylab.c.storage import StateStore
        from tools.verify_c_banc_m3_native_adjoint import load_inputs, load_npz

        paths = {key: e323.safe(value, root) for key, value in freeze["paths"].items()}
        native = load_inputs(json.loads(paths["e315_freeze"].read_text("utf8")), root=root)
        start = StateStore.load(paths["e318_start"])
        whole100 = StateStore.load(paths["e318_whole100"])
        final = StateStore.load(paths["e318_final"])
        if (start["control_tick"] != 0 or start["network"]["tick"] != 0
                or whole100["control_tick"] != 100 or whole100["network"]["tick"] != 2000
                or final["control_tick"] != 300 or final["network"]["tick"] != 6000
                or any(state["network"]["model_hash"] != native["network_identity"]
                       for state in (start, whole100, final))
                or any(state["adapter"]["hash"] != start["adapter"]["hash"]
                       for state in (whole100, final))
                or not e323.exact(start["network"]["rate"],
                                  native["states"]["initial_neural"]["rate"])
                or not e323.exact(start["network"]["adaptation"],
                                  native["states"]["initial_neural"]["adaptation"])):
            raise ValueError("Exact E318 neural/control origin and whole checkpoints required")
        chunks = sorted(paths["e318_chunks"].glob("full-drive-*.npz"))
        if len(chunks) != 12:
            raise ValueError("Twelve complete E318 chunks required")
        arrays = [load_npz(path) for path in chunks]
        ticks = np.concatenate([row["control_tick"] for row in arrays])
        network_ticks = np.concatenate([row["network_tick"] for row in arrays])
        drives = np.concatenate([row["drive"] for row in arrays])
        rates = np.concatenate([row["rate"] for row in arrays])
        adaptations = np.concatenate([row["adaptation"] for row in arrays])
        motors = np.concatenate([row["motor_rate"] for row in arrays])
        motor_masks = np.concatenate([row["motor_mask"] for row in arrays])
        calibration = load_npz(paths["e150_calibration"])
        mask = start["network"]["output_mask"]
        if (not e323.exact(ticks, np.arange(1, 301, dtype=np.int64))
                or not e323.exact(network_ticks, ticks * 20)
                or drives.dtype != np.float32 or drives.shape != (300, N)
                or rates.dtype != np.float32 or rates.shape != (300, N)
                or adaptations.dtype != np.float32 or adaptations.shape != (300, N)
                or motors.dtype != np.float32 or motors.shape != (300, 391)
                or mask.dtype != np.float32 or mask.shape != (N,)
                or not e323.exact(motors, rates[:, calibration["motor_ids"]])
                or not e323.exact(motor_masks,
                                  np.broadcast_to(mask[calibration["motor_ids"]], (300, 391)))
                or not all(np.isfinite(x).all() for x in (drives, rates, adaptations, motors))
                or not e323.exact(rates[99], whole100["network"]["rate"])
                or not e323.exact(adaptations[99], whole100["network"]["adaptation"])
                or not e323.exact(rates[-1], final["network"]["rate"])
                or not e323.exact(adaptations[-1], final["network"]["adaptation"])):
            raise ValueError("Complete ordered E318 controls/MN/boundaries required")
        raw = load_npz(paths["e318_raw"])
        if (not e323.exact(raw["control_tick"], np.arange(301, dtype=np.int64))
                or not e323.exact(raw["network_tick"], np.arange(301, dtype=np.int64) * 20)
                or raw["command"].dtype != np.float64 or raw["command"].shape != (301, 42)
                or raw["tendon"].dtype != np.float64 or raw["tendon"].shape != (301, 6)
                or raw["motor_rate32"].dtype != np.float32
                or raw["motor_rate32"].shape != (301, 391)
                or not e323.exact(raw["motor_rate32"],
                                  np.vstack((raw["motor_rate32"][0:1], motors)))
                or not np.isfinite(raw["command"]).all()
                or not np.isfinite(raw["tendon"]).all()):
            raise ValueError("Exact E318 raw decoder/MN clock required")
        startup = load_npz(paths["e103_startup"])
        if (startup["target_command"].dtype != np.float64
                or startup["target_command"].shape != (99, 42)
                or startup["target_tendon"].dtype != np.float64
                or startup["target_tendon"].shape != (99, 6)):
            raise ValueError("Original E103 startup99 required")
        return {"native": native, "start": start, "whole100": whole100,
                "final": final, "drives": drives, "rates": rates,
                "adaptations": adaptations, "motor_rates": motors,
                "raw": raw, "curve": e323.curve128(load_npz(paths["e85_reference"])),
                "startup": startup, "calibration": calibration,
                "parameters": load_npz(paths["e150_parameters"]),
                "e323_neutral_shape_loss": json.loads(paths["e323_result"].read_text("utf8"))
                ["runs"]["neutral"]["metrics"]["shape_loss"]}


def short_shape(q, curves):
    """Exactly E319's closed-segment geometry on four command rows."""
    import torch

    if (not isinstance(q, torch.Tensor) or q.dtype != torch.float64
            or q.shape != (4, 42) or not isinstance(curves, torch.Tensor)
            or curves.dtype != torch.float64 or curves.shape != (6, 128, 7)
            or q.device != curves.device):
        raise ValueError("Four f64 q rows and fixed M128 curves required")
    edge = torch.roll(curves, shifts=-1, dims=1) - curves
    norm = (edge * edge).sum(dim=-1)
    safe_norm = torch.where(norm > 0, norm, torch.ones_like(norm))
    x = q.reshape(4, 6, 7)
    delta = x[:, :, None, :] - curves[None, :, :, :]
    u = ((delta * edge[None]).sum(dim=-1) / safe_norm[None]).clamp(0, 1)
    nearest = curves[None] + u[..., None] * edge[None]
    per_leg = (((x[:, :, None] - nearest).square().sum(dim=-1) / 7)
               .min(dim=-1).values.mean(dim=0))
    torch._assert_async(torch.isfinite(per_leg).all(), "Finite E324 short shape required")
    return per_leg.mean(), per_leg


def _valid_array(value, shape, dtype, label):
    if (not isinstance(value, np.ndarray) or value.shape != shape
            or value.dtype != dtype or not np.isfinite(value).all()):
        raise ValueError("Finite exact E324 array required: " + label)


def _receipt(value):
    return e323._receipt(value)


def _host(value):
    return e323._host(value)


def _counters(receipts: dict, decoder_replay: int, continuation: bool,
              windows: int = 50) -> dict:
    if len(receipts) != 3 * windows + int(continuation):
        raise ValueError("E324 complete confirmed receipt roster required")
    confirmed = [row["confirmed"] for row in receipts.values()]
    if any(row is None for row in confirmed):
        raise ValueError("Unconfirmed E324 receipt")
    for name, row in receipts.items():
        if name == "continuation-minus":
            controls, backward = 300, False
        elif name.endswith("-neutral"):
            controls, backward = 4, True
        elif name.endswith(("-plus", "-minus")):
            controls, backward = 4, False
        else:
            raise ValueError("Unknown E324 receipt: " + name)
        expected_row = {"forward_controls": controls,
                        "forward_neural_ticks": 20 * controls,
                        "recompute_controls": controls if backward else 0,
                        "recompute_neural_ticks": 20 * controls if backward else 0,
                        "backward_blocks": controls if backward else 0,
                        "backward_block_ticks": 20 * controls if backward else 0}
        if row["confirmed"] != expected_row:
            raise ValueError("E324 per-run neural receipt differs: " + name)
    totals = {key: sum(row[key] for row in confirmed) for key in
              ("forward_controls", "forward_neural_ticks", "recompute_controls",
               "recompute_neural_ticks", "backward_blocks", "backward_block_ticks")}
    expected = {"forward_controls": 12 * windows + (300 if continuation else 0),
                "forward_neural_ticks": 240 * windows + (6000 if continuation else 0),
                "recompute_controls": 4 * windows,
                "recompute_neural_ticks": 80 * windows,
                "backward_blocks": 4 * windows,
                "backward_block_ticks": 80 * windows}
    if totals != expected or decoder_replay != 300:
        raise ValueError("E324 exact counted neural/decoder work differs")
    return totals | {"decoder_replay_controls": decoder_replay,
                     "body_calls": 0, "optimizer_steps": 0, "retry_count": 0}


def compute(data: dict, *, dependencies=None, device="cuda", checkpoint=None,
            deadline=None, on_receipt=None, on_progress=None, on_boundary=None,
            starts: tuple[int, ...] = STARTS,
            short_objective: Callable | None = None) -> tuple[dict, dict]:
    """Shared E324 numerical path; injected dependencies permit small generated checks.

    The original CLI uses fixed STARTS and the E319 short objective. A direct
    generated-only call may inject a smaller ordered subset and fake model.
    """
    import torch

    if (not starts or any(type(c) is not int or c < 100 or c > 296 for c in starts)
            or tuple(sorted(set(starts))) != starts
            or dependencies is None and starts != STARTS):
        raise ValueError("Fixed ordered E324 starts required")
    deps = Dependencies() if dependencies is None else dependencies
    save = checkpoint or (lambda _name, _arrays: None)
    timely = deadline or (lambda: None)
    report_receipt = on_receipt or (lambda _name, _value: None)
    progress = on_progress or (lambda _value: None)
    boundary = on_boundary or (lambda _name, _event: None)
    shape = short_objective or short_shape
    if (data["drives"].shape != (300, N)
            or data["motor_rates"].shape != (300, 391)
            or data["rates"].shape != (300, N)
            or data["adaptations"].shape != (300, N)):
        raise ValueError("E324 requires all 300 recorded controls and MN rows")
    result = {"conditional_fixed_neural_input": True, "window_starts": list(starts),
              "window_controls": 4, "steps_per_control": 20,
              "shared_direction_count": 1, "body_calls": 0,
              "optimizer_steps": 0, "decoder_replay_controls": 0,
              "decoder_replay_work": {"requested_controls": 0,
                                      "host_returned_controls": 0,
                                      "device_confirmed_controls": None},
              "windows": [], "runs": {}}
    receipts = {}

    def marked(name, action):
        timely()
        boundary(name, "requested")
        try:
            value = action()
        except BaseException as error:
            try:
                boundary(name, "failed")
            except Exception as secondary:  # noqa: BLE001
                error.add_note("E324 journal failure: " + repr(secondary))
            raise
        boundary(name, "host_returned")
        return value

    def retain_failure_receipt(name, receipt, error):
        if receipt is None:
            return
        try:
            receipt.confirm()
        except Exception as secondary:  # noqa: BLE001 - retain the neural error
            error.add_note("E324 receipt confirmation failure: " + repr(secondary))
        try:
            receipts[name] = _receipt(receipt)
            report_receipt(name, receipts[name])
        except Exception as secondary:  # noqa: BLE001 - retain the neural error
            error.add_note("E324 receipt journal failure: " + repr(secondary))

    rollout, objective, initial = marked("construct", lambda: deps.build(data, device))
    r0, z0, bank0, original0, all_drives, mask = initial
    if (all_drives.shape != (300, N) or len(data["motor_rates"]) != 300
            or data["motor_rates"].shape != (300, 391)
            or data["rates"].shape != (300, N)
            or data["adaptations"].shape != (300, N)):
        raise ValueError("E324 requires all 300 recorded controls and MN rows")
    timely()
    # The fixed decoder is replayed once from saved control0 and observed E318
    # motor rates. This reconstructs all 50 bank/original start boundaries.
    def replay():
        banks, originals, q_rows, tendon_rows = [bank0], [original0], [], []
        with torch.no_grad():
            for control, motor in enumerate(data["motor_rates"], start=1):
                result["decoder_replay_work"]["requested_controls"] = control
                decoded = rollout.decoder.step(
                    torch.as_tensor(np.ascontiguousarray(motor), device=rollout.device),
                    banks[-1], originals[-1], dt=.005)
                banks.append(decoded["bank"])
                originals.append(decoded["original_activation"])
                q_rows.append(decoded["q"])
                tendon_rows.append(decoded["tendon"])
                result["decoder_replay_controls"] = control
                result["decoder_replay_work"]["host_returned_controls"] = control
        return banks, originals, torch.stack(q_rows), torch.stack(tendon_rows)

    try:
        banks, originals, replay_q, replay_tendon = marked("decoder_replay", replay)
        # One host-returned progress record survives a later CUDA copy/save
        # failure, without 300 per-control disk writes.
        progress(result.copy())
        replay_arrays = {"q": _host(replay_q), "tendon": _host(replay_tendon),
                         "bank100": _host(banks[100]), "bank300": _host(banks[300]),
                         "original100": _host(originals[100]),
                         "original300": _host(originals[300])}
        save("decoder-replay", replay_arrays)
    except BaseException as error:
        try:
            progress(result.copy())
        except Exception as secondary:  # noqa: BLE001 - retain replay failure
            error.add_note("E324 replay progress journal failure: " + repr(secondary))
        raise
    result["decoder_replay_work"]["device_confirmed_controls"] = 300
    progress(result.copy())
    boundary("decoder_replay", "device_confirmed")
    for name, array, size in (("q", replay_arrays["q"], 42),
                               ("tendon", replay_arrays["tendon"], 6)):
        _valid_array(array, (300, size), np.float64, "decoder " + name)
        _valid_array(data["raw"]["command" if name == "q" else "tendon"],
                     (301, size), np.float64, "E318 " + name)
    q_error = float(np.max(np.abs(replay_arrays["q"] - data["raw"]["command"][1:])))
    tendon_error = float(np.max(np.abs(replay_arrays["tendon"]
                                      - data["raw"]["tendon"][1:])))
    replay_errors = {}
    for control, whole in ((100, data["whole100"]), (300, data["final"])):
        for key, size in (("bank", (391, 3)), ("original", (96,))):
            value = replay_arrays[key + str(control)]
            expected = (whole["adapter"]["bank"] if key == "bank" else
                        whole["adapter"]["original"]["activation"])
            _valid_array(value, size, np.float64, "replay " + key)
            _valid_array(expected, size, np.float64, "E318 " + key)
            replay_errors[key + str(control)] = float(np.max(np.abs(value - expected)))
    result["decoder_reproduction"] = {"q_max_abs_rad": q_error,
                                      "tendon_max_abs_command": tendon_error,
                                      **{key + "_max_abs": value for key, value in replay_errors.items()}}
    progress(result.copy())
    if (q_error > 1e-11 or tendon_error > 1e-11
            or any(value > 1e-10 for value in replay_errors.values())):
        raise ValueError("E324 original decoder replay reproduction failed")
    timely()

    tensor = lambda value: torch.as_tensor(np.ascontiguousarray(value), device=rollout.device)

    def start_state(c):
        if c == 0:
            return r0, z0, bank0, original0
        return tensor(data["rates"][c - 1]), tensor(data["adaptations"][c - 1]), banks[c], originals[c]

    def perform(c, label, efficacy, *, backward=False):
        name = f"window-{c:03d}-{label}"
        r, z, bank, original = start_state(c)
        receipt = None
        active = name + "_forward"
        try:
            timely()
            boundary(active, "requested")
            output = rollout.run(r, z, bank, original,
                                 all_drives[c:c + 4], efficacy, mask)
            receipt = output["receipt"]
            boundary(active, "host_returned")
            receipts[name] = _receipt(receipt)
            report_receipt(name, receipts[name])
            arrays = {key: _host(output[key]) for key in
                      ("q", "tendon", "rate", "adaptation", "bank", "original_activation")}
            save(name, arrays)
            receipt.confirm()
            receipts[name] = _receipt(receipt)
            report_receipt(name, receipts[name])
            boundary(active, "device_confirmed")
            active = None
            for key, size, dtype in (("q", (4, 42), np.float64),
                                     ("tendon", (4, 6), np.float64),
                                     ("rate", (N,), np.float32),
                                     ("adaptation", (N,), np.float32),
                                     ("bank", (391, 3), np.float64),
                                     ("original_activation", (96,), np.float64)):
                _valid_array(arrays[key], size, dtype, name + " " + key)
            reproduction = None
            if backward:
                raw = data["raw"]
                qerr = float(np.max(np.abs(arrays["q"] - raw["command"][c + 1:c + 5])))
                terr = float(np.max(np.abs(arrays["tendon"] - raw["tendon"][c + 1:c + 5])))
                endpoint_exact = (e323.exact(arrays["rate"], data["rates"][c + 3])
                                  and e323.exact(arrays["adaptation"],
                                                 data["adaptations"][c + 3]))
                expected_bank = _host(banks[c + 4])
                expected_original = _host(originals[c + 4])
                _valid_array(data["rates"][c + 3], (N,), np.float32,
                             "E318 window rate")
                _valid_array(data["adaptations"][c + 3], (N,), np.float32,
                             "E318 window adaptation")
                _valid_array(expected_bank, (391, 3), np.float64,
                             "replayed window bank")
                _valid_array(expected_original, (96,), np.float64,
                             "replayed window original")
                berr = float(np.max(np.abs(arrays["bank"] - expected_bank)))
                oerr = float(np.max(np.abs(arrays["original_activation"]
                                              - expected_original)))
                reproduction = {"q_max_abs_rad": qerr, "tendon_max_abs_command": terr,
                                "end_rate_adaptation_dtype_bytes_exact": endpoint_exact,
                                "bank_max_abs": berr, "original_max_abs": oerr}
                if (qerr > 1e-11 or terr > 1e-11 or not endpoint_exact
                        or berr > 1e-10 or oerr > 1e-10):
                    raise ValueError("E324 neutral window reproduction failed: " + name)
            loss, per_leg = shape(output["q"], objective.curves)
            loss_value = float(_host(loss))
            per_leg_value = _host(per_leg)
            if not np.isfinite(loss_value) or not np.isfinite(per_leg_value).all():
                raise ValueError("Nonfinite E324 short shape")
            gradient = None
            if backward:
                efficacy.grad = None
                active = name + "_backward"
                timely()
                boundary(active, "requested")
                (loss / len(starts)).backward()
                boundary(active, "host_returned")
                receipt.confirm()
                receipts[name] = _receipt(receipt)
                report_receipt(name, receipts[name])
                boundary(active, "device_confirmed")
                active = None
                gradient = _host(efficacy.grad)
                _valid_array(gradient, (GROUPS,), np.float64, name + " gradient")
                save(name + "-gradient", {"gradient": gradient})
            row = {"start_control": c, "label": label, "shape_loss": loss_value,
                   "per_leg_shape_loss": per_leg_value.tolist(),
                   "raw_file": name + ".npz", "receipt": _receipt(receipt)}
            if reproduction is not None:
                row["neutral_reproduction"] = reproduction
                row["gradient_file"] = name + "-gradient.npz"
            result["runs"][name] = row
            progress(result.copy())
            timely()
            return loss_value, gradient
        except BaseException as error:
            if active is not None:
                try:
                    boundary(active, "failed")
                except Exception as secondary:  # noqa: BLE001
                    error.add_note("E324 journal failure: " + repr(secondary))
            if receipt is None:
                receipt = getattr(error, "receipt", None)
            retain_failure_receipt(name, receipt, error)
            raise

    theta = torch.ones(GROUPS, dtype=torch.float64, device=rollout.device,
                       requires_grad=True)
    neutral_losses, gradients = [], []
    for c in starts:
        loss, gradient = perform(c, "neutral", theta, backward=True)
        neutral_losses.append(loss)
        gradients.append(gradient)
    neutral = float(np.mean(neutral_losses, dtype=np.float64))
    if abs(neutral - data["e323_neutral_shape_loss"]) > 1e-12:
        raise ValueError("E324 neutral reset mean differs from audited E323 shape")
    gradient = np.sum(np.stack(gradients), axis=0, dtype=np.float64)
    _valid_array(gradient, (GROUPS,), np.float64, "aggregated gradient")
    if not np.any(gradient):
        raise ValueError("E324 zero gradient cannot define a direction")
    direction = np.sign(gradient)
    plus = np.float32(1 + .001 * direction).astype(np.float64)
    minus = np.float32(1 - .001 * direction).astype(np.float64)
    if not np.any(plus != 1) or not np.any(minus != 1):
        raise ValueError("E324 f32 efficacy perturbation vanished")
    save("direction", {"gradient": gradient, "direction": direction,
                       "theta_plus": plus, "theta_minus": minus,
                       "window_gradients": np.stack(gradients)})
    selected = data["native"]["csr"]["data"][data["native"]["mask"]["positions"]]
    groups = data["native"]["mask"]["edge_parameters"]
    if len(selected) != len(groups) or not len(selected):
        raise ValueError("E324 selected source edge/group mismatch")
    weight_delta = {}
    for label, theta_value in (("plus", plus), ("minus", minus)):
        modified = np.float32(selected.astype(np.float64) * theta_value[groups])
        delta = modified.astype(np.float64) - selected.astype(np.float64)
        weight_delta[label] = {"changed_count": int(np.count_nonzero(delta)),
                               "min": float(delta.min()), "max": float(delta.max())}
    result["direction"] = {"nonzero_gradient_groups": int(np.count_nonzero(direction)),
                           "gradient_min": float(gradient.min()),
                           "gradient_max": float(gradient.max()),
                           "theta_plus_min": float(plus.min()),
                           "theta_plus_max": float(plus.max()),
                           "theta_minus_min": float(minus.min()),
                           "theta_minus_max": float(minus.max()),
                           "selected_weight_delta": weight_delta}
    progress(result.copy())

    plus_losses, minus_losses = [], []
    for c, baseline, g in zip(starts, neutral_losses, gradients, strict=True):
        plus_loss, _ = perform(c, "plus", tensor(plus))
        minus_loss, _ = perform(c, "minus", tensor(minus))
        plus_losses.append(plus_loss)
        minus_losses.append(minus_loss)
        observed = (plus_loss - minus_loss) / len(starts)
        predicted = float(np.sum(g * (plus - minus), dtype=np.float64))
        result["windows"].append({"start_control": c, "neutral_shape_loss": baseline,
                                  "plus_shape_loss": plus_loss,
                                  "minus_shape_loss": minus_loss,
                                  "observed_contribution": observed,
                                  "predicted_contribution": predicted,
                                  "gradient_file": f"window-{c:03d}-neutral-gradient.npz"})
        progress(result.copy())
    plus_mean = float(np.mean(plus_losses, dtype=np.float64))
    minus_mean = float(np.mean(minus_losses, dtype=np.float64))
    observed = plus_mean - minus_mean
    predicted = float(np.sum(gradient * (plus - minus), dtype=np.float64))
    comparison = e323.classify(observed, predicted)
    result.update(neutral_shape_loss=neutral, plus_shape_loss=plus_mean,
                  minus_shape_loss=minus_mean, comparison=comparison,
                  minus_improves_neutral=bool(minus_mean < neutral),
                  continuation_requested=False,
                  continuation_completed=False)
    progress(result.copy())

    if comparison["classification"] == "supported" and minus_mean < neutral:
        name = "continuation-minus"
        receipt = None
        active = name + "_forward"
        try:
            timely()
            boundary(active, "requested")
            result["continuation_requested"] = True
            progress(result.copy())
            output = rollout.run(r0, z0, bank0, original0, all_drives,
                                 tensor(minus), mask)
            receipt = output["receipt"]
            boundary(active, "host_returned")
            receipts[name] = _receipt(receipt)
            report_receipt(name, receipts[name])
            save(name, {key: _host(output[key]) for key in
                        ("q", "tendon", "rate", "adaptation", "bank",
                         "original_activation")})
            receipt.confirm()
            receipts[name] = _receipt(receipt)
            report_receipt(name, receipts[name])
            boundary(active, "device_confirmed")
            active = None
            for key, dimensions, dtype in (("q", (300, 42), np.float64),
                                           ("tendon", (300, 6), np.float64),
                                           ("rate", (N,), np.float32),
                                           ("adaptation", (N,), np.float32),
                                           ("bank", (391, 3), np.float64),
                                           ("original_activation", (96,), np.float64)):
                _valid_array(_host(output[key]), dimensions, dtype,
                             "continuation " + key)
            metrics = e323._metrics(objective.evaluate(output["q"], output["tendon"]))
            result["continuation_completed"] = True
            result["continuation"] = {"metrics": metrics,
                                      "receipt": _receipt(receipt),
                                      "raw_file": name + ".npz"}
            result["continuation_supported"] = bool(
                metrics["shape_loss"] < data["e323_neutral_shape_loss"]
                and metrics["startup_q_rms"] <= .018
                and metrics["startup_q_max"] <= .04
                and metrics["startup_tendon_rms"] <= .009
                and metrics["startup_tendon_max"] <= .025
                and metrics["startup_pass"])
            progress(result.copy())
            timely()
        except BaseException as error:
            if active is not None:
                try:
                    boundary(active, "failed")
                except Exception as secondary:  # noqa: BLE001
                    error.add_note("E324 journal failure: " + repr(secondary))
            if receipt is None:
                receipt = getattr(error, "receipt", None)
            retain_failure_receipt(name, receipt, error)
            raise
    else:
        result["continuation_supported"] = None
    result["actual_work"] = _counters(receipts, result["decoder_replay_controls"],
                                      result["continuation_requested"], len(starts))
    for key in ("forward_neural_ticks", "recompute_neural_ticks",
                "backward_block_ticks", "backward_blocks", "decoder_replay_controls"):
        if result["actual_work"][key] > LIMITS[key]:
            raise ValueError("E324 work cap exceeded: " + key)
    return result, receipts


def put_json(path: Path, value: dict) -> None:
    """Bounded atomic metadata replacement, including Windows ledger contention."""
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", "utf8")
    for attempt in range(3):
        try:
            os.replace(pending, path)
            return
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(.02)


def run(freeze_path: Path, freeze_sha256: str, review_path: Path,
        review_sha256: str, output: Path, *, root: Path = ROOT,
        gatekeeper=gate, dependencies=None, clock=time.perf_counter,
        writer=put_json, saver=np.savez_compressed, meter=e323.usage,
        device="cuda") -> dict:
    """One passively approved original attempt, with auditable partial cost."""
    began = clock()
    freeze = gatekeeper(freeze_path, freeze_sha256, review_path,
                        review_sha256, output, root=root)
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema": RESULT_SCHEMA, "diagnostic_id": "E324",
              "run_valid": False, "candidate_adopted": False,
              "biological_gait_proven": False, "F1_F8_complete": False,
              "F6_exposed": False, "M2_modified": False,
              "classification": "not_tested",
              "conditional_fixed_neural_input": True,
              "provenance": {"freeze_sha256": freeze_sha256,
                             "review_sha256": review_sha256,
                             "source_manifest": "source-manifest.json",
                             "e323_result_sha256": freeze["input_sha256"][PATHS["e323_result"]],
                             "e323_audit_sha256": freeze["input_sha256"][PATHS["e323_audit"]]},
              "publication": {"measured_boundary": "before_terminal_metadata_commit",
                              "external_audit_pins_required": ["result.json",
                                                               "advance-ledger.json"],
                              "metadata_excluded_from_manifest": True},
              "errors": []}
    ledger = {"limits": LIMITS.copy(), "retry_count": 0, "body_calls": 0,
              "optimizer_steps": 0, "attempts": [], "phases": {},
              "receipts": {}, "errors": []}
    first_error = None
    baseline = None
    science_began = clock()

    def failure(stage, error):
        nonlocal first_error
        entry = {"stage": stage, "type": type(error).__name__,
                 "message": str(error)[:500]}
        ledger["errors"].append(entry)
        if first_error is None:
            first_error = entry
        return entry

    def persist():
        ledger["elapsed_s"] = clock() - began
        writer(output / "advance-ledger.json", ledger)

    def check_resource(value):
        if baseline is None:
            return
        if value["device_total_bytes"] != baseline["device_total_bytes"]:
            raise RuntimeError("E324 device identity changed")
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
            raise MemoryError("E324 host/device resource limit exceeded")

    def deadline():
        if clock() - began >= LIMITS["internal_wall_s"]:
            raise TimeoutError("E324 internal deadline exceeded")
        if clock() - science_began >= LIMITS["science_wall_s"]:
            raise TimeoutError("E324 science deadline exceeded")

    def checkpoint(name, arrays):
        path = output / (name + ".npz")
        attempt = {"stage": "save_" + name, "requested": True,
                   "returned": None, "requested_elapsed_s": clock() - began}
        ledger["attempts"].append(attempt)
        persist()
        saver(path, **arrays)
        attempt["returned"] = True
        attempt["bytes"] = path.stat().st_size
        attempt["returned_elapsed_s"] = clock() - began
        persist()

    try:
        persist()
        # e323.usage initializes the Torch CUDA runtime before any E315 CuPy
        # loader is reached. It also samples total device use and Torch peak.
        baseline = meter()
        ledger["resource_baseline"] = baseline
        writer(output / "source-manifest.json", {
            "schema": "flylab.m3.e324.source-manifest.v1",
            "freeze_sha256": freeze_sha256, "review_sha256": review_sha256,
            "source_sha256": freeze["source_sha256"],
            "input_sha256": freeze["input_sha256"]})
        deps = Dependencies() if dependencies is None else dependencies
        ledger["phases"]["input_load"] = {
            "requested": True, "host_returned": None,
            "device_confirmed": None, "failed": False,
            "requested_elapsed_s": clock() - began}
        persist()
        try:
            data = deps.load(freeze, root=root)
        except BaseException as error:
            ledger["phases"]["input_load"]["failed"] = True
            try:
                persist()
            except Exception as secondary:  # noqa: BLE001 - keep original loader error
                error.add_note("E324 input-load journal failure: " + repr(secondary))
            raise
        ledger["phases"]["input_load"]["host_returned"] = True
        ledger["phases"]["input_load"]["last_event_elapsed_s"] = clock() - began
        persist()
        deadline()
        check_resource(meter())

        def on_receipt(name, value):
            ledger["receipts"][name] = value
            persist()

        def on_boundary(name, event):
            phases = ledger["phases"]
            if event == "requested":
                if name in phases:
                    raise ValueError("E324 phase was requested twice: " + name)
                phases[name] = {"requested": True, "host_returned": None,
                                "device_confirmed": None, "failed": False,
                                "requested_elapsed_s": clock() - began}
            elif event in ("host_returned", "device_confirmed"):
                phases[name][event] = True
            elif event == "failed":
                phases[name]["failed"] = True
            else:
                raise ValueError("Unknown E324 boundary event")
            phases[name]["last_event_elapsed_s"] = clock() - began
            persist()
            if ((event == "host_returned" and name == "construct")
                    or event == "device_confirmed"):
                check_resource(meter())
                deadline()

        def on_progress(value):
            result["partial"] = value
            if "decoder_replay_work" in value:
                ledger["decoder_replay_work"] = value["decoder_replay_work"].copy()
            persist()

        computed, receipts = compute(data, dependencies=deps, checkpoint=checkpoint,
                                     deadline=deadline, on_receipt=on_receipt,
                                     on_progress=on_progress, on_boundary=on_boundary,
                                     device=device)
        deadline()
        ledger["receipts"] = receipts
        result.update(computed)
        result["classification"] = computed["comparison"]["classification"]
        check_resource(meter())
    except Exception as error:  # noqa: BLE001 - preserve original diagnostic failure
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
            files = {path.relative_to(output).as_posix(): {
                "sha256": e323.sha256(path), "bytes": path.stat().st_size}
                for path in output.rglob("*") if path.is_file()
                and path.name not in ("output-manifest.json", "advance-ledger.json",
                                      "result.json", "publication-failure.json")
                and not path.name.endswith(".pending")}
            total_bytes = sum(path.stat().st_size for path in output.rglob("*")
                              if path.is_file())
            if total_bytes > LIMITS["output_bytes"]:
                raise OSError("E324 output size limit exceeded")
            writer(output / "output-manifest.json", {
                "schema": "flylab.m3.e324.output-manifest.v1", "files": files,
                "mutable_metadata_excluded": ["result.json", "advance-ledger.json",
                                              "publication-failure.json"],
                "external_audit_pins_required": True})
            if not (output / "output-manifest.json").is_file():
                raise OSError("E324 output manifest was not published")
            if sum(path.stat().st_size for path in output.rglob("*")
                   if path.is_file()) > LIMITS["output_bytes"]:
                raise OSError("E324 published output size limit exceeded")
        except Exception as error:  # noqa: BLE001
            failure("output_manifest", error)
        # All preliminary metadata/manifest work is inside this measured bound.
        ledger["publication_boundary"] = "before_terminal_metadata_commit"
        ledger["finalization_wall_s"] = clock() - finalization_began
        ledger["elapsed_s"] = clock() - began
        if ledger["finalization_wall_s"] > LIMITS["finalization_reserve_s"]:
            failure("finalization_deadline", TimeoutError("E324 finalization reserve exceeded"))
        if ledger["elapsed_s"] >= LIMITS["internal_wall_s"]:
            failure("internal_deadline", TimeoutError("E324 internal deadline exceeded"))
        result["first_error"] = first_error
        result["errors"] = ledger["errors"].copy()
        result["cost"] = ledger.copy()
        result["run_valid"] = first_error is None
        try:
            writer(output / "advance-ledger.json", ledger)
            writer(output / "result.json", result)
            if (json.loads((output / "advance-ledger.json").read_text("utf8")) != ledger
                    or json.loads((output / "result.json").read_text("utf8")) != result):
                raise OSError("E324 terminal metadata differs from return value")
        except Exception as error:  # noqa: BLE001
            failure("terminal_publication", error)
            result.update(run_valid=False, first_error=first_error,
                          errors=ledger["errors"].copy(), cost=ledger.copy())
            try:
                writer(output / "publication-failure.json", result)
            except Exception as secondary:  # noqa: BLE001
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
