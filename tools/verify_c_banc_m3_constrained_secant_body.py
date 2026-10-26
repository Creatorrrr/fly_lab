"""E327: one E326 efficacy in the actual SSP/u0 body feedback loop.

Only a root-reviewed immutable freeze may reach the original graph, CUDA or
body.  The attempt is single-body, single theta and failed-inclusive.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools import verify_c_banc_m3_constrained_secant as e326
from tools import verify_c_banc_m3_full_drive_capture as e318
from tools import verify_c_banc_m3_full_horizon_direction as e323

ROOT, BASE = e323.ROOT, e323.BASE
E326 = e326.OUTPUT
OUTPUT = BASE + "/e327-constrained-secant-body"
SCHEMA = "flylab.m3.e327.constrained-secant-body.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e327.constrained-secant-body.review.v1"
RESULT_SCHEMA = "flylab.m3.e327.constrained-secant-body.result.v1"
LIMITS = {"supervisor_wall_s": 80, "internal_wall_s": 70,
          "science_wall_s": 55, "finalization_reserve_s": 15,
          "body_constructors": 1, "neural_ticks": 6000,
          "physics_steps": 15000, "host_bytes": 6 << 30,
          "additional_gpu_bytes": 4 << 30, "output_bytes": 1 << 30,
          "minimum_free_disk_bytes": 4 << 30, "retry_count": 0,
          "optimizer_steps": 0}
PATHS = {**e326.PATHS, **e318.PATHS,
         "e326_freeze": BASE + "/e326-constrained-secant-freeze.json",
         "e326_review": BASE + "/e326-constrained-secant-source-review.json",
         "e326_result": E326 + "/result.json",
         "e326_audit": BASE + "/parent-e326-original-raw-audit.json",
         "e326_source_manifest": E326 + "/source-manifest.json",
         "e326_output_manifest": E326 + "/output-manifest.json",
         "e326_ledger": E326 + "/advance-ledger.json",
         "e326_step": E326 + "/constrained-step.npz",
         "e326_output": E326}
OWNED = ("tools/banc_m3_ssp_efficacy_trial.py",
         "tools/verify_c_banc_m3_constrained_secant_body.py",
         "tests/test_c_banc_ssp_efficacy_trial.py",
         "tests/test_c_banc_constrained_secant_body.py")


def source_roster(root: Path = ROOT):
    e326_freeze = json.loads(e323.safe(PATHS["e326_freeze"], root).read_text("utf8"))
    e318_freeze = json.loads(e323.safe(e323.REQUIRED_PATHS["e318_freeze"], root).read_text("utf8"))
    a, b = e326_freeze.get("source_sha256"), e318_freeze.get("source_sha256")
    if not isinstance(a, dict) or not isinstance(b, dict) or not a or not b:
        raise ValueError("Audited E326/E318 source closures required")
    return tuple(sorted(set(OWNED) | set(a) | set(b)))


def input_roster(root: Path = ROOT):
    """Include transitive producer inputs absent from the direct path aliases."""
    e326_freeze = json.loads(e323.safe(PATHS["e326_freeze"], root).read_text("utf8"))
    e318_freeze = json.loads(e323.safe(e323.REQUIRED_PATHS["e318_freeze"], root).read_text("utf8"))
    a, b = e326_freeze.get("input_sha256"), e318_freeze.get("input_sha256")
    if not isinstance(a, dict) or not isinstance(b, dict) or not a or not b:
        raise ValueError("Audited transitive input closures required")
    return tuple(sorted(e323._file_roster(PATHS, root) | set(a) | set(b)))


def gate(freeze_path: Path, freeze_sha256: str, review_path: Path,
         review_sha256: str, output: Path, *, root: Path = ROOT) -> dict:
    """No model/NPZ/CUDA/body import occurs until this full closure passes."""
    if output.exists() or output.resolve() != e323.safe(OUTPUT, root):
        raise ValueError("Fresh literal E327 output required")
    if (e323.sha256(freeze_path) != freeze_sha256
            or e323.sha256(review_path) != review_sha256):
        raise ValueError("E327 freeze/review digest differs")
    freeze = json.loads(freeze_path.read_text("utf8"))
    review = json.loads(review_path.read_text("utf8"))
    pins, sources = freeze.get("input_sha256"), freeze.get("source_sha256")
    if (freeze.get("schema") != SCHEMA or freeze.get("paths") != PATHS
            or freeze.get("limits") != LIMITS or freeze.get("output") != OUTPUT
            or freeze.get("candidate_adopted") is not False
            or not isinstance(pins, dict) or not isinstance(sources, dict)
            or set(pins) != set(input_roster(root))
            or set(sources) != set(source_roster(root))
            or review.get("schema") != REVIEW_SCHEMA
            or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha256
            or review.get("source_sha256") != sources):
        raise ValueError("Exact reviewed E327 immutable contract required")
    prior_freeze = json.loads(e323.safe(PATHS["e326_freeze"], root).read_text("utf8"))
    prior_review = json.loads(e323.safe(PATHS["e326_review"], root).read_text("utf8"))
    prior_result = json.loads(e323.safe(PATHS["e326_result"], root).read_text("utf8"))
    prior_audit = json.loads(e323.safe(PATHS["e326_audit"], root).read_text("utf8"))
    prior_source = json.loads(e323.safe(PATHS["e326_source_manifest"], root).read_text("utf8"))
    prior_manifest = json.loads(e323.safe(PATHS["e326_output_manifest"], root).read_text("utf8"))
    if (prior_freeze.get("schema") != e326.SCHEMA
            or prior_review.get("schema") != e326.REVIEW_SCHEMA
            or prior_review.get("approved") is not True
            or prior_review.get("blocking_defects") != []
            or prior_review.get("freeze_sha256") != pins[PATHS["e326_freeze"]]
            or prior_review.get("source_sha256") != prior_freeze.get("source_sha256")
            or prior_result.get("schema") != e326.RESULT_SCHEMA
            or prior_result.get("run_valid") is not True
            or prior_result.get("classification") != "supported"
            or prior_result.get("constrained_continuation_supported") is not True
            or prior_result.get("continuation_completed") is not True
            or prior_result.get("actual_work", {}).get("forward_neural_ticks") != 6000
            or prior_audit.get("schema") != "flylab.m3.e326.root-original-raw-audit.v1"
            or prior_audit.get("pass") is not True
            or prior_audit.get("constrained_continuation_supported") is not True
            or prior_audit.get("result_sha256") != pins[PATHS["e326_result"]]
            or prior_audit.get("advance_ledger_sha256") != pins[PATHS["e326_ledger"]]
            or prior_audit.get("output_manifest_sha256") != pins[PATHS["e326_output_manifest"]]
            or prior_source.get("source_sha256") != prior_freeze.get("source_sha256")
            or prior_source.get("input_sha256") != prior_freeze.get("input_sha256")
            or prior_manifest.get("schema") != "flylab.m3.e326.output-manifest.v1"):
        raise ValueError("Root-audited successful E326 source required")
    for name, digest in prior_freeze["source_sha256"].items():
        if sources.get(name) != digest:
            raise ValueError("E326 source closure differs: " + name)
    for name, digest in prior_freeze["input_sha256"].items():
        if pins.get(name) != digest:
            raise ValueError("E326 input closure differs: " + name)
    e318_freeze = json.loads(e323.safe(e323.REQUIRED_PATHS["e318_freeze"], root).read_text("utf8"))
    for name, digest in e318_freeze["source_sha256"].items():
        if sources.get(name) != digest:
            raise ValueError("E318 body source closure differs: " + name)
    for name, digest in e318_freeze["input_sha256"].items():
        if pins.get(name) != digest:
            raise ValueError("E318 body input closure differs: " + name)
    for name, digest in pins.items():
        if (not isinstance(digest, str)
                or e323.sha256(e323.safe(name, root)) != digest):
            raise ValueError("E327 input differs: " + name)
    for name, digest in sources.items():
        if (not isinstance(digest, str)
                or e323.sha256(e323.safe(name, root)) != digest):
            raise ValueError("E327 source differs: " + name)
    producer = e323.safe(E326, root)
    actual = {p.relative_to(producer).as_posix() for p in producer.rglob("*")
              if p.is_file() and p.name not in ("result.json", "advance-ledger.json",
                                                  "output-manifest.json")}
    if actual != set(prior_manifest.get("files", {})):
        raise ValueError("E326 output artifact roster differs")
    for name in actual:
        path = producer / name
        if prior_manifest["files"][name] != {
                "sha256": pins[path.relative_to(root).as_posix()],
                "bytes": path.stat().st_size}:
            raise ValueError("E326 output file differs: " + name)
    if shutil.disk_usage(output.parent).free < LIMITS["minimum_free_disk_bytes"]:
        raise OSError("E327 free disk reserve unavailable")
    return freeze


def _json(path: Path, value: dict):
    path.write_text(json.dumps(_plain(value), allow_nan=False, indent=2) + "\n", "utf8")


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def evaluate_saved(folder: Path, curve, startup):
    """Re-evaluate both candidate and saved neutral with the same frozen code."""
    import torch

    from tools import verify_c_banc_m3_resource_continuation as e310
    from tools.banc_m3_phase_free_shape import PhaseFreeShapeObjective

    load = e310.runner.protocol.load_npz
    raw = load(folder / "control-raw.npz")
    native = load(folder / "native-contact.npz")
    rows = json.loads((folder / "trace.json").read_text("utf8"))["samples"]
    if (len(rows) != 301 or len(native["time_s"]) != 15000
            or not np.array_equal(raw["control_tick"], np.arange(301))
            or not np.array_equal(raw["network_tick"], np.arange(301) * 20)
            or not np.allclose(raw["physics_time_s"], np.arange(301) * .005,
                               rtol=0, atol=2e-8)):
        raise ValueError("Complete actual 301-control / 15000-native clocks required")
    objective = PhaseFreeShapeObjective(curve, startup["target_command"],
                                        startup["target_tendon"], device="cpu")
    q, tendon = raw["command"][1:], raw["tendon"][1:]
    shape = e323._metrics(objective.evaluate(torch.as_tensor(q), torch.as_tensor(tendon)))
    combined = {"rows": rows, "native_arrays": native}
    clock = SimpleNamespace(neural_s=1.5, physical_s=1.5, clock_valid=True)
    short = e310.NativeDependencies.short_metrics(None, combined, folder, clock, {})
    summary = e310.metric_summary(short)
    return _plain({"shape": shape, "short": short, "summary": summary})


def classify(candidate: dict, neutral: dict, *, engineering_valid: bool):
    shape = candidate["shape"]
    gait = candidate["summary"]
    gates = {
        "engineering_valid": bool(engineering_valid),
        "startup_four": bool(shape["startup_q_rms"] <= .018
                             and shape["startup_q_max"] <= .04
                             and shape["startup_tendon_rms"] <= .009
                             and shape["startup_tendon_max"] <= .025),
        "shape_better_than_saved_neutral": bool(
            shape["shape_loss"] < neutral["shape"]["shape_loss"]),
        "forward": bool(gait["forward_mm"] >= 1.6790085779712631),
        "slip": bool(gait["slip_mm_s"] <= 1.8026436968249955),
        "bout": bool(gait["longest_continuous_forward_bout_s"] >= .3),
        "all_six_eligible_cycles": bool(all(
            gait["complete_cycles"].get(leg, 0) > 0
            for leg in ("lf", "lm", "lh", "rf", "rm", "rh"))),
        "native_health": bool(gait["health"].get("healthy") is True),
    }
    return gates, bool(all(gates.values()))


class Dependencies:
    """E318/E310 measured native path with one additional efficacy layer."""
    def __init__(self):
        from tools import banc_m3_ssp_efficacy_trial as layer
        from tools import verify_c_banc_m3_full_drive_capture as e318
        self.e318, self.layer = e318, layer
        self.base = e318.Dependencies()
        self.native = self.base.native
        self.wrapper = self.base.wrapper
        self.health_stop = self.base.health_stop

    def load(self, freeze):
        from tools.verify_c_banc_m3_native_adjoint import load_inputs, load_npz
        e315 = json.loads(e323.safe(PATHS["e315_freeze"]).read_text("utf8"))
        native = load_inputs(e315)
        data = self.base.load(freeze)
        step = load_npz(e323.safe(PATHS["e326_step"]))
        theta = step["theta"]
        if (theta.dtype != np.float64 or theta.shape != (e323.GROUPS,)
                or not np.isfinite(theta).all() or np.any(theta <= 0)
                or np.array_equal(theta, np.ones_like(theta))):
            raise ValueError("Single actual positive E326 f64 theta required")
        if not np.array_equal(native["mask"]["positions"], data["roster"]["edge_ids"]):
            raise ValueError("E29 selected CSR and E289 roster differ")
        prior = json.loads(e323.safe(PATHS["e326_result"]).read_text("utf8"))
        expected = np.float32(1 + float(prior["alpha"]["alpha_star"])
                              * (step["source_theta_minus"] - 1)).astype(np.float64)
        if (not np.array_equal(theta, expected)
                or float(step["alpha_star"]) != float(prior["alpha"]["alpha_star"])
                or not np.array_equal(theta.astype(np.float32).astype(np.float64), theta)):
            raise ValueError("E326 audited single theta differs")
        startup = load_npz(e323.safe(PATHS["e103_startup"]))
        reference = load_npz(e323.safe(PATHS["e85_reference"]))
        data.update(positions=native["mask"]["positions"],
                    edge_parameters=native["mask"]["edge_parameters"],
                    theta=theta, startup=startup, curve=e323.curve128(reference))
        return data

    def construct(self, data, ledger, persist, on_return):
        def callback():
            old = self.wrapper.SSPRecoveryResourceTrial.from_checkpoint(
                data["graph"], copy.deepcopy(data["saved"]))
            try:
                trial = self.layer.SSPEfficacyTrial(
                    old, data["positions"], data["edge_parameters"], data["theta"])
                trial._e291_arm = "neutral"
                return trial
            except Exception as error:
                self.wrapper.cleanup_owned(old, error)
                raise
        return self.native._construction("efficacy", callback, ledger,
                                         persist, on_return)

    def initial(self, session, data):
        observed = session.snapshot()
        inverse = self.layer._source_representation(
            observed, observed[self.layer.PATCH_KEY])
        self.wrapper.require_saved_whole_equal(
            inverse, data["saved"], "Exact original whole except efficacy layer")
        if (session.control_tick != 0 or session.network.tick != 0
                or abs(session.body.physics_time()) > 2e-8
                or session.adapter.identity != self.e318.ADAPTER_ID
                or session.network.identity == self.e318.NETWORK_ID
                or session.network.device != "cuda"
                or session.network.cuda_implementation != "packed"
                or session.network.interval_steps != 20):
            raise ValueError("Exact new SSP efficacy control0/native CUDA origin required")
        return {"source_whole_exact_including_wall": True,
                "candidate_network_identity": session.network.identity,
                "source_network_identity": self.e318.NETWORK_ID,
                "patch_identity": observed[self.layer.PATCH_KEY]["patch_identity"]}

    def capture(self, session, raw):
        return self.base.capture(session, raw)

    def save_raw(self, folder, raw, data, clock):
        return self.base.save_raw(folder, raw, data, clock)

    def __getattr__(self, name):
        return getattr(self.base, name)


def run(freeze_path: Path, freeze_sha256: str, review_path: Path,
        review_sha256: str, output: Path, *, root: Path = ROOT,
        gatekeeper=gate, dependencies=None, clock=time.perf_counter,
        writer=_json, meter=e323.usage) -> dict:
    began = clock()
    freeze = gatekeeper(freeze_path, freeze_sha256, review_path,
                        review_sha256, output, root=root)
    output.mkdir(parents=True, exist_ok=False)
    ledger = {"limits": LIMITS.copy(), "construction": [],
              "body_construction": [], "construction_requested": 0,
              "construction_completed": 0, "body_constructors": 0,
              "controls_requested": 0, "controls_completed": 0,
              "neural_requested": 0, "neural_host_returned": 0,
              "neural_device_confirmed": 0, "physics_steps_completed": 0,
              "retry_count": 0, "optimizer_steps": 0, "errors": []}
    result = {"schema": RESULT_SCHEMA, "diagnostic_id": "E327",
              "run_valid": False, "constrained_body_supported": False,
              "candidate_adopted": False, "biological_gait_proven": False,
              "F1_F8_complete": False, "F6_exposed": False, "M2_modified": False,
              "classification": "not_tested"}
    raw = {key: [] for key in ("rows", "controls", "recruitment", "activity", "native")}
    session = deps = detach = journal = native_clock = None
    holder, errors, first_error = {}, [], None
    science_began = clock()

    def persist():
        ledger["elapsed_s"] = clock() - began
        pending = output / "advance-ledger.pending.json"
        writer(pending, ledger)
        from tools.verify_c_banc_m3_full_drive_capture import replace_owned
        replace_owned(pending, output / "advance-ledger.json")

    def fail(stage, error):
        nonlocal first_error
        entry = {"stage": stage, "type": type(error).__name__,
                 "message": str(error)[:500]}
        errors.append(entry)
        if first_error is None:
            first_error = entry
        ledger["errors"] = errors.copy()
        result.update(run_valid=False, constrained_body_supported=False,
                      first_error=first_error, errors=errors.copy())

    def deadline():
        if clock() - began >= LIMITS["internal_wall_s"]:
            raise TimeoutError("E327 internal deadline exceeded")
        if clock() - science_began >= LIMITS["science_wall_s"]:
            raise TimeoutError("E327 science deadline exceeded")

    def check_resources(sample):
        if sample["device_total_bytes"] != baseline["device_total_bytes"]:
            raise RuntimeError("E327 CUDA device identity changed")
        ledger["resource_latest"] = sample
        ledger["additional_gpu_peak_bytes"] = max(
            ledger.get("additional_gpu_peak_bytes", 0),
            baseline["device_free_bytes"] - sample["device_free_bytes"])
        reserved = sample.get("torch_peak_reserved_bytes",
                              sample.get("gpu_peak_reserved_bytes", 0))
        initial = baseline.get("torch_reserved_bytes",
                               baseline.get("gpu_reserved_bytes", 0))
        ledger["additional_torch_peak_reserved_bytes"] = max(
            ledger.get("additional_torch_peak_reserved_bytes", 0),
            reserved - initial)
        if (sample["host_peak_bytes"] > LIMITS["host_bytes"]
                or ledger["additional_gpu_peak_bytes"] > LIMITS["additional_gpu_bytes"]
                or ledger["additional_torch_peak_reserved_bytes"]
                > LIMITS["additional_gpu_bytes"]):
            raise MemoryError("E327 host/device resource cap exceeded")

    try:
        persist()
        baseline = meter()  # e323 meter initializes Torch CUDA before CuPy.
        ledger["resource_baseline"] = baseline
        writer(output / "source-manifest.json", {
            "schema": "flylab.m3.e327.source-manifest.v1",
            "freeze_sha256": freeze_sha256, "review_sha256": review_sha256,
            "source_sha256": freeze["source_sha256"],
            "input_sha256": freeze["input_sha256"]})
        deps = Dependencies() if dependencies is None else dependencies
        data = deps.load(freeze)
        raw["ids"] = data["ids"]
        deadline()
        ledger["construction_requested"] = 1
        persist()
        session = deps.construct(data, ledger, persist,
                                 lambda value: holder.setdefault("session", value))
        ledger["construction_completed"] = 1
        ledger["body_constructors"] = 1
        if (len(ledger["construction"]) != 1
                or len(ledger["body_construction"]) != 1
                or any(row.get(key) != 0 for row in ledger["construction"]
                       for key in ("unknown_setup_steps", "unknown_capture_setup_steps"))
                or any(row.get(key) != 0 for row in ledger["body_construction"]
                       for key in ("unknown_step_entries", "active_unreturned_frames"))):
            raise RuntimeError("Exact one-body counted construction required")
        check_resources(meter())
        deadline()
        result["origin"] = deps.initial(session, data)
        deps.save_state(output / "observed-start", session)
        patch = session.ssp_efficacy_trial_patch
        np.savez_compressed(output / "patched-edge-roster.npz",
                            positions=patch["positions"],
                            edge_parameters=patch["edge_parameters"],
                            theta64=patch["theta64"], original32=patch["original32"],
                            actual32=patch["actual32"])
        from tools.probe_c_banc_m3_core_rhythm import source_matrix
        actual_csr = source_matrix(session.network)
        np.savez_compressed(output / "patched-csr.npz",
                            indptr=actual_csr.indptr, indices=actual_csr.indices,
                            data=actual_csr.data)
        del actual_csr
        if hasattr(deps, "trace_metadata"):
            raw["trace_metadata"] = deps.trace_metadata(session)
        detach = deps.install(session, raw["native"])
        native_clock = deps.base.e310.runner.native.AdvanceLedger()
        journal = deps.journal(output / "control-journal.jsonl")
        deps.capture(session, raw)
        for control in range(1, 301):
            deadline()
            ledger["controls_requested"] = control
            ledger["neural_requested"] = control * 20
            persist()
            before_native, before_tick = len(raw["native"]), session.network.tick
            event = deps.advance(session, "neutral", control, native_clock, journal)
            ledger["neural_host_returned"] = session.network.tick
            deps.capture(session, raw)
            deps.readout_receipt(journal, "neutral", control, event, True)
            if (session.control_tick != control or session.network.tick != before_tick + 20
                    or len(raw["native"]) != before_native + 50):
                raise ValueError("Incomplete E327 neural/native interval")
            ledger["neural_device_confirmed"] = control * 20
            ledger["controls_completed"] = control
            ledger["physics_steps_completed"] = len(raw["native"])
            native_last = deps.base.e310.runner.native.native_arrays(raw["native"][-50:])
            if (not np.isfinite(native_last["upright"]).all()
                    or np.any(native_last["upright"] < .9)
                    or np.any(native_last["nonfoot_contact"])):
                raise deps.health_stop("Measured E327 upright/nonfoot health stop")
            if session.fault or session.body.fault:
                raise deps.health_stop("Measured E327 session/body health stop")
            check_resources(meter())
            persist()
        deadline()
        if (not native_clock.clock_valid or abs(native_clock.neural_s - 1.5) > 1e-9
                or abs(native_clock.physical_s - 1.5) > 1e-9):
            raise ValueError("Exact full native/neural measured clock required")
        deps.save_state(output / "whole-300", session)
        deadline()
        result["science_complete"] = True
    except Exception as error:  # noqa: BLE001 - retain partial measured evidence
        fail("science", error)
        if session is not None:
            ledger["neural_host_returned_after_error"] = int(session.network.tick)
            ledger["control_tick_after_error"] = int(session.control_tick)
            ledger["native_samples_after_error"] = len(raw["native"])
        if deps is not None and isinstance(error, deps.health_stop):
            result["classification"] = "measured_native_health_stop"
            result["health_stop"] = {"control_attempted": ledger["controls_requested"],
                                     "native_samples": len(raw["native"])}
        result["science_error"] = traceback.format_exc(limit=8)[-3000:]
    finally:
        ledger["science_wall_s"] = clock() - science_began
        ledger["native_samples_observed"] = len(raw["native"])
        if native_clock is not None:
            ledger.update(clock_events=copy.deepcopy(native_clock.events),
                          clock_valid=native_clock.clock_valid,
                          actual_neural_s=native_clock.neural_s,
                          actual_physical_s=native_clock.physical_s)
        finalization_began = clock()
        if session is None and "session" in holder:
            session = holder["session"]
            ledger["owned_session_recovered"] = True
        for label, callback in (("native_detach", detach),
                                ("control_journal_close", journal.close if journal else None)):
            if callback:
                try:
                    callback()
                except Exception as error:  # noqa: BLE001 - retain cleanup failures
                    fail(label, error)
        if deps is not None and native_clock is not None and "data" in locals():
            try:
                for failure in deps.save_raw(output, raw, data, native_clock):
                    fail("raw_" + failure["artifact"], RuntimeError(failure["error"]))
            except Exception as error:  # noqa: BLE001 - preserve first failure
                fail("raw_publication", error)
        if session is not None and first_error is not None:
            try:
                deps.save_state(output / "partial-final", session)
            except Exception as error:  # noqa: BLE001 - preserve partial state
                fail("partial_whole", error)
                try:
                    deps.save_partial(output / "partial-components", session)
                except Exception as secondary:  # noqa: BLE001 - keep primary error
                    fail("partial_components", secondary)
        if session is not None:
            ledger["close_requested"] = True
            try:
                session.close()
                ledger["close_returned"] = True
            except Exception as error:  # noqa: BLE001 - keep science evidence
                fail("owned_close", error)
        if "baseline" in locals():
            try:
                check_resources(meter())
            except Exception as error:  # noqa: BLE001 - terminal resource accounting
                fail("terminal_resources", error)
        if first_error is None and deps is not None:
            try:
                if (ledger["controls_completed"] != 300
                        or ledger["neural_device_confirmed"] != 6000
                        or ledger["physics_steps_completed"] != 15000):
                    raise ValueError("Exact E327 one-body cost required")
                neutral = evaluate_saved(e323.safe(e323.E318, root),
                                         data["curve"], data["startup"])
                candidate = evaluate_saved(output, data["curve"], data["startup"])
                result["saved_neutral"] = neutral
                result["actual_body"] = candidate
                result["neutral_metrics"] = neutral["shape"]
                result["neutral_summary"] = neutral["summary"]
                result["metrics"] = candidate["shape"]
                result["short"] = candidate["short"]
                result["summary"] = candidate["summary"]
                result["shape_delta"] = float(candidate["shape"]["shape_loss"]
                                              - neutral["shape"]["shape_loss"])
                engineering = bool(ledger.get("clock_valid") and
                                   not errors and len(ledger["construction"]) == 1 and
                                   len(ledger["body_construction"]) == 1)
                gates, supported = classify(candidate, neutral,
                                            engineering_valid=engineering)
                result["gates"] = gates
                result["constrained_body_supported"] = supported
                result["classification"] = "supported_for_candidate_review" if supported else "rejected"
                result["run_valid"] = engineering
            except Exception as error:  # noqa: BLE001 - measured evaluation failure
                fail("evaluation", error)
        try:
            files = {p.relative_to(output).as_posix():
                     {"sha256": e323.sha256(p), "bytes": p.stat().st_size}
                     for p in output.rglob("*") if p.is_file()
                     and p.name not in ("result.json", "advance-ledger.json",
                                        "output-manifest.json", "publication-failure.json")
                     and not p.name.endswith(".pending.json")}
            if sum(v["bytes"] for v in files.values()) > LIMITS["output_bytes"]:
                raise OSError("E327 output cap exceeded")
            writer(output / "output-manifest.json", {
                "schema": "flylab.m3.e327.output-manifest.v1", "files": files,
                "mutable_metadata_excluded": ["result.json", "advance-ledger.json",
                                              "publication-failure.json"]})
        except Exception as error:  # noqa: BLE001 - terminal manifest failure
            fail("output_manifest", error)
        ledger["publication_boundary"] = "before_terminal_metadata_commit"
        ledger["finalization_wall_s"] = clock() - finalization_began
        ledger["elapsed_s"] = clock() - began
        if ledger["finalization_wall_s"] > LIMITS["finalization_reserve_s"]:
            fail("finalization_deadline", TimeoutError("E327 finalization cap exceeded"))
        if ledger["elapsed_s"] >= LIMITS["internal_wall_s"]:
            fail("internal_deadline", TimeoutError("E327 internal cap exceeded"))
        result.update(first_error=first_error, errors=errors.copy(),
                      cost=copy.deepcopy(ledger))
        if first_error is not None:
            result.update(run_valid=False, constrained_body_supported=False)
        for name, value in (("advance-ledger.json", ledger), ("result.json", result)):
            try:
                writer(output / name, value)
            except Exception as error:  # noqa: BLE001 - independent receipts
                fail("terminal_" + name, error)
        if first_error is not None:
            result.update(run_valid=False, constrained_body_supported=False,
                          first_error=first_error, errors=errors.copy(),
                          cost=copy.deepcopy(ledger))
            try:
                writer(output / "publication-failure.json", result)
                writer(output / "result.json", result)
                writer(output / "advance-ledger.json", ledger)
            except Exception as error:  # noqa: BLE001 - preserve invalid receipt
                fail("failure_journal", error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("freeze-sha256", "review-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = run(args.freeze, args.freeze_sha256, args.review,
                 args.review_sha256, args.output)
    print(json.dumps(_plain(result), allow_nan=False), flush=True)
    if not result["run_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
