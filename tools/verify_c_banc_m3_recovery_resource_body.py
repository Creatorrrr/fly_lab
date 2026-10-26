"""E301 recovery resource body gates, using the approved E291 executor."""
from __future__ import annotations

# Preserve the first scientific exception and all publication/cleanup failures.
# ruff: noqa: BLE001
import argparse
import copy
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import analyze_c_banc_m3_recovery_factor_saved as diagnostic
from tools import verify_c_banc_m3_core_afferent_resource_body as base

runner = base.prior.prior
OWNED = ("tools/verify_c_banc_m3_recovery_resource_body.py", "tests/test_c_banc_recovery_resource_body.py")
PRODUCERS = ("tools/banc_m3_recovery_resource_synapses.py", "tests/test_c_banc_recovery_resource_synapses.py",
             "tools/banc_m3_recovery_resource_trial.py", "tests/test_c_banc_recovery_resource_trial.py")
SOURCE_FILES = tuple(dict.fromkeys((*OWNED, *PRODUCERS, *base.SOURCE_FILES, *diagnostic.SOURCE_FILES)))
PATHS = base.PATHS
BASE_KEYS = tuple(key for key in base.BASE_KEYS if key != "neutral_network_identity")
LIMITS, STARTUP_LIMITS = copy.deepcopy(base.LIMITS), copy.deepcopy(base.STARTUP_LIMITS)
FREEZE_SCHEMA = "flylab.m3.e301.recovery-resource-body.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e301.recovery-resource-body.review.v1"
BASE_FREEZE_PATH = "verification/banc-natural-gait-m3-20260919/e296-core-afferent-resource-body-freeze.json"
OUTPUT_PATH = "verification/banc-natural-gait-m3-20260919/e301-recovery-resource-body"
ROSTER_PATH = "verification/banc-natural-gait-m3-20260919/e296-core-afferent-resource-roster.npz"
ROSTER_SHA = "22e04433d8d9e14e89b5a2d575dfdaa68d3e21a9d8fbfb0c5a08c8ea53d7a938"
safe = base.safe
verify_spec = base.verify_spec


def engine_api():
    from tools import banc_m3_recovery_resource_synapses
    return banc_m3_recovery_resource_synapses


def wrapper_api():
    from tools import banc_m3_recovery_resource_trial
    return banc_m3_recovery_resource_trial


def producer_approvals(freeze, pins):
    expected = {name: pins[name] for name in PRODUCERS}
    for alias, names in (("e299_source_review", PRODUCERS[:2]), ("e300_source_review", PRODUCERS[2:]),
                         ("recovery_tiny_qualification", PRODUCERS)):
        receipt = verify_spec(freeze[alias])
        if (receipt.get("approved") is not True or receipt.get("blocking_defects") != []
                or receipt.get("source_sha256") != {name: expected[name] for name in names}):
            raise ValueError("Current E299/E300 source and actual tiny qualification required: " + alias)


def base_lineage(freeze):
    if freeze["base_freeze"]["path"] != BASE_FREEZE_PATH:
        raise ValueError("Literal approved E296 freeze required")
    old, review, audit = (verify_spec(freeze[name]) for name in ("base_freeze", "base_review", "base_original_audit"))
    if (old.get("schema") != base.FREEZE_SCHEMA or old.get("diagnostic_id") != "E296" or old.get("owned_files") != list(base.OWNED)
            or old.get("policy") != base.POLICY or set(old.get("paths", {})) != PATHS
            or review.get("schema") != base.REVIEW_SCHEMA or review.get("approved") is not True or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze["base_freeze"]["sha256"] or set(review.get("source_sha256", {})) != set(base.OWNED)
            or audit.get("schema") != "flylab.m3.e296.root-original-raw-audit.v1" or audit.get("pass") is not True
            or audit.get("run_valid") is not True or any(audit.get(flag) is not False for flag in
                ("actual_startup_pass", "short_gait_pass", "candidate_adopted", "F1_F8_complete", "F6_exposed"))):
        raise ValueError("Approved validly rejected E296 actual provenance required")
    prior_base, inherited = base.base_lineage(old)
    base.core_lineage(old, prior_base)
    for key in base.BASE_KEYS:
        if old.get(key) != prior_base[key]:
            raise ValueError("Original E296 inherited context differs: " + key)
    pins = base.pin_groups(old["input_sha256"], old["reused_source_sha256"], old["preservation_sha256"],
                           review["source_sha256"], {old["design"]: old["design_sha256"]})
    if not set(base.SOURCE_FILES) <= set(pins) or any(pins.get(name) != digest for name, digest in inherited.items() if name != prior_base["design"]):
        raise ValueError("Complete frozen E296 execution sources required")
    arm = audit["arms"]["candidate"]
    if arm.get("controls") != 100 or arm.get("actual_neural_ticks") != 2000 or arm.get("native_samples") != 5000 or arm.get("core_afferent_resource_exactly_one") is not True:
        raise ValueError("Actual healthy100-control protected E296 audit required")
    return old, pins


def diagnostic_lineage(freeze, old):
    evidence, result, audit = (verify_spec(freeze[name]) for name in ("e298_freeze", "e298_result", "e298_root_audit"))
    if (evidence.get("schema") != diagnostic.FREEZE_SCHEMA or evidence.get("owned_files") != list(diagnostic.OWNED)
            or result.get("schema") != "flylab.m3.e298.recovery-factor-saved.result.v1" or result.get("analysis_valid") is not True
            or result.get("current_proxy_nontrivial") is not True
            or audit.get("schema") != "flylab.m3.e298.root-saved-audit.v1" or audit.get("pass") is not True
            or audit.get("current_proxy_nontrivial") is not True):
        raise ValueError("Approved actual E298 saved diagnostic lineage required")
    for receipt in (result, audit):
        if any(receipt.get(flag) is not False for flag in ("candidate_selected", "F1_F8_complete", "F6_exposed")):
            raise ValueError("E298 diagnostic must remain unpromoted")
    if result.get("causal_claim") is not False:
        raise ValueError("No causal E298 claim allowed")
    pins = base.pin_groups(evidence["source_sha256"], evidence["input_sha256"], {evidence["design"]: evidence["design_sha256"]})
    if not set(diagnostic.SOURCE_FILES) <= set(pins) or set(evidence.get("inputs", {})) != set(diagnostic.INPUT_NAMES):
        raise ValueError("Complete E298 actual source/input pins required")
    name = evidence["inputs"]["protected_roster"]
    if name != old["paths"]["protected_selection"] or pins.get(name) != old["input_sha256"][name]:
        raise ValueError("E298 must use the identical E296 protected roster")
    return pins


def gate(freeze_path, freeze_sha, review_path, review_sha, output):
    if runner.protocol.sha256(freeze_path) != freeze_sha or runner.protocol.sha256(review_path) != review_sha:
        raise ValueError("E301 freeze/review SHA mismatch")
    freeze, review = (json.loads(Path(path).read_text(encoding="utf-8")) for path in (freeze_path, review_path))
    if (freeze.get("schema") != FREEZE_SCHEMA or freeze.get("diagnostic_id") != "E301" or freeze.get("owned_files") != list(OWNED)
            or set(freeze.get("paths", {})) != PATHS or freeze.get("policy") != base.POLICY
            or freeze.get("candidate_adopted") is not False or freeze.get("output") != OUTPUT_PATH
            or freeze["paths"]["protected_selection"] != ROSTER_PATH or freeze["input_sha256"].get(ROSTER_PATH) != ROSTER_SHA):
        raise ValueError("Exact E301 protected-policy freeze required")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Approved matching E301 source review required")
    old, old_pins = base_lineage(freeze)
    diagnostic_lineage(freeze, old)
    for key in BASE_KEYS:
        if freeze.get(key) != old[key]:
            raise ValueError("Original E296 context differs: " + key)
    if freeze["paths"] != old["paths"]:
        raise ValueError("Every original E296 input path must remain unchanged")
    for name, digest in old["input_sha256"].items():
        if freeze["input_sha256"].get(name) != digest:
            raise ValueError("Original E296 input bytes differ")
    pins = base.pin_groups(freeze["input_sha256"], freeze["reused_source_sha256"], freeze["preservation_sha256"],
                           review["source_sha256"], {freeze["design"]: freeze["design_sha256"]})
    if not set(SOURCE_FILES) <= set(pins) or any(pins.get(name) != digest for name, digest in old_pins.items() if name != old["design"]):
        raise ValueError("All current E301 actual inherited source pins required")
    producer_approvals(freeze, pins)
    for alias, name in freeze["paths"].items():
        path = safe(name)
        files = [file for file in path.rglob("*") if file.is_file()] if path.is_dir() else [path]
        if not files or any(file.relative_to(ROOT).as_posix() not in freeze["input_sha256"] for file in files):
            raise ValueError("Every original input byte must be pinned: " + alias)
    for name, version in freeze["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Frozen E301 package differs")
    NativeDependencies().load(freeze)
    if output.resolve() != safe(OUTPUT_PATH) or output.exists() or not output.parent.is_dir():
        raise ValueError("Fresh fixed E301 output required")
    if shutil.disk_usage(output.parent).free < LIMITS["minimum_free_disk_bytes"]:
        raise OSError("E301 disk reserve unavailable")
    return freeze


def endpoint_factors(raw, data, arm):
    """Completed raw endpoint coefficients, never measurements of RK stages."""
    roster, ids = data["roster"], raw["ids"]
    columns = diagnostic.prior.prior.id_columns(ids, roster["source_ids"])
    records = [row for row in raw["activity"] if row.get("complete") is True]
    if not records:
        raise ValueError("No complete endpoint available")
    controls = np.asarray([row["control_tick"] for row in records], np.int64)
    network = np.asarray([row["network_tick"] for row in records], np.int64)
    physical = np.asarray([row["physics_time_s"] for row in records], np.float64)
    runner.layer.require_equal(controls, np.arange(len(controls), dtype=np.int64), "Completed endpoint prefix")
    runner.layer.require_equal(network, controls * 20, "Endpoint neural clock")
    if not np.isfinite(physical).all() or np.any(np.abs(physical - controls * .005) >= 2e-8):
        raise ValueError("Endpoint physical clock differs")
    u = roster["neutral_utilization"] if arm == "neutral" else roster["utilization"]
    tau = roster["recovery_tau"]
    active = u > 0
    factors = []
    for row in records:
        runner.validate_activity(row, len(ids), len(roster["source_ids"]))
        rate, resource = row["rate"][columns], row["synaptic_resource"]
        if np.any(rate[active] < 0) or not np.all(resource[~active] == 1):
            raise ValueError("Active nonnegative rate and exempt resource one required")
        g = np.minimum(1., resource * (1 + (tau * u) * rate.astype(np.float64))).astype(np.float32)
        if not np.isfinite(g).all() or np.any(g < resource.astype(np.float32)) or np.any(g > 1) or not np.all(g[~active] == 1):
            raise ValueError("Finite bounded endpoint recovery coefficient required")
        factors.append(g)
    exempt = (roster["weights32"] < 0) | np.isin(roster["post_ids"], data["core_ids"])
    if not np.all(u[roster["edge_source_slots"][exempt]] == 0):
        raise ValueError("Core-afferent/negative source exemption required")
    g32 = np.stack(factors)
    return {"source_ids": roster["source_ids"].copy(), "control_tick": controls, "network_tick": network,
            "physics_time_s": physical, "g32": g32}, {"derived_from_completed_activity": True, "RK_stage_measurements": False,
            "active_sources": int(active.sum()), "factor_nontrivial": bool(np.any(g32[:, active] < 1)), "selection_criterion": False}


class NativeDependencies(runner.NativeDependencies):
    def __init__(self, *, engine=None, wrapper=None, base_loader=None, construction=None):
        self._engine, self._wrapper = engine, wrapper
        self._base_loader = base.NativeDependencies().load if base_loader is None else base_loader
        self._construction = runner.short.construction_with_body_count if construction is None else construction
        self.resource_faults = []

    @property
    def engine(self):
        return engine_api() if self._engine is None else self._engine

    @property
    def wrapper(self):
        return wrapper_api() if self._wrapper is None else self._wrapper

    def load(self, freeze):
        if freeze["base_freeze"]["path"] != BASE_FREEZE_PATH:
            raise ValueError("Literal E296 validated data load required")
        old = verify_spec(freeze["base_freeze"])
        if old.get("schema") != base.FREEZE_SCHEMA or old.get("diagnostic_id") != "E296":
            raise ValueError("Approved original E296 freeze required")
        data = self._base_loader(old)
        roster, selection = data["roster"], data["selection"]
        self.wrapper.validate_selection_metadata(selection, 158706, roster["motor_ids"])
        for arm, u in (("neutral", roster["neutral_utilization"]), ("candidate", roster["utilization"])):
            metadata = self.engine.resource_metadata(data["saved"]["network"]["model_hash"], {**selection, "utilization": u})
            self.engine.checked_resource_metadata(metadata)
            identity = self.engine.resource_identity(metadata)
            if identity != freeze[arm + "_network_identity"] or identity == old[arm + "_network_identity"]:
                raise ValueError("Separate new recovery engine identity required: " + arm)
        data.update(freeze=freeze, core_ids=np.asarray(data["e296_lineage"]["core_ids"], np.int64))
        return data

    def construct(self, arm, data, ledger, persist, on_return):
        def callback():
            owned = self.wrapper.MonotoneRecruitmentTrial.from_checkpoint(data["graph"], copy.deepcopy(data["saved"]))
            try:
                selection = data["selection"]
                trial = self.wrapper.RecoveryResourceTrial(owned, selection["edge_ids"], selection["premotor_ids"],
                    data["roster"]["neutral_utilization"] if arm == "neutral" else selection["utilization"], selection["recovery_tau"])
                trial._e291_arm = arm
                return trial
            except Exception as error:
                self.wrapper.cleanup_owned(owned, error)
                raise
        return self._construction(arm, callback, ledger, persist, on_return)

    def validate_initial_original(self, state):
        return runner.e288.validate_zero_checkpoint(state)

    def initial(self, session, data, arm):
        try:
            saved = session.snapshot()
        except Exception as error:
            self.record_fault(session, arm, session.control_tick, error)
            raise
        changes, expected_changes = [], []
        actual = runner.prior.checkpoint_representation(self.wrapper.remove_resource_layer(saved), changes)
        expected = runner.prior.checkpoint_representation(data["saved"], expected_changes)
        runner.layer.require_equal(actual, expected, "Exact whole original recovery initial state")
        self.validate_initial_original(actual)
        if (session.network.identity != data["freeze"][arm + "_network_identity"] or session.adapter.identity != data["freeze"]["adapter_identity"]
                or session.control_tick != 0 or session.network.tick != 0 or abs(session.body.physics_time()) > 2e-8
                or session.network.device != "cuda" or session.network.cuda_implementation != "packed" or session.network.interval_steps != 20):
            raise ValueError("Exact original CUDA recovery clocks and identities required")
        runner.layer.require_equal(saved["network"]["synaptic_resource"], np.ones(len(data["source_ids"]), np.float64), "Initial resource one")
        return {"storage_representation_exact": True, "actual_representation_changes": changes, "expected_representation_changes": expected_changes,
                "numeric_tolerance": 0, "array_coercions": 0, "excluded_fields": []}

    def compare_neutral(self, raw, data):
        prefix = self.compare_neutral_prefix(raw, data)
        endpoint = runner.compare_neutral_endpoint(self.wrapper.remove_resource_layer(raw["neutral_end_snapshot"]), data["baseline_end"], data["saved"]["wall_s"])
        for row in raw["activity"]:
            runner.layer.require_equal(row["synaptic_resource"], np.ones(len(data["source_ids"]), np.float64), "Neutral resource one")
        return {**prefix, **endpoint}

    def compare_neutral_prefix(self, raw, data):
        return runner.prior.NativeDependencies.compare_neutral(self, raw, data)

    def record_fault(self, session, arm, index, error=None):
        # E299's getter copies device flags to host. Successful public readouts
        # already validate the merged packet; only consume exception evidence.
        if error is None:
            return
        flags = getattr(error, "resource_fault_flags", None)
        if flags is None:
            flags = getattr(error.__cause__, "resource_fault_flags", None)
        if flags is None:
            return
        try:
            values = np.asarray(flags)
            if not np.any(values):
                return
            record = {"arm": arm, "attempted_control": index, "actual_neural_tick": int(session.network.tick),
                      "actual_control_tick": int(session.control_tick), "flags": values.tolist(),
                      "classification": "neural_resource_validity_failure", "original_error": {"type": type(error).__name__, "message": str(error)}}
            self.resource_faults.append(record)
        except Exception as secondary:
            error.add_note("E301 fault evidence readout: " + repr(secondary))

    def advance(self, session, arm, index, clock, journal):
        try:
            event = super().advance(session, arm, index, clock, journal)
            return event
        except Exception as error:
            self.record_fault(session, arm, index, error)
            raise

    def capture(self, session, raw):
        try:
            return super().capture(session, raw)
        except Exception as error:
            self.record_fault(session, session._e291_arm, session.control_tick, error)
            raise

    def save_state(self, path, session):
        try:
            return super().save_state(path, session)
        except Exception as error:
            self.record_fault(session, session._e291_arm, session.control_tick, error)
            try:
                self.write(path.parent / "resource-faults.json", {"events": self.resource_faults})
            except Exception as secondary:
                error.add_note("Secondary E301 fault publication: " + repr(secondary))
            raise

    def save_raw(self, folder, raw, data, clock):
        failures = super().save_raw(folder, raw, data, clock)
        for name, callback in (("recovery-factor-endpoints.npz", lambda: self.save_endpoints(folder, raw, data)),
                               ("resource-faults.json", lambda: self.write(folder / "resource-faults.json", {"events": self.resource_faults}))):
            try:
                callback()
            except Exception as error:
                failures.append({"artifact": name, "error": repr(error)})
        return failures

    def save_endpoints(self, folder, raw, data):
        arrays, summary = endpoint_factors(raw, data, folder.name)
        np.savez_compressed(folder / "recovery-factor-endpoints.npz", **arrays)
        self.write(folder / "recovery-factor-endpoints.json", summary)

    def write(self, path, value):
        if path.name == "source-manifest.json":
            value = {**value, "schema": "flylab.m3.e301.sources.v1", "diagnostic_id": "E301",
                "owned_sha256": {name: runner.protocol.sha256(ROOT / name) for name in OWNED},
                "source_sha256": {**value.get("source_sha256", {}), **{name: runner.protocol.sha256(ROOT / name) for name in SOURCE_FILES if name not in OWNED}},
                "executed_body_runner": "tools.verify_c_banc_m3_recurrent_resource_body.execute"}
        return runner.NativeDependencies.write(self, path, value)


def execute(freeze, output, *, deps=None, budget=None, cleanup=None):
    if freeze.get("diagnostic_id") != "E301":
        raise ValueError("E301 diagnostic identity required")
    return runner.execute(freeze, output, deps=NativeDependencies() if deps is None else deps, budget=budget, cleanup=cleanup)


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    freeze = gate(freeze_path, freeze_sha, review_path, review_sha, output)
    import cupy
    import torch
    from threadpoolctl import threadpool_limits
    previous_threads, previous_env, limiter, cleaned = torch.get_num_threads(), {}, None, False
    def cleanup():
        nonlocal cleaned
        if cleaned:
            return
        cleaned = True
        failures = []
        for callback in ((lambda: limiter.restore_original_limits()) if limiter else lambda: None, lambda: torch.set_num_threads(previous_threads)):
            try:
                callback()
            except Exception as error:
                failures.append(error)
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if failures:
            for error in failures[1:]:
                failures[0].add_note("Secondary E301 cleanup: " + repr(error))
            raise failures[0]
    try:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            previous_env[name] = os.environ.get(name)
            os.environ[name] = "1"
        torch.set_num_threads(1)
        limiter = threadpool_limits(limits=1)
        deps = NativeDependencies()
        deps.gpu_free_before, _ = cupy.cuda.runtime.memGetInfo()
        result, error = execute(freeze, output, deps=deps, cleanup=cleanup)
        return 0 if error is None and result["run_valid"] else 1
    finally:
        cleanup()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("freeze", "source-review", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    for key in ("freeze-sha256", "source-review-sha256"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args(argv)
    return run(args.freeze, args.freeze_sha256, args.source_review, args.source_review_sha256, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
