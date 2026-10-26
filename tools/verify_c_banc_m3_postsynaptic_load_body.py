"""E313 fixed lambda0 qualification and lambda1 postsynaptic body diagnostic."""
from __future__ import annotations

# ruff: noqa: BLE001
import argparse
import copy
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import banc_m3_postsynaptic_load_trial as layer
from tools import verify_c_banc_m3_resource_continuation as e310

e309, e307, e306, e301, e305, runner = e310.e309, e310.e307, e310.e306, e310.e301, e310.e305, e310.runner
OWNED = ("tools/banc_m3_postsynaptic_load_trial.py", "tests/test_c_banc_postsynaptic_load_trial.py",
         "tools/verify_c_banc_m3_postsynaptic_load_body.py", "tests/test_c_banc_postsynaptic_load_body.py")
ENGINE_FILES = ("tools/banc_m3_postsynaptic_load.py", "tests/test_c_banc_postsynaptic_load.py")
SOURCE_FILES = tuple(dict.fromkeys((*OWNED, *ENGINE_FILES, *e310.SOURCE_FILES)))
FREEZE_SCHEMA = "flylab.m3.e313.postsynaptic-load-body.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e313.postsynaptic-load-body.review.v1"
RESULT_SCHEMA = "flylab.m3.e313.postsynaptic-load-body.result.v1"
DESIGN = "docs/BANC_M3_E313_POSTSYNAPTIC_BODY_DESIGN_20260928.md"
DESIGN_SHA = "316b351259430d845ee924284a223b9f63f4399f246b7da80d4903e98ed422aa"
ENGINE_DESIGN = "docs/BANC_M3_E312_POSTSYNAPTIC_LOAD_DESIGN_20260928.md"
ENGINE_DESIGN_SHA = "5b80252d83e926eac7163eddae25104535444b8e7e1cf89fd9c98a5f4c42d2ff"
OUTPUT_PATH = "verification/banc-natural-gait-m3-20260919/e313-postsynaptic-load-body"
PARAMETER_PATH = "verification/banc-natural-gait-m3-20260919/e313-load-model-parameters.npz"
PARAMETER_PREPARATION_PATH = "verification/banc-natural-gait-m3-20260919/parent-e313-load-parameters-preparation.json"
LIMITS = {**copy.deepcopy(e306.LIMITS), "internal_wall_s": 115, "science_wall_s": 100, "finalization_reserve_s": 15}
SUPERVISOR_WALL_S = 130
STARTUP_LIMITS = copy.deepcopy(e306.STARTUP_LIMITS)
LOAD_LAMBDA = {"neutral": 0., "candidate": 1.}
NEUTRAL_REFERENCES = {"neural": e307.OUTPUT_PATH + "/neutral/first-control-neural",
    **{key: e307.OUTPUT_PATH + "/neutral/" + filename for key, filename in
       (("trace", "trace.json"), ("native", "native-contact.npz"), ("recruitment", "native-recruitment-state.npz"), ("controls", "control-raw.npz"))}}
MATCHED_SHORT_PATH = e310.OUTPUT_PATH + "/neutral/arm-result.json"
E312_RECEIPTS = ("source_review", "qualification", "whole_integration")
safe = e306.safe


def validate_qualification(records, pins):
    review = records["source_review"]
    source = review.get("source_sha256", {})
    if review.get("approved") is not True or review.get("blocking_defects") != [] or set(source) != set(ENGINE_FILES):
        raise ValueError("Approved exact E312 source pair required")
    e306.require_pins(pins, source)
    qualification = records["qualification"]
    integration = records["whole_integration"]
    if (review.get("schema") != "flylab.m3.e312.root-source-review.v1"
            or qualification.get("schema") != "flylab.m3.e312.postsynaptic-load-tiny-qualification.v1"
            or qualification.get("approved") is not True or qualification.get("source_sha256") != source
            or any(qualification.get("checks", {}).get(key) is not True for key in ("cpu_pass", "cuda_pass", "independent_oracle_pass", "disk_future_exact"))
            or qualification.get("original_calls") != 0 or qualification.get("native_body_constructors") != 0
            or qualification.get("F1_F8_complete") is not False or qualification.get("F6_exposed") is not False
            or not isinstance(qualification.get("evidence_sha256"), dict) or not qualification["evidence_sha256"]
            or integration.get("schema") != "flylab.m3.e313.root-tiny-integration.v1" or integration.get("approved") is not True
            or integration.get("actual_tiny_neural") is not True or integration.get("native_body") is not False
            or not isinstance(integration.get("integration_path"), str)
            or not isinstance(integration.get("integration_sha256"), str)):
        raise ValueError("Actual approved E312 CPU/CUDA/oracle/future and fake-whole neural integration required")
    e306.require_pins(pins, qualification["evidence_sha256"])
    raw = e305.receipt({"path": integration["integration_path"], "sha256": integration["integration_sha256"]}, pins)
    if (raw.get("schema") != "flylab.m3.e313.real-neural-fake-body.v1" or raw.get("pass") is not True
            or raw.get("original_calls") != 0 or raw.get("native_body_calls") != 0
            or raw.get("exact_nonzero_disk_future") is not True):
        raise ValueError("Exact actual tiny neural and fake whole disk/future evidence required")
    if records["whole_integration"].get("wrapper_source_sha256") != {name: pins[name] for name in OWNED[:2]}:
        raise ValueError("Exact actual tiny whole integration wrapper source pair required")


def matched_lineage(freeze, pins):
    specs = freeze["e310_lineage"]
    required = ("freeze", "review", "result", "result_sha256", "source_manifest", "output_manifest", "root_audit", "correction", "neutral_arm_result")
    if set(specs) != set(required):
        raise ValueError("Complete E310 original/correction lineage required")
    paths = {"freeze": e310.OUTPUT_PATH + "-freeze.json", "review": e310.OUTPUT_PATH + "-source-review.json",
             **{key: e310.OUTPUT_PATH + "/" + filename for key, filename in
                (("result", "result.json"), ("result_sha256", "result-sha256.json"), ("source_manifest", "source-manifest.json"), ("output_manifest", "output-manifest.json"))},
             "root_audit": "verification/banc-natural-gait-m3-20260919/parent-e310-original-raw-audit.json",
             "correction": "verification/banc-natural-gait-m3-20260919/parent-e310-derived-contrast-correction.json", "neutral_arm_result": MATCHED_SHORT_PATH}
    if any(specs[key]["path"] != path for key, path in paths.items()):
        raise ValueError("Literal E310 frozen comparison paths required")
    records = {key: e305.receipt(spec, pins) for key, spec in specs.items()}
    old, result, audit = records["freeze"], records["result"], records["root_audit"]
    if (old.get("schema") != e310.FREEZE_SCHEMA or old.get("resource_limits") != e310.LIMITS
            or result.get("schema") != e310.RESULT_SCHEMA or result.get("run_valid") is not True or result.get("publication_status") != "complete"
            or audit.get("pass") is not True or audit.get("diagnostic_run_valid") is not True
            or audit.get("result_sha256") != specs["result"]["sha256"] or audit.get("output_manifest_sha256") != specs["output_manifest"]["sha256"]
            or records["result_sha256"].get("sha256") != specs["result"]["sha256"] or records["neutral_arm_result"] != result["arms"]["neutral"]):
        raise ValueError("Original valid audited E310 neutral comparison required")
    if any(result.get(k) is not False for k in ("candidate_adopted", "F1_F8_complete", "F2_complete", "F6_exposed", "next_separate_10s_eligible")):
        raise ValueError("Unpromoted E310 diagnostic must remain rejected")
    # E310's actually executed owned pair predates its approved reporting R1.
    # Those two old bytes live at archived paths, while current same-name
    # source pins point to R1. All other inherited paths remain exact.
    e306.require_pins(pins, old["input_sha256"],
        {name: digest for name, digest in old["reused_source_sha256"].items() if name not in e310.OWNED},
        old["preservation_sha256"])
    original_sources = records["source_manifest"]["owned_sha256"]
    if (set(original_sources) != set(e310.OWNED)
            or {name: old["reused_source_sha256"].get(name) for name in e310.OWNED} != original_sources
            or records["review"].get("source_sha256") != original_sources):
        raise ValueError("Actually executed E310 old owned bytes require frozen source/review/manifest agreement")
    archives = freeze["e310_executed_sources"]
    if set(archives) != set(e310.OWNED):
        raise ValueError("Actual prelaunch source pair archive required")
    for name, spec in archives.items():
        if spec["sha256"] != original_sources[name] or pins.get(spec["path"]) != spec["sha256"] or runner.protocol.sha256(safe(spec["path"])) != spec["sha256"]:
            raise ValueError("Exact actually executed E310 source archive required")
    for name, entry in records["output_manifest"]["files"].items():
        path = safe(e310.OUTPUT_PATH + "/" + name)
        if pins.get(path.relative_to(ROOT).as_posix()) != entry["sha256"] or runner.protocol.sha256(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError("Every original E310 output byte required")
    return records


def prepared_parameters(freeze, pins):
    spec = freeze["load_parameter_preparation"]
    if spec["path"] != PARAMETER_PREPARATION_PATH:
        raise ValueError("Literal independently prepared original threshold/tau receipt required")
    record = e305.receipt(spec, pins)
    if (record.get("schema") != "flylab.m3.e313.static-native-load-parameters.v1" or record.get("pass") is not True
            or record.get("output") != PARAMETER_PATH or record.get("output_sha256") != freeze["load_parameters"]["sha256"]
            or record.get("output_bytes") != safe(PARAMETER_PATH).stat().st_size
            or record.get("cells") != freeze["n"] or record.get("membrane_overrides") != 38
            or record.get("membrane_tau_exact_reconstruction") is not True
            or record.get("tau_is_membrane_not_adaptation") is not True
            or any(record.get(key) != 0 for key in ("original_calls", "model_constructors", "GraphStore_calls", "neural_body_physics_CUDA_solver_fit_calls"))):
        raise ValueError("Only audited original full membrane/threshold preparation required")
    e306.require_pins(pins, record["source_sha256"], record["input_sha256"])
    return record


def load_parameters(spec, count):
    """Hash-check binary native parameters without the JSON receipt reader."""
    if spec.get("path") != PARAMETER_PATH:
        raise ValueError("Literal native parameter NPZ required")
    path = safe(spec["path"])
    if runner.protocol.sha256(path) != spec.get("sha256"):
        raise ValueError("Native parameter NPZ SHA mismatch")
    params = runner.protocol.load_npz(path)
    if set(params) != {"threshold32", "tau32"} or any(
            v.dtype != np.float32 or v.shape != (count,)
            or not np.isfinite(v).all() or np.any(v <= 0) for v in params.values()):
        raise ValueError("Full positive original native threshold/tau32 required")
    return params


def gate(freeze_path, freeze_sha, review_path, review_sha, output):
    if runner.protocol.sha256(freeze_path) != freeze_sha or runner.protocol.sha256(review_path) != review_sha:
        raise ValueError("E313 freeze/review SHA mismatch")
    freeze, review = (json.loads(Path(p).read_text("utf8")) for p in (freeze_path, review_path))
    if (freeze.get("schema") != FREEZE_SCHEMA or freeze.get("diagnostic_id") != "E313" or freeze.get("owned_files") != list(OWNED)
            or freeze.get("resource_limits") != LIMITS or freeze.get("supervisor_wall_s") != SUPERVISOR_WALL_S
            or freeze.get("startup_limits") != STARTUP_LIMITS or freeze.get("load_lambda") != LOAD_LAMBDA
            or freeze.get("source_utilization") != 0 or freeze.get("neutral_references") != NEUTRAL_REFERENCES
            or freeze.get("load_parameters", {}).get("path") != PARAMETER_PATH or freeze.get("candidate_adopted") is not False
            or freeze.get("output") != OUTPUT_PATH or freeze.get("design") != DESIGN or freeze.get("design_sha256") != DESIGN_SHA):
        raise ValueError("Exact two-body fixed postsynaptic-load contract required")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Approved matching E313 source review required")
    pins = e301.base.pin_groups(freeze["input_sha256"], freeze["reused_source_sha256"], freeze["preservation_sha256"], review["source_sha256"],
        {DESIGN: DESIGN_SHA, ENGINE_DESIGN: ENGINE_DESIGN_SHA})
    if not set(SOURCE_FILES) <= set(pins):
        raise ValueError("Every actually reused source byte required")
    e310.e309_lineage(freeze, pins)
    if set(freeze["e312_lineage"]) != set(E312_RECEIPTS):
        raise ValueError("Complete E312 source/CPU/CUDA/whole qualifications required")
    records = {k: e305.receipt(spec, pins) for k, spec in freeze["e312_lineage"].items()}
    if (records["whole_integration"].get("source_review_sha256") != freeze["e312_lineage"]["source_review"]["sha256"]
            or records["whole_integration"].get("e312_qualification_sha256") != freeze["e312_lineage"]["qualification"]["sha256"]):
        raise ValueError("Exact integration producer bindings required")
    validate_qualification(records, pins)
    matched_lineage(freeze, pins)
    prepared_parameters(freeze, pins)
    for name, version in freeze["packages"].items():
        actual = platform.python_version() if name == "python" else importlib.metadata.version(name)
        if actual != version:
            raise ValueError("Pinned actual package differs")
    NativeDependencies().load(freeze)
    if output.resolve() != safe(OUTPUT_PATH) or output.exists() or not output.parent.is_dir():
        raise ValueError("Fresh literal E313 output required")
    if shutil.disk_usage(output.parent).free < LIMITS["minimum_free_disk_bytes"]:
        raise OSError("E313 disk reserve unavailable")
    return freeze


class NativeDependencies(e306.NativeDependencies):
    @property
    def wrapper(self):
        return layer if self._wrapper is None else self._wrapper

    def load(self, freeze):
        from flylab.c.storage import StateStore
        # This immutable loader validates old SSP provenance, not the new model.
        data = e306.NativeDependencies().load(freeze)
        # E306's inherited selection carries the E296 protected policy. The
        # E313 mechanism targets the complete original E289 edge roster.
        if freeze["input_sha256"].get(runner.ROSTER_PATH) != runner.ROSTER_SHA:
            raise ValueError("Exact original E289 full-roster input required")
        original_roster = runner.protocol.load_npz(safe(runner.ROSTER_PATH))
        selection = copy.deepcopy(runner.checked_roster(original_roster))
        selection["utilization"] = np.zeros_like(selection["utilization"])
        self.wrapper.validate_selection_metadata(selection, freeze["n"], data["motor_ids"])
        runner.layer.require_equal(selection["edge_ids"], data["selection"]["edge_ids"], "Inherited original selected edge IDs")
        data.update(selection=selection, roster=original_roster, source_ids=selection["source_ids"],
                    ids=np.unique(np.r_[selection["premotor_ids"], selection["motor_ids"]]).astype(np.int64))
        params = load_parameters(freeze["load_parameters"], freeze["n"])
        for arm, value in LOAD_LAMBDA.items():
            metadata = layer.network_api().load_metadata(freeze["source_network_identity"], selection, params["threshold32"], params["tau32"], value)
            layer.network_api().checked_load_metadata(metadata)
            if layer.network_api().load_identity(metadata) != freeze["load_network_identities"][arm]:
                raise ValueError("Computed distinct postsynaptic load identity differs")
        if len(set(freeze["load_network_identities"].values())) != 2 or any(identity in
                (freeze["neutral_network_identity"], freeze["source_network_identity"], freeze["candidate_network_identity"])
                for identity in freeze["load_network_identities"].values()):
            raise ValueError("Distinct new lambda0/lambda1 identities required")
        data.update(load_parameters=params, load_reference_neural=StateStore.load(safe(NEUTRAL_REFERENCES["neural"])),
            load_reference_rows=json.loads(safe(NEUTRAL_REFERENCES["trace"]).read_text("utf8"))["samples"],
            load_reference_native=runner.protocol.load_npz(safe(NEUTRAL_REFERENCES["native"])),
            load_reference_recruitment=runner.protocol.load_npz(safe(NEUTRAL_REFERENCES["recruitment"])),
            load_reference_controls=runner.protocol.load_npz(safe(NEUTRAL_REFERENCES["controls"])),
            matched_short=json.loads(safe(MATCHED_SHORT_PATH).read_text("utf8"))["short"])
        self.data = data
        return data

    def construct(self, arm, data, ledger, persist, on_return):
        def callback():
            base = self.wrapper.MonotoneRecruitmentTrial.from_checkpoint(data["graph"], copy.deepcopy(data["saved"]))
            try:
                for name, attr in (("threshold32", "threshold"), ("tau32", "tau")):
                    runner.layer.require_equal(getattr(base.network, attr).cpu().numpy(), data["load_parameters"][name], "Actual original " + name)
                s = data["selection"]
                session = self.wrapper.PostsynapticLoadTrial(base, s["edge_ids"], s["premotor_ids"], s["utilization"], s["recovery_tau"], LOAD_LAMBDA[arm])
                session._e291_arm = arm
                return session
            except Exception as error:
                self.wrapper.cleanup_owned(base, error)
                raise
        return self._construction(arm, callback, ledger, persist, on_return)

    def initial(self, session, data, arm):
        observed = session.snapshot()
        receipt = self.wrapper.require_saved_whole_equal(self.wrapper.remove_resource_layer(observed), data["saved"], "Entire original whole initial state")
        runner.e288.validate_zero_checkpoint(self.wrapper.remove_resource_layer(observed))
        network = observed["network"]
        runner.layer.require_equal(network["synaptic_resource"], np.ones(len(data["source_ids"]), np.float64), "Original selected source R1")
        runner.layer.require_equal(network["postsynaptic_load_state"], np.zeros(len(np.unique(data["selection"]["post_ids"])), np.float64), "Selected target h0")
        expected_metadata = layer.network_api().load_metadata(data["freeze"]["source_network_identity"], data["selection"],
            data["load_parameters"]["threshold32"], data["load_parameters"]["tau32"], LOAD_LAMBDA[arm])
        runner.layer.require_equal(network["postsynaptic_load"], expected_metadata, "Frozen actual postsynaptic load metadata")
        if (session.control_tick != 0 or session.network.tick != 0 or abs(session.body.physics_time()) > 2e-8
                or session.network.identity != data["freeze"]["load_network_identities"][arm] or session.adapter.identity != data["freeze"]["adapter_identity"]
                or session.network.device != "cuda" or session.network.cuda_implementation != "packed" or session.network.interval_steps != 20):
            raise ValueError("Actual new-model CUDA zero clocks and original adapter required")
        receipt.update(original_whole_exact=True, h_zero=True, R_one=True, load_lambda=LOAD_LAMBDA[arm])
        return receipt

    def capture(self, session, raw):
        e301.NativeDependencies.capture(self, session, raw)
        diagnostics = session.network.load_readout()
        raw.setdefault("load_diagnostics", []).append(copy.deepcopy(diagnostics))
        diagnostics["control_tick"] = session.control_tick
        raw["load_diagnostics"][-1]["control_tick"] = session.control_tick
        validate_diagnostic(diagnostics, session.network.tick, session.control_tick)
        expected_g = np.asarray(1. / (1. + LOAD_LAMBDA[session._e291_arm] * diagnostics["h"]), np.float32)
        runner.layer.require_equal(diagnostics["g"], expected_g, "Post-load endpoint factor")
        if session.control_tick == 1:
            state = raw["neutral_end_snapshot"]["network"] if session._e291_arm == "neutral" else session.network.snapshot()
            raw["ssp_first_control_neural"] = copy.deepcopy(state)
            raw["ssp_diagnostic_readouts"] = {"additional_neural_snapshots": int(session._e291_arm == "candidate"),
                "neutral_reused_whole_snapshot": session._e291_arm == "neutral"}
            if session._e291_arm == "neutral":
                e309.compare_neural_fields(state, self.data["load_reference_neural"])
                e309.compare_first_body_native(raw, self.data["load_reference_rows"], self.data["load_reference_native"])
                compare_neutral_readouts(raw, self.data)
                self.first_control_checks["neutral"] = {"ssp_neural_endpoint_exact": True, "neutral_prefix_valid": True, "load_metadata_separate": True, "candidate_first_neural_exact_required": False}
            else:
                self.first_control_checks["candidate"] = {"observed_first_neural": True, "neutral_first_neural_exact_claim": False}

    def compare_neutral(self, raw, data):
        from flylab.c.storage import StateStore
        saved = raw["neutral_end_snapshot"]
        StateStore.save(data["neutral_disk_path"], saved)
        receipt = self.wrapper.require_saved_whole_equal(StateStore.load(data["neutral_disk_path"]), saved, "Actual whole memory to disk")
        if self.first_control_checks.get("neutral", {}).get("neutral_prefix_valid") is not True:
            raise ValueError("Lambda0 exact SSP/u0 first interval required")
        return {**self.first_control_checks["neutral"], "whole_memory_disk_exact": True, "storage_comparison": receipt}

    def save_raw(self, folder, raw, data, clock):
        # E291 raw collectors remain unchanged; E301 source-R factors are not this mechanism.
        failures = runner.NativeDependencies.save_raw(self, folder, raw, data, clock)
        for name, callback in (("load-diagnostics.npz", lambda: save_diagnostics(folder / "load-diagnostics.npz", raw.get("load_diagnostics", []))),
            ("load-diagnostics.json", lambda: self.write(folder / "load-diagnostics.json", {"load_lambda": LOAD_LAMBDA[folder.name],
                "h_g_scope": "end-of-control endpoint", "gross_scope": "last SSPRK33 stage input before neuron overwrite; not endpoint", "gross_stage_arrays_preserved": True})),
            ("resource-faults.json", lambda: self.write(folder / "resource-faults.json", {"events": self.resource_faults})),
            ("first-control-neural", lambda: self.save_neural(folder / "first-control-neural", raw["ssp_first_control_neural"]))):
            if name == "first-control-neural" and "ssp_first_control_neural" not in raw:
                continue
            try:
                callback()
            except Exception as error:
                failures.append({"artifact": name, "error": repr(error)})
        return failures

    def startup(self, raw, data):
        result = runner.NativeDependencies.startup(self, raw, data)
        arrays = runner.native.native_arrays(raw["native"])
        upright, nonfoot = arrays["upright"], arrays["nonfoot_contact"]
        healthy = bool(len(upright) == 5000 and np.isfinite(upright).all() and np.min(upright) >= .9 and not np.any(nonfoot))
        result.update(native_health_pass=healthy, minimum_native_upright=float(np.min(upright)) if len(upright) else None,
                      nonfoot_contact_count=int(np.count_nonzero(nonfoot)), strict_E141_command_screen=result["strict_E141_startup_screen"])
        result["strict_E141_startup_screen"] = result["strict_E141_startup_screen"] and healthy
        return result

    def write(self, path, value):
        if path.name == "source-manifest.json":
            value = {**value, "schema": "flylab.m3.e313.sources.v1", "diagnostic_id": "E313",
                "owned_sha256": {name: runner.protocol.sha256(ROOT / name) for name in OWNED},
                "source_sha256": {**value.get("source_sha256", {}), **{name: runner.protocol.sha256(ROOT / name) for name in SOURCE_FILES if name not in OWNED}},
                "executed_body_runner": "tools.verify_c_banc_m3_postsynaptic_load_body.execute", "load_lambda": LOAD_LAMBDA, "source_utilization": 0}
        elif path.name == "trace.json":
            value = {**value, "controller": "original E150 body/controller with fixed SSP postsynaptic-load neural layer",
                     "load_lambda": LOAD_LAMBDA[path.parent.name], "source_utilization": 0}
        elif path.name == "arm-result.json":
            if value.get("validity") == "valid_recurrent_resource_body_diagnostic":
                value["validity"] = "valid_postsynaptic_load_body_diagnostic"
            value["diagnostic_id"] = "E313"
            value["load_lambda"] = LOAD_LAMBDA[path.parent.name]
        return runner.NativeDependencies.write(self, path, value)


def validate_diagnostic(row, tick, control):
    ids, h, g, gross = (row[key] for key in ("target_ids", "h", "g", "gross"))
    if (ids.dtype != np.int64 or ids.ndim != 1 or not len(ids) or np.any(ids[1:] <= ids[:-1])
            or h.dtype != np.float64 or h.shape != ids.shape or not np.isfinite(h).all() or np.any(h < 0) or np.any(h > 1)
            or g.dtype != np.float32 or g.shape != ids.shape or not np.isfinite(g).all() or np.any(g < .5) or np.any(g > 1)
            or gross.dtype != np.float32 or gross.shape != ids.shape or row["completed_tick"] != tick or tick != control * 20
            or row["gross_phase"] != "last-SSPRK33-stage-input"):
        raise ValueError("Actual finite bounded sorted postsynaptic diagnostic packet required")
    if row["gross_available"] is True:
        if not np.isfinite(gross).all() or np.any(gross < 0) or row["gross_tick"] != tick - 1:
            raise ValueError("Actual last-stage gross clock/nonnegative data required")
    elif row["gross_available"] is not False or control != 0 or row["gross_tick"] is not None:
        raise ValueError("Only initial reset may lack gross diagnostic")


def compare_neutral_readouts(raw, data):
    for i in (0, 1):
        for key in ("motor_rate32", "activation", "command", "tendon"):
            runner.layer.require_equal(raw["controls"][i][key], data["load_reference_controls"][key][i], "Lambda0 original SSP control " + key)
        for key in ("bank", "activation", "original_activation", "motor_rate"):
            runner.layer.require_equal(raw["recruitment"][i][key], data["load_reference_recruitment"][key][i], "Lambda0 original SSP recruitment " + key)


def save_diagnostics(path, records):
    if not records:
        np.savez_compressed(path, control_tick=np.empty(0, np.int64))
        return
    for row in records:
        runner.layer.require_equal(row["target_ids"], records[0]["target_ids"], "Fixed sorted targets")
    np.savez_compressed(path, target_ids=records[0]["target_ids"], **{key: np.asarray([row[key] for row in records]) for key in
        ("control_tick", "completed_tick", "h", "g", "gross", "gross_available")},
        gross_tick=np.asarray([-1 if row["gross_tick"] is None else row["gross_tick"] for row in records], np.int64),
        h_min=np.asarray([np.min(row["h"]) for row in records]), h_max=np.asarray([np.max(row["h"]) for row in records]),
        g_min=np.asarray([np.min(row["g"]) for row in records]), g_max=np.asarray([np.max(row["g"]) for row in records]))


def run_arm(arm, data, freeze, out, ledger, deps, budget):
    # E291 is correctly zero-origin for this protocol; no arm or global spoof.
    if arm == "neutral":
        data["neutral_disk_path"] = out / arm / "whole-memory-disk"
    return runner.run_arm(arm, data, freeze, out, ledger, deps, budget)


def execute(freeze, out, *, deps=None, budget=None, cleanup=None):
    deps = NativeDependencies() if deps is None else deps
    began = time.perf_counter()
    out.mkdir(exist_ok=False)
    ledger = {
        "schema": "flylab.m3.e313.ledger.v1",
        "arms": {},
        "construction": [],
        "body_construction": [],
        "original_attempts": 1,
        "retries": 0,
        "fit_calls": 0,
        "nonneutral_current_calls": 0,
    }
    result = {
        "schema": RESULT_SCHEMA,
        "validity": "incomplete",
        "publication_status": "pending",
        "candidate_adopted": False,
        "diagnostic_id": freeze.get("diagnostic_id", "E313"),
        "run_valid": False,
        "neutral_prefix_valid": False,
        "ssp_neural_endpoint_exact": False,
        "actual_startup_pass": False,
        "short_gait_pass": False,
        "adopted": False,
        "next_separate_10s_eligible": False,
        "natural_gait_complete": False,
        "F1_F8_complete": False,
        "arm_receipt_path": "tools.verify_c_banc_m3_postsynaptic_load_body.run_arm (delegates to tools.verify_c_banc_m3_recurrent_resource_body.run_arm)",
        "E272_executed": False,
        "F6_exposed": False,
        "F2_complete": False,
        "F3_F8_pass": False,
        "biological_claim": False,
        "ten_second_executed": False,
        "arms": {},
    }
    errors, primary = [], None

    def check():
        if time.perf_counter() - began >= LIMITS["science_wall_s"]:
            raise TimeoutError("E313 science deadline/reserve reached")
        if budget is not None:
            budget()
        elif hasattr(deps, "gpu_free_before"):
            import cupy
            import psutil

            memory = psutil.Process().memory_info()
            peak = getattr(memory, "peak_wset", memory.rss)
            free, _ = cupy.cuda.runtime.memGetInfo()
            added = max(0, deps.gpu_free_before - free)
            ledger["peak_rss_bytes"] = max(peak, ledger.get("peak_rss_bytes", 0))
            ledger["peak_added_gpu_bytes"] = max(
                added, ledger.get("peak_added_gpu_bytes", 0)
            )
            if (
                peak > LIMITS["host_peak_bytes"]
                or added > LIMITS["additional_gpu_peak_bytes"]
            ):
                raise MemoryError("E313 host/GPU cap exceeded")
            if shutil.disk_usage(out).free < LIMITS["minimum_free_disk_bytes"]:
                raise OSError("E313 output disk reserve reached")

    try:
        deps.write(
            out / "source-manifest.json",
            {
                "schema": "flylab.m3.e313.sources.v1",
                "input_sha256": freeze.get("input_sha256", {}),
                "source_sha256": freeze.get("reused_source_sha256", {}),
                "owned_sha256": {name: runner.protocol.sha256(ROOT / name) for name in OWNED},
            },
        )
        data = deps.load(freeze)
        neutral, primary = run_arm(
            "neutral", data, freeze, out, ledger, deps, check
        )
        result["arms"]["neutral"] = neutral
        result["neutral_prefix_valid"] = (
            primary is None
            and neutral.get("neutral_prefix", {}).get("neutral_prefix_valid") is True
            and neutral.get("neutral_prefix", {}).get("ssp_neural_endpoint_exact") is True
        )
        result["ssp_neural_endpoint_exact"] = result["neutral_prefix_valid"]
        if primary is None and not result["neutral_prefix_valid"]:
            raise ValueError("E305 SSP neutral endpoint required before candidate")
        if result["neutral_prefix_valid"]:
            candidate, primary = run_arm(
                "candidate", data, freeze, out, ledger, deps, check
            )
            result["arms"]["candidate"] = candidate
            result["actual_startup_pass"] = (
                primary is None
                and candidate.get("startup", {}).get("strict_E141_startup_screen")
                is True
            )
            if primary is None and candidate.get("completed_controls") == 300:
                result["short_adjudication"] = runner.assess_short(
                    data["baseline_short"], candidate.get("short", {})
                )
                result["short_gait_pass"] = result["short_adjudication"]["pass"]
            if primary is None and candidate.get("completed_controls") == 300:
                result["matched_contrast"] = e310.matched_contrast(e310.metric_summary(data["matched_short"]), e310.metric_summary(candidate["short"]))
            result["next_separate_10s_eligible"] = (
                result["actual_startup_pass"] and result["short_gait_pass"]
            )
        result["run_valid"] = primary is None
        result["validity"] = (
            "valid_postsynaptic_load_body_diagnostic"
            if primary is None
            else "invalid_or_incomplete"
        )
    except Exception as error:
        primary = error if primary is None else primary
        errors.append(
            {
                "stage": "execution",
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
    if cleanup is not None:
        try:
            cleanup()
        except Exception as error:
            primary = primary or error
            errors.append(
                {
                    "stage": "runtime cleanup",
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
    if primary is not None:
        result.update(
            run_valid=False,
            actual_startup_pass=False,
            short_gait_pass=False,
            next_separate_10s_eligible=False,
            validity="invalid_or_incomplete",
            first_error={"type": type(primary).__name__, "message": str(primary)},
        )
        errors.append(
            {
                "stage": "primary",
                "type": type(primary).__name__,
                "message": str(primary),
                "notes": getattr(primary, "__notes__", []),
            }
        )
    result["elapsed_s"] = time.perf_counter() - began
    result["cost"] = {
        "scientific_neural_ticks": sum(
            round(v.get("actual_neural_s", 0) / 0.00025)
            for v in ledger["arms"].values()
            if v.get("actual_neural_s") is not None
        ),
        "scientific_physics_steps": sum(
            round(v.get("actual_physical_s", 0) / 0.0001)
            for v in ledger["arms"].values()
            if v.get("actual_physical_s") is not None
        ),
        "unknown_cost_arms": [
            k
            for k, v in ledger["arms"].items()
            if v.get("actual_neural_s") is None or v.get("actual_physical_s") is None
        ],
        "construction_calls": len(ledger["construction"]),
        "known_executed_setup_ticks": sum(
            v.get("known_completed_setup_steps", 0) for v in ledger["construction"]
        ),
        "recorded_capture_setup_ticks": sum(
            v.get("recorded_capture_setup_steps", 0) for v in ledger["construction"]
        ),
        "known_settling_physics_steps": sum(
            v.get("known_completed_settling_steps", 0)
            for v in ledger["body_construction"]
        ),
        "unknown_setup_calls": [
            v.get("label")
            for v in ledger["construction"]
            if v.get("unknown_setup_steps", 0)
            or v.get("unknown_capture_setup_steps", 0)
        ],
        "unknown_body_construction_calls": [v.get("label") for v in ledger["body_construction"] if v.get("unknown_step_entries", 0)],
    }
    if (
        result["cost"]["scientific_neural_ticks"] > LIMITS["scientific_neural_ticks"]
        or result["cost"]["scientific_physics_steps"]
        > LIMITS["scientific_physics_steps"]
        or len(
            deps.sessions if hasattr(deps, "sessions") else ledger["body_construction"]
        )
        > 2
        or result["cost"]["unknown_cost_arms"]
        or result["cost"]["unknown_setup_calls"]
        or result["cost"]["unknown_body_construction_calls"]
    ):
        primary = primary or RuntimeError(
            "E313 scientific clocks/caps unavailable or exceeded"
        )
        result.update(
            run_valid=False,
            actual_startup_pass=False,
            short_gait_pass=False,
            next_separate_10s_eligible=False,
            validity="invalid_or_incomplete",
            first_error={"type": type(primary).__name__, "message": str(primary)},
        )
        errors.append({"stage": "cost", "message": str(primary)})
    finalization_began = time.perf_counter()
    try:
        deps.write(out / "advance-ledger.json", ledger)
        deps.write(out / "errors.json", {"errors": errors})
        manifest = runner.short.output_manifest(out, LIMITS["output_bytes"])
        manifest["schema"] = "flylab.m3.e313.output-manifest.v1"
        deps.write(out / "output-manifest.json", manifest)
        if (time.perf_counter() - began >= LIMITS["internal_wall_s"]
                or time.perf_counter() - finalization_began >= LIMITS["finalization_reserve_s"]):
            raise TimeoutError("E313 final publication deadline")
        result["publication_status"] = "complete"
        deps.write(out / "result.json", result)
        deps.write(out / "result-sha256.json", {"schema": "flylab.m3.e313.result-sha256.v1", "path": "result.json", "sha256": runner.protocol.sha256(out / "result.json")})
        if (time.perf_counter() - began >= LIMITS["internal_wall_s"]
                or time.perf_counter() - finalization_began >= LIMITS["finalization_reserve_s"]):
            raise TimeoutError("E313 late result publication deadline")
    except Exception as error:
        primary = primary or error
        result.update(
            run_valid=False,
            actual_startup_pass=False,
            short_gait_pass=False,
            next_separate_10s_eligible=False,
            validity="invalid_or_incomplete",
            first_error={"type": type(primary).__name__, "message": str(primary)},
        )
        for arm in result["arms"].values():
            arm["interpretation_validated"] = False
        result["publication_status"] = "invalid"
        errors.append({"stage": "publication", "type": type(error).__name__, "message": str(error)})
        def secondary(stage, failure):
            errors.append({"stage": stage, "type": type(failure).__name__, "message": str(failure)})
            primary.add_note("Secondary E313 " + stage + ": " + repr(failure))
        def clear_receipt(name):
            try:
                (out / name).unlink(missing_ok=True)
            except Exception as failure:
                secondary("invalidate " + name, failure)
        for name in ("output-manifest.json", "result-sha256.json"):
            clear_receipt(name)
        # Independent attempts make an earlier failed error write unable to
        # suppress the best available invalid result/status publication.
        for name, value in (("errors.json", {"errors": errors}), ("result.json", result),
                            ("publication-status.json", {"schema": "flylab.m3.e313.publication-status.v1", "status": "invalid",
                                "run_valid": False, "first_error": result["first_error"], "success_receipts_trusted": False})):
            try:
                deps.write(out / name, value)
            except Exception as failure:
                secondary("invalid publication " + name, failure)
                if name == "result.json":
                    clear_receipt(name)
        # Include secondary invalidation/write failures whenever still writable.
        try:
            deps.write(out / "errors.json", {"errors": errors})
        except Exception as failure:
            secondary("final errors publication", failure)
    return result, primary

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
                failures[0].add_note("Secondary E313 cleanup: " + repr(error))
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
