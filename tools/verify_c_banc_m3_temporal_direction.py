"""E330: one frozen temporal surrogate-VJP direction, never physical gait proof.

Import and the passive gate read only JSON/file hashes. Actual loading and CUDA
are reached only by an explicitly authorized root freeze/review/qualification.
The original CLI must be run once by the root's bounded 220s supervisor. All
older modules are reused read-only; their globals and output paths never change.
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

from tools import verify_c_banc_m3_full_horizon_direction as e323
from tools import verify_c_banc_m3_reset_window_direction as e324
from tools import verify_c_banc_m3_magnitude_direction as e325

ROOT, BASE = e323.ROOT, e323.BASE
N, GROUPS, STARTS = e324.N, e324.GROUPS, tuple(range(100, 300, 4))
OWNED = ("tools/verify_c_banc_m3_temporal_direction.py", "tests/test_c_banc_temporal_direction.py")
OUTPUT = BASE + "/e330-temporal-direction"
SCHEMA = "flylab.m3.e330.temporal-direction.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e330.temporal-direction.review.v1"
RESULT_SCHEMA = "flylab.m3.e330.temporal-direction.result.v1"
QUAL_SCHEMA = "flylab.m3.e330.generated-qualification.v1"
LIMITS = {"supervisor_wall_s": 220, "internal_wall_s": 200, "science_wall_s": 180,
          "finalization_reserve_s": 20, "host_bytes": 6 << 30, "additional_gpu_bytes": 8 << 30,
          "output_bytes": 2 << 30, "minimum_free_disk_bytes": 4 << 30,
          "forward_neural_ticks": 18000, "recompute_neural_ticks": 4000,
          "backward_block_ticks": 4000, "backward_blocks": 200,
          "decoder_replay_controls": 300, "forward_rollout_calls": 151,
          "backward_rollout_calls": 50, "original_processes": 1,
          "body_calls": 0, "optimizer_steps": 0, "retry_count": 0}
E329_SOURCES = ("tools/banc_m3_phase_free_temporal.py", "tools/analyze_c_banc_m3_temporal_objective.py",
                "tests/test_c_banc_phase_free_temporal.py", "tests/test_c_banc_temporal_objective.py")
PATHS = {**e325.PATHS,
         "e323_output": e323.OUTPUT,
         "e325_freeze": BASE + "/e325-magnitude-direction-freeze.json",
         "e325_review": BASE + "/e325-magnitude-direction-source-review.json",
         "e325_qualification": BASE + "/parent-e325-generated-qualification.json",
         "e325_result": e325.OUTPUT + "/result.json",
         "e325_audit": BASE + "/parent-e325-original-raw-audit.json",
         "e325_source_manifest": e325.OUTPUT + "/source-manifest.json",
         "e325_output_manifest": e325.OUTPUT + "/output-manifest.json",
         "e325_ledger": e325.OUTPUT + "/advance-ledger.json",
         "e325_output": e325.OUTPUT,
         "e329_qualification": BASE + "/parent-e329-qualification.json",
         "e329_review": BASE + "/parent-e329-final-source-review.json",
         "e329_cpu": BASE + "/parent-e329-second-cpu.json",
         "e329_cuda": BASE + "/parent-e329-second-cuda.json",
         "e329_contract": BASE + "/e329-temporal-saved-contract.json",
         "e329_result": BASE + "/e329-temporal-saved-characterization/result.json",
         "e329_audit": BASE + "/parent-e329-saved-raw-audit.json",
         "e329_design": "docs/BANC_M3_E329_TEMPORAL_LEARNING_DESIGN_20261001.md",
         "e330_design": "docs/BANC_M3_E330_TEMPORAL_DIRECTION_DESIGN_20261001.md",
         "e330_astra": BASE + "/parent-e330-astra-design-review.md",
         "e330_normalizer_astra": BASE + "/parent-e330-normalizer-design-addendum.md",
         "e330_qualification": BASE + "/parent-e330-generated-qualification.json"}
TERM_NAMES = ("training_loss", "successor_loss", "coordination_loss", "lr_coordination_loss")
METRIC_NAMES = (*TERM_NAMES, "shape_loss", "per_leg_successor_loss", "per_leg_shape_loss",
                "startup_q_rms", "startup_q_max", "startup_tendon_rms", "startup_tendon_max",
                "startup_pass")
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def bytes_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _closure(root):
    prior = read(e323.safe(PATHS["e325_freeze"], root))
    e329 = read(e323.safe(PATHS["e329_qualification"], root))
    qual = read(e323.safe(PATHS["e330_qualification"], root))
    sources = {**prior["source_sha256"], **e329["source_sha256"]}
    sources.update({name: e323.sha256(e323.safe(name, root)) for name in OWNED})
    roster = e323._file_roster(PATHS, root) | set(prior["input_sha256"])
    for proof in (e329, qual):
        roster.update(name for name in proof.get("evidence_sha256", {}) if name not in sources)
    contract = read(e323.safe(PATHS["e329_contract"], root))
    for descriptor in contract["inputs"].values():
        path = Path(descriptor["path"])
        if not path.is_absolute():
            path = e323.safe(PATHS["e329_contract"], root).parent / path
        roster.add(path.resolve().relative_to(root.resolve()).as_posix())
    return sources, roster


def make_freeze(*, root=ROOT):
    """Passive root helper; creates no original reservation or authorization."""
    sources, roster = _closure(root)
    saved = read(e323.safe(PATHS["e329_result"], root))
    return {"schema": SCHEMA, "paths": PATHS.copy(), "limits": LIMITS.copy(), "output": OUTPUT,
            "candidate_adopted": False, "original_execution_authorized": False,
            "original_supervisor_required": True, "source_sha256": sources,
            "input_sha256": {name: e323.sha256(e323.safe(name, root)) for name in sorted(roster)},
            "temporal_reference": {"curves": saved["reference_curves_metadata"],
                "normalization": saved["successor_normalization_metadata"],
                "normalization_f64": saved["successor_normalization_mse_rad"],
                "frequency_hz": 2.382760007990865, "control_dt_s": .005,
                "coordination_coefficient": 1, "unordered_pairs": 15,
                "previous_controls": [100, 299], "next_controls": [101, 300]},
            "gradient_semantics": "rounded native forward with real-equation surrogate VJP"}


def _check_map(expected, current, label):
    if not isinstance(expected, dict) or not expected or any(current.get(k) != v for k, v in expected.items()):
        raise ValueError("Incomplete or differing " + label)


def _approved(proof, schema, label):
    if (proof.get("schema") != schema or proof.get("approved") is not True
            or proof.get("blocking_defects") != []):
        raise ValueError("Approved " + label + " required")


def _prior_proof(stage, module, verdict, pins, sources, root):
    """Close prior freeze/review/producer/qualification/audit, without a payload."""
    alias = "e" + str(stage)
    audit_alias = "e324_audit_v2" if stage == 324 else alias + "_audit"
    frozen = read(e323.safe(PATHS[alias + "_freeze"], root))
    review = read(e323.safe(PATHS[alias + "_review"], root))
    source = read(e323.safe(PATHS[alias + "_source_manifest"], root))
    result = read(e323.safe(PATHS[alias + "_result"], root))
    audit = read(e323.safe(PATHS[audit_alias], root))
    if (frozen.get("schema") != module.SCHEMA
            or frozen.get("paths") != (e323.REQUIRED_PATHS if stage == 323 else module.PATHS)
            or frozen.get("limits") != module.LIMITS or frozen.get("output") != module.OUTPUT
            or frozen.get("candidate_adopted") is not False):
        raise ValueError("Exact prior freeze required: " + alias)
    _approved(review, module.REVIEW_SCHEMA, alias + " review")
    _check_map(frozen["input_sha256"], pins, alias + " inputs")
    _check_map(frozen["source_sha256"], sources, alias + " sources")
    if (review.get("freeze_sha256") != pins[PATHS[alias + "_freeze"]]
            or review.get("source_sha256") != frozen["source_sha256"]
            or source.get("schema") != f"flylab.m3.{alias}.source-manifest.v1"
            or source.get("freeze_sha256") != pins[PATHS[alias + "_freeze"]]
            or source.get("review_sha256") != pins[PATHS[alias + "_review"]]
            or source.get("source_sha256") != frozen["source_sha256"]
            or source.get("input_sha256") != frozen["input_sha256"]
            or result.get("schema") != module.RESULT_SCHEMA or result.get("run_valid") is not True
            or result.get("classification") != verdict
            or audit.get("schema") != f"flylab.m3.{alias}.root-original-raw-audit.{'v2' if stage == 324 else 'v1'}"
            or audit.get("pass") is not True
            or audit.get("diagnostic_run_valid") is not True or audit.get("verdict") != verdict
            or audit.get("result_sha256") != pins[PATHS[alias + "_result"]]
            or audit.get("output_manifest_sha256") != pins[PATHS[alias + "_output_manifest"]]
            or audit.get("advance_ledger_sha256") != pins[PATHS[alias + "_ledger"]]):
        raise ValueError("Prior root-audited producer linkage differs: " + alias)
    qual = read(e323.safe(PATHS[alias + "_qualification"], root))
    _approved(qual, ("flylab.m3.e323.generated-qualification-transfer.v1" if stage == 323
                     else f"flylab.m3.{alias}.generated-qualification.v1"), alias + " qualification")
    if qual.get("original_calls") != 0:
        raise ValueError("Prior qualification performed original work")
    for field in ("source_sha256", "evidence_sha256"):
        if qual.get(field):
            _check_map(qual[field], {**pins, **sources}, alias + " " + field)
    if stage == 323 and qual.get("current_source_sha256") != sources[module.OWNED[0]]:
        raise ValueError("E323 transferred qualification source differs")
    if stage == 324 and audit.get("audit_source_sha256") != pins[PATHS["e324_audit_v2_source"]]:
        raise ValueError("E324 root audit source differs")
    if stage in (324, 325) and (result.get("continuation_supported") is not False
                              or audit.get("continuation_supported") is not False):
        raise ValueError("Prior rejected continuation must remain rejected")
    manifest = read(e323.safe(PATHS[alias + "_output_manifest"], root))
    if manifest.get("schema") != f"flylab.m3.{alias}.output-manifest.v1":
        raise ValueError("Prior output manifest schema differs")
    files = {path.relative_to(e323.safe(module.OUTPUT, root)).as_posix()
             for path in e323.safe(module.OUTPUT, root).rglob("*") if path.is_file()
             and path.name not in ("result.json", "advance-ledger.json", "output-manifest.json")}
    if set(manifest.get("files", {})) != files:
        raise ValueError("Prior immutable producer output roster differs: " + alias)
    for relative, descriptor in manifest.get("files", {}).items():
        path = e323.safe(module.OUTPUT + "/" + relative, root)
        name = path.relative_to(root).as_posix()
        # E323 has a narrower declared manifest than later producers. Its
        # immutable entries still have to be pinned by their root closure.
        if descriptor != {"sha256": pins.get(name), "bytes": path.stat().st_size}:
            raise ValueError("Prior immutable producer file differs: " + name)


def _legacy_input_proof(pins, sources, root):
    """Keep E315/E318 and E317/E319 qualifications linked to the current bytes."""
    digests = {**pins, **sources}
    initial = read(e323.safe(PATHS["e315_freeze"], root))
    if initial.get("schema") != "flylab.m3.e315.native-adjoint-probe.freeze.v1":
        raise ValueError("Pinned E315 native input contract required")
    _check_map(initial["source_sha256"], sources, "E315 native sources")
    descriptor = initial["qualification"]
    if pins.get(descriptor["path"]) != descriptor["sha256"]:
        raise ValueError("E315 generated qualification pin differs")
    qualified = read(e323.safe(descriptor["path"], root))
    if (qualified.get("schema") != "flylab.m3.e314.native-adjoint-generated-qualification.v1"
            or qualified.get("approved") is not True):
        raise ValueError("Approved E314/E315 generated qualification required")
    _check_map(qualified["evidence_sha256"], digests, "E314/E315 generated evidence")
    for descriptor in initial["producer_reviews"]:
        if pins.get(descriptor["path"]) != descriptor["sha256"]:
            raise ValueError("E315 producer review pin differs")
    for descriptor in initial["inputs"].values():
        if "sha256" in descriptor:
            if pins.get(descriptor["path"]) != descriptor["sha256"]:
                raise ValueError("E315 original input pin differs")
        else:
            for name in ("manifest", "state"):
                path = descriptor["path"] + "/" + name + ".json"
                if pins.get(path) != descriptor[name + "_sha256"]:
                    raise ValueError("E315 original state metadata pin differs")
    for stage in (317, 319):
        alias = "e" + str(stage)
        review = read(e323.safe(PATHS[alias + "_review"], root))
        _approved(review, f"flylab.m3.{alias}.root-second-review.v1", alias + " source review")
        _check_map(review["source_sha256"], sources, alias + " reviewed source")
        qual = read(e323.safe(PATHS[alias + "_qualification"], root))
        _approved(qual, f"flylab.m3.{alias}.generated-qualification.v1", alias + " generated qualification")
        if qual.get("original_calls") != 0 or read(e323.safe(PATHS[alias + "_cuda"], root)).get("passed") is not True:
            raise ValueError("Original-zero prior generated CUDA qualification required")
        _check_map(qual["source_sha256"], digests, alias + " qualification source")
        _check_map(qual["evidence_sha256"], digests, alias + " generated evidence")
    original = read(e323.safe(PATHS["e318_freeze"], root))
    review = read(e323.safe(PATHS["e318_review"], root))
    producer = read(e323.safe(PATHS["e318_source_manifest"], root))
    _approved(review, "flylab.m3.e318.full-drive-capture.review.v1", "E318 source review")
    if (original.get("schema") != "flylab.m3.e318.full-drive-capture.freeze.v1"
            or producer.get("schema") != "flylab.m3.e318.source-manifest.v1"
            or review.get("freeze_sha256") != pins[PATHS["e318_freeze"]]
            or review.get("source_sha256") != original["source_sha256"]
            or producer.get("freeze_sha256") != pins[PATHS["e318_freeze"]]
            or producer.get("review_sha256") != pins[PATHS["e318_review"]]
            or producer.get("input_sha256") != original["input_sha256"]):
        raise ValueError("E318 complete approved producer input/source linkage differs")


def _e329_proof(freeze, pins, sources, root):
    qual = read(e323.safe(PATHS["e329_qualification"], root))
    review = read(e323.safe(PATHS["e329_review"], root))
    saved = read(e323.safe(PATHS["e329_result"], root))
    audit = read(e323.safe(PATHS["e329_audit"], root))
    contract = read(e323.safe(PATHS["e329_contract"], root))
    _approved(qual, "flylab.m3.e329.generated-and-saved-qualification.v1", "E329 qualification")
    _approved(review, "flylab.m3.e329.root-final-source-review.v1", "E329 review")
    if set(qual.get("source_sha256", {})) != set(E329_SOURCES) or review.get("source_sha256") != qual["source_sha256"]:
        raise ValueError("Complete E329 approved four-source roster required")
    _check_map(qual["source_sha256"], sources, "E329 approved source bytes")
    _check_map(qual["evidence_sha256"], {**pins, **sources}, "E329 evidence bytes")
    if (review.get("root_cpu_qualification_sha256") != pins[PATHS["e329_cpu"]]
            or review.get("root_cuda_qualification_sha256") != pins[PATHS["e329_cuda"]]
            or review.get("design_sha256") != pins[PATHS["e329_design"]]):
        raise ValueError("E329 root review qualification linkage differs")
    for alias in ("e329_cpu", "e329_cuda"):
        proof = read(e323.safe(PATHS[alias], root))
        if (proof.get("passed") is not True or not proof.get("checks")
                or not all(value is True for value in proof["checks"].values())
                or proof.get("objective_source_sha256") != sources[E329_SOURCES[0]]
                or proof.get("audit_source_sha256") != pins.get(BASE + "/parent-e329-independent-qualification.py")):
            raise ValueError("E329 independent CPU/CUDA linkage differs")
    if (saved.get("schema") != "flylab.m3.e329.temporal.saved-characterization.v1"
            or audit.get("schema") != "flylab.m3.e329.root-saved-raw-audit.v1"
            or audit.get("passed") is not True or not audit.get("checks")
            or not all(value is True for value in audit["checks"].values())
            or audit.get("result_sha256") != pins[PATHS["e329_result"]]
            or audit.get("contract_sha256") != pins[PATHS["e329_contract"]]
            or audit.get("audit_source_sha256") != pins.get(BASE + "/parent-e329-saved-raw-audit.py")
            or saved.get("pins", {}).get("contract", {}).get("sha256") != pins[PATHS["e329_contract"]]
            or contract.get("schema") != "flylab.m3.e329.temporal.saved-contract.v1"
            or contract.get("source_sha256") != qual["source_sha256"]):
        raise ValueError("E329 saved result/contract/root audit closure differs")
    for role, alias in (("reference", "e85_reference"), ("startup", "e103_startup"),
                        ("neutral", "e318_raw"), ("rejected", None)):
        descriptor = contract["inputs"][role]
        path = Path(descriptor["path"])
        if not path.is_absolute():
            path = e323.safe(PATHS["e329_contract"], root).parent / path
        name = path.resolve().relative_to(root.resolve()).as_posix()
        if (pins.get(name) != descriptor["sha256"] or saved["pins"]["inputs"][role]["sha256"] != descriptor["sha256"]
                or alias is not None and name != PATHS[alias]):
            raise ValueError("E329 input linkage differs: " + role)
    for name, descriptor in saved["pins"]["sources"].items():
        if sources.get(name) != descriptor["sha256"]:
            raise ValueError("E329 saved approved source differs")
    if (qual.get("e330_input_qualification_pass") is not True
            or saved.get("e330_input_qualification_pass") is not True
            or saved.get("valid_input_nonlocal_identifiability_pass") is not True
            or saved.get("reference_qualification", {}).get("pass") is not True
            or saved["arms"]["rejected"].get("classification") != "rejected"
            or saved.get("candidate_adopted") is not False):
        raise ValueError("E329 actual reference/input eligibility required; E327 stays rejected")
    expected = make_reference_contract(saved)
    if freeze.get("temporal_reference") != expected:
        raise ValueError("Exact E329 temporal reference constants and byte metadata required")
    for metadata, shape in ((expected["curves"], [6, 128, 7]), (expected["normalization"], [6])):
        if (metadata.get("shape") != shape or metadata.get("dtype") != "float64"
                or metadata.get("numpy_dtype") != "<f8" or metadata.get("layout") != "C"
                or metadata.get("leg_order") != list(LEGS)):
            raise ValueError("Fixed E329 reference shape/dtype/leg/C-byte order required")
    if expected["curves"].get("joint_index_order") != list(range(7)):
        raise ValueError("Fixed E329 joint index order required")
    vector = np.asarray(expected["normalization_f64"], dtype=np.float64)
    if (vector.shape != (6,) or not np.isfinite(vector).all() or np.any(vector <= 0)
            or bytes_sha(vector) != expected["normalization"]["c_order_sha256"]
            or audit.get("reference_curve_sha256") != expected["curves"]["c_order_sha256"]):
        raise ValueError("E329 JSON normalizer or curve approved byte hash differs")


def make_reference_contract(saved):
    return {"curves": saved["reference_curves_metadata"], "normalization": saved["successor_normalization_metadata"],
            "normalization_f64": saved["successor_normalization_mse_rad"],
            "frequency_hz": 2.382760007990865, "control_dt_s": .005,
            "coordination_coefficient": 1, "unordered_pairs": 15,
            "previous_controls": [100, 299], "next_controls": [101, 300]}


def gate(freeze_path, freeze_sha256, review_path, review_sha256, output, *, root=ROOT):
    if output.exists() or output.resolve() != e323.safe(OUTPUT, root):
        raise ValueError("Fresh literal E330 output required")
    if e323.sha256(freeze_path) != freeze_sha256 or e323.sha256(review_path) != review_sha256:
        raise ValueError("E330 freeze/review SHA differs")
    freeze, review = read(freeze_path), read(review_path)
    _approved(review, REVIEW_SCHEMA, "E330 source review")
    if (freeze.get("schema") != SCHEMA or freeze.get("paths") != PATHS or freeze.get("limits") != LIMITS
            or freeze.get("output") != OUTPUT or freeze.get("candidate_adopted") is not False
            or freeze.get("original_execution_authorized") is not True
            or freeze.get("original_supervisor_required") is not True
            or review.get("freeze_sha256") != freeze_sha256
            or review.get("source_sha256") != freeze.get("source_sha256")):
        raise ValueError("Explicit root-authorized exact E330 freeze required")
    sources, roster = _closure(root)
    if freeze.get("source_sha256") != sources or set(freeze.get("input_sha256", {})) != roster:
        raise ValueError("Exact complete E330 source/input closure required")
    pins = freeze["input_sha256"]
    if any(name in pins and pins[name] != digest for name, digest in sources.items()):
        raise ValueError("Contradictory E330 source/input pin")
    for name, digest in {**pins, **sources}.items():
        if (not isinstance(digest, str) or len(digest) != 64
                or e323.sha256(e323.safe(name, root)) != digest):
            raise ValueError("E330 pinned source/input differs: " + name)
    for stage, module, verdict in ((323, e323, "refuted"), (324, e324, "supported"), (325, e325, "supported")):
        _prior_proof(stage, module, verdict, pins, sources, root)
    _legacy_input_proof(pins, sources, root)
    # Reuse the exact original E318 producer/root-audit link carried by E323.
    old = read(e323.safe(PATHS["e318_freeze"], root))
    producer = read(e323.safe(PATHS["e318_source_manifest"], root))
    audit = read(e323.safe(PATHS["e318_audit"], root))
    result = read(e323.safe(PATHS["e318_result"], root))
    _check_map(old["input_sha256"], pins, "E318 original producer inputs")
    _check_map(old["source_sha256"], sources, "E318 original producer sources")
    if (producer.get("source_sha256") != old["source_sha256"] or result.get("run_valid") is not True
            or result.get("learning_input_approved") is not True or audit.get("pass") is not True
            or audit.get("learning_input_approved") is not True
            or audit.get("result_sha256") != pins[PATHS["e318_result"]]
            or audit.get("output_manifest_sha256") != pins[PATHS["e318_output_manifest"]]):
        raise ValueError("E318 approved original replay proof differs")
    manifest = read(e323.safe(PATHS["e318_output_manifest"], root))
    for name in pins:
        if name.startswith(e323.E318 + "/") and name not in (PATHS["e318_result"], PATHS["e318_output_manifest"]):
            if manifest["files"].get(name.removeprefix(e323.E318 + "/")) != {
                    "sha256": pins[name], "bytes": e323.safe(name, root).stat().st_size}:
                raise ValueError("E318 immutable producer bytes differ")
    _e329_proof(freeze, pins, sources, root)
    qual = read(e323.safe(PATHS["e330_qualification"], root))
    _approved(qual, QUAL_SCHEMA, "E330 generated qualification")
    if (qual.get("original_calls") != 0 or qual.get("cpu_pass") is not True or qual.get("cuda_pass") is not True
            or qual.get("source_sha256") != {name: sources[name] for name in OWNED}):
        raise ValueError("Current-source CPU/CUDA E330 qualification required")
    _check_map(qual.get("evidence_sha256"), {**pins, **sources}, "E330 generated evidence")
    if shutil.disk_usage(output.parent).free < LIMITS["minimum_free_disk_bytes"]:
        raise OSError("E330 disk reserve unavailable")
    return freeze


class Dependencies(e324.Dependencies):
    def load(self, freeze, root=ROOT):
        data = super().load(freeze, root=root)
        from tools.analyze_c_banc_m3_temporal_objective import validate_saved
        validate_saved(data["raw"])
        saved = read(e323.safe(PATHS["e329_result"], root))
        original_csr_path = read(e323.safe(PATHS["e315_freeze"], root))["inputs"]["actual_csr"]["path"]
        data.update(e329_neutral_terms={name: saved["arms"]["neutral"]["scores"][name] for name in TERM_NAMES},
                    temporal_reference=freeze["temporal_reference"],
                    handoff_source_pins={key: freeze["input_sha256"][PATHS[key]] for key in
                        ("e318_raw", "e150_calibration", "e150_parameters", "e329_result")},
                    original_control0_path=PATHS["e318_start"],
                    original_csr_file={"path": original_csr_path, "sha256": freeze["input_sha256"][original_csr_path]},
                    original_control0_pins={name: digest for name, digest in freeze["input_sha256"].items()
                        if name.startswith(PATHS["e318_start"] + "/")})
        return data

    def build(self, data, device):
        from tools.banc_m3_phase_free_temporal import PhaseFreeTemporalObjective
        rollout, _old_shape, initial = super().build(data, device)
        constants = data["temporal_reference"]
        if bytes_sha(data["curve"]) != constants["curves"]["c_order_sha256"]:
            raise ValueError("Rebuilt E329 curve bytes differ")
        objective = PhaseFreeTemporalObjective(data["curve"], data["startup"]["target_command"],
                                               data["startup"]["target_tendon"], device=device)
        data["normalizer_transfer"] = install_normalizer(objective, constants)
        return rollout, objective, initial


def install_normalizer(objective, constants):
    """Compare constructor math first, then verify the actual loss-read tensor bytes."""
    import torch
    frozen = np.asarray(constants["normalization_f64"], dtype=np.float64)
    if (frozen.shape != (6,) or not np.isfinite(frozen).all() or np.any(frozen <= 0)
            or bytes_sha(frozen) != constants["normalization"]["c_order_sha256"]):
        raise ValueError("Finite positive fixed E329 JSON f64 normalization bytes required")
    computed = e323._host(objective.successor_normalization)
    if (computed.shape != (6,) or computed.dtype != np.float64 or not np.isfinite(computed).all()
            or not np.allclose(computed, frozen, rtol=1e-12, atol=1e-15)):
        raise ValueError("Device-computed normalizer differs from frozen E329 constants")
    objective.successor_normalization = torch.tensor(frozen.copy(), dtype=torch.float64, device=objective.device)
    observed_fixed = e323._host(objective.successor_normalization)
    if bytes_sha(observed_fixed) != constants["normalization"]["c_order_sha256"]:
        raise ValueError("Actual loss denominator device bytes differ from frozen E329 vector")
    return {"device_computed_sha256": bytes_sha(computed), "frozen_source_sha256": bytes_sha(frozen),
            "device_value_comparison_pass": True, "rtol": 1e-12, "atol": 1e-15,
            "used_frozen_source_f64_vector": True, "actual_loss_denominator_sha256": bytes_sha(observed_fixed),
            "device_reduction_bytes_claimed_equal": bytes_sha(computed) == bytes_sha(frozen)}


def window_loss(objective, fixed_boundary, live_q, *, window_count=50):
    import torch
    if (live_q.shape != (4, 42) or fixed_boundary.shape != (42,)
            or type(window_count) is not int or window_count < 1):
        raise ValueError("Four live commands, one fixed original boundary, and positive window count required")
    previous = torch.cat((fixed_boundary.detach()[None], live_q[:3]), dim=0)
    terms = objective.evaluate_pairs(previous, live_q)
    return terms, terms["training_loss"] / window_count


def make_direction(gradient):
    if (not isinstance(gradient, np.ndarray) or gradient.dtype != np.float64
            or gradient.ndim != 1 or not len(gradient) or not np.isfinite(gradient).all()):
        raise ValueError("Finite f64 gradient required")
    maximum = float(np.max(np.abs(gradient)))
    if maximum == 0:
        return None
    direction = gradient / maximum
    plus = np.asarray(1 + .001 * direction, np.float32).astype(np.float64)
    minus = np.asarray(1 - .001 * direction, np.float32).astype(np.float64)
    if not np.isfinite(plus).all() or not np.isfinite(minus).all() or np.any(plus <= 0) or np.any(minus <= 0):
        raise ValueError("Positive exact f32-promoted efficacies required")
    return {"gradient": gradient.copy(), "direction": direction, "theta_plus": plus, "theta_minus": minus}


def classify_direction(neutral, plus, minus, gradient, theta_plus, theta_minus):
    if not all(np.isfinite(value) for value in (neutral, plus, minus)):
        raise ValueError("Finite neutral/plus/minus losses required")
    if (not isinstance(gradient, np.ndarray) or gradient.dtype != np.float64 or gradient.ndim != 1
            or not len(gradient) or not np.isfinite(gradient).all()
            or any(not isinstance(gain, np.ndarray) or gain.dtype != np.float64
                   or gain.shape != gradient.shape or not np.isfinite(gain).all() or np.any(gain <= 0)
                   or not np.array_equal(gain, gain.astype(np.float32).astype(np.float64))
                   for gain in (theta_plus, theta_minus))):
        raise ValueError("Finite f64 gradient and exact positive f32-promoted direction required")
    observed = float(plus - minus)
    with np.errstate(over="ignore", invalid="ignore"):
        predicted = float(np.sum(gradient * (theta_plus - theta_minus), dtype=np.float64))
    improvement = float(neutral - minus)
    if not np.isfinite(observed) or not np.isfinite(improvement):
        raise ValueError("Finite loss differences required")
    if not np.isfinite(predicted) or predicted < 0:
        raise ValueError("Finite nonnegative rounded-direction prediction required")
    meaningful = abs(observed) > 1e-9 and predicted > 1e-9
    relative = abs(observed - predicted) / predicted if meaningful else None
    classification = ("supported" if meaningful and observed > 0 and relative <= .05
                      else "refuted" if meaningful else "inconclusive")
    return {"local_direction": classification, "observed": observed, "predicted": predicted,
            "relative_error": relative, "local_descent": bool(improvement >= 1e-9),
            "neutral_minus_improvement": improvement}


def continuation_reasons(local_direction, local_descent, native_eligible):
    return [reason for condition, reason in ((local_direction != "supported", "local_direction_not_supported"),
        (local_descent is not True, "minus_has_no_meaningful_local_descent"),
        (native_eligible is not True, "minus_native_csr_ineligible")) if condition]


def patch_weights(native, theta):
    """Once-rounded selected weights; count full CSR uint32 palette, including +/-0."""
    csr, mask = native["csr"], native["mask"]
    values, positions, groups = csr["data"], mask["positions"], mask["edge_parameters"]
    if (values.dtype != np.float32 or values.ndim != 1 or not np.isfinite(values).all()
            or positions.dtype != np.int64 or groups.dtype != np.int64 or positions.shape != groups.shape
            or positions.ndim != 1 or not len(positions) or np.any(positions < 0) or np.any(positions >= len(values))
            or len(np.unique(positions)) != len(positions) or theta.dtype != np.float64 or theta.ndim != 1
            or not np.isfinite(theta).all() or np.any(theta <= 0) or np.any(groups < 0)
            or np.any(groups >= len(theta)) or not np.array_equal(np.unique(groups), np.arange(len(theta)))
            or not np.array_equal(theta, theta.astype(np.float32).astype(np.float64))):
        raise ValueError("Exact f32 source, positions/groups, and positive f32-promoted theta required")
    original = values[positions].copy()
    if np.any(original == 0):
        raise ValueError("Selected original edges must be nonzero")
    with np.errstate(over="raise", invalid="raise"):
        actual = np.asarray(original.astype(np.float64) * theta[groups], np.float32)
    actual[theta[groups] == 1] = original[theta[groups] == 1]
    if not np.isfinite(actual).all() or np.any(actual == 0) or np.any(np.signbit(actual) != np.signbit(original)):
        raise ValueError("Selected actual32 must preserve finite nonzero signs")
    patched = {name: csr[name].copy() for name in ("indptr", "indices", "data")}
    patched["data"][positions] = actual
    nodes, edges = len(patched["indptr"]) - 1, len(values)
    if (patched["indptr"].dtype != np.int32 or patched["indices"].dtype != np.int32
            or patched["indices"].shape != values.shape or patched["indptr"][0] != 0
            or patched["indptr"][-1] != edges or np.any(np.diff(patched["indptr"]) < 0)
            or np.any(patched["indices"] < 0) or np.any(patched["indices"] >= nodes)):
        raise ValueError("Original exact native CSR topology required")
    palette = int(len(np.unique(patched["data"].view(np.uint32))))
    changed = int(np.count_nonzero(actual.view(np.uint32) != original.view(np.uint32)))
    eligible = palette <= 16384 and nodes <= 2 ** 18 and edges < 2 ** 31
    metadata = {"eligible": bool(eligible), "full_csr_bit_palette_count": palette, "palette_limit": 16384,
                "node_count": nodes, "node_limit_inclusive": 2 ** 18,
                "edge_count": edges, "edge_limit_exclusive": 2 ** 31,
                "selected_actual32_sha256": bytes_sha(actual), "source_selected32_sha256": bytes_sha(original),
                "source_csr_data32_sha256": bytes_sha(values), "patched_csr_data32_sha256": bytes_sha(patched["data"]),
                "source_csr_indptr_sha256": bytes_sha(csr["indptr"]), "source_csr_indices_sha256": bytes_sha(csr["indices"]),
                "selected_positions_sha256": bytes_sha(positions), "selected_groups_sha256": bytes_sha(groups),
                "theta64_sha256": bytes_sha(theta), "theta32_sha256": bytes_sha(theta.astype(np.float32)),
                "selected_changed_bit_count": changed, "unchanged_theta_groups": int(np.count_nonzero(theta == 1)),
                "unselected_bytes_preserved": True, "neutral_selected_bytes_preserved": True,
                "signs_preserved": True, "body_constructed": False}
    return patched, actual, metadata


def metrics(terms):
    result = {}
    for name in METRIC_NAMES:
        if name not in terms:
            continue
        value = e323._host(terms[name])
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite temporal metric: " + name)
        result[name] = value.tolist()
    return result


def _counters(receipts, requested, replay_controls):
    if set(receipts) != set(requested) or len(requested) != len(set(requested)) or replay_controls != 300:
        raise ValueError("Exact conditional requested/confirmed receipt roster required")
    totals = dict.fromkeys(("forward_controls", "forward_neural_ticks", "recompute_controls",
                           "recompute_neural_ticks", "backward_blocks", "backward_block_ticks"), 0)
    backward_calls = 0
    for name in requested:
        count = 300 if name == "continuation-minus" else 4
        back = name.endswith("-neutral")
        expected = {"forward_controls": count, "forward_neural_ticks": 20 * count,
                    "recompute_controls": count if back else 0,
                    "recompute_neural_ticks": 20 * count if back else 0,
                    "backward_blocks": count if back else 0,
                    "backward_block_ticks": 20 * count if back else 0}
        if receipts[name].get("confirmed") != expected:
            raise ValueError("Incomplete or unexpected neural receipt: " + name)
        for key in totals:
            totals[key] += expected[key]
        backward_calls += int(back)
    totals.update(decoder_replay_controls=replay_controls, forward_rollout_calls=len(requested),
                  backward_rollout_calls=backward_calls, body_calls=0, optimizer_steps=0, retry_count=0)
    for key, value in totals.items():
        if key in LIMITS and value > LIMITS[key]:
            raise ValueError("E330 work limit exceeded: " + key)
    return totals


def compute(data, *, dependencies=None, device="cuda", checkpoint=None, deadline=None,
            on_receipt=None, on_progress=None, on_boundary=None, starts=STARTS):
    """Fixed original path; explicit injected dependencies permit generated fixtures."""
    import torch
    if (not starts or any(type(c) is not int or c not in STARTS for c in starts)
            or tuple(sorted(set(starts))) != starts or dependencies is None and starts != STARTS):
        raise ValueError("Ordered fixed E330 starts required")
    for name, shape, dtype in (("drives", (300, N), np.float32), ("rates", (300, N), np.float32),
                              ("adaptations", (300, N), np.float32), ("motor_rates", (300, 391), np.float32)):
        e324._valid_array(data[name], shape, dtype, name)
    deps = Dependencies() if dependencies is None else dependencies
    save = checkpoint or (lambda _name, _arrays: None)
    timely = deadline or (lambda: None)
    notify = on_progress or (lambda _value: None)
    boundary = on_boundary or (lambda _name, _event: None)
    report_receipt = on_receipt or (lambda _name, _value: None)
    receipts, requested = {}, []
    result = {"conditional_fixed_neural_input": True, "window_starts": list(starts),
              "gradient_semantics": "rounded native forward with real-equation surrogate VJP",
              "quantized_program_derivative_claimed": False, "full_horizon_gradient_claimed": False,
              "window_controls": 4, "steps_per_control": 20, "direction_count": 1,
              "decoder_replay_controls": 0, "decoder_replay_work": {
                  "requested_controls": 0, "host_returned_controls": 0, "device_confirmed_controls": None},
              "runs": {}, "windows": [], "requested_rollouts": requested,
              "local_direction": "not_tested", "local_descent": None,
              "native_eligibility": {"status": "not_tested"},
              "continuation": {"classification": "not_tested", "requested": False, "completed": False},
              "body_not_executed": True, "body_calls": 0, "F1_F8_complete": False,
              "candidate_adopted": False}

    def marked(name, action):
        timely()
        boundary(name, "requested")
        try:
            value = action()
        except BaseException as error:
            try:
                boundary(name, "failed")
            except Exception as secondary:  # noqa: BLE001
                error.add_note("E330 journal failure: " + repr(secondary))
            raise
        boundary(name, "host_returned")
        return value

    def retain_receipt(name, receipt, error):
        if receipt is None:
            return
        try:
            receipt.confirm()
        except Exception as secondary:  # noqa: BLE001
            error.add_note("E330 partial confirmation failure: " + repr(secondary))
        try:
            receipts[name] = e323._receipt(receipt)
            report_receipt(name, receipts[name])
        except Exception as secondary:  # noqa: BLE001
            error.add_note("E330 partial receipt journal failure: " + repr(secondary))

    rollout, objective, initial = marked("construct", lambda: deps.build(data, device))
    r0, z0, bank0, original0, drives, mask = initial
    result["normalizer_transfer"] = data.get("normalizer_transfer", {"generated_injected_objective": True})
    result["temporal_reference"] = data.get("temporal_reference", {"generated_injected_objective": True})
    result["loss_term_order"] = list(TERM_NAMES)
    expected_device = torch.device(device)
    if expected_device.type == "cuda" and expected_device.index is None:
        expected_device = torch.device("cuda", torch.cuda.current_device())

    def tensor(value):
        return torch.as_tensor(np.ascontiguousarray(value), device=rollout.device).clone()

    def device_tensor(value, shape, dtype, name):
        if (not isinstance(value, torch.Tensor) or value.device != expected_device
                or value.shape != shape or value.dtype != dtype):
            raise ValueError("Exact rollout device/shape/dtype required: " + name)

    rollout_device = torch.device(rollout.device)
    if rollout_device.type == "cuda" and rollout_device.index is None:
        rollout_device = torch.device("cuda", torch.cuda.current_device())
    if rollout_device != expected_device:
        raise ValueError("Requested rollout device differs")
    for value, shape, dtype, name in ((r0, (N,), torch.float32, "initial rate"),
        (z0, (N,), torch.float32, "initial adaptation"), (bank0, (391, 3), torch.float64, "initial bank"),
        (original0, (96,), torch.float64, "initial original"),
        (drives, (300, N), torch.float32, "held drives"), (mask, (N,), torch.float32, "output mask")):
        device_tensor(value, shape, dtype, name)

    def replay():
        banks, originals, q_rows, t_rows = [bank0.clone()], [original0.clone()], [], []
        with torch.no_grad():
            for control, motor in enumerate(data["motor_rates"], start=1):
                timely()
                result["decoder_replay_work"]["requested_controls"] = control
                decoded = rollout.decoder.step(tensor(motor), banks[-1].clone(), originals[-1].clone(), dt=.005)
                for key, shape in (("q", (42,)), ("tendon", (6,)), ("bank", (391, 3)), ("original_activation", (96,))):
                    device_tensor(decoded[key], shape, torch.float64, "decoder replay " + key)
                banks.append(decoded["bank"])
                originals.append(decoded["original_activation"])
                q_rows.append(decoded["q"])
                t_rows.append(decoded["tendon"])
                result["decoder_replay_controls"] = control
                result["decoder_replay_work"]["host_returned_controls"] = control
        return banks, originals, torch.stack(q_rows), torch.stack(t_rows)

    try:
        banks, originals, replay_q, replay_t = marked("decoder_replay", replay)
        notify(result.copy())
        replay_arrays = {"q": e323._host(replay_q), "tendon": e323._host(replay_t),
                         **{f"bank{c}": e323._host(banks[c]) for c in (100, 300)},
                         **{f"original{c}": e323._host(originals[c]) for c in (100, 300)}}
        save("decoder-replay", replay_arrays)
    except BaseException as error:
        try:
            notify(result.copy())
        except Exception as secondary:  # noqa: BLE001
            error.add_note("E330 replay journal failure: " + repr(secondary))
        raise
    result["decoder_replay_work"]["device_confirmed_controls"] = 300
    boundary("decoder_replay", "device_confirmed")
    reproduction = {}
    for key, width in (("q", 42), ("tendon", 6)):
        target = data["raw"]["command" if key == "q" else "tendon"]
        e324._valid_array(target, (301, width), np.float64, "saved " + key)
        e324._valid_array(replay_arrays[key], (300, width), np.float64, "decoder " + key)
        reproduction[key + "_max_abs"] = float(np.max(np.abs(replay_arrays[key] - target[1:])))
    for c, whole in ((100, data["whole100"]), (300, data["final"])):
        for key, dimensions in (("bank", (391, 3)), ("original", (96,))):
            target = whole["adapter"]["bank"] if key == "bank" else whole["adapter"]["original"]["activation"]
            e324._valid_array(target, dimensions, np.float64, "saved " + key)
            reproduction[key + str(c) + "_max_abs"] = float(np.max(np.abs(replay_arrays[key + str(c)] - target)))
    result["decoder_reproduction"] = reproduction
    notify(result.copy())
    if any(error > (1e-11 if key.startswith(("q_", "tendon_")) else 1e-10)
           for key, error in reproduction.items()):
        raise ValueError("E330 original decoder replay reproduction failed")

    def start_state(c):
        if c == 0:
            return r0.clone(), z0.clone(), bank0.clone(), original0.clone()
        return tensor(data["rates"][c - 1]), tensor(data["adaptations"][c - 1]), banks[c].clone(), originals[c].clone()

    def perform(c, label, theta, *, backward=False, continuation=False):
        name = "continuation-minus" if continuation else f"window-{c:03d}-{label}"
        requested.append(name)
        notify(result.copy())
        receipt, active = None, name + "_forward"
        try:
            r, z, bank, original = start_state(c)
            timely()
            boundary(active, "requested")
            with torch.set_grad_enabled(backward):
                output = rollout.run(r, z, bank, original, drives if continuation else drives[c:c + 4], theta, mask)
            receipt = output["receipt"]
            boundary(active, "host_returned")
            receipts[name] = e323._receipt(receipt)
            report_receipt(name, receipts[name])
            count = 300 if continuation else 4
            for key, dimensions, dtype in (("q", (count, 42), torch.float64), ("tendon", (count, 6), torch.float64),
                ("rate", (N,), torch.float32), ("adaptation", (N,), torch.float32),
                ("bank", (391, 3), torch.float64), ("original_activation", (96,), torch.float64)):
                device_tensor(output[key], dimensions, dtype, name + " " + key)
            arrays = {key: e323._host(output[key]) for key in
                      ("q", "tendon", "rate", "adaptation", "bank", "original_activation")}
            for key, dimensions, dtype in (("q", (count, 42), np.float64), ("tendon", (count, 6), np.float64),
                ("rate", (N,), np.float32), ("adaptation", (N,), np.float32),
                ("bank", (391, 3), np.float64), ("original_activation", (96,), np.float64)):
                e324._valid_array(arrays[key], dimensions, dtype, name + " " + key)
            neutral_reproduction = None
            if backward:
                qerr = float(np.max(np.abs(arrays["q"] - data["raw"]["command"][c + 1:c + 5])))
                terr = float(np.max(np.abs(arrays["tendon"] - data["raw"]["tendon"][c + 1:c + 5])))
                exact = e323.exact(arrays["rate"], data["rates"][c + 3]) and e323.exact(
                    arrays["adaptation"], data["adaptations"][c + 3])
                berr = float(np.max(np.abs(arrays["bank"] - e323._host(banks[c + 4]))))
                oerr = float(np.max(np.abs(arrays["original_activation"] - e323._host(originals[c + 4]))))
                neutral_reproduction = {"q_max_abs_rad": qerr, "tendon_max_abs_command": terr,
                    "end_rate_adaptation_dtype_bytes_exact": exact, "bank_max_abs": berr, "original_max_abs": oerr}
                if qerr > 1e-11 or terr > 1e-11 or not exact or berr > 1e-10 or oerr > 1e-10:
                    raise ValueError("E330 neutral original window reproduction failed")
            terms, scaled = ((objective.evaluate(output["q"], output["tendon"]), None) if continuation
                             else window_loss(objective, tensor(data["raw"]["command"][c]), output["q"], window_count=len(starts)))
            values = metrics(terms)
            arrays["loss_terms"] = np.asarray([values[key] for key in TERM_NAMES], np.float64)
            if not continuation:
                arrays["fixed_original_q_boundary"] = data["raw"]["command"][c].copy()
            save(name, arrays)
            receipt.confirm()
            receipts[name] = e323._receipt(receipt)
            report_receipt(name, receipts[name])
            boundary(active, "device_confirmed")
            active = None
            gradient = None
            if backward:
                theta.grad = None
                active = name + "_backward"
                timely()
                boundary(active, "requested")
                scaled.backward()
                boundary(active, "host_returned")
                receipt.confirm()
                receipts[name] = e323._receipt(receipt)
                report_receipt(name, receipts[name])
                boundary(active, "device_confirmed")
                active = None
                gradient = e323._host(theta.grad)
                e324._valid_array(gradient, (GROUPS,), np.float64, name + " gradient")
                save(name + "-gradient", {"gradient": gradient})
            row = {"start_control": c, "label": label, "metrics": values,
                   "raw_file": name + ".npz", "receipt": receipts[name]}
            if neutral_reproduction is not None:
                row.update(neutral_reproduction=neutral_reproduction, gradient_file=name + "-gradient.npz")
            result["runs"][name] = row
            notify(result.copy())
            timely()
            return values, gradient
        except BaseException as error:
            if active is not None:
                try:
                    boundary(active, "failed")
                except Exception as secondary:  # noqa: BLE001
                    error.add_note("E330 failure phase journal failure: " + repr(secondary))
            retain_receipt(name, receipt if receipt is not None else getattr(error, "receipt", None), error)
            raise

    theta = torch.ones(GROUPS, dtype=torch.float64, device=rollout.device, requires_grad=True)
    neutral, gradients = [], []
    for c in starts:
        value, gradient = perform(c, "neutral", theta, backward=True)
        neutral.append(value)
        gradients.append(gradient)
    neutral_mean = {key: float(np.mean([row[key] for row in neutral], dtype=np.float64)) for key in TERM_NAMES}
    if any(not np.isfinite(data["e329_neutral_terms"][key])
           or abs(neutral_mean[key] - data["e329_neutral_terms"][key]) > 1e-10 for key in TERM_NAMES):
        raise ValueError("E330 neutral 50-window terms differ from E329 full200-pair result")
    gradient = np.sum(np.stack(gradients), axis=0, dtype=np.float64)
    e324._valid_array(gradient, (GROUPS,), np.float64, "aggregate gradient")
    result["neutral_mean_terms"] = neutral_mean
    direction = make_direction(gradient)
    if direction is None:
        save("zero-gradient", {"gradient": gradient, "window_gradients": np.stack(gradients)})
        result.update(local_direction="inconclusive", local_descent=None, zero_gradient=True)
        result["continuation"]["reasons"] = ["finite_exact_zero_gradient"]
        result["native_eligibility"]["reason"] = "no_direction_created"
        result["actual_work"] = _counters(receipts, requested, result["decoder_replay_controls"])
        notify(result.copy())
        return result, receipts
    direction["window_gradients"] = np.stack(gradients)
    plus, minus = direction["theta_plus"], direction["theta_minus"]
    result["direction"] = {"gradient_max_abs": float(np.max(np.abs(gradient))),
        "nonzero_gradient_groups": int(np.count_nonzero(gradient)), "theta_plus_min": float(plus.min()),
        "theta_plus_max": float(plus.max()), "theta_minus_min": float(minus.min()), "theta_minus_max": float(minus.max()),
        "actual_rounded_displacement_sha256": bytes_sha(plus - minus)}
    patched = {}
    for label, gain in (("plus", plus), ("minus", minus)):
        csr, actual, info = marked("native_eligibility_" + label, lambda gain=gain: patch_weights(data["native"], gain))
        patched[label] = (csr, actual, info)
    result["native_eligibility"] = {label: info for label, (_, _, info) in patched.items()}
    notify(result.copy())  # immediately preserve palette eligibility before +/-
    save("direction", direction)
    save("theta-minus", {"theta64": minus, "theta32": minus.astype(np.float32)})
    for label, (csr, actual, _info) in patched.items():
        save("patched-csr-" + label, csr)
        save("selected-weights-" + label, {"actual32": actual,
            "original32": data["native"]["csr"]["data"][data["native"]["mask"]["positions"]],
            "positions": data["native"]["mask"]["positions"], "groups": data["native"]["mask"]["edge_parameters"]})
    plus_rows, minus_rows = [], []
    for c, baseline, g in zip(starts, neutral, gradients, strict=True):
        pv, _ = perform(c, "plus", tensor(plus))
        mv, _ = perform(c, "minus", tensor(minus))
        plus_rows.append(pv)
        minus_rows.append(mv)
        result["windows"].append({"start_control": c, "neutral": baseline, "plus": pv, "minus": mv,
            "observed_contribution": (pv["training_loss"] - mv["training_loss"]) / len(starts),
            "predicted_contribution": float(np.sum(g * (plus - minus), dtype=np.float64))})
        notify(result.copy())
    plus_mean = {key: float(np.mean([row[key] for row in plus_rows])) for key in TERM_NAMES}
    minus_mean = {key: float(np.mean([row[key] for row in minus_rows])) for key in TERM_NAMES}
    comparison = classify_direction(neutral_mean["training_loss"], plus_mean["training_loss"],
                                    minus_mean["training_loss"], gradient, plus, minus)
    result.update(comparison, plus_mean_terms=plus_mean, minus_mean_terms=minus_mean)
    reasons = continuation_reasons(result["local_direction"], result["local_descent"], patched["minus"][2]["eligible"])
    result["continuation"]["reasons"] = reasons
    notify(result.copy())
    if not reasons:
        result["continuation"]["requested"] = True
        value, _ = perform(0, "minus", tensor(minus), continuation=True)
        improvement = neutral_mean["training_loss"] - value["training_loss"]
        startup = bool(value["startup_pass"] and value["startup_q_rms"] <= .018 and value["startup_q_max"] <= .04
                       and value["startup_tendon_rms"] <= .009 and value["startup_tendon_max"] <= .025)
        result["continuation"].update(completed=True, metrics=value, improvement=improvement,
            startup_pass=startup, classification="supported" if improvement >= 1e-9 and startup else "rejected",
            original_control0_used=True, intermediate_saved_boundaries_overwritten=False)
    source_identity = data["native"].get("network_identity", "generated-source-network")
    candidate_identity = hashlib.sha256((source_identity + bytes_sha(minus)
        + patched["minus"][2]["patched_csr_data32_sha256"]).encode("ascii")).hexdigest()
    if patched["minus"][2]["selected_changed_bit_count"] and candidate_identity == source_identity:
        raise ValueError("Changed candidate content identity must differ from source identity")
    result["e331_handoff"] = {"eligible": result["continuation"]["classification"] == "supported",
        "theta_file": "theta-minus.npz", "theta64_sha256": bytes_sha(minus),
        "source_network_identity": source_identity, "candidate_content_identity": candidate_identity,
        "candidate_identity_kind": "source identity plus exact theta/CSR content; native identity must be recomputed by E331",
        "candidate_native_identity_claimed_equal_to_source": False,
        "original_control0_path": data.get("original_control0_path", "generated-control0"),
        "original_control0_pins": data.get("original_control0_pins", {}),
        "source_csr_file": data.get("original_csr_file", {}),
        "decoder_source_identity": data.get("start", {}).get("adapter", {}).get("hash", "generated-decoder"),
        "source_pins": data.get("handoff_source_pins", {}), "native_eligibility": patched["minus"][2],
        "startup_pass": result["continuation"].get("startup_pass"),
        "use_e330_terminal_state_as_body_start": False, "body_not_executed": True}
    result["actual_work"] = _counters(receipts, requested, result["decoder_replay_controls"])
    notify(result.copy())
    return result, receipts


put_json = e324.put_json

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
    result = {"schema": RESULT_SCHEMA, "diagnostic_id": "E330",
              "run_valid": False, "candidate_adopted": False,
              "biological_gait_proven": False, "F1_F8_complete": False,
              "F6_exposed": False, "M2_modified": False,
              "classification": "not_tested",
              "local_direction": "not_tested", "local_descent": None,
              "native_eligibility": {"status": "not_tested"},
              "continuation": {"classification": "not_tested", "requested": False, "completed": False},
              "body_not_executed": True,
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
    science_began = None
    science_deadline = began + LIMITS["internal_wall_s"] - LIMITS["finalization_reserve_s"]

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
            raise RuntimeError("E330 device identity changed")
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
            raise MemoryError("E330 host/device resource limit exceeded")

    def deadline():
        if clock() - began >= LIMITS["internal_wall_s"]:
            raise TimeoutError("E330 internal deadline exceeded")
        if clock() >= science_deadline:
            raise TimeoutError("E330 science deadline exceeded")

    def checkpoint(name, arrays):
        deadline()
        path = output / (name + ".npz")
        attempt = {"stage": "save_" + name, "requested": True,
                   "returned": None, "failed": False, "requested_elapsed_s": clock() - began}
        ledger["attempts"].append(attempt)
        persist()
        try:
            saver(path, **arrays)
        except Exception as error:  # noqa: BLE001
            attempt.update(failed=True, error_type=type(error).__name__, error_message=str(error)[:500])
            try:
                persist()
            except Exception as secondary:  # noqa: BLE001
                error.add_note("E330 failed-save journal failure: " + repr(secondary))
            raise
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
        check_resource(baseline)
        ledger["child_torch_cuda_before_input_loader"] = device == "cuda"
        writer(output / "source-manifest.json", {
            "schema": "flylab.m3.e330.source-manifest.v1",
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
                error.add_note("E330 input-load journal failure: " + repr(secondary))
            raise
        ledger["phases"]["input_load"]["host_returned"] = True
        ledger["phases"]["input_load"]["last_event_elapsed_s"] = clock() - began
        persist()
        deadline()
        check_resource(meter())
        science_began = clock()
        science_deadline = min(science_began + LIMITS["science_wall_s"],
                               began + LIMITS["internal_wall_s"] - LIMITS["finalization_reserve_s"])
        ledger["science_started_elapsed_s"] = science_began - began
        ledger["science_deadline_elapsed_s"] = science_deadline - began

        def on_receipt(name, value):
            ledger["receipts"][name] = value
            persist()

        def on_boundary(name, event):
            phases = ledger["phases"]
            if event == "requested":
                if name in phases:
                    raise ValueError("E330 phase was requested twice: " + name)
                phases[name] = {"requested": True, "host_returned": None,
                                "device_confirmed": None, "failed": False,
                                "requested_elapsed_s": clock() - began}
            elif event in ("host_returned", "device_confirmed"):
                phases[name][event] = True
            elif event == "failed":
                phases[name]["failed"] = True
            else:
                raise ValueError("Unknown E330 boundary event")
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
        result["classification"] = computed["local_direction"]
        check_resource(meter())
    except Exception as error:  # noqa: BLE001 - preserve original diagnostic failure
        failure("science", error)
        result["classification"] = "invalid"
        result["local_direction"] = "invalid"
        result["science_error"] = traceback.format_exc(limit=8)[-3000:]
    finally:
        ledger["science_wall_s"] = 0 if science_began is None else clock() - science_began
        finalization_began = clock()
        try:
            check_resource(meter())
        except Exception as error:  # noqa: BLE001
            failure("terminal_resources", error)
        result["first_error"] = first_error
        result["errors"] = ledger["errors"].copy()
        result["cost"] = ledger.copy()
        handoff = result.get("e331_handoff")
        if handoff is not None:
            handoff["output_file_sha256"] = {
                name: e323.sha256(output / name) for name in (
                    "theta-minus.npz", "patched-csr-minus.npz", "selected-weights-minus.npz")
                if (output / name).is_file()}
        try:
            writer(output / "receipts.json", {"schema": "flylab.m3.e330.receipts.v1",
                "receipts": ledger["receipts"],
                "requested_roster": result.get("requested_rollouts", result.get("partial", {}).get("requested_rollouts", []))})
        except Exception as error:  # noqa: BLE001
            failure("publish_receipts", error)
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
                raise OSError("E330 output size limit exceeded")
            writer(output / "output-manifest.json", {
                "schema": "flylab.m3.e330.output-manifest.v1", "files": files,
                "mutable_metadata_excluded": ["result.json", "advance-ledger.json",
                                              "publication-failure.json"],
                "external_audit_pins_required": True})
            if not (output / "output-manifest.json").is_file():
                raise OSError("E330 output manifest was not published")
            if sum(path.stat().st_size for path in output.rglob("*")
                   if path.is_file()) > LIMITS["output_bytes"]:
                raise OSError("E330 published output size limit exceeded")
        except Exception as error:  # noqa: BLE001
            failure("output_manifest", error)
        # All preliminary metadata/manifest work is inside this measured bound.
        ledger["publication_boundary"] = "before_terminal_metadata_commit"
        ledger["finalization_wall_s"] = clock() - finalization_began
        ledger["elapsed_s"] = clock() - began
        if ledger["finalization_wall_s"] > LIMITS["finalization_reserve_s"]:
            failure("finalization_deadline", TimeoutError("E330 finalization reserve exceeded"))
        if ledger["elapsed_s"] >= LIMITS["internal_wall_s"]:
            failure("internal_deadline", TimeoutError("E330 internal deadline exceeded"))
        result["first_error"] = first_error
        result["errors"] = ledger["errors"].copy()
        result["cost"] = ledger.copy()
        result["run_valid"] = first_error is None
        if handoff is not None and first_error is not None:
            handoff.update(eligible=False, ineligible_reason="diagnostic_or_publication_invalid")
        try:
            writer(output / "advance-ledger.json", ledger)
            writer(output / "result.json", result)
            if (json.loads((output / "advance-ledger.json").read_text("utf8")) != ledger
                    or json.loads((output / "result.json").read_text("utf8")) != result):
                raise OSError("E330 terminal metadata differs from return value")
        except Exception as error:  # noqa: BLE001
            failure("terminal_publication", error)
            result.update(run_valid=False, first_error=first_error,
                          errors=ledger["errors"].copy(), cost=ledger.copy())
            if handoff is not None:
                handoff.update(eligible=False, ineligible_reason="diagnostic_or_publication_invalid")
            try:
                writer(output / "publication-failure.json", result)
            except Exception as secondary:  # noqa: BLE001
                failure("publication_failure_journal", secondary)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("make-freeze", help="passive root helper; leaves original authorization false")
    prepare.add_argument("--output", type=Path, required=True)
    original = commands.add_parser("run", help="root-supervised original attempt only")
    for name in ("freeze", "review", "output"):
        original.add_argument("--" + name, type=Path, required=True)
    for name in ("freeze-sha256", "review-sha256"):
        original.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    if args.command == "make-freeze":
        with args.output.open("x", encoding="utf8") as stream:
            stream.write(json.dumps(make_freeze(), indent=2, allow_nan=False) + "\n")
        return
    result = run(args.freeze, args.freeze_sha256, args.review, args.review_sha256, args.output)
    print(json.dumps(result, allow_nan=False), flush=True)
    if not result["run_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
