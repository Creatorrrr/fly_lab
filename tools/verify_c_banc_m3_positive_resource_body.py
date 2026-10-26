"""E294 one static positive-only source resource policy; inherited body gates."""
from __future__ import annotations

# Preserve the first exception and subsequent runtime cleanup failures.
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
from tools import verify_c_banc_m3_recurrent_resource_body as prior

OWNED = ("tools/verify_c_banc_m3_positive_resource_body.py", "tests/test_c_banc_positive_resource_body.py")
SOURCE_FILES = tuple(dict.fromkeys((*OWNED, *prior.SOURCE_FILES)))
LIMITS, STARTUP_LIMITS = copy.deepcopy(prior.LIMITS), copy.deepcopy(prior.STARTUP_LIMITS)
FREEZE_SCHEMA = "flylab.m3.e294.positive-resource-body.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e294.positive-resource-body.review.v1"
PATHS = prior.PATHS | {"positive_selection"}
POLICY = "selected outgoing positive present and negative absent: u=.05; any negative including mixed: u=0"
BASE_ROSTER_PATH, BASE_ROSTER_SHA = prior.ROSTER_PATH, prior.ROSTER_SHA
BASE_FREEZE_PATH = "verification/banc-natural-gait-m3-20260919/e292-recurrent-resource-body-freeze.json"
OUTPUT_PATH = "verification/banc-natural-gait-m3-20260919/e294-positive-resource-body"
BASE_KEYS = ("graph_hash", "source_network_identity", "adapter_identity", "neutral_network_identity",
             "packages", "startup_limits", "resource_limits", "n", "edges", "premotor_count", "source_count", "motor_count",
             "neutral_utilization", "recovery_tau", "preservation_sha256", "e279_freeze", "e279_source_review",
             "e289_source_review", "e290_source_review", "tiny_qualification")
safe = prior.safe


def positive_source_selection(base_roster):
    """Copy original bytes; replace only u by immutable selected-edge sign policy."""
    keys = prior.layer.SELECTION_KEYS | {"neutral_utilization", "descriptive_edge_groups"}
    if not isinstance(base_roster, dict) or set(base_roster) != keys:
        raise ValueError("Exact original resource roster keys required")
    selected = {key: base_roster[key] for key in prior.layer.SELECTION_KEYS}
    prior.layer.validate_selection_metadata(selected, 158706, base_roster["motor_ids"])
    sources = len(base_roster["source_ids"])
    for key, value in (("utilization", .05), ("neutral_utilization", 0.), ("recovery_tau", .150)):
        prior.layer.require_equal(base_roster[key], np.full(sources, value, np.float64), "Original base " + key)
    groups = base_roster["descriptive_edge_groups"]
    if (not isinstance(groups, np.ndarray) or groups.dtype != np.int64
            or groups.shape != base_roster["edge_ids"].shape or np.any(groups < 0)):
        raise ValueError("Native descriptive groups required")
    slots, weights = base_roster["edge_source_slots"], base_roster["weights32"]
    positive, negative = np.zeros(sources, bool), np.zeros(sources, bool)
    np.logical_or.at(positive, slots, weights > 0)
    np.logical_or.at(negative, slots, weights < 0)
    result = copy.deepcopy(base_roster)
    result["utilization"] = np.where(positive & ~negative, .05, 0.).astype(np.float64)
    return result


def verify_spec(spec):
    if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
        raise ValueError("Exact pinned E294 receipt spec required")
    path = safe(spec["path"])
    if prior.protocol.sha256(path) != spec["sha256"]:
        raise ValueError("E294 receipt SHA mismatch")
    return json.loads(path.read_text(encoding="utf-8"))


def pin_groups(*groups):
    pins = {}
    for group in groups:
        for name, digest in group.items():
            if name in pins and pins[name] != digest:
                raise ValueError("Conflicting E294 lineage pin")
            pins[name] = digest
    for name, digest in pins.items():
        if not isinstance(digest, str) or len(digest) != 64 or prior.protocol.sha256(safe(name)) != digest:
            raise ValueError("Frozen E294/base bytes changed: " + name)
    return pins


def base_lineage(freeze):
    if freeze["base_freeze"]["path"] != BASE_FREEZE_PATH:
        raise ValueError("Literal frozen E292 baseline required")
    base, review, audit = (verify_spec(freeze[name]) for name in ("base_freeze", "base_review", "base_original_audit"))
    if (base.get("schema") != prior.FREEZE_SCHEMA or base.get("diagnostic_id") != "E292"
            or base.get("owned_files") != list(prior.OWNED) or set(base.get("paths", {})) != prior.PATHS
            or base.get("startup_limits") != STARTUP_LIMITS or base.get("resource_limits") != LIMITS
            or base.get("n") != 158706 or base.get("edges") != 725849 or base.get("source_count") != 7136
            or base.get("premotor_count") != 7136 or base.get("motor_count") != 391
            or base.get("neutral_utilization") != 0. or base.get("recovery_tau") != .150 or not base.get("packages")
            or base.get("utilization") != .05 or base.get("candidate_adopted") is not False
            or base["paths"]["selection"] != BASE_ROSTER_PATH
            or base["input_sha256"].get(BASE_ROSTER_PATH) != BASE_ROSTER_SHA
            or review.get("schema") != prior.REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != [] or review.get("freeze_sha256") != freeze["base_freeze"]["sha256"]
            or set(review.get("source_sha256", {})) != set(prior.OWNED)
            or audit.get("schema") != "flylab.m3.e292.root-original-raw-audit.v1"
            or audit.get("pass") is not True or audit.get("run_valid") is not True
            or audit.get("actual_startup_pass") is not False or audit.get("short_gait_pass") is not False
            or audit.get("candidate_adopted") is not False):
        raise ValueError("Approved validly rejected E292 baseline lineage required")
    pins = pin_groups(base["input_sha256"], base["reused_source_sha256"], base["preservation_sha256"],
                      review["source_sha256"], {base["design"]: base["design_sha256"]})
    if not set(prior.SOURCE_FILES) <= set(pins) or not base["preservation_sha256"]:
        raise ValueError("Complete frozen E292 sources/preservation required")
    old = verify_spec(base["e279_freeze"])
    old_review = verify_spec(base["e279_source_review"])
    if (old.get("schema") != prior.prior.FREEZE_SCHEMA or old_review.get("schema") != prior.prior.REVIEW_SCHEMA
            or old_review.get("approved") is not True or old_review.get("blocking_defects") != []
            or old_review.get("freeze_sha256") != base["e279_freeze"]["sha256"]):
        raise ValueError("Approved original E279 lineage required")
    for name, digest in old_review["source_sha256"].items():
        if pins.get(name) != digest:
            raise ValueError("Original E279 source lineage differs")
    for key in ("graph_hash", "source_network_identity", "adapter_identity", "packages"):
        if base[key] != old[key]:
            raise ValueError("Original E279 identity differs: " + key)
    for alias in prior.prior.PATHS - {"candidate", "predictions", "selection"}:
        if base["paths"][alias] != old["paths"][alias]:
            raise ValueError("Original baseline input path differs")
        path = safe(base["paths"][alias])
        files = [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else [path]
        for file in files:
            name = file.relative_to(ROOT).as_posix()
            if base["input_sha256"].get(name) != old["input_sha256"].get(name):
                raise ValueError("Original baseline input SHA differs")
    for alias in ("e289_source_review", "e290_source_review", "tiny_qualification"):
        receipt = verify_spec(base[alias])
        expected = {name: pins[name] for name in prior.PRODUCERS
                    if alias == "tiny_qualification" or ("synapses" if alias == "e289_source_review" else "trial") in name}
        if receipt.get("approved") is not True or receipt.get("blocking_defects") != [] or receipt.get("source_sha256") != expected:
            raise ValueError("Unchanged current producer/tiny approval required")
    return base, pins


def gate(freeze_path, freeze_sha, review_path, review_sha, output):
    if prior.protocol.sha256(freeze_path) != freeze_sha or prior.protocol.sha256(review_path) != review_sha:
        raise ValueError("E294 freeze/review SHA mismatch")
    freeze, review = (json.loads(Path(path).read_text(encoding="utf-8")) for path in (freeze_path, review_path))
    if (freeze.get("schema") != FREEZE_SCHEMA or freeze.get("diagnostic_id") != "E294"
            or freeze.get("owned_files") != list(OWNED) or freeze.get("policy") != POLICY
            or set(freeze.get("paths", {})) != PATHS or freeze.get("candidate_adopted") is not False
            or freeze.get("output") != OUTPUT_PATH):
        raise ValueError("Exact fixed E294 freeze required")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != [] or review.get("freeze_sha256") != freeze_sha
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Approved matching E294 source review required")
    base, base_pins = base_lineage(freeze)
    for key in BASE_KEYS:
        if freeze.get(key) != base[key]:
            raise ValueError("Frozen E292 original context changed: " + key)
    for alias in prior.PATHS:
        if freeze["paths"][alias] != base["paths"][alias]:
            raise ValueError("Original frozen E292 input path changed")
    for name, digest in base["input_sha256"].items():
        if freeze["input_sha256"].get(name) != digest:
            raise ValueError("Original frozen E292 input bytes changed")
    pins = pin_groups(freeze["input_sha256"], freeze["reused_source_sha256"], review["source_sha256"],
                      freeze["preservation_sha256"], {freeze["design"]: freeze["design_sha256"]})
    if not set(SOURCE_FILES) <= set(pins) or any(pins.get(name) != digest for name, digest in base_pins.items() if name != base["design"]):
        raise ValueError("Exact inherited E294 execution source pins required")
    for alias, name in freeze["paths"].items():
        path = safe(name)
        files = [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else [path]
        if not files or any(file.relative_to(ROOT).as_posix() not in freeze["input_sha256"] for file in files):
            raise ValueError("Every required E294 input byte must be pinned: " + alias)
    for name, version in freeze["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("E294 frozen package differs")
    NativeDependencies().load(freeze)
    if output.resolve() != safe(OUTPUT_PATH) or output.exists() or not output.parent.is_dir():
        raise ValueError("Fresh fixed E294 output required")
    if shutil.disk_usage(output.parent).free < LIMITS["minimum_free_disk_bytes"]:
        raise OSError("E294 disk reserve unavailable")
    return freeze


class NativeDependencies(prior.NativeDependencies):
    def load_base(self, base_freeze):
        return super().load(base_freeze)

    def load_selection(self, path):
        return prior.protocol.load_npz(safe(path))

    def load(self, freeze):
        if freeze["base_freeze"]["path"] != BASE_FREEZE_PATH:
            raise ValueError("Literal E292 baseline provenance required")
        base_freeze = verify_spec(freeze["base_freeze"])
        if base_freeze.get("schema") != prior.FREEZE_SCHEMA or base_freeze.get("diagnostic_id") != "E292":
            raise ValueError("Literal approved E292 baseline load required")
        # Literal E292 all-sign baseline is loaded and validated before policy
        # replacement. It is never presented as approval of the new candidate.
        data = self.load_base(base_freeze)
        roster = positive_source_selection(data["roster"])
        frozen_roster = self.load_selection(freeze["paths"]["positive_selection"])
        prior.layer.require_equal(frozen_roster, roster, "Exactly computed positive source roster")
        selection = {key: roster[key] for key in prior.layer.SELECTION_KEYS}
        metadata = prior.neural.resource_metadata(data["saved"]["network"]["model_hash"], selection)
        if (prior.neural.resource_identity(metadata) != freeze["candidate_network_identity"]
                or freeze["neutral_network_identity"] != base_freeze["neutral_network_identity"]):
            raise ValueError("Independently computed E294 candidate/neutral identity differs")
        data.update(selection=selection, roster=roster, freeze=freeze)
        data["e294_lineage"] = {"baseline_loaded_diagnostic": "E292",
            "new_candidate_separately_verified": True, "policy": POLICY,
            "nonzero_utilization_sources": int(np.count_nonzero(roster["utilization"]))}
        return data

    def write(self, path, value):
        if path.name == "source-manifest.json":
            value = {**value, "schema": "flylab.m3.e294.sources.v1", "diagnostic_id": "E294",
                "owned_sha256": {name: prior.protocol.sha256(ROOT / name) for name in OWNED},
                "source_sha256": {**value.get("source_sha256", {}),
                    **{name: prior.protocol.sha256(ROOT / name) for name in SOURCE_FILES if name not in OWNED}},
                "executed_body_runner": "tools.verify_c_banc_m3_recurrent_resource_body.execute"}
        return super().write(path, value)


def execute(freeze, output, *, deps=None, budget=None, cleanup=None):
    if freeze.get("diagnostic_id") != "E294":
        raise ValueError("E294 diagnostic identity required")
    return prior.execute(freeze, output, deps=NativeDependencies() if deps is None else deps, budget=budget, cleanup=cleanup)


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
        for callback in ((lambda: limiter.restore_original_limits()) if limiter else lambda: None,
                         lambda: torch.set_num_threads(previous_threads)):
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
            for failure in failures[1:]:
                failures[0].add_note("Secondary E294 runtime cleanup: " + repr(failure))
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
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--source-review", required=True, type=Path)
    parser.add_argument("--source-review-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    return run(args.freeze, args.freeze_sha256, args.source_review, args.source_review_sha256, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
