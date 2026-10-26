"""E296 original core-afferent source exemption; inherited body gates."""
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
from tools import verify_c_banc_m3_positive_resource_body as prior

OWNED = ("tools/verify_c_banc_m3_core_afferent_resource_body.py", "tests/test_c_banc_core_afferent_resource_body.py")
E295_OWNED = ("tools/analyze_c_banc_m3_core_resource_localization.py", "tests/test_c_banc_core_resource_localization.py")
SOURCE_FILES = tuple(dict.fromkeys((*OWNED, *prior.SOURCE_FILES, *E295_OWNED)))
LIMITS, STARTUP_LIMITS = copy.deepcopy(prior.LIMITS), copy.deepcopy(prior.STARTUP_LIMITS)
FREEZE_SCHEMA = "flylab.m3.e296.core-afferent-resource-body.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e296.core-afferent-resource-body.review.v1"
PATHS = prior.PATHS | {"core_reference", "protected_selection"}
POLICY = "E294 positive-only source u=.05; any selected outgoing into original core24 exempts shared source to u=0"
BASE_FREEZE_PATH = "verification/banc-natural-gait-m3-20260919/e294-positive-resource-body-freeze.json"
OUTPUT_PATH = "verification/banc-natural-gait-m3-20260919/e296-core-afferent-resource-body"
BASE_KEYS = prior.BASE_KEYS
CORE_REFERENCE_PATH = "verification/banc-natural-gait-m3-20260919/e257-lh-partition-body/baseline/core-activity.npz"
safe = prior.safe


def protected_source_selection(original_uniform_roster, core_ids):
    """Shared source exemption for any selected edge into fixed original core."""
    roster = prior.positive_source_selection(original_uniform_roster)
    if (not isinstance(core_ids, np.ndarray) or core_ids.dtype != np.int64 or core_ids.ndim != 1
            or not len(core_ids) or len(np.unique(core_ids)) != len(core_ids)
            or not np.isin(core_ids, roster["premotor_ids"]).all()):
        raise ValueError("Unique native original premotor core IDs required")
    chosen = np.isin(roster["post_ids"], core_ids)
    slots = np.unique(roster["edge_source_slots"][chosen])
    roster["utilization"][slots] = 0.
    return roster


def verify_spec(spec):
    if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
        raise ValueError("Exact pinned E296 receipt spec required")
    path = safe(spec["path"])
    if prior.prior.protocol.sha256(path) != spec["sha256"]:
        raise ValueError("E296 receipt SHA mismatch")
    return json.loads(path.read_text(encoding="utf-8"))


def pin_groups(*groups):
    pins = {}
    for group in groups:
        for name, digest in group.items():
            if name in pins and pins[name] != digest:
                raise ValueError("Conflicting E296 lineage pin")
            pins[name] = digest
    for name, digest in pins.items():
        if not isinstance(digest, str) or len(digest) != 64 or prior.prior.protocol.sha256(safe(name)) != digest:
            raise ValueError("Frozen E296/base bytes changed: " + name)
    return pins


def base_lineage(freeze):
    if freeze["base_freeze"]["path"] != BASE_FREEZE_PATH:
        raise ValueError("Literal frozen E294 baseline required")
    base, review, audit = (verify_spec(freeze[name]) for name in ("base_freeze", "base_review", "base_original_audit"))
    if (base.get("schema") != prior.FREEZE_SCHEMA or base.get("diagnostic_id") != "E294"
            or base.get("owned_files") != list(prior.OWNED) or set(base.get("paths", {})) != prior.PATHS
            or base.get("policy") != prior.POLICY or base.get("candidate_adopted") is not False
            or review.get("schema") != prior.REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != [] or review.get("freeze_sha256") != freeze["base_freeze"]["sha256"]
            or set(review.get("source_sha256", {})) != set(prior.OWNED)
            or audit.get("schema") != "flylab.m3.e294.root-original-raw-audit.v1"
            or audit.get("pass") is not True or audit.get("run_valid") is not True
            or audit.get("actual_startup_pass") is not False or audit.get("short_gait_pass") is not False
            or audit.get("candidate_adopted") is not False):
        raise ValueError("Approved validly rejected E294 baseline lineage required")
    # Validate the already frozen E294 -> E292 -> E279 approval lineage without
    # calling its fresh-output gate against an existing completed body run.
    e292, e292_pins = prior.base_lineage(base)
    for key in prior.BASE_KEYS:
        if base.get(key) != e292[key]:
            raise ValueError("Unchanged original E294/E292 context required: " + key)
    for alias in prior.prior.PATHS:
        if base["paths"][alias] != e292["paths"][alias]:
            raise ValueError("Original inherited baseline path differs")
    for name, digest in e292["input_sha256"].items():
        if base["input_sha256"].get(name) != digest:
            raise ValueError("Original inherited input SHA differs")
    pins = pin_groups(base["input_sha256"], base["reused_source_sha256"], base["preservation_sha256"],
                      review["source_sha256"], {base["design"]: base["design_sha256"]})
    if (not set(prior.SOURCE_FILES) <= set(pins)
            or any(pins.get(name) != digest for name, digest in e292_pins.items() if name != e292["design"])):
        raise ValueError("Complete unchanged E294 actual inherited source/preservation pins required")
    return base, pins


def core_lineage(freeze, base):
    evidence, audit = (verify_spec(freeze[name]) for name in ("e295_freeze", "e295_root_audit"))
    if (evidence.get("schema") != "flylab.m3.e295.core-resource-localization.freeze.v1"
            or evidence.get("owned_files") != list(E295_OWNED)
            or audit.get("schema") != "flylab.m3.e295.root-saved-audit.v1" or audit.get("pass") is not True
            or any(audit.get(key) is not False for key in ("candidate_selected", "causal_claim", "F1_F8_complete"))):
        raise ValueError("Frozen unpromoted E295 saved-analysis root evidence required")
    source_pins = pin_groups(evidence["source_sha256"])
    if not set(E295_OWNED) <= set(source_pins):
        raise ValueError("Actual E295 source producer SHA required")
    core_path = freeze["paths"]["core_reference"]
    if (core_path != CORE_REFERENCE_PATH or evidence["inputs"]["baseline_core_activity"] != core_path
            or evidence["input_sha256"].get(core_path) != freeze["input_sha256"].get(core_path)
            or evidence["inputs"]["positive_roster"] != base["paths"]["positive_selection"]
            or evidence["input_sha256"].get(base["paths"]["positive_selection"]) != base["input_sha256"].get(base["paths"]["positive_selection"])):
        raise ValueError("Original core-reference and E294 positive roster evidence must match")
    manifest_name = evidence["inputs"]["baseline_e257_output_manifest"]
    pin_groups({manifest_name: evidence["input_sha256"][manifest_name]})
    manifest = json.loads(safe(manifest_name).read_text(encoding="utf-8"))
    path, parent = safe(core_path), safe(manifest_name).parent
    if not path.is_relative_to(parent):
        raise ValueError("Original E257 core must belong to its producer manifest")
    entry = manifest.get("files", {}).get(path.relative_to(parent).as_posix())
    if (not isinstance(entry, dict) or entry.get("sha256") != freeze["input_sha256"][core_path]
            or entry.get("bytes") != path.stat().st_size):
        raise ValueError("Exact original E257 core producer raw SHA required")
    return evidence


def gate(freeze_path, freeze_sha, review_path, review_sha, output):
    if prior.prior.protocol.sha256(freeze_path) != freeze_sha or prior.prior.protocol.sha256(review_path) != review_sha:
        raise ValueError("E296 freeze/review SHA mismatch")
    freeze, review = (json.loads(Path(path).read_text(encoding="utf-8")) for path in (freeze_path, review_path))
    if (freeze.get("schema") != FREEZE_SCHEMA or freeze.get("diagnostic_id") != "E296"
            or freeze.get("owned_files") != list(OWNED) or freeze.get("policy") != POLICY
            or set(freeze.get("paths", {})) != PATHS or freeze.get("candidate_adopted") is not False
            or freeze.get("output") != OUTPUT_PATH):
        raise ValueError("Exact fixed E296 freeze required")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != [] or review.get("freeze_sha256") != freeze_sha
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Approved matching E296 source review required")
    base, base_pins = base_lineage(freeze)
    for key in BASE_KEYS:
        if freeze.get(key) != base[key]:
            raise ValueError("Frozen E294 original context changed: " + key)
    core_lineage(freeze, base)
    for alias in prior.PATHS:
        if freeze["paths"][alias] != base["paths"][alias]:
            raise ValueError("Original frozen E294 input path changed")
    for name, digest in base["input_sha256"].items():
        if freeze["input_sha256"].get(name) != digest:
            raise ValueError("Original frozen E294 input bytes changed")
    pins = pin_groups(freeze["input_sha256"], freeze["reused_source_sha256"], review["source_sha256"],
                      freeze["preservation_sha256"], {freeze["design"]: freeze["design_sha256"]})
    if not set(SOURCE_FILES) <= set(pins) or any(pins.get(name) != digest for name, digest in base_pins.items() if name != base["design"]):
        raise ValueError("Exact inherited E296 execution source pins required")
    for alias, name in freeze["paths"].items():
        path = safe(name)
        files = [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else [path]
        if not files or any(file.relative_to(ROOT).as_posix() not in freeze["input_sha256"] for file in files):
            raise ValueError("Every required E296 input byte must be pinned: " + alias)
    for name, version in freeze["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("E296 frozen package differs")
    NativeDependencies().load(freeze)
    if output.resolve() != safe(OUTPUT_PATH) or output.exists() or not output.parent.is_dir():
        raise ValueError("Fresh fixed E296 output required")
    if shutil.disk_usage(output.parent).free < LIMITS["minimum_free_disk_bytes"]:
        raise OSError("E296 disk reserve unavailable")
    return freeze


class NativeDependencies(prior.NativeDependencies):
    def load_e294(self, base_freeze):
        return super().load(base_freeze)

    def load(self, freeze):
        if freeze["base_freeze"]["path"] != BASE_FREEZE_PATH:
            raise ValueError("Literal E294 baseline provenance required")
        base = verify_spec(freeze["base_freeze"])
        if base.get("schema") != prior.FREEZE_SCHEMA or base.get("diagnostic_id") != "E294":
            raise ValueError("Literal approved E294 baseline load required")
        data = self.load_e294(base)
        uniform = self.load_selection(freeze["paths"]["selection"])
        positive = prior.positive_source_selection(uniform)
        prior.prior.layer.require_equal(positive, data["roster"], "Exactly loaded E294 positive baseline roster")
        reference = self.load_selection(freeze["paths"]["core_reference"])
        core_ids = reference.get("core_ids")
        if not isinstance(core_ids, np.ndarray) or core_ids.shape != (24,):
            raise ValueError("Frozen complete original core24 required")
        roster = protected_source_selection(uniform, core_ids)
        frozen = self.load_selection(freeze["paths"]["protected_selection"])
        prior.prior.layer.require_equal(frozen, roster, "Exactly computed protected source roster")
        selection = {key: roster[key] for key in prior.prior.layer.SELECTION_KEYS}
        metadata = prior.prior.neural.resource_metadata(data["saved"]["network"]["model_hash"], selection)
        if (prior.prior.neural.resource_identity(metadata) != freeze["candidate_network_identity"]
                or freeze["neutral_network_identity"] != base["neutral_network_identity"]):
            raise ValueError("Independently computed protected candidate/neutral identity differs")
        data.update(selection=selection, roster=roster, freeze=freeze)
        data["e296_lineage"] = {"baseline_loaded_diagnostic": "E294", "new_candidate_separately_verified": True,
            "policy": POLICY, "core_ids": core_ids.tolist(), "nonzero_utilization_sources": int(np.count_nonzero(roster["utilization"])),
            "shared_source_other_outgoing_also_exempt": True, "core_rate_preservation_claim": False}
        return data

    def write(self, path, value):
        if path.name == "source-manifest.json":
            value = {**value, "schema": "flylab.m3.e296.sources.v1", "diagnostic_id": "E296",
                "owned_sha256": {name: prior.prior.protocol.sha256(ROOT / name) for name in OWNED},
                "source_sha256": {**value.get("source_sha256", {}),
                    **{name: prior.prior.protocol.sha256(ROOT / name) for name in SOURCE_FILES if name not in OWNED}},
                "executed_body_runner": "tools.verify_c_banc_m3_recurrent_resource_body.execute"}
        # E294.write would relabel this manifest as E294. Use the approved E291
        # bounded writer directly, without changing either inherited module.
        return prior.prior.NativeDependencies.write(self, path, value)


def execute(freeze, output, *, deps=None, budget=None, cleanup=None):
    if freeze.get("diagnostic_id") != "E296":
        raise ValueError("E296 diagnostic identity required")
    return prior.prior.execute(freeze, output, deps=NativeDependencies() if deps is None else deps, budget=budget, cleanup=cleanup)


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
                failures[0].add_note("Secondary E296 runtime cleanup: " + repr(failure))
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
