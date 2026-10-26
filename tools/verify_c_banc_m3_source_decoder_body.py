"""E288 one selected source+decoder candidate, exact neutral then native body."""

from __future__ import annotations

# Keep first errors, actual ledgers and every partial publication/close failure.
# ruff: noqa: BLE001
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import banc_m3_monotone_recruitment as decoder
from tools import banc_m3_source_filter_trial as layer
from tools import verify_c_banc_m3_source_native_transfer as prior

protocol, native, short, gait, receipts = (
    prior.protocol,
    prior.native,
    prior.short,
    prior.gait,
    prior.receipts,
)
LEGS, STARTUP_LIMITS, LIMITS = (
    prior.LEGS,
    copy.deepcopy(prior.STARTUP_LIMITS),
    copy.deepcopy(prior.LIMITS),
)
FREEZE_SCHEMA, REVIEW_SCHEMA = (
    "flylab.m3.e288.source-decoder-body.freeze.v1",
    "flylab.m3.e288.source-decoder-body.review.v1",
)
OWNED = (
    "tools/verify_c_banc_m3_source_decoder_body.py",
    "tests/test_c_banc_source_decoder_body.py",
)
E287_SOURCES = (
    "tools/banc_m3_source_decoder_fit.py",
    "tools/fit_c_banc_m3_source_decoder.py",
    "tests/test_c_banc_source_decoder_fit.py",
)
SOURCE_FILES = (*OWNED, *prior.SOURCE_FILES, *E287_SOURCES)
PATHS = (prior.PATHS - {"predictions"}) | {
    "predictions",
    "candidate_metadata",
    "final_result",
    "selection_result",
    "e287_root_audit",
    "source_candidate",
    "baseline_neutral_final_checkpoint",
}
FACTOR_KEYS = (
    "tau",
    "kappa",
    "slopes",
    "membership",
    "motor_ids",
    "moment",
    "rest",
    "pool",
    "ltm",
    "fixed",
)
CANDIDATE_KEYS = {
    "coefficients32",
    "coefficients64",
    "motor_ids",
    "source_ids",
    "edge_ids",
    "membership",
    "tau",
    "kappa",
    "slopes",
    "low",
    "high",
    "moment",
    "rest",
    "pool",
    "ltm",
    "fixed",
}
BASELINE_FORWARD_MM = 1.6790085779712631
safe = prior.safe


def array_factor(values, keys):
    digest = hashlib.sha256()
    for name in keys:
        value = np.ascontiguousarray(values[name])
        digest.update(
            name.encode()
            + value.dtype.str.encode()
            + np.asarray(value.shape, np.int64).tobytes()
            + value.tobytes()
        )
    return digest.hexdigest()


def factor_identities(candidate):
    return {
        "source_factor_sha256": array_factor(candidate, ("coefficients32",)),
        "decoder_factor_sha256": array_factor(candidate, FACTOR_KEYS),
    }


def verify_spec(spec):
    if (
        not isinstance(spec, dict)
        or set(spec) != {"path", "sha256"}
        or protocol.sha256(safe(spec["path"])) != spec["sha256"]
    ):
        raise ValueError("Exact pinned E288 evidence spec required")
    return safe(spec["path"])


def selection_gate(freeze):
    candidate_sha = freeze["input_sha256"][freeze["paths"]["candidate"]]
    for alias, schema in (
        ("final_result", "flylab.m3.e287.source-decoder-fit.v1"),
        ("selection_result", "flylab.m3.e287.source-decoder-fit.v1"),
        ("e287_root_audit", "flylab.m3.e287.root-original-audit.v1"),
    ):
        value = json.loads(safe(freeze["paths"][alias]).read_text(encoding="utf-8"))
        if (
            value.get("schema") != schema
            or value.get("run_valid") is not True
            or value.get("selection_pass") is not True
            or value.get("candidate_sha256") != candidate_sha
        ):
            raise ValueError(
                "E287 valid independent selected candidate required: " + alias
            )
    metadata = json.loads(
        safe(freeze["paths"]["candidate_metadata"]).read_text(encoding="utf-8")
    )
    if (
        metadata.get("candidate_sha256") != candidate_sha
        or metadata.get("decoder_identity") != freeze["candidate_payload_identity"]
    ):
        raise ValueError("Exact E287 candidate payload/SHA required")
    return metadata


def gate(freeze_path, freeze_sha, review_path, review_sha, output):
    if (
        protocol.sha256(freeze_path) != freeze_sha
        or protocol.sha256(review_path) != review_sha
    ):
        raise ValueError("E288 freeze/review SHA mismatch")
    freeze, review = (
        json.loads(Path(p).read_text(encoding="utf-8"))
        for p in (freeze_path, review_path)
    )
    if (
        freeze.get("schema") != FREEZE_SCHEMA
        or freeze.get("owned_files") != list(OWNED)
        or set(freeze.get("paths", {})) != PATHS
        or freeze.get("startup_limits") != STARTUP_LIMITS
        or freeze.get("resource_limits") != LIMITS
        or freeze.get("n") != 158706
        or freeze.get("edges") != 1660
        or freeze.get("source_count") != 51
        or freeze.get("motor_count") != 391
        or freeze.get("candidate_adopted") is not False
    ):
        raise ValueError("Exact E288 scientific/resource freeze required")
    if (
        review.get("schema") != REVIEW_SCHEMA
        or review.get("approved") is not True
        or review.get("blocking_defects") != []
        or review.get("freeze_sha256") != freeze_sha
        or set(review.get("source_sha256", {})) != set(OWNED)
    ):
        raise ValueError("E288 approved matching source review required")
    pins = {}
    for group in (
        freeze["input_sha256"],
        freeze["reused_source_sha256"],
        review["source_sha256"],
        freeze.get("preservation_sha256", {}),
        {freeze["design"]: freeze["design_sha256"]},
    ):
        for name, digest in group.items():
            if name in pins and pins[name] != digest:
                raise ValueError("Conflicting E288 source/input SHA")
            pins[name] = digest
    if not set(SOURCE_FILES) <= set(pins):
        raise ValueError("All actual E288/E279/E287 source pins required")
    for name, digest in pins.items():
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or protocol.sha256(safe(name)) != digest
        ):
            raise ValueError("Frozen E288 bytes changed: " + name)
    for alias, name in freeze["paths"].items():
        path = safe(name)
        children = (
            [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else [path]
        )
        if not children or any(
            p.relative_to(ROOT).as_posix() not in freeze["input_sha256"]
            for p in children
        ):
            raise ValueError("Every required input byte must be pinned: " + alias)
    e287 = json.loads(verify_spec(freeze["e287_freeze"]).read_text(encoding="utf-8"))
    e287_review = json.loads(
        verify_spec(freeze["e287_source_review"]).read_text(encoding="utf-8")
    )
    if (
        e287_review.get("schema") != "flylab.m3.e287.source-review.v1"
        or e287_review.get("verdict") != "approved"
        or e287_review.get("freeze_sha256") != freeze["e287_freeze"]["sha256"]
        or set(e287_review.get("approved_sources", {})) != set(E287_SOURCES)
    ):
        raise ValueError("Approved E287 producer source review required")
    for name, digest in e287_review["approved_sources"].items():
        if pins.get(name) != digest:
            raise ValueError("E287 producer source pin differs")
    if not isinstance(e287, dict) or not e287.get("schema"):
        raise ValueError("Actual E287 freeze required")
    old = json.loads(verify_spec(freeze["e279_freeze"]).read_text(encoding="utf-8"))
    oldreview = json.loads(
        verify_spec(freeze["e279_source_review"]).read_text(encoding="utf-8")
    )
    if (
        old.get("schema") != prior.FREEZE_SCHEMA
        or oldreview.get("schema") != prior.REVIEW_SCHEMA
        or oldreview.get("approved") is not True
        or oldreview.get("freeze_sha256") != freeze["e279_freeze"]["sha256"]
    ):
        raise ValueError("Frozen approved E279 reusable path required")
    for name, digest in oldreview["source_sha256"].items():
        if pins.get(name) != digest:
            raise ValueError("Reused E279 source identity differs")
    for alias in prior.PATHS - {"candidate", "predictions"}:
        if freeze["paths"][alias] != old["paths"][alias]:
            raise ValueError("Original E279 input/context changed: " + alias)
    if freeze["paths"]["source_candidate"] != old["paths"]["candidate"]:
        raise ValueError("Fixed E277 source candidate required")
    for key in (
        "graph_hash",
        "source_network_identity",
        "adapter_identity",
        "neutral_network_identity",
        "candidate_network_identity",
        "packages",
        "prediction_joint_columns",
        "prediction_ltm_columns",
    ):
        if freeze.get(key) != old[key]:
            raise ValueError("Original native provenance changed: " + key)
    for name, version in freeze["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Frozen E288 package changed")
    prior.prediction_columns(freeze)
    selection_gate(freeze)
    c = protocol.load_npz(safe(freeze["paths"]["candidate"]))
    digest = hashlib.sha256(b"flylab.m3.e287.decoder-payload.v1")
    for name in sorted(c):
        value = np.ascontiguousarray(c[name])
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(np.asarray(value.shape, np.int64).tobytes())
        digest.update(value.tobytes())
    if digest.hexdigest() != freeze["candidate_payload_identity"]:
        raise ValueError("Exact E287 producer payload digest required")
    if not CANDIDATE_KEYS <= set(c) or factor_identities(c) != {
        k: freeze[k] for k in ("source_factor_sha256", "decoder_factor_sha256")
    }:
        raise ValueError("Exact source+decoder factors required")
    if (
        output.resolve() != safe(freeze["output"])
        or output.exists()
        or not output.parent.is_dir()
    ):
        raise ValueError("Fresh fixed E288 output required")
    if shutil.disk_usage(output.parent).free < LIMITS["minimum_free_disk_bytes"]:
        raise OSError("E288 disk reserve unavailable")
    return freeze


def validate_zero_checkpoint(saved):
    layer.validate_neutral(saved["network"])
    if saved.get("control_tick") != 0:
        raise ValueError("Original whole neutral control tick zero required")
    a = saved["adapter"]
    for path, value, shape in (
        ("bank", a["bank"], (391, 3)),
        ("activation", a["activation"], (96,)),
        ("original.activation", a["original"]["activation"], (96,)),
    ):
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.float64
            or value.shape != shape
            or not layer.equal(value, np.zeros(shape, np.float64))
        ):
            raise ValueError("Original neutral decoder zero bytes required: " + path)


def candidate_adapter_metadata(saved, candidate):
    """Actual native adapter construction from pure anatomical metadata; no body."""
    validate_zero_checkpoint(saved)
    old = saved["adapter"]
    parameters = old["parameters"]
    if not CANDIDATE_KEYS <= set(candidate):
        raise ValueError("Complete combined candidate payload required")
    for key in ("moment", "rest", "pool", "motor_ids", "ltm"):
        layer.require_equal(
            candidate[key], old["calibration"][key], "Original anatomical " + key
        )
    for key in ("tau", "kappa", "membership"):
        layer.require_equal(
            candidate[key], parameters[key], "Original fixed decoder " + key
        )

    def no_decode(*args, **kwargs):
        raise RuntimeError("Pure metadata adapter cannot drive a body")

    anatomical = SimpleNamespace(
        identity=old["original"]["hash"],
        calibration=copy.deepcopy(old["calibration"]),
        groups=copy.deepcopy(old["groups"]),
        activation=old["original"]["activation"].copy(),
        last_motor=copy.deepcopy(old["original"].get("last_motor", [])),
        decode=no_decode,
        snapshot=lambda: copy.deepcopy(old["original"]),
    )
    adapter = decoder.MonotoneRecruitmentAdapter(
        anatomical,
        candidate["tau"],
        candidate["kappa"],
        candidate["slopes"],
        candidate["membership"],
        parameters["motor_roster"],
    )
    fixed = np.zeros(96, bool)
    fixed[adapter.fixed_muscles] = True
    layer.require_equal(
        candidate["fixed"], fixed, "Original LTM/singleton fixed decoder rows"
    )
    supported = candidate["slopes"][candidate["membership"], np.arange(391)]
    layer.require_equal(candidate["low"], supported[:, :, 0], "Explicit low slopes")
    layer.require_equal(candidate["high"], supported[:, :, 1], "Explicit high slopes")
    if (
        candidate["coefficients32"].dtype != np.float32
        or candidate["coefficients32"].shape != (1660, 3)
        or not layer.equal(
            candidate["coefficients64"], candidate["coefficients32"].astype(np.float64)
        )
    ):
        raise ValueError("Exact widened fixed float32 source coefficients required")
    return adapter.snapshot()


def expected_candidate_checkpoint(saved, candidate):
    """Replace only approved parameter/hash and their derived metadata leaves."""
    metadata = candidate_adapter_metadata(saved, candidate)
    expected = copy.deepcopy(saved)
    allowed = [
        "adapter.hash",
        "adapter.parameters.slopes",
        "monotone_recruitment_patch.adapter_identity",
        "monotone_recruitment_patch.slopes",
    ]
    expected["adapter"]["hash"] = metadata["hash"]
    expected["adapter"]["parameters"]["slopes"] = metadata["parameters"][
        "slopes"
    ].copy()
    patch = expected["monotone_recruitment_patch"]
    patch["adapter_identity"] = metadata["hash"]
    patch["slopes"] = candidate["slopes"].copy()
    for key in ("fixed_muscles", "lower_budget", "upper_budget", "motor_minimum_slope"):
        expected["adapter"]["parameters"][key] = metadata["parameters"][key].copy()
        patch[key] = metadata["parameters"][key].copy()
        allowed.extend(
            ("adapter.parameters." + key, "monotone_recruitment_patch." + key)
        )
    # Assert the native fresh adapter matches this exact sparse replacement;
    # calibration/groups/original/last_motor/bank/activation are never excluded.
    layer.require_equal(
        expected["adapter"],
        metadata,
        "Native candidate adapter sparse state replacement",
    )
    return expected, allowed


def source_identity(saved_network, selection, coefficients):
    native = {
        "edge_ids": selection["edge_ids"],
        "coefficients64": coefficients,
        "coefficients32": coefficients.astype(np.float32),
        "free_motor_ids": selection["allowed_motor_ids"],
        "pre_ids": selection["pre_ids"],
        "post_ids": selection["post_ids"],
        "weights32": selection["original_weights"].astype(np.float32),
        "source_ids": selection["source_ids"],
        "edge_source_slots": np.searchsorted(
            selection["source_ids"], selection["pre_ids"]
        ).astype(np.int32),
    }
    metadata = {
        "schema": layer.SCHEMA,
        "source_model_hash": saved_network["model_hash"],
        "precision": layer.PRECISION,
        "tau": layer.TAU.copy(),
        **native,
    }
    return layer.filter_identity(metadata)


def assess_short(baseline, candidate):
    """Native [.5,1.5) health/forward/slip, eligible bouts and six full cycles."""
    b, c = baseline, candidate

    def finite(value):
        return type(value) in (int, float) and np.isfinite(value)

    bg, cg = b.get("gait", {}), c.get("gait", {})
    bw, cw = bg.get("window", {}), cg.get("window", {})
    window = all(
        w.get("start_s") == 0.5
        and w.get("end_s") == 1.5
        and w.get("sampled_duration_s") == 1.0
        for w in (bw, cw)
    )

    def eligible(w):
        bouts = w.get("bouts")
        if not isinstance(bouts, list):
            return None
        durations = []
        for bout in bouts:
            if not isinstance(bout, dict):
                return None
            start, end, samples = (bout.get(k) for k in ("start_s", "end_s", "samples"))
            if (
                not finite(start)
                or not finite(end)
                or not 0.5 <= start <= end < 1.5
                or type(samples) is not int
                or not 2 <= samples <= 200
            ):
                return None
            # Match the original analyzer's sample-count duration. Endpoint
            # subtraction can round an exact 61-sample .3 s bout below .3.
            duration = (samples - 1) * 0.005
            if duration < 0.3 or abs((end - start) - duration) > 1e-12:
                return None
            # Both endpoints must also belong to the frozen [.5, 1.5) 5 ms
            # control grid, rather than merely having a compatible difference.
            if any(
                abs((value - 0.5) - round((value - 0.5) / 0.005) * 0.005) > 1e-12
                for value in (start, end)
            ):
                return None
            durations.append(duration)
        return max(durations, default=0.0)

    be, ce = eligible(bw), eligible(cw)
    bf, cf = b.get("forward", {}), c.get("forward", {})
    bs, cs = b.get("slip", {}), c.get("slip", {})
    fkey, skey = "fixed_initial_heading_forward_mm", "load_weighted_mm_s"
    forward = bool(
        bf.get("available") is True
        and cf.get("available") is True
        and bf.get("window_samples") == cf.get("window_samples") == 200
        and bf.get("last_included_control") == cf.get("last_included_control") == 299
        and finite(bf.get(fkey))
        and finite(cf.get(fkey))
        and bf[fkey] == BASELINE_FORWARD_MM
        and cf[fkey] >= bf[fkey]
    )
    slip = bool(
        bs.get("available") is True
        and cs.get("available") is True
        and finite(bs.get(skey))
        and finite(cs.get(skey))
        and cs[skey] >= 0
        and bs[skey] >= 0
        and cs[skey] <= bs[skey]
        and cs.get("missing_selected_slip") == bs.get("missing_selected_slip") == 0
        and finite(bs.get("support_samples"))
        and bs["support_samples"] > 0
        and finite(bs.get("support_load_sum_bw"))
        and bs["support_load_sum_bw"] > 0
        and finite(cs.get("support_samples"))
        and cs["support_samples"] > 0
        and finite(cs.get("support_load_sum_bw"))
        and cs["support_load_sum_bw"] > 0
    )

    def complete_cycle(cycle):
        if not isinstance(cycle, dict):
            return False
        a, p, z = (cycle.get(k) for k in ("aep_s", "pep_s", "next_aep_s"))
        if not all(finite(v) for v in (a, p, z)) or not 0.5 <= a < p < z < 1.5:
            return False
        return any(
            bout["start_s"] <= a < z <= bout["end_s"]
            for bout in cw.get("bouts", [])
            if isinstance(bout, dict)
            and finite(bout.get("start_s"))
            and finite(bout.get("end_s"))
        )

    cycles = {}
    for leg in LEGS:
        br, cr = bg.get("events", {}).get(leg, {}), cg.get("events", {}).get(leg, {})
        bc, cc = br.get("complete_cycles"), cr.get("complete_cycles")
        cycles[leg] = bool(
            type(bc) is int
            and type(cc) is int
            and bc == 0
            and cc > bc
            and isinstance(cr.get("cycles"), list)
            and len(cr["cycles"]) == cc
            and all(complete_cycle(cycle) for cycle in cr["cycles"])
        )
    checks = {
        "same_complete_window": window,
        "native_health": all(
            v.get("health", {}).get(k) is True
            for v in (b, c)
            for k in (
                "healthy",
                "complete_1p5s",
                "clock_exact",
                "observed_clock_valid",
                "finite",
            )
        ),
        "health_thresholds": all(
            finite(v.get("health", {}).get("minimum_native_upright"))
            and v["health"]["minimum_native_upright"] >= 0.9
            and type(v["health"].get("nonfoot_contact_count")) is int
            and v["health"]["nonfoot_contact_count"] == 0
            for v in (b, c)
        ),
        "forward_nonworse": forward,
        "supported_slip_nonworse": slip,
        "eligible_bout": bool(
            be is not None and ce is not None and ce >= 0.3 and ce >= be
        ),
        "six_leg_complete_cycles": all(cycles.values()),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "cycles": cycles,
        "baseline_longest_eligible_bout_s": be,
        "candidate_longest_eligible_bout_s": ce,
        "window_s": [0.5, 1.5],
        "natural_gait_complete": False,
    }


class NativeDependencies(prior.NativeDependencies):
    def load(self, freeze):
        from flylab.c.graph import GraphStore
        from flylab.c.storage import StateStore

        paths = freeze["paths"]
        saved = StateStore.load(safe(paths["neutral_checkpoint"]))
        validate_zero_checkpoint(saved)
        candidate = protocol.load_npz(safe(paths["candidate"]))
        source = protocol.load_npz(safe(paths["source_candidate"]))
        layer.require_equal(
            candidate["coefficients32"],
            source["coefficients32"],
            "Unmodified E277 float32 source factor",
        )
        expected, allowed = expected_candidate_checkpoint(saved, candidate)
        if (
            expected["adapter"]["hash"] != freeze["candidate_adapter_identity"]
            or saved["adapter"]["hash"] != freeze["neutral_adapter_identity"]
        ):
            raise ValueError("Frozen actual native adapter identities changed")
        selection = protocol.load_npz(safe(paths["selection"]))
        for key, source_key in (
            ("motor_ids", "all_motor_ids"),
            ("source_ids", "source_ids"),
            ("edge_ids", "edge_ids"),
        ):
            layer.require_equal(
                candidate[key], selection[source_key], "Original selected " + key
            )
        if not layer.equal(selection["allowed_motor_ids"].shape, (343,)):
            raise ValueError("Exact original allowed343 source motor roster required")
        neutral_coefficients = np.tile(np.array([1.0, 0.0, 0.0]), (1660, 1))
        for arm, c in (
            ("neutral", neutral_coefficients),
            ("candidate", candidate["coefficients64"]),
        ):
            if (
                source_identity(saved["network"], selection, c)
                != freeze[arm + "_network_identity"]
            ):
                raise ValueError("Pure frozen source-network identity differs")
        graph = GraphStore.load(safe(paths["graph_bundle"]))
        if (
            graph.hash != freeze["graph_hash"]
            or saved["network"]["model_hash"] != freeze["source_network_identity"]
            or saved["adapter"]["hash"] != freeze["adapter_identity"]
        ):
            raise ValueError("Original graph/network/decoder source identities changed")
        prediction = protocol.load_npz(safe(paths["predictions"]))
        pq, pl = prediction["q"], prediction["ltm"]
        if (
            pq.shape != (101, 42)
            or pl.shape != (101, 6)
            or pq.dtype != np.float64
            or pl.dtype != np.float64
            or not np.isfinite(pq).all()
            or not np.isfinite(pl).all()
        ):
            raise ValueError("Complete E287 conditional startup prediction required")
        data = {
            "graph": graph,
            "saved": saved,
            "candidate": candidate,
            "coefficients": candidate["coefficients64"],
            "selection": selection,
            "allowed_motor_ids": selection["allowed_motor_ids"],
            "expected_candidate": expected,
            "allowed_changes": allowed,
            "baseline_end": StateStore.load(
                safe(paths["baseline_neutral_final_checkpoint"])
            ),
            "startup": protocol.load_npz(safe(paths["startup"])),
            "baseline_rows": json.loads(
                safe(paths["baseline_trace"]).read_text(encoding="utf-8")
            )["samples"],
            "baseline_native": protocol.load_npz(safe(paths["baseline_native"])),
            "baseline_recruitment": protocol.load_npz(
                safe(paths["baseline_recruitment"])
            ),
            "baseline_short": json.loads(
                safe(paths["baseline_short_result"]).read_text(encoding="utf-8")
            ),
            "predicted_q": pq,
            "predicted_ltm": pl,
            "freeze": freeze,
            "source_ids": selection["source_ids"],
            "motor_ids": candidate["motor_ids"],
        }
        data["ids"] = np.unique(np.r_[data["source_ids"], data["motor_ids"]]).astype(
            np.int64
        )
        if factor_identities(candidate) != {
            k: freeze[k] for k in ("source_factor_sha256", "decoder_factor_sha256")
        }:
            raise ValueError("Combined factor hashes changed")
        return data

    def construct(self, arm, data, ledger, persist, on_return):
        def callback():
            owner = None
            try:
                if arm == "neutral":
                    owner = decoder.MonotoneRecruitmentTrial.from_checkpoint(
                        data["graph"], copy.deepcopy(data["saved"])
                    )
                else:
                    source = copy.deepcopy(data["saved"])
                    source.pop("monotone_recruitment_patch")
                    source["schema"] = decoder.MotorTimescaleTrial.schema
                    source["adapter"] = copy.deepcopy(
                        data["saved"]["adapter"]["original"]
                    )
                    owner = decoder.MotorTimescaleTrial.from_checkpoint(
                        data["graph"], source
                    )
                    c = data["candidate"]
                    owner = decoder.MonotoneRecruitmentTrial(
                        owner, c["tau"], c["kappa"], c["slopes"], c["membership"]
                    )
                coefficients = (
                    np.tile(np.array([1.0, 0.0, 0.0]), (1660, 1))
                    if arm == "neutral"
                    else data["coefficients"].copy()
                )
                owner = layer.SourceFilterTrial(
                    owner,
                    data["selection"]["edge_ids"],
                    coefficients,
                    data["allowed_motor_ids"],
                )
                owner._e288_arm = arm
                return owner
            except Exception as error:
                if owner is not None:
                    layer.cleanup_owned(owner, error)
                raise

        return short.construction_with_body_count(
            arm, callback, ledger, persist, on_return
        )

    def initial(self, session, data, arm):
        saved = session.snapshot()
        changes = []
        actual = prior.checkpoint_representation(
            layer.remove_source_layer(saved), changes
        )
        expected = data["saved"] if arm == "neutral" else data["expected_candidate"]
        expected_changes = []
        expected = prior.checkpoint_representation(expected, expected_changes)
        layer.require_equal(
            actual, expected, "Exact whole source+decoder initial state"
        )
        validate_zero_checkpoint(actual)
        if (
            session.network.identity != data["freeze"][arm + "_network_identity"]
            or session.adapter.identity != data["freeze"][arm + "_adapter_identity"]
            or session.control_tick != 0
            or session.network.tick != 0
            or abs(session.body.physics_time()) > 2e-8
            or session.network.device != "cuda"
            or session.network.cuda_implementation != "packed"
            or session.network.interval_steps != 20
            or not layer.equal(
                saved["network"]["source_history"], np.zeros((51, 2), np.float64)
            )
        ):
            raise ValueError(
                "Exact CUDA neutral clocks/history and selected identities required"
            )
        return {
            "storage_representation_exact": True,
            "allowed_parameter_changes": []
            if arm == "neutral"
            else data["allowed_changes"],
            "actual_representation_changes": changes,
            "expected_representation_changes": expected_changes,
            "excluded_fields": [],
            "array_coercions": 0,
            "numeric_tolerance": 0,
        }

    def capture(self, session, raw):
        super().capture(session, raw)
        if session._e288_arm == "neutral" and session.control_tick == 1:
            raw["neutral_end_snapshot"] = session.snapshot()

    def compare_neutral(self, raw, data):
        result = super().compare_neutral(raw, data)
        expected = data["baseline_end"]
        if "source_filter_trial_patch" in expected:
            expected = layer.remove_source_layer(expected)
        actual_changes, expected_changes = [], []
        actual = prior.checkpoint_representation(
            layer.remove_source_layer(raw["neutral_end_snapshot"]), actual_changes
        )
        expected = prior.checkpoint_representation(expected, expected_changes)
        layer.require_equal(actual, expected, "Whole neutral .005 checkpoint exact")
        return {
            **result,
            "whole_end_exact": True,
            "excluded_fields": [],
            "array_coercions": 0,
            "numeric_tolerance": 0,
        }

    def save_raw(self, folder, raw, data, clock):
        failures = super().save_raw(folder, raw, data, clock)
        try:
            self.write(
                folder / "trace.json",
                {
                    "schema": "flylab.m3.e288.trace.v1",
                    "samples": raw["rows"],
                    "graph_hash": data["graph"].hash,
                    "controller": "E287 fixed source plus selected monotone decoder;original E150 physical/sensory/controller",
                    **raw.get("trace_metadata", {}),
                },
            )
        except Exception as error:
            failures.append({"artifact": "E288 trace.json", "error": repr(error)})
        return failures


def execute(freeze, out, *, deps=None, budget=None, cleanup=None):
    deps = NativeDependencies() if deps is None else deps
    began = time.perf_counter()
    out.mkdir(exist_ok=False)
    ledger = {
        "schema": "flylab.m3.e288.ledger.v1",
        "arms": {},
        "construction": [],
        "body_construction": [],
        "original_attempts": 1,
        "retries": 0,
        "fit_calls": 0,
        "nonneutral_current_calls": 0,
    }
    result = {
        "schema": "flylab.m3.e288.result.v1",
        "validity": "incomplete",
        "candidate_adopted": False,
        "diagnostic_id": freeze.get("diagnostic_id", "E288"),
        "E277_status": "rejected",
        "run_valid": False,
        "neutral_exact": False,
        "actual_startup_pass": False,
        "short_gait_pass": False,
        "next_separate_10s_eligible": False,
        "natural_gait_complete": False,
        "F1_F8_complete": False,
        "reused_arm_receipt_path": "tools.verify_c_banc_m3_source_native_transfer.run_arm",
        "E272_executed": False,
        "F6_exposed": False,
        "F2_complete": False,
        "F3_F8_pass": False,
        "ten_second_executed": False,
        "arms": {},
    }
    errors, primary = [], None

    def check():
        if time.perf_counter() - began >= LIMITS["science_wall_s"]:
            raise TimeoutError("E288 science deadline/reserve reached")
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
                raise MemoryError("E288 host/GPU cap exceeded")
            if shutil.disk_usage(out).free < LIMITS["minimum_free_disk_bytes"]:
                raise OSError("E288 output disk reserve reached")

    try:
        deps.write(
            out / "source-manifest.json",
            {
                "schema": "flylab.m3.e288.sources.v1",
                "input_sha256": freeze.get("input_sha256", {}),
                "source_sha256": freeze.get("reused_source_sha256", {}),
                "owned_sha256": {name: protocol.sha256(ROOT / name) for name in OWNED},
            },
        )
        data = deps.load(freeze)
        neutral, primary = prior.run_arm(
            "neutral", data, freeze, out, ledger, deps, check
        )
        result["arms"]["neutral"] = neutral
        result["neutral_exact"] = (
            primary is None
            and neutral.get("neutral_prefix", {}).get("exact") is True
            and neutral.get("neutral_prefix", {}).get("whole_end_exact") is True
        )
        if primary is None and not result["neutral_exact"]:
            raise ValueError("Whole neutral exact comparison required before candidate")
        if result["neutral_exact"]:
            candidate, primary = prior.run_arm(
                "candidate", data, freeze, out, ledger, deps, check
            )
            result["arms"]["candidate"] = candidate
            result["actual_startup_pass"] = (
                primary is None
                and candidate.get("startup", {}).get("strict_E141_startup_screen")
                is True
            )
            if primary is None and candidate.get("completed_controls") == 300:
                result["short_adjudication"] = assess_short(
                    data["baseline_short"], candidate.get("short", {})
                )
                result["short_gait_pass"] = result["short_adjudication"]["pass"]
            result["next_separate_10s_eligible"] = (
                result["actual_startup_pass"] and result["short_gait_pass"]
            )
        result["run_valid"] = primary is None
        result["validity"] = (
            "valid_source_decoder_body_diagnostic"
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
    ):
        primary = primary or RuntimeError(
            "E288 scientific clocks/caps unavailable or exceeded"
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
    try:
        deps.write(out / "advance-ledger.json", ledger)
        deps.write(out / "errors.json", {"errors": errors})
        manifest = short.output_manifest(out, LIMITS["output_bytes"])
        manifest["schema"] = "flylab.m3.e288.output-manifest.v1"
        deps.write(out / "output-manifest.json", manifest)
        if time.perf_counter() - began >= LIMITS["internal_wall_s"]:
            raise TimeoutError("E288 final publication deadline")
        deps.write(out / "result.json", result)
        if time.perf_counter() - began >= LIMITS["internal_wall_s"]:
            raise TimeoutError("E288 late result publication deadline")
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
        errors.append({"stage": "publication", "message": str(error)})
        (out / "output-manifest.json").unlink(missing_ok=True)
        try:
            deps.write(out / "errors.json", {"errors": errors})
            deps.write(out / "result.json", result)
        except Exception as secondary:
            primary.add_note("Secondary E288 invalid publication: " + repr(secondary))
            (out / "result.json").unlink(missing_ok=True)
    return result, primary


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    freeze = gate(freeze_path, freeze_sha, review_path, review_sha, output)
    import cupy
    import torch
    from threadpoolctl import threadpool_limits

    old_threads, old_env = torch.get_num_threads(), {}
    limiter, cleaned = None, False

    def cleanup():
        nonlocal cleaned
        if cleaned:
            return
        cleaned = True
        failures = []
        for callback in (
            (lambda: limiter.restore_original_limits()) if limiter else lambda: None,
            lambda: torch.set_num_threads(old_threads),
        ):
            try:
                callback()
            except Exception as error:
                failures.append(error)
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if failures:
            for error in failures[1:]:
                failures[0].add_note("Secondary runtime cleanup: " + repr(error))
            raise failures[0]

    try:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            old_env[name] = os.environ.get(name)
            os.environ[name] = "1"
        torch.set_num_threads(1)
        limiter = threadpool_limits(limits=1)
        deps = NativeDependencies()
        deps.gpu_free_before, _ = cupy.cuda.runtime.memGetInfo()
        result, error = execute(freeze, output, deps=deps, cleanup=cleanup)
        return (
            0
            if error is None
            and result["validity"] == "valid_source_decoder_body_diagnostic"
            else 1
        )
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
    return run(
        args.freeze,
        args.freeze_sha256,
        args.source_review,
        args.source_review_sha256,
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
