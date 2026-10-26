"""E334 one frozen constrained smooth secant, one fresh forward, no body.

Passive import/gate never import Torch or open NPZ. Original CLI requires fresh
root CPU/CUDA qualification, current source review and a separate 60s reservation.
Generated fixtures require explicit TestDimensions; no CLI dimension override.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
BASE = "verification/banc-natural-gait-m3-20260919"
PRIOR = BASE + "/e333-smooth-efficacy-direction"
OUTPUT = BASE + "/e334-smooth-constrained-secant"
N, GROUPS, CONTROLS = 158706, 8547, 300
OWNED = ("tools/verify_c_banc_m3_smooth_constrained_secant.py", "tests/test_c_banc_smooth_constrained_secant.py")
HELPER = "tools/verify_c_banc_m3_constrained_secant.py"
HELPER_SHA = "f39a29c5b6c56061e7311b34b75c8914454425e9e1cec3010d3552591cda868f"
SCHEMA = "flylab.m3.e334.smooth-constrained-secant.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e334.smooth-constrained-secant.review.v1"
RESULT_SCHEMA = "flylab.m3.e334.smooth-constrained-secant.result.v1"
QUAL_SCHEMA = "flylab.m3.e334.generated-qualification.v1"
LIMITS = {"supervisor_wall_s": 60, "internal_wall_s": 55, "science_wall_s": 45,
          "finalization_reserve_s": 10, "host_bytes": 6 << 30, "additional_gpu_bytes": 8 << 30,
          "output_bytes": 512 << 20, "minimum_free_disk_bytes": 4 << 30,
          "cpu_threads": 1, "gpu_count": 1, "forward_neural_ticks": 6000,
          "forward_rollout_calls": 1, "recompute_neural_ticks": 0, "backward_blocks": 0,
          "body_calls": 0, "optimizer_steps": 0, "retry_count": 0}
ALLOCATION = {"cumulative_before_s": 9405.698900999521, "cumulative_limit_s": 9660,
              "original_reserved_s": 60, "body_reserved_s": 0}
PATHS = {"prior_freeze": BASE + "/e333-smooth-efficacy-direction-freeze.json",
         "prior_review": BASE + "/parent-e333-original-source-review.json",
         "prior_decision": BASE + "/parent-e333-final-decision.json",
         "prior_audit": BASE + "/parent-e333-original-raw-audit.json",
         "prior_audit_source": BASE + "/parent-e333-original-raw-audit.py",
         "prior_audit_supervision": BASE + "/parent-e333-audit-supervision.json",
         "prior_supervision": BASE + "/parent-e333-original-supervision.json",
         "prior_contract": BASE + "/parent-e333-original-contract.json",
         "prior_result": PRIOR + "/result.json", "prior_ledger": PRIOR + "/advance-ledger.json",
         "prior_manifest": PRIOR + "/output-manifest.json", "prior_producer": PRIOR + "/source-manifest.json",
         "prior_receipts": PRIOR + "/receipts.json",
         "design": "docs/BANC_M3_E334_SMOOTH_CONSTRAINED_SECANT_DESIGN_20261001.md",
         "astra": BASE + "/parent-e334-astra-design-review.md",
         "draft": BASE + "/parent-e334-pre-astra-design.md",
         "invocation": BASE + "/parent-e334-astra-invocation.json",
         "events": BASE + "/parent-e334-astra-review-events.jsonl",
         "reflection": BASE + "/parent-e334-design-reflection.json",
         "secant": BASE + "/parent-e334-design-secant.json",
         "secant_source": BASE + "/parent-e334-design-secant.py",
         "secant_supervision": BASE + "/parent-e334-design-secant-supervision.json",
         "secant_contract": BASE + "/parent-e334-design-secant-contract.json",
         "closure": BASE + "/parent-e334-root-inherited-closure.json",
         "closure_source": BASE + "/parent-e334-root-inherited-closure.py",
         "closure_supervision": BASE + "/parent-e334-root-closure-supervision.json",
         "closure_contract": BASE + "/parent-e334-root-closure-contract.json",
         "prior_audit_contract": BASE + "/parent-e333-audit-contract.json",
         "qualification_source": BASE + "/parent-e334-generated-qualification.py",
         "qualification": BASE + "/parent-e334-generated-qualification.json"}
ANCHOR_SHA = {"prior_freeze": "40f4092f5aeca4ee4d67fdc7b8bee6ae55e89d8f2904b74ed009c4385b897d89",
              "prior_decision": "cfaf4aa9acbfe4c6ea272ddb96ca0632590cd2e4e184a35edc9b9a53cfe0647a",
              "design": "8459d17acb9958e89458d740d268b9f15d0530e751ce6cd4b6c9e107d77635e5",
              "astra": "1e3e2105943075f6b2838476b73027c83c90fea948d6061d6d7e445b82b3b3cb"}
THETA_SHA = "d7c8f19833c7364d8f345a5ae4dec2444a4be20606d5b23daf267e15525a7e40"
RAW_FILES = ("secant-inputs.npz", "theta-star.npz", "selected-weights-star.npz", "patched-csr-star.npz",
             "continuation-start.npz", "continuation-forward.npz", "continuation-metrics.npz")
GUARDS = {"q_rms": .018, "q_max": .04, "tendon_rms": .009, "tendon_max": .025}


@dataclass(frozen=True)
class TestDimensions:
    nodes: int
    groups: int
    controls: int = 300
    test_only: bool = True

    def __post_init__(self):
        if (self.test_only is not True or type(self.nodes) is not int or self.nodes < 391
                or type(self.groups) is not int or self.groups < 1 or self.controls != 300):
            raise ValueError("explicit generated N>=391/groups/300 dimensions required")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bytes_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read(path):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("duplicate JSON key")
            obj[key] = value
        return obj
    def invalid(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(Path(path).read_text("utf8"), object_pairs_hook=unique, parse_constant=invalid)


def safe(name, root=ROOT):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("path outside repository")
    return path


def pin_map(pins, root=ROOT, timely=lambda: None):
    if not isinstance(pins, dict) or not pins:
        raise ValueError("nonempty digest map required")
    for name, digest in pins.items():
        timely()
        if (not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                or sha256(safe(name, root)) != digest):
            raise ValueError("current SHA mismatch: " + name)


def _merge(target, added):
    for name, digest in added.items():
        if name in target and target[name] != digest:
            raise ValueError("inherited digest overwrite forbidden: " + name)
        target[name] = digest


def _closure(root):
    prior = read(safe(PATHS["prior_freeze"], root))
    if len(prior["source_sha256"]) != 180 or len(prior["input_sha256"]) != 1660:
        raise ValueError("exact inherited 180source/1660input required")
    sources, inputs = prior["source_sha256"].copy(), prior["input_sha256"].copy()
    _merge(sources, {HELPER: HELPER_SHA, **{name: sha256(safe(name, root)) for name in OWNED}})
    manifest = read(safe(PATHS["prior_manifest"], root))
    _merge(inputs, {PRIOR + "/" + name: row["sha256"] for name, row in manifest["files"].items()})
    _merge(inputs, {name: sha256(safe(name, root)) for name in PATHS.values()})
    for role in ("prior_decision", "qualification"):
        _merge(inputs, read(safe(PATHS[role], root)).get("evidence_sha256", {}))
    return prior, sources, inputs


def make_freeze(*, root=ROOT):
    """Passive; missing fresh generated qualification prevents original entry."""
    prior, sources, inputs = _closure(root)
    producer = read(safe(PATHS["secant"], root))
    return {"schema": SCHEMA, "paths": PATHS.copy(), "limits": LIMITS.copy(), "output": OUTPUT,
            "source_sha256": sources, "input_sha256": inputs,
            "smooth_constant_metadata": prior["smooth_constant_metadata"],
            "secant": {"alpha_star": producer["alpha_star"], "intersection": producer["intersection"],
                       "secant_intervals": producer["secant_intervals"], "theta_star64_bytes_sha256": THETA_SHA},
            "candidate_adopted": False, "original_execution_authorized": False,
            "original_supervisor_required": True, "allocation": None}


def _terminated(proof, *, success=True):
    if (proof.get("schema") != "flylab.m3.e267.root-supervision.v1"
            or (success and proof.get("success") is not True) or len(proof.get("attempts", [])) != 1
            or not proof.get("owned_pids") or any(psutil.pid_exists(pid) for pid in proof["owned_pids"])
            or proof["attempts"][0].get("killed") is not False
            or (success and proof["attempts"][0].get("exit_code") != 0)):
        raise ValueError("completed owned supervisor required")


def gate(freeze_path, freeze_sha256, review_path, review_sha256, output, *, root=ROOT, timely=lambda: None):
    """Strict closed producer/source/qualification proof before numerical keys."""
    timely()
    if (Path(output).exists() or Path(output).resolve() != safe(OUTPUT, root)
            or sha256(freeze_path) != freeze_sha256 or sha256(review_path) != review_sha256):
        raise ValueError("fresh literal E334 output/exact freeze-review SHA required")
    frozen, review = read(freeze_path), read(review_path)
    if (frozen.get("schema") != SCHEMA or frozen.get("paths") != PATHS or frozen.get("limits") != LIMITS
            or frozen.get("output") != OUTPUT or frozen.get("original_execution_authorized") is not True
            or frozen.get("original_supervisor_required") is not True or frozen.get("allocation") != ALLOCATION
            or frozen.get("candidate_adopted") is not False):
        raise ValueError("exact separately authorized/reserved E334 freeze required")
    sources, pins = frozen["source_sha256"], frozen["input_sha256"]
    pin_map(sources, root, timely)
    pin_map(pins, root, timely)
    if any(sources[name] != pins[name] for name in sources.keys() & pins.keys()):
        raise ValueError("contradictory source/input digest")
    for role, digest in ANCHOR_SHA.items():
        if pins.get(PATHS[role]) != digest:
            raise ValueError("immutable E334/prior anchor differs")
    prior, expected_sources, expected_pins = _closure(root)
    expected = make_freeze(root=root)
    if set(frozen) != set(expected) or any(frozen[key] != expected[key] for key in expected.keys() - {"allocation", "original_execution_authorized"}):
        raise ValueError("exact E334 closure/calculation constants required")
    if sources != expected_sources or pins != expected_pins:
        raise ValueError("exact inherited closure required")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != [] or review.get("source_sha256") != sources
            or review.get("freeze_sha256") != freeze_sha256):
        raise ValueError("current root source review required")
    qual = read(safe(PATHS["qualification"], root))
    if (qual.get("schema") != QUAL_SCHEMA or qual.get("approved") is not True or qual.get("blocking_defects") != []
            or qual.get("original_calls") != 0 or qual.get("source_sha256") != {name: sources[name] for name in OWNED}
            or not qual.get("evidence_sha256")):
        raise ValueError("fresh exact-own2 generated qualification required")
    for proof in (review, qual):
        for name, digest in proof.get("evidence_sha256", {}).items():
            if {**pins, **sources}.get(name) != digest:
                raise ValueError("review/qualification evidence differs")
    for mode in ("cpu", "cuda"):
        desc = qual.get(mode + "_report", {})
        if (set(desc) != {"path", "sha256"} or pins.get(desc["path"]) != desc["sha256"]
                or qual["evidence_sha256"].get(desc["path"]) != desc["sha256"]):
            raise ValueError("exact current CPU/CUDA report descriptor required")
        proof = read(safe(desc["path"], root))
        if (proof.get("schema") != "flylab.m3.e334.root-generated.v1" or proof.get("mode") != mode
                or proof.get("passed") is not True or not proof.get("checks")
                or not all(v is True for v in proof["checks"].values())
                or proof.get("source_sha256") != sources[OWNED[0]]
                or proof.get("audit_source_sha256") != pins[PATHS["qualification_source"]]
                or any(proof.get(key) != 0 for key in ("original_calls", "whole_BANC_calls", "body_calls"))):
            raise ValueError("current generated producer schema/source proof required")
    evidence = {role: read(safe(PATHS[role], root)) for role in ("prior_decision", "prior_audit", "prior_result",
        "prior_ledger", "prior_manifest", "prior_producer", "prior_review", "prior_receipts", "prior_supervision",
        "prior_audit_supervision", "prior_contract", "secant", "secant_supervision", "secant_contract",
        "closure", "closure_supervision", "closure_contract", "prior_audit_contract", "invocation", "reflection")}
    decision, audit, result, producer = (evidence[k] for k in ("prior_decision", "prior_audit", "prior_result", "prior_producer"))
    if (decision.get("schema") != "flylab.m3.e333.root-final-decision.v1" or decision.get("formal_verdict") != "invalid_deadline"
            or decision.get("run_valid") is not False or decision.get("effective_direction_qualification") is not False
            or decision.get("effective_continuation_qualification") is not False
            or decision.get("original_cumulative_s") != ALLOCATION["cumulative_before_s"]
            or decision.get("original_allocation_limit_s") != 9660 or decision.get("original_reserved_s") != 0
            or decision.get("whole_goal_status") != "active" or decision.get("body_calls") != 0
            or result.get("schema") != "flylab.m3.e333.smooth-efficacy-direction.result.v1" or result.get("run_valid") is not False
            or result.get("effective_direction_qualification") is not False or result.get("effective_continuation_qualification") is not False
            or result.get("first_error", {}).get("type") != "TimeoutError"
            or audit.get("schema") != "flylab.m3.e333.root-original-raw-audit.v1" or audit.get("passed") is not True
            or len(audit.get("checks", {})) != 3414 or not all(v is True for v in audit["checks"].values())
            or producer.get("schema") != "flylab.m3.e333.source-manifest.v1"
            or producer.get("source_sha256") != prior["source_sha256"] or producer.get("input_sha256") != prior["input_sha256"]
            or producer.get("freeze_sha256") != pins[PATHS["prior_freeze"]]
            or producer.get("review_sha256") != pins[PATHS["prior_review"]]):
        raise ValueError("E333 invalid/deadline/root3414/original producer required")
    for field, role in (("result_sha256", "prior_result"), ("advance_ledger_sha256", "prior_ledger"),
                        ("output_manifest_sha256", "prior_manifest"), ("source_manifest_sha256", "prior_producer"),
                        ("receipts_sha256", "prior_receipts"), ("supervision_sha256", "prior_supervision"),
                        ("freeze_sha256", "prior_freeze"), ("audit_source_sha256", "prior_audit_source")):
        if audit.get(field) != pins[PATHS[role]]:
            raise ValueError("E333 audit linkage differs: " + field)
    manifest = evidence["prior_manifest"]
    if (manifest.get("schema") != "flylab.m3.e333.output-manifest.v1" or len(manifest.get("files", {})) != 612
            or sum(name.endswith(".npz") for name in manifest["files"]) != 610
            or evidence["prior_receipts"].get("schema") != "flylab.m3.e333.receipts.v1"):
        raise ValueError("E333 exact 610raw+2JSON roster required")
    for name, desc in manifest["files"].items():
        if desc != {"sha256": pins[PRIOR + "/" + name], "bytes": safe(PRIOR + "/" + name, root).stat().st_size}:
            raise ValueError("E333 immutable manifest descriptor differs")
    receipt = evidence["prior_receipts"]["receipts"]["continuation-minus"]
    if receipt.get("confirmed") != {"forward_controls": 300, "forward_neural_ticks": 6000,
            "recompute_controls": 0, "recompute_neural_ticks": 0, "backward_blocks": 0, "backward_block_ticks": 0}:
        raise ValueError("E333 saved final forward receipt required, not qualification")
    _terminated(evidence["prior_supervision"], success=False)
    _terminated(evidence["prior_audit_supervision"])
    for role in ("secant_supervision", "closure_supervision"):
        _terminated(evidence[role])
    if evidence["prior_supervision"].get("contract_sha256") != pins[PATHS["prior_contract"]]:
        raise ValueError("E333 original supervisor contract link differs")
    if (evidence["prior_audit_supervision"].get("contract_sha256") != pins[PATHS["prior_audit_contract"]]
            or evidence["closure_supervision"].get("contract_sha256") != pins[PATHS["closure_contract"]]
            or evidence["prior_contract"].get("source_sha256") != prior["source_sha256"]
            or evidence["prior_review"].get("source_sha256") != prior["source_sha256"]
            or evidence["prior_review"].get("freeze_sha256") != pins[PATHS["prior_freeze"]]
            or result.get("provenance", {}).get("freeze_sha256") != pins[PATHS["prior_freeze"]]
            or result["provenance"].get("review_sha256") != pins[PATHS["prior_review"]]):
        raise ValueError("prior original/audit/root closure producer contract differs")
    secant = evidence["secant"]
    if (secant.get("schema") != "flylab.m3.e334.root-design-secant.v1"
            or secant.get("producer_source_sha256") != pins[PATHS["secant_source"]]
            or secant.get("e333_decision_sha256") != pins[PATHS["prior_decision"]]
            or secant.get("e333_audit_sha256") != pins[PATHS["prior_audit"]]
            or secant.get("theta_star64_bytes_sha256") != THETA_SHA
            or secant.get("old_gtheta_numeric_reads") != 0 or secant.get("original_calls") != 0
            or evidence["secant_supervision"].get("contract_sha256") != pins[PATHS["secant_contract"]]):
        raise ValueError("E334 fixed secant producer linkage differs")
    for name, digest in secant["input_sha256"].items():
        if pins.get(name) != digest:
            raise ValueError("secant consumed input differs")
    closure, invocation, reflection = (evidence[k] for k in ("closure", "invocation", "reflection"))
    if (closure.get("schema") != "flylab.m3.e334.root-inherited-closure.v1" or closure.get("passed") is not True
            or (closure.get("source_count"), closure.get("input_count")) != (180, 1660)
            or closure.get("freeze_sha256") != pins[PATHS["prior_freeze"]]
            or closure.get("producer_source_sha256") != pins[PATHS["closure_source"]]
            or closure.get("E326_helper_sha256") != HELPER_SHA
            or reflection.get("schema") != "flylab.m3.e334.root-design-reflection.v1"
            or reflection.get("reflected_design_sha256") != pins[PATHS["design"]]
            or reflection.get("review_sha256") != pins[PATHS["astra"]]
            or reflection.get("review_invocation_sha256") != pins[PATHS["invocation"]]
            or invocation.get("schema") != "flylab.m3.e334.actual-astra-invocation.v1"):
        raise ValueError("root closure/actual review reflection linkage differs")
    cli = invocation["CLI_current"]
    if (cli.get("model") != "gpt-6-astra" or cli.get("config", {}).get("model_reasoning_effort") != "max"
            or cli.get("completed") is not True or cli.get("exit_code") != 0
            or cli.get("review_sha256") != pins[PATHS["astra"]] or cli.get("events_sha256") != pins[PATHS["events"]]):
        raise ValueError("completed actual Astra max invocation required")
    timely()
    return frozen


def load_selected(path, keys):
    """Access ONLY named keys; old gradient archives must never be expanded."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in keys}


def _array(value, shape, dtype, label):
    if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
        raise ValueError("exact finite typed " + label + " required")
    return value


def startup_metrics(q, tendon, target_q, target_tendon):
    qe, te = q - target_q, tendon - target_tendon
    values = {"q_rms": float(np.sqrt(np.mean(qe ** 2))), "q_max": float(np.max(np.abs(qe))),
              "tendon_rms": float(np.sqrt(np.mean(te ** 2))), "tendon_max": float(np.max(np.abs(te)))}
    if not np.isfinite(list(values.values())).all():
        raise ValueError("nonfinite startup metrics")
    return values


def secant_candidate(q0, t0, q1, t1, target_q, target_tendon, theta_minus, *, expected=None):
    """Strict E334 wrapper around immutable E326 public closed-form helper."""
    from tools.verify_c_banc_m3_constrained_secant import feasible_alpha
    for label, value, shape in (("q0", q0, (99, 42)), ("q1", q1, (99, 42)), ("target_q", target_q, (99, 42)),
                               ("t0", t0, (99, 6)), ("t1", t1, (99, 6)), ("target_tendon", target_tendon, (99, 6))):
        _array(value, shape, np.float64, label)
    if not isinstance(theta_minus, np.ndarray) or theta_minus.ndim != 1 or not len(theta_minus):
        raise ValueError("theta_minus vector required")
    _array(theta_minus, theta_minus.shape, np.float64, "theta_minus")
    if np.any(theta_minus <= 0) or not np.array_equal(theta_minus, theta_minus.astype(np.float32).astype(np.float64)):
        raise ValueError("positive actual f32-promoted prior theta required")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        neutral = startup_metrics(q0, t0, target_q, target_tendon)
        if any(neutral[key] >= cap for key, cap in GUARDS.items()):
            raise ValueError("neutral must be strictly feasible")
        eq, et, dq, dt = q0 - target_q, t0 - target_tendon, q1 - q0, t1 - t0
        if not np.any(dq) and not np.any(dt):
            return {"classification": "inconclusive", "reason": "no_secant_information"}, None, {}
        coefficients = {}
        for label, e, d, cap in (("q", eq, dq, .018), ("tendon", et, dt, .009)):
            a, b, c = float(np.mean(d * d)), float(2 * np.mean(e * d)), float(np.mean(e * e) - cap * cap)
            discriminant = b * b - 4 * a * c
            if not np.isfinite([a, b, c, discriminant]).all() or a < 0 or discriminant < 0 or c >= 0:
                raise ValueError("finite feasible RMS quadratic required")
            if a == 0:
                if np.any(d):
                    raise ValueError("underflowed nonzero RMS delta")
                roots = None
            else:
                big = -.5 * (b + np.copysign(np.sqrt(discriminant), b))
                roots = sorted((big / a, c / big))
                if not np.isfinite(roots).all():
                    raise ValueError("finite RMS roots required")
            coefficients[label] = {"ABC": [a, b, c], "discriminant": discriminant, "roots": roots}
        # Explicit finite checks for each moving scalar root, including roots
        # the older helper does not use. Zero coordinates add no restriction.
        for e, d, cap in ((eq, dq, .04), (et, dt, .025)):
            moving = d != 0
            if not np.isfinite((cap - e[moving]) / d[moving]).all() or not np.isfinite((-cap - e[moving]) / d[moving]).all():
                raise ValueError("finite scalar roots required")
        info = feasible_alpha(q0, t0, q1, t1, target_q, target_tendon)
        alpha, maximum = info["alpha_star"], info["alpha_max"]
        if not np.isfinite([alpha, maximum]).all() or not 0 < alpha <= maximum <= 1:
            raise ValueError("positive finite secant interval required")
        theta = (1 + alpha * (theta_minus - 1)).astype(np.float32).astype(np.float64)
        if not np.isfinite(theta).all() or np.any(theta <= 0):
            raise ValueError("positive finite actual theta required")
        if np.all(theta == 1):
            return {"classification": "rejected", "reason": "rounded_unity_candidate"}, None, {}
        predicted = startup_metrics(q0 + alpha * dq, t0 + alpha * dt, target_q, target_tendon)
        if any(predicted[key] > cap for key, cap in GUARDS.items()):
            raise ValueError("predicted midpoint startup guard failed")
        intervals = {key: [0., value] for key, value in info["limits"].items()}
        if expected is not None and (not np.isclose(alpha, expected["alpha_star"], rtol=0, atol=1e-12)
                    or not np.allclose([0, maximum], expected["intersection"], rtol=0, atol=1e-12)
                    or any(not np.allclose(value, expected["secant_intervals"][key], rtol=0, atol=1e-12) for key, value in intervals.items())
                    or bytes_sha(theta) != expected["theta_star64_bytes_sha256"]):
            raise ValueError("fixed alpha/interval/actual theta SHA mismatch; no reselection")
    report = {"classification": "candidate", "alpha_star": alpha, "intersection": [0., maximum],
              "secant_intervals": intervals, "quadratics": coefficients, "predicted_startup": predicted,
              "theta_star64_bytes_sha256": bytes_sha(theta), "changed_groups": int(np.count_nonzero(theta != 1)),
              "affine_prediction_is_neural_proof": False, "f32_theta_is_exact_scalar_secant": False}
    raw = {"q_error": eq, "q_delta": dq, "tendon_error": et, "tendon_delta": dt,
           "oldtheta_minus": theta_minus, "alpha_star": np.array(alpha, np.float64),
           **{key: np.array(value, np.float64) for key, value in predicted.items()}}
    return report, theta, raw


class Dependencies:
    def load(self, freeze, root=ROOT):
        from tools.verify_c_banc_m3_smooth_efficacy_direction import (
            Dependencies as Previous,
        )
        prior = read(safe(PATHS["prior_freeze"], root))
        # The public old loader consumes its own inherited path vocabulary.
        # Supply a separate configuration, retaining all new source/input pins;
        # no old module global or frozen producer bytes are changed.
        loader_config = {**prior, "source_sha256": freeze["source_sha256"],
                         "input_sha256": freeze["input_sha256"],
                         "smooth_constant_metadata": freeze["smooth_constant_metadata"]}
        data = Previous().load(loader_config, root=root)
        minus = load_selected(safe(PRIOR + "/continuation-minus-forward.npz", root), ("q", "tendon"))
        direction = load_selected(safe(PRIOR + "/direction.npz", root), ("theta_minus",))
        _array(direction["theta_minus"], (GROUPS,), np.float64, "prior theta_minus")
        data.update(secant_q1=minus["q"][:99], secant_t1=minus["tendon"][:99],
                    theta_minus=direction["theta_minus"], expected_secant=freeze["secant"])
        return data

    def build(self, data, device):
        from tools.banc_m3_checkpointed_rollout import CheckpointedControlRollout
        from tools.verify_c_banc_m3_smooth_efficacy_direction import (
            Dependencies as Previous,
        )
        previous, objective, initial = Previous().build(data, device)
        rollout = CheckpointedControlRollout(previous.neural, previous.decoder, data["calibration"]["motor_ids"],
            steps_per_control=20, control_dt=.005, checkpoint_controls=300, max_controls=300)
        return rollout, objective, initial

    @staticmethod
    def host(value):
        return value.detach().cpu().numpy().copy()


def _metadata(array):
    return {"shape": list(array.shape), "dtype": array.dtype.str, "c_order_sha256": bytes_sha(array)}


def actual_input_proof(data, rollout, objective, initial, *, test_only=False, host=Dependencies.host):
    """Compare the actual used device tensor content against approved host inputs."""
    start, native = data["start"], data["native"]
    expected = (start["network"]["rate"], start["network"]["adaptation"], start["adapter"]["bank"],
                start["adapter"]["original"]["activation"], data["drives"], start["network"]["output_mask"])
    names = ("rate", "adaptation", "bank", "original_activation", "drives", "full_mask")
    proof, actual = {}, {}
    for name, value, approved in zip(names, initial, expected, strict=True):
        if not value.is_contiguous():
            raise ValueError("actual input tensor must be contiguous")
        if name == "drives":
            if tuple(value.shape) != approved.shape or str(value.dtype) != "torch." + approved.dtype.name:
                raise ValueError("actual input tensor dtype/shape differs: drives")
            h = hashlib.sha256()
            chunks, observed_rows = [], 0
            for c in range(0, 300, 25):
                observed = host(value[c:c + 25])
                _array(observed, (25, approved.shape[1]), approved.dtype, "actual drive host chunk")
                h.update(np.ascontiguousarray(observed).tobytes())
                chunks.append({"start_control": c + 1, **_metadata(observed)})
                observed_rows += observed.shape[0]
            metadata = {"shape": [observed_rows, observed.shape[1]], "dtype": observed.dtype.str,
                        "c_order_sha256": h.hexdigest()}
            proof["drives_tensor"] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            proof["drives_chunks"] = chunks
        else:
            observed = host(value)
            metadata = _metadata(observed)
            actual[name] = observed
        if metadata != _metadata(approved):
            raise ValueError("actual input tensor content differs: " + name)
        proof[name] = metadata
    ids = host(rollout.motor_ids)
    if _metadata(ids) != _metadata(data["calibration"]["motor_ids"]):
        raise ValueError("actual ordered motor IDs differ")
    proof["motor_ids"] = _metadata(ids)
    actual["motor_ids"] = ids
    for state in (data["whole100"], data["final"]):
        if _metadata(state["network"]["output_mask"]) != proof["full_mask"]:
            raise ValueError("full output mask changes across approved controls")
    if start.get("control_tick") != 0 or start["network"].get("tick") != 0:
        raise ValueError("actual original control0 clock required")
    proof["control_clock"] = {"control0": 0, "network_tick0": 0, "controls": 300, "steps_per_control": 20, "dt": .005}
    actual.update(control_tick=np.array(0, np.int64), network_tick=np.array(0, np.int64))
    tensor_specs = {}
    if test_only:
        # Explicit test-only small physiology is independently supplied by the
        # fixture. This escape is inaccessible through full original CLI.
        tensor_specs = data["test_tensor_specs"]
    else:
        for name in ("tau", "coefficient", "threshold", "cap", "gain"):
            tensor_specs["neural." + name] = (rollout.neural, name, native["physiology"][name])
        tensor_specs["neural.inverse_adaptation_tau"] = (rollout.neural, "inverse_adaptation_tau", native["physiology"]["inverse"])
        calibration, parameters = data["calibration"], data["parameters"]
        decoder = rollout.decoder
        for name in ("moment", "rest", "pool", "ltm"):
            tensor_specs["decoder." + name] = (decoder, name, calibration[name])
        for name in ("tau", "kappa"):
            tensor_specs["decoder." + name] = (decoder, name, parameters[name])
        tensor_specs["decoder.neutral"] = (decoder, "neutral", np.asarray(start["body"]["action"]["joint_angles"]))
        slopes = parameters["slopes"]
        tensor_specs["decoder.low"] = (decoder, "low", slopes[..., 0].reshape(96, 1173))
        tensor_specs["decoder.delta"] = (decoder, "delta", (slopes[..., 1] - slopes[..., 0]).reshape(96, 1173))
        counts = np.bincount(parameters["membership"], minlength=96)
        fixed = np.zeros(96, bool)
        fixed[np.union1d(calibration["ltm"], np.flatnonzero(counts == 1))] = True
        tensor_specs["decoder.fixed_mask"] = (decoder, "fixed_mask", fixed)
    for label, (owner, name, approved) in tensor_specs.items():
        observed = host(getattr(owner, name))
        if _metadata(observed) != _metadata(approved):
            raise ValueError("actual used tensor content differs: " + label)
        proof[label] = _metadata(observed)
    if float(rollout.neural.dt) != float(np.float32(.00025)):
        raise ValueError("actual neural dt differs")
    proof["neural.dt"] = _metadata(np.array(rollout.neural.dt, np.float32))
    if not test_only:
        attributes = {"curves": "curves", "normalization": "successor_normalization", "temperature": "temperature",
                      "future_anchors": "future_anchors", "phase_basis": "phase_basis"}
        for name, approved in data["smooth_constants"].items():
            observed = host(getattr(objective, attributes[name]))
            if _metadata(observed) != _metadata(approved) or objective.constant_metadata[name] != data["smooth_constant_metadata"][name]:
                raise ValueError("actual approved objective constant transfer differs")
        proof["smooth_constant_metadata"] = objective.constant_metadata
        proof["constructor_recomputed"] = {key: _metadata(value) for key, value in objective.constructor_recomputed.items()}
        operator, csr, mask = rollout.neural.operator, native["csr"], native["mask"]
        order = np.argsort(mask["positions"], kind="stable")
        for label, observed, approved in (("operator.data", operator._weights.data, csr["data"]),
            ("operator.indices", operator._weights.indices, csr["indices"]), ("operator.indptr", operator._weights.indptr, csr["indptr"]),
            ("operator.selected_positions", operator.selected_edge_ids, mask["positions"][order]),
            ("operator.selected_groups", operator.selected_group_ids, mask["edge_parameters"][order])):
            if _metadata(observed) != _metadata(approved):
                raise ValueError("actual dynamic operator original content differs: " + label)
            proof[label] = _metadata(observed)
    proof["test_only"] = test_only
    return proof, actual


def receipt_dict(receipt):
    names = ("schema", "status", "forward_controls_requested", "forward_controls_returned", "forward_neural_ticks_requested",
             "forward_neural_ticks_returned", "recompute_controls_requested", "recompute_controls_returned", "recompute_neural_ticks_requested",
             "recompute_neural_ticks_returned", "backward_blocks_requested", "backward_blocks_returned", "backward_block_ticks_returned",
             "confirm_requested", "confirm_returned", "confirmed")
    return {name: getattr(receipt, name) for name in names}


def check_receipt(row):
    expected = {"forward_controls": 300, "forward_neural_ticks": 6000, "recompute_controls": 0,
                "recompute_neural_ticks": 0, "backward_blocks": 0, "backward_block_ticks": 0}
    if row.get("confirmed") != expected or row.get("confirm_requested") != 1 or row.get("confirm_returned") != 1:
        raise ValueError("single confirmed 300/6000 forward-only receipt required")
    for key, value in expected.items():
        for suffix in ("requested", "returned"):
            if key + "_" + suffix == "backward_block_ticks_requested":
                continue
            if row.get(key + "_" + suffix) != value:
                raise ValueError("receipt requested/returned mismatch")


def compute(data, *, dependencies=None, device="cuda", test_dimensions=None, checkpoint=None,
            deadline=None, on_progress=None, on_boundary=None, on_receipt=None):
    import torch

    from tools.verify_c_banc_m3_temporal_direction import patch_weights
    if test_dimensions is None:
        if device != "cuda" or dependencies is not None:
            raise ValueError("original full CUDA path; fixtures require explicit TestDimensions")
        nodes, groups = N, GROUPS
    else:
        if not isinstance(test_dimensions, TestDimensions) or dependencies is None:
            raise ValueError("explicit generated dimensions/dependencies required")
        nodes, groups = test_dimensions.nodes, test_dimensions.groups
    deps = Dependencies() if dependencies is None else dependencies
    save, timely = checkpoint or (lambda name, arrays: None), deadline or (lambda: None)
    progress, boundary, report_receipt = on_progress or (lambda value: None), on_boundary or (lambda event: None), on_receipt or (lambda value: None)
    result = {"classification": "not_tested", "scientific_supported": False, "forward_calls": 0,
              "external_state_reinjection": 0, "body_calls": 0, "events": [],
              "acquisition": dict.fromkeys(("q", "tendon", "rate", "adaptation", "bank", "original_activation"), "unavailable"),
              "secondary_errors": []}
    receipts, receipt, confirm_attempted = {}, None, False
    def event(name, **fields):
        item = {"event": name, **fields}
        result["events"].append(item)
        boundary(item)
    def secondary(stage, exception):
        result["secondary_errors"].append({"stage": stage, "type": type(exception).__name__, "message": str(exception)})
    def confirm_once():
        nonlocal confirm_attempted
        if receipt is None or confirm_attempted:
            return
        confirm_attempted = True
        event("device_confirm_requested")
        receipt.confirm()
        event("device_confirm_returned")
    def publish_receipt():
        if receipt is not None:
            receipts["continuation-star"] = receipt_dict(receipt)
            report_receipt(receipts["continuation-star"])
    try:
        _array(data["drives"], (300, nodes), np.float32, "ordered drives")
        _array(data["theta_minus"], (groups,), np.float64, "prior theta_minus")
        info, theta, secant_raw = secant_candidate(data["raw"]["command"][1:100], data["raw"]["tendon"][1:100],
            data["secant_q1"], data["secant_t1"], data["startup"]["target_command"], data["startup"]["target_tendon"],
            data["theta_minus"], expected=data.get("expected_secant"))
        result["secant"] = info
        if theta is None:
            result.update(classification=info["classification"], reason=info["reason"])
            return result, receipts
        save("secant-inputs", secant_raw)
        save("theta-star", {"theta64": theta, "theta32": theta.astype(np.float32)})
        timely()
        patched, actual, eligibility = patch_weights(data["native"], theta)
        result["native_eligibility"] = eligibility
        positions, group = data["native"]["mask"]["positions"], data["native"]["mask"]["edge_parameters"]
        save("selected-weights-star", {"original32": data["native"]["csr"]["data"][positions],
                                       "actual32": actual, "positions": positions, "groups": group})
        save("patched-csr-star", patched)
        if not eligibility["eligible"]:
            result.update(classification="rejected", reason="native_ineligible")
            return result, receipts
        rollout, objective, initial = deps.build(data, device)
        if (rollout.neural.n != nodes or rollout.neural.operator.group_count != groups or rollout.checkpoint_controls != 300
                or rollout.max_controls != 300 or rollout.steps_per_control != 20 or rollout.control_dt != .005):
            raise ValueError("new checkpoint300/300 fixed E334 rollout required")
        proof, actual_start = actual_input_proof(data, rollout, objective, initial, test_only=test_dimensions is not None,
                                               host=getattr(deps, "host", Dependencies.host))
        result["actual_input_proof"] = proof
        save("continuation-start", actual_start)
        timely()
        theta_device = torch.tensor(theta, dtype=torch.float64, device=rollout.device)
        if _metadata(Dependencies.host(theta_device)) != _metadata(theta):
            raise ValueError("actual efficacy tensor differs")
        result["actual_theta"] = _metadata(theta)
        r, z, bank, original, drives, mask = initial
        event("neural_requested")
        result["forward_calls"] += 1
        with torch.no_grad():
            output = rollout.run(r, z, bank, original, drives, theta_device, mask)
        receipt = output["receipt"]
        acquired, first_acquisition_error = {}, None
        try:
            event("python_call_returned")
        except Exception as exception:  # noqa: BLE001
            first_acquisition_error = exception
        for key in ("q", "tendon", "rate", "adaptation", "bank", "original_activation"):
            try:
                acquired[key] = getattr(deps, "host", Dependencies.host)(output[key])
            except Exception as exception:  # noqa: BLE001
                result["acquisition"][key] = "unavailable"
                if first_acquisition_error is None:
                    first_acquisition_error = exception
                else:
                    secondary("acquisition_" + key, exception)
            else:
                result["acquisition"][key] = "acquired"
                try:
                    event("array_acquired", key=key)
                except Exception as exception:  # noqa: BLE001
                    if first_acquisition_error is None:
                        first_acquisition_error = exception
                    else:
                        secondary("acquisition_journal_" + key, exception)
        # Actual acquired raw precedes shape, finite, objective and deadline
        # validation. No synthesized missing arrays are written on failures.
        try:
            try:
                event("raw_save_requested")
            except Exception as exception:  # noqa: BLE001
                if first_acquisition_error is None:
                    first_acquisition_error = exception
                else:
                    secondary("raw_save_journal", exception)
            save("continuation-forward", acquired)
            event("raw_save_returned")
        except Exception as exception:
            if first_acquisition_error is not None:
                secondary("raw_save", exception)
                raise first_acquisition_error from exception
            raise
        if first_acquisition_error is not None:
            raise first_acquisition_error
        confirm_once()
        publish_receipt()
        timely()
        for key, shape, dtype in (("q", (300, 42), np.float64), ("tendon", (300, 6), np.float64),
                                 ("rate", (nodes,), np.float32), ("adaptation", (nodes,), np.float32),
                                 ("bank", (391, 3), np.float64), ("original_activation", (96,), np.float64)):
            _array(acquired[key], shape, dtype, key)
        terms = objective.evaluate(output["q"], output["tendon"])
        scalar = {}
        for key in ("training_loss", "successor_loss", "coordination_loss", "lr_coordination_loss",
                    "startup_q_rms", "startup_q_max", "startup_tendon_rms", "startup_tendon_max", "startup_pass"):
            value = Dependencies.host(terms[key])
            if value.shape != () or not np.isfinite(value).all():
                raise ValueError("finite scalar metric required")
            scalar[key] = value.item()
        numpy_guards = startup_metrics(acquired["q"][:99], acquired["tendon"][:99],
                                      data["startup"]["target_command"], data["startup"]["target_tendon"])
        for key, value in numpy_guards.items():
            torch_key = "startup_" + key
            if not np.isclose(scalar[torch_key], value, rtol=1e-10, atol=1e-10):
                raise ValueError("Torch/NumPy startup metric mismatch")
            if (scalar[torch_key] <= GUARDS[key]) != (value <= GUARDS[key]):
                raise ValueError("startup guard boundary disagreement; qualification withheld")
        startup_pass = all(numpy_guards[key] <= bound for key, bound in GUARDS.items())
        if bool(scalar["startup_pass"]) != startup_pass:
            raise ValueError("Torch/NumPy startup guard disagreement")
        improvement = float(data["e332_neutral_terms"]["training_loss"] - scalar["training_loss"])
        scalar["improvement"] = improvement
        result.update(metrics=scalar, actual_startup=numpy_guards,
                      predicted_actual_gap={key: numpy_guards[key] - info["predicted_startup"][key] for key in GUARDS})
        save("continuation-metrics", {key: np.asarray(value) for key, value in scalar.items()})
        check_receipt(receipts["continuation-star"])
        timely()
        supported = startup_pass and improvement >= 1e-9
        result.update(classification="conditional_neural_supported" if supported else "rejected", scientific_supported=supported,
                      handoff_correspondence="original CSR plus theta maps to once-rounded patched CSR; never apply theta to patched CSR",
                      candidate_native_identity=hashlib.sha256((eligibility["patched_csr_data32_sha256"] + bytes_sha(theta)).encode()).hexdigest())
        progress(result.copy())
        return result, receipts
    except Exception as exception:
        receipt = receipt if receipt is not None else getattr(exception, "receipt", None)
        try:
            confirm_once()
        except Exception as later:  # noqa: BLE001
            secondary("partial_confirm", later)
        try:
            publish_receipt()
        except Exception as later:  # noqa: BLE001
            secondary("receipt_journal", later)
        result["classification"] = "invalid"
        try:
            event("failed")
        except Exception as later:  # noqa: BLE001
            secondary("failure_journal", later)
        exception.partial = result
        exception.receipts = receipts
        raise


def write_json(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def repair_json(path, value):
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf8", dir=path.parent, prefix=path.name + ".repair-", delete=False) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def usage():
    values = {"host_rss_bytes": psutil.Process().memory_info().rss, "sampling_scope": "current child; external supervisor measures owned tree"}
    import sys
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        if torch.cuda.is_initialized():
            values["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            values["device_free_bytes"], values["device_total_bytes"] = torch.cuda.mem_get_info()
    return values


def run(freeze_path, freeze_sha256, review_path, review_sha256, output, *, root=ROOT, dependencies=None,
        test_dimensions=None, device="cuda", clock=time.perf_counter, writer=write_json, saver=np.savez,
        meter=usage, compute_fn=compute, gate_fn=gate, emergency=write_json, repair=repair_json, reader=read):
    """One fixed pre-gate clock; raw/first error survive every secondary sink."""
    s0 = clock()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("fresh output required")
    if test_dimensions is None and (dependencies is not None or device != "cuda" or output != safe(OUTPUT, root)):
        raise ValueError("original CLI fullN/G/300/CUDA only")
    ledger = {"schema": "flylab.m3.e334.advance-ledger.v1", "limits": LIMITS.copy(), "attempts": [], "events": [], "errors": []}
    result = {"schema": RESULT_SCHEMA, "run_valid": False, "classification": "invalid", "partial": {},
              "effective_neural_qualification": False, "handoff_eligible": False, "candidate_adopted": False,
              "F1_F8_complete": False, "biological_gait_proven": False, "F6_exposed": False, "M2_modified": False,
              "body_calls": 0, "body_not_executed": True, "E333_remains_invalid": True,
              "E330_local_direction": "refuted", "E327_body_verdict": "rejected", "whole_goal_status": "active",
              "provenance": {"freeze_sha256": freeze_sha256, "review_sha256": review_sha256},
              "publication": {"external_supervisor_and_root_byte_audit_required": True, "recovery_sink_failure_guarantees_disk_agreement": False}}
    errors, receipts, files, frozen = ledger["errors"], {}, {}, None
    science_end, final_end = s0 + 45, None
    baseline = meter()
    ledger["resource_baseline"] = baseline
    def record(stage, exception):
        errors.append({"stage": stage, "type": type(exception).__name__, "message": str(exception), "elapsed_s": clock() - s0})
    def resource_check():
        latest = meter()
        ledger["resource_latest"] = latest
        if latest.get("host_rss_bytes", 0) > LIMITS["host_bytes"]:
            raise MemoryError("host resource limit")
        if baseline.get("device_free_bytes") is not None and latest.get("device_free_bytes") is not None:
            delta = max(0, baseline["device_free_bytes"] - latest["device_free_bytes"])
            ledger["sampled_additional_gpu_bytes"] = max(ledger.get("sampled_additional_gpu_bytes", 0), delta)
            if delta > LIMITS["additional_gpu_bytes"]:
                raise MemoryError("additional GPU sampled limit")
        if latest.get("torch_peak_reserved_bytes", 0) > LIMITS["additional_gpu_bytes"]:
            raise MemoryError("Torch reserved limit")
    def timely():
        if clock() >= science_end:
            raise TimeoutError("E334 preparation/science absolute deadline")
        resource_check()
    def save(name, arrays):
        path = output / (name + ".npz")
        if path.name not in RAW_FILES or path.exists():
            raise ValueError("fixed unique raw7 roster required")
        attempt = {"file": path.name, "requested": True, "returned": False, "requested_elapsed_s": clock() - s0}
        ledger["attempts"].append(attempt)
        saver(path, **arrays)
        attempt.update(returned=True, returned_elapsed_s=clock() - s0)
        files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        # Acquisition/raw has already been recorded and written before timing
        # can fail here. No missing device values are synthesized.
        timely()
    try:
        frozen = gate_fn(freeze_path, freeze_sha256, review_path, review_sha256, output, root=root, timely=timely)
        output.mkdir(parents=True)
        if shutil.disk_usage(output).free < LIMITS["minimum_free_disk_bytes"]:
            raise OSError("free disk below4GiB")
        timely()
        if test_dimensions is None:
            import torch
            torch.set_num_threads(1)
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats()
            baseline.update(meter())
        deps = Dependencies() if dependencies is None else dependencies
        data = deps.load(frozen, root=root)
        timely()
        result["partial"], receipts = compute_fn(data, dependencies=dependencies, device=device, test_dimensions=test_dimensions,
            checkpoint=save, deadline=timely, on_progress=lambda value: result.update(partial=value),
            on_boundary=lambda event: ledger["events"].append({**event, "elapsed_s": clock() - s0}),
            on_receipt=lambda value: receipts.update({"continuation-star": value}))
        if result["partial"].get("forward_calls") == 1 and set(files) != set(RAW_FILES):
            raise ValueError("normal forward requires all7 raw files")
        pin_map(frozen["source_sha256"], root, timely)
        pin_map(frozen["input_sha256"], root, timely)
        if sha256(freeze_path) != freeze_sha256 or sha256(review_path) != review_sha256:
            raise ValueError("post-science freeze/review changed")
        timely()
        result.update(run_valid=True, classification=result["partial"]["classification"],
                      effective_neural_qualification=result["partial"].get("scientific_supported", False))
        result["handoff_eligible"] = result["effective_neural_qualification"]
    except Exception as exception:  # noqa: BLE001
        record("science", exception)
        result["partial"] = getattr(exception, "partial", result["partial"])
        receipts.update(getattr(exception, "receipts", {}))
        for secondary in result["partial"].get("secondary_errors", []):
            errors.append({**secondary, "elapsed_s": clock() - s0})
    final_begin = clock()
    final_end = min(s0 + 55, final_begin + 10)
    ledger.update(science_deadline_elapsed_s=45, final_deadline_elapsed_s=final_end - s0,
                  science_wall_s=final_begin - s0, original_calls=0 if test_dimensions is not None else int(frozen is not None),
                  body_calls=0, optimizer_steps=0, retry_count=0)
    def sync():
        first = errors[0] if errors else None
        common = {"run_valid": result["run_valid"], "classification": result["classification"], "first_error": first,
                  "errors": errors, "receipts": receipts, "elapsed_s": clock() - s0,
                  "effective_neural_qualification": result["effective_neural_qualification"], "handoff_eligible": result["handoff_eligible"]}
        result.update(common)
        ledger.update(common)
    def final_check():
        if clock() >= final_end:
            raise TimeoutError("E334 absolute final deadline")
        resource_check()
        if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > LIMITS["output_bytes"]:
            raise OSError("output limit exceeded")
    output.mkdir(parents=True, exist_ok=True)
    sync()
    try:
        final_check()
        writer(output / "source-manifest.json", {"schema": "flylab.m3.e334.source-manifest.v1", **result["provenance"],
               "source_sha256": {} if frozen is None else frozen["source_sha256"], "input_sha256": {} if frozen is None else frozen["input_sha256"]})
        writer(output / "receipts.json", {"schema": "flylab.m3.e334.receipts.v1", "receipts": receipts})
        for name in ("source-manifest.json", "receipts.json"):
            path = output / name
            files[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        writer(output / "output-manifest.json", {"schema": "flylab.m3.e334.output-manifest.v1", "files": files,
               "mutable_metadata_excluded": ["result.json", "advance-ledger.json", "publication-failure.json"]})
        writer(output / "advance-ledger.json", ledger)
        writer(output / "result.json", result)
        disk_result, disk_ledger, disk_manifest = (reader(output / name) for name in ("result.json", "advance-ledger.json", "output-manifest.json"))
        if disk_result != result or disk_ledger != ledger or disk_manifest["files"] != files:
            raise ValueError("terminal read-back/content agreement failure")
        for name, descriptor in files.items():
            if descriptor != {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size}:
                raise ValueError("final manifest bytes mismatch")
        final_check()
    except Exception as exception:  # noqa: BLE001
        record("publication", exception)
        result.update(run_valid=False, classification="invalid", effective_neural_qualification=False, handoff_eligible=False)
        sync()
        try:
            emergency(output / "publication-failure.json", {"schema": "flylab.m3.e334.publication-failure.v1", "run_valid": False,
                      "first_error": errors[0], "errors": errors, "elapsed_s": clock() - s0})
        except Exception as secondary:  # noqa: BLE001
            record("emergency_publication", secondary)
        sync()
        for name, value in (("advance-ledger.json", ledger), ("result.json", result)):
            try:
                repair(output / name, value)
            except Exception as secondary:  # noqa: BLE001
                record("terminal_repair_" + name, secondary)
                sync()
        try:
            disk_result, disk_ledger = reader(output / "result.json"), reader(output / "advance-ledger.json")
            if disk_result != result or disk_ledger != ledger:
                raise ValueError("repaired terminal agreement not established")
            final_check()
        except Exception as secondary:  # noqa: BLE001
            record("repair_validation", secondary)
            sync()
            # Last bounded best effort carries newly found failure/cost; no
            # deadline reset and no scientific qualification can be restored.
            for name, value in (("advance-ledger.json", ledger), ("result.json", result)):
                try:
                    repair(output / name, value)
                except Exception as later:  # noqa: BLE001
                    record("final_repair_" + name, later)
            sync()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    result = run(args.freeze, args.freeze_sha256, args.review, args.review_sha256, args.output)
    print(json.dumps({"schema": RESULT_SCHEMA, "run_valid": result["run_valid"], "classification": result["classification"], "first_error": result["first_error"]}))
    return 0 if result["run_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
