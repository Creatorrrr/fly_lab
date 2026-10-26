"""E339 fixed saved-command coordination proxy; no neural/body runtime.

Only ``main --run`` can read the four original archives, after ROOT's freeze
and review gate. Generated readers use the identical binding/scanner/decode
path in an isolated namespace. Scientific adoption belongs to ROOT.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "verification/banc-natural-gait-m3-20260919"
OUTPUT = BASE + "/e339-saved-coordination"
FREEZE = BASE + "/e339-saved-coordination-freeze.json"
REVIEW = BASE + "/parent-e339-original-source-review.json"
DEPS = BASE + "/parent-e339-direct-dependencies.json"
DESIGN = "docs/BANC_M3_E339_SAVED_COORDINATION_DESIGN_20261001.md"
ASTRA = BASE + "/parent-e339-astra-design-review.md"
REFLECTION = BASE + "/parent-e339-astra-reflection.json"
REGISTRATION = BASE + "/parent-e339-design-registration.json"
SNAPSHOT = BASE + "/parent-e339-pre-astra-design.md"
DISPATCH = BASE + "/parent-e339-sol-dispatch.json"
R1_DISPATCH = BASE + "/parent-e339-sol-r1-dispatch.json"
FIRST_REVIEW = BASE + "/parent-e339-root-first-source-review.json"
OWNED = ("tools/analyze_c_banc_m3_saved_coordination.py",
         "tests/test_c_banc_saved_coordination.py")
SCANNER = "tools/qualify_c_banc_m3_body_packet.py"
SCHEMA = "flylab.m3.e339.saved-coordination.freeze.v1"
PROOF_SCHEMA = "flylab.m3.e339.current-generated-proof.v1"
SUPERVISOR = BASE + "/parent-e267-supervise.py"
SUPERVISOR_SHA = "1d8682505f8060d3d8bcb50508811b9c4da531bdbf07c96200065bcc3324df8d"
LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
PAIRS = tuple((a, b) for a in range(6) for b in range(a + 1, 6))
LR_INDICES = (2, 7, 11)
FREQUENCY = 2.382760007990865
DT = .005
LIMITS = {"science_wall_s": 8, "internal_wall_s": 12, "finalization_s": 4,
          "supervisor_wall_s": 15, "host_bytes": 2 << 30,
          "output_bytes": 4 << 20, "cpu_threads": 1, "gpu_jobs": 0,
          "jobs": 1, "retry_count": 0}
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
REF_SPEC = {"curves": (6, 128, 7), "normalization": (6,), "temperature": (6,),
            "future_anchors": (6, 128, 7), "phase_basis": (128, 2)}
TRAJ_SPEC = {"q": (201, 42), "z": (201, 6, 2), "R": (201, 6),
             "coordination_pair_loss": (15,)}
_IDENTITIES = {
    "reference": (BASE + "/e332-smooth-temporal-saved/reference.npz",
                  "140e23c865d7e110cd93699b41612c1ffb02eb88dc100ac435c4dcba5acf6c01", 124353),
    "neutral": (BASE + "/e336-saved-gait-transfer/neutral-smooth.npz",
                "8b75c4b1d2cd7d4efc0fe2db0ad49ad50b7dbf2d9ca25b6f38359c9488603d4c", 1644664),
    "held": (BASE + "/e336-saved-gait-transfer/held-smooth.npz",
             "0ea9afacfb31fcb7595b1c37d235dfbfb39979f366c1a0866f0ec65b3cab1ee6", 1644664),
    "live": (BASE + "/e336-saved-gait-transfer/live-smooth.npz",
             "4d2ef1a8a22243b905adea9420ef7230e93e51d19d01e73615c37e84b9953ef4", 1644664),
}
FIXED = {
    DESIGN: "4982534e869c0d59e9d0f2c53b41743c90b4cfba1251022e6f6c3cde98897a10",
    ASTRA: "92ad44adf9b934b32afebe84197af6153c6aadc118413f5e1363abf567f8a7ce",
    DEPS: "10744674d6cf8ccdfe938949b4a610cabdad7f8643adaf312d30c524cf795fa7",
}


def runtime():
    """Lazy imports: production s0 and thread/GPU environment precede these."""
    global np, scan_archive
    if "np" not in globals():
        import numpy

        from tools.qualify_c_banc_m3_body_packet import scan_archive as scan
        np, scan_archive = numpy, scan


def safe(name, root):
    if (not isinstance(name, str) or not name or "\\" in name
            or Path(name).is_absolute() or any(x in ("", ".", "..") for x in name.split("/"))):
        raise ValueError("canonical relative path required")
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("path escapes root")
    return path


def digest(path, timely=lambda: None):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            timely()
            block = stream.read(1 << 20)
            if not block:
                break
            value.update(block)
    timely()
    return value.hexdigest()


def pin(path, timely=lambda: None):
    before = Path(path).stat().st_size
    sha = digest(path, timely)
    if Path(path).stat().st_size != before:
        raise ValueError("bytes changed while hashing")
    return {"sha256": sha, "bytes": before}


def read_json(path):
    return json.loads(Path(path).read_text("utf8"))


def schema_for(role):
    if role not in _IDENTITIES:
        raise ValueError("unknown role")
    return {k: {"shape": list(v), "dtype": "<f8", "finite": True}
            for k, v in (REF_SPEC if role == "reference" else TRAJ_SPEC).items()}


def original_bindings():
    return {role: {"path": name, "sha256": sha, "bytes": size,
                   "selected_schema": schema_for(role),
                   "header_observation_scope": "observation", "descriptor": None,
                   "original_archive_current_header_hash_or_values_observed_at_reflection": False}
            for role, (name, sha, size) in _IDENTITIES.items()}


def error_record(error, stage):
    return {"stage": stage, "type": type(error).__name__, "message": str(error)[:2048]}


class ReaderFailure(ValueError):
    def __init__(self, detail):
        super().__init__(detail["first_error"]["message"])
        self.detail = detail


def fixed_reader(root, role, binding, *, namespace="original", timely=lambda: None):
    """Actual selected-only decode, mandatory hash after np.load closes.

    Generated namespaces retain fixed role schemas, reject the real root and
    original hashes, and cannot be selected by the production CLI.
    """
    runtime()
    root = Path(root).resolve()
    detail = {"role": role, "namespace": namespace, "first_error": None,
              "secondary_errors": [], "selected_reads": [], "postdecode_verified": False}
    path = None
    before = None
    values = {}
    stage = "binding"
    try:
        if role not in _IDENTITIES:
            raise ValueError("unknown role")
        if namespace == "original":
            if root != ROOT or binding != original_bindings()[role]:
                raise ValueError("literal original binding/root differs")
        elif namespace == "generated":
            if root == ROOT or root.is_relative_to(ROOT / BASE) is False:
                raise ValueError("generated root must be isolated under verification base")
            if binding.get("sha256") in {v[1] for v in _IDENTITIES.values()}:
                raise ValueError("original identity forbidden in generated namespace")
            if binding.get("selected_schema") != schema_for(role):
                raise ValueError("generated selected role schema differs")
            if (binding.get("descriptor") is not None
                    or binding.get("header_observation_scope") != "observation"):
                raise ValueError("observation-only binding required")
        else:
            raise ValueError("unknown namespace")
        if (not isinstance(binding.get("sha256"), str)
                or re.fullmatch("[0-9a-f]{64}", binding["sha256"]) is None
                or type(binding.get("bytes")) is not int or binding["bytes"] <= 0):
            raise ValueError("typed binding pin required")
        path = safe(binding["path"], root)
        before = pin(path, timely)
        detail["predecode_pin"] = before
        if before != {k: binding[k] for k in ("sha256", "bytes")}:
            raise ValueError("fixed archive SHA/bytes differs")
        stage = "scan"
        observed = scan_archive(path, expected_sha256=binding["sha256"],
                                descriptor=None, descriptor_scope="observation", timely=timely)
        detail["header_observation"] = observed
        stage = "selected_schema"
        for key, spec in schema_for(role).items():
            member = observed["descriptor"]["members"].get(key)
            if member is None or any(member[k] != spec[k] for k in ("shape", "dtype")):
                raise ValueError("selected schema differs: " + key)
        stage = "selected_decode"
        with np.load(path, allow_pickle=False) as archive:
            for key, spec in schema_for(role).items():
                timely()
                # No iteration over archive values; extra headers are already scanned.
                array = archive[key]
                entry = {"path": binding["path"], "key": key, "shape": list(array.shape),
                         "dtype": array.dtype.str, "c_order_sha256": hashlib.sha256(
                             array.tobytes(order="C")).hexdigest()}
                detail["selected_reads"].append(entry)
                typed(array, tuple(spec["shape"]), key)
                values[key] = array.copy(order="C")
            timely()
    except Exception as error:  # noqa: BLE001 -- Preserve reader first error and posthash evidence.
        detail["first_error"] = error_record(error, stage)
        if hasattr(error, "detail"):
            detail["scanner_failure"] = error.detail
    finally:
        if path is not None and before is not None:
            try:
                after = pin(path, timely)
                detail["postdecode_pin"] = after
                if after != before:
                    raise ValueError("postdecode archive SHA/bytes differs")
                detail["postdecode_verified"] = True
            except Exception as error:  # noqa: BLE001 -- Posthash failures remain secondary to decode errors.
                record = error_record(error, "postdecode_hash")
                if detail["first_error"] is None:
                    detail["first_error"] = record
                else:
                    detail["secondary_errors"].append(record)
        else:
            detail["postdecode_unavailable_reason"] = "binding/prehash did not complete"
    if detail["first_error"] is not None:
        raise ReaderFailure(detail)
    return values, detail


def typed(array, shape, name):
    runtime()
    if (not isinstance(array, np.ndarray) or array.shape != shape or array.dtype.str != "<f8"
            or not np.isfinite(array).all()):
        raise ValueError("finite <f8 selected array required: " + name)
    return array


def close(actual, expected, name, *, rtol=1e-10, atol=1e-10):
    if (not np.isfinite(actual).all() or not np.isfinite(expected).all()
            or not np.allclose(actual, expected, rtol=rtol, atol=atol)):
        raise ValueError("numeric reconciliation differs: " + name)


def constants(curves):
    runtime()
    typed(curves, (6, 128, 7), "curves")
    coordinate = ((np.arange(128, dtype=np.float64) / 128 + FREQUENCY * DT) % 1) * 128
    lower = np.floor(coordinate)
    index = lower.astype(np.int64)
    future = curves[:, index] + (coordinate - lower)[None, :, None] * (
        curves[:, (index + 1) % 128] - curves[:, index])
    normalization = ((future - curves) ** 2).mean(axis=(1, 2))
    angle = 2 * np.pi * np.arange(128, dtype=np.float64) / 128
    return {"curves": curves.copy(), "normalization": normalization,
            "temperature": 7 * normalization, "future_anchors": future,
            "phase_basis": np.column_stack((np.cos(angle), np.sin(angle)))}


class Encoder:
    def __init__(self, reference):
        canonical = constants(reference["curves"])
        for key, shape in REF_SPEC.items():
            typed(reference[key], shape, key)
            close(reference[key], canonical[key], key, rtol=1e-12, atol=1e-15)
        if (reference["normalization"] <= 0).any() or (reference["temperature"] <= 0).any():
            raise ValueError("positive normalization/temperature required")
        self.reference = {k: reference[k].copy() for k in REF_SPEC}

    def encode(self, q):
        typed(q, (len(q), 42), "q")
        if not len(q):
            raise ValueError("nonempty q required")
        c = self.reference
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            delta = q.reshape(-1, 6, 1, 7) - c["curves"][None]
            logits = -np.sum(delta * delta, axis=-1) / c["temperature"][None, :, None]
            weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
            weights /= weights.sum(axis=-1, keepdims=True)
            z = np.einsum("tls,sk->tlk", weights, c["phase_basis"])
            resultant = np.linalg.norm(z, axis=-1)
        typed(z, (len(q), 6, 2), "encoded_z")
        typed(resultant, (len(q), 6), "encoded_R")
        return z, resultant


def wrapped(value):
    # Preserve an already in-range value exactly at the strict decision edge.
    return value - np.floor(value + .5)


def moments(z, resultant):
    """Cycles and masks, no epsilon/fill/unwrap/retiming/offset adjustment."""
    runtime()
    typed(z, (len(z), 6, 2), "z")
    typed(resultant, (len(z), 6), "R")
    close(resultant, np.linalg.norm(z, axis=-1), "R magnitude")
    if (resultant < 0).any() or (resultant > 1 + 1e-10).any():
        raise ValueError("moment magnitude out of range")
    valid = resultant >= .5
    unit = np.full_like(z, np.nan)
    unit[valid] = z[valid] / resultant[valid, None]
    pair_mask = np.column_stack([valid[:, a] & valid[:, b] for a, b in PAIRS])
    angles = np.full((len(z), 15), np.nan, dtype=np.float64)
    losses, radial, angular = [], [], []
    summaries = []
    for i, (a, b) in enumerate(PAIRS):
        mask = pair_mask[:, i]
        ua, ub = unit[mask, a], unit[mask, b]
        x = (ua * ub).sum(axis=-1)
        y = ua[:, 1] * ub[:, 0] - ua[:, 0] * ub[:, 1]
        angles[mask, i] = np.arctan2(y, x) / (2 * np.pi)
        samples = angles[mask, i]
        count = int(mask.sum())
        if count:
            mean = np.mean(np.exp(2j * np.pi * samples))
            magnitude = float(abs(mean))
            bias = float(np.angle(mean) / (2 * np.pi)) if magnitude > 1e-12 else None
        else:
            magnitude, bias = None, None
        deviation = (float(np.max(abs(wrapped(samples - bias))))
                     if count and bias is not None else None)
        dot = (z[:, a] * z[:, b]).sum(axis=-1)
        product = resultant[:, a] * resultant[:, b]
        loss = 1 - dot
        first, second = 1 - product, product - dot
        close(first + second, loss, "loss decomposition")
        if (first < -1e-10).any() or (second < -1e-10).any():
            raise ValueError("loss component negative beyond tolerance")
        losses.append(loss)
        radial.append(first)
        angular.append(second)
        summaries.append({"pair": [a, b], "legs": [LEGS[a], LEGS[b]],
                          "valid_count": count, "total_count": len(z),
                          "coverage": count / len(z) if len(z) else None,
                          "bias_cycles": bias, "resultant": magnitude,
                          "deviation_max_cycles": deviation,
                          "original_unmasked_loss_mean": float(loss.mean()) if len(z) else None,
                          "radial_loss_mean": float(first.mean()) if len(z) else None,
                          "angular_weighted_loss_mean": float(second.mean()) if len(z) else None,
                          "radial_loss_min": float(first.min()) if len(z) else None,
                          "angular_weighted_loss_min": float(second.min()) if len(z) else None})
    all6 = valid.all(axis=1)
    coherence = np.full(len(z), np.nan, dtype=np.float64)
    coherence[all6] = np.linalg.norm(unit[all6].mean(axis=1), axis=1)
    arrays = {"z": z, "R": resultant, "leg_valid": valid, "pair_valid": pair_mask,
              "pair_angle_cycles": angles, "all6_valid": all6, "coherence": coherence,
              "pair_loss": np.column_stack(losses), "radial_loss": np.column_stack(radial),
              "angular_weighted_loss": np.column_stack(angular)}
    summary = {"pairs": summaries, "all6_valid_count": int(all6.sum()), "total_count": len(z),
               "coherence_valid_mean": float(coherence[all6].mean()) if all6.any() else None,
               "all6_coverage": float(all6.mean()) if len(z) else None,
               "existing_coordination_coefficient": 1,
               "existing_coordination_loss": float(arrays["pair_loss"].mean()) if len(z) else None,
               "angular_population": "pair-valid normalized moments",
               "loss_population": "all unmasked selected rows including low R"}
    return summary, arrays


def decide(reference, live, live_arrays):
    """Exact integer coverage/exceed comparisons and strict angular boundary."""
    if reference["total_count"] != 128 or live["total_count"] != 200:
        raise ValueError("reference128/live200 decision population required")
    calibration = (reference["all6_valid_count"] >= 122 and all(
        row["valid_count"] >= 122 and row["bias_cycles"] is not None
        and row["deviation_max_cycles"] is not None and row["deviation_max_cycles"] <= .05
        for row in reference["pairs"]))
    coverage = live["all6_valid_count"] >= 180 and all(
        row["valid_count"] >= 180 for row in live["pairs"])
    table = []
    deviations = np.full((200, 15), np.nan)
    exceeds = np.zeros((200, 15), dtype=np.bool_)
    for i, (ref, row) in enumerate(zip(reference["pairs"], live["pairs"], strict=True)):
        if ref["pair"] != list(PAIRS[i]) or row["pair"] != list(PAIRS[i]):
            raise ValueError("fixed pair order differs")
        mask = live_arrays["pair_valid"][:, i]
        if int(mask.sum()) != row["valid_count"]:
            raise ValueError("live denominator differs")
        threshold = None
        count = None
        if ref["bias_cycles"] is not None and ref["deviation_max_cycles"] is not None:
            threshold = ref["deviation_max_cycles"] + .15
            deviations[mask, i] = abs(wrapped(
                live_arrays["pair_angle_cycles"][mask, i] - ref["bias_cycles"]))
            exceeds[mask, i] = deviations[mask, i] > threshold
            count = int(exceeds[:, i].sum())
        table.append({**row, "reference_bias_cycles": ref["bias_cycles"],
                      "reference_valid_count": ref["valid_count"],
                      "reference_deviation_max_cycles": ref["deviation_max_cycles"],
                      "threshold_cycles": threshold, "exceed_count": count,
                      "exceed_fraction": count / row["valid_count"]
                      if count is not None and row["valid_count"] else None,
                      "prominent": count is not None and row["valid_count"] > 0
                      and 4 * count >= row["valid_count"]})
    verdict = "proxy_limited"
    if calibration and coverage:
        verdict = ("angular_misalignment_prominent" if any(row["prominent"] for row in table)
                   else "no_prominent_angular_misalignment_at_fixed_rule")
    return {"suggested_diagnostic": verdict, "reference_calibration_pass": calibration,
            "live_coverage_pass": coverage, "pair_table": table,
            "lr_pair_table": [table[i] for i in LR_INDICES],
            "scope": "fixed-grid encoded proxy; not biological/F4/causal proof",
            "reference_coverage_min_count": 122, "live_coverage_min_count": 180,
            "exceed_test": "abs wrapped deviation > reference max + .15;4*exceed>=valid"}, {
                "reference_deviation_cycles": deviations, "exceed": exceeds}


def analyze(inputs, timely=lambda: None):
    timely()
    encoder = Encoder(inputs["reference"])
    # Curves already contain anatomical offsets; global phase index is shared.
    q_ref = inputs["reference"]["curves"].transpose(1, 0, 2).reshape(128, 42)
    rz, rr = encoder.encode(q_ref)
    ref_summary, ref_arrays = moments(rz, rr)
    summaries, arrays = {"reference": ref_summary}, {"reference": ref_arrays}
    for role in ("neutral", "held", "live"):
        timely()
        values = inputs[role]
        z, resultant = encoder.encode(values["q"])
        close(z, values["z"], role + " saved z all201")
        close(resultant, values["R"], role + " saved R all201")
        summary, numeric = moments(z[1:], resultant[1:])
        close(numeric["pair_loss"].mean(axis=0), values["coordination_pair_loss"],
              role + " pair loss last200")
        numeric.update(encoded_z_all201=z, encoded_R_all201=resultant)
        summaries[role], arrays[role] = summary, numeric
    diagnosis, live_extra = decide(summaries["reference"], summaries["live"], arrays["live"])
    arrays["live"].update(live_extra)
    timely()
    return {"diagnosis": diagnosis, "trajectories": summaries,
            "boundary_comparison_rows": 201, "statistics_rows": 200,
            "statistics_controls": "c101..300", "pair_order": [list(p) for p in PAIRS],
            "lr_indices": list(LR_INDICES), "reference_rows": 128}, arrays


def verify_producers(root, dependencies, timely=lambda: None):
    """Finite closed JSON states/explicit links only; never inherited leaves."""
    records = {}
    pins = dependencies["producer_text_sha256"]
    for name, sha in pins.items():
        path = safe(name, root)
        if digest(path, timely) != sha:
            raise ValueError("producer text pin differs: " + name)
        records[name] = read_json(path)
    def rec(name):
        return records[BASE + "/" + name]
    for number, count in ((336, 7141), (332, 2946)):
        final = rec(f"parent-e{number}-final-decision.json")
        audit = rec(f"parent-e{number}-saved-raw-audit.json")
        if final.get("analysis_valid") is not True or audit.get("passed") is not True:
            raise ValueError("producer completion not valid")
        if audit.get("first_error") is not None or len(audit["checks"]) != count or any(
                value is not True for value in audit["checks"].values()):
            raise ValueError("producer raw checks differ")
        required = ({"root_independent_raw_qualification": True, "raw_checks": count}
                    if number == 336 else {"input_qualification_pass": True,
                                          "root_raw_audit_checks": count})
        if any(final.get(key) != value for key, value in required.items()):
            raise ValueError("producer required states differ")
        linked = [f"parent-e{number}-saved-raw-audit.json",
                  f"e{number}-{'saved-gait-transfer' if number == 336 else 'smooth-temporal-saved'}/output-manifest.json"]
        linked += (["parent-e336-original-supervision.json"] if number == 336 else [
            "parent-e332-saved-supervision.json", "parent-e332-audit-supervision.json",
            "e332-smooth-temporal-saved/result.json"])
        for name in linked:
            full = BASE + "/" + name
            if final["evidence_sha256"].get(full) != pins[full]:
                raise ValueError("producer final evidence edge differs: " + name)
        supervisors = (["parent-e336-original-supervision.json"] if number == 336 else [
            "parent-e332-saved-supervision.json", "parent-e332-audit-supervision.json"])
        if any(rec(name).get("success") is not True for name in supervisors):
            raise ValueError("producer supervision not successful")
        if number == 336:
            if (audit.get("original_supervision_sha256") != pins[BASE + "/" + supervisors[0]]
                    or audit["terminal_sha256"].get("output-manifest.json") != pins[
                        BASE + "/e336-saved-gait-transfer/output-manifest.json"]):
                raise ValueError("E336 audit pin edge differs")
        elif (audit.get("supervision_sha256") != pins[BASE + "/" + supervisors[0]]
              or audit.get("result_sha256") != pins[BASE + "/e332-smooth-temporal-saved/result.json"]
              or audit.get("output_manifest_sha256") != pins[
                  BASE + "/e332-smooth-temporal-saved/output-manifest.json"]):
            raise ValueError("E332 audit pin edge differs")
    result = rec("e332-smooth-temporal-saved/result.json")
    if (result.get("analysis_valid") is not True or result.get("input_qualification_pass") is not True
            or result.get("first_error") is not None):
        raise ValueError("reference producer terminal differs")
    for role, binding in original_bindings().items():
        number = 332 if role == "reference" else 336
        manifest = rec(f"e{number}-{'smooth-temporal-saved' if number == 332 else 'saved-gait-transfer'}/output-manifest.json")
        if manifest["files"].get(Path(binding["path"]).name) != {
                key: binding[key] for key in ("sha256", "bytes")}:
            raise ValueError("original binding/producer manifest edge differs")
    return {"passed": True, "direct_producer_json_count": len(records),
            "closed_e336_checks": 7141, "closed_e332_checks": 2946,
            "scope": "saved closed-producer text evidence and explicit edges; no leaf rehash"}


def generated_path(name, root, *, branch=None, suffix=None):
    """Current finite text evidence only; never original NPZ or historical trees."""
    prefix = "e339-sol-" if branch == "sol" else "parent-e339-" if branch == "root" else None
    basename = name.removeprefix(BASE + "/") if isinstance(name, str) else ""
    if (not isinstance(name, str) or name != BASE + "/" + basename
            or re.fullmatch(r"[A-Za-z0-9_.-]+", basename) is None
            or not basename.startswith(prefix or ("e339-sol-", "parent-e339-"))
            or Path(name).suffix not in (".json", ".txt", ".py")
            or suffix is not None and not name.endswith(suffix)):
        raise ValueError("finite canonical generated evidence path required")
    return safe(name, root)


def verify_generated_proof(root, proof, owned, timely=lambda: None):
    """Closed two-branch current supervision proof and its direct byte closure.

    No report's historical/prior evidence maps are recursively traversed.
    Returned pins are added to the freeze and rechecked by the production gate.
    """
    if (not isinstance(proof, dict) or set(proof) != {
            "schema", "passed", "original_calls", "source_sha256", "branches", "evidence"}
            or proof["schema"] != PROOF_SCHEMA or proof["passed"] is not True
            or type(proof["original_calls"]) is not int or proof["original_calls"] != 0
            or proof["source_sha256"] != owned):
        raise ValueError("closed current generated source proof required")
    branches, evidence = proof["branches"], proof["evidence"]
    if (not isinstance(branches, dict) or set(branches) != {"sol", "root"}
            or not isinstance(evidence, dict) or not 6 <= len(evidence) <= 32):
        raise ValueError("finite Sol/ROOT generated branches required")
    contract_sources = set()
    # Read only byte-pinned canonical contract JSONs to determine allowed source
    # paths; source files outside BASE are permitted only via this exact roster.
    for branch, links in branches.items():
        if not isinstance(links, dict) or set(links) != {"contract", "supervision", "report"}:
            raise ValueError("closed generated branch links required")
        name = links["contract"]
        path = generated_path(name, root, branch=branch, suffix=".json")
        if name not in evidence or pin(path, timely) != evidence[name]:
            raise ValueError("missing/mutated generated contract byte pin")
        declared = read_json(path).get("source_sha256")
        if not isinstance(declared, dict) or not 3 <= len(declared) <= 16:
            raise ValueError("finite generated contract source roster required")
        for source in declared:
            safe(source, root)
            if not source.endswith(".py"):
                raise ValueError("generated contract sources must be Python text")
            contract_sources.add(source)
    verified = {}
    # Validate all declared byte pins before interpreting the six JSON records.
    for name, expected in evidence.items():
        if name in contract_sources:
            path = safe(name, root)
        else:
            path = generated_path(name, root)
        if (not isinstance(expected, dict) or set(expected) != {"sha256", "bytes"}
                or re.fullmatch("[0-9a-f]{64}", expected.get("sha256", "")) is None
                or type(expected.get("bytes")) is not int or not 0 <= expected["bytes"] <= 8 << 20):
            raise ValueError("strict finite generated byte pin required")
        if pin(path, timely) != expected:
            raise ValueError("generated evidence byte pin differs: " + name)
        verified[name] = copy.deepcopy(expected)
    required = set()
    for branch, links in branches.items():
        if not isinstance(links, dict) or set(links) != {"contract", "supervision", "report"}:
            raise ValueError("closed generated branch links required")
        for field, name in links.items():
            generated_path(name, root, branch=branch, suffix=".json")
            if name not in verified:
                raise ValueError("missing generated branch evidence: " + field)
        if len(set(links.values())) != 3:
            raise ValueError("distinct generated contract/supervision/report required")
        required.update(links.values())
        contract = read_json(safe(links["contract"], root))
        supervision = read_json(safe(links["supervision"], root))
        report = read_json(safe(links["report"], root))
        sources = contract.get("source_sha256")
        if not isinstance(sources, dict) or not 3 <= len(sources) <= 16:
            raise ValueError("finite generated contract source roster required")
        if any(sources.get(name) != sha for name, sha in owned.items()):
            raise ValueError("generated branch current owned source pins differ")
        if sources.get(SUPERVISOR) != SUPERVISOR_SHA:
            raise ValueError("generated supervisor source pin differs")
        for name, sha in sources.items():
            if name not in verified or verified[name]["sha256"] != sha:
                raise ValueError("generated contract source evidence differs")
            required.add(name)
        if (type(contract.get("original_calls")) is not int or contract["original_calls"] != 0
                or type(supervision.get("original_calls")) is not int or supervision["original_calls"] != 0
                or contract.get("supervisor_output") != links["supervision"]
                or supervision.get("schema") != "flylab.m3.e267.root-supervision.v1"
                or supervision.get("success") is not True
                or supervision.get("contract_sha256") != verified[links["contract"]]["sha256"]):
            raise ValueError("generated supervision/contract completion edge differs")
        commands, attempts = contract.get("commands"), supervision.get("attempts")
        if (not isinstance(commands, list) or not 1 <= len(commands) <= 4
                or not isinstance(attempts, list) or len(attempts) != len(commands)):
            raise ValueError("complete finite generated supervised commands required")
        for command, attempt in zip(commands, attempts, strict=True):
            argv = command.get("argv")
            if (not isinstance(argv, list) or not all(isinstance(x, str) for x in argv)
                    or len(argv) < 5 or argv[:4] != [".venv/Scripts/python.exe", "-X", "utf8", "-B"]
                    or "--run" in argv or attempt.get("argv") != argv
                    or attempt.get("requested") is not True or type(attempt.get("exit_code")) is not int
                    or attempt["exit_code"] != 0 or attempt.get("killed") is not False):
                raise ValueError("generated command did not successfully exit zero")
            for stream in ("stdout", "stderr"):
                name = command.get(stream)
                generated_path(name, root, branch=branch, suffix=".txt")
                if name not in verified:
                    raise ValueError("missing generated command log pin")
                required.add(name)
            if branch == "sol":
                if (argv[4:6] not in (["-m", "pytest"], ["-m", "ruff"])
                        or not any(name in argv for name in OWNED)):
                    raise ValueError("current Sol pytest/ruff source command required")
            elif argv[4] not in sources or links["report"] not in argv:
                raise ValueError("ROOT generated script/report command edge differs")
        if branch == "sol" and {x["argv"][5] for x in commands} != {"pytest", "ruff"}:
            raise ValueError("Sol generated supervision must include pytest and ruff")
        if (report.get("passed") is not True or not isinstance(report.get("source_sha256"), dict)
                or any(report["source_sha256"].get(name) != sha for name, sha in owned.items())
                or any(name not in sources or sources[name] != sha
                       for name, sha in report["source_sha256"].items())):
            raise ValueError("generated report current source/passed state differs")
        zero = report.get("original_calls", report.get("original_archive_header_value_hash_calls"))
        if type(zero) is not int or zero != 0:
            raise ValueError("generated report original calls must be zero")
        for key in ("original_archive_header_value_hash_calls", "original_gate_calls",
                    "original_body_neural_CUDA_F6_UI_calls"):
            if key in report and (type(report[key]) is not int or report[key] != 0):
                raise ValueError("generated report nonzero original calls")
        if branch == "sol":
            recorded_supervision = report.get("supervision", {})
            if (report.get("schema") != "flylab.m3.e339.sol-generated-qualification.v1"
                    or report.get("ruff_passed") is not True or report.get("pytest_passed") is not True
                    or type(report.get("pytest_items")) is not int or report["pytest_items"] <= 0
                    or report.get("real_selected_reader_generated") is not True
                    or report.get("real_metadata_freeze_symbolic_namespace") is not True
                    or report.get("loader_mock_success_used") is not False
                    or report.get("production_original_qualification") is not False
                    or recorded_supervision.get("path") != links["supervision"]
                    or recorded_supervision.get("sha256") != verified[links["supervision"]]["sha256"]
                    or recorded_supervision.get("success") is not True
                    or recorded_supervision.get("original_calls") != 0):
                raise ValueError("Sol real-generated qualification/supervision edge differs")
        else:
            checks = report.get("checks")
            if (re.fullmatch(r"flylab\.m3\.e339\.root-(first|second|final|current)-generated-(review|report)\.v1",
                             report.get("schema", "")) is None
                    or not isinstance(checks, dict) or not checks
                    or any(value is not True for value in checks.values())
                    or type(report.get("checks_count")) is not int or report["checks_count"] != len(checks)
                    or report.get("first_error") is not None
                    or report.get("source_review_passed", True) is not True
                    or report.get("R1_required", False) is not False
                    or any(value is not False for value in report.get("defects", {}).values())):
                raise ValueError("ROOT current generated report has unresolved failures")
    if required != set(verified):
        raise ValueError("generated evidence must be exactly the finite direct closure")
    return verified


def make_freeze(root, qualification, *, timely=lambda: None):
    """Metadata-only builder; ROOT saves final freeze after current qualification.

    The generated test namespace can contain symbolic producer JSON at the
    fixed relative names. No archive is opened or hashed by this helper.
    """
    root = Path(root).resolve()
    dependencies = read_json(safe(DEPS, root))
    if dependencies["original_bindings"] != original_bindings():
        raise ValueError("dependency original literal bindings differ")
    sources = {**dependencies["direct_source_sha256"]}
    for name in OWNED:
        sources[name] = digest(safe(name, root), timely)
    documents = (DESIGN, ASTRA, REFLECTION, REGISTRATION, SNAPSHOT, DISPATCH,
                 R1_DISPATCH, FIRST_REVIEW, DEPS, qualification)
    if not qualification.startswith(BASE + "/") or not qualification.endswith(".json"):
        raise ValueError("generated qualification canonical JSON required")
    evidence = {name: pin(safe(name, root), timely) for name in documents}
    for name, sha in FIXED.items():
        if evidence[name]["sha256"] != sha:
            raise ValueError("fixed design/dependency pin differs")
    for name, sha in sources.items():
        if digest(safe(name, root), timely) != sha:
            raise ValueError("direct source differs: " + name)
    proof = read_json(safe(qualification, root))
    owned = {name: sources[name] for name in OWNED}
    generated_evidence = verify_generated_proof(root, proof, owned, timely)
    evidence.update(generated_evidence)
    dispatch = read_json(safe(DISPATCH, root))
    if (dispatch.get("requested_model") != "gpt-6.1-sol"
            or dispatch.get("requested_reasoning_effort") != "high"):
        raise ValueError("actual implementation dispatch differs")
    reflection = read_json(safe(REFLECTION, root))
    if (reflection.get("all_required_items_reflected") is not True
            or reflection.get("reflected_design_sha256") != evidence[DESIGN]["sha256"]
            or reflection.get("review_sha256") != evidence[ASTRA]["sha256"]):
        raise ValueError("Astra reflection edges differ")
    for field, name in (("design_sha256", DESIGN), ("review_sha256", ASTRA),
                        ("reflection_sha256", REFLECTION), ("direct_dependencies_sha256", DEPS)):
        if dispatch.get(field) != evidence[name]["sha256"]:
            raise ValueError("actual dispatch byte edge differs: " + field)
    if (reflection.get("direct_dependencies_sha256") != evidence[DEPS]["sha256"]
            or reflection.get("requested_model") != "gpt-6-astra"
            or reflection.get("requested_effort") != "max"):
        raise ValueError("actual Astra/dependency reflection differs")
    registration = read_json(safe(REGISTRATION, root))
    if (registration.get("pre_review_design_sha256") != evidence[SNAPSHOT]["sha256"]
            or reflection.get("pre_review_design_sha256") != evidence[SNAPSHOT]["sha256"]
            or registration.get("actual_review_dispatch", {}).get("requested_model") != "gpt-6-astra"
            or registration.get("actual_review_dispatch", {}).get("requested_effort") != "max"
            or dispatch.get("owned_source") != list(OWNED)):
        raise ValueError("pre-review registration/actual dispatch edge differs")
    r1_dispatch = read_json(safe(R1_DISPATCH, root))
    first_review = read_json(safe(FIRST_REVIEW, root))
    if (r1_dispatch.get("requested_model") != "gpt-6.1-sol"
            or r1_dispatch.get("requested_reasoning_effort") != "high"
            or r1_dispatch.get("first_review_sha256") != evidence[FIRST_REVIEW]["sha256"]
            or r1_dispatch.get("first_source_sha256") != first_review.get("current_source_sha256")):
        raise ValueError("SAME Sol ONE R1 dispatch/review edge differs")
    producer = verify_producers(root, dependencies, timely)
    for name, sha in dependencies["producer_text_sha256"].items():
        evidence[name] = {**pin(safe(name, root), timely)}
        if evidence[name]["sha256"] != sha:
            raise ValueError("producer changed while freezing")
    return {"schema": SCHEMA, "output": OUTPUT, "limits": copy.deepcopy(LIMITS),
            "original_bindings": original_bindings(), "source_sha256": sources,
            "evidence": evidence, "qualification": qualification, "producer_states": producer,
            "scope": "finite direct dependencies; original4 selected-only; no inherited rehash",
            "science_adopted": False, "candidate_adopted": False, "F1_F8_complete": False}


def gate(freeze_sha, review_sha, timely=lambda: None):
    # Production root, binding/output/scope and freeze/review paths are literals.
    for sha in (freeze_sha, review_sha):
        if re.fullmatch("[0-9a-f]{64}", sha or "") is None:
            raise ValueError("explicit freeze/review SHA256 required")
    if digest(safe(FREEZE, ROOT), timely) != freeze_sha:
        raise ValueError("freeze byte pin differs")
    frozen = read_json(safe(FREEZE, ROOT))
    rebuilt = make_freeze(ROOT, frozen["qualification"], timely=timely)
    if frozen != rebuilt:
        raise ValueError("current finite freeze differs")
    if digest(safe(REVIEW, ROOT), timely) != review_sha:
        raise ValueError("ROOT source review byte pin differs")
    review = read_json(safe(REVIEW, ROOT))
    if (review.get("passed") is not True or review.get("freeze_sha256") != freeze_sha
            or review.get("source_sha256") != {k: frozen["source_sha256"][k] for k in OWNED}
            or review.get("generated_qualification_sha256") != frozen["evidence"][
                frozen["qualification"]]["sha256"]):
        raise ValueError("ROOT current freeze/source/proof review differs")
    return frozen


class Clock:
    def __init__(self, s0, now=time.perf_counter):
        self.s0, self.now = s0, now
        self.final_begin = None

    def science(self):
        if self.now() >= self.s0 + 8:
            raise TimeoutError("science deadline s0+8")

    def begin_final(self):
        self.final_begin = self.now()
        self.final()

    def final(self):
        if self.final_begin is None or self.now() >= min(self.s0 + 12, self.final_begin + 4):
            raise TimeoutError("finalization deadline min(s0+12, final_begin+4)")


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf8")


def terminal_success(terminal):
    """An error can never coexist with effective successful completion."""
    return (terminal.get("status") == "valid_provisional_diagnostic"
            and terminal.get("analysis_valid") is True and terminal.get("publication_valid") is True
            and terminal.get("first_error") is None and terminal.get("secondary_errors") == []
            and terminal.get("suggested_diagnostic") in (
                "proxy_limited", "angular_misalignment_prominent",
                "no_prominent_angular_misalignment_at_fixed_rule")
            and terminal.get("provisional") is True and terminal.get("science_adopted") is False
            and terminal.get("candidate_adopted") is False and terminal.get("F1_F8_complete") is False)


def invalidate(terminal, error=None, stage="terminal_coherence"):
    terminal.update(status="technical_invalid", analysis_valid=False,
                    publication_valid=False, suggested_diagnostic=None,
                    science_adopted=False, candidate_adopted=False, F1_F8_complete=False)
    if error is not None:
        record = error_record(error, stage)
        if terminal.get("first_error") is None:
            terminal["first_error"] = record
        else:
            terminal.setdefault("secondary_errors", []).append(record)
    return terminal


def coherent_terminal(terminal):
    """Preserve computed/evidence fields; revoke every inconsistent success."""
    if not terminal_success(terminal):
        error = None if terminal.get("first_error") is not None else ValueError("error-free coherent success required")
        invalidate(terminal, error)
    return terminal


def publish(output, terminal, arrays, ledger, clock, *, cap=4 << 20):
    """Fresh directory, nonselfreferencing manifest, byte/hash readback.

    On failure a distinct revocation is attempted; success files are never
    overwritten. ROOT checks external stdout/stderr/supervisor/audit scope.
    """
    output = Path(output)
    created = False
    phase = "create_output"
    terminal = coherent_terminal(copy.deepcopy(terminal))
    records = {}
    used = 0
    def write(name, payload):
        nonlocal used
        clock.final()
        # Retain headroom for terminal stdout, external receipt and revocation.
        if used + len(payload) > cap - (512 << 10):
            raise ValueError("worker retained output budget exceeded")
        path = output / name
        staging = output / (name + ".partial")
        with staging.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        used += len(payload)
        clock.final()
        if path.exists():
            raise FileExistsError("fresh target already exists")
        staging.rename(path)
        records[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        clock.final()
    try:
        clock.final()
        output.mkdir(exist_ok=False)
        created = True
        phase = "numeric_archives"
        import io
        for role, data in arrays.items():
            clock.final()
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **data)
            write(role + "-coordination.npz", buffer.getvalue())
        phase = "selected_read_ledger"
        write("selected-read-ledger.json", json_bytes(ledger))
        phase = "terminal_receipt"
        payload = json_bytes(terminal)
        write("result.json", payload)
        write("receipt.json", payload)
        phase = "manifest"
        manifest = {"schema": "flylab.m3.e339.output-manifest.v1", "files": copy.deepcopy(records),
                    "self_reference": False, "stdout_hashed": False,
                    "provisional": True, "science_adopted": False}
        manifest_bytes = json_bytes(manifest)
        write("output-manifest.json", manifest_bytes)
        phase = "readback"
        for name, expected in records.items():
            clock.final()
            path = output / name
            if pin(path, clock.final) != expected:
                raise ValueError("publication readback/hash differs: " + name)
        if read_json(output / "result.json") != terminal or read_json(output / "receipt.json") != terminal:
            raise ValueError("terminal/receipt readback differs")
        if (output / "output-manifest.json").read_bytes() != manifest_bytes:
            raise ValueError("manifest readback differs")
        clock.final()
        return terminal
    except Exception as error:  # noqa: BLE001 -- All publication failures invalidate provisional success.
        invalidate(terminal, error, "publication." + phase)
        if created:
            try:
                payload = json_bytes(terminal)
                if used + len(payload) > cap:
                    raise ValueError("revocation output cap exceeded")
                with (output / "invalid-revocation.json").open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            except Exception as revoke_error:  # noqa: BLE001 -- Preserve failed revocation for ROOT.
                terminal.setdefault("secondary_errors", []).append(error_record(revoke_error, "revocation"))
        return terminal


def worker(freeze_sha, review_sha, s0):
    clock = Clock(s0)
    terminal = {"schema": "flylab.m3.e339.saved-coordination.result.v1",
                "status": "technical_invalid", "analysis_valid": False,
                "publication_valid": False, "provisional": True,
                "suggested_diagnostic": None, "science_adopted": False,
                "candidate_adopted": False, "F1_F8_complete": False,
                "first_error": None, "secondary_errors": [], "limits": LIMITS,
                "freeze_sha256": freeze_sha, "review_sha256": review_sha,
                "clock_s0": s0, "science_deadline": s0 + 8, "internal_deadline": s0 + 12}
    arrays, ledger = {}, {"archives": {}, "scope": "actual selected arrays only"}
    stage = "imports"
    try:
        runtime()
        clock.science()
        stage = "gate"
        freeze = gate(freeze_sha, review_sha, clock.science)
        if safe(OUTPUT, ROOT).exists():
            raise FileExistsError("literal fresh output already exists")
        stage = "fixed_reader"
        inputs = {}
        for role, binding in freeze["original_bindings"].items():
            try:
                inputs[role], ledger["archives"][role] = fixed_reader(
                    ROOT, role, binding, timely=clock.science)
            except ReaderFailure as error:
                ledger["archives"][role] = error.detail
                raise
        stage = "science"
        analysis, arrays = analyze(inputs, clock.science)
        terminal.update(status="valid_provisional_diagnostic", analysis_valid=True,
                        publication_valid=True, suggested_diagnostic=analysis["diagnosis"]["suggested_diagnostic"],
                        analysis=analysis)
        clock.science()
    except Exception as error:  # noqa: BLE001 -- Worker preserves failure, never adopts science.
        invalidate(terminal, error, stage)
        if hasattr(error, "detail"):
            terminal["reader_failure"] = error.detail
    try:
        clock.begin_final()
        terminal.update(final_begin=clock.final_begin,
                        final_deadline=min(s0 + 12, clock.final_begin + 4))
        terminal = publish(safe(OUTPUT, ROOT), terminal, arrays, ledger, clock)
    except Exception as error:  # noqa: BLE001 -- Final deadline/publication failure is technical_invalid.
        invalidate(terminal, error, "final_deadline")
    return terminal


def main(argv=None):
    s0 = time.perf_counter()  # Before NumPy/scanner imports, parse and gate.
    for key in THREAD_ENV:
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review-sha256", required=True)
    args = parser.parse_args(argv)
    terminal = coherent_terminal(worker(args.freeze_sha256, args.review_sha256, s0))
    sys.stdout.write(json_bytes(terminal).decode("utf8"))
    sys.stdout.flush()
    return 0 if terminal_success(terminal) else 1


if __name__ == "__main__":
    raise SystemExit(main())
