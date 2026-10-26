"""Finite startup R ray and continuous arm lifecycle. Original admission is separate.

No arrays, numerical imports, checkpoints or bodies are opened before the same
ROOT grant/request/Source/input/debit-offset gate. The old alpha=.001 experiment
and its validators are not used to validate this new multi-alpha raw schema.
"""
# ruff: noqa: BLE001 -- preserve primary and cleanup failures in scientific evidence.
from __future__ import annotations

import time

ENTRY = time.perf_counter()
import argparse
import hashlib
import io
import json
import math
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = (.0009, .00075, .0005, .00025, .0001)
WINDOWS = (100, 164, 228, 292)
ROLES = ("input_audit", "ray", "candidate_audit", "neutral_body", "candidate_body",
         "neutral_audit", "candidate_body_audit", "close")
OUTER = (120, 1200, 120, 300, 300, 120, 120, 180)
CHILD = (110, 1180, 110, 285, 285, 110, 110, 170)
INTERNAL = (100, 1150, 100, 270, 270, 100, 100, 160)
SOURCE = "tools/verify_c_banc_m3_startup_ray.py"
SCHEMA = "flylab.m3.startup-ray"
STATE_KEYS = ("rate", "adaptation", "bank", "original_activation")
HELPER_SOURCES = {
    "tools/verify_c_banc_m3_reference_parameter_direction.py": (74213, "656415a13cde92d469b9ad281a16b969ce26b87e543bdc4161961ab621d7005b"),
    "tools/verify_c_banc_m3_startup_baseline.py": (90298, "4e853f8c4fae105102625816f3075d3b4b7d3ac7312506dd7b3e392e00607342"),
    "tools/verify_c_banc_m3_candidate_body_handoff.py": (94828, "5b43ee4ee34a8270ba123d11e99e9820b6f2d25644c027bc46a86ab9b6f7532a"),
    "tools/verify_c_banc_m3_common_descent_body.py": (83088, "7d123ea4d005f4c45c4cc386f94edd09eabab9a5e54b873dd855613ea272bc4e"),
    "tools/verify_c_banc_m3_coordination_parameter_direction.py": (87181, "fb4d2687baf9ad89b919b7c4692d7039992a319ffef082686933d1bbc563bb2d"),
    "tools/banc_m3_checkpointed_rollout.py": (21885, "49ad4c3b1f564335d1495ac7897c5fcc8b789732296b1ab7906bc9322b07788c"),
    "tools/banc_m3_ssp_adjoint.py": (11923, "d74d6abafd64205f085364f590a7a2f658010af4c8b79694694a4f248f8ceab4"),
    "tools/banc_m3_grouped_current_adjoint.py": (18975, "0456c4d85fe0b1ebe99d2902f5b43c4f2dc27e63695a98d6c89035c60878c748"),
}


def require(ok, label):
    if not ok:
        raise ValueError(label)


def exact_value(actual, expected, label):
    require(type(actual) is type(expected), label + " type")
    if isinstance(expected, dict):
        require(set(actual) == set(expected), label + " fields")
        for key in expected:
            exact_value(actual[key], expected[key], label + "." + key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), label + " length")
        for a, b in zip(actual, expected, strict=True):
            exact_value(a, b, label)
    else:
        require(actual == expected, label)


def path(name, root=ROOT):
    require(type(name) is str and name and "\\" not in name and not Path(name).is_absolute(), "relative path")
    p = root / name
    require(".." not in Path(name).parts and p.resolve().is_relative_to(root.resolve()), "contained path")
    for part in (p, *p.parents):
        if part == root.parent:
            break
        if part.exists():
            stat = part.lstat()
            require(not part.is_symlink() and not (getattr(stat, "st_file_attributes", 0) & 0x400), "no reparse")
    return p


def json_load(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    return json.loads(raw.decode("utf8", errors="strict"), object_pairs_hook=pairs,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def leaf(spec, root=ROOT, *, cap=262144, allow_empty=False):
    require(type(spec) is dict and set(spec) == {"path", "bytes", "sha256"}, "descriptor ABI")
    require(type(spec["bytes"]) is int and (0 if allow_empty else 1) <= spec["bytes"] <= cap and type(spec["sha256"]) is str
            and re.fullmatch("[0-9a-f]{64}", spec["sha256"]), "bounded typed descriptor")
    p = path(spec["path"], root)
    before = p.stat()
    require(before.st_size == spec["bytes"], "exact file size")
    raw = p.read_bytes()
    raw.decode("utf8", errors="strict")
    require(raw == p.read_bytes() and p.stat().st_mtime_ns == before.st_mtime_ns
            and hashlib.sha256(raw).hexdigest() == spec["sha256"], "cold readback")
    return raw


def spec_for(name, root=ROOT, *, cap=262144):
    p = path(name, root)
    before = p.stat()
    require(type(cap) is int and 0 < before.st_size <= cap, "bounded descriptor publication")
    raw = p.read_bytes()
    require(len(raw) == before.st_size and raw == p.read_bytes()
            and p.stat().st_mtime_ns == before.st_mtime_ns, "descriptor cold readback")
    return {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def source_leaf(spec, root=ROOT):
    """Explicit frozen installed Source only; never an Original array/checkpoint reader."""
    if not Path(spec["path"]).is_absolute():
        return leaf(spec, root, cap=2097152, allow_empty=True)
    require(set(spec) == {"path", "bytes", "sha256"} and type(spec["bytes"]) is int
            and 0 <= spec["bytes"] <= 2097152 and re.fullmatch("[0-9a-f]{64}", spec["sha256"])
            and Path(spec["path"]).suffix in (".py", ".ps1", ".json", ".txt", ".ini", ".md"), "exact foreign text Source")
    p = Path(spec["path"])
    for part in (p, *p.parents):
        stat = part.lstat()
        require(not part.is_symlink() and not (getattr(stat, "st_file_attributes", 0) & 0x400), "foreign Source reparse")
    before = p.stat()
    require(before.st_size == spec["bytes"], "foreign Source bytes")
    raw = p.read_bytes();raw.decode("utf8", errors="strict")
    require(raw == p.read_bytes() and p.stat().st_mtime_ns == before.st_mtime_ns
            and hashlib.sha256(raw).hexdigest() == spec["sha256"], "foreign cold Source fence")
    return raw


def finite_float(value, label, *, minimum=0):
    require(type(value) is float and math.isfinite(value) and value >= minimum, "typed finite " + label)
    return value


class Admission:
    """Finite science gate; ROOT issues the actual records and owns their accounting."""
    def __init__(self, request_path, stage, *, root=ROOT, entry=ENTRY):
        require(stage in ("pre_spawn", "pre_array"), "admission stage")
        self.root, self.entry, self.stage = root.resolve(), entry, stage
        self.request_spec = spec_for(request_path, root)
        self.request = json_load(leaf(self.request_spec, root))
        self.manifest_spec = self.request["manifest"]
        self.manifest = json_load(leaf(self.manifest_spec, root))
        m, q = self.manifest, self.request
        require(q["schema"] == SCHEMA + ".request.v1" and m["schema"] == SCHEMA + ".manifest.v1",
                "science request/manifest")
        self.role = q["role"]
        require(self.role in ROLES and q["mode"] == "original", "Original science role whitelist")
        i = ROLES.index(self.role)
        exact_value(m["alphas"], list(ALPHAS), "sealed descending ray")
        exact_value(m["roles"], list(ROLES), "fixed sequential roster")
        require(type(m["attempt_id"]) is str and re.fullmatch("startup-ray-[a-z0-9-]{1,48}", m["attempt_id"]), "attempt DATA")
        require(set(m["role_paths"]) == set(ROLES), "complete fixed role path roster")
        for role in ROLES:
            expected_paths = {"request": "verification/banc-natural-gait-m3-20260919/parent-" + m["attempt_id"] + "-" + role + "-request.json",
                              "output": "verification/banc-natural-gait-m3-20260919/parent-" + m["attempt_id"] + "/" + role}
            exact_value(m["role_paths"][role], expected_paths, "typed finite role path templates")
        exact_value(q["call_id"], m["attempt_id"] + "-" + self.role, "call identity")
        paths = m["role_paths"][self.role]
        exact_value(self.request_spec["path"], paths["request"], "request path")
        exact_value(q["output"], paths["output"], "output whitelist")
        self.output = path(q["output"], root)
        self.limits = q["limits"]
        require(set(m["role_limits"]) == set(ROLES), "complete reviewed per-role resources")
        exact_value(self.limits, m["role_limits"][self.role], "manifest to request exact limits")
        for key, value in (("outer_s", OUTER[i]), ("child_s", CHILD[i]), ("internal_s", INTERNAL[i])):
            exact_value(self.limits[key], value, "fixed full interval margin")
        require(type(self.limits["host_bytes"]) is int and self.limits["host_bytes"] > 0
                and type(self.limits["output_cap_bytes"]) is int and self.limits["output_cap_bytes"] > 0, "reviewed physical resource bounds")
        for key in ("grant", "source_freeze", "inputs", "local_debit_offset"):
            exact_value(q[key], m[key], "same " + key)
        require(type(q["artifacts"]) is dict, "finite role artifact bindings")
        for key in ("sample_cap", "GPU_delta_plus_allowance_cap_bytes"):
            require(type(self.limits[key]) is int and self.limits[key] > 0, "typed reviewed physical limit " + key)
        self.grant = json_load(leaf(m["grant"], root))
        self.freeze = json_load(leaf(m["source_freeze"], root))
        self.inputs = json_load(leaf(m["inputs"], root))
        g = self.grant
        require(g["schema"] == SCHEMA + ".original-grant.v1" and g["mode"] == "original", "actual dedicated Original grant")
        for key, expected in (("additional_s", 6000), ("global_limit_s", 16200), ("stage_limit_s", 6300),
                              ("science_local_limit_s", 6000), ("worker_max", 1), ("automatic_restarts", 0), ("reset", False)):
            exact_value(g[key], expected, "new grant " + key)
        exact_value(g["source_freeze"], m["source_freeze"], "grant Source")
        exact_value(g["inputs"], m["inputs"], "grant inputs")
        exact_value(g["local_debit_offset"], m["local_debit_offset"], "same issued debit offset")
        finite_float(g["original_cumulative_before_s"], "actual cumulative anchor")
        finite_float(g["original_stage_before_s"], "actual stage anchor")
        require(type(m["local_debit_offset"]) is int and m["local_debit_offset"] >= 0, "integer issued offset")
        before = json_load(leaf(g["before_ledger"], root, cap=524288))
        require(len(before["debits"]) == m["local_debit_offset"], "actual immutable offset anchor")
        exact_value(before["original_cumulative_used_s"], g["original_cumulative_before_s"], "actual unchanged cumulative anchor")
        exact_value(before["original_stage_used_s"], g["original_stage_before_s"], "actual unchanged stage anchor")
        self.before = before
        require(self.freeze["schema"] == SCHEMA + ".source-freeze.v1"
                and self.freeze["ROOT_review_cycle_complete"] is True, "actual reviewed Source freeze")
        exact_value(self.freeze["human_authorization"], g["human_authorization"], "same human delegation")
        for name in ("actual_ROOT_FIRST", "actual_ROOT_SECOND", "actual_Sol_FIRST"):
            leaf(self.freeze[name], root)
        self.fence()

    def check_sources(self):
        for name, (size, digest) in HELPER_SOURCES.items():
            exact_value(self.freeze["sources"][name], {"path": name, "bytes": size, "sha256": digest}, "reviewed reused helper Source")
        for name, descriptor in self.freeze["sources"].items():
            exact_value(name, descriptor["path"], "Source dictionary path")
            source_leaf(descriptor, self.root)
        require(SOURCE in self.freeze["sources"] and "tools/banc_m3_startup_ray_supervisor.py" in self.freeze["sources"], "actual science and physical-owner modules frozen")

    def check(self, stage=None):
        elapsed = time.perf_counter() - self.entry
        require(math.isfinite(elapsed) and elapsed >= 0, "same actual ENTRY clock")
        if elapsed > self.limits["internal_s"]:
            raise TimeoutError("science same-entry internal interval")
        return True

    def fence(self, stage=None):
        """Fresh admission/ownership/budget/Source at each physical boundary; no success cache."""
        self.check(stage)
        m, g, q = self.manifest, self.grant, self.request
        for descriptor, value in ((self.request_spec, q), (self.manifest_spec, m), (m["grant"], g),
                                  (m["source_freeze"], self.freeze), (m["inputs"], self.inputs)):
            exact_value(json_load(leaf(descriptor, self.root)), value, "cold exact parsed current context")
        leaf(g["before_ledger"], self.root, cap=524288)
        self.check_sources()  # Before importing the physical-owner helper, then again at the cold boundary end.
        live = json_load(leaf(spec_for(m["ledger_path"], self.root, cap=524288), self.root, cap=524288))
        offset = m["local_debit_offset"]
        exact_value(live["debits"][:offset], self.before["debits"], "all old charged costs preserved")
        expected = {"grant": m["grant"], "source_freeze": m["source_freeze"], "inputs": m["inputs"],
                    "local_debit_offset": offset, "worker_max": 1, "automatic_restarts": 0}
        exact_value(live["startup_ray_original_grants"][g["grant_id"]], expected, "actual complete grant event")
        for key in ("startup_ray_original_pending_actuals", "startup_ray_original_blocking_holds", "e360_pending_actuals", "e360_blocking_holds", "direct_science_pending_actuals", "direct_science_blocking_holds"):
            require(type(live[key]) is dict and not live[key], "no pending/unknown Original entry " + key)
        require(not live["e359_reservations"] and not live["e360_reservations"], "no unrelated retained reservations")
        workers = live["startup_ray_original_workers"]
        from tools import banc_m3_startup_ray_supervisor as supervisor
        exact_value(m["shared_original_lock_path"], supervisor.SHARED_LOCK, "existing actual shared Original exclusion")
        lock_path = path(m["shared_original_lock_path"], self.root)
        reservations = live["startup_ray_original_reservations"]
        if self.stage == "pre_spawn":
            exact_value(workers, {}, "sequential ownership closed before spawn")
            exact_value(reservations, {}, "no existing Original reservation")
            require(not lock_path.exists(), "actual shared lock competition blocks before arrays")
        else:
            require(type(workers) is dict and set(workers) == {q["call_id"]}, "one admitted physical worker")
            worker = workers[q["call_id"]]
            expected_keys = {"request", "grant", "state", "worker_max", "reserved_s", "lock_token",
                             "worker_pid", "worker_creation_ticks", "owner_pid", "owner_creation_ticks"}
            require(set(worker) == expected_keys, "complete physical worker lease")
            for key, value in (("request", self.request_spec), ("grant", m["grant"]), ("state", "begun"),
                               ("worker_max", 1), ("reserved_s", self.limits["outer_s"])):
                exact_value(worker[key], value, "physical lease " + key)
            exact_value(worker["worker_pid"], os.getpid(), "this actual pre-array worker PID")
            for key in ("worker_pid", "worker_creation_ticks", "owner_pid", "owner_creation_ticks"):
                require(type(worker[key]) is int and worker[key] > 0, "Native integer physical identity")
            require(worker["worker_pid"] != worker["owner_pid"], "separate actual supervisor/worker")
            exact_value(supervisor.creation_ticks(worker["worker_pid"]), worker["worker_creation_ticks"], "actual worker creation")
            exact_value(supervisor.creation_ticks(worker["owner_pid"]), worker["owner_creation_ticks"], "actual owner creation")
            lock = json_load(leaf(worker["lock_token"], self.root))
            exact_value(worker["lock_token"]["path"], m["shared_original_lock_path"], "same actual lock token path")
            exact_value(lock["call_id"], q["call_id"], "actual lock call")
            exact_value(lock["entry_counter_s"], self.entry, "same original absolute supervisor ENTRY")
            for key in ("owner_pid", "owner_creation_ticks"):
                exact_value(lock[key], worker[key], "same lock owner identity")
            require(supervisor.in_named_job(lock["job_name"]), "this worker actually contained in this owner's Job")
            outer_call = live["direct_science_calls"][q["call_id"]]
            exact_value(outer_call, {"request": lock["request"], "state": "admitted", "mode": "original", "reserved_s": self.limits["outer_s"]}, "actual Root-admitted supervisor call")
            outer_request = json_load(leaf(outer_call["request"], self.root))
            exact_value(outer_request["science_request"], self.request_spec, "actual outer to this same pre-array request")
            exact_value(reservations, {q["call_id"]: {"request": self.request_spec, "grant": m["grant"],
                                                   "reserved_s": self.limits["outer_s"]}}, "sole exact Original reservation")
        exact_value(live["original_jobs_running"], 0 if self.stage == "pre_spawn" else 1, "shared Original worker exclusion")
        exact_value(live["original_reserved_s"], 0 if self.stage == "pre_spawn" else self.limits["outer_s"], "actual sole Original reserved seconds")
        used, completed, failed, seen = [], [], [], set()
        for debit in live["debits"][offset:]:
            if debit["mode"] != "original":
                continue
            cost = json_load(leaf(debit["descriptor"], self.root))
            require(cost["schema"] == SCHEMA + ".full-return.v1" and cost["charged_once"] is True, "actual science once cost")
            require(cost["cost_id"] not in seen and cost["role"] in ROLES, "unique actual science return")
            seen.add(cost["cost_id"])
            exact_value(cost["grant"], m["grant"], "cost same grant")
            for field in ("source_freeze", "inputs", "local_debit_offset"):
                exact_value(cost[field], m[field], "cost same " + field)
            exact_value(cost["cost_id"], debit["cost_id"], "once cost identity")
            request = json_load(leaf(cost["request"], self.root))
            exact_value(request["call_id"], cost["cost_id"], "actual request/cost identity")
            exact_value(request["manifest"], self.manifest_spec, "actual paid request manifest")
            seconds = finite_float(cost["actual_outer_full_return_s"], "actual full return", minimum=1e-12)
            exact_value(debit["actual_outer_full_return_s"], seconds, "exact numeric debit")
            require(type(cost["exit_code"]) is int, "original actual exit")
            exact_value(debit["actual_returned_exit_code"], cost["exit_code"], "exact returned exit")
            exact_value(cost["all_owned_ended"], True, "prior actual owned close")
            exact_value(cost["reservation_released"], True, "prior reservation release")
            completed.append(cost["role"])
            if cost["exit_code"] != 0:
                failed.append(cost)
            used.append(seconds)
        i = ROLES.index(self.role)
        if self.role == "close":
            exact_value(completed, list(ROLES[:len(completed)]), "single finite prefix before mandatory close")
        else:
            require(not failed, "failed role cannot silently retry or advance")
            exact_value(completed, list(ROLES[:i]), "sequential actual role returns")
        current = 0.0
        for seconds in used:
            current += seconds  # exact ordered double arithmetic of the Native once writer
        cumulative = finite_float(g["original_cumulative_before_s"], "unchanged Original cumulative") + current
        stage_used = finite_float(g["original_stage_before_s"], "unchanged Original stage") + current
        exact_value(live["original_cumulative_limit_s"], 16200, "actual issued global limit")
        exact_value(live["original_stage_cap_s"], 6300, "actual issued stage limit")
        exact_value(live["startup_ray_original_local_limit_s"], 6000, "actual new science local limit")
        exact_value(live["original_cumulative_used_s"], cumulative, "actual ordered global used")
        exact_value(live["original_stage_used_s"], stage_used, "actual ordered stage used")
        carry = finite_float(g.get("science_local_used_before_s", 0.0), "preserved earlier scope spending")
        exact_value(live["startup_ray_original_local_used_s"], carry + current, "actual typed scope local used")
        exact_value(live["original_remaining_s"], 16200 - g["original_cumulative_before_s"] - current, "actual writer remaining order")
        necessary = sum(OUTER[i:])
        require(cumulative + necessary <= 16200 and stage_used + necessary <= 6300
                and carry + current + necessary <= 6000, "reserve all remaining audit/close/full returns")
        # Existing Original20 never supplies any authorization to this dedicated gate.
        if self.role in ("neutral_body", "candidate_body"):
            leaf(self.inputs["body_native_dependencies"], self.root)
            if self.role == "neutral_body":
                leaf(self.inputs["neutral_registration"]["body_inventory"], self.root)
        self.check_sources()
        return True


def array_digest(value):
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def normalized_direction(gradient):
    import numpy as np
    require(gradient.dtype == np.float64 and gradient.ndim == 1 and gradient.size > 0
            and np.isfinite(gradient).all(), "finite float64 summed R gradient")
    magnitude = float(np.max(np.abs(gradient)))
    return None if magnitude == 0 else -gradient / magnitude


def validate_actual_theta(actual):
    import numpy as np
    require(actual.dtype == np.float64 and actual.ndim == 1 and actual.size > 0
            and np.isfinite(actual).all() and np.all(actual > 0)
            and np.array_equal(actual, actual.astype(np.float32).astype(np.float64)), "actual f32-promoted positive theta")
    require(float(np.max(np.abs(actual - 1))) <= .001, "actual float32 delta bound without epsilon")
    return True


def proposal(gradient, alpha, csr, mask, previous=()):
    import numpy as np

    from tools import verify_c_banc_m3_reference_parameter_direction as ref
    require(type(alpha) is float and alpha in ALPHAS and 0 < alpha <= .001, "fixed requested alpha")
    direction = normalized_direction(gradient)
    if direction is None:
        return {"eligible_representation": False, "reason": "zero_R_gradient"}, None
    requested = 1 + alpha * direction
    actual = requested.astype(np.float32).astype(np.float64)
    evidence = {"alpha": alpha, "requested_max_delta": float(np.max(np.abs(requested - 1))),
                "actual_max_delta": float(np.max(np.abs(actual - 1))), "theta_sha256": array_digest(actual)}
    arrays = {"theta_requested": requested, "theta_actual": actual}
    if not np.isfinite(actual).all() or np.any(actual <= 0):
        return {**evidence, "eligible_representation": False, "reason": "nonfinite_or_nonpositive"}, arrays
    if evidence["actual_max_delta"] > .001:
        return {**evidence, "eligible_representation": False, "reason": "actual_float32_bound"}, arrays
    validate_actual_theta(actual)
    if np.array_equal(actual, np.ones_like(actual)):
        return {**evidence, "eligible_representation": False, "reason": "quantized_noop"}, arrays
    weights, palette = ref.native().patch_weights(csr, mask, actual)
    original = csr["data"][mask["positions"]]
    arrays["selected_weights_candidate"] = weights
    evidence.update(selected_sha256=array_digest(weights), palette=palette)
    if np.array_equal(weights.view(np.uint32), original.view(np.uint32)):
        return {**evidence, "eligible_representation": False, "reason": "deployed_noop"}, arrays
    if any(evidence["theta_sha256"] == row["theta_sha256"] or evidence["selected_sha256"] == row.get("selected_sha256") for row in previous):
        return {**evidence, "eligible_representation": False, "reason": "rounded_duplicate"}, arrays
    full = csr["data"].copy()
    full[mask["positions"]] = weights
    other = np.ones(len(full), bool)
    other[mask["positions"]] = False
    require(np.array_equal(full[other].view(np.uint32), csr["data"][other].view(np.uint32))
            and np.array_equal(np.signbit(weights), np.signbit(original)), "unselected/sign preservation")
    require(palette["full_data_c_order_sha256"] == array_digest(full), "actual patched full CSR bytes")
    return {**evidence, "eligible_representation": palette["native_deployable"],
            "reason": None if palette["native_deployable"] else "palette_cap"}, arrays


def choose_first(records):
    require(len(records) <= len(ALPHAS), "bounded alpha evaluations")
    for i, row in enumerate(records):
        exact_value(row["alpha"], ALPHAS[i], "fixed first-eligible order")
        if row.get("eligible") is True:
            require(i == len(records) - 1, "stop immediately after first eligibility")
            return i
    return None


def validate_decision_roster(records, evaluated, gradient):
    require(type(evaluated) is int and evaluated == sum(row.get("forward_evaluated") is True for row in records), "actual independently counted forward alpha rows")
    selected = choose_first(records)
    zero = normalized_direction(gradient) is None
    require((zero and not records) or (not zero and (selected is not None or len(records) == len(ALPHAS))), "nonzero ray cannot silently omit remaining fixed alphas")
    return selected


def analysis(raw):
    """Same R/O/Q/T/reconciliation math, with the new explicit alpha receipt."""
    import numpy as np

    from tools import verify_c_banc_m3_reference_parameter_direction as ref
    norm = np.repeat(raw["normalization"], 7)
    neutral = np.sum((raw["q_neutral"] - raw["target_q"]) ** 2 / norm, axis=(1, 2)) / 672
    gq = (raw["q_neutral"] - raw["target_q"]) / (336 * norm)
    require(np.allclose(neutral, raw["R_neutral"], rtol=1e-10, atol=1e-12)
            and np.allclose(gq, raw["gq_R"], rtol=1e-10, atol=1e-12)
            and np.allclose(raw["gtheta_R"].sum(axis=0), raw["sum_gtheta_R"], rtol=1e-10, atol=1e-12), "R/gradient arithmetic")
    candidate = np.sum((raw["q_candidate"] - raw["target_q"]) ** 2 / norm, axis=(1, 2)) / 672
    delta = raw["q_candidate"] - raw["q_neutral"]
    observed = candidate - neutral
    q = np.einsum("wij,wij->w", gq, delta)
    quadratic = q + np.sum(delta ** 2 / norm, axis=(1, 2)) / 672
    t = raw["gtheta_R"] @ (raw["theta_actual"] - 1)
    require(np.allclose(observed, quadratic, rtol=1e-10, atol=1e-12)
            and np.isclose(t.sum(), raw["sum_gtheta_R"] @ (raw["theta_actual"] - 1), rtol=1e-10, atol=1e-12), "unchanged quadratic/parameter identities")
    oq, ot = ref.native().reconciliation(observed, q), ref.native().reconciliation(observed, t)
    supported = -float(observed.sum()) > 1e-9 and -float(t.sum()) > 1e-9 and oq["passed"] and ot["passed"]
    return {"O": observed.tolist(), "Q": q.tolist(), "Q_exact": quadratic.tolist(), "T": t.tolist(),
            "O_Q": oq, "O_T": ot, "I_R": -float(observed.sum()), "supported": bool(supported),
            "meaningful": bool(oq["meaningful"] and ot["meaningful"])}


def deployed_rollout(bundle, packet, arrays):
    """Actually execute patched float32 weights with theta1; no artifact-only equivalence."""
    from scipy.sparse import csr_matrix

    from tools.banc_m3_checkpointed_rollout import CheckpointedControlRollout
    from tools.banc_m3_grouped_current_adjoint import GroupedCurrent
    from tools.banc_m3_ssp_adjoint import SSPAdjointModel
    source = packet["actual_csr"]
    data = source["data"].copy()
    data[packet["recurrent_mask"]["positions"]] = arrays["selected_weights_candidate"]
    weights = csr_matrix((data, source["indices"].copy(), source["indptr"].copy()), shape=(bundle["operator"].n,) * 2)
    old = bundle["rollout"].neural
    mask = packet["recurrent_mask"]
    operator = GroupedCurrent(weights, mask["positions"], mask["edge_parameters"], len(arrays["theta_actual"]), old.device)
    neural = SSPAdjointModel(operator, *(getattr(old, k) for k in ("tau", "coefficient", "threshold", "cap", "gain", "inverse_adaptation_tau")), old.dt)
    motor_ids = bundle["rollout"].motor_ids.detach().cpu().numpy()
    return CheckpointedControlRollout(neural, bundle["rollout"].decoder, motor_ids,
                                     steps_per_control=20, control_dt=.005, checkpoint_controls=1, max_controls=4)


def expected_work(evaluated):
    require(type(evaluated) is int and 0 <= evaluated <= 5, "actual evaluated alpha count")
    # Neutral R16 + startup99; per evaluated alpha: R16+16 and startup99+99.
    controls = 115 + 230 * evaluated
    return {"forward_controls_requested": controls, "forward_controls_returned": controls,
            "forward_neural_ticks_requested": controls * 20, "forward_neural_ticks_returned": controls * 20,
            "recompute_controls_requested": 16, "recompute_controls_returned": 16,
            "recompute_neural_ticks_requested": 320, "recompute_neural_ticks_returned": 320,
            "backward_blocks_requested": 16, "backward_blocks_returned": 16, "backward_block_ticks_returned": 320}


def validate_receipts(receipts, evaluated):
    from tools import verify_c_banc_m3_reference_parameter_direction as ref
    keys = tuple(expected_work(evaluated))
    for receipt in receipts.values():
        require(all(type(receipt[k]) is int and receipt[k] >= 0 for k in keys), "typed actual receipt counts")
        require(type(receipt["confirm_requested"]) is int and type(receipt["confirm_returned"]) is int
                and receipt["confirm_requested"] == receipt["confirm_returned"] > 0, "actual returned confirmation")
        for name in ("forward_controls", "forward_neural_ticks", "recompute_controls", "recompute_neural_ticks", "backward_blocks", "backward_block_ticks"):
            require(type(receipt["confirmed"][name]) is int
                    and receipt["confirmed"][name] == receipt[name + "_returned"], "actual device-confirmed work")
    require(ref.native().work_counts(receipts) == expected_work(evaluated), "new complete actual work")
    return True


class RayFailure(RuntimeError):
    def __init__(self, cause, raw, receipts, records, stage):
        super().__init__(str(cause))
        self.raw, self.receipts, self.records, self.stage = raw, receipts, records, stage


def collect_ray(packet, startup, target_q, check, progress=lambda value: None):
    import numpy as np
    import torch

    from tools import verify_c_banc_m3_common_descent_body as b
    from tools import verify_c_banc_m3_reference_parameter_direction as ref
    core = ref.native()
    raw = {"normalization": packet["reference"]["normalization"].copy(), "target_q": target_q.copy(),
           "window_starts": np.asarray(WINDOWS, np.int64)}
    receipts, records, graphs, proxy_losses = {}, [], [], []
    stage = "assembly"
    try:
        check("science")
        ref.validate_target(target_q, raw["normalization"])
        bundle = core.assemble(packet, device="cuda")
        device = bundle["rollout"].device
        tensor = lambda a: torch.as_tensor(a.copy(), device=device)
        mask = tensor(packet["output_mask"]["value"])
        norm = tensor(np.repeat(raw["normalization"], 7))

        def run(rollout, state, drives, theta, key):
            nonlocal stage
            stage = key
            check("science")
            try:
                result = rollout.run(*(tensor(state[k]) for k in STATE_KEYS), tensor(drives), theta, mask)
            except Exception as error:
                if hasattr(error, "receipt"):
                    receipts[key] = core.receipt_dict(error.receipt)
                raise
            receipts[key] = core.receipt_dict(result["receipt"])
            progress({"stage": stage, "raw": raw, "receipts": receipts, "records": records})
            try:
                result["receipt"].confirm()
            finally:
                receipts[key] = core.receipt_dict(result["receipt"])
            check("science")
            return result

        def startup99(rollout, theta, prefix):
            state = {k: startup["control0"][k].copy() for k in STATE_KEYS}
            q, tendon = [startup["control0"]["q"].copy()], [startup["control0"]["tendon"].copy()]
            for i in range(99):
                with torch.no_grad():
                    out = run(rollout, state, startup["drive"][i:i + 1], theta, prefix + "-" + str(i + 1))
                state = {k: out[k].detach().cpu().numpy().copy() for k in STATE_KEYS}
                q.append(out["q"].detach().cpu().numpy()[0].copy())
                tendon.append(out["tendon"].detach().cpu().numpy()[0].copy())
            return np.stack(q), np.stack(tendon), state

        ones = np.ones(8547, np.float64)
        # Existing rollout holds one saved drive vector for its twenty neural ticks/control.
        require(startup["drive"].dtype == np.float32 and startup["drive"].shape == (99, bundle["operator"].n), "separate full99 drive")
        for key, value in (("utilization", 0), ("resource", 1)):
            observed = startup["control0"][key]
            require(observed.dtype == np.float64 and observed.shape == (7136,) and np.isfinite(observed).all()
                    and np.all(observed == value), "actual neutral " + key)
        q0, t0, end0 = startup99(bundle["rollout"], tensor(ones), "neutral-startup")
        for name, actual in (("q", q0[1:]), ("tendon", t0[1:]), *(end0.items())):
            expected = startup["endpoint"][name]
            if name in ("rate", "adaptation"):
                require(core.exact(actual, expected), "neutral99 exact native " + name)
            else:
                require(np.allclose(actual, expected, rtol=0, atol=1e-11 if name in ("q", "tendon") else 1e-10), "neutral99 reproduction " + name)
        baseline = b.startup99(q0, t0, **startup["target"])
        require(baseline["passed"], "neutral99 valid reproduction/startup before ray selection")
        raw.update(neutral_startup_q=q0, neutral_startup_tendon=t0)
        for index, c in enumerate(WINDOWS):
            theta = torch.ones(8547, dtype=torch.float64, device=device, requires_grad=True)
            out = run(bundle["rollout"], packet[f"start-{c}"], packet[f"drive-{c}"]["drive"], theta, "R-neutral-" + str(c))
            endpoint = packet[f"endpoint-{c}"]
            for name in ("q", "tendon", "bank", "original_activation", "rate", "adaptation"):
                actual = out[name].detach().cpu().numpy()
                require(core.exact(actual, endpoint[name]) if name in ("rate", "adaptation") else
                        np.allclose(actual, endpoint[name], rtol=0, atol=1e-11 if name in ("q", "tendon") else 1e-10), "neutral R reproduction " + name)
            terms = core.window_loss(bundle["objective"], tensor(packet[f"start-{c}"]["fixed_original_q_boundary"]), out["q"])
            losses = core.loss_vector(terms)
            require(np.allclose(losses[[2, 0, 1, 3]], endpoint["loss_terms"], rtol=1e-10, atol=1e-11), "same neutral proxy loss order/reproduction")
            proxy_losses.append(losses)
            ell = ((out["q"] - tensor(target_q[index])).square() / norm).sum() / (16 * 42)
            graphs.append((out, theta, ell, "R-neutral-" + str(c)))
            raw["q_neutral"] = np.stack([x[0]["q"].detach().cpu().numpy() for x in graphs])
            raw["R_neutral"] = np.asarray([x[2].detach().cpu().item() for x in graphs], np.float64)
            progress({"stage": stage, "raw": raw, "receipts": receipts, "records": records})
        gradient_q, gradient_theta = [], []
        for out, theta, ell, key in graphs:
            stage = key + "-independent-backward"
            before = core.receipt_dict(out["receipt"])
            (gq,) = torch.autograd.grad(ell, out["q"], retain_graph=True)
            require(before == core.receipt_dict(out["receipt"]), "gq extraction has no neural work")
            gradient_q.append(gq.detach().cpu().numpy())
            raw["gq_R"] = np.stack(gradient_q)
            require(theta.is_leaf and theta.grad is None, "independent fresh leaf per R window")
            try:
                ell.backward()
            finally:
                receipts[key] = core.receipt_dict(out["receipt"])
            require(theta.grad is not None, "independent returned R VJP")
            gradient_theta.append(theta.grad.detach().cpu().numpy())
            raw["gtheta_R"] = np.stack(gradient_theta)
            try:
                out["receipt"].confirm()
            finally:
                receipts[key] = core.receipt_dict(out["receipt"])
            out["gq"] = gq.detach().cpu().numpy()
        raw.update(q_neutral=np.stack([x[0]["q"].detach().cpu().numpy() for x in graphs]),
                   R_neutral=np.asarray([x[2].detach().cpu().item() for x in graphs], np.float64),
                   gq_R=np.stack([x[0]["gq"] for x in graphs]),
                   gtheta_R=np.stack([x[1].grad.detach().cpu().numpy() for x in graphs]),
                   neutral_proxy_loss=np.stack(proxy_losses))
        raw["sum_gtheta_R"] = raw["gtheta_R"].sum(axis=0)
        require(np.isfinite(raw["sum_gtheta_R"]).all(), "finite actual summed R gradient")
        evaluated, selected = 0, None
        if normalized_direction(raw["sum_gtheta_R"]) is not None:
            for alpha in ALPHAS:
                evidence, arrays = proposal(raw["sum_gtheta_R"], alpha, packet["actual_csr"], packet["recurrent_mask"], records)
                row = {**evidence, "eligible": False, "forward_evaluated": False}
                records.append(row)
                if arrays:
                    for k, v in arrays.items():
                        raw[f"alpha-{len(records)-1}.{k}"] = v.copy()
                if not evidence["eligible_representation"]:
                    continue
                evaluated += 1
                deployed = deployed_rollout(bundle, packet, arrays)
                qs, deployed_qs, comparisons = [], [], {k: [] for k in ("tendon", *STATE_KEYS)}
                deployed_comparisons = {k: [] for k in comparisons}
                for c in WINDOWS:
                    with torch.no_grad():
                        out = run(bundle["rollout"], packet[f"start-{c}"], packet[f"drive-{c}"]["drive"], tensor(arrays["theta_actual"]), f"alpha-{len(records)-1}-R-{c}")
                        deployed_out = run(deployed, packet[f"start-{c}"], packet[f"drive-{c}"]["drive"], tensor(ones), f"alpha-{len(records)-1}-deployed-R-{c}")
                    for k in ("q", "tendon", *STATE_KEYS):
                        require(np.allclose(out[k].detach().cpu().numpy(), deployed_out[k].detach().cpu().numpy(), rtol=1e-10, atol=1e-12), "actual deployed R forward correspondence " + k)
                    qs.append(out["q"].detach().cpu().numpy())
                    deployed_qs.append(deployed_out["q"].detach().cpu().numpy())
                    for k, comparison in comparisons.items():
                        comparison.append(out[k].detach().cpu().numpy())
                        deployed_comparisons[k].append(deployed_out[k].detach().cpu().numpy())
                qa, ta, _ = startup99(bundle["rollout"], tensor(arrays["theta_actual"]), f"alpha-{len(records)-1}-startup")
                qd, td, _ = startup99(deployed, tensor(ones), f"alpha-{len(records)-1}-deployed-startup")
                require(np.allclose(qa, qd, rtol=1e-10, atol=1e-12) and np.allclose(ta, td, rtol=1e-10, atol=1e-12), "actual deployed startup forward correspondence")
                candidate_raw = {**raw, **arrays, "q_candidate": np.stack(qs)}
                answer = analysis(candidate_raw)
                startup_result = b.startup99(qd, td, **startup["target"])
                for k, v in {"q_candidate": np.stack(qs), "q_deployed": np.stack(deployed_qs), "startup_q": qd, "startup_tendon": td}.items():
                    raw[f"alpha-{len(records)-1}.{k}"] = v
                raw[f"alpha-{len(records)-1}.startup_q_surrogate"] = qa
                raw[f"alpha-{len(records)-1}.startup_tendon_surrogate"] = ta
                for k, comparison in comparisons.items():
                    raw[f"alpha-{len(records)-1}.R_{k}_surrogate"] = np.stack(comparison)
                    raw[f"alpha-{len(records)-1}.R_{k}_deployed"] = np.stack(deployed_comparisons[k])
                row.update(forward_evaluated=True, analysis=answer, startup=startup_result,
                           deployed_forward_correspondence=True, eligible=answer["supported"] and startup_result["passed"])
                progress({"stage": stage, "raw": raw, "receipts": receipts, "records": records})
                if row["eligible"]:
                    selected = len(records) - 1
                    break
        require(validate_decision_roster(records, evaluated, raw["sum_gtheta_R"]) == selected, "fixed first eligible stop")
        validate_receipts(receipts, evaluated)
        return raw, receipts, {"schema": SCHEMA + ".ray-record.v1", "alphas": list(ALPHAS), "records": records,
                               "selected": selected, "evaluated": evaluated, "neutral_startup": baseline,
                               "work": core.work_counts(receipts), "gradient_branch": "zero" if not records else "nonzero"}
    except Exception as error:
        raise RayFailure(error, raw, receipts, records, stage) from error


class Publisher:
    def __init__(self, admission):
        self.admission = admission
        self.folder = admission.output
        self.folder.mkdir(parents=True, exist_ok=False)
        self.total = 0

    def write(self, name, raw):
        self.admission.fence()
        require("/" not in name and "\\" not in name and name not in (".", ".."), "finite artifact member")
        self.total += len(raw)
        require(self.total <= self.admission.limits["output_cap_bytes"], "durable output total cap")
        p = self.folder / name
        with p.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        require(p.read_bytes() == raw, "exact durable artifact readback")
        self.admission.fence()
        return spec_for(str(p.relative_to(self.admission.root)).replace("\\", "/"), self.admission.root, cap=self.admission.limits["output_cap_bytes"])

    def json(self, name, value):
        from tools import verify_c_banc_m3_candidate_body_handoff as body
        return self.write(name, json.dumps(body.normalize(value), ensure_ascii=True, allow_nan=False, sort_keys=True).encode("utf8"))

    def arrays(self, name, arrays):
        import numpy as np
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        return self.write(name, buffer.getvalue())


def continuous_arm(data, arm, identity, record, timer, publisher, watch):
    """One construct per arm; observer/session/counters survive the 99→100 boundary."""
    from tools.banc_m3_contact_runtime_journal import phase
    journal = getattr(timer, "runtime_journal", None)
    import numpy as np

    from tools import verify_c_banc_m3_candidate_body_handoff as body
    from tools import verify_c_banc_m3_common_descent_body as b
    from tools import verify_c_banc_m3_startup_baseline as neutral
    require(arm in ("neutral", "candidate"), "distinct initializer arm")
    session = observer = factory = None
    rows, controls, samples, construction, counters = [], [], [], {}, b.new_counters()
    result = {"schema": SCHEMA + ".body-record.v1", "arm": arm, "candidate": identity,
              "run_valid": False, "classification": "technical_invalid", "first_error": None,
              "secondary_errors": [], "startup": None, "metrics": None, "gates": None}
    stage, reason = "construct", "duration"
    try:
        factory = b.OriginalFactory()
        with phase(timer, "pre-construct", opaque=True):
            timer.fence("pre_construct")
        with phase(timer, "construct", opaque=True):
            session = factory.construct(data, construction, lambda: timer.check("science"))
        if journal is not None:
            journal.acquire_work(counters, construction)  # actual returned constructor, before next opaque START
        require(not getattr(session, "closed", False), "one live owned instance")
        with phase(timer, "initial-transfer", opaque=True):
            timer.check("science")
            if arm == "neutral":
                data.pop("selected_weights", None)
                proof = neutral.initial_neutral(session, data, timer, b=b, factory=factory)
            else:
                proof = body.initial_body(session, data, identity, record, timer, b=b, factory=factory)
        result["transfer"] = proof
        publisher.json("transfer.json", proof)
        metadata = b.trace_metadata(session)
        row, boundary = b.capture(session)
        rows.append(row); controls.append(boundary); counters["completed_rows"] = 1
        observer = b.LiveObserver(session, counters, samples, timely=lambda: timer.check("science"),
                                  install_contact=factory.install_contact, synchronize=factory.synchronize).install()
        object_identity = (id(session), id(session.network), id(session.body), id(session.adapter), id(observer))
        if journal is not None:
            journal.acquire_work(counters, construction)  # returned transfer/control0; no control advance inferred
        with phase(timer, "control-loop", opaque=True):
            for control in range(1, 301):
                stage = "control-" + str(control)
                timer.check("science")
                counters["controls_requested"] += 1
                if journal is not None:
                    journal.acquire_work(counters, construction)  # in-memory attempted count only; no extra stream write
                try:
                    session.advance(1, capture=False)
                except Exception:
                    counters["advance_unreturned_unknown"] = True
                    if journal is not None:
                        journal.acquire_work(counters, construction)
                    raise
                counters["controls_host_returned"] += 1
                if journal is not None:
                    journal.acquire_work(counters, construction)  # host return is acquired before the existing sync
                timer.check("science")
                factory.synchronize()
                counters["physics_device_confirmed"] = counters["physics_host_returned"]
                if journal is not None:
                    journal.acquire_work(counters, construction)  # existing device confirmation actually returned
                timer.check("science")
                row, boundary = b.capture(session)
                rows.append(row); controls.append(boundary)
                require(object_identity == (id(session), id(session.network), id(session.body), id(session.adapter), id(observer)), "same arm instance/observer")
                require(session.control_tick == control and session.network.tick == control * 20
                        and len(samples) == control * 50 and abs(session.body.physics_time() - control * .005) <= 2e-8
                        and session._cpg_digest() == proof["cpg_hash"], "actual feedback/work/clock/CPG continuity")
                counters["controls_completed"] += 1
                counters["completed_rows"] = len(controls); counters["observer_samples"] = len(samples)
                if journal is not None:
                    journal.acquire_work(counters, construction)  # fixed acquired scalars; no array traversal
                body.monitor_check(watch.snapshot(), timer.limits)
                if control == 99:
                    body.verify_work(counters, construction, 99)
                    prefix_native = b.native_arrays(samples)
                    require(b.validate_native(prefix_native) == 4950 and np.all(prefix_native["upright"] >= .9)
                            and not prefix_native["nonfoot_contact"].any() and not any(r["fault"] for r in rows), "healthy known prefix before control100")
                    result["startup"] = b.startup99(np.stack([x["command"] for x in controls]),
                                                    np.stack([x["tendon"] for x in controls]), **data["startup"])
                    publisher.json("prefix99.json", {"rows": len(controls), "samples": len(samples),
                                                      "counters": counters, "startup": result["startup"]})
                    if not result["startup"]["passed"]:
                        reason = "startup_rejected"
                        break
        body.verify_work(counters, construction, counters["controls_completed"], full=reason == "duration")
        require(not counters.get("advance_unreturned_unknown", False), "no unknown advance")
        result.update(run_valid=True, reason=reason, classification="scientific_rejected")
    except Exception as error:
        body.fail(result, error, stage)
    finally:
        if factory and session:
            try:
                timer.check("science")  # no new GPU confirmation after science authority expires
                factory.synchronize()
                counters["physics_device_confirmed"] = counters["physics_host_returned"]
            except Exception as error:
                body.fail(result, error, "final-device-confirmation")
        if journal is not None:
            journal.acquire_work(counters, construction)
            journal.acquire_errors(result)
        body.finish_owned(observer, session, result, timer)
        result["all_owned_ended"] = bool(session is not None and getattr(session, "closed", False))
    result.update(counters=counters, construction=construction)
    if controls:
        try:
            timer.check("science")  # array/report processing remains science deadline guarded
            control_raw, native = b.control_arrays(controls, session, data), b.native_arrays(samples)
            publisher.arrays("control.npz", control_raw)
            publisher.arrays("native.npz", native)
            publisher.json("trace.json", b.lean_trace(rows, metadata))
            publisher.arrays("evaluation.npz", {role + "." + key: value for role in ("startup", "reference") for key, value in data[role].items()})
            if result["run_valid"] and reason == "duration":
                result["metrics"] = b.evaluate(publisher.folder, control_raw, native, data, result["startup"])
                result["gates"], passed = body.screen(result["metrics"], result["startup"])
                result["classification"] = "screen_accepted_pending_audit" if passed else "scientific_rejected"
            result["cpg_unchanged"] = session._cpg_digest() == result["transfer"]["cpg_hash"]
            require(result["cpg_unchanged"], "unchanged CPG after close")
            with phase(timer, "posthash", opaque=True):
                for item in data["ledger"]:
                    timer.check("science")
                    require(b.e.digest(path(item["path"], timer.root), timer.check) == item["expected_sha256"], "actual body input posthash")
            publisher.json("selected-ledger.json", body.normalize(data["ledger"]))
        except Exception as error:
            body.fail(result, error, "partial-or-full-raw-publication")
    if watch:
        try:
            result["sampler"] = watch.close()
            body.monitor_check(result["sampler"], timer.limits, closed=True)
        except Exception as error:
            body.fail(result, error, "sampler-close")
    with phase(timer, "science-publication", opaque=True):
        timer.check("science")
        if journal is not None:
            journal.acquire_work(counters, construction)
            journal.acquire_errors(result)
        publisher.json("body-report.json", body.normalize(result))
    return result


def arm_data(raw, inventory, admission, arm, *, b):
    """Same finite body loader semantics, explicit new offline dependency binding only."""
    import numpy as np

    from tools import verify_c_banc_m3_reference_parameter_direction as ref
    from tools.banc_m3_contact_runtime_journal import phase
    core, ledger = ref.native(), []
    selected = inventory["selected_archives"]
    require(set(selected) == {"recurrent_mask", "reference", "startup", "audited_candidate"}, "finite body selected roles")
    excluded = {selected["audited_candidate"]["path"]}
    if arm == "neutral":
        excluded.add(selected["reference"]["path"])
        registration = admission.inputs["neutral_registration"]
        bound_inventory = json_load(leaf(registration["body_inventory"], admission.root))
        exact_value(bound_inventory, inventory, "new neutral inventory equals admitted body inventory")
        exact_value(registration["body_native_dependencies"], admission.inputs["body_native_dependencies"], "same explicit offline dependency")
        exact_value(registration["source_input_roster"], {k: v for k, v in inventory["files"].items() if k not in excluded}, "same explicit neutral source133")
    allowed = {k: v for k, v in inventory["files"].items() if k not in excluded}
    require(len(allowed) == (133 if arm == "neutral" else 134), "finite original body source roster")
    with phase(admission, "input-original-files", opaque=False):
        admission.fence("pre_array")
        for name, descriptor in allowed.items():
            admission.fence("pre_array")
            p = path(name, admission.root)
            require(p.stat().st_size == descriptor["bytes"] and core.digest(p, admission.check) == descriptor["sha256"], "actual admitted Original body bytes")
            if p.suffix == ".npy":
                admission.fence("pre_array")
                with p.open("rb") as stream:
                    core.strict_npy_header(stream)
            ledger.append({"path": name, "expected_sha256": descriptor["sha256"], "bytes": descriptor["bytes"]})
    with phase(admission, "input-native-assets", opaque=False):
        dependencies = json_load(leaf(admission.inputs["body_native_dependencies"], admission.root))
        require(dependencies["network_asset_download_permitted"] is False and dependencies["mesh_cache_complete_currently"] is True, "new actual finite offline native assets")
        for name, descriptor in dependencies["native_constructor_asset_bytes"].items():
            admission.fence("pre_array")
            p = Path(name)
            require(p.is_absolute() and p.is_file() and not p.is_symlink() and p.stat().st_size == descriptor["bytes"], "exact admitted native asset")
            for ancestor in (p, *p.parents):
                require(not ancestor.is_symlink() and not (getattr(ancestor.lstat(), "st_file_attributes", 0) & 0x400), "native asset no reparse ancestor")
            require(core.digest(p, admission.check) == descriptor["sha256"], "exact offline native asset digest")
    with phase(admission, "input-whole-metadata", opaque=False):
        admission.fence("pre_array")
        b.whole_metadata(admission.root, inventory)
    with phase(admission, "input-resolve-api", opaque=False):
        api = b.resolve_original_api()
    with phase(admission, "input-graph-load", opaque=False):
        admission.fence("pre_array")
        graph = api["GraphStore"].load(path(inventory["graph_bundle"], admission.root))
    with phase(admission, "input-state-load", opaque=False):
        admission.fence("pre_array")
        saved = api["StateStore"].load(path(inventory["initial_whole"], admission.root), max_files=123)
        require(graph.hash == inventory["graph_hash"], "same source graph")
        api["recovery"].checked_checkpoint(saved)
    with phase(admission, "input-selected-arrays", opaque=False):
        arrays = {}
        roles = ("recurrent_mask", "startup") if arm == "neutral" else ("recurrent_mask", "reference", "startup")
        for role in roles:
            admission.fence("pre_array")
            arrays[role] = core.load_selected(admission.root, selected[role], ledger, timely=admission.check, generated=False)
    with phase(admission, "input-result-check", opaque=False):
        result = {"graph": graph, "saved": saved, "inventory": inventory, "ledger": ledger,
                  "theta": np.ones(8547, np.float64) if arm == "neutral" else raw["theta_actual"].copy(),
                  "positions": arrays["recurrent_mask"]["positions"], "groups": arrays["recurrent_mask"]["edge_parameters"], "startup": arrays["startup"]}
        if arm != "neutral":
            result.update(selected_weights=raw["selected_weights_candidate"].copy(), reference=arrays["reference"])
        require(result["positions"].dtype == result["groups"].dtype == np.int64
                and result["positions"].shape == result["groups"].shape
                and np.array_equal(np.unique(result["groups"]), np.arange(8547)), "actual source group coverage")
        return result


def load_inputs(admission):
    """The only numeric input path; caller must own this exact pre-array admission."""
    admission.fence("pre_array")
    require(admission.stage == "pre_array", "pre-array gate before numerical imports/loads")
    import numpy as np

    from tools import verify_c_banc_m3_reference_parameter_direction as ref
    core, ledger, inputs = ref.native(), [], admission.inputs
    packet = core.load_packet(admission.root, inputs["legacy_inventory"], timely=admission.check, ledger=ledger)
    target = ref.load_label(inputs["cadence_label"], ledger, root=admission.root, timely=admission.check)
    exact_value(ledger[-1]["sha256"], inputs["cadence_label"]["sha256"], "original cadence label bytes")
    ledger[-1]["expected_sha256"] = inputs["cadence_label"]["sha256"]
    exact_value(inputs["source_identity"], inputs["body_inventory"]["source_identity"], "same neural/body control0 assets")
    inventory = inputs["body_inventory"]["inventory"]
    from tools.banc_m3_startup_ray_inputs import load_e318
    startup, resolution = load_e318(admission, core, ledger, inventory)
    startup["resolution"] = resolution
    require(packet["metadata"]["model_identity"] == inventory["source_network_identity"]
            and packet["metadata"]["decoder_hash"] == inventory["source_adapter_identity"], "actual packet/body source identities")
    require(resolution["controls"] == [1, 99] and resolution["initial_control"] == 0, "actual ordered E318 drive1..99")
    require(all(startup["control0"][k].dtype == np.int64 and startup["control0"][k].shape == (1,)
                and startup["control0"][k].item() == 0 for k in ("control_tick", "network_tick")), "audited actual source control0")
    state_shapes = {"rate": (158706,), "adaptation": (158706,), "bank": (391, 3), "original_activation": (96,), "q": (42,), "tendon": (6,)}
    for key, shape in state_shapes.items():
        value = startup["control0"][key]
        dtype = np.float32 if key in ("rate", "adaptation") else np.float64
        require(value.dtype == dtype and value.shape == shape and np.isfinite(value).all(), "finite actual neutral control0 " + key)
    for key, shape in (("target_command", (99, 42)), ("target_tendon", (99, 6))):
        value = startup["target"][key]
        require(value.dtype == np.float64 and value.shape == shape and np.isfinite(value).all(), "finite startup target shape/dtype")
    for key, shape in state_shapes.items():
        value = startup["endpoint"][key]
        expected_shape = (99, *shape) if key in ("q", "tendon") else shape
        dtype = np.float32 if key in ("rate", "adaptation") else np.float64
        require(value.dtype == dtype and value.shape == expected_shape and np.isfinite(value).all(), "finite neutral99 endpoint " + key)
    for role, actual in (("startup", startup["target"]), ("reference", packet["reference"])):
        original = core.load_selected(admission.root, inventory["selected_archives"][role], ledger,
                                      timely=admission.check, generated=False)
        require(set(actual) == set(original) and all(core.exact(actual[k], original[k]) for k in actual),
                "ray/body same original " + role + " values")
    admission.check("science")
    return packet, startup, target, ledger


def saved_arrays(descriptor, admission):
    """Independent audit is numerical and is admitted before its first archive header."""
    import numpy as np
    admission.fence("pre_array")
    require(type(descriptor) is dict and set(descriptor) == {"path", "bytes", "sha256"}
            and type(descriptor["bytes"]) is int and 0 < descriptor["bytes"] <= admission.limits["output_cap_bytes"]
            and type(descriptor["sha256"]) is str and re.fullmatch("[0-9a-f]{64}", descriptor["sha256"]), "bounded raw archive descriptor")
    p = path(descriptor["path"], admission.root)
    before = p.stat()
    require(before.st_size == descriptor["bytes"] and hashlib.sha256(p.read_bytes()).hexdigest() == descriptor["sha256"], "immutable raw archive")
    with np.load(p, allow_pickle=False) as archive:
        out = {k: archive[k].copy() for k in archive.files}
    require(p.stat().st_size == before.st_size and p.stat().st_mtime_ns == before.st_mtime_ns
            and hashlib.sha256(p.read_bytes()).hexdigest() == descriptor["sha256"], "independent raw archive posthash")
    require(all(v.dtype.kind in "bifu" and v.dtype != object for v in out.values()), "numeric-only science raw")
    return out


def audit_ray(raw, receipts, record, packet, startup):
    """Independent saved alpha/representation/math/work audit, no body-based reselection."""
    import numpy as np

    from tools import verify_c_banc_m3_common_descent_body as b
    from tools import verify_c_banc_m3_reference_parameter_direction as ref
    exact_value(record["alphas"], list(ALPHAS), "saved fixed alpha sequence")
    require(record["schema"] == SCHEMA + ".ray-record.v1", "new ray record schema")
    selected = choose_first(record["records"])
    exact_value(selected, record["selected"], "single selected identity")
    validate_decision_roster(record["records"], record["evaluated"], raw["sum_gtheta_R"])
    validate_receipts(receipts, record["evaluated"])
    exact_value(ref.native().work_counts(receipts), record["work"], "saved actual requested/returned work")
    baseline = b.startup99(raw["neutral_startup_q"], raw["neutral_startup_tendon"], **startup["target"])
    exact_value(baseline, record["neutral_startup"], "independent neutral startup")
    require(baseline["passed"], "audited neutral prerequisite")
    previous = []
    for i, row in enumerate(record["records"]):
        evidence, arrays = proposal(raw["sum_gtheta_R"], row["alpha"], packet["actual_csr"], packet["recurrent_mask"], previous)
        previous.append(evidence)
        for key in evidence:
            exact_value(row[key], evidence[key], "independent representation " + key)
        if arrays:
            for key, value in arrays.items():
                require(np.array_equal(raw[f"alpha-{i}.{key}"], value), "exact saved deployment " + key)
        require(type(row["forward_evaluated"]) is bool and row["forward_evaluated"] is evidence["eligible_representation"], "no eligible representation silently skipped")
        if row["forward_evaluated"]:
            a = {**raw, **arrays, "q_candidate": raw[f"alpha-{i}.q_candidate"]}
            exact_value(analysis(a), row["analysis"], "independent R/O/Q/T")
            require(np.allclose(a["q_candidate"], raw[f"alpha-{i}.q_deployed"], rtol=1e-10, atol=1e-12), "saved deployed correspondence")
            for key in ("tendon", *STATE_KEYS):
                require(np.allclose(raw[f"alpha-{i}.R_{key}_surrogate"], raw[f"alpha-{i}.R_{key}_deployed"], rtol=1e-10, atol=1e-12), "independent saved deployment " + key)
            require(np.allclose(raw[f"alpha-{i}.startup_q_surrogate"], raw[f"alpha-{i}.startup_q"], rtol=1e-10, atol=1e-12)
                    and np.allclose(raw[f"alpha-{i}.startup_tendon_surrogate"], raw[f"alpha-{i}.startup_tendon"], rtol=1e-10, atol=1e-12), "independent saved startup deployment")
            fixed_startup = b.startup99(raw[f"alpha-{i}.startup_q"], raw[f"alpha-{i}.startup_tendon"], **startup["target"])
            exact_value(fixed_startup, row["startup"], "independent alpha startup")
            exact_value(row["eligible"], row["analysis"]["supported"] and fixed_startup["passed"], "saved actual eligibility")
        else:
            require(row["eligible"] is False, "no eligibility without actual forward")
    return {"schema": SCHEMA + ".candidate-audit.v1", "selected": selected, "raw_valid": True,
            "F1_F8": "INCOMPLETE", "physical_body_evaluated": False}


def audit_body(report, control, native, trace, evaluated, inventory, *, output):
    import numpy as np

    from tools import verify_c_banc_m3_candidate_body_handoff as body
    from tools import verify_c_banc_m3_common_descent_body as b
    from tools import verify_c_banc_m3_startup_baseline as neutral
    count = report["counters"]["controls_completed"]
    require(report["run_valid"] is True and report["all_owned_ended"] is True
            and report["first_error"] is None and not report["secondary_errors"], "closed valid arm before independent audit")
    require((count == 99 and report["reason"] == "startup_rejected") or (count == 300 and report["reason"] == "duration"), "explicit prefix/full lifecycle")
    body.verify_work(report["counters"], report["construction"], count, full=count == 300)
    require(not report["counters"].get("advance_unreturned_unknown", False), "no unknown body work")
    require(b.validate_native(native) == count * 50 and np.array_equal(control["control_tick"], np.arange(count + 1))
            and np.array_equal(control["network_tick"], np.arange(count + 1) * 20)
            and np.allclose(control["physics_time_s"], np.arange(count + 1) * .005, rtol=0, atol=2e-8), "independent complete clock/work rows")
    proof = report["transfer"]
    b.verify_chunk_coverage(proof)
    if report["arm"] == "neutral":
        neutral.validate_neutral_proof(proof, inventory, b=b)
    else:
        require(proof["candidate_CSR_data_sha256"] != proof["source_CSR_data_sha256"]
                and proof["selected_sha256"] == report["candidate"]["selected_sha256"]
                and proof["theta_sha256"] == report["candidate"]["theta_sha256"], "distinct admitted candidate transfer")
    require(array_digest(control["theta"]) == proof["theta_sha256"]
            and array_digest(control["selected_weights"]) == proof["selected_sha256"], "saved actual deployment identity")
    require(len(trace["samples"]) == count + 1 and trace["model_hash"] == proof["candidate_network_identity"]
            and trace["graph_hash"] == proof["graph_hash"] and trace["adapter_hash"] == proof["source_adapter_identity"], "same audited trace identities")
    for i, row in enumerate(trace["samples"]):
        b.lean_row(row, point_count=8)
        require(row["control_tick"] == i and abs(row["time_s"] - .005 * i) <= 2e-8 and not row["fault"], "independent trace health")
    data = {role: {k.split(".", 1)[1]: v for k, v in evaluated.items() if k.startswith(role + ".")}
            for role in ("startup", "reference")}
    startup = b.startup99(control["command"][:100], control["tendon"][:100], **data["startup"])
    exact_value(startup, report["startup"], "actual saved startup99")
    require(report["cpg_unchanged"] is True, "closed unchanged CPG")
    metrics, gates, passed = None, None, False
    if count == 300:
        metrics = b.evaluate(output, control, native, data, startup)
        gates, passed = body.screen(metrics, startup)
        exact_value(body.normalize(metrics), report["metrics"], "independent physical metrics")
        exact_value(gates, report["gates"], "unchanged short screen")
    else:
        require(not startup["passed"], "honest prefix rejection")
    return {"schema": SCHEMA + ".body-audit.v1", "arm": report["arm"], "startup": startup,
            "metrics": metrics, "gates": gates, "screen_accepted": passed, "F1_F8": "INCOMPLETE"}


def _worker(admission, publisher, watch):
    require(admission.stage == "pre_array", "same genuine worker pre-array")
    role, q = admission.role, admission.request
    try:
        if role in ("input_audit", "ray", "candidate_audit"):
            packet, startup, target, ledger = load_inputs(admission)
            if role == "input_audit":
                # Full neutral reproduction is performed again inside collect_ray before any selection.
                result = {"schema": SCHEMA + ".input-audit.v1", "input_roster": ledger,
                          "source_identity": admission.inputs["source_identity"], "arrays_loaded_and_checked": True,
                          "E318_resolution": startup["resolution"],
                          "neutral99_reproduction": "required_by_ray", "body_run": False}
                publisher.json("input-audit.json", result)
                return result
            if role == "ray":
                raw, receipts, record = collect_ray(packet, startup, target, admission.check)
                from tools import verify_c_banc_m3_reference_parameter_direction as ref
                for item in ledger:
                    require(ref.native().digest(path(item["path"], admission.root), admission.check) == item["expected_sha256"], "ray input post-calculation hash")
                raw_spec = publisher.arrays("ray-raw.npz", raw)
                receipt_spec = publisher.json("receipts.json", receipts)
                record_spec = publisher.json("ray-record.json", record)
                ledger_spec = publisher.json("selected-ledger.json", ledger)
                selected = record["selected"]
                candidate = None
                if selected is not None:
                    arrays = {key: raw[f"alpha-{selected}.{key}"] for key in ("theta_requested", "theta_actual", "selected_weights_candidate")}
                    selected_spec = publisher.arrays("selected-candidate.npz", arrays)
                    candidate = {"schema": SCHEMA + ".selected-candidate.v1", "alpha": record["records"][selected]["alpha"],
                                 "index": selected, "arrays": selected_spec, "theta_sha256": array_digest(arrays["theta_actual"]),
                                 "selected_sha256": array_digest(arrays["selected_weights_candidate"]),
                                 "palette": record["records"][selected]["palette"], "ray_raw": raw_spec,
                                 "ray_record": record_spec, "manifest": admission.manifest_spec, "grant": q["grant"]}
                    publisher.json("selected-candidate.json", candidate)
                return publisher.json("ray-manifest.json", {"schema": SCHEMA + ".ray-manifest.v1", "raw": raw_spec,
                                                           "receipts": receipt_spec, "record": record_spec, "ledger": ledger_spec,
                                                           "selected_candidate": candidate})
            manifest = json_load(leaf(q["artifacts"]["ray_manifest"], admission.root))
            raw = saved_arrays(manifest["raw"], admission)
            record = json_load(leaf(manifest["record"], admission.root))
            receipts = json_load(leaf(manifest["receipts"], admission.root, cap=admission.limits["output_cap_bytes"]))
            from tools import verify_c_banc_m3_candidate_body_handoff as body
            exact_value(json_load(leaf(manifest["ledger"], admission.root)), body.normalize(ledger), "independent original input read roster")
            result = audit_ray(raw, receipts, record, packet, startup)
            result.update(selected_candidate=manifest["selected_candidate"], ray_manifest=q["artifacts"]["ray_manifest"],
                          manifest=admission.manifest_spec, grant=q["grant"])
            publisher.json("candidate-audit.json", result)
            return result
        if role in ("neutral_body", "candidate_body"):
            from tools import verify_c_banc_m3_common_descent_body as b
            arm = "neutral" if role == "neutral_body" else "candidate"
            selected = json_load(leaf(q["artifacts"]["selected_candidate"], admission.root))
            audit = json_load(leaf(q["artifacts"]["candidate_audit"], admission.root))
            require(audit["raw_valid"] is True and audit["selected"] == selected["index"], "one immutable audited physical candidate")
            exact_value(audit["selected_candidate"], selected, "actual audited immutable candidate")
            exact_value(selected["manifest"], admission.manifest_spec, "candidate same science manifest")
            exact_value(selected["grant"], q["grant"], "candidate same scoped grant")
            arrays = saved_arrays(selected["arrays"], admission)
            require(array_digest(arrays["theta_actual"]) == selected["theta_sha256"]
                    and array_digest(arrays["selected_weights_candidate"]) == selected["selected_sha256"], "actual selected handoff identity")
            inventory = admission.inputs["body_inventory"]["inventory"]
            if arm == "neutral":
                data = arm_data(arrays, inventory, admission, arm, b=b)
                from tools import verify_c_banc_m3_reference_parameter_direction as ref
                admission.fence("pre_array")
                data["reference"] = ref.native().load_selected(admission.root, inventory["selected_archives"]["reference"], data["ledger"], timely=admission.check, generated=False)
            else:
                data = arm_data(arrays, inventory, admission, arm, b=b)
            return continuous_arm(data, arm, selected, {"palette": selected["palette"]}, admission, publisher, watch)
        if role in ("neutral_audit", "candidate_body_audit"):
            report = json_load(leaf(q["artifacts"]["body_report"], admission.root))
            control = saved_arrays(q["artifacts"]["control"], admission)
            native = saved_arrays(q["artifacts"]["native"], admission)
            evaluated = saved_arrays(q["artifacts"]["evaluation"], admission)
            trace = json_load(leaf(q["artifacts"]["trace"], admission.root, cap=admission.limits["output_cap_bytes"]))
            packet, startup, _target, _ledger = load_inputs(admission)
            from tools import verify_c_banc_m3_reference_parameter_direction as ref
            for role_name, actual in (("startup", startup["target"]), ("reference", packet["reference"])):
                require(set(actual) == {k.split(".", 1)[1] for k in evaluated if k.startswith(role_name + ".")}
                        and all(ref.native().exact(actual[k], evaluated[role_name + "." + k]) for k in actual), "independent actual original evaluation inputs")
            result = audit_body(report, control, native, trace, evaluated, admission.inputs["body_inventory"]["inventory"],
                                output=path(q["artifacts"]["trace"]["path"], admission.root).parent)
            publisher.json("body-audit.json", result)
            return result
        require(role == "close", "finite final close role")
        result = {"schema": SCHEMA + ".science-close.v1", "grant": q["grant"], "manifest": admission.manifest_spec,
                  "external_final_return_and_once_accounting": "ROOT_required", "F1_F8": "INCOMPLETE"}
        publisher.json("science-close.json", result)
        return result
    except Exception as error:
        partial = {"schema": SCHEMA + ".partial-failure.v1", "role": role, "error_type": type(error).__name__,
                   "error": str(error), "classification": "technical_invalid", "identical_retry_allowed": False}
        if isinstance(error, RayFailure):
            partial.update(stage=error.stage, receipts=error.receipts, records=error.records)
            try:
                publisher.arrays("partial-ray-raw.npz", error.raw)
            except Exception as secondary:
                partial["partial_raw_publication_error"] = str(secondary)
        try:
            publisher.json("failure.json", partial)
        except Exception as publication_error:
            error.add_note("failure publication: " + str(publication_error))
        raise


def worker(admission):
    # NVML baseline precedes the first native/numerical import for every numerical role.
    from tools import verify_c_banc_m3_candidate_body_handoff as body
    publisher = Publisher(admission)
    watch = None
    first = None
    try:
        if admission.role != "close":
            watch = body.gpu_watch({"limits": admission.limits})
        result = _worker(admission, publisher, watch)
        require(not isinstance(result, dict) or result.get("run_valid") is not False,
                "technical-invalid body cannot return a successful role exit")
        return body.normalize(result)
    except Exception as error:
        first = error
        raise
    finally:
        if watch:
            try:
                if watch.snapshot()["state"] != "CLOSED":
                    closed = watch.close()
                    body.monitor_check(closed, admission.limits, closed=True)
                    publisher.json("sampler.json", closed)
            except Exception as close_error:
                if first:
                    first.add_note("sampler close: " + str(close_error))
                else:
                    raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--pre-spawn", action="store_true")
    args = parser.parse_args(argv)
    entry = ENTRY
    if not args.pre_spawn:
        owner = json_load(os.environ["FLYLAB_DIRECT_SCIENCE_OWNER"].encode("utf8"))
        entry = finite_float(owner["entry_counter_s"], "actual supervisor ENTRY", minimum=1e-12)
    admission = Admission(args.request, "pre_spawn" if args.pre_spawn else "pre_array", entry=entry)
    if args.pre_spawn:
        print(json.dumps({"schema": SCHEMA + ".pre-spawn.v1", "request": admission.request_spec,
                          "manifest": admission.manifest_spec, "grant": admission.request["grant"]}))
        return 0
    result = worker(admission)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
