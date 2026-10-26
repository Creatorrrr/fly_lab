"""Finite K accounting checkpoint. Metadata only; current Source is a cold fence.

Accounting reconstructs every permitted field from the literal closed288 anchor.
It does not invoke the historical epoch proof graph or import scientific workers.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
import stat
import time
from pathlib import Path

from tools import banc_m3_conserved_resource as r

BASE = r.BASE
KEY = "direct_science_checkpoint_recovery"
ALLOCATION_KEY = "direct_science_checkpoint_allocation"
QUALIFICATION_KEY = "direct_science_checkpoint_qualification"
EPOCH = "contact-checkpoint-20261008-k"
ANCHOR = {"path": BASE + "/parent-j-lint-settled288-HOLD-20261008.json", "bytes": 460487,
          "sha256": "ba0cb692cae00e15a7693c8ae01279dc9d8b0af771380ad1af7c363051c9b55a"}
BEFORE_MAP = {"path": BASE + "/parent-k-before-20261008/source-before.json", "bytes": 15761,
              "sha256": "26666fb0be28e4b0efc68eda07c6764e63e8e339fe24c903fd81c0a9ec6b5a58"}
J_FREEZE = {"path": BASE + "/parent-contact-learning-source-freeze-20261008-j.json", "bytes": 255412,
            "sha256": "d2ea5057326a9a916dcfe59c3b9cbbdb5b7b61fc852bf23b5d314fb1269f1baa"}
J_BODY = {"path": BASE + "/parent-contact-body-runtime-20261008-j.json", "bytes": 15588,
          "sha256": "b49da9b777e6c640340c48c24225664b1987f70c9efc8e377d737b1357987e34"}
J_FAILED = {"path": BASE + "/parent-direct-science-20261008-contact-j-lint-full-return.json", "bytes": 1285,
            "sha256": "490fc81e9cf588ba742d9065844b3bcc711e2955f74a4171879ed561fd2f7742"}
J_TRACE = {"path": BASE + "/parent-direct-science-20261008-contact-j-lint-genuine-return.json", "bytes": 3473,
           "sha256": "ff5cd48349c0d6be0cac0e69a20682f5c6c5d23ce9d4c0c3569fd976044b422d"}
J_OUTER = {"path": BASE + "/parent-direct-science-20261008-contact-j-lint/outer.json", "bytes": 1493,
           "sha256": "5b4bd75dee8e7d3171d3fe517bffa4f39e8bdf2f99ab2c368f23231227b0df11"}
OLD_CAPS = dict(r.NEW_CAPS)
CAPS = {"generated_global": 2700, "generated_local": 1964, "original_global": 15200}
RECOVERY_PATH = BASE + "/parent-contact-checkpoint-recovery-20261008-k.json"
ALLOCATION_PATH = BASE + "/parent-contact-checkpoint-allocation-20261008-k.json"
SOURCE_PATH = BASE + "/parent-contact-learning-source-freeze-20261008-k.json"
REVIEW_PATH = BASE + "/parent-contact-checkpoint-reviewed-source-20261008-k.json"
CONTRACT_PATH = BASE + "/parent-contact-checkpoint-contract-20261008-k.json"
BODY_PATH = BASE + "/parent-contact-body-runtime-20261008-k.json"
ROSTER_PATH = BASE + "/parent-contact-learning-qualification-roster-20261008-k.json"
QUALIFICATION_PATH = BASE + "/parent-contact-learning-qualified-20261008-k.json"
BOOTSTRAP_CALL = "direct-science-20261008-contact-k-bootstrap"
QUAL_CALLS = tuple("direct-science-20261008-contact-k-" + x for x in ("lint", "math", "gate-a", "gate-b"))
CALLS = ("contact-learning-20261008-k-body", "contact-learning-20261008-k-body-audit")
SEQUENCE = (BOOTSTRAP_CALL, *QUAL_CALLS, *CALLS)
CAP_TRIPLES = ((90, 75, 15), (40, 35, 5), (35, 30, 5), (120, 110, 10), (360, 340, 20), (300, 285, 15), (120, 100, 20))
BOOTSTRAP_NAMES = (
    "test_gate_k_checkpoint_production_python_native_parity",
    "test_gate_k_checkpoint_exact_recovery_allocation_and_negatives",
    "test_gate_k_checkpoint_actual_atomic_foreign_lock_and_stale_raw",
    "test_gate_k_checkpoint_fee_source_drift_overrun_unknown_duplicate",
)
TEST_MODULE = "tests/test_c_banc_checkpoint_runtime.py"
MATH_NAMES = ("test_math_k_checkpoint_ordered_accounting_negative_remaining",
              "test_math_k_checkpoint_exact_prefix_types_and_legal_states")
REPLACEMENTS = (
    r.HELPER_SOURCE, BASE + "/parent-direct-science-account.ps1", "tools/banc_m3_startup_ray_supervisor.py",
    "tools/banc_m3_contact_learning_admission.py", BASE + "/parent-contact-learning-contract-native.ps1",
    "tools/banc_m3_contact_runtime_journal.py", "tools/verify_c_banc_m3_contact_learning.py",
    "tests/test_c_banc_contact_runtime_journal.py",
)
ADDITIONS = (
    "tools/banc_m3_checkpoint_runtime.py", "tools/banc_m3_checkpoint_runtime_native.ps1",
    "tools/banc_m3_checkpoint_legacy_epoch_native.ps1", TEST_MODULE,
    "tests/banc_m3_checkpoint_fixture.py", CONTRACT_PATH, BODY_PATH, ROSTER_PATH,
)
HOLD = "Actual closure or resource cap unresolved"
FAILURE_HOLD = 'K stage unsuccessful; successor blocked'


def keys(value, expected, label):
    r.need(type(value) is dict and set(value) == set(expected), label + " exact ordinal keys")


def integer(value, label, positive=False):
    r.need(type(value) is int and (not positive or value > 0), label + " integer, never bool/float")
    return value


def number(value, label, nonnegative=True):
    r.need(type(value) in (int, float) and math.isfinite(value) and (not nonnegative or value >= 0), label)
    return float(value)


def data(spec, root, cap=262144):
    lock = type(spec) is dict and spec.get('path') == r.SHARED_LOCK
    r.need(type(spec) is dict and type(spec.get('path')) is str
           and (Path(spec['path']).suffix == '.json' or lock),
           'metadata JSON or exact existing ownership token before any operand read')
    return r.read_pin(spec, root, min(cap, 16384) if lock else cap)


def metadata_pin(spec, path):
    keys(spec, ("path", "bytes", "sha256"), "pin")
    r.exact(spec["path"], path, "literal metadata path")
    r.need(0 < integer(spec["bytes"], "pin bytes") <= 524288
           and type(spec["sha256"]) is str and len(spec["sha256"]) == 64
           and all(c in "0123456789abcdef" for c in spec["sha256"]), "typed bounded pin")


def before_archive(spec, root):
    """Exact source/archive pair only; never execution authority."""
    mapping = data(BEFORE_MAP, root)
    for row in mapping["snapshots"]:
        if row["source"] == spec:
            r.exact(row["archive"]["bytes"], spec["bytes"], "whole archived bytes")
            r.exact(row["archive"]["sha256"], spec["sha256"], "whole archived SHA")
            return r.raw_pin(row["archive"], root, 524288)
    raise ValueError("no literal K-before exact pair")


def totals(ledger, policy, grant, caps):
    generated = original = 0.0
    seen = set()
    for index, debit in enumerate(ledger["debits"]):
        r.need(type(debit["cost_id"]) is str and debit["cost_id"] not in seen, "one debit per identity")
        seen.add(debit["cost_id"])
        r.need(debit["mode"] in ("generated", "original"), "finite debit mode")
        fee = number(debit["actual_outer_full_return_s"], "finite actual fee")
        if index >= 118 and debit["mode"] == "generated":
            generated += fee
        if index >= 165 and debit["mode"] == "original":
            original += fee
    stage = float(policy["generated_stage_before_s"]) + generated
    result = {"direct_science_generated_used_s": generated, "generated_stage_used_s": stage,
              "generated_shared_cumulative_used_s": float(policy["generated_shared_before_s"]) + generated,
              "generated_shared_remaining_s": float(caps["generated_global"]) - float(policy["generated_shared_baseline_s"]) - stage,
              "original_cumulative_used_s": float(grant["original_cumulative_before_s"]) + original,
              "original_stage_used_s": float(grant["original_stage_before_s"]) + original,
              "startup_ray_original_local_used_s": float(grant["science_local_used_before_s"]) + original,
              "original_remaining_s": float(caps["original_global"]) - float(grant["original_cumulative_before_s"]) - original}
    for value in result.values():
        number(value, "finite accounting including incurred negative remaining", nonnegative=False)
    return result


def anchor(root):
    closed = data(ANCHOR, root, 524288)
    r.need(len(closed["debits"]) == 288 and len(closed["direct_science_calls"]) == 170, "literal288/170")
    policy, grant = data(r.POLICY_PIN, root), data(r.J_PIN, root)
    r.exact(closed["direct_science_policy"], r.POLICY_PIN, "original policy pin")
    for k, v in totals(closed, policy, grant, OLD_CAPS).items():
        r.exact(closed[k], v, "ordered anchor " + k)
    return closed, policy, grant


def recovery(spec, root):
    metadata_pin(spec, RECOVERY_PATH)
    event = data(spec, root)
    keys(event, ("schema", "epoch", "before_ledger", "source_freeze", "failed_j", "released_hold",
                 "reason", "new_resource_grant_s", "refund_s"), "recovery")
    expected = {"schema": "flylab.m3.checkpoint-recovery.v1", "epoch": EPOCH, "before_ledger": ANCHOR,
                "source_freeze": event["source_freeze"], "failed_j": {"cost": J_FAILED, "trace": J_TRACE, "outer": J_OUTER},
                "released_hold": "direct-science-20261008-contact-j-lint",
                "reason": "KNOWN_CLOSED_ONCE_PAID_J_PER_CALL_OVERRUN_RETAINED",
                "new_resource_grant_s": 0, "refund_s": 0}
    r.exact(event, expected, "entire exact recovery event")
    metadata_pin(event["source_freeze"], SOURCE_PATH)
    cost, trace, outer = data(J_FAILED, root), data(J_TRACE, root), data(J_OUTER, root)
    r.exact(cost["actual_outer_full_return_s"], 48.607, "J fee remains48.607")
    r.exact(cost["exit_code"], 124, "J failure remains124")
    for k, v in (("cap_exceeded", True), ("charged_once", True), ("all_owned_ended", True), ("reservation_released", True)):
        r.exact(cost[k], v, "literal J known closure")
    r.exact(trace["full_return_s"], 48.607, "J genuine fee")
    for k, v in (("stdout_eof", True), ("stderr_eof", True), ("unknown_owned_identities", {})):
        r.exact(outer[k], v, "J actual closure")
    closed, policy, grant = anchor(root)
    r.exact(closed["direct_science_blocking_holds"], {expected["released_hold"]: HOLD}, "sole exact J HOLD")
    return event, closed, policy, grant


def recovered_snapshot(spec, root):
    _event, closed, _p, _g = recovery(spec, root)
    result = copy.deepcopy(closed)
    result[KEY] = spec
    result["direct_science_blocking_holds"] = {}
    return result


def roster(freeze, root):
    old = r.complete_roster(root)
    spec = freeze["sources"][ROSTER_PATH]
    new = data(spec, root)
    keys(new, (*old, "bootstrap"), "full K roster fields")
    r.exact(new['schema'], 'flylab.m3.contact-learning.qualification-roster.v3', 'distinct K roster schema')
    r.exact(new['epoch'], EPOCH, 'distinct K roster epoch')
    for module in new['indivisible_groups']:
        nodes = {row['node_id'] for row in new['gate'] if row['module'] == module}
        r.need(nodes <= {row['node_id'] for row in new['gate_a']} or nodes <= {row['node_id'] for row in new['gate_b']}, 'indivisible Native groups retained')
    for kind in ("math", "gate_a", "gate_b", "gate"):
        rows = new[kind]
        r.need(rows[:len(old[kind])] == old[kind], "entire old roster/parameter identities " + kind)
        r.need(len({row["node_id"] for row in rows}) == len(rows), "no duplicate roster")
        for row in rows:
            r.exact(row["node_id"], row["module"] + "::" + row["name"], "literal node")
            r.exact(row["classname"], row["module"][:-3].replace("/", "."), "literal classname")
    a, b, whole = ({row["node_id"] for row in new[k]} for k in ("gate_a", "gate_b", "gate"))
    r.need(not a & b and a | b == whole, "disjoint full gate union including indivisible old groups")
    r.exact([row["name"] for row in new["bootstrap"]], list(BOOTSTRAP_NAMES), "fixed bootstrap names")
    return new


def plan(call, freeze, root):
    r.need(call in SEQUENCE, "only seven literal K calls")
    i = SEQUENCE.index(call)
    limits = dict(zip(("outer_s", "child_s", "cleanup_s"), CAP_TRIPLES[i], strict=True))
    limits.update(host_bytes=2147483648 if i < 5 else 8589934592, output_cap_bytes=8388608 if i < 5 else 104857600)
    roles = ("execution", "lint", "numerical", "execution", "execution", "contact_candidate_body", "contact_body_audit")
    result = {"call": call, "role": roles[i], "limits": limits,
              "mode": "generated" if i < 5 else "original",
              "output": BASE + "/parent-" + call + ("" if i < 5 else "/result-supervision")}
    if i < 5:
        if i == 1:
            modules = r.generated_plan("lint", root)["argv"][3:]
            result["argv"] = ["-m", "ruff", "check", *modules, "tools/banc_m3_checkpoint_runtime.py", TEST_MODULE,
                              "tests/banc_m3_checkpoint_fixture.py"]
        else:
            kind = ("bootstrap", "lint", "math", "gate_a", "gate_b")[i]
            rows = roster(freeze, root)[kind]
            suffix = ("bootstrap", "lint", "math", "gate-a", "gate-b")[i]
            result["argv"] = ["-m", "pytest", *[row["node_id"] for row in rows], "--tb=short", "-q",
                              "--junitxml=" + BASE + "/parent-contact-learning-k-" + suffix + "-junit.xml"]
    else:
        contract = data(freeze["contract"], root)
        result["limits"] = {k: contract["limits"][roles[i]][k] for k in limits}
        r.exact(tuple(result["limits"][k] for k in ("outer_s", "child_s", "cleanup_s")), CAP_TRIPLES[i], "unchanged Original envelope")
    return result


def identity(q, request_spec, source_spec, root):
    metadata_pin(source_spec, SOURCE_PATH)
    freeze = data(source_spec, root)
    p = plan(q["call_id"], freeze, root)
    expected_keys = {"schema", "call_id", "mode", "role", "source_freeze", "limits", "output", "ledger_path", "shared_original_lock_path"}
    expected_keys |= {"generated_behavior", "generated_argv"} if p["mode"] == "generated" else {"grant", "science_request"}
    keys(q, expected_keys, "whole K supervision request")
    for k, v in (("schema", "flylab.m3.direct-science.supervision.v1"), ("source_freeze", source_spec),
                 ("call_id", p["call"]), ("mode", p["mode"]), ("role", p["role"]), ("limits", p["limits"]),
                 ("output", p["output"]), ("ledger_path", BASE + "/parent-e358-execution-ledger.json"),
                 ("shared_original_lock_path", r.SHARED_LOCK)):
        r.exact(q[k], v, "exact K request " + k)
    if q["mode"] == "generated":
        r.exact(q["generated_behavior"], "command", "one generated command")
        r.exact(q["generated_argv"], p["argv"], "whole exact K argv")
    else:
        r.exact(q["grant"], r.J_PIN, "unchanged J grant")
        science = data(q["science_request"], root)
        original_metadata(science, q, root)
        for k in ("call_id", "role", "grant", "source_freeze", "limits"):
            value = {x: science["limits"][x] for x in q["limits"]} if k == "limits" else science[k]
            r.exact(q[k], value, "same K science/outer " + k)
    r.exact(data(request_spec, root), q, "intact admitted request bytes")
    return freeze


def closure(outer):
    for key in ("worker_started", "creation_attempted", "job_created", "resume_attempted", "all_owned_ended", "reservation_released", "stdout_eof", "stderr_eof"):
        r.need(type(outer[key]) is bool, "mandatory typed " + key)
    state = outer["worker_creation_state"]
    r.need(state in ("NO_CREATE_ATTEMPT", "CREATED_SUSPENDED_NOT_RESUMED", "RESUMED", "UNCERTAIN_CREATION"), "finite creation state")
    r.need(outer["resume_outcome"] in ("NOT_ATTEMPTED", "RESUMED", "UNCERTAIN"), "finite resume outcome")
    keys(outer["unknown_owned_identities"], outer["unknown_owned_identities"], "unknown identities")
    r.need(len(outer['unknown_owned_identities']) <= 1024, 'bounded owned identities')
    for pid, ticks in outer['unknown_owned_identities'].items():
        r.need(type(pid) is str and pid.isascii() and pid.isdecimal() and int(pid) > 0 and str(int(pid)) == pid, 'typed unknown PID')
        integer(ticks, 'unknown creation ticks', positive=True)
    for k in ('owner_pid', 'owner_creation_ticks'):
        integer(outer[k], 'actual owner ' + k, positive=True)
    r.exact(outer['reservation_released'], outer['all_owned_ended'], 'release only with known whole closure')
    if state == "NO_CREATE_ATTEMPT":
        for k in ("worker_started", "creation_attempted", "resume_attempted", "stdout_eof", "stderr_eof"):
            r.exact(outer[k], False, "lexical no-create never observed EOF")
        for k in ("worker_pid", "worker_creation_ticks", "worker_exit_code"):
            r.exact(outer[k], None, "no synthetic identity/exit")
        r.exact(outer["resume_outcome"], "NOT_ATTEMPTED", "no synthetic resume")
        r.exact(outer['unknown_owned_identities'], {}, 'no invented no-create identities')
        return outer["all_owned_ended"] and outer["reservation_released"] and not outer["unknown_owned_identities"]
    r.exact(outer["creation_attempted"], True, "actual creation attempted")
    if state == "UNCERTAIN_CREATION":
        r.exact(outer["all_owned_ended"], False, "uncertain creation retains liability")
        r.exact(outer['reservation_released'], False, 'unknown reservation retained')
        r.exact(outer['job_created'], True, 'uncertainty belongs to actual owned Job')
        r.exact(outer['worker_started'], False, 'resume never confirmed')
        r.exact(outer['resume_outcome'], 'UNCERTAIN' if outer['resume_attempted'] else 'NOT_ATTEMPTED', 'finite uncertainty phase')
        if outer['resume_attempted']:
            for k in ('worker_pid', 'worker_creation_ticks'):
                integer(outer[k], 'known suspended identity ' + k, positive=True)
            r.exact(outer['unknown_owned_identities'].get(str(outer['worker_pid'])), outer['worker_creation_ticks'], 'retained suspended identity')
            r.need(outer['worker_pid'] != outer['owner_pid'], 'distinct observed suspended worker/owner')
        else:
            for k in ('worker_pid', 'worker_creation_ticks', 'worker_exit_code'):
                r.exact(outer[k], None, 'unconfirmed creation has no invented identity/exit')
            r.exact(outer['unknown_owned_identities'], {}, 'no observed identity before successful CreateProcess')
            for k in ('stdout_eof', 'stderr_eof'):
                r.exact(outer[k], False, 'no installed readers before confirmed creation')
        if outer['worker_exit_code'] is not None:
            integer(outer['worker_exit_code'], 'observed uncertain-resume worker exit')
        return False
    for k in ("worker_pid", "worker_creation_ticks", "owner_pid", "owner_creation_ticks"):
        integer(outer[k], "actual process " + k, positive=True)
    r.exact(outer["job_created"], True, "owned actual Job")
    r.need(outer['worker_pid'] != outer['owner_pid'], 'distinct worker/owner')
    if outer['worker_exit_code'] is not None:
        integer(outer['worker_exit_code'], 'actual worker exit')
    r.exact(outer['resume_attempted'], state == 'RESUMED', 'actual finite resume attempt')
    r.exact(outer["worker_started"], state == "RESUMED", "worker_started means resumed")
    r.exact(outer["resume_outcome"], "RESUMED" if state == "RESUMED" else "NOT_ATTEMPTED", "actual resume outcome")
    if outer['all_owned_ended']:
        integer(outer['worker_exit_code'], 'known closed worker exit')
        r.exact(outer['unknown_owned_identities'], {}, 'closed means no unknown identities')
        for k in ('stdout_eof', 'stderr_eof'):
            r.exact(outer[k], True, 'known closure requires actual two EOF')
    else:
        r.exact(outer['unknown_owned_identities'].get(str(outer['worker_pid'])), outer['worker_creation_ticks'], 'retain every unclosed worker identity')
    return (outer["all_owned_ended"] and outer["reservation_released"] and not outer["unknown_owned_identities"]
            and outer["stdout_eof"] and outer["stderr_eof"])


def fee_proof(cost_spec, debit, source_spec, root):
    """Intact immutable evidence, never current executable Source validation."""
    metadata_pin(cost_spec, BASE + "/parent-" + debit["cost_id"] + "-full-return.json")
    cost = data(cost_spec, root)
    request_spec = cost["request"] if debit["mode"] == "generated" else cost["supervisor_request"]
    q = data(request_spec, root)
    identity(q, request_spec, source_spec, root)
    trace, outer = data(cost["trace"], root, 2097152), data(cost["outer"], root, 1048576)
    for k in ("request", "source_freeze"):
        r.exact(trace[k], request_spec if k == "request" else source_spec, "genuine K trace binding")
    r.exact(outer["request"], request_spec, "immutable outer binding")
    r.exact(trace["schema"], "flylab.m3.direct-science.genuine-return.v1", "actual collector")
    r.exact(trace["phase"], "FINAL", "genuine first FINAL")
    r.exact(trace["first_error"], None, "no collector uncertainty")
    integer(trace["dispatch_ms"], "genuine dispatch", positive=True)
    integer(trace["first_final_ms"], "genuine first FINAL", positive=True)
    fee = (trace["first_final_ms"] - trace["dispatch_ms"]) / 1000.0
    r.need(number(fee, "positive exact fee") > 0, "positive elapsed")
    r.exact(trace["full_return_s"], fee, "genuine full return")
    for k, v in (("cost_id", q["call_id"]), ("mode", q["mode"]), ("source_freeze", source_spec),
                 ("actual_outer_full_return_s", fee), ("exit_code", trace["exit_code"]), ("charged_once", True),
                 ("cap_exceeded", fee > q["limits"]["outer_s"])):
        r.exact(cost[k], v, "once cost " + k)
    r.exact(debit, {"cost_id": q["call_id"], "mode": q["mode"], "actual_outer_full_return_s": fee,
                    "actual_returned_exit_code": trace["exit_code"], "descriptor": cost_spec}, "entire debit")
    integer(trace['exit_code'], 'genuine exit code')
    integer(outer['exit_code'], 'actual exit code')
    shell_pair = (trace['exit_code'] == 1 and outer['exit_code'] == 124 and outer['worker_exit_code'] == 124
                  and outer.get('timeout') is True and (outer.get('first_error') or {}).get('type') == 'TimeoutError'
                  and outer['all_owned_ended'] is True and outer['reservation_released'] is True
                  and outer['stdout_eof'] is True and outer['stderr_eof'] is True and not outer['unknown_owned_identities']
                  and trace.get('exec_args', {}).get('cmd', '').endswith('& .venv/Scripts/python.exe -m tools.banc_m3_startup_ray_supervisor --request ' + request_spec['path']))
    r.need(outer['exit_code'] == trace['exit_code'] or shell_pair, 'actual exit or proven existing shell deadline pair')
    base_keys = {'schema', 'cost_id', 'mode', 'request', 'source_freeze', 'trace', 'outer', 'actual_outer_full_return_s',
                 'exit_code', 'charged_once', 'all_owned_ended', 'reservation_released', 'cap_exceeded', 'worker_started',
                 'creation_attempted', 'worker_creation_state', 'resume_attempted', 'resume_outcome'}
    if q['mode'] == 'original':
        base_keys |= {'supervisor_request', 'role', 'grant', 'inputs', 'roster', 'contract', 'body_retry', 'local_debit_offset', 'resource_increment_s'}
    if shell_pair:
        base_keys |= {'native_supervisor_exit_code', 'actual_collector_shell_exit_code', 'shell_deadline_status_pair_observed'}
    keys(cost, base_keys, 'entire exact K cost layout')
    r.exact(cost['schema'], 'flylab.m3.direct-science.full-return.v1' if q['mode'] == 'generated' else 'flylab.m3.contact-learning.full-return.v1', 'exact mode cost schema')
    r.exact(outer['schema'], 'flylab.m3.direct-science.outer.v1', 'actual supervisor outer schema')
    fee_values(cost, q, request_spec, trace, outer, root)
    for field in ('all_owned_ended', 'reservation_released', 'worker_started', 'creation_attempted', 'worker_creation_state', 'resume_attempted', 'resume_outcome'):
        r.exact(cost[field], outer[field], 'cost actual creation/closure ' + field)
    closed = closure(outer)
    return q, cost, outer, closed


def fee_values(cost, q, request_spec, trace, outer, root):
    if q['mode'] == 'original':
        science = data(q['science_request'], root)
        r.exact(cost['request'], q['science_request'], 'same immutable admitted science request')
        r.exact(cost['supervisor_request'], request_spec, 'same immutable admitted supervision request')
        for field in ('role', 'grant', 'inputs', 'roster', 'contract', 'body_retry', 'local_debit_offset', 'resource_increment_s'):
            r.exact(cost[field], science[field], 'cost admitted science ' + field)
    if outer['exit_code'] != trace['exit_code']:
        for field, value in (('native_supervisor_exit_code', outer['exit_code']),
                             ('actual_collector_shell_exit_code', trace['exit_code']), ('shell_deadline_status_pair_observed', True)):
            r.exact(cost[field], value, 'actual observed shell pair ' + field)


def accounting_hold(ledger, mode, call):
    key = ('direct_science_' if mode == 'generated' else 'startup_ray_original_') + 'blocking_holds'
    hold = ledger[key].get(call)
    if type(hold) is dict:
        hold['accounting_hold'] = HOLD
    else:
        ledger[key][call] = HOLD


def expected_ownership(expected, q, request_spec, live, root, *, charged=False):
    """Exactly admitted/begun/closed-pending/unknown layouts; no broad tail keys."""
    call, mode = q["call_id"], q["mode"]
    prefix = "direct_science_" if mode == "generated" else "startup_ray_original_"
    w, res, pending, holds = (prefix + x for x in ("workers", "reservations", "pending_actuals", "blocking_holds"))
    worker, reservation, actual, hold = (live[k] for k in (w, res, pending, holds))
    if not worker and not actual and not hold and not charged:
        if mode == "generated":
            expected[res] = {call: {"request": request_spec, "reserved_s": q["limits"]["outer_s"]}}
        return "admitted"
    if not worker and not reservation and not hold:
        expected[pending] = {} if charged else {call: {"request": request_spec, "state": "actual_pending", "ownership_closed": True}}
        return "charged" if charged else "closed-pending"
    keys(worker, (call,), "sole actual owner")
    value = worker[call]
    id_keys = ("owner_pid", "owner_creation_ticks", "worker_pid", "worker_creation_ticks")
    allowed = ("request", "state", "lock_token", *id_keys) + (("grant", "worker_max", "reserved_s") if mode == "original" else ())
    keys(value, allowed, "whole typed owner")
    for k in id_keys:
        integer(value[k], "owner " + k, positive=True)
    r.need(value["owner_pid"] != value["worker_pid"], "distinct owned worker")
    metadata_pin(value["lock_token"], r.SHARED_LOCK)
    token = data(value["lock_token"], root, 16384)
    r.exact(token['schema'], 'flylab.m3.direct-science.lock-owner.v1', 'owned token schema')
    r.exact(token['request'], request_spec, 'owned request')
    r.exact(token['call_id'], call, 'same owned literal call')
    for key in ('owner_pid', 'owner_creation_ticks'):
        r.exact(token[key], value[key], 'same owned process identity')
    r.need(type(token['entry_counter_s']) is float and number(token['entry_counter_s'], 'same ENTRY') > 0,
           'positive finite original ENTRY')
    wanted = {"request": request_spec if mode == "generated" else q["science_request"], "state": "begun",
              "lock_token": value["lock_token"], **{k: value[k] for k in id_keys}}
    wanted_res = {"request": request_spec, "reserved_s": q["limits"]["outer_s"]}
    if mode == "original":
        wanted.update(grant=q["grant"], worker_max=1, reserved_s=q["limits"]["outer_s"])
        wanted_res = {"request": q["science_request"], "grant": q["grant"], "reserved_s": q["limits"]["outer_s"]}
        expected["original_jobs_running"] = 1
        expected["original_reserved_s"] = q["limits"]["outer_s"]
    expected[w], expected[res] = {call: wanted}, {call: wanted_res}
    if hold:
        expected[holds] = {call: {"request": request_spec, "state": "ownership_unknown", **{k: value[k] for k in id_keys}}}
        return "unknown"
    r.need(not charged, "charged unresolved owner requires unknown HOLD")
    return "begun"


def validate(ledger, root):
    """Return legal accounting state, permitting incurred negative remaining."""
    event, baseline, policy, grant = recovery(ledger[KEY], root)
    expected = copy.deepcopy(baseline)
    expected[KEY] = ledger[KEY]
    expected["direct_science_blocking_holds"] = {}
    source = event["source_freeze"]
    tail = ledger["debits"][288:]
    r.need(type(tail) is list and len(tail) <= len(SEQUENCE), "finite seven-call tail")
    caps = dict(OLD_CAPS)
    failed = False
    for index, debit in enumerate(tail):
        r.exact(debit["cost_id"], SEQUENCE[index], "next literal debit")
        r.need(not failed, "paid failure does not admit successors")
        q, cost, _outer, closed = fee_proof(debit["descriptor"], debit, source, root)
        reserved = q["limits"]["outer_s"]
        expected["debits"].append(debit)
        expected["direct_science_calls"][q["call_id"]] = {"request": cost["request"] if q["mode"] == "generated" else cost["supervisor_request"],
                                                       "state": "charged", "mode": q["mode"], "reserved_s": reserved}
        failed = cost["exit_code"] != 0 or cost["cap_exceeded"] or not closed
        if not closed:
            expected_ownership(expected, q, expected["direct_science_calls"][q["call_id"]]["request"], ledger, root, charged=True)
            owner_key = ('direct_science_' if q['mode'] == 'generated' else 'startup_ray_original_') + 'workers'
            for field in ('owner_pid', 'owner_creation_ticks', 'worker_pid', 'worker_creation_ticks'):
                r.exact(_outer[field], expected[owner_key][q['call_id']][field], 'unclosed evidence retains admitted identity')
        if closed and cost['exit_code'] != 0:
            expected[('direct_science_' if q['mode'] == 'generated' else 'startup_ray_original_') + 'blocking_holds'][q['call_id']] = FAILURE_HOLD
        if cost["cap_exceeded"]:
            accounting_hold(expected, q['mode'], q['call_id'])
    if ALLOCATION_KEY in ledger:
        r.need(len(tail) >= 1, "allocation after actual bootstrap")
        metadata_pin(ledger[ALLOCATION_KEY], ALLOCATION_PATH)
        allocation = data(ledger[ALLOCATION_KEY], root)
        keys(allocation, ("schema", "epoch", "before_ledger", "recovery", "source_freeze", "bootstrap", "bootstrap_junit",
                          "old_caps", "new_caps", "delta_s", "combined_global_s", "new_resource_grant_s", "refund_s"), "allocation event")
        r.exact(allocation["schema"], "flylab.m3.checkpoint-allocation.v1", "allocation schema")
        for k, v in (("epoch", EPOCH), ("recovery", ledger[KEY]), ("source_freeze", source), ("old_caps", OLD_CAPS),
                     ("new_caps", CAPS), ("delta_s", {"generated_global": 400, "generated_local": 400, "original_global": -400}),
                     ("combined_global_s", 17900), ("new_resource_grant_s", 0), ("refund_s", 0), ("bootstrap", tail[0]["descriptor"])):
            r.exact(allocation[k], v, "exact allocation " + k)
        before = data(allocation["before_ledger"], root, 524288)
        r.need(ALLOCATION_KEY not in before and len(before["debits"]) == 289, "fresh closed bootstrap before allocation")
        state = validate(before, root)
        eligibility(before, state)
        success(before["debits"][-1], source, root, "bootstrap", allocation["bootstrap_junit"])
        expected[ALLOCATION_KEY] = ledger[ALLOCATION_KEY]
        caps = dict(CAPS)
        for k, v in caps.items():
            expected[r.CAP_FIELDS[k]] = v
    else:
        r.need(len(tail) <= 1, "qualification cannot use old caps")
    if QUALIFICATION_KEY in ledger:
        r.need(ALLOCATION_KEY in ledger and len(tail) >= 5, "separate four actual costs before registration")
        metadata_pin(ledger[QUALIFICATION_KEY], QUALIFICATION_PATH)
        qualification(data(ledger[QUALIFICATION_KEY], root), ledger, source, root)
        expected[QUALIFICATION_KEY] = ledger[QUALIFICATION_KEY]
    r.need(len(tail) <= 5 or QUALIFICATION_KEY in ledger, "Original requires registered K qualification")
    expected.update(totals(expected, policy, grant, caps))
    active_calls = set(ledger["direct_science_calls"]) - set(expected["direct_science_calls"])
    r.need(len(active_calls) <= 1, "at most next admitted call")
    phase = "charged"
    if active_calls:
        r.need(not failed and len(tail) < len(SEQUENCE), "no call after failure/finite end")
        call = SEQUENCE[len(tail)]
        r.exact(active_calls, {call}, "no old/replay/alias admission")
        admitted = ledger["direct_science_calls"][call]
        keys(admitted, ("request", "state", "mode", "reserved_s"), "whole admission")
        q = data(admitted["request"], root)
        identity(q, admitted["request"], source, root)
        r.exact(admitted, {"request": admitted["request"], "state": "admitted", "mode": q["mode"], "reserved_s": q["limits"]["outer_s"]}, "exact admitted identity")
        r.need(len(tail) != 1 or ALLOCATION_KEY in ledger, "no lint before allocation")
        r.need(len(tail) != 5 or QUALIFICATION_KEY in ledger, "no body before K qualification")
        if len(tail) == 6:
            valid_body(tail[5], source, data(q['science_request'], root), root)
        expected["direct_science_calls"][call] = admitted
        phase = expected_ownership(expected, q, admitted["request"], ledger, root)
    # A legitimate cumulative breach adds the exact responsible HOLD, never erases a debit.
    if tail:
        last = tail[-1]
        breached = ((last["mode"] == "generated" and (expected["direct_science_generated_used_s"] > caps["generated_local"]
                    or expected["generated_shared_remaining_s"] < 0)) or
                    (last["mode"] == "original" and (expected["original_remaining_s"] < 0
                    or expected["original_stage_used_s"] > 6300 or expected["startup_ray_original_local_used_s"] > 6000)))
        if breached:
            accounting_hold(expected, last['mode'], last['cost_id'])
    r.exact(ledger, expected, "whole reconstructed K state; all immutable288 fields/calls retained")
    return {"caps": caps, "completed": len(tail), "phase": phase, "failed": failed, "source": source, "epoch": EPOCH}


def eligibility(ledger, state):
    r.need(state["phase"] == "charged" and not state["failed"], "known charged successful predecessor")
    for key in r.OWNER_KEYS:
        r.exact(ledger[key], {}, "closed eligible ownership " + key)
    r.need(ledger["generated_shared_remaining_s"] >= 0 and ledger["original_remaining_s"] >= 0
           and ledger["direct_science_generated_used_s"] <= state["caps"]["generated_local"], "remaining execution capacity")


def success(debit, source, root, kind, junit_spec=None):
    q, cost, _outer, closed = fee_proof(debit["descriptor"], debit, source, root)
    r.need(closed and cost["exit_code"] == 0 and cost["cap_exceeded"] is False and _outer["worker_started"] is True and _outer["worker_creation_state"] == "RESUMED", "actual successful resumed paid known closed role")
    if kind != "lint":
        freeze = data(source, root)
        metadata_pin(junit_spec, BASE + "/parent-contact-learning-k-" + kind.replace("_", "-") + "-junit.xml")
        r.successful_junit(junit_spec, roster(freeze, root)[kind], root)
    return q


def qualification(value, ledger, source, root):
    keys(value, ("schema", "epoch", "source_freeze", "actual_costs", "tests"), "K qualification")
    r.exact(value["schema"], "flylab.m3.direct-science.qualification.v8", "distinct K qualification")
    r.exact(value["epoch"], EPOCH, "same K epoch")
    r.exact(value["source_freeze"], source, "one tested Source")
    costs = ledger["debits"][289:293]
    r.exact(value["actual_costs"], [d["descriptor"] for d in costs], "four separately paid K qualification calls")
    r.need(len(costs) == 4 and len(value["tests"]) == 3, "four full costs/three full JUnits")
    for index, debit in enumerate(costs):
        r.exact(debit["cost_id"], QUAL_CALLS[index], "fixed role order")
        success(debit, source, root, ("lint", "math", "gate_a", "gate_b")[index], None if index == 0 else value["tests"][index-1])


def bounded_source(spec, root, cap):
    """The selected Source cap bounds each actual read, including installed text."""
    keys(spec, ('path', 'bytes', 'sha256'), 'whole Source descriptor')
    integer(spec['bytes'], 'Source bytes')
    r.need(0 <= spec['bytes'] <= cap, 'selected actual Source cap')
    path = Path(spec['path'])
    if not path.is_absolute():
        r.need(path.suffix in ('.py', '.ps1', '.js', '.json', '.txt', '.ini', '.md', '.xml'), 'relative text Source only')
        return r.raw_pin(spec, root, cap, allow_empty=True)
    old = data(J_FREEZE, root)['sources']
    r.need(spec['path'] in old, 'installed roots come only from exact inherited J vector')
    r.exact(spec, old[spec['path']], 'fixed installed Source path and pin')
    r.need(path.suffix in ('.py', '.ps1', '.json', '.txt', '.ini', '.md'), 'installed text extension')
    relative = path.relative_to(path.anchor).as_posix()
    operand = r._text_path(path.anchor, relative)
    before = os.lstat(operand)
    r.need(stat.S_ISREG(before.st_mode) and before.st_size == spec['bytes'], 'actual installed regular Source bytes')
    with operand.open('rb') as stream:
        raw = stream.read(cap + 1)
    raw.decode('utf8', errors='strict')
    with operand.open('rb') as stream:
        again = stream.read(cap + 1)
    r._text_path(path.anchor, relative)
    r.need(len(raw) <= cap and len(raw) == spec['bytes'] and raw == again
           and r._text_stamp(os.lstat(operand)) == r._text_stamp(before)
           and hashlib.sha256(raw).hexdigest() == spec['sha256'], 'whole capped installed Source readback/stamp/SHA')
    return raw


def source_fence(source, root, check=lambda: None):
    """Fresh whole current Source: call only after actual SharedLock acquisition."""
    metadata_pin(source, SOURCE_PATH)
    check()
    freeze = data(source, root)
    r.exact(freeze["ROOT_review_cycle_complete"], True, "ROOT adopted K reviewed Source")
    r.exact(freeze["qualification_epoch"], EPOCH, "distinct K Source")
    metadata_pin(freeze["reviewed_source"], REVIEW_PATH)
    review = data(freeze["reviewed_source"], root)
    keys(review, ('schema', 'epoch', 'sources', 'replacements', 'additions', 'before_source', 'root_reflection',
                  'astra_review', 'root_reviews', 'sol_reports'), 'complete K reviewed-source metadata')
    r.exact(review['schema'], 'flylab.m3.checkpoint-reviewed-source.v1', 'distinct K reviewed Source')
    r.exact(review['epoch'], EPOCH, 'K Source review epoch')
    r.exact(review['before_source'], J_FREEZE, 'exact J728 basis')
    original = data(J_FREEZE, root)
    old = original['sources']
    expected_freeze = copy.deepcopy(original)
    expected_freeze.update(qualification_epoch=EPOCH, reviewed_source=freeze['reviewed_source'],
                           body_retry=freeze['sources'][BODY_PATH], sources=freeze['sources'],
                           checkpoint_contract=freeze['sources'][CONTRACT_PATH])
    r.exact(freeze, expected_freeze, 'whole inherited freeze with only finite K changes')
    r.need(len(old) == 728 and set(freeze["sources"]) == set(old) | set(ADDITIONS), "entire J728 plus explicit additions")
    r.exact(review["sources"], freeze["sources"], "same reviewed vector")
    r.exact(review["replacements"], list(REPLACEMENTS), "all and only declared replacements")
    r.exact(review["additions"], list(ADDITIONS), "all and only declared new dependencies")
    for name, spec in old.items():
        if name not in REPLACEMENTS:
            r.exact(freeze["sources"][name], spec, "whole unchanged Source pin " + name)
    for name, spec in freeze["sources"].items():
        check()
        r.exact(spec['path'], name, 'same whole Source namespace')
        evidence = name in old and old[name]['bytes'] > 262144
        cap = 524288 if evidence else (65536 if name in ADDITIONS[:2] else 262144)
        r.need(spec['bytes'] <= cap, 'bounded Source/fixture or exact inherited evidence')
        if evidence:
            r.exact(spec, old[name], 'only fixed J inherited evidence may use524288')
            r.raw_pin(spec, root, cap)
        else:
            bounded_source(spec, root, cap)
    check()
    roster(freeze, root)
    body_binding_pin = freeze["sources"][BODY_PATH]
    r.exact(freeze["body_retry"], body_binding_pin, "static body in acyclic Source vector")
    return freeze


def dispatch(ledger, q, request_spec, root, *, phase):
    state = validate(ledger, root)
    r.need(phase in ("closed", "admitted"), "finite dispatch phase")
    if phase == "closed":
        eligibility(ledger, state)
        r.exact(q["call_id"], SEQUENCE[state["completed"]], "next literal call")
    else:
        r.exact(state["phase"], "admitted", "exact root admission before worker")
        r.exact(ledger["direct_science_calls"][q["call_id"]]["request"], request_spec, "same root-admitted K request")
    identity(q, request_spec, state["source"], root)
    if q["mode"] == "generated":
        envelope = 575 if q["call_id"] == QUAL_CALLS[0] else q["limits"]["outer_s"] + 20
        r.need(ledger["direct_science_generated_used_s"] + envelope <= state["caps"]["generated_local"]
               and ledger["generated_shared_remaining_s"] >= envelope, "complete paired Generated envelope")
    else:
        r.need(QUALIFICATION_KEY in ledger, "registered whole K qualification before Original")
        original_envelope = 600 if q['call_id'] == CALLS[0] else q['limits']['outer_s'] + 180
        for field, cap in (("original_cumulative_used_s", CAPS["original_global"]), ("original_stage_used_s", 6300), ("startup_ray_original_local_used_s", 6000)):
            r.need(ledger[field] + original_envelope <= cap, 'whole body/audit/close600 or conditional audit plus close180')
    for index in range(1, min(state['completed'], 5)):
        kind = ('lint', 'math', 'gate_a', 'gate_b')[index-1]
        junit = None if kind == 'lint' else r.text_spec(BASE + '/parent-contact-learning-k-' + kind.replace('_', '-') + '-junit.xml', root)
        success(ledger['debits'][288+index], state['source'], root, kind, junit)
    return state


def body_binding(q, contract, read):
    r.need(q["call_id"] in CALLS and q["role"] == ("contact_candidate_body", "contact_body_audit")[CALLS.index(q["call_id"])], "sole K body/audit mapping")
    r.exact(q["source_freeze"]["path"], SOURCE_PATH, "K execution Source")
    freeze = read(q["source_freeze"])
    r.exact(q["body_retry"], freeze["sources"][BODY_PATH], "immutable K body pin")
    value, old = read(q["body_retry"]), read(J_BODY)
    expected = copy.deepcopy(old)
    expected.update(schema="flylab.m3.contact-learning.body-runtime.v8", retry_id="contact-learning-20261008-k", calls=list(CALLS),
                    qualification_epoch=EPOCH, source_review_path=REVIEW_PATH, source_freeze_path=SOURCE_PATH,
                    checkpoint_contract_path=CONTRACT_PATH, checkpoint_recovery_path=RECOVERY_PATH,
                    checkpoint_allocation_path=ALLOCATION_PATH, historical_closed288=ANCHOR, failed_j_lint=J_FAILED)
    r.exact(value, expected, "whole J scientific body with only K routing additions")
    for key in ("contract", "grant", "inputs", "roster"):
        r.exact(q[key], value[key], "same candidate metadata " + key)
    r.exact(q["limits"], contract["limits"][q["role"]], "unchanged body/audit scientific envelopes")
    return value


def original_metadata(science, outer, root):
    """K common Original guarantees from immutable metadata, no J history/Source reads."""
    def read(pin):
        return data(pin, root)
    old = read(J_BODY)
    for key in ('contract', 'grant', 'inputs', 'roster'):
        r.exact(science[key], old[key], 'fixed immutable science pin before reading ' + key)
    metadata_pin(science['source_freeze'], SOURCE_PATH)
    metadata_pin(science['body_retry'], BODY_PATH)
    metadata_pin(science['qualification'], QUALIFICATION_PATH)
    contract = read(science['contract'])
    r.exact(contract['schema'], 'flylab.m3.contact-learning.stage.v1', 'fixed science contract schema')
    r.exact(contract['roles'], ['contact_direction', 'contact_candidate_audit', 'contact_candidate_body', 'contact_body_audit'], 'whole fixed science roles')
    body = body_binding(science, contract, read)
    for key, value in (('schema', 'flylab.m3.contact-learning.request.v1'), ('mode', 'original'),
                       ('ledger_path', BASE + '/parent-e358-execution-ledger.json'), ('shared_original_lock_path', r.SHARED_LOCK),
                       ('local_debit_offset', 165), ('resource_increment_s', 0),
                       ('output', BASE + '/parent-' + science['call_id'] + '/result')):
        r.exact(science[key], value, 'finite K science ' + key)
    r.exact(outer['mode'], 'original', 'same Original outer')
    r.exact(outer['output'], science['output'] + '-supervision', 'whole science/outer output relation')
    for key in ('call_id', 'role', 'grant', 'source_freeze', 'ledger_path', 'shared_original_lock_path'):
        r.exact(outer[key], science[key], 'same K Original ' + key)
    for key, value in outer['limits'].items():
        r.exact(science['limits'][key], value, 'whole exact outer science cap ' + key)
    r.exact(science['limits'], contract['limits'][science['role']], 'whole science limits including internal')
    roster, grant = read(science['roster']), read(science['grant'])
    r.exact(science['grant'], r.J_PIN, 'fixed existing grant')
    for key, value in (('schema', 'flylab.m3.startup-ray.original-grant.v1'), ('grant_id', 'startup-ray-original-20261007-j'),
                       ('reset', False), ('resource_increment_s', 0), ('local_debit_offset', 165), ('worker_max', 1), ('automatic_restarts', 0)):
        r.exact(grant[key], value, 'existing finite grant ' + key)
    r.exact(roster['schema'], 'flylab.m3.contact-conditioned-learning.input-roster.v1', 'fixed input roster schema')
    r.exact(contract['roster'], science['roster'], 'fixed contract roster')
    r.exact(roster['grant'], science['grant'], 'same immutable roster grant')
    r.exact(roster['original_inputs'], science['inputs'], 'same immutable input binder')
    r.exact(grant['inputs'], science['inputs'], 'same grant input lineage')
    inputs = read(science['inputs'])
    r.need(type(inputs) is dict, 'intact metadata input binder, no payload reads')
    r.exact(science['historical_manifest'], read(roster['previous_geometry_roster'])['historical_manifest'], 'fixed historical manifest lineage')
    read(science['historical_manifest'])
    freeze = read(science['source_freeze'])
    r.exact(freeze['schema'], 'flylab.m3.contact-learning.source-freeze.v1', 'same K freeze schema')
    r.exact(freeze['ROOT_review_cycle_complete'], True, 'actually reviewed immutable K freeze')
    r.exact(freeze['qualification_epoch'], EPOCH, 'finite K epoch')
    r.exact(freeze['sources'][science['contract']['path']], science['contract'], 'same frozen contract')
    r.exact(freeze['human_authorization'], grant['human_authorization'], 'unchanged human scope')
    qualified = read(science['qualification'])
    r.exact(qualified['schema'], 'flylab.m3.direct-science.qualification.v8', 'same K qualification schema')
    r.exact(qualified['epoch'], EPOCH, 'same K qualification epoch')
    r.exact(qualified['source_freeze'], science['source_freeze'], 'same admitted qualified freeze')
    before = data(science['before_ledger'], root, 524288)
    r.exact(before[QUALIFICATION_KEY], science['qualification'], 'before carries actual K qualification')
    r.exact(data(before[KEY], root)['source_freeze'], science['source_freeze'], 'same recovered immutable Source lineage')
    r.need(ALLOCATION_KEY in before, 'existing conserved allocation before Original')
    return body


def candidate(q, report, read):
    body = body_binding(q, read(q["contract"]), read)
    historic = read(body["historical_manifests"][1])
    audit = read(historic["artifacts"]["report"])
    r.exact(audit["candidate"], body["candidate"], "entire audited candidate-k")
    r.exact(report["candidate"], body["candidate"], "same whole candidate-k")
    r.exact(historic["artifacts"]["candidate_arrays"], body["candidate_arrays"], "literal array descriptor only")
    manifest = read(q["artifacts"]["manifest"])
    r.exact(manifest["source_freeze"], body["source_k"] if q["role"] == "contact_candidate_body" else q["source_freeze"], "immediate predecessor Source")
    r.exact(manifest["role"], "contact_candidate_audit" if q["role"] == "contact_candidate_body" else "contact_candidate_body", "immediate predecessor role")
    r.exact(manifest["run_valid"], True, "technically valid predecessor")
    r.exact(q["artifacts"], dict(manifest["artifacts"], manifest=q["artifacts"]["manifest"]), "all predecessor artifact pins")


def valid_body(debit, source, q, root):
    request, cost, _outer, closed = fee_proof(debit["descriptor"], debit, source, root)
    r.need(closed and cost["exit_code"] == 0 and not cost["cap_exceeded"] and request["call_id"] == CALLS[0], "audit only valid genuine body")
    manifest = data(q["artifacts"]["manifest"], root)
    r.exact(manifest["request"], cost["request"], "same actual body manifest")
    r.exact(manifest["run_valid"], True, "actual body run_valid")
    report = data(manifest["artifacts"]["report"], root)
    r.need(report["run_valid"] is True and report["first_error"] is None and not report["secondary_errors"], "actual report valid")
    candidate(q, report, lambda pin: data(pin, root))


def cold_contact(admission, label):
    """Preserve the existing deadline/Original owner/Job/fence without old history recursion."""
    from tools import banc_m3_startup_ray_supervisor as supervisor
    from tools import verify_c_banc_m3_startup_ray as ray
    q, root = admission.request, admission.root
    admission.deadline()
    for spec, expected, cap in ((admission.request_spec, q, 262144), (q["before_ledger"], admission.before, 524288),
                                (q["contract"], admission.contract, 262144), (q["grant"], admission.grant, 262144),
                                (q["inputs"], admission.inputs, 262144), (q["roster"], admission.roster, 262144)):
        r.exact(admission.read(spec, cap), expected, "fresh exact K metadata")
    body_binding(q, admission.contract, admission.read)
    source_fence(q["source_freeze"], root, admission.deadline)
    before_state = validate(admission.before, root)
    eligibility(admission.before, before_state)
    live = admission.read(ray.spec_for(q["ledger_path"], root, cap=524288), 524288)
    state = validate(live, root)
    r.exact(live["debits"], admission.before["debits"], "all original/generated actual debit history")
    r.exact(live[QUALIFICATION_KEY], q["qualification"], "registered K qualification")
    admitted = live["direct_science_calls"][q["call_id"]]
    outer = admission.read(admitted["request"])
    r.exact(outer["science_request"], admission.request_spec, "same science worker request")
    if admission.stage == "pre_spawn":
        r.exact(state["phase"], "admitted", "no K worker yet")
        r.need(not ray.path(r.SHARED_LOCK, root).exists() and not admission.output.exists(), "fresh physical exclusion/output")
    else:
        r.exact(state["phase"], "begun", "actual K Original begun owner")
        worker = live["startup_ray_original_workers"][q["call_id"]]
        r.exact(worker["worker_pid"], os.getpid(), "this actual worker")
        for name in ("owner", "worker"):
            r.exact(supervisor.creation_ticks(worker[name + "_pid"]), worker[name + "_creation_ticks"], "live creation identity")
        token = admission.read(worker["lock_token"])
        r.exact(ray.json_load(os.environ["FLYLAB_DIRECT_SCIENCE_OWNER"].encode("utf8")), token, "actual same ENTRY token")
        r.exact(token["entry_counter_s"], admission.entry, "no reset ENTRY")
        r.exact(token["request"], admitted["request"], "same owner request")
        r.need(supervisor.in_named_job(token["job_name"]), "same actual owned Job")
    r.need(label in ("initial", "pre_array", "publication", "pre_construct"), "unchanged cold boundary labels")
    admission.deadline()
    return True


def initial_refusal(root, request_spec, q, ledger_spec, raw, error, *, entry):
    """Lexical pre-create guard failure: no stream/EOF or fabricated fee."""
    from tools import banc_m3_startup_ray_supervisor as supervisor
    r.need(not r._text_path(root, r.SHARED_LOCK, write=True).exists(), "initial refusal before physical lock")
    live = r.read_pin(ledger_spec, root, 524288)
    state = validate(live, root)
    r.exact(state["phase"], "admitted", "already admitted intact call")
    identity(q, request_spec, state["source"], root)
    r.exact(r.raw_pin(ledger_spec, root, 524288), raw, "unchanged admitted raw")
    output = r._text_path(root, q["output"], write=True)
    r.need(not output.exists(), "fresh refusal output")
    output.mkdir(parents=True)
    outer = {"schema": "flylab.m3.direct-science.outer.v1", "request": request_spec, "exit_code": 1,
             "worker_exit_code": None, "worker_pid": None, "worker_creation_ticks": None,
             "owner_pid": os.getpid(), "owner_creation_ticks": supervisor.creation_ticks(os.getpid()),
             "all_owned_ended": True, "reservation_released": True, "unknown_owned_identities": {},
             "first_error": {"type": type(error).__name__, "message": str(error)[:8192]}, "secondary_errors": [],
             "timeout": isinstance(error, TimeoutError), "stdout_eof": False, "stderr_eof": False,
             "stdout": None, "stderr": None, "pre_spawn_refused": True, "job_created": False,
             "worker_started": False, "creation_attempted": False, "worker_creation_state": "NO_CREATE_ATTEMPT",
             "resume_attempted": False, "resume_outcome": "NOT_ATTEMPTED", "checkpoint_recovery": live[KEY],
             "initial_ledger": ledger_spec, "refusal_origin": "K_INITIAL_GUARD_BEFORE_CREATE",
             "observed_supervisor_interval_s": time.perf_counter() - entry}
    supervisor.durable(output / "outer.json", outer)
    return outer
