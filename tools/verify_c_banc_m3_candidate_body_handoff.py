"""E358 bounded candidate -> independent audit -> same-candidate body adapter.

ROOT prepares fresh contracts with build_contract, then runs --supervise CONTRACT.
The sole E267 job is --child CONTRACT --pre-spawn RECEIPT. Parent and child use
one perf_counter ENTRY, recorded durably before Popen; no original payload or
numerical library is imported by the parent. The passive reference module is
used only to reproduce its metadata-only make_freeze (never its original gate).

Contract v1 has the exact keys in CONTRACT_KEYS. ROOT must supply a current
adapter_qualification descriptor for original mode, previous once-charged
full-return descriptors, and dependency descriptors in inputs. A candidate
audit requires worker_exit; body requires worker_exit and candidate_audit_exit;
body_audit also requires body_exit. They must be this attempt's natural closed
E358 exits, and their provisional output envelopes are pinned by those exits.

The wrapper produces outer.json, NOT actual tool full-return. After the tool
returns ROOT saves actual_returned_exit_code and actual_outer_full_return_s in
an external JSON, and calls seal_full_return. This is the only accounting seal.
No internal elapsed number is substituted for an actual tool-return duration.
Late outer publication retains outer_revocation separately. Closure fields are
actual owned identity observations; failure remains chargeable and unqualified.
Outer work reserves .5 seconds within its existing cap for cleanup/publication,
and completion waits reserve .1 seconds for the final durable report.

Generated qualification: --execute-current-pytest CASE runs every function in
the current test file with setup/call/teardown receipts, no filter or external
pytest plugins. CASE keys: schema, source_sha256, expected_functions, receipt,
pytest_ini. ROOT seals its real supervised return separately. Fixture contracts
exercise the same real E346 -> E267 -> child -> durable publication path with
original_calls=0; they never load original scientific arrays or GPU libraries.
Generated suite caps are 20/18/16/15 seconds (outer/child/internal/science);
the standalone generated fixture keeps 8/6/5/4. Original caps are unchanged.
All success is provisional until ROOT's actual exit seal and independent audit.
F1-F8, M2/UI/F6, biology and natural gait are deliberately never certified here.
"""

from __future__ import annotations

import time

ENTRY = time.perf_counter()
# Keep the absolute ENTRY ahead of startup work, including stdlib imports.
import argparse  # noqa: E402
import ast  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from pathlib import Path, PurePosixPath  # noqa: E402
import re  # noqa: E402
import stat  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import zipfile  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = "verification/banc-natural-gait-m3-20260919"
OWNED = ("tools/verify_c_banc_m3_candidate_body_handoff.py",
         "tests/test_c_banc_candidate_body_handoff.py")
REGISTRATION = BASE + "/parent-e358-reflected-registration.json"
REGISTRATION_SHA = "2f78e695a9acc468cca4cdd401a7a214c180ff833ab2d5b604306f33bd8ead1c"
SUPERVISOR = BASE + "/parent-e267-supervise.py"
OBSERVER = BASE + "/parent-e346-supervise-identities.py"
PYTHON = ".venv/Scripts/python.exe"
BASELINE = 9893.754960899503
GENERATED_BASELINE = 80.1900156
CONTRACT_KEYS = {"schema", "mode", "role", "attempt", "attempt_id", "phase",
                 "original_calls", "registration", "source_sha256", "limits",
                 "commands", "supervisor_output", "max_tree_rss_bytes", "paths",
                 "cumulative_before_s", "cumulative_limit_s", "stage_used_s",
                 "stage_limit_s", "reserved_s", "pending_reservations", "prior_exits",
                 "adapter_qualification", "science_freeze", "inputs", "retry_evidence",
                 "retained_reserve_bytes", "cpu_threads"}
FIXTURE_LIMITS = {"outer_s": 8, "child_s": 6, "internal_s": 5, "science_s": 4,
                  "host_bytes": 2 << 30, "output_cap_bytes": 16 << 20}
SUITE_LIMITS = {"outer_s": 20, "child_s": 18, "internal_s": 16, "science_s": 15,
                "host_bytes": 2 << 30, "output_cap_bytes": 16 << 20}
OUTER_CLEANUP_MARGIN_S = .5
OUTER_REPORT_MARGIN_S = .1
FALSE_MILESTONES = {"F1_F8_complete": False, "biology_proven": False,
                    "natural_gait": False, "candidate_adopted": False,
                    "M2_UI_F6_changed": False}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def numeric(value, label, *, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            "typed finite " + label)
    return value


def typed_equal(actual, expected, label):
    require(type(actual) is type(expected), "exact typed " + label)
    if type(expected) is dict:
        require(set(actual) == set(expected), "literal keys " + label)
        for key, value in expected.items():
            typed_equal(actual[key], value, label + "." + key)
    elif type(expected) is list:
        require(len(actual) == len(expected), "literal length " + label)
        for a, b in zip(actual, expected, strict=True):
            typed_equal(a, b, label)
    else:
        require(actual == expected, "exact value " + label)


def strict_json(raw):
    def pairs(rows):
        out = {}
        for key, value in rows:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: require(False, "nonfinite JSON"),
                      parse_float=lambda v: numeric(float(v), "JSON", minimum=-math.inf))


def normalize(value):
    if type(value) in (str, int, bool) or value is None:
        return value
    if type(value) is float:
        numeric(value, "serialization", minimum=-math.inf)
        return value
    if type(value) in (tuple, list):
        return [normalize(v) for v in value]
    if type(value) is dict:
        require(all(type(k) is str for k in value), "JSON string keys")
        return {k: normalize(v) for k, v in value.items()}
    # No numpy import is needed to serialize already acquired numpy scalars.
    if type(value).__module__.startswith("numpy") and hasattr(value, "item"):
        return normalize(value.item())
    raise TypeError("unsupported finite JSON type " + type(value).__name__)


def json_bytes(value):
    return (json.dumps(normalize(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf8")


def safe(name, root=ROOT):
    require(type(name) is str and "\\" not in name and ":" not in name and "\0" not in name,
            "literal relative path")
    parts = PurePosixPath(name).parts
    require(parts and not name.startswith("/") and ".." not in parts
            and "." not in parts and PurePosixPath(name).as_posix() == name,
            "canonical relative path")
    path = root / name
    require(path.resolve().is_relative_to(root.resolve()), "contained path")
    for p in (path, *path.parents):
        if p.exists():
            require(not p.is_symlink() and not (getattr(p.lstat(), "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024)), "no source/output links")
        if p == root:
            break
    return path


def descriptor(name, raw):
    return {"path": name, "bytes": len(raw), "sha256": sha(raw)}


def file_spec(name, root=ROOT):
    return descriptor(name, safe(name, root).read_bytes())


def read_spec(spec, root=ROOT, *, text_only=True):
    require(type(spec) is dict and set(spec) == {"path", "bytes", "sha256"}
            and type(spec["bytes"]) is int and spec["bytes"] > 0
            and type(spec["sha256"]) is str and re.fullmatch("[0-9a-f]{64}", spec["sha256"]),
            "exact typed descriptor")
    if text_only:
        require(Path(spec["path"]).suffix in (".py", ".json", ".md", ".txt", ".ini"),
                "pre-array text only")
    path = safe(spec["path"], root)
    require(path.stat().st_size == spec["bytes"], "descriptor bytes")
    raw = path.read_bytes()
    require(sha(raw) == spec["sha256"], "descriptor SHA")
    return raw


def own_sources(root=ROOT):
    return {name: sha(safe(name, root).read_bytes()) for name in (*OWNED, SUPERVISOR, OBSERVER)}


def registration(root=ROOT):
    raw = safe(REGISTRATION, root).read_bytes()
    require(len(raw) == 54329 and sha(raw) == REGISTRATION_SHA, "fixed reviewed E358 registration")
    return strict_json(raw)


def closure_sources(reg, root=ROOT):
    pins = dict(reg["science_protected_text_sha256"])
    for name, digest in reg["body_source_sha256"].items():
        require(name not in pins or pins[name] == digest, "source pin collision")
        pins[name] = digest
        # The immutable body collector reports installed modules inside ROOT
        # relatively; its dependency descriptor lists the same files absolutely.
        # Authorize only that exact same pinned file, with collision rejection.
        path = Path(name)
        if path.is_absolute() and path.resolve().is_relative_to(Path(root).resolve()):
            alias = path.resolve().relative_to(Path(root).resolve()).as_posix()
            require(alias not in pins or pins[alias] == digest, "source alias pin collision")
            pins[alias] = digest
    for key in ("plan", "actual_Astra_review", "Astra_review_md", "ROOT_reflection",
                "science_registration", "science_qualification", "science_freeze",
                "body_inventory", "body_native_dependencies"):
        spec = reg[key]
        read_spec(spec, root)
        pins[spec["path"]] = spec["sha256"]
    pins[REGISTRATION] = REGISTRATION_SHA
    pins.update(own_sources(root))
    return pins


def check_sources(pins, root=ROOT):
    for name, digest in pins.items():
        require(type(digest) is str and re.fullmatch("[0-9a-f]{64}", digest), "typed source SHA")
        path = Path(name) if Path(name).is_absolute() else safe(name, root)
        require(path.suffix in (".py", ".json", ".md", ".txt", ".ini"), "source text closure")
        for p in (path, *path.parents):
            require(not p.is_symlink() and not (getattr(p.lstat(), "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024)), "source no reparse")
        require(sha(path.read_bytes()) == digest, "current source closure: " + name)


def passive_science(reg, root=ROOT):
    from tools import verify_c_banc_m3_reference_parameter_direction as ref
    frozen = ref.make_freeze(reg["science_qualification"]["path"], root=root)
    require(frozen == strict_json(read_spec(reg["science_freeze"], root))
            and frozen["original_authorized"] is False, "current E357 Q/F reproduction")
    return frozen


def names(attempt_id, role):
    require(type(attempt_id) is str and re.fullmatch("[a-z0-9][a-z0-9-]{0,55}", attempt_id),
            "literal attempt identity")
    stem = "parent-e346-e358-" + attempt_id + "-" + role
    base = BASE + "/" + stem
    return {"contract": base + ".json", "output": BASE + "/e358-" + attempt_id + "-" + role,
            "pre_spawn": base + "-pre-spawn.json", "outer": base + "-outer.json",
            "outer_revocation": base + "-outer-revocation.json",
            "actual_return": base + "-actual-tool-return.json", "full_exit": base + "-full-exit.json",
            "supervision": base + "-supervision.json", "stdout": base + "-child.stdout.txt",
            "stderr": base + "-child.stderr.txt", "witness_stdout": base + "-witness.stdout.txt",
            "witness_stderr": base + "-witness.stderr.txt"}


def build_contract(role, attempt_id, *, mode="original", attempt=1, prior_exits=(),
                   adapter_qualification=None, inputs=None, retry_evidence=None,
                   pending_reservations=None, root=ROOT):
    """Metadata helper only; ROOT writes returned exact bytes to paths.contract.

    pending_reservations holds remaining dependent roles (unique key e.g.
    candidate_audit_first/body/body_audit); each value is that role's outer cap.
    It must include the complete downstream candidate-audit/body/body-audit path.
    Prior exits include failures too. No original NPZ/NPY is read by this helper.
    """
    require(mode in ("original", "generated"), "mode")
    reg = registration(root) if mode == "original" else None
    limits = dict(reg["roles"][role] if reg else SUITE_LIMITS if role == "suite" else FIXTURE_LIMITS)
    paths = names(attempt_id, role)
    sources = closure_sources(reg, root) if reg else own_sources(root)
    previous = [strict_json(read_spec(v, root)) for v in prior_exits]
    used = sum(numeric(v["actual_outer_full_return_s"], "prior actual wall", minimum=1e-12)
               for v in previous)
    command = [PYTHON, "-X", "utf8", "-B", OWNED[0], "--child", paths["contract"],
               "--pre-spawn", paths["pre_spawn"]]
    return {"schema": "flylab.m3.e358.supervisor-contract.v1", "mode": mode,
            "role": role, "attempt": attempt, "attempt_id": attempt_id,
            "phase": "E358-" + role, "original_calls": int(mode == "original"),
            "registration": file_spec(REGISTRATION, root) if reg else None,
            "source_sha256": sources, "limits": limits,
            "commands": [{"argv": command, "stdout": paths["stdout"],
                          "stderr": paths["stderr"], "wall_s": limits["child_s"]}],
            "supervisor_output": paths["supervision"], "max_tree_rss_bytes": limits["host_bytes"],
            "paths": paths, "cumulative_before_s": (BASELINE if reg else GENERATED_BASELINE) + used,
            "cumulative_limit_s": 10200 if reg else 150, "stage_used_s": used,
            "stage_limit_s": 300 if reg else 150 - GENERATED_BASELINE,
            "reserved_s": limits["outer_s"], "pending_reservations": pending_reservations or {},
            "prior_exits": list(prior_exits), "adapter_qualification": adapter_qualification,
            "science_freeze": reg["science_freeze"] if reg else None,
            "inputs": inputs or {}, "retry_evidence": retry_evidence,
            "retained_reserve_bytes": min(1 << 20, limits["output_cap_bytes"] // 4), "cpu_threads": 1}


class Clock:
    def __init__(self, entry, limits, *, clock=time.perf_counter):
        self.entry, self.limits, self.clock = entry, limits, clock

    def check(self, stage="internal"):
        now = self.clock()
        require(math.isfinite(now) and now >= self.entry, "same monotone ENTRY")
        if now > self.entry + self.limits[stage + "_s"]:
            raise TimeoutError("E358 same-ENTRY " + stage + " deadline")

    def remaining(self, margin=0):
        self.check("outer")
        return max(0., self.entry + self.limits["outer_s"] - margin - self.clock())


def retained(c, root=ROOT):
    paths = c["paths"]
    stem = Path(paths["contract"]).stem
    files = [p for p in safe(BASE, root).glob(stem + "*") if p.is_file()]
    output = safe(paths["output"], root)
    if output.exists():
        files.extend(p for p in output.rglob("*") if p.is_file())
    return sum(p.stat().st_size for p in set(files))


def durable(path, payload, *, cap, check=lambda: None):
    require(type(payload) is bytes and len(payload) <= cap and not path.exists(), "fresh bounded publication")
    check()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    require(path.read_bytes() == payload, "fsync/closed/reread publication")
    check()
    return descriptor(path.name, payload)


def put(c, name, value, timer, *, raw=False, root=ROOT, emergency=False):
    require(re.fullmatch("[a-z0-9.-]+", name), "literal output member")
    payload = value if raw else json_bytes(value)
    def check():
        if not emergency:
            timer.check()
        reserve = 128 << 10 if emergency else c["retained_reserve_bytes"]
        require(retained(c, root) + (0 if safe(c["paths"]["output"] + "/" + name, root).exists()
                else len(payload)) + reserve <= c["limits"]["output_cap_bytes"], "recursive advance retention")
    path = safe(c["paths"]["output"] + "/" + name, root)
    durable(path, payload, cap=16 << 20 if name.endswith(".npz") else 2 << 20, check=check)
    return file_spec(c["paths"]["output"] + "/" + name, root)


def error_record(error, stage):
    return {"stage": stage, "type": type(error).__name__, "message": str(error)[:1024],
            "notes": [str(v)[:1024] for v in getattr(error, "__notes__", ())]}


def fail(result, error, stage):
    row = error_record(error, stage)
    if result["first_error"] is None:
        result["first_error"] = row
    else:
        result["secondary_errors"].append(row)
    result.update(run_valid=False, effective_qualified=False, revoked=True, classification="technical_invalid")


def envelope(c, contract_spec, receipt_spec):
    return {"schema": "flylab.m3.e358.provisional-envelope.v1", "role": c["role"],
            "attempt": c["attempt"], "attempt_id": c["attempt_id"], "mode": c["mode"],
            "source_sha256": c["source_sha256"], "contract": contract_spec,
            "pre_spawn": receipt_spec, "adapter_qualification": c["adapter_qualification"],
            "science_freeze": c["science_freeze"], "cumulative_before_s": c["cumulative_before_s"],
            "cumulative_after_s": None, "cumulative_limit_s": c["cumulative_limit_s"],
            "actual_full_return_pending": True, "run_valid": False, "effective_qualified": False,
            "first_error": None, "secondary_errors": [], "revoked": False,
            "classification": "not_evaluated", **FALSE_MILESTONES}


def charged_cost(spec, mode, root=ROOT):
    """Includes ROOT preparation/closure helpers, not only scientific jobs.

    Administrative costs use flylab.m3.e358.administrative-full-return.v1,
    mode, charged_once=true, cost_id, actual_outer_full_return_s and a descriptor
    actual_tool_return. The linked actual JSON has identical duration. ROOT
    measures and authors these; this adapter does not estimate their costs.
    """
    row = strict_json(read_spec(spec, root))
    require(row["schema"] in ("flylab.m3.e358.actual-full-return.v1",
                              "flylab.m3.e358.administrative-full-return.v1")
            and row["mode"] == mode and row["charged_once"] is True, "prior once-charged cost")
    cost = numeric(row["actual_outer_full_return_s"], "actual prior full return", minimum=1e-12)
    actual = strict_json(read_spec(row["actual_tool_return"], root))
    require(actual["actual_outer_full_return_s"] == cost, "external actual cost linkage")
    require(type(row["cost_id"]) is str and row["cost_id"], "unique actual cost identity")
    return row


def validate_qualification(spec, sources, root=ROOT):
    q = strict_json(read_spec(spec, root))
    require(q["schema"] == "flylab.m3.e358.adapter-qualification.v1"
            and q["source_sha256"] == own_sources(root)
            and q["original_calls"] == 0 and type(q["original_calls"]) is int
            and q["original_array_reads"] == 0 and q["closed"] is True,
            "current original0 adapter Q")
    for key in ("suite_exit", "fixture_exit"):
        row = closed_exit(q[key], root=root)
        require(row["mode"] == "generated" and row["role"] == key.removesuffix("_exit"), "generated-only new Q exact roles")
        value = strict_json(read_spec(row["envelope"], root))
        require(type(value["original_array_reads"]) is int and value["original_array_reads"] == 0, "actual fixture/suite original reads0")
        if key == "suite_exit":
            require(value["case_receipt"] == q["case_receipt"], "same actual suite receipt linkage")
    receipt = strict_json(read_spec(q["case_receipt"], root))
    validate_case_receipt(receipt, root=root)
    case = strict_json(read_spec(receipt["case_contract"], root))
    require(case["source_sha256"] == own_sources(root)
            and case["expected_functions"] == receipt["expected_functions"], "current controlled case contract")
    require(receipt["source_sha256"] == own_sources(root), "current case sources")
    require(all(sources.get(k) == v for k, v in own_sources(root).items()), "Q own closure")
    return q


def validate_contract(name, *, root=ROOT, child=False):
    path = safe(name, root)
    require(path.parent == safe(BASE, root) and path.name.startswith("parent-e346-e358-")
            and path.suffix == ".json", "E346-compatible fresh E358 contract path")
    raw = path.read_bytes()
    require(len(raw) <= 256 << 10, "contract metadata cap")
    c = strict_json(raw)
    require(type(c) is dict and set(c) == CONTRACT_KEYS, "literal E358 contract schema")
    require(c["schema"] == "flylab.m3.e358.supervisor-contract.v1" and c["mode"] in ("original", "generated"),
            "contract mode/schema")
    original = c["mode"] == "original"
    reg = registration(root) if original else None
    role = c["role"]
    require(role in (set(reg["roles"]) if original else {"fixture", "suite"}), "typed explicit role")
    require(type(c["attempt"]) is int and c["attempt"] in (1, 2), "typed bounded attempt")
    require((role == "candidate_retry") == (c["attempt"] == 2) or role in ("candidate_audit", "body", "body_audit"),
            "role/attempt match")
    paths = names(c["attempt_id"], role)
    require(c["paths"] == paths and name == paths["contract"] and c["phase"] == "E358-" + role,
            "literal exact role paths")
    expected_limits = reg["roles"][role] if reg else SUITE_LIMITS if role == "suite" else FIXTURE_LIMITS
    require(c["limits"] == expected_limits and all(type(c["limits"][k]) is type(v)
            for k, v in expected_limits.items()), "exact typed role policy")
    require(type(c["original_calls"]) is int and c["original_calls"] == int(original)
            and type(c["cpu_threads"]) is int and c["cpu_threads"] == 1,
            "literal original count/CPU thread1")
    argv = [PYTHON, "-X", "utf8", "-B", OWNED[0], "--child", name,
            "--pre-spawn", paths["pre_spawn"]]
    require(c["commands"] == [{"argv": argv, "stdout": paths["stdout"], "stderr": paths["stderr"],
                               "wall_s": expected_limits["child_s"]}]
            and type(c["commands"][0]["wall_s"]) is type(expected_limits["child_s"])
            and c["supervisor_output"] == paths["supervision"]
            and type(c["max_tree_rss_bytes"]) is int and c["max_tree_rss_bytes"] == expected_limits["host_bytes"],
            "exact sole job argv/wall/host/output")
    for key in ("cumulative_before_s", "cumulative_limit_s", "stage_used_s", "stage_limit_s", "reserved_s"):
        numeric(c[key], key)
    require(c["cumulative_limit_s"] == (10200 if original else 150)
            and c["stage_limit_s"] == (300 if original else 150 - GENERATED_BASELINE)
            and c["reserved_s"] == expected_limits["outer_s"]
            and type(c["cumulative_limit_s"]) is int
            and type(c["stage_limit_s"]) is (int if original else float)
            and type(c["reserved_s"]) is type(expected_limits["outer_s"]),
            "fixed typed cumulative/stage/reservation")
    prior = [charged_cost(v, c["mode"], root) for v in c["prior_exits"]]
    require(len({v["cost_id"] for v in prior}) == len(prior), "cost charged once, no replay")
    used = sum(v["actual_outer_full_return_s"] for v in prior)
    require(c["stage_used_s"] == used and type(c["stage_used_s"]) is type(used)
            and c["cumulative_before_s"] == (BASELINE if original else GENERATED_BASELINE) + used
            and type(c["cumulative_before_s"]) is float,
            "all actual preparation/execution/closure costs included")
    pending = c["pending_reservations"]
    require(type(pending) is dict, "typed downstream reservations")
    allowed_pending = {"candidate_audit_first": 8, "candidate_retry": 75,
                       "candidate_audit_retry": 8, "body": 130, "body_audit": 12}
    for key, value in pending.items():
        require(key in allowed_pending and type(value) is int and value == allowed_pending[key],
                "exact pending role reservation")
    if original:
        required = ({"candidate_audit_first", "body", "body_audit"} if role == "candidate_first"
                    else {"candidate_audit_retry", "body", "body_audit"} if role == "candidate_retry"
                    else {"body", "body_audit"} if role == "candidate_audit"
                    else {"body_audit"} if role == "body" else set())
        require(required <= set(pending), "complete dependent phase admission reserve")
    held = c["reserved_s"] + sum(pending.values())
    require(used + held <= c["stage_limit_s"] and c["cumulative_before_s"] + held <= c["cumulative_limit_s"],
            "atomic remaining cumulative and stage reservation")
    require(type(c["retained_reserve_bytes"]) is int
            and c["retained_reserve_bytes"] == min(1 << 20, expected_limits["output_cap_bytes"] // 4),
            "metadata/failure/outer report reserve")
    expected_sources = closure_sources(reg, root) if reg else own_sources(root)
    require(c["source_sha256"] == expected_sources, "closed current reviewed source roster")
    check_sources(expected_sources, root)
    require(c["registration"] == (file_spec(REGISTRATION, root) if reg else None), "reviewed fixed registration descriptor")
    frozen = None
    if original:
        require(c["science_freeze"] == reg["science_freeze"], "immutable science F descriptor")
        frozen = passive_science(reg, root)
        validate_qualification(c["adapter_qualification"], expected_sources, root)
    else:
        require(c["science_freeze"] is None and c["adapter_qualification"] is None
                and c["retry_evidence"] is None and c["attempt"] == 1, "fixture original authority absent")
    require(type(c["inputs"]) is dict, "typed role input descriptors")
    required_inputs = {"candidate_first": set(), "candidate_retry": set(),
                       "candidate_audit": {"worker_exit"},
                       "body": {"worker_exit", "candidate_audit_exit"},
                       "body_audit": {"worker_exit", "candidate_audit_exit", "body_exit"},
                       "fixture": set(), "suite": {"case_contract"}}
    require(set(c["inputs"]) == required_inputs[role], "finite role inputs, old candidate forbidden")
    for key, spec in c["inputs"].items():
        if key.endswith("_exit"):
            dep = closed_exit(spec, root=root)
            require(dep["attempt_id"] == c["attempt_id"] and dep["attempt"] == c["attempt"], "same current attempt handoff")
            expected_role = {"worker_exit": {"candidate_first", "candidate_retry"},
                             "candidate_audit_exit": {"candidate_audit"}, "body_exit": {"body"}}[key]
            require(dep["role"] in expected_role and dep["mode"] == "original", "dependency role")
            require(spec in c["prior_exits"], "dependency already actual charged")
        else:
            read_spec(spec, root)
    scientific_prior = [v for v in prior if v["schema"] == "flylab.m3.e358.actual-full-return.v1"]
    if original:
        worker_count = sum(v["role"] in ("candidate_first", "candidate_retry") for v in scientific_prior)
        require(worker_count <= (1 if c["attempt"] == 1 else 2), "candidate attempt count")
        if role == "candidate_first":
            require(worker_count == 0, "first candidate only once")
        if role == "candidate_retry":
            require(worker_count == 1 and not any(v["role"] == "candidate_retry" for v in scientific_prior), "one modified retry only")
        if role in ("candidate_audit", "body_audit"):
            require(not any(v["role"] == role and v["attempt"] == c["attempt"] for v in scientific_prior), "one current independent audit only")
        require(sum(v["role"] == "body" for v in scientific_prior) <= (1 if role == "body_audit" else 0),
                "no second body authorization")
        if c["attempt"] == 2:
            evidence = strict_json(read_spec(c["retry_evidence"], root))
            first = [v for v in scientific_prior if v["role"] == "candidate_first"]
            require(len(first) == 1 and evidence["first_cost_id"] == first[0]["cost_id"]
                    and evidence["change"] == "science45_to65_outer55_to75"
                    and evidence["math_source_changed"] is False
                    and evidence["same_backward_or_candidate_deadline"] is True
                    and type(evidence["phase_work_evidence"]) is dict and evidence["phase_work_evidence"],
                    "one evidence-based timing modified retry")
        else:
            require(c["retry_evidence"] is None, "no unused retry authority")
    freshness = [paths[k] for k in ("output", "outer", "outer_revocation", "actual_return", "full_exit")]
    if not child:
        freshness += [paths[k] for k in ("pre_spawn", "supervision", "stdout", "stderr", "witness_stdout", "witness_stderr")]
        target = safe(paths["supervision"], root)
        freshness += [str(target.with_name(target.stem + suffix).relative_to(root).as_posix())
                      for suffix in ("-identities.json", "-observer.stdout.txt", "-observer.stderr.txt")]
    require(all(not safe(v, root).exists() for v in freshness), "fresh outputs and no replay")
    return c, descriptor(name, raw), frozen


def pre_spawn(name, entry, *, root=ROOT):
    c, spec, frozen = validate_contract(name, root=root)
    timer = Clock(entry, c["limits"])
    timer.check("outer")
    import psutil
    owner = psutil.Process(os.getpid())
    receipt = {"schema": "flylab.m3.e358.actual-pre-spawn.v1", "contract": spec,
               "entry_perf_counter": entry, "parent_pid": owner.pid,
               "parent_create_time": owner.create_time(), "source_sha256": c["source_sha256"],
               "role": c["role"], "attempt_id": c["attempt_id"], "attempt": c["attempt"],
               "original_array_reads": 0, "scientific_native_imports": 0,
               "Popen_before_receipt": 0, "reservation_held": True,
               "cumulative_before_s": c["cumulative_before_s"], "cumulative_after_s": None,
               "cumulative_limit_s": c["cumulative_limit_s"], "reserved_s": c["reserved_s"],
               "pending_reservations": c["pending_reservations"], "limits": c["limits"],
               "stage_used_s": c["stage_used_s"], "stage_limit_s": c["stage_limit_s"],
               "original_calls": c["original_calls"], "max_tree_rss_bytes": c["max_tree_rss_bytes"],
               "retained_reserve_bytes": c["retained_reserve_bytes"], "cpu_threads": c["cpu_threads"]}
    require(not any(k == "torch" or k == "numpy" or k.startswith("flylab.") for k in sys.modules),
            "pre-spawn numeric/native imports0")
    durable(safe(c["paths"]["pre_spawn"], root), json_bytes(receipt), cap=256 << 10,
            check=lambda: timer.check("outer"))
    return c, spec, file_spec(c["paths"]["pre_spawn"], root), frozen


def pre_array(name, receipt_name, *, root=ROOT):
    c, spec, frozen = validate_contract(name, root=root, child=True)
    require(receipt_name == c["paths"]["pre_spawn"], "same actual pre-spawn receipt path")
    receipt_spec = file_spec(receipt_name, root)
    receipt = strict_json(read_spec(receipt_spec, root))
    expected = {"schema": "flylab.m3.e358.actual-pre-spawn.v1", "contract": spec,
                "source_sha256": c["source_sha256"], "reservation_held": True,
                "original_array_reads": 0, "scientific_native_imports": 0,
                "Popen_before_receipt": 0, "cumulative_after_s": None}
    for key in ("role", "attempt_id", "attempt", "cumulative_before_s", "cumulative_limit_s",
                "reserved_s", "pending_reservations", "limits", "stage_used_s", "stage_limit_s",
                "original_calls", "max_tree_rss_bytes", "retained_reserve_bytes", "cpu_threads"):
        expected[key] = c[key]
    require(set(receipt) == set(expected) | {"entry_perf_counter", "parent_pid", "parent_create_time"},
            "literal same-ENTRY receipt authority fields")
    for key, value in expected.items():
        typed_equal(receipt[key], value, "same-ENTRY pre-array authority " + key)
    entry = numeric(receipt["entry_perf_counter"], "ENTRY", minimum=1e-12)
    import psutil
    require(type(entry) is float and type(receipt["parent_pid"]) is int and receipt["parent_pid"] > 0
            and type(receipt["parent_create_time"]) is float,
            "typed actual parent identity")
    numeric(receipt["parent_create_time"], "parent creation time", minimum=1e-12)
    typed_equal(psutil.Process(receipt["parent_pid"]).create_time(), receipt["parent_create_time"],
            "actual live parent PID/create-time identity")
    require(not any(k == "torch" or k == "numpy" or k.startswith("flylab.") for k in sys.modules),
            "pre-array numeric/native imports0")
    timer = Clock(entry, c["limits"])
    timer.check("science")
    return c, spec, receipt_spec, frozen, timer


def closed_exit(spec, *, root=ROOT):
    row = charged_cost(spec, strict_json(read_spec(spec, root))["mode"], root)
    require(row["schema"] == "flylab.m3.e358.actual-full-return.v1" and row["qualified"] is True
            and row["exit_code"] == 0 and type(row["exit_code"]) is int
            and row["all_owned_ended"] is True and row["known_identity_cleanup"] == []
            and row["first_error"] is None and row["secondary_errors"] == []
            and row["actual_full_return_pending"] is False, "natural actual full-return closure")
    c = strict_json(read_spec(row["contract"], root))
    require(row["role"] == c["role"] and row["attempt_id"] == c["attempt_id"]
            and row["attempt"] == c["attempt"] and row["source_sha256"] == c["source_sha256"]
            and all(c["source_sha256"].get(k) == v for k, v in own_sources(root).items()),
            "current exact closed candidate/adapter identities")
    outer = strict_json(read_spec(row["outer"], root))
    require(outer["natural_closed"] is True and outer["contract"] == row["contract"]
            and outer["first_error"] is None and outer["secondary_errors"] == []
            and outer["known_identity_cleanup"] == [] and outer["envelope"] == row["envelope"],
            "external closure current outer linkage")
    value = strict_json(read_spec(row["envelope"], root))
    require(value["contract"] == row["contract"] and value["source_sha256"] == row["source_sha256"]
            and value["run_valid"] is True and value["first_error"] is None
            and value["secondary_errors"] == [] and value["revoked"] is False,
            "closed unrevoked envelope")
    folder = safe(c["paths"]["output"], root)
    require(not (folder / "failure.json").exists() and not (folder / "revocation.json").exists()
            and not safe(c["paths"]["outer_revocation"], root).exists()
            and row.get("outer_revocation") is None, "no late revocation")
    return row


def cleanup_matching(identities, process_api, *, timer=None):
    killed, errors = [], []
    for pid, created in reversed(list(identities.items())):
        try:
            if timer is not None:
                require(timer.remaining(OUTER_REPORT_MARGIN_S) > 0, "same-ENTRY cleanup time remaining")
            item = process_api.Process(pid)
            if item.create_time() == created:
                item.kill()
                killed.append({"pid": pid, "create_time": created})
        except process_api.NoSuchProcess:
            pass
        except Exception as error:
            errors.append(error_record(error, "matching-owned-cleanup"))
    return killed, errors


def owned_presence(identities, process_api):
    """A reused PID has ended the recorded identity; unreadable identity is unknown."""
    matching, unknown, errors = {}, {}, []
    for pid, created in identities.items():
        try:
            if process_api.Process(pid).create_time() == created:
                matching[pid] = created
        except process_api.NoSuchProcess:
            pass
        except Exception as error:
            unknown[pid] = created
            errors.append(error_record(error, "owned-identity-completion"))
    return matching, unknown, errors


def bounded_owned_completion(identities, timer, process_api, *, sleep=time.sleep):
    errors = []
    while True:
        matching, unknown, observed_errors = owned_presence(identities, process_api)
        errors.extend(observed_errors)
        if not matching or unknown:
            return matching, unknown, errors
        try:
            remaining = timer.remaining(OUTER_REPORT_MARGIN_S)
            if remaining <= 0:
                raise TimeoutError("E358 same-ENTRY owned completion deadline")
            sleep(min(.005, remaining))
        except Exception as error:
            errors.append(error_record(error, "owned-completion-wait"))
            return matching, unknown, errors


def publish_outer(c, record, timer, *, root=ROOT):
    """Retain a provisional write and a separate late revocation, never overwrite."""
    path = safe(c["paths"]["outer"], root)
    try:
        durable(path, json_bytes(record), cap=256 << 10, check=lambda: timer.check("outer"))
        timer.check("outer")
        require(retained(c, root) + (128 << 10) <= c["limits"]["output_cap_bytes"], "outer actual retained cap")
        timer.check("outer")
    except Exception as error:
        fail(record, error, "outer-publication")
        record["natural_closed"] = False
        # Emergency evidence claims no timely success and keeps the same ENTRY.
        target = c["paths"]["outer_revocation"] if path.exists() else c["paths"]["outer"]
        payload = json_bytes(record)
        require(retained(c, root) + len(payload) <= c["limits"]["output_cap_bytes"], "outer failure retained cap")
        durable(safe(target, root), payload, cap=256 << 10)
    return record


def supervise(name, *, entry=ENTRY, root=ROOT):
    """Observe the immutable observers; kill only PID/create-time owned identities."""
    import psutil
    lock = safe(BASE + "/parent-e358-original-active.lock", root)
    raw_c = strict_json(safe(name, root).read_bytes())
    held = raw_c.get("mode") == "original"
    token = {"pid": os.getpid(), "created": psutil.Process(os.getpid()).create_time(), "contract": name}
    if held:
        durable(lock, json_bytes(token), cap=1024)
    process = None
    identities, killed, secondary, first = {}, [], [], None
    c = spec = receipt = timer = None
    matching, unknown, missing_identities = {}, {}, []
    supervisor_killed = False
    try:
        c, spec, receipt, _ = pre_spawn(name, entry, root=root)
        timer = Clock(entry, c["limits"])
        argv = [PYTHON, "-X", "utf8", "-B", OBSERVER, name]
        with safe(c["paths"]["witness_stdout"], root).open("xb") as out, \
                safe(c["paths"]["witness_stderr"], root).open("xb") as err:
            require(timer.remaining(OUTER_CLEANUP_MARGIN_S) > 0, "outer cleanup/publication margin")
            process = subprocess.Popen(argv, cwd=root, stdout=out, stderr=err,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            owner = psutil.Process(process.pid)
            identities[owner.pid] = owner.create_time()
            while process.poll() is None:
                if timer.remaining(OUTER_CLEANUP_MARGIN_S) <= 0:
                    raise TimeoutError("E358 same-ENTRY outer work deadline with cleanup margin")
                try:
                    tree = [owner, *owner.children(recursive=True)]
                except psutil.NoSuchProcess:
                    tree = []
                rss = 0
                for item in tree:
                    try:
                        created = item.create_time()
                        require(item.pid not in identities or identities[item.pid] == created, "observed PID identity reuse")
                        identities[item.pid] = created
                        rss += item.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                require(rss <= c["limits"]["host_bytes"], "outer owned host tree cap")
                require(retained(c, root) + c["retained_reserve_bytes"] <= c["limits"]["output_cap_bytes"],
                        "outer stdout/report recursive retention")
                time.sleep(min(.005, timer.remaining(OUTER_CLEANUP_MARGIN_S)))
            code = process.returncode
            timer.check("outer")
        supervision = strict_json(safe(c["paths"]["supervision"], root).read_bytes())
        supervisor_killed = any(row["killed"] is not False for row in supervision["attempts"])
        identity_path = safe(c["paths"]["supervision"], root).with_name(
            Path(c["paths"]["supervision"]).stem + "-identities.json")
        identity = strict_json(identity_path.read_bytes())
        # Ingest actual descendant identities even when scientific work failed.
        for key, created in identity["pid_create_times"].items():
            pid = int(key)
            require(type(created) is float and math.isfinite(created) and created > 0,
                    "typed observer creation time")
            require(pid not in identities or identities[pid] == created, "matching observed ownership")
            identities[pid] = created
        missing_identities = sorted(set(supervision["owned_pids"]) - set(identities))
        secondary.extend({**row, "stage": "unchanged-creation-time-observer"} for row in identity["errors"])
        require(not missing_identities, "all observed owned identities known")
        require(supervision["success"] is True and supervision["contract_sha256"] == spec["sha256"]
                and len(supervision["attempts"]) == 1 and supervision["attempts"][0]["argv"] == c["commands"][0]["argv"]
                and supervision["attempts"][0]["killed"] is False and code == 0,
                "actual unchanged supervisor natural job exit")
        require(identity["qualified"] is True and identity["errors"] == []
                and identity["all_owned_pids_ended"] is True and identity["contract"] == spec,
                "actual unchanged creation-time observer closure")
    except Exception as error:
        first = error_record(error, "outer")
        if process is not None and process.pid not in identities:
            missing_identities.append(process.pid)
        if identities and timer is not None:
            killed, errors = cleanup_matching(identities, psutil, timer=timer)
            secondary.extend(errors)
    if c is None:
        if held and process is None:
            require(strict_json(lock.read_bytes()) == token, "unspawned reservation lock identity")
            lock.unlink()
        raise ValueError("E358 pre-spawn rejected: " + str(first))
    matching, unknown, closure_errors = bounded_owned_completion(identities, timer, psutil)
    secondary.extend(closure_errors)
    ended = process is not None and process.poll() is not None and not matching and not unknown and not missing_identities
    if not ended and first is None:
        first = error_record(RuntimeError("owned identities unresolved at outer completion"), "owned-completion")
    if closure_errors and first is None:
        first, secondary = secondary[0], secondary[1:]
    # Scientific failure can naturally close; technical cleanup cannot qualify.
    natural = ended and not killed and not supervisor_killed and not secondary and not missing_identities
    if first is not None and (first["type"] == "TimeoutError" or first["message"] != "actual unchanged supervisor natural job exit"):
        natural = False
    if held and ended:
        try:
            # Last actual identity check before relinquishing the reservation.
            matching, unknown, release_errors = owned_presence(identities, psutil)
            secondary.extend(release_errors)
            require(not matching and not unknown and not release_errors, "owned closure before reservation release")
            require(strict_json(lock.read_bytes()) == token, "owned reservation lock identity")
            lock.unlink()
            held = False
        except Exception as error:
            if first is None:
                first = error_record(error, "reservation-release")
            else:
                secondary.append(error_record(error, "reservation-release"))
            natural = False
            ended = not matching and not unknown and not missing_identities and process.poll() is not None
    output = c["paths"]["output"] + "/envelope.json"
    output_spec = file_spec(output, root) if safe(output, root).exists() else None
    record = {"schema": "flylab.m3.e358.outer-provisional.v1", "contract": spec,
              "pre_spawn": receipt, "envelope": output_spec, "natural_closed": bool(natural),
              "exit_code": process.returncode if process is not None else None,
              "observed_pid_create_times": {str(k): v for k, v in identities.items()},
              "all_owned_ended": bool(ended), "known_identity_cleanup": killed,
              "unchanged_supervisor_killed": supervisor_killed,
              "unresolved_matching_identities": {str(k): v for k, v in matching.items()},
              "unknown_owned_identities": {str(k): v for k, v in unknown.items()},
              "missing_owned_creation_times": missing_identities, "reservation_released": not held,
              "first_error": first, "secondary_errors": secondary,
              "elapsed_before_outer_publication_s": time.perf_counter() - entry,
              "actual_full_return_pending": True, "cumulative_before_s": c["cumulative_before_s"],
              "cumulative_after_s": None, "cumulative_limit_s": c["cumulative_limit_s"], **FALSE_MILESTONES}
    record = publish_outer(c, record, timer, root=root)
    return 0 if record["first_error"] is None and record["secondary_errors"] == [] else 1


def seal_full_return(contract_name, actual_tool_return, *, root=ROOT):
    """ROOT calls after the actual exec return. Returns a record; writes nothing.

    ROOT stores result under fresh E358 .json and debits it exactly once. The
    external return schema is ROOT-owned; required fields are the two actual_*
    names documented above. Its duration must include this entire wrapper call.
    """
    c = strict_json(safe(contract_name, root).read_bytes())
    require(actual_tool_return["path"] == c["paths"]["actual_return"], "literal external actual tool-return path")
    actual = strict_json(read_spec(actual_tool_return, root))
    wall = numeric(actual["actual_outer_full_return_s"], "external full-return", minimum=1e-12)
    code = actual["actual_returned_exit_code"]
    require(type(code) is int, "actual exit code")
    outer_spec = file_spec(c["paths"]["outer"], root)
    outer = strict_json(read_spec(outer_spec, root))
    require(outer["contract"] == file_spec(contract_name, root), "closed contract actual wrapper")
    pending = outer["envelope"]
    value = strict_json(read_spec(pending, root)) if pending else None
    errors = []
    def append_errors(record):
        if record:
            rows = [record.get("first_error"), *record.get("secondary_errors", [])]
            for index, row in enumerate(rows):
                # Snapshot linkage may repeat inherited errors, while repeated
                # actual errors inside a snapshot must all remain visible.
                if row is not None and errors.count(row) < rows[:index + 1].count(row):
                    errors.append(row)
    # Preserve earliest science failure, subsequent scientific/close errors,
    # then outer supervision/publication and actual full-return errors.
    append_errors(value)
    folder = safe(c["paths"]["output"], root)
    for member in ("failure.json", "revocation.json"):
        if (folder / member).exists():
            append_errors(strict_json((folder / member).read_bytes()))
    append_errors(outer)
    revocation_spec = None
    final_outer = outer
    if safe(c["paths"]["outer_revocation"], root).exists():
        revocation_spec = file_spec(c["paths"]["outer_revocation"], root)
        final_outer = strict_json(read_spec(revocation_spec, root))
        require(final_outer["contract"] == outer["contract"] and final_outer["envelope"] == pending,
                "outer revocation exact provisional linkage")
        append_errors(final_outer)
    if code != 0:
        errors.append(error_record(RuntimeError("actual tool returned exit code " + str(code)), "actual-full-return"))
    if wall > c["limits"]["outer_s"]:
        errors.append(error_record(TimeoutError("actual full-return exceeds same-ENTRY outer cap"), "actual-full-return"))
    if c["stage_used_s"] + wall > c["stage_limit_s"] or c["cumulative_before_s"] + wall > c["cumulative_limit_s"]:
        errors.append(error_record(RuntimeError("actual full-return exceeds registered cumulative/stage budget"), "actual-full-return"))
    valid = (code == 0 and wall <= c["limits"]["outer_s"] and outer["natural_closed"] is True
             and outer["all_owned_ended"] is True and outer["known_identity_cleanup"] == []
             and outer["first_error"] is None and outer["secondary_errors"] == [] and value is not None
             and value["run_valid"] is True and value["first_error"] is None and value["secondary_errors"] == []
             and value["revoked"] is False)
    valid &= not (folder / "failure.json").exists() and not (folder / "revocation.json").exists()
    valid &= revocation_spec is None and not errors
    valid &= c["stage_used_s"] + wall <= c["stage_limit_s"]
    valid &= c["cumulative_before_s"] + wall <= c["cumulative_limit_s"]
    valid &= retained(c, root) + (256 << 10) <= c["limits"]["output_cap_bytes"]
    return {"schema": "flylab.m3.e358.actual-full-return.v1", "mode": c["mode"],
            "role": c["role"], "attempt_id": c["attempt_id"], "attempt": c["attempt"],
            "cost_id": c["attempt_id"] + ":" + c["role"], "charged_once": True,
            "contract": file_spec(contract_name, root), "outer": outer_spec,
            "outer_revocation": revocation_spec, "envelope": pending,
            "source_sha256": c["source_sha256"], "original_calls": c["original_calls"],
            "actual_tool_return": actual_tool_return, "actual_outer_full_return_s": wall,
            "exit_code": code, "qualified": bool(valid), "all_owned_ended": outer["all_owned_ended"],
            "known_identity_cleanup": outer["known_identity_cleanup"], "first_error": errors[0] if errors else None,
            "unchanged_supervisor_killed": outer.get("unchanged_supervisor_killed", False),
            "secondary_errors": errors[1:], "natural_closed": final_outer["natural_closed"] is True,
            "unresolved_matching_identities": outer.get("unresolved_matching_identities", {}),
            "unknown_owned_identities": outer.get("unknown_owned_identities", {}),
            "missing_owned_creation_times": outer.get("missing_owned_creation_times", []),
            "reservation_released": outer.get("reservation_released", False), "actual_full_return_pending": False,
            "cumulative_before_s": c["cumulative_before_s"], "cumulative_after_s": c["cumulative_before_s"] + wall,
            "cumulative_limit_s": c["cumulative_limit_s"], "stage_after_s": c["stage_used_s"] + wall,
            "unused_current_reservation_released_s": max(0, c["reserved_s"] - wall)
            if outer["all_owned_ended"] is True and outer.get("reservation_released") is True else 0,
            **FALSE_MILESTONES}


def monitor_check(value, limits, *, closed=False):
    identity, memory = value["device_identity"], value["latest_memory_bytes"]
    require(type(value["sample_count"]) is int and 0 < value["sample_count"] <= limits["sample_cap"]
            and value["sampler_errors"] == [] and value["owned_sampler_pids"] == []
            and type(value["subprocess_calls"]) is int and value["subprocess_calls"] == 0
            and value["state"] == ("CLOSED" if closed else "RUNNING")
            and value["owned_thread_alive"] is (not closed)
            and type(value["shutdown_count"]) is int and value["shutdown_count"] == int(closed),
            "bounded sticky NVML lifecycle/subprocess0")
    require(value["mapping"] == "single physical non-MIG index0"
            and type(identity["physical_device_count"]) is int and identity["physical_device_count"] == 1
            and type(identity["physical_index"]) is int and identity["physical_index"] == 0
            and identity["mig_enabled"] is False and identity["driver"]
            and type(identity["uuid"]) is str and identity["uuid"].startswith("GPU-")
            and identity["handle_identity"] is not None
            and memory["uuid"] == identity["uuid"] and memory["handle_identity"] == identity["handle_identity"]
            and value["CUDA_VISIBLE_DEVICES"] in (None, "0", identity["uuid"]), "physical NVML mapping")
    for key in ("precontext_baseline_bytes", "sampled_total_peak_bytes", "sampled_additional_peak_bytes"):
        require(type(value[key]) is int and value[key] >= 0, "typed observed GPU bytes")
    require(type(memory["total"]) is int and memory["total"] > 0
            and value["precontext_baseline_bytes"] <= memory["total"]
            and value["sampled_total_peak_bytes"] <= memory["total"]
            and value["sampled_additional_peak_bytes"] == max(0, value["sampled_total_peak_bytes"] - value["precontext_baseline_bytes"])
            and value["quantization_allowance_bytes"] == 2 << 20
            and value["sampled_additional_peak_bytes"] + (2 << 20) <= limits["GPU_delta_plus_allowance_cap_bytes"],
            "sampled delta plus2MiB within unchanged role cap")
    return True


def gpu_watch(c):
    # This import is stdlib-only; start precedes native/science/torch imports.
    from tools.banc_m3_nvml_monitor import GPUMemoryWatch
    require("torch" not in sys.modules, "NVML baseline before Torch/native context")
    watch = GPUMemoryWatch(index=0, interval_s=.5)
    try:
        monitor_check(watch.start(), c["limits"])
    except Exception as error:
        closed = watch.close()
        error.add_note("NVML failed-start close: " + json.dumps(normalize(closed), allow_nan=False)[:4096])
        raise
    return watch


def npz_bytes(arrays, *, cap=16 << 20):
    import numpy as np
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    raw = stream.getvalue()
    require(len(raw) <= cap, "bounded new NPZ publication")
    return raw


def candidate(c, result, frozen, timer):
    watch, acquired, ledger = None, {}, []
    try:
        watch = gpu_watch(c)
        from tools import verify_c_banc_m3_reference_parameter_direction as ref
        core = ref.native()
        packet = core.load_packet(ROOT, frozen["legacy_inventory"], timely=lambda: timer.check("science"), ledger=ledger)
        target = ref.load_label(frozen["cadence_label"], ledger, timely=lambda: timer.check("science"))
        phase = ref.PhaseTrace(scope="E358-current-candidate", cuda_events=False)
        def progress(value):
            acquired.update(value)
            phase(value)
            monitor_check(watch.snapshot(), c["limits"])
        raw, receipts, record = ref.collect(packet, target_q=target, timely=lambda: timer.check("science"), progress=progress)
        record = normalize(record)
        require(all(c["source_sha256"].get(k) == v for k, v in record["actual_import_sha256"].items()), "current candidate actual imported source closure")
        closed = watch.close()
        monitor_check(closed, c["limits"], closed=True)
        record["sampler"] = closed
        # Existing science provenance and mathematics are deliberately unchanged.
        published = ref.publish_generated(safe(c["paths"]["output"]), raw, receipts, record,
                                          normalize(ledger), timely=timer.check)
        result.update(run_valid=True, classification="candidate_calculated",
                      science_manifest=file_spec(c["paths"]["output"] + "/manifest.json"),
                      science_result=file_spec(c["paths"]["output"] + "/result.json"),
                      branch=record["branch"], analysis=published["analysis"],
                      sampler=closed, phase_trace=phase.rows,
                      science_source2=published["source_sha256"])
    except Exception as error:
        fail(result, error, acquired.get("stage", "candidate-preload"))
        result["partial_receipts"] = normalize(acquired.get("receipts", {}))
        result["partial_record"] = normalize(acquired.get("record", {}))
        result["partial_stage"] = acquired.get("stage")
        if "phase" in locals():
            result["phase_trace"] = phase.rows
        if acquired.get("raw"):
            try:
                timer.check()
                put(c, "partial-raw.npz", npz_bytes(acquired["raw"]), timer, raw=True)
            except Exception as secondary:
                result["secondary_errors"].append(error_record(secondary, "retaining-partial-raw"))
    finally:
        if watch:
            try:
                result["sampler"] = watch.close()
                monitor_check(result["sampler"], c["limits"], closed=True)
                timer.check()
            except Exception as error:
                fail(result, error, "candidate-NVML-close")


def candidate_raw(exit_spec, frozen, timer, *, admission=False):
    closed = closed_exit(exit_spec)
    outer_value = strict_json(read_spec(closed["envelope"]))
    c = strict_json(read_spec(closed["contract"]))
    folder = safe(c["paths"]["output"])
    from tools import verify_c_banc_m3_reference_parameter_direction as ref
    result = strict_json(read_spec(outer_value["science_result"]))
    manifest = strict_json(read_spec(outer_value["science_manifest"]))
    ref.validate_publication_manifest(manifest)
    require(result["source_sha256"] == ref.own_sources() and result["qualified"] is True,
            "immutable source2 science result")
    for key in ("raw", "result", "receipts", "ledger"):
        spec = manifest[key]
        read_spec({**spec, "path": c["paths"]["output"] + "/" + spec["path"]}, text_only=key != "raw")
        timer.check("science")
    raw = ref.read_completed_raw(folder / "raw.npz", result["record"]["branch"], timely=lambda: timer.check("science"))
    analysis = ref.analyze(raw, result["record"])
    require(normalize(analysis) == result["analysis"]
            and ref.array_sha(raw["target_q"]) == result["target_c_order_sha256"]
            and ref.array_sha(raw["normalization"]) == result["normalization_c_order_sha256"],
            "same literal raw R/gq/O/Q_exact/T and target/norm")
    ledger = strict_json((folder / "selected-ledger.json").read_bytes())
    require(len(ledger) == 24 and ledger[-1]["path"] == frozen["cadence_label"]["path"]
            and ledger[-1]["slice_c_order_sha256"] == ref.array_sha(raw["target_q"])
            and ledger[0]["selected"]["normalization"]["c_order_sha256"] == ref.array_sha(raw["normalization"]),
            "actual selected ledger target and normalization")
    roles = frozen["legacy_inventory"]["role_order"]
    require(len(roles) == 23, "immutable selected23 roster")
    for item, role in zip(ledger[:-1], roles, strict=True):
        binding = frozen["legacy_inventory"]["bindings"][role]
        schema = binding["selected_schema"]
        require(item["path"] == binding["path"] and item["expected_sha256"] == binding["sha256"]
                and item["posthash_verified"] is True and set(item["selected"]) == set(schema)
                and len(item["numeric_reads"]) == len(schema) and set(item["numeric_reads"]) == set(schema),
                "closed literal selected ledger role/source")
        for key, field in schema.items():
            observed = item["selected"][key]
            require(observed["shape"] == field["shape"]
                    and observed["dtype"] == ref.native().dtype_metadata(field["dtype"])
                    and re.fullmatch("[0-9a-f]{64}", observed["c_order_sha256"]), "typed selected relationship")
    require(ledger[-1]["sha256"] == frozen["cadence_label"]["sha256"]
            and set(ledger[-1]["numeric_reads"]) == set(frozen["cadence_label"]["selected_members"])
            and ledger[-1]["extras_payload_header_reads"] == 0, "closed current label selection")
    receipts = strict_json((folder / "receipts.json").read_bytes())
    record = result["record"]
    require(ref.native().work_counts(receipts) == record["work"]
            and record["neural_backward_requested"] == record["neural_backward_returned"] == 4
            and record["all_neutral_reproduced_before_gradient"] is True,
            "actual immutable requested/returned work")
    for receipt in receipts.values():
        require(type(receipt["confirm_requested"]) is int
                and receipt["confirm_requested"] == receipt["confirm_returned"] > 0,
                "actual returned confirmation")
        for confirmed, returned in (("forward_controls", "forward_controls_returned"),
                                    ("forward_neural_ticks", "forward_neural_ticks_returned"),
                                    ("recompute_controls", "recompute_controls_returned"),
                                    ("recompute_neural_ticks", "recompute_neural_ticks_returned"),
                                    ("backward_blocks", "backward_blocks_returned"),
                                    ("backward_block_ticks", "backward_block_ticks_returned")):
            require(type(receipt[returned]) is int and receipt["confirmed"][confirmed] == receipt[returned],
                    "saved actual device-confirmed native work")
    identity = {"worker_exit": exit_spec, "raw": {**manifest["raw"], "path": c["paths"]["output"] + "/raw.npz"},
                "theta_sha256": ref.array_sha(raw["theta_actual"]) if "theta_actual" in raw else None,
                "selected_sha256": ref.array_sha(raw["selected_weights_candidate"]) if "selected_weights_candidate" in raw else None,
                "branch": record["branch"], "analysis": normalize(analysis)}
    if admission:
        import numpy as np
        theta = raw["theta_actual"]
        require(record["branch"] == "normal" and analysis["H356"] == "conditional_support"
                and np.any(theta != 1) and np.all(theta > 0)
                and np.array_equal(theta, theta.astype(np.float32).astype(np.float64))
                and record["palette"]["native_deployable"] is True
                and type(record["palette"]["full_native_palette_count"]) is int
                and 0 < record["palette"]["full_native_palette_count"] <= 16384,
                "same supported nonzero actualf32 candidate")
    return raw, record, identity


def candidate_audit(c, result, frozen, timer):
    raw, record, identity = candidate_raw(c["inputs"]["worker_exit"], frozen, timer)
    result.update(run_valid=True, classification="independent_raw_audit", candidate=identity,
                  independent_original_deployability_recomputed=False,
                  full_parameter_Jacobian_independently_proven=False)


def admitted_candidate(c, frozen, timer):
    audited = closed_exit(c["inputs"]["candidate_audit_exit"])
    value = strict_json(read_spec(audited["envelope"]))
    raw, record, identity = candidate_raw(c["inputs"]["worker_exit"], frozen, timer, admission=True)
    require(value["candidate"] == identity, "same independently audited candidate bytes")
    return raw, record, identity


def body_data(raw, inventory, timer, *, b):
    """Finite E345 graph/whole/mask/reference/startup; never old audited_candidate."""
    from tools import verify_c_banc_m3_reference_parameter_direction as ref
    core, ledger = ref.native(), []
    selected = inventory["selected_archives"]
    require(set(selected) == {"recurrent_mask", "reference", "startup", "audited_candidate"}, "finite body roles")
    rejected = selected["audited_candidate"]["path"]
    role_paths = {selected[k]["path"] for k in ("recurrent_mask", "reference", "startup")}
    allowed = {name: spec for name, spec in inventory["files"].items() if name != rejected}
    require(len(allowed) == 134 and rejected not in role_paths, "whole124/graph7/selected3 finite roster")
    for name, spec in allowed.items():
        timer.check("science")
        path = safe(name)
        require(path.stat().st_size == spec["bytes"] and core.digest(path, lambda: timer.check("science")) == spec["sha256"],
                "supervised original body input bytes/hash")
        if path.suffix == ".npy":
            with path.open("rb") as stream:
                core.strict_npy_header(stream)
        ledger.append({"path": name, "expected_sha256": spec["sha256"], "bytes": spec["bytes"], "role": spec.get("role")})
    dependencies = strict_json(read_spec(registration()["body_native_dependencies"]))
    require(dependencies["network_asset_download_permitted"] is False
            and dependencies["mesh_cache_complete_currently"] is True, "registered finite offline constructor assets")
    for name, spec in dependencies["native_constructor_asset_bytes"].items():
        timer.check("science")
        path = Path(name)
        require(path.is_absolute() and path.is_file() and not path.is_symlink()
                and path.stat().st_size == spec["bytes"]
                and core.digest(path, lambda: timer.check("science")) == spec["sha256"], "supervised finite native constructor asset bytes")
    b.whole_metadata(ROOT, inventory)
    api = b.resolve_original_api()
    graph = api["GraphStore"].load(safe(inventory["graph_bundle"]))
    saved = api["StateStore"].load(safe(inventory["initial_whole"]), max_files=123)
    require(graph.hash == inventory["graph_hash"], "same full source graph identity")
    api["recovery"].checked_checkpoint(saved)
    arrays = {role: core.load_selected(ROOT, selected[role], ledger, timely=lambda: timer.check("science"), generated=False)
              for role in ("recurrent_mask", "reference", "startup")}
    return {"graph": graph, "saved": saved, "inventory": inventory, "ledger": ledger,
            "theta": raw["theta_actual"].copy(), "selected_weights": raw["selected_weights_candidate"].copy(),
            "positions": arrays["recurrent_mask"]["positions"], "groups": arrays["recurrent_mask"]["edge_parameters"],
            "reference": arrays["reference"], "startup": arrays["startup"]}


def initial_body(session, data, candidate_identity, record, timer, *, b, factory):
    import numpy as np
    observed = session.snapshot()
    factory.efficacy.checked_checkpoint(observed)
    inverse = factory.efficacy._source_representation(observed, observed[factory.efficacy.PATCH_KEY])
    factory.recovery.require_saved_whole_equal(inverse, data["saved"], "E358 inverse whole exact including wall")
    require(session.control_tick == session.network.tick == 0 and abs(session.body.physics_time()) <= 2e-8
            and session.graph.hash == data["inventory"]["graph_hash"]
            and session.adapter.identity == data["inventory"]["source_adapter_identity"]
            and session.network.identity != data["inventory"]["source_network_identity"]
            and session.network.interval_steps == 20 and session.neural_steps_per_control == 20,
            "source/decoder/current candidate/control0 identities")
    proof = b.verify_transfer(session.network, data["source_csr"], data["positions"], data["groups"],
                              data["theta"], data["selected_weights"],
                              expected_palette=record["palette"]["full_native_palette_count"],
                              timely=lambda: timer.check("science"))
    require(proof["selected_sha256"] == candidate_identity["selected_sha256"]
            and proof["candidate_CSR_data_sha256"] == record["palette"]["full_data_c_order_sha256"]
            and proof["candidate_CSR_data_sha256"] != proof["source_CSR_data_sha256"], "current CSR distinct/source selected/full digest")
    _, boundary = b.capture(session)
    target_tendon = data["saved"]["adapter"]["original"]["original"]["original"]["tendon_activation"]
    require(b.e.exact(boundary["command"], np.asarray(data["saved"]["body"]["action"]["joint_angles"], np.float64))
            and b.e.exact(boundary["tendon"], target_tendon), "source control0 command/tendon exact")
    require(session._cpg_digest() == data["saved"]["cpg_hash"] == session.cpg_hash, "source CPG unchanged at transfer")
    proof.update(candidate=candidate_identity, source_inverse_whole_exact=True,
                 source_boundary_command_and_tendon_exact=True,
                 source_network_identity=data["inventory"]["source_network_identity"],
                 source_adapter_identity=session.adapter.identity, graph_hash=session.graph.hash,
                 theta_sha256=b.array_sha(data["theta"]), cpg_hash=session.cpg_hash)
    return proof


def verify_work(counters, construction, count, *, full=False):
    expected = {"body_constructors_requested": 1, "body_constructors_returned": 1,
                "settling_physics_requested": 2000, "settling_physics_host_returned": 2000,
                "neural_setup_requested": 320, "neural_setup_host_returned": 320,
                "neural_capture_requested": 160, "neural_capture_recorded": 160}
    require(all(type(construction[k]) is int and construction[k] == v for k, v in expected.items())
            and construction["unknown_work"] is False and construction["device_confirmed"] is True,
            "constructor exact requested/returned/device-confirmed work")
    require(type(count) is int and 0 <= count <= 300 and (not full or count == 300), "known completed controls")
    values = {"controls_requested": count, "controls_host_returned": count, "controls_completed": count,
              "neural_requested": count * 20, "neural_host_returned": count * 20,
              "neural_device_confirmed": count * 20, "physics_requested": count * 50,
              "physics_host_returned": count * 50, "physics_device_confirmed": count * 50,
              "observer_samples": count * 50, "completed_rows": count + 1}
    require(all(type(counters[k]) is int and counters[k] == v for k, v in values.items())
            and counters["neural_unreturned_unknown"] is False
            and counters["physics_unreturned_unknown"] is False,
            "typed exact controls/neural/native/rows, unknown work forbidden")
    return True


def finish_owned(observer, session, result, timer):
    """Always attempt detach then close current owner, preserving both errors."""
    for stage, callback in (("observer-detach", observer.close if observer else None),
                            ("owned-body-close", session.close if session else None)):
        if callback:
            try:
                callback()
                timer.check()
            except Exception as error:
                fail(result, error, stage)


def screen(metrics, startup):
    summary = metrics["summary"]
    gates = {"startup_four": startup.get("passed") is True,
             "forward": summary["forward_mm"] >= 1.6790085779712631,
             "slip": summary["slip_mm_s"] is not None and summary["slip_mm_s"] <= 1.8026436968249955,
             "bout": summary["longest_continuous_forward_bout_s"] >= .3,
             "all_six_cycles": all(type(summary["complete_cycles"].get(k)) is int
                                   and summary["complete_cycles"][k] > 0 for k in ("lf", "lm", "lh", "rf", "rm", "rh")),
             "native_health": summary["health"]["healthy"] is True}
    return gates, all(gates.values())


def body(c, result, frozen, timer):
    watch = session = observer = factory = None
    controls, rows, samples, construction, counters, data = [], [], [], {}, {}, None
    reason, stage = "duration", "body-preload"
    try:
        watch = gpu_watch(c)
        raw, record, identity = admitted_candidate(c, frozen, timer)
        from tools import verify_c_banc_m3_common_descent_body as b
        inventory = strict_json(read_spec(registration()["body_inventory"]))
        data = body_data(raw, inventory, timer, b=b)
        result["candidate"] = identity
        counters = b.new_counters()
        factory = b.OriginalFactory()
        stage = "construct"
        session = factory.construct(data, construction, lambda: timer.check("science"))
        stage = "initial-transfer"
        proof = initial_body(session, data, identity, record, timer, b=b, factory=factory)
        put(c, "transfer-chunks.json", proof, timer)
        result["transfer"] = proof
        metadata = b.trace_metadata(session)
        row, boundary = b.capture(session)
        rows.append(row)
        controls.append(boundary)
        counters["completed_rows"] = 1
        observer = b.LiveObserver(session, counters, samples, timely=lambda: timer.check("science"),
                                  install_contact=factory.install_contact, synchronize=factory.synchronize).install()
        for control in range(1, 301):
            stage = "control-" + str(control)
            timer.check("science")
            counters["controls_requested"] += 1
            # Any unreturned advance is unknown, even a native health exception.
            try:
                session.advance(1, capture=False)
            except Exception:
                counters["advance_unreturned_unknown"] = True
                raise
            counters["controls_host_returned"] += 1
            factory.synchronize()
            counters["physics_device_confirmed"] = counters["physics_host_returned"]
            row, boundary = b.capture(session)
            rows.append(row)
            controls.append(boundary)
            require(session.control_tick == control and session.network.tick == 20 * control
                    and len(samples) == 50 * control and abs(session.body.physics_time() - .005 * control) <= 2e-8
                    and session._cpg_digest() == proof["cpg_hash"], "actual sensory feedback clocks and CPG")
            counters["controls_completed"] += 1
            counters["completed_rows"] = len(controls)
            counters["observer_samples"] = len(samples)
            if control == 99:
                import numpy as np
                result["startup"] = b.startup99(np.stack([v["command"] for v in controls]),
                                                np.stack([v["tendon"] for v in controls]), **data["startup"])
                if not result["startup"]["passed"]:
                    reason = "startup_rejected"
                    break
            monitor_check(watch.snapshot(), c["limits"])
        verify_work(counters, construction, counters["controls_completed"], full=reason == "duration")
        require(counters.get("advance_unreturned_unknown", False) is False, "known advance work")
        result.update(run_valid=True, reason=reason, classification="scientific_rejected")
    except Exception as error:
        fail(result, error, stage)
    finally:
        if session and factory:
            try:
                factory.synchronize()
                counters["physics_device_confirmed"] = counters["physics_host_returned"]
            except Exception as error:
                fail(result, error, "final-body-confirmation")
        finish_owned(observer, session, result, timer)
    result.update(counters=counters, construction=construction)
    if controls and data:
        try:
            stage = "body-raw-publication"
            control_raw = b.control_arrays(controls, session, data)
            native = b.native_arrays(samples)
            b.validate_native(native)
            put(c, "control-raw.npz", npz_bytes(control_raw, cap=8 << 20), timer, raw=True)
            put(c, "native-contact.npz", npz_bytes(native, cap=8 << 20), timer, raw=True)
            put(c, "trace.json", b.lean_trace(rows, metadata), timer)
            evaluation = {"reference." + k: v for k, v in data["reference"].items()}
            evaluation.update({"startup." + k: v for k, v in data["startup"].items()})
            put(c, "evaluation-inputs.npz", npz_bytes(evaluation, cap=1 << 20), timer, raw=True)
            if result["run_valid"] and reason == "duration":
                metrics = b.evaluate(safe(c["paths"]["output"]), control_raw, native, data, result["startup"])
                gates, passed = screen(metrics, result["startup"])
                result.update(metrics=metrics, gates=gates,
                              classification="screen_accepted_pending_audit" if passed else "scientific_rejected",
                              legacy_smooth_shape_diagnostics={"smooth": metrics["smooth"], "shape": metrics["shape"]})
            result["cpg_unchanged"] = bool(session._cpg_digest() == result["transfer"]["cpg_hash"])
            require(result["cpg_unchanged"], "CPG exact through owned close")
            for item in data["ledger"]:
                timer.check()
                require(b.e.digest(safe(item["path"]), timer.check) == item["expected_sha256"], "body selected posthash")
            put(c, "selected-ledger.json", {"inputs": normalize(data["ledger"]), "candidate": result["candidate"]}, timer)
        except Exception as error:
            fail(result, error, stage)
    if watch:
        try:
            result["sampler"] = watch.close()
            monitor_check(result["sampler"], c["limits"], closed=True)
            timer.check()
        except Exception as error:
            fail(result, error, "body-NVML-close")
    if data:
        try:
            actual = b.actual_imports()
            result["actual_import_sha256"] = actual
            result["source_closure_mismatches"] = [
                {"path": k, "actual_sha256": v, "registered_sha256": c["source_sha256"].get(k)}
                for k, v in actual.items() if c["source_sha256"].get(k) != v
            ]
            require(not result["source_closure_mismatches"], "actual body imported source closure")
            put(c, "body-report.json", result, timer)
            roster = {"control-raw.npz", "native-contact.npz", "trace.json", "evaluation-inputs.npz",
                      "transfer-chunks.json", "selected-ledger.json", "body-report.json"}
            require(all(safe(c["paths"]["output"] + "/" + name).exists() for name in roster), "literal closed body member roster")
            files = {name: file_spec(c["paths"]["output"] + "/" + name) for name in sorted(roster)}
            result["body_manifest"] = put(c, "body-manifest.json", {"schema": "flylab.m3.e358.body-manifest.v1",
                                                                      "files": files, "all_fsynced_and_closed": True}, timer)
        except Exception as error:
            fail(result, error, "body-manifest-publication")


def load_literal_npz(spec, fields, timer, *, cap=8 << 20):
    """All exact member headers pass before the first numeric decode."""
    import numpy as np
    from tools import verify_c_banc_m3_reference_parameter_direction as ref
    raw_bytes = read_spec(spec, text_only=False)
    require(len(raw_bytes) <= cap, "audit archive bound")
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
        infos = archive.infolist()
        require(len(infos) == len(fields) and len({v.filename for v in infos}) == len(infos)
                and {v.filename for v in infos} == {k + ".npy" for k in fields}, "literal audit NPZ roster")
        for info in infos:
            timer.check("science")
            key = info.filename[:-4]
            shape, dtype, finite = fields[key]
            require(not info.is_dir() and not info.flag_bits & 1 and 0 < info.file_size <= cap, "bounded unencrypted member")
            with archive.open(info) as stream:
                header = ref.native().strict_npy_header(stream)
            require(header["shape"] == list(shape) and header["dtype"] == np.dtype(dtype).str
                    and header["fortran_order"] is False and not np.dtype(dtype).hasobject
                    and info.file_size == header["header_bytes"] + math.prod(shape) * np.dtype(dtype).itemsize,
                    "literal exact saved raw header")
        arrays = {v.filename[:-4]: np.load(io.BytesIO(archive.read(v)), allow_pickle=False) for v in infos}
    for k, v in arrays.items():
        require(not fields[k][2] or np.isfinite(v).all(), "finite saved raw " + k)
    return arrays


def body_audit(c, result, frozen, timer):
    import numpy as np
    from tools import verify_c_banc_m3_common_descent_body as b
    raw_candidate, _, identity = admitted_candidate(c, frozen, timer)
    closed = closed_exit(c["inputs"]["body_exit"])
    previous = strict_json(read_spec(closed["envelope"]))
    require(previous["candidate"] == identity, "same candidate body exit")
    manifest = strict_json(read_spec(previous["body_manifest"]))
    roster = {"control-raw.npz", "native-contact.npz", "trace.json", "evaluation-inputs.npz",
              "transfer-chunks.json", "selected-ledger.json", "body-report.json"}
    require(set(manifest) == {"schema", "files", "all_fsynced_and_closed"}
            and manifest["schema"] == "flylab.m3.e358.body-manifest.v1"
            and manifest["all_fsynced_and_closed"] is True and set(manifest["files"]) == roster, "literal body manifest")
    body_contract = strict_json(read_spec(closed["contract"]))
    files = manifest["files"]
    for name, spec in files.items():
        require(spec["path"] == body_contract["paths"]["output"] + "/" + name, "same body member paths")
        read_spec(spec, text_only=not name.endswith(".npz"))
        timer.check("science")
    report = strict_json(read_spec(files["body-report.json"]))
    require(report["candidate"] == identity and report["counters"] == previous["counters"]
            and report["construction"] == previous["construction"] and report["run_valid"] is True,
            "saved body report/envelope counters")
    count = report["counters"]["controls_completed"]
    require((report["reason"] == "duration" and count == 300)
            or (report["reason"] == "startup_rejected" and count == 99), "explicit full screen or startup prefix")
    verify_work(report["counters"], report["construction"], count, full=count == 300)
    require(report["counters"].get("advance_unreturned_unknown", False) is False, "no unknown advance work")
    n, m = count + 1, count * 50
    control_fields = {"control_tick": ((n,), "int64", True), "network_tick": ((n,), "int64", True),
                      "physics_time_s": ((n,), "float64", True), "command": ((n, 42), "float64", True),
                      "tendon": ((n, 6), "float64", True),
                      "theta": (raw_candidate["theta_actual"].shape, "float64", True),
                      "selected_weights": (raw_candidate["selected_weights_candidate"].shape, "float32", True)}
    control = load_literal_npz(files["control-raw.npz"], control_fields, timer)
    native_fields = {"time_s": ((m,), "float64", True), "normal_load_bw": ((m, 6), "float64", True),
                     "slip_mm_s": ((m, 6), "float64", False), "slip_valid": ((m, 6), "bool", True),
                     "nonfoot_contact": ((m,), "bool", True), "upright": ((m,), "float64", True)}
    native = load_literal_npz(files["native-contact.npz"], native_fields, timer)
    require(b.validate_native(native) == m and np.array_equal(control["control_tick"], np.arange(n))
            and np.array_equal(control["network_tick"], np.arange(n) * 20)
            and np.allclose(control["physics_time_s"], np.arange(n) * .005, rtol=0, atol=2e-8)
            and b.e.exact(control["theta"], raw_candidate["theta_actual"])
            and b.e.exact(control["selected_weights"], raw_candidate["selected_weights_candidate"]), "independent saved clocks/current theta/selected bytes")
    inventory = strict_json(read_spec(registration()["body_inventory"]))
    eval_fields = {role + "." + k: (tuple(v["shape"]), v["dtype"], True)
                   for role in ("reference", "startup") for k, v in inventory["selected_archives"][role]["selected_schema"].items()}
    evaluated = load_literal_npz(files["evaluation-inputs.npz"], eval_fields, timer, cap=1 << 20)
    data = {role: {k: evaluated[role + "." + k] for k in inventory["selected_archives"][role]["selected_schema"]}
            for role in ("reference", "startup")}
    ledger = strict_json(read_spec(files["selected-ledger.json"]))
    require(ledger["candidate"] == identity, "same body selected ledger candidate")
    for role in ("reference", "startup"):
        spec = inventory["selected_archives"][role]
        matches = [v for v in ledger["inputs"] if v["path"] == spec["path"] and "selected" in v]
        require(len(matches) == 1, "finite selected audit producer")
        for k, v in data[role].items():
            require(b.array_sha(v) == matches[0]["selected"][k]["c_order_sha256"], "saved evaluation input selected relationship")
    startup = b.startup99(control["command"][:100], control["tendon"][:100], **data["startup"])
    require(startup == report["startup"], "independent exact startup99")
    proof = strict_json(read_spec(files["transfer-chunks.json"]))
    b.verify_chunk_coverage(proof)
    require(proof == report["transfer"] and proof["candidate"] == identity
            and proof["selected_sha256"] == identity["selected_sha256"]
            and proof["theta_sha256"] == identity["theta_sha256"]
            and proof["source_inverse_whole_exact"] is True
            and proof["source_boundary_command_and_tendon_exact"] is True
            and proof["candidate_CSR_data_sha256"] != proof["source_CSR_data_sha256"]
            and type(proof["palette"]) is int and 0 < proof["palette"] <= 16384
            and report["cpg_unchanged"] is True, "saved transfer/source/CPG linkage, no GPU reacquisition")
    trace = strict_json(read_spec(files["trace.json"]))
    require(len(trace["samples"]) == n and trace["model_hash"] == proof["candidate_network_identity"]
            and trace["adapter_hash"] == proof["source_adapter_identity"] and trace["graph_hash"] == proof["graph_hash"],
            "finite samecandidate raw native trace")
    for tick, row in enumerate(trace["samples"]):
        b.lean_row(row, point_count=8)
        require(row["control_tick"] == tick and abs(row["time_s"] - tick * .005) <= 2e-8, "trace exact saved clock")
    metrics, gates, passed = None, None, False
    if count == 300:
        metrics = b.evaluate(safe(body_contract["paths"]["output"]), control, native, data, startup)
        gates, passed = screen(metrics, startup)
        require(normalize(metrics) == report["metrics"] and gates == report["gates"], "independent full physical screen")
    else:
        require(startup["passed"] is False, "partial startup rejection failstop")
    result.update(run_valid=True, classification="screen_accepted" if passed else "scientific_rejected",
                  candidate=identity, body_exit=c["inputs"]["body_exit"], startup=startup, metrics=metrics,
                  gates=gates, controls=count, saved_transfer_digest_audit=True,
                  independent_GPU_reacquisition=False, independent_body_reconstruction=False,
                  later_F1_F8_work="not_executed_requires_separate_measured_reservation")


def test_functions(root=ROOT):
    tree = ast.parse(safe(OWNED[1], root).read_text("utf8"))
    functions = [v.name for v in tree.body if isinstance(v, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and v.name.startswith("test_")]
    require(functions and len(functions) == len(set(functions)), "complete finite test function roster")
    return sorted(functions)


def validate_case_receipt(receipt, *, root=ROOT):
    require(receipt["schema"] == "flylab.m3.e358.complete-pytest-receipt.v1"
            and receipt["source_sha256"] == own_sources(root)
            and receipt["expected_functions"] == test_functions(root)
            and receipt["collected_functions"] == receipt["expected_functions"]
            and receipt["exit_code"] == 0 and receipt["collection_errors"] == []
            and receipt["original_calls"] == receipt["original_array_reads"] == 0
            and receipt["complete_current_cases"] is True, "all actual current test cases")
    require(set(receipt["phases"]) == set(receipt["expected_functions"]), "no hidden test filtering")
    for function, phases in receipt["phases"].items():
        require(set(phases) == {"setup", "call", "teardown"}
                and all(phases[k] == "passed" for k in phases), "every function setup/call/teardown passed: " + function)
    return True


def execute_current_pytest(case_name):
    raw = safe(case_name).read_bytes()
    case = strict_json(raw)
    require(set(case) == {"schema", "source_sha256", "expected_functions", "receipt", "pytest_ini"}
            and case["schema"] == "flylab.m3.e358.current-pytest-case.v1"
            and case["source_sha256"] == own_sources() and case["expected_functions"] == test_functions(),
            "controlled whole current pytest contract")
    ini = safe(case["pytest_ini"])
    require(ini.read_bytes().strip() == b"[pytest]", "empty owned pytest config")
    receipt_path = safe(case["receipt"])
    require(receipt_path.parent == safe(BASE) and receipt_path.name.startswith("parent-e358-")
            and not receipt_path.exists(), "fresh E358 case receipt")
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        os.environ.pop(key, None)
    import pytest
    observed = {"collected": [], "phases": {}, "collection_errors": []}
    class ReceiptPlugin:
        def pytest_collection_finish(self, session):
            for item in session.items:
                require(item.path.resolve() == safe(OWNED[1]).resolve(), "only current owned test file")
                require("[" not in item.name, "explicit unparameterized finite test functions")
                observed["collected"].append(item.name)
            require(sorted(observed["collected"]) == case["expected_functions"], "complete collection, no filter")
        def pytest_collectreport(self, report):
            if report.failed:
                observed["collection_errors"].append(str(report.longrepr)[:2048])
        def pytest_runtest_logreport(self, report):
            function = report.nodeid.split("::")[-1]
            phases = observed["phases"].setdefault(function, {})
            require(report.when not in phases, "one actual test phase")
            phases[report.when] = report.outcome
    code = int(pytest.main(["-q", "-p", "no:cacheprovider", "-c", case["pytest_ini"], OWNED[1]], plugins=[ReceiptPlugin()]))
    value = {"schema": "flylab.m3.e358.complete-pytest-receipt.v1", "source_sha256": own_sources(),
             "case_contract": descriptor(case_name, raw), "expected_functions": case["expected_functions"],
             "collected_functions": sorted(observed["collected"]), "phases": observed["phases"],
             "collection_errors": observed["collection_errors"], "exit_code": code,
             "original_calls": 0, "original_array_reads": 0, "complete_current_cases": code == 0}
    try:
        validate_case_receipt(value)
    except Exception:
        value["complete_current_cases"] = False
        code = code or 2
    durable(receipt_path, json_bytes(value), cap=256 << 10)
    return code


def child(name, receipt_name):
    c, spec, receipt_spec, frozen, timer = pre_array(name, receipt_name)
    result = envelope(c, spec, receipt_spec)
    try:
        if c["mode"] == "generated":
            if c["role"] == "suite":
                case = c["inputs"]["case_contract"]
                require(execute_current_pytest(case["path"]) == 0, "actual whole current pytest passed")
                value = strict_json(read_spec(case))
                result["case_receipt"] = file_spec(value["receipt"])
            else:
                # Literal generated integers, zero scientific imports/arrays.
                put(c, "fixture.json", {"value": [1, 2, 3], "original_array_reads": 0}, timer)
            result.update(run_valid=True, classification="generated_boundary_only", original_array_reads=0)
        elif c["role"] in ("candidate_first", "candidate_retry"):
            candidate(c, result, frozen, timer)
        elif c["role"] == "candidate_audit":
            candidate_audit(c, result, frozen, timer)
        elif c["role"] == "body":
            body(c, result, frozen, timer)
        elif c["role"] == "body_audit":
            body_audit(c, result, frozen, timer)
        timer.check()
        require(retained(c) + c["retained_reserve_bytes"] <= c["limits"]["output_cap_bytes"], "final recursive advance reserve")
    except Exception as error:
        fail(result, error, "child-final")
    try:
        result["elapsed_before_publication_s"] = time.perf_counter() - timer.entry
        put(c, "envelope.json", result, timer, emergency=not result["run_valid"])
        timer.check()
    except Exception as error:
        fail(result, error, "late-envelope-publication")
    if result["first_error"] is not None:
        put(c, "revocation.json", result, timer, emergency=True)
    print(json.dumps({"role": c["role"], "run_valid": result["run_valid"], "classification": result["classification"],
                      "actual_full_return_pending": True, **FALSE_MILESTONES}, allow_nan=False), flush=True)
    return 0 if result["run_valid"] else 1


def main(argv=None):
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--supervise")
    group.add_argument("--child")
    group.add_argument("--execute-current-pytest")
    parser.add_argument("--pre-spawn")
    args = parser.parse_args(argv)
    if args.supervise:
        require(args.pre_spawn is None, "parent creates actual receipt")
        return supervise(args.supervise)
    if args.child:
        require(args.pre_spawn is not None, "child requires actual receipt")
        return child(args.child, args.pre_spawn)
    require(args.pre_spawn is None, "whole-test entrypoint")
    return execute_current_pytest(args.execute_current_pytest)


if __name__ == "__main__":
    raise SystemExit(main())
