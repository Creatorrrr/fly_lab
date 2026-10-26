"""E355 NumPy forensic consistency only; E354 failed closure remains unknown."""

from __future__ import annotations

import time

ENTRY = time.perf_counter()
import argparse
import ast
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
BASE = "verification/banc-natural-gait-m3-20260919"
OWNED = (
    "tools/recover_c_banc_m3_reference_teacher_loss.py",
    "tests/test_c_banc_reference_teacher_recovery.py",
)
REGISTRATION = BASE + "/parent-e355-teacher-recovery-input-registration.json"
REG_SHA = "a9bc9ba62a81d41803b4ab00c6f4cb256feb691c66b42e506027f92b963331bb"
REFLECTION = BASE + "/parent-e355-reflected-design-registration.json"
REF_SHA = "476c8b53bc3e38fee1f4e04a38cb3226f0a5b82ac441f318fbf9db79f35972e4"
E354_SHA = "10b693b2e4817b78e1160476c7d185835ee862696c009a6f159e7bdccee5142a"
HELPER_SHA = "a9a5804b9b48adffb41c43c91931c5d1664b43e17624f1f0c70478a7105c1a04"
OUTPUT = BASE + "/e355-reference-teacher-forensic-audit"
CAP, EMERGENCY = 4 << 20, 64 << 10
CHECKS = (
    "CPU",
    "ruff",
    "independent_math",
    "raw_exact13",
    "failure_provenance",
    "private_gate",
    "supervisor_contract",
    "case_execution",
    "hidden_selection",
    "publication_deadline",
    "lazy_import",
    "strict_JSON",
)
DENIED = dict.fromkeys(
    (
        "E354_completion_recovered",
        "parameter_VJP_qualified",
        "gradient_step_improved",
        "candidate_adopted",
        "BANC_autonomy",
        "F1_F8_complete",
        "biology",
        "natural_gait",
        "M2_UI_F6_changed",
        "actual_CUDA_qualified",
    ),
    False,
)
_MODULES = {}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def module(path, digest):
    if path not in _MODULES:
        file = ROOT / path
        require(sha(file.read_bytes()) == digest, "frozen source pin")
        spec = importlib.util.spec_from_file_location("e355_" + file.stem, file)
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        _MODULES[path] = value
    return _MODULES[path]


def helper():
    return module("tools/recover_c_banc_m3_source_pose_report.py", HELPER_SHA)


def frozen():
    return module("tools/banc_m3_reference_cadence_teacher.py", E354_SHA)


def own_sources(root=ROOT):
    return {name: sha(helper().safe(name, root).read_bytes()) for name in OWNED}


def descriptor(name, raw):
    return {"path": name, "bytes": len(raw), "sha256": sha(raw)}


def json_bytes(value):
    return json.dumps(
        helper().lower(value), separators=(",", ":"), allow_nan=False
    ).encode("utf8")


def error_record(stage, error):
    return {"stage": stage, "type": type(error).__name__, "message": str(error)[:1024]}


def generated_reader(root):
    root = Path(root).resolve()
    require(
        not root.is_relative_to(ROOT)
        or root.relative_to(ROOT).as_posix().startswith(BASE + "/e355-sol-"),
        "owned fixture only",
    )
    r = helper().EvidenceReader(root)
    r.array_authorized = True
    return r


def failed_admission(reg, documents, *, root=ROOT):
    def doc(tail):
        return documents[BASE + "/" + tail]

    failed, outer, tool, old = (
        doc(n)
        for n in (
            "parent-e354-failed-full-return.json",
            "parent-e354-run-outer-witness.json",
            "parent-e354-run-tool-return.json",
            "parent-e346-e354-run-contract.json",
        )
    )
    decision = doc("parent-e354-final-decision.json")
    result, receipt = (
        doc("e354-reference-teacher-loss/" + n + ".json") for n in ("result", "receipt")
    )
    wall = failed["actual_outer_full_return_s"]
    require(
        failed["schema"] == "flylab.m3.e354.failed-actual-full-return.v1"
        and all(
            type(value) is int
            for value in (
                failed["exit_code"],
                failed["expected_exit_code"],
                tool["actual_returned_exit_code"],
                outer["exit_code"],
            )
        )
        and failed["exit_code"]
        == failed["expected_exit_code"]
        == tool["actual_returned_exit_code"]
        == outer["exit_code"]
        == 1,
        "exact known failed outer exit1",
    )
    require(
        type(wall) in (float, int)
        and math.isfinite(wall)
        and 0 < wall <= 10
        and wall == failed["charged_s"] == tool["actual_outer_full_return_s"]
        and failed["charged_once"] is True,
        "actual failed outer charge",
    )
    require(
        failed["source_sha256"]
        == reg["e354_source_sha256"]
        == result["source_sha256"]
        == decision["source_sha256"]
        and failed["worker_actual_exit_code"] is None
        and failed["worker_actual_exit_code_not_reconstructed"] is True
        and failed["supervision_report_available"] is False
        and decision["diagnostic_completion_qualified"] is False
        and decision["H354"] == "not_evaluated_due_failed_outer_closure",
        "unknown worker closure preserved",
    )
    require(
        outer["all_owned_ended"] is True
        and failed["all_owned_ended"] is True
        and outer["known_identity_cleanup"] == failed["known_identity_cleanup"] == []
        and outer["first_error"] is None
        and outer["secondary_errors"] == []
        and outer["owned_identities"] == failed["all_observed_PID_creation_identities"]
        and outer["owned_identities"],
        "observed failed identities closed",
    )
    require(
        all(
            type(x["pid"]) is int
            and type(x["create_time"]) in (float, int)
            and math.isfinite(x["create_time"])
            and x["create_time"] > 0
            and x["close"] == "ended"
            for x in outer["owned_identities"]
        ),
        "typed historical creation identities",
    )
    require(
        "cumulative_limit_s" not in old
        and old["original_calls"] == 1
        and all(
            old["source_sha256"].get(k) == v
            for k, v in reg["e354_source_sha256"].items()
        ),
        "exact missing cumulative failure",
    )
    stderr = documents[BASE + "/parent-e346-e354-run-supervision-observer.stderr.txt"]
    require(
        "KeyError: 'cumulative_limit_s'" in stderr
        and "parent-e267-supervise.py" in stderr,
        "registered E267 finally failure trace",
    )
    for spec, tail in (
        (failed["contract"], "parent-e346-e354-run-contract.json"),
        (outer["contract"], "parent-e346-e354-run-contract.json"),
        (failed["actual_tool_return"], "parent-e354-run-tool-return.json"),
        (failed["outer_witness"], "parent-e354-run-outer-witness.json"),
    ):
        require(
            spec == reg["failure_provenance"][BASE + "/" + tail],
            "failed descriptor linkage",
        )
    supervision = helper().safe(old["supervisor_output"], root)
    require(
        not supervision.exists()
        and not supervision.with_name(supervision.stem + "-identities.json").exists()
        and not helper()
        .safe(BASE + "/e354-reference-teacher-loss/failure.json", root)
        .exists(),
        "registered absent E354 reports",
    )
    require(
        result["receipt"] == receipt
        and receipt["status"] == "completed"
        and receipt["first_error"] is None
        and receipt["secondary_errors"] == []
        and set(receipt["acquired"]) == set(frozen().raw_roster())
        and receipt["missing"] == []
        and set(result["raw"]["selected_members"]) == set(frozen().raw_roster()),
        "declared complete exact13",
    )
    require(
        all(
            type(receipt[k]) is int and receipt[k] == 1
            for k in (
                "full_requested",
                "full_returned",
                "window_requested",
                "window_returned",
                "teacher_self_loss_direct_evaluations",
            )
        ),
        "typed declared gradient counts",
    )
    require(
        all(
            result["raw"][k] == reg["acquired_raw"][k]
            for k in ("bytes", "sha256", "selected_members")
        )
        and result["external_completion_qualified"] is False,
        "inherited acquired descriptor only",
    )
    return result, receipt


def metadata_preflight(root=ROOT, reader=None):
    r = reader or helper().EvidenceReader(root)

    def pinned(name, digest):
        file = helper().safe(name, root)
        return r.json({"path": name, "bytes": file.stat().st_size, "sha256": digest})

    reg, reflection = pinned(REGISTRATION, REG_SHA), pinned(REFLECTION, REF_SHA)
    require(
        reg["schema"] == "flylab.m3.e355.teacher-recovery-input-registration.v1"
        and reg["original_authorized"] is False
        and reflection["registration"]["sha256"] == REG_SHA,
        "current E355 reflection",
    )
    for spec in (reg["design"], reg["actual_review"], reg["review_md"]):
        r.bytes(spec)
    for name, digest in reg["consumed_sources"].items():
        require(
            name.endswith(".py")
            and sha(helper().safe(name, root).read_bytes()) == digest,
            "consumed source pin",
        )
    inherited = frozen().metadata_preflight(root, r)
    require(
        inherited["registration"]["payloads"] == reg["payloads"],
        "unchanged original3 selected7",
    )
    documents = {
        name: (r.bytes(spec).decode("utf8") if name.endswith(".txt") else r.json(spec))
        for name, spec in reg["failure_provenance"].items()
    }
    result, receipt = failed_admission(reg, documents, root=root)
    require(
        helper().safe(reg["acquired_raw"]["path"], root).stat().st_size
        == reg["acquired_raw"]["bytes"],
        "raw stat only before gate",
    )
    return {
        "reader": r,
        "registration": reg,
        "authorized": False,
        "declared_result": result,
        "declared_receipt": receipt,
    }


def recover(ctx, *, timely=lambda: None):
    require(ctx.get("authorized") is True, "E355 private gate before original bytes")
    r, reg = ctx["reader"], ctx["registration"]
    r.timely = timely
    timely()
    data = frozen().load_inputs(ctx)
    spec = reg["acquired_raw"]
    raw = r.npz(spec, frozen().raw_roster(), selected=spec["selected_members"])
    with zipfile.ZipFile(io.BytesIO(r.cache[spec["path"]])) as archive:
        require(
            set(archive.namelist()) == {k + ".npy" for k in frozen().raw_roster()},
            "forensic raw exact13 no extras",
        )
    timely()
    metrics = frozen().independent_check(
        data, raw, ctx["declared_receipt"], ctx["declared_result"]
    )
    timely()
    return {
        "schema": "flylab.m3.e355.teacher-forensic-report.v1",
        "status": "analysis_completed",
        "H355": "recovered_offline_label_loss_derivative_consistency",
        "metrics": metrics,
        "source_sha256": own_sources(),
        "registration_sha256": REG_SHA,
        "reflection_sha256": REF_SHA,
        "read_counts": r.read_counts,
        "consumed_digests": r.digests,
        "opened_members": [list(member) for member in r.opened_members],
        "measured_scope": "E355_NumPy_reader_and_direct_formula_only",
        "declared_E354_receipt": ctx["declared_receipt"],
        "worker_actual_exit_code": None,
        "E354_H354": "not_evaluated",
        "E354_outer_exit_code": 1,
        "Torch_autograd_NN_body_CPG_GPU_calls": 0,
        "external_completion_qualified": False,
        **DENIED,
    }


def ast_cases(file):
    tree = ast.parse(file.read_text("utf8"))
    require(
        not any(
            isinstance(n, ast.AsyncFunctionDef) and n.name.startswith("test_")
            for n in tree.body
        ),
        "synchronous complete test functions",
    )
    return sorted(
        n.name
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    )


def clean_pytest_environment():
    prior = {name: name in os.environ for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS")}
    for name in prior:
        os.environ.pop(name, None)
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return {
        "prior_variable_present": prior,
        "PYTEST_ADDOPTS_absent": True,
        "PYTEST_PLUGINS_absent": True,
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "raw_prior_values_recorded": False,
    }


def validate_case_contract(name, *, root=ROOT, before_execution=False):
    file = helper().safe(name, root)
    require(
        file.parent == helper().safe(BASE, root)
        and file.name.startswith(
            ("parent-e355-sol-FIRST-", "parent-e355-sol-R1-", "parent-e355-root-")
        )
        and file.suffix == ".json",
        "controlled case contract path",
    )
    raw = file.read_bytes()
    c = helper().strict_json(raw)
    own = own_sources(root)
    require(
        set(c)
        == {
            "schema",
            "source_sha256",
            "complete_suite",
            "expected_functions",
            "extra_tests",
            "empty_ini",
            "basetemp",
            "receipt_output",
        },
        "case contract has no selection/options fields",
    )
    require(
        c["schema"] == "flylab.m3.e355.pytest-case-contract.v1"
        and c["source_sha256"] == own,
        "current case contract sources",
    )
    reg_raw = helper().safe(REGISTRATION, root).read_bytes()
    require(sha(reg_raw) == REG_SHA, "case authority registration pin")
    reg = helper().strict_json(reg_raw)
    current = ast_cases(helper().safe(OWNED[1], root))
    require(
        current == sorted(reg["required_owned_test_functions"]),
        "mandatory current AST cases",
    )
    extras = c.get("extra_tests", [])
    require(type(extras) is list, "registered case extras")
    expected = {OWNED[1]: current}
    for s in extras:
        path = s["path"]
        require(
            type(path) is str
            and path.startswith(BASE + "/parent-e355-root-")
            and path.endswith(".py")
            and path not in expected,
            "controlled ROOT extra tests",
        )
        value = helper().safe(path, root).read_bytes()
        require(
            len(value) == s["bytes"] and sha(value) == s["sha256"],
            "current registered extra source",
        )
        expected[path] = ast_cases(helper().safe(path, root))
    require(
        c["complete_suite"] == list(expected) and c["expected_functions"] == expected,
        "complete current AST suite only",
    )
    ini = c["empty_ini"]
    require(ini == reg["empty_pytest_ini"], "pinned empty ini registration")
    ini_raw = helper().safe(ini["path"], root).read_bytes()
    require(
        len(ini_raw) == ini["bytes"]
        and sha(ini_raw) == ini["sha256"]
        and ini_raw.decode("utf8").splitlines() == ["[pytest]", "addopts ="],
        "empty ini no selection",
    )
    temp = Path(c["basetemp"])
    require(
        temp.is_absolute()
        and not temp.resolve().is_relative_to(root.resolve())
        and temp.name.startswith(("e355-sol-FIRST-", "e355-sol-R1-", "e355-root-"))
        and ".." not in temp.parts,
        "owned external pytest temp",
    )
    receipt = helper().safe(c["receipt_output"], root)
    require(
        receipt.parent == helper().safe(BASE, root)
        and receipt.name.startswith(
            ("parent-e355-sol-FIRST-", "parent-e355-sol-R1-", "parent-e355-root-")
        )
        and receipt.suffix == ".json",
        "owned case receipt",
    )
    if before_execution:
        require(
            not temp.exists() and not receipt.exists(),
            "fresh temp and case receipt before pytest",
        )
    return c, descriptor(name, raw)


def complete_case_receipt(receipt, expected):
    require(
        receipt["effective_environment"]["PYTEST_ADDOPTS_absent"] is True
        and receipt["effective_environment"]["PYTEST_PLUGINS_absent"] is True
        and receipt["effective_environment"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
        and receipt["effective_environment"]["raw_prior_values_recorded"] is False,
        "sanitized effective pytest environment",
    )
    require(
        type(receipt["pytest_main_requested"]) is int
        and type(receipt["pytest_main_returned"]) is int
        and receipt["pytest_main_requested"] == receipt["pytest_main_returned"] == 1
        and type(receipt["pytest_return_code"]) is int
        and receipt["pytest_return_code"] == 0
        and receipt["collection_errors"] == receipt["deselected"] == []
        and receipt["first_error"] is None
        and receipt["secondary_errors"] == [],
        "actual complete pytest return",
    )
    require(
        receipt["expected_functions"] == expected
        and receipt["basetemp_fresh_before_execution"] is True,
        "current case roster/fresh execution",
    )
    collected = receipt["collected"]
    require(
        type(collected) is list
        and collected
        and len(set(collected)) == len(collected)
        and len(collected) <= 256,
        "actual collected cases",
    )
    require(
        set(receipt["case_paths"]) == set(collected),
        "actual item paths for every collected nodeid",
    )
    bases = {file: set() for file in expected}
    for node in collected:
        fields = node.split("::")
        item_path = receipt["case_paths"][node]
        allowed_prefixes = {"", item_path}
        try:
            ini_relative = (
                PurePosixPath(item_path)
                .relative_to(PurePosixPath(receipt["empty_ini"]["path"]).parent)
                .as_posix()
            )
        except ValueError:
            pass
        else:
            allowed_prefixes.add(ini_relative)
        require(
            len(fields) == 2
            and item_path in expected
            and fields[0] in allowed_prefixes,
            "controlled top-level actual case",
        )
        bases[receipt["case_paths"][node]].add(fields[1].split("[", 1)[0])
    require(
        all(bases[file] == set(names) for file, names in expected.items()),
        "every current AST case executed",
    )
    phases = {node: {} for node in collected}
    for row in receipt["reports"]:
        require(
            row["nodeid"] in phases
            and row["when"] in ("setup", "call", "teardown")
            and row["when"] not in phases[row["nodeid"]],
            "one actual report per phase",
        )
        phases[row["nodeid"]][row["when"]] = row["outcome"]
    require(
        all(
            rows == {"setup": "passed", "call": "passed", "teardown": "passed"}
            for rows in phases.values()
        ),
        "all actual setup/call/teardown passed; no skips",
    )
    return True


def check_execution(r, execution, own, cap, *, case_role=False):
    values = {
        k: r.json(execution[k])
        for k in ("contract", "supervision", "identity", "full_exit")
    }
    c, s, i, f = (
        values[k] for k in ("contract", "supervision", "identity", "full_exit")
    )
    wall = f["full_exit_elapsed_s"]
    require(
        f["schema"] == "flylab.m3.e355.actual-full-return.v1"
        and f["source_sha256"] == own
        and f["scope"] == "prevalidation-through-observer-final-exit"
        and f["exit_code"] == 0
        and f["all_owned_ended"] is True
        and f["known_identity_cleanup"] == []
        and f["first_error"] is None
        and f["secondary_errors"] == []
        and type(wall) in (float, int)
        and math.isfinite(wall)
        and 0 < wall <= cap
        and wall == f["charged_s"] == f["actual_outer_full_return_s"]
        and f["charged_once"] is True
        and all(f[k] == execution[k] for k in ("contract", "supervision", "identity")),
        "actual complete outer full return",
    )
    require(
        all(c["source_sha256"].get(k) == v for k, v in own.items())
        and type(c["original_calls"]) is int
        and c["original_calls"] == f["original_calls"] == 0
        and s["success"] is True
        and s["max_tree_rss_bytes"] <= c["max_tree_rss_bytes"] <= 2 << 30
        and len(s["attempts"]) == len(c["commands"])
        and s["attempts"]
        and all(
            a["argv"] == b["argv"] and a["exit_code"] == 0 and not a["killed"]
            for a, b in zip(s["attempts"], c["commands"], strict=True)
        ),
        "actual source/argv/exit/RSS proof",
    )
    require(
        i["qualified"] is True
        and i["all_owned_pids_ended"] is True
        and i["errors"] == []
        and i["actual_supervisor_exit_code"] == 0
        and i["contract"] == execution["contract"]
        and i["supervision"] == execution["supervision"],
        "actual immutable identity proof",
    )
    outer, tool = r.json(f["outer_witness"]), r.json(f["actual_tool_return"])
    require(
        outer["all_owned_ended"] is True
        and outer["known_identity_cleanup"] == []
        and outer["first_error"] is None
        and outer["secondary_errors"] == []
        and outer["exit_code"] == tool["actual_returned_exit_code"] == 0
        and tool["actual_outer_full_return_s"] == wall
        and tool["tool_chunk_id"] == f["actual_tool_chunk_id"],
        "same actual returned tool and witness",
    )
    known = {
        x["pid"]: x["create_time"]
        for x in outer["owned_identities"]
        if x["close"] == "ended"
    }
    require(
        i["owned_pids"]
        and all(
            known.get(pid) == i["pid_create_times"][str(pid)]
            and i["present_after_return"][str(pid)] is False
            for pid in i["owned_pids"]
        ),
        "observed same owned creation identities ended",
    )
    if case_role:
        require(len(c["commands"]) == 1, "sole actual private pytest child")
        argv = c["commands"][0]["argv"]
        require(
            type(argv) is list
            and len(argv) == 7
            and argv[:6]
            == [
                ".venv/Scripts/python.exe",
                "-X",
                "utf8",
                "-B",
                OWNED[1],
                "--execute-whole-current-pytest",
            ],
            "actual controlled TEST entrypoint, not noop/flags",
        )
        case, case_spec = validate_case_contract(argv[6], root=r.root)
        link = f["pytest_case_execution"]
        require(
            link["case_contract"] == case_spec
            and c["source_sha256"].get(argv[6]) == case_spec["sha256"]
            and link["empty_ini"] == case["empty_ini"]
            and link["case_receipt"]["path"] == case["receipt_output"],
            "same job case/receipt/ini pins",
        )
        require(
            all(
                c["source_sha256"].get(item["path"]) == item["sha256"]
                for item in [case["empty_ini"], *case.get("extra_tests", [])]
            ),
            "same actual job ini/extra source pins",
        )
        receipt = r.json(link["case_receipt"])
        require(
            receipt["schema"] == "flylab.m3.e355.pytest-case-receipt.v1"
            and receipt["source_sha256"] == own
            and receipt["case_contract"] == case_spec
            and receipt["empty_ini"] == case["empty_ini"]
            and receipt["complete_current_cases"] is True,
            "current closed case receipt",
        )
        expected_args = [
            "-q",
            "-p",
            "no:cacheprovider",
            "--noconftest",
            "-c",
            case["empty_ini"]["path"],
            "--basetemp",
            case["basetemp"],
            *case["complete_suite"],
        ]
        require(
            receipt["pytest_args"] == expected_args, "actual fixed internal pytest argv"
        )
        complete_case_receipt(receipt, case["expected_functions"])
    return f


def make_freeze(qualification, root=ROOT, context=None):
    ctx = context or metadata_preflight(root)
    r = ctx["reader"]
    spec = descriptor(qualification, helper().safe(qualification, root).read_bytes())
    q, own = r.json(spec), own_sources(root)
    require(
        q["schema"] == "flylab.m3.e355.root-qualification.v1"
        and q["source_sha256"] == own
        and q["passed"] is True
        and type(q["original_calls"]) is int
        and q["original_calls"] == 0
        and set(q["checks"]) == set(CHECKS)
        and all(v is True for v in q["checks"].values()),
        "current ROOT qualification",
    )
    check_execution(r, q["execution"], own, 20, case_role=True)
    check_execution(r, q["ruff_execution"], own, 20)
    style = r.json(q["ruff_execution"]["contract"])["commands"]
    prefix = [".venv/Scripts/python.exe", "-X", "utf8", "-B", "-m", "ruff"]
    require(
        [row["argv"] for row in style]
        == [prefix + ["check", *OWNED], prefix + ["format", "--check", *OWNED]],
        "current actual Ruff/style role",
    )
    return {
        "schema": "flylab.m3.e355.forensic-freeze.v1",
        "source_sha256": own,
        "qualification": spec,
        "registration_sha256": REG_SHA,
        "reflection_sha256": REF_SHA,
        "design": ctx["registration"]["design"],
        "consumed_sources": ctx["registration"]["consumed_sources"],
        "original_authorized": False,
    }


def original_argv(freeze, review):
    return [
        ".venv/Scripts/python.exe",
        "-X",
        "utf8",
        "-B",
        OWNED[0],
        "--audit",
        "--freeze",
        freeze,
        "--review",
        review,
        "--output",
        OUTPUT,
    ]


def contract_paths(name, root=ROOT):
    path = helper().safe(name, root)
    require(
        path.parent == helper().safe(BASE, root)
        and path.name.startswith("parent-e346-e355-")
        and path.name.endswith("-contract.json"),
        "exact E355 E346 contract namespace",
    )
    tail = path.name[len("parent-e346-e355-") : -len("-contract.json")]
    require(tail and all(x.isalnum() or x == "-" for x in tail), "contract suffix")
    supervisor = BASE + "/parent-e346-e355-" + tail + "-supervision"
    outer = BASE + "/parent-e355-" + tail
    return {
        "supervision": supervisor + ".json",
        "identity": supervisor + "-identities.json",
        "observer_stdout": supervisor + "-observer.stdout.txt",
        "observer_stderr": supervisor + "-observer.stderr.txt",
        "stdout": outer + "-worker.stdout.txt",
        "stderr": outer + "-worker.stderr.txt",
        "witness_stdout": outer + "-witness.stdout",
        "witness_stderr": outer + "-witness.stderr",
        "outer_witness": outer + "-outer-witness.json",
        "full_exit": outer + "-full-exit.json",
        "tool_return": outer + "-tool-return.json",
        "output": OUTPUT,
    }


def validate_supervisor_contract(
    name,
    freeze,
    review,
    reservation,
    *,
    root=ROOT,
    phase="pre_spawn",
    reader=None,
    registration=None,
):
    """No arrays: use both immediately before spawn and in the owned live child."""
    require(phase in ("pre_spawn", "array_gate"), "contract validation phase")
    r = reader or helper().EvidenceReader(root)
    paths = contract_paths(name, root)
    raw = helper().safe(name, root).read_bytes()
    c = helper().strict_json(raw)
    reg = registration or r.json(
        descriptor(REGISTRATION, helper().safe(REGISTRATION, root).read_bytes())
    )
    require(
        reg["schema"] == "flylab.m3.e355.teacher-recovery-input-registration.v1"
        and sha(helper().safe(REGISTRATION, root).read_bytes()) == REG_SHA,
        "current contract registry",
    )
    own = own_sources(root)
    pins = {
        **reg["consumed_sources"],
        **own,
        REGISTRATION: REG_SHA,
        REFLECTION: REF_SHA,
    }
    for item in (freeze, review, reservation):
        pins[item["path"]] = item["sha256"]
        r.bytes(item)
    required = {
        "schema",
        "phase",
        "source_sha256",
        "commands",
        "supervisor_output",
        "max_tree_rss_bytes",
        "original_calls",
        "cumulative_before_s",
        "cumulative_limit_s",
        "outer_wall_s",
        "internal_wall_s",
        "freeze",
        "review",
        "reservation",
        "output",
    }
    require(required <= set(c), "all required top-level supervisor fields")
    require(
        c["schema"] == "flylab.m3.e355.original-supervisor-contract.v1"
        and c["phase"] == "E355 original forensic NumPy audit",
        "current original contract schema/phase",
    )
    for key, expected in (
        ("original_calls", 1),
        ("cumulative_limit_s", 9900),
        ("max_tree_rss_bytes", 2 << 30),
        ("outer_wall_s", 6),
        ("internal_wall_s", 4),
    ):
        require(type(c[key]) is int and c[key] == expected, "typed contract " + key)
    require(
        type(c["cumulative_before_s"]) is float
        and math.isfinite(c["cumulative_before_s"])
        and c["cumulative_before_s"] == 9871.733085099502,
        "finite cumulative before",
    )
    require(
        c["freeze"] == freeze
        and c["review"] == review
        and c["reservation"] == reservation
        and c["output"] == OUTPUT
        and c["supervisor_output"] == paths["supervision"],
        "contract finalized authority/output linkage",
    )
    require(
        type(c["commands"]) is list and len(c["commands"]) == 1, "sole original command"
    )
    job = c["commands"][0]
    require(
        type(job) is dict and {"argv", "wall_s", "stdout", "stderr"} <= set(job),
        "all nested command fields",
    )
    require(
        job["argv"] == original_argv(freeze["path"], review["path"])
        and type(job["wall_s"]) is int
        and job["wall_s"] == 4
        and job["stdout"] == paths["stdout"]
        and job["stderr"] == paths["stderr"],
        "sole exact argv/wall/logs",
    )
    require(
        type(c["source_sha256"]) is dict
        and all(c["source_sha256"].get(k) == v for k, v in pins.items()),
        "contract current authority/source pins",
    )
    for file, digest in c["source_sha256"].items():
        require(
            Path(file).suffix.lower() in (".py", ".json", ".ini", ".txt", ".md")
            and file not in (name, *paths.values()),
            "metadata-only contract source pin",
        )
        require(
            sha(helper().safe(file, root).read_bytes()) == digest,
            "final source pin " + file,
        )
    f, reviewed, reserved = (r.json(item) for item in (freeze, review, reservation))
    require(
        f["schema"] == "flylab.m3.e355.forensic-freeze.v1"
        and f["source_sha256"] == own
        and f["registration_sha256"] == REG_SHA
        and f["reflection_sha256"] == REF_SHA
        and f["original_authorized"] is False,
        "current frozen qualification authority",
    )
    require(
        reviewed["schema"] == "flylab.m3.e355.root-original-review.v1"
        and reviewed["approved"] is True
        and reviewed["source_sha256"] == own
        and reviewed["freeze"] == freeze
        and reviewed["reservation"] == reservation
        and reviewed["supervisor_contract_path"] == name
        and reviewed["registration_sha256"] == REG_SHA
        and reviewed["reflection_sha256"] == REF_SHA
        and reviewed["output"] == OUTPUT,
        "actual ROOT review authority",
    )
    require(
        reserved == reservation_value(freeze, review["path"], own)
        and all(
            type(reserved[k]) is int
            for k in (
                "cumulative_limit_s",
                "reserved_s",
                "audit_s",
                "internal_s",
                "original_calls",
                "retry",
                "body_calls",
                "NN_calls",
                "GPU_calls",
                "CPG_calls",
            )
        )
        and type(reserved["cumulative_before_s"]) is float,
        "finalized reservation authority",
    )
    require(
        f == make_freeze(f["qualification"]["path"], root=root),
        "current complete qualification before spawn or arrays",
    )
    live = {
        "stdout",
        "stderr",
        "observer_stdout",
        "observer_stderr",
        "witness_stdout",
        "witness_stderr",
    }
    for key, file in paths.items():
        exists = helper().safe(file, root).exists()
        if phase == "array_gate" and key in live:
            require(
                exists and helper().safe(file, root).is_file(),
                "owned live log at array gate " + key,
            )
        else:
            require(not exists, "fresh final/output path " + key)
    return {"contract": descriptor(name, raw), "paths": paths, "phase": phase}


def gate(freeze_name, review_name, *, root=ROOT, argv=None):
    ctx = metadata_preflight(root)
    r = ctx["reader"]
    fs = descriptor(freeze_name, helper().safe(freeze_name, root).read_bytes())
    rs = descriptor(review_name, helper().safe(review_name, root).read_bytes())
    f, review = r.json(fs), r.json(rs)
    contract_name = review["supervisor_contract_path"]
    require(
        f == make_freeze(f["qualification"]["path"], root, ctx),
        "current passive freeze",
    )
    own = own_sources(root)
    require(
        review["schema"] == "flylab.m3.e355.root-original-review.v1"
        and review["approved"] is True
        and review["source_sha256"] == own
        and review["freeze"] == fs
        and review["registration_sha256"] == REG_SHA
        and review["reflection_sha256"] == REF_SHA
        and review["output"] == OUTPUT
        and review["supervisor_contract_path"] == contract_name,
        "exact ROOT original review",
    )
    reserve_spec = review["reservation"]
    reserve = r.json(reserve_spec)
    fixed = reservation_value(fs, review_name, own)
    require(
        reserve == fixed
        and all(
            type(reserve[k]) is int
            for k in (
                "cumulative_limit_s",
                "reserved_s",
                "audit_s",
                "internal_s",
                "original_calls",
                "retry",
                "body_calls",
                "NN_calls",
                "GPU_calls",
                "CPG_calls",
            )
        )
        and type(reserve["cumulative_before_s"]) is float,
        "exact sole ROOT reservation",
    )
    require(argv == fixed["argv"][5:], "sole actual original CLI")
    validate_supervisor_contract(
        contract_name,
        fs,
        rs,
        reserve_spec,
        root=root,
        phase="array_gate",
        reader=r,
        registration=ctx["registration"],
    )
    require(
        all(
            os.environ.get(k) == "1"
            for k in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        )
        and shutil.disk_usage(helper().safe(BASE, root)).free >= 512 << 20,
        "CPU1 and disk reserve",
    )
    r.array_authorized = True
    ctx["authorized"] = True
    return ctx


def reservation_value(freeze, review_name, own):
    return {
        "schema": "flylab.m3.e355.root-reservation.v1",
        "cumulative_before_s": 9871.733085099502,
        "cumulative_limit_s": 9900,
        "reserved_s": 6,
        "audit_s": 6,
        "internal_s": 4,
        "original_calls": 1,
        "retry": 0,
        "body_calls": 0,
        "NN_calls": 0,
        "GPU_calls": 0,
        "CPG_calls": 0,
        "source_sha256": own,
        "freeze": freeze,
        "review_path": review_name,
        "registration_sha256": REG_SHA,
        "reflection_sha256": REF_SHA,
        "output": OUTPUT,
        "argv": original_argv(freeze["path"], review_name),
    }


def publish(output, name, value, *, timely=lambda: None, cap=CAP, reserve=EMERGENCY):
    timely()
    raw = json_bytes(value)
    used = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    require(
        len(raw) + used + reserve <= cap, "report plus emergency cap before publication"
    )
    with (output / name).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    timely()
    require(
        sum(p.stat().st_size for p in output.iterdir() if p.is_file()) <= cap,
        "post-close output cap",
    )
    return descriptor(name, raw)


def run_recovery(ctx, output, *, timely=lambda: None):
    """Saved provisional success never overrides a late close/deadline error."""
    first, secondary, report = None, [], None
    stage = "independent_recovery"
    try:
        report = recover(ctx, timely=timely)
        stage = "report_publication"
        publish(output, "report.json", report, timely=timely)
        timely()
        return report, 0
    except Exception as exc:  # noqa: BLE001 -- preserve sole-job first/late failures
        first = error_record(stage, exc)
    failure = {
        "schema": "flylab.m3.e355.teacher-forensic-failure.v1",
        "status": "technical_invalid",
        "H355": "not_evaluated",
        "first_error": first,
        "secondary_errors": secondary,
        "stage": stage,
        "late_error_has_precedence": True,
        "independent_calculation_completed": report is not None,
        "declared_E354_receipt": ctx.get("declared_receipt"),
        "worker_actual_exit_code": None,
        "E354_H354": "not_evaluated",
        "Torch_autograd_NN_body_CPG_GPU_calls": 0,
        "external_completion_qualified": False,
        **DENIED,
    }
    try:
        require(len(json_bytes(failure)) <= EMERGENCY, "bounded emergency64KiB")
        publish(output, "failure.json", failure, cap=CAP, reserve=0)
    except Exception as exc:  # noqa: BLE001 -- no old report rewrite or deletion
        secondary.append(error_record("emergency_publication", exc))
    return failure, 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--qualification")
    parser.add_argument("--freeze")
    parser.add_argument("--review")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.qualification:
        require(
            args.freeze
            and not args.audit
            and args.review is None
            and args.output is None,
            "sole passive freeze CLI",
        )
        value = make_freeze(args.qualification)
        file = helper().safe(args.freeze)
        require(
            file.parent == helper().safe(BASE) and not file.exists(),
            "fresh passive freeze file",
        )
        raw = json_bytes(value)
        with file.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        return 0
    output = None
    try:
        require(
            args.audit and args.output == OUTPUT and args.freeze and args.review,
            "fixed output/current approval fields",
        )

        def timely():
            require(
                time.perf_counter() - ENTRY <= 4,
                "internal four-second final-close deadline",
            )

        timely()
        ctx = gate(
            args.freeze,
            args.review,
            argv=list(argv if argv is not None else sys.argv[1:]),
        )
        timely()
        output = helper().safe(OUTPUT)
        output.mkdir()
        _, code = run_recovery(ctx, output, timely=timely)
        return code
    except Exception as exc:  # noqa: BLE001 -- default-denied gate never creates original output
        print(
            json.dumps(
                {
                    "status": "technical_invalid",
                    "stage": "private_gate" if output is None else "final_close",
                    "error": error_record("main", exc),
                    "original_authorized": False,
                }
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
