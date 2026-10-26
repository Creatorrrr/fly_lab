"""E335 single same-theta native body screen, with passive producer gates.

Generated CPU fixtures are explicit. Original execution is root-only, one body,
one direction, and never a biological gait or F1-F8 completion claim.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import numpy as np
import psutil

from tools import verify_c_banc_m3_smooth_constrained_secant as e334

ROOT, BASE = e334.ROOT, e334.BASE
N, GROUPS, CONTROLS = 158706, 8547, 300
OWNED = (
    "tools/verify_c_banc_m3_smooth_secant_body.py",
    "tests/test_c_banc_smooth_secant_body.py",
)
OUTPUT = BASE + "/e335-smooth-secant-body"
PREVIOUS = e334.OUTPUT
SCHEMA = "flylab.m3.e335.smooth-secant-body.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e335.smooth-secant-body.review.v1"
QUAL_SCHEMA = "flylab.m3.e335.generated-qualification.v1"
RESULT_SCHEMA = "flylab.m3.e335.smooth-secant-body.result.v1"
NORMAL_RAW_FILES = (
    "patched-edge-roster.npz",
    "patched-csr.npz",
    "actual-packed-csr.npz",
    "control-raw.npz",
    "native-contact.npz",
    "native-recruitment-state.npz",
    "activity.npz",
    "resource-description.json",
    "trace.json",
    "clock-events.json",
    "control-journal.jsonl",
    "live-input-events.json",
    "evaluation.json",
)
NORMAL_WHOLE_DIRECTORIES = ("observed-start", "whole-300")
LIMITS = {
    "supervisor_wall_s": 80,
    "internal_wall_s": 70,
    "science_wall_s": 60,
    "finalization_reserve_s": 10,
    "body_constructors": 1,
    "controls": 300,
    "neural_ticks": 6000,
    "physics_steps": 15000,
    "host_bytes": 6 << 30,
    "additional_gpu_bytes": 4 << 30,
    "output_bytes": 1 << 30,
    "minimum_free_disk_bytes": 4 << 30,
    "cpu_threads": 1,
    "gpu_count": 1,
    "retry_count": 0,
    "optimizer_steps": 0,
}
ALLOCATION = {
    "cumulative_before_s": 9431.526588899522,
    "cumulative_limit_s": 9660,
    "original_reserved_s": 80,
}
THETA_SHA = e334.THETA_SHA
CSR_SHA = "fabd71c6b7b1153cd060901f88edbb7964403598623c913cee2dc3005de5c249"
PATHS = {
    "e334_freeze": BASE + "/e334-smooth-constrained-secant-freeze.json",
    "e334_review": BASE + "/parent-e334-original-source-review.json",
    "e334_decision": BASE + "/parent-e334-final-decision.json",
    "e334_audit": BASE + "/parent-e334-original-raw-audit-v2.json",
    "e334_audit_source": BASE + "/parent-e334-original-raw-audit.py",
    "e334_supervision": BASE + "/parent-e334-original-supervision.json",
    "e334_contract": BASE + "/parent-e334-original-contract.json",
    "e334_audit_supervision": BASE + "/parent-e334-audit-v2-supervision.json",
    "e334_audit_contract": BASE + "/parent-e334-audit-v2-contract.json",
    "e334_result": PREVIOUS + "/result.json",
    "e334_ledger": PREVIOUS + "/advance-ledger.json",
    "e334_manifest": PREVIOUS + "/output-manifest.json",
    "e334_producer": PREVIOUS + "/source-manifest.json",
    "e334_receipts": PREVIOUS + "/receipts.json",
    "e318_freeze": BASE + "/e318-full-drive-capture-freeze.json",
    "e327_freeze": BASE + "/e327-constrained-secant-body-freeze.json",
    "e327_result": BASE + "/e327-constrained-secant-body/result.json",
    "e327_metrics_source": BASE + "/parent-e327-saved-metrics-v2.py",
    "design": "docs/BANC_M3_E335_SMOOTH_SECANT_BODY_DESIGN_20261001.md",
    "astra": BASE + "/parent-e335-astra-design-review.md",
    "reflection": BASE + "/parent-e335-astra-reflection.json",
    "invocation": BASE + "/parent-e335-astra-invocation.json",
    "qualification": BASE + "/parent-e335-generated-qualification.json",
}
ANCHORS = {
    "e334_decision": "6838c9282c2c6fd919567dd65464e267a48057bad42862902cf0d9d56e226d2a",
    "design": "022696f31975806d4f373c415ebdfe7ba776b2a8ff6fbd15549e33054da2c790",
    "astra": "0f66a9374bf04bef868ad476f68cd212c1b9b5915a2690bdfeb71f52848a79c3",
    "reflection": "632f561ebbc1cb6f87bd88d56409129e457cc7b1a928bf1e6b725c89a359b911",
}
sha256, bytes_sha, read, safe, pin_map = (
    e334.sha256,
    e334.bytes_sha,
    e334.read,
    e334.safe,
    e334.pin_map,
)


def put_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2) + "\n",
        encoding="utf8",
    )


def merge_pins(target, added):
    """Reject contradictory historical pins; never replace by current digest."""
    e334._merge(target, added)


def _closure(root):
    prior = read(safe(PATHS["e334_freeze"], root))
    old_body = read(safe(PATHS["e327_freeze"], root))
    body = read(safe(PATHS["e318_freeze"], root))
    if len(prior["source_sha256"]) != 183 or len(prior["input_sha256"]) != 2328:
        raise ValueError("exact inherited E334 183source/2328input required")
    if len(body["source_sha256"]) != 97 or len(body["input_sha256"]) != 402:
        raise ValueError("exact inherited E318 97source/402input required")
    sources, inputs = prior["source_sha256"].copy(), prior["input_sha256"].copy()
    for field, target in (("source_sha256", sources), ("input_sha256", inputs)):
        if any(target.get(k) != v for k, v in body[field].items()):
            raise ValueError("E318 producer must already be present in E334 closure")
    merge_pins(sources, old_body["source_sha256"])
    merge_pins(sources, {k: sha256(safe(k, root)) for k in OWNED})
    merge_pins(inputs, {k: sha256(safe(k, root)) for k in PATHS.values()})
    for role in ("e334_decision", "qualification"):
        merge_pins(inputs, read(safe(PATHS[role], root)).get("evidence_sha256", {}))
    invocation = read(safe(PATHS["invocation"], root))
    merge_pins(
        inputs, {BASE + "/" + k: v for k, v in invocation["files_sha256"].items()}
    )
    manifest = read(safe(PATHS["e334_manifest"], root))
    merge_pins(
        inputs, {PREVIOUS + "/" + k: v["sha256"] for k, v in manifest["files"].items()}
    )
    pure_sha = "b75f80920ba59886b93cf24fc157a3c483c51a92693fa43a3c33ff3055442117"
    if inputs[PATHS["e327_metrics_source"]] != pure_sha:
        raise ValueError("immutable independent saved gait evaluator differs")
    return prior, body, sources, inputs


def make_freeze(*, root=ROOT):
    prior, body, sources, inputs = _closure(root)
    return {
        "schema": SCHEMA,
        "paths": PATHS.copy(),
        "limits": LIMITS.copy(),
        "output": OUTPUT,
        "source_sha256": sources,
        "input_sha256": inputs,
        "e318_config": body,
        "smooth_constant_metadata": prior["smooth_constant_metadata"],
        "candidate": {
            "theta64_sha256": THETA_SHA,
            "patched_data32_sha256": CSR_SHA,
            "palette": 8040,
        },
        "candidate_adopted": False,
        "original_execution_authorized": False,
        "allocation": None,
        "original_supervisor_required": True,
    }


def _completed_supervision(proof, contract, digest, *, original_calls):
    # Historical PIDs may have been reused: completion is a historical producer
    # fact, not permission to signal or reinterpret an unrelated current PID.
    if (
        proof.get("schema") != "flylab.m3.e267.root-supervision.v1"
        or proof.get("success") is not True
        or len(proof.get("attempts", [])) != 1
        or not proof.get("owned_pids")
        or proof["attempts"][0].get("exit_code") != 0
        or proof["attempts"][0].get("killed") is not False
        or proof.get("original_calls") != original_calls
        or proof.get("contract_sha256") != digest
        or contract.get("original_calls") != original_calls
    ):
        raise ValueError("successful exact producer supervisor/contract required")


def verify_previous(proofs, pins, sources, *, root=ROOT):
    """Passive terminal/external/v2 proof checks, exposed for tamper fixtures."""
    prior, decision, result, ledger, producer, audit, manifest, receipts = (
        proofs[k]
        for k in (
            "freeze",
            "decision",
            "result",
            "ledger",
            "producer",
            "audit",
            "manifest",
            "receipts",
        )
    )
    if (
        decision.get("schema") != "flylab.m3.e334.root-final-decision.v1"
        or any(
            decision.get(k) is not True
            for k in (
                "run_valid",
                "effective_neural_qualification",
                "root_independent_raw_qualification",
                "handoff_eligible_for_same_candidate_body_design",
                "cost_charged_once",
                "owned_original_and_audit_processes_ended",
            )
        )
        or decision.get("original_cumulative_s") != 9431.526588899522
        or decision.get("original_allocation_limit_s") != 9660
        or decision.get("original_reserved_s") != 0
        or decision.get("whole_goal_status") != "active"
        or decision.get("body_calls") != 0
        or decision.get("candidate_adopted") is not False
        or decision.get("F1_F8_complete") is not False
        or decision.get("raw_checks") != 5094
    ):
        raise ValueError("E334 same-candidate completed final decision required")
    if (
        result.get("schema") != e334.RESULT_SCHEMA
        or ledger.get("schema") != "flylab.m3.e334.advance-ledger.v1"
        or producer.get("schema") != "flylab.m3.e334.source-manifest.v1"
        or manifest.get("schema") != "flylab.m3.e334.output-manifest.v1"
        or receipts.get("schema") != "flylab.m3.e334.receipts.v1"
        or result.get("run_valid") is not True
        or result.get("effective_neural_qualification") is not True
        or result.get("handoff_eligible") is not True
        or result.get("candidate_adopted") is not False
        or result.get("F1_F8_complete") is not False
        or result.get("body_calls") != 0
        or any(
            ledger.get(k) != result.get(k)
            for k in (
                "run_valid",
                "effective_neural_qualification",
                "handoff_eligible",
                "first_error",
                "errors",
            )
        )
        or result.get("first_error") is not None
        or result.get("errors") != []
    ):
        raise ValueError("E334 valid matching terminal schemas/fields required")
    if (
        producer.get("source_sha256") != prior["source_sha256"]
        or producer.get("input_sha256") != prior["input_sha256"]
        or producer.get("freeze_sha256") != pins[PATHS["e334_freeze"]]
        or producer.get("review_sha256") != pins[PATHS["e334_review"]]
        or result.get("provenance")
        != {
            "freeze_sha256": pins[PATHS["e334_freeze"]],
            "review_sha256": pins[PATHS["e334_review"]],
        }
    ):
        raise ValueError("E334 terminal producer links differ")
    if (
        audit.get("schema") != "flylab.m3.e334.root-original-raw-audit.v1"
        or audit.get("passed") is not True
        or len(audit.get("checks", {})) != 5094
        or not all(v is True for v in audit["checks"].values())
        or audit.get("audit_source_sha256") != pins[PATHS["e334_audit_source"]]
        or audit.get("freeze_sha256") != pins[PATHS["e334_freeze"]]
        or audit.get("review_sha256") != pins[PATHS["e334_review"]]
        or audit.get("supervision_sha256") != pins[PATHS["e334_supervision"]]
    ):
        raise ValueError("exact passed5094 v2 raw audit required")
    for name, digest in audit.get("terminal_sha256", {}).items():
        if pins.get(PREVIOUS + "/" + name) != digest:
            raise ValueError("audit terminal digest differs")
    if set(audit.get("terminal_sha256", {})) != {
        "result.json",
        "advance-ledger.json",
        "output-manifest.json",
        "source-manifest.json",
        "receipts.json",
    }:
        raise ValueError("complete audit terminal pins required")
    expected_receipt = {
        "forward_controls": 300,
        "forward_neural_ticks": 6000,
        "recompute_controls": 0,
        "recompute_neural_ticks": 0,
        "backward_blocks": 0,
        "backward_block_ticks": 0,
    }
    if set(receipts.get("receipts", {})) != {"continuation-star"}:
        raise ValueError("once6000 receipt roster required")
    rec = receipts["receipts"]["continuation-star"]
    if (
        rec.get("confirmed") != expected_receipt
        or rec.get("confirm_returned") != 1
        or rec.get("confirm_requested") != 1
        or result.get("receipts") != receipts["receipts"]
        or ledger.get("receipts") != receipts["receipts"]
        or rec.get("forward_neural_ticks_requested") != 6000
        or rec.get("forward_neural_ticks_returned") != 6000
    ):
        raise ValueError("E334 exact separately confirmed single forward required")
    roster = {*e334.RAW_FILES, "source-manifest.json", "receipts.json"}
    if set(manifest.get("files", {})) != roster:
        raise ValueError("E334 exact7raw+2metadata roster required")
    for name, desc in manifest["files"].items():
        if desc != {
            "sha256": pins.get(PREVIOUS + "/" + name),
            "bytes": safe(PREVIOUS + "/" + name, root).stat().st_size,
        }:
            raise ValueError("E334 immutable output descriptor differs")
    for source, digest in prior["source_sha256"].items():
        if sources.get(source) != digest:
            raise ValueError("E334 producer source replaced")
    for role in ("e334", "e334_audit"):
        _completed_supervision(
            proofs[role + "_supervision"],
            proofs[role + "_contract"],
            pins[PATHS[role + "_contract"]],
            original_calls=int(role == "e334"),
        )
    sup = proofs["e334_supervision"]
    if (
        sup.get("cumulative_after_s") != 9431.526588899522
        or sup.get("cumulative_limit_s") != 9660
        or sup.get("elapsed_s") != decision["original_supervisor_elapsed_s"]
        or decision.get("root_saved_audit_CPU_failed_included_s") != 10.433235699998477
    ):
        raise ValueError("historical failed-inclusive external cost differs")


def normalized_producer_path(value, *, root=ROOT):
    """Resolve slash/absolute aliases within the repository; reject escapes."""
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError("nonempty producer path required")
    windows = PureWindowsPath(value)
    if windows.drive and not windows.is_absolute():
        raise ValueError("drive-relative producer path is ambiguous")
    return safe(value.replace("\\", "/"), root)


def verify_generated_execution(
    proof, report, contract, supervision, *, role, root=ROOT
):
    """Bind the executed, pinned Python helper to this exact report artifact."""
    helper = normalized_producer_path(proof["audit_source_path"], root=root)
    executed_sources = {}
    for name, digest in contract.get("source_sha256", {}).items():
        path = normalized_producer_path(name, root=root)
        if path in executed_sources and executed_sources[path] != digest:
            raise ValueError("contradictory executed producer path aliases")
        executed_sources[path] = digest
    if (
        executed_sources.get(helper) != proof["audit_source_sha256"]
        or sha256(helper) != proof["audit_source_sha256"]
    ):
        raise ValueError("exact actual auditor helper must be an executed source pin")
    commands, attempts = contract.get("commands", []), supervision.get("attempts", [])
    if len(commands) != 1 or len(attempts) != 1:
        raise ValueError("sole generated command/attempt required")
    argv = commands[0].get("argv")
    if (
        not isinstance(argv, list)
        or not all(isinstance(v, str) for v in argv)
        or attempts[0].get("argv") != argv
    ):
        raise ValueError("actual supervisor and contract argv must be exactly equal")
    if (
        len(argv) < 7
        or argv[1:4] != ["-X", "utf8", "-B"]
        or argv[4].startswith("-")
        or helper.suffix != ".py"
        or normalized_producer_path(argv[0], root=root)
        != normalized_producer_path(".venv/Scripts/python.exe", root=root)
        or normalized_producer_path(argv[4], root=root) != helper
    ):
        raise ValueError("exact repository Python -X utf8 -B auditor helper required")
    outputs = {}
    arguments = argv[5:]
    if "--" in arguments:
        raise ValueError("producer output options must precede no option terminator")
    for flag in ("--output", "--decoder-output"):
        matches = []
        for index, value in enumerate(arguments):
            if value == flag:
                if index + 1 >= len(arguments) or arguments[index + 1].startswith("--"):
                    raise ValueError("producer output path missing")
                matches.append(arguments[index + 1])
            elif value.startswith(flag + "="):
                matches.append(value[len(flag) + 1 :])
        if len(matches) > 1:
            raise ValueError("ambiguous duplicate producer output option")
        if matches:
            outputs[flag] = normalized_producer_path(matches[0], root=root)
    flag = (
        "--decoder-output"
        if role == "decoder" and "--decoder-output" in outputs
        else "--output"
    )
    if outputs.get(flag) != normalized_producer_path(report["path"], root=root):
        raise ValueError("actual helper command must produce this exact report path")


def _generated_proof(qual, sources, pins, *, root):
    if (
        qual.get("schema") != QUAL_SCHEMA
        or qual.get("approved") is not True
        or qual.get("blocking_defects") != []
        or qual.get("source_sha256") != {k: sources[k] for k in OWNED}
        or qual.get("original_calls") != 0
        or not qual.get("evidence_sha256")
    ):
        raise ValueError("current generated own2 qualification required")
    for role, scope in (
        ("cpu", "fake-body"),
        ("cuda", "native-ssp"),
        ("decoder", "native-functional-decoder"),
    ):
        desc = qual.get(role + "_report", {})
        if set(desc) != {"path", "sha256"} or pins.get(desc["path"]) != desc["sha256"]:
            raise ValueError("exact separate generated report descriptor required")
        proof = read(safe(desc["path"], root))
        if (
            proof.get("schema") != "flylab.m3.e335.root-generated.v1"
            or proof.get("scope") != scope
            or proof.get("passed") is not True
            or not proof.get("checks")
            or not all(v is True for v in proof["checks"].values())
            or proof.get("source_sha256") != {k: sources[k] for k in OWNED}
            or any(
                proof.get(k) != 0
                for k in ("original_calls", "whole_BANC_calls", "body_calls")
            )
            or pins.get(proof.get("audit_source_path"))
            != proof.get("audit_source_sha256")
        ):
            raise ValueError("current separate scope/source generated proof required")
        for suffix in ("supervision", "contract"):
            producer_desc = qual.get(role + "_" + suffix, {})
            if (
                set(producer_desc) != {"path", "sha256"}
                or pins.get(producer_desc["path"]) != producer_desc["sha256"]
            ):
                raise ValueError(
                    "generated completed supervisor producer link required"
                )
        supervision = read(safe(qual[role + "_supervision"]["path"], root))
        contract = read(safe(qual[role + "_contract"]["path"], root))
        _completed_supervision(
            supervision,
            contract,
            qual[role + "_contract"]["sha256"],
            original_calls=0,
        )
        if any(contract.get("source_sha256", {}).get(k) != sources[k] for k in OWNED):
            raise ValueError("completed generated contract must cover current own2")
        verify_generated_execution(
            proof, desc, contract, supervision, role=role, root=root
        )
    sequence = qual.get("root_review_sequence", {})
    if set(sequence) != {"FIRST", "SECOND", "reflection"}:
        raise ValueError(
            "complete root FIRST/SECOND/reflection source sequence required"
        )
    for label, schema in (
        ("FIRST", "flylab.m3.e335.root-first-review.v1"),
        ("SECOND", "flylab.m3.e335.root-second-review.v1"),
        ("reflection", "flylab.m3.e335.root-final-source-reflection.v1"),
    ):
        desc = sequence[label]
        if set(desc) != {"path", "sha256"} or pins.get(desc["path"]) != desc["sha256"]:
            raise ValueError("source review sequence descriptor differs")
        proof = read(safe(desc["path"], root))
        if proof.get("schema") != schema:
            raise ValueError("root source review sequence schema differs")
        if label == "FIRST" and (
            set(proof.get("reviewed_source_sha256", {})) != set(OWNED)
            or type(proof.get("approved")) is not bool
            or not isinstance(proof.get("blocking_defects"), list)
        ):
            raise ValueError("preserved FIRST source/approval/defects required")
        if label != "FIRST" and (
            proof.get("current_source_sha256") != {k: sources[k] for k in OWNED}
            or proof.get("approved") is not True
            or proof.get("blocking_defects") != []
        ):
            raise ValueError("current approved SECOND/final-source reflection required")
        if label == "reflection" and (
            proof.get("actual_astra_reflection_sha256") != ANCHORS["reflection"]
            or proof.get("first_review_sha256") != sequence["FIRST"]["sha256"]
            or proof.get("second_review_sha256") != sequence["SECOND"]["sha256"]
        ):
            raise ValueError(
                "final-source reflection actual Astra/FIRST/SECOND links required"
            )


def gate(
    freeze_path,
    freeze_sha256,
    review_path,
    review_sha256,
    output,
    *,
    root=ROOT,
    timely=lambda: None,
):
    timely()
    if (
        Path(output).exists()
        or Path(output).resolve() != safe(OUTPUT, root)
        or sha256(freeze_path) != freeze_sha256
        or sha256(review_path) != review_sha256
    ):
        raise ValueError("fresh literal E335/exact freeze-review required")
    frozen, review = read(freeze_path), read(review_path)
    if (
        frozen.get("schema") != SCHEMA
        or frozen.get("paths") != PATHS
        or frozen.get("limits") != LIMITS
        or frozen.get("output") != OUTPUT
        or frozen.get("original_execution_authorized") is not True
        or frozen.get("allocation") != ALLOCATION
        or frozen.get("original_supervisor_required") is not True
    ):
        raise ValueError("exact root-authorized fixed E335 contract required")
    sources, pins = frozen["source_sha256"], frozen["input_sha256"]
    pin_map(sources, root, timely)
    pin_map(pins, root, timely)
    for role, digest in ANCHORS.items():
        if pins.get(PATHS[role]) != digest:
            raise ValueError("fixed reflected/previous anchor differs")
    prior, _body, expected_sources, expected_pins = _closure(root)
    expected = make_freeze(root=root)
    if set(frozen) != set(expected) or any(
        frozen[k] != expected[k]
        for k in expected.keys() - {"allocation", "original_execution_authorized"}
    ):
        raise ValueError("exact frozen E335 calculation/closure required")
    if (
        sources != expected_sources
        or pins != expected_pins
        or any(sources[k] != pins[k] for k in sources.keys() & pins.keys())
    ):
        raise ValueError("inherited producer collision/closure differs")
    if (
        review.get("schema") != REVIEW_SCHEMA
        or review.get("approved") is not True
        or review.get("blocking_defects") != []
        or review.get("source_sha256") != sources
        or review.get("freeze_sha256") != freeze_sha256
    ):
        raise ValueError("root FIRST/SECOND current source review required")
    qual = read(safe(PATHS["qualification"], root))
    for proof in (review, qual):
        if any(
            {**sources, **pins}.get(k) != v
            for k, v in proof.get("evidence_sha256", {}).items()
        ):
            raise ValueError("review/qualification producer pin mismatch")
    _generated_proof(qual, sources, pins, root=root)
    proofs = {
        role: read(safe(PATHS[alias], root))
        for role, alias in (
            ("decision", "e334_decision"),
            ("result", "e334_result"),
            ("ledger", "e334_ledger"),
            ("producer", "e334_producer"),
            ("audit", "e334_audit"),
            ("manifest", "e334_manifest"),
            ("receipts", "e334_receipts"),
            ("e334_supervision", "e334_supervision"),
            ("e334_contract", "e334_contract"),
            ("e334_audit_supervision", "e334_audit_supervision"),
            ("e334_audit_contract", "e334_audit_contract"),
        )
    }
    verify_previous({**proofs, "freeze": prior}, pins, sources, root=root)
    reflection = read(safe(PATHS["reflection"], root))
    if (
        reflection.get("all_M1_M7_reflected") is not True
        or reflection.get("reviewed_design_sha256") != ANCHORS["design"]
        or reflection.get("actual_review_sha256") != ANCHORS["astra"]
        or reflection.get("invocation_sha256") != pins[PATHS["invocation"]]
    ):
        raise ValueError("actual Astra/reflection producer link required")
    invocation = read(safe(PATHS["invocation"], root))
    cli = invocation.get("CLI_current", {})
    if (
        invocation.get("requested_model") != "gpt-6-astra"
        or invocation.get("requested_reasoning_effort") != "max"
        or cli.get("model") != "gpt-6-astra"
        or cli.get("config", {}).get("model_reasoning_effort") != "max"
        or cli.get("completed") is not True
        or cli.get("exit_code") != 0
        or cli.get("turn_completed_event", {}).get("type") != "turn.completed"
    ):
        raise ValueError("actual Astra/max completed invocation required")
    timely()
    return frozen


@dataclass(frozen=True)
class TestDimensions:
    nodes: int
    groups: int
    controls: int = 300
    test_only: bool = True

    def __post_init__(self):
        if (
            self.test_only is not True
            or type(self.nodes) is not int
            or self.nodes < 391
            or type(self.groups) is not int
            or self.groups < 1
            or self.controls != 300
        ):
            raise ValueError("explicit generated391+/groups/300 fixture required")


def _host(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "get"):
        value = value.get()
    return np.array(value, copy=True)


def _exact(a, b, label):
    if (
        not isinstance(a, np.ndarray)
        or not isinstance(b, np.ndarray)
        or a.dtype != b.dtype
        or a.shape != b.shape
        or a.tobytes() != b.tobytes()
    ):
        raise ValueError("exact dtype/shape/bytes differ: " + label)


def _typed(value, shape, dtype, label):
    e334._array(value, shape, dtype, label)
    return value


def verify_payload(
    native,
    theta_bundle,
    selected_bundle,
    patched_bundle,
    *,
    nodes=N,
    groups=GROUPS,
    expected_palette=8040,
):
    from tools.verify_c_banc_m3_temporal_direction import patch_weights

    theta = _typed(theta_bundle["theta64"], (groups,), np.float64, "theta64")
    _exact(theta_bundle["theta32"], theta.astype(np.float32), "theta32")
    if np.any(theta <= 0) or not np.array_equal(
        theta, theta.astype(np.float32).astype(np.float64)
    ):
        raise ValueError("positive actual f32-promoted efficacy required")
    positions, group = native["mask"]["positions"], native["mask"]["edge_parameters"]
    _exact(selected_bundle["positions"], positions, "ordered selected positions")
    _exact(selected_bundle["groups"], group, "ordered groups")
    csr, actual, info = patch_weights(native, theta)
    if (
        info["node_count"] != nodes
        or not info["eligible"]
        or (
            expected_palette is not None
            and info["full_csr_bit_palette_count"] != expected_palette
        )
    ):
        raise ValueError("fixed native palette/topology eligibility required")
    for key in ("data", "indices", "indptr"):
        _exact(patched_bundle[key], csr[key], "once-rounded CSR " + key)
    _exact(selected_bundle["actual32"], actual, "selected actual32")
    _exact(
        selected_bundle["original32"],
        native["csr"]["data"][positions],
        "selected original32",
    )
    return info


def verify_packed(network, csr, *, expected_palette=None):
    """Actual kernel buffers, pointer assertion, decoded topology and f32 bits."""
    if not callable(getattr(network, "_assert_pointers", None)):
        raise TypeError("actual captured pointer/version guard required")
    network._assert_pointers()
    if network.cuda_implementation != "packed":
        raise ValueError("actual packed kernel required")
    rowptr, columns, palette = (
        _host(getattr(network, k)) for k in ("_rowptr", "_columns", "_weight_values")
    )
    _typed(rowptr, csr["indptr"].shape, np.int32, "packed rowptr")
    _typed(columns, csr["indices"].shape, np.int32, "packed columns")
    _typed(palette, palette.shape, np.float32, "packed palette")
    if palette.ndim != 1 or len(palette) > 16384:
        raise ValueError("packed palette14bits required")
    bits = columns.view(np.uint32)
    index = bits >> 18
    cols = (bits & ((1 << 18) - 1)).astype(np.int32)
    if np.any(index >= len(palette)):
        raise ValueError("packed palette index outside actual buffers")
    _exact(rowptr, csr["indptr"], "packed row topology")
    _exact(cols, csr["indices"], "packed column order")
    _exact(
        palette[index].view(np.uint32),
        csr["data"].view(np.uint32),
        "packed weight f32 bits",
    )
    count = len(np.unique(csr["data"].view(np.uint32)))
    if len(palette) != count or (
        expected_palette is not None and count != expected_palette
    ):
        raise ValueError("actual packed full bit palette differs")
    return {
        "captured_pointer_version_guard": True,
        "packed_rowptr_sha256": bytes_sha(rowptr),
        "packed_columns_sha256": bytes_sha(columns),
        "packed_palette_sha256": bytes_sha(palette),
        "decoded_data32_sha256": bytes_sha(palette[index]),
        "palette": count,
        "device": str(getattr(network, "device", "unavailable")),
        "host_CSR_and_devicepacked_independently_compared": True,
    }


def verify_decoder(adapter, saved_adapter, approved):
    actual = adapter.snapshot()
    for key in ("bank", "activation"):
        _exact(actual[key], saved_adapter[key], "fresh decoder " + key)
    _exact(
        actual["original"]["activation"],
        saved_adapter["original"]["activation"],
        "fresh original activation",
    )
    for key in ("moment", "rest", "pool", "motor_ids", "ltm"):
        _exact(
            actual["calibration"][key],
            saved_adapter["calibration"][key],
            "native calibration " + key,
        )
        _exact(actual["calibration"][key], approved[key], "approved calibration " + key)
    for key in ("tau", "kappa", "membership"):
        _exact(
            actual["parameters"][key],
            saved_adapter["parameters"][key],
            "native parameter " + key,
        )
        _exact(actual["parameters"][key], approved[key], "approved parameter " + key)
    slopes = actual["parameters"]["slopes"]
    _exact(slopes, saved_adapter["parameters"]["slopes"], "native slopes")
    low = slopes[..., 0].reshape(96, 1173).copy()
    delta = (slopes[..., 1] - slopes[..., 0]).reshape(96, 1173).copy()
    fixed = np.zeros(96, bool)
    fixed[actual["parameters"]["fixed_muscles"]] = True
    for key, value in (
        ("low", low),
        ("delta", delta),
        ("fixed_mask", fixed),
        ("bank", actual["bank"]),
        ("original_activation", actual["original"]["activation"]),
    ):
        _exact(value, approved[key], "functional decoder " + key)
    ids = actual["calibration"]["motor_ids"]
    _typed(ids, (391,), np.int64, "ordered motor IDs")
    if not np.array_equal(ids, np.sort(np.unique(ids))):
        raise ValueError("391ordered unique motorIDs required")
    # AnatomicalRecruitmentAdapter.decode and MonotoneDecoder.step both use
    # the immutable .01-second activation constant; it is not a saved key.
    _exact(
        np.array(0.01, np.float64),
        approved["activation_tau"],
        "immutable activation tau",
    )
    _exact(np.asarray(adapter.body.neutral), approved["neutral"], "actual body neutral")
    return {
        "content_sha256": {
            key: bytes_sha(value)
            for key, value in {
                "low": low,
                "delta": delta,
                "fixed_mask": fixed,
                "motor_ids": ids,
                "bank": actual["bank"],
                "original_activation": actual["original"]["activation"],
            }.items()
        },
        "semantic_leaves": {
            "low": ["adapter", "parameters", "slopes"],
            "delta": ["adapter", "parameters", "slopes"],
            "fixed_mask": ["adapter", "parameters", "fixed_muscles"],
            "bank": ["adapter", "bank"],
            "original_activation": ["adapter", "original", "activation"],
            "calibration": ["adapter", "calibration"],
            "neutral": ["body", "action", "joint_angles"],
        },
        "activation_tau_s": 0.01,
        "activation_tau_producer": "tools/banc_m3_rhythmic_recruitment.py:AnatomicalRecruitmentAdapter.decode",
    }


def verify_construction(ledger):
    neural, body = ledger.get("construction", []), ledger.get("body_construction", [])
    if len(neural) != 1 or len(body) != 1:
        raise ValueError("exact one owned construction ledger required")
    n, b = neural[0], body[0]
    if (
        n.get("known_completed_setup_steps") != 320
        or n.get("recorded_capture_setup_steps") != 160
        or n.get("unknown_setup_steps") != 0
        or n.get("unknown_capture_setup_steps") != 0
        or b.get("known_completed_settling_steps") != 2000
        or b.get("unknown_step_entries") != 0
        or b.get("active_unreturned_frames") != 0
    ):
        raise ValueError("exact separate setup320/capture160/settling2000 required")
    if (
        n.get("interval_entries") != 24
        or n.get("interval_returns") != 24
        or n.get("returned_executed_setup_steps") != 320
        or n.get("returned_capture_setup_steps") != 160
        or n.get("synchronize_returned") is not True
        or n.get("profiler_restored") is not True
        or n.get("ownership_transferred") is not True
        or b.get("entered") != 2000
        or b.get("returned") != 2000
        or b.get("exception_unwinds") != 0
        or b.get("trace_restored") is not True
    ):
        raise ValueError(
            "returned/synchronized setup and restored owned profilers required"
        )
    return {
        "setup_neural_ticks": 320,
        "capture_recorded_ticks_subset": 160,
        "settling_physics_steps": 2000,
    }


class InstanceHooks:
    """Restore whether a method originally lived on the class or instance."""

    def __init__(self):
        self.saved = []

    def wrap(self, owner, name, factory):
        present = name in vars(owner)
        original = getattr(owner, name)
        self.saved.append((owner, name, present, vars(owner).get(name)))
        setattr(owner, name, factory(original))

    def detach(self):
        errors = []
        while self.saved:
            owner, name, present, value = self.saved.pop()
            try:
                if present:
                    setattr(owner, name, value)
                else:
                    delattr(owner, name)
            except Exception as error:  # noqa: BLE001
                errors.append(error)
        if errors:
            raise RuntimeError("instance restore failures: " + repr(errors))


class LiveInputObserver:
    """Capture actual encode calls, before and after descending addition.

    Returned-work receipts precede copies, finite checks and journals. Partial
    fields have separate control-ID arrays; absent data are never synthesized.
    """

    def __init__(
        self,
        session,
        output,
        ledger,
        *,
        nodes=N,
        chunk_controls=25,
        saver=np.savez,
        writer=put_json,
        clock=time.perf_counter,
    ):
        self.session, self.output, self.ledger = session, Path(output), ledger
        self.nodes, self.chunk_controls, self.saver, self.writer, self.clock = (
            nodes,
            chunk_controls,
            saver,
            writer,
            clock,
        )
        self.hooks, self.current, self.encoding = InstanceHooks(), None, False
        self.rows, self.events, self.chunks = [], [], []
        self.ledger.setdefault("readout_returns", [])
        self.ledger.setdefault("physics_steps_host_returned", 0)
        self.output.mkdir(exist_ok=True)
        self.io = {"write_s": 0.0, "hash_s": 0.0, "raw_payload_bytes": 0, "chunks": 0}

    def event(self, name, **extra):
        self.events.append(
            {
                "control": self.current["control"] if self.current else None,
                "event": name,
                "at_s": self.clock(),
                **extra,
            }
        )

    def field(self, name, value, shape, dtype):
        array = np.array(value, copy=True)
        self.current["fields"][name] = array  # Preserve even invalid/NaN acquired raw.
        _typed(array, shape, dtype, name)
        self.event(
            name + "_acquired",
            sha256=bytes_sha(array),
            dtype=array.dtype.str,
            shape=list(array.shape),
        )

    def install(self):
        body, adapter, network = (
            self.session.body,
            self.session.adapter,
            self.session.network,
        )

        def physics(original):
            def call(*args, **kwargs):
                value = original(*args, **kwargs)
                self.ledger["physics_steps_host_returned"] += 1
                return value

            return call

        # Installed before the original contact observer wraps this method.
        self.hooks.wrap(body, "_step_physics", physics)

        def pose(original):
            def call(*args, **kwargs):
                value = original(*args, **kwargs)
                if (
                    self.current is not None
                    and "prior_position" not in self.current["fields"]
                ):
                    self.field("prior_position", value[0], (3,), np.float64)
                    self.field("prior_rotation", value[1], (3, 3), np.float64)
                return value

            return call

        self.hooks.wrap(body, "pose", pose)

        def legs(original):
            def call(*args, **kwargs):
                value = original(*args, **kwargs)
                if self.encoding:
                    self.event("leg_observation_returned")
                    for key, shape in (
                        ("angles_rad", (42,)),
                        ("velocities_rad_s", (42,)),
                        ("support_load_bw", (6,)),
                        ("non_support_load_bw", (6,)),
                    ):
                        self.field(key, value[key], shape, np.float64)
                return value

            return call

        self.hooks.wrap(body, "leg_observation", legs)

        def knee(original):
            def call(*args, **kwargs):
                value = original(*args, **kwargs)
                if self.encoding:
                    self.event(
                        "knee_kinematics_returned",
                        include_direction=kwargs.get("include_direction"),
                    )
                    for key in ("flexion_rad", "velocity_rad_s"):
                        self.field("knee_" + key, value[key], (6,), np.float64)
                return value

            return call

        self.hooks.wrap(body, "knee_kinematics", knee)

        def encode(original):
            def call(*args, **kwargs):
                if self.current is None:
                    raise ValueError("encode outside owned control")
                self.event("encode_requested", sensory_cut=kwargs.get("sensory_cut"))
                self.encoding = True
                try:
                    value = original(*args, **kwargs)
                finally:
                    self.encoding = False
                self.event("encode_returned")
                self.field("encoded_drive", value, (self.nodes,), np.float32)
                return value

            return call

        self.hooks.wrap(adapter, "encode", encode)

        def readout(original):
            def call(*args, **kwargs):
                self.event("advance_readout_requested")
                self.field(
                    "actual_drive",
                    args[0] if args else kwargs["drive"],
                    (self.nodes,),
                    np.float32,
                )
                try:
                    value = original(*args, **kwargs)
                except BaseException as error:
                    receipt = getattr(error, "receipt", None)
                    try:
                        json.dumps(receipt, allow_nan=False)
                        self.current["neural_exception_receipt"] = copy.deepcopy(
                            receipt
                        )
                    except (TypeError, ValueError):
                        self.current["neural_exception_receipt_unserializable"] = repr(
                            receipt
                        )[:2000]
                    self.current["neural_exception_receipt_provided"] = (
                        receipt is not None
                    )
                    self.current["unreturned_neural_work_unknown"] = True
                    self.event("advance_readout_failed")
                    raise
                # Actual synchronous advance_readout returns a host MN vector.
                # Store the return receipt first, before any acquisition can fail.
                steps = int(args[2] if len(args) > 2 else kwargs["steps"])
                receipt = {
                    "control": self.current["control"],
                    "steps": steps,
                    "returned": True,
                    "device_confirmed": True,
                }
                self.ledger["readout_returns"].append(receipt)
                self.ledger["neural_host_returned"] += steps
                self.ledger["neural_device_confirmed"] += steps
                self.event("advance_readout_returned", steps=steps)
                self.field("motor_readout", value, (391,), np.float32)
                return value

            return call

        self.hooks.wrap(network, "advance_readout", readout)

        def advance(original):
            def call(*args, **kwargs):
                value = original(*args, **kwargs)
                count = int(args[0] if args else kwargs["controls"])
                self.ledger["controls_host_returned"] += count
                self.event("original_control_call_returned", requested_controls=count)
                return value

            return call

        self.hooks.wrap(self.session, "advance", advance)

    def begin(self, control):
        if self.current is not None:
            raise ValueError("previous observation unfinished")
        session = self.session
        self.current = {
            "control": int(control),
            "fields": {},
            "control_before": int(session.control_tick),
            "neural_before": int(session.network.tick),
            "body_before": float(session.body.physics_time()),
            "physics_returned_before": self.ledger["physics_steps_host_returned"],
        }
        self.event("control_requested")

    def end(self, *, returned=False, error=None):
        row = self.current
        if row is None:
            return
        row["control_returned"] = bool(returned)
        row["error"] = (
            None
            if error is None
            else {"type": type(error).__name__, "message": str(error)[:500]}
        )
        # Clock reads are independently fallible; never infer a confirmation.
        for label, callback in (
            ("control_after", lambda: int(self.session.control_tick)),
            ("neural_after", lambda: int(self.session.network.tick)),
            ("body_after", lambda: float(self.session.body.physics_time())),
        ):
            try:
                row[label] = callback()
            except Exception as secondary:  # noqa: BLE001
                row[label] = None
                row[label + "_error"] = repr(secondary)[:500]
        row["physics_returned_after"] = self.ledger["physics_steps_host_returned"]
        self.event("control_returned" if returned else "control_failed")
        self.rows.append(row)
        self.current = None
        if len(self.rows) >= self.chunk_controls:
            self.flush()
        if returned:
            mandatory = {
                "prior_position",
                "prior_rotation",
                "angles_rad",
                "velocities_rad_s",
                "support_load_bw",
                "non_support_load_bw",
                "knee_flexion_rad",
                "knee_velocity_rad_s",
                "encoded_drive",
                "actual_drive",
                "motor_readout",
            }
            if set(row["fields"]) != mandatory:
                raise ValueError("mandatory actual live observation missing")
            events = [v for v in self.events if v["control"] == row["control"]]
            for name in (
                "encode_requested",
                "encode_returned",
                "leg_observation_returned",
                "knee_kinematics_returned",
                "advance_readout_returned",
            ):
                if sum(v["event"] == name for v in events) != 1:
                    raise ValueError("actual single call order/count differs: " + name)
            if (
                next(v for v in events if v["event"] == "knee_kinematics_returned")[
                    "include_direction"
                ]
                is not False
            ):
                raise ValueError("original knee kwargs differ")

    def flush(self):
        if not self.rows:
            return
        rows = self.rows
        arrays = {"control_ids": np.asarray([r["control"] for r in rows], np.int64)}
        for name in sorted({k for r in rows for k in r["fields"]}):
            available = [r for r in rows if name in r["fields"]]
            arrays[name + "_control_ids"] = np.asarray(
                [r["control"] for r in available], np.int64
            )
            arrays[name] = np.stack([r["fields"][name] for r in available])
        for name, dtype in (
            ("control_before", np.int64),
            ("neural_before", np.int64),
            ("body_before", np.float64),
            ("control_after", np.int64),
            ("neural_after", np.int64),
            ("body_after", np.float64),
            ("physics_returned_before", np.int64),
            ("physics_returned_after", np.int64),
        ):
            available = [
                r for r in rows if r.get(name) is not None and np.isfinite(r[name])
            ]
            arrays[name + "_control_ids"] = np.asarray(
                [r["control"] for r in available], np.int64
            )
            arrays[name] = np.asarray([r[name] for r in available], dtype=dtype)
        path = (
            self.output
            / f"live-input-{rows[0]['control']:03d}-{rows[-1]['control']:03d}.npz"
        )
        began = self.clock()
        self.saver(path, **arrays)
        self.io["write_s"] += self.clock() - began
        began = self.clock()
        digest = sha256(path)
        self.io["hash_s"] += self.clock() - began
        self.io["raw_payload_bytes"] += sum(a.nbytes for a in arrays.values())
        self.io["chunks"] += 1
        self.chunks.append(
            {
                "file": path.name,
                "sha256": digest,
                "bytes": path.stat().st_size,
                "controls": arrays["control_ids"].tolist(),
                "arrays": {k: e334._metadata(v) for k, v in arrays.items()},
                "availability": [
                    {
                        k: v
                        for k, v in r.items()
                        if k != "fields"
                        and (not isinstance(v, float) or np.isfinite(v))
                    }
                    for r in rows
                ],
            }
        )
        self.rows = []

    def publish(self):
        self.flush()
        self.writer(
            self.output / "live-input-events.json",
            {
                "schema": "flylab.m3.e335.live-input.v1",
                "events": self.events,
                "chunks": self.chunks,
                "io": self.io,
                "nodes": self.nodes,
                "configuration": {
                    "dt": 0.005,
                    "neural_steps": 20,
                    "cuts": copy.deepcopy(self.session.cuts),
                    "descending_ids": np.asarray(self.session.descending).tolist(),
                    "descending_drive": float(self.session.p.descending_drive),
                    "navigation": self.session.navigation is not None,
                },
                "missing_fields_are_not_synthesized": True,
            },
        )

    def detach(self):
        self.hooks.detach()


def decoder_approved(saved, calibration, parameters, actual_proof):
    """Host decoder constants, compared to E334's actual used tensor metadata."""
    a = saved["adapter"]
    approved = {
        **{k: calibration[k] for k in ("moment", "rest", "pool", "motor_ids", "ltm")},
        **{k: parameters[k] for k in ("tau", "kappa", "membership")},
        "bank": a["bank"],
        "original_activation": a["original"]["activation"],
        "activation_tau": np.array(0.01, np.float64),
        "neutral": np.asarray(saved["body"]["action"]["joint_angles"]),
    }
    slopes = parameters["slopes"]
    approved["low"] = slopes[..., 0].reshape(96, 1173).copy()
    approved["delta"] = (slopes[..., 1] - slopes[..., 0]).reshape(96, 1173).copy()
    counts = np.bincount(parameters["membership"], minlength=96)
    fixed = np.zeros(96, bool)
    fixed[np.union1d(calibration["ltm"], np.flatnonzero(counts == 1))] = True
    approved["fixed_mask"] = fixed
    for key in (
        "moment",
        "rest",
        "pool",
        "ltm",
        "tau",
        "kappa",
        "neutral",
        "low",
        "delta",
        "fixed_mask",
    ):
        if actual_proof.get("decoder." + key) != e334._metadata(approved[key]):
            raise ValueError("E334 actual functional decoder proof differs: " + key)
    for key, alias in (
        ("bank", "bank"),
        ("original_activation", "original_activation"),
        ("motor_ids", "motor_ids"),
    ):
        if actual_proof.get(alias) != e334._metadata(approved[key]):
            raise ValueError("E334 actual start decoder bytes differ: " + key)
    return approved


class Dependencies:
    """Lazy actual native loader with separately inherited E318 configuration."""

    def __init__(self):
        from tools import banc_m3_ssp_efficacy_trial as layer
        from tools import verify_c_banc_m3_full_drive_capture as e318

        self.layer, self.e318 = layer, e318
        self.base = e318.Dependencies()
        self.native = self.base.native
        self.wrapper = self.base.wrapper
        self.health_stop = self.base.health_stop

    def load(self, freeze, *, root=ROOT):
        from tools import verify_c_banc_m3_full_horizon_direction as e323
        from tools import verify_c_banc_m3_smooth_efficacy_direction as e333
        from tools.verify_c_banc_m3_native_adjoint import load_inputs

        paths = e323.REQUIRED_PATHS
        data = self.base.load(freeze["e318_config"])
        native = load_inputs(read(safe(paths["e315_freeze"], root)), root=root)
        theta = e334.load_selected(
            safe(PREVIOUS + "/theta-star.npz", root), ("theta64", "theta32")
        )
        selected = e334.load_selected(
            safe(PREVIOUS + "/selected-weights-star.npz", root),
            ("positions", "groups", "original32", "actual32"),
        )
        patched = e334.load_selected(
            safe(PREVIOUS + "/patched-csr-star.npz", root),
            ("data", "indices", "indptr"),
        )
        proof = verify_payload(native, theta, selected, patched)
        if (
            bytes_sha(theta["theta64"]) != THETA_SHA
            or bytes_sha(patched["data"]) != CSR_SHA
        ):
            raise ValueError("single E334 candidate bytes required")
        _exact(
            data["roster"]["edge_ids"],
            native["mask"]["positions"],
            "original roster/mask",
        )
        calibration = e334.load_selected(
            safe(paths["e150_calibration"], root),
            ("moment", "rest", "pool", "motor_ids", "ltm"),
        )
        parameters = e334.load_selected(
            safe(paths["e150_parameters"], root),
            ("tau", "kappa", "slopes", "membership"),
        )
        prior_result = read(safe(PATHS["e334_result"], root))
        approved = decoder_approved(
            data["saved"],
            calibration,
            parameters,
            prior_result["partial"]["actual_input_proof"],
        )
        start = e334.load_selected(
            safe(PREVIOUS + "/continuation-start.npz", root),
            ("rate", "adaptation", "bank", "original_activation", "full_mask"),
        )
        for key, value in (
            ("rate", data["saved"]["network"]["rate"]),
            ("adaptation", data["saved"]["network"]["adaptation"]),
            ("bank", data["saved"]["adapter"]["bank"]),
            ("original_activation", data["saved"]["adapter"]["original"]["activation"]),
            ("full_mask", data["saved"]["network"]["output_mask"]),
        ):
            _exact(start[key], value, "E334 fresh start " + key)
        # Native full mask/control0/100/300 and physiological loader states.
        for whole in (data["saved"], data["e307"], data["e310"]):
            _exact(
                whole["network"]["output_mask"],
                native["states"]["initial_neural"]["output_mask"],
                "full output mask",
            )
        startup = e334.load_selected(
            safe(paths["e103_startup"], root), ("target_command", "target_tendon")
        )
        reference = e334.load_selected(
            safe(paths["e85_reference"], root), ("command", "phase", "time_s")
        )
        constants = e334.load_selected(
            safe(e333.SAVED + "/reference.npz", root),
            ("curves", "normalization", "temperature", "future_anchors", "phase_basis"),
        )
        _exact(
            constants["curves"], e323.curve128(reference), "approved reference curve"
        )
        data.update(
            native=native,
            theta=theta["theta64"],
            positions=native["mask"]["positions"],
            edge_parameters=native["mask"]["edge_parameters"],
            patched=patched,
            payload_proof=proof,
            decoder_approved=approved,
            startup=startup,
            curve=constants["curves"],
            smooth_constants=constants,
            smooth_constant_metadata=freeze["smooth_constant_metadata"],
            neutral_terms=e333.read(safe(e333.PATHS["e332_result"], root))[
                "reference_qualification"
            ]["E318_diagnostic"]["terms"],
            old_body=read(safe(PATHS["e327_result"], root)),
            root=root,
            E334_candidate_transfer_digest=prior_result["partial"][
                "candidate_native_identity"
            ],
        )
        return data

    def construct(self, data, ledger, persist, on_return):
        def callback():
            old = self.wrapper.SSPRecoveryResourceTrial.from_checkpoint(
                data["graph"], copy.deepcopy(data["saved"])
            )
            try:
                candidate = self.layer.SSPEfficacyTrial(
                    old, data["positions"], data["edge_parameters"], data["theta"]
                )
                candidate._e291_arm = "neutral"
                return candidate
            except BaseException as error:
                self.wrapper.cleanup_owned(old, error)
                raise

        return self.native._construction(
            "efficacy", callback, ledger, persist, on_return
        )

    def initial(self, session, data, output):
        from tools.probe_c_banc_m3_core_rhythm import source_matrix

        observed = session.snapshot()
        patch = observed[self.layer.PATCH_KEY]
        inverse = self.layer._source_representation(observed, patch)
        self.wrapper.require_saved_whole_equal(
            inverse, data["saved"], "E335 exact fresh original whole including wall"
        )
        if (
            session.control_tick != 0
            or session.network.tick != 0
            or abs(session.body.physics_time()) > 2e-8
            or session.adapter.identity != self.e318.ADAPTER_ID
            or session.network.identity == self.e318.NETWORK_ID
            or session.network.identity == data["E334_candidate_transfer_digest"]
            or session.network.device != "cuda"
            or session.network.interval_steps != 20
            or not session.fused_readout
        ):
            raise ValueError("fresh native CUDA/packed/fused20 control0 required")
        actual = source_matrix(session.network)
        host = {
            k: np.array(getattr(actual, k), copy=True)
            for k in ("indptr", "indices", "data")
        }
        for key, value in host.items():
            _exact(value, data["patched"][key], "actual host CSR " + key)
        packed = verify_packed(session.network, host, expected_palette=8040)
        np.savez(output / "patched-csr.npz", **host)
        copy_begin = time.perf_counter()
        buffers = {
            k: _host(getattr(session.network, k))
            for k in ("_rowptr", "_columns", "_weight_values")
        }
        copy_s = time.perf_counter() - copy_begin
        write_begin = time.perf_counter()
        np.savez(output / "actual-packed-csr.npz", **buffers)
        packed["raw_acquisition"] = {
            "extra_host_copy_s": copy_s,
            "uncompressed_write_s": time.perf_counter() - write_begin,
            "buffer_bytes": sum(a.nbytes for a in buffers.values()),
            "artifact_bytes": (output / "actual-packed-csr.npz").stat().st_size,
            "actual_arrays": {k: e334._metadata(v) for k, v in buffers.items()},
        }
        del buffers
        np.savez(
            output / "patched-edge-roster.npz",
            **{
                k: patch[k]
                for k in (
                    "positions",
                    "edge_parameters",
                    "theta64",
                    "original32",
                    "actual32",
                )
            },
        )
        decoder = verify_decoder(
            session.adapter, data["saved"]["adapter"], data["decoder_approved"]
        )
        return {
            "source_whole_exact_including_wall": True,
            "source_network_identity": self.e318.NETWORK_ID,
            "actual_native_model_hash": session.network.identity,
            "E334_candidate_transfer_digest": data["E334_candidate_transfer_digest"],
            "packed": packed,
            "packed_raw_file": "actual-packed-csr.npz",
            "host_CSR_raw_file": "patched-csr.npz",
            "decoder": decoder,
            "decoder_whole_root": "observed-start",
            "patch_identity": patch["patch_identity"],
        }

    def capture(self, session, raw):
        return self.base.capture(session, raw)

    def save_raw(self, folder, raw, data, clock):
        return self.base.save_raw(folder, raw, data, clock)

    def new_clock(self):
        return self.base.e310.runner.native.AdvanceLedger()

    def health(self, session, raw):
        native = self.base.e310.runner.native.native_arrays(raw["native"][-50:])
        if (
            not np.isfinite(native["upright"]).all()
            or np.any(native["upright"] < 0.9)
            or np.any(native["nonfoot_contact"])
            or session.fault
            or session.body.fault
        ):
            raise self.health_stop("E335 measured native health stop")

    def evaluate(self, output, data):
        import torch

        from tools import verify_c_banc_m3_constrained_secant_body as e327
        from tools import verify_c_banc_m3_smooth_efficacy_direction as e333
        from tools.banc_m3_smooth_temporal import SmoothTemporalObjective

        objective = SmoothTemporalObjective(
            data["curve"],
            data["startup"]["target_command"],
            data["startup"]["target_tendon"],
            device="cpu",
            normalization=data["smooth_constants"]["normalization"],
        )
        for name, array in objective.canonical_constants.items():
            _exact(array, data["smooth_constants"][name], "canonical smooth " + name)
            attr = {
                "curves": "curves",
                "normalization": "successor_normalization",
                "temperature": "temperature",
                "future_anchors": "future_anchors",
                "phase_basis": "phase_basis",
            }[name]
            _exact(_host(getattr(objective, attr)), array, "actual CPU smooth " + name)
            if (
                objective.constant_metadata[name]
                != data["smooth_constant_metadata"][name]
            ):
                raise ValueError("smooth constant metadata differs")
        neutral_folder = safe(self.e318.OUTPUT, data["root"])
        values = {}
        for label, folder in (("neutral", neutral_folder), ("candidate", output)):
            raw = e334.load_selected(folder / "control-raw.npz", ("command", "tendon"))
            values[label] = e327.evaluate_saved(folder, data["curve"], data["startup"])
            values[label]["smooth"] = e333.metrics(
                objective.evaluate(
                    torch.as_tensor(raw["command"][1:]),
                    torch.as_tensor(raw["tendon"][1:]),
                )
            )
        expected = data["old_body"]["saved_neutral"]
        require_close(values["neutral"]["shape"], expected["shape"])
        require_close(values["neutral"]["summary"], expected["summary"])
        if not np.allclose(
            values["neutral"]["smooth"]["training_loss"],
            data["neutral_terms"][0],
            rtol=1e-10,
            atol=1e-10,
        ):
            raise ValueError("E334/E332 neutral smooth score differs")
        put_json(
            output / "evaluation.json",
            {
                "schema": "flylab.m3.e335.evaluation.v1",
                **values,
                "smooth_constant_metadata": objective.constant_metadata,
            },
        )
        return values

    def __getattr__(self, name):
        return getattr(self.base, name)


def require_close(actual, expected):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError("neutral evaluation fields differ")
        for key in expected:
            require_close(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError("neutral evaluation lengths differ")
        for a, b in zip(actual, expected, strict=True):
            require_close(a, b)
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if not np.isfinite(actual) or not np.allclose(
            actual, expected, rtol=1e-10, atol=1e-10
        ):
            raise ValueError("neutral scientific evaluation differs")
    elif actual != expected:
        raise ValueError("neutral categorical evaluation differs")


def classify(candidate, neutral, *, engineering_valid):
    smooth, shape, gait = candidate["smooth"], candidate["shape"], candidate["summary"]
    gates = {
        "engineering_valid": engineering_valid is True,
        "startup_four": all(
            smooth[k] <= v
            for k, v in (
                ("startup_q_rms", 0.018),
                ("startup_q_max", 0.04),
                ("startup_tendon_rms", 0.009),
                ("startup_tendon_max", 0.025),
            )
        ),
        "smooth_better_than_saved_neutral": smooth["training_loss"]
        <= neutral["smooth"]["training_loss"] - 1e-9,
        "legacy_shape_better_than_saved_neutral": shape["shape_loss"]
        < neutral["shape"]["shape_loss"],
        "forward": gait["forward_mm"] >= 1.6790085779712631,
        "slip": gait["slip_mm_s"] <= 1.8026436968249955,
        "bout": gait["longest_continuous_forward_bout_s"] >= 0.3,
        "all_six_eligible_cycles": all(
            gait["complete_cycles"].get(leg, 0) > 0
            for leg in ("lf", "lm", "lh", "rf", "rm", "rh")
        ),
        "native_health": gait["health"].get("healthy") is True,
    }
    gates = {k: bool(v) for k, v in gates.items()}
    return gates, all(gates.values())


def new_ledger():
    return {
        "schema": "flylab.m3.e335.advance-ledger.v1",
        "limits": LIMITS.copy(),
        "construction": [],
        "body_construction": [],
        "construction_requested": 0,
        "construction_completed": 0,
        "body_constructors": 0,
        "controls_requested": 0,
        "controls_host_returned": 0,
        "controls_completed": 0,
        "neural_requested": 0,
        "neural_host_returned": 0,
        "neural_device_confirmed": 0,
        "physics_steps_host_returned": 0,
        "physics_steps_completed": 0,
        "readout_returns": [],
        "retry_count": 0,
        "optimizer_steps": 0,
        "errors": [],
        "phases": [],
    }


def lifecycle(
    data,
    dependencies,
    output,
    ledger,
    raw,
    *,
    deadline=lambda: None,
    persist=lambda: None,
    fail=None,
    clock=time.perf_counter,
    nodes=N,
    writer=put_json,
    saver=np.savez,
):
    """Own exactly one constructed session; cleanup/raw preservation precede eval.

    Public generated fixture hook. Original run() fixes nodes=N and uses only
    native dependencies. It does not restore or retry a body after failure.
    """
    deps = dependencies
    holder = {}
    session = observer = detach = journal = native_clock = None
    failures = []
    complete = False
    origin = None

    def failed(stage, error):
        failures.append(
            {
                "stage": stage,
                "type": type(error).__name__,
                "message": str(error)[:500],
                "notes": list(getattr(error, "__notes__", [])),
            }
        )
        if fail is not None:
            fail(stage, error)

    def phase(label, callback):
        began = clock()
        try:
            return callback()
        finally:
            ledger["phases"].append({"phase": label, "elapsed_s": clock() - began})

    try:
        deadline()
        ledger["construction_requested"] = 1
        persist()
        session = phase(
            "construct",
            lambda: deps.construct(
                data, ledger, persist, lambda value: holder.setdefault("session", value)
            ),
        )
        ledger.update(construction_completed=1, body_constructors=1)
        verify_construction(ledger)
        deadline()
        origin = phase("initial_proof", lambda: deps.initial(session, data, output))
        phase(
            "observed_start",
            lambda: deps.save_state(output / "observed-start", session),
        )
        if hasattr(deps, "trace_metadata"):
            raw["trace_metadata"] = deps.trace_metadata(session)
        observer = LiveInputObserver(
            session,
            output,
            ledger,
            nodes=nodes,
            saver=saver,
            writer=writer,
            clock=clock,
        )
        observer.install()  # Physics return counter is now inside contact wrapper.
        detach = deps.install(session, raw["native"])
        native_clock = deps.new_clock()
        journal = deps.journal(output / "control-journal.jsonl")
        deps.capture(session, raw)
        began = clock()
        for control in range(1, 301):
            deadline()
            ledger["controls_requested"] = control
            ledger["neural_requested"] = control * 20
            observer.begin(control)
            event = None
            before_native = len(raw["native"])
            before_returned = ledger["physics_steps_host_returned"]
            try:
                event = deps.advance(session, "neutral", control, native_clock, journal)
            except BaseException as error:
                try:
                    observer.end(error=error)
                except Exception as secondary:  # noqa: BLE001
                    error.add_note("observation end failure: " + repr(secondary))
                raise
            observer.end(returned=True)
            # A failed capture/journal cannot subtract synchronous work receipts.
            try:
                deps.capture(session, raw)
                deps.readout_receipt(journal, "neutral", control, event, True)
            except Exception as error:
                try:
                    deps.readout_receipt(
                        journal, "neutral", control, event, False, error=error
                    )
                except Exception as secondary:  # noqa: BLE001
                    error.add_note(
                        "original failed-readout journal failed: " + repr(secondary)
                    )
                raise
            if (
                session.control_tick != control
                or session.network.tick != control * 20
                or len(raw["native"]) != before_native + 50
                or ledger["physics_steps_host_returned"] != before_returned + 50
                or ledger["neural_device_confirmed"] != control * 20
            ):
                raise ValueError("separate returned/readout/native50 interval mismatch")
            deps.health(session, raw)
            ledger.update(
                controls_completed=control, physics_steps_completed=len(raw["native"])
            )
        ledger["phases"].append(
            {"phase": "live_controls_chunks", "elapsed_s": clock() - began}
        )
        deadline()
        if (
            not native_clock.clock_valid
            or native_clock.neural_s is None
            or native_clock.physical_s is None
            or abs(native_clock.neural_s - 1.5) > 2e-8
            or abs(native_clock.physical_s - 1.5) > 2e-8
            or ledger["physics_steps_host_returned"] != 15000
            or len(raw["rows"]) != 301
        ):
            raise ValueError(
                "complete synchronized native clocks/raw301/returned15000 required"
            )
        phase("whole_final", lambda: deps.save_state(output / "whole-300", session))
        observer.publish()
        deadline()
        complete = True
    except Exception as error:  # noqa: BLE001 - measured work survives faults
        failed("science", error)
    finally:
        if session is None and "session" in holder:
            session = holder["session"]
            ledger["owned_session_recovered"] = True
            ledger["body_constructors"] = 1
        if native_clock is not None:
            ledger.update(
                clock_events=copy.deepcopy(native_clock.events),
                clock_valid=native_clock.clock_valid,
                actual_neural_s=native_clock.neural_s,
                actual_physical_s=native_clock.physical_s,
            )
        ledger["native_samples_observed"] = len(raw["native"])
        if observer is not None:
            try:
                if observer.current is not None:
                    observer.end()
                observer.publish()
                ledger["live_input_io"] = copy.deepcopy(observer.io)
            except Exception as error:  # noqa: BLE001
                failed("live_partial_publication", error)
        # Original native contact observer first, then counter/other wrappers.
        for label, callback in (
            ("native_detach", detach),
            ("instance_detach", observer.detach if observer else None),
            ("journal_close", journal.close if journal else None),
        ):
            if callback is not None:
                try:
                    phase(label, callback)
                except Exception as error:  # noqa: BLE001
                    failed(label, error)
        if session is not None and failures:
            try:
                phase(
                    "partial_whole",
                    lambda: deps.save_state(output / "partial-final", session),
                )
            except Exception as error:  # noqa: BLE001
                failed("partial_whole", error)
                try:
                    deps.save_partial(output / "partial-components", session)
                except Exception as secondary:  # noqa: BLE001
                    failed("partial_components", secondary)
        if session is not None:
            ledger["close_requested"] = True
            try:
                phase("owned_close", session.close)
                ledger["close_returned"] = True
            except Exception as error:  # noqa: BLE001
                failed("owned_close", error)
        if native_clock is not None:
            try:
                for error in (
                    phase(
                        "raw_save",
                        lambda: deps.save_raw(output, raw, data, native_clock),
                    )
                    or []
                ):
                    failed("raw_" + error["artifact"], RuntimeError(error["error"]))
            except Exception as error:  # noqa: BLE001
                failed("raw_save", error)
        else:
            ledger["raw_save_unavailable"] = "native clock never constructed"
        try:
            deadline()
        except Exception as error:  # noqa: BLE001
            failed("science_postcleanup_deadline", error)
    return {
        "complete": complete and not failures,
        "origin": origin,
        "failures": failures,
        "first_error": failures[0] if failures else None,
        "body_owned": session is not None,
    }


def recursive_bytes(output):
    return sum(p.stat().st_size for p in Path(output).rglob("*") if p.is_file())


def file_manifest(output):
    excluded = {
        "result.json",
        "advance-ledger.json",
        "output-manifest.json",
        "emergency-result.json",
    }
    return {
        p.relative_to(output).as_posix(): {
            "sha256": sha256(p),
            "bytes": p.stat().st_size,
        }
        for p in sorted(Path(output).rglob("*"))
        if p.is_file() and p.relative_to(output).as_posix() not in excluded
    }


def usage():
    # Initialize Torch before any possible CuPy import (verified E321 order).
    import torch

    torch.set_num_threads(1)
    free, total = torch.cuda.mem_get_info()
    memory = psutil.Process().memory_info()
    return {
        "host_rss_bytes": memory.rss,
        "host_peak_bytes": getattr(memory, "peak_wset", memory.rss),
        "device_free_bytes": free,
        "device_total_bytes": total,
        "torch_reserved_bytes": torch.cuda.memory_reserved(),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


class GPUMemoryWatch:
    """Owned, read-only device0 SMI sampler started before any CUDA context."""

    def __init__(self, *, process_factory=subprocess.Popen):
        self.process_factory = process_factory
        self.process = None
        self.thread = None
        self.stderr_thread = None
        self.samples = []
        self.errors = []
        self.stderr = []
        self.ready = threading.Event()
        self.lock = threading.Lock()
        self.started = False
        self.closed = False
        self.terminated_owned_process = False

    def _error(self, error):
        with self.lock:
            self.errors.append(str(error)[:500])

    def start(self):
        if self.started or self.closed:
            raise RuntimeError("GPU sampler start is single-use")
        self.started = True
        try:
            self.process = self.process_factory(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "--id=0",
                    "--loop-ms=10",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )

            def consume():
                try:
                    for line in self.process.stdout:
                        sample = int(line.strip()) * (1 << 20)
                        if sample < 0:
                            raise ValueError("negative GPU memory sample")
                        with self.lock:
                            self.samples.append(sample)
                        self.ready.set()
                except Exception as error:  # noqa: BLE001
                    self._error("GPU sampler stdout: " + repr(error))
                    self.ready.set()

            def consume_stderr():
                try:
                    for line in self.process.stderr:
                        with self.lock:
                            self.stderr.append(line.strip()[:500])
                except Exception as error:  # noqa: BLE001
                    self._error("GPU sampler stderr: " + repr(error))

            self.thread = threading.Thread(target=consume, daemon=True)
            self.stderr_thread = threading.Thread(target=consume_stderr, daemon=True)
            self.thread.start()
            self.stderr_thread.start()
            if not self.ready.wait(5):
                raise RuntimeError("SMI baseline unavailable before CUDA")
            proof = self.snapshot()
            if not proof["sample_count"] or proof["errors"]:
                raise RuntimeError("pre-context GPU sampler baseline failed")
        except Exception as error:
            self._error("GPU sampler start: " + repr(error))
            raise
        return self

    def snapshot(self):
        with self.lock:
            samples, errors, stderr = (
                list(self.samples),
                list(self.errors),
                list(self.stderr),
            )
        ended = self.process is None or self.process.poll() is not None
        if self.started and self.process is not None and ended and not self.closed:
            errors.append("GPU sampler exited before owned close")
        errors.extend("GPU sampler stderr: " + line for line in stderr if line)
        baseline = samples[0] if samples else None
        peak = max(samples) if samples else None
        return {
            "schema": "flylab.m3.e335.gpu-memory-watch.v1",
            "measurement": "device-wide SMI samples from before CUDA through initial terminal readback/resource checks; final cleanup-evidence publication follows owned close; context-inclusive positive change plus1MiB quantization; unsampled peaks and external GPU work cannot be excluded",
            "device_index": 0,
            "pre_context_baseline_bytes": baseline,
            "sampled_peak_global_bytes": peak,
            "additional_sampled_including_context_bytes": None
            if not samples
            else max(0, peak - baseline) + (1 << 20),
            "quantization_allowance_bytes": 1 << 20,
            "sample_count": len(samples),
            "started": self.started,
            "closed": self.closed,
            "owned_pid": None if self.process is None else self.process.pid,
            "process_ended": ended,
            "stdout_thread_ended": self.thread is None or not self.thread.is_alive(),
            "stderr_thread_ended": self.stderr_thread is None
            or not self.stderr_thread.is_alive(),
            "terminated_owned_process": self.terminated_owned_process,
            "errors": errors,
        }

    def close(self):
        if self.closed:
            return self.snapshot()
        self.closed = True
        if self.process is not None:
            try:
                if self.process.poll() is None:
                    self.terminated_owned_process = True
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=3)
                else:
                    self._error("GPU sampler exited before owned close")
            except Exception as error:  # noqa: BLE001
                self._error("GPU sampler owned process close: " + repr(error))
        for thread in (self.thread, self.stderr_thread):
            if thread is not None:
                try:
                    thread.join(timeout=1)
                    if thread.is_alive():
                        self._error("GPU sampler owned thread not ended")
                except Exception as error:  # noqa: BLE001
                    self._error("GPU sampler owned thread close: " + repr(error))
        if self.process is not None and self.process.poll() is not None:
            for stream in (self.process.stdout, self.process.stderr):
                try:
                    stream.close()
                except Exception as error:  # noqa: BLE001
                    self._error("GPU sampler owned pipe close: " + repr(error))
        proof = self.snapshot()
        if proof["errors"] or not all(
            proof[key]
            for key in ("process_ended", "stdout_thread_ended", "stderr_thread_ended")
        ):
            raise RuntimeError("GPU sampler cleanup/measurement failed")
        return proof


def run(
    freeze_path,
    freeze_sha256,
    review_path,
    review_sha256,
    output,
    *,
    root=ROOT,
    dependencies=None,
    gatekeeper=gate,
    test_dimensions=None,
    clock=time.perf_counter,
    writer=put_json,
    emergency_writer=put_json,
    saver=np.savez,
    meter=None,
    watcher_factory=None,
):
    """Absolute gate/science/final clocks; generated injection is explicit only."""
    began = clock()
    science_deadline = began + 60
    final_deadline = None
    if test_dimensions is None:
        if (
            dependencies is not None
            or gatekeeper is not gate
            or watcher_factory is not None
            or meter is not None
        ):
            raise ValueError("injected body requires explicit TestDimensions")
        if any(
            os.environ.get(k) != "1"
            for k in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        ):
            raise ValueError("original CPUthread1 supervisor environment required")
        nodes = N
    else:
        if not isinstance(test_dimensions, TestDimensions) or dependencies is None:
            raise ValueError("generated fixture/dimensions required")
        nodes = test_dimensions.nodes
    output = Path(output)
    ledger = new_ledger()
    errors = []
    first_error = None
    baseline = None
    watcher = None
    result = {
        "schema": RESULT_SCHEMA,
        "run_valid": False,
        "body_screen_supported": False,
        "effective_body_qualification": False,
        "handoff_eligible": False,
        "classification": "not_tested",
        "candidate_adopted": False,
        "biological_gait_proven": False,
        "F1_F8_complete": False,
        "F6_exposed": False,
        "M2_modified": False,
        "original_calls": int(test_dimensions is None),
        "body_calls": 0,
        "generated_explicit_fixture": test_dimensions is not None,
        "external_completion_required": True,
        "root_terminal_raw_audit_required": True,
        "first_error": None,
        "errors": [],
        "provenance": {"freeze_sha256": freeze_sha256, "review_sha256": review_sha256},
    }

    def failed(stage, error):
        nonlocal first_error
        row = {
            "stage": stage,
            "type": type(error).__name__,
            "message": str(error)[:500],
            "notes": list(getattr(error, "__notes__", [])),
        }
        errors.append(row)
        if first_error is None:
            first_error = row
        result.update(
            run_valid=False,
            body_screen_supported=False,
            effective_body_qualification=False,
            handoff_eligible=False,
            first_error=first_error,
            errors=copy.deepcopy(errors),
            classification="invalid",
        )

    def resource_check():
        nonlocal baseline
        if recursive_bytes(output) > LIMITS["output_bytes"]:
            raise OSError("recursive output including failure/pending exceeds1GiB")
        sampled = 0
        if watcher is not None:
            proof = watcher.snapshot()
            ledger["gpu_sampler"] = copy.deepcopy(proof)
            result["gpu_sampler"] = copy.deepcopy(proof)
            if proof["errors"] or not proof["started"] or not proof["sample_count"]:
                raise RuntimeError("GPU sampler measurement unavailable or failed")
            sampled = proof["additional_sampled_including_context_bytes"]
            if not isinstance(sampled, int) or sampled < 0:
                raise ValueError("GPU sampler additional bytes invalid")
            ledger["additional_sampled_gpu_peak_bytes"] = max(
                sampled, ledger.get("additional_sampled_gpu_peak_bytes", 0)
            )
        value = meter() if meter is not None else usage()
        if baseline is None:
            baseline = value
            ledger["resource_baseline"] = value
        ledger["resource_latest"] = value
        if value["host_peak_bytes"] > LIMITS["host_bytes"]:
            raise MemoryError("owned host6GiB cap")
        if value.get("device_total_bytes") != baseline.get("device_total_bytes"):
            raise RuntimeError("GPU device identity changed")
        additional = max(
            0, baseline.get("device_free_bytes", 0) - value.get("device_free_bytes", 0)
        )
        peak = max(
            0,
            value.get("torch_peak_reserved_bytes", 0)
            - baseline.get("torch_reserved_bytes", 0),
        )
        ledger["additional_gpu_peak_bytes"] = max(
            additional, ledger.get("additional_gpu_peak_bytes", 0)
        )
        ledger["additional_torch_peak_reserved_bytes"] = max(
            peak, ledger.get("additional_torch_peak_reserved_bytes", 0)
        )
        if max(additional, peak, sampled) > LIMITS["additional_gpu_bytes"]:
            raise MemoryError("additional GPU4GiB cap")

    def timely():
        if clock() >= (science_deadline if final_deadline is None else final_deadline):
            raise TimeoutError("absolute E335 deadline exceeded")

    def science_check():
        timely()
        resource_check()

    # Passive original gate is before ANY CUDA meter or payload/model loader.
    freeze = gatekeeper(
        freeze_path,
        freeze_sha256,
        review_path,
        review_sha256,
        output,
        root=root,
        timely=timely,
    )
    timely()
    output.mkdir(parents=True, exist_ok=False)

    def persist():
        ledger["elapsed_s"] = clock() - began
        pending = output / "advance-ledger.pending.json"
        writer(pending, ledger)
        pending.replace(output / "advance-ledger.json")

    raw = {k: [] for k in ("rows", "controls", "recruitment", "activity", "native")}
    science = None
    data = None
    try:
        if shutil.disk_usage(output).free < LIMITS["minimum_free_disk_bytes"]:
            raise OSError("fresh output free disk4GiB required")
        if test_dimensions is None or watcher_factory is not None:
            watcher = GPUMemoryWatch() if watcher_factory is None else watcher_factory()
            watcher.start()  # Gate complete, before usage imports Torch/CUDA.
            ledger["gpu_sampler"] = copy.deepcopy(watcher.snapshot())
            result["gpu_sampler"] = copy.deepcopy(ledger["gpu_sampler"])
            timely()
        else:
            ledger["gpu_sampler"] = {
                "generated_cpu_only": True,
                "original_gpu_measured": False,
            }
        resource_check()  # Torch CUDA first; immutable loader may next import CuPy.
        writer(
            output / "source-manifest.json",
            {
                "schema": "flylab.m3.e335.source-manifest.v1",
                **result["provenance"],
                "source_sha256": freeze["source_sha256"],
                "input_sha256": freeze["input_sha256"],
            },
        )
        persist()
        deps = Dependencies() if dependencies is None else dependencies
        started = clock()
        data = deps.load(freeze, root=root)
        ledger["phases"].append(
            {"phase": "payload_load_gate", "elapsed_s": clock() - started}
        )
        raw["ids"] = data["ids"]
        science_check()
        science = lifecycle(
            data,
            deps,
            output,
            ledger,
            raw,
            deadline=science_check,
            persist=persist,
            fail=failed,
            clock=clock,
            nodes=nodes,
            writer=writer,
            saver=saver,
        )
        result["science"] = science
        result["science_complete"] = science["complete"]
        result["body_calls"] = ledger["body_constructors"]
        if science["complete"] and first_error is None:
            started = clock()
            evaluations = deps.evaluate(output, data)
            ledger["phases"].append(
                {"phase": "saved_evaluation", "elapsed_s": clock() - started}
            )
            result["evaluation"] = evaluations
            gates, supported = classify(
                evaluations["candidate"], evaluations["neutral"], engineering_valid=True
            )
            result.update(
                gates=gates,
                run_valid=True,
                body_screen_supported=supported,
                effective_body_qualification=supported,
                handoff_eligible=supported,
                classification="supported_short_body_screen"
                if supported
                else "rejected",
            )
        science_check()
        started = clock()
        pin_map(freeze["source_sha256"], root, timely)
        pin_map(freeze["input_sha256"], root, timely)
        ledger["phases"].append({"phase": "posthash", "elapsed_s": clock() - started})
        science_check()
    except Exception as error:  # noqa: BLE001
        failed("run_science", error)
    # No clock reset: both original absolute limits remain binding.
    final_begin = clock()
    final_deadline = min(began + 70, final_begin + 10)
    ledger["science_wall_s"] = final_begin - began
    ledger["absolute_clock"] = {
        "began_s": began,
        "science_deadline_s": science_deadline,
        "final_begin_s": final_begin,
        "final_deadline_s": final_deadline,
        "internal_deadline_s": began + 70,
    }

    def terminal_payloads():
        result.update(
            body_calls=ledger["body_constructors"],
            first_error=first_error,
            errors=copy.deepcopy(errors),
        )
        result["cost"] = {
            k: ledger[k]
            for k in (
                "body_constructors",
                "controls_requested",
                "controls_host_returned",
                "controls_completed",
                "neural_requested",
                "neural_host_returned",
                "neural_device_confirmed",
                "physics_steps_host_returned",
                "physics_steps_completed",
            )
        }
        ledger.update(
            {
                k: copy.deepcopy(result[k])
                for k in (
                    "run_valid",
                    "body_screen_supported",
                    "effective_body_qualification",
                    "handoff_eligible",
                    "classification",
                    "first_error",
                    "errors",
                )
            }
        )
        ledger["elapsed_s"] = clock() - began
        return {
            "schema": "flylab.m3.e335.output-manifest.v1",
            "files": file_manifest(output),
            "run_valid": result["run_valid"],
            "effective_body_qualification": result["effective_body_qualification"],
            "handoff_eligible": result["handoff_eligible"],
            "external_completion_required": True,
        }

    def publish(sink):
        manifest = terminal_payloads()
        sink(
            output / "receipts.json",
            {
                "schema": "flylab.m3.e335.receipts.v1",
                "readout_returns": ledger["readout_returns"],
                "neural_requested": ledger["neural_requested"],
                "neural_host_returned": ledger["neural_host_returned"],
                "neural_device_confirmed": ledger["neural_device_confirmed"],
                "physics_steps_host_returned": ledger["physics_steps_host_returned"],
                "native_samples_observed": ledger.get("native_samples_observed", 0),
                "clock_valid": ledger.get("clock_valid", False),
            },
        )
        manifest = terminal_payloads()  # Include immutable receipt after publication.
        sink(output / "advance-ledger.json", ledger)
        sink(output / "result.json", result)
        sink(output / "output-manifest.json", manifest)
        if (
            read(output / "result.json") != result
            or read(output / "advance-ledger.json") != ledger
            or read(output / "output-manifest.json") != manifest
        ):
            raise ValueError("terminal readback/disk-return disagreement")
        if manifest["files"] != file_manifest(output):
            raise ValueError("raw manifest changed during finalization")
        timely()
        resource_check()  # Include post-publication work and all files.

    def complete_publication():
        try:
            publish(writer)
        except Exception as error:  # noqa: BLE001
            failed("terminal_publication", error)
            try:
                publish(emergency_writer)
            except Exception as secondary:  # noqa: BLE001
                failed("terminal_repair", secondary)
                # Repair again after recording its failure; never renew deadlines.
                try:
                    publish(emergency_writer)
                except Exception as last:  # noqa: BLE001
                    failed("emergency_publication", last)
                    result["disk_agreement_established"] = False
                    try:
                        terminal_payloads()
                        emergency_writer(output / "result.json", result)
                        emergency_writer(output / "advance-ledger.json", ledger)
                        emergency_writer(output / "emergency-result.json", result)
                        timely()
                        resource_check()
                    except Exception as final_error:  # noqa: BLE001
                        result["emergency_sink_error"] = repr(final_error)[:500]
                        # Both absolute clocks stay binding after repair writes.
                        try:
                            terminal_payloads()
                            emergency_writer(output / "result.json", result)
                            emergency_writer(output / "advance-ledger.json", ledger)
                        except Exception:  # noqa: BLE001,S110 - sink failures already explicit
                            pass

    try:
        complete_publication()  # Sampler includes publication/readback/resource work.
    finally:
        if watcher is not None:
            try:
                watcher.close()  # Only the process/threads created by this instance.
            except Exception as error:  # noqa: BLE001
                failed("gpu_sampler_close", error)
            try:
                proof = watcher.snapshot()
                ledger["gpu_sampler"] = copy.deepcopy(proof)
                result["gpu_sampler"] = copy.deepcopy(proof)
                if proof["errors"] or not all(
                    proof[key]
                    for key in (
                        "closed",
                        "process_ended",
                        "stdout_thread_ended",
                        "stderr_thread_ended",
                    )
                ):
                    raise RuntimeError("GPU sampler final evidence invalid")
                resource_check()  # Include samples arriving while closing owned SMI.
                timely()
            except Exception as error:  # noqa: BLE001
                failed("gpu_sampler_final", error)
            complete_publication()  # Freeze ended proof; recheck recursive output/time.
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = run(
        args.freeze, args.freeze_sha256, args.review, args.review_sha256, args.output
    )
    print(json.dumps(value, ensure_ascii=True, allow_nan=False))
    return 0 if value["run_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
