"""E342 fixed four-window C-only native efficacy direction.

Import is passive. Original execution requires current finite producer closure
and a ROOT review. Generated axes are explicit and never exposed by the CLI.
The native rounded forward has a real-equation surrogate VJP; this module does
not claim an independently proved parameter Jacobian or biological gait.
"""

from __future__ import annotations

import time

_IMPORT_S0 = time.perf_counter()

import argparse
import ast
import copy
import hashlib
import io
import json
import math
import os
import re
import struct
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

for _key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_key] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = "verification/banc-natural-gait-m3-20260919"
OWNED = (
    "tools/verify_c_banc_m3_coordination_parameter_direction.py",
    "tests/test_c_banc_coordination_parameter_direction.py",
)
OUTPUT = BASE + "/e342-coordination-parameter-direction"
FREEZE = BASE + "/e342-coordination-parameter-direction-freeze.json"
REVIEW = BASE + "/parent-e342-original-source-review.json"
DESIGN = "docs/BANC_M3_E342_COORDINATION_PARAMETER_DIRECTION_DESIGN_20261001.md"
DESIGN_SHA = "99670ddc3bcd1f2542ea7be0ad1ae7b35ec6d18879425e2067a1b37fa6bb956f"
INVENTORY = BASE + "/parent-e342-selected-input-inventory.json"
INVENTORY_SHA = "d84599704814fa1ab24e576e812cb312bbc33234a90c585d431e9beb567d6eaa"
DIRECT = BASE + "/parent-e342-reflected-direct-dependencies.json"
DIRECT_SHA = "281fba9bdd895d399591bb97cbce5a951072604343c12640b8b63d8181c9416c"
REFLECTION = BASE + "/parent-e342-astra-reflection.json"
SUPERVISOR = BASE + "/parent-e267-supervise.py"
SUPERVISOR_SHA = "1d8682505f8060d3d8bcb50508811b9c4da531bdbf07c96200065bcc3324df8d"
WINDOWS = (100, 164, 228, 292)
SLICES = ((101, 0, 4), (151, 14, 18), (226, 3, 7), (276, 17, 21))
REFERENCE = ("curves", "normalization", "temperature", "future_anchors", "phase_basis")
LOSS_ORDER = ("S", "C", "total", "LR")
KIB = 1024
MIB = KIB * KIB
GIB = MIB * KIB
QUOTAS = {
    "raw.npz": 16 * MIB,
    "selected-ledger.json": 512 * KIB,
    "result.json": 256 * KIB,
    "receipts.json": 256 * KIB,
    "manifest.json": 16 * KIB,
    "stdout": 256 * KIB,
    "audit-report.json": 256 * KIB,
    "audit-stdout": 256 * KIB,
    "external-completion.json": 64 * KIB,
    "audit-external-completion.json": 64 * KIB,
    "stderr": 32 * KIB,
    "audit-stderr": 32 * KIB,
    "supervision.json": 16 * KIB,
    "audit-supervision.json": 16 * KIB,
    "pids.json": 8 * KIB,
    "audit-pids.json": 8 * KIB,
    "final-decision.json": 64 * KIB,
}
FAILURE_QUOTA = 4 * MIB
OUTPUT_CAP = 32 * MIB
FIXED = {
    "windows": list(WINDOWS),
    "controls": 4,
    "nodes": 158706,
    "edges": 11584852,
    "selected": 725849,
    "groups": 8547,
    "population": 7136,
    "steps_per_control": 20,
    "neural_dt": 0.00025,
    "control_dt": 0.005,
    "step": 0.001,
    "signal_floor": 1e-9,
    "relative_limit": 0.05,
    "science_s": 48,
    "internal_s": 60,
    "final_s": 5,
    "supervisor_s": 65,
    "host_cap": 6 * GIB,
    "additional_gpu_cap": 8 * GIB,
    "raw_cap": 16 * MIB,
    "output_cap": OUTPUT_CAP,
    "parameter_Jacobian_independently_proven": False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe(root, name):
    require(isinstance(name, str) and name and "\x00" not in name, "path string")
    text = name.replace("\\", "/")
    path = Path(text)
    target = (path if path.is_absolute() else Path(root) / path).resolve()
    target.relative_to(Path(root).resolve())
    require(not any(part == ".." for part in path.parts), "traversal")
    return target


def relative(root, name):
    return safe(root, str(name)).relative_to(Path(root).resolve()).as_posix()


def check_sha(value):
    require(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value), "SHA256")
    return value


def digest(path, timely=lambda: None):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(MIB):
            timely()
            checksum.update(block)
    timely()
    return checksum.hexdigest()


def read_json(path, limit=8 * MIB):
    require(0 < Path(path).stat().st_size <= limit, "JSON byte limit")

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def bad(value):
        raise ValueError("nonfinite JSON: " + value)

    return json.loads(
        Path(path).read_text("utf-8"), object_pairs_hook=pairs, parse_constant=bad
    )


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def merge(destination, additions, root):
    for name, pin in additions.items():
        name = relative(root, name)
        check_sha(pin)
        require(
            name not in destination or destination[name] == pin,
            "pin collision: " + name,
        )
        destination[name] = pin


def source_map(root=ROOT):
    return {name: digest(safe(root, name)) for name in OWNED}


@dataclass(frozen=True)
class TestDimensions:
    nodes: int = 400
    edges: int = 8
    selected: int = 8
    groups: int = 4
    population: int = 4
    steps_per_control: int = 1

    def __post_init__(self):
        require(
            all(type(v) is int and v > 0 for v in vars(self).values()),
            "test integer axes",
        )
        require(
            391 <= self.nodes <= 4096
            and self.edges <= 65536
            and self.groups <= self.selected <= self.edges
            and self.population <= self.nodes
            and self.steps_per_control in (1, 20),
            "small explicit generated axes",
        )


def axes(dimensions=None):
    require(
        dimensions is None or isinstance(dimensions, TestDimensions),
        "explicit test dimensions",
    )
    return (
        FIXED
        if dimensions is None
        else {
            **FIXED,
            **vars(dimensions),
            "control_dt": dimensions.steps_per_control * 0.00025,
        }
    )


def generated_path(root, name):
    path = safe(root, name)
    suffix = path.relative_to(safe(root, BASE))
    require(
        suffix.parts and suffix.parts[0].startswith(("e342-sol-", "parent-e342-")),
        "fresh E342 generated namespace",
    )
    require(path != safe(ROOT, OUTPUT), "original output forbidden")
    return path


def positive_time(value):
    return type(value) in (int, float) and 0 < value <= 90 and math.isfinite(value)


def positive_pids(value):
    return (
        isinstance(value, list)
        and bool(value)
        and all(type(v) is int and v > 0 for v in value)
        and len(set(value)) == len(value)
    )


def descriptor(root, value, evidence, timely=lambda: None):
    require(
        isinstance(value, dict) and set(value) == {"path", "sha256"}, "exact descriptor"
    )
    name = relative(root, value["path"])
    require(evidence.get(name) == check_sha(value["sha256"]), "descriptor evidence")
    path = safe(root, name)
    require(digest(path, timely) == value["sha256"], "descriptor bytes")
    return path


def closed_execution(
    root, branch, evidence, current, *, expected_exit=0, timely=lambda: None
):
    require(
        set(branch) == {"contract", "supervision", "report", "pids"}, "closed branch"
    )
    loaded = {
        k: read_json(descriptor(root, v, evidence, timely)) for k, v in branch.items()
    }
    contract, supervision, report, pids = (
        loaded[k] for k in ("contract", "supervision", "report", "pids")
    )
    require(
        contract.get("original_calls") == 0
        and type(contract["original_calls"]) is int
        and supervision.get("original_calls") == 0
        and type(supervision["original_calls"]) is int,
        "generated original counter",
    )
    require(
        contract.get("source_sha256", {}).get(SUPERVISOR) == SUPERVISOR_SHA,
        "supervisor source",
    )
    for name, pin in contract.get("source_sha256", {}).items():
        require(
            evidence.get(relative(root, name)) == check_sha(pin),
            "executed source evidence",
        )
        require(digest(safe(root, name), timely) == pin, "executed source bytes")
    require(
        all(contract["source_sha256"].get(k) == v for k, v in current.items()),
        "current executed source",
    )
    commands, attempts = contract.get("commands"), supervision.get("attempts")
    require(
        isinstance(commands, list)
        and len(commands) == 1
        and isinstance(attempts, list)
        and len(attempts) == 1,
        "sole completed command",
    )
    command, attempt = commands[0], attempts[0]
    argv = command.get("argv")
    require(
        isinstance(argv, list)
        and all(isinstance(v, str) for v in argv)
        and len(argv) >= 7
        and argv[:4] == [".venv/Scripts/python.exe", "-X", "utf8", "-B"]
        and attempt.get("argv") == argv,
        "exact Python supervised argv",
    )
    helper = relative(root, argv[4])
    require(
        helper == relative(root, report.get("audit_source_path", ""))
        and contract["source_sha256"].get(helper) == report.get("audit_source_sha256"),
        "actual report helper",
    )
    require(
        argv.count("--output") == 1
        and argv.index("--output") + 1 < len(argv)
        and relative(root, argv[argv.index("--output") + 1])
        == relative(root, branch["report"]["path"]),
        "actual report output",
    )
    require(
        supervision.get("contract_sha256") == branch["contract"]["sha256"]
        and relative(root, contract.get("supervisor_output", ""))
        == relative(root, branch["supervision"]["path"]),
        "supervisor contract linkage",
    )
    require(
        attempt.get("requested") is True
        and attempt.get("killed") is False
        and type(attempt.get("exit_code")) is int
        and attempt["exit_code"] == expected_exit
        and supervision.get("success") is (expected_exit == 0),
        "completed attempt",
    )
    require(
        positive_time(command.get("wall_s"))
        and positive_time(attempt.get("wall_s"))
        and positive_time(supervision.get("elapsed_s"))
        and attempt["wall_s"] <= supervision["elapsed_s"] <= command["wall_s"],
        "actual wall bound",
    )
    cap, peak = (
        contract.get("max_tree_rss_bytes"),
        supervision.get("max_tree_rss_bytes"),
    )
    require(
        type(cap) is int
        and 0 < cap <= 2 * GIB
        and type(peak) is int
        and 0 < peak <= cap,
        "actual RSS bound",
    )
    ids = supervision.get("owned_pids")
    require(
        positive_pids(ids)
        and positive_pids(attempt.get("pids"))
        and positive_pids(pids.get("owned_pids"))
        and ids == attempt["pids"] == pids["owned_pids"],
        "exact owned PID list",
    )
    presence = pids.get("present_after_return")
    require(
        pids.get("all_owned_pids_ended") is True
        and isinstance(presence, dict)
        and set(presence) == {str(v) for v in ids}
        and all(v is False for v in presence.values()),
        "exact absent owned PID map",
    )
    for name in ("stdout", "stderr"):
        path = relative(root, command.get(name, ""))
        require(
            path in evidence and digest(safe(root, path), timely) == evidence[path],
            "actual log pin",
        )
    require(
        report.get("source_sha256") == current
        and type(report.get("original_calls")) is int
        and report["original_calls"] == 0,
        "current report source/counter",
    )
    checks = report.get("checks")
    require(
        isinstance(checks, dict)
        and checks
        and all(type(v) is bool for v in checks.values()),
        "report checks",
    )
    if expected_exit == 0:
        require(
            report.get("passed") is True
            and report.get("first_error") is None
            and all(checks.values()),
            "successful generated report",
        )
    else:
        require(
            report.get("passed") is False
            and isinstance(report.get("first_error"), dict)
            and not all(checks.values()),
            "numerical rejection report",
        )
    return report


def validate_proof(root, proof, current, timely=lambda: None):
    require(
        proof.get("schema") == "flylab.m3.e342.current-generated-proof.v1"
        and proof.get("passed") is True
        and proof.get("source_sha256") == current
        and type(proof.get("original_calls")) is int
        and proof["original_calls"] == 0,
        "current generated proof",
    )
    evidence = proof.get("evidence_sha256")
    require(
        isinstance(evidence, dict) and 0 < len(evidence) <= 256, "finite proof evidence"
    )
    normalized = {}
    merge(normalized, evidence, root)
    require(len(normalized) == len(evidence), "proof path aliases")
    for name, pin in normalized.items():
        require(
            safe(root, name).stat().st_size <= 8 * MIB
            and digest(safe(root, name), timely) == pin,
            "proof bytes",
        )
    require(
        set(proof.get("branches", {})) == {"sol", "root", "native_cuda"},
        "three qualification branches",
    )
    reports = {
        k: closed_execution(root, v, normalized, current, timely=timely)
        for k, v in proof["branches"].items()
    }
    native = reports["native_cuda"]
    require(
        native.get("device") == "cuda"
        and native.get("steps_per_control") == 20
        and native.get("prohibited_old_calls") == 0,
        "actual generated native CUDA branch",
    )
    imports = native.get("actual_import_sha256")
    require(
        isinstance(imports, dict)
        and imports
        and all(normalized.get(k) == v for k, v in imports.items()),
        "actual imported source pins",
    )
    for name in (
        "tools/banc_m3_grouped_current_adjoint.py",
        "tools/banc_m3_ssp_adjoint.py",
        "tools/banc_m3_monotone_adjoint.py",
        "tools/banc_m3_checkpointed_rollout.py",
    ):
        require(name in imports, "actual native definition import")
    audit = proof.get("same_audit_cli", {})
    executable = descriptor(root, audit.get("source"), normalized, timely)
    qualification = read_json(
        descriptor(root, audit.get("qualification"), normalized, timely)
    )
    require(
        qualification.get("schema") == "flylab.m3.e342.audit-cli-qualification.v1"
        and qualification.get("passed") is True
        and qualification.get("source_sha256") == current,
        "same audit CLI qualification",
    )
    cases = qualification.get("cases", {})
    names = {
        "valid",
        "wrong_scale",
        "wrong_row",
        "wrong_boundary",
        "wrong_gradient",
        "wrong_actualtheta",
        "wrong_result",
    }
    require(set(cases) == names, "seven actual audit CLI cases")
    for name, branch in cases.items():
        report = closed_execution(
            root,
            branch,
            normalized,
            current,
            expected_exit=0 if name == "valid" else 1,
            timely=timely,
        )
        require(
            safe(root, report["audit_source_path"]) == executable,
            "same actual audit executable",
        )
    return imports


def validate_legacy_metadata(e318, e333, decision, receipts):
    checks = e318.get("checks")
    require(
        e318.get("pass") is True
        and e318.get("diagnostic_run_valid") is True
        and e318.get("learning_input_approved") is True
        and isinstance(checks, list)
        and len(checks) == 32507
        and all(
            isinstance(c, dict)
            and isinstance(c.get("name"), str)
            and c.get("pass") is True
            for c in checks
        ),
        "actual E318 list audit",
    )
    checks = e333.get("checks")
    require(
        e333.get("passed") is True
        and isinstance(checks, dict)
        and len(checks) == 3414
        and all(v is True for v in checks.values())
        and e333.get("run_valid") is False,
        "actual E333 dict audit with whole invalid",
    )
    require(
        decision.get("formal_verdict") == "invalid_deadline"
        and decision.get("run_valid") is False
        and decision.get("effective_direction_qualification") is False
        and decision.get("effective_continuation_qualification") is False,
        "E333 whole qualification remains invalid",
    )
    for start in WINDOWS:
        receipt = receipts.get("receipts", {}).get(f"window-{start}-neutral", {})
        require(receipt.get("status") == "backward_returned", "closed neutral producer")
        expected = {
            "forward_controls": 4,
            "forward_neural_ticks": 80,
            "recompute_controls": 4,
            "recompute_neural_ticks": 80,
            "backward_blocks": 4,
            "backward_block_ticks": 80,
        }
        require(
            receipt.get("confirmed") == expected
            and receipt.get("confirm_returned") == 2,
            "neutral producer device completion",
        )
        for key, value in expected.items():
            if key == "backward_block_ticks":
                require(
                    receipt.get(key + "_returned") == value, "producer returned ticks"
                )
            else:
                require(
                    receipt.get(key + "_requested") == value
                    and receipt.get(key + "_returned") == value,
                    "producer returned work",
                )


def make_freeze(*, generated_proof, root=ROOT):
    """Metadata/source-only preparation; never hashes an original array."""
    root = Path(root)
    require(root.resolve() == ROOT, "original preparation ROOT only")
    require(
        digest(safe(root, DESIGN)) == DESIGN_SHA
        and digest(safe(root, DIRECT)) == DIRECT_SHA
        and digest(safe(root, INVENTORY)) == INVENTORY_SHA,
        "reflected registration bytes",
    )
    direct, inventory = read_json(safe(root, DIRECT)), read_json(safe(root, INVENTORY))
    current = source_map(root)
    proof_path = safe(root, generated_proof)
    proof = read_json(proof_path)
    imports = validate_proof(root, proof, current)
    sources, metadata = {}, {}
    for name, entry in direct["direct_text_pins"].items():
        merge(
            sources if name.endswith(".py") else metadata, {name: entry["sha256"]}, root
        )
    merge(sources, current, root)
    merge(sources, imports, root)
    for name in (DESIGN, DIRECT, INVENTORY, REFLECTION, str(generated_proof)):
        merge(metadata, {name: digest(safe(root, name))}, root)
    for name, pin in {**sources, **metadata}.items():
        require(digest(safe(root, name)) == pin, "frozen direct text bytes: " + name)
    return {
        "schema": "flylab.m3.e342.freeze.v1",
        "fixed": copy.deepcopy(FIXED),
        "source_sha256": sources,
        "metadata_sha256": metadata,
        "actual_import_sha256": imports,
        "generated_proof": relative(root, generated_proof),
        "inventory": inventory,
        "output": OUTPUT,
        "original_retry": 0,
    }


def gate(freeze_sha256, review_sha256, *, timely=lambda: None):
    require(
        digest(safe(ROOT, FREEZE), timely) == check_sha(freeze_sha256)
        and digest(safe(ROOT, REVIEW), timely) == check_sha(review_sha256),
        "freeze/review bytes",
    )
    freeze, review = read_json(safe(ROOT, FREEZE)), read_json(safe(ROOT, REVIEW))
    require(
        freeze.get("schema") == "flylab.m3.e342.freeze.v1"
        and freeze.get("fixed") == FIXED
        and freeze.get("output") == OUTPUT
        and freeze.get("original_retry") == 0,
        "literal original contract",
    )
    current = source_map()
    require(
        review.get("approved") is True
        and review.get("blocking_defects") == []
        and review.get("source_sha256") == current
        and review.get("freeze_sha256") == freeze_sha256,
        "current ROOT original source review",
    )
    for group in ("source_sha256", "metadata_sha256"):
        normalized = {}
        merge(normalized, freeze.get(group, {}), ROOT)
        require(len(normalized) == len(freeze[group]), "frozen aliases")
        for name, pin in normalized.items():
            require(
                name.endswith((".py", ".json", ".md"))
                and digest(safe(ROOT, name), timely) == pin,
                "direct source/metadata pin",
            )
    require(
        all(freeze["source_sha256"].get(k) == v for k, v in current.items()),
        "current source closure",
    )
    require(
        freeze["metadata_sha256"].get(DESIGN) == DESIGN_SHA
        and freeze["metadata_sha256"].get(DIRECT) == DIRECT_SHA
        and freeze["metadata_sha256"].get(INVENTORY) == INVENTORY_SHA,
        "registration closure",
    )
    inventory = read_json(safe(ROOT, INVENTORY))
    require(
        freeze["inventory"] == inventory
        and inventory.get("input_count") == 23
        and inventory.get("selected_value_count") == 112
        and inventory.get("windows") == list(WINDOWS),
        "exact literal selected inventory",
    )
    imports = validate_proof(
        ROOT, read_json(safe(ROOT, freeze["generated_proof"])), current, timely
    )
    require(
        imports == freeze["actual_import_sha256"]
        and all(freeze["source_sha256"].get(k) == v for k, v in imports.items()),
        "actual import freeze",
    )
    direct = read_json(safe(ROOT, DIRECT))
    for name, entry in direct["direct_text_pins"].items():
        require(
            (
                freeze["source_sha256"]
                if name.endswith(".py")
                else freeze["metadata_sha256"]
            ).get(name)
            == entry["sha256"],
            "prior pin cannot be repinned",
        )
    validate_legacy_metadata(
        read_json(safe(ROOT, BASE + "/parent-e318-original-raw-audit.json")),
        read_json(safe(ROOT, BASE + "/parent-e333-original-raw-audit.json")),
        read_json(safe(ROOT, BASE + "/parent-e333-final-decision.json")),
        read_json(safe(ROOT, BASE + "/e333-smooth-efficacy-direction/receipts.json")),
    )
    for role, binding in inventory["bindings"].items():
        name = relative(ROOT, binding["path"])
        check_sha(binding["sha256"])
        require(
            type(binding["bytes"]) is int
            and binding["bytes"] > 0
            and safe(ROOT, name).stat().st_size == binding["bytes"],
            "input stat: " + role,
        )
        producer = read_json(safe(ROOT, binding["producer"]))
        local = (
            safe(ROOT, name)
            .relative_to(safe(ROOT, binding["producer"]).parent)
            .as_posix()
        )
        entry = producer.get("files", {}).get(local)
        if isinstance(entry, dict):
            require(
                entry.get("sha256") == binding["sha256"]
                and entry.get("bytes") == binding["bytes"],
                "selected producer manifest",
            )
        elif isinstance(entry, str):
            require(entry == binding["sha256"], "selected NPY producer manifest")
        else:
            require(
                role
                in (
                    "reference",
                    "actual_csr",
                    "motor_configuration",
                    "load_parameters",
                    "recurrent_mask",
                    "e289_roster",
                    "calibration",
                    "parameters",
                ),
                "missing producer pin",
            )
    require(not safe(ROOT, OUTPUT).exists(), "fresh original output")
    return freeze


def array_sha(value):
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def npy_file_sha(value):
    """Exactly StateStore's np.save(array, allow_pickle=False) file bytes."""
    import numpy as np

    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def dtype_metadata(value):
    import numpy as np

    require(isinstance(value, str), "dtype metadata string")
    dtype = np.dtype(value)
    require(
        not dtype.hasobject
        and dtype.fields is None
        and dtype.subdtype is None
        and dtype.kind in "bifu"
        and dtype.itemsize in (1, 2, 4, 8)
        and (dtype.itemsize == 1 or dtype.str.startswith("<")),
        "little endian scalar numeric dtype",
    )
    require(not value.startswith((">", "=")), "explicit incompatible byte order")
    return dtype.str


def strict_npy_header(stream):
    def take(count):
        value = stream.read(count)
        require(len(value) == count, "truncated NPY")
        return value

    require(take(6) == b"\x93NUMPY", "NPY magic")
    version = tuple(take(2))
    require(version in ((1, 0), (2, 0)), "NPY version")
    size = struct.unpack(
        "<H" if version == (1, 0) else "<I", take(2 if version == (1, 0) else 4)
    )[0]
    require(0 < size <= 65536, "bounded NPY header")
    text = take(size).decode("latin1")
    require(text.endswith("\n"), "NPY header newline")
    tree = ast.parse(text.strip(), mode="eval")
    require(
        isinstance(tree.body, ast.Dict)
        and len(tree.body.keys) == 3
        and {ast.literal_eval(k) for k in tree.body.keys}
        == {"descr", "fortran_order", "shape"},
        "strict NPY header mapping",
    )
    fields = ast.literal_eval(tree)
    require(
        type(fields["fortran_order"]) is bool
        and isinstance(fields["shape"], tuple)
        and len(fields["shape"]) <= 8
        and all(type(v) is int and v >= 0 for v in fields["shape"]),
        "typed NPY axes",
    )
    dtype = dtype_metadata(fields["descr"])
    return {
        "dtype": dtype,
        "shape": list(fields["shape"]),
        "fortran_order": fields["fortran_order"],
        "header_bytes": stream.tell(),
        "version": list(version),
    }


def load_selected(root, binding, ledger, *, timely=lambda: None, generated=False):
    import numpy as np

    path = (
        generated_path(root, binding["path"])
        if generated
        else safe(root, binding["path"])
    )
    schema = binding["selected_schema"]
    observation = {
        "path": relative(root, str(path)),
        "expected_sha256": binding["sha256"],
        "numeric_reads": [],
        "selected": {},
        "posthash_verified": False,
    }
    ledger.append(observation)
    require(path.stat().st_size == binding["bytes"], "selected input byte length")
    if binding["kind"] == "npz":
        from tools.qualify_c_banc_m3_body_packet import scan_archive

        observation["scanner"] = scan_archive(
            path,
            expected_sha256=binding["sha256"],
            descriptor_scope="observation",
            timely=timely,
        )
        headers = observation["scanner"]["descriptor"]["members"]
        for key, spec in schema.items():
            require(
                key in headers
                and headers[key]["shape"] == spec["shape"]
                and headers[key]["dtype"] == dtype_metadata(spec["dtype"])
                and headers[key]["fortran_order"] is spec.get("fortran_order", False),
                "selected header: " + key,
            )
        observation["header_only_extras"] = sorted(set(headers) - set(schema))
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in schema}
    elif binding["kind"] == "npy":
        require(
            set(schema) == {"value"} and digest(path, timely) == binding["sha256"],
            "NPY prehash",
        )
        with path.open("rb") as stream:
            header = strict_npy_header(stream)
        spec = schema["value"]
        require(
            header["dtype"] == dtype_metadata(spec["dtype"])
            and header["shape"] == spec["shape"]
            and header["fortran_order"] is False
            and path.stat().st_size
            == header["header_bytes"]
            + math.prod(header["shape"]) * np.dtype(header["dtype"]).itemsize,
            "bounded exact selected NPY",
        )
        observation["header"] = header
        arrays = {"value": np.load(path, allow_pickle=False)}
    else:
        raise ValueError("selected NPZ/NPY only")
    for key, value in arrays.items():
        timely()
        spec = schema[key]
        require(
            value.shape == tuple(spec["shape"])
            and value.dtype.str == dtype_metadata(spec["dtype"])
            and not value.dtype.hasobject
            and np.isfinite(value).all(),
            "finite typed selected value: " + key,
        )
        observation["numeric_reads"].append(key)
        observation["selected"][key] = {
            "shape": list(value.shape),
            "dtype": value.dtype.str,
            "c_order_sha256": array_sha(value),
            "npy_file_sha256": npy_file_sha(value),
        }
    require(
        digest(path, timely) == binding["sha256"]
        and path.stat().st_size == binding["bytes"],
        "selected posthash",
    )
    timely()
    observation["posthash_verified"] = True
    return arrays


def metadata_packet(root, inventory):
    initial_path = BASE + "/e305-ssp-first-control/neutral/initial-neural"
    observed_path = BASE + "/e318-full-drive-capture/observed-start"
    initial = read_json(safe(root, initial_path + "/state.json"))
    observed = read_json(safe(root, observed_path + "/state.json"))
    initial_manifest = read_json(safe(root, initial_path + "/manifest.json"))
    observed_manifest = read_json(safe(root, observed_path + "/manifest.json"))
    require(
        initial["model_hash"]
        == inventory["model_identity"]
        == observed["network"]["model_hash"]
        and initial["schema"] == "flylab.research-rate.v1"
        and initial["device"] == "cuda"
        and initial["dt"] == 0.00025
        and initial["tick"] == 0
        and initial["adaptation_gain"] == 8.0
        and initial["adaptation_tau_s"] == 0.3,
        "original native metadata",
    )
    require(
        initial["recurrent_resource"]["source_model_hash"]
        == "a31f899cd948a4297a73db66569e5edc2210ac9aa3601aa83e96cb9ff68ce244"
        and initial["motor_timescale"]["source_motor_identity"]
        == "a8310c1cff7e7dfa4b71b506fa3828a29b0450d8e62ddd17b611c22ee59ca0a8",
        "original physiological provenance",
    )
    require(
        observed["adapter"]["hash"] == inventory["decoder_hash"],
        "original decoder identity",
    )
    for role, owner, manifest, base, leaf in (
        (
            "inverse",
            initial["motor_timescale"],
            initial_manifest,
            initial_path,
            "inverse32",
        ),
        (
            "output_mask",
            observed["network"],
            observed_manifest,
            observed_path,
            "output_mask",
        ),
    ):
        pointer = owner[leaf]
        require(
            set(pointer) == {"__npy__"}
            and base + "/" + pointer["__npy__"] == inventory["bindings"][role]["path"]
            and manifest["files"][pointer["__npy__"]]
            == inventory["bindings"][role]["sha256"],
            "semantic NPY pointer",
        )
    for leaf, link in inventory["decoder_original_npy_links"].items():
        group, key = leaf.split(".")
        pointer = observed["adapter"][group][key]
        require(
            set(pointer) == {"__npy__"}
            and observed_path + "/" + pointer["__npy__"] == link["original_pointer"]
            and observed_manifest["files"][pointer["__npy__"]]
            == link["original_npy_sha256"],
            "decoder semantic pointer",
        )
    require(
        observed["body"]["action"]["joint_angles"] == inventory["decoder_neutral"],
        "JSON literal neutral",
    )
    return {
        "neutral": inventory["decoder_neutral"],
        "decoder_links": inventory["decoder_original_npy_links"],
        "model_identity": inventory["model_identity"],
        "decoder_hash": inventory["decoder_hash"],
    }


def load_packet(
    root, inventory, *, dimensions=None, metadata=None, timely=lambda: None, ledger=None
):
    import numpy as np

    generated = dimensions is not None
    ledger = [] if ledger is None else ledger
    meta = metadata_packet(root, inventory) if metadata is None else metadata
    require(generated or metadata is None, "original metadata override forbidden")
    packet = {"metadata": meta, "ledger": ledger, "bindings": inventory["bindings"]}
    for role in inventory["role_order"]:
        value = load_selected(
            root,
            inventory["bindings"][role],
            ledger,
            timely=timely,
            generated=generated,
        )
        if role.startswith("drive-"):
            start = int(role.split("-")[1])
            first, begin, end = SLICES[WINDOWS.index(start)]
            require(
                np.array_equal(
                    value["control_tick"], np.arange(first, first + 25, dtype=np.int64)
                )
                and np.array_equal(value["network_tick"], value["control_tick"] * 20),
                "complete drive chronology",
            )
            value = {
                "drive": value["drive"][begin:end].copy(),
                "source_slice": [begin, end],
            }
        packet[role] = value
    validate_packet(packet, dimensions=dimensions)
    return packet


def exact(left, right):
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def validate_packet(packet, *, dimensions=None):
    import numpy as np

    a = axes(dimensions)
    csr, mask, roster = (
        packet[k] for k in ("actual_csr", "recurrent_mask", "e289_roster")
    )
    require(
        csr["indptr"].dtype == np.int32
        and csr["indptr"].shape == (a["nodes"] + 1,)
        and csr["indices"].dtype == np.int32
        and csr["indices"].shape == (a["edges"],)
        and csr["data"].dtype == np.float32
        and csr["data"].shape == (a["edges"],),
        "actual CSR axes",
    )
    require(
        np.array_equal(csr["indptr"][[0, -1]], [0, a["edges"]])
        and np.all(np.diff(csr["indptr"]) >= 0)
        and np.all((csr["indices"] >= 0) & (csr["indices"] < a["nodes"])),
        "actual CSR topology",
    )
    for key, size in (
        ("population", a["population"]),
        ("positions", a["selected"]),
        ("edge_parameters", a["selected"]),
    ):
        require(
            mask[key].dtype == np.int64 and mask[key].shape == (size,),
            "integer anatomical selection",
        )
    ids, groups = mask["positions"], mask["edge_parameters"]
    require(
        len(np.unique(ids)) == len(ids)
        and np.all((ids >= 0) & (ids < a["edges"]))
        and np.array_equal(np.unique(groups), np.arange(a["groups"]))
        and exact(mask["population"], roster["premotor_ids"])
        and exact(ids, roster["edge_ids"])
        and exact(roster["weights32"], csr["data"][ids])
        and np.all(csr["data"][ids] != 0),
        "original efficacy roster",
    )
    config, load = packet["motor_configuration"], packet["load_parameters"]
    require(
        exact(load["tau32"], config["membrane_tau"])
        and np.all(load["tau32"] > 0)
        and np.all(load["threshold32"] == np.float32(7.5)),
        "original tau/threshold",
    )
    motors, inverse, gain = (
        config["motor_ids"],
        packet["inverse"]["value"],
        config["cell_gain"],
    )
    require(
        motors.dtype == np.int64
        and motors.shape == (391,)
        and np.array_equal(motors, np.unique(motors))
        and np.all((motors >= 0) & (motors < a["nodes"]))
        and inverse.dtype == np.float32
        and inverse.shape == (a["nodes"],)
        and np.all(inverse > 0)
        and gain.dtype == np.float32
        and gain.shape == (a["nodes"],)
        and np.all(gain >= 0),
        "full native physiology",
    )
    require(
        exact(inverse[motors], config["actual_motor_inverse32"])
        and exact(gain[motors], config["actual_motor_gain32"])
        and np.all(packet["output_mask"]["value"] == np.float32(1)),
        "motor override/all-native-output mask",
    )
    require(
        exact(motors, packet["calibration"]["motor_ids"]), "actual MN decoder order"
    )
    for leaf, link in packet["metadata"]["decoder_links"].items():
        role, key = leaf.split(".")
        require(
            npy_file_sha(packet[role][key]) == link["original_npy_sha256"],
            "serialized decoder file identity: " + leaf,
        )
    raw = packet["control_raw"]
    require(
        np.array_equal(raw["control_tick"], np.arange(301, dtype=np.int64))
        and np.array_equal(raw["network_tick"], raw["control_tick"] * 20)
        and exact(
            raw["command"][0],
            np.asarray(packet["metadata"]["neutral"], dtype=np.float64),
        ),
        "raw original clock/neutral",
    )
    for start in WINDOWS:
        state, endpoint = packet[f"start-{start}"], packet[f"endpoint-{start}"]
        require(
            int(state["source_control_tick"]) == start
            and int(state["source_neural_row_index"]) == start - 1
            and int(state["decoder_replay_boundary_index"]) == start
            and exact(state["fixed_original_q_boundary"], raw["command"][start])
            and exact(
                state["fixed_original_q_boundary"],
                endpoint["fixed_original_q_boundary"],
            ),
            "fixed original boundary/index",
        )
        require(
            all(
                np.all(state[k] >= 0)
                for k in ("rate", "adaptation", "bank", "original_activation")
            ),
            "nonnegative native start",
        )
    return True


@contextmanager
def prohibited_call_guard(record):
    previous = sys.getprofile()

    def guard(frame, event, arg):
        if event == "call":
            module = frame.f_globals.get("__name__", "")
            name = frame.f_code.co_name
            owner = frame.f_locals.get("self", frame.f_locals.get("cls"))
            owner_name = (
                owner.__name__ if isinstance(owner, type) else type(owner).__name__
            )
            old_worker = (
                module.startswith("tools.")
                and module != __name__
                and (
                    name in ("main", "compute", "run", "load_inputs", "metadata_worker")
                    or owner_name == "Dependencies"
                )
            )
            old_load = owner_name in ("StateStore", "GraphStore") and name in (
                "load",
                "__init__",
            )
            body = (
                owner_name in ("BancWalkingSession", "TetheredWorld", "WarpWorldBatch")
                and name == "__init__"
            )
            if old_worker or old_load or body:
                record["prohibited_old_calls"] += 1
                raise RuntimeError("prohibited old callable: " + module + "." + name)
        if previous is not None:
            previous(frame, event, arg)

    sys.setprofile(guard)
    try:
        yield
    finally:
        sys.setprofile(previous)


def actual_imports(root=ROOT):
    result = {}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if not name:
            continue
        path = Path(name).resolve()
        try:
            local = path.relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            continue
        if local.endswith(".py") and local.startswith(("tools/", "flylab/")):
            result[local] = digest(path)
    return result


def assemble(packet, *, device="cuda", dimensions=None):
    """Fresh current native assembly; no predecessor Dependencies/load calls."""
    a = axes(dimensions)
    validate_packet(packet, dimensions=dimensions)
    record = {"prohibited_old_calls": 0}
    with prohibited_call_guard(record):
        import numpy as np
        import torch
        from scipy.sparse import csr_matrix

        from tools.banc_m3_checkpointed_rollout import CheckpointedControlRollout
        from tools.banc_m3_grouped_current_adjoint import GroupedCurrent
        from tools.banc_m3_monotone_adjoint import MonotoneDecoder
        from tools.banc_m3_smooth_temporal import SmoothTemporalObjective
        from tools.banc_m3_ssp_adjoint import SSPAdjointModel

        torch.set_num_threads(1)
        csr = packet["actual_csr"]
        weights = csr_matrix(
            (csr["data"], csr["indices"], csr["indptr"]), shape=(a["nodes"], a["nodes"])
        )
        mask = packet["recurrent_mask"]
        operator = GroupedCurrent(
            weights, mask["positions"], mask["edge_parameters"], a["groups"], device
        )
        config, load = packet["motor_configuration"], packet["load_parameters"]
        coefficient = np.full(a["nodes"], np.float32(1 / 200), np.float32)
        cap = np.full(a["nodes"], np.float32(200), np.float32)
        neural = SSPAdjointModel(
            operator,
            load["tau32"],
            coefficient,
            load["threshold32"],
            cap,
            config["cell_gain"],
            packet["inverse"]["value"],
            0.00025,
        )
        calibration, parameters = packet["calibration"], packet["parameters"]
        decoder = MonotoneDecoder(
            calibration,
            parameters["tau"],
            parameters["kappa"],
            parameters["slopes"],
            parameters["membership"],
            np.asarray(packet["metadata"]["neutral"], np.float64),
            device=device,
        )
        rollout = CheckpointedControlRollout(
            neural,
            decoder,
            calibration["motor_ids"],
            steps_per_control=a["steps_per_control"],
            control_dt=a["control_dt"],
            checkpoint_controls=1,
            max_controls=4,
        )
        reference = packet["reference"]
        objective = SmoothTemporalObjective(
            reference["curves"],
            np.zeros((99, 42), np.float64),
            np.zeros((99, 6), np.float64),
            device=device,
            normalization=reference["normalization"],
        )
        require(
            all(
                exact(objective.canonical_constants[k], reference[k]) for k in REFERENCE
            ),
            "actual smooth constants bytes",
        )
        for name, expected in (
            ("tau", load["tau32"]),
            ("threshold", load["threshold32"]),
            ("coefficient", coefficient),
            ("cap", cap),
            ("gain", config["cell_gain"]),
            ("inverse_adaptation_tau", packet["inverse"]["value"]),
        ):
            require(
                exact(getattr(neural, name).detach().cpu().numpy(), expected),
                "actual native physiology tensor",
            )
        record["constant_metadata"] = objective.constant_metadata
        record["actual_import_sha256"] = actual_imports()
    return {
        "rollout": rollout,
        "objective": objective,
        "operator": operator,
        "record": record,
    }


def window_loss(objective, boundary, q):
    import torch

    require(
        tuple(boundary.shape) == (42,) and tuple(q.shape) == (4, 42),
        "four-control fixed boundary",
    )
    previous = torch.cat((boundary.detach().clone()[None], q[:3]), dim=0)
    return objective.evaluate_pairs(previous, q)


def loss_vector(terms):
    import numpy as np

    return np.array(
        [
            float(terms[k].detach().cpu())
            for k in (
                "successor_loss",
                "coordination_loss",
                "training_loss",
                "lr_coordination_loss",
            )
        ],
        np.float64,
    )


def receipt_dict(receipt):
    return {k: copy.deepcopy(v) for k, v in vars(receipt).items() if k != "device"}


def work_counts(receipts):
    keys = (
        "forward_controls_requested",
        "forward_controls_returned",
        "forward_neural_ticks_requested",
        "forward_neural_ticks_returned",
        "recompute_controls_requested",
        "recompute_controls_returned",
        "recompute_neural_ticks_requested",
        "recompute_neural_ticks_returned",
        "backward_blocks_requested",
        "backward_blocks_returned",
        "backward_block_ticks_returned",
    )
    return {k: sum(r.get(k, 0) for r in receipts.values()) for k in keys}


def make_direction(gradient):
    import numpy as np

    require(
        gradient.dtype == np.float64
        and gradient.ndim == 1
        and np.isfinite(gradient).all(),
        "finite newly computed G_C",
    )
    magnitude = float(np.max(np.abs(gradient)))
    if magnitude == 0:
        return None
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        direction = gradient / magnitude
        plus = np.float32(1 + 0.001 * direction).astype(np.float64)
        minus = np.float32(1 - 0.001 * direction).astype(np.float64)
    require(np.all(plus > 0) and np.all(minus > 0), "positive actual f32 efficacy")
    return plus, minus


def patch_weights(csr, mask, theta):
    import numpy as np

    require(
        theta.dtype == np.float64
        and exact(theta, theta.astype(np.float32).astype(np.float64))
        and np.isfinite(theta).all()
        and np.all(theta > 0),
        "f32-promoted efficacy",
    )
    data = csr["data"].copy()
    positions, groups = mask["positions"], mask["edge_parameters"]
    with np.errstate(over="raise", invalid="raise"):
        selected = np.float32(data[positions] * theta.astype(np.float32)[groups])
    data[positions] = selected
    count = len(np.unique(data.view(np.uint32)))
    require(
        np.isfinite(data).all()
        and np.array_equal(np.signbit(selected), np.signbit(csr["data"][positions])),
        "signed selected weights",
    )
    return selected, {
        "full_native_palette_count": count,
        "native_deployable": count <= 16384,
        "selected_c_order_sha256": array_sha(selected),
        "full_data_c_order_sha256": array_sha(data),
    }


def reconciliation(observed, predicted):
    import numpy as np

    require(
        observed.dtype == predicted.dtype == np.float64
        and observed.shape == predicted.shape == (4,)
        and np.isfinite(observed).all()
        and np.isfinite(predicted).all(),
        "finite four-window reconciliation",
    )
    o, p = float(observed.sum()), float(predicted.sum())
    denominator = max(float(np.abs(observed).sum()), float(np.abs(predicted).sum()))
    meaningful = abs(o) > 1e-9 and abs(p) > 1e-9
    signed_error = abs(o - p) / abs(p) if meaningful else None
    l1_error = (
        float(np.abs(observed - predicted).sum()) / denominator
        if denominator > 1e-9
        else None
    )
    same_sign = meaningful and ((o > 0) == (p > 0))
    return {
        "O": o,
        "prediction": p,
        "meaningful": meaningful,
        "same_sign": same_sign,
        "relative_error": signed_error,
        "window_L1_relative_error": l1_error,
        "passed": bool(
            same_sign
            and signed_error <= 0.05
            and l1_error is not None
            and l1_error <= 0.05
        ),
        "null_reason": None
        if meaningful and l1_error is not None
        else "no_direction_information",
    }


def classify(raw):
    import numpy as np

    oq, ot = reconciliation(raw["O"], raw["Q"]), reconciliation(raw["O"], raw["T"])
    delta = raw["theta_plus"] - raw["theta_minus"]
    aggregate = float(np.dot(raw["sum_gtheta_C"], delta))
    summed = float(raw["T"].sum())
    require(
        math.isfinite(aggregate)
        and math.isfinite(summed)
        and math.isclose(aggregate, summed, rel_tol=1e-10, abs_tol=1e-12)
        and aggregate >= -1e-9,
        "construction T aggregate invariant",
    )
    ic, is_ = float(raw["I_C"].sum()), float(raw["I_S"].sum())
    decrease = ic >= 1e-9
    tradeoff = (
        float(raw["minus_loss"][:, 0].mean())
        > float(raw["neutral_loss"][:, 0].mean()) + 1e-9
    )
    return {
        "observed_response": "C_decrease"
        if decrease
        else "C_increase"
        if ic <= -1e-9
        else "inconclusive",
        "I_C": ic,
        "I_S": is_,
        "O_Q": oq,
        "O_T": ot,
        "aggregate_T_from_sum_gradient": aggregate,
        "aggregate_T_from_windows": summed,
        "component_tradeoff": tradeoff,
        "parameter_output_transfer_mismatch": oq["passed"] and not ot["passed"],
        "direction_qualification": bool(
            decrease and oq["passed"] and ot["passed"] and not tradeoff
        ),
        "full_parameter_Jacobian_independently_proven": False,
    }


class CollectionFailure(RuntimeError):
    def __init__(self, error, raw, receipts, record, stage):
        super().__init__(str(error))
        self.error, self.raw, self.receipts, self.record, self.stage = (
            error,
            raw,
            receipts,
            record,
            stage,
        )


def collect(
    packet,
    *,
    device="cuda",
    dimensions=None,
    timely=lambda: None,
    progress=lambda value: None,
):
    import numpy as np
    import torch

    a = axes(dimensions)
    raw = {k: v.copy() for k, v in packet["reference"].items()}
    raw["boundary"] = np.stack(
        [packet[f"start-{c}"]["fixed_original_q_boundary"].copy() for c in WINDOWS]
    )
    receipts, record, graphs = {}, {"prohibited_old_calls": 0}, []
    stage = "assembly"

    def save(name, value):
        if name not in raw:
            raw[name] = value[None].copy()
        else:
            raw[name] = np.concatenate((raw[name], value[None]), axis=0)

    def emit():
        progress({"raw": raw, "receipts": receipts, "record": record, "stage": stage})

    try:
        timely()
        bundle = assemble(packet, device=device, dimensions=dimensions)
        rollout, objective = bundle["rollout"], bundle["objective"]
        record = bundle["record"]
        target = rollout.device
        tensor = lambda value: torch.as_tensor(value.copy(), device=target)
        mask = tensor(packet["output_mask"]["value"])

        def forward(c, arm, theta):
            nonlocal stage
            stage = f"{arm}-{c}-forward"
            key = f"window-{c}-{arm}"
            state = packet[f"start-{c}"]
            timely()
            try:
                result = rollout.run(
                    *(
                        tensor(state[k])
                        for k in ("rate", "adaptation", "bank", "original_activation")
                    ),
                    tensor(packet[f"drive-{c}"]["drive"]),
                    theta,
                    mask,
                )
            except Exception as error:
                if hasattr(error, "receipt"):
                    receipts[key] = receipt_dict(error.receipt)
                    emit()
                raise
            receipt = result["receipt"]
            receipts[key] = receipt_dict(receipt)
            # Host-acquired returned raw precedes confirmation/reproduction/science.
            save("q_" + arm, result["q"].detach().cpu().numpy())
            save("tendon_" + arm, result["tendon"].detach().cpu().numpy())
            if arm == "neutral":
                for name in ("rate", "adaptation", "bank", "original_activation"):
                    save("neutral_end_" + name, result[name].detach().cpu().numpy())
            emit()
            try:
                receipt.confirm()
            finally:
                receipts[key] = receipt_dict(receipt)
                emit()
            timely()
            return result, key

        for index, c in enumerate(WINDOWS):
            theta = torch.ones(
                a["groups"], dtype=torch.float64, device=target, requires_grad=True
            )
            result, key = forward(c, "neutral", theta)
            endpoint = packet[f"endpoint-{c}"]
            stage = f"neutral-{c}-reproduction"
            for name in ("q", "tendon", "bank", "original_activation"):
                tolerance = 1e-11 if name in ("q", "tendon") else 1e-10
                require(
                    np.allclose(
                        result[name].detach().cpu().numpy(),
                        endpoint[name],
                        rtol=0,
                        atol=tolerance,
                    ),
                    "neutral reproduction: " + name,
                )
            for name in ("rate", "adaptation"):
                require(
                    exact(result[name].detach().cpu().numpy(), endpoint[name]),
                    "neutral native bytes: " + name,
                )
            terms = window_loss(objective, tensor(raw["boundary"][index]), result["q"])
            losses = loss_vector(terms)
            save("neutral_loss", losses)
            # E333 input order is total,S,C,LR; our raw is S,C,total,LR.
            require(
                np.allclose(
                    losses[[2, 0, 1, 3]], endpoint["loss_terms"], rtol=1e-10, atol=1e-11
                ),
                "saved neutral loss order/reproduction",
            )
            graphs.append((result, theta, terms, key))
            emit()
        require(
            len(graphs) == 4
            and work_counts(receipts)["backward_blocks_requested"] == 0,
            "all neutral before new gradient",
        )
        record["all_neutral_reproduced_before_gradient"] = True
        for result, theta, terms, key in graphs:
            stage = key + "-C-backward"
            timely()
            receipt = result["receipt"]
            before = receipt_dict(receipt)
            ell = terms["coordination_loss"] / 4
            (gq,) = torch.autograd.grad(ell, result["q"], retain_graph=True)
            require(
                before == receipt_dict(receipt), "gq extraction performed neural work"
            )
            save("gq_C", gq.detach().cpu().numpy())
            require(
                theta.grad is None and theta.is_leaf, "independent fresh theta leaf"
            )
            emit()
            try:
                ell.backward()
            finally:
                receipts[key] = receipt_dict(receipt)
                emit()
            require(theta.grad is not None, "C-only theta VJP missing")
            save("gtheta_C", theta.grad.detach().cpu().numpy())
            emit()
            try:
                receipt.confirm()
            finally:
                receipts[key] = receipt_dict(receipt)
                emit()
            timely()
        raw["sum_gtheta_C"] = raw["gtheta_C"].sum(axis=0)
        require(
            all(np.isfinite(v).all() for v in raw.values()), "finite neutral/VJP raw"
        )
        direction = make_direction(raw["sum_gtheta_C"])
        if direction is None:
            record["branch"] = "zero_C_gradient"
        else:
            raw["theta_plus"], raw["theta_minus"] = direction
            palettes = {}
            for arm in ("plus", "minus"):
                raw["selected_weights_" + arm], palettes[arm] = patch_weights(
                    packet["actual_csr"], packet["recurrent_mask"], raw["theta_" + arm]
                )
            record["palettes"] = palettes
            emit()
            if not all(value["native_deployable"] for value in palettes.values()):
                record["branch"] = "native_palette_not_deployable"
            else:
                record["branch"] = "normal"
                for arm in ("plus", "minus"):
                    theta = tensor(raw["theta_" + arm])
                    for index, c in enumerate(WINDOWS):
                        with torch.no_grad():
                            result, _ = forward(c, arm, theta)
                            terms = window_loss(
                                objective, tensor(raw["boundary"][index]), result["q"]
                            )
                            save(arm + "_loss", loss_vector(terms))
                            emit()
                delta_q = raw["q_plus"] - raw["q_minus"]
                raw["O"] = (raw["plus_loss"][:, 1] - raw["minus_loss"][:, 1]) / 4
                raw["Q"] = np.sum(raw["gq_C"] * delta_q, axis=(1, 2))
                raw["T"] = raw["gtheta_C"] @ (raw["theta_plus"] - raw["theta_minus"])
                raw["I_C"] = (raw["neutral_loss"][:, 1] - raw["minus_loss"][:, 1]) / 4
                raw["I_S"] = (raw["neutral_loss"][:, 0] - raw["minus_loss"][:, 0]) / 4
        stage = "completed-raw"
        emit()
        require(all(np.isfinite(v).all() for v in raw.values()), "finite response raw")
        counts = work_counts(receipts)
        forward_controls = 48 if record["branch"] == "normal" else 16
        expected = {
            "forward_controls_requested": forward_controls,
            "forward_controls_returned": forward_controls,
            "forward_neural_ticks_requested": forward_controls * a["steps_per_control"],
            "forward_neural_ticks_returned": forward_controls * a["steps_per_control"],
            "recompute_controls_requested": 16,
            "recompute_controls_returned": 16,
            "recompute_neural_ticks_requested": 16 * a["steps_per_control"],
            "recompute_neural_ticks_returned": 16 * a["steps_per_control"],
            "backward_blocks_requested": 16,
            "backward_blocks_returned": 16,
            "backward_block_ticks_returned": 16 * a["steps_per_control"],
        }
        require(counts == expected, "actual returned native work")
        record["work"] = counts
        record["gq_neural_work"] = 0
        record["neural_backwards"] = 4
        record["control_dt"] = a["control_dt"]
        record["steps_per_control"] = a["steps_per_control"]
        return raw, receipts, record
    except Exception as error:
        raise CollectionFailure(error, raw, receipts, record, stage) from error


def analyze(raw, record):
    require(
        record.get("branch")
        in ("normal", "zero_C_gradient", "native_palette_not_deployable"),
        "completed branch",
    )
    if record["branch"] != "normal":
        return {
            "branch": record["branch"],
            "observed_response": "not_evaluated",
            "I_C": None,
            "I_S": None,
            "O_Q": None,
            "O_T": None,
            "component_tradeoff": None,
            "direction_qualification": False,
            "full_parameter_Jacobian_independently_proven": False,
        }
    return {
        "branch": "normal",
        **classify(raw),
        "arm_mean_losses": {
            arm: raw[arm + "_loss"].mean(axis=0).tolist()
            for arm in ("neutral", "plus", "minus")
        },
        "loss_order": list(LOSS_ORDER),
        "per_window": {k: raw[k].tolist() for k in ("O", "Q", "T", "I_C", "I_S")},
    }


class Clock:
    def __init__(self, clock=time.perf_counter, s0=None):
        self.clock = clock
        self.s0 = clock() if s0 is None else s0
        self.final_deadline = None

    def check(self, stage="science"):
        now = self.clock()
        limit = self.s0 + (48 if stage == "science" else 60)
        if self.final_deadline is not None:
            limit = min(
                limit if stage == "science" else self.s0 + 65, self.final_deadline
            )
        require(math.isfinite(now) and now >= self.s0, "monotone finite clock")
        if now > limit:
            raise TimeoutError("absolute " + stage + " deadline")

    def finalize(self):
        self.check("internal")
        self.final_deadline = min(self.s0 + 65, self.clock() + 5)


class GPUMemoryWatch:
    """Pre-context nvidia-smi total-device samples; own thread/process only."""

    def __init__(self, *, index=0, interval_s=0.1, sample=None):
        require(
            type(index) is int
            and index == 0
            and type(interval_s) in (int, float)
            and 0 < interval_s <= 1,
            "GPU sampler arguments",
        )
        self.index, self.interval_s = index, interval_s
        self._sample = self._smi if sample is None else sample
        self._stop = threading.Event()
        self._thread = None
        self._pids = set()
        self._errors = []
        self._samples = []

    def _smi(self):
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=flags,
        )
        self._pids.add(process.pid)
        try:
            stdout, stderr = process.communicate(timeout=2)
            require(
                process.returncode == 0 and not stderr.strip(),
                "GPU sampler process failure",
            )
            value = stdout.decode("ascii").strip()
            require(value.isdigit(), "GPU sampler integer MiB")
            return int(value) * MIB
        except Exception:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)
            raise

    def _once(self):
        value = self._sample()
        require(type(value) is int and value >= 0, "GPU sample bytes")
        self._samples.append(value)

    def start(self):
        require(self._thread is None, "sampler single start")
        self._once()

        def loop():
            while not self._stop.wait(self.interval_s):
                try:
                    self._once()
                except Exception as error:  # noqa: BLE001 -- Own sampler errors revoke qualification.
                    self._errors.append(
                        {"type": type(error).__name__, "message": str(error)[:512]}
                    )
                    self._stop.set()

        self._thread = threading.Thread(
            target=loop, name="E342-owned-GPU-memory", daemon=True
        )
        self._thread.start()
        return self.snapshot()

    def snapshot(self):
        baseline = self._samples[0] if self._samples else None
        peak = max(self._samples) if self._samples else None
        return {
            "precontext_baseline_bytes": baseline,
            "sampled_total_peak_bytes": peak,
            "sampled_additional_peak_bytes": max(0, peak - baseline)
            if baseline is not None
            else None,
            "quantization_allowance_bytes": 2 * MIB,
            "sample_count": len(self._samples),
            "sampler_errors": copy.deepcopy(self._errors),
            "owned_sampler_pids": sorted(self._pids),
            "owned_thread_alive": self._thread.is_alive() if self._thread else False,
            "context_included": True,
            "unsampled_or_external_work_exclusion_proven": False,
        }

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        result = self.snapshot()
        require(
            not result["owned_thread_alive"],
            "owned GPU sampler thread did not terminate",
        )
        return result


def usage(device):
    import psutil

    owner = psutil.Process()
    rss = owner.memory_info().rss
    for child in owner.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    result = {"tree_rss_bytes": rss, "torch_cuda_peak_bytes": None}
    if device == "cuda":
        import torch

        result["torch_cuda_peak_bytes"] = torch.cuda.max_memory_allocated()
    return result


class BudgetStream:
    def __init__(self, stream, quota, check):
        self.stream, self.quota, self.check = stream, quota, check

    def write(self, value):
        require(self.stream.tell() + len(value) <= self.quota, "staged per-file quota")
        self.check(len(value))
        return self.stream.write(value)

    def __getattr__(self, key):
        return getattr(self.stream, key)


class Store:
    def __init__(self, output, clock):
        require(not output.exists(), "fresh output directory")
        output.mkdir(parents=True)
        self.output, self.clock, self.written = output, clock, set()

    def check(self, extra=0, *, failure=False):
        actual = sum(p.stat().st_size for p in self.output.rglob("*") if p.is_file())
        reserved = sum(
            v
            for k, v in QUOTAS.items()
            if k not in self.written and k not in ("raw.npz",)
        )
        # Raw can be partial/absent on failure; its full quota is not a success claim.
        require(
            actual + extra + reserved + (0 if failure else FAILURE_QUOTA) <= OUTPUT_CAP,
            "recursive output/staging quota",
        )

    def put(self, name, value, *, failure=False):
        quota = FAILURE_QUOTA if failure else QUOTAS[name]
        require(
            len(value) <= quota and not (self.output / name).exists(),
            "fresh per-file publication quota",
        )
        self.clock.check(
            "final" if self.clock.final_deadline is not None else "internal"
        )
        self.check(len(value), failure=failure)
        path = self.output / name
        with path.open("xb") as stream:
            stream.write(value)
            stream.flush()
        self.written.add(name)
        require(path.read_bytes() == value, "published byte readback")
        self.check(failure=failure)
        return path

    def raw(self, arrays):
        import numpy as np

        require(not (self.output / "raw.npz").exists(), "raw written once")
        temporary = self.output / "raw.npz.partial"
        with temporary.open("xb") as stream:
            bounded = BudgetStream(stream, QUOTAS["raw.npz"], self.check)
            np.savez(bounded, **arrays)
            stream.flush()
        self.clock.check("internal")
        require(temporary.stat().st_size <= QUOTAS["raw.npz"], "raw cap")
        temporary.rename(self.output / "raw.npz")
        self.written.add("raw.npz")
        self.check()
        return self.output / "raw.npz"


def stdout_bytes(value):
    sys.stdout.buffer.write(value)
    sys.stdout.buffer.flush()


def worker(
    root,
    packet_loader,
    *,
    output,
    device,
    dimensions=None,
    clock=time.perf_counter,
    fault=lambda stage: None,
    printer=stdout_bytes,
    expected_imports=None,
    watcher=None,
    meter=usage,
    s0=None,
):
    """Shared original/generated lifecycle; numerical raw precedes science."""
    timer = Clock(clock, s0)
    terminal = {
        "schema": "flylab.m3.e342.result.v1",
        "worker_valid": False,
        "mode": "generated" if dimensions is not None else "original",
        "device": device,
        "analysis": {
            "observed_response": "not_evaluated",
            "direction_qualification": False,
        },
        "effective_direction_qualification": False,
        "first_error": None,
        "secondary_errors": [],
        "full_parameter_Jacobian_independently_proven": False,
        "body_calls": 0,
        "candidate_adopted": False,
        "F1_F8_complete": False,
        "biology_proven": False,
        "source_sha256": source_map(),
        "receipts": {},
        "work": {},
        "raw_roster": [],
        "external_completion_pending": True,
        "independent_audit_pending": True,
    }
    raw, ledger, record = {}, [], {}
    store = None
    stage = "output"
    closed_watch = False

    def failure(error, at):
        value = {
            "stage": at,
            "type": type(error).__name__,
            "message": str(error)[:1024],
        }
        if terminal["first_error"] is None:
            terminal["first_error"] = value
        else:
            terminal["secondary_errors"].append(value)
        terminal["worker_valid"] = False
        terminal["effective_direction_qualification"] = False

    def resource(*, record_snapshot=True):
        snapshot = meter(device)
        require(
            type(snapshot["tree_rss_bytes"]) is int
            and 0 < snapshot["tree_rss_bytes"] <= (2 * GIB if dimensions else 6 * GIB),
            "actual host memory",
        )
        if record_snapshot:
            terminal["resources"] = snapshot
        if watcher is not None:
            sampled = watcher.snapshot()
            if record_snapshot:
                terminal["gpu_memory_watch"] = sampled
            require(
                not sampled["sampler_errors"]
                and sampled["sample_count"] > 0
                and sampled["sampled_additional_peak_bytes"]
                + sampled["quantization_allowance_bytes"]
                <= 8 * GIB,
                "context-included sampled GPU memory",
            )

    def timely():
        timer.check()
        fault("deadline")

    try:
        store = Store(safe(root, output), timer)
        stage = "precontext"
        if watcher is not None and (
            dimensions is None or watcher.snapshot()["sample_count"] == 0
        ):
            watcher.start()
        stage = "typed-packet"
        packet = packet_loader(ledger, timely)
        timer.check()
        stage = "collect"
        try:
            raw, receipts, record = collect(
                packet, device=device, dimensions=dimensions, timely=timely
            )
        except CollectionFailure as error:
            raw, receipts, record = error.raw, error.receipts, error.record
            terminal["receipts"] = receipts
            raise error.error from error
        terminal["receipts"] = receipts
        terminal["work"] = work_counts(receipts)
        terminal["assembly"] = record
        resource()
        stage = "raw-save"
        fault(stage)
        store.raw(raw)
        terminal["raw_roster"] = sorted(raw)
        stage = "classification"
        fault(stage)
        terminal["analysis"] = analyze(raw, record)
        if expected_imports is not None:
            require(
                all(
                    expected_imports.get(k) == v
                    for k, v in record["actual_import_sha256"].items()
                ),
                "actual runtime import bytes frozen",
            )
        stage = "posthash"
        for entry in ledger:
            require(
                digest(safe(root, entry["path"]), timely) == entry["expected_sha256"],
                "post-calculation selected input hash",
            )
        store.put(
            "selected-ledger.json",
            json_bytes(
                {"schema": "flylab.m3.e342.selected-ledger.v1", "inputs": ledger}
            ),
        )
        resource()
        timer.check()
        terminal["worker_valid"] = True
    except Exception as error:  # noqa: BLE001 -- Preserve first failure with acquired raw and work.
        failure(error, stage)
    finally:
        if (
            raw
            and store is not None
            and not (store.output / "raw.npz").exists()
            and not (store.output / "raw.npz.partial").exists()
        ):
            try:
                store.raw(raw)
                terminal["raw_roster"] = sorted(raw)
            except Exception as error:  # noqa: BLE001 -- Partial publication must not replace first failure.
                failure(error, "partial-raw-save")
        terminal["work"] = work_counts(terminal["receipts"])
        terminal["assembly"] = record
    # Provisional identical result/receipts/stdout. Late errors revoke these bytes.
    stage = "finalization"
    try:
        timer.finalize()
        fault(stage)
        resource()
        terminal["elapsed_before_publication_s"] = clock() - timer.s0
        payload = json_bytes(terminal)
        require(len(payload) <= QUOTAS["stdout"], "actual stdout cap")
        store.put("result.json", payload)
        store.put("receipts.json", payload)
        stage = "manifest"
        manifest = {
            p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
            for p in store.output.iterdir()
            if p.is_file()
        }
        store.put(
            "manifest.json",
            json_bytes(
                {"schema": "flylab.m3.e342.output-manifest.v1", "files": manifest}
            ),
        )
        for name, entry in manifest.items():
            require(
                digest(store.output / name) == entry["sha256"],
                "nonself manifest readback",
            )
        fault("readback")
        timer.check("final")
        resource(record_snapshot=False)
        store.check()
        stage = "stdout"
        printer(payload)
        store.written.add("stdout")
        timer.check("final")
        fault("postpublication")
        resource(record_snapshot=False)
        if watcher is not None:
            terminal_watch = watcher.close()
            closed_watch = True
            require(
                not terminal_watch["sampler_errors"]
                and not terminal_watch["owned_thread_alive"]
                and terminal_watch["sampled_additional_peak_bytes"]
                + terminal_watch["quantization_allowance_bytes"]
                <= 8 * GIB,
                "sampler cleanup/resource",
            )
            store.put(
                "gpu-sampler-completion.json", json_bytes(terminal_watch), failure=True
            )
        timer.check("final")
        store.check()
    except Exception as error:  # noqa: BLE001 -- Late publication/resource failures revoke provisional bytes.
        failure(error, stage)
    finally:
        if watcher is not None and not closed_watch:
            try:
                terminal["gpu_sampler_cleanup"] = watcher.close()
                require(
                    not terminal["gpu_sampler_cleanup"]["sampler_errors"],
                    "sampler cleanup errors",
                )
            except Exception as error:  # noqa: BLE001 -- Preserve own cleanup failure as secondary evidence.
                failure(error, "sampler-close")
    if terminal["first_error"] is not None:
        terminal["analysis"]["direction_qualification"] = False
        terminal["analysis"]["observed_response"] = "not_evaluated"
        terminal["revoked"] = True
        try:
            value = json_bytes(terminal)
            require(len(value) <= FAILURE_QUOTA, "failure bytes")
            # The failure deadline is still absolute; no new clock or success repair.
            store.check(len(value), failure=True)
            with (store.output / "revocation.json").open("xb") as stream:
                stream.write(value)
            require(
                (store.output / "revocation.json").read_bytes() == value,
                "revocation readback",
            )
        except Exception as error:  # noqa: BLE001 -- Emergency failure cannot conceal the first error.
            terminal["secondary_errors"].append(
                {
                    "stage": "emergency-publication",
                    "type": type(error).__name__,
                    "message": str(error)[:1024],
                }
            )
        return terminal, 1
    return terminal, 0


def run_generated(
    root,
    packet,
    *,
    dimensions,
    output="diagnosis",
    device="cpu",
    clock=time.perf_counter,
    fault=lambda stage: None,
    printer=stdout_bytes,
    watcher=None,
    meter=usage,
):
    require(
        isinstance(dimensions, TestDimensions) and device in ("cpu", "cuda"),
        "explicit generated native mode",
    )
    prefix = packet["generated_prefix"]
    target = generated_path(root, prefix + "/" + output)
    require(not target.exists(), "fresh generated output")
    return worker(
        root,
        lambda ledger, timely: load_packet(
            root,
            packet["inventory"],
            dimensions=dimensions,
            metadata=packet["metadata"],
            timely=timely,
            ledger=ledger,
        ),
        output=relative(root, str(target)),
        device=device,
        dimensions=dimensions,
        clock=clock,
        fault=fault,
        printer=printer,
        watcher=watcher,
        meter=meter,
    )


def synthetic_fixture(root, *, dimensions, device="cpu", prefix=None):
    """Fresh generated 23-file packet, with native-produced neutral endpoints.

    Fixture preparation is separately labelled work, never an original read or
    one of the scientific collect forward counts. Every axis override is test-only.
    """
    import uuid

    import numpy as np
    import torch

    require(
        isinstance(dimensions, TestDimensions) and device in ("cpu", "cuda"),
        "generated fixture arguments",
    )
    require(
        dimensions.edges == dimensions.selected == 8
        and dimensions.groups == dimensions.population == 4,
        "this small fixture uses eight existing edges/four groups",
    )
    a = axes(dimensions)
    prefix = (
        BASE + "/e342-sol-fixture-" + uuid.uuid4().hex if prefix is None else prefix
    )
    directory = generated_path(root, prefix)
    require(not directory.exists(), "fresh fixture directory")
    directory.mkdir(parents=True)
    n = a["nodes"]
    ids = np.arange(391, dtype=np.int64)
    membership = ids % 96
    pool = np.zeros((96, 391), np.float64)
    pool[membership, ids] = 1 / np.bincount(membership, minlength=96)[membership]
    slopes = np.zeros((96, 391, 3, 2), np.float64)
    slopes[:, :, 1, :] = pool[:, :, None]
    calibration = {
        "moment": 0.2
        * np.sin(np.arange(42 * 96, dtype=np.float64).reshape(42, 96) * 0.17),
        "rest": np.zeros(42, np.float64),
        "pool": pool,
        "motor_ids": ids,
        "ltm": np.arange(6, dtype=np.int64),
    }
    parameters = {
        "tau": np.array([0.005, 0.01, 0.04], np.float64),
        "kappa": np.ones(391, np.float64),
        "slopes": slopes,
        "membership": membership,
    }
    phase = 2 * np.pi * np.arange(128, dtype=np.float64) / 128
    curves = np.empty((6, 128, 7), np.float64)
    for leg in range(6):
        for joint in range(7):
            curves[leg, :, joint] = (0.12 + 0.01 * joint) * np.sin(
                phase + leg * 0.4 + joint * 0.7
            )
    from tools.banc_m3_smooth_temporal import SmoothTemporalObjective

    objective = SmoothTemporalObjective(
        curves, np.zeros((99, 42), np.float64), np.zeros((99, 6), np.float64)
    )
    data = {
        "reference": objective.canonical_constants,
        "actual_csr": {
            "indptr": np.r_[0, 2, 4, 6, np.full(n - 3, 8)].astype(np.int32),
            "indices": np.tile(np.array([0, 1], np.int32), 4),
            "data": np.tile(np.array([0.01, -0.02], np.float32), 4),
        },
        "motor_configuration": {
            "cell_gain": np.full(n, 8, np.float32),
            "membrane_tau": np.full(n, 0.01, np.float32),
            "motor_ids": ids,
            "actual_motor_inverse32": np.full(391, np.float32(1 / 0.3), np.float32),
            "actual_motor_gain32": np.full(391, 8, np.float32),
        },
        "load_parameters": {
            "threshold32": np.full(n, 7.5, np.float32),
            "tau32": np.full(n, 0.01, np.float32),
        },
        "recurrent_mask": {
            "population": np.arange(4, dtype=np.int64),
            "positions": np.arange(8, dtype=np.int64),
            "edge_parameters": np.tile(np.arange(4, dtype=np.int64), 2),
        },
        "e289_roster": {
            "premotor_ids": np.arange(4, dtype=np.int64),
            "edge_ids": np.arange(8, dtype=np.int64),
            "weights32": np.tile(np.array([0.01, -0.02], np.float32), 4),
        },
        "calibration": calibration,
        "parameters": parameters,
        "inverse": {"value": np.full(n, np.float32(1 / 0.3), np.float32)},
        "output_mask": {"value": np.ones(n, np.float32)},
        "control_raw": {
            "command": np.zeros((301, 42), np.float64),
            "tendon": np.zeros((301, 6), np.float64),
            "control_tick": np.arange(301, dtype=np.int64),
            "network_tick": np.arange(301, dtype=np.int64) * 20,
        },
    }
    meta = {
        "neutral": calibration["rest"].tolist(),
        "model_identity": "generated-native-only",
        "decoder_hash": "generated-native-only",
        "decoder_links": {},
    }
    for role in ("calibration", "parameters"):
        for key, value in data[role].items():
            meta["decoder_links"][role + "." + key] = {
                "original_pointer": prefix + "/generated-" + role + "-" + key + ".npy",
                "original_npy_sha256": npy_file_sha(value),
            }
    for index, c in enumerate(WINDOWS):
        state = {
            "rate": np.linspace(1, 2, n, dtype=np.float32) + np.float32(index * 0.03),
            "adaptation": np.full(n, 0.2, np.float32),
            "bank": np.full((391, 3), 0.3, np.float64),
            "original_activation": np.full(96, 0.3, np.float64),
            "fixed_original_q_boundary": np.zeros(42, np.float64),
            "source_control_tick": np.array(c, np.int64),
            "source_neural_row_index": np.array(c - 1, np.int64),
            "decoder_replay_boundary_index": np.array(c, np.int64),
        }
        data[f"start-{c}"] = state
        data[f"endpoint-{c}"] = {
            **{
                k: v.copy()
                for k, v in state.items()
                if k
                in (
                    "rate",
                    "adaptation",
                    "bank",
                    "original_activation",
                    "fixed_original_q_boundary",
                )
            },
            "q": np.zeros((4, 42), np.float64),
            "tendon": np.zeros((4, 6), np.float64),
            "loss_terms": np.zeros(4, np.float64),
        }
        first, begin, end = SLICES[index]
        full = np.asarray(
            20
            + np.arange(25)[:, None] * 0.05
            + np.linspace(0, 3, n)[None]
            + index * 0.2,
            np.float32,
        )
        data[f"drive-{c}"] = {"drive": full[begin:end].copy()}
    packet = {**data, "metadata": meta}
    bundle = assemble(packet, device=device, dimensions=dimensions)
    rollout, target = bundle["rollout"], bundle["rollout"].device
    tensor = lambda value: torch.as_tensor(value.copy(), device=target)
    preparation = {}
    for index, c in enumerate(WINDOWS):
        state = data[f"start-{c}"]
        with torch.no_grad():
            result = rollout.run(
                *(
                    tensor(state[k])
                    for k in ("rate", "adaptation", "bank", "original_activation")
                ),
                tensor(data[f"drive-{c}"]["drive"]),
                torch.ones(4, dtype=torch.float64, device=target),
                tensor(data["output_mask"]["value"]),
            )
            result["receipt"].confirm()
            for key in (
                "q",
                "tendon",
                "rate",
                "adaptation",
                "bank",
                "original_activation",
            ):
                data[f"endpoint-{c}"][key] = result[key].detach().cpu().numpy().copy()
            terms = window_loss(
                bundle["objective"],
                tensor(state["fixed_original_q_boundary"]),
                result["q"],
            )
            data[f"endpoint-{c}"]["loss_terms"] = loss_vector(terms)[[2, 0, 1, 3]]
            preparation[str(c)] = receipt_dict(result["receipt"])
        first, begin, end = SLICES[index]
        full = np.asarray(
            20
            + np.arange(25)[:, None] * 0.05
            + np.linspace(0, 3, n)[None]
            + index * 0.2,
            np.float32,
        )
        data[f"drive-{c}"] = {
            "drive": full,
            "control_tick": np.arange(first, first + 25, dtype=np.int64),
            "network_tick": np.arange(first, first + 25, dtype=np.int64) * 20,
        }
    order = [
        "reference",
        "actual_csr",
        "motor_configuration",
        "load_parameters",
        "recurrent_mask",
        "e289_roster",
        "calibration",
        "parameters",
    ]
    order += [
        role for c in WINDOWS for role in (f"drive-{c}", f"start-{c}", f"endpoint-{c}")
    ]
    order += ["control_raw", "inverse", "output_mask"]
    bindings = {}
    for role in order:
        arrays = data[role]
        kind = "npy" if role in ("inverse", "output_mask") else "npz"
        path = directory / (role + "." + kind)
        if kind == "npy":
            np.save(path, arrays["value"], allow_pickle=False)
        else:
            extras = (
                {"source_weights": np.array([999], np.float32)}
                if role == "recurrent_mask"
                else {}
            )
            np.savez(path, **arrays, **extras)
        bindings[role] = {
            "path": relative(root, str(path)),
            "sha256": digest(path),
            "bytes": path.stat().st_size,
            "kind": kind,
            "selected_schema": {
                k: {"shape": list(v.shape), "dtype": v.dtype.str, "finite": True}
                for k, v in arrays.items()
            },
        }
    return {
        "generated_prefix": prefix,
        "metadata": meta,
        "inventory": {
            "bindings": bindings,
            "role_order": order,
            "input_count": 23,
            "selected_value_count": 112,
        },
        "fixture_preparation_receipts": preparation,
        "actual_import_sha256": bundle["record"]["actual_import_sha256"],
        "prohibited_old_calls": bundle["record"]["prohibited_old_calls"],
    }


def run(freeze_sha256, review_sha256):
    s0 = _IMPORT_S0
    timer = Clock(s0=s0)
    freeze = gate(freeze_sha256, review_sha256, timely=lambda: timer.check())
    watcher = GPUMemoryWatch()
    return worker(
        ROOT,
        lambda ledger, timely: load_packet(
            ROOT, freeze["inventory"], timely=timely, ledger=ledger
        ),
        output=OUTPUT,
        device="cuda",
        expected_imports=freeze["actual_import_sha256"],
        watcher=watcher,
        s0=s0,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review-sha256", required=True)
    options = parser.parse_args(argv)
    _terminal, code = run(options.freeze_sha256, options.review_sha256)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
