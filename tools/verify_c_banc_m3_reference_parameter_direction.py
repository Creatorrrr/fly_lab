"""E356 fresh reference-loss parameter direction; passive import, original gates."""

# ruff: noqa: E402
# ENTRY must precede library imports to include startup in the absolute deadline.

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
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
BASE = "verification/banc-natural-gait-m3-20260919"
OWNED = (
    "tools/verify_c_banc_m3_reference_parameter_direction.py",
    "tests/test_c_banc_reference_parameter_direction.py",
)
REGISTRATION = BASE + "/parent-e357-generated-registration-v2.json"
REGISTRATION_SHA = "4877de8d3f579edfec8fa2d3ef9c0e7a330862cdad2caaa0254ec6c4481c95d9"
REFLECTION = BASE + "/parent-e357-astra-reflection-v2.json"
REFLECTION_SHA = "a1cb7f8ccdc9efc829eac4d27962558d9511ec78d0ac6d6ed3996eb4902755b9"
CORE = "tools/verify_c_banc_m3_coordination_parameter_direction.py"
CORE_SHA = "fb4d2687baf9ad89b919b7c4692d7039992a319ffef082686933d1bbc563bb2d"
ADMIN = {
    BASE
    + "/parent-e356-root-exit-witness.py": "4ca99cf6859391880d81df2062a71de725b1d7a915231b449289d4e284ff900f",
    BASE
    + "/parent-e356-root-normalize-full-return.py": "054c90615f9d561957a8b4bd5506ce65555d18f631e777362fbbdadb9c29526f",
}
OUTPUT = BASE + "/e356-reference-parameter-direction"
AUDIT_OUTPUT = BASE + "/e356-reference-parameter-direction-audit"
WINDOWS = (100, 164, 228, 292)
_NATIVE = None


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def safe(name, root=ROOT):
    require(
        type(name) is str and "\\" not in name and ":" not in name,
        "literal relative path",
    )
    parts = PurePosixPath(name).parts
    require(
        parts
        and not name.startswith("/")
        and ".." not in parts
        and "." not in parts
        and PurePosixPath(name).as_posix() == name,
        "canonical relative path",
    )
    path = root / name
    require(
        path.resolve().is_relative_to(root.resolve()) and not path.is_symlink(),
        "root containment",
    )
    return path


def strict_json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def number(value):
        result = float(value)
        require(math.isfinite(result), "finite JSON number")
        return result

    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_float=number,
        parse_constant=lambda value: require(False, "JSON nonfinite " + value),
    )


def lower(value):
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        require(math.isfinite(value), "finite serializer")
        return value
    if type(value) is list:
        return [lower(v) for v in value]
    if type(value) is dict:
        require(all(type(k) is str for k in value), "string JSON keys")
        return {k: lower(v) for k, v in value.items()}
    raise TypeError("unsupported strict JSON type: " + type(value).__name__)


def report_value(value):
    if type(value) is tuple:
        return [report_value(v) for v in value]
    if type(value) is list:
        return [report_value(v) for v in value]
    if type(value) is dict:
        return {k: report_value(v) for k, v in value.items()}
    return lower(value)


def json_bytes(value):
    return (json.dumps(lower(value), indent=2, allow_nan=False) + "\n").encode("utf8")


def descriptor(name, raw):
    return {"path": name, "bytes": len(raw), "sha256": sha(raw)}


def own_sources(root=ROOT):
    return {name: sha(safe(name, root).read_bytes()) for name in OWNED}


def read_descriptor(spec, root=ROOT):
    require(
        type(spec) is dict
        and set(spec) == {"path", "bytes", "sha256"}
        and type(spec["bytes"]) is int,
        "literal descriptor",
    )
    require(
        Path(spec["path"]).suffix in (".py", ".json", ".ini", ".txt", ".md"),
        "text authority only",
    )
    raw = safe(spec["path"], root).read_bytes()
    require(len(raw) == spec["bytes"] and sha(raw) == spec["sha256"], "descriptor pin")
    return raw


def registry(root=ROOT):
    raw = safe(REGISTRATION, root).read_bytes()
    require(sha(raw) == REGISTRATION_SHA, "reflected registration pin")
    return strict_json(raw)


def metadata_preflight(root=ROOT, timely=lambda: None):
    reg = registry(root)
    require(
        sha(safe(REFLECTION, root).read_bytes()) == REFLECTION_SHA, "current reflection"
    )
    pins = {
        **reg["protected_source_sha256"],
        **reg["inherited_metadata_sha256"],
        **ADMIN,
    }
    for name, pin in pins.items():
        timely()
        require(
            Path(name).suffix in (".py", ".json", ".ini", ".txt", ".md"),
            "finite metadata/source scope",
        )
        require(
            sha(safe(name, root).read_bytes()) == pin, "protected current pin " + name
        )
    for key in (
        "design",
        "actual_review",
        "review_md",
        "empty_pytest_ini",
        "ROOT_closed_form_fixture",
    ):
        read_descriptor(reg[key], root)
    require(
        reg["M1_M2_M3_reflected"] is True and reg["original_authorized"] is False,
        "reflected authority only",
    )
    return reg


def native():
    global _NATIVE
    if _NATIVE is None:
        require(sha(safe(CORE).read_bytes()) == CORE_SHA, "pinned native primitive")
        spec = importlib.util.spec_from_file_location("e356_pinned_native", safe(CORE))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _NATIVE = module
    return _NATIVE


def array_sha(value):
    return sha(value.tobytes(order="C"))


def validate_target(target, normalization):
    import numpy as np

    require(
        target.dtype == np.float64
        and target.shape == (4, 4, 42)
        and target.flags.c_contiguous
        and np.isfinite(target).all(),
        "literal target",
    )
    require(
        normalization.dtype == np.float64
        and normalization.shape == (6,)
        and np.isfinite(normalization).all()
        and (normalization > 0).all(),
        "six positive normalization",
    )


def align_label(label):
    import numpy as np

    require(
        set(label) == {"command", "control_tick", "control_time_s"},
        "only three label members",
    )
    q, tick, clock = (label[k] for k in ("command", "control_tick", "control_time_s"))
    require(
        q.dtype == np.float64
        and q.shape == (301, 42)
        and np.isfinite(q).all()
        and q.flags.c_contiguous,
        "literal label commands",
    )
    require(
        tick.dtype == np.int64
        and tick.shape == (301,)
        and np.array_equal(tick, np.arange(301, dtype=np.int64)),
        "label tick",
    )
    require(
        clock.dtype == np.float64
        and clock.shape == (301,)
        and np.array_equal(clock, tick * 0.005),
        "label time",
    )
    return np.stack([q[c + 1 : c + 5].copy() for c in WINDOWS])


def load_label(binding, ledger, root=ROOT, timely=lambda: None):
    import numpy as np

    path = safe(binding["path"], root)
    timely()
    raw = path.read_bytes()
    require(
        len(raw) == binding["bytes"] and sha(raw) == binding["sha256"],
        "once label archive digest",
    )
    selected = binding["selected_members"]
    require(
        set(selected) == {"command", "control_tick", "control_time_s"},
        "label selected schema",
    )
    observation = {
        "path": binding["path"],
        "sha256": sha(raw),
        "numeric_reads": [],
        "selected": {},
        "selected_header_reads": [],
        "extras_payload_header_reads": 0,
    }
    ledger.append(observation)
    result = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        require(
            len(archive.namelist()) == len(set(archive.namelist())),
            "duplicate label member",
        )
        for key, spec in selected.items():
            timely()
            info = archive.getinfo(spec["member"])
            require(
                info.file_size == spec["uncompressed_bytes"]
                and info.compress_size == spec["compressed_bytes"]
                and info.CRC == spec["crc32"]
                and info.compress_type == spec["zip_compression"],
                "literal label ZIP metadata",
            )
            data = archive.read(info)
            stream = io.BytesIO(data)
            header = native().strict_npy_header(stream)
            require(
                header["shape"] == spec["shape"]
                and header["dtype"] == native().dtype_metadata(spec["dtype"])
                and header["fortran_order"] is False
                and header["header_bytes"] == spec["header_bytes"],
                "literal selected label header",
            )
            value = np.load(io.BytesIO(data), allow_pickle=False)
            require(
                value.shape == tuple(spec["shape"])
                and value.dtype.str == native().dtype_metadata(spec["dtype"])
                and value.flags.c_contiguous
                and np.isfinite(value).all(),
                "finite label payload",
            )
            result[key] = value
            observation["numeric_reads"].append(key)
            observation["selected_header_reads"].append(key)
            observation["selected"][key] = {
                "c_order_sha256": array_sha(value),
                "dtype": value.dtype.str,
                "shape": list(value.shape),
            }
    target = align_label(result)
    observation["slice_c_order_sha256"] = array_sha(target)
    return target


def make_direction(gradient):
    import numpy as np

    require(
        gradient.dtype == np.float64
        and gradient.ndim == 1
        and np.isfinite(gradient).all(),
        "finite fresh R gradient",
    )
    magnitude = float(np.max(np.abs(gradient)))
    require(magnitude > 0, "nonzero R gradient")
    requested = 1 - 0.001 * gradient / magnitude
    actual = requested.astype(np.float32).astype(np.float64)
    require((actual > 0).all(), "positive actual theta")
    return requested, actual


def validate_raw(raw, branch, *, dimensions=None):
    import numpy as np

    contract = registry()["raw_contract"]
    require(
        branch
        in (
            "normal",
            "zero_R_gradient",
            "quantized_zero_direction",
            "native_palette_not_deployable",
            "technical_partial",
        ),
        "known raw branch",
    )
    roster = set(contract["normal"])
    require(
        set(raw) <= roster
        if branch == "technical_partial"
        else set(raw) == set(contract[branch]),
        "exact acquired raw roster",
    )
    if dimensions is not None:
        require(
            isinstance(dimensions, native().TestDimensions),
            "explicit generated raw axes",
        )
    if "window_starts" in raw:
        require(
            np.array_equal(raw["window_starts"], np.asarray(WINDOWS, np.int64)),
            "literal four raw window starts",
        )
    if "normalization" in raw:
        require(
            raw["normalization"].shape == (6,) and np.all(raw["normalization"] > 0),
            "positive six raw normalization",
        )
    for key, value in raw.items():
        spec = contract["fields"][key]
        shape = list(spec["shape"])
        if dimensions is not None:
            shape = [
                {
                    158706: dimensions.nodes,
                    8547: dimensions.groups,
                    725849: dimensions.selected,
                }.get(v, v)
                for v in shape
            ]
        if branch == "technical_partial" and spec["partial_window_prefix"]:
            require(0 <= value.shape[0] <= 4, "actual partial prefix")
            shape[0] = value.shape[0]
        require(
            value.dtype.str == spec["dtype"]
            and value.shape == tuple(shape)
            and value.flags.c_contiguous
            and np.isfinite(value).all(),
            "raw literal axes " + key,
        )
        require(value.nbytes + 256 <= 4 << 20, "raw member cap")
    return True


def analyze(raw, record):
    import numpy as np

    branch = record["branch"]
    require(
        np.array_equal(raw["window_starts"], np.asarray(WINDOWS, np.int64)),
        "literal analysis window starts",
    )
    require(
        raw["normalization"].dtype == np.float64
        and raw["normalization"].shape == (6,)
        and np.isfinite(raw["normalization"]).all()
        and np.all(raw["normalization"] > 0),
        "analysis positive six normalization",
    )
    answer = {
        "branch": branch,
        "H356": "not_evaluated",
        "candidate_runtime_adoption": False,
        "natural_gait": False,
        "BANC_autonomy": False,
        "F1_F8_complete": False,
        "biology": False,
        "full_parameter_Jacobian_independently_proven": False,
        "independent_original_deployability_recomputed": False,
        "native_quantization_derivative_proven": False,
        "O": None,
        "Q": None,
        "Q_exact": None,
        "T": None,
    }
    norm = np.repeat(raw["normalization"], 7)
    R = np.sum((raw["q_neutral"] - raw["target_q"]) ** 2 / norm, axis=(1, 2)) / 672
    gq = (raw["q_neutral"] - raw["target_q"]) / (336 * norm)
    require(
        np.allclose(R, raw["R_neutral"], rtol=1e-10, atol=1e-12)
        and np.allclose(gq, raw["gq_R"], rtol=1e-10, atol=1e-12),
        "independent R/direct q arithmetic",
    )
    require(
        np.allclose(
            raw["sum_gtheta_R"], raw["gtheta_R"].sum(axis=0), rtol=1e-10, atol=1e-12
        ),
        "fresh gradient sum",
    )
    if branch != "zero_R_gradient":
        requested, actual = make_direction(raw["sum_gtheta_R"])
        require(
            np.array_equal(requested, raw["theta_requested"])
            and np.array_equal(actual, raw["theta_actual"]),
            "actual rounded theta",
        )
    if branch != "normal":
        answer["H356"] = "inconclusive"
        answer["observed_response_evaluated"] = False
        return answer
    delta = raw["q_candidate"] - raw["q_neutral"]
    candidate = (
        np.sum((raw["q_candidate"] - raw["target_q"]) ** 2 / norm, axis=(1, 2)) / 672
    )
    require(
        np.allclose(candidate, raw["R_candidate"], rtol=1e-10, atol=1e-12),
        "candidate R",
    )
    observed = candidate - R
    Q = np.einsum("wij,wij->w", gq, delta)
    exact = Q + np.sum(delta**2 / norm, axis=(1, 2)) / 672
    T = raw["gtheta_R"] @ (raw["theta_actual"] - 1)
    require(np.allclose(observed, exact, rtol=1e-10, atol=1e-12), "quadratic identity")
    require(
        np.isclose(
            T.sum(),
            raw["sum_gtheta_R"] @ (raw["theta_actual"] - 1),
            rtol=1e-10,
            atol=1e-12,
        ),
        "parameter dot identity",
    )
    oq, ot = native().reconciliation(observed, Q), native().reconciliation(observed, T)
    supported = (
        -float(observed.sum()) > 1e-9
        and -float(T.sum()) > 1e-9
        and oq["passed"]
        and ot["passed"]
    )
    meaningful = oq["meaningful"] and ot["meaningful"]
    answer.update(
        O=observed.tolist(),
        Q=Q.tolist(),
        Q_exact=exact.tolist(),
        T=T.tolist(),
        I_R=-float(observed.sum()),
        O_Q=oq,
        O_T=ot,
        H356="conditional_support"
        if supported
        else ("refuted" if meaningful else "inconclusive"),
        proxy_delta=(raw["candidate_proxy_loss"] - raw["neutral_proxy_loss"]).tolist(),
        observed_response_evaluated=True,
    )
    return answer


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
    file = safe(name, root)
    require(
        file.parent == safe(BASE, root)
        and file.name.startswith(
            ("e356-sol-FIRST-", "e356-sol-R1-", "parent-e356-root-")
        )
        and file.suffix == ".json",
        "controlled case contract path",
    )
    raw = file.read_bytes()
    c = strict_json(raw)
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
        c["schema"] == "flylab.m3.e356.pytest-case-contract.v1"
        and c["source_sha256"] == own,
        "current case contract sources",
    )
    reg_raw = safe(REGISTRATION, root).read_bytes()
    require(sha(reg_raw) == REGISTRATION_SHA, "case authority registration pin")
    reg = strict_json(reg_raw)
    current = ast_cases(safe(OWNED[1], root))
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
            and path.startswith(BASE + "/parent-e356-root-")
            and path.endswith(".py")
            and path not in expected,
            "controlled ROOT extra tests",
        )
        value = safe(path, root).read_bytes()
        require(
            len(value) == s["bytes"] and sha(value) == s["sha256"],
            "current registered extra source",
        )
        expected[path] = ast_cases(safe(path, root))
    require(
        c["complete_suite"] == list(expected) and c["expected_functions"] == expected,
        "complete current AST suite only",
    )
    ini = c["empty_ini"]
    require(ini == reg["empty_pytest_ini"], "pinned empty ini registration")
    ini_raw = safe(ini["path"], root).read_bytes()
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
        and temp.name.startswith(("e356-sol-FIRST-", "e356-sol-R1-", "e356-root-"))
        and ".." not in temp.parts,
        "owned external pytest temp",
    )
    receipt = safe(c["receipt_output"], root)
    require(
        receipt.parent == safe(BASE, root)
        and receipt.name.startswith(
            ("e356-sol-FIRST-", "e356-sol-R1-", "parent-e356-root-")
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


def collect(
    packet,
    *,
    target_q,
    device="cuda",
    dimensions=None,
    timely=lambda: None,
    progress=lambda value: None,
):
    import numpy as np
    import torch

    core = native()
    axes, assemble, receipt_dict = core.axes, core.assemble, core.receipt_dict
    work_counts, window_loss, loss_vector = (
        core.work_counts,
        core.window_loss,
        core.loss_vector,
    )
    patch_weights, exact = core.patch_weights, core.exact
    CollectionFailure = core.CollectionFailure
    a = axes(dimensions)
    raw = {
        "normalization": packet["reference"]["normalization"].copy(),
        "window_starts": np.asarray(WINDOWS, np.int64),
        "target_q": target_q.copy(),
    }
    validate_target(target_q, raw["normalization"])
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

    def emit(event="progress"):
        progress(
            {
                "raw": raw,
                "receipts": receipts,
                "record": record,
                "stage": stage,
                "event": event,
            }
        )

    try:
        timely()
        emit("begin")
        bundle = assemble(packet, device=device, dimensions=dimensions)
        rollout, objective = bundle["rollout"], bundle["objective"]
        record = bundle["record"]
        target = rollout.device
        emit("end")

        def tensor(value):
            return torch.as_tensor(value.copy(), device=target)

        mask = tensor(packet["output_mask"]["value"])
        targets = tensor(target_q).clone().detach()
        norm = tensor(np.repeat(raw["normalization"], 7)).clone().detach()
        record["neural_backward_requested"] = 0
        record["neural_backward_returned"] = 0

        def forward(c, arm, theta):
            nonlocal stage
            stage = f"{arm}-{c}-forward"
            key = f"window-{c}-{arm}"
            state = packet[f"start-{c}"]
            timely()
            emit("begin")
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
                emit("returned")
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
            if arm in ("neutral", "candidate"):
                for name in ("rate", "adaptation", "bank", "original_activation"):
                    save(arm + "_end_" + name, result[name].detach().cpu().numpy())
            emit()
            try:
                receipt.confirm()
            finally:
                receipts[key] = receipt_dict(receipt)
                emit("confirmed")
            timely()
            return result, key

        for index, c in enumerate(WINDOWS):
            theta = torch.ones(
                a["groups"], dtype=torch.float64, device=target, requires_grad=True
            )
            result, key = forward(c, "neutral", theta)
            endpoint = packet[f"endpoint-{c}"]
            stage = f"neutral-{c}-reproduction"
            emit("begin")
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
            save("neutral_proxy_loss", losses)
            # E333 input order is total,S,C,LR; our raw is S,C,total,LR.
            require(
                np.allclose(
                    losses[[2, 0, 1, 3]], endpoint["loss_terms"], rtol=1e-10, atol=1e-11
                ),
                "saved neutral loss order/reproduction",
            )
            ell = ((result["q"] - targets[index]).square() / norm).sum() / (16 * 42)
            save("R_neutral", ell.detach().cpu().numpy())
            graphs.append((result, theta, ell, key))
            emit("end")
        require(
            len(graphs) == 4
            and work_counts(receipts)["backward_blocks_requested"] == 0,
            "all neutral before new gradient",
        )
        record["all_neutral_reproduced_before_gradient"] = True
        for result, theta, ell, key in graphs:
            stage = key + "-R-backward"
            timely()
            receipt = result["receipt"]
            before = receipt_dict(receipt)
            (gq,) = torch.autograd.grad(ell, result["q"], retain_graph=True)
            require(
                before == receipt_dict(receipt), "gq extraction performed neural work"
            )
            save("gq_R", gq.detach().cpu().numpy())
            index = len(raw["gq_R"]) - 1
            require(
                np.allclose(
                    raw["gq_R"][index],
                    2
                    * (raw["q_neutral"][index] - target_q[index])
                    / (16 * 42 * np.repeat(raw["normalization"], 7)),
                    rtol=1e-10,
                    atol=1e-12,
                ),
                "fresh actual direct-q closed form",
            )
            require(
                theta.grad is None and theta.is_leaf, "independent fresh theta leaf"
            )
            emit()
            try:
                record["neural_backward_requested"] += 1
                emit("begin")
                ell.backward()
                record["neural_backward_returned"] += 1
                emit("returned")
            finally:
                receipts[key] = receipt_dict(receipt)
                emit()
            require(theta.grad is not None, "R-only theta VJP missing")
            save("gtheta_R", theta.grad.detach().cpu().numpy())
            emit()
            try:
                receipt.confirm()
            finally:
                receipts[key] = receipt_dict(receipt)
                emit("confirmed")
            timely()
        raw["sum_gtheta_R"] = raw["gtheta_R"].sum(axis=0)
        require(
            all(np.isfinite(v).all() for v in raw.values()), "finite neutral/VJP raw"
        )
        magnitude = float(np.max(np.abs(raw["sum_gtheta_R"])))
        if magnitude == 0:
            record["branch"] = "zero_R_gradient"
        else:
            raw["theta_requested"], raw["theta_actual"] = make_direction(
                raw["sum_gtheta_R"]
            )
            emit()
            if exact(raw["theta_actual"], np.ones(a["groups"], np.float64)):
                record["branch"] = "quantized_zero_direction"
            else:
                raw["selected_weights_candidate"], palette = patch_weights(
                    packet["actual_csr"], packet["recurrent_mask"], raw["theta_actual"]
                )
                record["palette"] = palette
                emit()
                if not palette["native_deployable"]:
                    record["branch"] = "native_palette_not_deployable"
                else:
                    record["branch"] = "normal"
                    theta = tensor(raw["theta_actual"])
                    for index, c in enumerate(WINDOWS):
                        with torch.no_grad():
                            result, _ = forward(c, "candidate", theta)
                            terms = window_loss(
                                objective, tensor(raw["boundary"][index]), result["q"]
                            )
                            save("candidate_proxy_loss", loss_vector(terms))
                            ell = (
                                (result["q"] - targets[index]).square() / norm
                            ).sum() / (16 * 42)
                            save("R_candidate", ell.detach().cpu().numpy())
                            emit()
        stage = "completed-raw"
        emit()
        require(all(np.isfinite(v).all() for v in raw.values()), "finite response raw")
        counts = work_counts(receipts)
        forward_controls = 32 if record["branch"] == "normal" else 16
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
        validate_raw(raw, record["branch"], dimensions=dimensions)
        return raw, receipts, record
    except Exception as error:
        raise CollectionFailure(error, raw, receipts, record, stage) from error


def make_freeze(qualification, *, root=ROOT):
    reg = metadata_preflight(root)
    raw = safe(qualification, root).read_bytes()
    q = strict_json(raw)
    own = own_sources(root)
    require(
        q["schema"] == "flylab.m3.e356.actual-full-return.v1"
        and q["source_sha256"] == own
        and type(q["original_calls"]) is int
        and q["original_calls"] == 0,
        "current generated full-return qualification",
    )
    require(
        type(q["exit_code"]) is int
        and q["exit_code"] == 0
        and q["all_owned_ended"] is True
        and q["known_identity_cleanup"] == []
        and q["first_error"] is None
        and q["secondary_errors"] == []
        and q["charged_once"] is True,
        "actual natural full exit",
    )
    whole_outer_s = reg.get("e357_generated_whole_outer_s", 40)
    require(
        type(whole_outer_s) in (int, float)
        and math.isfinite(whole_outer_s)
        and whole_outer_s == (65 if "e357_generated_whole_outer_s" in reg else 40),
        "strict prospective generated whole wall registration",
    )
    require(
        type(q["actual_outer_full_return_s"]) in (float, int)
        and math.isfinite(q["actual_outer_full_return_s"])
        and 0 < q["actual_outer_full_return_s"] <= whole_outer_s,
        "qualification actual wall",
    )
    linked = q["pytest_case_execution"]
    case, spec = validate_case_contract(linked["case_contract"]["path"], root=root)
    require(spec == linked["case_contract"], "linked current case")
    receipt = strict_json(read_descriptor(linked["case_receipt"], root))
    require(
        receipt["source_sha256"] == own
        and receipt["case_contract"] == spec
        and receipt["complete_current_cases"] is True,
        "same actual current suite",
    )
    complete_case_receipt(receipt, case["expected_functions"])
    if whole_outer_s == 65:
        require(
            q["registration_sha256"] == REGISTRATION_SHA
            and receipt["registration_sha256"] == REGISTRATION_SHA
            and receipt["whole_sampler_scope"] == "before_pytest_import_and_first_CUDA"
            and receipt["whole_sampler_started_before_pytest_import"] is True,
            "current E357 whole authority/precontext scope",
        )
        generated_sampler_check(receipt["whole_sampler_start"])
        generated_sampler_check(receipt["whole_sampler_closed"], closed=True)
    if reg.get("e357_fixture_revision") == 2:
        require(
            q["outer_wall_cap_s"] == 36 and q["actual_outer_full_return_s"] <= 36,
            "current v2-specific actual full outer36",
        )
    c = strict_json(read_descriptor(q["contract"], root))
    if reg.get("e357_fixture_revision") == 2:
        require(
            type(c["e357_outer_wall_s"]) is int
            and c["e357_outer_wall_s"] == 36
            and type(c["commands"][0]["wall_s"]) is int
            and c["commands"][0]["wall_s"] == 33,
            "current v2-specific outer36 child33 contract",
        )
    if whole_outer_s == 65:
        require(
            c["source_sha256"].get(REGISTRATION) == REGISTRATION_SHA,
            "current registration in generated whole closure",
        )
    require(
        c["original_calls"] == 0
        and c["commands"][0]["argv"]
        == [
            ".venv/Scripts/python.exe",
            "-X",
            "utf8",
            "-B",
            OWNED[1],
            "--execute-whole-current-pytest",
            spec["path"],
        ],
        "actual sole pytest entrypoint",
    )
    require(
        all(c["source_sha256"].get(k) == v for k, v in {**own, **ADMIN}.items()),
        "qualification current closure pins",
    )
    for name, pin in c["source_sha256"].items():
        require(
            Path(name).suffix in (".py", ".json", ".ini", ".txt", ".md")
            and sha(safe(name, root).read_bytes()) == pin,
            "qualification closure pin",
        )
    s = strict_json(read_descriptor(q["supervision"], root))
    identity = strict_json(read_descriptor(q["identity"], root))
    outer = strict_json(read_descriptor(q["outer_witness"], root))
    tool = strict_json(read_descriptor(q["actual_tool_return"], root))
    require(
        s["success"] is True
        and len(s["attempts"]) == 1
        and s["attempts"][0]["exit_code"] == 0
        and s["attempts"][0]["argv"] == c["commands"][0]["argv"]
        and s["contract_sha256"] == q["contract"]["sha256"],
        "actual E267 linkage",
    )
    require(
        identity["qualified"] is True
        and identity["all_owned_pids_ended"] is True
        and identity["errors"] == []
        and identity["contract"] == q["contract"]
        and identity["supervision"] == q["supervision"],
        "actual E346 identity linkage",
    )
    require(
        outer["all_owned_ended"] is True
        and outer["contract"] == q["contract"]
        and outer["exit_code"] == 0
        and outer["first_error"] is None
        and outer["secondary_errors"] == []
        and outer["known_identity_cleanup"] == [],
        "ROOT witness linkage",
    )
    require(
        tool["actual_returned_exit_code"] == 0
        and tool["actual_outer_full_return_s"] == q["actual_outer_full_return_s"],
        "actual tool-return linkage",
    )
    return {
        "schema": "flylab.m3.e356.reference-direction-freeze.v1",
        "source_sha256": own,
        "registration_sha256": REGISTRATION_SHA,
        "reflection_sha256": REFLECTION_SHA,
        "admin_source_sha256": ADMIN,
        "qualification": descriptor(qualification, raw),
        "original_authorized": False,
        "legacy_inventory": reg["legacy_inventory"],
        "cadence_label": reg["cadence_label"],
        "raw_contract": reg["raw_contract"],
    }


def original_argv(freeze, review, role="worker"):
    require(role in ("worker", "audit"), "literal original role")
    return [
        ".venv/Scripts/python.exe",
        "-X",
        "utf8",
        "-B",
        OWNED[0],
        "--run" if role == "worker" else "--audit",
        "--freeze",
        freeze,
        "--review",
        review,
        "--output",
        OUTPUT if role == "worker" else AUDIT_OUTPUT,
    ]


def contract_paths(name, root=ROOT):
    file = safe(name, root)
    require(
        file.parent == safe(BASE, root)
        and file.name.startswith("parent-e346-e356-")
        and file.suffix == ".json",
        "E346 E356 contract namespace",
    )
    stem = name.removesuffix(".json")
    prefix = BASE + "/parent-e356-" + file.stem.removeprefix("parent-e346-e356-")
    sup = stem + "-supervision"
    return {
        "supervision": sup + ".json",
        "identities": sup + "-identities.json",
        "observer_stdout": sup + "-observer.stdout.txt",
        "observer_stderr": sup + "-observer.stderr.txt",
        "stdout": prefix + "-worker.stdout.txt",
        "stderr": prefix + "-worker.stderr.txt",
        "witness_stdout": prefix + "-witness.stdout",
        "witness_stderr": prefix + "-witness.stderr",
        "pre_spawn_receipt": prefix + "-pre-spawn.json",
        "outer_witness": prefix + "-outer-witness.json",
        "full_exit": prefix + "-full-exit.json",
        "tool_return": prefix + "-tool-return.json",
    }


def validate_nested_job(c, paths, limits, role):
    """Reject malformed nested argv/deadline/log fields before authority reads."""
    for key in ("freeze", "review"):
        require(
            type(c[key]) is dict and type(c[key].get("path")) is str,
            "nested authority path " + key,
        )
    require(
        type(c["commands"]) is list and len(c["commands"]) == 1,
        "sole nested command list",
    )
    job = c["commands"][0]
    require(
        type(job) is dict and set(job) == {"argv", "wall_s", "stdout", "stderr"},
        "exact nested command fields",
    )
    require(
        type(job["wall_s"]) is type(limits[1]) and job["wall_s"] == limits[1],
        "typed nested child wall",
    )
    require(
        type(job["argv"]) is list
        and all(type(arg) is str for arg in job["argv"])
        and job["argv"]
        == original_argv(c["freeze"]["path"], c["review"]["path"], role),
        "exact nested original argv",
    )
    require(
        type(job["stdout"]) is str
        and type(job["stderr"]) is str
        and job["stdout"] == paths["stdout"]
        and job["stderr"] == paths["stderr"],
        "exact nested log paths",
    )


def validate_supervisor_contract(
    contract_name, *, phase, root=ROOT, timely=lambda: None
):
    """Passive final authority, called before Popen and again before arrays."""
    require(phase in ("pre_spawn", "pre_array"), "literal gate phase")
    reg = metadata_preflight(root, timely)
    paths = contract_paths(contract_name, root)
    raw = safe(contract_name, root).read_bytes()
    c = strict_json(raw)
    required = {
        "schema",
        "phase",
        "role",
        "source_sha256",
        "commands",
        "supervisor_output",
        "max_tree_rss_bytes",
        "original_calls",
        "cumulative_before_s",
        "cumulative_limit_s",
        "outer_wall_s",
        "internal_wall_s",
        "science_wall_s",
        "freeze",
        "review",
        "reservation",
        "output",
        "pre_spawn_receipt_output",
    }
    require(set(c) == required, "exact final contract fields")
    role = c["role"]
    require(
        role in ("worker", "audit")
        and c["schema"] == "flylab.m3.e356.original-supervisor-contract.v1"
        and c["phase"] == "E356 original " + role,
        "literal final role/schema",
    )
    output = OUTPUT if role == "worker" else AUDIT_OUTPUT
    limits = (22, 21, 20, 18) if role == "worker" else (3, 2.5, 2, 2)
    validate_nested_job(c, paths, limits, role)
    for key, expected in (
        ("original_calls", 1),
        ("max_tree_rss_bytes", 2 << 30),
        ("cumulative_limit_s", 9900),
        ("outer_wall_s", limits[0]),
        ("internal_wall_s", limits[2]),
        ("science_wall_s", limits[3]),
    ):
        require(
            type(c[key]) is type(expected) and c[key] == expected, "typed final " + key
        )
    before = c["cumulative_before_s"]
    require(
        type(before) is float
        and math.isfinite(before)
        and before >= reg["original_cumulative_before_s"]
        and before + limits[0] <= 9900,
        "finite remaining original allowance",
    )
    require(
        role != "worker" or before == reg["original_cumulative_before_s"],
        "worker exact cumulative baseline",
    )
    own = own_sources(root)
    authority = {
        **reg["protected_source_sha256"],
        **reg["inherited_metadata_sha256"],
        **ADMIN,
        **own,
        REGISTRATION: REGISTRATION_SHA,
        REFLECTION: REFLECTION_SHA,
    }
    docs = {}
    for key in ("freeze", "review", "reservation"):
        data = read_descriptor(c[key], root)
        docs[key] = strict_json(data)
        authority[c[key]["path"]] = c[key]["sha256"]
    require(c["source_sha256"] == authority, "finite exact final source closure")
    for name, pin in authority.items():
        timely()
        require(
            sha(safe(name, root).read_bytes()) == pin, "current final source " + name
        )
    f, review, reserve = (docs[k] for k in ("freeze", "review", "reservation"))
    require(
        f == make_freeze(f["qualification"]["path"], root=root),
        "current strict complete Q and F",
    )
    require(
        review
        == {
            "schema": "flylab.m3.e356.root-original-review.v1",
            "approved": True,
            "role": role,
            "source_sha256": own,
            "freeze": c["freeze"],
            "reservation": c["reservation"],
            "supervisor_contract_path": contract_name,
            "registration_sha256": REGISTRATION_SHA,
            "reflection_sha256": REFLECTION_SHA,
            "output": output,
        },
        "exact ROOT phase review",
    )
    required_reserve = {
        "schema": "flylab.m3.e356.root-reservation.v1",
        "source_sha256": own,
        "registration_sha256": REGISTRATION_SHA,
        "reflection_sha256": REFLECTION_SHA,
        "cumulative_before_s": before,
        "cumulative_limit_s": 9900,
        "combined_reserved_s": 25,
        "original_calls": 1,
        "retry": 0,
        "role": role,
        "outer_s": limits[0],
        "child_s": limits[1],
        "internal_s": limits[2],
        "science_s": limits[3],
        "freeze": c["freeze"],
        "review_path": c["review"]["path"],
        "output": output,
    }
    if role == "audit":
        worker = reserve.get("closed_worker")
        require(type(worker) is dict, "audit closed worker authority")
        closed = strict_json(read_descriptor(worker, root))
        require(
            closed["source_sha256"] == own
            and closed["original_calls"] == 1
            and closed["exit_code"] == 0
            and closed["charged_once"] is True
            and closed["all_owned_ended"] is True
            and closed["known_identity_cleanup"] == []
            and closed["first_error"] is None
            and closed["secondary_errors"] == [],
            "worker once charge before dependent audit",
        )
        require(
            before
            == reg["original_cumulative_before_s"]
            + closed["actual_outer_full_return_s"],
            "audit actual once-charge cumulative",
        )
        required_reserve["closed_worker"] = worker
    require(
        reserve == required_reserve
        and all(type(reserve[k]) is type(v) for k, v in required_reserve.items()),
        "typed final reservation",
    )
    require(
        c["output"] == output
        and c["supervisor_output"] == paths["supervision"]
        and c["pre_spawn_receipt_output"] == paths["pre_spawn_receipt"],
        "exact final output paths",
    )
    job = {
        "argv": original_argv(c["freeze"]["path"], c["review"]["path"], role),
        "wall_s": limits[1],
        "stdout": paths["stdout"],
        "stderr": paths["stderr"],
    }
    require(
        type(c["commands"]) is list
        and c["commands"] == [job]
        and type(c["commands"][0]["wall_s"]) is type(limits[1]),
        "sole actual command nested fields",
    )
    require(not safe(output, root).exists(), "fresh original output")
    live = {
        "stdout",
        "stderr",
        "observer_stdout",
        "observer_stderr",
        "witness_stdout",
        "witness_stderr",
    }
    for key, name in paths.items():
        exists = safe(name, root).exists()
        if phase == "pre_array" and key in live:
            require(exists and safe(name, root).is_file(), "live owned log")
        elif phase == "pre_array" and key == "pre_spawn_receipt":
            require(exists, "actual pre-spawn receipt required")
        else:
            require(not exists, "fresh final path " + key)
    if phase == "pre_array":
        pre_raw = safe(paths["pre_spawn_receipt"], root).read_bytes()
        require(len(pre_raw) <= 32 << 10, "bounded pre-spawn receipt")
        pre = strict_json(pre_raw)
        require(
            pre["schema"] == "flylab.m3.e356.actual-pre-spawn-receipt.v1"
            and pre["passed"] is True
            and pre["phase"] == "pre_spawn"
            and pre["contract"] == descriptor(contract_name, raw)
            and pre["source_sha256"] == own
            and pre["before_child_Popen"] is True
            and pre["same_ROOT_ENTRY"] is True
            and type(pre["original_array_reads"]) is int
            and pre["original_array_reads"] == 0
            and type(pre["Torch_native_body_CPG_GPU_calls"]) is int
            and pre["Torch_native_body_CPG_GPU_calls"] == 0,
            "same digest actual pre-spawn array gate",
        )
    timely()
    return {
        "passed": True,
        "phase": phase,
        "source_sha256": own,
        "contract_sha256": sha(raw),
        "contract": c,
        "paths": paths,
    }


class Clock:
    def __init__(self, *, entry=None, internal=20, science=18, clock=time.perf_counter):
        self.clock = clock
        self.entry = clock() if entry is None else entry
        self.internal, self.science = internal, science

    def check(self, stage="internal"):
        now = self.clock()
        require(math.isfinite(now) and now >= self.entry, "monotone absolute clock")
        if now > self.entry + (self.science if stage == "science" else self.internal):
            raise TimeoutError("E356 absolute " + stage + " deadline")


QUOTAS = {
    "raw.npz": 16 << 20,
    "result.json": 256 << 10,
    "receipts.json": 256 << 10,
    "selected-ledger.json": 512 << 10,
    "manifest.json": 32 << 10,
    "sampler.json": 32 << 10,
    "failure.json": 128 << 10,
    "revocation.json": 128 << 10,
    "audit.json": 256 << 10,
}
ADVANCE_EXTERNAL_RESERVE = (256 + 32 + 32 + 64 + 128 + 256 + 128) << 10


class Store:
    def __init__(self, output, clock):
        require(not output.exists(), "fresh output")
        output.mkdir(parents=True)
        self.output, self.clock, self.written = output, clock, set()

    def check(self, extra=0):
        self.clock.check()
        actual = sum(p.stat().st_size for p in self.output.rglob("*") if p.is_file())
        reserve = (
            sum(
                v for k, v in QUOTAS.items() if k not in self.written and k != "raw.npz"
            )
            + ADVANCE_EXTERNAL_RESERVE
        )
        require(actual + extra + reserve <= 24 << 20, "advance recursive24MiB reserve")

    def put(self, name, raw):
        require(
            name in QUOTAS and type(raw) is bytes and len(raw) <= QUOTAS[name],
            "per-file bound before first write",
        )
        path = self.output / name
        require(not path.exists(), "exclusive publication")
        self.check(len(raw))
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        self.written.add(name)
        self.check()
        require(path.read_bytes() == raw, "closed publication readback")
        self.check()
        return descriptor(name, raw)

    def raw(self, arrays):
        import numpy as np

        self.check()
        path = self.output / "raw.npz"
        require(not path.exists(), "raw exclusive publication")
        with path.open("xb") as stream:

            class Bounded:
                def write(inner, value):
                    require(stream.tell() + len(value) <= 16 << 20, "raw streamed cap")
                    self.check(len(value))
                    return stream.write(value)

                def __getattr__(inner, name):
                    return getattr(stream, name)

            np.savez(Bounded(), **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        self.written.add("raw.npz")
        self.check()
        raw = path.read_bytes()
        self.check()
        return descriptor("raw.npz", raw)


def sampler_check(value, *, closed=False, sample_limit=42):
    require(
        type(sample_limit) is int
        and sample_limit in (42, 132)
        and value["sample_count"] <= sample_limit
        and len(value["owned_sampler_pids"]) <= 42
        and value["sampler_errors"] == [],
        "bounded complete sampler identity/errors",
    )
    require(
        value["sampled_additional_peak_bytes"] + (2 << 20) <= 1 << 30,
        "context-included additional GPU cap",
    )
    if closed:
        require(value["owned_thread_alive"] is False, "sampler natural thread close")


def generated_sampler_check(value, *, closed=False):
    """E357 generated direct NVML observations; original sampler remains unchanged."""
    sampler_check(value, closed=closed, sample_limit=132)
    identity, latest = value["device_identity"], value["latest_memory_bytes"]
    require(
        type(value["sample_count"]) is int
        and 0 < value["sample_count"] <= 132
        and value["state"] == ("CLOSED" if closed else "RUNNING")
        and value["owned_thread_alive"] is (not closed)
        and value["shutdown_count"] == (1 if closed else 0)
        and value["owned_sampler_pids"] == []
        and value["mapping"] == "single physical non-MIG index0"
        and value["CUDA_VISIBLE_DEVICES"] in (None, "0", identity["uuid"])
        and type(identity["physical_device_count"]) is int
        and identity["physical_device_count"] == 1
        and type(identity["physical_index"]) is int
        and identity["physical_index"] == 0
        and identity["mig_enabled"] is False
        and identity["driver"]
        and identity["handle_identity"] is not None
        and latest["uuid"] == identity["uuid"]
        and latest["handle_identity"] == identity["handle_identity"]
        and type(value["precontext_baseline_bytes"]) is int
        and 0 <= value["precontext_baseline_bytes"] <= latest["total"]
        and type(value["sampled_total_peak_bytes"]) is int
        and value["sampled_total_peak_bytes"] <= latest["total"]
        and value["quantization_allowance_bytes"] == 2 << 20,
        "E357 generated NVML baseline/device/lifecycle",
    )


def generated_gpu_watch():
    from tools.banc_m3_nvml_monitor import GPUMemoryWatch

    return GPUMemoryWatch(index=0, interval_s=0.5)


class PhaseTrace:
    """Bounded metadata only; CUDA elapsed is read after existing confirmation."""

    def __init__(self, *, scope, cuda_events=False):
        self.scope, self.cuda_events = scope, cuda_events
        self.started, self.rows, self.pending = time.perf_counter(), [], {}

    def emit(self, stage, event, *, work=None, cuda_stream_ms=None):
        require(len(self.rows) < 256, "bounded phase trace")
        row = {
            "scope": self.scope,
            "stage": stage,
            "event": event,
            "cpu_perf_counter_elapsed_s": time.perf_counter() - self.started,
            "cuda_stream_ms": cuda_stream_ms,
        }
        if work is not None:
            row["work"] = {
                key: value
                for key, value in work.items()
                if type(value) is int and key.endswith(("requested", "returned"))
            }
        self.rows.append(row)
        print(json.dumps(row, allow_nan=False, separators=(",", ":")), flush=True)

    def __call__(self, payload):
        stage, event = payload["stage"], payload.get("event", "progress")
        elapsed = None
        timed = stage.endswith(("-forward", "-R-backward"))
        if self.cuda_events and timed:
            import torch

            if event == "begin":
                pair = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                pair[0].record()
                self.pending[stage] = pair
            elif event == "returned" and stage in self.pending:
                self.pending[stage][1].record()
            elif event == "confirmed" and stage in self.pending:
                begin, end = self.pending.pop(stage)
                # query is nonblocking; unavailable values remain null, never zero.
                if begin.query() and end.query():
                    measured = begin.elapsed_time(end)
                    require(
                        math.isfinite(measured) and measured >= 0,
                        "finite optional CUDA stream time",
                    )
                    elapsed = measured
        self.emit(
            stage,
            event,
            work=native().work_counts(payload["receipts"]),
            cuda_stream_ms=elapsed,
        )


def publish_generated(
    output,
    raw,
    receipts,
    record,
    ledger,
    *,
    dimensions=None,
    timely=lambda: None,
    late=lambda: None,
):
    clock = Clock()
    clock.check = lambda stage="internal": timely()
    store = Store(output, clock)
    first = None
    try:
        validate_raw(raw, record["branch"], dimensions=dimensions)
        analysis = analyze(raw, record)
        raw_spec = store.raw(raw)
        receipt_spec = store.put("receipts.json", json_bytes(report_value(receipts)))
        ledger_spec = store.put(
            "selected-ledger.json", json_bytes(report_value(ledger))
        )
        result = {
            "schema": "flylab.m3.e356.reference-direction-result.v1",
            "qualified": True,
            "source_sha256": own_sources(),
            "record": report_value(record),
            "analysis": analysis,
            "raw": raw_spec,
            "receipts": receipt_spec,
            "ledger": ledger_spec,
            "target_c_order_sha256": array_sha(raw["target_q"]),
            "normalization_c_order_sha256": array_sha(raw["normalization"]),
        }
        result_spec = store.put("result.json", json_bytes(result))
        store.put(
            "manifest.json",
            json_bytes(
                {
                    "schema": "flylab.m3.e356.closed-publication.v1",
                    "raw": raw_spec,
                    "result": result_spec,
                    "receipts": receipt_spec,
                    "ledger": ledger_spec,
                    "all_fsynced_and_closed": True,
                }
            ),
        )
        late()
        store.check()
        return result
    except Exception as error:
        first = {"type": type(error).__name__, "message": str(error)[:1024]}
        name = (
            "revocation.json" if (output / "result.json").exists() else "failure.json"
        )
        payload = json_bytes(
            {
                "external_success": False,
                "scientific_qualification": False,
                "H356": "not_evaluated",
                "first_error": first,
                "secondary_errors": [],
            }
        )
        path = output / name
        require(
            not path.exists() and len(payload) <= 128 << 10, "bounded emergency reserve"
        )
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        require(path.read_bytes() == payload, "closed emergency readback")
        raise


def gate(
    freeze_name, review_name, output, *, role, root=ROOT, timely=lambda: None, argv=None
):
    reviewed = strict_json(safe(review_name, root).read_bytes())
    proof = validate_supervisor_contract(
        reviewed["supervisor_contract_path"],
        phase="pre_array",
        root=root,
        timely=timely,
    )
    c = proof["contract"]
    require(
        c["role"] == role
        and c["freeze"]["path"] == freeze_name
        and c["review"]["path"] == review_name
        and c["output"] == output
        and argv == c["commands"][0]["argv"][5:],
        "actual original CLI linkage",
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
        ),
        "CPU thread1 environment",
    )
    return strict_json(read_descriptor(c["freeze"], root))


def read_completed_raw(path, branch, *, timely=lambda: None):
    """Validate all literal NPY headers before decoding any numeric payload."""
    import numpy as np

    contract = registry()["raw_contract"]
    require(
        branch
        in (
            "normal",
            "zero_R_gradient",
            "quantized_zero_direction",
            "native_palette_not_deployable",
        ),
        "completed raw audit branch only",
    )
    require(
        path.is_file() and path.stat().st_size <= 16 << 20,
        "raw archive bound before open",
    )
    names = {key + ".npy": key for key in contract[branch]}
    core = native()
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        require(
            len(infos) == len(names) <= 26
            and len({info.filename for info in infos}) == len(infos)
            and {info.filename for info in infos} == set(names),
            "exact unique literal completed raw members",
        )
        # This complete first pass opens only bounded selected headers.
        for info in infos:
            timely()
            require(
                not info.is_dir()
                and 0 < info.file_size <= 4 << 20
                and not (info.flag_bits & 1),
                "bounded unencrypted NPY member",
            )
            spec = contract["fields"][names[info.filename]]
            with archive.open(info) as stream:
                header = core.strict_npy_header(stream)
            require(
                header["dtype"] == spec["dtype"]
                and header["shape"] == spec["shape"]
                and header["fortran_order"] is False,
                "literal raw header before any numeric decode",
            )
            dtype = np.dtype(header["dtype"])
            require(
                not dtype.hasobject
                and dtype.str in ("<f4", "<f8", "<i8")
                and info.file_size
                == header["header_bytes"] + math.prod(spec["shape"]) * dtype.itemsize,
                "exact native header payload length",
            )
        # No np.load call is possible until every member passes the header pass.
        raw = {}
        for info in infos:
            timely()
            value = np.load(io.BytesIO(archive.read(info)), allow_pickle=False)
            raw[names[info.filename]] = value
    validate_raw(raw, branch)
    timely()
    return raw


def validate_publication_manifest(manifest):
    names = {
        "raw": "raw.npz",
        "result": "result.json",
        "receipts": "receipts.json",
        "ledger": "selected-ledger.json",
    }
    require(
        type(manifest) is dict
        and set(manifest) == {*names, "schema", "all_fsynced_and_closed"}
        and manifest["schema"] == "flylab.m3.e356.closed-publication.v1"
        and manifest["all_fsynced_and_closed"] is True,
        "literal closed publication manifest",
    )
    for key, name in names.items():
        spec = manifest[key]
        require(
            type(spec) is dict
            and set(spec) == {"path", "bytes", "sha256"}
            and spec["path"] == name
            and type(spec["bytes"]) is int
            and 0 < spec["bytes"] <= QUOTAS[name]
            and type(spec["sha256"]) is str
            and len(spec["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in spec["sha256"]),
            "literal bounded manifest member " + key,
        )
    return True


def audit_current(frozen, output, *, timely=lambda: None):

    folder = safe(OUTPUT)
    require(
        not (folder / "failure.json").exists()
        and not (folder / "revocation.json").exists(),
        "no revoked worker",
    )
    result = strict_json((folder / "result.json").read_bytes())
    manifest = strict_json((folder / "manifest.json").read_bytes())
    validate_publication_manifest(manifest)
    require(
        result["source_sha256"] == own_sources() and result["qualified"] is True,
        "current qualified worker record",
    )
    for key in ("raw", "result", "receipts", "ledger"):
        spec = manifest[key]
        raw_bytes = (folder / spec["path"]).read_bytes()
        timely()
        require(
            len(raw_bytes) == spec["bytes"] and sha(raw_bytes) == spec["sha256"],
            "audit actual manifest linkage",
        )
    require((folder / "raw.npz").stat().st_size <= 16 << 20, "raw archive bound")
    with zipfile.ZipFile(folder / "raw.npz") as zipped:
        require(
            all(i.file_size <= 4 << 20 for i in zipped.infolist())
            and len(zipped.namelist()) == len(set(zipped.namelist())),
            "bounded unique raw members",
        )
    raw = read_completed_raw(
        folder / "raw.npz", result["record"]["branch"], timely=timely
    )
    validate_raw(raw, result["record"]["branch"])
    analysis = analyze(raw, result["record"])
    require(
        analysis == result["analysis"]
        and array_sha(raw["target_q"]) == result["target_c_order_sha256"]
        and array_sha(raw["normalization"]) == result["normalization_c_order_sha256"],
        "independent new raw arithmetic/digests",
    )
    ledger = strict_json((folder / "selected-ledger.json").read_bytes())
    require(
        len(ledger) == 24
        and ledger[-1]["path"] == frozen["cadence_label"]["path"]
        and ledger[-1]["slice_c_order_sha256"] == array_sha(raw["target_q"])
        and ledger[0]["selected"]["normalization"]["c_order_sha256"]
        == array_sha(raw["normalization"]),
        "actual selected/slice/norm ledger",
    )
    receipts = strict_json((folder / "receipts.json").read_bytes())
    require(
        native().work_counts(receipts) == result["record"]["work"]
        and result["record"]["neural_backward_requested"]
        == result["record"]["neural_backward_returned"]
        == 4,
        "actual work linkage",
    )
    store = Store(safe(output), Clock(entry=ENTRY, internal=2, science=2))
    store.put(
        "audit.json",
        json_bytes(
            {
                "schema": "flylab.m3.e356.independent-raw-audit.v1",
                "passed": True,
                "analysis": analysis,
                "source_sha256": own_sources(),
                "worker_result": descriptor(
                    OUTPUT + "/result.json", (folder / "result.json").read_bytes()
                ),
                "independent_original_deployability_recomputed": False,
                "full_parameter_Jacobian_independently_proven": False,
            }
        ),
    )


def worker(frozen, output, clock):
    core = native()
    watch = core.GPUMemoryWatch(interval_s=0.5)
    watch.start()
    acquired = {}
    ledger = []
    first = None
    try:
        packet = core.load_packet(
            ROOT,
            frozen["legacy_inventory"],
            timely=lambda: clock.check("science"),
            ledger=ledger,
        )
        target = load_label(
            frozen["cadence_label"], ledger, timely=lambda: clock.check("science")
        )
        raw, receipts, record = collect(
            packet,
            target_q=target,
            timely=lambda: clock.check("science"),
            progress=lambda p: acquired.update(p),
        )
        clock.check()
        sampler_check(watch.snapshot())
        closed = watch.close()
        sampler_check(closed, closed=True)
        record["sampler"] = closed
        record["worker_original_deployability_pinned"] = True
        publish_generated(
            safe(output), raw, receipts, record, ledger, timely=clock.check
        )
    except Exception as error:
        first = error
        if not safe(output).exists():
            store = Store(safe(output), Clock())
            if acquired.get("raw"):
                validate_raw(acquired["raw"], "technical_partial")
                store.raw(acquired["raw"])
            store.put(
                "failure.json",
                json_bytes(
                    {
                        "external_success": False,
                        "scientific_qualification": False,
                        "H356": "not_evaluated",
                        "first_error": {
                            "type": type(error).__name__,
                            "message": str(error)[:1024],
                        },
                        "acquired_receipts": report_value(acquired.get("receipts", {})),
                        "acquired_record": report_value(acquired.get("record", {})),
                        "stage": acquired.get("stage"),
                    }
                ),
            )
        raise
    finally:
        try:
            closed = watch.close()
            sampler_check(closed, closed=True)
            clock.check()
        except Exception as error:
            path = safe(output) / "revocation.json"
            if safe(output).exists() and not path.exists():
                payload = json_bytes(
                    {
                        "external_success": False,
                        "scientific_qualification": False,
                        "H356": "not_evaluated",
                        "first_error": {
                            "type": type(first or error).__name__,
                            "message": str(first or error)[:1024],
                        },
                        "secondary_errors": []
                        if first is None
                        else [
                            {"type": type(error).__name__, "message": str(error)[:1024]}
                        ],
                    }
                )
                with path.open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true")
    action.add_argument("--audit", action="store_true")
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--qualification")
    parser.add_argument("--review")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if not args.run and not args.audit:
        require(
            args.qualification is not None
            and args.review is None
            and args.output is None
            and args.freeze == REGISTRATION,
            "passive freeze CLI",
        )
        print(json.dumps(make_freeze(args.qualification), allow_nan=False))
        return 0
    require(
        args.qualification is None
        and args.review is not None
        and args.output is not None,
        "original worker/audit CLI",
    )
    role = "worker" if args.run else "audit"
    clock = Clock(
        entry=ENTRY, internal=20 if args.run else 2, science=18 if args.run else 2
    )
    frozen = gate(
        args.freeze,
        args.review,
        args.output,
        role=role,
        timely=clock.check,
        argv=sys.argv[1:] if argv is None else argv,
    )
    if args.run:
        worker(frozen, args.output, clock)
    else:
        audit_current(frozen, args.output, timely=clock.check)
    clock.check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
