"""E355 generated NumPy cases and the controlled whole-current-pytest witness."""

from __future__ import annotations

import time

ENTRY = time.perf_counter()
import builtins
import copy
import io
import json
import os
import runpy
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def product():
    from tools import recover_c_banc_m3_reference_teacher_loss

    return recover_c_banc_m3_reference_teacher_loss


def labels():
    import numpy as np

    source = np.arange(12600, dtype=np.float64).reshape(300, 42) / 1000
    target = source.copy()
    target[100:] += np.linspace(0.01, 0.07, 42)
    return {
        "source_q": source,
        "target_q": target,
        "normalization": np.arange(1, 7, dtype=np.float64),
    }


def generated_acquisition():
    """Analytic NumPy fixture; no claimed measured E354/Torch acquisition."""
    import numpy as np

    p = product()
    data = labels()
    ix = np.asarray([[s + k for k in range(4)] for s in (100, 164, 228, 292)], np.int64)
    norm = np.repeat(data["normalization"], 7)
    e = data["source_q"] - data["target_q"]
    raw = {k: v.copy() for k, v in data.items()}
    raw.update(
        window_source_q=data["source_q"][ix].copy(),
        window_target_q=data["target_q"][ix].copy(),
        window_control_tick=ix + 1,
        full_gq=2 * e / (12600 * norm),
        window_gq=2 * e[ix] / (672 * norm),
    )
    for k, value in {
        "full_loss": np.mean(e * e / norm),
        "window_loss": np.mean(e[ix] ** 2 / norm),
        "prefix_loss": 0.0,
        "startup_loss": 0.0,
        "teacher_self_loss": 0.0,
    }.items():
        raw[k] = np.asarray(value, np.float64)
    receipt = {
        "schema": "flylab.m3.e354.teacher-receipt.v1",
        "status": "completed",
        "first_error": None,
        "secondary_errors": [],
        "missing": [],
        "acquired": sorted(raw),
        "full_requested": 1,
        "full_returned": 1,
        "window_requested": 1,
        "window_returned": 1,
        "teacher_self_loss_direct_evaluations": 1,
        "teacher_self_gradient_measured_autograd": 0,
        "teacher_self_gradient_scope": "algebraic_identity_zero",
        "target_requires_grad": False,
        "normalization_requires_grad": False,
        "full_is_leaf": True,
        "window_is_leaf": True,
        "device": "cpu",
        "dtype": "float64",
        "q_only": True,
        "NN_body_CPG_GPU_calls": 0,
    }
    result = {
        "status": "completed",
        "receipt": receipt,
        "H354": "qualified_offline_teacher_loss",
        "external_completion_qualified": False,
        "metrics": {
            k: float(raw[k])
            for k in (
                "full_loss",
                "window_loss",
                "prefix_loss",
                "startup_loss",
                "teacher_self_loss",
            )
        },
        **p.frozen().DENIED,
    }
    return data, raw, receipt, result


def pack(root, name, values, selected=None):
    import numpy as np

    p = product()
    buf = io.BytesIO()
    np.savez_compressed(buf, **values)
    payload = buf.getvalue()
    (root / name).write_bytes(payload)
    members = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        for key in selected or values:
            info = z.getinfo(key + ".npy")
            with z.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
                header = stream.tell()
            members[key] = {
                "member": key + ".npy",
                "zip_compression": info.compress_type,
                "compressed_bytes": info.compress_size,
                "uncompressed_bytes": info.file_size,
                "crc32": info.CRC,
                "npy_version": list(version),
                "shape": list(shape),
                "dtype": dtype.str,
                "fortran_order": order,
                "header_bytes": header,
            }
    return {**p.descriptor(name, payload), "selected_members": members}


def recovery_fixture(root):
    import numpy as np

    p = product()
    data, raw, receipt, result = generated_acquisition()
    ticks = np.arange(301, dtype=np.int64)
    literal = np.full((1, 42), -77.0, np.float64)
    payloads = {}
    for role, command, clock in (
        ("control", data["source_q"], "physics_time_s"),
        ("cadence", data["target_q"], "control_time_s"),
    ):
        values = {
            "command": np.concatenate([literal, command]),
            "control_tick": ticks,
            clock: ticks * 0.005,
            "unselected": np.asarray([float("nan")]),
        }
        payloads[role] = pack(
            root, role + ".npz", values, ("command", "control_tick", clock)
        )
    payloads["reference"] = pack(
        root,
        "reference.npz",
        {
            "normalization": data["normalization"],
            "unselected": np.asarray([float("inf")]),
        },
        ("normalization",),
    )
    spec = pack(root, "raw.npz", raw)
    return {
        "reader": p.generated_reader(root),
        "registration": {"payloads": payloads, "acquired_raw": spec},
        "authorized": True,
        "declared_receipt": receipt,
        "declared_result": result,
    }


def failed_metadata():
    p = product()
    reg = json.loads((p.ROOT / p.REGISTRATION).read_text("utf8"))
    docs = {
        name: (p.ROOT / name).read_text("utf8")
        if name.endswith(".txt")
        else json.loads((p.ROOT / name).read_text("utf8"))
        for name in reg["failure_provenance"]
    }
    return reg, docs


def test_independent_formula_and_label_bytes(tmp_path):
    import numpy as np

    p = product()
    ctx = recovery_fixture(tmp_path)
    report = p.recover(ctx)
    assert report["H355"] == "recovered_offline_label_loss_derivative_consistency"
    assert (
        report["worker_actual_exit_code"] is None
        and not report["E354_completion_recovered"]
    )
    assert (
        set(ctx["reader"].read_counts.values()) == {1}
        and len(ctx["reader"].opened_members) == 20
    )
    assert all(member != "unselected.npy" for _, member in ctx["reader"].opened_members)
    data, raw, _, _ = generated_acquisition()
    eps = 1e-5
    q = data["source_q"].copy()
    q[125, 13] += eps
    plus = np.mean((q - data["target_q"]) ** 2 / np.repeat(data["normalization"], 7))
    q[125, 13] -= 2 * eps
    minus = np.mean((q - data["target_q"]) ** 2 / np.repeat(data["normalization"], 7))
    assert abs((plus - minus) / (2 * eps) - raw["full_gq"][125, 13]) < 1e-11
    altered = copy.deepcopy(ctx["declared_result"])
    altered["metrics"]["window_loss"] += 0.1
    ctx["declared_result"] = altered
    import pytest

    with pytest.raises(ValueError, match="stored result math"):
        p.recover(ctx)


def test_raw_derivative_and_result_perturbations():
    import pytest

    p = product()
    for fault in ("gradient", "index", "bytes", "result"):
        data, raw, receipt, result = generated_acquisition()
        if fault == "gradient":
            raw["window_gq"][0, 0, 0] += 0.1
        elif fault == "index":
            raw["window_control_tick"][0, 0] += 1
        elif fault == "bytes":
            data["source_q"][0, 0] += 0.01
        else:
            result["metrics"]["full_loss"] += 0.1
        with pytest.raises(ValueError):
            p.frozen().independent_check(data, raw, receipt, result)


def test_raw_roster_index_nonfinite_partial(tmp_path):
    import numpy as np
    import pytest

    p = product()
    data, raw, receipt, result = generated_acquisition()
    for mode in ("partial", "nonfinite", "extras"):
        changed = copy.deepcopy(raw)
        if mode == "partial":
            changed.pop("window_gq")
        elif mode == "nonfinite":
            changed["full_gq"][0, 0] = np.nan
        else:
            changed["surplus"] = np.zeros(1)
        with pytest.raises(ValueError):
            p.frozen().independent_check(data, changed, receipt, result)
    spec = pack(tmp_path, "extra.npz", {**raw, "extra": np.zeros(1)})
    reader = p.generated_reader(tmp_path)
    with pytest.raises(ValueError, match="exact recovery raw"):
        reader.npz(spec, p.frozen().raw_roster())
    spec["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="pinned consumed"):
        reader.npz(spec, p.frozen().raw_roster())


def test_failed_outer_provenance_not_completed_worker(tmp_path):
    p = product()
    reg, docs = failed_metadata()
    result, receipt = p.failed_admission(reg, docs, root=tmp_path)
    assert result["status"] == receipt["status"] == "completed"
    assert (
        docs[p.BASE + "/parent-e354-failed-full-return.json"]["worker_actual_exit_code"]
        is None
    )
    assert (
        docs[p.BASE + "/parent-e354-final-decision.json"][
            "diagnostic_completion_qualified"
        ]
        is False
    )


def test_failed_outer_rejects_unknown_or_wrong_exit(tmp_path):
    import pytest

    p = product()
    reg, docs = failed_metadata()
    for field, value in (
        ("exit_code", 0),
        ("exit_code", None),
        ("worker_actual_exit_code", 0),
        ("all_owned_ended", False),
    ):
        changed = copy.deepcopy(docs)
        changed[p.BASE + "/parent-e354-failed-full-return.json"][field] = value
        with pytest.raises(ValueError):
            p.failed_admission(reg, changed, root=tmp_path)


def test_private_gate_denies_original_reads(tmp_path, monkeypatch):
    import pytest

    p = product()
    ctx = p.metadata_preflight()
    assert ctx["authorized"] is False and not ctx["reader"].array_authorized
    assert all(
        not name.endswith((".npz", ".npy", ".gz")) for name in ctx["reader"].read_counts
    )
    calls = []

    class Denied:
        def npz(self, *args, **kwargs):
            calls.append("array")
            raise AssertionError("array before gate")

    with pytest.raises(ValueError, match="private gate"):
        p.recover({"authorized": False, "reader": Denied()})
    monkeypatch.setattr(
        p,
        "metadata_preflight",
        lambda *args: (_ for _ in ()).throw(ValueError("unqualified default route")),
    )
    assert (
        p.main(
            [
                "--audit",
                "--freeze",
                "absent",
                "--review",
                "absent",
                "--output",
                p.OUTPUT,
            ]
        )
        == 1
    )
    assert calls == [] and not (tmp_path / p.OUTPUT).exists()


def supervisor_fixture(root, monkeypatch=None):
    p = product()
    reg = json.loads((p.ROOT / p.REGISTRATION).read_text("utf8"))
    files = {
        **reg["consumed_sources"],
        **p.own_sources(),
        p.REGISTRATION: p.REG_SHA,
        p.REFLECTION: p.REF_SHA,
    }
    for name in files:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((p.ROOT / name).read_bytes())
    specs = []
    name = p.BASE + "/parent-e346-e355-sol-FIRST-fixture-contract.json"
    rn = p.BASE + "/e355-sol-FIRST-fixture-review.json"
    resn = p.BASE + "/e355-sol-FIRST-fixture-reservation.json"
    own = p.own_sources(root)
    frozen = {
        "schema": "flylab.m3.e355.forensic-freeze.v1",
        "source_sha256": own,
        "registration_sha256": p.REG_SHA,
        "reflection_sha256": p.REF_SHA,
        "original_authorized": False,
        "qualification": {
            "path": p.BASE + "/e355-sol-FIRST-synthetic-qualification.json"
        },
    }
    for tail, value in (("freeze", frozen),):
        name = p.BASE + "/e355-sol-FIRST-fixture-" + tail + ".json"
        raw = p.json_bytes(value)
        (root / name).write_bytes(raw)
        specs.append(p.descriptor(name, raw))
    f = specs[0]
    reserved = p.reservation_value(f, rn, own)
    raw = p.json_bytes(reserved)
    (root / resn).write_bytes(raw)
    res = p.descriptor(resn, raw)
    name = p.BASE + "/parent-e346-e355-sol-FIRST-fixture-contract.json"
    reviewed = {
        "schema": "flylab.m3.e355.root-original-review.v1",
        "approved": True,
        "source_sha256": own,
        "freeze": f,
        "reservation": res,
        "supervisor_contract_path": name,
        "registration_sha256": p.REG_SHA,
        "reflection_sha256": p.REF_SHA,
        "output": p.OUTPUT,
    }
    raw = p.json_bytes(reviewed)
    (root / rn).write_bytes(raw)
    r = p.descriptor(rn, raw)
    specs = [f, r, res]
    if monkeypatch is not None:
        monkeypatch.setattr(p, "make_freeze", lambda *a, **k: frozen)
    paths = p.contract_paths(name, root)
    c = {
        "schema": "flylab.m3.e355.original-supervisor-contract.v1",
        "phase": "E355 original forensic NumPy audit",
        "source_sha256": {**files, **{s["path"]: s["sha256"] for s in specs}},
        "commands": [
            {
                "argv": p.original_argv(f["path"], r["path"]),
                "wall_s": 4,
                "stdout": paths["stdout"],
                "stderr": paths["stderr"],
            }
        ],
        "supervisor_output": paths["supervision"],
        "max_tree_rss_bytes": 2 << 30,
        "original_calls": 1,
        "cumulative_before_s": 9871.733085099502,
        "cumulative_limit_s": 9900,
        "outer_wall_s": 6,
        "internal_wall_s": 4,
        "freeze": f,
        "review": r,
        "reservation": res,
        "output": p.OUTPUT,
    }

    def save(value):
        (root / name).write_text(json.dumps(value), encoding="utf8")

    save(c)
    return c, name, specs, paths, save


def test_contract_missing_cumulative_and_nested_fields(tmp_path):
    import pytest

    p = product()
    c, name, specs, _, save = supervisor_fixture(tmp_path)
    for top, key in (
        (True, "cumulative_limit_s"),
        (True, "cumulative_before_s"),
        (False, "wall_s"),
        (False, "argv"),
        (False, "stderr"),
    ):
        changed = copy.deepcopy(c)
        del (changed if top else changed["commands"][0])[key]
        save(changed)
        with pytest.raises(ValueError, match="required|nested"):
            p.validate_supervisor_contract(name, *specs, root=tmp_path)
    assert not (tmp_path / p.OUTPUT).exists()


def test_contract_strict_paths_sources_wall_and_freshness(tmp_path, monkeypatch):
    import pytest

    p = product()
    c, name, specs, paths, save = supervisor_fixture(tmp_path, monkeypatch)
    assert (
        p.validate_supervisor_contract(name, *specs, root=tmp_path)["phase"]
        == "pre_spawn"
    )
    for key, value in (
        ("original_calls", True),
        ("cumulative_before_s", float("nan")),
        ("outer_wall_s", 8),
        ("supervisor_output", "elsewhere.json"),
    ):
        changed = copy.deepcopy(c)
        changed[key] = value
        save(changed)
        with pytest.raises(ValueError):
            p.validate_supervisor_contract(name, *specs, root=tmp_path)
    changed = copy.deepcopy(c)
    changed["source_sha256"][p.OWNED[0]] = "0" * 64
    save(changed)
    with pytest.raises(ValueError, match="source"):
        p.validate_supervisor_contract(name, *specs, root=tmp_path)
    save(c)
    for key in (
        "stdout",
        "stderr",
        "observer_stdout",
        "observer_stderr",
        "witness_stdout",
        "witness_stderr",
    ):
        (tmp_path / paths[key]).write_bytes(b"")
    with pytest.raises(ValueError, match="fresh"):
        p.validate_supervisor_contract(name, *specs, root=tmp_path)
    assert (
        p.validate_supervisor_contract(name, *specs, root=tmp_path, phase="array_gate")[
            "phase"
        ]
        == "array_gate"
    )
    (tmp_path / paths["supervision"]).write_bytes(b"{}")
    with pytest.raises(ValueError, match="fresh final"):
        p.validate_supervisor_contract(name, *specs, root=tmp_path, phase="array_gate")


def good_case_receipt():
    p = product()
    expected = {p.OWNED[1]: p.ast_cases(p.ROOT / p.OWNED[1])}
    nodes = [p.OWNED[1] + "::" + fn for fn in expected[p.OWNED[1]]]
    return {
        "effective_environment": {
            "PYTEST_ADDOPTS_absent": True,
            "PYTEST_PLUGINS_absent": True,
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "raw_prior_values_recorded": False,
        },
        "pytest_main_requested": 1,
        "pytest_main_returned": 1,
        "pytest_return_code": 0,
        "collection_errors": [],
        "deselected": [],
        "first_error": None,
        "secondary_errors": [],
        "expected_functions": expected,
        "basetemp_fresh_before_execution": True,
        "collected": nodes,
        "empty_ini": {"path": p.BASE + "/parent-e355-pytest-empty.ini"},
        "case_paths": {n: p.OWNED[1] for n in nodes},
        "reports": [
            {"nodeid": n, "when": w, "outcome": "passed"}
            for n in nodes
            for w in ("setup", "call", "teardown")
        ],
    }


def test_primary_cpu_requires_current_cases(monkeypatch):
    import pytest

    p = product()
    receipt = good_case_receipt()
    assert p.complete_case_receipt(receipt, receipt["expected_functions"])
    for counter in ("pytest_main_requested", "pytest_main_returned"):
        for wrong in (True, False, None, 0, 2, 1.0):
            changed = copy.deepcopy(receipt)
            changed[counter] = wrong
            with pytest.raises(ValueError, match="actual complete pytest return"):
                p.complete_case_receipt(changed, changed["expected_functions"])
    extra = p.BASE + "/parent-e355-root-FIRST-regressions.py"
    names = p.ast_cases(p.ROOT / extra)
    for prefix in (Path(extra).name, extra, ""):
        changed = copy.deepcopy(receipt)
        changed["expected_functions"][extra] = names
        for name in names:
            node = prefix + "::" + name
            changed["collected"].append(node)
            changed["case_paths"][node] = extra
            changed["reports"].extend(
                {"nodeid": node, "when": phase, "outcome": "passed"}
                for phase in ("setup", "call", "teardown")
            )
        preserved = copy.deepcopy(changed)
        assert p.complete_case_receipt(changed, changed["expected_functions"])
        assert changed == preserved
    for prefix in ("unrelated.py", "../" + Path(extra).name):
        changed = copy.deepcopy(preserved)
        old_node = changed["collected"][-1]
        wrong_node = prefix + "::" + old_node.split("::")[1]
        changed["collected"][-1] = wrong_node
        changed["case_paths"][wrong_node] = changed["case_paths"].pop(old_node)
        for report in changed["reports"]:
            if report["nodeid"] == old_node:
                report["nodeid"] = wrong_node
        with pytest.raises(ValueError, match="controlled top-level actual case"):
            p.complete_case_receipt(changed, changed["expected_functions"])
    own = p.own_sources()

    def spec(name):
        return {"path": name, "bytes": 2, "sha256": "1" * 64}

    execution = {
        k: spec(k) for k in ("contract", "supervision", "identity", "full_exit")
    }
    case_name = p.BASE + "/parent-e355-sol-FIRST-typed-case.json"
    case_spec = spec(case_name)
    case = {
        "empty_ini": spec("empty.ini"),
        "basetemp": "external-generated-only",
        "complete_suite": [p.OWNED[1]],
        "expected_functions": receipt["expected_functions"],
        "receipt_output": "case-receipt",
    }
    receipt.update(
        schema="flylab.m3.e355.pytest-case-receipt.v1",
        source_sha256=own,
        case_contract=case_spec,
        empty_ini=case["empty_ini"],
        complete_current_cases=True,
        pytest_args=[
            "-q",
            "-p",
            "no:cacheprovider",
            "--noconftest",
            "-c",
            "empty.ini",
            "--basetemp",
            case["basetemp"],
            p.OWNED[1],
        ],
    )
    argv = [
        ".venv/Scripts/python.exe",
        "-X",
        "utf8",
        "-B",
        p.OWNED[1],
        "--execute-whole-current-pytest",
        case_name,
    ]
    values = {
        "contract": {
            "source_sha256": {
                **own,
                case_name: case_spec["sha256"],
                "empty.ini": "1" * 64,
            },
            "original_calls": 0,
            "max_tree_rss_bytes": 2 << 30,
            "commands": [{"argv": argv}],
        },
        "supervision": {
            "success": True,
            "max_tree_rss_bytes": 1000,
            "attempts": [{"argv": argv.copy(), "exit_code": 0, "killed": False}],
        },
        "identity": {
            "qualified": True,
            "all_owned_pids_ended": True,
            "errors": [],
            "actual_supervisor_exit_code": 0,
            "contract": execution["contract"],
            "supervision": execution["supervision"],
            "owned_pids": [1],
            "pid_create_times": {"1": 123.0},
            "present_after_return": {"1": False},
        },
        "outer": {
            "all_owned_ended": True,
            "known_identity_cleanup": [],
            "first_error": None,
            "secondary_errors": [],
            "exit_code": 0,
            "owned_identities": [{"pid": 1, "create_time": 123.0, "close": "ended"}],
        },
        "tool": {
            "actual_returned_exit_code": 0,
            "actual_outer_full_return_s": 0.2,
            "tool_chunk_id": "typed-not-actual-job",
        },
        "case-receipt": receipt,
    }
    values["full_exit"] = {
        "schema": "flylab.m3.e355.actual-full-return.v1",
        "source_sha256": own,
        "scope": "prevalidation-through-observer-final-exit",
        "exit_code": 0,
        "all_owned_ended": True,
        "known_identity_cleanup": [],
        "first_error": None,
        "secondary_errors": [],
        "full_exit_elapsed_s": 0.2,
        "charged_s": 0.2,
        "actual_outer_full_return_s": 0.2,
        "charged_once": True,
        "original_calls": 0,
        **{k: execution[k] for k in ("contract", "supervision", "identity")},
        "outer_witness": spec("outer"),
        "actual_tool_return": spec("tool"),
        "actual_tool_chunk_id": "typed-not-actual-job",
        "pytest_case_execution": {
            "case_contract": case_spec,
            "empty_ini": case["empty_ini"],
            "case_receipt": spec("case-receipt"),
        },
    }

    class TypedReader:
        root = p.ROOT

        def json(self, descriptor):
            return values[descriptor["path"]]

    monkeypatch.setattr(p, "validate_case_contract", lambda *a, **k: (case, case_spec))
    assert (
        p.check_execution(TypedReader(), execution, own, 20, case_role=True)[
            "exit_code"
        ]
        == 0
    )
    for wrong in (
        [".venv/Scripts/python.exe", "-X", "utf8", "-B", "-c", "pass"],
        [
            ".venv/Scripts/python.exe",
            "-X",
            "utf8",
            "-B",
            "-m",
            "pytest",
            "--collect-only",
        ],
        [*argv, "-k", "one"],
    ):
        values["contract"]["commands"][0]["argv"] = wrong
        values["supervision"]["attempts"][0]["argv"] = wrong
        with pytest.raises(ValueError, match="controlled TEST"):
            p.check_execution(TypedReader(), execution, own, 20, case_role=True)
    for fault in ("noop", "skip", "missing_teardown", "deselected", "return"):
        r = copy.deepcopy(receipt)
        if fault == "noop":
            r["collected"] = []
        elif fault == "skip":
            r["reports"][1]["outcome"] = "skipped"
        elif fault == "missing_teardown":
            r["reports"].pop()
        elif fault == "deselected":
            r["deselected"] = [r["collected"][0]]
        else:
            r["pytest_return_code"] = 5
        with pytest.raises(ValueError):
            p.complete_case_receipt(r, r["expected_functions"])


def test_pytest_hidden_selection_removed_and_incomplete_rejected(monkeypatch):
    import pytest

    p = product()
    monkeypatch.setenv("PYTEST_ADDOPTS", "--collect-only -k never -m imaginary")
    monkeypatch.setenv("PYTEST_PLUGINS", "untrusted_plugin")
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "0")
    effective = p.clean_pytest_environment()
    assert effective["prior_variable_present"] == {
        "PYTEST_ADDOPTS": True,
        "PYTEST_PLUGINS": True,
    }
    assert (
        "PYTEST_ADDOPTS" not in os.environ
        and "PYTEST_PLUGINS" not in os.environ
        and os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    )
    assert (
        "never" not in json.dumps(effective)
        and effective["raw_prior_values_recorded"] is False
    )
    receipt = good_case_receipt()
    receipt["collected"].pop()
    receipt["case_paths"].pop(next(reversed(receipt["case_paths"])))
    receipt["reports"] = receipt["reports"][:-3]
    with pytest.raises(ValueError, match="every current AST"):
        p.complete_case_receipt(receipt, receipt["expected_functions"])


def test_publication_caps_and_late_close(tmp_path, monkeypatch):
    import pytest

    p = product()
    # Publish the complete recovered report, including the reader's member rows.
    # A mocked minimal report cannot catch non-JSON reader metadata.
    fixtures = tmp_path / "complete-recovery-inputs"
    fixtures.mkdir()
    complete_output = tmp_path / "complete-recovery-output"
    complete_output.mkdir()
    recovered, complete_code = p.run_recovery(
        recovery_fixture(fixtures), complete_output
    )
    assert complete_code == 0
    saved = json.loads((complete_output / "report.json").read_text("utf8"))
    assert saved == json.loads(p.json_bytes(recovered))
    assert len(saved["opened_members"]) == 20
    assert all(type(row) is list and len(row) == 2 for row in saved["opened_members"])
    assert set(saved["read_counts"].values()) == {1}
    assert not saved["external_completion_qualified"]
    assert not (complete_output / "failure.json").exists()
    with pytest.raises(ValueError, match="cap before"):
        p.publish(tmp_path, "huge.json", {"data": "x" * 300}, cap=500, reserve=400)
    assert not (tmp_path / "huge.json").exists()
    monkeypatch.setattr(
        p,
        "recover",
        lambda *a, **k: {"H355": "recovered_offline_label_loss_derivative_consistency"},
    )

    def late():
        if (tmp_path / "report.json").exists():
            raise TimeoutError("after success close")

    report, code = p.run_recovery(
        {"declared_receipt": {"full_returned": 1, "window_returned": 1}},
        tmp_path,
        timely=late,
    )
    assert (
        code == 1
        and report["H355"] == "not_evaluated"
        and report["late_error_has_precedence"]
        and report["independent_calculation_completed"]
    )
    assert (
        report["declared_E354_receipt"]["window_returned"] == 1
        and (tmp_path / "report.json").exists()
    )
    assert (
        json.loads((tmp_path / "failure.json").read_text())["first_error"]["type"]
        == "TimeoutError"
    )


def test_import_help_is_lazy(monkeypatch, capsys):
    import pytest

    p = product()
    original = builtins.__import__

    def guarded(name, *a, **k):
        if name.split(".")[0] in ("numpy", "torch", "flylab", "mujoco", "psutil"):
            raise AssertionError("heavy import before approval/help")
        return original(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", guarded)
    namespace = runpy.run_path(str(p.ROOT / p.OWNED[0]))
    assert namespace["_MODULES"] == {}
    with pytest.raises(SystemExit) as value:
        namespace["main"](["--help"])
    assert value.value.code == 0 and "--audit" in capsys.readouterr().out


def test_strict_json_duplicate_and_overflow():
    import numpy as np
    import pytest

    p = product()
    for raw in (b'{"x":1,"x":2}', b'{"x":1e999}', b'{"x":NaN}'):
        with pytest.raises(ValueError):
            p.helper().strict_json(raw)
    assert json.loads(
        p.json_bytes({"b": np.bool_(True), "i": np.int64(3), "f": np.float32(0.5)})
    ) == {"b": True, "i": 3, "f": 0.5}
    for bad in (
        np.array(0),
        np.array([1]),
        float("inf"),
        np.complex128(2j),
        {1: "bad"},
    ):
        with pytest.raises((ValueError, TypeError)):
            p.json_bytes(bad)


def execute_whole_current_pytest(contract_name):
    """Private real entrypoint: exactly one pytest.main, actual case/phase receipt."""
    p = product()
    case, spec = p.validate_case_contract(contract_name, before_execution=True)
    effective = p.clean_pytest_environment()
    args = [
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
    receipt = {
        "schema": "flylab.m3.e355.pytest-case-receipt.v1",
        "source_sha256": p.own_sources(),
        "case_contract": spec,
        "empty_ini": case["empty_ini"],
        "pytest_args": args,
        "expected_functions": case["expected_functions"],
        "effective_environment": effective,
        "basetemp_fresh_before_execution": True,
        "pytest_main_requested": 0,
        "pytest_main_returned": 0,
        "pytest_return_code": None,
        "collected": [],
        "case_paths": {},
        "reports": [],
        "collection_errors": [],
        "deselected": [],
        "first_error": None,
        "secondary_errors": [],
        "complete_current_cases": False,
    }

    class Observe:
        def pytest_collection_finish(self, session):
            receipt["collected"] = [item.nodeid for item in session.items]
            receipt["case_paths"] = {
                item.nodeid: item.path.resolve()
                .relative_to(p.ROOT.resolve())
                .as_posix()
                for item in session.items
            }

        def pytest_deselected(self, items):
            receipt["deselected"].extend(item.nodeid for item in items)

        def pytest_collectreport(self, report):
            if report.failed:
                receipt["collection_errors"].append(
                    {"nodeid": report.nodeid, "outcome": report.outcome}
                )

        def pytest_runtest_logreport(self, report):
            receipt["reports"].append(
                {
                    "nodeid": report.nodeid,
                    "when": report.when,
                    "outcome": report.outcome,
                }
            )

    try:
        import pytest

        receipt["pytest_main_requested"] = 1
        result = pytest.main(args, plugins=[Observe()])
        receipt["pytest_main_returned"] = 1
        receipt["pytest_return_code"] = int(result)
        p.complete_case_receipt(receipt, case["expected_functions"])
        receipt["complete_current_cases"] = True
    except Exception as exc:  # noqa: BLE001 -- preserve actual request/return/phases on failure
        receipt["first_error"] = p.error_record("whole_current_pytest", exc)
    raw = p.json_bytes(receipt)
    p.require(len(raw) <= 64 << 10, "bounded exclusive pytest case receipt")
    with (p.ROOT / case["receipt_output"]).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    p.require(
        time.perf_counter() - ENTRY <= 20,
        "generated whole-pytest includes receipt close",
    )
    return 0 if receipt["complete_current_cases"] else 1


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--execute-whole-current-pytest":
        raise SystemExit(
            "only the registered whole-current-pytest private route is supported"
        )
    raise SystemExit(execute_whole_current_pytest(sys.argv[2]))
