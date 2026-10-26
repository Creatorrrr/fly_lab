"""E354 generated CPU f64 labels, independent audit and bounded failure paths."""

import ast
import io
import json
import runpy
import types
import zipfile

import numpy as np
import pytest

from tools import banc_m3_reference_cadence_teacher as p


def data():
    source = np.arange(12600, dtype=np.float64).reshape(300, 42) / 1000
    target = source.copy()
    target[100:] += np.linspace(0.01, 0.07, 42)
    return {
        "source_q": source,
        "target_q": target,
        "normalization": np.arange(1, 7, dtype=np.float64),
    }


def test_teacher_derivative_index_isolation_and_finite_difference():
    t = p.torch_module()
    d = data()
    target, norm = t.tensor(d["target_q"]), t.tensor(d["normalization"])
    teacher = p.ReferenceCadenceTeacher(target, norm, device="cpu")
    q = t.tensor(d["source_q"], requires_grad=True)
    before = teacher.full(q)
    gf = t.autograd.grad(before, q)[0]
    qw = t.stack([q.detach()[s : s + 4] for s in p.STARTS]).requires_grad_(True)
    gw = t.autograd.grad(teacher.window(qw), qw)[0]
    assert np.allclose(
        gw.numpy(),
        (300 / 16) * np.stack([gf.numpy()[s : s + 4] for s in p.STARTS]),
        rtol=1e-12,
        atol=1e-14,
    )
    plus, minus = q.detach().clone(), q.detach().clone()
    plus[125, 13] += 1e-5
    minus[125, 13] -= 1e-5
    fd = (teacher.full(plus) - teacher.full(minus)) / 2e-5
    assert float(fd) == pytest.approx(float(gf[125, 13]), rel=1e-6, abs=1e-12)
    target.add_(100)
    norm.mul_(10)
    assert float(teacher.full(q)) == float(before)
    assert np.array_equal(
        teacher._window_target.numpy(),
        np.stack([d["target_q"][s : s + 4] for s in p.STARTS]),
    )
    assert not teacher._target.requires_grad and not teacher._norm.requires_grad
    assert (
        p.ReferenceCadenceTeacher(
            t.tensor(d["target_q"]), t.tensor(d["normalization"]), device="cpu"
        ).full(t.tensor(d["target_q"]))
        == 0
    )


def test_strict_dtype_device_nonfinite_and_overflow():
    t = p.torch_module()
    d = data()
    teacher = p.ReferenceCadenceTeacher(
        t.tensor(d["target_q"]), t.tensor(d["normalization"]), device="cpu"
    )
    for value in (
        t.zeros(300, 42, dtype=t.float32),
        t.empty(300, 42, dtype=t.float64, device="meta"),
        t.full((300, 42), float("nan"), dtype=t.float64),
        t.full((300, 42), 1e308, dtype=t.float64),
    ):
        with pytest.raises((ValueError, RuntimeError)):
            teacher.full(value)
    with pytest.raises(RuntimeError, match="positive"):
        p.ReferenceCadenceTeacher(
            t.tensor(d["target_q"]), t.zeros(6, dtype=t.float64), device="cpu"
        )
    with pytest.raises(RuntimeError, match="finite"):
        teacher.validate_gradient(t.full((300, 42), float("inf"), dtype=t.float64))
    tree = ast.parse((p.ROOT / p.OWNED[0]).read_text("utf8"))
    hot = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name in ("full", "window", "_input", "_finite")
    ]
    assert not any(
        isinstance(n, ast.Attribute)
        and n.attr in ("item", "cpu", "numpy", "to", "perf_counter")
        for fn in hot
        for n in ast.walk(fn)
    )


def test_actual_producer_publisher_independent_auditor(tmp_path):
    d = data()
    raw, receipt = p.produce(d)
    assert (
        receipt["full_requested"]
        == receipt["full_returned"]
        == receipt["window_requested"]
        == receipt["window_returned"]
        == 1
    )
    output = tmp_path / p.OUTPUT
    output.mkdir(parents=True)
    result, code = p.publish_worker(output, d, raw, receipt)
    assert code == 0 and len(raw) == 13
    roster = {f.name: p.descriptor(f.name, f.read_bytes()) for f in output.iterdir()}
    reader = p.generated_reader(tmp_path)
    ctx = {
        "authorized": True,
        "mode": "audit",
        "reader": reader,
        "worker_seal": {"expected_exit_code": 0, "artifacts": roster},
    }
    audit = p.audit_worker(ctx, d)
    assert audit["audit_passed"] and audit["H354"] == "qualified_offline_teacher_loss"
    assert audit["metrics"] == pytest.approx(result["metrics"], rel=1e-11, abs=1e-12)
    assert set(reader.read_counts.values()) == {1}
    assert len(reader.opened_members) == 13
    assert all(audit[k] is False for k in p.DENIED)


@pytest.mark.parametrize("fault", ["gradient", "index", "result"])
def test_independent_rejection(fault):
    d = data()
    raw, receipt = p.produce(d)
    result = {
        "status": "completed",
        "receipt": receipt,
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
        "H354": "qualified_offline_teacher_loss",
        "external_completion_qualified": False,
        **p.DENIED,
    }
    if fault == "gradient":
        raw["window_gq"][0, 0, 0] += 0.01
    elif fault == "index":
        raw["window_control_tick"][0, 0] += 1
    else:
        result["metrics"]["full_loss"] += 0.01
    with pytest.raises(ValueError, match="derivatives|index|stored result math"):
        p.independent_check(d, raw, receipt, result)


def test_late_gradient_partial_counts_and_publication(tmp_path):
    t = p.torch_module()
    calls = []

    def fail_second(loss, leaf):
        calls.append(tuple(leaf.shape))
        if len(calls) == 2:
            raise RuntimeError("late window gradient")
        return t.autograd.grad(loss, leaf)

    d = data()
    raw, receipt = p.produce(d, gradient=fail_second)
    assert receipt["full_requested"] == receipt["full_returned"] == 1
    assert receipt["window_requested"] == 1 and receipt["window_returned"] == 0
    assert "window_gq" in receipt["missing"] and "full_gq" in receipt["acquired"]
    output = tmp_path / "partial"
    output.mkdir()
    result, code = p.publish_worker(output, d, raw, receipt)
    assert code == 1 and result["first_error"]["message"] == "late window gradient"
    assert result["H354"] == "not_evaluated" and (output / "raw.npz").exists()
    saved = json.loads((output / "receipt.json").read_text())
    assert saved["acquired"] == sorted(raw) and saved["window_returned"] == 0


def test_late_publication_preserves_counts_and_saved_bytes(tmp_path):
    d = data()
    raw, receipt = p.produce(d)
    output = tmp_path / "late"
    output.mkdir()

    def late():
        if (output / "raw.npz").exists():
            raise TimeoutError("after raw close")

    report, code = p.publish_worker(output, d, raw, receipt, timely=late)
    assert code == 1 and report["late_error_has_precedence"]
    assert (
        report["receipt"]["full_returned"] == report["receipt"]["window_returned"] == 1
    )
    assert (output / "raw.npz").stat().st_size > 0 and not (
        output / "result.json"
    ).exists()
    assert report["H354"] == "not_evaluated"


def test_caps_before_raw_write(tmp_path, monkeypatch):
    d = data()
    raw, receipt = p.produce(d)
    monkeypatch.setattr(p, "CAP", 500)
    with pytest.raises(ValueError, match="failure64KiB"):
        p.publish_worker(tmp_path, d, raw, receipt)
    assert not (tmp_path / "raw.npz").exists()


def selected_fixture(root, name, values, selected):
    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    payload = stream.getvalue()
    (root / name).write_bytes(payload)
    members = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        for key in selected:
            info = z.getinfo(key + ".npy")
            with z.open(info) as child:
                version = np.lib.format.read_magic(child)
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(child)
                members[key] = {
                    "member": info.filename,
                    "shape": list(shape),
                    "dtype": str(dtype),
                    "fortran_order": fortran,
                    "npy_version": list(version),
                    "header_bytes": child.tell(),
                    "zip_compression": info.compress_type,
                    "compressed_bytes": info.compress_size,
                    "uncompressed_bytes": info.file_size,
                    "crc32": info.CRC,
                }
    return dict(p.descriptor(name, payload), selected_members=members)


def test_selected7_extras_unopened_cache_and_denied_preread(tmp_path):
    d = data()
    q, target = (
        np.vstack([np.full((1, 42), 9.0), d["source_q"]]),
        np.vstack([np.full((1, 42), 9.0), d["target_q"]]),
    )
    tick, clock = np.arange(301, dtype=np.int64), np.arange(301) * 0.005
    payloads = {
        "control": selected_fixture(
            tmp_path,
            "c.npz",
            {
                "command": q,
                "control_tick": tick,
                "physics_time_s": clock,
                "unopened": np.array([np.nan]),
            },
            ("command", "control_tick", "physics_time_s"),
        ),
        "reference": selected_fixture(
            tmp_path,
            "r.npz",
            {"normalization": d["normalization"], "unopened": np.array([np.inf])},
            ("normalization",),
        ),
        "cadence": selected_fixture(
            tmp_path,
            "t.npz",
            {
                "command": target,
                "control_tick": tick,
                "control_time_s": clock,
                "unopened": np.array([1]),
            },
            ("command", "control_tick", "control_time_s"),
        ),
    }
    reader = p.generated_reader(tmp_path)
    ctx = {
        "authorized": False,
        "reader": reader,
        "registration": {"payloads": payloads},
    }
    with pytest.raises(ValueError, match="own private gate"):
        p.load_inputs(ctx)
    assert reader.read_counts == {}
    ctx["authorized"] = True
    loaded = p.load_inputs(ctx)
    assert all(np.array_equal(loaded[k], d[k]) for k in d)
    p.load_inputs(ctx)
    assert set(reader.read_counts.values()) == {1} and len(reader.opened_members) == 14
    assert all(member != "unopened.npy" for _, member in reader.opened_members)
    denied = p.helper().EvidenceReader(tmp_path)
    with pytest.raises(ValueError, match="private gate"):
        denied.npz(
            payloads["control"],
            {"command": ((301, 42), "float64")},
            selected=payloads["control"]["selected_members"],
        )


def test_nonzero_completed_worker_before_newraw():
    reader = types.SimpleNamespace(
        npz=lambda *a, **k: pytest.fail("raw preread"),
        json=lambda *a: pytest.fail("JSON preread"),
    )
    with pytest.raises(ValueError, match="exit0 before newraw"):
        p.audit_worker(
            {
                "authorized": True,
                "mode": "audit",
                "reader": reader,
                "worker_seal": {"expected_exit_code": 1},
            },
            None,
        )


def test_passive_import_help_and_gate_metadata_no_original_reads(monkeypatch, capsys):
    loaded = runpy.run_path(str(p.ROOT / p.OWNED[0]), run_name="e354_passive")
    assert loaded["_HELPER"] is None
    monkeypatch.setattr(
        p, "torch_module", lambda: pytest.fail("Torch imported on help")
    )
    with pytest.raises(SystemExit) as ended:
        p.main(["--help"])
    assert ended.value.code == 0 and "offline" in capsys.readouterr().out
    with pytest.raises(ValueError, match="sole CLI"):
        p.main(["--run", "--freeze", "f", "--review", "r"])
    original = p.Path.open

    def guarded(self, *a, **k):
        assert self.suffix not in (".npz", ".npy", ".gz")
        return original(self, *a, **k)

    monkeypatch.setattr(p.Path, "open", guarded)
    ctx = p.metadata_preflight()
    assert not ctx["authorized"] and not ctx["reader"].array_authorized
    with pytest.raises(FileNotFoundError):
        p.gate(
            "run", "missing-e354-freeze.json", "missing-review.json", output=p.OUTPUT
        )


def test_primary_pytest_role_linked_generated_qualification(tmp_path, monkeypatch):
    # Historical process JSON supplies typed structure only; this altered in-memory
    # fixture is NOT a current execution proof and never authorizes actual inputs.
    prefix = p.BASE + "/parent-e354-sol-FIRST-04"
    full = json.loads((p.ROOT / (prefix + "-full-exit.json")).read_text())
    execution = {k: full[k] for k in ("contract", "supervision", "identity")}
    execution["full_exit"] = p.descriptor(prefix + "-full-exit.json", b"fixture-only")
    docs = {
        s["path"]: json.loads((p.ROOT / s["path"]).read_text())
        for s in [
            *execution.values(),
            full["outer_witness"],
            full["actual_tool_return"],
        ]
    }
    own = p.own_sources()
    contract = docs[execution["contract"]["path"]]
    supervision = docs[execution["supervision"]["path"]]
    docs[execution["full_exit"]["path"]]["source_sha256"] = own
    contract["source_sha256"].update(own)
    argv = list(contract["commands"][0]["argv"])
    q = {
        "schema": "flylab.m3.e354.root-qualification.v1",
        "source_sha256": own,
        "passed": True,
        "original_calls": 0,
        "checks": dict.fromkeys(p.CHECKS, True),
        "execution": execution,
    }
    (tmp_path / "q.json").write_bytes(p.json_bytes(q))
    docs["q.json"] = q
    reader = types.SimpleNamespace(
        json=lambda s: docs[s["path"]],
        bytes=lambda s: p.helper().safe(s["path"], tmp_path).read_bytes(),
    )
    context = {"reader": reader, "registration": {"design": {}, "consumed_sources": {}}}
    monkeypatch.setattr(p, "own_sources", lambda root=p.ROOT: own)
    assert p.make_freeze("q.json", tmp_path, context)["original_authorized"] is False
    bad = [
        [],
        argv[:4] + ["-c", "import time; time.sleep(.15)"],
        argv[:9],
        argv[:9] + ["tests/unrelated.py"],
        argv + [p.OWNED[1]],
        argv + ["--collect-only"],
        argv + ["-k", "small"],
        argv + ["--deselect", p.OWNED[1] + "::test_caps_before_raw_write"],
        argv[:9] + [p.OWNED[1], "--ignore", "tests"],
        argv[:9] + [p.OWNED[1] + "::test_caps_before_raw_write"],
        argv[:9] + ["--basetemp", str(p.ROOT), p.OWNED[1]],
        argv + [p.BASE + "/parent-e354-root-unregistered.py"],
    ]
    for actual in bad:
        contract["commands"] = [{"argv": actual}]
        supervision["attempts"][0]["argv"] = actual
        with pytest.raises(
            ValueError, match="pytest role|owned current|registered complete|basetemp"
        ):
            p.make_freeze("q.json", tmp_path, context)
    for commands in ([], [{"argv": argv}, {"argv": argv}]):
        contract["commands"] = commands
        supervision["attempts"] = [
            {"argv": c["argv"], "exit_code": 0, "killed": False} for c in commands
        ]
        with pytest.raises(ValueError, match="sole primary"):
            p.make_freeze("q.json", tmp_path, context)
    contract["commands"] = [{"argv": argv}]
    supervision["attempts"] = [{"argv": argv, "exit_code": 0, "killed": False}]
    contract["source_sha256"][p.OWNED[1]] = "stale"
    with pytest.raises(ValueError, match="current supervised"):
        p.make_freeze("q.json", tmp_path, context)
    contract["source_sha256"].update(own)
    extra = p.BASE + "/parent-e354-root-generated-fixture.py"
    file = tmp_path / extra
    file.parent.mkdir(parents=True)
    file.write_bytes(b"def test_generated_extra():\n    assert True\n")
    spec = p.descriptor(extra, file.read_bytes())
    q["extra_cpu_tests"] = [spec]
    contract["source_sha256"][extra] = spec["sha256"]
    contract["commands"][0]["argv"] = argv + [extra]
    supervision["attempts"][0]["argv"] = argv + [extra]
    assert p.make_freeze("q.json", tmp_path, context)["original_authorized"] is False
    q["extra_cpu_tests"] = [dict(spec, sha256="0" * 64)]
    with pytest.raises(ValueError, match="current extra"):
        p.make_freeze("q.json", tmp_path, context)
