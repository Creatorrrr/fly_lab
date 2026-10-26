"""E353 generated-only fake R loop, exact publication, independent audit and gates."""

import copy
import gzip
import json
import math
import runpy
import types
from pathlib import Path

import numpy as np
import pytest

from tools import probe_c_banc_m3_reference_cadence_body as p


def fixture():
    theta = np.arange(128) * math.tau / 128
    basis = np.c_[np.cos(theta), np.sin(theta)]
    curves = np.zeros((6, 128, 7))
    for leg in range(6):
        curves[leg, :, 0] = np.cos(theta)
        curves[leg, :, 1] = np.sin(theta)
    q = np.tile(curves[:, 20].reshape(42), (301, 1))
    q[0] = 9
    control = {
        "command": q,
        "tendon": np.zeros((301, 6)),
        "control_tick": np.arange(301, dtype=np.int64),
        "physics_time_s": np.arange(301) * 0.005,
    }
    reference = {
        "curves": curves,
        "phase_basis": basis,
        "temperature": np.full(6, 0.02),
        "normalization": np.ones(6),
        "future_anchors": curves.copy(),
    }
    native = {
        "time_s": np.arange(1, 15001) * 0.0001,
        "normal_load_bw": np.full((15000, 6), 0.1),
        "slip_mm_s": np.ones((15000, 6)),
        "slip_valid": np.ones((15000, 6), bool),
        "upright": np.ones(15000),
        "nonfoot_contact": np.zeros(15000, bool),
    }
    rows = []
    for c in range(301):
        t = c * 0.005
        origin = np.array([2 * t, 0.0, 0.0])
        feet = {}
        for leg in p.LEGS:
            points = np.c_[np.arange(8) * 0.1, np.zeros(8), np.arange(8) * 0.01]
            points[-1, 0] += 0.2 * math.cos(2 * math.pi * 10 * t)
            points[2, 2] += 0.03
            feet[leg] = (points + origin).tolist()
        rows.append(
            {
                "control_tick": c,
                "time_s": t,
                "physics_time_s": t,
                "qpos": [0.0] * 4,
                "qvel": [0.0] * 3,
                "ctrl": [0.0] * 42,
                "joint_targets_rad": ([0.0] * 42 if c == 0 else q[c].tolist()),
                "adhesion": [True] * 6,
                "tendon_inputs": {
                    **{leg + "_tarsus": 0.0 for leg in p.LEGS},
                    "abdomen_pitch": 0.0,
                    "abdomen_yaw": 0.0,
                },
                "neural_request": None,
                "fault": None,
                "body": {
                    "position": origin.tolist(),
                    "basis": np.eye(3).tolist(),
                    "yaw": 0.0,
                    "legs": feet,
                },
                "feet": {"load_bearing": [True] * 6},
                "upright": 1.0,
            }
        )
    metadata = {
        "schema": "generated",
        "samples": rows,
        "body_hash": "1b2a89084311ebd2ccbccbe2c57f2ec29fa937e115bb6cd4c7f6cd16ba7c2e6f",
        "model_hash": "label",
        "adapter_hash": "label",
        "graph_hash": "label",
        "parameters": {},
    }
    return {
        "control": control,
        "reference": reference,
        "native": native,
        "rows": rows,
        "metadata": metadata,
        "body_state": {},
        "body_metadata": {},
    }


class Counter:
    def __init__(self, body, stage, **kw):
        self.proof = {
            "steps_requested": 0,
            "steps_returned": 0,
            "absolute_clock_known": True,
            "actual_advance_s": 0.0,
            "method_restored": False,
        }
        body.counter = self

    def restore(self):
        self.proof["method_restored"] = True


class FakeDeps:
    def __init__(self, data, output, reg, health_at=None, prefix_fail=False):
        self.data = data
        self.output = output
        self.policy = reg["force_policy_value"]
        self.policy_sha256 = p.sha(json.dumps(self.policy, indent=2).encode())
        self.observation_policy = reg["observation_policy_value"]
        self.observations = {"B": []}
        self.actions = []
        self.before_read_errors = []
        self.action_adapter_requested = self.native_action_requested = (
            self.native_action_returned
        ) = 0
        self.health_at = health_at
        self.prefix_fail = prefix_fail
        self.compare_calls = []

        def arrays(samples):
            return {
                k: np.array([s[k] for s in samples], dtype=v.dtype).reshape(
                    (len(samples),) + v.shape[1:]
                )
                for k, v in data["native"].items()
            }

        def compare(rows, native, source, original, c):
            self.compare_calls.append(c)
            assert rows == source[: c + 1]
            for k, v in native.items():
                assert np.array_equal(v, original[k][: c * 50])
            if c == 100 and self.prefix_fail:
                raise ValueError("generated prefix mismatch")

        self.static = types.SimpleNamespace(
            native_arrays=arrays,
            verify_native=lambda values, **kw: None,
            typed=lambda v, s, *args: p.require(v.shape == s, "geometry shape"),
            compare_replay=compare,
        )

    def body_class(self, data):
        p.primitives().publish(self.output / "force-policy.json", self.policy)
        deps = self

        class Body:
            def __init__(self):
                self.counter.proof.update(
                    steps_requested=2000, steps_returned=2000, actual_advance_s=0.2
                )
                self.tick = 0
                self.native = 0
                self.command = np.zeros(42)

            def restore(self, state):
                self.tick = self.native = 0

            def cpg_state(self):
                return {"never_advanced": True}

            def physics_time(self):
                return self.native * 0.0001

            def close(self):
                deps.closed = True

        return Body

    counter = Counter

    def body_arguments(self, data):
        return (), {}

    def observed_configuration(self, body):
        return {"generated": True}

    def configure(self, body, data):
        return {"generated": True}

    def install(self, body, callback):
        self.callback = callback
        return lambda: None

    def action(self, body, row, command):
        self.action_adapter_requested += 1
        self.native_action_requested += 1
        event = {
            "arm": "B",
            "control_tick": row["control_tick"],
            "before": {"push_left": 0.0, "push_force": [0.0, 0.0, 0.0]},
            "after": {"push_left": 0.0, "push_force": [0.0, 0.0, 0.0]},
            "returned": False,
            "error": None,
        }
        self.actions.append(event)
        for _ in range(50):
            body.native += 1
            body.counter.proof["steps_requested"] += 1
            body.counter.proof["steps_returned"] += 1
            body.counter.proof["actual_advance_s"] = body.native * 0.0001
            record = {
                "native_index": body.native,
                "time_s": body.native * 0.0001,
                "source_upright": 1.0,
                "current_upright": 1.0,
                "time_available": True,
                "source_available": True,
                "current_available": True,
                "callback_attempted": True,
                "callback_returned": False,
                "errors": [],
            }
            self.observations["B"].append(record)
            sample = {
                k: v[body.native - 1].tolist() for k, v in self.data["native"].items()
            }
            self.callback(sample)
            record["callback_returned"] = True
            if self.health_at == body.native:
                record["current_upright"] = 0.8
                event["error"] = {"type": "HealthStop"}
                raise self.health_type("generated finite health")
        body.tick = row["control_tick"]
        body.command = command.copy()
        event["returned"] = True
        self.native_action_returned += 1

    def boundary(self, body, source):
        row = copy.deepcopy(source)
        row["joint_targets_rad"] = body.command.tolist()
        return row


@pytest.fixture
def seam(monkeypatch):
    real = p.pose()
    health = real.shared().HealthStop

    def trace(metadata, rows):
        return dict(copy.deepcopy(metadata), samples=rows)

    core = types.SimpleNamespace(HealthStop=health, body_trace=trace)
    fake = types.SimpleNamespace(shared=lambda: core, error_record=real.error_record)
    monkeypatch.setattr(p, "pose", lambda: fake)
    return core, health


def reg():
    return p.primitives().strict_json((p.ROOT / p.REGISTRATION).read_bytes())


def test_plan_independent_scalar_wrap_and_literal_row0():
    d = fixture()
    plan = p.build_plan(d["control"], d["reference"])
    verified = p.audit_plan(d["control"], d["reference"], plan)
    assert verified["maximum_command_error_rad"] <= 2e-12
    assert np.array_equal(plan["command"][:101], d["control"]["command"][:101])
    assert not np.array_equal(plan["command"][0], d["rows"][0]["joint_targets_rad"])
    assert (
        plan["blend_weight"][101] > 0
        and plan["blend_weight"][119] < 1
        and plan["blend_weight"][120] == pytest.approx(1)
    )
    assert np.count_nonzero(np.diff(plan["phi_R"]) < 0) >= 2
    plan["command"][170, 0] += 0.001
    with pytest.raises(ValueError, match="interpolation"):
        p.audit_plan(d["control"], d["reference"], plan)


def test_full_fake_loop_pack_read_audit(tmp_path, seam, monkeypatch):
    d = fixture()
    registration = reg()
    plan = p.build_plan(d["control"], d["reference"])
    output = tmp_path / p.OUTPUT
    output.mkdir(parents=True)
    deps = FakeDeps(d, output, registration)
    deps.health_type = seam[1]
    # New generated original3/cache/header/finite dossier loader, no actual inputs.
    source_root = tmp_path.parent / "f3"
    source_root.mkdir()
    reader0 = p.generated_reader(source_root)
    registration0 = copy.deepcopy(registration)
    registration0["payloads"] = {}
    for role in ("control", "native", "reference"):
        raw, members = p.npz_pack(d[role])
        (source_root / (role + ".npz")).write_bytes(raw)
        registration0["payloads"][role] = dict(
            p.descriptor(role + ".npz", raw), selected_members=members
        )
    finite = []
    documents = {
        "e335-smooth-secant-body/observed-start/state.json": {"body": {}},
        "e335-smooth-secant-body/trace.json": copy.deepcopy(d["metadata"]),
        "parent-e337-current-body-metadata.json": {},
        "e335-smooth-secant-body/observed-start/manifest.json": {"versions": "fixture"},
    }
    for row in documents["e335-smooth-secant-body/trace.json"]["samples"]:
        del row["physics_time_s"]
    for tail, value in documents.items():
        name = p.BASE + "/" + tail
        file = source_root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        raw = p.json_bytes(value)
        file.write_bytes(raw)
        finite.append(p.descriptor(name, raw))
    registration0["finite_JSON_dossier"] = finite
    deps.static._versions = lambda: "fixture"
    seam[0].libraries = lambda: (np, deps.static, None)
    fakepose = p.pose()
    fakepose.force = lambda: types.SimpleNamespace(
        validate_inputs=lambda state, trace, meta, policy: trace["samples"]
    )
    decoded = p.load_inputs(
        {"authorized": True, "registration": registration0, "reader": reader0}
    )
    assert np.array_equal(decoded["control"]["command"], d["control"]["command"])
    assert [row["physics_time_s"] for row in decoded["rows"]] == d["control"][
        "physics_time_s"
    ].tolist()
    assert decoded["rows"][0]["joint_targets_rad"] == [0.0] * 42
    assert all(n == 1 for n in reader0.read_counts.values())
    receipt, trace, native, observed = p.loop(decoded, plan, deps)
    assert (
        receipt["status"] == "completed"
        and deps.compare_calls == [0, 100]
        and len(deps.actions) == 300
    )
    assert trace["samples"][0]["joint_targets_rad"] == d["rows"][0]["joint_targets_rad"]
    result, code = p.publish_worker(
        output, receipt, trace, native, observed, plan, registration
    )
    assert code == 0 and p.primitives().usage(output) < p.CAP
    assert (
        min(v["complete_cycles"] for v in result["metrics"]["gait"]["events"].values())
        >= 8
    )
    assert {f.name for f in output.iterdir()} == set(
        registration["worker_output_roster"]
    ) - {"failure.json"}
    reader = p.generated_reader(tmp_path)
    roster = {f.name: p.descriptor(f.name, f.read_bytes()) for f in output.iterdir()}
    seam[0].libraries = lambda: (np, deps.static, None)
    ctx = {
        "authorized": True,
        "mode": "audit",
        "registration": registration,
        "reader": reader,
        "worker_seal": {"mode": "full", "artifacts": roster, "expected_exit_code": 0},
    }
    audit = p.audit_worker(ctx, d)
    assert audit["audit_passed"] is True and audit["metrics"] == p.primitives().lower(
        result["metrics"]
    )
    assert all(n == 1 for n in reader.read_counts.values())
    raw = (output / "trace.json.gz").read_bytes()
    assert p.primitives().strict_json(gzip.decompress(raw)) == p.primitives().lower(
        trace
    )


@pytest.mark.parametrize(
    "step,status", [(2500, "technical_invalid"), (5051, "scientific_partial")]
)
def test_dynamic_health_acquired_partial_no_padding(tmp_path, seam, step, status):
    d = fixture()
    output = tmp_path / "partial"
    output.mkdir()
    deps = FakeDeps(d, output, reg(), health_at=step)
    deps.health_type = seam[1]
    receipt, trace, native, observed = p.loop(
        d, p.build_plan(d["control"], d["reference"]), deps
    )
    assert (
        receipt["status"] == status
        and len(native["time_s"]) == step
        and len(observed["native_index"]) == step
        and receipt["controls_unreturned"] == 1
    )
    assert (
        p.evaluate(
            receipt, trace, native, observed, output / "trace.json.gz", "not-saved"
        )["H353"]["verdict"]
        == "not_evaluated"
    )


def test_prefix_failure_never_calls_c101(tmp_path, seam):
    d = fixture()
    output = tmp_path / "prefix"
    output.mkdir()
    deps = FakeDeps(d, output, reg(), prefix_fail=True)
    receipt, *_ = p.loop(d, p.build_plan(d["control"], d["reference"]), deps)
    assert (
        receipt["status"] == "technical_invalid"
        and receipt["controls_requested"] == 100
        and len(deps.actions) == 100
    )


def test_gait_gap_negative_actual_frozen_analyzer():
    d = fixture()
    model = p.primitives().gait_model(
        d["metadata"], Path("generated.json.gz"), "generated"
    )
    from tools import analyze_c_banc_biological_gait as gait

    model["speed"][::40] = 0
    model["moving"] = model["window"] & (model["speed"] > 0.5)
    assert all(
        v["complete_cycles"] == 0 for v in gait.analyze(model)[0]["events"].values()
    )


def test_gzip_crc_and_strict_overflow(tmp_path):
    trace = {"value": 1}
    decoded = p.json_bytes(trace)
    raw = gzip.compress(decoded, mtime=0)
    spec = dict(
        p.descriptor("trace.json.gz", raw),
        decoded_bytes=len(decoded),
        decoded_sha256=p.sha(decoded),
    )
    (tmp_path / "trace.json.gz").write_bytes(raw)
    r = p.primitives().EvidenceReader(tmp_path)
    assert p.read_gzip(r, spec) == trace
    corrupt = raw[:-8] + bytes([raw[-8] ^ 1]) + raw[-7:]
    (tmp_path / "bad.gz").write_bytes(corrupt)
    with pytest.raises(gzip.BadGzipFile):
        p.read_gzip(r, dict(spec, **p.descriptor("bad.gz", corrupt)))
    for bad in [b'{"x":1e999}', b'{"x":NaN}']:
        with pytest.raises(ValueError):
            p.primitives().strict_json(bad)


def test_npz_dtype_crc_and_nan_mask(tmp_path):
    d = fixture()
    d["native"]["slip_mm_s"][0, 0] = np.nan
    d["native"]["slip_valid"][0, 0] = False
    p.validate_native(d["native"])
    raw, members = p.npz_pack(d["native"])
    (tmp_path / "a.npz").write_bytes(raw)
    r = p.generated_reader(tmp_path)
    arrays = r.npz(
        p.descriptor("a.npz", raw),
        {k: (tuple(v["shape"]), v["dtype"]) for k, v in members.items()},
        selected=members,
    )
    p.validate_native(arrays)
    bad = copy.deepcopy(members)
    bad["time_s"]["crc32"] ^= 1
    with pytest.raises(ValueError, match="metadata"):
        r.npz(
            p.descriptor("a.npz", raw),
            {k: (tuple(v["shape"]), v["dtype"]) for k, v in members.items()},
            selected=bad,
        )
    arrays["slip_valid"][0, 0] = True
    with pytest.raises(ValueError, match="mask"):
        p.validate_native(arrays)


def test_reservation_caps_before_first_raw_and_emergency(tmp_path, seam, monkeypatch):
    output = tmp_path / "cap"
    output.mkdir()
    registration = reg()
    p.primitives().publish(
        output / "force-policy.json", registration["force_policy_value"]
    )
    receipt = {"first_error": None, "errors": [], "controls_returned": 0}
    monkeypatch.setattr(
        p, "pack", lambda *a: ({"trace.json.gz": b"x" * (p.CAP + 1)}, {})
    )
    monkeypatch.setattr(p, "evaluate", lambda *a: None)
    result, code = p.publish_worker(output, receipt, {}, None, None, None, registration)
    assert (
        code == 1
        and not (output / "trace.json.gz").exists()
        and (output / "failure.json").stat().st_size <= p.EMERGENCY
    )
    assert all(v["full_saved_proof"] is False for v in result["unpublished"])


def test_late_close_failure_preserves_written_raw(tmp_path, seam, monkeypatch):
    output = tmp_path / "late"
    output.mkdir()
    registration = reg()
    p.primitives().publish(
        output / "force-policy.json", registration["force_policy_value"]
    )
    receipt = {
        "first_error": None,
        "errors": [],
        "controls_returned": 0,
        "status": "technical_invalid",
    }
    monkeypatch.setattr(
        p,
        "pack",
        lambda *a: (
            {"trace.json.gz": b"x"},
            {"trace.json.gz": p.descriptor("trace.json.gz", b"x")},
        ),
    )
    monkeypatch.setattr(p, "evaluate", lambda *a: None)

    def late():
        if (output / "trace.json.gz").exists():
            raise TimeoutError("after close")

    result, code = p.publish_worker(
        output, receipt, {}, None, None, None, registration, timely=late
    )
    assert (
        code == 1
        and (output / "trace.json.gz").read_bytes() == b"x"
        and result["status"] == "technical_invalid"
    )


def test_real_metadata_stat_only_and_own_gate_no_original_read(monkeypatch):
    original = Path.open

    def limited(self, *args, **kwargs):
        assert self.suffix not in (".npz", ".npy", ".gz")
        assert not self.as_posix().endswith("/e335-smooth-secant-body/trace.json")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", limited)
    ctx = p.metadata_preflight()
    assert ctx["authorized"] is False
    with pytest.raises(ValueError, match="gate"):
        p.load_inputs(ctx)
    with pytest.raises(ValueError, match="sole"):
        p.main(["--run", "--freeze", "bad", "--review", "bad"])


def test_standalone_import_and_strict_scalar():
    loaded = runpy.run_path(
        str(p.ROOT / p.OWNED[0]), run_name="e353_generated_standalone"
    )
    assert loaded["ROOT"] == p.ROOT
    assert p.json_bytes({"x": np.bool_(True)}) == b'{"x":true}'
    for value in [np.array(0), np.float64(np.inf), np.longdouble(0.1)]:
        with pytest.raises((ValueError, TypeError)):
            p.json_bytes({"bad": value})


@pytest.mark.parametrize("prior", [None, {"stage": "original", "type": "ValueError"}])
def test_main_final_deadline_keeps_acquired_counts(tmp_path, seam, monkeypatch, prior):
    output = tmp_path / "main-late"
    counts = {
        "controls_requested": 300,
        "controls_returned": 300,
        "boundaries_acquired": 301,
        "native_samples_acquired": 15000,
        "pose_records_acquired": 15000,
        "controls_unreturned": 0,
    }
    receipt = dict(
        counts, status="completed", first_error=prior, errors=[prior] if prior else []
    )
    monkeypatch.setattr(
        p,
        "original_argv",
        lambda *a: [None] * 5 + ["--run", "--freeze", "f", "--review", "r"],
    )
    monkeypatch.setattr(
        p,
        "gate",
        lambda *a, **k: {
            "registration": {
                "force_policy_value": {},
                "observation_policy_value": {},
            }
        },
    )
    isolated = types.SimpleNamespace(**vars(p.primitives()))
    isolated.safe = lambda *a: output
    isolated.EvidenceReader = lambda **k: object()
    monkeypatch.setattr(p, "primitives", lambda: isolated)
    monkeypatch.setattr(p, "load_inputs", lambda *a: {"control": {}, "reference": {}})
    monkeypatch.setattr(p, "build_plan", lambda *a: {})
    p.pose().SourcePoseDependencies = lambda *a: object()
    monkeypatch.setattr(p, "loop", lambda *a, **k: (receipt, None, None, None))
    monkeypatch.setattr(
        p, "time", types.SimpleNamespace(perf_counter=lambda: p.ENTRY + 23)
    )

    def saved(*args, progress, **kwargs):
        raw = b"saved provisional result"
        (output / "result.json").write_bytes(raw)
        spec = p.descriptor("result.json", raw)
        progress.update(staged={"result.json": spec}, published={"result.json": spec})
        return {"status": "completed"}, 0

    monkeypatch.setattr(p, "publish_worker", saved)
    assert p.main(["--run", "--freeze", "f", "--review", "r"]) == 1
    report = json.loads((output / "failure.json").read_text())
    assert report["actual_counts"] == counts
    assert report["counter_status"] == "acquired_loop_receipt"
    assert report["late_error_has_precedence"] is True
    assert report["first_error"] == (prior or report["secondary_errors"][-1])
    assert report["secondary_errors"][-1]["type"] == "TimeoutError"
    assert report["published"][0]["full_saved_proof"] is True
    assert (output / "result.json").read_bytes() == b"saved provisional result"


def test_completed_failed_worker_exit_rejected_before_raw(tmp_path, seam):
    registration = reg()
    roster = {
        name: p.descriptor(name, b"unused")
        for name in registration["worker_output_roster"]
        if name != "failure.json"
    }
    read = []

    def json_only(spec):
        read.append(spec["path"])
        return {"status": "completed"} if spec["path"].endswith("receipt.json") else {}

    ctx = {
        "authorized": True,
        "mode": "audit",
        "registration": registration,
        "reader": types.SimpleNamespace(root=tmp_path, json=json_only),
        "worker_seal": {"mode": "full", "artifacts": roster, "expected_exit_code": 1},
    }
    with pytest.raises(ValueError, match="successful closed exit"):
        p.audit_worker(ctx, None)
    assert read == [p.OUTPUT + "/result.json", p.OUTPUT + "/receipt.json"]
