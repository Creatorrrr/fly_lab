"""Small E350 dispatch/publication spies; no real constructor/native physics."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import verify_c_banc_m3_source_pose_body as p


def registration():
    return {
        "force_policy_value": {
            "schema": "flylab.m3.e349.held-force-policy.v1",
            "historically_sampled_per_row": False,
            "recorded_origin": {"push_left": 0.0, "push_force": [0.0] * 3},
        },
        "observation_policy_value": {"schema": "flylab.m3.e350.source-pose-policy.v1"},
    }


class FakeMJ:
    mjtState = SimpleNamespace(mjSTATE_INTEGRATION=1)

    def __init__(self, current=0.96, fail=False):
        self.current, self.fail, self.forwarded = current, fail, []

    @staticmethod
    def MjData(model):
        return SimpleNamespace(upright=0.8)

    @staticmethod
    def mj_stateSize(*args):
        return 1

    @staticmethod
    def mj_getState(model, data, state, typ):
        state[0] = 1.0

    @staticmethod
    def mj_setState(*args):
        pass

    def mj_forward(self, model, data):
        self.forwarded.append(data)
        if self.fail:
            raise OSError("generated current forward unreadable")
        data.upright = self.current


class FakeBody:
    def __init__(self, *, current=0.96, fail=False, live=0.98):
        self.m, self.mj = object(), FakeMJ(current, fail)
        self.d = SimpleNamespace(upright=live, qpos=np.zeros(1), qvel=np.zeros(1))
        self.clock, self.fault = 0.0, None
        self._step_physics = self.step
        original = self.pose
        self.pose = lambda: original()  # Historical live-bound closure copied.

    def step(self):
        self.clock += 0.0001

    def pose(self):
        return None, np.diag([1.0, 1.0, self.d.upright]), None

    def physics_time(self):
        return self.clock

    def nonfoot_contact(self):
        return False


def dependencies(tmp_path, body=None):
    reg = registration()
    dep = p.SourcePoseDependencies(
        reg["force_policy_value"],
        tmp_path,
        reg["observation_policy_value"],
        contact_reader=lambda shadow: {
            "normal_load_bw": [0.1] * 6,
            "slip_mm_s": [None] * 6,
        },
    )
    if body is not None:
        dep._body_arm[id(body)] = "B"
    return dep


def test_actual_instance_install_dispatch_live_and_class_shadow(tmp_path):
    body = FakeBody()
    dep = dependencies(tmp_path, body)
    assert dep.install.__func__ is p.SourcePoseDependencies.install_source_pose
    assert dep.action.__func__ is p.force().HeldForceDependencies.held_action
    samples = []
    original = body._step_physics
    detach = dep.install(body, samples.append)
    body._step_physics()
    body._step_physics()
    detach()
    assert body._step_physics == original
    rows = dep.observations["B"]
    assert [r["native_index"] for r in rows] == [1, 2]
    assert [r["time_s"] for r in rows] == [0.0001, 0.0002]
    assert [r["source_upright"] for r in rows] == [0.98, 0.98]
    assert [r["current_upright"] for r in rows] == [0.96, 0.96]
    assert all(r["callback_returned"] for r in rows)
    assert [r["upright"] for r in samples] == [0.98, 0.98]
    assert all(data is not body.d for data in body.mj.forwarded)


@pytest.mark.parametrize("fault", ["low", "nonfinite", "read", "callback", "source"])
def test_last_acquired_sample_before_health_or_read_exception(tmp_path, fault):
    body = FakeBody(
        current=0.89 if fault == "low" else np.nan if fault == "nonfinite" else 0.96,
        fail=fault == "read",
        live=np.nan if fault == "source" else 0.98,
    )
    dep = dependencies(tmp_path, body)
    samples = []

    def callback(value):
        samples.append(value)
        if fault == "callback":
            raise p.shared().HealthStop("generated callback physical stop")

    detach = dep.install(body, callback)
    with pytest.raises(
        p.shared().HealthStop if fault in ("low", "callback") else ValueError
    ):
        body._step_physics()
    detach()
    assert len(dep.observations["B"]) == 1
    row = dep.observations["B"][0]
    assert row["native_index"] == 1 and row["time_s"] == 0.0001
    if fault in ("nonfinite", "read"):
        assert row["source_available"] and row["source_upright"] == 0.98
        assert not row["current_available"] and row["current_upright"] is None
        assert row["errors"] and samples == []
    elif fault == "source":
        assert not row["source_available"] and row["source_upright"] is None
        assert row["current_available"] and row["current_upright"] == 0.96
        assert samples == []
    else:
        assert row["current_available"] and len(samples) == 1
    assert len(body.mj.forwarded) == 1


def observed(dep, arm="B"):
    dep.observations[arm] = [
        {
            "native_index": 1,
            "time_s": 0.0001,
            "source_upright": 0.98,
            "current_upright": 0.96,
            "source_available": True,
            "current_available": True,
            "time_available": True,
            "callback_attempted": True,
            "callback_returned": True,
            "errors": [],
        }
    ]


def test_raw_schema_descriptor_actual_first_last(tmp_path):
    dep = dependencies(tmp_path)
    observed(dep)
    spec = p.publish_observations(dep, "B", tmp_path, p.shared().publish)
    assert spec == p.descriptor(tmp_path / "source-pose.json")
    metadata = json.loads((tmp_path / "source-pose.json").read_text())
    assert (
        metadata["count"] == 1
        and metadata["first"] == metadata["last"] == dep.observations["B"][0]
    )
    assert metadata["current_health_passed"] and metadata["recorded_health_passed"]
    assert metadata["raw"] == p.descriptor(tmp_path / "source-pose.npz")
    with np.load(tmp_path / "source-pose.npz", allow_pickle=False) as raw:
        assert raw["native_index"].dtype == np.dtype("int64")
        assert raw["current_upright"].dtype == np.dtype("float64")
        assert raw["current_available"].dtype == np.dtype(bool)
        assert raw["native_index"].tolist() == [1]
        assert raw["time_s"].tolist() == [0.0001]


def test_missing_current_is_masked_actual_error_not_zero(tmp_path):
    dep = dependencies(tmp_path)
    observed(dep)
    dep.observations["B"][0].update(
        current_available=False,
        current_upright=None,
        errors=[{"stage": "current_read", "type": "OSError", "message": "spy"}],
    )
    p.publish_observations(dep, "B", tmp_path, p.shared().publish)
    with np.load(tmp_path / "source-pose.npz", allow_pickle=False) as raw:
        assert np.isnan(raw["current_upright"][0]) and not raw["current_available"][0]
    metadata = json.loads((tmp_path / "source-pose.json").read_text())
    assert metadata["current_available_count"] == 0 and metadata["errors"]
    assert not metadata["current_health_passed"]


def test_compact_exact_paths_force_bytes_and_restore(tmp_path):
    dep = dependencies(tmp_path)
    output = tmp_path / "output"
    folder = output / "B"
    folder.mkdir(parents=True)
    value = {"samples": [{"foo": [1.0, 2.0], "neural_request": None}]}
    original = p.shared().publish
    with p.publication_scope(output, dep):
        p.shared().publish(folder / "trace.json", value)
        p.shared().publish(
            output / "force-policy.json", registration()["force_policy_value"]
        )
    assert p.shared().publish is original
    assert (folder / "trace.json").read_bytes() == json.dumps(
        value, separators=(",", ":"), allow_nan=False
    ).encode()
    assert (output / "force-policy.json").read_bytes() == json.dumps(
        registration()["force_policy_value"], indent=2, allow_nan=False
    ).encode()
    assert json.loads((folder / "trace.json").read_text()) == value
    with pytest.raises(FileExistsError), p.publication_scope(output, dep):
        p.shared().publish(folder / "trace.json", value)
    assert p.shared().publish is original


@pytest.mark.parametrize("stage", ["receipt", "evaluation"])
def test_inclusive_cap_before_next_arm_and_restore(tmp_path, monkeypatch, stage):
    core = p.shared()
    dep = dependencies(tmp_path)
    output = tmp_path / "output"
    folder = output / "B"
    folder.mkdir(parents=True)
    observed(dep)
    monkeypatch.setitem(core.LIMITS, "raw_arm_bytes", 4000)
    monkeypatch.setitem(core.LIMITS, "failure_bytes", 4000)
    ledger = {"status": "completed", "first_error": None, "errors": []}
    original = core.publish
    with (
        pytest.raises(ValueError, match="cap exceeded"),
        p.publication_scope(output, dep),
    ):
        if stage == "receipt":
            (folder / "trace.json").write_bytes(b"x" * 4000)
        core.publish(folder / "receipt.json", ledger)
        if stage == "evaluation":
            core.publish(folder / "evaluation.json", {"large": "x" * 4000})
        (output / "I").mkdir()  # Must never be reached after limit violation.
    assert not (output / "I").exists() and core.publish is original
    assert (folder / "source-pose.npz").exists() and (folder / "receipt.json").exists()
    assert ledger["source_pose_observation"] == p.descriptor(
        folder / "source-pose.json"
    )


def test_real_core_evaluation_cap_stops_roster_preserves_counts(tmp_path, monkeypatch):
    core = p.shared()
    dep = dependencies(tmp_path)
    output = tmp_path / "output"
    observed(dep)
    monkeypatch.setitem(core.LIMITS, "raw_arm_bytes", 6000)
    monkeypatch.setitem(core.LIMITS, "failure_bytes", 6000)
    monkeypatch.setattr(
        core,
        "plan",
        lambda *args: (
            {a: np.zeros((1, 42)) for a in ("B", "I", "C")},
            {a: np.zeros(1) for a in ("z", "R", "phi")},
        ),
    )
    called = []

    def replay(arm, command, data, dependencies, folder, **kw):
        called.append(arm)
        ledger = {
            "status": "completed",
            "first_error": None,
            "errors": [],
            "controls_requested": 1,
            "controls_returned": 1,
            "native_samples_acquired": 1,
            "configuration": {
                key: "spy"
                for key in (
                    "physics_identity",
                    "joint_names",
                    "config",
                    "body_options",
                    "physics_profile",
                    "model_hash",
                )
            },
        }
        core.publish(folder / "receipt.json", ledger)
        return ledger, {}, {}

    monkeypatch.setattr(core, "replay", replay)

    def engine(data, output, **kw):
        return core.run(
            data, output, evaluator=lambda *args, **kwargs: {"large": "x" * 7000}, **kw
        )

    result = p.run_loaded(
        {"control": {"command": None}, "reference": {}},
        output,
        registration(),
        dependencies=dep,
        engine_run=engine,
    )
    assert called == ["B"] and not (output / "I").exists()
    assert (
        result["status"] == "technical_invalid"
        and result["first_error"]["stage"] == "evaluation_publication"
    )
    assert result["arms"]["B"]["controls_requested"] == 1
    assert result["arms"]["B"]["native_samples_acquired"] == 1
    assert any(e["stage"] == "source_pose_finalization" for e in result["errors"])
    assert json.loads((output / "result.json").read_text()) == result
    assert (
        json.loads((output / "B" / "receipt.json").read_text()) == result["arms"]["B"]
    )


def test_new_raw_publication_failure_restores_publisher(tmp_path, monkeypatch):
    output = tmp_path / "output"
    folder = output / "B"
    folder.mkdir(parents=True)
    dep = dependencies(tmp_path)
    observed(dep)
    original = p.shared().publish

    def failing(file, value):
        if file.name == "source-pose.json":
            raise OSError("generated observation receipt sink failure")
        return original(file, value)

    monkeypatch.setattr(p.shared(), "publish", failing)
    with pytest.raises(OSError, match="sink failure"), p.publication_scope(output, dep):
        p.shared().publish(folder / "receipt.json", {"status": "technical_invalid"})
    assert p.shared().publish is failing and (folder / "source-pose.npz").exists()
    assert len(dep.observations["B"]) == 1


def test_actual_metadata_only_and_old_gate_never_called(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("original payload/body/gate was touched")

    monkeypatch.setattr(p.shared(), "selected", refuse)
    monkeypatch.setattr(p.force(), "original_gate", refuse)
    reg = p.metadata_preflight()
    assert reg["new_observer_publisher_gate_audit_inherited_native_proof"] is False
    assert len(reg["inherited_native_routine_AST_sha256"]) == 8


def test_loader_delegates_without_original_read(monkeypatch):
    sentinel = object()
    calls = []
    monkeypatch.setattr(
        p.force(),
        "load_original",
        lambda reg, **kw: (calls.append((reg, kw)), sentinel)[1],
    )
    reg = registration()
    assert p.load_original(reg, root=p.ROOT) is sentinel
    assert calls[0][0] is reg


def test_private_gate_fails_before_payload_and_sole_argv(tmp_path, monkeypatch):
    (tmp_path / "freeze.json").write_text(
        json.dumps({"qualification": {"path": "q.json"}})
    )
    monkeypatch.setattr(p, "metadata_preflight", lambda root: registration())
    monkeypatch.setattr(p, "make_freeze", lambda *args, **kw: {"different": True})
    with pytest.raises(ValueError, match="freeze differs"):
        p.original_gate("freeze.json", "review.json", root=tmp_path)
    assert p.original_argv("F.json", "R.json") == [
        ".venv/Scripts/python.exe",
        "-X",
        "utf8",
        "-B",
        p.OWNED[0],
        "--run",
        "--freeze",
        "F.json",
        "--review",
        "R.json",
        "--output",
        p.OUTPUT,
    ]


def test_new_E349_trace_compact_decoded_identity(tmp_path):
    # Authorized newly generated E349 body JSON only, no source/original arrays.
    file = p.ROOT / p.BASE / "e349-held-force-body/B/trace.json"
    value = json.loads(file.read_text("utf-8"))
    original = copy.deepcopy(value)
    raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    assert json.loads(raw) == original == value
    assert len(raw) < 4 << 20
    (tmp_path / "compact-proof.json").write_text(
        json.dumps(
            {
                "source": p.descriptor(
                    file, relative=p.BASE + "/e349-held-force-body/B/trace.json"
                ),
                "compact_bytes": len(raw),
                "decoded_identity": True,
                "original_NPZ_NPY_access": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
