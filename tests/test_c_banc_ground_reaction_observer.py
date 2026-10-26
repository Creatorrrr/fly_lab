"""Synthetic-only E185 shadow observer and source-gate checks."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

import tools
from tools import observe_c_banc_m3_ground_reaction as e185


@dataclass
class Contact:
    geom1: int
    geom2: int
    wrench: tuple[float, ...]
    frame: np.ndarray
    exclude: int = 0


class FakeMj:
    class mjtState:
        mjSTATE_INTEGRATION = 7

    def __init__(self):
        self.forward_data = []

    def MjData(self, model):
        return SimpleNamespace(
            time=0.0,
            contact=list(model.contacts),
            ncon=len(model.contacts),
            qpos=np.zeros(1),
            qvel=np.zeros(1),
            xpos=model.xpos.copy(),
            xipos=model.xipos.copy(),
        )

    def mj_stateSize(self, model, state):
        assert state == 7
        return 1

    def mj_getState(self, model, data, state, kind):
        state[0] = data.time

    def mj_setState(self, model, data, state, kind):
        data.time = state[0]

    def mj_forward(self, model, data):
        self.forward_data.append(data)

    def mj_contactForce(self, model, data, index, wrench):
        wrench[:] = data.contact[index].wrench


class FakeBody:
    def __init__(self, contacts=None):
        contacts = contacts or []
        self.m = SimpleNamespace(
            ngeom=60,
            nbody=5,
            contacts=contacts,
            body_mass=np.array([0.0, 1.0, 2.0, 3.0, 9.0]),
            body_parentid=np.array([0, 0, 1, 1, 0]),
            geom_bodyid=np.array([2] * 30 + [0, 0, 2, 1] + [0] * 26),
            opt=SimpleNamespace(gravity=np.array([0.0, 0.0, -10.0])),
            xipos=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0],
                    [8.0, 8.0, 8.0],
                ]
            ),
            xpos=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 2.0, 3.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
        )
        self.mj = FakeMj()
        self.d = self.mj.MjData(self.m)
        self.body_ids = np.array([1, 2, 3])
        self.mass = 6.0
        self.weight0 = 60.0
        self.thorax = 1
        self.model_hash = "synthetic"
        self.fault = None
        self.steps = 0
        mapping = np.full(60, -1, dtype=int)
        mapping[:30] = np.arange(30)
        leg_mapping = np.full(60, -1, dtype=int)
        leg_mapping[:30] = np.asarray([i // 5 * 8 + 3 + i % 5 for i in range(30)])
        leg_mapping[32] = 2
        self._force_maps = {"feet": (mapping, 30), "legs": (leg_mapping, 48)}
        self._ground_mask = np.zeros(60, dtype=bool)
        self._ground_mask[30] = True
        self._ground_mask[31] = True
        self.support_geom_ids = {30}

    def _step_physics(self):
        self.steps += 1
        self.d.time += 0.0001

    def physics_time(self):
        return self.d.time

    def pose(self):
        return self.d.xpos[self.thorax], np.eye(3), np.zeros(3)

    def nonfoot_contact(self):
        return False

    def physics_identity(self):
        return {"synthetic": True}


def diagnostics(body):
    return {"normal_load_bw": [0.0] * 6, "slip_mm_s": [None] * 6}


def contact(a, b, force, frame=None, exclude=0):
    return Contact(
        a,
        b,
        tuple(force) + (0.0, 0.0, 0.0),
        np.eye(3) if frame is None else np.asarray(frame),
        exclude,
    )


def test_force_sign_frame_filters_sums_projections_and_com():
    frame = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    body = FakeBody(
        [
            contact(0, 30, (2.0, 0.0, 0.0), frame),
            contact(30, 5, (0.0, 3.0, 0.0)),
            contact(1, 31, (500.0, 0.0, 0.0)),
            contact(2, 30, (500.0, 0.0, 0.0), exclude=1),
        ]
    )
    inventory = e185.fly_mass_inventory(body)
    row = e185.shadow_force_sample(body, inventory, np.eye(3))
    assert np.array_equal(row["segment_force"][0], [0.0, -2.0, 0.0])
    assert np.array_equal(row["segment_force"][5], [0.0, 3.0, 0.0])
    assert np.array_equal(row["leg_force"][0], [0.0, -2.0, 0.0])
    assert np.array_equal(row["leg_force"][1], [0.0, 3.0, 0.0])
    assert np.count_nonzero(row["segment_force"]) == 2
    assert np.array_equal(row["leg_forward_force"], np.zeros(6))
    assert np.array_equal(row["leg_heading_xy_force"], np.zeros(6))
    assert np.allclose(row["com_position"], [1.0, 1.5, 0.0])
    assert np.array_equal(row["segment_force_bw"][0], row["segment_force"][0] / 60.0)


@pytest.mark.parametrize(
    "ids, mass, match",
    [
        ([1, 2, 2], 5.0, "IDs"),
        ([1, 2, 5], 3.0, "IDs"),
        ([1, 2], 3.0, "missing"),
        ([1, 2, 3], 7.0, "mass"),
    ],
)
def test_mass_rejects_missing_duplicate_out_of_range(ids, mass, match):
    body = FakeBody()
    body.body_ids = np.asarray(ids)
    body.mass = mass
    with pytest.raises(ValueError, match=match):
        e185.fly_mass_inventory(body)


def test_tibia_and_other_body_floor_force_reveal_foot_coverage_gap():
    body = FakeBody(
        [contact(32, 30, (4.0, 0.0, 0.0)), contact(33, 30, (3.0, 0.0, 0.0))]
    )
    row = e185.shadow_force_sample(body, e185.fly_mass_inventory(body), np.eye(3))
    assert not row["segment_force"].any()
    assert np.array_equal(row["all_leg_segment_force"][2], [-4.0, 0.0, 0.0])
    assert np.array_equal(row["all_leg_force"][0], [-4.0, 0.0, 0.0])
    assert np.array_equal(row["full_fly_floor_force"], [-7.0, 0.0, 0.0])
    assert np.array_equal(row["nonleg_floor_force_residual"], [-3.0, 0.0, 0.0])
    assert row["full_fly_forward_force"] == -7.0
    assert row["all_leg_heading_xy_force"][0] == -4.0


def test_mass_rejects_weight_and_gravity_errors():
    body = FakeBody()
    body.weight0 = 600.0
    with pytest.raises(ValueError, match="weight"):
        e185.fly_mass_inventory(body)
    body = FakeBody()
    body.m.opt.gravity = np.array([0.0, np.nan, -10.0])
    with pytest.raises(ValueError, match="weight"):
        e185.fly_mass_inventory(body)
    body = FakeBody()
    body.m.opt.gravity = np.array([0.0, -10.0])
    with pytest.raises(ValueError, match="weight"):
        e185.fly_mass_inventory(body)


def test_single_native_step_original_tuple_and_restoration():
    body = FakeBody([contact(0, 30, (6.0, 0.0, 0.0))])
    old = body._step_physics
    native, forces, metadata = [], [], {}
    restore = e185.install_native_observer(
        body, native, forces, metadata, diagnostics, RuntimeError
    )
    try:
        body._step_physics()
    finally:
        restore()
    assert body.steps == 1
    assert len(body.mj.forward_data) == 1
    assert body.mj.forward_data[0] is not body.d
    assert body._step_physics == old
    assert len(native) == len(forces) == 1
    assert native[0][0] == pytest.approx(0.0001)
    assert native[0][3].tolist() == [False] * 6
    assert native[0][4:] == (False, 1.0)
    arrays = e185.force_arrays(forces)
    assert arrays["available"].tolist() == [True]
    assert arrays["time_s"].tolist() == pytest.approx([0.0001])
    assert metadata["body_mass_total"] == 6.0


def test_health_stop_and_partial_force_failure_keep_native_tuple():
    body = FakeBody([contact(0, 30, (2.0, 0.0, 0.0))])
    body.fault = "synthetic fault"
    native, forces = [], []
    restore = e185.install_native_observer(
        body, native, forces, {}, diagnostics, RuntimeError
    )
    with pytest.raises(RuntimeError, match="health guard"):
        body._step_physics()
    restore()
    assert len(native) == len(forces) == 1
    assert forces[0]["available"] is True
    assert body.steps == 1

    body = FakeBody([contact(0, 30, (2.0, 0.0, 0.0))])
    body.mj.mj_contactForce = lambda *_: (_ for _ in ()).throw(
        ValueError("force failed")
    )
    native, forces = [], []
    restore = e185.install_native_observer(
        body, native, forces, {}, diagnostics, RuntimeError
    )
    with pytest.raises(ValueError, match="force failed"):
        body._step_physics()
    restore()
    assert len(native) == len(forces) == 1
    assert native[0][0] == pytest.approx(0.0001)
    arrays = e185.force_arrays(forces)
    assert arrays["available"].tolist() == [False]
    assert "force failed" in arrays["error"][0]
    assert np.isnan(arrays["leg_force"]).all()


def test_invalid_gate_creates_only_attempt_and_charges_cost(tmp_path, monkeypatch):
    monkeypatch.setattr(e185, "OUTPUT", tmp_path / "e185")
    freeze = tmp_path / "bad.json"
    freeze.write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")
    args = [
        "--freeze",
        str(freeze),
        "--freeze-sha256",
        e185.sha256(freeze),
        "--source-review",
        str(tmp_path / "review.json"),
        "--source-review-sha256",
        "0" * 64,
    ]
    assert e185.main(args) == 1
    receipt = json.loads((e185.OUTPUT / "completion.json").read_text(encoding="utf-8"))
    assert receipt["stage"] == "source_gate"
    assert receipt["error"]["type"] == "ValueError"
    assert receipt["elapsed_wall_s"] > 0
    assert set(e185.OUTPUT.iterdir()) == {
        e185.OUTPUT / "attempt.json",
        e185.OUTPUT / "completion.json",
    }
    with pytest.raises(FileExistsError):
        e185.main(args)


def test_empty_force_arrays_preserve_shapes():
    arrays = e185.force_arrays([])
    assert arrays["segment_force"].shape == (0, 30, 3)
    assert arrays["leg_force"].shape == (0, 6, 3)
    assert arrays["time_s"].shape == (0,)


def test_monitor_unknown_overlap_and_missing_samples_fail_closed():
    assert not e185.assess_monitor([], [], False)["resource_clear"]
    good = {"queue": {"running": 0, "pending": 0}, "gpu_csv": ["synthetic"]}
    assert e185.assess_monitor([good], [], False)["resource_clear"]
    assert not e185.assess_monitor(
        [good, {**good, "queue_error": "unknown"}], [], False
    )["resource_clear"]
    assert not e185.assess_monitor(
        [good, {**good, "queue": {"running": 1, "pending": 0}}], [], False
    )["resource_clear"]
    assert not e185.assess_monitor([good], ["monitor failed"], False)["resource_clear"]
    assert not e185.assess_monitor([good], [], True)["resource_clear"]


def test_supervisor_caps_output_and_records_owned_pid(tmp_path, monkeypatch):
    output = tmp_path / "e185"
    output.mkdir()
    monkeypatch.setattr(e185, "OUTPUT", output)
    (output / "readiness.json").write_text("{}", encoding="utf-8")
    terminated = []

    class Process:
        pid = 12345

        def wait(self, timeout):
            return 0

    process = Process()
    monkeypatch.setattr(e185.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        e185.shutil, "disk_usage", lambda *_: SimpleNamespace(free=30 << 30)
    )
    fake = SimpleNamespace(
        psutil=SimpleNamespace(
            Process=lambda pid: SimpleNamespace(pid=pid, create_time=lambda: 42.0)
        ),
        _owned_children=lambda *args: None,
        _live_owned=lambda *_: [process],
        _terminate_owned_tree=lambda pid, known: (
            terminated.append((pid, dict(known)))
            or {"alive_pids": [], "errors": [], "terminated_pids": [pid]}
        ),
        _directory_bytes=lambda path: (1 << 30) + 1 if path.name == "baseline" else 0,
    )
    stop = __import__("threading").Event()
    receipt = e185.supervise_child(["synthetic"], fake, stop)
    assert receipt["status"] == "resource_guard"
    assert receipt["cap_reason"] == "baseline_raw_above_1GiB"
    assert receipt["owned_pid_birth_s"] == {12345: 42.0}
    assert terminated == [(12345, {12345: 42.0})]
    assert stop.is_set()
    assert (output / "child-started.json").is_file()
    assert (output / "child-supervisor.json").is_file()


@pytest.mark.parametrize("terminated", [[], [12345]])
def test_supervisor_distinguishes_exit_race_from_terminated_descendant(
    tmp_path, monkeypatch, terminated
):
    monkeypatch.setattr(e185, "OUTPUT", tmp_path)
    (tmp_path / "readiness.json").write_text("{}", encoding="utf-8")
    process = SimpleNamespace(pid=12345, wait=lambda timeout: 0)
    monkeypatch.setattr(e185.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        e185.shutil, "disk_usage", lambda *_: SimpleNamespace(free=30 << 30)
    )
    fake = SimpleNamespace(
        psutil=SimpleNamespace(
            Process=lambda pid: SimpleNamespace(pid=pid, create_time=lambda: 42.0)
        ),
        _owned_children=lambda *args: None,
        _live_owned=lambda *_: [process],
        _terminate_owned_tree=lambda *args: {
            "alive_pids": [],
            "errors": [],
            "terminated_pids": terminated,
        },
        _directory_bytes=lambda *_: 0,
    )
    receipt = e185.supervise_child(["synthetic"], fake, __import__("threading").Event())
    assert receipt["live_after_exit_snapshot"] == [12345]
    assert receipt["exit_snapshot_cleared_without_termination"] is (not terminated)
    assert receipt["status"] == (
        "completed" if not terminated else "unexpected_live_descendant"
    )


def test_child_restores_module_and_body_hook_on_original_failure(tmp_path, monkeypatch):
    output = tmp_path / "e185"
    output.mkdir()
    monkeypatch.setattr(e185, "OUTPUT", output)
    body = FakeBody()
    native_step = body._step_physics
    original_hook = object()
    protocol = SimpleNamespace(
        install_native_observer=original_hook,
        contact_diagnostics=diagnostics,
        NativeHealthStop=RuntimeError,
        joint_fit=SimpleNamespace(
            one_cpu_thread=lambda: (
                SimpleNamespace(restore_original_limits=lambda: None),
                None,
            )
        ),
    )
    gate = SimpleNamespace(
        mode="baseline",
        freeze={
            "graph_path": "graph",
            "graph_hash": "hash",
            "neutral_checkpoint": "neutral",
        },
    )

    def failed_run(*args, **kwargs):
        protocol.install_native_observer(body, [])
        body._step_physics()
        raise RuntimeError("synthetic run failed")

    fake = SimpleNamespace(
        protocol=protocol,
        verified_inputs=lambda *args, **kwargs: gate,
        _load_startup=lambda *_: {},
        GraphStore=SimpleNamespace(load=lambda *_: SimpleNamespace(hash="hash")),
        StateStore=SimpleNamespace(load=lambda *_: {}),
        run_arm=failed_run,
        LIMITS={},
    )
    monkeypatch.setitem(sys.modules, "tools.optimize_c_banc_m3_native_timing", fake)
    monkeypatch.setattr(tools, "optimize_c_banc_m3_native_timing", fake, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "flylab.c.walking_trace",
        SimpleNamespace(contact_diagnostics=diagnostics),
    )
    monkeypatch.setattr(
        e185,
        "sealed",
        lambda *_: {
            "schema": "flylab.m3.e185.readiness.v1",
            "freeze_sha256": "f" * 64,
            "source_review_sha256": "r" * 64,
            "observations": [{"ready": True}, {"ready": True}],
        },
    )
    monkeypatch.setattr(e185, "safe", lambda name: tmp_path / name)
    monkeypatch.setattr(
        e185.shutil, "disk_usage", lambda *_: SimpleNamespace(free=30 << 30)
    )
    monkeypatch.setattr(
        e185.np,
        "savez_compressed",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("archive failed")),
    )
    args = SimpleNamespace(
        freeze_sha256="f" * 64, source_review_sha256="r" * 64, readiness_sha256="s" * 64
    )
    freeze = {
        "e181_gate": {
            "freeze_path": "old-freeze",
            "freeze_sha256": "f" * 64,
            "source_review_path": "old-review",
            "source_review_sha256": "r" * 64,
        }
    }
    with pytest.raises(RuntimeError, match="synthetic run failed"):
        e185.child(args, freeze)
    assert protocol.install_native_observer is original_hook
    assert body._step_physics == native_step
    assert body.steps == 1
    with np.load(
        output / "ground-reaction-uncompressed.npz", allow_pickle=False
    ) as saved:
        assert saved["available"].tolist() == [True]
    metadata = json.loads(
        (output / "ground-reaction-metadata.json").read_text(encoding="utf-8")
    )
    assert "synthetic run failed" in metadata["primary_error"]
    assert metadata["raw_persistence_errors"][0]["artifact"] == "ground-reaction.npz"


def test_incomplete_original_result_keeps_invalid_check_without_comparison(
    tmp_path, monkeypatch
):
    output = tmp_path / "e185"
    output.mkdir()
    monkeypatch.setattr(e185, "OUTPUT", output)
    sentinel = object()
    protocol = SimpleNamespace(
        install_native_observer=sentinel,
        NativeHealthStop=RuntimeError,
        joint_fit=SimpleNamespace(
            one_cpu_thread=lambda: (
                SimpleNamespace(restore_original_limits=lambda: None),
                None,
            )
        ),
    )
    gate = SimpleNamespace(
        mode="baseline",
        freeze={
            "graph_path": "graph",
            "graph_hash": "hash",
            "neutral_checkpoint": "neutral",
        },
    )
    incomplete = {
        "validity": "incomplete",
        "technical_failure": True,
        "completed_controls": 7,
        "stop_reason": "execution_error",
    }

    def forbidden_compare(*args):
        raise AssertionError("partial trace must not be compared as 601 rows")

    fake = SimpleNamespace(
        protocol=protocol,
        verified_inputs=lambda *args, **kwargs: gate,
        _load_startup=lambda *_: {},
        GraphStore=SimpleNamespace(load=lambda *_: SimpleNamespace(hash="hash")),
        StateStore=SimpleNamespace(load=lambda *_: {}),
        run_arm=lambda *args, **kwargs: incomplete,
        _directory_bytes=lambda *_: 0,
        compare_baseline_prefix=forbidden_compare,
        LIMITS={},
    )
    monkeypatch.setitem(sys.modules, "tools.optimize_c_banc_m3_native_timing", fake)
    monkeypatch.setattr(tools, "optimize_c_banc_m3_native_timing", fake, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "flylab.c.walking_trace",
        SimpleNamespace(contact_diagnostics=diagnostics),
    )
    monkeypatch.setattr(
        e185,
        "sealed",
        lambda *_: {
            "schema": "flylab.m3.e185.readiness.v1",
            "freeze_sha256": "f" * 64,
            "source_review_sha256": "r" * 64,
            "observations": [{"ready": True}, {"ready": True}],
        },
    )
    monkeypatch.setattr(e185, "safe", lambda name: tmp_path / name)
    monkeypatch.setattr(
        e185.shutil, "disk_usage", lambda *_: SimpleNamespace(free=30 << 30)
    )
    args = SimpleNamespace(
        freeze_sha256="f" * 64, source_review_sha256="r" * 64, readiness_sha256="s" * 64
    )
    freeze = {
        "e181_gate": {
            "freeze_path": "old-freeze",
            "freeze_sha256": "f" * 64,
            "source_review_path": "old-review",
            "source_review_sha256": "r" * 64,
        },
        "e182_baseline_directory": "reference",
    }
    check = e185.child(args, freeze)
    assert check["observation_valid"] is False
    assert check["e181_result"]["stop_reason"] == "execution_error"
    assert check["e182_exact_comparison"] is None
    assert (output / "observation-check.json").is_file()
