"""Generated E348 contract tests and ONE scheduled actual tiny continuity probe."""

import copy
import inspect
import json
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import probe_c_banc_m3_shape_shared_timing_body as p


def fixture():
    phase = np.arange(128) * 2 * np.pi / 128
    curves = np.zeros((6, 128, 7), dtype=np.float64)
    for l in range(6):
        curves[l, :, 0] = np.cos(phase)
        curves[l, :, 1] = np.sin(phase)
        curves[l, :, 2] = l * 0.01
    command = np.empty((301, 42))
    for c in range(301):
        for l in range(6):
            command[c, l * 7 : (l + 1) * 7] = curves[l, (c + l * 8) % 128]
    reference = {
        "curves": curves,
        "temperature": np.full(6, 0.0001),
        "normalization": np.ones(6),
        "future_anchors": np.roll(curves, -1, axis=1),
        "phase_basis": np.stack((np.cos(phase), np.sin(phase)), axis=1),
    }
    rows = []
    for c in range(301):
        rows.append(
            {
                "control_tick": c,
                "time_s": c * 0.005,
                "physics_time_s": c * 0.005,
                "qpos": [0.0],
                "qvel": [0.0],
                "ctrl": [0.0],
                "joint_targets_rad": command[c].tolist(),
                "adhesion": [False] * 6,
                "tendon_inputs": {l + "_tarsus": 0.0 for l in p.LEGS},
                "fault": None,
                "body": {"legs": {l: [[0.0, 0.0, 0.0]] * 8 for l in p.LEGS}},
            }
        )
    # Actual held-action row0 deliberately differs from command0.
    rows[0]["joint_targets_rad"] = [0.0] * 42
    native = {
        "time_s": np.arange(1, 15001) * 0.0001,
        "normal_load_bw": np.ones((15000, 6)),
        "slip_mm_s": np.zeros((15000, 6)),
        "slip_valid": np.ones((15000, 6), bool),
        "upright": np.ones(15000),
        "nonfoot_contact": np.zeros(15000, bool),
    }
    return {
        "control": {"command": command},
        "reference": reference,
        "rows": rows,
        "native": native,
        "body_state": {},
        "metadata": {"samples": rows},
        "body_metadata": {},
    }


class Fake:
    def __init__(self):
        self.d = SimpleNamespace(time=0.0)
        self.steps = 0
        self.closed = False
        self.current = None

    def _step_physics(self):
        self.steps += 1
        self.d.time = self.steps * 0.0001

    def physics_time(self):
        return self.d.time

    def restore(self, state):
        pass

    def cpg_state(self):
        return {"fixed": True}

    def close(self):
        self.closed = True


class FakeDeps:
    def __init__(self, data, *, health=None, corrupt_prefix=False, technical=False):
        self.data = data
        self.health = health
        self.corrupt_prefix = corrupt_prefix
        self.technical = technical
        self.bodies = []
        _, static, _ = p.libraries()
        self.counter = static.InstancePhysicsCounter

    def body_class(self, data):
        return Fake

    def body_arguments(self, data):
        return (), {}

    def configure(self, body, data):
        self.bodies.append(body)
        body.arm = len(self.bodies)
        return {
            "physics_identity": "fixed",
            "joint_names": list(range(42)),
            "config": {},
            "body_options": {},
            "physics_profile": {},
            "model_hash": "fixed",
        }

    def install(self, body, callback):
        original = body._step_physics

        def step():
            original()
            n = body.steps - 1
            sample = {key: self.data["native"][key][n].tolist() for key in p.NATIVE}
            if body.arm == 2 and self.health is not None and n == self.health * 50:
                sample["upright"] = 0.8
            callback(sample)

        body._step_physics = step
        return lambda: setattr(body, "_step_physics", original)

    def action(self, body, row, command):
        body.current = command.copy()
        if self.technical and body.arm == 2 and row["control_tick"] == 102:
            raise OSError("technical fixture")
        for _ in range(50):
            body._step_physics()

    def boundary(self, body, source):
        row = copy.deepcopy(source)
        if body.current is not None:
            row["joint_targets_rad"] = body.current.tolist()
        if self.corrupt_prefix and body.arm == 2 and row["control_tick"] == 100:
            row["qpos"][0] = 1.0
        return row


def no_metrics(*args, full):
    return {
        "full_status": "not_tested",
        "physical": {"forward_mm": None, "slip_mm_s": None},
    }


def test_plan_fixed_prefix_transition_LF():
    data = fixture()
    out, encoded = p.plan(data["control"]["command"], data["reference"])
    for arm in ("B", "I", "C"):
        assert out[arm][:101].tobytes() == data["control"]["command"][:101].tobytes()
    assert not np.array_equal(out["I"][120:], out["C"][120:])
    _, static, _ = p.libraries()
    expected = static.cyclic_interpolate(
        data["reference"]["curves"], np.full((1, 6), encoded["phi"][20, 0])
    ).reshape(42)
    assert out["C"][120].tobytes() == expected.tobytes()
    w = 3 * 0.05**2 - 2 * 0.05**3
    target = static.cyclic_interpolate(
        data["reference"]["curves"], encoded["phi"][1:2]
    ).reshape(42)
    np.testing.assert_allclose(
        out["I"][101], (1 - w) * out["B"][101] + w * target, atol=1e-15, rtol=0
    )


@pytest.mark.parametrize(
    "health,corrupt,technical,arms,status",
    [
        (None, False, False, 3, "completed"),
        (100, False, False, 3, "completed"),
        (99, False, False, 2, "technical_invalid"),
        (None, True, False, 2, "technical_invalid"),
        (None, False, True, 2, "technical_invalid"),
    ],
)
def test_actual_prefix_and_partial_next_fresh(
    tmp_path, health, corrupt, technical, arms, status
):
    data = fixture()
    deps = FakeDeps(data, health=health, corrupt_prefix=corrupt, technical=technical)
    result = p.run_generated(
        data, tmp_path / "run", dependencies=deps, evaluator=no_metrics
    )
    assert len(deps.bodies) == arms
    assert result["status"] == status
    assert all(b.closed for b in deps.bodies)
    if arms == 3:
        assert result["arms"]["I"]["prefix_checked_before_c101"] is True
        assert result["H348"]["verdict"] == "not_evaluated"
    if health == 100:
        assert result["arms"]["I"]["status"] == "scientific_partial"
        assert result["arms"]["I"]["controls_unreturned"] == 1
        assert result["arms"]["C"]["controls_returned"] == 300


def descriptor(file, arrays):
    selected = {}
    with zipfile.ZipFile(file) as archive:
        for key in arrays:
            info = archive.getinfo(key + ".npy")
            with archive.open(info) as s:
                version = np.lib.format.read_magic(s)
                shape, order, dtype = np.lib.format.read_array_header_1_0(s)
                header = s.tell()
            selected[key] = {
                "member": info.filename,
                "dtype": dtype.name,
                "shape": list(shape),
                "fortran_order": order,
                "npy_version": list(version),
                "header_bytes": header,
                "zip_compression": info.compress_type,
                "compressed_bytes": info.compress_size,
                "uncompressed_bytes": info.file_size,
                "crc32": info.CRC,
            }
    return {
        "path": file.name,
        "bytes": file.stat().st_size,
        "sha256": p.digest(file.read_bytes()),
        "decode_members": list(arrays),
        "selected_members": selected,
    }


def test_selected_extra_is_unopened(tmp_path, monkeypatch):
    file = tmp_path / "selected.npz"
    np.savez_compressed(
        file, selected=np.zeros((3, 2)), extra=np.array([object()], object)
    )
    spec = descriptor(file, ["selected"])
    opened = []
    original = zipfile.ZipFile.open

    def guard(self, name, *args, **kwargs):
        assert getattr(name, "filename", name) != "extra.npy"
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", guard)
    assert p.selected(spec, root=tmp_path, opened=opened)["selected"].shape == (3, 2)
    assert opened == ["selected.npy"]
    spec["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="pin"):
        p.selected(spec, root=tmp_path)


def test_duplicate_traversal_refused(tmp_path):
    file = tmp_path / "bad.npz"
    with zipfile.ZipFile(file, "w") as archive:
        archive.writestr("../x.npy", b"x")
    spec = {
        "path": file.name,
        "bytes": file.stat().st_size,
        "sha256": p.digest(file.read_bytes()),
        "decode_members": [],
        "selected_members": {},
    }
    with pytest.raises(ValueError, match="unsafe"):
        p.selected(spec, root=tmp_path)


def test_hypothesis_dialect_no_adoption():
    physical = {
        "forward_mm": 1.7,
        "slip_mm_s": 1.8,
        "measured_native_healthy": True,
        "all_control_faults_clear": True,
    }
    metrics = {
        a: {
            "full_status": "evaluated",
            "physical": copy.deepcopy(physical),
            "gait": {
                "window": {"longest_forward_bout_s": 0.3},
                "events": {l: {"complete_cycles": 1} for l in p.LEGS},
            },
        }
        for a in ("B", "I", "C")
    }
    metrics["I"]["physical"].update(forward_mm=1.6, slip_mm_s=1.9)
    assert p.hypothesis(metrics)["supported"] is True
    metrics["C"]["gait"]["events"]["rh"]["complete_cycles"] = 0
    assert p.hypothesis(metrics)["supported"] is False


def test_current_metadata_only_preflight(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("no payload")

    monkeypatch.setattr(p, "selected", refuse)
    registration = p.metadata_preflight()
    assert set(registration["payloads"]) == {"control", "native", "reference"}


def test_root_input_failure_precedes_body_allocation(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "original_gate", lambda *a, **k: {})
    monkeypatch.setattr(p, "path", lambda name: tmp_path / name)

    def rejected(*a, **k):
        raise ValueError("unrecorded source force")

    def forbidden():
        raise AssertionError("body dependencies before input success")

    monkeypatch.setattr(p, "load_original", rejected)
    monkeypatch.setattr(p, "NativeDependencies", forbidden)
    assert p.main(["--run", "--freeze", "unused", "--review", "unused", "--output", "failed"]) == 1
    saved = json.loads((tmp_path / "failed" / "result.json").read_text("utf-8"))
    assert saved["body_constructors"] == saved["scientific_physics_steps"] == saved["neural_calls"] == 0
    assert saved["array_read_counts"] is None
    assert saved["first_error"]["stage"] == "input_load"


def test_root_empirical_analyzer_tuple_unpacking(tmp_path, monkeypatch):
    from tools import analyze_c_banc_biological_gait as gait

    report = {"window": {"longest_forward_bout_s": .35}, "events": {}}
    oracle = SimpleNamespace(physical=lambda *a, **k: {}, smooth=lambda *a: {}, shape=lambda *a: 0.)
    monkeypatch.setattr(p, "libraries", lambda: (np, None, oracle))
    monkeypatch.setattr(gait, "load_model", lambda file: {})
    monkeypatch.setattr(gait, "analyze", lambda data: (report, np.zeros((3, 6)), {}))
    result = p.evaluate(tmp_path, {"samples": [{"joint_targets_rad": [0.]*42}]}, {}, {"curves": None}, full=True)
    assert result["gait"] is report
    json.dumps(result, allow_nan=False)


def test_final_publication_deadline(tmp_path):
    data = fixture()

    def expired():
        raise TimeoutError("publication close deadline")

    result = p.run(
        data,
        tmp_path / "late",
        dependencies=FakeDeps(data),
        timely=lambda: None,
        final_timely=expired,
        evaluator=no_metrics,
        strict=False,
        generated=True,
    )
    assert result["status"] == "technical_invalid"
    assert result["arms"]["B"]["first_error"]["stage"] == "arm_publication"
    assert (tmp_path / "late" / "B" / "receipt.json").exists()


@pytest.mark.parametrize(
    "fault",
    [
        "planning",
        "identity",
        "evaluation",
        "evaluation_publication",
        "final_publication",
        "deadline",
        "partial",
    ],
)
def test_r1_orchestration_small_receipts(tmp_path, monkeypatch, fault):
    """Cheap injected orchestration preserves real reported counters, no traces."""
    original_publish = p.publish
    calls = []

    def fake_plan(*a):
        if fault == "planning":
            raise ValueError("planning fixture")
        return {arm: np.zeros((1, 1)) for arm in ("B", "I", "C")}, {
            k: np.zeros(1) for k in ("z", "R", "phi")
        }

    monkeypatch.setattr(p, "plan", fake_plan)

    def fake_replay(arm, command, data, deps, folder, **kw):
        calls.append(arm)
        proof = {
            k: "fixed"
            for k in (
                "physics_identity",
                "joint_names",
                "config",
                "body_options",
                "physics_profile",
                "model_hash",
            )
        }
        if fault == "identity" and arm == "I":
            proof["physics_identity"] = "different"
        partial = fault == "partial" and arm == "I"
        receipt = {
            "status": "scientific_partial" if partial else "completed",
            "configuration": proof,
            "first_error": None,
            "errors": [],
            "controls_requested": 101 if partial else 300,
            "controls_returned": 100 if partial else 300,
            "native_samples_acquired": 5001 if partial else 15000,
            "cleanup_completed": True,
        }
        original_publish(folder / "receipt.json", receipt)
        return receipt, {}, {}

    monkeypatch.setattr(p, "replay", fake_replay)

    def evaluate(*a, **kw):
        if fault == "evaluation":
            raise ArithmeticError("metric fixture")
        return {
            "full_status": "not_tested",
            "physical": {"forward_mm": None, "slip_mm_s": None},
        }

    tripped = []

    def write(file, value):
        if (
            (fault == "evaluation_publication" and file.name == "evaluation.json")
            or (fault == "final_publication" and file.name == "result.json")
        ) and not tripped:
            tripped.append(file.name)
            raise OSError("publication fixture")
        original_publish(file, value)

    monkeypatch.setattr(p, "publish", write)

    def deadline():
        if fault == "deadline":
            raise TimeoutError("post-close fixture")

    output = tmp_path / "r1"
    result = p.run(
        {"control": {"command": None}, "reference": None},
        output,
        dependencies=None,
        timely=lambda: None,
        final_timely=deadline,
        evaluator=evaluate,
        generated=True,
    )
    disk = json.loads((output / "result.json").read_text("utf-8"))
    assert disk == result
    assert result["external_completion_qualified"] is False
    if fault == "partial":
        assert calls == ["B", "I", "C"]
        assert result["arms"]["I"]["status"] == "scientific_partial"
        assert result["H348"]["verdict"] == "not_evaluated"
    else:
        assert result["status"] == "technical_invalid"
        assert result["first_error"] is not None
        if fault == "planning":
            assert calls == [] and result["arms"] == {}
        else:
            if fault not in ("deadline", "final_publication"):
                assert len(calls) == (2 if fault == "identity" else 1)
            for arm, receipt in result["arms"].items():
                assert (
                    receipt["controls_requested"] == receipt["controls_returned"] == 300
                )
                assert (
                    json.loads((output / arm / "receipt.json").read_text("utf-8"))
                    == receipt
                )


def test_r1_trace_lineage():
    trace = p.body_trace(
        {
            "schema": "old neural",
            "model_hash": "m",
            "adapter_hash": "a",
            "graph_hash": "g",
        },
        [],
    )
    assert trace["schema"] == "flylab.m3.e348.body-only-trace.v1"
    assert (
        trace["neural_execution"] == 0
        and trace["source_neural_metadata_is_label_only"] is True
    )
    assert trace["source_neural_labels"] == {
        "model_hash": "m",
        "adapter_hash": "a",
        "graph_hash": "g",
    }
    assert trace["landmarks"]["count"] == 8


def test_r1_canonical_body_arguments():
    metadata = p.read_json(p.ROOT / p.BASE / "parent-e337-current-body-metadata.json")
    args, kw = p.NativeDependencies.body_arguments({"body_metadata": metadata})
    assert args[0] == 42
    assert type(kw["body_options"].slope_degrees) is float
    assert kw["body_options"].slope_degrees == 10.0
    assert kw["body_options"].servo_profile == "tracking_all"
    assert kw["physics_profile"].backend == "cpu"
    assert type(args[2]["friction"]) is float


def tiny_native(output, entry):
    """Scheduled once by owned supervisor helper, never normal pytest collection.

    Generated seed42 literal checkpoint; both constructors include2000 settling.
    First body's continuous50 is compared to fresh second body's restore+50.
    A failed exact comparison is retained; no epsilon/retry adaptation.
    """
    started = time.perf_counter()
    module_import_s = started - entry
    _, static, _ = p.libraries()
    metadata = p.read_json(p.ROOT / p.BASE / "parent-e337-current-body-metadata.json")
    data = {"body_metadata": metadata, "body_state": {"modelHash": p.MODEL_HASH}}
    dep = p.NativeDependencies()
    report = {
        "schema": "flylab.m3.e348.tiny-native.v1",
        "source_sha256": p.sources(),
        "original_calls": 0,
        "neural_calls": 0,
        "CPG_advance_calls": 0,
        "passed": False,
        "external_completion_qualified": False,
        "stages": {
            "native_module_import_s": module_import_s,
            "library_metadata_load_s": time.perf_counter() - started,
        },
        "arms": [],
        "first_error": None,
    }
    bodies, detaches, counters = [], [], []
    reference_snapshot = reference_rows = reference_native = None
    try:
        began = time.perf_counter()
        cls = dep.body_class(data)
        args, kwargs = dep.body_arguments(data)
        report["stages"]["actual_class_argument_load_s"] = time.perf_counter() - began
        report["canonical_constructor_options"] = kwargs[
            "body_options"
        ].model_identity()
        report["legacy_OPTIONS_JSON"] = json.dumps(static.OPTIONS, sort_keys=True)
        from dataclasses import asdict

        report["canonical_OPTIONS_JSON"] = json.dumps(
            asdict(kwargs["body_options"]), sort_keys=True
        )
        report["API_signatures"] = {
            n: str(inspect.signature(getattr(cls, n)))
            for n in ("__init__", "restore", "snapshot", "step_joint_targets", "close")
        }
        for index in range(2):
            record = {}
            report["arms"].append(record)
            body = cls.__new__(cls)
            bodies.append(body)
            began = time.perf_counter()
            ctor = static.InstancePhysicsCounter(body, "constructor")
            counters.append(ctor)
            record["constructor"] = ctor.proof
            try:
                cls.__init__(body, *args, **kwargs)
            finally:
                ctor.restore()
                record["constructor_settling_s"] = time.perf_counter() - began
            assert ctor.proof["steps_requested"] == ctor.proof["steps_returned"] == 2000
            record["configuration_before_configure"] = dep.observed_configuration(body)
            record["configuration"] = dep.configure(body, data)
            if index == 0:
                reference_snapshot = body.snapshot()
                p.publish(
                    Path(output).with_name(Path(output).stem + "-generated-start.json"),
                    reference_snapshot,
                )
            else:
                began = time.perf_counter()
                body.restore(copy.deepcopy(reference_snapshot))
                record["restore_s"] = time.perf_counter() - began
            cpg = copy.deepcopy(body.cpg_state())
            tendon = {l + "_tarsus": 0.0 for l in p.LEGS}
            tendon.update(abdomen_pitch=0.0, abdomen_yaw=0.0)
            source = {
                "control_tick": 0,
                "tendon_inputs": tendon,
                "neural_request": {
                    "external_force_at_start": {
                        "push_left_s": 0.0,
                        "push_force": [0.0, 0.0, 0.0],
                    }
                },
            }
            samples = []
            counter = static.InstancePhysicsCounter(body, "science")
            counters.append(counter)
            record["scientific_physics"] = counter.proof
            detach = dep.install(body, samples.append)
            detaches.append(detach)
            began = time.perf_counter()
            anchor = dep.boundary(body, source)
            source.update(
                control_tick=1,
                joint_targets_rad=reference_snapshot["action"]["joint_angles"],
                adhesion=[
                    bool(x) for x in reference_snapshot["action"]["adhesion_onoff"]
                ],
            )
            dep.action(
                body, source, np.asarray(source["joint_targets_rad"], np.float64)
            )
            boundary = dep.boundary(body, source)
            record["control50_shadow_boundary_s"] = time.perf_counter() - began
            record["actual_clock_s"] = float(body.physics_time())
            assert (
                counter.proof["steps_requested"]
                == counter.proof["steps_returned"]
                == len(samples)
                == 50
            )
            assert body.cpg_state() == cpg
            for row in (anchor, boundary):
                for leg in p.LEGS:
                    static.typed(
                        np.asarray(row["body"]["legs"][leg], np.float64),
                        (8, 3),
                        "<f8",
                        "tiny8 geometry",
                    )
            native = static.native_arrays(samples)
            final = body.snapshot()
            p.publish(
                Path(output).with_name(
                    Path(output).stem + "-arm" + str(index) + "-states.json"
                ),
                {"anchor": anchor, "boundary": boundary, "final_snapshot": final},
            )
            with (
                Path(output)
                .with_name(Path(output).stem + "-arm" + str(index) + "-native.npz")
                .open("xb") as stream
            ):
                np.savez_compressed(stream, **native)
            record["native_samples_acquired"] = len(samples)
            if index == 0:
                reference_rows, reference_native, reference_final = (
                    [anchor, boundary],
                    native,
                    final,
                )
            else:
                assert record["configuration"] == report["arms"][0]["configuration"]
                for c, row in enumerate((anchor, boundary)):
                    for key in (
                        "qpos",
                        "qvel",
                        "ctrl",
                        "joint_targets_rad",
                        "adhesion",
                        "physics_time_s",
                    ):
                        static.exact(
                            np.asarray(row[key]),
                            np.asarray(reference_rows[c][key]),
                            "tiny continuous/restore " + key,
                        )
                for key in p.NATIVE:
                    static.exact(
                        native[key],
                        reference_native[key],
                        key,
                        masked_nan=key == "slip_mm_s",
                    )
                assert final == reference_final
            detach()
            detaches.remove(detach)
            counter.restore()
        report["both_continuations_completed"] = True
        report["passed"] = True
    except Exception as error:  # noqa: BLE001 -- preserve owned failure/cleanup evidence
        report["first_error"] = {
            "type": type(error).__name__,
            "message": str(error)[:2048],
        }
    finally:
        began = time.perf_counter()
        errors = []
        for callback in [
            *reversed(detaches),
            *(c.restore for c in reversed(counters) if not c.proof["method_restored"]),
        ]:
            try:
                callback()
            except Exception as error:  # noqa: BLE001 -- preserve owned failure/cleanup evidence
                errors.append(str(error))
        closure = []
        for index, body in reversed(list(enumerate(bodies))):
            item = {
                "allocated_body_index": index,
                "close_requested": True,
                "close_returned": False,
                "error": None,
            }
            try:
                body.close()
                item["close_returned"] = True
            except Exception as error:  # noqa: BLE001 -- owned close proof
                item["error"] = str(error)
                errors.append(str(error))
            closure.append(item)
        report["body_close_outcomes"] = closure
        report["actually_allocated_bodies"] = len(bodies)
        report["both_continuations_completed"] = report.get(
            "both_continuations_completed", False
        )
        report["cleanup_errors"] = errors
        report["all_owned_bodies_closed"] = bool(closure) and all(
            item["close_returned"] for item in closure
        )
        report["stages"]["cleanup_s"] = time.perf_counter() - began
        report["stages"]["native_total_s"] = time.perf_counter() - started
        report["passed"] &= not errors
        publication_began = time.perf_counter()
        p.publish(Path(output), report)
        # These fields belong to the caller's subsequent completion record;
        # they cannot truthfully be present in the file just closed above.
        report["post_publication"] = {
            "publication_close_s": time.perf_counter() - publication_began,
            "native_entry_to_publication_close_s": time.perf_counter() - entry,
        }
    return report
