"""E344 generated CPU fixtures; these do not qualify original construction."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import banc_m3_body_supervision as supervision
from tools import verify_c_banc_m3_common_descent_body as p


class HealthStop(RuntimeError):
    pass


def test_e346_actual_public_original_api():
    from flylab.c.graph import GraphStore
    from flylab.c.storage import StateStore

    api = p.resolve_original_api()
    assert api["GraphStore"] is GraphStore
    assert api["StateStore"] is StateStore
    assert GraphStore.__module__ == "flylab.c.graph"
    assert StateStore.__module__ == "flylab.c.storage"


def test_e346_actual_load_original_stops_at_exact_whole_metadata(monkeypatch):
    import functools

    api = p.resolve_original_api()
    reached, forbidden_calls = [], []

    class WholeMetadataBoundary(RuntimeError):
        pass

    marker = WholeMetadataBoundary("designated first whole_metadata boundary")
    inventory = {"inert_registered_metadata_boundary": "E346"}

    def boundary(root, passed_inventory):
        assert root == p.ROOT and passed_inventory is inventory
        reached.append("whole_metadata")
        raise marker

    def guard(function, label):
        @functools.wraps(function)
        def forbidden(*args, **kwargs):
            forbidden_calls.append(label)
            raise AssertionError("forbidden original operation: " + label)

        return forbidden

    for owner, name in (
        (api["GraphStore"], "load"),
        (api["StateStore"], "load"),
        (api["recovery"].SSPRecoveryResourceTrial, "from_checkpoint"),
    ):
        monkeypatch.setattr(
            owner, name, staticmethod(guard(getattr(owner, name), name))
        )
    for owner, name in (
        (api["recovery"], "checked_checkpoint"),
        (api["efficacy"].SSPEfficacyTrial, "__init__"),
        (api["native"], "install_native_observer"),
        (p.OriginalFactory, "__init__"),
        (p.e, "load_selected"),
        (np, "load"),
    ):
        monkeypatch.setattr(owner, name, guard(getattr(owner, name), name))
    old_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert path.suffix not in (".npy", ".npz"), "scientific payload access"
        return old_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(p, "whole_metadata", boundary)
    actual_resolver = p.resolve_original_api

    def watched_resolver():
        resolved = actual_resolver()
        reached.append("actual_imports_and_signature_bind")
        return resolved

    monkeypatch.setattr(p, "resolve_original_api", watched_resolver)
    with pytest.raises(WholeMetadataBoundary) as caught:
        p.load_original({"inventory": inventory}, timely=lambda: None)
    assert caught.value is marker
    assert reached == ["actual_imports_and_signature_bind", "whole_metadata"]
    assert forbidden_calls == []


def test_e346_public_signature_drift_rejected(monkeypatch):
    api = p.resolve_original_api()

    def wrong_state_load(path):
        raise AssertionError("signature probe must never call load")

    monkeypatch.setattr(api["StateStore"], "load", staticmethod(wrong_state_load))
    with pytest.raises(TypeError, match="max_files"):
        p.resolve_original_api()


def test_e345_registered_metadata_without_scientific_access(monkeypatch):
    """Use the actual registered JSON; sentinels fail on scientific operations."""

    def forbidden(*args, **kwargs):
        raise AssertionError("scientific sentinel reached")

    monkeypatch.setattr(p.e, "load_selected", forbidden)
    monkeypatch.setattr(p, "load_original", forbidden)
    monkeypatch.setattr(p, "GPUMemoryWatch", forbidden)
    monkeypatch.setattr(p, "OriginalFactory", forbidden)
    old_open = Path.open

    def metadata_open(path, *args, **kwargs):
        assert path.suffix not in (".npz", ".npy")
        return old_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", metadata_open)
    inv = p.validate_registered_inventory()
    before = p.pinned_json(p.ROOT, p.PREDECESSOR)
    assert p.preflight_inventory_metadata(inv, before)["files"] == 135
    for mutation in (
        "old131",
        "selected_omission",
        "alias",
        "binding",
        "member",
        "reference",
    ):
        bad = copy.deepcopy(inv)
        role = "startup"
        name = bad["selected_archives"][role]["path"]
        if mutation == "old131":
            bad["files"] = before["files"]
        elif mutation == "selected_omission":
            bad["files"][p.BASE + "/invented.npz"] = bad["files"].pop(name)
        elif mutation == "alias":
            bad["selected_archives"][role]["path"] = bad["selected_archives"][
                "reference"
            ]["path"]
        elif mutation == "binding":
            bad["files"][name]["sha256"] = "0" * 64
        elif mutation == "member":
            bad["selected_archives"][role]["selected_schema"]["target_command"][
                "shape"
            ] = [1]
        else:
            bad["initial_whole_array_references"][0]["path"] = name
        with pytest.raises((ValueError, AssertionError, RuntimeError)):
            p.preflight_inventory_metadata(bad, before)


def e345_supervision_fixture(root, label):
    prefix = p.BASE + "/e345-sol-" + label
    script = prefix + "-cpu.py"
    (root / p.BASE).mkdir(parents=True, exist_ok=True)
    (root / "tools").mkdir(exist_ok=True)
    for name in (supervision.SUPERVISOR, supervision.SELF):
        assert (root / name).is_file()
    (root / script).write_text("import time\ntime.sleep(.2)\n", "utf-8")
    # Fixed actual harmless CPU argv; truthy original_calls is only a metadata branch test.
    python = root / ".venv/Scripts/python.exe"
    python.parent.mkdir(parents=True, exist_ok=True)
    need_interpreter = python.exists()
    assert need_interpreter
    contract = {
        "phase": "e345-sol-cpu-contract",
        "original_calls": 1,
        "cumulative_before_s": 0.0,
        "cumulative_limit_s": 20.0,
        "retry_count": 0,
        "max_tree_rss_bytes": 256 * 1024**2,
        "source_sha256": {
            name: supervision.digest(root / name)
            for name in (script, supervision.SELF, supervision.SUPERVISOR)
        },
        "supervisor_output": prefix + "-supervisor.json",
        "commands": [
            {
                "argv": [".venv/Scripts/python.exe", "-X", "utf8", "-B", script],
                "wall_s": 1.0,
                "stdout": prefix + "-worker.stdout",
                "stderr": prefix + "-worker.stderr",
            }
        ],
    }
    authority = {
        "phase": contract["phase"],
        "original_calls": 1,
        "cumulative_before_s": 0.0,
        "cumulative_limit_s": 20.0,
        "phase_limit_s": 3.0,
        "job_wall_s": 1.0,
        "max_tree_rss_bytes": contract["max_tree_rss_bytes"],
        "argv": contract["commands"][0]["argv"],
        "generated_cpu": True,
        "science_original_calls": 0,
        "supervisor_stdout": prefix + "-outer.stdout",
        "supervisor_stderr": prefix + "-outer.stderr",
    }
    return prefix, contract, authority


@pytest.mark.parametrize(
    "case",
    [
        "truthy_finally",
        "missing_cumulative",
        "bool_cumulative",
        "nan_cumulative",
        "no_report",
    ],
)
def test_e345_supervision_recovery(tmp_path, monkeypatch, case):
    import uuid

    tmp_path = p.ROOT
    prefix, contract, authority = e345_supervision_fixture(
        tmp_path, case + "-" + uuid.uuid4().hex[:8]
    )
    if case == "missing_cumulative":
        del contract["cumulative_before_s"]
    elif case == "bool_cumulative":
        contract["cumulative_before_s"] = True
    elif case == "nan_cumulative":
        contract["cumulative_before_s"] = float("nan")
    contract_name, authority_name = (
        prefix + "-contract.json",
        prefix + "-authority.json",
    )
    (tmp_path / contract_name).write_text(json.dumps(contract), "utf-8")
    authority["contract_sha256"] = supervision.digest(tmp_path / contract_name)
    (tmp_path / authority_name).write_text(json.dumps(authority), "utf-8")
    if case == "no_report":
        actual_popen = supervision.subprocess.Popen

        def controlled_no_report(argv, **kwargs):
            # Real process exits without a supervisor report; no scientific code exists.
            return actual_popen(contract["commands"][0]["argv"], **kwargs)

        monkeypatch.setattr(supervision.subprocess, "Popen", controlled_no_report)
    result = supervision.supervise(
        contract_name,
        authority_name,
        supervision.digest(tmp_path / authority_name),
        prefix + "-receipt.json",
        prefix + "-pids.json",
        root=tmp_path,
    )
    assert result["science_original_calls"] == 0
    assert result["charged_s"] >= result["whole_elapsed_s"] > 0
    if case == "truthy_finally":
        assert result["technical_valid"] is True, result
        raw = supervision.read_json(tmp_path / contract["supervisor_output"])
        assert raw["original_calls"] == 1
        assert raw["cumulative_after_s"] == raw["elapsed_s"]
        assert result["all_owned_ended"] is True
    elif case in ("missing_cumulative", "bool_cumulative", "nan_cumulative"):
        assert result["spawned_supervisors"] == 0
        assert result["first_error"]["type"] == (
            "KeyError" if case == "missing_cumulative" else "ValueError"
        )
        assert result["all_owned_ended"] is False
    else:
        assert result["technical_valid"] is False
        assert result["supervisor_report_present"] is False
        assert result["spawned_supervisors"] == 1
        assert result["all_owned_ended"] is True
        assert result["first_error"]["type"] == "FileNotFoundError"


def row(tick, landmark_count=5):
    landmarks = {}
    for i, leg in enumerate(p.LEGS):
        x, y = (i % 3) * 0.4, 1 if i < 3 else -1
        landmarks[leg] = [
            [x, 0.0, y],
            [x + 0.1, 0.05, y + 0.15],
            [x + 0.2, 0.1, y + 0.3],
            [x + 0.3, 0.08, y + 0.5],
            [x + 0.4, 0.0, y + 0.7],
        ]
        if landmark_count == 8:
            landmarks[leg].extend(
                [[x + 0.4 + j * 0.03, 0.0, y + 0.7 + j * 0.05] for j in (1, 2, 3)]
            )
    return {
        "time_s": tick * 0.005,
        "control_tick": tick,
        "body": {
            "position": [tick * 0.005, 0.8, 0.0],
            "basis": np.eye(3).tolist(),
            "yaw": 0.0,
            "legs": landmarks,
        },
        "feet": {"load_bearing": [False] * 6},
        "upright": 1.0,
        "fault": None,
        "navigation": None,
    }


def chunk_proof(selected=None):
    selected = np.ones(8, np.float32) if selected is None else selected
    arrays = {
        "host.rowptr": np.asarray([0, 8], np.int32),
        "host.columns": np.arange(8, dtype=np.int32),
        "host.weights": selected,
        "packed.rowptr": np.asarray([0, 8], np.int32),
        "packed.columns": np.arange(8, dtype=np.int32),
        "packed.palette": np.ones(1, np.float32),
    }
    leaves = {}
    for name, value in arrays.items():
        digest = p.array_sha(value)
        leaves[name] = {
            "dtype": value.dtype.str,
            "count": value.size,
            "chunks": [
                {
                    "start": 0,
                    "end": value.size,
                    "element_count": value.size,
                    "dtype": value.dtype.str,
                    "bytes": value.nbytes,
                    "source_sha256": digest,
                    "expected_sha256": digest,
                    "actual_sha256": digest,
                }
            ],
        }
    return {
        "schema": "flylab.m3.e344.transfer-chunks.v1",
        "leaves": leaves,
        "palette": 1,
        "selected_sha256": p.array_sha(selected),
        "pointer_version_guards_passed": True,
        "generated_mock": True,
        "actual_GPU_observed": False,
        "collector_scope": "explicit synthetic CPU arrays and mock pointer guard",
    }


class FakeFactory:
    health_stop = HealthStop

    def __init__(
        self,
        *,
        stop_at=None,
        observation_fault=None,
        neural_fault=False,
        close_fault=False,
        q=0.0,
        landmark_count=5,
    ):
        (
            self.stop_at,
            self.observation_fault,
            self.neural_fault,
            self.close_fault,
            self.q,
        ) = stop_at, observation_fault, neural_fault, close_fault, q
        self.session = None
        self.landmark_count = landmark_count

    def construct(self, data, ledger, timely):
        timely()
        ledger.update(
            status="explicit_fake", original_factory_qualified=False, unknown_work=False
        )
        factory = self

        class Body:
            ticks = 0

            def _step_physics(self):
                self.ticks += 1

            def physics_time(self):
                return self.ticks * 0.0001

        class Network:
            tick = 0

            def advance_readout(self, drive, ids, steps):
                if factory.neural_fault:
                    raise RuntimeError("unreturned neural work")
                self.tick += steps
                return np.zeros(391, np.float32)

        class Session:
            def __init__(self):
                self.body, self.network, self.control_tick, self.closed = (
                    Body(),
                    Network(),
                    0,
                    False,
                )

            def advance(self, count, capture=False):
                assert count == 1 and capture is False
                self.network.advance_readout(
                    np.zeros(400, np.float32), np.arange(391, dtype=np.int64), 20
                )
                for _ in range(50):
                    self.body._step_physics()
                self.control_tick += 1

            def close(self):
                self.closed = True
                if factory.close_fault:
                    raise RuntimeError("owned close failed")

        self.session = Session()
        return self.session

    def install_contact(self, body, samples):
        original = body._step_physics

        def observed():
            answer = original()
            if body.ticks == self.observation_fault:
                raise RuntimeError("observer failed after native return")
            samples.append(
                (
                    body.physics_time(),
                    np.zeros(6),
                    np.full(6, np.nan),
                    np.zeros(6, bool),
                    False,
                    1.0,
                )
            )
            if body.ticks == self.stop_at:
                raise HealthStop("measured native health violation")
            return answer

        body._step_physics = observed
        return lambda: setattr(body, "_step_physics", original)

    def synchronize(self):
        pass

    def initial(self, session, data, timely):
        timely()
        return chunk_proof(data["selected_weights"])

    def metadata(self, session):
        return {
            "landmarks": {
                "scope": "synthetic",
                "count": self.landmark_count,
                "order": list(p.NATIVE_LANDMARKS)
                if self.landmark_count == 8
                else ["fixture" + str(i) for i in range(5)],
            },
            "body_hash": "b",
            "model_hash": "n",
            "adapter_hash": "a",
            "graph_hash": "g",
            "parameters": {},
        }

    def capture(self, session):
        return row(session.control_tick, self.landmark_count), {
            "control_tick": session.control_tick,
            "network_tick": session.network.tick,
            "physics_time_s": session.body.physics_time(),
            "command": np.full(42, self.q),
            "tendon": np.zeros(6),
        }


def fixture_data():
    return {
        "theta": np.ones(4, np.float64),
        "selected_weights": np.ones(8, np.float32),
        "ledger": [],
        "startup": {
            "target_command": np.zeros((99, 42)),
            "target_tendon": np.zeros((99, 6)),
        },
    }


def fixture_run(
    tmp_path,
    *,
    controls=1,
    factory=None,
    fault=None,
    clock=None,
    meter=None,
    watcher=None,
):
    name = p.BASE + "/e344-sol-case"
    kwargs = {
        "root": tmp_path,
        "output": name,
        "dimensions": p.GeneratedDimensions(controls),
        "printer": lambda payload: None,
        "meter": lambda device: {"tree_rss_bytes": 1000},
        "fault": fault,
    }
    if clock is not None:
        kwargs.update(clock=clock, s0=0.0)
    if meter is not None:
        kwargs["meter"] = meter
    if watcher is not None:
        kwargs["watcher"] = watcher
    factory = factory or FakeFactory()
    result, code = p.run_generated(factory, fixture_data(), **kwargs)
    return result, code, tmp_path / name, factory


def test_startup99_boundary_is_excluded():
    q, t = np.zeros((100, 42)), np.zeros((100, 6))
    q[0], t[0] = 99, 99
    assert p.startup99(q, t, np.zeros((99, 42)), np.zeros((99, 6)))["passed"]
    q[1] = 0.05
    assert not p.startup99(q, t, np.zeros((99, 42)), np.zeros((99, 6)))["passed"]


@pytest.mark.parametrize("shape", [(99, 42), (101, 42), (100, 41)])
def test_startup99_rejects_padding_and_other_shapes(shape):
    with pytest.raises(ValueError):
        p.startup99(
            np.zeros(shape), np.zeros((100, 6)), np.zeros((99, 42)), np.zeros((99, 6))
        )


def test_off_contact_nan_and_invalid_supported_slip():
    native = p.native_arrays(
        [
            (i * 0.0001, np.zeros(6), np.full(6, np.nan), np.zeros(6, bool), False, 1.0)
            for i in range(1, 5010)
        ]
    )
    assert p.validate_native(native) == 5009
    assert p.supported_slip(native) == {
        "available": False,
        "value": None,
        "reason": "no_supported_samples",
    }
    native["normal_load_bw"][-1, 0] = 0.02
    assert p.supported_slip(native)["reason"] == "invalid_supported_slip"
    native["slip_valid"][-1, 0], native["slip_mm_s"][-1, 0] = True, 2.0
    assert p.supported_slip(native)["value"] == 2.0


@pytest.mark.parametrize("mutation", ["missing", "overlap", "alter", "bytes", "dtype"])
def test_transfer_chunk_mutation(mutation):
    proof = chunk_proof()
    leaf, chunk = (
        proof["leaves"]["host.weights"],
        proof["leaves"]["host.weights"]["chunks"][0],
    )
    if mutation == "missing":
        leaf["chunks"] = []
    elif mutation == "overlap":
        chunk["start"] = 1
    elif mutation == "alter":
        chunk["actual_sha256"] = "b" * 64
    elif mutation == "bytes":
        chunk["bytes"] = 4 * p.e.MIB + 1
    else:
        chunk["dtype"] = "<f8"
    with pytest.raises(ValueError):
        p.verify_chunk_coverage(proof)


def test_success_disk_return_and_cleanup(tmp_path):
    result, code, output, factory = fixture_run(tmp_path)
    assert code == 0 and result["run_valid"] and factory.session.closed
    assert result["counters"]["neural_device_confirmed"] == 20
    assert result["counters"]["physics_device_confirmed"] == 50
    assert result["counters"]["completed_rows"] == 2
    assert (
        (output / "result.json").read_bytes()
        == (output / "receipts.json").read_bytes()
        == p.json_bytes(result)
    )
    assert (
        "manifest.json"
        not in json.loads((output / "manifest.json").read_text())["files"]
    )
    assert not result["effective_qualified"] and not result["biology_proven"]


def test_startup_reject_stops_before_control100(tmp_path):
    result, code, output, _ = fixture_run(
        tmp_path, controls=300, factory=FakeFactory(q=0.1)
    )
    assert code == 0 and result["reason"] == "startup_rejected"
    assert (
        result["metrics"] is None and result["classification"] == "scientific_rejected"
    )
    assert result["counters"]["controls_completed"] == 99
    assert result["counters"]["neural_host_returned"] == 1980
    assert result["counters"]["observer_samples"] == 4950
    with np.load(output / "control-raw.npz", allow_pickle=False) as archive:
        assert archive["command"].shape == (100, 42)


def test_midphysics_health_is_distinct_returned_work(tmp_path):
    result, code, _, _ = fixture_run(tmp_path, factory=FakeFactory(stop_at=7))
    c = result["counters"]
    assert (
        code == 0
        and result["reason"] == "native_health_stop"
        and result["metrics"] is None
    )
    assert c["neural_host_returned"] == c["neural_device_confirmed"] == 20
    assert (
        c["physics_host_returned"]
        == c["physics_device_confirmed"]
        == c["observer_samples"]
        == 7
    )
    assert c["controls_completed"] == 0 and c["completed_rows"] == 1


def test_observer_fault_preserves_native_return(tmp_path):
    result, code, output, factory = fixture_run(
        tmp_path, factory=FakeFactory(observation_fault=7)
    )
    assert code == 1 and result["counters"]["physics_host_returned"] == 7
    assert (
        result["counters"]["observer_samples"] == 6
        and result["counters"]["neural_host_returned"] == 20
    )
    assert factory.session.closed and (output / "native-contact.npz").exists()
    assert "observer failed" in result["first_error"]["message"]


def test_unreturned_neural_is_unknown_not_zero(tmp_path):
    result, code, _, _ = fixture_run(tmp_path, factory=FakeFactory(neural_fault=True))
    assert code == 1 and result["counters"]["neural_requested"] == 20
    assert result["counters"]["neural_unreturned_unknown"] is True
    assert result["counters"]["neural_host_returned"] == 0


@pytest.mark.parametrize("stage", ["raw-save", "readback", "postpublication"])
def test_partial_and_late_publication_failures(tmp_path, stage):
    def fault(at):
        if at == stage:
            raise RuntimeError("injected " + stage)

    result, code, output, factory = fixture_run(tmp_path, fault=fault)
    assert code == 1 and not result["run_valid"] and result["revoked"]
    assert result["counters"]["physics_host_returned"] == 50 and factory.session.closed
    assert (output / "control-raw.npz").exists() and (
        output / "revocation.json"
    ).exists()


def test_first_and_secondary_cleanup_failures(tmp_path):
    result, code, _, _ = fixture_run(
        tmp_path, factory=FakeFactory(observation_fault=7, close_fault=True)
    )
    assert code == 1 and "observer failed" in result["first_error"]["message"]
    assert any("owned close" in v["message"] for v in result["secondary_errors"])


def test_late_deadline_revokes_without_erasing_provisional(tmp_path):
    now = [0.0]

    def fault(stage):
        if stage == "postpublication":
            now[0] = 73.0

    result, code, output, _ = fixture_run(tmp_path, clock=lambda: now[0], fault=fault)
    assert code == 1 and result["first_error"]["type"] == "TimeoutError"
    assert json.loads((output / "result.json").read_text())["run_valid"] is True
    assert json.loads((output / "revocation.json").read_text())["run_valid"] is False


def test_resource_type_and_cap(tmp_path):
    result, code, _, _ = fixture_run(
        tmp_path, meter=lambda device: {"tree_rss_bytes": True}
    )
    assert code == 1 and "host6GiB" in result["first_error"]["message"]


def test_lean301_consumed_by_actual_gait(tmp_path):
    from tools.analyze_c_banc_biological_gait import analyze, load_model

    trace = p.lean_trace([row(i) for i in range(301)], FakeFactory().metadata(None))
    payload = p.json_bytes(trace)
    assert len(payload) < 2 * p.e.MIB
    destination = tmp_path / p.BASE / "e344-sol-generated-trace.json"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(payload)
    result, _, _ = analyze(load_model(destination))
    assert set(result["events"]) == set(p.LEGS)


def test_native8_lossless_contract_and_truncated_rejection():
    native_row = row(0, 8)
    metadata = FakeFactory(landmark_count=8).metadata(None)
    metadata["landmarks"]["scope"] = "actual_native"
    trace = p.lean_trace([native_row], metadata)
    assert trace["samples"][0]["body"]["legs"] == native_row["body"]["legs"]
    assert trace["landmarks"]["order"] == list(p.NATIVE_LANDMARKS)
    with pytest.raises(ValueError, match="measured landmarks"):
        p.lean_trace([row(0)], metadata)
    metadata["landmarks"]["order"] = list(reversed(p.NATIVE_LANDMARKS))
    with pytest.raises(ValueError, match="landmark contract"):
        p.lean_trace([native_row], metadata)


@pytest.mark.parametrize("byte_count", [0, -1, True])
def test_registered_empty_native_source(tmp_path, byte_count):
    root = tmp_path / "repo"
    root.mkdir()
    native = tmp_path / "__init__.py"
    native.write_bytes(b"")
    sha = p.e.digest(native)
    dependencies = {
        "native_library_sources": {
            native.as_posix(): {"sha256": sha, "bytes": byte_count}
        }
    }
    if type(byte_count) is int and byte_count == 0:
        assert p.resolve_source(root, native.as_posix(), sha, dependencies) == (
            native,
            native.as_posix(),
        )
    else:
        with pytest.raises(ValueError, match="native source descriptor"):
            p.resolve_source(root, native.as_posix(), sha, dependencies)


@pytest.mark.parametrize("override", ["factory", "clock", "meter", "output"])
def test_original_overrides_rejected_before_work(override):
    factory = FakeFactory()
    kwargs = {"s0": 0.0}
    if override == "clock":
        kwargs["clock"] = lambda: 0.0
    elif override == "meter":
        kwargs["meter"] = lambda _: {}
    elif override == "output":
        kwargs["output"] = "other"
    with pytest.raises(ValueError, match="original API"):
        p.lifecycle(factory, {}, **kwargs)
    assert factory.session is None


def test_whole_metadata_semantic122(tmp_path):
    base = p.BASE + "/e344-sol-generated-whole"
    destination = tmp_path / base
    destination.mkdir(parents=True)
    state = {"control_tick": 0, "network": {"tick": 0}, "leaves": {}}
    entries, references, files = {}, [], {}
    for i in range(122):
        name = f"array_{i:05d}.npy"
        state["leaves"][str(i)] = {"__npy__": name}
        (destination / name).write_bytes(b"generated-nonpayload-placeholder")
        files[base + "/" + name] = {"sha256": "a" * 64}
        entries[name] = "a" * 64
        references.append({"state_leaf": "leaves." + str(i), "path": base + "/" + name})
    (destination / "state.json").write_bytes(p.json_bytes(state))
    files[base + "/state.json"] = {"sha256": p.e.digest(destination / "state.json")}
    entries["state.json"] = files[base + "/state.json"]["sha256"]
    (destination / "manifest.json").write_bytes(
        p.json_bytes(
            {"schema": "flylab.state_store.v3", "files": entries, "versions": {}}
        )
    )
    inventory = {
        "initial_whole": base,
        "initial_whole_array_references": references,
        "files": files,
        "whole_runtime_versions": {},
    }
    assert p.whole_metadata(tmp_path, inventory) == state
    references[0]["state_leaf"] = "different"
    with pytest.raises(ValueError):
        p.whole_metadata(tmp_path, inventory)


def test_once_rounding_and_packed_full_coverage_cpu():
    import torch
    from scipy.sparse import csr_matrix

    from tools.banc_m3_ssp_efficacy_trial import patch_csr

    source = csr_matrix(
        (
            np.array([1.0, -2.0, 3.0, 4.0], np.float32),
            np.array([0, 2, 0, 2], np.int32),
            np.array([0, 2, 2, 4], np.int32),
        ),
        shape=(3, 3),
    )
    positions, groups, theta = (
        np.array([0, 3], np.int64),
        np.array([0, 1], np.int64),
        np.array([1.001, 0.999], np.float32).astype(np.float64),
    )
    candidate, _, selected = patch_csr(source, positions, groups, theta)
    palette = np.unique(candidate.data)
    ids = np.searchsorted(palette, candidate.data).astype(np.uint32)
    packed = (candidate.indices.astype(np.uint32) | (ids << 18)).view(np.int32)
    network = SimpleNamespace(
        n=3,
        device="cuda",
        cuda_implementation="packed",
        identity="fake-packed-CPU",
        weights=torch.sparse_csr_tensor(
            torch.tensor(candidate.indptr),
            torch.tensor(candidate.indices),
            torch.tensor(candidate.data),
            size=(3, 3),
        ),
        _rowptr=torch.tensor(candidate.indptr),
        _columns=torch.tensor(packed),
        _weight_values=torch.tensor(palette),
        _assert_pointers=lambda: None,
    )
    proof = p.verify_transfer(network, source, positions, groups, theta, selected)
    assert set(proof["leaves"]) == {
        "host.rowptr",
        "host.columns",
        "host.weights",
        "packed.rowptr",
        "packed.columns",
        "packed.palette",
    }
    network._columns[0] ^= 1
    with pytest.raises(ValueError):
        p.verify_transfer(network, source, positions, groups, theta, selected)


def test_full_fake_301_saved_publication(tmp_path, landmark_count=5):
    from tools.banc_m3_smooth_temporal import SmoothTemporalObjective

    phase = np.arange(128) * 2 * np.pi / 128
    curves = np.zeros((6, 128, 7), np.float64)
    curves[:, :, 0], curves[:, :, 1] = (
        np.cos(phase)[None, :] * 0.1,
        np.sin(phase)[None, :] * 0.1,
    )
    data = fixture_data()
    objective = SmoothTemporalObjective(
        curves,
        data["startup"]["target_command"],
        data["startup"]["target_tendon"],
        device="cpu",
    )
    data["reference"] = objective.canonical_constants
    data["saved_neutral"] = {
        "smooth": {"training_loss": 7.77211086209774},
        "shape": {"shape_loss": 0.002180594640846569},
    }
    output = p.BASE + "/e344-sol-generated-tests/full-301"
    printed = []
    result, code = p.run_generated(
        FakeFactory(landmark_count=landmark_count),
        data,
        root=tmp_path,
        output=output,
        dimensions=p.GeneratedDimensions(300),
        meter=lambda _: {"tree_rss_bytes": 1000},
        printer=printed.append,
    )
    assert code == 0 and result["counters"]["completed_rows"] == 301
    assert (
        result["counters"]["observer_samples"] == 15000
        and result["counters"]["neural_host_returned"] == 6000
    )
    assert result["metrics"] is not None and not result["gates"]["slip"]
    assert printed == [(tmp_path / output / "result.json").read_bytes()]
    (tmp_path / "worker.stdout.json").write_bytes(printed[0])
    np.savez(
        tmp_path / "fixture-inputs.npz",
        **data["reference"],
        **data["startup"],
        theta=data["theta"],
        selected_weights=data["selected_weights"],
        boundary_command=np.zeros(42),
        boundary_tendon=np.zeros(6),
    )
    (tmp_path / "fixture-descriptor.json").write_bytes(
        p.json_bytes(
            {
                "schema": "flylab.m3.e344.fake-full-fixture.v1",
                "publication": output,
                "fixture_inputs": "fixture-inputs.npz",
                "worker_stdout": "worker.stdout.json",
                "original_calls": 0,
                "actual_neural_calls": 0,
                "actual_body_calls": 0,
                "dimensions": {"controls": 300},
                "saved_neutral": data["saved_neutral"],
            }
        )
    )


def test_classification_existing_fixed_gates():
    metrics = {
        "smooth": {"training_loss": 1.0},
        "shape": {"shape_loss": 0.001},
        "summary": {
            "forward_mm": 2.0,
            "slip_mm_s": 1.0,
            "longest_continuous_forward_bout_s": 0.4,
            "complete_cycles": {k: 1 for k in p.LEGS},
            "health": {"healthy": True},
        },
    }
    neutral = {
        "smooth": {"training_loss": 7.77211086209774},
        "shape": {"shape_loss": 0.002180594640846569},
    }
    assert p.classify(metrics, neutral, {"passed": True}, engineering_valid=True)[1]
    metrics["summary"]["complete_cycles"]["rf"] = 0
    assert not p.classify(metrics, neutral, {"passed": True}, engineering_valid=True)[1]


def test_collision_and_strict_json():
    with pytest.raises(ValueError):
        p.merge_exact({"a": "1"}, {"a": "2"})
    with pytest.raises(ValueError):
        p.json_bytes({"value": float("nan")})


def test_original_namespace_forbidden(tmp_path):
    with pytest.raises(ValueError):
        p.generated_path(tmp_path, p.OUTPUT)


@pytest.mark.parametrize(
    "prefix",
    [
        "e344-sol-",
        "parent-e344-",
        "e345-sol-",
        "parent-e345-",
        "e346-sol-",
        "parent-e346-",
    ],
)
def test_generated_namespaces_accept_only_current_owners(tmp_path, prefix):
    name = p.BASE + "/" + prefix + "fixture/output"
    assert p.generated_path(tmp_path, name) == tmp_path / name


@pytest.mark.parametrize(
    "name",
    [
        p.BASE + "/parent-e343-fixture/output",
        p.BASE + "/e344-unknown/output",
        p.BASE + "/parent-e344-fixture/../other",
        "elsewhere/parent-e344-fixture/output",
    ],
)
def test_generated_namespaces_reject_unregistered_or_traversal(tmp_path, name):
    with pytest.raises(ValueError):
        p.generated_path(tmp_path, name)


def source_resolution_fixture(root):
    """Small new text files, including an external registered native source."""
    root.mkdir(parents=True, exist_ok=True)
    project = root / "project.py"
    native = root.parent / ("e344-sol-native-" + root.name + ".py")
    project.write_bytes(b"# generated project source\n")
    native.write_bytes(b"# generated external native source\n")
    digest = p.e.digest(native)
    dependencies = {
        "native_library_sources": {
            native.as_posix(): {"sha256": digest, "bytes": native.stat().st_size}
        }
    }
    return project, native, dependencies


def test_source_resolver_accepts_project_and_registered_external(tmp_path):
    root = tmp_path / "repo"
    project, native, dependencies = source_resolution_fixture(root)
    assert p.resolve_source(root, "project.py", p.e.digest(project), dependencies) == (
        project,
        "project.py",
    )
    assert p.resolve_source(
        root, native.as_posix(), p.e.digest(native), dependencies
    ) == (
        native,
        native.as_posix(),
    )


@pytest.mark.parametrize("change", ["unknown", "digest", "bytes", "traversal"])
def test_source_resolver_rejects_unregistered_or_changed_external(tmp_path, change):
    root = tmp_path / "repo"
    _, native, dependencies = source_resolution_fixture(root)
    digest, name = p.e.digest(native), native.as_posix()
    if change == "unknown":
        dependencies["native_library_sources"].clear()
    elif change == "digest":
        dependencies["native_library_sources"][name]["sha256"] = "a" * 64
    elif change == "bytes":
        dependencies["native_library_sources"][name]["bytes"] += 1
    else:
        name = str(root / ".." / native.name)
    with pytest.raises(ValueError):
        p.resolve_source(root, name, digest, dependencies)


def test_source_resolver_rejects_registered_symlink(tmp_path):
    root = tmp_path / "repo"
    _, native, dependencies = source_resolution_fixture(root)
    alias = native.with_suffix(".link.py")
    try:
        alias.symlink_to(native)
    except (OSError, NotImplementedError) as exc:
        pytest.skip("symlink creation unavailable: " + str(exc))
    digest = p.e.digest(native)
    dependencies["native_library_sources"][alias.as_posix()] = {
        "sha256": digest,
        "bytes": native.stat().st_size,
    }
    with pytest.raises(ValueError):
        p.resolve_source(root, alias.as_posix(), digest, dependencies)


def test_qualification_uses_author_independent_slots(tmp_path):
    assert p.QUALIFICATION_BRANCHES == {"cpu_smoke", "native_smoke", "root_audit"}
    old = {
        "schema": "flylab.m3.e344.current-generated-proof.v1",
        "passed": True,
        "source_sha256": {},
        "original_calls": 0,
        "evidence_sha256": {"placeholder": "a" * 64},
        "branches": {"sol_cpu": {}, "sol_native": {}, "root_audit": {}},
    }
    with pytest.raises(ValueError, match="finite current proof roots"):
        p.validate_qualification(tmp_path, old, {})


def execution_fixture(
    root, mutation=None, *, native_sources=None, helper_prefix="e344-sol-"
):
    base = root / p.BASE
    base.mkdir(parents=True)
    (base / "parent-e267-supervise.py").write_bytes(
        (p.ROOT / p.e.SUPERVISOR).read_bytes()
    )
    (root / p.DEPENDENCIES).write_bytes(
        p.json_bytes({"native_library_sources": native_sources or {}})
    )
    helper = p.BASE + "/" + helper_prefix + "fixture-helper.py"
    (root / helper).write_bytes(b"# explicitly fabricated generated metadata fixture\n")
    own = {helper: p.e.digest(root / helper)}
    sources = {**own, p.e.SUPERVISOR: p.e.SUPERVISOR_SHA}
    sources.update(
        {name: entry["sha256"] for name, entry in (native_sources or {}).items()}
    )
    names = {
        key: p.BASE + "/e344-sol-closed-" + key + ".json"
        for key in ("contract", "report", "supervision", "pids")
    }
    command = {
        "argv": [
            ".venv/Scripts/python.exe",
            "-X",
            "utf8",
            "-B",
            helper,
            "--output",
            names["report"],
        ],
        "wall_s": 5.0,
        "stdout": p.BASE + "/e344-sol-fixture.stdout",
        "stderr": p.BASE + "/e344-sol-fixture.stderr",
    }
    contract = {
        "commands": [command],
        "max_tree_rss_bytes": 6 * p.e.GIB,
        "source_sha256": sources,
        "original_calls": 0,
        "supervisor_output": names["supervision"],
    }
    attempt = {
        "argv": command["argv"].copy(),
        "wall_s": 1.0,
        "pids": [11, 12],
        "exit_code": 0,
        "requested": True,
        "killed": False,
    }
    supervision = {
        "attempts": [attempt],
        "elapsed_s": 1.1,
        "max_tree_rss_bytes": 1000,
        "owned_pids": [11, 12],
        "success": True,
        "original_calls": 0,
    }
    report = {
        "source_sha256": own,
        "original_calls": 0,
        "passed": True,
        "first_error": None,
        "checks": {"synthetic": True},
        "audit_source_path": helper,
        "audit_source_sha256": sources[helper],
    }
    pids = {
        "owned_pids": [11, 12],
        "present_after_return": {"11": False, "12": False},
        "all_owned_pids_ended": True,
        "pid_create_times": {"11": 100.0, "12": 101.0},
    }
    if mutation:
        mutation(contract, supervision, report, pids)
    loaded = {
        "contract": contract,
        "supervision": supervision,
        "report": report,
        "pids": pids,
    }
    for name in (command["stdout"], command["stderr"]):
        (root / name).write_bytes(b"")
    (root / names["contract"]).write_bytes(p.json_bytes(contract))
    supervision["contract_sha256"] = p.e.digest(root / names["contract"])
    for key in ("supervision", "report", "pids"):
        (root / names[key]).write_bytes(p.json_bytes(loaded[key]))
    evidence = {name: sha for name, sha in sources.items()}
    for name in [*names.values(), command["stdout"], command["stderr"]]:
        evidence[name] = p.e.digest(root / name)
    branch = {
        key: {"path": name, "sha256": evidence[name]} for key, name in names.items()
    }
    return branch, evidence, own


def test_closed_execution_positive_metadata_fixture(tmp_path):
    branch, evidence, own = execution_fixture(tmp_path)
    assert p.closed_execution(tmp_path, branch, evidence, own)["passed"]


def test_closed_execution_accepts_registered_external_and_root_helper(tmp_path):
    root = tmp_path / "repo"
    _, _, dependencies = source_resolution_fixture(root)
    branch, evidence, own = execution_fixture(
        root,
        native_sources=dependencies["native_library_sources"],
        helper_prefix="parent-e344-",
    )
    assert p.closed_execution(root, branch, evidence, own)["passed"]


@pytest.mark.parametrize("change", ["unknown", "digest", "bytes"])
def test_closed_execution_rejects_unregistered_external_metadata(tmp_path, change):
    root = tmp_path / "repo"
    _, native, dependencies = source_resolution_fixture(root)
    branch, evidence, own = execution_fixture(
        root, native_sources=dependencies["native_library_sources"]
    )
    spec = dependencies["native_library_sources"][native.as_posix()]
    if change == "unknown":
        dependencies["native_library_sources"].clear()
    elif change == "digest":
        spec["sha256"] = "a" * 64
    else:
        spec["bytes"] += 1
    (root / p.DEPENDENCIES).write_bytes(p.json_bytes(dependencies))
    with pytest.raises(ValueError):
        p.closed_execution(root, branch, evidence, own)


@pytest.mark.parametrize(
    "change",
    [
        "wall",
        "nanwall",
        "rss",
        "boolrss",
        "pidpresent",
        "pidmissing",
        "pidduplicate",
        "identity",
        "argv",
        "helper",
        "report",
        "originalbool",
    ],
)
def test_closed_execution_rejects_contradictory_metadata(tmp_path, change):
    def mutation(contract, supervision, report, pids):
        if change == "wall":
            supervision["elapsed_s"] = 6.0
        elif change == "nanwall":
            supervision["attempts"][0]["wall_s"] = True
        elif change == "rss":
            supervision["max_tree_rss_bytes"] = 6 * p.e.GIB + 1
        elif change == "boolrss":
            supervision["max_tree_rss_bytes"] = True
        elif change == "pidpresent":
            pids["present_after_return"]["11"] = True
        elif change == "pidmissing":
            del pids["present_after_return"]["12"]
        elif change == "pidduplicate":
            pids["owned_pids"] = [11, 11]
        elif change == "identity":
            del pids["pid_create_times"]["12"]
        elif change == "argv":
            supervision["attempts"][0]["argv"][-1] = "unrelated"
        elif change == "helper":
            report["audit_source_sha256"] = "a" * 64
        elif change == "report":
            report["audit_source_path"] = p.BASE + "/another.py"
        else:
            report["original_calls"] = False

    branch, evidence, own = execution_fixture(tmp_path, mutation)
    with pytest.raises((ValueError, KeyError)):
        p.closed_execution(tmp_path, branch, evidence, own)


class TinyNativeFactory:
    """Explicit generated400 fixture; never an original constructor substitute."""

    def __init__(self, packet):
        from tools import verify_c_banc_m3_e113_body as observer

        self.packet = packet
        self.health_stop, self.install_contact = (
            observer.NativeHealthStop,
            observer.install_native_observer,
        )
        self.encoding_calls = self.decoder_calls = 0

    def synchronize(self):
        import torch

        torch.cuda.synchronize()

    def construct(self, data, ledger, timely):
        import torch
        from scipy.sparse import csr_matrix

        from flylab.body import FlyGymBody
        from tools.banc_m3_monotone_adjoint import MonotoneDecoder
        from tools.banc_m3_ssp_efficacy_trial import patch_csr
        from tools.banc_m3_ssp_recovery_synapses import SSPRecoveryResourceNetwork

        torch.set_num_threads(1)
        packet = self.packet
        source = packet["actual_csr"]
        data["source_csr"] = csr_matrix(
            (source["data"], source["indices"], source["indptr"]), shape=(400, 400)
        )
        weights, _, selected_weights = patch_csr(
            data["source_csr"], data["positions"], data["groups"], data["theta"]
        )
        assert p.e.exact(selected_weights, data["selected_weights"])
        body = None
        try:
            with p.SetupObserver(ledger, timely):
                load, _config, roster = (
                    packet["load_parameters"],
                    packet["motor_configuration"],
                    packet["e289_roster"],
                )
                network = SSPRecoveryResourceNetwork(
                    weights,
                    np.full(400, 0.03, np.float32),
                    np.full(400, 1 / 200, np.float32),
                    load["threshold32"],
                    np.full(400, 200.0, np.float32),
                    adaptation_gain=8.0,
                    adaptation_tau_s=0.3,
                    dt=0.00025,
                    device="cuda",
                    cuda_implementation="packed",
                    capture_steps=20,
                    core_ids=np.arange(4, dtype=np.int64),
                    core_gain=8.0,
                    motor_ids=np.arange(9, 400, dtype=np.int64),
                    motor_gains=np.ones(391, np.float32),
                    motor_log_tau=np.zeros(391),
                    edge_ids=roster["edge_ids"],
                    premotor_ids=roster["premotor_ids"],
                    utilization=np.zeros(
                        len(np.unique(weights.indices[roster["edge_ids"]]))
                    ),
                    recovery_tau=np.full(
                        len(np.unique(weights.indices[roster["edge_ids"]])), 0.15
                    ),
                )
                body = FlyGymBody(
                    42,
                    {"bounds": [20.0, 20.0, 20.0], "obstacles": [], "cueOn": True},
                    {},
                    body_options={"model": "flybody", "tendons": "tarsi"},
                )
            self.synchronize()
            ledger["device_confirmed"] = True
            ledger["full_original_factory_qualified"] = False
            calibration = {k: v.copy() for k, v in packet["calibration"].items()}
            # The adjoint fixture allowed overlapping populations; native
            # recurrent resources require this explicit test-only disjoint map.
            calibration["motor_ids"] = np.arange(9, 400, dtype=np.int64)
            calibration["rest"] = body.neutral.copy()
            parameters = packet["parameters"]
            decoder = MonotoneDecoder(
                calibration,
                parameters["tau"],
                parameters["kappa"],
                parameters["slopes"],
                parameters["membership"],
                body.neutral.copy(),
                device="cuda",
            )
            factory = self

            class Session:
                def __init__(self):
                    self.body, self.network, self.control_tick, self.closed = (
                        body,
                        network,
                        0,
                        False,
                    )
                    self.adapter = SimpleNamespace(
                        identity="generated-functional-MonotoneDecoder",
                        motor_ids=calibration["motor_ids"],
                    )
                    self.graph = SimpleNamespace(hash="generated400-8edges")
                    self.p = SimpleNamespace(
                        state_dict=lambda: {
                            "generated": True,
                            "steps_per_control": 20,
                            "neural_dt": 0.00025,
                        }
                    )
                    self.tendon_inputs = {leg + "_tarsus": 0.0 for leg in p.LEGS}
                    self.rate, self.fault, self.navigation = (
                        np.zeros(391, np.float32),
                        None,
                        None,
                    )
                    self.bank = torch.zeros(
                        (391, 3), dtype=torch.float64, device="cuda"
                    )
                    self.activation = torch.zeros(
                        96, dtype=torch.float64, device="cuda"
                    )

                def advance(self, count, capture=False):
                    assert count == 1 and capture is False
                    # Measured sensors are encoded once, before actual native20.
                    observation = body.leg_observation()
                    knee = body.knee_kinematics(include_direction=True)
                    drive = np.zeros(400, np.float32)
                    drive[:4] = (
                        0.1 * np.abs(observation["angles_rad"][:4])
                        + observation["floor_normal_load_bw"][:4]
                    ).astype(np.float32)
                    assert np.isfinite(np.asarray(knee["flexion_rad"])).all()
                    factory.encoding_calls += 1
                    self.rate = network.advance_readout(
                        drive, self.adapter.motor_ids, 20
                    )
                    with torch.no_grad():
                        decoded = decoder.step(
                            torch.as_tensor(self.rate, device="cuda"),
                            self.bank,
                            self.activation,
                        )
                    self.bank, self.activation = (
                        decoded["bank"],
                        decoded["original_activation"],
                    )
                    target, tendon = (
                        decoded["q"].cpu().numpy(),
                        decoded["tendon"].cpu().numpy(),
                    )
                    factory.decoder_calls += 1
                    self.tendon_inputs = {
                        leg + "_tarsus": float(tendon[i])
                        for i, leg in enumerate(p.LEGS)
                    }
                    body.step_joint_targets(
                        target,
                        np.zeros(6, bool),
                        0.005,
                        tendon_inputs=self.tendon_inputs,
                    )
                    self.control_tick += 1

                def close(self):
                    body.close()
                    self.closed = True

            return Session()
        except Exception:
            if body is not None:
                body.close()
            raise

    def initial(self, session, data, timely):
        return p.verify_transfer(
            session.network,
            data["source_csr"],
            data["positions"],
            data["groups"],
            data["theta"],
            data["selected_weights"],
            timely=timely,
        )

    capture = staticmethod(p.capture)
    metadata = staticmethod(p.trace_metadata)
