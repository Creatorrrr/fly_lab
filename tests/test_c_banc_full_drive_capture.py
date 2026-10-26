"""Generated-only E318 observer, gate, chunk, and failure contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import verify_c_banc_m3_full_drive_capture as capture


class FakeTensor:
    def __init__(self, array):
        self.array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeNetwork:
    def __init__(self):
        self.tick = 0
        self.rate = FakeTensor(np.zeros(capture.N, np.float32))
        self.adaptation = FakeTensor(np.zeros(capture.N, np.float32))
        self.output_mask = FakeTensor(np.ones(capture.N, np.float32))
        self.received = []

    def advance_readout(self, drive, ids, steps):
        self.received.append((drive, ids, steps))
        self.tick += steps
        self.rate.array[np.asarray(ids)] = np.float32(self.tick / 20.)
        self.last_result = self.rate.array[np.asarray(ids)].copy()
        return self.last_result


class FakeBody:
    fault = False


class FakeSession:
    def __init__(self):
        self.network = FakeNetwork()
        self.control_tick = 0
        self.fault = False
        self.body = FakeBody()
        self.closed = False

    def close(self):
        self.closed = True


def make_observer(session=None, on_complete=None):
    session = FakeSession() if session is None else session
    records = []
    ids = np.arange(capture.MOTOR, dtype=np.int64)
    observer = capture.DriveObserver(session, ids, on_complete or records.append, lambda row: None)
    return session, observer, records, ids


def test_instance_observer_exact_pre_drive_post_mn_and_restoration():
    session, observer, records, ids = make_observer()
    assert "advance_readout" not in vars(session.network)
    observer.install()
    drive = np.zeros(capture.N, np.float32)
    drive[1000] = np.float32(2.5)
    result = session.network.advance_readout(drive, ids, 20)
    session.control_tick = 1
    assert len(records) == 1 and records[0]["device_confirmed"]
    assert records[0]["control"] == 1 and records[0]["network_tick"] == 20
    assert np.array_equal(result, records[0]["rate"][ids])
    assert result is session.network.last_result
    assert records[0]["motor_mask"].shape == (capture.MOTOR,)
    assert session.network.received[0][0] is drive and session.network.received[0][1] is ids
    drive[1000] = 0
    assert records[0]["drive"][1000] == np.float32(2.5)
    observer.restore()
    assert "advance_readout" not in vars(session.network)
    observer.restore()


def test_prior_instance_method_restored_by_identity_and_failed_entry_recorded():
    session, observer, _, ids = make_observer()
    prior = session.network.advance_readout
    session.network.advance_readout = prior
    observer.install()
    with pytest.raises(ValueError, match="entry"):
        session.network.advance_readout(np.zeros(5, np.float32), ids, 20)
    assert len(observer.records) == 1 and not observer.records[0]["requested"]
    observer.restore()
    assert session.network.advance_readout is prior


def test_host_failure_keeps_requested_drive_and_restores_instance():
    session, observer, _, ids = make_observer()

    def fail(_drive, _ids, _steps):
        raise RuntimeError("injected host failure")

    session.network.advance_readout = fail
    observer.install()
    with pytest.raises(RuntimeError, match="injected"):
        session.network.advance_readout(np.zeros(capture.N, np.float32), ids, 20)
    row = observer.records[0]
    assert row["requested"] and not row["host_returned"] and not row["device_confirmed"]
    assert row["drive"].shape == (capture.N,)
    observer.restore()
    assert session.network.advance_readout is fail


def test_post_return_host_copy_failure_is_not_device_confirmed():
    class FailingCopy(FakeTensor):
        def cpu(self):
            raise OSError("injected host copy failure")

    session, observer, _, ids = make_observer()
    session.network.adaptation = FailingCopy(np.zeros(capture.N, np.float32))
    observer.install()
    with pytest.raises(OSError, match="host copy"):
        session.network.advance_readout(np.zeros(capture.N, np.float32), ids, 20)
    row = observer.records[0]
    assert row["requested"] and row["host_returned"] and not row["device_confirmed"]
    observer.restore()
    assert "advance_readout" not in vars(session.network)


def test_off_by_one_and_motor_roster_rejected_before_original_call():
    session, observer, _, ids = make_observer()
    observer.install()
    with pytest.raises(ValueError, match="entry"):
        session.network.advance_readout(np.zeros(capture.N, np.float32), ids[::-1], 20)
    assert not session.network.received
    observer.restore()


def test_chunk_boundary_and_partial_preserve_distinct_files(tmp_path):
    chunks = capture.ChunkWriter(tmp_path, chunk_size=2)
    for control in range(1, 4):
        row = {"control": control, "network_tick": control * 20,
               "drive": np.full(3, control, np.float32), "rate": np.full(3, control, np.float32),
               "adaptation": np.zeros(3, np.float32), "motor_rate": np.ones(2, np.float32),
               "motor_mask": np.ones(2, np.float32)}
        chunks.add(row)
    assert [r["file"] for r in chunks.receipts] == ["full-drive-001-002.npz"]
    assert (tmp_path / "full-drive-001-002.npz").exists()
    chunks.partial()
    with np.load(tmp_path / "partial-drive.npz") as part:
        assert part["control_tick"].tolist() == [3]
    chunks.flush()
    assert [r["file"] for r in chunks.receipts] == ["full-drive-001-002.npz", "full-drive-003-003.npz"]


def test_second_chunk_storage_failure_preserves_first_chunk_and_pending_rows(tmp_path):
    saves = 0

    def saver(path, **arrays):
        nonlocal saves
        saves += 1
        if saves == 2:
            raise OSError("second chunk fixture")
        np.savez_compressed(path, **arrays)

    chunks = capture.ChunkWriter(tmp_path, saver=saver, chunk_size=1)
    first = {"control": 1, "network_tick": 20, "drive": np.zeros(2, np.float32),
             "rate": np.zeros(2, np.float32), "adaptation": np.zeros(2, np.float32),
             "motor_rate": np.zeros(1, np.float32), "motor_mask": np.ones(1, np.float32)}
    chunks.add(first)
    with pytest.raises(OSError, match="second chunk"):
        chunks.add({**first, "control": 2, "network_tick": 40})
    assert (tmp_path / "full-drive-001-001.npz").exists()
    assert len(chunks.receipts) == 1 and chunks.rows[0]["control"] == 2


def test_raw_comparator_detects_dtype_and_value_change():
    expected = {"observed_ids": np.arange(2, dtype=np.int64),
                "drive": np.zeros((2, 2), np.float32)}
    capture.compare_npz(expected, expected, "fixture")
    changed = {**expected, "drive": expected["drive"].astype(np.float64)}
    with pytest.raises(ValueError, match="Raw channel differs"):
        capture.compare_npz(changed, expected, "fixture")


class FakeWholeWrapper:
    def require_saved_whole_equal(self, actual, expected, _label):
        if actual.keys() != expected.keys() or any(not capture.exact(actual[key], expected[key])
                                                   for key in actual):
            raise ValueError("whole differs")


def test_only_whole_wall_excluded():
    expected = {"wall_s": 1., "rate": np.zeros(3, np.float32)}
    actual = {"wall_s": 2., "rate": np.zeros(3, np.float32)}
    assert capture.whole_equal_except_wall(FakeWholeWrapper(), actual, expected, "fixture")["whole_except_wall_exact"]
    actual["rate"][0] = 1
    with pytest.raises(capture.ReferenceMismatch, match="Whole endpoint differs"):
        capture.whole_equal_except_wall(FakeWholeWrapper(), actual, expected, "fixture")


def frozen_gate(tmp_path):
    paths = capture.PATHS
    for name in capture.SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source", encoding="utf8")
    sources = {name: capture.sha256(tmp_path / name) for name in capture.SOURCES}
    for alias, name in paths.items():
        path = tmp_path / name
        if alias in ("graph_bundle", "start_whole", "e307_final", "e310_final"):
            path.mkdir(parents=True, exist_ok=True)
            (path / "manifest.json").write_text("{}", encoding="utf8")
        elif alias in ("e307_raw", "e310_combined"):
            path.mkdir(parents=True, exist_ok=True)
            for item in capture.RAW:
                (path / item).write_text("raw", encoding="utf8")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("metadata", encoding="utf8")
    source_manifest = tmp_path / paths["e310_source_manifest"]
    source_manifest.write_text(json.dumps({"schema": "flylab.m3.e310.sources.v1",
                                           "source_sha256": {name: sources[name] for name in capture.SOURCES
                                                             if name not in capture.OWNED}}), encoding="utf8")
    for label in ("e307", "e310"):
        producer_root = (Path(paths["e307_raw"]).parent if label == "e307"
                         else Path(paths["e310_combined"]).parents[1])
        aliases = ("start_whole", "e307_final", "e307_raw") if label == "e307" else ("e310_final", "e310_combined")
        used = {}
        for alias in aliases:
            path = tmp_path / paths[alias]
            files = (p for p in path.rglob("*") if p.is_file()) if path.is_dir() else (path,)
            for file in files:
                used[file.relative_to(tmp_path / producer_root).as_posix()] = {
                    "sha256": capture.sha256(file), "bytes": file.stat().st_size}
        output_manifest = tmp_path / paths[label + "_output_manifest"]
        output_manifest.write_text(json.dumps({"schema": f"flylab.m3.{label}.output-manifest.v1",
                                               "files": used}), encoding="utf8")
        producer = tmp_path / paths[label + "_freeze"]
        producer.write_text(json.dumps({"graph_hash": capture.GRAPH_HASH,
                                        "adapter_identity": capture.ADAPTER_ID}), encoding="utf8")
        result = tmp_path / paths[label + "_result"]
        result.write_text(json.dumps({"run_valid": True}), encoding="utf8")
        approval = tmp_path / paths[label + "_review"]
        approval.write_text(json.dumps({"approved": True, "blocking_defects": [],
                                        "freeze_sha256": capture.sha256(producer)}), encoding="utf8")
        audit = tmp_path / paths[label + "_audit"]
        audit.write_text(json.dumps({"pass": True, "diagnostic_run_valid": True,
                                     "result_sha256": capture.sha256(result),
                                     "output_manifest_sha256": capture.sha256(tmp_path / paths[label + "_output_manifest"])}),
                         encoding="utf8")
    inputs = {}
    for alias, name in paths.items():
        path = tmp_path / name
        files = list(path.rglob("*")) if alias in ("graph_bundle", "start_whole", "e307_final", "e310_final") else (
            [path / item for item in capture.RAW] if alias in ("e307_raw", "e310_combined") else [path])
        for file in files:
            if file.is_file():
                inputs[file.relative_to(tmp_path).as_posix()] = capture.sha256(file)
    freeze = {"schema": capture.SCHEMA, "paths": paths, "limits": capture.LIMITS, "output": capture.OUTPUT,
              "network_identity": capture.NETWORK_ID, "adapter_identity": capture.ADAPTER_ID,
              "graph_hash": capture.GRAPH_HASH, "candidate_adopted": False,
              "source_sha256": sources, "input_sha256": inputs}
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf8")
    freeze_hash = capture.sha256(freeze_path)
    review = {"schema": capture.REVIEW_SCHEMA, "approved": True, "blocking_defects": [],
              "freeze_sha256": freeze_hash, "source_sha256": sources}
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf8")
    return freeze_path, freeze_hash, review_path, capture.sha256(review_path), freeze


def test_passive_gate_requires_exact_sources_inputs_review_and_fresh_output(tmp_path):
    args = frozen_gate(tmp_path)
    output = tmp_path / capture.OUTPUT
    assert "tools/banc_m3_motor_timescale.py" in capture.SOURCES
    assert "flylab/c/banc_walking.py" in capture.SOURCES
    assert capture.gate(*args[:4], output, root=tmp_path) == args[4]
    changed = tmp_path / capture.PATHS["e307_result"]
    changed.write_text("changed", encoding="utf8")
    with pytest.raises(ValueError, match="input differs"):
        capture.gate(*args[:4], output, root=tmp_path)


def test_gate_links_used_raw_to_audited_output_manifest_after_repin(tmp_path):
    freeze_path, _, review_path, _, freeze = frozen_gate(tmp_path)
    output_manifest = tmp_path / capture.PATHS["e307_output_manifest"]
    listing = json.loads(output_manifest.read_text("utf8"))
    listing["files"]["neutral/trace.json"]["sha256"] = "0" * 64
    output_manifest.write_text(json.dumps(listing), encoding="utf8")
    freeze["input_sha256"][capture.PATHS["e307_output_manifest"]] = capture.sha256(output_manifest)
    audit_path = tmp_path / capture.PATHS["e307_audit"]
    audit = json.loads(audit_path.read_text("utf8"))
    audit["output_manifest_sha256"] = capture.sha256(output_manifest)
    audit_path.write_text(json.dumps(audit), encoding="utf8")
    freeze["input_sha256"][capture.PATHS["e307_audit"]] = capture.sha256(audit_path)
    freeze_path.write_text(json.dumps(freeze), encoding="utf8")
    review = json.loads(review_path.read_text("utf8"))
    review["freeze_sha256"] = capture.sha256(freeze_path)
    review_path.write_text(json.dumps(review), encoding="utf8")
    with pytest.raises(ValueError, match="audited producer output"):
        capture.gate(freeze_path, capture.sha256(freeze_path), review_path,
                     capture.sha256(review_path), tmp_path / capture.OUTPUT, root=tmp_path)


class FakeJournal:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeClock:
    def __init__(self):
        self.events = []
        self.clock_valid = True
        self.neural_s = 0.
        self.physical_s = 0.


class FailingDependencies:
    class health_stop(RuntimeError):
        pass

    def __init__(self):
        self.session = FakeSession()
        self.detached = False
        self.journal_object = FakeJournal()
        self.native_records = None
        self.partial_calls = []

    def load(self, freeze):
        return {"ids": np.arange(2, dtype=np.int64), "motor_ids": np.arange(capture.MOTOR, dtype=np.int64),
                "source_ids": np.arange(2, dtype=np.int64), "freeze": freeze}

    def construct(self, _data, ledger, _persist, on_return):
        ledger["construction"].append({"unknown_setup_steps": 0, "unknown_capture_setup_steps": 0})
        ledger["body_construction"].append({"unknown_step_entries": 0, "active_unreturned_frames": 0})
        on_return(self.session)
        return self.session

    def initial(self, _session, _data):
        return {"fake": True}

    def save_state(self, path, _session):
        self.partial_calls.append(Path(path).name)
        if Path(path).name == "partial-final":
            raise OSError("secondary partial save failure")

    def save_partial(self, path, _session):
        self.partial_calls.append(Path(path).name)

    def trace_metadata(self, _session):
        return {}

    def install(self, _session, native_records):
        self.native_records = native_records
        return lambda: setattr(self, "detached", True)

    def clock(self):
        return FakeClock()

    def journal(self, _path):
        return self.journal_object

    def capture(self, session, raw):
        raw["rows"].append({"control_tick": session.control_tick})
        raw["activity"].append({"drive": np.zeros(2, np.float32)})

    def advance(self, session, _arm, control, _clock, _journal):
        if control == 2:
            raise RuntimeError("primary advance failure")
        session.network.advance_readout(np.zeros(capture.N, np.float32),
                                        np.arange(capture.MOTOR, dtype=np.int64), 20)
        session.control_tick = control
        self.native_records.extend({"sample": i} for i in range(50))
        return {}

    def readout_receipt(self, *_args):
        return None

    def save_raw(self, _folder, _raw, _data, _clock):
        return [{"artifact": "fixture", "error": "tertiary raw save failure"}]


def test_run_keeps_primary_error_partial_drive_restores_and_closes(tmp_path):
    freeze = {"source_sha256": {}, "input_sha256": {}}
    source = tmp_path / "freeze.json"
    source.write_text("{}", encoding="utf8")
    review = tmp_path / "review.json"
    review.write_text("{}", encoding="utf8")
    deps = FailingDependencies()
    result = capture.run(source, "x", review, "y", tmp_path / "out", dependencies=deps,
                         gatekeeper=lambda *_args, **_kwargs: freeze,
                         meter=lambda: {"host_rss_bytes": 1, "host_peak_bytes": 1,
                                        "gpu_reserved_bytes": 0, "gpu_peak_reserved_bytes": 0,
                                        "device_free_bytes": 10 << 30, "device_total_bytes": 16 << 30})
    assert not result["run_valid"] and result["first_error"]["message"] == "primary advance failure"
    assert any(row["stage"] == "partial_whole" for row in result["errors"])
    assert any(row["stage"] == "raw_fixture" for row in result["errors"])
    assert deps.session.closed and deps.detached and deps.journal_object.closed
    assert "advance_readout" not in vars(deps.session.network)
    assert (tmp_path / "out/full-drive-chunks/partial-drive.npz").exists()
    assert result["cost"]["observer_device_confirmed"] == 1


def test_total_device_free_delta_is_separate_from_torch_and_owns_session_on_failure(tmp_path):
    deps = FailingDependencies()
    calls = 0

    def meter():
        nonlocal calls
        calls += 1
        return {"host_rss_bytes": 1, "host_peak_bytes": 1,
                "gpu_reserved_bytes": 0, "gpu_peak_reserved_bytes": 0,
                "device_free_bytes": (10 if calls == 1 else 5) << 30,
                "device_total_bytes": 16 << 30}

    result = capture.run(tmp_path / "freeze", "x", tmp_path / "review", "y", tmp_path / "out",
                         dependencies=deps, gatekeeper=lambda *_args, **_kwargs: {"source_sha256": {}, "input_sha256": {}},
                         meter=meter)
    assert result["first_error"]["type"] == "MemoryError"
    assert result["cost"]["additional_gpu_peak_bytes"] == 5 << 30
    assert result["cost"]["additional_torch_peak_reserved_bytes"] == 0
    assert deps.session.closed


def test_success_publication_records_terminal_finalization_and_manifest(tmp_path, monkeypatch):
    class GoodDependencies(FailingDependencies):
        def advance(self, session, _arm, control, _clock, _journal):
            session.network.advance_readout(np.zeros(capture.N, np.float32),
                                            np.arange(capture.MOTOR, dtype=np.int64), 20)
            session.control_tick = control
            self.native_records.extend({"sample": i} for i in range(50))
            return {}

        def save_raw(self, _folder, _raw, _data, _clock):
            return []

    real_chunk = capture.ChunkWriter
    monkeypatch.setattr(capture, "CONTROLS", 1)
    monkeypatch.setattr(capture, "CHUNK", 1)
    monkeypatch.setattr(capture, "LIMITS", {**capture.LIMITS, "physics_steps": 50})
    monkeypatch.setattr(capture, "ChunkWriter", lambda folder, saver: real_chunk(folder, saver, chunk_size=1))
    monkeypatch.setattr(capture, "compare_outputs", lambda *_args: {"generated_complete": True})
    result = capture.run(tmp_path / "freeze", "x", tmp_path / "review", "y", tmp_path / "out",
                         dependencies=GoodDependencies(),
                         gatekeeper=lambda *_args, **_kwargs: {"source_sha256": {}, "input_sha256": {}},
                         meter=lambda: {"host_rss_bytes": 1, "host_peak_bytes": 1,
                                        "gpu_reserved_bytes": 0, "gpu_peak_reserved_bytes": 0,
                                        "device_free_bytes": 10 << 30, "device_total_bytes": 16 << 30})
    assert result["run_valid"] and result["cost"]["finalization_wall_s"] >= 0
    assert result["cost"]["publication_boundary"] == "after_preliminary_commit_before_terminal_commit"
    ledger = json.loads((tmp_path / "out/advance-ledger.json").read_text("utf8"))
    saved = json.loads((tmp_path / "out/result.json").read_text("utf8"))
    manifest = json.loads((tmp_path / "out/output-manifest.json").read_text("utf8"))
    assert saved["cost"] == ledger == result["cost"]
    for name in ("advance-ledger.json", "result.json"):
        assert manifest["files"][name]["sha256"] == capture.sha256(tmp_path / "out" / name)

    instant = [0.]

    def slow_first_publication(path, value):
        capture.put_json(path, value)
        if path.name == "output-manifest.json" and instant[0] == 0:
            instant[0] = 136.

    late = capture.run(tmp_path / "freeze", "x", tmp_path / "review", "y", tmp_path / "late",
                       dependencies=GoodDependencies(), clock=lambda: instant[0], writer=slow_first_publication,
                       gatekeeper=lambda *_args, **_kwargs: {"source_sha256": {}, "input_sha256": {}},
                       meter=lambda: {"host_rss_bytes": 1, "host_peak_bytes": 1,
                                      "gpu_reserved_bytes": 0, "gpu_peak_reserved_bytes": 0,
                                      "device_free_bytes": 10 << 30, "device_total_bytes": 16 << 30})
    assert not late["run_valid"] and not late["learning_input_approved"]
    assert {row["stage"] for row in late["errors"]} >= {"finalization_deadline", "internal_deadline"}
    assert json.loads((tmp_path / "late/result.json").read_text("utf8"))["cost"]["elapsed_s"] == 136.

    commits = [0]

    def terminal_manifest_failure(path, value):
        if path.name == "output-manifest.json":
            commits[0] += 1
            if commits[0] >= 2:
                raise OSError("terminal manifest failure")
        capture.put_json(path, value)

    failed = capture.run(tmp_path / "freeze", "x", tmp_path / "review", "y", tmp_path / "terminal",
                         dependencies=GoodDependencies(), writer=terminal_manifest_failure,
                         gatekeeper=lambda *_args, **_kwargs: {"source_sha256": {}, "input_sha256": {}},
                         meter=lambda: {"host_rss_bytes": 1, "host_peak_bytes": 1,
                                        "gpu_reserved_bytes": 0, "gpu_peak_reserved_bytes": 0,
                                        "device_free_bytes": 10 << 30, "device_total_bytes": 16 << 30})
    assert not failed["run_valid"] and not failed["learning_input_approved"]
    assert failed["first_error"]["stage"] == "output_manifest"
    assert failed["cost"]["terminal_commit_failed"]
    for name in ("result.json", "publication-failure.json"):
        saved = json.loads((tmp_path / "terminal" / name).read_text("utf8"))
        assert not saved["run_valid"] and saved["first_error"]["stage"] == "output_manifest"


def test_owned_replace_retries_only_bounded_file_sharing(tmp_path, monkeypatch):
    original = Path.replace
    calls, retries = [], []
    pending, final = tmp_path / "pending", tmp_path / "final"
    pending.write_text("complete", encoding="utf8")

    def sharing(self, target):
        calls.append(target)
        if len(calls) < 4:
            raise PermissionError("transient Windows sharing")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", sharing)
    monkeypatch.setattr(capture.time, "sleep", lambda _: None)
    capture.replace_owned(pending, final, lambda: retries.append(True))
    assert len(calls) == 4 and len(retries) == 3
    assert final.read_text("utf8") == "complete" and not pending.exists()

    calls.clear()
    monkeypatch.setattr(Path, "replace", lambda *_: (_ for _ in ()).throw(PermissionError("persistent")))
    with pytest.raises(PermissionError, match="persistent"):
        capture.replace_owned(tmp_path / "missing", final)
