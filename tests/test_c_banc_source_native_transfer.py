"""Generated E278 orchestration tests; no original native or neural work."""
import copy
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from tools import verify_c_banc_m3_source_native_transfer as probe

COUNTS = {"fake_sessions": 0, "fake_controls": 0, "fake_neural_ticks": 0,
          "fake_native_samples": 0, "fake_partial_component_snapshots": 0,
          "fake_partial_checkpoint_save_attempts": 0,
          "original_constructors": 0, "CUDA": 0, "physics": 0, "fit": 0}
BEGAN = time.perf_counter()


def freeze():
    return {"prediction_joint_columns": {leg: list(range(i * 7, (i + 1) * 7)) for i, leg in enumerate(probe.LEGS)},
            "prediction_ltm_columns": {leg: [i] for i, leg in enumerate(probe.LEGS)}}


class FakeHealthStop(RuntimeError):
    pass


class FakeSession:
    def __init__(self, arm, owner):
        COUNTS["fake_sessions"] += 1
        self.arm, self.owner = arm, owner
        self.control_tick, self.closed, self.fault = 0, False, False
        self.network = SimpleNamespace(tick=0)
        self.body = SimpleNamespace(fault=False, physics_time=lambda: self.time_s)
        self.time_s = 0.
        self.close_calls = 0

    def advance(self, controls):
        assert controls == 1
        COUNTS["fake_controls"] += 1
        self.network.tick += 20
        COUNTS["fake_neural_ticks"] += 20
        count = 27 if self.arm == "candidate" and self.owner.mode == "health" else 50
        for _ in range(count):
            self.time_s += .0001
            self.native.append((self.time_s, np.ones(6), np.full(6, np.nan), np.zeros(6, bool), False, 1.))
            COUNTS["fake_native_samples"] += 1
        if self.arm == "candidate" and self.owner.mode == "health":
            raise FakeHealthStop("measured fake health stop")
        if self.arm == "candidate" and self.owner.mode == "advance":
            raise RuntimeError("first fake advance error")
        self.control_tick += 1

    def close(self):
        self.close_calls += 1
        self.closed = True
        if self.owner.mode == "close":
            raise RuntimeError("fake close error")


class FakeDeps:
    health_stop = FakeHealthStop

    def __init__(self, mode="pass"):
        self.mode, self.sessions = mode, []
        self.saved_partial = 0
        self.journal_records = []

    def load(self, _freeze):
        return {"ids": np.arange(2, dtype=np.int64), "predicted_q": np.zeros((101, 42)),
                "predicted_ltm": np.zeros((101, 6))}

    def construct(self, arm, data, ledger, persist, on_return):
        if self.mode == "constructor" and arm == "candidate":
            raise RuntimeError("fake constructor error")
        session = FakeSession(arm, self)
        self.sessions.append(session)
        on_return(session)
        if self.mode == "post_return":
            raise OSError("fake post-return receipt error")
        return session

    def initial(self, session, data, arm):
        assert session.control_tick == session.network.tick == 0

    def install(self, session, records):
        session.native = records
        return lambda: None

    def capture(self, session, raw):
        row = {"control_tick": session.control_tick, "time_s": session.control_tick * .005,
               "joint_targets_rad": np.zeros(42), "tendon_inputs": {f"{leg}_tarsus": 0. for leg in probe.LEGS}}
        raw["rows"].append(row)
        if self.mode == "readout" and session.arm == "candidate" and session.control_tick == 2:
            raise RuntimeError("fake readout error")
        raw["controls"].append({"control_tick": session.control_tick})
        raw["recruitment"].append({"complete": True})
        raw["activity"].append({"control_tick": session.control_tick, "complete": True})

    def compare_neutral(self, raw, data):
        if self.mode == "neutral":
            raise ValueError("fake neutral mismatch")
        assert len(raw["rows"]) == 2 and len(raw["native"]) == 50
        return {"exact": True}

    def startup(self, raw, data):
        assert len(raw["rows"]) == 101
        return {"strict_E141_startup_screen": self.mode != "startup"}

    def save_state(self, path, session):
        if self.mode == "state" and path.name == "final":
            raise OSError("fake state storage error")
        if self.mode == "health" and path.name == "final" and session.arm == "candidate":
            raise ValueError("healthy snapshot only")
        self.write(path.with_suffix(".json"), {"control": session.control_tick})

    def save_partial(self, path, session):
        self.saved_partial += 1

    def journal(self, path):
        records = self.journal_records
        return SimpleNamespace(append=lambda value: records.append(copy.deepcopy(value)), close=lambda: None)

    def advance(self, session, arm, index, clock, journal):
        return probe.protocol.advance_control(session, arm, index, clock, journal)

    def readout_receipt(self, journal, arm, index, event, complete, error=None):
        probe.protocol.record_readout(journal, arm, index, event, complete=complete, error=error)

    def write(self, path, value):
        probe.receipts.write_receipt(path, value)
        if self.mode == "publication" and path.name == "output-manifest.json":
            raise OSError("fake final publication error")

    def save_raw(self, folder, raw, data, clock):
        self.write(folder / "fake-raw.json", {"rows": len(raw["rows"]), "native": len(raw["native"])})
        if self.mode == "raw":
            raise OSError("fake raw storage error")
        return []

    def short_metrics(self, raw, folder, clock, data):
        assert len(raw["rows"]) == 301 and len(raw["native"]) == 15000
        return {"available": True, "healthy": True, "descriptive_only": True}


def execute(tmp_path, mode):
    deps = FakeDeps(mode)
    result, primary = probe.execute(freeze(), tmp_path / "run", deps=deps, budget=lambda: None)
    return deps, result, primary


def test_neutral_mismatch_prevents_candidate_construction(tmp_path):
    deps, result, primary = execute(tmp_path, "neutral")
    assert str(primary) == "fake neutral mismatch"
    assert [s.arm for s in deps.sessions] == ["neutral"]
    assert deps.sessions[0].close_calls == 1
    assert result["validity"] == "invalid_or_incomplete"


def test_startup_failure_prevents_continuation(tmp_path):
    deps, result, primary = execute(tmp_path, "startup")
    assert primary is None
    candidate = deps.sessions[1]
    assert candidate.control_tick == 100
    assert result["arms"]["candidate"]["status"] == "strict_startup_failed"
    assert result["arms"]["candidate"]["short"]["available"] is False
    assert not result["candidate_adopted"] and result["E277_status"] == "rejected"


def test_passing_startup_uses_same_continuous_candidate(tmp_path):
    deps, result, primary = execute(tmp_path, "pass")
    assert primary is None and len(deps.sessions) == 2
    assert deps.sessions[1].control_tick == 300
    assert result["cost"]["scientific_neural_ticks"] == 6020
    assert result["cost"]["scientific_physics_steps"] == 15050
    assert result["cost"]["unknown_cost_arms"] == []
    assert result["arms"]["candidate"]["short"]["descriptive_only"]
    assert not result["F6_exposed"] and not result["ten_second_executed"]


def test_measured_health_stop_is_valid_diagnostic_and_preserves_partial(tmp_path):
    deps, result, primary = execute(tmp_path, "health")
    assert primary is None
    candidate = result["arms"]["candidate"]
    assert candidate["status"] == "measured_native_health_stop" and candidate["health_stop"]
    assert deps.saved_partial == 1 and deps.sessions[1].close_calls == 1
    assert result["cost"]["scientific_neural_ticks"] == 40
    assert result["cost"]["scientific_physics_steps"] == 77


@pytest.mark.parametrize("mode,message", [("advance", "first fake advance error"), ("readout", "fake readout error"),
    ("constructor", "fake constructor error"), ("raw", "fake raw storage error"),
    ("state", "fake state storage error"), ("close", "fake close error")])
def test_technical_errors_keep_first_and_close_all_returned_owners(tmp_path, mode, message):
    deps, result, primary = execute(tmp_path, mode)
    assert primary is not None and message in str(primary)
    assert result["validity"] == "invalid_or_incomplete"
    assert all(session.close_calls == 1 for session in deps.sessions)
    assert result["first_error"]["message"] == str(primary)
    if mode == "readout":
        assert any(record["stage"] == "readout_failed" for record in deps.journal_records)


def test_constructor_postreturn_error_recovers_transferred_owner(tmp_path):
    deps, result, primary = execute(tmp_path, "post_return")
    assert "post-return receipt" in str(primary)
    assert deps.sessions[0].close_calls == 1
    ledger = json.loads((tmp_path / "run/advance-ledger.json").read_text())
    assert ledger["arms"]["neutral"]["owned_session_recovered"] is True
    assert "candidate" not in result["arms"]


def test_prediction_metrics_exclude_endpoints_and_preserve_window_boundaries():
    q = np.zeros((101, 42))
    ltm = np.zeros((101, 6))
    q[[0, 100]] = 100
    q[20, 0] = .01
    report, effects = probe.phase_metrics(q, ltm, np.zeros_like(q), np.zeros_like(ltm), freeze())
    assert report["q"]["max_abs"] == .01
    assert effects["q_error"].shape == (99, 42)
    assert report["windows"][0]["samples"] == 19
    assert report["windows"][0]["q"]["max_abs"] == 0
    assert report["windows"][1]["first_control"] == 20
    assert report["windows"][1]["q"]["max_abs"] == .01
    assert report["legs"]["lf"]["q"]["max_abs"] == .01
    assert report["legs"]["lm"]["q"]["max_abs"] == 0
    assert report["material_prediction_mismatch"] is True


def test_wrong_phase_mapping_and_nonfinite_rejected():
    wrong = freeze()
    wrong["prediction_joint_columns"]["lh"][0] = 0
    with pytest.raises(ValueError):
        probe.prediction_columns(wrong)
    q = np.zeros((101, 42))
    q[1, 1] = np.nan
    with pytest.raises(ValueError):
        probe.phase_metrics(q, np.zeros((101, 6)), np.zeros_like(q), np.zeros((101, 6)), freeze())


def test_checkpoint_representation_matches_real_storage_boundary(tmp_path):
    from flylab.c.storage import StateStore

    original = {"tuple": (np.float64(-0.), np.int64(7)), "array": np.array([1., -0.], np.float32),
                "wall_s": np.float64(3.)}
    StateStore.save(tmp_path / "stored", original)
    restored = StateStore.load(tmp_path / "stored")
    changes = []
    equivalent = probe.checkpoint_representation(original, changes)
    probe.layer.require_equal(equivalent, restored, "exact persisted bytes")
    assert len(changes) == 4 and equivalent["array"] is original["array"]
    assert isinstance(original["tuple"], tuple)
    assert not probe.layer.equal(original, restored)


@pytest.mark.parametrize("kind", ["dtype", "ulp", "signed_zero", "wall"])
def test_checkpoint_representation_does_not_relax_values(kind):
    expected = {"array": np.array([1., -0.], np.float32), "scalar": -0., "wall_s": 3.}
    actual = copy.deepcopy(expected)
    if kind == "dtype":
        actual["array"] = actual["array"].astype(np.float64)
    elif kind == "ulp":
        actual["array"][0] = np.nextafter(actual["array"][0], np.float32(2.))
    elif kind == "signed_zero":
        actual["scalar"] = np.float64(0.)
    else:
        actual["wall_s"] = np.nextafter(3., 4.)
    with pytest.raises(ValueError):
        probe.layer.require_equal(probe.checkpoint_representation(actual, []), expected, "unchanged numeric equality")


def test_science_budget_prevents_constructor(tmp_path):
    deps = FakeDeps()
    def fail():
        raise TimeoutError("generated budget cap")
    result, error = probe.execute(freeze(), tmp_path / "run", deps=deps, budget=fail)
    assert not deps.sessions and "budget cap" in str(error)
    assert result["validity"] == "invalid_or_incomplete"


def test_cleanup_failure_invalidates_result_and_retains_scientific_cost(tmp_path):
    deps = FakeDeps("startup")
    def cleanup():
        raise RuntimeError("generated runtime cleanup error")
    result, error = probe.execute(freeze(), tmp_path / "run", deps=deps, budget=lambda: None, cleanup=cleanup)
    assert "runtime cleanup" in str(error)
    assert result["validity"] == "invalid_or_incomplete"
    assert result["cost"]["scientific_neural_ticks"] == 2020


def test_hash_gate_rejects_source_changes_before_any_construct(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    source_pins = {}
    for name in probe.SOURCE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated source", encoding="utf-8")
        source_pins[name] = probe.protocol.sha256(path)
    inputs = {}
    paths = {}
    for alias in probe.PATHS - {"predictions"}:
        name = "generated/" + alias
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated input", encoding="utf-8")
        paths[alias] = name
        inputs[name] = probe.protocol.sha256(path)
    paths["predictions"] = {leg: paths["candidate"] for leg in probe.LEGS}
    design = tmp_path / "design.md"
    design.write_text("generated design", encoding="utf-8")
    frozen = {**freeze(), "schema": probe.FREEZE_SCHEMA, "owned_files": list(probe.OWNED),
        "paths": paths, "startup_limits": probe.STARTUP_LIMITS, "resource_limits": probe.LIMITS,
        "n": 158706, "edges": 1660, "source_count": 51, "motor_count": 391,
        "neutral_network_identity": "neutral", "candidate_network_identity": "candidate",
        "E277_status": "rejected", "candidate_adopted": False, "input_sha256": inputs,
        "reused_source_sha256": source_pins, "design": "design.md", "design_sha256": probe.protocol.sha256(design),
        "packages": {}, "output": "out"}
    frozen_path, review_path = tmp_path / "freeze.json", tmp_path / "review.json"
    probe.protocol.write_json(frozen_path, frozen)
    fsha = probe.protocol.sha256(frozen_path)
    review = {"schema": probe.REVIEW_SCHEMA, "approved": True, "blocking_defects": [],
              "freeze_sha256": fsha, "source_sha256": {name: source_pins[name] for name in probe.OWNED}}
    probe.protocol.write_json(review_path, review)
    rsha = probe.protocol.sha256(review_path)
    monkeypatch.setattr(probe.shutil, "disk_usage", lambda _path: SimpleNamespace(free=5 << 30))
    assert probe.gate(frozen_path, fsha, review_path, rsha, tmp_path / "out")["schema"] == probe.FREEZE_SCHEMA
    (tmp_path / probe.SOURCE_FILES[2]).write_text("changed", encoding="utf-8")
    before = COUNTS["fake_sessions"]
    with pytest.raises(ValueError, match="pinned bytes"):
        probe.gate(frozen_path, fsha, review_path, rsha, tmp_path / "out")
    assert COUNTS["fake_sessions"] == before


def test_final_publication_failure_removes_success_manifest(tmp_path):
    deps, result, error = execute(tmp_path, "publication")
    assert "final publication error" in str(error)
    assert result["validity"] == "invalid_or_incomplete"
    assert all(not arm["interpretation_validated"] for arm in result["arms"].values())
    assert not (tmp_path / "run/output-manifest.json").exists()
    persisted = json.loads((tmp_path / "run/result.json").read_text())
    assert persisted["validity"] == "invalid_or_incomplete"
    assert all(session.close_calls == 1 for session in deps.sessions)


@pytest.mark.parametrize("journal_failure", [False, True])
def test_actual_partial_storage_failure_invalidates_measured_stop(tmp_path, monkeypatch, journal_failure):
    from flylab.c.storage import StateStore

    attempted = []
    original_save = StateStore.save
    original_write = probe.receipts.write_receipt
    def store(path, saved):
        COUNTS["fake_partial_checkpoint_save_attempts"] += 1
        attempted.append(path.name)
        if path.name == "adapter":
            raise OSError("generated adapter partial disk failure")
        return original_save(path, saved)
    def write(path, value):
        if journal_failure and path.name == "component-errors.json":
            raise OSError("generated component journal failure")
        return original_write(path, value)
    monkeypatch.setattr(StateStore, "save", store)
    monkeypatch.setattr(probe.receipts, "write_receipt", write)
    class PartialDeps(FakeDeps):
        def save_partial(self, path, session):
            def snapshot():
                COUNTS["fake_partial_component_snapshots"] += 1
                return {"generated": True}
            fake = SimpleNamespace(**{key: SimpleNamespace(snapshot=snapshot)
                for key in ("network", "adapter", "body")})
            probe.NativeDependencies().save_partial(path, fake)
    deps = PartialDeps("health")
    result, error = probe.execute(freeze(), tmp_path / "run", deps=deps, budget=lambda: None)
    assert error is not None and "adapter" in str(error)
    assert isinstance(error.__cause__, OSError)
    assert error.component_errors[0]["message"] == "generated adapter partial disk failure"
    assert attempted == ["network", "adapter", "body"]
    assert result["validity"] == "invalid_or_incomplete"
    assert result["arms"]["candidate"]["health_stop"]["message"] == "measured fake health stop"
    assert result["cost"]["scientific_neural_ticks"] == 40
    assert result["cost"]["scientific_physics_steps"] == 77
    assert all(session.close_calls == 1 for session in deps.sessions)
    path = tmp_path / "run/candidate/partial-final"
    assert StateStore.load(path / "network") == {"generated": True}
    assert StateStore.load(path / "body") == {"generated": True}
    if journal_failure:
        assert any("component journal failure" in note for note in error.__notes__)
        assert any("adapter partial disk failure" in note for note in error.__notes__)
    else:
        recorded = json.loads((path / "component-errors.json").read_text())
        assert recorded[0]["component"] == "adapter"


def test_actual_partial_capture_failures_attempt_all_components(tmp_path, monkeypatch):
    from flylab.c.storage import StateStore

    calls = []
    def capture(label):
        COUNTS["fake_partial_component_snapshots"] += 1
        calls.append(label)
        if label == "adapter":
            raise ValueError("generated adapter capture failure")
        if label == "body":
            raise OSError("generated body capture failure")
        COUNTS["fake_partial_checkpoint_save_attempts"] += 1
        return {"generated": True}
    fake = SimpleNamespace(**{label: SimpleNamespace(snapshot=lambda label=label: capture(label))
        for label in ("network", "adapter", "body")})
    with pytest.raises(RuntimeError, match="adapter, body") as caught:
        probe.NativeDependencies().save_partial(tmp_path / "partial", fake)
    assert calls == ["network", "adapter", "body"]
    assert [row["component"] for row in caught.value.component_errors] == ["adapter", "body"]
    assert isinstance(caught.value.__cause__, ValueError)
    assert StateStore.load(tmp_path / "partial/network") == {"generated": True}


def test_generated_work_receipt():
    print("E278_GENERATED_COST", {**COUNTS, "fake_neural_s": COUNTS["fake_neural_ticks"] * .00025,
                                  "fake_physical_s": COUNTS["fake_native_samples"] * .0001,
                                  "wall_s": time.perf_counter() - BEGAN})
    assert COUNTS["original_constructors"] == COUNTS["CUDA"] == COUNTS["physics"] == COUNTS["fit"] == 0
