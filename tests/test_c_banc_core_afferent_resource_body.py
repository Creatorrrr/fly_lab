"""Generated E296 shared-source core exemption; originals/model/body/CUDA zero."""
import copy
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from tools import verify_c_banc_m3_core_afferent_resource_body as probe


def metrics(candidate=False):
    return {
        "health": {
            "minimum_native_upright": 1.0,
            "nonfoot_contact_count": 0,
            **{
                key: True
                for key in (
                    "healthy",
                    "finite",
                    "complete_1p5s",
                    "clock_exact",
                    "observed_clock_valid",
                )
            },
        },
        "forward": {
            "available": True,
            "window_samples": 200,
            "last_included_control": 299,
            "fixed_initial_heading_forward_mm": probe.prior.prior.BASELINE_FORWARD_MM,
        },
        "slip": {
            "available": True,
            "load_weighted_mm_s": 1.0,
            "missing_selected_slip": 0,
            "support_samples": 20,
            "support_load_sum_bw": 1.0,
        },
        "gait": {
            "window": {
                "start_s": 0.5,
                "end_s": 1.5,
                "sampled_duration_s": 1.0,
                "bouts": [{"start_s": 0.5, "end_s": 1.2, "samples": 141}]
                if candidate
                else [],
            },
            "events": {
                leg: {
                    "complete_cycles": int(candidate),
                    "cycles": [{"aep_s": 0.6, "pep_s": 0.8, "next_aep_s": 1.0}]
                    if candidate
                    else [],
                }
                for leg in probe.prior.prior.LEGS
            },
        },
    }


COUNTS = {"fake_sessions": 0, "fake_controls": 0, "fake_neural_ticks": 0,
          "fake_native_samples": 0, "fake_partial_component_snapshots": 0,
          "fake_partial_checkpoint_save_attempts": 0,
          "original_constructors": 0, "CUDA": 0, "physics": 0, "fit": 0}
BEGAN = time.perf_counter()


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
        count = 27 if self.arm == "candidate" and self.owner.mode in ("health", "partial") else 50
        for _ in range(count):
            self.time_s += .0001
            self.native.append((self.time_s, np.ones(6), np.full(6, np.nan), np.zeros(6, bool), False, 1.))
            COUNTS["fake_native_samples"] += 1
        if self.arm == "candidate" and self.owner.mode in ("health", "partial"):
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
        return {"ids": np.arange(2, dtype=np.int64), "baseline_short": metrics()}

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
        if self.mode == "initial":
            raise ValueError("whole initial mismatch")

    def install(self, session, records):
        session.native = records
        return lambda: None

    def capture(self, session, raw):
        row = {"control_tick": session.control_tick, "time_s": session.control_tick * .005,
               "joint_targets_rad": np.zeros(42), "tendon_inputs": {f"{leg}_tarsus": 0. for leg in probe.prior.prior.LEGS}}
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
        return {"exact": True, "whole_end_exact": self.mode != "whole"}

    def startup(self, raw, data):
        assert len(raw["rows"]) == 101
        return {"strict_E141_startup_screen": self.mode != "startup"}

    def save_state(self, path, session):
        if self.mode == "state" and path.name == "final":
            raise OSError("fake state storage error")
        if self.mode in ("health", "partial") and path.name == "final" and session.arm == "candidate":
            raise ValueError("healthy snapshot only")
        self.write(path.with_suffix(".json"), {"control": session.control_tick})

    def save_partial(self, path, session):
        self.saved_partial += 1
        if self.mode == "partial":
            raise OSError("partial checkpoint failure")

    def journal(self, path):
        records = self.journal_records
        return SimpleNamespace(append=lambda value: records.append(copy.deepcopy(value)), close=lambda: None)

    def advance(self, session, arm, index, clock, journal):
        return probe.prior.prior.protocol.advance_control(session, arm, index, clock, journal)

    def readout_receipt(self, journal, arm, index, event, complete, error=None):
        probe.prior.prior.protocol.record_readout(journal, arm, index, event, complete=complete, error=error)

    def write(self, path, value):
        probe.prior.prior.receipts.write_receipt(path, value)
        if self.mode == "publication" and path.name == "output-manifest.json":
            raise OSError("fake final publication error")

    def save_raw(self, folder, raw, data, clock):
        self.write(folder / "fake-raw.json", {"rows": len(raw["rows"]), "native": len(raw["native"])})
        if self.mode == "raw":
            raise OSError("fake raw storage error")
        return []

    def short_metrics(self, raw, folder, clock, data):
        assert len(raw["rows"]) == 301 and len(raw["native"]) == 15000
        return metrics(self.mode != "short")





def roster():
    ids = np.arange(1, 5, dtype=np.int64)
    return {"premotor_ids": ids, "motor_ids": np.array([5], np.int64), "source_ids": ids.copy(),
            "edge_ids": np.arange(5, dtype=np.int64), "pre_ids": np.array([1, 1, 2, 3, 4], np.int64),
            "post_ids": np.array([2, 4, 3, 2, 4], np.int64), "edge_source_slots": np.array([0, 0, 1, 2, 3], np.int32),
            "weights32": np.array([2, 3, -1, 4, 5], np.float32), "utilization": np.full(4, .05),
            "neutral_utilization": np.zeros(4), "recovery_tau": np.full(4, .150), "descriptive_edge_groups": np.zeros(5, np.int64)}


def test_independent_shared_source_core_exemption_and_remaining_active():
    original = roster()
    before = copy.deepcopy(original)
    core = np.array([2], np.int64)
    candidate = probe.protected_source_selection(original, core)
    np.testing.assert_array_equal(candidate["utilization"], [0., 0., 0., .05])
    for key in original:
        probe.prior.prior.layer.require_equal(original[key], before[key], "Original input unchanged")
        assert not np.shares_memory(original[key], candidate[key])
        if key != "utilization":
            probe.prior.prior.layer.require_equal(candidate[key], original[key], "Other metadata exact")
    # Source1 also projects outside core2. Shared exemption preserves that edge
    # coefficient too; it does not prove future network rates or total currents.
    assert candidate["post_ids"][1] == 4 and candidate["utilization"][candidate["edge_source_slots"][1]] == 0
    assert np.count_nonzero(candidate["utilization"]) == 1
    np.testing.assert_array_equal(core, [2])


@pytest.mark.parametrize("core", [np.array([], np.int64), np.array([2, 2], np.int64), np.array([6], np.int64), np.array([-1], np.int64), np.array([2], np.int32), np.array([[2]], np.int64), [2]])
def test_invalid_core_rosters_rejected(core):
    with pytest.raises(ValueError):
        probe.protected_source_selection(roster(), core)


@pytest.mark.parametrize("field", ["utilization", "source_ids", "weights32"])
def test_original_uniform_roster_validation_is_inherited(field):
    value = roster()
    if field == "utilization":
        value[field][0] = 0
    elif field == "source_ids":
        value[field][1] = value[field][0]
    else:
        value[field] = value[field].astype(np.float64)
    with pytest.raises(ValueError):
        probe.protected_source_selection(value, np.array([2], np.int64))


def fake24roster():
    ids = np.arange(26, dtype=np.int64)
    return {"premotor_ids": ids, "motor_ids": np.array([26], np.int64), "source_ids": ids.copy(),
            "edge_ids": ids.copy(), "pre_ids": ids.copy(), "post_ids": np.roll(ids, -1),
            "edge_source_slots": ids.astype(np.int32), "weights32": np.ones(26, np.float32),
            "utilization": np.full(26, .05), "neutral_utilization": np.zeros(26), "recovery_tau": np.full(26, .150),
            "descriptive_edge_groups": np.zeros(26, np.int64)}


class GeneratedLoad(probe.NativeDependencies):
    def __init__(self, base, uniform, core_ids, protected):
        self.base, self.uniform, self.core_ids, self.protected = base, uniform, core_ids, protected
        self.loaded = []
        self.positive = probe.prior.positive_source_selection(uniform)
    def load_e294(self, value):
        self.loaded.append(copy.deepcopy(value))
        return {"roster": copy.deepcopy(self.positive), "saved": {"network": {"model_hash": self.base["source_network_identity"]}}}
    def load_selection(self, path):
        return copy.deepcopy({"uniform": self.uniform, "core": {"core_ids": self.core_ids}, "protected": self.protected}[path])


def load_fixture(monkeypatch):
    uniform, core = fake24roster(), np.arange(24, dtype=np.int64)
    protected = probe.protected_source_selection(uniform, core)
    base = {"schema": probe.prior.FREEZE_SCHEMA, "diagnostic_id": "E294",
            "source_network_identity": "1" * 64, "neutral_network_identity": "2" * 64}
    s = {key: protected[key] for key in probe.prior.prior.layer.SELECTION_KEYS}
    metadata = probe.prior.prior.neural.resource_metadata(base["source_network_identity"], s)
    freeze = {"diagnostic_id": "E296", "base_freeze": {"path": probe.BASE_FREEZE_PATH, "sha256": "0" * 64},
              "paths": {"selection": "uniform", "core_reference": "core", "protected_selection": "protected"},
              "candidate_network_identity": probe.prior.prior.neural.resource_identity(metadata),
              "neutral_network_identity": base["neutral_network_identity"]}
    monkeypatch.setattr(probe, "verify_spec", lambda _spec: base)
    return GeneratedLoad(base, uniform, core, protected), freeze


def test_literal_positive_baseline_then_separately_verified_protected_candidate(monkeypatch):
    deps, freeze = load_fixture(monkeypatch)
    data = deps.load(freeze)
    assert deps.loaded == [deps.base] and deps.loaded[0]["diagnostic_id"] == "E294"
    probe.prior.prior.layer.require_equal(data["roster"], deps.protected, "Protected actual metadata")
    assert np.count_nonzero(data["selection"]["utilization"]) == 2
    assert data["freeze"] is freeze and data["e296_lineage"]["baseline_loaded_diagnostic"] == "E294"
    assert not data["e296_lineage"]["core_rate_preservation_claim"]
    zero_s = {**data["selection"], "utilization": data["roster"]["neutral_utilization"]}
    old_zero = {key: probe.prior.positive_source_selection(deps.uniform)[key] for key in probe.prior.prior.layer.SELECTION_KEYS}
    old_zero["utilization"] = np.zeros(26)
    assert probe.prior.prior.neural.resource_identity(probe.prior.prior.neural.resource_metadata("1" * 64, zero_s)) == probe.prior.prior.neural.resource_identity(probe.prior.prior.neural.resource_metadata("1" * 64, old_zero))


@pytest.mark.parametrize("mutation", ["literal_path", "positive_baseline", "protected", "core_shape", "identity", "neutral"])
def test_loading_mismatch_fails_before_any_body(monkeypatch, mutation):
    deps, freeze = load_fixture(monkeypatch)
    if mutation == "literal_path":
        freeze["base_freeze"]["path"] = "different.json"
    elif mutation == "positive_baseline":
        deps.positive["utilization"][0] = 0
    elif mutation == "protected":
        deps.protected["utilization"][0] = .05
    elif mutation == "core_shape":
        deps.core_ids = deps.core_ids[:23]
    else:
        freeze["candidate_network_identity" if mutation == "identity" else "neutral_network_identity"] = "3" * 64
    with pytest.raises(ValueError):
        deps.load(freeze)


class ProtectedFakeDeps(FakeDeps):
    def load(self, freeze):
        data = super().load(freeze)
        data["roster"] = probe.protected_source_selection(roster(), np.array([2], np.int64))
        data["selection"] = {key: data["roster"][key] for key in probe.prior.prior.layer.SELECTION_KEYS}
        return data
    def construct(self, arm, data, ledger, persist, on_return):
        vector = data["roster"]["neutral_utilization"] if arm == "neutral" else data["selection"]["utilization"]
        np.testing.assert_array_equal(vector, np.zeros(4) if arm == "neutral" else [0., 0., 0., .05])
        return super().construct(arm, data, ledger, persist, on_return)


@pytest.mark.parametrize("mode,endpoints", [("neutral", [1]), ("startup", [1, 100]), ("pass", [1, 300])])
def test_inherited_body_orchestration_uses_protected_vector_and_original_branches(tmp_path, mode, endpoints):
    deps = ProtectedFakeDeps(mode)
    result, error = probe.execute({"diagnostic_id": "E296"}, tmp_path / "out", deps=deps, budget=lambda: None)
    assert [s.control_tick for s in deps.sessions] == endpoints
    assert all(s.close_calls == 1 for s in deps.sessions)
    assert result["diagnostic_id"] == "E296" and not result["candidate_adopted"] and not result["F1_F8_complete"]
    assert (error is not None) == (mode == "neutral")
    if mode == "pass":
        assert result["cost"]["scientific_neural_ticks"] == 6020
        assert result["next_separate_10s_eligible"]


def test_cleanup_preserves_actual_cost_and_invalidates_eligibility(tmp_path):
    def cleanup():
        raise OSError("generated cleanup")
    deps = ProtectedFakeDeps()
    result, error = probe.execute({"diagnostic_id": "E296"}, tmp_path / "out", deps=deps, budget=lambda: None, cleanup=cleanup)
    assert str(error) == "generated cleanup" and not result["run_valid"]
    assert result["cost"]["scientific_neural_ticks"] == 6020 and all(s.close_calls == 1 for s in deps.sessions)


def test_final_manifest_preserves_all_inherited_pins_and_actual_e296_ownership(tmp_path):
    extra = {"flylab/c/__init__.py": probe.prior.prior.protocol.sha256(probe.ROOT / "flylab/c/__init__.py")}
    path = tmp_path / "source-manifest.json"
    probe.NativeDependencies().write(path, {"source_sha256": extra, "input_sha256": {"uniform": "sha"}, "owned_sha256": {"wrong": "wrong"}})
    result = json.loads(path.read_text())
    assert result["schema"] == "flylab.m3.e296.sources.v1" and result["diagnostic_id"] == "E296"
    assert set(result["owned_sha256"]) == set(probe.OWNED)
    assert result["source_sha256"]["flylab/c/__init__.py"] == extra["flylab/c/__init__.py"]
    assert (set(probe.SOURCE_FILES) - set(probe.OWNED)) <= set(result["source_sha256"])
    assert result["input_sha256"] == {"uniform": "sha"}


def test_sha_gate_fails_before_any_dependencies(tmp_path):
    path = tmp_path / "freeze.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="SHA"):
        probe.gate(path, "0" * 64, path, "0" * 64, tmp_path / "out")


@pytest.mark.parametrize("mutation", [None, "source_bytes", "core_alias", "core_sha", "positive_sha", "producer_sha", "producer_bytes", "promotion"])
def test_core_reference_frozen_analysis_and_producer_provenance(tmp_path, monkeypatch, mutation):
    # Own-module fixture hooks only; inherited producers stay frozen/read-only.
    core = tmp_path / probe.CORE_REFERENCE_PATH
    core.parent.mkdir(parents=True)
    core.write_bytes(b"generated-core-reference")
    producer = core.parent.parent / "output-manifest.json"
    sha = probe.prior.prior.protocol.sha256
    producer.write_text(json.dumps({"files": {"baseline/core-activity.npz": {"sha256": sha(core), "bytes": core.stat().st_size}}}))
    producer_name = producer.relative_to(tmp_path).as_posix()
    source_pins = {}
    for name in probe.E295_OWNED:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"generated-producer-source")
        source_pins[name] = sha(path)
    evidence = {"schema": "flylab.m3.e295.core-resource-localization.freeze.v1", "owned_files": list(probe.E295_OWNED),
                "source_sha256": source_pins,
                "inputs": {"baseline_core_activity": probe.CORE_REFERENCE_PATH, "positive_roster": "positive.npz", "baseline_e257_output_manifest": producer_name},
                "input_sha256": {probe.CORE_REFERENCE_PATH: sha(core), "positive.npz": "1" * 64, producer_name: sha(producer)}}
    audit = {"schema": "flylab.m3.e295.root-saved-audit.v1", "pass": True,
             "candidate_selected": False, "causal_claim": False, "F1_F8_complete": False}
    freeze = {"e295_freeze": "evidence", "e295_root_audit": "audit", "paths": {"core_reference": probe.CORE_REFERENCE_PATH},
              "input_sha256": {probe.CORE_REFERENCE_PATH: sha(core)}}
    base = {"paths": {"positive_selection": "positive.npz"}, "input_sha256": {"positive.npz": "1" * 64}}
    monkeypatch.setattr(probe, "safe", lambda name: tmp_path / name)
    monkeypatch.setattr(probe, "verify_spec", lambda name: {"evidence": evidence, "audit": audit}[name])
    if mutation == "source_bytes":
        (tmp_path / probe.E295_OWNED[0]).write_bytes(b"changed")
    elif mutation == "core_alias":
        evidence["inputs"]["baseline_core_activity"] = "other.npz"
    elif mutation == "core_sha":
        evidence["input_sha256"][probe.CORE_REFERENCE_PATH] = "2" * 64
    elif mutation == "positive_sha":
        evidence["input_sha256"]["positive.npz"] = "2" * 64
    elif mutation == "producer_sha":
        producer.write_text("{}")
    elif mutation == "producer_bytes":
        manifest = json.loads(producer.read_text())
        manifest["files"]["baseline/core-activity.npz"]["bytes"] += 1
        producer.write_text(json.dumps(manifest))
        evidence["input_sha256"][producer_name] = sha(producer)
    elif mutation == "promotion":
        audit["causal_claim"] = True
    if mutation:
        with pytest.raises(ValueError):
            probe.core_lineage(freeze, base)
    else:
        assert probe.core_lineage(freeze, base) is evidence


def test_generated_cost_receipt():
    elapsed = time.perf_counter() - BEGAN
    assert elapsed < 60
    print(json.dumps({"wall_s": elapsed, "fake_counts": COUNTS,
                      "original": 0, "model": 0, "body": 0, "CUDA": 0, "physics": 0, "solver": 0, "fit": 0}))
