"""Generated E294 sign policy/lineage tests; original/model/body/CUDA zero."""
import copy
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from tools import verify_c_banc_m3_positive_resource_body as probe


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
            "fixed_initial_heading_forward_mm": probe.prior.BASELINE_FORWARD_MM,
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
                for leg in probe.prior.LEGS
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
               "joint_targets_rad": np.zeros(42), "tendon_inputs": {f"{leg}_tarsus": 0. for leg in probe.prior.LEGS}}
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
        return probe.prior.protocol.advance_control(session, arm, index, clock, journal)

    def readout_receipt(self, journal, arm, index, event, complete, error=None):
        probe.prior.protocol.record_readout(journal, arm, index, event, complete=complete, error=error)

    def write(self, path, value):
        probe.prior.receipts.write_receipt(path, value)
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
    pre = np.array([1, 2, 3, 3], np.int64)
    return {"premotor_ids": np.array([1, 2, 3], np.int64), "motor_ids": np.array([4], np.int64),
            "edge_ids": np.array([0, 1, 2, 3], np.int64), "pre_ids": pre, "post_ids": np.array([2, 3, 1, 2], np.int64),
            "source_ids": np.array([1, 2, 3], np.int64), "edge_source_slots": np.array([0, 1, 2, 2], np.int32),
            "weights32": np.array([2, -3, 4, -5], np.float32), "utilization": np.full(3, .05),
            "neutral_utilization": np.zeros(3), "recovery_tau": np.full(3, .150),
            "descriptive_edge_groups": np.zeros(4, np.int64)}


def test_static_positive_negative_mixed_source_rule_and_owned_copies():
    original = roster()
    before = copy.deepcopy(original)
    candidate = probe.positive_source_selection(original)
    np.testing.assert_array_equal(candidate["utilization"], [.05, 0., 0.])
    for key in original:
        probe.prior.layer.require_equal(original[key], before[key], "Input unchanged " + key)
        assert not np.shares_memory(candidate[key], original[key])
        if key != "utilization":
            probe.prior.layer.require_equal(candidate[key], original[key], "Original bytes " + key)
    weights, slots = candidate["weights32"], candidate["edge_source_slots"]
    assert np.all(candidate["utilization"][slots[weights < 0]] == 0)
    np.testing.assert_array_equal(candidate["neutral_utilization"], np.zeros(3))


def test_negative_and_mixed_sources_keep_original_expression_even_for_small_resource():
    selected = probe.positive_source_selection(roster())
    for edge in np.flatnonzero(selected["weights32"] < 0):
        source = selected["edge_source_slots"][edge]
        value = probe.prior.neural.resource_edge_value(selected["weights32"][edge], np.float32(7), np.float32(1), .1, selected["utilization"][source])
        original = probe.prior.neural.resource_edge_value(selected["weights32"][edge], np.float32(7), np.float32(1), 1., 0.)
        assert value.tobytes() == original.tobytes()
    # Stable u0 resource remains one; its derivative has no activity depletion.
    u = selected["utilization"]
    derivative = probe.prior.neural.resource_derivative(np.full(5, 7, np.float32), np.ones(3), selected["source_ids"], u, selected["recovery_tau"])
    assert np.array_equal(derivative[u == 0], np.zeros(2))


@pytest.mark.parametrize("mutation", ["keys", "source_order", "duplicate", "slots", "weight_dtype", "weight_zero", "nonfinite", "utilization", "neutral", "tau", "groups"])
def test_policy_rejects_invalid_original_roster(mutation):
    value = roster()
    if mutation == "keys":
        value["extra"] = np.zeros(1)
    elif mutation == "source_order":
        value["source_ids"] = value["source_ids"][::-1].copy()
    elif mutation == "duplicate":
        value["edge_ids"][1] = 0
    elif mutation == "slots":
        value["edge_source_slots"][0] = 1
    elif mutation == "weight_dtype":
        value["weights32"] = value["weights32"].astype(np.float64)
    elif mutation == "weight_zero":
        value["weights32"][0] = 0
    elif mutation == "nonfinite":
        value["weights32"][0] = np.nan
    elif mutation in ("utilization", "neutral", "tau"):
        value[{"utilization": "utilization", "neutral": "neutral_utilization", "tau": "recovery_tau"}[mutation]][0] += .01
    else:
        value["descriptive_edge_groups"] = value["descriptive_edge_groups"].astype(np.int32)
    with pytest.raises(ValueError):
        probe.positive_source_selection(value)


def computed_freeze(base, candidate):
    s = {key: candidate[key] for key in probe.prior.layer.SELECTION_KEYS}
    metadata = probe.prior.neural.resource_metadata(base["source_network_identity"], s)
    return {"diagnostic_id": "E294", "base_freeze": {"path": probe.BASE_FREEZE_PATH, "sha256": "0" * 64},
            "paths": {"positive_selection": "candidate.npz"},
            "neutral_network_identity": base["neutral_network_identity"],
            "candidate_network_identity": probe.prior.neural.resource_identity(metadata)}


class GeneratedLoad(probe.NativeDependencies):
    def __init__(self, base, original, frozen):
        self.base, self.original, self.frozen = base, original, frozen
        self.base_loads = []
    def load_base(self, freeze):
        self.base_loads.append(copy.deepcopy(freeze))
        return {"roster": copy.deepcopy(self.original), "saved": {"network": {"model_hash": self.base["source_network_identity"]}}}
    def load_selection(self, path):
        assert path == "candidate.npz"
        return copy.deepcopy(self.frozen)


def test_literal_baseline_load_then_new_candidate_independently_checked(monkeypatch):
    original = roster()
    candidate = probe.positive_source_selection(original)
    base = {"schema": probe.prior.FREEZE_SCHEMA, "diagnostic_id": "E292", "source_network_identity": "1" * 64, "neutral_network_identity": "2" * 64}
    freeze = computed_freeze(base, candidate)
    monkeypatch.setattr(probe, "verify_spec", lambda spec: base)
    deps = GeneratedLoad(base, original, candidate)
    data = deps.load(freeze)
    assert deps.base_loads == [base] and deps.base_loads[0]["diagnostic_id"] == "E292"
    assert data["freeze"] is freeze and data["e294_lineage"]["new_candidate_separately_verified"]
    assert data["e294_lineage"]["baseline_loaded_diagnostic"] == "E292"
    np.testing.assert_array_equal(data["selection"]["utilization"], [.05, 0., 0.])
    probe.prior.layer.require_equal(original, roster(), "Original base roster unchanged")


@pytest.mark.parametrize("mutation", ["utilization", "weights", "identity", "neutral_identity"])
def test_new_frozen_candidate_or_identity_mismatch_rejected(monkeypatch, mutation):
    original = roster()
    candidate = probe.positive_source_selection(original)
    base = {"schema": probe.prior.FREEZE_SCHEMA, "diagnostic_id": "E292", "source_network_identity": "1" * 64, "neutral_network_identity": "2" * 64}
    freeze = computed_freeze(base, candidate)
    monkeypatch.setattr(probe, "verify_spec", lambda spec: base)
    if mutation == "utilization":
        candidate["utilization"][2] = .05
    elif mutation == "weights":
        candidate["weights32"][0] += 1
    else:
        freeze["candidate_network_identity" if mutation == "identity" else "neutral_network_identity"] = "3" * 64
    with pytest.raises(ValueError):
        GeneratedLoad(base, original, candidate).load(freeze)


class SelectedFakeDeps(FakeDeps):
    def load(self, freeze):
        data = super().load(freeze)
        data["roster"] = probe.positive_source_selection(roster())
        data["selection"] = {key: data["roster"][key] for key in probe.prior.layer.SELECTION_KEYS}
        return data
    def construct(self, arm, data, ledger, persist, on_return):
        vector = data["roster"]["neutral_utilization"] if arm == "neutral" else data["selection"]["utilization"]
        np.testing.assert_array_equal(vector, np.zeros(3) if arm == "neutral" else [.05, 0., 0.])
        return super().construct(arm, data, ledger, persist, on_return)


@pytest.mark.parametrize("mode,endpoints", [("neutral", [1]), ("startup", [1, 100]), ("pass", [1, 300])])
def test_inherited_actual_orchestration_routes_selected_vector_and_branches(tmp_path, mode, endpoints):
    deps = SelectedFakeDeps(mode)
    result, error = probe.execute({"diagnostic_id": "E294"}, tmp_path / "out", deps=deps, budget=lambda: None)
    assert [s.control_tick for s in deps.sessions] == endpoints
    assert all(s.closed and s.close_calls == 1 for s in deps.sessions)
    assert result["diagnostic_id"] == "E294"
    assert not result["candidate_adopted"] and not result["F1_F8_complete"] and not result["F6_exposed"]
    assert (error is not None) == (mode == "neutral")
    if mode == "pass":
        assert result["cost"]["scientific_neural_ticks"] == 6020
        assert result["next_separate_10s_eligible"]


def test_cleanup_failure_preserves_inherited_cost(tmp_path):
    def cleanup():
        raise OSError("generated cleanup error")
    deps = SelectedFakeDeps()
    result, error = probe.execute({"diagnostic_id": "E294"}, tmp_path / "out", deps=deps, budget=lambda: None, cleanup=cleanup)
    assert str(error) == "generated cleanup error" and not result["run_valid"]
    assert result["cost"]["scientific_neural_ticks"] == 6020
    assert all(s.close_calls == 1 for s in deps.sessions)


def test_source_manifest_has_actual_e294_owner(tmp_path):
    path = tmp_path / "source-manifest.json"
    inherited = {"flylab/c/__init__.py": probe.prior.protocol.sha256(probe.ROOT / "flylab/c/__init__.py")}
    probe.NativeDependencies().write(path, {"owned_sha256": {"wrong_owner": "wrong"}, "source_sha256": inherited, "input_sha256": {"input": "pin"}})
    value = json.loads(path.read_text())
    assert set(value["owned_sha256"]) == set(probe.OWNED)
    assert set(value["source_sha256"]) == (set(probe.SOURCE_FILES) - set(probe.OWNED)) | set(inherited)
    assert value["source_sha256"]["flylab/c/__init__.py"] == inherited["flylab/c/__init__.py"]
    assert value["diagnostic_id"] == "E294" and value["input_sha256"] == {"input": "pin"}


def test_gate_sha_failure_precedes_dependencies(tmp_path):
    path = tmp_path / "freeze.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="SHA"):
        probe.gate(path, "0" * 64, path, "0" * 64, tmp_path / "out")


def test_pin_groups_conflicts_and_mutated_bytes(tmp_path, monkeypatch):
    path = tmp_path / "generated.json"
    path.write_text("{}")
    monkeypatch.setattr(probe, "safe", lambda name: tmp_path / name)
    digest = probe.prior.protocol.sha256(path)
    assert probe.pin_groups({"generated.json": digest}) == {"generated.json": digest}
    with pytest.raises(ValueError, match="Conflicting"):
        probe.pin_groups({"generated.json": digest}, {"generated.json": "0" * 64})
    path.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        probe.pin_groups({"generated.json": digest})



def generated_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    def safe(name):
        from pathlib import Path
        relative = Path(name)
        target = (tmp_path / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not target.is_relative_to(tmp_path):
            raise ValueError("Repository relative frozen input required")
        return target
    monkeypatch.setattr(probe, "safe", safe)
    def put(name, value):
        path = safe(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return {"path": name, "sha256": probe.prior.protocol.sha256(path)}
    sources = {name: put(name, {"fake_source": name})["sha256"] for name in probe.SOURCE_FILES}
    paths = {alias: "inputs/" + alias + ".json" for alias in probe.prior.PATHS}
    paths["selection"] = probe.BASE_ROSTER_PATH
    inputs = {name: put(name, {"fake_input": name})["sha256"] for name in paths.values()}
    monkeypatch.setattr(probe, "BASE_ROSTER_SHA", inputs[probe.BASE_ROSTER_PATH])
    packages = {"numpy": probe.importlib.metadata.version("numpy")}
    common = {"graph_hash": "graph", "source_network_identity": "1" * 64, "adapter_identity": "adapter",
              "neutral_network_identity": "2" * 64, "packages": packages}
    old = {"schema": probe.prior.prior.FREEZE_SCHEMA, "paths": paths, "input_sha256": inputs, **common}
    old_spec = put("receipts/e279-freeze.json", old)
    old_review = put("receipts/e279-review.json", {"schema": probe.prior.prior.REVIEW_SCHEMA,
        "approved": True, "blocking_defects": [], "freeze_sha256": old_spec["sha256"],
        "source_sha256": {name: sources[name] for name in probe.prior.prior.OWNED}})
    base_design = put("base-design.json", {})
    preserved = put("preserved.json", {"M2": "unchanged"})
    base = {"schema": probe.prior.FREEZE_SCHEMA, "diagnostic_id": "E292", "owned_files": list(probe.prior.OWNED),
        "paths": paths, "input_sha256": inputs,
        "reused_source_sha256": {name: sources[name] for name in probe.prior.SOURCE_FILES if name not in probe.prior.OWNED},
        "preservation_sha256": {preserved["path"]: preserved["sha256"]}, "design": base_design["path"],
        "design_sha256": base_design["sha256"], "startup_limits": probe.STARTUP_LIMITS, "resource_limits": probe.LIMITS,
        "n": 158706, "edges": 725849, "premotor_count": 7136, "source_count": 7136, "motor_count": 391,
        "utilization": .05, "neutral_utilization": 0., "recovery_tau": .150, "candidate_adopted": False,
        "e279_freeze": old_spec, "e279_source_review": old_review, **common}
    for alias in ("e289_source_review", "e290_source_review", "tiny_qualification"):
        expected = {name: sources[name] for name in probe.prior.PRODUCERS
                    if alias == "tiny_qualification" or ("synapses" if alias == "e289_source_review" else "trial") in name}
        base[alias] = put("receipts/" + alias + ".json", {"approved": True, "blocking_defects": [], "source_sha256": expected})
    base_spec = put(probe.BASE_FREEZE_PATH, base)
    base_review = put("receipts/base-review.json", {"schema": probe.prior.REVIEW_SCHEMA, "approved": True,
        "blocking_defects": [], "freeze_sha256": base_spec["sha256"],
        "source_sha256": {name: sources[name] for name in probe.prior.OWNED}})
    audit = put("receipts/base-audit.json", {"schema": "flylab.m3.e292.root-original-raw-audit.v1",
        "pass": True, "run_valid": True, "actual_startup_pass": False, "short_gait_pass": False, "candidate_adopted": False})
    design = put("new-design.json", {})
    selection = put("inputs/positive-selection.json", {})
    freeze = {"schema": probe.FREEZE_SCHEMA, "diagnostic_id": "E294", "owned_files": list(probe.OWNED),
        "policy": probe.POLICY, "base_freeze": base_spec, "base_review": base_review, "base_original_audit": audit,
        "paths": {**paths, "positive_selection": selection["path"]},
        "input_sha256": {**inputs, selection["path"]: selection["sha256"]},
        "reused_source_sha256": {name: sources[name] for name in probe.SOURCE_FILES if name not in probe.OWNED},
        "candidate_adopted": False, "candidate_network_identity": "3" * 64,
        "design": design["path"], "design_sha256": design["sha256"], "output": probe.OUTPUT_PATH,
        **{key: base[key] for key in probe.BASE_KEYS}}
    safe(probe.OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    class NoOriginal:
        def load(self, value):
            assert value["diagnostic_id"] == "E294"
            return {}
    monkeypatch.setattr(probe, "NativeDependencies", NoOriginal)
    def call():
        spec = put("freeze.json", freeze)
        review = put("review.json", {"schema": probe.REVIEW_SCHEMA, "approved": True, "blocking_defects": [],
            "freeze_sha256": spec["sha256"], "source_sha256": {name: sources[name] for name in probe.OWNED}})
        return probe.gate(safe(spec["path"]), spec["sha256"], safe(review["path"]), review["sha256"], safe(probe.OUTPUT_PATH))
    return freeze, call, safe


def test_complete_generated_lineage_gate_without_original_load(tmp_path, monkeypatch):
    freeze, call, _ = generated_gate(tmp_path, monkeypatch)
    assert call() == freeze


@pytest.mark.parametrize("mutation", ["base_spec", "base_review", "base_audit", "new_selection", "source", "package", "criteria", "policy", "output"])
def test_frozen_new_or_base_lineage_tampering_fails_closed(tmp_path, monkeypatch, mutation):
    freeze, call, safe = generated_gate(tmp_path, monkeypatch)
    if mutation == "base_spec":
        freeze["base_freeze"]["sha256"] = "0" * 64
    elif mutation in ("base_review", "base_audit"):
        alias = "base_review" if mutation == "base_review" else "base_original_audit"
        path = safe(freeze[alias]["path"])
        value = json.loads(path.read_text())
        value["approved" if mutation == "base_review" else "pass"] = False
        path.write_text(json.dumps(value))
        freeze[alias]["sha256"] = probe.prior.protocol.sha256(path)
    elif mutation == "new_selection":
        safe(freeze["paths"]["positive_selection"]).write_text("mutated")
    elif mutation == "source":
        safe(probe.prior.OWNED[0]).write_text("mutated")
    elif mutation == "package":
        freeze["packages"] = {"numpy": "invalid"}
    elif mutation == "criteria":
        freeze["startup_limits"] = {**freeze["startup_limits"], "q_RMS": .019}
    elif mutation == "policy":
        freeze["policy"] = "all positive edges regardless of mixed source"
    else:
        safe(probe.OUTPUT_PATH).mkdir()
    with pytest.raises(ValueError):
        call()


def test_generated_cost_receipt():
    assert time.perf_counter() - BEGAN < 60
    print(json.dumps({"wall_s": time.perf_counter() - BEGAN, "fake_counts": COUNTS,
                      "original": 0, "model": 0, "CUDA": 0, "body": 0, "physics": 0, "fit": 0, "solver": 0}))
