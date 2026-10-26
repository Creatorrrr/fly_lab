"""E301 explicit fake producers/sessions; originals and real neural ticks zero."""
import copy
import hashlib
import importlib.util
import json
import time
from types import SimpleNamespace

import numpy as np
import psutil
import pytest
from threadpoolctl import threadpool_limits

from tools import verify_c_banc_m3_recovery_resource_body as probe

# Read-only reuse of an already approved fake session; no inherited tests run.
spec = importlib.util.spec_from_file_location("e301_approved_fake", probe.ROOT / probe.base.OWNED[1])
approved = importlib.util.module_from_spec(spec)
spec.loader.exec_module(approved)
BEGAN = time.perf_counter()


@pytest.fixture(autouse=True)
def single_thread():
    with threadpool_limits(limits=1):
        yield


class FakeEngine:
    def resource_metadata(self, source, selection):
        return {"schema": "fake-qualified-recovery", "source": source, **copy.deepcopy(selection)}
    def checked_resource_metadata(self, metadata):
        assert metadata["schema"] == "fake-qualified-recovery"
        return metadata
    def resource_identity(self, metadata):
        checksum = hashlib.sha256((metadata["source"] + metadata["schema"]).encode())
        for key in sorted(probe.runner.layer.SELECTION_KEYS):
            checksum.update(key.encode() + metadata[key].tobytes())
        return checksum.hexdigest()


class FakeWrapper:
    validate_selection_metadata = staticmethod(probe.runner.layer.validate_selection_metadata)
    def __init__(self):
        self.created, self.cleaned = [], []
        self.fail = False
        self.MonotoneRecruitmentTrial = SimpleNamespace(from_checkpoint=lambda graph, saved: SimpleNamespace(graph=graph, saved=saved))
    def RecoveryResourceTrial(self, owned, edges, premotor, u, tau):
        self.created.append((owned, edges.copy(), premotor.copy(), u.copy(), tau.copy()))
        if self.fail:
            raise ValueError("fake wrapper construction")
        return SimpleNamespace(network=SimpleNamespace(), owner=owned)
    def cleanup_owned(self, owned, error):
        self.cleaned.append((owned, error))
    def remove_resource_layer(self, state):
        saved = copy.deepcopy(state)
        saved["network"]["model_hash"] = saved.pop("recovery_resource_trial_patch")
        saved["network"].pop("synaptic_resource")
        return saved


def fixture(monkeypatch):
    s = {"premotor_ids": np.arange(3, dtype=np.int64), "motor_ids": np.array([3], np.int64), "source_ids": np.arange(3, dtype=np.int64),
         "edge_ids": np.arange(4, dtype=np.int64), "pre_ids": np.array([0, 0, 1, 2], np.int64), "post_ids": np.array([0, 1, 1, 2], np.int64),
         "edge_source_slots": np.array([0, 0, 1, 2], np.int32), "weights32": np.array([2, 4, -3, 5], np.float32),
         "utilization": np.array([.05, 0, 0]), "recovery_tau": np.full(3, .15), "neutral_utilization": np.zeros(3)}
    selection = {key: s[key] for key in probe.runner.layer.SELECTION_KEYS}
    old = {"schema": probe.base.FREEZE_SCHEMA, "diagnostic_id": "E296", "candidate_network_identity": "1" * 64, "neutral_network_identity": "0" * 64}
    saved = {"schema": "original", "wall_s": 0., "network": {"model_hash": "2" * 64, "tick": 0, "rate": np.zeros(4, np.float32)}, "body": {"clock": 0}}
    data = {"saved": saved, "selection": selection, "roster": s, "graph": "fake-graph", "source_ids": s["source_ids"], "motor_ids": s["motor_ids"],
            "ids": np.arange(4, dtype=np.int64), "e296_lineage": {"core_ids": [2]}}
    engine, wrapper = FakeEngine(), FakeWrapper()
    called = []
    def loader(value):
        called.append(copy.deepcopy(value))
        return copy.deepcopy(data)
    freeze = {"diagnostic_id": "E301", "base_freeze": {"path": probe.BASE_FREEZE_PATH, "sha256": "0" * 64}, "adapter_identity": "3" * 64}
    for arm, u in (("candidate", s["utilization"]), ("neutral", s["neutral_utilization"])):
        freeze[arm + "_network_identity"] = engine.resource_identity(engine.resource_metadata(saved["network"]["model_hash"], {**selection, "utilization": u}))
    monkeypatch.setattr(probe, "verify_spec", lambda _value: old)
    def construction(_arm, callback, _ledger, _persist, on_return):
        value = callback()
        on_return(value)
        return value
    deps = probe.NativeDependencies(engine=engine, wrapper=wrapper, base_loader=loader, construction=construction)
    return deps, freeze, old, data, called


def test_original_load_with_two_independent_new_identities_and_unchanged_policy(monkeypatch):
    deps, freeze, old, original, called = fixture(monkeypatch)
    data = deps.load(freeze)
    assert called == [old] and data["freeze"] is freeze
    assert freeze["candidate_network_identity"] != freeze["neutral_network_identity"]
    for key in original["roster"]:
        probe.runner.layer.require_equal(data["roster"][key], original["roster"][key], "Frozen policy unchanged")
    for arm in ("neutral", "candidate"):
        session = deps.construct(arm, data, {}, lambda: None, lambda _session: None)
        assert session._e291_arm == arm
        np.testing.assert_array_equal(deps.wrapper.created[-1][3], data["roster"]["neutral_utilization"] if arm == "neutral" else data["selection"]["utilization"])
        np.testing.assert_array_equal(deps.wrapper.created[-1][4], data["selection"]["recovery_tau"])


@pytest.mark.parametrize("fault", ["literal", "candidate", "neutral", "reuse_old"])
def test_new_engine_identity_and_literal_source_load_reject(monkeypatch, fault):
    deps, freeze, old, _, _ = fixture(monkeypatch)
    if fault == "literal":
        freeze["base_freeze"]["path"] = "wrong"
    else:
        freeze[("candidate" if fault in ("candidate", "reuse_old") else "neutral") + "_network_identity"] = old["candidate_network_identity"] if fault == "reuse_old" else "9" * 64
    with pytest.raises(ValueError):
        deps.load(freeze)


def test_failed_wrapper_construction_cleans_original_owner(monkeypatch):
    deps, freeze, _, _, _ = fixture(monkeypatch)
    data = deps.load(freeze)
    deps.wrapper.fail = True
    with pytest.raises(ValueError, match="wrapper") as caught:
        deps.construct("candidate", data, {}, lambda: None, lambda _session: None)
    assert deps.wrapper.cleaned[0][1] is caught.value


class WholeDeps(probe.NativeDependencies):
    def validate_initial_original(self, _state):
        return None  # Tiny source fixture; production uses unchanged E288 validator.
    def compare_neutral_prefix(self, _raw, _data):
        return {"exact": True}


def wrapped_state(data, identity):
    state = copy.deepcopy(data["saved"])
    state["recovery_resource_trial_patch"] = state["network"]["model_hash"]
    state["network"].update(model_hash=identity, synaptic_resource=np.ones(3, np.float64))
    return state


def test_new_layer_initial_removal_and_independent_wall_neutral_scope(monkeypatch):
    deps, freeze, _, _, _ = fixture(monkeypatch)
    data = deps.load(freeze)
    whole = WholeDeps(engine=deps.engine, wrapper=deps.wrapper, base_loader=lambda _f: data)
    state = wrapped_state(data, freeze["neutral_network_identity"])
    session = SimpleNamespace(snapshot=lambda: copy.deepcopy(state), control_tick=0,
        network=SimpleNamespace(identity=freeze["neutral_network_identity"], tick=0, device="cuda", cuda_implementation="packed", interval_steps=20),
        adapter=SimpleNamespace(identity=freeze["adapter_identity"]), body=SimpleNamespace(physics_time=lambda: 0.))
    assert whole.initial(session, data, "neutral")["excluded_fields"] == []
    baseline = copy.deepcopy(data["saved"])
    baseline["wall_s"], state["wall_s"] = .1, .2
    data["baseline_end"] = baseline
    raw = {"neutral_end_snapshot": state, "activity": [{"synaptic_resource": np.ones(3, np.float64)}]}
    result = whole.compare_neutral(raw, data)
    assert result["excluded_fields"] == ["wall_s"] and result["whole_end_exact"]
    state["body"]["clock"] = .00001
    with pytest.raises(ValueError):
        whole.compare_neutral(raw, data)


def endpoint_fixture(monkeypatch):
    deps, freeze, _, _, _ = fixture(monkeypatch)
    data = deps.load(freeze)
    rows = []
    for control in range(3):
        rows.append({"complete": True, "control_tick": control, "network_tick": control * 20, "physics_time_s": control * .005,
            "rate": np.array([100, -1e-5, 0, 0], np.float32), "drive": np.zeros(4, np.float32), "adaptation": np.zeros(4, np.float32),
            "synaptic_resource": np.array([1. if control == 0 else .4, 1, 1], np.float64)})
    return {"ids": data["ids"], "activity": rows}, data


def test_derived_endpoint_factor_labels_exemptions_and_incomplete_exclusion(monkeypatch):
    raw, data = endpoint_fixture(monkeypatch)
    raw["activity"].append({"control_tick": 3, "complete": False})
    arrays, summary = probe.endpoint_factors(raw, data, "candidate")
    assert arrays["g32"].dtype == np.float32 and arrays["g32"].shape == (3, 3)
    np.testing.assert_array_equal(arrays["g32"][:, 1:], np.ones((3, 2), np.float32))
    np.testing.assert_allclose(arrays["g32"][1:, 0], .7)
    assert summary["factor_nontrivial"] and not summary["RK_stage_measurements"] and not summary["selection_criterion"]


@pytest.mark.parametrize("fault", ["active_negative", "inactive_R", "clock", "neural", "core_u"])
def test_endpoint_derivation_rejections(monkeypatch, fault):
    raw, data = endpoint_fixture(monkeypatch)
    if fault == "active_negative":
        raw["activity"][1]["rate"][0] = -1e-6
    elif fault == "inactive_R":
        raw["activity"][1]["synaptic_resource"][1] = .9
    elif fault == "clock":
        raw["activity"][1]["physics_time_s"] += .001
    elif fault == "neural":
        raw["activity"][1]["network_tick"] += 1
    else:
        data["roster"]["utilization"][2] = .05
    with pytest.raises(ValueError):
        probe.endpoint_factors(raw, data, "candidate")


class RecoveryFakeDeps(approved.FakeDeps):
    def __init__(self, mode, guard, data):
        super().__init__(mode)
        self.guard, self.data, self.original_fault = guard, data, None
    def load(self, _freeze):
        return {**self.data, "baseline_short": approved.metrics()}
    def construct(self, arm, data, ledger, persist, on_return):
        session = super().construct(arm, data, ledger, persist, on_return)
        session._e291_arm = arm
        class PacketValidatedNetwork:
            tick = 0
            @property
            def resource_fault_flags(self):
                raise AssertionError("No successful-path device flag host copy")
        session.network = PacketValidatedNetwork()
        if self.mode == "resource" and arm == "candidate":
            def fault_advance(_count):
                approved.COUNTS["fake_controls"] += 1
                approved.COUNTS["fake_neural_ticks"] += 7
                session.network.tick = 7
                self.original_fault = RuntimeError("original fake RK resource fault")
                self.original_fault.resource_fault_flags = 4
                raise self.original_fault
            session.advance = fault_advance
        return session
    def capture(self, session, raw):
        super().capture(session, raw)
        raw["activity"][-1].update(network_tick=session.network.tick, physics_time_s=session.body.physics_time(),
            rate=np.zeros(4, np.float32), drive=np.zeros(4, np.float32), adaptation=np.zeros(4, np.float32), synaptic_resource=np.ones(3, np.float64))
    def advance(self, session, arm, index, clock, journal):
        return self.guard.advance(session, arm, index, clock, journal)
    def save_raw(self, folder, raw, data, clock):
        result = super().save_raw(folder, raw, data, clock)
        self.guard.save_endpoints(folder, raw, data)
        self.guard.write(folder / "resource-faults.json", {"events": self.guard.resource_faults})
        return result
    def save_state(self, path, session):
        if self.mode == "resource" and session.arm == "candidate" and path.name == "final":
            raise ValueError("sticky fault whole snapshot unavailable")
        return super().save_state(path, session)
    def save_partial(self, path, session):
        if self.mode == "resource":
            self.saved_partial += 1
            raise RuntimeError("faulted network component snapshot unavailable")
        return super().save_partial(path, session)


@pytest.mark.parametrize("mode,endpoints", [("neutral", [1]), ("startup", [1, 100]), ("pass", [1, 300]), ("resource", [1, 0]), ("health", [1, 0])])
def test_qualified_executor_gates_same_session_and_fault_costs(tmp_path, monkeypatch, mode, endpoints):
    guard, freeze, _, _, _ = fixture(monkeypatch)
    data = guard.load(freeze)
    deps = RecoveryFakeDeps(mode, guard, data)
    result, error = probe.execute(freeze, tmp_path / "out", deps=deps, budget=lambda: None)
    assert [session.control_tick for session in deps.sessions] == endpoints
    assert all(session.close_calls == 1 for session in deps.sessions)
    assert result["schema"] == "flylab.m3.e291.result.v1" and result["diagnostic_id"] == "E301"
    assert not result["candidate_adopted"] and not result["F1_F8_complete"]
    if mode == "resource":
        assert error is deps.original_fault and result["cost"]["scientific_neural_ticks"] == 27
        assert result["cost"]["scientific_physics_steps"] == 50 and not result["run_valid"]
        assert deps.saved_partial == 1
        records = json.loads((tmp_path / "out/candidate/resource-faults.json").read_text())["events"]
        assert records[-1]["flags"] == 4 and records[-1]["actual_neural_tick"] == 7
        assert records[-1]["original_error"]["message"] == str(error)
    elif mode == "pass":
        assert result["cost"]["scientific_neural_ticks"] == 6020 and result["next_separate_10s_eligible"]
    elif mode == "health":
        assert result["cost"]["scientific_neural_ticks"] == 40 and result["cost"]["scientific_physics_steps"] == 77
        assert result["arms"]["candidate"]["status"] == "measured_native_health_stop" and not guard.resource_faults
    else:
        assert not result["next_separate_10s_eligible"]


@pytest.mark.parametrize("fault", [None, "not_approved", "wrong_pair", "old_tiny"])
def test_exact_current_producer_reviews_and_four_source_tiny(monkeypatch, fault):
    pins = {name: str(index) * 64 for index, name in enumerate(probe.PRODUCERS)}
    specs = {key: key for key in ("e299_source_review", "e300_source_review", "recovery_tiny_qualification")}
    receipts = {key: {"approved": True, "blocking_defects": [], "source_sha256": {name: pins[name] for name in names}}
                for key, names in (("e299_source_review", probe.PRODUCERS[:2]), ("e300_source_review", probe.PRODUCERS[2:]), ("recovery_tiny_qualification", probe.PRODUCERS))}
    if fault == "not_approved":
        receipts["recovery_tiny_qualification"]["approved"] = False
    elif fault == "wrong_pair":
        receipts["e299_source_review"]["source_sha256"] = receipts["e300_source_review"]["source_sha256"]
    elif fault == "old_tiny":
        receipts["recovery_tiny_qualification"]["source_sha256"] = {name: "0" * 64 for name in probe.runner.PRODUCERS}
    monkeypatch.setattr(probe, "verify_spec", lambda name: receipts[name])
    if fault:
        with pytest.raises(ValueError):
            probe.producer_approvals(specs, pins)
    else:
        probe.producer_approvals(specs, pins)


def test_actual_owner_manifest_and_inherited_pin_preservation(tmp_path, monkeypatch):
    # E299 is still in development; fixture injects a declared fake source set.
    inherited = probe.runner.OWNED[0]
    monkeypatch.setattr(probe, "SOURCE_FILES", (*probe.OWNED, inherited))
    path = tmp_path / "source-manifest.json"
    probe.NativeDependencies().write(path, {"source_sha256": {"extra-frozen-source": "f" * 64}, "input_sha256": {"roster": "pinned"}})
    value = json.loads(path.read_text())
    assert value["diagnostic_id"] == "E301" and value["schema"] == "flylab.m3.e301.sources.v1"
    assert set(value["owned_sha256"]) == set(probe.OWNED)
    assert inherited in value["source_sha256"] and value["source_sha256"]["extra-frozen-source"] == "f" * 64
    assert value["executed_body_runner"] == "tools.verify_c_banc_m3_recurrent_resource_body.execute"


def test_sha_gate_and_generated_budget(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="SHA"):
        probe.gate(path, "0" * 64, path, "0" * 64, tmp_path / "out")
    memory = psutil.Process().memory_info()
    peak = getattr(memory, "peak_wset", memory.rss)
    elapsed = time.perf_counter() - BEGAN
    assert elapsed < 45 and peak < 1 << 30
    print(json.dumps({"generated_wall_s": elapsed, "host_peak_bytes": peak, "fake_counts": approved.COUNTS,
                      **{key: 0 for key in ("original", "model", "neural", "body", "physics", "CUDA", "solver", "fit")}}))
