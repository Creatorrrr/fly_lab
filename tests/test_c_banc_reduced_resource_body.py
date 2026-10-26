"""E309 fake-only qualification and pure parameter metadata; originals zero."""
import copy
import importlib.util
import json
import time
from types import SimpleNamespace

import numpy as np
import psutil
import pytest
from threadpoolctl import threadpool_limits

from tools import verify_c_banc_m3_reduced_resource_body as probe

spec = importlib.util.spec_from_file_location("e309_approved_fake", probe.ROOT / probe.e306.OWNED[1])
approved = importlib.util.module_from_spec(spec)
spec.loader.exec_module(approved)
BEGAN = time.perf_counter()
COUNTS = {"fixtures": 0, "first_control_snapshots": 0, "metadata_calls": 0}
SMALL = {"sources": 3, "active": 2, "edges": 4, "motors": 1}


@pytest.fixture(autouse=True)
def single_thread():
    with threadpool_limits(limits=1):
        yield


def original_selection():
    return {"edge_ids": np.arange(4, dtype=np.int64), "premotor_ids": np.arange(3, dtype=np.int64), "motor_ids": np.array([3], np.int64),
            "pre_ids": np.array([0, 0, 1, 2], np.int64), "post_ids": np.array([0, 1, 1, 2], np.int64), "source_ids": np.arange(3, dtype=np.int64),
            "edge_source_slots": np.array([0, 0, 1, 2], np.int32), "weights32": np.array([1, 2, 3, -4], np.float32),
            "utilization": np.array([.05, .05, 0.], np.float64), "recovery_tau": np.full(3, .15, np.float64)}


def snapshot(tick, model_hash="reduced-fake", metadata=None):
    resource = np.ones(3, np.float64)
    if tick > 0:
        resource[:2] = .9999 if tick == 20 else .8
    return {"model_hash": model_hash, "tick": tick, "rate": np.full(4, tick / 20, np.float32), "adaptation": np.zeros(4, np.float32),
            "drive": np.zeros(4, np.float32), "output_mask": np.ones(4, np.float32), "synaptic_resource": resource,
            "recurrent_resource": copy.deepcopy(metadata or {"schema": "fake-new", "source_model_hash": "source-fake"})}


class FakeEngine:
    def checked_resource_metadata(self, metadata):
        COUNTS["metadata_calls"] += 1
        assert metadata["schema"] == "fake-new"
        return metadata


def reference_prefix():
    records, time_s = [], 0.
    for _ in range(50):
        time_s += .0001
        records.append((time_s, np.ones(6), np.full(6, np.nan), np.zeros(6, bool), False, 1.))
    return probe.runner.native.native_arrays(records)


def reference_rows():
    return [{"control_tick": index, "time_s": index * .005, "joint_targets_rad": [0.] * 42,
             "tendon_inputs": {f"{leg}_tarsus": 0. for leg in probe.runner.LEGS}} for index in range(2)]


class FakeDeps(approved.FakeDeps):
    def __init__(self, mode="pass"):
        super().__init__(mode)
        COUNTS["fixtures"] += 1
        self.guard = probe.NativeDependencies(engine=FakeEngine(), wrapper=approved.FakeWrapper())
        self.first_control_checks = self.guard.first_control_checks
        self.resource_summary = {}

    def load(self, freeze):
        data = super().load(freeze)
        original = original_selection()
        selection = probe.reduced_selection(original, expected_counts=SMALL)
        data.update(ids=np.arange(4, dtype=np.int64), source_ids=np.arange(3, dtype=np.int64), selection=selection,
                    baseline_selection=original, saved={"network": {"rate": np.zeros(4, np.float32)}},
                    freeze={"reduced_candidate_network_identity": "reduced-fake", "candidate_network_identity": "old-active-fake",
                            "neutral_network_identity": "u0-fake", "source_network_identity": "source-fake"},
                    reference_neural=snapshot(20, "u0-fake"), reference_rows=reference_rows(), reference_native=reference_prefix(),
                    reduced_metadata={"schema": "fake-new", "source_model_hash": "source-fake"})
        data["reference_neural"]["synaptic_resource"] = np.ones(3, np.float64)
        self.guard.data = data
        self.data = data
        return data

    def construct(self, arm, data, ledger, persist, on_return):
        assert arm == "candidate"
        session = super().construct(arm, data, ledger, persist, on_return)
        def current():
            COUNTS["first_control_snapshots"] += 1
            state = snapshot(session.network.tick)
            if self.mode == "first_neural" and session.control_tick == 1:
                state["rate"][0] += .1
            elif self.mode == "first_resource" and session.control_tick == 1:
                state["synaptic_resource"][2] = .9
            elif self.mode == "first_identity" and session.control_tick == 1:
                state["model_hash"] = "old-active-fake"
            return state
        session.network.snapshot = current
        if self.mode == "range":
            def failed(_count):
                counts = approved.approved.COUNTS
                counts["fake_controls"] += 1
                counts["fake_neural_ticks"] += 7
                session.network.tick = 7
                self.original_error = RuntimeError("first fake reduced SSP range error")
                self.original_error.resource_fault_flags = 4
                raise self.original_error
            session.advance = failed
        return session

    def capture(self, session, raw):
        approved.approved.FakeDeps.capture(self, session, raw)
        state = snapshot(session.network.tick)
        raw["activity"][-1].update(rate=state["rate"], adaptation=state["adaptation"], drive=state["drive"], synaptic_resource=state["synaptic_resource"])
        if self.mode == "first_body" and session.control_tick == 1:
            raw["rows"][-1]["joint_targets_rad"][8] = .1
        if session.control_tick == 1:
            self.guard.first_control(session, raw)

    def save_raw(self, folder, raw, data, clock):
        if any(row.get("complete") for row in raw["activity"]):
            resource = np.stack([row["synaptic_resource"] for row in raw["activity"] if row["complete"]])
            factors = np.where(resource < .999, np.float32(.9), np.float32(1))
            self.resource_summary = probe.summarize_resources(raw, data, {"g32": factors})
        return super().save_raw(folder, raw, data, clock)

    def short_metrics(self, raw, folder, clock, data):
        result = super().short_metrics(raw, folder, clock, data)
        if self.mode == "short_fail":
            result["forward"]["fixed_initial_heading_forward_mm"] = .1
        return result


@pytest.mark.parametrize("mode,controls,valid", [("pass", 300, True), ("startup", 100, True), ("short_fail", 300, True),
    ("initial", 0, False), ("first_neural", 1, False), ("first_body", 1, False), ("first_resource", 1, False), ("first_identity", 1, False),
    ("range", 0, False), ("health", 0, True), ("readout", 2, False), ("state", 300, False), ("close", 300, False),
    ("raw", 300, False), ("constructor", None, False), ("post_return", 0, False), ("publication", 300, False)])
def test_single_candidate_startup_same_session_short_and_fault_cost(tmp_path, mode, controls, valid):
    deps = FakeDeps(mode)
    cleanup = []
    result, error = probe.execute({"diagnostic_id": "E309"}, tmp_path / "out", deps=deps, budget=lambda: None, cleanup=lambda: cleanup.append(1))
    assert cleanup == [1] and result["run_valid"] is valid and len(deps.sessions) == int(controls is not None)
    assert set(result["arms"]) == {"candidate"} and result["schema"] == probe.RESULT_SCHEMA
    if deps.sessions:
        assert deps.sessions[0].arm == "candidate" and deps.sessions[0].control_tick == controls and deps.sessions[0].close_calls == 1
    assert not result["candidate_adopted"] and not result["F1_F8_complete"] and not result["F6_exposed"] and not result["biological_claim"]
    assert not result["whole_network_reference_exact_claim"] and result["candidate_utilization"] == .005
    if mode in ("pass", "startup", "short_fail"):
        assert error is None and result["first_control_neural_fields_exact"] and result["first_control_body_native_exact"]
        assert result["first_control_comparison"]["fields"] == list(probe.NEURAL_FIELDS)
        assert result["resource_perturbation_observed"] and result["resource_observation"]["active_sources"] == 2
        assert result["cost"]["scientific_neural_ticks"] == controls * 20 and result["cost"]["scientific_physics_steps"] == controls * 50
        assert result["actual_startup_pass"] is (mode != "startup")
        assert result["short_gait_pass"] is (mode == "pass") and result["next_separate_10s_eligible"] is (mode == "pass")
        if mode == "startup":
            assert result["arms"]["candidate"]["short"]["available"] is False
        pin = json.loads((tmp_path / "out/result-sha256.json").read_text())
        assert pin["sha256"] == probe.runner.protocol.sha256(tmp_path / "out/result.json")
    elif mode == "range":
        assert error is deps.original_error and result["cost"]["scientific_neural_ticks"] == 7 and result["cost"]["scientific_physics_steps"] == 0
        assert deps.saved_partial == 1 and deps.guard.resource_faults[-1]["flags"] == 4
    elif mode == "health":
        assert error is None and result["cost"]["scientific_neural_ticks"] == 20 and result["cost"]["scientific_physics_steps"] == 27
        assert not result["next_separate_10s_eligible"]
    elif mode.startswith("first_"):
        assert result["cost"]["scientific_neural_ticks"] == 20 and "candidate" in deps.saved_neural
        assert not result["actual_startup_pass"]
    if not valid:
        assert error is not None and not result["next_separate_10s_eligible"]


def test_literal_float64_active_mask_only_and_actual_metadata_identity():
    old = original_selection()
    before = copy.deepcopy(old)
    candidate = probe.reduced_selection(old, expected_counts=SMALL)
    np.testing.assert_array_equal(candidate["utilization"], np.array([.005, .005, 0.], np.float64))
    assert candidate["utilization"].tobytes() != (old["utilization"] * .1).tobytes()
    for key in old:
        probe.runner.layer.require_equal(old[key], before[key], "Fake caller unchanged")
        if key != "utilization":
            probe.runner.layer.require_equal(candidate[key], old[key], "Fake unchanged selected bytes")
        assert not np.shares_memory(candidate[key], old[key])
    # Pure metadata only: no neural/model constructor or advance.
    engine = probe.e306.engine_api()
    a = engine.resource_identity(engine.resource_metadata("a" * 64, old))
    b = engine.resource_identity(engine.resource_metadata("a" * 64, candidate))
    assert a != b
    freeze = {"reduced_candidate_network_identity": b, "candidate_network_identity": a, "neutral_network_identity": "b" * 64, "source_network_identity": "a" * 64}
    probe.validate_reduced_identity(b, freeze)
    with pytest.raises(ValueError):
        probe.validate_reduced_identity(a, freeze)


@pytest.mark.parametrize("fault", ["dtype", "active_count", "old_value", "tau", "edge_count", "missing"])
def test_reduced_selection_rejects_changed_baseline_contract(fault):
    original = original_selection()
    if fault == "dtype":
        original["utilization"] = original["utilization"].astype(np.float32)
    elif fault == "active_count":
        original["utilization"][0] = 0.
    elif fault == "old_value":
        original["utilization"][0] = .04
    elif fault == "tau":
        original["recovery_tau"][0] = .2
    elif fault == "edge_count":
        original["edge_ids"] = original["edge_ids"][:3]
    else:
        original.pop("weights32")
    with pytest.raises(ValueError):
        probe.reduced_selection(original, expected_counts=SMALL)


def test_first_dynamic_fields_exact_with_new_metadata_and_resource_honest():
    actual, reference = snapshot(20), snapshot(20, "u0-fake")
    reference["synaptic_resource"] = np.ones(3)
    reference["recurrent_resource"]["schema"] = "fake-neutral-metadata"
    result = probe.compare_neural_fields(actual, reference)
    assert result["neural_fields_exact"] and not result["whole_network_reference_exact_claim"]
    assert result["excluded_from_reference_equality"] == ["synaptic_resource", "recurrent_resource", "model_hash"]
    actual["drive"][0] = 1.
    with pytest.raises(ValueError):
        probe.compare_neural_fields(actual, reference)


@pytest.mark.parametrize("fault", ["native_dtype", "native_clock", "body", "count"])
def test_first_body_native_mismatch_blocks(fault):
    native = reference_prefix()
    raw = {"rows": reference_rows(), "native": []}
    for index in range(50):
        raw["native"].append(tuple(native[key][index] for key in ("time_s", "normal_load_bw", "slip_mm_s", "slip_valid", "nonfoot_contact", "upright")))
    reference = reference_rows()
    if fault == "native_dtype":
        native["normal_load_bw"] = native["normal_load_bw"].astype(np.float32)
    elif fault == "native_clock":
        native["time_s"] = native["time_s"].copy()
        native["time_s"][40] += .0001
    elif fault == "body":
        reference[1]["joint_targets_rad"][8] = .1
    else:
        raw["native"].pop()
    with pytest.raises(ValueError):
        probe.compare_first_body_native(raw, reference, native)


@pytest.mark.parametrize("fault", [None, "promoted", "audit", "trajectory", "cost"])
def test_e308_valid_saved_only_lineage_semantics(fault):
    flags = {key: False for key in ("candidate_adopted", "F1_F8_complete", "F2_complete", "F6_exposed", "biological_claim", "causal_claim")}
    cost = {key: 0 for key in ("model_calls", "neural_ticks", "physics_steps", "cuda_calls", "fit_calls", "solver_calls", "retries")}
    result = {"schema": probe.e308.SCHEMA, "run_valid": True, "publication_status": "complete", "full_current_claim": False, **flags, "cost": cost}
    audit = {"schema": "flylab.m3.e308.root-saved-raw-audit.v1", "pass": True, "diagnostic_run_valid": True, "all_source_input_and_output_pins_verified": True, **flags, "cost": copy.deepcopy(cost)}
    if fault == "promoted":
        result["causal_claim"] = True
    elif fault == "audit":
        audit["pass"] = False
    elif fault == "trajectory":
        result["cost"]["model_calls"] = audit["cost"]["model_calls"] = 1
    elif fault == "cost":
        audit["cost"]["neural_ticks"] = 1
    if fault:
        with pytest.raises(ValueError):
            probe.validate_e308(result, audit)
    else:
        probe.validate_e308(result, audit)


@pytest.mark.parametrize("fail_result", [False, True])
def test_primary_range_survives_output_unlink_and_cleanup(tmp_path, monkeypatch, fail_result):
    out = tmp_path / "out"
    class ConflictDeps(FakeDeps):
        def __init__(self):
            super().__init__("range")
            self.result_writes = 0
        def write(self, path, value):
            if path.name == "result-sha256.json":
                raise PermissionError("fake SHA lock")
            if path.name == "result.json":
                self.result_writes += 1
                if fail_result and self.result_writes > 1:
                    raise PermissionError("fake invalid result lock")
            return super().write(path, value)
    deps = ConflictDeps()
    old_unlink = probe.Path.unlink
    def locked(path, *args, **kwargs):
        if path.parent == out and path.name in {"output-manifest.json", "result-sha256.json", "result.json"}:
            raise PermissionError("fake stale success lock")
        return old_unlink(path, *args, **kwargs)
    def cleanup():
        raise OSError("fake runtime cleanup secondary")
    with monkeypatch.context() as context:
        context.setattr(probe.Path, "unlink", locked)
        result, error = probe.execute({"diagnostic_id": "E309"}, out, deps=deps, budget=lambda: None, cleanup=cleanup)
    assert error is deps.original_error and not result["run_valid"] and result["publication_status"] == "invalid"
    status = json.loads((out / "publication-status.json").read_text())
    assert status["success_receipts_trusted"] is False
    errors = json.loads((out / "errors.json").read_text())["errors"]
    assert any(row["stage"] == "runtime cleanup" for row in errors)
    assert any(row["stage"] == "invalidate output-manifest.json" for row in errors)


def test_real_initial_logic_on_generated_new_layer_only():
    selection = probe.reduced_selection(original_selection(), expected_counts=SMALL)
    network = snapshot(0)
    source = copy.deepcopy(network)
    source.pop("recurrent_resource")
    source.pop("synaptic_resource")
    source["model_hash"] = "source-fake"
    saved = {"network": source, "body": {"clock": 0.}, "wall_s": 0.}
    state = {**copy.deepcopy(saved), "network": network}
    class InitialWrapper:
        def remove_resource_layer(self, value):
            result = copy.deepcopy(value)
            result["network"].pop("recurrent_resource")
            result["network"].pop("synaptic_resource")
            result["network"]["model_hash"] = "source-fake"
            return result
    class InitialGuard(probe.NativeDependencies):
        def validate_initial_original(self, value):
            assert value["network"]["tick"] == 0
    guard = InitialGuard(wrapper=InitialWrapper())
    session = SimpleNamespace(snapshot=lambda: copy.deepcopy(state), network=SimpleNamespace(identity="reduced-fake", tick=0, device="cuda", cuda_implementation="packed", interval_steps=20, selection=selection),
                              adapter=SimpleNamespace(identity="adapter-fake"), body=SimpleNamespace(physics_time=lambda: 0.), control_tick=0)
    data = {"saved": saved, "selection": selection, "reduced_metadata": network["recurrent_resource"], "source_ids": selection["source_ids"],
            "freeze": {"reduced_candidate_network_identity": "reduced-fake", "adapter_identity": "adapter-fake"}}
    result = guard.initial(session, data, "candidate")
    assert result["storage_representation_exact"] and not result["whole_network_reference_exact_claim"]
    state["body"]["clock"] = .005
    with pytest.raises(ValueError):
        guard.initial(session, data, "candidate")
    with pytest.raises(ValueError, match="only constructs"):
        probe.NativeDependencies().construct("neutral", {}, {}, lambda: None, lambda _value: None)



def test_before_after_metadata_and_selection_are_published(tmp_path):
    class RawDeps(probe.NativeDependencies):
        def inherited_save_raw(self, *_args):
            return []
    old = original_selection()
    reduced = probe.reduced_selection(old, expected_counts=SMALL)
    data = {"baseline_selection": old, "selection": reduced, "baseline_metadata": {"schema": "fake-old", "utilization": old["utilization"]},
            "reduced_metadata": {"schema": "fake-new", "utilization": reduced["utilization"]},
            "freeze": {"candidate_network_identity": "old-fake", "reduced_candidate_network_identity": "reduced-fake"}}
    assert RawDeps().save_raw(tmp_path, {}, data, None) == []
    metadata = json.loads((tmp_path / "reduced-parameter-metadata.json").read_text())
    assert metadata["baseline_metadata"]["utilization"] == [.05, .05, 0.]
    assert metadata["reduced_metadata"]["utilization"] == [.005, .005, 0.]
    with np.load(tmp_path / "reduced-parameter-selection.npz", allow_pickle=False) as arrays:
        assert arrays["baseline_utilization"].tobytes() == old["utilization"].tobytes()
        assert arrays["reduced_utilization"].tobytes() == reduced["utilization"].tobytes()
        assert arrays["baseline_weights32"].tobytes() == arrays["reduced_weights32"].tobytes()


def test_manifest_and_generated_budget(tmp_path):
    path = tmp_path / "source-manifest.json"
    probe.NativeDependencies().write(path, {"source_sha256": {}, "input_sha256": {}})
    source = json.loads(path.read_text())
    assert source["executed_body_runner"] == "tools.verify_c_banc_m3_reduced_resource_body.execute"
    assert source["reused_arm_runner"] == "tools.verify_c_banc_m3_recurrent_resource_body.run_arm"
    assert source["neutral_session_constructed"] is False and source["executed_candidate_utilization"] == .005
    assert set(probe.SOURCE_FILES) <= set(source["source_sha256"]) | set(source["owned_sha256"])
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    with pytest.raises(ValueError, match="SHA"):
        probe.gate(bad, "0" * 64, bad, "0" * 64, tmp_path / "out")
    peak = getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss)
    elapsed = time.perf_counter() - BEGAN
    assert elapsed < 60 and peak < 1536 << 20
    print(json.dumps({"generated_wall_s": elapsed, "host_peak_bytes": peak, "e309_fake_counts": COUNTS,
                      "e306_fake_counts": approved.COUNTS, "e291_fake_counts": approved.approved.COUNTS,
                      **{key: 0 for key in ("original", "model", "neural", "body", "physics", "CUDA", "solver", "fit")}}))
