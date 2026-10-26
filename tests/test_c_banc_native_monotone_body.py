"""Generated fake-only tests for E150 native monotone orchestration."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import verify_c_banc_m3_native_monotone_body as e150


def test_preflight_bad_freeze_stops_before_review_or_model(tmp_path, monkeypatch):
    entered = []
    monkeypatch.setattr(e150.protocol.GraphStore, "load",
                        lambda *_: entered.append("graph"))
    with pytest.raises(ValueError, match="Exact E150 freeze"):
        e150.verified_inputs(tmp_path / "missing", "0" * 64,
                             tmp_path / "review", "0" * 64)
    assert entered == []


def test_trial_factory_routes_source_then_wrapper_without_mutation(monkeypatch):
    calls = []

    class Base:
        network = object()

        @classmethod
        def from_checkpoint(cls, graph, saved):
            calls.append(("base", graph, saved["schema"]))
            return cls()

        def snapshot(self):
            return {"schema": "source", "network": {"model_hash": "stub"}}

    class Wrapper:
        def __init__(self, base, tau, kappa, slopes, membership):
            self.network = base.network
            calls.append(("wrap", type(base).__name__, tau.shape, kappa.shape,
                          slopes.shape, membership.shape))

        @classmethod
        def from_checkpoint(cls, graph, saved):
            calls.append(("restore", graph, saved["schema"]))
            return "restored"

    values = {"tau": np.zeros(3), "kappa": np.zeros(391),
        "slopes": np.zeros((96, 391, 3, 2)),
        "membership": np.zeros(391, np.int64)}
    factory = e150.TrialFactory(values, base_cls=Base, wrapper_cls=Wrapper)
    monkeypatch.setattr(e150, "verify_wrapper_provenance", lambda *_: None)
    monkeypatch.setattr(e150, "csr_fingerprint", lambda _: {"stub": True})
    monkeypatch.setattr(e150.protocol, "exact", lambda *_: None)
    factory.from_checkpoint("g", {"schema": "source"})
    assert factory.from_checkpoint("g", {"schema": e150.monotone.TRIAL_SCHEMA}) == "restored"
    assert [row[0] for row in calls] == ["base", "wrap", "restore"]
    assert factory.metadata()["source_restores"] == 1
    assert factory.metadata()["wrapper_restores"] == 1
    assert factory.metadata()["source_checks"][0]["before_csr"] == {"stub": True}


def test_isolated_native_body_changes_exactly_four_private_globals():
    original = e150.protocol._native_body
    before = {key: original.__globals__[key] for key in e150.OVERRIDES}
    observer = object()
    saver = object()
    factory = object()
    isolated = e150.isolated_native_body(factory, observer, saver)
    assert isolated.__code__ is original.__code__
    assert isolated.__defaults__ is original.__defaults__
    assert isolated.__closure__ is original.__closure__
    assert isolated.e150_overrides == e150.OVERRIDES
    assert {key: original.__globals__[key] for key in e150.OVERRIDES} == before
    assert isolated.__globals__["MotorTimescaleTrial"] is factory
    assert isolated.__globals__["_observe"] is observer
    assert isolated.__globals__["_save_body_raw"] is saver
    assert isolated.__globals__["INTERNAL_LIMIT_S"] == 250.
    assert all(isolated.__globals__[key] is value for key, value in
               original.__globals__.items() if key not in e150.OVERRIDES)


@pytest.mark.parametrize("changed", ["body", "network"])
def test_wrapper_provenance_rejects_changed_root_or_network(changed):
    keys = ("graph_hash", "parameters", "seed", "world", "body", "network",
        "cuts", "control_tick", "initial_position", "signed_forward_mm",
        "cpg_hash", "wall_s", "navigation")
    original = {key: {"value": key} for key in keys}
    source = json.loads(json.dumps(original))
    wrapped = json.loads(json.dumps(original))
    wrapped[changed] = {"value": "changed"}
    wrapped["monotone_recruitment_patch"] = {
        "source_motor_timescale_patch": {"tau": 1}}
    network = SimpleNamespace(identity="network")
    body = SimpleNamespace(model_hash="body")
    base = SimpleNamespace(network=network, body=body,
                           motor_timescale_patch={"tau": 1})
    candidate = SimpleNamespace(network=network, body=body,
                                snapshot=lambda: wrapped)
    with pytest.raises(AssertionError, match="wrapper source"):
        e150.verify_wrapper_provenance(original, source, base, candidate)


def fake_session(readout_error=None):
    class Network:
        tick = 20

        def readout(self, indices):
            if readout_error is not None:
                raise readout_error
            return np.arange(len(indices), dtype=np.float64)

    original = SimpleNamespace(activation=np.arange(96, dtype=np.float64))
    adapter = SimpleNamespace(bank=np.arange(391 * 3, dtype=np.float64).reshape(391, 3),
        activation=np.arange(96, dtype=np.float64) + 1, original=original,
        calibration={"motor_ids": np.arange(391, dtype=np.int64)})
    return SimpleNamespace(control_tick=1, network=Network(), adapter=adapter,
        body=SimpleNamespace(physics_time=lambda: .005))


def test_observer_records_complete_bank_and_keeps_original_readout():
    records = []

    def original(_session, _detailed, available):
        available.setdefault("fields", {})["rate"] = np.array([4.])
        return available["fields"]

    available = {}
    fields = e150.make_observer(records, original)(
        fake_session(), np.array([1]), available)
    assert records[0]["complete"] is True
    assert records[0]["bank"].shape == (391, 3)
    assert records[0]["motor_rate"].shape == (391,)
    assert fields["rate"].item() == 4
    assert np.array_equal(fields["monotone_bank"], records[0]["bank"])


def test_observer_publishes_partial_before_readout_failure():
    primary = RuntimeError("readout")
    records = []
    available = {}
    with pytest.raises(RuntimeError) as failure:
        e150.make_observer(records)(fake_session(primary), np.array([1]), available)
    assert failure.value is primary
    assert available["e150_native"] is records[0]
    assert records[0]["bank"].shape == (391, 3)
    assert records[0]["complete"] is False
    assert records[0]["error"] == repr(primary)


def test_native_state_saves_partial_availability(tmp_path):
    records = [{"control_tick": 0, "network_tick": 0, "time_s": 0.,
        "bank": np.zeros((391, 3)), "complete": False}]
    path = tmp_path / "state.npz"
    e150.save_native_state(path, records)
    with np.load(path, allow_pickle=False) as raw:
        assert raw["schema"].item() == "flylab.m3.e150.native-monotone-state.v1"
        assert raw["bank_available"].item()
        assert not raw["activation_available"].item()
        assert np.isnan(raw["activation"]).all()


def test_raw_wrapper_runs_original_and_reports_extra_save_failure(tmp_path):
    calls = []
    primary = OSError("state")
    saver = e150.make_raw_saver([], original=lambda *_a, **_k: [],
        save_state=lambda *_: (calls.append("state") or (_ for _ in ()).throw(primary)),
        relabel=lambda *_: None)
    errors = saver(tmp_path)
    assert calls == ["state"]
    assert errors[0]["artifact"] == "native-recruitment-state.npz"
    assert errors[0]["message"] == "state"


def test_raw_wrapper_preserves_original_primary_and_notes_secondary(tmp_path):
    primary = RuntimeError("original")
    secondary = OSError("state")

    def original(*_args, **_kwargs):
        raise primary

    def save(*_args):
        raise secondary

    with pytest.raises(RuntimeError) as failure:
        e150.make_raw_saver([], original=original, save_state=save,
                            relabel=lambda *_: None)(tmp_path)
    assert failure.value is primary
    assert repr(secondary) in primary.__notes__[0]


def test_raw_wrapper_reports_trace_metadata_failure(tmp_path):
    errors = e150.make_raw_saver([], original=lambda *_a, **_k: [],
        save_state=lambda *_: None,
        relabel=lambda *_: (_ for _ in ()).throw(OSError("trace")))(tmp_path)
    assert len(errors) == 1
    assert errors[0]["artifact"] == "trace.json E150 provenance"
    assert errors[0]["error"] == "OSError" and errors[0]["message"] == "trace"


class Clock:
    clock_valid = True
    neural_s = .1
    physical_s = .1


def cut_fixture():
    neutral_target = np.arange(42, dtype=np.float64)
    rows = []
    records = []
    for index in range(20):
        cut = index < 4
        rows.append({"joint_targets_rad": (neutral_target if cut else
            neutral_target + 1).tolist(),
            "motor_rates": np.zeros(391) if cut else np.ones(391),
            "tendon_inputs": {**{key: 0. if cut else 1. for key in e150.TENDON_KEYS},
                              **e150.NON_MOTOR_TENDON_HOLDS},
            "qpos": np.zeros(1), "qvel": np.zeros(1), "ctrl": np.zeros(1),
            "fault": False})
        records.append({"complete": True,
            "bank": np.zeros((391, 3)) if cut else np.ones((391, 3)),
            "activation": np.zeros(96) if cut else np.ones(96),
            "original_activation": np.zeros(96) if cut else np.ones(96),
            "motor_rate": np.zeros(391) if cut else np.ones(391)})
    return rows, records, neutral_target


def test_cut_restart_validation_requires_all_counters_and_real_restart():
    rows, records, neutral = cut_fixture()
    result = e150.validate_cut_restart(rows, records, neutral, Clock(), 1000)
    assert result["accepted"] is True
    assert result["F2_motor_transfer_evidence"] is True
    assert result["F7_complete"] is False
    assert result["preserved_non_motor_tendon_holds"] == {
        "abdomen_pitch": 0.2, "abdomen_yaw": 0.2}
    assert e150.validate_cut_restart(rows, records, neutral, Clock(), 999)[
        "accepted"] is False
    records[-1]["activation"].fill(0)
    for record in records[4:]:
        record["activation"].fill(0)
    assert e150.validate_cut_restart(rows, records, neutral, Clock(), 1000)[
        "accepted"] is False


@pytest.mark.parametrize("mutation", ["missing_hold", "changed_hold", "nan_hold",
                                      "unknown_tendon", "nonzero_motor_cut"])
def test_cut_scope_preserves_original_abdomen_holds_and_zeroes_modeled_motor_tendons(mutation):
    rows, records, neutral = cut_fixture()
    tendon = rows[0]["tendon_inputs"]
    if mutation == "missing_hold":
        tendon.pop("abdomen_pitch")
    elif mutation == "changed_hold":
        tendon["abdomen_pitch"] = 0.
    elif mutation == "nan_hold":
        tendon["abdomen_yaw"] = np.nan
    elif mutation == "unknown_tendon":
        tendon["unknown"] = 0.
    else:
        tendon["lf_tarsus"] = .01
    assert not e150.validate_cut_restart(rows, records, neutral, Clock(), 1000)["accepted"]


@pytest.mark.parametrize("mutation", ["fault", "nan_command", "missing_tendon",
                                       "nan_state"])
def test_cut_restart_rejects_fault_nonfinite_and_incomplete_tendons(mutation):
    rows, records, neutral = cut_fixture()
    if mutation == "fault":
        rows[7]["fault"] = True
    elif mutation == "nan_command":
        rows[7]["joint_targets_rad"][0] = np.nan
    elif mutation == "missing_tendon":
        rows[7]["tendon_inputs"].pop(e150.TENDON_KEYS[0])
    else:
        records[7]["bank"][0, 0] = np.nan
    assert e150.validate_cut_restart(rows, records, neutral, Clock(), 1000)[
        "accepted"] is False


def test_full_main_gate_rejects_health_partial_and_resource_stops():
    body = {"validity": "valid_full_BANC_native_transfer_diagnostic",
        "technical_failure": False, "stop_reason": "duration",
        "measured_health_stop": False, "clock_valid": True,
        "main_completed_controls": 2000, "main_native_samples": 100000,
        "clone_native_samples": 1000, "actual_neural_s": 10.1,
        "actual_physical_s": 10.1, "raw_persistence_errors": [],
        "analysis_errors": []}
    assert e150.full_main_complete(body)
    for key, value in (("measured_health_stop", True),
                       ("stop_reason", "resource_guard"),
                       ("main_completed_controls", 1999)):
        changed = {**body, key: value}
        assert not e150.full_main_complete(changed)


def test_close_preserving_keeps_first_error_and_cleanup_note():
    primary = RuntimeError("first")
    cleanup = OSError("close")
    answer = e150.close_preserving(primary, "owner", lambda: (_ for _ in ()).throw(cleanup))
    assert answer is primary and repr(cleanup) in primary.__notes__[0]
    assert e150.close_preserving(None, "owner",
        lambda: (_ for _ in ()).throw(cleanup)) is cleanup


def test_required_artifacts_reports_missing(tmp_path):
    with pytest.raises(ValueError, match="required artifacts missing"):
        e150.required_artifacts(tmp_path)


def test_resource_checks_use_frozen_boundaries():
    assert e150.resource_checks(prepare_wall=60, body_wall=180,
        process_wall=250, raw_bytes=4 << 30, free_disk_before=12 << 30,
        clock_valid=True) == []
    assert e150.resource_checks(prepare_wall=61, body_wall=181,
        process_wall=251, raw_bytes=(4 << 30) + 1,
        free_disk_before=(12 << 30) - 1, clock_valid=False) == [
            "minimum_free_disk_bytes", "prepare_wall_limit_s",
            "body_wall_limit_s", "internal_wall_limit_s",
            "maximum_raw_bytes", "clock_valid"]


def test_manifest_measurement_samples_after_delayed_required_save(tmp_path):
    time_state = [0.]

    def manifest(_out):
        time_state[0] = 251.
        return {"sha256": "x"}

    answer = e150.persist_manifest_measure(tmp_path, 0.,
        clock=lambda: time_state[0], persist_manifest=manifest,
        measure_raw=lambda _out: 17)
    assert answer == ({"sha256": "x"}, 251., 17)
    assert "internal_wall_limit_s" in e150.resource_checks(prepare_wall=0,
        body_wall=0, process_wall=answer[1], raw_bytes=answer[2],
        free_disk_before=12 << 30, clock_valid=True)


def test_manifest_measurement_preserves_persistence_primary(tmp_path):
    primary = OSError("manifest")
    with pytest.raises(OSError) as failure:
        e150.persist_manifest_measure(tmp_path, 0.,
            persist_manifest=lambda _out: (_ for _ in ()).throw(primary),
            measure_raw=lambda _out: (_ for _ in ()).throw(AssertionError()))
    assert failure.value is primary


def test_trace_relabel_changes_only_metadata_and_keeps_numeric_samples(tmp_path):
    path = tmp_path / "trace.json"
    samples = [{"time_s": 0., "joint": [1., 2.]}]
    e150.protocol.write_json(path, {"schema": "legacy", "controller": "E141",
                                    "samples": samples, "other": {"number": 4}})
    e150.relabel_trace(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["schema"] == "flylab.m3.e150.native-monotone-body-trace.v1"
    assert value["controller"].startswith("frozen E149")
    assert value["E150_provenance"]["original_protocol_metadata"] == {
        "schema": "legacy", "controller": "E141"}
    assert value["samples"] == samples and value["other"] == {"number": 4}


def test_load_parameters_rejects_incomplete_without_model(tmp_path):
    path = tmp_path / "parameters.npz"
    np.savez(path, tau=np.zeros(3))
    with pytest.raises(ValueError, match="Complete frozen"):
        e150.load_parameters(path)


def test_spec_and_result_gates_remain_no_adoption():
    source = e150.Path(e150.__file__).read_text(encoding="utf-8")
    assert '"candidate_adopted": False' in source
    assert '"F6_exposed": False' in source
    assert '"active_UI_changed": False' in source
    assert '"whole_goal_complete": False' in source
