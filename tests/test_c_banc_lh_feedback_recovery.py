"""Generated E262 held-input and publication contracts; no model or body construction."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import probe_c_banc_m3_lh_feedback_recovery as recovery


def _held(abdomen=.2):
    return {"abdomen_pitch": abdomen, "abdomen_yaw": .2,
            **{f"{leg}_tarsus": .19 + index * .001 for index, leg in
               enumerate(("lf", "lm", "lh", "rf", "rm", "rh"))}}


def _session(held):
    names = list(reversed(list(held)))
    body = SimpleNamespace(tendon_control=SimpleNamespace(names=names))
    adapter = SimpleNamespace(tendon_inputs={key: held[key] for key in held
                                             if key.endswith("_tarsus")})
    return SimpleNamespace(body=body, adapter=adapter, control_tick=55,
                           tendon_inputs={"wrong_extra": .7, "abdomen_pitch": .5})


def test_full_eight_held_restore_removes_pollution_and_retains_nondefault_abdomen():
    held = _held(.17)
    session = _session(held)
    receipt = recovery.install_held_inputs(session, held)
    assert set(session.tendon_inputs) == set(held)
    assert session.tendon_inputs == held
    assert receipt["tendon_names"] == session.body.tendon_control.names
    assert receipt["tendon_inputs"]["abdomen_pitch"] == .17
    session.tendon_inputs.clear()
    session.tendon_inputs.update({"intruder": .9, "abdomen_pitch": .8})
    recovery.install_held_inputs(session, held)
    assert session.tendon_inputs == held


def test_adapter_mismatch_or_incomplete_tendon_roster_rejected():
    held = _held()
    session = _session(held)
    session.adapter.tendon_inputs["lf_tarsus"] += 1e-6
    with pytest.raises(ValueError, match="adapter six"):
        recovery.install_held_inputs(session, held)
    session = _session(held)
    session.body.tendon_control.names.pop()
    with pytest.raises(ValueError, match="roster"):
        recovery.install_held_inputs(session, held)
    assert "wrong_extra" in session.tendon_inputs


def test_sidecar_exact_schema_and_saved_control55():
    held = _held()
    sidecar = {"schema": "flylab.m3.e262.held-tendon-inputs.v1",
               "control_tick": 55, "tendon_inputs": held}
    assert recovery.validated_held_inputs(sidecar) == held
    with pytest.raises(ValueError):
        recovery.validated_held_inputs({**sidecar, "control_tick": 54})
    with pytest.raises(ValueError):
        recovery.validated_held_inputs({**sidecar,
            "tendon_inputs": {**held, "extra": .2}})


def test_complete_restore_explicitly_reinstalls_held_after_component_restore(monkeypatch):
    held = _held(.13)
    session = _session(held)
    def underlying(s, fork):
        assert fork == {"control_tick": 55}
        s.tendon_inputs.clear()
        s.tendon_inputs.update(s.adapter.tendon_inputs)
    monkeypatch.setattr(recovery.e261, "restore_in_place", underlying)
    monkeypatch.setattr(recovery, "exact", lambda *args: None)
    session.snapshot = lambda: {"control_tick": session.control_tick}
    receipt = recovery.restore_complete(session, {"control_tick": 55}, held)
    assert session.tendon_inputs == held
    assert receipt["tendon_inputs"]["abdomen_pitch"] == .13


def test_dynamic_six_tarsus_endpoint_is_valid_while_abdomen_is_held():
    start = _held(.17)
    session = _session(start)
    recovery.install_held_inputs(session, start)
    changed = {key: (value + .003 if key.endswith("_tarsus") else value)
               for key, value in start.items()}
    session.adapter.tendon_inputs = {key: changed[key] for key in changed
                                     if key.endswith("_tarsus")}
    session.tendon_inputs.update(session.adapter.tendon_inputs)
    end = recovery.validate_end_held(session, start, {"tendon_inputs": changed})
    assert end["tendon_inputs"] == changed
    assert end["tendon_inputs"] != start
    session.tendon_inputs["abdomen_pitch"] = .2
    with pytest.raises(ValueError, match="abdomen_pitch"):
        recovery.validate_end_held(session, start, {"tendon_inputs": changed})


def test_science_error_precedes_both_later_sidecar_failures(tmp_path):
    start = _held()
    session = _session(start)
    recovery.install_held_inputs(session, start)
    first = ValueError("original scientific failure")
    errors = []
    def broken(path, value):
        raise OSError("generated sidecar error: " + path.name)
    got, end = recovery.save_arm_held_sidecars(
        tmp_path, recovery.held_receipt(session), session, first,
        errors, writer=broken)
    assert got is first
    assert end is not None
    assert [item["stage"] for item in errors] == [
        "start held sidecar", "end held sidecar"]
    assert len(first.__notes__) == 2


def test_late_publication_revokes_both_verdict_locations_preserving_numeric_raw(tmp_path):
    raw = tmp_path / "contrast.npz"
    np.savez(raw, q=np.array([1.]))
    recovery.e261.write_json(tmp_path / "contrast.json", {
        "H261_supported": True, "H262_supported": True,
        "classification": "supported_positive_return", "D": .001})
    result = {"validity": "valid_short_closed_loop_diagnostic",
              "H262": {"H261_supported": True, "H262_supported": True,
                       "classification": "supported_positive_return"}}
    first = [True]
    def writer(path, value):
        if path.name == "result.json" and first[0]:
            first[0] = False
            raise OSError("late result write")
        recovery.e261.write_json(path, value)
    error = recovery.finalize(tmp_path, {"segments": {}}, result, [], None,
                              recovery.time.perf_counter(), 1_000_000, writer=writer)
    assert isinstance(error, OSError)
    assert result["validity"] == "invalid_or_incomplete"
    assert result["H262"]["H262_supported"] is False
    assert json.loads((tmp_path / "contrast.json").read_text())["H262_supported"] is False
    assert raw.exists()
