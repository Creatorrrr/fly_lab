"""Generated fixtures only. Actual neural/body/gait evidence is not exercised here."""
from types import SimpleNamespace

import numpy as np
import pytest

from tools import analyze_c_banc_m3_startup_tracking as analysis
from tools import observe_c_banc_m3_startup_tracking as capture


def metadata(*, verified=True, enabled=True):
    return {"control_clamp": {"status": "VERIFIED_COMPILED_FLAG" if verified else "UNVERIFIED", "enabled": enabled},
            "actuator_ctrllimited": [True] + [False] * 41,
            "actuator_ctrlrange_rad": [[-1., 1.]] * 42}


def fixture_session(*, c=100, neural_error=False, physics_error=False):
    marker = object()

    class Network:
        def __init__(self):
            self.tick, self.calls = 20 * c, []
            self.rate = np.arange(512, dtype=np.float32)
            self.adaptation = np.zeros(512, np.float32)
            self.resource = np.ones(2, np.float64)
            self._resource_sources = np.asarray([5, 10], np.int64)
            self._resource_u = np.zeros(2, np.float64)
            self._resource_tau = np.ones(2, np.float64)
            self.output_mask = np.ones(512, np.float32)

        def advance_readout(self, drive, ids, steps):
            self.calls.append((drive, ids, steps))
            drive.fill(77)  # the capture must retain the original finalized drive
            self.rate.fill(99)
            if neural_error:
                raise RuntimeError("original neural failure")
            self.tick += steps
            return marker

    class Body:
        def __init__(self):
            self.now, self.calls = .005 * c, 0
            self.act_ids = self.qpos_ids = np.arange(42)
            self.d = SimpleNamespace(ctrl=np.ones(42, np.float64) * 2, qpos=np.zeros(42, np.float64))

        def physics_time(self):
            return self.now

        def _step_physics(self, *args, **kwargs):
            self.calls += 1
            self.now += .0001
            self.d.qpos += .1
            if physics_error:
                raise RuntimeError("original post-step health stop")
            return marker

    adapter = SimpleNamespace(motor_ids=np.arange(391, dtype=np.int64), bank=np.ones((391, 3), np.float64),
                              original=SimpleNamespace(activation=np.ones(96, np.float64)))
    adapter.encode = lambda: (_ for _ in ()).throw(AssertionError("encode must not be called"))
    s = SimpleNamespace(control_tick=c, network=Network(), body=Body(), adapter=adapter)
    return s, marker


def tracker(s):
    return capture.TrackingObserver(s, np, metadata(), lambda label: True, n=512, resource_n=2, synthetic=True)


def test_finalized_drive_pre_state_one_call_identity_and_no_alias():
    s, marker = fixture_session()
    t = tracker(s).install()
    drive = np.arange(512, dtype=np.float32)
    before = drive.copy()
    try:
        answer = s.network.advance_readout(drive, s.adapter.motor_ids, 20)
        assert answer is marker and len(s.network.calls) == 1
        assert s.network.calls[0][0] is drive and s.network.calls[0][1] is s.adapter.motor_ids
        assert np.array_equal(t.context["drive"][0], before)
        assert np.array_equal(t.context["rate"][0], before)
        assert t.context["pre_control"][0] == 100 and t.context["drive_control_end"][0] == 101
        assert t.context["pre_network"][0] == 2000 and t.context["drive_network_pre"][0] == 2000
        assert t.context["u_zero"][0] and t.context["resource_one"][0]
        assert np.all(drive == 77) and t.context["rate"][0, 0] == 0 and np.all(s.network.rate == 99)
    finally:
        t.close()
    assert "advance_readout" not in vars(s.network) and "_step_physics" not in vars(s.body)


def test_neural_clock_off_by_one_refuses_before_original_call():
    s, _ = fixture_session()
    s.network.tick += 20
    t = tracker(s).install()
    try:
        with pytest.raises(ValueError, match="c generates c\\+1"):
            s.network.advance_readout(np.zeros(512, np.float32), s.adapter.motor_ids, 20)
        assert not s.network.calls and not t.context["pre_valid"].any()
    finally:
        t.close()


def test_native_pre_ctrl_pre_q_and_post_q_have_distinct_clocks():
    s, marker = fixture_session(c=0)
    t = tracker(s).install()
    try:
        token = object()
        assert s.body._step_physics(token, exact=token) is marker
        assert s.body.calls == 1 and t.started == t.returned == 1
        a = t.payload()
        assert a["native_control_end"].tolist() == [1] and a["native_substep"].tolist() == [0]
        assert a["native_pre_time_s"].tolist() == [0.] and a["native_post_time_s"].tolist() == [.0001]
        assert np.all(a["native_pre_q_rad"] == 0) and np.all(a["native_post_q_rad"] == .1)
        assert a["native_raw_ctrl_rad"][0, 0] == 2 and a["native_effective_ctrl_rad"][0, 0] == 1
        assert a["native_effective_ctrl_rad"][0, 1] == 2  # unlimited, even with nominal range
    finally:
        t.close()


@pytest.mark.parametrize("which", ["neural", "physics"])
def test_exception_preserves_original_error_and_restores_inherited_method(which):
    s, _ = fixture_session(c=100 if which == "neural" else 0, neural_error=which == "neural", physics_error=which == "physics")
    t = tracker(s).install()
    try:
        with pytest.raises(RuntimeError, match="original"):
            if which == "neural":
                s.network.advance_readout(np.zeros(512, np.float32), s.adapter.motor_ids, 20)
            else:
                s.body._step_physics()
        if which == "physics":
            assert t.started == 1 and t.returned == 0
            assert t.a["native_post_observed"][0] and not t.a["native_returned"][0]
            assert s.body.calls == 1
        else:
            assert len(s.network.calls) == 1 and t.context["drive_valid"][0]
    finally:
        t.close()
    assert "advance_readout" not in vars(s.network) and "_step_physics" not in vars(s.body)


def test_nested_live_shadow_restores_true_pre_live_instance_ownership():
    s, _ = fixture_session(c=0)
    inherited = capture.MethodBinding(s.body, "_step_physics")
    own_token = lambda: "owned"
    s.network.advance_readout = own_token
    owned = capture.MethodBinding(s.network, "advance_readout")
    pre_live = s.body._step_physics
    s.body._step_physics = lambda: pre_live()
    t = tracker(s).install()
    t.close()
    assert "_step_physics" in vars(s.body)  # pre-tracking Live shadow remains until Live cleanup
    s.body._step_physics = pre_live  # legacy LiveObserver.close behavior
    inherited.restore()
    owned.restore()
    assert "_step_physics" not in vars(s.body)
    assert vars(s.network)["advance_readout"] is own_token


def test_effective_is_null_when_flags_unverified_and_constraint_range_is_not_clamp():
    raw = np.ones((2, 42), np.float64) * 2
    assert capture.derived_effective(raw, metadata(verified=False), np) is None
    assert np.array_equal(capture.derived_effective(raw, metadata(enabled=False), np), raw)
    effective = capture.derived_effective(raw, metadata(), np)
    assert np.all(effective[:, 0] == 1) and np.all(effective[:, 1:] == 2)
    assert np.all(raw == 2)


def test_lags_use_identical_pairs_denominators_and_no_outside_official_window():
    ticks = np.arange(301, dtype=np.int64)
    request = np.broadcast_to(ticks[:, None], (301, 42)).astype(np.float64).copy()
    actual = request - 1
    result = analysis.common_lags(request, actual, ticks, np)
    assert result["common_base_ticks"] == list(range(100, 296))
    assert [v["pair_count"] for v in result["lags"]] == [196] * 5
    assert [v["scalar_denominator"] for v in result["lags"]] == [196 * 42] * 5
    assert result["lags"][1]["error_rad"]["rms"] == 0
    assert result["includes_outside_window"] is False and result["lag_selection_is_causal_or_acceptance"] is False
    partial = analysis.common_lags(request[:103], actual[:103], ticks[:103], np)
    assert all(v["pair_count"] == 0 and v["error_rad"]["rms"] is None for v in partial["lags"])


def test_two_end_contact_transition_masks_partition_without_relabeling():
    pre = np.asarray([True, False, True, False])
    post = np.asarray([True, False, False, True])
    states = analysis.endpoint_states(pre, post, np)
    assert [int(states[v].sum()) for v in ("loaded_both", "unloaded_both", "transition")] == [1, 1, 2]
    assert np.all(sum(states.values()) == 1)


def test_missing_capture_bitmap_and_failed_outer_cannot_be_complete():
    s, _ = fixture_session(c=0)
    t = tracker(s)
    assert t.summary()["capture_complete"] is False
    assert t.summary()["pre_complete_bitmap"] == [False] * 4
    t.context["pre_valid"][:] = True
    t.context["drive_valid"][:] = True
    t.a["boundary_valid"][:] = True
    # Returned native proof and actual work are still absent.
    assert t.summary()["capture_complete"] is False
    assert analysis.compare_values(np.zeros((1, 42)), np.zeros((301, 42)), np)["comparable"] is False


def test_trace_comparison_detects_contact_and_basis_drift_without_zip_tolerance():
    a = {"samples": [{"control_tick": 0, "body": {"basis": [[1., 0.], [0., 1.]]}, "feet": {"load_bearing": [True]}}]}
    b = {"samples": [{"control_tick": 0, "body": {"basis": [[1., .1], [0., 1.]]}, "feet": {"load_bearing": [False]}}]}
    result = analysis.compare_trace(a, b, np)
    assert result["exact"] is False and result["numeric_leaf_comparison"]["max_abs"] == 1
    assert any("basis" in path for path in result["first_mismatch_paths"])
    assert any("load_bearing" in path for path in result["first_mismatch_paths"])


def physical_names_and_map():
    indices = np.arange(42, dtype=np.int32)[::-1].reshape(6, 7)
    names = [None] * 42
    for leg, row in zip(capture.LEGS, indices, strict=True):
        suffixes = (f"/c_thorax-{leg}_coxa-yaw", f"/c_thorax-{leg}_coxa-pitch", f"/c_thorax-{leg}_coxa-roll",
                    f"/{leg}_coxa-{leg}_trochanterfemur-pitch", f"/{leg}_coxa-{leg}_trochanterfemur-roll",
                    f"/{leg}_trochanterfemur-{leg}_tibia-pitch", f"/{leg}_tibia-{leg}_tarsus1-pitch")
        for i, suffix in zip(row, suffixes, strict=True):
            names[int(i)] = "fly/namespace" + suffix
    return names, indices


def test_actual_namespace_hip_names_use_explicit_adapter_indices():
    names, indices = physical_names_and_map()
    groups = capture.leg_joint_indices(indices, names, np)
    assert list(groups) == ["lf", "lm", "lh", "rf", "rm", "rh"]
    assert groups["lf"] == indices[0].tolist() and not any(name.startswith("LF_") for name in names)
    wrong = indices.copy()
    wrong[[0, 3]] = wrong[[3, 0]]
    with pytest.raises(ValueError, match="binding"):
        capture.leg_joint_indices(wrong, names, np)
    with pytest.raises(ValueError, match="unique adapter"):
        capture.leg_joint_indices(np.zeros((6, 7), np.int64), names, np)


def test_invalid_slip_nan_is_preserved_and_valid_nonfinite_is_rejected(tmp_path):
    import hashlib
    import json
    slip = np.asarray([[np.nan, 1., 2., 3., 4., 5.]], np.float64)
    valid = np.asarray([[False, True, True, True, True, True]])
    raw = {"slip_mm_s": slip, "slip_valid": valid}
    report = analysis.compare_native_slip(raw, raw, np)
    assert report["exact"] and report["nan_position_exact"] and report["a_validity"]["invalid_nan_count"] == 1
    assert report["both_valid_values"]["count"] == 5 and report["finite_pair_values"]["rms"] == 0
    json.dumps(report, allow_nan=False)
    archive = tmp_path / "native.npz"
    np.savez(archive, **raw)
    descriptor = {"path": "native.npz", "bytes": archive.stat().st_size, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
    admission = SimpleNamespace(root=tmp_path, check=lambda label: True, fence=lambda label: True,
                                limits={"input_file_cap_bytes": 96 << 20, "selected_expanded_cap_bytes": 208 << 20})
    saved = analysis.SavedInputs(admission, np).arrays(descriptor, {k: analysis.NATIVE_FIELDS[k] for k in raw})
    assert np.isnan(saved["slip_mm_s"][0, 0]) and not saved["slip_mm_s"].flags.writeable
    changed = {"slip_mm_s": slip.copy(), "slip_valid": valid.copy()}
    changed["slip_valid"][0, 0] = True
    with pytest.raises(ValueError, match="valid native slip"):
        analysis.validate_native_slip(changed, np)
    changed["slip_valid"][0, 0] = False
    changed["slip_mm_s"][0, 1] = np.nan
    with pytest.raises(ValueError, match="valid native slip"):
        analysis.compare_native_slip(raw, changed, np)


def test_physical_boundary_pairs_native_post_window_and_equal_support_threshold():
    pairs = analysis.control_pairs(np.arange(301), np)
    assert pairs["physical200_end100..299"].tolist() == list(range(100, 300))
    assert pairs["both_in_window199_end101..299"].tolist() == list(range(101, 300))
    assert 300 not in pairs["physical200_end100..299"]
    native = {"time_s": np.asarray([.4999, .5, 1.4999, 1.5]), "normal_load_bw": np.tile([.019999, .02, .020001, 0., .01, .03], (4, 1))}
    window, support = analysis.native_window_and_support(native, np)
    assert window.tolist() == [False, True, True, False]
    assert support[1].tolist() == [False, True, True, False, False, True]


def test_tracking_publisher_reserves_streams_and_records_actual_bytes(tmp_path):
    admission = SimpleNamespace(role="tracking_diagnosis", root=tmp_path, output=tmp_path / "result", fence=lambda *args: True,
                                limits={"output_cap_bytes": 4 << 20, "host_bytes": 2 << 30})
    publisher = capture.bounded_publisher(admission)
    publisher.write("bounded.bin", b"x" * (1 << 20))
    assert publisher.usage()["durable_result_bytes_so_far"] == 1 << 20
    assert publisher.usage()["reserved_supervision_bytes"] == 3 << 20
    with pytest.raises(ValueError, match="three MiB"):
        publisher.write("overflow.bin", b"x")
    assert not (admission.output / "overflow.bin").exists()


def test_omitted_interpolate_uses_actual_default_true_but_tau_zero_holds():
    default = capture.interpolation_contract(False, False, 0.)
    assert default["interpolate"] is True and default["interpolate_argument"].startswith("omitted")
    assert default["interpolation"] == "held raw ctrl"
    continuous = capture.interpolation_contract(True, False, .03)
    cut = capture.interpolation_contract(True, True, .03)
    assert continuous["interpolate"] is True and "exponential" in continuous["interpolation"]
    assert cut["interpolate"] is False and cut["interpolation"] == "held raw ctrl"
