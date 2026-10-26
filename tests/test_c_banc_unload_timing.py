import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "e161", ROOT / "tools/analyze_c_banc_m3_unload_timing.py"
)
e161 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(e161)


def test_world_z_not_tangent_or_world_y_and_closes():
    jac = np.zeros((2, 3, 20))
    vel = np.ones((2, 20))
    jac[:, 1, :] = 1000  # Must be ignored.
    jac[:, 2, :] = np.arange(20)
    legs = np.array([0, 1])
    root = np.arange(6)
    leg_dofs = np.array(
        [
            [6, 7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18, 19],
            [6, 7, 8, 9, 10, 11, 12],
        ]
        * 2
    )
    # Use distinct nonoverlapping addresses across root plus all legs in real schema.
    leg_dofs = np.arange(6, 48).reshape(6, 7)
    jac = np.pad(jac, ((0, 0), (0, 0), (0, 28)))
    vel = np.pad(vel, ((0, 0), (0, 28)), constant_values=1)
    out = e161.decompose_world_z(jac, vel, legs, root, leg_dofs)
    assert np.allclose(out["contact_z_total"], np.arange(20).sum())
    assert np.max(np.abs(out["contact_z_closure_error"])) == 0
    assert np.all(out["contact_z_root"] == np.arange(6).sum())


def test_contact_weighting_and_missing_are_nan_with_mask():
    out = e161.weighted_native_leg(
        np.array([0, 0, 1]),
        np.array([4, 4, 0]),
        np.array([1.0, 3.0, 2.0]),
        {"value": np.array([2.0, 6.0, 9.0])},
        3,
    )
    assert out["value"][0, 4] == pytest.approx(5.0)
    assert out["contact_load_sum"][0, 4] == 4
    assert not out["contact_available"][2, 4]
    assert np.isnan(out["value"][2, 4])


def test_transition_stability_and_window_boundary_reasons():
    load = np.zeros(800)
    load[0:250] = 0.03
    load[300:550] = 0.03
    load[700:] = 0.03
    raw, included, reasons = e161.find_unload_transitions(load)
    assert raw.tolist() == [250, 550]
    assert included.tolist() == [550]
    assert reasons == ["unstable_post_nonsupport", "included"]
    boundary = np.r_[np.full(150, 0.03), np.zeros(650)]
    raw, included, reasons = e161.find_unload_transitions(boundary)
    assert raw.tolist() == [150] and included.size == 0
    assert reasons == ["incomplete_alignment_window"]
    unstable = np.r_[np.full(210, 0.03), np.zeros(20), np.full(20, 0.03), np.zeros(550)]
    _, _, reasons = e161.find_unload_transitions(unstable)
    assert "unstable_post_nonsupport" in reasons


def test_command_projection_uses_control_delta_and_all_15_groups():
    activation = np.zeros((4, 96))
    activation[2, 65:80] = np.arange(1, 16)
    moment = np.zeros((42, 96))
    moment[28:35, 65:80] = 1
    jz = np.ones((1, 7))
    delta, joint_derivative, projected = e161.command_rate_projection(
        activation, np.arange(4), moment, np.array([2]), jz
    )
    assert delta.shape == projected.shape == (1, 15)
    assert joint_derivative.shape == (1, 7, 15)
    assert np.allclose(delta[0], np.arange(1, 16) / 0.005)
    assert np.allclose(projected[0], 0.02 * 7 * delta[0])
    moment[28:35, 78] = 0
    _, _, projected = e161.command_rate_projection(
        activation, np.arange(4), moment, np.array([2]), jz
    )
    assert projected[0, 13] == 0  # Diagnostic projection only; not physical influence.


@pytest.mark.parametrize("fault", ["clock", "index", "dof"])
def test_fail_closed_schema_alignment_faults(fault):
    if fault == "dof":
        with pytest.raises(ValueError, match="overlaps"):
            e161.decompose_world_z(
                np.zeros((1, 3, 48)),
                np.zeros((1, 48)),
                np.array([0]),
                np.arange(6),
                np.tile(np.arange(6, 13), (6, 1)),
            )
        return
    load = np.r_[np.full(300, 0.03), np.zeros(500)]
    if fault == "clock":
        with pytest.raises(ValueError, match="clock"):
            e161.command_rate_projection(
                np.zeros((3, 96)),
                np.array([0, 2, 3]),
                np.zeros((42, 96)),
                np.array([2]),
                np.zeros((1, 7)),
            )
    else:
        with pytest.raises(ValueError, match="out of range"):
            e161.weighted_native_leg(
                np.array([800]),
                np.array([4]),
                np.array([1.0]),
                {"x": np.array([1.0])},
                len(load),
            )


def test_verified_inputs_rejects_hashes_before_json_read(tmp_path):
    freeze = tmp_path / "freeze.json"
    review = tmp_path / "review.json"
    freeze.write_text("not json", encoding="utf-8")
    review.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="freeze SHA"):
        e161.verified_inputs(freeze, "bad", review, "bad")


def test_trace_ltm_is_separate_from_position_moment_projection():
    samples = [
        {
            "control_tick": tick,
            "time_s": tick * 0.005,
            "joint_targets_rad": np.zeros(42).tolist(),
            "tendon_inputs": {"rm_tarsus": tick / 10},
        }
        for tick in range(2001)
    ]
    out = e161.trace_arrays(
        {"samples": samples}, {"ltm": np.array([1, 2, 3, 4, 78, 5])}
    )
    assert out["rm_trace_joint_targets"].shape == (2001, 7)
    assert out["rm_ltm_tendon"][10] == 1


def generated_full_schema():
    counts = np.full(10000, 2, dtype=np.int64)
    counts[:4420] = 3
    offsets = np.r_[0, np.cumsum(counts)]
    native_index = np.repeat(np.arange(10000), counts)
    within = np.arange(24420) - np.repeat(offsets[:-1], counts)
    leg = (native_index + within) % 6
    contact_load = np.full(24420, 0.01)
    flat = native_index * 6 + leg
    native_load = np.bincount(flat, weights=contact_load, minlength=60000).reshape(
        10000, 6
    )
    slip_valid = native_load >= e161.SUPPORT_BW
    slip = np.full((10000, 6), np.nan)
    slip[slip_valid] = 0
    receipt_names = (
        "step_requested",
        "integration_complete",
        "state_captured",
        "state_copy_complete",
        "shadow_forward_complete",
        "observer_returned",
        "diagnostics_returned",
        "nonfoot_returned",
        "observation_complete",
        "source_verified",
    )
    native_qvel = np.zeros((10000, 108))
    native = {
        "native_index": np.arange(10000),
        "time_s": 1.0001 + np.arange(10000) * 0.0001,
        "control_tick": 201 + np.arange(10000) // 50,
        "qvel": native_qvel,
        "ctrl": np.zeros((10000, 82)),
        "normal_load_bw": native_load,
        "slip_mm_s": slip,
        "slip_valid": slip_valid,
        "force_forward_bw": -np.ones((10000, 6)),
        **{key: np.ones(10000, dtype=bool) for key in receipt_names},
    }
    contact = {
        "offsets": offsets,
        "native_index": native_index,
        "leg": leg,
        "normal_load_bw": contact_load,
        "qvel": native_qvel[native_index],
        "relative_jacobian": np.zeros((24420, 3, 108)),
        "ap_own7": np.zeros((24420, 7)),
    }
    layout = {
        "root_dof_addresses": list(range(6)),
        "leg_dof_addresses": [
            [42 + 11 * leg_index + joint for joint in range(7)]
            for leg_index in range(6)
        ],
        "actuator_ids": list(range(8, 50)),
    }
    recruitment = {
        "activation": np.zeros((2001, 96)),
        "control_tick": np.arange(2001),
        "complete": np.ones(2001, bool),
        "activation_available": np.ones(2001, bool),
    }
    adapter = {"moment": np.zeros((42, 96))}
    features = {
        "time_s": np.arange(2001) * 0.005,
        "speed_mm_s": np.zeros(2001),
    }
    return contact, native, layout, recruitment, adapter, features


def test_generated_full_schema_keeps_contact_and_native_closure_distinct():
    out = e161.analyze_arrays(*generated_full_schema())
    assert out["contact_z_closure_error"].shape == (24420,)
    assert out["native_z_closure_error"].shape == (10000, 6)
    assert np.isfinite(out["contact_z_closure_error"]).all()
    assert np.array_equal(np.isfinite(out["native_z_total"]), out["contact_available"])
    assert np.isnan(out["native_z_total"][~out["contact_available"]]).all()
    assert out["subset_supported_z_own7"].shape == (6, 7)
    assert out["subset_supported_rm_command_projection_z"].shape == (15,)
    assert out["event_mean_rm_command_projection_z"].shape == (401, 15)


def test_late_ledger_failure_revokes_success_and_preserves_primary(
    tmp_path, monkeypatch
):
    (tmp_path / "raw-derived.npz").write_bytes(b"raw")
    original = e161.atomic_json

    def fail_ledger(path, value):
        if path.name == "ledger.json":
            raise OSError("generated late ledger failure")
        original(path, value)

    monkeypatch.setattr(e161, "atomic_json", fail_ledger)
    result = {
        "accepted": True,
        "publication_complete": True,
        "diagnostic_valid": True,
        "primary_error": {"error_type": "ValueError", "error": "primary"},
    }
    out = e161.finalize_publication(tmp_path, result, {}, e161.time.perf_counter())
    assert not out["accepted"]
    assert not out["publication_complete"]
    assert not out["diagnostic_valid"]
    assert out["primary_error"]["error"] == "primary"
    persisted = __import__("json").loads((tmp_path / "result.json").read_text())
    assert not persisted["accepted"]
    assert any(item["stage"] == "ledger" for item in persisted["publication_errors"])


def test_successful_publication_manifests_every_final_file(tmp_path):
    (tmp_path / "raw-derived.npz").write_bytes(b"raw")
    result = {
        "accepted": True,
        "publication_complete": True,
        "diagnostic_valid": True,
    }
    out = e161.finalize_publication(
        tmp_path, result, {"schema": "generated"}, e161.time.perf_counter()
    )
    assert out["accepted"] and out["publication_complete"]
    manifest = __import__("json").loads((tmp_path / "output-manifest.json").read_text())
    assert set(manifest) == {"raw-derived.npz", "ledger.json", "result.json"}
    assert manifest["result.json"]["sha256"] == e161.sha256(tmp_path / "result.json")


def test_numeric_mismatch_is_rejected_after_raw_is_preserved(tmp_path):
    arrays = {
        "contact_z_closure_error": np.zeros(2),
        "native_z_total": np.array([[0.0, np.nan]]),
        "contact_available": np.array([[True, False]]),
        "contact_load_sum": np.array([[0.1, 0.0]]),
        "native_load": np.array([[0.2, 0.0]]),
        "native_rm_target_ctrl": np.zeros((1, 7)),
    }
    e161.atomic_npz(tmp_path / "raw-derived.npz", **arrays)
    with pytest.raises(ValueError, match="contact load"):
        e161.validate_numerical_contract(arrays, np.zeros((1, 7)))
    assert (tmp_path / "raw-derived.npz").is_file()


def test_rm_summary_masks_missing_projection_and_retains_unloaded_requests():
    contact, native, layout, recruitment, adapter, features = generated_full_schema()
    contact["normal_load_bw"] *= 3
    native["normal_load_bw"] *= 3
    native["slip_valid"] = native["normal_load_bw"] >= e161.SUPPORT_BW
    native["slip_mm_s"][native["slip_valid"]] = 0
    contact["relative_jacobian"][:, 2, 91] = -2.5
    recruitment["activation"][:, 74] = np.arange(2001) * 0.005
    adapter["moment"][33, 74] = -1
    out = e161.analyze_arrays(contact, native, layout, recruitment, adapter, features)
    assert out["subset_supported_rm_command_projection_z"][9] == pytest.approx(0.05)
    assert np.isfinite(out["subset_supported_rm_command_projection_z"]).all()
    assert np.allclose(out["native_rm_activation_derivative"][:, 9], 1)
    assert np.allclose(out["native_rm_joint_command_derivative"][:, 5, 9], -0.02)
    unavailable = ~out["contact_available"][:, 4]
    assert unavailable.any()
    assert np.isnan(out["native_rm_command_projection_z"][unavailable]).all()
    assert np.isfinite(out["native_rm_joint_command_derivative"][unavailable]).all()


@pytest.mark.parametrize(
    "target", ["ledger.json", "result.json", "output-manifest.json"]
)
@pytest.mark.parametrize("persistent", [False, True])
def test_second_publication_failure_revokes_persisted_success(
    tmp_path, monkeypatch, target, persistent
):
    import json

    original = e161.atomic_json
    calls = 0

    def fail_second(path, value):
        nonlocal calls
        if path.name == target:
            calls += 1
            if calls == 2 or (persistent and calls > 2):
                raise OSError("Generated late publication failure")
        original(path, value)

    monkeypatch.setattr(e161, "atomic_json", fail_second)
    result = {"accepted": True, "diagnostic_valid": True, "publication_complete": True}
    out = e161.finalize_publication(tmp_path, result, {}, e161.time.perf_counter())
    assert not out["accepted"] and not out["publication_complete"]
    result_path = tmp_path / "result.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        assert not saved["accepted"] and not saved["publication_complete"]
    else:
        assert persistent and target == "result.json"
        assert len(list(tmp_path.glob("invalid-result-*.json"))) == 1
    if not persistent:
        manifest = json.loads((tmp_path / "output-manifest.json").read_text())
        assert all(
            e161.sha256(tmp_path / name) == row["sha256"]
            for name, row in manifest.items()
        )
