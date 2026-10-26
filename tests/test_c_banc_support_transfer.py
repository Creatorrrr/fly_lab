"""Synthetic E122 contract tests; no frozen trajectory is decoded here."""

from pathlib import Path

import numpy as np
import pytest

from tools import analyze_c_banc_m3_support_transfer as e122


def synthetic_calibration():
    ids = np.arange(391) + 2000
    observed = np.r_[ids[10:], ids[:10], np.arange(55) + 100]
    membership = np.arange(391) % 96
    pool = np.zeros((96, 391))
    for muscle in range(96):
        columns = np.flatnonzero(membership == muscle)
        pool[muscle, columns] = 1 / len(columns)
    muscles = []
    for i in range(96):
        muscles.append({"leg": e122.LEGS[i // 16],
                        "muscle": "long_tendon_muscle" if i % 16 == 15 else f"muscle_{i}",
                        "columns": np.flatnonzero(pool[i]).tolist()})
    moment = np.zeros((42, 96)); moment[0, 0] = 1; moment[0, 1] = -2
    rest = np.linspace(-.1, .1, 42)
    ltm = np.arange(6) * 16 + 15
    cal = dict(motor_ids=ids, pool=pool, membership=membership,
               moment=moment, rest=rest, ltm=ltm)
    meta = dict(motor_ids=ids.copy(), observed_ids=observed.copy())
    signal = dict(observed_ids=observed.copy())
    anatomy = dict(graph_hash="synthetic-graph",
                   motor_roster=[f"flywire:banc:888:{720575941000000000 + i}" for i in ids],
                   muscles=muscles)
    start = dict(graph_hash="synthetic-graph", adapter={"groups": muscles,
        "activation": np.zeros(96), "original": {"original": {"filtered": np.zeros(391)}},
        "calibration": {key: cal[key].copy() for key in ("motor_ids", "moment", "rest", "pool", "ltm")}})
    return cal, meta, signal, anatomy, start


def test_motor_identity_order_and_pool_membership():
    cal, meta, signal, anatomy, start = synthetic_calibration()
    value = e122.decoder_calibration(cal, meta, signal, anatomy, start)
    assert np.array_equal(value[0], cal["motor_ids"])
    meta["motor_ids"] = meta["motor_ids"][::-1]
    with pytest.raises(ValueError, match="motor IDs"):
        e122.decoder_calibration(cal, meta, signal, anatomy, start)
    meta["motor_ids"] = cal["motor_ids"].copy()
    cal["membership"][0] = 1
    with pytest.raises(ValueError, match="declared muscle"):
        e122.decoder_calibration(cal, meta, signal, anatomy, start)


def test_observed_rate_id_order_and_stored_decoder_comparison():
    cal, meta, signal, anatomy, start = synthetic_calibration()
    ids, observed, pool, moment, rest, ltm, _, _ = e122.decoder_calibration(cal, meta, signal, anatomy, start)
    recorded = np.zeros((2000, 446)); recorded[:, 0] = 4; recorded[:, 10] = 7
    columns = [int(np.flatnonzero(observed == motor_id)[0]) for motor_id in ids]
    filtered, active, request, tendon = e122.decode_recorded(recorded[:, columns], pool, moment, rest, ltm)
    assert filtered.shape == (2000, 391)
    rows = [{}] + [{"joint_targets_rad": request[i],
                    "tendon_inputs": {f"{leg}_tarsus": tendon[i, j] for j, leg in enumerate(e122.LEGS)}}
                   for i in range(2000)]
    signal["muscle_activation"] = active.copy()
    assert max(e122.compare_decoder(filtered, active, request, tendon, signal, rows, moment, rest).values()) < 1e-12
    # Swapping stored motor columns changes the resulting decoder output.
    wrong_columns = columns.copy(); wrong_columns[0], wrong_columns[10] = wrong_columns[10], wrong_columns[0]
    _, wrong_active, wrong_request, wrong_tendon = e122.decode_recorded(recorded[:, wrong_columns], pool, moment, rest, ltm)
    with pytest.raises(ValueError, match="Recorded decoder mismatch"):
        e122.compare_decoder(filtered, wrong_active, wrong_request, wrong_tendon, signal, rows, moment, rest)


def test_native_endpoint_grouping_validity_and_fixed_window():
    times = np.arange(1, 100001) * .0001
    loads = np.zeros((100000, 6)); slip = np.full((100000, 6), np.nan)
    # Endpoint state 100 is .5s; its native samples are (.495,.5].
    loads[4950:5000, 0] = .1; slip[4950:5000, 0] = 2.0
    native = dict(time_s=times, normal_load_bw=loads, slip_mm_s=slip,
                  upright=np.ones(100000), nonfoot_contact=np.zeros(100000, bool))
    _, l, s, valid, supported, provenance = e122.native_intervals(native)
    assert provenance == "derived_isfinite_E85"
    assert valid[4950:5000, 0].all()
    assert not valid[5000, 0]
    grouped = e122.native_group_stats(times, l, s, supported, [100])
    assert grouped["native_samples"] == 50
    assert grouped["first_native_s"] == pytest.approx(.4951)
    assert grouped["last_native_s"] == pytest.approx(.5)
    assert grouped["supported_slip_load_weighted_mm_s"] == pytest.approx(2)
    assert np.count_nonzero((times >= .5) & (times < 10)) == 95000
    native["slip_valid"] = valid.copy(); native["slip_valid"][4950, 0] = False
    with pytest.raises(ValueError, match="slip_valid"):
        e122.native_intervals(native)
    native.pop("slip_valid"); slip[4950, 0] = np.nan
    with pytest.raises(ValueError, match="Supported native slip"):
        e122.native_intervals(native)
    slip[4950, 0] = 2; native["time_s"] = times.copy(); native["time_s"][5] += 3e-8
    with pytest.raises(ValueError, match="clock"):
        e122.native_intervals(native)


def test_factorized_signed_contributions_preserve_rest_and_cancellation():
    cal, _, _, anatomy, _ = synthetic_calibration()
    filtered = np.zeros((4, 391)); filtered[:, 0] = [1, 2, 3, 4]; filtered[:, 1] = [4, 3, 2, 1]
    active = filtered @ cal["pool"].T
    result = e122.contribution_stats(filtered, active, cal["pool"], cal["moment"],
                                     cal["rest"], np.ones(4, bool), anatomy["muscles"], anatomy["motor_roster"])
    assert np.allclose(np.asarray(result["mn_signed_mean_rad_42x391"]).sum(axis=1),
                       np.asarray(result["muscle_signed_mean_rad_42x96"]).sum(axis=1))
    assert result["positive_mean_sum_rad"][0] > 0
    assert result["negative_mean_sum_rad"][0] < 0
    assert result["request_mean_rad"][0] == pytest.approx(result["rest_rad"][0] + result["linear_net_mean_rad"][0])
    assert result["largest_mn_mean"][0]["source"].startswith("flywire:banc:888:")


def test_tracking_ratio_is_scale_invariant_for_finite_nonzero_motion():
    request = np.zeros((4, 42))
    request[:, 0] = [-1., 1., -1., 1.]
    mask = np.ones(4, bool)
    for scale in (1., 1e-13):
        result = e122.tracking_stats(scale * request, 2 * scale * request, mask)
        ratio = result["per_joint"]["measured_to_requested_dynamic_ratio"]
        assert ratio[0] == pytest.approx(2.)
        assert ratio[1] is None


def test_start_held_action_and_actual_neural_zero_without_rest_equality():
    qpos = np.arange(3, dtype=float); qvel = np.arange(2, dtype=float)
    requested = np.full(42, .3); measured = np.full(42, .2)
    row = {"qpos": qpos, "qvel": qvel, "joint_targets_rad": requested,
           "joint_angles_rad": measured}
    start = {"control_tick": 0, "network": {"tick": 0, "rate": np.zeros(391)},
             "body": {"modelHash": e122.BODY_SHA,
                      "state": np.r_[.2, qpos, qvel],
                      "action": {"joint_angles": requested.copy()}}}
    e122.verify_start_anchor(start, row)  # The synthetic decoder rest can differ by .3.
    anchor = {"qpos": qpos, "qvel": qvel, "joint_angles": measured}
    e122.verify_shared_anchor(anchor, {k: v.copy() for k, v in anchor.items()}, "E113")
    changed = {k: v.copy() for k, v in anchor.items()}; changed["joint_angles"][0] += .01
    with pytest.raises(ValueError, match="initial physical state"):
        e122.verify_shared_anchor(changed, anchor, "E113")
    start["network"]["rate"][0] = 1
    with pytest.raises(ValueError, match="neural rate"):
        e122.verify_start_anchor(start, row)


def test_decoder_registry_survives_raw_save_failure(tmp_path, monkeypatch):
    registry = {"decoder_attempts": 1, "decoder_completed": 1, "decoder_steps": 2000,
                "raw_arrays": {}, "raw_saved": set(), "comparison_errors": {}}
    raw = {"filtered_rate": np.arange(3)}
    registry["raw_arrays"]["E113"] = raw
    registry["comparison_errors"]["E113"] = {"activation": .01}
    original_savez = e122.np.savez
    def fail_raw(path, **arrays):
        if str(path).endswith("E113-decoder-raw.npz"):
            raise OSError("raw save failure")
        return original_savez(path, **arrays)
    monkeypatch.setattr(e122.np, "savez", fail_raw)
    with pytest.raises(OSError, match="raw save failure"):
        e122.save_decoder_artifacts(tmp_path, "E113", raw, registry["comparison_errors"]["E113"], registry)
    assert registry["decoder_steps"] == 2000 and registry["decoder_completed"] == 1
    assert np.array_equal(registry["raw_arrays"]["E113"]["filtered_rate"], [0, 1, 2])
    assert (tmp_path / "E113-decoder-pass.json").exists()
    assert (tmp_path / "E113-decoder-comparison-raw.json").exists()


def test_plot_primary_error_survives_secondary_close(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    original_subplots = plt.subplots
    def fake_subplots(*args, **kwargs):
        fig, axes = original_subplots(*args, **kwargs)
        def bad_save(*a, **k):
            raise ValueError("primary savefig")
        fig.savefig = bad_save
        return fig, axes
    def bad_close(*args, **kwargs):
        raise OSError("secondary close")
    monkeypatch.setattr(plt, "subplots", fake_subplots)
    monkeypatch.setattr(plt, "close", bad_close)
    curve = {"t": np.array([.5, 10.]), "speed": np.array([0., 0.]),
             "tracking_rms_rad": np.array([0., 0.]),
             "supported_forward_foot_fraction": np.array([0., 0.])}
    with pytest.raises(ValueError, match="primary savefig") as caught:
        e122.plot_comparison(tmp_path, {"synthetic": curve})
    assert "Secondary E122 plot close" in " ".join(caught.value.__notes__)


def test_output_existence_deadline_and_primary_secondary_preservation(tmp_path, monkeypatch):
    out = tmp_path / "existing"; out.mkdir()
    with pytest.raises(FileExistsError):
        e122.run(out, tmp_path / "unused")
    with pytest.raises(TimeoutError):
        e122.deadline_check(0, 25, 3, "new dataset")
    original_save = e122.save_json
    def broken_freeze(_):
        raise ValueError("primary freeze failure")
    def fail_one_save(path, value):
        if Path(path).name == "summary.json":
            raise OSError("secondary save failure")
        return original_save(path, value)
    monkeypatch.setattr(e122, "frozen_inputs", broken_freeze)
    monkeypatch.setattr(e122, "save_json", fail_one_save)
    out = tmp_path / "new"
    with pytest.raises(ValueError, match="primary freeze failure") as caught:
        e122.run(out, tmp_path / "freeze")
    assert "Secondary E122 failure" in " ".join(caught.value.__notes__)
    assert (out / "execution-raw.json").exists()
    assert (out / "secondary-errors.json").exists()
