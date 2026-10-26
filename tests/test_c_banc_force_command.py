"""Synthetic adversarial tests for the E186 saved-data analyzer."""

import json

import numpy as np
import pytest

from tools import analyze_c_banc_m3_force_command as e186


def test_native_sign_is_split_before_binning_and_anchor_is_missing():
    force = np.zeros((100, 6))
    load = np.full((100, 6), 0.03)
    force[:25, 0] = -2
    force[25:50, 0] = 2
    force[50:, 0] = -1
    load[50:, 0] = 0.01
    slip = np.ones_like(force)
    valid = np.ones_like(force, dtype=bool)
    time = np.arange(1, 101) * 0.0001
    bins = e186.native_sign_bins(force, load, slip, valid, time, time.copy())
    assert bins["count"].shape == (3, 6, 3)
    np.testing.assert_array_equal(bins["count"][0], 0)
    np.testing.assert_array_equal(bins["count"][1, 0], [25, 0, 25])
    np.testing.assert_allclose(bins["signed_force_sum_bw"][1, 0], [-50, 0, 50])
    np.testing.assert_array_equal(bins["count"][2, 0], 0)
    assert bins["negative_magnitude_sum_bw"][1, 0] == 50
    with pytest.raises(ValueError, match="supported slip"):
        broken = valid.copy()
        broken[0, 0] = False
        e186.native_sign_bins(force, load, slip, broken, time, time)


def test_conditional_mean_retains_missing_denominator_and_valid_counts():
    values = np.array([[1.0, np.nan], [3.0, 9.0], [100.0, 100.0]])
    count = np.zeros((3, 6))
    count[:2, 0] = [2, 1]
    magnitude = np.zeros_like(count)
    magnitude[:2, 0] = [1, 3]
    mean, valid, magmean, magvalid = e186.conditional_mean(
        values, count, magnitude, np.array([0, 0])
    )
    np.testing.assert_allclose(mean, [5 / 3, 9])
    np.testing.assert_array_equal(valid, [3, 1])
    np.testing.assert_allclose(magmean, [2.5, 9])
    np.testing.assert_array_equal(magvalid, [4, 3])
    empty = e186.conditional_mean(
        values, np.zeros_like(count), magnitude * 0, np.array([0, 0])
    )
    assert np.isnan(empty[0]).all() and np.array_equal(empty[1], [0, 0])


def test_interval_motion_preserves_row30_and_missing_anchor():
    target = np.zeros((4, 42))
    actual = np.zeros_like(target)
    target[1:, 30] = [2, 3, 4]
    actual[1:, 30] = [1, 2, 3]
    q = np.zeros((4, 96, 42))
    q[1:, 74, 30] = [0, 1, 2]
    row_leg = np.repeat(np.arange(6), 7)
    motion = e186.joint_motion(target, actual, q, row_leg)
    assert np.isnan(motion["joint_target_rate_rad_s"][:2]).all()
    assert np.isnan(motion["joint_contribution_rate_rad_s"][:2]).all()
    assert motion["joint_measured_velocity_rad_s"][1, 30] == 200
    assert motion["joint_start_discrepancy_rad"][1, 30] == 2
    assert motion["joint_tracking_error_rad"][2, 30] == 1
    assert motion["joint_alignment_cosine"][2, 4] == 1
    assert np.isnan(motion["joint_alignment_cosine"][1, 4])
    assert np.isnan(motion["joint_alignment_cosine"][2, 0])


def test_reconstruction_full_width_fixed_tau_and_negative_hinge():
    membership = np.repeat(np.arange(77), 4)
    membership = np.r_[membership, np.arange(64), np.arange(77, 96)].astype(np.int64)
    assert len(membership) == 391
    pool = np.zeros((96, 391))
    for m in range(96):
        pool[m, membership == m] = 1 / np.count_nonzero(membership == m)
    fixed = np.arange(77, 96, dtype=np.int64)
    slopes = np.zeros((96, 391, 3, 2))
    slopes[membership, np.arange(391), 1, 0] = pool[membership, np.arange(391)]
    slopes[membership, np.arange(391), 1, 1] = pool[membership, np.arange(391)]
    slopes[0, membership == 0, 0, 0] = 0.2
    slopes[0, membership == 0, 0, 1] = 0.1
    rate = np.zeros((4, 391))
    rate[1:] = 2
    bank = np.zeros((4, 391, 3))
    tau = np.array([0.005, 0.01, 0.04])
    alpha = -np.expm1(-0.005 / tau)
    original = np.zeros((4, 96))
    for i in range(1, 4):
        bank[i] = bank[i - 1] + alpha * (rate[i, :, None] - bank[i - 1])
        original[i] = original[i - 1] + alpha[1] * (pool @ rate[i] - original[i - 1])
    kappa = np.ones(391) * 0.1
    by_mn = (
        slopes[membership, np.arange(391), :, 0][None] * bank
        + (
            slopes[membership, np.arange(391), :, 1]
            - slopes[membership, np.arange(391), :, 0]
        )[None]
        * np.maximum(bank - kappa[None, :, None], 0)
    ).sum(axis=2)
    activation = np.column_stack(
        [by_mn[:, membership == m].sum(axis=1) for m in range(96)]
    )
    cal = {
        "tau": tau,
        "membership": membership,
        "fixed_muscles": fixed,
        "pool": pool,
        "slopes": slopes,
        "kappa": kappa,
        "ltm": fixed[:6],
        "initial_bank": bank[0],
        "initial_activation": activation[0],
        "initial_original_activation": original[0],
    }
    actual, residuals = e186.reconstruct(rate, bank, activation, original, cal)
    np.testing.assert_allclose(actual, by_mn, atol=1e-12)
    assert all(value < 1e-12 for value in residuals.values())
    assert (
        slopes[0, membership == 0, 0, 1] - slopes[0, membership == 0, 0, 0] < 0
    ).all()
    damaged = bank.copy()
    damaged[2, 0, 1] += 0.01
    with pytest.raises(ValueError, match="bank residual"):
        e186.reconstruct(rate, damaged, activation, original, cal)


def test_event_windows_keep_all_events_and_full_recover_band():
    events = [{"index": 131, "selected": True}, {"index": 139, "selected": False}]
    windows = e186._windows(events)
    np.testing.assert_array_equal(windows["startup99"], np.arange(1, 100))
    np.testing.assert_array_equal(windows["evaluation"], np.arange(100, 600))
    assert len(windows["event131:pre"]) == 10
    assert len(windows["event131:recover"]) == 11
    assert "event139:recover" in windows


def test_gate_rejects_tampered_freeze_before_original_read(tmp_path, monkeypatch):
    monkeypatch.setattr(e186, "ROOT", tmp_path)
    design = tmp_path / e186.DESIGN
    design.parent.mkdir(parents=True)
    design.write_text("design", encoding="utf-8")
    metadata = tmp_path / e186.METADATA
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({"schema": "invalid"}), encoding="utf-8")
    supervisor = (
        tmp_path / "verification/banc-natural-gait-m3-20260919/parent-e186-supervise.py"
    )
    supervisor.write_text("supervisor", encoding="utf-8")
    out = tmp_path / e186.OUTPUT
    freeze = tmp_path / "freeze.json"
    review = tmp_path / "review.json"
    freeze.write_text(
        json.dumps(
            {
                "schema": "flylab.m3.e186.force-command.freeze.v1",
                "design_path": e186.DESIGN,
                "design_sha256": e186.digest(design),
                "metadata_path": e186.METADATA,
                "metadata_sha256": e186.digest(metadata),
                "output_path": e186.OUTPUT,
                "parameters": e186.PARAMETERS,
                "parent_supervisor_sha256": e186.digest(supervisor),
            }
        ),
        encoding="utf-8",
    )
    review.write_text(
        json.dumps(
            {
                "schema": "flylab.m3.e186.force-command.source-review.v1",
                "approved": True,
                "blocking_defects": [],
                "freeze_sha256": e186.digest(freeze),
            }
        ),
        encoding="utf-8",
    )
    design.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="design identity"):
        e186.gate(freeze, review, out)
    design.write_text("design", encoding="utf-8")
    review.write_text(
        json.dumps(
            {
                "schema": "flylab.m3.e186.force-command.source-review.v1",
                "approved": True,
                "blocking_defects": [],
                "freeze_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="parent source approval"):
        e186.gate(freeze, review, out)


def complete_synthetic_gate(tmp_path, monkeypatch):
    """Reach all source/input checks using generated files only."""
    monkeypatch.setattr(e186, "ROOT", tmp_path)
    design = tmp_path / e186.DESIGN
    design.parent.mkdir(parents=True)
    design.write_text("generated design", encoding="utf-8")
    supervisor_rel = (
        "verification/banc-natural-gait-m3-20260919/parent-e186-supervise.py"
    )
    supervisor = tmp_path / supervisor_rel
    supervisor.parent.mkdir(parents=True, exist_ok=True)
    supervisor.write_text("generated supervisor", encoding="utf-8")
    prior_rel = "verification/banc-natural-gait-m3-20260919/generated-prior.json"
    prior = tmp_path / prior_rel
    prior.write_text(
        json.dumps(
            {
                "saved_raw_analysis_accepted": True,
                "child_observation_valid": True,
                "original_driver_accepted": False,
            }
        ),
        encoding="utf-8",
    )
    inputs = {prior_rel: e186.digest(prior)}
    for i in range(198):
        rel = f"generated-inputs/input-{i:03}.bin"
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes([i % 251]))
        inputs[rel] = e186.digest(path)
    sources = {}
    for i in range(17):
        rel = f"generated-sources/source-{i:02}.py"
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# generated {i}\n", encoding="utf-8")
        sources[rel] = e186.digest(path)
    metadata = tmp_path / e186.METADATA
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(
        json.dumps(
            {
                "schema": "flylab.m3.e186.force-command.metadata.v1",
                "design_path": e186.DESIGN,
                "design_sha256": e186.digest(design),
                "cumulative_before_s": e186.PARAMETERS["inherited_cumulative_wall_s"],
                "stage_reserve_s": 45,
                "original_analysis_runs": 0,
                "inputs": inputs,
                "source_hashes": sources,
                "previous_review_path": prior_rel,
            }
        ),
        encoding="utf-8",
    )
    all_sources = sources.copy()
    for rel in e186.OWNED:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# generated owned\n", encoding="utf-8")
        all_sources[rel] = e186.digest(path)
    freeze = tmp_path / "freeze.json"
    freeze_payload = {
        "schema": "flylab.m3.e186.force-command.freeze.v1",
        "design_path": e186.DESIGN,
        "design_sha256": e186.digest(design),
        "metadata_path": e186.METADATA,
        "metadata_sha256": e186.digest(metadata),
        "output_path": e186.OUTPUT,
        "parameters": e186.PARAMETERS,
        "parent_supervisor_sha256": e186.digest(supervisor),
        "source_hashes": all_sources,
        "input_sha256": inputs,
    }
    freeze.write_text(json.dumps(freeze_payload), encoding="utf-8")
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "schema": "flylab.m3.e186.force-command.source-review.v1",
                "approved": True,
                "blocking_defects": [],
                "freeze_sha256": e186.digest(freeze),
                "design_sha256": e186.digest(design),
                "metadata_sha256": e186.digest(metadata),
                "source_hashes": all_sources,
            }
        ),
        encoding="utf-8",
    )
    return (
        freeze,
        review,
        tmp_path / e186.OUTPUT,
        tmp_path / next(iter(inputs)),
        tmp_path / next(iter(sources)),
    )


def test_complete_gate_rejects_source_and_input_tampering(tmp_path, monkeypatch):
    freeze, review, out, input_path, source_path = complete_synthetic_gate(
        tmp_path, monkeypatch
    )
    assert e186.gate(freeze, review, out)[1] == e186.digest(freeze)
    source_path.write_text("mutated", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        e186.gate(freeze, review, out)
    source_path.write_text("# generated 0\n", encoding="utf-8")
    input_path.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="input changed"):
        e186.gate(freeze, review, out)


def test_existing_output_refused_and_publish_error_receipt(tmp_path, monkeypatch):
    freeze, review, out, input_path, _ = complete_synthetic_gate(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="fixed output"):
        e186.execute(freeze, review, input_path)
    out.mkdir()
    with pytest.raises(ValueError, match="output exists"):
        e186.execute(freeze, review, out)
    assert not (out / "terminal.json").exists()
    out.rmdir()
    monkeypatch.setattr(e186, "gate", lambda *args: ({}, "f" * 64))
    monkeypatch.setattr(
        e186,
        "analyze",
        lambda metadata: ({"schema": "generated"}, {"a": np.array([1])}),
    )
    monkeypatch.setattr(
        e186.np,
        "savez_compressed",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("generated publication failure")
        ),
    )
    with pytest.raises(OSError, match="publication failure"):
        e186.execute(freeze, review, out)
    receipt = json.loads((out / "terminal.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "failed" and receipt["error"]["type"] == "OSError"


def test_generated_force_inventory_and_bw_heading_identity():
    n = 30000
    segment = np.zeros((n, 30, 3))
    leg_segment = np.zeros((n, 48, 3))
    rotation = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    axis = np.broadcast_to([1.0, 0, 0], (n, 3)).copy()
    force = {
        "available": np.ones(n, dtype=bool),
        "finite": np.ones(n, dtype=bool),
        "error": np.full(n, ""),
        "thorax_rotation": rotation,
        "forward": axis,
        "heading_xy": axis,
        "segment_force": segment,
        "leg_force": segment.reshape(n, 6, 5, 3).sum(axis=2),
        "all_leg_segment_force": leg_segment,
        "all_leg_force": leg_segment.reshape(n, 6, 8, 3).sum(axis=2),
        "full_fly_floor_force": np.zeros((n, 3)),
        "nonleg_floor_force_residual": np.zeros((n, 3)),
    }
    force["all_leg_segment_force"][0, 0, 0] = 20
    force["all_leg_force"][0, 0, 0] = 20
    force["full_fly_floor_force"][0, 0] = 20
    for prefix in ("leg", "all_leg"):
        for label, projection in (("forward", axis), ("heading_xy", axis)):
            force[f"{prefix}_{label}_force"] = np.einsum(
                "tlj,tj->tl", force[f"{prefix}_force"], projection
            )
    for label in ("forward", "heading_xy"):
        force[f"full_fly_{label}_force"] = force["full_fly_floor_force"][:, 0]
    for name, value in list(force.items()):
        if name.endswith(("force", "residual")):
            force[name + "_bw"] = value / 2
    inventory = {
        "body_ids": [1],
        "body_mass": [0.2],
        "body_mass_total": 0.2,
        "body_mass_reported": 0.2,
        "weight0": 2.0,
        "gravity": [0, 0, -10],
        "body_model_hash": "body",
        "leg_order": list(e186.LEGS),
    }
    metadata = {
        "schema": "flylab.m3.e185.ground-reaction-metadata.v1",
        "inventory": inventory,
        "force_samples": n,
        "original_native_samples": n,
        "primary_error": None,
        "cleanup_errors": [],
        "raw_persistence_errors": [],
    }
    assert e186.validate_force_semantics(force, metadata, "body")["body_weight"] == 2
    force["all_leg_force_bw"][0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="force/BW"):
        e186.validate_force_semantics(force, metadata, "body")
