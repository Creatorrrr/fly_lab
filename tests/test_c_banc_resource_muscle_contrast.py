"""Generated-only E308 arithmetic, selective ownership, and failure contracts."""

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from tools import analyze_c_banc_m3_resource_muscle_contrast as a


@pytest.fixture
def case():
    t, n, m, j = 101, 3, 2, 2
    meta = {
        "motor_ids": np.array([5, 3, 4], np.int64),
        "membership": np.array([0, 0, 1], np.int64),
        "fixed_muscles": np.array([1], np.int64),
        "ltm": np.array([1], np.int64),
        "groups": [
            {"columns": [0, 1], "leg": "lf", "muscle": "variable"},
            {"columns": [2], "leg": "lm", "muscle": "fixed"},
        ],
        "moment": np.array([[-2.0, 1.0], [0.5, -1.0]], np.float64),
        "rest": np.zeros(j, np.float64),
        "kappa": np.ones(n, np.float64),
        "tau": np.array([0.005, 0.01, 0.04]),
        "slopes": np.zeros((m, n, 3, 2), np.float64),
    }
    meta["slopes"][0, 0, :, 0] = 0.2
    meta["slopes"][0, 0, :, 1] = 0.1
    meta["slopes"][0, 1, :, 0] = 0.1
    meta["slopes"][0, 1, :, 1] = 0.3
    observed = np.array([4, 0, 5, 2, 1, 3], np.int64)
    source = np.array([2, 0, 1], np.int64)

    def arm(scale):
        clock = {
            "control_tick": np.arange(t, dtype=np.int64),
            "network_tick": np.arange(t, dtype=np.int64) * 20,
            "physics_time_s": np.arange(t, dtype=np.float64) * 0.005,
            "complete": np.ones(t, bool),
        }
        bank = np.ones((t, n, 3), np.float64) * scale
        original = np.zeros((t, m), np.float64)
        original[:, 1] = 0.1 * scale
        activation = original.copy()
        terms = a.motor_contributions(bank, meta, 0)
        activation[:, 0] = np.sum(terms["low"], axis=(1, 2)) + np.sum(
            terms["hinge"], axis=(1, 2)
        )
        rate = np.tile(np.arange(6, dtype=np.float32), (t, 1)) * scale
        cols = [list(observed).index(v) for v in meta["motor_ids"]]
        r = {
            **clock,
            "bank": bank,
            "activation": activation,
            "original_activation": original,
            "motor_rate": rate[:, cols].astype(np.float64),
        }
        r.update(
            {
                key + "_available": np.ones(t, bool)
                for key in ("bank", "activation", "original_activation", "motor_rate")
            }
        )
        activity = {
            **clock,
            "observed_ids": observed,
            "motor_ids": meta["motor_ids"],
            "source_ids": source,
            "rate": rate,
            "drive": rate + 0.5,
            "adaptation": rate + 0.25,
            "synaptic_resource": np.ones((t, 3), np.float64),
        }
        factors = {**clock, "source_ids": source, "g32": np.ones((t, 3), np.float32)}
        return {
            "q": meta["rest"] + 0.02 * (activation @ meta["moment"].T),
            "ltm": 0.2 * np.tanh(activation[:, [1]] / 10),
            **{
                k: r[k]
                for k in ("bank", "activation", "original_activation", "motor_rate")
            },
            "recruitment": r,
            "activity": activity,
            "factors": factors,
            "observed_ids": observed,
            "source_ids": source,
            "start_mask": np.ones(7, np.float32),
            "final_mask": np.ones(7, np.float32),
            "cuts_unchanged_false": True,
        }

    active, neutral = arm(2), arm(1)
    # Existing positive/negative/zero incoming edges, no selected motor target.
    csr = {
        "indptr": np.array([0, 0, 0, 0, 2, 3, 6, 6], np.int32),
        "indices": np.array([0, 1, 2, 0, 1, 2], np.int32),
        "data": np.array([1, -2, 0, 2, -3, 0], np.float32),
    }
    roster = {"edge_ids": np.empty(0, np.int64)}
    target = active["q"][1:100].copy()
    target[4, 1] -= 0.2
    # Largest contrast occurs at joint0/time0; distinct startup max control5/joint1.
    return active, neutral, meta, target, csr, roster, ["joint-a", "joint-b"]


def test_signed_all_muscle_transfer_and_distinct_peaks(case):
    summary, raw = a.analyze(*case)
    assert summary["run_valid"]
    assert raw["muscle_contributions"].shape == (101, 2, 2)
    assert np.max(abs(raw["muscle_contributions"].sum(axis=2) - raw["delta_q"])) < 1e-15
    assert summary["largest_active_minus_neutral"]["joint"] == 0
    assert summary["separate_active_startup_error_peak"]["joint"] == 1
    assert summary["separate_active_startup_error_peak"]["control_tick"] == 5
    assert len(summary["all_joint_maxima"]) == 2
    assert (
        raw["muscle_contributions"][0, 0, 0] < 0 < raw["muscle_contributions"][0, 0, 1]
    )
    assert summary["major_muscles"][1]["motor_decomposition_available"] is False
    assert summary["motors"][0]["full_endpoint_sum_available"]
    assert summary["full_current_claim"] is False and summary["causal_claim"] is False


def test_hinge_negative_delta_is_not_negative_synapse(case):
    meta = case[2]
    bank = case[0]["bank"]
    terms = a.motor_contributions(bank, meta, 0)
    assert terms["hinge"][0, 0, 0] < 0 and terms["low"][0, 0, 0] > 0
    assert np.array_equal(terms["columns"], [0, 1])
    # Literal piecewise low/high curve, independently of the hinge helper.
    expected = np.minimum(bank[:, [0, 1]], 1) * meta["slopes"][0, [0, 1], :, 0]
    expected += np.maximum(bank[:, [0, 1]] - 1, 0) * meta["slopes"][0, [0, 1], :, 1]
    np.testing.assert_allclose(
        terms["low"] + terms["hinge"], expected, rtol=0, atol=1e-16
    )


def test_control_zero_anchor_is_retained_not_absolute_decoded(case):
    case[0]["q"][0] = 20
    case[1]["q"][0] = 20
    case[0]["ltm"][0] = 0.25
    case[1]["ltm"][0] = 0.25
    summary, raw = a.analyze(*case)
    assert (
        summary["run_valid"] and not summary["anchor_absolute_reconstruction_assessed"]
    )
    assert np.array_equal(raw["active_anchor_q"], [20, 20])


def test_reconstruction_failure_retains_actual_residual(case):
    case[0]["q"][8, 1] += 0.01
    summary, raw = a.analyze(*case)
    assert summary["run_valid"] is False
    assert summary["checks"]["active_q"]["max_abs"] > 0.009
    assert raw["delta_q_residual"][8, 1] < -0.009


def test_neutral_nonunity_resource_cannot_be_valid_contrast(case):
    case[1]["factors"]["g32"][30, 1] = 0.9
    summary, _ = a.analyze(*case)
    assert (
        not summary["run_valid"]
        and not summary["checks"]["neutral_resource_unity"]["pass"]
    )


def test_real_rss_measurement_and_public_contract():
    assert 0 < a.peak_rss_bytes() < 1536 * 1024**2
    assert "final_state_dir" in a.ARM_KEYS
    assert a.CONTRACT["neural_ticks"] == 0 and a.CONTRACT["supervisor_s"] == 20


def test_no_input_mutation(case):
    before = copy.deepcopy(case)
    a.analyze(*case)

    def equal(x, y):
        if isinstance(x, np.ndarray):
            return np.array_equal(x, y)
        if isinstance(x, dict):
            return all(equal(v, y[k]) for k, v in x.items())
        if isinstance(x, (tuple, list)):
            return all(equal(v, w) for v, w in zip(x, y, strict=True))
        return x == y

    assert equal(before, case)


@pytest.mark.parametrize(
    "defect",
    [
        "clock",
        "dtype",
        "nan",
        "availability",
        "order",
        "negative",
        "group",
        "factor_order",
        "source_missing",
        "mask",
    ],
)
def test_invalid_recordings_rejected(case, defect):
    active, neutral, meta = case[:3]
    if defect == "clock":
        active["activity"]["network_tick"][7] += 1
    if defect == "dtype":
        active["bank"] = active["bank"].astype(np.float32)
    if defect == "nan":
        active["q"][4, 0] = np.nan
    if defect == "availability":
        active["recruitment"]["activation_available"][4] = False
    if defect == "order":
        neutral["observed_ids"] = neutral["observed_ids"][::-1].copy()
    if defect == "negative":
        active["activity"]["rate"][2, 0] = -1
    if defect == "group":
        meta["groups"][0]["columns"] = [1, 0]
    if defect == "factor_order":
        active["factors"]["source_ids"] = np.array([0, 2, 1], np.int64)
    if defect == "source_missing":
        active["source_ids"] = np.array([6, 0, 1], np.int64)
    if defect == "mask":
        active["final_mask"][0] = 0.5
    with pytest.raises(ValueError):
        a.analyze(*case)


def test_missing_incoming_is_nan_not_zero_filled(case):
    active, neutral, _, _, csr = case[:5]
    csr["indices"][3] = 6
    summary, raw = a.incoming_contributions(
        active, neutral, csr, 5, unit_masks=True, no_direct_resource=True
    )
    assert summary["missing_ids"] == [6] and summary["coverage"] == pytest.approx(2 / 3)
    assert not summary["full_endpoint_sum_available"]
    assert np.isnan(raw["terms64"][:, 0]).all() and np.isnan(raw["full_sum"]).all()
    assert np.isfinite(raw["observed_partial_sum"]).all()


@pytest.mark.parametrize("mode", ["mask", "direct_resource", "cuts"])
def test_full_endpoint_sum_requires_all_conditions(case, mode):
    active, _, _, _, _, roster = case[:6]
    if mode == "mask":
        active["final_mask"][0] = 0
    if mode == "direct_resource":
        roster["edge_ids"] = np.array([0], np.int64)
    if mode == "cuts":
        active["cuts_unchanged_false"] = False
    summary, raw = a.analyze(*case)
    assert all(not m["full_endpoint_sum_available"] for m in summary["motors"])
    assert np.isnan(raw["motor_5_full_sum"]).all()
    assert "motor_5_active_drive32" in raw and "motor_5_adaptation_delta64" in raw


def test_input_signs_float64_and_unsorted_ids(case):
    active, neutral, _, _, csr = case[:5]
    summary, raw = a.incoming_contributions(
        active, neutral, csr, 5, unit_masks=True, no_direct_resource=True
    )
    # ID0 occupies observed column1; ID1 occupies column4.
    assert raw["positive"][0] == 2
    assert raw["negative"][0] == -12
    assert raw["zero"][0] == 0
    assert raw["full_sum"][0] == -10 and summary["missing_ids"] == []


def state_fixture(tmp_path):
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    values = {
        "mask": np.ones(7, np.float32),
        "kappa": np.array([1.0]),
        "huge": np.ones(100, np.float32),
    }
    files = {}
    for name, value in values.items():
        np.save(directory / (name + ".npy"), value, allow_pickle=False)
        files[name + ".npy"] = a.sha256(directory / (name + ".npy"))
    state = {
        "body": {"modelHash": "body"},
        "parameters": {"neural_dt_s": 0.00025},
        "cuts": {"motor": False},
        "control_tick": 0,
        "network": {
            "tick": 0,
            "output_mask": {"__npy__": "mask.npy"},
            "rate": {"__npy__": "huge.npy"},
        },
        "adapter": {
            "hash": "adapter",
            "schema": "flylab.m3-monotone-recruitment-adapter.v1",
            "groups": [],
            "parameters": {"kappa": {"__npy__": "kappa.npy"}},
            "calibration": {},
            "bank": [],
            "activation": [],
            "original": {"activation": []},
        },
    }
    a.write_json(directory / "state.json", state)
    files["state.json"] = a.sha256(directory / "state.json")
    manifest = {
        "schema": "flylab.state_store.v3",
        "files": files,
        "state_hash": hashlib.sha256(a.canonical(files)).hexdigest(),
    }
    a.write_json(directory / "manifest.json", manifest)
    pins = a.Pins(
        tmp_path,
        {p.relative_to(tmp_path).as_posix(): a.sha256(p) for p in directory.iterdir()},
    )
    return directory, state, manifest, pins


def test_selective_real_shaped_modelhash_no_neural_unpack(tmp_path, monkeypatch):
    _, _, _, pins = state_fixture(tmp_path)
    original = np.load
    loaded = []

    def load(path, **kwargs):
        loaded.append(Path(path).name)
        return original(path, **kwargs)

    monkeypatch.setattr(np, "load", load)
    state = a.selective_state(pins, "checkpoint", adapter=True)
    assert state["body_hash"] == "body" and set(loaded) == {"mask.npy", "kappa.npy"}
    assert "huge.npy" not in loaded


@pytest.mark.parametrize(
    "defect", ["hash", "identity", "escape", "malformed", "missing"]
)
def test_checkpoint_manifest_and_leaf_tamper(tmp_path, defect):
    directory, state, manifest, pins = state_fixture(tmp_path)
    if defect == "hash":
        (directory / "mask.npy").write_bytes(b"changed")
    elif defect == "identity":
        manifest["state_hash"] = "0" * 64
    elif defect == "escape":
        state["network"]["output_mask"] = {"__npy__": "../mask.npy"}
    elif defect == "malformed":
        state["network"]["output_mask"] = {"__npy__": "mask.npy", "other": 1}
    elif defect == "missing":
        state["network"]["output_mask"] = {"__npy__": "missing.npy"}
    if defect in ("escape", "malformed", "missing"):
        a.write_json(directory / "state.json", state)
        manifest["files"]["state.json"] = a.sha256(directory / "state.json")
        manifest["state_hash"] = hashlib.sha256(
            a.canonical(manifest["files"])
        ).hexdigest()
    if defect != "hash":
        a.write_json(directory / "manifest.json", manifest)
        pins = a.Pins(
            tmp_path,
            {
                p.relative_to(tmp_path).as_posix(): a.sha256(p)
                for p in directory.iterdir()
            },
        )
    with pytest.raises(ValueError):
        a.selective_state(pins, "checkpoint")


def fake_run(tmp_path, case, **overrides):
    output = tmp_path / a.OUTPUT
    freeze = {
        "input_sha256": {"fake": "0" * 64},
        "owned_sha256": {},
        "reference_source_sha256": {},
        "design": {"path": a.DESIGN, "sha256": a.DESIGN_SHA},
    }
    args = {
        "root": tmp_path,
        "gate": lambda *args, **kwargs: (freeze, None, {"fake": True}),
        "loader": lambda *args: case,
        "rss": lambda: 1,
    }
    args.update(overrides)
    return lambda: a.run(
        tmp_path / "freeze", "0" * 64, tmp_path / "review", "0" * 64, output, **args
    ), output


def test_success_publication_manifest_and_no_promotion(tmp_path, case):
    run, output = fake_run(tmp_path, case)
    result = run()
    receipt = a.read_json(output / "result-sha256.json")
    assert result["run_valid"] and receipt["sha256"] == a.sha256(output / "result.json")
    manifest = a.read_json(output / "output-manifest.json")
    assert all(
        a.sha256(output / name) == entry["sha256"]
        for name, entry in manifest["files"].items()
    )
    assert (
        result["cost"]["analysis_completed"] == 1 and result["cost"]["model_calls"] == 0
    )
    assert not result["F1_F8_complete"] and not result["candidate_adopted"]
    assert a.verify_publication(output)["run_valid"]


def test_late_publication_and_locked_seals_explicitly_invalidated(
    tmp_path, case, monkeypatch
):
    primary = OSError("after manifest write")
    locked = OSError("seal lock")
    original_unlink = Path.unlink

    def unlink(path, *args, **kwargs):
        if path.name in ("result-sha256.json", "output-manifest.json"):
            raise locked
        return original_unlink(path, *args, **kwargs)

    def writer(path, value):
        a.write_json(path, value)
        if path.name == "output-manifest.json":
            raise primary

    monkeypatch.setattr(Path, "unlink", unlink)
    run, output = fake_run(tmp_path, case, writer=writer)
    with pytest.raises(OSError) as caught:
        run()
    assert caught.value is primary and primary.cost_receipt["analysis_completed"] == 1
    assert (output / "result-sha256.json").exists() and (
        output / "output-manifest.json"
    ).exists()
    assert (output / "invalidation.json").exists() and len(primary.__notes__) >= 2
    assert (
        a.read_json(output / "result.json")["publication_status"] == "invalid_partial"
    )
    assert "muscle_contributions" in primary.partial_raw
    with pytest.raises(ValueError, match="invalidated"):
        a.verify_publication(output)


@pytest.mark.parametrize("defect", [None, "wrong_mask", "wrong_time", "extra_target"])
def test_exact_original99_reference_schema(defect):
    startup = {
        "time_s": np.arange(2001, dtype=np.float64) * 0.005,
        "mask": np.zeros(2001, bool),
        "target_command": np.zeros((99, 2), np.float64),
    }
    startup["mask"][1:100] = True
    if defect == "wrong_mask":
        startup["mask"][0] = True
        startup["mask"][99] = False
    if defect == "wrong_time":
        startup["time_s"][4] += 0.1
    if defect == "extra_target":
        startup["target_command"] = np.zeros((100, 2), np.float64)
    if defect is None:
        a.validate_startup_reference(startup, 2)
    else:
        with pytest.raises(ValueError):
            a.validate_startup_reference(startup, 2)


@pytest.mark.parametrize("arm", ["active", "neutral"])
@pytest.mark.parametrize("audit_bio", [False, True])
def test_actual_shaped_producer_flags_cost_review_and_manifest(
    tmp_path, monkeypatch, arm, audit_bio
):
    number = 306 if arm == "active" else 307
    count = 2 if number == 306 else 1
    prefix = (
        f"flylab.m3.e{number}.ssp-{'recovery' if number == 306 else 'neutral'}-body."
    )
    cost = {
        "construction_calls": count,
        "known_executed_setup_ticks": 280 * count,
        "recorded_capture_setup_ticks": 140 * count,
        "known_settling_physics_steps": 2000 * count,
        "scientific_neural_ticks": 2020 if number == 306 else 2000,
        "scientific_physics_steps": 5050 if number == 306 else 5000,
        "unknown_cost_arms": [],
        "unknown_setup_calls": [],
    }
    result = {
        "schema": prefix + "result.v1",
        "diagnostic_id": f"E{number}",
        "run_valid": True,
        "publication_status": "complete",
        **dict.fromkeys(a.FLAGS, False),
        "actual_startup_pass": arm == "neutral",
        "neutral_prefix_valid": True,
        "ssp_neural_endpoint_exact": True,
        "cost": cost,
    }
    if number == 306:
        result.pop("biological_claim")
    audit = {
        "schema": f"flylab.m3.e{number}.root-original-raw-audit.v1",
        "pass": True,
        "diagnostic_run_valid": True,
        "actual_startup_pass" if number == 306 else "original_strict_startup_pass": arm
        == "neutral",
        "candidate_rejected": True,
        "cost": cost,
        **dict.fromkeys(
            ("candidate_adopted", "F1_F8_complete", "F2_complete", "F6_exposed"), False
        ),
        "biological_claim": audit_bio,
    }
    entry = {
        "completed_controls": 100,
        "native_samples": 5000,
        "close_returned": True,
        "clock_valid": True,
        "continuation_entered": False,
        "actual_neural_s": 0.5,
        "actual_physical_s": 0.5,
    }
    ledger = {
        "original_attempts": 1,
        "retries": 0,
        "fit_calls": 0,
        "arms": {"candidate" if number == 306 else "neutral": entry},
    }
    for name, value in (
        ("result", result),
        ("audit", audit),
        ("ledger", ledger),
        ("freeze", {"schema": prefix + "freeze.v1"}),
    ):
        a.write_json(tmp_path / (name + ".json"), value)
    result_hash = a.sha256(tmp_path / "result.json")
    monkeypatch.setitem(a.RESULT_SHAS, arm, result_hash)
    a.write_json(tmp_path / "result-sha.json", {"sha256": result_hash})
    a.write_json(
        tmp_path / "review.json",
        {
            "schema": prefix + "review.v1",
            "approved": True,
            "blocking_defects": [],
            "freeze_sha256": a.sha256(tmp_path / "freeze.json"),
            "source_sha256": {},
        },
    )
    a.write_json(
        tmp_path / "sources.json",
        {
            "schema": f"flylab.m3.e{number}.sources.v1",
            "diagnostic_id": f"E{number}",
            "source_sha256": {},
            "owned_sha256": {},
        },
    )
    a.write_json(
        tmp_path / "manifest.json",
        {
            "files": {
                "result.json": {
                    "sha256": result_hash,
                    "bytes": (tmp_path / "result.json").stat().st_size,
                }
            }
        },
    )
    pins = a.Pins(tmp_path, {p.name: a.sha256(p) for p in tmp_path.iterdir()})
    spec = {
        "result": "result.json",
        "root_audit": "audit.json",
        "ledger": "ledger.json",
        "result_sha": "result-sha.json",
        "review": "review.json",
        "freeze": "freeze.json",
        "source_manifest": "sources.json",
        "output_manifest": "manifest.json",
    }
    if audit_bio:
        with pytest.raises(ValueError, match="promotion"):
            a.validate_producer(arm, spec, pins)
    else:
        assert a.validate_producer(arm, spec, pins)["startup_pass"] is (
            arm == "neutral"
        )


@pytest.mark.parametrize(
    "artifact",
    [
        "input-receipt.json",
        "source-manifest.json",
        "summary.json",
        "result.json",
        "result-sha256.json",
        "output-manifest.json",
    ],
)
def test_publication_error_identity_partial_counts_no_stale_receipt(
    tmp_path, case, artifact
):
    original = OSError("primary " + artifact)

    def writer(path, value):
        if path.name == artifact:
            raise original
        a.write_json(path, value)

    run, output = fake_run(tmp_path, case, writer=writer)
    with pytest.raises(OSError) as caught:
        run()
    assert caught.value is original
    assert original.partial_result["publication_status"] == "invalid_partial"
    assert (
        not (output / "result-sha256.json").exists()
        and not (output / "output-manifest.json").exists()
    )
    assert original.cost_receipt["gate_completed"] == 1
    assert original.cost_receipt["analysis_completed"] == int(
        artifact not in ("input-receipt.json", "source-manifest.json")
    )


def test_secondary_storage_failure_does_not_replace_primary(tmp_path, case):
    primary = ValueError("load")
    secondary = OSError("partial storage")

    def loader(*args):
        raise primary

    def writer(path, value):
        if path.name == "failure.json":
            raise secondary
        a.write_json(path, value)

    run, _ = fake_run(tmp_path, case, loader=loader, writer=writer)
    with pytest.raises(ValueError) as caught:
        run()
    assert caught.value is primary and any(
        "partial storage" in n for n in primary.__notes__
    )
    assert (
        primary.cost_receipt["load_requested"] == 1
        and primary.cost_receipt["load_completed"] == 0
    )
    assert primary.cost_receipt["analysis_requested"] == 0


def test_deadline_halts_without_load_or_retry(tmp_path, case):
    times = iter([0, 17])
    run, _ = fake_run(tmp_path, case, wall_clock=lambda: next(times))
    with pytest.raises(ValueError) as caught:
        run()
    assert (
        caught.value.cost_receipt["load_requested"] == 0
        and caught.value.cost_receipt["retries"] == 0
    )


def test_existing_output_is_never_overwritten(tmp_path, case):
    run, output = fake_run(tmp_path, case)
    output.mkdir(parents=True)
    with pytest.raises(ValueError, match="already exists"):
        run()


def test_forbidden_imports_absent():
    import ast

    source = Path(a.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    names += [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not any(
        name and (name.startswith(("torch", "flylab", "cupy", "scipy")))
        for name in names
    )


def test_schema_gate_rejects_wrong_contract_before_input_load(tmp_path):
    freeze = {
        "schema": a.FREEZE_SCHEMA,
        "contract": {},
        "output": a.OUTPUT,
        "design": {"path": a.DESIGN, "sha256": a.DESIGN_SHA},
    }
    a.write_json(tmp_path / "freeze.json", freeze)
    a.write_json(tmp_path / "review.json", {})
    with pytest.raises(ValueError, match="contract"):
        a.verified_gate(
            tmp_path / "freeze.json",
            a.sha256(tmp_path / "freeze.json"),
            tmp_path / "review.json",
            a.sha256(tmp_path / "review.json"),
            root=tmp_path,
        )


@pytest.mark.parametrize(
    "field",
    ["run_valid", "publication_status", "candidate_adopted", "actual_startup_pass"],
)
def test_failed_or_promoted_producer_never_passes(tmp_path, monkeypatch, field):
    result = {
        "schema": "flylab.m3.e306.ssp-recovery-body.result.v1",
        "diagnostic_id": "E306",
        "run_valid": True,
        "publication_status": "complete",
        **dict.fromkeys(a.FLAGS, False),
        "actual_startup_pass": False,
        "neutral_prefix_valid": True,
        "ssp_neural_endpoint_exact": True,
    }
    result[field] = {
        "run_valid": False,
        "publication_status": "partial",
        "candidate_adopted": True,
        "actual_startup_pass": True,
    }[field]
    for name, value in (("result", result), ("audit", {}), ("ledger", {})):
        a.write_json(tmp_path / (name + ".json"), value)
    digest = a.sha256(tmp_path / "result.json")
    a.write_json(tmp_path / "sha.json", {"sha256": digest})
    monkeypatch.setitem(a.RESULT_SHAS, "active", digest)
    pins = a.Pins(tmp_path, {p.name: a.sha256(p) for p in tmp_path.iterdir()})
    with pytest.raises(ValueError):
        a.validate_producer(
            "active",
            {
                "result": "result.json",
                "root_audit": "audit.json",
                "ledger": "ledger.json",
                "result_sha": "sha.json",
            },
            pins,
        )
