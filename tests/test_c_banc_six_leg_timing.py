import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "e163", ROOT / "tools/analyze_c_banc_m3_six_leg_timing.py"
)
e163 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(e163)

FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "e163_fixture",
    ROOT
    / "verification/banc-natural-gait-m3-20260919/parent-e163-generated-fixture.py",
)
fixture = importlib.util.module_from_spec(FIXTURE_SPEC)
assert FIXTURE_SPEC.loader
FIXTURE_SPEC.loader.exec_module(fixture)


def generated_data():
    data = fixture.make_fixture()
    data["old_events"]["offset_s"] = np.arange(-20, 21) * e163.CONTROL_DT
    return data


def generated_run_environment(tmp_path, monkeypatch):
    data, paths = fixture.save_fixture(tmp_path / "fixture")
    data["old_events"]["offset_s"] = np.arange(-20, 21) * e163.CONTROL_DT
    with paths["old_events"].open("wb") as stream:
        np.savez_compressed(stream, **data["old_events"])
    for relative in e163.OWNED:
        owned = tmp_path / relative
        owned.parent.mkdir(parents=True, exist_ok=True)
        owned.write_text("generated owned source", encoding="utf-8")
    counts = {
        "original_processes": 1,
        "automatic_retries": 0,
        "body_models": 0,
        "state_store_instances": 0,
        "physics_steps": 0,
        "neural_steps": 0,
        "fit_calls": 0,
        "candidate_selections": 0,
        "F6_exposed": False,
    }
    freeze = {
        "paths": {key: str(path) for key, path in paths.items()},
        "sample_contract": {"group_legs": data["group_legs"]},
        "body_hash": data["body_hash"],
        "counts": counts,
    }
    review = {
        "source_sha256": {
            relative: e163.sha256(tmp_path / relative) for relative in e163.OWNED
        }
    }
    freeze_path = tmp_path / "freeze.json"
    review_path = tmp_path / "review.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    review_path.write_text(json.dumps(review), encoding="utf-8")
    frozen = {
        str(path.relative_to(tmp_path)).replace("\\", "/"): e163.sha256(path)
        for path in paths.values()
    }
    monkeypatch.setattr(e163, "ROOT", tmp_path)
    monkeypatch.setattr(e163, "_path", lambda value, key: Path(value["paths"][key]))
    monkeypatch.setattr(
        e163,
        "verified_inputs",
        lambda *unused: (freeze, review),
    )
    monkeypatch.setattr(e163, "frozen_hashes", lambda unused: frozen)
    monkeypatch.setattr(e163, "verify_hash_map", lambda *unused: None)
    monkeypatch.setattr(e163, "threadpool_info", lambda: [{"num_threads": 1}])
    return (
        freeze_path,
        e163.sha256(freeze_path),
        review_path,
        e163.sha256(review_path),
    )


def test_quadratic_peak_and_phase_have_no_synthetic_end_brackets():
    times = np.arange(9, dtype=float) * 0.005
    values = np.array([0, 0, 1, 4, 1, 0, 1, 3, 1], dtype=float)
    peaks = e163.quadratic_peaks(times, values, prominence=1)
    assert peaks.tolist() == pytest.approx([0.015, 0.035])
    query = np.array([0.0, 0.015, 0.025, 0.035, 0.04])
    phase, available = e163.phase_trace(query, peaks)
    assert available.tolist() == [False, True, True, True, False]
    assert phase[1:4].tolist() == pytest.approx([0.0, 0.5, 0.0])
    assert np.isnan(phase[[0, 4]]).all()


def test_all_15_pair_windows_wrap_and_drift():
    query = np.arange(0.5, 10.0, 0.025)
    base = np.mod(query / 0.2, 1)
    offsets = np.array([0.95, 0.05, 0.25, 0.45, 0.65, 0.85])
    phase = np.mod(base[:, None] + offsets, 1)
    result = e163.pair_phase_statistics(phase, query)
    assert result["pair_indices"].shape == (15, 2)
    assert result["pair_mean"].shape == (15, 3)
    assert np.all(result["pair_R"] > 0.999999)
    assert np.all(result["pair_max_drift"] < 1e-12)
    assert result["pair_time_consistent"].all()
    # LF-LM is 0.95-0.05 = 0.90 modulo one, not a linear -0.10 mean.
    assert result["pair_mean"][0, 0] == pytest.approx(0.9)


def test_pair_statistics_require_common_six_leg_bracket():
    query = np.arange(0.5, 10.0, 0.025)
    phase = np.zeros((len(query), 6))
    phase[0, 5] = np.nan
    result = e163.pair_phase_statistics(phase, query)
    assert not result["pair_common_bracket"][0]
    assert np.all(result["pair_count"][:, 0] == 79)


def test_bimodal_events_require_second_harmonic():
    values = np.tile([0.1, 0.6], 20)
    mean1, r1, count1 = e163.circular_summary(values, 1)
    mean2, r2, count2 = e163.circular_summary(values, 2)
    assert count1 == count2 == 40
    assert r1 < 1e-12
    assert r2 == pytest.approx(1.0)
    assert mean2 == pytest.approx(0.1)
    assert np.isfinite(
        mean1
    )  # Weak-R mean exists but is not evidence of a strong relation.


def test_crossings_use_frozen_window_and_missing_brackets_remain_nan():
    times = np.arange(101) * 0.005
    speed = np.ones(101)
    speed[[20, 60]] = 0
    window = np.zeros(101, bool)
    window[10:100] = True
    events = e163.interruption_events(speed, times, window)
    assert events["crossing_index"].tolist() == [20, 60]
    assert events["event_index"].tolist() == [60]
    phase, available = e163.phase_trace(times, np.array([0.15, 0.25]))
    assert not available[20] and available[40] and not available[60]
    assert np.isnan(phase[[20, 60]]).all()


def test_native_sign_split_precedes_block_mean():
    force = np.zeros((10000, 6))
    force[:, 0] = np.tile(np.r_[np.ones(25) * 2, np.ones(25) * -2], 200)
    load = np.ones((10000, 6)) * 0.03
    result = e163.native_force_blocks(force, load)
    assert np.allclose(result["block_mean_force"][:, 0], 0)
    assert np.allclose(result["block_positive_impulse"][:, 0], 0.005)
    assert np.allclose(result["block_negative_impulse"][:, 0], 0.005)
    assert np.max(np.abs(result["block_impulse_closure"])) == 0


def test_force_event_pre_is_minus19_through_zero_and_post_is_1_through20():
    records = e163.force_event_records(np.array([205, 250, 390]))
    assert records["force_event_tick"].tolist() == [250]
    assert records["force_event_exclusion_reason"].tolist() == [
        "pre_outside_native_1_2s",
        "included",
        "post_outside_native_1_2s",
    ]
    values = np.arange(200, dtype=float)[:, None]
    summary = e163.force_pre_post(
        values, values * 2, records["force_event_block_index"]
    )
    # tick 250 has block index 49. Pre ticks 231..250 => indices30..49.
    assert summary["event_pre_positive"][0, 0] == np.arange(30, 50).sum()
    assert summary["event_post_positive"][0, 0] == np.arange(50, 70).sum()


def test_phase_bins_preserve_zero_and_missing_bins():
    phase = np.array([0.0, 1 / 12, np.nextafter(1.0, 0), np.nan])
    values = np.array([1.0, 2.0, 3.0, 99.0])
    result = e163.phase_bin_statistics(phase, values, np.ones(4, bool))
    assert result["count"][[0, 1, 11]].tolist() == [1, 1, 1]
    assert np.isnan(result["mean"][2:11]).all()
    assert not result["available"][2]


def test_zero_event_force_summary_is_explicit_nan():
    grid = np.empty((0, 41), dtype=np.int64)
    result = e163.force_pre_post(np.zeros((200, 6)), np.zeros((200, 6)), grid)
    assert result["event_pre_positive"].shape == (0, 6)
    assert np.isnan(result["event_pre_positive_mean"]).all()
    assert np.isnan(result["event_post_negative_median"]).all()


def test_numpy_scalar_json_conversion_is_strict():
    value = e163._jsonable(
        {
            "integer": np.int64(3),
            "float": np.float64(2.5),
            "missing": np.float64(np.nan),
            "flag": np.bool_(True),
        }
    )
    assert value == {"integer": 3, "float": 2.5, "missing": None, "flag": True}
    json.dumps(value, allow_nan=False)


def test_late_publication_failure_revokes_success(tmp_path, monkeypatch):
    (tmp_path / "raw-derived.npz").write_bytes(b"raw")
    original = e163.atomic_json

    def fail_manifest(path, value):
        if path.name == "output-manifest.json":
            raise OSError("generated manifest failure")
        original(path, value)

    monkeypatch.setattr(e163, "atomic_json", fail_manifest)
    result = {"accepted": True, "publication_complete": True, "diagnostic_valid": True}
    output = e163.finalize(tmp_path, result, {}, e163.time.perf_counter())
    assert not output["accepted"]
    assert not output["publication_complete"]
    assert any("publication" in item["stage"] for item in output["publication_errors"])
    if (tmp_path / "result.json").exists():
        persisted = json.loads((tmp_path / "result.json").read_text())
        assert not persisted["accepted"]


@pytest.mark.parametrize("failed_name", ["result.json", "output-manifest.json"])
def test_persistent_second_publication_failure_quarantines_success(
    tmp_path, monkeypatch, failed_name
):
    (tmp_path / "raw-derived.npz").write_bytes(b"raw")
    original = e163.atomic_json
    calls = {failed_name: 0}

    def fail_from_second(path, value):
        if path.name == failed_name:
            calls[failed_name] += 1
            if calls[failed_name] >= 2:
                raise OSError("generated persistent second-write failure")
        original(path, value)

    monkeypatch.setattr(e163, "atomic_json", fail_from_second)
    initial = {
        "accepted": True,
        "publication_complete": True,
        "diagnostic_valid": True,
        "candidate_adopted": True,
        "whole_goal_complete": True,
        "F6_exposed": True,
    }
    returned = e163.finalize(tmp_path, initial, {}, e163.time.perf_counter())
    assert not returned["accepted"]
    assert not returned["candidate_adopted"]
    assert not returned["whole_goal_complete"]
    assert not returned["F6_exposed"]
    receipt = tmp_path / "result.json"
    assert not receipt.exists() or not json.loads(receipt.read_text())["accepted"]
    assert list(tmp_path.glob("*.invalid-*.json"))


def test_unique_atomic_temp_ignores_old_partial_temp(tmp_path):
    (tmp_path / "result.json.tmp").write_text("partial", encoding="utf-8")
    e163.atomic_json(tmp_path / "result.json", {"accepted": False})
    assert json.loads((tmp_path / "result.json").read_text()) == {"accepted": False}
    assert (tmp_path / "result.json.tmp").read_text() == "partial"


def test_complete_generated_analysis_and_postraw_numeric_validation():
    arrays, summary = e163.analyze_loaded(generated_data())
    e163.validate_numeric_contract(arrays)
    assert summary["interruption_event_count"] == 46
    assert arrays["event_e1_phase_window"].shape == (46, 41, 6)
    assert arrays["event_e1_phase_available_window"].shape == (46, 41, 6)
    assert arrays["block_control_tick"].tolist() == list(range(201, 401))
    assert arrays["block_endpoint_time_s"][-1] == pytest.approx(2.0)
    assert arrays["crossing_exclusion_reason"].shape == (46,)
    assert arrays["forcebin_lf_positive_available"].dtype == np.dtype(bool)


def test_numeric_mismatch_can_be_saved_before_it_fails(tmp_path):
    data = generated_data()
    data["native"]["force_forward_bw"] = data["native"]["force_forward_bw"].copy()
    data["native"]["force_forward_bw"][10, 0] += 0.25
    arrays, _ = e163.analyze_loaded(data)
    e163.atomic_npz(tmp_path / "raw-derived.npz", **arrays)
    with pytest.raises(ValueError, match="World-force heading projection"):
        e163.validate_numeric_contract(arrays)
    with np.load(tmp_path / "raw-derived.npz", allow_pickle=False) as raw:
        assert raw["residual_force_projection"][10, 0] == pytest.approx(-0.25)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda d: d["recruitment"]["network_tick"].__setitem__(10, 201),
            "Recruitment clock",
        ),
        (
            lambda d: d["recruitment"]["physics_time_s"].__setitem__(10, 9.0),
            "Recruitment clock",
        ),
        (lambda d: d["trace"].__setitem__("body_hash", "wrong"), "Trace rows/ticks"),
        (
            lambda d: d["adapter"]["motor_ids"].__setitem__(
                0, d["adapter"]["motor_ids"][1]
            ),
            "Motor IDs",
        ),
        (
            lambda d: d["native"]["source_verified"].__setitem__(10, False),
            "Native receipt",
        ),
        (lambda d: d["adapter"]["pool"].__setitem__((0, 0), -1.0), "MN pool mapping"),
    ],
)
def test_generated_clock_body_mapping_and_receipt_fail_closed(mutate, message):
    data = generated_data()
    mutate(data)
    with pytest.raises(ValueError, match=message):
        e163.analyze_loaded(data)


def test_phase_absence_does_not_remove_force_event():
    data = generated_data()
    data["signals"]["rate"][:, e163.SOURCE_COLUMNS[0]] = 0
    arrays, _ = e163.analyze_loaded(data)
    assert arrays["force_event_tick"].tolist() == [260, 300, 340, 380]
    assert np.isnan(arrays["force_event_e1_phase"][:, :, 0]).all()


def test_full_generated_run_publishes_terminal_manifest(tmp_path, monkeypatch):
    args = generated_run_environment(tmp_path, monkeypatch)
    output = tmp_path / "output"
    result = e163.run(*args, output)
    assert result["accepted"]
    assert result["diagnostic_valid"]
    manifest = json.loads((output / "output-manifest.json").read_text())
    expected = {
        path.name for path in output.iterdir() if path.name != "output-manifest.json"
    }
    assert set(manifest) == expected
    for name, receipt in manifest.items():
        assert receipt["sha256"] == e163.sha256(output / name)


def test_full_generated_run_numeric_failure_keeps_raw(tmp_path, monkeypatch):
    args = generated_run_environment(tmp_path, monkeypatch)
    original = e163.analyze_loaded

    def mismatched(data):
        arrays, summary = original(data)
        arrays["residual_target"] = arrays["residual_target"].copy()
        arrays["residual_target"][0, 0] = 1.0
        return arrays, summary

    monkeypatch.setattr(e163, "analyze_loaded", mismatched)
    output = tmp_path / "output"
    result = e163.run(*args, output)
    assert not result["accepted"]
    assert result["primary_error"]["error"] == "Target reconstruction differs"
    with np.load(output / "raw-derived.npz", allow_pickle=False) as raw:
        assert raw["residual_target"][0, 0] == 1.0
    assert not json.loads((output / "result.json").read_text())["accepted"]


def test_traceback_logging_failure_preserves_primary(tmp_path, monkeypatch):
    monkeypatch.setattr(
        e163,
        "verified_inputs",
        lambda *unused: (_ for _ in ()).throw(ValueError("generated primary")),
    )
    monkeypatch.setattr(e163, "threadpool_info", lambda: [{"num_threads": 1}])
    original_write_text = Path.write_text

    def fail_traceback(self, *args, **kwargs):
        if self.name == "traceback.txt":
            raise OSError("generated logging failure")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_traceback)
    freeze = tmp_path / "freeze.json"
    review = tmp_path / "review.json"
    freeze.write_text("{}", encoding="utf-8")
    review.write_text("{}", encoding="utf-8")
    result = e163.run(
        freeze,
        e163.sha256(freeze),
        review,
        e163.sha256(review),
        tmp_path / "output",
    )
    assert result["primary_error"]["error"] == "generated primary"
    assert any(
        error["stage"] == "diagnostic_logging" for error in result["secondary_errors"]
    )
    assert not result["accepted"]


def test_verified_inputs_rejects_hash_before_json(tmp_path):
    freeze = tmp_path / "freeze.json"
    review = tmp_path / "review.json"
    freeze.write_text("invalid", encoding="utf-8")
    review.write_text("invalid", encoding="utf-8")
    with pytest.raises(ValueError, match="freeze SHA"):
        e163.verified_inputs(freeze, "bad", review, "bad")


@pytest.mark.parametrize("late_fault", ["deadline", "bytes"])
def test_final_publication_resource_overrun_revokes_success(
    tmp_path, monkeypatch, late_fault
):
    monkeypatch.setattr(e163, "threadpool_info", lambda: [{"num_threads": 1}])
    now = [0.0]
    monkeypatch.setattr(e163.time, "perf_counter", lambda: now[0])
    if late_fault == "bytes":
        monkeypatch.setattr(e163, "RAW_LIMIT", 4096)
    original = e163.atomic_json
    manifest_calls = [0]

    def late_overrun(path, value):
        original(path, value)
        if path.name == "output-manifest.json":
            manifest_calls[0] += 1
            if manifest_calls[0] == 2:
                if late_fault == "deadline":
                    now[0] = e163.INTERNAL_S + 1
                else:
                    (tmp_path / "late-bytes.bin").write_bytes(b"x" * 9000)

    monkeypatch.setattr(e163, "atomic_json", late_overrun)
    result = e163.finalize(
        tmp_path,
        {"accepted": True, "publication_complete": True, "diagnostic_valid": True},
        {},
        0.0,
    )
    assert not result["accepted"]
    assert not json.loads((tmp_path / "result.json").read_text())["accepted"]
    manifest = json.loads((tmp_path / "output-manifest.json").read_text())
    for name, receipt in manifest.items():
        assert receipt["sha256"] == e163.sha256(tmp_path / name)


def test_failed_terminal_hash_does_not_claim_verified_post_hashes(
    tmp_path, monkeypatch
):
    args = generated_run_environment(tmp_path, monkeypatch)

    def fail_post(*unused):
        raise ValueError("generated frozen hash changed")

    monkeypatch.setattr(e163, "verify_hash_map", fail_post)
    result = e163.run(*args, tmp_path / "output")
    assert not result["accepted"]
    assert result["source_hashes_pre"]
    assert result["source_hashes_post"] == {}
    assert result["frozen_hashes_post_verified"] is False
    assert any(row["stage"] == "post_frozen" for row in result["secondary_errors"])
