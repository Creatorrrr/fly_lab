"""Generated E207 inverse and two-call fail-closed mock-device checks only."""

import json

import numpy as np
import pytest

from tools import verify_c_banc_m3_threshold_initialization as e207


def generated_parameters():
    return {"coefficient": np.full(391, .1),
        "threshold": np.ones(391), "initial_adaptation": np.zeros(391),
        "initial_rate": np.zeros(391), "cap": np.full(391, 100.),
        "tau": np.full(391, .03), "adaptation_gain": np.full(391, 6.),
        "inverse_adaptation_tau": np.full(391, 3.),
        "motor_ids": np.arange(391)}


def generated_response():
    rate = np.ones((2001, 391), np.float64)
    rate[:, :178] = 0
    return {"rate": rate, "adaptation": rate.copy(),
        "rate_sensitivity": np.repeat(rate[:, :, None], 7, axis=2),
        "adaptation_sensitivity": np.repeat(rate[:, :, None], 7, axis=2)}


def test_mask_bias_inverse_range_and_initial_state_immutability():
    response = generated_response()
    free = np.arange(343)
    weights = np.zeros((391, 7))
    eligible = e207.silent_free_mask(response, weights, free, np.arange(178))
    assert int(np.count_nonzero(eligible)) == 178
    p = generated_parameters()
    before_r, before_z = p["initial_rate"].copy(), p["initial_adaptation"].copy()
    original_weights = weights.copy()
    seeded = e207.seed_bias(np.ones(391), p, weights, eligible)
    np.testing.assert_allclose(seeded["phi0"], .01, atol=1e-10, rtol=1e-10)
    np.testing.assert_array_equal(seeded["weights"][~eligible], weights[~eligible])
    np.testing.assert_array_equal(seeded["weights"][eligible, 1:], 0)
    np.testing.assert_array_equal(weights, original_weights)
    np.testing.assert_array_equal(p["initial_rate"], before_r)
    np.testing.assert_array_equal(p["initial_adaptation"], before_z)
    assert np.any(seeded["weights"][eligible, 0] != 0)
    wrong = generated_response()
    wrong["adaptation"][0, 0] = 1
    with pytest.raises(ValueError, match="eligible columns"):
        e207.silent_free_mask(wrong, weights, free, np.arange(178))
    bad_cap = generated_parameters()
    bad_cap["cap"][:178] = .005
    with pytest.raises(ValueError, match="cap"):
        e207.seed_bias(np.ones(391), bad_cap, weights, eligible)
    bad_range = generated_parameters()
    bad_range["threshold"][:178] = 100
    with pytest.raises(ValueError, match="tanh domain"):
        e207.seed_bias(np.ones(391), bad_range, weights, eligible)


def test_two_call_mock_fidelity_nonfinite_and_unchanged_tamper(
        tmp_path, monkeypatch):
    names = ("current", "parameters", "start", "trace", "targets", "mask",
        "e205_response", "e205_decoded", "e205_selected_raw",
        "e205_preconditioner", "zero_gradient_audit")
    monkeypatch.setattr(e207, "verify_gate", lambda *args: (
        {"paths": {name: name for name in names}, "input_sha256": {},
         "prior_cumulative_s": 1.}, {"source_sha256": {}}))
    monkeypatch.setattr(e207, "repository_path", lambda name, **kwargs: tmp_path / name)
    p = generated_parameters()
    response = generated_response()
    weights = np.zeros((391, 7))
    decoded = {"bank": np.zeros((2001, 391, 3)),
        "activation": np.zeros((2001, 96)), "command": np.zeros((2001, 42)),
        "tendon": np.zeros((2001, 6))}
    with (tmp_path / "parameters").open("wb") as stream:
        np.savez_compressed(stream, **p)
    with (tmp_path / "mask").open("wb") as stream:
        np.savez_compressed(stream, free_columns=np.arange(343),
                            fixed_columns=np.arange(343, 391))
    with (tmp_path / "e205_response").open("wb") as stream:
        np.savez_compressed(stream, **response, weights=weights)
    with (tmp_path / "e205_decoded").open("wb") as stream:
        np.savez_compressed(stream, **decoded)
    with (tmp_path / "e205_selected_raw").open("wb") as stream:
        np.savez_compressed(stream, weights=weights)
    with (tmp_path / "e205_preconditioner").open("wb") as stream:
        np.savez_compressed(stream, D=np.zeros((2001, 391, 7)),
                            initial_weights=weights)
    (tmp_path / "zero_gradient_audit").write_text(json.dumps({
        "zero_free_count": 178, "free_motor_columns": list(range(178))}))
    (tmp_path / "trace").write_text("{}")
    start = {"adapter": {"calibration": {"ltm": np.array([15, 30, 46, 63, 78, 94])}}}
    from flylab.c.storage import StateStore
    monkeypatch.setattr(StateStore, "load", lambda path: start)
    monkeypatch.setattr(e207, "load_targets", lambda *args: {
        "indices": np.arange(100, 1924), "baseline": {"command_rms": 1.}})
    original_load = np.load

    class FakeCurrent:
        shape = (160000, 391)
        dtype = np.dtype(np.float32)

        def __getitem__(self, key):
            assert key == 0
            return np.ones(391, np.float32)

    monkeypatch.setattr(e207.np, "load", lambda path, *args, **kwargs: (
        FakeCurrent() if str(path) == str(tmp_path / "current")
        else original_load(path, *args, **kwargs)))

    class FakeDecoder:
        def __init__(self, source):
            pass

        def forward(self, rates):
            return {key: value.copy() for key, value in decoded.items()}

    monkeypatch.setattr(e207, "DecoderAdjoint", FakeDecoder)
    monkeypatch.setattr(e207, "command_diagonal_scale", lambda *args: {
        "D": np.zeros((2001, 391, 7)),
        "h": np.full((343, 7), 1e-5 + 2e-4 / 2401),
        "scale": np.ones((343, 7)), "exponent": np.zeros((343, 7), int),
        "free_columns": np.arange(343)})
    monkeypatch.setattr(e207, "direct_metrics", lambda *args: {"command_rms": 1.})
    monkeypatch.setattr(e207, "feasible", lambda *args: False)
    behavior = {"kind": "normal"}

    class FakeDevice:
        def __init__(self, current, parameters):
            self.stats = {"launch_requested": 0, "launch_synchronized": 0,
                "launch_completion_unknown": 0, "conditional_ticks_completed": 0,
                "stage_inputs_consumed": 0, "current_uploads": 1}

        def evaluate(self, full_weights):
            self.stats["launch_requested"] += 1
            self.stats["launch_synchronized"] += 1
            self.stats["conditional_ticks_completed"] += 40000
            self.stats["stage_inputs_consumed"] += 160000
            result = {key: value.copy() for key, value in response.items()}
            if self.stats["launch_requested"] == 1 and behavior["kind"] == "baseline_mismatch":
                result["rate"][0, 0] = .1
            if self.stats["launch_requested"] == 2:
                result["rate"][:, :178] = .001
                if behavior["kind"] == "nonfinite":
                    result["adaptation"][1, 0] = np.nan
                if behavior["kind"] == "unchanged_tamper":
                    result["rate"][2, 200] += .1
            return result

    monkeypatch.setattr(e207, "PersistentLocalResponse", FakeDevice)
    for kind, expected_calls, error in (
            ("baseline_mismatch", 1, "bit-exact E205"),
            ("nonfinite", 2, "adaptation"),
            ("unchanged_tamper", 2, "unchanged MN"),
            ("normal", 2, None)):
        behavior["kind"] = kind
        output = tmp_path / kind
        if error:
            with pytest.raises(ValueError, match=error):
                e207.run("unused", "unused", "unused", "unused", output)
            assert json.loads((output / "result.json").read_text())[
                "validity"] == "incomplete_or_invalid"
        else:
            result = e207.run("unused", "unused", "unused", "unused", output)
            assert result["recruited_count"] == 178
        ledger = json.loads((output / "ledger.json").read_text())
        assert ledger["device_calls_requested"] == expected_calls
        assert ledger["device_calls_returned"] == expected_calls
        assert ledger["device"]["current_uploads"] == 1
        assert (output / "baseline-response.npz").is_file()
        assert (output / "seeded-response.npz").is_file() == (expected_calls == 2)
        if kind == "nonfinite":
            with original_load(output / "seeded-response.npz") as saved:
                assert np.isnan(saved["adaptation"][1, 0])
