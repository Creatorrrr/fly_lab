"""Synthetic tests only; no checkpoint, graph, neural or FlyBody construction."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools import probe_c_banc_m3_native_objective as pilot


def generated():
    pool = np.array([[0.5, 0.5, 0], [0, 0, 1]], np.float64)
    canonical = np.zeros((2, 3, 3, 2), np.float64)
    canonical[:, :, 1, :] = pool[:, :, None]
    original = canonical.copy()
    original[0, :2, 1, 0] *= 0.8
    original[0, :2, 1, 1] *= 1.2
    original[0, :2, 0, :] = 0.05
    return original, pool, np.array([1], np.int64), canonical


def test_alpha_zero_is_bit_exact_and_independent():
    original, pool, fixed, _ = generated()
    original[0, 0, 0, 0] = -0.0
    result = pilot.blend_slopes(original, pool, fixed, 0.0)
    assert result.tobytes() == original.tobytes()
    result[0, 0, 0, 0] = 17
    assert original[0, 0, 0, 0] == 0


def test_canonical_convex_blend_and_fixed_rows():
    original, pool, fixed, canonical = generated()
    result = pilot.blend_slopes(original, pool, fixed, 0.01)
    expected = 0.99 * original + 0.01 * canonical
    expected[fixed] = original[fixed]
    np.testing.assert_array_equal(result, expected)
    assert result[fixed].tobytes() == original[fixed].tobytes()
    assert np.all(result >= 0)


@pytest.mark.parametrize("change", ["alpha", "dtype", "pool", "fixed", "nan"])
def test_blend_rejects_invalid_inputs(change):
    original, pool, fixed, _ = generated()
    alpha = 0.01
    if change == "alpha":
        alpha = 0.02
    elif change == "dtype":
        original = original.astype(np.float32)
    elif change == "pool":
        pool = pool[:, :2]
    elif change == "fixed":
        fixed = np.array([1, 1])
    else:
        original[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        pilot.blend_slopes(original, pool, fixed, alpha)


def test_partial_and_measured_health_outcomes_are_distinct():
    partial = pilot.summarize_arm(
        "execution_error", False, 31, 1550, 0.155, 0.155, True, None, [], []
    )
    assert partial["validity"] == "incomplete"
    assert partial["technical_failure"] is True
    health = pilot.summarize_arm(
        "native_health_guard", True, 31, 1551, 0.155, 0.1551, True, None, [], []
    )
    assert health["validity"] == "measured_native_health_stop"
    assert health["technical_failure"] is False
    assert pilot.compare_arms(health, health)["paired_valid"] is False
    assert health["F3_final"] is False


def test_full_three_second_diagnostic_still_not_final_success():
    outcome = pilot.summarize_arm(
        "duration", False, 600, 30000, 3.0, 3.0, True, True, [], []
    )
    assert outcome["validity"] == "valid_3s_diagnostic"
    pair = pilot.compare_arms(outcome, outcome)
    assert pair["paired_valid"] is True
    assert pair["candidate_adopted"] is False
    assert not pair["F3_final"] and not pair["F4_final"] and not pair["F5_final"]


def test_pair_requires_positive_denominators_and_supported_slip():
    baseline = pilot.summarize_arm(
        "duration", False, 600, 30000, 3.0, 3.0, True, True, [], []
    )
    candidate = dict(baseline)
    candidate["physical_3s"] = {
        "fixed_initial_heading_net_forward_mm": 1.0,
        "net_over_path": 0.8,
        "native_slip_load_weighted_mm_s": 0.2,
        "longest_forward_bout_s": 0.4,
        "forward_eligible_complete_cycles": dict.fromkeys(pilot.protocol.LEGS, 1),
    }
    baseline["physical_3s"] = dict(candidate["physical_3s"])
    assert pilot.compare_arms(baseline, candidate)["development_entry"] is True
    baseline["physical_3s"]["fixed_initial_heading_net_forward_mm"] = 0
    assert pilot.compare_arms(baseline, candidate)["development_entry"] is False
    baseline["physical_3s"]["fixed_initial_heading_net_forward_mm"] = 1
    baseline["physical_3s"]["native_slip_load_weighted_mm_s"] = None
    assert pilot.compare_arms(baseline, candidate)["development_entry"] is False


def test_gate_failure_precedes_model_and_output(monkeypatch, tmp_path):
    # A direct call cannot construct a model or create an output directory.
    def forbidden(*args, **kwargs):
        raise AssertionError("model entered before gate")

    monkeypatch.setattr(pilot, "restore_arm", forbidden)
    output = tmp_path / "original"
    with pytest.raises(ValueError, match="Frozen E180"):
        pilot.run_arm(None, {}, 0.0, output, startup={}, limits=pilot.LIMITS)
    assert not output.exists()


def test_cli_help_enters_no_model():
    script = Path(pilot.__file__)
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(script), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--arm {baseline,candidate}" in result.stdout


@pytest.mark.parametrize("failure_stage", ["advance", "readout"])
def test_fake_runner_preserves_anchor_and_failure_raw(
    monkeypatch, tmp_path, failure_stage
):
    class Network:
        tick = 0
        device = "cuda"
        identity = "net"

    class Body:
        fault = False
        model_hash = "body"

        def __init__(self):
            self.time_s = 0.0
            self.native_sink = None

        def physics_time(self):
            return self.time_s

    class Adapter:
        identity = "adapter"

        def __init__(self):
            self.calibration = {"motor_ids": np.arange(391)}

    class Session:
        def __init__(self):
            self.network = Network()
            self.body = Body()
            self.adapter = Adapter()
            self.control_tick = 0
            self.fault = False
            self.monotone_recruitment_patch = {
                "tau": np.zeros(3),
                "kappa": np.zeros(391),
                "slopes": np.zeros((96, 391, 3, 2)),
                "membership": np.zeros(391, np.int64),
                "motor_ids": np.arange(391),
                "fixed_muscles": np.arange(19),
            }
            self.closed = False
            self.calls = 0

        def snapshot(self):
            return {"schema": "fake", "tick": self.control_tick}

        def advance(self, _controls):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("synthetic advance failure")
            self.network.tick += 20
            self.control_tick += 1
            self.body.time_s += 0.005
            self.body.native_sink.extend(
                (
                    index * 0.0001,
                    np.zeros(6),
                    np.full(6, np.nan),
                    np.zeros(6, bool),
                    False,
                    1.0,
                )
                for index in range(1, 51)
            )

        def close(self):
            self.closed = True

    session = Session()
    states = []
    monkeypatch.setattr(pilot, "_neutral", lambda _saved: None)
    monkeypatch.setattr(pilot, "restore_arm", lambda *_args: session)
    monkeypatch.setattr(
        pilot.protocol,
        "profiled_construction",
        lambda _label, callback, rows, persist, **_kwargs: (
            rows.append({"returned": True}) or persist() or callback()
        ),
    )
    monkeypatch.setattr(
        pilot.StateStore, "save", lambda path, _state: Path(path).mkdir()
    )
    monkeypatch.setattr(
        pilot,
        "capture_control",
        lambda _session: {
            "time_s": _session.body.time_s,
            "control_tick": _session.control_tick,
            "fault": None,
        },
    )
    monkeypatch.setattr(
        pilot.protocol,
        "install_native_observer",
        lambda _body, _native: setattr(_body, "native_sink", _native) or (lambda: None),
    )

    def make_observer(records, original):
        def observe(current, *_args):
            records.append({"control_tick": current.control_tick})
            states.append(current.control_tick)
            if failure_stage == "readout" and current.control_tick == 1:
                raise RuntimeError("synthetic readout failure")

        return observe

    monkeypatch.setattr(pilot.e150, "make_observer", make_observer)
    monkeypatch.setattr(
        pilot.e150,
        "save_native_state",
        lambda path, records: Path(path).write_text(str(len(records))),
    )
    neutral = {
        "monotone_recruitment_patch": {"adapter_identity": "src"},
        "network": {"model_hash": "net"},
        "parameters": {},
    }
    gate = pilot.ValidatedGate(
        {"schema": pilot.SCHEMA, "maximum_raw_bytes": 1 << 30}, "0" * 64, "baseline"
    )
    out = tmp_path / "failure"
    result = pilot.run_arm(
        type("Graph", (), {"hash": "graph", "n": 158706})(),
        neutral,
        0.0,
        out,
        startup={},
        limits=pilot.LIMITS,
        gate=gate,
    )
    assert result["validity"] == "incomplete"
    assert (
        result["primary_error"] == f"RuntimeError('synthetic {failure_stage} failure')"
    )
    assert (out / "trace.jsonl").read_text().count("\n") == 2
    assert (out / "native-recruitment-state.npz").read_text() == "2"
    assert result["completed_controls"] == 1
    assert result["native_samples"] == 50
    assert result["actual_neural_s"] == pytest.approx(0.005)
    assert (out / "final").is_dir()
    assert (out / "clock-events.json").exists()
    assert session.closed
