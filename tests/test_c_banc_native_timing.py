"""Generated-only E181 tests: no original graph, checkpoint, or FlyBody run."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psutil
import pytest

from tools import optimize_c_banc_m3_native_timing as timing


def generated_slopes():
    slopes = np.zeros((3, 2, 3, 2), np.float64)
    slopes[0, 0, :, 0] = (0.2, 0.3, 0.1)
    slopes[0, 0, :, 1] = (0.1, 0.4, 0.2)
    slopes[0, 1, :, 0] = (0.4, 0.2, 0.1)
    slopes[0, 1, :, 1] = (0.3, 0.2, 0.2)
    slopes[1, 0, 1, :] = (0.5, 0.5)
    slopes[2, 0, 1, :] = (0.5, 0.5)
    return slopes, ("lf", "lm", "rh"), np.array([1, 2], np.int64)


def test_tilt_zero_and_fixed_are_byte_exact():
    slopes, legs, fixed = generated_slopes()
    slopes[0, 1, 0, 1] = -0.0
    zero = timing.tilt_slopes(slopes, timing.TAU, legs, fixed, np.zeros(6, np.float64))
    assert zero.tobytes() == slopes.tobytes()
    tilted = timing.tilt_slopes(
        slopes,
        timing.TAU,
        legs,
        fixed,
        np.array([0.125, 0.125, 0, 0, 0, -0.125], np.float64),
    )
    assert tilted[1:].tobytes() == slopes[1:].tobytes()
    assert tilted[0].tobytes() != slopes[0].tobytes()
    assert np.array_equal(tilted == 0, slopes == 0)
    np.testing.assert_allclose(
        tilted.sum(axis=2), slopes.sum(axis=2), atol=2e-14, rtol=2e-14
    )


def test_tilt_min_surplus_scalar_oracle_and_budget():
    slopes, legs, fixed = generated_slopes()
    theta = np.array([-0.125, 0, 0, 0, 0, 0], np.float64)
    actual = timing.tilt_slopes(slopes, timing.TAU, legs, fixed, theta)
    weights = [(float(tau) / 0.01) ** float(theta[0]) for tau in timing.TAU]
    for motor in range(2):
        low = slopes[0, motor, :, 0]
        high = slopes[0, motor, :, 1]
        common = [min(float(x), float(y)) for x, y in zip(low, high, strict=True)]
        low_extra = [float(x) - x0 for x, x0 in zip(low, common, strict=True)]
        high_extra = [float(x) - x0 for x, x0 in zip(high, common, strict=True)]

        def scalar(parts):
            total = sum(parts)
            weighted = sum(x * w for x, w in zip(parts, weights, strict=True))
            return [
                x * w * total / weighted if weighted else 0.0
                for x, w in zip(parts, weights, strict=True)
            ]

        c = scalar(common)
        a = scalar(low_extra)
        b = scalar(high_extra)
        np.testing.assert_allclose(actual[0, motor, :, 0], np.add(c, a), atol=1e-15)
        np.testing.assert_allclose(actual[0, motor, :, 1], np.add(c, b), atol=1e-15)
    assert np.isclose(
        np.minimum(actual[0, :, :, 0], actual[0, :, :, 1]).sum(),
        np.minimum(slopes[0, :, :, 0], slopes[0, :, :, 1]).sum(),
    )
    assert np.isclose(
        np.maximum(actual[0, :, :, 0], actual[0, :, :, 1]).sum(),
        np.maximum(slopes[0, :, :, 0], slopes[0, :, :, 1]).sum(),
    )


@pytest.mark.parametrize("invalid", ["theta", "tau", "nan", "fixed", "leg"])
def test_tilt_rejects_invalid_contract(invalid):
    slopes, legs, fixed = generated_slopes()
    tau = timing.TAU.copy()
    theta = np.zeros(6, np.float64)
    if invalid == "theta":
        theta[0] = 0.126
    elif invalid == "tau":
        tau[0] = 0.006
    elif invalid == "nan":
        slopes[0, 0, 0, 0] = np.nan
    elif invalid == "fixed":
        fixed = np.array([1, 1], np.int64)
    else:
        legs = ("lf", "lm", "no_leg")
    with pytest.raises(ValueError):
        timing.tilt_slopes(slopes, tau, legs, fixed, theta)


def test_physical_forward_deficit_exact_window():
    window = (np.arange(601) * 0.005 >= 0.5) & (np.arange(601) * 0.005 < 3)
    speed = np.full(601, 0.6)
    speed[window] = np.linspace(0, 1, 500)
    expected = np.maximum(0, 0.5 - speed[window]).mean()
    assert timing.forward_deficit(speed, window) == pytest.approx(expected)
    speed[100] = np.nan
    with pytest.raises(ValueError):
        timing.forward_deficit(speed, window)


def test_same_center_signed_backtracking_and_duplicate_ledger():
    center = (0.0,) * 6
    seen = {center}
    calls = []

    def screen(theta):
        calls.append(theta)
        return {"strict_E141_startup_screen": abs(theta[0]) <= 0.03125}

    plus = timing.preflight_direction(center, "lf", "plus", screen, seen)
    minus = timing.preflight_direction(center, "lf", "minus", screen, seen)
    assert plus["selected_theta"][0] == 0.03125
    assert minus["selected_theta"][0] == -0.03125
    assert [a["status"] for a in plus["attempts"]] == [
        "startup_reject",
        "startup_reject",
        "feasible",
    ]
    assert len(calls) == 6
    repeated = timing.preflight_direction(center, "lf", "plus", screen, seen)
    assert repeated["selected_theta"][0] == 0.015625
    assert [a["status"] for a in repeated["attempts"]] == [
        "duplicate",
        "duplicate",
        "duplicate",
        "feasible",
    ]
    assert len(calls) == 7


def result(objective, forward=1.0, ratio=0.8, slip=0.2, validity="valid_3s_diagnostic"):
    return {
        "validity": validity,
        "startup_pass": True,
        "technical_failure": False,
        "measured_health_stop": False,
        "objective_mm_s": objective,
        "physical_3s": {
            "fixed_initial_heading_net_forward_mm": forward,
            "net_over_path": ratio,
            "native_slip_load_weighted_mm_s": slip,
        },
    }


def test_selection_requires_constraints_and_one_percent_gain():
    center = result(0.3)
    plus_theta = [0.125, 0, 0, 0, 0, 0]
    minus_theta = [-0.125, 0, 0, 0, 0, 0]
    choice = timing.select_axis(
        np.zeros(6),
        center,
        {
            "plus": {"theta": plus_theta, "result": result(0.296)},
            "minus": {"theta": minus_theta, "result": result(0.20, slip=0.21)},
        },
        center,
    )
    assert choice["selected_direction"] == "plus"
    assert choice["directions"]["minus"]["eligible"] is False
    no_gain = timing.select_axis(
        np.zeros(6),
        center,
        {"plus": {"theta": plus_theta, "result": result(0.298)}},
        center,
    )
    assert no_gain["selected_direction"] is None
    incomplete = timing.select_axis(
        np.zeros(6),
        center,
        {"plus": {"theta": plus_theta, "result": result(0.1, validity="incomplete")}},
        center,
    )
    assert incomplete["selected_direction"] is None
    assert no_gain["F3_final"] is False and no_gain["candidate_adopted"] is False


def test_gate_precedes_model_and_output(tmp_path):
    out = tmp_path / "arm"
    with pytest.raises(ValueError, match="Frozen E181"):
        timing.run_arm(None, {}, (0.0,) * 6, out, startup={}, limits=timing.LIMITS)
    assert not out.exists()


def test_cli_help_has_no_original_execution():
    script = Path(timing.__file__)
    completed = subprocess.run(
        [
            str(
                Path(__file__).resolve().parents[1] / ".venv" / "Scripts" / "python.exe"
            ),
            "-X",
            "utf8",
            str(script),
            "--help",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert (
        "--baseline" in completed.stdout
        and "--axis {lf,lm,lh,rf,rm,rh}" in completed.stdout
    )
    assert "--child-request" not in completed.stdout


def test_fake_child_preserves_partial_neural_and_body_raw(monkeypatch, tmp_path):
    class Network:
        tick = 0
        device = "cuda"
        identity = "synthetic-network"

    class Body:
        fault = False
        model_hash = "synthetic-body"

        def __init__(self):
            self.time_s = 0.0
            self.native_sink = None

        def physics_time(self):
            return self.time_s

    class Session:
        def __init__(self):
            self.network = Network()
            self.body = Body()
            self.adapter = type(
                "Adapter",
                (),
                {
                    "identity": "synthetic-adapter",
                    "calibration": {"motor_ids": np.arange(391)},
                },
            )()
            self.control_tick = 0
            self.fault = False
            self.closed = False
            self.calls = 0
            self.monotone_recruitment_patch = {
                "tau": timing.TAU,
                "kappa": np.zeros(391),
                "slopes": np.zeros((96, 391, 3, 2)),
                "membership": np.zeros(391, np.int64),
                "motor_ids": np.arange(391),
                "fixed_muscles": np.arange(19),
            }

        def snapshot(self):
            return {"synthetic": self.control_tick}

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
    monkeypatch.setattr(timing, "_neutral", lambda _saved: None)
    monkeypatch.setattr(timing, "restore_arm", lambda *_args: session)
    monkeypatch.setattr(
        timing.protocol,
        "profiled_construction",
        lambda _label, callback, rows, persist, **_kwargs: (
            rows.append({"returned": True}) or persist() or callback()
        ),
    )
    monkeypatch.setattr(
        timing.StateStore, "save", lambda path, _state: Path(path).mkdir()
    )
    monkeypatch.setattr(
        timing,
        "capture_control",
        lambda current: {
            "time_s": current.body.time_s,
            "control_tick": current.control_tick,
            "fault": None,
        },
    )
    monkeypatch.setattr(
        timing.protocol,
        "install_native_observer",
        lambda body, native: setattr(body, "native_sink", native) or (lambda: None),
    )
    monkeypatch.setattr(
        timing.e150,
        "make_observer",
        lambda records, original: (
            lambda current, *_args: records.append(
                {"control_tick": current.control_tick}
            )
        ),
    )
    monkeypatch.setattr(
        timing.e150,
        "save_native_state",
        lambda path, records: Path(path).write_text(str(len(records))),
    )
    neutral = {
        "monotone_recruitment_patch": {"adapter_identity": "source"},
        "network": {"model_hash": "synthetic-network"},
        "parameters": {},
    }
    gate = timing.ValidatedGate(
        {"schema": timing.SCHEMA, "maximum_raw_bytes": 1 << 30},
        "0" * 64,
        "1" * 64,
        "baseline",
    )
    output = tmp_path / "partial"
    outcome = timing.run_arm(
        type("Graph", (), {"hash": "synthetic-graph", "n": 158706})(),
        neutral,
        (0.0,) * 6,
        output,
        startup={},
        limits=timing.LIMITS,
        gate=gate,
    )
    assert outcome["technical_failure"] is True
    assert outcome["validity"] == "incomplete"
    assert outcome["completed_controls"] == 1
    assert outcome["native_samples"] == 50
    assert outcome["objective_mm_s"] is None
    assert (output / "trace.jsonl").read_text().count("\n") == 2
    assert (output / "native-recruitment-state.npz").read_text() == "2"
    assert (output / "final").is_dir()
    assert (output / "clock-events.json").is_file()
    assert session.closed


def test_windows_venv_redirector_is_live_driver_ancestor():
    driver = psutil.Process(os.getpid())
    request = {"parent_pid": driver.pid, "parent_created_s": driver.create_time()}
    code = (
        "import json,os; from tools import optimize_c_banc_m3_native_timing as e; "
        "r=json.loads(os.environ['E181_SYNTHETIC_ANCESTRY']); "
        "print(json.dumps({'ppid':os.getppid(),'ancestor':e.driver_ancestor(r)}))"
    )
    env = dict(os.environ, E181_SYNTHETIC_ANCESTRY=json.dumps(request))
    completed = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    observation = json.loads(completed.stdout)
    assert observation["ancestor"] is True
    assert (
        timing.driver_ancestor(
            {"parent_pid": driver.pid, "parent_created_s": driver.create_time() - 5}
        )
        is False
    )


def test_supervisor_real_generated_child_persists_started_and_streams(tmp_path):
    request = tmp_path / "baseline-request.json"
    request.write_text('{"synthetic":true}', encoding="utf-8")
    code = "import sys; print('generated stdout', flush=True); print('generated stderr', file=sys.stderr, flush=True)"
    receipt = timing.supervise_command(
        [sys.executable, "-c", code], request, timeout_s=15
    )
    assert receipt["status"] == "completed"
    assert receipt["started"] is True and receipt["completed"] is True
    assert receipt["exit_code"] == 0 and receipt["cost_exact"] is True
    assert timing._attempt_accounting(request)["arms_exact"] == 1
    started = json.loads((tmp_path / "baseline-request-started.json").read_text())
    assert started["pid"] == receipt["pid"] and started["started"] is True
    assert "generated stdout" in (tmp_path / "baseline-request-stdout.txt").read_text()
    assert "generated stderr" in (tmp_path / "baseline-request-stderr.txt").read_text()


def test_supervisor_spawn_failure_receipt_is_provably_unstarted(tmp_path):
    request = tmp_path / "plus-request.json"
    request.write_text('{"synthetic":true}', encoding="utf-8")
    receipt = timing.supervise_command(
        [str(tmp_path / "does-not-exist.exe")], request, timeout_s=2
    )
    assert receipt["status"] == "spawn_error"
    assert receipt["started"] is False and receipt["completed"] is False
    assert receipt["spawn_wall_s"] >= 0
    assert (tmp_path / "plus-request-supervisor.json").exists()
    accounting = timing._attempt_accounting(request)
    assert accounting["arms_exact"] == 0 and accounting["arms_upper_bound"] == 0


def test_supervisor_timeout_kills_generated_descendant_tree(tmp_path):
    request = tmp_path / "minus-request.json"
    request.write_text('{"synthetic":true}', encoding="utf-8")
    code = (
        "import subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
        "print(p.pid,flush=True);time.sleep(30)"
    )
    receipt = timing.supervise_command(
        [sys.executable, "-c", code], request, timeout_s=2
    )
    assert receipt["started"] is True and receipt["timed_out"] is True
    assert receipt["status"] == "timed_out"
    assert receipt["termination"]["alive_pids"] == []
    assert receipt["accounted_wall_s"] >= timing.SUPERVISOR_S
    assert timing._attempt_accounting(request)["cost_exact"] is False
    spawned = int((tmp_path / "minus-request-stdout.txt").read_text().splitlines()[0])
    assert not psutil.pid_exists(spawned)


def test_missing_supervisor_receipt_counts_unknown_not_zero(tmp_path):
    request = tmp_path / "plus-request.json"
    request.write_text('{"synthetic":true}', encoding="utf-8")
    accounting = timing._attempt_accounting(request)
    assert accounting["arms_exact"] is None
    assert accounting["arms_upper_bound"] == 1
    assert accounting["accounted_wall_s"] == timing.SUPERVISOR_S


def test_driver_child_gate_failure_still_has_live_and_final_receipts(tmp_path):
    freeze = tmp_path / "synthetic-freeze.json"
    review = tmp_path / "synthetic-review.json"
    freeze.write_text("{}", encoding="utf-8")
    review.write_text("{}", encoding="utf-8")
    args = SimpleNamespace(
        freeze=freeze,
        freeze_sha256="0" * 64,
        source_review=review,
        source_review_sha256="1" * 64,
    )
    gate = timing.ValidatedGate(
        {"schema": timing.SCHEMA}, "0" * 64, "1" * 64, "baseline"
    )
    request = {
        "schema": "flylab.m3.e181.child-request.v1",
        "mode": "baseline",
        "arm": "baseline",
        "theta": [0.0] * 6,
        "parent_pid": os.getpid(),
        "parent_created_s": psutil.Process(os.getpid()).create_time(),
        "freeze_sha256": gate.freeze_sha256,
        "source_review_sha256": gate.source_review_sha256,
    }
    path = tmp_path / "baseline-request.json"
    with pytest.raises(timing.ResourceGuard):
        timing._launch_child(args, gate, request, path, began=0.0)
    final = json.loads((tmp_path / "baseline-request-supervisor.json").read_text())
    assert final["status"] == "child_nonzero"
    assert final["started"] is True and final["completed"] is False
    assert final["accounted_wall_s"] >= timing.SUPERVISOR_S
    assert (tmp_path / "baseline-request-started.json").is_file()
    assert (tmp_path / "baseline-request-stderr.txt").is_file()
    assert path.is_file()
