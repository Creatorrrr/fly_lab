"""E334 generated CPU qualification; no original numerical payload loads."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import verify_c_banc_m3_smooth_constrained_secant as subject
from tools.banc_m3_checkpointed_rollout import CheckpointedControlRollout
from tools.banc_m3_smooth_temporal import SmoothTemporalObjective
from tools.verify_c_banc_m3_temporal_direction import patch_weights

torch.set_num_threads(1)


def secant_inputs():
    q0, t0 = np.zeros((99, 42)), np.zeros((99, 6))
    q1 = q0.copy()
    q1[0, 0] = .08
    return q0, t0, q1, t0.copy(), q0.copy(), t0.copy(), np.array([.8, 1.2, 1], np.float32).astype(np.float64)


def generated_fixture(*, failure=None, checkpoint=300):
    """Real public N391 checkpoint fixture; synthetic fault injection is explicit."""
    nodes = 391
    dims = subject.TestDimensions(nodes, 3)
    phi = 2 * np.pi * np.arange(128) / 128
    curve = np.stack([np.stack([.1 * np.sin(phi + j * .2) for j in range(7)], -1) for _ in range(6)])
    objective = SmoothTemporalObjective(curve, np.zeros((99, 42)), np.zeros((99, 6)))
    native = {"csr": {"data": np.array([.25, -.5, .125, -0.], np.float32), "indices": np.arange(4, dtype=np.int32),
                      "indptr": np.r_[np.arange(5, dtype=np.int32), np.full(387, 4, np.int32)]},
              "mask": {"positions": np.arange(3, dtype=np.int64), "edge_parameters": np.arange(3, dtype=np.int64)}}
    state = {"control_tick": 0, "network": {"tick": 0, "rate": np.zeros(nodes, np.float32),
             "adaptation": np.zeros(nodes, np.float32), "output_mask": np.ones(nodes, np.float32)},
             "adapter": {"bank": np.zeros((391, 3)), "original": {"activation": np.zeros(96)}}}
    q0, _t0, q1, t1, tq, tt, oldtheta = secant_inputs()
    data = {"native": native, "start": state, "whole100": state, "final": state,
            "drives": np.full((300, nodes), .2, np.float32), "calibration": {"motor_ids": np.arange(391, dtype=np.int64)},
            "raw": {"command": np.vstack((q0, np.zeros((202, 42)))), "tendon": np.zeros((301, 6))},
            "startup": {"target_command": tq, "target_tendon": tt}, "secant_q1": q1, "secant_t1": t1,
            "theta_minus": oldtheta, "e332_neutral_terms": {"training_loss": 10000.0}}
    class Neural:
        n = nodes
        device = torch.device("cpu")
        dt = float(np.float32(.00025))
        def __init__(self):
            self.operator = SimpleNamespace(group_count=3)
            self.tau = torch.full((nodes,), .01, dtype=torch.float32)
        def prepare_efficacy(self, theta):
            return theta
        def step(self, r, z, drive, theta, mask):
            return (drive + (theta[0] - 1).float() * .001) * mask, z
    class Decoder:
        device = torch.device("cpu")
        def step(self, motor, bank, original, *, dt):
            assert dt == .005
            return {"q": torch.ones(42, dtype=torch.float64) * motor[0].double() * .0001,
                    "tendon": torch.zeros(6, dtype=torch.float64), "bank": bank + .001,
                    "original_activation": original + .001}
    class Deps:
        def __init__(self):
            self.calls, self.confirm_calls = [], []
            self.rollout = CheckpointedControlRollout(Neural(), Decoder(), data["calibration"]["motor_ids"],
                steps_per_control=20, control_dt=.005, checkpoint_controls=checkpoint, max_controls=300)
            approved_tau = self.rollout.neural.tau.numpy().copy()
            data["test_tensor_specs"] = {"neural.tau": (self.rollout.neural, "tau", approved_tau)}
            original_run = self.rollout.run
            def run(r, z, bank, original, drives, theta, mask):
                self.calls.append(tuple(subject.Dependencies.host(x) for x in (r, z, bank, original, drives, theta, mask)))
                output = original_run(r, z, bank, original, drives, theta, mask)
                confirm = output["receipt"].confirm
                def once():
                    self.confirm_calls.append(1)
                    if failure == "confirm":
                        raise RuntimeError("synthetic confirm failure")
                    return confirm()
                output["receipt"].confirm = once
                if failure == "nonfinite":
                    output["q"][0, 0] = torch.nan
                if failure == "shape":
                    output["tendon"] = output["tendon"][:1]
                return output
            self.rollout.run = run
        def load(self, freeze, root):
            return data
        def build(self, data, device):
            if failure == "changed_tensor":
                self.rollout.neural.tau[0] = .02
            tensor = lambda x: torch.tensor(x)
            initial = tuple(tensor(x) for x in (state["network"]["rate"], state["network"]["adaptation"],
                state["adapter"]["bank"], state["adapter"]["original"]["activation"], data["drives"], state["network"]["output_mask"]))
            return self.rollout, objective, initial
        @staticmethod
        def host(value):
            if failure == "acquisition" and tuple(value.shape) == (300, 42):
                raise RuntimeError("synthetic acquisition failure")
            return subject.Dependencies.host(value)
    return data, Deps(), dims, objective


def test_closed_form_midpoint_and_independent_bisection36_cases():
    rng = np.random.default_rng(334)
    for _ in range(36):
        q0, t0, _, _, tq, tt, theta = secant_inputs()
        q0 += rng.normal(size=q0.shape) * .002
        t0 += rng.normal(size=t0.shape) * .001
        q1 = q0 + rng.normal(size=q0.shape) * rng.uniform(.02, .2)
        t1 = t0 + rng.normal(size=t0.shape) * rng.uniform(.01, .08)
        info, candidate, _ = subject.secant_candidate(q0, t0, q1, t1, tq, tt, theta)
        def feasible(alpha, q0=q0, q1=q1, t0=t0, t1=t1, tq=tq, tt=tt):
            q, t = q0 + alpha * (q1 - q0) - tq, t0 + alpha * (t1 - t0) - tt
            return np.sqrt(np.mean(q ** 2)) <= .018 and np.max(np.abs(q)) <= .04 and np.sqrt(np.mean(t ** 2)) <= .009 and np.max(np.abs(t)) <= .025
        low, high = 0., 1.
        if feasible(1):
            low = 1.
        else:
            for _ in range(60):
                mid = (low + high) / 2
                if feasible(mid):
                    low = mid
                else:
                    high = mid
        assert info["alpha_star"] == pytest.approx(low / 2, abs=1e-12)
        assert np.array_equal(candidate, (1 + low / 2 * (theta - 1)).astype(np.float32).astype(np.float64))


def test_stationary_coordinates_whole_zero_and_exact_theta_pin():
    values = secant_inputs()
    info, theta, _ = subject.secant_candidate(*values)
    assert info["intersection"] == [0., .5]
    assert info["alpha_star"] == .25
    expected = {key: info[key] for key in ("alpha_star", "intersection", "secant_intervals", "theta_star64_bytes_sha256")}
    assert subject.secant_candidate(*values, expected=expected)[1].tobytes() == theta.tobytes()
    wrong = {**expected, "theta_star64_bytes_sha256": "0" * 64}
    with pytest.raises(ValueError, match="no reselection"):
        subject.secant_candidate(*values, expected=wrong)
    stationary = list(values)
    stationary[2], stationary[3] = values[0].copy(), values[1].copy()
    assert subject.secant_candidate(*stationary)[0]["reason"] == "no_secant_information"
    unity = list(values)
    unity[-1] = np.ones(3)
    assert subject.secant_candidate(*unity)[0]["reason"] == "rounded_unity_candidate"


@pytest.mark.parametrize("failure", ["dtype", "shape", "nonfinite", "strict", "overflow", "underflow"])
def test_secant_wrapper_rejects_degenerate_protocol_inputs(failure):
    values = list(secant_inputs())
    if failure == "dtype":
        values[0] = values[0].astype(np.float32)
    elif failure == "shape":
        values[0] = values[0][:98]
    elif failure == "nonfinite":
        values[2][0, 0] = np.nan
    elif failure == "strict":
        values[0][0, 0] = .04
    elif failure == "overflow":
        values[2][0, 0] = 1e308
    else:
        values[2][:] = 1e-300
    with pytest.raises((ValueError, FloatingPointError)):
        subject.secant_candidate(*values)


def test_selective_direction_loader_refuses_any_gradient_access(monkeypatch):
    touched = []
    class Archive:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def __getitem__(self, key):
            touched.append(key)
            assert key == "theta_minus"
            return np.ones(3)
    def load(path, *, allow_pickle):
        assert allow_pickle is False
        return Archive()
    monkeypatch.setattr(subject.np, "load", load)
    assert set(subject.load_selected("generated-direction", ("theta_minus",))) == {"theta_minus"}
    assert touched == ["theta_minus"]


def test_inherited_digest_cannot_be_overwritten():
    pins = {"old": "a"}
    with pytest.raises(ValueError, match="overwrite"):
        subject._merge(pins, {"old": "b"})
    assert pins == {"old": "a"}


def test_native_once_round_gain1_signzeros_and_full_palette():
    data, _, _, _ = generated_fixture()
    theta = np.array([.95, 1., 1.05], np.float32).astype(np.float64)
    csr, actual, info = patch_weights(data["native"], theta)
    original = data["native"]["csr"]["data"]
    expected = (original[:3].astype(np.float64) * theta).astype(np.float32)
    assert actual.tobytes() == expected.tobytes()
    assert csr["data"][1].tobytes() == original[1].tobytes()
    assert csr["data"][3].tobytes() == original[3].tobytes()
    assert np.array_equal(np.signbit(actual), np.signbit(original[:3]))
    assert info["full_csr_bit_palette_count"] == len(np.unique(csr["data"].view(np.uint32)))
    big = dict(data["native"])
    big["csr"] = {"data": np.arange(1, 17001, dtype=np.float32), "indices": np.zeros(17000, np.int32),
                  "indptr": np.r_[0, np.full(391, 17000, np.int32)]}
    assert patch_weights(big, theta)[2]["eligible"] is False


def test_real_checkpoint1_and300_bytes_receipts_and_single_forward():
    data, deps, dims, _ = generated_fixture()
    snapshots = {}
    partial, receipts = subject.compute(data, dependencies=deps, device="cpu", test_dimensions=dims,
        checkpoint=lambda name, raw: snapshots.update({name: raw}))
    assert {name + ".npz" for name in snapshots} == set(subject.RAW_FILES)
    assert len(deps.calls) == len(deps.confirm_calls) == 1
    assert partial["scientific_supported"] and partial["external_state_reinjection"] == 0
    subject.check_receipt(receipts["continuation-star"])
    assert all(np.asarray(value).shape == () for value in snapshots["continuation-metrics"].values())
    data2, deps2, _, _ = generated_fixture(checkpoint=1)
    rollout, _, initial = deps2.build(data2, "cpu")
    r, z, bank, original, drives, mask = initial
    theta = torch.tensor(subject.secant_candidate(*secant_inputs())[1])
    with torch.no_grad():
        output = rollout.run(r, z, bank, original, drives, theta, mask)
    output["receipt"].confirm()
    for key in ("q", "tendon", "rate", "adaptation", "bank", "original_activation"):
        assert subject.Dependencies.host(output[key]).tobytes() == snapshots["continuation-forward"][key].tobytes()
    assert deps.calls[0][2].tobytes() == data["start"]["adapter"]["bank"].tobytes()
    assert deps.calls[0][3].tobytes() == data["start"]["adapter"]["original"]["activation"].tobytes()


def test_actual_drive_tensor_and_each_host_chunk_dtype_shape_are_observed():
    data, deps, _, objective = generated_fixture()
    rollout, _, initial = deps.build(data, "cpu")
    proof, _ = subject.actual_input_proof(data, rollout, objective, initial, test_only=True)
    assert proof["drives_tensor"] == {"shape": [300, 391], "dtype": "torch.float32"}
    assert len(proof["drives_chunks"]) == 12
    assert [chunk["start_control"] for chunk in proof["drives_chunks"]] == list(range(1, 301, 25))
    assert all(chunk["shape"] == [25, 391] and chunk["dtype"] == "<f4" for chunk in proof["drives_chunks"])
    wrong = list(initial)
    wrong[4] = initial[4].view(torch.int32)
    assert wrong[4].numpy().tobytes() == data["drives"].tobytes()
    with pytest.raises(ValueError, match="dtype/shape differs: drives"):
        subject.actual_input_proof(data, rollout, objective, tuple(wrong), test_only=True)
    seen = []
    def wrong_host_dtype(value):
        observed = subject.Dependencies.host(value)
        if tuple(value.shape) == (25, 391):
            seen.append(1)
            if len(seen) == 2:
                return observed.view(np.int32)
        return observed
    with pytest.raises(ValueError, match="actual drive host chunk"):
        subject.actual_input_proof(data, rollout, objective, initial, test_only=True, host=wrong_host_dtype)
    assert len(seen) == 2
    def wrong_host_shape(value):
        observed = subject.Dependencies.host(value)
        return observed.reshape(5, 1955) if tuple(value.shape) == (25, 391) else observed
    with pytest.raises(ValueError, match="actual drive host chunk"):
        subject.actual_input_proof(data, rollout, objective, initial, test_only=True, host=wrong_host_shape)


@pytest.mark.parametrize("failure", ["nonfinite", "shape", "confirm", "acquisition", "changed_tensor"])
def test_failure_raw_before_validation_confirm_once_and_actual_tensor_check(failure):
    data, deps, dims, _ = generated_fixture(failure=failure)
    snapshots = {}
    with pytest.raises((ValueError, RuntimeError)) as caught:
        subject.compute(data, dependencies=deps, device="cpu", test_dimensions=dims,
            checkpoint=lambda name, raw: snapshots.update({name: raw}))
    partial = caught.value.partial
    assert len(deps.confirm_calls) <= 1
    if failure == "changed_tensor":
        assert not deps.calls and "continuation-forward" not in snapshots
    else:
        assert "continuation-forward" in snapshots
    if failure == "nonfinite":
        assert np.isnan(snapshots["continuation-forward"]["q"]).any()
    if failure == "acquisition":
        assert "q" not in snapshots["continuation-forward"] and partial["acquisition"]["q"] == "unavailable"
        assert "tendon" in snapshots["continuation-forward"]


def test_acquired_event_survives_journal_and_save_failure():
    data, deps, dims, _ = generated_fixture()
    saved = {}
    def boundary(event):
        if event["event"] == "array_acquired" and event["key"] == "q":
            raise OSError("first acquired-journal failure")
    def save(name, raw):
        saved[name] = raw
        if name == "continuation-forward":
            raise OSError("second raw-sink failure")
    with pytest.raises(OSError, match="first acquired-journal failure") as caught:
        subject.compute(data, dependencies=deps, device="cpu", test_dimensions=dims, checkpoint=save, on_boundary=boundary)
    assert caught.value.partial["acquisition"]["q"] == "acquired"
    assert "q" in saved["continuation-forward"] and len(deps.confirm_calls) == 1
    assert any(row["stage"] == "raw_save" for row in caught.value.partial["secondary_errors"])


def run_fixture(tmp_path, *, clock=None, writer=None, emergency=None, repair=None, reader=None, compute_fn=None):
    _data, deps, dims, _ = generated_fixture()
    path = tmp_path / "pin.txt"
    path.write_text("generated", encoding="utf8")
    frozen = {"source_sha256": {"pin.txt": subject.sha256(path)}, "input_sha256": {"pin.txt": subject.sha256(path)}}
    kwargs = {key: value for key, value in (("clock", clock), ("writer", writer), ("emergency", emergency),
              ("repair", repair), ("reader", reader), ("compute_fn", compute_fn)) if value is not None}
    output = tmp_path / "output"
    result = subject.run(path, subject.sha256(path), path, subject.sha256(path), output, root=tmp_path,
        dependencies=deps, test_dimensions=dims, device="cpu", gate_fn=lambda *args, **kwargs: frozen,
        meter=lambda: {"host_rss_bytes": 0}, **kwargs)
    return result, output


def test_normal_raw_uncompressed_unique7_and_terminal_coherence(tmp_path):
    import zipfile
    result, output = run_fixture(tmp_path)
    assert result["run_valid"] and result["effective_neural_qualification"]
    assert {path.name for path in output.glob("*.npz")} == set(subject.RAW_FILES)
    for path in output.glob("*.npz"):
        with zipfile.ZipFile(path) as archive:
            assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
    disk = subject.read(output / "result.json")
    ledger = subject.read(output / "advance-ledger.json")
    assert disk == result and ledger["first_error"] is None and ledger["run_valid"]


@pytest.mark.parametrize("failure", ["postcommit", "readback", "output"])
def test_publication_late_failure_retracts_disk_and_return_without_clock_reset(tmp_path, failure, monkeypatch):
    elapsed = [0.]
    def writer(path, value):
        subject.write_json(path, value)
        if path.name == "result.json" and failure == "postcommit":
            elapsed[0] = 56.
    def reader(path):
        value = subject.read(path)
        if failure == "readback":
            elapsed[0] = 11.
        return value
    if failure == "output":
        monkeypatch.setitem(subject.LIMITS, "output_bytes", 1)
    result, output = run_fixture(tmp_path, clock=lambda: elapsed[0], writer=writer, reader=reader)
    assert not result["run_valid"] and not result["effective_neural_qualification"] and not result["handoff_eligible"]
    assert subject.read(output / "result.json")["run_valid"] is False
    assert subject.read(output / "advance-ledger.json")["first_error"] == result["first_error"]
    assert result["first_error"]["stage"] == "publication"
    if failure in ("postcommit", "readback"):
        assert result["first_error"]["type"] == "TimeoutError"


def test_first_compute_failure_survives_all_later_sink_failures(tmp_path):
    def fail_compute(*args, **kwargs):
        raise ValueError("primary compute failure")
    def unavailable(*args, **kwargs):
        raise OSError("secondary sink unavailable")
    result, _ = run_fixture(tmp_path, compute_fn=fail_compute, writer=unavailable, emergency=unavailable, repair=unavailable)
    assert not result["run_valid"] and result["first_error"]["message"] == "primary compute failure"
    assert len(result["errors"]) >= 5


def test_pre_gate_absolute_clock_and_fixed_cli_dimensions(tmp_path, monkeypatch):
    path = tmp_path / "empty.json"
    path.write_text("{}", encoding="utf8")
    touched = []
    monkeypatch.setattr(subject.np, "load", lambda *args, **kwargs: touched.append(args))
    with pytest.raises(ValueError):
        subject.gate(path, subject.sha256(path), path, subject.sha256(path), tmp_path / subject.OUTPUT, root=tmp_path)
    assert not touched
    for kwargs in ({"device": "cpu"}, {"dependencies": object()}):
        with pytest.raises(ValueError, match="fullN"):
            subject.run(path, subject.sha256(path), path, subject.sha256(path), tmp_path / subject.OUTPUT, root=tmp_path, **kwargs)
    called = []
    def original_run(*args, **kwargs):
        called.append(kwargs)
        return {"run_valid": False, "classification": "invalid", "first_error": {"message": "unqualified"}}
    monkeypatch.setattr(subject, "run", original_run)
    assert subject.main(["--freeze", str(path), "--freeze-sha256", "x", "--review", str(path), "--review-sha256", "y", "--output", str(tmp_path / subject.OUTPUT)]) == 1
    assert called == [{}]
    assert (subject.N, subject.GROUPS, subject.CONTROLS) == (158706, 8547, 300)
