"""Generated-data tests for the E149 Stage-A adapter; no real model is opened."""

import copy
import hashlib
import json
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from tools import banc_m3_monotone_recruitment as native


def generated_anatomy():
    counts = np.full(96, 4, dtype=int)
    counts[:10] = 5
    counts[10] = 1
    assert counts.sum() == 391
    membership = np.repeat(np.arange(96), counts).astype(np.int64)
    membership = membership[np.random.default_rng(14900).permutation(391)]
    pool = np.zeros((96, 391), dtype=np.float64)
    for muscle in range(96):
        members = np.flatnonzero(membership == muscle)
        pool[muscle, members] = 1 / len(members)
    leg_counts = np.array([17, 15, 16, 17, 15, 16])
    leg_for_muscle = np.repeat(("lf", "lm", "lh", "rf", "rm", "rh"), leg_counts)
    ltm = np.array([15, 30, 46, 63, 78, 94], dtype=np.int64)
    groups = [
        {
            "leg": leg_for_muscle[i],
            "muscle": "long_tendon_muscle" if i in ltm else f"muscle_{i}",
            "columns": np.flatnonzero(membership == i).tolist(),
        }
        for i in range(96)
    ]
    moment = np.zeros((42, 96), dtype=np.float64)
    moment[np.arange(42), np.arange(42)] = np.linspace(-0.4, 0.4, 42)
    calibration = {
        "moment": moment,
        "rest": np.linspace(-0.2, 0.2, 42),
        "pool": pool,
        "motor_ids": np.arange(391, dtype=np.int64),
        "ltm": ltm,
    }
    slopes = np.zeros((96, 391, 3, 2), dtype=np.float64)
    canonical = np.zeros_like(slopes)
    canonical[:, :, 1, :] = pool[:, :, None]
    fixed = set(calibration["ltm"]) | {10}
    for muscle in range(96):
        members = np.flatnonzero(membership == muscle)
        if muscle in fixed:
            slopes[muscle] = canonical[muscle]
        elif muscle == 11:  # Duplicate slopes: exact linear reduction.
            value = np.array([0.2, 0.5, 0.3]) / len(members)
            slopes[muscle, members, :, 0] = value
            slopes[muscle, members, :, 1] = value
        else:
            slopes[muscle, members, :, 0] = np.array([0.05, 0.70, 0.05]) / len(members)
            slopes[muscle, members, :, 1] = np.array([0.08, 1.00, 0.12]) / len(members)
    return calibration, groups, membership, slopes


class FakeOriginal:
    schema = "fake-anatomical.v1"

    def __init__(self, calibration, groups):
        self.calibration = {key: value.copy() for key, value in calibration.items()}
        self.groups = copy.deepcopy(groups)
        self.motor_ids = self.calibration["motor_ids"]
        payload = {key: value.tolist() for key, value in self.calibration.items()}
        self.identity = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        self.activation = np.zeros(96, dtype=np.float64)
        self.last_motor = []
        self.body = SimpleNamespace(neutral=np.linspace(-0.3, 0.3, 42))

    @property
    def tendon_inputs(self):
        return {
            self.groups[i]["leg"] + "_tarsus": float(
                0.2 * np.tanh(self.activation[i] / 10)
            )
            for i in self.calibration["ltm"]
        }

    def decode(self, rates, *, motor_cut=False, dt=0.005):
        rates = np.asarray(rates, dtype=np.float64)
        if motor_cut:
            self.activation.fill(0)
            target = self.body.neutral.copy()
        else:
            pooled = self.calibration["pool"] @ rates
            self.activation += -np.expm1(-dt / 0.01) * (pooled - self.activation)
            target = (
                self.calibration["rest"]
                + 0.02 * self.calibration["moment"] @ self.activation
            )
        self.last_motor = [{"tick_rate_sum": float(rates.sum()), "cut": motor_cut}]
        return target, np.arange(6) % 2 == 0

    def snapshot(self):
        return {
            "schema": self.schema,
            "hash": self.identity,
            "activation": self.activation.copy(),
            "last_motor": copy.deepcopy(self.last_motor),
        }

    def restore(self, saved):
        if saved.get("schema") != self.schema or saved.get("hash") != self.identity:
            raise ValueError("fake original identity")
        active = np.asarray(saved["activation"])
        if (
            active.shape != (96,)
            or active.dtype != np.float64
            or not np.isfinite(active).all()
        ):
            raise ValueError("fake original activation")
        self.activation[:] = active
        self.last_motor = copy.deepcopy(saved["last_motor"])


def make_adapter():
    calibration, groups, membership, slopes = generated_anatomy()
    original = FakeOriginal(calibration, groups)
    tau = native.FILTER_TAU_S.copy()
    kappa = np.linspace(0, 3, 391, dtype=np.float64)
    roster = [f"flywire:banc:888:{720575940000000000 + i}" for i in range(391)]
    return native.MonotoneRecruitmentAdapter(
        original, tau, kappa, slopes, membership, roster
    ), (tau, kappa, slopes, membership, roster)


def test_uneven_anatomy_nonlinear_duplicate_and_fixed_outputs():
    adapter, _ = make_adapter()
    rates = np.linspace(0, 8, 391)
    target, adhesion = adapter.decode(rates, dt=0.007)
    assert target.shape == (42,) and adhesion.shape == (6,)
    np.testing.assert_array_equal(
        adapter.activation[adapter.fixed_muscles],
        adapter.original.activation[adapter.fixed_muscles],
    )
    np.testing.assert_array_equal(adapter.tendon_inputs, adapter.original.tendon_inputs)
    members = np.flatnonzero(adapter.membership == 11)
    low = adapter.slopes[11, members, :, 0]
    expected = np.sum(low * adapter.bank[members])
    assert adapter.activation[11] == expected
    nonlinear = 12
    members = np.flatnonzero(adapter.membership == nonlinear)
    low = adapter.slopes[nonlinear, members, :, 0]
    high = adapter.slopes[nonlinear, members, :, 1]
    explicit = np.sum(low * adapter.bank[members]) + np.sum(
        (high - low)
        * np.maximum(adapter.bank[members] - adapter.kappa[members, None], 0)
    )
    assert adapter.activation[nonlinear] == explicit


def test_variable_dt_prefix_snapshot_restore_and_defensive_copies():
    adapter, parameters = make_adapter()
    tau, kappa, slopes, membership, roster = parameters
    rng = np.random.default_rng(14901)
    rates = rng.uniform(0, 12, size=(7, 391))
    dts = [0.003, 0.011, 0.005, 0.02, 0.001, 0.05, 0.007]
    for index in range(3):
        adapter.decode(rates[index], dt=dts[index])
    checkpoint = adapter.snapshot()
    restored, _ = make_adapter()
    restored.restore(checkpoint)
    for index in range(3, 7):
        left = adapter.decode(rates[index], dt=dts[index])
        right = restored.decode(rates[index], dt=dts[index])
        np.testing.assert_array_equal(left[0], right[0])
        np.testing.assert_array_equal(adapter.bank, restored.bank)
        np.testing.assert_array_equal(adapter.activation, restored.activation)
    tau[:] = 99
    kappa[:] = 99
    slopes[:] = 99
    membership[:] = 0
    roster[0] = "changed"
    np.testing.assert_array_equal(adapter.tau, native.FILTER_TAU_S)
    assert adapter.kappa[0] == 0 and adapter.motor_roster[0] != "changed"
    checkpoint["bank"][:] = 99
    checkpoint["parameters"]["slopes"][:] = 99
    assert not np.all(adapter.bank == 99) and not np.all(adapter.slopes == 99)


def test_constructor_preserves_progressed_original_fixed_state_and_snapshot():
    calibration, groups, membership, slopes = generated_anatomy()
    original = FakeOriginal(calibration, groups)
    original.decode(np.linspace(1, 9, 391), dt=0.013)
    original_fixed = original.activation[calibration["ltm"]].copy()
    original_last = copy.deepcopy(original.last_motor)
    adapter = native.MonotoneRecruitmentAdapter(
        original,
        native.FILTER_TAU_S.copy(),
        np.linspace(0, 3, 391),
        slopes,
        membership,
        [f"root:{i}" for i in range(391)],
    )
    np.testing.assert_array_equal(
        adapter.activation[calibration["ltm"]], original_fixed
    )
    assert adapter.last_motor == original_last
    assert np.all(adapter.activation[adapter.variable_muscles] == 0)
    saved = adapter.snapshot()
    adapter.restore(saved)
    native.exact(saved, adapter.snapshot(), "progressed constructor checkpoint")


def test_complex_rates_rejected_before_any_state_change_without_warning():
    adapter, _ = make_adapter()
    adapter.decode(np.linspace(0, 2, 391), dt=0.007)
    before = adapter.snapshot()
    bad = np.linspace(0, 2, 391).astype(np.complex128)
    bad[0] += 1j
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="rates and actual timestep"):
            adapter.decode(bad, dt=0.011)
    assert caught == []
    native.exact(before, adapter.snapshot(), "complex input rejection state")


def test_cut_zero_and_restart_matches_fresh_zero_state():
    adapter, _ = make_adapter()
    rng = np.random.default_rng(14902)
    adapter.decode(rng.uniform(0, 8, 391), dt=0.013)
    target, _ = adapter.decode(np.zeros(391), motor_cut=True, dt=0.017)
    np.testing.assert_array_equal(target, adapter.body.neutral)
    assert np.all(adapter.bank == 0) and np.all(adapter.activation == 0)
    assert np.all(adapter.original.activation == 0)
    fresh, _ = make_adapter()
    restart_rate = rng.uniform(0, 8, 391)
    restarted = adapter.decode(restart_rate, dt=0.009)[0]
    expected = fresh.decode(restart_rate, dt=0.009)[0]
    np.testing.assert_array_equal(restarted, expected)
    np.testing.assert_array_equal(adapter.bank, fresh.bank)


@pytest.mark.parametrize(
    "change,match",
    [
        (
            lambda tau, kappa, slopes, membership: slopes.__setitem__(
                (12, 0, 0, 0), -1
            ),
            "nonnegative",
        ),
        (
            lambda tau, kappa, slopes, membership: membership.__setitem__(0, 1),
            "Membership",
        ),
        (
            lambda tau, kappa, slopes, membership: slopes.__setitem__(
                (15, slice(None), slice(None), slice(None)), 0
            ),
            "Fixed LTM",
        ),
        (
            lambda tau, kappa, slopes, membership: tau.__setitem__(0, 0.006),
            "filter tau",
        ),
    ],
)
def test_parameter_validation(change, match):
    calibration, groups, membership, slopes = generated_anatomy()
    tau = native.FILTER_TAU_S.copy()
    kappa = np.ones(391, dtype=np.float64)
    change(tau, kappa, slopes, membership)
    with pytest.raises(ValueError, match=match):
        native.MonotoneRecruitmentAdapter(
            FakeOriginal(calibration, groups),
            tau,
            kappa,
            slopes,
            membership,
            [f"root:{i}" for i in range(391)],
        )


def test_restore_rejects_state_parameter_and_support_mutation():
    adapter, _ = make_adapter()
    adapter.decode(np.linspace(0, 5, 391), dt=0.005)
    saved = adapter.snapshot()
    bad = copy.deepcopy(saved)
    bad["bank"][0, 0] = np.nan
    with pytest.raises(ValueError, match="filter bank"):
        adapter.restore(bad)
    bad = copy.deepcopy(saved)
    bad["activation"][12] += 0.1
    with pytest.raises(ValueError, match="disagrees"):
        adapter.restore(bad)
    bad = copy.deepcopy(saved)
    bad["parameters"]["slopes"][0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="parameters"):
        adapter.restore(bad)
    bad = copy.deepcopy(saved)
    bad["last_motor"][0]["cut"] = True
    with pytest.raises(ValueError, match="Last motor"):
        adapter.restore(bad)


class FakeGraph:
    def __init__(self):
        self.nodes = [
            {"id": f"flywire:banc:888:{720575940000000000 + i}"} for i in range(391)
        ]


def fake_base(adapter):
    base = object.__new__(native.MotorTimescaleTrial)
    base.graph = FakeGraph()
    base.adapter = adapter
    base.network = SimpleNamespace(marker="unchanged-network")
    base.body = adapter.body
    base.tendon_inputs = adapter.tendon_inputs.copy()
    base.closed = False
    base.close_count = 0
    base.motor_timescale_patch = {"schema": "fake-timescale", "future": np.arange(3)}
    return base


def install_fake_trial_methods(monkeypatch):
    def snapshot(self):
        return {
            "schema": self.schema,
            "adapter": self.adapter.snapshot(),
            "network": {"marker": self.network.marker},
            "motor_timescale_patch": copy.deepcopy(self.motor_timescale_patch),
            "tendon_inputs": copy.deepcopy(self.tendon_inputs),
            "closed": self.closed,
        }

    def restore_base(cls, graph, saved):
        calibration, groups, _, _ = generated_anatomy()
        original = FakeOriginal(calibration, groups)
        original.restore(saved["adapter"])
        base = fake_base(original)
        base.graph = graph
        base.network.marker = saved["network"]["marker"]
        base.motor_timescale_patch = copy.deepcopy(saved["motor_timescale_patch"])
        base.tendon_inputs = copy.deepcopy(saved["tendon_inputs"])
        base.closed = saved["closed"]
        return base

    def close(self):
        self.close_count = getattr(self, "close_count", 0) + 1
        self.closed = True

    monkeypatch.setattr(native.MotorTimescaleTrial, "snapshot", snapshot)
    monkeypatch.setattr(
        native.MotorTimescaleTrial, "from_checkpoint", classmethod(restore_base)
    )
    monkeypatch.setattr(native.MotorTimescaleTrial, "close", close)


def test_nested_trial_restoration_keeps_network_and_source_provenance(monkeypatch):
    install_fake_trial_methods(monkeypatch)
    calibration, groups, membership, slopes = generated_anatomy()
    original = FakeOriginal(calibration, groups)
    original.decode(np.linspace(0, 3, 391), dt=0.012)
    boundary_fixed = original.activation[calibration["ltm"]].copy()
    boundary_last = copy.deepcopy(original.last_motor)
    base = fake_base(original)
    trial = native.MonotoneRecruitmentTrial(
        base, native.FILTER_TAU_S.copy(), np.linspace(0, 2, 391), slopes, membership
    )
    source_network = trial.network
    source_identity = trial.monotone_recruitment_patch["source_adapter_identity"]
    np.testing.assert_array_equal(
        trial.adapter.activation[calibration["ltm"]], boundary_fixed
    )
    assert trial.adapter.last_motor == boundary_last
    saved = trial.snapshot()
    clone = native.MonotoneRecruitmentTrial.from_checkpoint(trial.graph, saved)
    assert clone.network.marker == source_network.marker == "unchanged-network"
    assert (
        clone.monotone_recruitment_patch["source_adapter_identity"] == source_identity
    )
    np.testing.assert_array_equal(clone.adapter.bank, trial.adapter.bank)
    assert clone.snapshot()["schema"] == native.TRIAL_SCHEMA
    future_rate = np.linspace(0, 4, 391)
    np.testing.assert_array_equal(
        clone.adapter.decode(future_rate, dt=0.009)[0],
        trial.adapter.decode(future_rate, dt=0.009)[0],
    )
    saved["adapter"]["bank"][:] = 99
    assert not np.all(clone.adapter.bank == 99)


def test_nested_restore_failure_closes_and_preserves_primary(monkeypatch):
    install_fake_trial_methods(monkeypatch)
    calibration, groups, membership, slopes = generated_anatomy()
    trial = native.MonotoneRecruitmentTrial(
        fake_base(FakeOriginal(calibration, groups)),
        native.FILTER_TAU_S.copy(),
        np.ones(391),
        slopes,
        membership,
    )
    saved = trial.snapshot()
    saved["adapter"]["activation"][12] += 1
    closed = []

    def close_with_secondary(self):
        self.closed = True
        closed.append(True)
        raise OSError("secondary cleanup")

    monkeypatch.setattr(native.MotorTimescaleTrial, "close", close_with_secondary)
    with pytest.raises(ValueError, match="Activation disagrees") as caught:
        native.MonotoneRecruitmentTrial.from_checkpoint(trial.graph, saved)
    assert closed
    assert "Body cleanup failed" in " ".join(caught.value.__notes__)
