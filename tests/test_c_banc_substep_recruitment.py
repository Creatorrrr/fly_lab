"""Generated-only tests for the E156 causal substep helper."""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import banc_m3_monotone_recruitment as monotone
from tools import banc_m3_substep_recruitment as substep


def generated_anatomy(*, dense_signed_moment=False, decreasing_high=False):
    counts = np.full(96, 4, dtype=int)
    counts[:10] = 5
    counts[10] = 1
    assert counts.sum() == 391
    membership = np.repeat(np.arange(96), counts).astype(np.int64)
    membership = membership[np.random.default_rng(15600).permutation(391)]
    pool = np.zeros((96, 391), dtype=np.float64)
    for muscle in range(96):
        members = np.flatnonzero(membership == muscle)
        pool[muscle, members] = 1 / len(members)
    leg_counts = np.array([17, 15, 16, 17, 15, 16])
    legs = np.repeat(("lf", "lm", "lh", "rf", "rm", "rh"), leg_counts)
    ltm = np.array([15, 30, 46, 63, 78, 94], dtype=np.int64)
    groups = [
        {
            "leg": legs[muscle],
            "muscle": "long_tendon_muscle" if muscle in ltm else f"m{muscle}",
            "columns": np.flatnonzero(membership == muscle).tolist(),
        }
        for muscle in range(96)
    ]
    if dense_signed_moment:
        moment = np.random.default_rng(15601).normal(0, 0.2, size=(42, 96))
    else:
        moment = np.zeros((42, 96), dtype=np.float64)
        moment[np.arange(42), np.arange(42)] = np.linspace(0.1, 0.8, 42)
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
    fixed = set(ltm) | {10}
    for muscle in range(96):
        members = np.flatnonzero(membership == muscle)
        if muscle in fixed:
            slopes[muscle] = canonical[muscle]
        else:
            low = np.array([0.05, 0.70, 0.05]) / len(members)
            if decreasing_high and muscle == 11:
                high = np.array([0.03, 0.60, 0.02]) / len(members)
            else:
                high = np.array([0.12, 1.00, 0.18]) / len(members)
            slopes[muscle, members, :, 0] = low
            slopes[muscle, members, :, 1] = high
    return calibration, groups, membership, slopes


class FakeOriginal:
    schema = "generated-original.v1"

    def __init__(self, calibration, groups):
        self.calibration = {key: value.copy() for key, value in calibration.items()}
        self.groups = copy.deepcopy(groups)
        self.identity = hashlib.sha256(
            json.dumps(
                {key: value.tolist() for key, value in calibration.items()},
                sort_keys=True,
            ).encode()
        ).hexdigest()
        self.activation = np.zeros(96, dtype=np.float64)
        self.last_motor = []
        self.body = SimpleNamespace(neutral=np.linspace(-0.3, 0.3, 42))

    def decode(self, rates, *, motor_cut=False, dt=0.005):
        rate = np.asarray(rates, dtype=np.float64)
        if motor_cut:
            self.activation.fill(0)
            target = self.body.neutral.copy()
        else:
            pooled = self.calibration["pool"] @ rate
            self.activation += -np.expm1(-dt / 0.01) * (pooled - self.activation)
            target = self.calibration["rest"] + 0.02 * (
                self.calibration["moment"] @ self.activation
            )
        self.last_motor = [{"sum": float(rate.sum()), "cut": motor_cut}]
        return target, np.zeros(6, dtype=bool)

    def snapshot(self):
        return {
            "schema": self.schema,
            "hash": self.identity,
            "activation": self.activation.copy(),
            "last_motor": copy.deepcopy(self.last_motor),
        }

    def restore(self, saved):
        self.activation[:] = saved["activation"]
        self.last_motor = copy.deepcopy(saved["last_motor"])


def make_adapter(*, dense_signed_moment=False, decreasing_high=False):
    calibration, groups, membership, slopes = generated_anatomy(
        dense_signed_moment=dense_signed_moment,
        decreasing_high=decreasing_high,
    )
    original = FakeOriginal(calibration, groups)
    roster = tuple(f"flywire:banc:888:{900000000000000000 + i}" for i in range(391))
    adapter = monotone.MonotoneRecruitmentAdapter(
        original,
        monotone.FILTER_TAU_S.copy(),
        np.linspace(0.05, 2.5, 391, dtype=np.float64),
        slopes,
        membership,
        roster,
    )
    return adapter


def snapshot_state(adapter):
    return {
        "bank": adapter.bank.copy(),
        "activation": adapter.activation.copy(),
        "fixed": adapter.original.activation.copy(),
        "identity": adapter.identity,
        "neutral": adapter.body.neutral.copy(),
    }


def assert_state_equal(adapter, saved):
    np.testing.assert_array_equal(adapter.bank, saved["bank"])
    np.testing.assert_array_equal(adapter.activation, saved["activation"])
    np.testing.assert_array_equal(adapter.original.activation, saved["fixed"])
    np.testing.assert_array_equal(adapter.body.neutral, saved["neutral"])
    assert adapter.identity == saved["identity"]


def decoded_interval(adapter, rates, dt):
    bank0 = adapter.bank.copy()
    fixed0 = adapter.original.activation.copy()
    endpoint, _ = adapter.decode(rates, dt=dt)
    decoded = snapshot_state(adapter)
    return bank0, fixed0, endpoint, decoded


def readonly_view(adapter):
    return SimpleNamespace(
        schema=adapter.schema,
        identity=adapter.identity,
        original=SimpleNamespace(
            identity=adapter.original.identity,
            activation=adapter.original.activation.copy(),
        ),
        body=SimpleNamespace(neutral=adapter.body.neutral.copy()),
        calibration={key: value.copy() for key, value in adapter.calibration.items()},
        groups=copy.deepcopy(adapter.groups),
        tau=adapter.tau.copy(),
        kappa=adapter.kappa.copy(),
        slopes=adapter.slopes.copy(),
        membership=adapter.membership.copy(),
        fixed_muscles=adapter.fixed_muscles.copy(),
        variable_muscles=adapter.variable_muscles.copy(),
        lower_budget=adapter.lower_budget.copy(),
        upper_budget=adapter.upper_budget.copy(),
        motor_minimum_slope=adapter.motor_minimum_slope.copy(),
        motor_roster=tuple(adapter.motor_roster),
        bank=adapter.bank.copy(),
        activation=adapter.activation.copy(),
    )


def test_endpoint_scalar_midpoint_and_immutable_state_and_inputs():
    adapter = make_adapter()
    adapter.decode(np.linspace(0.2, 2.0, 391), dt=0.003)
    rates = np.linspace(0.5, 8.0, 391)
    bank0, fixed0, endpoint, decoded = decoded_interval(adapter, rates, 0.005)
    inputs = (bank0.copy(), fixed0.copy(), rates.copy(), endpoint.copy())

    result = substep.compute_substep_joint_targets(
        adapter,
        bank0,
        fixed0,
        rates,
        endpoint,
        motor_cut=False,
        dt=0.005,
        physics_dt=0.0001,
    )
    assert result.joint_targets.shape == (50, 42)
    np.testing.assert_array_equal(result.joint_targets[-1], endpoint)
    assert max(result.endpoint_error.values()) <= 1e-12

    index = 12
    t = result.relative_time_s[index]
    bank = bank0 + (-np.expm1(-t / adapter.tau))[None, :] * (rates[:, None] - bank0)
    pooled = adapter.calibration["pool"] @ rates
    fixed = fixed0 + (-np.expm1(-t / 0.01)) * (pooled - fixed0)
    active = np.empty(96)
    active[adapter.fixed_muscles] = fixed[adapter.fixed_muscles]
    for muscle in adapter.variable_muscles:
        members = np.flatnonzero(adapter.membership == muscle)
        low = adapter.slopes[muscle, members, :, 0]
        high = adapter.slopes[muscle, members, :, 1]
        active[muscle] = np.sum(low * bank[members]) + np.sum(
            (high - low) * np.maximum(bank[members] - adapter.kappa[members, None], 0)
        )
    expected = adapter.calibration["rest"] + 0.02 * (
        adapter.calibration["moment"] @ active
    )
    np.testing.assert_allclose(
        result.joint_targets[index], expected, atol=1e-15, rtol=0
    )
    assert_state_equal(adapter, decoded)
    for actual, expected_input in zip((bank0, fixed0, rates, endpoint), inputs):
        np.testing.assert_array_equal(actual, expected_input)
    assert not result.joint_targets.flags.writeable
    assert not result.relative_time_s.flags.writeable


def test_validated_readonly_adapter_view_uses_same_api_without_isinstance():
    adapter = make_adapter()
    rates = np.linspace(0.25, 4.5, 391)
    bank0, fixed0, endpoint, _ = decoded_interval(adapter, rates, 0.005)
    direct = substep.compute_substep_joint_targets(
        adapter,
        bank0,
        fixed0,
        rates,
        endpoint,
        motor_cut=False,
        dt=0.005,
        physics_dt=0.0001,
    )
    view = readonly_view(adapter)
    replay = substep.compute_substep_joint_targets(
        view,
        bank0,
        fixed0,
        rates,
        endpoint,
        motor_cut=False,
        dt=0.005,
        physics_dt=0.0001,
    )
    np.testing.assert_array_equal(direct.joint_targets, replay.joint_targets)
    np.testing.assert_array_equal(direct.relative_time_s, replay.relative_time_s)


def test_three_timescales_knot_crossing_fixed_singleton_and_ltm():
    adapter = make_adapter()
    rates = np.linspace(2.0, 20.0, 391)
    bank0, fixed0, endpoint, _ = decoded_interval(adapter, rates, 0.01)
    result = substep.CausalSubstepRecruitment(adapter).compute(
        bank0,
        fixed0,
        rates,
        endpoint,
        motor_cut=False,
        dt=0.01,
        physics_dt=0.0001,
    )
    # Positive diagonal moments make the first 42 generated targets monotone.
    delta = np.diff(result.joint_targets[:, :42], axis=0)
    assert np.all(delta >= -1e-15)
    motor = np.flatnonzero(adapter.membership == 11)[0]
    trajectory = bank0[motor] + (
        -np.expm1(-result.relative_time_s[:, None] / adapter.tau)
    ) * (rates[motor] - bank0[motor])
    assert trajectory[-1, 0] > trajectory[-1, 2]
    assert np.any(trajectory[:, 1] < adapter.kappa[motor])
    assert np.any(trajectory[:, 1] > adapter.kappa[motor])
    for muscle in np.r_[adapter.calibration["ltm"], 10]:
        assert muscle in adapter.fixed_muscles


def test_decreasing_positive_high_slopes_and_dense_signed_moment():
    adapter = make_adapter(dense_signed_moment=True, decreasing_high=True)
    adapter.decode(np.linspace(9.0, 1.0, 391), dt=0.004)
    rates = np.linspace(0.2, 12.0, 391)
    bank0, fixed0, endpoint, _ = decoded_interval(adapter, rates, 0.005)
    result = substep.compute_substep_joint_targets(
        adapter,
        bank0,
        fixed0,
        rates,
        endpoint,
        motor_cut=False,
        dt=0.005,
        physics_dt=0.0001,
    )
    members = np.flatnonzero(adapter.membership == 11)
    low = adapter.slopes[11, members, :, 0]
    high = adapter.slopes[11, members, :, 1]
    assert np.all(high > 0) and np.all(high < low)
    assert np.all(adapter.calibration["moment"] != 0)

    sample = 27
    t = result.relative_time_s[sample]
    bank = bank0 + (-np.expm1(-t / adapter.tau))[None, :] * (rates[:, None] - bank0)
    pooled = adapter.calibration["pool"] @ rates
    fixed = fixed0 + (-np.expm1(-t / 0.01)) * (pooled - fixed0)
    active = np.empty(96)
    active[adapter.fixed_muscles] = fixed[adapter.fixed_muscles]
    for muscle in adapter.variable_muscles:
        muscle_members = np.flatnonzero(adapter.membership == muscle)
        muscle_low = adapter.slopes[muscle, muscle_members, :, 0]
        muscle_high = adapter.slopes[muscle, muscle_members, :, 1]
        active[muscle] = np.sum(muscle_low * bank[muscle_members]) + np.sum(
            (muscle_high - muscle_low)
            * np.maximum(bank[muscle_members] - adapter.kappa[muscle_members, None], 0)
        )
    expected = adapter.calibration["rest"] + 0.02 * (
        adapter.calibration["moment"] @ active
    )
    np.testing.assert_allclose(
        result.joint_targets[sample], expected, atol=1e-15, rtol=0
    )


def test_motor_cut_is_immediate_neutral_then_regular_restart():
    adapter = make_adapter()
    adapter.decode(np.full(391, 9.0), dt=0.005)
    bank0 = adapter.bank.copy()
    fixed0 = adapter.original.activation.copy()
    endpoint, _ = adapter.decode(np.full(391, 7.0), motor_cut=True, dt=0.017)
    result = substep.compute_substep_joint_targets(
        adapter,
        bank0,
        fixed0,
        np.full(391, 7.0),
        endpoint,
        motor_cut=True,
        dt=0.017,
        physics_dt=0.0001,
    )
    assert result.joint_targets.shape == (170, 42)
    np.testing.assert_array_equal(
        result.joint_targets,
        np.broadcast_to(adapter.body.neutral, result.joint_targets.shape),
    )
    restart_rates = np.linspace(0, 5, 391)
    bank1, fixed1, target1, _ = decoded_interval(adapter, restart_rates, 0.002)
    restart = substep.compute_substep_joint_targets(
        adapter,
        bank1,
        fixed1,
        restart_rates,
        target1,
        motor_cut=False,
        dt=0.002,
        physics_dt=0.0001,
    )
    np.testing.assert_array_equal(restart.joint_targets[-1], target1)


def test_maximum_interval_is_vectorized_as_five_hundred_native_rows():
    adapter = make_adapter()
    rates = np.linspace(0.1, 6.0, 391)
    bank0, fixed0, endpoint, _ = decoded_interval(adapter, rates, 0.05)
    result = substep.compute_substep_joint_targets(
        adapter,
        bank0,
        fixed0,
        rates,
        endpoint,
        motor_cut=False,
        dt=0.05,
        physics_dt=0.0001,
    )
    assert result.joint_targets.shape == (500, 42)
    assert result.relative_time_s[-1] == 0.05
    np.testing.assert_array_equal(result.joint_targets[-1], endpoint)


def test_restore_determinism_and_future_rate_causality():
    adapter = make_adapter()
    adapter.decode(np.linspace(0, 3, 391), dt=0.004)
    checkpoint = adapter.snapshot()
    rate = np.linspace(1, 7, 391)
    bank0, fixed0, endpoint, _ = decoded_interval(adapter, rate, 0.005)
    first = substep.compute_substep_joint_targets(
        adapter,
        bank0,
        fixed0,
        rate,
        endpoint,
        motor_cut=False,
        dt=0.005,
        physics_dt=0.0001,
    )
    frozen_targets = first.joint_targets.copy()
    frozen_times = first.relative_time_s.copy()
    # A real later decode cannot mutate the already returned interval.
    future = np.full(391, 1000.0)
    adapter.decode(future, dt=0.005)
    np.testing.assert_array_equal(first.joint_targets, frozen_targets)
    np.testing.assert_array_equal(first.relative_time_s, frozen_times)

    restored = make_adapter()
    restored.restore(checkpoint)
    bank1, fixed1, endpoint1, _ = decoded_interval(restored, rate, 0.005)
    second = substep.compute_substep_joint_targets(
        restored,
        bank1,
        fixed1,
        rate,
        endpoint1,
        motor_cut=False,
        dt=0.005,
        physics_dt=0.0001,
    )
    np.testing.assert_array_equal(first.joint_targets, second.joint_targets)
    np.testing.assert_array_equal(first.relative_time_s, second.relative_time_s)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("bank", np.zeros((390, 3)), "pre-decode filter bank"),
        ("fixed", np.zeros(95), "pre-decode fixed activation"),
        ("rates", np.zeros(390), "391-MN rates"),
        ("rates", np.full(391, -1.0), "391-MN rates"),
        ("rates", np.full(391, np.nan), "391-MN rates"),
        ("rates", np.full(391, np.inf), "391-MN rates"),
        ("rates", np.full(391, 1 + 2j), "391-MN rates"),
        ("target", np.zeros(41), "endpoint target"),
    ],
)
def test_invalid_dynamic_shapes_values_are_rejected(field, value, match):
    adapter = make_adapter()
    rates = np.ones(391)
    bank0, fixed0, endpoint, _ = decoded_interval(adapter, rates, 0.005)
    values = {"bank": bank0, "fixed": fixed0, "rates": rates, "target": endpoint}
    values[field] = value
    with pytest.raises(ValueError, match=match):
        substep.compute_substep_joint_targets(
            adapter,
            values["bank"],
            values["fixed"],
            values["rates"],
            values["target"],
            motor_cut=False,
            dt=0.005,
            physics_dt=0.0001,
        )


@pytest.mark.parametrize(
    ("dt", "physics_dt", "match"),
    [
        (True, 0.0001, "control dt"),
        (0.0, 0.0001, "control dt"),
        (0.051, 0.0001, "control dt"),
        (0.005, True, "physics dt"),
        (0.005, 0.0002, "physics dt"),
        (0.00505, 0.0001, "must be an integer"),
    ],
)
def test_invalid_time_contract_is_rejected(dt, physics_dt, match):
    adapter = make_adapter()
    rates = np.ones(391)
    bank0, fixed0, endpoint, _ = decoded_interval(adapter, rates, 0.005)
    with pytest.raises(ValueError, match=match):
        substep.compute_substep_joint_targets(
            adapter,
            bank0,
            fixed0,
            rates,
            endpoint,
            motor_cut=False,
            dt=dt,
            physics_dt=physics_dt,
        )


def test_three_native_substeps_accept_float_roundoff():
    adapter = make_adapter()
    rates = np.linspace(0, 10, 391)
    bank0, fixed0, endpoint, _ = decoded_interval(adapter, rates, 0.0003)
    result = substep.compute_substep_joint_targets(
        adapter,
        bank0,
        fixed0,
        rates,
        endpoint,
        motor_cut=False,
        dt=0.0003,
        physics_dt=0.0001,
    )
    assert result.joint_targets.shape == (3, 42)
    np.testing.assert_array_equal(result.joint_targets[-1], endpoint)


def test_original_identity_change_after_construction_is_rejected():
    adapter = make_adapter()
    rates = np.ones(391)
    bank0, fixed0, endpoint, _ = decoded_interval(adapter, rates, 0.005)
    helper = substep.CausalSubstepRecruitment(adapter)
    adapter.original.identity = "0" * 64
    with pytest.raises(ValueError, match="fixed transfer changed"):
        helper.compute(
            bank0,
            fixed0,
            rates,
            endpoint,
            motor_cut=False,
            dt=0.005,
            physics_dt=0.0001,
        )


@pytest.mark.parametrize("field", ["bank", "fixed", "activation"])
def test_each_decoded_endpoint_state_mismatch_is_rejected(field):
    adapter = make_adapter()
    rates = np.linspace(0.5, 5.0, 391)
    bank0, fixed0, endpoint, _ = decoded_interval(adapter, rates, 0.005)
    if field == "bank":
        adapter.bank[0, 0] += 1e-6
    elif field == "fixed":
        adapter.original.activation[0] += 1e-6
    else:
        adapter.activation[0] += 1e-6
    with pytest.raises(ValueError, match="Reconstructed endpoint"):
        substep.compute_substep_joint_targets(
            adapter,
            bank0,
            fixed0,
            rates,
            endpoint,
            motor_cut=False,
            dt=0.005,
            physics_dt=0.0001,
        )


def test_schema_mapping_mutation_and_endpoint_mismatch_fail_closed():
    adapter = make_adapter()
    original_schema = adapter.schema
    adapter.schema = "wrong"
    with pytest.raises(ValueError, match="schema"):
        substep.CausalSubstepRecruitment(adapter)
    adapter.schema = original_schema

    original_identity = adapter.identity
    adapter.identity = "0" * 64
    with pytest.raises(ValueError, match="identity mismatch"):
        substep.CausalSubstepRecruitment(adapter)
    adapter.identity = original_identity

    helper = substep.CausalSubstepRecruitment(adapter)
    adapter.membership.flags.writeable = True
    adapter.membership[[0, 1]] = adapter.membership[[1, 0]]
    with pytest.raises(ValueError, match="fixed transfer changed"):
        helper.compute(
            np.zeros((391, 3)),
            np.zeros(96),
            np.ones(391),
            np.zeros(42),
            motor_cut=False,
            dt=0.005,
            physics_dt=0.0001,
        )

    adapter = make_adapter()
    rates = np.ones(391)
    bank0, fixed0, endpoint, _ = decoded_interval(adapter, rates, 0.005)
    with pytest.raises(ValueError, match="Reconstructed endpoint"):
        substep.compute_substep_joint_targets(
            adapter,
            bank0,
            fixed0,
            rates,
            endpoint + 1e-6,
            motor_cut=False,
            dt=0.005,
            physics_dt=0.0001,
        )
