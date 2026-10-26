"""Pure/mock E217 geometry and conditional basis checks; actual CUDA is deferred."""

from __future__ import annotations

import os

import numpy as np
import pytest

from tools.banc_m3_grouped_response_basis import (
    BASIS_COUNT,
    PROBE,
    group_affine_readout,
    planned_gpu_bytes,
    run_grouped_basis_cuda,
    select_grouped_prototypes,
    simulate_grouped_basis_numpy,
)
from tools.prepare_c_banc_m3_grouped_response_basis import (
    validate_group_affine,
    validate_raw_basis,
    verify_gate,
)


def test_weighted_group_fps_ties_and_own_centers():
    descriptor = np.zeros((70, 3, 9), np.float64)
    descriptor[:, 0, 0] = np.arange(70)
    descriptor[:, 1, 0] = 2 * np.arange(70)
    descriptor[:, 2, 0] = 7  # degenerate own descriptor
    membership = np.array([0, 0, 1])
    pool = np.array([[1., 3., 0.], [0., 0., 2.]])
    cap = np.array([2., 1., 1.])
    result = select_grouped_prototypes(descriptor, membership, pool, cap)
    rows = result["anchor_rows"]
    assert rows.shape == (2, 64)
    assert np.array_equal(rows[:, 0], [0, 0])
    assert rows[0, 1] == 69
    assert np.array_equal(rows[1], np.arange(64))
    assert len(np.unique(rows[0])) == 64
    assert np.allclose(result["owner_weights"][:2], [2/5, 3/5])
    assert np.array_equal(result["centers"][1], descriptor[rows[0], 1])
    assert result["sigma2"][2] == 1.
    assert np.all(result["sigma2"] > 0)


def test_sigma_excludes_zero_nearest_other_after_nearest_selection():
    descriptor = np.zeros((65, 1, 9), np.float64)
    descriptor[32:, 0, 0] = 2.
    result = select_grouped_prototypes(descriptor, np.array([0]),
                                       np.ones((1, 1)), np.ones(1))
    rows = result["anchor_rows"][0]
    assert len(np.unique(rows)) == 64
    assert np.count_nonzero(descriptor[rows, 0, 0] == 0) == 32
    assert np.count_nonzero(descriptor[rows, 0, 0] == 2) == 32
    # Every selected center has a duplicate: nearest-other is zero even
    # though a strictly positive farther-center distance exists.
    assert result["sigma2"][0] == 1.


def _small_basis_inputs(ticks=8):
    rows = ticks * 4
    grid = np.arange(rows, dtype=np.float64)[:, None]
    positive = np.maximum(0, .09 + .04*np.sin(grid * [.23, .47, .61]))
    negative = np.maximum(0, .03 + .02*np.cos(grid * [.19, .37, .43]))
    drive = .06*np.sin(grid * [.31, .49, .17])
    return (positive.astype(np.float32), negative.astype(np.float32),
            drive.astype(np.float32))


def test_numpy_grouped_basis_probe_two_memories_and_mass():
    positive, negative, drive = _small_basis_inputs()
    scale = np.array([[.3, .2, .1], [.4, .1, .15], [.25, .14, .12]])
    inverse = np.array([17., 37., 71.], np.float64)
    centers = np.empty((3, 64, 9), np.float64)
    for m in range(3):
        for j in range(64):
            centers[m, j] = .05*np.sin((m+1)*(j+1)*np.arange(1, 10)/17)
    sigma2 = np.array([.03, .07, .11])
    initial = np.array([.1, .2, .3])
    tau = np.array([.014, .016, .018])
    cap = np.array([100., 80., 90.])
    raw = simulate_grouped_basis_numpy(positive, negative, drive, scale,
        inverse, centers, sigma2, initial, tau, cap, sample_every=1)
    assert raw["B"].shape == (9, 3, 64)
    assert np.array_equal(raw["H"][0], initial)
    assert np.array_equal(raw["h_fast"][0], np.zeros((3, 3)))
    assert np.max(np.abs(raw["h_fast"]-raw["h_slow"])) > 1e-8
    reconstructed = raw["H"] + cap[None] * np.sum(raw["B"]*PROBE, axis=2)
    assert len(PROBE) == BASIS_COUNT
    assert np.allclose(raw["probe_rate"], reconstructed, atol=1e-12, rtol=1e-12)
    assert np.min(raw["B"]) > -1e-12
    assert np.max(np.sum(raw["B"], axis=2) - (1-raw["H_factor"])) < 1e-12
    old = {"B": np.zeros((9, 3, 16)), "H": raw["H"].copy(),
           "H_factor": raw["H_factor"].copy()}
    old["B"][:, :, 0] = np.sum(raw["B"], axis=2)
    checks = validate_raw_basis(raw, {key: raw[key].copy() for key in (
        "h_fast", "h_slow", "descriptor")}, initial, cap, old)
    assert checks["B_sum_old_max_abs"] == 0
    old["B"][2, 0, 0] += 1e-4
    with pytest.raises(ValueError, match="envelope"):
        validate_raw_basis(raw, {key: raw[key] for key in (
            "h_fast", "h_slow", "descriptor")}, initial, cap, old)


def test_group_affine_canonical_pool_and_fixed_ltm():
    samples, motors = 3, 391
    membership = np.arange(motors) % 96
    # Place exactly 48 fixed MN in the final six original muscle groups.
    fixed = np.flatnonzero(membership >= 90)
    membership[fixed] = 90 + np.arange(len(fixed)) % 6
    # Include additional old owners so exactly 48 fixed MN are present.
    membership[np.arange(343, 391)] = 90 + np.arange(48) % 6
    membership[:343] = np.arange(343) % 90
    fixed = np.arange(343, 391)
    free = np.arange(343)
    pool = np.zeros((96, motors))
    pool[membership, np.arange(motors)] = 1.
    moment = np.zeros((42, 96))
    moment[0, 0] = .3
    rest = np.zeros(42)
    ltm = np.arange(90, 96)
    groups = [{"leg": "lf"} for _ in range(96)]
    for group, leg in zip(ltm, ("lf", "lm", "lh", "rf", "rm", "rh")):
        groups[group]["leg"] = leg
    B = np.zeros((samples, motors, 64))
    B[1:, 0, 0] = [0.2, .3]
    H = np.zeros((samples, motors))
    original = H.copy()
    old = {"rate_intercept": H.copy(),
        "q_intercept": np.zeros((samples, 42)),
        "tendon_intercept": np.zeros((samples, 6)),
        "activation_intercept": np.zeros((samples, 96)),
        "D": np.zeros((samples, motors, 16))}
    old["D"][1:, 0, 0] = B[1:, 0, 0]
    cap = np.ones(motors)
    affine = group_affine_readout({"B": B, "H": H}, cap, membership,
        pool, moment, rest, ltm, groups, free, fixed, original, old)
    alpha = -np.expm1(-.5)
    expected = alpha*.2
    assert affine["D_group"].shape == (samples, 96, 64)
    assert affine["D_group"][1, 0, 0] == pytest.approx(expected)
    assert np.max(np.abs(affine["D_group"][:, 90:])) == 0
    assert np.allclose(affine["q_probe_affine"],
                       affine["q_probe_direct"], atol=1e-15)
    # Build the independently old 16-column filtered envelope.
    old["D"][1, 0, 0] = expected
    old["D"][2, 0, 0] = expected + alpha*(.3-expected)
    canonical = {"activation": np.zeros((samples, 96)),
                 "tendon": np.zeros((samples, 6))}
    metrics = validate_group_affine(affine, old, membership, fixed,
                                    ltm, canonical)
    assert metrics["fixed_ltm_tendon_exact"]


def test_gpu_plan_one_call_and_partial_readback_without_cuda():
    plan = planned_gpu_bytes((32, 3), sample_every=1)
    assert plan["channel_bytes"] == 32*3*3*4
    assert plan["output_bytes"] == 9*3*82*8
    with pytest.raises(ValueError):
        planned_gpu_bytes((31, 3), sample_every=1)

    class FakeStream:
        def synchronize(self):
            return None

    class FakeCuda:
        @staticmethod
        def get_current_stream():
            return FakeStream()

    class FakeCp:
        float64 = np.float64
        cuda = FakeCuda()

        @staticmethod
        def empty(shape, dtype):
            return np.zeros(shape, dtype)

        @staticmethod
        def asarray(value, dtype=None):
            return np.asarray(value, dtype=dtype)

        @staticmethod
        def asnumpy(value):
            return value.copy()

    class FakeContext:
        cp = FakeCp()
        n = 3
        ticks = 8
        sample_every = 1
        calls_requested = 0
        calls_completed = 0
        parameter_transfer_count = 0
        parameter_upload_bytes = 0
        e = i = d = np.zeros((32, 3), np.float32)

        def __init__(self):
            self.launch_events = []
            self.current_output = None
            self.partial_host = {}

        @staticmethod
        def _measure():
            return None

        @staticmethod
        def kernel(grid, block, args):
            assert grid == (1,) and block == (128,)
            assert len(args) == 20

    context = FakeContext()
    centers = np.zeros((3, 64, 9))
    raw, _ = run_grouped_basis_cuda(context, np.ones((3, 3)),
        np.ones(3, np.float64), {"centers": centers, "sigma2": np.ones(3)},
        np.zeros(3), np.ones(3), np.ones(3), sample_every=1)
    assert raw["B"].shape == (9, 3, 64)
    assert context.calls_requested == context.calls_completed == 1
    assert context.parameter_transfer_count == 7
    assert context.launch_events[0]["readback_complete"]
    with pytest.raises(RuntimeError):
        run_grouped_basis_cuda(context, np.ones((3, 3)),
            np.ones(3, np.float64), {"centers": centers, "sigma2": np.ones(3)},
            np.zeros(3), np.ones(3), np.ones(3), sample_every=1)


def test_gate_fails_without_review(tmp_path):
    with pytest.raises(ValueError):
        verify_gate(tmp_path / "absent.json", "wrong",
                    tmp_path / "absent-review.json", "wrong")


def test_generated_grouped_cuda_parity(tmp_path):
    """Parent-only after FIRST review: exactly one small CUDA launch, no model."""
    if os.environ.get("FLYLAB_E217_GENERATED_CUDA") != "1":
        pytest.skip("actual generated CUDA is parent-controlled after FIRST")
    from tools.banc_m3_grouped_response_basis import PersistentGroupedInputs

    positive, negative, drive = _small_basis_inputs(ticks=8)
    scale = np.ones((3, 3))
    inverse = np.array([17., 37., 71.])
    centers = np.empty((3, 64, 9))
    for m in range(3):
        for j in range(64):
            centers[m, j] = .05*np.sin((m+1)*(j+1)*np.arange(1, 10)/17)
    sigma2 = np.array([.03, .07, .11])
    initial = np.array([.1, .2, .3])
    tau = np.array([.014, .016, .018])
    cap = np.array([100., 80., 90.])
    context = PersistentGroupedInputs(positive, negative, drive,
                                      sample_every=1)
    raw, _ = run_grouped_basis_cuda(context, scale, inverse,
        {"centers": centers, "sigma2": sigma2}, initial, tau, cap,
        sample_every=1)
    np.savez(tmp_path / "generated-grouped-cuda-raw.npz", **raw)
    reference = simulate_grouped_basis_numpy(positive, negative, drive,
        scale, inverse, centers, sigma2, initial, tau, cap, sample_every=1)
    assert context.calls_requested == context.calls_completed == 1
    for key in raw:
        assert np.allclose(raw[key], reference[key], atol=2e-9, rtol=2e-11), key
