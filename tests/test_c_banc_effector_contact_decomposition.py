"""Synthetic CPU-only E229 geometry, raw-retention, and aggregation checks."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tools import observe_c_banc_m3_effector_contact_decomposition as probe


def mapping():
    legs = np.full(4, -1, np.int64)
    legs[1] = 24  # RF first link, leg index 3.
    return {"geom_body_ids": [0, 1, 2, 3],
            "fly_body_ids": [1, 2], "support_geom_ids": [0],
            "ground_mask": [True, False, False, False],
            "leg_geom_map": legs, "weight0": 2.}


def classify(**changes):
    options = dict(geom1=0, geom2=1, excluded=False,
        frame=np.eye(3), wrench=np.array([1., 2., 0., 0., 0., 0.]),
        position=np.array([.1, .2, .3]), **mapping())
    options.update(changes)
    return probe.classify_all_contact(**options)


def test_all_active_floor_fly_sign_segment_and_nonpositive_raw():
    row = classify()
    assert (row["sign"], row["leg"], row["segment"]) == (1, 3, 24)
    np.testing.assert_array_equal(row["force_world_bw"], [.5, 1., 0.])
    assert row["normal_load_bw"] == .5
    assert row["positive_leg"]
    reverse = classify(geom1=1, geom2=0)
    assert reverse["sign"] == -1
    np.testing.assert_array_equal(reverse["force_world_bw"], [-.5, -1., 0.])
    nonleg = classify(geom2=2)
    assert nonleg["leg"] == -1 and nonleg["segment"] == -1
    assert not nonleg["positive_leg"]
    zero = classify(wrench=np.zeros(6))
    assert zero["normal_load_bw"] == 0. and not zero["positive_leg"]
    assert classify(excluded=True) is None
    assert classify(geom2=3) is None


def test_e157_point_jacobian_retains_root_own_residual_closure():
    qvel = np.arange(1., 15.)
    jfly = np.zeros((3, 14))
    jfly[0, :] = 1.
    jfloor = np.zeros_like(jfly)
    d = probe.e157.decompose_contact(np.eye(3), qvel, jfly, jfloor,
        np.array([0., 1., 0.]), np.arange(6), np.arange(6, 13),
        np.array([2., 1., 0., 0., 0., 0.]), 1, 2.)
    # Normal is x; tangent y. jfly has only normal x, hence zero tangent speed.
    assert d["ap_total"] == 0.
    assert d["power_total"] == 0.
    assert max(d[key] for key in ("ap_closure_error", "ap_direct_error",
        "power_closure_error", "power_direct_error", "projection_error")) == 0.


def test_second_jacobian_failure_retains_first_and_all_contact_raw(tmp_path):
    class MJ:
        calls = 0

        def mj_contactForce(self, _m, _d, _id, wrench):
            wrench[:] = [1., 0., 0., 0., 0., 0.]

        def mj_jac(self, _m, _d, dest, _rot, _point, _body):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("generated floor Jacobian error")
            dest[1, 0] = 1.

    contact = SimpleNamespace(geom1=0, geom2=1, exclude=False,
                              frame=np.eye(3), pos=np.zeros(3))
    body = SimpleNamespace(m=SimpleNamespace(nv=14), mj=MJ(),
        support_geom_ids={0}, _ground_mask=np.array([True, False]))
    body.m.geom_bodyid = np.array([0, 1])
    shadow = SimpleNamespace(d=SimpleNamespace(contact=[contact], ncon=1,
        qvel=np.arange(14., dtype=float)))
    registry = probe.new_registry(1, 14, 1)
    probe.append(registry["native_state"], {
        "native_index": 0, "time_s": .0001, "control_tick": 1608,
        "qpos": np.zeros(1), "qvel": np.arange(14., dtype=float),
        "ctrl": np.zeros(1), "state_captured": True,
        "shadow_forward_returned": True, "observation_returned": False})
    ledger = {"excluded_contact_count": 0,
        "contact_force_calls_requested": 0, "contact_force_calls_returned": 0,
        "contact_rows": 0, "nonleg_contact_count": 0,
        "nonpositive_leg_contact_count": 0,
        "jacobian_calls_requested": 0, "jacobian_calls_returned": 0,
        "decomposition_rows": 0}
    layout = {"leg_geom_map": [-1, 24], "geom_body_ids": [0, 1],
              "support_geom_ids": [0], "ground_mask": [True, False],
              "root_dof_addresses": list(range(6)),
              "leg_dof_addresses": [list(range(6, 13))] * 6}
    inventory = {"body_ids": [1], "weight0": 2.}
    with pytest.raises(RuntimeError, match="floor Jacobian"):
        probe.observe_contact_step(body, shadow, registry, layout, inventory,
            np.array([0., 1., 0.]), ledger,
            {"contact_rows_max": 16000, "jacobian_calls_max": 32000})
    state, contacts, attempts, decomposition = probe.arrays(registry)
    assert len(state["native_index"]) == 1
    assert len(contacts["native_index"]) == 1
    assert len(attempts["contact_row"]) == 1
    assert attempts["jfly_requested"][0] and attempts["jfly_returned"][0]
    assert attempts["jfloor_requested"][0] and not attempts["jfloor_returned"][0]
    assert not attempts["decomposition_complete"][0]
    assert len(decomposition["contact_row"]) == 0
    assert ledger["jacobian_calls_requested"] == 2
    assert ledger["jacobian_calls_returned"] == 1
    probe.save_npz(tmp_path / "contacts-all.npz", contacts)
    probe.save_npz(tmp_path / "jacobian-attempts.npz", attempts)
    with np.load(tmp_path / "contacts-all.npz") as raw:
        assert raw["contact_id"].tolist() == [0]
        assert raw["positive_leg"].tolist() == [True]
    with np.load(tmp_path / "jacobian-attempts.npz") as raw:
        assert raw["jfly_returned"].tolist() == [True]
        assert raw["jfloor_returned"].tolist() == [False]


def test_unique_support_time_load_weighted_velocity_unweighted_power():
    contacts = {"native_index": np.array([0, 0, 1]),
        "leg": np.array([0, 0, 0]),
        "normal_load_bw": np.array([1., 3., 2.]),
        "force_world_bw": np.array([[-1., 0., 0.], [-2., 0., 0.],
                                    [2., 0., 0.]])}
    decomposition = {"native_index": np.array([0, 0, 1]),
        "leg": np.array([0, 0, 0]),
        "normal_load_bw": np.array([1., 3., 2.]),
        "ap_total": np.array([1., 3., 5.]),
        "ap_root": np.zeros(3),
        "ap_own7": np.array([[1.] + [0.] * 6, [3.] + [0.] * 6,
                            [5.] + [0.] * 6]),
        "ap_residual": np.zeros(3), "slip_speed": np.array([1., 3., 5.]),
        "power_total": np.array([2., 4., 6.]),
        "power_root": np.zeros(3),
        "power_own7": np.array([[2.] + [0.] * 6, [4.] + [0.] * 6,
                               [6.] + [0.] * 6]),
        "power_residual": np.zeros(3)}
    report, loads, forces = probe.aggregate_phase(contacts, decomposition,
        {"lead": [1608, 1620]}, .02, np.array([1., 0., 0.]))
    supported = report["lead"]["lf"]["support"]
    braking = report["lead"]["lf"]["braking"]
    assert supported["unique_native_samples"] == 2
    assert supported["positive_contact_count"] == 3
    assert supported["supported_time_s"] == pytest.approx(.0002)
    assert supported["AP_load_weighted_mm_s"]["ap_total"] == pytest.approx(20. / 6.)
    assert supported["power_integral_BW_mm"]["power_total"] == pytest.approx(.0012)
    assert braking["unique_native_samples"] == 1
    assert braking["positive_contact_count"] == 2
    assert braking["power_integral_BW_mm"]["power_total"] == pytest.approx(.0006)
    assert report["lead"]["rf"]["support"]["AP_load_weighted_mm_s"][
        "ap_total"] is None
    assert loads[0, 0] == 4. and forces[0, 0] == -3.


def test_gate_denies_unreviewed_inputs_without_body(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        probe.gate(tmp_path / "missing", "bad", tmp_path / "missing-review",
                   "bad", tmp_path / "out")
