"""Generated-only E253 decoder, CSR, concentration, and receipt checks."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from tools import analyze_c_banc_m3_lh_reductor_input as e253


def motor_roster() -> np.ndarray:
    anchors = [(-1, -100), *zip(e253.MOTOR_COLUMNS, e253.MOTOR_IDS, strict=True),
               (391, 158706)]
    result = np.empty(391, np.int64)
    for (left_col, left_id), (right_col, right_id) in pairwise(anchors):
        for col in range(left_col + 1, min(right_col + 1, 391)):
            result[col] = left_id + (right_id-left_id)*(col-left_col)//(right_col-left_col)
    assert np.array_equal(result[list(e253.MOTOR_COLUMNS)], e253.MOTOR_IDS)
    assert len(np.unique(result)) == 391
    return result


def csr_fixture(neurons: int, source: np.ndarray, bridge: np.ndarray,
                motor_ids: np.ndarray) -> tuple[dict, dict, dict]:
    rows: dict[int, list[tuple[int, float, bool]]] = {}
    for i, pre in enumerate(np.r_[source, source[:2]]):
        post = int(bridge[i % 4])
        rows.setdefault(post, []).append((int(pre), .001, True))
    for m, post in enumerate(motor_ids):
        rows[int(post)] = [(int(item), (-1 if m == 0 and j == 1 else 1), False)
                           for j, item in enumerate(bridge)]
        rows[int(post)].extend([(int(source[0]), -2, False),
                                (int(motor_ids[-1]), .5, False),
                                (777, -3, False)])
    ptr = np.zeros(neurons + 1, np.int32)
    index = []
    weight = []
    patch_ids = []
    patch_pre = []
    patch_post = []
    for post in range(neurons):
        for pre, value, patched in rows.get(post, []):
            if patched:
                patch_ids.append(len(index))
                patch_pre.append(pre)
                patch_post.append(post)
            index.append(pre)
            weight.append(value)
        ptr[post + 1] = len(index)
    before = {"indptr": ptr, "indices": np.asarray(index, np.int32),
              "data": np.asarray(weight, np.float32)}
    after = {key: value.copy() for key, value in before.items()}
    after["data"][patch_ids] = .03
    patch = {"edge_ids": np.asarray(patch_ids, np.int64),
             "pre_ids": np.asarray(patch_pre, np.int64),
             "post_ids": np.asarray(patch_post, np.int64),
             "source_values": np.full(25, .001, np.float32),
             "target_values": np.full(25, .03, np.float32)}
    assert len(patch_ids) == 25
    return before, after, patch


def fixture() -> dict:
    rows, motors, muscles, neurons = 301, 391, 96, 158706
    roster = motor_roster()
    columns = np.asarray(e253.MOTOR_COLUMNS)
    target = np.asarray(e253.MOTOR_IDS)
    bridge = np.asarray(e253.BRIDGE_IDS)
    source = np.arange(500, 523, dtype=np.int64)
    assert len(np.unique(np.r_[source, bridge, roster])) == 418
    member = np.arange(motors, dtype=np.int64) % muscles
    member[member == 41] = 40
    member[columns] = 41
    pool = np.zeros((muscles, motors), np.float64)
    counts = np.bincount(member, minlength=muscles)
    pool[member, np.arange(motors)] = 1/counts[member]
    moment = np.zeros((42, muscles), np.float64)
    moment[17, 41] = -3
    groups = [{"columns": np.flatnonzero(member == m).tolist(),
               "leg": "lh" if m == 41 else "lf",
               "muscle": "femur_reductor_muscle" if m == 41 else f"other_{m}"}
              for m in range(muscles)]
    before, after, patch = csr_fixture(neurons, source, bridge, target)
    selection = {"muscle_index": 41, "muscle": groups[41],
                 "motor_ids": target.tolist(), "bridge_ids": bridge.tolist()}
    reachable = roster[:55]
    route = {"rule": "generated fixed selection", "selected": {
        "port": 85, "kind": "claw", "reachable_count": 55,
        "source_rows": source.tolist(), "bridge_rows": bridge.tolist(),
        "reachable_motor_rows": reachable.tolist()}}
    result = {"calibration": {"motor_ids": roster, "membership": member,
                              "pool": pool, "moment": moment, "rest": np.zeros(42),
                              "ltm": np.arange(6, dtype=np.int64)},
              "muscles": {"muscles": groups}, "selection": selection,
              "route_selection": route, "edge_patch": patch,
              "baseline_csr": before, "candidate_csr": after}
    slopes = np.zeros((muscles, motors, 3, 2), np.float64)
    slopes[41, columns, :, 0] = 1
    slopes[41, columns, :, 1] = .5
    slopes[41, columns[0], 0, 1] = 2
    kappa = np.full(motors, .5)
    tick = np.arange(rows, dtype=np.int64)
    for arm in ("baseline", "candidate"):
        active = arm == "candidate"
        bank = np.zeros((rows, motors, 3))
        if active:
            bank[1:, columns[0], 0] = 1
        activation = np.zeros((rows, muscles))
        if active:
            activation[1:, 41] = 1.5
        command = .02*activation @ moment.T
        motor_rate = np.zeros((rows, motors), np.float32)
        activity_rate = np.zeros((rows, 82), np.float32)
        if active:
            activity_rate[1:, 23] = 2
        decoder = {"output_mask": np.ones(neurons, np.float32),
                   "tau": np.asarray([.005, .01, .04]),
                   "kappa": kappa.copy(), "slopes": slopes.copy(),
                   "membership": member.copy(),
                   "fixed_muscles": np.arange(10, dtype=np.int64)}
        result[arm] = {"activity": {"observed_ids": np.r_[source, bridge, reachable],
                                    "source_ids": source.copy(),
                                    "bridge_ids": bridge.copy(),
                                    "reachable_motor_ids": reachable.copy(),
                                    "rate": activity_rate},
                       "control": {"control_tick": tick.copy(),
                                   "network_tick": tick*20,
                                   "physics_time_s": tick*.005,
                                   "motor_rate32": motor_rate,
                                   "activation": activation,
                                   "command": command},
                       "recruitment": {"control_tick": tick.copy(),
                                       "network_tick": tick*20,
                                       "physics_time_s": tick*.005,
                                       "complete": np.ones(rows, bool),
                                       "bank_available": np.ones(rows, bool),
                                       "activation_available": np.ones(rows, bool),
                                       "motor_rate_available": np.ones(rows, bool),
                                       "bank": bank,
                                       "activation": activation.copy(),
                                       "motor_rate": motor_rate.astype(np.float64)},
                       "decoder": decoder}
    return result


def analyze(inputs, *, raw=None, progress=None):
    return e253.analyze_arrays(inputs, raw=raw, progress=progress)


def test_exact_decoder_hinge_and_concentration():
    report, raw = analyze(fixture())
    assert report["H253_single_observed_route"] is True
    assert report["motor_concentration"]["winner_id"] == e253.MOTOR_IDS[0]
    assert report["bridge_concentration"]["winner_id"] == e253.BRIDGE_IDS[0]
    np.testing.assert_allclose(raw["mn_activation_delta"][1:, 0], 1.5)
    assert raw["slope_high"][0, 0] > raw["slope_low"][0, 0]
    assert raw["slope_high"][1, 0] < raw["slope_low"][1, 0]
    assert raw["hinge_candidate"][1, 0, 0]
    np.testing.assert_allclose(raw["joint17_mn_command_delta"][1:, 0], -.09)
    assert report["cases"]["64"]["mn_activation_delta"][0] == 1.5


def test_observed_and_unobserved_current_are_distinct():
    inputs = fixture()
    inputs["baseline"]["decoder"]["output_mask"][e253.BRIDGE_IDS[0]] = 0
    inputs["candidate"]["decoder"]["output_mask"][e253.BRIDGE_IDS[0]] = 0
    report, raw = analyze(inputs)
    assert report["H253_single_observed_route"] is False
    assert report["bridge_concentration"]["available"] is False
    missing = ~raw["incoming_observed"]
    assert missing.any()
    assert np.isnan(raw["incoming_current_delta"][:, missing]).all()
    assert raw["incoming_observed_column"][missing].min() == -1
    assert report["coverage"][str(e253.MOTOR_IDS[0])]["unobserved_edges"] > 0
    assert report["observed_subtotal_is_full_recurrent_current"] is False


def test_tie_break_and_zero_denominator():
    tied = e253._concentration(np.array([2., 2., 0., 0., 0.]),
                                np.array([90, 10, 30, 40, 50]))
    assert tied["winner_id"] == 10
    empty = e253._concentration(np.zeros(4), np.array([7, 6, 5, 4]))
    assert empty["available"] is False and empty["winner_fraction"] is None
    inputs = fixture()
    inputs["candidate"]["recruitment"]["bank"][:] = 0
    inputs["candidate"]["control"]["activation"][:] = 0
    inputs["candidate"]["recruitment"]["activation"][:] = 0
    inputs["candidate"]["control"]["command"][:] = 0
    report, _ = analyze(inputs)
    assert report["motor_concentration"]["available"] is False
    assert report["bridge_concentration"]["available"] is False
    assert report["H253_single_observed_route"] is False


def test_bridge_stage_does_not_switch_to_another_motor():
    inputs = fixture()
    selected_id = e253.MOTOR_IDS[0]
    for arm in ("baseline_csr", "candidate_csr"):
        csr = inputs[arm]
        start, stop = csr["indptr"][selected_id:selected_id + 2]
        incoming = csr["indices"][start:stop]
        bridge_edge = np.isin(incoming, e253.BRIDGE_IDS)
        csr["data"][start:stop][bridge_edge] = 0
    report, raw = analyze(inputs)
    assert report["motor_concentration"]["winner_id"] == selected_id
    assert report["bridge_concentration"]["available"] is False
    assert report["H253_single_observed_route"] is False
    assert np.all(raw["bridge_direct_delta"][:, 0] == 0)


def test_permuted_source_graph_id_order_keeps_bridge_current():
    inputs = fixture()
    _, reference = analyze(inputs)
    selected = inputs["route_selection"]["selected"]
    permutation = np.arange(23)[::-1]
    selected["source_rows"] = np.asarray(selected["source_rows"])[permutation].tolist()
    for arm in ("baseline", "candidate"):
        activity = inputs[arm]["activity"]
        activity["source_ids"] = activity["source_ids"][permutation]
        activity["observed_ids"][:23] = activity["source_ids"]
        activity["rate"][:, :23] = activity["rate"][:, permutation]
    _, reordered = analyze(inputs)
    np.testing.assert_array_equal(reordered["bridge_direct_delta"],
                                  reference["bridge_direct_delta"])


def test_incoming_row_change_and_partial_failure():
    inputs = fixture()
    target = e253.MOTOR_IDS[0]
    edge = int(inputs["candidate_csr"]["indptr"][target])
    inputs["candidate_csr"]["data"][edge] += .2
    with pytest.raises(ValueError, match="outside exact 25"):
        analyze(inputs)
    inputs = fixture()
    raw = {}
    def progress(stage):
        if stage == "decoder":
            raise TimeoutError("generated checkpoint")
    with pytest.raises(TimeoutError, match="generated checkpoint"):
        analyze(inputs, raw=raw, progress=progress)
    assert "bank_contribution_delta" in raw
    assert "incoming_edge_id" not in raw


def test_late_publication_revokes_nested_support_and_latest_manifest(tmp_path: Path,
                                                                      monkeypatch):
    result = {"validity": "valid_saved_diagnostic", "H253_single_observed_route": True,
              "analysis": {"H253_single_observed_route": True,
                           "motor_concentration": {"supported": True},
                           "bridge_concentration": {"supported": True}},
              "finalization_pending": True}
    original = e253.publish
    attempts = 0
    def late_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        original(*args, **kwargs)
        if attempts == 1:
            raise OSError("generated late failure")
    monkeypatch.setattr(e253, "publish", late_once)
    error = e253.settle(tmp_path, result, {"x": np.array([1.])},
                        {"errors": []}, lambda: 1., float("inf"))
    assert isinstance(error, OSError)
    assert result["H253_single_observed_route"] is None
    assert result["analysis"]["H253_single_observed_route"] is None
    assert result["analysis"]["motor_concentration"]["supported"] is None
    saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert saved["validity"] == "incomplete_or_invalid"
    manifest = json.loads((tmp_path / "output-manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]["errors.json"]["sha256"] == e253.sha(tmp_path / "errors.json")


def test_hash_gate_before_original_arrays(tmp_path: Path):
    fake = tmp_path / "fake-freeze.json"
    fake.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="freeze hash"):
        e253.gate(str(fake), "0"*64, str(fake), "0"*64)
