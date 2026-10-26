"""Generated E259 decoder, observed CSR, and publication checks."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from tools import analyze_c_banc_m3_lh_extensor_input as e259


def fixture() -> dict:
    n, motors, muscles, rows = 158706, 391, 96, 301
    cols = np.array(e259.MOTOR_COLUMNS)
    ids = np.array(e259.MOTOR_IDS)
    bridges = np.array(e259.BRIDGE_IDS)
    anchors = [(-1, -100), *zip(e259.MOTOR_COLUMNS, e259.MOTOR_IDS, strict=True),
               (391, n)]
    roster = np.empty(motors, np.int64)
    for (left_col, left_id), (right_col, right_id) in pairwise(anchors):
        for col in range(left_col+1, min(right_col+1, motors)):
            roster[col] = left_id + (right_id-left_id)*(col-left_col)//(right_col-left_col)
    assert np.array_equal(np.sort(roster), roster)
    source = np.arange(500, 523, dtype=np.int64)
    core = np.array([1000+3*i for i in range(22)] + [bridges[1], bridges[3]], np.int64)
    core.sort()
    extra = core[~np.isin(core, bridges)]
    member = np.arange(motors, dtype=np.int64) % muscles
    member[member == 44] = 43
    member[cols] = 44
    pool = np.zeros((muscles, motors))
    pool[member, np.arange(motors)] = 1/np.bincount(member, minlength=muscles)[member]
    moment = np.zeros((42, muscles))
    moment[19, 44] = 1.8088900095777785
    moment[19, 43] = 3  # Other muscle shares joint19.
    groups = [{"columns": np.flatnonzero(member == j).tolist(),
               "leg": "lh" if j == 44 else "lf",
               "muscle": "tibia_extensor_muscle" if j == 44 else f"other_{j}"}
              for j in range(muscles)]
    edges = {int(bridges[2]): [(int(s), .125, True) for s in source[:6]]}
    for motor in ids:
        edges[int(motor)] = [(int(b), 1., False) for b in bridges]
        edges[int(motor)] += [(int(source[0]), -.5, False),
                              (int(roster[0]), .25, False),
                              (int(extra[0]), -.75, False), (777, 2., False)]
    ptr = np.zeros(n+1, np.int32)
    pres, weights, patch_edges = [], [], []
    for post in range(n):
        for pre, weight, patched in edges.get(post, []):
            if patched:
                patch_edges.append(len(pres))
            pres.append(pre)
            weights.append(weight)
        ptr[post+1] = len(pres)
    base_csr = {"indptr": ptr, "indices": np.array(pres, np.int32),
                "data": np.array(weights, np.float32)}
    cand_csr = {key: value.copy() for key, value in base_csr.items()}
    cand_csr["data"][patch_edges] = np.float32(.25)
    result = {"calibration": {"motor_ids": roster, "membership": member,
                               "pool": pool, "moment": moment, "rest": np.zeros(42),
                               "ltm": np.arange(6, dtype=np.int64)},
              "muscles": {"muscles": groups},
              "selection": {"muscle_index": 44, "muscle": groups[44],
                            "motor_ids": ids.tolist(), "bridge_ids": bridges.tolist(),
                            "joint_column": 19, "hypothesized_bridge": int(bridges[2]),
                            "moment_coefficient": float(moment[19, 44])},
              "route_selection": {"rule": "generated", "selected": {
                  "port": 85, "kind": "claw", "reachable_count": 55,
                  "source_rows": source.tolist(), "bridge_rows": bridges.tolist(),
                  "reachable_motor_rows": roster[:55].tolist()}},
              "edge_patch": {"edge_ids": np.array(patch_edges, np.int64),
                             "pre_ids": source[:6].copy(),
                             "post_ids": np.full(6, bridges[2], np.int64),
                             "source_values": np.full(6, float(np.float32(.125))),
                             "target_values": np.full(6, float(np.float32(.25)))},
              "baseline_csr": base_csr, "candidate_csr": cand_csr,
              "core_roster": {"original_24_node_rows": core.tolist()},
              "observation_roster": {"original24": core.tolist(),
                                     "additional_core_ids": extra.tolist()},
              "e258_result": {"validity": "valid_saved_diagnostic",
                              "finalization_pending": False, "H258_same_muscle": False,
                              "analysis": {"H258_same_muscle": False,
                                  "first_new_violation": {"control": 56, "joint_column": 19},
                                  "maximum_candidate_error": {"control": 70,
                                                              "joint_column": 19}}},
              "e258_output_review": {"all_passed": True, "H258_same_muscle": False},
              "e258_final_review": {"validity": "valid_saved_diagnostic",
                                    "all_passed": True, "H258_same_muscle": False},
              "e257_output_review": {"all_passed": True},
              "e257_final_review": {"all_passed": True, "H257_supported": False},
              "e258_supervisor": {"exit_code": 0}}
    slopes = np.zeros((muscles, motors, 3, 2))
    slopes[44, cols, :, 0] = 1
    slopes[44, cols, :, 1] = .5
    slopes[44, cols[0], 0, 1] = 2
    tick = np.arange(rows, dtype=np.int64)
    for arm in ("baseline", "candidate"):
        is_candidate = arm == "candidate"
        bank = np.zeros((rows, motors, 3))
        bank[1:, cols[0], 0] = 0 if is_candidate else 1
        bank[1:, cols[1], 0] = .25 if is_candidate else .5
        own = bank[:, cols]
        low = slopes[44, cols, :, 0]
        high = slopes[44, cols, :, 1]
        value = low[None]*np.minimum(own, .5)+high[None]*np.maximum(own-.5, 0)
        activation = np.zeros((rows, muscles))
        activation[:, 44] = value.sum(axis=(1, 2))
        rate = np.zeros((rows, 82), np.float32)
        if not is_candidate:
            rate[1:, 25] = 2
        motor_rate = np.zeros((rows, motors), np.float32)
        core_rate = np.zeros((rows, 24), np.float32)
        core_rate[:, np.flatnonzero(core == bridges[1])[0]] = rate[:, 24]
        core_rate[:, np.flatnonzero(core == bridges[3])[0]] = rate[:, 26]
        clock = {"control_tick": tick.copy(), "network_tick": tick*20,
                 "physics_time_s": tick*.005}
        result[arm] = {
            "activity": {"observed_ids": np.r_[source, bridges, roster[:55]],
                         "source_ids": source.copy(), "bridge_ids": bridges.copy(),
                         "reachable_motor_ids": roster[:55].copy(),
                         "rate": rate, "drive": np.zeros_like(rate),
                         "adaptation": np.zeros_like(rate)},
            "core": {"core_ids": core.copy(), **clock, "rate": core_rate,
                     "drive": np.zeros_like(core_rate),
                     "adaptation": np.zeros_like(core_rate)},
            "control": {**clock, "motor_rate32": motor_rate,
                        "activation": activation, "command": .02*activation @ moment.T},
            "recruitment": {**clock, "complete": np.ones(rows, bool),
                            "bank_available": np.ones(rows, bool),
                            "activation_available": np.ones(rows, bool),
                            "motor_rate_available": np.ones(rows, bool),
                            "bank": bank, "activation": activation.copy(),
                            "motor_rate": motor_rate.astype(np.float64)},
            "decoder": {"output_mask": np.ones(n, np.float32),
                        "tau": np.array([.005, .01, .04]),
                        "kappa": np.full(motors, .5), "slopes": slopes.copy(),
                        "membership": member.copy(),
                        "fixed_muscles": np.arange(10, dtype=np.int64)}}
    difference = (result["candidate"]["control"]["activation"]
                  - result["baseline"]["control"]["activation"])
    contribution = np.zeros((rows, 42, muscles))
    contribution[:, 19, 44] = .02*moment[19, 44]*difference[:, 44]
    result["e258_raw"] = {"muscle_joint_contribution": contribution,
                          "difference_activation96": difference}
    return result


def test_negative_route_and_exact_muscle_projection():
    report, raw = e259.analyze_arrays(fixture())
    assert report["H259_single_inhibitory_route"] is True
    assert report["motor_concentration"]["winner_id"] == e259.MOTOR_IDS[0]
    assert report["bridge_concentration"]["fixed_bridge_fraction"] == 1
    assert report["cases"]["56"]["joint19_mn_command_delta"][0] < 0
    assert raw["category_current_delta"].shape == (301, 2, 7)
    assert np.isnan(raw["secant_slope"][0]).all()


def test_hinge_tiny_delta_and_zero_bank_secant():
    inputs = fixture()
    bank = inputs["candidate"]["recruitment"]["bank"]
    bank[2, e259.MOTOR_COLUMNS[0], 0] = 1-1e-12
    bank[3, e259.MOTOR_COLUMNS[0], 0] = .5+1e-12
    activation = inputs["candidate"]["control"]["activation"]
    activation[2, 44] += 1.5-2e-12
    activation[3, 44] += .5+2e-12
    inputs["candidate"]["recruitment"]["activation"] = activation.copy()
    inputs["candidate"]["control"]["command"] = .02*activation @ inputs["calibration"]["moment"].T
    difference = activation-inputs["baseline"]["control"]["activation"]
    inputs["e258_raw"]["difference_activation96"] = difference
    inputs["e258_raw"]["muscle_joint_contribution"][:, 19, 44] = (
        .02*inputs["calibration"]["moment"][19, 44]*difference[:, 44])
    _, raw = e259.analyze_arrays(inputs)
    assert raw["secant_available"][2, 0, 0]
    assert raw["secant_slope"][2, 0, 0] == 2
    assert raw["secant_available"][3, 0, 0]
    assert 1 <= raw["secant_slope"][3, 0, 0] <= 2
    assert not raw["secant_available"][0].any()


def test_zero_denominator_tie_and_unobserved_nan():
    assert e259._concentration(np.array([2., 2.]), np.array([90, 10]))["winner_id"] == 10
    inputs = fixture()
    inputs["candidate"]["recruitment"]["bank"] = inputs["baseline"]["recruitment"]["bank"].copy()
    inputs["candidate"]["control"]["activation"] = inputs["baseline"]["control"]["activation"].copy()
    inputs["candidate"]["recruitment"]["activation"] = inputs["baseline"]["control"]["activation"].copy()
    inputs["e258_raw"]["difference_activation96"][:] = 0
    inputs["e258_raw"]["muscle_joint_contribution"][:] = 0
    report, raw = e259.analyze_arrays(inputs)
    assert report["H259_single_inhibitory_route"] is False
    assert report["motor_concentration"]["available"] is False
    assert np.isnan(raw["incoming_current_delta"][:, ~raw["incoming_observed"]]).all()
    assert report["coverage"][str(e259.MOTOR_IDS[0])]["unobserved_edges"] > 0


def test_fixed_bridge_does_not_switch():
    inputs = fixture()
    for arm in ("baseline_csr", "candidate_csr"):
        csr = inputs[arm]
        start, stop = csr["indptr"][e259.MOTOR_IDS[0]:e259.MOTOR_IDS[0]+2]
        csr["data"][start:stop][csr["indices"][start:stop] == e259.FIXED_BRIDGE] = 0
    report, _ = e259.analyze_arrays(inputs)
    assert report["motor_concentration"]["available"] is True
    assert report["bridge_concentration"]["supported"] is False
    assert report["H259_single_inhibitory_route"] is False


def test_float64_rate_subtraction_and_patch_roundtrip():
    inputs = fixture()
    inputs["baseline"]["core"]["rate"][1, 0] = np.float32(1e-5)
    inputs["candidate"]["core"]["rate"][1, 0] = np.float32(200)
    _, raw = e259.analyze_arrays(inputs)
    expected = float(np.float32(200))-float(np.float32(1e-5))
    assert raw["observed_rate_delta"][1, 418] == expected
    inputs["edge_patch"]["source_values"][0] += 1e-10
    with pytest.raises(ValueError, match="exact f32 roundtrip"):
        e259.validate(inputs)


def test_partial_raw_and_six_edge_provenance():
    inputs = fixture()
    target = e259.MOTOR_IDS[0]
    inputs["candidate_csr"]["data"][inputs["candidate_csr"]["indptr"][target]] += .2
    with pytest.raises(ValueError, match="outside exact six"):
        e259.analyze_arrays(inputs)
    raw = {}
    def stop(stage):
        if stage == "decoder":
            raise TimeoutError("generated stop")
    with pytest.raises(TimeoutError, match="generated stop"):
        e259.analyze_arrays(fixture(), raw=raw, progress=stop)
    assert "bank_contribution_delta" in raw and "incoming_edge_id" not in raw


def test_late_publication_revokes_nested_support(tmp_path: Path, monkeypatch):
    result = {"validity": "valid_saved_diagnostic", "H259_single_inhibitory_route": True,
              "analysis": {"H259_single_inhibitory_route": True,
                           "motor_concentration": {"supported": True},
                           "bridge_concentration": {"supported": True}},
              "finalization_pending": True}
    original = e259.publish
    attempts = 0
    def late_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        original(*args, **kwargs)
        if attempts == 1:
            raise OSError("generated late failure")
    monkeypatch.setattr(e259, "publish", late_once)
    error = e259.settle(tmp_path, result, {"x": np.array([1.])},
                        {"errors": []}, lambda: 1., float("inf"))
    assert isinstance(error, OSError)
    assert result["H259_single_inhibitory_route"] is None
    assert result["analysis"]["H259_single_inhibitory_route"] is None
    saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert saved["validity"] == "incomplete_or_invalid"
    manifest = json.loads((tmp_path / "output-manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]["errors.json"]["sha256"] == e259.sha(tmp_path / "errors.json")


def test_hash_gate_before_arrays(tmp_path: Path):
    fake = tmp_path / "fake-freeze.json"
    fake.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="freeze hash"):
        e259.gate(str(fake), "0"*64, str(fake), "0"*64)


def test_exact_run_path_before_any_directory_or_input(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(e259, "BASE", tmp_path)
    mkdir_calls = []

    class ReachedMkdir(Exception):
        pass

    def stop_before_write(path, *args, **kwargs):
        mkdir_calls.append(path)
        raise ReachedMkdir

    monkeypatch.setattr(Path, "mkdir", stop_before_write)
    intended = tmp_path / "e259-lh-extensor-input"
    with pytest.raises(ReachedMkdir):
        e259.run("", "", "", "", str(intended))
    assert mkdir_calls == [intended]
    for other in ("E259-lh-reductor-input", "e259-lh-reductor-input",
                  "e259-lh-extensor-input-other"):
        with pytest.raises(ValueError, match="Exact E259 write-once output path"):
            e259.run("", "", "", "", str(tmp_path / other))
    assert mkdir_calls == [intended]
    assert not intended.exists()
