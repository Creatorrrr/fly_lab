"""Generated-only E249 saved endpoint attribution checks."""

import json

import numpy as np
import pytest

from tools import analyze_c_banc_m3_lh_input_slope as e249


def fixture():
    n = 12
    source = [1, 2]
    bridge = [3, 4, 5, 6]
    motor = [7]
    # Declared W[post,pre], including a masked edge and a zero-weight edge.
    edges = [(3, 0, 0), (3, 1, 10), (3, 2, -10),
             (4, 1, 5), (4, 2, -5), (5, 1, 5), (5, 2, -5),
             (6, 1, 1), (6, 2, -1),
             (7, 0, -2), (7, 1, 1), (7, 3, 2)]
    ptr = np.zeros(n+1, np.int64)
    for post, *_ in edges:
        ptr[post+1] += 1
    ptr = np.cumsum(ptr)
    csr = {"indptr": ptr, "indices": np.asarray([pre for _, pre, _ in edges]),
           "data": np.asarray([weight for _, _, weight in edges], np.float32)}
    states = {}
    for name in e249.STATES:
        rate = np.zeros(n, np.float32)
        rate[1:3] = 2 if name == "intact_high" else 1
        mask = np.ones(n, np.float32)
        mask[0] = 0
        drive = np.zeros(n, np.float32)
        drive[3:6] = 500
        drive[6] = 7.5
        states[name] = {"rate": rate, "output_mask": mask,
                        "drive": drive, "adaptation": np.zeros(n, np.float32),
                        "tick": 40000 if name == "baseline" else 40800}
    configuration = {"membrane_tau": np.full(n, .3)}
    return csr, configuration, states, source, bridge, motor


def test_full_csr_groups_mask_sign_cancellation_and_kink():
    csr, configuration, states, source, bridge, motor = fixture()
    summary, raw = e249.analyze_arrays(csr, configuration, states, source,
                                        bridge, motor, strict_original=False)
    assert raw["row_ids"].tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert raw["row_group"].tolist() == [0, 0, 1, 1, 1, 1, 2]
    row3 = np.flatnonzero(raw["edge_post"] == 3)
    assert raw["edge_id"][row3].tolist() == [0, 1, 2]
    assert raw["edge_pre"][row3].tolist() == [0, 1, 2]
    assert raw["edge_native_weight"][row3].tolist() == [0., 10., -10.]
    assert raw["edge_group"][row3].tolist() == [2, 0, 0]
    row7 = np.flatnonzero(raw["edge_post"] == 7)
    assert raw["edge_group"][row7].tolist() == [2, 0, 1]
    assert raw["signed"][1, 2].tolist() == [0., 0., 0.]
    assert raw["positive"][1, 2, 0] == 10
    assert raw["negative"][1, 2, 0] == -10
    assert raw["gross"][1, 2, 0] == 20
    assert raw["margin"][1, 5] == 0
    assert raw["slope"][1, 5] == 0  # exact rectification kink
    assert raw["rectified"][1, 5] and not raw["saturated"][1, 5]
    assert raw["completed_states"] == 8
    assert np.all(np.isfinite(raw["warp_bridge_current32"]))
    assert summary["H249_saturation"] is True
    assert summary["H249_small_direct_input"] is True
    assert raw["cancellation_zero_gross"].tolist() == [False]*4
    assert np.all(raw["cancellation_net_to_gross"] == 0)


def test_hypotheses_independent_and_observed_rate_not_static_phi():
    csr, configuration, states, source, bridge, motor = fixture()
    states["intact_high"]["rate"][2] = 1
    high_input, high_raw = e249.analyze_arrays(csr, configuration, states,
        source, bridge, motor, strict_original=False)
    assert high_input["H249_saturation"] is True
    assert high_input["H249_small_direct_input"] is False
    assert high_raw["cancellation_signed"][0] == 10
    assert high_raw["cancellation_net_to_gross"][0] == 1
    assert high_raw["contrast_rate_delta"][0] == 0
    assert high_raw["contrast_phi_delta"][0] > 0

    csr, configuration, states, source, bridge, motor = fixture()
    states["intact_low"]["drive"][4] = 0
    states["intact_high"]["drive"][4] = 0
    low_slope, _ = e249.analyze_arrays(csr, configuration, states,
        source, bridge, motor, strict_original=False)
    assert low_slope["H249_saturation"] is False
    assert low_slope["H249_small_direct_input"] is True


def test_partial_saved_state_prefix_and_zero_gross():
    csr, configuration, states, source, bridge, motor = fixture()
    raw = {}

    def fail(_name, completed):
        if completed == 1:
            raise RuntimeError("generated partial")

    with pytest.raises(RuntimeError, match="generated partial"):
        e249.analyze_arrays(csr, configuration, states, source, bridge,
                            motor, raw=raw, progress=fail, strict_original=False)
    assert raw["completed_states"] == 1
    assert np.isfinite(raw["signed"][0]).all()
    assert np.isnan(raw["signed"][1:]).all()

    csr, configuration, states, source, bridge, motor = fixture()
    states["intact_high"]["rate"][1:3] = states["intact_low"]["rate"][1:3]
    _, raw = e249.analyze_arrays(csr, configuration, states, source,
                                  bridge, motor, strict_original=False)
    assert raw["cancellation_zero_gross"].all()
    assert np.array_equal(raw["cancellation_net_to_gross"], np.zeros(4))


def test_late_publication_retracts_both_summary_and_result(tmp_path, monkeypatch):
    summary = {"H249_saturation": True, "H249_small_direct_input": False,
               "source_bridge_delta_rms_normalized_cap200": .2}
    result = {"validity": "valid_saved_endpoint_diagnostic",
              "H249_saturation": True, "H249_small_direct_input": False,
              "summary": summary, "verdict_status": "tested_saved_endpoint_only"}
    ledger = {"errors": []}
    actual = e249.publish
    calls = 0

    def fault(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("generated late receipt")
        return actual(*args, **kwargs)

    monkeypatch.setattr(e249, "publish", fault)
    first = e249.settle(tmp_path, result, summary, {"x": np.array([2])},
        {"nodes": []}, ledger, elapsed=lambda: .1,
        deadline=e249.time.perf_counter()+10)
    assert isinstance(first, OSError)
    final = json.loads((tmp_path / "result.json").read_bytes())
    standalone = json.loads((tmp_path / "summary.json").read_bytes())
    assert final["H249_saturation"] is None
    assert final["H249_small_direct_input"] is None
    assert final["summary"]["H249_saturation"] is None
    assert standalone["H249_small_direct_input"] is None
    assert standalone["source_bridge_delta_rms_normalized_cap200"] == .2
    manifest = json.loads((tmp_path / "output-manifest.json").read_bytes())
    for name, record in manifest["files"].items():
        assert e249.prior.sha(tmp_path / name) == record["sha256"]

    def persistent(*_args, **_kwargs):
        raise OSError("generated persistent publication")

    stuck = tmp_path / "stuck"
    stuck.mkdir()
    e249.prior.save_json(stuck / "summary.json", {"H249_saturation": True})
    e249.prior.save_json(stuck / "output-manifest.json", {"stale": True})
    monkeypatch.setattr(e249, "publish", persistent)
    nested = {"H249_saturation": True, "H249_small_direct_input": True,
              "source_bridge_delta_rms_normalized_cap200": .03}
    result2 = {"H249_saturation": True, "H249_small_direct_input": True,
               "summary": nested, "validity": "valid_saved_endpoint_diagnostic"}
    e249.settle(stuck, result2, nested, {}, {}, {"errors": []},
        elapsed=lambda: .1, deadline=e249.time.perf_counter()+10)
    assert not (stuck / "output-manifest.json").exists()
    assert json.loads((stuck / "summary.json").read_bytes())["H249_saturation"] is None
    assert json.loads((stuck / "result.json").read_bytes())["finalization_pending"] is True


def test_hash_gate_before_checkpoint_load(monkeypatch, tmp_path):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("checkpoint opened")

    monkeypatch.setattr(e249, "load", forbidden)
    with pytest.raises(ValueError, match="freeze SHA"):
        e249.run(tmp_path / "absent", "wrong", tmp_path / "review", "wrong",
                 tmp_path / "generated-out", generated_test_output=True)
    receipt = json.loads((tmp_path / "generated-out" / "result.json").read_bytes())
    assert receipt["loaded_input_roles"] == []
    assert receipt["original_calculation_count"] == 0
