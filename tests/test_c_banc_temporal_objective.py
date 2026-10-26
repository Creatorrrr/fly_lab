"""Generated-only CPU checks for the frozen saved-characterization CLI."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import analyze_c_banc_m3_temporal_objective as subject
from tools.banc_m3_phase_free_temporal import PhaseFreeTemporalObjective


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def reference():
    times = np.arange(1, 2001, dtype=np.float64) * .005
    global_phase = 2 * np.pi * 2.382760007990865 * times
    phases = global_phase[:, None] + np.arange(6)[None, :] * .31
    command = np.zeros((2000, 42))
    for leg in range(6):
        command[:, 7 * leg] = (.5 + leg * .1) * np.cos(phases[:, leg])
        command[:, 7 * leg + 1] = (.6 + leg * .08) * np.sin(phases[:, leg])
        command[:, 7 * leg + 2:7 * leg + 7] = .03 * leg
    return {"time_s": times, "phase": phases, "command": command}


def raw_saved(curves, shift=0):
    p = np.repeat((.137 + np.arange(301) * subject.FREQUENCY_HZ * .005)[:, None], 6, axis=1)
    p[:, [0, 3]] += shift
    return {"control_tick": np.arange(301, dtype=np.int64),
            "network_tick": np.arange(301, dtype=np.int64) * 20,
            "physics_time_s": np.arange(301, dtype=np.float64) * .005,
            "command": subject.numpy_interpolate(curves, p), "tendon": np.zeros((301, 6))}


def generated_contract(folder):
    ref = reference()
    curves = subject.curve128(ref)
    arrays = {"reference": ref,
              "startup": {"target_command": np.zeros((99, 42)), "target_tendon": np.zeros((99, 6))},
              "neutral": raw_saved(curves), "rejected": raw_saved(curves, .125)}
    inputs = {}
    for role, array in arrays.items():
        path = folder / (role + ".npz")
        np.savez(path, **array)
        inputs[role] = {"path": path.name, "sha256": subject.sha256(path)}
    contract = {"schema": subject.SCHEMA, "inputs": inputs,
                "source_sha256": {name: subject.sha256(subject.ROOT / name) for name in subject.OWNED_SOURCES}}
    path = folder / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf8")
    return path, contract


def test_exact_e319_builder_lf_sorted_periodic_interpolation():
    ref = reference()
    actual = subject.curve128(ref)
    mask = ref["time_s"] >= 5
    phase = (ref["phase"][mask, 0] / (2 * np.pi)) % 1
    indices = np.argsort(phase)
    phase = phase[indices]
    values = ref["command"][mask][indices]
    expected = np.empty((6, 128, 7))
    extended_phase = np.concatenate(([phase[-1] - 1], phase, [phase[0] + 1]))
    for leg in range(6):
        for joint in range(7):
            column = values[:, leg * 7 + joint]
            expected[leg, :, joint] = np.interp(np.arange(128) / 128, extended_phase,
                                               np.concatenate(([column[-1]], column, [column[0]])))
    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.c_contiguous
    shifted = {key: value.copy() for key, value in ref.items()}
    shifted["phase"][:, 1:] += 1.5
    np.testing.assert_array_equal(subject.curve128(shifted), actual)


def test_reference_qualification_fixed_real_shape_roundtrip_counterexamples():
    curves = subject.curve128(reference())
    obj = PhaseFreeTemporalObjective(curves, np.zeros((99, 42)), np.zeros((99, 6)))
    result = subject.qualify_reference(obj)
    assert result["pass"]
    assert all(result["gates"].values())
    assert result["sample_count"] == 260
    assert result["successor_round_trip_max_rad"] <= 1e-10
    assert result["phase_round_trip_max_cycle"] <= 1e-8
    assert result["scores"]["one_lr_pair_shift_0_125_cycle"]["coordination_loss"] > .1


def test_numpy_tiny_negative_and_subnormal_seam_uses_direct_anchor_expectation():
    curves = subject.curve128(reference())
    values = np.array([-1e-20, np.nextafter(0., -np.inf), -0., 0.,
                       np.nextafter(0., np.inf)], dtype=np.float64)
    phases = np.repeat(values[:, None], 6, axis=1)
    actual = subject.numpy_interpolate(curves, phases)
    expected = np.repeat(curves[:, 0].reshape(1, 42), len(values), axis=0)
    expected[-1] += (values[-1] * 128 * (curves[:, 1] - curves[:, 0])).reshape(42)
    np.testing.assert_array_equal(actual, expected)
    assert np.isfinite(actual).all()


def test_adjacent_endpoints_and_seam_are_not_nonlocal_but_repeated_loop_is():
    curves = subject.curve128(reference())
    obj = PhaseFreeTemporalObjective(curves, np.zeros((99, 42)), np.zeros((99, 6)))
    p = np.repeat(np.array([.9997, 0., .0003])[:, None], 6, axis=1)
    q = obj.interpolate(torch.tensor(p))
    result = subject.projection_diagnostics(obj.project(q))
    assert result["nonlocal_tie_count"] == 0
    assert np.max(result["circular_jump_cycle"]) <= .0003 + 1e-12
    repeated = np.tile(curves[:, ::2], (1, 2, 1))
    ambiguous = PhaseFreeTemporalObjective(repeated, np.zeros((99, 42)), np.zeros((99, 6)))
    projection = ambiguous.project(torch.tensor(repeated[:, 10].reshape(1, 42)))
    diagnostic = subject.projection_diagnostics(projection)
    assert diagnostic["nonlocal_tie_count"] == 6
    assert all(diagnostic["nonlocal_tie"][0])
    assert not subject.qualify_reference(ambiguous)["pass"]
    json.dumps(diagnostic, allow_nan=False)


def test_generated_cli_finite_raw_scores_preserves_rejection_and_no_overwrite(tmp_path):
    path, _ = generated_contract(tmp_path)
    output = tmp_path / "analysis.json"
    subject.main(["--contract", str(path), "--output", str(output)])
    result = json.loads(output.read_text(encoding="utf8"))
    curves = subject.curve128(reference())
    curves_meta = result["reference_curves_metadata"]
    assert curves_meta["shape"] == [6, 128, 7]
    assert curves_meta["dtype"] == "float64"
    assert curves_meta["numpy_dtype"] == curves.dtype.str
    assert curves_meta["layout"] == "C"
    assert curves_meta["leg_order"] == list(subject.LEGS)
    assert curves_meta["joint_index_order"] == list(range(7))
    assert curves_meta["c_order_sha256"] == hashlib.sha256(curves.tobytes(order="C")).hexdigest()
    normalizer = np.array(result["successor_normalization_mse_rad"], dtype=np.float64)
    norm_meta = result["successor_normalization_metadata"]
    assert norm_meta["shape"] == [6] and norm_meta["dtype"] == "float64"
    assert norm_meta["leg_order"] == list(subject.LEGS)
    assert norm_meta["c_order_sha256"] == hashlib.sha256(normalizer.tobytes(order="C")).hexdigest()
    assert norm_meta["per_leg_f64_sha256"] == {
        name: hashlib.sha256(normalizer[i:i + 1].tobytes(order="C")).hexdigest()
        for i, name in enumerate(subject.LEGS)}
    assert result["reference_qualification"]["pass"]
    assert result["valid_input_nonlocal_identifiability_pass"]
    assert result["e330_input_qualification_pass"]
    assert result["arms"]["neutral"]["experiment"] == "E318"
    assert result["arms"]["rejected"]["experiment"] == "E327"
    assert result["arms"]["rejected"]["classification"] == "rejected"
    assert result["arms"]["neutral"]["steady_pair_inputs_projection"]["phase_proxy_cycle"]
    assert len(result["arms"]["neutral"]["steady_pair_inputs_projection"]["phase_proxy_cycle"]) == 201
    for name in ("original_calls", "model_calls", "body_calls", "CUDA_calls", "optimizer_steps"):
        assert result[name] == 0
    for name in ("biological_pass", "F1_F8_pass", "candidate_adopted", "natural_gait_success",
                 "e330_original_execution_authorized", "ranking_is_acceptance_gate"):
        assert result[name] is False
    before = output.read_bytes()
    with pytest.raises(ValueError, match="new path"):
        subject.main(["--contract", str(path), "--output", str(output)])
    assert output.read_bytes() == before


@pytest.mark.parametrize("kind", ["input_hash", "source_hash", "missing_source", "extra_input"])
def test_all_pins_before_any_numeric_read(tmp_path, monkeypatch, kind):
    path, contract = generated_contract(tmp_path)
    if kind == "input_hash":
        contract["inputs"]["rejected"]["sha256"] = "0" * 64
    elif kind == "source_hash":
        contract["source_sha256"][subject.OWNED_SOURCES[0]] = "0" * 64
    elif kind == "missing_source":
        del contract["source_sha256"][subject.OWNED_SOURCES[-1]]
    else:
        contract["inputs"]["unexpected"] = contract["inputs"]["reference"]
    path.write_text(json.dumps(contract), encoding="utf8")
    def forbidden(*args, **kwargs):
        pytest.fail("numeric load attempted before all hashes/contracts passed")
    monkeypatch.setattr(np, "load", forbidden)
    with pytest.raises(ValueError):
        subject.main(["--contract", str(path), "--output", str(tmp_path / "out.json")])


def test_contract_rejects_duplicate_json_keys_and_nonfinite_constants(tmp_path):
    path = tmp_path / "bad.json"
    for text in ('{"schema":1,"schema":2}', '{"schema":NaN}'):
        path.write_text(text, encoding="utf8")
        with pytest.raises(ValueError):
            subject.verify_contract(path)


def test_saved_clock_failure_precedes_any_reference_qualification(tmp_path, monkeypatch):
    path, contract = generated_contract(tmp_path)
    raw_path = tmp_path / "rejected.npz"
    raw = raw_saved(subject.curve128(reference()))
    raw["network_tick"][100] += 1
    np.savez(raw_path, **raw)
    contract["inputs"]["rejected"]["sha256"] = subject.sha256(raw_path)
    path.write_text(json.dumps(contract), encoding="utf8")
    def forbidden(*args, **kwargs):
        pytest.fail("reference analysis attempted before all saved clocks passed")
    monkeypatch.setattr(subject, "qualify_reference", forbidden)
    with pytest.raises(ValueError, match="network_tick"):
        subject.main(["--contract", str(path), "--output", str(tmp_path / "out.json")])


def test_npz_reads_required_arrays_only_and_rejects_missing_keys(tmp_path):
    path = tmp_path / "generated.npz"
    np.savez(path, command=np.zeros((1, 42)), unused_object=np.array([object()], dtype=object))
    assert set(subject.load_npz(path, ("command",))) == {"command"}
    with pytest.raises(ValueError, match="arrays missing"):
        subject.load_npz(path, ("missing",))


@pytest.mark.parametrize("kind", ["control", "network", "timestamp", "nan_time", "float_ticks",
                                  "q_dtype", "nan_q", "wrong_t_shape", "reordered"])
def test_saved_shape_finite_dtype_and_ordered_clocks(kind):
    raw = raw_saved(subject.curve128(reference()))
    if kind == "control":
        raw["control_tick"][10] = 9
    elif kind == "network":
        raw["network_tick"][10] += 1
    elif kind == "timestamp":
        raw["physics_time_s"][10] += .001
    elif kind == "nan_time":
        raw["physics_time_s"][10] = np.nan
    elif kind == "float_ticks":
        raw["control_tick"] = raw["control_tick"].astype(np.float64)
    elif kind == "q_dtype":
        raw["command"] = raw["command"].astype(np.float32)
    elif kind == "nan_q":
        raw["command"][100, 0] = np.inf
    elif kind == "wrong_t_shape":
        raw["tendon"] = raw["tendon"][:300]
    else:
        raw = {key: value[::-1].copy() for key, value in raw.items()}
    with pytest.raises(ValueError):
        subject.validate_saved(raw)


@pytest.mark.parametrize("kind", ["nan", "wrong_shape", "wrong_dt", "cadence", "empty_late"])
def test_reference_invalid_finite_clock_and_fixed_cadence(kind):
    ref = reference()
    if kind == "nan":
        ref["command"][0, 0] = np.nan
    elif kind == "wrong_shape":
        ref["phase"] = ref["phase"][:, :5]
    elif kind == "wrong_dt":
        ref["time_s"][10] += .001
    elif kind == "cadence":
        ref["phase"] *= 1.01
    else:
        ref["time_s"] -= 20
    with pytest.raises(ValueError):
        subject.curve128(ref)


def test_module_hot_methods_have_no_host_scalar_copy_or_neural_imports():
    import ast
    source = Path(subject.ROOT / subject.OWNED_SOURCES[0]).read_text(encoding="utf8")
    tree = ast.parse(source)
    hot = {"_input", "_sample", "interpolate", "project", "evaluate_pairs", "evaluate"}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in hot:
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
                    assert call.func.attr not in {"item", "cpu", "numpy", "detach", "tolist"}
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all("flylab" not in ast.unparse(node) and "flygym" not in ast.unparse(node) for node in imports)
