"""E336 explicit generated NumPy/I/O fixtures; original numerical reads0."""

import json
import sys
import time

import numpy as np
import pytest

from tools import analyze_c_banc_m3_saved_gait_transfer as transfer


def reference():
    phase = np.arange(128, dtype=np.float64) / 128
    curves = np.zeros((6, 128, 7), np.float64)
    for leg in range(6):
        angle = 2 * np.pi * (phase + leg / 6)
        curves[leg, :, 0] = np.cos(angle)
        curves[leg, :, 1] = np.sin(angle)
        curves[leg, :, 2:] = leg / 10
    return transfer.canonical_constants(curves)


def commands(phase, constants):
    coordinate = (np.asarray(phase) % 1) * 128
    floor = np.floor(coordinate)
    index = floor.astype(np.int64) % 128
    curves = constants["curves"]
    q = (
        curves[:, index]
        + (coordinate - floor)[None, :, None]
        * (np.roll(curves, -1, axis=1) - curves)[:, index]
    )
    return q.transpose(1, 0, 2).reshape(-1, 42).copy()


def test_value_only_known_motion_and_leg_offsets():
    c = reference()
    q = commands(np.arange(201) * transfer.INC, c)
    encoded = transfer.ValueEncoding(c).encode(q)
    assert set(encoded) == {
        "weights",
        "target",
        "z",
        "R",
        "entropy",
        "minimum_anchor_distance_rad",
    }
    assert np.allclose(encoded["weights"].sum(axis=-1), 1)
    assert np.all(encoded["R"] > 0.99)
    arrays, metrics = transfer.smooth_diagnostics(c, q)
    for leg in transfer.LEGS:
        row = metrics["phases"][leg]
        assert row["valid_edge_count"] == 200 and row["reverse_count"] == 0
        assert len(row["segments"]) == 1
        assert abs(row["segments"][0]["signed_progress"] - 200 * transfer.INC) < 1e-4
    assert arrays["successor_residual"].shape == (200, 6, 7)
    assert metrics["boundaries"] == 201 and metrics["edges"] == 200
    assert np.allclose(encoded["z"][:, 0], encoded["z"][:, 5], atol=1e-12)
    assert not any("jac" in key or "grad" in key for key in arrays)


def test_constant_command_and_symmetric_center_not_dynamic_claim():
    c = reference()
    q = np.repeat(commands([0.2], c), 201, axis=0)
    arrays, metrics = transfer.smooth_diagnostics(c, q)
    assert np.array_equal(arrays["command_increment"], np.zeros((200, 42)))
    assert all(
        row["signed_progress_sum_within_segments"] == 0
        for row in metrics["phases"].values()
    )
    center = c["curves"].mean(axis=1).reshape(1, 42)
    encoded = transfer.ValueEncoding(c).encode(np.repeat(center, 201, axis=0))
    assert np.max(encoded["R"]) < 1e-12
    _, phase = transfer.phase_segments(encoded["z"], encoded["R"])
    assert all(
        row["valid_edge_count"] == 0 and not row["segments"] for row in phase.values()
    )


@pytest.mark.parametrize(
    "key", ["normalization", "temperature", "future_anchors", "phase_basis"]
)
def test_canonical_reconstruction_rejects_tamper(key):
    c = reference()
    c[key].flat[0] += 0.01
    with pytest.raises(ValueError):
        transfer.ValueEncoding(c)


def test_stored_canonical_bytes_used_consistently_and_nonpositive_rejected():
    c = reference()
    c["future_anchors"].flat[0] += 1e-16
    value = transfer.ValueEncoding(c)
    assert value.constants["future_anchors"].tobytes() == c["future_anchors"].tobytes()
    with pytest.raises(ValueError):
        transfer.ValueEncoding(
            transfer.canonical_constants(np.zeros((6, 128, 7), np.float64))
        )


@pytest.mark.parametrize("fault", ["dtype", "nan", "overflow", "empty"])
def test_value_input_guards(fault):
    q = np.zeros((201, 42), np.float64)
    if fault == "dtype":
        q = q.astype(np.float32)
    elif fault == "nan":
        q[0, 0] = np.nan
    elif fault == "overflow":
        q[0, 0] = 1e308
    else:
        q = q[:0]
    with pytest.raises((ValueError, FloatingPointError)):
        transfer.ValueEncoding(reference()).encode(q)


def test_phase_threshold_gap_half_turn_and_reverse_without_unwrap():
    angle = np.array([0.0, 0.1, 0.2, 0.3, 0.2, 0.1]) * 2 * np.pi
    z = np.tile(np.stack((np.cos(angle), np.sin(angle)), axis=-1)[:, None], (1, 6, 1))
    R = np.full((6, 6), 0.5, np.float64)
    R[2] = np.nextafter(0.5, 0)
    a, rows = transfer.phase_segments(z, R)
    assert rows["lf"]["valid_edge_count"] == 3 and rows["lf"]["reverse_count"] == 2
    assert len(rows["lf"]["segments"]) == 2
    assert np.isnan(a["delta"][1:3]).all()
    assert rows["lf"]["segments"][0]["edge_count"] == 1
    opposite = np.tile(np.array([[[1.0, 0.0]], [[-1.0, 0.0]]]), (1, 6, 1))
    _, rows = transfer.phase_segments(opposite, np.ones((2, 6), np.float64))
    assert rows["lf"]["valid_edge_count"] == 0
    with pytest.raises(ValueError):
        transfer.phase_segments(
            z, R, controls=np.array([100, 101, 103, 104, 105, 106], np.int64)
        )


def test_tracking_sign_axes_windows_ratios_and_zero_denominator():
    c = reference()
    q = commands(np.arange(200) * transfer.INC, c)
    actual = 0.5 * q
    a, metrics = transfer.tracking_statistics(
        q, actual, q - actual, c["normalization"], c["curves"]
    )
    assert a["command_increment"].shape == (199, 42)
    assert metrics["controls"] == [100, 299] and metrics["boundaries"] == 200
    assert all(
        value == pytest.approx(0.5)
        for i, value in enumerate(metrics["actual_span_over_command_span"])
        if i % 7 < 2
    )
    assert all(
        value is None
        for i, value in enumerate(metrics["actual_span_over_command_span"])
        if i % 7 >= 2
    )
    np.testing.assert_allclose(a["tracking"], actual - q)
    with pytest.raises(ValueError):
        transfer.tracking_statistics(
            q, actual, actual - q, c["normalization"], c["curves"]
        )


def native_fixture():
    a = {
        "time_s": np.arange(1, 15001, dtype=np.float64) * 0.0001,
        "normal_load_bw": np.zeros((15000, 6), np.float64),
        "slip_mm_s": np.full((15000, 6), np.nan, np.float64),
        "slip_valid": np.zeros((15000, 6), bool),
        "upright": np.ones(15000),
        "nonfoot_contact": np.zeros(15000, bool),
    }
    a["normal_load_bw"][:, :2] = [0.1, 0.2]
    a["slip_mm_s"][:, :2] = [1.0, 9.0]
    a["slip_valid"][:, :2] = True
    return a


def test_native_exact_weighted_mask_not_leg_average_and_no_load_null():
    a = native_fixture()
    m = transfer.native_statistics(a)
    assert (
        m["actual_mask_count"] == 10000
        and m["first_native_tick"] == 5000
        and m["last_native_tick"] == 14999
    )
    assert m["weighted_slip_mm_s"] == pytest.approx((0.1 * 1 + 0.2 * 9) / 0.3)
    assert m["per_leg"]["lh"]["weighted_slip_mm_s"] is None
    a["normal_load_bw"].fill(0)
    m = transfer.native_statistics(a)
    assert m["weighted_slip_mm_s"] is None and m["load_denominator_bw"] == 0


@pytest.mark.parametrize(
    "which", ["lower_minus", "lower_plus", "upper_minus", "upper_plus"]
)
def test_native_endpoint_jitter_keeps_actual_mask(which):
    a = native_fixture()
    index = 4999 if which.startswith("lower") else 14999
    a["time_s"][index] += -1e-12 if which.endswith("minus") else 1e-12
    m = transfer.native_statistics(a)
    expected = [index + 1] if which in ("lower_minus", "upper_minus") else []
    assert m["nominal_mask_difference_ticks"] == expected


@pytest.mark.parametrize("fault", ["invalid_slip", "nan_load", "clock", "bool_dtype"])
def test_native_validity_guards(fault):
    a = native_fixture()
    if fault == "invalid_slip":
        a["slip_valid"][6000, 0] = False
    elif fault == "nan_load":
        a["normal_load_bw"][0, 0] = np.nan
    elif fault == "clock":
        a["time_s"][0] += 1e-5
    else:
        a["slip_valid"] = a["slip_valid"].astype(np.uint8)
    with pytest.raises(ValueError):
        transfer.native_statistics(a)


def tonic_fixture():
    encoded = np.full((25, 8), 1e20, np.float32)
    encoded[:, 3] = 0.1234567
    config = {"descending_ids": [0, 1], "descending_drive": 5000.0, "navigation": True}
    profile = {
        "ports": [3],
        "bilateral_descending": [0, 1],
        "port_background": -5000,
        "forward_drive": 5000.0,
    }
    actual = encoded.copy()
    actual[:, [0, 1]] += np.float32(5000)
    tonic = np.zeros(8, np.float32)
    tonic[3] = -5000
    tonic[[0, 1]] = 5000
    actual += tonic
    return encoded, actual, config, profile


def test_f32_tonic_order_large_finite_rounding_and_promoted_difference():
    encoded, actual, config, profile = tonic_fixture()
    assert transfer.verify_tonic(encoded, actual, config, profile)["verified"]
    assert np.array_equal(actual[:, :2], encoded[:, :2])
    assert not np.all(
        actual[:, 3].astype(np.float64) - encoded[:, 3].astype(np.float64) == -5000
    )
    m = transfer.difference_statistics(actual, encoded)
    expected = actual.astype(np.float64) - encoded.astype(np.float64)
    np.testing.assert_array_equal(m["changed_count"], np.full(25, 1))
    np.testing.assert_allclose(m["l2"], np.sqrt((expected * expected).sum(axis=1)))
    actual[0, 7] = np.nextafter(actual[0, 7], np.float32(np.inf))
    with pytest.raises(ValueError):
        transfer.verify_tonic(encoded, actual, config, profile)


def test_selected_typed_loading_never_decodes_unselected_object(tmp_path):
    p = tmp_path / "e336-generated-selected.npz"
    np.savez(
        p,
        allowed=np.arange(3, dtype=np.float32),
        forbidden=np.arange(5, dtype=np.float64),
    )
    reads = []
    a = transfer.load_selected(
        p, {"allowed": {"shape": [3], "dtype": "<f4"}}, reads=reads
    )
    assert list(a) == ["allowed"] and len(reads) == 1
    with pytest.raises(ValueError):
        transfer.load_selected(p, {"allowed": {"shape": [3], "dtype": "<f8"}})


@pytest.mark.parametrize("alias", ["backslash", "absolute", "relative_dot"])
def test_pins_aliases_and_collision_escape(tmp_path, alias):
    p = tmp_path / "producer.py"
    p.write_text("# fixture", "utf8")
    name = (
        str(p).replace("/", "\\")
        if alias == "backslash"
        else str(p)
        if alias == "absolute"
        else "./producer.py"
    )
    target = {}
    transfer.merge_pins(target, {name: transfer.sha256(p)}, root=tmp_path)
    assert target == {"producer.py": transfer.sha256(p)}
    with pytest.raises(ValueError):
        transfer.merge_pins(target, {"producer.py": "0" * 64}, root=tmp_path)
    with pytest.raises(ValueError):
        transfer.safe("../escape.py", tmp_path)


def execution_fixture(tmp_path):
    helper = tmp_path / "fixture.py"
    helper.write_text("# generated-only producer", "utf8")
    report = {
        "audit_source_path": "fixture.py",
        "audit_source_sha256": transfer.sha256(helper),
    }
    desc = {"path": "report.json"}
    argv = [
        ".venv/Scripts/python.exe",
        "-X",
        "utf8",
        "-B",
        "fixture.py",
        "--output",
        "report.json",
        "--fixtures",
    ]
    contract = {
        "commands": [{"argv": argv}],
        "source_sha256": {"fixture.py": transfer.sha256(helper)},
    }
    sup = {
        "schema": "flylab.m3.e267.root-supervision.v1",
        "success": True,
        "owned_pids": [1],
        "attempts": [{"argv": argv.copy(), "exit_code": 0, "killed": False}],
    }
    return report, desc, contract, sup


def test_execution_helper_report_and_exact_completed_argv(tmp_path):
    transfer.verify_execution(*execution_fixture(tmp_path), root=tmp_path)


@pytest.mark.parametrize(
    "fault", [None, "source_missing", "source_collision", "log_missing", "log_changed"]
)
def test_transitive_executed_source_and_completed_logs_pinned(tmp_path, fault):
    _, _, contract, _ = execution_fixture(tmp_path)
    imported = tmp_path / "imported.py"
    imported.write_text("# independent imported helper", "utf8")
    contract["source_sha256"]["imported.py"] = transfer.sha256(imported)
    command = contract["commands"][0]
    for key in ("stdout", "stderr"):
        command[key] = "e336-generated." + key
        (tmp_path / command[key]).write_text("completed " + key, "utf8")
    freeze = {
        "source_sha256": contract["source_sha256"].copy(),
        "input_sha256": {
            command[k]: transfer.sha256(tmp_path / command[k])
            for k in ("stdout", "stderr")
        },
    }
    if fault == "source_missing":
        del freeze["source_sha256"]["imported.py"]
    elif fault == "source_collision":
        freeze["input_sha256"]["./imported.py"] = "0" * 64
    elif fault == "log_missing":
        del freeze["input_sha256"][command["stderr"]]
    elif fault == "log_changed":
        (tmp_path / command["stdout"]).write_text("changed", "utf8")
    if fault is None:
        transfer.verify_producer_closure(contract, freeze, root=tmp_path)
    else:
        with pytest.raises(ValueError):
            transfer.verify_producer_closure(contract, freeze, root=tmp_path)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_helper",
        "helper_bytes",
        "argv",
        "python_c",
        "report_path",
        "failed",
        "omitted",
        "duplicate_output",
    ],
)
def test_execution_rejects_fabricated_producer_updates(tmp_path, fault):
    report, desc, contract, sup = execution_fixture(tmp_path)
    if fault == "missing_helper":
        contract["source_sha256"] = {}
    elif fault == "helper_bytes":
        (tmp_path / "fixture.py").write_text("# changed", "utf8")
    elif fault == "argv":
        sup["attempts"][0]["argv"][-1] = "--other"
    elif fault == "python_c":
        contract["commands"][0]["argv"][4] = "-c"
        sup["attempts"][0]["argv"] = contract["commands"][0]["argv"].copy()
    elif fault == "report_path":
        desc["path"] = "other-report.json"
    elif fault == "failed":
        sup["attempts"][0]["exit_code"] = 1
    elif fault == "omitted":
        sup["attempts"] = []
    else:
        contract["commands"][0]["argv"].extend(["--output", "report.json"])
        sup["attempts"][0]["argv"] = contract["commands"][0]["argv"].copy()
    with pytest.raises(ValueError):
        transfer.verify_execution(report, desc, contract, sup, root=tmp_path)


class FakeDependencies:
    def __init__(self, fault=None):
        self.fault, self.loads = fault, 0

    def load(self, _freeze, *, root, timely, reads):
        self.loads += 1
        timely()
        reads.append({"path": "synthetic", "key": "fixture"})
        if self.fault == "load":
            raise ValueError("first generated load")
        return reference()

    def compute(self, data, _freeze, save, **_kwargs):
        a, m = transfer.smooth_diagnostics(
            data, commands(np.arange(201) * transfer.INC, data)
        )
        save(
            "generated-smooth.npz",
            {key: a[key] for key in ("q", "z", "R", "phase_delta", "phase_valid")},
        )
        if self.fault == "compute":
            raise ValueError("first generated compute after raw")
        return {"generated_only": True, "smooth": m}


def invoke(tmp_path, deps, **kwargs):
    p = tmp_path / "e336-generated-producer.json"
    transfer.put_json(p, {"generated": True})
    pins = {p.name: transfer.sha256(p)}
    output = tmp_path / "e336-generated-output"
    value = transfer.run(
        tmp_path / "freeze",
        "f",
        tmp_path / "review",
        "r",
        output,
        root=tmp_path,
        dependencies=deps,
        test_dimensions=transfer.TestDimensions(8),
        gatekeeper=kwargs.pop(
            "gatekeeper",
            lambda *_a, **_k: {"source_sha256": pins, "input_sha256": pins},
        ),
        meter=kwargs.pop("meter", lambda: {"host_peak_bytes": 1}),
        **kwargs,
    )
    return value, output


def test_generated_lifecycle_disk_roster_and_no_runtime_import(tmp_path):
    result, output = invoke(tmp_path, FakeDependencies())
    assert (
        result["analysis_valid"]
        and not result["candidate_adopted"]
        and not result["F1_F8_complete"]
    )
    assert transfer.read(output / "result.json") == result
    assert transfer.read(output / "output-manifest.json")[
        "files"
    ] == transfer.file_manifest(output)
    assert (
        result["original_analysis_calls"]
        == result["neural_calls"]
        == result["body_calls"]
        == result["CUDA_calls"]
        == 0
    )
    assert "torch" not in sys.modules and "cupy" not in sys.modules


@pytest.mark.parametrize(
    "fault",
    [
        "load",
        "compute",
        "writer",
        "readback",
        "deadline",
        "late_host",
        "posthash",
        "save",
    ],
)
def test_lifecycle_partial_first_error_and_publication_revoke(tmp_path, fault):
    now, changed = [0.0], [False]
    deps = FakeDependencies(fault)

    def writer(path, value):
        transfer.put_json(path, value)
        if path.name == "diagnosis.json" and fault == "posthash":
            (tmp_path / "e336-generated-producer.json").write_text("changed", "utf8")
        if path.name == "result.json" and not changed[0]:
            changed[0] = True
            if fault == "writer":
                raise OSError("first generated writer")
            if fault == "readback":
                path.write_text("{}", "utf8")
            if fault == "deadline":
                now[0] = 36.0

    def saver(path, **arrays):
        np.savez(path, **arrays)
        if fault == "save":
            raise OSError("first generated save after raw")

    result, output = invoke(
        tmp_path,
        deps,
        writer=writer,
        saver=saver,
        clock=lambda: now[0],
        meter=lambda: {
            "host_peak_bytes": (3 << 30) if fault == "late_host" and changed[0] else 1
        },
    )
    assert not result["analysis_valid"] and result["first_error"]
    assert transfer.read(output / "result.json") == result
    ledger = transfer.read(output / "advance-ledger.json")
    assert (
        ledger["first_error"] == result["first_error"]
        and ledger["absolute_clock"]["final_deadline"] == 5.0
    )
    if fault != "load":
        assert list(output.glob("*.npz"))


def test_gate_rejection_before_selective_loader(tmp_path):
    deps = FakeDependencies()

    def reject(*_a, **_k):
        raise ValueError("passive gate rejected")

    with pytest.raises(ValueError, match="passive gate"):
        invoke(tmp_path, deps, gatekeeper=reject)
    assert deps.loads == 0


def test_absolute_clock_started_before_gate_never_reset_for_science(tmp_path):
    deps, now = FakeDependencies(), [100.0]

    def slow_gate(*_a, **_k):
        now[0] += 30
        return {"source_sha256": {}, "input_sha256": {}}

    with pytest.raises(TimeoutError, match="absolute"):
        invoke(tmp_path, deps, gatekeeper=slow_gate, clock=lambda: now[0])
    assert deps.loads == 0 and not (tmp_path / "e336-generated-output").exists()


def test_postpublication_recursive64MiB_fault_simulated_without_allocating_payload(
    tmp_path, monkeypatch
):
    made = [False]
    actual_counter = transfer.recursive_bytes
    monkeypatch.setattr(
        transfer,
        "recursive_bytes",
        lambda path: (64 << 20) + 1 if made[0] else actual_counter(path),
    )

    def writer(path, value):
        transfer.put_json(path, value)
        if path.name == "result.json" and not made[0]:
            made[0] = True

    result, output = invoke(tmp_path, FakeDependencies(), writer=writer)
    assert not result["analysis_valid"]
    assert result["first_error"]["message"] == "recursive output64MiB limit"
    assert (output / "generated-smooth.npz").exists()
    assert transfer.read(output / "result.json") == result


def test_original_cli_has_no_generated_dimension_option():
    with pytest.raises(SystemExit) as error:
        transfer.main(
            [
                "--freeze",
                "f",
                "--freeze-sha256",
                "a",
                "--review",
                "r",
                "--review-sha256",
                "b",
                "--output",
                "o",
                "--test-dimensions",
                "8",
            ]
        )
    assert error.value.code == 2


@pytest.mark.parametrize("injection", ["clock", "writer", "emergency_writer", "saver"])
def test_original_api_rejects_clock_and_sink_injections_before_gate(
    tmp_path, injection
):
    def never_called(*_args, **_kwargs):
        raise AssertionError("original injection must be rejected before use")

    with pytest.raises(ValueError, match="explicit TestDimensions"):
        transfer.run(
            tmp_path / "missing-freeze.json",
            "generated",
            tmp_path / "missing-review.json",
            "generated",
            tmp_path / "unused-output",
            **{injection: never_called},
        )
    assert not (tmp_path / "unused-output").exists()


def live_fixture():
    ids = np.arange(1, 26, dtype=np.int64)
    model = {
        "actual": np.repeat(
            np.arange(301, dtype=np.float64)[:, None] / 1000, 42, axis=1
        ),
        "knee": np.full((301, 6), 90.0, np.float64),
        "position": np.repeat(np.arange(301, dtype=np.float64)[:, None], 3, axis=1)
        @ transfer.S.T,
        "basis": np.tile(transfer.S, (301, 1, 1)),
    }
    raw = {
        "motor_rate32": np.repeat(np.arange(301, dtype=np.float32)[:, None], 4, axis=1)
    }
    chunk = {"control_ids": ids}
    for key, shape in transfer.LIVE_FIELDS.items():
        chunk[key] = np.zeros((25, *shape), np.float64)
    chunk["angles_rad"] = model["actual"][ids - 1].copy()
    chunk["knee_flexion_rad"].fill(np.pi / 2)
    chunk["prior_position"] = np.repeat(
        (ids - 1).astype(np.float64)[:, None], 3, axis=1
    )
    chunk["prior_rotation"] = np.tile(np.eye(3), (25, 1, 1))
    chunk["encoded_drive"] = np.zeros((25, 8), np.float32)
    chunk["actual_drive"] = chunk["encoded_drive"].copy()
    chunk["motor_readout"] = raw["motor_rate32"][ids].copy()
    for key in transfer.LIVE_CLOCKS:
        value = ids - 1 if key.endswith("before") else ids.copy()
        if key.startswith("neural"):
            value = value * 20
        elif key.startswith("physics_returned"):
            value = value * 50
        elif key.startswith("body"):
            value = value.astype(np.float64) * 0.005
        chunk[key] = value
    for key in list(chunk):
        if key != "control_ids":
            chunk[key + "_control_ids"] = ids.copy()
    rows, events = [], []
    for i, c in enumerate(ids):
        rows.append(
            {
                "control": int(c),
                "control_returned": True,
                "error": None,
                **{key: chunk[key][i].item() for key in transfer.LIVE_CLOCKS},
            }
        )
        for name in transfer.EVENT_ORDER:
            event = {"control": int(c), "event": name}
            if name == "knee_kinematics_returned":
                event["include_direction"] = False
            events.append(event)
        for key in {
            **transfer.LIVE_FIELDS,
            "encoded_drive": (),
            "actual_drive": (),
            "motor_readout": (),
        }:
            events.append(
                {
                    "control": int(c),
                    "event": key + "_acquired",
                    "sha256": transfer.bytes_sha(chunk[key][i]),
                }
            )
    descriptor = {
        "controls": ids.tolist(),
        "availability": rows,
        "arrays": {key: transfer.metadata(a) for key, a in chunk.items()},
    }
    return chunk, descriptor, events, raw, model


def test_live_prior_boundary_knee_complement_coordinate_and_MN_alignment():
    proof = transfer.verify_live_chunk(*live_fixture(), nodes=8, motors=4)
    assert proof["prior_boundary_geometry"] and proof["motor_raw_exact"]
    assert proof["receptor_load_is_not_native_floor_load"]


@pytest.mark.parametrize(
    "fault",
    [
        "IDs",
        "dtype",
        "knee",
        "prior_row",
        "MN",
        "availability",
        "clock",
        "event_count",
        "event_order",
        "kwargs",
        "field_digest",
    ],
)
def test_live_rejects_row_clock_event_MN_tamper(fault):
    chunk, desc, events, raw, model = live_fixture()
    if fault == "IDs":
        chunk["angles_rad_control_ids"][0] = 2
    elif fault == "dtype":
        chunk["prior_position"] = chunk["prior_position"].astype(np.float32)
    elif fault == "knee":
        model["knee"][0, 0] = 45
    elif fault == "prior_row":
        model["actual"][0] = model["actual"][1]
    elif fault == "MN":
        raw["motor_rate32"][1, 0] += 1
    elif fault == "availability":
        desc["availability"][0]["control_returned"] = False
    elif fault == "clock":
        chunk["neural_after"][0] += 20
    elif fault == "event_count":
        events.append(events[0].copy())
    elif fault == "event_order":
        events[1], events[2] = events[2], events[1]
    elif fault == "kwargs":
        next(e for e in events if e["event"] == "knee_kinematics_returned")[
            "include_direction"
        ] = True
    else:
        next(e for e in events if e["event"] == "actual_drive_acquired")["sha256"] = (
            "0" * 64
        )
    with pytest.raises(ValueError):
        transfer.verify_live_chunk(chunk, desc, events, raw, model, nodes=8, motors=4)


def inventory_fixture(tmp_path, dtype=np.float32, fortran=False):
    path = tmp_path / "e336-generated-array.npz"
    values = np.arange(6).reshape(2, 3).astype(dtype)
    np.savez(path, value=np.asfortranarray(values) if fortran else values)
    h = transfer.npz_headers(path)["value"]
    pin = transfer.sha256(path)
    inputs = {path.name: pin}
    inventory = {
        "schema": "flylab.m3.e336.root-input-inventory.v1",
        "passed": True,
        "array_payload_numeric_reads": 0,
        "inherited_source_sha256": {},
        "inherited_input_sha256": inputs,
        "files": {path.name: {"sha256": pin, "bytes": path.stat().st_size}},
        "protected_F6": {"payload_paths": []},
        "archives": {
            path.name: {
                "sha256": pin,
                "bytes": path.stat().st_size,
                "members": {
                    "value": {
                        "shape": h["shape"],
                        "dtype": h["dtype"],
                        "fortran_order": h["fortran"],
                        "zip_compression": h["compression"],
                        "compressed_bytes": h["compressed_bytes"],
                        "uncompressed_bytes": h["uncompressed_bytes"],
                        "npy_version": h["npy_version"],
                        "header_bytes": h["header_bytes"],
                        "crc32": h["crc32"],
                    }
                },
            }
        },
    }
    return inventory, inputs


def test_inventory_metadata_and_protected_payload_scope(tmp_path):
    inventory, inputs = inventory_fixture(tmp_path)
    transfer.verify_inventory(inventory, {}, inputs, root=tmp_path)
    inventory["protected_F6"]["payload_paths"] = list(inputs)
    with pytest.raises(ValueError, match="protected F6"):
        transfer.verify_inventory(inventory, {}, inputs, root=tmp_path)


@pytest.mark.parametrize("dtype", ["float64", "float32", "int64", "bool"])
@pytest.mark.parametrize("fortran", [False, True])
def test_actual_external_inventory_dtype_dialect_without_payload_conversion(
    tmp_path, dtype, fortran
):
    inventory, inputs = inventory_fixture(tmp_path, dtype=dtype, fortran=fortran)
    path = next(iter(inputs))
    member = inventory["archives"][path]["members"]["value"]
    member["dtype"] = str(np.dtype(dtype))
    transfer.verify_inventory(inventory, {}, inputs, root=tmp_path)
    reads = []
    freeze = {"input_members": inventory["archives"]}
    values = transfer.Dependencies.selected(path, freeze, tmp_path, lambda: None, reads)
    array = values["value"]
    assert array.dtype == np.dtype(dtype)
    assert array.flags.f_contiguous == fortran
    np.testing.assert_array_equal(array, np.arange(6).reshape(2, 3).astype(dtype))
    assert reads[0]["dtype"] == np.dtype(dtype).str
    assert reads[0]["c_order_sha256"] == transfer.bytes_sha(array)


@pytest.mark.parametrize(
    "fault",
    ["different_width", "different_kind", "different_endian", "different_fortran"],
)
def test_external_dialect_rejects_genuine_type_endian_or_order_mismatch(
    tmp_path, fault
):
    inventory, inputs = inventory_fixture(tmp_path, dtype="float64")
    path = next(iter(inputs))
    member = inventory["archives"][path]["members"]["value"]
    member["dtype"] = "float64"
    if fault == "different_width":
        member["dtype"] = "float32"
    elif fault == "different_kind":
        member["dtype"] = "int64"
    elif fault == "different_endian":
        member["dtype"] = ">f8" if np.little_endian else "<f8"
    else:
        member["fortran_order"] = True
    with pytest.raises(ValueError):
        transfer.verify_inventory(inventory, {}, inputs, root=tmp_path)
    with pytest.raises(ValueError):
        transfer.Dependencies.selected(
            path, {"input_members": inventory["archives"]}, tmp_path, lambda: None, []
        )


def test_big_endian_dtype_metadata_stays_big_endian_and_never_casts(tmp_path):
    inventory, inputs = inventory_fixture(tmp_path, dtype=">f8")
    path = next(iter(inputs))
    inventory["archives"][path]["members"]["value"]["dtype"] = ">f8"
    transfer.verify_inventory(inventory, {}, inputs, root=tmp_path)
    values = transfer.Dependencies.selected(
        path, {"input_members": inventory["archives"]}, tmp_path, lambda: None, []
    )
    assert values["value"].dtype.str == ">f8"
    assert (
        values["value"].tobytes() == np.arange(6).reshape(2, 3).astype(">f8").tobytes()
    )


@pytest.mark.parametrize("invalid", [None, False, 0, "", "object", "(2,)f8"])
def test_invalid_dtype_metadata_cannot_default_to_float64(tmp_path, invalid):
    inventory, inputs = inventory_fixture(tmp_path, dtype="float64")
    path = next(iter(inputs))
    inventory["archives"][path]["members"]["value"]["dtype"] = invalid
    with pytest.raises(ValueError):
        transfer.verify_inventory(inventory, {}, inputs, root=tmp_path)
    with pytest.raises(ValueError):
        transfer.Dependencies.selected(
            path, {"input_members": inventory["archives"]}, tmp_path, lambda: None, []
        )


@pytest.mark.parametrize(
    "field",
    [
        "dtype",
        "shape",
        "fortran_order",
        "npy_version",
        "header_bytes",
        "zip_compression",
        "compressed_bytes",
        "uncompressed_bytes",
        "crc32",
    ],
)
def test_inventory_rejects_exact_header_tamper(tmp_path, field):
    inventory, inputs = inventory_fixture(tmp_path)
    spec = inventory["archives"][next(iter(inputs))]["members"]["value"]
    spec[field] = "wrong"
    with pytest.raises(ValueError):
        transfer.verify_inventory(inventory, {}, inputs, root=tmp_path)


def test_concrete_pinned_source_axis_ordering_static_only():
    sources = {
        name: transfer.sha256(transfer.ROOT / name)
        for name in (
            "flylab/c/walking_trace.py",
            "flylab/body.py",
            "flylab/c/banc_walking.py",
            "flylab/common.py",
        )
    }
    proof = transfer.source_correspondence(sources)
    assert (
        proof["flylab/body.py"]["exact_expressions"][0]
        == "self.d.qpos[self.qpos_ids].copy()"
    )
    with pytest.raises(ValueError):
        transfer.source_correspondence({})


def trace_fixture(tmp_path):
    rows = []
    for c in range(301):
        origin = np.array([-2 * c * 0.005, 0.0, 0.0])
        legs = {}
        for leg in transfer.LEGS:
            points = origin + np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.3, 0.0],
                    [2.0, 0.3, 0.0],
                    [2.5, 0.3, 0.0],
                ]
            )
            legs[leg] = points.tolist()
        rows.append(
            {
                "control_tick": c,
                "time_s": c * 0.005,
                "body": {
                    "position": origin.tolist(),
                    "basis": np.eye(3).tolist(),
                    "yaw": 0.0,
                    "legs": legs,
                },
                "joint_angles_rad": [0.0] * 42,
                "joint_targets_rad": [0.0] * 42,
                "joint_tracking_error_rad": [0.0] * 42,
                "adhesion": [True] * 6,
                "feet": {"load_bearing": [True] * 6},
                "upright": 1.0,
                "fault": None,
                "navigation": None,
            }
        )
    path = tmp_path / "e336-generated-trace.json"
    path.write_text(
        json.dumps(
            {
                "samples": rows,
                "body_hash": "fixture",
                "model_hash": "fixture",
                "adapter_hash": "fixture",
                "graph_hash": "fixture",
                "parameters": {},
            },
            separators=(",", ":"),
            allow_nan=False,
        ),
        "utf8",
    )
    raw = {
        "command": np.zeros((301, 42), np.float64),
        "tendon": np.zeros((301, 6), np.float64),
        "control_tick": np.arange(301, dtype=np.int64),
        "network_tick": np.arange(301, dtype=np.int64) * 20,
        "physics_time_s": np.arange(301) * 0.005,
    }
    return path, raw


def test_official_geometry_signed_forward_slip_bout_cycles_and_startup(tmp_path):
    path, raw = trace_fixture(tmp_path)
    model = transfer.trace_data(path, raw)
    arrays, physical = transfer.physical_metrics(
        raw,
        native_fixture(),
        model,
        reference()["curves"],
        {"target_command": np.zeros((99, 42)), "target_tendon": np.zeros((99, 6))},
    )
    official = physical["official"]
    assert official["forward_mm"] == pytest.approx(-1.99)
    assert official["slip_mm_s"] == pytest.approx(19 / 3)
    assert (
        official["startup_pass"]
        and official["healthy"]
        and not official["short_fixed_pass"]
    )
    assert official["longest_continuous_forward_bout_s"] == 0
    assert all(v == 0 for v in official["complete_cycles"].values())
    assert np.allclose(arrays["geometric_knee_interior_deg"], 90)
    assert physical["exploratory_sensitivity_is_descriptive_only"]


@pytest.mark.parametrize("fault", ["target", "error", "clock", "brake"])
def test_trace_same_window_correspondence_guards(tmp_path, fault):
    path, raw = trace_fixture(tmp_path)
    value = transfer.read(path)
    row = value["samples"][100]
    if fault == "target":
        row["joint_targets_rad"][0] += 0.1
    elif fault == "error":
        row["joint_tracking_error_rad"][0] += 0.1
    elif fault == "clock":
        raw["physics_time_s"][100] += 1e-4
    else:
        row["navigation"] = {"brake_request": True}
    path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False), "utf8")
    with pytest.raises(ValueError):
        transfer.trace_data(path, raw)


def test_full25_chunk_stored_deflated_repeated12_reads_within64MiB(tmp_path):
    folder = tmp_path / "e336-generated-full25"
    folder.mkdir()
    nodes = 158706
    matrix = np.repeat(np.arange(25, dtype=np.float32)[:, None], nodes, axis=1)
    matrix[:, ::31] = 0.1234567
    values = {
        "encoded": matrix,
        "actual": matrix + np.float32(1),
        "held": matrix + np.float32(2),
    }
    schema = {key: {"shape": [25, nodes], "dtype": "<f4"} for key in values}
    started = time.perf_counter()
    stored, compressed = folder / "stored.npz", folder / "deflated.npz"
    np.savez(stored, **values)
    np.savez_compressed(compressed, **values)
    del values, matrix
    write_s = time.perf_counter() - started
    started = time.perf_counter()
    for index in range(12):
        path = stored if index % 2 == 0 else compressed
        chunk = transfer.load_selected(path, schema)
        result = transfer.difference_statistics(chunk["actual"], chunk["held"])
        assert np.array_equal(result["changed_count"], np.full(25, nodes))
        assert np.allclose(result["rms"], 1)
        del chunk, result
    h1, h2 = transfer.npz_headers(stored), transfer.npz_headers(compressed)
    assert all(v["compression"] == 0 for v in h1.values())
    assert all(v["compression"] == 8 for v in h2.values())
    output_bytes = transfer.recursive_bytes(folder)
    assert output_bytes < 64 << 20
    report = {
        "schema": "flylab.m3.e336.sol-generated-workload.v1",
        "passed": True,
        "repeated_reads": 12,
        "generated_only": True,
        "original_analysis_calls": 0,
        "neural_calls": 0,
        "body_calls": 0,
        "CUDA_calls": 0,
        "full25_shape": [25, nodes],
        "write_s": write_s,
        "read_compute_s": time.perf_counter() - started,
        "output_bytes": output_bytes,
        "original40second_completion_proven": False,
    }
    transfer.put_json(folder / "io-report.json", report)
    print("E336_SOL_GENERATED_IO " + json.dumps(report))
