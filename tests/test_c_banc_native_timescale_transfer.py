import ast
import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tools import verify_c_banc_m3_native_timescale_transfer as runner
from tools.verify_c_banc_m3_e113_body import AdvanceLedger


class FailingReceipt:
    def __init__(self, failure=None):
        self.failure = failure
        self.calls = 0
        self.snapshots = []

    def __call__(self):
        self.calls += 1
        self.snapshots.append(self.calls)
        if self.calls == self.failure:
            raise OSError(f"receipt-{self.calls}")


def _generated_candidate(tmp_path):
    edges = np.asarray([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23], np.int64)
    motors = np.asarray([4, 7, 9, 12, 15, 18], np.int64)
    edge_mask = np.zeros((6, len(edges)), bool)
    motor_mask = np.zeros((6, len(motors)), bool)
    muscle_mask = np.zeros((6, 6), bool)
    for leg in range(6):
        edge_mask[leg, 2 * leg:2 * leg + 2] = True
        motor_mask[leg, leg] = True
        muscle_mask[leg, leg] = True
    calibration_ids = np.arange(30, dtype=np.int64)
    values = np.linspace(.01, .3, len(calibration_ids), dtype=np.float32)
    values[edges[1::2]] *= -1
    source_values = values.copy()
    source_values[edges[0]] = np.float32(values[edges[0]] * (1 - 2e-7))
    source_values[edges[1]] = np.float32(values[edges[1]] * (1 + 2e-7))
    calibration = {
        "edge_ids": calibration_ids,
        "source_values": source_values,
        "values": values.copy(),
        "individual_edge_ids": edges.copy(),
        "individual_edge_efficacies": np.ones(len(edges), np.float64),
        "preserved": np.arange(5, dtype=np.int16),
    }
    fixture = {
        "edge_ids": edges,
        "motor_ids": motors,
        "pool": np.eye(6, dtype=np.float64),
        "membership": np.arange(6, dtype=np.int64),
        "leg_edge_mask": edge_mask,
        "leg_motor_mask": motor_mask,
        "leg_muscle_mask": muscle_mask,
        "delta_bounds": np.tile(
            np.asarray([runner.LOG_TAU_LOWER, runner.LOG_TAU_UPPER]), (len(edges), 1)),
        "base_fit_weights": values[edges].astype(np.float64),
    }
    selected = []
    all_delta = np.linspace(runner.LOG_TAU_LOWER, runner.LOG_TAU_UPPER,
                            len(edges), dtype=np.float64)
    all_gain = np.linspace(0, 8, 6, dtype=np.float64)
    all_ell = np.linspace(runner.LOG_TAU_LOWER, runner.LOG_TAU_UPPER,
                          6, dtype=np.float64)
    for leg in range(6):
        edge_columns = np.flatnonzero(edge_mask[leg])
        motor_columns = np.flatnonzero(motor_mask[leg])
        muscle_columns = np.flatnonzero(muscle_mask[leg])
        command = np.full((3, 2001, 42), -1000. - leg, np.float64)
        command[:, :, 7 * leg:7 * leg + 7] = leg + np.arange(7)
        raw = {
            "delta": all_delta[edge_columns],
            "gain": all_gain[muscle_columns],
            "log_tau": all_ell[muscle_columns],
            "edge_ids": edges[edge_columns],
            "motor_indices": motor_columns.astype(np.int32),
            "muscle_indices": muscle_columns.astype(np.int32),
            "rate": np.full((3, 2001, 1), leg + .1, np.float64),
            "adaptation": np.full((3, 2001, 1), leg + .2, np.float64),
            "activation": np.full((3, 2001, 1), leg + .3, np.float64),
            "command": command,
            "tendon": np.full((3, 2001, 1), leg + .4, np.float64),
        }
        path = tmp_path / f"leg-{leg}.npz"
        np.savez(path, **raw)
        selected.append({"leg": leg, "source": {"raw_path": path.name,
            "raw_sha256": runner.sha256(path)}, "original_guard_pass": leg not in (2, 5),
            "ratio": .97 + leg / 1000})
    return ({"selected_legs": selected}, fixture, calibration, all_delta,
            all_gain, all_ell)


def test_generated_candidate_assembly_exact_mapping_and_no_clipping(tmp_path):
    freeze, fixture, calibration, delta, gain, ell = _generated_candidate(tmp_path)
    parameters, changed, expected, report = runner.assemble_candidate(
        freeze, fixture, calibration,
        path_resolver=lambda relative: tmp_path / relative)
    assert np.array_equal(parameters["delta"], delta)
    assert np.array_equal(parameters["gain"], gain)
    assert np.array_equal(parameters["log_tau"], ell)
    target64 = fixture["base_fit_weights"] * np.exp(delta)
    assert np.array_equal(parameters["native_fit_values64"], target64)
    assert np.array_equal(parameters["native_fit_values32"], target64.astype(np.float32))
    mapped = parameters["calibration_positions"]
    assert np.array_equal(changed["values"][mapped], target64.astype(np.float32))
    other = np.ones(len(changed["values"]), bool)
    other[mapped] = False
    assert np.array_equal(changed["values"][other], calibration["values"][other])
    assert np.array_equal(changed["source_values"], calibration["source_values"])
    assert report["failed_guard_legs"] == [2, 5]
    assert report["clipped"] is False
    assert report["derived_efficacy_max"] > 10
    assert expected["rate"].shape == (3, 2001, 6)
    assert expected["command"].shape == (3, 2001, 42)
    for leg in range(6):
        assert np.all(expected["rate"][:, :, leg] == leg + .1)
        assert np.all(expected["activation"][:, :, leg] == leg + .3)
        assert np.all(expected["command"][:, :, 7 * leg] == leg)
        assert np.all(expected["tendon"][:, :, leg] == leg + .4)


def test_generated_candidate_rejects_bounds_identity_and_hash(tmp_path):
    freeze, fixture, calibration, *_ = _generated_candidate(tmp_path)
    bad = copy.deepcopy(freeze)
    bad["selected_legs"][0]["source"]["raw_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="raw changed"):
        runner.assemble_candidate(bad, fixture, calibration,
                                  path_resolver=lambda relative: tmp_path / relative)
    raw_path = tmp_path / freeze["selected_legs"][0]["source"]["raw_path"]
    raw = runner.load_npz(raw_path)
    raw["delta"][0] = runner.LOG_TAU_LOWER - np.finfo(float).eps
    np.savez(raw_path, **raw)
    freeze["selected_legs"][0]["source"]["raw_sha256"] = runner.sha256(raw_path)
    with pytest.raises(ValueError, match="bounds changed"):
        runner.assemble_candidate(freeze, fixture, calibration,
                                  path_resolver=lambda relative: tmp_path / relative)


def test_conditional_comparison_distinguishes_numeric_mismatch():
    expected = {key: np.arange(6, dtype=float).reshape(2, 3)
                for key in ("rate", "adaptation", "activation", "command", "tendon")}
    actual = {key: value.copy() for key, value in expected.items()}
    report = runner.compare_conditional(actual, expected, tuple(expected))
    assert report["accepted"] is True
    actual["rate"][0, 0] += 2e-8
    report = runner.compare_conditional(actual, expected, tuple(expected))
    assert report["accepted"] is False
    assert report["fields"]["rate"]["finite"] is True
    actual["rate"][0, 0] = np.nan
    report = runner.compare_conditional(actual, expected, tuple(expected))
    assert report["fields"]["rate"]["finite"] is False


class GeneratedConditionalEngine:
    contexts = 3
    ticks = 4
    samples = 3

    def __init__(self, before_solve):
        self.before_solve = before_solve
        self.solve_calls = 0
        self.problem = {"edge_ids": np.asarray([3, 5], np.int64),
                        "motor_indices": np.asarray([0], np.int32),
                        "muscle_indices": np.asarray([0], np.int32)}

    def solve(self, delta, gain, log_tau, *, derivative):
        self.before_solve()
        self.solve_calls += 1
        assert derivative is False
        raw = {key: np.full((3, 3, 1), index, np.float64)
               for index, key in enumerate(("rate", "adaptation", "activation",
                                             "command", "tendon"))}
        raw["branch_bits"] = np.zeros((3, 4, 1), np.uint8)
        timing = {"progress": {"completion_class": "known_complete"},
                  "context_rk4_ticks": 12, "conditional_model_s": .003,
                  "decoder_endpoint_ticks": 6, "wall_s": .01}
        return {"rate": object()}, raw, timing


def test_conditional_forward_persists_request_before_solve_and_raw(tmp_path):
    ledger = {}
    snapshots = []

    def persist():
        snapshots.append(copy.deepcopy(ledger["conditional"]))

    engine = GeneratedConditionalEngine(lambda: (
        snapshots[-1]["solve_requested"] is True
        and snapshots[-1]["entered"] is False) or pytest.fail(
            "solve entered before durable request receipt"))
    parameters = {"delta": np.zeros(2), "gain": np.zeros(1),
                  "log_tau": np.zeros(1)}
    raw = runner._conditional_forward(engine, parameters, tmp_path, ledger, persist)
    assert engine.solve_calls == 1
    assert ledger["conditional"]["solve_returned"] is True
    assert ledger["conditional"]["raw_saved"] is True
    assert runner.sha256(tmp_path / "conditional-forward-raw.npz") == (
        ledger["conditional"]["raw_sha256"])
    assert np.array_equal(raw["delta"], parameters["delta"])


def test_conditional_post_return_receipt_failure_keeps_raw_and_cost(tmp_path):
    ledger = {}
    receipt = FailingReceipt(3)
    engine = GeneratedConditionalEngine(lambda: None)
    parameters = {"delta": np.zeros(2), "gain": np.zeros(1),
                  "log_tau": np.zeros(1)}
    with pytest.raises(OSError, match="receipt-3"):
        runner._conditional_forward(engine, parameters, tmp_path, ledger, receipt)
    row = ledger["conditional"]
    assert engine.solve_calls == 1
    assert row["solve_returned"] is True
    assert row["post_return_error"] is True
    assert row["progress"]["completion_class"] == "known_complete"
    assert (tmp_path / "conditional-forward-raw.npz").exists()


@pytest.mark.parametrize("failed_call", [1, 2])
def test_durable_preentry_failure_is_known_zero(failed_call):
    receipt = FailingReceipt(failed_call)
    row = {}
    with pytest.raises(OSError, match=f"receipt-{failed_call}"):
        runner.durable_preentry(row, receipt, "generated", request_field="solve_requested")
    assert row["requested"] is False
    assert row["solve_requested"] is False
    assert row["entered"] is False
    assert row["known_zero_work"] is True


def test_accounted_operation_post_return_receipt_preserves_completed_work():
    receipt = FailingReceipt(3)
    calls = []
    rows = []
    with pytest.raises(OSError, match="receipt-3"):
        runner.accounted_operation("generated", lambda: calls.append(1) or 7,
                                   rows, receipt)
    assert calls == [1]
    assert rows[0]["entered"] is True
    assert rows[0]["returned"] is True
    assert rows[0]["post_return_error"] is True
    assert rows[0]["stage"] == "generated_post_return_receipt_failed"


def test_accounted_operation_callback_failure_keeps_primary():
    receipt = FailingReceipt()
    rows = []

    def fail():
        raise LookupError("primary-generated")

    with pytest.raises(LookupError, match="primary-generated"):
        runner.accounted_operation("generated", fail, rows, receipt)
    assert rows[0]["entered"] is True
    assert rows[0]["returned"] is False
    assert rows[0]["stage"] == "generated_failed"


class GeneratedIntervals:
    interval_steps = 3

    def interval(self):
        return self.interval_steps


class GeneratedOwner:
    def __init__(self, body=None):
        self.closed = False
        self.body = body

    def close(self):
        if self.body is not None:
            self.body.close()
        self.closed = True


class GeneratedSharedBody:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


def test_profiled_construction_counts_steps_sync_and_restores_profiler():
    receipt = FailingReceipt()
    rows = []
    generated = GeneratedIntervals()
    sync = []
    owner = GeneratedOwner()

    def previous(frame, event, arg):
        del frame, event, arg

    original = sys.getprofile()
    sys.setprofile(previous)
    try:
        result = runner.profiled_construction("generated",
            lambda: (generated.interval(), generated.interval(), owner)[-1],
            rows, receipt, interval_code=GeneratedIntervals.interval.__code__,
            synchronize=lambda: sync.append(1))
        assert sys.getprofile() is previous
    finally:
        sys.setprofile(original)
    assert result is owner and owner.closed is False
    assert sync == [1]
    assert rows[0]["interval_entries"] == 2
    assert rows[0]["interval_returns"] == 2
    assert rows[0]["known_completed_setup_steps"] == 6
    assert rows[0]["recorded_capture_setup_steps"] == 0
    assert rows[0]["unknown_setup_steps"] == 0


def test_profiled_construction_sync_failure_closes_returned_owner():
    receipt = FailingReceipt()
    rows = []
    generated = GeneratedIntervals()
    owner = GeneratedOwner()

    def bad_sync():
        raise RuntimeError("sync-generated")

    with pytest.raises(RuntimeError, match="sync-generated"):
        runner.profiled_construction("generated",
            lambda: (generated.interval(), owner)[-1], rows, receipt,
            interval_code=GeneratedIntervals.interval.__code__, synchronize=bad_sync)
    assert owner.closed is True
    assert rows[0]["entered_steps"] == 3
    assert rows[0]["known_completed_setup_steps"] == 0
    assert rows[0]["unknown_setup_steps"] == 3


def test_profiled_construction_exception_unwind_is_conservatively_unknown():
    receipt = FailingReceipt()
    rows = []

    class PartialInterval:
        interval_steps = 20

        def __init__(self):
            self.completed = 0

        def interval(self):
            for index in range(self.interval_steps):
                if index == 3:
                    raise ValueError("generated partial interval")
                self.completed += 1

    interval = PartialInterval()
    with pytest.raises(ValueError, match="generated partial interval"):
        runner.profiled_construction("partial", interval.interval, rows, receipt,
            interval_code=PartialInterval.interval.__code__,
            capture_probe=lambda: False, synchronize=lambda: None)
    assert interval.completed == 3
    assert rows[0]["interval_entries"] == 1
    assert rows[0]["interval_returns"] == 1
    assert rows[0]["known_completed_setup_steps"] == 0
    assert rows[0]["unknown_setup_steps"] == 20


def test_profiled_construction_separates_capture_from_executed_warmup():
    receipt = FailingReceipt()
    rows = []
    generated = GeneratedIntervals()
    states = iter((False, False, True))
    runner.profiled_construction("capture", lambda: (
        generated.interval(), generated.interval(), generated.interval()),
        rows, receipt, interval_code=GeneratedIntervals.interval.__code__,
        capture_probe=lambda: next(states), synchronize=lambda: None)
    row = rows[0]
    assert row["executed_interval_entries"] == 2
    assert row["captured_interval_entries"] == 1
    assert row["known_completed_setup_steps"] == 6
    assert row["recorded_capture_setup_steps"] == 3
    assert row["unknown_setup_steps"] == 0
    assert row["unknown_capture_setup_steps"] == 0


@pytest.mark.parametrize("label", ["apply_motor_gain", "apply_motor_timescale"])
@pytest.mark.parametrize("failure", ["sync", "receipt"])
def test_profiled_wrapper_transfers_shared_owner_before_post_return_failure(
        label, failure):
    body = GeneratedSharedBody()
    previous = GeneratedOwner(body)
    current = {"owner": previous}
    receipt = FailingReceipt(3 if failure == "receipt" else None)

    def transfer(_candidate):
        current["owner"] = None

    def synchronize():
        if failure == "sync":
            raise RuntimeError("generated synchronization failure")

    with pytest.raises((OSError, RuntimeError)):
        runner.profiled_construction(label,
            lambda: GeneratedOwner(previous.body), [], receipt,
            interval_code=GeneratedIntervals.interval.__code__,
            capture_probe=lambda: False, synchronize=synchronize,
            on_return=transfer)
    if current["owner"] is not None and not current["owner"].closed:
        current["owner"].close()
    assert body.close_calls == 1


class GeneratedBody:
    def __init__(self):
        self.time = 0.

    def physics_time(self):
        return self.time


class GeneratedNetwork:
    def __init__(self):
        self.tick = 0


class GeneratedSession:
    def __init__(self):
        self.control_tick = 0
        self.network = GeneratedNetwork()
        self.body = GeneratedBody()
        self.advance_calls = 0

    def advance(self, controls):
        self.advance_calls += 1
        self.control_tick += controls
        self.network.tick += 20 * controls
        self.body.time += .005 * controls


class GeneratedJournal:
    def __init__(self, failure=None):
        self.failure = failure
        self.rows = []

    def append(self, row):
        if len(self.rows) + 1 == self.failure:
            raise OSError("journal-generated")
        self.rows.append(copy.deepcopy(row))


def test_control_receipts_separate_request_clock_and_readout():
    session = GeneratedSession()
    clock = AdvanceLedger()
    journal = GeneratedJournal()
    event = runner.advance_control(session, "main", 1, clock, journal)
    runner.record_readout(journal, "main", 1, event, complete=True)
    assert [row["stage"] for row in journal.rows] == [
        "intent", "requested", "advance_returned", "readout_returned"]
    assert journal.rows[1]["entered"] is False
    assert journal.rows[2]["advance_returned"] is True
    assert journal.rows[2]["synchronized_readout"] is False
    assert journal.rows[3]["synchronized_readout"] is True
    assert session.network.tick == 20
    assert clock.neural_s == .005
    assert clock.physical_s == .005


def test_control_request_receipt_failure_performs_zero_work():
    session = GeneratedSession()
    clock = AdvanceLedger()
    journal = GeneratedJournal(failure=2)
    with pytest.raises(OSError, match="journal-generated"):
        runner.advance_control(session, "main", 1, clock, journal)
    assert session.advance_calls == 0
    assert session.network.tick == 0
    assert clock.events == []
    assert [row["stage"] for row in journal.rows] == ["intent"]


def test_control_return_receipt_failure_preserves_actual_clock():
    session = GeneratedSession()
    clock = AdvanceLedger()
    journal = GeneratedJournal(failure=3)
    with pytest.raises(OSError, match="journal-generated"):
        runner.advance_control(session, "main", 1, clock, journal)
    assert session.advance_calls == 1
    assert session.network.tick == 20
    assert len(clock.events) == 1
    assert clock.events[0]["clock_valid"] is True


def test_file_control_journal_summarizes_four_distinct_stages(tmp_path):
    journal = runner.ControlJournal(tmp_path / "journal.jsonl")
    try:
        for stage in ("intent", "requested", "advance_returned", "readout_returned"):
            journal.append({"owner": "main", "stage": stage})
        summary = journal.summary()
    finally:
        journal.close()
    assert summary["rows"] == 4
    assert summary["by_stage"] == {"intent": 1, "requested": 1,
        "advance_returned": 1, "readout_returned": 1}
    assert len((tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()) == 4


def _startup_rows(error):
    rows = [{"time_s": 0., "joint_targets_rad": np.zeros(42),
             "tendon_inputs": {f"{leg}_tarsus": 0. for leg in runner.LEGS}}]
    for index in range(1, 100):
        rows.append({"time_s": index * .005,
            "joint_targets_rad": np.full(42, error),
            "tendon_inputs": {f"{leg}_tarsus": 0. for leg in runner.LEGS}})
    return rows


def test_strict_startup_is_separate_from_historical_e121():
    startup = {"target_command": np.zeros((99, 42)),
               "target_tendon": np.zeros((99, 6))}
    rows = _startup_rows(.019)
    native = [{}] * 4950
    strict = runner.strict_startup(rows, native, startup,
        {"q_RMS": .018, "q_max_abs": .04, "LTM_RMS": .009,
         "LTM_max_abs": .025})
    historical = runner.e121.original_startup(rows, native, startup)
    assert strict["complete_original99_prefix"] is True
    assert strict["strict_E141_startup_screen"] is False
    assert historical["original_E100_E103_startup_screen"] is True


@pytest.mark.parametrize(("raw_errors", "analysis_errors"), [
    ([{"error": "raw"}], []), ([], [{"error": "analysis"}])])
def test_health_stop_with_late_technical_failure_is_invalid(
        raw_errors, analysis_errors):
    summary = {"complete_10s_native_health": False,
               "development_screens": {"F3_numeric": False,
                    "F4_numeric": False, "F5_numeric": False}}
    validity, verdict, technical = runner.native_outcome(
        "native_health_guard", True, summary, False,
        {"strict_E141_startup_screen": False}, raw_errors, analysis_errors)
    assert (validity, verdict, technical) == (
        "incomplete", "invalid_not_tested", True)


def test_health_stop_without_technical_failure_remains_observed_refutation():
    summary = {"complete_10s_native_health": False,
               "development_screens": {"F3_numeric": False,
                    "F4_numeric": False, "F5_numeric": False}}
    assert runner.native_outcome("native_health_guard", True, summary, False,
        {"strict_E141_startup_screen": False}, [], []) == (
            "measured_native_health_stop", "refuted_health_stop", False)


def test_first_control_health_stop_preserves_empty_signal_matrix_shapes():
    detailed = np.arange(446, dtype=np.int64)
    values = runner._signal_arrays([], detailed)
    for key in ("rate", "recurrent_current", "drive", "adaptation"):
        assert values[key].shape == (0, len(detailed))
        assert values[key].dtype == np.float32
    assert values["muscle_activation"].shape == (0, 96)
    assert values["muscle_activation"].dtype == np.float32
    rates0 = np.vstack((np.zeros((1, len(detailed))), values["rate"]))
    assert rates0.shape == (1, len(detailed))


def test_exact_arrays_requires_dtype_shape_and_bytes():
    expected = {"x": np.asarray([1, 2], np.float64)}
    runner.exact_arrays({"x": expected["x"].copy()}, expected, "generated")
    with pytest.raises(ValueError, match="changed"):
        runner.exact_arrays({"x": expected["x"].astype(np.float32)}, expected,
                            "generated")


def test_static_source_contract_has_frozen_layers_and_no_old_runner_call():
    path = Path(runner.__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    options = {node.args[0].value for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument" and node.args
        and isinstance(node.args[0], ast.Constant)}
    assert options == {"--freeze", "--freeze-sha256", "--source-review",
                       "--source-review-sha256", "--output"}
    assert "MotorAdaptationTrial(base" in source
    assert "MotorTimescaleTrial(motor" in source
    assert "MotorTimescaleTrial.from_checkpoint" in source
    assert "derivative=False" in source
    assert "e121.run_body" not in source
    assert "e121.save_raw" not in source
    assert "setattr(" not in source
    assert "accepted_for_single_original_native_transfer" in source
    assert "conditional-forward-raw.npz" in source
    assert "advance-journal.jsonl" in source
    assert "network-layer-identities.json" in source
    assert source.index("cuda_runtime_entered = True") < source.index(
        "engine = CudaAdaptationTimescaleEngine")
    assert source.count("on_return=transfer_") == 2


def test_construction_totals_never_turn_unknown_into_zero():
    rows = [{"returned": True, "interval_entries": 3, "interval_returns": 2,
             "known_completed_setup_steps": 40, "unknown_setup_steps": 20,
             "synchronize_returned": True}]
    total = runner._construction_totals(rows)
    assert total == {"attempts": 1, "returned": 1, "interval_entries": 3,
        "interval_returns": 2, "executed_interval_entries": 0,
        "captured_interval_entries": 0, "unknown_capture_state_entries": 0,
        "known_completed_setup_steps": 40, "recorded_capture_setup_steps": 0,
        "unknown_setup_steps": 20, "all_synchronized": True,
        "unknown_capture_setup_steps": 0,
        "all_profilers_restored": False}


def test_cleanup_preserves_first_failure_and_records_secondary():
    primary = ValueError("primary")
    secondary = []

    def fail():
        raise RuntimeError("cleanup")

    result = runner._close_owned(primary, (("generated", fail),), secondary)
    assert result is primary
    assert secondary[0]["stage"] == "generated"
    assert any("cleanup" in note for note in primary.__notes__)


def test_body_raw_attempts_are_independent_without_any_model(tmp_path):
    graph = SimpleNamespace(hash="generated-graph")
    neutral = {"network": {}, "parameters": {}}
    clock = AdvanceLedger()
    errors = runner._save_body_raw(tmp_path, graph, neutral, None, [], [], None,
                                   [], [], clock, {"experiment": "generated"}, [])
    assert any(item["artifact"] == "trace.json" for item in errors)
    assert (tmp_path / "clock-events.json").exists()
    assert (tmp_path / "clock-summary.json").exists()
    assert (tmp_path / "native-contact.npz").exists()
    assert (tmp_path / "clone-native-contact.npz").exists()


def test_hash_and_json_helpers_are_deterministic_and_reject_nan(tmp_path):
    path = tmp_path / "receipt.json"
    runner.write_json(path, {"value": np.int64(3), "path": Path("x")})
    assert json.loads(path.read_text(encoding="utf-8")) == {"path": "x", "value": 3}
    first = runner.sha256(path)
    assert first == runner.sha256(path)
    with pytest.raises(ValueError):
        runner.write_json(path, {"bad": math.nan})


def test_cuda_cleanup_guard_skips_preentry_and_cleans_entered_scope(tmp_path):
    calls = []

    class FakeCuda:
        @staticmethod
        def is_available():
            calls.append("available")
            return True

        @staticmethod
        def synchronize():
            calls.append("torch_sync")

        @staticmethod
        def empty_cache():
            calls.append("torch_empty")

    fake_torch = SimpleNamespace(cuda=FakeCuda())
    fake_device = SimpleNamespace(synchronize=lambda: calls.append("cupy_sync"))
    fake_pool = SimpleNamespace(free_all_blocks=lambda: calls.append("cupy_free"))
    fake_cupy = SimpleNamespace(cuda=SimpleNamespace(Device=lambda: fake_device),
                                get_default_memory_pool=lambda: fake_pool)
    ledger = {}
    assert runner.cleanup_cuda_runtime(False, ledger, torch_module=fake_torch,
                                       cupy_module=fake_cupy) == []
    assert calls == []
    assert ledger["cuda_cleanup"]["stage"] == "skipped_before_cuda_entry"
    assert runner.cleanup_cuda_runtime(True, ledger, torch_module=fake_torch,
                                       cupy_module=fake_cupy) == []
    assert calls == ["available", "torch_sync", "torch_empty",
                     "cupy_sync", "cupy_free"]
    assert ledger["cuda_cleanup"]["performed"] is True

    sync_calls = []
    fake_preentry_torch = SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: sync_calls.append("unexpected"),
        empty_cache=lambda: sync_calls.append("unexpected_empty")))
    with (patch.dict(sys.modules, {"torch": fake_preentry_torch}),
          pytest.raises(ValueError, match="Exact E141 v2 freeze SHA required")):
        runner.run(tmp_path / "unused-freeze.json", "bad-sha",
                   tmp_path / "unused-review.json", "bad",
                   tmp_path / "bad-gate")
    assert sync_calls == []
