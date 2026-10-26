"""Explicit synthetic CPU fixtures only; no original NPZ numerical payloads."""

import copy
import io
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from tools import verify_c_banc_m3_smooth_secant_body as body


def evaluation(candidate=True):
    return {
        "smooth": {
            "training_loss": 0.8 if candidate else 1.0,
            "startup_q_rms": 0.01,
            "startup_q_max": 0.02,
            "startup_tendon_rms": 0.004,
            "startup_tendon_max": 0.01,
        },
        "shape": {"shape_loss": 0.9 if candidate else 1.0},
        "summary": {
            "forward_mm": 1.7,
            "slip_mm_s": 1.79,
            "longest_continuous_forward_bout_s": 0.31,
            "complete_cycles": {leg: 1 for leg in ("lf", "lm", "lh", "rf", "rm", "rh")},
            "health": {"healthy": True},
        },
    }


def construction(ledger):
    ledger["construction"].append(
        {
            "known_completed_setup_steps": 320,
            "recorded_capture_setup_steps": 160,
            "unknown_setup_steps": 0,
            "unknown_capture_setup_steps": 0,
            "interval_entries": 24,
            "interval_returns": 24,
            "returned_executed_setup_steps": 320,
            "returned_capture_setup_steps": 160,
            "synchronize_returned": True,
            "profiler_restored": True,
            "ownership_transferred": True,
        }
    )
    ledger["body_construction"].append(
        {
            "known_completed_settling_steps": 2000,
            "unknown_step_entries": 0,
            "active_unreturned_frames": 0,
            "entered": 2000,
            "returned": 2000,
            "exception_unwinds": 0,
            "trace_restored": True,
        }
    )


class FakeBody:
    def __init__(self):
        self.time = 0.0
        self.fault = False
        self.neutral = np.zeros(42)
        self.steps = 0

    def pose(self):
        return np.array([self.time, 0.0, 0.0]), np.eye(3), np.zeros(6)

    def physics_time(self):
        return self.time

    def leg_observation(self):
        return {
            "angles_rad": np.full(42, self.time),
            "velocities_rad_s": np.ones(42),
            "support_load_bw": np.ones(6),
            "non_support_load_bw": np.zeros(6),
        }

    def knee_kinematics(self, *, include_direction=True):
        return {"flexion_rad": np.full(6, self.time), "velocity_rad_s": np.ones(6)}

    def _step_physics(self):
        self.steps += 1
        self.time += 0.0001


class FakeAdapter:
    def __init__(self, session, nodes):
        self.body = session.body
        self.nodes = nodes
        self.motor_ids = np.arange(391, dtype=np.int64)

    def encode(self, *, sensory_cut=False):
        observation = self.body.leg_observation()
        self.body.knee_kinematics(include_direction=False)
        return np.full(self.nodes, observation["angles_rad"][0] + 0.125, np.float32)


class FakeNetwork:
    def __init__(self):
        self.tick = 0
        self.fail = None

    def advance_readout(self, drive, ids, steps):
        self.tick += steps
        if self.fail == "partial_neural":
            error = RuntimeError("first unreturned native neural")
            error.receipt = {
                "entered_steps": steps,
                "returned_steps": 0,
                "unknown_steps": steps,
            }
            raise error
        return drive[ids].copy()


class FakeSession:
    def __init__(self, nodes=400):
        self.body = FakeBody()
        self.network = FakeNetwork()
        self.adapter = FakeAdapter(self, nodes)
        self.control_tick = 0
        self.closed = False
        self.close_calls = 0
        self.fault = False
        self.cuts = {"sensory": False, "motor": False}
        self.descending = np.array([0, 1], np.int64)
        self.p = SimpleNamespace(descending_drive=0.25)
        self.navigation = None
        self.fail = None

    def advance(self, controls):
        for _control in range(controls):
            self.body.pose()
            drive = self.adapter.encode(sensory_cut=False)
            drive[self.descending] += self.p.descending_drive
            self.network.advance_readout(drive, self.adapter.motor_ids, 20)
            for step in range(50):
                if self.fail == "partial_physics" and step == 4:
                    raise RuntimeError("first partial physics")
                self.body._step_physics()
            self.control_tick += 1
        return {"control_tick": self.control_tick}

    def close(self):
        self.close_calls += 1
        self.closed = True


class FakeClock:
    def __init__(self):
        self.events = []
        self.clock_valid = True
        self.neural_s = 0.0
        self.physical_s = 0.0


class FakeJournal:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeDependencies:
    def __init__(self, fault=None, nodes=400):
        self.session = FakeSession(nodes)
        self.fault = fault
        self.detached = False
        self.partial = False
        self.session.fail = fault
        self.session.network.fail = fault
        self.journal_object = FakeJournal()
        self.native = []
        self.raw_saved = False

    def load(self, _freeze, *, root):
        return {"ids": np.arange(2, dtype=np.int64)}

    def construct(self, _data, ledger, persist, on_return):
        construction(ledger)
        on_return(self.session)
        if self.fault == "constructor_persist":
            raise RuntimeError("first constructor persist")
        return self.session

    def initial(self, *_args):
        return {"generated_explicit_fixture": True}

    def save_state(self, path, session):
        if path.name == "partial-final":
            self.partial = True
        path.mkdir(exist_ok=False)
        body.put_json(
            path / "state.json",
            {
                "control_tick": session.control_tick,
                "network_tick": session.network.tick,
            },
        )

    def save_partial(self, *_args):
        pass

    def install(self, session, records):
        self.native = records
        original = session.body._step_physics

        def observed():
            original()
            if self.fault == "contact" and session.body.steps == 3:
                raise RuntimeError("first contact acquisition")
            records.append(session.body.time)

        session.body._step_physics = observed

        def detach():
            session.body._step_physics = original
            self.detached = True
            if self.fault == "detach":
                raise RuntimeError("first detach")

        return detach

    def new_clock(self):
        return FakeClock()

    def journal(self, path):
        return self.journal_object

    def capture(self, session, raw):
        if session.control_tick == 1 and self.fault == "capture":
            raise RuntimeError("first capture")
        raw["rows"].append({"control_tick": session.control_tick})
        raw["controls"].append(
            {
                "command": np.full(42, session.control_tick / 1000.0),
                "tendon": np.zeros(6),
            }
        )

    def advance(self, session, _arm, control, clock, _journal):
        session.advance(1)
        clock.neural_s += 0.005
        clock.physical_s += 0.005
        clock.events.append({"control": control})
        if self.fault == "journal":
            raise RuntimeError("first journal")
        if self.fault == "clock":
            clock.clock_valid = False
            raise RuntimeError("first native clock")
        return {"control": control}

    def readout_receipt(self, *_args):
        pass

    def health(self, *_args):
        if self.fault == "health":
            raise RuntimeError("first health stop")

    def save_raw(self, folder, raw, _data, _clock):
        self.raw_saved = True
        np.savez(
            folder / "control-raw.npz",
            command=np.stack([r["command"] for r in raw["controls"]]),
            tendon=np.stack([r["tendon"] for r in raw["controls"]]),
        )
        np.savez(folder / "native-contact.npz", time_s=np.asarray(raw["native"]))
        body.put_json(folder / "trace.json", {"samples": raw["rows"]})
        if self.fault == "raw_save":
            raise RuntimeError("first raw save")
        return []

    def evaluate(self, *_args):
        return {"candidate": evaluation(), "neutral": evaluation(False)}


def meter():
    return {
        "host_peak_bytes": 1,
        "device_free_bytes": 16 << 30,
        "device_total_bytes": 16 << 30,
        "torch_reserved_bytes": 0,
        "torch_peak_reserved_bytes": 0,
    }


def invoke(tmp_path, deps, **kwargs):
    output = tmp_path / "e335-generated"
    producer = tmp_path / "e335-generated-producer.json"
    body.put_json(producer, {"schema": "synthetic-fixture", "original_calls": 0})
    pins = {producer.name: body.sha256(producer)}
    return body.run(
        tmp_path / "freeze",
        "f",
        tmp_path / "review",
        "r",
        output,
        root=tmp_path,
        dependencies=deps,
        test_dimensions=body.TestDimensions(deps.session.adapter.nodes, 1),
        gatekeeper=lambda *_args, **_kwargs: {
            "source_sha256": pins,
            "input_sha256": pins,
        },
        meter=kwargs.pop("meter", meter),
        **kwargs,
    ), output


class FakeMemoryWatch:
    """Explicit CPU-only lifecycle proof; no SMI/CUDA process is started."""

    def __init__(self, fault=None):
        self.fault = fault
        self.started = False
        self.closed = False
        self.close_calls = 0
        self.peak = 7 << 20
        self.errors = []

    def start(self):
        self.started = True
        if self.fault == "start":
            self.errors.append("first sampler baseline")
            raise RuntimeError("first sampler baseline")
        return self

    def snapshot(self):
        return {
            "schema": "flylab.m3.e335.gpu-memory-watch.v1",
            "generated_cpu_only": True,
            "pre_context_baseline_bytes": 4 << 20,
            "sampled_peak_global_bytes": self.peak,
            "additional_sampled_including_context_bytes": max(0, self.peak - (4 << 20))
            + (1 << 20),
            "quantization_allowance_bytes": 1 << 20,
            "sample_count": int(self.started),
            "started": self.started,
            "closed": self.closed,
            "process_ended": self.closed,
            "stdout_thread_ended": self.closed,
            "stderr_thread_ended": self.closed and self.fault != "thread",
            "owned_pid": None,
            "errors": self.errors.copy(),
        }

    def close(self):
        self.close_calls += 1
        self.closed = True
        if self.fault == "close":
            self.errors.append("sampler close failed")
            raise OSError("sampler close failed")
        return self.snapshot()


class FakeSMIProcess:
    def __init__(self, stdout="4\n7\n5\n", stderr=""):
        self.pid = 123456
        self.stdout, self.stderr = io.StringIO(stdout), io.StringIO(stderr)
        self.returncode = None
        self.terminate_calls = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1
        self.returncode = 0

    def wait(self, timeout):
        return self.returncode


def test_public_gpu_sampler_cpu_process_fixture_baseline_peak_and_owned_close():
    process = FakeSMIProcess()
    calls = []

    def factory(argv, **kwargs):
        calls.append((argv, kwargs))
        return process

    watch = body.GPUMemoryWatch(process_factory=factory)
    assert not calls  # Constructor/import performs no SMI/Torch/CUDA work.
    watch.start()
    watch.thread.join(timeout=1)
    proof = watch.snapshot()
    assert proof["pre_context_baseline_bytes"] == 4 << 20
    assert proof["sampled_peak_global_bytes"] == 7 << 20
    assert proof["additional_sampled_including_context_bytes"] == 4 << 20
    assert proof["sample_count"] == 3 and not proof["closed"]
    assert "external GPU work cannot be excluded" in proof["measurement"]
    assert "unsampled peaks" in proof["measurement"]
    assert calls[0][0] == [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
        "--id=0",
        "--loop-ms=10",
    ]
    proof = watch.close()
    assert all(
        proof[k]
        for k in (
            "closed",
            "process_ended",
            "stdout_thread_ended",
            "stderr_thread_ended",
        )
    )
    assert proof["owned_pid"] == process.pid and proof["terminated_owned_process"]
    assert process.terminate_calls == 1
    assert watch.close() == proof and process.terminate_calls == 1
    with pytest.raises(RuntimeError, match="single-use"):
        watch.start()


@pytest.mark.parametrize(
    "failure", ["invalid_sample", "stderr", "unexpected_exit", "launch"]
)
def test_public_sampler_measurement_failure_preserved_on_owned_cleanup(failure):
    process = FakeSMIProcess(
        stdout="4\nN/A\n" if failure == "invalid_sample" else "4\n",
        stderr="SMI failure\n" if failure == "stderr" else "",
    )

    def factory(*_args, **_kwargs):
        if failure == "launch":
            raise OSError("no SMI")
        return process

    watch = body.GPUMemoryWatch(process_factory=factory)
    if failure == "launch":
        with pytest.raises((RuntimeError, OSError, ValueError)):
            watch.start()
    else:
        try:
            watch.start()
        except RuntimeError:
            assert failure in ("stderr", "invalid_sample")
    if watch.thread is not None:
        watch.thread.join(timeout=1)
    if failure == "unexpected_exit":
        process.returncode = 2
    with pytest.raises(RuntimeError):
        watch.close()
    proof = watch.snapshot()
    assert proof["errors"] and proof["closed"] and proof["process_ended"]
    assert proof["stdout_thread_ended"] and proof["stderr_thread_ended"]


def test_run_sampler_starts_before_meter_and_closed_evidence_matches_disk(tmp_path):
    watch = FakeMemoryWatch()
    states = []

    def tracked_meter():
        assert watch.started
        states.append(watch.closed)
        return meter()

    result, output = invoke(
        tmp_path, FakeDependencies(), meter=tracked_meter, watcher_factory=lambda: watch
    )
    assert result["run_valid"] and watch.close_calls == 1
    assert states[0] is False and states[-1] is True
    assert result["gpu_sampler"]["closed"] and not result["gpu_sampler"]["errors"]
    ledger = body.read(output / "advance-ledger.json")
    assert ledger["gpu_sampler"] == result["gpu_sampler"]
    assert ledger["additional_sampled_gpu_peak_bytes"] == 4 << 20
    assert ledger["additional_torch_peak_reserved_bytes"] == 0
    assert body.read(output / "result.json") == result
    assert result["cost"]["neural_host_returned"] == 6000


@pytest.mark.parametrize(
    "fault",
    [
        "start",
        "close",
        "thread",
        "postpublication_peak",
        "cleanup_deadline",
        "earlier_body_error",
    ],
)
def test_run_sampler_failures_revoke_and_preserve_first_error_work(tmp_path, fault):
    watch = FakeMemoryWatch(
        fault
        if fault in ("start", "close", "thread")
        else "close"
        if fault == "earlier_body_error"
        else None
    )
    now = [0.0]
    deps = FakeDependencies("contact" if fault == "earlier_body_error" else None)
    meter_calls = []

    def tracked_meter():
        meter_calls.append(True)
        return meter()

    def writer(path, value):
        body.put_json(path, value)
        if path.name == "result.json" and fault == "postpublication_peak":
            watch.peak = (4 << 20) + (4 << 30)

    original_close = watch.close

    def close():
        value = original_close()
        if fault == "cleanup_deadline":
            now[0] = 71.0
        return value

    watch.close = close
    result, output = invoke(
        tmp_path,
        deps,
        meter=tracked_meter,
        watcher_factory=lambda: watch,
        writer=writer,
        clock=lambda: now[0],
    )
    assert not result["run_valid"] and not result["handoff_eligible"]
    assert not result["effective_body_qualification"] and watch.close_calls == 1
    assert body.read(output / "result.json") == result
    assert (
        body.read(output / "advance-ledger.json")["gpu_sampler"]
        == result["gpu_sampler"]
    )
    if fault == "start":
        assert not meter_calls and result["cost"]["neural_host_returned"] == 0
        assert result["first_error"]["message"] == "first sampler baseline"
    elif fault == "earlier_body_error":
        assert result["first_error"]["message"] == "first contact acquisition"
        assert result["cost"]["neural_host_returned"] == 20
        assert result["cost"]["physics_steps_host_returned"] == 3
    else:
        assert (
            result["science_complete"]
            and result["cost"]["neural_host_returned"] == 6000
        )
        assert result["cost"]["physics_steps_host_returned"] == 15000
    assert result["gpu_sampler"]["closed"]


def test_actual_observation_feedback_and_original_call_order(tmp_path):
    deps = FakeDependencies()
    result, output = invoke(tmp_path, deps)
    assert result["run_valid"] and result["body_screen_supported"]
    assert (
        not result["candidate_adopted"]
        and not result["biological_gait_proven"]
        and not result["F1_F8_complete"]
    )
    ledger = body.read(output / "advance-ledger.json")
    assert ledger["neural_host_returned"] == ledger["neural_device_confirmed"] == 6000
    assert (
        ledger["physics_steps_host_returned"]
        == ledger["physics_steps_completed"]
        == 15000
    )
    assert ledger["controls_host_returned"] == ledger["controls_completed"] == 300
    assert (
        len(ledger["readout_returns"]) == 300
        and deps.session.close_calls == 1
        and deps.detached
    )
    assert "_step_physics" not in vars(deps.session.body) and "encode" not in vars(
        deps.session.adapter
    )
    raw = body.e334.load_selected(
        output / "live-input-001-025.npz",
        ("encoded_drive", "actual_drive", "motor_readout"),
    )
    assert raw["encoded_drive"][1, 5] > raw["encoded_drive"][0, 5]
    assert np.array_equal(raw["actual_drive"][:, 2:], raw["encoded_drive"][:, 2:])
    assert np.array_equal(
        raw["actual_drive"][:, :2], raw["encoded_drive"][:, :2] + np.float32(0.25)
    )
    assert np.array_equal(raw["actual_drive"][:, :391], raw["motor_readout"])
    events = body.read(output / "live-input-events.json")["events"]
    names = [r["event"] for r in events if r["control"] == 1]
    assert (
        names.index("leg_observation_returned")
        < names.index("encode_returned")
        < names.index("advance_readout_requested")
        < names.index("advance_readout_returned")
    )
    assert body.read(output / "result.json") == result


@pytest.mark.parametrize(
    "fault,neural,physics",
    [
        ("capture", 20, 50),
        ("journal", 20, 50),
        ("contact", 20, 3),
        ("partial_neural", 0, 0),
        ("partial_physics", 20, 4),
        ("clock", 20, 50),
        ("health", 20, 50),
        ("constructor_persist", 0, 0),
        ("detach", 6000, 15000),
        ("raw_save", 6000, 15000),
    ],
)
def test_failure_preserves_returned_work_first_error_and_own_cleanup(
    tmp_path, fault, neural, physics
):
    deps = FakeDependencies(fault)
    result, output = invoke(tmp_path, deps)
    assert not result["run_valid"] and not result["handoff_eligible"]
    assert result["first_error"]["message"].startswith("first")
    assert result["cost"]["neural_host_returned"] == neural
    assert result["cost"]["physics_steps_host_returned"] == physics
    assert deps.session.close_calls == 1
    if fault != "constructor_persist":
        assert deps.raw_saved and deps.journal_object.closed and deps.detached
    if fault == "contact":
        receipt = body.read(output / "receipts.json")
        assert (
            receipt["native_samples_observed"] == 2
            and receipt["physics_steps_host_returned"] == 3
        )
    if fault == "partial_neural":
        assert (
            deps.session.network.tick == 20
            and result["cost"]["neural_device_confirmed"] == 0
        )
        availability = body.read(output / "live-input-events.json")["chunks"][0][
            "availability"
        ][0]
        assert availability["neural_exception_receipt"]["unknown_steps"] == 20
        assert availability["unreturned_neural_work_unknown"] is True
    if fault in ("journal", "clock"):
        assert result["cost"]["controls_host_returned"] == 1
    assert body.read(output / "result.json") == result


def test_nan_actual_return_raw_retained_without_json_nan(tmp_path):
    session = FakeSession()
    ledger = body.new_ledger()
    session.network.advance_readout = lambda _drive, _ids, steps: np.full(
        391, np.nan, np.float32
    )
    observer = body.LiveInputObserver(
        session, tmp_path / "e335-generated-nan", ledger, nodes=400
    )
    observer.install()
    observer.begin(1)
    session.body.pose()
    drive = session.adapter.encode(sensory_cut=False)
    with pytest.raises(ValueError):
        session.network.advance_readout(drive, session.adapter.motor_ids, 20)
    observer.end()
    observer.publish()
    observer.detach()
    assert ledger["neural_host_returned"] == 20
    raw = body.e334.load_selected(
        observer.output / "live-input-001-001.npz", ("motor_readout",)
    )
    assert np.isnan(raw["motor_readout"]).all()
    assert "NaN" not in (observer.output / "live-input-events.json").read_text("utf8")


@pytest.mark.parametrize("sink", ["writer", "readback", "deadline"])
def test_publication_invalidates_qualification_and_preserves_disk(tmp_path, sink):
    deps = FakeDependencies()
    now = [0.0]
    changed = [False]

    def writer(path, value):
        body.put_json(path, value)
        if path.name == "result.json" and not changed[0]:
            changed[0] = True
            if sink == "writer":
                raise OSError("first terminal sink")
            if sink == "readback":
                path.write_text("{}", "utf8")
            if sink == "deadline":
                now[0] = 71.0

    result, output = invoke(tmp_path, deps, writer=writer, clock=lambda: now[0])
    assert (
        not result["run_valid"]
        and not result["effective_body_qualification"]
        and not result["handoff_eligible"]
    )
    assert (
        result["science_complete"]
        and result["first_error"]["stage"] == "terminal_publication"
    )
    assert body.read(output / "result.json") == result
    ledger = body.read(output / "advance-ledger.json")
    assert (
        ledger["first_error"] == result["first_error"]
        and ledger["absolute_clock"]["final_deadline_s"] == 10.0
    )
    assert result["cost"]["neural_host_returned"] == 6000


def test_both_sinks_fail_without_hiding_measured_science(tmp_path):
    def broken(path, value):
        if path.name == "result.json":
            raise OSError("terminal inaccessible")
        body.put_json(path, value)

    result, _output = invoke(
        tmp_path, FakeDependencies(), writer=broken, emergency_writer=broken
    )
    assert not result["run_valid"] and result["disk_agreement_established"] is False
    assert (
        result["first_error"]["message"] == "terminal inaccessible"
        and result["science_complete"]
    )
    assert result["cost"]["neural_device_confirmed"] == 6000


def native_fixture():
    data = np.array([-0.125, 0.25, 0.0, -0.0], np.float32)
    csr = {
        "data": data,
        "indices": np.array([1, 2, 3, 4], np.int32),
        "indptr": np.r_[0, 4, np.full(399, 4)].astype(np.int32),
    }
    native = {
        "csr": csr,
        "mask": {
            "positions": np.array([0, 1], np.int64),
            "edge_parameters": np.array([0, 1], np.int64),
        },
    }
    theta = np.array([1.0, 1.125], np.float64)
    patched = {k: v.copy() for k, v in csr.items()}
    patched["data"][1] = np.float32(np.float64(data[1]) * theta[1])
    bundle = {
        "positions": native["mask"]["positions"].copy(),
        "groups": native["mask"]["edge_parameters"].copy(),
        "original32": data[:2].copy(),
        "actual32": patched["data"][:2].copy(),
    }
    return (
        native,
        {"theta64": theta, "theta32": theta.astype(np.float32)},
        bundle,
        patched,
    )


def test_once_round_neutral_topology_signed_zero_palette():
    native, theta, selected, patched = native_fixture()
    info = body.verify_payload(
        native, theta, selected, patched, nodes=400, groups=2, expected_palette=4
    )
    assert (
        info["eligible"]
        and patched["data"].view(np.uint32)[2] != patched["data"].view(np.uint32)[3]
    )
    assert native["csr"]["data"][1] == np.float32(0.25)
    assert patched["data"][0].tobytes() == native["csr"]["data"][0].tobytes()


@pytest.mark.parametrize(
    "tamper", ["theta64", "theta32", "order", "groups", "dtype", "topology", "twice"]
)
def test_payload_tamper_rejected_before_construct(tamper):
    native, theta, selected, patched = native_fixture()
    if tamper == "theta64":
        theta["theta64"][1] = 1.1
    if tamper == "theta32":
        theta["theta32"][1] = 2
    if tamper == "order":
        selected["positions"] = selected["positions"][::-1].copy()
    if tamper == "groups":
        native["mask"]["edge_parameters"][1] = 0
        selected["groups"][1] = 0
    if tamper == "dtype":
        patched["data"] = patched["data"].astype(np.float64)
    if tamper == "topology":
        patched["indices"][0] = 2
    if tamper == "twice":
        patched["data"][1] *= np.float32(1.125)
    with pytest.raises(ValueError):
        body.verify_payload(
            native, theta, selected, patched, nodes=400, groups=2, expected_palette=4
        )


def packed_fixture():
    _native, _theta, _selected, csr = native_fixture()
    bits, inverse = np.unique(csr["data"].view(np.uint32), return_inverse=True)
    network = SimpleNamespace(
        cuda_implementation="packed",
        device="cpu generated",
        _rowptr=csr["indptr"].copy(),
        _columns=(
            (inverse.astype(np.uint32) << 18) | csr["indices"].astype(np.uint32)
        ).view(np.int32),
        _weight_values=bits.view(np.float32),
        _assert_pointers=lambda: None,
    )
    return network, csr


@pytest.mark.parametrize(
    "tamper", [None, "row", "column", "palette", "pointer", "dtype"]
)
def test_actual_packed_buffer_proof_is_independent_of_host_csr(tamper):
    network, csr = packed_fixture()
    if tamper == "row":
        network._rowptr[1] -= 1
    if tamper == "column":
        network._columns[0] ^= 1
    if tamper == "palette":
        network._weight_values[0] += np.float32(1.0)
    if tamper == "pointer":
        network._assert_pointers = lambda: (_ for _ in ()).throw(
            ValueError("captured address changed")
        )
    if tamper == "dtype":
        network._columns = network._columns.astype(np.int64)
    if tamper is None:
        proof = body.verify_packed(network, csr, expected_palette=4)
        assert (
            proof["captured_pointer_version_guard"]
            and proof["host_CSR_and_devicepacked_independently_compared"]
        )
    else:
        with pytest.raises(ValueError):
            body.verify_packed(network, csr, expected_palette=4)


def decoder_fixture():
    slopes = np.arange(96 * 391 * 3 * 2, dtype=np.float64).reshape(96, 391, 3, 2) / 1e6
    calibration = {
        "moment": np.zeros((42, 96)),
        "rest": np.zeros(42),
        "pool": np.zeros((96, 391)),
        "motor_ids": np.arange(391, dtype=np.int64),
        "ltm": np.arange(6, dtype=np.int64),
    }
    params = {
        "tau": np.array([0.005, 0.01, 0.04]),
        "kappa": np.ones(391),
        "membership": np.arange(391, dtype=np.int64) % 96,
        "slopes": slopes,
        "fixed_muscles": np.arange(6, dtype=np.int64),
    }
    state = {
        "parameters": params,
        "calibration": calibration,
        "bank": np.zeros((391, 3)),
        "activation": np.zeros(96),
        "original": {"activation": np.zeros(96)},
    }
    fixed = np.zeros(96, bool)
    fixed[:6] = True
    approved = {
        **calibration,
        **{k: params[k] for k in ("tau", "kappa", "membership")},
        "bank": state["bank"],
        "original_activation": state["original"]["activation"],
        "activation_tau": np.array(0.01),
        "neutral": np.zeros(42),
        "low": slopes[..., 0].reshape(96, 1173).copy(),
        "delta": (slopes[..., 1] - slopes[..., 0]).reshape(96, 1173).copy(),
        "fixed_mask": fixed,
    }
    adapter = SimpleNamespace(
        snapshot=lambda: copy.deepcopy(state),
        body=SimpleNamespace(neutral=np.zeros(42)),
    )
    return adapter, state, approved


@pytest.mark.parametrize(
    "tamper",
    [
        None,
        "low",
        "delta",
        "bank",
        "original_activation",
        "activation_tau",
        "neutral",
        "motor_ids",
    ],
)
def test_decoder_actual_constants_and_observed_semantic_leaves(tamper):
    adapter, saved, approved = decoder_fixture()
    if tamper is not None:
        approved = copy.deepcopy(approved)
        approved[tamper].flat[0] += 1
        with pytest.raises(ValueError):
            body.verify_decoder(adapter, saved, approved)
    else:
        proof = body.verify_decoder(adapter, saved, approved)
        assert proof["semantic_leaves"]["low"] == ["adapter", "parameters", "slopes"]
        assert proof["activation_tau_s"] == 0.01


def test_construction_setup_capture_settling_separate_and_unknown_invalid():
    ledger = body.new_ledger()
    construction(ledger)
    assert body.verify_construction(ledger)["capture_recorded_ticks_subset"] == 160
    ledger["construction"][0]["unknown_setup_steps"] = 1
    with pytest.raises(ValueError):
        body.verify_construction(ledger)


@pytest.mark.parametrize(
    "tamper",
    [
        "smooth",
        "startup",
        "shape",
        "forward",
        "slip",
        "bout",
        "cycles",
        "health",
        "engineering",
    ],
)
def test_all_scientific_gates_remain_conjunctive(tamper):
    candidate, neutral = evaluation(), evaluation(False)
    if tamper == "smooth":
        candidate["smooth"]["training_loss"] = 1.0
    if tamper == "startup":
        candidate["smooth"]["startup_q_max"] = 0.040001
    if tamper == "shape":
        candidate["shape"]["shape_loss"] = 1.0
    if tamper == "forward":
        candidate["summary"]["forward_mm"] = 1.6
    if tamper == "slip":
        candidate["summary"]["slip_mm_s"] = 1.81
    if tamper == "bout":
        candidate["summary"]["longest_continuous_forward_bout_s"] = 0.29
    if tamper == "cycles":
        candidate["summary"]["complete_cycles"]["rh"] = 0
    if tamper == "health":
        candidate["summary"]["health"]["healthy"] = False
    gates, passed = body.classify(
        candidate, neutral, engineering_valid=tamper != "engineering"
    )
    assert not passed and any(v is False for v in gates.values())


def test_collision_and_gate_rejection_precede_payload(monkeypatch, tmp_path):
    pins = {"source": "old"}
    with pytest.raises(ValueError):
        body.merge_pins(pins, {"source": "new"})
    assert pins == {"source": "old"}

    def prohibited(*_args, **_kwargs):
        raise AssertionError("payload must not be read")

    monkeypatch.setattr(np, "load", prohibited)
    freeze = tmp_path / "e335-generated-freeze.json"
    review = tmp_path / "e335-generated-review.json"
    body.put_json(freeze, {"schema": "wrong"})
    body.put_json(review, {})
    with pytest.raises(ValueError):
        body.gate(
            freeze,
            body.sha256(freeze),
            review,
            body.sha256(review),
            body.safe(body.OUTPUT, tmp_path),
            root=tmp_path,
        )
    with pytest.raises(ValueError):
        body.run(
            freeze,
            "f",
            review,
            "r",
            tmp_path / "e335-generated-out",
            dependencies=FakeDependencies(),
        )
    with pytest.raises(SystemExit):
        body.main(["--nodes", "400"])


def test_generated_fullsize_300_live_chunk_io_hash_and_readback(tmp_path):
    output = tmp_path / "e335-generated-fullsize-io"
    session = FakeSession(body.N)
    ledger = body.new_ledger()
    observer = body.LiveInputObserver(session, output, ledger, nodes=body.N)
    observer.install()
    began = time.perf_counter()
    for control in range(1, 301):
        observer.begin(control)
        session.body.pose()
        drive = session.adapter.encode(sensory_cut=False)
        drive[session.descending] += session.p.descending_drive
        session.network.advance_readout(drive, session.adapter.motor_ids, 20)
        session.control_tick = control
        session.body.time = control * 0.005
        observer.end(returned=True)
    observer.publish()
    observer.detach()
    generated_s = time.perf_counter() - began
    chunks = sorted(output.glob("live-input-*.npz"))
    assert len(chunks) == 12
    observed = 0
    read_begin = time.perf_counter()
    for path in chunks:
        desc = next(c for c in observer.chunks if c["file"] == path.name)
        assert body.sha256(path) == desc["sha256"]
        with np.load(path, allow_pickle=False) as archive:
            assert (
                archive["encoded_drive"].shape
                == archive["actual_drive"].shape
                == (25, body.N)
            )
            assert archive["motor_readout"].shape == (25, 391)
            assert np.array_equal(
                archive["actual_drive"][:, :2],
                archive["encoded_drive"][:, :2] + np.float32(0.25),
            )
            for key, array_desc in desc["arrays"].items():
                assert body.e334._metadata(archive[key]) == array_desc
        observed += 25
    assert observed == 300 and observer.io["raw_payload_bytes"] >= 300 * body.N * 4 * 2
    report = {
        "schema": "flylab.m3.e335.sol-generated-io.v1",
        "passed": True,
        "original_calls": 0,
        "body_calls": 0,
        "nodes": body.N,
        "controls": 300,
        "chunk_count": 12,
        "generated_write_s": generated_s,
        "readback_hash_s": time.perf_counter() - read_begin,
        "output_bytes": body.recursive_bytes(output),
        "io": observer.io,
        "original60second_completion_proven": False,
    }
    # Retain the report in the fresh test-output directory and print metadata
    # into the supervisor-owned stdout, without copying large fixture arrays.
    body.put_json(output / "io-report.json", report)
    print(
        "E335_SOL_GENERATED_IO " + json.dumps(report, allow_nan=False, sort_keys=True)
    )


def previous_fixture(tmp_path):
    """All fabricated producer JSON and byte descriptors; no original payload."""
    pins = {path: "a" * 64 for path in body.PATHS.values()}
    sources = {"synthetic-producer.py": "b" * 64}
    prior = {
        "source_sha256": sources.copy(),
        "input_sha256": {"synthetic-input": "c" * 64},
    }
    rec = {
        "confirmed": {
            "forward_controls": 300,
            "forward_neural_ticks": 6000,
            "recompute_controls": 0,
            "recompute_neural_ticks": 0,
            "backward_blocks": 0,
            "backward_block_ticks": 0,
        },
        "confirm_returned": 1,
        "confirm_requested": 1,
        "forward_neural_ticks_requested": 6000,
        "forward_neural_ticks_returned": 6000,
    }
    receipts = {
        "schema": "flylab.m3.e334.receipts.v1",
        "receipts": {"continuation-star": rec},
    }
    result = {
        "schema": body.e334.RESULT_SCHEMA,
        "run_valid": True,
        "effective_neural_qualification": True,
        "handoff_eligible": True,
        "candidate_adopted": False,
        "F1_F8_complete": False,
        "body_calls": 0,
        "first_error": None,
        "errors": [],
        "provenance": {
            "freeze_sha256": pins[body.PATHS["e334_freeze"]],
            "review_sha256": pins[body.PATHS["e334_review"]],
        },
        "receipts": receipts["receipts"],
    }
    ledger = {
        k: copy.deepcopy(result[k])
        for k in (
            "run_valid",
            "effective_neural_qualification",
            "handoff_eligible",
            "first_error",
            "errors",
            "receipts",
        )
    }
    ledger["schema"] = "flylab.m3.e334.advance-ledger.v1"
    decision = {
        "schema": "flylab.m3.e334.root-final-decision.v1",
        **{
            key: True
            for key in (
                "run_valid",
                "effective_neural_qualification",
                "root_independent_raw_qualification",
                "handoff_eligible_for_same_candidate_body_design",
                "cost_charged_once",
                "owned_original_and_audit_processes_ended",
            )
        },
        "original_cumulative_s": 9431.526588899522,
        "original_allocation_limit_s": 9660,
        "original_reserved_s": 0,
        "whole_goal_status": "active",
        "body_calls": 0,
        "candidate_adopted": False,
        "F1_F8_complete": False,
        "raw_checks": 5094,
        "original_supervisor_elapsed_s": 25.0,
        "root_saved_audit_CPU_failed_included_s": 10.433235699998477,
    }
    producer = {
        "schema": "flylab.m3.e334.source-manifest.v1",
        **prior,
        **result["provenance"],
    }
    files = {}
    for name in (*body.e334.RAW_FILES, "source-manifest.json", "receipts.json"):
        path = body.safe(body.PREVIOUS + "/" + name, tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"explicit synthetic non-NPZ bytes")
        digest = body.sha256(path)
        pins[body.PREVIOUS + "/" + name] = digest
        files[name] = {"sha256": digest, "bytes": path.stat().st_size}
    audit = {
        "schema": "flylab.m3.e334.root-original-raw-audit.v1",
        "passed": True,
        "checks": {str(i): True for i in range(5094)},
        "audit_source_sha256": pins[body.PATHS["e334_audit_source"]],
        "freeze_sha256": pins[body.PATHS["e334_freeze"]],
        "review_sha256": pins[body.PATHS["e334_review"]],
        "supervision_sha256": pins[body.PATHS["e334_supervision"]],
        "terminal_sha256": {
            name: pins.get(body.PREVIOUS + "/" + name, "a" * 64)
            for name in (
                "result.json",
                "advance-ledger.json",
                "output-manifest.json",
                "source-manifest.json",
                "receipts.json",
            )
        },
    }
    manifest = {"schema": "flylab.m3.e334.output-manifest.v1", "files": files}
    proofs = {
        "freeze": prior,
        "decision": decision,
        "result": result,
        "ledger": ledger,
        "producer": producer,
        "receipts": receipts,
        "audit": audit,
        "manifest": manifest,
    }
    for role, original in (("e334", 1), ("e334_audit", 0)):
        proofs[role + "_contract"] = {"original_calls": original}
        proofs[role + "_supervision"] = {
            "schema": "flylab.m3.e267.root-supervision.v1",
            "success": True,
            "attempts": [{"exit_code": 0, "killed": False}],
            "owned_pids": [123],
            "original_calls": original,
            "contract_sha256": pins[body.PATHS[role + "_contract"]],
            "elapsed_s": 25.0,
            "cumulative_after_s": 9431.526588899522,
            "cumulative_limit_s": 9660,
        }
    return proofs, pins, sources


@pytest.mark.parametrize(
    "tamper",
    [
        None,
        "first_audit",
        "false_check",
        "terminal",
        "producer",
        "supervisor",
        "contract",
        "receipt",
        "ledger",
        "roster",
        "decision",
    ],
)
def test_generated_previous_terminal_producer_v2_exact_links(tmp_path, tamper):
    proofs, pins, sources = previous_fixture(tmp_path)
    if tamper == "first_audit":
        proofs["audit"]["passed"] = False
        proofs["audit"]["formal_qualification"] = True
    if tamper == "false_check":
        proofs["audit"]["checks"]["0"] = False
    if tamper == "terminal":
        proofs["audit"]["terminal_sha256"]["result.json"] = "d" * 64
    if tamper == "producer":
        proofs["producer"]["source_sha256"] = {"replaced": "d" * 64}
    if tamper == "supervisor":
        proofs["e334_audit_supervision"]["success"] = False
    if tamper == "contract":
        proofs["e334_contract"]["original_calls"] = 0
    if tamper == "receipt":
        proofs["receipts"]["receipts"]["continuation-star"]["confirm_returned"] = 0
    if tamper == "ledger":
        proofs["ledger"]["handoff_eligible"] = False
    if tamper == "roster":
        proofs["manifest"]["files"].pop("theta-star.npz")
    if tamper == "decision":
        proofs["decision"]["body_calls"] = 1
    if tamper is None:
        body.verify_previous(proofs, pins, sources, root=tmp_path)
    else:
        with pytest.raises(ValueError):
            body.verify_previous(proofs, pins, sources, root=tmp_path)


def qualification_fixture(tmp_path, *, first_requires_r1=False):
    pins = {}
    sources = {}

    def save(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        body.put_json(path, value)
        digest = body.sha256(path)
        pins[name] = digest
        return {"path": name, "sha256": digest}

    for name in body.OWNED:
        sources[name] = save(name, {"synthetic_source": True})["sha256"]
    helper = save("e335-generated-helper.py", {"synthetic_audit": True})
    qual = {
        "schema": body.QUAL_SCHEMA,
        "approved": True,
        "blocking_defects": [],
        "source_sha256": sources,
        "original_calls": 0,
        "evidence_sha256": {},
        "root_review_sequence": {},
    }
    for role, scope in (
        ("cpu", "fake-body"),
        ("cuda", "native-ssp"),
        ("decoder", "native-functional-decoder"),
    ):
        report_name = "e335-generated-" + role + "-report.json"
        argv = [
            ".venv/Scripts/python.exe",
            "-X",
            "utf8",
            "-B",
            helper["path"],
            "--output",
            report_name,
        ]
        contract = save(
            "e335-generated-" + role + "-contract.json",
            {
                "original_calls": 0,
                "source_sha256": {**sources, helper["path"]: helper["sha256"]},
                "commands": [{"argv": argv}],
            },
        )
        supervision = save(
            "e335-generated-" + role + "-supervision.json",
            {
                "schema": "flylab.m3.e267.root-supervision.v1",
                "success": True,
                "attempts": [{"exit_code": 0, "killed": False, "argv": argv}],
                "owned_pids": [123],
                "original_calls": 0,
                "contract_sha256": contract["sha256"],
            },
        )
        report = save(
            report_name,
            {
                "schema": "flylab.m3.e335.root-generated.v1",
                "scope": scope,
                "passed": True,
                "checks": {"fixture": True},
                "source_sha256": sources,
                "audit_source_path": helper["path"],
                "audit_source_sha256": helper["sha256"],
                "original_calls": 0,
                "whole_BANC_calls": 0,
                "body_calls": 0,
            },
        )
        qual[role + "_contract"] = contract
        qual[role + "_supervision"] = supervision
        qual[role + "_report"] = report
    seq = qual["root_review_sequence"]
    seq["FIRST"] = save(
        "e335-generated-first.json",
        {
            "schema": "flylab.m3.e335.root-first-review.v1",
            "reviewed_source_sha256": sources,
            "approved": not first_requires_r1,
            "blocking_defects": ["preserved FIRST"] if first_requires_r1 else [],
        },
    )
    seq["SECOND"] = save(
        "e335-generated-second.json",
        {
            "schema": "flylab.m3.e335.root-second-review.v1",
            "current_source_sha256": sources,
            "approved": True,
            "blocking_defects": [],
        },
    )
    seq["reflection"] = save(
        "e335-generated-reflection.json",
        {
            "schema": "flylab.m3.e335.root-final-source-reflection.v1",
            "current_source_sha256": sources,
            "approved": True,
            "blocking_defects": [],
            "actual_astra_reflection_sha256": body.ANCHORS["reflection"],
            "first_review_sha256": seq["FIRST"]["sha256"],
            "second_review_sha256": seq["SECOND"]["sha256"],
        },
    )
    qual["evidence_sha256"] = pins.copy()
    return qual, sources, pins


@pytest.mark.parametrize(
    "tamper",
    [
        None,
        "scope",
        "source",
        "checks",
        "supervisor",
        "producer",
        "missing_first",
        "second_source",
        "reflection",
        "first_requires_r1",
    ],
)
def test_current_generated_scope_producers_and_root_review_sequence(tmp_path, tamper):
    qual, sources, pins = qualification_fixture(
        tmp_path, first_requires_r1=tamper == "first_requires_r1"
    )
    if tamper in ("scope", "source", "checks"):
        path = tmp_path / qual["cuda_report"]["path"]
        value = body.read(path)
        if tamper == "scope":
            value["scope"] = "fake-body"
        if tamper == "source":
            value["source_sha256"] = {"old": "d" * 64}
        if tamper == "checks":
            value["checks"]["fixture"] = False
        body.put_json(path, value)
    if tamper == "supervisor":
        path = tmp_path / qual["cpu_supervision"]["path"]
        value = body.read(path)
        value["attempts"][0]["killed"] = True
        body.put_json(path, value)
    if tamper == "producer":
        qual["decoder_contract"]["sha256"] = "d" * 64
    if tamper == "missing_first":
        qual["root_review_sequence"].pop("FIRST")
    if tamper == "second_source":
        path = tmp_path / qual["root_review_sequence"]["SECOND"]["path"]
        value = body.read(path)
        value["current_source_sha256"] = {}
        body.put_json(path, value)
    if tamper == "reflection":
        path = tmp_path / qual["root_review_sequence"]["reflection"]["path"]
        value = body.read(path)
        value["actual_astra_reflection_sha256"] = "d" * 64
        body.put_json(path, value)
    if tamper in (None, "first_requires_r1"):
        body._generated_proof(qual, sources, pins, root=tmp_path)
    else:
        with pytest.raises(ValueError):
            body._generated_proof(qual, sources, pins, root=tmp_path)


def refresh_producer_fixture(tmp_path, qual, pins, role, contract, supervision):
    """Rehash the fabricated producer DAG so rejection reaches argv binding."""
    contract_desc, supervision_desc = (
        qual[role + "_contract"],
        qual[role + "_supervision"],
    )
    body.put_json(tmp_path / contract_desc["path"], contract)
    contract_desc["sha256"] = body.sha256(tmp_path / contract_desc["path"])
    pins[contract_desc["path"]] = contract_desc["sha256"]
    supervision["contract_sha256"] = contract_desc["sha256"]
    body.put_json(tmp_path / supervision_desc["path"], supervision)
    supervision_desc["sha256"] = body.sha256(tmp_path / supervision_desc["path"])
    pins[supervision_desc["path"]] = supervision_desc["sha256"]
    qual["evidence_sha256"] = pins.copy()


@pytest.mark.parametrize(
    "tamper",
    [
        "helper_missing",
        "other_helper",
        "helper_pin_bytes",
        "helper_file_bytes",
        "other_report",
        "attempt_argv",
        "slash_mismatch",
        "unrelated_python_c",
        "python_flags",
        "duplicate_output",
        "missing_output",
        "drive_relative_helper",
        "outside_helper",
        "alias_collision",
    ],
)
def test_generated_execution_rehash_tamper_binds_actual_helper_and_report(
    tmp_path, tamper
):
    qual, sources, pins = qualification_fixture(tmp_path)
    contract = body.read(tmp_path / qual["cpu_contract"]["path"])
    supervision = body.read(tmp_path / qual["cpu_supervision"]["path"])
    report = body.read(tmp_path / qual["cpu_report"]["path"])
    helper, argv = report["audit_source_path"], contract["commands"][0]["argv"]
    if tamper == "helper_missing":
        contract["source_sha256"].pop(helper)
    if tamper == "other_helper":
        other = tmp_path / "e335-generated-other-helper.py"
        other.write_bytes((tmp_path / helper).read_bytes())
        contract["source_sha256"][other.name] = body.sha256(other)
        argv[4] = other.name
    if tamper == "helper_pin_bytes":
        contract["source_sha256"][helper] = "d" * 64
    if tamper == "helper_file_bytes":
        (tmp_path / helper).write_bytes(b"changed synthetic helper bytes")
    if tamper == "other_report":
        argv[6] = "e335-generated-unrelated-report.json"
    if tamper == "unrelated_python_c":
        contract["commands"][0]["argv"] = [
            ".venv/Scripts/python.exe",
            "-c",
            "print('unrelated')",
        ]
    if tamper == "python_flags":
        argv[3] = "-E"
    if tamper == "duplicate_output":
        argv.extend(["--output", "e335-generated-unrelated-report.json"])
    if tamper == "missing_output":
        del argv[5:]
    if tamper == "drive_relative_helper":
        argv[4] = "C:e335-generated-helper.py"
    if tamper == "outside_helper":
        argv[4] = "../e335-generated-helper.py"
    if tamper == "alias_collision":
        contract["source_sha256"]["./" + helper] = "d" * 64
    supervision["attempts"][0]["argv"] = copy.deepcopy(contract["commands"][0]["argv"])
    if tamper == "attempt_argv":
        supervision["attempts"][0]["argv"] = ["different supervisor argv"]
    if tamper == "slash_mismatch":
        supervision["attempts"][0]["argv"][0] = ".venv\\Scripts\\python.exe"
    refresh_producer_fixture(tmp_path, qual, pins, "cpu", contract, supervision)
    assert supervision["contract_sha256"] == qual["cpu_contract"]["sha256"]
    with pytest.raises(ValueError):
        body._generated_proof(qual, sources, pins, root=tmp_path)


@pytest.mark.parametrize(
    "alias", ["absolute_backslashes", "relative_dot", "equal_output_option"]
)
def test_generated_execution_safe_windows_path_aliases(tmp_path, alias):
    qual, sources, pins = qualification_fixture(tmp_path)
    contract = body.read(tmp_path / qual["cpu_contract"]["path"])
    supervision = body.read(tmp_path / qual["cpu_supervision"]["path"])
    report = body.read(tmp_path / qual["cpu_report"]["path"])
    argv = contract["commands"][0]["argv"]
    if alias == "absolute_backslashes":
        for index in (0, 4, 6):
            argv[index] = str((tmp_path / argv[index]).resolve()).replace("/", "\\")
        digest = contract["source_sha256"].pop(report["audit_source_path"])
        contract["source_sha256"][
            str((tmp_path / report["audit_source_path"]).resolve())
        ] = digest
    if alias == "relative_dot":
        argv[4] = ".\\" + argv[4]
        argv[6] = "./" + argv[6]
        digest = contract["source_sha256"].pop(report["audit_source_path"])
        contract["source_sha256"][".\\" + report["audit_source_path"]] = digest
    if alias == "equal_output_option":
        argv[5:] = ["--output=" + argv[6], "--fixtures", "e335-generated-fixtures"]
    supervision["attempts"][0]["argv"] = copy.deepcopy(argv)
    refresh_producer_fixture(tmp_path, qual, pins, "cpu", contract, supervision)
    body._generated_proof(qual, sources, pins, root=tmp_path)


@pytest.mark.parametrize("tamper", [False, True])
def test_generated_cuda_decoder_share_one_exact_completed_command(tmp_path, tamper):
    qual, sources, pins = qualification_fixture(tmp_path)
    contract = body.read(tmp_path / qual["cuda_contract"]["path"])
    supervision = body.read(tmp_path / qual["cuda_supervision"]["path"])
    argv = contract["commands"][0]["argv"]
    argv.extend(["--decoder-output", qual["decoder_report"]["path"]])
    if tamper:
        argv[-1] = "e335-generated-other-decoder-report.json"
    supervision["attempts"][0]["argv"] = copy.deepcopy(argv)
    refresh_producer_fixture(tmp_path, qual, pins, "cuda", contract, supervision)
    qual["decoder_contract"] = copy.deepcopy(qual["cuda_contract"])
    qual["decoder_supervision"] = copy.deepcopy(qual["cuda_supervision"])
    if tamper:
        with pytest.raises(ValueError):
            body._generated_proof(qual, sources, pins, root=tmp_path)
    else:
        body._generated_proof(qual, sources, pins, root=tmp_path)
