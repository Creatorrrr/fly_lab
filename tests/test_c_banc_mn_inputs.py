"""Synthetic E188 observer/source/numeric contracts; no original model is run."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import observe_c_banc_m3_mn_inputs as e188
from tools.analyze_c_banc_m3_mn_current import conditioned
from tools.analyze_c_banc_m3_saved_current import warp_current


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def clone(self):
        return Tensor(self.value.copy())

    def numel(self):
        return self.value.size

    def __getitem__(self, key):
        return Tensor(self.value[key.value if isinstance(key, Tensor) else key])


class Weights:
    def __init__(self, ptr, pre, value):
        self.ptr = Tensor(ptr)
        self.pre = Tensor(pre)
        self.value = Tensor(value)

    def crow_indices(self):
        return self.ptr

    def col_indices(self):
        return self.pre

    def values(self):
        return self.value


class Network:
    device = "cuda"
    n = 158706
    cuda_implementation = "packed"
    identity = "network"

    def __init__(self, inventory):
        self.tick = 0
        self.rate = Tensor(np.arange(self.n, dtype=np.float32) / 1000)
        self.drive = Tensor(np.zeros(self.n, np.float32))
        self.adaptation = Tensor(np.zeros(self.n, np.float32))
        self.output_mask = Tensor(np.ones(self.n, np.float32))
        self.torch = SimpleNamespace(
            equal=lambda a, b: np.array_equal(a.value, b.value),
            as_tensor=lambda a, device=None: Tensor(np.asarray(a)),
        )
        ptr = np.zeros(self.n + 1, np.int64)
        ptr[e188.POST + 1 :] = e188.EDGES
        pre = np.array([row["pre_id"] for row in inventory], np.int64)
        value = np.array([row["actual_weight"] for row in inventory], np.float32)
        self.weights = Weights(ptr, pre, value)
        unique, inverse = np.unique(value.view(np.uint32), return_inverse=True)
        entries = pre.astype(np.uint32) | (inverse.astype(np.uint32) << 18)
        self._rowptr = Tensor(ptr.astype(np.int32))
        self._columns = Tensor(entries.view(np.int32))
        self._weight_values = Tensor(unique.view(np.float32))
        self.mutate = False

    def _current(self, rate):
        if self.mutate:
            self.rate.value[e188.POST] += 1
        value = np.zeros(self.n, np.float32)
        a = e188.POST
        pre = self.weights.pre.value
        value[a] = warp_current(
            self.weights.value.value, rate.value[pre], self.output_mask.value[pre]
        )
        return Tensor(value)


def inventory():
    return [
        {
            "edge_id": i,
            "post_id": e188.POST,
            "pre_id": 100 + i,
            "actual_weight": float(np.float32((i % 7 - 3) / 10)),
        }
        for i in range(e188.EDGES)
    ]


def metadata():
    return {
        "row_inventory": inventory(),
        "expected_graph_hash": "graph",
        "expected_network_identity": "network",
        "expected_adapter_identity": "adapter",
        "expected_body_hash": "body",
    }


def session(inventory=None):
    rows = inventory if inventory is not None else globals()["inventory"]()
    net = Network(rows)
    return SimpleNamespace(
        network=net,
        graph=SimpleNamespace(n=158706, hash="graph"),
        adapter=SimpleNamespace(identity="adapter"),
        body=SimpleNamespace(model_hash="body", physics_time=lambda: 0.0),
        control_tick=0,
    )


def test_runtime_csr_packed_order_and_weight_bits():
    rows = inventory()
    net = Network(rows)
    assert e188.validate_csr(net, rows)["packed_weight_bits_equal"] is True
    net.weights.pre.value[10], net.weights.pre.value[11] = (
        net.weights.pre.value[11],
        net.weights.pre.value[10],
    )
    with pytest.raises(ValueError, match="CSR edge/pre order"):
        e188.validate_csr(net, rows)
    net = Network(rows)
    net._columns.value[4] = np.int32(net._columns.value[4] ^ 1)
    with pytest.raises(ValueError, match="packed pre/lookup index"):
        e188.validate_csr(net, rows)
    net = Network(rows)
    net.weights.value.value[3] = np.nextafter(net.weights.value.value[3], np.float32(1))
    with pytest.raises(ValueError, match="runtime CSR weights"):
        e188.validate_csr(net, rows)


def test_factory_preserves_callback_return_exception_and_restoration():
    original_return = {"same": object()}
    calls = []

    def factory(*_a, **_k):
        def callback(*_args):
            calls.append("original")
            return original_return

        return callback

    fake = SimpleNamespace(e150=SimpleNamespace(make_observer=factory))
    collector = e188.Collector(metadata())
    restore = e188.install_factory(fake, collector)
    try:
        callback = fake.e150.make_observer([], original=None)
        assert callback(session(), np.array([], np.int64), {}) is original_return
        assert calls == ["original"] and len(collector.rows) == 1
    finally:
        restore()
    assert fake.e150.make_observer is factory

    def failing(*_a, **_k):
        def callback(*_args):
            raise RuntimeError("original failure")

        return callback

    fake.e150.make_observer = failing
    restore = e188.install_factory(fake, collector)
    try:
        with pytest.raises(RuntimeError, match="original failure"):
            fake.e150.make_observer()(session(), None, {})
        assert len(collector.rows) == 1
        assert "original failure" in collector.error
    finally:
        restore()
    assert fake.e150.make_observer is failing


def test_observer_rejects_clock_and_readout_mutation():
    collector = e188.Collector(metadata())
    sess = session()
    sess.network.tick = 1
    with pytest.raises(ValueError, match="clock/order"):
        collector.capture(sess)
    assert collector.rows[0]["complete"] is False
    assert collector.arrays()["network_tick"].tolist() == [1]
    collector = e188.Collector(metadata())
    sess = session()
    sess.network.mutate = True
    with pytest.raises(ValueError, match="readout changed state"):
        collector.capture(sess)
    assert len(collector.rows) == 1 and not collector.rows[0]["complete"]


def test_mid_read_failure_preserves_attempted_tick_and_partial_values(monkeypatch):
    collector = e188.Collector(metadata())
    original = e188.tensor_array

    def fail_after_first(value, dtype, shape, label):
        if label == "pre mask":
            raise RuntimeError("synthetic mid-read failure")
        return original(value, dtype, shape, label)

    monkeypatch.setattr(e188, "tensor_array", fail_after_first)
    with pytest.raises(RuntimeError, match="mid-read"):
        collector.capture(session())
    result = collector.arrays()
    assert result["control_tick"].tolist() == [0]
    assert result["complete"].tolist() == [False]
    assert result["pre_rate_available"].tolist() == [True]
    assert result["pre_mask_available"].tolist() == [False]
    assert np.isfinite(result["pre_rate"]).all()
    assert np.isnan(result["pre_mask"]).all()
    assert "mid-read" in result["row_error"][0]


def test_cleanup_and_persistence_failures_do_not_replace_primary(tmp_path, monkeypatch):
    collector = e188.Collector(metadata())

    def fail(message):
        raise RuntimeError(message)

    monkeypatch.setattr(
        e188.np, "savez_compressed", lambda *_a, **_k: fail("save failed")
    )

    class Monitor:
        def join(self, timeout):
            assert timeout == 5

        def is_alive(self):
            return False

    (tmp_path / "resource-monitor.jsonl").write_text(
        '{"phase":"monitor","resource_mode":"queue_empty","resource_clear":true,"queue":{"running":0,"pending":0},"gpu_csv":["2026/09/24 12:00:00.000, 0, 0, 842, 16384, 10, 300, 400"]}\n'
    )
    e182 = SimpleNamespace(
        _resource_rows=lambda path: [
            json.loads(path.read_text(encoding="utf-8").strip())
        ]
    )
    errors, check = e188.finish_observation(
        collector,
        tmp_path,
        lambda: fail("restore failed"),
        SimpleNamespace(restore_original_limits=lambda: fail("thread failed")),
        Monitor(),
        __import__("threading").Event(),
        [],
        e182,
    )
    assert len(errors) == 3
    assert check["resource_clear"] is True
    assert (tmp_path / "incoming-signals-uncompressed.npz").exists()
    assert (tmp_path / "resource-monitor-check.json").exists()
    assert (tmp_path / "observer-errors.json").exists()
    original = ValueError("original exception")
    with pytest.raises(ValueError, match="original exception"):
        e188.propagate_observation_failure(original, errors)
    with pytest.raises(ValueError, match="cleanup/persistence"):
        e188.propagate_observation_failure(None, errors)


def test_array_materialization_failure_still_finalizes_monitor(tmp_path, monkeypatch):
    collector = e188.Collector(metadata())
    monkeypatch.setattr(
        collector,
        "arrays",
        lambda: (_ for _ in ()).throw(RuntimeError("array materialization failed")),
    )

    class Monitor:
        def join(self, timeout):
            assert timeout == 5

        def is_alive(self):
            return False

    errors, check = e188.finish_observation(
        collector,
        tmp_path,
        None,
        None,
        Monitor(),
        __import__("threading").Event(),
        [],
        SimpleNamespace(
            _resource_rows=lambda _path: [
                {"phase": "monitor", "resource_mode": "queue_empty", "resource_clear": True, "queue": {"running": 0, "pending": 0}, "gpu_csv": ["2026/09/24 12:00:00.000, 0, 0, 842, 16384, 10, 300, 400"]}
            ]
        ),
    )
    assert errors[0]["artifact"] == "incoming_arrays"
    assert check["resource_clear"] is True
    assert (tmp_path / "resource-monitor-check.json").exists()


def test_decomposition_all_edges_cuda_bits_and_rounding():
    weights = np.asarray([r["actual_weight"] for r in inventory()], np.float32)
    base = np.asarray([np.float32(0.001 * i) for i in range(e188.EDGES)])
    rates = np.broadcast_to(base, (e188.ROWS, e188.EDGES)).copy()
    rates[100:, 7] *= np.float32(1.3)
    masks = np.ones_like(rates)
    masks[:, 14] = 0
    actual = np.array(
        [warp_current(weights, rates[i], masks[i]) for i in range(e188.ROWS)],
        np.float32,
    )
    known = np.zeros(e188.EDGES, bool)
    known[:10] = True
    values = e188.decomposition(
        {
            "pre_rate": rates,
            "pre_mask": masks,
            "weights": weights,
            "post_current": actual,
        },
        known,
    )
    assert (
        np.max(np.abs(values["full_sum"] - values["known_sum"] - values["unknown_sum"]))
        < 1e-12
    )
    assert np.array_equal(
        (values["full_sum"] + values["rounding_residual"])
        .astype(np.float32)
        .view(np.uint32),
        actual.view(np.uint32),
    )
    bad = actual.copy()
    bad[25] = np.nextafter(bad[25], np.float32(np.inf))
    with pytest.raises(ValueError, match="bit mismatch"):
        e188.decomposition(
            {
                "pre_rate": rates,
                "pre_mask": masks,
                "weights": weights,
                "post_current": bad,
            },
            known,
        )


def test_zero_force_count_is_nan_and_count_zero():
    values = np.array([[1.0, np.nan], [2.0, 3.0]])
    count = np.zeros((2, 6, 3), np.int64)
    count[0, 4, 2] = 4
    magnitude = np.zeros((2, 6), np.float64)
    result = conditioned(values, np.array([0, 1]), count, magnitude)
    assert np.isnan(result[0][0]).all()
    assert np.array_equal(result[1][0], [0, 0])
    assert result[0][2, 0] == 1 and np.isnan(result[0][2, 1])
    assert np.isnan(result[2]).all() and np.array_equal(result[3], [0, 0])


def test_raw_baseline_mismatch_fails(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        root.mkdir()
        (root / "trace.json").write_text(
            json.dumps({"samples": [{"control_tick": i} for i in range(e188.ROWS)]}),
            encoding="utf-8",
        )
        np.savez(root / "native-recruitment-state.npz", control_tick=np.arange(601))
        np.savez(root / "native-contact.npz", time_s=np.arange(30000))
        for boundary in ("start", "final"):
            (root / boundary).mkdir()
            np.save(root / boundary / "state.npy", np.array([1.0, 2.0]))
            (root / boundary / "state.json").write_text(
                json.dumps(
                    {"network": {"rate": {"__npy__": "state.npy"}}, "wall_s": 0}
                ),
                encoding="utf-8",
            )
    (b / "trace.json").write_text(
        json.dumps(
            {
                "samples": [
                    {"control_tick": i + (1 if i == 7 else 0)} for i in range(e188.ROWS)
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="trace raw mismatch"):
        e188.compare_baseline(a, b)


def test_source_gate_rejects_stale_hash_and_output_collision(tmp_path, monkeypatch):
    monkeypatch.setattr(e188, "ROOT", tmp_path)
    monkeypatch.setattr(e188, "DESIGN", "design.md")
    monkeypatch.setattr(e188, "METADATA", "metadata.json")
    monkeypatch.setattr(e188, "SUPERVISOR", "supervisor.py")
    monkeypatch.setattr(e188, "AMENDMENT", "amendment.md")
    monkeypatch.setattr(e188, "SERVICE_AMENDMENT", "service-amendment.md")
    monkeypatch.setattr(e188, "OUTPUT", "output")
    paths = [
        "design.md",
        "amendment.md",
        "service-amendment.md",
        "supervisor.py",
        *e188.OWNED,
        *e188.RESOURCE_OWNED,
        "source.py",
        "input.dat",
        "prior-e187.json",
        "prior-e185.json",
        "e187/analysis.json",
        "old-freeze.json",
        "old-review.json",
    ]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    (tmp_path / "prior-e187.json").write_text(
        json.dumps(
            {
                "schema": "flylab.m3.e187.parent-final-review.v1",
                "run_status": "completed",
                "run_validity": "valid_saved_current_diagnostic",
                "native_runs": 0,
                "boundary_current_bit_exact_rows": 25,
                "candidate_adopted": False,
            }
        )
    )
    (tmp_path / "prior-e185.json").write_text('{"saved_raw_analysis_accepted":true}')
    hashes = {
        name: e188.sha256(tmp_path / name)
        for name in (*e188.OWNED, *e188.RESOURCE_OWNED, "source.py")
    }
    inputs = {
        name: e188.sha256(tmp_path / name)
        for name in ("input.dat", "e187/analysis.json")
    }
    meta = {
        "schema": "flylab.m3.e188.all-incoming.metadata.v1",
        "design_path": "design.md",
        "design_sha256": e188.sha256(tmp_path / "design.md"),
        "owned_files": list(e188.OWNED),
        "reused_sources": {"source.py": hashes["source.py"]},
        "inputs": inputs,
        "original_native_runs": 0,
        "F6_exposed": 0,
        "candidate_adopted": False,
        "cumulative_before_s": e188.PARAMETERS["cumulative_before_s"],
        "cumulative_limit_s": e188.PARAMETERS["cumulative_cap_s"],
        "e187_directory": "e187",
        "previous_review_path": "prior-e187.json",
        "e185_review_path": "prior-e185.json",
        "e181_gate": {
            "freeze_path": "old-freeze.json",
            "freeze_sha256": e188.sha256(tmp_path / "old-freeze.json"),
            "source_review_path": "old-review.json",
            "source_review_sha256": e188.sha256(tmp_path / "old-review.json"),
        },
        "row_inventory": inventory(),
    }
    (tmp_path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    freeze = {
        "schema": "flylab.m3.e188.all-incoming.freeze.v1",
        "design_path": "design.md",
        "design_sha256": meta["design_sha256"],
        "metadata_path": "metadata.json",
        "metadata_sha256": e188.sha256(tmp_path / "metadata.json"),
        "output_path": "output",
        "parameters": e188.PARAMETERS,
        "parent_supervisor_sha256": e188.sha256(tmp_path / "supervisor.py"),
        "amendment_path": "amendment.md",
        "amendment_sha256": e188.sha256(tmp_path / "amendment.md"),
        "service_amendment_path": "service-amendment.md",
        "service_amendment_sha256": e188.sha256(tmp_path / "service-amendment.md"),
        "source_hashes": hashes,
        "input_sha256": inputs,
    }
    (tmp_path / "freeze.json").write_text(json.dumps(freeze), encoding="utf-8")
    review = {
        "schema": "flylab.m3.e188.all-incoming.source-review.v1",
        "approved": True,
        "blocking_defects": [],
        "freeze_sha256": e188.sha256(tmp_path / "freeze.json"),
        "source_hashes": hashes,
        "design_sha256": meta["design_sha256"],
        "metadata_sha256": freeze["metadata_sha256"],
        "amendment_sha256": freeze["amendment_sha256"],
        "service_amendment_sha256": freeze["service_amendment_sha256"],
    }
    (tmp_path / "review.json").write_text(json.dumps(review), encoding="utf-8")
    assert (
        e188.gate(
            tmp_path / "freeze.json", tmp_path / "review.json", tmp_path / "output"
        )[0]["row_inventory"]
        == meta["row_inventory"]
    )
    (tmp_path / "service-amendment.md").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="stopped service amendment changed"):
        e188.gate(
            tmp_path / "freeze.json", tmp_path / "review.json", tmp_path / "output"
        )
    (tmp_path / "service-amendment.md").write_text("{}", encoding="utf-8")
    (tmp_path / "prior-e187.json").write_text('{"run_validity":"valid"}')
    with pytest.raises(ValueError, match="E187 parent review"):
        e188.gate(
            tmp_path / "freeze.json", tmp_path / "review.json", tmp_path / "output"
        )
    (tmp_path / "prior-e187.json").write_text(
        json.dumps(
            {
                "schema": "flylab.m3.e187.parent-final-review.v1",
                "run_status": "completed",
                "run_validity": "valid_saved_current_diagnostic",
                "native_runs": 0,
                "boundary_current_bit_exact_rows": 25,
                "candidate_adopted": False,
            }
        )
    )
    (tmp_path / "input.dat").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="input changed"):
        e188.gate(
            tmp_path / "freeze.json", tmp_path / "review.json", tmp_path / "output"
        )
    (tmp_path / "output").mkdir()
    with pytest.raises(ValueError, match="output exists"):
        e188.gate(
            tmp_path / "freeze.json", tmp_path / "review.json", tmp_path / "output"
        )


def test_parent_readiness_pins_receipt_and_requires_fresh_complete_proof(tmp_path, monkeypatch):
    monkeypatch.setattr(e188, "ROOT", tmp_path)
    monkeypatch.setattr(e188, "PARENT_VALIDATION", "receipt.json")
    monkeypatch.setattr(e188.time, "perf_counter", lambda: 110.0)
    row = {"phase": "readiness", "ready": True, "resource_clear": True, "resource_mode": "service_stopped"}
    receipt = {"resource_readiness_finished_monotonic_s": 100.0, "resource_readiness": [row, row]}
    path = tmp_path / "receipt.json"

    def save():
        path.write_text(json.dumps(receipt), encoding="utf-8")
        return e188.sha256(path)

    digest = save()
    assert e188.parent_readiness(digest) == [row, row]
    receipt["resource_readiness_finished_monotonic_s"] = 99.99
    with pytest.raises(ValueError, match="expired"):
        e188.parent_readiness(save())
    receipt["resource_readiness_finished_monotonic_s"] = 111.0
    with pytest.raises(ValueError, match="expired"):
        e188.parent_readiness(save())
    receipt["resource_readiness_finished_monotonic_s"] = 105.0
    row["ready"] = False
    with pytest.raises(ValueError, match="not clear"):
        e188.parent_readiness(save())
    with pytest.raises(ValueError, match="receipt changed"):
        e188.parent_readiness(digest)


def test_terminal_write_failure_cannot_exit_success(tmp_path, monkeypatch):
    failure = OSError("terminal disk write failed")

    def fail(*_args):
        raise failure

    monkeypatch.setattr(e188, "write_new", fail)
    with pytest.raises(OSError) as captured:
        e188.persist_terminal(tmp_path, {"status": "complete"}, None)
    assert captured.value is failure


def test_terminal_write_failure_preserves_primary_exception(
    tmp_path, monkeypatch, capsys
):
    primary = RuntimeError("original observation failure")

    def fail(*_args):
        raise OSError("terminal disk write failed")

    monkeypatch.setattr(e188, "write_new", fail)
    with pytest.raises(RuntimeError) as captured:
        try:
            raise primary
        finally:
            e188.persist_terminal(tmp_path, {"status": "failed"}, primary)
    assert captured.value is primary
    assert "terminal disk write failed" in capsys.readouterr().err


@pytest.fixture
def parent_validation_fixture(tmp_path, monkeypatch):
    import psutil

    monkeypatch.setattr(e188, "ROOT", tmp_path)
    monkeypatch.setattr(e188, "SUPERVISOR", "supervisor.py")
    monkeypatch.setattr(e188, "PARENT_VALIDATION", "receipt.json")
    (tmp_path / "supervisor.py").write_text("# synthetic parent\n")
    (tmp_path / "input.txt").write_text("original bytes")
    parent = SimpleNamespace(
        is_running=lambda: True,
        create_time=lambda: 123.5,
        cwd=lambda: str(tmp_path),
        cmdline=lambda: ["python", str(tmp_path / "supervisor.py")],
    )
    monkeypatch.setattr(psutil, "Process", lambda _pid: parent)
    inputs = {"input.txt": e188.sha256(tmp_path / "input.txt")}
    receipt = {
        "schema": "flylab.m3.e188.parent-input-validation.v1",
        "validated": True,
        "freeze_sha256": "f" * 64,
        "source_review_sha256": "a" * 64,
        "metadata_sha256": "b" * 64,
        "input_sha256": inputs,
        "input_stamps": e188.input_stamps(inputs),
        "parent": {"pid": e188.os.getppid(), "create_time": 123.5},
        "issued_monotonic_s": e188.time.perf_counter(),
    }

    def validate():
        path = tmp_path / "receipt.json"
        path.write_text(json.dumps(receipt), encoding="utf-8")
        e188.validate_parent_inputs(
            inputs, "f" * 64, "a" * 64, "b" * 64, e188.sha256(path)
        )

    return receipt, parent, validate


def test_parent_receipt_requires_exact_live_supervisor(parent_validation_fixture):
    receipt, parent, validate = parent_validation_fixture
    validate()
    receipt["parent"]["pid"] += 1
    with pytest.raises(ValueError, match="different parent"):
        validate()
    receipt["parent"]["pid"] = e188.os.getppid()
    parent.create_time = lambda: 999.5
    with pytest.raises(ValueError, match="identity changed"):
        validate()
    parent.create_time = lambda: 123.5
    parent.is_running = lambda: False
    with pytest.raises(ValueError, match="identity changed"):
        validate()


def test_parent_receipt_rejects_expiry_and_wrong_command(parent_validation_fixture):
    receipt, parent, validate = parent_validation_fixture
    receipt["issued_monotonic_s"] -= 146
    with pytest.raises(ValueError, match="expired"):
        validate()
    receipt["issued_monotonic_s"] = e188.time.perf_counter()
    parent.cmdline = lambda: ["python", "another.py"]
    with pytest.raises(ValueError, match="supervisor command"):
        validate()


def test_parent_receipt_rejects_changed_input(tmp_path, parent_validation_fixture):
    _receipt, _parent, validate = parent_validation_fixture
    (tmp_path / "input.txt").write_text("changed input bytes")
    with pytest.raises(ValueError, match="input stat changed"):
        validate()


def test_parent_receipt_rejects_wrong_bindings(parent_validation_fixture, tmp_path):
    receipt, _parent, validate = parent_validation_fixture
    receipt["metadata_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="receipt contract"):
        validate()
    path = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="receipt SHA"):
        e188.validate_parent_inputs({}, "f" * 64, "a" * 64, "b" * 64, "0" * 64)
    assert path.is_file()


@pytest.mark.skipif(e188.os.name != "nt", reason="Windows venv launcher contract")
def test_parent_receipt_checks_single_venv_launcher(
    parent_validation_fixture, monkeypatch
):
    import psutil

    receipt, parent, validate = parent_validation_fixture
    receipt["parent"]["pid"] = e188.os.getppid() + 100
    arguments = ["-X", "utf8", "observer.py", "--parent-validation-sha256", "example"]
    current = SimpleNamespace(cmdline=lambda: ["base-python.exe", *arguments])
    launcher = SimpleNamespace(
        is_running=lambda: True,
        ppid=lambda: receipt["parent"]["pid"],
        exe=lambda: e188.sys.executable,
        cmdline=lambda: [e188.sys.executable, *arguments],
    )
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda pid=None: (
            current
            if pid is None
            else (parent if pid == receipt["parent"]["pid"] else launcher)
        ),
    )
    validate()
    launcher.cmdline = lambda: [e188.sys.executable, "unrelated.py"]
    with pytest.raises(ValueError, match="unverified venv launcher"):
        validate()
    launcher.cmdline = lambda: [e188.sys.executable, *arguments]
    launcher.exe = lambda: e188.sys.executable + ".foreign"
    with pytest.raises(ValueError, match="unverified venv launcher"):
        validate()
