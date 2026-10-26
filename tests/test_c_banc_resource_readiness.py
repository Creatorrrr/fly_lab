"""Synthetic resource evidence only; these tests never run BANC or CUDA."""

from __future__ import annotations

import errno
import json
import os
import socket
import threading
from types import SimpleNamespace
from typing import ClassVar

import psutil
import pytest

from tools import banc_m3_resource_readiness as readiness

GPU = "2026/09/24 12:00:00.000, 0, 0, 842, 16384, 10, 300, 400"


def source(queue=None, queue_error=None, gpu=GPU, clients=None):
    def sample():
        row = {"gpu_csv": [gpu]}
        if queue_error:
            row["queue_error"] = queue_error
        else:
            row["queue"] = queue if queue is not None else readiness.EMPTY_QUEUE
        return row

    return SimpleNamespace(
        _sample_resources=sample,
        _client_status=lambda: {"16248": "absent", "35496": "reused"} if clients is None else clients,
        CLIENTS={16248: "born", 35496: "born"},
        _queue_counts=lambda: queue if queue is not None else readiness.EMPTY_QUEUE,
    )


def proof(**changes):
    row = {
        "current_pid": 100,
        "tcp_connect": "refused",
        "tcp_connect_error": "ConnectionRefusedError(10061)",
        "tcp_connect_errno": 10061,
        "listener_inventory_complete": True,
        "listener_8188": False,
        "listeners_8188": [],
        "process_inventory_complete": True,
        "allowed_process_family": [{"pid": 100, "ppid": 0, "create_time": 1.0, "relation": "self"}],
        "python_processes": [{"pid": 100, "ppid": 0, "name": "python.exe", "create_time": 1.0}],
        "foreign_python": [],
    }
    row.update(changes)
    return row


@pytest.mark.parametrize("phase", ["readiness", "monitor"])
def test_empty_queue_requires_actual_empty_clients_and_valid_gpu(phase):
    assert readiness._sample(source(), phase)["resource_mode"] == "queue_empty"
    assert not readiness._sample(source(queue={"running": 1, "pending": 0}), phase)["resource_clear"]
    assert not readiness._sample(source(clients={"16248": "unknown:AccessDenied", "35496": "reused"}), phase)["resource_clear"]
    assert not readiness._sample(source(gpu="bad"), phase)["resource_clear"]


@pytest.mark.parametrize(
    "change",
    [
        {"tcp_connect": "timeout"},
        {"tcp_connect": "error"},
        {"listener_8188": True, "listeners_8188": [{"pid": 9, "port": 8188}]},
        {"listener_inventory_complete": False, "port_error": "AccessDenied"},
        {"process_inventory_complete": False, "process_error": "AccessDenied"},
        {"foreign_python": [{"pid": 200, "name": "python.exe", "create_time": 2.0}]},
    ],
)
def test_stopped_service_fail_closed_for_incomplete_or_conflicting_proof(monkeypatch, change):
    monkeypatch.setattr(readiness, "_probe_service", lambda: proof(**change))
    row = readiness._sample(source(queue_error="URLError(ConnectionRefusedError(10061))"), "readiness")
    assert row["resource_mode"] == "blocked"
    assert row["queue_error"]
    assert not row["ready"]


def test_stopped_service_ready_only_when_gpu_idle(monkeypatch):
    monkeypatch.setattr(readiness, "_probe_service", proof)
    e182 = source(queue_error="raw refused")
    ready = readiness._sample(e182, "readiness")
    assert ready["resource_mode"] == "service_stopped"
    assert ready["ready"] and ready["queue_error"] == "raw refused"
    busy_gpu = "2026/09/24 12:00:00.000, 0, 80, 12000, 16384, 180, 2200, 4000"
    assert not readiness._sample(source(queue_error="raw refused", gpu=busy_gpu), "readiness")["ready"]
    running = readiness._sample(source(queue_error="raw refused", gpu=busy_gpu), "monitor")
    assert running["resource_mode"] == "service_stopped"
    assert readiness.resource_clear(running)
    assert not readiness.resource_clear(ready)


def test_connected_after_error_blocks_until_fresh_queue_sample(monkeypatch):
    monkeypatch.setattr(readiness, "_probe_service", lambda: proof(tcp_connect="connected"))
    busy = readiness._sample(source(queue={"running": 1, "pending": 0}, queue_error="initial failure"), "monitor")
    assert busy["queue_error"] == "initial failure"
    assert not readiness.resource_clear(busy)
    assert not readiness.resource_clear(readiness._sample(source(queue_error="initial failure"), "monitor"))
    empty = readiness._sample(source(), "monitor")
    assert empty["resource_mode"] == "queue_empty"
    assert "queue_error" not in empty


@pytest.mark.parametrize(
    "clients",
    [{}, {"16248": "absent"}, {"16248": "absent", "35496": "reused", "999": "absent"}, {"16248": "absent", "35496": "unknown:AccessDenied"}],
)
def test_exact_client_key_set_required(clients):
    assert not readiness._sample(source(clients=clients), "readiness")["ready"]


def test_readiness_two_samples_and_monitor_errors_retained(tmp_path, monkeypatch):
    monkeypatch.setattr(readiness, "_probe_service", proof)
    monkeypatch.setattr(readiness.time, "sleep", lambda seconds: sleep_times.append(seconds))
    sleep_times = []
    e182 = source(queue_error="raw refused")
    rows = readiness.readiness(e182)
    assert sleep_times == [2]
    assert len(rows) == 2 and all(row["ready"] for row in rows)
    stop = threading.Event()
    stop.set()
    errors = []
    readiness.monitor(tmp_path / "monitor.jsonl", stop, errors, e182)
    assert not errors
    assert not (tmp_path / "monitor.jsonl").read_text(encoding="utf-8")
    stop.clear()
    stop.wait = lambda seconds: True
    readiness.monitor(tmp_path / "second.jsonl", stop, errors, e182)
    monitored = json.loads((tmp_path / "second.jsonl").read_text(encoding="utf-8"))
    assert monitored["queue_error"] == "raw refused"
    assert readiness.resource_clear(monitored)
    readiness.monitor(tmp_path / "second.jsonl", stop, errors, e182)
    assert errors and "FileExistsError" in errors[0]


class FakeProcess:
    instances: ClassVar[dict[int, FakeProcess]] = {}

    def __init__(self, pid, name, ppid, born):
        self.pid, self._name, self._ppid, self._born = pid, name, ppid, born
        self.instances[pid] = self

    def name(self):
        return self._name

    def ppid(self):
        return self._ppid

    def create_time(self):
        return self._born

    def children(self, recursive):
        assert recursive is True
        descendants = {self.pid}
        while True:
            before = len(descendants)
            descendants.update(p.pid for p in self.instances.values() if p._ppid in descendants)
            if len(descendants) == before:
                return [self.instances[pid] for pid in descendants - {self.pid}]


@pytest.fixture(autouse=True)
def clear_fake_process_registry():
    FakeProcess.instances = {}


def test_probe_accepts_exact_process_family_and_refused_socket(monkeypatch):
    current = os.getpid()
    processes = {
        1: FakeProcess(1, "powershell.exe", 0, 1.0),
        current: FakeProcess(current, "python.exe", 1, 2.0),
        900: FakeProcess(900, "pypy.exe", current, 3.0),
    }
    monkeypatch.setattr(psutil, "pids", lambda: list(processes))
    monkeypatch.setattr(psutil, "Process", lambda pid: processes[pid])
    monkeypatch.setattr(psutil, "net_connections", lambda kind: [])

    def refused(*args, **kwargs):
        raise ConnectionRefusedError(errno.ECONNREFUSED, "refused")

    monkeypatch.setattr(socket, "create_connection", refused)
    result = readiness._probe_service()
    assert result["tcp_connect"] == "refused"
    assert result["listener_inventory_complete"]
    assert result["process_inventory_complete"]
    assert result["foreign_python"] == []
    assert {item["relation"] for item in result["allowed_process_family"]} == {"ancestor", "self", "descendant"}
    assert all("ppid" in item for item in result["allowed_process_family"] + result["python_processes"])


def test_probe_rejects_sibling_and_cousin_python(monkeypatch):
    current = os.getpid()
    processes = {
        1: FakeProcess(1, "explorer.exe", 0, 1.0),
        10: FakeProcess(10, "powershell.exe", 1, 2.0),
        current: FakeProcess(current, "python.exe", 10, 3.0),
        900: FakeProcess(900, "python.exe", 10, 4.0),
        901: FakeProcess(901, "pypy.exe", 1, 5.0),
        902: FakeProcess(902, "python.exe", current, 6.0),
    }
    monkeypatch.setattr(psutil, "pids", lambda: list(processes))
    monkeypatch.setattr(psutil, "Process", lambda pid: processes[pid])
    monkeypatch.setattr(psutil, "net_connections", lambda kind: [])
    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionRefusedError(errno.ECONNREFUSED, "refused")))
    result = readiness._probe_service()
    assert result["process_inventory_complete"]
    assert {item["pid"] for item in result["foreign_python"]} == {900, 901}
    assert {item["pid"] for item in result["allowed_process_family"]} == {1, 10, current, 902}


@pytest.mark.parametrize(
    "relative_parent,ancestor_birth,ancestor_ppid",
    [(True, 4.0, 0), (True, 1.0, -1), (False, 1.0, 0)],
)
def test_probe_invalid_ancestry_fails_closed(monkeypatch, relative_parent, ancestor_birth, ancestor_ppid):
    current = os.getpid()
    parent = 10 if relative_parent else 999
    processes = {
        current: FakeProcess(current, "python.exe", parent, 3.0),
        10: FakeProcess(10, "powershell.exe", current if ancestor_ppid == -1 else ancestor_ppid, ancestor_birth),
    }
    monkeypatch.setattr(psutil, "pids", lambda: list(processes))
    monkeypatch.setattr(psutil, "Process", lambda pid: processes[pid])
    monkeypatch.setattr(psutil, "pid_exists", lambda pid: True)
    monkeypatch.setattr(psutil, "net_connections", lambda kind: [])
    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionRefusedError(errno.ECONNREFUSED, "refused")))
    result = readiness._probe_service()
    assert not result["process_inventory_complete"]
    assert "process_error" in result


def test_probe_verified_absent_ancestor_boundary(monkeypatch):
    current = os.getpid()
    processes = {current: FakeProcess(current, "python.exe", 999, 3.0)}
    monkeypatch.setattr(psutil, "pids", lambda: list(processes))
    monkeypatch.setattr(psutil, "Process", lambda pid: processes[pid])
    monkeypatch.setattr(psutil, "pid_exists", lambda pid: False)
    monkeypatch.setattr(psutil, "net_connections", lambda kind: [])
    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionRefusedError(errno.ECONNREFUSED, "refused")))
    result = readiness._probe_service()
    assert result["process_inventory_complete"]
    assert result["ancestor_terminal"] == {"reason": "absent_parent", "pid": 999, "child_pid": current, "verified_absent": True}
    assert {item["pid"] for item in result["allowed_process_family"]} == {current}


def test_probe_access_denied_and_foreign_python(monkeypatch):
    current = os.getpid()
    processes = {
        current: FakeProcess(current, "python.exe", 0, 1.0),
        999: FakeProcess(999, "pythonw.exe", 0, 2.0),
    }
    monkeypatch.setattr(psutil, "pids", lambda: list(processes))
    monkeypatch.setattr(psutil, "Process", lambda pid: processes[pid])
    monkeypatch.setattr(psutil, "net_connections", lambda kind: [])
    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError()))
    result = readiness._probe_service()
    assert result["tcp_connect"] == "timeout"
    assert result["foreign_python"][0]["pid"] == 999
    monkeypatch.setattr(psutil, "Process", lambda pid: (_ for _ in ()).throw(psutil.AccessDenied(pid)))
    result = readiness._probe_service()
    assert not result["process_inventory_complete"]
    assert "AccessDenied" in result["process_error"]


def test_probe_listener_and_port_access_denied(monkeypatch):
    current = os.getpid()
    process = FakeProcess(current, "python.exe", 0, 1.0)
    monkeypatch.setattr(psutil, "pids", lambda: [current])
    monkeypatch.setattr(psutil, "Process", lambda pid: process)
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ConnectionRefusedError(errno.ECONNREFUSED, "refused")
        ),
    )
    listener = SimpleNamespace(
        status=psutil.CONN_LISTEN,
        laddr=SimpleNamespace(ip="127.0.0.1", port=8188),
        pid=5,
    )
    monkeypatch.setattr(psutil, "net_connections", lambda kind: [listener])
    result = readiness._probe_service()
    assert result["listener_inventory_complete"]
    assert result["listener_8188"]
    assert result["listeners_8188"][0]["pid"] == 5
    monkeypatch.setattr(
        psutil,
        "net_connections",
        lambda kind: (_ for _ in ()).throw(psutil.AccessDenied(5)),
    )
    result = readiness._probe_service()
    assert not result["listener_inventory_complete"]
    assert "AccessDenied" in result["port_error"]


def test_gpu_all_indices_valid_and_idle_threshold_applies_to_every_gpu(monkeypatch):
    monkeypatch.setattr(readiness, "_probe_service", proof)
    second = "2026/09/24 12:00:00.000, 1, 6, 900, 16384, 12, 300, 400"
    e182 = source(queue_error="raw refused")
    e182._sample_resources = lambda: {"queue_error": "raw refused", "gpu_csv": [GPU, second]}
    row = readiness._sample(e182, "readiness")
    assert not row["ready"]
    assert len(row["gpu_telemetry"]) == 2
    assert readiness._sample(e182, "monitor")["resource_mode"] == "service_stopped"
    e182._sample_resources = lambda: {"queue_error": "raw refused", "gpu_csv": [GPU, GPU]}
    row = readiness._sample(e182, "monitor")
    assert row["resource_mode"] == "blocked"
    assert "duplicate" in row["gpu_telemetry_error"]
