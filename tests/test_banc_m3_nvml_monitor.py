"""Injected CPU reader only: no device/DLL/context access."""

import ctypes
import subprocess
import threading

import pytest

from tools import banc_m3_nvml_monitor as m

UUID = "GPU-12345678-1234-1234-1234-123456789abc"


class Reader:
    def __init__(
        self,
        values=(m.GIB, m.GIB + 8 * m.GIB - 2 * m.MIB),
        *,
        count=1,
        mig=False,
        fail=None,
    ):
        self.values, self.count, self.mig, self.fail = values, count, mig, fail
        self.initialized = False
        self.calls, self.shutdowns = 0, 0

    def init(self):
        if self.fail == "init":
            raise RuntimeError("init failure")
        self.initialized = True
        return {
            "physical_device_count": self.count,
            "physical_index": 0,
            "mig_enabled": self.mig,
            "uuid": UUID,
            "driver": "injected",
            "handle_identity": 11,
            "dll_path": "injected-only",
        }

    def read(self):
        if self.fail == "read":
            raise RuntimeError("read failure")
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return {
            "total": 16 * m.GIB,
            "used": value,
            "free": 16 * m.GIB - value,
            "uuid": UUID,
            "handle_identity": 11,
        }

    def shutdown(self):
        self.shutdowns += 1
        self.initialized = False
        if self.fail == "shutdown":
            raise RuntimeError("shutdown failure")


def test_running_bytes_baseline_gate_close_and_no_subprocess(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("subprocess/DLL/identity capture forbidden")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(ctypes, "CDLL", forbidden)
    identity = type("Identity", (), {"capture": forbidden})()
    reader = Reader()
    watch = m.GPUMemoryWatch(reader=reader, identity=identity, cuda_visible_devices="0")
    live = watch.start()
    assert live["owned_thread_alive"] and live["qualified"]
    assert live["context_included"] is True
    assert live["CUDA_context_creation_order_verified"] is False
    assert live["actual_neural_body_execution_verified"] is False
    assert (
        live["qualification_scope"]
        == "sampler lifecycle/device mapping/memory observations only"
    )
    watch._once()
    assert watch.start()["precontext_baseline_bytes"] == m.GIB
    closed = watch.close()
    assert closed["CUDA_context_creation_order_verified"] is False
    assert closed["actual_neural_body_execution_verified"] is False
    assert (
        closed["sampled_additional_peak_bytes"] + closed["quantization_allowance_bytes"]
        == 8 * m.GIB
    )
    assert closed["qualified"] and closed["owned_sampler_pids"] == []
    assert watch.close()["shutdown_count"] == reader.shutdowns == 1


@pytest.mark.parametrize(
    "visibility",
    ["", "1", "0,1", "GPU-1234", "MIG-123", "GPU-99999999-1234-1234-1234-123456789abc"],
)
def test_ambiguous_mapping_refused(visibility):
    reader = Reader()
    watch = m.GPUMemoryWatch(reader=reader, cuda_visible_devices=visibility)
    result = watch.start()
    assert not result["qualified"] and result["sampler_errors"]
    assert reader.calls == 0 and reader.shutdowns == 1


@pytest.mark.parametrize("visibility", ["0", UUID, None])
def test_supported_mapping(visibility, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    watch = m.GPUMemoryWatch(reader=Reader(), cuda_visible_devices=visibility)
    assert watch.start()["qualified"]
    assert watch.close()["qualified"]


@pytest.mark.parametrize(
    "reader",
    [
        Reader(count=2),
        Reader(mig=True),
        Reader(fail="init"),
        Reader(fail="read"),
        Reader(fail="shutdown"),
    ],
)
def test_sticky_failure_and_single_cleanup(reader):
    watch = m.GPUMemoryWatch(reader=reader, cuda_visible_devices="0")
    watch.start()
    result = watch.close()
    watch.start()
    watch.close()
    assert not result["qualified"] and result["sampler_errors"]
    assert reader.shutdowns == (0 if reader.fail == "init" else 1)


def test_no_shutdown_race_with_blocked_query():
    entered, release = threading.Event(), threading.Event()

    class Blocked(Reader):
        def read(self):
            if self.calls:
                entered.set()
                assert release.wait(timeout=1)
            return super().read()

    reader = Blocked()
    watch = m.GPUMemoryWatch(
        reader=reader, cuda_visible_devices="0", interval_s=0.001, join_timeout_s=0.01
    )
    watch.start()
    try:
        assert entered.wait(timeout=0.5)
        result = watch.close()
        assert (
            result["owned_thread_alive"]
            and not result["qualified"]
            and reader.shutdowns == 0
        )
    finally:
        release.set()
        watch._thread.join(timeout=0.5)
        final = watch.close()
    assert (
        not final["owned_thread_alive"]
        and reader.shutdowns == 1
        and not final["qualified"]
    )


def test_over_cap_and_wrong_units_stay_invalid():
    watch = m.GPUMemoryWatch(
        reader=Reader((m.GIB, m.GIB + 8 * m.GIB)), cuda_visible_devices="0"
    )
    watch.start()
    watch._once()
    assert not watch.close()["memory_limit_passed"]
    reader = Reader((True,))
    failed = m.GPUMemoryWatch(reader=reader, cuda_visible_devices="0").start()
    assert not failed["qualified"] and reader.shutdowns == 1


def test_memory_abi_uint64_layout():
    assert ctypes.sizeof(m.MemoryInfo) == 24
    assert [key for key, dtype in m.MemoryInfo._fields_] == ["total", "free", "used"]
    assert all(dtype is ctypes.c_uint64 for key, dtype in m.MemoryInfo._fields_)


def test_checked_public_ctypes_abi_injected_dll(monkeypatch):
    calls = []

    class Function:
        def __init__(self, name):
            self.name = name

        def __call__(self, *args):
            calls.append(self.name)
            if self.name == "nvmlDeviceGetCount_v2":
                args[0]._obj.value = 1
            elif self.name == "nvmlDeviceGetHandleByIndex_v2":
                args[1]._obj.value = 11
            elif self.name == "nvmlDeviceGetUUID":
                args[1].value = UUID.encode()
            elif self.name == "nvmlSystemGetDriverVersion":
                args[0].value = b"injected-driver"
            elif self.name == "nvmlDeviceGetMigMode":
                return 3
            elif self.name == "nvmlDeviceGetMemoryInfo":
                value = args[1]._obj
                value.total, value.free, value.used = 16 * m.GIB, 15 * m.GIB, m.GIB
            return 0

    names = (
        "nvmlInit_v2",
        "nvmlShutdown",
        "nvmlDeviceGetCount_v2",
        "nvmlDeviceGetHandleByIndex_v2",
        "nvmlDeviceGetUUID",
        "nvmlSystemGetDriverVersion",
        "nvmlDeviceGetMemoryInfo",
        "nvmlDeviceGetMigMode",
    )
    dll = type("InjectedDLL", (), {})()
    for name in names:
        setattr(dll, name, Function(name))

    def fake_loader(path):
        assert path == "C:/Windows/System32/nvml.dll"
        return dll

    monkeypatch.setattr(ctypes, "CDLL", fake_loader)
    watch = m.GPUMemoryWatch(reader=m.NVMLReader(), cuda_visible_devices="0")
    assert watch.start()["precontext_baseline_bytes"] == m.GIB
    assert watch.close()["qualified"]
    assert calls.count("nvmlInit_v2") == calls.count("nvmlShutdown") == 1
    assert all(
        getattr(dll, name).restype is ctypes.c_int
        and isinstance(getattr(dll, name).argtypes, list)
        for name in names
    )
