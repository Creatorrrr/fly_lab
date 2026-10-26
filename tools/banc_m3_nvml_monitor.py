"""Process-free, single physical non-MIG GPU memory monitoring in bytes."""

import ctypes
import math
import os
import re
import threading

DLL_PATH = "C:/Windows/System32/nvml.dll"
MIB, GIB = 1048576, 1073741824
UUID = re.compile(
    r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z"
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


class MemoryInfo(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_uint64),
        ("free", ctypes.c_uint64),
        ("used", ctypes.c_uint64),
    ]


class NVMLReader:
    """Checked public C ABI; the DLL is loaded only by init()."""

    def __init__(self):
        self.initialized = False
        self.dll = None
        self.handle = ctypes.c_void_p()
        self.metadata = None

    def _call(self, name, *args):
        status = getattr(self.dll, name)(*args)
        require(status == 0, name + " status=" + str(status))

    def init(self):
        require(not self.initialized, "single NVML init")
        self.dll = ctypes.CDLL(DLL_PATH)
        declarations = {
            "nvmlInit_v2": [],
            "nvmlShutdown": [],
            "nvmlDeviceGetCount_v2": [ctypes.POINTER(ctypes.c_uint)],
            "nvmlDeviceGetHandleByIndex_v2": [
                ctypes.c_uint,
                ctypes.POINTER(ctypes.c_void_p),
            ],
            "nvmlDeviceGetUUID": [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_uint,
            ],
            "nvmlSystemGetDriverVersion": [
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_uint,
            ],
            "nvmlDeviceGetMemoryInfo": [ctypes.c_void_p, ctypes.POINTER(MemoryInfo)],
            "nvmlDeviceGetMigMode": [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint),
                ctypes.POINTER(ctypes.c_uint),
            ],
        }
        for name, arguments in declarations.items():
            function = getattr(self.dll, name)
            function.argtypes, function.restype = arguments, ctypes.c_int
        self._call("nvmlInit_v2")
        self.initialized = True
        count = ctypes.c_uint()
        self._call("nvmlDeviceGetCount_v2", ctypes.byref(count))
        require(count.value == 1, "single physical GPU required")
        self._call("nvmlDeviceGetHandleByIndex_v2", 0, ctypes.byref(self.handle))
        require(bool(self.handle.value), "non-null physical handle")
        uuid, driver = ctypes.create_string_buffer(96), ctypes.create_string_buffer(96)
        self._call("nvmlDeviceGetUUID", self.handle, uuid, len(uuid))
        self._call("nvmlSystemGetDriverVersion", driver, len(driver))
        current, pending = ctypes.c_uint(), ctypes.c_uint()
        status = self.dll.nvmlDeviceGetMigMode(
            self.handle, ctypes.byref(current), ctypes.byref(pending)
        )
        # Public NOT_SUPPORTED means this physical device does not support MIG.
        require(status in (0, 3), "checked MIG query status=" + str(status))
        require(
            status == 3 or (current.value == pending.value == 0),
            "non-MIG whole-device required",
        )
        self.metadata = {
            "dll_path": DLL_PATH,
            "physical_device_count": count.value,
            "physical_index": 0,
            "uuid": uuid.value.decode("ascii"),
            "driver": driver.value.decode("ascii"),
            "mig_enabled": False,
            "mig_query": "not_supported" if status == 3 else "disabled",
            "handle_identity": self.handle.value,
        }
        return dict(self.metadata)

    def read(self):
        require(
            self.initialized and self.metadata is not None, "NVML initialized handle"
        )
        value = MemoryInfo()
        self._call("nvmlDeviceGetMemoryInfo", self.handle, ctypes.byref(value))
        return {
            "uuid": self.metadata["uuid"],
            "handle_identity": self.handle.value,
            "total": int(value.total),
            "free": int(value.free),
            "used": int(value.used),
        }

    def shutdown(self):
        require(self.initialized, "shutdown matches successful init")
        # Never retry even if NVML shutdown itself reports an error.
        self.initialized = False
        self._call("nvmlShutdown")


class GPUMemoryWatch:
    """Injected reader has init()->metadata,read()->byte sample,shutdown()."""

    def __init__(
        self,
        *,
        index=0,
        interval_s=0.1,
        identity=None,
        reader=None,
        join_timeout_s=1,
        cuda_visible_devices=None,
    ):
        require(type(index) is int and index == 0, "physical index0 only")
        for value, maximum in ((interval_s, 1), (join_timeout_s, 2)):
            require(
                type(value) in (int, float)
                and math.isfinite(value)
                and 0 < value <= maximum,
                "bounded monitor interval/join",
            )
        self.interval_s, self.join_timeout_s = interval_s, join_timeout_s
        self.identity = identity  # Compatibility only; never captures subprocesses.
        self.reader = reader if reader is not None else NVMLReader()
        self.visibility = (
            os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible_devices is None
            else cuda_visible_devices
        )
        self._lifecycle = threading.RLock()
        self._lock = threading.Lock()
        self._stop, self._thread = threading.Event(), None
        self._state, self._errors = "NEW", []
        self._initialized, self._shutdown_count = False, 0
        self._metadata, self._baseline, self._peak, self._latest = (
            None,
            None,
            None,
            None,
        )
        self._count = 0

    def _error(self, error):
        with self._lock:
            self._errors.append(
                {"type": type(error).__name__, "message": str(error)[:512]}
            )
        self._stop.set()

    def _once(self):
        value = self.reader.read()
        require(
            value["uuid"] == self._metadata["uuid"]
            and value["handle_identity"] == self._metadata["handle_identity"],
            "same fresh UUID/handle",
        )
        require(
            all(
                type(value[key]) is int and 0 <= value[key] < 2**64
                for key in ("total", "used", "free")
            )
            and value["total"] > 0
            and value["used"] <= value["total"]
            and value["free"] <= value["total"],
            "uint64 memory bytes",
        )
        with self._lock:
            require(
                self._latest is None or value["total"] == self._latest["total"],
                "same total memory",
            )
            self._baseline = value["used"] if self._baseline is None else self._baseline
            self._peak = max(value["used"], self._peak if self._peak is not None else 0)
            self._latest = dict(value)
            self._count += 1

    def start(self):
        with self._lifecycle:
            if self._state != "NEW":
                return self.snapshot()
            self._state = "STARTING"
            try:
                self._metadata = self.reader.init()
                self._initialized = True
                metadata = self._metadata
                require(
                    metadata["physical_device_count"] == 1
                    and type(metadata["physical_device_count"]) is int
                    and metadata["physical_index"] == 0
                    and metadata["mig_enabled"] is False
                    and isinstance(metadata["uuid"], str)
                    and UUID.fullmatch(metadata["uuid"])
                    and metadata.get("driver")
                    and metadata.get("handle_identity") is not None,
                    "single non-MIG device identity",
                )
                require(
                    self.visibility in (None, "0", metadata["uuid"]),
                    "CUDA ordinal mapping ambiguous",
                )
                self._once()

                def loop():
                    while not self._stop.wait(self.interval_s):
                        try:
                            self._once()
                        except Exception as error:  # noqa: BLE001 -- Sticky monitor failure.
                            self._error(error)

                self._thread = threading.Thread(
                    target=loop, name="E347-NVML-memory", daemon=True
                )
                self._thread.start()
                self._state = "RUNNING"
            except Exception as error:  # noqa: BLE001 -- Preserve failed start and cleanup.
                self._initialized = (
                    self._initialized
                    or getattr(self.reader, "initialized", False) is True
                )
                self._error(error)
                self._state = "FAILED"
                self.close()
            return self.snapshot()

    def snapshot(self):
        with self._lock:
            alive = self._thread.is_alive() if self._thread else False
            delta = (
                None if self._baseline is None else max(0, self._peak - self._baseline)
            )
            memory_passed = delta is not None and delta + 2 * MIB <= 8 * GIB
            good_state = (self._state == "RUNNING" and alive) or (
                self._state == "CLOSED" and not alive and self._shutdown_count == 1
            )
            return {
                "precontext_baseline_bytes": self._baseline,
                "sampled_total_peak_bytes": self._peak,
                "sampled_additional_peak_bytes": delta,
                "quantization_allowance_bytes": 2 * MIB,
                "sample_count": self._count,
                "sampler_errors": [dict(v) for v in self._errors],
                "owned_sampler_pids": [],
                "owned_thread_alive": alive,
                "context_included": True,
                "context_included_scope": "measurement policy includes any context memory present in device samples",
                "qualification_scope": "sampler lifecycle/device mapping/memory observations only",
                "CUDA_context_creation_order_verified": False,
                "actual_neural_body_execution_verified": False,
                "unsampled_or_external_work_exclusion_proven": False,
                "state": self._state,
                "device_identity": None
                if self._metadata is None
                else dict(self._metadata),
                "latest_memory_bytes": None
                if self._latest is None
                else dict(self._latest),
                "CUDA_VISIBLE_DEVICES": self.visibility,
                "mapping": "single physical non-MIG index0",
                "shutdown_count": self._shutdown_count,
                "qualified": bool(
                    good_state and self._count and not self._errors and memory_passed
                ),
                "memory_limit_passed": memory_passed,
                "subprocess_calls": 0,
            }

    def close(self):
        with self._lifecycle:
            self._stop.set()
            if self._thread and self._thread.ident is not None:
                self._thread.join(timeout=self.join_timeout_s)
                if self._thread.is_alive():
                    if not any(
                        v["message"] == "owned NVML query thread still alive"
                        for v in self._errors
                    ):
                        self._error(TimeoutError("owned NVML query thread still alive"))
                    self._state = "FAILED"
                    return self.snapshot()
            if self._initialized and not self._shutdown_count:
                self._shutdown_count = 1
                self._initialized = False
                try:
                    self.reader.shutdown()
                except Exception as error:  # noqa: BLE001 -- Failed shutdown stays sticky.
                    self._error(error)
            self._state = "CLOSED"
            return self.snapshot()
