"""Finite stdlib direct supervisor. ROOT owns admission and actual once-cost commit."""
# ruff: noqa: BLE001 -- record every primary, stream and cleanup failure before exit.
from __future__ import annotations

import time

ENTRY = time.perf_counter()
import argparse
import ctypes as ct
import hashlib
import json
import math
import os
import secrets
import subprocess
import sys
import threading
import traceback
from ctypes import wintypes as wt
from pathlib import Path

from tools.banc_m3_conserved_resource import SharedLock

ROOT = Path(__file__).resolve().parents[1]
BASE = "verification/banc-natural-gait-m3-20260919"
SHARED_LOCK = BASE + "/parent-e358-original-active.lock"
SCHEMA = "flylab.m3.direct-science"
MAX_PROCESSES = 32
TRACKING_ROLES = ("tracking_neutral", "tracking_candidate", "tracking_diagnosis")
TRACKING_STREAM_CAP = 1 << 20
TRACKING_METADATA_CAP = 64 << 10


def require(ok, message):
    if not ok:
        raise ValueError(message)


def kernel():
    require(os.name == "nt", "Windows exact process identity required")
    k = ct.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "OpenProcess": ([wt.DWORD, wt.BOOL, wt.DWORD], wt.HANDLE),
        "GetCurrentProcess": ([], wt.HANDLE),
        "GetProcessTimes": ([wt.HANDLE, ct.POINTER(wt.FILETIME), ct.POINTER(wt.FILETIME), ct.POINTER(wt.FILETIME), ct.POINTER(wt.FILETIME)], wt.BOOL),
        "GetExitCodeProcess": ([wt.HANDLE, ct.POINTER(wt.DWORD)], wt.BOOL),
        "CloseHandle": ([wt.HANDLE], wt.BOOL),
        "CreateJobObjectW": ([ct.c_void_p, wt.LPCWSTR], wt.HANDLE),
        "OpenJobObjectW": ([wt.DWORD, wt.BOOL, wt.LPCWSTR], wt.HANDLE),
        "AssignProcessToJobObject": ([wt.HANDLE, wt.HANDLE], wt.BOOL),
        "IsProcessInJob": ([wt.HANDLE, wt.HANDLE, ct.POINTER(wt.BOOL)], wt.BOOL),
        "SetInformationJobObject": ([wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD], wt.BOOL),
        "QueryInformationJobObject": ([wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD, ct.POINTER(wt.DWORD)], wt.BOOL),
        "TerminateJobObject": ([wt.HANDLE, wt.UINT], wt.BOOL),
        "CreatePipe": ([ct.POINTER(wt.HANDLE), ct.POINTER(wt.HANDLE), ct.c_void_p, wt.DWORD], wt.BOOL),
        "SetHandleInformation": ([wt.HANDLE, wt.DWORD, wt.DWORD], wt.BOOL),
        "ReadFile": ([wt.HANDLE, ct.c_void_p, wt.DWORD, ct.POINTER(wt.DWORD), ct.c_void_p], wt.BOOL),
        "CreateFileW": ([wt.LPCWSTR, wt.DWORD, wt.DWORD, ct.c_void_p, wt.DWORD, wt.DWORD, wt.HANDLE], wt.HANDLE),
        "CreateProcessW": ([wt.LPCWSTR, wt.LPWSTR, ct.c_void_p, ct.c_void_p, wt.BOOL, wt.DWORD, ct.c_void_p, wt.LPCWSTR, ct.c_void_p, ct.c_void_p], wt.BOOL),
        "ResumeThread": ([wt.HANDLE], wt.DWORD),
        "TerminateProcess": ([wt.HANDLE, wt.UINT], wt.BOOL),
    }
    for name, (args, returns) in signatures.items():
        getattr(k, name).argtypes = args
        getattr(k, name).restype = returns
    return k


def checked(ok):
    if not ok:
        raise ct.WinError(ct.get_last_error())


def handle_creation_ticks(k, handle):
    values = [wt.FILETIME() for _ in range(4)]
    checked(k.GetProcessTimes(handle, *(ct.byref(v) for v in values)))
    return (values[0].dwHighDateTime << 32) | values[0].dwLowDateTime


def creation_ticks(pid):
    require(type(pid) is int and pid > 0, "exact integer PID")
    k = kernel()
    h = k.OpenProcess(0x1000, False, pid)
    checked(h)
    try:
        actual = handle_creation_ticks(k, h)
        exit_code = wt.DWORD()
        checked(k.GetExitCodeProcess(h, ct.byref(exit_code)))
        require(exit_code.value == 259, "live exact identity")
        return actual
    finally:
        checked(k.CloseHandle(h))


def in_named_job(name):
    require(type(name) is str and name.startswith("Local\\flylab-startup-ray-"), "exact owned Job name")
    k = kernel()
    h = k.OpenJobObjectW(0x0004, False, name)
    checked(h)
    try:
        actual = wt.BOOL()
        checked(k.IsProcessInJob(k.GetCurrentProcess(), h, ct.byref(actual)))
        return bool(actual.value)
    finally:
        checked(k.CloseHandle(h))


class Security(ct.Structure):
    _fields_ = [("length", wt.DWORD), ("descriptor", ct.c_void_p), ("inherit", wt.BOOL)]


class Startup(ct.Structure):
    _fields_ = [("cb", wt.DWORD), ("reserved", wt.LPWSTR), ("desktop", wt.LPWSTR), ("title", wt.LPWSTR),
                ("x", wt.DWORD), ("y", wt.DWORD), ("xsize", wt.DWORD), ("ysize", wt.DWORD),
                ("xchars", wt.DWORD), ("ychars", wt.DWORD), ("fill", wt.DWORD), ("flags", wt.DWORD),
                ("show", wt.WORD), ("reserved_size", wt.WORD), ("reserved_bytes", ct.c_void_p),
                ("stdin", wt.HANDLE), ("stdout", wt.HANDLE), ("stderr", wt.HANDLE)]


class ProcessInfo(ct.Structure):
    _fields_ = [("process", wt.HANDLE), ("thread", wt.HANDLE), ("pid", wt.DWORD), ("tid", wt.DWORD)]


class BasicLimits(ct.Structure):
    _fields_ = [("process_user", ct.c_int64), ("job_user", ct.c_int64), ("flags", wt.DWORD),
                ("min_working", ct.c_size_t), ("max_working", ct.c_size_t), ("active_limit", wt.DWORD),
                ("affinity", ct.c_size_t), ("priority", wt.DWORD), ("scheduling", wt.DWORD)]


class IoCounters(ct.Structure):
    _fields_ = [(name, ct.c_uint64) for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]


class ExtendedLimits(ct.Structure):
    _fields_ = [("basic", BasicLimits), ("io", IoCounters), ("process_memory", ct.c_size_t),
                ("job_memory", ct.c_size_t), ("peak_process", ct.c_size_t), ("peak_job", ct.c_size_t)]


class Accounting(ct.Structure):
    _fields_ = [(name, ct.c_int64) for name in ("user", "kernel", "period_user", "period_kernel")] + \
               [(name, wt.DWORD) for name in ("page_faults", "total", "active", "terminated")]


class ProcessList(ct.Structure):
    _fields_ = [("assigned", wt.DWORD), ("count", wt.DWORD), ("pids", ct.c_size_t * MAX_PROCESSES)]


class OwnedJob:
    """Assign while suspended; no descendant can execute outside this Job."""
    def __init__(self, name, host_bytes):
        require(type(host_bytes) is int and host_bytes > 0, "exact Job host limit")
        self.k = kernel()
        ct.set_last_error(0)
        self.handle = self.k.CreateJobObjectW(None, name)
        checked(self.handle)
        require(ct.get_last_error() != 183, "fresh uniquely named Job")
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000 | 0x0008 | 0x0200  # kill-on-close, process-count, Job memory
        limits.basic.active_limit = MAX_PROCESSES
        limits.job_memory = host_bytes
        checked(self.k.SetInformationJobObject(self.handle, 9, ct.byref(limits), ct.sizeof(limits)))
        self.process = self.thread = None
        self.reads = []
        self.writes = []
        self.pid = None
        self.identities = {}
        self.observed_handles = []

    def spawn(self, argv, root, environment):
        security = Security(ct.sizeof(Security), None, True)
        for _ in range(2):
            read, write = wt.HANDLE(), wt.HANDLE()
            checked(self.k.CreatePipe(ct.byref(read), ct.byref(write), ct.byref(security), 0))
            checked(self.k.SetHandleInformation(read, 1, 0))
            self.reads.append(read)
            self.writes.append(write)
        stdin = self.k.CreateFileW("NUL", 0x80000000, 3, ct.byref(security), 3, 0x80, None)
        checked(stdin != ct.c_void_p(-1).value)
        startup = Startup()
        startup.cb = ct.sizeof(startup)
        startup.flags = 0x100
        startup.stdin, startup.stdout, startup.stderr = stdin, self.writes[0], self.writes[1]
        info = ProcessInfo()
        command = ct.create_unicode_buffer(subprocess.list2cmdline(argv))
        env = ct.create_unicode_buffer("\0".join(k + "=" + v for k, v in sorted(environment.items())) + "\0\0")
        try:
            checked(self.k.CreateProcessW(argv[0], command, None, None, True,
                                         0x08000000 | 0x00000004 | 0x00000400, env, str(root),
                                         ct.byref(startup), ct.byref(info)))
            self.process, self.thread, self.pid = info.process, info.thread, int(info.pid)
            checked(self.k.AssignProcessToJobObject(self.handle, self.process))
            self.identities[str(self.pid)] = creation_ticks(self.pid)
        except Exception:
            if self.process:
                checked(self.k.TerminateProcess(self.process, 124))
            raise
        finally:
            checked(self.k.CloseHandle(stdin))
            for handle in self.writes:
                checked(self.k.CloseHandle(handle))
            self.writes = []

    def resume(self):
        require(self.thread is not None, "known suspended primary worker")
        checked(self.k.ResumeThread(self.thread) != 0xFFFFFFFF)
        checked(self.k.CloseHandle(self.thread))
        self.thread = None

    def active(self):
        info = Accounting()
        checked(self.k.QueryInformationJobObject(self.handle, 1, ct.byref(info), ct.sizeof(info), None))
        return int(info.active)

    def observe(self):
        pids = ProcessList()
        checked(self.k.QueryInformationJobObject(self.handle, 3, ct.byref(pids), ct.sizeof(pids), None))
        require(pids.count <= MAX_PROCESSES, "bounded actual Job roster")
        for pid in pids.pids[:pids.count]:
            key = str(int(pid))
            if key in self.identities:
                # The original process handle stays held, so its PID cannot be reused.
                continue
            held = self.k.OpenProcess(0x1000, False, int(pid))
            if not held:
                # A disappearing member cannot be declared known while the Job is active.
                require(self.active() == 0, "Job member identity not known while owned Job active")
                continue
            self.observed_handles.append(held)
            membership = wt.BOOL()
            checked(self.k.IsProcessInJob(held, self.handle, ct.byref(membership)))
            require(membership.value, "exact observed process belongs to this actual Job")
            self.identities[key] = handle_creation_ticks(self.k, held)

    def exit_code(self):
        if self.process is None:
            return None
        code = wt.DWORD()
        checked(self.k.GetExitCodeProcess(self.process, ct.byref(code)))
        return None if code.value == 259 else int(code.value)

    def terminate(self):
        checked(self.k.TerminateJobObject(self.handle, 124))

    def close(self):
        for handle in (*self.reads, *self.observed_handles, self.thread, self.process, self.handle):
            if handle:
                checked(self.k.CloseHandle(handle))
        self.reads = []
        self.observed_handles = []
        self.thread = self.process = self.handle = None


def durable(p, value):
    raw = json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True).encode("utf8")
    with p.open("xb") as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    require(p.read_bytes() == raw, "durable publication readback")
    return {"path": str(p), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def transition(root, ledger_path, q, request_spec, token_spec, identity, state, *, lock):
    """One serialized writer while shared lock is held; costs/debits never changed here."""
    from tools import banc_m3_conserved_resource as resource
    from tools import verify_c_banc_m3_startup_ray as ray

    require(isinstance(lock, resource.SharedLock), "actual local SharedLock capability, never a foreign token")
    lock.authenticate(ray.path(SHARED_LOCK, root))
    ray.exact_value(lock.value["request"], request_spec, "this locally created owner request")
    ray.exact_value(lock.value["call_id"], q["call_id"], "this locally created owner call")
    ray.exact_value(lock.value["owner_pid"], os.getpid(), "this actual owner PID")
    ray.exact_value(creation_ticks(os.getpid()), lock.value["owner_creation_ticks"], "same owner process creation")
    ray.exact_value(ray.spec_for(SHARED_LOCK, root), token_spec, "actual locally owned token bytes")
    for key in ("owner_pid", "owner_creation_ticks"):
        ray.exact_value(identity[key], lock.value[key], "same local transition identity")
    p = ray.path(ledger_path, root)
    raw = p.read_bytes()
    require(len(raw) <= 524288, "bounded actual accounting ledger")
    ledger = ray.json_load(raw)
    caps = resource.current_caps(ledger, root, generated_budget_limit)
    from tools import banc_m3_checkpoint_runtime as checkpoint
    if caps not in (resource.NEW_CAPS, checkpoint.CAPS):
        generated_budget_limit(ledger, root)  # OLD_CAPS alone is not legacy transfer validation.
    original = q["mode"] == "original"
    prefix = "startup_ray_original_" if original else "direct_science_"
    call = q["call_id"]
    workers = ledger.setdefault(prefix + "workers", {})
    reservations = ledger[prefix + "reservations"]
    pending = ledger[prefix + "pending_actuals"]
    holds = ledger[prefix + "blocking_holds"]
    if state == "begin":
        require(not workers and not pending and not holds, "sole serial owner, no pending/unknown work")
        if original:
            ray.exact_value(ledger["direct_science_calls"][call], {"request": request_spec, "state": "admitted", "mode": "original", "reserved_s": q["limits"]["outer_s"]}, "real Root-admitted Original outer request")
            require(not reservations and ledger["original_jobs_running"] == 0 and ledger["original_reserved_s"] == 0, "no existing Original reservation")
            reservations[call] = {"request": q["science_request"], "grant": q["grant"], "reserved_s": q["limits"]["outer_s"]}
            ledger["original_reserved_s"] = q["limits"]["outer_s"]
            workers[call] = {"request": q["science_request"], "grant": q["grant"], "state": "begun",
                             "worker_max": 1, "reserved_s": q["limits"]["outer_s"], "lock_token": token_spec, **identity}
            ledger["original_jobs_running"] = 1
        else:
            require(ledger["direct_science_calls"][call]["state"] == "admitted", "fresh generated admission")
            workers[call] = {"request": request_spec, "state": "begun", "lock_token": token_spec, **identity}
            ray.exact_value(reservations, {call: {"request": request_spec, "reserved_s": q["limits"]["outer_s"]}}, "exact preadmitted generated reservation")
    elif state == "closed":
        require(call in workers and call in reservations and not pending and not holds, "same known returned call")
        workers.pop(call)
        reservations.pop(call)
        pending[call] = {"request": request_spec, "state": "actual_pending", "ownership_closed": True}
        if original:
            ledger["original_jobs_running"] = 0
            ledger["original_reserved_s"] = 0
        # ROOT alone changes direct_science_calls.state to charged after genuine outer return.
    else:
        require(state == "unknown" and call in workers, "retain exact unknown owner")
        holds[call] = {"request": request_spec, "state": "ownership_unknown", **identity}
    resource.atomic_ledger_publish(p, raw, ledger, lock,
                                  validate=lambda current: resource.current_caps(current, root, generated_budget_limit))


def tracking_pre_spawn_refusal(root, request_spec, q, error, *, entry):
    """Actual refusal in this supervisor before any Job/CreateProcess attempt.

    This path creates no lease, reservation, worker, numeric import, or array read.
    It is not an admission certificate and is never used for ordinary worker errors.
    """
    from tools import banc_m3_startup_tracking_admission as tracking
    from tools import verify_c_banc_m3_startup_ray as ray
    require(q["mode"] == "original" and q["role"] in TRACKING_ROLES, "new tracking refusal only")
    require(q["call_id"] == tracking.CALLS[TRACKING_ROLES.index(q["role"])], "fixed refused call")
    tracking.exact_value(q["grant"], tracking.GRANT, "refusal preserves actual admitted J grant")
    expected = tracking.limits_for(q["role"])
    for key in ("outer_s", "child_s", "cleanup_s", "host_bytes", "output_cap_bytes"):
        tracking.exact_value(q["limits"][key], expected[key], "actual refused role bound")
    tracking.exact_value(q["output"], tracking.BASE + "/parent-" + q["call_id"] + "/result-supervision", "fresh finite refused output")
    live = ray.json_load(ray.leaf(ray.spec_for(q["ledger_path"], root, cap=524288), root, cap=524288))
    tracking.exact_value(live["direct_science_calls"][q["call_id"]],
                         {"request": request_spec, "state": "admitted", "mode": "original", "reserved_s": expected["outer_s"]}, "actual current admitted refused request")
    require(not live["startup_ray_original_workers"] and not live["startup_ray_original_reservations"]
            and not live["startup_ray_original_pending_actuals"] and not live["startup_ray_original_blocking_holds"]
            and live["original_jobs_running"] == 0 and live["original_reserved_s"] == 0
            and not ray.path(SHARED_LOCK, root).exists(), "refusal before actual shared lease/Job")
    output = ray.path(q["output"], root)
    require(not output.exists(), "refused call cannot publish twice")
    output.mkdir(parents=True)
    streams = {}
    for label in ("stdout", "stderr"):
        p = output / (label + ".bin")
        with p.open("xb") as stream:
            stream.flush();os.fsync(stream.fileno())
        streams[label] = {"path": p.relative_to(root).as_posix(), "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    message = str(error)
    outer = {"schema": SCHEMA + ".outer.v1", "request": request_spec, "exit_code": 1, "worker_exit_code": None,
             "all_owned_ended": True, "reservation_released": True, "unknown_owned_identities": {},
             "first_error": {"type": type(error).__name__, "message": message[:8192], "original_message_chars": len(message), "message_truncated": len(message) > 8192},
             "secondary_errors": [], "timeout": isinstance(error, TimeoutError), "stdout_eof": True, "stderr_eof": True,
             "stdout": streams["stdout"], "stderr": streams["stderr"], "pre_spawn_refused": True,
             "worker_started": False, "job_created": False, "worker_pid": None, "worker_creation_ticks": None,
             "owner_pid": os.getpid(), "owner_creation_ticks": creation_ticks(os.getpid()),
             "original_payload_accesses": 0, "payload_access_proof": "tracking pre-spawn stdlib admission path; no Job/CreateProcess/numeric worker attempted",
             "refusal_binding": {"schema": tracking.SCHEMA + ".pre-spawn-refusal-binding.v1", "role": q["role"], "call_id": q["call_id"],
                                 "science_request": q["science_request"], "grant": q["grant"], "inputs": tracking.INPUTS, "roster": tracking.ROSTER,
                                 "source_freeze": q["source_freeze"], "local_debit_offset": 165, "resource_increment_s": 0},
             "observed_supervisor_interval_s": time.perf_counter() - entry,
             "output_usage": {"durable_result_bytes": 0, "stdout_bytes": 0, "stderr_bytes": 0,
                              "stream_cap_each_bytes": TRACKING_STREAM_CAP, "metadata_cap_bytes": TRACKING_METADATA_CAP,
                              "logical_combined_cap_bytes": expected["output_cap_bytes"], "combined_bytes_including_outer": 0}}
    publish_tracking_outer(output, outer, root, None)
    return outer


def live_replay_pre_spawn_refusal(root, request_spec, q, error, *, entry):
    """Actual refusal in this supervisor before any Job/CreateProcess attempt.

    This path creates no lease, reservation, worker, numeric import, or array read.
    It is not an admission certificate and is never used for ordinary worker errors.
    """
    from tools import banc_m3_live_replay_admission as tracking
    from tools import verify_c_banc_m3_startup_ray as ray
    require(q["mode"] == "original" and q["role"] == "live_replay", "new tracking refusal only")
    require(q["call_id"] == tracking.CALLS[0], "fixed refused call")
    tracking.exact_value(q["grant"], tracking.GRANT, "refusal preserves actual admitted J grant")
    expected = tracking.limits_for(q["role"])
    for key in ("outer_s", "child_s", "cleanup_s", "host_bytes", "output_cap_bytes"):
        tracking.exact_value(q["limits"][key], expected[key], "actual refused role bound")
    tracking.exact_value(q["output"], tracking.BASE + "/parent-" + q["call_id"] + "/result-supervision", "fresh finite refused output")
    live = ray.json_load(ray.leaf(ray.spec_for(q["ledger_path"], root, cap=524288), root, cap=524288))
    tracking.exact_value(live["direct_science_calls"][q["call_id"]],
                         {"request": request_spec, "state": "admitted", "mode": "original", "reserved_s": expected["outer_s"]}, "actual current admitted refused request")
    require(not live["startup_ray_original_workers"] and not live["startup_ray_original_reservations"]
            and not live["startup_ray_original_pending_actuals"] and not live["startup_ray_original_blocking_holds"]
            and live["original_jobs_running"] == 0 and live["original_reserved_s"] == 0
            and not ray.path(SHARED_LOCK, root).exists(), "refusal before actual shared lease/Job")
    output = ray.path(q["output"], root)
    require(not output.exists(), "refused call cannot publish twice")
    output.mkdir(parents=True)
    streams = {}
    for label in ("stdout", "stderr"):
        p = output / (label + ".bin")
        with p.open("xb") as stream:
            stream.flush();os.fsync(stream.fileno())
        streams[label] = {"path": p.relative_to(root).as_posix(), "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    message = str(error)
    outer = {"schema": SCHEMA + ".outer.v1", "request": request_spec, "exit_code": 1, "worker_exit_code": None,
             "all_owned_ended": True, "reservation_released": True, "unknown_owned_identities": {},
             "first_error": {"type": type(error).__name__, "message": message[:8192], "original_message_chars": len(message), "message_truncated": len(message) > 8192},
             "secondary_errors": [], "timeout": isinstance(error, TimeoutError), "stdout_eof": True, "stderr_eof": True,
             "stdout": streams["stdout"], "stderr": streams["stderr"], "pre_spawn_refused": True,
             "worker_started": False, "job_created": False, "worker_pid": None, "worker_creation_ticks": None,
             "owner_pid": os.getpid(), "owner_creation_ticks": creation_ticks(os.getpid()),
             "original_payload_accesses": 0, "payload_access_proof": "live replay pre-spawn stdlib admission path; no Job/CreateProcess/numeric worker attempted",
             "refusal_binding": {"schema": tracking.SCHEMA + ".pre-spawn-refusal-binding.v1", "role": q["role"], "call_id": q["call_id"],
                                 "science_request": q["science_request"], "grant": q["grant"], "inputs": tracking.INPUTS, "roster": tracking.ROSTER,
                                 "source_freeze": q["source_freeze"], "local_debit_offset": 165, "resource_increment_s": 0},
             "observed_supervisor_interval_s": time.perf_counter() - entry,
             "output_usage": {"durable_result_bytes": 0, "stdout_bytes": 0, "stderr_bytes": 0,
                              "stream_cap_each_bytes": TRACKING_STREAM_CAP, "metadata_cap_bytes": TRACKING_METADATA_CAP,
                              "logical_combined_cap_bytes": expected["output_cap_bytes"], "combined_bytes_including_outer": 0}}
    publish_tracking_outer(output, outer, root, None)
    return outer


def geometry_pre_spawn_refusal(root, request_spec, q, error, *, entry):
    """Actual refusal in this supervisor before any Job/CreateProcess attempt.

    This path creates no lease, reservation, worker, numeric import, or array read.
    It is not an admission certificate and is never used for ordinary worker errors.
    """
    from tools import banc_m3_local_foot_geometry_admission as tracking
    from tools import verify_c_banc_m3_startup_ray as ray
    require(q["mode"] == "original" and q["role"] in tracking.ROLES, "new tracking refusal only")
    require(q["call_id"] == tracking.CALLS[tracking.ROLES.index(q["role"])], "fixed refused call")
    tracking.exact_value(q["grant"], tracking.GRANT, "refusal preserves actual admitted J grant")
    expected = tracking.limits_for(q["role"])
    for key in ("outer_s", "child_s", "cleanup_s", "host_bytes", "output_cap_bytes"):
        tracking.exact_value(q["limits"][key], expected[key], "actual refused role bound")
    tracking.exact_value(q["output"], tracking.BASE + "/parent-" + q["call_id"] + "/result-supervision", "fresh finite refused output")
    live = ray.json_load(ray.leaf(ray.spec_for(q["ledger_path"], root, cap=524288), root, cap=524288))
    tracking.exact_value(live["direct_science_calls"][q["call_id"]],
                         {"request": request_spec, "state": "admitted", "mode": "original", "reserved_s": expected["outer_s"]}, "actual current admitted refused request")
    require(not live["startup_ray_original_workers"] and not live["startup_ray_original_reservations"]
            and not live["startup_ray_original_pending_actuals"] and not live["startup_ray_original_blocking_holds"]
            and live["original_jobs_running"] == 0 and live["original_reserved_s"] == 0
            and not ray.path(SHARED_LOCK, root).exists(), "refusal before actual shared lease/Job")
    output = ray.path(q["output"], root)
    require(not output.exists(), "refused call cannot publish twice")
    output.mkdir(parents=True)
    streams = {}
    for label in ("stdout", "stderr"):
        p = output / (label + ".bin")
        with p.open("xb") as stream:
            stream.flush();os.fsync(stream.fileno())
        streams[label] = {"path": p.relative_to(root).as_posix(), "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    message = str(error)
    outer = {"schema": SCHEMA + ".outer.v1", "request": request_spec, "exit_code": 1, "worker_exit_code": None,
             "all_owned_ended": True, "reservation_released": True, "unknown_owned_identities": {},
             "first_error": {"type": type(error).__name__, "message": message[:8192], "original_message_chars": len(message), "message_truncated": len(message) > 8192},
             "secondary_errors": [], "timeout": isinstance(error, TimeoutError), "stdout_eof": True, "stderr_eof": True,
             "stdout": streams["stdout"], "stderr": streams["stderr"], "pre_spawn_refused": True,
             "worker_started": False, "job_created": False, "worker_pid": None, "worker_creation_ticks": None,
             "owner_pid": os.getpid(), "owner_creation_ticks": creation_ticks(os.getpid()),
             "original_payload_accesses": 0, "payload_access_proof": "local foot geometry pre-spawn stdlib admission path; no Job/CreateProcess/numeric worker attempted",
             "refusal_binding": {"schema": tracking.SCHEMA + ".pre-spawn-refusal-binding.v1", "role": q["role"], "call_id": q["call_id"],
                                 "science_request": q["science_request"], "grant": q["grant"], "inputs": tracking.INPUTS, "roster": tracking.ROSTER,
                                 "source_freeze": q["source_freeze"], "local_debit_offset": 165, "resource_increment_s": 0},
             "observed_supervisor_interval_s": time.perf_counter() - entry,
             "output_usage": {"durable_result_bytes": 0, "stdout_bytes": 0, "stderr_bytes": 0,
                              "stream_cap_each_bytes": TRACKING_STREAM_CAP, "metadata_cap_bytes": TRACKING_METADATA_CAP,
                              "logical_combined_cap_bytes": expected["output_cap_bytes"], "combined_bytes_including_outer": 0}}
    publish_tracking_outer(output, outer, root, None)
    return outer


def contact_pre_spawn_refusal(root, request_spec, q, error, *, entry):
    """Actual refusal in this supervisor before any Job/CreateProcess attempt.

    This path creates no lease, reservation, worker, numeric import, or array read.
    It is not an admission certificate and is never used for ordinary worker errors.
    """
    from tools import banc_m3_contact_learning_admission as tracking
    from tools import verify_c_banc_m3_startup_ray as ray
    require(q["mode"] == "original" and q["role"] in tracking.ROLES, "new tracking refusal only")
    contract = tracking.stage_contract(root)
    retry = None
    if q["call_id"] in tracking.C_RETRY_CALLS + tracking.D_RUNTIME_CALLS + tracking.E_RUNTIME_CALLS + tracking.F_RUNTIME_CALLS + tracking.G_RUNTIME_CALLS:
        science = ray.json_load(ray.leaf(q["science_request"], root))
        def read_metadata(spec):
            return ray.json_load(ray.leaf(spec, root))
        retry = tracking.retry_binding(science, contract, read_metadata)
        require(retry is not None, "fixed c/d/e refusal descriptor required")
        for key in ("call_id", "role", "grant", "source_freeze"):
            tracking.exact_value(science[key], q[key], "actual admitted retry refusal " + key)
    else:
        require(q["call_id"] == contract["calls"][tracking.ROLES.index(q["role"])], "fixed refused call")
    tracking.exact_value(q["grant"], tracking.GRANT, "refusal preserves actual admitted J grant")
    expected = tracking.limits_for(q["role"], root=root)
    for key in ("outer_s", "child_s", "cleanup_s", "host_bytes", "output_cap_bytes"):
        tracking.exact_value(q["limits"][key], expected[key], "actual refused role bound")
    tracking.exact_value(q["output"], tracking.BASE + "/parent-" + q["call_id"] + "/result-supervision", "fresh finite refused output")
    live = ray.json_load(ray.leaf(ray.spec_for(q["ledger_path"], root, cap=524288), root, cap=524288))
    tracking.exact_value(live["direct_science_calls"][q["call_id"]],
                         {"request": request_spec, "state": "admitted", "mode": "original", "reserved_s": expected["outer_s"]}, "actual current admitted refused request")
    require(not live["startup_ray_original_workers"] and not live["startup_ray_original_reservations"]
            and not live["startup_ray_original_pending_actuals"] and not live["startup_ray_original_blocking_holds"]
            and live["original_jobs_running"] == 0 and live["original_reserved_s"] == 0
            and not ray.path(SHARED_LOCK, root).exists(), "refusal before actual shared lease/Job")
    output = ray.path(q["output"], root)
    require(not output.exists(), "refused call cannot publish twice")
    output.mkdir(parents=True)
    streams = {}
    for label in ("stdout", "stderr"):
        p = output / (label + ".bin")
        with p.open("xb") as stream:
            stream.flush();os.fsync(stream.fileno())
        streams[label] = {"path": p.relative_to(root).as_posix(), "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    message = str(error)
    outer = {"schema": SCHEMA + ".outer.v1", "request": request_spec, "exit_code": 1, "worker_exit_code": None,
             "all_owned_ended": True, "reservation_released": True, "unknown_owned_identities": {},
             "first_error": {"type": type(error).__name__, "message": message[:8192], "original_message_chars": len(message), "message_truncated": len(message) > 8192},
             "secondary_errors": [], "timeout": isinstance(error, TimeoutError), "stdout_eof": True, "stderr_eof": True,
             "stdout": streams["stdout"], "stderr": streams["stderr"], "pre_spawn_refused": True,
             "worker_started": False, "job_created": False, "worker_pid": None, "worker_creation_ticks": None,
             "owner_pid": os.getpid(), "owner_creation_ticks": creation_ticks(os.getpid()),
             "original_payload_accesses": 0, "payload_access_proof": "contact learning pre-spawn stdlib admission path; no Job/CreateProcess/numeric worker attempted",
             "refusal_binding": {"schema": tracking.SCHEMA + ".pre-spawn-refusal-binding.v1", "role": q["role"], "call_id": q["call_id"],
                                 "science_request": q["science_request"], "grant": q["grant"], "inputs": tracking.INPUTS, "roster": tracking.ROSTER,
                                 "source_freeze": q["source_freeze"], "local_debit_offset": 165, "resource_increment_s": 0},
             "observed_supervisor_interval_s": time.perf_counter() - entry,
             "output_usage": {"durable_result_bytes": 0, "stdout_bytes": 0, "stderr_bytes": 0,
                              "stream_cap_each_bytes": TRACKING_STREAM_CAP, "metadata_cap_bytes": TRACKING_METADATA_CAP,
                              "logical_combined_cap_bytes": expected["output_cap_bytes"], "combined_bytes_including_outer": 0}}
    if retry is not None:
        outer["refusal_binding"]["body_retry"] = science["body_retry"]
    publish_tracking_outer(output, outer, root, None)
    return outer


def directory_bytes(directory, root):
    from tools import verify_c_banc_m3_startup_ray as ray
    if not directory.exists():
        return 0
    require(directory.is_dir(), "finite output directory")
    total = 0
    for item in directory.rglob("*"):
        safe = ray.path(item.relative_to(root).as_posix(), root)
        if safe.is_file():
            total += safe.stat().st_size
    return total


def publish_tracking_outer(output, outer, root, result_directory):
    result_bytes = directory_bytes(result_directory, root) if result_directory is not None else 0
    supervision_bytes = directory_bytes(output, root)
    usage = outer.setdefault("output_usage", {})
    usage.update(durable_result_bytes=result_bytes,
                 stdout_bytes=outer["stdout"]["bytes"] if outer.get("stdout") else (output / "stdout.bin").stat().st_size if (output / "stdout.bin").exists() else 0,
                 stderr_bytes=outer["stderr"]["bytes"] if outer.get("stderr") else (output / "stderr.bin").stat().st_size if (output / "stderr.bin").exists() else 0,
                 stream_cap_each_bytes=TRACKING_STREAM_CAP, metadata_cap_bytes=TRACKING_METADATA_CAP)
    combined_cap = usage["logical_combined_cap_bytes"]
    require(result_bytes <= combined_cap - (3 << 20) and usage["stdout_bytes"] <= TRACKING_STREAM_CAP
            and usage["stderr_bytes"] <= TRACKING_STREAM_CAP, "actual tracking result/stream reserve")
    usage["combined_bytes_including_outer"] = result_bytes + supervision_bytes
    for _ in range(8):
        raw = json.dumps(outer, ensure_ascii=True, allow_nan=False, sort_keys=True).encode("utf8")
        require(len(raw) <= TRACKING_METADATA_CAP, "tracking outer metadata64KiB cap")
        expected = result_bytes + supervision_bytes + len(raw)
        require(expected <= combined_cap, "logical result plus opaque streams plus outer cap")
        if usage["combined_bytes_including_outer"] == expected:
            durable(output / "outer.json", outer)
            require(directory_bytes(output, root) + result_bytes == expected, "actual durable combined byte readback")
            return
        usage["combined_bytes_including_outer"] = expected
    raise ValueError("finite outer metadata byte count")


def worker_environment(role, token, executable, *, inherited=None):
    """Actual spawn environment, observable before any child numerical imports."""
    environment = dict(os.environ if inherited is None else inherited)
    environment["FLYLAB_DIRECT_SCIENCE_OWNER"] = json.dumps(token, allow_nan=False)
    # The Windows venv redirector must not replace the suspended Job primary.
    environment["__PYVENV_LAUNCHER__"] = executable
    if role in ("saved_startup_transfer", "tracking_diagnosis", "geometry_analyze", "contact_direction", "contact_candidate_audit", "contact_body_audit"):
        for thread_option in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[thread_option] = "1"
    return environment


def generated_transferred_totals(ledger, policy, root, ray):
    """Mirror the production Native ordered totals for transferred allocations."""
    before = ray.json_load(ray.leaf(policy["before_ledger"], root, cap=524288))
    expected = {"schema": "flylab.m3.direct-science.policy.v1", "generated_local_limit_s": 300,
                "generated_shared_limit_s": 1700, "new_generated_grant_s": 0,
                "original_authorized": False, "minimum_close_reserved_s": 20, "local_debit_offset": 118,
                "generated_shared_baseline_s": 80.1900156}
    for key, value in expected.items():
        ray.exact_value(policy[key], value, "unchanged transferred initial policy")
    require(len(before["debits"]) == 118, "exact initial debit offset118")
    ray.exact_value(ledger["debits"][:118], before["debits"], "initial debit prefix preserved")
    for prior_key, live_key in (("generated_stage_before_s", "generated_stage_used_s"),
                               ("generated_shared_before_s", "generated_shared_cumulative_used_s")):
        require(type(policy[prior_key]) in (int, float) and math.isfinite(policy[prior_key])
                and policy[prior_key] >= 0, "finite typed initial carry")
        ray.exact_value(policy[prior_key], before[live_key], "exact initial carry")
    ray.exact_value(before["generated_shared_cumulative_limit_s"], 1700, "initial shared cap")
    ray.exact_value(ledger["generated_shared_cumulative_limit_s"], 1700, "live shared cap")
    ray.exact_value(before["generated_shared_remaining_s"],
                    1700.0 - policy["generated_shared_baseline_s"] - policy["generated_stage_before_s"], "initial remaining order")
    total = 0.
    for debit in ledger["debits"][118:]:
        if debit["mode"] == "generated":
            value = debit["actual_outer_full_return_s"]
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "typed finite generated debit")
            total += float(value)
    expected_totals = {"direct_science_generated_used_s": total,
                       "generated_stage_used_s": float(policy["generated_stage_before_s"]) + total,
                       "generated_shared_cumulative_used_s": float(policy["generated_shared_before_s"]) + total}
    expected_totals["generated_shared_remaining_s"] = 1700.0 - policy["generated_shared_baseline_s"] - expected_totals["generated_stage_used_s"]
    for key, value in expected_totals.items():
        require(type(ledger[key]) in (int, float) and math.isfinite(ledger[key]) and ledger[key] >= 0,
                "typed finite transferred live total")
        ray.exact_value(ledger[key], value, "exact live ordered generated accounting")


def generated_budget_limit(ledger, root):
    """Validate the pinned allocation event without changing any budget or history."""
    from tools import banc_m3_checkpoint_harness_runtime as harness
    from tools import banc_m3_checkpoint_runtime as checkpoint
    from tools import banc_m3_conserved_resource as resource
    from tools import verify_c_banc_m3_startup_ray as ray
    if harness.KEY in ledger:
        checkpoint = harness
    if checkpoint.KEY in ledger:
        return checkpoint.validate(ledger, root)['caps']['generated_local']
    if resource.J_KEY in ledger or resource.I_KEY in ledger or resource.H_KEY in ledger or resource.G_KEY in ledger or resource.ALLOCATION_KEY in ledger or any(ledger.get(resource.CAP_FIELDS[key]) == value for key, value in resource.NEW_CAPS.items()):
        caps = resource.current_caps(ledger, root, generated_budget_limit)
        return caps["generated_local"]
    limit = ledger["direct_science_generated_limit_s"]
    require(type(limit) is int and limit in (300, 380, 460, 600, 900, 960, 964), "finite generated effective allocation")
    if limit < 964:
        require("direct_science_generated_runtime_residual_transfer" not in ledger, "no residual pointer in lower allocation")
    if limit < 960:
        require("direct_science_generated_runtime_transfer" not in ledger, "no runtime pointer in lower allocation")
    if limit < 900:
        require("direct_science_generated_continuation_transfer" not in ledger, "no continuation pointer in lower allocation")
    if limit < 600:
        require("direct_science_generated_final_transfer" not in ledger, "no final pointer in lower allocation")
    if limit == 300:
        require("direct_science_generated_transfer" not in ledger
                and "direct_science_generated_additional_transfer" not in ledger, "no unreflected transfer")
        return limit
    event = ray.json_load(ray.leaf(ledger["direct_science_generated_transfer"], root))
    expected = {"schema": "flylab.m3.contact-metadata.generated-transfer.v1",
                "transfer_id": "contact-metadata-20261007-80", "transfer_s": 80,
                "from_local_limit_s": 300, "to_local_limit_s": 380, "new_resource_grant_s": 0}
    for key, value in expected.items():
        ray.exact_value(event[key], value, "one fixed generated allocation transfer")
    ray.exact_value(event["initial_policy"], ledger["direct_science_policy"], "unchanged initial policy")
    policy = ray.json_load(ray.leaf(event["initial_policy"], root))
    generated_transferred_totals(ledger, policy, root, ray)
    require(policy["generated_local_limit_s"] == 300 and policy["generated_shared_limit_s"] == 1700
            and policy["new_generated_grant_s"] == 0 and policy["minimum_close_reserved_s"] == 20,
            "initial allocation and closing reserve preserved")
    authority = ray.json_load(ray.leaf(event["direct_human_auto_authority"], root))
    require(authority["schema"] == "flylab.m3.contact-metadata.current-human-autonomy.v1"
            and authority["direct_human_authority"] is True and authority["automatic_authority"] is True
            and authority["new_resource_grant_s"] == 0
            and authority["literal_latest_user"] == "앞으로 내 승인 받지 말고 스스로 판단해서 진행해줘. 개입할 상황이면 내가 알아서 개입할게",
            "pinned direct human autonomy without resource grant")
    anchor = ray.json_load(ray.leaf(event["before_ledger"], root, cap=524288))
    require("direct_science_generated_transfer" not in anchor and "direct_science_generated_final_transfer" not in anchor
            and "direct_science_generated_continuation_transfer" not in anchor
            and anchor["direct_science_generated_limit_s"] == 300,
            "one transfer anchored before reallocation")
    ray.exact_value(anchor["direct_science_policy"], ledger["direct_science_policy"], "anchor same policy")
    ray.exact_value(ledger["debits"][:len(anchor["debits"])], anchor["debits"], "transfer debit history preserved")
    for call, prior in anchor["direct_science_calls"].items():
        require(prior["state"] == "charged", "transfer calls already charged")
        ray.exact_value(ledger["direct_science_calls"][call], prior, "transfer prior call unchanged")
    require(anchor["generated_shared_remaining_s"] >= 380.0 - anchor["direct_science_generated_used_s"],
            "shared allocation covers entire effective unspent limit")
    generated_transferred_totals(anchor, policy, root, ray)
    require("direct_science_generated_additional_transfer" not in anchor, "first anchor has no additional transfer")
    if limit == 380:
        require("direct_science_generated_additional_transfer" not in ledger, "380 has only first transfer")
        return limit
    extra = ray.json_load(ray.leaf(ledger["direct_science_generated_additional_transfer"], root))
    expected_extra = {"schema": "flylab.m3.contact-body-retry.generated-transfer.v1",
                      "transfer_id": "contact-body-retry-20261007-80", "transfer_s": 80,
                      "from_local_limit_s": 380, "to_local_limit_s": 460, "new_resource_grant_s": 0}
    for key, value in expected_extra.items():
        ray.exact_value(extra[key], value, "one fixed additional allocation")
    ray.exact_value(extra["initial_policy"], ledger["direct_science_policy"], "additional same initial policy")
    ray.exact_value(extra["first_transfer"], ledger["direct_science_generated_transfer"], "additional same first transfer")
    ray.exact_value(extra["direct_human_auto_authority"], event["direct_human_auto_authority"], "same validated autonomy")
    previous = ray.json_load(ray.leaf(extra["before_ledger"], root, cap=524288))
    require(previous["direct_science_generated_limit_s"] == 380
            and "direct_science_generated_additional_transfer" not in previous
            and "direct_science_generated_final_transfer" not in previous
            and "direct_science_generated_continuation_transfer" not in previous, "additional anchored at380 only")
    ray.exact_value(previous["direct_science_policy"], ledger["direct_science_policy"], "additional predecessor policy")
    ray.exact_value(previous["direct_science_generated_transfer"], extra["first_transfer"], "additional predecessor first event")
    require(generated_budget_limit(previous, root) == 380, "complete valid380 predecessor")
    ray.exact_value(ledger["debits"][:len(previous["debits"])], previous["debits"], "additional preserved debit prefix")
    for call, prior in previous["direct_science_calls"].items():
        require(prior["state"] == "charged", "additional predecessor calls charged")
        ray.exact_value(ledger["direct_science_calls"][call], prior, "additional preserved calls")
    require(previous["generated_shared_remaining_s"] >= 460.0 - previous["direct_science_generated_used_s"],
            "shared allocation covers complete additional unspent limit")
    if limit == 460:
        return limit
    final = ray.json_load(ray.leaf(ledger["direct_science_generated_final_transfer"], root))
    expected_final = {"schema": "flylab.m3.contact-body-retry.generated-final-transfer.v1",
                      "transfer_id": "contact-body-retry-20261007-final-140", "transfer_s": 140,
                      "from_local_limit_s": 460, "to_local_limit_s": 600, "new_resource_grant_s": 0}
    for key, value in expected_final.items():
        ray.exact_value(final[key], value, "one fixed final allocation")
    ray.exact_value(final["initial_policy"], ledger["direct_science_policy"], "final same initial policy")
    ray.exact_value(final["first_transfer"], ledger["direct_science_generated_transfer"], "final same first transfer")
    ray.exact_value(final["additional_transfer"], ledger["direct_science_generated_additional_transfer"], "final same additional transfer")
    ray.exact_value(final["direct_human_auto_authority"], event["direct_human_auto_authority"], "final same validated autonomy")
    previous_final = ray.json_load(ray.leaf(final["before_ledger"], root, cap=524288))
    require(previous_final["direct_science_generated_limit_s"] == 460
            and "direct_science_generated_final_transfer" not in previous_final
            and "direct_science_generated_continuation_transfer" not in previous_final, "final anchored at460 only")
    for key, field in (("direct_science_policy", "initial_policy"), ("direct_science_generated_transfer", "first_transfer"),
                       ("direct_science_generated_additional_transfer", "additional_transfer")):
        ray.exact_value(previous_final[key], final[field], "final predecessor exact " + field)
    require(generated_budget_limit(previous_final, root) == 460, "complete valid460 predecessor")
    ray.exact_value(ledger["debits"][:len(previous_final["debits"])], previous_final["debits"], "final preserved debit prefix")
    for call, prior in previous_final["direct_science_calls"].items():
        require(prior["state"] == "charged", "final predecessor calls charged")
        ray.exact_value(ledger["direct_science_calls"][call], prior, "final preserved calls")
    require(previous_final["generated_shared_remaining_s"] >= 600.0 - previous_final["direct_science_generated_used_s"],
            "shared allocation covers complete final unspent limit")
    if limit == 600:
        return limit
    continuation = ray.json_load(ray.leaf(ledger["direct_science_generated_continuation_transfer"], root))
    expected_continuation = {"schema": "flylab.m3.contact-body-retry.generated-continuation-transfer.v1",
                             "transfer_id": "contact-body-retry-20261007-continuation-300", "transfer_s": 300,
                             "from_local_limit_s": 600, "to_local_limit_s": 900, "new_resource_grant_s": 0}
    for key, value in expected_continuation.items():
        ray.exact_value(continuation[key], value, "one fixed continuation allocation")
    links = (("direct_science_policy", "initial_policy"), ("direct_science_generated_transfer", "first_transfer"),
             ("direct_science_generated_additional_transfer", "additional_transfer"), ("direct_science_generated_final_transfer", "final_transfer"))
    for key, field in links:
        ray.exact_value(continuation[field], ledger[key], "continuation same " + field)
    ray.exact_value(continuation["direct_human_auto_authority"], event["direct_human_auto_authority"], "continuation same validated autonomy")
    predecessor = ray.json_load(ray.leaf(continuation["before_ledger"], root, cap=524288))
    require(predecessor["direct_science_generated_limit_s"] == 600
            and "direct_science_generated_continuation_transfer" not in predecessor, "continuation anchored at600 only")
    for key, field in links:
        ray.exact_value(predecessor[key], continuation[field], "continuation predecessor exact " + field)
    require(generated_budget_limit(predecessor, root) == 600, "complete valid600 predecessor")
    ray.exact_value(ledger["debits"][:len(predecessor["debits"])], predecessor["debits"], "continuation preserved debit prefix")
    for call, prior in predecessor["direct_science_calls"].items():
        require(prior["state"] == "charged", "continuation predecessor calls charged")
        ray.exact_value(ledger["direct_science_calls"][call], prior, "continuation preserved calls")
    require(predecessor["generated_shared_remaining_s"] >= 900.0 - predecessor["direct_science_generated_used_s"],
            "shared allocation covers complete continuation unspent limit")
    if limit == 900:
        return limit
    runtime = ray.json_load(ray.leaf(ledger["direct_science_generated_runtime_transfer"], root))
    from tools.banc_m3_contact_learning_admission import D_HISTORY_PINS
    ray.exact_value(ledger["direct_science_generated_runtime_transfer"], D_HISTORY_PINS["runtime_transfer"], "one actual registered runtime pointer")
    links = links + (("direct_science_generated_continuation_transfer", "continuation_transfer"),)
    expected_runtime = {"schema": "flylab.m3.contact-body-runtime.generated-transfer.v1",
                        "transfer_id": "contact-body-runtime-20261007-60",
                        "from_local_limit_s": 900, "to_local_limit_s": 960,
                        "transfer_s": 60, "new_resource_grant_s": 0,
                        "before_ledger": D_HISTORY_PINS["closed276"],
                        "direct_human_auto_authority": event["direct_human_auto_authority"]}
    expected_runtime.update({field: ledger[key] for key, field in links})
    ray.exact_value(runtime, expected_runtime, "exact thirteen runtime event fields")
    anchor_runtime = ray.json_load(ray.leaf(runtime["before_ledger"], root, cap=524288))
    require(len(anchor_runtime["debits"]) == 276 and len(anchor_runtime["direct_science_calls"]) == 158
            and anchor_runtime["direct_science_generated_limit_s"] == 900
            and "direct_science_generated_runtime_transfer" not in anchor_runtime, "exact closed276 runtime anchor before sole allocation")
    for key, field in links:
        ray.exact_value(anchor_runtime[key], runtime[field], "runtime anchor unchanged " + field)
    require(generated_budget_limit(anchor_runtime, root) == 900, "complete valid900 predecessor")
    ray.exact_value(ledger["debits"][:276], anchor_runtime["debits"], "runtime complete ordered276 prefix")
    require(len({d["cost_id"] for d in ledger["debits"]}) == len(ledger["debits"]), "no duplicate runtime debit")
    for call, prior in anchor_runtime["direct_science_calls"].items():
        require(prior["state"] == "charged", "all runtime predecessor calls charged")
        ray.exact_value(ledger["direct_science_calls"][call], prior, "runtime all158 prior calls preserved")
    unused = 960.0 - anchor_runtime["direct_science_generated_used_s"]
    require(0 <= unused <= anchor_runtime["generated_shared_remaining_s"]
            and 0 <= 960.0-ledger["direct_science_generated_used_s"] <= ledger["generated_shared_remaining_s"], "full runtime unused conservation within same shared1700")
    if limit == 960:
        return limit
    from tools.banc_m3_contact_learning_admission import RUNTIME_RECOVERY_PINS
    ray.exact_value(ledger["direct_science_generated_runtime_residual_transfer"], RUNTIME_RECOVERY_PINS["residual_transfer"], "exact registered residual pointer")
    residual = ray.json_load(ray.leaf(RUNTIME_RECOVERY_PINS["residual_transfer"], root))
    residual_links = links + (("direct_science_generated_runtime_transfer", "runtime_transfer"),)
    expected_residual = {"schema": "flylab.m3.contact-body-runtime.generated-residual-transfer.v1",
                         "transfer_id": "contact-body-runtime-20261007-residual-4", "from_local_limit_s": 960,
                         "to_local_limit_s": 964, "transfer_s": 4, "new_resource_grant_s": 0,
                         "before_ledger": RUNTIME_RECOVERY_PINS["closed277"], "direct_human_auto_authority": runtime["direct_human_auto_authority"]}
    expected_residual.update({field: ledger[key] for key, field in residual_links})
    ray.exact_value(residual, expected_residual, "exact residual fourteen fields")
    residual_anchor = ray.json_load(ray.leaf(residual["before_ledger"], root, cap=524288))
    require(len(residual_anchor["debits"]) == 277 and len(residual_anchor["direct_science_calls"]) == 159
            and "direct_science_generated_runtime_residual_transfer" not in residual_anchor, "exact fresh closed277 residual anchor")
    require(generated_budget_limit(residual_anchor, root) == 960, "complete old960 predecessor")
    for key, field in residual_links:
        ray.exact_value(residual_anchor[key], residual[field], "residual unchanged " + field)
    ray.exact_value(ledger["debits"][:277], residual_anchor["debits"], "all residual277 debits preserved")
    for call, prior in residual_anchor["direct_science_calls"].items():
        require(prior["state"] == "charged", "all159 residual prior calls charged")
        ray.exact_value(ledger["direct_science_calls"][call], prior, "all159 residual calls unchanged")
    require(0 <= 964.0-residual_anchor["direct_science_generated_used_s"] <= residual_anchor["generated_shared_remaining_s"]
            and 0 <= 964.0-ledger["direct_science_generated_used_s"] <= ledger["generated_shared_remaining_s"], "complete residual unused conservation")
    return limit


def _supervise(name, *, root=ROOT, entry=ENTRY):
    from tools import verify_c_banc_m3_startup_ray as ray
    root = root.resolve()
    from tools import banc_m3_conserved_resource as resource
    require(type(name) is str and name.endswith(".json"), "supervisor text request metadata only")
    request_spec = resource.text_spec(name, root)
    q = resource.read_pin(request_spec, root)
    require(q["schema"] == SCHEMA + ".supervision.v1" and q["mode"] in ("original", "generated"), "finite direct dispatch schema/mode")
    require(type(q["call_id"]) is str and q["call_id"] and type(q["role"]) is str, "exact call/role identity")
    require(q["shared_original_lock_path"] == SHARED_LOCK, "existing shared Original lock, no private substitute")
    for key in ("outer_s", "child_s", "cleanup_s", "output_cap_bytes", "host_bytes"):
        require(type(q["limits"][key]) is int and q["limits"][key] > 0, "typed finite direct limits")
    limits = q["limits"]
    require(limits["child_s"] + limits["cleanup_s"] <= limits["outer_s"], "separate cleanup/publication reserve")
    # The F restriction is global and precedes every Original-specific admission.
    from tools import banc_m3_conserved_resource as resource
    ray.exact_value(q["ledger_path"], BASE + "/parent-e358-execution-ledger.json", "one actual dispatch ledger")
    dispatch_spec = resource.text_spec(q["ledger_path"], root, 524288)
    dispatch_raw = resource.raw_pin(dispatch_spec, root, 524288)
    dispatch_ledger = ray.json_load(dispatch_raw)
    from tools import banc_m3_checkpoint_harness_runtime as harness
    from tools import banc_m3_checkpoint_runtime as checkpoint
    if harness.KEY in dispatch_ledger:
        checkpoint = harness
    k_dispatch = checkpoint.KEY in dispatch_ledger
    if k_dispatch:
        try:
            f_dispatch = checkpoint.dispatch(dispatch_ledger, q, request_spec, root, phase='admitted')
            checkpoint.source_fence(q['source_freeze'], root)
        except Exception as error:
            return checkpoint.initial_refusal(root, request_spec, q, dispatch_spec, dispatch_raw, error, entry=entry)
    elif resource.J_KEY in dispatch_ledger:
        try:
            f_dispatch = resource.prequalification_dispatch(dispatch_ledger, q, request_spec, root, generated_budget_limit, phase="admitted")
        except Exception as error:
            try:
                return j_initial_guard_refusal(root, request_spec, q, dispatch_spec, dispatch_raw, error, entry=entry)
            except Exception as publication_error:
                error.add_note("J initial-refusal identity/publication failed; actual return remains PENDING/HOLD: " + str(publication_error))
                raise error

    elif resource.I_KEY in dispatch_ledger:
        try:
            f_dispatch = resource.prequalification_dispatch(dispatch_ledger, q, request_spec, root, generated_budget_limit, phase="admitted")
        except Exception as error:
            try:
                return i_initial_guard_refusal(root, request_spec, q, dispatch_spec, dispatch_raw, error, entry=entry)
            except Exception as publication_error:
                error.add_note("I initial-refusal identity/publication failed; actual return remains PENDING/HOLD: " + str(publication_error))
                raise error
    elif resource.H_KEY in dispatch_ledger:
        # Only this first global guard is caught, before SharedLock/OwnedJob.
        try:
            f_dispatch = resource.prequalification_dispatch(dispatch_ledger, q, request_spec, root, generated_budget_limit, phase="admitted")
        except Exception as error:
            try:
                return h_initial_guard_refusal(root, request_spec, q, dispatch_spec, dispatch_raw, error, entry=entry)
            except Exception as publication_error:
                error.add_note("H initial-refusal identity/publication failed; actual return remains PENDING/HOLD: " + str(publication_error))
                raise error
    elif resource.G_KEY in dispatch_ledger:
        # This lexical catch is solely the initial global G guard. No lock or Job
        # constructor has been attempted; the locked repeat below has no such catch.
        try:
            f_dispatch = resource.prequalification_dispatch(dispatch_ledger, q, request_spec, root, generated_budget_limit, phase="admitted")
        except Exception as error:
            try:
                return g_initial_guard_refusal(root, request_spec, q, dispatch_spec, dispatch_raw, error, entry=entry)
            except Exception as publication_error:
                error.add_note("G initial-refusal identity/publication failed; actual return remains PENDING/HOLD: " + str(publication_error))
                raise error
    else:
        f_dispatch = resource.prequalification_dispatch(dispatch_ledger, q, request_spec, root, generated_budget_limit, phase="admitted")
    f_dispatch_raw = dispatch_raw if f_dispatch is not None else None
    tracking_role = q["mode"] == "original" and q["role"] in TRACKING_ROLES
    replay_role = q["mode"] == "original" and q["role"] == "live_replay"
    geometry_role = q["mode"] == "original" and q["role"] in ("geometry_capture", "geometry_analyze")
    contact_role = q["mode"] == "original" and q["call_id"].startswith("contact-learning-")
    if contact_role:
        try:
            from tools import banc_m3_contact_learning_admission as contact
            science = contact.Admission(q["science_request"]["path"], "pre_spawn", root=root, entry=entry)
        except Exception as error:
            if k_dispatch:
                return checkpoint.initial_refusal(root, request_spec, q, dispatch_spec, dispatch_raw, error, entry=entry)
            return contact_pre_spawn_refusal(root, request_spec, q, error, entry=entry)
    if geometry_role:
        try:
            from tools import banc_m3_local_foot_geometry_admission as geometry
            science = geometry.Admission(q["science_request"]["path"], "pre_spawn", root=root, entry=entry)
        except Exception as error:
            return geometry_pre_spawn_refusal(root, request_spec, q, error, entry=entry)
    if replay_role:
        try:
            from tools import banc_m3_live_replay_admission as replay
            science = replay.Admission(q["science_request"]["path"], "pre_spawn", root=root, entry=entry)
        except Exception as error:
            return live_replay_pre_spawn_refusal(root, request_spec, q, error, entry=entry)
    if tracking_role:
        try:
            from tools import banc_m3_startup_tracking_admission as tracking
            science = tracking.Admission(q["science_request"]["path"], "pre_spawn", root=root, entry=entry)
        except Exception as error:
            return tracking_pre_spawn_refusal(root, request_spec, q, error, entry=entry)
    if tracking_role or replay_role or geometry_role or contact_role:
        freeze = science.freeze  # already deadline-first cold-fenced by this admission
    else:
        freeze = resource.read_pin(q["source_freeze"], root) if f_dispatch is not None and f_dispatch["epoch"] == resource.G_EPOCH else ray.json_load(ray.leaf(q["source_freeze"], root))
        require(freeze["ROOT_review_cycle_complete"] is True, "actual reviewed direct sources")
        if not k_dispatch:
            for source in freeze["sources"].values():
                (resource.source_text if f_dispatch is not None and f_dispatch["epoch"] == resource.G_EPOCH else ray.source_leaf)(source, root)
        require("tools/banc_m3_startup_ray_supervisor.py" in freeze["sources"] and ray.SOURCE in freeze["sources"], "actual two production entrypoints frozen")
    if q["mode"] == "original":
        if contact_role:
            require(science.role in contact.ROLES, "one fixed new contact role")
        elif geometry_role:
            require(science.role in geometry.ROLES, "one of two admitted geometry roles")
        elif replay_role:
            require(science.role == "live_replay", "one replay-only Original admission")
        elif q["role"] in TRACKING_ROLES:
            require(tracking_role, "actual tracking pre-spawn admitted")
        elif q["role"] == "saved_startup_transfer":
            from tools import banc_m3_saved_original_admission as saved
            science = saved.Admission(q["science_request"]["path"], "pre_spawn", root=root, entry=entry)
        else:
            science = ray.Admission(q["science_request"]["path"], "pre_spawn", root=root, entry=entry)
        ray.exact_value(science.request_spec, q["science_request"], "actual requested science entry")
        ray.exact_value(science.request["call_id"], q["call_id"], "same actual science call")
        ray.exact_value(science.role, q["role"], "same actual science role")
        ray.exact_value(science.request["grant"], q["grant"], "same actual grant")
        ray.exact_value(science.request["source_freeze"], q["source_freeze"], "same direct source freeze")
        for key in ("outer_s", "child_s", "host_bytes", "output_cap_bytes"):
            ray.exact_value(limits[key], science.limits[key], "same role resource bound")
        if contact_role:
            module = "tools.analyze_c_banc_m3_contact_learning" if q["role"] in ("contact_candidate_audit", "contact_body_audit") else "tools.verify_c_banc_m3_contact_learning"
            argv = [sys._base_executable, "-m", module, "--request", q["science_request"]["path"]]
        elif geometry_role:
            module = "tools.observe_c_banc_m3_local_foot_geometry" if q["role"] == "geometry_capture" else "tools.analyze_c_banc_m3_local_foot_geometry"
            argv = [sys._base_executable, "-m", module, "--request", q["science_request"]["path"]]
        elif replay_role:
            argv = [sys._base_executable, "-m", "tools.verify_c_banc_m3_live_replay", "--request", q["science_request"]["path"]]
        elif q["role"] in TRACKING_ROLES:
            module = "tools.analyze_c_banc_m3_startup_tracking" if q["role"] == "tracking_diagnosis" else "tools.observe_c_banc_m3_startup_tracking"
            argv = [sys._base_executable, "-m", module, "--request", q["science_request"]["path"]]
        elif q["role"] == "saved_startup_transfer":
            argv = [sys._base_executable, "-m", "tools.analyze_c_banc_m3_startup_ray_transfer", "--request", q["science_request"]["path"]]
        else:
            argv = [sys._base_executable, "-m", "tools.verify_c_banc_m3_startup_ray", "--request", q["science_request"]["path"]]
    else:
        ledger = ray.json_load(ray.leaf(ray.spec_for(q["ledger_path"], root, cap=524288), root, cap=524288))
        expected = {"request": request_spec, "state": "admitted", "mode": "generated", "reserved_s": limits["outer_s"]}
        ray.exact_value(ledger["direct_science_calls"][q["call_id"]], expected, "actual generated call admitted")
        ray.exact_value(ledger["direct_science_reservations"], {q["call_id"]: {"request": request_spec, "reserved_s": limits["outer_s"]}}, "one exact generated reservation")
        require(not ledger["direct_science_pending_actuals"] and not ledger["direct_science_blocking_holds"], "no generated pending/unknown or duplicate replay")
        used = ray.finite_float(ledger["direct_science_generated_used_s"], "generated direct used")
        ray.finite_float(ledger["generated_shared_cumulative_used_s"], "actual shared generated used")
        from tools import banc_m3_conserved_resource as resource
        caps = resource.current_caps(ledger, root, generated_budget_limit)
        effective_limit = caps["generated_local"] if caps in (resource.NEW_CAPS, checkpoint.CAPS) else generated_budget_limit(ledger, root)
        i_gate_b = (q["call_id"] == resource.I_QUAL_CALLS[3] and f_dispatch is not None
                    and f_dispatch["epoch"] == resource.I_EPOCH and limits == resource.i_generated_plan("gate-b", root)["limits"])
        j_gate_b = (q["call_id"] == resource.J_QUAL_CALLS[3] and f_dispatch is not None
                    and f_dispatch["epoch"] == resource.J_EPOCH and limits == resource.j_generated_plan("gate-b", root)["limits"])
        l_large = (k_dispatch and checkpoint is harness and f_dispatch['epoch'] == harness.EPOCH
                   and q['call_id'] in (harness.BOOTSTRAP_CALL, harness.QUAL_CALLS[2], harness.QUAL_CALLS[3])
                   and limits == harness.plan(q['call_id'], freeze, root)['limits']
                   and f_dispatch['caps'] == caps)
        k_gate_b = k_dispatch and checkpoint is not harness and q['call_id'] == checkpoint.QUAL_CALLS[3]
        require((limits["outer_s"] <= 120 or i_gate_b or j_gate_b or k_gate_b or l_large) and used + limits["outer_s"] + (25 if checkpoint is harness and q["call_id"] in harness.QUAL_CALLS else 20) <= effective_limit
                and limits["outer_s"] + (25 if checkpoint is harness and q["call_id"] in harness.QUAL_CALLS else 20) <= ledger["generated_shared_remaining_s"], "actual paired Generated budget fit; same ordered remaining and close20")
        behavior = q["generated_behavior"]
        require(behavior in ("normal", "nonzero", "timeout-descendant", "command"), "finite generated behavior whitelist")
        if behavior == "command":
            extra = q["generated_argv"]
            require(type(extra) is list and all(type(a) is str and a for a in extra) and len(extra) >= 2 and extra[0] == "-m" and extra[1] in ("ruff", "pytest"), "only actual frozen lint/numerical/gate tests command")
            if k_dispatch:
                plan = checkpoint.plan(q['call_id'], freeze, root)
                ray.exact_value(extra, plan['argv'], 'whole exact K bootstrap/qualification command')
                ray.exact_value(limits, plan['limits'], 'literal K envelope')
            elif q["call_id"] in resource.J_QUAL_CALLS:
                role = ("lint", "math", "gate-a", "gate-b")[resource.J_QUAL_CALLS.index(q["call_id"])]
                plan = resource.j_generated_plan(role, root)
                ray.exact_value(extra, plan["argv"], "whole unchanged J19/21/32 qualification argv")
                ray.exact_value(limits, plan["limits"], "fixed J time/host/output bounds")
                require(f_dispatch is not None and f_dispatch["epoch"] == resource.J_EPOCH and caps == resource.NEW_CAPS,
                        "actual global J guard before each role")
                if role == "lint":
                    require(used + 575 <= effective_limit and ledger["generated_shared_remaining_s"] >= 575,
                            "entire J555 plus close20 in both pools")

            elif q["call_id"] in resource.I_QUAL_CALLS:
                role = ("lint", "math", "gate-a", "gate-b")[resource.I_QUAL_CALLS.index(q["call_id"])]
                plan = resource.i_generated_plan(role, root)
                ray.exact_value(extra, plan["argv"], "whole unchanged I19/21/32 qualification argv")
                ray.exact_value(limits, plan["limits"], "fixed I time/host/output bounds")
                require(f_dispatch is not None and f_dispatch["epoch"] == resource.I_EPOCH and caps == resource.NEW_CAPS,
                        "actual global I guard before each role")
                if role == "lint":
                    require(used + 575 <= effective_limit and ledger["generated_shared_remaining_s"] >= 575,
                            "entire I555 plus close20 in both pools")
            elif q["call_id"] in resource.G_QUAL_CALLS:
                role = ("lint", "math", "gate-a", "gate-b")[resource.G_QUAL_CALLS.index(q["call_id"])]
                plan = resource.g_generated_plan(role, root)
                ray.exact_value(extra, plan["argv"], "entire unchanged19/21/32 G qualification command")
                ray.exact_value(limits, plan["limits"], "same complete G time/host/output limits")
                require(f_dispatch is not None and f_dispatch["epoch"] == resource.G_EPOCH and caps == resource.NEW_CAPS,
                        "actual global G guard before each role")
                if role == "lint":
                    require(used + 300 <= effective_limit and ledger["generated_shared_remaining_s"] >= 300,
                            "whole G280 and separate close20 in both pools")
            elif q["call_id"] in resource.F_QUAL_CALLS:
                role = ("lint", "math", "gate-a", "gate-b")[resource.F_QUAL_CALLS.index(q["call_id"])]
                plan = resource.f_generated_plan(role, root)
                ray.exact_value(extra, plan["argv"], "entire fixed F qualification on unchanged complete E roster")
                ray.exact_value(limits, plan["limits"], "exact F host/output/clock limits")
                require(f_dispatch is not None and caps == resource.NEW_CAPS, "global F guard before every command")
                if role == "lint":
                    require(used + 300 <= effective_limit and ledger["generated_shared_remaining_s"] >= 300,
                            "full four-call280 plus close20 envelope before first F dispatch")
            elif q["call_id"] in resource.QUAL_CALLS:
                role = ("lint", "math", "gate-a", "gate-b")[resource.QUAL_CALLS.index(q["call_id"])]
                plan = resource.generated_plan(role, root)
                ray.exact_value(extra, plan["argv"], "entire fixed complete four-call qualification command")
                ray.exact_value(q["role"], plan["role"], "same actual generated qualification role")
                ray.exact_value(q["source_freeze"]["path"], resource.E_SOURCE_PATH, "same current E Source")
                for key, value in zip(("outer_s", "child_s", "cleanup_s"), plan["caps"], strict=True):
                    ray.exact_value(limits[key], value, "fixed full Generated qualification limits")
                require(caps == resource.NEW_CAPS, "allocation before qualification")
                if role == "lint":
                    require(used + 300 <= effective_limit and ledger["generated_shared_remaining_s"] >= 300,
                            "full lint15/math25/gate-A120/gate-B120/close20 envelope reserve before first dispatch")
            elif q["call_id"] == resource.BOOTSTRAP_CALL:
                ray.exact_value(extra, ["-m", "pytest", resource.BOOTSTRAP_NODE, "--tb=short", "-q",
                                       "--junitxml=" + BASE + "/parent-conserved-resource-bootstrap-junit.xml"], "one fixed complete bootstrap")
                ray.exact_value((limits["outer_s"], limits["child_s"], limits["cleanup_s"]), (11, 8, 3), "existing old-budget bootstrap envelope; fit remains unproven")
                require(resource.ALLOCATION_KEY not in ledger, "future allocation never funds its bootstrap")
                require(resource.BOOTSTRAP_DISPATCHABLE, "NOT_DISPATCHABLE: complete outer11/child8/cleanup3 bootstrap fit remains UNKNOWN; no new probe authorization")
            for item in extra[2:]:
                source_module = item.split("::", 1)[0]
                if source_module.endswith(".py"):
                    require(source_module in freeze["sources"], "reviewed exact generated Source/module/parameterized target")
            argv = [sys._base_executable, *extra]
        else:
            argv = [sys._base_executable, "-m", "tools.banc_m3_startup_ray_supervisor", "--generated-worker", behavior]
    output = ray.path(q["output"], root)
    require(not output.exists(), "fresh actual request/output, duplicate cannot dispatch")
    if f_dispatch is None:
        output.mkdir(parents=True)
    token = {"schema": SCHEMA + ".lock-owner.v1", "call_id": q["call_id"], "request": request_spec,
             "owner_pid": os.getpid(), "owner_creation_ticks": creation_ticks(os.getpid()), "entry_counter_s": entry}
    token["job_name"] = "Local\\flylab-startup-ray-" + str(token["owner_pid"]) + "-" + secrets.token_hex(16)
    lock = job = None
    readers, eof, stream_errors = [], [False, False], []
    first, secondary, timed_out = None, [], False
    begun = False
    observed_worker_exit = None
    identity = {"owner_pid": token["owner_pid"], "owner_creation_ticks": token["owner_creation_ticks"], "worker_pid": None, "worker_creation_ticks": None}
    outer = {"schema": SCHEMA + ".outer.v1", "request": request_spec, "exit_code": 1,
             "all_owned_ended": False, "reservation_released": False, "unknown_owned_identities": {},
             "first_error": None, "timeout": False, "stdout_eof": False, "stderr_eof": False}
    if k_dispatch:
        outer.update(worker_started=False, creation_attempted=False, job_created=False,
                     worker_creation_state='NO_CREATE_ATTEMPT', resume_attempted=False, resume_outcome='NOT_ATTEMPTED')
    try:
        lock_path = ray.path(SHARED_LOCK, root)
        lock = SharedLock(lock_path, token)
        token_spec = ray.spec_for(SHARED_LOCK, root)
        ray.leaf(request_spec, root)
        (resource.raw_pin if f_dispatch is not None and f_dispatch["epoch"] == resource.G_EPOCH else ray.leaf)(q["source_freeze"], root)
        if not k_dispatch:
            for source in freeze["sources"].values():
                (resource.source_text if f_dispatch is not None and f_dispatch["epoch"] == resource.G_EPOCH else ray.source_leaf)(source, root)
        if f_dispatch is not None:
            lock.authenticate(lock_path)
            fresh_raw = ray.leaf(ray.spec_for(q["ledger_path"], root, cap=524288), root, cap=524288)
            require(fresh_raw == f_dispatch_raw, "exact F admitted ledger bytes unchanged under actual common exclusion")
            resource.prequalification_dispatch(ray.json_load(fresh_raw), q, request_spec, root, generated_budget_limit, phase="admitted")
            require(not output.exists(), "F output is still unused after exact second state/Source check")
            output.mkdir(parents=True)
        if k_dispatch:
            checkpoint.source_fence(q['source_freeze'], root,
                                    lambda: require(time.perf_counter() < entry + limits['child_s'], 'same ENTRY K locked Source deadline'))
        job = OwnedJob(token["job_name"], limits["host_bytes"])
        if k_dispatch:
            outer['job_created'] = True
        environment = worker_environment(q["role"], token, sys.executable)
        deadline = entry + min(limits["child_s"], limits["outer_s"] - limits["cleanup_s"])
        if time.perf_counter() >= deadline:
            timed_out = True
            raise TimeoutError("same ENTRY pre-spawn child interval expired")
        if k_dispatch:
            outer.update(creation_attempted=True, worker_creation_state='UNCERTAIN_CREATION')
        job.spawn(argv, root, environment)
        if k_dispatch:
            outer['worker_creation_state'] = 'CREATED_SUSPENDED_NOT_RESUMED'
        identity.update(worker_pid=job.pid, worker_creation_ticks=job.identities[str(job.pid)])
        transition(root, q["ledger_path"], q, request_spec, token_spec, identity, "begin", lock=lock)
        begun = True
        def drain(index):
            try:
                total = 0
                with (output / ("stdout.bin" if index == 0 else "stderr.bin")).open("xb") as stream:
                    buffer = ct.create_string_buffer(65536)
                    while True:
                        count = wt.DWORD()
                        ok = job.k.ReadFile(job.reads[index], buffer, len(buffer), ct.byref(count), None)
                        if not ok:
                            if ct.get_last_error() == 109:
                                eof[index] = True
                                break
                            checked(ok)
                        if count.value == 0:
                            eof[index] = True
                            break
                        total += count.value
                        require(total <= (TRACKING_STREAM_CAP if tracking_role or replay_role or geometry_role or contact_role else limits["output_cap_bytes"]), "bounded opaque stream, never truncate to PASS")
                        stream.write(buffer.raw[:count.value])
                    stream.flush();os.fsync(stream.fileno())
            except Exception as error:
                stream_errors.append({"type": type(error).__name__, "message": str(error)})
        for index in (0, 1):
            thread = threading.Thread(target=drain, args=(index,), daemon=True)
            thread.start();readers.append(thread)
        if time.perf_counter() >= deadline:
            timed_out = True
            raise TimeoutError("same ENTRY pre-resume child interval expired")
        if k_dispatch:
            outer.update(resume_attempted=True, resume_outcome='UNCERTAIN', worker_creation_state='UNCERTAIN_CREATION')
        job.resume()
        if k_dispatch:
            outer.update(worker_started=True, resume_outcome='RESUMED', worker_creation_state='RESUMED')
        while job.active() != 0:
            job.observe()
            if stream_errors:
                raise RuntimeError("opaque stream failure: " + repr(stream_errors))
            if time.perf_counter() >= deadline:
                timed_out = True
                raise TimeoutError("same ENTRY child interval, owned Job termination required")
            time.sleep(.01)
        outer["exit_code"] = job.exit_code()
        if time.perf_counter() >= deadline:
            timed_out = True
            raise TimeoutError("primary completion observed after same ENTRY child deadline")
        require(type(outer["exit_code"]) is int, "genuine known primary exit")
    except Exception as error:
        first = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        if job:
            try:
                if k_dispatch and outer['worker_creation_state'] == 'UNCERTAIN_CREATION' and not outer['resume_attempted'] and str(job.pid) in job.identities:
                    # CreateProcess and creation identity were observed even if
                    # later suspended setup failed. No resume or EOF is implied.
                    identity.update(worker_pid=job.pid, worker_creation_ticks=job.identities[str(job.pid)])
                    outer['worker_creation_state'] = 'CREATED_SUSPENDED_NOT_RESUMED'
                if job.active() != 0:
                    job.terminate()
                cleanup_end = entry + limits["outer_s"] - .1
                while job.active() != 0 and time.perf_counter() < cleanup_end:
                    time.sleep(.01)
                for thread in readers:
                    thread.join(max(0., cleanup_end - time.perf_counter()))
                observed_worker_exit = job.exit_code()
                known = job.active() == 0 and observed_worker_exit is not None and all(eof) and not stream_errors
                if k_dispatch and not outer['creation_attempted']:
                    known = job.active() == 0 and not job.identities and not stream_errors
                    outer['reservation_released'] = bool(known)
                if k_dispatch and outer['worker_creation_state'] == 'UNCERTAIN_CREATION':
                    known = False
                outer["exit_code"] = job.exit_code() if job.exit_code() is not None else 1
                outer["all_owned_ended"] = bool(known)
                if not known:
                    outer["unknown_owned_identities"] = dict(job.identities)
                if begun:
                    transition(root, q["ledger_path"], q, request_spec, token_spec, identity, "closed" if known else "unknown", lock=lock)
                    outer["reservation_released"] = bool(known)
                if lock and known:
                    lock.release(True);lock = None
            except Exception as error:
                captured = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
                if first is None:
                    first = captured
                else:
                    secondary.append(captured)
                outer["all_owned_ended"] = False
                outer["reservation_released"] = False
                outer["unknown_owned_identities"] = dict(job.identities)
            if all(not thread.is_alive() for thread in readers):
                job.close()
        elif lock:
            # No CreateProcess was attempted, so this owner's exclusion may be released.
            lock.release(True);lock = None
            outer["all_owned_ended"] = True
            if k_dispatch:
                outer['reservation_released'] = True
        if k_dispatch and not output.exists():
            output.mkdir(parents=True)
        outer["worker_exit_code"] = (observed_worker_exit if identity['worker_pid'] is not None else None) if k_dispatch else outer["exit_code"]
        outer["exit_code"] = 124 if timed_out else (1 if first or not outer["all_owned_ended"] else outer["exit_code"])
        outer.update(identity)
        outer.update(first_error=first, secondary_errors=secondary, timeout=timed_out, stdout_eof=eof[0], stderr_eof=eof[1])
        for index, label in enumerate(("stdout", "stderr")):
            p = output / (label + ".bin")
            if p.exists() and eof[index]:
                raw = p.read_bytes()
                outer[label] = {"path": str(p.relative_to(root)).replace("\\", "/"), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            else:
                outer[label] = None
        # This is an internal observation, not the outer tool's actual full-return duration/cost.
        outer["observed_supervisor_interval_s"] = time.perf_counter() - entry
        if tracking_role or replay_role or geometry_role or contact_role:
            # Primary errors remain failures; bounded metadata may summarize long tracebacks.
            for record in [outer.get("first_error"), *outer.get("secondary_errors", [])]:
                if record and len(record.get("traceback", "")) > 8192:
                    record["traceback_original_chars"] = len(record["traceback"])
                    record["traceback"] = record["traceback"][:8192]
                    record["traceback_truncated"] = True
            if not k_dispatch:
                outer["worker_started"], outer["job_created"] = bool(job and getattr(job, "pid", None)), bool(job)
            outer["pre_spawn_refused"] = False
            outer["output_usage"] = {"logical_combined_cap_bytes": limits["output_cap_bytes"]}
            publish_tracking_outer(output, outer, root, ray.path(science.request["output"], root))
        else:
            durable(output / "outer.json", outer)
    return outer



def g_initial_guard_refusal(root, request_spec, q, ledger_spec, admitted_raw, error, *, entry):
    """Actual supervisor observation only at the lexical initial G guard failure."""
    from tools import banc_m3_conserved_resource as resource
    from tools import verify_c_banc_m3_startup_ray as ray
    require(type(admitted_raw) is bytes and len(admitted_raw) <= 524288, "actual saved admitted G raw bytes")
    resource.exact(ledger_spec, {"path": q["ledger_path"], "bytes": len(admitted_raw),
                               "sha256": hashlib.sha256(admitted_raw).hexdigest()}, "actual initial ledger identity")
    resource.exact(resource.raw_pin(ledger_spec, root, 524288), admitted_raw, "actual unchanged initial G raw ledger")
    resource.exact(resource.read_pin(request_spec, root), q, "actual admitted request before refusal publication")
    ledger = ray.json_load(admitted_raw)
    pointer = resource.g_admitted_identity(ledger, q, request_spec, root)
    try:
        resource._text_path(root, SHARED_LOCK)
    except FileNotFoundError:
        pass  # Every parent was lstat checked; no actual SharedLock attempt occurred.
    else:
        raise ValueError("foreign exclusion prevents initial G refusal publication")
    message = str(error)
    outer = {"schema": SCHEMA + ".outer.v1", "request": request_spec, "source_freeze": q["source_freeze"],
             "qualification_amendment": pointer, "initial_ledger": ledger_spec,
             "exit_code": 1, "worker_exit_code": None, "worker_pid": None, "worker_creation_ticks": None,
             "owner_pid": os.getpid(), "owner_creation_ticks": creation_ticks(os.getpid()),
             "all_owned_ended": True, "reservation_released": True, "unknown_owned_identities": {},
             "first_error": {"type": type(error).__name__, "message": message[:8192],
                             "original_message_chars": len(message), "message_truncated": len(message) > 8192},
             "secondary_errors": [], "timeout": isinstance(error, TimeoutError),
             "pre_spawn_refused": True, "job_created": False, "worker_started": False,
             "stdout_eof": True, "stderr_eof": True, "stdout": None, "stderr": None,
             "stream_state": "NO_WORKER_STREAMS_OPENED", "runtime_reservation_created": False,
             "runtime_reservation_state": "NEVER_CREATED", "root_admission_reservation_state": "STILL_PRESENT_PENDING_NATIVE_COMMIT",
             "closure_flag_meaning": "No owned Job/process/streams were ever created; no worker EOF reads or runtime cleanup claimed",
             "ledger_mutated": False, "charged": False, "qualified": False, "runtime_worker_pass": False,
             "original_payload_accesses": 0, "refusal_origin": "ACTUAL_G_INITIAL_GUARD_BEFORE_LOCK_JOB_ATTEMPT",
             "observed_supervisor_interval_s": time.perf_counter() - entry}
    raw = json.dumps(outer, ensure_ascii=True, allow_nan=False, sort_keys=True).encode("utf8")
    require(len(raw) <= TRACKING_METADATA_CAP and len(raw) <= q["limits"]["output_cap_bytes"], "bounded actual G refusal metadata")
    output = resource.text_new_directory(root, q["output"])
    durable(output / "outer.json", outer)  # CreateNew/fsync/full readback; no streams.
    return outer
def h_initial_guard_refusal(root, request_spec, q, ledger_spec, admitted_raw, error, *, entry):
    """Actual lexical H first-guard failure; no lock/Job/stream was attempted."""
    from tools import banc_m3_conserved_resource as resource
    from tools import verify_c_banc_m3_startup_ray as ray

    require(type(admitted_raw) is bytes and len(admitted_raw) <= 524288, "actual saved admitted H raw")
    resource.exact(ledger_spec, {"path": q["ledger_path"], "bytes": len(admitted_raw),
                               "sha256": hashlib.sha256(admitted_raw).hexdigest()}, "actual saved H raw identity")
    resource.exact(resource.raw_pin(ledger_spec, root, 524288), admitted_raw, "unchanged exact admitted H raw")
    resource.exact(resource.read_pin(request_spec, root), q, "exact actual admitted H request")
    ledger = ray.json_load(admitted_raw)
    pointer = resource.h_admitted_identity(ledger, q, request_spec, root)
    try:
        resource._text_path(root, SHARED_LOCK)
    except FileNotFoundError:
        pass  # lstat-checked parents; no lock attempt was made.
    else:
        raise ValueError("foreign exclusion prevents initial H refusal publication")
    message = str(error)
    outer = {"schema": SCHEMA + ".outer.v1", "request": request_spec, "source_freeze": q["source_freeze"],
             "qualification_amendment": ledger[resource.G_KEY], "qualification_lint_repair": pointer, "initial_ledger": ledger_spec,
             "exit_code": 1, "worker_exit_code": None, "worker_pid": None, "worker_creation_ticks": None,
             "owner_pid": os.getpid(), "owner_creation_ticks": creation_ticks(os.getpid()),
             "all_owned_ended": True, "reservation_released": True, "unknown_owned_identities": {},
             "first_error": {"type": type(error).__name__, "message": message[:8192],
                             "original_message_chars": len(message), "message_truncated": len(message) > 8192},
             "secondary_errors": [], "timeout": isinstance(error, TimeoutError),
             "pre_spawn_refused": True, "job_created": False, "worker_started": False,
             "stdout_eof": True, "stderr_eof": True, "stdout": None, "stderr": None,
             "stdout_eof_observed": False, "stderr_eof_observed": False,
             "stream_state": "NO_WORKER_STREAMS_OPENED", "runtime_reservation_created": False,
             "runtime_reservation_state": "NEVER_CREATED", "root_admission_reservation_state": "STILL_PRESENT_PENDING_NATIVE_COMMIT",
             "closure_flag_meaning": "No owned Job/process/streams were ever created; EOF flags are vacuous protocol closure, no EOF read or runtime cleanup observed",
             "ledger_mutated": False, "charged": False, "qualified": False, "runtime_worker_pass": False,
             "original_payload_accesses": 0, "refusal_origin": "ACTUAL_H_INITIAL_GUARD_BEFORE_LOCK_JOB_ATTEMPT",
             "observed_supervisor_interval_s": time.perf_counter() - entry}
    raw = json.dumps(outer, ensure_ascii=True, allow_nan=False, sort_keys=True).encode("utf8")
    require(len(raw) <= TRACKING_METADATA_CAP and len(raw) <= q["limits"]["output_cap_bytes"], "bounded actual H refusal metadata")
    # Recheck the saved admitted bytes and exclusion immediately before publication.
    resource.exact(resource.raw_pin(ledger_spec, root, 524288), admitted_raw, "saved H admitted raw remains unchanged")
    try:
        resource._text_path(root, SHARED_LOCK)
    except FileNotFoundError:
        pass
    else:
        raise ValueError("foreign exclusion appeared before initial H publication")
    output = resource.text_new_directory(root, q["output"])
    durable(output / "outer.json", outer)
    return outer


def i_initial_guard_refusal(root, request_spec, q, ledger_spec, admitted_raw, error, *, entry):
    """Actual lexical I first-guard failure; no lock/Job/stream was attempted."""
    from tools import banc_m3_conserved_resource as resource
    from tools import verify_c_banc_m3_startup_ray as ray

    require(type(admitted_raw) is bytes and len(admitted_raw) <= 524288, "actual saved admitted I raw")
    resource.exact(ledger_spec, {"path": q["ledger_path"], "bytes": len(admitted_raw),
                               "sha256": hashlib.sha256(admitted_raw).hexdigest()}, "actual saved I raw identity")
    resource.exact(resource.raw_pin(ledger_spec, root, 524288), admitted_raw, "unchanged exact admitted I raw")
    resource.exact(resource.read_pin(request_spec, root), q, "exact actual admitted I request")
    ledger = ray.json_load(admitted_raw)
    pointer = resource.i_admitted_identity(ledger, q, request_spec, root)
    try:
        resource._text_path(root, SHARED_LOCK)
    except FileNotFoundError:
        pass  # lstat-checked parents; no lock attempt was made.
    else:
        raise ValueError("foreign exclusion prevents initial I refusal publication")
    message = str(error)
    outer = {"schema": SCHEMA + ".outer.v1", "request": request_spec, "source_freeze": q["source_freeze"],
             "qualification_amendment": ledger[resource.G_KEY], "qualification_lint_repair": ledger[resource.H_KEY], "preflight_recovery": pointer, "initial_ledger": ledger_spec,
             "exit_code": 1, "worker_exit_code": None, "worker_pid": None, "worker_creation_ticks": None,
             "owner_pid": os.getpid(), "owner_creation_ticks": creation_ticks(os.getpid()),
             "all_owned_ended": True, "reservation_released": True, "unknown_owned_identities": {},
             "first_error": {"type": type(error).__name__, "message": message[:8192],
                             "original_message_chars": len(message), "message_truncated": len(message) > 8192},
             "secondary_errors": [], "timeout": isinstance(error, TimeoutError),
             "pre_spawn_refused": True, "job_created": False, "worker_started": False,
             "stdout_eof": True, "stderr_eof": True, "stdout": None, "stderr": None,
             "stdout_eof_observed": False, "stderr_eof_observed": False,
             "stream_state": "NO_WORKER_STREAMS_OPENED", "runtime_reservation_created": False,
             "runtime_reservation_state": "NEVER_CREATED", "root_admission_reservation_state": "STILL_PRESENT_PENDING_NATIVE_COMMIT",
             "closure_flag_meaning": "No owned Job/process/streams were ever created; EOF flags are vacuous protocol closure, no EOF read or runtime cleanup observed",
             "ledger_mutated": False, "charged": False, "qualified": False, "runtime_worker_pass": False,
             "original_payload_accesses": 0, "refusal_origin": "ACTUAL_I_INITIAL_GUARD_BEFORE_LOCK_JOB_ATTEMPT",
             "observed_supervisor_interval_s": time.perf_counter() - entry}
    raw = json.dumps(outer, ensure_ascii=True, allow_nan=False, sort_keys=True).encode("utf8")
    require(len(raw) <= TRACKING_METADATA_CAP and len(raw) <= q["limits"]["output_cap_bytes"], "bounded actual I refusal metadata")
    # Recheck the saved admitted bytes and exclusion immediately before publication.
    resource.exact(resource.raw_pin(ledger_spec, root, 524288), admitted_raw, "saved I admitted raw remains unchanged")
    try:
        resource._text_path(root, SHARED_LOCK)
    except FileNotFoundError:
        pass
    else:
        raise ValueError("foreign exclusion appeared before initial I publication")
    output = resource.text_new_directory(root, q["output"])
    durable(output / "outer.json", outer)
    return outer


def j_initial_guard_refusal(root, request_spec, q, ledger_spec, admitted_raw, error, *, entry):
    """Actual lexical J first-guard failure; no lock/Job/stream was attempted."""
    from tools import banc_m3_conserved_resource as resource
    from tools import verify_c_banc_m3_startup_ray as ray

    require(type(admitted_raw) is bytes and len(admitted_raw) <= 524288, "actual saved admitted J raw")
    resource.exact(ledger_spec, {"path": q["ledger_path"], "bytes": len(admitted_raw),
                               "sha256": hashlib.sha256(admitted_raw).hexdigest()}, "actual saved J raw identity")
    resource.exact(resource.raw_pin(ledger_spec, root, 524288), admitted_raw, "unchanged exact admitted J raw")
    resource.exact(resource.read_pin(request_spec, root), q, "exact actual admitted J request")
    ledger = ray.json_load(admitted_raw)
    pointer = resource.j_admitted_identity(ledger, q, request_spec, root)
    try:
        resource._text_path(root, SHARED_LOCK)
    except FileNotFoundError:
        pass  # lstat-checked parents; no lock attempt was made.
    else:
        raise ValueError("foreign exclusion prevents initial J refusal publication")
    message = str(error)
    outer = {"schema": SCHEMA + ".outer.v1", "request": request_spec, "source_freeze": q["source_freeze"],
             "qualification_amendment": ledger[resource.G_KEY], "qualification_lint_repair": ledger[resource.H_KEY], "preflight_recovery": ledger[resource.I_KEY], "qualification_lint_cleanup": pointer, "initial_ledger": ledger_spec,
             "exit_code": 1, "worker_exit_code": None, "worker_pid": None, "worker_creation_ticks": None,
             "owner_pid": os.getpid(), "owner_creation_ticks": creation_ticks(os.getpid()),
             "all_owned_ended": True, "reservation_released": True, "unknown_owned_identities": {},
             "first_error": {"type": type(error).__name__, "message": message[:8192],
                             "original_message_chars": len(message), "message_truncated": len(message) > 8192},
             "secondary_errors": [], "timeout": isinstance(error, TimeoutError),
             "pre_spawn_refused": True, "job_created": False, "worker_started": False,
             "stdout_eof": True, "stderr_eof": True, "stdout": None, "stderr": None,
             "stdout_eof_observed": False, "stderr_eof_observed": False,
             "stream_state": "NO_WORKER_STREAMS_OPENED", "runtime_reservation_created": False,
             "runtime_reservation_state": "NEVER_CREATED", "root_admission_reservation_state": "STILL_PRESENT_PENDING_NATIVE_COMMIT",
             "closure_flag_meaning": "No owned Job/process/streams were ever created; EOF flags are vacuous protocol closure, no EOF read or runtime cleanup observed",
             "ledger_mutated": False, "charged": False, "qualified": False, "runtime_worker_pass": False,
             "original_payload_accesses": 0, "refusal_origin": "ACTUAL_J_INITIAL_GUARD_BEFORE_LOCK_JOB_ATTEMPT",
             "observed_supervisor_interval_s": time.perf_counter() - entry}
    raw = json.dumps(outer, ensure_ascii=True, allow_nan=False, sort_keys=True).encode("utf8")
    require(len(raw) <= TRACKING_METADATA_CAP and len(raw) <= q["limits"]["output_cap_bytes"], "bounded actual J refusal metadata")
    # Recheck the saved admitted bytes and exclusion immediately before publication.
    resource.exact(resource.raw_pin(ledger_spec, root, 524288), admitted_raw, "saved J admitted raw remains unchanged")
    try:
        resource._text_path(root, SHARED_LOCK)
    except FileNotFoundError:
        pass
    else:
        raise ValueError("foreign exclusion appeared before initial J publication")
    resource.exact(resource.read_pin(request_spec, root), q, "saved J request remains unchanged")
    resource.j_admitted_identity(ray.json_load(resource.raw_pin(ledger_spec, root, 524288)), q, request_spec, root)
    output = resource.text_new_directory(root, q["output"])
    durable(output / "outer.json", outer)
    return outer



def bootstrap_pre_spawn_refusal(root, request_spec, q, error, *, entry):
    """Record an actual admitted bootstrap failure before any Job/output attempt.

    Existing output or exclusion prevents this narrow proof. Such an outcome
    remains unresolved for ROOT; absence alone never establishes process closure.
    """
    from tools import banc_m3_conserved_resource as resource
    from tools import verify_c_banc_m3_startup_ray as ray

    root = root.resolve()
    resource.exact((q["call_id"], q["mode"], q["role"], q["generated_behavior"]),
                   (resource.BOOTSTRAP_CALL, "generated", "execution", "command"), "fixed actual bootstrap refusal")
    resource.exact(tuple(q["limits"][key] for key in ("outer_s", "child_s", "cleanup_s")),
                   (11, 8, 3), "same actual old-budget refusal limits")
    ledger_name = BASE + "/parent-e358-execution-ledger.json"
    resource.exact(q["ledger_path"], ledger_name, "same actual ledger")
    resource.exact(q["shared_original_lock_path"], SHARED_LOCK, "same physical exclusion")
    resource.exact(q["output"], BASE + "/parent-" + resource.BOOTSTRAP_CALL, "fixed fresh bootstrap output")
    resource.exact(q["limits"]["output_cap_bytes"], 8 << 20, "fixed complete bounded output")
    live = ray.json_load(ray.leaf(ray.spec_for(ledger_name, root, cap=524288), root, cap=524288))
    resource.exact(live["direct_science_calls"][q["call_id"]],
                   {"request": request_spec, "state": "admitted", "mode": "generated", "reserved_s": 11},
                   "real unchanged Native admission before any worker")
    resource.exact(live["direct_science_reservations"],
                   {q["call_id"]: {"request": request_spec, "reserved_s": 11}}, "sole real reservation")
    for key in ("direct_science_workers", "direct_science_pending_actuals", "direct_science_blocking_holds",
                "startup_ray_original_workers", "startup_ray_original_reservations"):
        resource.exact(live[key], {}, "no worker/unknown transition before refusal")
    output = ray.path(q["output"], root)
    require(not output.exists() and not ray.path(SHARED_LOCK, root).exists(), "before actual output/lock/Job attempt")
    output.mkdir(parents=True)
    streams = {}
    for label in ("stdout", "stderr"):
        target = output/(label + ".bin")
        with target.open("xb") as stream:
            stream.flush();os.fsync(stream.fileno())
        require(target.read_bytes() == b"", "actual empty unstarted worker stream")
        streams[label] = {"path": target.relative_to(root).as_posix(), "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    message = str(error)
    outer = {"schema": SCHEMA + ".outer.v1", "request": request_spec, "exit_code": 1, "worker_exit_code": None,
             "all_owned_ended": True, "reservation_released": True, "unknown_owned_identities": {},
             "first_error": {"type": type(error).__name__, "message": message[:8192],
                             "original_message_chars": len(message), "message_truncated": len(message) > 8192},
             "secondary_errors": [], "timeout": isinstance(error, TimeoutError), "stdout_eof": True, "stderr_eof": True,
             "stdout": streams["stdout"], "stderr": streams["stderr"], "pre_spawn_refused": True,
             "worker_started": False, "job_created": False, "worker_pid": None, "worker_creation_ticks": None,
             "owner_pid": os.getpid(), "owner_creation_ticks": creation_ticks(os.getpid()),
             "original_payload_accesses": 0, "refusal_origin": "ACTUAL_FIXED_BOOTSTRAP_BEFORE_JOB_ATTEMPT",
             "observed_supervisor_interval_s": time.perf_counter() - entry,
             "output_usage": {"logical_combined_cap_bytes": q["limits"]["output_cap_bytes"]}}
    publish_tracking_outer(output, outer, root, None)
    return outer


def supervise(name, *, root=ROOT, entry=ENTRY):
    try:
        return _supervise(name, root=root, entry=entry)
    except Exception as error:
        from tools import verify_c_banc_m3_startup_ray as ray
        try:
            request_spec = ray.spec_for(name, root)
            q = ray.json_load(ray.leaf(request_spec, root))
            if q.get("call_id") != "direct-science-20261008-conserved-bootstrap":
                raise
            return bootstrap_pre_spawn_refusal(root, request_spec, q, error, entry=entry)
        except Exception as refusal_error:
            if refusal_error is not error:
                error.add_note("Bootstrap initial-refusal publication failed; preserve actual return PENDING/HOLD: " + str(refusal_error))
            raise error


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--request")
    parser.add_argument("--generated-worker", choices=("normal", "nonzero", "timeout-descendant"))
    args = parser.parse_args(argv)
    if args.generated_worker:
        require("FLYLAB_DIRECT_SCIENCE_OWNER" in os.environ, "only actual owned generated fixture")
        print(json.dumps({"actual_worker_pid": os.getpid(), "python_prefix": sys.prefix}), flush=True)
        if args.generated_worker == "timeout-descendant":
            # Keep both descendant and primary alive beyond the admitted short fixture deadline.
            child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"], creationflags=0x08000000)
            print(json.dumps({"descendant_pid": child.pid}), flush=True)
            # Generated-only opaque fixture: existing actual Job kills an unfinished
            # operation after its short test bound; this never constructs a body.
            from tools.banc_m3_contact_runtime_journal import RuntimeJournal
            token = json.loads(os.environ["FLYLAB_DIRECT_SCIENCE_OWNER"])
            from tools import verify_c_banc_m3_startup_ray as ray
            request = ray.json_load(ray.leaf(token["request"], ROOT))
            journal = RuntimeJournal(float(token["entry_counter_s"]), token, role="process")
            journal.bind(request, token["request"])
            with journal.phase("construct", opaque=True):
                time.sleep(60)
        print("generated stdout", flush=True)
        print("generated stderr", file=sys.stderr, flush=True)
        return 7 if args.generated_worker == "nonzero" else 0
    require(args.request, "exact admitted supervisor request")
    outer = supervise(args.request)
    print(json.dumps(outer, allow_nan=False, ensure_ascii=True), flush=True)
    return outer["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
