"""E188 resource-only readiness and monitor for an empty or stopped queue service."""

# Broad catches preserve raw resource failures and fail the readiness check.
# ruff: noqa: BLE001

from __future__ import annotations

import csv
import errno
import json
import math
import os
import re
import socket
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

EMPTY_QUEUE = {"running": 0, "pending": 0}
PORT = 8188


def _gpu_telemetry(row: dict[str, Any]) -> list[dict[str, Any]]:
    if row.get("gpu_error") or not isinstance(row.get("gpu_csv"), list):
        raise ValueError("GPU sample missing or failed")
    lines = row["gpu_csv"]
    if not lines:
        raise ValueError("GPU list empty")
    result = []
    seen = set()
    for line in lines:
        fields = next(csv.reader([line], skipinitialspace=True))
        if len(fields) != 8:
            raise ValueError("GPU CSV field count")
        stamp, index, util, used, total, power, graphics, memory = (
            field.strip() for field in fields
        )
        # nvidia-smi's query-gpu timestamp has this format on the E182 path.
        datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S.%f")  # noqa: DTZ007 -- format validation only
        if not re.fullmatch(r"\d+", index):
            raise ValueError("GPU index")
        number = int(index)
        if number in seen:
            raise ValueError("duplicate GPU index")
        seen.add(number)
        values = [float(value) for value in (util, used, total, power, graphics, memory)]
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("GPU numeric telemetry")
        usage, used_mib, total_mib, watts, graphics_mhz, memory_mhz = values
        if usage > 100 or total_mib <= 0 or used_mib > total_mib:
            raise ValueError("GPU range")
        result.append(
            {
                "timestamp": stamp,
                "index": number,
                "utilization_pct": usage,
                "memory_used_mib": used_mib,
                "memory_total_mib": total_mib,
                "power_w": watts,
                "graphics_mhz": graphics_mhz,
                "memory_mhz": memory_mhz,
            }
        )
    return result


def _probe_service() -> dict[str, Any]:
    proof: dict[str, Any] = {
        "current_pid": os.getpid(),
        "tcp_connect": "error",
        "listener_inventory_complete": False,
        "listener_8188": None,
        "listeners_8188": [],
        "process_inventory_complete": False,
        "allowed_process_family": [],
        "ancestor_terminal": None,
        "python_processes": [],
        "foreign_python": None,
    }
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=3):
            proof["tcp_connect"] = "connected"
    except OSError as error:
        proof["tcp_connect_error"] = repr(error)
        proof["tcp_connect_errno"] = error.errno
        proof["tcp_connect_winerror"] = getattr(error, "winerror", None)
        if isinstance(error, TimeoutError) or error.errno in (errno.ETIMEDOUT, 10060):
            proof["tcp_connect"] = "timeout"
        elif error.errno in (errno.ECONNREFUSED, 10061) or getattr(
            error, "winerror", None
        ) == 10061:
            proof["tcp_connect"] = "refused"
    try:
        connections = psutil.net_connections(kind="tcp")
        listeners = []
        for connection in connections:
            if connection.status != psutil.CONN_LISTEN or not connection.laddr:
                continue
            if connection.laddr.port == PORT:
                listeners.append(
                    {
                        "pid": connection.pid,
                        "address": connection.laddr.ip,
                        "port": connection.laddr.port,
                        "status": connection.status,
                    }
                )
        proof["listeners_8188"] = listeners
        proof["listener_8188"] = bool(listeners)
        proof["listener_inventory_complete"] = True
    except (psutil.Error, OSError, ValueError) as error:
        proof["port_error"] = repr(error)
    try:
        records = {}
        processes = {}
        for pid in psutil.pids():
            process = psutil.Process(pid)
            processes[pid] = process
            records[pid] = {"pid": pid, "name": process.name()}

        def identity(pid: int) -> dict[str, Any]:
            record = records[pid]
            if "ppid" not in record:
                record["ppid"] = processes[pid].ppid()
                record["create_time"] = processes[pid].create_time()
            return record

        current = proof["current_pid"]
        if current not in records:
            raise ValueError("current process absent from inventory")
        family: dict[int, str] = {current: "self"}
        parent = identity(current)["ppid"]
        child = current
        while parent:
            if parent in family:
                raise ValueError("process ancestry cycle")
            if parent not in records:
                if psutil.pid_exists(parent):
                    raise ValueError("ancestor missing from inventory but live")
                proof["ancestor_terminal"] = {
                    "reason": "absent_parent",
                    "pid": parent,
                    "child_pid": child,
                    "verified_absent": True,
                }
                break
            if identity(parent)["create_time"] > identity(child)["create_time"]:
                raise ValueError("ancestor created after child")
            family[parent] = "ancestor"
            child = parent
            parent = records[parent]["ppid"]
        if not parent:
            proof["ancestor_terminal"] = {"reason": "root", "pid": 0}
        descendant_pids = {p.pid for p in processes[current].children(recursive=True)}
        if not descendant_pids <= set(records):
            raise ValueError("descendant changed during process inventory")
        for pid in descendant_pids:
            identity(pid)
        descendants = {current}
        changed = True
        while changed:
            changed = False
            for pid in descendant_pids:
                record = records[pid]
                if pid not in family and record["ppid"] in descendants:
                    if records[record["ppid"]]["create_time"] > record["create_time"]:
                        raise ValueError("descendant created before parent")
                    family[pid] = "descendant"
                    descendants.add(pid)
                    changed = True
        if descendants - {current} != descendant_pids:
            raise ValueError("descendant topology changed during inventory")
        proof["allowed_process_family"] = [
            {
                "pid": pid,
                "ppid": records[pid]["ppid"],
                "create_time": records[pid]["create_time"],
                "relation": relation,
            }
            for pid, relation in sorted(family.items())
        ]
        python = [
            dict(identity(pid))
            for pid, record in sorted(records.items())
            if re.match(r"^(python|pypy)", record["name"], re.IGNORECASE)
        ]
        proof["python_processes"] = python
        proof["foreign_python"] = [
            process for process in python if process["pid"] not in family
        ]
        proof["process_inventory_complete"] = True
    except (psutil.Error, OSError, ValueError) as error:
        proof["process_error"] = repr(error)
    return proof


def _sample(e182: Any, phase: str) -> dict[str, Any]:
    try:
        row = e182._sample_resources()
    except Exception as error:
        row = {"sample_error": repr(error)}
    row["phase"] = phase
    try:
        row["clients"] = e182._client_status()
    except Exception as error:
        row["client_error"] = repr(error)
    try:
        row["gpu_telemetry"] = _gpu_telemetry(row)
    except (TypeError, ValueError, OverflowError) as error:
        row["gpu_telemetry_error"] = repr(error)
    if row.get("queue_error"):
        row["service_probe"] = _probe_service()
    queue = row.get("queue") if "queue_error" not in row else None
    clients = row.get("clients")
    expected_clients = {str(pid) for pid in e182.CLIENTS}
    clients_clear = isinstance(clients, dict) and set(clients) == expected_clients and all(
        state in ("absent", "reused") for state in clients.values()
    ) and not row.get("client_error")
    gpu_valid = "gpu_telemetry" in row and not row.get("gpu_telemetry_error")
    proof = row.get("service_probe", {})
    stopped = bool(
        row.get("queue_error")
        and proof.get("tcp_connect") == "refused"
        and proof.get("listener_inventory_complete") is True
        and proof.get("listener_8188") is False
        and proof.get("process_inventory_complete") is True
        and proof.get("foreign_python") == []
    )
    idle = bool(
        gpu_valid
        and all(
            gpu["utilization_pct"] <= 5 and gpu["memory_used_mib"] <= 2048
            for gpu in row["gpu_telemetry"]
        )
    )
    if queue == EMPTY_QUEUE and clients_clear and gpu_valid:
        row["resource_mode"] = "queue_empty"
        row["resource_clear"] = True
    elif stopped and clients_clear and gpu_valid and (phase == "monitor" or idle):
        row["resource_mode"] = "service_stopped"
        row["resource_clear"] = True
    else:
        row["resource_mode"] = "blocked"
        row["resource_clear"] = False
    row["ready"] = row["resource_clear"] if phase == "readiness" else False
    return row


def readiness(e182: Any) -> list[dict[str, Any]]:
    first = _sample(e182, "readiness")
    time.sleep(2)
    return [first, _sample(e182, "readiness")]


def monitor(path: Path, stop: threading.Event, errors: list[str], e182: Any) -> None:
    try:
        with path.open("x", encoding="utf-8") as stream:
            while not stop.is_set():
                stream.write(json.dumps(_sample(e182, "monitor"), sort_keys=True) + "\n")
                stream.flush()
                if stop.wait(2):
                    break
    except Exception as error:
        errors.append(repr(error))


def resource_clear(row: dict[str, Any]) -> bool:
    return bool(
        isinstance(row, dict)
        and row.get("phase") == "monitor"
        and row.get("resource_mode") in ("queue_empty", "service_stopped")
        and row.get("resource_clear") is True
    )
