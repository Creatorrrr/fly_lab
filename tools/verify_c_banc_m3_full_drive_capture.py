"""E318: one bounded, observational SSP/u0 full-drive body replay."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = "verification/banc-natural-gait-m3-20260919"
SCHEMA = "flylab.m3.e318.full-drive-capture.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e318.full-drive-capture.review.v1"
RESULT_SCHEMA = "flylab.m3.e318.full-drive-capture.result.v1"
OUTPUT = BASE + "/e318-full-drive-capture"
NETWORK_ID = "f9ea25308cf5e074047bf16afba55de9fac89411ca48127f2c37cc4e6e34c7be"
ADAPTER_ID = "1fd7b86939a6846e01c4837fb5c2fbe24ce8c776e3331c1f75921a376c53b067"
GRAPH_HASH = "9bd8468f5e5e40c3e6daaffde9f444466825ba8e69189f1e347568bff85920ff"
N, MOTOR, CONTROLS, CHUNK = 158706, 391, 300, 25
LIMITS = {"supervisor_wall_s": 150, "internal_wall_s": 135, "science_wall_s": 105,
          "finalization_reserve_s": 30, "body_constructors": 1, "neural_ticks": 6000,
          "physics_steps": 15000, "host_bytes": 6 << 30, "additional_gpu_bytes": 4 << 30,
          "output_bytes": 2 << 30, "minimum_free_disk_bytes": 4 << 30}
OWNED = ("tools/verify_c_banc_m3_full_drive_capture.py", "tests/test_c_banc_full_drive_capture.py")
REUSED = (
    "tools/verify_c_banc_m3_resource_continuation.py",
    "tools/verify_c_banc_m3_recovery_resource_body.py",
    "tools/verify_c_banc_m3_recurrent_resource_body.py",
    "tools/verify_c_banc_m3_ssp_first_control.py",
    "tools/verify_c_banc_m3_native_timescale_transfer.py",
    "tools/verify_c_banc_m3_source_native_transfer.py",
    "tools/banc_m3_ssp_recovery_synapses.py",
    "tools/banc_m3_ssp_recovery_trial.py",
    "flylab/c/banc_walking.py", "flylab/c/rate_body.py", "flylab/c/research_rate.py",
    "flylab/c/walking_trace.py", "flylab/c/storage.py",
)
PATHS = {
    "graph_bundle": "data/acquisitions/banc888-windows-20260910/bundle",
    "roster": BASE + "/e289-recurrent-resource-roster.npz",
    "start_whole": BASE + "/e307-ssp-neutral-body/neutral/observed-start",
    "e307_final": BASE + "/e307-ssp-neutral-body/neutral/final",
    "e310_final": BASE + "/e310-resource-continuation/neutral/final",
    "e307_raw": BASE + "/e307-ssp-neutral-body/neutral",
    "e310_combined": BASE + "/e310-resource-continuation/neutral/combined",
    "e307_freeze": BASE + "/e307-ssp-neutral-body-freeze.json",
    "e310_freeze": BASE + "/e310-resource-continuation-freeze.json",
    "e307_review": BASE + "/e307-ssp-neutral-body-source-review.json",
    "e310_review": BASE + "/e310-resource-continuation-source-review.json",
    "e307_result": BASE + "/e307-ssp-neutral-body/result.json",
    "e310_result": BASE + "/e310-resource-continuation/result.json",
    "e307_source_manifest": BASE + "/e307-ssp-neutral-body/source-manifest.json",
    "e310_source_manifest": BASE + "/e310-resource-continuation/source-manifest.json",
    "e307_output_manifest": BASE + "/e307-ssp-neutral-body/output-manifest.json",
    "e310_output_manifest": BASE + "/e310-resource-continuation/output-manifest.json",
    "e307_audit": BASE + "/parent-e307-original-raw-audit.json",
    "e310_audit": BASE + "/parent-e310-original-raw-audit.json",
}
RAW = ("trace.json", "native-contact.npz", "activity.npz", "control-raw.npz", "native-recruitment-state.npz")


def source_roster(root: Path = ROOT) -> tuple[str, ...]:
    """Use the audited E310 code roster, without importing any producer module."""
    manifest = json.loads((root / PATHS["e310_source_manifest"]).read_text("utf8"))
    source = manifest.get("source_sha256")
    if not isinstance(source, dict) or manifest.get("schema") != "flylab.m3.e310.sources.v1":
        raise ValueError("Audited E310 source roster required")
    closure = {name for name in source if name.startswith(("flylab/", "tools/")) and name.endswith(".py")}
    if not closure or not set(REUSED).issubset(closure):
        raise ValueError("Incomplete E310 implementation closure")
    return tuple(sorted(set(OWNED) | closure))


SOURCES = source_roster()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def safe(name: str, root: Path = ROOT) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Repo-relative path required")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Input escapes repository")
    return path


def put_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n", encoding="utf8")


def replace_owned(pending: Path, final: Path, on_retry=None) -> None:
    """Retry only transient Windows file sharing, never scientific work."""
    for attempt in range(4):
        try:
            pending.replace(final)
            return
        except PermissionError:
            if attempt == 3:
                raise
            if on_retry is not None:
                on_retry()
            time.sleep(.01 * (2 ** attempt))


def usage() -> dict:
    import cupy
    import psutil
    import torch

    memory = psutil.Process().memory_info()
    free, total = cupy.cuda.runtime.memGetInfo()
    report = {"host_rss_bytes": memory.rss, "host_peak_bytes": getattr(memory, "peak_wset", memory.rss),
              "device_free_bytes": free, "device_total_bytes": total,
              "gpu_reserved_bytes": 0, "gpu_peak_reserved_bytes": 0}
    if torch.cuda.is_available():
        report["gpu_reserved_bytes"] = torch.cuda.memory_reserved()
        report["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    return report


def gate(freeze_path: Path, freeze_sha256: str, review_path: Path, review_sha256: str,
         output: Path, *, root: Path = ROOT) -> dict:
    """Passive hash/review gate; imports no model, graph, storage, body, or CUDA."""
    if output.exists() or output.resolve() != safe(OUTPUT, root):
        raise ValueError("Fresh literal E318 output required")
    if sha256(freeze_path) != freeze_sha256 or sha256(review_path) != review_sha256:
        raise ValueError("Frozen E318 freeze/review hash differs")
    freeze = json.loads(freeze_path.read_text("utf8"))
    review = json.loads(review_path.read_text("utf8"))
    if (freeze.get("schema") != SCHEMA or freeze.get("paths") != PATHS
            or freeze.get("limits") != LIMITS or freeze.get("output") != OUTPUT
            or freeze.get("network_identity") != NETWORK_ID or freeze.get("adapter_identity") != ADAPTER_ID
            or freeze.get("graph_hash") != GRAPH_HASH or freeze.get("candidate_adopted") is not False):
        raise ValueError("Exact E318 neutral observational contract required")
    pins = freeze.get("input_sha256")
    if not isinstance(pins, dict) or not pins:
        raise ValueError("Complete input file pins required")
    required = set()
    for alias, name in PATHS.items():
        path = safe(name, root)
        if alias in ("graph_bundle", "start_whole", "e307_final", "e310_final"):
            files = [p for p in path.rglob("*") if p.is_file()]
        elif alias in ("e307_raw", "e310_combined"):
            files = [path / item for item in RAW]
        else:
            files = [path]
        if not files:
            raise ValueError("Missing E318 input: " + alias)
        required.update(file.relative_to(root).as_posix() for file in files)
    if set(pins) != required:
        raise ValueError("Exact used E318 input pin roster required")
    for name, digest in pins.items():
        if sha256(safe(name, root)) != digest:
            raise ValueError("E318 input differs: " + name)
    sources = freeze.get("source_sha256")
    if not isinstance(sources, dict) or set(sources) != set(source_roster(root)):
        raise ValueError("Exact E318 source roster required")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != [] or review.get("freeze_sha256") != freeze_sha256
            or review.get("source_sha256") != sources):
        raise ValueError("Approved matching E318 review required")
    for name, digest in sources.items():
        if sha256(safe(name, root)) != digest:
            raise ValueError("E318 source differs: " + name)
    for label in ("e307", "e310"):
        producer = json.loads(safe(PATHS[label + "_freeze"], root).read_text("utf8"))
        approval = json.loads(safe(PATHS[label + "_review"], root).read_text("utf8"))
        result = json.loads(safe(PATHS[label + "_result"], root).read_text("utf8"))
        audit = json.loads(safe(PATHS[label + "_audit"], root).read_text("utf8"))
        if (producer.get("graph_hash") != GRAPH_HASH or producer.get("adapter_identity") != ADAPTER_ID
                or approval.get("approved") is not True or approval.get("blocking_defects") != []
                or approval.get("freeze_sha256") != pins[PATHS[label + "_freeze"]]
                or result.get("run_valid") is not True or audit.get("pass") is not True
                or audit.get("diagnostic_run_valid") is not True
                or audit.get("result_sha256") != pins[PATHS[label + "_result"]]
                or audit.get("output_manifest_sha256") != pins[PATHS[label + "_output_manifest"]]):
            raise ValueError("Exact audited E307/E310 producer metadata required: " + label)
        output_manifest = json.loads(safe(PATHS[label + "_output_manifest"], root).read_text("utf8"))
        if output_manifest.get("schema") != f"flylab.m3.{label}.output-manifest.v1":
            raise ValueError("Audited producer output manifest required: " + label)
        producer_root = (Path(PATHS["e307_raw"]).parent if label == "e307"
                         else Path(PATHS["e310_combined"]).parents[1])
        used_aliases = ("start_whole", "e307_final", "e307_raw") if label == "e307" else ("e310_final", "e310_combined")
        for alias in used_aliases:
            path = safe(PATHS[alias], root)
            files = ((path / item for item in RAW) if alias in ("e307_raw", "e310_combined")
                     else (p for p in path.rglob("*") if p.is_file()))
            for file in files:
                name = file.relative_to(root).as_posix()
                key = Path(name).relative_to(producer_root).as_posix()
                if output_manifest.get("files", {}).get(key) != {"sha256": pins[name], "bytes": file.stat().st_size}:
                    raise ValueError("Used E318 file differs from audited producer output: " + name)
    if shutil.disk_usage(output.parent).free < LIMITS["minimum_free_disk_bytes"]:
        raise OSError("E318 free disk reserve unavailable")
    return freeze


def exact(left, right) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes(order="C") == b.tobytes(order="C")


class ReferenceMismatch(ValueError):
    """A complete observed field disagrees with a pinned E307/E310 reference."""


class DriveObserver:
    """Instance-only interception, preserving the original bound method and input bytes."""

    def __init__(self, session, motor_ids, on_complete, journal):
        self.session, self.network = session, session.network
        self.motor_ids = np.asarray(motor_ids).copy()
        self.on_complete, self.journal = on_complete, journal
        self.records: list[dict] = []
        self.original = None
        self.previous_instance = None
        self.had_instance = False
        self.installed = False

    def install(self):
        if self.installed:
            raise RuntimeError("Observer already installed")
        self.had_instance = "advance_readout" in vars(self.network)
        if self.had_instance:
            self.previous_instance = vars(self.network)["advance_readout"]
        self.original = self.network.advance_readout
        self.network.advance_readout = self._intercept
        self.installed = True
        return self.restore

    def restore(self):
        if not self.installed:
            return
        if self.had_instance:
            self.network.advance_readout = self.previous_instance
        else:
            delattr(self.network, "advance_readout")
        self.installed = False

    def _intercept(self, drive, motor_ids, steps):
        control = self.session.control_tick + 1
        before = self.network.tick
        record = {"control": control, "network_before": before, "requested": False,
                  "host_returned": False, "device_confirmed": False}
        self.records.append(record)
        values = np.asarray(drive)
        ids = np.asarray(motor_ids)
        if (control != len(self.records) or before != (control - 1) * 20
                or values.dtype != np.float32 or values.shape != (N,)
                or not np.isfinite(values).all() or not exact(ids, self.motor_ids)
                or steps != 20):
            self.journal(record)
            raise ValueError("Exact E318 native readout entry required")
        record["drive"] = values.copy()
        record["requested"] = True
        self.journal(record)
        result = self.original(drive, motor_ids, steps)
        record["host_returned"] = True
        self.journal(record)
        rate = self.network.rate.detach().cpu().numpy().copy()
        adaptation = self.network.adaptation.detach().cpu().numpy().copy()
        mask = self.network.output_mask.detach().cpu().numpy().copy()
        motor = np.asarray(result)
        if (rate.dtype != np.float32 or adaptation.dtype != np.float32
                or rate.shape != (N,) or adaptation.shape != (N,)
                or mask.dtype != np.float32 or mask.shape != (N,)
                or motor.dtype != np.float32 or motor.shape != (MOTOR,)
                or self.network.tick != before + 20 or not np.isfinite(rate).all()
                or not np.isfinite(adaptation).all() or not np.isfinite(motor).all()
                or np.any(rate < 0) or np.any(adaptation < 0)
                or not np.isin(mask, [0., 1.]).all()
                or not exact(motor, rate[self.motor_ids])):
            raise ValueError("Actual complete E318 neural/MN endpoint differs")
        record.update(rate=rate, adaptation=adaptation, motor_rate=motor.copy(),
                      motor_mask=mask[self.motor_ids].copy(), network_tick=self.network.tick,
                      device_confirmed=True)
        self.journal(record)
        self.on_complete(record)
        return result


class ChunkWriter:
    def __init__(self, folder: Path, saver=np.savez_compressed, chunk_size: int = CHUNK):
        self.folder, self.saver, self.chunk_size = folder, saver, chunk_size
        self.rows: list[dict] = []
        self.receipts: list[dict] = []

    def add(self, record: dict):
        if record["control"] != sum(row["count"] for row in self.receipts) + len(self.rows) + 1:
            raise ValueError("Noncontiguous E318 full-drive control")
        self.rows.append(record)
        if len(self.rows) == self.chunk_size:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        start, end = self.rows[0]["control"], self.rows[-1]["control"]
        name = f"full-drive-{start:03d}-{end:03d}.npz"
        pending = self.folder / (name + ".pending.npz")
        final = self.folder / name
        arrays = {"control_tick": np.asarray([row["control"] for row in self.rows], np.int64),
                  "network_tick": np.asarray([row["network_tick"] for row in self.rows], np.int64)}
        for key in ("drive", "rate", "adaptation", "motor_rate", "motor_mask"):
            arrays[key] = np.stack([row[key] for row in self.rows])
        self.saver(pending, **arrays)
        replace_owned(pending, final)
        self.receipts.append({"file": name, "start": start, "end": end, "count": len(self.rows),
                              "sha256": sha256(final), "bytes": final.stat().st_size})
        self.rows.clear()

    def partial(self):
        if not self.rows:
            return
        arrays = {"control_tick": np.asarray([row["control"] for row in self.rows], np.int64)}
        for key in ("drive", "rate", "adaptation", "motor_rate", "motor_mask"):
            rows = [row[key] for row in self.rows if key in row]
            if rows:
                arrays[key] = np.stack(rows)
        self.saver(self.folder / "partial-drive.npz", **arrays)


def whole_equal_except_wall(wrapper, observed: dict, expected: dict, label: str):
    if set(observed) != set(expected) or "wall_s" not in observed:
        raise ReferenceMismatch("Whole endpoint fields differ: " + label)
    comparison = copy.deepcopy(observed)
    comparison["wall_s"] = expected["wall_s"]
    try:
        wrapper.require_saved_whole_equal(comparison, expected, label)
    except ValueError as error:
        raise ReferenceMismatch("Whole endpoint differs: " + label) from error
    return {"whole_except_wall_exact": True, "actual_wall_s": observed["wall_s"],
            "reference_wall_s": expected["wall_s"], "excluded_fields": ["wall_s"]}


def compare_npz(actual: dict, expected: dict, label: str, *, prefix: int | None = None):
    if set(actual) != set(expected):
        raise ReferenceMismatch("Raw channel roster differs: " + label)
    for key, value in expected.items():
        sample = actual[key] if prefix is None or key in ("observed_ids", "motor_ids", "source_ids", "schema") else actual[key][:prefix]
        if not exact(sample, value):
            raise ReferenceMismatch("Raw channel differs: " + label + "/" + key)


def compare_outputs(raw: dict, observer: DriveObserver, produced: Path, data: dict, deps):
    from tools import verify_c_banc_m3_recurrent_resource_body as runner

    expected = data["reference"]
    if len(raw["rows"]) != CONTROLS + 1 or len(raw["native"]) != CONTROLS * 50:
        raise ValueError("Complete 301 endpoint/15000 native raw required")
    if len(observer.records) != CONTROLS or not all(row["device_confirmed"] for row in observer.records):
        raise ValueError("Complete 300 full-drive/MN observations required")
    for i, row in enumerate(raw["activity"][1:], 1):
        if not exact(row["drive"], observer.records[i - 1]["drive"][raw["ids"]]):
            raise ValueError("Actual full drive differs from selected endpoint drive")
    for name in RAW:
        if name == "trace.json":
            actual_rows = json.loads((produced / name).read_text("utf8"))["samples"]
            for label, reference in (("E310", expected["e310"]["trace.json"]),):
                if runner.protocol.jsonable(actual_rows) != reference["samples"]:
                    raise ReferenceMismatch(label + " complete trace samples differ")
            if runner.protocol.jsonable(actual_rows[:101]) != expected["e307"]["trace.json"]["samples"]:
                raise ReferenceMismatch("E307 prefix trace samples differ")
        else:
            actual = runner.protocol.load_npz(produced / name)
            compare_npz(actual, expected["e310"][name], "E310/" + name)
            if name == "native-contact.npz":
                prefix = 5000
            else:
                prefix = 101
            compare_npz(actual, expected["e307"][name], "E307/" + name, prefix=prefix)
    return {"e307_prefix_exact": True, "e310_full_exact": True,
            "selected_drive_exact": True, "control_endpoints": 301, "native_samples": 15000}


class Dependencies:
    """Heavy dependencies are imported only after the passive gate."""

    def __init__(self):
        from tools import verify_c_banc_m3_recurrent_resource_body as runner
        from tools import verify_c_banc_m3_resource_continuation as e310

        self.e310, self.runner = e310, runner
        self.native = e310.NativeDependencies()
        self.wrapper = self.native.wrapper
        self.health_stop = self.native.health_stop

    def load(self, freeze):
        from flylab.c.graph import GraphStore
        from flylab.c.storage import StateStore

        runner = self.runner
        p = freeze["paths"]
        graph = GraphStore.load(safe(p["graph_bundle"]))
        saved = StateStore.load(safe(p["start_whole"]))
        e307 = StateStore.load(safe(p["e307_final"]))
        final = StateStore.load(safe(p["e310_final"]))
        roster = runner.protocol.load_npz(safe(p["roster"]))
        runner.checked_roster(roster)
        original_freeze = json.loads(safe(p["e307_freeze"]).read_text("utf8"))
        if (graph.hash != GRAPH_HASH or original_freeze["paths"]["graph_bundle"] != p["graph_bundle"]
                or original_freeze["graph_hash"] != GRAPH_HASH
                or saved["network"]["model_hash"] != NETWORK_ID or saved["adapter"]["hash"] != ADAPTER_ID
                or saved["control_tick"] != 0 or saved["network"]["tick"] != 0):
            raise ValueError("Exact current E307 original graph/start identity required")
        for label, value, control in (("E307", e307, 100), ("E310", final, 300)):
            if value["control_tick"] != control or value["network"]["tick"] != control * 20 or value["network"]["model_hash"] != NETWORK_ID:
                raise ValueError(label + " frozen whole identity/clocks differ")
        for label in ("e307", "e310"):
            result = json.loads(safe(p[label + "_result"]).read_text("utf8"))
            audit = json.loads(safe(p[label + "_audit"]).read_text("utf8"))
            if (result.get("run_valid") is not True or audit.get("pass") is not True
                    or audit.get("diagnostic_run_valid") is not True):
                raise ValueError("Audited E307/E310 producer result required: " + label)
        reference = {}
        for label, alias in (("e307", "e307_raw"), ("e310", "e310_combined")):
            folder = safe(p[alias])
            reference[label] = {name: json.loads((folder / name).read_text("utf8")) if name == "trace.json"
                                else runner.protocol.load_npz(folder / name) for name in RAW}
        ids = np.unique(np.r_[roster["premotor_ids"], roster["motor_ids"]]).astype(np.int64)
        return {"graph": graph, "saved": saved, "e307": e307, "e310": final,
                "roster": roster, "ids": ids, "motor_ids": roster["motor_ids"],
                "source_ids": roster["source_ids"], "reference": reference, "freeze": freeze}

    def construct(self, data, ledger, persist, on_return):
        return self.native.construct("neutral", data, ledger, persist, on_return)

    def initial(self, session, data):
        observed = session.snapshot()
        self.wrapper.require_saved_whole_equal(observed, data["saved"], "E318 control0 full whole including wall")
        if (session.control_tick != 0 or session.network.tick != 0 or abs(session.body.physics_time()) > 2e-8
                or session.network.identity != NETWORK_ID or session.adapter.identity != ADAPTER_ID
                or session.network.device != "cuda" or session.network.cuda_implementation != "packed"
                or session.network.interval_steps != 20 or not session.fused_readout
                or not np.all(observed["network"]["synaptic_resource"] == 1)):
            raise ValueError("Exact SSP/u0 control0 replay origin required")
        return {"whole_start_exact_including_wall": True, "control_tick": 0, "neural_tick": 0}

    def capture(self, session, raw):
        return self.e310.NativeDependencies.capture(self.native, session, raw)

    def save_raw(self, folder, raw, data, clock):
        return self.runner.NativeDependencies.save_raw(self.native, folder, raw, data, clock)

    def __getattr__(self, name):
        return getattr(self.native, name)


def run(freeze_path: Path, freeze_sha256: str, review_path: Path, review_sha256: str,
        output: Path, *, root: Path = ROOT, gatekeeper=gate, dependencies=None,
        clock=time.perf_counter, writer=put_json, saver=np.savez_compressed, meter=usage) -> dict:
    began = clock()
    freeze = gatekeeper(freeze_path, freeze_sha256, review_path, review_sha256, output, root=root)
    output.mkdir(parents=True, exist_ok=False)
    chunk_folder = output / "full-drive-chunks"
    chunk_folder.mkdir()
    ledger = {"construction": [], "body_construction": [], "construction_requested": 0,
              "construction_completed": 0, "controls_requested": 0, "controls_completed": 0,
              "observer_requested": 0, "observer_host_returned": 0, "observer_device_confirmed": 0,
              "chunk_saved_controls": 0, "neural_ticks_device_confirmed": 0,
              "physics_steps_completed": 0, "retry_count": 0, "body_constructors": 0,
              "limits": LIMITS.copy(), "errors": []}
    result = {"schema": RESULT_SCHEMA, "diagnostic_id": "E318", "run_valid": False,
              "learning_input_approved": False, "candidate_adopted": False,
              "biological_gait_proven": False, "classification": "not_tested"}
    source = {"schema": "flylab.m3.e318.source-manifest.v1", "freeze_sha256": freeze_sha256,
              "review_sha256": review_sha256, "source_sha256": freeze["source_sha256"],
              "input_sha256": freeze["input_sha256"]}
    raw = {key: [] for key in ("rows", "controls", "recruitment", "activity", "native")}
    chunk = ChunkWriter(chunk_folder, saver=saver)
    session = observer = detach = control_journal = deps = native_clock = None
    holder, errors, first_error = {}, [], None
    science_start = clock()

    def count_replace_retry():
        ledger["ledger_replace_retries"] = ledger.get("ledger_replace_retries", 0) + 1

    def persist():
        ledger["elapsed_s"] = clock() - began
        pending = output / "advance-ledger.pending.json"
        writer(pending, ledger)
        replace_owned(pending, output / "advance-ledger.json", count_replace_retry)

    def fail(stage, error):
        nonlocal first_error
        row = {"stage": stage, "type": type(error).__name__, "message": str(error)[:500]}
        errors.append(row)
        if first_error is None:
            first_error = row
        ledger["errors"] = errors.copy()
        result.update(run_valid=False, learning_input_approved=False,
                      first_error=first_error, errors=errors.copy())
        return row

    def check_resources(sample):
        if sample["device_total_bytes"] != baseline_usage["device_total_bytes"]:
            raise RuntimeError("E318 CUDA device memory identity changed")
        ledger["resources"] = sample
        ledger["additional_gpu_peak_bytes"] = max(
            ledger.get("additional_gpu_peak_bytes", 0),
            0, baseline_usage["device_free_bytes"] - sample["device_free_bytes"])
        ledger["additional_torch_peak_reserved_bytes"] = max(
            ledger.get("additional_torch_peak_reserved_bytes", 0),
            0, sample["gpu_peak_reserved_bytes"] - baseline_usage["gpu_reserved_bytes"])
        if (sample["host_peak_bytes"] > LIMITS["host_bytes"]
                or ledger["additional_gpu_peak_bytes"] > LIMITS["additional_gpu_bytes"]):
            raise MemoryError("E318 host/device-total GPU limit exceeded")

    def check_internal_deadline():
        if clock() - began >= LIMITS["internal_wall_s"]:
            raise TimeoutError("E318 internal wall deadline exceeded")

    try:
        persist()
        baseline_usage = meter()
        ledger["resource_baseline"] = baseline_usage
        writer(output / "source-manifest.json", source)
        deps = Dependencies() if dependencies is None else dependencies
        data = deps.load(freeze)
        raw["ids"] = data["ids"]
        ledger["construction_requested"] = 1
        persist()
        session = deps.construct(data, ledger, persist, lambda value: holder.setdefault("session", value))
        ledger["construction_completed"] = 1
        ledger["body_constructors"] = 1
        check_resources(meter())
        check_internal_deadline()
        if (len(ledger["construction"]) != 1 or len(ledger["body_construction"]) != 1
                or any(row.get(key) != 0 for row in ledger["construction"]
                       for key in ("unknown_setup_steps", "unknown_capture_setup_steps"))
                or any(row.get(key) != 0 for row in ledger["body_construction"]
                       for key in ("unknown_step_entries", "active_unreturned_frames"))):
            raise RuntimeError("Exact counted one-body constructor and settling required")
        result["origin"] = deps.initial(session, data)
        deps.save_state(output / "observed-start", session)
        if hasattr(deps, "trace_metadata"):
            raw["trace_metadata"] = deps.trace_metadata(session)
        detach = deps.install(session, raw["native"])
        native_clock = deps.e310.runner.native.AdvanceLedger() if hasattr(deps, "e310") else deps.clock()
        control_journal = deps.journal(output / "control-journal.jsonl")

        def observer_journal(record):
            ledger["observer_requested"] = sum(row["requested"] for row in observer.records)
            ledger["observer_host_returned"] = sum(row["host_returned"] for row in observer.records)
            ledger["observer_device_confirmed"] = sum(row["device_confirmed"] for row in observer.records)
            ledger["neural_ticks_device_confirmed"] = ledger["observer_device_confirmed"] * 20
            persist()

        observer = DriveObserver(session, data["motor_ids"], chunk.add, observer_journal)
        observer.install()
        deps.capture(session, raw)
        for control in range(1, CONTROLS + 1):
            check_internal_deadline()
            if clock() - science_start > LIMITS["science_wall_s"]:
                raise TimeoutError("E318 internal science deadline exceeded")
            ledger["controls_requested"] = control
            persist()
            before_native = len(raw["native"])
            event = deps.advance(session, "neutral", control, native_clock, control_journal)
            ledger["controls_completed"] = control
            deps.capture(session, raw)
            deps.readout_receipt(control_journal, "neutral", control, event, True)
            if (session.control_tick != control or session.network.tick != control * 20
                    or len(raw["native"]) != before_native + 50
                    or len(observer.records) != control or not observer.records[-1]["device_confirmed"]):
                raise ValueError("Incomplete E318 control/neural/native interval")
            native_last = deps.e310.runner.native.native_arrays(raw["native"][-50:]) if hasattr(deps, "e310") else None
            if native_last is not None and (not np.isfinite(native_last["upright"]).all()
                                            or np.any(native_last["upright"] < .9)
                                            or np.any(native_last["nonfoot_contact"])):
                raise deps.health_stop("Measured E318 upright/nonfoot health stop")
            ledger["physics_steps_completed"] = len(raw["native"])
            ledger["chunk_saved_controls"] = sum(row["count"] for row in chunk.receipts)
            check_resources(meter())
            if control in (100, 300):
                observed = session.snapshot()
                expected = data["e307" if control == 100 else "e310"]
                result["whole_" + str(control)] = whole_equal_except_wall(deps.wrapper, observed, expected, "E318 control" + str(control))
                deps.save_state(output / ("whole-" + str(control)), session)
            persist()
            if session.fault or session.body.fault:
                raise deps.health_stop("Measured E318 session/body health stop")
        chunk.flush()
        ledger["chunk_saved_controls"] = sum(row["count"] for row in chunk.receipts)
        if len(chunk.receipts) != CONTROLS // CHUNK or ledger["chunk_saved_controls"] != CONTROLS:
            raise ValueError("Complete E318 25-control chunks required")
        if clock() - science_start > LIMITS["science_wall_s"]:
            raise TimeoutError("E318 science deadline exceeded after final control")
        check_internal_deadline()
        result["science_complete"] = True
    except Exception as error:  # noqa: BLE001 - preserve native and partial failure
        fail("science", error)
        if isinstance(error, ReferenceMismatch):
            result["classification"] = "reference_mismatch"
        elif deps is not None and isinstance(error, deps.health_stop):
            result["classification"] = "measured_native_health_stop"
            result["health_stop"] = {"control_attempted": ledger["controls_requested"],
                                     "native_samples": len(raw["native"])}
        else:
            result["classification"] = "not_tested"
        result["science_error"] = traceback.format_exc(limit=8)[-3000:]
    finally:
        ledger["science_wall_s"] = clock() - science_start
        ledger["native_samples_observed"] = len(raw["native"])
        if native_clock is not None:
            ledger["clock_events"] = copy.deepcopy(native_clock.events)
            ledger["clock_valid"] = native_clock.clock_valid
            ledger["actual_neural_s"] = native_clock.neural_s
            ledger["actual_physical_s"] = native_clock.physical_s
        finalization_start = clock()
        if session is None and "session" in holder:
            session = holder["session"]
            ledger["owned_session_recovered"] = True
        for label, callback in (("observer_restore", observer.restore if observer else None),
                                ("native_detach", detach),
                                ("control_journal_close", control_journal.close if control_journal else None)):
            if callback:
                try:
                    callback()
                except Exception as error:  # noqa: BLE001 - record all cleanup failures
                    fail(label, error)
        try:
            chunk.partial()
        except Exception as error:  # noqa: BLE001 - retain first science failure
            fail("partial_chunk", error)
        if observer is not None and observer.records and not observer.records[-1].get("device_confirmed"):
            failed = observer.records[-1]
            try:
                writer(output / "failed-drive-observer.json", {key: value for key, value in failed.items()
                                                               if not isinstance(value, np.ndarray)})
                if "drive" in failed:
                    saver(output / "failed-drive-observer.npz", drive=failed["drive"],
                          control_tick=np.asarray(failed["control"], np.int64))
            except Exception as error:  # noqa: BLE001 - preserve first observer failure
                fail("failed_drive_artifact", error)
        if deps is not None and native_clock is not None and "data" in locals():
            try:
                for failure in deps.save_raw(output, raw, data, native_clock):
                    fail("raw_" + failure["artifact"], RuntimeError(failure["error"]))
            except Exception as error:  # noqa: BLE001 - preserve other raw components
                fail("raw_publication", error)
        if session is not None and first_error is not None:
            try:
                deps.save_state(output / "partial-final", session)
            except Exception as error:  # noqa: BLE001 - guarded unhealthy snapshot
                fail("partial_whole", error)
                try:
                    deps.save_partial(output / "partial-components", session)
                except Exception as secondary:  # noqa: BLE001 - preserve first failure
                    fail("partial_components", secondary)
        if session is not None:
            ledger["close_requested"] = True
            try:
                session.close()
                ledger["close_returned"] = True
            except Exception as error:  # noqa: BLE001 - record owned close failure
                fail("owned_close", error)
        if "baseline_usage" in locals():
            try:
                check_resources(meter())
            except Exception as error:  # noqa: BLE001 - terminal resource failure invalidates run
                fail("terminal_resources", error)
        if first_error is None and deps is not None:
            try:
                result["comparison"] = compare_outputs(raw, observer, output, data, deps)
                if (ledger["controls_completed"] != CONTROLS or ledger["observer_device_confirmed"] != CONTROLS
                        or ledger["chunk_saved_controls"] != CONTROLS
                        or ledger["physics_steps_completed"] != LIMITS["physics_steps"]):
                    raise ValueError("Complete E318 fixed resource counts required")
                result.update(run_valid=True, learning_input_approved=True,
                              classification="same_current_ssp_input_observed")
            except Exception as error:  # noqa: BLE001 - comparison failure is not success
                fail("comparison", error)
                result["classification"] = "reference_mismatch" if isinstance(error, ReferenceMismatch) else "not_tested"
        result["chunks"] = chunk.receipts.copy()
        files = {}
        try:
            files = {path.relative_to(output).as_posix(): {"sha256": sha256(path), "bytes": path.stat().st_size}
                     for path in output.rglob("*") if path.is_file() and path.name not in
                     ("advance-ledger.json", "result.json", "publication-failure.json", "output-manifest.json")}
        except Exception as error:  # noqa: BLE001 - keep earlier artifacts and primary error
            fail("manifest_input_scan", error)

        def publication_pass():
            ledger["errors"] = errors.copy()
            if first_error is not None:
                result.update(run_valid=False, learning_input_approved=False,
                              first_error=first_error, errors=errors)
            else:
                result.update(first_error=None, errors=[])
            result["cost"] = copy.deepcopy(ledger)
            for name, value in (("advance-ledger.json", ledger), ("result.json", result)):
                try:
                    writer(output / name, value)
                except Exception as error:  # noqa: BLE001 - keep trying independent artifacts
                    fail("publication_" + name, error)
            if first_error is not None:
                result.update(run_valid=False, learning_input_approved=False,
                              first_error=first_error, errors=errors, cost=copy.deepcopy(ledger))
                try:
                    writer(output / "publication-failure.json", result)
                    writer(output / "result.json", result)
                except Exception as error:  # noqa: BLE001 - retain primary failure
                    fail("failure_journal", error)
            try:
                for name in ("advance-ledger.json", "result.json", "publication-failure.json"):
                    path = output / name
                    if path.is_file():
                        files[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
                if sum(row["bytes"] for row in files.values()) > LIMITS["output_bytes"]:
                    raise OSError("E318 output size limit exceeded")
                writer(output / "output-manifest.json", {"schema": "flylab.m3.e318.output-manifest.v1", "files": files})
            except Exception as error:  # noqa: BLE001 - next bounded pass records failure
                fail("output_manifest", error)

        # The first pass measures publication. The second is a bounded terminal commit;
        # its own self-referential duration is excluded from the recorded boundary.
        ledger["publication_boundary"] = "before_preliminary_commit"
        publication_pass()
        ledger["publication_boundary"] = "after_preliminary_commit_before_terminal_commit"
        ledger["finalization_wall_s"] = clock() - finalization_start
        ledger["elapsed_s"] = clock() - began
        if ledger["finalization_wall_s"] > LIMITS["finalization_reserve_s"]:
            fail("finalization_deadline", TimeoutError("E318 finalization reserve exceeded"))
        if ledger["elapsed_s"] >= LIMITS["internal_wall_s"]:
            fail("internal_deadline", TimeoutError("E318 internal wall deadline exceeded"))
        terminal_errors_before = len(errors)
        publication_pass()
        if len(errors) > terminal_errors_before:
            # The final manifest can fail after result.json was written. Never return
            # success in that case; preserve an explicit failure without an unbounded
            # self-referential publication loop. The prior manifest is not approved.
            ledger["terminal_commit_failed"] = True
            for name in ("publication-failure.json", "result.json", "advance-ledger.json"):
                result.update(cost=copy.deepcopy(ledger), first_error=first_error,
                              errors=errors.copy(), run_valid=False, learning_input_approved=False)
                try:
                    writer(output / name, ledger if name == "advance-ledger.json" else result)
                except Exception as error:  # noqa: BLE001 - returned failure remains authoritative
                    fail("terminal_failure_" + name, error)
            result["cost"] = copy.deepcopy(ledger)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("freeze-sha256", "review-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = run(args.freeze, args.freeze_sha256, args.review, args.review_sha256, args.output)
    print(json.dumps(result, allow_nan=False), flush=True)
    if not result["run_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
