"""E351 once report recovery; preserved E350 worker/audit remain rejected."""

from __future__ import annotations

import time

ENTRY = time.perf_counter()
import argparse
import ast
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
# The sole CLI is a file invocation; its default import path is tools/, not ROOT.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = "verification/banc-natural-gait-m3-20260919"
OWNED = (
    "tools/recover_c_banc_m3_source_pose_report.py",
    "tests/test_c_banc_source_pose_report_recovery.py",
)
REGISTRATION = BASE + "/parent-e351-recovery-input-registration.json"
REGISTRATION_SHA = "f8d81f0af676ceec55d276d3738297c4d9313b0265167c521e01b6e8432e6248"
REFLECTION = BASE + "/parent-e351-reflected-design-registration.json"
REFLECTION_SHA = "803aa0384d0d6964cb18e73817d83df7f901aa21ac5b79e5dac741053be50ae0"
OUTPUT = BASE + "/e351-source-pose-report"
CORE = "tools/probe_c_banc_m3_shape_shared_timing_body.py"
CORE_SHA = "97d3f518b90b9af4d6daefdb5431232e1649fbe9159a9ae0d63e4c866bdc9c2d"
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
ARMS = ("B", "I", "C")
MODEL_HASH = "1b2a89084311ebd2ccbccbe2c57f2ec29fa937e115bb6cd4c7f6cd16ba7c2e6f"
OUTPUT_CAP, STAGE_CAP = 4 << 20, 64 << 20
_CORE = None


def numpy():
    import numpy as np

    return np


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def safe(name, root=ROOT):
    if (
        type(name) is not str
        or "\\" in name
        or PurePosixPath(name).is_absolute()
        or any(x in ("", ".", "..") for x in name.split("/"))
    ):
        raise ValueError("unsafe relative path")
    file = (root / name).resolve()
    if not file.is_relative_to(root.resolve()):
        raise ValueError("path escapes root")
    return file


def shared():
    global _CORE
    if _CORE is None:
        file = ROOT / CORE
        if sha(file.read_bytes()) != CORE_SHA:
            raise ValueError("frozen predicate source differs")
        spec = importlib.util.spec_from_file_location("e351_frozen_hypothesis", file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CORE = module
    return _CORE


def lower(value):
    """Exact scalar JSON boundary, applied AFTER the frozen predicate."""
    kind = type(value)
    if value is None or kind in (bool, int, str):
        return value
    if kind is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite JSON float")
        return value
    if kind is list:
        return [lower(v) for v in value]
    if kind is dict:
        if any(type(key) is not str for key in value):
            raise TypeError("native string JSON keys required")
        return {key: lower(v) for key, v in value.items()}
    np = numpy()
    if kind is np.bool_:
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if kind in (np.float16, np.float32, np.float64):
        if not np.isfinite(value):
            raise ValueError("nonfinite NumPy scalar")
        return float(value)
    raise TypeError("unsupported JSON value type: " + kind.__name__)


def strict_json(raw):
    def reject(value):
        raise ValueError("nonfinite JSON token: " + value)

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite JSON numeric value")
        return number

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        raw, parse_constant=reject, parse_float=finite_float, object_pairs_hook=pairs
    )


def failure(stage, error):
    return {"stage": stage, "type": type(error).__name__, "message": str(error)[:1024]}


def fail_report(error, *, stage="recovery", prior=None):
    # A rejected arbitrary report value must not poison the strict error sink.
    # Retain safe provenance, never repair/relabel a rejected scientific value.
    keys = (
        "original_worker_status",
        "original_worker_first_error",
        "original_worker_errors",
        "original_worker_H348",
        "original_audit",
        "original_worker_adopted",
        "recovery_body_rerun",
        "original_worker_retry_count",
        "body_evidence_qualified",
        "candidate_adopted",
        "F1_F8_complete",
        "biology_proven",
        "orientation_replayed_by_independent_audit",
        "new_body_NN_physics_GPU_calls",
        "source_sha256",
        "registration_sha256",
        "reflection_sha256",
        "consumed_input_digests",
        "consumed_read_counts",
        "no_second_fallback_original_read",
        "first_error",
        "errors",
    )
    result = {key: value for key, value in (prior or {}).items() if key in keys}
    result.update(
        schema="flylab.m3.e351.source-pose-report.v1",
        diagnostic_verified=False,
        scientific_diagnostic_status="technical_invalid",
        diagnostic_H348={
            "verdict": "technical_invalid",
            "supported": None,
            "gates": None,
        },
    )
    item = failure(stage, error)
    result.setdefault("errors", []).append(item)
    result.setdefault("first_error", item)
    if result["first_error"] is None:
        result["first_error"] = item
    result["external_completion_qualified"] = False
    result["independent_metrics"] = None
    result["late_error_has_precedence"] = True
    return result


def usage(folder):
    return sum(file.stat().st_size for file in folder.rglob("*") if file.is_file())


def publish(
    file,
    report,
    *,
    timely=lambda: None,
    output_cap=OUTPUT_CAP,
    stage_cap=STAGE_CAP,
    opener=None,
    count_folder=True,
):
    raw = json.dumps(lower(report), indent=2, allow_nan=False).encode("utf-8")
    retained = usage(file.parent) if count_folder else 0
    if len(raw) > output_cap or retained + len(raw) > output_cap:
        raise ValueError("report output cap exceeded before publication")
    if retained + len(raw) > stage_cap:
        raise ValueError("report stage cap exceeded before publication")
    timely()
    with (opener or (lambda name: name.open("xb")))(file) as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    timely()  # Includes flush/fsync/close; success file alone is insufficient.
    retained = usage(file.parent) if count_folder else file.stat().st_size
    if retained > output_cap or retained > stage_cap:
        raise ValueError("report output/stage cap exceeded after close")


def publish_outcome(output, calculate, *, timely=lambda: None, failure_context=dict):
    """Late publication failures invalidate provisional success, preserving it."""
    output.mkdir(parents=True, exist_ok=False)
    report = None
    try:
        report = calculate()
        publish(output / "report.json", report, timely=timely)
        return report, 0 if report["diagnostic_verified"] else 1
    except Exception as error:  # noqa: BLE001 -- owned calculation/publication boundary
        report = fail_report(
            error, stage="calculation_or_publication", prior=report or failure_context()
        )
        report["provisional_report_present"] = (output / "report.json").exists()
        try:
            publish(output / "failure.json", report, timely=timely)
        except Exception as secondary:  # noqa: BLE001 -- late sink/deadline retains external rejection
            report["errors"].append(failure("failure_publication", secondary))
        return report, 1


class EvidenceReader:
    """Bounded cached bytes: hash/header/decode reuse the same single read."""

    def __init__(self, root=ROOT, *, timely=lambda: None, generated=False):
        self.root, self.timely = Path(root), timely
        self.cache, self.digests, self.read_counts, self.json_cache = {}, {}, {}, {}
        self.opened_members = []
        self.array_authorized = generated
        if generated and self.root.resolve().is_relative_to(ROOT.resolve()):
            relative = self.root.resolve().relative_to(ROOT.resolve()).as_posix()
            if not relative.startswith(BASE + "/e351-sol-r1-"):
                raise ValueError(
                    "generated reader must use owned E351 fixture namespace"
                )

    def bytes(self, spec, *, cap=16 << 20, array=False):
        name = spec["path"]
        if name.endswith((".npz", ".npy")) and not array:
            raise ValueError("scientific bytes require authorized selected read")
        if array and not self.array_authorized:
            raise ValueError("E351 private gate required before array bytes")
        self.timely()
        file = safe(name, self.root)
        if (
            type(spec["bytes"]) is not int
            or not 0 <= spec["bytes"] <= cap
            or file.stat().st_size != spec["bytes"]
        ):
            raise ValueError("bounded input descriptor size differs")
        if name not in self.cache:
            if sum(len(v) for v in self.cache.values()) + spec["bytes"] > STAGE_CAP:
                raise ValueError("bounded cached input bytes exceeded")
            with file.open("rb") as stream:
                raw = stream.read(spec["bytes"] + 1)
            if len(raw) != spec["bytes"]:
                raise ValueError("input changed or truncated")
            self.cache[name] = raw
            self.read_counts[name] = 1
            self.digests[name] = sha(raw)
        raw = self.cache[name]
        if len(raw) != spec["bytes"] or (
            spec.get("sha256") is not None and self.digests[name] != spec["sha256"]
        ):
            raise ValueError("pinned consumed input differs: " + name)
        self.timely()
        return raw

    def json(self, spec):
        raw = self.bytes(spec)
        if spec["path"] not in self.json_cache:
            self.json_cache[spec["path"]] = strict_json(raw)
        return self.json_cache[spec["path"]]

    def npz(self, spec, roster, *, selected=None):
        np = numpy()
        raw = self.bytes(spec, array=True)
        result = {}
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = archive.infolist()
            names = [v.filename for v in infos]
            if (
                len(names) > 128
                or len(set(names)) != len(names)
                or any(
                    "\\" in n
                    or n.startswith("/")
                    or not n.endswith(".npy")
                    or any(p in ("", ".", "..") for p in n.split("/"))
                    for n in names
                )
            ):
                raise ValueError("unsafe/duplicate archive member")
            if selected is None and set(names) != {key + ".npy" for key in roster}:
                raise ValueError("exact recovery raw member roster differs")
            for key, (shape, dtype) in roster.items():
                self.timely()
                member = selected[key] if selected is not None else None
                info = archive.getinfo(member["member"] if member else key + ".npy")
                if member:
                    for attr, field in (
                        ("compress_type", "zip_compression"),
                        ("compress_size", "compressed_bytes"),
                        ("file_size", "uncompressed_bytes"),
                        ("CRC", "crc32"),
                    ):
                        if getattr(info, attr) != member[field]:
                            raise ValueError("selected ZIP metadata differs")
                if info.file_size > 4 << 20:
                    raise ValueError("bounded selected allocation exceeded")
                self.opened_members.append((spec["path"], info.filename))
                with archive.open(info) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version != (1, 0) or (
                        member and version != tuple(member["npy_version"])
                    ):
                        raise ValueError("selected NPY version differs")
                    actual_shape, fortran, actual_dtype = (
                        np.lib.format.read_array_header_1_0(
                            stream, max_header_size=1024
                        )
                    )
                    if (
                        actual_shape != tuple(shape)
                        or actual_dtype != np.dtype(dtype)
                        or fortran
                        or actual_dtype.hasobject
                        or actual_dtype.fields
                        or actual_dtype.subdtype
                        or stream.tell() + math.prod(shape) * actual_dtype.itemsize
                        != info.file_size
                        or (
                            member
                            and (
                                stream.tell() != member["header_bytes"]
                                or member["fortran_order"] is not False
                            )
                        )
                    ):
                        raise ValueError("selected bounded NPY header differs")
                    payload = stream.read(info.file_size)
                    if len(payload) != math.prod(shape) * actual_dtype.itemsize:
                        raise ValueError("selected payload truncated")
                    value = (
                        np.frombuffer(payload, dtype=actual_dtype).reshape(shape).copy()
                    )
                if value.dtype.kind == "f" and np.isinf(value).any():
                    raise ValueError("Inf in selected/raw channel")
                result[key] = value
        return result


def own_sources(root=ROOT):
    return {name: sha(safe(name, root).read_bytes()) for name in OWNED}


def require(condition, label):
    if not condition:
        raise ValueError(label)


def admit(worker, old_audit, admission, worker_sources):
    require(
        worker.get("schema") == admission["worker_schema"]
        and worker.get("source_sha256") == worker_sources
        and worker.get("status") == admission["worker_status"]
        and worker.get("analysis_completed") is False
        and worker.get("first_error") == admission["worker_singleton_error"]
        and worker.get("errors") == [admission["worker_singleton_error"]]
        and worker.get("H348") == admission["worker_H348"],
        "exact singleton publication rejection admission",
    )
    require(
        old_audit.get("schema") == "flylab.m3.e350.bounded-independent-audit-failure.v1"
        and old_audit.get("passed") is False
        and old_audit.get("body_evidence_qualified") is False
        and old_audit.get("H348") == "not_evaluated"
        and old_audit.get("worker_first_error") == admission["worker_singleton_error"]
        and old_audit.get("audit_first_error")
        == {"type": "ValueError", "message": "independent H348 conjunction"},
        "only frozen conjunction audit failure is admissible",
    )
    require(
        list(worker["arms"]) == list(ARMS), "exact completed B/I/C admission roster"
    )
    for arm, ledger in worker["arms"].items():
        require(
            ledger["status"] == "completed"
            and ledger["first_error"] is None
            and ledger["errors"] == []
            and ledger["cleanup_completed"] is True
            and ledger["constructor_returned"] is True
            and type(ledger["controls_requested"]) is int
            and ledger["controls_requested"] == 300
            and type(ledger["controls_returned"]) is int
            and ledger["controls_returned"] == 300
            and type(ledger["native_samples_acquired"]) is int
            and ledger["native_samples_acquired"] == 15000
            and ledger["controls_unreturned"] == 0,
            arm + " completed clean count admission",
        )
        for phase, count, duration in (
            ("constructor", 2000, 0.2),
            ("scientific_physics", 15000, 1.5),
        ):
            clock = ledger[phase]
            require(
                clock["tracking_available"] is True
                and clock["method_restored"] is True
                and clock["absolute_clock_known"] is True
                and type(clock["steps_requested"]) is int
                and clock["steps_requested"] == count
                and type(clock["steps_returned"]) is int
                and clock["steps_returned"] == count,
                arm + " real restored phase counters",
            )
            for key in ("absolute_before_s", "absolute_after_s", "actual_advance_s"):
                require(
                    type(clock[key]) in (float, int) and math.isfinite(clock[key]),
                    "finite actual phase clock",
                )
            require(
                abs(clock["actual_advance_s"] - duration) <= 1e-9
                and abs(
                    clock["absolute_after_s"] - clock["absolute_before_s"] - duration
                )
                <= 1e-9,
                "exact settling/science duration",
            )
            expected_errors = (
                [admission["allowed_constructor_probe_error"]]
                if phase == "constructor"
                else []
            )
            require(
                clock["clock_read_errors"] == expected_errors,
                "only exact before-init missing-d probe retained",
            )
        require(
            ledger.get("baseline_full_exact") is True
            if arm == "B"
            else ledger["prefix_checked_before_c101"] is True,
            "B full / I-C prefix admission",
        )
        require(
            abs(ledger["constructor"]["absolute_before_s"]) <= 1e-9
            and abs(
                ledger["scientific_physics"]["absolute_before_s"]
                - ledger["constructor"]["absolute_after_s"]
            )
            <= 1e-9,
            "settling/science absolute clock linkage",
        )
    require(
        worker["neural_calls"] == worker["CPG_advance_calls"] == 0
        and worker["candidate_adopted"] is False
        and worker["F1_F8_complete"] is False
        and worker["biology_proven"] is False,
        "preserved body diagnostic scope",
    )


def metadata_preflight(root=ROOT, *, reader=None):
    reader = reader or EvidenceReader(root)
    for name, pin in ((REGISTRATION, REGISTRATION_SHA), (REFLECTION, REFLECTION_SHA)):
        spec = {"path": name, "bytes": safe(name, root).stat().st_size, "sha256": pin}
        reader.json(spec)
    reg, reflected = reader.json_cache[REGISTRATION], reader.json_cache[REFLECTION]
    require(
        reg["schema"] == "flylab.m3.e351.report-recovery-registration.v1"
        and reflected["original_authorized"] is False
        and reg["original_authorized"] is False,
        "E351 ROOT reflection required",
    )
    for spec in (reg["design"], reg["actual_review"], reg["E350_registration"]):
        reader.bytes(spec)
    source_registration = reader.json(reg["E350_registration"])
    require(
        reg["payloads"] == source_registration["payloads"],
        "same selected original three payloads",
    )
    for group in (reg["consumed_sources"], reg["executed_source_snapshots"]):
        for spec in group.values():
            reader.bytes(spec)
    for name, pin in reg["force_policy_value"]["source_sha256"].items():
        spec = {"path": name, "bytes": safe(name, root).stat().st_size, "sha256": pin}
        reader.bytes(spec)
    for spec in reg["finite_provenance"].values():
        reader.bytes(spec)
    worker_spec = reg["preserved_JSON"][reg["worker_folder"] + "/result.json"]
    worker = reader.json(worker_spec)
    audit_name = BASE + "/parent-e350-independent-body-audit.json"
    old_audit = reader.json(reg["finite_provenance"][audit_name])
    admit(worker, old_audit, reg["admission"], reg["worker_source_sha256"])
    full = reader.json(
        reg["finite_provenance"][BASE + "/parent-e350-original-full-exit.json"]
    )
    identity = reader.json(full["identity"])
    require(
        full["qualified"] is True
        and full["actual_original_worker_calls"] == 1
        and full["worker_retry_count"] == 0
        and identity["qualified"] is True
        and identity["all_owned_pids_ended"] is True
        and identity["errors"] == []
        and identity["owned_pids"]
        and set(map(str, identity["owned_pids"])) == set(identity["pid_create_times"])
        and all(v is False for v in identity["present_after_return"].values()),
        "actual rejected-worker full-exit identity qualification required",
    )
    for spec in list(reg["payloads"].values()) + list(reg["preserved_raw"].values()):
        require(
            safe(spec["path"], root).stat().st_size == spec["bytes"],
            "raw stat-only descriptor differs",
        )
    return {
        "registration": reg,
        "reader": reader,
        "worker": worker,
        "old_audit": old_audit,
        "original_full_exit": full,
    }


def exact(a, b, label, *, nan=False):
    np = numpy()
    a, b = np.asarray(a), np.asarray(b)
    require(a.shape == b.shape and a.dtype == b.dtype, label + " type/shape")
    if a.dtype.kind == "f":
        require(
            not np.isinf(a).any() and not np.isinf(b).any(), label + " Inf rejected"
        )
        if nan:
            require(np.array_equal(np.isnan(a), np.isnan(b)), label + " NaN positions")
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        else:
            require(np.isfinite(a).all() and np.isfinite(b).all(), label + " finite")
    require(a.tobytes() == b.tobytes(), label + " exact finite bytes")


def fixed_configuration(reader, registration, config, before):
    """Read literal pinned CONFIG/OPTIONS/PROFILE without importing old runtime."""
    name = "tools/probe_c_banc_m3_current_shape_counterfactual.py"
    tree = ast.parse(
        reader.bytes(registration["consumed_sources"][name]).decode("utf-8")
    )
    literals = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in (
                    "CONFIG",
                    "OPTIONS",
                    "PROFILE",
                ):
                    literals[target.id] = ast.literal_eval(node.value)
    require(
        config["config"] == literals["CONFIG"]
        and config["body_options"] == literals["OPTIONS"]
        and config["physics_profile"] == literals["PROFILE"]
        and config["physics_identity"]["profile"] == literals["PROFILE"]
        and config["physics_identity"]["body_options"] == literals["OPTIONS"]
        and config["physics_identity"]["precision"] == "float64"
        and config["physics_identity"]["timestep"] == 0.0001,
        "pinned fixed configuration/options/profile",
    )
    require(
        all(
            before[key] == config[key]
            for key in (
                "config",
                "body_options",
                "physics_profile",
                "physics_identity",
                "model_hash",
                "joint_names",
            )
        ),
        "before/after no-op configuration linkage",
    )


def original_payloads(reader, registration):
    payloads = {}
    for role, spec in registration["payloads"].items():
        require(
            spec["scope"] == "historical_selected"
            and spec["unselected_members_allowed_but_not_opened"] is True
            and set(spec["decode_members"]) == set(spec["selected_members"]),
            "registered selected roles",
        )
        roster = {
            key: (tuple(m["shape"]), m["dtype"])
            for key, m in spec["selected_members"].items()
        }
        payloads[role] = reader.npz(spec, roster, selected=spec["selected_members"])
    np = numpy()
    for role, arrays in payloads.items():
        for key, value in arrays.items():
            require(
                key == "slip_mm_s" or np.isfinite(value).all(),
                role + " selected finite " + key,
            )
    return payloads


def independent_plan(control, reference, plan):
    """Arithmetic retained from frozen ROOT E350 audit a156e557..., lines95-139."""
    np = numpy()
    q, curves, basis = control["command"], reference["curves"], reference["phase_basis"]
    exact(plan["B"], q, "B source plan")
    for arm in ("I", "C"):
        exact(plan[arm][:101], q[:101], arm + " command prefix")
    maximum_error = 0.0
    for c in range(100, 301):
        phases = []
        for leg in range(6):
            d = np.square(curves[leg] - q[c, leg * 7 : leg * 7 + 7]).sum(axis=1)
            logits = -d / float(reference["temperature"][leg])
            weights = [math.exp(float(v - logits.max())) for v in logits]
            total = math.fsum(weights)
            z = [
                math.fsum(w * float(basis[k, j]) for k, w in enumerate(weights)) / total
                for j in range(2)
            ]
            radius = math.hypot(*z)
            require(radius >= 0.5, "fixed reliable phase")
            phases.append((math.atan2(z[1], z[0]) / math.tau) % 1)
            require(
                np.allclose(plan["z"][c - 100, leg], z, rtol=0, atol=1e-12),
                "independent phase encoder",
            )
            require(
                math.isclose(
                    float(plan["R"][c - 100, leg]), radius, rel_tol=0, abs_tol=1e-12
                ),
                "saved phase reliability",
            )
            require(
                abs((float(plan["phi"][c - 100, leg]) - phases[-1] + 0.5) % 1 - 0.5)
                <= 1e-12,
                "saved circular phase",
            )
        if c == 100:
            continue
        x = min(1.0, max(0.0, (c * 0.005 - 0.5) / 0.1))
        alpha = x * x * (3 - 2 * x)
        for arm in ("I", "C"):
            target = []
            for leg in range(6):
                phase = phases[leg] if arm == "I" else phases[0]
                position = phase * 128
                lo = math.floor(position) % 128
                fraction = position - math.floor(position)
                target.extend(
                    (1 - fraction) * curves[leg, lo]
                    + fraction * curves[leg, (lo + 1) % 128]
                )
            expected = (
                np.asarray(target)
                if c >= 120
                else (1 - alpha) * q[c] + alpha * np.asarray(target)
            )
            maximum_error = max(
                maximum_error, float(np.abs(plan[arm][c] - expected).max())
            )
            require(
                np.allclose(plan[arm][c], expected, atol=1e-12, rtol=0),
                arm + " fixed independent reference",
            )
    return maximum_error


def gait_model(payload, file, digest):
    """Pinned gait.load_model body bcceb5d1..., with cached JSON/source digest only.

    Coordinate/clock arithmetic is unchanged; no JSON-reader monkeypatch or reread.
    """
    from scipy.ndimage import gaussian_filter1d

    from tools import analyze_c_banc_biological_gait as gait

    np = numpy()
    rows = payload["samples"]
    t = np.array([r["time_s"] for r in rows])
    dt = gait.validate_clock(t)
    if not np.isclose(dt, 0.005) or any(
        r["control_tick"] != rows[0]["control_tick"] + i for i, r in enumerate(rows)
    ):
        raise ValueError("Expected complete 5 ms control-boundary recording")
    basis = np.array([r["body"]["basis"] for r in rows])
    if not np.allclose(np.einsum("tji,tjk->tik", basis, basis), np.eye(3), atol=1e-9):
        raise ValueError("Body basis is not orthogonal")
    origin = np.array([r["body"]["position"] for r in rows])
    points = np.array([[r["body"]["legs"][leg] for leg in LEGS] for r in rows])
    ego = np.einsum("tlpj,tjk->tlpk", points - origin[:, None, None], basis)
    lengths = np.median(
        np.linalg.norm(np.diff(ego, axis=2), axis=-1).sum(axis=-1), axis=0
    )
    brake_times = [
        r["time_s"] for r in rows if (r.get("navigation") or {}).get("brake_request")
    ]
    end = min(t[-1], 10.0, min(brake_times) if brake_times else np.inf)
    window = (t >= 0.5) & (t < end)
    speed = gait.position_forward_speed(origin, basis[:, :, 0], dt)
    yaw = np.unwrap([r["body"]["yaw"] for r in rows])
    yaw_rate = gaussian_filter1d(np.gradient(yaw, dt), 0.02 / dt, mode="nearest")
    return {
        "t": t,
        "dt": dt,
        "ap": ego[:, :, -1, 0] / lengths,
        "knee": gait.geometric_knee(ego),
        "window": window,
        "moving": window & (speed > 0.5),
        "straight": np.abs(yaw_rate) < 0.5,
        "speed": speed,
        "yaw_rate": yaw_rate,
        "contact": np.array([r["feet"]["load_bearing"] for r in rows], dtype=bool),
        "position": origin,
        "upright": np.array([r["upright"] for r in rows]),
        "faults": [r["fault"] for r in rows if r["fault"]],
        "lengths": lengths,
        "start_s": 0.5,
        "end_s": float(end),
        "source": {
            "path": str(file.resolve()),
            "sha256": digest,
            "kind": "model",
            "body_hash": payload["body_hash"],
            "model_hash": payload["model_hash"],
            "adapter_hash": payload["adapter_hash"],
            "graph_hash": payload["graph_hash"],
            "parameters": payload["parameters"],
            "foot": "tarsus5 origin proxy",
            "length_unit": "mm",
            "speed_source": "Thorax coordinate derivative, projected onto physical forward axis, Gaussian sigma 20 ms; not boundary body.speed",
        },
    }


def physical(native, trace, current):
    """Independent fsum forward/slip as frozen ROOT audit; Python health bools."""
    np = numpy()
    rows = trace["samples"]
    basis0 = np.asarray(rows[0]["body"]["basis"], np.float64)[[0, 2], 0]
    norm = math.hypot(*basis0)
    require(norm > 0, "nonzero initial projected forward basis")
    displacement = (
        np.asarray(rows[299]["body"]["position"])[[0, 2]]
        - np.asarray(rows[100]["body"]["position"])[[0, 2]]
    )
    forward = math.fsum(
        float(displacement[j]) * float(basis0[j]) / norm for j in range(2)
    )
    support = ((native["time_s"] >= 0.5) & (native["time_s"] < 1.5))[:, None] & (
        native["normal_load_bw"] >= 0.02
    )
    slip, status = None, "unavailable_support"
    if (
        support.any()
        and native["slip_valid"][support].all()
        and np.isfinite(native["slip_mm_s"][support]).all()
    ):
        slip = math.fsum(
            float(a) * float(b)
            for a, b in zip(
                native["normal_load_bw"][support], native["slip_mm_s"][support]
            )
        ) / math.fsum(float(v) for v in native["normal_load_bw"][support])
        status = "evaluated"
    return {
        "status": "evaluated",
        "forward_mm": forward,
        "slip_mm_s": slip,
        "slip_status": status,
        "minimum_native_upright": float(native["upright"].min()),
        "nonfoot_contact_count": int(native["nonfoot_contact"].sum()),
        "measured_native_healthy": bool(
            native["upright"].min() >= 0.9 and not native["nonfoot_contact"].any()
        ),
        "all_control_faults_clear": all(r["fault"] is None for r in rows),
        "current_forwarded_healthy": bool(
            np.isfinite(current).all() and current.min() >= 0.9
        ),
    }


def compare_evaluation(
    worker_metric, evaluation, independently_measured, gait_measured
):
    require(
        evaluation == worker_metric,
        "on-disk evaluation and worker metrics exact linkage",
    )
    require(evaluation["full_status"] == "evaluated", "preserved full evaluation")
    for key, actual in independently_measured.items():
        if key == "current_forwarded_healthy":
            continue  # New conservative channel is independently verified separately.
        saved = evaluation["physical"][key]
        if actual is None or type(actual) is bool or type(actual) is str:
            require(
                type(saved) is type(actual) and saved == actual,
                "independent physical " + key,
            )
        else:
            require(
                type(saved) in (float, int)
                and math.isfinite(saved)
                and math.isclose(saved, actual, rel_tol=0, abs_tol=1e-12),
                "independent physical " + key,
            )
    require(
        evaluation["gait"] == gait_measured,
        "pinned empirical gait independently recomputed",
    )


def diagnostic_hypothesis(metrics):
    """Never lower predicate inputs: health `is True` remains frozen semantics."""
    result = shared().hypothesis(metrics, True)
    if result["gates"] is not None:
        i, c = metrics["I"]["physical"], metrics["C"]["physical"]
        gait = metrics["C"]["gait"]
        independent = {
            "forward_C_better_I": c["forward_mm"] - i["forward_mm"] > 1e-9,
            "slip_C_better_I": c["slip_mm_s"] - i["slip_mm_s"] < -1e-9,
            "absolute_forward": c["forward_mm"] >= 1.6790085779712631,
            "absolute_slip": c["slip_mm_s"] <= 1.8026436968249955,
            "bout": gait["window"]["longest_forward_bout_s"] >= 0.3,
            "six_cycles": all(
                gait["events"][leg]["complete_cycles"] > 0 for leg in LEGS
            ),
            "health": c["measured_native_healthy"] is True
            and c["all_control_faults_clear"] is True,
        }
        require(
            result["gates"] == independent, "independently equivalent fixed seven gates"
        )
    return result


def recover_evidence(context):
    """No original reads unless this context passed the private CLI gate."""
    reader, reg, worker = context["reader"], context["registration"], context["worker"]
    require(reader.array_authorized, "E351 original gate must precede recovery")
    np = numpy()
    from tools import analyze_c_banc_biological_gait as gait

    root, folder = reader.root, safe(reg["worker_folder"], reader.root)
    specs = {s["path"]: s for s in reg["finite_JSON_dossier"]}

    def preserved(relative):
        return reader.json(reg["preserved_JSON"][reg["worker_folder"] + "/" + relative])

    policy, held = preserved("force-policy.json"), preserved("held-force-receipt.json")
    require(policy == reg["force_policy_value"], "exact reviewed force policy")
    policy_name = reg["worker_folder"] + "/force-policy.json"
    for key, name in (
        ("force_policy", "force-policy.json"),
        ("held_force_receipt", "held-force-receipt.json"),
    ):
        spec = reg["preserved_JSON"][reg["worker_folder"] + "/" + name]
        require(
            worker[key]
            == {"path": name, "bytes": spec["bytes"], "sha256": spec["sha256"]},
            "actual final force descriptor",
        )
    require(
        held["schema"] == "flylab.m3.e349.held-force-receipt.v1"
        and held["policy_sha256"] == reader.digests[policy_name],
        "force receipt policy linkage",
    )
    counters = worker["force_call_counters"]
    require(
        worker["force_call_counters_status"] == "instrumented"
        and worker["force_before_read_errors"] == []
        and counters
        == {
            "adapter_requested": 900,
            "native_action_requested": 900,
            "native_action_returned": 900,
        }
        and all(type(value) is int for value in counters.values()),
        "actual force requests/returns",
    )
    require(len(held["actions"]) == 900, "full force observation count")
    for index, row in enumerate(held["actions"]):
        require(
            row["arm"] == ARMS[index // 300]
            and type(row["control_tick"]) is int
            and row["control_tick"] == index % 300 + 1
            and row["returned"] is True
            and row["error"] is None
            and row["before"] == policy["recorded_origin"]
            and row["after"] == policy["recorded_origin"],
            "actual ordered held-force before/after",
        )
    state = reader.json(
        specs[BASE + "/e335-smooth-secant-body/observed-start/state.json"]
    )
    require(
        {k: state["body"][k] for k in ("push_left", "push_force")}
        == policy["recorded_origin"],
        "recorded explicit no-force origin",
    )
    source_trace = reader.json(specs[BASE + "/e335-smooth-secant-body/trace.json"])
    source = source_trace["samples"]
    require(len(source) == 301, "exact source boundary rows")
    payloads = original_payloads(reader, reg)
    control, reference, native_source = (
        payloads[key] for key in ("control", "reference", "native")
    )
    plan_roster = {
        "B": ((301, 42), "float64"),
        "I": ((301, 42), "float64"),
        "C": ((301, 42), "float64"),
        "z": ((201, 6, 2), "float64"),
        "R": ((201, 6), "float64"),
        "phi": ((201, 6), "float64"),
    }
    plan = reader.npz(reg["preserved_raw"]["shape_plan"], plan_roster)
    require(all(np.isfinite(a).all() for a in plan.values()), "finite shape plan")
    maximum_plan_error = independent_plan(control, reference, plan)
    exact(
        control["control_tick"],
        np.arange(301, dtype=np.int64),
        "original control indices",
    )
    require(
        np.allclose(
            control["physics_time_s"], np.arange(301) * 0.005, rtol=0, atol=2e-8
        ),
        "original control clock",
    )
    native_roster = {
        key: (tuple(value.shape), value.dtype) for key, value in native_source.items()
    }
    observation_roster = {
        key: (
            (15000,),
            "int64"
            if key == "native_index"
            else "bool"
            if key.endswith("_available")
            else "float64",
        )
        for key in (
            "native_index",
            "time_s",
            "source_upright",
            "current_upright",
            "time_available",
            "source_available",
            "current_available",
        )
    }
    metrics, identity, cap_bytes = {}, None, {}
    for arm in ARMS:
        reader.timely()
        ledger, trace, evaluation, observer = (
            preserved(arm + "/" + name)
            for name in (
                "receipt.json",
                "trace.json",
                "evaluation.json",
                "source-pose.json",
            )
        )
        require(ledger == worker["arms"][arm], "actual on-disk arm receipt linkage")
        config = ledger["configuration"]
        fixed_configuration(
            reader, reg, config, ledger["configuration_before_configure"]
        )
        actual_identity = {
            k: config[k]
            for k in (
                "physics_identity",
                "joint_names",
                "config",
                "body_options",
                "physics_profile",
                "model_hash",
            )
        }
        if identity is None:
            identity = actual_identity
        require(
            actual_identity == identity
            and config["model_hash"] == MODEL_HASH
            and len(config["joint_names"]) == len(set(config["joint_names"])) == 42
            and config["activation_trajectory_tau_s"] == 0
            and config["active_length"] is False
            and config["graded_adhesion"] is False,
            "actual fixed independent arm identity/options/order",
        )
        require(
            trace["body_hash"] == MODEL_HASH
            and all(
                trace[k] == source_trace[k]
                for k in ("model_hash", "adapter_hash", "graph_hash", "parameters")
            ),
            "trace physical identity and source neural labels",
        )
        native = reader.npz(reg["preserved_raw"][arm + "_native"], native_roster)
        for key, value in native.items():
            require(
                key == "slip_mm_s" or np.isfinite(value).all(),
                "finite native channel " + key,
            )
        require(
            np.isfinite(native["normal_load_bw"]).all()
            and (native["normal_load_bw"] >= 0).all()
            and np.array_equal(native["slip_valid"], np.isfinite(native["slip_mm_s"])),
            "native load/slip masks",
        )
        require(
            np.allclose(
                native["time_s"], np.arange(1, 15001) * 0.0001, atol=2e-8, rtol=0
            ),
            "exact acquired native clock",
        )
        observer_spec = reg["preserved_JSON"][
            reg["worker_folder"] + "/" + arm + "/source-pose.json"
        ]
        local_descriptor = {
            "path": "source-pose.json",
            "bytes": observer_spec["bytes"],
            "sha256": observer_spec["sha256"],
        }
        require(
            ledger["source_pose_observation"]
            == worker["source_pose_observations"][arm]
            == local_descriptor,
            "actual final/arm observer descriptor",
        )
        require(
            observer["schema"] == "flylab.m3.e350.source-pose-observation.v1"
            and observer["arm"] == arm
            and observer["source_sha256"] == reg["worker_source_sha256"]
            and observer["policy"] == reg["observation_policy_value"]
            and observer["count"]
            == observer["source_available_count"]
            == observer["current_available_count"]
            == observer["time_available_count"]
            == 15000
            and observer["errors"] == []
            and observer["orientation_replayed_by_independent_audit"] is False
            and observer["new_observer_native_qualified"] is False,
            "exact actual observer schema/source/policy/counts",
        )
        raw_spec = reg["preserved_raw"][arm + "_source_pose"]
        require(
            observer["raw"]
            == {
                "path": "source-pose.npz",
                "bytes": raw_spec["bytes"],
                "sha256": raw_spec["sha256"],
            },
            "acquisition-pinned observer raw",
        )
        observed = reader.npz(raw_spec, observation_roster)
        exact(
            observed["native_index"],
            np.arange(1, 15001, dtype=np.int64),
            "actual one-based observation indices",
        )
        for channel, mask in (
            ("time_s", "time_available"),
            ("source_upright", "source_available"),
            ("current_upright", "current_available"),
        ):
            require(
                observed[mask].all() and np.isfinite(observed[channel]).all(),
                "completed finite available observation " + channel,
            )
        exact(
            observed["time_s"],
            native["time_s"],
            "same acquired native observation time",
        )
        exact(
            observed["source_upright"],
            native["upright"],
            "same source-live/native upright",
        )
        for key, index in (("first", 0), ("last", 14999)):
            row = observer[key]
            require(
                row["native_index"] == index + 1
                and row["errors"] == []
                and all(
                    row[flag] is True
                    for flag in (
                        "time_available",
                        "source_available",
                        "current_available",
                        "callback_attempted",
                        "callback_returned",
                    )
                ),
                "actual first/last flags",
            )
            require(
                all(
                    row[channel] == float(observed[channel][index])
                    for channel in ("time_s", "source_upright", "current_upright")
                ),
                "actual first/last values",
            )
        require(
            observer["current_health_passed"] is True
            and observer["recorded_health_passed"] is True
            and (observed["current_upright"] >= 0.9).all()
            and (observed["source_upright"] >= 0.9).all(),
            "independent dual health",
        )
        rows = trace["samples"]
        require(len(rows) == 301, "all acquired control boundaries")
        prefix = 300 if arm == "B" else 100
        for c, row in enumerate(rows):
            require(
                type(row["control_tick"]) is int
                and row["control_tick"] == c
                and abs(row["physics_time_s"] - c * 0.005) <= 2e-8
                and abs(row["time_s"] - c * 0.005) <= 2e-8,
                "actual boundary clock",
            )
            require(
                row["neural_request"] is None
                and row["fault"] is None
                and row["source_label"]["source_neural_metadata_is_label_only"] is True,
                "no neural execution or boundary faults",
            )
            for key in ("qpos", "qvel", "ctrl", "joint_targets_rad"):
                values = np.asarray(row[key], np.float64)
                require(
                    values.shape == np.asarray(source[c][key], np.float64).shape
                    and np.isfinite(values).all(),
                    "all-row finite physical state/action " + key,
                )
            for leg in LEGS:
                geometry = np.asarray(row["body"]["legs"][leg], np.float64)
                require(
                    geometry.shape == (8, 3) and np.isfinite(geometry).all(),
                    "actual all8 anatomical geometry",
                )
                if c <= prefix:
                    exact(
                        geometry,
                        np.asarray(source[c]["body"]["legs"][leg], np.float64),
                        "source geometry prefix",
                    )
            require(
                row["adhesion"] == source[c]["adhesion"]
                and len(row["adhesion"]) == 6
                and all(type(v) is bool for v in row["adhesion"])
                and row["tendon_inputs"] == source[c]["tendon_inputs"],
                "complete original other actions",
            )
            require(
                set(row["tendon_inputs"])
                == {leg + "_tarsus" for leg in LEGS} | {"abdomen_pitch", "abdomen_yaw"}
                and all(
                    type(value) in (float, int) and math.isfinite(value)
                    for value in row["tendon_inputs"].values()
                ),
                "exact finite eight tendon channels",
            )
            for optional in ("joint_stiffness_scale", "adhesion_control"):
                require(
                    row.get(optional) == source[c].get(optional),
                    "unchanged optional action flags",
                )
            if c:
                exact(
                    np.asarray(row["joint_targets_rad"], np.float64),
                    plan[arm][c],
                    "actual planned command",
                )
                exact(
                    np.asarray(source[c]["joint_targets_rad"], np.float64),
                    control["command"][c],
                    "original source command",
                )
                exact(
                    np.asarray(
                        [source[c]["tendon_inputs"][leg + "_tarsus"] for leg in LEGS],
                        np.float64,
                    ),
                    control["tendon"][c],
                    "original source tendon",
                )
            if c <= prefix:
                for key in ("qpos", "qvel", "ctrl", "joint_targets_rad"):
                    exact(
                        np.asarray(row[key], np.float64),
                        np.asarray(source[c][key], np.float64),
                        "source boundary prefix " + key,
                    )
        for key in native:
            exact(
                native[key][: prefix * 50],
                native_source[key][: prefix * 50],
                "source native prefix " + key,
                nan=key == "slip_mm_s",
            )
        cap_bytes[arm] = usage(folder / arm)
        require(cap_bytes[arm] <= 16 << 20, "actual inclusive completed arm cap")
        measured_physical = physical(native, trace, observed["current_upright"])
        require(
            measured_physical["measured_native_healthy"] is True
            and measured_physical["all_control_faults_clear"] is True
            and measured_physical["current_forwarded_healthy"] is True,
            "independent complete health",
        )
        trace_name = reg["worker_folder"] + "/" + arm + "/trace.json"
        measured_gait = gait.analyze(
            gait_model(trace, safe(trace_name, root), reader.digests[trace_name])
        )[0]
        compare_evaluation(
            worker["metrics"][arm], evaluation, measured_physical, measured_gait
        )
        metrics[arm] = {
            "full_status": "evaluated",
            "physical": measured_physical,
            "gait": measured_gait,
        }
    require(usage(folder) <= STAGE_CAP, "preserved stage64MiB")
    derived = diagnostic_hypothesis(
        metrics
    )  # Typed values FIRST; lowering only publisher.
    return {
        "schema": "flylab.m3.e351.source-pose-report.v1",
        "diagnostic_verified": True,
        "scientific_diagnostic_status": "verified_from_preserved_technical_rejected_run",
        "original_worker_status": worker["status"],
        "original_worker_first_error": worker["first_error"],
        "original_worker_errors": worker["errors"],
        "original_worker_H348": worker["H348"],
        "original_audit": context["old_audit"],
        "original_worker_adopted": False,
        "recovery_body_rerun": False,
        "original_worker_retry_count": 0,
        "body_evidence_qualified": False,
        "external_completion_qualified": False,
        "candidate_adopted": False,
        "F1_F8_complete": False,
        "biology_proven": False,
        "orientation_replayed_by_independent_audit": False,
        "new_body_NN_physics_GPU_calls": 0,
        "original_evidence_calls": 1,
        "source_sha256": own_sources(),
        "registration_sha256": REGISTRATION_SHA,
        "reflection_sha256": REFLECTION_SHA,
        "original_worker_source_sha256": reg["worker_source_sha256"],
        "consumed_input_digests": reader.digests,
        "consumed_read_counts": reader.read_counts,
        "raw_digest_provenance": {
            key: {
                "path": spec["path"],
                "sha256": reader.digests[spec["path"]],
                "digest_provenance": spec["digest_provenance"],
            }
            for key, spec in reg["preserved_raw"].items()
        },
        "original_payloads": reg["payloads"],
        "selected_members_opened": [list(v) for v in reader.opened_members],
        "original_full_exit": context["original_full_exit"],
        "arm_inclusive_bytes": cap_bytes,
        "independent_metrics": metrics,
        "maximum_plan_error": maximum_plan_error,
        "diagnostic_H348": derived,
        "first_error": None,
        "errors": [],
        "elapsed_s_before_publication": time.perf_counter() - ENTRY,
    }


def failure_provenance(context):
    worker, reader = context["worker"], context["reader"]
    return {
        "original_worker_status": worker["status"],
        "original_worker_first_error": worker["first_error"],
        "original_worker_errors": worker["errors"],
        "original_worker_H348": worker["H348"],
        "original_audit": context["old_audit"],
        "original_worker_adopted": False,
        "original_worker_retry_count": 0,
        "recovery_body_rerun": False,
        "body_evidence_qualified": False,
        "candidate_adopted": False,
        "F1_F8_complete": False,
        "biology_proven": False,
        "orientation_replayed_by_independent_audit": False,
        "new_body_NN_physics_GPU_calls": 0,
        "source_sha256": own_sources(),
        "registration_sha256": REGISTRATION_SHA,
        "reflection_sha256": REFLECTION_SHA,
        "consumed_input_digests": reader.digests,
        "consumed_read_counts": reader.read_counts,
        "no_second_fallback_original_read": True,
    }


def make_freeze(qualification, *, root=ROOT, context=None):
    context = context or metadata_preflight(root)
    reg, reader = context["registration"], context["reader"]
    qual_file = safe(qualification, root)
    qual_spec = {
        "path": qualification,
        "bytes": qual_file.stat().st_size,
        "sha256": sha(qual_file.read_bytes()),
    }
    qual = reader.json(qual_spec)
    own = own_sources(root)
    checks = (
        "CPU",
        "ruff",
        "predicate_first",
        "publication",
        "admission",
        "reader_cap",
        "deadline",
        "metadata_guards",
        "selected_headers",
        "evaluation_linkage",
    )
    require(
        qual.get("schema") == "flylab.m3.e351.root-qualification.v1"
        and qual.get("source_sha256") == own
        and qual.get("passed") is True
        and qual.get("original_calls") == 0
        and all(qual.get("checks", {}).get(k) is True for k in checks),
        "current ROOT SECOND qualification required",
    )
    execution = qual["execution"]
    contract, supervision, full_exit = (
        reader.json(execution[k]) for k in ("contract", "supervision", "full_exit")
    )
    require(
        all(contract["source_sha256"].get(k) == v for k, v in own.items())
        and contract["original_calls"] == 0
        and supervision["success"] is True
        and supervision["original_calls"] == 0
        and supervision["contract_sha256"] == execution["contract"]["sha256"]
        and full_exit["all_owned_ended"] is True
        and full_exit["exit_code"] == 0
        and full_exit["first_error"] is None
        and full_exit["secondary_errors"] == []
        and full_exit["owned_identities"]
        and all(
            x["close"] == "ended"
            and type(x["pid"]) is int
            and type(x["create_time"]) in (float, int)
            and math.isfinite(x["create_time"])
            for x in full_exit["owned_identities"]
        )
        and contract["commands"]
        and len(contract["commands"]) == len(supervision["attempts"])
        and all(
            a["argv"] == j["argv"] and a["exit_code"] == 0 and a["killed"] is False
            for a, j in zip(supervision["attempts"], contract["commands"])
        ),
        "current source/argv/PID full-exit qualification",
    )
    cost = full_exit["charged_s"]
    require(
        type(cost) in (float, int) and math.isfinite(cost) and 0 < cost <= 20,
        "generated bounded actual full-exit cost",
    )
    return {
        "schema": "flylab.m3.e351.report-recovery-freeze.v1",
        "source_sha256": own,
        "design": reg["design"],
        "registration_sha256": REGISTRATION_SHA,
        "reflection_sha256": REFLECTION_SHA,
        "qualification": qual_spec,
        "consumed_sources": reg["consumed_sources"],
        "preserved_worker_source_sha256": reg["worker_source_sha256"],
        "admission": reg["admission"],
        "prospective": reg["prospective"],
        "original_authorized": False,
        "recovery_body_rerun": False,
    }


def original_argv(freeze, review):
    return [
        ".venv/Scripts/python.exe",
        "-X",
        "utf8",
        "-B",
        OWNED[0],
        "--recover",
        "--freeze",
        freeze,
        "--review",
        review,
        "--output",
        OUTPUT,
    ]


def original_gate(freeze_name, review_name, *, root=ROOT, output=OUTPUT, reader=None):
    context = metadata_preflight(root, reader=reader)
    reader = context["reader"]

    def read(name):
        file = safe(name, root)
        return reader.json({"path": name, "bytes": file.stat().st_size, "sha256": None})

    freeze, review = read(freeze_name), read(review_name)
    require(
        freeze
        == make_freeze(freeze["qualification"]["path"], root=root, context=context),
        "current exact E351 freeze required",
    )
    freeze_sha = reader.digests[freeze_name]
    own = own_sources(root)
    require(
        review.get("schema") == "flylab.m3.e351.root-original-review.v1"
        and review.get("approved") is True
        and review.get("source_sha256") == own
        and review.get("freeze_sha256") == freeze_sha
        and review.get("registration_sha256") == REGISTRATION_SHA
        and review.get("reflection_sha256") == REFLECTION_SHA
        and review.get("output") == OUTPUT
        and output == OUTPUT,
        "current private E351 ROOT approval required",
    )
    reservation = reader.json(review["reservation"])
    expected = {
        "schema": "flylab.m3.e351.root-reservation.v1",
        "cumulative_before_s": 9843.038036599502,
        "cumulative_limit_s": 9900,
        "reserved_s": 8,
        "audit_wall_s": 8,
        "audit_internal_s": 7,
        "original_calls": 1,
        "retry_count": 0,
        "body_calls": 0,
        "neural_calls": 0,
        "GPU_calls": 0,
        "physics_calls": 0,
        "CPG_calls": 0,
        "source_sha256": own,
        "freeze_sha256": freeze_sha,
        "registration_sha256": REGISTRATION_SHA,
        "reflection_sha256": REFLECTION_SHA,
        "review_path": review_name,
        "output": OUTPUT,
        "argv": original_argv(freeze_name, review_name),
    }
    require(
        all(
            type(reservation.get(k)) is type(v) and reservation.get(k) == v
            for k, v in expected.items()
        ),
        "exact E351 once/cumulative/source/argv reservation",
    )
    require(
        not safe(output, root).exists()
        and shutil.disk_usage(root).free >= 512 << 20
        and all(
            os.environ.get(k) == "1"
            for k in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        ),
        "fresh output/CPU1/free disk before arrays",
    )
    reader.array_authorized = (
        True  # Sole production capability granted after every gate.
    )
    return context


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--qualification")
    parser.add_argument("--review")
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--recover", action="store_true")
    args = parser.parse_args(argv)
    if args.recover:
        require(
            list(sys.argv[1:] if argv is None else argv)
            == original_argv(args.freeze, args.review)[5:],
            "exact sole E351 recovery CLI argv",
        )

    def timely():
        if time.perf_counter() - ENTRY > 7:
            raise TimeoutError(
                "E351 inclusive import/read/hash/audit/publication/close7s"
            )
        import psutil

        if psutil.Process().memory_info().rss > 2 << 30:
            raise MemoryError("E351 host2GiB")

    reader = EvidenceReader(timely=timely)
    if not args.recover:
        if not args.qualification:
            parser.error("passive freeze requires --qualification")
        frozen = make_freeze(
            args.qualification, context=metadata_preflight(reader=reader)
        )
        file = safe(args.freeze)
        publish(file, frozen, timely=timely, count_folder=False)
        return 0
    context = original_gate(args.freeze, args.review, output=args.output, reader=reader)
    report, code = publish_outcome(
        safe(args.output),
        lambda: recover_evidence(context),
        timely=timely,
        failure_context=lambda: failure_provenance(context),
    )
    if code:
        print(
            json.dumps(
                lower(
                    {
                        "first_error": report["first_error"],
                        "errors": report["errors"],
                        "diagnostic_verified": False,
                    }
                ),
                allow_nan=False,
            ),
            file=sys.stderr,
        )
    return code


if __name__ == "__main__":
    sys.exit(main())
