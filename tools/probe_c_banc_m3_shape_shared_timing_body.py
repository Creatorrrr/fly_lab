"""E348 fixed B/I/C body-only diagnostic.

CLI: --freeze OUTPUT --qualification ROOT_JSON creates a passive source freeze;
--run --freeze INPUT --review ROOT_JSON --output FRESH_DIR executes only after
current ROOT qualification/reservation. Python tests inject data/dependencies into
run_generated; that seam is unavailable from the original CLI. No neural/CPG
advance and no observer pose writes. Full-exit qualification remains external.
"""

from __future__ import annotations

import time

ENTRY = time.perf_counter()  # Includes later NumPy/helper imports and publication.
import argparse
import copy
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
BASE = "verification/banc-natural-gait-m3-20260919"
OWNED = (
    "tools/probe_c_banc_m3_shape_shared_timing_body.py",
    "tests/test_c_banc_shape_shared_timing_body.py",
)
OUTPUT = BASE + "/e348-shape-shared-timing-body"
REGISTRATION = BASE + "/parent-e348-selective-input-registration-v2.json"
REFLECTION = BASE + "/parent-e348-reflected-design-registration.json"
PINS = {
    REGISTRATION: "dfafe8b892fcc51fd545ec634a79dd07d7f13a0989183a6b5b62cb1aefb41844",
    REFLECTION: "831590fccc4a158dab2677542d564fb0080a1fa643405d0f7faf2f4c0e8a75b5",
    "docs/BANC_M3_E348_SHAPE_AND_SHARED_TIMING_BODY_DESIGN_20261002.md": "9865e220ef50a8a6a3e6ea7974b73f273c9930c53e2fc5bc33ebe4dcbc87db5d",
    BASE
    + "/parent-e348-astra-design-review.md": "2ad5415283338fd1bdbed7bc61659b3005790aa3bf4969f0e08d5e583e91fd16",
    BASE
    + "/parent-e348-astra-design-review.json": "d31ca34d3aabaf6e054a5c81b2f0d420742c63fd6a31b89980701270ec906f52",
    "tools/probe_c_banc_m3_current_shape_counterfactual.py": "3f8daa90b25ec0e4ec873589d56fa36363ba165072b5bd883cbe685a22ce4ce9",
    BASE
    + "/parent-e344-independent-metrics.py": "5da9074b2f1f5b0f9c2ba02bff0543e358d6c0ce62075a7677fc4d6ff5aa10ef",
    "tools/analyze_c_banc_biological_gait.py": "bcceb5d1e9f26293ffc800d09a1fbf063a1ab784d73cf5489b74c57ecdd14760",
}
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
LINKS = (
    "coxa",
    "trochanterfemur",
    "tibia",
    "tarsus1",
    "tarsus2",
    "tarsus3",
    "tarsus4",
    "tarsus5",
)
NATIVE = (
    "time_s",
    "normal_load_bw",
    "slip_mm_s",
    "slip_valid",
    "upright",
    "nonfoot_contact",
)
LIMITS = {
    "science_s": 70,
    "internal_s": 85,
    "outer_s": 95,
    "audit_s": 8,
    "final_reserve_s": 15,
    "host_bytes": 2 << 30,
    "raw_arm_bytes": 16 << 20,
    "stage_bytes": 64 << 20,
    "failure_bytes": 4 << 20,
    "stdout_bytes": 64 << 10,
    "free_disk_bytes": 512 << 20,
    "cpu_threads": 1,
    "neural_gpu_jobs": 0,
    "retry_count": 0,
}
MODEL_HASH = "1b2a89084311ebd2ccbccbe2c57f2ec29fa937e115bb6cd4c7f6cd16ba7c2e6f"


def path(name, root=ROOT):
    if (
        not isinstance(name, str)
        or "\\" in name
        or PurePosixPath(name).is_absolute()
        or any(x in ("", ".", "..") for x in name.split("/"))
    ):
        raise ValueError("unsafe relative path")
    candidate = (root / name).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("path escapes root")
    return candidate


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(file, *, cap=16 << 20):
    if file.stat().st_size > cap:
        raise ValueError("bounded JSON exceeded")
    return json.loads(
        file.read_text("utf-8"),
        parse_constant=lambda x: (_ for _ in ()).throw(
            ValueError("nonfinite JSON " + x)
        ),
    )


def checked_json(spec, root=ROOT):
    file = path(spec["path"], root)
    if (
        type(spec["bytes"]) is not int
        or not 0 < spec["bytes"] <= 16 << 20
        or file.stat().st_size != spec["bytes"]
    ):
        raise ValueError("JSON byte pin differs")
    raw = file.read_bytes()
    if digest(raw) != spec["sha256"]:
        raise ValueError("JSON pin differs")
    return json.loads(
        raw,
        parse_constant=lambda x: (_ for _ in ()).throw(
            ValueError("nonfinite JSON " + x)
        ),
    )


def sources(root=ROOT):
    return {name: digest(path(name, root).read_bytes()) for name in OWNED}


def metadata_preflight(root=ROOT):
    # Only these finite text/JSON/PY pins; payloads and referenced files unopened.
    for name, pin in PINS.items():
        file = path(name, root)
        if file.stat().st_size > 256 << 10 or digest(file.read_bytes()) != pin:
            raise ValueError("current static pin differs: " + name)
    registration = read_json(path(REGISTRATION, root))
    if registration.get(
        "schema"
    ) != "flylab.m3.e348.selective-input-registration.v2" or set(
        registration["payloads"]
    ) != {"control", "native", "reference"}:
        raise ValueError("selective registration differs")
    roles = {
        "control": ("command", "tendon", "control_tick", "physics_time_s"),
        "native": NATIVE,
        "reference": (
            "curves",
            "normalization",
            "temperature",
            "future_anchors",
            "phase_basis",
        ),
    }
    seen = set()
    for role, spec in registration["payloads"].items():
        path(spec["path"], root)
        if (
            spec["path"] in seen
            or set(spec["decode_members"]) != set(roles[role])
            or set(spec["selected_members"]) != set(roles[role])
            or spec.get("unselected_members_allowed_but_not_opened") is not True
        ):
            raise ValueError("selected role/member binding differs")
        seen.add(spec["path"])
        if type(spec["bytes"]) is not int or not 0 < spec["bytes"] <= 16 << 20:
            raise ValueError("bounded payload metadata")
        for key, member in spec["selected_members"].items():
            if (
                member["member"] != key + ".npy"
                or member["fortran_order"] is not False
                or member["npy_version"] != [1, 0]
                or member["dtype"] not in ("float64", "int64", "bool")
            ):
                raise ValueError("selected descriptor differs")
    return registration


def libraries():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import numpy as np

    from tools import probe_c_banc_m3_current_shape_counterfactual as static

    file = path(BASE + "/parent-e344-independent-metrics.py")
    if digest(file.read_bytes()) != PINS[BASE + "/parent-e344-independent-metrics.py"]:
        raise ValueError("oracle pin differs")
    spec = importlib.util.spec_from_file_location("e348_independent_oracle", file)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    return np, static, oracle


def selected(spec, *, root=ROOT, timely=lambda: None, opened=None):
    """Read/hash one bounded archive once, open only registered selected members."""
    np, _, _ = libraries()
    timely()
    file = path(spec["path"], root)
    if file.stat().st_size != spec["bytes"] or spec["bytes"] > 16 << 20:
        raise ValueError("payload bytes differ")
    raw = file.read_bytes()
    if digest(raw) != spec["sha256"]:
        raise ValueError("payload pin differs")
    result = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        if (
            len(names) > 128
            or len(set(names)) != len(names)
            or any(
                "\\" in n
                or n.startswith("/")
                or any(p in ("", ".", "..") for p in n.split("/"))
                or not n.endswith(".npy")
                for n in names
            )
        ):
            raise ValueError("unsafe/duplicate archive member")
        for key in spec["decode_members"]:
            timely()
            member = spec["selected_members"][key]
            info = archive.getinfo(member["member"])
            for attr, field in (
                ("compress_type", "zip_compression"),
                ("compress_size", "compressed_bytes"),
                ("file_size", "uncompressed_bytes"),
                ("CRC", "crc32"),
            ):
                if getattr(info, attr) != member[field]:
                    raise ValueError("selected ZIP metadata differs")
            if info.file_size > 4 << 20:
                raise ValueError("selected allocation exceeded")
            if opened is not None:
                opened.append(info.filename)
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                if version != tuple(member["npy_version"]) or version != (1, 0):
                    raise ValueError("selected NPY version differs")
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(
                    stream, max_header_size=1024
                )
                if (
                    list(shape) != member["shape"]
                    or dtype != np.dtype(member["dtype"])
                    or dtype.hasobject
                    or fortran
                    or stream.tell() != member["header_bytes"]
                    or stream.tell() + math.prod(shape) * dtype.itemsize
                    != info.file_size
                ):
                    raise ValueError("selected bounded header differs")
                payload = stream.read(info.file_size)
                if len(payload) != math.prod(shape) * dtype.itemsize:
                    raise ValueError("selected payload truncated")
                value = np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
            if key != "slip_mm_s" and not np.isfinite(value).all():
                raise ValueError("selected nonfinite value")
            result[key] = value
    timely()
    return result


def load_original(registration, *, root=ROOT, timely=lambda: None):
    # Access only finite dossier entries required by this body experiment.
    dossier = registration["finite_JSON_dossier"]
    specs = (
        {s["path"]: s for s in dossier}
        if isinstance(dossier, list)
        else {s["path"]: s for s in dossier.values()}
    )

    def finite(name):
        timely()
        return checked_json(specs[BASE + "/" + name], root)

    metadata = finite("parent-e337-current-body-metadata.json")
    state = finite("e335-smooth-secant-body/observed-start/state.json")
    trace = finite("e335-smooth-secant-body/trace.json")
    manifest = finite("e335-smooth-secant-body/observed-start/manifest.json")
    selected_meta = metadata["selected"]
    if (
        any(
            state.get(k) != selected_meta[k]
            for k in ("schema", "seed", "world", "parameters", "control_tick")
        )
        or sorted(state["body"]) != selected_meta["body_keys"]
    ):
        raise ValueError("literal body metadata differs")

    def literal(value):
        if isinstance(value, dict):
            if "__npy__" in value:
                raise ValueError("literal body forbids leaf references")
            for v in value.values():
                literal(v)
        elif isinstance(value, list):
            for v in value:
                literal(v)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite literal body")

    literal(state["body"])
    if len(state["body"]["state"]) != 924 or state["body"]["modelHash"] != MODEL_HASH:
        raise ValueError("literal integration924/model differs")
    np, static, _ = libraries()
    if manifest["versions"] != static._versions():
        raise ValueError("runtime package versions differ")
    data = {
        "body_state": state["body"],
        "body_metadata": metadata,
        "metadata": trace,
        "rows": trace["samples"],
    }
    data.update(
        {
            role: selected(spec, root=root, timely=timely)
            for role, spec in registration["payloads"].items()
        }
    )
    rows = data["rows"]
    raw = data["control"]
    if (
        len(rows) != 301
        or [r["control_tick"] for r in rows] != list(range(301))
        or trace["body_hash"] != MODEL_HASH
        or trace["parameters"] != selected_meta["parameters"]
    ):
        raise ValueError("source trace metadata differs")
    static.exact(
        raw["control_tick"], np.arange(301, dtype=np.int64), "source control clock"
    )
    if not np.allclose(
        raw["physics_time_s"], np.arange(301) * 0.005, atol=2e-8, rtol=0
    ):
        raise ValueError("source physical clock differs")
    for c, row in enumerate(rows):
        row["physics_time_s"] = float(raw["physics_time_s"][c])
        if c:
            static.validate_action(row)
            static.exact(
                np.asarray(row["joint_targets_rad"], np.float64),
                raw["command"][c],
                "source q",
            )
            static.exact(
                np.asarray(
                    [row["tendon_inputs"][leg + "_tarsus"] for leg in LEGS], np.float64
                ),
                raw["tendon"][c],
                "source tendon",
            )
    static.verify_native(data["native"], count=15000)
    return data


def plan(command, reference):
    np, static, oracle = libraries()
    oracle.typed(command, (301, 42), "command")
    z, _ = oracle.encode(command[100:301], reference)
    R = np.linalg.norm(z, axis=2)
    if z.shape != (201, 6, 2) or not np.isfinite(z).all() or not np.all(R >= 0.5):
        raise ValueError("fixed201 six-leg source phase reliability")
    phi = (np.arctan2(z[..., 1], z[..., 0]) / (2 * np.pi)) % 1
    out = {"B": command.copy(), "I": command.copy(), "C": command.copy()}
    for c in range(101, 301):
        x = min(1.0, max(0.0, (c * 0.005 - 0.5) / 0.1))
        weight = x * x * (3 - 2 * x)
        for arm in ("I", "C"):
            phase = phi[c - 100] if arm == "I" else np.full(6, phi[c - 100, 0])
            target = static.cyclic_interpolate(
                reference["curves"], phase[None]
            ).reshape(42)
            out[arm][c] = (
                target if c >= 120 else (1 - weight) * command[c] + weight * target
            )
    return out, {
        "z": z,
        "R": R,
        "phi": phi,
        "fixed_leader": "lf",
        "anatomical_offsets_added": 0,
    }


class HealthStop(RuntimeError):
    pass


class NativeDependencies:
    def __init__(self):
        _, static, _ = libraries()
        self.static = static
        for key in ("body_class", "configure", "install", "boundary"):
            setattr(self, key, getattr(static.Dependencies, key))
        self.action = static.applied_action
        self.counter = static.InstancePhysicsCounter

    @staticmethod
    def body_arguments(data):
        # BancWalkingSession's canonical defaults matter to JSON model identity:
        # slope_degrees is 10.0, while the old E337 dict used integer 10.
        from flylab.body_options import BodyOptions
        from flylab.engine import config_values
        from flylab.physics import PhysicsProfile

        selected_meta = data["body_metadata"]["selected"]
        options = BodyOptions(
            model="flybody",
            actuation="whole_body",
            servo_profile="tracking_all",
            tendons="all",
        )
        return (
            selected_meta["seed"],
            copy.deepcopy(selected_meta["world"]),
            config_values(None),
        ), {
            "optimized": True,
            "initial_pose": None,
            "physics_profile": PhysicsProfile(),
            "body_options": options,
        }

    @staticmethod
    def observed_configuration(body):
        from dataclasses import asdict

        return {
            "config": copy.deepcopy(body.config),
            "body_options": asdict(body.body_options),
            "physics_profile": asdict(body.physics_profile),
            "physics_identity": body.physics_identity(),
            "model_hash": body.model_hash,
            "joint_names": list(body.joint_names),
        }


def publish(file, value):
    raw = json.dumps(value, indent=2, allow_nan=False).encode("utf-8")
    with file.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def update_receipt(file, value):
    """Replace only this new run's receipt, never raw outputs or old evidence."""
    pending = file.with_name(file.name + ".pending")
    publish(pending, value)
    os.replace(pending, file)


def record_error(record, stage, error):
    entry = {"stage": stage, "type": type(error).__name__, "message": str(error)[:1024]}
    record.setdefault("errors", []).append(entry)
    if record.get("first_error") is None:
        record["first_error"] = entry
    return entry


def body_trace(metadata, rows):
    trace = {k: copy.deepcopy(v) for k, v in metadata.items() if k != "samples"}
    trace.update(
        schema="flylab.m3.e348.body-only-trace.v1",
        neural_execution=0,
        source_neural_metadata_is_label_only=True,
        source_trace_schema=metadata.get("schema"),
        samples=rows,
        landmarks={"scope": "actual_native", "count": 8, "order": list(LINKS)},
    )
    trace["source_neural_labels"] = {
        k: metadata.get(k) for k in ("model_hash", "adapter_hash", "graph_hash")
    }
    return trace


def usage(output):
    return sum(f.stat().st_size for f in output.rglob("*") if f.is_file())


def replay(
    arm, command, data, deps, output, *, timely, final_timely=lambda: None, strict=True
):
    np, static, _ = libraries()
    rows, samples = [], []
    ledger = {
        "arm": arm,
        "controls_requested": 0,
        "controls_returned": 0,
        "prefix_checked_before_c101": False,
        "status": "technical_invalid",
        "cleanup_completed": False,
        "first_error": None,
        "errors": [],
        "geometry_order": list(LINKS),
    }
    body = ctor = counter = detach = cpg = None
    began = time.perf_counter()

    def fail(stage, error):
        record_error(ledger, stage, error)

    def observer(sample):
        samples.append(copy.deepcopy(sample))
        static.verify_native(static.native_arrays([sample]))
        if sample.get("numeric_fault"):
            raise ValueError("native numeric fault")
        if sample["nonfoot_contact"] or sample["upright"] < 0.9:
            raise HealthStop("observed native physical health-stop")
        if len(samples) % 50 == 0:
            timely()

    try:
        timely()
        cls = deps.body_class(data)
        body = cls.__new__(cls)
        ctor = deps.counter(body, "constructor", timely=timely)
        ledger["constructor"] = ctor.proof
        args, kwargs = deps.body_arguments(data)
        try:
            cls.__init__(body, *args, **kwargs)
            ledger["constructor_returned"] = True
        finally:
            ctor.restore()
        if strict and (
            ctor.proof["steps_requested"] != 2000
            or ctor.proof["steps_returned"] != 2000
            or not ctor.proof["absolute_clock_known"]
            or abs(ctor.proof["actual_advance_s"] - 0.2) > 1e-9
        ):
            raise ValueError("constructor2000 settling proof differs")
        if hasattr(deps, "observed_configuration"):
            ledger["configuration_before_configure"] = deps.observed_configuration(body)
        ledger["configuration"] = deps.configure(body, data)
        body.restore(copy.deepcopy(data["body_state"]))
        cpg = copy.deepcopy(body.cpg_state())
        counter = deps.counter(body, "science", timely=timely)
        ledger["scientific_physics"] = counter.proof
        detach = deps.install(body, observer)
        rows.append(deps.boundary(body, data["rows"][0]))
        for c in range(1, 301):
            timely()
            if arm != "B" and c == 101:
                static.compare_replay(
                    rows,
                    static.native_arrays(samples),
                    data["rows"],
                    data["native"],
                    100,
                )
                ledger["prefix_checked_before_c101"] = True
            ledger["controls_requested"] += 1
            deps.action(body, data["rows"][c], command[c])
            ledger["controls_returned"] += 1
            rows.append(deps.boundary(body, data["rows"][c]))
            if (
                len(samples) != c * 50
                or abs(float(body.physics_time()) - c * 0.005) > 2e-8
            ):
                raise ValueError("actual native50/control clock differs")
            if rows[-1].get("fault"):
                raise ValueError("boundary fault")
            for leg in LEGS:
                static.typed(
                    np.asarray(rows[-1]["body"]["legs"][leg], np.float64),
                    (8, 3),
                    "<f8",
                    "actual8 geometry",
                )
        if strict and (
            counter.proof["steps_requested"] != 15000
            or counter.proof["steps_returned"] != 15000
        ):
            raise ValueError("requested/returned15000 differs")
        if arm == "B":
            static.compare_replay(
                rows, static.native_arrays(samples), data["rows"], data["native"], 300
            )
            ledger["baseline_full_exact"] = True
        ledger["status"] = "completed"
    except Exception as error:  # noqa: BLE001 -- preserve owned failure/cleanup evidence
        fail("replay", error)
        if (
            isinstance(error, HealthStop)
            and arm != "B"
            and ledger["prefix_checked_before_c101"]
        ):
            ledger["status"] = "scientific_partial"
    finally:
        if body is not None and cpg is not None:
            try:
                if body.cpg_state() != cpg:
                    raise ValueError("CPG advanced")
            except Exception as error:  # noqa: BLE001 -- preserve owned failure/cleanup evidence
                fail("cpg_guard", error)
                ledger["status"] = "technical_invalid"
        closed = True
        for stage, callback in (
            ("detach", detach),
            (
                "counter_restore",
                counter.restore
                if counter and not counter.proof["method_restored"]
                else None,
            ),
            (
                "constructor_restore",
                ctor.restore if ctor and not ctor.proof["method_restored"] else None,
            ),
            ("body_close", getattr(body, "close", None)),
        ):
            if callback:
                try:
                    callback()
                except Exception as error:  # noqa: BLE001 -- preserve owned failure/cleanup evidence
                    fail(stage, error)
                    closed = False
                    ledger["status"] = "technical_invalid"
        ledger["cleanup_completed"] = closed and body is not None
    ledger["native_samples_acquired"] = len(samples)
    ledger["controls_unreturned"] = (
        ledger["controls_requested"] - ledger["controls_returned"]
    )
    ledger["elapsed_s"] = time.perf_counter() - began
    trace = body_trace(data["metadata"], rows)
    native = static.native_arrays(samples)
    try:
        publish(output / "trace.json", trace)
        with (output / "native-contact.npz").open("xb") as stream:
            np.savez_compressed(stream, **native)
        cap = (
            LIMITS["raw_arm_bytes"]
            if ledger["status"] == "completed"
            else LIMITS["failure_bytes"]
        )
        if usage(output) > cap:
            raise ValueError("arm output cap exceeded")
        publish(output / "receipt.json", ledger)
        final_timely()
    except Exception as error:  # noqa: BLE001 -- preserve failed publication counters
        ledger["status"] = "technical_invalid"
        fail("arm_publication", error)
        try:
            update_receipt(output / "receipt.json", ledger)
        except Exception as secondary:  # noqa: BLE001 -- failed sink retained externally
            fail("arm_receipt_publication", secondary)
    return ledger, trace, native


def evaluate(folder, trace, native, reference, *, full):
    np, _, oracle = libraries()
    physics = oracle.physical(native, trace, full=full)
    if not full:
        return {
            "physical": physics,
            "gait": None,
            "smooth": None,
            "shape": None,
            "full_status": "not_tested",
        }
    from tools import analyze_c_banc_biological_gait as gait

    command = np.asarray([r["joint_targets_rad"] for r in trace["samples"]], np.float64)
    return {
        "physical": physics,
        "gait": gait.analyze(gait.load_model(folder / "trace.json"))[0],
        "smooth": oracle.smooth(command, reference),
        "shape": oracle.shape(command, reference["curves"]),
        "full_status": "evaluated",
    }


def hypothesis(metrics, engineering_valid=True):
    if not engineering_valid:
        return {"verdict": "technical_invalid", "supported": None, "gates": None}
    if any(
        metrics.get(k, {}).get("full_status") != "evaluated" for k in ("B", "I", "C")
    ):
        return {"verdict": "not_evaluated", "supported": None, "gates": None}
    i, c = metrics["I"]["physical"], metrics["C"]["physical"]
    gait = metrics["C"]["gait"]
    if any(
        v is None
        for v in (i["forward_mm"], i["slip_mm_s"], c["forward_mm"], c["slip_mm_s"])
    ):
        return {"verdict": "not_evaluated", "supported": None, "gates": None}
    gates = {
        "forward_C_better_I": c["forward_mm"] - i["forward_mm"] > 1e-9,
        "slip_C_better_I": c["slip_mm_s"] - i["slip_mm_s"] < -1e-9,
        "absolute_forward": c["forward_mm"] >= 1.6790085779712631,
        "absolute_slip": c["slip_mm_s"] <= 1.8026436968249955,
        "bout": gait["window"]["longest_forward_bout_s"] >= 0.3,
        "six_cycles": all(gait["events"][leg]["complete_cycles"] > 0 for leg in LEGS),
        "health": c["measured_native_healthy"] is True
        and c["all_control_faults_clear"] is True,
    }
    return {
        "verdict": "supported" if all(gates.values()) else "partial_or_inconclusive",
        "supported": all(gates.values()),
        "gates": gates,
        "I_minus_B": {
            k: i[k] - metrics["B"]["physical"][k]
            if metrics["B"]["physical"][k] is not None
            else None
            for k in ("forward_mm", "slip_mm_s")
        },
        "C_minus_I": {k: c[k] - i[k] for k in ("forward_mm", "slip_mm_s")},
    }


def run_generated(
    data, output, *, dependencies, timely=lambda: None, evaluator=evaluate, strict=False
):
    """Generated-only injected data/body API; fixed300/50/100 science dimensions."""
    return run(
        data,
        output,
        dependencies=dependencies,
        timely=timely,
        evaluator=evaluator,
        strict=strict,
        generated=True,
    )


def run(
    data,
    output,
    *,
    dependencies,
    timely,
    final_timely=lambda: None,
    evaluator=evaluate,
    strict=True,
    generated=False,
):
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "flylab.m3.e348.body-result.v1",
        "generated": generated,
        "arms": {},
        "metrics": {},
        "status": "technical_invalid",
        "historical_compiled_identity": "not_recorded",
        "historical_joint_order": "not_recorded",
        "neural_calls": 0,
        "CPG_advance_calls": 0,
        "candidate_adopted": False,
        "F1_F8_complete": False,
        "biology_proven": False,
        "pure_coupling_causality": False,
        "muscle_force_causality": False,
        "external_completion_qualified": False,
        "analysis_completed": False,
        "first_error": None,
        "errors": [],
    }
    identity = None
    ledger = folder = None
    stage = "planning"
    try:
        np, _, _ = libraries()
        timely()
        commands, encoded = plan(data["control"]["command"], data["reference"])
        stage = "plan_publication"
        with (output / "shape-plan.npz").open("xb") as stream:
            np.savez_compressed(
                stream, **commands, **{k: encoded[k] for k in ("z", "R", "phi")}
            )
        for arm in ("B", "I", "C"):
            ledger = folder = None
            stage = "arm_prepare"
            timely()
            folder = output / arm
            folder.mkdir()
            stage = "arm_replay"
            ledger, trace, native = replay(
                arm,
                commands[arm],
                data,
                dependencies,
                folder,
                timely=timely,
                final_timely=final_timely,
                strict=strict,
            )
            result["arms"][arm] = ledger
            if ledger["status"] == "technical_invalid":
                if ledger.get("first_error"):
                    result["first_error"] = ledger["first_error"]
                    result["errors"].extend(ledger["errors"])
                break
            stage = "arm_identity"
            proof = ledger["configuration"]
            current = {
                k: proof[k]
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
                identity = current
            elif current != identity:
                raise ValueError("actual independent arm identity/order differs")
            stage = "evaluation"
            result["metrics"][arm] = evaluator(
                folder,
                trace,
                native,
                data["reference"],
                full=ledger["status"] == "completed",
            )
            stage = "evaluation_publication"
            publish(folder / "evaluation.json", result["metrics"][arm])
        valid = len(result["arms"]) == 3 and all(
            a["status"] != "technical_invalid" for a in result["arms"].values()
        )
        result["status"] = "completed" if valid else "technical_invalid"
        result["analysis_completed"] = valid
        stage = "hypothesis"
        result["H348"] = hypothesis(result["metrics"], valid)
    except Exception as error:  # noqa: BLE001 -- narrow owned orchestration boundary
        result["status"] = "technical_invalid"
        result["analysis_completed"] = False
        record_error(result, stage, error)
        if ledger is not None:
            ledger["status"] = "technical_invalid"
            record_error(ledger, stage, error)
            try:
                update_receipt(folder / "receipt.json", ledger)
            except Exception as secondary:  # noqa: BLE001 -- sink failure is evidence
                record_error(result, "arm_receipt_publication", secondary)
                record_error(ledger, "arm_receipt_publication", secondary)
        result["H348"] = hypothesis(result["metrics"], False)
    result.setdefault("H348", hypothesis(result["metrics"], False))
    result["timing_elapsed_s"] = time.perf_counter() - ENTRY
    try:
        publish(output / "result.json", result)
        if usage(output) > LIMITS["stage_bytes"]:
            raise ValueError("stage output cap exceeded")
        final_timely()  # Includes writing/stream close; process exit is external.
    except Exception as error:  # noqa: BLE001 -- bounded final publication boundary
        record_error(result, "final_publication", error)
        result["status"] = "technical_invalid"
        result["analysis_completed"] = False
        result["H348"] = hypothesis(result["metrics"], False)
        try:
            update_receipt(output / "result.json", result)
        except Exception as secondary:  # noqa: BLE001 -- unavailable sink is external
            record_error(result, "final_receipt_publication", secondary)
    return result


def make_freeze(qualification, *, root=ROOT):
    registration = metadata_preflight(root)
    own = sources(root)
    qual = read_json(path(qualification, root), cap=256 << 10)
    if (
        qual.get("schema") != "flylab.m3.e348.root-qualification.v1"
        or qual.get("source_sha256") != own
        or qual.get("original_calls") != 0
        or qual.get("passed") is not True
    ):
        raise ValueError("current ROOT generated qualification required")
    checks = (
        "CPU",
        "ruff",
        "selected_extras_unopened",
        "prefix_before_c101",
        "postprefix_health_continues",
        "native_continuity_exact",
        "native8_geometry",
        "native_API",
        "CPG_unchanged",
        "timing_margin_accepted",
    )
    if any(qual.get("checks", {}).get(k) is not True for k in checks):
        raise ValueError("current qualification checks incomplete")
    evidence = qual.get("evidence", [])
    if len(evidence) < 3:
        raise ValueError("native/timing/test execution evidence required")
    for spec in evidence:
        record = checked_json(spec, root)
        if (
            record.get("source_sha256") != own
            or record.get("passed") is not True
            or record.get("original_calls") != 0
        ):
            raise ValueError("current executed evidence differs")
    execution = qual.get("execution")
    if not isinstance(execution, dict) or set(execution) != {
        "contract",
        "supervision",
        "full_exit",
    }:
        raise ValueError("ROOT actual execution receipts required")
    contract, supervision, full_exit = (
        checked_json(execution[k], root)
        for k in ("contract", "supervision", "full_exit")
    )
    if (
        contract.get("original_calls") != 0
        or any(contract.get("source_sha256", {}).get(k) != v for k, v in own.items())
        or supervision.get("schema") != "flylab.m3.e267.root-supervision.v1"
        or supervision.get("success") is not True
        or supervision.get("contract_sha256") != execution["contract"]["sha256"]
        or supervision.get("original_calls") != 0
        or full_exit.get("all_owned_ended") is not True
        or full_exit.get("exit_code") != 0
        or full_exit.get("first_error") is not None
        or not full_exit.get("owned_identities")
        or any(
            x.get("close") != "ended"
            or type(x.get("pid")) is not int
            or not isinstance(x.get("create_time"), (int, float))
            or not math.isfinite(x["create_time"])
            for x in full_exit["owned_identities"]
        )
        or not 0 < full_exit.get("charged_s", math.inf) <= 35
    ):
        raise ValueError("current executed supervision/identity closure differs")
    jobs = contract.get("commands", [])
    attempts = supervision.get("attempts", [])
    if (
        not jobs
        or len(jobs) != len(attempts)
        or any(
            a.get("argv") != j["argv"]
            or a.get("exit_code") != 0
            or a.get("killed") is not False
            for j, a in zip(jobs, attempts)
        )
    ):
        raise ValueError("actual current generated argv/exit differs")
    return {
        "schema": "flylab.m3.e348.body-freeze.v1",
        "source_sha256": own,
        "static_sha256": PINS,
        "limits": LIMITS,
        "registration_sha256": PINS[REGISTRATION],
        "qualification": {
            "path": qualification,
            "bytes": path(qualification, root).stat().st_size,
            "sha256": digest(path(qualification, root).read_bytes()),
        },
        "input_metadata": registration,
        "original_authorized": False,
    }


def original_gate(freeze_file, review_file, *, root=ROOT, expected_output=OUTPUT):
    registration = metadata_preflight(root)
    freeze = read_json(path(freeze_file, root))
    current = make_freeze(freeze["qualification"]["path"], root=root)
    if freeze != current:
        raise ValueError("current freeze differs")
    review = read_json(path(review_file, root), cap=256 << 10)
    if (
        review.get("schema") != "flylab.m3.e348.root-original-review.v1"
        or review.get("approved") is not True
        or review.get("freeze_sha256") != digest(path(freeze_file, root).read_bytes())
        or review.get("source_sha256") != sources(root)
        or review.get("output") != expected_output
        or expected_output != OUTPUT
    ):
        raise ValueError("ROOT original review differs")
    reservation = checked_json(review["reservation"], root)
    expected = {
        "schema": "flylab.m3.e348.root-reservation.v1",
        "cumulative_before_s": 9780.954023399501,
        "cumulative_limit_s": 9900,
        "reserved_s": 103,
        "body_wall_s": 95,
        "audit_wall_s": 8,
        "original_calls": 1,
        "retry_count": 0,
        "output": OUTPUT,
        "source_sha256": sources(root),
    }
    if (
        any(
            type(reservation.get(k)) is not type(v) or reservation.get(k) != v
            for k, v in expected.items()
        )
        or reservation.get("freeze_sha256") != review["freeze_sha256"]
    ):
        raise ValueError("exact current reservation required")
    argv = [
        ".venv/Scripts/python.exe",
        "-X",
        "utf8",
        "-B",
        OWNED[0],
        "--run",
        "--freeze",
        freeze_file,
        "--review",
        review_file,
        "--output",
        OUTPUT,
    ]
    if (
        reservation.get("argv") != argv
        or reservation.get("review_path") != review_file
        or reservation.get("reflection_sha256") != PINS[REFLECTION]
    ):
        raise ValueError("sole original argv/current review linkage differs")
    if (
        path(expected_output, root).exists()
        or shutil.disk_usage(root).free < LIMITS["free_disk_bytes"]
    ):
        raise ValueError("fresh output/free disk required")
    if any(
        os.environ.get(k) != "1"
        for k in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    ):
        raise ValueError("CPU1 environment required before imports")
    return registration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--qualification")
    parser.add_argument("--review")
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)

    def timely(final=False):
        if time.perf_counter() - ENTRY > LIMITS["internal_s" if final else "science_s"]:
            raise TimeoutError("E348 inclusive publication/science deadline")
        import psutil

        if psutil.Process().memory_info().rss > LIMITS["host_bytes"]:
            raise MemoryError("E348 host2GiB")

    if not args.run:
        if not args.qualification:
            parser.error("passive freeze requires --qualification")
        publish(path(args.freeze), make_freeze(args.qualification))
        timely(True)
        return 0
    registration = original_gate(args.freeze, args.review, expected_output=args.output)
    timely()
    try:
        data = load_original(registration, timely=timely)
    except Exception as error:  # noqa: BLE001 -- input rejection precedes any body allocation
        folder = path(args.output)
        folder.mkdir(parents=True, exist_ok=False)
        failure = {
            "schema": "flylab.m3.e348.input-failure.v1",
            "status": "technical_invalid",
            "stage": "input_load",
            "neural_calls": 0,
            "body_constructors": 0,
            "scientific_physics_steps": 0,
            "array_read_counts": None,
            "array_read_counts_status": "not_instrumented",
            "candidate_adopted": False,
            "F1_F8_complete": False,
            "first_error": None,
            "errors": [],
        }
        record_error(failure, "input_load", error)
        publish(folder / "result.json", failure)
        timely(True)
        print(json.dumps(failure, allow_nan=False), file=sys.stderr)
        return 1
    result = run(
        data,
        path(args.output),
        dependencies=NativeDependencies(),
        timely=timely,
        final_timely=lambda: timely(True),
    )
    timely(True)
    if result["status"] != "completed":
        print(
            json.dumps(
                {
                    "schema": "flylab.m3.e348.failure-summary.v1",
                    "first_error": result["first_error"],
                    "errors": result["errors"],
                },
                allow_nan=False,
            ),
            file=sys.stderr,
        )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
