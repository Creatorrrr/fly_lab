"""E353 sole reference cadence R; teacher diagnostic, never autonomous gait."""

from __future__ import annotations

import time

ENTRY = time.perf_counter()
import argparse
import copy
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = "verification/banc-natural-gait-m3-20260919"
OWNED = (
    "tools/probe_c_banc_m3_reference_cadence_body.py",
    "tests/test_c_banc_reference_cadence_body.py",
)
REGISTRATION = BASE + "/parent-e353-cadence-input-registration.json"
REG_SHA = "20c1eca01023101b720716c26d014e29985bf79f06e60a78971a6b42c6a3afbe"
REFLECTION = BASE + "/parent-e353-reflected-design-registration.json"
REF_SHA = "086c5a18c3c63d7c90308f0729ec7af0c8cd3f49f3981bc2de684cf69657397c"
OUTPUT = BASE + "/e353-reference-cadence-body"
AUDIT_OUTPUT = BASE + "/e353-reference-cadence-audit"
FREQUENCY = 2.382760007990865
CAP = 4 << 20
EMERGENCY = 64 << 10
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
CHECKS = (
    "CPU",
    "ruff",
    "plan",
    "row0_prefix",
    "dynamic_health",
    "full_synthetic_pack_audit",
    "gait_events",
    "gzip_npz",
    "publication_caps",
    "late_failure",
    "private_gates",
    "standalone",
)
_MODULES = {}


def require(ok, label):
    if not ok:
        raise ValueError(label)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def module(name, digest):
    require(sha((ROOT / name).read_bytes()) == digest, "frozen PY source " + name)
    if name not in _MODULES:
        spec = importlib.util.spec_from_file_location(
            "e353_" + Path(name).stem, ROOT / name
        )
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        _MODULES[name] = result
    return _MODULES[name]


def primitives():
    return module(
        "tools/recover_c_banc_m3_source_pose_report.py",
        "a9a5804b9b48adffb41c43c91931c5d1664b43e17624f1f0c70478a7105c1a04",
    )


def pose():
    return module(
        "tools/verify_c_banc_m3_source_pose_body.py",
        "d488c305dd8e8a11922c4543d78557117638d818d2a5e0b67c986616d66666c1",
    )


def own_sources(root=ROOT):
    return {n: sha(primitives().safe(n, root).read_bytes()) for n in OWNED}


def descriptor(name, raw):
    return {"path": name, "bytes": len(raw), "sha256": sha(raw)}


def json_bytes(value):
    return json.dumps(
        primitives().lower(value), separators=(",", ":"), allow_nan=False
    ).encode("utf8")


def metadata_preflight(root=ROOT, reader=None):
    p = primitives()
    r = reader or p.EvidenceReader(root)
    reg = r.json({"path": REGISTRATION, "bytes": 51249, "sha256": REG_SHA})
    f = p.safe(REFLECTION, root)
    reflection = r.json(
        {"path": REFLECTION, "bytes": f.stat().st_size, "sha256": REF_SHA}
    )
    require(
        reg["schema"] == "flylab.m3.e353.cadence-input-registration.v1"
        and reflection["registration"]["sha256"] == REG_SHA
        and reflection["original_authorized"] is False,
        "E353 reflected registry",
    )
    for s in (reg["design"], reg["actual_review"], reg["review_md"]):
        r.bytes(s)
    for name, pin in reg["consumed_sources"].items():
        require(name.endswith(".py"), "finite PY closure")
        f = p.safe(name, root)
        r.bytes({"path": name, "bytes": f.stat().st_size, "sha256": pin})
    require(
        set(reg["allowed_original_npz_paths"])
        == {s["path"] for s in reg["payloads"].values()}
        and set(reg["payloads"]) == {"control", "native", "reference"},
        "exact original3",
    )
    for s in list(reg["payloads"].values()) + reg["finite_JSON_dossier"]:
        require(
            p.safe(s["path"], root).stat().st_size == s["bytes"], "stat-only dossier"
        )
    inherited = {n: r.json(s) for n, s in reg["inherited_provenance"].items()}
    final = inherited[BASE + "/parent-e351-final-decision.json"]
    report = inherited[final["report"]["path"]]
    identity = inherited[final["identity"]["path"]]
    require(
        final["diagnostic_completion_qualified"] is True
        and final["H348"] == reg["inherited_H348"] == report["diagnostic_H348"]
        and final["verified_metrics_summary"]["C"] == reg["historical_C"]
        and report["external_completion_qualified"] is False
        and report["body_evidence_qualified"] is False
        and identity["qualified"] is True
        and identity["all_owned_pids_ended"] is True
        and identity["errors"] == [],
        "qualified historical C; unchanged provisional scopes",
    )
    require(
        reg["new_raw_roster"]
        == [
            "trace.json.gz",
            "native-contact.npz",
            "source-pose.npz",
            "cadence-plan.npz",
        ]
        and len(reg["worker_output_roster"]) == 8
        and reg["cadence_policy"]["frequency_Hz"] == FREQUENCY,
        "fixed R roster/frequency",
    )
    require(
        r.json(reg["force_policy"]) == reg["force_policy_value"]
        and r.json(reg["observation_policy"]) == reg["observation_policy_value"],
        "fixed policies",
    )
    return {
        "registration": reg,
        "reader": r,
        "inherited": inherited,
        "authorized": False,
    }


def check_execution(
    reader,
    execution,
    own,
    cap,
    *,
    expected_original=0,
    expected_argv=None,
    expected_exit=0,
):
    contract, supervision, identity, full = (
        reader.json(execution[k])
        for k in ("contract", "supervision", "identity", "full_exit")
    )
    require(
        all(contract["source_sha256"].get(k) == v for k, v in own.items())
        and contract["original_calls"] == expected_original
        and supervision["contract_sha256"] == execution["contract"]["sha256"]
        and supervision["original_calls"] == expected_original
        and supervision["success"] is (expected_exit == 0)
        and len(contract["commands"]) == len(supervision["attempts"]) == 1
        and supervision["attempts"][0]["argv"] == contract["commands"][0]["argv"]
        and supervision["attempts"][0]["exit_code"] == expected_exit
        and supervision["attempts"][0]["killed"] is False
        and supervision["max_tree_rss_bytes"] <= 2 << 30,
        "executed current source argv caps",
    )
    if expected_argv is not None:
        require(
            contract["commands"][0]["argv"] == expected_argv, "sealed worker sole argv"
        )
    require(
        identity["qualified"] is True
        and identity["actual_supervisor_exit_code"] == expected_exit
        and identity["all_owned_pids_ended"] is True
        and identity["errors"] == []
        and identity["contract"] == execution["contract"]
        and identity["supervision"] == execution["supervision"]
        and identity["owned_pids"]
        and all(
            identity["present_after_return"][str(pid)] is False
            and type(identity["pid_create_times"][str(pid)]) in (float, int)
            and math.isfinite(identity["pid_create_times"][str(pid)])
            for pid in identity["owned_pids"]
        ),
        "actual live PID/create-time ended",
    )
    wall = full["full_exit_elapsed_s"]
    require(
        full["schema"] == "flylab.m3.e353.actual-full-return.v1"
        and full["scope"] == "prevalidation-through-observer-final-exit"
        and full["exit_code"] == expected_exit
        and full["all_owned_ended"] is True
        and full["known_identity_cleanup"] == []
        and full["source_sha256"] == own
        and full["identity"] == execution["identity"]
        and full["contract"] == execution["contract"]
        and full["supervision"] == execution["supervision"]
        and type(wall) in (float, int)
        and math.isfinite(wall)
        and 0 < wall <= cap
        and full["charged_s"] == wall
        and full["actual_outer_full_return_s"] == wall
        and full["first_error"] is None
        and full["secondary_errors"] == [],
        "measured outer full-return, not narrow elapsed",
    )
    return contract, supervision, identity, full


def make_freeze(qualification, root=ROOT, context=None):
    ctx = context or metadata_preflight(root)
    r = ctx["reader"]
    file = primitives().safe(qualification, root)
    s = descriptor(qualification, file.read_bytes())
    q = r.json(s)
    own = own_sources(root)
    require(
        q["schema"] == "flylab.m3.e353.root-qualification.v1"
        and q["source_sha256"] == own
        and q["passed"] is True
        and type(q["original_calls"]) is int
        and q["original_calls"] == 0
        and all(q["checks"].get(k) is True for k in CHECKS),
        "current ROOT qualification",
    )
    check_execution(r, q["execution"], own, 20)
    return {
        "schema": "flylab.m3.e353.cadence-freeze.v1",
        "source_sha256": own,
        "registration_sha256": REG_SHA,
        "reflection_sha256": REF_SHA,
        "qualification": s,
        "design": ctx["registration"]["design"],
        "consumed_sources": ctx["registration"]["consumed_sources"],
        "prospective": ctx["registration"]["prospective"],
        "original_authorized": False,
    }


def original_argv(mode, freeze, review):
    return [
        ".venv/Scripts/python.exe",
        "-X",
        "utf8",
        "-B",
        OWNED[0],
        "--" + mode,
        "--freeze",
        freeze,
        "--review",
        review,
        "--output",
        OUTPUT if mode == "run" else AUDIT_OUTPUT,
    ]


def gate(mode, freeze_name, review_name, *, root=ROOT, output=None, reader=None):
    require(mode in ("run", "audit"), "sole mode")
    ctx = metadata_preflight(root, reader)
    r = ctx["reader"]
    p = primitives()

    def read(name):
        f = p.safe(name, root)
        return r.json({"path": name, "bytes": f.stat().st_size, "sha256": None})

    freeze, review = read(freeze_name), read(review_name)
    require(
        freeze == make_freeze(freeze["qualification"]["path"], root, ctx),
        "exact current freeze",
    )
    own = own_sources(root)
    target = OUTPUT if mode == "run" else AUDIT_OUTPUT
    require(
        output == target
        and review["schema"] == "flylab.m3.e353.root-original-review.v1"
        and review["mode"] == mode
        and review["approved"] is True
        and review["source_sha256"] == own
        and review["freeze_sha256"] == r.digests[freeze_name]
        and review["registration_sha256"] == REG_SHA
        and review["reflection_sha256"] == REF_SHA
        and review["output"] == target,
        "own private gate review",
    )
    reserve = r.json(review["reservation"])
    cap = 25 if mode == "run" else 6
    expected = {
        "schema": "flylab.m3.e353.root-reservation.v1",
        "mode": mode,
        "cumulative_limit_s": 9900,
        "reserved_s": cap,
        "wall_s": cap,
        "internal_s": 22 if mode == "run" else 5,
        "original_calls": 1,
        "retry_count": 0,
        "body_calls": 1 if mode == "run" else 0,
        "neural_calls": 0,
        "CPG_calls": 0,
        "GPU_calls": 0,
        "source_sha256": own,
        "freeze_sha256": r.digests[freeze_name],
        "registration_sha256": REG_SHA,
        "reflection_sha256": REF_SHA,
        "review_path": review_name,
        "output": target,
        "argv": original_argv(mode, freeze_name, review_name),
    }
    require(
        all(
            type(reserve.get(k)) is type(v) and reserve.get(k) == v
            for k, v in expected.items()
        ),
        "exact mode reservation argv",
    )
    if mode == "run":
        require(
            type(reserve["cumulative_before_s"]) is float
            and reserve["cumulative_before_s"] == 9846.861104799502,
            "current cumulative run",
        )
    else:
        sealed = r.json(review["worker_seal"])
        require(
            sealed["schema"] == "flylab.m3.e353.closed-worker-seal.v1"
            and sealed["source_sha256"] == own
            and sealed["registration_sha256"] == REG_SHA
            and sealed["output"] == OUTPUT,
            "sealed same current worker",
        )
        require(
            type(sealed["expected_exit_code"]) is int
            and sealed["expected_exit_code"] in (0, 1),
            "closed worker expected exit",
        )
        _, _, _, full = check_execution(
            r,
            sealed["execution"],
            own,
            25,
            expected_original=1,
            expected_argv=original_argv(
                "run", sealed["freeze_path"], sealed["review_path"]
            ),
            expected_exit=sealed["expected_exit_code"],
        )
        require(
            type(reserve["cumulative_before_s"]) is float
            and reserve["cumulative_before_s"] == 9846.861104799502 + full["charged_s"]
            and reserve["cumulative_before_s"] + 6 <= 9900,
            "once charged worker cumulative",
        )
        ctx["worker_seal"] = sealed
    require(
        not p.safe(target, root).exists()
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
        "fresh output/CPU1/disk",
    )
    ctx["authorized"] = True
    ctx["mode"] = mode
    ctx["review"] = review
    r.array_authorized = True
    return ctx


def load_inputs(ctx):
    require(ctx["authorized"], "private gate before scientific bytes")
    r, reg = ctx["reader"], ctx["registration"]
    p = primitives()
    payload = {}
    for role, s in reg["payloads"].items():
        payload[role] = r.npz(
            s,
            {
                k: (tuple(v["shape"]), v["dtype"])
                for k, v in s["selected_members"].items()
            },
            selected=s["selected_members"],
        )
    specs = {s["path"]: s for s in reg["finite_JSON_dossier"]}

    def read(tail):
        return r.json(specs[BASE + "/" + tail])

    state = read("e335-smooth-secant-body/observed-start/state.json")
    trace = read("e335-smooth-secant-body/trace.json")
    meta = read("parent-e337-current-body-metadata.json")
    manifest = read("e335-smooth-secant-body/observed-start/manifest.json")
    rows = pose().force().validate_inputs(state, trace, meta, reg["force_policy_value"])
    np = p.numpy()
    core = pose().shared()
    _, static, _ = core.libraries()
    require(manifest["versions"] == static._versions(), "fixed native versions")
    p.exact(
        payload["control"]["control_tick"],
        np.arange(301, dtype=np.int64),
        "control indices",
    )
    require(
        np.allclose(
            payload["control"]["physics_time_s"],
            np.arange(301) * 0.005,
            atol=2e-8,
            rtol=0,
        ),
        "control clock",
    )
    for c, row in enumerate(rows):
        require(abs(row["time_s"] - c * 0.005) <= 2e-8, "source boundary clock")
        row["physics_time_s"] = float(payload["control"]["physics_time_s"][c])
        if c:
            p.exact(
                np.asarray(row["joint_targets_rad"], np.float64),
                payload["control"]["command"][c],
                "original c>=1 command",
            )
            p.exact(
                np.asarray(
                    [row["tendon_inputs"][leg + "_tarsus"] for leg in LEGS], np.float64
                ),
                payload["control"]["tendon"][c],
                "original tendon",
            )
    static.verify_native(payload["native"], count=15000)
    return dict(
        payload, rows=rows, metadata=trace, body_state=state["body"], body_metadata=meta
    )


def build_plan(control, reference):
    np = primitives().numpy()
    q = np.asarray(control["command"])
    curves = reference["curves"]
    basis = reference["phase_basis"]
    z = np.zeros((201, 6, 2))
    reliability = np.zeros((201, 6))
    source_phi = np.zeros((201, 6))
    for c in range(100, 301):
        for leg in range(6):
            logits = (
                -np.square(curves[leg] - q[c, leg * 7 : leg * 7 + 7]).sum(axis=1)
                / reference["temperature"][leg]
            )
            weights = np.exp(logits - logits.max())
            weights /= weights.sum()
            point = weights @ basis
            z[c - 100, leg] = point
            reliability[c - 100, leg] = np.linalg.norm(point)
            source_phi[c - 100, leg] = (np.arctan2(point[1], point[0]) / math.tau) % 1
    require(
        np.isfinite(z).all() and (reliability >= 0.5).all(),
        "all original source reliability>=.5",
    )
    ticks = np.arange(301, dtype=np.int64)
    times = ticks * 0.005
    phi = (source_phi[0, 0] + FREQUENCY * (np.arange(100, 301) * 0.005 - 0.5)) % 1
    weights = np.zeros(301)
    commands = q.copy()
    for c in range(101, 301):
        x = min(1.0, max(0.0, (c * 0.005 - 0.5) / 0.1))
        weights[c] = x * x * (3 - 2 * x)
        position = phi[c - 100] * 128
        lo = math.floor(position) % 128
        fraction = position - math.floor(position)
        target = (
            (1 - fraction) * curves[:, lo] + fraction * curves[:, (lo + 1) % 128]
        ).reshape(42)
        commands[c] = (
            target if c >= 120 else (1 - weights[c]) * q[c] + weights[c] * target
        )
    return {
        "command": commands,
        "control_tick": ticks,
        "control_time_s": times,
        "source_z": z,
        "source_R": reliability,
        "source_phi": source_phi,
        "source_control_tick": np.arange(100, 301, dtype=np.int64),
        "phi_R": phi,
        "blend_weight": weights,
    }


def audit_plan(control, reference, plan):
    """Independent scalar Gaussian/fsum and own R interpolation; no producer builder."""
    np = primitives().numpy()
    p = primitives()
    q = control["command"]
    curves = reference["curves"]
    basis = reference["phase_basis"]
    anchor = None
    maximum = 0.0
    p.exact(plan["control_tick"], np.arange(301, dtype=np.int64), "plan control ticks")
    p.exact(
        plan["source_control_tick"],
        np.arange(100, 301, dtype=np.int64),
        "source phase ticks",
    )
    p.exact(plan["control_time_s"], np.arange(301) * 0.005, "plan clock")
    p.exact(plan["command"][:101], q[:101], "literal source plan prefix")
    for c in range(100, 301):
        for leg in range(6):
            squared = [
                math.fsum(
                    (float(curves[leg, k, j]) - float(q[c, leg * 7 + j])) ** 2
                    for j in range(7)
                )
                for k in range(128)
            ]
            logits = [-v / float(reference["temperature"][leg]) for v in squared]
            top = max(logits)
            weights = [math.exp(v - top) for v in logits]
            denominator = math.fsum(weights)
            point = [
                math.fsum(w * float(basis[k, j]) for k, w in enumerate(weights))
                / denominator
                for j in range(2)
            ]
            radius = math.hypot(*point)
            phase = (math.atan2(point[1], point[0]) / math.tau) % 1
            require(
                radius >= 0.5
                and np.allclose(
                    plan["source_z"][c - 100, leg], point, atol=2e-12, rtol=0
                )
                and abs(plan["source_R"][c - 100, leg] - radius) <= 2e-12
                and abs((plan["source_phi"][c - 100, leg] - phase + 0.5) % 1 - 0.5)
                <= 2e-12,
                "independent source encoder",
            )
            if c == 100 and leg == 0:
                anchor = phase
        phase = (anchor + FREQUENCY * (c * 0.005 - 0.5)) % 1
        require(
            abs((plan["phi_R"][c - 100] - phase + 0.5) % 1 - 0.5) <= 2e-12,
            "independent R phase",
        )
        if c == 100:
            continue
        x = min(1.0, max(0.0, (c * 0.005 - 0.5) / 0.1))
        weight = x * x * (3 - 2 * x)
        position = phase * 128
        k = math.floor(position)
        fraction = position - k
        target = [
            (1 - fraction) * float(curves[leg, k % 128, j])
            + fraction * float(curves[leg, (k + 1) % 128, j])
            for leg in range(6)
            for j in range(7)
        ]
        expected = (
            np.array(target)
            if c >= 120
            else np.array(
                [(1 - weight) * float(q[c, j]) + weight * target[j] for j in range(42)]
            )
        )
        error = float(np.abs(plan["command"][c] - expected).max())
        maximum = max(maximum, error)
        require(
            error <= 2e-12 and abs(plan["blend_weight"][c] - weight) <= 2e-12,
            "independent interpolation/blend",
        )
    require(np.array_equal(plan["blend_weight"][:101], np.zeros(101)), "prefix blend0")
    return {"anchor": float(anchor), "maximum_command_error_rad": maximum}


def loop(data, plan, deps, *, timely=lambda: None):
    """One R orchestration; no old replay/run/publisher/global patch."""
    np = primitives().numpy()
    core = pose().shared()
    static = deps.static
    rows, samples = [], []
    receipt = {
        "schema": "flylab.m3.e353.arm-receipt.v1",
        "experiment_arm": "R",
        "dependency_arm": "B",
        "status": "technical_invalid",
        "controls_requested": 0,
        "controls_returned": 0,
        "prefix_checked_before_c101": False,
        "constructor_returned": False,
        "cleanup_completed": False,
        "first_error": None,
        "errors": [],
    }
    body = ctor = counter = detach = cpg = None

    def fail(stage, error):
        item = pose().error_record(stage, error)
        receipt["errors"].append(item)
        if receipt["first_error"] is None:
            receipt["first_error"] = item

    def observe(sample):
        samples.append(copy.deepcopy(sample))
        static.verify_native(static.native_arrays([sample]))
        require(not sample.get("numeric_fault"), "native numeric fault")
        if sample["nonfoot_contact"] or sample["upright"] < 0.9:
            raise core.HealthStop("actual source native health")
        if len(samples) % 50 == 0:
            timely()

    try:
        timely()
        cls = deps.body_class(data)
        body = cls.__new__(cls)
        ctor = deps.counter(body, "constructor", timely=timely)
        receipt["constructor"] = ctor.proof
        args, kwargs = deps.body_arguments(data)
        try:
            cls.__init__(body, *args, **kwargs)
            receipt["constructor_returned"] = True
        finally:
            ctor.restore()
        require(
            ctor.proof["steps_requested"] == ctor.proof["steps_returned"] == 2000
            and ctor.proof["absolute_clock_known"]
            and abs(ctor.proof["actual_advance_s"] - 0.2) <= 1e-9,
            "constructor2000",
        )
        receipt["configuration_before_configure"] = deps.observed_configuration(body)
        receipt["configuration"] = deps.configure(body, data)
        body.restore(copy.deepcopy(data["body_state"]))
        cpg = copy.deepcopy(body.cpg_state())
        counter = deps.counter(body, "science", timely=timely)
        receipt["scientific_physics"] = counter.proof
        detach = deps.install(body, observe)
        rows.append(deps.boundary(body, data["rows"][0]))
        static.compare_replay(
            rows, static.native_arrays(samples), data["rows"], data["native"], 0
        )
        for c in range(1, 301):
            timely()
            if c == 101:
                static.compare_replay(
                    rows,
                    static.native_arrays(samples),
                    data["rows"],
                    data["native"],
                    100,
                )
                receipt["prefix_checked_before_c101"] = True
            receipt["controls_requested"] += 1
            deps.action(body, data["rows"][c], plan["command"][c])
            receipt["controls_returned"] += 1
            rows.append(deps.boundary(body, data["rows"][c]))
            require(
                len(samples) == c * 50
                and abs(float(body.physics_time()) - c * 0.005) <= 2e-8,
                "native50/control clock",
            )
            require(not rows[-1].get("fault"), "boundary fault")
            for leg in LEGS:
                static.typed(
                    np.asarray(rows[-1]["body"]["legs"][leg], np.float64),
                    (8, 3),
                    "<f8",
                    "actual8 geometry",
                )
        require(
            counter.proof["steps_requested"]
            == counter.proof["steps_returned"]
            == 15000,
            "science15000",
        )
        receipt["status"] = "completed"
    except Exception as error:  # noqa: BLE001 -- actual acquisition and dynamic health identity
        fail("loop", error)
        if isinstance(error, core.HealthStop) and receipt["prefix_checked_before_c101"]:
            records = deps.observations.get("B", [])
            finite = all(
                r["time_available"]
                and r["source_available"]
                and r["current_available"]
                and all(
                    math.isfinite(r[k])
                    for k in ("time_s", "source_upright", "current_upright")
                )
                for r in records
            )
            force_clean = not deps.before_read_errors and all(
                e["after"] is not None and e["after"] == e["before"]
                for e in deps.actions
            )
            if finite and force_clean:
                receipt["status"] = "scientific_partial"
    finally:
        if body is not None and cpg is not None:
            try:
                require(body.cpg_state() == cpg, "CPG unchanged")
            except Exception as error:  # noqa: BLE001 -- preserve actual acquisition/cleanup failures
                fail("CPG", error)
                receipt["status"] = "technical_invalid"
        closed = body is not None
        for stage, callback in [
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
            ("close", getattr(body, "close", None)),
        ]:
            if callback:
                try:
                    callback()
                except Exception as error:  # noqa: BLE001 -- preserve actual acquisition/cleanup failures
                    fail(stage, error)
                    closed = False
                    receipt["status"] = "technical_invalid"
        receipt["cleanup_completed"] = closed
    records = deps.observations.get("B", [])
    receipt.update(
        boundaries_acquired=len(rows),
        native_samples_acquired=len(samples),
        pose_records_acquired=len(records),
        controls_unreturned=receipt["controls_requested"]
        - receipt["controls_returned"],
        pose_callback_attempted=sum(r["callback_attempted"] for r in records),
        pose_callback_returned=sum(r["callback_returned"] for r in records),
        force_counters={
            "adapter_requested": deps.action_adapter_requested,
            "native_action_requested": deps.native_action_requested,
            "native_action_returned": deps.native_action_returned,
        },
        force_before_read_errors=deps.before_read_errors,
        held_force_receipt={
            "schema": "flylab.m3.e349.held-force-receipt.v1",
            "policy_sha256": deps.policy_sha256,
            "actions": deps.actions,
        },
        source_pose={
            "experiment_arm": "R",
            "dependency_arm": "B",
            "policy": deps.observation_policy,
            "count": len(records),
            "first": records[0] if records else None,
            "last": records[-1] if records else None,
            "errors": [
                {"native_index": r["native_index"], "errors": r["errors"]}
                for r in records
                if r["errors"]
            ],
            "source_available_count": sum(r["source_available"] for r in records),
            "current_available_count": sum(r["current_available"] for r in records),
            "time_available_count": sum(r["time_available"] for r in records),
            "orientation_replayed_by_independent_audit": False,
        },
    )
    trace = core.body_trace(data["metadata"], rows)
    trace.update(experiment_arm="R", dependency_arm="B")
    observed = {
        k: np.array(
            [r[k] if r[k] is not None else np.nan for r in records], dtype=np.float64
        )
        for k in ("time_s", "source_upright", "current_upright")
    }
    observed["native_index"] = np.array(
        [r["native_index"] for r in records], dtype=np.int64
    )
    for k in ("time_available", "source_available", "current_available"):
        observed[k] = np.array([r[k] for r in records], dtype=bool)
    return receipt, trace, static.native_arrays(samples), observed


def full_contract(receipt, trace, native, observed):
    np = primitives().numpy()
    require(
        receipt["status"] == "completed"
        and receipt["cleanup_completed"] is True
        and receipt["constructor_returned"] is True
        and receipt["prefix_checked_before_c101"] is True
        and receipt["experiment_arm"] == "R"
        and receipt["dependency_arm"] == "B"
        and len(trace["samples"]) == 301
        and receipt["controls_requested"] == receipt["controls_returned"] == 300
        and receipt["boundaries_acquired"] == 301
        and receipt["native_samples_acquired"]
        == receipt["pose_records_acquired"]
        == receipt["pose_callback_attempted"]
        == receipt["pose_callback_returned"]
        == 15000
        and receipt["errors"] == [],
        "full actual R counts/cleanup",
    )
    for key, count in (("constructor", 2000), ("scientific_physics", 15000)):
        proof = receipt[key]
        require(
            proof["steps_requested"] == proof["steps_returned"] == count
            and proof["method_restored"] is True
            and proof["absolute_clock_known"] is True
            and abs(proof["actual_advance_s"] - count * 0.0001) <= 2e-8,
            "full counters",
        )
    require(
        receipt["force_counters"]
        == {
            "adapter_requested": 300,
            "native_action_requested": 300,
            "native_action_returned": 300,
        }
        and receipt["force_before_read_errors"] == []
        and len(receipt["held_force_receipt"]["actions"]) == 300,
        "actual force counts",
    )
    for c, event in enumerate(receipt["held_force_receipt"]["actions"], 1):
        require(
            event["arm"] == "B"
            and event["control_tick"] == c
            and event["returned"] is True
            and event["error"] is None
            and event["before"]
            == event["after"]
            == {"push_left": 0.0, "push_force": [0.0, 0.0, 0.0]},
            "actual force observations",
        )
    validate_native(native)
    require(
        set(observed)
        == {
            "native_index",
            "time_s",
            "source_upright",
            "current_upright",
            "time_available",
            "source_available",
            "current_available",
        },
        "observer roster",
    )
    for k, v in observed.items():
        dtype = np.dtype(
            "int64"
            if k == "native_index"
            else "bool"
            if k.endswith("_available")
            else "float64"
        )
        require(
            v.shape == (15000,) and v.dtype == dtype, "actual observer full typed count"
        )
        if k.endswith("_available"):
            require(v.all(), "all observer available")
        else:
            require(np.isfinite(v).all(), "finite full observation")
    require(
        np.array_equal(observed["native_index"], np.arange(1, 15001))
        and np.array_equal(observed["time_s"], native["time_s"])
        and np.array_equal(observed["source_upright"], native["upright"]),
        "actual observation index/time/source",
    )
    require(
        all(
            np.isfinite(np.asarray(r["body"]["legs"][leg])).all()
            and np.asarray(r["body"]["legs"][leg]).shape == (8, 3)
            for r in trace["samples"]
            for leg in LEGS
        ),
        "actual full8 geometry",
    )
    return bool(
        (observed["source_upright"] >= 0.9).all()
        and (observed["current_upright"] >= 0.9).all()
        and not native["nonfoot_contact"].any()
        and all(r["fault"] is None for r in trace["samples"])
    )


def validate_native(native, count=15000):
    np = primitives().numpy()
    require(
        set(native)
        == {
            "time_s",
            "normal_load_bw",
            "slip_mm_s",
            "slip_valid",
            "upright",
            "nonfoot_contact",
        },
        "native roster",
    )
    for k, v in native.items():
        require(
            v.shape
            == (
                (count, 6)
                if k in ("normal_load_bw", "slip_mm_s", "slip_valid")
                else (count,)
            )
            and v.dtype
            == np.dtype(
                "bool" if k in ("slip_valid", "nonfoot_contact") else "float64"
            ),
            "native dtype/shape",
        )
        require(
            not np.isinf(v).any() and (k == "slip_mm_s" or np.isfinite(v).all()),
            "native nonfinite",
        )
    require(
        np.allclose(
            native["time_s"], np.arange(1, count + 1) * 0.0001, atol=2e-8, rtol=0
        )
        and np.array_equal(native["slip_valid"], np.isfinite(native["slip_mm_s"]))
        and (native["normal_load_bw"] >= 0).all(),
        "native clock/masks/load",
    )


def evaluate(receipt, trace, native, observed, file, digest):
    if receipt["status"] != "completed":
        return {
            "full_status": "not_tested",
            "physical": None,
            "gait": None,
            "H353": {"verdict": "not_evaluated", "supported": None, "gates": None},
        }
    health = full_contract(receipt, trace, native, observed)
    p = primitives()
    np = p.numpy()
    rows = trace["samples"]
    origin = np.asarray(rows[0]["body"]["basis"], np.float64)[[0, 2], 0]
    origin /= np.linalg.norm(origin)
    forward = math.fsum(
        float(v) * float(w)
        for v, w in zip(
            np.asarray(rows[299]["body"]["position"])[[0, 2]]
            - np.asarray(rows[100]["body"]["position"])[[0, 2]],
            origin,
        )
    )
    window = (native["time_s"] >= 0.5) & (native["time_s"] < 1.5)
    support = (native["normal_load_bw"] >= 0.02) & window[:, None]
    denominator = math.fsum(float(v) for v in native["normal_load_bw"][support])
    missing = int((support & ~native["slip_valid"]).sum())
    slip = (
        None
        if not denominator or missing
        else math.fsum(
            float(a) * float(b)
            for a, b in zip(
                native["normal_load_bw"][support], native["slip_mm_s"][support]
            )
        )
        / denominator
    )
    from tools import analyze_c_banc_biological_gait as gait

    report = gait.analyze(p.gait_model(trace, file, digest))[0]
    gates = {
        "absolute_forward": forward >= 1.6790085779712631,
        "absolute_slip": slip is not None and slip <= 1.8026436968249955,
        "bout": report["window"]["longest_forward_bout_s"] >= 0.3,
        "six_cycles": all(report["events"][leg]["complete_cycles"] > 0 for leg in LEGS),
        "dual_health_and_engineering": health,
    }
    supported = all(gates.values())
    return {
        "full_status": "evaluated",
        "physical": {
            "forward_mm": forward,
            "slip_mm_s": slip,
            "slip_missing_count": missing,
            "measured_dual_healthy": health,
        },
        "gait": report,
        "H353": {
            "verdict": "supported_for_later_learning_eligibility"
            if supported
            else "partial_or_inconclusive",
            "supported": supported,
            "gates": gates,
        },
    }


def npz_pack(values):
    np = primitives().numpy()
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **values)
    raw = buffer.getvalue()
    members = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for info in archive.infolist():
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(
                    stream, max_header_size=1024
                )
                members[info.filename[:-4]] = {
                    "member": info.filename,
                    "dtype": str(dtype),
                    "shape": list(shape),
                    "fortran_order": fortran,
                    "npy_version": list(version),
                    "header_bytes": stream.tell(),
                    "zip_compression": info.compress_type,
                    "compressed_bytes": info.compress_size,
                    "uncompressed_bytes": info.file_size,
                    "crc32": info.CRC,
                }
    return raw, members


def pack(trace, native, observed, plan):
    decoded = json_bytes(trace)
    require(len(decoded) <= 16 << 20, "decoded trace16MiB")
    raw = gzip.compress(decoded, mtime=0)
    staged = {"trace.json.gz": raw}
    descriptors = {
        "trace.json.gz": dict(
            descriptor("trace.json.gz", raw),
            decoded_bytes=len(decoded),
            decoded_sha256=sha(decoded),
            encoding="gzip_strict_lossless_JSON",
        )
    }
    for name, values in [
        ("native-contact.npz", native),
        ("source-pose.npz", observed),
        ("cadence-plan.npz", plan),
    ]:
        raw, members = npz_pack(values)
        staged[name] = raw
        descriptors[name] = dict(
            descriptor(name, raw),
            selected_members=members,
            digest_provenance="actual_new_worker_acquisition",
        )
    require(sum(map(len, staged.values())) + len(decoded) <= 64 << 20, "staging64MiB")
    return staged, descriptors


def write_raw(file, raw, *, timely=lambda: None, opener=None):
    timely()
    with (opener or (lambda f: f.open("xb")))(file) as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    timely()


def emergency(
    output,
    error,
    receipt,
    staged,
    published,
    *,
    timely=lambda: None,
    stage="publication",
):
    def limited(spec, saved):
        return {
            "path": spec["path"],
            "bytes": spec["bytes"],
            "sha256": spec["sha256"],
            "missing": not saved,
            "full_saved_proof": saved,
        }

    result = {
        "schema": "flylab.m3.e353.emergency.v1",
        "status": "technical_invalid",
        "experiment_arm": "R",
        "dependency_arm": "B",
        "first_error": receipt.get("first_error") or pose().error_record(stage, error),
        "secondary_errors": (
            receipt.get("errors", []) + [pose().error_record(stage, error)]
        )[:16],
        "actual_counts": {
            k: receipt.get(k)
            for k in (
                "controls_requested",
                "controls_returned",
                "boundaries_acquired",
                "native_samples_acquired",
                "pose_records_acquired",
                "controls_unreturned",
            )
        },
        "staged": [limited(s, False) for s in staged.values()][:8],
        "published": [limited(s, True) for s in published.values()][:8],
        "unpublished": [
            limited(s, False) for k, s in staged.items() if k not in published
        ][:8],
        "metrics": None,
        "counter_status": "acquired_loop_receipt",
        "late_error_has_precedence": True,
    }
    raw = json_bytes(result)
    require(
        len(raw) <= EMERGENCY and primitives().usage(output) + len(raw) <= CAP,
        "emergency64KiB/total4MiB",
    )
    write_raw(output / "failure.json", raw, timely=timely)
    return result


def publish_worker(
    output,
    receipt,
    trace,
    native,
    observed,
    plan,
    reg,
    *,
    timely=lambda: None,
    opener=None,
    progress=None,
):
    staged, descriptors, published = {}, {}, {}
    if progress is not None:
        progress.update(staged={}, published=published)
    try:
        policy_raw = (output / "force-policy.json").read_bytes()
        require(
            primitives().strict_json(policy_raw) == reg["force_policy_value"],
            "actual force policy",
        )
        policy = descriptor("force-policy.json", policy_raw)
        published["force-policy.json"] = policy
        staged, descriptors = pack(trace, native, observed, plan)
        receipt.update(
            cadence_policy=reg["cadence_policy"],
            raw=descriptors,
            force_policy=policy,
            planned_controls=300,
            executed_controls=receipt["controls_returned"],
        )
        metrics = evaluate(
            receipt,
            trace,
            native,
            observed,
            output / "trace.json.gz",
            descriptors["trace.json.gz"]["sha256"],
        )
        result = {
            "schema": "flylab.m3.e353.reference-cadence-result.v1",
            "status": receipt["status"],
            "experiment_arm": "R",
            "dependency_arm": "B",
            "source_sha256": own_sources(),
            "registration_sha256": REG_SHA,
            "reflection_sha256": REF_SHA,
            "raw": descriptors,
            "force_policy": policy,
            "metrics": metrics,
            "receipt": receipt,
            "historical_C": reg["historical_C"],
            "inherited_H348": reg["inherited_H348"],
            "external_completion_qualified": False,
            "body_evidence_qualified": False,
            "candidate_adopted": False,
            "F1_F8_complete": False,
            "biology_proven": False,
            "NN_CPG_GPU_calls": 0,
            "first_error": receipt["first_error"],
            "errors": receipt["errors"],
        }
        staged["receipt.json"] = json_bytes(receipt)
        staged["result.json"] = json_bytes(result)
        all_desc = {k: descriptor(k, v) for k, v in staged.items()}
        if progress is not None:
            progress["staged"] = all_desc
        require(
            set(staged) | {"force-policy.json"} <= set(reg["worker_output_roster"])
            and primitives().usage(output)
            + sum(len(v) for v in staged.values())
            + EMERGENCY
            <= CAP,
            "reserve policy/raw/receipt/result/emergency before first raw write",
        )
        require(sum(len(v) for v in staged.values()) <= 64 << 20, "stage64MiB")
        for name, raw in staged.items():
            write_raw(output / name, raw, timely=timely, opener=opener)
            published[name] = all_desc[name]
        require(primitives().usage(output) <= CAP, "actual4MiB")
        timely()
        return result, 0 if receipt["status"] == "completed" else 1
    except Exception as error:  # noqa: BLE001 -- preserve acquired and partial writes/late failures
        all_desc = {k: descriptor(k, v) for k, v in staged.items()}
        try:
            result = emergency(
                output, error, receipt, all_desc, published, timely=timely
            )
        except Exception as secondary:  # noqa: BLE001 -- preserve actual acquisition/cleanup failures
            result = {
                "schema": "flylab.m3.e353.emergency.v1",
                "status": "technical_invalid",
                "first_error": pose().error_record("publication", error),
                "secondary_errors": [pose().error_record("emergency", secondary)],
            }
        return result, 1


def read_gzip(reader, spec):
    raw = reader.bytes(spec, cap=CAP)
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        decoded = stream.read((16 << 20) + 1)
        require(len(decoded) <= 16 << 20, "bounded gzip decode")
        require(stream.read(1) == b"", "gzip EOF")
    require(
        len(decoded) == spec["decoded_bytes"]
        and sha(decoded) == spec["decoded_sha256"],
        "decoded acquisition digest",
    )
    return primitives().strict_json(decoded)


def audit_worker(ctx, data):
    require(ctx["authorized"] and ctx.get("mode") == "audit", "audit private gate")
    p = primitives()
    np = p.numpy()
    reg = ctx["registration"]
    r = ctx["reader"]
    sealed = ctx["worker_seal"]
    roster = sealed["artifacts"]
    require(
        len(roster) <= 8
        and set(roster) <= set(reg["worker_output_roster"])
        and p.usage(p.safe(OUTPUT, r.root)) <= CAP,
        "sealed worker roster/cap",
    )

    def spec(name):
        s = dict(roster[name])
        s["path"] = OUTPUT + "/" + name
        return s

    if sealed["mode"] == "metadata_only_failure":
        require(
            "failure.json" in roster
            and sealed["expected_worker_status"] == "technical_invalid",
            "metadata-only failed worker",
        )
        failure = r.json(spec("failure.json"))
        require(failure["status"] == "technical_invalid", "failure metadata")
        return {
            "schema": "flylab.m3.e353.reference-cadence-audit.v1",
            "audit_passed": False,
            "mode": "metadata_only_failure",
            "original_worker_status": "technical_invalid",
            "metrics": None,
            "full_saved_proof": False,
            "body_rerun": False,
            "candidate_adopted": False,
            "F1_F8_complete": False,
            "biology_proven": False,
        }
    require(
        set(roster) == set(reg["worker_output_roster"]) - {"failure.json"},
        "full saved roster",
    )
    result = r.json(spec("result.json"))
    receipt = r.json(spec("receipt.json"))
    if receipt["status"] == "completed":
        require(
            sealed.get("expected_exit_code") == 0
            and type(sealed.get("expected_exit_code")) is int,
            "completed worker requires successful closed exit",
        )
    policy = r.json(spec("force-policy.json"))
    require(
        result["receipt"] == receipt
        and policy == reg["force_policy_value"]
        and result["source_sha256"] == own_sources()
        and result["registration_sha256"] == REG_SHA
        and result["reflection_sha256"] == REF_SHA
        and result["status"] == receipt["status"]
        and result["external_completion_qualified"] is False
        and result["body_evidence_qualified"] is False,
        "current result/receipt/policy linkage",
    )
    require(
        result["force_policy"]
        == {k: roster["force-policy.json"][k] for k in ("path", "bytes", "sha256")}
        and receipt["held_force_receipt"]["policy_sha256"]
        == roster["force-policy.json"]["sha256"],
        "saved policy descriptor",
    )
    for name in reg["new_raw_roster"]:
        require(
            all(
                result["raw"][name][k] == roster[name][k]
                for k in ("path", "bytes", "sha256")
            )
            and result["raw"] == receipt["raw"],
            "actual acquisition descriptors",
        )
    trace_spec = spec("trace.json.gz")
    trace_spec.update(
        {
            k: result["raw"]["trace.json.gz"][k]
            for k in ("decoded_bytes", "decoded_sha256")
        }
    )
    trace = read_gzip(r, trace_spec)
    values = {}
    for name in reg["new_raw_roster"][1:]:
        s = spec(name)
        members = result["raw"][name]["selected_members"]
        values[name] = r.npz(
            s,
            {k: (tuple(v["shape"]), v["dtype"]) for k, v in members.items()},
            selected=members,
        )
    plan = values["cadence-plan.npz"]
    require(set(plan) == set(reg["cadence_plan_members"]), "exact9 plan")
    for k, v in plan.items():
        require(
            v.shape == tuple(reg["cadence_plan_members"][k]["shape"])
            and str(v.dtype) == reg["cadence_plan_members"][k]["dtype"]
            and np.isfinite(v).all(),
            "finite exact plan headers",
        )
    verified = audit_plan(data["control"], data["reference"], plan)
    native = values["native-contact.npz"]
    observed = values["source-pose.npz"]
    rows = trace["samples"]
    validate_native(native, len(native["time_s"]))
    require(
        len(rows) == receipt["boundaries_acquired"]
        and len(native["time_s"]) == receipt["native_samples_acquired"]
        and set(observed)
        == {
            "native_index",
            "time_s",
            "source_upright",
            "current_upright",
            "time_available",
            "source_available",
            "current_available",
        }
        and len(observed["native_index"]) == receipt["pose_records_acquired"],
        "honest actual raw lengths",
    )
    require(
        np.array_equal(
            observed["native_index"], np.arange(1, len(observed["native_index"]) + 1)
        ),
        "acquired observation indices",
    )
    for channel, mask in [
        ("time_s", "time_available"),
        ("source_upright", "source_available"),
        ("current_upright", "current_available"),
    ]:
        require(
            observed[mask].dtype == np.dtype("bool")
            and observed[channel].dtype == np.dtype("float64")
            and not np.isinf(observed[channel]).any()
            and np.array_equal(np.isfinite(observed[channel]), observed[mask]),
            "actual observation mask vs NaN",
        )
    require(
        trace["experiment_arm"] == "R"
        and trace["dependency_arm"] == "B"
        and trace["body_hash"] == reg["source_model_hash"]
        and all(
            trace[k] == data["metadata"][k]
            for k in ("model_hash", "adapter_hash", "graph_hash", "parameters")
        ),
        "trace source labels/model",
    )
    core = pose().shared()
    _, static, _ = core.libraries()
    for c, row in enumerate(rows):
        require(
            row["control_tick"] == c
            and abs(row["time_s"] - c * 0.005) <= 2e-8
            and abs(row["physics_time_s"] - c * 0.005) <= 2e-8
            and row["neural_request"] is None,
            "actual trace clock/labels",
        )
        for key in ("qpos", "qvel", "ctrl", "joint_targets_rad"):
            require(
                np.asarray(row[key]).shape == np.asarray(data["rows"][c][key]).shape
                and np.isfinite(row[key]).all(),
                "actual trace finite state",
            )
        for leg in LEGS:
            require(
                np.asarray(row["body"]["legs"][leg]).shape == (8, 3)
                and np.isfinite(row["body"]["legs"][leg]).all(),
                "actual8 geometry",
            )
        for key in (
            "adhesion",
            "tendon_inputs",
            "joint_stiffness_scale",
            "adhesion_control",
        ):
            require(row.get(key) == data["rows"][c].get(key), "unchanged other actions")
        if c:
            require(
                np.allclose(
                    row["joint_targets_rad"], plan["command"][c], atol=2e-12, rtol=0
                ),
                "actual R independent command",
            )
    if rows:
        static.compare_replay(
            rows[:1],
            {k: v[:0] for k, v in native.items()},
            data["rows"],
            data["native"],
            0,
        )
    if receipt["prefix_checked_before_c101"]:
        static.compare_replay(
            rows[:101],
            {k: v[:5000] for k, v in native.items()},
            data["rows"],
            data["native"],
            100,
        )
    if receipt["status"] == "completed":
        metrics = evaluate(
            receipt,
            trace,
            native,
            observed,
            p.safe(OUTPUT + "/trace.json.gz", r.root),
            r.digests[OUTPUT + "/trace.json.gz"],
        )
        require(p.lower(metrics) == result["metrics"], "independent physical/gait/H353")
        passed = True
    else:
        require(
            receipt["status"] in ("scientific_partial", "technical_invalid"),
            "partial/technical branch",
        )
        metrics = {
            "full_status": "not_tested",
            "physical": None,
            "gait": None,
            "H353": {"verdict": "not_evaluated", "supported": None, "gates": None},
        }
        require(result["metrics"] == metrics, "no fabricated partial full metrics")
        passed = False
    return {
        "schema": "flylab.m3.e353.reference-cadence-audit.v1",
        "audit_passed": passed,
        "original_worker_status": receipt["status"],
        "metrics": metrics,
        "independent_plan": verified,
        "source_sha256": own_sources(),
        "registration_sha256": REG_SHA,
        "reflection_sha256": REF_SHA,
        "consumed_digests": r.digests,
        "read_counts": r.read_counts,
        "full_saved_proof": passed,
        "body_rerun": False,
        "external_completion_qualified": False,
        "candidate_adopted": False,
        "F1_F8_complete": False,
        "biology_proven": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--qualification")
    parser.add_argument("--review")
    parser.add_argument("--output")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run", action="store_true")
    group.add_argument("--audit", action="store_true")
    a = parser.parse_args(argv)
    mode = "run" if a.run else "audit" if a.audit else None
    if mode:
        require(
            list(sys.argv[1:] if argv is None else argv)
            == original_argv(mode, a.freeze, a.review)[5:],
            "exact sole CLI",
        )

    def timely():
        if time.perf_counter() - ENTRY > (22 if mode == "run" else 5):
            raise TimeoutError("E353 inclusive entry through close")
        import psutil

        if psutil.Process().memory_info().rss > 2 << 30:
            raise MemoryError("host2GiB")

    reader = primitives().EvidenceReader(timely=timely)
    receipt, output, stage = None, None, "gate"
    progress = {"staged": {}, "published": {}}
    try:
        if not mode:
            require(a.qualification is not None, "passive qualification")
            value = make_freeze(
                a.qualification, context=metadata_preflight(reader=reader)
            )
            primitives().publish(
                primitives().safe(a.freeze), value, timely=timely, count_folder=False
            )
            return 0
        ctx = gate(mode, a.freeze, a.review, output=a.output, reader=reader)
        output = primitives().safe(a.output)
        if mode == "audit":
            data = (
                None
                if ctx["worker_seal"]["mode"] == "metadata_only_failure"
                else load_inputs(ctx)
            )
            report = audit_worker(ctx, data)
            output.mkdir()
            primitives().publish(output / "report.json", report, timely=timely)
            timely()
            return 0 if report["audit_passed"] else 1
        output.mkdir()
        stage = "input_loading"
        data = load_inputs(ctx)
        stage = "planning"
        plan = build_plan(data["control"], data["reference"])
        stage = "dependency_initialization"
        deps = pose().SourcePoseDependencies(
            ctx["registration"]["force_policy_value"],
            output,
            ctx["registration"]["observation_policy_value"],
        )
        stage = "loop"
        receipt, trace, native, observed = loop(data, plan, deps, timely=timely)
        stage = "publication"
        result, code = publish_worker(
            output,
            receipt,
            trace,
            native,
            observed,
            plan,
            ctx["registration"],
            timely=timely,
            progress=progress,
        )
        if code:
            receipt = dict(receipt)
            receipt["first_error"] = receipt.get("first_error") or result.get(
                "first_error"
            )
            receipt["errors"] = result.get("secondary_errors", receipt["errors"])
            print(json_bytes(result).decode(), file=sys.stderr)
        stage = "main_finalization"
        timely()
        return code
    except Exception as error:  # noqa: BLE001 -- strict bounded gate/load/audit/publication rejection
        if receipt is not None:
            try:
                report = emergency(
                    output,
                    error,
                    receipt,
                    progress["staged"],
                    progress["published"],
                    stage=stage,
                )
            except Exception as secondary:  # noqa: BLE001 -- preserve failure and acquired counts if emergency sink fails
                report = {
                    "schema": "flylab.m3.e353.emergency.v1",
                    "status": "technical_invalid",
                    "first_error": receipt.get("first_error")
                    or pose().error_record(stage, error),
                    "secondary_errors": (
                        receipt.get("errors", [])
                        + [pose().error_record(stage, error)]
                        + [pose().error_record("emergency", secondary)]
                    )[:16],
                    "actual_counts": {
                        k: receipt.get(k)
                        for k in (
                            "controls_requested",
                            "controls_returned",
                            "boundaries_acquired",
                            "native_samples_acquired",
                            "pose_records_acquired",
                            "controls_unreturned",
                        )
                    },
                    "counter_status": "acquired_loop_receipt",
                    "late_error_has_precedence": True,
                }
            print(json_bytes(report).decode(), file=sys.stderr)
            return 1
        report = {
            "schema": "flylab.m3.e353.emergency.v1",
            "status": "technical_invalid",
            "first_error": {
                "stage": "main",
                "type": type(error).__name__,
                "message": str(error)[:1024],
            },
            "secondary_errors": [],
            "actual_counts": None,
            "counter_status": "not_instrumented_before_loop",
            "candidate_adopted": False,
        }
        print(json_bytes(report).decode(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())


def generated_reader(root):
    """Explicit fixture-only capability; never generated=True on historical raw."""
    resolved = Path(root).resolve()
    require(
        not resolved.is_relative_to(ROOT)
        or resolved.relative_to(ROOT).as_posix().startswith(BASE + "/e353-sol-"),
        "owned E353 fixture namespace",
    )
    reader = primitives().EvidenceReader(resolved)
    reader.array_authorized = True
    return reader
