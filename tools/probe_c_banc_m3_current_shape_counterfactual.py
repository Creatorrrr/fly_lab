"""E337 fixed, body-only stored-output counterfactual; no neural execution.

Original numerical inputs and FlyBody construction are lazy and follow the
passive current-source gate. TestDimensions is the only injection boundary.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import analyze_c_banc_m3_saved_gait_transfer as saved

BASE = "verification/banc-natural-gait-m3-20260919"
OUTPUT = BASE + "/e337-current-shape-counterfactual"
OWNED = (
    "tools/probe_c_banc_m3_current_shape_counterfactual.py",
    "tests/test_c_banc_current_shape_counterfactual.py",
)
SCHEMA = "flylab.m3.e337.current-shape-counterfactual.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e337.current-shape-counterfactual.review.v1"
RESULT_SCHEMA = "flylab.m3.e337.current-shape-counterfactual.result.v1"
QUAL_SCHEMA = "flylab.m3.e337.generated-qualification.v1"
LIMITS = {
    "supervisor_wall_s": 80,
    "internal_wall_s": 70,
    "science_wall_s": 60,
    "finalization_reserve_s": 10,
    "host_bytes": 2 << 30,
    "output_bytes": 128 << 20,
    "minimum_free_disk_bytes": 512 << 20,
    "cpu_threads": 1,
    "gpu_jobs": 0,
    "retry_count": 0,
    "body_constructors": 2,
    "controls_per_arm": 300,
    "native_steps_per_arm": 15000,
}
PATHS = {
    "previous": BASE + "/e336-saved-gait-transfer-freeze.json",
    "decision": BASE + "/parent-e336-final-decision.json",
    "design": "docs/BANC_M3_E337_CURRENT_SHAPE_COUNTERFACTUAL_DESIGN_20261001.md",
    "astra": BASE + "/parent-e337-astra-design-review.md",
    "reflection": BASE + "/parent-e337-astra-reflection-v2.json",
    "prior_reflection": BASE + "/parent-e337-astra-reflection.json",
    "prior_design": BASE + "/parent-e337-reflected-design-v1.md",
    "metadata": BASE + "/parent-e337-current-body-metadata.json",
    "invocation": BASE + "/parent-e337-astra-invocation.json",
    "qualification": BASE + "/parent-e337-qualification.json",
}
FIXED = {
    "previous": "7dbc9c6ed9496d03a112068e688f771679389b429f2ff3841b24eb1d96c09a6c",
    "design": "b7c6bdbaf4e29ec619cf761e77a03c34d525b3e5e582af6da559e2c4c878ef7c",
    "astra": "5843c896e4f380e50381f695dfb6e3b821f6da71051d05fddd953949b5a82295",
    "reflection": "0b25df576ba58e10342129665a3ecf9190dac650f693b896bf52fcfed93edbba",
    "prior_reflection": "7ea050e18d759c5156be553d6769cd3097e7ca1eb23e63764c361755762a1c33",
    "prior_design": "4d729cfdca7989a31e3edb5d1610dd099b68b584e7fe5841f332cca7593dfc6f",
    "metadata": "5634fc22519f98177fb37a79107d0e88c7b976fa60ee8798165afad11003d7a0",
    "invocation": "17b305c51b35edff1874416b63bedf140a9d3d0c91b9cbe526fd23f88ae0e745",
}
OPTIONS = {
    "model": "flybody",
    "terrain": "flat",
    "terrain_seed": 0,
    "slope_degrees": 10,
    "render_camera": False,
    "actuation": "whole_body",
    "servo_profile": "tracking_all",
    "attachment": "free",
    "tendons": "all",
}
PROFILE = {
    "backend": "cpu",
    "noslip_iterations": 5,
    "multiccd": True,
    "max_contacts": 1024,
    "max_constraints": 4096,
    "cuda_graph": True,
    "control_backend": "cpu",
}
CONFIG = {
    "mode": "walk",
    "task": "explore",
    "goalAngle": 1,
    "altitude": 0.8,
    "sensorNoise": 0.018,
    "graphGain": 1,
    "motorCoupled": True,
    "friction": 1,
    "gravity": 1,
}
NATIVE_KEYS = (
    "time_s",
    "normal_load_bw",
    "slip_mm_s",
    "slip_valid",
    "upright",
    "nonfoot_contact",
)
NORMAL_RAW_FILES = (
    "shape-plan.npz",
    "baseline/trace.json",
    "baseline/native-contact.npz",
    "candidate/trace.json",
    "candidate/native-contact.npz",
    "baseline/evaluation.json",
    "candidate/evaluation.json",
)


def read(path):
    return saved.read(path)


def sha256(path, timely=lambda: None):
    return saved.sha256(path, timely)


def put_json(path, value):
    """Atomic replacement in the new owned output, with durable file contents."""
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("w", encoding="utf8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_npz(path, arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def typed(value, shape, dtype, label, *, finite=True):
    return saved.typed(value, shape, dtype, label, finite=finite)


def exact(actual, expected, label="exact", *, masked_nan=False):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or a.dtype.str != b.dtype.str:
        raise ValueError(label + " dtype/shape differs")
    if masked_nan:
        if a.dtype.kind != "f" or not np.array_equal(np.isnan(a), np.isnan(b)):
            raise ValueError(label + " NaN mask differs")
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(label + " nonfinite exact state")
    if a.tobytes(order="C") != b.tobytes(order="C"):
        raise ValueError(label + " bytes differ")


def cyclic_interpolate(curves, phase):
    """Query global-phase curves directly; anatomical offsets are already stored."""
    curves = typed(curves, (6, 128, 7), "<f8", "curves")
    phase = typed(phase, (len(phase), 6), "<f8", "phase")
    coordinate = np.remainder(phase, 1.0) * 128
    floor = np.floor(coordinate)
    index = floor.astype(np.int64) % 128
    leg = np.arange(6)[None, :]
    a = curves[leg, index]
    return a + (coordinate - floor)[..., None] * (curves[leg, (index + 1) % 128] - a)


def shape_plan(command, live, curves):
    """Fixed rowc policy; no phase unwrapping, gap bridging, or reverse repair."""
    command = typed(command, (301, 42), "<f8", "command")
    q = typed(live["q"], (201, 42), "<f8", "live q")
    exact(q, command[100:301], "E336 live q source")
    control = typed(live["control_tick"], (201,), "<i8", "live controls")
    exact(control, np.arange(100, 301, dtype=np.int64), "live row order")
    z = typed(live["z"], (201, 6, 2), "<f8", "live z")
    R = typed(live["R"], (201, 6), "<f8", "live R")
    if np.any(R < 0.5) or np.any(R > 1 + 1e-12):
        raise ValueError("all201 boundary R>=.5 required")
    if not np.allclose(np.linalg.norm(z, axis=-1), R, rtol=0, atol=1e-12):
        raise ValueError("z/R disagreement")
    cross = z[:-1, :, 0] * z[1:, :, 1] - z[:-1, :, 1] * z[1:, :, 0]
    dot = np.einsum("tlk,tlk->tl", z[:-1], z[1:])
    delta = np.arctan2(cross, dot) / (2 * np.pi)
    reliable = (R[:-1] >= 0.5) & (R[1:] >= 0.5) & (np.abs(delta) != 0.5)
    if not reliable.all():
        raise ValueError("all200 edges reliable without exact half-turn required")
    phase = np.remainder(np.arctan2(z[..., 1], z[..., 0]) / (2 * np.pi), 1)
    reference = cyclic_interpolate(curves, phase).reshape(201, 42)
    x = np.clip((np.arange(100, 301) * 0.005 - 0.5) / 0.1, 0, 1)
    blend = 3 * x**2 - 2 * x**3
    result = command.copy(order="C")
    result[101:] = (1 - blend[1:, None]) * command[101:] + blend[1:, None] * reference[
        1:
    ]
    # The endpoints are exact requests, including signed zero.
    result[120:] = reference[20:]
    exact(result[:101], command[:101], "direct prefix copy")
    return {
        "command": result,
        "phase": phase,
        "blend": blend,
        "reference": reference,
        "reliable_edges": reliable,
        "signed_phase_delta": delta,
    }


def _merge(target, additions, root):
    saved.merge_pins(target, additions, root=root)


def astra_file_pins(invocation):
    """Resolve the fixed invocation's semantic aliases to its real artifacts."""
    files = {
        "snapshot": BASE + "/parent-e337-pre-astra-design.md",
        "prompt": BASE + "/parent-e337-astra-review-prompt.txt",
        "review": PATHS["astra"],
        "events": BASE + "/parent-e337-astra-review-events.jsonl",
        "stderr": BASE + "/parent-e337-astra-review-stderr.txt",
    }
    pins = invocation.get("files_sha256")
    if (not isinstance(pins, dict) or set(pins) != set(files)
            or pins["snapshot"] != invocation.get("input_snapshot_sha256")
            or pins["review"] != FIXED["astra"]):
        raise ValueError("fixed Astra artifact aliases and identities required")
    return {files[alias]: pin for alias, pin in pins.items()}


def make_freeze(*, root=ROOT):
    """Build immutable inherited closure; never include mutable cost/checkpoint."""
    previous = read(saved.safe(PATHS["previous"], root))
    sources = previous["source_sha256"].copy()
    inputs = previous["input_sha256"].copy()
    inherited_sources, inherited_inputs = sources.copy(), inputs.copy()
    _merge(sources, {name: sha256(saved.safe(name, root)) for name in OWNED}, root)
    _merge(
        inputs, {name: sha256(saved.safe(name, root)) for name in PATHS.values()}, root
    )
    decision = read(saved.safe(PATHS["decision"], root))
    _merge(inputs, decision["evidence_sha256"], root)
    # The completed result manifest adds all E336 result bytes, not new analysis.
    manifest_name = saved.OUTPUT + "/output-manifest.json"
    manifest = read(saved.safe(manifest_name, root))
    for name, info in manifest["files"].items():
        pin = info["sha256"] if isinstance(info, dict) else info
        _merge(inputs, {saved.OUTPUT + "/" + name: pin}, root)
    qualification = read(saved.safe(PATHS["qualification"], root))
    _merge(inputs, qualification["evidence_sha256"], root)
    invocation = read(saved.safe(PATHS["invocation"], root))
    _merge(
        inputs,
        astra_file_pins(invocation),
        root,
    )
    return {
        "schema": SCHEMA,
        "paths": PATHS.copy(),
        "output": OUTPUT,
        "limits": LIMITS.copy(),
        "source_sha256": sources,
        "input_sha256": inputs,
        "inherited_source_sha256": inherited_sources,
        "inherited_input_sha256": inherited_inputs,
        "input_members": previous["input_members"],
        "reference": previous["reference"],
        "startup": previous["startup"],
        "original_body_authorized": False,
        "allocation": None,
        "candidate_adopted": False,
        "F1_F8_complete": False,
    }


def _completed(contract, supervision, digest, original_calls):
    saved._completed(contract, supervision, digest, original_calls=original_calls)


def gate(
    freeze_path,
    freeze_sha256,
    review_path,
    review_sha256,
    output,
    *,
    root=ROOT,
    timely=lambda: None,
):
    """Passive SHA/schema/producer gate: zero numerical payload or constructor."""
    if Path(output).exists() or Path(output).resolve() != saved.safe(OUTPUT, root):
        raise ValueError("fresh literal E337 output required")
    for path, pin in ((freeze_path, freeze_sha256), (review_path, review_sha256)):
        if sha256(path, timely) != pin:
            raise ValueError("freeze/review SHA differs")
    freeze, review = read(freeze_path), read(review_path)
    expected = make_freeze(root=root)
    for key, value in expected.items():
        if (
            key not in {"original_body_authorized", "allocation"}
            and freeze.get(key) != value
        ):
            raise ValueError("exact E337 freeze differs: " + key)
    if freeze.get("original_body_authorized") is not True or freeze.get(
        "allocation"
    ) != {
        "cumulative_before_s": 9513.922589299525,
        "cumulative_limit_s": 9660,
        "original_body_reserved_s": 80,
    }:
        raise ValueError("root prospective original80 authorization required")
    if (
        review.get("schema") != REVIEW_SCHEMA
        or review.get("approved") is not True
        or review.get("blocking_defects") != []
        or review.get("freeze_sha256") != freeze_sha256
        or review.get("source_sha256") != freeze["source_sha256"]
    ):
        raise ValueError("approved current root source review required")
    for alias, pin in FIXED.items():
        if freeze["input_sha256"].get(PATHS[alias]) != pin:
            raise ValueError("fixed reviewed input differs: " + alias)
    for mapping in (freeze["source_sha256"], freeze["input_sha256"]):
        saved._pin_map(mapping, root, timely)
    previous = read(saved.safe(PATHS["previous"], root))
    if (
        len(previous["source_sha256"]) != 205
        or len(previous["input_sha256"]) != 2737
        or freeze["inherited_source_sha256"] != previous["source_sha256"]
        or freeze["inherited_input_sha256"] != previous["input_sha256"]
    ):
        raise ValueError("exact inherited205/2737 closure required")
    saved.verify_previous(previous["input_sha256"], root=root)
    saved.verify_inventory(
        read(saved.safe(saved.PATHS["inventory"], root)),
        previous["inherited_source_sha256"],
        previous["inherited_input_sha256"],
        root=root,
    )
    decision = read(saved.safe(PATHS["decision"], root))
    if (
        decision.get("analysis_valid") is not True
        or decision.get("root_independent_raw_qualification") is not True
        or decision.get("raw_checks") != 7141
        or decision.get("prior_E335_body_verdict") != "rejected"
        or decision.get("candidate_adopted") is not False
    ):
        raise ValueError("completed valid E336 / preserved rejected E335 required")
    for prefix, calls in (("original", 1), ("saved-raw-audit", 0)):
        cpath = BASE + "/parent-e336-" + prefix + "-contract.json"
        spath = BASE + "/parent-e336-" + prefix + "-supervision.json"
        _completed(
            read(saved.safe(cpath, root)),
            read(saved.safe(spath, root)),
            freeze["input_sha256"][cpath],
            calls,
        )
    audit = read(saved.safe(BASE + "/parent-e336-saved-raw-audit.json", root))
    if (
        audit.get("passed") is not True
        or not audit.get("checks")
        or not all(v is True for v in audit["checks"].values())
    ):
        raise ValueError("all completed saved raw audit checks required")
    reflection = read(saved.safe(PATHS["reflection"], root))
    if set(reflection.get("accepted", {})) != {"M" + str(i) for i in range(1, 8)}:
        raise ValueError("actual Astra M1-M7 reflection required")
    own = {name: freeze["source_sha256"][name] for name in OWNED}
    qual = read(saved.safe(PATHS["qualification"], root))
    if (
        qual.get("schema") != QUAL_SCHEMA
        or qual.get("approved") is not True
        or qual.get("blocking_defects") != []
        or qual.get("source_sha256") != own
        or not qual.get("evidence_sha256")
    ):
        raise ValueError("current own2 generated qualification required")
    for key in ("cpu_report", "cpu_contract", "cpu_supervision"):
        desc = qual[key]
        if (
            set(desc) != {"path", "sha256"}
            or freeze["input_sha256"].get(desc["path"]) != desc["sha256"]
        ):
            raise ValueError("exact completed qualification descriptor required")
    report = read(saved.safe(qual["cpu_report"]["path"], root))
    if (
        report.get("schema") != "flylab.m3.e337.root-generated.v1"
        or report.get("passed") is not True
        or report.get("source_sha256") != own
        or not report.get("checks")
        or not all(v is True for v in report["checks"].values())
        or any(
            report.get(k) != 0 for k in ("original_calls", "neural_calls", "CUDA_calls")
        )
    ):
        raise ValueError("independent generated CPU proof required")
    contract, supervision = (
        read(saved.safe(qual[k]["path"], root))
        for k in ("cpu_contract", "cpu_supervision")
    )
    _completed(contract, supervision, qual["cpu_contract"]["sha256"], 0)
    if any(contract["source_sha256"].get(name) != pin for name, pin in own.items()):
        raise ValueError("executed generated producer own2 differs")
    saved.verify_execution(report, qual["cpu_report"], contract, supervision, root=root)
    saved.verify_producer_closure(contract, freeze, root=root)
    timely()
    return freeze


@dataclass(frozen=True)
class TestDimensions:
    controls: int = 300
    native_per_control: int = 50
    prefix_controls: int = 100

    def __post_init__(self):
        if (
            self.controls != 300
            or self.native_per_control != 50
            or self.prefix_controls != 100
        ):
            raise ValueError("generated fixtures retain full300/50/100 dimensions")


class HealthStop(RuntimeError):
    pass


def native_arrays(samples):
    arrays = {}
    for key in NATIVE_KEYS:
        shape = (
            (len(samples), 6)
            if key in {"normal_load_bw", "slip_mm_s", "slip_valid"}
            else (len(samples),)
        )
        dtype = np.bool_ if key in {"slip_valid", "nonfoot_contact"} else np.float64
        arrays[key] = np.asarray([s[key] for s in samples], dtype=dtype).reshape(shape)
    return arrays


def npz_headers(path):
    """Bounded metadata-only inspection, rejecting unsafe unselected members too."""
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > 128 or sum(i.file_size for i in members) > 128 << 20:
            raise ValueError("NPZ header declared allocation too large")
        for member in members:
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("unsupported NPY header")
                if (
                    dtype.hasobject
                    or dtype.fields is not None
                    or dtype.subdtype is not None
                ):
                    raise ValueError("object/structured/subarray member forbidden")
                if (
                    not all(type(v) is int and v >= 0 for v in shape)
                    or math.prod(shape) * dtype.itemsize + stream.tell()
                    != member.file_size
                ):
                    raise ValueError("NPZ declared shape/size differs")
    return saved.npz_headers(path)


def verify_native(arrays, *, count=None):
    n = len(arrays["time_s"]) if count is None else count
    if set(arrays) != set(NATIVE_KEYS):
        raise ValueError("native fields differ")
    for key in NATIVE_KEYS:
        shape = (n, 6) if key in {"normal_load_bw", "slip_mm_s", "slip_valid"} else (n,)
        dtype = "|b1" if key in {"slip_valid", "nonfoot_contact"} else "<f8"
        typed(arrays[key], shape, dtype, key, finite=key != "slip_mm_s")
    slip, valid, load = (
        arrays[k] for k in ("slip_mm_s", "slip_valid", "normal_load_bw")
    )
    if np.any(load < 0) or not np.array_equal(np.isfinite(slip), valid):
        raise ValueError("native slip validity/finite mask differs")
    if not np.isnan(slip[~valid]).all() or not valid[load >= 0.02].all():
        raise ValueError("supported slip required; unloaded invalid must be NaN")
    if not np.isfinite(arrays["upright"]).all():
        raise ValueError("finite native upright required")


def compare_replay(rows, native, expected_rows, expected_native, controls):
    if len(rows) != controls + 1 or len(native["time_s"]) != controls * 50:
        raise ValueError("complete replay comparator required")
    for c, row in enumerate(rows):
        expected = expected_rows[c]
        for key in ("qpos", "qvel", "ctrl", "joint_targets_rad", "adhesion"):
            dtype = np.bool_ if key == "adhesion" else np.float64
            exact(
                np.asarray(row[key], dtype=dtype),
                np.asarray(expected[key], dtype=dtype),
                "control" + str(c) + " " + key,
            )
        if row["control_tick"] != c or row["time_s"] != expected["time_s"]:
            raise ValueError("boundary actual time/order differs")
        if row.get("physics_time_s") != expected.get("physics_time_s"):
            raise ValueError("actual boundary physics clock differs")
        if row["tendon_inputs"] != expected["tendon_inputs"]:
            raise ValueError("all tendon dictionary channels differ")
    verify_native(native, count=controls * 50)
    for key in NATIVE_KEYS:
        exact(
            native[key],
            expected_native[key][: controls * 50],
            key,
            masked_nan=key == "slip_mm_s",
        )
    saved.close(
        native["time_s"],
        np.arange(1, controls * 50 + 1) * 0.0001,
        "nominal native clock",
        atol=2e-8,
        rtol=0,
    )


def classify(candidate, baseline, *, engineering_valid=True, health_rejection=False):
    """H337 only: baseline-relative fixed conjunction, distinct from absolute gates."""
    if not engineering_valid:
        return {"verdict": "technical_invalid", "supported": None, "gates": None}
    if health_rejection:
        return {"verdict": "health_rejection", "supported": False, "gates": None}
    gates = {
        "forward_bout": candidate["longest_continuous_forward_bout_s"] >= 0.3,
        "all_six_eligible_cycles": all(
            candidate["complete_cycles"].get(leg, 0) >= 1 for leg in saved.LEGS
        ),
        "forward_relative": candidate["forward_mm"] >= baseline["forward_mm"],
        "slip_relative": (
            candidate["slip_mm_s"] is not None
            and baseline["slip_mm_s"] is not None
            and candidate["slip_mm_s"] <= baseline["slip_mm_s"]
        ),
    }
    supported = all(gates.values())
    return {
        "verdict": "supported" if supported else "not_supported",
        "supported": supported,
        "gates": gates,
    }


def descriptive(gait):
    cv, lr = {}, {}
    for leg in saved.LEGS:
        event = gait["events"][leg]
        sufficient = (
            event["complete_cycles"] >= 3 and event["recurrence_evaluable"] is True
        )
        cv[leg] = {
            "value": event["period_cv"] if sufficient else None,
            "reason": None if sufficient else "fewer_than_three_complete_cycles",
        }
    for a, b in (("lf", "rf"), ("lm", "rm"), ("lh", "rh")):
        pair = gait["pair_phase"][a + "-" + b]
        enough = all(gait["events"][leg]["complete_cycles"] >= 3 for leg in (a, b))
        overlap = pair["samples"] > 0
        lr[a + "-" + b] = {
            "value": copy.deepcopy(pair) if enough and overlap else None,
            "reason": None
            if enough and overlap
            else "fewer_than_three_complete_cycles"
            if not enough
            else "no_finite_pair_overlap",
        }
    return {"period_cv": cv, "lr_phase": lr}


class Dependencies:
    """Selective original loader; construction imports occur only after gate/load."""

    def load(self, freeze, *, root=ROOT, timely=lambda: None, reads=None):
        prefix = BASE + "/e335-smooth-secant-body"
        metadata = read(saved.safe(PATHS["metadata"], root))
        reader = saved.module(
            saved.safe(saved.PATHS["reader"], root), "e337_body_reader"
        )
        state = reader.LimitedState(
            saved.safe(prefix + "/observed-start", root), timely=timely
        )
        selected = metadata["selected"]
        for key in ("schema", "seed", "world", "parameters", "control_tick"):
            if state.state.get(key) != selected[key]:
                raise ValueError("explicit observed-start metadata differs: " + key)
        if (
            sorted(state.state["body"]) != selected["body_keys"]
            or state.state["body"]["modelHash"] != selected["body_model_hash"]
        ):
            raise ValueError("selected body schema/model differs")
        if state.manifest["versions"] != _versions():
            raise ValueError("observed-start package versions differ")

        def decode(value, path):
            if isinstance(value, dict) and set(value) == {"__npy__"}:
                header = state.headers[value["__npy__"]]
                array = state.leaf(
                    path, dtype=header["dtype"], shape=tuple(header["shape"])
                )
                if not np.isfinite(array).all():
                    raise ValueError("finite body-only leaf required")
                if reads is not None:
                    reads.append(
                        dict(
                            path=prefix + "/observed-start",
                            semantic_leaf=path,
                            **saved.metadata(array),
                        )
                    )
                return array.tolist()
            if isinstance(value, dict):
                return {k: decode(v, path + [k]) for k, v in value.items()}
            if isinstance(value, list):
                return [decode(v, path + [i]) for i, v in enumerate(value)]
            return value

        body_state = decode(state.state["body"], ["body"])
        trace = read(saved.safe(prefix + "/trace.json", root))
        rows = trace["samples"]
        if (
            len(rows) != 301
            or [r["control_tick"] for r in rows] != list(range(301))
            or trace["body_hash"] != selected["body_model_hash"]
            or trace["parameters"] != selected["parameters"]
        ):
            raise ValueError("trace metadata and explicit body contract differ")
        command = self.selected(
            prefix + "/control-raw.npz",
            ("command", "control_tick", "physics_time_s"),
            freeze,
            root,
            timely,
            reads,
        )
        typed(command["command"], (301, 42), "<f8", "source command")
        exact(command["control_tick"], np.arange(301, dtype=np.int64), "control order")
        saved.close(
            command["physics_time_s"],
            np.arange(301) * 0.005,
            "source boundary clock",
            rtol=0,
            atol=2e-8,
        )
        for c, row in enumerate(rows):
            row["physics_time_s"] = float(command["physics_time_s"][c])
            if c:
                exact(
                    np.asarray(row["joint_targets_rad"], np.float64),
                    command["command"][c],
                    "source trace q",
                )
            if c and row.get("command_interval") != {
                "start_tick": c - 1,
                "end_tick": c,
                "start_s": (c - 1) * 0.005,
                "end_s": c * 0.005,
            }:
                raise ValueError("source command interval differs")
            if c:
                validate_action(row)
        native = self.selected(
            prefix + "/native-contact.npz", NATIVE_KEYS, freeze, root, timely, reads
        )
        verify_native(native, count=15000)
        live_name = saved.OUTPUT + "/live-smooth.npz"
        headers = npz_headers(saved.safe(live_name, root))
        schema = {
            key: {"shape": headers[key]["shape"], "dtype": headers[key]["dtype"]}
            for key in ("q", "z", "R", "phase_phase", "phase_valid")
        }
        live = saved.load_selected(
            saved.safe(live_name, root), schema, reads=reads, timely=timely
        )
        live["control_tick"] = np.arange(100, 301, dtype=np.int64)
        expected_phase = np.arctan2(live["z"][..., 1], live["z"][..., 0]) / (2 * np.pi)
        exact(live["phase_phase"], expected_phase, "saved phase encoding")
        typed(live["phase_valid"], (200, 6), "|b1", "saved phase validity")
        if not live["phase_valid"].all():
            raise ValueError("saved200 reliable edges required")
        constants = self.selected(
            freeze["reference"], ("curves",), freeze, root, timely, reads
        )
        startup = saved.load_selected(
            saved.safe(freeze["startup"], root),
            {
                "target_command": {"shape": [99, 42], "dtype": "<f8"},
                "target_tendon": {"shape": [99, 6], "dtype": "<f8"},
            },
            reads=reads,
            timely=timely,
        )
        return {
            "rows": rows,
            "command": command["command"],
            "native": native,
            "body_state": body_state,
            "metadata": trace,
            "body_metadata": metadata,
            "curves": constants["curves"],
            "live": live,
            "startup": startup,
        }

    @staticmethod
    def selected(name, keys, freeze, root, timely, reads):
        descriptor = freeze["input_members"][name]
        headers = npz_headers(saved.safe(name, root))
        if set(headers) != set(descriptor):
            raise ValueError("full frozen archive key roster differs")
        for key, spec in descriptor.items():
            if headers[key]["shape"] != spec["shape"] or headers[key][
                "dtype"
            ] != saved.dtype_metadata(spec["dtype"]):
                raise ValueError("full frozen archive typed header differs")
        schema = {
            key: {
                "shape": descriptor[key]["shape"],
                "dtype": descriptor[key]["dtype"],
                "finite": key != "slip_mm_s",
            }
            for key in keys
        }
        return saved.load_selected(
            saved.safe(name, root), schema, reads=reads, timely=timely
        )

    @staticmethod
    def body_class(data):
        from flylab.body import FlyGymBody

        return FlyGymBody

    @staticmethod
    def body_arguments(data):
        selected = data["body_metadata"]["selected"]
        return (
            selected["seed"],
            copy.deepcopy(selected["world"]),
            CONFIG.copy(),
        ), {
            "optimized": True,
            "initial_pose": None,
            "physics_profile": PROFILE.copy(),
            "body_options": OPTIONS.copy(),
        }

    @staticmethod
    def configure(body, data):
        from flylab.c.banc_mechanics import (
            configure_activation_trajectory,
            configure_active_length,
            configure_adhesion_units,
            configure_finite_pad_contact,
            configure_torque_capacity,
        )

        for function, value in (
            (configure_torque_capacity, 0),
            (configure_activation_trajectory, 0),
            (configure_active_length, False),
            (configure_finite_pad_contact, False),
            (configure_adhesion_units, False),
        ):
            function(body, value)
        if getattr(body, "graded_adhesion", False):
            raise ValueError("graded adhesion forbidden")
        proof = {
            "config": body.config,
            "body_options": asdict(body.body_options),
            "physics_profile": asdict(body.physics_profile),
            "physics_identity": body.physics_identity(),
            "model_hash": body.model_hash,
            "joint_names": list(body.joint_names),
            "activation_trajectory_tau_s": getattr(
                body, "joint_target_activation_tau_s", 0
            ),
            "active_length": hasattr(body, "active_length_scale"),
            "graded_adhesion": False,
        }
        if (
            proof["config"] != CONFIG
            or proof["body_options"] != OPTIONS
            or proof["physics_profile"] != PROFILE
            or proof["model_hash"] != data["body_state"]["modelHash"]
            or len(proof["joint_names"]) != 42
            or len(set(proof["joint_names"])) != 42
            or proof["activation_trajectory_tau_s"] != 0
            or proof["active_length"]
        ):
            raise ValueError("actual configured body differs")
        proof["original_compiled_identity_direct_comparison"] = "not_recorded"
        proof["original_compiled_joint_names_direct_comparison"] = "not_recorded"
        return proof

    @staticmethod
    def install(body, callback):
        from flylab.c.walking_trace import contact_diagnostics

        original = body._step_physics
        shadow = copy.copy(body)
        shadow.d = body.mj.MjData(body.m)
        typ = body.mj.mjtState.mjSTATE_INTEGRATION
        state = np.empty(body.mj.mj_stateSize(body.m, typ))

        def step():
            original()
            body.mj.mj_getState(body.m, body.d, state, typ)
            body.mj.mj_setState(body.m, shadow.d, state, typ)
            body.mj.mj_forward(body.m, shadow.d)
            feet = contact_diagnostics(shadow)
            _, rotation, _ = shadow.pose()
            slip = [np.nan if v is None else v for v in feet["slip_mm_s"]]
            callback(
                {
                    "time_s": float(body.physics_time()),
                    "normal_load_bw": feet["normal_load_bw"],
                    "slip_mm_s": slip,
                    "slip_valid": np.isfinite(slip).tolist(),
                    "nonfoot_contact": bool(shadow.nonfoot_contact()),
                    "upright": float(rotation[2, 2]),
                    "numeric_fault": bool(
                        not np.isfinite(body.d.qpos).all()
                        or not np.isfinite(body.d.qvel).all()
                        or body.fault
                    ),
                }
            )
            if (
                not np.isfinite(body.d.qpos).all()
                or not np.isfinite(body.d.qvel).all()
                or body.fault
            ):
                raise ValueError("native body numeric/fault state")

        body._step_physics = step
        return lambda: setattr(body, "_step_physics", original)

    @staticmethod
    def boundary(body, source_row):
        from flylab.c.walking_trace import capture_control

        session = SimpleNamespace(
            body=body,
            control_tick=source_row["control_tick"],
            tendon_inputs=source_row["tendon_inputs"],
            rate=np.empty(0),
            fault=body.fault,
            _trace_neural_request=None,
        )
        row = capture_control(session)
        row["physics_time_s"] = float(body.physics_time())
        row["source_label"] = {
            "control_tick": source_row["control_tick"],
            "neural_execution": 0,
            "network_tick": source_row.get("network_tick"),
        }
        row["neural_request"] = {
            "external_force_at_start": copy.deepcopy(
                (source_row.get("neural_request") or {}).get("external_force_at_start")
            )
        }
        return row

    @staticmethod
    def evaluate(folder, data):
        from tools import analyze_c_banc_biological_gait as gait

        raw = {
            "command": np.asarray(
                [
                    r["joint_targets_rad"]
                    for r in read(folder / "trace.json")["samples"]
                ],
                np.float64,
            ),
            "tendon": np.asarray(
                [
                    [r["tendon_inputs"][leg + "_tarsus"] for leg in saved.LEGS]
                    for r in read(folder / "trace.json")["samples"]
                ],
                np.float64,
            ),
        }
        model = gait.load_model(folder / "trace.json")
        model["basis"] = np.asarray(
            [r["body"]["basis"] for r in read(folder / "trace.json")["samples"]]
        )
        model["adhesion"] = np.asarray(
            [r["adhesion"] for r in read(folder / "trace.json")["samples"]], bool
        )
        native = saved.load_selected(
            folder / "native-contact.npz",
            {
                key: {
                    "shape": [15000, 6]
                    if key in {"normal_load_bw", "slip_mm_s", "slip_valid"}
                    else [15000],
                    "dtype": "|b1"
                    if key in {"slip_valid", "nonfoot_contact"}
                    else "<f8",
                    "finite": key != "slip_mm_s",
                }
                for key in NATIVE_KEYS
            },
        )
        arrays, metrics = saved.physical_metrics(
            raw, native, model, data["curves"], data["startup"], gait_module=gait
        )
        metrics["descriptive"] = descriptive(metrics["gait"])
        metrics["F5_separate"] = {
            leg: {
                "minimum_deg": float(minimum),
                "span_deg": float(metrics["geometry_knee_span_deg"][index]),
                "passed": bool(metrics["geometry_knee_span_deg"][index] >= minimum),
            }
            for leg, index, minimum in (
                ("lf", 0, 25),
                ("rf", 3, 30),
                ("lh", 2, 40),
                ("rh", 5, 35),
            )
        }
        return arrays, metrics


def _versions():
    from flylab.c.storage import runtime_versions

    return runtime_versions()


def validate_action(row):
    q = typed(np.asarray(row["joint_targets_rad"], np.float64), (42,), "<f8", "row q")
    if np.any(np.abs(q) > 10):
        raise ValueError("row request magnitude")
    typed(np.asarray(row["adhesion"]), (6,), "|b1", "row adhesion")
    tendon = row["tendon_inputs"]
    if (
        not isinstance(tendon, dict)
        or len(tendon) != 8
        or not np.isfinite(list(tendon.values())).all()
    ):
        raise ValueError("full eight-channel tendon dictionary required")
    force = row["neural_request"]["external_force_at_start"]
    if set(force) != {"push_left_s", "push_force"} or not math.isfinite(
        force["push_left_s"]
    ):
        raise ValueError("explicit source external force required")
    typed(np.asarray(force["push_force"], np.float64), (3,), "<f8", "external force")
    for key, shape in (("joint_stiffness_scale", (42,)), ("adhesion_control", (6,))):
        if key in row:
            typed(np.asarray(row[key], np.float64), shape, "<f8", key)


def applied_action(body, row, command):
    validate_action(row)
    force = row["neural_request"]["external_force_at_start"]
    body.push_left = force["push_left_s"]
    body.push_force = np.asarray(force["push_force"], np.float64).copy()
    kwargs = {
        key: copy.deepcopy(row[key])
        for key in ("joint_stiffness_scale", "adhesion_control")
        if key in row
    }
    body.step_joint_targets(
        command,
        np.asarray(row["adhesion"], bool),
        0.005,
        tendon_inputs=copy.deepcopy(row["tendon_inputs"]),
        interpolate=True,
        **kwargs,
    )


def _error(stage, error):
    return {"stage": stage, "type": type(error).__name__, "message": str(error)}


class InstancePhysicsCounter:
    """Track only an already-owned instance, including unreturned native steps."""

    def __init__(
        self, body, phase, *, journal=None, timely=lambda: None, fail=lambda *_: None
    ):
        self.body, self.phase = body, phase
        self.journal, self.timely, self.fail = journal, timely, fail
        self.had_method = "_step_physics" in vars(body)
        self.previous_attribute = vars(body).get("_step_physics")
        self.original = getattr(body, "_step_physics", None)
        self.proof = {
            "tracking_available": callable(self.original),
            "steps_requested": 0 if callable(self.original) else None,
            "steps_returned": 0 if callable(self.original) else None,
            "absolute_before_s": None,
            "absolute_after_s": None,
            "absolute_clock_known": False,
            "actual_advance_s": None,
            "clock_read_errors": [],
            "method_restored": False,
        }
        self.read_clock("before_init_or_science", initial=True)
        if callable(self.original):
            body._step_physics = self.step

    def read_clock(self, stage, *, initial=False):
        try:
            value = float(self.body.d.time)
            if not math.isfinite(value):
                raise ValueError("nonfinite absolute physics clock")
        except Exception as error:  # noqa: BLE001
            self.proof["clock_read_errors"].append(_error(stage, error))
            self.proof["absolute_clock_known"] = False
            self.proof["absolute_after_s"] = None
            self.proof["actual_advance_s"] = None
            return None
        if initial or (
            self.proof["absolute_before_s"] is None and stage.endswith("_step_before")
        ):
            self.proof["absolute_before_s"] = value
        self.proof["absolute_after_s"] = value
        self.proof["absolute_clock_known"] = True
        self.proof["actual_advance_s"] = (
            None
            if self.proof["absolute_before_s"] is None
            else value - self.proof["absolute_before_s"]
        )
        return value

    def record(self, event):
        if self.journal is not None:
            self.journal(event)

    def step(self, *args, **kwargs):
        self.proof["steps_requested"] += 1
        before = self.read_clock(self.phase + "_step_before")
        if self.phase == "constructor" and self.proof["steps_requested"] == 1:
            self.record(
                {
                    "event": "constructor_step_requested",
                    "step": self.proof["steps_requested"],
                    "absolute_before_s": before,
                    "absolute_clock_known": before is not None,
                }
            )
        returned = False
        try:
            value = self.original(*args, **kwargs)
            returned = True
            self.proof["steps_returned"] += 1
            return value
        except Exception as error:
            self.fail(self.phase + "_native_step", error)
            raise
        finally:
            after = self.read_clock(self.phase + "_step_after")
            if self.phase == "constructor" and (
                not returned or self.proof["steps_requested"] % 250 == 0
            ):
                try:
                    self.record(
                        {
                            "event": "constructor_step_finished_or_partial",
                            "step": self.proof["steps_requested"],
                            "python_returned": returned,
                            "steps_requested": self.proof["steps_requested"],
                            "steps_returned": self.proof["steps_returned"],
                            "absolute_after_s": after,
                            "absolute_clock_known": after is not None,
                            "actual_advance_s": self.proof["actual_advance_s"],
                            "unrecorded_steps_after_this_record": "unknown",
                        }
                    )
                except Exception as secondary:
                    self.fail("constructor_journal", secondary)
                    raise
            if self.proof["steps_requested"] % 50 == 0:
                self.timely()

    def restore(self):
        self.read_clock(self.phase + "_final_absolute_clock")
        if callable(self.original):
            if self.had_method:
                self.body._step_physics = self.previous_attribute
            else:
                del self.body._step_physics
        self.proof["method_restored"] = True


def replay_arm(
    label,
    data,
    command,
    output,
    dependencies,
    *,
    timely=lambda: None,
    writer=put_json,
    saver=save_npz,
    original=False,
):
    """Own the object before __init__, journal each requested/returned action."""
    output.mkdir()
    rows, samples, events, errors = [], [], [], []
    ledger = {
        "schema": "flylab.m3.e337.arm-ledger.v1",
        "arm": label,
        "construction_requested": True,
        "construction_returned": False,
        "controls_requested": 0,
        "controls_returned": 0,
        "native_samples_acquired": 0,
        "partial_physics_known": True,
        "actual_physical_s": None,
        "health_stop": False,
        "first_error": None,
        "errors": errors,
        "cleanup_completed": False,
    }
    body = detach = construction_counter = science_counter = None
    chunks, chunk_start = [], 0
    journal = (output / "control-journal.jsonl").open("x", encoding="utf8")

    def fail(stage, error):
        record = _error(stage, error)
        errors.append(record)
        if ledger["first_error"] is None:
            ledger["first_error"] = record

    def append(value):
        journal.write(json.dumps(value, allow_nan=False) + "\n")
        journal.flush()
        os.fsync(journal.fileno())

    def flush_chunk():
        nonlocal chunk_start
        if len(samples) == chunk_start:
            return
        name = "native-chunk-" + str(len(chunks)).zfill(3) + ".npz"
        saver(output / name, native_arrays(samples[chunk_start:]))
        chunks.append(
            {
                "path": name,
                "sha256": sha256(output / name),
                "first_sample": chunk_start,
                "sample_count": len(samples) - chunk_start,
            }
        )
        chunk_start = len(samples)
        writer(
            output / "native-chunks.json",
            {"schema": "flylab.m3.e337.native-chunks.v1", "completed": chunks},
        )

    def observer(sample):
        # Acquisition precedes validation/guard, preserving the failing sample.
        samples.append(copy.deepcopy(sample))
        ledger["native_samples_acquired"] = len(samples)
        native = native_arrays([sample])
        verify_native(native)
        if sample.get("numeric_fault"):
            raise ValueError("native numeric/body fault")
        if native["nonfoot_contact"][0] or native["upright"][0] < 0.9:
            raise HealthStop("measured native health stop")
        if len(samples) % 50 == 0:
            timely()

    try:
        timely()
        cls = dependencies.body_class(data)
        body = cls.__new__(cls)  # Ownership exists even if initialization raises.
        args, kwargs = dependencies.body_arguments(data)
        construction_counter = InstancePhysicsCounter(
            body, "constructor", journal=append, timely=timely, fail=fail
        )
        ledger["construction_physics"] = construction_counter.proof
        writer(output / "construction-ledger.json", construction_counter.proof)
        append(
            {
                "event": "constructor_requested",
                "absolute_clock": copy.deepcopy(construction_counter.proof),
            }
        )
        try:
            cls.__init__(body, *args, **kwargs)
            ledger["construction_returned"] = True
        except Exception as error:
            fail("constructor", error)
            raise
        finally:
            try:
                construction_counter.restore()
            except Exception as error:
                fail("constructor_method_restore", error)
                raise
            append(
                {
                    "event": "constructor_finished_or_partial",
                    "python_returned": ledger["construction_returned"],
                    "absolute_clock": copy.deepcopy(construction_counter.proof),
                }
            )
            writer(output / "construction-ledger.json", construction_counter.proof)
        if original and (
            not construction_counter.proof["tracking_available"]
            or construction_counter.proof["steps_requested"] != 2000
            or construction_counter.proof["steps_returned"] != 2000
            or not construction_counter.proof["absolute_clock_known"]
            or construction_counter.proof["actual_advance_s"] is None
            or abs(construction_counter.proof["actual_advance_s"] - 0.2) > 1e-9
        ):
            raise ValueError("original constructor2000/.2 settling proof required")
        ledger["construction_returned"] = True
        ledger["configuration"] = dependencies.configure(body, data)
        body.restore(copy.deepcopy(data["body_state"]))
        if hasattr(body, "cpg_state"):
            ledger["restored_cpg"] = copy.deepcopy(body.cpg_state())
        science_counter = InstancePhysicsCounter(
            body, "science", timely=timely, fail=fail
        )
        ledger["scientific_physics"] = science_counter.proof
        if original and not science_counter.proof["tracking_available"]:
            raise ValueError(
                "original scientific native requested/returned tracker required"
            )
        detach = dependencies.install(body, observer)
        rows.append(dependencies.boundary(body, data["rows"][0]))
        append({"event": "anchor", "row": rows[0]})
        for c in range(1, 301):
            timely()
            event = {
                "control_tick": c,
                "requested": True,
                "python_returned": False,
                "physics_time_before_s": float(body.physics_time()),
                "physics_time_after_s": None,
                "acquired_before": len(samples),
                "acquired_after": None,
                "clock_error": None,
                "command": command[c].tolist(),
                "external_force": copy.deepcopy(
                    data["rows"][c]["neural_request"]["external_force_at_start"]
                ),
            }
            events.append(event)
            ledger["controls_requested"] += 1
            append(dict(event="requested", **event))
            try:
                applied_action(body, data["rows"][c], command[c])
                event["python_returned"] = True
                ledger["controls_returned"] += 1
            except Exception as error:
                ledger["health_stop"] = isinstance(error, HealthStop)
                fail("replay", error)
                raise
            finally:
                event["acquired_after"] = len(samples)
                try:
                    event["physics_time_after_s"] = float(body.physics_time())
                    ledger["actual_physical_s"] = event["physics_time_after_s"]
                except Exception as error:  # noqa: BLE001
                    ledger["partial_physics_known"] = False
                    event["clock_error"] = _error("actual_clock", error)
                    fail("actual_clock", error)
                try:
                    append(dict(event="finished_or_partial", **event))
                except Exception as error:
                    fail("control_journal", error)
                    raise
            timely()
            if (
                event["physics_time_after_s"] is None
                or abs(event["physics_time_after_s"] - c * 0.005) > 2e-8
                or event["acquired_after"] - event["acquired_before"] != 50
            ):
                raise ValueError("actual control clock/native50 differs")
            row = dependencies.boundary(body, data["rows"][c])
            rows.append(row)
            append({"event": "boundary", "row": row})
            if c % 25 == 0:
                flush_chunk()
            if row.get("fault"):
                raise ValueError("boundary body fault")
        if original and (
            science_counter.proof["steps_requested"] != 15000
            or science_counter.proof["steps_returned"] != 15000
        ):
            raise ValueError(
                "original scientific15000 requested/returned proof required"
            )
    except Exception as error:  # noqa: BLE001
        ledger["health_stop"] = isinstance(error, HealthStop)
        if not errors or errors[-1]["message"] != str(error):
            fail("replay", error)
    finally:
        if body is not None and "restored_cpg" in ledger:
            try:
                if body.cpg_state() != ledger["restored_cpg"]:
                    raise ValueError("restored CPG state changed without authorization")
            except Exception as error:  # noqa: BLE001
                fail("cpg_guard", error)
        for stage, callback in (
            ("native_chunk", flush_chunk),
            ("hook_restore", detach),
            (
                "constructor_method_restore",
                construction_counter.restore
                if construction_counter
                and not construction_counter.proof["method_restored"]
                else None,
            ),
            (
                "scientific_method_restore",
                science_counter.restore if science_counter else None,
            ),
            ("body_close", getattr(body, "close", None)),
            ("journal_close", journal.close),
        ):
            if callback is not None:
                try:
                    callback()
                except Exception as error:  # noqa: BLE001
                    fail(stage, error)
        ledger["cleanup_completed"] = not any(
            e["stage"]
            in {
                "hook_restore",
                "scientific_method_restore",
                "constructor_method_restore",
                "body_close",
                "journal_close",
            }
            for e in errors
        )
        trace = {
            k: copy.deepcopy(data["metadata"].get(k))
            for k in (
                "parameters",
                "body_hash",
                "model_hash",
                "adapter_hash",
                "graph_hash",
            )
        }
        trace.update(
            schema="flylab.m3.e337.body-only-trace.v1",
            samples=rows,
            neural_execution=0,
            source_neural_metadata_is_label_only=True,
        )
        for stage, callback in (
            ("trace_save", lambda: writer(output / "trace.json", trace)),
            (
                "native_save",
                lambda: saver(output / "native-contact.npz", native_arrays(samples)),
            ),
            ("arm_ledger", lambda: writer(output / "advance-ledger.json", ledger)),
        ):
            try:
                callback()
            except Exception as error:  # noqa: BLE001
                fail(stage, error)
    return {
        "rows": rows,
        "native": native_arrays(samples),
        "ledger": ledger,
        "events": events,
    }


def recursive_bytes(output):
    return sum(p.stat().st_size for p in Path(output).rglob("*") if p.is_file())


def file_manifest(output):
    return {
        p.relative_to(output).as_posix(): {
            "sha256": sha256(p),
            "bytes": p.stat().st_size,
        }
        for p in sorted(Path(output).rglob("*"))
        if p.is_file()
        and not (
            p.parent == output
            and p.name in {"result.json", "advance-ledger.json", "output-manifest.json"}
        )
        and not p.name.endswith(".pending")
    }


def usage():
    owner = psutil.Process()
    total = owner.memory_info().rss
    for child in owner.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total


def run(
    freeze_path=None,
    freeze_sha256=None,
    review_path=None,
    review_sha256=None,
    output=None,
    *,
    root=ROOT,
    test_dimensions=None,
    dependencies=None,
    clock=None,
    writer=None,
    saver=None,
    meter=None,
    reader=None,
    emergency_writer=None,
):
    """One fixed sequential job. Original mode rejects every injected hook."""
    injected = (dependencies, clock, writer, saver, meter, reader, emergency_writer)
    if test_dimensions is None and any(v is not None for v in injected):
        raise ValueError("original mode forbids custom dependencies/clock/sinks")
    if test_dimensions is not None and not isinstance(test_dimensions, TestDimensions):
        raise TypeError("explicit generated TestDimensions required")
    generated = test_dimensions is not None
    if generated and (
        dependencies is None
        or getattr(type(dependencies), "load", None) is Dependencies.load
    ):
        raise ValueError("generated mode requires an explicit generated loader")
    clock = clock or time.perf_counter
    writer, saver, reader = writer or put_json, saver or save_npz, reader or read
    emergency_writer = emergency_writer or put_json
    dependencies, meter = dependencies or Dependencies(), meter or usage
    s0 = clock()  # Pre-gate start; never reset on either arm or repair.
    science_deadline = s0 + 60
    output = Path(output or saved.safe(OUTPUT, root))
    errors = []
    result = {
        "schema": RESULT_SCHEMA,
        "analysis_valid": False,
        "effective_validity": False,
        "classification": "technical_invalid",
        "science": None,
        "first_error": None,
        "errors": [],
        "publication_valid": False,
        "disk_agreement_established": False,
        "root_independent_raw_audit": False,
        "original_calls": 0,
        "original_job_requested": not generated,
        "neural_calls": 0,
        "CUDA_calls": 0,
        "candidate_adopted": False,
        "F1_F8_complete": False,
    }
    ledger = {
        "schema": "flylab.m3.e337.advance-ledger.v1",
        "numerical_reads": [],
        "arms": {},
        "s0": s0,
        "science_deadline": science_deadline,
        "final_deadline": None,
        "errors": [],
        "first_error": None,
        "generated": generated,
        "body_constructors_requested": 0,
    }
    final_deadline = None

    def fail(stage, error):
        record = _error(stage, error)
        errors.append(record)
        if result["first_error"] is None:
            result["first_error"] = record
        result.update(
            analysis_valid=False,
            effective_validity=False,
            publication_valid=False,
            classification="technical_invalid",
        )

    def check():
        deadline = science_deadline if final_deadline is None else final_deadline
        if clock() > deadline:
            raise TimeoutError("absolute E337 deadline exceeded")
        if meter() > LIMITS["host_bytes"]:
            raise MemoryError("host tree2GiB exceeded")
        cap = 64 << 20 if generated else LIMITS["output_bytes"]
        if output.exists() and recursive_bytes(output) > cap:
            raise OSError("output128MiB exceeded")
        if shutil.disk_usage(output.parent).free < LIMITS["minimum_free_disk_bytes"]:
            raise OSError("free512MiB required")

    try:
        if generated:
            if output.exists():
                raise ValueError("fresh generated output required")
            freeze = {}
        else:
            freeze = gate(
                freeze_path,
                freeze_sha256,
                review_path,
                review_sha256,
                output,
                root=root,
                timely=check,
            )
            result["original_calls"] = 1
        check()
        output.mkdir()
        data = dependencies.load(
            freeze, root=root, timely=check, reads=ledger["numerical_reads"]
        )
        plan = shape_plan(data["command"], data["live"], data["curves"])
        saver(output / "shape-plan.npz", plan)
        check()
        for label in ("baseline", "candidate"):
            ledger["body_constructors_requested"] += 1
            raw = replay_arm(
                label,
                data,
                data["command"] if label == "baseline" else plan["command"],
                output / label,
                dependencies,
                timely=check,
                writer=writer,
                saver=saver,
                original=not generated,
            )
            ledger["arms"][label] = raw["ledger"]
            phases = [
                arm.get(key, {})
                for arm in ledger["arms"].values()
                for key in ("construction_physics", "scientific_physics")
            ]
            for field in ("steps_requested", "steps_returned"):
                values = [phase.get(field) for phase in phases]
                ledger["total_native_" + field] = (
                    sum(values) if all(v is not None for v in values) else None
                )
            # Raw is saved before replay validation or scientific classification.
            arm = raw["ledger"]
            if label == "baseline":
                if arm["first_error"] is not None:
                    errors.extend(copy.deepcopy(arm["errors"]))
                    result["first_error"] = copy.deepcopy(arm["first_error"])
                    raise ValueError(
                        "baseline incomplete/failed: " + str(arm["first_error"])
                    )
                compare_replay(
                    raw["rows"], raw["native"], data["rows"], data["native"], 300
                )
                baseline = raw
                ledger["baseline_exact"] = True
            else:
                if arm["first_error"] is not None and len(raw["rows"]) < 101:
                    errors.extend(copy.deepcopy(arm["errors"]))
                    result["first_error"] = copy.deepcopy(arm["first_error"])
                    raise ValueError("candidate prefix incomplete")
                compare_replay(
                    raw["rows"][:101],
                    {k: v[:5000] for k, v in raw["native"].items()},
                    baseline["rows"],
                    baseline["native"],
                    100,
                )
                ledger["prefix_exact"] = True
                if arm.get("configuration") != baseline["ledger"].get("configuration"):
                    raise ValueError("actual configured arm identities differ")
                measured_stop = (
                    arm["health_stop"]
                    and arm["cleanup_completed"]
                    and len(raw["rows"]) >= 101
                )
                if arm["first_error"] is not None and not (
                    measured_stop
                    and len(arm["errors"]) == 1
                    and arm["first_error"]["type"] == "HealthStop"
                ):
                    errors.extend(copy.deepcopy(arm["errors"]))
                    result["first_error"] = copy.deepcopy(arm["first_error"])
                    raise ValueError(
                        "candidate technical failure: " + str(arm["first_error"])
                    )
                if measured_stop:
                    result["science"] = classify({}, {}, health_rejection=True)
                else:
                    verify_native(raw["native"], count=15000)
                    metrics = {}
                    for name in ("baseline", "candidate"):
                        arrays, evaluation = dependencies.evaluate(output / name, data)
                        saver(output / name / "geometry.npz", arrays)
                        writer(output / name / "evaluation.json", evaluation)
                        metrics[name] = evaluation
                    result["metrics"] = metrics
                    result["science"] = classify(
                        metrics["candidate"]["official"],
                        metrics["baseline"]["official"],
                    )
                result.update(
                    analysis_valid=True, classification=result["science"]["verdict"]
                )
            check()
        if not generated:
            if result["classification"] != "health_rejection" and (
                ledger["total_native_steps_requested"] != 34000
                or ledger["total_native_steps_returned"] != 34000
            ):
                raise ValueError("two original arms total34000 incl settling required")
            saved._pin_map(freeze["source_sha256"], root, check)
            saved._pin_map(freeze["input_sha256"], root, check)
            writer(output / "source-manifest.json", freeze)
        check()
    except Exception as error:  # noqa: BLE001
        fail("science", error)
    final_begin = clock()
    final_deadline = min(s0 + 70, final_begin + 10)
    ledger.update(final_begin=final_begin, final_deadline=final_deadline)

    def sync():
        result["errors"] = copy.deepcopy(errors)
        ledger.update(
            first_error=result["first_error"],
            errors=copy.deepcopy(errors),
            analysis_valid=result["analysis_valid"],
            effective_validity=result["effective_validity"],
            elapsed_s=clock() - s0,
        )

    def publish(sink, *, repair=False):
        # Invalid repair preserves a coherent terminal result even after the
        # deadline. It never changes the recorded absolute deadline or validity.
        if not repair:
            check()
            result.update(
                publication_valid=True,
                effective_validity=result["analysis_valid"],
                disk_agreement_established=True,
            )
        sync()
        manifest = {
            "schema": "flylab.m3.e337.output-manifest.v1",
            "files": file_manifest(output),
            "analysis_valid": result["analysis_valid"],
            "external_completion_required": True,
        }
        sink(output / "advance-ledger.json", ledger)
        sink(output / "result.json", result)
        sink(output / "output-manifest.json", manifest)
        if (
            reader(output / "result.json") != result
            or reader(output / "advance-ledger.json") != ledger
        ):
            raise ValueError("terminal readback differs")
        if (
            reader(output / "output-manifest.json") != manifest
            or file_manifest(output) != manifest["files"]
        ):
            raise ValueError("terminal raw manifest/readback differs")
        if not repair:
            check()
        else:
            # Still observe terminal time/resources without granting validity.
            ledger["repair_finished_after_deadline"] = clock() > final_deadline

    try:
        if not output.is_dir():
            raise OSError("owned output unavailable")
        publish(writer)
    except Exception as error:  # noqa: BLE001
        fail("publication", error)
        result["disk_agreement_established"] = False
        for attempt in range(2):
            try:
                publish(emergency_writer, repair=True)
                result["disk_agreement_established"] = True
                sync()
                emergency_writer(output / "result.json", result)
                emergency_writer(output / "advance-ledger.json", ledger)
                # Every last repair write is verified; returning sinks can
                # publish corrupt bytes without raising an I/O exception.
                if (
                    reader(output / "result.json") != result
                    or reader(output / "advance-ledger.json") != ledger
                    or reader(output / "output-manifest.json")
                    != {
                        "schema": "flylab.m3.e337.output-manifest.v1",
                        "files": file_manifest(output),
                        "analysis_valid": result["analysis_valid"],
                        "external_completion_required": True,
                    }
                ):
                    raise ValueError("final repair readback differs")
                clock()  # Observe the unchanged absolute clock after final reads.
                break
            except Exception as secondary:  # noqa: BLE001
                fail("repair" + str(attempt + 1), secondary)
                result["disk_agreement_established"] = False
    if not result["disk_agreement_established"]:
        sync()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "freeze-sha256", "review", "review-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output", default=OUTPUT)
    args = parser.parse_args(argv)
    result = run(
        args.freeze,
        args.freeze_sha256,
        args.review,
        args.review_sha256,
        saved.safe(args.output, ROOT),
    )
    print(json.dumps(result, allow_nan=False))
    return 0 if result["effective_validity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
