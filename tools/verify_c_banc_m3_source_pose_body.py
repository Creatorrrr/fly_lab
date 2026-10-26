"""E350 source-live pose recovery; new observer/publisher are CPU-qualified only."""

from __future__ import annotations

import time

ENTRY = time.perf_counter()
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "verification/banc-natural-gait-m3-20260919"
OWNED = (
    "tools/verify_c_banc_m3_source_pose_body.py",
    "tests/test_c_banc_source_pose_body.py",
)
OUTPUT = BASE + "/e350-source-pose-body"
REGISTRATION = BASE + "/parent-e350-selective-input-registration.json"
REGISTRATION_SHA = "bd1611d4c88abc316ad3b0fc65be19e721b633d250f8af8fc24819b9ed427666"
REFLECTION = BASE + "/parent-e350-reflected-design-registration.json"
REFLECTION_SHA = "e5cc5dd63a7da7284d5083e80ab50a35226a94a09670e307fcd2d1c655241770"
FORCE_SOURCE = "tools/verify_c_banc_m3_held_force_body.py"
FORCE_SHA = "8990f10a6319e30c5a8e730801eacda5925b093816eb9fed073f33c635063352"
_FORCE = None


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def force():
    global _FORCE
    if _FORCE is None:
        file = ROOT / FORCE_SOURCE
        if sha(file.read_bytes()) != FORCE_SHA:
            raise ValueError("unchanged E349 force source differs")
        spec = importlib.util.spec_from_file_location("e350_unchanged_force", file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _FORCE = module
    return _FORCE


def shared():
    return force().shared()


def own_sources(root=ROOT):
    return {name: sha(shared().path(name, root).read_bytes()) for name in OWNED}


def metadata_preflight(root=ROOT):
    core = shared()
    for name, pin in ((REGISTRATION, REGISTRATION_SHA), (REFLECTION, REFLECTION_SHA)):
        if sha(core.path(name, root).read_bytes()) != pin:
            raise ValueError("E350 reflected registration differs")
    reg = core.read_json(core.path(REGISTRATION, root))
    reflected = core.read_json(core.path(REFLECTION, root))
    previous = force().metadata_preflight(root)
    if (
        reg["schema"] != "flylab.m3.e350.selective-input-registration.v1"
        or reflected["original_authorized"] is not False
        or reg["payloads"] != previous["payloads"]
        or reg["force_policy_value"] != previous["force_policy_value"]
        or reg["inherited_native_routine_AST_sha256"]
        != previous["inherited_native_routine_AST_sha256"]
    ):
        raise ValueError("unchanged input/science inheritance differs")
    for key in (
        "design",
        "actual_review",
        "E349_registration",
        "observation_policy",
        "current_force_qualification",
        "E349_final_decision",
    ):
        spec = reg[key]
        file = core.path(spec["path"], root)
        if (
            file.stat().st_size != spec["bytes"]
            or sha(file.read_bytes()) != spec["sha256"]
        ):
            raise ValueError("finite E350 descriptor differs: " + key)
    for spec in reg["current_force_sources"].values():
        file = core.path(spec["path"], root)
        if (
            file.stat().st_size != spec["bytes"]
            or sha(file.read_bytes()) != spec["sha256"]
        ):
            raise ValueError("protected force source differs")
    policy = core.checked_json(reg["observation_policy"], root)
    if policy != reg["observation_policy_value"]:
        raise ValueError("exact source-pose policy differs")
    for name, pin in policy["source_sha256"].items():
        if previous["force_policy_value"]["source_sha256"].get(name) != pin:
            raise ValueError("reviewed pose source chain differs")
    core.checked_json(policy["E349_diagnostic"], root)
    return reg


def load_original(registration, *, root=ROOT, timely=lambda: None):
    return force().load_original(registration, root=root, timely=timely)


def error_record(stage, error):
    return {"stage": stage, "type": type(error).__name__, "message": str(error)[:1024]}


class SourcePoseDependencies(force().HeldForceDependencies):
    def __init__(self, policy, output, observation_policy, *, contact_reader=None):
        super().__init__(policy, output)
        self.observation_policy = copy.deepcopy(observation_policy)
        self.contact_reader = contact_reader
        self.observations = {}
        # NativeDependencies.__init__ installs an instance install attribute.
        self.install = self.install_source_pose

    def install_source_pose(self, body, callback):
        np, _, _ = shared().libraries()
        if self.contact_reader is None:
            from flylab.c.walking_trace import contact_diagnostics

            contact = contact_diagnostics
        else:
            contact = self.contact_reader
        arm = self._body_arm[id(body)]
        records = self.observations.setdefault(arm, [])
        original = body._step_physics
        shadow = copy.copy(body)
        shadow.d = body.mj.MjData(body.m)
        typ = body.mj.mjtState.mjSTATE_INTEGRATION
        state = np.empty(body.mj.mj_stateSize(body.m, typ))

        def step():
            original()
            record = {
                "native_index": len(records) + 1,
                "time_s": None,
                "source_upright": None,
                "current_upright": None,
                "source_available": False,
                "current_available": False,
                "time_available": False,
                "callback_attempted": False,
                "callback_returned": False,
                "errors": [],
            }
            records.append(record)  # Actual step, including failed final observation.
            try:
                stamp = float(body.physics_time())
                if not math.isfinite(stamp):
                    raise ValueError("nonfinite actual native time")
                record.update(time_s=stamp, time_available=True)
            except Exception as error:  # noqa: BLE001 -- actual time read availability
                record["errors"].append(error_record("time_read", error))
            for key, reader in (("source", lambda: body.pose()),):
                try:
                    value = float(reader()[1][2, 2])
                    if not math.isfinite(value):
                        raise ValueError(
                            "nonfinite source-live upright: " + repr(value)
                        )
                    record[key + "_upright"] = value
                    record[key + "_available"] = True
                except Exception as error:  # noqa: BLE001 -- actual pose read availability
                    record["errors"].append(error_record(key + "_read", error))
            try:
                body.mj.mj_getState(body.m, body.d, state, typ)
                body.mj.mj_setState(body.m, shadow.d, state, typ)
                body.mj.mj_forward(body.m, shadow.d)  # Shadow only.
                value = float(type(shadow).pose(shadow)[1][2, 2])
                if not math.isfinite(value):
                    raise ValueError(
                        "nonfinite current-forwarded upright: " + repr(value)
                    )
                record.update(current_upright=value, current_available=True)
            except Exception as error:  # noqa: BLE001 -- retain acquired step on shadow read failure
                record["errors"].append(error_record("current_read", error))
            if record["errors"]:
                raise ValueError(
                    "source-pose observation unavailable: " + str(record["errors"])
                )
            feet = contact(shadow)
            slip = [np.nan if value is None else value for value in feet["slip_mm_s"]]
            sample = {
                "time_s": record["time_s"],
                "normal_load_bw": feet["normal_load_bw"],
                "slip_mm_s": slip,
                "slip_valid": np.isfinite(slip).tolist(),
                "nonfoot_contact": bool(shadow.nonfoot_contact()),
                "upright": record["source_upright"],
                "numeric_fault": bool(
                    not np.isfinite(body.d.qpos).all()
                    or not np.isfinite(body.d.qvel).all()
                    or body.fault
                ),
            }
            record["callback_attempted"] = True
            try:
                callback(sample)
                record["callback_returned"] = True
            except Exception as error:
                record["errors"].append(error_record("callback", error))
                raise
            if sample["numeric_fault"]:
                raise ValueError("native body numeric/fault state")
            if record["current_upright"] < 0.9:
                raise shared().HealthStop(
                    "current-forwarded native physical health-stop"
                )

        body._step_physics = step
        return lambda: setattr(body, "_step_physics", original)


def descriptor(file, *, relative=None):
    return {
        "path": relative or file.name,
        "bytes": file.stat().st_size,
        "sha256": sha(file.read_bytes()),
    }


def compact_publish(file, value):
    raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
    with file.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def publish_observations(deps, arm, folder, publisher):
    np, _, _ = shared().libraries()
    rows = deps.observations.get(arm, [])
    raw_file, meta_file = folder / "source-pose.npz", folder / "source-pose.json"
    if not raw_file.exists():
        arrays = {
            key: np.asarray(
                [r[key] if r[key] is not None else np.nan for r in rows],
                dtype=np.float64,
            )
            for key in ("time_s", "source_upright", "current_upright")
        }
        arrays["native_index"] = np.asarray(
            [r["native_index"] for r in rows], dtype=np.int64
        )
        for key in ("time_available", "source_available", "current_available"):
            arrays[key] = np.asarray([r[key] for r in rows], dtype=np.bool_)
        with raw_file.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
    if not meta_file.exists():
        publisher(
            meta_file,
            {
                "schema": "flylab.m3.e350.source-pose-observation.v1",
                "arm": arm,
                "source_sha256": own_sources(),
                "policy": deps.observation_policy,
                "raw": descriptor(raw_file),
                "count": len(rows),
                "source_available_count": sum(r["source_available"] for r in rows),
                "current_available_count": sum(r["current_available"] for r in rows),
                "time_available_count": sum(r["time_available"] for r in rows),
                "first": rows[0] if rows else None,
                "last": rows[-1] if rows else None,
                "errors": [
                    {"native_index": r["native_index"], "errors": r["errors"]}
                    for r in rows
                    if r["errors"]
                ],
                "current_health_passed": bool(rows)
                and all(
                    r["current_available"] and r["current_upright"] >= 0.9 for r in rows
                ),
                "recorded_health_passed": bool(rows)
                and all(
                    r["source_available"] and r["source_upright"] >= 0.9 for r in rows
                ),
                "orientation_replayed_by_independent_audit": False,
                "new_observer_native_qualified": False,
            },
        )
    return descriptor(meta_file)


def check_caps(folder, status, output):
    core = shared()
    cap = core.LIMITS["raw_arm_bytes" if status == "completed" else "failure_bytes"]
    if core.usage(folder) > cap:
        raise ValueError("E350 inclusive raw/receipt/evaluation arm cap exceeded")
    if core.usage(output) > core.LIMITS["stage_bytes"]:
        raise ValueError("E350 inclusive stage cap exceeded")


@contextmanager
def publication_scope(output, deps, final_timely=lambda: None):
    """Sole-worker scope. Exact trace paths only; restore core binding in finally."""
    core = shared()
    previous = core.publish
    folders = {Path(output) / arm: arm for arm in ("B", "I", "C")}

    def publish(file, value):
        file = Path(file)
        arm = folders.get(file.parent)
        if arm and file.name == "trace.json":
            return compact_publish(file, value)
        if arm and file.name in ("receipt.json", "receipt.json.pending"):
            value["source_pose_observation"] = publish_observations(
                deps, arm, file.parent, previous
            )
        previous(file, value)
        if arm and file.name in ("receipt.json", "evaluation.json"):
            status = (
                value["status"]
                if file.name == "receipt.json"
                else core.read_json(file.parent / "receipt.json")["status"]
            )
            check_caps(file.parent, status, output)
            final_timely()  # Receipt and evaluation stream close before next arm.

    core.publish = publish
    try:
        yield
    finally:
        core.publish = previous


def run_loaded(
    data,
    output,
    registration,
    *,
    timely=lambda: None,
    final_timely=lambda: None,
    dependencies=None,
    engine_run=None,
):
    core, old = shared(), force()
    output = Path(output)
    deps = dependencies or SourcePoseDependencies(
        registration["force_policy_value"],
        output,
        registration["observation_policy_value"],
    )
    with publication_scope(output, deps, final_timely):
        result = old.run_loaded(
            data,
            output,
            registration["force_policy_value"],
            timely=timely,
            final_timely=final_timely,
            dependencies=deps,
            engine_run=engine_run,
        )
    result["inherited_force_result_schema"] = result["schema"]
    result["schema"] = "flylab.m3.e350.source-pose-result.v1"
    result["source_sha256"] = own_sources()
    result["source_pose_observations"] = {}
    for arm, ledger in result["arms"].items():
        folder = output / arm
        try:
            spec = publish_observations(deps, arm, folder, core.publish)
            ledger["source_pose_observation"] = spec
            result["source_pose_observations"][arm] = spec
            core.update_receipt(folder / "receipt.json", ledger)
            check_caps(folder, ledger["status"], output)
        except Exception as error:  # noqa: BLE001 -- narrow owned arm publication/cap boundary
            core.record_error(result, "source_pose_finalization", error)
            core.record_error(ledger, "source_pose_finalization", error)
            ledger["status"] = result["status"] = "technical_invalid"
            result["analysis_completed"] = False
            result["H348"] = core.hypothesis(result["metrics"], False)
            try:
                core.update_receipt(folder / "receipt.json", ledger)
            except Exception as secondary:  # noqa: BLE001 -- failed owned receipt sink
                core.record_error(result, "source_pose_final_receipt", secondary)
    try:
        core.update_receipt(output / "result.json", result)
        if core.usage(output) > core.LIMITS["stage_bytes"]:
            raise ValueError("E350 inclusive final stage cap exceeded")
        final_timely()
    except Exception as error:  # noqa: BLE001 -- narrow final publication/deadline boundary
        core.record_error(result, "source_pose_result_publication", error)
        result["status"] = "technical_invalid"
        result["analysis_completed"] = False
        result["H348"] = core.hypothesis(result["metrics"], False)
        try:
            core.update_receipt(output / "result.json", result)
        except Exception as secondary:  # noqa: BLE001 -- bounded final receipt attempt
            core.record_error(result, "source_pose_result_receipt", secondary)
    return result


def make_freeze(qualification, *, root=ROOT):
    core, reg = shared(), metadata_preflight(root)
    own = own_sources(root)
    qual = core.read_json(core.path(qualification, root), cap=256 << 10)
    checks = (
        "CPU",
        "ruff",
        "actual_install_binding",
        "live_class_dispatch",
        "last_sample",
        "unavailable_reads",
        "early_arm_caps",
        "publisher_restored",
        "compact_identity",
        "private_gate",
        "source_AST_reuse",
    )
    if (
        qual.get("schema") != "flylab.m3.e350.root-qualification.v1"
        or qual.get("source_sha256") != own
        or qual.get("passed") is not True
        or qual.get("original_calls") != 0
        or any(qual.get("checks", {}).get(k) is not True for k in checks)
    ):
        raise ValueError("current E350 ROOT qualification required")
    execution = qual["execution"]
    contract, supervision, full_exit = (
        core.checked_json(execution[k], root)
        for k in ("contract", "supervision", "full_exit")
    )
    if (
        any(contract["source_sha256"].get(k) != v for k, v in own.items())
        or contract.get("original_calls") != 0
        or supervision.get("success") is not True
        or supervision.get("original_calls") != 0
        or supervision.get("contract_sha256") != execution["contract"]["sha256"]
        or full_exit.get("all_owned_ended") is not True
        or full_exit.get("exit_code") != 0
        or full_exit.get("first_error") is not None
        or full_exit.get("secondary_errors")
        or not full_exit.get("owned_identities")
        or any(
            x.get("close") != "ended"
            or type(x.get("pid")) is not int
            or isinstance(x.get("create_time"), bool)
            or not isinstance(x.get("create_time"), (float, int))
            or not math.isfinite(x["create_time"])
            for x in full_exit["owned_identities"]
        )
        or len(contract["commands"]) != len(supervision["attempts"])
        or not contract["commands"]
        or any(
            a.get("argv") != j["argv"]
            or a.get("exit_code") != 0
            or a.get("killed") is not False
            for a, j in zip(supervision["attempts"], contract["commands"])
        )
    ):
        raise ValueError("current E350 supervised argv/source/PID closure required")
    cost = full_exit.get("charged_s")
    if (
        isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(cost)
        or not 0 < cost <= 20
    ):
        raise ValueError("finite generated cost required")
    return {
        "schema": "flylab.m3.e350.source-pose-freeze.v1",
        "source_sha256": own,
        "registration_sha256": REGISTRATION_SHA,
        "reflection_sha256": REFLECTION_SHA,
        "current_core": reg["current_core"],
        "current_force_sources": reg["current_force_sources"],
        "inherited_AST_sha256": reg["inherited_native_routine_AST_sha256"],
        "inherited_native_evidence": reg["inherited_actual_native_evidence"],
        "evaluate_API_exception": reg["evaluate_exception"],
        "observation_policy": reg["observation_policy"],
        "qualification": descriptor(
            core.path(qualification, root), relative=qualification
        ),
        "prospective": reg["prospective"],
        "original_authorized": False,
        "new_observer_publisher_gate_inherited_native_proof": False,
    }


def original_argv(freeze, review):
    return [
        ".venv/Scripts/python.exe",
        "-X",
        "utf8",
        "-B",
        OWNED[0],
        "--run",
        "--freeze",
        freeze,
        "--review",
        review,
        "--output",
        OUTPUT,
    ]


def original_gate(freeze_name, review_name, *, root=ROOT, output=OUTPUT):
    core, reg = shared(), metadata_preflight(root)
    freeze = core.read_json(core.path(freeze_name, root))
    if freeze != make_freeze(freeze["qualification"]["path"], root=root):
        raise ValueError("current E350 freeze differs")
    freeze_sha = sha(core.path(freeze_name, root).read_bytes())
    review = core.read_json(core.path(review_name, root), cap=256 << 10)
    if (
        review.get("schema") != "flylab.m3.e350.root-original-review.v1"
        or review.get("approved") is not True
        or review.get("source_sha256") != own_sources(root)
        or review.get("freeze_sha256") != freeze_sha
        or review.get("registration_sha256") != REGISTRATION_SHA
        or review.get("reflection_sha256") != REFLECTION_SHA
        or output != OUTPUT
        or review.get("output") != OUTPUT
    ):
        raise ValueError("current private E350 ROOT approval required")
    reservation = core.checked_json(review["reservation"], root)
    expected = {
        "schema": "flylab.m3.e350.root-reservation.v1",
        "cumulative_before_s": 9798.25293289951,
        "cumulative_limit_s": 9900,
        "reserved_s": 91,
        "body_wall_s": 85,
        "internal_s": 80,
        "science_s": 65,
        "finalization_s": 15,
        "audit_wall_s": 6,
        "audit_internal_s": 5,
        "original_calls": 1,
        "retry_count": 0,
        "source_sha256": own_sources(root),
        "freeze_sha256": freeze_sha,
        "registration_sha256": REGISTRATION_SHA,
        "reflection_sha256": REFLECTION_SHA,
        "review_path": review_name,
        "output": OUTPUT,
        "argv": original_argv(freeze_name, review_name),
    }
    if any(
        type(reservation.get(k)) is not type(v) or reservation.get(k) != v
        for k, v in expected.items()
    ):
        raise ValueError("exact E350 once/cumulative reservation linkage required")
    if (
        core.path(output, root).exists()
        or shutil.disk_usage(root).free < core.LIMITS["free_disk_bytes"]
        or any(
            os.environ.get(k) != "1"
            for k in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        )
    ):
        raise ValueError("fresh output/CPU1/free disk required")
    return reg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--qualification")
    parser.add_argument("--review")
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)

    def timely(final=False):
        if time.perf_counter() - ENTRY > (80 if final else 65):
            raise TimeoutError(
                "E350 inclusive import/read/science/observer/publication deadline"
            )
        import psutil

        if psutil.Process().memory_info().rss > shared().LIMITS["host_bytes"]:
            raise MemoryError("E350 host2GiB")

    if not args.run:
        if not args.qualification:
            parser.error("passive freeze requires --qualification")
        shared().publish(shared().path(args.freeze), make_freeze(args.qualification))
        timely(True)
        return 0
    reg = original_gate(args.freeze, args.review, output=args.output)
    output = shared().path(args.output)
    try:
        timely()
        data = load_original(reg, timely=timely)
    except Exception as error:  # noqa: BLE001 -- input failure precedes any constructor
        output.mkdir(parents=True, exist_ok=False)
        result = force().initial_result("input_load")
        result.update(
            body_constructors=0,
            scientific_physics_steps=0,
            array_read_counts=None,
            array_read_counts_status="not_instrumented",
        )
        shared().record_error(result, "input_load", error)
        result = force().publish_final(
            result,
            output,
            reg["force_policy_value"],
            None,
            final_timely=lambda: timely(True),
        )
        result.update(
            schema="flylab.m3.e350.source-pose-result.v1",
            source_sha256=own_sources(),
            source_pose_observations={},
        )
        shared().update_receipt(output / "result.json", result)
    else:
        result = run_loaded(
            data, output, reg, timely=timely, final_timely=lambda: timely(True)
        )
    timely(True)
    if result["status"] != "completed":
        print(
            json.dumps(
                {"first_error": result["first_error"], "errors": result["errors"]},
                allow_nan=False,
            ),
            file=sys.stderr,
        )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
