"""E349 checkpoint-held no-force recovery; unchanged E348 B/I/C science.

--freeze FILE --qualification ROOT_JSON creates a passive current-source freeze.
--run --freeze FILE --review ROOT_JSON --output FRESH_FIXED_DIR is the sole
prospective original argv. No original execution is authorized by this module.
The Python loader/spy seams are for generated qualification only.
"""

from __future__ import annotations

import time

ENTRY = time.perf_counter()  # Before inherited library imports/input reads.
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "verification/banc-natural-gait-m3-20260919"
OWNED = (
    "tools/verify_c_banc_m3_held_force_body.py",
    "tests/test_c_banc_held_force_body.py",
)
OUTPUT = BASE + "/e349-held-force-body"
CORE = "tools/probe_c_banc_m3_shape_shared_timing_body.py"
CORE_SHA = "97d3f518b90b9af4d6daefdb5431232e1649fbe9159a9ae0d63e4c866bdc9c2d"
REGISTRATION = BASE + "/parent-e349-selective-input-registration.json"
REGISTRATION_SHA = "b2e55a43352b195e19eb11d1dcb63f0e680bba13abc5f8ae47b495bfec910005"
REFLECTION = BASE + "/parent-e349-reflected-design-registration.json"
REFLECTION_SHA = "a577a9d2ce1ab3c50c4288feb3a73c469ac3510dfdaf46f70f9d93e79ed4c3a8"
TENDONS = tuple(leg + "_tarsus" for leg in ("lf", "lm", "lh", "rf", "rm", "rh")) + (
    "abdomen_pitch",
    "abdomen_yaw",
)
REUSED = (
    "libraries",
    "selected",
    "plan",
    "NativeDependencies",
    "body_trace",
    "replay",
    "run",
    "hypothesis",
)
_CORE = None


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def shared():
    global _CORE
    if _CORE is None:
        file = ROOT / CORE
        if sha(file.read_bytes()) != CORE_SHA:
            raise ValueError("current inherited core source differs")
        spec = importlib.util.spec_from_file_location("e349_unchanged_body_core", file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CORE = module
    return _CORE


def own_sources(root=ROOT):
    return {name: sha(shared().path(name, root).read_bytes()) for name in OWNED}


def metadata_preflight(root=ROOT):
    """Finite JSON/PY only. Never open a referenced scientific array/archive."""
    core = shared()
    for name, pin in (
        (CORE, CORE_SHA),
        (REGISTRATION, REGISTRATION_SHA),
        (REFLECTION, REFLECTION_SHA),
    ):
        file = core.path(name, root)
        if file.stat().st_size > 256 << 10 or sha(file.read_bytes()) != pin:
            raise ValueError("E349 current registration/source pin differs: " + name)
    reg = core.read_json(core.path(REGISTRATION, root))
    reflected = core.read_json(core.path(REFLECTION, root))
    if (
        reg.get("schema") != "flylab.m3.e349.selective-input-registration.v1"
        or reflected.get("original_authorized") is not False
    ):
        raise ValueError("ROOT reflected pre-execution registration required")
    for key in (
        "design",
        "actual_review",
        "current_core",
        "force_policy",
        "inherited_actual_native_evidence",
        "inherited_actual_native_full_exit",
    ):
        spec = reg[key]
        if spec["path"].endswith(".py"):
            file = core.path(spec["path"], root)
            if (
                file.stat().st_size != spec["bytes"]
                or sha(file.read_bytes()) != spec["sha256"]
            ):
                raise ValueError("current static source descriptor differs")
        elif spec["path"].endswith(".md"):
            file = core.path(spec["path"], root)
            if (
                file.stat().st_size != spec["bytes"]
                or sha(file.read_bytes()) != spec["sha256"]
            ):
                raise ValueError("reflected design differs")
        else:
            core.checked_json(spec, root)
    policy = core.checked_json(reg["force_policy"], root)
    if (
        policy != reg["force_policy_value"]
        or policy.get("schema") != "flylab.m3.e349.held-force-policy.v1"
        or policy.get("historically_sampled_per_row") is not False
    ):
        raise ValueError("exact reviewed force-policy value differs")
    freeze = core.checked_json(policy["original_JSON"]["freeze"], root)
    # Fixed reviewed call chain, not a new imported trial or a word-search proof.
    # The final Astra authority pins this finite source chain; any changed link
    # fails before construction. No whole inherited closure is traversed.
    for name, pin in policy["source_sha256"].items():
        if not name.endswith(".py") or freeze["source_sha256"].get(name) != pin:
            raise ValueError("reviewed batch call-path/frozen source differs")
        file = core.path(name, root)
        if file.stat().st_size > 256 << 10 or sha(file.read_bytes()) != pin:
            raise ValueError("actual batch source no longer matches reviewed chain")
    tree = ast.parse(core.path(CORE, root).read_text("utf-8"))
    routine = {getattr(node, "name", ""): node for node in tree.body}
    actual = {
        name: sha(ast.dump(routine[name], include_attributes=False).encode())
        for name in REUSED
    }
    if actual != reg["inherited_native_routine_AST_sha256"]:
        raise ValueError("unchanged-eight routine AST inheritance differs")
    api = core.checked_json(
        reg["evaluate_exception"]["actual_current_API_amendment"], root
    )
    identity = core.checked_json(reg["evaluate_exception"]["actual_PID_receipt"], root)
    if (
        api["source_sha256"].get(CORE) != CORE_SHA
        or api.get("CPU_tests") != 2
        or identity.get("qualified") is not True
        or identity.get("all_owned_pids_ended") is not True
    ):
        raise ValueError("current separately tested gait tuple API required")
    native = core.checked_json(reg["inherited_actual_native_evidence"], root)
    exit_record = core.checked_json(reg["inherited_actual_native_full_exit"], root)
    if (
        native.get("passed") is not True
        or native.get("both_continuations_completed") is not True
        or native.get("all_owned_bodies_closed") is not True
        or exit_record.get("all_owned_ended") is not True
        or exit_record.get("exit_code") != 0
    ):
        raise ValueError(
            "explicit historical unchanged-routine native evidence differs"
        )
    required = {
        "control": {"command", "tendon", "control_tick", "physics_time_s"},
        "native": set(core.NATIVE),
        "reference": {
            "curves",
            "normalization",
            "temperature",
            "future_anchors",
            "phase_basis",
        },
    }
    if set(reg["payloads"]) != set(required):
        raise ValueError("same sole three payload roles required")
    paths = set()
    for role, spec in reg["payloads"].items():
        core.path(spec["path"], root)
        if (
            spec["path"] in paths
            or set(spec["decode_members"]) != required[role]
            or set(spec["selected_members"]) != required[role]
            or spec.get("unselected_members_allowed_but_not_opened") is not True
        ):
            raise ValueError("same selected payload binding differs")
        paths.add(spec["path"])
    return reg


def validate_action(row):
    """Own null-request-safe validator; full source-native eight-channel ABI."""
    np, static, _ = shared().libraries()
    q = static.typed(
        np.asarray(row["joint_targets_rad"], np.float64), (42,), "<f8", "source q42"
    )
    if np.any(np.abs(q) > 10):
        raise ValueError("source q magnitude")
    static.typed(np.asarray(row["adhesion"]), (6,), "|b1", "source adhesion6 boolean")
    tendon = row["tendon_inputs"]
    if (
        not isinstance(tendon, dict)
        or set(tendon) != set(TENDONS)
        or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            for v in tendon.values()
        )
    ):
        raise ValueError("exact original eight tendon channels required")
    for key, shape in (("joint_stiffness_scale", (42,)), ("adhesion_control", (6,))):
        if key in row:
            static.typed(
                np.asarray(row[key], np.float64), shape, "<f8", "source " + key
            )
    if "neural_request" not in row or row["neural_request"] is not None:
        raise ValueError("explicit sampled-null request dialect required")


def validate_inputs(state, trace, metadata, policy):
    core = shared()
    selected = metadata["selected"]
    if (
        any(
            state.get(k) != selected[k]
            for k in ("schema", "seed", "world", "parameters", "control_tick")
        )
        or sorted(state["body"]) != selected["body_keys"]
    ):
        raise ValueError("literal checkpoint body metadata differs")

    def literal(value):
        if isinstance(value, dict):
            if "__npy__" in value:
                raise ValueError("literal body forbids array leaf references")
            for v in value.values():
                literal(v)
        elif isinstance(value, list):
            for v in value:
                literal(v)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite literal body")

    literal(state["body"])
    body = state["body"]
    if len(body["state"]) != 924 or body["modelHash"] != core.MODEL_HASH:
        raise ValueError("literal924 fixed native model required")
    origin = {"push_left": body["push_left"], "push_force": body["push_force"]}
    if (
        not isinstance(origin["push_left"], (int, float))
        or isinstance(origin["push_left"], bool)
        or not math.isfinite(origin["push_left"])
        or not isinstance(origin["push_force"], list)
        or len(origin["push_force"]) != 3
        or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            for v in origin["push_force"]
        )
        or origin != policy["recorded_origin"]
        or origin["push_left"] != 0
        or any(v != 0 for v in origin["push_force"])
    ):
        raise ValueError("explicit recorded finite no-force origin required")
    rows = trace["samples"]
    if (
        len(rows) != 301
        or [r["control_tick"] for r in rows] != list(range(301))
        or trace["body_hash"] != core.MODEL_HASH
        or trace["parameters"] != selected["parameters"]
    ):
        raise ValueError("source301 metadata differs")
    parameters = state["parameters"]
    if (
        parameters.get("active_length", False) is not False
        or parameters.get("continuous_joint_activation", False) is not False
        or parameters.get("adhesion_model", "legacy") != "legacy"
    ):
        raise ValueError("reviewed inactive optional-control provenance differs")
    for row in rows:
        validate_action(row)
        if "joint_stiffness_scale" in row or "adhesion_control" in row:
            raise ValueError("reviewed original optional-control dialect differs")
    return rows


def load_original(registration, *, root=ROOT, timely=lambda: None):
    """Same3 selected payloads; no old loader/validator/whole network import."""
    core = shared()
    specs = {s["path"]: s for s in registration["finite_JSON_dossier"]}

    def read(name):
        timely()
        return core.checked_json(specs[BASE + "/" + name], root)

    metadata = read("parent-e337-current-body-metadata.json")
    state = read("e335-smooth-secant-body/observed-start/state.json")
    trace = read("e335-smooth-secant-body/trace.json")
    manifest = read("e335-smooth-secant-body/observed-start/manifest.json")
    rows = copy.deepcopy(
        validate_inputs(state, trace, metadata, registration["force_policy_value"])
    )
    np, static, _ = core.libraries()
    if manifest["versions"] != static._versions():
        raise ValueError("source native package versions differ")
    data = {
        "body_state": state["body"],
        "body_metadata": metadata,
        "metadata": trace,
        "rows": rows,
    }
    data.update(
        {
            role: core.selected(spec, root=root, timely=timely)
            for role, spec in registration["payloads"].items()
        }
    )
    raw = data["control"]
    static.exact(
        raw["control_tick"], np.arange(301, dtype=np.int64), "source control clock"
    )
    if not np.allclose(
        raw["physics_time_s"], np.arange(301) * 0.005, atol=2e-8, rtol=0
    ):
        raise ValueError("source native boundary clock differs")
    for c, row in enumerate(rows):
        row["physics_time_s"] = float(raw["physics_time_s"][c])
        if c:
            static.exact(
                np.asarray(row["joint_targets_rad"], np.float64),
                raw["command"][c],
                "source q",
            )
            static.exact(
                np.asarray(
                    [row["tendon_inputs"][leg + "_tarsus"] for leg in core.LEGS],
                    np.float64,
                ),
                raw["tendon"][c],
                "source tendon",
            )
    static.verify_native(data["native"], count=15000)
    return data


def read_held(body):
    np, _, _ = shared().libraries()
    left = body.push_left
    if (
        isinstance(left, (bool, np.bool_))
        or not isinstance(left, (int, float, np.integer, np.floating))
        or not math.isfinite(float(left))
    ):
        raise ValueError("unreadable/nonfinite actual held duration")
    force = np.asarray(body.push_force)
    if (
        force.dtype.kind not in "fiu"
        or force.shape != (3,)
        or not np.isfinite(force).all()
    ):
        raise ValueError("unreadable/nonfinite actual held vector")
    return {"push_left": float(left), "push_force": [float(v) for v in force]}


def check_empty(value):
    if value["push_left"] != 0 or any(v != 0 for v in value["push_force"]):
        raise ValueError("actual checkpoint-held empty force changed")


class HeldForceReadError(RuntimeError):
    pass


class HeldForceDependencies(shared().NativeDependencies):
    def __init__(self, policy, output):
        super().__init__()
        self.policy = copy.deepcopy(policy)
        self.policy_sha256 = sha(json.dumps(policy, indent=2, allow_nan=False).encode())
        self.output = Path(output)
        self.actions, self.before_read_errors = [], []
        self.action_adapter_requested = self.native_action_requested = (
            self.native_action_returned
        ) = 0
        self._body_class, self._configure = self.body_class, self.configure
        self._arm_count, self._body_arm = 0, {}
        # Base __init__ assigns instance methods; subclass declarations alone
        # would leave the null-request validator and force-writing action active.
        self.body_class = self.prepare_body_class
        self.configure = self.configure_arm
        self.action = self.held_action
        self.boundary = self.held_boundary

    def ensure_policy(self):
        file = self.output / "force-policy.json"
        if not file.exists():
            shared().publish(file, self.policy)
        if sha(file.read_bytes()) != self.policy_sha256:
            raise ValueError("owned exact force-policy publication differs")

    def prepare_body_class(self, data):
        self.ensure_policy()  # Core has created fresh output; before constructor.
        return self._body_class(data)

    def configure_arm(self, body, data):
        self._arm_count += 1
        if self._arm_count > 3:
            raise ValueError("sole fixed B/I/C arm roster")
        self._body_arm[id(body)] = ("B", "I", "C")[self._arm_count - 1]
        return self._configure(body, data)

    def held_action(self, body, row, command):
        core = shared()
        self.action_adapter_requested += 1
        arm = self._body_arm[id(body)]
        try:
            before = read_held(body)
        except Exception as error:
            self.before_read_errors.append(
                {
                    "arm": arm,
                    "control_tick": row["control_tick"],
                    "type": type(error).__name__,
                    "message": str(error)[:1024],
                }
            )
            raise
        event = {
            "arm": arm,
            "control_tick": row["control_tick"],
            "before": before,
            "after": None,
            "returned": False,
            "error": None,
        }
        self.actions.append(event)
        primary = after_error = None
        try:
            check_empty(before)
            validate_action(row)
            np, static, _ = core.libraries()
            static.typed(command, (42,), "<f8", "fixed planned command")
            kwargs = {
                key: copy.deepcopy(row[key])
                for key in ("joint_stiffness_scale", "adhesion_control")
                if key in row
            }
            self.native_action_requested += 1
            body.step_joint_targets(
                command,
                np.asarray(row["adhesion"]),
                0.005,
                tendon_inputs=copy.deepcopy(row["tendon_inputs"]),
                interpolate=True,
                **kwargs,
            )
            event["returned"] = True
            self.native_action_returned += 1
        except Exception as error:  # noqa: BLE001 -- preserve native HealthStop type
            primary = error
            event["error"] = {
                "stage": "action",
                "type": type(error).__name__,
                "message": str(error)[:1024],
            }
        finally:
            try:
                event["after"] = read_held(body)
                check_empty(event["after"])
            except Exception as error:  # noqa: BLE001 -- actual unreadable after stays null
                after_error = error
                record = {
                    "stage": "after_read_or_check",
                    "type": type(error).__name__,
                    "message": str(error)[:1024],
                }
                if event["error"] is None:
                    event["error"] = record
                else:
                    event["error"]["secondary_error"] = record
        if after_error is not None:
            raise HeldForceReadError(str(after_error)) from primary
        if primary is not None:
            raise primary

    def held_boundary(self, body, source_row):
        # Read native state/contact/geometry with unchanged helper, no old action.
        row = self.static.Dependencies.boundary(body, source_row)
        actual = read_held(body)
        check_empty(actual)
        row["neural_request"] = None
        row["checkpoint_held_force"] = {
            "policy_sha256": self.policy_sha256,
            "actual_boundary_held": actual,
            "historically_sampled_per_row": False,
        }
        row["source_label"]["source_neural_metadata_is_label_only"] = True
        return row


def initial_result(stage):
    return {
        "schema": "flylab.m3.e349.held-force-result.v1",
        "status": "technical_invalid",
        "stage": stage,
        "arms": {},
        "metrics": {},
        "first_error": None,
        "errors": [],
        "analysis_completed": False,
        "external_completion_qualified": False,
        "neural_calls": 0,
        "CPG_advance_calls": 0,
        "candidate_adopted": False,
        "F1_F8_complete": False,
        "biology_proven": False,
        "H348": {"verdict": "technical_invalid", "supported": None, "gates": None},
    }


def publish_final(result, output, policy, deps, *, final_timely):
    core = shared()
    result["force_policy"] = result["held_force_receipt"] = None
    try:
        if not (output / "force-policy.json").exists():
            core.publish(output / "force-policy.json", policy)
        policy_spec = {
            "path": "force-policy.json",
            "bytes": (output / "force-policy.json").stat().st_size,
            "sha256": sha((output / "force-policy.json").read_bytes()),
        }
        if core.read_json(output / "force-policy.json") != policy:
            raise ValueError("exact reviewed force policy differs")
        result["force_policy"] = policy_spec
        receipt = {
            "schema": "flylab.m3.e349.held-force-receipt.v1",
            "policy_sha256": policy_spec["sha256"],
            "actions": deps.actions if deps else [],
        }
        core.publish(output / "held-force-receipt.json", receipt)
        result["held_force_receipt"] = {
            "path": "held-force-receipt.json",
            "bytes": (output / "held-force-receipt.json").stat().st_size,
            "sha256": sha((output / "held-force-receipt.json").read_bytes()),
        }
    except Exception as error:  # noqa: BLE001 -- narrow owned force publication
        core.record_error(result, "force_publication", error)
        result["status"] = "technical_invalid"
        result["analysis_completed"] = False
        result["H348"] = core.hypothesis(result["metrics"], False)
    result["source_sha256"] = own_sources()
    result["current_core_sha256"] = CORE_SHA
    result["source_neural_metadata_is_label_only"] = True
    result["timing_elapsed_s"] = time.perf_counter() - ENTRY
    result["timing_elapsed_scope"] = (
        "own ENTRY through final-result publication start; inclusive close deadline checked separately"
    )
    result["force_call_counters"] = (
        {
            "adapter_requested": deps.action_adapter_requested,
            "native_action_requested": deps.native_action_requested,
            "native_action_returned": deps.native_action_returned,
        }
        if deps
        else None
    )
    result["force_call_counters_status"] = "instrumented" if deps else "not_started"
    result["force_before_read_errors"] = deps.before_read_errors if deps else []
    try:
        core.update_receipt(output / "result.json", result)
        if core.usage(output) > core.LIMITS["stage_bytes"]:
            raise ValueError("inclusive stage output limit exceeded")
        final_timely()  # Includes both new artifacts and result stream close.
    except Exception as error:  # noqa: BLE001 -- unavailable sink retains external failure
        core.record_error(result, "final_publication", error)
        result["status"] = "technical_invalid"
        result["analysis_completed"] = False
        result["H348"] = core.hypothesis(result["metrics"], False)
        try:
            core.update_receipt(output / "result.json", result)
        except Exception as secondary:  # noqa: BLE001 -- bounded final attempt only
            core.record_error(result, "final_receipt_publication", secondary)
    return result


def run_loaded(
    data,
    output,
    policy,
    *,
    timely=lambda: None,
    final_timely=lambda: None,
    dependencies=None,
    engine_run=None,
):
    """Generated spies may inject dependencies/engine; original CLI cannot."""
    core = shared()
    output = Path(output)
    if output.exists():
        raise ValueError("fresh owned output required")
    deps = dependencies or HeldForceDependencies(policy, output)
    try:
        result = (engine_run or core.run)(
            data, output, dependencies=deps, timely=timely, final_timely=final_timely
        )
        result["inherited_body_result_schema"] = result["schema"]
        result["schema"] = "flylab.m3.e349.held-force-result.v1"
    except Exception as error:
        if isinstance(error, FileExistsError):
            raise  # Never publish into an output directory owned by another run.
        result = initial_result("body_orchestration")
        result["body_counter_status"] = "not_available_from_failed_orchestration"
        core.record_error(result, "body_orchestration", error)
        output.mkdir(parents=True, exist_ok=True)
    return publish_final(result, output, policy, deps, final_timely=final_timely)


def make_freeze(qualification, *, root=ROOT):
    core = shared()
    reg = metadata_preflight(root)
    own = own_sources(root)
    qual = core.read_json(core.path(qualification, root), cap=256 << 10)
    checks = (
        "CPU",
        "ruff",
        "instance_adapters",
        "no_force_writes",
        "null_request_loader",
        "policy_provenance",
        "before_after_failure",
        "publication_failure",
        "source_AST_reuse",
    )
    if (
        qual.get("schema") != "flylab.m3.e349.root-qualification.v1"
        or qual.get("source_sha256") != own
        or qual.get("passed") is not True
        or qual.get("original_calls") != 0
        or any(qual.get("checks", {}).get(k) is not True for k in checks)
    ):
        raise ValueError("current E349 ROOT qualification required")
    execution = qual["execution"]
    contract, supervisor, exit_record = (
        core.checked_json(execution[k], root)
        for k in ("contract", "supervision", "full_exit")
    )
    if (
        any(contract["source_sha256"].get(k) != v for k, v in own.items())
        or contract.get("original_calls") != 0
        or supervisor.get("contract_sha256") != execution["contract"]["sha256"]
        or supervisor.get("success") is not True
        or supervisor.get("original_calls") != 0
        or exit_record.get("all_owned_ended") is not True
        or exit_record.get("exit_code") != 0
        or exit_record.get("first_error") is not None
        or not exit_record.get("owned_identities")
        or any(
            x.get("close") != "ended"
            or type(x.get("pid")) is not int
            or isinstance(x.get("create_time"), bool)
            or not isinstance(x.get("create_time"), (int, float))
            or not math.isfinite(x["create_time"])
            for x in exit_record["owned_identities"]
        )
    ):
        raise ValueError("current qualified source/actual full-exit identities differ")
    jobs, attempts = contract["commands"], supervisor["attempts"]
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
        raise ValueError("current generated sole argv/exits differ")
    cost = exit_record.get("charged_s")
    if (
        isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(cost)
        or not 0 < cost <= 20
    ):
        raise ValueError("bounded current generated execution charge required")
    return {
        "schema": "flylab.m3.e349.held-force-freeze.v1",
        "source_sha256": own,
        "current_core_sha256": CORE_SHA,
        "registration_sha256": REGISTRATION_SHA,
        "reflection_sha256": REFLECTION_SHA,
        "inherited_AST_sha256": reg["inherited_native_routine_AST_sha256"],
        "inherited_native_evidence": reg["inherited_actual_native_evidence"],
        "evaluate_API_exception": reg["evaluate_exception"],
        "force_policy": reg["force_policy"],
        "qualification": {
            "path": qualification,
            "bytes": core.path(qualification, root).stat().st_size,
            "sha256": sha(core.path(qualification, root).read_bytes()),
        },
        "limits": core.LIMITS,
        "original_authorized": False,
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
    core = shared()
    reg = metadata_preflight(root)
    freeze = core.read_json(core.path(freeze_name, root))
    if freeze != make_freeze(freeze["qualification"]["path"], root=root):
        raise ValueError("current E349 freeze differs")
    review = core.read_json(core.path(review_name, root), cap=256 << 10)
    freeze_sha = sha(core.path(freeze_name, root).read_bytes())
    if (
        review.get("schema") != "flylab.m3.e349.root-original-review.v1"
        or review.get("approved") is not True
        or review.get("source_sha256") != own_sources(root)
        or review.get("freeze_sha256") != freeze_sha
        or review.get("registration_sha256") != REGISTRATION_SHA
        or review.get("reflection_sha256") != REFLECTION_SHA
        or output != OUTPUT
        or review.get("output") != OUTPUT
    ):
        raise ValueError("current ROOT private E349 approval required")
    reservation = core.checked_json(review["reservation"], root)
    expected = {
        "schema": "flylab.m3.e349.root-reservation.v1",
        "cumulative_before_s": 9781.550494199497,
        "cumulative_limit_s": 9900,
        "reserved_s": 103,
        "body_wall_s": 95,
        "audit_wall_s": 8,
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
        raise ValueError("exact current once/cumulative reservation linkage required")
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
        raise ValueError("fresh output/CPU1/free disk required before loading")
    return reg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--qualification")
    parser.add_argument("--review")
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    core = shared()

    def timely(final=False):
        if (
            time.perf_counter() - ENTRY
            > core.LIMITS["internal_s" if final else "science_s"]
        ):
            raise TimeoutError(
                "E349 own inclusive import/read/science/publication deadline"
            )
        import psutil

        if psutil.Process().memory_info().rss > core.LIMITS["host_bytes"]:
            raise MemoryError("E349 host2GiB")

    if not args.run:
        if not args.qualification:
            parser.error("passive freeze requires --qualification")
        core.publish(core.path(args.freeze), make_freeze(args.qualification))
        timely(True)
        return 0
    reg = original_gate(args.freeze, args.review, output=args.output)
    output = core.path(args.output)
    try:
        timely()
        data = load_original(reg, timely=timely)
    except Exception as error:  # noqa: BLE001 -- no constructor reached during input load
        output.mkdir(parents=True, exist_ok=False)
        result = initial_result("input_load")
        result.update(
            body_constructors=0,
            scientific_physics_steps=0,
            array_read_counts=None,
            array_read_counts_status="not_instrumented",
        )
        core.record_error(result, "input_load", error)
        result = publish_final(
            result,
            output,
            reg["force_policy_value"],
            None,
            final_timely=lambda: timely(True),
        )
    else:
        result = run_loaded(
            data,
            output,
            reg["force_policy_value"],
            timely=timely,
            final_timely=lambda: timely(True),
        )
    if result["status"] != "completed":
        print(
            json.dumps(
                {"first_error": result["first_error"], "errors": result["errors"]},
                allow_nan=False,
            ),
            file=sys.stderr,
        )
    timely(True)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
