"""Small generated spies/loader sentinels; no constructor or physical stepping."""

import copy
import json

import numpy as np
import pytest

from tools import verify_c_banc_m3_held_force_body as p


def row():
    return {
        "control_tick": 1,
        "joint_targets_rad": [0.0] * 42,
        "adhesion": [False] * 6,
        "tendon_inputs": {name: 0.0 for name in p.TENDONS},
        "neural_request": None,
    }


def policy():
    return {
        "schema": "flylab.m3.e349.held-force-policy.v1",
        "historically_sampled_per_row": False,
        "recorded_origin": {"push_left": 0.0, "push_force": [0.0, 0.0, 0.0]},
    }


class SpyBody:
    def __init__(self, *, error=None, unreadable_after=False, left=0.0):
        self._left, self._force = left, [0.0, 0.0, 0.0]
        self.calls, self.arguments = 0, None
        self.error, self.unreadable_after = error, unreadable_after

    @property
    def push_left(self):
        return self._left

    @property
    def push_force(self):
        if self.calls and self.unreadable_after:
            raise RuntimeError("actual after unreadable")
        return self._force

    def __setattr__(self, name, value):
        if name in ("push_left", "push_force"):
            raise AssertionError("forbidden force write/reset")
        object.__setattr__(self, name, value)

    def step_joint_targets(self, *args, **kwargs):
        self.calls += 1
        self.arguments = (args, kwargs)
        if self.error:
            raise self.error


def deps(tmp_path, body):
    value = p.HeldForceDependencies(policy(), tmp_path)
    value._configure = lambda *_: {"test": "no construction"}
    value.configure(body, {})
    return value


def test_actual_instance_action_rebound_no_force_write_full_ABI(tmp_path):
    body = SpyBody()
    dep = deps(tmp_path, body)
    source = row()
    source["joint_stiffness_scale"] = [1.0] * 42
    source["adhesion_control"] = [0.0] * 6
    original = copy.deepcopy(source)
    command = np.full(42, 0.1, dtype=np.float64)
    assert dep.action.__func__ is p.HeldForceDependencies.held_action
    assert dep.boundary.__func__ is p.HeldForceDependencies.held_boundary
    dep.action(body, source, command)
    assert source == original and source["neural_request"] is None
    args, kw = body.arguments
    assert args[0] is command and args[1].dtype == np.dtype(bool) and args[2] == 0.005
    assert kw == {
        "tendon_inputs": source["tendon_inputs"],
        "interpolate": True,
        "joint_stiffness_scale": source["joint_stiffness_scale"],
        "adhesion_control": source["adhesion_control"],
    }
    assert dep.actions == [
        {
            "arm": "B",
            "control_tick": 1,
            "before": policy()["recorded_origin"],
            "after": policy()["recorded_origin"],
            "returned": True,
            "error": None,
        }
    ]
    assert body.calls == dep.native_action_requested == dep.native_action_returned == 1


@pytest.mark.parametrize("unreadable", [False, True])
def test_failed_call_actual_after_or_null_error(tmp_path, unreadable):
    primary = ValueError("native spy failed")
    body = SpyBody(error=primary, unreadable_after=unreadable)
    dep = deps(tmp_path, body)
    with pytest.raises(p.HeldForceReadError if unreadable else ValueError):
        dep.action(body, row(), np.zeros(42))
    event = dep.actions[0]
    assert (
        event["returned"] is False and event["error"]["message"] == "native spy failed"
    )
    if unreadable:
        assert (
            event["after"] is None
            and event["error"]["secondary_error"]["message"]
            == "actual after unreadable"
        )
    else:
        assert event["after"] == policy()["recorded_origin"]
    assert dep.native_action_requested == 1 and dep.native_action_returned == 0


def test_health_exception_is_preserved_for_shared_prefix_rules(tmp_path):
    body = SpyBody(error=p.shared().HealthStop("observed generated health"))
    dep = deps(tmp_path, body)
    with pytest.raises(p.shared().HealthStop):
        dep.action(body, row(), np.zeros(42))
    assert dep.actions[0]["after"] == policy()["recorded_origin"]


@pytest.mark.parametrize("bad", ["force", "adhesion", "tendon", "request"])
def test_invalid_before_or_source_stops_before_spy_call(tmp_path, bad):
    body = SpyBody(left=0.01 if bad == "force" else 0.0)
    dep = deps(tmp_path, body)
    source = row()
    if bad == "adhesion":
        source["adhesion"] = [0] * 6
    if bad == "tendon":
        source["tendon_inputs"]["unknown"] = source["tendon_inputs"].pop("abdomen_yaw")
    if bad == "request":
        source["neural_request"] = {"unreviewed": True}
    with pytest.raises(p.HeldForceReadError if bad == "force" else ValueError):
        dep.action(body, source, np.zeros(42))
    assert body.calls == dep.native_action_requested == dep.native_action_returned == 0
    assert dep.actions[0]["returned"] is False
    assert dep.actions[0]["before"]["push_left"] == body.push_left


def test_boundary_labels_only_not_synthetic_neural_request(tmp_path, monkeypatch):
    body = SpyBody()
    dep = deps(tmp_path, body)

    def boundary(actual, source):
        assert actual is body and source["neural_request"] is None
        return {
            "source_label": {"neural_execution": 0},
            "neural_request": {"external_force_at_start": None},
        }

    monkeypatch.setattr(dep.static.Dependencies, "boundary", boundary)
    result = dep.boundary(body, row())
    assert result["neural_request"] is None
    assert (
        result["checkpoint_held_force"]["actual_boundary_held"]
        == policy()["recorded_origin"]
    )
    assert result["checkpoint_held_force"]["historically_sampled_per_row"] is False
    assert result["source_label"]["source_neural_metadata_is_label_only"] is True


def loader_fixture(monkeypatch):
    core = p.shared()
    selected = {
        "schema": "generated",
        "seed": 42,
        "world": {},
        "parameters": {"joint_activation_tau_s": 0.09, "fast_coxa_activation": True},
        "control_tick": 0,
        "body_keys": ["modelHash", "push_force", "push_left", "state"],
    }
    state = {
        k: copy.deepcopy(selected[k])
        for k in ("schema", "seed", "world", "parameters", "control_tick")
    }
    state["body"] = {
        "state": [0.0] * 924,
        "modelHash": core.MODEL_HASH,
        "push_left": 0.0,
        "push_force": [0.0, 0.0, 0.0],
    }
    rows = [dict(row(), control_tick=c) for c in range(301)]
    trace = {
        "samples": rows,
        "body_hash": core.MODEL_HASH,
        "parameters": copy.deepcopy(selected["parameters"]),
    }
    np_, static, _ = core.libraries()
    assert np_ is np

    def old_validator(*a):
        raise AssertionError("old null-request validator was called")

    monkeypatch.setattr(static, "validate_action", old_validator)
    values = {
        "parent-e337-current-body-metadata.json": {"selected": selected},
        "e335-smooth-secant-body/observed-start/state.json": state,
        "e335-smooth-secant-body/trace.json": trace,
        "e335-smooth-secant-body/observed-start/manifest.json": {
            "versions": static._versions()
        },
    }
    reg = {
        "finite_JSON_dossier": [{"path": p.BASE + "/" + name} for name in values],
        "force_policy_value": policy(),
        "payloads": {
            role: {"path": role} for role in ("control", "native", "reference")
        },
    }
    monkeypatch.setattr(
        core,
        "checked_json",
        lambda spec, root: values[spec["path"].removeprefix(p.BASE + "/")],
    )
    return reg, state, trace


def test_loader_own_validator_then_exact_first_payload_sentinel(monkeypatch):
    reg, state, trace = loader_fixture(monkeypatch)
    original = copy.deepcopy(trace)

    class FirstSelected(Exception):
        pass

    opened = []

    def selected(spec, **kw):
        opened.append(spec["path"])
        raise FirstSelected("EXACT GENERATED FIRST SELECTED")

    monkeypatch.setattr(p.shared(), "selected", selected)
    with pytest.raises(FirstSelected, match="^EXACT GENERATED FIRST SELECTED$"):
        p.load_original(reg)
    assert opened == ["control"] and trace == original
    assert all(r["neural_request"] is None for r in trace["samples"])
    assert state["parameters"]["joint_activation_tau_s"] == 0.09


@pytest.mark.parametrize(
    "bad",
    ["missing_origin", "nonzero", "nonfinite", "mixed", "missing_request", "optional"],
)
def test_loader_rejects_policy_before_any_payload(monkeypatch, bad):
    reg, state, trace = loader_fixture(monkeypatch)
    if bad == "missing_origin":
        state["body"].pop("push_force")
    elif bad == "nonzero":
        state["body"]["push_force"][0] = 1.0
    elif bad == "nonfinite":
        state["body"]["push_left"] = float("nan")
    elif bad == "mixed":
        trace["samples"][55]["neural_request"] = {}
    elif bad == "missing_request":
        trace["samples"][56].pop("neural_request")
    else:
        state["parameters"]["active_length"] = True
        trace["parameters"]["active_length"] = True

    def forbidden(*a, **k):
        raise AssertionError("payload touched before valid metadata")

    monkeypatch.setattr(p.shared(), "selected", forbidden)
    with pytest.raises((ValueError, KeyError)):
        p.load_original(reg)


@pytest.mark.parametrize("fault", [None, "planning", "publication", "deadline"])
def test_small_output_policy_receipt_hash_binding_and_failures(
    tmp_path, monkeypatch, fault
):
    output = tmp_path / "tiny"
    body = SpyBody()
    dep = deps(output, body)

    def engine(data, folder, **kw):
        folder.mkdir()
        if fault == "planning":
            raise ValueError("generated plan failed before body")
        dep.action(body, row(), np.zeros(42))
        result = p.initial_result("generated stub")
        result["status"] = "completed"
        p.shared().publish(folder / "result.json", result)
        return result

    original = p.shared().publish

    def writer(file, value):
        if fault == "publication" and file.name == "held-force-receipt.json":
            raise OSError("generated receipt sink failed")
        original(file, value)

    monkeypatch.setattr(p.shared(), "publish", writer)

    def deadline():
        if fault == "deadline":
            raise TimeoutError("after-close generated deadline")

    result = p.run_loaded(
        {}, output, policy(), dependencies=dep, engine_run=engine, final_timely=deadline
    )
    assert json.loads((output / "result.json").read_text("utf-8")) == result
    assert json.loads((output / "force-policy.json").read_text("utf-8")) == policy()
    if fault:
        assert (
            result["status"] == "technical_invalid"
            and result["first_error"] is not None
        )
        assert result["H348"]["verdict"] == "technical_invalid"
    else:
        assert result["force_policy"]["sha256"] == p.sha(
            (output / "force-policy.json").read_bytes()
        )
        assert result["held_force_receipt"]["sha256"] == p.sha(
            (output / "held-force-receipt.json").read_bytes()
        )
        held = json.loads((output / "held-force-receipt.json").read_text("utf-8"))
        assert (
            held["policy_sha256"] == result["force_policy"]["sha256"]
            and held["actions"] == dep.actions
        )
    assert result["force_call_counters"]["native_action_requested"] == (
        0 if fault == "planning" else 1
    )


def test_actual_registered_metadata_AST_inheritance_without_payloads(monkeypatch):
    core = p.shared()

    def refuse(*a, **k):
        raise AssertionError("original payload or body was touched")

    monkeypatch.setattr(core, "selected", refuse)
    monkeypatch.setattr(core, "load_original", refuse)
    reg = p.metadata_preflight()
    assert set(reg["inherited_native_routine_AST_sha256"]) == set(p.REUSED)
    assert reg["force_policy_value"]["historically_sampled_per_row"] is False


def test_registered_exact_policy_written_before_stub_class(tmp_path):
    reg = p.shared().read_json(p.ROOT / p.REGISTRATION)
    policy_value = reg["force_policy_value"]
    dep = p.HeldForceDependencies(policy_value, tmp_path)
    marker = object()

    def cls(data):
        assert (
            json.loads((tmp_path / "force-policy.json").read_text("utf-8"))
            == policy_value
        )
        return marker

    dep._body_class = cls
    assert dep.body_class({}) is marker
    assert dep.actions == [] and dep.native_action_requested == 0


def test_unknown_reviewed_source_chain_rejected_metadata_only(monkeypatch):
    core = p.shared()
    original = core.checked_json

    def reader(spec, root):
        value = original(spec, root)
        if spec["path"].endswith("e335-smooth-secant-body-freeze.json"):
            first = "tools/verify_c_banc_m3_smooth_secant_body.py"
            value["source_sha256"][first] = "0" * 64
        return value

    monkeypatch.setattr(core, "checked_json", reader)
    with pytest.raises(ValueError, match="call-path"):
        p.metadata_preflight()


def test_private_gate_rejects_before_any_original_array(monkeypatch):
    core = p.shared()
    monkeypatch.setattr(p, "metadata_preflight", lambda *a: {})
    monkeypatch.setattr(
        core, "read_json", lambda *a, **k: {"qualification": {"path": "generated"}}
    )
    monkeypatch.setattr(p, "make_freeze", lambda *a, **k: {})
    with pytest.raises(ValueError, match="freeze"):
        p.original_gate(
            p.BASE + "/e349-sol-generated-freeze.json",
            p.BASE + "/e349-sol-generated-review.json",
        )
    assert p.original_argv("F", "R") == [
        ".venv/Scripts/python.exe",
        "-X",
        "utf8",
        "-B",
        p.OWNED[0],
        "--run",
        "--freeze",
        "F",
        "--review",
        "R",
        "--output",
        p.OUTPUT,
    ]
