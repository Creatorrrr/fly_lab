"""Generated E337 mathematics, body ownership, durable replay, terminal failures."""

import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools import probe_c_banc_m3_current_shape_counterfactual as probe


def test_astra_provenance_aliases_are_existing_artifact_paths_not_basenames():
    invocation = {
        "input_snapshot_sha256": "a" * 64,
        "files_sha256": {"snapshot": "a" * 64, "prompt": "b" * 64,
                         "review": probe.FIXED["astra"], "events": "c" * 64,
                         "stderr": "d" * 64},
    }
    pins = probe.astra_file_pins(invocation)
    assert pins[probe.BASE + "/parent-e337-pre-astra-design.md"] == "a" * 64
    assert pins[probe.PATHS["astra"]] == probe.FIXED["astra"]
    assert all(name.startswith(probe.BASE + "/parent-e337-") for name in pins)
    assert not any(probe.BASE + "/" + alias in pins for alias in invocation["files_sha256"])
    invocation["files_sha256"]["unknown"] = "e" * 64
    with pytest.raises(ValueError, match="aliases"):
        probe.astra_file_pins(invocation)


def generated_data():
    phase = (np.arange(201)[:, None] * -0.013 + np.arange(6)[None, :] * 0.091) % 1
    angles = 2 * np.pi * phase
    z = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    curves = np.empty((6, 128, 7), np.float64)
    for leg in range(6):
        for i in range(128):
            curves[leg, i] = 0.1 * np.sin(
                2 * np.pi * (i / 128 + leg / 13) + np.arange(7) / 9
            )
    command = np.zeros((301, 42), np.float64)
    command[:101, 1] = -0.0
    rows = []
    for c in range(301):
        rows.append(
            {
                "control_tick": c,
                "time_s": c * 0.005,
                "qpos": [0.0],
                "qvel": [0.0],
                "ctrl": [0.0],
                "joint_targets_rad": command[c].tolist(),
                "adhesion": [True] * 6,
                "tendon_inputs": {"tendon" + str(i): float(i) / 10 for i in range(8)},
                "joint_stiffness_scale": [1.0] * 42,
                "adhesion_control": [0.5] * 6,
                "neural_request": {
                    "external_force_at_start": {
                        "push_left_s": 0.01,
                        "push_force": [1.0, 2.0, 3.0],
                    }
                },
                "fault": None,
            }
        )
    native = {
        "time_s": np.arange(1, 15001) * 0.0001,
        "normal_load_bw": np.full((15000, 6), 0.1),
        "slip_mm_s": np.full((15000, 6), 2.0),
        "slip_valid": np.ones((15000, 6), bool),
        "upright": np.ones(15000),
        "nonfoot_contact": np.zeros(15000, bool),
    }
    return {
        "command": command,
        "rows": rows,
        "native": native,
        "curves": curves,
        "live": {
            "q": command[100:].copy(),
            "z": z,
            "R": np.linalg.norm(z, axis=-1),
            "control_tick": np.arange(100, 301, dtype=np.int64),
        },
        "body_state": {},
        "metadata": {},
        "startup": {},
    }


@pytest.fixture
def data():
    return generated_data()


def scalar_reference(curves, phase):
    values = np.empty((len(phase), 6, 7))
    for row in range(len(phase)):
        for leg in range(6):
            x = phase[row, leg] % 1 * 128
            i = int(np.floor(x))
            values[row, leg] = curves[leg, i] * (1 - (x - i)) + curves[
                leg, (i + 1) % 128
            ] * (x - i)
    return values


def test_wrapped_interpolation_direct_global_offsets(data):
    phase = np.array(
        [
            [0, 1 - 1e-14, 0.125, 0.5, 0.75, 1e-14],
            [1, -1e-14, 1.125, -0.5, -1.25, 2.00001],
        ],
        np.float64,
    )
    actual = probe.cyclic_interpolate(data["curves"], phase)
    np.testing.assert_allclose(
        actual, scalar_reference(data["curves"], phase), rtol=0, atol=3e-17
    )


def test_plan_row_alignment_signed_zero_blend_reverse(data):
    plan = probe.shape_plan(data["command"], data["live"], data["curves"])
    assert plan["command"][:101].tobytes() == data["command"][:101].tobytes()
    assert np.signbit(plan["command"][:101, 1]).all()
    assert plan["phase"].shape == (201, 6)
    assert plan["blend"].shape == (201,)
    assert plan["reference"].shape == (201, 42)
    assert plan["reliable_edges"].shape == (200, 6)
    assert plan["reliable_edges"].all()
    assert (plan["signed_phase_delta"] < 0).all()
    np.testing.assert_allclose(
        plan["reference"],
        scalar_reference(data["curves"], plan["phase"]).reshape(201, 42),
        atol=3e-17,
    )
    assert plan["command"][120:].tobytes() == plan["reference"][20:].tobytes()
    weight = 3 * 0.05**2 - 2 * 0.05**3
    np.testing.assert_allclose(
        plan["command"][101], weight * plan["reference"][1], rtol=0, atol=1e-17
    )
    assert plan["blend"][0] == 0 and plan["blend"][20] == 1


@pytest.mark.parametrize("boundary", [0, 1, 200])
def test_all_boundaries_reliable_no_fallback(data, boundary):
    data["live"]["R"][boundary, 5] = 0.499
    with pytest.raises(ValueError, match="all201"):
        probe.shape_plan(data["command"], data["live"], data["curves"])


def test_exact_halfturn_at_first_modified_edge_rejected(data):
    data["live"]["z"][:2, 0] = [[1.0, 0.0], [-1.0, 0.0]]
    data["live"]["R"][:2, 0] = 1
    with pytest.raises(ValueError, match="half-turn"):
        probe.shape_plan(data["command"], data["live"], data["curves"])


@pytest.mark.parametrize("field", ["q", "control_tick", "z", "R"])
def test_phase_source_scope_and_numeric_fault(data, field):
    if field == "q":
        data["live"][field][0, 0] = 0.001
    elif field == "control_tick":
        data["live"][field][0] = 99
    else:
        data["live"][field].flat[-1] = np.nan
    with pytest.raises(ValueError):
        probe.shape_plan(data["command"], data["live"], data["curves"])


def test_exact_finite_bits_and_missing_nan_payloads():
    with pytest.raises(ValueError, match="bytes"):
        probe.exact(np.array([-0.0]), np.array([0.0]))
    a = np.array([np.nan, 1.0])
    b = a.copy()
    b.view(np.uint64)[0] = 0x7FF8000000000001
    probe.exact(a, b, masked_nan=True)
    b[0] = 0
    with pytest.raises(ValueError, match="NaN mask"):
        probe.exact(a, b, masked_nan=True)


def test_dtype_metadata_str_and_dtype_str_selected_header(tmp_path):
    path = tmp_path / "fixture.npz"
    probe.save_npz(
        path, {"q": np.ones((2, 42)), "motor_rate32": np.zeros((2, 391), np.float32)}
    )
    reads = []
    values = probe.saved.load_selected(
        path, {"q": {"shape": [2, 42], "dtype": "float64"}}, reads=reads
    )
    assert values["q"].dtype.str == "<f8"
    assert [r["key"] for r in reads] == ["q"]
    with pytest.raises(ValueError, match="header"):
        probe.saved.load_selected(path, {"q": {"shape": [2, 42], "dtype": "int64"}})


def test_native_weighted_slip_actual_time_mask_and_unsupported(data):
    native = data["native"]
    native["time_s"][4999] = 0.5 - 1e-15
    native["normal_load_bw"][:] = 0
    native["slip_mm_s"][:] = np.nan
    native["slip_valid"][:] = False
    native["normal_load_bw"][5000, 0] = 0.02
    native["normal_load_bw"][5001, 0] = 0.06
    native["slip_mm_s"][5000:5002, 0] = [1.0, 3.0]
    native["slip_valid"][5000:5002, 0] = True
    probe.verify_native(native)
    report = probe.saved.native_statistics(native)
    assert report["weighted_slip_mm_s"] == pytest.approx(2.5)
    assert report["actual_mask_count"] == 9999
    assert report["nominal_mask_difference_ticks"] == [5000]
    native["normal_load_bw"][:] = 0
    native["slip_mm_s"][:] = np.nan
    native["slip_valid"][:] = False
    assert probe.saved.native_statistics(native)["weighted_slip_mm_s"] is None


def test_h337_relative_conjunction_not_old_absolute_gate():
    baseline = {"forward_mm": 0.1, "slip_mm_s": 8.0}
    candidate = {
        "forward_mm": 0.2,
        "slip_mm_s": 7.0,
        "longest_continuous_forward_bout_s": 0.3,
        "complete_cycles": {leg: 1 for leg in probe.saved.LEGS},
    }
    assert probe.classify(candidate, baseline)["supported"] is True
    for key, value in (
        ("forward_mm", 0.09),
        ("slip_mm_s", 9.0),
        ("longest_continuous_forward_bout_s", 0.299),
        ("slip_mm_s", None),
    ):
        bad = dict(candidate, **{key: value})
        assert probe.classify(bad, baseline)["supported"] is False
    candidate["complete_cycles"]["rh"] = 0
    assert probe.classify(candidate, baseline)["supported"] is False
    assert probe.classify({}, {}, engineering_valid=False)["supported"] is None


def test_descriptive_three_cycle_and_pair_overlap():
    gait = {
        "events": {
            leg: {"complete_cycles": 1, "recurrence_evaluable": False, "period_cv": 0.1}
            for leg in probe.saved.LEGS
        },
        "pair_phase": {
            a + "-" + b: {"samples": 2, "mean": 0.1}
            for a, b in (("lf", "rf"), ("lm", "rm"), ("lh", "rh"))
        },
    }
    before = copy.deepcopy(gait)
    summary = probe.descriptive(gait)
    assert all(v["value"] is None for v in summary["period_cv"].values())
    assert all(v["value"] is None for v in summary["lr_phase"].values())
    assert gait == before
    for event in gait["events"].values():
        event.update(complete_cycles=3, recurrence_evaluable=True)
    gait["pair_phase"]["lh-rh"]["samples"] = 0
    summary = probe.descriptive(gait)
    assert summary["period_cv"]["lf"]["value"] == 0.1
    assert summary["lr_phase"]["lf-rf"]["value"]["samples"] == 2
    assert summary["lr_phase"]["lh-rh"]["reason"] == "no_finite_pair_overlap"


class FakeBody:
    def __init__(self, owner):
        self.owner = owner
        owner.bodies.append(self)
        self.time = 0.0
        self.count = 0
        self.closed = False
        self.callback = None
        self.q = np.zeros(42)
        self.q[1] = -0.0
        self.push_force = np.zeros(3)
        if owner.constructor_failure:
            raise RuntimeError("partial constructor")

    def restore(self, state):
        self.restored = state

    def physics_time(self):
        if self.owner.clock_failure and self.count:
            raise RuntimeError("clock unavailable")
        return self.time

    def step_joint_targets(self, q, adhesion, dt, **kwargs):
        assert dt == 0.005 and kwargs["interpolate"] is True
        assert self.push_left == 0.01 and np.array_equal(self.push_force, [1, 2, 3])
        assert kwargs["tendon_inputs"] == self.owner.data["rows"][1]["tendon_inputs"]
        assert kwargs["joint_stiffness_scale"] == [1.0] * 42
        assert kwargs["adhesion_control"] == [0.5] * 6
        assert adhesion.dtype == np.bool_
        self.q = q.copy()
        c = self.count // 50 + 1
        for i in range(50):
            self.count += 1
            self.time = self.count * 0.0001
            sample = {
                key: value[self.count - 1].tolist()
                for key, value in self.owner.data["native"].items()
            }
            health_arm = "baseline" if len(self.owner.bodies) == 1 else "candidate"
            if (
                self.owner.health_control == c
                and i == 7
                and self.owner.health_arm == health_arm
            ):
                sample["upright"] = 0.8
            self.callback(sample)
            if self.owner.fail_control == c and i == 2:
                raise RuntimeError("partial physics")

    def close(self):
        self.closed = True
        if self.owner.close_failure:
            raise OSError("close failure")


class GeneratedDependencies:
    def __init__(
        self,
        data,
        *,
        fail_control=None,
        health_control=None,
        health_arm="baseline",
        constructor_failure=False,
        close_failure=False,
        clock_failure=False,
        mismatch=False,
        hook_failure=False,
    ):
        self.data = data
        self.fail_control = fail_control
        self.health_control = health_control
        self.health_arm = health_arm
        self.constructor_failure = constructor_failure
        self.close_failure = close_failure
        self.clock_failure = clock_failure
        self.mismatch = mismatch
        self.hook_failure = hook_failure
        self.bodies = []
        self.loads = 0

    def load(self, freeze, **kwargs):
        self.loads += 1
        return self.data

    def body_class(self, data):
        return FakeBody

    def body_arguments(self, data):
        return (self,), {}

    def configure(self, body, data):
        return {"generated": True, "original_constructor": False}

    def install(self, body, callback):
        body.callback = callback

        def detach():
            body.callback = None
            if self.hook_failure:
                raise OSError("hook restore failure")

        return detach

    def boundary(self, body, source):
        row = copy.deepcopy(source)
        row["joint_targets_rad"] = body.q.tolist()
        if self.mismatch and source["control_tick"] == 300:
            row["qpos"] = [0.1]
        return row

    def evaluate(self, folder, data):
        return {"knee": np.zeros((301, 6))}, {
            "official": {
                "forward_mm": 0.2,
                "slip_mm_s": 2.0,
                "longest_continuous_forward_bout_s": 0.3,
                "complete_cycles": {leg: 1 for leg in probe.saved.LEGS},
            }
        }


def generated_run(tmp_path, data, dependencies=None, **kwargs):
    dependencies = dependencies or GeneratedDependencies(data)
    # The immutable external supervisor measures the real process tree. Keep
    # resource-failure injections explicit without repeating Windows process
    # enumeration at every synthetic control in unrelated mathematical tests.
    kwargs.setdefault("meter", lambda: 0)
    result = probe.run(
        output=tmp_path / "output",
        test_dimensions=probe.TestDimensions(),
        dependencies=dependencies,
        **kwargs,
    )
    return result, dependencies


def test_full_sequential_replay_durable_chunks_and_no_neural_calls(tmp_path, data):
    result, deps = generated_run(tmp_path, data)
    assert result["effective_validity"] and result["science"]["supported"]
    assert len(deps.bodies) == 2 and all(
        b.closed and b.callback is None for b in deps.bodies
    )
    assert (
        result["original_calls"] == result["neural_calls"] == result["CUDA_calls"] == 0
    )
    ledger = probe.read(tmp_path / "output/advance-ledger.json")
    assert ledger["baseline_exact"] and ledger["prefix_exact"]
    for arm in ("baseline", "candidate"):
        folder = tmp_path / "output" / arm
        chunks = probe.read(folder / "native-chunks.json")["completed"]
        assert len(chunks) == 12 and all(v["sample_count"] == 1250 for v in chunks)
        with np.load(folder / "native-contact.npz") as archive:
            actual = archive["time_s"]
        combined = np.concatenate(
            [np.load(folder / v["path"])["time_s"] for v in chunks]
        )
        assert combined.tobytes() == actual.tobytes()
        journal = [
            json.loads(line)
            for line in (folder / "control-journal.jsonl").read_text().splitlines()
        ]
        assert sum(v["event"] == "requested" for v in journal) == 300
        assert sum(v["event"] == "boundary" for v in journal) == 300
    assert probe.read(tmp_path / "output/result.json") == result


def test_baseline_mismatch_blocks_candidate_construction(tmp_path, data):
    result, deps = generated_run(
        tmp_path, data, GeneratedDependencies(data, mismatch=True)
    )
    assert not result["analysis_valid"] and len(deps.bodies) == 1
    assert result["first_error"]["message"].endswith("qpos bytes differ")


@pytest.mark.parametrize(
    "kwargs,primary",
    [
        ({"constructor_failure": True, "close_failure": True}, "partial constructor"),
        (
            {"fail_control": 1, "close_failure": True, "hook_failure": True},
            "partial physics",
        ),
        ({"fail_control": 1, "clock_failure": True}, "partial physics"),
    ],
)
def test_partial_ownership_actual_clock_first_error_secondary_cleanup(
    tmp_path, data, kwargs, primary
):
    deps = GeneratedDependencies(data, **kwargs)
    result, _ = generated_run(tmp_path, data, deps)
    assert not result["analysis_valid"] and result["first_error"]["message"] == primary
    assert deps.bodies[0].closed
    arm = probe.read(tmp_path / "output/baseline/advance-ledger.json")
    if not kwargs.get("constructor_failure"):
        assert arm["controls_requested"] == 1 and arm["controls_returned"] == 0
        assert arm["native_samples_acquired"] == 3
        assert arm["actual_physical_s"] == (
            None if kwargs.get("clock_failure") else 3 * 0.0001
        )
        assert arm["partial_physics_known"] is not kwargs.get("clock_failure", False)
    assert len(result["errors"]) >= 2


def test_health_stop_baseline_invalid_raw_failing_sample_saved(tmp_path, data):
    result, deps = generated_run(
        tmp_path, data, GeneratedDependencies(data, health_control=1)
    )
    assert not result["analysis_valid"] and len(deps.bodies) == 1
    assert result["first_error"]["type"] == "HealthStop"
    with np.load(tmp_path / "output/baseline/native-contact.npz") as archive:
        assert archive["upright"][-1] == 0.8 and len(archive["upright"]) == 8


@pytest.mark.parametrize("control", [99, 101])
def test_candidate_health_prefix_transition(tmp_path, data, control):
    result, deps = generated_run(
        tmp_path,
        data,
        GeneratedDependencies(data, health_control=control, health_arm="candidate"),
    )
    if control == 99:
        assert not result["effective_validity"] and result["science"] is None
        assert result["first_error"]["type"] == "HealthStop"
        assert len(deps.bodies) == 2 and all(b.closed for b in deps.bodies)
        return
    assert result["effective_validity"] and result["analysis_valid"]
    assert result["science"]["verdict"] == "health_rejection"
    assert "metrics" not in result and result["first_error"] is None
    assert len(deps.bodies) == 2 and all(b.closed for b in deps.bodies)
    arm = probe.read(tmp_path / "output/candidate/advance-ledger.json")
    assert arm["controls_returned"] == 100 and arm["native_samples_acquired"] == 5008


@pytest.mark.parametrize("failure", ["writer", "saver", "meter"])
def test_generated_sink_and_resource_failure_returns_invalid(tmp_path, data, failure):
    def sink(*args):
        raise OSError("sink failure")

    kwargs = {failure: sink if failure != "meter" else lambda: 3 << 30}
    result, deps = generated_run(tmp_path, data, **kwargs)
    assert not result["effective_validity"]
    if failure != "writer":
        assert not deps.bodies
    assert result["first_error"]


def test_all_terminal_sinks_failure_does_not_escape_or_hide_primary(tmp_path, data):
    def sink(path, value):
        if path.name in {"result.json", "advance-ledger.json", "output-manifest.json"}:
            raise OSError("emergency disk failure")
        probe.put_json(path, value)

    deps = GeneratedDependencies(data, constructor_failure=True)
    result, _ = generated_run(tmp_path, data, deps, writer=sink, emergency_writer=sink)
    assert not result["effective_validity"] and not result["disk_agreement_established"]
    assert result["first_error"]["message"] == "partial constructor"
    assert len(result["errors"]) >= 4


def test_postcommit_deadline_retracts_disk_success(tmp_path, data):
    now = [0.0]

    def writer(path, value):
        probe.put_json(path, value)
        if path.name == "result.json" and value["analysis_valid"]:
            now[0] = 71.0

    result, _ = generated_run(tmp_path, data, writer=writer, clock=lambda: now[0])
    assert not result["analysis_valid"] and not result["effective_validity"]
    assert probe.read(tmp_path / "output/result.json") == result
    ledger = probe.read(tmp_path / "output/advance-ledger.json")
    assert ledger["final_deadline"] == 10 and ledger["repair_finished_after_deadline"]
    assert not ledger["analysis_valid"]


def test_final_read_deadline_cannot_reset(tmp_path, data):
    now = [0.0]

    def reader(path):
        value = probe.read(path)
        if path.name == "output-manifest.json":
            now[0] = 71.0
        return value

    result, _ = generated_run(tmp_path, data, reader=reader, clock=lambda: now[0])
    assert not result["effective_validity"]
    assert probe.read(tmp_path / "output/result.json") == result


@pytest.mark.parametrize(
    "hook",
    ["dependencies", "clock", "writer", "saver", "meter", "reader", "emergency_writer"],
)
def test_original_rejects_custom_hooks_before_load_or_output(tmp_path, hook):
    with pytest.raises(ValueError, match="original mode"):
        probe.run(output=tmp_path / "never", **{hook: lambda: None})
    assert not (tmp_path / "never").exists()


def test_original_gate_sha_failure_before_numerical_load(tmp_path, data):
    freeze, review = tmp_path / "freeze.json", tmp_path / "review.json"
    probe.put_json(freeze, {})
    probe.put_json(review, {})
    root = tmp_path
    output = root / probe.OUTPUT
    output.parent.mkdir(parents=True)
    with pytest.raises(ValueError, match="SHA"):
        probe.gate(freeze, "0" * 64, review, "0" * 64, output, root=root)
    assert not output.exists()


def test_cli_fixed_full_dimensions_and_no_test_injection(monkeypatch):
    captured = []
    monkeypatch.setattr(
        probe,
        "run",
        lambda *args, **kwargs: (
            captured.append((args, kwargs)) or {"effective_validity": False}
        ),
    )
    assert (
        probe.main(
            [
                "--freeze",
                "f",
                "--freeze-sha256",
                "x",
                "--review",
                "r",
                "--review-sha256",
                "y",
            ]
        )
        == 1
    )
    assert captured[0][1] == {} and captured[0][0][-1] == probe.ROOT / probe.OUTPUT
    with pytest.raises(ValueError):
        probe.TestDimensions(controls=2)


def test_generated_small_mujoco_partial_native_and_restoration(tmp_path):
    import mujoco as mj

    model = mj.MjModel.from_xml_string(
        """<mujoco><option timestep="0.0001" gravity="0 0 0"/>
      <worldbody><body><joint type="slide" axis="1 0 0"/><geom type="sphere" size=".01" mass="1"/>
      </body></worldbody><actuator><motor joint="0"/></actuator></mujoco>""".replace(
            "joint type=", 'joint name="joint" type='
        ).replace('joint="0"', 'joint="joint"')
    )
    original = mj.MjData(model)
    typ = mj.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mj.mj_stateSize(model, typ))
    mj.mj_getState(model, original, state, typ)
    original.ctrl[0] = 1
    times = []
    for _ in range(53):
        mj.mj_step(model, original)
        times.append(original.time)
    restored = mj.MjData(model)
    mj.mj_setState(model, restored, state, typ)
    restored.ctrl[0] = 1
    for _ in range(53):
        mj.mj_step(model, restored)
    assert original.qpos.tobytes() == restored.qpos.tobytes()
    assert original.qvel.tobytes() == restored.qvel.tobytes()
    assert times[-1] == original.time and original.time > 0.005
    # Exercise the production instance hook on a distinct generated model.
    # With no foot contacts its six missing slips must remain masked NaNs.
    actual = mj.MjData(model)
    body = SimpleNamespace(
        mj=mj,
        m=model,
        d=actual,
        fault=None,
        weight0=1.0,
        support_geom_ids=set(),
        _force_maps={"feet": (np.full(model.ngeom, -1, dtype=np.int64), None)},
        physics_time=lambda: actual.time,
        pose=lambda: (np.zeros(3), np.eye(3), np.zeros(6)),
        nonfoot_contact=lambda: False,
        _step_physics=lambda: mj.mj_step(model, actual),
    )
    samples = []
    original_step = body._step_physics
    detach = probe.Dependencies.install(body, samples.append)
    try:
        for _ in range(53):
            body._step_physics()
    finally:
        detach()
    assert body._step_physics is original_step
    native = probe.native_arrays(samples)
    probe.verify_native(native, count=53)
    assert not native["slip_valid"].any() and np.isnan(native["slip_mm_s"]).all()
    assert native["time_s"][-1] == actual.time
    assert "torch" not in sys.modules and "cupy" not in sys.modules


def test_r1_named_four_f5_knees_using_evaluator_stub(tmp_path, data, monkeypatch):
    from tools import analyze_c_banc_biological_gait as gait

    rows = copy.deepcopy(data["rows"])
    for row in rows:
        row["body"] = {"basis": np.eye(3).tolist()}
        row["tendon_inputs"] = {leg + "_tarsus": 0.1 for leg in probe.saved.LEGS}
    metrics = {
        "geometry_knee_span_deg": [26.0, 3.0, 41.0, 31.0, 4.0, 36.0],
        "gait": {
            "events": {
                leg: {
                    "complete_cycles": 3,
                    "recurrence_evaluable": True,
                    "period_cv": 0.1,
                }
                for leg in probe.saved.LEGS
            },
            "pair_phase": {
                a + "-" + b: {"samples": 3, "mean": 0.5}
                for a, b in (("lf", "rf"), ("lm", "rm"), ("lh", "rh"))
            },
        },
    }
    monkeypatch.setattr(probe, "read", lambda *_: {"samples": rows})
    monkeypatch.setattr(gait, "load_model", lambda *_: {})
    monkeypatch.setattr(probe.saved, "load_selected", lambda *_: {})
    monkeypatch.setattr(
        probe.saved,
        "physical_metrics",
        lambda *_args, **_kwargs: ({}, copy.deepcopy(metrics)),
    )
    _, report = probe.Dependencies.evaluate(tmp_path, {"curves": None, "startup": None})
    assert report["F5_separate"] == {
        "lf": {"minimum_deg": 25.0, "span_deg": 26.0, "passed": True},
        "rf": {"minimum_deg": 30.0, "span_deg": 31.0, "passed": True},
        "lh": {"minimum_deg": 40.0, "span_deg": 41.0, "passed": True},
        "rh": {"minimum_deg": 35.0, "span_deg": 36.0, "passed": True},
    }


def test_r1_corrupt_last_repair_checked_after_early_constructor_failure(tmp_path, data):
    corruptions = []

    def writer(path, value):
        if path.name == "output-manifest.json":
            raise OSError("first terminal publication failure")
        probe.put_json(path, value)

    def emergency(path, value):
        probe.put_json(path, value)
        if path.name == "result.json" and value.get("disk_agreement_established"):
            damaged = dict(value, corrupted_last_repair=True)
            probe.put_json(path, damaged)
            corruptions.append(path)

    result, _ = generated_run(
        tmp_path,
        data,
        GeneratedDependencies(data, constructor_failure=True),
        writer=writer,
        emergency_writer=emergency,
    )
    assert corruptions and not result["disk_agreement_established"]
    assert not result["effective_validity"]
    assert result["first_error"]["message"] == "partial constructor"
    assert any(
        e["message"] == "final repair readback differs" for e in result["errors"]
    )


class PartialNativeConstructor(FakeBody):
    def __init__(self, owner):
        super().__init__(owner)
        self.d = SimpleNamespace(time=0.0)
        self.constructor_calls = 0
        for _ in range(3):
            self._step_physics()
        raise RuntimeError("constructor after three actual native steps")

    def _step_physics(self):
        self.constructor_calls += 1
        self.d.time = self.constructor_calls * 0.0001


def test_r1_partial_constructor_physics_owned_count_restore_and_cleanup(tmp_path, data):
    class Deps(GeneratedDependencies):
        def body_class(self, data):
            return PartialNativeConstructor

    deps = Deps(data, close_failure=True)
    result, _ = generated_run(tmp_path, data, deps)
    assert (
        result["first_error"]["message"]
        == "constructor after three actual native steps"
    )
    assert any(e["message"] == "close failure" for e in result["errors"])
    body = deps.bodies[0]
    assert body.closed and "_step_physics" not in vars(body)
    assert PartialNativeConstructor._step_physics is type(body)._step_physics
    proof = probe.read(tmp_path / "output/baseline/construction-ledger.json")
    assert proof["steps_requested"] == proof["steps_returned"] == 3
    assert proof["absolute_before_s"] == 0 and proof["absolute_after_s"] == 3 * 0.0001
    assert proof["actual_advance_s"] == 3 * 0.0001
    assert proof["absolute_clock_known"] and proof["method_restored"]
    journal = [
        json.loads(v)
        for v in (tmp_path / "output/baseline/control-journal.jsonl")
        .read_text()
        .splitlines()
    ]
    final = [v for v in journal if v["event"] == "constructor_finished_or_partial"][-1]
    assert (
        final["python_returned"] is False
        and final["absolute_clock"]["steps_returned"] == 3
    )


def test_r1_unreturned_constructor_step_keeps_known_absolute_advance(tmp_path, data):
    class Body(PartialNativeConstructor):
        def _step_physics(self):
            super()._step_physics()
            if self.constructor_calls == 3:
                raise RuntimeError("native integration advanced before return failure")

    class Deps(GeneratedDependencies):
        def body_class(self, data):
            return Body

    result, deps = generated_run(tmp_path, data, Deps(data))
    assert (
        result["first_error"]["message"]
        == "native integration advanced before return failure"
    )
    proof = probe.read(tmp_path / "output/baseline/construction-ledger.json")
    assert proof["steps_requested"] == 3 and proof["steps_returned"] == 2
    assert proof["actual_advance_s"] == 3 * 0.0001 and proof["absolute_clock_known"]
    assert deps.bodies[0].closed and "_step_physics" not in vars(deps.bodies[0])


def test_r1_constructor_unknown_clock_is_not_zero(tmp_path, data):
    class Body(PartialNativeConstructor):
        def _step_physics(self):
            super()._step_physics()
            if self.constructor_calls == 3:
                del self.d
                raise RuntimeError("partial constructor clock lost")

    class Deps(GeneratedDependencies):
        def body_class(self, data):
            return Body

    result, _ = generated_run(tmp_path, data, Deps(data))
    assert result["first_error"]["message"] == "partial constructor clock lost"
    proof = probe.read(tmp_path / "output/baseline/construction-ledger.json")
    assert proof["absolute_after_s"] is None and proof["actual_advance_s"] is None
    assert proof["absolute_clock_known"] is False and proof["clock_read_errors"]


def test_r1_owned_counters_complete_2000_15000_34000_without_class_patch():
    class Body:
        def __init__(self):
            self.d = SimpleNamespace(time=0.0)
            self.steps = 0
            for _ in range(2000):
                self._step_physics()

        def _step_physics(self):
            self.steps += 1
            self.d.time = self.steps * 0.0001

    original = Body._step_physics
    total = 0
    for _ in range(2):
        body = Body.__new__(Body)
        events = []
        constructor = probe.InstancePhysicsCounter(
            body, "constructor", journal=events.append
        )
        Body.__init__(body)
        constructor.restore()
        assert (
            constructor.proof["steps_requested"]
            == constructor.proof["steps_returned"]
            == 2000
        )
        assert constructor.proof["actual_advance_s"] == 0.2
        assert len(events) == 9 and "_step_physics" not in vars(body)
        assert events[-1]["steps_requested"] == events[-1]["steps_returned"] == 2000
        assert events[-1]["actual_advance_s"] == 0.2
        assert events[-1]["unrecorded_steps_after_this_record"] == "unknown"
        science = probe.InstancePhysicsCounter(body, "science")
        for _ in range(15000):
            body._step_physics()
        science.restore()
        assert (
            science.proof["steps_requested"] == science.proof["steps_returned"] == 15000
        )
        assert science.proof["actual_advance_s"] == pytest.approx(1.5)
        total += constructor.proof["steps_returned"] + science.proof["steps_returned"]
    assert total == 34000 and Body._step_physics is original


def test_r1_constructor_start_sink_failure_closes_owned_uninitialized_instance(
    tmp_path, data
):
    owned = []

    class Body:
        def __new__(cls):
            body = object.__new__(cls)
            body.closed = False
            owned.append(body)
            return body

        def __init__(self, *_):
            raise AssertionError("constructor must not run after start sink failure")

        def _step_physics(self):
            raise AssertionError("native must not run after start sink failure")

        def close(self):
            self.closed = True

    class Deps(GeneratedDependencies):
        def body_class(self, data):
            return Body

    def writer(path, value):
        if path.name == "construction-ledger.json":
            raise OSError("constructor start ledger unavailable")
        probe.put_json(path, value)

    result, _ = generated_run(tmp_path, data, Deps(data), writer=writer)
    assert result["first_error"]["message"] == "constructor start ledger unavailable"
    assert not result["analysis_valid"] and len(owned) == 1 and owned[0].closed
    assert "_step_physics" not in vars(owned[0])
    proof = probe.read(tmp_path / "output/baseline/advance-ledger.json")[
        "construction_physics"
    ]
    assert proof["steps_requested"] == proof["steps_returned"] == 0
    assert proof["absolute_before_s"] is None and proof["actual_advance_s"] is None
    assert not proof["absolute_clock_known"] and proof["method_restored"]
