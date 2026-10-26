"""Small generated model/wrapper/metadata checks; no Original numerical input."""
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco as mj
import numpy as np
import pytest

from tools import analyze_c_banc_m3_local_foot_geometry as analyze
from tools import banc_m3_local_foot_geometry_admission as admission
from tools import observe_c_banc_m3_local_foot_geometry as capture


class SmallPublisher:
    def __init__(self, folder):
        self.folder, self.total, self.result_cap = folder, 0, 93 << 20

    def write(self, name, raw):
        p = self.folder / name
        p.write_bytes(raw)
        self.total += len(raw)
        return {"path": str(p), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    def json(self, name, value):
        return self.write(name, json.dumps(value, allow_nan=False).encode("utf8"))


def synthetic_model(mocap):
    pieces = ['<mujoco><compiler angle="radian"/><worldbody>']
    if mocap:
        pieces.append('<body name="marker" mocap="true" pos=".1 .2 .3"><geom size=".02"/></body>')
    pieces.append('<body name="fly/c_thorax" pos="1 2 3" quat=".98 .1 .05 .02"><freejoint name="root"/><geom size=".05"/>')
    pieces.append('<body name="passive"><joint name="passive" axis="0 1 0"/><geom size=".04"/>')
    for leg in capture.LEGS:
        for j in range(7):
            name = 'fly/' + leg + ('_tarsus5' if j == 6 else '_segment' + str(j))
            pieces.append(f'<body name="{name}" pos=".02 .03 .01"><joint name="{leg}_j{j}" axis="0 0 1"/><geom size=".01" mass=".001"/>')
        pieces.extend(['</body>'] * 7)
    pieces.append('</body></body></worldbody><actuator>')
    for leg in capture.LEGS:
        for j in range(7):
            pieces.append(f'<position name="{leg}_a{j}" joint="{leg}_j{j}" ctrllimited="true" ctrlrange="-1 1"/>')
    pieces.append('</actuator></mujoco>')
    return mj.MjModel.from_xml_string(''.join(pieces))


def model_metadata(model):
    joints = np.arange(2, 44, dtype=np.int64)[::-1].copy()
    act = np.arange(42, dtype=np.int64)[::-1].copy()
    qpos, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    tips = np.asarray([mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'fly/' + leg + '_tarsus5') for leg in capture.LEGS], np.int64)
    thorax = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'fly/c_thorax')
    control = {"joint_ids": joints.tolist(), "qpos_ids": qpos.tolist(), "actuator_ids": act.tolist(), "joint_names": [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, int(i)) for i in joints],
               "activation_tau_s": 0., "interpolation": "held raw ctrl", "ctrl_to_q_mapping": {"status": "VERIFIED_UNIT_GEAR_JOINT_HINGE"},
               "control_clamp": {"status": "VERIFIED_COMPILED_FLAG", "enabled": True}, "actuator_ctrllimited": [True] * 42,
               "actuator_ctrlrange_rad": model.actuator_ctrlrange[act].tolist(), "actuator_gear": model.actuator_gear[act].tolist(),
               "actuator_trntype": model.actuator_trntype[act].tolist(), "disableflags": int(model.opt.disableflags), "enableflags": int(model.opt.enableflags)}
    return {"dimensions": capture.model_dimensions(model), "fk_identity": capture.fk_identity(model, np),
            "active_joint_ids": joints.tolist(), "active_qpos_ids": qpos.tolist(), "active_dof_ids": dofs.tolist(),
            "active_joint_names": control["joint_names"], "tip_ids": tips.tolist(), "tip_names": ['fly/' + leg + '_tarsus5' for leg in capture.LEGS],
            "thorax_id": thorax, "thorax_name": 'fly/c_thorax', "control_mapping": control, "mujoco_version": mj.__version__,
            "control_identity": capture.control_identity(model, act, np), "control_end_verified": True,
            "control_boundary_verified": {str(c): True for c in capture.CONTROLS}, "physics_t0_absolute_s": .2}


def synthetic_pose(model, metadata):
    data = mj.MjData(model)
    pose = {k: np.zeros(shape, dtype) for k, (shape, dtype) in capture.pose_fields(metadata["dimensions"]).items()}
    qpos = np.asarray(metadata["active_qpos_ids"], np.int64)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for i, c in enumerate(capture.CONTROLS):
        data.qpos[:] = model.qpos0
        data.qpos[7] = .17  # nonzero passive hinge, outside active42
        data.qpos[qpos] = .0001 * c
        data.qvel[:] = .003
        if model.nmocap:
            data.mocap_pos[:] = [.2, -.1, .4]
            data.mocap_quat[:] = [1., 0., 0., 0.]
        data.time = metadata["physics_t0_absolute_s"] + .005 * c
        mj.mj_forward(model, data)
        for key in ("qpos", "qvel", "mocap_pos", "mocap_quat"):
            pose[key][i] = getattr(data, key)
        pose["control"][i], pose["network"][i], pose["valid"][i] = c, 20*c, True
        pose["model_time_absolute_s"][i], pose["physics_time_normalized_s"][i] = data.time, .005*c
        pose["tip_world_mm"][i], pose["thorax_points_mm"][i] = analyze.read_points(data, metadata["tip_ids"], metadata["thorax_id"], np)
        pose["normal_load_bw"][i] = [.02, 0., .03, .0, .04, .01] if i % 2 == 0 else [.02, 0., 0., .03, .04, .01]
        pose["recorded_load_bearing"][i] = pose["normal_load_bw"][i] > .02
        if c in capture.ANCHORS:
            j = capture.ANCHORS.index(c)
            for leg, tip in enumerate(metadata["tip_ids"]):
                mj.mj_jacBody(model, data, jp, jr, tip)
                pose["jacobian_full_nv"][j, leg] = jp
            pose["anchor_control"][j], pose["jacobian_valid"][j], pose["interval_valid"][j] = c, True, True
            pose["interval_raw_ctrl_rad"][j] = np.broadcast_to(data.qpos[qpos] + .001, (50, 42))
    return pose


@pytest.mark.parametrize("mocap", [False, True])
def test_math_model_roundtrip_full_context_J_FD_344_and_qpos_dof_offset(tmp_path, mocap):
    model = synthetic_model(mocap)
    metadata, publisher = model_metadata(model), SmallPublisher(tmp_path)
    pose = synthetic_pose(model, metadata)
    assert np.all(np.asarray(metadata["active_qpos_ids"]) != metadata["active_dof_ids"])
    assert np.all(pose["qpos"][:, :3] != 0) and (pose["mocap_pos"].nbytes > 0) == mocap
    spec, receipt = capture.save_model_once(mj, model, np, publisher, sum(v.nbytes for v in pose.values()), admission.limits_for("geometry_capture"))
    assert receipt["save_requested"] == receipt["save_returned"] == 1 and receipt["buffer_plus_copy_peak_bytes"] == 2 * spec["bytes"]
    work = analyze.empty_work()
    restored, data = analyze.construct_pair(mj, tmp_path / 'model.mjb', work, lambda label: None)
    checks, work, raw = analyze.reconstruct(restored, data, pose, metadata, mj, np, lambda label: None, receipts=work)
    assert all(value["passed"] for value in checks.values()), checks
    assert work["forward_requested"] == work["forward_returned"] == 344 and work["cpu_jacobian_returned"] == 24
    assert work["model_constructor_requested"] == work["model_constructor_returned"] == work["data_constructor_requested"] == work["data_constructor_returned"] == 1
    assert raw["CPU_J_full_nv"].shape[-1] == model.nv and raw["FD_J_active"].shape[-1] == 42
    assert work["physics_steps"] == work["Torch_CUDA"] == work["body_constructors"] == 0
    rows, categories = analyze.local_table(pose, metadata, np, control_identity_verified=True)
    assert len(rows) == 24 and sum(v["count"] for v in categories.values()) == 24
    assert rows[0]["loaded_pre"] and rows[0]["recorded_load_pre"] is False  # >=.02 differs from recorded strict boolean
    assert set(rows[0]["observed_world_delta"]) == {"start_thorax_AP_mm", "world_vertical_mm", "start_body_dorsal_mm"}
    assert "unattributed_kinematic_residual" in rows[0]


def test_math_model_size_guard_refuses_before_buffer_or_save_and_mapping_rejects_drift(tmp_path):
    attempted = []
    work = capture.empty_model_work()
    publisher = SmallPublisher(tmp_path)
    fake = SimpleNamespace(mj_sizeModel=lambda model: (96 << 20) + 1, mj_saveModel=lambda *args, **kwargs: attempted.append(True))
    with pytest.raises(ValueError, match="mj_sizeModel"):
        capture.save_model_once(fake, object(), SimpleNamespace(empty=lambda *args: attempted.append(True)), publisher, 16,
                                admission.limits_for("geometry_capture"), work=work)
    assert attempted == []
    preflight = json.loads((tmp_path / "model-size-preflight.json").read_bytes())
    assert preflight["actual_size_bytes"] == (96 << 20) + 1 and preflight["actual_size_type"] == "builtins.int"
    assert preflight["size_requested"] == preflight["size_returned"] == 1 and preflight["save_requested"] == 0
    result, artifacts = {"first_error": {"message": "original size failure"}, "capture_complete": False}, {}
    capture.geometry_artifacts(None, result, artifacts, publisher, serialization=work)
    assert result["first_error"]["message"] == "original size failure" and not result["geometry"]["complete"]
    assert result["geometry"]["serialization"]["buffer_requested"] == 0 and "geometry_work" in artifacts
    model = synthetic_model(False)
    metadata = model_metadata(model)
    metadata["active_dof_ids"] = metadata["active_dof_ids"][::-1]
    with pytest.raises(ValueError, match="addresses"):
        analyze.validate_mapping(model, metadata, mj, np)
    metadata = model_metadata(model)
    model.qpos0[7] += .1
    with pytest.raises(ValueError, match="FK"):
        analyze.validate_mapping(model, metadata, mj, np)


def test_math_model_save_failure_keeps_actual_requested_returned_receipt(tmp_path):
    work = capture.empty_model_work()

    def save_failure(*args, **kwargs):
        raise RuntimeError("injected model save failure")

    fake = SimpleNamespace(mj_sizeModel=lambda model: 64, mj_saveModel=save_failure)
    publisher = SmallPublisher(tmp_path)
    with pytest.raises(RuntimeError, match="model save failure"):
        capture.save_model_once(fake, object(), np, publisher, 16, admission.limits_for("geometry_capture"), work=work)
    assert work["actual_size_bytes"] == 64 and work["buffer_requested"] == work["buffer_returned"] == 1
    assert work["save_requested"] == 1 and work["save_returned"] == work["copy_requested"] == work["publication_requested"] == 0
    result = {"first_error": {"message": "injected model save failure"}, "capture_complete": False}
    capture.geometry_artifacts(None, result, {}, publisher, serialization=work)
    receipt = json.loads((tmp_path / "model-serialization-work.json").read_bytes())
    assert receipt["save_requested"] == 1 and receipt["save_returned"] == 0 and not (tmp_path / "model.mjb").exists()


def test_math_nonzero_t0_restored_absolute_clock_and_wrong_channels_refused():
    model, work = synthetic_model(False), analyze.empty_work()
    metadata = model_metadata(model)
    pose, data = synthetic_pose(model, metadata), mj.MjData(model)
    analyze.restore_pose(data, pose, 0, np)
    assert data.time == .7 and pose["physics_time_normalized_s"][0] == .5
    for channel in ("model_time_absolute_s", "physics_time_normalized_s"):
        wrong = {k: v.copy() for k, v in pose.items()}
        wrong[channel][0] += .2
        with pytest.raises(ValueError, match="boundaries"):
            analyze.reconstruct(model, data, wrong, metadata, mj, np, lambda label: None, receipts=work)
    assert work["forward_requested"] == work["forward_returned"] == 0
    # A passed model/data pair never claims that reconstruct created it.
    assert work["model_constructor_requested"] == work["data_constructor_requested"] == 0


def test_math_compiled_control_drift_and_common_drift_separate_qualification():
    model = synthetic_model(False)
    metadata = model_metadata(model)
    pose = synthetic_pose(model, metadata)
    model.actuator_ctrlrange[0, 0] -= .1
    with pytest.raises(ValueError, match="compiled control identity"):
        analyze.validate_mapping(model, metadata, mj, np)
    rows, _ = analyze.local_table(pose, metadata, np)
    assert all(row["J_command"] is None and row["command_status"] == "UNVERIFIED" for row in rows)
    model.actuator_ctrlrange[0, 0] += .1
    model.opt.disableflags = int(model.opt.disableflags) | int(mj.mjtDisableBit.mjDSBL_CLAMPCTRL)
    with pytest.raises(ValueError, match="compiled control identity"):
        analyze.validate_mapping(model, metadata, mj, np)
    # Inject one real scalar drift into the same common comparator/predicate.
    from tools.analyze_c_banc_m3_startup_tracking import compare_values
    expected, current = np.zeros(2, np.float64), np.zeros(2, np.float64)
    result = analyze.initial_result()
    result.update(run_valid=True, checks={"fixture": {"passed": True}}, control_mapping_CPU_verified=True)
    result["baseline_comparison"] = {"common_values_exact": compare_values(current, expected, np)["exact"]}
    analyze.qualification(result)
    assert result["qualified"] and result["next_learning_input_qualified"]
    current[1] = 1e-15
    result["baseline_comparison"]["common_values_exact"] = compare_values(current, expected, np)["exact"]
    analyze.qualification(result)
    assert result["run_valid"] and result["FK_numerical_passed"] and not result["common_values_exact"]
    assert not result["qualified"] and not result["geometry_passed"] and not result["historical_correspondence_qualified"]
    assert not result["next_learning_input_qualified"] and result["classification"] == "scientific_rejected"


class InjectedPublisher:
    def __init__(self, refused=()):
        self.refused, self.records, self.attempts, self.total = set(refused), {}, [], 0

    def record(self, name, value):
        self.attempts.append(name)
        if name in self.refused or "ALL" in self.refused:
            raise OSError("injected bounded writer refusal: " + name)
        self.records[name] = copy.deepcopy(value)
        self.total += 1

    def arrays(self, name, value):
        self.record(name, value)

    def json(self, name, value):
        self.record(name, value)

    def write(self, name, value):
        self.record(name, value)


def test_math_import_constructor_and_guarded_writer_failures_keep_primary_partial_work(monkeypatch, capsys):
    for key in analyze.THREAD_ENV:
        monkeypatch.setenv(key, "1")
    fake = SimpleNamespace(role="geometry_analyze", stage="pre_array", fence=lambda label: None, check=lambda label: None)
    writer = InjectedPublisher(("geometry-analysis.json",))
    monkeypatch.setattr(analyze, "publisher_for", lambda q: writer)

    def import_failure():
        raise ImportError("injected numerical import failure")
    monkeypatch.setattr(analyze, "numerical_modules", import_failure)
    with pytest.raises(ValueError, match="numerical import failure"):
        analyze.worker(fake)
    report = writer.records["geometry-publication-failure.json"]
    assert report["first_error"]["stage"] == "numerical-import"
    assert report["secondary_errors"][0]["stage"] == "publish-report-json"
    assert report["work"]["model_constructor_requested"] == report["work"]["data_constructor_requested"] == 0
    assert not report["run_valid"] and not report["publication_valid"] and not report["qualified"]

    from tools import analyze_c_banc_m3_startup_tracking as old
    model_spec = {"path": "small-generated-model.mjb", "bytes": 16, "sha256": "0" * 64}
    bundle = {"model": model_spec, "manifest": "manifest", "body_report": "body_report", "geometry_metadata": "metadata", "geometry": "pose"}
    ledger = []

    class Selected:
        def __init__(self, q, np):
            self.ledger, self.expanded = ledger, 0
        def json(self, spec):
            return {"role": "geometry_capture", "run_valid": True, "capture_complete": True,
                    "artifacts": {k: v for k, v in bundle.items() if k != "manifest"}} if spec == "manifest" else {"geometry": {"complete": True}} if spec == "body_report" else {"dimensions": {"nq": 1, "nv": 1, "nmocap": 0}}
        def arrays(self, spec, fields):
            return {}
        def digest(self, spec):
            return Path(spec["path"]), 42
        def posthash(self):
            assert self.ledger == [(model_spec, 42)]  # failure still checks the actual MJB read
            return [{"fixture_model_posthash": True}]

    def data_failure(model):
        assert ledger == [(model_spec, 42)]
        raise RuntimeError("injected data constructor failure")
    constructors = SimpleNamespace(MjModel=SimpleNamespace(from_binary_path=lambda path: object()), MjData=data_failure)
    monkeypatch.setattr(old, "SavedInputs", Selected)
    monkeypatch.setattr(analyze, "numerical_modules", lambda: (constructors, np))
    writer = InjectedPublisher()
    monkeypatch.setattr(analyze, "publisher_for", lambda q: writer)
    fake.request, fake.limits = {"artifacts": bundle}, {"host_bytes": 2 << 30}
    with pytest.raises(ValueError, match="data constructor failure"):
        analyze.worker(fake)
    report = writer.records["geometry-analysis.json"]
    assert report["first_error"]["stage"] == "CPU-model-one-pair" and report["input_pre_post_hashes"] == [{"fixture_model_posthash": True}]
    assert report["work"]["model_constructor_requested"] == report["work"]["model_constructor_returned"] == report["work"]["data_constructor_requested"] == 1
    assert report["work"]["data_constructor_returned"] == report["work"]["forward_requested"] == 0
    def model_failure(path):
        raise RuntimeError("injected model constructor failure")
    work = analyze.empty_work()
    with pytest.raises(RuntimeError, match="model constructor failure"):
        analyze.construct_pair(SimpleNamespace(MjModel=SimpleNamespace(from_binary_path=model_failure)), Path("synthetic.mjb"), work, fake.check)
    assert work["model_constructor_requested"] == 1 and work["model_constructor_returned"] == work["data_constructor_requested"] == 0

    first = copy.deepcopy(report["first_error"])
    refused = InjectedPublisher(("geometry-analysis.npz", "geometry-analysis.json"))
    analyze.publish_result(refused, {"fixture": np.zeros(1)}, [], report)
    final = refused.records["geometry-publication-failure.json"]
    assert final["first_error"] == first and final["work"] == report["work"]
    assert {v["stage"] for v in final["secondary_errors"]} >= {"publish-raw", "publish-report-json"}
    technically_valid = analyze.initial_result()
    technically_valid.update(run_valid=True, checks={"fixture": {"passed": True}}, control_mapping_CPU_verified=True,
                             baseline_comparison={"common_values_exact": True})
    analyze.qualification(technically_valid)
    analyze.publish_result(InjectedPublisher(("geometry-analysis.npz",)), {"fixture": np.zeros(1)}, [], technically_valid)
    assert technically_valid["first_error"]["stage"] == "publish-raw" and not technically_valid["run_valid"] and not technically_valid["qualified"]
    writer = InjectedPublisher(("ALL",))
    monkeypatch.setattr(analyze, "publisher_for", lambda q: writer)
    ledger.clear()
    with pytest.raises(ValueError, match="data constructor failure"):
        analyze.worker(fake)
    stderr = capsys.readouterr().err
    assert "data constructor failure" in stderr and "publish-failure-json" in stderr and "model_constructor_returned" in stderr


def test_math_post_forward_wrapper_one_call_copy_and_inherited_restoration_on_error(tmp_path, monkeypatch):
    from tools import observe_c_banc_m3_startup_tracking as tracking
    model = synthetic_model(False)
    metadata, data = model_metadata(model), mj.MjData(model)

    class BaseTracker:
        def __init__(self, session, np, metadata, check):
            self.s, self.np, self.metadata, self.check = session, np, metadata, check
            self.a = {"native_control_end": np.full(50, 101, np.int64), "native_returned": np.ones(50, bool), "native_raw_ctrl_rad": np.zeros((50, 42))}
        def install(self):
            return self
        def boundary(self, row, boundary):
            pass
        def close(self):
            pass

    class Body:
        def physics_time(self):
            return float(self.d.time - self.t0)

        def _forward_physics(self):
            self.calls += 1
            if self.fail:
                raise RuntimeError("original forward failure")
            self.d.qpos[self.qpos_ids[0]] = .123
            mj.mj_forward(self.m, self.d)

    body = Body()
    body.m, body.d, body.mj, body.calls, body.fail, body.t0 = model, data, mj, 0, False, .2
    body.joint_ids = np.asarray(metadata["active_joint_ids"], np.int64)
    body.qpos_ids, body._tip_ids, body.thorax = np.asarray(metadata["active_qpos_ids"]), np.asarray(metadata["tip_ids"]), metadata["thorax_id"]
    session = SimpleNamespace(body=body, control_tick=100, network=SimpleNamespace(tick=2000))
    monkeypatch.setattr(tracking, "TrackingObserver", BaseTracker)
    timer = SimpleNamespace(limits=admission.limits_for("geometry_capture"))
    observer = capture.geometry_factory(SmallPublisher(tmp_path), timer)(session, np, metadata["control_mapping"], lambda label: None)
    observer.install()
    observer.forward_requested = observer.forward_returned = 99
    data.time = .7
    body._forward_physics()
    observer.boundary({"feet": {"normal_load_bw": [.03]*6, "load_bearing": [True]*6}}, {})
    assert body.calls == 1 and observer.g["valid"][0] and observer.g["qpos"][0, body.qpos_ids[0]] == .123
    assert observer.jac_requested == observer.jac_returned == 6
    assert observer.g["model_time_absolute_s"][0] == observer.last_forward_time == .7
    assert abs(observer.g["physics_time_normalized_s"][0] - .5) < 1e-15 and observer.geometry_metadata["physics_t0_absolute_s"] == .2
    body.fail = True
    with pytest.raises(RuntimeError, match="original forward failure"):
        body._forward_physics()
    observer.close()
    assert '_forward_physics' not in vars(body) and body.calls == 2
    assert not observer.geometry_summary()["complete"] and observer.forward_requested != observer.forward_returned


@pytest.mark.parametrize("cleanup_stage", ("close", "publish"))
def test_math_capture_cleanup_failure_preserves_primary_input_error(tmp_path, monkeypatch, cleanup_stage):
    from tools import observe_c_banc_m3_startup_tracking as tracking
    from tools import verify_c_banc_m3_candidate_body_handoff as body
    from tools import verify_c_banc_m3_startup_ray as ray

    primary = OSError("injected first selected-input failure")
    calls = []

    class Watch:
        def snapshot(self):
            return {"state": "RUNNING"}

        def close(self):
            calls.append("close")
            if cleanup_stage == "close":
                raise RuntimeError("injected sampler close failure")
            return {"state": "CLOSED"}

    class Publisher:
        def json(self, name, value):
            calls.append("publish")
            raise RuntimeError("injected sampler publication failure")

    def input_failure(path):
        raise primary

    monkeypatch.setattr(body, "gpu_watch", lambda request: Watch())
    monkeypatch.setattr(tracking, "bounded_publisher", lambda request: Publisher())
    monkeypatch.setattr(ray, "leaf", lambda spec, root: tmp_path / "generated-not-read.json")
    monkeypatch.setattr(ray, "json_load", input_failure)
    request = SimpleNamespace(role="geometry_capture", stage="pre_array", fence=lambda label: None,
                              root=tmp_path, limits=admission.limits_for("geometry_capture"),
                              roster={"selected_candidate": {}})
    with pytest.raises(OSError, match="first selected-input failure") as raised:
        capture.worker(request)
    assert raised.value is primary
    assert calls == (["close"] if cleanup_stage == "close" else ["close", "publish"])
    expected_cleanup = "sampler close failure" if cleanup_stage == "close" else "sampler publication failure"
    assert any("sampler cleanup" in note and expected_cleanup in note for note in primary.__notes__)


def test_gate_lazy_imports_CPU_threads_and_deadline_before_source_reads(monkeypatch):
    from tools import banc_m3_startup_ray_supervisor as supervisor
    code = "import sys,os,json; from tools import observe_c_banc_m3_local_foot_geometry, analyze_c_banc_m3_local_foot_geometry, banc_m3_local_foot_geometry_admission; assert not any(k in sys.modules for k in ('numpy','mujoco','torch','flygym')); print(json.dumps({k:os.environ.get(k) for k in analyze_c_banc_m3_local_foot_geometry.THREAD_ENV}))"
    inherited = dict(os.environ, **dict.fromkeys(analyze.THREAD_ENV, "3"))
    token = {"fixture": "generated pre-import environment observation"}
    environment = supervisor.worker_environment("geometry_analyze", token, sys.executable, inherited=inherited)
    outcome = subprocess.run([sys.executable, '-c', code], env=environment, capture_output=True, timeout=10, check=False)
    assert outcome.returncode == 0, outcome.stderr
    assert json.loads(outcome.stdout) == dict.fromkeys(analyze.THREAD_ENV, "1")
    assert json.loads(environment["FLYLAB_DIRECT_SCIENCE_OWNER"]) == token and environment["__PYVENV_LAUNCHER__"] == sys.executable
    assert {k: inherited[k] for k in analyze.THREAD_ENV} == dict.fromkeys(analyze.THREAD_ENV, "3")
    other = supervisor.worker_environment("geometry_capture", token, sys.executable, inherited=inherited)
    assert {k: other[k] for k in analyze.THREAD_ENV} == dict.fromkeys(analyze.THREAD_ENV, "3")
    touched = []
    monkeypatch.setattr(admission, 'spec_for', lambda *args: touched.append(args))
    monkeypatch.setattr(admission.time, 'perf_counter', lambda: 371.)
    with pytest.raises(TimeoutError):
        admission.Admission('never-open.json', 'pre_spawn', entry=100.)
    assert touched == []


def historical_fixture():
    def read(spec):
        require_path = admission.path(spec['path'], admission.ROOT)
        assert require_path.suffix == '.json' and require_path.stat().st_size <= 1048576
        raw = require_path.read_bytes()
        assert len(raw) == spec['bytes'] and hashlib.sha256(raw).hexdigest() == spec['sha256']
        return json.loads(raw)
    roster, grant = read(admission.ROSTER), read(admission.GRANT)
    before = read(roster['failed_geometry_attempt']['closed_ledger'])
    q = {'role': 'geometry_capture', 'artifacts': {}}
    def archived(item):
        raw = Path(item['archive_path']).read_bytes()
        assert len(raw) <= 2097152
        return raw
    def closed(cost, debit):
        assert cost['cost_id'] == debit['cost_id'] and cost['charged_once'] and cost['all_owned_ended'] and cost['reservation_released']
        return cost
    return before, grant, roster, q, read, closed, archived


def test_gate_native_failed_work_accepts_JSON_int64_and_rejects_value_or_type_drift(tmp_path):
    construction = {"body_constructors_requested": 1, "body_constructors_returned": 1,
                    "neural_setup_requested": 320, "neural_setup_host_returned": 320,
                    "neural_capture_requested": 160, "neural_capture_recorded": 160,
                    "settling_physics_requested": 2000, "settling_physics_host_returned": 2000}
    counters = dict.fromkeys(("controls_requested", "controls_host_returned", "controls_completed", "neural_requested",
                             "neural_host_returned", "neural_device_confirmed", "physics_requested", "physics_host_returned", "physics_device_confirmed"), 0)
    small = tmp_path / 'generated-work.json'
    small.write_text(json.dumps({"construction": construction, "counters": counters}), encoding='utf8')
    script = tmp_path / 'check-production-work.ps1'
    source = Path(admission.ROOT) / admission.BASE / 'parent-direct-science-account.ps1'
    script.write_text("""param([string]$Source,[string]$Small)
$ErrorActionPreference='Stop'
$tokens=$null;$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($Source,[ref]$tokens,[ref]$errors)
if($errors.Count){throw 'Production Native parse errors'}
foreach($name in @('Need','Same','Geometry-FailedWork')){
  $defs=@($ast.FindAll({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -ceq $name},$true))
  if($defs.Count -ne 1){throw 'Exactly one actual production function'}
  Invoke-Expression $defs[0].Extent.Text
}
$report=Get-Content -LiteralPath $Small -Raw|ConvertFrom-Json -AsHashtable
Geometry-FailedWork $report
$report.counters.controls_completed=[long]1
$rejected=$false;try{Geometry-FailedWork $report}catch{$rejected=$_.Exception.Message -match 'normal interval'}
if(-not $rejected){throw 'Wrong normal work accepted'}
$report.counters.controls_completed=[long]0
$report.construction.body_constructors_returned=[double]1
$rejected=$false;try{Geometry-FailedWork $report}catch{$rejected=$_.Exception.Message -match 'construction type'}
if(-not $rejected){throw 'Wrong constructor type accepted'}
Write-Output 'production accepted JSON int64; rejected wrong work and double constructor'
""", encoding='utf8')
    result = subprocess.run(['pwsh', '-NoProfile', '-File', str(script), '-Source', str(source), '-Small', str(small)],
                            capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'production accepted JSON int64' in result.stdout


def test_gate_immutable_closed200_history_archive8plus6_and_failed_replay_refusal():
    f = historical_fixture()
    original = copy.deepcopy(f[0])
    assert admission.validate_history(*f) > 0 and f[0] == original
    original_read = f[4]
    def failed_read(spec):
        value = original_read(spec)
        if spec == f[2]['replay_costs'][3]:
            value['exit_code'] = 1
        return value
    with pytest.raises(ValueError, match='completed replay'):
        admission.validate_history(*f[:4], failed_read, *f[5:])
    def altered_failed_capture(spec):
        value = original_read(spec)
        if spec == f[2]['failed_geometry_attempt']['body_report']:
            value['counters']['controls_completed'] = 1
        return value
    with pytest.raises(ValueError, match='failed-a normal interval'):
        admission.validate_history(*f[:4], altered_failed_capture, *f[5:])
    f[0]['debits'][0]['cost_id'] = 'changed-historical-row'
    with pytest.raises(ValueError, match='historical debit'):
        admission.validate_history(*f)


@pytest.mark.parametrize('role', admission.ROLES)
def test_gate_actual_pre_spawn_refusal_and_temp_native_once_only(tmp_path, monkeypatch, role):
    from tools import banc_m3_startup_ray_supervisor as supervisor
    def write(name, value):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value, allow_nan=False), encoding='utf8')
        return {'path': name, 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    call, limits = admission.CALLS[admission.ROLES.index(role)], admission.limits_for(role)
    science = write(admission.BASE + '/synthetic-geometry-science.json', {'schema': admission.SCHEMA+'.request.v1', 'mode': 'original', 'role': role, 'limits': {}})
    freeze = write(admission.BASE + '/synthetic-geometry-freeze.json', {'fixture': 'generated metadata refusal'})
    q = {'schema': 'flylab.m3.direct-science.supervision.v1', 'mode': 'original', 'role': role, 'call_id': call, 'science_request': science,
         'source_freeze': freeze, 'grant': admission.GRANT, 'ledger_path': admission.LEDGER, 'shared_original_lock_path': admission.SHARED_LOCK,
         'output': admission.BASE+'/parent-'+call+'/result-supervision', 'limits': {k: limits[k] for k in ('outer_s','child_s','cleanup_s','host_bytes','output_cap_bytes')}}
    request = write(admission.BASE+'/synthetic-geometry-owner.json', q)
    debits = [{'cost_id': 'synthetic-old-'+str(i), 'mode': 'generated'} for i in range(165)]
    before = write(admission.BASE+'/synthetic-geometry-before.json', {'debits': debits})
    policy = write(admission.BASE+'/synthetic-geometry-policy.json', {'schema':'flylab.m3.direct-science.policy.v1','generated_local_limit_s':300,'original_authorized':False,'before_ledger':before,'local_debit_offset':165})
    ledger = {'debits':debits,'direct_science_policy':policy,'direct_science_calls':{call:{'request':request,'state':'admitted','mode':'original','reserved_s':limits['outer_s']}},
              'startup_ray_original_workers':{},'startup_ray_original_reservations':{},'startup_ray_original_pending_actuals':{},'startup_ray_original_blocking_holds':{},
              'direct_science_pending_actuals':{},'direct_science_blocking_holds':{},'original_jobs_running':0,'original_reserved_s':0}
    write(admission.LEDGER, ledger)
    monkeypatch.setattr(supervisor, 'creation_ticks', lambda pid: 999)
    monkeypatch.setattr(supervisor.time, 'perf_counter', lambda: 101.)
    def no_job(*args):
        raise AssertionError('No Job in genuine pre-spawn refusal')
    monkeypatch.setattr(supervisor, 'OwnedJob', no_job)
    outer = supervisor.supervise(request['path'], root=tmp_path, entry=100.)
    assert outer['pre_spawn_refused'] and not outer['worker_started'] and not outer['job_created'] and outer['original_payload_accesses'] == 0
    assert (tmp_path/q['output']/'stdout.bin').read_bytes() == (tmp_path/q['output']/'stderr.bin').read_bytes() == b''
    with pytest.raises(ValueError, match='publish twice'):
        supervisor.supervise(request['path'], root=tmp_path, entry=100.)
    assert os.name == 'nt' and shutil.which('pwsh')
    grant_target = tmp_path/admission.GRANT['path']
    grant_target.parent.mkdir(parents=True, exist_ok=True)
    grant_target.write_bytes((admission.ROOT/admission.GRANT['path']).read_bytes())
    g = json.loads(grant_target.read_text(encoding='utf8'))
    ledger['startup_ray_original_grants']={g['grant_id']:{'grant':admission.GRANT,'source_freeze':g['source_freeze'],'inputs':g['inputs'],'local_debit_offset':165,'worker_max':1,'automatic_restarts':0}}
    write(admission.LEDGER, ledger)
    trace_name=admission.BASE+'/synthetic-geometry-return.json'
    write(trace_name, {'schema':'flylab.m3.direct-science.genuine-return.v1','phase':'FINAL','first_error':None,'dispatch_ms':1000,'first_final_ms':1125,'full_return_s':.125,'exit_code':1,'request':request,'source_freeze':freeze})
    source=(admission.ROOT/admission.BASE/'parent-direct-science-account.ps1').read_text(encoding='utf8')
    anchor="$taskRoot=[IO.Path]::GetFullPath('C:/Users/ckthd/Dev/Projects/fly_lab')"
    assert source.count(anchor) == 1
    script=tmp_path/'synthetic-geometry-account.ps1'
    script.write_text(source.replace(anchor,"$taskRoot=[IO.Path]::GetFullPath('"+tmp_path.as_posix().replace("'","''")+"')"),encoding='utf8')
    command=[shutil.which('pwsh'),'-NoProfile','-File',str(script),'-Action','commit','-TracePath',trace_name]
    first=subprocess.run(command,capture_output=True,timeout=10,check=False)
    assert first.returncode == 0, first.stderr
    after=json.loads((tmp_path/admission.LEDGER).read_text(encoding='utf8'))
    assert len(after['debits']) == 166 and after['debits'][-1]['actual_outer_full_return_s'] == .125
    cost=json.loads((tmp_path/after['debits'][-1]['descriptor']['path']).read_text(encoding='utf8'))
    assert cost['schema'] == admission.SCHEMA+'.full-return.v1' and cost['original_payload_accesses'] == 0
    duplicate=subprocess.run(command,capture_output=True,timeout=10,check=False)
    assert duplicate.returncode != 0 and b'Fresh once commit' in duplicate.stderr
    assert len(json.loads((tmp_path/admission.LEDGER).read_text(encoding='utf8'))['debits']) == 166
