"""Small original0 E358 boundary tests; no BANC arrays, torch, GPU or body."""

import copy
import subprocess
import time
from types import SimpleNamespace
import uuid
import sys

import pytest

from tools import verify_c_banc_m3_candidate_body_handoff as h


def generated_root(tmp_path):
    for name in (*h.OWNED, h.SUPERVISOR, h.OBSERVER):
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes((h.ROOT / name).read_bytes())
    (tmp_path / h.BASE).mkdir(exist_ok=True, parents=True)
    return tmp_path


def fixture_contract(root, *, attempt_id="test-fixture"):
    c = h.build_contract("fixture", attempt_id, mode="generated", root=root)
    h.durable(h.safe(c["paths"]["contract"], root), h.json_bytes(c), cap=256 << 10)
    return c


def rewrite_contract(c, root):
    # Only generated ephemeral fixtures; no historical evidence overwritten.
    path = h.safe(c["paths"]["contract"], root)
    path.write_bytes(h.json_bytes(c))


def good_monitor(*, closed=True):
    identity = {"physical_device_count": 1, "physical_index": 0, "mig_enabled": False,
                "uuid": "GPU-12345678-1234-1234-1234-123456789abc", "handle_identity": "fake0", "driver": "fixture"}
    return {"sample_count": 1, "sampler_errors": [], "owned_sampler_pids": [], "subprocess_calls": 0,
            "state": "CLOSED" if closed else "RUNNING", "owned_thread_alive": not closed,
            "shutdown_count": int(closed), "device_identity": identity,
            "latest_memory_bytes": {"uuid": identity["uuid"], "handle_identity": "fake0", "total": 16 << 30},
            "mapping": "single physical non-MIG index0", "CUDA_VISIBLE_DEVICES": None,
            "precontext_baseline_bytes": 10, "sampled_total_peak_bytes": 10,
            "sampled_additional_peak_bytes": 0, "quantization_allowance_bytes": 2 << 20}


def good_work(count=99):
    construction = {"body_constructors_requested": 1, "body_constructors_returned": 1,
                    "settling_physics_requested": 2000, "settling_physics_host_returned": 2000,
                    "neural_setup_requested": 320, "neural_setup_host_returned": 320,
                    "neural_capture_requested": 160, "neural_capture_recorded": 160,
                    "unknown_work": False, "device_confirmed": True}
    counters = {"controls_requested": count, "controls_host_returned": count, "controls_completed": count,
                "neural_requested": 20 * count, "neural_host_returned": 20 * count,
                "neural_device_confirmed": 20 * count, "physics_requested": 50 * count,
                "physics_host_returned": 50 * count, "physics_device_confirmed": 50 * count,
                "observer_samples": 50 * count, "completed_rows": count + 1,
                "neural_unreturned_unknown": False, "physics_unreturned_unknown": False}
    return counters, construction


def test_real_generated_cli_supervisor_publication_closed_return():
    # Retain the fresh real E358 fixture evidence; the surrounding suite cost
    # includes this subprocess, so ROOT does not debit it a second time.
    attempt_id = "generated-fixture-" + uuid.uuid4().hex[:16]
    c = fixture_contract(h.ROOT, attempt_id=attempt_id)
    began = time.perf_counter()
    completed = subprocess.run([str(h.ROOT / h.PYTHON), "-X", "utf8", "-B", h.OWNED[0],
                                "--supervise", c["paths"]["contract"]], cwd=h.ROOT,
                               capture_output=True, timeout=10, check=False)
    wall = time.perf_counter() - began
    assert completed.returncode == 0, completed.stderr.decode("utf8", errors="replace")
    actual_name = c["paths"]["actual_return"]
    h.durable(h.safe(actual_name), h.json_bytes({"actual_returned_exit_code": completed.returncode,
                                               "actual_outer_full_return_s": wall}), cap=1024)
    sealed = h.seal_full_return(c["paths"]["contract"], h.file_spec(actual_name))
    exit_name = c["paths"]["full_exit"]
    h.durable(h.safe(exit_name), h.json_bytes(sealed), cap=256 << 10)
    assert h.closed_exit(h.file_spec(exit_name))["qualified"] is True
    receipt = h.strict_json(h.safe(c["paths"]["pre_spawn"]).read_bytes())
    value = h.strict_json(h.read_spec(sealed["envelope"]))
    assert receipt["original_array_reads"] == receipt["scientific_native_imports"] == 0
    assert value["original_array_reads"] == 0 and value["actual_full_return_pending"] is True
    assert not value["natural_gait"] and not value["F1_F8_complete"]


def test_contract_missing_and_typed_budget_reject_before_array(tmp_path):
    root = generated_root(tmp_path)
    c = fixture_contract(root)
    original = copy.deepcopy(c)
    for key, value in (("cumulative_before_s", True), ("reserved_s", None), ("stage_used_s", -1),
                       ("cumulative_limit_s", 151), ("stage_limit_s", 1000)):
        bad = copy.deepcopy(original)
        bad[key] = value
        rewrite_contract(bad, root)
        with pytest.raises((ValueError, TypeError)):
            h.validate_contract(bad["paths"]["contract"], root=root)
    missing = copy.deepcopy(original)
    missing.pop("reserved_s")
    rewrite_contract(missing, root)
    with pytest.raises(ValueError, match="schema"):
        h.validate_contract(original["paths"]["contract"], root=root)
    assert not h.safe(original["paths"]["output"], root).exists()


def test_bad_role_path_argv_Q_and_source_rejected(tmp_path):
    root = generated_root(tmp_path)
    c = fixture_contract(root)
    mutations = []
    for key, value in (("role", "body"), ("adapter_qualification", {"path": "old.json"}),
                       ("attempt", True), ("pending_reservations", {"body": 129})):
        bad = copy.deepcopy(c)
        bad[key] = value
        mutations.append(bad)
    bad = copy.deepcopy(c)
    bad["commands"][0]["argv"][-1] = "../old-receipt.json"
    mutations.append(bad)
    bad = copy.deepcopy(c)
    bad["source_sha256"][h.OWNED[0]] = "0" * 64
    mutations.append(bad)
    bad = copy.deepcopy(c)
    bad["paths"]["output"] = h.BASE + "/e343-common-descent-direction"
    mutations.append(bad)
    for bad in mutations:
        rewrite_contract(bad, root)
        with pytest.raises(ValueError):
            h.validate_contract(c["paths"]["contract"], root=root)
    assert not h.safe(c["paths"]["output"], root).exists()


def test_registered_installed_source_alias_is_same_file_and_rejects_collision(tmp_path, monkeypatch):
    root = generated_root(tmp_path)
    relative = ".venv/Lib/site-packages/flygym/__init__.py"
    file = h.safe(relative, root)
    file.parent.mkdir(parents=True)
    file.write_bytes(b"literal generated source fixture\n")
    digest = h.sha(file.read_bytes())
    reg = {"science_protected_text_sha256": {}, "body_source_sha256": {file.as_posix(): digest}}
    for key in ("plan", "actual_Astra_review", "Astra_review_md", "ROOT_reflection",
                "science_registration", "science_qualification", "science_freeze",
                "body_inventory", "body_native_dependencies"):
        reg[key] = {"path": "literal-metadata.json", "bytes": 2, "sha256": "0" * 64}
    monkeypatch.setattr(h, "read_spec", lambda *args: b"{}")
    pins = h.closure_sources(reg, root)
    assert pins[file.as_posix()] == pins[relative] == digest
    assert set(pins) == {file.as_posix(), relative, "literal-metadata.json", h.REGISTRATION, *h.own_sources(root)}
    reg["science_protected_text_sha256"][relative] = "1" * 64
    with pytest.raises(ValueError, match="alias pin collision"):
        h.closure_sources(reg, root)


def test_stale_output_and_pre_spawn_receipt_replay(tmp_path):
    root = generated_root(tmp_path)
    c = fixture_contract(root)
    h.safe(c["paths"]["output"], root).mkdir()
    with pytest.raises(ValueError, match="fresh"):
        h.validate_contract(c["paths"]["contract"], root=root)
    # Mismatched receipts fail before numerical imports; fake metadata only.
    c2 = fixture_contract(root, attempt_id="other")
    bad = {"schema": "flylab.m3.e358.actual-pre-spawn.v1", "contract": h.file_spec(c["paths"]["contract"], root),
           "source_sha256": c2["source_sha256"], "role": "fixture", "attempt_id": "other", "attempt": 1,
           "reservation_held": True, "original_array_reads": 0, "scientific_native_imports": 0,
           "Popen_before_receipt": 0, "cumulative_before_s": c2["cumulative_before_s"],
           "reserved_s": c2["reserved_s"], "pending_reservations": {}}
    h.durable(h.safe(c2["paths"]["pre_spawn"], root), h.json_bytes(bad), cap=256 << 10)
    with pytest.raises(ValueError, match="authority"):
        h.pre_array(c2["paths"]["contract"], c2["paths"]["pre_spawn"], root=root)


def test_changed_selected_candidate_and_mismatched_attempt_rejected(monkeypatch):
    identity = {"worker_exit": {"path": "current.json"}, "selected_sha256": "1" * 64, "theta_sha256": "2" * 64}
    changed = {**identity, "selected_sha256": "3" * 64}
    monkeypatch.setattr(h, "closed_exit", lambda spec: {"envelope": {"path": "audit.json"}})
    monkeypatch.setattr(h, "read_spec", lambda spec: h.json_bytes({"candidate": changed}))
    monkeypatch.setattr(h, "candidate_raw", lambda *args, **kw: ({}, {}, identity))
    c = {"inputs": {"candidate_audit_exit": {}, "worker_exit": {}}}
    with pytest.raises(ValueError, match="same independently"):
        h.admitted_candidate(c, {}, None)
    changed["selected_sha256"] = identity["selected_sha256"]
    changed["worker_exit"] = {"path": "replayed-old.json"}
    with pytest.raises(ValueError, match="same independently"):
        h.admitted_candidate(c, {}, None)


def test_valid_receipt_mutations_reject_exact_typed_authority(tmp_path, monkeypatch):
    root = generated_root(tmp_path)
    c = fixture_contract(root)
    owner = SimpleNamespace(pid=123, create_time=lambda: 456.)
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=lambda pid: owner))
    h.pre_spawn(c["paths"]["contract"], time.perf_counter(), root=root)
    path = h.safe(c["paths"]["pre_spawn"], root)
    good = h.strict_json(path.read_bytes())
    assert h.pre_array(c["paths"]["contract"], c["paths"]["pre_spawn"], root=root)[0] == c
    for key, value in (("original_array_reads", False), ("scientific_native_imports", False),
                       ("Popen_before_receipt", False), ("attempt", True), ("cumulative_limit_s", 150.),
                       ("cumulative_before_s", 0.), ("stage_limit_s", True), ("stage_used_s", False),
                       ("reserved_s", 8.), ("original_calls", False), ("cpu_threads", True),
                       ("max_tree_rss_bytes", float(2 << 30)), ("retained_reserve_bytes", True)):
        bad = copy.deepcopy(good)
        bad[key] = value
        path.write_bytes(h.json_bytes(bad))
        with pytest.raises(ValueError, match="authority"):
            h.pre_array(c["paths"]["contract"], c["paths"]["pre_spawn"], root=root)
    bad = copy.deepcopy(good)
    bad["limits"]["outer_s"] = 8.
    path.write_bytes(h.json_bytes(bad))
    with pytest.raises(ValueError, match="authority"):
        h.pre_array(c["paths"]["contract"], c["paths"]["pre_spawn"], root=root)
    assert not h.safe(c["paths"]["output"], root).exists()


def test_NVML_empty_overlimit_error_and_late_close_rejected():
    limits = {"sample_cap": 132, "GPU_delta_plus_allowance_cap_bytes": 1 << 30}
    good = good_monitor()
    assert h.monitor_check(good, limits, closed=True)
    for key, value in (("sample_count", 0), ("sample_count", 133), ("sampler_errors", [{"message": "late"}]),
                       ("owned_thread_alive", True), ("shutdown_count", 0), ("subprocess_calls", 1)):
        bad = copy.deepcopy(good)
        bad[key] = value
        with pytest.raises(ValueError):
            h.monitor_check(bad, limits, closed=True)
    good["sampled_additional_peak_bytes"] = (1 << 30) - (2 << 20)
    good["sampled_total_peak_bytes"] = good["precontext_baseline_bytes"] + good["sampled_additional_peak_bytes"]
    assert h.monitor_check(good, limits, closed=True)
    good["sampled_additional_peak_bytes"] += 1
    good["sampled_total_peak_bytes"] += 1
    with pytest.raises(ValueError, match="delta"):
        h.monitor_check(good, limits, closed=True)


def test_real_direct_NVML_fake_reader_sticky_failure_shutdown_once():
    from tools.banc_m3_nvml_monitor import GPUMemoryWatch
    identity = good_monitor()["device_identity"]
    class Reader:
        shutdown_count = 0
        def init(self):
            return identity
        def read(self):
            return {"uuid": identity["uuid"], "handle_identity": "fake0", "total": 16 << 30, "used": 100, "free": (16 << 30) - 100}
        def shutdown(self):
            self.shutdown_count += 1
            raise RuntimeError("late fake shutdown")
    reader = Reader()
    watch = GPUMemoryWatch(index=0, interval_s=.5, reader=reader)
    assert watch.start()["sample_count"] == 1
    closed = watch.close()
    assert closed["sampler_errors"] and closed["state"] == "CLOSED"
    assert watch.close()["shutdown_count"] == reader.shutdown_count == 1
    with pytest.raises(ValueError):
        h.monitor_check(closed, {"sample_cap": 132, "GPU_delta_plus_allowance_cap_bytes": 1 << 30}, closed=True)


def test_constructor_unknown_and_control_work_reject():
    counters, construction = good_work()
    assert h.verify_work(counters, construction, 99)
    for key in ("neural_capture_recorded", "settling_physics_host_returned", "neural_setup_requested"):
        bad = copy.deepcopy(construction)
        bad[key] -= 1
        with pytest.raises(ValueError, match="constructor"):
            h.verify_work(counters, bad, 99)
    construction["unknown_work"] = True
    with pytest.raises(ValueError):
        h.verify_work(counters, construction, 99)
    counters, construction = good_work(300)
    assert h.verify_work(counters, construction, 300, full=True)
    counters["physics_unreturned_unknown"] = True
    with pytest.raises(ValueError, match="unknown"):
        h.verify_work(counters, construction, 300, full=True)


def test_owned_detach_then_close_preserves_first_secondary_and_notes():
    calls = []
    class Owner:
        def __init__(self, name):
            self.name = name
        def close(self):
            calls.append(self.name)
            error = RuntimeError(self.name)
            error.add_note("owned failure detail")
            raise error
    result = {"first_error": None, "secondary_errors": []}
    h.finish_owned(Owner("detach"), Owner("body"), result, SimpleNamespace(check=lambda: None))
    assert calls == ["detach", "body"]
    assert result["first_error"]["message"] == "detach"
    assert result["secondary_errors"][0]["message"] == "body"
    assert result["first_error"]["notes"] == ["owned failure detail"] and result["revoked"]


def test_cleanup_only_matching_observed_PID_creation_identity():
    kills = []
    class Gone(Exception):
        pass
    class Process:
        def __init__(self, pid):
            self.pid = pid
        def create_time(self):
            return {1: 11., 2: 99.}[self.pid]
        def kill(self):
            kills.append(self.pid)
    killed, errors = h.cleanup_matching({1: 11., 2: 22.}, SimpleNamespace(Process=Process, NoSuchProcess=Gone))
    assert kills == [1] and killed == [{"pid": 1, "create_time": 11.}] and errors == []


def test_absolute_ENTRY_is_not_reset_for_cleanup():
    timer = h.Clock(10., {"internal_s": 5, "science_s": 4, "outer_s": 8}, clock=lambda: 16.)
    with pytest.raises(TimeoutError):
        timer.check()
    result = {"first_error": None, "secondary_errors": []}
    h.finish_owned(None, SimpleNamespace(close=lambda: None), result, timer)
    assert result["first_error"]["type"] == "TimeoutError" and timer.entry == 10.


def test_owned_descendant_completion_reuse_unknown_and_timeout():
    state = {"now": 10., "unknown": False, "end": 10.015}
    class Gone(Exception):
        pass
    class Process:
        def __init__(self, pid):
            self.pid = pid
        def create_time(self):
            if self.pid == 2:
                return 99.  # Recorded PID2 identity has ended; replacement is unowned.
            if state["unknown"]:
                raise PermissionError("unreadable owned descendant")
            if state["now"] >= state["end"]:
                raise Gone()
            return 11.
    api = SimpleNamespace(Process=Process, NoSuchProcess=Gone)
    timer = h.Clock(10., {"outer_s": 1}, clock=lambda: state["now"])
    def sleep(amount):
        state["now"] += amount
    matching, unknown, errors = h.bounded_owned_completion({1: 11., 2: 22.}, timer, api, sleep=sleep)
    assert matching == unknown == {} and errors == []
    state.update(now=10., unknown=True)
    matching, unknown, errors = h.bounded_owned_completion({1: 11.}, timer, api, sleep=sleep)
    assert matching == {} and unknown == {1: 11.} and errors[0]["type"] == "PermissionError"
    state.update(now=10., unknown=False, end=99.)
    matching, unknown, errors = h.bounded_owned_completion({1: 11., 2: 22.}, timer, api, sleep=sleep)
    assert matching == {1: 11.} and unknown == {} and errors[-1]["type"] == "TimeoutError"
    assert state["now"] <= 10.9 and timer.entry == 10.


def test_late_outer_publication_full_return_preserves_science_errors(tmp_path, monkeypatch):
    root = generated_root(tmp_path)
    c = fixture_contract(root)
    spec = h.file_spec(c["paths"]["contract"], root)
    folder = h.safe(c["paths"]["output"], root)
    folder.mkdir()
    science = h.error_record(RuntimeError("earliest science failure"), "candidate-backward")
    close = h.error_record(RuntimeError("later owned close failure"), "native-close")
    value = {"run_valid": False, "revoked": True, "first_error": science, "secondary_errors": [close]}
    h.durable(folder / "envelope.json", h.json_bytes(value), cap=4096)
    outer_error = h.error_record(RuntimeError("actual unchanged supervisor natural job exit"), "outer")
    record = {"contract": spec, "envelope": h.file_spec(c["paths"]["output"] + "/envelope.json", root),
              "natural_closed": True, "all_owned_ended": True, "known_identity_cleanup": [],
              "first_error": outer_error, "secondary_errors": [], "reservation_released": True}
    state = {"now": 0.}
    timer = h.Clock(0., c["limits"], clock=lambda: state["now"])
    original_durable = h.durable
    def late(path, payload, **kwargs):
        if path == h.safe(c["paths"]["outer"], root):
            def checked():
                kwargs["check"]()
            checked()
            original_durable(path, payload, cap=kwargs["cap"])
            state["now"] = 9.
            checked()
        else:
            original_durable(path, payload, **kwargs)
    monkeypatch.setattr(h, "durable", late)
    revoked = h.publish_outer(c, record, timer, root=root)
    assert revoked["first_error"] == outer_error and revoked["secondary_errors"][0]["type"] == "TimeoutError"
    assert h.safe(c["paths"]["outer"], root).exists() and h.safe(c["paths"]["outer_revocation"], root).exists()
    actual = {"actual_returned_exit_code": 1, "actual_outer_full_return_s": 9.}
    original_durable(h.safe(c["paths"]["actual_return"], root), h.json_bytes(actual), cap=4096)
    seal = h.seal_full_return(c["paths"]["contract"], h.file_spec(c["paths"]["actual_return"], root), root=root)
    assert seal["qualified"] is False and seal["charged_once"] is True and seal["all_owned_ended"] is True
    assert seal["first_error"] == science and seal["secondary_errors"][0] == close
    assert outer_error in seal["secondary_errors"] and seal["outer_revocation"] is not None
    assert seal["natural_closed"] is False
    assert seal["cumulative_after_s"] == c["cumulative_before_s"] + 9.
    assert timer.entry == 0.


def test_late_publication_revokes_and_retains_provisional(tmp_path, monkeypatch):
    root = generated_root(tmp_path)
    c = fixture_contract(root)
    spec = h.file_spec(c["paths"]["contract"], root)
    timer = h.Clock(time.perf_counter(), c["limits"])
    monkeypatch.setattr(h, "pre_array", lambda *args: (c, spec, spec, None, timer))
    original_put, original_retained = h.put, h.retained
    state = {"late": False}
    def write(c, name, value, timer, **kwargs):
        returned = original_put(c, name, value, timer, root=root, **kwargs)
        if name == "envelope.json":
            state["late"] = True
        return returned
    def check(stage="internal"):
        if state["late"]:
            raise TimeoutError("late publication deadline")
    monkeypatch.setattr(h, "put", write)
    monkeypatch.setattr(h, "retained", lambda c, root_override=root: original_retained(c, root_override))
    monkeypatch.setattr(timer, "check", check)
    assert h.child(c["paths"]["contract"], c["paths"]["pre_spawn"]) == 1
    folder = h.safe(c["paths"]["output"], root)
    provisional = h.strict_json((folder / "envelope.json").read_bytes())
    revoked = h.strict_json((folder / "revocation.json").read_bytes())
    assert provisional["run_valid"] is True and revoked["run_valid"] is False
    assert revoked["first_error"]["type"] == "TimeoutError"


def test_strict_finite_JSON_and_numpy_scalar_tuple_normalization():
    class Scalar:
        __module__ = "numpy"
        def item(self):
            return 3.
    assert h.strict_json(h.json_bytes({"value": (Scalar(), 4)})) == {"value": [3., 4]}
    for raw in (b'{"a":1,"a":2}', b'{"v":NaN}', b'{"v":1e999}'):
        with pytest.raises(ValueError):
            h.strict_json(raw)
    with pytest.raises(ValueError):
        h.json_bytes({"v": float("nan")})


def test_R_screen_ignores_old_smooth_shape_and_requires_all_six():
    metrics = {"summary": {"forward_mm": 1.6790085779712631, "slip_mm_s": 1.8026436968249955,
                           "longest_continuous_forward_bout_s": .3,
                           "complete_cycles": {leg: 1 for leg in ("lf", "lm", "lh", "rf", "rm", "rh")},
                           "health": {"healthy": True}}, "smooth": {"training_loss": 1000}, "shape": {"shape_loss": 1000}}
    assert h.screen(metrics, {"passed": True})[1]
    metrics["summary"]["complete_cycles"]["rh"] = 0
    assert not h.screen(metrics, {"passed": True})[1]


def test_all_current_pytest_phases_required_no_hidden_filter(monkeypatch):
    expected = ["test_a", "test_b"]
    monkeypatch.setattr(h, "test_functions", lambda root: expected)
    monkeypatch.setattr(h, "own_sources", lambda root: {"test": "source"})
    receipt = {"schema": "flylab.m3.e358.complete-pytest-receipt.v1", "source_sha256": {"test": "source"},
               "expected_functions": expected, "collected_functions": expected, "exit_code": 0,
               "collection_errors": [], "original_calls": 0, "original_array_reads": 0, "complete_current_cases": True,
               "phases": {v: {k: "passed" for k in ("setup", "call", "teardown")} for v in expected}}
    assert h.validate_case_receipt(receipt)
    receipt["phases"]["test_b"].pop("teardown")
    with pytest.raises(ValueError, match="setup/call/teardown"):
        h.validate_case_receipt(receipt)
