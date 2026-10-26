"""Small generated contact mathematics/owner fixtures; no Original numerical input."""
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from tools import banc_m3_contact_learning as s
from tools import banc_m3_contact_learning_admission as gate


def contract():
    return json.loads((gate.ROOT / gate.STAGE_SPEC).read_text(encoding="utf8"))


def frozen():
    j = np.zeros((4, 6, 3, 42), np.float64)
    for window in range(4):
        for leg in range(6):
            j[window, leg, :, 7 * leg:7 * leg + 3] = np.eye(3)
    loaded = np.zeros((4, 6), bool)
    for window, count in enumerate((1, 2, 4, 5)):
        loaded[window, :count] = True
    return {"J": j, "qpos": np.zeros((4, 42)), "anterior": np.tile([1., 0., 0.], (4, 1)),
            "loaded": loaded, "unloaded": ~loaded, "limits": np.tile([-.2, .2], (42, 1)),
            "limited": np.ones(42, bool), "clamp_enabled": True}


def test_math_global_loss_unequal_window_masks_analytic_seed_and_keep_zero():
    import torch
    f, c = frozen(), contract()["loss"]
    nominal = np.full((4, 4, 42), .01, np.float64)
    d0 = np.stack([s.projection(nominal[i, 0], f, i, np)[0] for i in range(4)])
    baseline = s.evaluate(nominal, f, d0, c, np)
    assert baseline["keep"] == 0 and baseline["unloaded_RMS_mm"] == 0
    assert baseline["stance"] == pytest.approx(2 * .01**2 / c["s_mm"]**2)
    proposed = nominal.copy()
    proposed[:, 0] *= .8
    proposed[:, 1:] = 123.  # other controls never enter the loss
    independent = torch.tensor(proposed, dtype=torch.float64, requires_grad=True)
    loss = torch.zeros((), dtype=torch.float64)
    for window in range(4):
        d = torch.einsum("lci,i->lc", torch.tensor(f["J"][window]), independent[window, 0])
        loaded, unloaded = torch.tensor(f["loaded"][window]), torch.tensor(f["unloaded"][window])
        loss = loss + (torch.relu(d[loaded, 0]).square().sum() + torch.relu(d[loaded, 2]).square().sum()) / (12 * .02**2)
        loss = loss + .25 * (d[unloaded] - torch.tensor(d0[window][f["unloaded"][window]])).square().sum() / (36 * .02**2)
    loss.backward()
    calculated = s.evaluate(proposed, f, d0, c, np)
    assert calculated["total"] == pytest.approx(loss.item())
    for window in range(4):
        _, seed = s.window_loss(proposed[window], f, d0, window, c, np)
        assert np.allclose(seed, independent.grad.numpy()[window], atol=1e-12, rtol=0)
        assert np.array_equal(seed[1:], np.zeros((3, 42)))
        _, baseline_seed = s.window_loss(nominal[window], f, d0, window, c, np)
        assert np.all(baseline_seed[0, np.flatnonzero(np.repeat(f["unloaded"][window], 7))] == 0)


def test_math_clamp_boundary_disabled_relu_zero_and_nonfinite_refusal():
    f = frozen()
    q = np.zeros(42)
    q[:6] = [-.3, -.2, -.1, 0., .2, .3]
    _, derivative, _ = s.projection(q, f, 0, np)
    assert np.array_equal(derivative[:6], [0, 0, 1, 1, 0, 0])
    f["clamp_enabled"] = False
    d, derivative, _ = s.projection(q, f, 0, np)
    assert np.array_equal(derivative, np.ones(42)) and d[0, 0] == -.3
    q = np.zeros((4, 42))
    _, seed = s.window_loss(q, f, np.zeros((4, 6, 3)), 0, contract()["loss"], np)
    assert np.array_equal(seed, np.zeros((4, 42)))
    q[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        s.window_loss(q, f, np.zeros((4, 6, 3)), 0, contract()["loss"], np)


def test_math_fixed_probe_eligibility_rounding_and_negative_retention():
    from tools import verify_c_banc_m3_live_replay as replay
    c = contract()["loss"]
    baseline = {"total": 2., "stance": 2.}
    current = {"total": 1., "stance": .9, "unloaded_RMS_mm": .005}
    assert s.finite_probe_pass(baseline, current, {"total": 9.}, True)
    assert not s.finite_probe_pass(baseline, current, {"total": 9.}, False)
    assert s.eligible(baseline, current, {"passed": True}, True, True, c)
    for key in ("total", "stance", "unloaded_RMS_mm"):
        wrong = dict(current, **{key: 3.})
        assert not s.eligible(baseline, wrong, {"passed": True}, True, True, c)
    source = np.asarray([.25, -.125, 3.], np.float32)
    selected = (source[:2] * np.float32(1.0001)).astype(np.float32)
    deployed = replay.selected_deployment(source, np.arange(2, dtype=np.int64), selected, np)
    assert np.array_equal(deployed[:2], selected) and deployed[2] == source[2]
    assert not np.array_equal(deployed[:2], selected * np.float32(1.0001))  # no double theta
    assert np.array_equal((np.ones(2) + 1e-10).astype(np.float32), np.ones(2, np.float32))  # honest rounded no-op


def test_math_actual_generated_checkpoint_first_seed_and_partial_receipt():
    import torch

    from tests.test_c_banc_checkpointed_rollout import case
    from tools.verify_c_banc_m3_coordination_parameter_direction import receipt_dict
    torch.set_num_threads(1)
    rollout, inputs = case(4, steps=20, checkpoint=1)
    # Only efficacy is a new independent leaf; frozen sensory/state are constants.
    values = [v.detach().clone() for v in inputs]
    values[5].requires_grad_(True)
    output = rollout.run(*values)
    q = output["q"].detach().cpu().numpy()
    f = frozen()
    f["qpos"][:] = q[0] - .01  # nonzero stance seed through actual generated checkpoint backward
    d0 = np.stack([s.projection(q[0], f, j, np)[0] for j in range(4)])
    _, seed = s.window_loss(q, f, d0, 0, contract()["loss"], np)
    output["q"].backward(torch.tensor(seed))
    output["receipt"].confirm()
    r = receipt_dict(output["receipt"])
    assert r["forward_controls_returned"] == r["recompute_controls_returned"] == r["backward_blocks_returned"] == 4
    assert r["forward_neural_ticks_returned"] == r["recompute_neural_ticks_returned"] == r["backward_block_ticks_returned"] == 80
    assert values[5].grad is not None and torch.isfinite(values[5].grad).all() and torch.any(values[5].grad != 0)
    rollout, inputs = case(4, steps=20, checkpoint=1)
    rollout.neural.fault["at"] = 2
    with pytest.raises(Exception) as failed:
        rollout.run(*inputs)
    receipt = failed.value.receipt
    receipt.confirm()
    r = receipt_dict(receipt)
    assert r["forward_neural_ticks_requested"] == 2 and r["forward_neural_ticks_returned"] == 1
    assert r["forward_controls_requested"] == 1 and r["forward_controls_returned"] == 0
    assert r["confirmed"]["forward_controls"] == 0


def test_math_header_member_plan_large_startup_and_zero_mocap():
    assert s.member_bytes((99, 158706), "<f4", 64 << 20) == 62847576
    assert s.member_bytes((8, 0, 3), "<f8", 64 << 20) == 0
    for shape, dtype in (((99, 200000), "<f4"), ((-1,), "<f8"), ((3,), "object")):
        with pytest.raises(ValueError):
            s.member_bytes(shape, dtype, 64 << 20)


def test_math_independent_qualified_saved_projection_and_drift_refusal():
    f, q = frozen(), np.full((4, 42), .01)
    measured = np.stack([s.projection(q[j], f, j, np)[0] for j in range(4)])
    rows = []
    for j, c in enumerate(s.STARTS):
        for leg, name in enumerate(("lf", "lm", "lh", "rf", "rm", "rh")):
            rows.append({"anchor_control": c, "leg": name, "command_status": "VERIFIED_ACTUAL_HELD_CLAMP",
                         "category": "loaded_loaded" if f["loaded"][j, leg] else "unloaded_unloaded",
                         "J_command": {"start_thorax_AP_mm": float(measured[j, leg, 0]), "world_vertical_mm": float(measured[j, leg, 2])}})
    analysis = {"qualified": True, "common_values_exact": True, "local_table": rows}
    assert s.qualified_projection(measured, f, analysis, np)["max_abs_mm"] == 0
    wrong = copy.deepcopy(analysis)
    wrong["local_table"][7]["J_command"]["world_vertical_mm"] += 1e-6
    with pytest.raises(ValueError, match="matches qualified"):
        s.qualified_projection(measured, f, wrong, np)
    wrong = copy.deepcopy(analysis)
    wrong["local_table"][7]["anchor_control"] += 1
    with pytest.raises(ValueError, match="row identity"):
        s.qualified_projection(measured, f, wrong, np)


def test_math_returned_window_seed_gradient_and_prefix_survive_later_failure():
    import torch

    from tests.test_c_banc_checkpointed_rollout import case
    from tools.verify_c_banc_m3_coordination_parameter_direction import receipt_dict
    raw, report = {}, {}
    phases = s.PhaseBuffers(raw, report, np)
    for key, shape in (("q", (4, 4, 42)), ("tendon", (4, 4, 6)), ("seed", (4, 4, 42)), ("gradient", (4, 2))):
        phases.allocate(key, shape)
    rollout, inputs = case(4, steps=20, checkpoint=1)
    values = [v.detach().clone() for v in inputs]
    values[5].requires_grad_(True)
    out = rollout.run(*values)
    q, tendon = out["q"].detach().numpy().copy(), out["tendon"].detach().numpy().copy()
    phases.returned("q", 0, q)
    phases.returned("tendon", 0, tendon)
    seed = np.zeros_like(q)
    seed[0] = 1.
    phases.returned("seed", 0, seed)
    out["q"].backward(torch.tensor(seed))
    phases.returned("gradient", 0, values[5].grad.detach().numpy().copy())
    out["receipt"].confirm()
    completed = receipt_dict(out["receipt"])
    rollout.neural.fault["at"] = len(rollout.neural.trace) + 1
    with pytest.raises(RuntimeError, match="generated neural fault") as failure:
        rollout.run(*values)
    partial = receipt_dict(failure.value.receipt)
    assert completed["forward_controls_returned"] == 4 and partial["forward_controls_returned"] == 0
    assert np.array_equal(raw["q"][0], q) and np.array_equal(raw["seed"][0], seed)
    assert np.array_equal(raw["gradient"][0], values[5].grad.detach().numpy())
    assert all(np.array_equal(raw[k + ".valid"], [True, False, False, False]) for k in ("q", "tendon", "seed", "gradient"))
    rollout, inputs = case(1, steps=20, checkpoint=1)
    phases.allocate("startup_prefix", (100, 42))
    phases.returned("startup_prefix", 0, np.zeros(42))  # known generated control0
    prefix = rollout.run(*inputs)
    phases.returned("startup_prefix", 1, prefix["q"].detach().numpy()[0])
    rollout.neural.fault["at"] = len(rollout.neural.trace) + 1
    with pytest.raises(RuntimeError, match="generated neural fault"):
        rollout.run(*inputs)
    assert raw["startup_prefix.valid"].sum() == 2 and report["retained_phase_fields"]["startup_prefix"]["returned"] == 2
    assert len(raw) < 256 and all(v.nbytes <= 64 << 20 for v in raw.values())


def test_math_minimal_saved_E318_endpoint_native_precision_and_mixed_headers(tmp_path):
    n = 2
    chunk = {"control_tick": np.arange(76, 101, dtype=np.int64), "network_tick": np.arange(76, 101, dtype=np.int64) * 20,
             "rate": np.arange(25 * n, dtype=np.float32).reshape(25, n), "adaptation": np.zeros((25, n), np.float32)}
    recruitment = {"control_tick": np.arange(301, dtype=np.int64), "network_tick": np.arange(301, dtype=np.int64) * 20,
                   "bank": np.zeros((301, 1, 3)), "original_activation": np.zeros((301, 2)),
                   "complete": np.ones(301, bool), "bank_available": np.ones(301, bool), "original_activation_available": np.ones(301, bool)}
    raw = {"neutral_endpoint-rate": chunk["rate"][23].copy(), "neutral_endpoint-adaptation": chunk["adaptation"][23].copy(),
           "neutral_endpoint-bank": recruitment["bank"][99].copy(), "neutral_endpoint-original_activation": recruitment["original_activation"][99].copy()}
    assert all(v["passed"] for v in s.compare_E318_endpoint(raw, chunk, recruitment, np).values())
    wrong = {k: v.copy() for k, v in raw.items()}
    wrong["neutral_endpoint-rate"][0] = np.nextafter(wrong["neutral_endpoint-rate"][0], np.float32(np.inf))
    with pytest.raises(ValueError, match="E318 endpoint rate"):
        s.compare_E318_endpoint(wrong, chunk, recruitment, np)
    raw["neutral_endpoint-bank"][0, 0] = 1e-11
    assert s.compare_E318_endpoint(raw, chunk, recruitment, np)["bank"]["passed"]
    raw["neutral_endpoint-bank"][0, 0] = 1e-9
    with pytest.raises(ValueError, match="E318 endpoint bank"):
        s.compare_E318_endpoint(raw, chunk, recruitment, np)
    archive = tmp_path / "mixed.npz"
    np.savez(archive, rate=np.ones(2, np.float32), palette=np.asarray([0, 65535], np.uint16), ids=np.asarray([2**40], np.uint64), flag=np.ones(2, bool))
    class Admission:
        root = tmp_path
        limits: ClassVar = {"input_file_cap_bytes": 1 << 20, "member_cap_bytes": 64 << 20, "selected_expanded_cap_bytes": 512 << 20}
        def check(self, label):
            return True
        def fence(self, label):
            return True
    saved = s.Saved(Admission(), np)
    spec = {"path": archive.name, "bytes": archive.stat().st_size, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
    planned = saved.arrays(spec, plan_only=True)
    assert saved.actual_bytes == 0 and planned["expanded_bytes"] == 22
    selected = saved.arrays(spec)
    assert selected["palette"].dtype == np.uint16 and selected["ids"].dtype == np.uint64 and saved.actual_bytes == 22


def test_math_maximum_receipts_small_report_and_late_manifest_failure(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    from tools.analyze_c_banc_m3_contact_learning import audit_backend_guard
    from tools.banc_m3_checkpointed_rollout import RolloutReceipt
    from tools.verify_c_banc_m3_contact_learning import publish
    from tools.verify_c_banc_m3_coordination_parameter_direction import receipt_dict
    from tools.verify_c_banc_m3_startup_ray import Publisher
    assert json.loads(s.wire({"n": np.int64(7), "x": np.float64(.5), "ok": np.bool_(True)}, s.REPORT_CAP)) == {"n": 7, "x": .5, "ok": True}
    with pytest.raises(ValueError):
        s.wire({"nonfinite": np.float64(np.nan)}, s.REPORT_CAP)
    with monkeypatch.context() as modules:
        modules.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False)))
        audit_backend_guard("contact_body_audit")
        with pytest.raises(ValueError, match="CPU tensor"):
            audit_backend_guard("contact_candidate_audit")
        modules.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True)))
        with pytest.raises(ValueError, match="no CUDA"):
            audit_backend_guard("contact_body_audit")
    # Synthetic maximum-size metadata, not executed1297 Original controls.
    receipts = {}
    controls = [1] * 99 + [4] * 4 + [4] * 8 + ([4] * 8 + [1] * 198) * 5
    for i, count in enumerate(controls):
        r = RolloutReceipt("cpu")
        r.forward_controls_requested = r.forward_controls_returned = count
        r.forward_neural_ticks_requested = r.forward_neural_ticks_returned = count * 20
        if 99 <= i < 103:
            r.recompute_controls_requested = r.recompute_controls_returned = r.backward_blocks_requested = r.backward_blocks_returned = count
            r.recompute_neural_ticks_requested = r.recompute_neural_ticks_returned = r.backward_block_ticks_returned = count * 20
        r.confirm()
        receipts["longest-phase-label-deployed-startup-" + str(i)] = receipt_dict(r)
    work = s.work_summary(receipts, contract(), evaluated=5, probes=2, complete=True)
    report = {"run_valid": True, "first_error": None, "secondary_errors": [], "work": work, "receipts": receipts,
              "candidate_present": False, "classification": "generated maximum metadata", "receipt_record_count": len(receipts)}
    branch = {"clip_derivative": [1] * 42, "clip_lower_headroom": [1e308] * 42, "clip_upper_headroom": [-1e308] * 42,
              "AP_active": [True] * 6, "Z_active": [False] * 6, "SSP_internal_branches": "NOT_OBSERVED"}
    report["alphas"] = [{"index": i, "loss": {"windows": [{"branches": branch}] * 4}, "eligible": False} for i in range(5)]
    report["retained_phase_fields"] = {"longest-retained-phase-key-" + str(i): {"returned": 100, "capacity": 100, "valid_bitmap": "bounded-validity", "invalid_storage_is_evidence": False} for i in range(64)}
    assert work["forward_controls_returned"] == 1297 and len(s.wire(receipts, s.RECEIPTS_CAP)) <= s.RECEIPTS_CAP
    assert len(s.wire({k: v for k, v in report.items() if k != "receipts"}, s.REPORT_CAP)) < s.REPORT_CAP
    class Admission:
        root, output = tmp_path, tmp_path / "result"
        limits: ClassVar = {"output_cap_bytes": 16 << 20}
        role = "contact_direction"
        request: ClassVar = {"source_freeze": {"fixture": True}, "contract": {"fixture": True}}
        request_spec: ClassVar = {"fixture": True}
        def fence(self, label="publication"):
            return True
    admission, raw = Admission(), {"returned": np.asarray([3.], np.float64)}
    publisher = Publisher(admission)
    original_write = publisher.write
    def injected(name, payload):
        if name == "contact-manifest.json":
            raise OSError("injected terminal manifest failure")
        return original_write(name, payload)
    monkeypatch.setattr(publisher, "write", injected)
    artifacts = publish(publisher, raw, report, [], admission)
    before = (admission.output / "contact-report.json").read_bytes()
    terminal = json.loads((admission.output / "contact-terminal-publication-failure.json").read_bytes())
    assert report["run_valid"] is False and terminal["first_error"]["stage"] == "manifest-publication"
    assert json.loads(before)["run_valid"] is True and "receipts" not in json.loads(before)
    assert len(before) == artifacts["report"]["bytes"] <= s.REPORT_CAP and "manifest-publication" in capsys.readouterr().err
    assert (admission.output / "contact-report.json").read_bytes() == before
    assert not s.secant_diagnostic(-1., 100.)["secant_agreement"] and s.secant_diagnostic(-1., 100.)["secant_agreement_is_gate"] is False


def test_gate_existing_continuous_arm_pre_construct_dispatch_and_nonphysical_refusal(monkeypatch):
    from tools import verify_c_banc_m3_common_descent_body as body
    from tools import verify_c_banc_m3_startup_ray as ray
    timer = object.__new__(gate.Admission)
    timer.entry, timer._deadline_s, timer.role, timer.stage = 100., 270, "contact_candidate_body", "pre_array"
    monkeypatch.setattr(gate.time, "perf_counter", lambda: 101.)
    cold, constructed = [], []
    def injected_cold(label):
        cold.append(label)  # only the complete cold-gate dependency is synthetic
        return True
    monkeypatch.setattr(timer, "_cold_fence", injected_cold)
    class Factory:
        def construct(self, data, construction, check):
            assert cold[-1] == "pre_construct"
            constructed.append(True)
            raise RuntimeError("generated sentinel after actual cold-boundary dispatch")
    class Publisher:
        def json(self, name, value):
            return {"fixture": name}
    monkeypatch.setattr(body, "OriginalFactory", Factory)
    result = ray.continuous_arm({}, "candidate", {}, {}, timer, Publisher(), None)
    assert constructed == [True] and cold == ["pre_construct"] and "generated sentinel" in result["first_error"]["message"]
    timer.role = "contact_direction"
    with pytest.raises(ValueError, match="physical role only"):
        timer.fence("pre_construct")
    assert cold == ["pre_construct"] and constructed == [True]


def test_gate_deadline_lazy_CPU_environment_and_immutable215_history(monkeypatch):
    from tools import banc_m3_startup_ray_supervisor as supervisor
    code = "import sys,os,json;from tools import banc_m3_contact_learning,banc_m3_contact_learning_admission,verify_c_banc_m3_contact_learning,analyze_c_banc_m3_contact_learning;assert not any(k in sys.modules for k in ('torch','numpy','mujoco','flygym'));print(json.dumps([os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')]))"
    env = supervisor.worker_environment("contact_candidate_audit", {"generated": True}, sys.executable)
    observed = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, timeout=5, check=False)
    assert observed.returncode == 0 and json.loads(observed.stdout) == ["1"] * 4
    touched = []
    monkeypatch.setattr(gate.time, "perf_counter", lambda: 1250.)
    monkeypatch.setattr(gate, "spec_for", lambda *a: touched.append(a))
    with pytest.raises(TimeoutError):
        gate.Admission("not-opened.json", "pre_spawn", entry=100.)
    assert touched == []
    def read(spec, cap=1048576):
        p = gate.path(spec["path"], gate.ROOT)
        assert p.suffix == ".json" and p.stat().st_size <= cap
        raw = p.read_bytes()
        assert len(raw) == spec["bytes"] and hashlib.sha256(raw).hexdigest() == spec["sha256"]
        return json.loads(raw)
    roster, grant = read(gate.ROSTER), read(gate.GRANT)
    before = read(roster["closed_215"])
    spec = {"path": gate.STAGE_SPEC, "bytes": (gate.ROOT/gate.STAGE_SPEC).stat().st_size, "sha256": hashlib.sha256((gate.ROOT/gate.STAGE_SPEC).read_bytes()).hexdigest()}
    q = {"role": gate.ROLES[0], "contract": spec, "artifacts": {}}
    from tools.banc_m3_saved_original_admission import closed_cost
    def closed(cost, debit):
        return closed_cost(cost, debit, gate.ROOT)
    def archive(item):
        return Path(item["archive_path"]).read_bytes()
    original = copy.deepcopy(before)
    assert gate.validate_history(before, grant, roster, q, read, closed, archive) > 0 and before == original
    before["debits"][0]["cost_id"] = "tampered-old-debit"
    with pytest.raises(ValueError, match="unchanged"):
        gate.validate_history(before, grant, roster, q, read, closed, archive)


@pytest.mark.parametrize("negative", [None, "float_count", "missing_confirm", "unknown_requested", "double_backward_ticks"])
def test_gate_actual_production_Native_Int64_positive_and_negatives(tmp_path, negative):
    evaluated, probes = 2, 2
    controls = 115 + 16 * probes + 230 * evaluated
    work = {"confirmed": {}, "body_constructors": 0, "physics_steps": 0, "warmup_ticks": 0, "capture_ticks": 0,
            "backward_block_ticks_are_additional_forward": False, "backward_block_ticks_returned": 320}
    for key, count in (("forward_controls", controls), ("forward_neural_ticks", controls*20), ("recompute_controls", 16), ("recompute_neural_ticks", 320), ("backward_blocks", 16)):
        work[key+"_requested"] = work[key+"_returned"] = work["confirmed"][key] = count
    work["confirmed"]["backward_block_ticks"] = 320
    report = {"run_valid": True, "first_error": None, "secondary_errors": [], "evaluated_alphas": evaluated, "probe_evaluations": probes, "work": work}
    receipt = {k: v for k, v in work.items() if k.endswith(("_requested", "_returned"))}
    receipt.update(confirmed=copy.deepcopy(work["confirmed"]), confirm_requested=1, confirm_returned=1)
    complete = {"synthetic aggregate for production typed gate only": receipt}
    report["receipt_record_count"] = 1
    if negative == "float_count":
        work["forward_controls_returned"] = float(controls)
    elif negative == "missing_confirm":
        del work["confirmed"]["forward_controls"]
    elif negative == "unknown_requested":
        work["forward_controls_requested"] += 1
    elif negative == "double_backward_ticks":
        work["backward_block_ticks_are_additional_forward"] = True
    payload = tmp_path / "report.json"
    payload.write_text(json.dumps(report), encoding="utf8")
    receipt_payload = tmp_path / "receipts.json"
    receipt_payload.write_text(json.dumps(complete), encoding="utf8")
    source = (gate.ROOT/gate.BASE/"parent-direct-science-account.ps1").read_text(encoding="utf8")
    # Exact production Need/Same functions, no copied permissive comparator.
    definitions = source[source.index("function Need("):source.index("function Write-New(")]
    helper = (gate.ROOT/gate.BASE/"parent-contact-learning-contract-native.ps1").as_posix().replace("'", "''")
    stage = (gate.ROOT/gate.STAGE_SPEC).as_posix().replace("'", "''")
    script = tmp_path / "work-fixture.ps1"
    script.write_text("$ErrorActionPreference='Stop'\n"+definitions+"\n. '"+helper+"'\n$c=Get-Content -Raw '"+stage+"'|ConvertFrom-Json -AsHashtable\n$r=Get-Content -Raw '"+payload.as_posix().replace("'", "''")+"'|ConvertFrom-Json -AsHashtable\n$u=Get-Content -Raw '"+receipt_payload.as_posix().replace("'", "''")+"'|ConvertFrom-Json -AsHashtable\nNeed ($r.evaluated_alphas -is [long]) 'Actual parsed Int64'\n$null=Contact-Work $r $c $u\n", encoding="utf8")
    observed = subprocess.run([shutil.which("pwsh"), "-NoProfile", "-File", str(script)], capture_output=True, timeout=5, check=False)
    assert (observed.returncode == 0) == (negative is None), observed.stderr


@pytest.mark.parametrize('role', (gate.ROLES[0], gate.ROLES[3]))
def test_gate_contact_pre_spawn_refusal_and_temp_native_once_only(tmp_path, monkeypatch, role):
    from tools import banc_m3_startup_ray_supervisor as supervisor
    def write(name, value):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value, allow_nan=False), encoding='utf8')
        return {'path': name, 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    stage = contract()
    call, limits = stage["calls"][gate.ROLES.index(role)], stage["limits"][role]
    stage_spec = write(gate.STAGE_SPEC, stage)
    science = write(gate.BASE + '/synthetic-geometry-science.json', {'schema': gate.SCHEMA+'.request.v1', 'mode': 'original', 'role': role, 'limits': {}, 'contract': stage_spec})
    freeze = write(gate.BASE + '/synthetic-geometry-freeze.json', {'fixture': 'generated metadata refusal'})
    q = {'schema': 'flylab.m3.direct-science.supervision.v1', 'mode': 'original', 'role': role, 'call_id': call, 'science_request': science,
         'source_freeze': freeze, 'grant': gate.GRANT, 'ledger_path': gate.LEDGER, 'shared_original_lock_path': gate.SHARED_LOCK,
         'output': gate.BASE+'/parent-'+call+'/result-supervision', 'limits': {k: limits[k] for k in ('outer_s','child_s','cleanup_s','host_bytes','output_cap_bytes')}}
    request = write(gate.BASE+'/synthetic-geometry-owner.json', q)
    debits = [{'cost_id': 'synthetic-old-'+str(i), 'mode': 'generated'} for i in range(165)]
    before = write(gate.BASE+'/synthetic-geometry-before.json', {'debits': debits})
    policy = write(gate.BASE+'/synthetic-geometry-policy.json', {'schema':'flylab.m3.direct-science.policy.v1','generated_local_limit_s':300,'original_authorized':False,'before_ledger':before,'local_debit_offset':165})
    ledger = {'debits':debits,'direct_science_policy':policy,'direct_science_calls':{call:{'request':request,'state':'admitted','mode':'original','reserved_s':limits['outer_s']}},
              'startup_ray_original_workers':{},'startup_ray_original_reservations':{},'startup_ray_original_pending_actuals':{},'startup_ray_original_blocking_holds':{},
              'direct_science_pending_actuals':{},'direct_science_blocking_holds':{},'original_jobs_running':0,'original_reserved_s':0}
    write(gate.LEDGER, ledger)
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
    grant_target = tmp_path/gate.GRANT['path']
    grant_target.parent.mkdir(parents=True, exist_ok=True)
    grant_target.write_bytes((gate.ROOT/gate.GRANT['path']).read_bytes())
    g = json.loads(grant_target.read_text(encoding='utf8'))
    ledger['startup_ray_original_grants']={g['grant_id']:{'grant':gate.GRANT,'source_freeze':g['source_freeze'],'inputs':g['inputs'],'local_debit_offset':165,'worker_max':1,'automatic_restarts':0}}
    write(gate.LEDGER, ledger)
    trace_name=gate.BASE+'/synthetic-geometry-return.json'
    write(trace_name, {'schema':'flylab.m3.direct-science.genuine-return.v1','phase':'FINAL','first_error':None,'dispatch_ms':1000,'first_final_ms':1125,'full_return_s':.125,'exit_code':1,'request':request,'source_freeze':freeze})
    source=(gate.ROOT/gate.BASE/'parent-direct-science-account.ps1').read_text(encoding='utf8')
    anchor="$taskRoot=[IO.Path]::GetFullPath('C:/Users/ckthd/Dev/Projects/fly_lab')"
    assert source.count(anchor) == 1
    helper=tmp_path/gate.BASE/'parent-contact-learning-contract-native.ps1'
    helper.write_bytes((gate.ROOT/gate.BASE/'parent-contact-learning-contract-native.ps1').read_bytes())
    script=tmp_path/'synthetic-contact-account.ps1'
    script.write_text(source.replace(anchor,"$taskRoot=[IO.Path]::GetFullPath('"+tmp_path.as_posix().replace("'","''")+"')"),encoding='utf8')
    command=[shutil.which('pwsh'),'-NoProfile','-File',str(script),'-Action','commit','-TracePath',trace_name]
    first=subprocess.run(command,capture_output=True,timeout=10,check=False)
    assert first.returncode == 0, first.stderr
    after=json.loads((tmp_path/gate.LEDGER).read_text(encoding='utf8'))
    assert len(after['debits']) == 166 and after['debits'][-1]['actual_outer_full_return_s'] == .125
    cost=json.loads((tmp_path/after['debits'][-1]['descriptor']['path']).read_text(encoding='utf8'))
    assert cost['schema'] == gate.SCHEMA+'.full-return.v1' and cost['original_payload_accesses'] == 0
    duplicate=subprocess.run(command,capture_output=True,timeout=10,check=False)
    assert duplicate.returncode != 0 and b'Fresh once commit' in duplicate.stderr
    assert len(json.loads((tmp_path/gate.LEDGER).read_text(encoding='utf8'))['debits']) == 166
