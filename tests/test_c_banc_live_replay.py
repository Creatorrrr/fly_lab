"""New generated replay fixtures only; never open Original numerical arrays."""
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import banc_m3_live_replay_admission as admission
from tools import verify_c_banc_m3_live_replay as replay


def recorded_fixture(offset=0.):
    n, sources = 391, 3
    starts, ends = np.asarray(replay.STARTS, np.int64), np.asarray(replay.ENDS, np.int64)
    context = {
        "pre_control": starts, "pre_network": 20 * starts, "pre_time_s": .005 * starts, "pre_valid": np.ones(4, bool),
        "drive_control_end": ends, "drive_network_pre": 20 * (ends - 1), "drive_time_pre_s": .005 * (ends - 1),
        "drive_valid": np.ones(16, bool), "rate": np.zeros((4, n), np.float32), "adaptation": np.zeros((4, n), np.float32),
        "bank": np.zeros((4, 391, 3), np.float64), "original_activation": np.zeros((4, 96), np.float64),
        "resource": np.ones((4, sources), np.float64), "u_zero": np.ones(4, bool), "resource_one": np.ones(4, bool),
        "drive": np.arange(16 * n, dtype=np.float32).reshape(16, n), "source_ids": np.arange(sources, dtype=np.int64),
        "utilization": np.zeros(sources, np.float64), "recovery_tau_s": np.ones(sources, np.float64),
        "motor_ids": np.arange(391, dtype=np.int64), "output_mask": np.ones(n, np.float32),
    }
    for j in range(4):
        for k, name in enumerate(("rate", "adaptation", "bank", "original_activation")):
            context[name][j] = offset + 10 * j + k + 1
    ticks = np.arange(301, dtype=np.int64)
    control = {"control_tick": ticks, "network_tick": 20 * ticks, "physics_time_s": .005 * ticks}
    packet = {"actual_csr": {"indices": np.arange(sources, dtype=np.int32)},
              "recurrent_mask": {"positions": np.arange(sources, dtype=np.int64)},
              "calibration": {"motor_ids": context["motor_ids"].copy()},
              "motor_configuration": {"motor_ids": context["motor_ids"].copy()},
              "output_mask": {"value": context["output_mask"].copy()},
              "metadata": {"model_identity": "generated-neural", "decoder_hash": "generated-decoder"}}
    report = {"run_valid": True, "capture_complete": True, "all_owned_ended": True,
              "context_array_sha256": {k: replay.sha(v) for k, v in context.items()},
              "transfer": {"source_network_identity": "generated-neural", "source_adapter_identity": "generated-decoder"}}
    return context, control, report, packet


def test_math_recorded_windows_clocks_states_and_two_arm_bytes_are_bound():
    neutral, candidate = recorded_fixture(), recorded_fixture(100.)
    for fixture in (neutral, candidate):
        context = fixture[0]
        result = replay.validate_recorded(*fixture, np, n=391, resource_n=3)
        assert result["windows"] == [100, 164, 228, 292]
        assert result["drive_ends"] == [101, 102, 103, 104, 165, 166, 167, 168, 229, 230, 231, 232, 293, 294, 295, 296]
        assert result["post_neural_decoder_endpoints"] == "NOT_OBSERVED"
        assert all(context[name].dtype == np.float64 for name in ("bank", "original_activation", "resource"))
    # Respective report identity refuses exchanging the arms even though shapes match.
    with pytest.raises(ValueError, match="byte identity"):
        replay.validate_recorded(candidate[0], neutral[1], neutral[2], neutral[3], np, n=391, resource_n=3)


@pytest.mark.parametrize("mutation", ["pre_clock", "drive_clock", "drive_order", "missing_pre", "missing_drive", "resource", "utilization", "motor_order", "source_order", "state_precision"])
def test_math_recorded_input_refuses_wrong_window_clock_order_missing_or_precision(mutation):
    context, control, report, packet = recorded_fixture()
    if mutation == "pre_clock":
        context["pre_network"][0] += 20
    elif mutation == "drive_clock":
        context["drive_time_pre_s"][0] += .005
    elif mutation == "drive_order":
        context["drive_control_end"][[0, 1]] = context["drive_control_end"][[1, 0]]
    elif mutation == "missing_pre":
        context["pre_valid"][2] = False
    elif mutation == "missing_drive":
        context["drive_valid"][15] = False
    elif mutation == "resource":
        context["resource"][0, 0] = .5
    elif mutation == "utilization":
        context["utilization"][0] = .1
    elif mutation == "motor_order":
        context["motor_ids"] = context["motor_ids"][::-1].copy()
    elif mutation == "source_order":
        context["source_ids"] = context["source_ids"][::-1].copy()
    else:
        context["rate"] = context["rate"].astype(np.float64)
    # Keep bytes internally honest so each refusal exercises semantic input checks.
    report["context_array_sha256"] = {k: replay.sha(v) for k, v in context.items()}
    with pytest.raises(ValueError):
        replay.validate_recorded(context, control, report, packet, np, n=391, resource_n=3)


def test_math_deployment_uses_saved_float32_patch_once_and_preserves_other_edges():
    source = np.asarray([.8, -2., 3., 4., 5.], np.float32)
    positions = np.asarray([1, 4], np.int64)
    selected = np.asarray([-2.0002, 5.0005], np.float32)
    prior = source.copy()
    deployed = replay.selected_deployment(source, positions, selected, np)
    assert np.array_equal(deployed[positions], selected)
    assert np.array_equal(deployed[[0, 2, 3]], prior[[0, 2, 3]]) and np.array_equal(source, prior)
    assert not np.array_equal(deployed[positions], (selected * 1.0001).astype(np.float32))
    with pytest.raises(ValueError):
        replay.selected_deployment(source, np.asarray([1, 1], np.int64), selected, np)


class FixtureNeural:
    """Tiny generated operator; actual checkpointed forward/receipt path is used."""
    def __init__(self):
        self.device, self.n, self.dt = torch.device("cpu"), 391, np.float32(.00025)
        self.operator = SimpleNamespace(group_count=1)
        self.ticks, self.prepares, self.first_states, self.efficacies = 0, 0, [], []

    def prepare_efficacy(self, efficacy):
        self.prepares += 1
        self.efficacies.append(efficacy.detach().clone())
        return efficacy

    def step(self, rate, adaptation, drive, efficacy, mask):
        if self.ticks % 80 == 0:
            self.first_states.append(tuple(v.detach().clone() for v in (rate, adaptation, drive)))
        self.ticks += 1
        return rate + drive * self.dt, adaptation + torch.ones_like(adaptation) * self.dt


class FixtureDecoder:
    def __init__(self):
        self.device, self.calls, self.first_states = torch.device("cpu"), 0, []

    def step(self, motors, bank, original, *, dt):
        assert dt == .005
        if self.calls % 4 == 0:
            self.first_states.append((bank.detach().clone(), original.detach().clone()))
        self.calls += 1
        return {"q": motors[:42].to(torch.float64) + bank[0, 0], "tendon": motors[:6].to(torch.float64) + original[0],
                "bank": bank + dt, "original_activation": original + dt}


def test_math_existing_rollout_restores_each_recorded_state_and_returns_exactly640_ticks_without_warmup():
    from tools.banc_m3_checkpointed_rollout import CheckpointedControlRollout
    neural, decoder = FixtureNeural(), FixtureDecoder()
    rollout = CheckpointedControlRollout(neural, decoder, np.arange(391, dtype=np.int64), steps_per_control=20,
                                         control_dt=.005, checkpoint_controls=1, max_controls=4)
    assert neural.ticks == decoder.calls == neural.prepares == 0
    receipts, contexts = {}, []
    for arm_index, arm in enumerate(replay.ARMS):
        context = recorded_fixture(arm_index * 100.)[0]
        contexts.append(context)
        before = {key: replay.sha(value) for key, value in context.items()}
        for j, start in enumerate(replay.STARTS):
            state = [torch.as_tensor(context[name][j].copy()) for name in ("rate", "adaptation", "bank", "original_activation")]
            drives = torch.as_tensor(context["drive"][4*j:4*j+4].copy())
            with torch.no_grad():
                output = rollout.run(*state, drives, torch.ones(1, dtype=torch.float64), torch.as_tensor(context["output_mask"].copy()))
            receipt = output["receipt"]
            receipt.confirm()
            receipts[arm + "." + str(start)] = {key: value for key, value in vars(receipt).items() if key != "device"}
            assert output["q"].shape == (4, 42) and output["tendon"].shape == (4, 6)
        assert before == {key: replay.sha(value) for key, value in context.items()}
    work = replay.verify_receipts(receipts, full=True)
    assert neural.ticks == work["forward_neural_ticks_device_confirmed"] == 640
    assert decoder.calls == work["forward_controls_device_confirmed"] == 32 and neural.prepares == 8
    for k in range(8):
        context, j = contexts[k // 4], k % 4
        assert np.array_equal(neural.first_states[k][0].numpy(), context["rate"][j])
        assert np.array_equal(neural.first_states[k][1].numpy(), context["adaptation"][j])
        assert np.array_equal(neural.first_states[k][2].numpy(), context["drive"][4*j])
        assert np.array_equal(decoder.first_states[k][0].numpy(), context["bank"][j])
        assert np.array_equal(decoder.first_states[k][1].numpy(), context["original_activation"][j])
        assert torch.equal(neural.efficacies[k], torch.ones(1, dtype=torch.float64))
    broken = copy.deepcopy(receipts)
    broken[next(iter(broken))]["forward_neural_ticks_requested"] += 1
    with pytest.raises(ValueError):
        replay.verify_receipts(broken, full=True)
    assert work["recompute_neural_ticks_requested"] == work["backward_blocks_requested"] == work["warmup_ticks"] == 0


def test_math_raw_command_mismatch_is_rejected_at_fixed_tolerance_and_endpoint_report_stays_unobserved():
    expected = np.zeros((4, 42), np.float64)
    actual = expected.copy()
    actual[0, 0] = 1.1e-11
    rejected = replay.compare_output(actual, expected, np)
    assert rejected["passed"] is False and rejected["rtol"] == 0 and rejected["max_abs"] == 1.1e-11
    assert rejected["rms"] > 0 and rejected["exact"] is False
    actual[0, 0] = 1e-11
    assert replay.compare_output(actual, expected, np)["passed"]
    result = {"run_valid": True, "output_correspondence_passed": False,
              "comparisons": {"neutral.100": {"command": rejected, "tendon": replay.compare_output(np.zeros((4, 6)), np.zeros((4, 6)), np)}}}
    text = replay.markdown(result)
    assert "run_valid=True" in text and "output_correspondence_passed=False" in text and "NOT_OBSERVED" in text
    assert "VJP" in text and "후보 채택 false" in text and "INCOMPLETE" in text
    actual[0, 0] = np.nan
    with pytest.raises(ValueError):
        replay.compare_output(actual, expected, np)


def test_gate_new_runner_and_helper_import_do_not_load_numerical_or_body_modules():
    script = "import sys; from tools import verify_c_banc_m3_live_replay, banc_m3_live_replay_admission; " + \
             "assert not any(x in sys.modules for x in ('numpy','torch','scipy','mujoco','flygym','tools.verify_c_banc_m3_common_descent_body'))"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr


def test_gate_deadline_precedes_request_read_and_hot_callbacks_do_not_rehash_Source(monkeypatch):
    touched = []
    monkeypatch.setattr(admission.time, "perf_counter", lambda: 301.)
    monkeypatch.setattr(admission, "spec_for", lambda *args: touched.append(args))
    with pytest.raises(TimeoutError):
        admission.Admission("never-open.json", "pre_spawn", entry=100.)
    assert touched == []
    live = admission.Admission.__new__(admission.Admission)
    live.entry, live._deadline_s = 100., 200
    live.fence = lambda label: touched.append(label)
    monkeypatch.setattr(admission.time, "perf_counter", lambda: 101.)
    live.check("science")
    assert touched == []
    live.check("publication")
    assert touched == ["publication"]
    limits = admission.limits_for("live_replay")
    assert [limits[k] for k in ("outer_s", "child_s", "internal_s", "cleanup_s")] == [240, 220, 200, 20]
    assert limits["output_cap_bytes"] == 16 << 20
    with pytest.raises(ValueError):
        admission.limits_for("tracking_neutral")


def history_fixture():
    values, debits = {}, [{"cost_id": f"old-{i}", "mode": "generated"} for i in range(165)]

    def pin(name, value):
        values[name] = value
        return {"path": name, "bytes": 1, "sha256": "0" * 64}

    grant = {"source_freeze": {"historical_J": True}, "inputs": admission.INPUTS, "original_cumulative_before_s": 1000.,
             "original_stage_before_s": 100., "science_local_used_before_s": 371.257}
    roster = {"tracking_source_freeze": pin("historic-c", {"sources": {f"source-{i}": {"bytes": 1, "sha256": "1" * 64} for i in range(8)}}),
              "tracking_source_archive": pin("archive-c", {"schema": "flylab.m3.tracking-source-archive.v1", "sources": [
                  {"original_path": f"source-{i}", "archive_path": f"archive-{i}", "bytes": 1, "sha256": "1" * 64} for i in range(8)]}),
              "actual_tracking_result": pin("actual", {"status": "ACTUAL_TRACKING_CAPTURE_COMPLETE", "candidate_adopted": False}),
              "tracking_roster": {"historical_roster": True}, "tracking_costs": [], "captures": {}}
    archive_hash = hashlib.sha256(b"x").hexdigest()
    values["archive-readnames"] = []
    for i in range(8):
        values["historic-c"]["sources"][f"source-{i}"]["sha256"] = archive_hash
        values["archive-c"]["sources"][i]["sha256"] = archive_hash
        values["archive-bytes-source-" + str(i)] = b"x"
    for i in range(14):
        name = f"J-{i}" if i < 8 else f"startup-ray-20261007-j-saved-transfer-{i-7:02d}" if i < 11 else f"tracking-{i-11}"
        cost = {"cost_id": name, "mode": "original", "grant": admission.GRANT, "local_debit_offset": 165, "actual_outer_full_return_s": 1.,
                "exit_code": 1 if i in (8, 9) else 0}
        if i < 8:
            cost.update(schema="flylab.m3.startup-ray.full-return.v1", role=admission.OLD_ROLES[i], source_freeze=grant["source_freeze"], inputs=grant["inputs"])
        elif i < 11:
            cost.update(schema="flylab.m3.startup-ray.saved-diagnosis.full-return.v1")
        else:
            role = ("tracking_neutral", "tracking_candidate", "tracking_diagnosis")[i-11]
            cost.update(schema="flylab.m3.startup-tracking.full-return.v1", role=role, cap_exceeded=False,
                        source_freeze=roster["tracking_source_freeze"], roster=roster["tracking_roster"])
        spec = pin(name, cost)
        debits.append({"cost_id": name, "mode": "original", "descriptor": spec})
        if i >= 11:
            roster["tracking_costs"].append(spec)
    for i in range(17):
        name = f"generated-{i}"
        cost = {"schema": "flylab.m3.direct-science.full-return.v1", "cost_id": name, "charged_once": True, "actual_outer_full_return_s": .1}
        debits.append({"cost_id": name, "mode": "generated", "descriptor": pin(name, cost), "actual_outer_full_return_s": .1})
    for role in ("tracking_neutral", "tracking_candidate"):
        artifacts = {key: {"path": role + "." + key, "bytes": 1, "sha256": "2" * 64} for key in ("context", "control", "body_report")}
        roster["captures"][role] = {**artifacts, "manifest": pin(role + ".manifest", {"role": role, "run_valid": True, "capture_complete": True, "artifacts": artifacts})}
    before = {"debits": debits, "original_cumulative_used_s": 1014., "original_stage_used_s": 114.,
              "startup_ray_original_local_used_s": 385.257, "original_remaining_s": 15186.}
    # Match exact ordered double summation rather than a rounded carry literal.
    before["startup_ray_original_local_used_s"] = grant["science_local_used_before_s"] + 14.
    def read(spec):
        return values[spec["path"]]

    def closed(cost, debit):
        return cost

    def archive_read(item):
        values["archive-readnames"].append(item["original_path"])
        return values["archive-bytes-" + item["original_path"]]

    return before, grant, roster, {}, read, closed, archive_read, values


def test_gate_history_keeps_closed_tracking3_saved_failures_generated_tail_and_archived_Source():
    fixture = history_fixture()
    original = copy.deepcopy(fixture[0])
    assert admission.validate_history(*fixture[:7]) == 14.
    assert fixture[0] == original and len(original["debits"]) == 196
    assert fixture[7]["archive-readnames"] == [f"source-{i}" for i in range(8)]
    # No current canonical Source is opened by this historical metadata lineage.
    fixture[7]["historic-c"]["sources"]["source-0"]["sha256"] = "3" * 64
    with pytest.raises(ValueError, match="archive/historical"):
        admission.validate_history(*fixture[:7])


@pytest.mark.parametrize("mutation", ["duplicate", "lost_tracking", "saved_failure_rewritten", "failed_capture", "carry_reset"])
def test_gate_history_refuses_rewritten_once_costs_or_incomplete_recordings(mutation):
    f = history_fixture()
    if mutation == "duplicate":
        f[0]["debits"].append(copy.deepcopy(f[0]["debits"][-1]))
    elif mutation == "lost_tracking":
        f[0]["debits"] = [d for d in f[0]["debits"] if d["cost_id"] != "tracking-1"]
    elif mutation == "saved_failure_rewritten":
        f[7]["startup-ray-20261007-j-saved-transfer-01"]["exit_code"] = 0
    elif mutation == "failed_capture":
        f[7]["tracking_candidate.manifest"]["capture_complete"] = False
    else:
        f[0]["startup_ray_original_local_used_s"] = 14.
    with pytest.raises(ValueError):
        admission.validate_history(*f[:7])


def test_gate_archive_reader_refuses_same_length_archived_Source_byte_drift():
    fixture = history_fixture()
    fixture[7]["archive-bytes-source-2"] = b"z"
    with pytest.raises(ValueError, match="actual archived Source bytes"):
        admission.validate_history(*fixture[:7])
    assert fixture[7]["archive-readnames"] == ["source-0", "source-1", "source-2"]


def planner_fixture():
    from tools import analyze_c_banc_m3_startup_tracking as tracking
    bindings = {}
    for role, fields in replay.PACKET_SCHEMA.items():
        bindings[role] = {"path": "generated-" + role, "bytes": 93333788 if role == "actual_csr" else 4096,
                          "kind": "npy" if role in ("inverse", "output_mask") else "npz",
                          "selected_schema": {key: {"shape": list(shape), "dtype": dtype, "finite": True} for key, (shape, dtype) in fields.items()}}
    captures = {arm: {"context": {"path": arm + ".context", "bytes": 16319804},
                      "control": {"path": arm + ".control", "bytes": 3096370}} for arm in replay.ARMS}
    saved_fields = {"context": tracking.CONTEXT_FIELDS,
                    "control": {key: (tuple(301 if axis is None else axis for axis in shape), dtype)
                                for key, (shape, dtype) in tracking.CONTROL_FIELDS.items()}}
    return bindings, captures, {"path": "generated-candidate", "bytes": 3040972}, admission.limits_for("live_replay"), saved_fields


def test_math_metadata_plan_accepts_actual46MiB_members_and_reconciles_combined_bytes_without_allocation(monkeypatch):
    fixture = planner_fixture()

    def no_array(*args, **kwargs):
        raise AssertionError("Metadata planning cannot allocate/load numerical arrays")

    for name in ("empty", "zeros", "ones", "array", "asarray", "load"):
        monkeypatch.setattr(np, name, no_array)
    plan = replay.plan_inputs(*fixture)
    assert plan["packet"]["actual_csr"]["member_bytes"]["data"] == 46339408
    assert plan["packet"]["actual_csr"]["member_bytes"]["indices"] == 46339408
    assert plan["packet_expanded_bytes"] == 119710856 and plan["saved_expanded_bytes"] == 35846308
    assert plan["combined_expanded_bytes"] == 155557164 < fixture[3]["selected_expanded_cap_bytes"]
    small = {"member_bytes": {"a": 8, "b": 16}}
    assert replay.reconcile_arrays({"a": SimpleNamespace(nbytes=8), "b": SimpleNamespace(nbytes=16)}, small) == 24
    with pytest.raises(ValueError, match="actual selected nbytes"):
        replay.reconcile_arrays({"a": SimpleNamespace(nbytes=9), "b": SimpleNamespace(nbytes=16)}, small)


@pytest.mark.parametrize("mutation", ["legacy64", "dtype", "shape", "file96", "combined208", "saved16"])
def test_math_metadata_plan_refuses_wrong_shape_dtype_member_file_or_combined_cap_without_allocation(mutation):
    bindings, captures, candidate, limits, saved_fields = planner_fixture()
    if mutation == "legacy64":
        with pytest.raises(ValueError, match="member allocation cap"):
            replay.schema_nbytes([16777217], "<f4", 64 << 20)
        limits["legacy_packet_member_cap_bytes"] = 16 << 20
    elif mutation == "dtype":
        bindings["actual_csr"]["selected_schema"]["data"]["dtype"] = "|O"
    elif mutation == "shape":
        bindings["actual_csr"]["selected_schema"]["data"]["shape"] = [-11584852]
    elif mutation == "file96":
        bindings["actual_csr"]["bytes"] = (96 << 20) + 1
    elif mutation == "combined208":
        limits["selected_expanded_cap_bytes"] = 155557163
    else:
        limits["member_cap_bytes"] = 1 << 20
    with pytest.raises(ValueError):
        replay.plan_inputs(bindings, captures, candidate, limits, saved_fields)


class InjectedWriter:
    def __init__(self, directory, fail_names):
        self.directory, self.fail_names, self.total = directory, set(fail_names), 0

    def write(self, name, raw):
        if name in self.fail_names:
            raise OSError("injected bounded writer failure: " + name)
        (self.directory / name).write_bytes(raw)
        self.total += len(raw)

    def json(self, name, value):
        self.write(name, json.dumps(value, allow_nan=False).encode("utf8"))

    def arrays(self, name, values):
        self.write(name, b"small generated raw marker")


@pytest.mark.parametrize("fail_name", ["raw.npz", "report.md", "report.json"])
def test_math_injected_publication_failure_preserves_primary_error_partial_work_and_invalid_report(tmp_path, fail_name):
    primary = {"stage": "numerical-load", "type": "ImportError", "message": "injected original numerical import refusal"}
    result = {"run_valid": False, "output_correspondence_passed": False, "classification": "technical_invalid",
              "first_error": copy.deepcopy(primary), "secondary_errors": [], "comparisons": {},
              "receipts": {"neutral.100": {"forward_controls_requested": 2, "forward_neural_ticks_returned": 25}},
              "work": {"forward_controls_requested": 2, "forward_neural_ticks_returned": 25}}
    writer = InjectedWriter(tmp_path, {fail_name})
    replay.publish_result(writer, {}, [], result)
    assert result["first_error"] == primary and any(fail_name in e["message"] for e in result["secondary_errors"])
    target = "publication-failure.json" if fail_name == "report.json" else "report.json"
    published = json.loads((tmp_path / target).read_text(encoding="utf8"))
    assert published["run_valid"] is False and published["output_correspondence_passed"] is False
    assert published["first_error"] == primary and published["work"]["forward_neural_ticks_returned"] == 25
    assert json.loads((tmp_path / "receipts.json").read_text(encoding="utf8")) == result["receipts"]


def test_math_successful_forward_is_invalidated_when_raw_publication_fails(tmp_path):
    result = {"run_valid": True, "output_correspondence_passed": True, "classification": "conditional_output_correspondence",
              "first_error": None, "secondary_errors": [], "comparisons": {}, "receipts": {}, "work": {"forward_neural_ticks_returned": 640}}
    replay.publish_result(InjectedWriter(tmp_path, {"raw.npz"}), {}, [], result)
    published = json.loads((tmp_path / "report.json").read_text(encoding="utf8"))
    assert published["run_valid"] is False and published["classification"] == "technical_invalid"
    assert published["first_error"]["stage"] == "publish-raw" and published["work"]["forward_neural_ticks_returned"] == 640


def refusal_fixture(tmp_path, monkeypatch):
    from tools import banc_m3_startup_ray_supervisor as supervisor

    def write(name, value):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, allow_nan=False), encoding="utf8")
        return {"path": name, "bytes": target.stat().st_size, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}

    call = admission.CALLS[0]
    science = write(admission.BASE + "/synthetic-replay-science.json", {"schema": admission.SCHEMA + ".request.v1", "mode": "original", "role": "live_replay", "limits": {}})
    freeze = write(admission.BASE + "/synthetic-replay-source.json", {"fixture": "generated-only refusal"})
    limits = admission.limits_for("live_replay")
    q = {"schema": "flylab.m3.direct-science.supervision.v1", "mode": "original", "role": "live_replay", "call_id": call,
         "science_request": science, "source_freeze": freeze, "grant": admission.GRANT,
         "shared_original_lock_path": admission.SHARED_LOCK, "ledger_path": admission.LEDGER,
         "output": admission.BASE + "/parent-" + call + "/result-supervision",
         "limits": {key: limits[key] for key in ("outer_s", "child_s", "cleanup_s", "host_bytes", "output_cap_bytes")}}
    request = write(admission.BASE + "/synthetic-replay-owner.json", q)
    before = {"debits": [{"cost_id": f"synthetic-old-{i}", "mode": "generated"} for i in range(165)]}
    before_spec = write(admission.BASE + "/synthetic-replay-before.json", before)
    policy = write(admission.BASE + "/synthetic-replay-policy.json", {"schema": "flylab.m3.direct-science.policy.v1", "generated_local_limit_s": 300,
                       "original_authorized": False, "before_ledger": before_spec, "local_debit_offset": 165})
    ledger = {"debits": before["debits"], "direct_science_policy": policy,
              "direct_science_calls": {call: {"request": request, "state": "admitted", "mode": "original", "reserved_s": 240}},
              "startup_ray_original_workers": {}, "startup_ray_original_reservations": {}, "startup_ray_original_pending_actuals": {},
              "startup_ray_original_blocking_holds": {}, "direct_science_pending_actuals": {}, "direct_science_blocking_holds": {},
              "original_jobs_running": 0, "original_reserved_s": 0}
    write(admission.LEDGER, ledger)
    monkeypatch.setattr(supervisor, "creation_ticks", lambda pid: 999)
    monkeypatch.setattr(supervisor.time, "perf_counter", lambda: 101.)
    attempts = []

    def no_job(*args):
        attempts.append(args)
        raise AssertionError("No Job during actual stdlib pre-spawn refusal")

    monkeypatch.setattr(supervisor, "OwnedJob", no_job)
    outer = supervisor.supervise(request["path"], root=tmp_path, entry=100.)
    return supervisor, write, q, request, ledger, outer, attempts


def test_gate_replay_pre_spawn_refusal_preserves_empty_streams_noWorker_and_rejects_second_publication(tmp_path, monkeypatch):
    supervisor, _, q, request, _, outer, attempts = refusal_fixture(tmp_path, monkeypatch)
    assert outer["exit_code"] == 1 and outer["pre_spawn_refused"] and outer["worker_started"] is False and outer["job_created"] is False
    assert outer["original_payload_accesses"] == 0 and outer["worker_pid"] is None and attempts == []
    output = tmp_path / q["output"]
    assert (output / "stdout.bin").read_bytes() == (output / "stderr.bin").read_bytes() == b""
    prior = (output / "outer.json").read_bytes()
    assert outer["output_usage"]["combined_bytes_including_outer"] == len(prior) <= 64 << 10
    assert outer["output_usage"]["logical_combined_cap_bytes"] == 16 << 20
    with pytest.raises(ValueError, match="publish twice"):
        supervisor.supervise(request["path"], root=tmp_path, entry=100.)
    assert (output / "outer.json").read_bytes() == prior and attempts == []


def test_gate_native_once_writer_charges_only_temp_refusal_and_rejects_duplicate(tmp_path, monkeypatch):
    _, write, q, request, ledger, _, attempts = refusal_fixture(tmp_path, monkeypatch)
    assert os.name == "nt" and shutil.which("pwsh"), "Assigned Windows host required"
    root = Path(admission.__file__).resolve().parents[1]
    original_ledger = root / admission.LEDGER
    original_digest = hashlib.sha256(original_ledger.read_bytes()).hexdigest()
    grant_target = tmp_path / admission.GRANT["path"]
    grant_target.parent.mkdir(parents=True, exist_ok=True)
    grant_target.write_bytes((root / admission.GRANT["path"]).read_bytes())
    assert hashlib.sha256(grant_target.read_bytes()).hexdigest() == admission.GRANT["sha256"]
    grant = json.loads(grant_target.read_text(encoding="utf8"))
    ledger["startup_ray_original_grants"] = {grant["grant_id"]: {"grant": admission.GRANT, "source_freeze": grant["source_freeze"], "inputs": grant["inputs"],
                                                              "local_debit_offset": 165, "worker_max": 1, "automatic_restarts": 0}}
    write(admission.LEDGER, ledger)
    trace_name = admission.BASE + "/synthetic-replay-genuine-return.json"
    write(trace_name, {"schema": "flylab.m3.direct-science.genuine-return.v1", "phase": "FINAL", "first_error": None,
                       "dispatch_ms": 1000, "first_final_ms": 1125, "full_return_s": .125, "exit_code": 1,
                       "request": request, "source_freeze": q["source_freeze"]})
    source = (root / admission.BASE / "parent-direct-science-account.ps1").read_text(encoding="utf8")
    anchor = "$taskRoot=[IO.Path]::GetFullPath('C:/Users/ckthd/Dev/Projects/fly_lab')"
    assert source.count(anchor) == 1
    script = tmp_path / "synthetic-replay-account.ps1"
    script.write_text(source.replace(anchor, "$taskRoot=[IO.Path]::GetFullPath('" + tmp_path.as_posix().replace("'", "''") + "')"), encoding="utf8")
    command = [shutil.which("pwsh"), "-NoProfile", "-File", str(script), "-Action", "commit", "-TracePath", trace_name]
    first = subprocess.run(command, capture_output=True, timeout=15, check=False)
    assert first.returncode == 0, first.stderr
    after = json.loads((tmp_path / admission.LEDGER).read_text(encoding="utf8"))
    assert len(after["debits"]) == 166 and after["debits"][-1]["cost_id"] == admission.CALLS[0]
    assert after["debits"][-1]["actual_outer_full_return_s"] == .125
    cost = json.loads((tmp_path / after["debits"][-1]["descriptor"]["path"]).read_text(encoding="utf8"))
    assert cost["schema"] == admission.SCHEMA + ".full-return.v1" and cost["charged_once"] and cost["pre_spawn_refused"]
    assert cost["worker_started"] is False and cost["original_payload_accesses"] == 0
    duplicate = subprocess.run(command, capture_output=True, timeout=15, check=False)
    assert duplicate.returncode != 0 and b"Fresh once commit" in duplicate.stderr
    assert len(json.loads((tmp_path / admission.LEDGER).read_text(encoding="utf8"))["debits"]) == 166
    assert hashlib.sha256(original_ledger.read_bytes()).hexdigest() == original_digest and attempts == []
