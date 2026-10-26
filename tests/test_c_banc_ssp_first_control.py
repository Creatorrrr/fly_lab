"""Small explicit fake circuits only; no production network construction."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import verify_c_banc_m3_ssp_first_control as m


def freeze():
    return {"neutral_network_identity": "new-neutral", "candidate_network_identity": "new-candidate",
            "source_sha256": {"inherited": "pinned"}, "input_sha256": {}, "packages": {}}


def state(arm, tick=0):
    return {"rate": np.array([1., 2., 3.], np.float32), "adaptation": np.zeros(3, np.float32),
            "drive": np.full(3, 4., np.float32), "output_mask": np.ones(3, np.float32),
            "synaptic_resource": np.array([1., 1.], np.float64), "tick": tick,
            "model_hash": "new-" + arm, "recurrent_resource": {"identity": "new-" + arm}}


class FakeWriter(m.prior.Writer):
    def __init__(self, output, fail=None):
        super().__init__(output)
        self.fail, self.failed, self.saved = fail, False, []
    def json(self, name, value):
        if name == self.fail and not self.failed:
            self.failed = True
            raise OSError("publication original")
        return super().json(name, value)
    def checkpoint(self, name, value):
        if name == self.fail and not self.failed:
            self.failed = True
            raise OSError("snapshot publication original")
        self.saved.append(name)
        self.raw(name.replace("/", "-") + ".npz", {"rate": value["rate"]})


class FakeNetwork:
    def __init__(self, owner, arm):
        self.owner, self.arm, self.tick = owner, arm, 0
    def advance(self, drive, steps):
        assert steps == 20
        np.testing.assert_array_equal(drive, np.full(3, 4., np.float32))
        self.owner.calls.append((self.arm, steps))
        if self.owner.failure == self.arm:
            self.tick = 7
            error = RuntimeError("scientific original")
            error.resource_fault_flags = 32
            self.owner.original = error
            raise error
        self.tick = 20
    def snapshot(self):
        if self.owner.failure == self.arm:
            raise RuntimeError("sticky fault prohibits snapshot")
        value = state(self.arm, self.tick)
        value["rate"] += np.float32(.25 if self.arm == "neutral" else .5)
        return value


class FakeDependencies:
    def __init__(self, failure=None, construct_failure=False, cleanup_failure=False):
        self.failure, self.construct_failure, self.cleanup_failure = failure, construct_failure, cleanup_failure
        self.calls, self.live, self.max_live, self.cleanups = [], 0, 0, 0
        self.original = None
    def load(self, f):
        selections = {arm: {"source_ids": np.array([2, 0], np.int64),
                      "utilization": np.array([0., .05 if arm == "candidate" else 0.], np.float64),
                      "recovery_tau": np.full(2, .15, np.float64)} for arm in ("neutral", "candidate")}
        reference = state("old")
        reference["model_hash"] = "old-RK4"
        return {"selections": selections, "drive": state("neutral")["drive"], "rk4_reference": reference}
    def construct(self, arm, data, profiles, persist, on_return):
        assert self.live == 0
        self.live += 1
        self.max_live = max(self.live, self.max_live)
        net = FakeNetwork(self, arm)
        on_return(net)
        profiles.append({"setup_ticks": 40, "capture_ticks": 20, "unknown_ticks": 0})
        if self.construct_failure:
            raise OSError("post-return receipt original")
        return net
    def initial(self, net, data, arm):
        return state(arm)
    def final(self, net, data, arm):
        value = net.snapshot()
        return value, m.endpoint_factor(value, data["selections"][arm])
    def release(self, net):
        self.live -= 1
    def cleanup(self):
        self.cleanups += 1
        if self.cleanup_failure and self.original is not None:
            raise OSError("cleanup secondary")
    def resource_check(self, output):
        return {"continuous_gpu_peak_measured": False, "host_peak_bytes": 123}


def run(tmp_path, deps=None, fail=None, clock=lambda: 0.):
    output = tmp_path / "fresh"
    writer = FakeWriter(output, fail)
    deps = deps or FakeDependencies()
    result, error = m.execute(freeze(), output, dependencies=deps, writer=writer, clock=clock)
    return result, error, deps, writer, output


def test_sequential_one_advance_numeric_difference_allowed(tmp_path):
    result, error, deps, writer, output = run(tmp_path)
    assert error is None and result["run_valid"]
    assert deps.calls == [("neutral", 20), ("candidate", 20)]
    assert deps.max_live == 1 and deps.live == 0 and deps.cleanups == 2
    assert len(writer.saved) == 4
    assert result["cost"]["scientific_ticks"] == 40
    assert result["numeric_comparisons"]["new_neutral_minus_original_RK4_rate"]["max_abs"] == .25
    assert not result["neutral_RK4_byteexact_required"]
    assert all(result[key] is False for key in m.FLAGS)
    saved = json.loads((output / "result.json").read_text())
    assert saved["schema"] == m.RESULT_SCHEMA and saved["diagnostic_id"] == "E305"
    assert json.loads((output / "result-sha256.json").read_text())["sha256"] == m.prior.sha256(output / "result.json")
    manifest = json.loads((output / "output-manifest.json").read_text())
    assert "source-manifest.json" in manifest["files"] and "result.json" not in manifest["files"]
    source = json.loads((output / "source-manifest.json").read_text())
    assert set(source["owned_sha256"]) == set(m.OWNED) and source["source_sha256"] == freeze()["source_sha256"]


@pytest.mark.parametrize("arm,expected", [("neutral", 7), ("candidate", 27)])
def test_original_fault_partial_ticks_no_retry_and_cleanup(tmp_path, arm, expected):
    deps = FakeDependencies(failure=arm, cleanup_failure=True)
    result, error, deps, _, _ = run(tmp_path, deps)
    assert error is deps.original and str(error) == "scientific original"
    assert not result["run_valid"] and deps.live == 0
    assert result["cost"]["scientific_ticks"] == expected
    row = result["cost"]["arms"][arm]
    assert row["actual_tick"] == 7 and row["advance_failed"] == 1
    assert row["resource_fault_flags"] == 32 and row["partial_neural_snapshot_available"] is False
    assert len(deps.calls) == (1 if arm == "neutral" else 2)
    assert any("cleanup" in note for note in error.__notes__)


def test_constructor_return_receipt_failure_releases_owner(tmp_path):
    result, error, deps, _, _ = run(tmp_path, FakeDependencies(construct_failure=True))
    assert str(error) == "post-return receipt original" and not result["run_valid"]
    assert deps.live == 0 and deps.cleanups == 1 and not deps.calls
    assert result["cost"]["models_returned"] == 1


def test_pre_arm_publication_failure_cleanup(tmp_path):
    result, error, deps, _, _ = run(tmp_path, fail="source-manifest.json")
    assert str(error) == "publication original" and not result["run_valid"]
    assert deps.cleanups == 1 and not deps.calls
    assert result["cost"]["models_requested"] == 0


def test_constructor_constants_and_explicit_profiler():
    calls, ownership = [], []
    def network(*args, **kwargs):
        calls.append((args, kwargs))
        return object()
    def profiler(label, callback, rows, persist, on_return):
        assert label == "e305_ssp_candidate"
        value = callback()
        on_return(value)
        rows.append({"setup_ticks": 40, "capture_ticks": 20})
        persist()
        return value
    selection = {"edge_ids": np.array([1]), "premotor_ids": np.array([2]),
                 "utilization": np.array([.05]), "recovery_tau": np.array([.15])}
    data = {"network": {"core_adaptation": {"ids": np.array([2]), "gain": 3.},
                        "motor_adaptation": {"ids": np.array([1]), "gains": np.array([2.])},
                        "motor_timescale": {"log_tau": np.array([-.5])}},
            "configuration": {"membrane_tau": np.array([.02, .03], np.float32)},
            "weights": "explicit fake CSR", "selections": {"candidate": selection}}
    deps = m.NativeDependencies(engine=SimpleNamespace(SSPRecoveryResourceNetwork=network), profiler=profiler)
    value = deps.construct("candidate", data, [], lambda: None, ownership.append)
    assert ownership == [value]
    args, kwargs = calls[0]
    assert kwargs["dt"] == .00025 and kwargs["capture_steps"] == 20 and kwargs["cuda_implementation"] == "packed"
    assert kwargs["adaptation_gain"] == 8. and kwargs["adaptation_tau_s"] == .3
    np.testing.assert_array_equal(args[2], np.ones(2, np.float32))
    np.testing.assert_array_equal(args[3], np.full(2, 7.5, np.float32))
    np.testing.assert_array_equal(args[4], np.full(2, 200., np.float32))


@pytest.mark.parametrize("name", ["neutral/initial-neural", "neutral/final-neural", "output-manifest.json", "result.json", "result-sha256.json"])
def test_publication_failures_cannot_leave_success_receipt(tmp_path, name):
    result, error, deps, _, output = run(tmp_path, fail=name)
    assert error is not None and not result["run_valid"] and deps.live == 0
    assert json.loads((output / "result.json").read_text())["run_valid"] is False


def test_deadline_prevents_next_constructor(tmp_path):
    deps = FakeDependencies()
    def clock():
        return 46. if deps.cleanups else 0.
    result, error, deps, _, _ = run(tmp_path, deps, clock=clock)
    assert isinstance(error, TimeoutError) and not result["run_valid"]
    assert deps.calls == [("neutral", 20)] and result["cost"]["models_requested"] == 1


def test_endpoint_unsorted_source_ids_and_unity():
    value = state("candidate")
    value["synaptic_resource"] = np.array([.5, 1.], np.float64)
    selection = {"source_ids": np.array([2, 0]), "utilization": np.array([.05, 0.]), "recovery_tau": np.array([.15, .15])}
    expected = np.array([.5 * (1 + .15 * .05 * 3), 1], np.float32)
    np.testing.assert_array_equal(m.endpoint_factor(value, selection), expected)


@pytest.mark.parametrize("mutation", ["negative_rate", "negative_adaptation", "resource", "inactive_resource", "drive", "mask", "identity", "tick"])
def test_native_endpoint_guards_with_explicit_fake(mutation):
    value = state("candidate", 20)
    data = {"network": state("candidate"), "drive": value["drive"].copy(), "freeze": freeze(),
            "selections": {"candidate": {"source_ids": np.array([2, 0]), "utilization": np.array([.05, 0.]),
                                          "recovery_tau": np.array([.15, .15])}}}
    if mutation == "negative_rate":
        value["rate"][0] = -1e-9
    elif mutation == "negative_adaptation":
        value["adaptation"][1] = -1e-9
    elif mutation == "resource":
        value["synaptic_resource"][0] = 1.001
    elif mutation == "inactive_resource":
        value["synaptic_resource"][1] = .9
    elif mutation == "drive":
        value["drive"][0] = 0
    elif mutation == "mask":
        value["output_mask"][0] = 0
    elif mutation == "identity":
        value["model_hash"] = "old"
    else:
        value["tick"] = 19
    deps = m.NativeDependencies(engine=SimpleNamespace(resource_identity=lambda metadata: metadata["identity"]))
    with pytest.raises(ValueError):
        deps.final(SimpleNamespace(snapshot=lambda: value), data, "candidate")


@pytest.mark.parametrize("kind", ["dtype", "shape", "nonfinite"])
def test_numeric_definitions_strict(kind):
    left, right = np.ones(2, np.float32), np.ones(2, np.float32)
    if kind == "dtype":
        right = right.astype(np.float64)
    elif kind == "shape":
        right = right[:1]
    else:
        right[0] = np.nan
    with pytest.raises(ValueError):
        m.differences(left, right)


def qualification_fixture(tmp_path):
    path = tmp_path / "real-receipt.json"
    path.write_text("{}")
    pins = {name: str(index) * 64 for index, name in enumerate(m.PRODUCERS)}
    review = {"approved": True, "blocking_defects": [], "source_sha256": pins}
    qual = {**review, "schema": m.QUALIFICATION_SCHEMA, "checks": dict(m.QUALIFICATION_CHECKS),
            "evidence_sha256": {path.name: m.prior.sha256(path)}}
    return review, qual, pins


def test_qualification_current_receipt_bytes(tmp_path):
    review, qual, pins = qualification_fixture(tmp_path)
    m.validate_qualification(review, qual, pins, tmp_path)
    (tmp_path / "real-receipt.json").write_text("changed")
    with pytest.raises(ValueError):
        m.validate_qualification(review, qual, pins, tmp_path)


@pytest.mark.parametrize("key", ["cuda_pass", "disk_future_exact", "independent_oracle_pass", "cpu_pass"])
def test_all_actual_qualification_checks_required(tmp_path, key):
    review, qual, pins = qualification_fixture(tmp_path)
    qual["checks"][key] = False
    with pytest.raises(ValueError):
        m.validate_qualification(review, qual, pins, tmp_path)


def test_pin_conflict_and_receipt_changed_bytes(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text("{}")
    digest = m.prior.sha256(path)
    assert m.receipt({"path": path.name, "sha256": digest}, root=tmp_path) == {}
    with pytest.raises(ValueError):
        m.pin_bytes([{path.name: digest}, {path.name: "0" * 64}], tmp_path)
    path.write_text("[]")
    with pytest.raises(ValueError):
        m.receipt({"path": path.name, "sha256": digest}, root=tmp_path)


def test_old_frozen_load_context_separate_new_identity(tmp_path):
    path = tmp_path / m.BASE_FREEZE_PATH
    path.parent.mkdir(parents=True)
    old = {"schema": m.prior.FREEZE_SCHEMA, "inputs": {"neutral_final": "literal"}, "candidate_network_identity": "old-candidate"}
    path.write_text(json.dumps(old))
    f = {**freeze(), "inputs": old["inputs"], "e302_freeze": {"path": m.BASE_FREEZE_PATH, "sha256": m.prior.sha256(path)}}
    seen = []
    def loader(context):
        seen.append(context)
        return {"selection": {"utilization": np.array([.05, 0.], np.float64)}, "rk4_reference": {"model_hash": "old-RK4"}}
    engine = SimpleNamespace(resource_metadata=lambda source, selection: {"id": "new-candidate" if selection["utilization"].any() else "new-neutral"},
                             checked_resource_metadata=lambda value: value, resource_identity=lambda value: value["id"])
    data = m.load_inputs(f, engine=engine, old_loader=loader, root=tmp_path)
    assert seen == [old]
    np.testing.assert_array_equal(data["selections"]["candidate"]["utilization"], [.05, 0.])
    np.testing.assert_array_equal(data["selections"]["neutral"]["utilization"], [0., 0.])
    f["neutral_network_identity"] = "old-RK4"
    with pytest.raises(ValueError):
        m.load_inputs(f, engine=engine, old_loader=loader, root=tmp_path)


def test_e302_fault_cost_cross_validation():
    fault = {"attempted_tick": 10, "completed_tick": 9, "stage_index": 3, "actual_flags": 4, "negative_source_ids": [96198], "valid": False}
    ledger = {"science": {"completed_ticks": 9, "partial_ticks": 1, "advance_requested": 10, "advance_completed": 9, "advance_failed": 1}, "body_constructors": 0}
    flags = {"candidate_adopted": False, "F1_F8_complete": False, "F6_exposed": False}
    result = {**flags, "schema": m.prior.RESULT_SCHEMA, "diagnostic_id": "E302", "run_valid": True, "first_fault_observed": True,
              "all_valid_factors_unity_before_first_fault": True, "first_fault": fault, "cost": ledger,
              "rows": [{"valid": True, "factor_nonunity_count": 0} for _ in range(39)] + [fault]}
    audit = {**flags, "schema": "flylab.m3.e302.root-original-raw-audit.v1", "pass": True, "diagnostic_run_valid": True,
             "prior_valid_factor_stages": 39, "all_prior_factors_one": True, "first_fault": fault}
    m.validate_e302(result, ledger, audit)
    changed = copy.deepcopy(result)
    changed["rows"][0]["factor_nonunity_count"] = 1
    with pytest.raises(ValueError):
        m.validate_e302(changed, ledger, audit)
