"""Generated CPU E288 checks: fake owners and pure adapter metadata only."""

import copy
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pytest

from tools import verify_c_banc_m3_source_decoder_body as probe

BEGAN = time.perf_counter()


def fixture_module(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(name + ".py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


old = fixture_module("test_c_banc_source_native_transfer")
anatomy = fixture_module("test_c_banc_native_monotone_recruitment")


def metrics(candidate=False):
    return {
        "health": {
            "minimum_native_upright": 1.0,
            "nonfoot_contact_count": 0,
            **{
                key: True
                for key in (
                    "healthy",
                    "finite",
                    "complete_1p5s",
                    "clock_exact",
                    "observed_clock_valid",
                )
            },
        },
        "forward": {
            "available": True,
            "window_samples": 200,
            "last_included_control": 299,
            "fixed_initial_heading_forward_mm": probe.BASELINE_FORWARD_MM,
        },
        "slip": {
            "available": True,
            "load_weighted_mm_s": 1.0,
            "missing_selected_slip": 0,
            "support_samples": 20,
            "support_load_sum_bw": 1.0,
        },
        "gait": {
            "window": {
                "start_s": 0.5,
                "end_s": 1.5,
                "sampled_duration_s": 1.0,
                "bouts": [{"start_s": 0.5, "end_s": 1.2, "samples": 141}]
                if candidate
                else [],
            },
            "events": {
                leg: {
                    "complete_cycles": int(candidate),
                    "cycles": [{"aep_s": 0.6, "pep_s": 0.8, "next_aep_s": 1.0}]
                    if candidate
                    else [],
                }
                for leg in probe.LEGS
            },
        },
    }


class FakeDeps(old.FakeDeps):
    def load(self, freeze):
        return {**super().load(freeze), "baseline_short": metrics()}

    def initial(self, *args):
        super().initial(*args)
        if self.mode == "initial":
            raise ValueError("whole initial changed")
        return {"storage_representation_exact": True}

    def compare_neutral(self, *args):
        return {
            **super().compare_neutral(*args),
            "whole_end_exact": self.mode != "whole",
        }

    def short_metrics(self, *args):
        super().short_metrics(*args)
        return metrics(self.mode != "short")

    def save_partial(self, path, session):
        if self.mode == "partial":
            raise OSError("partial component storage")
        return super().save_partial(path, session)


def execute(tmp_path, mode="pass", cleanup=None):
    deps = FakeDeps(mode)
    result, error = probe.execute(
        old.freeze(), tmp_path / "out", deps=deps, budget=lambda: None, cleanup=cleanup
    )
    return deps, result, error


def test_pass_is_continuous_and_only_next_stage_eligible(tmp_path):
    deps, result, error = execute(tmp_path)
    assert error is None and result["run_valid"] and result["neutral_exact"]
    assert (
        result["actual_startup_pass"]
        and result["short_gait_pass"]
        and result["next_separate_10s_eligible"]
    )
    assert [s.control_tick for s in deps.sessions] == [1, 300]
    assert all(s.closed and s.close_calls == 1 for s in deps.sessions)
    assert (
        result["cost"]["scientific_neural_ticks"] == 6020
        and result["cost"]["scientific_physics_steps"] == 15050
    )
    for key in (
        "candidate_adopted",
        "natural_gait_complete",
        "F1_F8_complete",
        "F6_exposed",
        "ten_second_executed",
    ):
        assert result[key] is False


@pytest.mark.parametrize("mode", ["neutral", "whole", "initial"])
def test_neutral_failures_prevent_candidate(tmp_path, mode):
    deps, result, error = execute(tmp_path, mode)
    assert error is not None and len(deps.sessions) == 1 and not result["run_valid"]
    assert all(s.closed for s in deps.sessions)


def test_startup_rejection_is_valid_and_stops_at100(tmp_path):
    deps, result, error = execute(tmp_path, "startup")
    assert error is None and result["run_valid"] and not result["actual_startup_pass"]
    assert (
        deps.sessions[1].control_tick == 100
        and not result["next_separate_10s_eligible"]
    )
    assert result["cost"]["scientific_neural_ticks"] == 2020


def test_healthstop_is_valid_measured_prefix(tmp_path):
    deps, result, error = execute(tmp_path, "health")
    assert error is None and result["run_valid"] and not result["actual_startup_pass"]
    assert deps.saved_partial == 1 and result["arms"]["candidate"]["health_stop"]
    assert (
        result["cost"]["scientific_neural_ticks"] == 40
        and result["cost"]["scientific_physics_steps"] == 77
    )


@pytest.mark.parametrize(
    "mode",
    [
        "constructor",
        "post_return",
        "advance",
        "readout",
        "raw",
        "state",
        "close",
        "publication",
    ],
)
def test_technical_failures_retain_cost_and_close_owners(tmp_path, mode):
    deps, result, error = execute(tmp_path, mode)
    assert (
        error is not None
        and not result["run_valid"]
        and not result["next_separate_10s_eligible"]
    )
    assert all(s.closed for s in deps.sessions)
    assert (
        result["cost"]["scientific_neural_ticks"] >= 0
        and (tmp_path / "out" / "errors.json").exists()
    )


def test_cleanup_failure_retains_full_scientific_cost(tmp_path):
    def fail():
        raise OSError("cleanup failed")

    _, result, error = execute(tmp_path, cleanup=fail)
    assert str(error) == "cleanup failed" and not result["run_valid"]
    assert result["cost"]["scientific_neural_ticks"] == 6020


@pytest.mark.parametrize(
    "path,value",
    [
        (("gait", "window", "start_s"), 0.0),
        (("gait", "window", "end_s"), 1.6),
        (("health", "complete_1p5s"), False),
        (("health", "finite"), False),
        (("forward", "window_samples"), 199),
        (("forward", "fixed_initial_heading_forward_mm"), float("nan")),
        (("slip", "support_samples"), 0),
        (("slip", "support_load_sum_bw"), None),
        (("slip", "missing_selected_slip"), 1),
        (("slip", "load_weighted_mm_s"), 2.0),
        (("gait", "events", probe.LEGS[0], "complete_cycles"), True),
        (("gait", "events", probe.LEGS[1], "complete_cycles"), -1),
        (("gait", "events", probe.LEGS[2], "cycles"), []),
        (("gait", "window", "bouts"), []),
    ],
)
def test_missing_or_changed_short_metrics_fail(path, value):
    candidate = metrics(True)
    node = candidate
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = value
    assert not probe.assess_short(metrics(), candidate)["pass"]


def payload():
    adapter, args = anatomy.make_adapter()
    tau, kappa, slopes, membership, _roster = args
    network = {
        "model_hash": "fake-native",
        "rate": np.zeros(2, np.float32),
        "adaptation": np.zeros(2, np.float32),
        "drive": np.zeros(2, np.float32),
        "output_mask": np.ones(2, np.float32),
        "tick": 0,
    }
    saved = {
        "control_tick": 0,
        "network": network,
        "adapter": adapter.snapshot(),
        "body": {"position": np.zeros(3), "wall_s": 1.0},
        "held": {"sensory": np.zeros(4)},
        "monotone_recruitment_patch": {
            "adapter_identity": adapter.identity,
            "slopes": slopes.copy(),
        },
    }
    for key in ("fixed_muscles", "lower_budget", "upper_budget", "motor_minimum_slope"):
        saved["monotone_recruitment_patch"][key] = saved["adapter"]["parameters"][
            key
        ].copy()
    new = slopes.copy()
    free = np.setdiff1d(np.arange(96), adapter.fixed_muscles)
    new[free] *= 0.99
    fixed = np.zeros(96, bool)
    fixed[adapter.fixed_muscles] = True
    coeff = np.tile(np.array([1.0, 0.0, 0.0], np.float32), (1660, 1))
    c = {
        **{key: value.copy() for key, value in adapter.calibration.items()},
        "tau": tau.copy(),
        "kappa": kappa.copy(),
        "membership": membership.copy(),
        "slopes": new,
        "fixed": fixed,
        "coefficients32": coeff,
        "coefficients64": coeff.astype(np.float64),
        "source_ids": np.arange(51, dtype=np.int64),
        "edge_ids": np.arange(1660, dtype=np.int64),
        "low": new[membership, np.arange(391), :, 0],
        "high": new[membership, np.arange(391), :, 1],
    }
    return saved, c


def test_native_identity_and_sparse_checkpoint_changes_preserve_whole_state():
    saved, c = payload()
    before = copy.deepcopy(saved)
    expected, allowed = probe.expected_candidate_checkpoint(saved, c)
    probe.layer.require_equal(saved, before, "caller immutable")
    assert expected["adapter"]["hash"] != saved["adapter"]["hash"]
    for key in (
        "bank",
        "activation",
        "original",
        "last_motor",
        "calibration",
        "groups",
    ):
        probe.layer.require_equal(expected["adapter"][key], saved["adapter"][key], key)
    for key in ("body", "held", "network"):
        probe.layer.require_equal(expected[key], saved[key], key)
    assert "adapter" not in allowed and "monotone_recruitment_patch" not in allowed
    assert probe.factor_identities(c)["decoder_factor_sha256"] != probe.array_factor(
        {**c, "slopes": saved["adapter"]["parameters"]["slopes"]}, probe.FACTOR_KEYS
    )


@pytest.mark.parametrize(
    "key",
    ["tau", "kappa", "membership", "moment", "rest", "fixed", "coefficients64", "low"],
)
def test_native_candidate_rejects_changed_fixed_parameters(key):
    saved, c = payload()
    c[key] = c[key].copy()
    c[key].flat[0] = (
        not c[key].flat[0]
        if c[key].dtype == bool
        else c[key].flat[0] + (1 if c[key].dtype.kind in "iu" else 0.01)
    )
    with pytest.raises(ValueError):
        probe.expected_candidate_checkpoint(saved, c)


def test_partial_saves_use_actual_aggregate_failure(tmp_path, monkeypatch):
    from flylab.c.storage import StateStore

    calls = []

    class Component:
        def snapshot(self):
            return {"raw": np.zeros(1)}

    def fail(path, snapshot):
        calls.append(str(path))
        if "adapter" in str(path):
            raise OSError("adapter component failed")

    monkeypatch.setattr(StateStore, "save", fail)
    session = type(
        "Session",
        (),
        {
            "network": Component(),
            "adapter": Component(),
            "body": Component(),
            "control_tick": 0,
        },
    )()
    with pytest.raises(Exception, match="component|partial") as error:
        probe.NativeDependencies().save_partial(tmp_path / "partial", session)
    assert len(calls) == 3 and any(
        "adapter component failed" in n for n in getattr(error.value, "__notes__", [])
    )


def test_sha_gate_precedes_dependencies(tmp_path):
    file = tmp_path / "f.json"
    file.write_text("{}")
    with pytest.raises(ValueError, match="SHA"):
        probe.gate(file, "0" * 64, file, "0" * 64, tmp_path / "out")


@pytest.mark.parametrize("arm", ["neutral", "candidate"])
def test_constructor_unwraps_then_wraps_once_and_reports_owner(
    tmp_path, monkeypatch, arm
):
    saved, c = payload()
    calls = []

    class Owner:
        def close(self):
            calls.append("close")

    class Timescale:
        schema = "generated-timescale"

        @staticmethod
        def from_checkpoint(graph, state):
            calls.append(("timescale", state))
            assert "monotone_recruitment_patch" not in state
            assert probe.layer.equal(state["adapter"], saved["adapter"]["original"])
            return Owner()

    class Monotone:
        @staticmethod
        def from_checkpoint(graph, state):
            calls.append(("neutral", state))
            return Owner()

        def __new__(cls, owner, tau, kappa, slopes, membership):
            calls.append(("monotone", slopes.copy()))
            return owner

    def source(owner, edges, coefficients, allowed):
        calls.append(("source", coefficients.copy()))
        return owner

    def count(arm, callback, ledger, persist, on_return):
        owner = callback()
        on_return(owner)
        return owner

    monkeypatch.setattr(probe.decoder, "MotorTimescaleTrial", Timescale)
    monkeypatch.setattr(probe.decoder, "MonotoneRecruitmentTrial", Monotone)
    monkeypatch.setattr(probe.layer, "SourceFilterTrial", source)
    monkeypatch.setattr(probe.short, "construction_with_body_count", count)
    owners = []
    data = {
        "saved": saved,
        "candidate": c,
        "graph": None,
        "coefficients": c["coefficients64"],
        "selection": {"edge_ids": c["edge_ids"]},
        "allowed_motor_ids": np.arange(343),
    }
    owner = probe.NativeDependencies().construct(
        arm, data, {}, lambda: None, owners.append
    )
    assert owners == [owner] and owner._e288_arm == arm
    assert [item[0] for item in calls] == (
        ["neutral", "source"]
        if arm == "neutral"
        else ["timescale", "monotone", "source"]
    )
    probe.layer.require_equal(
        calls[-1][1], c["coefficients64"], "source factor unchanged"
    )


@pytest.mark.parametrize("kind", ["missing", "boundary", "reversed"])
def test_actual_cycle_requires_three_internal_extrema_in_eligible_bout(kind):
    candidate = metrics(True)
    cycle = candidate["gait"]["events"][probe.LEGS[0]]["cycles"][0]
    if kind == "missing":
        cycle.pop("pep_s")
    elif kind == "boundary":
        cycle["next_aep_s"] = 1.5
    else:
        cycle["pep_s"] = 1.1
    assert not probe.assess_short(metrics(), candidate)["pass"]


@pytest.mark.parametrize(
    "alias", ["final_result", "selection_result", "e287_root_audit"]
)
def test_selection_requires_all_three_independent_matching_receipts(
    tmp_path, monkeypatch, alias
):
    paths = {
        key: key + ".json"
        for key in (
            "final_result",
            "selection_result",
            "e287_root_audit",
            "candidate_metadata",
        )
    }
    paths["candidate"] = "candidate.npz"
    sha = "1" * 64
    for key in ("final_result", "selection_result", "e287_root_audit"):
        value = {
            "schema": "flylab.m3.e287.root-original-audit.v1"
            if key == "e287_root_audit"
            else "flylab.m3.e287.source-decoder-fit.v1",
            "run_valid": True,
            "selection_pass": key != alias,
            "candidate_sha256": sha,
        }
        (tmp_path / paths[key]).write_text(json.dumps(value))
    (tmp_path / paths["candidate_metadata"]).write_text(
        json.dumps({"candidate_sha256": sha, "decoder_identity": "payload"})
    )
    monkeypatch.setattr(probe, "safe", lambda name: tmp_path / name)
    with pytest.raises(ValueError, match="selected"):
        probe.selection_gate(
            {
                "paths": paths,
                "input_sha256": {"candidate.npz": sha},
                "candidate_payload_identity": "payload",
            }
        )


def test_actual_partial_storage_failure_invalidates_health_stop(tmp_path, monkeypatch):
    from flylab.c.storage import StateStore

    calls = []

    def save(path, state):
        calls.append(str(path))
        if "adapter" in str(path):
            raise OSError("actual adapter save failed")

    monkeypatch.setattr(StateStore, "save", save)

    class PartialDeps(FakeDeps):
        def save_partial(self, path, session):
            session.network.snapshot = lambda: {"raw": np.zeros(1)}
            session.body.snapshot = lambda: {"raw": np.zeros(1)}
            session.adapter = type(
                "Adapter", (), {"snapshot": lambda self: {"raw": np.zeros(1)}}
            )()
            return probe.NativeDependencies().save_partial(path, session)

    deps = PartialDeps("health")
    result, error = probe.execute(
        old.freeze(), tmp_path / "out", deps=deps, budget=lambda: None
    )
    assert error is not None and not result["run_valid"] and len(calls) == 3
    assert (
        result["arms"]["candidate"]["health_stop"]
        and result["cost"]["scientific_physics_steps"] == 77
    )
    assert all(s.closed for s in deps.sessions)


def boundary_metrics():
    candidate = metrics(True)
    candidate["gait"]["window"]["bouts"] = [
        {"start_s": 0.55, "end_s": 0.85, "samples": 61}
    ]
    for leg in probe.LEGS:
        candidate["gait"]["events"][leg]["cycles"] = [
            {"aep_s": 0.57, "pep_s": 0.7, "next_aep_s": 0.83}
        ]
    return candidate


def test_exact_61_sample_bout_passes_despite_endpoint_subtraction():
    candidate = boundary_metrics()
    assert 0.85 - 0.55 < 0.3
    result = probe.assess_short(metrics(), candidate)
    assert result["pass"] and result["candidate_longest_eligible_bout_s"] == 0.3


@pytest.mark.parametrize(
    "change",
    [
        {"samples": 60, "end_s": 0.845},
        {"samples": 61.0},
        {"samples": True},
        {"samples": None},
        {"samples": -1},
        {"samples": 201},
        {"end_s": 0.849},
        {"start_s": 0.5501, "end_s": 0.8501},
        {"start_s": 0.49, "end_s": 0.79},
        {"end_s": 1.5},
    ],
)
def test_malformed_bout_samples_and_endpoints_rejected(change):
    candidate = boundary_metrics()
    candidate["gait"]["window"]["bouts"][0].update(change)
    assert not probe.assess_short(metrics(), candidate)["pass"]


def test_generated_cost_receipt():
    assert time.perf_counter() - BEGAN < 90
    print(
        json.dumps(
            {
                "wall_s": time.perf_counter() - BEGAN,
                "fake_counts": old.COUNTS,
                "tiny_solver": 0,
                "original": 0,
                "CUDA": 0,
                "body": 0,
            }
        )
    )
