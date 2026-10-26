"""Small generated CPU mathematics/metadata/faults; no native CPU worker."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import verify_c_banc_m3_common_descent_direction as p


def test_proxy_formula_and_explicit_limitations():
    total = np.arange(16, dtype=np.float64).reshape(4, 4) / 100
    c = total * 2
    tq = np.arange(672, dtype=np.float64).reshape(4, 4, 42) / 1000
    result = p.reconstruct(total, tq, c, tq * 3)
    np.testing.assert_allclose(result["gtheta_S_proxy"], total * 10.5, rtol=1e-15)
    np.testing.assert_allclose(result["gq_S_reconstructed"], tq * 9.5, rtol=1e-15)
    assert all(v is False for v in p.LIMITATIONS.values())
    with pytest.raises(FloatingPointError):
        p.reconstruct(np.full((4, 4), 1e308), tq, c, tq)


@pytest.mark.parametrize("scale", [1e-300, 1e-200, 1.0, 1e308])
def test_scaled_unit_avoids_squared_underflow_and_norm_overflow(scale):
    value = np.array([scale, -scale], np.float64)
    unit, norms = p.scaled_unit(value)
    np.testing.assert_allclose(unit, [2**-0.5, -(2**-0.5)], rtol=1e-15)
    assert norms["max_abs"] == scale and norms["scaled_L2"] == pytest.approx(2**0.5)
    assert np.isfinite(unit).all()


def test_bisector_and_actual_f32_delta():
    c, s = np.array([3.0, 0.0, -1.0]), np.array([0.0, 2.0, -1.0])
    raw, info = p.common_direction(c, s)
    u = c / np.linalg.norm(c) + s / np.linalg.norm(s)
    np.testing.assert_allclose(raw["direction"], u / max(abs(u)), rtol=1e-15)
    assert info["branch"] == "normal"
    assert (
        np.dot(c, raw["actual_delta_theta"]) > 0
        and np.dot(s, raw["actual_delta_theta"]) > 0
    )
    np.testing.assert_array_equal(
        raw["theta_plus"], np.float32(1 + 0.001 * raw["direction"]).astype(np.float64)
    )
    assert not np.array_equal(raw["actual_delta_theta"], 0.002 * raw["direction"])


@pytest.mark.parametrize(
    "c,s,branch",
    [
        ([0.0, 0.0], [1.0, 2.0], "zero_component_gradient"),
        ([1.0, 2.0], [-1.0, -2.0], "common_descent_unavailable"),
        (
            [1e-200, 2e-200],
            [2e-200, 1e-200],
            "actual_rounded_common_descent_uninformative",
        ),
        ([1.0, 0.0], [-1.0, 1e-12], "actual_rounded_common_descent_uninformative"),
    ],
)
def test_early_geometry_has_truthful_acquired_roster(c, s, branch):
    raw, record = p.common_direction(np.array(c), np.array(s))
    assert record["branch"] == branch and "q_plus" not in raw
    assert ("theta_plus" in raw) is (
        branch == "actual_rounded_common_descent_uninformative"
    )
    result = p.analyze(raw, record)
    assert result["observed_response"] == "not_evaluated" and result["I_C"] is None
    assert not result["direction_qualification"]


def normal_raw():
    c, s = np.array([2.0, 0.3, -0.2]), np.array([0.5, 1.0, 0.1])
    raw, record = p.common_direction(c, s)
    rows = np.array([0.5, 0.3, -0.1, 0.3])[:, None]
    raw.update(
        gtheta_C=rows * c, gtheta_S_proxy=rows * s, sum_gtheta_C=c, sum_gtheta_S_proxy=s
    )
    for component, vectors in (("C", raw["gtheta_C"]), ("S", raw["gtheta_S_proxy"])):
        observed = (vectors * raw["actual_delta_theta"]).sum(axis=1)
        for key in ("O", "Q", "T"):
            raw[key + "_" + component] = observed.copy()
        raw["I_" + component] = observed / 2
    neutral = np.tile(np.array([3.0, 2.0, 5.0, 0.1]), (4, 1))
    plus, minus = neutral.copy(), neutral.copy()
    for component, column in (("S", 0), ("C", 1)):
        plus[:, column] += 2 * raw["O_" + component]
        minus[:, column] -= 2 * raw["O_" + component]
    raw.update(neutral_loss=neutral, plus_loss=plus, minus_loss=minus)
    return raw, record


def test_joint_actual_reduction_separate_from_proxy_scope():
    raw, record = normal_raw()
    result = p.analyze(raw, record)
    assert result["direction_qualification"] and result["joint_measured_reduction"]
    assert result["S_nonincrease_guard"] and raw["T_C"][2] < 0
    assert all(result[k] is False for k in p.LIMITATIONS)


def test_actual_S_guard_is_not_replaced_by_total_or_proxy_positive_dot():
    raw, record = normal_raw()
    raw["minus_loss"][:, 0] = raw["neutral_loss"][:, 0] + 2e-4
    raw["I_S"] = np.full(4, -5e-5)
    result = p.analyze(raw, record)
    assert result["aggregate_T_S"] > 0 and not result["S_nonincrease_guard"]
    assert (
        not result["direction_qualification"] and not result["joint_measured_reduction"]
    )


def test_signed_aggregate_does_not_hide_window_L1_cancellation():
    result = p.reconciliation(
        np.array([10.0, -9.0, 0.0, 0.0]), np.array([9.0, -8.0, 0.0, 0.0])
    )
    assert result["relative_error"] == 0 and not result["passed"]
    result = p.reconciliation(np.zeros(4), np.zeros(4))
    assert (
        result["relative_error"] is None
        and result["null_reason"] == "no_direction_information"
    )


def test_q_only_gradient_scale_and_live_predecessor_finite_difference():
    from tools.banc_m3_smooth_temporal import SmoothTemporalObjective

    phase = 2 * np.pi * np.arange(128) / 128
    curves = np.stack(
        [
            np.stack(
                [0.1 * np.sin(phase + j * 0.6 + l * 0.2) for j in range(7)], axis=-1
            )
            for l in range(6)
        ]
    )
    objective = SmoothTemporalObjective(
        curves, np.zeros((99, 42)), np.zeros((99, 6)), device="cpu"
    )
    boundary = torch.zeros(42, dtype=torch.float64, requires_grad=True)
    q = (
        torch.linspace(-0.03, 0.04, 168, dtype=torch.float64)
        .reshape(4, 42)
        .requires_grad_(True)
    )
    terms = p.window_loss(objective, boundary, q)
    cg = torch.autograd.grad(terms["coordination_loss"] / 4, q, retain_graph=True)[0]
    sg = torch.autograd.grad(terms["successor_loss"] / 4, q, retain_graph=True)[0]
    total = torch.autograd.grad(terms["training_loss"] / 50, q)[0]
    torch.testing.assert_close(12.5 * total - cg, sg, rtol=1e-10, atol=1e-12)
    direction = torch.sin(torch.arange(168, dtype=torch.float64) * 0.31).reshape(4, 42)
    direction /= torch.linalg.vector_norm(direction)
    for name, gradient in (("coordination_loss", cg), ("successor_loss", sg)):
        for step in (1e-6, 5e-7):
            fd = (
                p.window_loss(objective, boundary, q + step * direction)[name]
                - p.window_loss(objective, boundary, q - step * direction)[name]
            ) / (8 * step)
            assert float(fd.detach()) == pytest.approx(
                float((gradient * direction).sum()), rel=1e-6, abs=1e-9
            )
    assert boundary.grad is None


def test_own_generated_namespace_typed_loader_and_no_old_guard_patch(tmp_path):
    name = p.BASE + "/e343-sol-first-typed/values.npz"
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    np.savez(
        path, value=np.arange(4, dtype=np.float64), ignored=np.array([999], np.int64)
    )
    binding = {
        "path": name,
        "sha256": p.e.digest(path),
        "bytes": path.stat().st_size,
        "kind": "npz",
        "selected_schema": {
            "value": {"shape": [4], "dtype": "float64", "finite": True}
        },
    }
    ledger = []
    result = p.load_selected(tmp_path, binding, ledger, generated=True)
    assert set(result) == {"value"} and ledger
    with pytest.raises(ValueError):
        p.e.generated_path(tmp_path, name)
    bad = copy.deepcopy(binding)
    bad["selected_schema"]["value"]["dtype"] = ">f8"
    with pytest.raises(ValueError):
        p.load_selected(tmp_path, bad, [], generated=True)
    bad["path"] = p.BASE + "/e343-common-descent-direction/raw.npz"
    with pytest.raises(ValueError):
        p.load_selected(tmp_path, bad, [], generated=True)


def proof_fixture(root):
    base = p.BASE + "/e343-sol-first-metadata"
    current = {}
    for name in p.OWNED:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("fabricated metadata source " + name).encode())
        current[name] = p.e.digest(path)
    supervisor = root / p.e.SUPERVISOR
    supervisor.parent.mkdir(parents=True, exist_ok=True)
    supervisor.write_bytes((p.ROOT / p.e.SUPERVISOR).read_bytes())
    helper = base + "/helper.py"
    (root / helper).parent.mkdir(parents=True, exist_ok=True)
    (root / helper).write_bytes(b"# fabricated metadata only\n")
    names = {
        key: base + "/" + key + ".json"
        for key in ("contract", "supervision", "report", "pids")
    }
    argv = [
        ".venv/Scripts/python.exe",
        "-X",
        "utf8",
        "-B",
        helper,
        "--output",
        names["report"],
    ]
    sources = {
        **current,
        helper: p.e.digest(root / helper),
        p.e.SUPERVISOR: p.e.SUPERVISOR_SHA,
    }
    for key in ("stdout", "stderr"):
        (root / (base + "/" + key + ".txt")).write_bytes(b"")
    data = {
        "contract": {
            "source_sha256": sources,
            "commands": [
                {
                    "argv": argv,
                    "wall_s": 1,
                    "stdout": base + "/stdout.txt",
                    "stderr": base + "/stderr.txt",
                }
            ],
            "original_calls": 0,
            "supervisor_output": names["supervision"],
            "max_tree_rss_bytes": 2 * p.e.GIB,
        },
        "supervision": {
            "contract_sha256": "placeholder",
            "original_calls": 0,
            "success": True,
            "elapsed_s": 0.2,
            "max_tree_rss_bytes": p.e.MIB,
            "owned_pids": [11, 12],
            "attempts": [
                {
                    "argv": argv,
                    "requested": True,
                    "exit_code": 0,
                    "killed": False,
                    "wall_s": 0.1,
                    "pids": [11, 12],
                }
            ],
        },
        "report": {
            "passed": True,
            "checks": {"fabricated_only": True},
            "first_error": None,
            "original_calls": 0,
            "source_sha256": current,
            "audit_source_path": helper,
            "audit_source_sha256": sources[helper],
        },
        "pids": {
            "owned_pids": [11, 12],
            "present_after_return": {"11": False, "12": False},
            "all_owned_pids_ended": True,
            "pid_create_times": {"11": 1.2, "12": 1.3},
        },
    }
    return names, current, data


def close_fixture(root, names, data):
    for key in ("contract", "supervision", "report", "pids"):
        if key == "supervision":
            data[key]["contract_sha256"] = p.e.digest(root / names["contract"])
        (root / names[key]).write_bytes(p.e.json_bytes(data[key]))
    paths = [
        *names.values(),
        *data["contract"]["source_sha256"],
        data["contract"]["commands"][0]["stdout"],
        data["contract"]["commands"][0]["stderr"],
    ]
    evidence = {name: p.e.digest(root / name) for name in paths}
    branch = {
        key: {"path": name, "sha256": evidence[name]} for key, name in names.items()
    }
    return branch, evidence


def test_completed_producer_metadata_positive(tmp_path):
    names, current, data = proof_fixture(tmp_path)
    branch, evidence = close_fixture(tmp_path, names, data)
    assert p.closed_execution(tmp_path, branch, evidence, current)["passed"]


@pytest.mark.parametrize(
    "fault",
    [
        "wall",
        "RSS",
        "presence",
        "argv",
        "helper",
        "source",
        "creation_missing",
        "creation_bool",
        "creation_nan",
    ],
)
def test_producer_coherent_resource_PID_helper_byte_tamper(tmp_path, fault):
    names, current, data = proof_fixture(tmp_path)
    if fault == "wall":
        data["supervision"]["elapsed_s"] = 2
    elif fault == "RSS":
        data["supervision"]["max_tree_rss_bytes"] = 3 * p.e.GIB
    elif fault == "presence":
        data["pids"]["present_after_return"]["11"] = True
    elif fault == "argv":
        data["supervision"]["attempts"][0]["argv"] = ["python", "-c", "pass"]
    elif fault == "helper":
        data["report"]["audit_source_path"] = p.OWNED[0]
    elif fault == "creation_missing":
        del data["pids"]["pid_create_times"]["11"]
    elif fault == "creation_bool":
        data["pids"]["pid_create_times"]["11"] = True
    elif fault == "creation_nan":
        data["pids"]["pid_create_times"]["11"] = "NaN"
    branch, evidence = close_fixture(tmp_path, names, data)
    if fault == "source":
        (tmp_path / p.OWNED[0]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        p.closed_execution(tmp_path, branch, evidence, current)


@pytest.mark.parametrize(
    "stage", ["raw-save", "classification", "readback", "postpublication", "stdout"]
)
def test_publication_failure_preserves_acquired_raw_and_first_error(
    tmp_path, monkeypatch, stage
):
    raw = {"component_max_abs": np.zeros(2)}
    monkeypatch.setattr(
        p, "collect", lambda *a, **k: (raw, {}, {"branch": "zero_component_gradient"})
    )

    def fault(at):
        if at == stage:
            raise RuntimeError("injected " + at)

    def printer(value):
        if stage == "stdout":
            raise OSError("injected stdout")

    output = p.BASE + "/e343-sol-first-fault-" + stage
    result, code = p.worker(
        lambda *a: {},
        root=tmp_path,
        output=output,
        device="cpu",
        dimensions=p.TestDimensions(),
        fault=fault,
        printer=printer,
        meter=lambda device: {"tree_rss_bytes": p.e.MIB},
    )
    assert code == 1 and result["first_error"] and not result["worker_valid"]
    assert (tmp_path / output / "raw.npz").exists()
    assert (
        p.e.read_json(tmp_path / output / "revocation.json")["first_error"]
        == result["first_error"]
    )
    assert not result["effective_direction_qualification"]


def test_successful_result_receipt_stdout_bytes_and_late_revocation(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        p,
        "collect",
        lambda *a, **k: (
            {"component_max_abs": np.zeros(2)},
            {},
            {"branch": "zero_component_gradient"},
        ),
    )
    captured = []
    output = p.BASE + "/e343-sol-first-success"
    result, code = p.worker(
        lambda *a: {},
        root=tmp_path,
        output=output,
        dimensions=p.TestDimensions(),
        device="cpu",
        printer=captured.append,
        meter=lambda device: {"tree_rss_bytes": p.e.MIB},
    )
    assert (
        code == 0
        and captured[0]
        == p.e.json_bytes(result)
        == (tmp_path / output / "result.json").read_bytes()
    )
    assert captured[0] == (tmp_path / output / "receipts.json").read_bytes()
    now = [0.0]

    def late(value):
        now[0] = 36.0

    result, code = p.worker(
        lambda *a: {},
        root=tmp_path,
        output=p.BASE + "/e343-sol-first-late",
        dimensions=p.TestDimensions(),
        device="cpu",
        clock=lambda: now[0],
        printer=late,
        meter=lambda device: {"tree_rss_bytes": p.e.MIB},
    )
    assert code == 1 and result["first_error"]["type"] == "TimeoutError"


def test_new_clock_limits_without_old_clock_mutation():
    now = [0.0]
    own, old = p.Clock(lambda: now[0]), p.e.Clock(lambda: now[0])
    now[0] = 26.0
    with pytest.raises(TimeoutError):
        own.check()
    old.check()
    now[0] = 29.0
    own.finalize()
    assert own.final_deadline == 34.0
    now[0] = 34.1
    with pytest.raises(TimeoutError):
        own.check("final")


@pytest.mark.parametrize("override", ["clock", "printer", "meter", "root"])
def test_original_API_overrides_rejected_before_loader(tmp_path, override):
    kwargs = {override: tmp_path if override == "root" else lambda *a: None}
    with pytest.raises(ValueError):
        p.worker(lambda *a: pytest.fail("original loader called"), **kwargs)


def test_literal_CLI_axes_override_rejected():
    with pytest.raises(SystemExit):
        p.main(
            [
                "--run",
                "--freeze-sha256",
                "0" * 64,
                "--review-sha256",
                "0" * 64,
                "--device",
                "cpu",
            ]
        )


def test_CPU_fake_sampler_preserves_resource_caveat():
    watch = p.GPUMemoryWatch(sample=lambda: 20 * p.e.MIB)
    assert watch.start()["precontext_baseline_bytes"] == 20 * p.e.MIB
    closed = watch.close()
    assert not closed["owned_thread_alive"] and closed["context_included"]
    assert not closed["unsampled_or_external_work_exclusion_proven"]


def metadata_inputs():
    get = lambda name: p.e.read_json(p.ROOT / p.BASE / name)
    return [
        get("parent-e343-selected-input-inventory-reflected.json"),
        get("parent-e342-selected-input-inventory.json"),
        get("e333-smooth-efficacy-direction/result.json"),
        get("e333-smooth-efficacy-direction/receipts.json"),
        get("e333-smooth-efficacy-direction/output-manifest.json"),
        get("parent-e333-original-raw-audit.json"),
        get("parent-e333-final-decision.json"),
        get("e342-coordination-parameter-direction/result.json"),
        get("parent-e342-final-decision.json"),
        get("parent-e342-audit-report.json"),
        get("e342-coordination-parameter-direction/manifest.json"),
    ]


def test_actual_readonly_producer_JSON_dialects_no_array_reads():
    data = metadata_inputs()
    p.validate_saved_metadata(*data)
    get = lambda name: p.e.read_json(p.ROOT / p.BASE / name)
    registration = get("parent-e343-reflected-design-registration.json")
    p.validate_prior_completion(
        data[8],
        get("parent-e342-original-external-completion.json"),
        get("parent-e342-audit-external-completion.json"),
        registration,
    )
    pins = p.producer_pins(registration, data[0])
    assert (
        pins[p.BASE + "/e332-smooth-temporal-saved/output-manifest.json"]
        == "acfe3cde1dccd8aa6b2e2dc0b35697b91c245ca82229cf47606ac5029f5f16f0"
    )


@pytest.mark.parametrize(
    "fault",
    [
        "whole_recovery",
        "partial_location",
        "mean50",
        "run_gradient",
        "receipt",
        "C_raw",
        "source",
    ],
)
def test_historical_metadata_tamper_does_not_launder_whole_or_local_VJP(fault):
    data = metadata_inputs()
    if fault == "whole_recovery":
        data[6]["run_valid"] = True
    elif fault == "partial_location":
        data[2]["runs"] = data[2]["partial"].pop("runs")
    elif fault == "mean50":
        data[2]["partial"]["window_starts"] = list(p.WINDOWS)
    elif fault == "run_gradient":
        data[2]["partial"]["runs"]["window-100-neutral"]["gradient_file"] = (
            "another.npz"
        )
    elif fault == "receipt":
        data[3]["receipts"]["window-100-neutral"]["backward_blocks_returned"] = 0
    elif fault == "C_raw":
        data[0]["bindings"]["audited_C_vjp"]["sha256"] = "0" * 64
    elif fault == "source":
        data[9]["source_sha256"][p.E342_SOURCE] = "0" * 64
    with pytest.raises((ValueError, KeyError)):
        p.validate_saved_metadata(*data)


def fake_no_back_packet():
    from tools.banc_m3_checkpointed_rollout import RolloutReceipt
    from tools.banc_m3_smooth_temporal import SmoothTemporalObjective

    phase = 2 * np.pi * np.arange(128) / 128
    curves = np.stack(
        [
            np.stack(
                [0.1 * np.sin(phase + j * 0.6 + l * 0.2) for j in range(7)], axis=-1
            )
            for l in range(6)
        ]
    )
    objective = SmoothTemporalObjective(
        curves, np.zeros((99, 42)), np.zeros((99, 6)), device="cpu"
    )
    packet = {
        "reference": objective.canonical_constants,
        "output_mask": {"value": np.ones(400, np.float32)},
        "actual_csr": {"data": np.tile(np.array([0.01, -0.02], np.float32), 4)},
        "recurrent_mask": {
            "positions": np.arange(8),
            "edge_parameters": np.tile(np.arange(4), 2),
        },
    }
    cg, sg, gt, gtotq = [], [], [], []
    for index, c in enumerate(p.WINDOWS):
        q = (
            torch.linspace(-0.03, 0.04, 168, dtype=torch.float64)
            .reshape(4, 42)
            .requires_grad_(True)
        )
        boundary = torch.zeros(42, dtype=torch.float64)
        terms = p.window_loss(objective, boundary, q)
        cg.append(
            torch.autograd.grad(terms["coordination_loss"] / 4, q, retain_graph=True)[
                0
            ].numpy()
        )
        sg.append(
            torch.autograd.grad(terms["successor_loss"] / 4, q, retain_graph=True)[
                0
            ].numpy()
        )
        gtotq.append(torch.autograd.grad(terms["training_loss"] / 50, q)[0].numpy())
        gparam = np.array([1.0, 0.3, 0.2, 0.1]) / 4
        sparam = np.array([0.2, 1.0, 0.1, 0.3]) / 4
        gt.append(gparam)
        packet[f"legacy_total_gradient-{c}"] = {"gradient": (gparam + sparam) / 12.5}
        packet[f"legacy_total_gq-{c}"] = {"gq": gtotq[-1]}
        packet[f"start-{c}"] = {
            "rate": np.zeros(400, np.float32),
            "adaptation": np.zeros(400, np.float32),
            "bank": np.zeros((391, 3)),
            "original_activation": np.zeros(96),
            "fixed_original_q_boundary": np.zeros(42),
        }
        packet[f"endpoint-{c}"] = {
            **packet[f"start-{c}"],
            "q": q.detach().numpy(),
            "tendon": np.zeros((4, 6)),
            "loss_terms": p.loss_vector(terms)[[2, 0, 1, 3]],
        }
        packet[f"drive-{c}"] = {"drive": np.zeros((4, 400), np.float32)}
    packet["audited_C_vjp"] = {
        "gtheta_C": np.stack(gt),
        "sum_gtheta_C": np.stack(gt).sum(axis=0),
        "gq_C": np.stack(cg),
    }

    class Rollout:
        device = torch.device("cpu")
        calls = 0
        bad_last = False

        def run(self, rate, adaptation, bank, original, drive, theta, mask):
            assert not torch.is_grad_enabled() and not theta.requires_grad
            c = p.WINDOWS[self.calls % 4]
            self.calls += 1
            endpoint = packet[f"endpoint-{c}"]
            q = torch.tensor(endpoint["q"]) + (theta.sum() - 4) * 0.01
            if self.bad_last and self.calls == 4:
                q += 0.1
            receipt = RolloutReceipt("cpu")
            receipt.forward_controls_requested = receipt.forward_controls_returned = 4
            receipt.forward_neural_ticks_requested = (
                receipt.forward_neural_ticks_returned
            ) = 4
            receipt.status = "forward_returned"
            return {
                "q": q,
                "tendon": torch.tensor(endpoint["tendon"]),
                "rate": rate,
                "adaptation": adaptation,
                "bank": bank,
                "original_activation": original,
                "receipt": receipt,
            }

    rollout = Rollout()
    return packet, {
        "rollout": rollout,
        "objective": objective,
        "record": {"actual_import_sha256": {}, "prohibited_old_calls": 0},
    }


def test_current_no_grad_neutral_plus_minus_receipts_zero_neural_backwards(monkeypatch):
    packet, bundle = fake_no_back_packet()
    monkeypatch.setattr(p, "assemble", lambda *a, **k: bundle)
    raw, receipts, record = p.collect(
        packet, device="cpu", dimensions=p.TestDimensions()
    )
    assert (
        record["branch"] == "normal"
        and record["all_neutral_reproduced_before_direction"]
    )
    assert (
        record["work"]["forward_controls_returned"] == 48
        and record["new_neural_backwards"] == 0
    )
    assert all(
        value == 0
        for key, value in record["work"].items()
        if key.startswith(("backward", "recompute"))
    )
    assert (
        raw["q_plus"].shape == raw["q_minus"].shape == (4, 4, 42)
        and len(receipts) == 12
    )


def test_neutral_mismatch_retains_four_returns_before_proxy_or_arms(monkeypatch):
    packet, bundle = fake_no_back_packet()
    bundle["rollout"].bad_last = True
    monkeypatch.setattr(p, "assemble", lambda *a, **k: bundle)
    with pytest.raises(p.e.CollectionFailure) as caught:
        p.collect(packet, device="cpu", dimensions=p.TestDimensions())
    assert caught.value.raw["q_neutral"].shape == (4, 4, 42)
    assert "gtheta_C" not in caught.value.raw and "q_plus" not in caught.value.raw
    assert p.e.work_counts(caught.value.receipts)["backward_blocks_requested"] == 0


@pytest.mark.parametrize(
    "branch",
    [
        "zero_component_gradient",
        "common_descent_unavailable",
        "native_palette_not_deployable",
    ],
)
def test_current_early_no_invented_arms_or_recompute(monkeypatch, branch):
    packet, bundle = fake_no_back_packet()
    if branch == "zero_component_gradient":
        packet["audited_C_vjp"]["gtheta_C"].fill(0)
        packet["audited_C_vjp"]["sum_gtheta_C"].fill(0)
    elif branch == "common_descent_unavailable":
        for c in p.WINDOWS:
            packet[f"legacy_total_gradient-{c}"]["gradient"].fill(0)
    else:
        patch = p.patch_weights

        def impossible(*args):
            weights, record = patch(*args)
            return weights, {
                **record,
                "full_native_palette_count": 16385,
                "native_deployable": False,
            }

        monkeypatch.setattr(p, "patch_weights", impossible)
    monkeypatch.setattr(p, "assemble", lambda *a, **k: bundle)
    raw, _, record = p.collect(packet, device="cpu", dimensions=p.TestDimensions())
    assert record["branch"] == branch and "q_plus" not in raw and "O_C" not in raw
    assert (
        record["work"]["forward_controls_returned"] == 16
        and bundle["rollout"].calls == 4
    )
    assert ("selected_weights_plus" in raw) is (
        branch == "native_palette_not_deployable"
    )


def test_record_current_CPU_definition_imports():
    import os

    name = os.environ.get("E343_IMPORT_REPORT")
    if name:
        Path(name).write_bytes(p.e.json_bytes(p.e.actual_imports()))
