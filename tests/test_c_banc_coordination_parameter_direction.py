"""Generated E342 only: no original array/hash/gate/model execution."""

from __future__ import annotations

import copy
import hashlib
import io
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import verify_c_banc_m3_coordination_parameter_direction as e


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(e.json_bytes(value))


def qualification_fixture(root):
    """Explicit fabricated metadata for positive/negative validator tests."""
    base = e.BASE + "/e342-sol-metadata"
    source = {
        name: hashlib.sha256(("generated " + name).encode()).hexdigest()
        for name in e.OWNED
    }
    for name in e.OWNED:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated " + name, "utf-8")
    supervisor = root / e.SUPERVISOR
    supervisor.parent.mkdir(parents=True, exist_ok=True)
    supervisor.write_bytes((e.ROOT / e.SUPERVISOR).read_bytes())
    helper = base + "/helper.py"
    (root / helper).parent.mkdir(parents=True, exist_ok=True)
    (root / helper).write_text("# fabricated metadata fixture only\n", "utf-8")
    names = {
        k: base + "/" + k + ".json"
        for k in ("contract", "supervision", "report", "pids")
    }
    report = {
        "schema": "flylab.m3.e342.sol-generated.v1",
        "passed": True,
        "first_error": None,
        "checks": {"fixture": True},
        "source_sha256": source,
        "original_calls": 0,
        "audit_source_path": helper,
        "audit_source_sha256": e.digest(root / helper),
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
    for key in ("stdout", "stderr"):
        (root / (base + "/" + key + ".txt")).write_bytes(b"")
    contract = {
        "source_sha256": {
            **source,
            e.SUPERVISOR: e.SUPERVISOR_SHA,
            helper: report["audit_source_sha256"],
        },
        "commands": [
            {
                "argv": argv,
                "wall_s": 1,
                "stdout": base + "/stdout.txt",
                "stderr": base + "/stderr.txt",
            }
        ],
        "supervisor_output": names["supervision"],
        "original_calls": 0,
        "max_tree_rss_bytes": 2 * e.GIB,
    }
    supervision = {
        "contract_sha256": "placeholder",
        "original_calls": 0,
        "success": True,
        "elapsed_s": 0.2,
        "max_tree_rss_bytes": e.MIB,
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
    }
    pids = {
        "owned_pids": [11, 12],
        "all_owned_pids_ended": True,
        "present_after_return": {"11": False, "12": False},
    }
    return {
        "root": root,
        "source": source,
        "names": names,
        "helper": helper,
        "data": {
            "contract": contract,
            "supervision": supervision,
            "report": report,
            "pids": pids,
        },
    }


def close_fixture(fixture):
    root, names, data = fixture["root"], fixture["names"], fixture["data"]
    write_json(root / names["contract"], data["contract"])
    data["supervision"]["contract_sha256"] = e.digest(root / names["contract"])
    for key in ("supervision", "report", "pids"):
        write_json(root / names[key], data[key])
    paths = [
        *names.values(),
        *data["contract"]["source_sha256"],
        data["contract"]["commands"][0]["stdout"],
        data["contract"]["commands"][0]["stderr"],
    ]
    evidence = {p: e.digest(root / p) for p in paths}
    branch = {k: {"path": p, "sha256": evidence[p]} for k, p in names.items()}
    return branch, evidence


def test_closed_execution_positive(tmp_path):
    fixture = qualification_fixture(tmp_path)
    branch, evidence = close_fixture(fixture)
    assert e.closed_execution(tmp_path, branch, evidence, fixture["source"])["passed"]


@pytest.mark.parametrize(
    "kind",
    [
        "wall",
        "attempt_wall",
        "wall_bool",
        "wall_nan",
        "rss",
        "rss_bool",
        "pid_present",
        "pid_extra",
        "pid_duplicate",
        "pid_mismatch",
        "helper_missing",
        "other_helper",
        "other_output",
        "argv_mismatch",
        "log_tamper",
        "source_tamper",
    ],
)
def test_closed_execution_rejects_coherent_tamper(tmp_path, kind):
    fixture = qualification_fixture(tmp_path)
    c, s, r, p = (
        fixture["data"][k] for k in ("contract", "supervision", "report", "pids")
    )
    if kind == "wall":
        s["elapsed_s"] = 2
    elif kind == "attempt_wall":
        s["attempts"][0]["wall_s"] = 0.3
    elif kind == "wall_bool":
        s["elapsed_s"] = True
    elif kind == "wall_nan":
        s["elapsed_s"] = "NaN"
    elif kind == "rss":
        s["max_tree_rss_bytes"] = 3 * e.GIB
    elif kind == "rss_bool":
        s["max_tree_rss_bytes"] = True
    elif kind == "pid_present":
        p["present_after_return"]["11"] = True
    elif kind == "pid_extra":
        p["present_after_return"]["99"] = False
    elif kind == "pid_duplicate":
        s["owned_pids"] = [11, 11]
    elif kind == "pid_mismatch":
        p["owned_pids"] = [11, 13]
    elif kind == "helper_missing":
        del c["source_sha256"][fixture["helper"]]
    elif kind == "other_helper":
        r["audit_source_path"] = e.OWNED[0]
    elif kind == "other_output":
        c["commands"][0]["argv"][-1] = fixture["names"]["pids"]
    elif kind == "argv_mismatch":
        s["attempts"][0]["argv"] = ["python", "-c", "pass"]
    branch, evidence = close_fixture(fixture)
    if kind == "log_tamper":
        (tmp_path / c["commands"][0]["stdout"]).write_bytes(b"changed")
    if kind == "source_tamper":
        (tmp_path / fixture["helper"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        e.closed_execution(tmp_path, branch, evidence, fixture["source"])


def test_legacy_actual_list_dict_dialects_and_whole_invalid():
    a = {
        "pass": True,
        "diagnostic_run_valid": True,
        "learning_input_approved": True,
        "checks": [{"name": str(i), "pass": True} for i in range(32507)],
    }
    b = {
        "passed": True,
        "run_valid": False,
        "checks": {str(i): True for i in range(3414)},
    }
    decision = {
        "formal_verdict": "invalid_deadline",
        "run_valid": False,
        "effective_direction_qualification": False,
        "effective_continuation_qualification": False,
    }
    expected = {
        "forward_controls": 4,
        "forward_neural_ticks": 80,
        "recompute_controls": 4,
        "recompute_neural_ticks": 80,
        "backward_blocks": 4,
        "backward_block_ticks": 80,
    }
    receipt = {
        "status": "backward_returned",
        "confirm_returned": 2,
        "confirmed": expected,
    }
    for key, value in expected.items():
        receipt[key + "_returned"] = value
        if key != "backward_block_ticks":
            receipt[key + "_requested"] = value
    receipts = {"receipts": {f"window-{c}-neutral": receipt for c in e.WINDOWS}}
    e.validate_legacy_metadata(a, b, decision, receipts)
    bad = copy.deepcopy(a)
    bad["checks"] = {str(i): True for i in range(32507)}
    with pytest.raises(ValueError):
        e.validate_legacy_metadata(bad, b, decision, receipts)
    decision["effective_direction_qualification"] = True
    with pytest.raises(ValueError):
        e.validate_legacy_metadata(a, b, decision, receipts)


@pytest.mark.parametrize(
    "dtype,order", [("<f8", "C"), ("<f8", "F"), ("<f4", "C"), ("<i8", "F")]
)
def test_canonical_npy_serialization_matches_file(tmp_path, dtype, order):
    value = np.array(np.arange(12).reshape(3, 4), dtype=dtype, order=order)
    path = tmp_path / "value.npy"
    np.save(path, value, allow_pickle=False)
    assert e.npy_file_sha(value) == e.digest(path)
    assert e.array_sha(value) != e.digest(path)
    with path.open("rb") as stream:
        header = e.strict_npy_header(stream)
    assert header["fortran_order"] is (order == "F")
    assert header["dtype"] == np.dtype(dtype).str


@pytest.mark.parametrize("dtype", [">f8", "=f8", "O", "U4"])
def test_incompatible_dtype_rejected(dtype):
    with pytest.raises(ValueError):
        e.dtype_metadata(dtype)


def binding_for(root, name, arrays, kind="npz"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "npz":
        np.savez(path, **arrays)
    else:
        np.save(path, arrays["value"], allow_pickle=False)
    return {
        "path": name,
        "sha256": e.digest(path),
        "bytes": path.stat().st_size,
        "kind": kind,
        "selected_schema": {
            k: {"shape": list(v.shape), "dtype": v.dtype.str, "finite": True}
            for k, v in arrays.items()
        },
    }


def test_selected_reader_extras_are_header_only(tmp_path):
    name = e.BASE + "/e342-sol-reader/safe.npz"
    binding = binding_for(
        tmp_path,
        name,
        {"take": np.array([1], np.float32), "extra": np.array([999], np.int64)},
    )
    del binding["selected_schema"]["extra"]
    ledger = []
    got = e.load_selected(tmp_path, binding, ledger, generated=True)
    assert set(got) == {"take"}
    assert ledger[0]["header_only_extras"] == ["extra"]
    assert ledger[0]["numeric_reads"] == ["take"] and ledger[0]["posthash_verified"]


def test_unsafe_extra_rejected_before_decode(tmp_path, monkeypatch):
    name = e.BASE + "/e342-sol-reader/unsafe.npz"
    binding = binding_for(
        tmp_path,
        name,
        {"take": np.array([1], np.float32), "unsafe": np.array([object()], object)},
    )
    del binding["selected_schema"]["unsafe"]

    def no_decode(*args, **kwargs):
        raise AssertionError("payload decoder called")

    monkeypatch.setattr(np, "load", no_decode)
    with pytest.raises(ValueError):
        e.load_selected(tmp_path, binding, [], generated=True)


def test_strict_npy_length_endian_fortran_and_nonfinite(tmp_path):
    name = e.BASE + "/e342-sol-reader/value.npy"
    binding = binding_for(tmp_path, name, {"value": np.array([1], np.float32)}, "npy")
    assert (
        e.load_selected(tmp_path, binding, [], generated=True)["value"].dtype
        == np.float32
    )
    binding["selected_schema"]["value"]["dtype"] = "<f8"
    with pytest.raises(ValueError):
        e.load_selected(tmp_path, binding, [], generated=True)
    binding = binding_for(
        tmp_path,
        e.BASE + "/e342-sol-reader/nonfinite.npy",
        {"value": np.array([np.nan], np.float32)},
        "npy",
    )
    with pytest.raises(ValueError):
        e.load_selected(tmp_path, binding, [], generated=True)


def test_direction_is_actual_float32_and_once_scaled_signed_weights():
    gradient = np.array([2.0, -1.0, 0.25, 0.0], np.float64)
    plus, minus = e.make_direction(gradient)
    assert np.array_equal(plus, np.float32(1 + 0.001 * gradient / 2).astype(np.float64))
    assert np.array_equal(
        minus, np.float32(1 - 0.001 * gradient / 2).astype(np.float64)
    )
    csr = {"data": np.array([-0.2, 0.3, 0.0, -0.0], np.float32)}
    mask = {
        "positions": np.array([0, 1], np.int64),
        "edge_parameters": np.array([0, 1], np.int64),
    }
    selected, report = e.patch_weights(csr, mask, plus)
    assert np.array_equal(
        selected, np.float32(csr["data"][:2] * plus.astype(np.float32)[:2])
    )
    assert np.array_equal(np.signbit(selected), np.signbit(csr["data"][:2]))
    assert report["full_native_palette_count"] == 4
    assert e.make_direction(np.zeros(4, np.float64)) is None


def test_modified_full_palette_guard_counts_zero_signs():
    csr = {"data": np.arange(1, 16385, dtype=np.float32)}
    csr["data"] = np.r_[csr["data"], np.float32(0), np.float32(-0.0)]
    mask = {
        "positions": np.array([0], np.int64),
        "edge_parameters": np.array([0], np.int64),
    }
    _, report = e.patch_weights(csr, mask, np.array([1.0010000467300415], np.float64))
    assert (
        report["full_native_palette_count"] > 16384 and not report["native_deployable"]
    )


def raw_classification():
    plus, minus = e.make_direction(np.ones(4, np.float64))
    g = np.eye(4, dtype=np.float64) * 0.1
    t = g @ (plus - minus)
    return {
        "theta_plus": plus,
        "theta_minus": minus,
        "gtheta_C": g,
        "sum_gtheta_C": g.sum(0),
        "T": t,
        "O": t.copy(),
        "Q": t.copy(),
        "I_C": t * 0.5,
        "I_S": np.zeros(4, np.float64),
        "neutral_loss": np.tile(np.array([1.0, 1.0, 2.0, 1.0]), (4, 1)),
        "minus_loss": np.tile(np.array([1.0, 0.99, 1.99, 1.0]), (4, 1)),
    }


def test_observation_reconciliation_and_tradeoff_separate():
    raw = raw_classification()
    assert e.classify(raw)["direction_qualification"]
    raw["O"] *= 2
    report = e.classify(raw)
    assert (
        report["observed_response"] == "C_decrease"
        and not report["direction_qualification"]
    )
    raw = raw_classification()
    raw["minus_loss"][:, 0] += 0.1
    assert (
        e.classify(raw)["component_tradeoff"]
        and not e.classify(raw)["direction_qualification"]
    )


def test_window_L1_cancellation_negative_components_and_zero_information():
    observed = np.array([1.0, -1.0, 0.01, 0.0], np.float64)
    predicted = np.array([2.0, -2.0, 0.01, 0.0], np.float64)
    report = e.reconciliation(observed, predicted)
    assert report["relative_error"] < 1e-10 and not report["passed"]
    assert e.reconciliation(-np.ones(4), -np.ones(4))["passed"]
    report = e.reconciliation(np.zeros(4), np.zeros(4))
    assert (
        report["relative_error"] is None and report["window_L1_relative_error"] is None
    )
    assert report["null_reason"] == "no_direction_information"


def test_aggregate_T_negative_or_mismatch_is_technical_error():
    raw = raw_classification()
    raw["T"] *= -1
    with pytest.raises(ValueError):
        e.classify(raw)


def test_boundary_previous_linkage_and_gq_scale_finite_difference():
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
    objective = SmoothTemporalObjective(curves, np.zeros((99, 42)), np.zeros((99, 6)))
    boundary = torch.zeros(42, dtype=torch.float64, requires_grad=True)
    q = (
        torch.linspace(-0.03, 0.04, 168, dtype=torch.float64)
        .reshape(4, 42)
        .requires_grad_(True)
    )
    terms = e.window_loss(objective, boundary, q)
    loss = terms["coordination_loss"] / 4
    (gq,) = torch.autograd.grad(loss, q)
    direction = torch.sin(torch.arange(168, dtype=torch.float64) * 0.31).reshape(4, 42)
    direction = direction / torch.linalg.vector_norm(direction)
    for step in (1e-6, 5e-7):
        fd = (
            e.window_loss(objective, boundary, q + step * direction)["coordination_loss"]
            - e.window_loss(objective, boundary, q - step * direction)["coordination_loss"]
        ) / (8 * step)
        assert float(fd.detach()) == pytest.approx(
            float((gq * direction).sum()), rel=1e-6, abs=1e-9
        )
    assert boundary.grad is None
    successor = e.window_loss(objective, boundary, q)["successor_loss"]
    (predecessor,) = torch.autograd.grad(successor, q)
    assert torch.isfinite(predecessor).all()
    assert terms["previous_encoding"]["target"].requires_grad


@pytest.fixture(scope="module")
def native_fixture(tmp_path_factory):
    dimensions = e.TestDimensions()
    previous = os.environ.get("E342_GENERATED_REUSE_ROOT")
    if previous:
        root = Path(previous).resolve()
        root.relative_to(e.ROOT / e.BASE / "e342-sol-first-generated-small-01")
        prefix = e.BASE + "/e342-sol-current-native"
        from tools.qualify_c_banc_m3_body_packet import scan_archive

        order = [
            "reference",
            "actual_csr",
            "motor_configuration",
            "load_parameters",
            "recurrent_mask",
            "e289_roster",
            "calibration",
            "parameters",
        ]
        order += [
            role
            for c in e.WINDOWS
            for role in (f"drive-{c}", f"start-{c}", f"endpoint-{c}")
        ]
        order += ["control_raw", "inverse", "output_mask"]
        bindings = {}
        for role in order:
            kind = "npy" if role in ("inverse", "output_mask") else "npz"
            path = e.generated_path(root, prefix + "/" + role + "." + kind)
            if kind == "npz":
                observed = scan_archive(
                    path, expected_sha256=e.digest(path), descriptor_scope="observation"
                )
                schema = {
                    k: {"shape": v["shape"], "dtype": v["dtype"], "finite": True}
                    for k, v in observed["descriptor"]["members"].items()
                    if k != "source_weights"
                }
            else:
                with path.open("rb") as stream:
                    header = e.strict_npy_header(stream)
                schema = {
                    "value": {
                        "shape": header["shape"],
                        "dtype": header["dtype"],
                        "finite": True,
                    }
                }
            bindings[role] = {
                "path": prefix + "/" + role + "." + kind,
                "kind": kind,
                "bytes": path.stat().st_size,
                "sha256": e.digest(path),
                "selected_schema": schema,
            }
        meta = {
            "neutral": [0.0] * 42,
            "decoder_links": {},
            "model_identity": "generated-native-only",
            "decoder_hash": "generated-native-only",
        }
        for role in ("calibration", "parameters"):
            for key, value in e.load_selected(
                root, bindings[role], [], generated=True
            ).items():
                meta["decoder_links"][role + "." + key] = {
                    "original_pointer": prefix
                    + "/generated-"
                    + role
                    + "-"
                    + key
                    + ".npy",
                    "original_npy_sha256": e.npy_file_sha(value),
                }
        packet = {
            "generated_prefix": prefix,
            "inventory": {
                "bindings": bindings,
                "role_order": order,
                "input_count": 23,
                "selected_value_count": 112,
            },
            "metadata": meta,
        }
    else:
        root = tmp_path_factory.mktemp("e342-native")
        packet = e.synthetic_fixture(
            root, dimensions=dimensions, prefix=e.BASE + "/e342-sol-current-native"
        )
    return root, dimensions, packet


def test_actual_current_native_worker_cpu(native_fixture):
    root, dimensions, packet = native_fixture
    output = []
    terminal, code = e.run_generated(
        root, packet, dimensions=dimensions, printer=output.append
    )
    assert code == 0, terminal["first_error"]
    assert (
        terminal["worker_valid"]
        and terminal["assembly"]["all_neutral_reproduced_before_gradient"]
    )
    assert (
        terminal["assembly"]["gq_neural_work"] == 0
        and terminal["assembly"]["neural_backwards"] == 4
    )
    assert terminal["work"]["backward_blocks_returned"] == 16
    assert terminal["work"]["forward_controls_returned"] == 48
    assert terminal["assembly"]["control_dt"] == 0.00025
    target = root / packet["generated_prefix"] / "diagnosis"
    assert e.read_json(target / "result.json") == terminal
    assert (
        (target / "result.json").read_bytes()
        == (target / "receipts.json").read_bytes()
        == output[0]
    )
    with np.load(target / "raw.npz", allow_pickle=False) as raw:
        assert raw["gq_C"].shape == (4, 4, 42) and raw["gtheta_C"].shape == (4, 4)
        assert raw["neutral_end_bank"].shape == (4, 391, 3)
        assert raw["neutral_end_original_activation"].shape == (4, 96)
        assert np.array_equal(raw["sum_gtheta_C"], raw["gtheta_C"].sum(0))
        assert np.isclose(
            raw["T"].sum(),
            raw["sum_gtheta_C"] @ (raw["theta_plus"] - raw["theta_minus"]),
            rtol=1e-10,
            atol=1e-12,
        )


def fake_bundle(packet, *, zero=False, bad_last=False):
    from tools.banc_m3_checkpointed_rollout import RolloutReceipt
    from tools.banc_m3_smooth_temporal import SmoothTemporalObjective

    objective = SmoothTemporalObjective(
        packet["reference"]["curves"],
        np.zeros((99, 42)),
        np.zeros((99, 6)),
        normalization=packet["reference"]["normalization"],
    )
    for c in e.WINDOWS:
        q = np.linspace(-0.02, 0.03, 168).reshape(4, 42)
        packet[f"endpoint-{c}"]["q"] = q.copy()
        packet[f"endpoint-{c}"]["loss_terms"] = e.loss_vector(
            e.window_loss(
                objective, torch.zeros(42, dtype=torch.float64), torch.tensor(q)
            )
        )[[2, 0, 1, 3]]

    class Native(torch.autograd.Function):
        @staticmethod
        def forward(ctx, theta, base, receipt):
            ctx.receipt = receipt
            ctx.shape = theta.shape
            ctx.zero = zero
            return base + (theta.sum() - len(theta)) * 0.01

        @staticmethod
        def backward(ctx, gradient):
            receipt = ctx.receipt
            for key in (
                "recompute_controls_requested",
                "recompute_controls_returned",
                "backward_blocks_requested",
                "backward_blocks_returned",
            ):
                setattr(receipt, key, 4)
            receipt.recompute_neural_ticks_requested = (
                receipt.recompute_neural_ticks_returned
            ) = 4
            receipt.backward_block_ticks_returned = 4
            receipt.status = "backward_returned"
            value = gradient.sum() * (0 if ctx.zero else 0.01)
            return value.expand(ctx.shape), None, None

    class Rollout:
        device = torch.device("cpu")
        calls = 0

        def run(self, rate, adaptation, bank, original, drives, theta, mask):
            receipt = RolloutReceipt("cpu")
            receipt.forward_controls_requested = receipt.forward_controls_returned = 4
            receipt.forward_neural_ticks_requested = (
                receipt.forward_neural_ticks_returned
            ) = 4
            receipt.status = "forward_returned"
            c = e.WINDOWS[self.calls % 4]
            self.calls += 1
            q = Native.apply(theta, torch.tensor(packet[f"endpoint-{c}"]["q"]), receipt)
            if bad_last and self.calls == 4:
                q = q + 0.1
            endpoint = packet[f"endpoint-{c}"]
            return {
                "q": q,
                "tendon": torch.tensor(endpoint["tendon"]),
                "rate": torch.tensor(endpoint["rate"]),
                "adaptation": torch.tensor(endpoint["adaptation"]),
                "bank": torch.tensor(endpoint["bank"]),
                "original_activation": torch.tensor(endpoint["original_activation"]),
                "receipt": receipt,
            }

    return {
        "rollout": Rollout(),
        "objective": objective,
        "record": {"prohibited_old_calls": 0, "actual_import_sha256": {}},
    }


@pytest.mark.parametrize("branch", ["zero_C_gradient", "native_palette_not_deployable"])
def test_truthful_neutral_only_early_rosters(native_fixture, monkeypatch, branch):
    root, dimensions, wrapper = native_fixture
    packet = e.load_packet(
        root, wrapper["inventory"], dimensions=dimensions, metadata=wrapper["metadata"]
    )
    bundle = fake_bundle(packet, zero=branch == "zero_C_gradient")
    monkeypatch.setattr(e, "assemble", lambda *a, **k: bundle)
    if branch == "native_palette_not_deployable":
        patch = e.patch_weights

        def impossible(*args):
            weights, report = patch(*args)
            return weights, {
                **report,
                "full_native_palette_count": 16385,
                "native_deployable": False,
            }

        monkeypatch.setattr(e, "patch_weights", impossible)
    raw, _receipts, record = e.collect(packet, dimensions=dimensions, device="cpu")
    assert record["branch"] == branch
    assert "q_plus" not in raw and "O" not in raw and "I_C" not in raw
    assert ("theta_plus" in raw) is (branch == "native_palette_not_deployable")
    assert record["work"]["forward_controls_returned"] == 16
    assert record["work"]["backward_blocks_returned"] == 16
    assert e.analyze(raw, record)["observed_response"] == "not_evaluated"


def test_all_neutral_must_reproduce_before_any_backward(native_fixture, monkeypatch):
    root, dimensions, wrapper = native_fixture
    packet = e.load_packet(
        root, wrapper["inventory"], dimensions=dimensions, metadata=wrapper["metadata"]
    )
    bundle = fake_bundle(packet, bad_last=True)
    monkeypatch.setattr(e, "assemble", lambda *a, **k: bundle)
    with pytest.raises(e.CollectionFailure) as caught:
        e.collect(packet, dimensions=dimensions, device="cpu")
    assert "gtheta_C" not in caught.value.raw
    assert e.work_counts(caught.value.receipts)["backward_blocks_requested"] == 0
    assert caught.value.raw["q_neutral"].shape == (4, 4, 42)


def test_fixture_header_row_and_serialization_tamper(native_fixture):
    root, dimensions, packet = native_fixture
    data = e.load_packet(
        root, packet["inventory"], dimensions=dimensions, metadata=packet["metadata"]
    )
    data["start-164"]["source_neural_row_index"] = np.array(164, np.int64)
    with pytest.raises(ValueError):
        e.validate_packet(data, dimensions=dimensions)
    data = e.load_packet(
        root, packet["inventory"], dimensions=dimensions, metadata=packet["metadata"]
    )
    data["calibration"]["moment"][0, 0] += 0.1
    with pytest.raises(ValueError):
        e.validate_packet(data, dimensions=dimensions)


def test_prohibited_old_calls_guard():
    record = {"prohibited_old_calls": 0}

    class StateStore:
        def load(self):
            return 1

    with pytest.raises(RuntimeError), e.prohibited_call_guard(record):
        StateStore().load()
    assert record["prohibited_old_calls"] == 1


def small_worker_raw():
    raw = raw_classification()
    raw.update(
        {
            "plus_loss": raw["neutral_loss"].copy(),
            "gq_C": np.zeros((4, 4, 42)),
            "boundary": np.zeros((4, 42)),
            "q_neutral": np.zeros((4, 4, 42)),
        }
    )
    return raw


@pytest.mark.parametrize(
    "stage", ["raw-save", "classification", "readback", "postpublication", "stdout"]
)
def test_publication_failures_preserve_raw_first_error(tmp_path, monkeypatch, stage):
    raw = small_worker_raw()
    monkeypatch.setattr(
        e,
        "collect",
        lambda *a, **k: (raw, {}, {"branch": "normal", "actual_import_sha256": {}}),
    )

    def fault(at):
        if at == stage:
            raise RuntimeError("injected " + stage)

    def printer(value):
        if stage == "stdout":
            raise OSError("injected stdout")

    path = e.BASE + "/e342-sol-failure-" + stage
    result, code = e.worker(
        tmp_path,
        lambda *a: {},
        output=path,
        device="cpu",
        dimensions=e.TestDimensions(),
        fault=fault,
        printer=printer,
        meter=lambda device: {"tree_rss_bytes": e.MIB},
    )
    assert code == 1 and not result["worker_valid"] and result["first_error"]
    assert not result["effective_direction_qualification"]
    assert (tmp_path / path / "raw.npz").exists()
    assert (
        e.read_json(tmp_path / path / "revocation.json")["first_error"]
        == result["first_error"]
    )
    if (tmp_path / path / "result.json").exists() and stage in (
        "readback",
        "postpublication",
        "stdout",
    ):
        assert e.read_json(tmp_path / path / "result.json")["worker_valid"]


def test_late_deadline_not_reset(tmp_path, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(
        e, "collect", lambda *a, **k: (small_worker_raw(), {}, {"branch": "normal"})
    )

    def printer(value):
        now[0] = 66

    result, code = e.worker(
        tmp_path,
        lambda *a: {},
        output=e.BASE + "/e342-sol-deadline",
        device="cpu",
        dimensions=e.TestDimensions(),
        clock=lambda: now[0],
        printer=printer,
        meter=lambda device: {"tree_rss_bytes": e.MIB},
    )
    assert code == 1 and result["first_error"]["type"] == "TimeoutError"


def test_sampler_precontext_snapshot_close_and_errors():
    watcher = e.GPUMemoryWatch(sample=lambda: 20 * e.MIB)
    assert watcher.start()["precontext_baseline_bytes"] == 20 * e.MIB
    result = watcher.close()
    assert result["sample_count"] >= 1 and not result["owned_thread_alive"]
    assert (
        result["context_included"]
        and not result["unsampled_or_external_work_exclusion_proven"]
    )
    with pytest.raises(ValueError):
        e.GPUMemoryWatch(sample=lambda: True).start()


def test_original_cli_does_not_expose_axes_device_window_limit():
    with pytest.raises(SystemExit):
        e.main(
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


def test_generated_paths_do_not_accept_original_or_traversal(tmp_path):
    with pytest.raises(ValueError):
        e.generated_path(tmp_path, e.OUTPUT)
    with pytest.raises(ValueError):
        e.generated_path(tmp_path, e.BASE + "/e342-sol-test/../bad")


def test_bounded_staging_and_print_quota(tmp_path):
    timer = e.Clock()
    store = e.Store(tmp_path / "store", timer)
    with pytest.raises(ValueError):
        store.put("result.json", b"x" * (256 * e.KIB + 1))
    stream = io.BytesIO()
    bounded = e.BudgetStream(stream, 4, lambda extra: None)
    bounded.write(b"1234")
    with pytest.raises(ValueError):
        bounded.write(b"5")


def test_record_actual_definition_imports():
    path = os.environ.get("E342_GENERATED_IMPORT_REPORT")
    if path:
        target = e.generated_path(e.ROOT, path)
        assert not target.exists()
        target.write_bytes(e.json_bytes(e.actual_imports()))
