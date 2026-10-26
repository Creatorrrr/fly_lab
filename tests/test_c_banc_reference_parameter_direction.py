"""Controlled E356 generated cases; private actual whole-pytest entrypoint."""

# ruff: noqa: E402
# The controlled direct script must add its workspace before passive tool import.

from __future__ import annotations

import copy
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import verify_c_banc_m3_reference_parameter_direction as m


def execute_whole(case_name):
    started = time.perf_counter()
    contract, spec = m.validate_case_contract(case_name, before_execution=True)
    # Set before pytest imports the test module and any scientific libraries.
    early_cpu_threads = {}
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
        early_cpu_threads[name] = os.environ[name]
    environment = m.clean_pytest_environment()
    receipt = {
        "schema": "flylab.m3.e356.pytest-case-receipt.v1",
        "source_sha256": m.own_sources(),
        "case_contract": spec,
        "expected_functions": contract["expected_functions"],
        "empty_ini": contract["empty_ini"],
        "basetemp": contract["basetemp"],
        "basetemp_fresh_before_execution": True,
        "effective_environment": environment,
        "pytest_main_requested": 0,
        "pytest_main_returned": 0,
        "pytest_return_code": None,
        "collected": [],
        "case_paths": {},
        "reports": [],
        "collection_errors": [],
        "deselected": [],
        "first_error": None,
        "secondary_errors": [],
        "complete_current_cases": False,
        "registration_sha256": m.REGISTRATION_SHA,
        "whole_sampler_scope": "before_pytest_import_and_first_CUDA",
        "whole_sampler_started_before_pytest_import": False,
        "whole_sampler_start": None,
        "whole_sampler_closed": None,
        "timing_scope": "CPU perf_counter through NVML/import/pytest/sampler close; excludes receipt publication; CUDA stream events case-local",
    }

    def failure(error):
        value = {"type": type(error).__name__, "message": str(error)[:1024]}
        if receipt["first_error"] is None:
            receipt["first_error"] = value
        else:
            receipt["secondary_errors"].append(value)
        receipt["complete_current_cases"] = False
        print(
            m.json_bytes({"stage": "whole-failure", **value}).decode("utf8"), flush=True
        )

    class Plugin:
        def pytest_collection_modifyitems(self, session, config, items):
            receipt["collected"] = [item.nodeid for item in items]
            receipt["case_paths"] = {
                item.nodeid: item.path.resolve().relative_to(ROOT).as_posix()
                for item in items
            }

        def pytest_collectreport(self, report):
            if report.failed:
                receipt["collection_errors"].append(str(report.longrepr)[:4096])

        def pytest_deselected(self, items):
            receipt["deselected"].extend(item.nodeid for item in items)

        def pytest_runtest_logstart(self, nodeid, location):
            print(
                m.json_bytes(
                    {
                        "stage": "case-start",
                        "nodeid": nodeid,
                        "cpu_perf_counter_elapsed_s": time.perf_counter() - started,
                    }
                ).decode("utf8"),
                flush=True,
            )

        def pytest_runtest_logreport(self, report):
            receipt["reports"].append(
                {
                    "nodeid": report.nodeid,
                    "when": report.when,
                    "outcome": report.outcome,
                    "duration_s": report.duration,
                }
            )
            print(
                m.json_bytes(
                    {
                        "stage": "case-phase",
                        **receipt["reports"][-1],
                        "cpu_perf_counter_elapsed_s": time.perf_counter() - started,
                    }
                ).decode("utf8"),
                flush=True,
            )
            if report.failed:
                # Preserve an actual failure before a later process deadline.
                print(str(report.longrepr)[:2048], flush=True)

    argv = [
        "-q",
        "-s",
        "-p",
        "no:cacheprovider",
        "-c",
        str(m.safe(contract["empty_ini"]["path"])),
        "--noconftest",
        "--basetemp",
        contract["basetemp"],
        *[str(m.safe(name)) for name in contract["complete_suite"]],
    ]
    receipt["actual_pytest_main_argv"] = argv
    receipt["early_science_thread_environment"] = early_cpu_threads
    watch = m.generated_gpu_watch()
    try:
        receipt["whole_sampler_start"] = watch.start()
        m.generated_sampler_check(receipt["whole_sampler_start"])
        receipt["whole_sampler_started_before_pytest_import"] = True
        print(
            m.json_bytes({"stage": "whole-sampler", "event": "started"}).decode("utf8"),
            flush=True,
        )
        import pytest

        receipt["pytest_main_requested"] += 1
        code = int(pytest.main(argv, plugins=[Plugin()]))
        receipt["pytest_main_returned"] += 1
        receipt["pytest_return_code"] = code
        m.complete_case_receipt(receipt, contract["expected_functions"])
        receipt["complete_current_cases"] = True
    except BaseException as error:  # noqa: BLE001 -- Preserve interrupted private entrypoint.
        failure(error)
    finally:
        try:
            print(
                m.json_bytes({"stage": "whole-sampler", "event": "close-start"}).decode(
                    "utf8"
                ),
                flush=True,
            )
            receipt["whole_sampler_closed"] = watch.close()
            m.generated_sampler_check(receipt["whole_sampler_closed"], closed=True)
            print(
                m.json_bytes({"stage": "whole-sampler", "event": "closed"}).decode(
                    "utf8"
                ),
                flush=True,
            )
        except BaseException as error:  # noqa: BLE001 -- Keep first and secondary closure failures.
            failure(error)
        receipt["actual_entrypoint_elapsed_s"] = time.perf_counter() - started
        try:
            m.require(
                receipt["source_sha256"] == m.own_sources(),
                "same current source2 after actual pytest",
            )
        except BaseException as error:  # noqa: BLE001 -- Source closure failure stays in receipt.
            failure(error)
        try:
            raw = m.json_bytes(receipt)
            m.require(len(raw) <= 256 << 10, "case receipt cap")
            with m.safe(contract["receipt_output"]).open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            m.require(
                m.safe(contract["receipt_output"]).read_bytes() == raw,
                "closed case receipt",
            )
        except BaseException as error:  # noqa: BLE001 -- Publication failure cannot qualify.
            failure(error)
    return 0 if receipt["complete_current_cases"] else 1


# Science imports occur only inside the actual private pytest run.
if __name__ != "__main__":
    import numpy as np
    import pytest

    @pytest.fixture(scope="session")
    def cpu(tmp_path_factory):
        trace = m.PhaseTrace(scope="whole-initial-fixture-steps1", cuda_events=True)
        trace.emit("initial-fixture", "preparing")
        import torch

        # synthetic_fixture constructs a CPU objective before assemble sets this.
        torch.set_num_threads(1)
        assert torch.get_num_threads() == 1
        core = m.native()
        root = tmp_path_factory.mktemp("e356-sol-controlled")
        dimensions = core.TestDimensions(steps_per_control=1)
        fixture = core.synthetic_fixture(root, dimensions=dimensions, device="cuda")
        trace.emit("initial-fixture", "done")
        trace.emit("packet-load", "preparing")
        packet = core.load_packet(
            root,
            fixture["inventory"],
            dimensions=dimensions,
            metadata=fixture["metadata"],
        )
        trace.emit("packet-load", "done")
        target = np.stack([packet[f"endpoint-{c}"]["q"] for c in m.WINDOWS]) + 0.01
        raw, receipts, record = m.collect(
            packet,
            target_q=target,
            device="cuda",
            dimensions=dimensions,
            progress=trace,
        )
        record["generated_actual_torch_num_threads"] = torch.get_num_threads()
        assert record["generated_actual_torch_num_threads"] == 1
        return root, dimensions, fixture, packet, target, raw, receipts, record


def test_strict_metadata_and_finite_registration():
    reg = m.metadata_preflight()
    assert len(reg["legacy_inventory"]["role_order"]) == 23
    assert len(reg["raw_contract"]["normal"]) == 26
    for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}'):
        with pytest.raises(ValueError):
            m.strict_json(raw)
    with pytest.raises(TypeError):
        m.json_bytes({"tuple": (1, 2)})
    assert m.json_bytes(m.report_value({"tuple": (1, 2)}))


def test_passive_import_and_pre_array_gate():
    tree = m.ast.parse(m.safe(m.OWNED[0]).read_text("utf8"))
    imports = [n for n in tree.body if isinstance(n, (m.ast.Import, m.ast.ImportFrom))]
    assert all(
        "torch" not in m.ast.unparse(n) and "numpy" not in m.ast.unparse(n)
        for n in imports
    )
    with pytest.raises(ValueError):
        m.validate_supervisor_contract("../invalid.json", phase="pre_array")
    with pytest.raises(ValueError):
        m.validate_supervisor_contract("unused", phase="unknown")


def test_current_whole_case_qualification():
    names = m.ast_cases(m.safe(m.OWNED[1]))
    assert names == sorted(m.registry()["required_owned_test_functions"])
    with pytest.raises((FileNotFoundError, ValueError)):
        m.make_freeze(m.BASE + "/e356-sol-no-qualification.json")


def test_final_supervisor_contract_required_fields(tmp_path, monkeypatch):
    paths = m.contract_paths(m.BASE + "/parent-e346-e356-sol-FIRST-example.json")
    assert paths["pre_spawn_receipt"].endswith(
        "parent-e356-sol-FIRST-example-pre-spawn.json"
    )
    base = tmp_path / m.BASE
    base.mkdir(parents=True)
    # The registration pin cannot be bypassed by a generated-root contract.
    (base / Path(m.REGISTRATION).name).write_bytes(b"{}")
    with pytest.raises(ValueError, match="registration pin"):
        m.validate_supervisor_contract(
            m.BASE + "/parent-e346-e356-sol-FIRST-example.json",
            phase="pre_spawn",
            root=tmp_path,
        )
    assert "--run" in m.original_argv("F", "R")
    assert "--audit" in m.original_argv("F", "R", "audit")
    assert all("dimension" not in arg for arg in m.original_argv("F", "R"))
    # Negative-only generated C fixtures pass a mocked passive metadata boundary.
    # They are never a Q, reservation, pre-spawn witness, or original authority.
    boundary = []

    def generated_metadata(root, timely):
        boundary.append(
            {"original_reads": 0, "science_calls": 0, "generated_mock_only": True}
        )
        return {"original_cumulative_before_s": 9872.876823299503}

    monkeypatch.setattr(m, "metadata_preflight", generated_metadata)
    contract_name = m.BASE + "/parent-e346-e356-sol-FIRST-negative-only.json"
    fake_paths = m.contract_paths(contract_name, tmp_path)
    F, R = {"path": "generated-F.json"}, {"path": "generated-R.json"}
    good_shape = {
        "schema": "flylab.m3.e356.original-supervisor-contract.v1",
        "phase": "E356 original worker",
        "role": "worker",
        "source_sha256": {},
        "commands": [
            {
                "argv": m.original_argv(F["path"], R["path"]),
                "wall_s": 21,
                "stdout": fake_paths["stdout"],
                "stderr": fake_paths["stderr"],
            }
        ],
        "supervisor_output": fake_paths["supervision"],
        "max_tree_rss_bytes": 2 << 30,
        "original_calls": 1,
        "cumulative_before_s": 9872.876823299503,
        "cumulative_limit_s": 9900,
        "outer_wall_s": 22,
        "internal_wall_s": 20,
        "science_wall_s": 18,
        "freeze": F,
        "review": R,
        "reservation": {},
        "output": m.OUTPUT,
        "pre_spawn_receipt_output": fake_paths["pre_spawn_receipt"],
    }
    probes = []
    missing = copy.deepcopy(good_shape)
    del missing["commands"][0]["wall_s"]
    probes.append(missing)
    for bad_wall in (True, None, 20, 21.0):
        value = copy.deepcopy(good_shape)
        value["commands"][0]["wall_s"] = bad_wall
        probes.append(value)
    for field, bad_value in (
        ("commands", [None]),
        ("commands", None),
        ("outer_wall_s", True),
        ("internal_wall_s", None),
        ("science_wall_s", 20),
    ):
        value = copy.deepcopy(good_shape)
        value[field] = bad_value
        probes.append(value)
    for value in probes:
        m.safe(contract_name, tmp_path).write_bytes(m.json_bytes(value))
        with pytest.raises(ValueError):
            m.validate_supervisor_contract(
                contract_name, phase="pre_spawn", root=tmp_path
            )
    assert len(boundary) == len(probes) and all(
        row["original_reads"] == row["science_calls"] == 0 for row in boundary
    )


def test_teacher_clock_and_four_window_alignment():
    tick = np.arange(301, dtype=np.int64)
    label = {
        "command": np.repeat(tick[:, None], 42, axis=1).astype(np.float64),
        "control_tick": tick,
        "control_time_s": tick * 0.005,
    }
    target = m.align_label(label)
    assert target[:, 0, 0].tolist() == [101, 165, 229, 293]
    label["control_time_s"] = label["control_time_s"] + 0.001
    with pytest.raises(ValueError, match="label time"):
        m.align_label(label)


def test_independent_R_loss_and_direct_q_derivative():
    import torch

    fixture = m.strict_json(m.read_descriptor(m.registry()["ROOT_closed_form_fixture"]))
    q = torch.full((4, 42), 3.0, dtype=torch.float64, requires_grad=True)
    loss = ((q - 1).square() / 2).sum() / 672
    (gradient,) = torch.autograd.grad(loss, q)
    assert float(loss) == fixture["R_neutral"] / 4
    assert np.all(
        gradient.numpy() == fixture["gq_numerator"] / fixture["gq_denominator"]
    )


def test_all_neutral_fidelity_precedes_R_backward(cpu):
    _, dimensions, _, packet, target, _, _, record = cpu
    assert record["all_neutral_reproduced_before_gradient"] is True
    broken = copy.deepcopy(packet)
    broken["endpoint-292"]["q"][0, 0] += 1
    with pytest.raises(m.native().CollectionFailure) as caught:
        m.collect(broken, target_q=target, device="cuda", dimensions=dimensions)
    assert caught.value.record["neural_backward_requested"] == 0
    assert (
        m.native().work_counts(caught.value.receipts)["backward_blocks_requested"] == 0
    )
    assert caught.value.raw["q_neutral"].shape[0] == 4


def test_fresh_native_R_parameter_VJP_cpu(cpu):
    import torch
    from scipy.sparse import csr_matrix

    from tools.banc_m3_grouped_current_adjoint import GroupedCurrent
    from tools.banc_m3_ssp_adjoint import SSPAdjointModel

    weights = csr_matrix(np.array([[0.01, -0.02], [0.01, -0.02]], np.float32))
    operator = GroupedCurrent(
        weights, np.arange(4, dtype=np.int64), np.arange(4, dtype=np.int64), 4, "cpu"
    )
    model = SSPAdjointModel(
        operator,
        np.full(2, 0.01, np.float32),
        np.full(2, 0.005, np.float32),
        np.full(2, 7.5, np.float32),
        np.full(2, 200, np.float32),
        np.full(2, 8, np.float32),
        np.full(2, 1 / 0.3, np.float32),
        0.00025,
    )
    gradients = []
    for _ in range(4):
        theta = torch.ones(4, dtype=torch.float64, requires_grad=True)
        assert theta.is_leaf and theta.grad is None
        rate, _ = model.step(
            torch.ones(2),
            torch.full((2,), 0.2),
            torch.full((2,), 20.0),
            theta,
            torch.ones(2),
        )
        q = rate.to(torch.float64).repeat(84).reshape(4, 42)
        R = (q.square() / 2).sum() / 672
        (gq,) = torch.autograd.grad(R, q, retain_graph=True)
        assert torch.equal(gq, q / 672)
        R.backward()
        gradients.append(theta.grad.numpy().copy())
    assert np.any(np.stack(gradients) != 0)
    _, dimensions, _, _, _, raw, receipts, record = cpu
    assert (
        record["neural_backward_requested"] == record["neural_backward_returned"] == 4
    )
    assert raw["gtheta_R"].shape == (4, 4)
    assert np.any(raw["gtheta_R"] != 0)
    assert record["gq_neural_work"] == 0
    assert m.native().work_counts(receipts)["recompute_controls_returned"] == 16
    assert m.validate_raw(raw, "normal", dimensions=dimensions)


def test_actual_f32_direction_and_anatomical_deployability(cpu):
    _, _, _, packet, _, raw, _, record = cpu
    requested, actual = m.make_direction(raw["sum_gtheta_R"])
    assert np.array_equal(actual, requested.astype(np.float32).astype(np.float64))
    selected, palette = m.native().patch_weights(
        packet["actual_csr"], packet["recurrent_mask"], actual
    )
    assert np.array_equal(selected, raw["selected_weights_candidate"])
    assert palette["native_deployable"] is True and record["palette"] == palette


def test_output_quadratic_and_parameter_response_analysis(cpu):
    fixture = m.strict_json(m.read_descriptor(m.registry()["ROOT_closed_form_fixture"]))
    raw = {
        "normalization": np.full(6, 2.0, np.float64),
        "q_neutral": np.full((4, 4, 42), 3.0, np.float64),
        "target_q": np.ones((4, 4, 42), np.float64),
        "gtheta_R": np.full((4, 4), 0.125, np.float64),
        "sum_gtheta_R": np.full(4, 0.5, np.float64),
        "gq_R": np.full((4, 4, 42), 1 / 336, np.float64),
        "R_neutral": np.full(4, 0.5, np.float64),
        "neutral_proxy_loss": np.zeros((4, 4)),
        "candidate_proxy_loss": np.zeros((4, 4)),
    }
    raw["theta_requested"], raw["theta_actual"] = m.make_direction(raw["sum_gtheta_R"])
    raw["q_candidate"] = raw["q_neutral"] + fixture["theta_actual_f32_promoted"] - 1
    raw["R_candidate"] = np.full(
        4, (2 + fixture["theta_actual_f32_promoted"] - 1) ** 2 / 8
    )
    raw["window_starts"] = np.asarray(m.WINDOWS, np.int64)
    answer = m.analyze(raw, {"branch": "normal"})
    assert answer["H356"] == "conditional_support"
    assert sum(answer["O"]) == pytest.approx(fixture["R_delta_exact"], abs=1e-12)
    assert sum(answer["Q"]) == pytest.approx(fixture["Q_and_T_exact"], abs=1e-12)
    assert answer["full_parameter_Jacobian_independently_proven"] is False
    assert m.analyze(cpu[5], cpu[7])["candidate_runtime_adoption"] is False


def test_actual_requested_returned_work_and_partial_failure(cpu):
    work = m.native().work_counts(cpu[6])
    assert work["forward_controls_requested"] == work["forward_controls_returned"] == 32
    assert (
        work["recompute_controls_requested"]
        == work["recompute_controls_returned"]
        == 16
    )
    _, dimensions, _, packet, target, _, _, _ = cpu
    calls = []

    def deadline():
        calls.append(1)
        if len(calls) > 3:
            raise TimeoutError("actual partial deadline")

    with pytest.raises(m.native().CollectionFailure) as caught:
        m.collect(
            packet,
            target_q=target,
            device="cuda",
            dimensions=dimensions,
            timely=deadline,
        )
    assert caught.value.record["neural_backward_requested"] == 0
    m.validate_raw(caught.value.raw, "technical_partial", dimensions=dimensions)


def test_no_signal_and_palette_rejection_no_proposal(cpu):
    _, dimensions, _, packet, _, _, _, _ = cpu
    target = np.stack([packet[f"endpoint-{c}"]["q"] for c in m.WINDOWS])
    raw, receipts, record = m.collect(
        packet, target_q=target, device="cuda", dimensions=dimensions
    )
    assert record["branch"] == "zero_R_gradient" and len(raw) == 15
    assert record["neural_backward_returned"] == 4
    assert m.native().work_counts(receipts)["forward_controls_returned"] == 16
    assert m.analyze(raw, record)["O"] is None
    # An explicit larger generated graph tests the actual full palette rejection.
    bigger = copy.deepcopy(packet)
    count = 16382
    old = bigger["actual_csr"]
    old["data"] = np.r_[old["data"], np.arange(1, count + 1, dtype=np.float32)]
    # Add canonical rows with strictly increasing presynaptic IDs. First8
    # selected edges and their anatomical grouping remain the generated base.
    nodes = dimensions.nodes
    first_row = nodes - (count + nodes - 1) // nodes
    columns = np.tile(np.arange(nodes, dtype=np.int32), (count + nodes - 1) // nodes)
    old["indices"] = np.r_[old["indices"], columns[:count]]
    old["indptr"][first_row:] = 8 + np.minimum(
        np.arange(nodes - first_row + 1, dtype=np.int32) * nodes, count
    )
    original_palette = len(np.unique(old["data"].view(np.uint32)))
    assert original_palette <= 16384
    axes = m.native().TestDimensions(edges=8 + count, selected=8, steps_per_control=1)
    bundle = m.native().assemble(bigger, device="cuda", dimensions=axes)
    import torch

    def tensor(value):
        return torch.as_tensor(value.copy(), device="cuda")

    for c in m.WINDOWS:
        state = bigger[f"start-{c}"]
        with torch.no_grad():
            end = bundle["rollout"].run(
                *(
                    tensor(state[k])
                    for k in ("rate", "adaptation", "bank", "original_activation")
                ),
                tensor(bigger[f"drive-{c}"]["drive"]),
                torch.ones(4, dtype=torch.float64, device="cuda"),
                tensor(bigger["output_mask"]["value"]),
            )
            end["receipt"].confirm()
            for k in (
                "rate",
                "adaptation",
                "bank",
                "original_activation",
                "q",
                "tendon",
            ):
                bigger[f"endpoint-{c}"][k] = end[k].cpu().numpy().copy()
            terms = m.native().window_loss(
                bundle["objective"],
                tensor(state["fixed_original_q_boundary"]),
                end["q"],
            )
            bigger[f"endpoint-{c}"]["loss_terms"] = m.native().loss_vector(terms)[
                [2, 0, 1, 3]
            ]
    palette_target = np.stack([bigger[f"endpoint-{c}"]["q"] for c in m.WINDOWS]) + 0.01
    raw, receipts, record = m.collect(
        bigger, target_q=palette_target, device="cuda", dimensions=axes
    )
    assert record["branch"] == "native_palette_not_deployable" and len(raw) == 18
    assert record["palette"]["full_native_palette_count"] > 16384
    assert record["neural_backward_returned"] == 4
    assert m.native().work_counts(receipts)["forward_controls_returned"] == 16


def test_full_generated_publication_and_late_revocation(cpu, tmp_path):
    output = tmp_path / "e356-sol-publication"
    result = m.publish_generated(
        output, cpu[5], cpu[6], cpu[7], cpu[3]["ledger"], dimensions=cpu[1]
    )
    assert result["qualified"] is True
    manifest = m.strict_json((output / "manifest.json").read_bytes())
    assert manifest["all_fsynced_and_closed"] is True
    assert m.validate_publication_manifest(manifest) is True
    for field, value in (
        ("path", "../outside.npz"),
        ("bytes", True),
        ("bytes", (16 << 20) + 1),
        ("sha256", "invalid"),
    ):
        malformed = copy.deepcopy(manifest)
        malformed["raw"][field] = value
        with pytest.raises(ValueError):
            m.validate_publication_manifest(malformed)
    for key in ("raw", "result", "receipts", "ledger"):
        spec = manifest[key]
        assert m.descriptor(spec["path"], (output / spec["path"]).read_bytes()) == spec

    def late():
        raise OSError("generated late close failure")

    other = tmp_path / "e356-sol-revocation"
    with pytest.raises(OSError):
        m.publish_generated(
            other, cpu[5], cpu[6], cpu[7], [], dimensions=cpu[1], late=late
        )
    revoked = m.strict_json((other / "revocation.json").read_bytes())
    assert (
        revoked["external_success"] is False
        and revoked["scientific_qualification"] is False
    )


def test_resources_absolute_deadline_and_sampler_cleanup(tmp_path, monkeypatch):
    timer = m.Clock(entry=10.0, clock=lambda: 31.0)
    with pytest.raises(TimeoutError):
        timer.check()
    watch = m.native().GPUMemoryWatch(interval_s=0.5, sample=lambda: 1)
    watch.start()
    closed = watch.close()
    m.sampler_check(closed, closed=True)
    assert closed["owned_thread_alive"] is False
    closed["owned_sampler_pids"] = list(range(43))
    with pytest.raises(ValueError):
        m.sampler_check(closed, closed=True)
    store = m.Store(tmp_path / "e356-sol-quota", m.Clock())
    with pytest.raises(ValueError):
        store.put("manifest.json", b"x" * (33 << 10))
    assert list(store.output.iterdir()) == []
    # Explicitly generated zero arrays with original literal header dimensions;
    # these are not copied inputs or a valid original acquisition/qualification.
    schema = m.registry()["raw_contract"]
    generated = {
        key: np.zeros(
            schema["fields"][key]["shape"], dtype=schema["fields"][key]["dtype"]
        )
        for key in schema["zero_R_gradient"]
    }
    generated["normalization"][:] = 1
    generated["window_starts"][:] = m.WINDOWS
    malformed = dict(generated)
    malformed["window_starts"] = malformed["window_starts"].astype(np.float64)
    wrong_header = tmp_path / "e356-sol-generated-wrong-header.npz"
    np.savez_compressed(wrong_header, **malformed)
    extras = tmp_path / "e356-sol-generated-extra-object.npz"
    np.savez_compressed(
        extras, **generated, forbidden=np.array([object()], dtype=object)
    )
    decode_calls = []

    def forbidden_numeric_decode(*args, **kwargs):
        decode_calls.append(1)
        raise AssertionError("numeric decoding must follow every literal header")

    with monkeypatch.context() as scope:
        scope.setattr(np, "load", forbidden_numeric_decode)
        for path, branch in (
            (wrong_header, "zero_R_gradient"),
            (extras, "zero_R_gradient"),
            (wrong_header, "technical_partial"),
            (wrong_header, "unknown"),
        ):
            with pytest.raises(ValueError):
                m.read_completed_raw(path, branch)
    assert decode_calls == []
    nonfinite = dict(generated)
    nonfinite["q_neutral"] = generated["q_neutral"].copy()
    nonfinite["q_neutral"][0, 0, 0] = np.nan
    bad_value = tmp_path / "e356-sol-generated-nonfinite.npz"
    np.savez_compressed(bad_value, **nonfinite)
    with pytest.raises(ValueError):
        m.read_completed_raw(bad_value, "zero_R_gradient")
    bad_clock = dict(generated)
    bad_clock["window_starts"] = np.asarray([100, 164, 228, 293], np.int64)
    with pytest.raises(ValueError, match="window starts"):
        m.validate_raw(bad_clock, "zero_R_gradient")
    bad_norm = dict(generated)
    bad_norm["normalization"] = np.zeros(6, np.float64)
    with pytest.raises(ValueError, match="normalization"):
        m.validate_raw(bad_norm, "zero_R_gradient")


def test_private_pytest_selection_nodeids_and_typed_counts(monkeypatch):
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k forbidden")
    monkeypatch.setenv("PYTEST_PLUGINS", "forbidden")
    env = m.clean_pytest_environment()
    assert env["prior_variable_present"] == {
        "PYTEST_ADDOPTS": True,
        "PYTEST_PLUGINS": True,
    }
    assert "PYTEST_ADDOPTS" not in os.environ and "PYTEST_PLUGINS" not in os.environ
    assert os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    receipt = {
        "effective_environment": env,
        "pytest_main_requested": True,
        "pytest_main_returned": 1,
        "pytest_return_code": 0,
        "collection_errors": [],
        "deselected": [],
        "first_error": None,
        "secondary_errors": [],
    }
    with pytest.raises(ValueError, match="complete pytest"):
        m.complete_case_receipt(receipt, {})


def test_small_native_cuda_R_direction_end_to_end(tmp_path):
    import torch

    torch.set_num_threads(1)
    assert torch.get_num_threads() == 1

    core = m.native()
    dimensions = core.TestDimensions(steps_per_control=20)
    watch = m.generated_gpu_watch()
    trace = m.PhaseTrace(
        scope="native20-case-local-delta-after-session-context", cuda_events=True
    )
    try:
        m.generated_sampler_check(watch.start())
        assert torch.cuda.is_available(), (
            "mandatory actual native CUDA, no skip/fallback"
        )
        trace.emit("native20-fixture", "preparing")
        fixture = core.synthetic_fixture(tmp_path, dimensions=dimensions, device="cuda")
        trace.emit("native20-fixture", "done")
        trace.emit("packet-load", "preparing")
        packet = core.load_packet(
            tmp_path,
            fixture["inventory"],
            dimensions=dimensions,
            metadata=fixture["metadata"],
        )
        trace.emit("packet-load", "done")
        target = np.stack([packet[f"endpoint-{c}"]["q"] for c in m.WINDOWS]) + 0.01
        raw, receipts, record = m.collect(
            packet,
            target_q=target,
            device="cuda",
            dimensions=dimensions,
            progress=trace,
        )
        work = core.work_counts(receipts)
        assert (
            work["forward_controls_returned"] == 32
            and work["forward_neural_ticks_returned"] == 640
        )
        assert (
            work["recompute_controls_returned"] == 16
            and work["recompute_neural_ticks_returned"] == 320
        )
        assert (
            work["backward_blocks_returned"] == 16
            and record["neural_backward_returned"] == 4
        )
        trace.emit("publication", "start")
        result = m.publish_generated(
            tmp_path / "e356-sol-native-complete",
            raw,
            receipts,
            record,
            packet["ledger"],
            dimensions=dimensions,
        )
        trace.emit("publication", "complete")
        assert (
            result["qualified"] is True
            and result["analysis"]["candidate_runtime_adoption"] is False
        )
    finally:
        trace.emit("case-local-sampler", "close-start")
        closed = watch.close()
        m.generated_sampler_check(closed, closed=True)
        trace.emit("case-local-sampler", "closed")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--execute-whole-current-pytest":
        raise SystemExit("only controlled whole current pytest entrypoint")
    raise SystemExit(execute_whole(sys.argv[2]))
