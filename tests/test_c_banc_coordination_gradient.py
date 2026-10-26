"""Generated CPU math/reader/provenance/publication qualification, original0."""

from __future__ import annotations

import copy
import io
import json
import shutil

import numpy as np
import pytest

from tools import analyze_c_banc_m3_coordination_gradient as m


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(m.json_bytes(value))


@pytest.fixture(scope="session")
def samples(tmp_path_factory):
    root = tmp_path_factory.mktemp("arrays")
    m.generated_root(root)
    phase = 2 * np.pi * np.arange(128, dtype=np.float64) / 128
    # An independent regular generated curve; no saved numerical input.
    curves = np.empty((6, 128, 7), dtype="<f8")
    for leg in range(6):
        for joint in range(7):
            curves[leg, :, joint] = (0.7 + 0.03 * joint) * np.sin(
                phase + 0.11 * joint
            ) + 0.05 * leg
    coordinate = (
        (np.arange(128, dtype=np.float64) / 128 + 2.382760007990865 * 0.005) % 1
    ) * 128
    lower = np.floor(coordinate).astype(np.int64)
    future = curves[:, lower] + (coordinate - lower)[None, :, None] * (
        curves[:, (lower + 1) % 128] - curves[:, lower]
    )
    n = np.mean((future - curves) ** 2, axis=(1, 2))
    reference = {
        "curves": curves,
        "future_anchors": future,
        "normalization": n,
        "temperature": 7 * n,
        "phase_basis": np.column_stack((np.cos(phase), np.sin(phase))),
    }
    trajectories, bindings = {}, {}
    ref_path = root / "reference.npz"
    np.savez(ref_path, **reference, ignored_numeric=np.arange(7, dtype=np.int64))
    for role_index, role in enumerate(m.ROLES):
        rows = np.arange(201, dtype=np.float64)
        q = np.empty((201, 6, 7), dtype="<f8")
        for leg in range(6):
            angle = 0.033 * rows + 0.27 * leg + 0.17 * role_index
            for joint in range(7):
                q[:, leg, joint] = (0.7 + 0.03 * joint) * np.sin(
                    angle + 0.11 * joint
                ) + 0.05 * leg
        q = q.reshape(201, 42)
        delta = q.reshape(201, 6, 1, 7) - curves[None]
        logits = -np.sum(delta**2, axis=-1) / reference["temperature"][None, :, None]
        w = np.exp(logits - logits.max(axis=-1, keepdims=True))
        w /= w.sum(axis=-1, keepdims=True)
        z = np.einsum("tls,sa->tla", w, reference["phase_basis"])
        pair = np.column_stack(
            [1 - (z[1:, a] * z[1:, b]).sum(axis=-1) for a, b in m.PAIRS]
        )
        trajectories[role] = {
            "q": q,
            "z": z,
            "R": np.linalg.norm(z, axis=-1),
            "coordination_pair_loss": pair.mean(axis=0),
        }
        np.savez(
            root / (role + ".npz"),
            **trajectories[role],
            unselected_integer=np.arange(11, dtype=np.int32),
        )
    for role in ("reference", *m.ROLES):
        original = m.original_bindings()[role]
        bindings[role] = {
            "path": role + ".npz",
            **m.pin(root / (role + ".npz")),
            "selected_schema": original["selected_schema"],
            "descriptor": None,
            "header_observation_scope": "observation",
        }
    return root, reference, trajectories, bindings


@pytest.fixture(scope="session")
def analyzed(samples):
    _, ref, trajectories, _ = samples
    arrays, report = m.analyze_trajectory(ref, trajectories["live"])
    return arrays, report


def test_actual_primary_analytic_fd(samples, analyzed):
    _, ref, trajectories, _ = samples
    arrays, report = analyzed
    boundary, live = m.split_windows(trajectories["live"]["q"])
    assert len(m.finite_difference_checks(ref, boundary, live, arrays)) == 12
    for name in ("g_successor", "g_coordination", "g_total"):
        np.testing.assert_allclose(
            arrays[name], arrays["analytic_" + name], rtol=1e-8, atol=2e-10
        )
    np.testing.assert_allclose(
        arrays["g_total"],
        arrays["g_successor"] + arrays["g_coordination"],
        rtol=1e-8,
        atol=2e-10,
    )
    assert report["gradient_reconciled"] and report["window_count"] == 50
    assert report["coefficients"] == {"S": 1, "C": 1}
    assert len(report["per_window"]) == 50 and list(report["per_leg"]) == list(m.LEGS)
    assert report["placeholder_startup_unused"] and not report["startup_guard_called"]


def test_actual_four_pair_window_scaling(samples, analyzed):
    _, reference, trajectories, _ = samples
    arrays, _ = analyzed
    b, x = m.split_windows(trajectories["live"]["q"])
    objective = m.make_objective(reference)
    boundary = m.torch.tensor(b[0].copy(), dtype=m.torch.float64, device="cpu")
    live = m.torch.tensor(x[0], dtype=m.torch.float64, device="cpu", requires_grad=True)
    terms = objective.evaluate_pairs(m.torch.cat((boundary[None], live[:3])), live)
    gradient = m.torch.autograd.grad(terms["training_loss"] / 50, live)[0].numpy()
    np.testing.assert_allclose(gradient, arrays["g_total"][0], rtol=1e-8, atol=2e-10)
    assert not np.allclose(gradient * 50, arrays["g_total"][0], rtol=1e-8, atol=2e-10)


def test_detached_boundary_and_internal_previous(samples, analyzed):
    _, ref, trajectories, _ = samples
    arrays, _ = analyzed
    b, x = m.split_windows(trajectories["live"]["q"])
    before = b.copy()
    changed = x.copy()
    changed[0, 3] += 0.2
    later = m.numpy_gradients(ref, b, changed)
    np.testing.assert_array_equal(b, before)
    np.testing.assert_array_equal(later["g_total"][1], arrays["analytic_g_total"][1])
    assert not b.flags.writeable and not np.shares_memory(b, x)
    np.testing.assert_array_equal(arrays["successor_previous_internal_path"][:, 0], 0)
    np.testing.assert_allclose(
        arrays["successor_next_path"][:, :3]
        + arrays["successor_previous_internal_path"][:, 1:],
        arrays["analytic_g_successor"][:, :3],
    )
    assert (
        np.max(np.abs(arrays["analytic_g_successor"] - arrays["successor_next_path"]))
        > 1e-5
    )
    # Full-trace rolling boundary would wrongly let x[0,3] change window1.
    bad_boundary = before.copy()
    bad_boundary[1] = changed[0, 3]
    bad = m.numpy_gradients(ref, bad_boundary, changed)
    assert np.max(np.abs(bad["g_total"][1] - later["g_total"][1])) > 1e-5


def test_all_fifteen_pairs(samples, analyzed):
    _, _, trajectories, _ = samples
    arrays, report = analyzed
    z = trajectories["live"]["z"][1:]
    expected = np.column_stack(
        [1 - (z[:, a] * z[:, b]).sum(axis=1) for a in range(6) for b in range(a + 1, 6)]
    )
    np.testing.assert_allclose(arrays["pair_losses"], expected, rtol=1e-10, atol=1e-10)
    assert len(m.PAIRS) == 15
    assert report["losses"]["coordination_loss"] == pytest.approx(
        expected.mean(), abs=1e-10
    )
    assert abs(expected.mean() - expected[:, [2, 7, 11]].mean()) > 1e-3


@pytest.mark.parametrize("key", list(m.REF_SPEC))
def test_actual_constants_reject_mismatch(samples, key):
    _, reference, _, _ = samples
    bad = {k: a.copy() for k, a in reference.items()}
    bad[key].flat[0] += 0.001
    with pytest.raises(ValueError):
        m.make_objective(bad)


def test_direction_literal_shape_and_unit_norm():
    directions = m.fixed_fd_directions()
    for direction, factor, fn in zip(
        directions, (0.37, 0.53), (np.sin, np.cos), strict=True
    ):
        expected = fn(factor * (np.arange(8400, dtype=np.float64) + 1))
        np.testing.assert_array_equal(
            direction.ravel(), expected / np.linalg.norm(expected)
        )
        assert direction.shape == (50, 4, 42) and np.linalg.norm(
            direction
        ) == pytest.approx(1)


@pytest.mark.parametrize(
    "s,c,branch,reason",
    [
        (1.0, 0.0, "output_gradient_inconclusive", "exact_zero"),
        (-1.0, 1.0, "output_gradient_inconclusive", "exact_zero"),
        (2e-12, 1e-12, "output_gradient_inconclusive", "below_declared_precision"),
        (-2.0, 1.0, "combined_output_descent_opposes_coordination", None),
        (10.0, 1.0, "successor_output_gradient_dominates", None),
        (9.999, 1.0, "no_fixed_output_gradient_obstacle", None),
        (0.0, 1.0, "no_fixed_output_gradient_obstacle", None),
    ],
)
def test_precision_branch_raw_null_rules(s, c, branch, reason):
    gs, gc = (np.full((50, 4, 42), v, dtype="<f8") for v in (s, c))
    report = m.classify_gradients(gs, gc)
    row = report["aggregate"]
    assert report["classification"] == branch and report["reason"] == reason
    assert row["N"] == 8400 and row["norms"]["C"] == pytest.approx(
        np.sqrt(8400) * abs(c)
    )
    assert row["eta"]["C"] == pytest.approx(
        np.sqrt(8400) * 2e-10 + 1e-8 * row["norms"]["C"]
    )
    if reason:
        assert row["cosine_C_total"] is None and row["norm_ratio_S_over_C"] is None
        assert row["relative_coordination_progress"] is None
    if s == 0:
        assert (
            row["cosine_C_S"] is None
            and row["cosine_C_S_undefined_reason"] == "S_exact_zero"
        )
    if s == 10:
        assert (
            row["D_C"] < 0
            and row["dots"]["C_S"] > 0
            and row["relative_coordination_progress"] == pytest.approx(11)
        )


def test_branch_instability_does_not_choose_path():
    value = m.compare_branches(
        {"classification": "successor_output_gradient_dominates", "reason": None},
        {"classification": "no_fixed_output_gradient_obstacle", "reason": None},
    )
    assert value == {
        "classification": "output_gradient_inconclusive",
        "reason": "branch_unstable",
    }


def circle_reference():
    phase = 2 * np.pi * np.arange(128, dtype=np.float64) / 128
    curves = np.zeros((6, 128, 7), dtype="<f8")
    curves[:, :, 0], curves[:, :, 1] = np.cos(phase), np.sin(phase)
    # Value-only existing canonical routine, never a gradient expected oracle.
    m.runtime()
    return m.reader.constants(curves)


@pytest.mark.parametrize("half", [False, True])
def test_same_and_half_phase_have_radial_derivative(half):
    ref = circle_reference()
    live = np.zeros((50, 4, 6, 7), dtype="<f8")
    live[:, :, :, 0] = 1
    if half:
        live[:, :, 3:, 0] = -1
    b = live[:, 0].reshape(50, 42).copy()
    cpu = m.cpu_gradients(ref, b, live.reshape(50, 4, 42))
    analytic = m.numpy_gradients(ref, b, live.reshape(50, 4, 42))
    np.testing.assert_allclose(
        cpu["g_coordination"], analytic["g_coordination"], rtol=1e-8, atol=2e-10
    )
    assert np.linalg.norm(cpu["g_coordination"]) > 1e-6


def test_symmetric_center_is_precision_inconclusive():
    ref = circle_reference()
    x = np.zeros((50, 4, 42), dtype="<f8")
    value = m.cpu_gradients(ref, x[:, 0].copy(), x)
    assert np.linalg.norm(value["g_coordination"]) < 1e-12
    assert (
        m.classify_gradients(value["g_successor"], value["g_coordination"])[
            "classification"
        ]
        == "output_gradient_inconclusive"
    )


def test_finite_overflow_is_technical_error():
    with pytest.raises((FloatingPointError, ValueError)):
        m.classify_gradients(np.full((50, 4, 42), 1e200), np.full((50, 4, 42), 1e200))
    with pytest.raises(ValueError):
        m.json_bytes({"not_finite": float("nan")})


def test_exact_raw_array_roster_and_quota(analyzed):
    arrays, _ = analyzed
    payload = m.array_bytes(arrays)
    assert len(payload) <= 640 << 10
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        assert set(archive.files) == set(m.ARRAY_SPEC)
        for key, shape in m.ARRAY_SPEC.items():
            assert archive[key].shape == shape and archive[key].dtype.str == "<f8"
    with pytest.raises(ValueError):
        m.array_bytes(arrays | {"forbidden_jacobian": np.zeros(1)})


def test_actual_selected_reader(samples):
    root, _, _, bindings = samples
    m.runtime()
    for role, binding in bindings.items():
        values, detail = m.reader.fixed_reader(
            root, role, binding, namespace="generated"
        )
        assert set(values) == set(binding["selected_schema"])
        assert detail["postdecode_verified"]
        assert detail["predecode_pin"] == detail["postdecode_pin"]
        assert len(detail["selected_reads"]) == len(values)
        compact = m.compact_read(detail)
        assert compact["observed_member_count"] == len(values) + 1
        assert len(m.json_bytes(compact)) < 8 << 10


def test_unsafe_extra_header_rejected_before_decode(samples, tmp_path):
    _, reference, _, bindings = samples
    path = tmp_path / "unsafe.npz"
    np.savez(path, **reference, forbidden=np.array([object()], dtype=object))
    binding = bindings["reference"] | {"path": path.name, **m.pin(path)}
    with pytest.raises(m.reader.ReaderFailure) as caught:
        m.reader.fixed_reader(tmp_path, "reference", binding, namespace="generated")
    assert caught.value.detail["selected_reads"] == []
    assert caught.value.detail["postdecode_verified"]


def test_actual_reader_posthash_detects_change(samples, tmp_path, monkeypatch):
    source, _, _, bindings = samples
    path = tmp_path / "reference.npz"
    shutil.copyfile(source / "reference.npz", path)
    binding = bindings["reference"] | {"path": path.name, **m.pin(path)}
    real_typed = m.reader.typed
    changed = False

    def mutate(array, shape, name):
        nonlocal changed
        result = real_typed(array, shape, name)
        if not changed:
            with path.open("ab") as stream:
                stream.write(b"postdecode_tamper")
            changed = True
        return result

    monkeypatch.setattr(m.reader, "typed", mutate)
    with pytest.raises(m.reader.ReaderFailure) as caught:
        m.reader.fixed_reader(tmp_path, "reference", binding, namespace="generated")
    assert caught.value.detail["first_error"]["stage"] == "postdecode_hash"
    assert caught.value.detail["postdecode_verified"] is False


@pytest.mark.parametrize(
    "root", [m.ROOT, m.ROOT / m.BASE, m.ROOT / m.BASE / "e339-old"]
)
def test_generated_root_rejects_original_or_old_namespace(root):
    with pytest.raises(ValueError):
        m.generated_root(root)


def test_original_hash_forbidden_before_read(samples):
    root, _, _, bindings = samples
    bad = copy.deepcopy(bindings)
    bad["live"]["sha256"] = m.IDENTITIES["live"][1]
    with pytest.raises(ValueError, match="original SHA"):
        m.generated_bindings(root, bad)


@pytest.mark.parametrize(
    "path", ["../escape", "a/../b", "a\\b", "/absolute", "a//b", "a/./b"]
)
def test_paths_reject_unsafe(path, tmp_path):
    with pytest.raises(ValueError):
        m.safe(path, tmp_path)


def test_real_generated_worker_disk_return_stdout(samples):
    root, _, _, bindings = samples
    published = []
    terminal, code = m.generated_worker(
        root, bindings, output="normal", printer=published.append
    )
    assert code == 0 and terminal["analysis_valid"] and terminal["publication_valid"]
    output = root / "normal"
    assert (
        (output / "result.json").read_bytes()
        == (output / "receipt.json").read_bytes()
        == published[0]
        == m.json_bytes(terminal)
    )
    manifest = m.read_json(output / "output-manifest.json")
    assert "output-manifest.json" not in manifest["files"]
    assert set(manifest["files"]) == {
        "neutral-gradient.npz",
        "held-gradient.npz",
        "live-gradient.npz",
        "selected-read-ledger.json",
        "result.json",
        "receipt.json",
    }
    assert sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) < 4 << 20
    assert not terminal["science_adopted"] and not terminal["neural_improvement"]
    assert terminal["neural_body_CUDA_parameter_VJP_calls"] == 0


@pytest.mark.parametrize(
    "kind",
    ["rename", "readback", "late_science", "late_final", "stdout", "post_stdout"],
)
def test_actual_publication_failures_retain_raw(samples, kind):
    root, _, _, bindings = samples
    now, published = [0.0], []

    def fault(stage):
        if kind == "rename" and stage == "before_rename:neutral-gradient.npz":
            raise OSError("generated rename failure")
        if kind == "readback" and stage == "before_readback":
            path = root / kind / "live-gradient.npz"
            with path.open("ab") as stream:
                stream.write(b"generated_readback_tamper")
        if kind == "late_science" and stage == "after_raw:live":
            now[0] = 13
        if kind == "late_final" and stage == "before_stdout":
            now[0] = 17
        if kind == "post_stdout" and stage == "after_stdout":
            now[0] = 17

    def printer(payload):
        if kind == "stdout":
            raise OSError("generated stdout sink failed")
        published.append(payload)

    terminal, code = m.generated_worker(
        root, bindings, output=kind, clock=lambda: now[0], printer=printer, fault=fault
    )
    assert (
        code == 1
        and not terminal["analysis_valid"]
        and terminal["suggested_diagnostic"] == "not_evaluated"
    )
    assert terminal["first_error"] is not None
    output = root / kind
    if kind == "rename":
        assert (output / "neutral-gradient.npz.partial").is_file()
        assert not (output / "neutral-gradient.npz").exists()
    else:
        assert (output / "neutral-gradient.npz").is_file()
        assert (output / "live-gradient.npz").is_file()
    path = output / (
        "revocation.json" if (output / "result.json").exists() else "failure.json"
    )
    assert m.read_json(path)["first_error"] == terminal["first_error"]
    if kind in ("readback", "late_final", "stdout", "post_stdout"):
        assert m.read_json(output / "result.json")[
            "analysis_valid"
        ]  # immutable, explicitly revoked
        assert terminal["revokes_provisional_result_sha256"] == m.digest(
            output / "result.json"
        )
    if kind == "post_stdout":
        assert published and m.read_json(output / "result.json") == json.loads(
            published[0]
        )


def test_reconciliation_failure_keeps_finite_acquired_raw(samples):
    root, _, _, bindings = samples
    # Change a generated saved claim, retain freshly pinned source arrays.
    m.runtime()
    with np.load(root / "live.npz", allow_pickle=False) as archive:
        values = {k: archive[k].copy() for k in m.TRAJ_SPEC}
    values["coordination_pair_loss"][0] += 0.1
    np.savez(root / "mismatch.npz", **values)
    bad = copy.deepcopy(bindings)
    bad["live"].update(path="mismatch.npz", **m.pin(root / "mismatch.npz"))
    terminal, code = m.generated_worker(
        root, bad, output="mismatch", printer=lambda _: None
    )
    assert code == 1 and terminal["first_error"]["type"] == "NumericalFailure"
    assert (root / "mismatch" / "live-gradient.npz").is_file()
    assert terminal["analysis"].keys() == {"neutral", "held"}


def test_stage_and_missing_reservations(tmp_path):
    output = tmp_path / "bounded"
    output.mkdir()
    budget = m.Budget(output)
    assert sum(budget.caps.values()) + budget.FAILURE_CAP == 4064 << 10
    assert budget.check()["missing_and_headroom_reserved_bytes"] == 4 << 20
    path = output / "live-gradient.npz"
    # A tiny test-only cap exercises the same actual stage/target accounting;
    # array_bytes separately checks the original 640KiB serialization bound.
    budget.caps[path] = 16
    path.write_bytes(b"x" * 16)
    stage = path.with_name(path.name + ".partial")
    stage.write_bytes(b"x")
    with pytest.raises(ValueError, match="stage/target"):
        budget.check()
    with pytest.raises(FileExistsError):
        m.write_bytes(path, b"must not overwrite", budget)
    assert path.stat().st_size == 16


def test_failure_publication_preserves_first_and_emergency(tmp_path, monkeypatch):
    output = tmp_path / "failed"
    output.mkdir()
    terminal = m.invalid_terminal(
        {"stage": "first", "type": "ValueError", "message": "first remains"},
        [],
        s0=0,
        mode="generated",
    )
    real_write = m.write_bytes

    def fail_primary(path, payload, budget, **kwargs):
        if path.name == "failure.json":
            raise OSError("generated failure sink")
        return real_write(path, payload, budget, **kwargs)

    monkeypatch.setattr(m, "write_bytes", fail_primary)
    m.preserve_failure(output, terminal, m.Budget(output))
    saved = m.read_json(output / "emergency-failure.json")
    assert saved["first_error"]["message"] == "first remains"
    assert saved["secondary_errors"][0]["stage"] == "failure_publication"


def test_absolute_clock_is_never_reset():
    now = [5.0]
    watch = m.Clock(0, clock=lambda: now[0])
    watch.science()
    now[0] = 11
    watch.begin_final()
    assert watch.final_deadline == 15
    now[0] = 14
    watch.begin_final()
    assert watch.final_deadline == 15
    now[0] = 15.1
    with pytest.raises(TimeoutError):
        watch.final()


def qualification_fixture(root):
    """Synthetic closed metadata only; not an actual original execution proof."""
    m.generated_root(root)
    for name in (*m.OWNED, m.SUPERVISOR):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(m.ROOT / name, target)
    audit = root / m.AUDIT_SOURCE
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text("# generated synthetic metadata helper\n", "utf8")
    helper = m.BASE + "/e340-sol-fixture-helper.py"
    (root / helper).write_text("# generated branch metadata helper\n", "utf8")
    current = m.source_map(root)
    source_pins = {
        k: m.pin(root / k) for k in (*m.OWNED, m.SUPERVISOR, helper, m.AUDIT_SOURCE)
    }
    evidence = {}

    def execution(name, case=None):
        prefix = m.BASE + "/e340-sol-fixture-" + name
        branch = {
            k: prefix + "-" + k + ".json"
            for k in ("contract", "supervision", "report", "pids")
        }
        script = m.AUDIT_SOURCE if case is not None else helper
        argv = [
            ".venv/Scripts/python.exe",
            "-X",
            "utf8",
            "-B",
            script,
            "--output",
            branch["report"],
        ]
        if case:
            argv.extend(("--generated-case", case))
        command = {
            "argv": argv,
            "stdout": prefix + ".stdout.txt",
            "stderr": prefix + ".stderr.txt",
            "wall_s": 20,
        }
        sources = {
            k: v["sha256"]
            for k, v in source_pins.items()
            if k != (helper if case else m.AUDIT_SOURCE)
        }
        contract = {
            "source_sha256": sources,
            "commands": [command],
            "supervisor_output": branch["supervision"],
            "max_tree_rss_bytes": 2 << 30,
            "original_calls": 0,
        }
        dump(root / branch["contract"], contract)
        expected = 0 if case in (None, "valid") else 1
        supervision = {
            "schema": "flylab.m3.e267.root-supervision.v1",
            "contract_sha256": m.digest(root / branch["contract"]),
            "success": expected == 0,
            "original_calls": 0,
            "elapsed_s": 0.1,
            "max_tree_rss_bytes": 1 << 20,
            "owned_pids": [11, 12],
            "attempts": [
                {
                    "argv": argv,
                    "exit_code": expected,
                    "killed": False,
                    "pids": [11, 12],
                    "wall_s": 0.08,
                }
            ],
        }
        report = {
            "schema": "flylab.m3.e340.synthetic-report.v1",
            "source_sha256": current,
            "audit_source_path": script,
            "audit_source_sha256": sources[script],
            "original_calls": 0,
            "passed": expected == 0,
            "first_error": None if expected == 0 else {"type": "generated rejection"},
            "checks": {"generated": expected == 0},
        }
        dump(root / branch["supervision"], supervision)
        dump(root / branch["report"], report)
        dump(
            root / branch["pids"],
            {
                "owned_pids": [11, 12],
                "all_owned_pids_ended": True,
                "present_after_return": {"11": False, "12": False},
            },
        )
        for key in ("stdout", "stderr"):
            (root / command[key]).write_text("", "utf8")
        for edge_name in (
            *branch.values(),
            command["stdout"],
            command["stderr"],
            *sources,
        ):
            evidence[edge_name] = m.pin(root / edge_name)
        return branch

    branches = {name: execution(name) for name in ("sol", "root")}
    cases = {case: execution(case, case) for case in m.AUDIT_CASES}
    qual_path = m.BASE + "/e340-sol-fixture-audit-qualification.json"
    qualification = {
        "schema": "flylab.m3.e340.audit-cli-qualification.v1",
        "passed": True,
        "original_calls": 0,
        "source_sha256": current,
        "audit_source_sha256": m.digest(audit),
        "cases": cases,
    }
    dump(root / qual_path, qualification)
    evidence[qual_path] = m.pin(root / qual_path)
    proof = {
        "schema": m.PROOF_SCHEMA,
        "passed": True,
        "original_calls": 0,
        "source_sha256": current,
        "branches": branches,
        "same_audit_cli": {"source": m.AUDIT_SOURCE, "qualification": qual_path},
        "evidence": evidence,
    }
    return proof, current


def test_finite_closed_proof_positive(tmp_path):
    proof, current = qualification_fixture(tmp_path)
    result = m.verify_current_proof(tmp_path, proof, current)
    assert result["cases_closed"] == list(m.AUDIT_CASES)
    assert result["branches_closed"] == ["sol", "root"]


@pytest.mark.parametrize(
    "kind",
    [
        "helper_missing",
        "other_helper",
        "helper_bytes",
        "other_output",
        "argv_mismatch",
        "stale_source",
        "unrelated",
        "missing_case",
        "missing_log",
        "live_pid",
        "original_counter",
    ],
)
def test_closed_proof_rejects_tamper(tmp_path, kind):
    proof, current = qualification_fixture(tmp_path)
    branch = proof["branches"]["root"]
    contract = m.read_json(tmp_path / branch["contract"])
    supervision = m.read_json(tmp_path / branch["supervision"])
    report = m.read_json(tmp_path / branch["report"])
    if kind == "helper_missing":
        del contract["source_sha256"][report["audit_source_path"]]
    elif kind == "other_helper":
        contract["commands"][0]["argv"][4] = m.AUDIT_SOURCE
        supervision["attempts"][0]["argv"] = contract["commands"][0]["argv"].copy()
    elif kind == "helper_bytes":
        (tmp_path / report["audit_source_path"]).write_text(
            "changed generated helper", "utf8"
        )
    elif kind == "other_output":
        contract["commands"][0]["argv"][6] = proof["branches"]["sol"]["report"]
        supervision["attempts"][0]["argv"] = contract["commands"][0]["argv"].copy()
    elif kind == "argv_mismatch":
        supervision["attempts"][0]["argv"].append("unrelated")
    elif kind == "stale_source":
        report["source_sha256"][m.OWNED[0]] = "0" * 64
    elif kind == "unrelated":
        path = m.BASE + "/e340-sol-unrelated.txt"
        (tmp_path / path).write_text("not an execution edge", "utf8")
        proof["evidence"][path] = m.pin(tmp_path / path)
    elif kind == "missing_case":
        path = proof["same_audit_cli"]["qualification"]
        qual = m.read_json(tmp_path / path)
        del qual["cases"]["wrong_detach"]
        dump(tmp_path / path, qual)
        proof["evidence"][path] = m.pin(tmp_path / path)
    elif kind == "missing_log":
        del proof["evidence"][contract["commands"][0]["stdout"]]
    elif kind == "live_pid":
        dump(
            tmp_path / branch["pids"],
            {"owned_pids": [11, 12], "all_owned_pids_ended": False},
        )
        proof["evidence"][branch["pids"]] = m.pin(tmp_path / branch["pids"])
    elif kind == "original_counter":
        contract["original_calls"] = 1
    # The attack repins the changed producer, so rejection must follow edges/state.
    dump(tmp_path / branch["contract"], contract)
    supervision["contract_sha256"] = m.digest(tmp_path / branch["contract"])
    dump(tmp_path / branch["supervision"], supervision)
    dump(tmp_path / branch["report"], report)
    for key in ("contract", "supervision", "report"):
        proof["evidence"][branch[key]] = m.pin(tmp_path / branch[key])
    with pytest.raises(ValueError):
        m.verify_current_proof(tmp_path, proof, current)


def test_merge_does_not_overwrite_historical_pin():
    value = {"sha256": "a" * 64, "bytes": 1}
    pins = {"tools/historical.py": value.copy()}
    with pytest.raises(ValueError, match="collision"):
        m.merge_pins(pins, {"tools/historical.py": {"sha256": "b" * 64, "bytes": 1}})
    assert pins["tools/historical.py"] == value


def test_no_text_only_neural_module_imports():
    assert "tools.verify_c_banc_m3_temporal_direction" not in m.sys.modules
    assert "tools.verify_c_banc_m3_smooth_efficacy_direction" not in m.sys.modules
    assert m.os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert all(m.os.environ[k] == "1" for k in m.THREAD_ENV)
    assert m.torch.get_num_threads() == 1


def test_tiny_nonzero_norm_is_not_filled_with_zero():
    gs, gc = np.full((50, 4, 42), 2e-200), np.full((50, 4, 42), 1e-200)
    row = m.classify_gradients(gs, gc)
    assert row["reason"] == "below_declared_precision"
    assert row["aggregate"]["norms"]["C"] == pytest.approx(
        np.sqrt(8400) * 1e-200, abs=0, rel=1e-14
    )
    assert row["aggregate"]["norms"]["C"] > 0
    assert row["aggregate"]["cosine_C_total"] is None


def predecessor_fixture(root):
    """Finite generated JSON producer states; no archive/header/value/hash calls."""
    prefix = m.BASE + "/parent-e339-"
    folder = m.BASE + "/e339-saved-coordination/"
    source = {
        "tools/analyze_c_banc_m3_saved_coordination.py": "a" * 64,
        "tests/test_c_banc_saved_coordination.py": "b" * 64,
    }
    frozen_name = m.BASE + "/e339-saved-coordination-freeze.json"
    frozen = {
        "original_bindings": m.original_bindings(),
        "source_sha256": source | {"tools/immutable-reader.py": "c" * 64},
    }
    dump(root / frozen_name, frozen)
    dump(
        root / (prefix + "current-generated-qualification.json"),
        {"passed": True, "source_sha256": source},
    )
    review = {
        "passed": True,
        "ROOT_SECOND_completed": True,
        "remaining_ROOT_source_fix_required": False,
        "source_sha256": source,
        "freeze_sha256": m.digest(root / frozen_name),
        "generated_qualification_sha256": m.digest(
            root / (prefix + "current-generated-qualification.json")
        ),
    }
    dump(root / (prefix + "original-source-review.json"), review)
    result = {
        "status": "valid_provisional_diagnostic",
        "analysis_valid": True,
        "publication_valid": True,
        "first_error": None,
        "secondary_errors": [],
        "freeze_sha256": review["freeze_sha256"],
        "review_sha256": m.digest(root / (prefix + "original-source-review.json")),
    }
    dump(root / (folder + "result.json"), result)
    dump(
        root / (folder + "output-manifest.json"),
        {"files": {"result.json": m.pin(root / (folder + "result.json"))}},
    )
    dump(
        root / (prefix + "saved-raw-audit.json"),
        {"passed": True, "checks": {str(i): True for i in range(5591)}},
    )
    evidence = {folder + "result.json": m.pin(root / (folder + "result.json"))}
    for role in ("original", "audit"):
        contract_name = prefix + role + "-contract.json"
        command = {"argv": ["generated", role]}
        dump(root / contract_name, {"commands": [command], "original_calls": 1})
        supervision_name = prefix + role + "-supervision.json"
        dump(
            root / supervision_name,
            {
                "attempts": [
                    {"argv": command["argv"], "exit_code": 0, "killed": False}
                ],
                "success": True,
                "original_calls": 1,
                "owned_pids": [11, 12],
                "contract_sha256": m.digest(root / contract_name),
            },
        )
        dump(
            root / (prefix + role + "-pids.json"),
            {
                "all_owned_pids_absent": True,
                "present_after_return": {"11": False, "12": False},
            },
        )
        for suffix in (".stdout.txt", ".stderr.txt"):
            (root / (prefix + role + suffix)).write_text("", "utf8")
        role_evidence = {
            name: m.pin(root / name)
            for name in (
                contract_name,
                supervision_name,
                prefix + role + "-pids.json",
                prefix + role + ".stdout.txt",
                prefix + role + ".stderr.txt",
            )
        }
        evidence.update(role_evidence)
        dump(
            root / (prefix + role + "-external-receipt.json"),
            {"passed": True, "first_error": None, "evidence": role_evidence},
        )
    dump(
        root / (prefix + "final-decision.json"),
        {
            "schema": "flylab.m3.e339.root-final-decision.v1",
            "analysis_valid": True,
            "root_independent_raw_qualification": True,
            "science_adopted": True,
            "candidate_adopted": False,
            "F1_F8_complete": False,
            "raw_checks": 5591,
            "original_cumulative_s": 9567.165820199525,
            "original_limit_s": 9660,
            "evidence": evidence,
        },
    )
    pins = {
        path.relative_to(root).as_posix(): m.pin(path)
        for path in root.rglob("*")
        if path.is_file()
    }
    return pins


def test_predecessor_extra_import_sources_do_not_erase_owned_edge(tmp_path):
    pins = predecessor_fixture(tmp_path)
    report = m.verify_predecessor_states(tmp_path, pins)
    assert (
        report["direct_raw_checks"] == 5591 and not report["recursive_inherited_rehash"]
    )


def test_predecessor_wrong_owned_source_is_rejected(tmp_path):
    pins = predecessor_fixture(tmp_path)
    name = m.BASE + "/e339-saved-coordination-freeze.json"
    frozen = m.read_json(tmp_path / name)
    frozen["source_sha256"]["tools/analyze_c_banc_m3_saved_coordination.py"] = "d" * 64
    dump(tmp_path / name, frozen)
    with pytest.raises(ValueError, match="producer state"):
        m.verify_predecessor_states(tmp_path, pins)


def test_current_actual_worker_after_source_refinement(samples):
    root, _, _, bindings = samples
    published = []
    terminal, code = m.generated_worker(
        root, bindings, output="current-normal", printer=published.append
    )
    assert code == 0 and terminal["analysis_valid"]
    output = root / "current-normal"
    assert (
        (output / "result.json").read_bytes()
        == (output / "receipt.json").read_bytes()
        == published[0]
        == m.json_bytes(terminal)
    )
    assert terminal["science_deadline"] - terminal["clock_s0"] == 12
    assert terminal["internal_deadline"] - terminal["clock_s0"] == 16
    assert terminal["final_deadline"] <= terminal["internal_deadline"]


def test_current_poststdout_storage_failure_revokes(samples):
    root, _, _, bindings = samples
    output = root / "current-overhead"
    published = []

    def fault(stage):
        if stage == "after_stdout":
            (output / "injected-overhead.bin").write_bytes(b"x" * ((32 << 10) + 1))

    terminal, code = m.generated_worker(
        root, bindings, output=output.name, fault=fault, printer=published.append
    )
    assert code == 1 and terminal["first_error"]["stage"] == "postpublication"
    assert "quota" in terminal["first_error"]["message"]
    revoked = m.read_json(output / "revocation.json")
    assert (
        revoked["first_error"] == terminal["first_error"]
        and not revoked["analysis_valid"]
    )
    assert m.read_json(output / "result.json")["analysis_valid"]
    assert m.json_bytes(m.read_json(output / "result.json")) == published[0]
    assert sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) < 4 << 20


def test_current_partial_save_error_does_not_replace_first(samples):
    root, _, trajectories, bindings = samples
    values = {k: a.copy() for k, a in trajectories["neutral"].items()}
    values["coordination_pair_loss"][0] += 0.1
    np.savez(root / "current-mismatch-neutral.npz", **values)
    bad = copy.deepcopy(bindings)
    bad["neutral"].update(
        path="current-mismatch-neutral.npz",
        **m.pin(root / "current-mismatch-neutral.npz"),
    )

    def fault(stage):
        if stage == "before_rename:neutral-gradient.npz":
            raise OSError("generated partial raw rename failure")

    terminal, code = m.generated_worker(
        root, bad, output="current-partial-failure", fault=fault, printer=lambda _: None
    )
    assert code == 1 and terminal["first_error"]["type"] == "NumericalFailure"
    assert "saved15pair" in terminal["first_error"]["message"]
    assert terminal["secondary_errors"][0]["stage"] == "partial_raw_publication"
    assert (root / "current-partial-failure" / "neutral-gradient.npz.partial").is_file()
    assert (
        m.read_json(root / "current-partial-failure" / "failure.json")["first_error"]
        == terminal["first_error"]
    )


def test_bounded_actual_small_publication_rejects_late_clock(tmp_path):
    output = tmp_path / "small-publication"
    output.mkdir()
    budget = m.Budget(output)
    now = [0.0]
    watch = m.Clock(0, clock=lambda: now[0])
    watch.begin_final()

    def fault(stage):
        if stage == "before_rename:result.json":
            now[0] = 5

    with pytest.raises(TimeoutError):
        m.write_bytes(
            output / "result.json", b"{}\n", budget, timely=watch.final, fault=fault
        )
    assert (output / "result.json").exists()  # late after rename is invalid, retained


def test_terminal_stdout_payload_is_bounded_before_print(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    budget = m.Budget(tmp_path)
    with pytest.raises(ValueError, match="stdout quota"):
        budget.before_stdout(b"x" * ((256 << 10) + 1))
    assert budget.stdout_bytes == 0


@pytest.fixture(scope="module")
def completed_resource_fixture(tmp_path_factory):
    # One shared generated source/helper set, fresh small receipts per fault.
    root = tmp_path_factory.mktemp("resource-proof")
    proof, current = qualification_fixture(root)
    return root, proof, current


def resource_receipts(fixture, label, mutate):
    root, proof, current = fixture
    original = proof["branches"]["root"]
    prefix = m.BASE + "/e340-sol-r1-resource-" + label
    branch = {
        key: prefix + "-" + key + ".json"
        for key in ("contract", "supervision", "report", "pids")
    }
    contract, supervision, report, pids = (
        copy.deepcopy(m.read_json(root / original[key]))
        for key in ("contract", "supervision", "report", "pids")
    )
    contract["supervisor_output"] = branch["supervision"]
    contract["commands"][0]["argv"][6] = branch["report"]
    contract["commands"][0]["stdout"] = prefix + ".stdout.txt"
    contract["commands"][0]["stderr"] = prefix + ".stderr.txt"
    supervision["attempts"][0]["argv"] = contract["commands"][0]["argv"].copy()
    mutate(contract, supervision, pids)
    # Unsafe NaN/Infinity serialization is intentional only in rejected fixtures;
    # production read_json must reject it before any resource acceptance.
    for key, value in (
        ("contract", contract),
        ("supervision", supervision),
        ("report", report),
        ("pids", pids),
    ):
        (root / branch[key]).write_text(
            json.dumps(value, allow_nan=True) + "\n", "utf8"
        )
    supervision["contract_sha256"] = m.digest(root / branch["contract"])
    (root / branch["supervision"]).write_text(
        json.dumps(supervision, allow_nan=True) + "\n", "utf8"
    )
    evidence = copy.deepcopy(proof["evidence"])
    for name in branch.values():
        evidence[name] = m.pin(root / name)
    for key in ("stdout", "stderr"):
        name = contract["commands"][0][key]
        (root / name).write_text("", "utf8")
        evidence[name] = m.pin(root / name)
    return root, branch, evidence, current


def test_completed_resource_actual_schema_positive(completed_resource_fixture):
    root, proof, current = completed_resource_fixture
    m.closed_execution(root, proof["branches"]["root"], proof["evidence"], current)

    # Equality at the declared cap is allowed; this is metadata only, not a run.
    def at_cap(contract, supervision, pids):
        supervision["elapsed_s"] = contract["commands"][0]["wall_s"]
        supervision["attempts"][0]["wall_s"] = supervision["elapsed_s"]
        supervision["max_tree_rss_bytes"] = contract["max_tree_rss_bytes"]

    m.closed_execution(
        *resource_receipts(completed_resource_fixture, "valid-cap", at_cap)
    )


@pytest.mark.parametrize(
    "target,value",
    [
        ("declared_wall", None),
        ("declared_wall", True),
        ("declared_wall", "1"),
        ("declared_wall", float("nan")),
        ("declared_wall", float("inf")),
        ("total_wall", None),
        ("total_wall", True),
        ("total_wall", "0.1"),
        ("total_wall", 0),
        ("total_wall", -1),
        ("total_wall", float("nan")),
        ("total_wall", float("inf")),
        ("total_wall", 21),
        ("attempt_wall", None),
        ("attempt_wall", True),
        ("attempt_wall", "0.08"),
        ("attempt_wall", 0),
        ("attempt_wall", -1),
        ("attempt_wall", float("nan")),
        ("attempt_wall", float("inf")),
        ("attempt_wall", 21),
        ("attempt_wall", 0.11),
        ("actual_RSS", None),
        ("actual_RSS", True),
        ("actual_RSS", "1048576"),
        ("actual_RSS", 1048576.0),
        ("actual_RSS", 0),
        ("actual_RSS", -1),
        ("actual_RSS", float("nan")),
        ("actual_RSS", float("inf")),
        ("actual_RSS", 3 << 30),
    ],
)
def test_completed_resource_times_RSS_reject(
    completed_resource_fixture, target, value, request
):
    label = request.node.nodeid.rsplit("[", 1)[-1].replace("]", "").replace(".", "_")

    def mutate(contract, supervision, pids):
        if target == "declared_wall":
            owner, key = contract["commands"][0], "wall_s"
        elif target == "total_wall":
            owner, key = supervision, "elapsed_s"
        elif target == "attempt_wall":
            owner, key = supervision["attempts"][0], "wall_s"
        else:
            owner, key = supervision, "max_tree_rss_bytes"
        if value is None:
            del owner[key]
        else:
            owner[key] = value

    with pytest.raises(ValueError):
        m.closed_execution(
            *resource_receipts(completed_resource_fixture, label, mutate)
        )


@pytest.mark.parametrize(
    "kind",
    [
        "reported_overrun",
        "supervisor_missing",
        "supervisor_empty",
        "supervisor_duplicate",
        "supervisor_boolean",
        "supervisor_zero",
        "supervisor_negative",
        "supervisor_float",
        "record_missing",
        "record_empty",
        "record_duplicate",
        "record_boolean",
        "record_reordered",
        "attempt_missing",
        "attempt_mismatch",
        "presence_missing",
        "presence_key_missing",
        "presence_extra",
        "presence_true",
        "presence_zero",
        "presence_null",
        "presence_wrong_type",
    ],
)
def test_completed_resource_PID_contradictions_reject(completed_resource_fixture, kind):
    def mutate(contract, supervision, pids):
        if kind == "reported_overrun":
            contract["commands"][0]["wall_s"], supervision["elapsed_s"] = 1, 2
        elif kind == "supervisor_missing":
            del supervision["owned_pids"]
        elif kind.startswith("supervisor_"):
            supervision["owned_pids"] = {
                "empty": [],
                "duplicate": [11, 11],
                "boolean": [True, 12],
                "zero": [0, 12],
                "negative": [-11, 12],
                "float": [11.0, 12],
            }[kind.removeprefix("supervisor_")]
        elif kind == "record_missing":
            del pids["owned_pids"]
        elif kind.startswith("record_"):
            pids["owned_pids"] = {
                "empty": [],
                "duplicate": [11, 11],
                "boolean": [True, 12],
                "reordered": [12, 11],
            }[kind.removeprefix("record_")]
        elif kind == "attempt_missing":
            del supervision["attempts"][0]["pids"]
        elif kind == "attempt_mismatch":
            supervision["attempts"][0]["pids"] = [11]
        elif kind == "presence_missing":
            del pids["present_after_return"]
        elif kind == "presence_key_missing":
            del pids["present_after_return"]["11"]
        elif kind == "presence_extra":
            pids["present_after_return"]["13"] = False
        elif kind == "presence_wrong_type":
            pids["present_after_return"] = [False, False]
        else:
            pids["present_after_return"]["11"] = {
                "presence_true": True,
                "presence_zero": 0,
                "presence_null": None,
            }[kind]

    with pytest.raises(ValueError):
        m.closed_execution(*resource_receipts(completed_resource_fixture, kind, mutate))
