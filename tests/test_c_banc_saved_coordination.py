"""Generated-only E339 qualification, with independent scalar calculations."""

import copy
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from tools import analyze_c_banc_m3_saved_coordination as c


@pytest.fixture(autouse=True)
def loaded_runtime():
    c.runtime()


def reference():
    curves = np.empty((6, 128, 7), dtype="<f8")
    offsets = (0, .31, .63, .47, .82, .12)
    for leg in range(6):
        for s in range(128):
            phi = 2 * math.pi * (s / 128 + offsets[leg])
            curves[leg, s] = [math.sin(phi + j * .71) * (1 + .1 * j)
                              + .05 * math.cos(2 * phi + j) + leg * .13 for j in range(7)]
    # Scalar wrapped interpolation and independent reduction over all7 joints.
    future = np.empty_like(curves)
    norm = np.zeros(6, dtype="<f8")
    for leg in range(6):
        for s in range(128):
            coordinate = ((s / 128 + 2.382760007990865 * .005) % 1) * 128
            lo = math.floor(coordinate)
            for j in range(7):
                future[leg, s, j] = curves[leg, lo, j] + (coordinate - lo) * (
                    curves[leg, (lo + 1) % 128, j] - curves[leg, lo, j])
                norm[leg] += (future[leg, s, j] - curves[leg, s, j]) ** 2 / (128 * 7)
    basis = np.array([[math.cos(2 * math.pi * s / 128), math.sin(2 * math.pi * s / 128)]
                      for s in range(128)], dtype="<f8")
    return {"curves": curves, "future_anchors": future, "normalization": norm,
            "temperature": 7 * norm, "phase_basis": basis}


def scalar_encode(q, ref, coefficient=1):
    z = np.zeros((len(q), 6, 2))
    for t in range(len(q)):
        for leg in range(6):
            logits = [-coefficient * sum((q[t, 7 * leg + j] - ref["curves"][leg, s, j]) ** 2
                                         for j in range(7)) / ref["temperature"][leg]
                      for s in range(128)]
            peak = max(logits)
            weights = [math.exp(x - peak) for x in logits]
            denominator = sum(weights)
            for axis in range(2):
                z[t, leg, axis] = sum(w * ref["phase_basis"][s, axis]
                                      for s, w in enumerate(weights)) / denominator
    return z, np.sqrt((z * z).sum(axis=-1))


def generated_inputs():
    ref = reference()
    phases = np.arange(201) % 128
    q = ref["curves"][:, phases].transpose(1, 0, 2).reshape(201, 42)
    z, resultant = c.Encoder(ref).encode(q)
    pair_loss = np.array([np.mean(1 - (z[1:, a] * z[1:, b]).sum(axis=-1))
                          for a in range(6) for b in range(a + 1, 6)])
    values = {"q": q, "z": z, "R": resultant, "coordination_pair_loss": pair_loss}
    return {"reference": ref, **{k: copy.deepcopy(values) for k in ("neutral", "held", "live")}}


def archive(root, role, values, extra=None):
    name = role + ".npz"
    np.savez(root / name, **values, **(extra or {}))
    return {"path": name, **c.pin(root / name), "selected_schema": c.schema_for(role),
            "header_observation_scope": "observation", "descriptor": None}


def read_generated(root, role, binding):
    return c.fixed_reader(root, role, binding, namespace="generated")


def test_scalar_constants_and_encoder_no_half_coefficient_or_offset():
    ref = reference()
    vector_constants = c.constants(ref["curves"])
    for key in ref:
        np.testing.assert_allclose(vector_constants[key], ref[key], rtol=1e-12, atol=1e-15)
    q = ref["curves"][:, [0, 11, 57, 127]].transpose(1, 0, 2).reshape(4, 42)
    scalar_z, scalar_r = scalar_encode(q, ref)
    z, resultant = c.Encoder(ref).encode(q)
    np.testing.assert_allclose(z, scalar_z, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(resultant, scalar_r, rtol=1e-12, atol=1e-14)
    wrong_z, _ = scalar_encode(q, ref, coefficient=.5)
    assert np.max(abs(wrong_z - z)) > 1e-3
    shifted = q.reshape(4, 6, 7).copy()
    for leg in range(6):
        shifted[:, leg] = ref["curves"][leg, (np.array([0, 11, 57, 127]) + 17 * leg) % 128]
    wrong_z, _ = c.Encoder(ref).encode(shifted.reshape(4, 42))
    assert np.max(abs(wrong_z - z)) > .5


@pytest.mark.parametrize("field", ["normalization", "temperature", "future_anchors", "phase_basis"])
def test_constant_tamper(field):
    ref = reference()
    ref[field].flat[0] += .001
    with pytest.raises(ValueError, match="reconciliation"):
        c.Encoder(ref)


def test_global_reference_200_rows_and_all201_reconciliation():
    inputs = generated_inputs()
    analysis, arrays = c.analyze(inputs)
    assert analysis["diagnosis"]["suggested_diagnostic"] == "no_prominent_angular_misalignment_at_fixed_rule"
    assert analysis["statistics_rows"] == 200
    assert analysis["pair_order"] == [[a, b] for a in range(6) for b in range(a + 1, 6)]
    assert [c.PAIRS[i] for i in c.LR_INDICES] == [(0, 3), (1, 4), (2, 5)]
    q_ref = inputs["reference"]["curves"].transpose(1, 0, 2).reshape(128, 42)
    expected, _ = scalar_encode(q_ref[[0, 31, 76]], inputs["reference"])
    np.testing.assert_allclose(arrays["reference"]["z"][[0, 31, 76]], expected, atol=1e-14)
    inputs["live"]["z"][0, 0, 0] += .01
    with pytest.raises(ValueError, match="all201"):
        c.analyze(inputs)


def test_row_zero_not_in_pair_loss():
    inputs = generated_inputs()
    ref = inputs["reference"]
    inputs["live"]["q"][0, :7] = ref["curves"][0, 64]
    z, resultant = c.Encoder(ref).encode(inputs["live"]["q"])
    inputs["live"].update(z=z, R=resultant)
    _, arrays = c.analyze(inputs)
    np.testing.assert_allclose(arrays["live"]["pair_loss"].mean(0),
                               inputs["live"]["coordination_pair_loss"])
    inputs["live"]["coordination_pair_loss"] = np.array([
        np.mean(1 - (z[:, a] * z[:, b]).sum(axis=-1)) for a, b in c.PAIRS])
    with pytest.raises(ValueError, match="last200"):
        c.analyze(inputs)


def moment_values(phases, radius=.8):
    phases = np.asarray(phases)
    return np.stack((np.cos(2 * np.pi * phases), np.sin(2 * np.pi * phases)), axis=-1) * radius


def test_same_half_phase_loss_algebra_lowR_and_wrap():
    phases = np.zeros((10, 6))
    phases[:, 3] = .5
    z = moment_values(phases)
    summary, arrays = c.moments(z, np.linalg.norm(z, axis=-1))
    assert abs(summary["pairs"][2]["bias_cycles"]) == pytest.approx(.5)
    assert arrays["angular_weighted_loss"][:, 2] == pytest.approx(np.full(10, 1.28))
    assert arrays["pair_loss"][:, 2] == pytest.approx(np.full(10, 1.64))
    np.testing.assert_allclose(arrays["radial_loss"] + arrays["angular_weighted_loss"], arrays["pair_loss"])
    zero = z.copy()
    zero[:, 0] = 0
    summary, arrays = c.moments(zero, np.linalg.norm(zero, axis=-1))
    assert summary["pairs"][0]["valid_count"] == 0
    assert summary["pairs"][0]["bias_cycles"] is None
    assert np.isnan(arrays["pair_angle_cycles"][:, 0]).all()
    assert np.isfinite(arrays["pair_loss"]).all()
    assert summary["coherence_valid_mean"] is None
    phases[:, 0] = [.49, -.49] * 5
    phases[:, 3] = 0
    z = moment_values(phases)
    summary, _ = c.moments(z, np.linalg.norm(z, axis=-1))
    assert abs(summary["pairs"][0]["bias_cycles"]) == pytest.approx(.5)
    assert summary["pairs"][0]["deviation_max_cycles"] == pytest.approx(.01)


def test_exact_R_boundary_and_null_live_mean():
    phases = np.zeros((200, 6))
    phases[100:, 0] = .5
    z = moment_values(phases, radius=.5)
    summary, arrays = c.moments(z, np.linalg.norm(z, axis=-1))
    assert summary["pairs"][0]["valid_count"] == 200
    assert summary["pairs"][0]["bias_cycles"] is None
    assert summary["pairs"][0]["resultant"] <= 1e-12
    lower = np.nextafter(.5, 0)
    z = moment_values(np.zeros((200, 6)), radius=lower)
    low, _ = c.moments(z, np.linalg.norm(z, axis=-1))
    assert low["all6_valid_count"] == 0
    reference_summary = copy.deepcopy(summary)
    reference_summary.update(total_count=128, all6_valid_count=128)
    for row in reference_summary["pairs"]:
        row.update(valid_count=128, bias_cycles=0., deviation_max_cycles=0.)
    diagnosis, _ = c.decide(reference_summary, summary, arrays)
    assert diagnosis["suggested_diagnostic"] == "angular_misalignment_prominent"


def decision_fixture(valid=180, ref_valid=122, count=45):
    z = moment_values(np.zeros((200, 6)))
    live, arrays = c.moments(z, np.linalg.norm(z, axis=-1))
    ref = copy.deepcopy(live)
    ref.update(total_count=128, all6_valid_count=ref_valid)
    live["all6_valid_count"] = valid
    for row in ref["pairs"]:
        row.update(total_count=128, valid_count=ref_valid, bias_cycles=0., deviation_max_cycles=.0375)
    for row in live["pairs"]:
        row.update(valid_count=valid)
    arrays["pair_valid"][valid:] = False
    arrays["pair_angle_cycles"][:] = .1875  # exact edge, must not exceed
    arrays["pair_angle_cycles"][:count, 0] = np.nextafter(.1875, 1)
    return ref, live, arrays


@pytest.mark.parametrize("valid,ref_valid,count,expected", [
    (180, 122, 45, "angular_misalignment_prominent"),
    (180, 122, 44, "no_prominent_angular_misalignment_at_fixed_rule"),
    (179, 122, 45, "proxy_limited"), (180, 121, 45, "proxy_limited"),
    (200, 128, 49, "no_prominent_angular_misalignment_at_fixed_rule"),
    (200, 128, 50, "angular_misalignment_prominent"),
])
def test_integer_coverage_and_strict_exceed_boundary(valid, ref_valid, count, expected):
    ref, live, arrays = decision_fixture(valid, ref_valid, count)
    decision, _ = c.decide(ref, live, arrays)
    assert decision["suggested_diagnostic"] == expected
    assert decision["pair_table"][0]["exceed_count"] == count
    assert decision["pair_table"][1]["exceed_count"] == 0


def test_reference_calibration_boundary_null_pair_order_and_denominator():
    ref, live, arrays = decision_fixture()
    ref["pairs"][0]["deviation_max_cycles"] = .05
    assert c.decide(ref, live, arrays)[0]["reference_calibration_pass"]
    ref["pairs"][0]["deviation_max_cycles"] = np.nextafter(.05, 1)
    assert c.decide(ref, live, arrays)[0]["suggested_diagnostic"] == "proxy_limited"
    ref["pairs"][0]["bias_cycles"] = None
    assert c.decide(ref, live, arrays)[0]["pair_table"][0]["exceed_count"] is None
    ref["pairs"][0]["pair"] = [0, 2]
    with pytest.raises(ValueError, match="pair order"):
        c.decide(ref, live, arrays)
    ref, live, arrays = decision_fixture()
    live["pairs"][0]["valid_count"] += 1
    with pytest.raises(ValueError, match="denominator"):
        c.decide(ref, live, arrays)


def test_real_selected_reader_extra_not_decoded(tmp_path, monkeypatch):
    inputs = generated_inputs()
    binding = archive(tmp_path, "reference", inputs["reference"], extra={"extra": np.arange(8)})
    reads = []
    actual = np.lib.npyio.NpzFile.__getitem__
    def selected_only(self, key):
        reads.append(key)
        assert key in c.REF_SPEC
        return actual(self, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", selected_only)
    values, ledger = read_generated(tmp_path, "reference", binding)
    assert reads == list(c.REF_SPEC)
    assert set(values) == set(c.REF_SPEC)
    assert "extra" in ledger["header_observation"]["descriptor"]["members"]
    assert ledger["header_observation"]["descriptor_scope"] == "observation"
    assert ledger["postdecode_verified"]
    assert [x["key"] for x in ledger["selected_reads"]] == list(c.REF_SPEC)


@pytest.mark.parametrize("kind", ["unsafe_extra", "wrong_shape", "wrong_dtype", "wrong_sha", "nan"])
def test_reader_rejects_generated_faults(tmp_path, kind):
    ref = reference()
    extra = {"extra": np.array([object()], dtype=object)} if kind == "unsafe_extra" else None
    if kind == "wrong_shape":
        ref["curves"] = ref["curves"][:5]
    if kind == "wrong_dtype":
        ref["curves"] = ref["curves"].astype("<f4")
    if kind == "nan":
        ref["curves"][0, 0, 0] = np.nan
    binding = archive(tmp_path, "reference", ref, extra)
    if kind == "wrong_sha":
        binding["sha256"] = "f" * 64
    with pytest.raises(c.ReaderFailure) as caught:
        read_generated(tmp_path, "reference", binding)
    detail = caught.value.detail
    assert detail["first_error"] is not None
    if kind in ("wrong_shape", "wrong_dtype", "unsafe_extra", "wrong_sha"):
        assert detail["selected_reads"] == []
    if kind == "nan":
        assert detail["first_error"]["stage"] == "selected_decode"


@pytest.mark.parametrize("decode_error", [False, True])
def test_postdecode_change_and_preserve_first_error(tmp_path, monkeypatch, decode_error):
    ref = reference()
    if decode_error:
        ref["normalization"][0] = np.nan
    binding = archive(tmp_path, "reference", ref)
    actual = np.lib.npyio.NpzFile.__exit__
    def close_then_mutate(self, *args):
        returned = actual(self, *args)
        with (tmp_path / binding["path"]).open("ab") as stream:
            stream.write(b"changed_after_actual_selected_decode")
        return returned
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__exit__", close_then_mutate)
    with pytest.raises(c.ReaderFailure) as caught:
        read_generated(tmp_path, "reference", binding)
    detail = caught.value.detail
    assert not detail["postdecode_verified"]
    assert len(detail["selected_reads"]) > 0
    if decode_error:
        assert detail["first_error"]["stage"] == "selected_decode"
        assert detail["secondary_errors"][0]["stage"] == "postdecode_hash"
    else:
        assert detail["first_error"]["stage"] == "postdecode_hash"


def test_original_binding_and_generated_namespace_are_not_cli_overrides(tmp_path):
    binding = c.original_bindings()["live"]
    with pytest.raises(c.ReaderFailure, match="literal original binding/root"):
        c.fixed_reader(tmp_path, "live", binding)
    with pytest.raises(c.ReaderFailure, match="original identity forbidden"):
        read_generated(tmp_path, "live", binding)
    changed = copy.deepcopy(binding)
    changed["path"] = "different.npz"
    with pytest.raises(c.ReaderFailure, match="literal original binding/root"):
        c.fixed_reader(c.ROOT, "live", changed)
    with pytest.raises(SystemExit):
        c.main(["--run", "--root", str(tmp_path), "--freeze-sha256", "0" * 64,
                "--review-sha256", "0" * 64])


@pytest.fixture(scope="module")
def symbolic_freeze(tmp_path_factory):
    root = tmp_path_factory.mktemp("symbolic_closed_metadata")
    deps = c.read_json(c.ROOT / c.DEPS)
    names = set(deps["producer_text_sha256"]) | set(deps["direct_source_sha256"]) | set(c.OWNED)
    names |= {c.DEPS, c.DESIGN, c.ASTRA, c.REFLECTION, c.REGISTRATION, c.SNAPSHOT,
              c.DISPATCH, c.R1_DISPATCH, c.FIRST_REVIEW, c.SUPERVISOR}
    # Text metadata only, no source NPZ/header/hash/array access.
    for name in names:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(c.ROOT / name, target)
    qualification = c.BASE + "/e339-sol-r1-symbolic-current-proof.json"
    symbolic_generated_proof(root, qualification)
    return root, qualification


def symbolic_generated_proof(root, qualification):
    """Synthetic supervision records test validation; no real execution claim."""
    owned = {name: c.digest(root / name) for name in c.OWNED}
    branches, evidence = {}, {}
    for branch in ("sol", "root"):
        prefix = "e339-sol-r1-symbolic" if branch == "sol" else "parent-e339-symbolic"
        links = {field: c.BASE + "/" + prefix + "-" + field + ".json"
                 for field in ("contract", "supervision", "report")}
        sources = {**owned, c.SUPERVISOR: c.digest(root / c.SUPERVISOR)}
        commands = []
        if branch == "root":
            script = c.BASE + "/parent-e339-symbolic-generated.py"
            (root / script).write_text("# symbolic generated supervisor fixture\n", "utf8")
            sources[script] = c.digest(root / script)
        for i in range(2 if branch == "sol" else 1):
            stdout = c.BASE + "/" + prefix + f"-{i}-stdout.txt"
            stderr = c.BASE + "/" + prefix + f"-{i}-stderr.txt"
            (root / stdout).write_text("symbolic generated log\n", "utf8")
            (root / stderr).write_text("", "utf8")
            argv = [".venv/Scripts/python.exe", "-X", "utf8", "-B"]
            argv += (["-m", "ruff", "check", *c.OWNED] if i == 0 else ["-m", "pytest", c.OWNED[1]]) if branch == "sol" else [script, links["report"]]
            commands.append({"argv": argv, "stdout": stdout, "stderr": stderr, "wall_s": 10})
            evidence[stdout], evidence[stderr] = c.pin(root / stdout), c.pin(root / stderr)
        contract = {"source_sha256": sources, "commands": commands, "original_calls": 0,
                    "supervisor_output": links["supervision"], "max_tree_rss_bytes": 2 << 30}
        (root / links["contract"]).write_bytes(c.json_bytes(contract))
        supervision = {"schema": "flylab.m3.e267.root-supervision.v1", "success": True,
                       "contract_sha256": c.digest(root / links["contract"]), "original_calls": 0,
                       "attempts": [{"argv": x["argv"], "requested": True, "exit_code": 0, "killed": False}
                                    for x in commands]}
        (root / links["supervision"]).write_bytes(c.json_bytes(supervision))
        report = {"passed": True, "source_sha256": owned, "original_calls": 0}
        if branch == "sol":
            report.update(schema="flylab.m3.e339.sol-generated-qualification.v1", ruff_passed=True,
                          pytest_passed=True, pytest_items=1, real_selected_reader_generated=True,
                          real_metadata_freeze_symbolic_namespace=True, loader_mock_success_used=False,
                          production_original_qualification=False, supervision={
                              "path": links["supervision"], "sha256": c.digest(root / links["supervision"]),
                              "success": True, "original_calls": 0})
        else:
            report.update(schema="flylab.m3.e339.root-current-generated-report.v1",
                          checks={"independent_generated_fixture": True}, checks_count=1,
                          first_error=None, source_review_passed=True, R1_required=False)
        (root / links["report"]).write_bytes(c.json_bytes(report))
        for name in {*links.values(), *sources}:
            evidence[name] = c.pin(root / name)
        branches[branch] = links
    proof = {"schema": c.PROOF_SCHEMA, "passed": True, "source_sha256": owned,
             "original_calls": 0, "branches": branches, "evidence": evidence}
    (root / qualification).write_bytes(c.json_bytes(proof))
    return proof


def test_real_freeze_builder_finite_closed_producers_and_current_source(symbolic_freeze):
    root, qualification = symbolic_freeze
    frozen = c.make_freeze(root, qualification)
    assert frozen["producer_states"]["closed_e336_checks"] == 7141
    assert frozen["producer_states"]["closed_e332_checks"] == 2946
    assert frozen["original_bindings"] == c.original_bindings()
    assert set(c.read_json(root / qualification)["evidence"]).issubset(frozen["evidence"])
    assert frozen["science_adopted"] is False
    source = root / c.OWNED[0]
    before = source.read_bytes()
    try:
        source.write_bytes(before + b"\n# generated mutation\n")
        with pytest.raises(ValueError, match="current generated source"):
            c.make_freeze(root, qualification)
    finally:
        source.write_bytes(before)


def test_producer_linkage_failure_without_archive_calls(symbolic_freeze):
    root, _ = symbolic_freeze
    deps = c.read_json(root / c.DEPS)
    name = c.BASE + "/parent-e336-final-decision.json"
    path = root / name
    before = path.read_bytes()
    try:
        final = c.read_json(path)
        final["evidence_sha256"][c.BASE + "/parent-e336-saved-raw-audit.json"] = "e" * 64
        path.write_bytes(c.json_bytes(final))
        deps["producer_text_sha256"][name] = c.digest(path)
        with pytest.raises(ValueError, match="evidence edge"):
            c.verify_producers(root, deps)
    finally:
        path.write_bytes(before)


def test_symbolic_production_gate_real_builder_and_byte_pins(symbolic_freeze, monkeypatch):
    root, qualification = symbolic_freeze
    frozen = c.make_freeze(root, qualification)
    freeze_path, review_path = root / c.FREEZE, root / c.REVIEW
    freeze_path.write_bytes(c.json_bytes(frozen))
    frozen_sha = c.digest(freeze_path)
    review = {"passed": True, "freeze_sha256": frozen_sha,
              "source_sha256": {name: frozen["source_sha256"][name] for name in c.OWNED},
              "generated_qualification_sha256": frozen["evidence"][qualification]["sha256"]}
    review_path.write_bytes(c.json_bytes(review))
    review_sha = c.digest(review_path)
    monkeypatch.setattr(c, "ROOT", root)
    assert c.gate(frozen_sha, review_sha) == frozen
    with pytest.raises(ValueError, match="freeze byte pin"):
        c.gate("e" * 64, review_sha)
    with pytest.raises(ValueError, match="review byte pin"):
        c.gate(frozen_sha, "e" * 64)
    review["generated_qualification_sha256"] = "e" * 64
    review_path.write_bytes(c.json_bytes(review))
    with pytest.raises(ValueError, match="current freeze/source/proof"):
        c.gate(frozen_sha, c.digest(review_path))


@pytest.mark.parametrize("branch", ["sol", "root"])
@pytest.mark.parametrize("fault", ["stale_source", "original_calls", "failed_supervision",
                                  "contract_edge", "failed_exit", "missing_log", "mutated_log",
                                  "report_failed", "source_pin", "missing_contract"])
def test_closed_generated_proof_rejects_branch_faults(symbolic_freeze, branch, fault):
    root, qualification = symbolic_freeze
    proof = c.read_json(root / qualification)
    links = proof["branches"][branch]
    owned = {name: c.digest(root / name) for name in c.OWNED}
    modified = {}
    def mutate(name, update):
        modified[name] = (root / name).read_bytes()
        value = c.read_json(root / name)
        update(value)
        (root / name).write_bytes(c.json_bytes(value))
        proof["evidence"][name] = c.pin(root / name)
    try:
        if fault == "stale_source":
            mutate(links["report"], lambda x: x["source_sha256"].update({c.OWNED[0]: "a" * 64}))
        elif fault == "original_calls":
            mutate(links["report"], lambda x: x.update(original_calls=1))
        elif fault == "failed_supervision":
            mutate(links["supervision"], lambda x: x.update(success=False))
        elif fault == "contract_edge":
            mutate(links["supervision"], lambda x: x.update(contract_sha256="a" * 64))
        elif fault == "failed_exit":
            mutate(links["supervision"], lambda x: x["attempts"][0].update(exit_code=1))
        elif fault == "report_failed":
            mutate(links["report"], lambda x: x.update(passed=False))
        elif fault == "source_pin":
            mutate(links["contract"], lambda x: x["source_sha256"].update({c.OWNED[1]: "a" * 64}))
        elif fault == "missing_contract":
            del proof["evidence"][links["contract"]]
        else:
            name = c.read_json(root / links["contract"])["commands"][0]["stdout"]
            if fault == "missing_log":
                del proof["evidence"][name]
            else:
                modified[name] = (root / name).read_bytes()
                (root / name).write_text("changed generated stdout", "utf8")
        with pytest.raises(ValueError):
            c.verify_generated_proof(root, proof, owned)
    finally:
        for name, before in modified.items():
            (root / name).write_bytes(before)


def test_naked_proof_unsafe_extra_and_historical_failure_cannot_qualify(symbolic_freeze):
    root, qualification = symbolic_freeze
    proof = c.read_json(root / qualification)
    owned = proof["source_sha256"]
    with pytest.raises(ValueError, match="closed current"):
        c.verify_generated_proof(root, {"passed": True, "source_sha256": owned}, owned)
    with pytest.raises(ValueError, match="finite canonical"):
        extra = copy.deepcopy(proof)
        extra["evidence"]["../outside.json"] = {"sha256": "a" * 64, "bytes": 1}
        c.verify_generated_proof(root, extra, owned)
    with pytest.raises(ValueError, match="exactly the finite direct closure"):
        extra = copy.deepcopy(proof)
        extra["evidence"][c.REFLECTION] = c.pin(root / c.REFLECTION)
        c.verify_generated_proof(root, extra, owned)
    name = proof["branches"]["root"]["report"]
    before = (root / name).read_bytes()
    try:
        historical = c.read_json(root / name)
        historical.update(source_review_passed=False, R1_required=True)
        (root / name).write_bytes(c.json_bytes(historical))
        proof["evidence"][name] = c.pin(root / name)
        with pytest.raises(ValueError, match="unresolved failures"):
            c.verify_generated_proof(root, proof, owned)
    finally:
        (root / name).write_bytes(before)


def valid_terminal():
    return {"status": "valid_provisional_diagnostic", "analysis_valid": True,
            "publication_valid": True, "suggested_diagnostic": "proxy_limited",
            "provisional": True, "science_adopted": False, "candidate_adopted": False,
            "F1_F8_complete": False, "first_error": None, "secondary_errors": []}


def publication_clock():
    clock = c.Clock(0, now=lambda: 1)
    clock.begin_final()
    return clock


def test_real_reader_analysis_publication_end_to_end_generated(tmp_path):
    inputs = generated_inputs()
    decoded, ledger = {}, {"archives": {}}
    for role, values in inputs.items():
        binding = archive(tmp_path, role, values)
        decoded[role], ledger["archives"][role] = read_generated(tmp_path, role, binding)
    analysis, arrays = c.analyze(decoded)
    terminal = valid_terminal()
    terminal.update(analysis=analysis, suggested_diagnostic=analysis["diagnosis"]["suggested_diagnostic"])
    output = tmp_path / "published"
    result = c.publish(output, terminal, arrays, ledger, publication_clock())
    assert result == terminal == c.read_json(output / "result.json") == c.read_json(output / "receipt.json")
    manifest = c.read_json(output / "output-manifest.json")
    assert "output-manifest.json" not in manifest["files"]
    for name, expected in manifest["files"].items():
        assert c.pin(output / name) == expected
    with np.load(output / "live-coordination.npz", allow_pickle=False) as values:
        assert values["z"].shape == (200, 6, 2)
        assert values["encoded_z_all201"].shape == (201, 6, 2)
        assert values["exceed"].shape == (200, 15)


@pytest.mark.parametrize("fault", ["rename", "readback", "deadline", "cap"])
def test_publication_failures_are_invalid_and_revoked(tmp_path, monkeypatch, fault):
    output = tmp_path / "published"
    clock = publication_clock()
    if fault == "rename":
        def fail_rename(*args):
            raise OSError("injected rename failure")
        monkeypatch.setattr(Path, "rename", fail_rename)
    if fault == "readback":
        actual = c.pin
        def wrong_readback(path, timely=lambda: None):
            result = actual(path, timely)
            if Path(path).name == "result.json":
                result["sha256"] = "0" * 64
            return result
        monkeypatch.setattr(c, "pin", wrong_readback)
    if fault == "deadline":
        calls = [0]
        def final_clock():
            calls[0] += 1
            return 1 if calls[0] < 8 else 12
        clock.now = final_clock
    result = c.publish(output, valid_terminal(), {}, {}, clock, cap=(512 << 10) + 1 if fault == "cap" else 4 << 20)
    assert result["status"] == "technical_invalid"
    assert result["analysis_valid"] is False and result["publication_valid"] is False
    assert result["suggested_diagnostic"] is None
    assert result["science_adopted"] is False
    assert (output / "invalid-revocation.json").exists()


def test_preexisting_output_is_untouched(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "result.json"
    marker.write_bytes(b"existing immutable bytes")
    before = hashlib.sha256(marker.read_bytes()).hexdigest()
    result = c.publish(output, valid_terminal(), {}, {}, publication_clock())
    assert result["status"] == "technical_invalid"
    assert hashlib.sha256(marker.read_bytes()).hexdigest() == before
    assert list(output.iterdir()) == [marker]


def test_clock_absolute_deadlines_include_imports_gate_and_final():
    now = [7.999]
    clock = c.Clock(0, now=lambda: now[0])
    clock.science()
    now[0] = 8
    with pytest.raises(TimeoutError):
        clock.science()
    now[0] = 9
    clock.begin_final()
    now[0] = 11.999
    clock.final()
    now[0] = 12
    with pytest.raises(TimeoutError):
        clock.final()
    now[0] = 1
    clock = c.Clock(0, now=lambda: now[0])
    clock.begin_final()
    now[0] = 5
    with pytest.raises(TimeoutError):
        clock.final()


def test_late_science_failure_real_generated_reader_publication_and_main_exit(tmp_path, monkeypatch, capsys):
    inputs = generated_inputs()
    bindings = {role: archive(tmp_path, role, values) for role, values in inputs.items()}
    late = [False]
    actual_reader, actual_analyze, actual_clock = c.fixed_reader, c.analyze, c.Clock
    class LateClock(actual_clock):
        def __init__(self, s0):
            super().__init__(s0, now=lambda: 8.1 if late[0] else 1.)
    def actual_generated_reader(root, role, binding, **kwargs):
        return actual_reader(tmp_path, role, binding, namespace="generated", timely=kwargs["timely"])
    def compute_then_expire(*args):
        result = actual_analyze(*args)
        late[0] = True
        return result
    output = tmp_path / "late-science-output"
    monkeypatch.setattr(c, "OUTPUT", output.relative_to(c.ROOT).as_posix())
    monkeypatch.setattr(c, "gate", lambda *args: {"original_bindings": bindings})
    monkeypatch.setattr(c, "fixed_reader", actual_generated_reader)
    monkeypatch.setattr(c, "analyze", compute_then_expire)
    monkeypatch.setattr(c, "Clock", LateClock)
    terminal = c.worker("0" * 64, "0" * 64, 0)
    assert terminal["first_error"]["type"] == "TimeoutError"
    assert terminal["status"] == "technical_invalid"
    assert not terminal["analysis_valid"] and not terminal["publication_valid"]
    assert terminal["suggested_diagnostic"] is None
    assert terminal["analysis"]["diagnosis"]["suggested_diagnostic"] is not None
    assert c.read_json(output / "result.json") == terminal == c.read_json(output / "receipt.json")
    with np.load(output / "live-coordination.npz", allow_pickle=False) as values:
        assert values["encoded_z_all201"].shape == (201, 6, 2)
    # CLI semantics replay this actual failure terminal without any original call.
    monkeypatch.setattr(c, "worker", lambda *args: terminal)
    assert c.main(["--run", "--freeze-sha256", "0" * 64, "--review-sha256", "0" * 64]) == 1
    assert json.loads(capsys.readouterr().out) == terminal


@pytest.mark.parametrize("location", ["first_error", "secondary_errors"])
def test_publish_and_main_reject_success_markers_with_any_error(tmp_path, monkeypatch, capsys, location):
    terminal = valid_terminal()
    error = {"stage": "late.science", "type": "TimeoutError", "message": "retained failure"}
    terminal[location] = error if location == "first_error" else [error]
    output = tmp_path / "incoherent-terminal"
    result = c.publish(output, terminal, {}, {}, publication_clock())
    assert result["status"] == "technical_invalid" and not c.terminal_success(result)
    assert result["suggested_diagnostic"] is None
    assert c.read_json(output / "result.json") == result
    assert result[location] == terminal[location]
    monkeypatch.setattr(c, "worker", lambda *args: copy.deepcopy(terminal))
    assert c.main(["--run", "--freeze-sha256", "0" * 64, "--review-sha256", "0" * 64]) == 1
    serialized = json.loads(capsys.readouterr().out)
    assert serialized["status"] == "technical_invalid" and not serialized["publication_valid"]
