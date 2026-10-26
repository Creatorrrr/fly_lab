"""E333 single smooth offline efficacy direction; passive import and gate.

Only root-qualified, prospectively allocated original CLI runs can reach a
neural loader. Generated tests inject explicit dimensions and dependencies.
Original efficacy gradients from E330 are never read. A direction qualification
is bounded surrogate finite-response evidence, not gait or biological proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil

from tools import analyze_c_banc_m3_smooth_temporal as e332
from tools import verify_c_banc_m3_full_horizon_direction as e323
from tools import verify_c_banc_m3_reset_window_direction as e324
from tools import verify_c_banc_m3_temporal_direction as e330

ROOT, BASE = e323.ROOT, e323.BASE
N, GROUPS, STARTS = e324.N, e324.GROUPS, tuple(range(100, 300, 4))
CONTROLS = 300
OWNED = ("tools/verify_c_banc_m3_smooth_efficacy_direction.py", "tests/test_c_banc_smooth_efficacy_direction.py")
OUTPUT = BASE + "/e333-smooth-efficacy-direction"
SAVED = BASE + "/e332-smooth-temporal-saved"
SCHEMA = "flylab.m3.e333.smooth-efficacy-direction.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e333.smooth-efficacy-direction.review.v1"
RESULT_SCHEMA = "flylab.m3.e333.smooth-efficacy-direction.result.v1"
QUAL_SCHEMA = "flylab.m3.e333.generated-qualification.v1"
LIMITS = {**e330.LIMITS, "supervisor_wall_s": 240, "internal_wall_s": 220,
          "science_wall_s": 200, "finalization_reserve_s": 20, "cpu_threads": 1, "gpu_count": 1}
PATHS = {**e330.PATHS,
    "e332_contract": BASE + "/e332-smooth-temporal-contract.json",
    "e332_result": SAVED + "/result.json", "e332_ledger": SAVED + "/advance-ledger.json",
    "e332_manifest": SAVED + "/output-manifest.json", "e332_first": BASE + "/parent-e332-first-source-review.json",
    "e332_passive": BASE + "/parent-e332-passive-freeze.json",
    "e332_audit": BASE + "/parent-e332-saved-raw-audit.json",
    "e332_audit_source": BASE + "/parent-e332-saved-raw-audit.py",
    "e332_supervision": BASE + "/parent-e332-saved-supervision.json",
    "e332_final": BASE + "/parent-e332-final-decision.json",
    "e332_report": "docs/BANC_M3_E332_SMOOTH_TEMPORAL_RESULTS_20261001.md",
    "e332_cpu": BASE + "/parent-e332-first-cpu.json",
    "e332_cuda": BASE + "/parent-e332-first-cuda.json",
    "e332_publication": BASE + "/parent-e332-first-publication.json",
    "e332_qualification_source": BASE + "/parent-e332-independent-qualification.py",
    "e332_math_source": BASE + "/parent-e332-independent-math.py",
    "e332_publication_source": BASE + "/parent-e332-publication-probe.py",
    "e333_design": "docs/BANC_M3_E333_SMOOTH_EFFICACY_DIRECTION_DESIGN_20261001.md",
    "e333_astra": BASE + "/parent-e333-astra-design-review.md",
    "e333_astra_invocation": BASE + "/parent-e333-astra-invocation.json",
    "e333_astra_events": BASE + "/parent-e333-astra-review-v2-events.jsonl",
    "e333_pre_astra": BASE + "/parent-e333-pre-astra-design.md",
    "e333_qualification_source": BASE + "/parent-e333-generated-hookup-audit.py",
    "e333_qualification": BASE + "/parent-e333-generated-qualification.json"}
PROOF_SHA = {
    "e332_contract": "0197af57f25414babbc4e54d9e07886359d6fff71990bec3cf1bea446eb4d5a8",
    "e332_final": "fa6a9d3f06373f1efff046c8b901e81745ee1dffc9ca70ca89273b075c0d2d21",
    "e333_design": "640533e71d358bbff34a6a437e9b1bc26c20d6ad49f328528426bcd741352ceb",
    "e333_astra": "bb37f020a3b6e4654d5934903be727eaa96b3caa0c80c982e43d5a29f98eec57"}
TERM_NAMES = e330.TERM_NAMES
METRIC_NAMES = (*TERM_NAMES, "per_leg_successor_loss", "startup_q_rms", "startup_q_max",
                "startup_tendon_rms", "startup_tendon_max", "startup_pass")
bytes_sha = e332.bytes_sha
read = e332.read
window_loss = e330.window_loss
make_direction = e330.make_direction
patch_weights = e330.patch_weights
put_json = e324.put_json


@dataclass(frozen=True)
class TestDimensions:
    """Explicit generated-only dimensions; original CLI provides no override."""
    nodes: int
    groups: int
    starts: tuple[int, ...]
    test_only: bool = True

    def __post_init__(self):
        if (self.test_only is not True or type(self.nodes) is not int or self.nodes < 1
                or type(self.groups) is not int or self.groups < 1 or not self.starts
                or tuple(sorted(set(self.starts))) != self.starts
                or any(type(c) is not int or c not in STARTS for c in self.starts)):
            raise ValueError("explicit generated dimensions and ordered allowed starts required")


def _closure(root):
    prior = read(e323.safe(PATHS["e332_contract"], root))
    sources = {**prior["source_sha256"], **{name: e323.sha256(e323.safe(name, root)) for name in OWNED}}
    inputs = dict(prior["input_sha256"])
    manifest = read(e323.safe(PATHS["e332_manifest"], root))
    inputs.update({SAVED + "/" + name: row["sha256"] for name, row in manifest["files"].items()})
    for role, name in PATHS.items():
        if role.startswith(("e332_", "e333_")):
            inputs[name] = e323.sha256(e323.safe(name, root))
    for role in ("e332_first", "e332_final", "e333_qualification"):
        proof = read(e323.safe(PATHS[role], root))
        inputs.update(proof.get("evidence_sha256", {}))
    return prior, sources, inputs


def make_freeze(*, root=ROOT):
    """Passive JSON/hashes only. Authorization and allocation remain unset."""
    _, sources, inputs = _closure(root)
    saved = read(e323.safe(PATHS["e332_result"], root))
    return {"schema": SCHEMA, "paths": PATHS.copy(), "limits": LIMITS.copy(), "output": OUTPUT,
        "source_sha256": sources, "input_sha256": inputs, "candidate_adopted": False,
        "original_execution_authorized": False, "original_supervisor_required": True, "allocation": None,
        "smooth_constant_metadata": saved["reference_qualification"]["constant_metadata"],
        "gradient_semantics": "rounded native forward with real-equation surrogate VJP",
        "calculation": {"window_starts": list(STARTS), "loss_divisor": 50, "primary_max_relative_error": .05,
                        "window_L1_max_relative_error": .05, "magnitude_step": .001,
                        "meaningful_signal": 1e-9, "startup_guards": [.018, .04, .009, .025]}}


def _qualification_reports(qualification, sources, inputs, root):
    """Passive CPU/CUDA producer reports, source identities and exact byte links."""
    digests = {**inputs, **sources}
    if not qualification.get("evidence_sha256"):
        raise ValueError("nonempty current generated qualification evidence required")
    for mode in ("cpu", "cuda"):
        descriptor = qualification.get(mode + "_report", {})
        if (set(descriptor) != {"path", "sha256"} or digests.get(descriptor["path"]) != descriptor["sha256"]
                or qualification["evidence_sha256"].get(descriptor["path"]) != descriptor["sha256"]):
            raise ValueError("exact generated " + mode + " report descriptor required")
        proof = read(e323.safe(descriptor["path"], root))
        if (proof.get("schema") != "flylab.m3.e333.root-generated-hookup.v1" or proof.get("mode") != mode
                or proof.get("passed") is not True or not proof.get("checks")
                or not all(value is True for value in proof["checks"].values())
                or proof.get("source_sha256") != sources[OWNED[0]]
                or proof.get("audit_source_sha256") != inputs[PATHS["e333_qualification_source"]]
                or proof.get("math_helper_sha256") != inputs[PATHS["e332_math_source"]]
                or proof.get("checkpoint_fixture_sha256") != digests["tests/test_c_banc_checkpointed_rollout.py"]
                or any(proof.get(field) != 0 for field in ("original_calls", "whole_BANC_calls", "body_calls"))):
            raise ValueError("current generated " + mode + " producer qualification required")


def gate(freeze_path, freeze_sha256, review_path, review_sha256, output, *, root=ROOT):
    """All inherited source/input/producer proof bytes before Torch/NPZ/load."""
    if (Path(output).resolve() != e323.safe(OUTPUT, root) or Path(output).exists()
            or e323.sha256(freeze_path) != freeze_sha256 or e323.sha256(review_path) != review_sha256):
        raise ValueError("fresh literal E333 output and exact freeze/review SHA required")
    frozen, review = read(freeze_path), read(review_path)
    if (frozen.get("schema") != SCHEMA or frozen.get("paths") != PATHS or frozen.get("limits") != LIMITS
            or frozen.get("output") != OUTPUT or frozen.get("candidate_adopted") is not False
            or frozen.get("original_execution_authorized") is not True
            or frozen.get("original_supervisor_required") is not True):
        raise ValueError("exact authorized E333 freeze required")
    e332.pin_map(frozen["source_sha256"], root)
    e332.pin_map(frozen["input_sha256"], root)
    for role, digest in PROOF_SHA.items():
        if frozen["input_sha256"].get(PATHS[role]) != digest:
            raise ValueError("immutable prior/design SHA differs: " + role)
    prior, sources, inputs = _closure(root)
    if (len(prior["source_sha256"]) != 178 or len(prior["input_sha256"]) != 1542
            or frozen["source_sha256"] != sources or frozen["input_sha256"] != inputs):
        raise ValueError("exact 178source/1542input inherited closure required")
    e332.gate(e323.safe(PATHS["e332_contract"], root), PROOF_SHA["e332_contract"], Path(output), root=root)
    expected = make_freeze(root=root)
    for key in expected.keys() - {"allocation", "original_execution_authorized"}:
        if frozen.get(key) != expected[key]:
            raise ValueError("exact frozen calculation/constant contract required: " + key)
    if set(frozen) != set(expected):
        raise ValueError("exact E333 freeze fields required")
    allocation = {"cumulative_before_s": 9200.673308899522, "previous_limit_s": 9360,
                  "prospective_stage_allocation_s": 300, "cumulative_limit_s": 9660,
                  "original_reserved_s": 240, "body_reserved_s": 0}
    if frozen["allocation"] != allocation:
        raise ValueError("prospective root allocation and 240s reservation required")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != [] or review.get("freeze_sha256") != freeze_sha256
            or review.get("source_sha256") != sources):
        raise ValueError("current root-approved E333 source/freeze review required")
    qual = read(e323.safe(PATHS["e333_qualification"], root))
    if (qual.get("schema") != QUAL_SCHEMA or qual.get("approved") is not True
            or qual.get("blocking_defects") != [] or qual.get("original_calls") != 0
            or qual.get("source_sha256") != {name: sources[name] for name in OWNED}):
        raise ValueError("current source-qualified E333 generated proof required")
    for proof in (review, qual):
        for name, digest in proof.get("evidence_sha256", {}).items():
            if {**inputs, **sources}.get(name) != digest:
                raise ValueError("E333 source review/qualification evidence differs")
    _qualification_reports(qual, sources, inputs, root)
    result, ledger, audit, first, final, supervisor = (read(e323.safe(PATHS[role], root)) for role in
        ("e332_result", "e332_ledger", "e332_audit", "e332_first", "e332_final", "e332_supervision"))
    manifest = read(e323.safe(PATHS["e332_manifest"], root))
    roster = {"reference.npz", "reference-cases.npz", "e318-diagnostic.npz", "e327-diagnostic.npz",
              *(f"window-{c}.npz" for c in STARTS)}
    if set(manifest.get("files", {})) != roster:
        raise ValueError("exact 54 immutable E332 raw files required")
    for name, descriptor in manifest["files"].items():
        if descriptor != {"sha256": inputs[SAVED + "/" + name], "bytes": e323.safe(SAVED + "/" + name, root).stat().st_size}:
            raise ValueError("E332 raw manifest descriptor differs")
    for obj in (result, ledger):
        if (obj.get("analysis_valid") is not True or obj.get("input_qualification_pass") is not True
                or obj.get("first_error") is not None or obj.get("errors") != [] or obj.get("completed_windows") != 50):
            raise ValueError("completed valid E332 qualification/result/ledger required")
    passive = read(e323.safe(PATHS["e332_passive"], root))
    if (result.get("schema") != e332.RESULT_SCHEMA
            or ledger.get("schema") != "flylab.m3.e332.smooth-temporal.ledger.v1"
            or manifest.get("schema") != "flylab.m3.e332.smooth-temporal.manifest.v1"
            or {row.get("file") for row in ledger.get("attempts", [])} != roster
            or len(ledger.get("attempts", [])) != 54 or any(row.get("returned") is not True for row in ledger["attempts"])
            or first.get("passive_freeze_sha256") != inputs[PATHS["e332_passive"]]
            or first.get("saved_contract_sha256") != inputs[PATHS["e332_contract"]]
            or final.get("report") != PATHS["e332_report"] or final.get("report_sha256") != inputs[PATHS["e332_report"]]
            or passive.get("passed") is not True or passive.get("source_count") != 178 or passive.get("input_count") != 1542
            or passive.get("contract_sha256") != PROOF_SHA["e332_contract"]):
        raise ValueError("E332 passive/FIRST/report/raw producer linkage required")
    for mode, count in (("cpu",33), ("cuda",41), ("publication",4)):
        proof = read(e323.safe(PATHS["e332_" + mode], root))
        if (proof.get("passed") is not True or len(proof.get("checks", {})) != count
                or not all(value is True for value in proof["checks"].values())
                or proof.get("source_sha256") != sources[e332.OWNED[1 if mode == "publication" else 0]]
                or proof.get("audit_source_sha256") != inputs[PATHS["e332_publication_source" if mode == "publication" else "e332_qualification_source"]]
                or (mode != "publication" and proof.get("math_source_sha256") != inputs[PATHS["e332_math_source"]])):
            raise ValueError("E332 " + mode + " current source qualification required")
    if (result.get("source_sha256") != prior["source_sha256"] or result.get("input_sha256") != prior["input_sha256"]
            or result.get("contract_sha256") != PROOF_SHA["e332_contract"]
            or audit.get("passed") is not True or len(audit.get("checks", {})) != 2946
            or not all(v is True for v in audit["checks"].values())
            or first.get("approved") is not True or first.get("blocking_defects") != []
            or first.get("source_sha256") != {name: sources[name] for name in e332.OWNED}
            or (first.get("CPU_checks"), first.get("CUDA_checks"), first.get("publication_checks")) != (33, 41, 4)
            or final.get("analysis_valid") is not True or final.get("input_qualification_pass") is not True
            or final.get("root_raw_audit_checks") != 2946 or final.get("E330_local_direction") != "refuted"
            or final.get("original_cumulative_s") != 9200.673308899522 or final.get("original_allocation_limit_s") != 9360
            or final.get("original_reserved_s") != 0):
        raise ValueError("E332 source/2946audit/final proof linkage required")
    for field, role in (("result_sha256", "e332_result"), ("advance_ledger_sha256", "e332_ledger"),
                        ("output_manifest_sha256", "e332_manifest"), ("contract_sha256", "e332_contract"),
                        ("supervision_sha256", "e332_supervision"), ("audit_source_sha256", "e332_audit_source")):
        if audit.get(field) != inputs[PATHS[role]]:
            raise ValueError("E332 audit producer SHA link differs")
    if (supervisor.get("success") is not True or len(supervisor.get("attempts", [])) != 1
            or supervisor["attempts"][0].get("exit_code") != 0 or supervisor["attempts"][0].get("killed") is not False
            or not supervisor.get("owned_pids") or any(psutil.pid_exists(pid) for pid in supervisor["owned_pids"])):
        raise ValueError("terminated E332 supervisor required")
    ref, aggregate = result["reference_qualification"], result["aggregate"]
    if (ref.get("reference_progress_pass") is not True or ref.get("known_static_discrimination") is not True
            or ref.get("static_margin", 0) <= 1e-8 or aggregate.get("finite_command_qualification") is not True
            or aggregate.get("e_signed") is None or aggregate.get("e_L1") is None
            or aggregate["e_signed"] > .05 or aggregate["e_L1"] > .05
            or any(not np.all(np.asarray(value) > 1e-8) for value in ref["progress_margins"].values())):
        raise ValueError("E332 fixed progress/static/command margins required")
    invocation = read(e323.safe(PATHS["e333_astra_invocation"], root))
    cli = invocation.get("CLI_current", {})
    if (cli.get("model") != "gpt-6-astra" or cli.get("config", {}).get("model_reasoning_effort") != "max"
            or cli.get("exit_code") != 0 or cli.get("completed") is not True
            or cli.get("review_sha256") != inputs[PATHS["e333_astra"]]
            or cli.get("events_sha256") != inputs[PATHS["e333_astra_events"]]):
        raise ValueError("actual completed Astra/max invocation linkage required")
    return frozen


def classify_direction(neutral, plus, minus, window_gradients, theta_plus, theta_minus):
    """Primary T-denominator AND window-L1; negative individual T is legal."""
    arrays = [np.asarray(x) for x in (neutral, plus, minus, window_gradients, theta_plus, theta_minus)]
    n, p, m, g, tp, tm = arrays
    if (n.ndim != 1 or len(n) < 1 or p.shape != n.shape or m.shape != n.shape
            or g.ndim != 2 or g.shape[0] != len(n) or tp.shape != (g.shape[1],) or tm.shape != tp.shape
            or any(x.dtype != np.float64 or not np.isfinite(x).all() for x in arrays)
            or any(np.any(x <= 0) or not np.array_equal(x, x.astype(np.float32).astype(np.float64)) for x in (tp, tm))):
        raise ValueError("finite f64 losses/gradients and positive actual f32-promoted theta required")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        delta = tp - tm
        o = (p-m)/len(n)
        t = np.sum(g * delta[None], axis=1)
        observed, predicted = float(o.sum()), float(t.sum())
        improvement = float(n.mean()-m.mean())
        denominator = max(float(np.abs(o).sum()), float(np.abs(t).sum()))
        l1_error = None if denominator == 0 else float(np.abs(o-t).sum()) / denominator
    if not all(np.isfinite(x) for x in (observed, predicted, improvement, denominator)) or predicted < 0:
        raise ValueError("finite nonnegative aggregate prediction required")
    meaningful = abs(observed) > 1e-9 and predicted > 1e-9
    relative = abs(observed-predicted)/predicted if meaningful else None
    if any(x is not None and not np.isfinite(x) for x in (relative,l1_error)):
        raise ValueError("finite direction error ratios required")
    primary = "supported" if meaningful and observed > 0 and relative <= .05 else ("refuted" if meaningful else "inconclusive")
    l1_supported = bool(denominator > 0 and l1_error <= .05)
    supported = bool(primary == "supported" and l1_supported)
    return {"primary_local_direction": primary, "local_direction": "supported" if supported else ("refuted" if primary == "supported" else primary),
            "direction_supported": supported, "observed": observed, "predicted": predicted,
            "relative_error": relative, "window_L1_denominator": denominator, "window_L1_relative_error": l1_error,
            "window_L1_supported": l1_supported, "no_direction_information": not meaningful or denominator == 0,
            "signal_status": {"observed_meaningful": abs(observed) > 1e-9,
                              "predicted_meaningful": predicted > 1e-9,
                              "both_exact_zero": observed == 0 and predicted == 0},
            "local_descent": bool(improvement >= 1e-9), "neutral_minus_improvement": improvement,
            "window_O": o.tolist(), "window_T": t.tolist()}


def metrics(terms):
    result = {}
    for name in METRIC_NAMES:
        if name in terms:
            value = e323._host(terms[name])
            if not np.isfinite(value).all():
                raise ValueError("nonfinite smooth metric: " + name)
            result[name] = value.tolist()
    return result


def validate_masks(data, nodes=N):
    initial = data["native"]["states"]["initial_neural"]["output_mask"]
    for label in ("start", "whole100", "final"):
        mask = data[label]["network"]["output_mask"]
        e324._valid_array(mask, (nodes,), np.float32, label + " full output mask")
        if not e323.exact(mask, initial):
            raise ValueError("full control0/100/300/native supplied output masks differ")
    return {label: bytes_sha(data[label]["network"]["output_mask"]) for label in ("start", "whole100", "final")}


class Dependencies(e324.Dependencies):
    def load(self, freeze, root=ROOT):
        data = super().load(freeze, root=root)
        data["full_mask_proof"] = validate_masks(data)
        e332._clock_commands(data["raw"], "E318")
        saved = read(e323.safe(PATHS["e332_result"], root))
        original_csr_path = read(e323.safe(PATHS["e315_freeze"], root))["inputs"]["actual_csr"]["path"]
        data.update(smooth_constant_metadata=freeze["smooth_constant_metadata"],
            smooth_constants=e332.load_npz(e323.safe(SAVED + "/reference.npz", root),
                ("curves", "normalization", "temperature", "future_anchors", "phase_basis")),
            e332_neutral_terms=dict(zip(TERM_NAMES, saved["reference_qualification"]["E318_diagnostic"]["terms"], strict=True)),
            handoff_source_pins=freeze["source_sha256"].copy(),
            handoff_input_pins={PATHS[key]: freeze["input_sha256"][PATHS[key]] for key in
                ("e318_raw", "e150_calibration", "e150_parameters", "e332_result")},
            original_csr_file={"path": original_csr_path, "sha256": freeze["input_sha256"][original_csr_path]},
            original_control0_path=PATHS["e318_start"],
            original_control0_pins={name: digest for name, digest in freeze["input_sha256"].items() if name.startswith(PATHS["e318_start"] + "/")})
        return data

    def build(self, data: dict, device: str):
        import torch
        from scipy.sparse import csr_matrix

        from tools.banc_m3_checkpointed_rollout import CheckpointedControlRollout
        from tools.banc_m3_grouped_current_adjoint import GroupedCurrent
        from tools.banc_m3_monotone_adjoint import MonotoneDecoder
        from tools.banc_m3_smooth_temporal import SmoothTemporalObjective
        from tools.banc_m3_ssp_adjoint import SSPAdjointModel

        native, start = data["native"], data["start"]
        a, m, p = native["csr"], native["mask"], native["physiology"]
        csr = csr_matrix((a["data"], a["indices"], a["indptr"]), shape=(N, N))
        operator = GroupedCurrent(csr, m["positions"], m["edge_parameters"], GROUPS, device)
        neural = SSPAdjointModel(operator, p["tau"], p["coefficient"], p["threshold"],
                                 p["cap"], p["gain"], p["inverse"], .00025)
        adapter = start["adapter"]
        calibration = {key: data["calibration"][key] for key in
                       ("moment", "rest", "pool", "motor_ids", "ltm")}
        parameters = data["parameters"]
        # StateStore preserves BodySnapshot action vectors as JSON lists.
        neutral = np.asarray(start["body"]["action"]["joint_angles"])
        if (adapter["hash"] != "1fd7b86939a6846e01c4837fb5c2fbe24ce8c776e3331c1f75921a376c53b067"
                or not all(e323.exact(adapter["calibration"][key], calibration[key])
                           for key in calibration)
                or not all(e323.exact(adapter["parameters"][key], parameters[key])
                           for key in ("tau", "kappa", "slopes", "membership"))
                or not isinstance(neutral, np.ndarray) or neutral.dtype != np.float64
                or neutral.shape != (42,) or not np.isfinite(neutral).all()):
            raise ValueError("E318 start adapter differs from fixed E150 decoder")
        decoder = MonotoneDecoder(calibration, parameters["tau"], parameters["kappa"],
                                  parameters["slopes"], parameters["membership"],
                                  neutral, device=device)
        rollout = CheckpointedControlRollout(neural, decoder, calibration["motor_ids"],
                                             steps_per_control=20, control_dt=.005,
                                             checkpoint_controls=1, max_controls=CONTROLS)
        objective = SmoothTemporalObjective(data["curve"],
                                            data["startup"]["target_command"],
                                            data["startup"]["target_tendon"], device=device,
                                            normalization=data["smooth_constants"]["normalization"])
        for name, array in objective.canonical_constants.items():
            if (bytes_sha(array) != bytes_sha(data["smooth_constants"][name])
                    or objective.constant_metadata[name] != data["smooth_constant_metadata"][name]):
                raise ValueError("actual approved smooth constant bytes differ: " + name)
        data["constant_transfer"] = objective.constant_metadata.copy()
        tensor = lambda value: torch.as_tensor(np.ascontiguousarray(value), device=device)
        initial = (tensor(start["network"]["rate"]), tensor(start["network"]["adaptation"]),
                   tensor(adapter["bank"]), tensor(adapter["original"]["activation"]),
                   tensor(data["drives"]), tensor(start["network"]["output_mask"]))
        return rollout, objective, initial


def _counters(receipts, requested, replay_controls):
    if set(receipts) != set(requested) or len(requested) != len(set(requested)) or replay_controls != 300:
        raise ValueError("Exact conditional requested/confirmed receipt roster required")
    totals = dict.fromkeys(("forward_controls", "forward_neural_ticks", "recompute_controls",
                           "recompute_neural_ticks", "backward_blocks", "backward_block_ticks"), 0)
    backward_calls = 0
    for name in requested:
        count = 300 if name == "continuation-minus" else 4
        back = name.endswith("-neutral")
        expected = {"forward_controls": count, "forward_neural_ticks": 20 * count,
                    "recompute_controls": count if back else 0,
                    "recompute_neural_ticks": 20 * count if back else 0,
                    "backward_blocks": count if back else 0,
                    "backward_block_ticks": 20 * count if back else 0}
        if receipts[name].get("confirmed") != expected:
            raise ValueError("Incomplete or unexpected neural receipt: " + name)
        for key, value in expected.items():
            for suffix in ("requested", "returned"):
                field = key + "_" + suffix
                if field == "backward_block_ticks_requested":
                    continue  # Actual rollout publishes only returned block ticks.
                if receipts[name].get(field) != value:
                    raise ValueError("incomplete requested/returned receipt: " + name + " " + field)
        if (receipts[name].get("confirm_requested") != 1 + int(back)
                or receipts[name].get("confirm_returned") != 1 + int(back)):
            raise ValueError("exact distinct forward/backward confirmations required")
        for key in totals:
            totals[key] += expected[key]
        backward_calls += int(back)
    totals.update(decoder_replay_controls=replay_controls, forward_rollout_calls=len(requested),
                  backward_rollout_calls=backward_calls, body_calls=0, optimizer_steps=0, retry_count=0)
    for key, value in totals.items():
        if key in LIMITS and value > LIMITS[key]:
            raise ValueError("E333 work limit exceeded: " + key)
    return totals


def compute(data, *, dependencies=None, device="cuda", checkpoint=None, deadline=None,
            on_receipt=None, on_progress=None, on_boundary=None, test_dimensions=None):
    """Fixed original path; explicit injected dependencies permit generated fixtures."""
    import torch
    if test_dimensions is None:
        if dependencies is not None and device != "cuda":
            raise ValueError("generated CPU work requires explicit TestDimensions")
        nodes, groups, starts = N, GROUPS, STARTS
    else:
        if not isinstance(test_dimensions, TestDimensions) or dependencies is None:
            raise ValueError("explicit generated dependencies/dimensions required")
        nodes, groups, starts = test_dimensions.nodes, test_dimensions.groups, test_dimensions.starts
    N_local, GROUPS_local = nodes, groups
    for name, shape, dtype in (("drives", (300, N_local), np.float32), ("rates", (300, N_local), np.float32),
                              ("adaptations", (300, N_local), np.float32), ("motor_rates", (300, 391), np.float32)):
        e324._valid_array(data[name], shape, dtype, name)
    deps = Dependencies() if dependencies is None else dependencies
    save = checkpoint or (lambda _name, _arrays: None)
    timely = deadline or (lambda: None)
    notify = on_progress or (lambda _value: None)
    boundary = on_boundary or (lambda _name, _event: None)
    report_receipt = on_receipt or (lambda _name, _value: None)
    receipts, requested = {}, []
    command_rows, neutral_gqs = {}, {}
    result = {"conditional_fixed_neural_input": True, "window_starts": list(starts),
              "gradient_semantics": "rounded native forward with real-equation surrogate VJP",
              "new_efficacy_gradient_computed": False, "old_E330_gtheta_reused": False,
              "quantized_program_derivative_claimed": False, "full_horizon_gradient_claimed": False,
              "window_controls": 4, "steps_per_control": 20, "direction_count": 1,
              "decoder_replay_controls": 0, "decoder_replay_work": {
                  "requested_controls": 0, "host_returned_controls": 0, "device_confirmed_controls": None},
              "runs": {}, "windows": [], "requested_rollouts": requested,
              "local_direction": "not_tested", "local_descent": None,
              "native_eligibility": {"status": "not_tested"},
              "continuation": {"classification": "not_tested", "requested": False, "completed": False},
              "body_not_executed": True, "body_calls": 0, "F1_F8_complete": False,
              "candidate_adopted": False}

    def marked(name, action):
        timely()
        boundary(name, "requested")
        try:
            value = action()
        except BaseException as error:
            try:
                boundary(name, "failed")
            except Exception as secondary:  # noqa: BLE001
                error.add_note("E333 journal failure: " + repr(secondary))
            raise
        boundary(name, "host_returned")
        return value

    def retain_receipt(name, receipt, error):
        if receipt is None:
            return
        try:
            receipt.confirm()
        except Exception as secondary:  # noqa: BLE001
            error.add_note("E333 partial confirmation failure: " + repr(secondary))
        try:
            receipts[name] = e323._receipt(receipt)
            report_receipt(name, receipts[name])
        except Exception as secondary:  # noqa: BLE001
            error.add_note("E333 partial receipt journal failure: " + repr(secondary))

    rollout, objective, initial = marked("construct", lambda: deps.build(data, device))
    r0, z0, bank0, original0, drives, mask = initial
    result["constant_transfer"] = data.get("constant_transfer", {"generated_injected_objective": True})
    result["smooth_constant_metadata"] = data.get("smooth_constant_metadata", {"generated_injected_objective": True})
    result["loss_term_order"] = list(TERM_NAMES)
    result["full_mask_proof"] = data.get("full_mask_proof", {"generated_explicit_fixture": True})
    expected_device = torch.device(device)
    if expected_device.type == "cuda" and expected_device.index is None:
        expected_device = torch.device("cuda", torch.cuda.current_device())

    def tensor(value):
        return torch.as_tensor(np.ascontiguousarray(value), device=rollout.device).clone()

    def device_tensor(value, shape, dtype, name):
        if (not isinstance(value, torch.Tensor) or value.device != expected_device
                or value.shape != shape or value.dtype != dtype):
            raise ValueError("Exact rollout device/shape/dtype required: " + name)

    rollout_device = torch.device(rollout.device)
    if rollout_device.type == "cuda" and rollout_device.index is None:
        rollout_device = torch.device("cuda", torch.cuda.current_device())
    if rollout_device != expected_device:
        raise ValueError("Requested rollout device differs")
    for value, shape, dtype, name in ((r0, (N_local,), torch.float32, "initial rate"),
        (z0, (N_local,), torch.float32, "initial adaptation"), (bank0, (391, 3), torch.float64, "initial bank"),
        (original0, (96,), torch.float64, "initial original"),
        (drives, (300, N_local), torch.float32, "held drives"), (mask, (N_local,), torch.float32, "output mask")):
        device_tensor(value, shape, dtype, name)

    def replay():
        banks, originals, q_rows, t_rows = [bank0.clone()], [original0.clone()], [], []
        with torch.no_grad():
            for control, motor in enumerate(data["motor_rates"], start=1):
                timely()
                result["decoder_replay_work"]["requested_controls"] = control
                decoded = {}
                try:
                    decoded = rollout.decoder.step(tensor(motor), banks[-1].clone(), originals[-1].clone(), dt=.005)
                    for key, shape in (("q", (42,)), ("tendon", (6,)), ("bank", (391, 3)), ("original_activation", (96,))):
                        device_tensor(decoded[key], shape, torch.float64, "decoder replay " + key)
                    banks.append(decoded["bank"])
                    originals.append(decoded["original_activation"])
                    q_rows.append(decoded["q"])
                    t_rows.append(decoded["tendon"])
                    result["decoder_replay_controls"] = control
                    result["decoder_replay_work"]["host_returned_controls"] = control
                except BaseException as error:
                    partial = {"banks": np.stack([e323._host(x) for x in banks]),
                               "originals": np.stack([e323._host(x) for x in originals])}
                    if q_rows:
                        partial.update(q=np.stack([e323._host(x) for x in q_rows]), tendon=np.stack([e323._host(x) for x in t_rows]))
                    for key, value in decoded.items():
                        try:
                            partial["failed_control_" + key] = e323._host(value)
                        except Exception as secondary:  # noqa: BLE001
                            error.add_note("unavailable decoder array " + key + ": " + repr(secondary))
                    try:
                        save("decoder-replay-failed", partial)
                    except Exception as secondary:  # noqa: BLE001
                        error.add_note("partial replay save failed: " + repr(secondary))
                    raise
        replay_q, replay_t = torch.stack(q_rows), torch.stack(t_rows)
        replay_arrays = {"q": e323._host(replay_q), "tendon": e323._host(replay_t),
                         "banks": np.stack([e323._host(value) for value in banks]),
                         "originals": np.stack([e323._host(value) for value in originals]),
                         "control_tick": np.arange(301, dtype=np.int64),
                         **{f"bank{c}": e323._host(banks[c]) for c in (100, 300)},
                         **{f"original{c}": e323._host(originals[c]) for c in (100, 300)}}
        # Acquired data is committed before journaling can fail at a boundary.
        save("decoder-replay", replay_arrays)
        return banks, originals, replay_arrays

    try:
        banks, originals, replay_arrays = marked("decoder_replay", replay)
        notify(result.copy())
    except BaseException as error:
        try:
            notify(result.copy())
        except Exception as secondary:  # noqa: BLE001
            error.add_note("E333 replay journal failure: " + repr(secondary))
        raise
    result["decoder_replay_work"]["device_confirmed_controls"] = 300
    boundary("decoder_replay", "device_confirmed")
    reproduction = {}
    for key, width in (("q", 42), ("tendon", 6)):
        target = data["raw"]["command" if key == "q" else "tendon"]
        e324._valid_array(target, (301, width), np.float64, "saved " + key)
        e324._valid_array(replay_arrays[key], (300, width), np.float64, "decoder " + key)
        reproduction[key + "_max_abs"] = float(np.max(np.abs(replay_arrays[key] - target[1:])))
    for c, whole in ((100, data["whole100"]), (300, data["final"])):
        for key, dimensions in (("bank", (391, 3)), ("original", (96,))):
            target = whole["adapter"]["bank"] if key == "bank" else whole["adapter"]["original"]["activation"]
            e324._valid_array(target, dimensions, np.float64, "saved " + key)
            reproduction[key + str(c) + "_max_abs"] = float(np.max(np.abs(replay_arrays[key + str(c)] - target)))
    result["decoder_reproduction"] = reproduction
    notify(result.copy())
    if any(error > (1e-11 if key.startswith(("q_", "tendon_")) else 1e-10)
           for key, error in reproduction.items()):
        raise ValueError("E333 original decoder replay reproduction failed")

    def start_state(c):
        if c == 0:
            return r0.clone(), z0.clone(), bank0.clone(), original0.clone()
        return tensor(data["rates"][c - 1]), tensor(data["adaptations"][c - 1]), banks[c].clone(), originals[c].clone()

    def perform(c, label, theta, *, backward=False, continuation=False):
        name = "continuation-minus" if continuation else f"window-{c:03d}-{label}"
        requested.append(name)
        notify(result.copy())
        receipt, active = None, name + "_forward"
        try:
            r, z, bank, original = start_state(c)
            start_arrays = {"rate": e323._host(r), "adaptation": e323._host(z),
                "bank": e323._host(bank), "original_activation": e323._host(original),
                "fixed_original_q_boundary": data["raw"]["command"][c].copy(),
                "source_control_tick": np.array(c, np.int64),
                "source_neural_row_index": np.array(c-1, np.int64),
                "decoder_replay_boundary_index": np.array(c, np.int64)}
            save(name + "-start", start_arrays)
            timely()
            boundary(active, "requested")
            with torch.set_grad_enabled(backward):
                output = rollout.run(r, z, bank, original, drives if continuation else drives[c:c + 4], theta, mask)
            receipt = output.get("receipt")
            count = 300 if continuation else 4
            arrays, unavailable = {}, {}
            for key in ("q", "tendon", "rate", "adaptation", "bank", "original_activation"):
                try:
                    arrays[key] = e323._host(output[key])
                except Exception as acquisition_error:  # noqa: BLE001
                    unavailable[key] = {"type": type(acquisition_error).__name__, "message": str(acquisition_error)}
            receipt_snapshot = None
            if receipt is not None:
                try:
                    receipt_snapshot = e323._receipt(receipt)
                    receipts[name] = receipt_snapshot
                except Exception as acquisition_error:  # noqa: BLE001
                    unavailable["receipt"] = {"type": type(acquisition_error).__name__, "message": str(acquisition_error)}
            else:
                unavailable["receipt"] = {"type": "ValueError", "message": "receipt missing"}
            result["runs"][name] = {"forward_raw_file": name + "-forward.npz" if arrays else None,
                "unavailable_arrays": unavailable, "receipt": receipt_snapshot, "scientific_metrics_available": False}
            if arrays:
                save(name + "-forward", arrays)
            boundary(active, "host_returned")
            if receipt_snapshot is not None:
                report_receipt(name, receipt_snapshot)
            notify(result.copy())
            if unavailable:
                raise ValueError("forward arrays unavailable: " + ",".join(unavailable))
            receipt.confirm()
            receipts[name] = e323._receipt(receipt)
            report_receipt(name, receipts[name])
            boundary(active, "device_confirmed")
            active = None
            for key, dimensions, dtype in (("q", (count, 42), torch.float64), ("tendon", (count, 6), torch.float64),
                ("rate", (N_local,), torch.float32), ("adaptation", (N_local,), torch.float32),
                ("bank", (391, 3), torch.float64), ("original_activation", (96,), torch.float64)):
                device_tensor(output[key], dimensions, dtype, name + " " + key)
            for key, dimensions, dtype in (("q", (count, 42), np.float64), ("tendon", (count, 6), np.float64),
                ("rate", (N_local,), np.float32), ("adaptation", (N_local,), np.float32),
                ("bank", (391, 3), np.float64), ("original_activation", (96,), np.float64)):
                e324._valid_array(arrays[key], dimensions, dtype, name + " " + key)
            neutral_reproduction = None
            if backward:
                qerr = float(np.max(np.abs(arrays["q"] - data["raw"]["command"][c + 1:c + 5])))
                terr = float(np.max(np.abs(arrays["tendon"] - data["raw"]["tendon"][c + 1:c + 5])))
                exact = e323.exact(arrays["rate"], data["rates"][c + 3]) and e323.exact(
                    arrays["adaptation"], data["adaptations"][c + 3])
                berr = float(np.max(np.abs(arrays["bank"] - e323._host(banks[c + 4]))))
                oerr = float(np.max(np.abs(arrays["original_activation"] - e323._host(originals[c + 4]))))
                neutral_reproduction = {"q_max_abs_rad": qerr, "tendon_max_abs_command": terr,
                    "end_rate_adaptation_dtype_bytes_exact": exact, "bank_max_abs": berr, "original_max_abs": oerr}
                if qerr > 1e-11 or terr > 1e-11 or not exact or berr > 1e-10 or oerr > 1e-10:
                    raise ValueError("E333 neutral original window reproduction failed")
            terms, scaled = ((objective.evaluate(output["q"], output["tendon"]), None) if continuation
                             else window_loss(objective, tensor(data["raw"]["command"][c]), output["q"], window_count=len(starts)))
            values = metrics(terms)
            arrays["loss_terms"] = np.asarray([values[key] for key in TERM_NAMES], np.float64)
            if not continuation:
                arrays["fixed_original_q_boundary"] = data["raw"]["command"][c].copy()
            save(name, arrays)
            command_rows[name] = arrays["q"].copy()
            gradient = None
            if backward:
                before_gq = e323._receipt(receipt)
                gq = torch.autograd.grad(scaled, output["q"], retain_graph=True, create_graph=False)[0]
                gq_array = e323._host(gq)
                save(name + "-gq", {"gq": gq_array})
                after_gq = e323._receipt(receipt)
                if before_gq != after_gq:
                    raise ValueError("gq request changed neural work receipt")
                e324._valid_array(gq_array, (4,42), np.float64, name + " gq")
                neutral_gqs[c] = gq_array
                result["runs"][name]["gq_neural_receipt_unchanged"] = True
                theta.grad = None
                active = name + "_backward"
                timely()
                boundary(active, "requested")
                scaled.backward()
                if theta.grad is None:
                    raise ValueError("theta gradient missing after single backward")
                gradient = e323._host(theta.grad)
                save(name + "-gradient", {"gradient": gradient})
                boundary(active, "host_returned")
                receipt.confirm()
                receipts[name] = e323._receipt(receipt)
                report_receipt(name, receipts[name])
                boundary(active, "device_confirmed")
                active = None
                e324._valid_array(gradient, (GROUPS_local,), np.float64, name + " gradient")
            row = {"start_control": c, "label": label, "metrics": values,
                   "raw_file": name + ".npz", "forward_raw_file": name + "-forward.npz",
                   "start_file": name + "-start.npz", "receipt": receipts[name],
                   "gq_file": name + "-gq.npz" if backward else None, "theta_backward_calls": int(backward)}
            if backward:
                row.update(gq_receipt_before=before_gq, gq_receipt_after=after_gq, gq_neural_receipt_unchanged=True)
            if neutral_reproduction is not None:
                row.update(neutral_reproduction=neutral_reproduction, gradient_file=name + "-gradient.npz")
            result["runs"][name] = row
            notify(result.copy())
            timely()
            return values, gradient
        except BaseException as error:
            if active is not None:
                try:
                    boundary(active, "failed")
                except Exception as secondary:  # noqa: BLE001
                    error.add_note("E333 failure phase journal failure: " + repr(secondary))
            retain_receipt(name, receipt if receipt is not None else getattr(error, "receipt", None), error)
            if backward and theta.grad is not None:
                try:
                    save(name + "-gradient-partial", {"gradient": e323._host(theta.grad)})
                except Exception as secondary:  # noqa: BLE001
                    error.add_note("partial gradient save failed: " + repr(secondary))
            try:
                notify(result.copy())
            except Exception as secondary:  # noqa: BLE001
                error.add_note("partial scientific journal failed: " + repr(secondary))
            raise

    theta = torch.ones(GROUPS_local, dtype=torch.float64, device=rollout.device, requires_grad=True)
    neutral, gradients = [], []
    for c in starts:
        value, gradient = perform(c, "neutral", theta, backward=True)
        neutral.append(value)
        gradients.append(gradient)
    neutral_mean = {key: float(np.mean([row[key] for row in neutral], dtype=np.float64)) for key in TERM_NAMES}
    if any(not np.isfinite(data["e332_neutral_terms"][key])
           or abs(neutral_mean[key] - data["e332_neutral_terms"][key]) > 1e-10 for key in TERM_NAMES):
        raise ValueError("E333 neutral 50-window terms differ from E332 full200-pair result")
    gradient = np.sum(np.stack(gradients), axis=0, dtype=np.float64)
    e324._valid_array(gradient, (GROUPS_local,), np.float64, "aggregate gradient")
    result["new_efficacy_gradient_computed"] = True
    result["neutral_mean_terms"] = neutral_mean
    direction = make_direction(gradient)
    if direction is None:
        save("zero-gradient", {"gradient": gradient, "window_gradients": np.stack(gradients)})
        result.update(local_direction="inconclusive", primary_local_direction="inconclusive",
                      direction_supported=False, no_direction_information=True, relative_error=None,
                      window_L1_relative_error=None, window_L1_denominator=None, window_L1_supported=False,
                      local_descent=None, zero_gradient=True)
        result["continuation"]["reasons"] = ["finite_exact_zero_gradient"]
        result["native_eligibility"]["reason"] = "no_direction_created"
        result["actual_work"] = _counters(receipts, requested, result["decoder_replay_controls"])
        notify(result.copy())
        return result, receipts
    direction["window_gradients"] = np.stack(gradients)
    plus, minus = direction["theta_plus"], direction["theta_minus"]
    direction.update(theta_plus_from_one=plus-1, theta_minus_from_one=minus-1,
                     between_candidates_delta=plus-minus)
    result["direction"] = {"gradient_max_abs": float(np.max(np.abs(gradient))),
        "nonzero_gradient_groups": int(np.count_nonzero(gradient)), "theta_plus_min": float(plus.min()),
        "theta_plus_max": float(plus.max()), "theta_minus_min": float(minus.min()), "theta_minus_max": float(minus.max()),
        "actual_rounded_displacement_sha256": bytes_sha(plus - minus)}
    patched = {}
    for label, gain in (("plus", plus), ("minus", minus)):
        csr, actual, info = marked("native_eligibility_" + label, lambda gain=gain: patch_weights(data["native"], gain))
        patched[label] = (csr, actual, info)
    result["native_eligibility"] = {label: info for label, (_, _, info) in patched.items()}
    notify(result.copy())  # immediately preserve palette eligibility before +/-
    save("direction", direction)
    save("theta-minus", {"theta64": minus, "theta32": minus.astype(np.float32)})
    for label, (csr, actual, _info) in patched.items():
        save("patched-csr-" + label, csr)
        save("selected-weights-" + label, {"actual32": actual,
            "original32": data["native"]["csr"]["data"][data["native"]["mask"]["positions"]],
            "positions": data["native"]["mask"]["positions"], "groups": data["native"]["mask"]["edge_parameters"]})
    save("selected-weights-between", {"actual_plus32": patched["plus"][1], "actual_minus32": patched["minus"][1],
         "between_candidates_actual32_delta_f64": patched["plus"][1].astype(np.float64)-patched["minus"][1].astype(np.float64)})
    plus_rows, minus_rows = [], []
    for c, baseline, g in zip(starts, neutral, gradients, strict=True):
        pv, _ = perform(c, "plus", tensor(plus))
        mv, _ = perform(c, "minus", tensor(minus))
        plus_rows.append(pv)
        minus_rows.append(mv)
        dq = command_rows[f"window-{c:03d}-plus"] - command_rows[f"window-{c:03d}-minus"]
        o = (pv["training_loss"]-mv["training_loss"])/len(starts)
        t = float(np.sum(g*(plus-minus)))
        q_response = float(np.sum(neutral_gqs[c]*dq))
        if not all(np.isfinite(x) for x in (o,t,q_response)):
            raise ValueError("nonfinite window O/Q/T")
        if not np.isclose(o-t, (o-q_response)+(q_response-t), rtol=1e-12, atol=1e-12):
            raise ValueError("O/Q/T residual closure differs")
        save(f"window-{c}-contributions", {"delta_q": dq, "O": np.array(o), "Q": np.array(q_response),
            "T": np.array(t), "O_minus_Q": np.array(o-q_response), "Q_minus_T": np.array(q_response-t)})
        result["windows"].append({"start_control": c, "neutral": baseline, "plus": pv, "minus": mv,
            "observed_contribution": o, "predicted_contribution": t, "command_contribution": q_response,
            "O_minus_Q": o-q_response, "Q_minus_T": q_response-t})
        notify(result.copy())
    plus_mean = {key: float(np.mean([row[key] for row in plus_rows])) for key in TERM_NAMES}
    minus_mean = {key: float(np.mean([row[key] for row in minus_rows])) for key in TERM_NAMES}
    comparison = classify_direction(np.array([row["training_loss"] for row in neutral]),
        np.array([row["training_loss"] for row in plus_rows]), np.array([row["training_loss"] for row in minus_rows]),
        np.stack(gradients), plus, minus)
    result.update(comparison, plus_mean_terms=plus_mean, minus_mean_terms=minus_mean)
    reasons = e330.continuation_reasons(result["local_direction"], result["local_descent"], patched["minus"][2]["eligible"])
    result["continuation"]["reasons"] = reasons
    notify(result.copy())
    if not reasons:
        result["continuation"]["requested"] = True
        value, _ = perform(0, "minus", tensor(minus), continuation=True)
        improvement = neutral_mean["training_loss"] - value["training_loss"]
        startup = bool(value["startup_pass"] and value["startup_q_rms"] <= .018 and value["startup_q_max"] <= .04
                       and value["startup_tendon_rms"] <= .009 and value["startup_tendon_max"] <= .025)
        result["continuation"].update(completed=True, metrics=value, improvement=improvement,
            startup_pass=startup, classification="supported" if improvement >= 1e-9 and startup else "rejected",
            original_control0_used=True, intermediate_saved_boundaries_overwritten=False)
    result["residual_rankings"] = {key: [row["start_control"] for row in sorted(result["windows"], key=lambda row: abs(row[key]), reverse=True)]
                                  for key in ("O_minus_Q", "Q_minus_T")}
    source_identity = data["native"].get("network_identity", "generated-source-network")
    candidate_identity = hashlib.sha256((source_identity + bytes_sha(minus)
        + patched["minus"][2]["patched_csr_data32_sha256"]).encode("ascii")).hexdigest()
    if patched["minus"][2]["selected_changed_bit_count"] and candidate_identity == source_identity:
        raise ValueError("Changed candidate content identity must differ from source identity")
    result["e334_handoff"] = {"eligible": result["continuation"]["classification"] == "supported",
        "theta_file": "theta-minus.npz", "theta64_sha256": bytes_sha(minus),
        "source_network_identity": source_identity, "candidate_content_identity": candidate_identity,
        "candidate_identity_kind": "source identity plus exact theta/CSR content; native identity must be recomputed by E334",
        "candidate_native_identity_claimed_equal_to_source": False,
        "original_control0_path": data.get("original_control0_path", "generated-control0"),
        "original_control0_pins": data.get("original_control0_pins", {}),
        "source_csr_file": data.get("original_csr_file", {}),
        "decoder_source_identity": data.get("start", {}).get("adapter", {}).get("hash", "generated-decoder"),
        "source_pins": data.get("handoff_source_pins", {}), "native_eligibility": patched["minus"][2],
        "input_pins": data.get("handoff_input_pins", {}),
        "approved_smooth_constants": data.get("smooth_constant_metadata", {}),
        "reset_control0_arrays_file": "continuation-minus-start.npz",
        "startup_pass": result["continuation"].get("startup_pass"),
        "use_e333_terminal_state_as_body_start": False, "body_not_executed": True}
    result["actual_work"] = _counters(receipts, requested, result["decoder_replay_controls"])
    notify(result.copy())
    return result, receipts


def _output_bytes(output):
    return sum(p.stat().st_size for p in output.rglob("*") if p.is_file())


def run(freeze_path, freeze_sha256, review_path, review_sha256, output, *, root=ROOT,
        gatekeeper=gate, dependencies=None, clock=time.perf_counter, writer=put_json,
        saver=np.savez_compressed, meter=e323.usage, device="cuda", test_dimensions=None,
        compute_fn=compute, emergency_writer=e332.write_json, repair_writer=e332._replace_terminal_json,
        output_meter=_output_bytes):
    """One attempt, strict finite JSON, partial raw and terminal qualification repair.

    Recovery sink failure cannot guarantee disk/return agreement; the returned
    state remains invalid and requires external supervisor/root byte audit.
    """
    began = clock()
    freeze = gatekeeper(freeze_path, freeze_sha256, review_path, review_sha256, Path(output), root=root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ledger = {"schema": "flylab.m3.e333.advance-ledger.v1", "limits": LIMITS.copy(), "attempts": [],
              "phases": {}, "receipts": {}, "errors": [], "retry_count": 0,
              "body_calls": 0, "optimizer_steps": 0, "original_calls": int(test_dimensions is None)}
    result = {"schema": RESULT_SCHEMA, "run_valid": False, "classification": "not_tested",
              "local_direction": "not_tested", "effective_direction_qualification": False,
              "effective_continuation_qualification": False, "candidate_adopted": False,
              "F1_F8_complete": False, "biological_gait_proven": False, "whole_goal_status": "active",
              "E330_local_direction": "refuted", "E327_body_verdict": "rejected", "F6_exposed": False,
              "M2_modified": False, "body_not_executed": True, "body_calls": 0,
              "provenance": {"freeze_sha256": freeze_sha256, "review_sha256": review_sha256},
              "publication": {"external_supervisor_and_root_byte_audit_required": True,
                              "recovery_sink_failure_guarantees_disk_agreement": False},
              "interpretation": {"neural_Jacobian_independently_proven": False, "loss_decrease_proves_motion": False,
                                 "static_fixedpoints_excluded": False, "high_R_proves_true_phase": False,
                                 "gradient_is_quantized_program_derivative": False}}
    baseline, science_began = None, None
    science_deadline = began + LIMITS["internal_wall_s"] - LIMITS["finalization_reserve_s"]

    def failure(stage, exception):
        ledger["errors"].append({"stage": stage, "type": type(exception).__name__,
                                 "message": str(exception)[:1000], "elapsed_s": clock()-began,
                                 "notes": [str(note) for note in getattr(exception, "__notes__", [])]})

    def persist():
        ledger["elapsed_s"] = clock()-began
        writer(output / "advance-ledger.json", ledger)

    def deadline():
        if clock() >= science_deadline:
            raise TimeoutError("E333 preparation/science deadline exceeded")

    def check_resource(value):
        if baseline is None:
            return
        if value["device_total_bytes"] != baseline["device_total_bytes"]:
            raise RuntimeError("E333 device identity changed")
        process = psutil.Process()
        rss = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        ledger["owned_tree_sampled_peak_bytes"] = max(ledger.get("owned_tree_sampled_peak_bytes", 0), rss)
        ledger["resource_latest"] = value
        ledger["additional_gpu_peak_bytes"] = max(ledger.get("additional_gpu_peak_bytes", 0),
            baseline["device_free_bytes"]-value["device_free_bytes"])
        ledger["additional_torch_peak_reserved_bytes"] = max(ledger.get("additional_torch_peak_reserved_bytes", 0),
            value["torch_peak_reserved_bytes"]-baseline["torch_reserved_bytes"])
        if (max(rss, value["host_peak_bytes"]) > LIMITS["host_bytes"]
                or ledger["additional_gpu_peak_bytes"] > LIMITS["additional_gpu_bytes"]
                or ledger["additional_torch_peak_reserved_bytes"] > LIMITS["additional_gpu_bytes"]):
            raise MemoryError("E333 owned-tree/device resource limit exceeded")

    def checkpoint(name, arrays):
        # Failure arrays may contain NaN/Inf, but their JSON descriptors never do.
        if Path(name).name != name or (output / (name + ".npz")).exists():
            raise ValueError("fresh owned raw basename required")
        path = output / (name + ".npz")
        entry = {"file": path.name, "requested": True, "returned": False,
                 "requested_elapsed_s": clock()-began}
        ledger["attempts"].append(entry)
        try:
            saver(path, **arrays)
            entry.update(returned=True, bytes=path.stat().st_size, sha256=e323.sha256(path),
                         returned_elapsed_s=clock()-began)
        except Exception as exception:
            entry.update(error_type=type(exception).__name__, error_message=str(exception))
            raise
        if output_meter(output) > LIMITS["output_bytes"]:
            raise OSError("E333 raw output limit exceeded")
        persist()
        deadline()

    def on_receipt(name, value):
        ledger["receipts"][name] = value
        persist()

    def on_progress(value):
        result["partial"] = value
        persist()

    def on_boundary(name, event):
        if event == "requested":
            if name in ledger["phases"]:
                raise ValueError("phase requested twice: " + name)
            ledger["phases"][name] = {"requested": True, "host_returned": False, "device_confirmed": False, "failed": False}
        elif event in ("host_returned", "device_confirmed", "failed"):
            ledger["phases"][name][event] = True
        else:
            raise ValueError("unknown phase event")
        ledger["phases"][name]["elapsed_s"] = clock()-began
        persist()
        if event == "device_confirmed" or (name == "construct" and event == "host_returned"):
            check_resource(meter())
            deadline()

    try:
        persist()
        deadline()
        if shutil.disk_usage(output).free < LIMITS["minimum_free_disk_bytes"]:
            raise OSError("E333 free disk below 4GiB")
        import torch
        torch.set_num_threads(1)
        if device == "cuda":
            torch.cuda.init()
            torch.empty(1, device="cuda")
            torch.cuda.synchronize()
            ledger["child_torch_cuda_before_input_loader"] = True
            ledger["torch_device_name"] = torch.cuda.get_device_name()
        elif test_dimensions is None:
            raise ValueError("original E333 requires CUDA/full dimensions")
        baseline = meter()
        ledger["resource_baseline"] = baseline
        ledger["resource_peak_limitation"] = "sampled owned-tree and total-device use; unseen transient peaks not excluded"
        check_resource(baseline)
        writer(output / "source-manifest.json", {"schema": "flylab.m3.e333.source-manifest.v1",
            "freeze_sha256": freeze_sha256, "review_sha256": review_sha256,
            "source_sha256": freeze["source_sha256"], "input_sha256": freeze["input_sha256"]})
        deps = Dependencies() if dependencies is None else dependencies
        on_boundary("input_load", "requested")
        try:
            data = deps.load(freeze, root=root)
        except BaseException as error:
            try:
                on_boundary("input_load", "failed")
            except Exception as secondary:  # noqa: BLE001
                error.add_note("input load journal failure: " + repr(secondary))
            raise
        on_boundary("input_load", "host_returned")
        deadline()
        check_resource(meter())
        science_began = clock()
        science_deadline = min(science_began+LIMITS["science_wall_s"], began+LIMITS["internal_wall_s"]-LIMITS["finalization_reserve_s"])
        ledger.update(science_started_elapsed_s=science_began-began, science_deadline_elapsed_s=science_deadline-began)
        computed, receipts = compute_fn(data, dependencies=deps, device=device, checkpoint=checkpoint,
            deadline=deadline, on_receipt=on_receipt, on_progress=on_progress, on_boundary=on_boundary,
            test_dimensions=test_dimensions)
        result.update(computed)
        ledger["receipts"] = receipts
        deadline()
        e332.pin_map(freeze["source_sha256"], root, deadline)
        e332.pin_map(freeze["input_sha256"], root, deadline)
        if e323.sha256(freeze_path) != freeze_sha256 or e323.sha256(review_path) != review_sha256:
            raise ValueError("post-computation freeze/review SHA differs")
        check_resource(meter())
    except Exception as exception:  # noqa: BLE001
        failure("science", exception)
    finalization_began = clock()
    ledger["science_wall_s"] = 0 if science_began is None else finalization_began-science_began

    def terminal_state():
        valid = not ledger["errors"]
        result.update(run_valid=valid, first_error=ledger["errors"][0] if ledger["errors"] else None,
                      errors=ledger["errors"].copy(), elapsed_s=clock()-began)
        result["classification"] = result.get("local_direction", "not_tested") if valid else "invalid"
        result["effective_direction_qualification"] = bool(valid and result.get("direction_supported") is True)
        result["effective_continuation_qualification"] = bool(valid and result.get("continuation", {}).get("classification") == "supported")
        if "e334_handoff" in result:
            result["e334_handoff"]["eligible"] = result["effective_continuation_qualification"]
        ledger.update(run_valid=valid, first_error=result["first_error"], elapsed_s=result["elapsed_s"],
                      effective_direction_qualification=result["effective_direction_qualification"],
                      effective_continuation_qualification=result["effective_continuation_qualification"],
                      finalization_wall_s=clock()-finalization_began,
                      publication_boundary="before_terminal_metadata_commit")
        result["cost"] = ledger.copy()

    try:
        check_resource(meter())
        writer(output / "receipts.json", {"schema": "flylab.m3.e333.receipts.v1", "receipts": ledger["receipts"],
             "requested_roster": result.get("requested_rollouts", result.get("partial", {}).get("requested_rollouts", []))})
        files = {p.relative_to(output).as_posix(): {"sha256": e323.sha256(p), "bytes": p.stat().st_size}
                 for p in output.rglob("*") if p.is_file() and p.name not in
                 ("result.json", "advance-ledger.json", "output-manifest.json", "publication-failure.json")
                 and not p.name.endswith(".pending")}
        writer(output / "output-manifest.json", {"schema": "flylab.m3.e333.output-manifest.v1", "files": files,
            "mutable_metadata_excluded": ["result.json", "advance-ledger.json", "publication-failure.json"],
            "external_audit_pins_required": True})
        if "e334_handoff" in result:
            result["e334_handoff"]["output_file_sha256"] = {name: files[name]["sha256"] for name in
                ("theta-minus.npz", "patched-csr-minus.npz", "selected-weights-minus.npz", "continuation-minus-start.npz") if name in files}
        terminal_state()
        writer(output / "advance-ledger.json", ledger)
        writer(output / "result.json", result)
        if clock()-began >= LIMITS["internal_wall_s"] or clock()-finalization_began > LIMITS["finalization_reserve_s"]:
            raise TimeoutError("E333 postcommit finalization/internal deadline exceeded")
        if output_meter(output) > LIMITS["output_bytes"]:
            raise OSError("E333 postcommit output limit exceeded")
        if read(output / "result.json") != result or read(output / "advance-ledger.json") != ledger:
            raise OSError("terminal disk/result/ledger mismatch")
    except Exception as exception:  # noqa: BLE001
        failure("publication", exception)
        terminal_state()
        try:
            emergency_writer(output / "publication-failure.json", result)
        except Exception as secondary:  # noqa: BLE001
            failure("emergency_publication", secondary)
        terminal_state()
        repaired = []
        for name, value in (("advance-ledger.json", ledger), ("result.json", result)):
            try:
                repair_writer(output / name, value)
                repaired.append((name, value))
            except Exception as secondary:  # noqa: BLE001
                failure("terminal_repair_" + name, secondary)
                terminal_state()
        if any(error["stage"].startswith("terminal_repair_") for error in ledger["errors"]):
            terminal_state()
            for name, value in repaired:
                try:
                    repair_writer(output / name, value)
                except Exception as secondary:  # noqa: BLE001
                    failure("terminal_refresh_" + name, secondary)
            terminal_state()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("make-freeze")
    prepare.add_argument("--output", type=Path, required=True)
    original = commands.add_parser("run")
    for name in ("freeze", "review", "output"):
        original.add_argument("--" + name, type=Path, required=True)
    original.add_argument("--freeze-sha256", required=True)
    original.add_argument("--review-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "make-freeze":
        e332.write_json(args.output, make_freeze())
        return 0
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = run(args.freeze, args.freeze_sha256, args.review, args.review_sha256, args.output)
    print(json.dumps({"run_valid": result["run_valid"], "classification": result["classification"],
                      "first_error": result["first_error"]}, allow_nan=False))
    return 0 if result["run_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

