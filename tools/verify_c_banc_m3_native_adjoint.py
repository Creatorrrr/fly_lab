"""E315: bounded native SSP adjoint qualification, never a gait candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import psutil
import torch
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "flylab.m3.e315.native-adjoint-probe.freeze.v1"
QUALIFICATION = "flylab.m3.e314.native-adjoint-generated-qualification.v1"
OWNED = ("tools/verify_c_banc_m3_native_adjoint.py", "tests/test_c_banc_native_adjoint.py")
PRODUCERS = (
    "tools/banc_m3_grouped_current_adjoint.py",
    "tests/test_c_banc_grouped_current_adjoint.py",
    "tools/banc_m3_ssp_adjoint.py",
    "tests/test_c_banc_ssp_adjoint.py",
)
SOURCES = (*OWNED, *PRODUCERS)
INPUTS = ("actual_csr", "motor_configuration", "load_parameters", "recurrent_mask", "e289_roster")
STATES = ("initial_neural", "first_control_neural", "final_whole")
N, EDGES, POPULATION, GROUPS = 158706, 11584852, 7136, 8547
SELECTED, DT, EPS = 725849, .00025, .001
SCIENCE_S, FINALIZATION_S = 45., 5.


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe(path: str, *, root: Path = ROOT) -> Path:
    if not isinstance(path, str) or not path or Path(path).is_absolute():
        raise ValueError("Repo-relative input path required")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Input escapes repository")
    return resolved


def receipt(entry: dict, *, root: Path = ROOT, key: str = "sha256") -> Path:
    if not isinstance(entry, dict) or set(entry) != {"path", key}:
        raise ValueError("Exact path/hash descriptor required")
    path = safe(entry["path"], root=root)
    if digest(path) != entry[key]:
        raise ValueError("Pinned input hash differs: " + entry["path"])
    return path


def gate(freeze_path: Path, output: Path, *, root: Path = ROOT) -> tuple[dict, dict]:
    if output.exists():
        raise FileExistsError("Fresh output directory required")
    freeze = json.loads(freeze_path.read_text("utf8"))
    if freeze.get("schema") != SCHEMA or freeze.get("candidate_adopted") is not False:
        raise ValueError("E315 diagnostic-only freeze required")
    sources = freeze.get("source_sha256")
    if not isinstance(sources, dict) or set(sources) != set(SOURCES):
        raise ValueError("Exact six E315/E314 source pins required")
    for name in SOURCES:
        if digest(safe(name, root=root)) != sources[name]:
            raise ValueError("Source differs: " + name)
    qualification_path = receipt(freeze.get("qualification"), root=root)
    qualification = json.loads(qualification_path.read_text("utf8"))
    checks = qualification.get("checks", {})
    if (qualification.get("schema") != QUALIFICATION or qualification.get("approved") is not True
            or qualification.get("source_sha256") != {name: sources[name] for name in PRODUCERS}
            or not isinstance(qualification.get("evidence_sha256"), dict)
            or not qualification["evidence_sha256"]
            or any(checks.get(key) is not True for key in
                   ("cpu_independent_pass", "cuda_neutral_exact", "cuda_adjoint_pass"))
            or any(qualification.get(key) != 0 for key in
                   ("original_model_runs", "body_runs", "training_runs"))):
        raise ValueError("Approved E314 CPU/CUDA qualification required")
    for name, value in qualification["evidence_sha256"].items():
        if digest(safe(name, root=root)) != value:
            raise ValueError("E314 evidence differs: " + name)
    inputs = freeze.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {*INPUTS, *STATES}:
        raise ValueError("Exact E315 input aliases required")
    for name in INPUTS:
        receipt(inputs[name], root=root)
    for name in STATES:
        entry = inputs[name]
        if not isinstance(entry, dict) or set(entry) != {"path", "manifest_sha256", "state_sha256"}:
            raise ValueError("Pinned StateStore descriptor required")
        folder = safe(entry["path"], root=root)
        for file, field in (("manifest.json", "manifest_sha256"), ("state.json", "state_sha256")):
            if digest(folder / file) != entry[field]:
                raise ValueError("Pinned StateStore file differs")
    if not isinstance(freeze.get("producer_reviews", []), list):
        raise TypeError("Producer review descriptors required")
    for entry in freeze.get("producer_reviews", []):
        receipt(entry, root=root)
    if freeze.get("network_identity") != "f9ea25308cf5e074047bf16afba55de9fac89411ca48127f2c37cc4e6e34c7be":
        raise ValueError("Exact SSP/u0 network identity required")
    return freeze, qualification


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def exact(left: np.ndarray, right: np.ndarray) -> bool:
    return (left.dtype == right.dtype and left.shape == right.shape
            and left.tobytes(order="C") == right.tobytes(order="C"))


def same_tree(left, right) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and exact(left, right)
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys()
                and all(same_tree(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(same_tree(a, b) for a, b in zip(left, right, strict=True)))
    return type(left) is type(right) and left == right


def load_inputs(freeze: dict, *, root: Path = ROOT, npz_loader=load_npz, state_loader=None) -> dict:
    if state_loader is None:
        from flylab.c.storage import StateStore

        state_loader = StateStore.load
    arrays = {key: npz_loader(safe(freeze["inputs"][key]["path"], root=root)) for key in INPUTS}
    states = {key: state_loader(safe(freeze["inputs"][key]["path"], root=root)) for key in STATES}
    csr, config, load, mask, roster = (arrays[key] for key in INPUTS)
    require(set(csr) == {"indptr", "indices", "data", "selected_edge_ids", "selected_values"}, "Actual CSR schema")
    require(csr["indptr"].dtype == np.int32 and csr["indptr"].shape == (N + 1,)
            and csr["indices"].dtype == np.int32 and csr["indices"].shape == (EDGES,)
            and csr["data"].dtype == np.float32 and csr["data"].shape == (EDGES,), "Actual CSR dimensions")
    require(set(load) == {"threshold32", "tau32"}, "Pinned E313 load schema")
    require(set(mask) == {"population", "positions", "edge_parameters", "source_weights"}, "E29 mask schema")
    require(mask["population"].dtype == np.int64 and mask["population"].shape == (POPULATION,)
            and mask["positions"].dtype == np.int64 and mask["positions"].shape == (SELECTED,)
            and mask["edge_parameters"].dtype == np.int64 and mask["edge_parameters"].shape == (SELECTED,), "E29 selection dimensions")
    require(exact(mask["population"], roster["premotor_ids"])
            and exact(mask["positions"], roster["edge_ids"]), "E29 and E289 masks differ")
    require(np.array_equal(np.unique(mask["edge_parameters"]), np.arange(GROUPS)), "Original group count/order")
    require(exact(roster["weights32"], csr["data"][mask["positions"]]),
            "Current selected weights differ")
    require(np.array_equal(csr["indptr"][[0, -1]], [0, EDGES])
            and np.all(np.diff(csr["indptr"]) >= 0)
            and np.all((csr["indices"] >= 0) & (csr["indices"] < N))
            and np.isfinite(csr["data"]).all(), "Valid original CSR required")
    physiology = {"tau": load["tau32"], "threshold": load["threshold32"],
                  "gain": config["cell_gain"]}
    for key, value in physiology.items():
        require(value.dtype == np.float32 and value.shape == (N,) and np.isfinite(value).all(),
                "Resolved physiology: " + key)
    require(exact(physiology["tau"], config["membrane_tau"])
            and np.all(physiology["threshold"] == np.float32(7.5))
            and np.all(physiology["tau"] > 0), "Original threshold/membrane tau required")
    initial, first, whole = (states[key] for key in STATES)
    final = whole["network"]
    require(initial["schema"] == "flylab.research-rate.v1"
            and initial["recurrent_resource"]["schema"] == "flylab.m3-ssprk3-recovery-resource.v1"
            and initial["recurrent_resource"]["source_model_hash"]
            == "a31f899cd948a4297a73db66569e5edc2210ac9aa3601aa83e96cb9ff68ce244"
            and initial["motor_timescale"]["source_motor_identity"]
            == "a8310c1cff7e7dfa4b71b506fa3828a29b0450d8e62ddd17b611c22ee59ca0a8",
            "Exact SSP source hierarchy required")
    for name, state, tick in (("initial", initial, 0), ("first", first, 20), ("final", final, 2000)):
        require(state["tick"] == tick and state["dt"] == DT and state["device"] == "cuda"
                and state["adaptation_gain"] == 8. and state["adaptation_tau_s"] == .3
                and state["schema"] == initial["schema"],
                "Exact native state metadata: " + name)
        for key in ("rate", "adaptation", "drive", "output_mask"):
            value = state[key]
            require(value.dtype == np.float32 and value.shape == (N,) and np.isfinite(value).all(),
                    "Finite native state: " + name + "/" + key)
        require(np.all(state["rate"] >= 0) and np.all(state["adaptation"] >= 0)
                and np.isin(state["output_mask"], [0., 1.]).all(), "Nonnegative state/binary mask")
        require(state["model_hash"] == initial["model_hash"]
                and state["model_hash"] == freeze["network_identity"]
                and same_tree(state["motor_timescale"], initial["motor_timescale"])
                and same_tree(state["recurrent_resource"], initial["recurrent_resource"]),
                "Same native neural identity/kinetics")
    require(exact(initial["rate"], np.zeros(N, np.float32))
            and exact(initial["adaptation"], np.zeros(N, np.float32))
            and exact(initial["drive"], np.zeros(N, np.float32))
            and np.all(initial["output_mask"] == 1)
            and np.all(initial["synaptic_resource"] == 1)
            and np.all(first["synaptic_resource"] == 1)
            and np.all(final["synaptic_resource"] == 1), "Neutral u0 states required")
    inverse = initial["motor_timescale"]["inverse32"]
    require(inverse.dtype == np.float32 and inverse.shape == (N,)
            and np.isfinite(inverse).all() and np.all(inverse > 0), "Actual full inverse adaptation tau")
    require(config["motor_ids"].dtype == np.int64 and config["motor_ids"].shape == (391,)
            and np.all((config["motor_ids"] >= 0) & (config["motor_ids"] < N))
            and len(np.unique(config["motor_ids"])) == 391,
            "Exact motor override IDs required")
    require(np.array_equal(config["actual_motor_inverse32"], inverse[config["motor_ids"]])
            and np.array_equal(config["actual_motor_gain32"], physiology["gain"][config["motor_ids"]]),
            "Motor gain/tau overrides differ")
    physiology["coefficient"] = np.full(N, np.float32(1. / 200.), np.float32)
    physiology["cap"] = np.full(N, 200., np.float32)
    physiology["inverse"] = inverse
    return {"csr": csr, "mask": mask, "roster": roster, "states": states,
            "physiology": physiology, "network_identity": initial["model_hash"]}


def screen(observed: float, predicted: float) -> dict:
    finite = bool(np.isfinite(observed) and np.isfinite(predicted))
    meaningful = finite and abs(observed) > 1e-7 and abs(predicted) > 1e-7
    relative = abs(observed - predicted) / abs(predicted) if meaningful else None
    classification = ("supported" if meaningful and np.sign(observed) == np.sign(predicted)
                      and relative <= .05 else "refuted" if meaningful else "inconclusive")
    return {"observed_delta": observed if np.isfinite(observed) else None,
            "predicted_delta": predicted if np.isfinite(predicted) else None,
            "relative_error": relative, "supported": classification == "supported",
            "classification": classification}


def compute(data: dict, ledger: dict, *, device: str = "cuda", operator_type=None,
            model_type=None, partial=None, deadline=None, journal=None, checkpoint=None) -> tuple[dict, dict]:
    if operator_type is None:
        from tools.banc_m3_grouped_current_adjoint import GroupedCurrent

        operator_type = GroupedCurrent
    if model_type is None:
        from tools.banc_m3_ssp_adjoint import SSPAdjointModel

        model_type = SSPAdjointModel
    a, m, p = data["csr"], data["mask"], data["physiology"]
    partial = {} if partial is None else partial
    journal = (lambda _ledger: None) if journal is None else journal
    checkpoint = (lambda _partial: None) if checkpoint is None else checkpoint
    def timely():
        if deadline is not None and time.perf_counter() > deadline:
            raise TimeoutError("E315 internal science deadline exceeded")
    timely()
    weights = csr_matrix((a["data"], a["indices"], a["indptr"]), shape=(N, N))
    ledger["operator_constructor_requested"] += 1
    journal(ledger)
    operator = operator_type(weights, m["positions"], m["edge_parameters"], GROUPS, device)
    ledger["operator_constructor_completed"] += 1
    ledger["model_constructor_requested"] += 1
    journal(ledger)
    model = model_type(operator, p["tau"], p["coefficient"], p["threshold"], p["cap"],
                       p["gain"], p["inverse"], DT)
    ledger["model_constructor_completed"] += 1
    ledger["constructors_completed"] += 1
    journal(ledger)
    t = lambda value: torch.as_tensor(value.copy(), device=device)
    theta = torch.ones(GROUPS, dtype=torch.float64, device=device)
    initial, first, final = (data["states"][key] for key in STATES)
    final = final["network"]
    r, z = t(initial["rate"]), t(initial["adaptation"])
    drive, mask = t(first["drive"]), t(first["output_mask"])
    prefix_efficacy = model.prepare_efficacy(theta) if hasattr(model, "prepare_efficacy") else theta
    with torch.no_grad():
        for _ in range(20):
            timely()
            ledger["forward_requested"] += 1
            journal(ledger)
            r, z = model.step(r, z, drive, prefix_efficacy, mask)
            ledger["forward_completed"] += 1
            journal(ledger)
        prefix_rate, prefix_adaptation = r.cpu().numpy().copy(), z.cpu().numpy().copy()
        partial.update(first_rate=prefix_rate, first_adaptation=prefix_adaptation)
        ledger["forward_device_confirmed"] = 20
        journal(ledger)
        checkpoint(partial)
    require(exact(prefix_rate, first["rate"]) and exact(prefix_adaptation, first["adaptation"])
            and exact(first["drive"], drive.cpu().numpy())
            and exact(first["output_mask"], mask.cpu().numpy()), "First20 native endpoint mismatch")
    ledger["neutral_exact"] = True
    journal(ledger)
    theta = torch.ones(GROUPS, dtype=torch.float64, device=device, requires_grad=True)
    nominal_efficacy = model.prepare_efficacy(theta) if hasattr(model, "prepare_efficacy") else theta
    base_r, base_z = t(final["rate"]), t(final["adaptation"])
    drive, mask = t(final["drive"]), t(final["output_mask"])
    r, z = base_r, base_z
    for _ in range(4):
        timely()
        ledger["forward_requested"] += 1
        journal(ledger)
        r, z = model.step(r, z, drive, nominal_efficacy, mask)
        ledger["forward_completed"] += 1
        journal(ledger)
    ids = torch.as_tensor(m["population"].copy(), device=device)
    nominal = r[ids].to(torch.float64).mean()
    nominal_rate = r.detach().cpu().numpy().copy()
    nominal_adaptation = z.detach().cpu().numpy().copy()
    nominal_value = float(nominal.detach().item())
    partial.update(nominal_rate=nominal_rate, nominal_adaptation=nominal_adaptation,
                   nominal_loss64=np.asarray(nominal_value, np.float64))
    ledger["forward_device_confirmed"] = 24
    journal(ledger)
    checkpoint(partial)
    ledger["backward_requested"] = 4
    journal(ledger)
    timely()
    nominal.backward()
    ledger["backward_completed"] = 4
    journal(ledger)
    gradient = theta.grad.detach().cpu().numpy().copy()
    ledger["backward_device_confirmed"] = 4
    journal(ledger)
    partial["gradient"] = gradient
    checkpoint(partial)
    require(gradient.shape == (GROUPS,) and np.isfinite(gradient).all(), "Finite whole-state gradient required")
    direction = np.sign(gradient).astype(np.float32)
    partial["direction"] = direction
    losses = []
    perturbed = []
    with torch.no_grad():
        for sign in (1., -1.):
            effective = (np.ones(GROUPS, np.float32) + np.float32(sign * EPS) * direction).astype(np.float32)
            changed = t(effective)
            changed_efficacy = model.prepare_efficacy(changed) if hasattr(model, "prepare_efficacy") else changed
            rr, zz = base_r, base_z
            for _ in range(4):
                timely()
                ledger["forward_requested"] += 1
                journal(ledger)
                rr, zz = model.step(rr, zz, drive, changed_efficacy, mask)
                ledger["forward_completed"] += 1
                journal(ledger)
            end_rate, end_adaptation = rr.detach().cpu().numpy().copy(), zz.detach().cpu().numpy().copy()
            losses.append(float(rr[ids].to(torch.float64).mean().item()))
            require(np.isfinite(losses[-1]), "Finite perturbed loss required")
            ledger["forward_device_confirmed"] = 28 if sign > 0 else 32
            journal(ledger)
            perturbed.append(effective)
            prefix = "plus" if sign > 0 else "minus"
            partial[prefix + "_rate"] = end_rate
            partial[prefix + "_adaptation"] = end_adaptation
            partial["theta_plus32" if sign > 0 else "theta_minus32"] = effective
            partial["plus_loss64" if sign > 0 else "minus_loss64"] = np.asarray(losses[-1], np.float64)
            checkpoint(partial)
    predicted = float(np.dot(gradient, perturbed[0].astype(np.float64) - perturbed[1].astype(np.float64)))
    require(np.isfinite(predicted), "Finite effective prediction required")
    result = screen(losses[0] - losses[1], predicted)
    result.update(nominal_loss=nominal_value, plus_loss=losses[0],
                  minus_loss=losses[1], neutral_exact=True)
    raw = {"first_rate": prefix_rate, "first_adaptation": prefix_adaptation,
           "gradient": gradient, "direction": direction, "theta_plus32": perturbed[0],
           "theta_minus32": perturbed[1], "losses64": np.asarray([nominal_value, *losses]),
           "nominal_rate": nominal_rate, "nominal_adaptation": nominal_adaptation}
    partial.update(raw)
    return result, partial.copy()


def publish(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def publish_failure(writer, path: Path, value: dict, primary: BaseException) -> None:
    try:
        writer(path, value)
    except Exception as second:  # noqa: BLE001 - keep original publication failure
        primary.add_note("Failure journal write failed: " + type(second).__name__ + ": " + str(second)[:500])


def usage() -> dict:
    memory = psutil.Process().memory_info()
    report = {"host_rss_bytes": memory.rss,
              "host_peak_bytes": getattr(memory, "peak_wset", memory.rss)}
    if torch.cuda.is_available():
        report["gpu_reserved_bytes"] = torch.cuda.memory_reserved()
        report["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    else:
        report["gpu_reserved_bytes"] = 0
        report["gpu_peak_reserved_bytes"] = 0
    return report


def run(freeze_path: Path, output: Path, *, root: Path = ROOT, loader=load_inputs,
        calculator=compute, clock=time.perf_counter, saver=np.savez, writer=publish,
        meter=usage, gatekeeper=gate) -> dict:
    began = clock()
    freeze, qualification = gatekeeper(freeze_path, output, root=root)
    if shutil.disk_usage(output.parent).free < 2 << 30:
        raise RuntimeError("E315 requires at least 2 GiB free output disk")
    output.mkdir(parents=True, exist_ok=False)
    ledger = {"forward_requested": 0, "forward_completed": 0, "backward_requested": 0,
              "backward_completed": 0, "forward_device_confirmed": 0,
              "backward_device_confirmed": 0, "constructors_completed": 0,
              "operator_constructor_requested": 0, "operator_constructor_completed": 0,
              "model_constructor_requested": 0, "model_constructor_completed": 0,
              "neutral_exact": False,
              "original_attempts": 1, "body_constructors": 0, "physics_steps": 0,
              "fit_calls": 0, "retry_count": 0, "science_budget_s": SCIENCE_S,
              "finalization_budget_s": FINALIZATION_S, "supervisor_budget_s": 60.}
    source = {"schema": "flylab.m3.e315.source-manifest.v1", "freeze_sha256": digest(freeze_path),
              "source_sha256": freeze["source_sha256"], "inputs": freeze["inputs"],
              "qualification": freeze["qualification"],
              "qualification_evidence": qualification["evidence_sha256"],
              "producer_reviews": freeze.get("producer_reviews", [])}
    raw = {}
    first_error = None
    science_start = clock()
    def advance(current):
        current["elapsed_s"] = clock() - began
        pending = output / "advance-ledger.pending.json"
        writer(pending, current)
        pending.replace(output / "advance-ledger.json")

    def partial_checkpoint(current):
        pending = output / "partial-raw.pending.npz"
        saver(pending, **current)
        pending.replace(output / "partial-raw.npz")

    try:
        advance(ledger)
        baseline_usage = meter()
        writer(output / "source-manifest.json", source)
        data = loader(freeze, root=root)
        result, complete_raw = calculator(data, ledger, partial=raw, deadline=science_start + SCIENCE_S,
                                          journal=advance, checkpoint=partial_checkpoint)
        raw.update(complete_raw)
        require(ledger["neutral_exact"] is True and ledger["forward_device_confirmed"] == 32
                and ledger["backward_device_confirmed"] == 4
                and ledger["forward_requested"] == ledger["forward_completed"] == 32
                and ledger["backward_requested"] == ledger["backward_completed"] == 4,
                "Complete device-confirmed bounded E315 calculation required")
        ending_usage = meter()
        ledger["resources"] = ending_usage
        ledger["additional_gpu_peak_reserved_bytes"] = max(
            0, ending_usage["gpu_peak_reserved_bytes"] - baseline_usage["gpu_reserved_bytes"])
        if ending_usage["host_peak_bytes"] > 6 << 30 or ledger["additional_gpu_peak_reserved_bytes"] > 2 << 30:
            raise MemoryError("E315 host/GPU resource limit exceeded")
        if clock() - science_start > SCIENCE_S:
            raise TimeoutError("E315 internal science deadline exceeded")
        result.update(schema="flylab.m3.e315.native-adjoint-probe.v1", run_valid=True,
                      candidate_adopted=False, biological_gait_proven=False)
    except Exception as error:  # noqa: BLE001 - preserve arbitrary native failure
        first_error = {"type": type(error).__name__, "message": str(error)[:500],
                       "traceback": traceback.format_exc(limit=8)[-3000:]}
        result = {"schema": "flylab.m3.e315.native-adjoint-probe.v1", "run_valid": False,
                  "classification": "not_tested", "candidate_adopted": False,
                  "biological_gait_proven": False, "first_error": first_error}
    ledger["science_wall_s"] = clock() - science_start
    finalization_start = clock()
    ledger["wall_s"] = clock() - began
    ledger["forward_host_returned"] = ledger["forward_completed"]
    ledger["backward_host_returned"] = ledger["backward_completed"]
    ledger["science_forward_ticks"] = ledger["forward_device_confirmed"]
    ledger["science_backward_ticks"] = ledger["backward_device_confirmed"]
    ledger["forward_unconfirmed_host_returns"] = max(0, ledger["forward_completed"] - ledger["forward_device_confirmed"])
    ledger["backward_unconfirmed_host_returns"] = max(0, ledger["backward_completed"] - ledger["backward_device_confirmed"])
    storage_failures = []
    if raw:
        try:
            saver(output / "raw.npz", **raw)
        except Exception as error:  # noqa: BLE001 - record failed artifact write
            result["run_valid"] = False
            result["classification"] = "not_tested"
            storage_failures.append({"file": "raw.npz", "type": type(error).__name__,
                                     "message": str(error)[:500]})
            first_error = first_error or storage_failures[0]
    result["first_error"] = first_error
    result["storage_failures"] = storage_failures
    ledger["finalization_wall_s"] = clock() - finalization_start
    ledger["wall_s"] = clock() - began
    result["cost"] = ledger.copy()
    failures = []
    for name, value in (("ledger.json", ledger), ("result.json", result)):
        try:
            writer(output / name, value)
        except Exception as error:  # noqa: BLE001 - attempt remaining journals
            failures.append({"file": name, "type": type(error).__name__, "message": str(error)[:500]})
    if failures:
        result["run_valid"] = False
        result["classification"] = "not_tested"
        result["publication_failures"] = failures
        result["first_error"] = result["first_error"] or failures[0]
        primary = RuntimeError("E315 publication failure: " + failures[0]["message"])
        publish_failure(writer, output / "publication-failure.json", result, primary)
        raise primary
    try:
        files = {path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
                 for path in output.iterdir() if path.is_file()}
        if sum(entry["bytes"] for entry in files.values()) > 200 << 20:
            raise RuntimeError("E315 output limit exceeded")
        writer(output / "output-manifest.json", {"schema": "flylab.m3.e315.output-manifest.v1", "files": files})
    except Exception as error:
        result["run_valid"] = False
        result["classification"] = "not_tested"
        result["publication_failures"] = [{"file": "output-manifest.json", "type": type(error).__name__,
                                             "message": str(error)[:500]}]
        result["first_error"] = result["first_error"] or result["publication_failures"][0]
        publish_failure(writer, output / "publication-failure.json", result, error)
        try:
            writer(output / "result.json", result)
        except Exception as second:  # noqa: BLE001 - preserve primary publication error
            error.add_note("Invalid result rewrite failed: " + type(second).__name__ + ": " + str(second)[:500])
        raise
    if clock() - finalization_start > FINALIZATION_S:
        result["run_valid"] = False
        result["classification"] = "not_tested"
        result["first_error"] = result["first_error"] or {"type": "TimeoutError",
                                                       "message": "E315 internal finalization deadline exceeded"}
        ledger["finalization_wall_s"] = clock() - finalization_start
        ledger["wall_s"] = clock() - began
        result["cost"] = ledger.copy()
        writer(output / "ledger.json", ledger)
        writer(output / "result.json", result)
        writer(output / "publication-failure.json", result)
        files = {path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
                 for path in output.iterdir() if path.is_file() and path.name != "output-manifest.json"}
        writer(output / "output-manifest.json", {"schema": "flylab.m3.e315.output-manifest.v1", "files": files})
        raise TimeoutError("E315 internal finalization deadline exceeded")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.freeze, args.output)
    print(json.dumps(result, allow_nan=False), flush=True)
    if not result["run_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
