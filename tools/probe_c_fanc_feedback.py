"""E25: explicit FANC core plus functional claw cells, retaining known feedback.

This conditional reference model does not replace BANC or certify physical
control. Missing connections and sensory intrinsic parameters are recorded.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.probe_c_banc_reference import checked_csv, sha, write

MATRIX_SHA = "260cffc79df112dd4be52b0d079fbd017e170c0008f666f6906f001fcd529140"
TABLE_SHA = "b1d4788f7c41ef654e7ad4554bd8a60f583df722079eab729d88e11d37bb53ee"
FECO_SHA = "478eece92cc641b03ab69f85904ca84da094c05fd7dfcc69070d9e9d268e1b4c"
DOWN_SHA = "c48bce9f989084b41c5ed19fdb89a58c4e8f1cb14fbb91a88784d5f0d5fab7f3"
UP_SHA = "a96962b443765b8f33d4277416a97b5507c3d6f98a97ebe4d9176a4ba411f1a7"
E22_MANIFEST_SHA = "46d8ce8519a72404133151460f736a6c7242ef05817631222728754c97a99709"
LABELS = ("q1", "q1.4", "q2", "q2.3")
PROFILE_KEYS = ("tau", "a", "threshold", "cap")
PARAMETERS = ((0.02, 0.002), (1, 0.1), (7.5, 0.6), (200, 10))


def extend_claw_graph(base, roots, signs, claw_roots, synapses):
    """Merge by synapse ID, count parallel contacts, and retain excluded edges."""
    roots = list(roots)
    claw_roots = list(claw_roots)
    if (
        len(set(roots)) != len(roots)
        or len(set(claw_roots)) != len(claw_roots)
        or set(roots) & set(claw_roots)
        or base.shape != (len(roots), len(roots))
        or len(signs) != len(roots)
        or not np.isfinite(base).all()
        or not np.array_equal(base, np.rint(base))
        or not set(signs) <= {-1, 0, 1}
        or np.any((np.sign(base) != np.asarray(signs)[:, None]) & (base != 0))
    ):
        raise ValueError("Invalid core or claw identity/sign alignment")
    full_roots = roots + claw_roots
    lookup = {root: i for i, root in enumerate(full_roots)}
    all_signs = np.concatenate((signs, np.ones(len(claw_roots))))
    full = np.zeros((len(full_roots), len(full_roots)), base.dtype)
    full[: len(roots), : len(roots)] = base
    sensory = set(claw_roots)
    seen = {}
    for row in synapses:
        if row["valid"] != "t":
            continue
        pair = (row["pre_pt_root_id"], row["post_pt_root_id"])
        identity = row["id"]
        if not isinstance(identity, str) or not identity:
            raise ValueError("Synapse ID must be an exact nonempty string")
        if identity in seen and seen[identity] != pair:
            raise ValueError("Conflicting endpoints for one synapse ID")
        seen[identity] = pair
    retained, excluded = [], []
    for identity, (pre, post) in seen.items():
        if pre not in sensory and post not in sensory:
            continue
        record = {"synapse_id": identity, "pre_root": pre, "post_root": post}
        if pre not in lookup or post not in lookup:
            excluded.append(record)
            continue
        sign = int(all_signs[lookup[pre]])
        if sign == 0:
            raise ValueError("Unassigned sign on a retained sensory feedback source")
        full[lookup[pre], lookup[post]] += sign
        retained.append({**record, "sign": sign})
    np.testing.assert_array_equal(full[: len(roots), : len(roots)], base)
    return full, full_roots, retained, excluded


def motor_groups(table):
    motors = [i for i, row in enumerate(table) if row["class"] == "motor neuron"]
    flex = [i for i in motors if table[i]["motor module"].startswith("tibia flex ")]
    extend = [i for i in motors if table[i]["motor module"] == "tibia extend"]
    if len(motors) != 69 or len(flex) != 15 or len(extend) != 2:
        raise ValueError("Unexpected source motor module membership")
    return {"flex": flex, "extend": extend, "all_motor": motors}


def prepare(args):
    import jax

    from tools.verify_c_rate_reference import load_author

    began = time.perf_counter()
    root = args.evidence_root
    matrix_path = root / "fanc-reference-01/W_BDN2toMN_20250107_corrected.npy"
    table_path = root / "fanc-sensory-source-01/wTable_BDN2toMN_20250107_corrected.csv"
    feco_path = root / "feco-source-01/feco_annotation_table.csv"
    down_path = root / "fanc-sensory-source-01/feco_downstream_connections.csv"
    up_path = root / "fanc-sensory-source-01/feco_upstream_connections.csv"
    if sha(matrix_path) != MATRIX_SHA:
        raise ValueError("Pinned FANC matrix changed")
    base = np.load(matrix_path, allow_pickle=False)
    table = checked_csv(table_path, TABLE_SHA)
    annotation = checked_csv(feco_path, FECO_SHA)
    down = checked_csv(down_path, DOWN_SHA)
    up = checked_csv(up_path, UP_SHA)
    roots = [row["pt_root_id"] for row in table]
    if (
        len(roots) != 803
        or [int(row["w_idx"]) for row in table] != list(range(803))
        or roots[0] != "648518346459693060"
        or table[0]["w_type"] != "DNg100"
        or np.count_nonzero(base) != 65606
    ):
        raise ValueError("Fixed reference core ordering changed")
    groups = motor_groups(table)
    claw = sorted(
        (
            row
            for row in annotation
            if row["valid"] == "t" and row["cell_type"] in ("claw_ext", "claw_flx")
        ),
        key=lambda row: row["pt_root_id"],
    )
    if len(claw) != 21 or sum(row["cell_type"] == "claw_ext" for row in claw) != 8:
        raise ValueError("Fixed functional claw population changed")
    full, ids, retained, excluded = extend_claw_graph(
        base,
        roots,
        [int(row["sign"]) for row in table],
        [row["pt_root_id"] for row in claw],
        down + up,
    )
    if len(retained) != 3252 or len(excluded) != 782:
        raise ValueError("Fixed sensory contact coverage changed")
    areas = np.array([float(row["surf_area_um2"] or "nan") for row in table])
    if (
        np.isfinite(areas).sum() != 801
        or np.isinf(areas).any()
        or (areas <= 0).any()
        or not np.isclose(np.nanmedian(areas), 4183.823234, rtol=0, atol=1e-9)
    ):
        raise ValueError("Fixed reference areas changed")
    full_areas = np.concatenate((areas, np.full(21, np.nan)))
    scope = load_author(root / "rate-reference")
    keys = jax.random.split(jax.random.PRNGKey(1), 5)
    arrays = {
        "signed_counts": full,
        "root_ids": np.asarray(ids),
        "areas_um2": full_areas,
    }
    for profile in ("mean", "seed1"):
        params = {}
        for index, (key, (mean, std)) in enumerate(zip(PROFILE_KEYS, PARAMETERS)):
            params[key] = (
                np.full((1, 803), mean, np.float32)
                if profile == "mean"
                else scope["sample_trunc_normal"](keys[index], mean, std, (1024, 803))[
                    :1
                ]
            )
        # Preserve the author's array backend through size scaling. Converting
        # the JAX draws to NumPy first changes float32 rounding of a/theta.
        params["a"], params["threshold"] = scope["set_sizes"](
            areas, params["a"], params["threshold"]
        )
        for (key, value), (mean, _) in zip(params.items(), PARAMETERS):
            # Every added cell has the explicit median-area assumption, hence
            # size ratio one. Append only after fixing all original 803 values.
            values = np.concatenate(
                (np.asarray(value[0], dtype=np.float32), np.full(21, mean, np.float32))
            )
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError("Invalid prepared parameters")
            arrays[profile + "_" + key] = values
    old_manifest = root / "descending-evidence-manifest.json"
    if sha(old_manifest) != E22_MANIFEST_SHA:
        raise ValueError("Prior input evidence manifest changed")
    hashes = {
        str(Path(row["path"])): row["sha256"]
        for row in json.loads(old_manifest.read_text())["evidence"]
    }
    input_sources = {}
    for label in LABELS:
        path = root / "banc-conditioned-01" / ("mean-" + label + ".npz")
        if sha(path) != hashes[str(path.relative_to(root))]:
            raise ValueError("Saved engineering input changed")
        with np.load(path, allow_pickle=False) as data:
            input_roots = data["root_ids"][data["input_indices"]].tolist()
            # The old BANC IDs identify saved signal columns, not FANC homologs.
            positive = data["input_drive"][:, input_roots.index("720575941439451477")]
            negative = data["input_drive"][:, input_roots.index("720575941478502209")]
            signals = np.stack(
                [
                    negative if row["cell_type"] == "claw_ext" else positive
                    for row in claw
                ],
                axis=1,
            )
        if signals.shape != (600, 21) or np.any(signals[200:]):
            raise ValueError("Input onset/removal alignment changed")
        arrays[label + "_drive"] = signals
        input_sources[label] = {"path": str(path), "sha256": sha(path)}
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out / "bundle.npz", **arrays)
    write(args.out / "table.json", table)
    write(args.out / "claw.json", claw)
    write(args.out / "retained-synapses.json", retained)
    write(args.out / "excluded-synapses.json", excluded)
    sources = (
        Path(__file__),
        Path("tools/verify_c_rate_reference.py"),
        Path("tools/probe_c_banc_reference.py"),
        Path("flylab/c/research_rate.py"),
        Path("GOAL_PLAN.md"),
    )
    (args.out / "source").mkdir()
    for path in sources:
        (args.out / "source" / path.name).write_bytes(path.read_bytes())
    write(
        args.out / "spec.json",
        {
            "test": "E25 FANC803 plus 21 functional claw cells",
            "preparation_revision": 2,
            "size_scaling_order": "original 803 author backend first, sensory means appended last",
            "base_neurons": 803,
            "sensory_neurons": 21,
            "total_neurons": 824,
            "missing_measured_areas": 23,
            "sensory_area_assumption": "original core median",
            "added_sensory_parameters": "mean tau=.02,a=1,threshold=7.5,cap=200 in both profiles",
            "base_seed1_draw_shape": [1024, 803],
            "base_seed1_selected_row": 0,
            "retained_sensory_synapses": len(retained),
            "excluded_sensory_synapses": len(excluded),
            "base_pairs": int(np.count_nonzero(base)),
            "full_pairs": int(np.count_nonzero(full)),
            "motor_groups": groups,
            "sensory_sources": input_sources,
            "source_files": {
                str(path): sha(path)
                for path in (matrix_path, table_path, feco_path, down_path, up_path)
            },
            "sources": {str(path): sha(path) for path in sources},
            "outputs": {
                path.name: sha(path) for path in args.out.iterdir() if path.is_file()
            },
            "wall_seconds": time.perf_counter() - began,
            "biological_validation": False,
            "complete_connectome": False,
            "exact_paper_run_reproduction": False,
            "controller_adoption": False,
        },
    )
    if time.perf_counter() - began > 60:
        raise TimeoutError("E25 parameter preparation wall limit")
    print(
        json.dumps(
            {
                "prepared_neurons": 824,
                "full_pairs": int(np.count_nonzero(full)),
                "bundle_sha256": sha(args.out / "bundle.npz"),
                "wall_seconds": time.perf_counter() - began,
            }
        ),
        flush=True,
    )


def run(args):
    from scipy.sparse import csr_matrix

    from flylab.c.research_rate import ResearchRateNetwork

    began = time.perf_counter()
    preparation = json.loads((args.prepared / "spec.json").read_text(encoding="utf8"))
    if preparation.get("preparation_revision") != 2:
        raise ValueError(
            "E25 preparation revision 2 required; preserve the original author parameter backend"
        )
    for name, expected in preparation["outputs"].items():
        if sha(args.prepared / name) != expected:
            raise ValueError("Prepared FANC bundle changed")
    for name, expected in preparation["source_files"].items():
        if sha(Path(name)) != expected:
            raise ValueError("FANC source changed after preparation")
    with np.load(args.prepared / "bundle.npz", allow_pickle=False) as data:
        bundle = {key: data[key].copy() for key in data.files}
    table = json.loads((args.prepared / "table.json").read_text(encoding="utf8"))
    roots = bundle["root_ids"]
    weights = bundle.pop("signed_counts")
    if weights.shape != (824, 824) or len(roots) != 824 or len(set(roots)) != 824:
        raise ValueError("Prepared graph dimensions or IDs changed")
    groups = motor_groups(table)
    if groups != preparation["motor_groups"]:
        raise ValueError("Motor module membership changed")
    sensory_indices = np.arange(803, 824, dtype=np.int64)
    tibia_indices = groups["flex"] + groups["extend"]
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "source").mkdir()
    sources = (
        Path(__file__),
        Path("tools/probe_c_banc_reference.py"),
        Path("flylab/c/research_rate.py"),
        Path("GOAL_PLAN.md"),
    )
    for path in sources:
        (args.out / "source" / path.name).write_bytes(path.read_bytes())
    specification = {
        "test": "E25 FANC functional claw reciprocal network diagnostic",
        "preparation_revision": preparation["preparation_revision"],
        "prepared_path": str(args.prepared),
        "prepared_spec_sha256": sha(args.prepared / "spec.json"),
        "bundle_sha256": sha(args.prepared / "bundle.npz"),
        "sources": {str(path): sha(path) for path in sources},
        "motor_groups": groups,
        "sensory_indices": sensory_indices.tolist(),
        "dn_root": str(roots[0]),
        "dn_index": 0,
        "dn_units": 150,
        "dn_start_ms": 20,
        "branch_start_ms": 1600,
        "all_inputs_off_ms": 1800,
        "end_ms": 2200,
        "neural_model_limit_s": 22.8,
        "wall_limit_s": 180,
        "missing_sensory_connection_count": 782,
        "missing_measured_area_count": 23,
        "existing_BANC_fast_feti_criterion": "preserved, not tested by this FANC module diagnostic",
        "physical_test_qualified": False,
        "controller_adoption": False,
    }
    write(args.out / "spec.json", specification)
    cases = []
    control_errors = {}
    simulated_ms = 0

    def integrate(network, name, profile, *, cut=False, initial=None, label=None):
        nonlocal simulated_ms
        start = 0 if initial is None else 1600
        count = 2200 - start
        network.reset()
        if initial is not None:
            state = network.snapshot()
            state["rate"] = initial.copy()
            state["tick"] = 16000
            state["drive"][0] = 150
            network.restore(state)
        if cut:
            network.set_muted(sensory_indices)
        rates = np.zeros((count + 1, network.n), np.float32)
        rates[0] = network.readout()
        if initial is not None:
            np.testing.assert_array_equal(rates[0], initial)
        targets = (
            np.concatenate(([0], sensory_indices)).astype(np.int64)
            if network.n == 824
            else np.array([0], np.int64)
        )
        inputs = np.zeros((count, len(targets)), np.float32)
        absolute_ms = start + np.arange(count)
        inputs[(absolute_ms >= 20) & (absolute_ms < 1800), 0] = 150
        if label is not None:
            if initial is None or network.n != 824:
                raise ValueError("Sensory inputs require the extended preserved state")
            inputs[:, 1:] = bundle[label + "_drive"]
        drive = np.zeros(network.n, np.float32)
        for tick in range(count):
            drive.fill(0)
            drive[targets] = inputs[tick]
            network.advance(drive)
            rates[tick + 1] = network.readout()
            simulated_ms += 1
            if simulated_ms > 22800 or time.perf_counter() - began > 180:
                raise TimeoutError("E25 cumulative model/wall limit")
        first_tail = 1751 - start
        last_tail = 1801 - start
        motor_tail = rates[first_tail:last_tail].astype(np.float64).mean(axis=0)
        final_tail = rates[-50:].astype(np.float64).mean(axis=0)
        valid = bool(
            np.isfinite(rates).all() and (rates >= 0).all() and network.tick == 22000
        )
        output = args.out / (name + ".npz")
        np.savez_compressed(
            output,
            rates=rates,
            root_ids=roots[: network.n],
            input_indices=targets,
            input_drive=inputs,
            time_s=(start + np.arange(count + 1)) * 0.001,
            neural_tick=network.tick,
            output_mask=network.output_mask.cpu().numpy(),
        )
        case = {
            "name": name,
            "profile": profile,
            "condition": label or "dn_only",
            "cut": cut,
            "neurons": network.n,
            "start_ms": start,
            "valid": valid,
            "model_hash": network.identity,
            "device": network.device,
            "neural_model_seconds": count * 0.001,
            "restored_state_max_error": 0.0 if initial is not None else None,
            "module_stimulus_tail": {
                key: float(motor_tail[ids].mean()) for key, ids in groups.items()
            },
            "module_final_tail": {
                key: float(final_tail[ids].mean()) for key, ids in groups.items()
            },
            "final_max_tibia_neuron_tail": float(final_tail[tibia_indices].max()),
            "final_active_neurons_ge_2": int((rates[-1] >= 2).sum()),
            "motor_stimulus_tail": [
                {
                    "root_id": str(roots[i]),
                    "source_type": table[i]["cell_type"],
                    "source_module": table[i]["motor module"],
                    "mean_rate_units": float(motor_tail[i]),
                }
                for i in groups["all_motor"]
            ],
            "trace_sha256": sha(output),
        }
        cases.append(case)
        write(args.out / "cases.json", cases)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in case.items()
                    if key != "motor_stimulus_tail"
                }
            ),
            flush=True,
        )
        return rates

    for profile in ("mean", "seed1"):
        parameters = [bundle[profile + "_" + key] for key in PROFILE_KEYS]
        core = ResearchRateNetwork(
            csr_matrix(weights[:803, :803].T.astype(np.float32) * np.float32(0.03)),
            *[value[:803] for value in parameters],
            device="cuda",
        )
        baseline = integrate(core, profile + "-base-dn", profile)
        del core
        gc.collect()
        extended = ResearchRateNetwork(
            csr_matrix(weights.T.astype(np.float32) * np.float32(0.03)),
            *parameters,
            device="cuda",
        )
        no_sensory = integrate(extended, profile + "-augmented-dn", profile)
        start_state = no_sensory[1600].copy()
        cut_control = integrate(
            extended, profile + "-augmented-dn-cut", profile, cut=True
        )
        error = float(np.abs(cut_control[:, :803] - baseline).max())
        control_errors[profile] = error
        if error > 0.001:
            raise ValueError("Extended cut network differs from the original core")
        for label in LABELS:
            for cut in (False, True):
                suffix = "-cut" if cut else "-full"
                integrate(
                    extended,
                    profile + "-" + label + suffix,
                    profile,
                    cut=cut,
                    initial=start_state,
                    label=label,
                )
        del extended
        gc.collect()
    qualification = {}
    for profile in ("mean", "seed1"):
        selected = {row["name"]: row for row in cases if row["profile"] == profile}
        one = selected[profile + "-q1-full"]["module_stimulus_tail"]
        two = selected[profile + "-q2-full"]["module_stimulus_tail"]
        cut_one = selected[profile + "-q1-cut"]["module_stimulus_tail"]
        cut_two = selected[profile + "-q2-cut"]["module_stimulus_tail"]
        directional = (
            one["flex"] >= 2
            and one["extend"] <= 0.2 * one["flex"]
            and two["extend"] >= 2
            and two["flex"] <= 0.2 * two["extend"]
        )
        contrasts = [
            one["flex"] - one["extend"] - cut_one["flex"] + cut_one["extend"],
            two["extend"] - two["flex"] - cut_two["extend"] + cut_two["flex"],
        ]
        causal = min(contrasts) >= 2
        decayed = all(
            selected[profile + "-" + label + "-full"]["final_max_tibia_neuron_tail"] < 2
            for label in LABELS
        )
        qualification[profile] = {
            "bidirectional_module_transfer": directional,
            "full_minus_cut_directional_contrast": contrasts,
            "sensory_causal_module_contrast": causal,
            "all_tibia_neurons_decay_below_2": decayed,
            "conditional_reference_hypothesis_supported": directional
            and causal
            and decayed,
            "BANC_or_physical_control_qualified": False,
        }
    report = {
        "status": "COMPLETE",
        "validity": "valid" if all(row["valid"] for row in cases) else "invalid",
        "cases": cases,
        "qualification": qualification,
        "base_vs_extended_cut_max_errors": control_errors,
        "neural_model_seconds": simulated_ms * 0.001,
        "physics_model_seconds": 0,
        "wall_seconds": time.perf_counter() - began,
        "BANC_fast_feti_criterion": "not_tested",
        "F3_F6": "fail",
        "controller_adoption": False,
        "full_goal_completed": False,
    }
    write(args.out / "report.json", report)
    print(
        json.dumps(
            {
                "qualification": qualification,
                "model_seconds": report["neural_model_seconds"],
                "wall_seconds": report["wall_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument(
        "--evidence-root", type=Path, default=Path("verification/body-control-20260911")
    )
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("out must not exist; preserve earlier evidence")
    if args.mode == "run" and args.prepared is None:
        parser.error("run requires --prepared")
    try:
        (prepare if args.mode == "prepare" else run)(args)
    except Exception as error:
        if args.out.is_dir() and not (args.out / "report.json").exists():
            write(
                args.out / "failure.json", {"status": "INVALID", "error": repr(error)}
            )
        raise
