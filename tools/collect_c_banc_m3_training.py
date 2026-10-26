"""Collect full BANC responses to a physically verified projected teacher.

The saved boundary currents are for a reduced training feasibility diagnostic,
not an autonomous controller. Final evaluation must remove the teacher body,
stored currents, stored sensory data, and all teacher-state resets.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.banc_walking import BancWalkingSession
from flylab.c.graph import GraphStore
from flylab.c.research_annotations import rate_weights
from flylab.c.storage import StateStore
from flylab.c.target_sense import TargetObservation
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.verify_c_banc_observation import exact


def choose_core(graph, weights, adapter):
    eligible = np.array([
        n.get("super_class") == "ventral_nerve_cord_intrinsic"
        and bool(n.get("cell_type")) for n in graph.nodes
    ])
    signs = np.zeros(graph.n)
    for j, w in zip(graph.indices, weights.data):
        signs[j] = np.sign(w)
    selected, selections = set(adapter.motor_ids.tolist()), []
    for leg, groups in enumerate(adapter.muscles):
        for name, motor, _ in groups:
            strength = np.asarray(abs(weights[motor]).sum(axis=0)).ravel()
            for sign in (1, -1):
                candidates = np.flatnonzero(eligible & (signs == sign) & (strength > 0))
                # Stable graph index breaks count ties; no activity/label selection.
                order = candidates[np.argsort(-strength[candidates], kind="stable")[:2]]
                selected.update(order.tolist())
                selections.append({"leg_index": leg, "muscle": name, "sign": sign,
                    "neurons": [graph.nodes[i]["id"] for i in order],
                    "strength": strength[order].tolist()})
    return np.array(sorted(selected)), selections


def run(source, projected, identification, graph_path, out, boundary_dt_ms=5, reference=None):
    out.mkdir(parents=True, exist_ok=False)
    encoding_path = Path("verification/banc-natural-gait-m3-20260919/e05-afferent/candidate-source/flylab/c/afferent_rate.py")
    spec = {
        "experiment": "M3-E07" if boundary_dt_ms == 5 else "M3-E07b",
        "hypothesis": "Corrected afferent transmission plus existing anatomical premotor/motor recurrence provides an identifiable training substrate for the projected gait",
        "competing_explanations": ["Sensor or boundary information remains inadequate", "Selected recurrent parameterization is inadequate", "Optimization/finite training data fail"],
        "source": "E06b original-rest actual physical command trace, 2.2s",
        "input": "Normal 5ms receptor path and full 0.25ms adaptive BANC, E05 isolated-cell afferent rate18 correction, held public encoder bearing0 distance20; teacher drives only the body",
        "subset_selection": "All 391 anatomical leg motors plus top2 excitatory/top2 inhibitory typed VNC intrinsic direct inputs per muscle, chosen from source synaptic strength before any activity/target fitting",
        "boundary": "W0[core,outside] @ recorded outside rates, stored directly. Never Jfull minus learned internal current. Only internal selected existing edges may later be trained; boundary edges remain unchanged.",
        "state_contract": "Student neural rate/adaptation and muscle activation must evolve continuously; no teacher-state reset per frame/period; labels/time never enter student forward",
        "windows_s": {"train": [.5, 1.2], "validation": [1.5, 2.2]},
        "validity": "Every replay qpos/qvel/ctrl byte exact; full roster/edge count retained; actual CUDA; finite complete 5ms data and same clock",
        "decision": "Collection is diagnostic only. Learned parameter candidates require teacher-free full-network physics before any improvement claim; all F1-F8 unchanged",
        "limits": "2.2model sec /300wall sec, one CUDA job; no fit during collection",
        "tool_sha256": sha256(__file__),
        "trace_sha256": sha256(projected / "trace.json"),
        "projected_signals_sha256": sha256(projected / "signals.npz"),
        "encoding_sha256": sha256(encoding_path),
        "boundary_dt_ms": boundary_dt_ms,
        "reference_sha256": sha256(reference / "signals.npz") if reference else None,
    }
    write_json(out / "spec.json", spec)
    (out / "runner.py").write_bytes(Path(__file__).read_bytes())
    module_spec = importlib.util.spec_from_file_location("flylab.c._m3_archived_afferent", encoding_path)
    encoding_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(encoding_module)
    saved = StateStore.load(source / "start")
    graph = GraphStore.load(graph_path)
    session = BancWalkingSession.from_checkpoint(graph, saved)
    began = time.perf_counter()
    try:
        if session.network.device != "cuda":
            raise ValueError("Actual CUDA required")
        body, adapter, network = session.body, session.adapter, session.network
        if boundary_dt_ms == 1:
            # Observation granularity alone changes; all original RK steps remain.
            initial_neural = network.snapshot()
            network.interval_steps = 4
            network.graph = None
            with network.torch.inference_mode():
                network._interval()
                network._interval()
                network.torch.cuda.synchronize()
                network.graph = network.torch.cuda.CUDAGraph()
                with network.torch.cuda.graph(network.graph):
                    network._interval()
            network.restore(initial_neural)
        exact(saved["body"], body.snapshot(), "E07 initial body")
        weights, _ = rate_weights(graph)
        weights = session.navigation.encoder.calibrate_weights(weights, adapter.motor_ids)
        core, selections = choose_core(graph, weights, adapter)
        global_to_local = np.full(graph.n, -1)
        global_to_local[core] = np.arange(len(core))
        selected_edges = np.concatenate([
            np.arange(graph.indptr[i], graph.indptr[i+1])[
                global_to_local[graph.indices[graph.indptr[i]:graph.indptr[i+1]]] >= 0]
            for i in core])
        core_matrix = weights[core][:, core].toarray()
        boundary = weights[core].copy()
        boundary.data[global_to_local[boundary.indices] >= 0] = 0
        boundary.eliminate_zeros()
        original = np.load(identification / "signals.npz")
        observed = np.union1d(original["presyn_ids"], core)
        afferents = np.unique(adapter._sensory_ids)
        encoding = encoding_module.AfferentRateEncoding(
            afferents,
            cap=network.cap.detach().cpu().numpy()[afferents],
            slope=network.a.detach().cpu().numpy()[afferents],
            threshold=network.threshold.detach().cpu().numpy()[afferents],
            adaptation_gain=network.adaptation_gain,
            feature_gain=adapter.sensory_gain, target_rate=18,
        )
        tonic = session.navigation.encoder.drive(TargetObservation(0,20,True))
        if np.count_nonzero(tonic) != 22 or not np.all(tonic[session.descending] == 5000) or not np.all(tonic[session.navigation.encoder.ports] == -5000):
            raise AssertionError("Fixed forward input mismatch")
        teacher = json.loads((projected / "trace.json").read_text())["samples"][1:441]
        projection = np.load(projected / "signals.npz")
        write_json(out / "topology.json", {
            "core_nodes": [graph.nodes[i] for i in core], "selections": selections,
            "graph_hash": graph.hash, "neurons": graph.n, "edges": len(graph.indices),
            "core_neurons": len(core), "internal_edges_including_zero": len(selected_edges),
            "nonzero_internal_edges": int(np.count_nonzero(core_matrix)),
            "boundary_edges_nonzero": boundary.nnz,
            "encoding_hash": encoding.identity, "all_source_nodes_preserved": True,
        })
        rr, zz, outside = [], [], []

        def capture():
            full = network.rate.detach().cpu().numpy().copy()
            rr.append(full[core].copy())
            zz.append(network.adaptation.detach().cpu().numpy()[core].copy())
            outside.append(boundary @ full)
            return full[observed].copy()

        all_pre = [capture()]
        drive_rows, sensory = [], []
        for tick, row in enumerate(teacher, 1):
            if time.perf_counter()-began > 300:
                raise TimeoutError("E07 wall limit")
            drive = encoding(adapter.encode()) + tonic
            sensory.append(adapter.last_features.copy())
            for _ in range(5 // boundary_dt_ms):
                drive_rows.append(drive[core].copy())
                network.advance(drive, boundary_dt_ms * 4)
                observed_rates = capture()
            all_pre.append(observed_rates)
            body.step_joint_targets(np.asarray(row["joint_targets_rad"]),
                np.asarray(row["adhesion"], bool), tendon_inputs=row["tendon_inputs"])
            for name in ("qpos", "qvel", "ctrl"):
                if getattr(body.d, name).tobytes() != np.asarray(row[name]).tobytes():
                    raise AssertionError(f"E07 teacher physics mismatch at {tick} {name}")
            session.control_tick = tick
            if tick % 100 == 0:
                print(json.dumps({"control_tick": tick, "wall_s": time.perf_counter()-began}), flush=True)
        session._clocks()
        data = {"time_s": np.arange(len(rr))*boundary_dt_ms/1000, "core": core,
            "motor_ids": adapter.motor_ids, "observed_ids": observed,
            "rate": np.asarray(rr), "adaptation": np.asarray(zz),
            "outside_current": np.asarray(outside), "drive": np.asarray(drive_rows),
            "observed_rates": np.asarray(all_pre), "sensory": np.asarray(sensory),
            "internal_weights": core_matrix, "selected_edge_indices": selected_edges,
            "rest": projection["rest"], "moment": projection["moment"],
            "target": projection["command"], "teacher_muscle_rates": projection["rates"],
            "source_parameters": np.array([.02,1.,7.5,200.,8.,.3]),
            "joints": adapter.joints}
        if reference:
            prior = np.load(reference / "signals.npz")
            for name in ("rate", "adaptation", "outside_current"):
                exact(prior[name], data[name][::5], "E07b observed " + name)
            for name in ("observed_rates", "sensory", "target", "core", "internal_weights"):
                exact(prior[name], data[name], "E07b source " + name)
        if any(not np.isfinite(v).all() for v in data.values()):
            raise ValueError("Nonfinite training data")
        np.savez_compressed(out / "signals.npz", **data)
        StateStore.save(out / "final-diagnostic", {
            "schema": "flylab.m3-teacher-collection.v1",
            "not_a_walking_session_checkpoint": True, "spec": spec,
            "body": body.snapshot(), "network": network.snapshot(),
            "adapter": adapter.snapshot(), "control_tick": session.control_tick,
            "encoding_hash": encoding.identity,
        })
        result = {"experiment": spec["experiment"], "run_status": "completed", "validity": "valid",
            "hypothesis_verdict": "not_yet_tested", "adoption": "not_applicable", "milestone_outcome": "pending",
            "model_s": session.control_tick*.005, "wall_s": time.perf_counter()-began,
            "device": network.device, "source_neural_model_hash": network.identity,
            "graph_hash": graph.hash, "encoding_hash": encoding.identity,
            "body_hash": body.model_hash, "exact_physical_steps": session.control_tick,
            "neurons": graph.n, "edges": len(graph.indices), "core_neurons": len(core),
            "internal_edges": len(selected_edges), "signal_sha256": sha256(out / "signals.npz")}
        result["boundary_dt_ms"] = boundary_dt_ms
        result["reference_control_neural_states_bit_exact"] = bool(reference)
        write_json(out / "result.json", result)
        print(json.dumps(result),flush=True)
    finally:
        session.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--projected",type=Path,required=True)
    p.add_argument("--identification",type=Path,required=True)
    p.add_argument("--graph",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--boundary-dt-ms",type=int,choices=(1,5),default=5)
    p.add_argument("--exact-reference",type=Path)
    a=p.parse_args()
    run(a.source,a.projected,a.identification,a.graph,a.out,a.boundary_dt_ms,a.exact_reference)
