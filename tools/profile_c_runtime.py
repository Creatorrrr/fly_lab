#!/usr/bin/env python3
"""Profile a checkpoint continuation without changing the live experiment."""
from pathlib import Path
import argparse
import cProfile
import hashlib
import json
import pstats
import sys
import time


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=root)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=.2)
    args = p.parse_args()
    if not 0 < args.seconds <= 5 or abs(args.seconds/.005-round(args.seconds/.005)) > 1e-8:
        p.error('Use 5ms multiples up to five model seconds')
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source_root.resolve()))
    import numpy as np
    from flylab.c.graph import GraphStore
    from flylab.c.ports import PortBindings
    from flylab.c.engine import CEngine
    from flylab.c.storage import StateStore
    graph = GraphStore.load(root/'data/fafb783/bundle')
    binding = PortBindings(graph, json.loads((root/'data/fafb783/bindings.json').read_text()))
    state = StateStore.load(args.checkpoint)
    engine = CEngine.from_checkpoint(graph, binding, state)
    try:
        body = engine.body
        body.mj.mj_saveModel(body.m, str(args.out/'body.mjb'), None)
        np.savez(args.out/'body_state.npz', **{name: np.array(getattr(body.d, name)) for name in
                    ('qpos', 'qvel', 'act', 'ctrl', 'qacc_warmstart', 'xfrc_applied', 'time', 'mocap_pos', 'mocap_quat', 'userdata')})
        profile = cProfile.Profile()
        begun = time.perf_counter()
        profile.enable()
        for i in range(round(args.seconds/.005)):
            frame = engine.step(1)
        profile.disable()
        elapsed = time.perf_counter()-begun
        profile.dump_stats(str(args.out/'runtime.pstats'))
        with (args.out/'runtime.txt').open('w') as f:
            pstats.Stats(profile, stream=f).sort_stats('cumulative').print_stats(75)
            pstats.Stats(profile, stream=f).sort_stats('tottime').print_stats(40)
        StateStore.save(args.out/'final_checkpoint', engine.checkpoint())
        result = dict(status='PASS' if not engine.fault else 'FAIL',
                      source_root=str(args.source_root.resolve()), source_checkpoint=str(args.checkpoint),
                      source_manifest_sha256=hashlib.sha256((args.checkpoint/'manifest.json').read_bytes()).hexdigest(),
                      model_seconds=args.seconds, profiled_wall_seconds=elapsed,
                      frame=frame, body_model=dict(nq=body.m.nq,nv=body.m.nv,nu=body.m.nu,
                        ngeom=body.m.ngeom, integrator=int(body.m.opt.integrator),
                        solver=int(body.m.opt.solver), noslip_iterations=body.m.opt.noslip_iterations))
        (args.out/'report.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='frame'}, indent=2))
    finally:
        engine.close()


if __name__ == '__main__':
    main()
