#!/usr/bin/env python3
"""Reproduce tendon routing reversals without applying any muscle commands."""
import argparse
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from flylab.c.integrity import file_hash, write_json
from flylab.c.muscles import MuscleRig
from flylab.c.muscle_calibration import sweep


def run(out, samples=401):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    write_json(out/'spec.json', dict(samples=samples, angle_domain='full upstream joint range',
        tolerances=dict(analytic=1e-9, reference=1e-9, finite_difference=1e-8),
        geometry_modified=False, physics_simulated=False))
    rig = MuscleRig()
    result = sweep(rig, samples)
    import flygym
    mocap = Path(flygym.__file__).parent.parent/'flygym_demo/muscle_imitation/assets/mocap/qpos/0002.npy'
    if mocap.is_file():
        q = np.load(mocap, allow_pickle=False)
        if q.ndim != 2 or q.shape[1] != 7: raise ValueError('Unexpected upstream clip shape')
        shutil.copyfile(mocap, out/'mocap_0002_qpos.npy')
        lo, hi = map(float, (q[:,6].min(), q[:,6].max()))
        result['reference_clip'] = dict(sha256=file_hash(mocap), samples=len(q),
            tibia_angle_range_rad=[lo,hi], interval_s=.002,
            crossings_inside_clip_range=[r for r in result['zero_crossings'] if lo <= r['zero_angle_rad'] <= hi],
            interpretation='One provided motion clip; not the complete physiological range')
    else: result['reference_clip'] = dict(status='UNAVAILABLE')
    write_json(out/'model.json', rig.metadata)
    shutil.copyfile(rig.source, out/'upstream.xml'); (out/'fixture.xml').write_text(rig.xml)
    write_json(out/'report.json', result)
    print(result['status'], result['errors'], result['zero_crossings'])
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True); p.add_argument('--samples', type=int, default=401)
    args = p.parse_args(); result = run(args.out, args.samples)
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
