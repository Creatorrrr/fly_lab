"""Compare the official retinal transform with cached CPU and ordered CUDA."""
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from flylab.retina_compute import RetinaCompute
from flylab.c.integrity import write_json


def main():
    from flygym.vision.retina import Retina
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    r=Retina();raw=np.random.default_rng(42).integers(0,256,(2,r.nrows,r.ncols,3),dtype=np.uint8)
    reference=lambda:np.asarray([r.raw_image_to_hex_pxls(r.correct_fisheye(im)) for im in raw],np.float32)
    cpu,gpu=RetinaCompute(r,'cpu'),RetinaCompute(r,'cuda')
    paths=dict(official_cpu=reference,cpu_fallback=lambda:cpu.process(raw,include_rgb=False)[1],cuda=lambda:gpu.process(raw,include_rgb=False)[1])
    expected=reference();timings={}
    for name,fn in paths.items():
        np.testing.assert_array_equal(expected,fn());samples=[]
        for _ in range(5):
            start=time.perf_counter()
            for _ in range(30):fn()
            samples.append((time.perf_counter()-start)/30)
        timings[name]=float(np.median(samples))
    result=dict(exact=True,seconds_per_stereo_pair=timings,speedup=timings['official_cpu']/timings['cuda'],
        scope='warmed optical transform + host transfers; native raster rendering and neural integration excluded')
    write_json(a.out/'report.json',result);print(result)


if __name__=='__main__':main()
