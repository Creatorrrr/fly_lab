import os
import unittest
import numpy as np


@unittest.skipUnless(os.environ.get('FLYLAB_NATIVE_TESTS')=='1','Official Retina reference required')
class RetinaComputeParity(unittest.TestCase):
    def test_cached_optics_and_cuda_match_official_pixels(self):
        import numba
        from flygym.vision.retina import Retina
        from flylab.retina_compute import RetinaCompute
        original_threads=numba.get_num_threads();numba.set_num_threads(1)
        try:
            r=Retina();cpu=RetinaCompute(r,'cpu')
            gpu=RetinaCompute(r,'cuda') if os.environ.get('FLYLAB_CUDA_BATCH_TESTS')=='1' else None
            random=np.random.default_rng(42)
            for mode in ('black','white','random','stripes'):
                a=random.integers(0,256,(2,r.nrows,r.ncols,3),dtype=np.uint8)
                if mode=='black':a[:]=0
                elif mode=='white':a[:]=255
                elif mode=='stripes':a[:,::2]=0
                pixels=np.asarray([r.correct_fisheye(im) for im in a])
                expected=np.asarray([r.raw_image_to_hex_pxls(im) for im in pixels],np.float32)
                actual,values=cpu.process(a)
                np.testing.assert_array_equal(actual,pixels);np.testing.assert_array_equal(values,expected)
                if gpu:
                    actual,values=gpu.process(a)
                    np.testing.assert_array_equal(actual,pixels);np.testing.assert_array_equal(values,expected)
                    actual,values=gpu.process(a,include_rgb=False)
                    self.assertIsNone(actual);np.testing.assert_array_equal(values,expected)
        finally:numba.set_num_threads(original_threads)


if __name__=='__main__':unittest.main()
