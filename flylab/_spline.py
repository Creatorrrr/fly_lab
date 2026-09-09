"""Evaluate the pinned SciPy cubic coefficients without changing their order.

This is the dx=0 power-series evaluation used by scipy.interpolate._ppoly,
with six independent leg splines batched into one native call. No fast math.
"""
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def _evaluate(coefficients, offsets, intervals):
    out=np.empty((6,7),dtype=np.float64)
    for leg in range(6):
        s=offsets[leg]
        interval=intervals[leg]
        for dof in range(7):
            value=0.;power=1.
            for k in range(4):
                value=value+coefficients[leg,3-k,interval,dof]*power
                if k<3:power*=s
            out[leg,dof]=value
    return out


class BatchedSplines:
    def __init__(self, steps, legs):
        splines=[steps._psi_funcs[leg] for leg in legs]
        self.x=np.array(splines[0].x,copy=True)
        if any(s.extrapolate!='periodic' or not np.array_equal(s.x,self.x) or s.c.shape!=(4,len(self.x)-1,7) for s in splines):
            raise ValueError('Batched evaluator requires matching periodic cubic leg splines')
        self.coefficients=np.array([s.c for s in splines],order='C')
        self.evaluate(np.zeros(6))  # Compile before simulation clocks/timing start.

    def evaluate(self, phases):
        wrapped=self.x[0]+(phases-self.x[0])%(self.x[-1]-self.x[0])
        intervals=np.minimum(np.searchsorted(self.x,wrapped,side='right')-1,len(self.x)-2)
        return _evaluate(self.coefficients,wrapped-self.x[intervals],intervals)
