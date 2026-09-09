"""Pure helpers: no physics dependency. Coordinates at the UI boundary: mm/y-up."""
from __future__ import annotations
import copy, math
import numpy as np

TAU = math.tau
S = np.array([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]]) # right-handed: (x,y,z) -> (x,z,-y)
def to_ui(v): return (S @ np.asarray(v, dtype=float)).tolist()
def to_physics(v): return S.T @ np.asarray(v, dtype=float)
def wrap(x): return math.atan2(math.sin(x), math.cos(x))
def clamp(x,a=0.,b=1.): return max(a,min(b,float(x)))
def clone(x): return copy.deepcopy(x)
def number(v, name='number', lo=-1e12, hi=1e12):
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not lo<=v<=hi:
        raise ValueError(f'{name}: 유효한 수치 범위는 {lo}..{hi}입니다.')
    return float(v)
def integer(v,name='integer',lo=0,hi=2**32-1):
    if type(v) is not int or not lo<=v<=hi: raise ValueError(f'{name}: 정수 {lo}..{hi} 필요')
    return v

def safe_tree(x, depth=0):
    if depth>28: raise ValueError('JSON nesting limit')
    if x is None or isinstance(x,(str,bool)): return
    if isinstance(x,(int,float)): number(x); return
    if isinstance(x,list):
        if len(x)>200000: raise ValueError('Array limit')
        for v in x: safe_tree(v,depth+1)
    elif isinstance(x,dict):
        if len(x)>20000: raise ValueError('Object limit')
        for k,v in x.items():
            if not isinstance(k,str): raise ValueError('String keys required')
            safe_tree(v,depth+1)
    else: raise ValueError('Not a JSON value')

class RNG:
    """Same xorshift32 as A, for numerical parity and deterministic interventions."""
    def __init__(self,seed=42): self.state=(int(seed)&0xffffffff) or 1
    def next(self):
        x=self.state; x^=(x<<13)&0xffffffff; x^=x>>17; x^=(x<<5)&0xffffffff
        self.state=x&0xffffffff; return self.state/4294967296
    def signed(self): return 2*self.next()-1
