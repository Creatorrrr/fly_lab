"""Explicit backend requests fail visibly; auto prefers working CUDA, then MPS."""
from functools import lru_cache


@lru_cache(maxsize=1)
def backend_availability():
    result={'exp_lif_cpu_reference':dict(available=True,device='CPU')}
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount()<1: raise RuntimeError('No NVIDIA GPU')
        probe=cp.RawKernel('extern "C" __global__ void probe(int* x) { x[0]=7; }','probe')
        value=cp.zeros(1,cp.int32); probe((1,),(1,),(value,))
        if int(value.get()[0])!=7: raise RuntimeError('CUDA kernel probe failed')
        name=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name']
        result['exp_lif_cuda']=dict(available=True,device=name.decode() if isinstance(name,bytes) else name,
                                   cupy=cp.__version__,runtime=cp.cuda.runtime.runtimeGetVersion())
    except Exception as exc:
        result['exp_lif_cuda']=dict(available=False,reason=str(exc))
    try:
        import torch
        if not torch.backends.mps.is_available() or not hasattr(torch.mps,'compile_shader'):
            raise RuntimeError('Apple MPS shader runtime unavailable')
        result['exp_lif_mps']=dict(available=True,device='Apple GPU',torch=torch.__version__)
    except Exception as exc:
        result['exp_lif_mps']=dict(available=False,reason=str(exc))
    return result


def resolve_backend(backend='auto'):
    from .neural import NEURAL_BACKENDS
    if backend=='auto':
        available=backend_availability()
        return next(name for name in ('exp_lif_cuda','exp_lif_mps','exp_lif_cpu_reference') if available[name]['available'])
    if backend not in NEURAL_BACKENDS: raise ValueError('Unknown neural backend: '+str(backend))
    return backend


def resolve_research_device(device='auto'):
    import torch
    if device=='auto':
        device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device not in ('cpu','mps','cuda'): raise ValueError('Research device must be auto, cpu, mps or cuda')
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('BLOCKED_CUDA: install CUDA-enabled PyTorch')
    if device=='mps' and not torch.backends.mps.is_available(): raise RuntimeError('BLOCKED_MPS')
    return device
