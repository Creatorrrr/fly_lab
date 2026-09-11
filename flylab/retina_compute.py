"""Cached official optical remap and ordered CUDA ommatidial reductions."""
import numpy as np


class RetinaCompute:
    def __init__(self,retina,backend='auto'):
        if backend not in ('auto','cpu','cuda'):raise ValueError('Invalid retinal compute backend')
        self.retina=retina
        n=retina.nrows*retina.ncols
        # Ask the official transform to remap encoded pixel addresses once. This
        # preserves its integer rounding and out-of-bounds decisions exactly.
        ids=np.arange(1,n+1,dtype=np.uint32).reshape(retina.nrows,retina.ncols)
        encoded=np.stack([ids&255,(ids>>8)&255,(ids>>16)&255],axis=-1).astype(np.uint8)
        mapped=retina.correct_fisheye(encoded).astype(np.int32)
        self.lookup=(mapped[...,0]+(mapped[...,1]<<8)+(mapped[...,2]<<16)-1).ravel()
        self.backend='cpu';self.cp=None
        if backend!='cpu':
            try:
                import cupy as cp
                if cp.cuda.runtime.getDeviceCount():self.cp=cp
            except (ImportError,RuntimeError):
                if backend=='cuda':raise
            if backend=='cuda' and self.cp is None:raise RuntimeError('Retinal CUDA device unavailable')
        if self.cp is not None:
            cp=self.cp;self.stream=cp.cuda.Stream(non_blocking=True)
            flat=retina.ommatidia_id_map.ravel();valid=np.flatnonzero(flat>0)
            ordered=valid[np.argsort(flat[valid],kind='stable')].astype(np.int32)
            counts=np.bincount(flat[valid],minlength=retina.num_ommatidia_per_eye+1)[1:]
            with self.stream:
                self.indices=cp.asarray(ordered);self.ptr=cp.asarray(np.r_[0,np.cumsum(counts)].astype(np.int32))
                self.mapping=cp.asarray(self.lookup);self.pale=cp.asarray(retina.pale_type_mask.astype(np.int32))
                self.raw=cp.empty((2,retina.nrows,retina.ncols,3),cp.uint8)
                self.output=cp.empty((2,retina.num_ommatidia_per_eye,2),cp.float32)
            self.kernel=cp.RawKernel(r'''
            extern "C" __global__ void retina(const unsigned char* raw,
                const int* lookup,const int* pixels,const int* ptr,const int* pale,
                float* out,int n,int m) {
                int i=blockIdx.x*blockDim.x+threadIdx.x,eye=blockIdx.y;
                if(i>=m)return;
                int ch=pale[i]; double sum=0.;int count=ptr[i+1]-ptr[i];
                for(int j=ptr[i];j<ptr[i+1];j++) {
                    int source=lookup[pixels[j]];
                    if(source>=0)sum+=double(raw[(eye*n+source)*3+ch+1])/double(count);
                }
                out[(eye*m+i)*2]=0.;out[(eye*m+i)*2+1]=0.;
                out[(eye*m+i)*2+ch]=float(sum/255.);
            }''','retina',options=('--fmad=false',))
            self.kernel.compile();self.backend='cuda'

    def process(self,images,*,include_rgb=True):
        r=self.retina;value=np.asarray(images)
        expected=(2,r.nrows,r.ncols,3)
        if value.shape!=expected or value.dtype!=np.uint8:raise ValueError('Two native uint8 eye images required')
        corrected=None
        if self.cp is None:
            # The official compiled remap is faster than a NumPy gather on CPU.
            corrected=np.asarray([r.correct_fisheye(im) for im in value])
        elif include_rgb:
            corrected=value.reshape(2,-1,3)[:,np.maximum(self.lookup,0)].copy()
            corrected[:,self.lookup<0]=0
            corrected=corrected.reshape(expected)
        if self.cp is None:
            ommatidia=np.asarray([r.raw_image_to_hex_pxls(im) for im in corrected],np.float32)
        else:
            with self.stream:
                self.raw.set(value,stream=self.stream)
                self.kernel(((r.num_ommatidia_per_eye+127)//128,2),(128,),
                    (self.raw,self.mapping,self.indices,self.ptr,self.pale,self.output,
                     np.int32(r.nrows*r.ncols),np.int32(r.num_ommatidia_per_eye)))
                ommatidia=self.output.get(stream=self.stream)
        return corrected if include_rgb else None,ommatidia
