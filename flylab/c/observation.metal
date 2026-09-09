#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

// These kernels observe state only. No value is fed back into LIF integration.
kernel void read_state(device const float* v, device const float* rate,
                       device const long* count, device const int* indices,
                       device uchar* output, constant uint& n, uint i [[thread_position_in_grid]]) {
    if (i>=n) return;
    device float* values=reinterpret_cast<device float*>(output);
    device long* counts=reinterpret_cast<device long*>(output+8*n);
    int j=indices[i];values[i]=v[j];values[n+i]=rate[j];counts[i]=count[j];
}

kernel void summarize(device const float* v, device const float* rate,
                      device const long* count, device uchar* output,
                      constant uint& n, uint block [[thread_position_in_grid]]) {
    uint begin=block*256;
    if (begin>=n) return;
    float lo=INFINITY,hi=-INFINITY,sum=0;long spikes=0;
    for (uint j=begin;j<min(begin+256,n);++j) {
        lo=min(lo,v[j]);hi=max(hi,v[j]);sum+=rate[j];spikes+=count[j];
    }
    device float* row=reinterpret_cast<device float*>(output+24*block);
    row[0]=lo;row[1]=hi;row[2]=sum;row[3]=0;
    *reinterpret_cast<device long*>(output+24*block+16)=spikes;
}
