#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

kernel void integrate(
    device float* v, device float* h, device float* rate,
    device long* refractory, device long* counts, device float* queue,
    device const uchar* suppress, device const uchar* mute, device uchar* emitted,
    device const float* x, device const float* c, device const int* capture_map,
    device uchar* events, device const int* pulse_ptr, device const float* pulses,
    device atomic_int* health, device const int* out_ptr, device const int* out_indices,
    device atomic_uint* active_rows, constant uint& n, constant ulong& tick,
    constant uint& slot, constant uint& refractory_ticks, constant uint& k,
    constant uint& channels, constant uint& pulse_width,
    uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float current = h[i] + queue[slot*n+i];
    for (int j=pulse_ptr[i]; j<pulse_ptr[i+1]; ++j) current = current + pulses[k*pulse_width+j];
    bool eligible = tick >= ulong(refractory[i]);
    float base = c[0] + x[i];
    float delta = v[i] - c[0];
    delta = delta - x[i];
    delta = delta * c[3];
    float next = base + delta;
    float synaptic = current * c[6];
    next = next + synaptic;
    current = current * c[4];
    if (!eligible) next = c[1];
    bool spike = eligible && next >= c[2] && !suppress[i];
    emitted[i] = spike && !mute[i];
    if (emitted[i]) {
        for (int j=out_ptr[i]; j<out_ptr[i+1]; ++j)
            atomic_store_explicit(active_rows+out_indices[j], 1u, memory_order_relaxed);
    }
    if (spike) { next = c[1]; refractory[i] = long(tick+1+refractory_ticks); counts[i] += 1; }
    float r = rate[i] * c[5];
    r = r + (spike ? c[7] : 0.0f);
    v[i] = next; h[i] = current; rate[i] = r;
    if (!isfinite(next) || !isfinite(current) || !isfinite(r)) atomic_store_explicit(health, 1, memory_order_relaxed);
    int channel = capture_map[i];
    if (channel >= 0) events[k*channels+channel] = spike;
}

kernel void propagate(device const int* ptr, device const int* indices,
                      device const float* weight, device const uchar* emitted,
                      device atomic_uint* active_rows, device float* queue, constant uint& n, constant uint& slot,
                      uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    bool active = atomic_load_explicit(active_rows+i, memory_order_relaxed);
    atomic_store_explicit(active_rows+i, 0u, memory_order_relaxed);
    float sum = 0.0f;
    if (active) {
        for (int j=ptr[i]; j<ptr[i+1]; ++j) {
            float contribution = weight[j] * float(emitted[indices[j]]);
            sum = sum + contribution;
        }
    }
    queue[slot*n+i] = sum;
}
