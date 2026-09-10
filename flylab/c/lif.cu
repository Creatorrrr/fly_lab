// Ordered float32 LIF; compile with --fmad=false, without fast math.
extern "C" __global__ void integrate(
    float* v, float* h, float* rate, long long* refractory, long long* counts,
    float* queue, const unsigned char* suppress, const unsigned char* mute,
    unsigned char* emitted, const float* x, const float* c,
    const int* capture_map, unsigned char* events, const int* pulse_ptr,
    const float* pulses, int* health, const int* out_ptr, const int* out_indices,
    unsigned int* active_rows, const unsigned long long* clock,
    int n, int slots, int refractory_ticks, int k, int channels, int pulse_width,
    int integration, double rate_jump) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long tick = clock[0] + k;
    int slot = tick % slots;
    float current = h[i] + queue[(long long)slot*n+i];
    float external = v[i];
    for (int j=pulse_ptr[i]; j<pulse_ptr[i+1]; ++j) {
        if (integration == 2) external = external + pulses[(long long)k*pulse_width+j];
        else current = current + pulses[(long long)k*pulse_width+j];
    }
    bool eligible = tick >= (unsigned long long)refractory[i];
    if (integration == 2 && pulse_ptr[i] < pulse_ptr[i+1]) eligible = true;
    float base = c[0] + x[i];
    float delta = external - c[0];
    delta = delta - x[i];
    delta = delta * c[3];
    float next = base + delta;
    float synaptic = current * c[6];
    next = next + synaptic;
    if (integration != 2 || eligible) current = current * c[4];
    if (!eligible) next = c[1];
    bool spike = eligible && (integration == 2 ? next > c[2] : next >= c[2]) && !suppress[i];
    emitted[i] = spike && !mute[i];
    if (emitted[i]) {
        for (int j=out_ptr[i]; j<out_ptr[i+1]; ++j) atomicExch(active_rows+out_indices[j],1u);
    }
    if (spike) {
        if (integration != 0) current = 0.0f;
        next = c[1]; refractory[i] = tick+1+refractory_ticks; counts[i] += 1;
    }
    float r = rate[i] * c[5];
    // NumPy's bool * Python-float rate jump is float64 before in-place casting.
    r = (float)((double)r + (spike ? rate_jump : 0.0));
    v[i] = next; h[i] = current; rate[i] = r;
    if (!isfinite(next) || !isfinite(current) || !isfinite(r)) atomicExch(health,1);
    int channel = capture_map[i];
    if (channel >= 0) events[(long long)k*channels+channel] = spike;
}

extern "C" __global__ void propagate(
    const int* ptr, const int* indices, const float* weight,
    const unsigned char* emitted, unsigned int* active_rows, float* queue,
    const unsigned long long* clock, int n, int slots, int k) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    float sum = 0.0f;
    if (active_rows[i]) {
        for (int j=ptr[i]; j<ptr[i+1]; ++j) {
            float contribution = weight[j] * (float)emitted[indices[j]];
            sum = sum + contribution;
        }
    }
    active_rows[i] = 0;
    int slot = (clock[0]+k) % slots;
    queue[(long long)slot*n+i] = sum;
}

extern "C" __global__ void read_state(
    const float* v, const float* rate, const long long* count,
    const int* indices, unsigned char* output, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    float* values = (float*)output;
    long long* counts = (long long*)(output+8*n);
    int j=indices[i]; values[i]=v[j]; values[n+i]=rate[j]; counts[i]=count[j];
}

extern "C" __global__ void summarize(
    const float* v, const float* rate, const long long* count,
    unsigned char* output, int n) {
    int block=blockDim.x*blockIdx.x+threadIdx.x, begin=block*256;
    if (begin >= n) return;
    float lo=__int_as_float(0x7f800000), hi=-lo, sum=0; long long spikes=0;
    for (int j=begin; j<min(begin+256,n); ++j) {
        lo=fminf(lo,v[j]); hi=fmaxf(hi,v[j]); sum+=rate[j]; spikes+=count[j];
    }
    float* row=(float*)(output+24*block);
    row[0]=lo; row[1]=hi; row[2]=sum; row[3]=0;
    *((long long*)(output+24*block+16))=spikes;
}
