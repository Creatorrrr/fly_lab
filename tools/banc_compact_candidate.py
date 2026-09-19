"""Unmeasured CUDA candidate: exact24-bit CSR entries with sparse escapes.

This file may be imported for CPU packing without importing Torch or CuPy.
No production implementation is changed. A CUDA execution needs a later,
separately authorized experiment; the present study checks packing on CPU.
"""

from dataclasses import dataclass

import numpy as np

from flylab.c.adaptive_fused import SOURCE as POINTWISE_SOURCE
from flylab.c.adaptive_rate import AdaptiveRateNetwork

PAYLOAD_MASK = np.uint32((1 << 18) - 1)
ESCAPE_TAG = np.uint32(63)


@dataclass
class Packed24:
    words: np.ndarray
    hot_lookup: np.ndarray
    escapes: np.ndarray
    hot_original_ids: np.ndarray
    edge_count: int


def pack_entries(entries, lookup):
    """Encode the original packed entry bits; preserve the float lookup bits."""
    entries = np.asarray(entries)
    lookup = np.asarray(lookup)
    if entries.ndim != 1 or entries.dtype != np.uint32:
        raise ValueError("Original uint32 entry vector required")
    if lookup.ndim != 1 or lookup.dtype != np.float32 or not len(lookup) <= 2**14:
        raise ValueError("Original float32 lookup required")
    identifiers = entries >> 18
    if identifiers.size and int(identifiers.max()) >= len(lookup):
        raise ValueError("Entry lookup index out of bounds")
    counts = np.bincount(identifiers, minlength=len(lookup))
    hot_ids = np.argsort(-counts, kind="stable")[:63].astype(np.uint32)
    tags = np.full(len(lookup), 63, np.uint32)
    tags[hot_ids] = np.arange(len(hot_ids), dtype=np.uint32)
    edge_tags = tags[identifiers]
    is_escape = edge_tags == ESCAPE_TAG
    escapes = entries[is_escape].copy()
    if len(escapes) > 2**18:
        raise ValueError("Escape table exceeds18-bit payload")
    encoded = (entries & PAYLOAD_MASK) | (edge_tags << 18)
    encoded[is_escape] = np.arange(len(escapes), dtype=np.uint32) | np.uint32(63 << 18)
    # One guard word makes the second aligned32-bit load legal even for the
    # final entry. Byte packing and <u4 interpretation make endianness explicit.
    byte_count = 3 * len(entries)
    storage = np.zeros(((byte_count + 3) // 4 + 1) * 4, np.uint8)
    data = storage[:byte_count].reshape(-1, 3)
    for byte in range(3):
        data[:, byte] = encoded >> (8 * byte)
    return Packed24(
        storage.view("<u4"),
        lookup[hot_ids].copy(),
        escapes,
        hot_ids,
        len(entries),
    )


def unpack_entries(packed):
    """CPU mirror of the aligned two-word load and funnel-shift decode."""
    byte_offsets = np.arange(packed.edge_count, dtype=np.uint64) * 3
    word_offsets = byte_offsets >> 2
    shifts = (byte_offsets & 3) * 8
    low = packed.words[word_offsets].astype(np.uint64)
    high = packed.words[word_offsets + 1].astype(np.uint64)
    value = ((low | (high << 32)) >> shifts).astype(np.uint32) & np.uint32(0xFFFFFF)
    tags, payload = value >> 18, value & PAYLOAD_MASK
    hot = tags != ESCAPE_TAG
    result = np.empty(packed.edge_count, np.uint32)
    result[hot] = payload[hot] | (packed.hot_original_ids[tags[hot]] << 18)
    result[~hot] = packed.escapes[payload[~hot]]
    return result


SOURCE = (
    POINTWISE_SOURCE.split('extern "C" __global__ void csr_input', 1)[0]
    + r"""
extern "C" __global__ void compact_csr_input(
    const int* rowptr, const unsigned int* words, const float* lookup,
    const float* hot_lookup, const unsigned int* escapes,
    const float* masked, const float* drive, const float* threshold,
    const float* adaptation, const float* coefficient, float* input, int n) {
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= n) return;
    float value = 0.0f;
    for (int edge = rowptr[row] + lane; edge < rowptr[row + 1]; edge += 32) {
        unsigned int byte_offset = (unsigned int)edge * 3u;
        unsigned int word_offset = byte_offset >> 2;
        unsigned int shift = (byte_offset & 3u) << 3;
        // Both addresses are32-bit aligned. The allocation includes a guard.
        unsigned int item = __funnelshift_r(words[word_offset], words[word_offset + 1], shift) & 0xffffffu;
        unsigned int tag = item >> 18;
        unsigned int column = item & 0x3ffffu;
        float weight;
        if (tag == 63u) {
            item = escapes[column];
            column = item & 0x3ffffu;
            weight = lookup[item >> 18];
        } else {
            weight = hot_lookup[tag];
        }
        value += weight * masked[column];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    if (lane == 0) {
        value = add_exact(drive[row], value);
        value = sub_exact(value, threshold[row]);
        value = sub_exact(value, adaptation[row]);
        input[row] = mul_exact(coefficient[row], value);
    }
}
"""
)


class CompactCandidate(AdaptiveRateNetwork):
    def _init_fusion(self):
        # Runs in the existing warmup before CUDA Graph capture. This method
        # has deliberately not been executed during the CPU-only study.
        super()._init_fusion()
        original = self._columns.detach().cpu().numpy().view(np.uint32)
        lookup = self._weight_values.detach().cpu().numpy()
        packed = pack_entries(original, lookup)
        if unpack_entries(packed).tobytes() != original.tobytes():
            raise AssertionError("Compact entry roundtrip differs")
        self._compact_words = self.torch.as_tensor(
            packed.words.view(np.int32), device=self.device
        )
        self._compact_hot = self.torch.as_tensor(packed.hot_lookup, device=self.device)
        self._compact_escapes = self.torch.as_tensor(
            packed.escapes.view(np.int32), device=self.device
        )
        self._compact_kernel = self._cupy.RawKernel(
            SOURCE, "compact_csr_input", options=("--fmad=false",)
        )
        self._compact_kernel.compile()

    def _launch_fused(self, name, tensors, scalars=()):
        if name != "csr_input":
            return super()._launch_fused(name, tensors, scalars)
        inputs = (
            tensors[0],
            self._compact_words,
            tensors[2],
            self._compact_hot,
            self._compact_escapes,
            *tensors[3:],
        )
        self._compact_kernel(
            ((self.n + 7) // 8,),
            (256,),
            tuple(np.uint64(t.data_ptr()) for t in inputs) + (np.int32(self.n),),
        )
