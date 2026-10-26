"""E312 normalized engineering postsynaptic load, not a conductance model.

Selected existing products retain sign and packed warp reduction order. All
presynaptic resources are neutral; the new f64 state belongs to selected targets.
"""

from __future__ import annotations

import copy
import hashlib

import numpy as np

from tools import banc_m3_ssp_recovery_synapses as ssp
from tools.banc_m3_motor_timescale import MotorTimescaleNetwork
from tools.banc_m3_source_filtered_synapses import _add, _equal, _mul

SCHEMA = "flylab.m3-postsynaptic-load.v1"
PRECISION = (
    ssp.PRECISION + ";E312:all-u0/R1;selected-original-product:f32-FTZ;"
    "U:abs-selected-products,original-warp32-f32-FTZ;"
    "L=f64(U)/(f64(threshold32)+f64(U));q=dt/f64(tau32);"
    "hFE=(1-q)*h+q*L;SSPRK33-h:f64-no-FMA;"
    "g64=1/(1+lambda*h);g32:noFTZ-conversion;"
    "selected-product*g32:f32-FTZ;lambda0:original-expression;"
    "same-old-stage-h/U-before-neural-overwrite;sticky-load/no-repair"
)
FAULT_LOAD = 64
FAULT_GROSS = 128
FAULT_LOAD_FACTOR = 256
SELECTION_KEYS = ssp.SELECTION_KEYS
_META_KEYS = {
    "schema",
    "precision",
    "source_model_hash",
    "selection",
    "target_ids",
    "threshold32",
    "tau32",
    "load_lambda",
}


def checked_selection(
    weights, edge_ids, premotor_ids, utilization, recovery_tau, motor_ids
):
    selection = ssp.checked_selection(
        weights, edge_ids, premotor_ids, utilization, recovery_tau, motor_ids
    )
    if np.any(selection["utilization"] != 0):
        raise ValueError("E312 requires every presynaptic utilization to be zero")
    return selection


def _lambda(value):
    if type(value) not in (int, float) or value not in (0, 1):
        raise ValueError("Exactly lambda0 or lambda1 required")
    return float(value)


def _state(value, count, name):
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.float64
        or value.shape != (count,)
        or not np.isfinite(value).all()
        or np.any((value < 0) | (value > 1))
    ):
        raise ValueError("Finite aligned f64 [0,1] " + name + " required")
    return value


def load_metadata(source_model_hash, selection, threshold32, tau32, load_lambda):
    neutral = ssp.resource_metadata(source_model_hash, selection)
    if np.any(neutral["utilization"] != 0):
        raise ValueError("No active source resource in E312")
    targets = np.unique(neutral["post_ids"])
    threshold, tau = np.asarray(threshold32), np.asarray(tau32)
    if (
        threshold.dtype != np.float32
        or tau.dtype != np.float32
        or threshold.ndim != 1
        or tau.shape != threshold.shape
        or len(threshold) <= targets.max()
        or not np.isfinite(threshold[targets]).all()
        or np.any(threshold[targets] <= 0)
        or not np.isfinite(tau[targets]).all()
        or np.any(tau[targets] <= 0)
    ):
        raise ValueError("Original positive selected-target threshold32/tau32 required")
    return {
        "schema": SCHEMA,
        "precision": PRECISION,
        "source_model_hash": source_model_hash,
        "selection": {k: neutral[k].copy() for k in SELECTION_KEYS},
        "target_ids": targets.copy(),
        "threshold32": threshold[targets].copy(),
        "tau32": tau[targets].copy(),
        "load_lambda": _lambda(load_lambda),
    }


def checked_load_metadata(metadata):
    if (
        not isinstance(metadata, dict)
        or set(metadata) != _META_KEYS
        or metadata.get("schema") != SCHEMA
        or metadata.get("precision") != PRECISION
    ):
        raise ValueError("Exact postsynaptic-load schema/precision required")
    neutral = ssp.resource_metadata(
        metadata["source_model_hash"], metadata["selection"]
    )
    targets = np.unique(neutral["post_ids"])
    if (
        not _equal(targets, metadata["target_ids"])
        or np.any(neutral["utilization"] != 0)
        or any(
            not isinstance(metadata[k], np.ndarray)
            or metadata[k].dtype != np.float32
            or metadata[k].shape != targets.shape
            or not np.isfinite(metadata[k]).all()
            or np.any(metadata[k] <= 0)
            for k in ("threshold32", "tau32")
        )
    ):
        raise ValueError(
            "Exact target ordering, neutral selection and native parameters required"
        )
    _lambda(metadata["load_lambda"])
    return copy.deepcopy(metadata)


def load_identity(metadata):
    metadata = checked_load_metadata(metadata)
    checksum = hashlib.sha256(
        (metadata["source_model_hash"] + SCHEMA + PRECISION).encode()
    )
    for key, value in sorted(
        {
            **metadata["selection"],
            **{k: metadata[k] for k in ("target_ids", "threshold32", "tau32")},
        }.items()
    ):
        checksum.update(key.encode() + value.dtype.str.encode())
        checksum.update(
            np.asarray(value.shape, np.int64).tobytes() + value.tobytes(order="C")
        )
    checksum.update(np.float64(metadata["load_lambda"]).tobytes())
    return checksum.hexdigest()


def load_factors(h, load_lambda):
    _state(h, len(h), "load state")
    value = _lambda(load_lambda)
    return (1.0 / (1.0 + value * h)).astype(np.float32)


def load_euler(h, gross, threshold32, tau32, dt):
    """Owned f64 Euler map using a completed original f32 gross reduction."""
    _state(h, len(h), "load state")
    U, threshold, tau = map(np.asarray, (gross, threshold32, tau32))
    if (
        any(v.dtype != np.float32 or v.shape != h.shape for v in (U, threshold, tau))
        or not np.isfinite(U).all()
        or np.any(U < 0)
        or not np.isfinite(threshold).all()
        or np.any(threshold <= 0)
        or not np.isfinite(tau).all()
        or np.any(tau <= 0)
        or type(dt) not in (int, float)
        or not np.isfinite(dt)
        or dt <= 0
        or dt > float(tau.min()) / 2
    ):
        raise ValueError(
            "Finite nonnegative gross, positive native parameters and resolved dt required"
        )
    U64 = U.astype(np.float64)
    L = U64 / (threshold.astype(np.float64) + U64)
    q = dt / tau.astype(np.float64)
    result = (1.0 - q) * h + q * L
    _state(result, len(h), "Euler load state")
    return result


def packed_load_oracle(weights, rate, mask, selection, h, load_lambda):
    """Literal original warp f32 current and selected gross, with owned outputs."""
    if (
        not isinstance(rate, np.ndarray)
        or rate.dtype != np.float32
        or rate.shape != (weights.shape[0],)
        or not np.isfinite(rate).all()
        or np.any(rate < 0)
        or not isinstance(mask, np.ndarray)
        or mask.dtype != np.float32
        or mask.shape != rate.shape
        or not np.isin(mask, (0, 1)).all()
    ):
        raise ValueError(
            "Finite nonnegative native rate and binary native mask required"
        )
    selection = checked_selection(
        weights,
        *(
            selection[k]
            for k in (
                "edge_ids",
                "premotor_ids",
                "utilization",
                "recovery_tau",
                "motor_ids",
            )
        ),
    )
    targets = np.unique(selection["post_ids"])
    factors = load_factors(_state(h, len(targets), "load state"), load_lambda)
    chosen = set(selection["edge_ids"].tolist())
    slots = dict(zip(targets.tolist(), range(len(targets)), strict=True))
    current = np.zeros(len(rate), np.float32)
    gross = np.zeros(len(targets), np.float32)
    for row in range(len(rate)):
        net, total = np.zeros(32, np.float32), np.zeros(32, np.float32)
        slot = slots.get(row)
        for lane in range(32):
            for edge in range(weights.indptr[row] + lane, weights.indptr[row + 1], 32):
                pre = weights.indices[edge]
                masked = np.float32(rate[pre] * mask[pre])  # Original noFTZ mask.
                original = _mul(weights.data[edge], masked)
                if edge in chosen:
                    total[lane] = _add(total[lane], np.abs(original))
                    if load_lambda != 0:
                        original = _mul(original, factors[slot])
                net[lane] = _add(net[lane], original)
        for offset in (16, 8, 4, 2, 1):
            old_net, old_total = net.copy(), total.copy()
            for lane in range(32 - offset):
                net[lane] = _add(old_net[lane], old_net[lane + offset])
                total[lane] = _add(old_total[lane], old_total[lane + offset])
        current[row] = net[0]
        if slot is not None:
            gross[slot] = total[0]
    if not np.isfinite(gross).all() or np.any(gross < 0):
        raise ssp.prior._fault_error(FAULT_GROSS)
    return current, gross


_LOAD_SOURCE = r"""
extern "C" __global__ void load_csr(
    const int* rowptr,const unsigned int* entries,const float* lookup,
    const float* masked,const float* drive,const float* threshold,
    const float* adaptation,const float* coefficient,float* input,
    const int* selected,const int* slots,const double* h,float* gross,float* factor,
    int* fault,float lambda,int n) {
    int lane=threadIdx.x&31,row=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
    if (row>=n) return;
    int slot=slots[row];
    double oldh=slot<0?0.0:h[slot];
    int flags=(!isfinite(oldh)||oldh<0.0||oldh>1.0)?64:0;
    double g=1.0/(1.0+(double)lambda*oldh);
    if (!isfinite(g)||g<0.5||g>1.0) flags|=256;
    float gf=recovery_f64_to_f32(g);
    float value=0.0f,total=0.0f;
    for (int edge=rowptr[row]+lane;edge<rowptr[row+1];edge+=32) {
        unsigned int item=entries[edge];
        float original=lookup[item>>18]*masked[item&0x3ffff];
        if (selected[edge]>=0) {
            total+=fabsf(original);
            if (lambda==0.0f) {
                value+=lookup[item>>18]*masked[item&0x3ffff];
            } else {
                value+=original*gf;
            }
        } else {
            value+=lookup[item>>18]*masked[item&0x3ffff];
        }
    }
    for (int offset=16;offset>0;offset>>=1) {
        value+=__shfl_down_sync(0xffffffff,value,offset);
        total+=__shfl_down_sync(0xffffffff,total,offset);
    }
    if (lane==0) {
        if (slot>=0) {
            gross[slot]=total;factor[slot]=gf;
            if (recovery_bad_rate(total)) flags|=128;
        }
        if (flags) atomicOr(fault,flags);
        value=add_exact(drive[row],value);
        value=sub_exact(value,threshold[row]);value=sub_exact(value,adaptation[row]);
        input[row]=mul_exact(coefficient[row],value);
    }
}
extern "C" __global__ void load_stage(const double* h,const double* base,
    const float* gross,const float* threshold,const float* tau,double* next,
    int* fault,double dt,double b,double w,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    double U=recovery_f32_to_f64(gross[i]);
    double th=recovery_f32_to_f64(threshold[i]),time=recovery_f32_to_f64(tau[i]);
    double L=U/(th+U),q=dt/time;
    double fe=(1.0-q)*h[i]+q*L;
    double nh=b*base[i]+w*fe;
    if (!isfinite(nh)||nh<0.0||nh>1.0) atomicOr(fault,64);
    next[i]=nh;
}
"""


def load_cuda_source(fuse_tanh=True):
    return ssp.resource_cuda_source(fuse_tanh) + _LOAD_SOURCE


class PostsynapticLoadNetwork(ssp.SSPRecoveryResourceNetwork):
    """Coupled SSP r/z/h, constant R1; lambda0 retains SSP neural bytes."""

    def __init__(
        self,
        *args,
        edge_ids,
        premotor_ids,
        utilization,
        recovery_tau,
        load_lambda,
        **kwargs,
    ):
        selection = checked_selection(
            args[0],
            edge_ids,
            premotor_ids,
            utilization,
            recovery_tau,
            kwargs.get("motor_ids"),
        )
        self.load_lambda = _lambda(load_lambda)
        self._load_targets_numpy = np.unique(selection["post_ids"])
        self._load_threshold_numpy = np.asarray(args[3], np.float32)[
            self._load_targets_numpy
        ].copy()
        self._load_tau_numpy = np.asarray(args[1], np.float32)[
            self._load_targets_numpy
        ].copy()
        # Resolve all inputs before any native construction/warmup.
        load_metadata(
            "0" * 64,
            selection,
            np.asarray(args[3], np.float32),
            np.asarray(args[1], np.float32),
            self.load_lambda,
        )
        super().__init__(
            *args,
            edge_ids=edge_ids,
            premotor_ids=premotor_ids,
            utilization=utilization,
            recovery_tau=recovery_tau,
            **kwargs,
        )
        self._load_metadata = load_metadata(
            self.source_timescale_identity,
            self._selection,
            self._native_constants["threshold"],
            self._native_constants["tau"],
            self.load_lambda,
        )
        self.identity = load_identity(self._load_metadata)
        self._inherited_metadata["model_hash"] = self.identity
        self.postsynaptic_load_state.zero_()
        self._resource_fault.zero_()
        self.load_gross.zero_()
        self.load_factor.fill_(1)
        self._load_names = (
            "postsynaptic_load_state",
            "_load_work",
            "load_gross",
            "load_factor",
            "_load_target_ids",
            "_load_row_slots",
            "_load_threshold",
            "_load_tau",
        )
        self._load_pointers = {k: getattr(self, k).data_ptr() for k in self._load_names}
        self._load_versions = {
            k: getattr(self, k)._version
            for k in self._load_names[4:]
            if not getattr(self, k).is_inference()
        }
        self._lambda_original = self.load_lambda
        self._load_gross_valid = False

    def _init_resource(self):
        super()._init_resource()
        if hasattr(self, "postsynaptic_load_state"):
            return
        t = self.torch
        slots = np.full(self.n, -1, np.int32)
        slots[self._load_targets_numpy] = np.arange(
            len(self._load_targets_numpy), dtype=np.int32
        )
        with t.inference_mode(False):
            self.postsynaptic_load_state = t.zeros(
                len(self._load_targets_numpy), dtype=t.float64, device=self.device
            )
            self._load_work = t.empty_like(self.postsynaptic_load_state)
            self.load_gross = t.zeros_like(
                self.postsynaptic_load_state, dtype=t.float32
            )
            self.load_factor = t.ones_like(self.load_gross)
            for key, value in (
                ("_load_target_ids", self._load_targets_numpy),
                ("_load_row_slots", slots),
                ("_load_threshold", self._load_threshold_numpy),
                ("_load_tau", self._load_tau_numpy),
            ):
                setattr(self, key, t.as_tensor(value.copy(), device=self.device))

    def _assert_pointers(self):
        super()._assert_pointers()
        if hasattr(self, "_load_pointers") and (
            self.load_lambda != self._lambda_original
            or any(
                getattr(self, key).data_ptr() != pointer
                for key, pointer in self._load_pointers.items()
            )
            or any(
                getattr(self, key)._version != version
                for key, version in self._load_versions.items()
            )
        ):
            raise ValueError("Captured postsynaptic pointer/parameter changed")

    def _assert_parameters(self):
        super()._assert_parameters()
        if hasattr(self, "_load_metadata"):
            if load_identity(self._load_metadata) != self.identity:
                raise ValueError("Postsynaptic metadata identity changed")
            slots = np.full(self.n, -1, np.int32)
            slots[self._load_targets_numpy] = np.arange(
                len(self._load_targets_numpy), dtype=np.int32
            )
            for key, expected in (
                ("_load_target_ids", self._load_metadata["target_ids"]),
                ("_load_row_slots", slots),
                ("_load_threshold", self._load_metadata["threshold32"]),
                ("_load_tau", self._load_metadata["tau32"]),
            ):
                if not _equal(getattr(self, key).cpu().numpy(), expected):
                    raise ValueError("Owned postsynaptic parameter bytes changed")

    @property
    def load_metadata(self):
        return copy.deepcopy(self._load_metadata)

    @property
    def load_fault_flags(self):
        """Diagnostic host copy only; normal readout uses the inherited one packet."""
        return self.resource_fault_flags

    def _init_fusion(self):
        self._init_resource()
        MotorTimescaleNetwork._init_fusion(self)
        source = load_cuda_source(self.fuse_tanh)
        for name in ("load_csr", "load_stage", "recovery_factor", "ssp_neuron"):
            kernel = self._cupy.RawKernel(source, name, options=("--fmad=false",))
            kernel.compile()
            self._fused_kernels[name] = kernel

    def _step_fused(self):
        if not hasattr(self, "_fused_kernels"):
            self._init_fusion()
        masked, _, _, inputs, activation, *_ = self._fused_buffers
        r, z, h = self.rate, self.adaptation, self.postsynaptic_load_state
        wr, wz = self._ssp_buffers
        self.torch.mul(r, self.output_mask, out=masked)
        with self._cupy.cuda.ExternalStream(
            self.torch.cuda.current_stream().cuda_stream
        ):
            for stage, (b, w) in enumerate(((0.0, 1.0), (0.75, 0.25), (1 / 3, 2 / 3))):
                self._recovery_launch(
                    "recovery_factor",
                    (
                        self._resource_sources,
                        r,
                        self.resource,
                        self._resource_u,
                        self._resource_tau,
                        self.resource_factor,
                        self._resource_fault,
                    ),
                )
                tensors = (
                    self._rowptr,
                    self._columns,
                    self._weight_values,
                    masked,
                    self.drive,
                    self.threshold,
                    z,
                    self.coefficient,
                    inputs,
                    self._resource_edge_source,
                    self._load_row_slots,
                    h,
                    self.load_gross,
                    self.load_factor,
                    self._resource_fault,
                )
                self._fused_kernels["load_csr"](
                    ((self.n + 7) // 8,),
                    (256,),
                    tuple(np.uint64(v.data_ptr()) for v in tensors)
                    + (np.float32(self.load_lambda), np.int32(self.n)),
                )
                if self.fuse_tanh:
                    activation = inputs
                else:
                    self.torch.tanh(inputs, out=activation)
                nr, nz, nh = (
                    (self.rate, self.adaptation, self.postsynaptic_load_state)
                    if stage == 2
                    else (wr, wz, self._load_work)
                )
                # h consumes old-stage masked products before the neural stage overwrites r.
                tensors = (
                    h,
                    self.postsynaptic_load_state,
                    self.load_gross,
                    self._load_threshold,
                    self._load_tau,
                    nh,
                    self._resource_fault,
                )
                count = len(self._load_targets_numpy)
                self._fused_kernels["load_stage"](
                    ((count + 255) // 256,),
                    (256,),
                    tuple(np.uint64(v.data_ptr()) for v in tensors)
                    + (
                        np.float64(self.dt),
                        np.float64(b),
                        np.float64(w),
                        np.int32(count),
                    ),
                )
                tensors = (
                    activation,
                    self.cap,
                    r,
                    z,
                    self.tau,
                    self._cell_gain,
                    self._cell_inverse_tau,
                    self.rate,
                    self.adaptation,
                    self.output_mask,
                    nr,
                    nz,
                    masked,
                    self._resource_fault,
                )
                self._fused_kernels["ssp_neuron"](
                    ((self.n + 255) // 256,),
                    (256,),
                    tuple(np.uint64(v.data_ptr()) for v in tensors)
                    + (
                        np.float32(self.dt),
                        np.float32(b),
                        np.float32(w),
                        np.int32(self.n),
                    ),
                )
                r, z, h = nr, nz, nh

    def _step(self):
        if self.device == "cuda":
            return self._step_fused()
        t = self.torch
        r0, z0, h0 = self.rate, self.adaptation, self.postsynaptic_load_state
        r, z, h = r0, z0, h0
        wr, wz = self._ssp_buffers
        for stage, (b, w) in enumerate(((0.0, 1.0), (0.75, 0.25), (1 / 3, 2 / 3))):
            self._native_guard(r, z)
            self._cpu_factors(r, self.resource)  # Sticky fault / neutral R1 guard.
            try:
                current, gross = packed_load_oracle(
                    self._source_weights,
                    r.numpy(),
                    self.output_mask.numpy(),
                    self._selection,
                    h.numpy(),
                    self.load_lambda,
                )
                next_h = b * h0.numpy() + w * load_euler(
                    h.numpy(),
                    gross,
                    self._load_threshold.numpy(),
                    self._load_tau.numpy(),
                    self.dt,
                )
                _state(next_h, len(h), "SSP load state")
            except (ValueError, RuntimeError) as error:
                flags = getattr(error, "resource_fault_flags", FAULT_LOAD)
                self._resource_fault.bitwise_or_(flags)
                raise ssp.prior._fault_error(
                    int(self._resource_fault.numpy()[0])
                ) from error
            self.load_gross.copy_(t.as_tensor(gross))
            self.load_factor.copy_(
                t.as_tensor(load_factors(h.numpy(), self.load_lambda))
            )
            current = t.as_tensor(current)
            inputs = self.drive + current
            inputs = inputs - self.threshold
            inputs = inputs - z
            inputs = self.coefficient * inputs
            target = t.clamp_min(self.cap * t.tanh(inputs), 0.0)
            qr, qz = self.dt / self.tau, self.dt * self._cell_inverse_tau
            rfe, zfe = (
                (1.0 - qr) * r + qr * target,
                (1.0 - qz) * z + qz * (self._cell_gain * r),
            )
            nr, nz = b * r0 + w * rfe, b * z0 + w * zfe
            self._native_guard(nr, nz)
            if stage == 2:
                r0.copy_(nr)
                z0.copy_(nz)
                h0.copy_(t.as_tensor(next_h))
            else:
                wr.copy_(nr)
                wz.copy_(nz)
                self._load_work.copy_(t.as_tensor(next_h))
                r, z, h = wr, wz, self._load_work

    def _record_final_faults(self):
        super()._record_final_faults()
        h = self.postsynaptic_load_state
        bad = (~self.torch.isfinite(h)).any() | ((h < 0) | (h > 1)).any()
        self._resource_fault.bitwise_or_(
            (bad.to(self.torch.int32) * FAULT_LOAD).reshape(1)
        )

    def snapshot(self):
        self._readout_packet(None)  # No successful snapshot of sticky fault.
        return {
            **super().snapshot(),
            "postsynaptic_load": self.load_metadata,
            "postsynaptic_load_state": self.postsynaptic_load_state.cpu()
            .numpy()
            .copy(),
        }

    def _prevalidate(self, saved):
        if not isinstance(saved, dict) or not _equal(
            saved.get("postsynaptic_load"), self._load_metadata
        ):
            raise ValueError("Exact postsynaptic-load identity required")
        h = _state(
            saved.get("postsynaptic_load_state"),
            len(self._load_targets_numpy),
            "saved load state",
        )
        if saved.get("tick") == 0 and np.any(h != 0):
            raise ValueError("Zero-tick load must be zero")
        inherited = {
            k: v
            for k, v in saved.items()
            if k not in ("postsynaptic_load", "postsynaptic_load_state")
        }
        staged = super()._prevalidate(inherited)
        staged["postsynaptic_load_state"] = self.torch.as_tensor(
            h.copy(), device=self.device
        )
        return staged

    def reset(self):
        super().reset()
        self.postsynaptic_load_state.zero_()
        self.load_gross.zero_()
        self.load_factor.fill_(1)
        self._load_gross_valid = False

    def restore(self, saved):
        super().restore(saved)
        self.load_gross.zero_()
        self.load_factor.fill_(1)
        self._load_gross_valid = False

    def _advance(self, drive, steps, *, readout_indices=None):
        self._load_gross_valid = False
        result = super()._advance(drive, steps, readout_indices=readout_indices)
        self._load_gross_valid = True
        return result

    def load_readout(self):
        """One diagnostic packet: endpoint h/g, separately labelled last-stage U."""
        self._assert_pointers()
        self._record_final_faults()
        h = self.postsynaptic_load_state
        g = 1.0 / (1.0 + self.load_lambda * h)
        packet = self.torch.cat(
            (
                h,
                g,
                self.load_gross.to(self.torch.float64),
                self._resource_fault.to(self.torch.float64),
            )
        )
        values = packet.detach().cpu().numpy()
        if values[-1] != 0:
            raise ssp.prior._fault_error(int(values[-1]))
        n = len(self._load_targets_numpy)
        return {
            "target_ids": self._load_metadata["target_ids"].copy(),
            "h": values[:n].copy(),
            "g": values[n : 2 * n].astype(np.float32),
            "gross": values[2 * n : 3 * n].astype(np.float32),
            "gross_available": self._load_gross_valid,
            "gross_phase": "last-SSPRK33-stage-input",
            "gross_tick": self.tick - 1 if self._load_gross_valid else None,
            "completed_tick": self.tick,
        }

    def import_base_snapshot(self, saved):
        if (
            not isinstance(saved, dict)
            or saved.get("model_hash") != self.source_timescale_identity
        ):
            raise ValueError("Exact original MotorTimescale identity required")
        if (
            type(saved.get("tick")) is not int
            or saved["tick"] != 0
            or any(
                not isinstance(saved.get(k), np.ndarray)
                or saved[k].dtype != np.float32
                or saved[k].shape != (self.n,)
                or not np.all(saved[k] == 0)
                for k in ("rate", "adaptation", "drive")
            )
            or not isinstance(saved.get("output_mask"), np.ndarray)
            or saved["output_mask"].dtype != np.float32
            or saved["output_mask"].shape != (self.n,)
            or not np.all(saved["output_mask"] == 1)
        ):
            raise ValueError(
                "Base import requires tick0/rate0/adaptation0/drive0/mask1"
            )
        candidate = copy.deepcopy(saved)
        candidate.update(
            model_hash=self.identity,
            recurrent_resource=copy.deepcopy(self._metadata),
            synaptic_resource=np.ones(len(self._selection["source_ids"]), np.float64),
            postsynaptic_load=self.load_metadata,
            postsynaptic_load_state=np.zeros(len(self._load_targets_numpy), np.float64),
        )
        self.restore(candidate)

    def load_current_oracle(self, *, rate=None, h=None):
        self._assert_parameters()
        self._readout_packet(None)
        r = self.rate.cpu().numpy().copy() if rate is None else rate
        state = self.postsynaptic_load_state.cpu().numpy().copy() if h is None else h
        return packed_load_oracle(
            self._source_weights,
            r,
            self.output_mask.cpu().numpy(),
            self._selection,
            state,
            self.load_lambda,
        )

    def _current(self, rate):
        if self.load_lambda:
            raise RuntimeError("Joint load state required; use load_current_oracle")
        return super()._current(rate)

    def _derivative_pair(self, rate, adaptation):
        if self.load_lambda:
            raise RuntimeError("Joint load state required; use load_current_oracle")
        return super()._derivative_pair(rate, adaptation)

    def resource_current_oracle(self, *, rate=None, resource=None):
        if self.load_lambda:
            raise RuntimeError("Joint load state required; use load_current_oracle")
        return super().resource_current_oracle(rate=rate, resource=resource)
