"""E343 fixed common direction of submitted C and rounded-adjoint S proxy.

No direct S parameter VJP identity or float32 adjoint linearity is claimed.
Import is passive; original execution requires current producer/source review.
"""

from __future__ import annotations

import time

_IMPORT_S0 = time.perf_counter()

import argparse
import copy
import math
import os
import subprocess
import sys
import threading
from pathlib import Path

for _key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_key] = "1"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import verify_c_banc_m3_coordination_parameter_direction as e

BASE = e.BASE
OWNED = (
    "tools/verify_c_banc_m3_common_descent_direction.py",
    "tests/test_c_banc_common_descent_direction.py",
)
DESIGN = "docs/BANC_M3_E343_COMMON_DESCENT_DIRECTION_DESIGN_20261002.md"
DESIGN_SHA = "056de6c1b75e69e0dad35497e06a483ecee425b3ff5d51f8defebb123e0404e6"
REGISTRATION = BASE + "/parent-e343-reflected-design-registration.json"
REGISTRATION_SHA = "e643662e8cb924894387c6d9d432e0b4a254d6c4728cc63b1982629189cada08"
INVENTORY = BASE + "/parent-e343-selected-input-inventory-reflected.json"
INVENTORY_SHA = "df16603f01b41d8b96e80ec0aa184b9635826faa6fad7bb3fe2dd1001b69c851"
E342_SOURCE = "tools/verify_c_banc_m3_coordination_parameter_direction.py"
E342_SHA = "fb4d2687baf9ad89b919b7c4692d7039992a319ffef082686933d1bbc563bb2d"
SEALED = BASE + "/parent-e342-sealed-native-input.json"
SEALED_SHA = "f7d3f8a086468fe72fa7a466978e10b6aaccceec6c122a08dd5e1d8369076e9d"
OUTPUT = BASE + "/e343-common-descent-direction"
FREEZE = BASE + "/e343-common-descent-direction-freeze.json"
REVIEW = BASE + "/parent-e343-original-source-review.json"
WINDOWS = e.WINDOWS
SCALE = 12.5
FIXED = {
    **e.FIXED,
    "science_s": 25,
    "internal_s": 30,
    "supervisor_s": 35,
    "legacy_mean_windows": 50,
    "current_mean_windows": 4,
    "reconstruction_scale": SCALE,
    "new_neural_backwards": 0,
    "direct_S_VJP_numerical_byte_identity_proven": False,
    "internal_float32_adjoint_linearity_proven": False,
    "parameter_proxy_independently_proven": False,
}
LIMITATIONS = {
    "direct_S_VJP_numerical_byte_identity_proven": False,
    "internal_float32_adjoint_linearity_proven": False,
    "parameter_proxy_independently_proven": False,
    "q_only_derivative_check_proves_parameter_proxy": False,
    "full_parameter_Jacobian_independently_proven": False,
}
require = e.require
TestDimensions = e.TestDimensions
receipt_dict = e.receipt_dict
patch_weights = e.patch_weights
reconciliation = e.reconciliation
window_loss = e.window_loss
loss_vector = e.loss_vector
assemble = e.assemble
Store = e.Store
usage = e.usage


def source_map(root=ROOT):
    return {name: e.digest(e.safe(root, name)) for name in OWNED}


def generated_path(root, name):
    target = e.safe(root, name)
    parts = target.relative_to(e.safe(root, BASE)).parts
    require(
        parts
        and parts[0].startswith(
            (
                "e343-sol-first-",
                "parent-e343-audit-",
                "e342-sol-first-native-cuda-fixture-01",
            )
        ),
        "explicit generated namespace",
    )
    return target


def load_selected(root, binding, ledger, *, timely=lambda: None, generated=False):
    if generated:
        generated_path(root, binding["path"])
    # E343 owns its generated namespace; the old E342 namespace guard is not patched.
    return e.load_selected(root, binding, ledger, timely=timely, generated=False)


def load_packet(
    root, inventory, *, dimensions=None, metadata=None, timely=lambda: None, ledger=None
):
    import numpy as np

    generated = dimensions is not None
    require(
        not generated or isinstance(dimensions, TestDimensions),
        "explicit generated dimensions",
    )
    require(generated or metadata is None, "original metadata override forbidden")
    ledger = [] if ledger is None else ledger
    roles = inventory["role_order"]
    require(
        len(roles) == len(set(roles)) == 32
        and inventory["input_count"] == 32
        and inventory["selected_value_count"] == 123
        and set(roles) == set(inventory["bindings"]),
        "literal32/123",
    )
    packet = {
        "metadata": e.metadata_packet(root, inventory)
        if metadata is None
        else metadata,
        "ledger": ledger,
        "bindings": inventory["bindings"],
    }
    for role in roles:
        value = load_selected(
            root,
            inventory["bindings"][role],
            ledger,
            timely=timely,
            generated=generated,
        )
        if role.startswith("drive-"):
            start = int(role.split("-")[1])
            first, begin, end = e.SLICES[WINDOWS.index(start)]
            require(
                np.array_equal(
                    value["control_tick"], np.arange(first, first + 25, dtype=np.int64)
                )
                and np.array_equal(value["network_tick"], value["control_tick"] * 20),
                "complete drive chronology",
            )
            value = {
                "drive": value["drive"][begin:end].copy(),
                "source_slice": [begin, end],
            }
        packet[role] = value
    e.validate_packet(packet, dimensions=dimensions)
    groups = e.axes(dimensions)["groups"]
    shapes = {"gtheta_C": (4, groups), "sum_gtheta_C": (groups,), "gq_C": (4, 4, 42)}
    for key, shape in shapes.items():
        valid_array(packet["audited_C_vjp"][key], shape)
    require(
        e.exact(
            packet["audited_C_vjp"]["gtheta_C"].sum(axis=0),
            packet["audited_C_vjp"]["sum_gtheta_C"],
        ),
        "audited C vector sum identity",
    )
    for c in WINDOWS:
        valid_array(packet[f"legacy_total_gradient-{c}"]["gradient"], (groups,))
        valid_array(packet[f"legacy_total_gq-{c}"]["gq"], (4, 42))
    return packet


def valid_array(value, shape=None):
    import numpy as np

    require(
        isinstance(value, np.ndarray)
        and value.dtype == np.float64
        and (shape is None or value.shape == shape)
        and np.isfinite(value).all(),
        "finite float64 array",
    )
    return value


def reconstruct(total_gtheta, total_gq, c_gtheta, c_gq):
    import numpy as np

    valid_array(total_gtheta)
    require(
        total_gtheta.ndim == 2 and total_gtheta.shape[0] == 4,
        "four submitted parameter windows",
    )
    valid_array(c_gtheta, total_gtheta.shape)
    valid_array(total_gq, (4, 4, 42))
    valid_array(c_gq, (4, 4, 42))
    with np.errstate(over="raise", invalid="raise"):
        proxy = SCALE * total_gtheta - c_gtheta
        output = SCALE * total_gq - c_gq
        sum_c, sum_s = c_gtheta.sum(axis=0), proxy.sum(axis=0)
    for value in (proxy, output, sum_c, sum_s):
        valid_array(value)
    return {
        "gtheta_S_proxy": proxy,
        "gq_S_reconstructed": output,
        "sum_gtheta_C": sum_c,
        "sum_gtheta_S_proxy": sum_s,
    }


def scaled_unit(vector):
    import numpy as np

    valid_array(vector)
    require(vector.ndim == 1 and vector.size > 0, "nonempty vector")
    magnitude = float(np.max(np.abs(vector)))
    if magnitude == 0:
        return None, {"max_abs": 0.0, "scaled_L2": 0.0, "L2": 0.0}
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        scaled = vector / magnitude
        norm = float(np.sqrt(np.dot(scaled, scaled)))
        unit = scaled / norm
    require(
        math.isfinite(norm) and norm > 0 and np.isfinite(unit).all(), "scaled L2 unit"
    )
    absolute = magnitude * norm
    return unit, {
        "max_abs": magnitude,
        "scaled_L2": norm,
        "L2": absolute if math.isfinite(absolute) else None,
        "L2_null_reason": None
        if math.isfinite(absolute)
        else "not_representable_float64",
    }


def common_direction(c_vector, s_proxy_vector):
    import numpy as np

    valid_array(c_vector)
    valid_array(s_proxy_vector, c_vector.shape)
    uc, nc = scaled_unit(c_vector)
    us, ns = scaled_unit(s_proxy_vector)
    info = {
        "branch": "zero_component_gradient",
        "norm_C": nc,
        "norm_S_proxy": ns,
        "positive_dot_scope": "submitted C and reconstructed rounded-adjoint S proxy only",
        **LIMITATIONS,
    }
    raw = {
        "component_max_abs": np.array([nc["max_abs"], ns["max_abs"]], np.float64),
        "component_scaled_L2": np.array([nc["scaled_L2"], ns["scaled_L2"]], np.float64),
    }
    if uc is None or us is None:
        return raw, info
    raw.update(
        unit_C=uc,
        unit_S_proxy=us,
        cosine=np.array(np.dot(uc, us), np.float64),
        bisector=uc + us,
    )
    info["cosine"] = float(raw["cosine"])
    magnitude = float(np.max(np.abs(raw["bisector"])))
    if magnitude == 0:
        info["branch"] = "common_descent_unavailable"
        return raw, info
    direction = raw["bisector"] / magnitude
    with np.errstate(over="raise", invalid="raise"):
        plus = np.float32(1 + 0.001 * direction).astype(np.float64)
        minus = np.float32(1 - 0.001 * direction).astype(np.float64)
        delta = plus - minus
        tc, ts = float(np.dot(c_vector, delta)), float(np.dot(s_proxy_vector, delta))
    require(
        math.isfinite(tc)
        and math.isfinite(ts)
        and np.all(plus > 0)
        and np.all(minus > 0),
        "finite actual rounded geometry",
    )
    raw.update(
        direction=direction,
        theta_plus=plus,
        theta_minus=minus,
        actual_delta_theta=delta,
        geometry_T_C=np.array(tc, np.float64),
        geometry_T_S_proxy=np.array(ts, np.float64),
    )
    info.update(
        branch="normal"
        if tc > 1e-9 and ts > 1e-9
        else "actual_rounded_common_descent_uninformative",
        actual_T_C=tc,
        actual_T_S_proxy=ts,
    )
    return raw, info


def analyze(raw, record):
    import numpy as np

    branch = record.get("branch")
    require(
        branch
        in (
            "normal",
            "zero_component_gradient",
            "common_descent_unavailable",
            "actual_rounded_common_descent_uninformative",
            "native_palette_not_deployable",
        ),
        "completed branch",
    )
    result = {
        "branch": branch,
        "parameter_S_semantics": "surrogate proxy from two rounded adjoint outputs",
        "observed_response": "not_evaluated",
        "direction_qualification": False,
        "joint_measured_reduction": False,
        "I_C": None,
        "I_S": None,
        "C_O_Q": None,
        "C_O_T": None,
        "S_O_Q": None,
        "S_O_T": None,
        **LIMITATIONS,
    }
    if branch != "normal":
        return result
    for component in ("C", "S"):
        for comparison in ("Q", "T"):
            result[f"{component}_O_{comparison}"] = reconciliation(
                raw[f"O_{component}"], raw[f"{comparison}_{component}"]
            )
        aggregate = float(
            np.dot(
                raw["sum_gtheta_C" if component == "C" else "sum_gtheta_S_proxy"],
                raw["actual_delta_theta"],
            )
        )
        summed = float(raw[f"T_{component}"].sum())
        require(
            math.isfinite(aggregate)
            and math.isfinite(summed)
            and math.isclose(aggregate, summed, rel_tol=1e-10, abs_tol=1e-12)
            and aggregate > 1e-9,
            "rounded common dot/window invariant",
        )
        result[f"aggregate_T_{component}"] = aggregate
        result[f"I_{component}"] = float(raw[f"I_{component}"].sum())
    neutral_s, minus_s = (
        float(raw["neutral_loss"][:, 0].mean()),
        float(raw["minus_loss"][:, 0].mean()),
    )
    s_guard = minus_s <= neutral_s + 1e-9
    reduction = result["I_C"] >= 1e-9
    result.update(
        observed_response="C_decrease"
        if reduction
        else "C_increase"
        if result["I_C"] <= -1e-9
        else "inconclusive",
        S_nonincrease_guard=s_guard,
        component_tradeoff=not s_guard,
        joint_measured_reduction=reduction and result["I_S"] >= 1e-9,
        direction_qualification=reduction
        and s_guard
        and all(result[k]["passed"] for k in ("C_O_Q", "C_O_T", "S_O_Q", "S_O_T")),
        loss_order=list(e.LOSS_ORDER),
        arm_mean_losses={
            arm: raw[arm + "_loss"].mean(axis=0).tolist()
            for arm in ("neutral", "plus", "minus")
        },
        per_window={
            k: raw[k].tolist()
            for k in ("O_C", "Q_C", "T_C", "I_C", "O_S", "Q_S", "T_S", "I_S")
        },
    )
    return result


def collect(
    packet,
    *,
    device="cuda",
    dimensions=None,
    timely=lambda: None,
    progress=lambda value: None,
):
    import numpy as np
    import torch

    a = e.axes(dimensions)
    raw = {k: v.copy() for k, v in packet["reference"].items()}
    raw["boundary"] = np.stack(
        [packet[f"start-{c}"]["fixed_original_q_boundary"].copy() for c in WINDOWS]
    )
    receipts, record = {}, {"prohibited_old_calls": 0, **LIMITATIONS}
    stage = "assembly"

    def save(name, value):
        raw[name] = (
            value[None].copy()
            if name not in raw
            else np.concatenate((raw[name], value[None]), axis=0)
        )

    def emit():
        progress(
            {
                "raw_roster": sorted(raw),
                "receipts": copy.deepcopy(receipts),
                "stage": stage,
            }
        )

    try:
        timely()
        bundle = assemble(packet, device=device, dimensions=dimensions)
        record.update(bundle["record"])
        require(
            not any(
                n in sys.modules
                for n in (
                    "tools.verify_c_banc_m3_smooth_efficacy_direction",
                    "tools.verify_c_banc_m3_temporal_direction",
                )
            ),
            "old E333/E330 runtime import forbidden",
        )
        rollout, objective, target = (
            bundle["rollout"],
            bundle["objective"],
            bundle["rollout"].device,
        )
        tensor = lambda value: torch.as_tensor(value.copy(), device=target)
        mask = tensor(packet["output_mask"]["value"])

        def forward(c, arm, theta):
            nonlocal stage
            stage = f"{arm}-{c}-forward"
            key = f"window-{c}-{arm}"
            state = packet[f"start-{c}"]
            timely()
            try:
                with torch.no_grad():
                    output = rollout.run(
                        *(
                            tensor(state[k])
                            for k in (
                                "rate",
                                "adaptation",
                                "bank",
                                "original_activation",
                            )
                        ),
                        tensor(packet[f"drive-{c}"]["drive"]),
                        theta,
                        mask,
                    )
            except Exception as error:
                if hasattr(error, "receipt"):
                    receipts[key] = receipt_dict(error.receipt)
                raise
            receipt = output["receipt"]
            receipts[key] = receipt_dict(receipt)
            for name in ("q", "tendon"):
                save(name + "_" + arm, output[name].detach().cpu().numpy())
            if arm == "neutral":
                for name in ("rate", "adaptation", "bank", "original_activation"):
                    save("neutral_end_" + name, output[name].detach().cpu().numpy())
            emit()
            try:
                receipt.confirm()
            finally:
                receipts[key] = receipt_dict(receipt)
                emit()
            timely()
            return output, receipt, key

        outputs = []
        for index, c in enumerate(WINDOWS):
            output, receipt, key = forward(
                c,
                "neutral",
                torch.ones(a["groups"], dtype=torch.float64, device=target),
            )
            stage = f"neutral-{c}-reproduction"
            endpoint = packet[f"endpoint-{c}"]
            for name in ("q", "tendon", "bank", "original_activation"):
                require(
                    np.allclose(
                        output[name].detach().cpu().numpy(),
                        endpoint[name],
                        rtol=0,
                        atol=1e-11 if name in ("q", "tendon") else 1e-10,
                    ),
                    "neutral reproduction: " + name,
                )
            for name in ("rate", "adaptation"):
                require(
                    e.exact(output[name].detach().cpu().numpy(), endpoint[name]),
                    "neutral native bytes: " + name,
                )
            with torch.no_grad():
                terms = window_loss(
                    objective, tensor(raw["boundary"][index]), output["q"]
                )
            losses = loss_vector(terms)
            save("neutral_loss", losses)
            require(
                np.allclose(
                    losses[[2, 0, 1, 3]], endpoint["loss_terms"], rtol=1e-10, atol=1e-11
                ),
                "saved neutral loss order",
            )
            outputs.append((output, receipt, key))
        record["all_neutral_reproduced_before_direction"] = True
        raw["legacy_total_gtheta"] = np.stack(
            [packet[f"legacy_total_gradient-{c}"]["gradient"].copy() for c in WINDOWS]
        )
        raw["legacy_total_gq"] = np.stack(
            [packet[f"legacy_total_gq-{c}"]["gq"].copy() for c in WINDOWS]
        )
        raw.update({k: v.copy() for k, v in packet["audited_C_vjp"].items()})
        raw.update(
            reconstruct(
                raw["legacy_total_gtheta"],
                raw["legacy_total_gq"],
                raw["gtheta_C"],
                raw["gq_C"],
            )
        )
        for index, (output, receipt, key) in enumerate(outputs):
            stage = key + "-q-only"
            before = receipt_dict(receipt)
            q = output["q"].detach().clone().requires_grad_(True)
            terms = window_loss(objective, tensor(raw["boundary"][index]), q)
            cg = torch.autograd.grad(
                terms["coordination_loss"] / 4, q, retain_graph=True
            )[0]
            sg = torch.autograd.grad(terms["successor_loss"] / 4, q)[0]
            require(
                before == receipt_dict(receipt),
                "q-only derivative changed neural receipt",
            )
            save("q_only_gq_C", cg.detach().cpu().numpy())
            save("q_only_gq_S", sg.detach().cpu().numpy())
            require(
                np.allclose(
                    raw["q_only_gq_C"][-1], raw["gq_C"][index], rtol=1e-8, atol=2e-10
                )
                and np.allclose(
                    raw["q_only_gq_S"][-1],
                    raw["gq_S_reconstructed"][index],
                    rtol=1e-8,
                    atol=2e-10,
                ),
                "audited/reconstructed q-only derivative",
            )
            emit()
            timely()
        record["q_only_gradient_neural_work"] = 0
        geometry, info = common_direction(
            raw["sum_gtheta_C"], raw["sum_gtheta_S_proxy"]
        )
        raw.update(geometry)
        record.update(info)
        emit()
        if record["branch"] == "normal":
            record["palettes"] = {}
            for arm in ("plus", "minus"):
                raw["selected_weights_" + arm], record["palettes"][arm] = patch_weights(
                    packet["actual_csr"], packet["recurrent_mask"], raw["theta_" + arm]
                )
            emit()
            if not all(v["native_deployable"] for v in record["palettes"].values()):
                record["branch"] = "native_palette_not_deployable"
            else:
                for arm in ("plus", "minus"):
                    for index, c in enumerate(WINDOWS):
                        output, _, _ = forward(c, arm, tensor(raw["theta_" + arm]))
                        with torch.no_grad():
                            terms = window_loss(
                                objective, tensor(raw["boundary"][index]), output["q"]
                            )
                        save(arm + "_loss", loss_vector(terms))
                deltaq = raw["q_plus"] - raw["q_minus"]
                for component, column, gq, gt in (
                    ("C", 1, "gq_C", "gtheta_C"),
                    ("S", 0, "gq_S_reconstructed", "gtheta_S_proxy"),
                ):
                    with np.errstate(over="raise", invalid="raise"):
                        raw["O_" + component] = (
                            raw["plus_loss"][:, column] - raw["minus_loss"][:, column]
                        ) / 4
                        raw["Q_" + component] = (raw[gq] * deltaq).sum(axis=(1, 2))
                        raw["T_" + component] = (
                            raw[gt] * raw["actual_delta_theta"]
                        ).sum(axis=1)
                        raw["I_" + component] = (
                            raw["neutral_loss"][:, column]
                            - raw["minus_loss"][:, column]
                        ) / 4
        stage = "completed-raw"
        require(
            all(np.isfinite(v).all() for v in raw.values()),
            "finite common-direction raw",
        )
        expected = 48 if record["branch"] == "normal" else 16
        work = e.work_counts(receipts)
        require(
            work["forward_controls_requested"]
            == work["forward_controls_returned"]
            == expected
            and work["forward_neural_ticks_requested"]
            == work["forward_neural_ticks_returned"]
            == expected * a["steps_per_control"]
            and all(
                value == 0
                for key, value in work.items()
                if key.startswith(("recompute", "backward"))
            ),
            "returned no-back native work",
        )
        record.update(
            work=work,
            new_neural_backwards=0,
            recompute_controls=0,
            steps_per_control=a["steps_per_control"],
            control_dt=a["control_dt"],
        )
        emit()
        return raw, receipts, record
    except Exception as error:
        raise e.CollectionFailure(error, raw, receipts, record, stage) from error


class Clock:
    def __init__(self, clock=time.perf_counter, s0=None):
        self.clock = clock
        self.s0 = clock() if s0 is None else s0
        self.final_deadline = None

    def check(self, stage="science"):
        now = self.clock()
        require(
            type(now) in (int, float) and math.isfinite(now) and now >= self.s0,
            "monotone finite clock",
        )
        limit = self.s0 + (25 if stage == "science" else 30)
        if self.final_deadline is not None:
            limit = min(self.s0 + 35, self.final_deadline)
        if now > limit:
            raise TimeoutError("absolute E343 " + stage + " deadline")

    def finalize(self):
        self.check("internal")
        self.final_deadline = min(self.s0 + 35, self.clock() + 5)


class ProcessIdentityWatch:
    """Record creation times while own children exist, before later PID absence."""

    def __init__(self):
        self.create_times = {}
        self.errors = []
        self.stop = threading.Event()
        self.thread = None

    def capture(self, pid):
        import psutil

        try:
            value = psutil.Process(pid).create_time()
            require(
                math.isfinite(value) and value > 0, "positive process creation time"
            )
            self.create_times[str(pid)] = value
        except psutil.NoSuchProcess:
            pass

    def start(self):
        import psutil

        owner = psutil.Process()
        self.capture(owner.pid)
        parent = owner.parent()
        if parent is not None and any(
            Path(v).name == Path(sys.argv[0]).name for v in parent.cmdline()
        ):
            self.capture(parent.pid)

        def loop():
            while not self.stop.wait(0.02):
                try:
                    for process in owner.children(recursive=True):
                        self.capture(process.pid)
                except (psutil.Error, RuntimeError) as error:
                    self.errors.append(str(error)[:256])

        self.thread = threading.Thread(
            target=loop, name="E343-own-process-identities", daemon=True
        )
        self.thread.start()
        return self

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1)
        return {
            "pid_create_times": dict(self.create_times),
            "errors": list(self.errors),
            "owned_thread_alive": self.thread.is_alive() if self.thread else False,
        }


class GPUMemoryWatch(e.GPUMemoryWatch):
    def __init__(self, *, identity=None, **kwargs):
        self.identity = identity
        super().__init__(**kwargs)

    def _smi(self):
        process = subprocess.Popen(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self._pids.add(process.pid)
        if self.identity is not None:
            self.identity.capture(process.pid)
        try:
            stdout, stderr = process.communicate(timeout=2)
            require(
                process.returncode == 0
                and not stderr.strip()
                and stdout.decode("ascii").strip().isdigit(),
                "sampler integer MiB",
            )
            return int(stdout.strip()) * e.MIB
        except Exception:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)
            raise


def stage_guard(root, extra=0):
    files = []
    for item in e.safe(root, BASE).glob("e343-sol-first-*"):
        files.extend(item.rglob("*")) if item.is_dir() else files.append(item)
    total = sum(p.stat().st_size for p in files if p.is_file())
    require(
        total + extra + e.FAILURE_QUOTA <= 32 * e.MIB,
        "generated retained/staging/failure quota",
    )
    return total


def generated_write(root, name, payload):
    path = generated_path(root, name)
    require(not path.exists(), "fresh generated evidence")
    stage_guard(root, len(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
    require(path.read_bytes() == payload, "generated write readback")
    stage_guard(root)
    return path


def prepare_generated(
    root=ROOT,
    *,
    prefix,
    device="cuda",
    timely=lambda: None,
    progress=lambda value: None,
):
    """Four generated total/50 VJPs; separate preparation, not E333 whole50."""
    import io

    import numpy as np
    import torch

    require(
        Path(root).resolve() == ROOT and device == "cuda",
        "sealed generated CUDA preparation",
    )
    directory = generated_path(root, prefix)
    require(
        not directory.exists() and Path(prefix).name.startswith("e343-sol-first-"),
        "fresh own fixture prefix",
    )
    require(e.digest(e.safe(root, SEALED)) == SEALED_SHA, "sealed native metadata pin")
    seal = e.read_json(e.safe(root, SEALED))
    require(
        seal["completion_valid"] is True
        and seal["packet_descriptor"]["sha256"]
        == "b9380e44a77ec286883ca341502682b0bd76ba65171fb5f1d63bd68adccbbcdd",
        "approved sealed generated packet",
    )
    for name, binding in seal["fixture"]["inventory"]["bindings"].items():
        path = generated_path(root, binding["path"])
        require(
            e.digest(path, timely) == binding["sha256"],
            "sealed generated input bytes: " + name,
        )
    descriptor = seal["packet_descriptor"]
    require(
        e.digest(e.safe(root, descriptor["path"]), timely) == descriptor["sha256"],
        "sealed packet file bytes",
    )
    for name, entry in seal["publication_pins"].items():
        require(
            e.digest(generated_path(root, seal["publication"] + "/" + name), timely)
            == entry["sha256"],
            "sealed publication bytes",
        )
    wrapper = copy.deepcopy(seal["fixture"])
    dimensions = TestDimensions(**seal["dimensions"])
    ledger = []
    packet = e.load_packet(
        root,
        wrapper["inventory"],
        dimensions=dimensions,
        metadata=wrapper["metadata"],
        ledger=ledger,
        timely=timely,
    )
    bundle = assemble(packet, dimensions=dimensions, device=device)
    rollout, target = bundle["rollout"], bundle["rollout"].device
    tensor = lambda value: torch.as_tensor(value.copy(), device=target)
    inventory = wrapper["inventory"]
    preparation = {}

    def checkpoint(stage):
        progress(
            {
                "stage": stage,
                "receipts": copy.deepcopy(preparation),
                "new_selected_bindings": {
                    k: v
                    for k, v in inventory["bindings"].items()
                    if k.startswith("legacy_total_")
                },
            }
        )

    def save(role, arrays, name=None):
        stream = io.BytesIO()
        np.savez(stream, **arrays)
        name = prefix + "/" + role + ".npz" if name is None else name
        path = generated_write(root, name, stream.getvalue())
        inventory["bindings"][role] = {
            "path": e.relative(root, str(path)),
            "kind": "npz",
            "sha256": e.digest(path),
            "bytes": path.stat().st_size,
            "selected_schema": {
                key: {
                    "shape": list(value.shape),
                    "dtype": value.dtype.str,
                    "finite": True,
                }
                for key, value in arrays.items()
            },
        }
        inventory["role_order"].append(role)

    for c in WINDOWS:
        state = packet[f"start-{c}"]
        theta = torch.ones(
            dimensions.groups, dtype=torch.float64, device=target, requires_grad=True
        )
        timely()
        output = rollout.run(
            *(
                tensor(state[k])
                for k in ("rate", "adaptation", "bank", "original_activation")
            ),
            tensor(packet[f"drive-{c}"]["drive"]),
            theta,
            tensor(packet["output_mask"]["value"]),
        )
        receipt = output["receipt"]
        preparation[str(c)] = receipt_dict(receipt)
        checkpoint(f"{c}-forward-returned")
        try:
            receipt.confirm()
        finally:
            preparation[str(c)] = receipt_dict(receipt)
            checkpoint(f"{c}-forward-confirm")
        terms = window_loss(
            bundle["objective"], tensor(state["fixed_original_q_boundary"]), output["q"]
        )
        scaled = terms["training_loss"] / 50
        before = receipt_dict(receipt)
        gradient_q = torch.autograd.grad(scaled, output["q"], retain_graph=True)[0]
        require(
            before == receipt_dict(receipt), "preparation gq changed neural receipt"
        )
        try:
            scaled.backward()
        finally:
            preparation[str(c)] = receipt_dict(receipt)
            checkpoint(f"{c}-backward-returned")
        save(
            f"legacy_total_gradient-{c}",
            {"gradient": theta.grad.detach().cpu().numpy().copy()},
        )
        save(f"legacy_total_gq-{c}", {"gq": gradient_q.detach().cpu().numpy().copy()})
        try:
            receipt.confirm()
        finally:
            preparation[str(c)] = receipt_dict(receipt)
            checkpoint(f"{c}-backward-confirm")
        timely()
    path = seal["publication"] + "/raw.npz"
    inventory["bindings"]["audited_C_vjp"] = {
        "path": path,
        "kind": "npz",
        **seal["publication_pins"]["raw.npz"],
        "selected_schema": {
            "gtheta_C": {
                "shape": [4, dimensions.groups],
                "dtype": "<f8",
                "finite": True,
            },
            "sum_gtheta_C": {
                "shape": [dimensions.groups],
                "dtype": "<f8",
                "finite": True,
            },
            "gq_C": {"shape": [4, 4, 42], "dtype": "<f8", "finite": True},
        },
    }
    inventory["role_order"].append("audited_C_vjp")
    inventory.update(
        input_count=32, selected_value_count=123, npz_count=30, npy_count=2
    )
    wrapper.update(
        generated_prefix=prefix,
        fixture_preparation_receipts=preparation,
        preparation_window_count=4,
        legacy_objective_scaling_window_count=50,
        E333_whole50_completed_claimed=False,
        actual_import_sha256=bundle["record"]["actual_import_sha256"],
        sealed_generated_packet_sha256=descriptor["sha256"],
        **LIMITATIONS,
    )
    return wrapper, dimensions


def worker(
    packet_loader,
    *,
    root=ROOT,
    output=OUTPUT,
    dimensions=None,
    device="cuda",
    clock=time.perf_counter,
    fault=None,
    printer=e.stdout_bytes,
    watcher=None,
    meter=usage,
    expected_imports=None,
    s0=None,
):
    if dimensions is None:
        require(
            Path(root).resolve() == ROOT
            and output == OUTPUT
            and device == "cuda"
            and clock is time.perf_counter
            and fault is None
            and printer is e.stdout_bytes
            and meter is usage,
            "original lifecycle overrides forbidden",
        )
    timer = Clock(clock, s0)
    fault = (lambda stage: None) if fault is None else fault
    terminal = {
        "schema": "flylab.m3.e343.result.v1",
        "worker_valid": False,
        "mode": "generated" if dimensions is not None else "original",
        "device": device,
        "source_sha256": source_map(),
        "analysis": {
            "observed_response": "not_evaluated",
            "direction_qualification": False,
        },
        "first_error": None,
        "secondary_errors": [],
        "receipts": {},
        "work": {},
        "raw_roster": [],
        "effective_direction_qualification": False,
        "external_completion_pending": True,
        "independent_audit_pending": True,
        "body_calls": 0,
        "candidate_adopted": False,
        "F1_F8_complete": False,
        "biology_proven": False,
        **LIMITATIONS,
    }
    raw, ledger, record, store = {}, [], {}, None
    stage, closed_watch = "output", False

    def failure(error, at):
        entry = {
            "stage": at,
            "type": type(error).__name__,
            "message": str(error)[:1024],
        }
        if terminal["first_error"] is None:
            terminal["first_error"] = entry
        else:
            terminal["secondary_errors"].append(entry)
        terminal["worker_valid"] = False
        terminal["effective_direction_qualification"] = False

    def timely():
        timer.check()
        fault("deadline")

    def resources(record_snapshot=True):
        snapshot = meter(device)
        require(
            type(snapshot["tree_rss_bytes"]) is int
            and 0
            < snapshot["tree_rss_bytes"]
            <= (2 * e.GIB if dimensions else 6 * e.GIB),
            "actual host cap",
        )
        if record_snapshot:
            terminal["resources"] = snapshot
        if watcher is not None:
            sampled = watcher.snapshot()
            require(
                not sampled["sampler_errors"]
                and sampled["sample_count"] > 0
                and sampled["sampled_additional_peak_bytes"]
                + sampled["quantization_allowance_bytes"]
                <= 8 * e.GIB,
                "context-included sampled GPU cap",
            )
            if record_snapshot:
                terminal["gpu_memory_watch"] = sampled

    try:
        store = Store(e.safe(root, output), timer)
        stage = "precontext"
        if watcher is not None and watcher.snapshot()["sample_count"] == 0:
            watcher.start()
        stage = "selected-load"
        packet = packet_loader(ledger, timely)
        timely()
        stage = "collect"
        try:
            raw, receipts, record = collect(
                packet, device=device, dimensions=dimensions, timely=timely
            )
        except e.CollectionFailure as error:
            raw, receipts, record = error.raw, error.receipts, error.record
            terminal["receipts"] = receipts
            raise error.error from error
        terminal["receipts"] = receipts
        terminal["work"] = e.work_counts(receipts)
        terminal["assembly"] = record
        resources()
        stage = "raw-save"
        fault(stage)
        store.raw(raw)
        terminal["raw_roster"] = sorted(raw)
        stage = "classification"
        fault(stage)
        terminal["analysis"] = analyze(raw, record)
        if expected_imports is not None:
            require(
                all(
                    expected_imports.get(k) == v
                    for k, v in record["actual_import_sha256"].items()
                ),
                "current actual imported bytes",
            )
        stage = "posthash"
        for entry in ledger:
            require(
                e.digest(e.safe(root, entry["path"]), timely)
                == entry["expected_sha256"],
                "postcalculation selected bytes",
            )
        store.put(
            "selected-ledger.json",
            e.json_bytes(
                {"schema": "flylab.m3.e343.selected-ledger.v1", "inputs": ledger}
            ),
        )
        resources()
        timely()
        terminal["worker_valid"] = True
    except Exception as error:  # noqa: BLE001 - preserve acquired work on lifecycle failure.
        failure(error, stage)
    finally:
        if (
            raw
            and store is not None
            and not (store.output / "raw.npz").exists()
            and not (store.output / "raw.npz.partial").exists()
        ):
            try:
                store.raw(raw)
                terminal["raw_roster"] = sorted(raw)
            except Exception as error:  # noqa: BLE001 - preserve original error before secondary I/O.
                failure(error, "partial-raw-save")
        terminal["work"] = e.work_counts(terminal["receipts"])
        terminal["assembly"] = record
    try:
        stage = "finalization"
        timer.finalize()
        fault(stage)
        resources()
        terminal["elapsed_before_publication_s"] = clock() - timer.s0
        payload = e.json_bytes(terminal)
        require(len(payload) <= e.QUOTAS["stdout"], "actual stdout cap")
        store.put("result.json", payload)
        store.put("receipts.json", payload)
        stage = "manifest"
        manifest = {
            p.name: {"sha256": e.digest(p), "bytes": p.stat().st_size}
            for p in store.output.iterdir()
            if p.is_file()
        }
        store.put(
            "manifest.json",
            e.json_bytes(
                {"schema": "flylab.m3.e343.output-manifest.v1", "files": manifest}
            ),
        )
        for name, entry in manifest.items():
            require(
                e.digest(store.output / name) == entry["sha256"],
                "nonself manifest readback",
            )
        fault("readback")
        timer.check("final")
        resources(False)
        store.check()
        stage = "stdout"
        printer(payload)
        store.written.add("stdout")
        timer.check("final")
        fault("postpublication")
        resources(False)
        if watcher is not None:
            completed = watcher.close()
            closed_watch = True
            require(
                not completed["sampler_errors"]
                and not completed["owned_thread_alive"]
                and completed["sampled_additional_peak_bytes"]
                + completed["quantization_allowance_bytes"]
                <= 8 * e.GIB,
                "sampler own cleanup/resource",
            )
            store.put(
                "gpu-sampler-completion.json", e.json_bytes(completed), failure=True
            )
        timer.check("final")
        store.check()
    except Exception as error:  # noqa: BLE001 - late failures revoke provisional publication.
        failure(error, stage)
    finally:
        if watcher is not None and not closed_watch:
            try:
                terminal["sampler_cleanup"] = watcher.close()
                require(
                    not terminal["sampler_cleanup"]["sampler_errors"],
                    "sampler close failure",
                )
            except Exception as error:  # noqa: BLE001 - owned cleanup errors cannot escape receipts.
                failure(error, "sampler-close")
    if terminal["first_error"] is not None:
        terminal["analysis"]["direction_qualification"] = False
        terminal["analysis"]["observed_response"] = "not_evaluated"
        terminal["revoked"] = True
        try:
            value = e.json_bytes(terminal)
            require(len(value) <= e.FAILURE_QUOTA, "failure payload cap")
            store.check(len(value), failure=True)
            with (store.output / "revocation.json").open("xb") as stream:
                stream.write(value)
            require(
                (store.output / "revocation.json").read_bytes() == value,
                "revocation readback",
            )
        except Exception as error:  # noqa: BLE001 - emergency failure is retained as secondary.
            terminal["secondary_errors"].append(
                {
                    "stage": "emergency-publication",
                    "type": type(error).__name__,
                    "message": str(error)[:1024],
                }
            )
        return terminal, 1
    return terminal, 0


def run_generated(
    root, wrapper, *, dimensions, output="diagnosis", device="cpu", **options
):
    require(
        isinstance(dimensions, TestDimensions) and device in ("cpu", "cuda"),
        "explicit generated native axes",
    )
    target = generated_path(root, wrapper["generated_prefix"] + "/" + output)
    require(not target.exists(), "fresh generated output")
    return worker(
        lambda ledger, timely: load_packet(
            root,
            wrapper["inventory"],
            dimensions=dimensions,
            metadata=wrapper["metadata"],
            ledger=ledger,
            timely=timely,
        ),
        root=root,
        output=e.relative(root, str(target)),
        dimensions=dimensions,
        device=device,
        **options,
    )


def validate_saved_metadata(
    inventory,
    base_inventory,
    e333_result,
    e333_receipts,
    e333_manifest,
    e333_audit,
    e333_decision,
    e342_result,
    e342_decision,
    e342_audit,
    e342_manifest,
):
    require(
        inventory["input_count"] == 32
        and inventory["selected_value_count"] == 123
        and inventory["npz_count"] == 30
        and inventory["npy_count"] == 2,
        "literal selected counts",
    )
    first = inventory["role_order"][:23]
    require(
        first == base_inventory["role_order"]
        and all(
            inventory["bindings"][k] == base_inventory["bindings"][k] for k in first
        ),
        "same approved first23 packet",
    )
    for key in (
        "model_identity",
        "decoder_hash",
        "decoder_neutral",
        "decoder_original_npy_links",
        "windows",
    ):
        require(inventory[key] == base_inventory[key], "same original packet metadata")
    require(
        e333_audit.get("passed") is True
        and e333_audit.get("run_valid") is False
        and isinstance(e333_audit.get("checks"), dict)
        and len(e333_audit["checks"]) == 3414
        and all(v is True for v in e333_audit["checks"].values()),
        "closed E333 audit not whole recovery",
    )
    require(
        e333_decision.get("formal_verdict") == "invalid_deadline"
        and e333_decision.get("run_valid") is False
        and e333_decision.get("effective_direction_qualification") is False
        and e333_decision.get("effective_continuation_qualification") is False
        and e333_result.get("run_valid") is False,
        "E333 whole remains invalid",
    )
    partial = e333_result.get("partial", {})
    require(
        partial.get("window_starts") == list(range(100, 300, 4))
        and partial.get("window_controls") == 4
        and partial.get("steps_per_control") == 20,
        "actual E333 local50 semantics",
    )
    for c in WINDOWS:
        run = partial["runs"][f"window-{c}-neutral"]
        receipt = e333_receipts["receipts"][f"window-{c}-neutral"]
        expected = {
            "forward_controls": 4,
            "forward_neural_ticks": 80,
            "recompute_controls": 4,
            "recompute_neural_ticks": 80,
            "backward_blocks": 4,
            "backward_block_ticks": 80,
        }
        require(
            receipt.get("status") == "backward_returned"
            and receipt.get("confirm_returned") == 2
            and receipt.get("confirmed") == expected
            and all(
                receipt.get(key + "_returned") == value
                for key, value in expected.items()
            )
            and all(
                receipt.get(key + "_requested") == value
                for key, value in expected.items()
                if key != "backward_block_ticks"
            ),
            "closed individual neutral work",
        )
        require(
            run.get("start_control") == c
            and run.get("label") == "neutral"
            and run.get("theta_backward_calls") == 1
            and run.get("gq_neural_receipt_unchanged") is True
            and run.get("gq_receipt_before") == run.get("gq_receipt_after")
            and run.get("receipt") == e333_receipts["receipts"][f"window-{c}-neutral"],
            "individual saved neutral returned VJP",
        )
        reproduction = run.get("neutral_reproduction", {})
        require(
            reproduction.get("end_rate_adaptation_dtype_bytes_exact") is True
            and 0 <= reproduction["q_max_abs_rad"] <= 1e-11
            and 0 <= reproduction["tendon_max_abs_command"] <= 1e-11
            and 0 <= reproduction["bank_max_abs"] <= 1e-10
            and 0 <= reproduction["original_max_abs"] <= 1e-10,
            "actual saved neutral reproduction",
        )
        for role, field in (
            (f"legacy_total_gradient-{c}", "gradient_file"),
            (f"legacy_total_gq-{c}", "gq_file"),
        ):
            binding = inventory["bindings"][role]
            require(
                Path(binding["path"]).name == run[field]
                and binding["window_start"] == c
                and binding["objective_scaling_window_count"] == 50
                and e333_manifest["files"][run[field]]["sha256"] == binding["sha256"]
                and e333_manifest["files"][run[field]]["bytes"] == binding["bytes"],
                "selected legacy producer file",
            )
    require(
        e342_result.get("worker_valid") is True
        and e342_decision.get("run_valid") is True
        and e342_decision.get("independent_audit_passed") is True
        and e342_decision.get("independent_audit_checks") == 910
        and e342_decision.get("direction_learning_adopted") is False
        and e342_audit.get("passed") is True
        and e342_audit.get("check_count") == 910
        and e342_audit.get("first_error") is None
        and e342_audit.get("source_sha256", {}).get(E342_SOURCE) == E342_SHA
        and e342_decision.get("source_sha256")
        == e342_audit.get("source_sha256")
        == e342_result.get("source_sha256"),
        "audited E342 source and rejection",
    )
    binding = inventory["bindings"]["audited_C_vjp"]
    require(
        e342_manifest["files"]["raw.npz"] == e342_decision["output_pins"]["raw.npz"]
        and e342_manifest["files"]["raw.npz"]["sha256"] == binding["sha256"]
        and e342_manifest["files"]["raw.npz"]["bytes"] == binding["bytes"],
        "audited C raw identity",
    )
    require(
        e342_result["assembly"].get("all_neutral_reproduced_before_gradient") is True
        and e342_result["assembly"].get("neural_backwards") == 4
        and e342_result["assembly"].get("steps_per_control") == 20
        and e342_result["assembly"].get("control_dt") == 0.005,
        "same audited C local scale/clock",
    )


def closed_execution(
    root, branch, evidence, current, *, expected_exit=0, timely=lambda: None
):
    report = e.closed_execution(
        root, branch, evidence, current, expected_exit=expected_exit, timely=timely
    )
    contract = e.read_json(e.descriptor(root, branch["contract"], evidence, timely))
    require(contract["commands"][0]["wall_s"] <= 60, "E343 generated stage wall")
    pids = e.read_json(e.descriptor(root, branch["pids"], evidence, timely))
    identities = pids.get("pid_create_times", {})
    require(
        all(
            type(identities.get(str(pid))) in (int, float)
            and math.isfinite(identities[str(pid)])
            and identities[str(pid)] > 0
            for pid in pids["owned_pids"]
        ),
        "owned creation-time evidence",
    )
    return report


def validate_proof(root, proof, current, timely=lambda: None):
    require(
        proof.get("schema") == "flylab.m3.e343.current-generated-proof.v1"
        and proof.get("passed") is True
        and proof.get("source_sha256") == current
        and type(proof.get("original_calls")) is int
        and proof["original_calls"] == 0,
        "current E343 proof",
    )
    evidence = proof.get("evidence_sha256", {})
    require(
        0 < len(evidence) <= 256
        and set(proof.get("branches", {})) == {"sol", "root", "native_cuda"},
        "finite three-branch proof",
    )
    normalized = {}
    e.merge(normalized, evidence, root)
    require(len(normalized) == len(evidence), "proof path aliases")
    for name, pin in normalized.items():
        if not name.endswith((".py", ".json", ".md", ".txt")):
            generated_path(root, name)
        require(
            e.safe(root, name).stat().st_size <= 8 * e.MIB
            and e.digest(e.safe(root, name), timely) == pin,
            "finite proof bytes",
        )
    reports = {
        key: closed_execution(root, branch, evidence, current, timely=timely)
        for key, branch in proof["branches"].items()
    }
    native = reports["native_cuda"]
    require(
        native.get("device") == "cuda"
        and native.get("steps_per_control") == 20
        and native.get("prohibited_old_calls") == 0
        and native.get("current_new_neural_backwards") == 0,
        "actual generated CUDA no-back producer",
    )
    imports = native.get("actual_import_sha256", {})
    require(
        imports
        and all(evidence.get(k) == v for k, v in imports.items())
        and imports.get(E342_SOURCE) == E342_SHA
        and all(
            imports.get(k) == v for k, v in current.items() if k.startswith("tools/")
        ),
        "current actual imported bytes",
    )
    audit = proof["same_audit_cli"]
    executable = e.descriptor(root, audit["source"], evidence, timely)
    qualification = e.read_json(
        e.descriptor(root, audit["qualification"], evidence, timely)
    )
    require(
        qualification.get("schema") == "flylab.m3.e343.audit-cli-qualification.v1"
        and qualification.get("passed") is True
        and qualification.get("source_sha256") == current,
        "same qualified audit executable",
    )
    cases = qualification.get("cases", {})
    require(
        set(cases)
        == {
            "valid",
            "wrong_scale",
            "wrong_row",
            "wrong_boundary",
            "wrong_gradient",
            "wrong_actualtheta",
            "wrong_result",
        },
        "seven actual audit CLI cases",
    )
    for name, branch in cases.items():
        report = closed_execution(
            root,
            branch,
            evidence,
            current,
            expected_exit=0 if name == "valid" else 1,
            timely=timely,
        )
        require(
            e.safe(root, report["audit_source_path"]) == executable,
            "same actual audit source",
        )
    return imports


def make_freeze(*, generated_proof):
    for name, pin in (
        (DESIGN, DESIGN_SHA),
        (REGISTRATION, REGISTRATION_SHA),
        (INVENTORY, INVENTORY_SHA),
        (E342_SOURCE, E342_SHA),
    ):
        require(e.digest(e.safe(ROOT, name)) == pin, "reflected fixed bytes")
    registration = e.read_json(e.safe(ROOT, REGISTRATION))
    current = source_map()
    imports = validate_proof(ROOT, e.read_json(e.safe(ROOT, generated_proof)), current)
    sources, metadata = {}, {}
    e.merge(sources, registration["direct_source_sha256"], ROOT)
    e.merge(sources, current, ROOT)
    e.merge(sources, imports, ROOT)
    for name, entry in registration["direct_metadata_pins"].items():
        e.merge(metadata, {name: entry["sha256"]}, ROOT)
    inventory = e.read_json(e.safe(ROOT, INVENTORY))
    e.merge(metadata, producer_pins(registration, inventory), ROOT)
    for name in (REGISTRATION, str(generated_proof)):
        e.merge(metadata, {name: e.digest(e.safe(ROOT, name))}, ROOT)
    for name, pin in {**sources, **metadata}.items():
        require(
            name.endswith((".py", ".json", ".md"))
            and e.digest(e.safe(ROOT, name)) == pin,
            "current direct text closure",
        )
    return {
        "schema": "flylab.m3.e343.freeze.v1",
        "fixed": copy.deepcopy(FIXED),
        "source_sha256": sources,
        "metadata_sha256": metadata,
        "actual_import_sha256": imports,
        "generated_proof": e.relative(ROOT, generated_proof),
        "inventory": inventory,
        "output": OUTPUT,
        "original_retry": 0,
    }


def producer_pins(registration, inventory):
    """Selected producer JSON pins inherited from closed freezes, never repinned."""
    older = []
    for name, group in (
        (
            BASE + "/e342-coordination-parameter-direction-freeze.json",
            "metadata_sha256",
        ),
        (BASE + "/e333-smooth-efficacy-direction-freeze.json", "input_sha256"),
    ):
        require(
            e.digest(e.safe(ROOT, name))
            == registration["direct_metadata_pins"][name]["sha256"],
            "prior freeze producer bytes",
        )
        older.append(e.read_json(e.safe(ROOT, name))[group])
    pins = {}
    direct = {
        name: entry["sha256"]
        for name, entry in registration["direct_metadata_pins"].items()
    }
    for binding in inventory["bindings"].values():
        name = binding["producer"]
        candidates = [mapping[name] for mapping in (direct, *older) if name in mapping]
        require(
            name.endswith(".json") and candidates and len(set(candidates)) == 1,
            "selected producer existing pin",
        )
        e.merge(pins, {name: candidates[0]}, ROOT)
    return pins


def validate_prior_completion(final, worker_marker, audit_marker, registration):
    for marker, kind in ((worker_marker, "worker"), (audit_marker, "audit")):
        require(
            marker.get("completion_valid") is True
            and marker.get("kind") == kind
            and type(marker.get("original_calls")) is int
            and marker["original_calls"] == 1
            and marker.get("first_error") is None,
            "closed prior external completion",
        )
        for key in ("contract", "supervision", "pids"):
            descriptor = marker[key]
            require(
                final["evidence"][descriptor["path"]]
                == {"sha256": descriptor["sha256"], "bytes": descriptor["bytes"]},
                "prior completed producer edge",
            )
    require(
        worker_marker.get("worker_valid") is True
        and worker_marker.get("source_sha256") == final["source_sha256"]
        and worker_marker.get("output_pins") == final["output_pins"],
        "prior worker output/source identity",
    )
    report = audit_marker["report_pin"]
    name = BASE + "/parent-e342-audit-report.json"
    require(
        report == final["evidence"][name]
        and report["sha256"] == registration["direct_metadata_pins"][name]["sha256"]
        and audit_marker.get("source_sha256") == final["source_sha256"]
        and audit_marker.get("check_count") == 910,
        "prior audit report edge",
    )


def gate(freeze_sha256, review_sha256, *, timely=lambda: None):
    require(
        e.digest(e.safe(ROOT, FREEZE), timely) == e.check_sha(freeze_sha256)
        and e.digest(e.safe(ROOT, REVIEW), timely) == e.check_sha(review_sha256),
        "literal freeze/review bytes",
    )
    freeze, review = (
        e.read_json(e.safe(ROOT, FREEZE)),
        e.read_json(e.safe(ROOT, REVIEW)),
    )
    current = source_map()
    require(
        freeze.get("schema") == "flylab.m3.e343.freeze.v1"
        and freeze.get("fixed") == FIXED
        and freeze.get("output") == OUTPUT
        and freeze.get("original_retry") == 0
        and review.get("approved") is True
        and review.get("blocking_defects") == []
        and review.get("source_sha256") == current
        and review.get("freeze_sha256") == freeze_sha256,
        "current literal ROOT source review",
    )
    registration = e.read_json(e.safe(ROOT, REGISTRATION))
    for group in ("source_sha256", "metadata_sha256"):
        mapping = freeze[group]
        normalized = {}
        e.merge(normalized, mapping, ROOT)
        require(len(normalized) == len(mapping), "frozen aliases")
        for name, pin in mapping.items():
            require(
                name.endswith((".py", ".json", ".md"))
                and e.digest(e.safe(ROOT, name), timely) == pin,
                "frozen direct text bytes",
            )
    for name, pin in {**registration["direct_source_sha256"], **current}.items():
        require(freeze["source_sha256"].get(name) == pin, "no source repinning")
    for name, entry in registration["direct_metadata_pins"].items():
        require(
            freeze["metadata_sha256"].get(name) == entry["sha256"],
            "no producer metadata repinning",
        )
    require(
        freeze["metadata_sha256"].get(REGISTRATION) == REGISTRATION_SHA
        and freeze["metadata_sha256"].get(DESIGN) == DESIGN_SHA
        and freeze["metadata_sha256"].get(INVENTORY) == INVENTORY_SHA,
        "reflected closure",
    )
    inventory = e.read_json(e.safe(ROOT, INVENTORY))
    require(freeze["inventory"] == inventory, "exact original inventory")
    for name, pin in producer_pins(registration, inventory).items():
        require(
            freeze["metadata_sha256"].get(name) == pin,
            "selected inherited producer pin",
        )
    imports = validate_proof(
        ROOT, e.read_json(e.safe(ROOT, freeze["generated_proof"])), current, timely
    )
    require(
        imports == freeze["actual_import_sha256"]
        and all(freeze["source_sha256"].get(k) == v for k, v in imports.items()),
        "current import closure",
    )
    get = lambda name: e.read_json(e.safe(ROOT, BASE + "/" + name))
    validate_prior_completion(
        get("parent-e342-final-decision.json"),
        get("parent-e342-original-external-completion.json"),
        get("parent-e342-audit-external-completion.json"),
        registration,
    )
    validate_saved_metadata(
        inventory,
        get("parent-e342-selected-input-inventory.json"),
        get("e333-smooth-efficacy-direction/result.json"),
        get("e333-smooth-efficacy-direction/receipts.json"),
        get("e333-smooth-efficacy-direction/output-manifest.json"),
        get("parent-e333-original-raw-audit.json"),
        get("parent-e333-final-decision.json"),
        get("e342-coordination-parameter-direction/result.json"),
        get("parent-e342-final-decision.json"),
        get("parent-e342-audit-report.json"),
        get("e342-coordination-parameter-direction/manifest.json"),
    )
    for role, binding in inventory["bindings"].items():
        require(
            type(binding["bytes"]) is int
            and binding["bytes"] > 0
            and e.safe(ROOT, binding["path"]).stat().st_size == binding["bytes"],
            "selected input stat",
        )
        producer = e.read_json(e.safe(ROOT, binding["producer"]))
        local = (
            e.safe(ROOT, binding["path"])
            .relative_to(e.safe(ROOT, binding["producer"]).parent)
            .as_posix()
        )
        entry = producer.get("files", {}).get(local)
        if isinstance(entry, dict):
            require(
                entry == {"sha256": binding["sha256"], "bytes": binding["bytes"]},
                "selected exact producer descriptor",
            )
        elif isinstance(entry, str):
            require(entry == binding["sha256"], "selected NPY producer pin")
        else:
            require(
                role in inventory["role_order"][:8], "missing selected producer pin"
            )
    require(not e.safe(ROOT, OUTPUT).exists(), "fresh literal output")
    return freeze


def run(freeze_sha256, review_sha256):
    timer = Clock(s0=_IMPORT_S0)
    freeze = gate(freeze_sha256, review_sha256, timely=lambda: timer.check())
    return worker(
        lambda ledger, timely: load_packet(
            ROOT, freeze["inventory"], ledger=ledger, timely=timely
        ),
        watcher=GPUMemoryWatch(),
        expected_imports=freeze["actual_import_sha256"],
        s0=_IMPORT_S0,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review-sha256", required=True)
    arguments = parser.parse_args(argv)
    return run(arguments.freeze_sha256, arguments.review_sha256)[1]


if __name__ == "__main__":
    raise SystemExit(main())
