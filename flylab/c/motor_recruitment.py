"""Inspectable rate-to-MTU calibration hypotheses, with no runtime pose policy."""

import math

import numpy as np

from .integrity import digest, finite
from .muscle_calibration import direction_status, interior_angle
from .muscles import MUSCLES, moment_matrix


class TorqueBalancedRecruitment:
    """Match active torque at one fixed reference by reducing the stronger MTU.

    This engineering calibration changes no upstream muscle parameters and
    never observes the moving joint. It is not a measured firing/force curve.
    """

    def __init__(self, rig, *, neutral_rad=math.pi / 2, rate_half=100.0):
        neutral_rad = finite(
            neutral_rad, "recruitment reference interior angle", 0, math.pi
        )
        self.rate_half = finite(rate_half, "recruitment half-rate", 1, 1000)
        mj, model = rig.mj, rig.model
        candidate = mj.MjData(model)
        low, high = map(float, model.jnt_range[0])

        def angle_at(q):
            candidate.qpos[0] = q
            mj.mj_forward(model, candidate)
            return interior_angle(model, candidate)

        if not angle_at(high) < neutral_rad < angle_at(low):
            raise ValueError(
                "Reference angle is outside the decreasing interior-angle range"
            )
        for _ in range(60):
            midpoint = (low + high) / 2
            if angle_at(midpoint) > neutral_rad:
                low = midpoint
            else:
                high = midpoint
        reference_q = (low + high) / 2
        measured = angle_at(reference_q)
        arms = moment_matrix(mj, model, candidate)[rig.act_ids, 0]
        if direction_status({"moment_arm_mm": arms})["status"] != "CONSISTENT":
            raise ValueError(
                "Calibration reference has singular/reversed muscle geometry"
            )
        gains = np.array(
            [
                mj.mju_muscleGain(
                    candidate.actuator_length[i],
                    0.0,
                    model.actuator_lengthrange[i],
                    model.actuator_acc0[i],
                    model.actuator_gainprm[i, :9],
                )
                for i in rig.act_ids
            ]
        )
        coefficients = arms * gains
        if (
            not np.isfinite(coefficients).all()
            or not coefficients[0] > 1e-12
            or not coefficients[1] < -1e-12
            or abs(measured - neutral_rad) > 1e-10
        ):
            raise ValueError("Invalid reference active-torque coefficients")
        strength = np.abs(coefficients)
        self.scale = strength.min() / strength
        self.scale.flags.writeable = False
        self.metadata = {
            "schema": "flylab.torque-balanced-recruitment.v1",
            "body_hash": rig.identity,
            "muscles": list(MUSCLES),
            "reference_interior_angle_rad": measured,
            "reference_q_rad": reference_q,
            "reference_velocity_rad_s": 0.0,
            "torque_per_activation": coefficients.tolist(),
            "activation_scale": self.scale.tolist(),
            "rate_half": self.rate_half,
            "rule": "fixed activation_scale * rate / (rate_half + rate)",
            "runtime_pose_input": False,
            "upstream_muscle_parameters_changed": False,
            "calibration": "engineering torque balance; physiological recruitment unmeasured",
            "unit_contract": rig.metadata["unit_contract"],
            "biological_validation": False,
        }
        self.identity = digest(self.metadata)

    def encode(self, rates):
        if any(
            isinstance(value, (bool, np.bool_))
            for value in np.asarray(rates, dtype=object).flat
        ):
            raise ValueError("Boolean values are not motor rates")
        rates = np.asarray(rates)
        if (
            rates.shape != (2,)
            or rates.dtype.kind not in "fiu"
            or not np.isfinite(rates).all()
            or np.any(rates < 0)
        ):
            raise ValueError("Two finite nonnegative motor rates required")
        rates = rates.astype(np.float64)
        return self.scale * (rates / self.rate_half) / (1.0 + rates / self.rate_half)
