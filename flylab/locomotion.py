"""Allocation-reduced execution of the pinned FlyGym hybrid controller.

The original CPG integrator and reflex update methods remain authoritative.
Invariant lookups are cached; the optional native spline batch evaluates the
original SciPy coefficients in the same arithmetic order without fast math.
"""
import numpy as np


class CachedHybridStepper:
    def __init__(self, controller, *, compiled_splines=True):
        from flygym_demo.complex_terrain.common import dof_spec_to_jointdof, LocomotionAction
        from flygym_demo.complex_terrain.hybrid_controller import (
            _CORRECTION_VECTORS, _RIGHT_LEG_CORRECTION_SIGN)
        self.controller = controller
        self.Action = LocomotionAction
        self.legs = controller.legs
        steps = controller.preprogrammed_steps
        source = [dof_spec_to_jointdof(leg, spec) for leg in self.legs for spec in steps.dofs_per_leg]
        self.order = np.asarray([source.index(dof) for dof in controller.output_dof_order])
        self.vectors = np.array([_CORRECTION_VECTORS[leg[1]] *
            (_RIGHT_LEG_CORRECTION_SIGN if leg.startswith('r') else 1) for leg in self.legs])
        self.increment_values = np.array([0., .8, 0., -.1, 0.])
        self._knots_key = None
        self._refresh_knots()
        self.splines = None
        if compiled_splines:
            try:
                from ._spline import BatchedSplines
            except ImportError:
                pass  # Cached upstream evaluation still works without optional JIT.
            else:
                self.splines = BatchedSplines(steps, self.legs)

    def _refresh_knots(self):
        c = self.controller
        key = (c.swing_extension, tuple(tuple(c.preprogrammed_steps.swing_period[leg]) for leg in self.legs))
        if key != self._knots_key:
            self.knots = []
            for start, end in key[1]:
                self.knots.append(np.array([start, np.mean([start, end]), end+c.swing_extension,
                                            np.mean([end, 2*np.pi]), 2*np.pi]))
            self._knots_key = key

    def step(self, descending, obs):
        c = self.controller
        self._refresh_knots()
        c.cpg_network.intrinsic_amps = np.repeat(np.abs(descending[:, np.newaxis]), 3, axis=1).ravel()
        frequencies = c._base_intrinsic_freqs.copy()
        frequencies[:3] *= 1 if descending[0] >= 0 else -1
        frequencies[3:] *= 1 if descending[1] >= 0 else -1
        c.cpg_network.intrinsic_freqs = frequencies
        leg_to_correct = c._select_retraction_leg(obs)
        if leg_to_correct is not None and c.retraction_correction[leg_to_correct] > c.retraction_persistence_initiation_threshold:
            c.retraction_persistence_counter[leg_to_correct] = 1
        c._update_persistence_counter()
        stumbling = c._get_stumbling_mask(obs)
        c.cpg_network.step()
        spline_values=self.splines.evaluate(c.cpg_network.curr_phases) if self.splines else None
        angles = np.empty((6, 7))
        adhesion = np.empty(6, dtype=bool)
        corrections = np.zeros(6)
        for i, leg in enumerate(self.legs):
            c._update_retraction_correction(i, leg_to_correct)
            c._update_stumbling_correction(i, stumbling[i])
            if c.retraction_correction[i] > 0:
                correction = c.retraction_correction[i]
                c.stumbling_correction[i] = 0
            else:
                correction = c.stumbling_correction[i]
            phase = c.cpg_network.curr_phases[i]
            magnitude = c.cpg_network.curr_magnitudes[i]
            if spline_values is None:
                leg_angles = c.preprogrammed_steps.get_joint_angles(leg, phase, magnitude)
            else:
                neutral=c.preprogrammed_steps.neutral_pos[leg][:,0]
                leg_angles=neutral+magnitude*(spline_values[i]-neutral)
            correction = np.clip(correction, 0, c.max_correction)
            gain = float(np.interp(phase % (2*np.pi), self.knots[i], self.increment_values))
            angles[i] = leg_angles + correction*gain*self.vectors[i]
            corrections[i] = correction*gain
            adhesion[i] = c._get_adhesion_onoff(leg, phase) if c.enable_adhesion else False
        c.last_info = dict(net_corrections=corrections.copy(),
            retraction_correction=c.retraction_correction.copy(), stumbling_correction=c.stumbling_correction.copy(),
            stumbling_mask=stumbling.copy(), leg_to_correct_retraction=leg_to_correct)
        return self.Action(joint_angles=angles.ravel()[self.order], adhesion_onoff=adhesion)
