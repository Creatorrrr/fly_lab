import unittest

import numpy as np
import torch

from tools.collect_c_banc_m3_motor_teacher import (
    ContinuousMotorTeacher,
    MotorTeacher,
    held_motor_state,
)
from tools.identify_c_banc_m3_dynamic_targets import (
    continuous_neural_filter,
    inverse_operator,
)
from tools.probe_c_banc_m3_boundary import rk4
from tools.train_c_banc_m3_motor_commands import MotorCommandNetwork
from tools.train_c_banc_m3_recurrent import CalibrationNetwork
from tools.verify_c_banc_m3_full_trial import calibrated_weights


def network(recurrent=True):
    return CalibrationNetwork(
        np.array([[.2,-.1],[.4,0]],np.float32),np.array([0,1]),
        np.array([[0,1],[2,0]]),np.array([[1,0]],np.float32),
        np.array([[1]],np.float32),np.array([.3],np.float32),
        np.array([True,False]),recurrent=recurrent)


class CalibrationTests(unittest.TestCase):
    def test_recruitment_cut_clears_retained_joint_and_tendon_output(self):
        from types import SimpleNamespace

        from tools.banc_m3_rhythmic_recruitment import AnatomicalRecruitmentAdapter

        original = SimpleNamespace(identity='test-source', motor_ids=np.arange(391),
            body=SimpleNamespace(neutral=np.linspace(-.2, .2, 42)))
        columns = np.array_split(np.arange(391), 96)
        pool = np.zeros((96, 391))
        for i, ids in enumerate(columns):
            pool[i, ids] = 1 / len(ids)
        groups = [{'leg': leg, 'muscle': str(k)}
            for leg in ('lf', 'lm', 'lh', 'rf', 'rm', 'rh') for k in range(16)]
        calibration = {'moment': np.ones((42, 96)) / 96,
            'rest': original.body.neutral + .1, 'pool': pool,
            'motor_ids': original.motor_ids, 'ltm': np.arange(15, 96, 16)}
        adapter = AnatomicalRecruitmentAdapter(original, calibration, groups)
        active, _ = adapter.decode(np.full(391, 80))
        self.assertTrue(np.all(active > original.body.neutral))
        self.assertTrue(all(v > 0 for v in adapter.tendon_inputs.values()))
        cut, adhesion = adapter.decode(np.full(391, 200), motor_cut=True)
        np.testing.assert_array_equal(cut, original.body.neutral)
        self.assertFalse(adhesion.any())
        self.assertFalse(adapter.activation.any())
        self.assertTrue(all(v == 0 for v in adapter.tendon_inputs.values()))
        # A steady-state cancellation must not conceal an inadmissible
        # zero-activation starting command at the physical actuator boundary.
        calibration['rest'][8] = 15.99
        with self.assertRaises(ValueError):
            AnatomicalRecruitmentAdapter(original, calibration, groups)

    def test_periodic_recruitment_filter_matches_cyclic_recurrence_and_gradient(self):
        from tools.fit_c_banc_m3_rhythmic_recruitment import periodic_filter

        values = torch.tensor(np.random.default_rng(43).normal(size=(30, 3)),
            dtype=torch.float64, requires_grad=True)
        beta = np.exp(-.005 / .01)
        alpha = 1 - beta
        # Solve the last state from the recurrence's independent geometric sum,
        # then walk forward once; no FFT appears in this reference.
        previous = alpha * sum(beta ** (29 - k) * values[k] for k in range(30)) / (1 - beta ** 30)
        rows = []
        for row in values:
            previous = beta * previous + alpha * row
            rows.append(previous)
        expected = torch.stack(rows)
        actual = periodic_filter(values, .005, .01)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        expected_grad, = torch.autograd.grad(expected.square().sum(), values, retain_graph=True)
        actual_grad, = torch.autograd.grad(actual.square().sum(), values)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)

    def test_core_adaptation_uses_only_selected_cell_gain_and_matches_equal_gain(self):
        from scipy.sparse import csr_matrix

        from flylab.c.adaptive_rate import AdaptiveRateNetwork
        from tools.banc_m3_core_adaptation import CoreAdaptationNetwork

        args = (csr_matrix(np.array([[.2, -.1, .3], [.4, 0., -.2], [-.5, .3, .1]], np.float32)),
            np.full(3, .02), np.ones(3), np.full(3, 7.5), np.full(3, 200))
        kw = {'adaptation_gain': 8, 'adaptation_tau_s': .3, 'dt': .00025, 'device': 'cpu'}
        base = AdaptiveRateNetwork(*args, **kw)
        same = CoreAdaptationNetwork(*args, core_ids=[1], core_gain=8, **kw)
        changed = CoreAdaptationNetwork(*args, core_ids=[1], core_gain=0, **kw)
        r = torch.tensor([11., 23., 7.]); z = torch.tensor([2., 3., 4.])
        dr, dz = changed._derivative_pair(r, z)
        expected = (torch.tensor([8., 0., 8.]) * r - z) / .3
        torch.testing.assert_close(dz, expected, rtol=0, atol=0)
        torch.testing.assert_close(dr, base._derivative_pair(r, z)[0], rtol=0, atol=0)
        drive = np.array([40., 50., 60.], np.float32)
        base.advance(drive, 20); same.advance(drive, 20)
        np.testing.assert_array_equal(base.snapshot()['rate'], same.snapshot()['rate'])
        np.testing.assert_array_equal(base.snapshot()['adaptation'], same.snapshot()['adaptation'])
        changed.advance(drive, 20)
        self.assertEqual(float(changed.adaptation[1]), 0)
        self.assertTrue((changed.adaptation[[0, 2]] > 0).all())
        saved = changed.snapshot()
        changed.advance(drive, 20)
        future = changed.snapshot()
        changed.restore(saved); changed.advance(drive, 20)
        from tools.verify_c_banc_observation import exact
        exact(future, changed.snapshot())
        with self.assertRaises(ValueError):
            same.restore(saved)

    def test_four_knee_delivery_keeps_other_actuators_and_cut_disconnects(self):
        from types import SimpleNamespace

        from flylab.c.rate_body import RateBodyAdapter
        from tests.test_c_rate_body import fixture
        from tools.analyze_c_banc_biological_gait import LEGS
        from tools.probe_c_banc_m3_knee_delivery import SELECTED
        from tools.verify_c_banc_m3_knee_trial import KneeAdapter

        graph, body, _ = fixture()
        body.tendon_control = SimpleNamespace(index={leg + '_tarsus': i for i, leg in enumerate(LEGS)},
            limits=np.tile([-1., 1.], (6, 1)))
        original = RateBodyAdapter(graph, body, joint_activation_tau_s=.09,
            fast_coxa_activation=True, ltm_tendons=True)
        corrected = KneeAdapter(RateBodyAdapter(graph, body, joint_activation_tau_s=.09,
            fast_coxa_activation=True, ltm_tendons=True), [1.5, 1.8, 3.5, 1.3])
        untouched = np.setdiff1d(np.arange(42), original.joints[SELECTED, 5])
        rates = np.random.default_rng(71).uniform(0, 150, len(original.motor_ids))
        for _ in range(8):
            expected, _ = original.decode(rates)
            actual, _ = corrected.decode(rates)
            np.testing.assert_array_equal(actual[untouched], expected[untouched])
            np.testing.assert_array_equal(corrected.tendon_activation, original.tendon_activation)
        actual, adhesion = corrected.decode(rates, motor_cut=True)
        np.testing.assert_array_equal(actual, body.neutral)
        self.assertFalse(adhesion.any())
        self.assertFalse(corrected.offset.any())
        self.assertFalse(corrected.tendon_activation.any())

    def test_four_knee_state_restore_requires_same_identity_and_repeats_future(self):
        import copy

        from flylab.c.rate_body import RateBodyAdapter
        from tests.test_c_rate_body import fixture
        from tools.verify_c_banc_m3_knee_trial import KneeAdapter
        from tools.verify_c_banc_observation import exact

        graph, body, _ = fixture()
        adapter = KneeAdapter(RateBodyAdapter(graph, body), [1.5, 1.8, 3.5, 1.3])
        rates = np.random.default_rng(72).uniform(0, 150, len(adapter.motor_ids))
        adapter.decode(rates)
        saved = adapter.snapshot()
        expected = adapter.decode(rates[::-1])[0]
        future = adapter.snapshot()
        adapter.restore(saved)
        np.testing.assert_array_equal(adapter.decode(rates[::-1])[0], expected)
        exact(future, adapter.snapshot(), 'four-knee future')
        wrong = copy.deepcopy(saved)
        wrong['gains'][0] += .01
        with self.assertRaises((ValueError, AssertionError)):
            adapter.restore(wrong)
        wrong = copy.deepcopy(saved)
        wrong['hash'] = 'different'
        with self.assertRaises(ValueError):
            adapter.restore(wrong)

    def test_recurrent_mask_preserves_zero_motor_and_unknown_side_boundaries(self):
        from types import SimpleNamespace

        from scipy.sparse import csr_matrix

        from tools.banc_m3_recurrent_mask import recurrent_mask
        nodes=[{'id':f'n{i}','super_class':'ventral_nerve_cord_intrinsic',
            'cell_type':'A','soma_side':side} for i,side in enumerate(
                ('left','left','right','unknown','left'))]
        nodes[0]['super_class']='motor';nodes[4]['super_class']='sensory'
        nodes[3]['cell_type']=''
        w=csr_matrix((np.array([.5,-.3,.1,.2,0.,-.4,.7,.2,.5,.8,.6,-.7,.1]),
            np.array([1,2,3,4,1,2,3,4,1,3,1,2,1]),np.array([0,4,8,10,12,13])),shape=(5,5))
        saved=w.copy()
        population,positions,groups,keys=recurrent_mask(SimpleNamespace(nodes=nodes),w,np.array([0]))
        np.testing.assert_array_equal(population,[1,2,3])
        posts=np.repeat(np.arange(5),np.diff(w.indptr))[positions]
        self.assertTrue(np.isin(posts,population).all())
        self.assertTrue(np.isin(w.indices[positions],population).all())
        self.assertTrue((w.data[positions]!=0).all())
        self.assertEqual(len(positions),6)
        for post,position,group in zip(posts,positions,groups):
            key=keys[group];pre=w.indices[position]
            self.assertEqual(key[1],int(np.sign(w.data[position])))
            if 3 in (post,pre):self.assertEqual(key[2],'unknown')
            if post==3:self.assertEqual(key[0],'n3')
        np.testing.assert_array_equal(w.data,saved.data)

    def test_export_retains_mask_sign_and_unselected_rows(self):
        m=network(False)
        with torch.no_grad():
            m.log_efficacy.fill_(10)
            m.log_tau_ratio.fill_(-10)
            m.log_threshold_ratio.fill_(10)
        m.project_parameters()
        w,tau,threshold=m.physical_parameters()
        np.testing.assert_array_equal(np.sign(w.detach().numpy()),np.sign(m.base_weights.numpy()))
        np.testing.assert_array_equal(w.detach()[1],m.base_weights[1])
        np.testing.assert_allclose(tau.detach(),[.0125,.02],rtol=1e-6)
        np.testing.assert_allclose(threshold.detach(),[15,7.5],rtol=1e-6)


    def test_continuous_state_matches_independent_rate_equation(self):
        m=network()
        currents=np.array([[[60,40],[61,41],[62,42]],[[64,42],[65,43],[66,44]]],np.float32)
        _,r,z=m(torch.tensor(currents),torch.zeros(2),torch.zeros(2))
        reference_r,reference_z=np.zeros(2),np.zeros(2)
        for b in currents:
            reference_r,reference_z=rk4(reference_r,reference_z,b[[0,1,1,2]],
                m.base_weights.numpy(),.005)
        np.testing.assert_allclose(r.detach().numpy()*200,reference_r,rtol=2e-6)
        np.testing.assert_allclose(z.detach().numpy()*200,reference_z,rtol=2e-6)


    def test_parameter_gradient_matches_finite_difference(self):
        m=network()
        b=torch.full((5,3,2),60.)
        def objective():
            return m(b,torch.zeros(2),torch.zeros(2))[0].square().mean()
        objective().backward()
        actual=float(m.log_efficacy.grad[0])
        with torch.no_grad():
            m.log_efficacy[0]=.002
            plus=float(objective())
            m.log_efficacy[0]=-.002
            minus=float(objective())
        assert abs(actual)>1e-6
        np.testing.assert_allclose(actual,(plus-minus)/.004,rtol=.01,atol=1e-5)


    def test_zero_input_has_no_hidden_phase_generator(self):
        m=network()
        q,r,z=m(torch.zeros((20,3,2)),torch.zeros(2),torch.zeros(2))
        np.testing.assert_array_equal(q.detach().numpy(),np.full((20,1),.3,np.float32))
        assert torch.count_nonzero(r)==0
        assert torch.count_nonzero(z)==0

    def test_inverse_cascade_transpose_is_an_adjoint(self):
        import time
        rng=np.random.default_rng(42)
        operator=inverse_operator(rng.normal(size=(42,6)),11,time.perf_counter()+10)
        x=rng.normal(size=operator.shape[1]);y=rng.normal(size=operator.shape[0])
        np.testing.assert_allclose(y@(operator@x),x@operator.rmatvec(y),rtol=1e-12,atol=1e-12)

    def test_motor_teaching_matches_independent_coupled_ode(self):
        from types import SimpleNamespace

        from scipy.integrate import solve_ivp
        weights = np.array([[.2,-.1],[.4,0.]])
        initial = np.array([1.,2.,3.,4.])
        target = 30.
        n = SimpleNamespace(torch=torch,device='cpu',dt=.00025,
            rate=torch.tensor(initial[:2]),adaptation=torch.tensor(initial[2:]))
        def ordinary(r,z):
            drive=40+torch.tensor(weights)@r-7.5-z
            return ((200*torch.tanh(drive/200)).clamp_min(0)-r)/.02,(8*r-z)/.3
        n._derivative_pair=ordinary
        teacher=MotorTeacher(n,np.array([0]))
        teacher.target.fill_(target)
        for _ in range(80):
            teacher.step()
        def rhs(t,x):
            r,z=x[:2],x[2:]
            dr=(np.maximum(200*np.tanh((40+weights@r-7.5-z)/200),0)-r)/.02
            dr[0]=(target-r[0])/.02
            return np.r_[dr,(8*r-z)/.3]
        independent=solve_ivp(rhs,(0,.02),initial,rtol=1e-11,atol=1e-12).y[:,-1]
        np.testing.assert_allclose(np.r_[n.rate,n.adaptation],independent,rtol=2e-6,atol=1e-7)
        r,z=held_motor_state(initial[0],initial[2],target,.02)
        np.testing.assert_allclose([r,z],independent[[0,2]],rtol=1e-10)
        self.assertGreater(float(teacher.correction.sum()),0)

    def test_continuous_target_filter_matches_state_space(self):
        from scipy.linalg import expm
        rng=np.random.default_rng(19)
        v=rng.uniform(0,199,size=(31,5))
        transition=expm(np.array([[-100.,0.,100.],[50.,-50.,0.],[0.,0.,0.]])*.005)
        state=np.zeros((2,5));rates=[]
        for row in v:
            state=transition[:2,:2]@state+transition[:2,2,None]*row
            rates.append(state[1].copy())
        np.testing.assert_allclose(continuous_neural_filter(v),rates,atol=2e-12,rtol=2e-14)
        dual=rng.normal(size=v.shape)
        np.testing.assert_allclose(np.sum(continuous_neural_filter(v)*dual),np.sum(v*continuous_neural_filter(dual,transpose=True)),rtol=1e-12)

    def test_smooth_motor_teaching_matches_independent_ode(self):
        from types import SimpleNamespace

        from scipy.integrate import solve_ivp
        n=SimpleNamespace(torch=torch,device='cpu',dt=.00025,
            rate=torch.tensor([1.],dtype=torch.float64),adaptation=torch.tensor([3.],dtype=torch.float64))
        n._derivative_pair=lambda r,z: (-r/.02,(8*r-z)/.3)
        teacher=ContinuousMotorTeacher(n,np.array([0]))
        teacher.target=teacher.target.double();teacher.activation=teacher.activation.double()
        teacher.activation.fill_(2);teacher.target.fill_(30)
        for _ in range(80):teacher.step()
        expected=solve_ivp(lambda t,x: [(x[2]-x[0])/.02,(8*x[0]-x[1])/.3,(30-x[2])/.01],
            (0,.02),[1,3,2],rtol=1e-11,atol=1e-12).y[:,-1]
        np.testing.assert_allclose(np.r_[n.rate,n.adaptation,teacher.activation],expected,rtol=1e-10,atol=1e-10)

    def test_pooled_command_gradient_and_native_conditional_replay(self):
        from tools.fit_c_banc_m3_motor_currents import conditional_commands
        m=MotorCommandNetwork(np.ones(2),np.array([[0],[1]]),np.zeros((2,2)),
            np.zeros((2,2),int),np.array([[.5,.5]]),np.ones((1,1)),np.zeros(1))
        x=torch.full((2,40,1),.3);drive=torch.zeros((40,2))
        q,_,_,_=m(x,drive,native=True)
        data={'motor_ids':np.array([0,1]),'moment':np.ones((1,1)),
            'rest':np.zeros(1),'motor_group':np.array([0,0])}
        expected,_=conditional_commands(np.full((40,2),60.),np.zeros((40,2)),data)
        np.testing.assert_allclose(q.detach(),expected,rtol=2e-6,atol=1e-7)
        loss=q.square().mean();loss.backward();gradient=float(m.efficacy.grad[0])
        values=[]
        with torch.no_grad():
            for theta in (1.002,.998):
                m.efficacy[0]=theta
                values.append(float(m(x,drive,native=True)[0].square().mean()))
        np.testing.assert_allclose(gradient,(values[0]-values[1])/.004,rtol=.005)

    def test_student_motor_recurrence_matches_independent_ode(self):
        from scipy.integrate import solve_ivp
        w=np.array([[.2,-.15],[.25,0.]])
        m=MotorCommandNetwork(np.ones(2),np.array([[0],[1]]),w,np.zeros((2,2),int),
            np.array([[.5,.5]]),np.ones((1,1)),np.zeros(1))
        # Align the initial rectifier crossing to a native step boundary; this
        # tests recurrence, separately from off-grid nonsmooth event accuracy.
        _,rates,_,_=m(torch.full((2,40,1),.1875),torch.zeros((40,2)),student_motor=True,native=True)
        def rhs(t,x):
            r,z=x[:2],x[2:]
            external=.1875*min(t/.0025,1)
            u=np.maximum(np.tanh(external+w@r-7.5/200-z),0)
            return np.r_[(u-r)/.02,(8*r-z)/.3]
        expected=solve_ivp(rhs,(0,.1),np.zeros(4),rtol=1e-10,atol=1e-12,max_step=.0002).y[:2,-1]
        np.testing.assert_allclose(rates[-1].detach(),expected,rtol=2e-6,atol=1e-8)

    def test_truncated_credit_preserves_all_forward_state_values(self):
        m=MotorCommandNetwork(np.ones(2),np.array([[0],[1]]),np.array([[.2,-.15],[.25,0.]]),
            np.zeros((2,2),int),np.array([[.5,.5]]),np.ones((1,1)),np.zeros(1))
        x=torch.full((2,80,1),.3);drive=torch.zeros((80,2))
        full=m(x,drive,student_motor=True)
        truncated=m(x,drive,student_motor=True,truncate_controls=10)
        for a,b in zip(full[:3],truncated[:3]):
            np.testing.assert_array_equal(a.detach(),b.detach())
        truncated[0].square().mean().backward()
        self.assertTrue(bool(torch.isfinite(m.efficacy.grad).all()))

    def test_full_candidate_export_preserves_unselected_and_zero_edges(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from scipy.sparse import csr_matrix
        source=csr_matrix((np.array([1.,-.3,0.,.4],np.float32),[0,0,1,2],[0,1,3,4]),shape=(3,3))
        session=SimpleNamespace(adapter=SimpleNamespace(motor_ids=np.array([1])),
            navigation=SimpleNamespace(encoder=SimpleNamespace(calibrate_weights=lambda w,ids:w)))
        candidate={'motor_ids':np.array([1]),'incoming_source_data':np.array([-.3],np.float32),
            'incoming_data':np.array([-.6],np.float32),'incoming_indptr':np.array([0,1]),'incoming_indices':np.array([0])}
        with patch('tools.verify_c_banc_m3_full_trial.rate_weights',return_value=(source.copy(),None)):
            actual,count=calibrated_weights(None,session,candidate)
        self.assertEqual(count,1)
        np.testing.assert_array_equal(actual.indptr,source.indptr)
        np.testing.assert_array_equal(actual.indices,source.indices)
        np.testing.assert_array_equal(actual.data, np.array([1.,-.6,0.,.4],np.float32))
        candidate['incoming_data'][0]=.6
        with patch('tools.verify_c_banc_m3_full_trial.rate_weights',return_value=(source.copy(),None)), self.assertRaises(ValueError):
            calibrated_weights(None,session,candidate)

    def test_boundary_replay_keeps_matched_nonzero_initial_state(self):
        from scipy.integrate import solve_ivp
        from scipy.sparse import csr_matrix

        from tools.probe_c_banc_m3_closed_boundary import motor_replay
        initial=(np.array([2.]),np.array([3.]),np.array([1.]))
        command,rates=motor_replay(np.full((8,1),60.),np.array([60.]),csr_matrix([[.2]]),
            np.ones((1,1)),np.ones((1,1)),np.array([.1]),initial)
        solved=solve_ivp(lambda t,x: [(max(200*np.tanh((60+.2*x[0]-7.5-x[1])/200),0)-x[0])/.02,
            (8*x[0]-x[1])/.3],(0,.02),[2,3],t_eval=np.arange(1,5)*.005,rtol=1e-11,atol=1e-12)
        np.testing.assert_allclose(rates[:,0],solved.y[0],rtol=1e-8,atol=1e-8)
        a=1.;expected=[]
        for rate in solved.y[0]:
            a+=(1-np.exp(-.5))*(rate-a);expected.append(.1+.02*a)
        np.testing.assert_allclose(command[:,0],expected,rtol=1e-8,atol=1e-8)
        np.testing.assert_array_equal(np.concatenate(initial),[2,3,1])

    def test_full_sparse_derivative_includes_untrained_feedback(self):
        from scipy.sparse import csr_matrix

        from tools.train_c_banc_m3_full_circuit import AnatomicalOperator, SparseCurrent
        source=csr_matrix(np.array([[.1,.4,0],[.3,-.2,.1],[0,.25,.2]]))
        positions=np.array([2,3,5]);groups=np.array([0,1,0])
        operator=AnatomicalOperator(source,positions,groups,2,dtype=torch.float64)
        theta=torch.tensor([1.3,.8],dtype=torch.float64,requires_grad=True)
        initial=torch.tensor([.1,.2,.3],dtype=torch.float64,requires_grad=True)
        operator.prepare(theta);r=initial
        for _ in range(4):r=torch.tanh(SparseCurrent.apply(r,theta,operator)+.2)
        loss=r.square().sum();loss.backward()
        actual_r=initial.grad.clone();actual_theta=theta.grad.clone()
        t=theta.detach().clone().requires_grad_();start=initial.detach().clone().requires_grad_()
        entries=torch.as_tensor(source.data).clone();entries[positions]*=t[groups]
        dense=torch.zeros((3,3),dtype=torch.float64)
        dense[np.repeat(np.arange(3),np.diff(source.indptr)),source.indices]=entries
        expected=start
        for _ in range(4):expected=torch.tanh(dense@expected+.2)
        expected.square().sum().backward()
        np.testing.assert_allclose(r.detach(),expected.detach(),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(actual_r,start.grad,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(actual_theta,t.grad,rtol=1e-12,atol=1e-12)

    def test_sparse_derivative_rejects_changed_pending_weights(self):
        from scipy.sparse import csr_matrix

        from tools.train_c_banc_m3_full_circuit import AnatomicalOperator, SparseCurrent
        operator=AnatomicalOperator(csr_matrix([[.5]]),np.array([0]),np.array([0]),1,dtype=torch.float64)
        theta=torch.ones(1,dtype=torch.float64,requires_grad=True)
        operator.prepare(theta);value=SparseCurrent.apply(torch.ones(1,dtype=torch.float64),theta,operator)
        with torch.no_grad():theta.add_(.1)
        operator.prepare(theta)
        with self.assertRaisesRegex(ValueError,'Weights changed'):
            value.sum().backward()

    def test_full_circuit_block_carry_preserves_native_values(self):
        from scipy.sparse import csr_matrix

        from tools.train_c_banc_m3_full_circuit import (
            AnatomicalOperator,
            FullCircuit,
            checkpointed_gradient,
        )
        source=csr_matrix(np.array([[.1,.4,0],[.3,-.2,.1],[0,.25,.2]],np.float32))
        operator=AnatomicalOperator(source,np.array([2,3,5]),np.array([0,1,0]),2)
        calibration={'efficacy':np.array([1.3,.8],np.float32),'motor_ids':np.array([1,2]),
            'moment':np.ones((1,1)),'rest':np.array([.1])}
        initial={'rate':np.array([2.,3.,1.]),'adaptation':np.array([1.,2.,3.])}
        parameters={'tau':np.full(3,.02),'a':np.ones(3),'threshold':np.full(3,7.5),'cap':np.full(3,200.)}
        model=FullCircuit(operator,calibration,np.array([[.5,.5]]),np.arange(3),initial,parameters)
        drive=torch.tensor([[20.,30.,0.]]*6)
        with torch.inference_mode():
            whole=model(drive,dt=.00025,return_state=True)
            first=model(drive[:3],dt=.00025,return_state=True)
            second=model(drive[3:],dt=.00025,return_state=True,initial_state=tuple(v.detach() for v in first[2]))
        for i in (0,1):np.testing.assert_array_equal(whole[i],torch.cat((first[i],second[i])))
        for a,b in zip(whole[2],second[2]):np.testing.assert_array_equal(a,b)
        target=torch.full((6,1),.2);mask=torch.ones(6,dtype=torch.bool);scale=torch.ones(1)
        direct=model(drive,dt=.00025)[0]
        loss=(direct-target).square().mean()+1e-4*(model.efficacy-1).square().mean()
        loss.backward();expected_gradient=model.efficacy.grad.detach().clone()
        value,command,gradient,_=checkpointed_gradient(model,drive,target,mask,scale,block_controls=2)
        np.testing.assert_allclose(value,float(loss.detach()),rtol=1e-6,atol=1e-8)
        np.testing.assert_array_equal(command,direct.detach())
        np.testing.assert_allclose(gradient,expected_gradient,rtol=2e-6,atol=1e-9)

    def test_shared_weight_branches_keep_coupled_gradients(self):
        from scipy.sparse import csr_matrix

        from tools.train_c_banc_m3_full_circuit import AnatomicalOperator, SparseCurrent
        source=csr_matrix(np.array([[.1,.4],[.3,-.2]]))
        operator=AnatomicalOperator(source,np.array([2,3]),np.array([0,1]),2,dtype=torch.float64)
        theta=torch.tensor([1.3,.8],dtype=torch.float64,requires_grad=True)
        branches=[]
        for delta in (0.,-.01,.01):
            operator.prepare(theta)
            value=torch.tensor([.2+delta,.3],dtype=torch.float64)
            for _ in range(3):value=torch.tanh(SparseCurrent.apply(value,theta,operator)+.2)
            branches.append(value)
        loss=sum(v.square().mean() for v in branches)/3+sum((v-branches[0]).square().mean() for v in branches[1:])/2
        loss.backward();actual=theta.grad.clone()
        t=theta.detach().clone().requires_grad_();matrix=torch.as_tensor(source.toarray()).clone()
        matrix[1,:]*=t;expected=[]
        for delta in (0.,-.01,.01):
            value=torch.tensor([.2+delta,.3],dtype=torch.float64)
            for _ in range(3):value=torch.tanh(matrix@value+.2)
            expected.append(value)
        dense_loss=sum(v.square().mean() for v in expected)/3+sum((v-expected[0]).square().mean() for v in expected[1:])/2
        dense_loss.backward()
        np.testing.assert_allclose(actual,t.grad,rtol=1e-12,atol=1e-12)

    def test_cell_size_patch_preserves_unselected_cells_and_input_units(self):
        from tools.banc_m3_cell_size import patch_parameters

        gain=np.array([1.,.8,1.2],np.float32)
        threshold=np.array([7.5,8.,6.],np.float32)
        before=(gain.copy(),threshold.copy())
        size=np.array([4.,.5],np.float32)
        ids=np.array([0,2])
        a,t=patch_parameters(gain,threshold,ids,gain[ids]/size,threshold[ids]*size)
        currents=np.array([[100.,20.,-10.],[300.,-5.,60.]],np.float64)
        actual=np.maximum(200*np.tanh(a*(currents-t)/200),0)
        normalized=currents.copy()
        normalized[:,ids]/=size
        expected=np.maximum(200*np.tanh(gain*(normalized-threshold)/200),0)
        np.testing.assert_allclose(actual,expected,rtol=1e-7,atol=1e-7)
        np.testing.assert_array_equal(a[[1]],gain[[1]])
        np.testing.assert_array_equal(t[[1]],threshold[[1]])
        for invalid in (np.array([0,0]),np.array([0,3]),np.array([0.,2.])):
            with self.assertRaises(ValueError):
                patch_parameters(gain,threshold,invalid,[1.,1.],[7.5,7.5])
        np.testing.assert_array_equal(gain,before[0])
        np.testing.assert_array_equal(threshold,before[1])

    def test_core_wave_forcing_reaches_unforced_cells_at_native_rk_stages(self):
        from types import SimpleNamespace

        from scipy.integrate import solve_ivp

        from tools.probe_c_banc_m3_core_teacher import CoreWaveTeacher

        initial=np.array([20.,2.,0.,4.])
        n=SimpleNamespace(torch=torch,device='cpu',dt=.00025,
            rate=torch.tensor(initial[:2]),adaptation=torch.tensor(initial[2:]))
        def ordinary(r,z):
            return torch.stack((r[0]*0,3*r[0]-4*r[1])),(8*r-z)/.3
        n._derivative_pair=ordinary
        table=(20+1000*np.arange(161)*.000125)[:,None]
        teacher=CoreWaveTeacher(n,np.array([0]),table)
        for _ in range(80):teacher.step()
        def rhs(t,x):
            core=20+1000*t
            return [1000.,3*core-4*x[1],0.,(8*x[1]-x[3])/.3]
        expected=solve_ivp(rhs,(0,.02),initial,rtol=1e-11,atol=1e-12).y[:,-1]
        np.testing.assert_allclose(np.r_[n.rate,n.adaptation],expected,rtol=1e-10,atol=1e-10)
        self.assertEqual(int(teacher.cursor),160)
        self.assertGreater(float(teacher.correction.sum()),0)

    def test_efficacy_patch_retains_topology_and_rejects_sign_or_source_changes(self):
        from scipy.sparse import csr_matrix

        from tools.banc_m3_synaptic_efficacy import patch_weights

        source = csr_matrix(np.array([[0., .2, -.3], [1., 0., .4], [-.1, 0., 0.]], np.float32))
        original = source.copy()
        ids = np.array([1, 3])
        values = np.array([-.6, .1], np.float32)
        candidate = patch_weights(source, ids, source.data[ids], values)
        np.testing.assert_array_equal(candidate.indptr, original.indptr)
        np.testing.assert_array_equal(candidate.indices, original.indices)
        np.testing.assert_array_equal(candidate.data[ids], values)
        np.testing.assert_array_equal(candidate.data[[0, 2, 4]], original.data[[0, 2, 4]])
        np.testing.assert_array_equal(source.data, original.data)
        for changed in ([.6, .1], [0., .1], [-.6, float('nan')]):
            with self.assertRaises(ValueError):
                patch_weights(source, ids, source.data[ids], changed)
        with self.assertRaises(ValueError):
            patch_weights(source, ids, [-.31, .4], values)
        for invalid in (np.array([1, 1]), np.array([1, 5]), np.array([1., 3.])):
            with self.assertRaises(ValueError):
                patch_weights(source, invalid, source.data[ids], values)

    def test_harmonic_relay_measurement_distinguishes_amplitude_phase_and_constant(self):
        from tools.probe_c_banc_m3_premotor_relay import harmonic_signal

        time = np.arange(300) * .005
        values = np.column_stack((30 + 12 * np.cos(2 * np.pi * time / .15 - .4),
                                  np.full(300, 17.)))
        measured = harmonic_signal(time, values)
        np.testing.assert_allclose(measured['explained_fraction'], [1., 0.], atol=1e-12)
        np.testing.assert_allclose(measured['peak_to_peak_first_harmonic'], [24., 0.], atol=1e-12)
        self.assertAlmostEqual(measured['phase_cycle'][0], .4 / (2 * np.pi))

    def test_native_tau_patch_matches_analytical_relaxation_and_separates_identity(self):
        from scipy.sparse import csr_matrix

        from tools.banc_m3_core_adaptation import CoreAdaptationNetwork
        from tools.banc_m3_intrinsic_recruitment import patch_tau

        source_tau = np.full(2, .02, np.float32)
        changed_tau = patch_tau(source_tau, np.array([1]), [.0465910855723])
        np.testing.assert_array_equal(source_tau, np.full(2, .02, np.float32))
        self.assertEqual(changed_tau[0], source_tau[0])
        def make(tau):
            return CoreAdaptationNetwork(csr_matrix((2, 2), dtype=np.float32), tau,
                np.ones(2), np.full(2, 7.5), np.full(2, 200), adaptation_gain=8,
                adaptation_tau_s=.3, dt=.00025, device='cpu', core_ids=[0, 1], core_gain=0)
        network = make(changed_tau)
        drive = np.array([30., 50.], np.float32)
        network.advance(drive, 80)
        target = 200 * np.tanh((drive.astype(float) - 7.5) / 200)
        expected = target * (1 - np.exp(-.02 / changed_tau.astype(float)))
        np.testing.assert_allclose(network.readout(), expected, atol=5e-6, rtol=1e-6)
        np.testing.assert_array_equal(network.adaptation, np.zeros(2))
        with self.assertRaises(ValueError):
            make(source_tau).restore(network.snapshot())
        for ids, values in (([1, 1], [.02, .03]), ([2], [.02]), ([1.], [.02]),
                ([1], [0.]), ([1], [float('nan')])):
            with self.assertRaises(ValueError):
                patch_tau(source_tau, np.asarray(ids), values)

    def test_event_phase_recovers_known_offset_and_frequency_mismatch(self):
        from tools.fit_c_banc_m3_phase_convergence import event_phase

        times = np.arange(20001) * .00025
        rates = np.zeros((len(times), 5))
        rates[:, 0] = 50 + 40*np.cos(2*np.pi*times/.15)
        rates[:, 4] = 50 + 40*np.cos(2*np.pi*(times/.15+.25))
        t, phase = event_phase({'time_s': times, 'rate': rates})
        np.testing.assert_allclose((phase+.25+.5) % 1-.5, 0, atol=1e-12)
        rates[:, 0] = 50 + 40*np.cos(2*np.pi*times/.1503)
        t, phase = event_phase({'time_s': times, 'rate': rates})
        self.assertAlmostEqual(np.polyfit(t, phase, 1)[0], 1/.1503-1/.15, places=7)

    def test_native_chain_tangent_matches_finite_parameter_response(self):
        from tools.fit_c_banc_m3_native_chain_sensitivity import tangent_step
        from tools.probe_c_banc_m3_pair_coupling import rate_step

        weights=np.array([[.1,-.8],[1.2,.3]])
        direction=np.array([[0.,0.],[1.2,0.]])
        current=np.array([20.,40.]);tau=np.array([.04,.02]);mask=np.array([1.,0.])
        initial=np.array([10.,5.]);state=np.column_stack((initial,np.zeros((2,2))))
        for _ in range(160):
            state=tangent_step(state,weights,current,tau,direction,mask)
        for parameter in range(2):
            values=[]
            for sign in (-1,1):
                theta=sign*1e-5
                w=weights+(np.exp(theta)-1)*direction if parameter==0 else weights
                t=tau*np.exp(theta*mask) if parameter==1 else tau
                rate=initial.copy()
                for _ in range(160):
                    rate=rate_step(rate,w,current,t)
                values.append(rate)
            np.testing.assert_allclose(state[:,parameter+1],(values[1]-values[0])/2e-5,atol=2e-8,rtol=2e-8)

    def test_peak_sensitivity_recovers_frequency_and_relative_phase(self):
        from tools.fit_c_banc_m3_native_chain_sensitivity import (
            pair_feature,
            phase_features,
        )

        times=np.arange(20001)*.00025
        phase=times[:,None]/.15+np.array([0.,.25])
        states=np.zeros((len(times),2,3))
        states[:,:,0]=50+40*np.cos(2*np.pi*phase)
        states[:,:,1]=-40*np.sin(2*np.pi*phase)*(2*np.pi*times[:,None])
        states[:,1,2]=-40*np.sin(2*np.pi*phase[:,1])*2*np.pi
        features=phase_features(times,states,[0,1])
        mean,gradient=pair_feature(features,0,1)
        self.assertAlmostEqual(mean,.75,places=10)
        np.testing.assert_allclose(gradient,[0.,-1.],atol=1e-5)
        np.testing.assert_allclose(features[3][:,0],-.15**2,atol=1e-6)
