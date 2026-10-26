"""Small generated E242 algebra, ownership, and failure accounting only."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from scipy import sparse

from tools import fit_c_banc_m3_matched_teacher as fit
from tools.fit_c_banc_m3_causal_neural_basis import quadratic_objective


def test_adjacent_train_q_velocity_objective_and_gradient():
    # Same row helper used by the real 1200-control build. No validation row.
    joint = np.array([[[0., 0.], [1., 0.]],
                      [[1., 2.], [2., 1.]],
                      [[2., 1.], [3., 2.]]])
    residual = np.array([[1., -1.], [.5, -.5], [.2, -.2]])
    matrix, offset = fit.normalized_train_rows(joint, residual, 2., 3.)
    q = (joint[1:] @ np.array([.3, .7]) + residual[1:])
    v = np.diff(joint @ np.array([.3, .7]) + residual, axis=0)/.005
    expected = np.r_[q.ravel()/(np.sqrt(4)*2),
                     v.ravel()/(np.sqrt(4)*3)]
    assert np.allclose(matrix @ np.array([.3, .7]) + offset, expected)
    P, linear, constant = quadratic_objective(matrix, offset, ridge=1e-8)
    x = np.array([.3, .7])
    direct = expected @ expected + 1e-8/2 * (x @ x)
    symmetric = lambda z: fit.symmetric_p_matvec(P,z)
    assert np.isclose(.5*x @ symmetric(x) + linear @ x + constant, direct)
    for i in range(2):
        epsilon = 1e-6
        e = np.eye(2)[i]*epsilon
        f = lambda z: .5*z @ symmetric(z) + linear @ z + constant
        assert np.isclose((f(x+e)-f(x-e))/(2*epsilon),
                          (symmetric(x) + linear)[i], rtol=2e-8, atol=2e-8)
    # A held/validation suffix cannot enter the passed training rows.
    assert len(matrix) == 8


def test_train_scale_uses_controls_one_to_1200_and_previous_control():
    original = np.zeros((2001, 42))
    target = np.zeros_like(original)
    for leg in range(6):
        sl = slice(leg*7, (leg+1)*7)
        original[:, sl] = .1*(leg+1)*np.arange(2001)[:, None]
    reference = {}
    for i, leg in enumerate(fit.LEGS):
        error = original[:, i*7:(i+1)*7]
        reference[leg] = {"q": fit.rms(error[1:1201]),
                          "velocity": fit.rms(np.diff(error[:1201],axis=0)/.005)}
    assert fit.training_scales(original, target, reference) == reference
    target[1201:] = 1e9
    assert fit.training_scales(original, target, reference) == reference
    reference["lf"]["q"] *= 1.01
    with pytest.raises(ValueError, match="TRAIN-only scale"):
        fit.training_scales(original, target, reference)


def test_muscle_column_order_fixed_rows_and_float32_decoder():
    by_leg = [np.arange(offset, offset+width//64) for offset, width in
              zip((0, 16, 30, 45, 61, 75), fit.WIDTHS)]
    # These generated group IDs are disjoint and cover90; six LTM stay zero.
    x = np.arange(5760, dtype=np.float64)/5760
    beta = fit.beta_from_x(x, by_leg)
    cursor=0
    for columns in by_leg:
        width=len(columns)*64
        assert np.array_equal(beta[columns],
            x[cursor:cursor+width].reshape(len(columns),64))
        cursor+=width
    assert np.array_equal(beta[90:], np.zeros((6,64)))
    B = np.zeros((3, 4, 2)); B[1:, 0, 0] = 1.; B[1:, 2, 1] = .3
    H = np.zeros((3, 4)); H[:, 0] = .123456789123
    member = np.array([0, 2, 1, 2])
    coefficients = np.array([[.5, 0], [.1, .25], [0, 0]])
    original = np.zeros((3,4), np.float32)
    original[:,1] = np.array([.2,.3,.4], np.float32)
    full, applied = fit.assemble_rates(B,H,coefficients,member,
        np.ones(4), np.array([0,2,3]), np.array([1]), original)
    assert full[1,0] == H[1,0]+.5
    assert applied[1,0] == np.float32(full[1,0])
    assert np.array_equal(applied[:,1], original[:,1])
    assert full[0,0] == H[0,0]


def test_metrics_held_row_and_split_and_eligibility():
    target = np.zeros((2001,42))
    baseline = np.ones_like(target)*.1
    candidate = np.ones_like(target)*.05
    candidate[0] = baseline[0] = target[0] = 3.  # held row excluded
    original = fit.command_metrics(baseline,target)
    current = fit.command_metrics(candidate,target)
    limits = {"global_q_rms_fraction_train_and_validation": .8,
        "global_velocity_rms_fraction_train_and_validation": 1.,
        "each_leg_q_rms_fraction_train_and_validation": 1.,
        "each_leg_velocity_rms_fraction_train_and_validation": 1.,
        "startup_q_rms_fraction": .8, "startup_q_max_fraction": 1.}
    assert fit.eligibility(current,original,limits)["eligible"]
    assert np.isclose(current["global"]["train"]["q_max_abs_rad"], .05)
    assert np.isclose(current["legs"]["lf"]["train"]["velocity_max_abs_rad_s"],10.)
    candidate[1300,0] = 10.
    assert "lf.validation.q_rms_rad" in fit.eligibility(
        fit.command_metrics(candidate,target),original,limits)["reasons"]


def test_box_kkt_raw_shape_and_projection_objective():
    width = 5760
    problem = {"P": sparse.eye(width,format="csc"),
               "linear": np.zeros(width),
               "A": sparse.vstack((-sparse.eye(width),sparse.eye(width)),format="csc"),
               "b": np.r_[np.zeros(width),np.ones(width)],
               "objective_constant": 2.}
    x = np.zeros(width); x[0] = -5e-9
    s = problem["b"]-problem["A"] @ x
    raw = {"solver_x": x, "solver_s": s, "solver_z": np.zeros(width*2)}
    info = {"status": "Solved", "objective_primal": .5*x@x,
        "objective_dual": 0., "primal_residual": 0., "dual_residual": 0.,
        "solve_time_s": .1, "process_call_wall_s": .1}
    numerical = {"raw_box_tolerance":1e-8,"primal_residual_inf":1e-7,
        "stationarity_inf":1e-7,"nonnegative_tolerance":1e-8,
        "complementarity_max_abs":1e-7,"objective_atol":1e-8,
        "objective_rtol":1e-8}
    checks, valid = fit.assess_solver(problem,raw,info,numerical)
    assert valid and np.isclose(checks["raw_box_violation"],5e-9)
    projected = np.clip(x,0,1)
    assert fit.objective_value(problem,projected) <= fit.objective_value(problem,x)
    raw["solver_s"] = np.r_[s[:-1],np.nan]
    with pytest.raises(ValueError,match="raw solver"):
        fit.assess_solver(problem,raw,info,numerical)


def test_box_block_order_signs_and_objective_constant():
    left = sparse.csc_matrix([[2., .4], [0., 3.]])
    right = sparse.csc_matrix([[5.]])
    problem = fit.assemble_box_blocks([left,right],
        [np.array([1.,2.]),np.array([3.])], [.25,.75],
        [np.array([4,5]),np.array([9])])
    x = np.array([.2,.4,.7])
    assert np.array_equal(problem["group_columns"], [4,5,9])
    assert np.array_equal(problem["leg_offsets"], [0,2,3])
    assert np.array_equal(problem["A"].toarray(),
        np.r_[-np.eye(3),np.eye(3)])
    assert np.array_equal(problem["b"], [0,0,0,1,1,1])
    expected = .5*x @ np.array([[2,.4,0],[.4,3,0],[0,0,5]]) @ x
    expected += np.array([1,2,3]) @ x + 1.
    assert np.isclose(fit.objective_value(problem,x), expected)


def test_raw_solver_saved_before_status_check_and_one_call(tmp_path):
    class Returned:
        x = [0.]*5760
        s = [0.]*11520
        z = [0.]*11520
        status = "MaxTime"
        iterations = 2
        obj_val = 0.
        obj_val_dual = 0.
        r_prim = 0.
        r_dual = 0.
        solve_time = .01
    class FakeSolver:
        def solve(self):
            return Returned()
    calls = []
    def factory(*args):
        calls.append(args)
        return FakeSolver()
    width = 5760
    problem = {"P":sparse.eye(width,format="csc"),
        "linear":np.zeros(width),
        "A":sparse.vstack((-sparse.eye(width),sparse.eye(width)),format="csc"),
        "b":np.r_[np.zeros(width),np.ones(width)],"objective_constant":0.}
    ledger = {"constructor_entered":False,"solver_calls":0}
    raw, info = fit.solve_once(problem,tmp_path,ledger,lambda: 0.,factory)
    assert len(calls)==1 and info["status"]=="MaxTime"
    assert np.array_equal(np.load(tmp_path/"solver-raw.npz")["solver_x"],raw["solver_x"])
    assert ledger["solver_calls"]==1 and ledger["solver_returned"]
    with pytest.raises(RuntimeError,match="second"):
        fit.solve_once(problem,tmp_path,ledger,lambda: 0.,factory)
    assert len(calls)==1


def test_actual_leg_builder_is_train_only_and_preserves_column_order(monkeypatch):
    joint = np.zeros((1201, 7, 64))
    joint[:, 0, 0] = np.arange(1201)*.1
    joint[:, 1, 63] = np.arange(1201)*-.2
    calls = []
    def fake_jacobian(source, moment, columns, leg):
        calls.append((len(source), columns.copy(), leg))
        return joint
    monkeypatch.setattr(fit.grouped, "group_joint_matrix", fake_jacobian)
    q0 = np.zeros((2001,42))
    target = np.zeros_like(q0)
    q0[:1201,0] = .4
    matrix, offset = fit.train_leg_matrix(np.empty((2001,1)),q0,target,
        np.empty((42,96)),np.array([17]),0,2.,3.)
    assert calls[0][0] == 1201 and np.array_equal(calls[0][1],[17])
    assert matrix.shape == (16800,64)
    assert np.isclose(matrix[0,0], .1/(np.sqrt(8400)*2))
    assert np.isclose(matrix[8400,0], 20/(np.sqrt(8400)*3))
    assert np.isclose(matrix[8401,63], -40/(np.sqrt(8400)*3))
    target[1201:] = 1e12
    matrix2, offset2 = fit.train_leg_matrix(np.empty((2001,1)),q0,target,
        np.empty((42,96)),np.array([17]),0,2.,3.)
    assert np.array_equal(matrix,matrix2) and np.array_equal(offset,offset2)


def test_returned_solver_raw_survives_post_return_ledger_failure(tmp_path,monkeypatch):
    class Returned:
        x=[.25]*5760; s=[.5]*11520; z=[.125]*11520
        status="Solved"; iterations=4; obj_val=1.; obj_val_dual=1.
        r_prim=0.; r_dual=0.; solve_time=.01
    class FakeSolver:
        def solve(self):
            return Returned()
    calls=[]
    def factory(*_):
        calls.append(1)
        return FakeSolver()
    original=fit.save_json
    def fail_after_return(path,value):
        if str(path).endswith("ledger.json") and value.get("solver_returned"):
            raise OSError("post-return ledger failure")
        return original(path,value)
    monkeypatch.setattr(fit,"save_json",fail_after_return)
    width=5760
    problem={"P":sparse.eye(width,format="csc"),"linear":np.zeros(width),
        "A":sparse.eye(11520,width,format="csc"),"b":np.ones(11520)}
    ledger={"constructor_entered":False,"solver_calls":0}
    with pytest.raises(OSError,match="post-return ledger failure"):
        fit.solve_once(problem,tmp_path,ledger,lambda:0.,factory)
    assert len(calls)==1 and ledger["solver_returned"]
    with np.load(tmp_path/"solver-raw.npz") as raw:
        assert np.array_equal(raw["solver_x"], [.25]*5760)
        assert np.array_equal(raw["solver_s"], [.5]*11520)
        assert np.array_equal(raw["solver_z"], [.125]*11520)
    assert json.loads((tmp_path/"solver-info.json").read_text())["status"]=="Solved"


def test_constructor_prewrite_failure_does_not_claim_invocation(tmp_path,monkeypatch):
    calls=[]
    original=fit.save_json
    def fail_prewrite(path,value):
        if str(path).endswith("ledger.json") and value.get("constructor_entered"):
            raise OSError("pre-constructor ledger")
        return original(path,value)
    monkeypatch.setattr(fit,"save_json",fail_prewrite)
    width=5760
    problem={"P":sparse.eye(width,format="csc"),"linear":np.zeros(width),
        "A":sparse.eye(11520,width,format="csc"),"b":np.ones(11520)}
    ledger={"constructor_entered":False,"solver_calls":0}
    with pytest.raises(OSError,match="pre-constructor ledger"):
        fit.solve_once(problem,tmp_path,ledger,lambda:0.,lambda *_:calls.append(1))
    assert calls==[] and ledger["constructor_requested"]
    assert not ledger["constructor_entered"] and ledger["solver_calls"]==0


def test_complete_candidate_finite_held_fixed_and_both_tendons():
    candidate={"beta":np.zeros((96,64)),
        "rate64":np.zeros((2001,391)),
        "decoder_input32":np.zeros((2001,391),np.float32),
        "q64":np.zeros((2001,42)),"q32":np.zeros((2001,42)),
        "activation64":np.zeros((2001,96)),
        "activation32":np.zeros((2001,96)),
        "tendon64":np.zeros((2001,6)),"tendon32":np.zeros((2001,6)),
        "affine_q":np.zeros((2001,42))}
    arrays={"fixed_columns":np.arange(48),"ltm":np.arange(6),
        "motor_rate32":np.zeros((2001,391),np.float32),
        "held0":np.zeros(42),"activation0":np.zeros(96)}
    tol={"tendon_atol":2e-12}
    assert fit.validate_candidate(candidate,arrays,np.zeros((2001,6)),tol)=={
        "tendon64":0.,"tendon32":0.}
    candidate["tendon64"][0,0]=3e-12
    with pytest.raises(ValueError,match="tendon"):
        fit.validate_candidate(candidate,arrays,np.zeros((2001,6)),tol)
    candidate["tendon64"][0,0]=0
    candidate["q32"][3,0]=np.nan
    with pytest.raises(ValueError,match="nonfinite"):
        fit.validate_candidate(candidate,arrays,np.zeros((2001,6)),tol)
    candidate["q32"][3,0]=0
    candidate["beta"][0,0]=.1
    with pytest.raises(ValueError,match="LTM"):
        fit.validate_candidate(candidate,arrays,np.zeros((2001,6)),tol)


def test_primary_error_survives_late_publication_error(tmp_path, monkeypatch):
    primary = ValueError("science failure")
    real = fit.save_json
    def fail_ledger(path,value):
        if str(path).endswith("ledger.json"):
            raise OSError("late ledger write")
        return real(path,value)
    monkeypatch.setattr(fit,"save_json",fail_ledger)
    ledger={"errors":[]}
    result={"validity":"valid_conditional_matched_teacher_fit"}
    returned=fit.finish_receipts(tmp_path,result,ledger,primary,lambda:.1)
    assert returned is primary
    assert result["validity"]=="incomplete_or_invalid"
    assert json.loads((tmp_path/"result.json").read_text())["primary_error"]["message"]=="science failure"
    assert any(x["stage"]=="ledger" for x in ledger["errors"])
    assert (tmp_path/"output-manifest.json").exists()


@pytest.mark.parametrize("failed_file",["manifest","result"])
def test_late_publication_failure_invalidates_and_rehashes(tmp_path,monkeypatch,
                                                           failed_file):
    ledger={"errors":[]}
    result={"validity":"valid_conditional_matched_teacher_fit",
        "candidate_eligible":True,
        "metrics":{"eligibility":{"eligible":True,"reasons":[]}}}
    seen=[]
    if failed_file=="manifest":
        original=fit.manifest
        def once(path):
            seen.append(1)
            if len(seen)==1:
                raise OSError("first manifest failure")
            return original(path)
        monkeypatch.setattr(fit,"manifest",once)
    else:
        original=fit.save_json
        def once(path,value):
            if str(path).endswith("result.json"):
                seen.append(1)
                if len(seen)==1:
                    raise OSError("first result failure")
            return original(path,value)
        monkeypatch.setattr(fit,"save_json",once)
    primary=fit.finish_receipts(tmp_path,result,ledger,None,lambda:1.)
    assert isinstance(primary,OSError) and len(seen)>=2
    assert result["validity"]=="incomplete_or_invalid"
    assert result["candidate_eligible"] is False
    assert result["metrics"]["eligibility"]["eligible"] is False
    saved=json.loads((tmp_path/"result.json").read_text())
    assert saved["primary_error"]["type"]=="OSError"
    listed=json.loads((tmp_path/"output-manifest.json").read_text())["files"]
    assert listed["ledger.json"]["sha256"]==hashlib.sha256(
        (tmp_path/"ledger.json").read_bytes()).hexdigest()


def test_deadline_crossed_during_manifest_republishes_invalid(tmp_path,monkeypatch):
    original=fit.manifest
    crossed=[False]
    def advancing(path):
        value=original(path)
        crossed[0]=True
        return value
    monkeypatch.setattr(fit,"manifest",advancing)
    result={"validity":"valid_conditional_matched_teacher_fit",
            "candidate_eligible":True}
    ledger={"errors":[]}
    primary=fit.finish_receipts(tmp_path,result,ledger,None,
        lambda:151. if crossed[0] else 149.)
    assert isinstance(primary,TimeoutError)
    assert result["validity"]=="incomplete_or_invalid"
    assert result["candidate_eligible"] is False
    stored=json.loads((tmp_path/"result.json").read_text())
    assert stored["primary_error"]["type"]=="TimeoutError"
    manifest=json.loads((tmp_path/"output-manifest.json").read_text())
    assert manifest["files"]["ledger.json"]["sha256"]==hashlib.sha256(
        (tmp_path/"ledger.json").read_bytes()).hexdigest()


def test_time_reserve_blocks_without_solver():
    with pytest.raises(TimeoutError):
        fit.ensure_solve_budget(lambda:51.)
    fit.ensure_solve_budget(lambda:50.)
