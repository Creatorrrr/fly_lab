"""Generated CPU E333 fixtures, never original payloads or neural loaders."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import verify_c_banc_m3_smooth_efficacy_direction as subject
from tools.banc_m3_checkpointed_rollout import CheckpointedControlRollout
from tools.banc_m3_smooth_temporal import SmoothTemporalObjective


def native():
    return {"csr": {"data": np.array([.25, -.5, .125, -0.], np.float32),
                    "indices": np.arange(4, dtype=np.int32),
                    "indptr": np.r_[np.arange(5, dtype=np.int32), np.full(387,4,np.int32)]},
            "mask": {"positions": np.arange(3, dtype=np.int64), "edge_parameters": np.arange(3, dtype=np.int64)},
            "states": {"initial_neural": {"output_mask": np.ones(391, np.float32)}},
            "network_identity": "generated-network"}


def generated_fixture(*, zero=False, failure=None):
    phi = 2*np.pi*np.arange(128, dtype=np.float64)/128
    curves = np.stack([np.stack([.2+.1*np.sin(phi+.2*j) for j in range(7)], -1) for _ in range(6)])
    smooth = SmoothTemporalObjective(curves, np.full((99,42), .21), np.zeros((99,6)))
    dims = subject.TestDimensions(391, 3, (100,104))
    ticks = np.arange(1,301, dtype=np.float32)
    held = np.repeat(ticks[:,None], 391, axis=1)
    bank0, original0 = np.zeros((391,3)), np.zeros(96)
    def state(c):
        return {"network": {"output_mask": np.ones(391, np.float32)},
                "adapter": {"bank": bank0+c/1000, "original": {"activation": original0+c/1000}, "hash": "generated-decoder"}}
    data = {"native": native(), "start": state(0), "whole100": state(100), "final": state(300),
            "drives": held, "rates": held.copy(), "adaptations": held.copy(),
            "motor_rates": np.repeat(ticks[:,None],391,axis=1),
            "raw": {"command": np.full((301,42), .21), "tendon": np.zeros((301,6))}}
    class Objective:
        def evaluate_pairs(self, previous, q):
            terms = smooth.evaluate_pairs(previous,q)
            if zero:
                for key in subject.TERM_NAMES:
                    terms[key] = terms[key]*0
            return terms
        def evaluate(self,q,tendon):
            return smooth.evaluate(q,tendon)
    objective = Objective()
    baseline = objective.evaluate_pairs(torch.full((4,42), .21, dtype=torch.float64),
                                         torch.full((4,42), .21, dtype=torch.float64))
    data["e332_neutral_terms"] = subject.metrics(baseline)

    class Neural:
        n = 391
        device = torch.device("cpu")
        dt = np.float32(.00025)
        def __init__(self):
            self.operator = SimpleNamespace(group_count=3)
        def prepare_efficacy(self, theta):
            return theta
        def step(self,r,z,drive,theta,mask):
            if failure == "backward" and torch.is_grad_enabled():
                raise RuntimeError("generated backward failure")
            shift = 1000*torch.dot(theta-1, torch.tensor([1.,.4,.05], dtype=torch.float64))
            return (drive.to(torch.float64)+shift).to(torch.float32), drive.clone()

    class Decoder:
        device = torch.device("cpu")
        def step(self,motor,bank,original,*,dt):
            assert dt == .005
            offset = .0005*(motor[0].to(torch.float64)-(original[0]*1000+1))
            return {"q": torch.full((42,),.21,dtype=torch.float64)+offset,
                    "tendon": torch.zeros(6,dtype=torch.float64), "bank": bank+.001,
                    "original_activation": original+.001}
    class Deps:
        def __init__(self):
            self.load_called = False
            self.calls = []
            self.rollout = CheckpointedControlRollout(Neural(), Decoder(), np.arange(391,dtype=np.int64),
                steps_per_control=20, control_dt=.005, checkpoint_controls=1, max_controls=300)
            original_run = self.rollout.run
            def run(r,z,bank,original,drives,theta,mask):
                self.calls.append({"r": r.detach().numpy().copy(), "bank": bank.detach().numpy().copy(),
                    "original": original.detach().numpy().copy(), "theta": theta.detach().numpy().copy(), "count":len(drives)})
                output = original_run(r,z,bank,original,drives,theta,mask)
                if failure == "dtype":
                    output["q"] = output["q"].float()
                elif failure == "nonfinite":
                    output["tendon"][0,0] = torch.nan
                elif failure == "reproduction":
                    output["q"] = output["q"]+.001
                elif failure == "confirmation":
                    def fail_confirm():
                        raise RuntimeError("generated confirmation failure")
                    output["receipt"].confirm = fail_confirm
                elif failure == "gq_receipt":
                    def count_gq(grad):
                        output["receipt"].recompute_controls_requested += 1
                        return grad
                    output["q"].register_hook(count_gq)
                elif failure == "incomplete":
                    receipt = output["receipt"]
                    original_confirm = receipt.confirm
                    def incomplete_confirm():
                        original_confirm()
                        if receipt.confirm_returned == 2:
                            receipt.backward_blocks_returned -= 1
                    receipt.confirm = incomplete_confirm
                elif failure == "startup" and len(drives) == 300:
                    output["tendon"] = output["tendon"]+.1
                return output
            self.rollout.run = run
        def load(self,freeze,root):
            self.load_called = True
            return data
        def build(self,data,device):
            return self.rollout, objective, (torch.zeros(391,dtype=torch.float32), torch.zeros(391,dtype=torch.float32),
                torch.tensor(bank0), torch.tensor(original0), torch.tensor(held), torch.ones(391,dtype=torch.float32))
    return data, Deps(), dims, smooth


def test_f64_smooth_window_new_gq_gtheta_full_previous_and_FD():
    _, _, _, objective = generated_fixture()
    theta = torch.ones(3,dtype=torch.float64,requires_grad=True)
    rng = np.random.default_rng(333)
    matrix = torch.tensor(rng.normal(size=(4,42,3))*.001)
    baseline = torch.tensor(rng.normal(size=(4,42))*.01+.2)
    boundary = torch.full((42,),.2,dtype=torch.float64,requires_grad=True)
    q = baseline+torch.einsum("tjg,g->tj",matrix,theta-1)
    _, ell = subject.window_loss(objective,boundary,q)
    gq = torch.autograd.grad(ell,q,retain_graph=True)[0]
    assert theta.grad is None and boundary.grad is None
    ell.backward()
    expected = torch.einsum("tj,tjg->g",gq,matrix)
    torch.testing.assert_close(theta.grad,expected,rtol=1e-10,atol=1e-10)
    fd = []
    for i in range(3):
        d = torch.zeros(3,dtype=torch.float64); d[i]=1e-6
        plus = subject.window_loss(objective,boundary,baseline+torch.einsum("tjg,g->tj",matrix,d))[1]
        minus = subject.window_loss(objective,boundary,baseline+torch.einsum("tjg,g->tj",matrix,-d))[1]
        fd.append((plus-minus).item()/2e-6)
    np.testing.assert_allclose(theta.grad.numpy(),fd,rtol=1e-5,atol=1e-7)
    assert boundary.grad is None


def test_real_generated_checkpointed_receipt_gq_zero_neural_work_and_conditional_roster():
    data,deps,dims,_ = generated_fixture()
    snapshots = {}
    result,receipts = subject.compute(data,dependencies=deps,device="cpu",test_dimensions=dims,
        checkpoint=lambda name,arrays:snapshots.update({name:{key:np.array(value,copy=True) for key,value in arrays.items()}}))
    assert len(receipts) in (6,7) and result["actual_work"]["backward_rollout_calls"] == 2
    assert result["direction_supported"] and result["continuation"]["requested"], result
    assert result["actual_work"]["decoder_replay_controls"] == 300
    assert snapshots["decoder-replay"]["banks"].shape == (301,391,3)
    assert snapshots["decoder-replay"]["originals"].shape == (301,96)
    for c in dims.starts:
        row = result["runs"][f"window-{c}-neutral"]
        assert row["gq_receipt_before"] == row["gq_receipt_after"] and row["theta_backward_calls"] == 1
        assert snapshots[f"window-{c}-neutral-gq"]["gq"].shape == (4,42)
        assert snapshots[f"window-{c}-neutral-gradient"]["gradient"].shape == (3,)
        assert snapshots[f"window-{c}-neutral-start"]["source_neural_row_index"] == c-1
        np.testing.assert_allclose(snapshots[f"window-{c}-neutral-start"]["bank"], snapshots["decoder-replay"]["banks"][c], atol=0,rtol=0)
    assert all(call["count"]==4 for call in deps.calls[:-1])
    if result["continuation"]["requested"]:
        assert deps.calls[-1]["count"] == 300
        np.testing.assert_array_equal(deps.calls[-1]["r"],np.zeros(391,np.float32))
        np.testing.assert_array_equal(deps.calls[-1]["bank"],np.zeros((391,3)))
        assert result["continuation"]["intermediate_saved_boundaries_overwritten"] is False
    direction = snapshots["direction"]
    np.testing.assert_array_equal(direction["between_candidates_delta"],direction["theta_plus"]-direction["theta_minus"])
    for row in result["windows"]:
        assert row["observed_contribution"]-row["predicted_contribution"] == pytest.approx(row["O_minus_Q"]+row["Q_minus_T"],abs=1e-12)
    assert not result["candidate_adopted"]


def test_zero_new_gradient_has_no_perturbation_or_continuation():
    data,deps,dims,_ = generated_fixture(zero=True)
    result,receipts = subject.compute(data,dependencies=deps,device="cpu",test_dimensions=dims)
    assert result["zero_gradient"] and result["local_direction"] == "inconclusive" and len(receipts) == 2
    assert result["no_direction_information"] and not result["direction_supported"]
    assert result["window_L1_relative_error"] is None and result["window_L1_denominator"] is None
    assert not result["continuation"]["requested"] and len(deps.calls) == 2


def classify(o,t):
    count = len(o)
    tp,tm = np.array([1.001],np.float32).astype(np.float64),np.array([.999],np.float32).astype(np.float64)
    gradients = np.array(t,np.float64)[:,None]/(tp-tm)
    neutral = np.full(count,100.,np.float64)
    return subject.classify_direction(neutral,neutral+np.array(o,np.float64)*count/2,
        neutral-np.array(o,np.float64)*count/2,gradients,tp,tm)


def test_primary_and_L1_cancellation_negative_window_zero_and_nonfinite():
    result = classify([1,1],[101,-99])
    assert result["primary_local_direction"] == "supported" and not result["window_L1_supported"]
    assert not result["direction_supported"]
    result = classify([100,-99],[100,-99.05])
    assert result["window_L1_supported"] and result["primary_local_direction"] == "refuted"
    result = classify([2,-1],[2,-1])
    assert result["direction_supported"]
    result = classify([0,0],[0,0])
    assert result["window_L1_relative_error"] is None and result["local_direction"] == "inconclusive"
    with pytest.raises(ValueError):
        classify([1,1],[-2,-2])
    with pytest.raises(ValueError):
        classify([np.nan],[1])


def test_actual_magnitude_and_full_palette_once_rounding_signed_zero():
    source = native()
    g = np.array([1., .25, 1e-10])
    direction = subject.make_direction(g)
    assert direction["theta_plus"][2] == direction["theta_minus"][2] == 1
    patched,selected,metadata = subject.patch_weights(source,direction["theta_minus"])
    expected = (source["csr"]["data"][:3].astype(np.float64)*direction["theta_minus"]).astype(np.float32)
    np.testing.assert_array_equal(selected.view(np.uint32),expected.view(np.uint32))
    assert patched["data"].view(np.uint32)[3] == source["csr"]["data"].view(np.uint32)[3]
    assert metadata["unselected_bytes_preserved"]
    values = np.r_[source["csr"]["data"],np.arange(20000,dtype=np.float32)+1]
    source["csr"] = {"data":values,"indices":np.arange(len(values),dtype=np.int32)%4,
                      "indptr":np.array([0,1,2,3,len(values)],np.int32)}
    assert not subject.patch_weights(source,direction["theta_minus"])[2]["eligible"]


def test_full_masks_detect_unselected_neuron_not_just_motor_rows():
    data,_,_,_ = generated_fixture()
    assert len(subject.validate_masks(data,391)) == 3
    data["whole100"]["network"]["output_mask"][390] = 0
    with pytest.raises(ValueError,match="supplied output masks"):
        subject.validate_masks(data,391)


@pytest.mark.parametrize("failure",["dtype","nonfinite","reproduction","confirmation","backward","gq_receipt","incomplete"])
def test_forward_raw_and_partial_receipt_precede_failure_validation(failure):
    data,deps,dims,_ = generated_fixture(failure=failure)
    snapshots,receipts = {},{}
    with pytest.raises((RuntimeError,ValueError)):
        subject.compute(data,dependencies=deps,device="cpu",test_dimensions=dims,
            checkpoint=lambda name,arrays:snapshots.update({name:{key:np.array(value,copy=True) for key,value in arrays.items()}}),
            on_receipt=lambda name,value:receipts.update({name:value}))
    assert "window-100-neutral-forward" in snapshots and "window-100-neutral" in receipts
    if failure == "nonfinite":
        assert np.isnan(snapshots["window-100-neutral-forward"]["tendon"][0,0])
    elif failure == "dtype":
        assert snapshots["window-100-neutral-forward"]["q"].dtype == np.float32
    if failure in ("backward","gq_receipt"):
        assert "window-100-neutral-gq" in snapshots
        assert receipts["window-100-neutral"]["backward_blocks_returned"] < 4


def test_acquired_raw_precedes_failing_host_returned_journal():
    data,deps,dims,_ = generated_fixture()
    snapshots = {}
    def boundary(name,event):
        if name == "window-100-neutral_forward" and event == "host_returned":
            raise OSError("generated host journal failure")
    with pytest.raises(OSError,match="host journal"):
        subject.compute(data,dependencies=deps,device="cpu",test_dimensions=dims,on_boundary=boundary,
            checkpoint=lambda name,arrays:snapshots.update({name:arrays}))
    assert snapshots["window-100-neutral-forward"]["q"].shape == (4,42)


def test_continuation_guard_rejects_startup_on_same_resetless_minus():
    data,deps,dims,_ = generated_fixture(failure="startup")
    result,_ = subject.compute(data,dependencies=deps,device="cpu",test_dimensions=dims)
    assert result["direction_supported"] and result["continuation"]["requested"]
    assert result["continuation"]["classification"] == "rejected" and not result["continuation"]["startup_pass"]
    assert not result["e334_handoff"]["eligible"]


def test_native_ineligible_still_completes_local_arms_and_blocks_continuation():
    data,deps,dims,_ = generated_fixture()
    values=np.r_[data["native"]["csr"]["data"],np.arange(20000,dtype=np.float32)+1]
    data["native"]["csr"]={"data":values,"indices":np.arange(len(values),dtype=np.int32)%391,
        "indptr":np.r_[np.array([0,1,2,3,len(values)],np.int32),np.full(387,len(values),np.int32)]}
    result,receipts=subject.compute(data,dependencies=deps,device="cpu",test_dimensions=dims)
    assert result["direction_supported"] and len(receipts) == 6 and len(deps.calls) == 6
    assert not result["native_eligibility"]["minus"]["eligible"] and not result["continuation"]["requested"]


def test_original_dimension_override_disallowed_without_explicit_fixture():
    data,deps,_,_ = generated_fixture()
    with pytest.raises(ValueError,match="explicit TestDimensions"):
        subject.compute(data,dependencies=deps,device="cpu")
    assert subject.N == 158706 and subject.GROUPS == 8547 and len(subject.STARTS) == 50


def test_passive_gate_rejects_before_loader_Torch_or_NPZ(tmp_path,monkeypatch):
    frozen = tmp_path/"freeze.json"; frozen.write_text("{}",encoding="utf8")
    review = tmp_path/"review.json"; review.write_text("{}",encoding="utf8")
    touched = []
    monkeypatch.setattr(subject.np,"load",lambda *a,**k:touched.append("payload"))
    with pytest.raises(ValueError):
        subject.gate(frozen,subject.e323.sha256(frozen),review,subject.e323.sha256(review),tmp_path/subject.OUTPUT,root=tmp_path)
    assert touched == []


def run_fixture(tmp_path,*,compute=None,writer=subject.put_json,clock=None,meter=None,saver=np.savez_compressed,
                emergency_writer=subject.e332.write_json,repair_writer=subject.e332._replace_terminal_json,
                output_meter=subject._output_bytes):
    _data,deps,dims,_ = generated_fixture()
    source = tmp_path/"generated.py"; source.write_text("# generated",encoding="utf8")
    frozen = {"source_sha256":{"generated.py":subject.e323.sha256(source)},"input_sha256":{"generated.py":subject.e323.sha256(source)}}
    path = tmp_path/"freeze.json"; subject.e332.write_json(path,frozen)
    review = tmp_path/"review.json"; subject.e332.write_json(review,{})
    kwargs = {} if clock is None else {"clock":clock}
    if meter is None:
        meter=lambda:{"host_peak_bytes":1,"device_total_bytes":1<<34,"device_free_bytes":1<<34,
                      "torch_reserved_bytes":0,"torch_peak_reserved_bytes":0}
    if compute is None:
        compute=subject.compute
    result=subject.run(path,subject.e323.sha256(path),review,subject.e323.sha256(review),tmp_path/"output",root=tmp_path,
        gatekeeper=lambda *a,**k:frozen,dependencies=deps,device="cpu",test_dimensions=dims,
        compute_fn=compute,writer=writer,meter=meter,saver=saver,
        emergency_writer=emergency_writer,repair_writer=repair_writer,output_meter=output_meter,**kwargs)
    return result,deps


def fake_supported(data,**kwargs):
    return {"local_direction":"supported","direction_supported":True,"continuation":{"classification":"supported"},
            "e334_handoff":{"eligible":True},"requested_rollouts":[]},{}


def test_generated_valid_run_final_disk_ledger_manifest_and_raw_roster(tmp_path):
    result,_=run_fixture(tmp_path)
    assert result["run_valid"] and result["effective_direction_qualification"]
    # This fixture's conditional local descent does not persist in steady200.
    assert result["continuation"]["requested"] and result["continuation"]["completed"]
    assert result["continuation"]["improvement"] < 1e-9 and result["continuation"]["classification"] == "rejected"
    assert not result["effective_continuation_qualification"] and not result["e334_handoff"]["eligible"]
    assert subject.read(tmp_path/"output/result.json") == result
    ledger=subject.read(tmp_path/"output/advance-ledger.json")
    assert ledger == result["cost"] and ledger["original_calls"] == 0
    manifest=subject.read(tmp_path/"output/output-manifest.json")
    raw={name for name in manifest["files"] if name.endswith(".npz")}
    assert len(raw) == 35  # decoder/global8 + neutral10 + local12 + contributions2 + continuation3
    for name,row in manifest["files"].items():
        path=tmp_path/"output"/name
        assert row == {"sha256":subject.e323.sha256(path),"bytes":path.stat().st_size}


def test_passive_generated_report_requires_current_source_and_actual_cpu_cuda_links(tmp_path):
    sources={subject.OWNED[0]:"a"*64,"tests/test_c_banc_checkpointed_rollout.py":"b"*64}
    inputs={subject.PATHS["e333_qualification_source"]:"c"*64,subject.PATHS["e332_math_source"]:"d"*64}
    qualification={"evidence_sha256":{}}
    for mode in ("cpu","cuda"):
        path=tmp_path/(mode+".json")
        proof={"schema":"flylab.m3.e333.root-generated-hookup.v1","mode":mode,"passed":True,"checks":{"generated":True},
            "source_sha256":"a"*64,"audit_source_sha256":"c"*64,"math_helper_sha256":"d"*64,
            "checkpoint_fixture_sha256":"b"*64,"original_calls":0,"whole_BANC_calls":0,"body_calls":0}
        subject.e332.write_json(path,proof)
        digest=subject.e323.sha256(path)
        inputs[path.name]=digest
        qualification["evidence_sha256"][path.name]=digest
        qualification[mode+"_report"]={"path":path.name,"sha256":digest}
    subject._qualification_reports(qualification,sources,inputs,tmp_path)
    sources[subject.OWNED[0]]="e"*64
    with pytest.raises(ValueError,match="producer qualification"):
        subject._qualification_reports(qualification,sources,inputs,tmp_path)


def test_run_partial_save_error_first_error_and_finite_json(tmp_path):
    def saver(path,**arrays):
        path.write_bytes(b"partial")
        raise OSError("generated raw save failure")
    result,_=run_fixture(tmp_path,saver=saver)
    assert not result["run_valid"] and result["first_error"]["message"] == "generated raw save failure"
    assert result["cost"]["attempts"][0]["returned"] is False
    assert subject.read(tmp_path/"output/result.json") == result


def test_run_postcommit_deadline_revokes_effective_qualification_and_handoff(tmp_path):
    elapsed=[0.]
    def writer(path,value):
        subject.put_json(path,value)
        if path.name == "result.json":elapsed[0]=221.
    result,_=run_fixture(tmp_path,compute=fake_supported,writer=writer,clock=lambda:elapsed[0])
    assert result["direction_supported"] and result["continuation"]["classification"] == "supported"
    assert not result["run_valid"] and not result["effective_direction_qualification"]
    assert not result["e334_handoff"]["eligible"] and result["first_error"]["stage"] == "publication"
    assert subject.read(tmp_path/"output/result.json") == result
    assert subject.read(tmp_path/"output/advance-ledger.json")["elapsed_s"] == result["elapsed_s"] == 221.


def test_postcommit_output_limit_revokes_effective_qualification(tmp_path):
    committed=[False]
    def writer(path,value):
        subject.put_json(path,value)
        if path.name == "result.json":committed[0]=True
    def output_meter(path):
        return subject.LIMITS["output_bytes"]+1 if committed[0] else subject._output_bytes(path)
    result,_=run_fixture(tmp_path,compute=fake_supported,writer=writer,output_meter=output_meter)
    assert result["direction_supported"] and not result["run_valid"] and not result["e334_handoff"]["eligible"]
    assert result["first_error"]["message"] == "E333 postcommit output limit exceeded"
    assert subject.read(tmp_path/"output/result.json") == result
    assert subject.read(tmp_path/"output/advance-ledger.json")["run_valid"] is False


def test_all_publication_repairs_fail_return_invalid_external_audit_required(tmp_path):
    def writer(path,value):
        if path.name == "result.json":raise OSError("terminal write failed")
        subject.put_json(path,value)
    def fail(path,value):
        raise OSError("recovery sink failed")
    result,_=run_fixture(tmp_path,compute=fake_supported,writer=writer,emergency_writer=fail,repair_writer=fail)
    assert not result["run_valid"] and not result["effective_direction_qualification"]
    assert not result["e334_handoff"]["eligible"] and result["publication"]["external_supervisor_and_root_byte_audit_required"]
    assert not (tmp_path/"output/result.json").exists()
    assert result["first_error"]["message"] == "terminal write failed"
    assert {x["stage"] for x in result["errors"]} >= {"publication","emergency_publication","terminal_repair_result.json"}


def test_run_preparation_consumes_science_boundary_and_blocks_loader(tmp_path):
    elapsed=[0.]
    def writer(path,value):
        subject.put_json(path,value)
        if path.name == "advance-ledger.json":elapsed[0]=201.
    result,deps=run_fixture(tmp_path,compute=fake_supported,writer=writer,clock=lambda:elapsed[0])
    assert not result["run_valid"] and not deps.load_called
    assert "deadline" in result["first_error"]["message"]


def test_primary_error_and_secondary_emergency_failure_are_retained(tmp_path):
    def compute(data,**kwargs):
        kwargs["checkpoint"]("failure-forward",{"q":np.full((4,42),np.nan)})
        raise ValueError("primary generated science failure")
    def writer(path,value):
        if path.name == "result.json":raise OSError("second publication failure")
        subject.put_json(path,value)
    original = subject.e332.write_json
    def emergency(path,value):
        if path.name == "publication-failure.json":raise OSError("third emergency failure")
        original(path,value)
    result,_=run_fixture(tmp_path,compute=compute,writer=writer,emergency_writer=emergency)
    assert result["first_error"]["message"] == "primary generated science failure"
    assert [x["stage"] for x in result["errors"]] == ["science","publication","emergency_publication"]
    assert np.isnan(subject.e332.load_npz(tmp_path/"output/failure-forward.npz",("q",))["q"]).all()
    assert subject.read(tmp_path/"output/result.json") == result
