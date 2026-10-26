"""K paid Generated qualification. Temporary ledgers and fees are synthetic."""
import copy
import hashlib
import json
import os

import pytest

from tests import banc_m3_checkpoint_fixture as historical_fixture
from tests import banc_m3_checkpoint_harness_fixture as f
from tools import banc_m3_checkpoint_harness_runtime as k
from tools import banc_m3_conserved_resource as r

_HISTORY_BASE = 'verification/banc-natural-gait-m3-20260919'
D19_DELTA_FIELDS={"direct_science_checkpoint_harness_s28_recovery","direct_science_blocking_holds"}


def test_math_l_checkpoint_ordered_accounting_negative_remaining(tmp_path):
    fixture = f.case(tmp_path, "generated-l-negative-accounting")
    fixture.admit()
    overrun=historical_fixture.d19_synthetic_overrun(f.d19_golden(k,fixture))
    fixture.python_charge(overrun, 124, suspended=True)
    state = k.validate(fixture.ledger, fixture.root)
    assert fixture.ledger["generated_shared_remaining_s"] < 0
    assert fixture.ledger["direct_science_generated_used_s"] == k.data(f.d19_golden(k,fixture)["independent_baseline"],fixture.root,524288)["direct_science_generated_used_s"] + overrun
    assert fixture.ledger["debits"][-1]["actual_outer_full_return_s"] == overrun
    assert fixture.ledger["direct_science_blocking_holds"] == {k.BOOTSTRAP_CALL: k.HOLD}
    f.negative(lambda: k.eligibility(fixture.ledger, state))
    _a, policy, grant = k.pinned_anchor(fixture.root)
    totals = k.totals(fixture.ledger, policy, grant, k.CAPS)
    for key, value in totals.items():
        assert fixture.ledger[key] == value and type(fixture.ledger[key]) is float
    assert totals["original_remaining_s"] == 12870.0 - grant["original_cumulative_before_s"] - sum(
        d["actual_outer_full_return_s"] for d in fixture.ledger["debits"][165:] if d["mode"] == "original")


def test_math_l_checkpoint_exact_prefix_types_and_legal_states(tmp_path):
    fixture = f.case(tmp_path, "generated-l-types")
    baseline = copy.deepcopy(fixture.ledger)
    assert k.validate(baseline, fixture.root)["phase"] == "charged"
    for key, value in (("original_jobs_running", True), ("direct_science_generated_limit_s", 1964),
                       ("original_remaining_s", 0.0), ("direct_science_policy", None), ("extra", 0)):
        f.negative(lambda key=key, value=value: k.validate(f.mutation(baseline, key, value), fixture.root))
    wrong = copy.deepcopy(baseline)
    wrong["debits"][0]["actual_outer_full_return_s"] += .001
    f.negative(lambda: k.validate(wrong, fixture.root))
    assert fixture.admit()["phase"] == "admitted"
    call = k.BOOTSTRAP_CALL
    token = {"schema": "flylab.m3.direct-science.lock-owner.v1", "call_id": call,
             "request": fixture.request_spec, "owner_pid": 411, "owner_creation_ticks": 412,
             "entry_counter_s": 1.0, "job_name": "Local\\generated-synthetic-L"}
    token_spec = fixture.put(r.SHARED_LOCK, token)
    identity = {"owner_pid": 411, "owner_creation_ticks": 412, "worker_pid": 413, "worker_creation_ticks": 414}
    fixture.ledger["direct_science_workers"] = {call: {"request": fixture.request_spec, "state": "begun", "lock_token": token_spec, **identity}}
    assert k.validate(fixture.ledger, fixture.root)["phase"] == "begun"
    fixture.ledger["direct_science_blocking_holds"] = {call: {"request": fixture.request_spec, "state": "ownership_unknown", **identity}}
    assert k.validate(fixture.ledger, fixture.root)["phase"] == "unknown"
    f.negative(lambda: k.eligibility(fixture.ledger, k.validate(fixture.ledger, fixture.root)))
    fixture.ledger["direct_science_workers"] = {}
    fixture.ledger["direct_science_blocking_holds"] = {}
    fixture.ledger["direct_science_reservations"] = {}
    fixture.ledger["direct_science_pending_actuals"] = {call: {"request": fixture.request_spec, "state": "actual_pending", "ownership_closed": True}}
    assert k.validate(fixture.ledger, fixture.root)["phase"] == "closed-pending"
    wrong = copy.deepcopy(fixture.ledger)
    wrong["direct_science_pending_actuals"][call]["ownership_closed"] = 1
    f.negative(lambda: k.validate(wrong, fixture.root))
    duplicate = b'{"a":1,"a":2}'
    from tools.verify_c_banc_m3_startup_ray import json_load
    f.negative(lambda: json_load(duplicate))
    f.negative(lambda: json_load(b'{"fee":NaN}'))



def test_gate_l_checkpoint_production_python_native_parity(tmp_path):
    v57 = f.case(tmp_path, "generated-l-parity")
    r.d19_fixed4_controls(k,v57)
    def catalog_refusal(entered,message):
        def unexpected_catalog(*args,**kwargs):
            entered.append(True)
            raise AssertionError(message)
        return unexpected_catalog
    # D4: fixed same318 continuity and raw no-write negatives in this existing node.
    assert len(k.PRODUCTION_REASONS) == 133 and set(k.PRODUCTION_REASONS) == set(range(13)) | set(range(14,20)) | set(range(22,36)) | set(range(38,138))
    assert set(k.PRODUCTION_REASONS).isdisjoint({-1,13,20,21,36,37,138,139,2147483647})
    q8 = k.refinement_before(v57.root)
    u9=r.raw_pin(k.data(k.data(k.CURRENT_CONTRACT,v57.root)['d19_fixed_history'],v57.root)['authenticated_actual'],v57.root,524288)
    assert k.refinement_input(q8, u9, v57.root) == q8
    assert set(v57.ledger)|({k.ALLOCATION_KEY} if k.data(k.CURRENT_CONTRACT,v57.root)['d19_predecessor_allocation_present'] else set()) == set(q8)
    assert {name for name in q8 if name not in v57.ledger or q8[name] != v57.ledger[name]} == D19_DELTA_FIELDS
    contract = k.data(k.CURRENT_CONTRACT, v57.root)
    predecessor = next(row["immutable_archive"] for row in contract["actual320"]["archives"] if row["logical_current_before"]["path"] == k.CONTRACT_PATH)
    middle = k.data(predecessor, v57.root)
    oldest_pin = next(row["immutable_archive"] for row in middle["actual320"]["archives"] if row["logical_current_before"]["path"] == k.CONTRACT_PATH)
    bridges = (k.data(oldest_pin, v57.root)["actual320"], middle["actual320"], contract["actual320"])
    # Execute only each authenticated current fixture's actual finite residual stage.
    from pathlib import Path
    from unittest.mock import patch
    review = k.data(v57.freeze["reviewed_source"], v57.root)
    final_pin = next(pin for pin in review["root_reviews"] if pin["path"] == k.BASE+"/s28-f18-root/residual03/final-v2.json")
    actual_final = k.data(final_pin, v57.root)
    actual_data = k.data
    # Source-authenticated fixture code runs its actual paid stage in isolated roots.
    p56 = []
    for bridge in bridges:
        cost = actual_data(bridge["old_cost"], v57.root)
        p56 += [bridge["old_request"], bridge["old_cost"], cost["trace"], cost["outer"]]
    assert len(p56) == len({pin["path"] for pin in p56}) == 12
    for module, modname in ((f,"k"),(historical_fixture,"h")):
        path = Path(module.__file__).resolve().relative_to(f.admission.ROOT).as_posix()
        assert r.raw_pin(v57.freeze["sources"][path],f.admission.ROOT,262144) == Path(module.__file__).read_bytes()
        assert module.Fixture._stage_paid.__code__.co_argcount == 3
        assert module.Fixture._stage_paid.__code__.co_varnames[:3] == ("self","current_contract",modname)
        z56 = tmp_path/("paid-stage-"+modname);z56.mkdir()
        receiver = object.__new__(module.Fixture);receiver.root=z56
        with patch.object(module.admission,"ROOT",v57.root):
            module.Fixture._stage_paid(receiver,contract,k)
            for pin in p56:
                assert (z56/pin["path"]).read_bytes() == r.raw_pin(pin,v57.root,k.reader_cap(pin["path"]))
                operand = v57.root/pin["path"];saved = operand.read_bytes()
                try:
                    operand.write_bytes(saved+b"\n")
                    with pytest.raises(ValueError):module.Fixture._stage_paid(receiver,contract,k)
                    operand.unlink()
                    with pytest.raises(FileNotFoundError):module.Fixture._stage_paid(receiver,contract,k)
                finally:operand.write_bytes(saved)
            for case in ("alias","hash","size","bool"):
                changed = copy.deepcopy(contract)
                p = changed["actual320"]["old_request"]
                if case == "alias":p["path"] += ".json"
                elif case == "hash":p["sha256"] = "0"*64
                elif case == "size":p["bytes"] += 1
                else:p["bytes"] = True
                with pytest.raises((ValueError,FileNotFoundError) if case == "alias" else ValueError):
                    module.Fixture._stage_paid(receiver,changed,k)
            for owner,pin in ((contract,k.CURRENT_CONTRACT),(middle,predecessor)):
                for logical in (k.CONTRACT_PATH,k.REVIEW_PATH):
                    for duplicate in (False,True):
                        changed = copy.deepcopy(owner)
                        rows = changed["actual320"]["archives"]
                        selected = [row for row in rows if row["logical_current_before"]["path"] == logical]
                        changed["actual320"]["archives"] = [row for row in rows if row not in selected]+(selected*2 if duplicate else [])
                        def decoded(p,root,cap=262144,changed=changed,pin=pin):
                            return copy.deepcopy(changed) if p == pin else actual_data(p,root,cap)
                        current = changed if owner is contract else contract
                        with patch.object(k,"data",side_effect=decoded), pytest.raises(ValueError):
                            module.Fixture._stage_paid(receiver,current,k)
                sf = owner["actual320"]["adoption_before_Source"]
                for case in ("association","layout"):
                    def source_decoded(p,root,cap=262144,sf=sf,case=case):
                        v = actual_data(p,root,cap)
                        if p == sf:
                            v = copy.deepcopy(v)
                            if case == "association":v["checkpoint_contract"]["sha256"] = "0"*64
                            else:v["sources"] = dict(reversed(list(v["sources"].items())))
                        return v
                    with patch.object(k,"data",side_effect=source_decoded), pytest.raises(ValueError):
                        module.Fixture._stage_paid(receiver,contract,k)
    for module, modname, reviewname in ((f,"k","review"),(historical_fixture,"h","current_review")):
        module_path = Path(module.__file__).resolve().relative_to(f.admission.ROOT).as_posix()
        assert r.raw_pin(v57.freeze["sources"][module_path],f.admission.ROOT,262144) == Path(module.__file__).read_bytes()
        assert module.Fixture._stage_residual.__code__.co_argcount == 3
        assert module.Fixture._stage_residual.__code__.co_varnames[:3] == ("self",reviewname,modname)
        for case in ("positive","duplicate-final","missing-final","missing-field","missing-pin","extra-pin","reordered","wrong-path","wrong-hash","wrong-size","bool-size","extra-key"):
            z56 = tmp_path/("residual-"+modname+"-"+case);z56.mkdir()
            r.fixture_text_write(z56,final_pin["path"],r.raw_pin(final_pin,f.admission.ROOT,262144))
            case_review = copy.deepcopy(review);v56 = copy.deepcopy(actual_final)
            if case == "duplicate-final":case_review["root_reviews"].append(copy.deepcopy(final_pin))
            elif case == "missing-final":case_review["root_reviews"].remove(final_pin)
            elif case == "missing-field":del v56["residual_provenance"]
            elif case == "missing-pin":v56["residual_provenance"].pop()
            elif case == "extra-pin":v56["residual_provenance"].append(copy.deepcopy(v56["residual_provenance"][0]))
            elif case == "reordered":v56["residual_provenance"].reverse()
            elif case == "wrong-path":v56["residual_provenance"][0]["path"] += "-alias"
            elif case == "wrong-hash":v56["residual_provenance"][0]["sha256"] = "0"*64
            elif case == "wrong-size":v56["residual_provenance"][0]["bytes"] += 1
            elif case == "bool-size":v56["residual_provenance"][0]["bytes"] = True
            elif case == "extra-key":v56["residual_provenance"][0]["extra"] = 1
            def residual_reader(spec,root,cap=262144,v56=v56):
                value=actual_data(spec,root,cap)
                return copy.deepcopy(v56) if spec == final_pin else value
            receiver=object.__new__(module.Fixture);receiver.root=z56
            with patch.object(k,"data",side_effect=residual_reader):
                if case == "positive":
                    module.Fixture._stage_residual(receiver,case_review,k)
                    for pin in actual_final["residual_provenance"]:assert (z56/pin["path"]).read_bytes()==r.raw_pin(pin,f.admission.ROOT,262144)
                else:
                    with pytest.raises(KeyError if case == "missing-field" else ValueError) as caught:
                        module.Fixture._stage_residual(receiver,case_review,k)
                    if case == "missing-field":assert caught.value.args == ("residual_provenance",)
    for drift in ("Source","review"):
        def report_drift(spec,root,cap=262144,drift=drift):
            value=actual_data(spec,root,cap)
            if spec["path"] == k.SOURCE_PATH and drift == "Source":value=copy.deepcopy(value);value["ROOT_review_cycle_complete"]=False
            if spec["path"] == k.REVIEW_PATH and drift == "review":value=copy.deepcopy(value);value["sources"][k.CONTRACT_PATH]=dict(k.CURRENT_CONTRACT,sha256="0"*64)
            return value
        with patch.object(k,"data",side_effect=report_drift):
            f.negative(lambda:f.archive_pair(final_pin))
            f.negative(lambda:historical_fixture.exact_source_operand(final_pin))
    assert len(q8["debits"]) == 390 and len(q8["direct_science_calls"]) == 272
    assert len(k._PAID_LEDGER_PATHS) == 9
    expected_original = tuple(_HISTORY_BASE+'/'+p for p in ('s28-f14-root/a318.json','s28-f13-root/a318.json','s28-f12-root/a318.json','s28-f11-root/a318.json','s28-f10-root/a318.json','s28-f9-root/a318.json','s28-f8-root/a318.json','s28-f7-root/a318.json','s28-f6-root/a318.json','s28-f5-root/a318.json','s28-f4-root/a318.json','s28-anchor318-before-recovery.json','s28-recovered318-before-bootstrap.json','parent-s28-bootstrap-settled319-before-allocation-20261009.json','parent-s28-qualification-settled323-before-body-20261009.json','parent-s28-body-settled324-before-audit-20261009.json','s28-bootstrap-settled319-HOLD-20261009.json','s27-anchor317-before-recovery.json','s27-recovered317-before-bootstrap.json','parent-s27-bootstrap-settled318-before-allocation-20261009.json','parent-s27-qualification-settled322-before-body-20261009.json','parent-s27-body-settled323-before-audit-20261009.json','s27-bootstrap-settled318-HOLD-20261009.json','s25-bootstrap-settled317-HOLD-20261009.json','s26-anchor317-before-recovery.json','s26-recovered317-before-bootstrap.json','parent-s26-bootstrap-settled318-before-allocation-20261009.json','parent-s26-qualification-settled322-before-body-20261009.json','parent-s26-body-settled323-before-audit-20261009.json','s26-bootstrap-settled318-HOLD-20261009.json','s24-bootstrap-settled316-HOLD-20261009.json','s25-anchor316-before-recovery.json','s25-recovered316-before-bootstrap.json','parent-s25-bootstrap-settled317-before-allocation-20261009.json','parent-s25-qualification-settled321-before-body-20261009.json','parent-s25-body-settled322-before-audit-20261009.json','s25-bootstrap-settled317-HOLD-20261009.json','s23-bootstrap-settled315-HOLD-20261009.json','s24-anchor315-before-recovery.json','s24-recovered315-before-bootstrap.json','parent-s24-bootstrap-settled316-before-allocation-20261009.json','parent-s24-qualification-settled320-before-body-20261009.json','parent-s24-body-settled321-before-audit-20261009.json','s24-bootstrap-settled316-HOLD-20261009.json','parent-l12-cold-settled306-HOLD-20261008.json','parent-l11-bootstrap-settled305-HOLD-20261008.json','parent-l10-cold-settled304-HOLD-20261008.json','parent-l9r2-bootstrap-settled303-HOLD-20261008.json','parent-l9-cold-settled302-HOLD-20261008.json','parent-l8-bootstrap-settled301-HOLD-20261008.json','parent-l7-cold-settled300-HOLD-20261008.json','s17-bootstrap-settled311-HOLD-20261009.json','s18-anchor311-after-repair20.json','parent-l12-bootstrap-admitted305-before-cold-settlement-20261008.json','parent-l10-bootstrap-admitted303-before-cold-settlement-20261008.json','parent-l9-bootstrap-admitted301-before-cold-settlement-20261008.json','parent-e358-execution-ledger.json','parent-l13r2-bootstrap-settled307-before-allocation-20261008.json','parent-l13r2-qualification-settled311-before-body-20261008.json','parent-l13r2-body-settled312-before-audit-20261008.json','parent-l7-bootstrap-admitted299-before-cold-settlement-20261008.json','parent-l6-math-settled299-HOLD-20261008.json','parent-l3-bootstrap-settled292-before-allocation-20261008.json','generated-l13r2-allocation-before.json','generated-l13r2-original-before.json','parent-s15-bootstrap-settled309-before-allocation-20261009.json','parent-s15-qualification-settled313-before-body-20261009.json','parent-s15-body-settled314-before-audit-20261009.json','parent-l13r2-bootstrap-settled307-HOLD-20261008.json','s14-anchor307-after-repair400.json','parent-s14-bootstrap-settled308-before-allocation-20261009.json','parent-s14-qualification-settled312-before-body-20261009.json','parent-s14-body-settled313-before-audit-20261009.json','s14-bootstrap-settled308-HOLD-20261009.json','s15-bootstrap-settled309-HOLD-20261009.json','parent-s16-bootstrap-settled310-before-allocation-20261009.json','parent-s16-qualification-settled314-before-body-20261009.json','parent-s16-body-settled315-before-audit-20261009.json','s16-anchor309-after-repair400-v2.json','s16-recovered309-before-bootstrap.json','parent-s17-bootstrap-settled311-before-allocation-20261009.json','parent-s17-qualification-settled315-before-body-20261009.json','parent-s17-body-settled316-before-audit-20261009.json','s16-bootstrap-settled310-HOLD-20261009.json','s17-recovered310-before-bootstrap.json','parent-s18-bootstrap-settled312-before-allocation-20261009.json','parent-s18-qualification-settled316-before-body-20261009.json','parent-s18-body-settled317-before-audit-20261009.json','s18-bootstrap-settled312-HOLD-20261009.json','s19-anchor312-after-repair200.json','s18-recovered311-before-bootstrap.json','parent-s19-bootstrap-settled313-before-allocation-20261009.json','parent-s19-qualification-settled317-before-body-20261009.json','parent-s19-body-settled318-before-audit-20261009.json','s19-bootstrap-settled313-HOLD-20261009.json','s22-anchor313-after-repair110.json','s19-recovered312-before-bootstrap.json','parent-s22-bootstrap-settled314-before-allocation-20261009.json','parent-s22-qualification-settled318-before-body-20261009.json','parent-s22-body-settled319-before-audit-20261009.json','s22-bootstrap-settled314-HOLD-20261009.json','s23-anchor314-after-grant2000.json','parent-s23-bootstrap-settled315-before-allocation-20261009.json','parent-s23-qualification-settled319-before-body-20261009.json','parent-s23-body-settled320-before-audit-20261009.json','s28-f15b-root/terminal319-before.json','s28-f15-root/admitted318-before.json','s28-f15-root/settlement-pre-backup.json','s28-f15-root/settlement-replace-backup.json','s28-f12-root/a318.json','s28-f14-root/a318.json','s27-bootstrap-settled318-HOLD-20261009.json','s28-f13-root/a318.json','s28-anchor318-before-recovery.json'))
    expected_paid = (_HISTORY_BASE+'/s28-f16-root/before/verification/banc-natural-gait-m3-20260919/parent-e358-execution-ledger.json', _HISTORY_BASE+'/s28-f15c-root/before-full4-canonical319.json', _HISTORY_BASE+'/s28-f15c-root/full4-admitted319.json', _HISTORY_BASE+'/s28-f18-root/b320.json', _HISTORY_BASE+'/s28-f18-root/a320.json', _HISTORY_BASE+'/s28-f18-root/t321.json')
    assert k._PAID_LEDGER_PATHS[:6] == expected_paid
    expected_paid += tuple(k.BASE+'/s28-f18-root/residual02/r3-'+p+'.json' for p in ('before-admission-canonical321','admitted321','terminal322'))
    assert k._OLD_LEDGER_PATHS == expected_original
    assert k._PAID_LEDGER_PATHS == expected_paid
    assert k.LEDGER_METADATA_PATHS==expected_original+expected_paid+historical_fixture.d19_r35_current_paths(k)
    assert k._OLD_LEDGER_PATHS is not k.LEDGER_METADATA_PATHS
    assert k.SEQUENCE == ('direct-science-20261009-contact-s28-bootstrap-r64', 'direct-science-20261009-contact-s28-r64-lint', 'direct-science-20261009-contact-s28-r64-math', 'direct-science-20261009-contact-s28-r64-gate-a', 'direct-science-20261009-contact-s28-r64-gate-b', 'contact-learning-20261009-s28-r64-body', 'contact-learning-20261009-s28-r64-body-audit')
    class OriginalPathSubclass(str): pass
    for path in expected_original:
        assert k.reader_cap(OriginalPathSubclass(path)) == 524288
    plan = k.plan(k.BOOTSTRAP_CALL, v57.freeze, v57.root)
    assert plan["argv"] == ["-m","pytest",*[k.TEST_MODULE+"::"+n for n in k.BOOTSTRAP_NAMES],"--tb=short","-q","--junitxml="+k.BASE+"/parent-contact-learning-s28-r64-bootstrap-junit.xml","--maxfail=1"]
    assert plan["output"] == k.BASE + "/parent-" + k.BOOTSTRAP_CALL
    for bridge in bridges:
        for field in ("before_full4", "admitted_full4", "terminal"):
            pin = bridge[field]
            assert pin["path"] in k._PAID_LEDGER_PATHS and k.reader_cap(pin["path"]) == 524288
            assert k.data(pin, v57.root, 524288) == json.loads(r.raw_pin(pin, v57.root, 524288))
            wrong = dict(pin, path=pin["path"]+".json")
            f.negative(lambda wrong=wrong: k.data(wrong, v57.root, 524288))
    class PathSubclass(str): pass
    for path in k._PAID_LEDGER_PATHS:
        assert k.reader_cap(PathSubclass(path)) == 262144
        f.negative(lambda path=path: k.data({'path':PathSubclass(path), 'bytes':1, 'sha256':"0"*64}, v57.root, 524288))
    for suffix, cap in ((k.REPLACEMENTS[0],131072),(k.REPLACEMENTS[1],131072),(k.BASE+"/parent-direct-science-account.ps1",393216)):
        path = k.BASE+"/s28-f18-root/before/"+suffix
        assert k.reader_cap(path) == cap
        pin = next(pin for pin in bridges[1]["raw_pins"] if pin["path"] == path)
        assert len(r.raw_pin(pin, v57.root, cap)) == pin["bytes"]
        f.negative(lambda pin=pin: k.data(pin, v57.root, 524288))
    ledger_raw = (v57.root / v57.ledger_path).read_bytes()
    f.negative(lambda: k.refinement_input(v57.ledger, u9, v57.root))  # immediate duplicate
    for stale in (u9 + b"\n", r.raw_pin(k.S15_PINS["anchor"], v57.root, 524288)):
        f.negative(lambda stale=stale: k.refinement_input(q8, stale, v57.root))
    for name, value in ((k.KEY, v57.recovery), ("direct_science_workers", {"foreign": {}}),
                        ("direct_science_blocking_holds", {"foreign": "HOLD"}),
                        ("direct_science_generated_limit_s", 5795)):
        f.negative(lambda name=name, value=value: k.refinement_input(f.mutation(q8, name, value), u9, v57.root))
    continuity_pins = [k.S15_PINS[name] for name in ("previous_recovery", "previous_source", "previous_contract", "previous_catalog", "technical_refusal", "refinement_anchor")]
    continuity_gate = k.data(k.S15_PINS["technical_refusal"], v57.root)
    continuity_pins += [continuity_gate[name] for name in ("actual_tool_return", "actual_known_compiler_type_refusal", "journal")]
    continuity_pins += [{'path':_HISTORY_BASE+path,'bytes':size,'sha256':digest} for path,size,digest in (('/s28-f14-root/a318.json',495559,'fadfb1604441f5c2cc481e15b07eeaf10a7a7529cedc8969bc496196ea221274'),('/s28-recovery10.json',1903,'27f8b420a0f7cdb3447aac416cd2ac71cf499ee8b5ea719ff05c968fd6f1184e'),('/s28-f14-root/s1-before.json',262140,'f1b802af060e385f03955b5cbba5830cc9ee42ccf757b9800016a45afd7cabfd'),('/s28-f14-root/c5-before.json',97623,'9690bd2ec7a65d33432071d09d9050c6d949c92d6a20852fc92072737f7dcb41'),('/s28-f14-root/h28-before.json',261912,'e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e'),('/s28-f13-root/actual-fast-warm-gate.json',2818,'042051f83aaa9c347c72d943e76d0958cf22afeccb5fd3c929bf0241cd5f5a1f'),('/s28-f13-root/s1-before.json',262140,'39b4910b21c51042a5d68f3da51061ce2d57ead51ec5ff4cdcf82554a6493ef0'),('/s28-f13-root/c5-before.json',74674,'3a87bc0f34b8a36ddd5153940f2b17fccbdd990b32459aea28ae92e51be58b0a'),('/s28-f12-root/actual-fast-warm-gate.json',5074,'9c9348756e9dacf4ee53a320e50acc22d25c8fe9028419dccbc35819356d7b7e'),('/s28-f12-root/actual-tool-return.json',1846,'71341a5233ecc031b45876ab24d1f7e4305ed79c738aa19f3efa71dc6c250162'),('/s28-f12-root/actual-layout-compatibility-diagnostic.json',1331,'efc4694c9651f842832549809e1613b9900ab71004da098bb82dc24c29ee8a6d'),('/s28-catalog-leaves-300021e7370a4dc8b163ef22f0245f31.jsonl',6393,'c4bb7e972e5ed5950e554983344f5d34849008f9a2a9a984a874d1ff8795926e'),('/s28-f13-root/actual-tool-return.json',2798,'e76c924ab89dd27752d2f8a1afe68bd6839f363471886c073b3b14fce0d5d2c0'),('/s28-f13-root/actual-compiler-type-refusal.json',1968,'448f06a882edd1757895f90dfd4bd1c7bfad38f414fcdda3909a0eacfa34e407'),('/s28-catalog-leaves-30944ee8fe084e2aac23de9d47e4de80.jsonl',496,'b3ebd169ebfc3f700ec42a0c3b7b9292f35b7561cb6f7d100c71bf3835669606'),('/s28-recovery11.json',1882,'0098806a84d0c18cf606482710d15e73dbe21defae360badba88a6c909db4c26'),('/s28-f13-root/a318.json',495559,'fadfb1604441f5c2cc481e15b07eeaf10a7a7529cedc8969bc496196ea221274'),)] + [{'path': _HISTORY_BASE+'/s27-bootstrap-settled318-HOLD-20261009.json', 'sha256': '70175b4bc56c19412b64500343ebc72bc538a7b3776dd60733d43efeeb59476e', 'bytes': 495417}] + [{'path':_HISTORY_BASE+path,'bytes':size,'sha256':digest} for path,size,digest in (('/e28.json',2443,'e5e284cbaa1f67f1dd6e14bdf3fe74c5f902d751c387f6850700323c23609c93'),('/s28-f13-root/h28-before.json',261912,'e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e'),('/s27-actual.json',3365,'fcf5b1fa42a8348fa52f85e0259cc3b3554be5d96206fd5e77f4fcc54e01eb94'),)]
    for continuity_pin in continuity_pins:
        operand = v57.root / continuity_pin["path"]
        saved = operand.read_bytes()
        try:
            k.recovery(v57.recovery, v57.root)  # warm proof before each fresh raw check
            operand.write_bytes(saved + b"\n")
            f.negative(lambda: k.recovery(v57.recovery, v57.root))
            operand.unlink()
            f.negative(lambda: k.recovery(v57.recovery, v57.root))
        finally:
            operand.write_bytes(saved)
        assert (v57.root / v57.ledger_path).read_bytes() == ledger_raw
    # D6: meaningful four direct provenance cold/warm refusals dominate catalog/cache.
    from unittest.mock import patch
    direct_contract = k.data(v57.freeze["sources"][k.CONTRACT_PATH],v57.root)
    direct_pins = tuple(direct_contract["provenance"][name] for name in ("root_design","astra_markdown","astra_json","root_reflection"))
    direct_ledger_raw = (v57.root / v57.ledger_path).read_bytes()
    for direct_pin in direct_pins:
        operand = v57.root / direct_pin["path"]
        saved = operand.read_bytes()
        for warm in (False,True):
            for bad in ("mutated","missing"):
                k._HISTORY_PROOF.clear()
                if warm:k.recovery(v57.recovery,v57.root)
                entered = []
                unexpected_catalog=catalog_refusal(entered,"direct provenance must refuse before catalog/backend or proof-cache lookup")
                try:
                    if bad == "mutated":operand.write_bytes(saved+b"\n")
                    else:operand.unlink()
                    with patch.object(k,"catalog_auth",side_effect=unexpected_catalog):
                        f.negative(lambda:k.recovery(v57.recovery,v57.root))
                    assert not entered and not k._HISTORY_PROOF and not k._HISTORY_BUSY
                    assert (v57.root / v57.ledger_path).read_bytes() == direct_ledger_raw
                finally:operand.write_bytes(saved)
    actual_direct_reader = k.data
    for warm in (False,True):
        for mismatch in ("Source-contract","review-contract","contract-alias","contract-oversized","contract-missing","contract-review"):
            k._HISTORY_PROOF.clear()
            if warm:k.recovery(v57.recovery,v57.root)
            entered = []
            def mismatched_direct_reader(spec,root,cap=262144,mismatch=mismatch):
                value = actual_direct_reader(spec,root,cap)
                if mismatch == "Source-contract" and spec["path"] == k.SOURCE_PATH:
                    value = copy.deepcopy(value);value["checkpoint_contract"] = k.S15_PINS["previous_contract"]
                elif mismatch == "review-contract" and spec["path"] == k.REVIEW_PATH:
                    value = copy.deepcopy(value);value["sources"][k.CONTRACT_PATH] = k.S15_PINS["previous_contract"]
                elif mismatch.startswith("contract-") and spec["path"] == k.CONTRACT_PATH:
                    value = copy.deepcopy(value);p = value["provenance"]
                    if mismatch == "contract-alias":p["root_design"]["path"] += "-alias"
                    elif mismatch == "contract-oversized":p["root_design"]["bytes"] = 262145
                    elif mismatch == "contract-missing":p.pop("root_design")
                    else:p["root_coverage"] = copy.deepcopy(p["root_design"])
                return value
            unexpected_catalog=catalog_refusal(entered,"Source review c5 disagreement reached catalog")
            with patch.object(k,"data",side_effect=mismatched_direct_reader),patch.object(k,"catalog_auth",side_effect=unexpected_catalog):
                f.negative(lambda:k.recovery(v57.recovery,v57.root))
            assert not entered and not k._HISTORY_PROOF and not k._HISTORY_BUSY
            assert (v57.root / v57.ledger_path).read_bytes() == direct_ledger_raw
    # Every current governance operand is fresh before first catalog and cache return.
    current_review=k.data(k.data(v57.source,v57.root)["reviewed_source"],v57.root)
    for warm in (False,True):
        for governance_pin in (*current_review["root_reviews"],*current_review["sol_reports"]):
            for damage in ("missing","mutated"):
                k._HISTORY_PROOF.clear()
                if warm:k.recovery(v57.recovery,v57.root)
                governance_path=v57.root/governance_pin["path"];saved_governance=governance_path.read_bytes();entered=[]
                try:
                    if damage=="missing":governance_path.unlink()
                    else:governance_path.write_bytes(saved_governance+b"\n")
                    with patch.object(k,"catalog_auth",side_effect=lambda *a,entered=entered,**kw:entered.append(True)):
                        f.negative(lambda:k.recovery(v57.recovery,v57.root))
                    assert not entered and not k._HISTORY_PROOF and not k._HISTORY_BUSY
                    assert (v57.root/v57.ledger_path).read_bytes()==direct_ledger_raw
                finally:governance_path.write_bytes(saved_governance)
    # D15B: authored within the same fixed4 node; not an unpaid probe.
    import time

    from tools import banc_m3_startup_ray_supervisor as supervisor
    v57.admit()
    admitted_raw=(v57.root/v57.ledger_path).read_bytes()
    admitted_spec=r.text_spec(v57.ledger_path,v57.root,524288)
    failing=[]
    def fail_dispatch(*args,**kwargs):
        failing.append(True);raise NameError("injected source-fence dependency")
    with patch.object(k,"validate",side_effect=fail_dispatch),patch.object(k,"source_fence",side_effect=fail_dispatch),patch.object(k,"plan",side_effect=AssertionError("no reporting plan")),patch.object(k,"roster",side_effect=AssertionError("no reporting roster")),patch.object(k,"identity",side_effect=AssertionError("no reporting identity")),patch.object(k,"original_metadata",side_effect=AssertionError("no reporting Original")):
        try:k.dispatch(v57.ledger,v57.q,v57.request_spec,v57.root,phase="admitted")
        except NameError as original:
            outer=k.initial_refusal(v57.root,v57.request_spec,v57.q,admitted_spec,admitted_raw,original,entry=time.perf_counter())
        else:raise AssertionError("injected dispatch did not fail")
    assert len(failing)==1 and outer["owner_pid"]==os.getpid() and outer["owner_creation_ticks"]==supervisor.creation_ticks(os.getpid())
    assert outer["first_error"]["type"]=="NameError" and outer["worker_pid"] is None and outer["worker_creation_state"]=="NO_CREATE_ATTEMPT"
    assert outer["reservation_released"] is outer["all_owned_ended"] is True and not outer["stdout_eof"] and not outer["stderr_eof"]
    assert (v57.root/v57.ledger_path).read_bytes()==admitted_raw
    # R1 scoped reporting assertion; global historical f.negative stays unchanged.
    def reporting_refusal(invoke, original, secondary_type, secondary_label, output_path, *, allow_empty_output=False):
        before_ledger=(v57.root/v57.ledger_path).read_bytes()
        request_path=v57.root/v57.request_spec["path"];before_request=request_path.read_bytes()
        lock_path=v57.root/r.SHARED_LOCK;before_lock=lock_path.read_bytes() if lock_path.exists() else None
        def output_snapshot():
            if not output_path.exists():return None
            assert output_path.is_dir()
            return tuple((p.relative_to(output_path).as_posix(),p.is_dir(),None if p.is_dir() else p.read_bytes()) for p in sorted(output_path.rglob("*")))
        before_output=output_snapshot()
        try:invoke(original)
        except RuntimeError as uncertain:
            assert uncertain.__cause__ is original
            assert str(uncertain).startswith("original "+type(original).__name__+": "+str(original)+"; independent refusal uncertainty ")
            assert "uncertainty "+secondary_type.__name__+": "+secondary_label in str(uncertain)
        else:raise AssertionError("uncertain reporting refusal did not raise RuntimeError")
        assert (v57.root/v57.ledger_path).read_bytes()==before_ledger
        assert request_path.read_bytes()==before_request
        assert (lock_path.read_bytes() if lock_path.exists() else None)==before_lock
        # Full ledger bytes include admission, reservations, owners, counters and fee maps.
        if before_output is not None:assert output_snapshot()==before_output
        elif allow_empty_output:
            # Publication may leave its fresh empty directory; never claim a closed outer result.
            assert output_snapshot() in (None,()) and not (output_path/"outer.json").exists()
        else:assert not output_path.exists()
    reporting_refusal(lambda original:k.initial_refusal(v57.root,v57.request_spec,v57.q,admitted_spec,admitted_raw,original,entry=time.perf_counter()),NameError("injected"),ValueError,"output",v57.root/v57.q["output"])
    # Reporting negatives preserve admission even if the original validator is broken.
    import shutil
    shutil.rmtree(v57.root/v57.q["output"])
    # First Source dependency failure is retained without a second validation attempt.
    source_calls=[]
    def broken_source(*args,**kwargs):
        source_calls.append(True);raise NameError("injected first Source fence failure")
    def validator_enters_source(*args,**kwargs):return k.source_fence(v57.source,v57.root)
    with patch.object(k,"validate",side_effect=validator_enters_source),patch.object(k,"source_fence",side_effect=broken_source),patch.object(k,"original_metadata",side_effect=AssertionError("reporting Original read")):
        try:k.dispatch(v57.ledger,v57.q,v57.request_spec,v57.root,phase="admitted")
        except NameError as first_source:
            source_outer=k.initial_refusal(v57.root,v57.request_spec,v57.q,admitted_spec,admitted_raw,first_source,entry=time.perf_counter())
        else:raise AssertionError("Source failure did not propagate")
    assert source_calls==[True] and source_outer["first_error"]["message"]=="injected first Source fence failure"
    assert source_outer["worker_pid"] is None and (v57.root/v57.ledger_path).read_bytes()==admitted_raw
    shutil.rmtree(v57.root/v57.q["output"])
    for bad in ("raw-drift","foreign-HOLD","foreign-worker","wrong-reservation","unknown-actor","physical-lock","publication"):
        live=copy.deepcopy(v57.ledger)
        if bad=="foreign-HOLD":live["direct_science_blocking_holds"]={"foreign":"HOLD"}
        if bad=="foreign-worker":live["direct_science_workers"]={"foreign":{}}
        if bad=="wrong-reservation":live["direct_science_reservations"][k.BOOTSTRAP_CALL]["reserved_s"]=179
        if bad=="raw-drift":
            (v57.root/v57.ledger_path).write_bytes(admitted_raw+b"\n");wrong_spec=admitted_spec;wrong_raw=admitted_raw
        else:
            v57.ledger=live;v57.publish();wrong_spec=r.text_spec(v57.ledger_path,v57.root,524288);wrong_raw=(v57.root/v57.ledger_path).read_bytes()
        if bad=="physical-lock":v57.put(r.SHARED_LOCK,{"foreign":"owner"})
        try:
            if bad=="unknown-actor":
                with patch.object(supervisor,"creation_ticks",return_value=0):reporting_refusal(lambda original,wrong_spec=wrong_spec,wrong_raw=wrong_raw:k.initial_refusal(v57.root,v57.request_spec,v57.q,wrong_spec,wrong_raw,original,entry=time.perf_counter()),NameError("original retained"),ValueError,"actor",v57.root/v57.q["output"])
            elif bad=="publication":
                with patch.object(supervisor,"durable",side_effect=OSError("publication uncertainty")):
                    reporting_refusal(lambda original,wrong_spec=wrong_spec,wrong_raw=wrong_raw:k.initial_refusal(v57.root,v57.request_spec,v57.q,wrong_spec,wrong_raw,original,entry=time.perf_counter()),NameError("original retained"),OSError,"publication uncertainty",v57.root/v57.q["output"],allow_empty_output=True)
            else:reporting_refusal(lambda original,wrong_spec=wrong_spec,wrong_raw=wrong_raw:k.initial_refusal(v57.root,v57.request_spec,v57.q,wrong_spec,wrong_raw,original,entry=time.perf_counter()),NameError("original retained"),ValueError,{"raw-drift":"file","foreign-HOLD":"layout","foreign-worker":"layout","wrong-reservation":"scalar","physical-lock":"lock"}[bad],v57.root/v57.q["output"])
            assert not (v57.root/v57.q["output"]/"outer.json").exists()
            assert (v57.root/v57.ledger_path).read_bytes()==(admitted_raw+b"\n" if bad=="raw-drift" else wrong_raw)
        finally:
            (v57.root/v57.ledger_path).write_bytes(admitted_raw)
            v57.ledger=k.data(admitted_spec,v57.root,524288)
            if (v57.root/r.SHARED_LOCK).exists():(v57.root/r.SHARED_LOCK).unlink()
            if (v57.root/v57.q["output"]).exists():shutil.rmtree(v57.root/v57.q["output"])
    # Original pre-create reporting has BOTH reservation maps empty; no science read.
    saved_q,saved_request,saved_ledger=copy.deepcopy(v57.q),copy.deepcopy(v57.request_spec),copy.deepcopy(v57.ledger)
    original_call=k.SEQUENCE[5]
    original_limits=dict(zip(("outer_s","child_s","cleanup_s"),k.CAP_TRIPLES[5],strict=True))
    original_limits.update(host_bytes=8589934592,output_cap_bytes=26214400)
    v57.q={"schema":"flylab.m3.direct-science.supervision.v1","call_id":original_call,"mode":"original","role":"contact_candidate_body","source_freeze":v57.source,"limits":original_limits,"output":k.BASE+"/parent-"+original_call+"/result-supervision","ledger_path":v57.ledger_path,"shared_original_lock_path":r.SHARED_LOCK,"grant":r.J_PIN,"science_request":{"path":k.BASE+"/NEVER_READ_ORIGINAL_METADATA.json","bytes":1,"sha256":"0"*64}}
    v57.request_spec=v57.put(k.BASE+"/D15B-authored-original-refusal-request.json",v57.q)
    v57.ledger=copy.deepcopy(saved_ledger)
    v57.ledger["direct_science_calls"][k.BOOTSTRAP_CALL]["state"]="charged"
    v57.ledger["direct_science_calls"][original_call]={"request":v57.request_spec,"state":"admitted","mode":"original","reserved_s":original_limits["outer_s"]}
    for owner_key in r.OWNER_KEYS:v57.ledger[owner_key]={}
    v57.publish();original_raw=(v57.root/v57.ledger_path).read_bytes();original_pin=r.text_spec(v57.ledger_path,v57.root,524288)
    original_output=v57.root/v57.q["output"]
    try:
        with patch.object(k,"validate",side_effect=AssertionError("second validator")),patch.object(k,"source_fence",side_effect=AssertionError("second Source fence")),patch.object(k,"original_metadata",side_effect=AssertionError("Original metadata read")),patch.object(k,"data",side_effect=AssertionError("metadata dependency read")),patch.object(r,"read_pin",side_effect=AssertionError("Original/NPZ read")),patch.object(r,"raw_pin",side_effect=AssertionError("Original/NPZ raw read")):
            original_outer=k.initial_refusal(v57.root,v57.request_spec,v57.q,original_pin,original_raw,NameError("Original validation failure"),entry=time.perf_counter())
        assert original_outer["worker_pid"] is None and original_outer["worker_creation_state"]=="NO_CREATE_ATTEMPT" and original_outer["first_error"]["message"]=="Original validation failure"
        assert original_outer["reservation_released"] is original_outer["all_owned_ended"] is True
        assert not original_outer["stdout_eof"] and not original_outer["stderr_eof"] and (v57.root/v57.ledger_path).read_bytes()==original_raw
        shutil.rmtree(original_output)
        for reservation_key in ("direct_science_reservations","startup_ray_original_reservations"):
            wrong=copy.deepcopy(v57.ledger);wrong[reservation_key]={original_call:{"request":v57.request_spec,"reserved_s":original_limits["outer_s"]}}
            wrong_raw=json.dumps(wrong,ensure_ascii=True).encode();(v57.root/v57.ledger_path).write_bytes(wrong_raw);wrong_pin=r.text_spec(v57.ledger_path,v57.root,524288)
            reporting_refusal(lambda original,wrong_pin=wrong_pin,wrong_raw=wrong_raw:k.initial_refusal(v57.root,v57.request_spec,v57.q,wrong_pin,wrong_raw,original,entry=time.perf_counter()),NameError("Original validation failure"),ValueError,"layout",original_output)
            assert not original_output.exists() and (v57.root/v57.ledger_path).read_bytes()==wrong_raw
            (v57.root/v57.ledger_path).write_bytes(original_raw)
    finally:
        if original_output.exists():shutil.rmtree(original_output)
        v57.q,v57.request_spec,v57.ledger=saved_q,saved_request,saved_ledger
        v57.publish()
    # R1 positive baseline and each reached mutation are independent of negative success.
    fixed_bridge=k.terminal_bridge(v57.root)
    positive_terminal_baseline=k.terminal_baseline(v57.root)
    for decoded_bad in ("cost","receipt","actor","duplicate-old-fee","tail-index","unknown-partial"):
        original_reader=k.data
        target=fixed_bridge[decoded_bad]["path"] if decoded_bad in ("cost","receipt","actor") else fixed_bridge["terminal"]["path"]
        reached=[];mutated=[]
        def decoded_corruption(pin,root,cap=262144,original_reader=original_reader,target=target,reached=reached,decoded_bad=decoded_bad,mutated=mutated):
            value=copy.deepcopy(original_reader(pin,root,cap))
            if pin["path"]==target:
                reached.append((pin["path"],cap));before_value=copy.deepcopy(value)
                if decoded_bad=="cost":value["reported_tool_wall_s"]=0.4983085
                elif decoded_bad=="receipt":value["cost"]["sha256"]="0"*64
                elif decoded_bad=="actor":value["pid"]=26437
                elif decoded_bad=="duplicate-old-fee":value["debits"].append(copy.deepcopy(value["debits"][-1]))
                elif decoded_bad=="tail-index":value["debits"][-1],value["debits"][-2]=value["debits"][-2],value["debits"][-1]
                else:value.pop("direct_science_s28_exec_precreate_failure")
                assert value!=before_value
                mutated.append((pin["path"],decoded_bad))
            return value
        intended_guard={"cost":"mixed fee","receipt":"scalar","actor":"scalar","duplicate-old-fee":"319/201","tail-index":"bits","unknown-partial":"nine"}[decoded_bad]
        try:
            with patch.object(k,"data",side_effect=decoded_corruption):k.terminal_baseline(v57.root)
        except ValueError as refusal:
            assert reached==[(target,262144 if decoded_bad in ("cost","receipt","actor") else 524288)]
            assert mutated==[(target,decoded_bad)] and intended_guard in str(refusal)
        else:raise AssertionError("decoded mutation did not reach its intended refusal")
        # TypeError is deliberately not caught: a reader/default failure must fail this control.
        assert (v57.root/v57.ledger_path).read_bytes()==admitted_raw
        k.whole_exact(k.terminal_baseline(v57.root),positive_terminal_baseline)
    f.d19_python_history_raw(k,v57,admitted_raw)
    import shutil
    if (v57.root/v57.q["output"]).exists():shutil.rmtree(v57.root/v57.q["output"])
    v57.ledger=copy.deepcopy(k.recovered_snapshot(v57.recovery,v57.root));v57.publish()
    event_path = v57.root / k.RECOVERY_PATH
    event_raw = event_path.read_bytes()
    event = k.data(v57.recovery, v57.root)
    for name, value in (("previous_recovery", v57.recovery), ("old_source", v57.source),
                        ("audit", k.S15_PINS["previous_recovery"]), ("reset", True),
                        ("new_resource_grant_s", 1)):
        try:
            event_path.write_bytes(json.dumps(f.mutation(event, name, value), ensure_ascii=True).encode("utf8"))
            wrong_spec = r.text_spec(k.RECOVERY_PATH, v57.root, 262144)
            f.negative(lambda wrong_spec=wrong_spec: k.recovery(wrong_spec, v57.root))
        finally:
            event_path.write_bytes(event_raw)
        assert (v57.root / v57.ledger_path).read_bytes() == ledger_raw
    cached = k.recovery(v57.recovery, v57.root)
    expected_proof = copy.deepcopy(cached)
    historical_entry = f.d19_history_entry(k,v57)
    historical_value = next(iter(historical_entry.values()))
    cached[0]["source_freeze"]["bytes"] = 1
    cached[1]["debits"][0]["cost_id"] = "returned-mutation"
    cached[2]["returned"] = True
    cached[3]["returned"] = [None, False, 1.25]
    r.exact(k.recovery(v57.recovery, v57.root), expected_proof, "detached typed proof on every return")
    for warm in (False, True):
        k._HISTORY_PROOF.clear()
        if warm:
            k._HISTORY_PROOF.update(copy.deepcopy(historical_entry))
            r.exact(k.recovery(v57.recovery, v57.root), expected_proof, "warm snapshot control ready")
        snapshot = k.recovered_snapshot(copy.deepcopy(v57.recovery), v57.root)
        expected_snapshot = f.d19_snapshot_expected(k,v57,v57.recovery)
        r.exact(snapshot, expected_snapshot, "whole detached cold/warm snapshot")
        k._HISTORY_PROOF.clear()
        if warm:
            k._HISTORY_PROOF.update(copy.deepcopy(historical_entry))
            r.exact(k.recovery(v57.recovery, v57.root), expected_proof, "warm validation control ready")
        live = copy.deepcopy(snapshot)
        k.validate(live, v57.root)
        r.exact(live, expected_snapshot, "whole live input preserved by cold/warm validation")
        r.d19_whole(tuple(k._HISTORY_PROOF.items()),tuple((historical_entry if warm else {}).items()))
        r.exact(k.recovery(v57.recovery, v57.root), expected_proof, "validation preserves next whole recovery")
        r.exact(k.terminal_baseline(v57.root)[0], historical_value[0], "validation preserves whole original baseline")
        snapshot["debits"][0]["cost_id"] = "snapshot-array-mutation"
        snapshot["direct_science_calls"][next(iter(snapshot["direct_science_calls"]))]["state"] = "snapshot-map-mutation"
        snapshot[k.KEY]["bytes"] = 1
        snapshot[k.REPAIR_KEY]["bytes"] = 1
        r.exact(k.recovery(v57.recovery, v57.root), expected_proof, "snapshot mutations preserve next whole recovery")
        r.exact(k.data(k.data(k.CURRENT_CONTRACT, v57.root)["actual320"]["terminal"], v57.root, 524288), historical_value[0], "snapshot mutations preserve pinned whole baseline")
        r.exact(k.terminal_baseline(v57.root)[0], historical_value[0], "cold/warm snapshot preserves whole typed anchor")
    from unittest.mock import patch
    actual_reader = k.data
    for warm in (False, True):
        k._HISTORY_PROOF.clear()
        if warm:
            r.exact(k.recovery(v57.recovery, v57.root), expected_proof, "warm reentry control ready")
        entered = []
        def reenter(spec, root, cap=262144, entered=entered):
            if not entered:
                entered.append(True)
                assert k._HISTORY_BUSY
                f.negative(lambda: k.recovery(v57.recovery, v57.root))
                assert k._HISTORY_BUSY  # Inner rejection retains the outer capability.
            return actual_reader(spec, root, cap)
        with patch.object(k, "data", side_effect=reenter):
            r.exact(k.recovery(v57.recovery, v57.root), expected_proof, "actual cold/warm reader reentry rejected")
        assert entered == [True] and not k._HISTORY_BUSY
        r.exact(k.recovery(v57.recovery, v57.root), expected_proof, "guard cleanup positive return")
    f.d19_python_catalog_raw(k,v57,ledger_raw)
    historical_fixture.d19_opaque_controls(k,v57,ledger_raw)
    empty_root = tmp_path / "different-history-root"
    empty_root.mkdir()
    f.negative(lambda: k.recovery(v57.recovery, empty_root))
    for value in (True, float(v57.recovery["bytes"]), v57.recovery["path"] + "-alias"):
        field = "path" if type(value) is str else "bytes"
        f.negative(lambda value=value, field=field: k.recovery(f.mutation(v57.recovery, field, value), v57.root))
    f.negative(lambda: k._history_operand(k.FAIL_REPORT, v57.root, True, "json"))
    f.negative(lambda: k._history_operand(k.FAIL_REPORT, v57.root, 262144, "alias"))
    catalog = k.data(k.S15_PINS["catalog"], v57.root)
    positive_policy = k.catalog_auth(catalog, v57.root)
    altered = []
    for field, value in (("cap", True), ("cap", 65536), ("kind", "alias")):
        wrong = copy.deepcopy(catalog)
        wrong["rows"][0][field] = value
        altered.append(wrong)
    wrong = copy.deepcopy(catalog)
    wrong["rows"][0]["pin"]["path"] += "-alias"
    altered.append(wrong)
    wrong = copy.deepcopy(catalog)
    wrong["rows"] = wrong["rows"][1:]
    altered.append(wrong)
    wrong = copy.deepcopy(catalog)
    wrong["rows"][0], wrong["rows"][1] = wrong["rows"][1], wrong["rows"][0]
    altered.append(wrong)
    for wrong in altered:
        f.negative(lambda wrong=wrong: k.catalog_auth(wrong, v57.root))
        r.exact(k.catalog_auth(catalog, v57.root), positive_policy, "whole catalog positive after mutation")
    empty_pin = next(row["pin"] for row in catalog["rows"] if row["pin"]["path"] == k.EMPTY_PATHS[1])
    f.negative(lambda: k._history_operand(dict(empty_pin,path=k.BASE+"/empty-alias.bin"), v57.root, 262144, "raw0"))
    f.negative(lambda: k._history_operand(k.FAIL_REPORT, v57.root, 524288, "raw"))
    v57.admit()
    expected = k.validate(v57.ledger, v57.root)
    lock_value = {'schema': 'flylab.m3.direct-science.lock-owner.v1', 'fixture_synthetic': True}
    lock_spec = v57.put(r.SHARED_LOCK, lock_value)
    assert k.data(lock_spec, v57.root, 16384) == lock_value
    foreign_lock = v57.put(k.BASE + '/generated-l-foreign-metadata.lock', lock_value)
    f.negative(lambda: k.data(foreign_lock, v57.root, 16384))
    from unittest.mock import patch
    with patch.object(r, "j_history", side_effect=AssertionError("legacy recursion")), patch.object(r, "j_lint_cleanup", side_effect=AssertionError("legacy recursion")):
        assert r.current_caps(v57.ledger, v57.root, lambda *_: (_ for _ in ()).throw(AssertionError("legacy fallback"))) == k.CAPS
    original = f.case(tmp_path, 'generated-l-original-metadata')
    assert original.original_evidence()[3] is True
    bad_science, bad_cost = [], []
    for index, (key, value) in enumerate((('schema', 'wrong'), ('mode', 'generated'), ('role', 'contact_body_audit'),
            ('call_id', k.CALLS[0] + '-alias'), ('output', original.science['output'] + '-alias'),
            ('inputs', original.science['contract']), ('grant', original.science['contract']),
            ('roster', original.science['contract']), ('local_debit_offset', True), ('resource_increment_s', 1),
            ('historical_manifest', original.science['contract']), ('source_freeze', k.J_FREEZE))):
        wrong = f.mutation(original.science, key, value)
        f.negative(lambda wrong=wrong: k.original_metadata(wrong, original.q, original.root))
        bad_science.append(original.put(k.BASE + '/generated-l-science-negative-' + str(index) + '.json', wrong))
    wrong = copy.deepcopy(original.science)
    wrong['limits']['internal_s'] += 1
    f.negative(lambda: k.original_metadata(wrong, original.q, original.root))
    bad_science.append(original.put(k.BASE + '/generated-l-science-internal-negative.json', wrong))
    trace, outer = k.data(original.trace_spec, original.root), k.data(original.outer_spec, original.root)
    for index, key in enumerate(('request', 'supervisor_request', 'role', 'grant', 'inputs', 'roster', 'contract', 'body_retry', 'local_debit_offset', 'resource_increment_s')):
        wrong = f.mutation(original.cost, key, 164 if key == 'local_debit_offset' else (1 if key == 'resource_increment_s' else 'tampered'))
        f.negative(lambda wrong=wrong: k.fee_values(wrong, original.q, original.request_spec, trace, outer, original.root))
        pin = original.put(original.cost_spec['path'], wrong)
        debit = dict(original.debit, descriptor=pin)
        f.negative(lambda pin=pin, debit=debit: k.fee_proof(pin, debit, original.source, original.root))
        bad_cost.append(original.put(k.BASE + '/generated-l-cost-negative-' + str(index) + '.json', wrong))
    original.put(original.cost_spec['path'], original.cost)
    shell_trace, shell_outer = copy.deepcopy(trace), copy.deepcopy(outer)
    shell_trace.update(exit_code=1, exec_args={'cmd': '& .venv/Scripts/python.exe -m tools.banc_m3_startup_ray_supervisor --request ' + original.request_spec['path']})
    shell_outer.update(exit_code=124, worker_exit_code=124, timeout=True, first_error={'type': 'TimeoutError'})
    shell_cost = dict(original.cost, exit_code=1, native_supervisor_exit_code=124,
                      actual_collector_shell_exit_code=1, shell_deadline_status_pair_observed=True)
    k.fee_values(shell_cost, original.q, original.request_spec, shell_trace, shell_outer, original.root)
    shell_bad = []
    for key, value in (('native_supervisor_exit_code', 1), ('actual_collector_shell_exit_code', 124), ('shell_deadline_status_pair_observed', 1)):
        wrong = f.mutation(shell_cost, key, value)
        f.negative(lambda wrong=wrong: k.fee_values(wrong, original.q, original.request_spec, shell_trace, shell_outer, original.root))
        shell_bad.append(wrong)
    shell_pin = original.put(k.BASE + '/generated-l-shell-pair-values.json', {'trace': shell_trace, 'outer': shell_outer, 'valid': shell_cost, 'invalid': shell_bad})
    science_names = ','.join("'" + x['path'] + "'" for x in bad_science)
    cost_names = ','.join("'" + x['path'] + "'" for x in bad_cost)
    native = original.native("""
$q=K-Read (K-BoundedPin 'REQUEST');$s=K-Read $q.science_request;$null=K-OriginalMetadata $s $q
$cost=K-Read (K-BoundedPin 'COST');$d=@{cost_id=$q.call_id;mode='original';actual_outer_full_return_s=1.0;actual_returned_exit_code=[long]0;descriptor=(K-BoundedPin 'COST')};$null=K-FeeProof $d.descriptor $d $q.source_freeze
$t=K-Read $cost.trace;$o=K-Read $cost.outer
foreach($name in @(SCIENCE)){$rejected=$false;try{$null=K-OriginalMetadata (K-Read (K-BoundedPin $name)) $q}catch{$rejected=$true};Need $rejected 'Actual Native common science metadata negative'}
foreach($name in @(COSTS)){$rejected=$false;try{K-FeeValues (K-Read (K-BoundedPin $name)) $q (K-BoundedPin 'REQUEST') $t $o}catch{$rejected=$true};Need $rejected 'Actual Native admitted cost value negative'}
$pair=K-Read (K-BoundedPin 'SHELL');Need (K-ExitMatch $pair.trace $pair.outer (K-BoundedPin 'REQUEST')) 'Actual immutable shell deadline pair';K-FeeValues $pair.valid $q (K-BoundedPin 'REQUEST') $pair.trace $pair.outer
foreach($value in $pair.invalid){$rejected=$false;try{K-FeeValues $value $q (K-BoundedPin 'REQUEST') $pair.trace $pair.outer}catch{$rejected=$true};Need $rejected 'Exact Native observed shell-pair field'}
Write-Output 'PASS synthetic Original immutable metadata and cost value parity'
""".replace('REQUEST', original.request_spec['path']).replace('COSTS', cost_names).replace('COST', original.cost_spec['path']).replace('SCIENCE', science_names).replace('SHELL', shell_pin['path']))
    f.observed_pass(native)
    negatives = []
    for index, (key, value) in enumerate((("original_jobs_running", True), ("direct_science_generated_limit_s", 1964),
                                         ("startup_ray_original_grants", {}), ("extra", 0))):
        wrong = f.mutation(v57.ledger, key, value)
        f.negative(lambda wrong=wrong: k.validate(wrong, v57.root))
        negatives.append(v57.put(k.BASE + "/generated-l-native-negative-" + str(index) + ".json",
                                     {"field": key, "value": value}))
    paths = ",".join("'" + pin['path'] + "'" for pin in negatives)
    old_source = k.data(k.SOURCE11, v57.root)
    old_account_pin = dict(old_source["sources"][k.BASE + "/parent-direct-science-account.ps1"],
                           path=k.BASE + "/l12-before/" + k.BASE + "/parent-direct-science-account.ps1")
    old_account = k.bounded_source(old_account_pin, v57.root, 262144).decode("utf8")
    begin, end = old_account.index("function Same("), old_account.index("function Write-New(")
    old_same = old_account[begin:end].replace("Same(", "HistoricalSame(").replace(";Same ", ";HistoricalSame ").replace("{Same ", "{HistoricalSame ")
    iterative_case = historical_fixture.d19_r41_reference() + old_same + "\n$script:NewSameEntered=0;$script:ManagedSameEntered=0\n" + r'''
function PrimitiveCompare($a,$b){
$oldError=$null;$newError=$null;try{HistoricalSame $a $b 'matrix'}catch{$oldError=$_.Exception.Message}
$script:NewSameEntered++
try{Same $a $b 'matrix'}catch{$newError=$_.Exception.Message}
Need ($oldError -ceq $newError) ('Old/new Same first label '+$oldError+' / '+$newError)
$managedError=$null;$script:ManagedSameEntered++
try{K-Same $a $b 'matrix'}catch{$managedError=$_.Exception.Message}
Need ($newError -ceq $managedError) ('Generic/wrapper same19 exact first error '+$newError+' / '+$managedError)
}
PrimitiveCompare $null $null;PrimitiveCompare $null @();PrimitiveCompare @() $null
PrimitiveCompare @() @();PrimitiveCompare @(1) @(1);PrimitiveCompare @(1) @(1.0)
PrimitiveCompare @($null,$false,@(2,'x'),@{late=$null}) @($null,$false,@(2,'x'),@{late=$null})
PrimitiveCompare ([ordered]@{early=@{x=1};late=@{x=2}}) ([ordered]@{early=@{x=3};late=@{x=4}})
PrimitiveCompare ([ordered]@{early=@{x=3};missing=1}) ([ordered]@{early=@{x=1};late=1})
PrimitiveCompare @{a=1;b=2} ([ordered]@{b=2;a=1});PrimitiveCompare @{a=1} @{A=1}
PrimitiveCompare (Conserved-Json '{"a":1,"A":2}') (Conserved-Json '{"A":2,"a":1}')
PrimitiveCompare @{a=1} @{a=1;b=2};PrimitiveCompare @{a=1} @{b=1}
PrimitiveCompare $true 1;PrimitiveCompare 1 1.0;PrimitiveCompare ([int]1) ([long]1);PrimitiveCompare 'X' 'x'
PrimitiveCompare @(1,@(2,@{late=3})) @(1,@(2,@{late=4}))
Need ($script:NewSameEntered -eq 19) 'Actual new Same matrix entry count'

Need ($script:ManagedSameEntered -eq 19) 'Same literal19 pairs enter actual wrapper'
function ManagedBoundary($a,$b,[string]$tag){
 $genericError=$null;$managedError=$null
 try{Same $a $b ('boundary.'+$tag)}catch{$genericError=$_.Exception.Message}
 try{K-Same $a $b ('boundary.'+$tag)}catch{$managedError=$_.Exception.Message}
 Need ($genericError -ceq $managedError) ('Actual same-pair success/reject/first-error '+$tag)
}
ManagedBoundary @{Count=1;x=2} @{Count=1;x=2} 'Count-shadow'
ManagedBoundary @{Keys=@('x');x=2} @{Keys=@('x');x=2} 'Keys-shadow'
ManagedBoundary @{a=1} @{A=1} 'key-case-defer'
ManagedBoundary ([ordered]@{a=1;b=2}) ([ordered]@{b=2;a=1}) 'key-order'
ManagedBoundary ([ordered]@{early=2;missing=1}) ([ordered]@{early=1;late=1}) 'multiple-errors-original-order'
ManagedBoundary $null $null 'null';ManagedBoundary @() @() 'empty-array';ManagedBoundary @{} @{} 'empty-map'
ManagedBoundary $true $true 'bool';ManagedBoundary $true ([int]1) 'bool-int'
ManagedBoundary ([int]1) ([int]1) 'int';ManagedBoundary ([long]1) ([long]1) 'long'
ManagedBoundary ([int]1) ([long]1) 'int-long-defer';ManagedBoundary ([long]1) ([double]1) 'long-double-defer'
ManagedBoundary ([double]1.25) ([double]1.25) 'finite-double'
ManagedBoundary ([double]1.0) ([double]::BitIncrement([double]1.0)) 'adjacent-double'
ManagedBoundary ([double]0.0) ([double]-0.0) 'signed-zero'
ManagedBoundary ([double]::NaN) ([double]::NaN) 'nan-defer'
ManagedBoundary ([double]::PositiveInfinity) ([double]::PositiveInfinity) 'infinity-defer'
ManagedBoundary @{x=@([double]::NegativeInfinity)} @{x=@([double]::NegativeInfinity)} 'nested-nonfinite-defer'
ManagedBoundary ([decimal]1) ([decimal]1) 'unsupported-decimal'
$custom=[Collections.Generic.Dictionary[string,object]]::new();$custom.Add('x',[int]1)
ManagedBoundary $custom $custom 'custom-dictionary-defer'
$rank=[Array]::CreateInstance([object],[int[]]@(1,1));$rank.SetValue([int]1,0,0)
ManagedBoundary $rank $rank 'rank2-defer'
$lower=[Array]::CreateInstance([object],[int[]]@(1),[int[]]@(1));$lower.SetValue([int]1,1)
ManagedBoundary $lower $lower 'lower-bound-defer'
ManagedBoundary ([psobject]::new('x')) ([psobject]::new('x')) 'PSObject-no-unwrapping'
$ets=[psobject]::new(@{x=[int]1});Add-Member -InputObject $ets -NotePropertyName Count -NotePropertyValue 2 -Force
ManagedBoundary $ets @{x=[int]1} 'ETS-Count-shadow-defer'
ManagedBoundary 'X' 'x' 'string-case';ManagedBoundary ([string][char]233) ('e'+[char]769) 'InvariantCulture-string'
S41ReferenceControls
$backend=$script:KManagedCache;$origin=$script:KManagedOrigin
Need ($backend.type -is [type] -and $origin.Length -eq 6 -and [object]::ReferenceEquals($backend.type,$origin[0])) 'Actual Add-Type Type/Assembly/Method provenance'
$t=$backend.type;$entryBefore=$script:R41[0];$completeBefore=$script:R41[1]
$same=@{a=@([long]1,$false,[double]1.25,'x')};K-Same $same $same 'No reference-identity positive skip'
Need ($script:R41[0] -eq $entryBefore+1 -and $script:R41[1] -eq $completeBefore+1 -and $t::Entries -eq 0 -and $t::Matches -eq 0) 'Reference same object'
foreach($bad in @('foreign-type','foreign-assembly','foreign-method','partial','stale-source','stale-digest','null-cache')){
 $fake=@{};foreach($key in $backend.Keys){$fake[$key]=$backend[$key]}
 switch($bad){
  'foreign-type' {$fake.type=[string]}
  'foreign-assembly' {$fake.assembly=[string].Assembly}
  'foreign-method' {$fake.method=[string].GetMethod('IsNullOrEmpty',[type[]]@([string]))}
  'partial' {$fake.Remove('method')}
  'stale-source' {$fake.source+=' '}
  'stale-digest' {$fake.source_hash=('0'*64)}
  'null-cache' {$fake=$null}
 }
 try{$script:KManagedCache=$fake;$no=$script:R41[0];$failed=$false;try{K-Same ([int]1) ([int]1) 'cache-exception-cannot-fallback'}catch{$failed=$true};Need ($failed -and $script:R41[0] -eq $no) ('Actual foreign/partial/stale backend refusal '+$bad)}
 finally{$script:KManagedCache=$backend}
 K-Same ([int]1) ([int]1) 'Warm original backend after explicit cache refusal'
 Need ([object]::ReferenceEquals($script:KManagedOrigin,$origin)) 'Failed cache never adopts foreign origin'
}
# Same actual typed pair is constructed before both clocks. No one-sided parse/clone/read.
$timingPin=(K-S15Pins).anchor;$timingLeft=K-Read $timingPin 524288;$timingRight=K-Clone $timingLeft
$leftBefore=($timingLeft|ConvertTo-Json -Depth 100 -Compress);$rightBefore=($timingRight|ConvertTo-Json -Depth 100 -Compress)
$genericStart=[Diagnostics.Stopwatch]::GetTimestamp();Same $timingLeft $timingRight 'same typed whole314 pair';$genericEnd=[Diagnostics.Stopwatch]::GetTimestamp()
$entries=[long]$backend.type.GetField('Entries').GetValue($null);$matches=[long]$backend.type.GetField('Matches').GetValue($null);$re=$script:R41[0];$rc=$script:R41[1]
$managedStart=[Diagnostics.Stopwatch]::GetTimestamp();K-Same $timingLeft $timingRight 'same typed whole314 pair';$managedEnd=[Diagnostics.Stopwatch]::GetTimestamp()
Need ($script:R41[0] -eq $re+1 -and $script:R41[1] -eq $rc+1 -and [long]$backend.type.GetField('Entries').GetValue($null) -eq $entries -and [long]$backend.type.GetField('Matches').GetValue($null) -eq $matches) 'Actual REFERENCE_SAME_V1 comparison; managed not invoked'
Need ($genericStart -ge 0 -and $genericEnd -ge $genericStart -and $managedStart -ge $genericEnd -and $managedEnd -ge $managedStart) 'Actual monotonic same-pair timing/order'
Need ([string]::Equals($leftBefore,($timingLeft|ConvertTo-Json -Depth 100 -Compress),[StringComparison]::Ordinal) -and [string]::Equals($rightBefore,($timingRight|ConvertTo-Json -Depth 100 -Compress),[StringComparison]::Ordinal)) 'Actual large inputs read-only through both comparators'
$observation=[ordered]@{schema='flylab.m3.reference-same-observation.v1';profile='REFERENCE_SAME_V1';reference_entries=$script:R41[0];reference_completions=$script:R41[1];input_provenance=$timingPin;same_typed_objects_for_both=$true;order=@('Generic Same','REFERENCE_SAME_V1');frequency=[long][Diagnostics.Stopwatch]::Frequency;generic_start_ticks=$genericStart;generic_end_ticks=$genericEnd;reference_start_ticks=$managedStart;reference_end_ticks=$managedEnd;backend_entries=[long]$backend.type.GetField('Entries').GetValue($null);backend_matches=[long]$backend.type.GetField('Matches').GetValue($null);template_hash=$backend.template_hash;rendered_source_hash=$backend.source_hash;returned_type=$backend.type.FullName;returned_assembly=$backend.assembly.FullName;method=$backend.method.ToString();compile_start_ticks=$backend.compile_start_ticks;compile_end_ticks=$backend.compile_end_ticks;warm_reuses=$backend.warm_reuses;compile_in_overall_fee=$true;cold_call_in_pair=$false;JIT_state='WARM_FROM_EARLIER_ACTUAL_ADMITTED_CASES_NOT_ISOLATED';scope='ONE_PAIR_OBSERVATION_NO_GENERAL_SPEEDUP_OR_CAUSAL_CLAIM'}
Write-Output ('S41_REFERENCE_OBSERVATION '+($observation|ConvertTo-Json -Depth 30 -Compress))

$script:ActualKJson=(Get-Command K-Json).ScriptBlock;$script:NewJsonEntered=0
function K-Json($raw){$script:NewJsonEntered++; & $script:ActualKJson $raw}
try{
foreach($text in @('[]','[1]','null','1','true','"x"','[null,false,1,1.25,{"x":[2]}]','{"a":1,"A":2}','[{"a":1},{"a":2}]')){
$old=@(Conserved-Json $text);$before=$script:NewJsonEntered;$new=@(K-Json $text)
Need ($script:NewJsonEntered -eq $before+1) 'Actual new JSON positive entered';HistoricalSame $new $old 'Original reader output parity'
}
$depth=(('['*65)+'0'+(']'*65));[string[]]$negatives=@('{"a":1,"a":2}','[0,{"late":[{"a":1,"\u0061":2}]}]','[0,{"late":[{"b":1,"b":2}]}]','{"a":1,}','/*x*/{}','[NaN]',$depth)
Need ($negatives -is [array] -and $negatives.Length -eq 7) 'Exact seven JSON negatives';foreach($item in $negatives){Need ($item -is [string]) 'Every negative string'}
Need ($depth.Length -eq 131 -and [regex]::Matches($depth,'\[').Count -eq 65 -and [regex]::Matches($depth,'\]').Count -eq 65) 'Whole depth131/open65/close65'
for($ordinal=0;$ordinal -lt $negatives.Length;$ordinal++){$text=$negatives[$ordinal];$diagnostic='ordinal='+[string]$ordinal+' input='+$text
$oldError=$null;$newError=$null;try{$null=Conserved-Json $text}catch{$oldError=$_.Exception.Message}
Need ($null -ne $oldError) ('Retained reader rejects malformed JSON '+$diagnostic);$before=$script:NewJsonEntered
try{$null=K-Json $text}catch{$newError=$_.Exception.Message}
Need ($script:NewJsonEntered -eq $before+1 -and $null -ne $newError) ('Actual new JSON negative entered '+$diagnostic)
if($text.Contains('"a"') -or $text.Contains('"b"')){Need ($oldError -ceq $newError) ('Whole duplicate rejection parity '+$diagnostic)}
}
$bad='{"a":1,"a":2}';$name=$kb+'/generated-l13r2-actual-json-negative.json';[IO.File]::WriteAllBytes((Safe $name),$utf8.GetBytes($bad));$pin=K-BoundedPin $name
$before=$script:NewJsonEntered;$error=$null;try{$null=K-Read $pin}catch{$error=$_.Exception.Message}
Need ($script:NewJsonEntered -eq $before+1 -and $error -ceq 'Duplicate raw JSON event/key rejected') 'Actual bounded K-Read duplicate rejection'
$v=K-Json '{"a":[{"x":1}]}';$c=K-Clone $v;$c.a[0].x=[long]2;Need ($v.a[0].x -eq 1 -and $c.a[0].x -eq 2) 'Actual K-Clone detached tree'
}finally{Set-Item -LiteralPath Function:K-Json -Value $script:ActualKJson}
Write-Output 'PASS REFERENCE_SAME_V1 actual Same/JSON semantics and negatives'
'''
    receipts = v57.native_batch((
        ("lock", """
Need ($script:KManagedSourceAccepted -cne $true -and $null -eq $script:KManagedCache -and $null -eq $script:KManagedOrigin) 'Preacceptance accepted/cache/origin untouched before guard'
$guardMessage=$null
try{K-MatchInit}catch{$guardMessage=$_.Exception.Message}
Need ($guardMessage -ceq 'Managed backend only inside accepted owned/admitted action') 'Exact preacceptance managed guard exception.Message'
Need ($script:KManagedSourceAccepted -cne $true -and $null -eq $script:KManagedCache -and $null -eq $script:KManagedOrigin) 'Preacceptance accepted/cache/origin untouched after guard'

Need ($script:KManagedSourceAccepted -cne $true -and $null -eq $script:KCatalogCache -and $null -eq $script:KCatalogOrigin -and $null -eq $script:KManagedCache -and $null -eq $script:KManagedOrigin) 'Preacceptance both backends untouched before catalog guard'
$catalogGuardMessage=$null
try{K-CatalogInit}catch{$catalogGuardMessage=$_.Exception.Message}
Need ($catalogGuardMessage -ceq 'Catalog backend only inside accepted owned/admitted action') 'Exact preacceptance catalog guard exception.Message'
Need ($script:KManagedSourceAccepted -cne $true -and $null -eq $script:KCatalogCache -and $null -eq $script:KCatalogOrigin -and $null -eq $script:KManagedCache -and $null -eq $script:KManagedOrigin) 'Preacceptance both backends untouched after catalog guard'
"""+historical_fixture.d19_powershell_member_contract()+"""



$capProfiles=@(
 @{path='tools/banc_m3_checkpoint_harness_runtime.py';cap=[long]131072},
 @{path='tools/banc_m3_checkpoint_harness_runtime_native.ps1';cap=[long]131072},
 @{path=$kb+'/parent-direct-science-account.ps1';cap=[long]393216},
 @{path=$kb+'/s23-before/tools/banc_m3_checkpoint_harness_runtime.py';cap=[long]65536},
 @{path=$kb+'/s23-before/tools/banc_m3_checkpoint_harness_runtime_native.ps1';cap=[long]65536},
 @{path=$kb+'/s23-before/'+$kb+'/parent-direct-science-account.ps1';cap=[long]262144},
 @{path=$kb+'/s24-before/tools/banc_m3_checkpoint_harness_runtime.py';cap=[long]131072},
 @{path=$kb+'/s24-before/tools/banc_m3_checkpoint_harness_runtime_native.ps1';cap=[long]131072},
 @{path=$kb+'/s24-before/'+$kb+'/parent-direct-science-account.ps1';cap=[long]393216},
 @{path=$kb+'/s25-before/tools/banc_m3_checkpoint_harness_runtime.py';cap=[long]131072},
 @{path=$kb+'/s25-before/tools/banc_m3_checkpoint_harness_runtime_native.ps1';cap=[long]131072},
 @{path=$kb+'/s25-before/'+$kb+'/parent-direct-science-account.ps1';cap=[long]393216},
 @{path=$kb+'/s26-before/tools/banc_m3_checkpoint_harness_runtime.py';cap=[long]131072},
 @{path=$kb+'/s26-before/tools/banc_m3_checkpoint_harness_runtime_native.ps1';cap=[long]131072},
 @{path=$kb+'/s26-before/'+$kb+'/parent-direct-science-account.ps1';cap=[long]393216},
 @{path=$kb+'/s27-before/tools/banc_m3_checkpoint_harness_runtime.py';cap=[long]131072},
 @{path=$kb+'/s27-before/tools/banc_m3_checkpoint_harness_runtime_native.ps1';cap=[long]131072},
 @{path=$kb+'/s27-before/'+$kb+'/parent-direct-science-account.ps1';cap=[long]393216},
 @{path=$kb+'/s28-before/tools/banc_m3_checkpoint_harness_runtime.py';cap=[long]131072},
 @{path=$kb+'/s28-before/tools/banc_m3_checkpoint_harness_runtime_native.ps1';cap=[long]131072},
 @{path=$kb+'/s28-before/'+$kb+'/parent-direct-science-account.ps1';cap=[long]393216})
Need ($capProfiles.Length -eq 21) 'Exact current3/older3/S24 archive3/S25 archive3 cap boundaries'
foreach($profile in $capProfiles){
 Same (K-ReaderCap $profile.path) $profile.cap 'Exact literal path chooses its current or historical cap'
 $boundary=Safe $profile.path;$existed=[IO.File]::Exists($boundary);$saved=if($existed){[IO.File]::ReadAllBytes($boundary)}else{$null}
 $null=[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($boundary))
 $bytes=[Text.Encoding]::ASCII.GetBytes(('x'*($profile.cap+1)))
 try{
  [IO.File]::WriteAllBytes($boundary,$bytes);$rejected=$false
  try{$null=K-SourceRaw $boundary (K-ReaderCap $profile.path)}catch{$rejected=$true}
  Need $rejected ('Actual exact cap+1 refusal '+$profile.path)
  [IO.File]::WriteAllBytes($boundary,[byte[]]$bytes[0..($profile.cap-1)])
  $actual=K-SourceRaw $boundary (K-ReaderCap $profile.path)
  Need ($actual -is [byte[]] -and $actual.LongLength -eq $profile.cap) 'Whole exact selected cap read'
 }finally{if($existed){[IO.File]::WriteAllBytes($boundary,$saved)}else{[IO.File]::Delete($boundary)}}
}
$empty=K-SourceRaw (Safe ($kb+'/parent-direct-science-20261009-contact-s14-bootstrap/stderr.bin')) 262144
Need ($empty -is [byte[]] -and $empty.Length -eq 0) 'Production reader returns one empty byte[]'
$single=K-SourceRaw (Safe ($kb+'/parent-direct-science-20261009-contact-s14-bootstrap/stdout.bin')) 262144
Need ($single -is [byte[]] -and $single.Length -eq 1 -and $single[0] -eq 70) 'Actual one-byte F remains byte[]'
$path=Safe ($kb+'/generated-l-reader-return.bin');$ascii=[Text.Encoding]::ASCII.GetBytes(('x'*65536))
try{
 [IO.File]::WriteAllBytes($path,$ascii);$raw=K-SourceRaw $path 65536
 Need ($raw -is [byte[]] -and $raw.Length -eq 65536) 'Production buffer-boundary byte[]'
 Same ([Convert]::ToBase64String($raw)) ([Convert]::ToBase64String($ascii)) 'Complete independent boundary reads'
 [IO.File]::WriteAllBytes($path,[byte[]]@(255));$rejected=$false;try{$null=K-SourceRaw $path 65536}catch{$rejected=$true};Need $rejected 'Production strict UTF8 rejection'
 [IO.File]::WriteAllBytes($path,([byte[]]($ascii+@([byte]120))));$rejected=$false;try{$null=K-SourceRaw $path 65536}catch{$rejected=$true};Need $rejected 'Production cap+1 rejected before read'

$expectedLedger=K-Json '["@/s28-f15b-root/terminal319-before.json","@/s28-f15-root/admitted318-before.json","@/s28-f15-root/settlement-pre-backup.json","@/s28-f15-root/settlement-replace-backup.json","@/s28-f12-root/a318.json","@/s28-f14-root/a318.json","@/s27-bootstrap-settled318-HOLD-20261009.json","@/s28-f13-root/a318.json","@/s28-anchor318-before-recovery.json","@/s28-f14-root/a318.json","@/s28-f13-root/a318.json","@/s28-f12-root/a318.json","@/s28-f11-root/a318.json","@/s28-f10-root/a318.json","@/s28-f9-root/a318.json","@/s28-f8-root/a318.json","@/s28-f7-root/a318.json","@/s28-f6-root/a318.json","@/s28-f5-root/a318.json","@/s28-f4-root/a318.json","@/s28-anchor318-before-recovery.json","@/s28-recovered318-before-bootstrap.json","@/parent-s28-bootstrap-settled319-before-allocation-20261009.json","@/parent-s28-qualification-settled323-before-body-20261009.json","@/parent-s28-body-settled324-before-audit-20261009.json","@/s28-bootstrap-settled319-HOLD-20261009.json","@/s27-anchor317-before-recovery.json","@/s27-recovered317-before-bootstrap.json","@/parent-s27-bootstrap-settled318-before-allocation-20261009.json","@/parent-s27-qualification-settled322-before-body-20261009.json","@/parent-s27-body-settled323-before-audit-20261009.json","@/s27-bootstrap-settled318-HOLD-20261009.json","@/s25-bootstrap-settled317-HOLD-20261009.json","@/s26-anchor317-before-recovery.json","@/s26-recovered317-before-bootstrap.json","@/parent-s26-bootstrap-settled318-before-allocation-20261009.json","@/parent-s26-qualification-settled322-before-body-20261009.json","@/parent-s26-body-settled323-before-audit-20261009.json","@/s26-bootstrap-settled318-HOLD-20261009.json","@/s24-bootstrap-settled316-HOLD-20261009.json","@/s25-anchor316-before-recovery.json","@/s25-recovered316-before-bootstrap.json","@/parent-s25-bootstrap-settled317-before-allocation-20261009.json","@/parent-s25-qualification-settled321-before-body-20261009.json","@/parent-s25-body-settled322-before-audit-20261009.json","@/s25-bootstrap-settled317-HOLD-20261009.json","@/s23-bootstrap-settled315-HOLD-20261009.json","@/s24-anchor315-before-recovery.json","@/s24-recovered315-before-bootstrap.json","@/parent-s24-bootstrap-settled316-before-allocation-20261009.json","@/parent-s24-qualification-settled320-before-body-20261009.json","@/parent-s24-body-settled321-before-audit-20261009.json","@/s24-bootstrap-settled316-HOLD-20261009.json","@/parent-l12-cold-settled306-HOLD-20261008.json","@/parent-l12-bootstrap-admitted305-before-cold-settlement-20261008.json","@/parent-l11-bootstrap-settled305-HOLD-20261008.json","@/parent-l10-cold-settled304-HOLD-20261008.json","@/parent-l10-bootstrap-admitted303-before-cold-settlement-20261008.json","@/parent-l9r2-bootstrap-settled303-HOLD-20261008.json","@/parent-l9-cold-settled302-HOLD-20261008.json","@/parent-l8-bootstrap-settled301-HOLD-20261008.json","@/parent-l7-cold-settled300-HOLD-20261008.json","@/parent-e358-execution-ledger.json","@/parent-l13r2-bootstrap-settled307-before-allocation-20261008.json","@/parent-l13r2-qualification-settled311-before-body-20261008.json","@/parent-l13r2-body-settled312-before-audit-20261008.json","@/parent-l9-bootstrap-admitted301-before-cold-settlement-20261008.json","@/parent-l7-bootstrap-admitted299-before-cold-settlement-20261008.json","@/parent-l6-math-settled299-HOLD-20261008.json","@/parent-l3-bootstrap-settled292-before-allocation-20261008.json","@/generated-l13r2-allocation-before.json","@/generated-l13r2-original-before.json","@/s16-bootstrap-settled310-HOLD-20261009.json","@/s16-bootstrap-settled310-HOLD-20261009.json","@/parent-s15-bootstrap-settled309-before-allocation-20261009.json","@/parent-s15-qualification-settled313-before-body-20261009.json","@/parent-s15-body-settled314-before-audit-20261009.json","@/parent-l13r2-bootstrap-settled307-HOLD-20261008.json","@/s14-anchor307-after-repair400.json","@/parent-s14-bootstrap-settled308-before-allocation-20261009.json","@/parent-s14-qualification-settled312-before-body-20261009.json","@/parent-s14-body-settled313-before-audit-20261009.json","@/s14-bootstrap-settled308-HOLD-20261009.json","@/s15-bootstrap-settled309-HOLD-20261009.json","@/parent-s16-bootstrap-settled310-before-allocation-20261009.json","@/parent-s16-qualification-settled314-before-body-20261009.json","@/parent-s16-body-settled315-before-audit-20261009.json","@/s16-anchor309-after-repair400-v2.json","@/s16-recovered309-before-bootstrap.json","@/parent-s17-bootstrap-settled311-before-allocation-20261009.json","@/parent-s17-qualification-settled315-before-body-20261009.json","@/parent-s17-body-settled316-before-audit-20261009.json","@/s17-bootstrap-settled311-HOLD-20261009.json","@/s18-anchor311-after-repair20.json","@/s17-recovered310-before-bootstrap.json","@/parent-s18-bootstrap-settled312-before-allocation-20261009.json","@/parent-s18-qualification-settled316-before-body-20261009.json","@/parent-s18-body-settled317-before-audit-20261009.json","@/s18-bootstrap-settled312-HOLD-20261009.json","@/s19-anchor312-after-repair200.json","@/s18-recovered311-before-bootstrap.json","@/parent-s19-bootstrap-settled313-before-allocation-20261009.json","@/parent-s19-qualification-settled317-before-body-20261009.json","@/parent-s19-body-settled318-before-audit-20261009.json","@/s19-bootstrap-settled313-HOLD-20261009.json","@/s22-anchor313-after-repair110.json","@/s19-recovered312-before-bootstrap.json","@/parent-s22-bootstrap-settled314-before-allocation-20261009.json","@/parent-s22-qualification-settled318-before-body-20261009.json","@/parent-s22-body-settled319-before-audit-20261009.json","@/s22-bootstrap-settled314-HOLD-20261009.json","@/s23-anchor314-after-grant2000.json","@/parent-s23-bootstrap-settled315-before-allocation-20261009.json","@/parent-s23-qualification-settled319-before-body-20261009.json","@/parent-s23-body-settled320-before-audit-20261009.json","@/s28-f16-root/before/@/parent-e358-execution-ledger.json","@/s28-f15c-root/before-full4-canonical319.json","@/s28-f15c-root/full4-admitted319.json","@/s28-f18-root/b320.json","@/s28-f18-root/a320.json","@/s28-f18-root/t321.json","@/s28-f18-root/residual02/r3-before-admission-canonical321.json","@/s28-f18-root/residual02/r3-admitted321.json","@/s28-f18-root/residual02/r3-terminal322.json","@/s28-f19-root/transition44/allocation-before.json","@/s28-f19-root/transition44/original-body-before.json","@/s28-f19-root/transition44/original-audit-before.json","@/s28-f19-root/transition44/cut/independent-expected-paid363.json","@/s28-f19-root/transition43/paid363-pre-spawn.json","@/s28-f19-root/transition43/paid362.json","@/s28-f19-root/transition45/allocation-before.json","@/s28-f19-root/transition45/original-body-before.json","@/s28-f19-root/transition45/original-audit-before.json","@/s28-f19-root/transition45/cut/independent-expected-paid364.json","@/s28-f19-root/transition44/paid364.json","@/s28-f19-root/transition46/allocation-before.json","@/s28-f19-root/transition46/original-body-before.json","@/s28-f19-root/transition46/original-audit-before.json","@/s28-f19-root/transition46/cut/independent-expected-paid365.json","@/s28-f19-root/transition45/paid365.json","@/s28-f19-root/transition47/allocation-before.json","@/s28-f19-root/transition47/original-body-before.json","@/s28-f19-root/transition47/original-audit-before.json","@/s28-f19-root/transition47/cut/independent-expected-paid366.json","@/s28-f19-root/transition46/paid366.json","@/s28-f19-root/transition48/allocation-before.json","@/s28-f19-root/transition48/original-body-before.json","@/s28-f19-root/transition48/original-audit-before.json","@/s28-f19-root/transition48/cut/independent-expected-paid367.json","@/s28-f19-root/transition47/paid367.json","@/s28-f19-root/transition49/allocation-before.json","@/s28-f19-root/transition49/original-body-before.json","@/s28-f19-root/transition49/original-audit-before.json","@/s28-f19-root/transition49/cut/independent-expected-paid368.json","@/s28-f19-root/transition48/paid368.json","@/s28-f19-root/transition50/allocation-before.json","@/s28-f19-root/transition50/original-body-before.json","@/s28-f19-root/transition50/original-audit-before.json","@/s28-f19-root/transition50/cut/independent-expected-paid369.json","@/s28-f19-root/transition49/paid369.json","@/s28-f19-root/transition51/allocation-before.json","@/s28-f19-root/transition51/original-body-before.json","@/s28-f19-root/transition51/original-audit-before.json","@/s28-f19-root/transition51/cut/independent-expected-paid370.json","@/s28-f19-root/transition50/paid370.json","@/s28-f19-root/transition52/allocation-before.json","@/s28-f19-root/transition52/original-body-before.json","@/s28-f19-root/transition52/original-audit-before.json","@/s28-f19-root/transition52/cut/independent-expected-paid371.json","@/s28-f19-root/transition51/paid371.json","@/s28-f19-root/transition53/allocation-before.json","@/s28-f19-root/transition53/original-body-before.json","@/s28-f19-root/transition53/original-audit-before.json","@/s28-f19-root/transition53/cut/independent-expected-paid372.json","@/s28-f19-root/transition52/paid372.json","@/s28-f19-root/transition54/allocation-before.json","@/s28-f19-root/transition54/original-body-before.json","@/s28-f19-root/transition54/original-audit-before.json","@/s28-f19-root/transition54/cut/independent-expected-paid373.json","@/s28-f19-root/transition53/paid373.json","@/s28-f19-root/transition55/cut/p.json","@/s28-f19-root/transition54/paid374.json","@/s28-f19-root/transition56/p.json","@/s28-f19-root/transition55/paid375.json",R66_PATHS]'.Replace('@','verification/banc-natural-gait-m3-20260919');Same @(K-LedgerPaths) @($expectedLedger) 'Independent Native original115 plus finite9 exact order/duplicates';Same @(K-Calls) (K-Json '["direct-science-20261009-contact-s28-bootstrap-r64","direct-science-20261009-contact-s28-r64-lint","direct-science-20261009-contact-s28-r64-math","direct-science-20261009-contact-s28-r64-gate-a","direct-science-20261009-contact-s28-r64-gate-b","contact-learning-20261009-s28-r64-body","contact-learning-20261009-s28-r64-body-audit"]') 'Fixed seven roles'
$contract=K-Read $script:KCurrentContract;$whole=$script:KWhole;$oldOutput=@(& $script:KPredecessor $contract.actual320 $whole);$oldestOutput=@(& $script:KPredecessor $oldOutput[0].actual320 $whole);Need ($oldOutput.Length -eq 1 -and $oldOutput[0] -is [Collections.IDictionary]) 'Predecessor one dictionary no extra output'

$prefixBody=(Get-Item -LiteralPath Function:K-ReaderCap).Definition
$prefixMatch=[regex]::Match($prefixBody,'foreach\\(\\$pf in (.*?)\\)\\{if')
Need $prefixMatch.Success 'Actual production selector prefix expression present'
$readerPrefixes=@(Invoke-Expression $prefixMatch.Groups[1].Value)
Need ($readerPrefixes.Length -eq 2 -and $readerPrefixes[0] -is [string] -and $readerPrefixes[1] -is [string]) 'Actual production selector emits two strings'
Same $readerPrefixes @(($kb+'/s28-f16-root/before/'),($kb+'/s28-f18-root/before/')) 'Actual ordered finite reader prefixes'
$residualSix=K-Json '[{"pin":{"path":"verification/banc-natural-gait-m3-20260919/s28-f16-root/before/tools/banc_m3_checkpoint_harness_runtime.py","bytes":98851,"sha256":"f4c9099304095bcb433b2bfe6871e6537936763270310d17502300fe3ec7c084"},"cap":131072},{"pin":{"path":"verification/banc-natural-gait-m3-20260919/s28-f16-root/before/tools/banc_m3_checkpoint_harness_runtime_native.ps1","bytes":126857,"sha256":"20d35b8f9dfe61ebcd58643ff195dac4de7211708a9081e5edae8cb882c2da1d"},"cap":131072},{"pin":{"path":"verification/banc-natural-gait-m3-20260919/s28-f16-root/before/verification/banc-natural-gait-m3-20260919/parent-direct-science-account.ps1","bytes":324356,"sha256":"aa56466be84939acbc4f27325b7aa202ea3ca71f34da79aae413292bb6d17c0d"},"cap":393216},{"pin":{"path":"verification/banc-natural-gait-m3-20260919/s28-f18-root/before/tools/banc_m3_checkpoint_harness_runtime.py","bytes":99523,"sha256":"8dc2a47468ce82c1730e609ad7c0932da3885749ccb07ef7523dcc3a3ea61f7e"},"cap":131072},{"pin":{"path":"verification/banc-natural-gait-m3-20260919/s28-f18-root/before/tools/banc_m3_checkpoint_harness_runtime_native.ps1","bytes":130307,"sha256":"a1db2a1e7a508ba5736e8372ceab684c2e81082bebda149c7364b7fc0e3643c2"},"cap":131072},{"pin":{"path":"verification/banc-natural-gait-m3-20260919/s28-f18-root/before/verification/banc-natural-gait-m3-20260919/parent-direct-science-account.ps1","bytes":327806,"sha256":"99407d9cbfc591c326216ff4482241df45d5a66ea36e89ee1eaf55acd7080807"},"cap":393216}]'
Need ($residualSix.Length -eq 6) 'Six independently pinned old/current raw operands'
foreach($row in $residualSix){
 $cap=K-ReaderCap $row.pin.path
 Need ($cap -is [long] -and $cap -eq $row.cap) 'Actual exact operand Int64 cap'
 $result=@(K-Raw $row.pin $cap)
 Need ($result.Length -eq 1 -and $result[0] -is [byte[]] -and $result[0].LongLength -eq $row.pin.bytes) 'Actual one whole byte array per archived operand'
 $independent=[IO.File]::ReadAllBytes((Safe $row.pin.path))
 Same ([long]$independent.LongLength) $row.pin.bytes 'Independent whole archived bytes'
 Same ([Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($independent)).ToLowerInvariant()) $row.pin.sha256 'Independent archived SHA'
 Same ([Convert]::ToBase64String($result[0])) ([Convert]::ToBase64String($independent)) 'Actual K-Raw equals full independently pinned bytes'
}
Need ((Get-Item Function:K-Recovery).Definition.Contains('Same @([long]$t.debits.Length,(Conserved-Count')) 'Reviewed production count boundary'
$countPairs=@(@($oldestOutput[0].actual320.terminal,[long]320,[long]202),@($oldOutput[0].actual320.terminal,[long]321,[long]203),@($contract.actual320.terminal,[long]322,[long]204))
foreach($row in $countPairs){$value=K-Read $row[0] 524288;$counts=@([long]$value.debits.Length,(Conserved-Count $value.direct_science_calls));Need ($counts.Length -eq 2 -and $counts[0] -is [long] -and $counts[1] -is [long]) 'Both actual terminal counts Int64';Same $counts @($row[1],$row[2]) 'Exact actual terminal counts'}
$strict=$false;try{Same ([int]1) ([long]1) 'Residual strict'}catch{$strict=$_.Exception.Message -ceq 'Residual strict type'};Need $strict 'Unchanged Int32 Int64 refusal'
foreach($bridge in @($oldestOutput[0].actual320,$oldOutput[0].actual320,$contract.actual320)){foreach($name in @('before_full4','admitted_full4','terminal')){$pin=$bridge[$name];Same (K-ReaderCap $pin.path) ([long]524288) 'Finite paid ledger selector';$value=K-Read $pin 524288;Need ($value -is [Collections.IDictionary]) 'Actual finite paid ledger metadata acceptance'}}
$paths=K-Paths;foreach($logical in @($paths.contract,$paths.review)){foreach($duplicate in @($false,$true)){$wrong=K-Clone $contract.actual320;$matching=@($wrong.archives|Where-Object {$_.logical_current_before.path -ceq $logical});$wrong.archives=@($wrong.archives|Where-Object {$_.logical_current_before.path -cne $logical});if($duplicate){$wrong.archives+=@($matching)+@($matching)};$rejected=$false;try{$null=& $script:KPredecessor $wrong $whole}catch{$wanted=if($logical -ceq $paths.contract){'Unique predecessor contract'}else{'Unique predecessor review'};Need ($_.Exception.Message.Contains($wanted)) 'Intended predecessor selection guard, never unrelated hash gate';$rejected=$true};Need $rejected 'Missing/duplicate predecessor row refused before selected read'}}
}finally{[IO.File]::Delete($path)}
""".replace("R66_PATHS", ",".join('"verification/banc'+n+'/'+p+'.json"' for n in ('57','58','59','60','61','62','63','64') for p in ('p','t','a','b','u'))) + "$value=K-Read (K-BoundedPin '" + r.SHARED_LOCK + "') 16384;Same $value.schema 'flylab.m3.direct-science.lock-owner.v1' 'Existing literal ownership JSON';$rejected=$false;try{$null=K-Read (K-BoundedPin '" + foreign_lock['path'] + "') 16384}catch{$rejected=$true};Need $rejected 'Foreign lock metadata refused before read';Write-Output 'PASS synthetic existing ownership token reader'"),
        ("typed-state", "$l=K-Read (K-BoundedPin $ledgerName) 524288;$s=K-Validate $l;$s|ConvertTo-Json -Depth 10 -Compress"),
        ("legacy-control", "function Load-CheckpointLegacy {throw 'forbidden K fee loader'};function Conserved-J-History {throw 'forbidden K recursion'};$l=K-Read (K-BoundedPin $ledgerName) 524288;$s=K-Validate $l;K-Same $s.caps (K-Read (K-Read $script:KCurrentContract).d19_current_golden).new_caps 'Current golden caps';Write-Output 'PASS synthetic K production validator'"),
        ("cache-copy-drift-policy-four-mutations", historical_fixture.d19_early_snapshot_history(f,k,v57)+k.d19_opaque_native(k,v57,f.D19_OPAQUE_TAIL)+"""
$s19PhaseJournal=@{root=$taskRoot;stream=$null;encoding=[Text.UTF8Encoding]::new($false,$true);phases=@('detached','companion3','snapshots','reentry','clone','raw-drift','root-descriptor','catalog','mutations','final');ordinal=[long]0;ticks=[long]0;frequency=[long][Diagnostics.Stopwatch]::Frequency;primary=$null}
function Write-S19CasePhase([string]$phase,[string]$boundary){
 [long]$s19PhaseTicks=[Diagnostics.Stopwatch]::GetTimestamp()
 Need ($s19PhaseJournal.ordinal -is [long] -and $s19PhaseJournal.ordinal -ge 0 -and $s19PhaseJournal.ordinal -lt 20 -and $phase -ceq $s19PhaseJournal.phases[[int]($s19PhaseJournal.ordinal -shr 1)] -and $boundary -ceq @('begin','end')[[int]($s19PhaseJournal.ordinal % 2)]) 'S19 exact fixed phase order'
 Need ($s19PhaseTicks -ge 0 -and $s19PhaseTicks -ge $s19PhaseJournal.ticks -and $s19PhaseJournal.frequency -gt 0 -and $s19PhaseJournal.frequency -eq [Diagnostics.Stopwatch]::Frequency) 'S19 monotonic same-frequency ticks'
 $s19PhaseJSON=([ordered]@{schema='flylab.m3.s19.native-case-phase.v1';ordinal=[long]$s19PhaseJournal.ordinal;phase=$phase;boundary=$boundary;ticks=$s19PhaseTicks;frequency=[long]$s19PhaseJournal.frequency;root=$s19PhaseJournal.root}|ConvertTo-Json -Depth 10 -Compress)
 $s19PhaseBytes=$s19PhaseJournal.encoding.GetBytes($s19PhaseJSON+[char]10)
 Need ($s19PhaseJournal.stream.Position+$s19PhaseBytes.LongLength -le 8192) 'S19 phase journal byte cap'
 $s19PhaseJournal.stream.Write($s19PhaseBytes,0,$s19PhaseBytes.Length);$s19PhaseJournal.stream.Flush()
 $s19PhaseJournal.ticks=$s19PhaseTicks;$s19PhaseJournal.ordinal++
}
try{
 $s19PhaseJournal.stream=[IO.File]::Open([IO.Path]::Combine($s19PhaseJournal.root,'s19-case-phases.jsonl'),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
Write-S19CasePhase 'detached' 'begin'
$sp=K-BoundedPin (K-Paths).recovery;$proof=K-Recovery $sp;$positive=K-Clone $proof
$proof.event.source_freeze.bytes=[long]1;$proof.anchor.debits[0].cost_id='returned-mutation';$proof.policy.returned=$true;$proof.grant.returned=@($null,$false,[double]1.25)
K-Same (K-Recovery $sp) $positive 'Detached whole typed cached proof'
Write-S19CasePhase 'detached' 'end'
Write-S19CasePhase 'companion3' 'begin'
foreach($corruption in @('malformed-text','valid-stale-json','typed-nested')){
 $output=@(& $script:R14Legacy $script:R14Key (K-Clone $script:R14Seed) (K-Clone $script:R14Event));Need ($output.Length -eq 1 -and $output[0] -is [Collections.IDictionary]) 'Actual legacy consumer one dictionary';$null=& $script:R14Check $output[0];$output[0].event.source_freeze.bytes=1L;$output[0].anchor.debits[0].cost_id='returned-legacy';$output[0].policy.returned=$true;$output[0].grant.returned=@($null,$false,[double]1.25);$output=@(& $script:R14Legacy $script:R14Key (K-Clone $script:R14Seed) (K-Clone $script:R14Event));Need ($output.Length -eq 1) 'Detached legacy warm one output';$null=& $script:R14Check $output[0]
 $key=@($script:KPC.Keys)[0]
 switch($corruption){
  'malformed-text' {$script:KPJ[1]='{'}
  'valid-stale-json' {$stale=K-Clone $script:KPC[$key];$stale.anchor.debits[0].cost_id='companion-stale';$script:KPJ[1]=($stale|ConvertTo-Json -Depth 100 -Compress)}
  'typed-nested' {$script:KPC[$key].anchor.debits[0].cost_id='cache-typed-stale'}
 }
 $rejected=$false;try{$null=& $script:R14Legacy $script:R14Key (K-Clone $script:R14Seed) (K-Clone $script:R14Event)}catch{$rejected=$true}
 Need ($rejected -and $script:KPC.Count -eq 0 -and $null -eq $script:KPJ -and -not $script:KPB) ('Companion rejection, paired reset and guard cleanup '+$corruption)
 Need ($script:KManagedSourceAccepted -ceq $false -and $null -eq $script:KCatalogManifestContract) 'Legacy rejection invalidates accepted Source'
 $null=K-SourceFence (K-BoundedPin (K-Paths).source)
 $output=@(& $script:R14Legacy $script:R14Key (K-Clone $script:R14Seed) (K-Clone $script:R14Event));Need ($output.Length -eq 1) 'Next cold legacy one output';$null=& $script:R14Check $output[0]
}
Write-S19CasePhase 'companion3' 'end'
Write-S19CasePhase 'snapshots' 'begin'
foreach($warm in @($false,$true)){
 $script:KPC.Clear();$script:KPJ=$null;if($warm){K-Same (K-Recovery $sp) $positive 'Warm snapshot control ready'}
 $snapshot=K-RecoveredSnapshot (K-Clone $sp);$expected=K-Json (K-Raw $script:R14SnapshotPin 524288)
 K-Same $snapshot $expected 'Whole detached cold/warm snapshot'
 $script:KPC.Clear();$script:KPJ=$null;if($warm){K-Same (K-Recovery $sp) $positive 'Warm validation control ready'}
 $live=K-Clone $snapshot;$null=K-Validate $live
 K-Same $live $expected 'Whole live input preserved by cold/warm validation'
 Need ($script:KPC.Count -eq 0 -and $null -eq $script:KPJ) 'Current recovery has no historical cache producer';K-Same (K-Recovery $sp) $positive 'Whole current anchor/policy/grant preserved'
 K-Same (K-Recovery $sp) $positive 'Validation preserves next whole recovery'
 K-Same (K-PinnedAnchor).anchor (K-Read (K-S15Pins).anchor 524288) 'Whole independent frozen318 preserved'
 $snapshot.debits[0].cost_id='snapshot-array-mutation';$map=@(Conserved-Keys $snapshot.direct_science_calls)[0];$snapshot.direct_science_calls[$map].state='snapshot-map-mutation';$snapshot[(K-Key)].bytes=[long]1;$snapshot.direct_science_checkpoint_repair_s16_400.bytes=[long]1
 K-Same (K-Recovery $sp) $positive 'Snapshot mutations preserve next whole recovery'
 K-Same (K-PinnedAnchor).anchor (K-Read (K-S15Pins).anchor 524288) 'Whole independent frozen318 preserved'
 K-Same (K-PinnedAnchor).anchor (K-Read (K-S15Pins).anchor 524288) 'Whole independent frozen318 preserved'
}
Write-S19CasePhase 'snapshots' 'end'
Write-S19CasePhase 'reentry' 'begin'
$script:R1ActualRead=(Get-Command K-Read).ScriptBlock;$script:R1RecoveryPin=$sp
foreach($warm in @($false,$true)){
 $script:KPC.Clear();$script:KPJ=$null;if($warm){K-Same (K-Recovery $sp) $positive 'Warm reentry control ready'}
 $script:R1ReaderEntered=0
 function K-Read($_pin,[long]$cap=262144){
  if($script:R1ReaderEntered -eq 0){
   $script:R1ReaderEntered++;Need $script:KPB 'Guard before actual reader'
   $rejected=$false;try{$null=K-Recovery $script:R1RecoveryPin}catch{$rejected=$true}
   Need ($rejected -and $script:KPB) 'Inner rejection retains outer capability'
  }
  & $script:R1ActualRead $_pin $cap
 }
 try{K-Same (K-Recovery $sp) $positive 'Actual cold/warm reader reentry rejected'}
 finally{Set-Item -LiteralPath Function:K-Read -Value $script:R1ActualRead}
 Need ($script:R1ReaderEntered -eq 1 -and -not $script:KPB) 'Owned guard cleanup'
 K-Same (K-Recovery $sp) $positive 'Guard cleanup positive return'
}
""" + historical_fixture.d19_retained_consumer_reentry() + """
Write-S19CasePhase 'reentry' 'end'
Write-S19CasePhase 'clone' 'begin'
$check=@{empty=@();null=$null;bool=$false;integer=[long]1;number=[double]1.25;map=@{x=@($true,[long]2)}}
$clone=K-Clone $check;Same $clone $check 'Empty/null/bool/Int64/double/order clone';$clone.map.x[0]=$false;Need ($check.map.x[0] -ceq $true) 'Nested independent clone'
Write-S19CasePhase 'clone' 'end'
Write-S19CasePhase 'raw-drift' 'begin'
$names=@(($taskBase+'/l9-before/tools/banc_m3_checkpoint_harness_runtime.py'),((K-Read (K-HistoryPins).report).nested_failure.path),($taskBase+'/s14-before/tools/banc_m3_checkpoint_harness_runtime.py'),($taskBase+'/parent-direct-science-20261008-contact-l13r2-bootstrap/stderr.bin'))
foreach($name in $names){$path=Safe $name;$raw=[IO.File]::ReadAllBytes($path);$null=K-CatalogVerified (K-Read (K-S15Pins).catalog);try{$drift=if($raw.Length){$copy=[byte[]]$raw.Clone();$copy[-1]=$copy[-1] -bxor 1;,$copy}else{,[byte[]]@(120)};[IO.File]::WriteAllBytes($path,$drift);$rejected=$false;try{$null=K-CatalogVerified (K-Read (K-S15Pins).catalog)}catch{$rejected=$true};Need $rejected 'Warm altered complete dependency/direct archive';[IO.File]::WriteAllBytes($path,$raw);$null=K-CatalogVerified (K-Read (K-S15Pins).catalog);[IO.File]::Delete($path);$rejected=$false;try{$null=K-CatalogVerified (K-Read (K-S15Pins).catalog)}catch{$rejected=$true};Need $rejected 'Warm missing dependency/direct archive'}finally{[IO.File]::WriteAllBytes($path,$raw)}}
Write-S19CasePhase 'raw-drift' 'end'
Write-S19CasePhase 'root-descriptor' 'begin'
$root=$taskRoot;try{$taskRoot=$root+'/different-history-root';$rejected=$false;try{$null=K-Recovery $sp}catch{$rejected=$true};Need $rejected 'Different root no cache hit'}finally{$taskRoot=$root}
foreach($value in @($true,[double]$sp.bytes)){$wrong=K-Clone $sp;$wrong.bytes=$value;$rejected=$false;try{$null=K-Recovery $wrong}catch{$rejected=$true};Need $rejected 'Warm strict descriptor before hit'}
Write-S19CasePhase 'root-descriptor' 'end'
Write-S19CasePhase 'catalog' 'begin'
$catalog=K-Read (K-S15Pins).catalog;$policy=K-CatalogVerified $catalog
# D12 exact finite source-owned diagnostic metadata, with caller13 separate.
$tableAbsent=$script:KCatalogCache.type.GetMethod('TableAbsent',[Reflection.BindingFlags]'NonPublic,Static')
Need ($tableAbsent.DeclaringType -eq $script:KCatalogCache.type -and $tableAbsent.ReturnType -eq [bool]) 'Actual source-owned TableAbsent method'
$parameters=$tableAbsent.GetParameters();Need ($parameters.Length -eq 6 -and $parameters[4].ParameterType -eq [int] -and $parameters[5].ParameterType -eq [int].MakeByRefType() -and $parameters[5].IsOut) 'Fixed out-int metadata with original four operands'
$resurrection=$script:KCatalogCache.type.GetMethod('Resurrection',[Reflection.BindingFlags]'NonPublic,Static')
Need ($resurrection.DeclaringType -eq $script:KCatalogCache.type -and $resurrection.ReturnType -eq [bool]) 'Actual source-owned Resurrection method'
$rp=$resurrection.GetParameters();Need ($rp.Length -eq 5 -and $rp[0].ParameterType -eq [object] -and $rp[1].ParameterType -eq [type] -and $rp[2].ParameterType -eq [string] -and $rp[3].ParameterType -eq [int] -and $rp[4].ParameterType -eq [int].MakeByRefType() -and $rp[4].IsOut) 'Exact Resurrection5 ABI, no invocation or injected metadata'
$finite=@(0..12)+@(14..19)+@(22..35)+@(38..137);Need ($finite.Count -eq 133 -and 13 -notin $finite -and 20 -notin $finite -and 21 -notin $finite -and 36 -notin $finite -and 37 -notin $finite) 'Exact finite production reason union'
# D12 SOURCE_AUTHORED_NOT_RUN: reserve unchanged133 production rows plus exact caller13; no additional rows/process/node.
Need ($script:KCatalogJournal.ordinal -le 1018) 'Existing1152 cap conditional room for exact134 finite rows; whole control fit NOT_RUN'
foreach($reason in $finite){K-CatalogJournal 'catalog-support' 'END' @([int]1,[int]-1,[int]0,[int]$reason,[int]1,'UNKNOWN')}
K-CatalogJournal 'catalog-support' 'END' @([int]0,[int]-1,[int]0,[int]13,[int]0,'DIAGNOSTIC_UNAVAILABLE')
foreach($reason in @(-1,13,20,21,36,37,138,139,2147483647)){$rejected=$false;try{K-CatalogJournal 'catalog-support' 'END' @([int]1,[int]-1,[int]0,[int]$reason,[int]1,'UNKNOWN')}catch{$rejected=$true};Need $rejected 'Invalid production reason cannot serialize'}
$continuityPins=@();foreach($name in @('refinement_anchor','previous_recovery','previous_source','previous_contract','previous_catalog','technical_refusal')){$continuityPins+=,(K-S15Pins)[$name]};$continuityGate=K-Read (K-S15Pins).technical_refusal;foreach($name in @('actual_tool_return','actual_known_compiler_type_refusal','journal')){$continuityPins+=,$continuityGate[$name]};foreach($pin in @(@{'path'='verification/banc-natural-gait-m3-20260919/s28-f14-root/a318.json';'bytes'=[long]495559;'sha256'='fadfb1604441f5c2cc481e15b07eeaf10a7a7529cedc8969bc496196ea221274'},@{'path'='verification/banc-natural-gait-m3-20260919/s28-recovery10.json';'bytes'=[long]1903;'sha256'='27f8b420a0f7cdb3447aac416cd2ac71cf499ee8b5ea719ff05c968fd6f1184e'},@{'path'='verification/banc-natural-gait-m3-20260919/s28-f14-root/s1-before.json';'bytes'=[long]262140;'sha256'='f1b802af060e385f03955b5cbba5830cc9ee42ccf757b9800016a45afd7cabfd'},@{'path'='verification/banc-natural-gait-m3-20260919/s28-f14-root/c5-before.json';'bytes'=[long]97623;'sha256'='9690bd2ec7a65d33432071d09d9050c6d949c92d6a20852fc92072737f7dcb41'},@{'path'='verification/banc-natural-gait-m3-20260919/s28-f14-root/h28-before.json';'bytes'=[long]261912;'sha256'='e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e'},@{'path'='verification/banc-natural-gait-m3-20260919/s28-f13-root/actual-fast-warm-gate.json';'bytes'=[long]2818;'sha256'='042051f83aaa9c347c72d943e76d0958cf22afeccb5fd3c929bf0241cd5f5a1f'},@{'path'='verification/banc-natural-gait-m3-20260919/s28-f13-root/s1-before.json';'bytes'=[long]262140;'sha256'='39b4910b21c51042a5d68f3da51061ce2d57ead51ec5ff4cdcf82554a6493ef0'},@{'path'='verification/banc-natural-gait-m3-20260919/s28-f13-root/c5-before.json';'bytes'=[long]74674;'sha256'='3a87bc0f34b8a36ddd5153940f2b17fccbdd990b32459aea28ae92e51be58b0a'},@{'path'='verification/banc-natural-gait-m3-20260919/s28-f12-root/actual-fast-warm-gate.json';'bytes'=[long]5074;'sha256'='9c9348756e9dacf4ee53a320e50acc22d25c8fe9028419dccbc35819356d7b7e'},@{'path'='verification/banc-natural-gait-m3-20260919/s28-f12-root/actual-tool-return.json';'bytes'=[long]1846;'sha256'='71341a5233ecc031b45876ab24d1f7e4305ed79c738aa19f3efa71dc6c250162'},@{'path'='verification/banc-natural-gait-m3-20260919/s28-f12-root/actual-layout-compatibility-diagnostic.json';'bytes'=[long]1331;'sha256'='efc4694c9651f842832549809e1613b9900ab71004da098bb82dc24c29ee8a6d'},@{'path'='verification/banc-natural-gait-m3-20260919/s28-catalog-leaves-300021e7370a4dc8b163ef22f0245f31.jsonl';'bytes'=[long]6393;'sha256'='c4bb7e972e5ed5950e554983344f5d34849008f9a2a9a984a874d1ff8795926e'},@{'path'='verification/banc-natural-gait-m3-20260919/s28-f13-root/actual-tool-return.json';'bytes'=[long]2798;'sha256'='e76c924ab89dd27752d2f8a1afe68bd6839f363471886c073b3b14fce0d5d2c0'},@{'path'='verification/banc-natural-gait-m3-20260919/s28-f13-root/actual-compiler-type-refusal.json';'bytes'=[long]1968;'sha256'='448f06a882edd1757895f90dfd4bd1c7bfad38f414fcdda3909a0eacfa34e407'},@{'path'='verification/banc-natural-gait-m3-20260919/s28-catalog-leaves-30944ee8fe084e2aac23de9d47e4de80.jsonl';'bytes'=[long]496;'sha256'='b3ebd169ebfc3f700ec42a0c3b7b9292f35b7561cb6f7d100c71bf3835669606'},@{'path'='verification/banc-natural-gait-m3-20260919/s28-recovery11.json';'bytes'=[long]1882;'sha256'='0098806a84d0c18cf606482710d15e73dbe21defae360badba88a6c909db4c26'},@{'path'='verification/banc-natural-gait-m3-20260919/s28-f13-root/a318.json';'bytes'=[long]495559;'sha256'='fadfb1604441f5c2cc481e15b07eeaf10a7a7529cedc8969bc496196ea221274'},@{'path'='verification/banc-natural-gait-m3-20260919/s27-bootstrap-settled318-HOLD-20261009.json';'sha256'='70175b4bc56c19412b64500343ebc72bc538a7b3776dd60733d43efeeb59476e';'bytes'=[long]495417},@{'path'='verification/banc-natural-gait-m3-20260919/e28.json';'bytes'=[long]2443;'sha256'='e5e284cbaa1f67f1dd6e14bdf3fe74c5f902d751c387f6850700323c23609c93'},@{'path'='verification/banc-natural-gait-m3-20260919/s28-f13-root/h28-before.json';'bytes'=[long]261912;'sha256'='e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e'},@{'path'='verification/banc-natural-gait-m3-20260919/s27-actual.json';'bytes'=[long]3365;'sha256'='fcf5b1fa42a8348fa52f85e0259cc3b3554be5d96206fd5e77f4fcc54e01eb94'})){$continuityPins+=,$pin}
foreach($pin in $continuityPins){
 $path=Safe $pin.path;$saved=K-Raw $pin (K-ReaderCap $pin.path);$null=K-Recovery $sp
 try{[IO.File]::WriteAllBytes($path,([byte[]]($saved+@([byte]10))));$rejected=$false;try{$null=K-Recovery $sp}catch{$rejected=$true};Need $rejected 'Fresh bounded continuity proof raw before warm reuse';[IO.File]::Delete($path);$rejected=$false;try{$null=K-Recovery $sp}catch{$rejected=$true};Need $rejected 'Missing continuity proof before warm reuse'}finally{[IO.File]::WriteAllBytes($path,$saved)}
}
# D6 four finite direct pins: cold/warm missing/mutated refusal before catalog/cache.
$directFreeze=K-Read (K-Read $sp).source_freeze;$directContract=K-Read $directFreeze.checkpoint_contract
$directPins=@($directContract.provenance.root_design,$directContract.provenance.astra_markdown,$directContract.provenance.astra_json,$directContract.provenance.root_reflection)
$directLedgerRaw=[IO.File]::ReadAllBytes((Safe $ledgerName));$savedDirectCatalog=(Get-Item -LiteralPath Function:K-CatalogVerified).ScriptBlock
foreach($pin in $directPins){
 $path=Safe $pin.path;$saved=K-Raw $pin 262144
 foreach($warm in @($false,$true)){foreach($bad in @('mutated','missing')){
  $script:KPC.Clear();$script:KPJ=$null;if($warm){$null=K-Recovery $sp}
  $script:D6CatalogEntered=[long]0
  Set-Item -LiteralPath Function:K-CatalogVerified -Value {param($v) $script:D6CatalogEntered++;throw 'Direct provenance must refuse before catalog/backend/cache'}
  try{
   if($bad -ceq 'mutated'){[IO.File]::WriteAllBytes($path,([byte[]]($saved+@([byte]10))))}else{[IO.File]::Delete($path)}
   $rejected=$false;try{$null=K-Recovery $sp}catch{$rejected=$true}
   Need ($rejected -and $script:D6CatalogEntered -eq 0 -and $script:KPC.Count -eq 0 -and $null -eq $script:KPJ -and -not $script:KPB) 'Four direct operands refuse before catalog/cache, paired cleanup'
   Same ([Convert]::ToBase64String([IO.File]::ReadAllBytes((Safe $ledgerName)))) ([Convert]::ToBase64String($directLedgerRaw)) 'Whole raw ledger no write fee owner leak'
  }finally{[IO.File]::WriteAllBytes($path,$saved);Set-Item -LiteralPath Function:K-CatalogVerified -Value $savedDirectCatalog}
 }}
}
$script:D6ActualRead=(Get-Item -LiteralPath Function:K-Read).ScriptBlock
foreach($warm in @($false,$true)){foreach($script:D6Mismatch in @('Source-contract','review-contract','contract-alias','contract-oversized','contract-missing','contract-review')){
 $script:KPC.Clear();$script:KPJ=$null;if($warm){$null=K-Recovery $sp};$script:D6CatalogEntered=[long]0
 Set-Item -LiteralPath Function:K-CatalogVerified -Value {param($v) $script:D6CatalogEntered++;throw 'Source review c5 mismatch reached catalog'}
 Set-Item -LiteralPath Function:K-Read -Value {
  param($_p,[long]$cap=262144)
  $value=& $script:D6ActualRead $_p $cap
  if($script:D6Mismatch -ceq 'Source-contract' -and $_p.path -ceq 'verification/banc-natural-gait-m3-20260919/s28-s1.json'){$value=K-Clone $value;$value.checkpoint_contract=(K-S15Pins).previous_contract}
  elseif($script:D6Mismatch -ceq 'review-contract' -and $_p.path -ceq (K-Paths).review){$value=K-Clone $value;$value.sources[(K-Paths).contract]=(K-S15Pins).previous_contract}
  elseif($script:D6Mismatch.StartsWith('contract-') -and $_p.path -ceq (K-Paths).contract){
   $value=K-Clone $value
   switch($script:D6Mismatch){'contract-alias'{$value.provenance.root_design.path+='-alias'} 'contract-oversized'{$value.provenance.root_design.bytes=[long]262145} 'contract-missing'{$value.provenance.Remove('root_design')} 'contract-review'{$value.provenance.root_coverage=K-Clone $value.provenance.root_design}}
  }
  return $value
 }
 try{
  $rejected=$false;try{$null=K-Recovery $sp}catch{$rejected=$true}
  Need ($rejected -and $script:D6CatalogEntered -eq 0 -and $script:KPC.Count -eq 0 -and $null -eq $script:KPJ -and -not $script:KPB) 'Source review c5 schema descriptor agreement before catalog/cache'
  Same ([Convert]::ToBase64String([IO.File]::ReadAllBytes((Safe $ledgerName)))) ([Convert]::ToBase64String($directLedgerRaw)) 'Whole raw mismatch refusal no fee or owner write'
 }finally{Set-Item -LiteralPath Function:K-Read -Value $script:D6ActualRead;Set-Item -LiteralPath Function:K-CatalogVerified -Value $savedDirectCatalog}
}}
# S28 R1 same admitted case: pure read-view positives and conservative unsafe negatives.
$viewMethod=$script:KCatalogCache.type.GetMethod('ReadView',[Reflection.BindingFlags]'NonPublic,Static')
Need ($viewMethod.DeclaringType -eq $script:KCatalogCache.type -and $viewMethod.ReturnType -eq [bool]) 'Actual source-owned private read-view control'
foreach($viewValue in @($null,$false,[int]0,[long]0,[double]0,[double]-0.0,'plain')){
 $viewArgs=[object[]]::new(2);$viewArgs[0]=$viewValue
 Same (K-CatalogInvoke $viewMethod $viewArgs) $true 'Bare/null/scalar view positive in existing owned case'
 if($null -eq $viewValue){Same $viewArgs[1] $null 'Bare null view preserved without engine access'}else{Need ([object]::ReferenceEquals($viewArgs[1],$viewValue)) 'Read-view same reference and original boxing'}
}
"""+historical_fixture.d19_readview_box_identity()+"""
"""+historical_fixture.d19_readview_unsafe_members()+"""
$mapStorage=$script:KCatalogCache.type.GetMethod('MapStorage',[Reflection.BindingFlags]'NonPublic,Static')
$viewArgs=[object[]]::new(1);$viewArgs[0]=[Collections.Specialized.OrderedDictionary]::new()
Same (K-CatalogInvoke $mapStorage $viewArgs) $false 'Cold ordered storage refused before lazy Keys/enumerator writes'
$viewArgs[0]=[Collections.Hashtable]::new([StringComparer]::CurrentCulture)
Same (K-CatalogInvoke $mapStorage $viewArgs) $false 'Unlisted comparer identity refused without comparer test calls'
"""+historical_fixture.d19_typed_source_field_weak()+"""
# D9 actual same-box CWT association controls in this existing admitted process only.
$d7Flags=[Reflection.BindingFlags]'NonPublic,Static';$d7Instance=[Reflection.BindingFlags]'DeclaredOnly,NonPublic,Instance'
$d7Member=[PSObject].Assembly.GetType('System.Management.Automation.PSMemberInfoInternalCollection'+[char]96+'1',$true).MakeGenericType([System.Management.Automation.PSMemberInfo])
$d7Parent=[PSObject].Assembly.GetType('System.Management.Automation.PSMemberInfoCollection'+[char]96+'1',$true).MakeGenericType([System.Management.Automation.PSMemberInfo])
Need ($d7Member.BaseType -eq $d7Parent -and $d7Parent.BaseType -eq [object] -and $d7Parent.GetFields([Reflection.BindingFlags]'DeclaredOnly,Public,NonPublic,Instance').Length -eq 0) 'Actual exact storage-free parent layout'
$d7Fields=$d7Member.GetFields([Reflection.BindingFlags]'DeclaredOnly,Public,NonPublic,Instance,Static')
Need ($d7Fields.Length -eq 2) 'Actual all-visibility two declared fields including static'
$d7Storage=$d7Member.GetField('_members',$d7Instance);$d7Hidden=$d7Member.GetField('_countHidden',$d7Instance)
foreach($d7Field in @($d7Storage,$d7Hidden)){Need ($d7Field.DeclaringType -eq $d7Member -and $d7Field.IsPrivate -and -not $d7Field.IsStatic -and -not $d7Field.IsInitOnly -and -not $d7Field.IsLiteral -and -not $d7Field.IsDefined([ThreadStaticAttribute],$false)) 'Actual exact mutable instance nonliteral nonthread fields'}
Need ($d7Storage.FieldType -eq [Collections.Specialized.OrderedDictionary] -and $d7Hidden.FieldType -eq [int]) 'Actual backing field types'
$d7TableField=[PSObject].GetField('s_instanceMembersResurrectionTable',$d7Flags);$d7NamesField=[PSObject].GetField('s_typeNamesResurrectionTable',$d7Flags)
$d7Table=$d7TableField.GetValue($null);$d7Names=$d7NamesField.GetValue($null)
$d7Add=$d7Table.GetType().GetMethod('Add',[type[]]@([object],$d7Member));$d7Remove=$d7Table.GetType().GetMethod('Remove',[type[]]@([object]));$d7Lookup=$d7Table.GetType().GetMethod('TryGetValue',[type[]]@([object],$d7Member.MakeByRefType()))
$d7NameType=[PSObject].Assembly.GetType('System.Management.Automation.Runspaces.ConsolidatedString',$true)
$d7NameAdd=$d7Names.GetType().GetMethod('Add',[type[]]@([object],$d7NameType));$d7NameRemove=$d7Names.GetType().GetMethod('Remove',[type[]]@([object]));$d7NameLookup=$d7Names.GetType().GetMethod('TryGetValue',[type[]]@([object],$d7NameType.MakeByRefType()))
$d7Dispatch=[Reflection.MethodBase].GetMethod('Invoke',[type[]]@([object],[object[]]));$d7FieldRead=[Reflection.FieldInfo].GetMethod('GetValue',[type[]]@([object]));$d7ObjectType=[object].GetMethod('GetType',[type[]]@())
$d7Inert=$script:KCatalogCache.type.GetMethod('InertMembers',[Reflection.BindingFlags]'NonPublic,Static')
$d7IP=$d7Inert.GetParameters();Need ($d7Inert.DeclaringType -eq $script:KCatalogCache.type -and $d7Inert.ReturnType -eq [bool] -and $d7IP.Length -eq 4 -and $d7IP[0].ParameterType -eq [object] -and $d7IP[1].ParameterType -eq [object] -and $d7IP[2].ParameterType -eq [type] -and $d7IP[3].ParameterType -eq [int].MakeByRefType() -and $d7IP[3].IsOut) 'Actual private InertMembers4 ABI'
foreach($d7Wrong in @($null,[object]::new())){$d7IA=[object[]]@($null,$d7Wrong,$d7Member,[int]0);Need (-not (R28-SourceInvoke $d7Inert $d7IA) -and $d7IA[3] -eq 57) 'Real private helper wrong-type57, distinct from typed-CWT reachability'}
$d7Slot=$script:KCatalogCache.type.GetField('RefusalSlot',[Reflection.BindingFlags]'NonPublic,Static');$d7Seq=$script:KCatalogCache.type.GetField('RefusalSequence',[Reflection.BindingFlags]'NonPublic,Static')
$d7Ledger=[IO.File]::ReadAllBytes((Safe $ledgerName));$script:D7GetterCalls=[long]0
$d8All=[Reflection.BindingFlags]'DeclaredOnly,Public,NonPublic,Instance,Static'
$d8DF=[Reflection.BindingFlags]'DeclaredOnly,NonPublic,Instance'
$d8DT=[Collections.Specialized.OrderedDictionary];$d8LT=[Collections.ArrayList]
$d8DFNames=@('_objectsArray','_objectsTable','_initialCapacity','_comparer','_readOnly','_siInfo');$d8DFields=@();foreach($d8N in $d8DFNames){$d8DFields+=,$d8DT.GetField($d8N,$d8DF)}
$d8LFields=@();foreach($d8N in @('_items','_size','_version')){$d8LFields+=,$d8LT.GetField($d8N,$d8DF)}
$d8DAll=$d8DT.GetFields($d8All);$d8LAll=$d8LT.GetFields($d8All)
Need ($d8DT.BaseType -eq [object] -and $d8DAll.Length -cin @(6,10) -and $d8LT.BaseType -eq [object] -and $d8LAll.Length -cin @(3,4)) 'Independent coupled BCL metadata6/10 and3/4'
$d8LiteralMask=0
foreach($d8F in $d8DAll){
 if($d8F.Name -cin $d8DFNames){$d8I=[array]::IndexOf($d8DFNames,$d8F.Name);$d8Types=@([Collections.ArrayList],[Collections.Hashtable],[int],[Collections.IEqualityComparer],[bool],[Runtime.Serialization.SerializationInfo]);Need ($d8F.DeclaringType -eq $d8DT -and $d8F.FieldType -eq $d8Types[$d8I] -and $d8F.IsPrivate -and -not $d8F.IsStatic -and -not $d8F.IsLiteral -and $d8F.IsInitOnly -eq ($d8I -eq 5) -and [int]$d8F.Attributes -eq $(if($d8I -eq 5){33}else{1}) -and $d8F.GetRequiredCustomModifiers().Length -eq 0 -and $d8F.GetOptionalCustomModifiers().Length -eq 0 -and -not $d8F.IsDefined([ThreadStaticAttribute],$false)) 'Exact dictionary six storage fields'}
 else{$d8LI=[array]::IndexOf(@('KeyComparerName','ArrayListName','ReadOnlyName','InitCapacityName'),$d8F.Name);Need ($d8LI -ge 0 -and $d8F.IsPrivate -and $d8F.IsStatic -and $d8F.IsLiteral -and -not $d8F.IsInitOnly -and [int]$d8F.Attributes -eq 32849 -and $d8F.GetRequiredCustomModifiers().Length -eq 0 -and $d8F.GetOptionalCustomModifiers().Length -eq 0 -and $d8F.FieldType -eq [string] -and ($d8LiteralMask -band (1 -shl $d8LI)) -eq 0) 'Exact unique optional dictionary literal';$d8LiteralMask=$d8LiteralMask -bor (1 -shl $d8LI)}
}
Need ($d8LiteralMask -eq $(if($d8DAll.Length -eq 10){15}else{0})) 'Coupled literal mask0/15'
$d8Constant=$d8LT.GetField('_defaultCapacity',[Reflection.BindingFlags]'DeclaredOnly,NonPublic,Static')
Need (($null -ne $d8Constant) -eq ($d8LAll.Length -eq 4)) 'Coupled ArrayList optional constant metadata'
if($null -ne $d8Constant){$d8RawConstant=$d8Constant.GetRawConstantValue();Need ($d8Constant.IsPrivate -and $d8Constant.IsStatic -and $d8Constant.IsLiteral -and -not $d8Constant.IsInitOnly -and [int]$d8Constant.Attributes -eq 32849 -and $d8Constant.GetRequiredCustomModifiers().Length -eq 0 -and $d8Constant.GetOptionalCustomModifiers().Length -eq 0 -and $d8Constant.FieldType -eq [int] -and $d8RawConstant.GetType() -eq [int] -and $d8RawConstant -eq 4) 'Mandatory rawInt32 four if optional constant exists'}
foreach($d8F in $d8LAll){if($d8F.Name -cne '_defaultCapacity'){$d8I=[array]::IndexOf(@('_items','_size','_version'),$d8F.Name);Need ($d8I -ge 0 -and $d8F.DeclaringType -eq $d8LT -and $d8F.IsPrivate -and -not $d8F.IsStatic -and -not $d8F.IsInitOnly -and -not $d8F.IsLiteral -and [int]$d8F.Attributes -eq 1 -and $d8F.GetRequiredCustomModifiers().Length -eq 0 -and $d8F.GetOptionalCustomModifiers().Length -eq 0 -and $d8F.FieldType -eq $(if($d8I -eq 0){[object[]]}else{[int]}) -and -not $d8F.IsDefined([ThreadStaticAttribute],$false)) 'Exact three mutable ArrayList storage fields'}}
$d8StoreHelper=$script:KCatalogCache.type.GetMethod('PristineStore',[Reflection.BindingFlags]'NonPublic,Static')
$d8Wrong=[object[]]@([object]::new(),$null,$null,[int]0);Need (-not $d8StoreHelper.Invoke($null,$d8Wrong) -and $d8Wrong[3] -eq 62) 'Private helper62; typed field branch NOT CLAIMED'
$d9Helper=$script:KCatalogCache.type.GetMethod('HiddenDiagnostic',[Reflection.BindingFlags]'NonPublic,Static')
$d9Params=$d9Helper.GetParameters();Need ($d9Params.Length -eq 6 -and $d9Params[0].ParameterType -eq [object] -and $d9Params[1].ParameterType -eq [object] -and $d9Params[2].ParameterType -eq [object] -and $d9Params[3].ParameterType -eq [int] -and $d9Params[4].ParameterType -eq [Reflection.FieldInfo] -and $d9Params[5].ParameterType -eq [Reflection.FieldInfo]) 'Actual private HiddenDiagnostic6 ABI'
$d9Wrong=[object[]]@($null,$null,[object]::new(),[int]1,$null,$null)
Need ((R28-SourceInvoke $d9Helper $d9Wrong) -eq 71) 'Private helper wrong-store71 only; no typed-CWT reachability claim'
$d8Modes=@('default','initialized-empty','ignorecase-empty','lazy-empty','table-empty','capacity-one','readonly','serialization','ordinal','foreign-comparer','list-wrapper','list-populated','list-versioned','list-capacity','list-covariant','script-member','hidden','names','fresh-observation','diag-negative','diag-high','diag-null-array','diag-list-null','diag-list-four-table','diag-list-unknown-table','diag-list-mismatch','diag-list-capacity','diag-list-wrapper','diag-list-zero-version','diag-list-too-long','diag-list-covariant','diag-five-null','diag-five-table','diag-five-spoof-null','diag-five-spoof-table','diag-five-bad-key','diag-five-case','diag-five-duplicate','diag-five-bad-type')
foreach($d8Wrapped in @($false,$true)){
foreach($d7Mode in $d8Modes){
 $d7Args=[object[]]::new(2);$d7Args[0]=[long](8000000+100*[int]$d8Wrapped+[array]::IndexOf($d8Modes,$d7Mode))
 $d9EntryBefore=$null;$d9EntryFields=$null
 $d7Collection=[Activator]::CreateInstance($d7Member,$true);$d7Store=$null;$d7HiddenValue=[int]0;$d7Reason=[int]0
 if($d7Mode -cin @('initialized-empty','ignorecase-empty','lazy-empty','table-empty','capacity-one','readonly','serialization','ordinal','foreign-comparer','list-wrapper','list-populated','list-versioned','list-capacity','list-covariant','script-member')){$d7Store=[Collections.Specialized.OrderedDictionary]::new()}
 if($d7Mode -ceq 'ignorecase-empty'){$d7Store=[Collections.Specialized.OrderedDictionary]::new([StringComparer]::OrdinalIgnoreCase)}
 if($d7Mode -ceq 'lazy-empty'){$null=$d7Store.GetEnumerator()}
 if($d7Mode -ceq 'table-empty'){$d8DFields[1].SetValue($d7Store,[Collections.Hashtable]::new());$d7Reason=64}
 if($d7Mode -ceq 'capacity-one'){$d7Store=[Collections.Specialized.OrderedDictionary]::new(1);$d7Reason=64}
 if($d7Mode -ceq 'readonly'){$d8DFields[4].SetValue($d7Store,$true);$d7Reason=64}
 if($d7Mode -ceq 'serialization'){$d8Info=[Runtime.Serialization.SerializationInfo]::new($d8DT,[Runtime.Serialization.FormatterConverter]::new());$d8DFields[5].SetValue($d7Store,$d8Info);$d7Reason=64}
 if($d7Mode -ceq 'ordinal'){$d8DFields[3].SetValue($d7Store,[StringComparer]::Ordinal);$d7Reason=64}
 if($d7Mode -ceq 'foreign-comparer'){$d8DFields[3].SetValue($d7Store,[StringComparer]::InvariantCulture);$d7Reason=64}
 if($d7Mode.StartsWith('list-')){
  $d8List=[Collections.ArrayList]::new(0);$d7Reason=66
  if($d7Mode -ceq 'list-wrapper'){$d8List=[Collections.ArrayList]::ReadOnly($d8List);$d7Reason=65}
  if($d7Mode -ceq 'list-populated'){$null=$d8List.Add('fresh-isolated-value')}
  if($d7Mode -ceq 'list-versioned'){$null=$d8List.Add('removed');$d8List.RemoveAt(0);$d8List.Capacity=0}
  if($d7Mode -ceq 'list-capacity'){$d8List=[Collections.ArrayList]::new(1)}
  if($d7Mode -ceq 'list-covariant'){$d8LFields[0].SetValue($d8List,[string[]]::new(0))}
  $d8DFields[0].SetValue($d7Store,$d8List)
  Need ($null -eq $d8DFields[1].GetValue($d7Store)) 'Independent denied list with table-null reaches66/65'
 }
 if($d7Mode -ceq 'script-member'){$d7Store.Add('x',[System.Management.Automation.PSScriptProperty]::new('x',{$script:D7GetterCalls++;throw 'D8 getter must never execute'}));$d7Reason=64}
 if($d7Mode -ceq 'hidden'){$d7HiddenValue=[int]1;$d7Reason=70}
 if($d7Mode -ceq 'diag-negative'){$d7HiddenValue=[int]-1;$d7Reason=68}
 if($d7Mode -ceq 'diag-high'){$d7HiddenValue=[int]6;$d7Reason=69}
 if($d7Mode -ceq 'diag-null-array'){$d7HiddenValue=[int]1;$d7Store=[Collections.Specialized.OrderedDictionary]::new();$d7Reason=74}
 if($d7Mode.StartsWith('diag-list-') -or $d7Mode.StartsWith('diag-five-')){
  $d7Store=[Collections.Specialized.OrderedDictionary]::new();$d9List=[Collections.ArrayList]::new(0)
  $d7HiddenValue=if($d7Mode.StartsWith('diag-five-')){[int]5}else{[int]1}
  if($d7Mode -ceq 'diag-list-four-table'){$d7HiddenValue=[int]4}
  $d9Size=$d7HiddenValue;$d9Items=[object[]]::new($d9Size)
  for($d9I=0;$d9I -lt $d9Size;$d9I++){$d9Items[$d9I]='fresh-bounded-isolated'}
  $d7Reason=77
  if($d7Mode -ceq 'diag-list-four-table'){$d8DFields[1].SetValue($d7Store,[Collections.Hashtable]::new());$d7Reason=78}
  if($d7Mode -ceq 'diag-list-unknown-table'){$d8DFields[1].SetValue($d7Store,[Collections.Hashtable]::Synchronized([Collections.Hashtable]::new()));$d7Reason=60}
  if($d7Mode -ceq 'diag-list-mismatch'){$d7HiddenValue=[int]2;$d7Reason=79}
  if($d7Mode -ceq 'diag-list-capacity'){$d8DFields[2].SetValue($d7Store,[int]1);$d7Reason=73}
  if($d7Mode -ceq 'diag-list-wrapper'){$d9List=[Collections.ArrayList]::ReadOnly($d9List);$d7Reason=75}
  if($d7Mode -ceq 'diag-list-zero-version'){$d7Reason=76}
  if($d7Mode -ceq 'diag-list-too-long'){$d9Items=[object[]]::new(17);$d7Reason=76}
  if($d7Mode -ceq 'diag-list-covariant'){$d9Items=[string[]]::new(1);$d7Reason=76}
  if($d7Mode.StartsWith('diag-five-')){
   # Manual fresh constructor controls are not engine-origin evidence. All remain refused.
   $d9Internal=[PSObject].Assembly.GetType('System.Management.Automation.PSInternalMemberSet',$true)
   $d9InternalCtor=$d9Internal.GetConstructor([Reflection.BindingFlags]'Public,NonPublic,Instance',$null,[type[]]@([string],[PSObject]),$null)
   $d9CodeCtor=[System.Management.Automation.PSCodeProperty].GetConstructor([Reflection.BindingFlags]'Public,NonPublic,Instance',$null,[type[]]@([string]),$null)
   Need ($null -ne $d9InternalCtor -and $null -ne $d9CodeCtor) 'Existing source-feasible SMA constructors; unavailable is SOURCE_ABI_ONLY, never substitute a class'
   $d9OwnerArgs=[object[]]::new(1);$d9OwnerArgs[0]=[object]::new();$d9OwnerSlot=[object[]]::new(1);$d9OwnerSlot[0]=[PSObject].GetConstructor([type[]]@([object])).Invoke($d9OwnerArgs)
   $d9OwnerDispatch=[object[]]::new(2);$d9OwnerDispatch[0]=$d9OwnerSlot[0];Need ($d7Dispatch.Invoke($d7ObjectType,$d9OwnerDispatch) -eq [PSObject]) 'Actual original fresh PSObject owner argument slot'
   $d9Names=@('pstypenames','psadapted','psextended','psobject','psbase')
   $d9EntryFields=@([Collections.DictionaryEntry].GetField('_key',$d8DF),[Collections.DictionaryEntry].GetField('_value',$d8DF))
   Need ([Collections.DictionaryEntry].IsValueType -and [Collections.DictionaryEntry].BaseType -eq [ValueType] -and [Collections.DictionaryEntry].GetFields($d8All).Length -eq 2) 'Independent original boxed-entry layout'
   foreach($d9F in $d9EntryFields){Need ($null -ne $d9F -and $d9F.DeclaringType -eq [Collections.DictionaryEntry] -and $d9F.FieldType -eq [object] -and $d9F.IsPrivate -and -not $d9F.IsStatic -and -not $d9F.IsInitOnly -and -not $d9F.IsLiteral -and -not $d9F.IsDefined([ThreadStaticAttribute],$false)) 'Independent mutable Object key/value fields'}
   for($d9I=0;$d9I -lt 5;$d9I++){
    $d9Name=$d9Names[$d9I];$d9MemberName=if($d7Mode -cin @('diag-five-spoof-null','diag-five-spoof-table')){'custom-spoof-member'}else{$d9Name}
    if($d9Name -ceq 'psextended'){$d9Value=[System.Management.Automation.PSMemberSet]::new($d9MemberName)}
    elseif($d9Name -ceq 'pstypenames'){$d9CtorArgs=[object[]]::new(1);$d9CtorArgs[0]=$d9MemberName;$d9Value=$d9CodeCtor.Invoke($d9CtorArgs)}
    else{$d9CtorArgs=[object[]]::new(2);$d9CtorArgs[0]=$d9MemberName;$d9CtorArgs[1]=$d9OwnerSlot[0];$d9Value=$d9InternalCtor.Invoke($d9CtorArgs)}
    $d9Key=$d9Name
    if($d7Mode -ceq 'diag-five-bad-key' -and $d9I -eq 2){$d9Key='unrecognized'}
    if($d7Mode -ceq 'diag-five-case' -and $d9I -eq 2){$d9Key='PSEXTENDED'}
    if($d7Mode -ceq 'diag-five-duplicate' -and $d9I -eq 4){$d9Key='psobject'}
    if($d7Mode -ceq 'diag-five-bad-type' -and $d9I -eq 2){$d9Value=[System.Management.Automation.PSNoteProperty]::new('psextended','spoof')}
    $d9Items[$d9I]=[Collections.DictionaryEntry]::new($d9Key,$d9Value)
   }
   $d7Reason=82
   if($d7Mode -cin @('diag-five-table','diag-five-spoof-table')){$d8DFields[1].SetValue($d7Store,[Collections.Hashtable]::new());$d7Reason=60}
   if($d7Mode -cin @('diag-five-bad-key','diag-five-case','diag-five-duplicate','diag-five-bad-type')){$d7Reason=80}
  }
  if($d7Mode -cne 'diag-list-wrapper'){
   $d8LFields[0].SetValue($d9List,$d9Items);$d8LFields[1].SetValue($d9List,[int]$d9Size);$d8LFields[2].SetValue($d9List,[int]1)
   if($d7Mode -ceq 'diag-list-zero-version'){$d8LFields[2].SetValue($d9List,[int]0)}
  }
  $d8DFields[0].SetValue($d7Store,$d9List)
 }
 $d7Storage.SetValue($d7Collection,$d7Store);$d7Hidden.SetValue($d7Collection,$d7HiddenValue)
 $d7Base=[object[]]::new(1);$d7Base[0]=$d7Args[0]
 if($d8Wrapped){
  $d7WC=[object[]]::new(1);$d7WC[0]=$d7Args[0];$d7Args[0]=[PSObject].GetConstructor([type[]]@([object])).Invoke($d7WC)
  $d7WA=[object[]]::new(1);$d7WA[0]=$d7Args[0];$d7DI=[object[]]::new(2);$d7DI[0]=$d7Args[0]
  Need ($d7Dispatch.Invoke($d7ObjectType,$d7DI) -eq [PSObject]) 'Actual RAW PSObject kept in production argument slot, no implicit unwrap'
  $d7Immediate=[PSObject].GetProperty('ImmediateBaseObject').GetGetMethod();$d7Base[0]=$d7Dispatch.Invoke($d7Immediate,$d7DI)
  foreach($d7Direct in @('_instanceMembers','_typeNames','_adapterSet')){Need ($null -eq $d7FieldRead.Invoke([PSObject].GetField($d7Direct,$d7Instance),$d7WA)) 'Wrapped direct metadata null, avoid reason4 preemption'}
  $d7Bits=[Convert]::ToByte($d7FieldRead.Invoke([PSObject].GetField('_flags',$d7Instance),$d7WA))
  Need (($d7Bits -band (-bnot 16)) -eq 0 -and ($d7Bits -band 16) -eq 0) 'Wrapped exact expression selects base box'
 }
 $d7Key=[object[]]::new(1);$d7Key[0]=$d7Base[0]
 $d7AddArgs=[object[]]::new(2);$d7AddArgs[0]=$d7Key[0];$d7AddArgs[1]=$d7Collection
 $d7LookupArgs=[object[]]::new(2);$d7LookupArgs[0]=$d7Key[0]
 $d7NameArgs=[object[]]::new(2);$d7NameArgs[0]=$d7Key[0]
 $d7Cleanup=[object[]]::new(1);$d7Cleanup[0]=$d7Key[0]
 Need ([object]::ReferenceEquals($d7Base[0],$d7Key[0]) -and [object]::ReferenceEquals($d7Key[0],$d7AddArgs[0]) -and [object]::ReferenceEquals($d7Key[0],$d7LookupArgs[0])) 'Same original boxed key argument slots, no reboxed positive'
 if(-not $d8Wrapped){Need ([object]::ReferenceEquals($d7Args[0],$d7Key[0])) 'Production input slot is SAME association box before call'}
 $d7Added=$false;$d7NamesAdded=$false
 try{
  Need (-not $d7Lookup.Invoke($d7Table,$d7LookupArgs)) 'Fresh isolated key has no prior association'
  $null=$d7Add.Invoke($d7Table,$d7AddArgs);$d7Added=$true
  if($d7Mode -ceq 'names'){
   $d7Ctor=$d7NameType.GetConstructor([Reflection.BindingFlags]'Public,NonPublic,Instance',$null,[type[]]@([Collections.Generic.IEnumerable[string]]),$null)
   Need ($null -ne $d7Ctor) 'Actual existing ConsolidatedString enumerable constructor'
   $d7CtorArgs=[object[]]::new(1);$d7CtorArgs[0]=[string[]]@('D7.fixture.names');$d7NameArgs[1]=$d7Ctor.Invoke($d7CtorArgs)
   Need ([object]::ReferenceEquals($d7NameArgs[0],$d7Key[0])) 'Names same key slot'
   $null=$d7NameAdd.Invoke($d7Names,$d7NameArgs);$d7NamesAdded=$true;$d7Reason=56
  }
  Need ($d7Lookup.Invoke($d7Table,$d7LookupArgs) -and [object]::ReferenceEquals($d7LookupArgs[1],$d7AddArgs[1])) 'Real SAME box association SAME collection before production'
  if($d7NamesAdded){$d7NL=[object[]]::new(2);$d7NL[0]=$d7Key[0];Need ($d7NameLookup.Invoke($d7Names,$d7NL) -and [object]::ReferenceEquals($d7NL[1],$d7NameArgs[1])) 'Real names association before production'}
  $d9CommonStore=$d7Storage.GetValue($d7Collection);$d9CommonHidden=$d7Hidden.GetValue($d7Collection)
  Need ([object]::ReferenceEquals($d9CommonStore,$d7Store) -and $d9CommonHidden.GetType() -eq [int] -and $d9CommonHidden -eq $d7HiddenValue) 'Direct common fields before actual production call'
  $d8Before=[object[]]::new(6);$d8ListBefore=[object[]]::new(3)
 if($null -ne $d7Store){
  for($d8I=0;$d8I -lt 6;$d8I++){$d8Before[$d8I]=$d8DFields[$d8I].GetValue($d7Store)}
  if($d7Mode -cin @('initialized-empty','ignorecase-empty','lazy-empty')){
   Need ($null -eq $d8Before[1] -and $d8Before[2].GetType() -eq [int] -and $d8Before[2] -eq 0 -and ($null -eq $d8Before[3] -or [object]::ReferenceEquals($d8Before[3],[StringComparer]::OrdinalIgnoreCase)) -and $d8Before[4].GetType() -eq [bool] -and -not $d8Before[4] -and $null -eq $d8Before[5]) 'Independently proved pristine dictionary scalar/reference state'
   Need (($d7Mode -ceq 'lazy-empty') -eq ($null -ne $d8Before[0])) 'Positive expected lazy list allocation only during setup'
  }
  if($null -ne $d8Before[0]){
   if($d8Before[0].GetType() -eq $d8LT){
    for($d8I=0;$d8I -lt 3;$d8I++){$d8ListBefore[$d8I]=$d8LFields[$d8I].GetValue($d8Before[0])}
    if($d7Mode -ceq 'lazy-empty'){Need ($d8ListBefore[0].GetType() -eq [object[]] -and $d8ListBefore[0].Length -eq 0 -and $d8ListBefore[1] -eq 0 -and $d8ListBefore[2] -eq 0) 'Independent exact Object[] length0 size0 version0'}
   }else{Need ($d7Mode -cin @('list-wrapper','diag-list-wrapper')) 'Only deliberate denied existing BCL wrapper'}
  }
 }
 if($d7Mode.StartsWith('diag-five-')){
  $d9EntryBefore=[object[]]::new(15)
  for($d9I=0;$d9I -lt 5;$d9I++){
   $d9EntryBefore[3*$d9I]=$d9Items[$d9I]
   $d9EntryBefore[3*$d9I+1]=$d9EntryFields[0].GetValue($d9EntryBefore[3*$d9I])
   $d9EntryBefore[3*$d9I+2]=$d9EntryFields[1].GetValue($d9EntryBefore[3*$d9I])
  }
 }
 $d7SavedSlot=$d7Slot.GetValue($null);$d7Generation=$d7Seq.GetValue($null)
  try{
   $d7Reset=[object[]]@(1,[Threading.Thread]::CurrentThread.ManagedThreadId,$d7Generation,0,0,0,-1,0,'NONE');$d7Slot.SetValue($null,$d7Reset)
   $d7Supported=$viewMethod.Invoke($null,$d7Args)
   Need ($d7Supported -eq ($d7Reason -eq 0)) ('D9 actual ReadView outcome '+$d7Mode)
   Need ([object]::ReferenceEquals($d7Args[1],$d7Base[0])) 'ReadView returns SAME original base box on accepted and refused controls'
   $d7ActualSlot=$d7Slot.GetValue($null);Need ($d7ActualSlot[5] -eq $d7Reason -and $d7Seq.GetValue($null) -eq $d7Generation) ('D9 exact first refusal '+$d7Mode)
  }finally{$d7Slot.SetValue($null,$d7SavedSlot)}
  $d7LookupArgs[1]=$null
  Need ($d7Lookup.Invoke($d7Table,$d7LookupArgs) -and [object]::ReferenceEquals($d7LookupArgs[1],$d7AddArgs[1]) -and [object]::ReferenceEquals($d7LookupArgs[0],$d7Key[0])) 'Real SAME box association SAME collection after production'
  Need ([object]::ReferenceEquals($d7Storage.GetValue($d7Collection),$d9CommonStore) -and $d7Hidden.GetValue($d7Collection) -eq $d9CommonHidden) 'Same directly observed common refs/scalars after production'
  if($d7NamesAdded){$d7NL[1]=$null;Need ($d7NameLookup.Invoke($d7Names,$d7NL) -and [object]::ReferenceEquals($d7NL[1],$d7NameArgs[1])) 'Names association unchanged after production'}
  if($null -ne $d7Store){
   for($d8I=0;$d8I -lt 6;$d8I++){$d8After=$d8DFields[$d8I].GetValue($d7Store);if($d8I -cin @(2,4)){Need ($d8After -ceq $d8Before[$d8I]) 'Backing scalar unchanged'}else{Need ([object]::ReferenceEquals($d8After,$d8Before[$d8I])) 'Backing reference unchanged'}}
   if($null -ne $d8Before[0] -and $d8Before[0].GetType() -eq $d8LT){for($d8I=0;$d8I -lt 3;$d8I++){$d8After=$d8LFields[$d8I].GetValue($d8Before[0]);if($d8I -eq 0){Need ([object]::ReferenceEquals($d8After,$d8ListBefore[$d8I])) 'Input items array reference unchanged'}else{Need ($d8After -ceq $d8ListBefore[$d8I]) 'Input list scalar unchanged'}}}
  }
  if($null -ne $d9EntryBefore){
   for($d9I=0;$d9I -lt 5;$d9I++){
    Need ([object]::ReferenceEquals($d9Items[$d9I],$d9EntryBefore[3*$d9I])) 'Original boxed entry ref unchanged'
    Need ([object]::ReferenceEquals($d9EntryFields[0].GetValue($d9EntryBefore[3*$d9I]),$d9EntryBefore[3*$d9I+1])) 'Original key ref unchanged'
    Need ([object]::ReferenceEquals($d9EntryFields[1].GetValue($d9EntryBefore[3*$d9I]),$d9EntryBefore[3*$d9I+2])) 'Original value ref unchanged'
   }
  }
  if($d8Wrapped){
   Need ([object]::ReferenceEquals($d7Args[0],$d7WA[0]) -and $d7Dispatch.Invoke($d7ObjectType,$d7DI) -eq [PSObject]) 'Same original RAW wrapper after production'
   foreach($d7Direct in @('_instanceMembers','_typeNames','_adapterSet')){Need ($null -eq $d7FieldRead.Invoke([PSObject].GetField($d7Direct,$d7Instance),$d7WA)) 'Same wrapper direct metadata refs after production'}
   Need ([Convert]::ToByte($d7FieldRead.Invoke([PSObject].GetField('_flags',$d7Instance),$d7WA)) -eq $d7Bits) 'Same wrapper flags after production'
  }
  Need ([object]::ReferenceEquals($d7TableField.GetValue($null),$d7Table) -and [object]::ReferenceEquals($d7NamesField.GetValue($null),$d7Names)) 'Each site retains engine identity'
  Same ([Convert]::ToBase64String([IO.File]::ReadAllBytes((Safe $ledgerName)))) ([Convert]::ToBase64String($d7Ledger)) 'Each site whole raw ledger equality'
  Need ($script:D7GetterCalls -eq 0) 'Each site callback0'
  if($d7Mode -ceq 'fresh-observation'){
   $d7Changed=[Collections.Specialized.OrderedDictionary]::new(1);$d7Storage.SetValue($d7Collection,$d7Changed)
   $d7IA=[object[]]@($d7Key[0],$d7Collection,$d7Member,[int]0);Need (-not (R28-SourceInvoke $d7Inert $d7IA) -and $d7IA[3] -eq 64) 'Actual capacity1 sequential later refusal64; no within-call61/67 claim'
   Need ([object]::ReferenceEquals($d7Storage.GetValue($d7Collection),$d7Changed)) 'Denied observation does not write fresh storage'
  }
 }finally{
  try{if($d7NamesAdded){Need ($d7NameRemove.Invoke($d7Names,$d7Cleanup)) 'Remove only fresh names fixture entry'}}
  finally{if($d7Added){Need ($d7Remove.Invoke($d7Table,$d7Cleanup)) 'Remove only fresh members fixture entry'}}
 }
 $d7LookupArgs[1]=$null;Need (-not $d7Lookup.Invoke($d7Table,$d7LookupArgs)) 'Fresh association cleanup complete'
 if($d7NamesAdded){$d7NL[1]=$null;Need (-not $d7NameLookup.Invoke($d7Names,$d7NL)) 'Fresh names association cleanup complete'}
}
}
Need ($script:D7GetterCalls -eq 0) 'Actual zero callbacks including populated ScriptProperty'
Need ([object]::ReferenceEquals($d7TableField.GetValue($null),$d7Table) -and [object]::ReferenceEquals($d7NamesField.GetValue($null),$d7Names)) 'Production preserves original engine CWT identities'
Same ([Convert]::ToBase64String([IO.File]::ReadAllBytes((Safe $ledgerName)))) ([Convert]::ToBase64String($d7Ledger)) 'D7 complete raw ledger no fee/owner/input write'
# S28 ROOT residual controls: a separate Source-bound helper in the same admitted process.
$fixtureSource=@'
using System;
using System.Collections;
using System.Collections.Specialized;
using System.Collections.ObjectModel;
using System.Management.Automation;
using System.Management.Automation.Runspaces;
using System.Reflection;
public sealed class S28PurityFixture_a91caea129ac11a18a774ad8aa744778be869cd53380ef1060900857c623e1e4 : PSPropertyAdapter, IEqualityComparer
{
public static long EqualCalls, HashCalls;
public static readonly long[] AdapterCalls = new long[9];
static void Need(bool value, string label) { if (!value) throw new InvalidOperationException(label); }
static Exception Callback(int i) { AdapterCalls[i]++; return new InvalidOperationException("Adapter callback executed"); }
bool IEqualityComparer.Equals(object a, object b) { EqualCalls++; throw new InvalidOperationException("Comparer Equals executed"); }
int IEqualityComparer.GetHashCode(object value) { HashCalls++; throw new InvalidOperationException("Comparer GetHashCode executed"); }
public override Collection<PSAdaptedProperty> GetProperties(object value) { throw Callback(0); }
public override PSAdaptedProperty GetProperty(object value, string name) { throw Callback(1); }
public override string GetPropertyTypeName(PSAdaptedProperty property) { throw Callback(2); }
public override object GetPropertyValue(PSAdaptedProperty property) { throw Callback(3); }
public override bool IsGettable(PSAdaptedProperty property) { throw Callback(4); }
public override bool IsSettable(PSAdaptedProperty property) { throw Callback(5); }
public override void SetPropertyValue(PSAdaptedProperty property, object value) { throw Callback(6); }
public override Collection<string> GetTypeNameHierarchy(object value) { throw Callback(7); }
public override PSAdaptedProperty GetFirstPropertyOrDefault(object value, MemberNamePredicate predicate) { throw Callback(8); }
static FieldInfo Field(Type type, string name, Type fieldType)
{
var f=type.GetField(name,BindingFlags.NonPublic|BindingFlags.Instance|BindingFlags.DeclaredOnly);
Need(f!=null && f.DeclaringType==type && f.FieldType==fieldType && f.IsPrivate && !f.IsStatic && !f.IsInitOnly,"Exact fresh test field");
return f;
}
static void Refuse(Type catalog, MethodInfo support, object value, int reason)
{
var flags=BindingFlags.NonPublic|BindingFlags.Static|BindingFlags.DeclaredOnly;
var slotField=catalog.GetField("RefusalSlot",flags);var seqField=catalog.GetField("RefusalSequence",flags);
Need(slotField!=null && seqField!=null && slotField.DeclaringType==catalog && seqField.DeclaringType==catalog && slotField.FieldType==typeof(object[]) && seqField.FieldType==typeof(long) && slotField.IsPrivate && seqField.IsPrivate && slotField.IsDefined(typeof(ThreadStaticAttribute),false) && seqField.IsDefined(typeof(ThreadStaticAttribute),false),"Exact source-owned test refusal metadata");
long prior=(long)seqField.GetValue(null);
object result=support.Invoke(null,new object[]{value});
Need(result is bool && !(bool)result,"Intended unsafe support refusal");
object[] slot=(object[])slotField.GetValue(null);
Need(slot.Length==9 && (int)slot[0]==1 && (int)slot[1]==Environment.CurrentManagedThreadId && (long)slot[2]==prior+1 && (long)seqField.GetValue(null)==prior+1 && (int)slot[5]==reason,"Fresh generation intended refusal reason");
}
public static object[] Check(Type catalog, MethodInfo support)
{
Need(catalog!=typeof(S28PurityFixture_a91caea129ac11a18a774ad8aa744778be869cd53380ef1060900857c623e1e4) && catalog.Assembly!=typeof(S28PurityFixture_a91caea129ac11a18a774ad8aa744778be869cd53380ef1060900857c623e1e4).Assembly && support.DeclaringType==catalog && support.IsStatic && support.IsPublic && support.ReturnType==typeof(bool) && support.GetParameters().Length==1 && support.GetParameters()[0].ParameterType==typeof(object),"Separate actual source-owned catalog and fixture");
var flags=BindingFlags.NonPublic|BindingFlags.Static|BindingFlags.DeclaredOnly;
var map=catalog.GetMethod("MapStorage",flags,null,new Type[]{typeof(object)},null);
Need(map!=null && map.DeclaringType==catalog && map.ReturnType==typeof(bool),"Exact private map test target");
var comparer=new S28PurityFixture_a91caea129ac11a18a774ad8aa744778be869cd53380ef1060900857c623e1e4();
var hash=new Hashtable((IEqualityComparer)comparer);var ordered=new OrderedDictionary((IEqualityComparer)comparer);
var hf=Field(typeof(Hashtable),"_keycomparer",typeof(IEqualityComparer));var of=Field(typeof(OrderedDictionary),"_comparer",typeof(IEqualityComparer));
var oa=Field(typeof(OrderedDictionary),"_objectsArray",typeof(ArrayList));var ot=Field(typeof(OrderedDictionary),"_objectsTable",typeof(Hashtable));
object oaBefore=oa.GetValue(ordered),otBefore=ot.GetValue(ordered);
Need(Object.ReferenceEquals(hf.GetValue(hash),comparer) && Object.ReferenceEquals(of.GetValue(ordered),comparer) && hash.GetType()==typeof(Hashtable) && ordered.GetType()==typeof(OrderedDictionary) && EqualCalls==0 && HashCalls==0,"Actual empty custom comparer storage");
Need(!(bool)map.Invoke(null,new object[]{hash}) && !(bool)map.Invoke(null,new object[]{ordered}),"Comparer rejected before any custom dispatch");
Refuse(catalog,support,hash,7);Refuse(catalog,support,ordered,7);
Need(EqualCalls==0 && HashCalls==0 && Object.ReferenceEquals(hf.GetValue(hash),comparer) && Object.ReferenceEquals(of.GetValue(ordered),comparer) && Object.ReferenceEquals(oa.GetValue(ordered),oaBefore) && Object.ReferenceEquals(ot.GetValue(ordered),otBefore),"Zero comparer calls and unchanged cold storage");
Type ps=typeof(PSObject),setType=ps.GetNestedType("AdapterSet",BindingFlags.NonPublic);
Need(setType!=null && setType.Assembly==ps.Assembly,"Exact engine AdapterSet test type");
var factory=ps.GetMethod("CreateThirdPartyAdapterSet",flags,null,new Type[]{typeof(Type),typeof(PSPropertyAdapter)},null);
Need(factory!=null && factory.DeclaringType==ps && factory.ReturnType==setType && !factory.IsPublic && factory.IsStatic,"Exact internal test-only adapter factory");
var adapter=new S28PurityFixture_a91caea129ac11a18a774ad8aa744778be869cd53380ef1060900857c623e1e4();object set=factory.Invoke(null,new object[]{typeof(Hashtable),adapter});
Need(set!=null && set.GetType()==setType,"Actual constructed adapter set");
Type engineAdapter=ps.Assembly.GetType("System.Management.Automation.Adapter",true,false),third=ps.Assembly.GetType("System.Management.Automation.ThirdPartyAdapter",true,false);
var originalField=Field(setType,"<OriginalAdapter>k__BackingField",engineAdapter);
object originalAdapter=originalField.GetValue(set);Need(originalAdapter!=null && originalAdapter.GetType()==third,"Actual third-party adapter linkage");
var external=third.GetField("_externalAdapter",BindingFlags.NonPublic|BindingFlags.Instance|BindingFlags.DeclaredOnly);
Need(external!=null && external.DeclaringType==third && external.IsPrivate && external.IsInitOnly && external.FieldType==typeof(PSPropertyAdapter) && Object.ReferenceEquals(external.GetValue(originalAdapter),adapter),"Actual callback adapter identity");
var adapterField=Field(ps,"_adapterSet",setType);var input=new PSObject((object)1L);
adapterField.SetValue(input,set);Need(Object.ReferenceEquals(adapterField.GetValue(input),set),"Actual fresh adapter link");
Refuse(catalog,support,input,4);
Need(Object.ReferenceEquals(adapterField.GetValue(input),set),"Unchanged adapter link");
for(int i=0;i<AdapterCalls.Length;i++)Need(AdapterCalls[i]==0,"Zero actual adapter callbacks");
var currentMethod=catalog.GetMethod("CurrentTable",flags,null,Type.EmptyTypes,null);
Need(currentMethod!=null && currentMethod.DeclaringType==catalog && currentMethod.ReturnType==typeof(object),"Exact existing current-table read target");
object current=currentMethod.Invoke(null,Array.Empty<object>());
var foreign=new TypeTable(Array.Empty<string>());Need(!Object.ReferenceEquals(foreign,current),"Actual foreign identity");
var weak=new WeakReference<TypeTable>(foreign);TypeTable target;
Need(weak.TryGetTarget(out target) && Object.ReferenceEquals(target,foreign),"Actual live captured target before");
var weakField=Field(ps,"_typeTable",typeof(WeakReference<TypeTable>));var captured=new PSObject((object)1L);
weakField.SetValue(captured,weak);Need(Object.ReferenceEquals(weakField.GetValue(captured),weak),"Exact fresh weak capture link");
Refuse(catalog,support,captured,3);
Need(Object.ReferenceEquals(weakField.GetValue(captured),weak) && weak.TryGetTarget(out target) && Object.ReferenceEquals(target,foreign),"Unchanged live capture after");
GC.KeepAlive(foreign);
var link=Field(ps,"_immediateBaseObject",typeof(object));var first=new PSObject((object)1L);var second=new PSObject((object)1L);
link.SetValue(first,second);link.SetValue(second,first);
Need(Object.ReferenceEquals(link.GetValue(first),second) && Object.ReferenceEquals(link.GetValue(second),first),"Actual two-node cycle before");
Refuse(catalog,support,first,10);
Need(Object.ReferenceEquals(link.GetValue(first),second) && Object.ReferenceEquals(link.GetValue(second),first),"Unchanged cycle after");
object terminal=(object)1L;var layers=new PSObject[9];
for(int i=0;i<layers.Length;i++)layers[i]=new PSObject(terminal);
for(int i=0;i<layers.Length;i++)link.SetValue(layers[i],i+1<layers.Length?(object)layers[i+1]:terminal);
for(int i=0;i<layers.Length;i++){Need(Object.ReferenceEquals(link.GetValue(layers[i]),i+1<layers.Length?(object)layers[i+1]:terminal),"Actual nine-layer link before");for(int j=0;j<i;j++)Need(!Object.ReferenceEquals(layers[i],layers[j]),"Distinct test wrappers");}
Refuse(catalog,support,layers[0],9);
for(int i=0;i<layers.Length;i++)Need(Object.ReferenceEquals(link.GetValue(layers[i]),i+1<layers.Length?(object)layers[i+1]:terminal),"Unchanged nine-layer links after");
var positive=new PSObject(terminal);Need((bool)support.Invoke(null,new object[]{positive}) && Object.ReferenceEquals(link.GetValue(positive),terminal),"Actual acyclic transparent positive");
return new object[]{"custom_comparer0","adapter9_callbacks0","live_foreign_capture_reason3","cycle_reason10","actual_depth9_reason9","acyclic_positive"};
}
public static void EmptyTemplate(Type catalog)
{
var flags=BindingFlags.NonPublic|BindingFlags.Static|BindingFlags.DeclaredOnly;
var keys=catalog.GetMethod("Keys",flags,null,new Type[]{typeof(object),typeof(string[])},null);
var empty=catalog.GetMethod("EmptyPin",flags,null,new Type[]{typeof(IDictionary),typeof(IDictionary)},null);
Need(keys!=null && empty!=null && keys.DeclaringType==catalog && empty.DeclaringType==catalog,"Exact actual raw0 verifier branches");
var template=new Hashtable();template.Add("path","");template.Add("bytes",0L);template.Add("sha256","e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
var pin=new Hashtable();pin.Add("path","verification/banc-natural-gait-m3-20260919/parent-direct-science-20261009-contact-s27-bootstrap/stderr.bin");pin.Add("bytes",0L);pin.Add("sha256","e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
var keyField=Field(typeof(Hashtable),"_keys",typeof(ICollection));var valueField=Field(typeof(Hashtable),"_values",typeof(ICollection));
object beforeKeys=keyField.GetValue(template),beforeValues=valueField.GetValue(template);Need(beforeKeys==null && beforeValues==null,"Fresh raw0 template no cached views");
keys.Invoke(null,new object[]{template,new string[]{"path","bytes","sha256"}});empty.Invoke(null,new object[]{pin,template});
Need(Object.ReferenceEquals(keyField.GetValue(template),beforeKeys) && Object.ReferenceEquals(valueField.GetValue(template),beforeValues),"Actual raw0 verification branch does not initialize collection views");
}
}
'@
$fixtureName='S28PurityFixture_a91caea129ac11a18a774ad8aa744778be869cd53380ef1060900857c623e1e4';$fixtureHash='56e5e6f1d56d80da5a4b5ecbebaf83fbcb8f883fbab20d2c6c5c99de09981fd3'
Need (([Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($utf8.GetBytes($fixtureSource)))).ToLowerInvariant() -ceq $fixtureHash) 'Whole fixed test helper source'
$catalogTypeBefore=$script:KCatalogCache.type;$catalogAssemblyBefore=$script:KCatalogCache.assembly;$catalogOriginBefore=$script:KCatalogOrigin
if($null -eq $script:S28FixtureCache){
 $fixtureTypes=@(Add-Type -TypeDefinition $fixtureSource -PassThru -ErrorAction Stop -WarningAction Stop)
 Need ($fixtureTypes.Length -eq 1 -and $fixtureTypes[0] -is [type] -and $fixtureTypes[0].FullName -ceq $fixtureName -and $fixtureTypes[0].Assembly -ne $catalogAssemblyBefore) 'Separate actual one returned fixture Type/Assembly'
 $fixtureType=$fixtureTypes[0];$fixtureCheck=$fixtureType.GetMethod('Check',[type[]]@([type],[Reflection.MethodInfo]));$fixtureEmpty=$fixtureType.GetMethod('EmptyTemplate',[type[]]@([type]))
 Need ($fixtureCheck.DeclaringType -eq $fixtureType -and $fixtureCheck.ReturnType -eq [object[]] -and $fixtureCheck.IsStatic -and $fixtureEmpty.DeclaringType -eq $fixtureType -and $fixtureEmpty.ReturnType -eq [void] -and $fixtureEmpty.IsStatic) 'Exact test helper methods'
 $script:S28FixtureCache=@{type=$fixtureType;assembly=$fixtureType.Assembly;check=$fixtureCheck;empty=$fixtureEmpty;source=$fixtureSource;hash=$fixtureHash}
}
$fixture=$script:S28FixtureCache
# D5 non-overridable predicate profiles; no production MethodInfo injection.
$profileInterface=$fixture.type.GetInterfaceMap([Collections.IEqualityComparer])
Need ($profileInterface.TargetMethods.Length -eq 2) 'Existing explicit-interface fixture profile'
$profileFinal=$profileInterface.TargetMethods[0]
$profilePlain=[object].GetMethod('GetType',[type[]]@())
$profileOpen=[object].GetMethod('ToString',[type[]]@())
Need ($profilePlain.DeclaringType -eq [object] -and -not $profilePlain.IsVirtual -and -not $profilePlain.IsStatic) 'Existing BCL nonvirtual profile'
Need ($profileFinal.DeclaringType -eq $fixture.type -and $profileFinal.IsVirtual -and $profileFinal.IsFinal -and -not $profileFinal.IsStatic) 'Existing explicit-interface virtual-final profile'
Need ($profileOpen.DeclaringType -eq [object] -and $profileOpen.IsVirtual -and -not $profileOpen.IsFinal -and -not $profileOpen.IsStatic) 'Existing base virtual-nonfinal profile'
foreach($profile in @(@($profilePlain,$true),@($profileFinal,$true),@($profileOpen,$false))){
 $methodProfile=$profile[0];$eligible=(-not ($methodProfile.IsVirtual -and -not $methodProfile.IsFinal))
 Need ($eligible -eq $profile[1]) 'Narrow non-overridable metadata eligibility predicate'
}
Need ($fixture.type.GetField('EqualCalls').GetValue($null) -eq 0 -and $fixture.type.GetField('HashCalls').GetValue($null) -eq 0) 'Metadata profiles execute zero callbacks'
Need ($fixture.type -is [type] -and $fixture.type.Assembly -eq $fixture.assembly -and $fixture.assembly -ne $catalogAssemblyBefore -and $fixture.type.FullName -ceq $fixtureName -and [string]::Equals($fixture.source,$fixtureSource,[StringComparison]::Ordinal) -and $fixture.hash -ceq $fixtureHash -and $fixture.check.DeclaringType -eq $fixture.type -and $fixture.empty.DeclaringType -eq $fixture.type) 'Whole retained actual fixture identity, no global name lookup'
"""+historical_fixture.d19_clr_fixture_calls()+historical_fixture.d19_r32_local_contract()+"""
# D13 genuine reserved graph controls in the existing admitted process, SOURCE_AUTHORED_NOT_RUN.
# No genuine cache/flag is cleared. Every mutation below targets a fresh owned graph and is restored.
$d13All=[Reflection.BindingFlags]'DeclaredOnly,Public,NonPublic,Instance,Static';$d13Inst=[Reflection.BindingFlags]'DeclaredOnly,Public,NonPublic,Instance'
$d13Flags=[PSObject].GetField('_flags',$d7Instance);$d13Base=[PSObject].GetField('_immediateBaseObject',$d7Instance)
$d13OwnerMembers=[PSObject].GetField('_instanceMembers',$d7Instance);$d13CacheField=[PSObject].GetField('_typeNames',$d7Instance);$d13AdapterField=[PSObject].GetField('_adapterSet',$d7Instance)
$d13Read=[Reflection.FieldInfo].GetMethod('GetValue',[type[]]@([object]));$d13Write=[Reflection.FieldInfo].GetMethod('SetValue',[type[]]@([object],[object]))
$d13Observe=$script:KCatalogCache.type.GetMethod('ReservedObservation',[Reflection.BindingFlags]'NonPublic,Static')
$d13Kinds=@([long],[int],[bool],[double])
$d13Modes=@('genuine','faithful-code-equivalent','flags','member-name','hidden-flag','reserved-flag','instance-flag','serialize-flag','wrong-owner','equal-distinct-key','getter','setter','proxy-owner','proxy-collections','internal-cache','inherit','cache-key','cache-hierarchy','cache-direct-list','cache-link','adapter-link','table-count','table-writer','table-comparer','table-version','table-too-large','hash','probe-stop','extra-entry','table-list-divergence','list-version','cache-extra-names','cache-unused','cache-version','cache-array-bound','adapter-backing')
foreach($d13Kind in $d13Kinds){foreach($d13Wrapped in @($false,$true)){foreach($d13Mode in $d13Modes){
 $d13Key=[object[]]::new(1);$d13Key[0]=if($d13Kind -eq [long]){[long](19000000+[array]::IndexOf($d13Modes,$d13Mode))}elseif($d13Kind -eq [int]){[int](190000+[array]::IndexOf($d13Modes,$d13Mode))}elseif($d13Kind -eq [bool]){[Runtime.Serialization.FormatterServices]::GetUninitializedObject([bool])}else{[double](190000.25+[array]::IndexOf($d13Modes,$d13Mode))}
 $d13Lookup=[object[]]::new(2);$d13Lookup[0]=$d13Key[0];$d13NameLookup=[object[]]::new(2);$d13NameLookup[0]=$d13Key[0]
 $d13OwnerCtor=[object[]]::new(1);$d13OwnerCtor[0]=$d13Key[0];$d13Owner=[object[]]::new(1);$d13Owner[0]=[PSObject].GetConstructor([type[]]@([object])).Invoke($d13OwnerCtor)
 $d13HTBefore=$null;$d13Undo=[Collections.Generic.List[object[]]]::new();$d13BucketUndo=[Collections.Generic.List[object[]]]::new();$d13Owned=$false;$d13OldSlot=$d7Slot.GetValue($null)
 try{
  Need (-not $d7Lookup.Invoke($d7Table,$d13Lookup) -and -not $d7NameLookup.Invoke($d7Names,$d13NameLookup)) 'Fresh SAME scalar box has no CWT association'
  $d13OwnerDispatch=[object[]]::new(2);$d13OwnerDispatch[0]=$d13Owner[0]
  $d13Proxy=$d7Dispatch.Invoke([PSObject].GetProperty('Members').GetGetMethod(),$d13OwnerDispatch)
  $d13Generate=$d13Proxy.GetType().GetMethod('GenerateAllReservedMembers',[Reflection.BindingFlags]'DeclaredOnly,NonPublic,Instance')
  Need ($null -ne $d13Generate -and $d13Generate.GetParameters().Length -eq 0 -and $d13Generate.ReturnType -eq [void]) 'Actual genuine GenerateAllReservedMembers setup, not five-generator surrogate'
  $d13GenerateDispatch=[object[]]::new(2);$d13GenerateDispatch[0]=$d13Proxy
  $d13Owned=$true;$null=$d7Dispatch.Invoke($d13Generate,$d13GenerateDispatch)
  Need ($d7Lookup.Invoke($d7Table,$d13Lookup)) 'Genuine generator original-key CWT association'
  $d13Collection=$d13Lookup[1];$d13Store=$d7Storage.GetValue($d13Collection);$d13List=$d8DFields[0].GetValue($d13Store);$d13Table=$d8DFields[1].GetValue($d13Store)
  $d13HTFields=@('_buckets','_count','_version','_isWriterInProgress','_keycomparer') | ForEach-Object {[Collections.Hashtable].GetField($_,$d13Inst)}
  $d13HTBefore=[object[]]::new(5);for($d13I=0;$d13I -lt 5;$d13I++){$d13HTBefore[$d13I]=$d13HTFields[$d13I].GetValue($d13Table)}
  $d13HTAllFields=[Collections.Hashtable].GetFields($d13Inst);$d13HTAllBefore=[object[]]::new($d13HTAllFields.Length);for($d13I=0;$d13I -lt $d13HTAllFields.Length;$d13I++){$d13HTAllBefore[$d13I]=$d13HTAllFields[$d13I].GetValue($d13Table)}
  $d13OriginalBuckets=$d13HTBefore[0];$d13OriginalRecords=[object[]]::new($d13OriginalBuckets.Length);for($d13I=0;$d13I -lt $d13OriginalBuckets.Length;$d13I++){$d13OriginalRecords[$d13I]=$d13OriginalBuckets.GetValue($d13I)}
  $d13Items=$d8LFields[0].GetValue($d13List);$d13Keys=[object[]]::new(5);$d13Values=[object[]]::new(5);$d13Entries=[object[]]::new(5)
  $d13EF=@([Collections.DictionaryEntry].GetField('_key',$d8DF),[Collections.DictionaryEntry].GetField('_value',$d8DF));$d13Index=@{}
  for($d13I=0;$d13I -lt 5;$d13I++){$d13Entries[$d13I]=$d13Items[$d13I];$d13Keys[$d13I]=$d13EF[0].GetValue($d13Entries[$d13I]);$d13Values[$d13I]=$d13EF[1].GetValue($d13Entries[$d13I]);$d13Index[$d13Keys[$d13I]]=$d13I}
  Need ($d13Index.GetType() -eq [Collections.Hashtable] -and $d13Index.Count -eq 5) 'R30 owned exact index'
  $d13Seen=@{};foreach($d13Name in @('psbase','psadapted','psextended','psobject','pstypenames')){
   Need ($d13Index.ContainsKey($d13Name)) 'R30 literal key present';$d13Position=$d13Index[$d13Name]
   Need ($d13Position -is [int] -and $d13Position.GetType() -eq [int] -and $d13Position -ge 0 -and $d13Position -lt 5) 'R30 actual Int32 before array selection'
   Need (-not $d13Seen.ContainsKey($d13Position) -and $d13Keys[$d13Position] -is [string] -and [StringComparer]::Ordinal.Equals($d13Keys[$d13Position],$d13Name)) 'R30 distinct original key position';$d13Seen[$d13Position]=$true
  }
  $d13Code=$d13Values[$d13Index['pstypenames']];$d13Extended=$d13Values[$d13Index['psextended']];$d13Internal=$d13Values[$d13Index['psbase']]
  $d13MFields=[System.Management.Automation.PSMemberInfo].GetFields($d13Inst);$d13FlagBefore=$d13Read.Invoke($d13Flags,$d13Owner)
  Need ([Convert]::ToByte($d13FlagBefore) -eq 2 -and [object]::ReferenceEquals($d13Read.Invoke($d13Base,$d13Owner),$d13Key[0]) -and [object]::ReferenceEquals($d13Read.Invoke($d13OwnerMembers,$d13Owner),$d13Collection)) 'Generated flag and SAME key/owner/collection before source'
  $d13Cache=$d13Read.Invoke($d13CacheField,$d13Owner);$d13Adapter=$d13Read.Invoke($d13AdapterField,$d13Owner)
  $d13First=[object[]]@($d13Key[0],$d13Collection,$d13Table,$d13Keys,$d13Values,$null,$null)
  Need ($d13Observe.Invoke($null,$d13First)) 'Genuine unmodified graph reaches complete guard before mutation, not a preempted negative'
  $d13Input=[object[]]::new(2);$d13Input[0]=$d13Key[0]
  if($d13Wrapped){$d13WrapperCtor=[object[]]::new(1);$d13WrapperCtor[0]=$d13Key[0];$d13Input[0]=[PSObject].GetConstructor([type[]]@([object])).Invoke($d13WrapperCtor);$d13Wrapper=[object[]]::new(1);$d13Wrapper[0]=$d13Input[0];foreach($d13F in @('_instanceMembers','_typeNames','_adapterSet')){Need ($null -eq $d13Read.Invoke([PSObject].GetField($d13F,$d7Instance),$d13Wrapper)) 'Actually clean wrapper retains original scalar route'}}
  $d13Reset=[object[]]@(1,[Threading.Thread]::CurrentThread.ManagedThreadId,$d7Seq.GetValue($null),0,0,0,-1,0,'NONE');$d7Slot.SetValue($null,$d13Reset)
  Need ($viewMethod.Invoke($null,$d13Input) -and [object]::ReferenceEquals($d13Input[1],$d13Key[0])) 'Actual genuine raw/clean-wrapper positive SAME output box'
  $d13Second=[object[]]@($d13Key[0],$d13Collection,$d13Table,$d13Keys,$d13Values,$null,$null);Need ($d13Observe.Invoke($null,$d13Second)) 'Same complete observed graph after production'
  Need ($d13First[5].Length -eq $d13Second[5].Length -and $d13First[6].Length -eq $d13Second[6].Length) 'Complete finite snapshots length'
  for($d13I=0;$d13I -lt $d13First[5].Length;$d13I++){Need ([object]::ReferenceEquals($d13First[5][$d13I],$d13Second[5][$d13I])) 'All observed graph/cache/proxy/table reference identities unchanged'}
  for($d13I=0;$d13I -lt $d13First[6].Length;$d13I++){Need ($d13First[6][$d13I] -ceq $d13Second[6][$d13I]) 'All observed flag/cache/list/hash/storage scalars unchanged'}
  if($d13Mode.StartsWith('cache-')){
   # Negative cache setup uses an owned full equivalent, retaining the genuine shared cache intact.
   $d13CC=$d7NameType.GetConstructor([Reflection.BindingFlags]'Public,NonPublic,Instance',$null,[type[]]@([Collections.Generic.IEnumerable[string]],[bool]),$null)
   Need ($null -ne $d13CC) 'Existing interned constructor for separately labelled owned faithful default cache'
   $d13CA=[object[]]::new(2);$d13CA[0]=[string[]]@($d13Kind.FullName,'System.ValueType','System.Object');$d13CA[1]=$true;$d13OwnedCache=$d13CC.Invoke($d13CA)
   $d13PriorCache=[object[]]::new(1);$d13PriorCache[0]=$d13Cache;$d13Undo.Add([object[]]@($d13CacheField,$d13Owner,$d13PriorCache))
   $d13CacheSet=[object[]]::new(2);$d13CacheSet[0]=$d13Owner[0];$d13CacheSet[1]=$d13OwnedCache;$null=$d13Write.Invoke($d13CacheField,$d13CacheSet);$d13Cache=$d13OwnedCache
   $d13EquivalentObservation=[object[]]@($d13Key[0],$d13Collection,$d13Table,$d13Keys,$d13Values,$null,$null);Need ($d13Observe.Invoke($null,$d13EquivalentObservation)) 'Separately labelled complete reconstructed default cache before its reachable negative'
  }
  $d13Target=$null;$d13Field=$null;$d13New=$null;$d13FieldArgs=$null
  switch -CaseSensitive($d13Mode){
   'flags' {$d13Field=$d13Flags;$d13FieldArgs=$d13Owner;$d13New=[Enum]::ToObject($d13Flags.FieldType,[byte]3)}
   'member-name' {$d13Target=$d13Extended;$d13Field=[System.Management.Automation.PSMemberInfo].GetField('name',$d13Inst);$d13New='foreign-name'}
   'hidden-flag' {$d13Target=$d13Extended;$d13Field=[System.Management.Automation.PSMemberInfo].GetField('<IsHidden>k__BackingField',$d13Inst);$d13New=$false}
   'reserved-flag' {$d13Target=$d13Extended;$d13Field=[System.Management.Automation.PSMemberInfo].GetField('<IsReservedMember>k__BackingField',$d13Inst);$d13New=$false}
   'instance-flag' {$d13Target=$d13Extended;$d13Field=[System.Management.Automation.PSMemberInfo].GetField('<IsInstance>k__BackingField',$d13Inst);$d13New=$false}
   'serialize-flag' {$d13Target=$d13Extended;$d13Field=[System.Management.Automation.PSMemberInfo].GetField('<ShouldSerialize>k__BackingField',$d13Inst);$d13New=$true}
   'wrong-owner' {$d13Target=$d13Code;$d13Field=[System.Management.Automation.PSMemberInfo].GetField('instance',$d13Inst);$d13Other=[object[]]::new(1);$d13Other[0]=[object]::new();$d13New=[PSObject].GetConstructor([type[]]@([object])).Invoke($d13Other)}
   'equal-distinct-key' {$d13Field=$d13Base;$d13FieldArgs=$d13Owner;$d13New=if($d13Kind -eq [long]){[long]19000009}elseif($d13Kind -eq [int]){[int]190009}elseif($d13Kind -eq [bool]){[Runtime.Serialization.FormatterServices]::GetUninitializedObject([bool])}else{[double]190009.25};Need (-not [object]::ReferenceEquals($d13New,$d13Key[0]) -and $d13New -ceq $d13Key[0]) 'Equal-valued distinct scalar box negative, not key replacement'}
   'getter' {$d13Target=$d13Code;$d13Field=[System.Management.Automation.PSCodeProperty].GetField('<GetterCodeReference>k__BackingField',$d13Inst);$d13New=[Environment].GetProperty('TickCount').GetGetMethod()}
   'setter' {$d13Target=$d13Code;$d13Field=[System.Management.Automation.PSCodeProperty].GetField('<SetterCodeReference>k__BackingField',$d13Inst);$d13New=[Environment].GetProperty('TickCount').GetGetMethod()}
   'proxy-owner' {$d13Target=[System.Management.Automation.PSMemberSet].GetField('_members',$d13Inst).GetValue($d13Extended);$d13Field=$d13Target.GetType().GetField('_memberSetOwner',$d13Inst);$d13New=$d13Internal}
   'proxy-collections' {$d13Target=[System.Management.Automation.PSMemberSet].GetField('_members',$d13Inst).GetValue($d13Extended);$d13Field=$d13Target.GetType().GetField('<Collections>k__BackingField',$d13Inst);$d13New=[System.Management.Automation.PSMemberSet].GetField('s_emptyMemberCollection',$d13All).GetValue($null)}
   'internal-cache' {$d13Target=$d13Internal;$d13Field=[System.Management.Automation.PSMemberSet].GetField('internalMembers',$d13Inst);$d13New=$d13Collection}
   'inherit' {$d13Target=$d13Internal;$d13Field=[System.Management.Automation.PSMemberSet].GetField('inheritMembers',$d13Inst);$d13New=$false}
   'cache-key' {$d13Target=$d13Cache;$d13Field=$d7NameType.GetField('<Key>k__BackingField',$d13Inst);$d13New='System.Object'}
   'cache-hierarchy' {$d13NameItems=[Collections.ObjectModel.Collection[string]].GetField('items',$d13Inst).GetValue($d13Cache);$d13NameList=[Collections.ObjectModel.ReadOnlyCollection[string]].GetField('list',$d13Inst).GetValue($d13NameItems);$d13Target=$d13NameList;$d13Field=[Collections.Generic.List[string]].GetField('_items',$d13Inst);$d13New=[string[]]@($d13Kind.FullName,'System.String','System.Object')}
   'cache-direct-list' {$d13Target=$d13Cache;$d13Field=[Collections.ObjectModel.Collection[string]].GetField('items',$d13Inst);$d13New=[Collections.Generic.List[string]]::new()}
   'cache-link' {$d13NameItems=[Collections.ObjectModel.Collection[string]].GetField('items',$d13Inst).GetValue($d13Cache);$d13Target=$d13NameItems;$d13Field=[Collections.ObjectModel.ReadOnlyCollection[string]].GetField('list',$d13Inst);$d13New=[Collections.Generic.List[string]]::new()}
   'adapter-link' {$d13Field=$d13AdapterField;$d13FieldArgs=$d13Owner;$d13New=[Activator]::CreateInstance($d13AdapterField.FieldType,[Reflection.BindingFlags]'Public,NonPublic,Instance',$null,[object[]]@($null,$null),$null)}
   'adapter-backing' {$d13Target=$d13Adapter;$d13Field=$d13AdapterField.FieldType.GetField('<OriginalAdapter>k__BackingField',$d13Inst);$d13New=$null}
   'cache-extra-names' {$d13NameItems=[Collections.ObjectModel.Collection[string]].GetField('items',$d13Inst).GetValue($d13Cache);$d13NameList=[Collections.ObjectModel.ReadOnlyCollection[string]].GetField('list',$d13Inst).GetValue($d13NameItems);$d13Target=$d13NameList;$d13Field=[Collections.Generic.List[string]].GetField('_size',$d13Inst);$d13New=[int]4}
   'cache-unused' {$d13NameItems=[Collections.ObjectModel.Collection[string]].GetField('items',$d13Inst).GetValue($d13Cache);$d13NameList=[Collections.ObjectModel.ReadOnlyCollection[string]].GetField('list',$d13Inst).GetValue($d13NameItems);$d13Target=$d13NameList;$d13Field=[Collections.Generic.List[string]].GetField('_items',$d13Inst);$d13New=[string[]]@($d13Kind.FullName,'System.ValueType','System.Object','foreign-unused')}
   'cache-version' {$d13NameItems=[Collections.ObjectModel.Collection[string]].GetField('items',$d13Inst).GetValue($d13Cache);$d13Target=[Collections.ObjectModel.ReadOnlyCollection[string]].GetField('list',$d13Inst).GetValue($d13NameItems);$d13Field=[Collections.Generic.List[string]].GetField('_version',$d13Inst);$d13New=[int]33}
   'cache-array-bound' {$d13NameItems=[Collections.ObjectModel.Collection[string]].GetField('items',$d13Inst).GetValue($d13Cache);$d13Target=[Collections.ObjectModel.ReadOnlyCollection[string]].GetField('list',$d13Inst).GetValue($d13NameItems);$d13Field=[Collections.Generic.List[string]].GetField('_items',$d13Inst);$d13New=[string[]]::new(17)}
   'table-count' {$d13Target=$d13Table;$d13Field=[Collections.Hashtable].GetField('_count',$d13Inst);$d13New=[int]4}
   'table-writer' {$d13Target=$d13Table;$d13Field=[Collections.Hashtable].GetField('_isWriterInProgress',$d13Inst);$d13New=$true}
   'table-comparer' {$d13Target=$d13Table;$d13Field=[Collections.Hashtable].GetField('_keycomparer',$d13Inst);$d13New=[Activator]::CreateInstance($fixture.type)}
   'table-version' {$d13Target=$d13Table;$d13Field=[Collections.Hashtable].GetField('_version',$d13Inst);$d13New=[int]33}
   'table-too-large' {$d13Target=$d13Table;$d13Field=[Collections.Hashtable].GetField('_buckets',$d13Inst);$d13OldBuckets=$d13Field.GetValue($d13Table);$d13New=[Array]::CreateInstance($d13OldBuckets.GetType().GetElementType(),65)}
   'list-version' {$d13Target=$d13List;$d13Field=[Collections.ArrayList].GetField('_version',$d13Inst);$d13New=[int]33}
   'table-list-divergence' {$d13FieldArgs=[object[]]::new(1);$d13FieldArgs[0]=$d13Items[$d13Index['pstypenames']];$d13Field=$d13EF[1];$d13New=[System.Management.Automation.PSCodeProperty]::new('pstypenames',$script:R26Getter)}
   'extra-entry' {$d13Table.Add('foreign-alias',[System.Management.Automation.PSNoteProperty]::new('foreign-alias',0));$d13Undo.Add([object[]]@($null,$d13Table,'foreign-alias'))}
   'hash' {$d13Buckets=[Collections.Hashtable].GetField('_buckets',$d13Inst).GetValue($d13Table);$d13BF=$d13Buckets.GetType().GetElementType().GetField('hash_coll',$d13All);$d13KF=$d13Buckets.GetType().GetElementType().GetField('key',$d13All);for($d13I=0;$d13I -lt $d13Buckets.Length;$d13I++){$d13Record=$d13Buckets.GetValue($d13I);if([object]::ReferenceEquals($d13KF.GetValue($d13Record),$d13Keys[0])){$d13BucketUndo.Add([object[]]@($d13Buckets,$d13I,$d13Buckets.GetValue($d13I)));$d13BF.SetValue($d13Record,([int]$d13BF.GetValue($d13Record) -bxor 1));$d13Buckets.SetValue($d13Record,$d13I);break}}}
   'probe-stop' {
    $d13Buckets=[Collections.Hashtable].GetField('_buckets',$d13Inst).GetValue($d13Table);$d13BT=$d13Buckets.GetType().GetElementType();$d13BF=$d13BT.GetField('hash_coll',$d13All);$d13KF=$d13BT.GetField('key',$d13All)
    $d13Hash=([long][StringComparer]::OrdinalIgnoreCase.GetHashCode([string]$d13Keys[0])) -band 2147483647;$d13Start=[int]($d13Hash % $d13Buckets.Length);$d13AtStart=$d13Buckets.GetValue($d13Start)
    $d13BucketUndo.Add([object[]]@($d13Buckets,$d13Start,$d13Buckets.GetValue($d13Start)))
    if([object]::ReferenceEquals($d13KF.GetValue($d13AtStart),$d13Keys[0])){
     for($d13I=0;$d13I -lt $d13Buckets.Length;$d13I++){$d13OtherRecord=$d13Buckets.GetValue($d13I);$d13OtherKey=$d13KF.GetValue($d13OtherRecord);if($null -ne $d13OtherKey -and -not [object]::ReferenceEquals($d13OtherKey,$d13Keys[0])){$d13BucketUndo.Add([object[]]@($d13Buckets,$d13I,$d13Buckets.GetValue($d13I)));$d13Buckets.SetValue($d13AtStart,$d13I);$d13AtStart=$d13OtherRecord;break}}
    }
    $d13BF.SetValue($d13AtStart,([int]$d13BF.GetValue($d13AtStart) -band 2147483647));$d13Buckets.SetValue($d13AtStart,$d13Start)
   }
   'faithful-code-equivalent' {
    $d13Getter=[System.Management.Automation.PSCodeProperty].GetField('<GetterCodeReference>k__BackingField',$d13Inst).GetValue($d13Code);$d13Equivalent=[System.Management.Automation.PSCodeProperty]::new('pstypenames',$d13Getter)
    foreach($d13F in $d13MFields){$d13CopyArgs=[object[]]::new(2);$d13CopyArgs[0]=$d13Equivalent;$d13CopyArgs[1]=if($d13F.Name -ceq 'instance'){$d13Owner[0]}else{$d13F.GetValue($d13Code)};$null=$d13Write.Invoke($d13F,$d13CopyArgs)}
    $d13FieldArgs=[object[]]::new(1);$d13FieldArgs[0]=$d13Items[$d13Index['pstypenames']];$d13Field=$d13EF[1];$d13New=$d13Equivalent
    $d13Table[$d13Keys[$d13Index['pstypenames']]]=$d13Equivalent;$d13Undo.Add([object[]]@($null,$d13Table,$d13Keys[$d13Index['pstypenames']],$d13Code))
   }
  }
  if($null -ne $d13Field){
   if($null -eq $d13FieldArgs){$d13FieldArgs=[object[]]::new(1);$d13FieldArgs[0]=$d13Target}
   $d13Old=[object[]]::new(1);$d13Old[0]=$d13Read.Invoke($d13Field,$d13FieldArgs);$d13Undo.Add([object[]]@($d13Field,$d13FieldArgs,$d13Old))
   $d13SetArgs=[object[]]::new(2);$d13SetArgs[0]=$d13FieldArgs[0];$d13SetArgs[1]=$d13New;R31Check 'before';$null=$d13Write.Invoke($d13Field,$d13SetArgs);R31Check 'after'
  }
  # Snapshot the deliberately changed raw field and complete table before the production denial/positive.
  $d13MutationBefore=[object[]]::new(1);if($null -ne $d13Field){$d13MutationBefore[0]=$d13Read.Invoke($d13Field,$d13FieldArgs)}
  $d13HTCall=[object[]]::new(5);for($d13I=0;$d13I -lt 5;$d13I++){$d13HTCall[$d13I]=$d13HTFields[$d13I].GetValue($d13Table)}
  $d13HTAllCall=[object[]]::new($d13HTAllFields.Length);for($d13I=0;$d13I -lt $d13HTAllFields.Length;$d13I++){$d13HTAllCall[$d13I]=$d13HTAllFields[$d13I].GetValue($d13Table)}
  $d13CallBuckets=$d13HTCall[0];$d13CallBF=@('key','val','hash_coll') | ForEach-Object {$d13CallBuckets.GetType().GetElementType().GetField($_,$d13All)}
  $d13CallPairs=[Collections.Generic.List[object[]]]::new();for($d13I=0;$d13I -lt $d13CallBuckets.Length;$d13I++){$d13R=$d13CallBuckets.GetValue($d13I);$d13CallPairs.Add([object[]]@($d13CallBF[0].GetValue($d13R),$d13CallBF[1].GetValue($d13R),$d13CallBF[2].GetValue($d13R)))}
  $d13Input[1]=$null;$d13Reset=[object[]]@(1,[Threading.Thread]::CurrentThread.ManagedThreadId,$d7Seq.GetValue($null),0,0,0,-1,0,'NONE');$d7Slot.SetValue($null,$d13Reset)
  $d13Positive=$d13Mode -cin @('genuine','faithful-code-equivalent')
  $r31Return=$viewMethod.Invoke($null,$d13Input);R31View $r31Return
  Need (($d7Slot.GetValue($null))[5] -eq $(if($d13Positive){0}elseif($d13Mode -ceq 'list-version'){76}else{60})) 'Positive0, inherited list-prefix76, or reached new grouped guard60'
  if($null -ne $d13Field){$d13MutationAfter=[object[]]::new(1);$d13MutationAfter[0]=$d13Read.Invoke($d13Field,$d13FieldArgs);if($d13Field.FieldType.IsValueType){Need ($d13MutationBefore[0] -ceq $d13MutationAfter[0]) 'Trusted changed field scalar unchanged by source'}else{Need ([object]::ReferenceEquals($d13MutationBefore[0],$d13MutationAfter[0])) 'Original changed field reference unchanged by source'}}
  for($d13I=0;$d13I -lt 5;$d13I++){$d13After=$d13HTFields[$d13I].GetValue($d13Table);if($d13I -cin @(1,2,3)){Need ($d13After -ceq $d13HTCall[$d13I]) 'Complete table scalars unchanged by source'}else{Need ([object]::ReferenceEquals($d13After,$d13HTCall[$d13I])) 'Complete table reference links unchanged by source'}}
  for($d13I=0;$d13I -lt $d13HTAllFields.Length;$d13I++){$d13After=$d13HTAllFields[$d13I].GetValue($d13Table);if($d13HTAllFields[$d13I].FieldType.IsValueType){Need ($d13After -ceq $d13HTAllCall[$d13I]) 'Every Hashtable field scalar unchanged by source'}else{Need ([object]::ReferenceEquals($d13After,$d13HTAllCall[$d13I])) 'Every Hashtable field reference unchanged by source'}}
  for($d13I=0;$d13I -lt $d13CallBuckets.Length;$d13I++){$d13R=$d13CallBuckets.GetValue($d13I);Need ([object]::ReferenceEquals($d13CallBF[0].GetValue($d13R),$d13CallPairs[$d13I][0]) -and [object]::ReferenceEquals($d13CallBF[1].GetValue($d13R),$d13CallPairs[$d13I][1]) -and $d13CallBF[2].GetValue($d13R) -ceq $d13CallPairs[$d13I][2]) 'Every Bucket raw key/value/hash unchanged by source'}
  $d13Lookup[1]=$null;Need ($d7Lookup.Invoke($d7Table,$d13Lookup) -and [object]::ReferenceEquals($d13Lookup[1],$d13Collection) -and [object]::ReferenceEquals($d13Lookup[0],$d13Key[0])) 'Every control keeps same original key/value association'
  Need ($script:D7GetterCalls -eq 0 -and $script:R26Counter.GetValue($null) -eq 0 -and $fixture.type.GetField('EqualCalls').GetValue($null) -eq 0 -and $fixture.type.GetField('HashCalls').GetValue($null) -eq 0) 'Source dispatches zero supplied getter/comparer callbacks'
  foreach($d13N in $fixture.type.GetField('AdapterCalls').GetValue($null)){Need ($d13N -eq 0) 'Source dispatches zero supplied adapter callbacks'}
 }finally{
  try{
   for($d13I=$d13Undo.Count-1;$d13I -ge 0;$d13I--){$d13U=$d13Undo[$d13I];if($null -eq $d13U[0]){if($d13U.Length -eq 4){$d13U[1][$d13U[2]]=$d13U[3]}else{$d13U[1].Remove($d13U[2])}}else{$d13SetArgs=[object[]]::new(2);$d13SetArgs[0]=$d13U[1][0];$d13SetArgs[1]=$d13U[2][0];$null=$d13Write.Invoke($d13U[0],$d13SetArgs)}}
   for($d13I=$d13BucketUndo.Count-1;$d13I -ge 0;$d13I--){$d13U=$d13BucketUndo[$d13I];$d13U[0].SetValue($d13U[2],$d13U[1])}
   if($null -ne $d13HTBefore){for($d13I=0;$d13I -lt $d13HTAllFields.Length;$d13I++){$d13HTAllFields[$d13I].SetValue($d13Table,$d13HTAllBefore[$d13I])};for($d13I=0;$d13I -lt $d13OriginalBuckets.Length;$d13I++){$d13OriginalBuckets.SetValue($d13OriginalRecords[$d13I],$d13I)}};R31Check 'undo'

  }finally{
   try{$d7Slot.SetValue($null,$d13OldSlot)}finally{if($d13Owned){$d13Cleanup=[object[]]::new(1);$d13Cleanup[0]=$d13Key[0];$null=$d7NameRemove.Invoke($d7Names,$d13Cleanup);$null=$d7Remove.Invoke($d7Table,$d13Cleanup)}}
  }
 }
 $d13Lookup[1]=$null;$d13NameLookup[1]=$null;Need (-not $d7Lookup.Invoke($d7Table,$d13Lookup) -and -not $d7NameLookup.Invoke($d7Names,$d13NameLookup)) 'Only owned fresh associations removed'
  Same ([Convert]::ToBase64String([IO.File]::ReadAllBytes((Safe $ledgerName)))) ([Convert]::ToBase64String($d7Ledger)) 'All generated controls preserve whole raw ledger and fee/owner state'
}}}

# Disk trust negatives use original hardpin and real cold/warm IO.
"""+historical_fixture.d19_r35_loader_entry()+"""
$c5Path=Safe ($kb+'/c5.json');$c5Raw=[IO.File]::ReadAllBytes($c5Path)
$loaderCache=$script:KCatalogCache;$loaderOrigin=$script:KCatalogOrigin;$loaderManifest=$script:KCatalogManifestContract
$loaderWarm=$loaderCache.warm_reuses;$loaderCompileStart=$loaderCache.compile_start_ticks;$loaderCompileEnd=$loaderCache.compile_end_ticks
try{
 foreach($temperature in @('cold','warm')){
  foreach($diskCase in @('drift','missing','historical','cross-epoch')){"""+historical_fixture.d19_r35_loader_entry()+"""
   try{
    if($temperature -ceq 'cold'){$script:KCatalogCache=$null;$script:KCatalogOrigin=$null}else{$script:KCatalogCache=$loaderCache;$script:KCatalogOrigin=$loaderOrigin}
    if($diskCase -ceq 'missing'){[IO.File]::Delete($c5Path)}
    elseif($diskCase -ceq 'historical'){$historical=K-Raw @{path=$kb+'/s28-before/'+$kb+'/c5.json';bytes=[long]14263;sha256='688beff537b45b8163d956d09ab3e6c70192729f5aa81ce5b34f8cb66f3d262d'} 262144;[IO.File]::WriteAllBytes($c5Path,$historical)}
    elseif($diskCase -ceq 'cross-epoch'){$diskTree=K-Json $utf8.GetString($c5Raw);$diskTree.epoch='foreign';[IO.File]::WriteAllBytes($c5Path,$utf8.GetBytes(($diskTree|ConvertTo-Json -Depth 100 -Compress)+[char]10))}
    else{$driftRaw=[byte[]]::new($c5Raw.Length+1);[Array]::Copy($c5Raw,$driftRaw,$c5Raw.Length);$driftRaw[-1]=32;[IO.File]::WriteAllBytes($c5Path,$driftRaw)}
    $loaderError=$null;try{K-CatalogInit}catch{$loaderError=$_.Exception.Message}
    Need ($null -ne $loaderError) ('Actual hardpin fresh disk refusal '+$temperature+'/'+$diskCase)
    $r35DiskExpected=if($diskCase -ceq 'missing'){@('Bounded actual regular Source')}elseif($diskCase -ceq 'cross-epoch'){@('Whole K evidence bytes','Whole K SHA')}else{@('Whole K evidence bytes')}
    Need ($loaderError -cin $r35DiskExpected) 'Exact reached disk hardpin refusal'
    if($temperature -ceq 'cold'){Need ($null -eq $script:KCatalogCache -and $null -eq $script:KCatalogOrigin) 'Cold failed loader never compiles/adopts cache'}
    else{Need ([object]::ReferenceEquals($script:KCatalogCache,$loaderCache) -and [object]::ReferenceEquals($script:KCatalogOrigin,$loaderOrigin)) 'Warm failed loader keeps actual cache/origin'}
    Same @($loaderCache.warm_reuses,$loaderCache.compile_start_ticks,$loaderCache.compile_end_ticks) @($loaderWarm,$loaderCompileStart,$loaderCompileEnd) 'Loader failures never count warm success or compile'
   }finally{[IO.File]::WriteAllBytes($c5Path,$c5Raw);$script:KCatalogCache=$loaderCache;$script:KCatalogOrigin=$loaderOrigin}
  }
 }
}finally{[IO.File]::WriteAllBytes($c5Path,$c5Raw);$script:KCatalogCache=$loaderCache;$script:KCatalogOrigin=$loaderOrigin;$script:KCatalogManifestContract=$loaderManifest}
# Synthetic internal-envelope branches: every shim first invokes the saved real reader.
$actualLoaderRead=(Get-Item -LiteralPath Function:K-Read).ScriptBlock
$r35EnvelopeLabels=@{
 'schema'='Exact authenticated catalog code envelope schema'
 'extra-key'='Exact genuine dictionary field count'
 'missing-key'='Exact genuine dictionary field count'
 'source-type'='Typed catalog code envelope'
 'source-code'='Exact sole catalog type identifier'
 'rendered-hash'='Hardcoded rendered catalog source identity'
 'template-hash'='Hardcoded catalog template identity'
 'epoch'='Current provenance contract schema epoch'
}
try{
 foreach($script:S28EnvelopeCase in @('schema','extra-key','missing-key','source-type','source-code','rendered-hash','template-hash','epoch')){
  Set-Item -LiteralPath Function:K-Read -Value $actualLoaderRead
"""+historical_fixture.d19_r35_loader_entry()+"""
  Set-Item -LiteralPath Function:K-Read -Value {
   param($_p)
   $actualTree=& $actualLoaderRead $_p
   if($_p.path -ceq ($kb+'/c5.json')){
    $detachedTree=K-Clone $actualTree
    switch($script:S28EnvelopeCase){
     'schema' {$detachedTree.catalog_backend_code.schema='foreign'}
     'extra-key' {$detachedTree.catalog_backend_code.extra=$true}
     'missing-key' {$detachedTree.catalog_backend_code.Remove('source')}
     'source-type' {$detachedTree.catalog_backend_code.source=[long]1}
     'source-code' {$detachedTree.catalog_backend_code.source='public class Foreign {}'}
     'rendered-hash' {$detachedTree.catalog_backend_code.rendered_sha256='0'*64}
     'template-hash' {$detachedTree.catalog_backend_code.template_sha256='0'*64}
     'epoch' {$detachedTree.epoch='foreign'}
    }
    return $detachedTree
   }
   return $actualTree
  }
  $loaderError=$null;try{K-CatalogInit}catch{$loaderError=$_.Exception.Message}
  Need ($null -ne $loaderError) ('Synthetic envelope guard after actual fresh authentication '+$script:S28EnvelopeCase)
  Same $loaderError $r35EnvelopeLabels[$script:S28EnvelopeCase] 'Exact reached envelope refusal'
  Need ([object]::ReferenceEquals($script:KCatalogCache,$loaderCache) -and [object]::ReferenceEquals($script:KCatalogOrigin,$loaderOrigin)) 'Synthetic refusal keeps actual cache/origin'
  Same @($loaderCache.warm_reuses,$loaderCache.compile_start_ticks,$loaderCache.compile_end_ticks) @($loaderWarm,$loaderCompileStart,$loaderCompileEnd) 'Synthetic malformed envelope never compiles or counts warm success'
 }
}finally{Set-Item -LiteralPath Function:K-Read -Value $actualLoaderRead;$script:S28EnvelopeCase=$null}
Need ($null -ne $script:KCatalogCache -and $null -ne $script:KCatalogOrigin) 'Actual production catalog backend entered'
$catalogMetrics=K-CatalogInvoke $script:KCatalogCache.snapshot ([object[]]@())

Need (-not $script:KCatalogCache.source.Contains('static readonly long[] WideCaps')) 'S27 exact scalar cap policy has no constant primitive aggregate'
Same $catalogMetrics[0] ([long]$catalog.rows.Length) 'Actual N rows started'
Same $catalogMetrics[1] ([long]$catalog.rows.Length) 'Actual N rows authenticated'
Same $catalogMetrics[2] ([long](2*$catalog.rows.Length)) 'Actual 2N fresh FileShare.Read opens'
Same $catalogMetrics[3] ([long](2*$catalog.rows.Length)) 'Actual 2N complete capped read passes'
Same $catalogMetrics[4] ([long]$catalog.rows.Length) 'Actual N raw content SHA calls'
Need ($catalogMetrics[7] -is [string] -and $catalogMetrics[7].Length -eq 16 -and $catalogMetrics[8].Length -eq 12) 'Bounded actual order checksum and failed-prefix witness'

$actualPath=[IO.Path]::GetFullPath((Safe $catalog.rows[0].pin.path))
Same $catalogMetrics[8][1] ('0:open:0:'+$actualPath) 'Actual canonical FileStream open path witness'
Same $catalogMetrics[8][2] ('0:pass:0:'+$actualPath) 'Actual completed first read path witness'
Same $catalogMetrics[8][3] ('0:open:1:'+$actualPath) 'Actual distinct second open path witness'
Same $catalogMetrics[8][4] ('0:pass:1:'+$actualPath) 'Actual completed second read path witness'
Same $catalogMetrics[8][5] ('0:sha:0:'+$actualPath) 'Actual first read buffer raw content SHA path witness'
Need ($script:KCatalogJournal.context -ceq 'generated') 'Actual existing Generated observation journal context'
Same $script:KCatalogJournal.command ([IO.Path]::GetFullPath($PSCommandPath)) 'Actual existing Generated wrapper command'
Same $script:KCatalogJournal.root ([IO.Path]::GetFullPath($taskRoot)) 'Actual contained Generated observation root'
Same $script:KCatalogJournal.path ([IO.Path]::Combine([IO.Path]::GetDirectoryName($PSCommandPath),'s28-catalog-leaves.jsonl')) 'Same observation journal name preserved'
$d16EntryRaw=[Environment]::GetEnvironmentVariable('FLYLAB_DIRECT_SCIENCE_OWNER');$existingParent=K-Json $d16EntryRaw
Same $script:KCatalogJournal.identity (([IO.Path]::GetFullPath($PSCommandPath))+'|'+[BitConverter]::DoubleToInt64Bits($existingParent.entry_counter_s)) 'Journal bound to existing actual parent ENTRY, no new clock'
$actualJournal=$script:KCatalogJournal;$journalOrdinal=$actualJournal.ordinal;$journalBytes=$actualJournal.bytes
$d16JournalBefore=K-Clone $actualJournal;$d16JournalRaw=K-SourceRaw $actualJournal.path 524288
$d16EntryCases=@(
 @{scope='request';field='path';value='verification/banc-natural-gait-m3-20260919/s28-f15c-root/full4-request.json';label='Literal K metadata path'},
 @{scope='request';field='bytes';value=$true;label='K exact integer, never bool or float'},
 @{scope='request';field='bytes';value=[double]1;label='K exact integer, never bool or float'},
 @{scope='request';field='bytes';value=[long]0;label='Typed bounded K pin'},
 @{scope='request';field='bytes';value=[long]262145;label='Actual bounded parent request descriptor'},
 @{scope='request';field='sha256';value=('G'*64);label='Typed bounded K pin'},
 @{scope='request';field='sha256';value=$true;label='Typed bounded K pin'},
 @{scope='owner';field='owner_pid';value=[long]0;label='Same actual typed parent ENTRY identity'},
 @{scope='owner';field='owner_pid';value=$true;label='Same actual typed parent ENTRY identity'},
 @{scope='owner';field='owner_creation_ticks';value=[long]0;label='Same actual typed parent ENTRY identity'},
 @{scope='owner';field='owner_creation_ticks';value=[double]1;label='Same actual typed parent ENTRY identity'},
 @{scope='owner';field='entry_counter_s';value=[double]0;label='Same actual typed parent ENTRY identity'},
 @{scope='owner';field='entry_counter_s';value=[long]1;label='Same actual typed parent ENTRY identity'},
 @{scope='owner';field='job_name';value='Local\foreign';label='Same actual typed parent ENTRY identity'},
 @{scope='owner';field='job_name';value=('Local\flylab-startup-ray-'+('x'*257));label='Same actual typed parent ENTRY identity'}
)
foreach($d16EntryCase in $d16EntryCases){
 $d16WrongEntry=K-Clone $existingParent
 if($d16EntryCase.scope -ceq 'request'){$d16WrongEntry.request[$d16EntryCase.field]=$d16EntryCase.value}else{$d16WrongEntry[$d16EntryCase.field]=$d16EntryCase.value}
 $d16EntryError=$null
 try{
  [Environment]::SetEnvironmentVariable('FLYLAB_DIRECT_SCIENCE_OWNER',($d16WrongEntry|ConvertTo-Json -Depth 100 -Compress))
  try{K-CatalogJournal 'catalog-support' 'BEGIN'}catch{$d16EntryError=$_.Exception.Message}
  Same $d16EntryError $d16EntryCase.label 'Exact inherited ENTRY journal grammar guard'
 }finally{[Environment]::SetEnvironmentVariable('FLYLAB_DIRECT_SCIENCE_OWNER',$d16EntryRaw)}
 Same ([Environment]::GetEnvironmentVariable('FLYLAB_DIRECT_SCIENCE_OWNER')) $d16EntryRaw 'Exact real inherited environment restoration'
 Same $script:KCatalogJournal $d16JournalBefore 'Complete retained journal state unchanged after token refusal'
 Need ([Object]::ReferenceEquals($actualJournal,$script:KCatalogJournal)) 'ENTRY grammar refusal never replaces journal'
 Same ([Convert]::ToBase64String((K-SourceRaw $actualJournal.path 524288))) ([Convert]::ToBase64String($d16JournalRaw)) 'ENTRY grammar refusal appends no actual journal bytes'
}
foreach($change in @(@{field='context';label='Same authenticated journal context; unknown context never reused'},@{field='identity';label='Same authenticated journal context; unknown context never reused'},@{field='path';label='Same authenticated journal context; unknown context never reused'},@{field='command';label='No partial foreign journal context adoption'},@{field='root';label='No partial foreign journal context adoption'},@{field='ordinal';label='Exact retained journal counters'})){
 $savedJournalValue=$actualJournal[$change.field];$journalError=$null
 try{
  $actualJournal[$change.field]=if($change.field -ceq 'ordinal'){[double]0}else{'foreign'}
  try{K-CatalogJournal 'catalog-support' 'BEGIN'}catch{$journalError=$_.Exception.Message}
  Same $journalError $change.label 'Exact foreign/partial journal context refusal before IO'
  Same $actualJournal.bytes $journalBytes 'Rejected journal context wrote zero bytes'
 }finally{$actualJournal[$change.field]=$savedJournalValue}
 Same $actualJournal.ordinal $journalOrdinal 'Rejected journal context retains prior actual ordinal'
}
Need ([object]::ReferenceEquals($actualJournal,$script:KCatalogJournal)) 'Generated warm journal context never replaced or reset'
Same (K-CatalogAuth $catalog) $policy 'One existing whole positive: entire Native old/new detached typed policy'
$detached=$policy;$beforePath=$catalog.rows[0].pin.path;$detached[0][0]='returned-policy-mutation'
Same $catalog.rows[0].pin.path $beforePath 'Detached row output cannot mutate catalog';$detached[0][0]=$beforePath
"""+historical_fixture.d19_r35_loader_entry()+"""
$warm=$script:KCatalogCache.warm_reuses;K-CatalogInit;Need ($script:KCatalogCache.warm_reuses -eq $warm+1) 'Actual retained catalog warm reuse'
$cache=$script:KCatalogCache;$origin=$script:KCatalogOrigin
foreach($field in @('type','assembly','support','verify','snapshot','source','source_hash','template_hash')){
"""+historical_fixture.d19_r35_loader_entry()+"""
 $saved=$cache[$field]
 try{$cache[$field]=if($field -cin @('source','source_hash','template_hash')){'foreign'}else{$null};$message=$null;try{K-CatalogInit}catch{$message=$_.Exception.Message};Need ($null -ne $message) 'Catalog foreign/partial/stale cache refusal';Same $message $(if($field -cin @('type','assembly')){'No foreign actual catalog Type/Assembly'}elseif($field -cin @('support','verify','snapshot')){'No foreign or stale actual catalog Method'}else{'No stale catalog literal/source/hash cache'}) 'Exact reached cache metadata refusal'}finally{$cache[$field]=$saved}
}
Need ([object]::ReferenceEquals($script:KCatalogOrigin,$origin)) 'Actual catalog origin never replaced during refusal'
K-CatalogInit
$script:S26LegacyReader=$null
function Invoke-S26CatalogError($value,[string]$expected,[bool]$io=$false,[bool]$observeOld=$false){
 $oldMessage=$null;$newMessage=$null;$oldType=$null;$newType=$null
 try{
  if($observeOld){
   $script:S26LegacyReader=(Get-Item -LiteralPath Function:\\K-SourceRaw).ScriptBlock
   Need ($script:S26LegacyReader -is [scriptblock]) 'Actual script legacy reader'
   $observer={param($a,$c,$o=$false);$script:S26OldReadPrefix.Add($a);$script:S26OldReadEvents.Add(@($a,$c,$o));& $script:S26LegacyReader $a $c $o}
   Set-Item -LiteralPath Function:\\K-SourceRaw -Value $observer
   Need ([object]::ReferenceEquals((Get-Item -LiteralPath Function:\\K-SourceRaw).ScriptBlock,$observer)) 'Actual script legacy observer installed'
  }
  try{$null=K-CatalogAuth $value}catch{$cause=$_.Exception.GetBaseException();$oldMessage=$cause.Message;$oldType=$cause.GetType()}
 }finally{
  if($observeOld -and $null -ne $script:S26LegacyReader){
   Set-Item -LiteralPath Function:\\K-SourceRaw -Value $script:S26LegacyReader
   Need ([object]::ReferenceEquals((Get-Item -LiteralPath Function:\\K-SourceRaw).ScriptBlock,$script:S26LegacyReader)) 'Actual script legacy reader restored'
   $script:S26LegacyReader=$null
  }
 }
 try{$null=K-CatalogVerified $value}catch{$cause=$_.Exception.GetBaseException();$newMessage=$cause.Message;$newType=$cause.GetType()}
 Need ($null -ne $oldMessage -and $null -ne $newMessage) 'Old and production catalog both rejected'
 Same $newMessage $oldMessage 'Exact catalog first label/underlying cause message'
 if($io){Same $newType $oldType 'Exact catalog underlying IO/UTF8 cause type'}else{Same $newMessage $expected 'Exact intended first catalog label'}
}
# Recognized malformed builtin fields are rejected at their original pre-IO site.
foreach($change in @(@{key='ordinal';value=$true;label='K exact integer, never bool or float'},@{key='ordinal';value=[long]1;label='Catalog ordinal'},@{key='reason';value=[long]1;label='Reader reason'},@{key='kind';value='foreign';label='Reader kind'},@{key='cap';value=[double]262144;label='K exact integer, never bool or float'},@{key='cap';value=[long]1;label='Path cap every reader'})){
 $wrong=K-Clone $catalog;$wrong.rows[0][$change.key]=$change.value
 Invoke-S26CatalogError $wrong $change.label
 $actual=K-CatalogInvoke $script:KCatalogCache.snapshot ([object[]]@());Same $actual[2] ([long]0) 'Pre-IO malformed builtin has zero actual opens'
}
$wrong=K-Clone $catalog;$wrong.schema='FLYLAB.M3.S26.ROOT-static-historical-catalog.v1';Invoke-S26CatalogError $wrong 'Catalog schema'
$wrong=K-Clone $catalog;$wrong.rows[0]['Keys']='shadow';Invoke-S26CatalogError $wrong 'Exact genuine dictionary field count'
$empty=@($catalog.rows|Where-Object {$_.kind -ceq 'raw0'})[0]
$wrong=K-Clone $catalog;$wrong.rows[0].pin=K-Clone $empty.pin;$wrong.rows[0].kind='raw0';$wrong.rows[0].cap=[long]262144;$wrong.rows[0].pin.bytes=[int]0
Invoke-S26CatalogError $wrong 'Exact empty pin.bytes type'
# Unsupported representation is decided by the pure method before any reader.
$supportArgs=[object[]]::new(1)
foreach($value in @([pscustomobject]@{schema=$catalog.schema;rows=$catalog.rows},[Collections.Generic.Dictionary[string,object]]::new(),[Array]::CreateInstance([object],[int[]]@(1),[int[]]@(1)))){
 $supportArgs[0]=$value;Same (K-CatalogInvoke $script:KCatalogCache.support $supportArgs) $false 'Exact custom/PSObject/lower-bound unsupported domain'
}
$unsupported=K-Clone $catalog;$unsupported.schema=$catalog.schema+[char]0;$supportArgs[0]=$unsupported
Same (K-CatalogInvoke $script:KCatalogCache.support $supportArgs) $false 'Uncertain culture/control-string whole-old selection before IO'
$selected=Safe $catalog.rows[0].pin.path;$saved=[IO.File]::ReadAllBytes($selected)
$actualSourceReader=(Get-Item -LiteralPath Function:\\K-SourceRaw).ScriptBlock
Need ($actualSourceReader -is [scriptblock]) 'Actual caller legacy reader'
try{
 [IO.File]::Delete($selected)
 $wrong=K-Clone $catalog;$wrong.rows[1].ordinal=[long]999
 $script:S26OldReadPrefix=[Collections.Generic.List[string]]::new()
 $script:S26OldReadEvents=[Collections.Generic.List[object]]::new()
 $oldMessage=$null;$newMessage=$null;$oldType=$null;$newType=$null
 $s55w={param($s55n,$s55d);Need ($s55n -ceq 's26-reader-slot-observation.json' -or $s55n -ceq 's26-script-reader-observation.json') 'Finite reader diagnostic';$s55b=$utf8.GetBytes(($s55d|ConvertTo-Json -Depth 20 -Compress));Need ($s55b.Length -le 262144) 'Bounded actual reader diagnostic';$s55f=[IO.File]::Open(($taskRoot+'/'+$s55n),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None);try{$s55f.Write($s55b,0,$s55b.Length)}finally{$s55f.Dispose()}}
 $script:S26LegacyReader=$null
 try{
  $s55v=$null;$s55e=$null;try{$s55v=${function:K-SourceRaw}}catch{$s55e=$_}
  & $s55w 's26-reader-slot-observation.json' ([ordered]@{null=($null -eq $s55v);type=$(if($null -ne $s55v){[type]::GetTypeFromHandle([type]::GetTypeHandle($s55v)).FullName});scriptblock=($s55v -is [scriptblock]);caller_ref=[object]::ReferenceEquals($actualSourceReader,$s55v);retrieval_error=($null -ne $s55e);error_type=$(if($null -ne $s55e){$s55e.Exception.GetType().FullName});error_message=$(if($null -ne $s55e){$s55e.Exception.Message});error_stack=$(if($null -ne $s55e){$s55e.ScriptStackTrace});namespace='function:K-SourceRaw'})
  if($null -ne $s55e){throw $s55e}
  Need ($s55v -is [scriptblock]) 'Actual script legacy reader'
  $s57c=Get-Command -Name K-SourceRaw -ErrorAction Stop
  Need ($s57c -is [System.Management.Automation.FunctionInfo] -and [object]::ReferenceEquals($s57c.ScriptBlock,$s55v)) 'Actual visible legacy command before installation'
  $script:S26LegacyReader=$s55v
  $s26SameSlot=[object]::ReferenceEquals($actualSourceReader,$script:S26LegacyReader)
  $observer={param($a,$c,$o=$false);$script:S26OldReadPrefix.Add($a);$script:S26OldReadEvents.Add(@($a,$c,$o));& $script:S26LegacyReader $a $c $o}
  Set-Item -LiteralPath Function:\\K-SourceRaw -Value $observer
  $s26Installed=[object]::ReferenceEquals(${function:K-SourceRaw},$observer);Need $s26Installed 'Actual script legacy observer installed'
  $s57c=Get-Command -Name K-SourceRaw -ErrorAction Stop
  $s57i=$s57c -is [System.Management.Automation.FunctionInfo] -and [object]::ReferenceEquals($s57c.ScriptBlock,$observer)
  Need $s57i 'Actual visible legacy observer installed'
  try{$null=K-CatalogAuth $wrong}catch{$cause=$_.Exception.GetBaseException();$oldMessage=$cause.Message;$oldType=$cause.GetType()}
 }finally{
  if($null -ne $script:S26LegacyReader){
   Set-Item -LiteralPath Function:\\K-SourceRaw -Value $script:S26LegacyReader
   $s26Restored=[object]::ReferenceEquals(${function:K-SourceRaw},$script:S26LegacyReader);Need $s26Restored 'Actual script legacy reader restored'
   $s57c=Get-Command -Name K-SourceRaw -ErrorAction Stop
   $s57r=$s57c -is [System.Management.Automation.FunctionInfo] -and [object]::ReferenceEquals($s57c.ScriptBlock,$script:S26LegacyReader)
   Need $s57r 'Actual visible legacy reader restored'
   $script:S26LegacyReader=$null
  }
 }
 try{$null=K-CatalogVerified $wrong}catch{$cause=$_.Exception.GetBaseException();$newMessage=$cause.Message;$newType=$cause.GetType()}
 Need ($null -ne $oldMessage -and $null -ne $newMessage) 'Old and production catalog both rejected'
 Same $newMessage $oldMessage 'Exact catalog first label/underlying cause message'
 Same $newMessage 'Bounded actual regular Source' 'Exact intended first catalog label'
 $s26Snap=K-CatalogInvoke $script:KCatalogCache.snapshot ([object[]]@())
 $s26Diag=[ordered]@{prefix_count=[long]$script:S26OldReadPrefix.Count;event_count=[long]$script:S26OldReadEvents.Count;prefix=@($script:S26OldReadPrefix);events=@($script:S26OldReadEvents);selected=$selected;cap=[long]262144;opaque=$false;visible_same_script=$s26SameSlot;installed=$s26Installed;restored=$s26Restored;old_message=$oldMessage;new_message=$newMessage;old_type=$oldType.FullName;new_type=$newType.FullName;snapshot=$s26Snap;resolved_installed=$s57i;resolved_restored=$s57r}
 & $s55w 's26-script-reader-observation.json' $s26Diag
 Same @($script:S26OldReadPrefix) @((Safe (K-Paths).contract),$selected) 'Prefix'
 Same @($script:S26OldReadEvents) @(@((Safe (K-Paths).contract),262144L,$false),@($selected,262144L,$false)) 'Events'
 $actual=K-CatalogInvoke $script:KCatalogCache.snapshot ([object[]]@())
 Same @($actual[0],$actual[1],$actual[2],$actual[3],$actual[4]) @([long]1,[long]0,[long]0,[long]0,[long]0) 'Actual compiled failed prefix, no invented row/open/hash success'
 Same $actual[8] @('0:row:0') 'Exact actual first failed read order prefix'
}finally{
 try{Set-Item -LiteralPath Function:\\K-SourceRaw -Value $actualSourceReader}
 finally{[IO.File]::WriteAllBytes($selected,$saved)}
}
# Both fresh passes finish before strict UTF8 rejection; no IO failure fallback.
try{
 [IO.File]::WriteAllBytes($selected,[byte[]]@(255))
 $wrong=K-Clone $catalog;$wrong.rows[0].pin.bytes=[long]1;$wrong.rows[0].pin.sha256=([Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([byte[]]@(255)))).ToLowerInvariant()
 Invoke-S26CatalogError $wrong '' $true
 $actual=K-CatalogInvoke $script:KCatalogCache.snapshot ([object[]]@())
 Same @($actual[0],$actual[1],$actual[2],$actual[3],$actual[4]) @([long]1,[long]0,[long]2,[long]2,[long]0) 'Actual failed UTF8 two passes, zero content SHA'
}finally{[IO.File]::WriteAllBytes($selected,$saved)}
foreach($change in @(@{key='cap';value=$true},@{key='cap';value=[double]262144},@{key='cap';value=[long]65536},@{key='kind';value='alias'})){$wrong=K-Clone $catalog;$wrong.rows[0][$change.key]=$change.value;$rejected=$false;try{$null=K-CatalogVerified $wrong}catch{$rejected=$true};Need $rejected 'Catalog typed/path/kind policy';Same (K-CatalogVerified $catalog) $policy 'Whole positive catalog after mutation'}
$wrong=K-Clone $catalog;$wrong.rows[0].pin.path+='-alias';$rejected=$false;try{$null=K-CatalogVerified $wrong}catch{$rejected=$true};Need $rejected 'Catalog alias rejected'
$wrong=K-Clone $catalog;$swap=$wrong.rows[0];$wrong.rows[0]=$wrong.rows[1];$wrong.rows[1]=$swap;$rejected=$false;try{$null=K-CatalogVerified $wrong}catch{$rejected=$true};Need $rejected 'Catalog order rejected'
$wrong=K-Clone $catalog;$wrong.rows=@($wrong.rows|Select-Object -Skip 1);$rejected=$false;try{$null=K-CatalogVerified $wrong}catch{$rejected=$true};Need $rejected 'Catalog omitted row rejected';Same (K-CatalogVerified $catalog) $policy 'Whole catalog final positive'
$row=@{pin=(K-HistoryPins).report;cap=[long]262144;kind='alias'};$rejected=$false;try{K-HistoryOperand $row}catch{$rejected=$true};Need $rejected 'Changed reader policy rejected'
Write-S19CasePhase 'catalog' 'end'

Write-S19CasePhase 'mutations' 'begin'
$baseline=K-Read (K-BoundedPin $ledgerName) 524288;$baselineState=K-Validate $baseline
$positive=K-Clone $baseline;K-Same $positive $baseline 'Whole positive Native ledger clone'
Same (K-Validate $positive) $baselineState 'Positive Native production validator control'
$expected=@(@{field='original_jobs_running';value=$true},@{field='direct_science_generated_limit_s';value=[long]1964},@{field='startup_ray_original_grants';value=@{}},@{field='extra';value=[long]0})
$names=@(PATHS);Need ($names.Length -eq $expected.Length) 'Whole four mutation operands'
for($index=0;$index -lt $names.Length;$index++){
    $mutation=K-Read (K-BoundedPin $names[$index]);K-Keys $mutation @('field','value')
    Same $mutation $expected[$index] 'Exact intended field/value mutation before validator'
    $wrong=K-Clone $baseline;K-Same $wrong $baseline 'Whole clone before single mutation'
    $wrong[$mutation.field]=$mutation.value
    $rejected=$false;$validatorInvoked=$false
    try{$validatorInvoked=$true;$null=K-Validate $wrong}catch{$rejected=$true}
    Need ($validatorInvoked -and $rejected) 'Production K-Validate rejects actual prefix/type/mixed caps/grant mutation'
    K-Same $baseline $positive 'Negative validation preserves whole baseline and positive clone'
    Same (K-Validate $baseline) $baselineState 'Whole positive baseline still passes'
}
Write-S19CasePhase 'mutations' 'end'
Write-S19CasePhase 'final' 'begin'
$rejected=$false;try{$null=Conserved-Json '{"a":1,"a":2}'}catch{$rejected=$true}
Need $rejected 'Actual Native duplicate JSON';Write-Output 'PASS synthetic Native negative parity'
Write-S19CasePhase 'final' 'end'
}catch{$s19PhaseJournal.primary=$_;throw}finally{
 $s19PhaseCleanup=$null
 if($null -ne $s19PhaseJournal.stream){
  try{$s19PhaseJournal.stream.Flush()}catch{$s19PhaseCleanup=$_}
  try{$s19PhaseJournal.stream.Dispose()}catch{if($null -eq $s19PhaseCleanup){$s19PhaseCleanup=$_}}
 }
 if($null -ne $s19PhaseCleanup){
  if($null -ne $s19PhaseJournal.primary){try{[Console]::Error.WriteLine('S19 phase journal secondary cleanup failure')}catch{}}
  else{throw $s19PhaseCleanup}
 }
}
""".replace("PATHS", paths)),
        ("iterative-primitives", iterative_case),
    ))
    actual = json.loads(receipts[1]["payload"][-1])
    assert actual == expected
    # These negatives reach the real exclusion witness; they never claim PID lookup or request IO was reached.
    from unittest.mock import patch

    from tests.test_c_banc_contact_runtime_journal import conserved_run_native
    from tools import banc_m3_startup_ray_supervisor as supervisor
    entry_raw = os.environ["FLYLAB_DIRECT_SCIENCE_OWNER"]
    entry = json.loads(entry_raw)
    actual_root = f.admission.ROOT
    lock_pin = r.text_spec(f.admission.SHARED_LOCK, actual_root, 16384)
    r.exact(r.read_pin(lock_pin, actual_root, 16384), entry, "real inherited parent exclusion witness before negatives")
    actual_request = r.read_pin(entry["request"], actual_root, 262144)
    assert actual_request["call_id"] == entry["call_id"]
    assert entry["request"]["sha256"] != "0" * 64
    protected = (lock_pin, entry["request"], r.text_spec(actual_request["ledger_path"], actual_root, 524288))
    protected_caps = {lock_pin["path"]: 16384, entry["request"]["path"]: 262144, actual_request["ledger_path"]: 524288}
    protected_raw = {p["path"]: r.raw_pin(p, actual_root, protected_caps[p["path"]]) for p in protected}
    journal_raw = {p: p.read_bytes() for p in v57.root.glob("generated-native-observation-*/s28-catalog-leaves.jsonl")}
    entry_changes = (
        ("request", "path", k.BASE + "/s28-f15c-root/full4-request.json"),
        ("request", "sha256", "0" * 64),
        ("request", "bytes", entry["request"]["bytes"] + 1),
        ("owner", "owner_pid", entry["owner_pid"] + 1),
        ("owner", "owner_creation_ticks", entry["owner_creation_ticks"] + 1),
        ("owner", "entry_counter_s", entry["entry_counter_s"] + 1.0),
        ("owner", "job_name", entry["job_name"] + "-foreign"),
    )
    for scope, field, value in entry_changes:
        wrong_entry = copy.deepcopy(entry)
        target = wrong_entry["request"] if scope == "request" else wrong_entry
        target[field] = value
        assert wrong_entry != entry
        observation = f.observation_directory(v57)
        with patch.dict(os.environ, {"FLYLAB_DIRECT_SCIENCE_OWNER": json.dumps(wrong_entry, ensure_ascii=True, allow_nan=False)}), patch.object(
                supervisor, "OwnedJob", side_effect=AssertionError("ENTRY refusal unexpectedly reached Job construction")) as job_guard:
            rejected = conserved_run_native(v57.root, [], observation_root=observation)
            assert job_guard.call_count == 0
        assert os.environ["FLYLAB_DIRECT_SCIENCE_OWNER"] == entry_raw
        assert rejected.pre_spawn_refused and rejected.worker_pid is None and rejected.worker_creation_ticks is None
        assert rejected.stdout == rejected.stderr == b"" and not rejected.stdout_eof and not rejected.stderr_eof
        assert rejected.all_owned_ended and not rejected.timeout and not rejected.secondary_errors
        guard_path = ".request." + field if scope == "request" else "." + field
        assert rejected.first_error == {"type": "ValueError", "message": "same live parent exclusion witness" + guard_path}
        assert not rejected.identities and not rejected.unknown_owned_identities and rejected.entry is None
        # The unchanged wrapper intentionally publishes three observation artifacts even before spawn.
        assert all((observation / name).is_file() for name in (
            "generated-conserved-native-observation.json", "generated-conserved-native-stdout.bin", "generated-conserved-native-stderr.bin"))
        assert {p: p.read_bytes() for p in v57.root.glob("generated-native-observation-*/s28-catalog-leaves.jsonl")} == journal_raw
        for p in protected:
            assert r.raw_pin(p, actual_root, protected_caps[p["path"]]) == protected_raw[p["path"]]
    # Separately exercise the genuine bounded request reader and its first refusal labels.
    for field, value, label in (
            ("bytes", True, "typed finite text pin"),
            ("sha256", True, "typed finite text pin"),
            ("sha256", "G" * 64, "typed finite text pin"),
            ("bytes", 262145, "typed finite text pin"),
            ("bytes", entry["request"]["bytes"] + 1, "exact text size before bounded read"),
            ("sha256", "0" * 64, "full text repeated bytes/stamp/hash")):
        wrong_request = copy.deepcopy(entry["request"])
        wrong_request[field] = value
        try:
            r.read_pin(wrong_request, actual_root, 262144)
        except ValueError as error:
            assert str(error) == label
        else:
            raise AssertionError("malformed actual ENTRY request reader input accepted")
        for p in protected:
            assert r.raw_pin(p, actual_root, protected_caps[p["path"]]) == protected_raw[p["path"]]


def test_gate_l_checkpoint_exact_recovery_allocation_and_negatives(tmp_path):
    fixture = f.case(tmp_path, "generated-l-deltas")
    contract = k.data(k.CURRENT_CONTRACT, fixture.root)
    expected_sequence = f.d19_golden(k,fixture)["sequence"]
    k.whole_exact(contract["sequence"], expected_sequence)
    assert list(contract["sequence"][0]) == ["precondition", "mode", "caps", "call"]
    assert list(contract["sequence"][0]["caps"]) == ["cleanup_s", "outer_s", "child_s"]
    assert all(type(value) is int for value in contract["sequence"][0]["caps"].values())
    for wrong_first in (k.BOOTSTRAP_CALL, dict(contract["sequence"][0], extra=0), dict(contract["sequence"][0], caps={"cleanup_s":20,"outer_s":4000,"child_s":True}), dict(contract["sequence"][0], caps={"outer_s":4000,"cleanup_s":20,"child_s":3980})):
        wrong_sequence = copy.deepcopy(expected_sequence)
        wrong_sequence[0] = wrong_first
        f.negative(lambda wrong_sequence=wrong_sequence: k.whole_exact(wrong_sequence, expected_sequence))
    historical_map = r.read_pin(historical_fixture.L4_BEFORE, f.admission.ROOT, 262144)
    requested = historical_map["inputs"][0]["source"]
    row = historical_fixture.l4_before_row(requested, historical_map)
    assert row == historical_map["inputs"][0]
    for field, value in (("ordinal", True), ("cap", 524288)):
        wrong = copy.deepcopy(historical_map)
        wrong["inputs"][0][field] = value
        f.negative(lambda wrong=wrong: historical_fixture.l4_before_row(requested, wrong))
    for field, value in (("path", r.BASE + "/l4-a/01.txt"), ("bytes", 1), ("sha256", "0" * 64)):
        wrong = copy.deepcopy(historical_map)
        wrong["inputs"][0]["archive"][field] = value
        f.negative(lambda wrong=wrong: historical_fixture.l4_before_row(requested, wrong))
    tamper_root = tmp_path / "generated-historical-tamper"
    tamper_root.mkdir()
    map_raw = r.raw_pin(historical_fixture.L4_BEFORE, f.admission.ROOT, 262144)
    archive_raw = r.raw_pin(row["archive"], f.admission.ROOT, row["cap"])
    r.fixture_text_write(tamper_root, historical_fixture.L4_BEFORE["path"], map_raw)
    r.fixture_text_write(tamper_root, row["archive"]["path"], archive_raw)
    assert r.raw_pin(row["archive"], tamper_root, row["cap"]) == archive_raw
    (tamper_root / row["archive"]["path"]).write_bytes(archive_raw + b"# drift\n")
    f.negative(lambda: r.raw_pin(row["archive"], tamper_root, row["cap"]))
    (tamper_root / historical_fixture.L4_BEFORE["path"]).write_bytes(map_raw + b" ")
    f.negative(lambda: r.read_pin(historical_fixture.L4_BEFORE, tamper_root, 262144))
    ambiguous = copy.deepcopy(historical_map)
    ambiguous["inputs"][1] = copy.deepcopy(ambiguous["inputs"][0])
    f.negative(lambda: historical_fixture.l4_before_row(requested, ambiguous))
    fixture.full_source()
    from tests.test_c_banc_contact_runtime_journal import conserved_native_parent_call
    for call in (k.BOOTSTRAP_CALL, k.QUAL_CALLS[2], k.QUAL_CALLS[3]):
        parent_q = {'call_id':call, 'mode':"generated", 'role':"execution"}
        assert conserved_native_parent_call(parent_q, k, r)
        assert not conserved_native_parent_call(dict(parent_q, mode="original"), k, r)
        assert not conserved_native_parent_call(dict(parent_q, role="numerical"), k, r)
    for call in ("direct-science-20261009-contact-s26-bootstrap", "direct-science-20261009-contact-s26-gate-a", "direct-science-20261009-contact-s26-gate-b"):
        parent_q = {'call_id':call, 'mode':"generated", 'role':"execution"}
        assert conserved_native_parent_call(parent_q, k, r)
        assert not conserved_native_parent_call(dict(parent_q, mode="original"), k, r)
        assert not conserved_native_parent_call(dict(parent_q, role="numerical"), k, r)
    for call in ("direct-science-20261009-contact-s27-bootstrap", "direct-science-20261009-contact-s27-gate-a", "direct-science-20261009-contact-s27-gate-b"):
        parent_q = {'call_id':call, 'mode':"generated", 'role':"execution"}
        assert conserved_native_parent_call(parent_q, k, r)
        assert not conserved_native_parent_call(dict(parent_q, mode="original"), k, r)
        assert not conserved_native_parent_call(dict(parent_q, role="numerical"), k, r)
    for call in (k.QUAL_CALLS[0], k.QUAL_CALLS[1], "foreign-bootstrap", k.BOOTSTRAP_CALL+"-alias", "direct-science-20261009-contact-s29-bootstrap", "direct-science-20261009-contact-s29-gate-a", "direct-science-20261009-contact-s29-gate-b"):
        assert not conserved_native_parent_call({'call_id':call, 'mode':"generated", 'role':"execution"}, k, r)
    new_map = k.data({'path': _HISTORY_BASE+'/s24-before-index.json', 'bytes': 4901, 'sha256': '494e1646a6f567fd67e4a98122ed7d1bfd4c0104f4176dee1c681143e25adcc0'}, fixture.root)
    for pair in new_map["pairs"]:
        assert historical_fixture.s24_before_row(pair["before"], new_map, k) == pair
        assert historical_fixture.s24_before_row(pair["archive"], new_map, k) == pair
    for field, value in (("schema", "foreign"), ("pairs", new_map["pairs"][:-1])):
        f.negative(lambda field=field, value=value: historical_fixture.s24_before_row(new_map["pairs"][0]["before"], f.mutation(new_map, field, value), k))
    for field, value in (("cap", True), ("cap", 65536), ("before", dict(new_map["pairs"][0]["before"], path=k.REPLACEMENTS[1])), ("archive", dict(new_map["pairs"][0]["archive"], sha256="0"*64))):
        wrong_map = copy.deepcopy(new_map)
        wrong_map["pairs"][0][field] = value
        f.negative(lambda wrong_map=wrong_map: historical_fixture.s24_before_row(new_map["pairs"][0]["before"], wrong_map, k))
    # Real selected exact-source reader refuses missing/drift without historical fallback.
    from unittest.mock import patch
    for operand_pin, selected_name in ((new_map["pairs"][0]["before"], new_map["pairs"][0]["archive"]["path"]), (new_map["pairs"][0]["archive"], new_map["pairs"][0]["archive"]["path"]), (fixture.freeze["sources"][k.REPLACEMENTS[0]], k.REPLACEMENTS[0])):
        selected_path = fixture.root / selected_name
        saved = selected_path.read_bytes()
        with patch.object(f.admission, "ROOT", fixture.root):
            assert historical_fixture.exact_source_operand(operand_pin) == saved
            try:
                selected_path.write_bytes(saved+b"# drift\n")
                f.negative(lambda operand_pin=operand_pin: historical_fixture.exact_source_operand(operand_pin))
                selected_path.unlink()
                f.negative(lambda operand_pin=operand_pin: historical_fixture.exact_source_operand(operand_pin))
            finally:
                selected_path.write_bytes(saved)
    # New S25 exact adapter, preserving all preceding S24 checks.
    new_map = k.data({'path': _HISTORY_BASE+'/s25-before-index.json', 'bytes': 4901, 'sha256': 'f9c9e4326750fcd02c6de81efdf9ea233883bb90786652f7c44b2e9da5bd88a8'}, fixture.root)
    for pair in new_map["pairs"]:
        assert historical_fixture.s25_before_row(pair["before"], new_map, k) == pair
        assert historical_fixture.s25_before_row(pair["archive"], new_map, k) == pair
    for field, value in (("schema", "foreign"), ("pairs", new_map["pairs"][:-1])):
        f.negative(lambda field=field, value=value: historical_fixture.s25_before_row(new_map["pairs"][0]["before"], f.mutation(new_map, field, value), k))
    for field, value in (("cap", True), ("cap", 65536), ("before", dict(new_map["pairs"][0]["before"], path=k.REPLACEMENTS[1])), ("archive", dict(new_map["pairs"][0]["archive"], sha256="0"*64))):
        wrong_map = copy.deepcopy(new_map)
        wrong_map["pairs"][0][field] = value
        f.negative(lambda wrong_map=wrong_map: historical_fixture.s25_before_row(new_map["pairs"][0]["before"], wrong_map, k))
    # Real selected exact-source reader refuses missing/drift without historical fallback.
    from unittest.mock import patch
    for operand_pin, selected_name in ((new_map["pairs"][0]["before"], new_map["pairs"][0]["archive"]["path"]), (new_map["pairs"][0]["archive"], new_map["pairs"][0]["archive"]["path"]), (fixture.freeze["sources"][k.REPLACEMENTS[0]], k.REPLACEMENTS[0])):
        selected_path = fixture.root / selected_name
        saved = selected_path.read_bytes()
        with patch.object(f.admission, "ROOT", fixture.root):
            assert historical_fixture.exact_source_operand(operand_pin) == saved
            try:
                selected_path.write_bytes(saved+b"# drift\n")
                f.negative(lambda operand_pin=operand_pin: historical_fixture.exact_source_operand(operand_pin))
                selected_path.unlink()
                f.negative(lambda operand_pin=operand_pin: historical_fixture.exact_source_operand(operand_pin))
            finally:
                selected_path.write_bytes(saved)
    # New S25 exact adapter, preserving all preceding S24 checks.
    # New S26 exact adapter, preserving all preceding S25 checks.
    new_map = k.data({'path': _HISTORY_BASE+'/s26-before-index.json', 'bytes': 4913, 'sha256': 'd31419aab4a3afe9eeefae40190e639e3f2dcf1895d1264831ab1c8ad8efd502'}, fixture.root)
    for pair in new_map["pairs"]:
        assert historical_fixture.s26_before_row(pair["before"], new_map, k) == pair
        assert historical_fixture.s26_before_row(pair["archive"], new_map, k) == pair
    for field, value in (("schema", "foreign"), ("pairs", new_map["pairs"][:-1])):
        f.negative(lambda field=field, value=value: historical_fixture.s26_before_row(new_map["pairs"][0]["before"], f.mutation(new_map, field, value), k))
    for field, value in (("cap", True), ("cap", 65536), ("before", dict(new_map["pairs"][0]["before"], path=k.REPLACEMENTS[1])), ("archive", dict(new_map["pairs"][0]["archive"], sha256="0"*64))):
        wrong_map = copy.deepcopy(new_map)
        wrong_map["pairs"][0][field] = value
        f.negative(lambda wrong_map=wrong_map: historical_fixture.s26_before_row(new_map["pairs"][0]["before"], wrong_map, k))
    # Real selected exact-source reader refuses missing/drift without historical fallback.
    from unittest.mock import patch
    for operand_pin, selected_name in ((new_map["pairs"][0]["before"], new_map["pairs"][0]["archive"]["path"]), (new_map["pairs"][0]["archive"], new_map["pairs"][0]["archive"]["path"]), (fixture.freeze["sources"][k.REPLACEMENTS[0]], k.REPLACEMENTS[0])):
        selected_path = fixture.root / selected_name
        saved = selected_path.read_bytes()
        with patch.object(f.admission, "ROOT", fixture.root):
            assert historical_fixture.exact_source_operand(operand_pin) == saved
            try:
                selected_path.write_bytes(saved+b"# drift\n")
                f.negative(lambda operand_pin=operand_pin: historical_fixture.exact_source_operand(operand_pin))
                selected_path.unlink()
                f.negative(lambda operand_pin=operand_pin: historical_fixture.exact_source_operand(operand_pin))
            finally:
                selected_path.write_bytes(saved)
    # New S27 exact adapter, preserving all preceding S26 checks.
    new_map = k.data({'path': _HISTORY_BASE+'/s27-before-index.json', 'bytes': 4915, 'sha256': 'f0c88cff5f55698a7325134bacea2cf89507617ca47e7ece911bd785f611346c'}, fixture.root)
    for pair in new_map["pairs"]:
        assert historical_fixture.s27_before_row(pair["before"], new_map, k) == pair
        assert historical_fixture.s27_before_row(pair["archive"], new_map, k) == pair
    for field, value in (("schema", "foreign"), ("pairs", new_map["pairs"][:-1])):
        f.negative(lambda field=field, value=value: historical_fixture.s27_before_row(new_map["pairs"][0]["before"], f.mutation(new_map, field, value), k))
    for field, value in (("cap", True), ("cap", 65536), ("before", dict(new_map["pairs"][0]["before"], path=k.REPLACEMENTS[1])), ("archive", dict(new_map["pairs"][0]["archive"], sha256="0"*64))):
        wrong_map = copy.deepcopy(new_map)
        wrong_map["pairs"][0][field] = value
        f.negative(lambda wrong_map=wrong_map: historical_fixture.s27_before_row(new_map["pairs"][0]["before"], wrong_map, k))
    # Real selected exact-source reader refuses missing/drift without historical fallback.
    from unittest.mock import patch
    for operand_pin, selected_name in ((new_map["pairs"][0]["before"], new_map["pairs"][0]["archive"]["path"]), (new_map["pairs"][0]["archive"], new_map["pairs"][0]["archive"]["path"]), (fixture.freeze["sources"][k.REPLACEMENTS[0]], k.REPLACEMENTS[0])):
        selected_path = fixture.root / selected_name
        saved = selected_path.read_bytes()
        with patch.object(f.admission, "ROOT", fixture.root):
            assert historical_fixture.exact_source_operand(operand_pin) == saved
            try:
                selected_path.write_bytes(saved+b"# drift\n")
                f.negative(lambda operand_pin=operand_pin: historical_fixture.exact_source_operand(operand_pin))
                selected_path.unlink()
                f.negative(lambda operand_pin=operand_pin: historical_fixture.exact_source_operand(operand_pin))
            finally:
                selected_path.write_bytes(saved)
    # New S28 exact adapter, preserving all preceding S27 checks.
    new_map = k.data({'path': _HISTORY_BASE+'/s28-before-index.json', 'bytes': 4915, 'sha256': '7c9ca2a35817b59ab1f91bbf699f48e1ff40a8394044b2c9d911b1e305255bd5'}, fixture.root)
    for pair in new_map["pairs"]:
        assert historical_fixture.s28_before_row(pair["before"], new_map, k) == pair
        assert historical_fixture.s28_before_row(pair["archive"], new_map, k) == pair
    for field, value in (("schema", "foreign"), ("pairs", new_map["pairs"][:-1])):
        f.negative(lambda field=field, value=value: historical_fixture.s28_before_row(new_map["pairs"][0]["before"], f.mutation(new_map, field, value), k))
    for field, value in (("cap", True), ("cap", 65536), ("before", dict(new_map["pairs"][0]["before"], path=k.REPLACEMENTS[1])), ("archive", dict(new_map["pairs"][0]["archive"], sha256="0"*64))):
        wrong_map = copy.deepcopy(new_map)
        wrong_map["pairs"][0][field] = value
        f.negative(lambda wrong_map=wrong_map: historical_fixture.s28_before_row(new_map["pairs"][0]["before"], wrong_map, k))
    # Real selected exact-source reader refuses missing/drift without historical fallback.
    from unittest.mock import patch
    for operand_pin, selected_name in ((new_map["pairs"][0]["before"], new_map["pairs"][0]["archive"]["path"]), (new_map["pairs"][0]["archive"], new_map["pairs"][0]["archive"]["path"]), (fixture.freeze["sources"][k.REPLACEMENTS[0]], k.REPLACEMENTS[0])):
        selected_path = fixture.root / selected_name
        saved = selected_path.read_bytes()
        with patch.object(f.admission, "ROOT", fixture.root):
            assert historical_fixture.exact_source_operand(operand_pin) == saved
            try:
                selected_path.write_bytes(saved+b"# drift\n")
                f.negative(lambda operand_pin=operand_pin: historical_fixture.exact_source_operand(operand_pin))
                selected_path.unlink()
                f.negative(lambda operand_pin=operand_pin: historical_fixture.exact_source_operand(operand_pin))
            finally:
                selected_path.write_bytes(saved)
    # Admitted Generated gate only: current3 and historical3 exact cap+1 controls.
    cap_profiles = ((k.S15_PINS["refinement_anchor"]["path"], 524288),(k.REPLACEMENTS[0], 131072), (k.REPLACEMENTS[1], 131072),
                    (k.BASE + "/parent-direct-science-account.ps1", 393216),
                    (k.BASE + "/s23-before/" + k.REPLACEMENTS[0], 65536),
                    (k.BASE + "/s23-before/" + k.REPLACEMENTS[1], 65536),
                    (k.BASE + "/s23-before/" + k.BASE + "/parent-direct-science-account.ps1", 262144),
                    (k.BASE + "/s24-before/" + k.REPLACEMENTS[0], 131072),
                    (k.BASE + "/s24-before/" + k.REPLACEMENTS[1], 131072),
                    (k.BASE + "/s24-before/" + k.BASE + "/parent-direct-science-account.ps1", 393216),
                    (k.BASE + "/s25-before/" + k.REPLACEMENTS[0], 131072),
                    (k.BASE + "/s25-before/" + k.REPLACEMENTS[1], 131072),
                    (k.BASE + "/s25-before/" + k.BASE + "/parent-direct-science-account.ps1", 393216),
                    (k.BASE + "/s26-before/" + k.REPLACEMENTS[0], 131072),
                    (k.BASE + "/s26-before/" + k.REPLACEMENTS[1], 131072),
                    (k.BASE + "/s26-before/" + k.BASE + "/parent-direct-science-account.ps1", 393216),
                    (k.BASE + "/s27-before/" + k.REPLACEMENTS[0], 131072),
                    (k.BASE + "/s27-before/" + k.REPLACEMENTS[1], 131072),
                    (k.BASE + "/s27-before/" + k.BASE + "/parent-direct-science-account.ps1", 393216),
                    (k.BASE + "/s28-before/" + k.REPLACEMENTS[0], 131072),
                    (k.BASE + "/s28-before/" + k.REPLACEMENTS[1], 131072),
                    (k.BASE + "/s28-before/" + k.BASE + "/parent-direct-science-account.ps1", 393216))
    for name, cap in cap_profiles:
        assert k.reader_cap(name) == cap
        operand = fixture.root / name
        existed = operand.exists()
        saved = operand.read_bytes() if existed else None
        operand.parent.mkdir(parents=True, exist_ok=True)
        try:
            above = b"x" * (cap + 1)
            operand.write_bytes(above)
            descriptor = {'path':name, 'bytes':len(above), 'sha256':hashlib.sha256(above).hexdigest()}
            f.negative(lambda descriptor=descriptor,name=name: k.bounded_source(descriptor, fixture.root, k.reader_cap(name)))
            f.negative(lambda descriptor=descriptor,name=name: r.raw_pin(descriptor, fixture.root, k.reader_cap(name)))
            exact = above[:-1]
            operand.write_bytes(exact)
            descriptor = {'path':name, 'bytes':len(exact), 'sha256':hashlib.sha256(exact).hexdigest()}
            assert k.bounded_source(descriptor, fixture.root, k.reader_cap(name)) == exact
        finally:
            if existed:
                operand.write_bytes(saved)
            else:
                operand.unlink()
    # Synthetic old-current >262144 exercises disjointness, never installed18 reclassification.
    from unittest.mock import patch
    actual_data = k.data
    for current_name in (k.REPLACEMENTS[0], k.REPLACEMENTS[1], k.BASE + "/parent-direct-science-account.ps1"):
        synthetic_before = actual_data(k.S15_PINS["before"], fixture.root)
        synthetic_before["sources"][current_name] = dict(synthetic_before["sources"][current_name], bytes=262145, sha256="0" * 64)
        def current_before_view(spec, root, cap=262144, synthetic_before=synthetic_before):
            return copy.deepcopy(synthetic_before) if spec == k.S15_PINS["before"] else actual_data(spec, root, cap)
        with patch.object(k, "data", side_effect=current_before_view):
            k.source_fence(fixture.source, fixture.root)
    current_path = fixture.root / "tools/banc_m3_checkpoint_harness_runtime.py"
    current_raw = current_path.read_bytes()
    current_path.write_bytes(current_raw + b"# current drift\n")
    f.negative(lambda: k.source_fence(fixture.source, fixture.root))
    current_path.write_bytes(current_raw)
    review = k.data(fixture.freeze["reviewed_source"], fixture.root)
    direct_contract = k.data(fixture.freeze["sources"][k.CONTRACT_PATH],fixture.root)
    provenance_pins = tuple(direct_contract["provenance"][name] for name in ("root_design","astra_markdown","astra_json","root_reflection"))
    for pin in provenance_pins:
        metadata_path = fixture.root / pin["path"]
        original_bytes = metadata_path.read_bytes()
        try:
            metadata_path.write_bytes(original_bytes + b"\n")
            f.negative(lambda: k.source_fence(fixture.source, fixture.root))
            metadata_path.unlink()
            f.negative(lambda: k.source_fence(fixture.source, fixture.root))
        finally:
            metadata_path.write_bytes(original_bytes)
    bad_reviews = []
    wrong = copy.deepcopy(review)
    wrong["astra_review"].pop("json")
    bad_reviews.append(wrong)
    wrong = copy.deepcopy(review)
    wrong["root_reflection"]["path"] += "-alias"
    bad_reviews.append(wrong)
    wrong = copy.deepcopy(review)
    wrong["root_reflection"].pop("sha256")
    bad_reviews.append(wrong)
    wrong = copy.deepcopy(review)
    wrong["root_reflection"]["bytes"] = 262145
    bad_reviews.append(wrong)
    for wrong in bad_reviews:
        f.negative(lambda wrong=wrong: k.provenance(wrong, fixture.root))
    bad_pins = [fixture.put(k.BASE + "/generated-l13r2-provenance-negative-" + str(i) + ".json", value)
                for i, value in enumerate(bad_reviews)]
    native_paths = ",".join("'" + pin["path"] + "'" for pin in bad_pins)
    from unittest.mock import patch
    bounded = fixture.put(k.BASE + '/generated-l-read-cap.py', 'synthetic Source text')
    reads = []
    real_raw = r.raw_pin
    def observed_raw(spec, root, cap, **kwargs):
        reads.append(cap)
        return real_raw(spec, root, cap, **kwargs)
    with patch.object(r, 'raw_pin', side_effect=observed_raw), patch.object(r, 'source_text', side_effect=AssertionError('old2MiB Source reader')):
        k.bounded_source(bounded, fixture.root, 65536)
    assert reads == [65536]
    f.negative(lambda: k.bounded_source(bounded, fixture.root, bounded['bytes'] - 1))
    relative_path = fixture.root / bounded['path']
    assert relative_path.read_bytes() == r.raw_pin(bounded, fixture.root, 65536)
    oversized = r.fixture_text_write(fixture.root, k.BASE + '/generated-l-oversized.py', b'x' * 65537)
    f.negative(lambda: k.bounded_source(oversized, fixture.root, 65536))
    foreign = dict(bounded, path=str(relative_path))
    f.negative(lambda: k.bounded_source(foreign, fixture.root, 65536))
    cut=k.data(k.data(k.CURRENT_CONTRACT,fixture.root)['d19_fixed_history'],fixture.root);bl=k.data(cut['independent_expected'],fixture.root,524288)
    assert len(k.PRODUCTION_REASONS) == 133 and set(k.PRODUCTION_REASONS) == set(range(13)) | set(range(14,20)) | set(range(22,36)) | set(range(38,138))
    assert set(k.PRODUCTION_REASONS).isdisjoint({-1,13,20,21,36,37,138,139,2147483647})
    _actual320 = k.data(k.CURRENT_CONTRACT, fixture.root)["actual320"]
    pc = k.refinement_before(fixture.root)
    fixture.ledger = pc
    (fixture.root/fixture.ledger_path).write_bytes(r.raw_pin(cut['authenticated_actual'],fixture.root,524288))
    fixture.native_batch((
        ("provenance", """
$f=K-Read (K-BoundedPin 'SOURCE');$rv=K-Read $f.reviewed_source;$null=K-Provenance $rv
$currentDirect=K-Read $rv.sources[(K-Paths).contract]
foreach($pin in @($currentDirect.provenance.root_design,$currentDirect.provenance.astra_markdown,$currentDirect.provenance.astra_json,$currentDirect.provenance.root_reflection)){
    $path=Safe $pin.path;$raw=K-Raw $pin 262144
    try{
        [IO.File]::WriteAllBytes($path,([byte[]]($raw+@([byte]10))))
        $rejected=$false;try{$null=K-SourceFence (K-BoundedPin 'SOURCE')}catch{$rejected=$true}
        Need $rejected 'Production Source fence rejects mutated exact provenance bytes'
        [IO.File]::Delete($path)
        $rejected=$false;try{$null=K-SourceFence (K-BoundedPin 'SOURCE')}catch{$rejected=$true}
        Need $rejected 'Production Source fence rejects missing exact provenance file'
    }finally{[IO.File]::WriteAllBytes($path,$raw)}
}
foreach($name in @(NEGATIVES)){
    $wrong=K-Read (K-BoundedPin $name);$rejected=$false
    try{K-Provenance $wrong}catch{$rejected=$true}
    Need $rejected 'Exact current provenance keys/path/cap rejection'
}

$script:R1ActualCurrentRead=(Get-Command K-Read).ScriptBlock
foreach($currentName in @('tools/banc_m3_checkpoint_harness_runtime.py','tools/banc_m3_checkpoint_harness_runtime_native.ps1',($kb+'/parent-direct-science-account.ps1'))){
 $script:R1SyntheticCurrentBefore=K-Read $script:KBeforeSource
 $script:R1SyntheticCurrentBefore.sources[$currentName].bytes=[long]262145
 $script:R1SyntheticCurrentBefore.sources[$currentName].sha256=('0'*64)
 function K-Read($_pin,[long]$cap=262144){
  if($_pin.path -ceq $script:KBeforeSource.path){return K-Clone $script:R1SyntheticCurrentBefore}
  & $script:R1ActualCurrentRead $_pin $cap
 }
 try{$null=K-SourceFence (K-BoundedPin 'SOURCE')}
 finally{Set-Item -LiteralPath Function:K-Read -Value $script:R1ActualCurrentRead}
}
Write-Output 'PASS exact current3 excluded from synthetic inherited-large classification'
$null=K-Provenance $rv;Write-Output 'PASS current provenance raw bytes, exact paths, shape and bounded caps'
""".replace("SOURCE", fixture.source["path"]).replace("NEGATIVES", native_paths)),
        ("read-cap", "$rejected=$false;try{$null=K-SourceRaw (Safe 'OVERSIZED') 65536}catch{$rejected=$true};Need $rejected 'Actual Source cap, not descriptor only';$raw=K-SourceRaw (Safe 'BOUNDED') 65536;Need ($raw.Length -lt 65536) 'Whole bounded text';Write-Output 'PASS synthetic actual Source read caps'".replace('OVERSIZED', oversized['path']).replace('BOUNDED', bounded['path'])),
        ("recovery", """$null=Start-LedgerTransaction 'generated-synthetic-L-recovery'
try{
 $beforeRaw=[byte[]]$script:ledgerTransaction.before.Clone();$l=Conserved-Json $beforeRaw
 foreach($badCase in @('pointer','owners','other-stage','caps','stale')){
  $wrong=K-Clone $l;$wrongRaw=[byte[]]$beforeRaw.Clone()
  switch($badCase){'pointer'{$wrong[(K-Key)]=K-BoundedPin 'RECOVERY'} 'owners'{$wrong.direct_science_workers=@{foreign=@{}}} 'other-stage'{$wrong.debits=@($wrong.debits)+@($wrong.debits[-1])} 'caps'{$wrong.direct_science_generated_limit_s=[long]5795} 'stale'{$wrongRaw=[byte[]]($wrongRaw+@([byte]10))}}
  $rejected=$false;try{$null=K-RecoveryAction $wrong (K-BoundedPin 'RECOVERY') $wrongRaw}catch{$rejected=$true};Need $rejected 'Bounded current318 owner pointer stage cap or raw refusal'
  Same ([Convert]::ToBase64String([IO.File]::ReadAllBytes((Safe $ledgerName)))) ([Convert]::ToBase64String($beforeRaw)) 'Failure leaves whole fixture ledger raw unchanged'
 }
 $rp=K-BoundedPin 'RECOVERY';foreach($p in @($rp,[ordered]@{path=$rp.path;bytes=$rp.bytes;sha256=$rp.sha256},[ordered]@{sha256=$rp.sha256;bytes=$rp.bytes;path=$rp.path})){$null=K-Recovery $p;D19-R4Whole (K-RecoveredSnapshot $p) (K-RecoveryAction $l $p $beforeRaw)}
  $a=K-RecoveryAction $l (K-BoundedPin 'RECOVERY') $beforeRaw;$null=K-Validate $a
  $bad=K-Clone $a;$p=$bad[(K-Key)];$bad[(K-Key)]=[ordered]@{sha256=$p.sha256;bytes=$p.bytes;path=$p.path};$message=$null;try{$null=K-Validate $bad}catch{$message=$_.Exception.Message};Same $message 'R4 metadata proof: dictionary order' 'D19 actual ledger retains strict order'
 $comparison=K-Clone $a;foreach($n in @(NATIVE_DELTA)){$comparison[$n]=K-Clone $l[$n]};D19-R4Whole $comparison $l
 $null=Save-Ledger $a;Write-Output 'PASS synthetic recovery'
}finally{if($script:ledgerTransaction -and -not $script:ledgerTransaction.uncertain){Stop-LedgerTransaction $script:ledgerTransaction}}
$null=Start-LedgerTransaction 'generated-synthetic-L-duplicate-refinement'
try{$duplicateRaw=[byte[]]$script:ledgerTransaction.before.Clone();$l=Conserved-Json $duplicateRaw;$rejected=$false;try{$null=K-RecoveryAction $l (K-BoundedPin 'RECOVERY') $duplicateRaw}catch{$rejected=$true};Need $rejected 'Immediate duplicate refinement rejected';Same ([Convert]::ToBase64String([IO.File]::ReadAllBytes((Safe $ledgerName)))) ([Convert]::ToBase64String($duplicateRaw)) 'Duplicate leaves refined raw unchanged'}finally{if($script:ledgerTransaction -and -not $script:ledgerTransaction.uncertain){Stop-LedgerTransaction $script:ledgerTransaction}}""".replace("RECOVERY", k.RECOVERY_PATH).replace("NATIVE_DELTA", ",".join("'"+n+"'" for n in sorted(D19_DELTA_FIELDS)))),
    ))
    rc = k.data(r.text_spec(fixture.ledger_path, fixture.root, 524288), fixture.root, 524288)
    assert rc == k.recovered_snapshot(fixture.recovery, fixture.root)
    assert set(rc)|({k.ALLOCATION_KEY} if k.data(k.CURRENT_CONTRACT,fixture.root)['d19_predecessor_allocation_present'] else set()) == set(pc)
    assert {name for name in pc if name not in rc or pc[name] != rc[name]} == D19_DELTA_FIELDS
    for name in set(pc) - D19_DELTA_FIELDS:
        assert rc[name] == pc[name]
    assert rc["debits"] == bl["debits"] and rc["direct_science_calls"] == bl["direct_science_calls"]
    assert set(rc)|({k.ALLOCATION_KEY} if k.data(k.CURRENT_CONTRACT,fixture.root)['d19_predecessor_allocation_present'] else set()) == set(bl)
    assert rc["direct_science_blocking_holds"] == {}
    assert bl["direct_science_blocking_holds"] == {"direct-science-20261009-contact-s28-bootstrap-r63": "L stage unsuccessful; successor blocked"}
    for key in set(bl) - D19_DELTA_FIELDS:
        assert rc[key] == bl[key]
    fixture.ledger = rc
    fixture.admit()
    debit = fixture.python_charge()
    rows = k.roster(fixture.freeze, fixture.root)["bootstrap"]
    xml = '<testsuites name="pytest tests"><testsuite name="pytest" tests="4" failures="0" errors="0" skipped="0">' + ''.join(
        '<testcase classname="' + x["classname"] + '" name="' + x["name"] + '" />' for x in rows) + '</testsuite></testsuites>'
    junit = r.fixture_text_write(fixture.root, k.BASE + "/parent-contact-learning-s28-r64-bootstrap-junit.xml", xml.encode("utf8"))
    before = fixture.put(k.BASE + "/generated-l13r2-allocation-before.json", fixture.ledger)
    event = {"schema": "flylab.m3.checkpoint-allocation.v1", "epoch": k.EPOCH, "before_ledger": before, "recovery": fixture.recovery,
             "source_freeze": fixture.source, "bootstrap": debit["descriptor"], "bootstrap_junit": junit,
             "old_caps": k.CAPS, "new_caps": k.CAPS, "delta_s": {"generated_global": 0, "generated_local": 0, "original_global": 0},
             "combined_global_s": 101400, "new_resource_grant_s": 0, "refund_s": 0}
    fixture.put(k.ALLOCATION_PATH, event)
    result = fixture.native(f.d19_native_allocation(k))
    f.observed_pass(result)
    after = k.data(r.text_spec(fixture.ledger_path, fixture.root, 524288), fixture.root, 524288)
    assert k.validate(after, fixture.root)["caps"] == k.CAPS
    changed = {key for key in fixture.ledger if fixture.ledger[key] != after[key]}
    assert changed == set()  # Successful continuation adds only its pointer; caps/remaining stay exact.
    assert set(after) - set(fixture.ledger) == {k.ALLOCATION_KEY}
    assert after["debits"] == fixture.ledger["debits"]
    f.negative(lambda: k.validate(f.mutation(after, "original_cumulative_limit_s", 15600), fixture.root))


def test_gate_l_checkpoint_actual_atomic_foreign_lock_and_stale_raw(tmp_path):
    fixture = f.case(tmp_path, "generated-l-atomic")
    from unittest.mock import patch
    actual_data = k.data
    staged_contract = actual_data(k.CURRENT_CONTRACT, fixture.root)
    for path in (k.CONTRACT_PATH, k.REVIEW_PATH):
        for duplicate in (False, True):
            modified = copy.deepcopy(staged_contract)
            matching = [row for row in modified["actual320"]["archives"] if row["logical_current_before"]["path"] == path]
            modified["actual320"]["archives"] = [row for row in modified["actual320"]["archives"] if row["logical_current_before"]["path"] != path] + (matching*2 if duplicate else [])
            def decoded(pin, root, cap=262144, modified=modified):
                return copy.deepcopy(modified) if pin == k.CURRENT_CONTRACT else actual_data(pin, root, cap)
            for factory, label in ((f.case, "current"), (historical_fixture.case, "historical")):
                with patch.object(k, "data", side_effect=decoded):
                    try:
                        factory(tmp_path, "fit-stage-"+label+str(duplicate)+path.rsplit("/",1)[-1])
                    except ValueError as error:
                        assert "unique predecessor" in str(error), str(error)
                    else:
                        raise AssertionError("intended predecessor selection guard did not refuse")
    path = fixture.root / fixture.ledger_path
    before = path.read_bytes()
    lock_path = fixture.root / r.SHARED_LOCK
    import time

    from tools import banc_m3_startup_ray_supervisor as supervisor
    q,fixture.request_spec=fixture.request()
    token={'schema':'flylab.m3.direct-science.lock-owner.v1','call_id':q['call_id'],'request':fixture.request_spec,'owner_pid':os.getpid(),'owner_creation_ticks':supervisor.creation_ticks(os.getpid()),'entry_counter_s':time.perf_counter(),'job_name':'Local\\generated-synthetic-atomic'}
    lock = r.SharedLock(lock_path, token)
    try:
        f.negative(lambda: r.SharedLock(lock_path, token))
        changed = copy.deepcopy(fixture.ledger)
        f.negative(lambda: r.atomic_ledger_publish(path, b"stale", changed, lock, validate=lambda value: k.validate(value, fixture.root)))
        assert path.read_bytes() == before
        raw = r.atomic_ledger_publish(path, before, changed, lock, validate=lambda value: k.validate(value, fixture.root))
        assert path.read_bytes()==raw
        r.d19_whole(json.loads(raw),changed)
        for alteration in ("missing-pointer","old-namespace","order","owner"):
            bad=copy.deepcopy(changed)
            if alteration=="missing-pointer":bad.pop(k.KEY)
            elif alteration=="old-namespace":bad[k.KEY]["path"]=k.BASE+"/s28-d19w-recovery.json"
            elif alteration=="order":bad[k.KEY]=dict(reversed(list(bad[k.KEY].items())))
            else:lock.value["owner_creation_ticks"]+=1
            f.negative(lambda bad=bad:r.atomic_ledger_publish(path,raw,bad,lock,validate=lambda value:None))
            assert path.read_bytes()==raw
            if alteration=="owner":lock.value["owner_creation_ticks"]-=1
        assert lock_path.read_bytes() == lock.raw
    finally:
        lock.release(True)
    assert not lock_path.exists()
    f.d19_atomic_legacy_positive(tmp_path)
    result = fixture.native("$null=Start-LedgerTransaction 'generated-synthetic-L-atomic';try{$l=Conserved-Json $script:ledgerTransaction.before;$pin=Save-Ledger $l;Same (K-Read $pin 524288) $l 'Actual atomic replace/readback';Write-Output 'PASS synthetic Native atomic'}finally{if($script:ledgerTransaction -and -not $script:ledgerTransaction.uncertain){Stop-LedgerTransaction $script:ledgerTransaction}}")
    f.observed_pass(result)


def test_gate_l_checkpoint_fee_source_drift_overrun_unknown_duplicate(tmp_path):
    fixture = f.case(tmp_path, "generated-l-fee")
    f.d19_fee_association_controls(tmp_path)
    fixture.full_source()
    admission_only = copy.deepcopy(fixture.ledger)
    fixture.admit()
    overrun=historical_fixture.d19_synthetic_overrun(f.d19_golden(k,fixture))
    assert k.dispatch(admission_only, fixture.q, fixture.request_spec, fixture.root, phase="closed")["phase"] == "charged"
    admitted_only = copy.deepcopy(fixture.ledger)
    assert k.dispatch(admitted_only, fixture.q, fixture.request_spec, fixture.root, phase="admitted")["phase"] == "admitted"
    fixture.evidence(overrun, 124, suspended=True)
    synthetic_trace=k.data(fixture.trace_spec,fixture.root)
    assert (synthetic_trace["full_return_s"],(synthetic_trace["first_final_ms"]-synthetic_trace["dispatch_ms"])/1000.0)==(overrun,overrun)
    import base64
    import struct
    def whole_paid(actual,want):
        r.exact(actual,want,"independent whole paid ledger")
        if type(want) is float: assert struct.pack('>d',actual)==struct.pack('>d',want)
        if type(want) is dict:
            assert list(actual)==list(want)
            for key in want:whole_paid(actual[key],want[key])
        elif type(want) is list:
            for left,right in zip(actual,want,strict=True):whole_paid(left,right)
    # Bounded damage follows lawful whole-Source admission and known closed evidence.
    damaged = fixture.root / "tools/banc_m3_contact_learning.py"
    original = r.raw_pin(fixture.freeze["sources"]["tools/banc_m3_contact_learning.py"], fixture.root, 262144)
    damage = original + b'\nraise RuntimeError("D16 bounded executable drift")\n'
    assert len(damage) <= 262144 and damage != original
    # The one Native invocation warms valid Source before writing this exact damage.
    def refuse_same_damage():
        def exact_source_refusal(function):
            try:
                function()
            except ValueError as error:
                assert str(error) == "exact text size before bounded read"
            else:
                raise AssertionError("same damaged Source was accepted")
        assert damaged.read_bytes() == damage
        exact_source_refusal(lambda: k.source_fence(fixture.source, fixture.root))
        exact_source_refusal(lambda: k.dispatch(admission_only, fixture.q, fixture.request_spec, fixture.root, phase="closed"))
        exact_source_refusal(lambda: k.dispatch(admitted_only, fixture.q, fixture.request_spec, fixture.root, phase="admitted"))
        exact_source_refusal(lambda: k.catalog_auth(k.data(k.S15_PINS["catalog"], fixture.root), fixture.root))
        exact_source_refusal(lambda: k.catalog_auth(k.data(k.S15_PINS["catalog"], fixture.root), fixture.root))
    def wire_snapshot(value):
        raw = json.dumps(value, ensure_ascii=True, allow_nan=False).encode("utf8")
        assert 0 < len(raw) <= 524288
        whole_paid(json.loads(raw), value)
        return base64.b64encode(raw).decode("ascii"), len(raw), hashlib.sha256(raw).hexdigest()
    closed_wire = wire_snapshot(admission_only)
    admitted_wire = wire_snapshot(admitted_only)
    original_pin = fixture.freeze["sources"]["tools/banc_m3_contact_learning.py"]
    native_fee = """
[Console]::Error.WriteLine('D19_FEE_STAGE wire '+[Diagnostics.Stopwatch]::GetTimestamp()+' '+[Diagnostics.Stopwatch]::Frequency);$d16ClosedRaw=[Convert]::FromBase64String('CLOSED_B64');Need ($d16ClosedRaw.Length -eq CLOSED_BYTES -and $d16ClosedRaw.Length -le 524288) 'Whole closed snapshot bytes';Same ([Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($d16ClosedRaw)).ToLowerInvariant()) 'CLOSED_SHA' 'Whole closed snapshot SHA';$d16Closed=K-Json $d16ClosedRaw
$d16AdmittedRaw=[Convert]::FromBase64String('ADMITTED_B64');Need ($d16AdmittedRaw.Length -eq ADMITTED_BYTES -and $d16AdmittedRaw.Length -le 524288) 'Whole admitted snapshot bytes';Same ([Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($d16AdmittedRaw)).ToLowerInvariant()) 'ADMITTED_SHA' 'Whole admitted snapshot SHA';$d16Admitted=K-Json $d16AdmittedRaw
$d16RequestPin=K-Json 'REQUEST_PIN';$d16Request=K-Read $d16RequestPin
[Console]::Error.WriteLine('D19_FEE_STAGE dispatch '+[Diagnostics.Stopwatch]::GetTimestamp()+' '+[Diagnostics.Stopwatch]::Frequency);Same (K-Dispatch $d16Closed $d16Request $d16RequestPin 'closed').phase 'charged' 'Valid closed dispatch before damage';Same (K-Dispatch $d16Admitted $d16Request $d16RequestPin 'admitted').phase 'admitted' 'Valid admitted dispatch before damage'
[Console]::Error.WriteLine('D19_FEE_STAGE source '+[Diagnostics.Stopwatch]::GetTimestamp()+' '+[Diagnostics.Stopwatch]::Frequency);$d16Accepted=K-SourceFence (K-BoundedPin (K-Paths).source);$d16OriginalPin=K-Json 'ORIGINAL_PIN';Same $d16Accepted.sources['tools/banc_m3_contact_learning.py'] $d16OriginalPin 'Original operand belongs to accepted Source';$d16Original=K-Raw $d16OriginalPin 262144
[Console]::Error.WriteLine('D19_FEE_STAGE catalog '+[Diagnostics.Stopwatch]::GetTimestamp()+' '+[Diagnostics.Stopwatch]::Frequency);$d16Catalog=K-Read (K-S15Pins).catalog;$null=K-CatalogVerified $d16Catalog;$null=K-CatalogVerified $d16Catalog
$d16Cache=$script:KCatalogCache;$d16Origin=@($script:KCatalogOrigin);$d16Match=$script:KManagedCache;$d16MatchOrigin=@($script:KManagedOrigin)
$d16Types=@($d16Cache.type,$d16Cache.assembly,$d16Match.type,$d16Match.assembly);foreach($d16Type in $d16Types){Need ($null -ne $d16Type) 'Actual retained Type and Assembly before damage'}
$d16Damage=[Convert]::FromBase64String('DAMAGE_B64');Need ($d16Damage.Length -eq DAMAGE_BYTES -and $d16Damage.Length -le 262144) 'Exact bounded damage bytes';Same ([Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($d16Damage)).ToLowerInvariant()) 'DAMAGE_SHA' 'Exact damage SHA'
$d16Path=Safe 'tools/banc_m3_contact_learning.py';[IO.File]::WriteAllBytes($d16Path,$d16Damage)
$d16DamageCheck={Same ([Convert]::ToBase64String((K-SourceRaw $d16Path 262144))) ([Convert]::ToBase64String($d16Damage)) 'Identical damage before and after settlement'}
$d16TypesCheck={Need ([Object]::ReferenceEquals($d16Cache,$script:KCatalogCache) -and [Object]::ReferenceEquals($d16Match,$script:KManagedCache)) 'Retained actual caches';Same @($d16Origin,$d16MatchOrigin) @($script:KCatalogOrigin,$script:KManagedOrigin) 'Retained exact Type origins';$d16Now=@($script:KCatalogCache.type,$script:KCatalogCache.assembly,$script:KManagedCache.type,$script:KManagedCache.assembly);for($d16I=0;$d16I -lt 4;$d16I++){Need ([Object]::ReferenceEquals($d16Types[$d16I],$d16Now[$d16I])) 'Same actual Type and Assembly references'}}
$d16Refuse={foreach($d16Case in @('fence','closed','admitted','catalog','warm')){& $d16DamageCheck;$d16Refusal=$null;try{switch($d16Case){'fence'{$null=K-SourceFence (K-BoundedPin (K-Paths).source)}'closed'{$null=K-Dispatch $d16Closed $d16Request $d16RequestPin 'closed'}'admitted'{$null=K-Dispatch $d16Admitted $d16Request $d16RequestPin 'admitted'}default{$null=K-CatalogVerified $d16Catalog}}}catch{$d16Refusal=$_.Exception.Message};Same $d16Refusal 'Whole K evidence bytes' ('Exact Source first refusal '+$d16Case);Need (-not $script:KManagedSourceAccepted -and $null -eq $script:KCatalogManifestContract -and $script:KPC.Count -eq 0 -and $null -eq $script:KPJ) 'Damaged Source acceptance/cache cleanup';& $d16TypesCheck}}
$d16Primary=$null
try{
[Console]::Error.WriteLine('D19_FEE_STAGE refuse '+[Diagnostics.Stopwatch]::GetTimestamp()+' '+[Diagnostics.Stopwatch]::Frequency); & $d16Refuse
 [Console]::Error.WriteLine('D19_FEE_STAGE commit '+[Diagnostics.Stopwatch]::GetTimestamp()+' '+[Diagnostics.Stopwatch]::Frequency);$null=Start-LedgerTransaction 'generated-synthetic-L-commit';$d16Live=Conserved-Json $script:ledgerTransaction.before
 $d16Receipt=K-Commit $d16Live 'TRACE_PATH'
 [Console]::Error.WriteLine('D19_FEE_STAGE post-commit '+[Diagnostics.Stopwatch]::GetTimestamp()+' '+[Diagnostics.Stopwatch]::Frequency);& $d16DamageCheck
 K-Keys $d16Receipt @('committed','cost','actual_s','exit_code','ledger');Same $d16Receipt.committed $true 'Known committed fee receipt';Same $d16Receipt.actual_s ([double]SYNTHETIC_S) 'Exact synthetic overrun receipt';Same $d16Receipt.exit_code ([long]124) 'Exact synthetic timeout receipt'
 K-PinShape $d16Receipt.cost 'COST_PATH';Need ($d16Receipt.cost.bytes -le 262144) 'Bounded once-paid cost receipt';$d16Cost=K-Read $d16Receipt.cost
 K-PinShape $d16Receipt.ledger $ledgerName;Same $d16Receipt.ledger (K-BoundedPin $ledgerName 524288) 'Exact receipt canonical raw identity';$d16Paid=K-Read $d16Receipt.ledger 524288;Same $d16Paid.debits[-1].descriptor $d16Receipt.cost 'Exact canonical paid cost readback';Same $d16Cost.cost_id $d16Request.call_id 'Exact paid request readback'
 & $d16Refuse
 [Console]::Error.WriteLine('D19_FEE_STAGE restore '+[Diagnostics.Stopwatch]::GetTimestamp()+' '+[Diagnostics.Stopwatch]::Frequency);[IO.File]::WriteAllBytes($d16Path,$d16Original);$null=K-Raw $d16OriginalPin 262144
 $null=K-SourceFence (K-BoundedPin (K-Paths).source);$null=K-CatalogVerified $d16Catalog;$null=K-CatalogVerified $d16Catalog;& $d16TypesCheck
 [IO.File]::WriteAllBytes($d16Path,$d16Damage);& $d16DamageCheck
 [Console]::Error.WriteLine('D19_FEE_STAGE end '+[Diagnostics.Stopwatch]::GetTimestamp()+' '+[Diagnostics.Stopwatch]::Frequency);$d16Receipt|ConvertTo-Json -Depth 100 -Compress
}catch{$d16Primary=$_.Exception;throw}finally{if($null -ne $script:ledgerTransaction -and -not $script:ledgerTransaction.uncertain){try{Stop-LedgerTransaction $script:ledgerTransaction}catch{if($null -eq $d16Primary){throw};[Console]::Error.WriteLine('D16 fee secondary owner cleanup failure')}}}
"""
    replacements = {
        "CLOSED_B64": closed_wire[0], "CLOSED_BYTES": str(closed_wire[1]), "CLOSED_SHA": closed_wire[2],
        "ADMITTED_B64": admitted_wire[0], "ADMITTED_BYTES": str(admitted_wire[1]), "ADMITTED_SHA": admitted_wire[2],
        "REQUEST_PIN": json.dumps(fixture.request_spec, ensure_ascii=True, allow_nan=False),
        "ORIGINAL_PIN": json.dumps(original_pin, ensure_ascii=True, allow_nan=False),
        "DAMAGE_B64": base64.b64encode(damage).decode("ascii"), "DAMAGE_BYTES": str(len(damage)),
        "DAMAGE_SHA": hashlib.sha256(damage).hexdigest(), "TRACE_PATH": fixture.trace_spec["path"],
        "COST_PATH": k.BASE + "/parent-" + fixture.q["call_id"] + "-full-return.json",
        "SYNTHETIC_S": repr(overrun),
    }
    for token, value in replacements.items():
        assert "'" not in value and native_fee.count(token) == 1
        native_fee = native_fee.replace(token, value)
    result = fixture.native(native_fee)
    f.observed_pass(result)
    refuse_same_damage()
    live = k.data(r.text_spec(fixture.ledger_path, fixture.root, 524288), fixture.root, 524288)
    assert live["debits"][-1]["actual_outer_full_return_s"] == overrun and live["generated_shared_remaining_s"] < 0
    assert live["direct_science_blocking_holds"] == {k.BOOTSTRAP_CALL: k.HOLD}
    # Independent sealed paid326, exact grant/event and immutable observations.
    expected = f.d19_snapshot_expected(k,fixture,fixture.recovery)
    request = k.data(fixture.request_spec, fixture.root)
    plan = k.plan(request["call_id"], k.data(fixture.source, fixture.root), fixture.root)
    expected["direct_science_calls"][request["call_id"]] = {"request": fixture.request_spec, "state":"admitted", "mode":plan["mode"], "reserved_s":plan["limits"]["outer_s"]}
    # Current insertion-preserving expected ledger; external evidence encoders unchanged.
    paid_pin = r.text_spec(k.BASE+"/parent-"+request["call_id"]+"-full-return.json",fixture.root)
    trace = k.data(fixture.trace_spec, fixture.root)
    whole_paid(trace,synthetic_trace)
    expected["debits"].append({"cost_id":request["call_id"],"mode":plan["mode"],"actual_outer_full_return_s":overrun,"actual_returned_exit_code":trace["exit_code"],"descriptor":paid_pin})
    expected["direct_science_calls"][request["call_id"]]["state"]="charged"
    for key in ("direct_science_workers","direct_science_reservations","direct_science_pending_actuals"): expected[key]={}
    expected["direct_science_blocking_holds"][request["call_id"]]=k.HOLD
    expected.update(f.d19_independent_totals(k,fixture,expected))
    whole_paid(live,expected)
    refuse_same_damage()
    duplicate = fixture.native_commit(entry_marker=True)
    f.observed_rejection(duplicate)
    assert b'ENTER Start-LedgerTransaction' in duplicate.stdout and b'ENTER K-Commit' in duplicate.stdout
    assert b'Once actual admitted call' in duplicate.stderr
    assert duplicate.returncode != 0
    unchanged = k.data(r.text_spec(fixture.ledger_path, fixture.root, 524288), fixture.root, 524288)
    assert unchanged == live
    unknown = f.case(tmp_path, "generated-l-unknown")
    unknown.admit()
    unknown.unknown_owner()
    token_path = unknown.root / r.SHARED_LOCK
    token_raw = token_path.read_bytes()
    before_raw = (unknown.root / unknown.ledger_path).read_bytes()
    outer = unknown.evidence(3000.0, 124, uncertain=True)
    assert k.closure(outer) is False
    illegal = []
    for index, (key, value) in enumerate((('all_owned_ended', True), ('reservation_released', True),
            ('worker_started', True), ('resume_outcome', 'RESUMED'), ('worker_pid', None), ('unknown_owned_identities', {}))):
        wrong = f.mutation(outer, key, value)
        f.negative(lambda wrong=wrong: k.closure(wrong))
        illegal.append(unknown.put(k.BASE + '/generated-l-uncertainty-negative-' + str(index) + '.json', wrong))
    names = ','.join("'" + x['path'] + "'" for x in illegal)
    unknown_receipts = unknown.native_batch((
        ("uncertainty-layout", "foreach($name in @(NAMES)){$rejected=$false;try{$null=K-Closure (K-Read (K-BoundedPin $name))}catch{$rejected=$true};Need $rejected 'Exact Native uncertainty layout'};Write-Output 'PASS synthetic uncertainty metadata parity'".replace('NAMES', names)),
        ("typed-unknown-state", "function Load-CheckpointLegacy {throw 'forbidden unknown fee loader'};$l=K-Read (K-BoundedPin $ledgerName) 524288;$s=K-Validate $l;$s|ConvertTo-Json -Depth 20 -Compress"),
    ))
    unknown_evidence_bytes = {pin['path']: (unknown.root / pin['path']).read_bytes()
                              for pin in (unknown.trace_spec, unknown.outer_spec)}
    blocked = unknown.native_commit(entry_marker=True)
    f.observed_rejection(blocked)
    assert b'ENTER Start-LedgerTransaction' in blocked.stdout and b'ENTER K-Commit' not in blocked.stdout
    assert blocked.returncode != 0 and token_path.read_bytes() == token_raw
    assert (unknown.root / unknown.ledger_path).read_bytes() == before_raw
    assert not (unknown.root / (k.BASE + '/parent-' + k.BOOTSTRAP_CALL + '-full-return.json')).exists()
    receipt = unknown.python_charge(3000.0, 124, uncertain=True)
    assert receipt == {'committed': False, 'status': 'UNCOMMITTED_OWNERSHIP_UNKNOWN',
                       'cost_published': False, 'owner_and_reservation_retained': True}
    assert (unknown.root / unknown.ledger_path).read_bytes() == before_raw
    assert token_path.read_bytes() == token_raw and not (unknown.root / (k.BASE + '/parent-' + k.BOOTSTRAP_CALL + '-full-return.json')).exists()
    assert all((unknown.root / path).read_bytes() == raw for path, raw in unknown_evidence_bytes.items())
    unknown_state = k.validate(unknown.ledger, unknown.root)
    assert unknown_state['phase'] == 'unknown' and unknown_state['completed'] == 0
    assert unknown.ledger['direct_science_workers'] and unknown.ledger['direct_science_reservations']
    assert json.loads(unknown_receipts[1]["payload"][-1]) == unknown_state and token_path.read_bytes() == token_raw
    pending = f.case(tmp_path, 'generated-l-uncommitted-unknown')
    pending.admit()
    pending.evidence(3000.0, 124, uncertain=True)
    pending_before = (pending.root / pending.ledger_path).read_bytes()
    receipt = pending.native_commit()
    f.observed_pass(receipt)
    observed = json.loads(receipt.stdout)
    assert observed['committed'] is False and observed['status'] == 'UNCOMMITTED_OWNERSHIP_UNKNOWN'
    assert observed['cost_published'] is False and observed['owner_and_reservation_retained'] is True
    assert (pending.root / pending.ledger_path).read_bytes() == pending_before
    assert not (pending.root / (k.BASE + '/parent-' + k.BOOTSTRAP_CALL + '-full-return.json')).exists()
    refused = f.case(tmp_path, "generated-l-no-create")
    refused.admit()
    outer = refused.evidence(1.0, 1, no_create=True)
    assert k.closure(outer) is True and outer["stdout_eof"] is False and outer["stderr_eof"] is False
    outer["stdout_eof"] = True
    f.negative(lambda: k.closure(outer))
    no_create_before = (refused.root / refused.ledger_path).read_bytes()
    f.negative(lambda: refused.python_charge(1.0, True, no_create=True))
    assert (refused.root / refused.ledger_path).read_bytes() == no_create_before
    assert not (refused.root / (k.BASE + '/parent-' + k.BOOTSTRAP_CALL + '-full-return.json')).exists()
    refused.evidence(1.0, 1, no_create=True)
    paid_refusal = refused.native_commit()
    f.observed_pass(paid_refusal)
    refused_live = k.data(r.text_spec(refused.ledger_path, refused.root, 524288), refused.root, 524288)
    assert refused_live['debits'][-1]['actual_outer_full_return_s'] == 1.0
    assert refused_live['direct_science_blocking_holds'] == {k.BOOTSTRAP_CALL: k.FAILURE_HOLD}
    assert k.validate(refused_live, refused.root)['failed'] is True
    wrong = copy.deepcopy(live)
    wrong["startup_ray_original_grants"] = {}
    f.negative(lambda: k.validate(wrong, fixture.root))
