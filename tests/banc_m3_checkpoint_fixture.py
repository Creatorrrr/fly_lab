"""Generated temporary-root operands. Synthetic costs never qualify the real ledger."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

from tools import banc_m3_checkpoint_runtime as k
from tools import banc_m3_conserved_resource as r
from tools import banc_m3_contact_learning_admission as admission

_HISTORY_BASE = 'verification/banc-natural-gait-m3-20260919'
L_BEFORE = {"path": r.BASE + "/parent-l-checkpoint-harness-design-inputs-20261008.json", "bytes": 16537,
            "sha256": "f5589e98fc56c4990b8d151cdf6607b6e36eba7a8e54af4aab6d01cdfd21c7ba"}
L2_BEFORE = {"path": r.BASE + "/parent-l2-root-residual-inputs-20261008.json", "bytes": 13942,
             "sha256": "235e26b33a45d58ed6ff08e31f2d214e064dab19e45f5c551ad7d98b26e44e6d"}

L3_BEFORE = {"bytes":3210,"sha256":"bffa1740d71b24fea45d3bfebc71324e0cfdabc3ef924cd8b5edcc51677944b7","path":"verification/banc-natural-gait-m3-20260919/l3-before.json"}

L4_BEFORE = {"path":"verification/banc-natural-gait-m3-20260919/l4-before.json","bytes":5028,"sha256":"b3bf9fb7c089c8358b392857f538e6714e4a6d4a75a19bd045a07cf0dc2c82df"}

K_ACCOUNT = {"path": r.BASE + "/parent-contact-checkpoint-harness-k-account-source-20261008.txt", "bytes": 242201,
             "sha256": "2d195eac16397d030596ae4f2df9121b1610fb54f976519bfa4b51d65494a43c"}
K_CONTACT = {"path": r.BASE + "/parent-contact-checkpoint-harness-k-contact-source-20261008.txt", "bytes": 122720,
             "sha256": "57399f904a1ccf9e8ed7829dcd70a53ae22ec77501f3f18aca9401fd950f39f7"}


def l4_before_row(pin, mapping):
    """Historical operands only. Current executable Source never falls back."""
    r.exact(mapping["schema"], "flylab.m3.l4-exact-before.v1", "exact historical map schema")
    rows = mapping["inputs"]
    paths = (
        "tools/banc_m3_checkpoint_harness_runtime.py", "tools/banc_m3_checkpoint_harness_runtime_native.ps1",
        "tests/banc_m3_checkpoint_fixture.py", "tests/banc_m3_checkpoint_harness_fixture.py",
        "tests/test_c_banc_checkpoint_harness_runtime.py", "tests/test_c_banc_contact_runtime_journal.py",
        "tools/banc_m3_conserved_resource.py", r.BASE + "/parent-direct-science-account.ps1",
        "tools/banc_m3_checkpoint_runtime.py", "tools/banc_m3_contact_learning_admission.py",
        "tools/banc_m3_contact_runtime_journal.py", "tools/banc_m3_startup_ray_supervisor.py",
        "tools/verify_c_banc_m3_contact_learning.py",
    )
    r.need(type(rows) is list and len(rows) == len(paths), "all13 immutable before operands")
    selected = None
    for index, row in enumerate(rows):
        r.exact(set(row), {"ordinal", "source", "archive", "cap"}, "whole archive row")
        r.need(type(row["ordinal"]) is int and row["ordinal"] == index, "typed literal ordinal")
        cap = 65536 if index < 2 else 262144
        r.need(type(row["cap"]) is int and row["cap"] == cap, "fixed selected actual cap")
        source, archive = row["source"], row["archive"]
        for operand in (source, archive):
            r.exact(set(operand), {"path", "bytes", "sha256"}, "whole exact historical pin")
            r.need(type(operand["bytes"]) is int and 0 < operand["bytes"] <= cap, "typed bounded before bytes")
            digest = operand["sha256"]
            r.need(type(digest) is str and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                   "typed whole before SHA")
        r.exact(source["path"], paths[index], "literal historical Source namespace")
        r.exact(archive["path"], r.BASE + f"/l4-a/{index:02d}.txt", "literal read-only archive path")
        r.exact(archive["bytes"], source["bytes"], "whole source/archive bytes")
        r.exact(archive["sha256"], source["sha256"], "whole source/archive SHA")
        if source == pin:
            r.exact(source, pin, "whole requested historical tuple")
            r.need(selected is None, "ambiguous historical operand refused")
            selected = row
    return selected


def s28_before_row(pin, mapping, current):
    current.keys(mapping, ("schema","pairs","before_review","before_source","before_ledger","project_execution"), "S28 exact index")
    r.exact(mapping["schema"], "flylab.m3.s28.ROOT-before11.v1", "S28 schema")
    r.exact(mapping["before_source"], {'path': _HISTORY_BASE+'/s27-s1.json', 'sha256': '35b6cf63018f61f3fde34d47045d16f5e3c97dc53bac64b1fe7814aed0aa61c6', 'bytes': 262139}, "executed Source basis")
    r.exact(mapping["before_ledger"], {'path': _HISTORY_BASE+'/s27-bootstrap-settled318-HOLD-20261009.json', 'sha256': '70175b4bc56c19412b64500343ebc72bc538a7b3776dd60733d43efeeb59476e', 'bytes': 495417}, "whole318 basis")
    r.exact(mapping["before_review"], {'path': _HISTORY_BASE+'/s27-r1.json', 'sha256': '562162a4a496884bbd7a2f489a1c4875e1018d54a41e49625dc3c413a19681ce', 'bytes': 247286}, "executed review basis")
    r.exact(mapping["project_execution"], "NOT_RUN_SOURCE_ONLY", "immutable index source only")
    rows = mapping["pairs"]
    r.need(type(rows) is list and len(rows) == 11, "all11 before/archive pairs")
    selected = None
    for index, row in enumerate(rows):
        current.keys(row, ("cap","before","archive"), "S28 pair")
        cap = 131072 if index < 2 else 393216 if index == 7 else 262144
        r.need(type(row["cap"]) is int and row["cap"] == cap, "exact selected archive cap")
        for operand in (row["before"], row["archive"]):
            current.metadata_pin(operand, operand["path"])
            r.need(operand["bytes"] <= cap, "whole bounded before pin")
        r.exact(row["before"]["path"], current.REPLACEMENTS[index], "all11 exact order/path")
        r.exact(row["archive"], dict(row["before"], path=current.BASE+"/s28-before/"+row["before"]["path"]), "whole exact S28 archive pair")
        r.exact(current.reader_cap(row["archive"]["path"]), cap, "same exact archive reader cap")
        if pin == row["before"] or pin == row["archive"]:
            r.need(selected is None, "ambiguous selected pair refused")
            selected = row
    return selected


def s27_before_row(pin, mapping, current):
    current.keys(mapping, ("schema","pairs","before_review","before_source","before_ledger","project_execution"), "S27 exact index")
    r.exact(mapping["schema"], "flylab.m3.s27.ROOT-before11.v1", "S27 schema")
    r.exact(mapping["before_source"], {'bytes': 262139, 'sha256': 'a6c5a19bd4d61b87b8ba5c31774369eaab12cd55d63a2d7dd7ba15e2a4aecc27', 'path': _HISTORY_BASE+'/s26-s1.json'}, "executed Source basis")
    r.exact(mapping["before_ledger"], {'bytes': 494291, 'sha256': '994ba49395dd242db5d7ae44890e1df6bd3f88a6b6401f6dc605b4f6d85e5ef1', 'path': _HISTORY_BASE+'/s25-bootstrap-settled317-HOLD-20261009.json'}, "whole317 basis")
    r.exact(mapping["before_review"], {'bytes': 247806, 'sha256': '89dfe65c1d166fd7259325af504181257a28ee78bd354ef645052b89a96ac48f', 'path': _HISTORY_BASE+'/s26-r1.json'}, "executed review basis")
    r.exact(mapping["project_execution"], "NOT_RUN_SOURCE_ONLY", "immutable index source only")
    rows = mapping["pairs"]
    r.need(type(rows) is list and len(rows) == 11, "all11 before/archive pairs")
    selected = None
    for index, row in enumerate(rows):
        current.keys(row, ("cap","before","archive"), "S27 pair")
        cap = 131072 if index < 2 else 393216 if index == 7 else 262144
        r.need(type(row["cap"]) is int and row["cap"] == cap, "exact selected archive cap")
        for operand in (row["before"], row["archive"]):
            current.metadata_pin(operand, operand["path"])
            r.need(operand["bytes"] <= cap, "whole bounded before pin")
        r.exact(row["before"]["path"], current.REPLACEMENTS[index], "all11 exact order/path")
        r.exact(row["archive"], dict(row["before"], path=current.BASE+"/s27-before/"+row["before"]["path"]), "whole exact S27 archive pair")
        r.exact(current.reader_cap(row["archive"]["path"]), cap, "same exact archive reader cap")
        if pin == row["before"] or pin == row["archive"]:
            r.need(selected is None, "ambiguous selected pair refused")
            selected = row
    return selected


def s26_before_row(pin, mapping, current):
    current.keys(mapping, ("schema","pairs","before_review","before_source","before_ledger","project_execution"), "S26 exact index")
    r.exact(mapping["schema"], "flylab.m3.s26.ROOT-before11.v1", "S26 schema")
    r.exact(mapping["before_source"], {'bytes': 262138, 'sha256': 'e42e64997a039fa7ae86daa851929682d98b8843255d92449cc4cbfc5b5735e1', 'path': _HISTORY_BASE+'/s25-s1.json'}, "executed Source basis")
    r.exact(mapping["before_ledger"], {'bytes': 494291, 'sha256': '994ba49395dd242db5d7ae44890e1df6bd3f88a6b6401f6dc605b4f6d85e5ef1', 'path': _HISTORY_BASE+'/s25-bootstrap-settled317-HOLD-20261009.json'}, "whole317 basis")
    r.exact(mapping["before_review"], {'bytes': 247285, 'sha256': '8b112721a2815dcb10ec3c84275590821cfe1f7961e1cafd19083849e5419963', 'path': _HISTORY_BASE+'/s25-r1.json'}, "executed review basis")
    r.exact(mapping["project_execution"], "NOT_RUN_SOURCE_ONLY", "immutable index source only")
    rows = mapping["pairs"]
    r.need(type(rows) is list and len(rows) == 11, "all11 before/archive pairs")
    selected = None
    for index, row in enumerate(rows):
        current.keys(row, ("cap","before","archive"), "S26 pair")
        cap = 131072 if index < 2 else 393216 if index == 7 else 262144
        r.need(type(row["cap"]) is int and row["cap"] == cap, "exact selected archive cap")
        for operand in (row["before"], row["archive"]):
            current.metadata_pin(operand, operand["path"])
            r.need(operand["bytes"] <= cap, "whole bounded before pin")
        r.exact(row["before"]["path"], current.REPLACEMENTS[index], "all11 exact order/path")
        r.exact(row["archive"], dict(row["before"], path=current.BASE+"/s26-before/"+row["before"]["path"]), "whole exact S26 archive pair")
        r.exact(current.reader_cap(row["archive"]["path"]), cap, "same exact archive reader cap")
        if pin == row["before"] or pin == row["archive"]:
            r.need(selected is None, "ambiguous selected pair refused")
            selected = row
    return selected


def s25_before_row(pin, mapping, current):
    current.keys(mapping, ("schema","pairs","before_review","before_source","before_ledger","project_execution"), "S25 exact index")
    r.exact(mapping["schema"], "flylab.m3.s25.ROOT-before11.v1", "S25 schema")
    r.exact(mapping["before_source"], {'path': _HISTORY_BASE+'/s24-s1.json', 'bytes': 262138, 'sha256': 'fc69b16aa9f5556fefdb8d7e4937af5805873fd99378dbd193e6e7f5f5d17558'}, "executed Source basis")
    r.exact(mapping["before_ledger"], {'path': _HISTORY_BASE+'/s24-bootstrap-settled316-HOLD-20261009.json', 'bytes': 493186, 'sha256': '2a3b07e6a8efb7f5958ee72f5b89e2db8ea14d993a518f6125738e0fbccbc20a'}, "whole316 basis")
    r.exact(mapping["before_review"], {'path': _HISTORY_BASE+'/s24-r1.json', 'bytes': 247400, 'sha256': '7b2bca3ec1e791a2505f84775a9115d35d1d0d226cfc2cd511d7c47e3b9cd940'}, "executed review basis")
    r.exact(mapping["project_execution"], "NOT_RUN", "immutable index source only")
    rows = mapping["pairs"]
    r.need(type(rows) is list and len(rows) == 11, "all11 before/archive pairs")
    selected = None
    for index, row in enumerate(rows):
        current.keys(row, ("cap","before","archive"), "S25 pair")
        cap = 131072 if index < 2 else 393216 if index == 7 else 262144
        r.need(type(row["cap"]) is int and row["cap"] == cap, "exact selected archive cap")
        for operand in (row["before"], row["archive"]):
            current.metadata_pin(operand, operand["path"])
            r.need(operand["bytes"] <= cap, "whole bounded before pin")
        r.exact(row["before"]["path"], current.REPLACEMENTS[index], "all11 exact order/path")
        r.exact(row["archive"], dict(row["before"], path=current.BASE+"/s25-before/"+row["before"]["path"]), "whole exact S25 archive pair")
        r.exact(current.reader_cap(row["archive"]["path"]), cap, "same exact archive reader cap")
        if pin == row["before"] or pin == row["archive"]:
            r.need(selected is None, "ambiguous selected pair refused")
            selected = row
    return selected


def s24_before_row(pin, mapping, current):
    current.keys(mapping, ("schema","pairs","before_review","before_source","before_ledger","project_execution"), "S24 exact index")
    r.exact(mapping["schema"], "flylab.m3.s24.ROOT-before11.v1", "S24 schema")
    r.exact(mapping["before_source"], {'sha256': '939e7797d6795ed9041f445742e60c98228585e60bc7a9e6fca0d574196684f8', 'bytes': 262138, 'path': _HISTORY_BASE+'/s23-s1.json'}, "executed Source basis")
    r.exact(mapping["before_ledger"], {'sha256': '94adec2cd28a3486e06b8efcb9ff54b152fe1199993ba378f10edec033b49a1a', 'bytes': 492062, 'path': _HISTORY_BASE+'/s23-bootstrap-settled315-HOLD-20261009.json'}, "whole315 basis")
    r.exact(mapping["before_review"], {'path': _HISTORY_BASE+'/s23-r1.json', 'bytes': 247742, 'sha256': '32b96e0bd22eac0e360cf797f291f10d1e0ad062af789ee73323d614baa8480c'}, "executed review basis")
    r.exact(mapping["project_execution"], "NOT_RUN", "immutable index source only")
    rows = mapping["pairs"]
    r.need(type(rows) is list and len(rows) == 11, "all11 before/archive pairs")
    selected = None
    for index, row in enumerate(rows):
        current.keys(row, ("cap","before","archive"), "S24 pair")
        cap = 131072 if index < 2 else 393216 if index == 7 else 262144
        r.need(type(row["cap"]) is int and row["cap"] == cap, "exact selected archive cap")
        for operand in (row["before"], row["archive"]):
            current.metadata_pin(operand, operand["path"])
            r.need(operand["bytes"] <= cap, "whole bounded before pin")
        r.exact(row["before"]["path"], current.REPLACEMENTS[index], "all11 exact order/path")
        r.exact(row["archive"], dict(row["before"], path=current.BASE+"/s24-before/"+row["before"]["path"]), "whole exact S24 archive pair")
        r.exact(current.reader_cap(row["archive"]["path"]), cap, "same exact archive reader cap")
        if pin == row["before"] or pin == row["archive"]:
            r.need(selected is None, "ambiguous selected pair refused")
            selected = row
    return selected


D15B_METADATA_PINS = [{'path': _HISTORY_BASE+'/s28-f15b-root/terminal319-before.json', 'bytes': 394685, 'sha256': 'c8a74a40a0d2bfef0e630524508d819c2b106cbce06f6f2b93364f5e4b4b1482'}, {'path': _HISTORY_BASE+'/s28-f15-root/admitted318-before.json', 'bytes': 496266, 'sha256': 'ea814b9271554155b9f71673ac4d8c5a4e2a1fcb5a3cc5a20cdf57dd59cb405d'}, {'path': _HISTORY_BASE+'/s28-f15-root/settlement-pre-backup.json', 'bytes': 496266, 'sha256': 'ea814b9271554155b9f71673ac4d8c5a4e2a1fcb5a3cc5a20cdf57dd59cb405d'}, {'path': _HISTORY_BASE+'/s28-f15-root/settlement-replace-backup.json', 'bytes': 496266, 'sha256': 'ea814b9271554155b9f71673ac4d8c5a4e2a1fcb5a3cc5a20cdf57dd59cb405d'}, {'path': _HISTORY_BASE+'/s28-exec-precreate-cost.json', 'bytes': 2452, 'sha256': '359c0dd97f2baf9f063d9a26c72fcd5269c3a66665be7a4004fdf8cf2278abb4'}, {'path': _HISTORY_BASE+'/s28-exec-precreate-settlement.json', 'bytes': 2916, 'sha256': '79c2967af54a4a28d7dd580c98e068396f1141b61043361c64befaf72190df31'}, {'path': _HISTORY_BASE+'/s28-f15-root/settlement-owner-token.json', 'bytes': 212, 'sha256': 'c8ef54795ae8bc5fe801478578436c66cbc5b11c9b6995bbc5ad10a88236bf02'}, {'path': _HISTORY_BASE+'/s28-f15-root/actual-settlement-audit.json', 'bytes': 2459, 'sha256': '8cb8eeb7decead2e8db9dd4195e571a667d5b6457ff72165721e9d8fdb8b4401'}, {'path': _HISTORY_BASE+'/s28-f15-root/settlement-actual-tool-capture.json', 'bytes': 13865, 'sha256': '6c02d1939b022ced5f7b45f71712d1d97cae4bef54dc4284f93f58c8827f6f63'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/full4-failure.json', 'bytes': 6976, 'sha256': '978021662924b9a1c15bfac7cc5699a58535d43eececd894d58b542d7395ff0c'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/full4-actual-tool-capture.json', 'bytes': 6464, 'sha256': '72ccd3b9621fe0979c3d51737fc8d3f6c818548ed017e09165ee7502b307e247'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/full4-request.json', 'bytes': 1371, 'sha256': 'bdd63628db50c52b0a6ebf9f0f769848d79bcd241558397a55084b25dbdec90b'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f15b-root/before/verification/banc-natural-gait-m3-20260919/c5.json', 'bytes': 124922, 'sha256': '9d65539915dbe98563ae5599df9d91c422541411f51608bc5ebe14fab8655ed8'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f15b-root/before/verification/banc-natural-gait-m3-20260919/s28-s1.json', 'bytes': 262142, 'sha256': 'b8f00cb6fe06b0fa611b10b62d60f39d5a8e130887767f9e6c791b0973e35bbc'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f15b-root/before/verification/banc-natural-gait-m3-20260919/s28-r1.json', 'bytes': 246958, 'sha256': '06479a5a995aa0d0eb7a3b2a7d062506005ae5e2cfe6d2b37a74c9b27f5bca97'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f15b-root/before/verification/banc-natural-gait-m3-20260919/s28-recovery12.json', 'bytes': 1869, 'sha256': '26d3727cb1ade441017cc5dfe98a474a740aec206e55dbb5aa0adedb8d0b9add'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-recovery10.json', 'bytes': 1903, 'sha256': '27f8b420a0f7cdb3447aac416cd2ac71cf499ee8b5ea719ff05c968fd6f1184e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/s1-before.json', 'bytes': 262140, 'sha256': '39b4910b21c51042a5d68f3da51061ce2d57ead51ec5ff4cdcf82554a6493ef0'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f12-root/a318.json', 'bytes': 495558, 'sha256': '1221bf613896fd8f929b0dbfd93c1e4bc8e922f4c873e3869d5a7780d53c6d54'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/a318.json', 'bytes': 495559, 'sha256': 'fadfb1604441f5c2cc481e15b07eeaf10a7a7529cedc8969bc496196ea221274'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f12-root/actual-tool-return.json', 'bytes': 1846, 'sha256': '71341a5233ecc031b45876ab24d1f7e4305ed79c738aa19f3efa71dc6c250162'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/c5-before.json', 'bytes': 74674, 'sha256': '3a87bc0f34b8a36ddd5153940f2b17fccbdd990b32459aea28ae92e51be58b0a'}, {'path': 'verification/banc-natural-gait-m3-20260919/h28.json', 'bytes': 261912, 'sha256': 'e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f12-root/actual-layout-compatibility-diagnostic.json', 'bytes': 1331, 'sha256': 'efc4694c9651f842832549809e1613b9900ab71004da098bb82dc24c29ee8a6d'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-catalog-leaves-300021e7370a4dc8b163ef22f0245f31.jsonl', 'bytes': 6393, 'sha256': 'c4bb7e972e5ed5950e554983344f5d34849008f9a2a9a984a874d1ff8795926e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f12-root/fast-counter-expectation.json', 'bytes': 2331, 'sha256': '99e971e5a752697e8ab86e9fbf461acd4052d1d27da3a5437f05db14fe050b5d'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-recovery9.json', 'bytes': 1902, 'sha256': 'a4cf49fb579aeb3114fd800f092080b5e6a54b8f71ae835121e8cd34a1d4a28d'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f12-root/s1-before.json', 'bytes': 262140, 'sha256': '1cf30bee866a147de013d0023a2ea02d8734005178d145d7f06952f058ef4aba'}, {'path': 'verification/banc-natural-gait-m3-20260919/s27-bootstrap-settled318-HOLD-20261009.json', 'sha256': '70175b4bc56c19412b64500343ebc72bc538a7b3776dd60733d43efeeb59476e', 'bytes': 495417}, {'path': 'verification/banc-natural-gait-m3-20260919/e28.json', 'bytes': 2443, 'sha256': 'e5e284cbaa1f67f1dd6e14bdf3fe74c5f902d751c387f6850700323c23609c93'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f12-root/h28-before.json', 'bytes': 261912, 'sha256': 'e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s27-actual.json', 'bytes': 3365, 'sha256': 'fcf5b1fa42a8348fa52f85e0259cc3b3554be5d96206fd5e77f4fcc54e01eb94'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f11-root/actual-fast-warm-gate.json', 'bytes': 4977, 'sha256': '9bbf1418212eedfb12b4aaa57cac0d613c9b900b0f9e2a0b4a4d7dfa1f63c0ce'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-recovery11.json', 'bytes': 1882, 'sha256': '0098806a84d0c18cf606482710d15e73dbe21defae360badba88a6c909db4c26'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/s1-before.json', 'bytes': 262140, 'sha256': 'f1b802af060e385f03955b5cbba5830cc9ee42ccf757b9800016a45afd7cabfd'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/actual-tool-return.json', 'bytes': 2798, 'sha256': 'e76c924ab89dd27752d2f8a1afe68bd6839f363471886c073b3b14fce0d5d2c0'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-catalog-leaves-30944ee8fe084e2aac23de9d47e4de80.jsonl', 'bytes': 496, 'sha256': 'b3ebd169ebfc3f700ec42a0c3b7b9292f35b7561cb6f7d100c71bf3835669606'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/c5-before.json', 'bytes': 97623, 'sha256': '9690bd2ec7a65d33432071d09d9050c6d949c92d6a20852fc92072737f7dcb41'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/actual-compiler-type-refusal.json', 'bytes': 1968, 'sha256': '448f06a882edd1757895f90dfd4bd1c7bfad38f414fcdda3909a0eacfa34e407'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/a318.json', 'bytes': 495559, 'sha256': 'fadfb1604441f5c2cc481e15b07eeaf10a7a7529cedc8969bc496196ea221274'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/h28-before.json', 'bytes': 261912, 'sha256': 'e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f12-root/actual-fast-warm-gate.json', 'bytes': 5074, 'sha256': '9c9348756e9dacf4ee53a320e50acc22d25c8fe9028419dccbc35819356d7b7e'}, {'bytes': 495417, 'sha256': '70175b4bc56c19412b64500343ebc72bc538a7b3776dd60733d43efeeb59476e', 'path': 'verification/banc-natural-gait-m3-20260919/s28-anchor318-before-recovery.json'}, {'path': 'verification/banc-natural-gait-m3-20260919/s27-s1.json', 'sha256': '35b6cf63018f61f3fde34d47045d16f5e3c97dc53bac64b1fe7814aed0aa61c6', 'bytes': 262139}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/h28-before.json', 'bytes': 261912, 'sha256': 'e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/actual-fast-warm-gate.json', 'bytes': 2818, 'sha256': '042051f83aaa9c347c72d943e76d0958cf22afeccb5fd3c929bf0241cd5f5a1f'}, {'path': 'docs/M3_28D15C.md', 'bytes': 10675, 'sha256': '1c1337578a8982b240bf743bb31ad421fde9cd71b54c7e5650b32295a7d43055'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-d15c-astra/review.md', 'bytes': 16179, 'sha256': '79b900d090ca7fe61f0322a3dff5099ebbdf50444d91cd113c6e709a3dd21290'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-d15c-astra/review.json', 'bytes': 14885, 'sha256': '69c41861683acb00b7f784fc2d41601c9d8dbe1611345e9337c672c1ca0895d3'}, {'path': 'docs/M3_28R15C.md', 'bytes': 9488, 'sha256': '578b086dfe080b3119ce75404abdcb7143d715a9251d60396ad6fe332bf27a33'}]

def exact_source_operand(pin):
    if pin["path"] == 'verification/banc-natural-gait-m3-20260919/s28-f18-root/before/verification/banc-natural-gait-m3-20260919/s28-s1.json':
        from tools import banc_m3_checkpoint_harness_runtime as current
        r.exact(pin, current.CONTINUATION_BEFORE_SOURCE, "whole compiled predecessor fixture pin")
        contract = current.data(current.CURRENT_CONTRACT, admission.ROOT)
        r.exact(contract["actual320"]["adoption_before_Source"], pin, "current authenticated predecessor before staging")
        return r.raw_pin(pin, admission.ROOT, 262144)
    if pin["path"] in ('docs/M3_28D18.md', 'verification/banc-natural-gait-m3-20260919/s28-d18-astra/review.md', 'verification/banc-natural-gait-m3-20260919/s28-d18-astra/review.json', 'docs/M3_28R18.md'):
        from tools import banc_m3_checkpoint_harness_runtime as current
        contract = current.data(current.CURRENT_CONTRACT, admission.ROOT)
        declared = tuple(contract["provenance"][field] for field in ("root_design", "astra_markdown", "astra_json", "root_reflection"))
        current.metadata_pin(pin, pin["path"])
        r.need(pin in declared, "only compiled current D18 direct provenance")
        r.need(pin["bytes"] <= 262144, "bounded current D18 direct provenance")
        return r.raw_pin(pin, admission.ROOT, 262144)
    if pin["path"] in ('verification/banc-natural-gait-m3-20260919/s28-f18-root/FIRST.md', 'verification/banc-natural-gait-m3-20260919/s28-f18-root/SECOND.md', 'verification/banc-natural-gait-m3-20260919/s28-f18-root/residual03/final-v2.md', 'verification/banc-natural-gait-m3-20260919/s28-f18-root/SECOND.json', 'verification/banc-natural-gait-m3-20260919/s28-f18-root/residual03/final-v2.json', 'verification/banc-natural-gait-m3-20260919/s28-f18-sol/FIRST.json', 'verification/banc-natural-gait-m3-20260919/s28-f18-sol/sole-R1/R1.json'):
        from tools import banc_m3_checkpoint_harness_runtime as current
        freeze = current.data(r.text_spec(current.SOURCE_PATH, admission.ROOT, 262144), admission.ROOT)
        r.exact(freeze["checkpoint_contract"], current.CURRENT_CONTRACT, "compiled current contract before report staging")
        current.metadata_pin(freeze["reviewed_source"], current.REVIEW_PATH)
        review = current.data(freeze["reviewed_source"], admission.ROOT)
        r.exact(review["sources"][current.CONTRACT_PATH], current.CURRENT_CONTRACT, "current reviewed contract before report staging")
        r.exact((freeze["ROOT_review_cycle_complete"], freeze["qualification_epoch"], freeze["sources"]), (True, current.EPOCH, review["sources"]), "complete current Source/review before report staging")
        r.exact(freeze["reviewed_source"], r.text_spec(current.REVIEW_PATH, admission.ROOT, 262144), "actual current review before report staging")
        current.metadata_pin(pin, pin["path"])
        r.need(pin in (*review["root_reviews"], *review["sol_reports"]), "only actually declared current D18 review/report")
        r.need(pin["bytes"] <= 262144, "bounded current D18 report")
        return r.raw_pin(pin, admission.ROOT, 262144)
    if pin["path"] in ('verification/banc-natural-gait-m3-20260919/s28-f15c-root/FIRST.md', 'verification/banc-natural-gait-m3-20260919/s28-f15c-root/SECOND.md', 'verification/banc-natural-gait-m3-20260919/s28-f15c-root/final.md', 'verification/banc-natural-gait-m3-20260919/s28-f15c-root/SECOND.json', 'verification/banc-natural-gait-m3-20260919/s28-f15c-root/final.json', 'verification/banc-natural-gait-m3-20260919/s28-f15c-sol/FIRST.json', 'verification/banc-natural-gait-m3-20260919/s28-f15c-sol/R1.json'):
        return r.raw_pin(pin,admission.ROOT,262144)
    if pin in D15B_METADATA_PINS:
        return r.raw_pin(pin,admission.ROOT,524288 if pin['bytes']>262144 else 262144)
    if pin["path"] in ('docs/M3_28D15C.md', 'verification/banc-natural-gait-m3-20260919/s28-d15c-astra/review.md', 'verification/banc-natural-gait-m3-20260919/s28-d15c-astra/review.json', 'docs/M3_28R15C.md'):
        return r.raw_pin(pin,admission.ROOT,262144)
    if pin in [{'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/a318.json', 'bytes': 495559, 'sha256': 'fadfb1604441f5c2cc481e15b07eeaf10a7a7529cedc8969bc496196ea221274'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-recovery10.json', 'bytes': 1903, 'sha256': '27f8b420a0f7cdb3447aac416cd2ac71cf499ee8b5ea719ff05c968fd6f1184e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/s1-before.json', 'bytes': 262140, 'sha256': 'f1b802af060e385f03955b5cbba5830cc9ee42ccf757b9800016a45afd7cabfd'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/c5-before.json', 'bytes': 97623, 'sha256': '9690bd2ec7a65d33432071d09d9050c6d949c92d6a20852fc92072737f7dcb41'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/h28-before.json', 'bytes': 261912, 'sha256': 'e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/actual-fast-warm-gate.json', 'bytes': 2818, 'sha256': '042051f83aaa9c347c72d943e76d0958cf22afeccb5fd3c929bf0241cd5f5a1f'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/s1-before.json', 'bytes': 262140, 'sha256': '39b4910b21c51042a5d68f3da51061ce2d57ead51ec5ff4cdcf82554a6493ef0'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/c5-before.json', 'bytes': 74674, 'sha256': '3a87bc0f34b8a36ddd5153940f2b17fccbdd990b32459aea28ae92e51be58b0a'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f12-root/actual-fast-warm-gate.json', 'bytes': 5074, 'sha256': '9c9348756e9dacf4ee53a320e50acc22d25c8fe9028419dccbc35819356d7b7e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f12-root/actual-tool-return.json', 'bytes': 1846, 'sha256': '71341a5233ecc031b45876ab24d1f7e4305ed79c738aa19f3efa71dc6c250162'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f12-root/actual-layout-compatibility-diagnostic.json', 'bytes': 1331, 'sha256': 'efc4694c9651f842832549809e1613b9900ab71004da098bb82dc24c29ee8a6d'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-catalog-leaves-300021e7370a4dc8b163ef22f0245f31.jsonl', 'bytes': 6393, 'sha256': 'c4bb7e972e5ed5950e554983344f5d34849008f9a2a9a984a874d1ff8795926e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/actual-tool-return.json', 'bytes': 2798, 'sha256': 'e76c924ab89dd27752d2f8a1afe68bd6839f363471886c073b3b14fce0d5d2c0'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/actual-compiler-type-refusal.json', 'bytes': 1968, 'sha256': '448f06a882edd1757895f90dfd4bd1c7bfad38f414fcdda3909a0eacfa34e407'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-catalog-leaves-30944ee8fe084e2aac23de9d47e4de80.jsonl', 'bytes': 496, 'sha256': 'b3ebd169ebfc3f700ec42a0c3b7b9292f35b7561cb6f7d100c71bf3835669606'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-recovery11.json', 'bytes': 1882, 'sha256': '0098806a84d0c18cf606482710d15e73dbe21defae360badba88a6c909db4c26'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/a318.json', 'bytes': 495559, 'sha256': 'fadfb1604441f5c2cc481e15b07eeaf10a7a7529cedc8969bc496196ea221274'}, {'path': 'verification/banc-natural-gait-m3-20260919/s27-bootstrap-settled318-HOLD-20261009.json', 'sha256': '70175b4bc56c19412b64500343ebc72bc538a7b3776dd60733d43efeeb59476e', 'bytes': 495417}, {'path': 'verification/banc-natural-gait-m3-20260919/e28.json', 'bytes': 2443, 'sha256': 'e5e284cbaa1f67f1dd6e14bdf3fe74c5f902d751c387f6850700323c23609c93'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/h28-before.json', 'bytes': 261912, 'sha256': 'e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s27-actual.json', 'bytes': 3365, 'sha256': 'fcf5b1fa42a8348fa52f85e0259cc3b3554be5d96206fd5e77f4fcc54e01eb94'}]:
        return r.raw_pin(pin,admission.ROOT,524288 if pin["bytes"]>262144 else 262144)
    """Current exact pin or immutable historical tuple; current fences do not fall back."""
    path = pin["path"]
    from tools import banc_m3_checkpoint_harness_runtime as current
    if path in current.REPLACEMENTS or path.startswith((current.BASE+"/s28-before/",current.BASE+"/s27-before/",current.BASE+"/s26-before/",current.BASE+"/s25-before/",current.BASE+"/s24-before/")):
        freeze = r.read_pin(r.text_spec(current.SOURCE_PATH, admission.ROOT, 262144), admission.ROOT)
        if path in current.REPLACEMENTS and pin == freeze["sources"][path]:
            # Current missing/drift is an actual refusal, outside every historical catch.
            return r.raw_pin(pin, admission.ROOT, current.reader_cap(path))
        selected = s28_before_row(pin, r.read_pin({'path': 'verification/banc-natural-gait-m3-20260919/s28-before-index.json', 'bytes': 4915, 'sha256': '7c9ca2a35817b59ab1f91bbf699f48e1ff40a8394044b2c9d911b1e305255bd5'}, admission.ROOT), current)
        if selected is not None:
            return r.raw_pin(selected["archive"], admission.ROOT, selected["cap"])
        r.need(not path.startswith(current.BASE+"/s28-before/"), "unknown S28 archive tuple refused")
        selected = s27_before_row(pin, r.read_pin({'path': 'verification/banc-natural-gait-m3-20260919/s27-before-index.json', 'bytes': 4915, 'sha256': 'f0c88cff5f55698a7325134bacea2cf89507617ca47e7ece911bd785f611346c'}, admission.ROOT), current)
        if selected is not None:
            return r.raw_pin(selected["archive"], admission.ROOT, selected["cap"])
        r.need(not path.startswith(current.BASE+"/s27-before/"), "unknown S27 archive tuple refused")
        selected = s26_before_row(pin, r.read_pin({'path': 'verification/banc-natural-gait-m3-20260919/s26-before-index.json', 'bytes': 4913, 'sha256': 'd31419aab4a3afe9eeefae40190e639e3f2dcf1895d1264831ab1c8ad8efd502'}, admission.ROOT), current)
        if selected is not None:
            return r.raw_pin(selected["archive"], admission.ROOT, selected["cap"])
        r.need(not path.startswith(current.BASE+"/s26-before/"), "unknown S26 archive tuple refused")
        selected = s25_before_row(pin, r.read_pin({'path': 'verification/banc-natural-gait-m3-20260919/s25-before-index.json', 'bytes': 4901, 'sha256': 'f9c9e4326750fcd02c6de81efdf9ea233883bb90786652f7c44b2e9da5bd88a8'}, admission.ROOT), current)
        if selected is not None:
            return r.raw_pin(selected["archive"], admission.ROOT, selected["cap"])
        r.need(not path.startswith(current.BASE+"/s25-before/"), "unknown S25 archive tuple refused")
        selected = s24_before_row(pin, r.read_pin({'path': 'verification/banc-natural-gait-m3-20260919/s24-before-index.json', 'bytes': 4901, 'sha256': '494e1646a6f567fd67e4a98122ed7d1bfd4c0104f4176dee1c681143e25adcc0'}, admission.ROOT), current)
        if selected is not None:
            # Known current-before miss/drift is an actual failure, outside historical catch.
            return r.raw_pin(selected["archive"], admission.ROOT, selected["cap"])
        r.need(not path.startswith(current.BASE+"/s24-before/"), "unknown S24 archive tuple refused")
        mapping = r.read_pin({'path': 'verification/banc-natural-gait-m3-20260919/s23-before-index.json', 'bytes': 5499, 'sha256': '2fdc50e275d2a4fededd022ce4ecd6b24c7c496e10766e4ea138fdfa597ae113'}, admission.ROOT)
        r.need(len(mapping["inputs"]) == 11, "whole immutable S23 before11")
        for index, row in enumerate(mapping["inputs"]):
            r.exact(row["source"]["path"], current.REPLACEMENTS[index], "exact before11 order/path")
            cap = 65536 if index < 2 else 262144
            r.exact(row["cap"], cap, "unchanged historical before11 cap")
            r.exact(row["archive"], dict(row["source"], path=current.BASE+"/s23-before/"+row["source"]["path"]), "whole exact before11 archive pair")
            if pin == row["source"]:
                return r.raw_pin(row["archive"], admission.ROOT, cap)
    old = r.read_pin({"path": r.BASE + "/parent-contact-learning-source-freeze-20261008-k.json",
                     "bytes": 257763, "sha256": "66ffae7fb088befd1e1155c93db65bb62005ec59853ee3ba17708e854bda1c0f"},
                    admission.ROOT)["sources"]
    cap = 524288 if path in old and old[path] == pin and pin["bytes"] > 262144 else 262144
    if path in (k.ANCHOR["path"], r.BASE + "/parent-k-bootstrap-settled289-HOLD-20261008.json",
                r.BASE + "/parent-l-bootstrap-settled290-HOLD-20261008.json",
                r.BASE + "/parent-l2-bootstrap-settled291-before-allocation-20261008.json",
                r.BASE + "/parent-l2-qualification-settled295-before-body-20261008.json",
                r.BASE + "/parent-l2-body-settled296-before-audit-20261008.json",
                r.BASE + "/parent-l3-bootstrap-settled292-before-allocation-20261008.json",
                r.BASE + "/parent-l3-qualification-settled296-before-body-20261008.json",
                r.BASE + "/parent-l3-body-settled297-before-audit-20261008.json",
                r.BASE + "/parent-l3-lint-settled293-HOLD-20261008.json",
                r.BASE + "/parent-l4-bootstrap-settled294-before-allocation-20261008.json",
                r.BASE + "/parent-l4-bootstrap-settled294-HOLD-20261008.json",
                r.BASE + "/parent-l5-bootstrap-settled295-before-allocation-20261008.json",
                r.BASE + "/parent-l5-qualification-settled299-before-body-20261008.json",
                r.BASE + "/parent-l5-body-settled300-before-audit-20261008.json",
                r.BASE + "/parent-l5-lint-settled296-HOLD-20261008.json",
                r.BASE + "/parent-l6-bootstrap-settled297-before-allocation-20261008.json",
                r.BASE + "/parent-l6-qualification-settled301-before-body-20261008.json",
                r.BASE + "/parent-l6-body-settled302-before-audit-20261008.json",
                r.BASE + "/parent-l6-math-settled299-HOLD-20261008.json",
                r.BASE + "/parent-l7-bootstrap-settled300-before-allocation-20261008.json",
                r.BASE + "/parent-l7-qualification-settled304-before-body-20261008.json",
                r.BASE + "/parent-l7-body-settled305-before-audit-20261008.json",
                r.BASE + "/parent-l7-bootstrap-admitted299-before-cold-settlement-20261008.json",
                r.BASE + "/parent-l7-cold-settled300-HOLD-20261008.json",
                r.BASE + "/parent-l8-bootstrap-settled301-HOLD-20261008.json",
                r.BASE + "/parent-l9-bootstrap-settled302-before-allocation-20261008.json",
                r.BASE + "/parent-l9-qualification-settled306-before-body-20261008.json",
                r.BASE + "/parent-l9-body-settled307-before-audit-20261008.json",
                r.BASE + "/parent-l9-bootstrap-admitted301-before-cold-settlement-20261008.json",
                r.BASE + "/parent-l9-cold-settled302-HOLD-20261008.json",
                r.BASE + "/parent-l9r2-bootstrap-settled303-before-allocation-20261008.json",
                r.BASE + "/parent-l9r2-qualification-settled307-before-body-20261008.json",
                r.BASE + "/parent-l9r2-body-settled308-before-audit-20261008.json",
                r.BASE + "/parent-l9r2-bootstrap-settled303-HOLD-20261008.json",
                r.BASE + "/parent-l10-bootstrap-admitted303-before-cold-settlement-20261008.json",
                r.BASE + "/parent-l10-cold-settled304-HOLD-20261008.json",
                r.BASE + "/parent-l11-bootstrap-settled305-HOLD-20261008.json",
                r.BASE + "/parent-l12-cold-settled306-HOLD-20261008.json",
                r.BASE + "/parent-l12-bootstrap-admitted305-before-cold-settlement-20261008.json",
                r.BASE + "/parent-l13r2-bootstrap-settled307-before-allocation-20261008.json",
                r.BASE + "/parent-l13r2-qualification-settled311-before-body-20261008.json",
                r.BASE + "/parent-l13r2-body-settled312-before-audit-20261008.json",
                r.BASE + "/parent-l12-bootstrap-settled306-before-allocation-20261008.json",
                r.BASE + "/parent-l12-qualification-settled310-before-body-20261008.json",
                r.BASE + "/parent-l12-body-settled311-before-audit-20261008.json",
                r.BASE + "/parent-l11-bootstrap-settled305-before-allocation-20261008.json",
                r.BASE + "/parent-l11-qualification-settled309-before-body-20261008.json",
                r.BASE + "/parent-l11-body-settled310-before-audit-20261008.json",
                r.BASE + "/parent-l10-bootstrap-settled304-before-allocation-20261008.json",
                r.BASE + "/parent-l10-qualification-settled308-before-body-20261008.json",
                r.BASE + "/parent-l10-body-settled309-before-audit-20261008.json",
                r.BASE + "/parent-l8-bootstrap-settled301-before-allocation-20261008.json",
                r.BASE + "/parent-l8-qualification-settled305-before-body-20261008.json",
                r.BASE + "/parent-l8-body-settled306-before-audit-20261008.json",
                r.BASE + "/parent-l4-qualification-settled298-before-body-20261008.json",
                r.BASE + "/parent-l4-body-settled299-before-audit-20261008.json",
                r.BASE + "/parent-l13r2-bootstrap-settled307-HOLD-20261008.json",
                r.BASE + "/s14-anchor307-after-repair400.json",
                r.BASE + "/parent-s14-bootstrap-settled308-before-allocation-20261009.json",
                r.BASE + "/parent-s14-qualification-settled312-before-body-20261009.json",
                r.BASE + "/parent-s14-body-settled313-before-audit-20261009.json",
                r.BASE + "/s14-bootstrap-settled308-HOLD-20261009.json",
                r.BASE + "/parent-s15-bootstrap-settled309-before-allocation-20261009.json",
                r.BASE + "/parent-s15-qualification-settled313-before-body-20261009.json",
                r.BASE + "/parent-s15-body-settled314-before-audit-20261009.json",
                r.BASE + "/s15-bootstrap-settled309-HOLD-20261009.json",
                r.BASE + "/s16-anchor309-after-repair400-v2.json",
                r.BASE + "/parent-s16-bootstrap-settled310-before-allocation-20261009.json",
                r.BASE + "/parent-s16-qualification-settled314-before-body-20261009.json",
                r.BASE + "/parent-s16-body-settled315-before-audit-20261009.json",r.BASE+"/s16-bootstrap-settled310-HOLD-20261009.json",r.BASE+"/s16-recovered309-before-bootstrap.json",r.BASE+"/parent-s17-bootstrap-settled311-before-allocation-20261009.json",r.BASE+"/parent-s17-qualification-settled315-before-body-20261009.json",r.BASE+"/parent-s17-body-settled316-before-audit-20261009.json",r.BASE+"/s17-bootstrap-settled311-HOLD-20261009.json",r.BASE+"/s18-anchor311-after-repair20.json",r.BASE+"/s17-recovered310-before-bootstrap.json",r.BASE+"/parent-s18-bootstrap-settled312-before-allocation-20261009.json",r.BASE+"/parent-s18-qualification-settled316-before-body-20261009.json",r.BASE+"/parent-s18-body-settled317-before-audit-20261009.json",r.BASE+"/s18-bootstrap-settled312-HOLD-20261009.json",r.BASE+"/s19-anchor312-after-repair200.json",r.BASE+"/s18-recovered311-before-bootstrap.json",r.BASE+"/parent-s19-bootstrap-settled313-before-allocation-20261009.json",r.BASE+"/parent-s19-qualification-settled317-before-body-20261009.json",r.BASE+"/parent-s19-body-settled318-before-audit-20261009.json",r.BASE+"/s19-bootstrap-settled313-HOLD-20261009.json",r.BASE+"/s22-anchor313-after-repair110.json",r.BASE+"/s19-recovered312-before-bootstrap.json",r.BASE+"/parent-s22-bootstrap-settled314-before-allocation-20261009.json",r.BASE+"/parent-s22-qualification-settled318-before-body-20261009.json",r.BASE+"/parent-s22-body-settled319-before-audit-20261009.json",r.BASE+"/s22-bootstrap-settled314-HOLD-20261009.json",r.BASE+"/s23-anchor314-after-grant2000.json",r.BASE+"/parent-s23-bootstrap-settled315-before-allocation-20261009.json",r.BASE+"/parent-s23-qualification-settled319-before-body-20261009.json",r.BASE+"/parent-s23-body-settled320-before-audit-20261009.json",'verification/banc-natural-gait-m3-20260919/s23-bootstrap-settled315-HOLD-20261009.json','verification/banc-natural-gait-m3-20260919/s24-anchor315-before-recovery.json','verification/banc-natural-gait-m3-20260919/s24-recovered315-before-bootstrap.json','verification/banc-natural-gait-m3-20260919/parent-s24-bootstrap-settled316-before-allocation-20261009.json','verification/banc-natural-gait-m3-20260919/parent-s24-qualification-settled320-before-body-20261009.json','verification/banc-natural-gait-m3-20260919/parent-s24-body-settled321-before-audit-20261009.json','verification/banc-natural-gait-m3-20260919/s24-bootstrap-settled316-HOLD-20261009.json','verification/banc-natural-gait-m3-20260919/s24-bootstrap-settled316-HOLD-20261009.json','verification/banc-natural-gait-m3-20260919/s25-anchor316-before-recovery.json','verification/banc-natural-gait-m3-20260919/s25-recovered316-before-bootstrap.json','verification/banc-natural-gait-m3-20260919/parent-s25-bootstrap-settled317-before-allocation-20261009.json','verification/banc-natural-gait-m3-20260919/parent-s25-qualification-settled321-before-body-20261009.json','verification/banc-natural-gait-m3-20260919/parent-s25-body-settled322-before-audit-20261009.json','verification/banc-natural-gait-m3-20260919/s25-bootstrap-settled317-HOLD-20261009.json','verification/banc-natural-gait-m3-20260919/s25-bootstrap-settled317-HOLD-20261009.json','verification/banc-natural-gait-m3-20260919/s26-anchor317-before-recovery.json','verification/banc-natural-gait-m3-20260919/s26-recovered317-before-bootstrap.json','verification/banc-natural-gait-m3-20260919/parent-s26-bootstrap-settled318-before-allocation-20261009.json','verification/banc-natural-gait-m3-20260919/parent-s26-qualification-settled322-before-body-20261009.json','verification/banc-natural-gait-m3-20260919/parent-s26-body-settled323-before-audit-20261009.json','verification/banc-natural-gait-m3-20260919/s26-bootstrap-settled318-HOLD-20261009.json','verification/banc-natural-gait-m3-20260919/s27-anchor317-before-recovery.json','verification/banc-natural-gait-m3-20260919/s27-recovered317-before-bootstrap.json','verification/banc-natural-gait-m3-20260919/parent-s27-bootstrap-settled318-before-allocation-20261009.json','verification/banc-natural-gait-m3-20260919/parent-s27-qualification-settled322-before-body-20261009.json','verification/banc-natural-gait-m3-20260919/parent-s27-body-settled323-before-audit-20261009.json','verification/banc-natural-gait-m3-20260919/s27-bootstrap-settled318-HOLD-20261009.json','verification/banc-natural-gait-m3-20260919/s28-anchor318-before-recovery.json','verification/banc-natural-gait-m3-20260919/s28-recovered318-before-bootstrap.json','verification/banc-natural-gait-m3-20260919/parent-s28-bootstrap-settled319-before-allocation-20261009.json','verification/banc-natural-gait-m3-20260919/parent-s28-qualification-settled323-before-body-20261009.json','verification/banc-natural-gait-m3-20260919/parent-s28-body-settled324-before-audit-20261009.json','verification/banc-natural-gait-m3-20260919/s28-bootstrap-settled319-HOLD-20261009.json'):
        cap = 524288
    if path in (r.BASE+"/parent-direct-science-20261008-contact-l11-bootstrap/stderr.bin", r.BASE+"/parent-direct-science-20261008-contact-l13r2-bootstrap/stderr.bin", r.BASE+"/parent-direct-science-20261009-contact-s14-bootstrap/stderr.bin", r.BASE+"/parent-direct-science-20261009-contact-s15-bootstrap/stderr.bin",r.BASE+"/parent-direct-science-20261009-contact-s16-bootstrap/stderr.bin",r.BASE+"/parent-direct-science-20261009-contact-s19-bootstrap/stderr.bin",r.BASE+"/parent-direct-science-20261009-contact-s23-bootstrap/stderr.bin",r.BASE+"/parent-direct-science-20261009-contact-s24-bootstrap/stderr.bin",r.BASE+"/parent-direct-science-20261009-contact-s25-bootstrap/stderr.bin"):
        r.exact(pin, {'path':path,'bytes':0,'sha256':"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}, "exact empty historical log")
        return r.raw_pin(pin, admission.ROOT, 262144, allow_empty=True)
    if path in ("tools/banc_m3_checkpoint_harness_runtime.py", "tools/banc_m3_checkpoint_harness_runtime_native.ps1") or path.endswith(("/tools/banc_m3_checkpoint_harness_runtime.py", "/tools/banc_m3_checkpoint_harness_runtime_native.ps1")):
        cap = 65536
    try:
        return r.raw_pin(pin, admission.ROOT, cap)
    except (ValueError, OSError):
        mapping = r.read_pin(L4_BEFORE, admission.ROOT, 262144)
        row = l4_before_row(pin, mapping)
        if row is not None:
            return r.raw_pin(row["archive"], admission.ROOT, row["cap"])
        for map_pin in (L3_BEFORE, L2_BEFORE, L_BEFORE):
            mapping = r.read_pin(map_pin, admission.ROOT, 262144)
            for row in mapping["inputs"]:
                if row["source"] == pin and row["archive"] is not None:
                    r.exact(row["archive"]["bytes"], pin["bytes"], "whole exact archive bytes")
                    r.exact(row["archive"]["sha256"], pin["sha256"], "whole exact archive SHA")
                    return r.raw_pin(row["archive"], admission.ROOT, row["cap"])
        return k.before_archive(pin, admission.ROOT)

def observation_directory(fixture):
    """Monotonic exclusive directories retain every previous real Native observation."""
    fixture._native_observation_index = getattr(fixture, '_native_observation_index', 0) + 1
    path = fixture.root / f"generated-native-observation-{fixture._native_observation_index:06d}"
    path.mkdir(exist_ok=False)
    return path

def stage_legacy(root):
    """Executed dedicated Source differs from G/H/I historical metadata operands."""
    spec = {"path": "tools/banc_m3_checkpoint_legacy_epoch_native.ps1", "bytes": 126162,
            "sha256": "a790ccde241733ed7e23bc8bbba63d4a9833c02708567907b385ef0d33af8116"}
    for pin in (spec, r.IMMUTABLE_RECORD_PIN):
        target = root / pin["path"]
        raw = r.raw_pin(pin, admission.ROOT, 262144)
        if target.exists():
            r.exact(target.read_bytes(), raw, "same immutable executable/record staged")
        else:
            r.fixture_text_write(root, pin["path"], raw)


def archive_pair(spec):
    return exact_source_operand(spec)


def parent(q, token):
    """Called after all existing actual parent PID/Job/token assertions."""
    from tools.banc_m3_startup_ray_supervisor import generated_budget_limit
    ledger = r.read_pin(r.text_spec(q["ledger_path"], admission.ROOT, 524288), admission.ROOT, 524288)
    state = k.validate(ledger, admission.ROOT)
    r.exact(state["phase"], "begun", "actual K parent owns this running fixture")
    r.exact(state["completed"], 0 if q["call_id"] == k.BOOTSTRAP_CALL else 4, "actual K progress before bootstrap/gate-B")
    r.exact(r.current_caps(ledger, admission.ROOT, generated_budget_limit),
            k.OLD_CAPS if q["call_id"] == k.BOOTSTRAP_CALL else k.CAPS, "actual paired K parent caps")
    k.identity(q, token["request"], state["source"], admission.ROOT)
    r.exact(ledger["direct_science_calls"][q["call_id"]],
            {"state": "admitted", "mode": "generated", "request": token["request"], "reserved_s": q["limits"]["outer_s"]},
            "actual K parent admission; no rewritten token")
    r.exact(ledger["direct_science_workers"][q["call_id"]]["lock_token"], r.text_spec(r.SHARED_LOCK, admission.ROOT),
            "same exact live physical parent exclusion")
    return k.CAP_TRIPLES[0 if q["call_id"] == k.BOOTSTRAP_CALL else 4]


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True)
        from tools import banc_m3_checkpoint_harness_runtime as h
        self.copy(h.S15_PINS["catalog"])
        for row in h.data(h.S15_PINS["catalog"], self.root)["rows"]:
            raw = h._history_operand(row["pin"], admission.ROOT, row["cap"], row["kind"])
            t56 = self.root / row["pin"]["path"]
            if t56.exists(): r.exact(t56.read_bytes(), raw, "same catalog operand")
            else: r.fixture_text_write(self.root, row["pin"]["path"], raw)
        for name in ("refinement_anchor", "previous_recovery", "previous_source", "previous_contract", "previous_catalog", "technical_refusal"):
            pin = h.S15_PINS[name]
            raw = r.raw_pin(pin, admission.ROOT, h.reader_cap(pin["path"]))
            t56 = self.root / pin["path"]
            if t56.exists(): r.exact(t56.read_bytes(), raw, "same fixed continuity operand")
            else: r.fixture_text_write(self.root, pin["path"], raw)
        gate = h.data(h.S15_PINS["technical_refusal"], self.root)
        for name in ("actual_tool_return", "actual_known_compiler_type_refusal", "journal"):
            pin = gate[name]
            raw = r.raw_pin(pin, admission.ROOT, h.reader_cap(pin["path"]))
            t56 = self.root / pin["path"]
            if t56.exists(): r.exact(t56.read_bytes(), raw, "same fixed historical proof")
            else: r.fixture_text_write(self.root, pin["path"], raw)
        # Fixed D14 archives, full D12 history and partial D13/R11 raw operands.
        for pin in [{'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/a318.json', 'bytes': 495559, 'sha256': 'fadfb1604441f5c2cc481e15b07eeaf10a7a7529cedc8969bc496196ea221274'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-recovery10.json', 'bytes': 1903, 'sha256': '27f8b420a0f7cdb3447aac416cd2ac71cf499ee8b5ea719ff05c968fd6f1184e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/s1-before.json', 'bytes': 262140, 'sha256': 'f1b802af060e385f03955b5cbba5830cc9ee42ccf757b9800016a45afd7cabfd'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/c5-before.json', 'bytes': 97623, 'sha256': '9690bd2ec7a65d33432071d09d9050c6d949c92d6a20852fc92072737f7dcb41'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/h28-before.json', 'bytes': 261912, 'sha256': 'e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/actual-fast-warm-gate.json', 'bytes': 2818, 'sha256': '042051f83aaa9c347c72d943e76d0958cf22afeccb5fd3c929bf0241cd5f5a1f'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/s1-before.json', 'bytes': 262140, 'sha256': '39b4910b21c51042a5d68f3da51061ce2d57ead51ec5ff4cdcf82554a6493ef0'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/c5-before.json', 'bytes': 74674, 'sha256': '3a87bc0f34b8a36ddd5153940f2b17fccbdd990b32459aea28ae92e51be58b0a'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f12-root/actual-fast-warm-gate.json', 'bytes': 5074, 'sha256': '9c9348756e9dacf4ee53a320e50acc22d25c8fe9028419dccbc35819356d7b7e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f12-root/actual-tool-return.json', 'bytes': 1846, 'sha256': '71341a5233ecc031b45876ab24d1f7e4305ed79c738aa19f3efa71dc6c250162'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f12-root/actual-layout-compatibility-diagnostic.json', 'bytes': 1331, 'sha256': 'efc4694c9651f842832549809e1613b9900ab71004da098bb82dc24c29ee8a6d'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-catalog-leaves-300021e7370a4dc8b163ef22f0245f31.jsonl', 'bytes': 6393, 'sha256': 'c4bb7e972e5ed5950e554983344f5d34849008f9a2a9a984a874d1ff8795926e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/actual-tool-return.json', 'bytes': 2798, 'sha256': 'e76c924ab89dd27752d2f8a1afe68bd6839f363471886c073b3b14fce0d5d2c0'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/actual-compiler-type-refusal.json', 'bytes': 1968, 'sha256': '448f06a882edd1757895f90dfd4bd1c7bfad38f414fcdda3909a0eacfa34e407'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-catalog-leaves-30944ee8fe084e2aac23de9d47e4de80.jsonl', 'bytes': 496, 'sha256': 'b3ebd169ebfc3f700ec42a0c3b7b9292f35b7561cb6f7d100c71bf3835669606'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-recovery11.json', 'bytes': 1882, 'sha256': '0098806a84d0c18cf606482710d15e73dbe21defae360badba88a6c909db4c26'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/a318.json', 'bytes': 495559, 'sha256': 'fadfb1604441f5c2cc481e15b07eeaf10a7a7529cedc8969bc496196ea221274'}, {'path': 'verification/banc-natural-gait-m3-20260919/s27-bootstrap-settled318-HOLD-20261009.json', 'sha256': '70175b4bc56c19412b64500343ebc72bc538a7b3776dd60733d43efeeb59476e', 'bytes': 495417}, {'path': 'verification/banc-natural-gait-m3-20260919/e28.json', 'bytes': 2443, 'sha256': 'e5e284cbaa1f67f1dd6e14bdf3fe74c5f902d751c387f6850700323c23609c93'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/h28-before.json', 'bytes': 261912, 'sha256': 'e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s27-actual.json', 'bytes': 3365, 'sha256': 'fcf5b1fa42a8348fa52f85e0259cc3b3554be5d96206fd5e77f4fcc54e01eb94'}]:
            raw = r.raw_pin(pin, admission.ROOT, 524288 if pin["bytes"]>262144 else 262144)
            t56 = self.root / pin["path"]
            if t56.exists(): r.exact(t56.read_bytes(),raw,"same fixed D14 continuity raw")
            else: r.fixture_text_write(self.root,pin["path"],raw)
        h8 = h.data(h.S15_PINS["previous_recovery"], self.root)
        for name in ("before_ledger","previous_recovery","previous_source","original_ledger","repair","catalog","latest_failure","technical_refusal"):
            pin = h8[name]
            raw = r.raw_pin(pin, admission.ROOT, 524288 if name in ("before_ledger","original_ledger") else 262144)
            t56 = self.root / pin["path"]
            if t56.exists(): r.exact(t56.read_bytes(), raw, "same full literal historical R10 operand")
            else: r.fixture_text_write(self.root, pin["path"], raw)
        current = self.copy(r.text_spec(h.SOURCE_PATH, admission.ROOT))
        c56 = h.data(current, self.root)
        self.freeze = c56
        self.full_source()
        self.copy(c56["sources"][h.CONTRACT_PATH])
        self.copy(c56["reviewed_source"])
        current_review = h.data(c56["reviewed_source"], self.root)
        current_contract = h.data(c56["sources"][h.CONTRACT_PATH], self.root)
        self._stage_paid(current_contract, h)
        for pin in D15B_METADATA_PINS:
            t56=self.root/pin["path"]
            raw=r.raw_pin(pin,admission.ROOT,524288 if pin["bytes"]>262144 else 262144)
            if t56.exists():r.exact(t56.read_bytes(),raw,"same D15B staged raw")
            else:r.fixture_text_write(self.root,pin["path"],raw)
        for name,path in (("root_design","docs/M3_28D18.md"),("astra_markdown",h.BASE+"/s28-d18-astra/review.md"),("astra_json",h.BASE+"/s28-d18-astra/review.json"),("root_reflection","docs/M3_28R18.md")):
            pin = current_contract["provenance"][name]
            h.metadata_pin(pin,path)
            raw = r.raw_pin(pin,admission.ROOT,262144)
            t56 = self.root / path
            if t56.exists():r.exact(t56.read_bytes(),raw,"same finite direct provenance")
            else:r.fixture_text_write(self.root,path,raw)
        b8 = current_contract["pinned_checkpoint"]
        for pin in (b8["coverage"], *b8["provenance"].values(), current_review["root_reflection"], *current_review["astra_review"].values(), *current_review["root_reviews"], *current_review["sol_reports"]):
            self.copy(pin)
        e8 = [pin for pin in current_review["root_reviews"] if pin["path"] == h.BASE+"/s28-f18-root/SECOND.json"]
        r.need(len(e8) == 1, "unique current ROOT SECOND fit report")
        d8 = h.data(e8[0], self.root)
        a57 = d8["fit_provenance"]
        d57 = ("docs/M3_28D18_FIT.md", h.BASE+"/s28-d18-fit-astra/review.md", h.BASE+"/s28-d18-fit-astra/review.json", "docs/M3_28R18_FIT.md")
        r.need(type(a57) is list and len(a57) == 4, "exact four supplemental fit pins")
        for b57, c57 in zip(a57, d57):
            h.metadata_pin(b57, c57)
            r.need(type(b57) is dict and type(b57["path"]) is str and b57["bytes"] <= 262144, "bounded exact supplemental metadata pin")
            r.fixture_text_write(self.root, c57, r.raw_pin(b57, admission.ROOT, 262144))
        self._stage_residual(current_review, h)
        pins = (k.ANCHOR, r.POLICY_PIN, r.J_PIN, k.J_FAILED, k.J_TRACE, k.J_OUTER, k.J_BODY, k.J_FREEZE, r.ROSTER_PIN)
        for pin in pins:
            self.copy(pin)
        self._staged_current_source = copy.deepcopy(current)
        self._staged_current_freeze = copy.deepcopy(c56)
        self._legacy_source_active = False
        self.source = self.copy(r.text_spec(k.SOURCE_PATH, admission.ROOT, 262144))
        self.freeze = k.data(self.source, self.root)
        for name in (k.CONTRACT_PATH, k.BODY_PATH, k.ROSTER_PATH):
            self.copy(self.freeze["sources"][name])
        self.copy(self.freeze["reviewed_source"])
        self.recovery = self.put(k.RECOVERY_PATH, {"schema": "flylab.m3.checkpoint-recovery.v1", "epoch": k.EPOCH,
            "before_ledger": k.ANCHOR, "source_freeze": self.source,
            "failed_j": {"cost": k.J_FAILED, "trace": k.J_TRACE, "outer": k.J_OUTER},
            "released_hold": "direct-science-20261008-contact-j-lint",
            "reason": "KNOWN_CLOSED_ONCE_PAID_J_PER_CALL_OVERRUN_RETAINED", "new_resource_grant_s": 0, "refund_s": 0})
        self.ledger = k.recovered_snapshot(self.recovery, self.root)
        self.ledger_path = k.BASE + "/parent-e358-execution-ledger.json"
        self.publish()

    def _stage_paid(self, current_contract, h):
        # D18 finite paid stage begin: two ancestors and all three paid steps.
        def stage_paid_pin(pin):
            raw = r.raw_pin(pin, admission.ROOT, h.reader_cap(pin["path"]))
            t56 = self.root / pin["path"]
            if t56.exists(): r.exact(t56.read_bytes(), raw, "same finite paid raw")
            else: r.fixture_text_write(self.root, pin["path"], raw)
        paid = current_contract["actual320"]
        h.whole_exact(paid["adoption_before_Source"], h.CONTINUATION_BEFORE_SOURCE)
        for step, cid in enumerate(("direct-science-20261009-contact-s28-bootstrap-r3", "direct-science-20261009-contact-s28-bootstrap-r2", "direct-science-20261009-contact-s28-bootstrap-r1")):
            r.exact(paid["old_failed_call"], cid, "fixed staged paid identity")
            for pin in paid["raw_pins"]: stage_paid_pin(pin)
            for pin in (paid["old_request"], paid["old_cost"]): stage_paid_pin(pin)
            n8 = h.data(paid["old_request"], self.root)
            m8 = h.data(paid["old_cost"], self.root)
            h.whole_exact(m8["request"], paid["old_request"])
            r.exact((n8["call_id"], m8["cost_id"]), (cid, cid), "same finite request/cost identity")
            r.exact(paid["old_cost"]["path"], h.BASE+"/parent-"+cid+"-full-return.json", "fixed finite cost path")
            for pin in (m8["trace"], m8["outer"]): stage_paid_pin(pin)
            if step < 2:
                c8 = [row for row in paid["archives"] if row["logical_current_before"]["path"] == h.CONTRACT_PATH]
                p8 = [row for row in paid["archives"] if row["logical_current_before"]["path"] == h.REVIEW_PATH]
                r.need(len(c8) == len(p8) == 1, "finite unique predecessor contract/review")
                cr, rr = c8[0], p8[0]
                for pin in (paid["adoption_before_Source"], cr["immutable_archive"], rr["immutable_archive"]): stage_paid_pin(pin)
                k8 = h.data(paid["adoption_before_Source"], self.root)
                j8 = h.data(rr["immutable_archive"], self.root)
                h.whole_exact(k8["reviewed_source"], rr["logical_current_before"])
                h.whole_exact([k8["checkpoint_contract"], k8["sources"][h.CONTRACT_PATH], j8["sources"][h.CONTRACT_PATH]], [cr["logical_current_before"]]*3)
                h.whole_exact(k8["sources"], j8["sources"])
                paid = h.data(cr["immutable_archive"], self.root)["actual320"]
        # D18 finite paid stage end.

    def _stage_residual(self, current_review, h):
        residual_reports = [pin for pin in current_review["root_reviews"] if pin["path"] == h.BASE+"/s28-f18-root/residual03/final-v2.json"]
        r.need(len(residual_reports) == 1, "unique authenticated residual ROOT final report")
        g57 = h.data(residual_reports[0], self.root)
        e57 = g57["residual_provenance"]
        f57 = ("docs/M3_28D18_RESIDUAL3.md", h.BASE+"/s28-f18-root/residual03/astra/review.md", h.BASE+"/s28-f18-root/residual03/astra/review.json", "docs/M3_28R18_RESIDUAL3.md")
        r.need(type(e57) is list and len(e57) == 4, "exact four residual provenance pins")
        for r56, x56 in zip(e57, f57):
            r.need(type(r56) is dict and list(r56) == ["path","bytes","sha256"] and type(r56["path"]) is str and type(r56["bytes"]) is int and 0 < r56["bytes"] <= 262144 and type(r56["sha256"]) is str, "typed bounded residual descriptor")
            h.metadata_pin(r56, x56)
            u56 = r.raw_pin(r56, admission.ROOT, 262144)
            r.fixture_text_write(self.root, x56, u56)

    def copy(self, pin):
        raw = exact_source_operand(pin)
        target = self.root / pin["path"]
        if target.exists():
            r.exact(target.read_bytes(), raw, "exact pre-staged Source/evidence")
            return pin
        return r.fixture_text_write(self.root, pin["path"], raw)

    def put(self, name, value):
        return r.fixture_text_write(self.root, name, json.dumps(value, ensure_ascii=True, allow_nan=False).encode("utf8"))

    def publish(self):
        return self.put(self.ledger_path, self.ledger)

    def full_source(self):
        transition = hasattr(self, "source") and not getattr(self, "_legacy_source_active", False)
        current_freeze = None
        if transition:
            from tools import banc_m3_checkpoint_harness_runtime as current
            legacy = {"path": k.SOURCE_PATH, "bytes": 257763,
                      "sha256": "66ffae7fb088befd1e1155c93db65bb62005ec59853ee3ba17708e854bda1c0f"}
            r.exact(self.source, legacy, "independently pinned historical K Source phase")
            r.exact(self.freeze, k.data(legacy, self.root), "whole historical freeze before transition")
            r.exact(self._staged_current_source, r.text_spec(current.SOURCE_PATH, admission.ROOT, 262144),
                    "independently pinned current Source phase")
            current_freeze = current.data(self._staged_current_source, self.root)
            r.exact(self._staged_current_freeze, current_freeze, "whole staged current freeze before transition")
        for pin in self.freeze["sources"].values():
            if not Path(pin["path"]).is_absolute():
                prior = current_freeze["sources"][pin["path"]] if transition else None
                if transition and prior != pin:
                    current_raw = exact_source_operand(prior)
                    cap = 524288 if prior["bytes"] > 262144 else 262144
                    r.exact(r.raw_pin(prior, self.root, cap), current_raw, "exact current bytes before K phase replacement")
                    legacy_raw = exact_source_operand(pin)
                    staged = r.fixture_text_write(self.root, pin["path"], legacy_raw)
                    r.exact(staged, pin, "whole independently pinned historical replacement")
                    cap = 524288 if pin["bytes"] > 262144 else 262144
                    r.exact(r.raw_pin(pin, self.root, cap), legacy_raw, "whole historical replacement readback")
                else:
                    self.copy(pin)
        if transition:
            self._legacy_source_active = True

    def request(self, call=k.BOOTSTRAP_CALL):
        p = k.plan(call, self.freeze, self.root)
        q = {"schema": "flylab.m3.direct-science.supervision.v1", "call_id": call, "mode": p["mode"], "role": p["role"],
             "source_freeze": self.source, "limits": p["limits"], "output": p["output"], "ledger_path": self.ledger_path,
             "shared_original_lock_path": r.SHARED_LOCK, "generated_behavior": "command", "generated_argv": p["argv"]}
        spec = self.put(k.BASE + "/parent-" + call + "-request.json", q)
        return q, spec

    def admit(self):
        self.q, self.request_spec = self.request()
        k.dispatch(self.ledger, self.q, self.request_spec, self.root, phase="closed")
        self.ledger["direct_science_calls"][self.q["call_id"]] = {"request": self.request_spec, "state": "admitted", "mode": "generated", "reserved_s": 90}
        self.ledger["direct_science_reservations"] = {self.q["call_id"]: {"request": self.request_spec, "reserved_s": 90}}
        self.publish()
        return k.validate(self.ledger, self.root)

    def evidence(self, seconds=1.0, exit_code=0, suspended=False, uncertain=False, no_create=False):
        """Synthetic immutable observations; production validators are the subject."""
        call = self.q["call_id"]
        state = "NO_CREATE_ATTEMPT" if no_create else ("UNCERTAIN_CREATION" if uncertain else ("CREATED_SUSPENDED_NOT_RESUMED" if suspended else "RESUMED"))
        outer = {"schema": "flylab.m3.direct-science.outer.v1", "request": self.request_spec, "exit_code": exit_code,
                 "owner_pid": 411, "owner_creation_ticks": 412, "worker_pid": None if no_create else 413,
                 "worker_creation_ticks": None if no_create else 414, "worker_exit_code": None if no_create else exit_code,
                 "job_created": not no_create, "creation_attempted": not no_create, "worker_started": not (suspended or uncertain or no_create),
                 "worker_creation_state": state, "resume_attempted": not (suspended or no_create),
                 "resume_outcome": "NOT_ATTEMPTED" if suspended or no_create else ("UNCERTAIN" if uncertain else "RESUMED"),
                 "all_owned_ended": not uncertain, "reservation_released": not uncertain,
                 "unknown_owned_identities": {"413": 414} if uncertain else {}, "stdout_eof": not (uncertain or no_create),
                 "stderr_eof": not (uncertain or no_create), "timeout": exit_code == 124, "first_error": None,
                 "fixture_synthetic": True}
        self.outer_spec = self.put(self.q["output"] + "/outer.json", outer)
        dispatch = 1800000000000
        trace = {"schema": "flylab.m3.direct-science.genuine-return.v1", "phase": "FINAL", "first_error": None,
                 "call_id": call, "request": self.request_spec, "source_freeze": self.source,
                 "dispatch_ms": dispatch, "first_final_ms": dispatch + round(seconds * 1000),
                 "full_return_s": seconds, "exit_code": exit_code, "fixture_synthetic": True}
        self.trace_spec = self.put(k.BASE + "/parent-" + call + "-genuine-return.json", trace)
        if not (uncertain or no_create) and call in self.ledger["direct_science_calls"] and self.ledger["direct_science_calls"][call]["state"]=="admitted":
            self.ledger["direct_science_reservations"]={};self.ledger["direct_science_workers"]={}
            self.ledger["direct_science_pending_actuals"]={call:{"request":self.request_spec,"state":"actual_pending","ownership_closed":True}}
            self.ledger=json.loads(json.dumps(self.ledger,ensure_ascii=True,allow_nan=False,sort_keys=True));self.publish()
        return outer

    def unknown_owner(self):
        call = self.q['call_id']
        identity = {'owner_pid': 411, 'owner_creation_ticks': 412, 'worker_pid': 413, 'worker_creation_ticks': 414}
        token = {'schema': 'flylab.m3.direct-science.lock-owner.v1', 'call_id': call, 'request': self.request_spec,
                 'owner_pid': 411, 'owner_creation_ticks': 412, 'entry_counter_s': 1.0, 'job_name': 'Local\\generated-synthetic-K'}
        token_spec = self.put(r.SHARED_LOCK, token)
        self.ledger['direct_science_workers'] = {call: {'request': self.request_spec, 'state': 'begun', 'lock_token': token_spec, **identity}}
        self.ledger['direct_science_blocking_holds'] = {call: {'request': self.request_spec, 'state': 'ownership_unknown', **identity}}
        self.publish()
        return k.validate(self.ledger, self.root)

    def original_evidence(self):
        """Pure immutable-metadata fee/identity fixture; never an Original execution."""
        body = k.data(self.freeze['sources'][k.BODY_PATH], self.root)
        for key in ('contract', 'grant', 'inputs', 'roster'):
            self.copy(body[key])
        roster = k.data(body['roster'], self.root)
        previous = k.data(self.copy(roster['previous_geometry_roster']), self.root)
        self.copy(previous['historical_manifest'])
        qualified = self.put(k.QUALIFICATION_PATH, {'schema': 'flylab.m3.direct-science.qualification.v8', 'epoch': k.EPOCH,
            'source_freeze': self.source, 'actual_costs': [], 'tests': [], 'fixture_synthetic': True})
        before = copy.deepcopy(self.ledger)
        before[k.QUALIFICATION_KEY] = qualified
        before[k.ALLOCATION_KEY] = {'fixture_synthetic': True}
        before_spec = self.put(k.BASE + '/generated-k-original-before.json', before)
        contract = k.data(body['contract'], self.root)
        call = k.CALLS[0]
        self.science = {'schema': 'flylab.m3.contact-learning.request.v1', 'mode': 'original', 'role': 'contact_candidate_body',
            'call_id': call, 'output': k.BASE + '/parent-' + call + '/result', 'source_freeze': self.source,
            'body_retry': self.freeze['sources'][k.BODY_PATH], 'limits': contract['limits']['contact_candidate_body'],
            'ledger_path': self.ledger_path, 'shared_original_lock_path': r.SHARED_LOCK, 'local_debit_offset': 165,
            'resource_increment_s': 0, 'historical_manifest': previous['historical_manifest'], 'qualification': qualified,
            'before_ledger': before_spec, **{key: body[key] for key in ('contract', 'grant', 'inputs', 'roster')}}
        science_spec = self.put(k.BASE + '/generated-k-original-science.json', self.science)
        p = k.plan(call, self.freeze, self.root)
        self.q = {'schema': 'flylab.m3.direct-science.supervision.v1', 'call_id': call, 'mode': 'original', 'role': p['role'],
            'source_freeze': self.source, 'limits': p['limits'], 'output': p['output'], 'ledger_path': self.ledger_path,
            'shared_original_lock_path': r.SHARED_LOCK, 'grant': body['grant'], 'science_request': science_spec}
        self.request_spec = self.put(k.BASE + '/generated-k-original-supervision.json', self.q)
        outer = self.evidence()
        self.cost = {'schema': 'flylab.m3.contact-learning.full-return.v1', 'cost_id': call, 'mode': 'original',
            'request': science_spec, 'supervisor_request': self.request_spec, 'source_freeze': self.source,
            'trace': self.trace_spec, 'outer': self.outer_spec, 'actual_outer_full_return_s': 1.0, 'exit_code': 0,
            'charged_once': True, 'cap_exceeded': False, **{key: outer[key] for key in ('all_owned_ended', 'reservation_released',
            'worker_started', 'worker_creation_state', 'creation_attempted', 'resume_attempted', 'resume_outcome')},
            **{key: self.science[key] for key in ('role', 'grant', 'inputs', 'roster', 'contract', 'body_retry', 'local_debit_offset', 'resource_increment_s')}}
        self.cost_spec = self.put(k.BASE + '/parent-' + call + '-full-return.json', self.cost)
        self.debit = {'cost_id': call, 'mode': 'original', 'actual_outer_full_return_s': 1.0, 'actual_returned_exit_code': 0, 'descriptor': self.cost_spec}
        return k.fee_proof(self.cost_spec, self.debit, self.source, self.root)

    def python_charge(self, seconds=1.0, exit_code=0, **creation):
        outer = self.evidence(seconds, exit_code, **creation)
        closed = k.closure(outer)
        cost = {"schema": "flylab.m3.direct-science.full-return.v1", "cost_id": self.q["call_id"], "mode": "generated",
                "request": self.request_spec, "source_freeze": self.source, "trace": self.trace_spec, "outer": self.outer_spec,
                "actual_outer_full_return_s": seconds, "exit_code": exit_code, "charged_once": True, "cap_exceeded": seconds > 90,
                "all_owned_ended": outer["all_owned_ended"], "reservation_released": outer["reservation_released"],
                "worker_started": outer["worker_started"], "worker_creation_state": outer["worker_creation_state"],
                "creation_attempted": outer["creation_attempted"], "resume_attempted": outer["resume_attempted"], "resume_outcome": outer["resume_outcome"]}
        self.cost_spec = self.put(k.BASE + "/parent-" + self.q["call_id"] + "-full-return.json", cost)
        debit = {"cost_id": self.q["call_id"], "mode": "generated", "actual_outer_full_return_s": seconds,
                 "actual_returned_exit_code": exit_code, "descriptor": self.cost_spec}
        self.ledger["debits"].append(debit)
        self.ledger["direct_science_calls"][self.q["call_id"]]["state"] = "charged"
        if closed:
            self.ledger["direct_science_reservations"] = {}
            self.ledger["direct_science_pending_actuals"] = {}
        self.ledger.update(k.totals(self.ledger, k.data(r.POLICY_PIN, self.root), k.data(r.J_PIN, self.root), k.OLD_CAPS))
        if closed and exit_code != 0:
            self.ledger['direct_science_blocking_holds'][self.q['call_id']] = k.FAILURE_HOLD
        if seconds > 90 or self.ledger["generated_shared_remaining_s"] < 0:
            k.accounting_hold(self.ledger, 'generated', self.q['call_id'])
        self.publish()
        k.validate(self.ledger, self.root)
        return debit

    def native(self, body):
        r.d19_legacy_fixture_binding(self)
        from tests.test_c_banc_contact_runtime_journal import (
            conserved_native_definitions,
            conserved_run_native,
        )
        host = shutil.which("pwsh")
        r.need(host is not None, "actual Native qualification requires pwsh; never skipped")
        source = conserved_native_definitions(self.root, account_pin=K_ACCOUNT, contact_pin=K_CONTACT) + "\n" + body
        observation = observation_directory(self)
        wrapper = observation / "generated-native.ps1"
        with wrapper.open("x", encoding="utf8") as stream:
            stream.write(source)
        return conserved_run_native(self.root, [host, "-NoProfile", "-File", str(wrapper)], observation_root=observation)

    def native_commit(self):
        return self.native("""
$null=Start-LedgerTransaction 'generated-synthetic-K-commit'
try{$l=Conserved-Json $script:ledgerTransaction.before;$receipt=K-Commit $l 'TRACE';$receipt|ConvertTo-Json -Depth 100 -Compress}
finally{if($null -ne $script:ledgerTransaction -and -not $script:ledgerTransaction.uncertain){Stop-LedgerTransaction $script:ledgerTransaction}}
""".replace("TRACE", self.trace_spec["path"]))


def observed_pass(result):
    assert result.returncode == 0, result.stdout.decode("utf8", errors="replace") + result.stderr.decode("utf8", errors="replace")
    assert result.all_owned_ended and result.stdout_eof and result.stderr_eof and not result.unknown_owned_identities
    assert result.first_error is None and not result.secondary_errors and not result.timeout


def case(root, name):
    return Fixture(root / name)


def negative(function):
    try:
        function()
    except (ValueError, OSError, KeyError, TypeError):
        return
    raise AssertionError("malformed metadata unexpectedly accepted")


def mutation(base, key, value):
    result = copy.deepcopy(base)
    result[key] = value
    return result


def d19_opaque_controls(k,f,lr):
 import hashlib
 r=k.r;ct=k.data(k.CURRENT_CONTRACT,f.root);cut=k.data(ct['d19_fixed_history'],f.root);rows=k.data(cut['historical_inventory'],f.root)
 assert all(type(x['opaque_stream']) is bool for x in rows)
 streams=[x for x in rows if x['opaque_stream']];assert streams
 refuse=r.d19_opaque_refuse
 for x in streams:
  p=x['archive'];cap=x['cap'];path=r._text_path(f.root,p['path']);saved=path.read_bytes();read=lambda p=p,cap=cap:r.raw_pin(p,f.root,cap,allow_empty=p['bytes']==0,opaque=True)
  assert read()==saved and len(saved)==p['bytes'] and hashlib.sha256(saved).hexdigest()==p['sha256']
  r.d19_opaque_write(k,f,p,saved,cap)
  for flag in (None,0,'true',[]):refuse(lambda p=p,cap=cap,saved=saved,flag=flag:r.raw_pin(p,f.root,cap,allow_empty=not saved,opaque=flag),ValueError,'literal opaque stream option')
  _d19_opaque_default(r,p,f.root,cap,saved)
  try:
   if saved:
    path.write_bytes(saved[:-1]+bytes([saved[-1]^1]));refuse(read,ValueError,'full text repeated bytes/stamp/hash')
   path.write_bytes(saved+b'x');refuse(read,ValueError,'exact text size before bounded read')
   path.unlink()
   try:read()
   except FileNotFoundError as e:assert e.filename==str(path)
   else:raise AssertionError('missing stream')
  finally:path.write_bytes(saved)
  assert read()==saved
 for p,cap in ((k.CURRENT_CONTRACT,262144),(k.data(f.source,f.root)['sources']['tools/banc_m3_checkpoint_harness_runtime.py'],131072)):
  refuse(lambda p=p,cap=cap:r.raw_pin(p,f.root,cap,opaque=True),ValueError,'literal opaque stream option')
 refuse(lambda:r.fixture_text_write(f.root,'unrelated.bin',b'x',opaque=True),ValueError,'literal opaque stream option')
 assert (f.root/f.ledger_path).read_bytes()==lr


def d19_synthetic_overrun(g):
 import math
 import struct
 values=(g['sequence'][0]['caps']['outer_s'],g['new_caps']['generated_local'],g['totals']['direct_science_generated_used_s'],g['totals']['generated_shared_remaining_s'])
 assert all(type(v) in (int,float) and math.isfinite(v) for v in values)
 outer,cap,used,shared=values;bounds=(outer,cap-used,shared)
 assert all(v>0 for v in bounds)
 value=float(math.ceil(max(bounds)+1.0))
 assert math.isfinite(value) and all(value>v for v in bounds) and 0<value<=9007199254740991/1000
 ms=round(value*1000);assert 0<ms<=9007199254740991 and ms==value*1000
 assert struct.pack('>d',ms/1000.0)==struct.pack('>d',value)
 return value


def _d19_opaque_default(r,p,root,cap,saved):
 try:saved.decode('utf8',errors='strict')
 except UnicodeDecodeError as wanted:
  try:r.raw_pin(p,root,cap,allow_empty=not saved)
  except UnicodeDecodeError as got:assert (got.encoding,got.object,got.start,got.end,got.reason)==(wanted.encoding,wanted.object,wanted.start,wanted.end,wanted.reason)
  else:raise AssertionError('strict UTF8 default')
 else:assert r.raw_pin(p,root,cap,allow_empty=not saved)==saved


def d19_retained_consumer_reentry():
 return r'''
$script:R20ActualJson=(Get-Command K-Json).ScriptBlock
foreach($warm in @($false,$true)){
 $script:KPC.Clear();$script:KPJ=$null
 $null=K-SourceFence (K-BoundedPin (K-Paths).source)
 if($warm){$output=@(& $script:R14Legacy $script:R14Key (K-Clone $script:R14Seed) (K-Clone $script:R14Event));Need ($output.Length -eq 1 -and $output[0] -is [Collections.IDictionary]) 'Retained warm prime one dictionary';$null=& $script:R14Check $output[0]}
 $script:R20OuterSeed=K-Clone $script:R14Seed;$script:R20OuterEvent=K-Clone $script:R14Event
 $script:R20InnerSeed=K-Clone $script:R14Seed;$script:R20InnerEvent=K-Clone $script:R14Event
 $script:R20ReaderEntered=0;$R20OuterRejected=$false
 try{
  function K-Json($a){
   $pair=$script:KPJ
   if($script:R20ReaderEntered -eq 0 -and $script:KPB -ceq $true -and $pair -is [array] -and $pair.Length -eq 2 -and $pair[0] -is [string] -and $pair[0] -ceq $script:R14Key -and $pair[1] -is [string] -and $pair[1].Length -gt 0 -and $a -is [string] -and $a -ceq $pair[1]){
    $R20CompanionArgument=$a;$script:R20ReaderEntered++
    Need ($script:KPC.Count -eq 1 -and $script:KPC.ContainsKey($script:R14Key)) 'Actual retained companion cache before reader'
    $R20InnerRejected=$false;try{$null=& $script:R14Legacy $script:R14Key $script:R20InnerSeed $script:R20InnerEvent}catch{$R20InnerRejected=$_.Exception.Message -ceq 'Root/reentrancy'}
    Need ($R20InnerRejected -and $script:KPB -ceq $true -and $script:KManagedSourceAccepted -ceq $false -and $null -eq $script:KCatalogManifestContract -and $script:KPC.Count -eq 0 -and $null -eq $script:KPJ) 'Exact retained inner refusal preserves outer capability and clears acceptance/cache'
    return (& $script:R20ActualJson $R20CompanionArgument)
   }
   return (& $script:R20ActualJson $a)
  }
  try{$null=& $script:R14Legacy $script:R14Key $script:R20OuterSeed $script:R20OuterEvent}catch{$R20OuterRejected=$_.Exception.Message -ceq 'Managed backend only inside accepted owned/admitted action'}
 }finally{Set-Item -LiteralPath Function:K-Json -Value $script:R20ActualJson}
 Need ([object]::ReferenceEquals((Get-Command K-Json).ScriptBlock,$script:R20ActualJson)) 'Exact original parser restored'
 Need ($R20OuterRejected -and $script:R20ReaderEntered -eq 1 -and $script:KPB -ceq $false -and $script:KManagedSourceAccepted -ceq $false -and $null -eq $script:KCatalogManifestContract -and $script:KPC.Count -eq 0 -and $null -eq $script:KPJ) 'Exact retained outer refusal and owned cleanup'
 $null=K-SourceFence (K-BoundedPin (K-Paths).source)
 $output=@(& $script:R14Legacy $script:R14Key (K-Clone $script:R14Seed) (K-Clone $script:R14Event));Need ($output.Length -eq 1 -and $output[0] -is [Collections.IDictionary]) 'Reauthenticated retained cold recovery one dictionary';$null=& $script:R14Check $output[0]
}
'''


def d19_early_snapshot_history(f,k,fixture):
    value=f.d19_accepted_history(k,fixture)
    marker="$script:R14Legacy=[scriptblock]::Create"
    assert value.count(marker)==1
    return value.replace(marker,"$null=K-Json (K-Raw $script:R14SnapshotPin 524288)\n"+marker)


def d19_readview_box_identity():
    return r'''
$r23Cache=$script:KCatalogCache;$r23Origin=$script:KCatalogOrigin
$r23Type=$r23Cache.type;$r23Assembly=$r23Cache.assembly
$r23Source=$r23Cache.source;$r23SourceHash=$r23Cache.source_hash;$r23TemplateHash=$r23Cache.template_hash
$r23Accepted=$script:KManagedSourceAccepted;$r23Manifest=$script:KCatalogManifestContract
$r23Types=@($script:R26HelperType);$r23Method=$script:R26Verify
Need ($r23Types.Length -eq 1 -and $r23Types[0] -is [type] -and [object]::ReferenceEquals($r23Types[0],$script:R26HelperType) -and [object]::ReferenceEquals($r23Types[0].Assembly,$script:R26HelperAssembly) -and $r23Types[0] -ne $r23Type) 'R26 exact early Type Assembly retained distinct from Source'
Need ($null -ne $r23Method -and [object]::ReferenceEquals($r23Method,$script:R26Verify) -and [object]::ReferenceEquals($r23Method.DeclaringType,$r23Types[0]) -and $r23Method.IsPublic -and $r23Method.IsStatic -and -not $r23Method.IsGenericMethod -and $r23Method.ReturnType -eq [bool]) 'R26 exact early Verify MethodInfo retained'
Need ($null -ne $script:R26Getter -and [object]::ReferenceEquals($script:R26Getter.DeclaringType,$r23Types[0]) -and $null -ne $script:R26Counter -and [object]::ReferenceEquals($script:R26Counter.DeclaringType,$r23Types[0])) 'R26 exact early getter and counter declaring Type'
Same ($script:R26Counter.GetValue($null)) ([long]0) 'R26 no supplied getter during intervening cases'
Need ($script:R26Bindings -is [object[]] -and $script:R26Bindings.Length -eq 5 -and [object]::ReferenceEquals($script:R26Bindings[0],$script:R26HelperType) -and [object]::ReferenceEquals($script:R26Bindings[1],$script:R26HelperAssembly) -and [object]::ReferenceEquals($script:R26Bindings[2],$script:R26Verify) -and [object]::ReferenceEquals($script:R26Bindings[3],$script:R26Getter) -and [object]::ReferenceEquals($script:R26Bindings[4],$script:R26Counter)) 'R26 exact five early captured metadata references survive case scope'
Need ($script:R27Bindings -is [object[]] -and $script:R27Bindings.Length -eq 5 -and [object]::ReferenceEquals($script:R27Bindings[0],$script:R27Script) -and [object]::ReferenceEquals($script:R27Bindings[1],$script:R27Smoke) -and [object]::ReferenceEquals($script:R27Bindings[2],$script:R27Unsafe) -and [object]::ReferenceEquals($script:R27Bindings[3],$script:R27Check) -and [object]::ReferenceEquals($script:R27Bindings[4],$script:R27Empty)) 'R27 exact early ScriptBlock and bridge references retained'
foreach($method in @($script:R27Smoke,$script:R27Unsafe,$script:R27Check,$script:R27Empty)){Need ([object]::ReferenceEquals($method.DeclaringType,$script:R26HelperType) -and [object]::ReferenceEquals($method.DeclaringType.Assembly,$script:R26HelperAssembly)) 'R27 bridge declaring identities retained'}
Same $script:viewGetterCalls ([long]0) 'R27 no ScriptProperty callback before late unsafe control'
Same ($script:R26Counter.GetValue($null)) ([long]0) 'R27 no CodeProperty callback before late unsafe control'
Need ($script:R28Bindings -is [object[]] -and $script:R28Bindings.Length -eq 3 -and [object]::ReferenceEquals($script:R28Bindings[0],$script:R28Smoke) -and [object]::ReferenceEquals($script:R28Bindings[1],$script:R28Invoke) -and [object]::ReferenceEquals($script:R28Bindings[2],$script:R28Weak)) 'R28 exact early bridge references retained'
foreach($method in @($script:R28Smoke,$script:R28Invoke,$script:R28Weak)){Need ([object]::ReferenceEquals($method.DeclaringType,$script:R26HelperType) -and [object]::ReferenceEquals($method.DeclaringType.Assembly,$script:R26HelperAssembly)) 'R28 exact bridge declaring identities retained'}
Same $script:viewGetterCalls ([long]0) 'R28 no ScriptProperty callbacks before typed Source controls'
Same ($script:R26Counter.GetValue($null)) ([long]0) 'R28 no CodeProperty callbacks before typed Source controls'
function R28-SourceInvoke([object]$method,[object]$arguments) {
 $packed=[object[]]::new(4);$packed[0]=$method;$packed[1]=$script:KCatalogCache.type;$packed[2]=$script:KCatalogCache.assembly;$packed[3]=$arguments
 return ,(K-CatalogInvoke $script:R28Invoke $packed)
}
Need ($script:R29Bindings -is [object[]] -and $script:R29Bindings.Length -eq 1 -and [object]::ReferenceEquals($script:R29Bindings[0],$script:R29Field)) 'R29 exact early captured helper reference'
Need ($script:R29Field -is [Reflection.MethodInfo] -and [object]::ReferenceEquals($script:R29Field.DeclaringType,$script:R26HelperType) -and [object]::ReferenceEquals($script:R29Field.DeclaringType.Assembly,$script:R26HelperAssembly)) 'R29 independent early helper type/assembly'
Same ($script:R26Counter.GetValue($null)) ([long]0) 'R29 late counter0 without reset'
$r23Args=[object[]]::new(3);$r23Args[0]=$viewMethod;$r23Args[1]=$r23Type;$r23Args[2]=$r23Assembly
Same (K-CatalogInvoke $r23Method $r23Args) $true 'R23 CLR original-box identity on actual Source ReadView'
Need ([object]::ReferenceEquals($script:KCatalogCache,$r23Cache) -and [object]::ReferenceEquals($script:KCatalogOrigin,$r23Origin) -and [object]::ReferenceEquals($r23Cache.type,$r23Type) -and [object]::ReferenceEquals($r23Cache.assembly,$r23Assembly) -and [object]::ReferenceEquals($viewMethod.DeclaringType,$r23Type)) 'R23 Source Type Assembly cache origin identities retained'
Need ([object]::ReferenceEquals($script:KCatalogManifestContract,$r23Manifest)) 'R23 accepted Source provenance object retained'
Same $script:KManagedSourceAccepted $r23Accepted 'R23 Source acceptance retained'
Same $r23Cache.source $r23Source 'R23 exact Source literal retained'
Same $r23Cache.source_hash $r23SourceHash 'R23 exact rendered hash retained'
Same $r23Cache.template_hash $r23TemplateHash 'R23 exact template hash retained'
'''


def d19_powershell_member_contract():
    return r'''
$r23Code=@'
using System;
using System.Reflection;
using System.Management.Automation;
public static class D19ReadViewBoxIdentityR23 {
 public static long GetterCalls=0;
 public static object UnsafeGetter(PSObject value) { GetterCalls++; throw new InvalidOperationException("R26 supplied CodeProperty getter must not run"); }
 private static object InputBase(object value,string name) {
  object[] seen=new object[8];
  for(int depth=0;value!=null && value.GetType()==typeof(PSObject) && depth<8;depth++) {
   for(int i=0;i<depth;i++)
    if(Object.ReferenceEquals(seen[i],value))
     throw new InvalidOperationException("R25 helper input cycle: "+name);
   seen[depth]=value;
   object next=((PSObject)value).ImmediateBaseObject;
   if(Object.ReferenceEquals(value,next))
    throw new InvalidOperationException("R25 helper input cycle: "+name);
   value=next;
  }
  if(value!=null && value.GetType()==typeof(PSObject))
   throw new InvalidOperationException("R25 helper input depth: "+name);
  return value;
 }
 public static bool Verify(object methodInput,object typeInput,object assemblyInput) {
  MethodInfo method=InputBase(methodInput,"method") as MethodInfo;
  if(method==null)throw new InvalidOperationException("R25 helper input kind: MethodInfo");
  Type sourceType=InputBase(typeInput,"type") as Type;
  if(sourceType==null)throw new InvalidOperationException("R25 helper input kind: Type");
  Assembly sourceAssembly=InputBase(assemblyInput,"assembly") as Assembly;
  if(sourceAssembly==null)throw new InvalidOperationException("R25 helper input kind: Assembly");
  if(method==null || sourceType==null || sourceAssembly==null ||
     !Object.ReferenceEquals(method.DeclaringType,sourceType) ||
     !Object.ReferenceEquals(sourceType.Assembly,sourceAssembly) ||
     method.Name!="ReadView" || !method.IsPrivate || !method.IsStatic ||
     method.IsGenericMethod || method.ContainsGenericParameters || method.ReturnType!=typeof(bool))
   throw new InvalidOperationException("R23 actual Source ReadView identity/ABI");
  ParameterInfo[] parameters=method.GetParameters();
  if(parameters.Length!=2 || parameters[0].ParameterType!=typeof(object) || parameters[0].IsOut ||
     parameters[1].ParameterType!=typeof(object).MakeByRefType() || !parameters[1].IsOut)
   throw new InvalidOperationException("R23 exact object/out-object parameters");
  object original=(object)0L;
  PSObject wrapper=new PSObject(original);
  object expected=wrapper.ImmediateBaseObject;
  object[] args=new object[]{wrapper,null};
  if(wrapper.GetType()!=typeof(PSObject) || !Object.ReferenceEquals(expected,original) ||
     !Object.ReferenceEquals(args[0],wrapper))
   throw new InvalidOperationException("R23 CLR constructor/getter precondition");
  object answer=method.Invoke(null,args);
  if(answer==null || answer.GetType()!=typeof(bool) || !(bool)answer)
   throw new InvalidOperationException("R23 actual ReadView declined transparent wrapper");
  if(!Object.ReferenceEquals(args[1],original) || !Object.ReferenceEquals(args[1],expected))
   throw new InvalidOperationException("R23 actual ReadView returned different original box");
  if(!Object.ReferenceEquals(args[0],wrapper) || !Object.ReferenceEquals(wrapper.ImmediateBaseObject,original))
   throw new InvalidOperationException("R23 ReadView changed original input/getter identity");
  return true;
 }
 private static void Require(bool ok,string text) { if(!ok)throw new InvalidOperationException("R27 "+text); }
 private static MethodInfo Method(object input,string name) {
  MethodInfo value=InputBase(input,name) as MethodInfo;
  Require(value!=null,"metadata MethodInfo "+name);return value;
 }
 private static Type Kind(object input,string name) {
  Type value=InputBase(input,name) as Type;
  Require(value!=null,"metadata Type "+name);return value;
 }
 private static Assembly Module(object input,string name) {
  Assembly value=InputBase(input,name) as Assembly;
  Require(value!=null,"metadata Assembly "+name);return value;
 }
 private static void Signature(MethodInfo m,Type owner,string name,Type result,bool publicly,Type[] parameters) {
  Require(Object.ReferenceEquals(m.DeclaringType,owner) && m.Name==name && m.IsStatic &&
   m.IsPublic==publicly && (publicly || m.IsPrivate) && !m.IsGenericMethod && !m.ContainsGenericParameters &&
   m.ReturnType==result,"actual method identity/ABI "+name);
  ParameterInfo[] p=m.GetParameters();Require(p.Length==parameters.Length,"parameter count "+name);
  for(int i=0;i<p.Length;i++)Require(p[i].ParameterType==parameters[i] && p[i].IsOut==(parameters[i].IsByRef),"parameter ABI "+name);
 }
 private static MethodInfo Getter(object input) {
  MethodInfo getter=Method(input,"getter");
  Signature(getter,typeof(D19ReadViewBoxIdentityR23),"UnsafeGetter",typeof(object),true,new Type[]{typeof(PSObject)});
  return getter;
 }
 private static ScriptBlock Script(object input) {
  ScriptBlock script=InputBase(input,"script") as ScriptBlock;
  Require(script!=null && script.GetType()==typeof(ScriptBlock),"original script kind");return script;
 }
 private static PSObject Make(int kind,ScriptBlock script,MethodInfo getter) {
  string original=new string(new char[]{'b','a','s','e'});
  PSObject wrapper=new PSObject(original);
  Require(wrapper.GetType()==typeof(PSObject) && Object.ReferenceEquals(wrapper.ImmediateBaseObject,original),"fresh original wrapper");
  if(kind==4) { wrapper.TypeNames.Insert(0,"foreign");Require(wrapper.TypeNames[0]=="foreign","type-name mutation");return wrapper; }
  PSMemberInfo member;
  if(kind==0)member=new PSScriptProperty("x",script);
  else if(kind==1)member=new PSNoteProperty("x",(long)1);
  else if(kind==2)member=new PSAliasProperty("x","Length");
  else { var code=new PSCodeProperty("x",getter);Require(Object.ReferenceEquals(code.GetterCodeReference,getter),"code exact getter");member=code; }
  wrapper.Members.Add(member);
  PSMemberInfo stored=wrapper.Members["x"];
  Require(stored!=null && stored.GetType()==member.GetType() && stored.Name=="x","instance member kind/name retained");
  if(kind==3)Require(Object.ReferenceEquals(((PSCodeProperty)stored).GetterCodeReference,getter),"stored code exact getter");
  return wrapper;
 }
 public static bool ConstructorSmoke(object scriptInput,object getterInput) {
  ScriptBlock script=Script(scriptInput);MethodInfo getter=Getter(getterInput);
  for(int i=0;i<5;i++) { PSObject wrapper=Make(i,script,getter);Require(wrapper.GetType()==typeof(PSObject),"smoke exact wrapper"); }
  Require(GetterCalls==0,"smoke zero CodeProperty executions");return true;
 }
 public static bool UnsafeReadView(object methodInput,object typeInput,object assemblyInput,object scriptInput,object getterInput) {
  MethodInfo method=Method(methodInput,"ReadView");Type source=Kind(typeInput,"Source");Assembly module=Module(assemblyInput,"Source");
  Require(Object.ReferenceEquals(source.Assembly,module) && source!=typeof(D19ReadViewBoxIdentityR23),"independent actual Source type/assembly");
  Signature(method,source,"ReadView",typeof(bool),false,new Type[]{typeof(object),typeof(object).MakeByRefType()});
  ScriptBlock script=Script(scriptInput);MethodInfo getter=Getter(getterInput);
  for(int i=0;i<5;i++) {
   PSObject wrapper=Make(i,script,getter);object original=wrapper.ImmediateBaseObject;
   object[] args=new object[]{wrapper,null};Require(Object.ReferenceEquals(args[0],wrapper),"original unsafe input");
   object result=method.Invoke(null,args);
   Require(result!=null && result.GetType()==typeof(bool) && !(bool)result,"actual unsafe Source refusal");
   Require(Object.ReferenceEquals(args[0],wrapper) && Object.ReferenceEquals(wrapper.ImmediateBaseObject,original),"unsafe wrapper/base retained");
  }
  Require(GetterCalls==0,"unsafe zero CodeProperty executions");return true;
 }
 private static object Fixture(object methodInput,object fixtureInput,object fixtureAssemblyInput,object sourceInput,object sourceAssemblyInput,object supportInput,bool check) {
  MethodInfo method=Method(methodInput,"fixture");Type fixture=Kind(fixtureInput,"fixture");Assembly fixtureAssembly=Module(fixtureAssemblyInput,"fixture");
  Type source=Kind(sourceInput,"Source");Assembly sourceAssembly=Module(sourceAssemblyInput,"Source");MethodInfo support=Method(supportInput,"support");
  Require(Object.ReferenceEquals(fixture.Assembly,fixtureAssembly) && Object.ReferenceEquals(source.Assembly,sourceAssembly) &&
   fixture!=source && fixtureAssembly!=sourceAssembly && fixture!=typeof(D19ReadViewBoxIdentityR23),"independent fixture/Source identities");
  Require(support.Name=="Supports" && support.IsPublic && support.IsStatic && !support.IsGenericMethod && !support.ContainsGenericParameters &&
   Object.ReferenceEquals(support.DeclaringType,source) && support.ReturnType==typeof(bool),"actual Source support identity");
  ParameterInfo[] p=support.GetParameters();Require(p.Length==1 && p[0].ParameterType==typeof(object) && !p[0].IsOut,"actual Source support ABI");
  Signature(method,fixture,check?"Check":"EmptyTemplate",check?typeof(object[]):typeof(void),true,
   check?new Type[]{typeof(Type),typeof(MethodInfo)}:new Type[]{typeof(Type)});
  object result=method.Invoke(null,check?new object[]{source,support}:new object[]{source});
  if(check)Require(result!=null && result.GetType()==typeof(object[]),"actual Check object[] result");
  else Require(result==null,"actual EmptyTemplate void result");
  return result;
 }
 public static object[] FixtureCheck(object method,object fixture,object fixtureAssembly,object source,object sourceAssembly,object support) {
  return (object[])Fixture(method,fixture,fixtureAssembly,source,sourceAssembly,support,true);
 }
 public static bool FixtureEmpty(object method,object fixture,object fixtureAssembly,object source,object sourceAssembly,object support) {
  Fixture(method,fixture,fixtureAssembly,source,sourceAssembly,support,false);return true;
 }
 public static bool MetadataSmoke(object typeInput,object assemblyInput,object getterInput,object counterInput) {
  Type type=Kind(typeInput,"smoke");Assembly module=Module(assemblyInput,"smoke");MethodInfo getter=Method(getterInput,"smoke getter");
  FieldInfo counter=InputBase(counterInput,"smoke counter") as FieldInfo;
  Require(Object.ReferenceEquals(type,typeof(D19ReadViewBoxIdentityR23)) && Object.ReferenceEquals(type.Assembly,module),"metadata smoke independent type/assembly");
  Signature(getter,type,"UnsafeGetter",typeof(object),true,new Type[]{typeof(PSObject)});
  Require(counter!=null && Object.ReferenceEquals(counter.DeclaringType,type) && counter.Name=="GetterCalls" && counter.IsPublic && counter.IsStatic && counter.FieldType==typeof(long),"metadata smoke exact field");
  Require(GetterCalls==0,"metadata smoke zero CodeProperty executions");return true;
 }
 public static object InvokeSource(object methodInput,object typeInput,object assemblyInput,object argumentsInput) {
  MethodInfo method=Method(methodInput,"typed Source");Type source=Kind(typeInput,"typed Source");Assembly module=Module(assemblyInput,"typed Source");
  Require(Object.ReferenceEquals(source.Assembly,module) && source!=typeof(D19ReadViewBoxIdentityR23),"typed independent actual Source type/assembly");
  Type[] types;Type result;int[] metadata;
  if(method.Name=="Field") { types=new Type[]{typeof(Type),typeof(string),typeof(Type),typeof(bool),typeof(bool),typeof(bool)};result=typeof(FieldInfo);metadata=new int[]{0,2}; }
  else if(method.Name=="InertMembers") { types=new Type[]{typeof(object),typeof(object),typeof(Type),typeof(int).MakeByRefType()};result=typeof(bool);metadata=new int[]{2}; }
  else if(method.Name=="HiddenDiagnostic") { types=new Type[]{typeof(object),typeof(object),typeof(object),typeof(int),typeof(FieldInfo),typeof(FieldInfo)};result=typeof(int);metadata=new int[]{4,5}; }
  else throw new InvalidOperationException("R28 only three typed Source test methods");
  Signature(method,source,method.Name,result,false,types);
  object container=InputBase(argumentsInput,"controlled argument container");
  Require(container!=null && container.GetType()==typeof(object[]),"exact original object[] container");
  object[] original=(object[])container;Require(original.Length==types.Length,"typed Source argument count");
  object[] snapshot=(object[])original.Clone();object[] working=(object[])original.Clone();
  foreach(int index in metadata) {
   object value=InputBase(original[index],"typed metadata slot");
   if(types[index]==typeof(Type))Require(value is Type,"typed Type metadata");
   else Require(value==null || value is FieldInfo,"nullable typed FieldInfo metadata");
   working[index]=value;
  }
  ParameterInfo[] parameters=method.GetParameters();
  try { return method.Invoke(null,working); }
  finally {
   for(int i=0;i<original.Length;i++) {
    if(parameters[i].IsOut)original[i]=working[i];
    else Require(Object.ReferenceEquals(original[i],snapshot[i]),"typed Source original non-out slot/reference retained");
   }
   Require(Object.ReferenceEquals(container,original),"typed Source original container retained");
  }
 }
 public static bool DeadWeakSame(object methodInput,object typeInput,object assemblyInput) {
  MethodInfo method=Method(methodInput,"WeakSame");Type source=Kind(typeInput,"WeakSame");Assembly module=Module(assemblyInput,"WeakSame");
  Require(Object.ReferenceEquals(source.Assembly,module) && source!=typeof(D19ReadViewBoxIdentityR23),"WeakSame independent actual Source type/assembly");
  Signature(method,source,"WeakSame",typeof(bool),false,new Type[]{typeof(object),typeof(object)});
  var captured=new WeakReference<System.Management.Automation.Runspaces.TypeTable>(null);object current=new object();
  System.Management.Automation.Runspaces.TypeTable target;
  Require(captured!=null && captured.GetType()==typeof(WeakReference<System.Management.Automation.Runspaces.TypeTable>) &&
   !captured.TryGetTarget(out target) && target==null && current!=null && current.GetType()==typeof(object),"nonnull exact dead weak capture and fresh current");
  object[] args=new object[]{captured,current};object answer=method.Invoke(null,args);
  Require(answer!=null && answer.GetType()==typeof(bool) && (bool)answer,"actual Source dead WeakSame positive");
  Require(Object.ReferenceEquals(args[0],captured) && Object.ReferenceEquals(args[1],current),"actual WeakSame original input references retained");
  return true;
 }
 public static string FieldRefusal(object methodInput,object typeInput,object assemblyInput,object argumentsInput) {
  MethodInfo method=Method(methodInput,"Field refusal");Type source=Kind(typeInput,"Field refusal");Assembly module=Module(assemblyInput,"Field refusal");
  Require(Object.ReferenceEquals(source.Assembly,module) && source!=typeof(D19ReadViewBoxIdentityR23),"Field refusal independent actual Source");
  Type[] types=new Type[]{typeof(Type),typeof(string),typeof(Type),typeof(bool),typeof(bool),typeof(bool)};
  Signature(method,source,"Field",typeof(FieldInfo),false,types);
  object container=InputBase(argumentsInput,"Field refusal container");
  Require(container!=null && container.GetType()==typeof(object[]),"Field refusal exact original object[]");
  object[] original=(object[])container;
  Require(original.Length==6,"Field refusal exact six arguments");
  Require(Kind(original[0],"Field refusal owner")==typeof(PSObject) && Kind(original[2],"Field refusal storage")==typeof(object),"Field refusal exact Type metadata");
  Require(original[1]!=null && original[1].GetType()==typeof(string) && (string)original[1]=="__missing_layout__","Field refusal literal missing name");
  for(int i=3;i<6;i++)Require(original[i]!=null && original[i].GetType()==typeof(bool) && !(bool)original[i],"Field refusal actual false flags");
  Require(GetterCalls==0,"Field refusal CodeProperty counter before");
  object[] snapshot=(object[])original.Clone();Exception cause=null;TargetInvocationException caught=null;
  try { InvokeSource(methodInput,typeInput,assemblyInput,argumentsInput); }
  catch(TargetInvocationException wrapper) { caught=wrapper;cause=wrapper.InnerException; }
  finally {
   Require(Object.ReferenceEquals(container,original),"Field refusal original container retained");
   for(int i=0;i<6;i++)Require(Object.ReferenceEquals(original[i],snapshot[i]),"Field refusal original non-out carrier retained");
   Require(GetterCalls==0,"Field refusal CodeProperty counter after");
  }
  string evidence="R29 Field refusal actual wrapper="+(caught==null?"<none>":caught.GetType().FullName)+" inner="+(cause==null?"<none>":cause.GetType().FullName);
  bool exact=caught!=null && caught.GetType()==typeof(TargetInvocationException) && cause!=null && cause.GetType()==typeof(InvalidOperationException);
  if(exact) {
   string actual=cause.Message;evidence+=" message="+(actual.Length<=512?actual:actual.Substring(0,512)+" [truncated]");
  }
  Require(exact,evidence);
  Require(cause.InnerException==null && cause.Message=="Fixed field shape",evidence);
  MethodInfo target=cause.TargetSite as MethodInfo;
  Require(target!=null && Object.ReferenceEquals(target.DeclaringType,source) && Object.ReferenceEquals(target.DeclaringType.Assembly,module),evidence+" target declaring Source");
  Signature(target,source,"Field",typeof(FieldInfo),false,types);
  Require(target.Module.Equals(method.Module) && target.MetadataToken==method.MetadataToken,evidence+" target module/token");
  return cause.Message;
 }
}
'@
$r26Types=@(Add-Type -TypeDefinition $r23Code -PassThru -ErrorAction Stop -WarningAction Stop)
Need ($r26Types.Length -eq 1 -and $r26Types[0] -is [type] -and $r26Types[0].FullName -ceq 'D19ReadViewBoxIdentityR23') 'R26 sole early helper Type'
$script:R26HelperType=$r26Types[0];$script:R26HelperAssembly=$r26Types[0].Assembly
$script:R26Verify=$r26Types[0].GetMethod('Verify',[Reflection.BindingFlags]'Public,Static',$null,[type[]]@([object],[object],[object]),$null)
$script:R26Getter=$r26Types[0].GetMethod('UnsafeGetter',[Reflection.BindingFlags]'Public,Static',$null,[type[]]@([System.Management.Automation.PSObject]),$null)
$script:R26Counter=$r26Types[0].GetField('GetterCalls',[Reflection.BindingFlags]'Public,Static')
Need ($null -ne $script:R26Verify -and [object]::ReferenceEquals($script:R26Verify.DeclaringType,$script:R26HelperType) -and $script:R26Verify.IsPublic -and $script:R26Verify.IsStatic -and -not $script:R26Verify.IsGenericMethod -and $script:R26Verify.ReturnType -eq [bool]) 'R26 early Verify exact object ABI'
Need ($null -ne $script:R26Getter -and [object]::ReferenceEquals($script:R26Getter.DeclaringType,$script:R26HelperType) -and $script:R26Getter.IsPublic -and $script:R26Getter.IsStatic -and -not $script:R26Getter.IsGenericMethod -and $script:R26Getter.ReturnType -eq [object]) 'R26 public static nonvoid getter'
$r26Parameters=$script:R26Getter.GetParameters()
Need ($r26Parameters.Length -eq 1 -and $r26Parameters[0].ParameterType -eq [System.Management.Automation.PSObject] -and -not $r26Parameters[0].IsOut) 'R26 getter exact one PSObject argument'
Need ($null -ne $script:R26Counter -and [object]::ReferenceEquals($script:R26Counter.DeclaringType,$script:R26HelperType) -and $script:R26Counter.IsPublic -and $script:R26Counter.IsStatic -and $script:R26Counter.FieldType -eq [long]) 'R26 exact execution counter field'
Same ($script:R26Counter.GetValue($null)) ([long]0) 'R26 getter counter initially zero'
$r26Property=[System.Management.Automation.PSCodeProperty]::new('x',$script:R26Getter)
Need ($r26Property.GetType() -eq [System.Management.Automation.PSCodeProperty] -and [object]::ReferenceEquals($r26Property.GetterCodeReference,$script:R26Getter)) 'R26 public constructor retains exact valid getter'
Same ($script:R26Counter.GetValue($null)) ([long]0) 'R26 constructor executes zero supplied getters'
Need ($script:KManagedSourceAccepted -cne $true -and $null -eq $script:KCatalogCache -and $null -eq $script:KCatalogOrigin -and $null -eq $script:KManagedCache -and $null -eq $script:KManagedOrigin) 'Preacceptance both backends untouched after R26 helper constructor'
$script:R26Bindings=[object[]]::new(5);$script:R26Bindings[0]=$script:R26HelperType;$script:R26Bindings[1]=$script:R26HelperAssembly;$script:R26Bindings[2]=$script:R26Verify;$script:R26Bindings[3]=$script:R26Getter;$script:R26Bindings[4]=$script:R26Counter
$script:viewGetterCalls=[long]0
$script:R27Script={$script:viewGetterCalls++;throw 'Must not call custom getter'}
$r27Methods=[object[]]::new(4)
$r27Signatures=@(@('ConstructorSmoke',2,[bool]),@('UnsafeReadView',5,[bool]),@('FixtureCheck',6,[object[]]),@('FixtureEmpty',6,[bool]))
for($i=0;$i -lt 4;$i++){
 $signature=$r27Signatures[$i];$types=[type[]]::new($signature[1]);for($j=0;$j -lt $types.Length;$j++){$types[$j]=[object]}
 $method=$script:R26HelperType.GetMethod($signature[0],[Reflection.BindingFlags]'Public,Static',$null,$types,$null)
 Need ($null -ne $method -and [object]::ReferenceEquals($method.DeclaringType,$script:R26HelperType) -and $method.IsPublic -and $method.IsStatic -and -not $method.IsGenericMethod -and -not $method.ContainsGenericParameters -and $method.ReturnType -eq $signature[2]) 'R27 early exact bridge metadata'
 $parameters=$method.GetParameters();Need ($parameters.Length -eq $types.Length) 'R27 bridge arity';foreach($parameter in $parameters){Need ($parameter.ParameterType -eq [object] -and -not $parameter.IsOut) 'R27 object input ABI'}
 $r27Methods[$i]=$method
}
$script:R27Smoke=$r27Methods[0];$script:R27Unsafe=$r27Methods[1];$script:R27Check=$r27Methods[2];$script:R27Empty=$r27Methods[3]
$script:R27Bindings=[object[]]::new(5);$script:R27Bindings[0]=$script:R27Script;for($i=0;$i -lt 4;$i++){$script:R27Bindings[$i+1]=$r27Methods[$i]}
$r27SmokeArgs=[object[]]::new(2);$r27SmokeArgs[0]=$script:R27Script;$r27SmokeArgs[1]=$script:R26Getter
Same (K-CatalogInvoke $script:R27Smoke $r27SmokeArgs) $true 'R27 constructor-only CLR member and type-name smoke'
Same $script:viewGetterCalls ([long]0) 'R27 smoke zero supplied ScriptProperty executions'
Same ($script:R26Counter.GetValue($null)) ([long]0) 'R27 smoke zero supplied CodeProperty executions'
Need ($script:KManagedSourceAccepted -cne $true -and $null -eq $script:KCatalogCache -and $null -eq $script:KCatalogOrigin -and $null -eq $script:KManagedCache -and $null -eq $script:KManagedOrigin) 'Preacceptance both backends untouched after R27 constructor smoke'
$r28Methods=[object[]]::new(3);$r28Signatures=@(@('MetadataSmoke',4,[bool]),@('InvokeSource',4,[object]),@('DeadWeakSame',3,[bool]))
for($i=0;$i -lt 3;$i++){
 $signature=$r28Signatures[$i];$types=[type[]]::new($signature[1]);for($j=0;$j -lt $types.Length;$j++){$types[$j]=[object]}
 $method=$script:R26HelperType.GetMethod($signature[0],[Reflection.BindingFlags]'Public,Static',$null,$types,$null)
 Need ($null -ne $method -and [object]::ReferenceEquals($method.DeclaringType,$script:R26HelperType) -and $method.IsPublic -and $method.IsStatic -and -not $method.IsGenericMethod -and -not $method.ContainsGenericParameters -and $method.ReturnType -eq $signature[2]) 'R28 early exact bridge metadata'
 $parameters=$method.GetParameters();Need ($parameters.Length -eq $types.Length) 'R28 bridge arity';foreach($parameter in $parameters){Need ($parameter.ParameterType -eq [object] -and -not $parameter.IsOut) 'R28 object input ABI'}
 $r28Methods[$i]=$method
}
$script:R28Smoke=$r28Methods[0];$script:R28Invoke=$r28Methods[1];$script:R28Weak=$r28Methods[2]
$script:R28Bindings=[object[]]::new(3);for($i=0;$i -lt 3;$i++){$script:R28Bindings[$i]=$r28Methods[$i]}
$r28SmokeArgs=[object[]]::new(4);$r28SmokeArgs[0]=$script:R26HelperType;$r28SmokeArgs[1]=$script:R26HelperAssembly;$r28SmokeArgs[2]=$script:R26Getter;$r28SmokeArgs[3]=$script:R26Counter
Same (K-CatalogInvoke $script:R28Smoke $r28SmokeArgs) $true 'R28 metadata-only Type MethodInfo FieldInfo smoke'
Same $script:viewGetterCalls ([long]0) 'R28 smoke zero supplied ScriptProperty executions'
Same ($script:R26Counter.GetValue($null)) ([long]0) 'R28 smoke zero supplied CodeProperty executions'
Need ($script:KManagedSourceAccepted -cne $true -and $null -eq $script:KCatalogCache -and $null -eq $script:KCatalogOrigin -and $null -eq $script:KManagedCache -and $null -eq $script:KManagedOrigin) 'Preacceptance both backends untouched after R28 metadata smoke'
$script:R29Field=$script:R26HelperType.GetMethod('FieldRefusal',[Reflection.BindingFlags]'Public,Static')
Need ($script:R29Field -is [Reflection.MethodInfo] -and [object]::ReferenceEquals($script:R29Field.DeclaringType,$script:R26HelperType) -and $script:R29Field.IsPublic -and $script:R29Field.IsStatic -and -not $script:R29Field.IsGenericMethod -and $script:R29Field.ReturnType -eq [string]) 'R29 exact early FieldRefusal metadata ABI'
$r29Parameters=$script:R29Field.GetParameters();Need ($r29Parameters.Length -eq 4) 'R29 object4 metadata'
foreach($r29Parameter in $r29Parameters){Need ($r29Parameter.ParameterType -eq [object] -and -not $r29Parameter.IsOut) 'R29 exact object metadata'}
$script:R29Bindings=[object[]]::new(1);$script:R29Bindings[0]=$script:R29Field
Same ($script:R26Counter.GetValue($null)) ([long]0) 'R29 early metadata-only counter0'
$r30Index=@{'psbase'=0;'psadapted'=1;'psextended'=2;'psobject'=3;'pstypenames'=4}
Need ($r30Index.GetType() -eq [Collections.Hashtable] -and $r30Index.Count -eq 5) 'R30 owned smoke index'
$r30Names=@('psbase','psadapted','psextended','psobject','pstypenames');for($r30I=0;$r30I -lt 5;$r30I++){$r30Value=$r30Index[$r30Names[$r30I]];Need ($r30Value -is [int] -and $r30Value.GetType() -eq [int] -and $r30Value -eq $r30I) 'R30 literal key actual Int32 smoke'}
'''


def d19_r32_local_contract():
    return r'''
function R31Check([string]$r31Phase){
 if($d13Mode -cnotin @('faithful-code-equivalent','table-list-divergence')){return}
 if($r31Phase -ceq 'undo' -and $d13Undo.Count -eq 0){return}
 $r31Position=$d13Index['pstypenames'];$r31Saved=$d13Undo[$d13Undo.Count-1]
 Need ([object]::ReferenceEquals($d8LFields[0].GetValue($d13List),$d13Items) -and $d13Items[$r31Position].GetType() -eq [Collections.DictionaryEntry]) ('R31 original backing/box '+$r31Phase)
 Need ($d13EF[0].DeclaringType -eq [Collections.DictionaryEntry] -and $d13EF[0].Name -ceq '_key' -and $d13EF[0].FieldType -eq [object] -and $d13EF[1].DeclaringType -eq [Collections.DictionaryEntry] -and $d13EF[1].Name -ceq '_value' -and $d13EF[1].FieldType -eq [object] -and [object]::ReferenceEquals($d13Field,$d13EF[1])) ('R31 exact entry metadata '+$r31Phase)
 Need ([object]::ReferenceEquals($d13Items[$r31Position],$d13Entries[$r31Position]) -and [object]::ReferenceEquals($d13Items[$r31Position],$d13FieldArgs[0]) -and [object]::ReferenceEquals($d13Items[$r31Position],$d13SetArgs[0]) -and [object]::ReferenceEquals($r31Saved[1],$d13FieldArgs) -and [object]::ReferenceEquals($d13Items[$r31Position],$r31Saved[1][0])) ('R31 original box through setter/undo '+$r31Phase)
 Need ([object]::ReferenceEquals($d13EF[0].GetValue($d13Items[$r31Position]),$d13Keys[$r31Position]) -and [object]::ReferenceEquals($r31Saved[2][0],$d13Code)) ('R31 original key/old value '+$r31Phase)
 if($r31Phase -ceq 'after'){
  Need ([object]::ReferenceEquals($d13EF[1].GetValue($d13Items[$r31Position]),$d13New)) 'R31 actual original entry changed'
  if($d13Mode -ceq 'faithful-code-equivalent'){
   Need ([object]::ReferenceEquals($d13Table[$d13Keys[$r31Position]],$d13New)) 'R31 faithful actual table/list equivalence'
   $r31NewMember=[object[]]::new(1);$r31NewMember[0]=$d13Equivalent;$r31OldMember=[object[]]::new(1);$r31OldMember[0]=$d13Code
   foreach($r31Field in $d13MFields){
    if($r31Field.Name -ceq 'instance'){Need ([object]::ReferenceEquals($d13Read.Invoke($r31Field,$r31NewMember),$d13Owner[0])) 'R31 copied original owner identity'}
    elseif($r31Field.FieldType.IsValueType){Need ($d13Read.Invoke($r31Field,$r31NewMember) -ceq $d13Read.Invoke($r31Field,$r31OldMember)) 'R31 copied field scalar'}
    else{Need ([object]::ReferenceEquals($d13Read.Invoke($r31Field,$r31NewMember),$d13Read.Invoke($r31Field,$r31OldMember))) 'R31 copied field reference'}
   }
  }else{Need ([object]::ReferenceEquals($d13Table[$d13Keys[$r31Position]],$d13Code) -and -not [object]::ReferenceEquals($d13New,$d13Code)) 'R31 intentional table/list divergence'}
 }else{
  Need ([object]::ReferenceEquals($d13EF[1].GetValue($d13Items[$r31Position]),$d13Code)) ('R31 original entry value '+$r31Phase)
  if($r31Phase -ceq 'undo'){Need ([object]::ReferenceEquals($d13Table[$d13Keys[$r31Position]],$d13Code)) 'R31 complete original table/list restored'}
 }
}
function R31View([bool]$r31Return){
 $r31Output=[object]::ReferenceEquals($d13Input[1],$d13Key[0]);$r31Slot=$d7Slot.GetValue($null)
 Need ($r31Return -eq $d13Positive -and $r31Output) ('Sequential source-owned graph {0}/{1}/{2}; returned={3}; output={4}; stage={5}; no within-call race or origin claim' -f $d13Kind.FullName,$d13Wrapped,$d13Mode,$r31Return,$r31Output,$r31Slot[5])
}
Need ((Get-Item Function:R31Check -ErrorAction Stop).ScriptBlock -is [scriptblock] -and (Get-Item Function:R31View -ErrorAction Stop).ScriptBlock -is [scriptblock]) 'R32 actual local catalog guard definitions'
'''


def d19_readview_unsafe_members():
    return r'''
$r27UnsafeArgs=[object[]]::new(5);$r27UnsafeArgs[0]=$viewMethod;$r27UnsafeArgs[1]=$script:KCatalogCache.type;$r27UnsafeArgs[2]=$script:KCatalogCache.assembly;$r27UnsafeArgs[3]=$script:R27Script;$r27UnsafeArgs[4]=$script:R26Getter
Same (K-CatalogInvoke $script:R27Unsafe $r27UnsafeArgs) $true 'Instance Script/Note/Alias/Code and type-name storage decline in CLR before member execution'
Same $script:viewGetterCalls ([long]0) 'Actual zero custom ScriptProperty executions'
Same ($script:R26Counter.GetValue($null)) ([long]0) 'Zero supplied CodeProperty getter executions'
Need ([object]::ReferenceEquals($script:KCatalogCache,$r23Cache) -and [object]::ReferenceEquals($script:KCatalogOrigin,$r23Origin) -and [object]::ReferenceEquals($script:KCatalogManifestContract,$r23Manifest) -and [object]::ReferenceEquals($script:KCatalogCache.type,$r23Type) -and [object]::ReferenceEquals($script:KCatalogCache.assembly,$r23Assembly)) 'R27 unsafe actual Source cache origin provenance retained'
Same $script:KManagedSourceAccepted $r23Accepted 'R27 unsafe Source acceptance retained'
Same $r23Cache.source $r23Source 'R27 unsafe Source literal retained'
Same $r23Cache.source_hash $r23SourceHash 'R27 unsafe rendered hash retained'
Same $r23Cache.template_hash $r23TemplateHash 'R27 unsafe template hash retained'
'''


def d19_clr_fixture_calls():
    return r'''
$fixtureArgs=[object[]]::new(6);$fixtureArgs[0]=$fixture.check;$fixtureArgs[1]=$fixture.type;$fixtureArgs[2]=$fixture.assembly;$fixtureArgs[3]=$script:KCatalogCache.type;$fixtureArgs[4]=$script:KCatalogCache.assembly;$fixtureArgs[5]=$script:KCatalogCache.support
$fixtureResults=K-CatalogInvoke $script:R27Check $fixtureArgs
Same $fixtureResults @('custom_comparer0','adapter9_callbacks0','live_foreign_capture_reason3','cycle_reason10','actual_depth9_reason9','acyclic_positive') 'Actual bounded custom comparer/adapter/capture/cycle/depth controls'
Need ($fixtureResults[4] -ceq 'actual_depth9_reason9') 'Nine wrappers bounded unsupported, no BaseObject'
$emptyArgs=[object[]]::new(6);$emptyArgs[0]=$fixture.empty;$emptyArgs[1]=$fixture.type;$emptyArgs[2]=$fixture.assembly;$emptyArgs[3]=$script:KCatalogCache.type;$emptyArgs[4]=$script:KCatalogCache.assembly;$emptyArgs[5]=$script:KCatalogCache.support;Same (K-CatalogInvoke $script:R27Empty $emptyArgs) $true 'Actual original EmptyTemplate bridge'
Need ([object]::ReferenceEquals($script:KCatalogCache.type,$catalogTypeBefore) -and [object]::ReferenceEquals($script:KCatalogCache.assembly,$catalogAssemblyBefore) -and [object]::ReferenceEquals($script:KCatalogOrigin,$catalogOriginBefore)) 'Production catalog Type/Assembly/origin preserved after test-only setup'
'''


def d19_typed_source_field_weak():
    return r'''
$viewField=$script:KCatalogCache.type.GetMethod('Field',[Reflection.BindingFlags]'NonPublic,Static')
$missingArgs=[object[]]@([PSObject],'__missing_layout__',[object],$false,$false,$false)
$r29Fields=[object[]]::new(4);$r29Fields[0]=$viewField;$r29Fields[1]=$script:KCatalogCache.type;$r29Fields[2]=$script:KCatalogCache.assembly;$r29Fields[3]=$missingArgs
try{$missingError=K-CatalogInvoke $script:R29Field $r29Fields}catch{$r29Display=[string]$_.Exception.Message;if($r29Display.Length -gt 1024){$r29Display=$r29Display.Substring(0,1024)+' [diagnostic truncated]'};throw ('R29 actual Field refusal rejected; outer='+$_.Exception.GetType().FullName+' message='+$r29Display)}
Same $missingError 'Fixed field shape' 'Unknown layout has fixed pure refusal; no engine writes'
Same $script:viewGetterCalls ([long]0) 'R29 Field refusal zero ScriptProperty callbacks'
Same ($script:R26Counter.GetValue($null)) ([long]0) 'R29 Field refusal zero CodeProperty callbacks'
$weakMethod=$script:KCatalogCache.type.GetMethod('WeakSame',[Reflection.BindingFlags]'NonPublic,Static')
$r28WeakArgs=[object[]]::new(3);$r28WeakArgs[0]=$weakMethod;$r28WeakArgs[1]=$script:KCatalogCache.type;$r28WeakArgs[2]=$script:KCatalogCache.assembly
Same (K-CatalogInvoke $script:R28Weak $r28WeakArgs) $true 'Dead captured reference uses existing current table only'
'''

def d19_r35_current_paths(k):
    q=k.BASE+'/s28-f19-root/'
    return (q+'transition-cut/independent-expected-paid328.json',q+'residual09/v2/paid328.json')+(q+'transition11/cut/independent-expected-paid329.json',q+'transition10/paid329.json')+(q+'transition12/cut/independent-expected-paid330.json',q+'transition11/paid330.json')+(q+'transition13/cut/independent-expected-paid331.json',q+'transition12/paid331.json')+(q+'transition14/cut/independent-expected-paid332.json',q+'transition13/paid332.json')+(q+'transition15/cut/independent-expected-paid333.json',q+'transition14/paid333.json')+(q+'transition16/cut/independent-expected-paid334.json',q+'transition15/paid334.json')+(q+'transition17/cut/independent-expected-paid335.json',q+'transition16/paid335.json')+(q+'transition18/cut/independent-expected-paid336.json',q+'transition17/paid336.json')+(q+'transition19/cut/independent-expected-paid337.json',q+'transition18/paid337.json')+(q+'transition20/cut/independent-expected-paid338.json',q+'transition19/paid338.json')+(q+'transition21/cut/independent-expected-paid339.json',q+'transition20/paid339.json')+(q+'transition22/cut/independent-expected-paid340.json',q+'transition21/paid340.json')+(q+'transition23/cut/independent-expected-paid341.json',q+'transition22/paid341.json')+(q+'transition24/cut/independent-expected-paid342.json',q+'transition23/paid342.json')+(q+'transition25/cut/independent-expected-paid343.json',q+'transition24/paid343.json')+(q+'transition26/cut/independent-expected-paid344.json',q+'transition25/paid344.json')+(q+'transition27/cut/independent-expected-paid345.json',q+'transition26/paid345.json')+(q+'transition28/cut/independent-expected-paid346.json',q+'transition27/paid346.json')+(q+'transition29/cut/independent-expected-paid347.json',q+'transition28/paid347.json')+(q+'transition30/cut/independent-expected-paid348.json',q+'transition29/paid348.json')+(q+'transition31/cut/independent-expected-paid349.json',q+'transition30/paid349.json')+(q+'transition32/cut/independent-expected-paid350.json',q+'transition31/paid350.json')+(q+'transition33/cut/independent-expected-paid351.json',q+'transition32/paid351.json')+(q+'transition34/cut/independent-expected-paid352.json',q+'transition33/paid352.json')+(q+'transition35/cut/independent-expected-paid353.json',q+'transition34/paid353.json')+(q+'transition36/cut/independent-expected-paid354.json',q+'transition35/paid354.json')+(q+'transition37/cut/independent-expected-paid355.json',q+'transition36/paid355.json')+(q+'transition38/cut/independent-expected-paid356.json',q+'transition37/paid356.json')+(q+'transition39/cut/independent-expected-paid357.json',q+'transition38/paid357.json')+(q+'transition40/cut/independent-expected-paid358.json',q+'transition39/paid358.json')+(q+'transition41/cut/independent-expected-paid359.json',q+'transition40/paid359.json')+(q+'transition42/cut/independent-expected-paid360.json',q+'transition41/paid360.json')+(q+'transition43/cut/independent-expected-paid361.json',q+'transition42/paid361.json')+(q+'transition36/paid353.json',)+tuple(q+'transition'+p for p in ('44/allocation-before.json', '44/original-body-before.json', '44/original-audit-before.json', '44/cut/independent-expected-paid363.json', '43/paid363-pre-spawn.json', '43/paid362.json', '45/allocation-before.json', '45/original-body-before.json', '45/original-audit-before.json', '45/cut/independent-expected-paid364.json', '44/paid364.json', '46/allocation-before.json', '46/original-body-before.json', '46/original-audit-before.json', '46/cut/independent-expected-paid365.json', '45/paid365.json', '47/allocation-before.json', '47/original-body-before.json', '47/original-audit-before.json', '47/cut/independent-expected-paid366.json', '46/paid366.json', '48/allocation-before.json', '48/original-body-before.json', '48/original-audit-before.json', '48/cut/independent-expected-paid367.json', '47/paid367.json', '49/allocation-before.json', '49/original-body-before.json', '49/original-audit-before.json', '49/cut/independent-expected-paid368.json', '48/paid368.json', '50/allocation-before.json', '50/original-body-before.json', '50/original-audit-before.json', '50/cut/independent-expected-paid369.json', '49/paid369.json', '51/allocation-before.json', '51/original-body-before.json', '51/original-audit-before.json', '51/cut/independent-expected-paid370.json', '50/paid370.json', '52/allocation-before.json', '52/original-body-before.json', '52/original-audit-before.json', '52/cut/independent-expected-paid371.json', '51/paid371.json', '53/allocation-before.json', '53/original-body-before.json', '53/original-audit-before.json', '53/cut/independent-expected-paid372.json', '52/paid372.json', '54/allocation-before.json', '54/original-body-before.json', '54/original-audit-before.json', '54/cut/independent-expected-paid373.json', '53/paid373.json', '55/cut/p.json', '54/paid374.json', '56/p.json', '55/paid375.json'))+tuple('verification/banc'+p for p in ('57/p.json', '57/t.json', '57/a.json', '57/b.json', '57/u.json', '58/p.json', '58/t.json', '58/a.json', '58/b.json', '58/u.json', '59/p.json', '59/t.json', '59/a.json', '59/b.json', '59/u.json', '60/p.json', '60/t.json', '60/a.json', '60/b.json', '60/u.json', '61/p.json', '61/t.json', '61/a.json', '61/b.json', '61/u.json',*(f'6{n}/{p}.json'for n in'234'for p in'ptabu')))

def d19_r35_loader_entry():
    return r'''
$null=K-SourceFence (K-BoundedPin (K-Paths)[$zk])
Need ($script:KManagedSourceAccepted -ceq $true) 'Genuine Source re-entry accepted'
K-PinShape $script:KCatalogManifestContract (K-Paths).contract
Same $script:KCatalogManifestContract $script:KCurrentContract 'Genuine Source re-entry current manifest'
'''


def d19_r41_reference():
    return r'''function S41ReferenceControls{
$e=$script:R41[0];$c=$script:R41[1];$x=$null;try{K-Same 1 2 'S41'}catch{$x=$_.Exception.Message};Need ($x -ceq 'S41' -and $script:R41[0] -eq $e+1 -and $script:R41[1] -eq $c) 'S41 counts'
$script:S41Read=${function:K-Read};$rv=K-Read (K-BoundedPin (K-Paths).review)
try{foreach($bad in @($null,1,'bad','missing')){$script:S41Bad=$bad;function K-Read($p){$v=& $script:S41Read $p;if($p.path -ceq (K-Paths).contract){if($script:S41Bad -ceq 'missing'){$null=$v.Remove('comparison_profile')}else{$v.comparison_profile=$script:S41Bad}};return $v};$e=$script:R41[0];$f=$false;try{$null=K-Provenance $rv}catch{$f=$_.Exception.Message.StartsWith('Reference profile')};Need ($f -and $script:R41[0] -eq $e) 'S41 profile'}}finally{Set-Item Function:K-Read $script:S41Read};$null=K-Provenance $rv
}
'''
