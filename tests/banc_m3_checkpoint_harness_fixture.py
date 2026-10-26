"""Generated temporary-root operands. Synthetic costs never qualify the real ledger."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

from tests.banc_m3_checkpoint_fixture import exact_source_operand, observation_directory
from tools import banc_m3_checkpoint_harness_runtime as k
from tools import banc_m3_conserved_resource as r
from tools import banc_m3_contact_learning_admission as admission

_HISTORY_BASE = 'verification/banc-natural-gait-m3-20260919'


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
    r.exact(state["completed"], {k.BOOTSTRAP_CALL: 0, k.QUAL_CALLS[2]: 3, k.QUAL_CALLS[3]: 4}[q["call_id"]], "actual K progress before bootstrap/gate-B")
    r.exact(r.current_caps(ledger, admission.ROOT, generated_budget_limit),
            k.CAPS, "actual paired K parent caps")
    k.identity(q, token["request"], state["source"], admission.ROOT)
    r.exact(ledger["direct_science_calls"][q["call_id"]],
            {"state": "admitted", "mode": "generated", "request": token["request"], "reserved_s": q["limits"]["outer_s"]},
            "actual K parent admission; no rewritten token")
    r.exact(ledger["direct_science_workers"][q["call_id"]]["lock_token"], r.text_spec(r.SHARED_LOCK, admission.ROOT),
            "same exact live physical parent exclusion")
    return k.CAP_TRIPLES[{k.BOOTSTRAP_CALL: 0, k.QUAL_CALLS[2]: 3, k.QUAL_CALLS[3]: 4}[q["call_id"]]]


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True)
        r.d19_stage_fixture(self,admission.ROOT)
        # Fixed catalog staging authenticates every operand; it does not replay history.
        for pin in [{'path': _HISTORY_BASE+'/s28-f15b-root/terminal319-before.json', 'bytes': 394685, 'sha256': 'c8a74a40a0d2bfef0e630524508d819c2b106cbce06f6f2b93364f5e4b4b1482'}, {'path': _HISTORY_BASE+'/s28-f15-root/admitted318-before.json', 'bytes': 496266, 'sha256': 'ea814b9271554155b9f71673ac4d8c5a4e2a1fcb5a3cc5a20cdf57dd59cb405d'}, {'path': _HISTORY_BASE+'/s28-f15-root/settlement-pre-backup.json', 'bytes': 496266, 'sha256': 'ea814b9271554155b9f71673ac4d8c5a4e2a1fcb5a3cc5a20cdf57dd59cb405d'}, {'path': _HISTORY_BASE+'/s28-f15-root/settlement-replace-backup.json', 'bytes': 496266, 'sha256': 'ea814b9271554155b9f71673ac4d8c5a4e2a1fcb5a3cc5a20cdf57dd59cb405d'}, {'path': _HISTORY_BASE+'/s28-exec-precreate-cost.json', 'bytes': 2452, 'sha256': '359c0dd97f2baf9f063d9a26c72fcd5269c3a66665be7a4004fdf8cf2278abb4'}, {'path': _HISTORY_BASE+'/s28-exec-precreate-settlement.json', 'bytes': 2916, 'sha256': '79c2967af54a4a28d7dd580c98e068396f1141b61043361c64befaf72190df31'}, {'path': _HISTORY_BASE+'/s28-f15-root/settlement-owner-token.json', 'bytes': 212, 'sha256': 'c8ef54795ae8bc5fe801478578436c66cbc5b11c9b6995bbc5ad10a88236bf02'}, {'path': _HISTORY_BASE+'/s28-f15-root/actual-settlement-audit.json', 'bytes': 2459, 'sha256': '8cb8eeb7decead2e8db9dd4195e571a667d5b6457ff72165721e9d8fdb8b4401'}, {'path': _HISTORY_BASE+'/s28-f15-root/settlement-actual-tool-capture.json', 'bytes': 13865, 'sha256': '6c02d1939b022ced5f7b45f71712d1d97cae4bef54dc4284f93f58c8827f6f63'}, {'path': _HISTORY_BASE+'/s28-f14-root/full4-failure.json', 'bytes': 6976, 'sha256': '978021662924b9a1c15bfac7cc5699a58535d43eececd894d58b542d7395ff0c'}, {'path': _HISTORY_BASE+'/s28-f14-root/full4-actual-tool-capture.json', 'bytes': 6464, 'sha256': '72ccd3b9621fe0979c3d51737fc8d3f6c818548ed017e09165ee7502b307e247'}, {'path': _HISTORY_BASE+'/s28-f14-root/full4-request.json', 'bytes': 1371, 'sha256': 'bdd63628db50c52b0a6ebf9f0f769848d79bcd241558397a55084b25dbdec90b'}, {'path': _HISTORY_BASE+'/s28-f15b-root/before/verification/banc-natural-gait-m3-20260919/c5.json', 'bytes': 124922, 'sha256': '9d65539915dbe98563ae5599df9d91c422541411f51608bc5ebe14fab8655ed8'}, {'path': _HISTORY_BASE+'/s28-f15b-root/before/verification/banc-natural-gait-m3-20260919/s28-s1.json', 'bytes': 262142, 'sha256': 'b8f00cb6fe06b0fa611b10b62d60f39d5a8e130887767f9e6c791b0973e35bbc'}, {'path': _HISTORY_BASE+'/s28-f15b-root/before/verification/banc-natural-gait-m3-20260919/s28-r1.json', 'bytes': 246958, 'sha256': '06479a5a995aa0d0eb7a3b2a7d062506005ae5e2cfe6d2b37a74c9b27f5bca97'}, {'path': _HISTORY_BASE+'/s28-f15b-root/before/verification/banc-natural-gait-m3-20260919/s28-recovery12.json', 'bytes': 1869, 'sha256': '26d3727cb1ade441017cc5dfe98a474a740aec206e55dbb5aa0adedb8d0b9add'}, {'path': _HISTORY_BASE+'/s28-recovery10.json', 'bytes': 1903, 'sha256': '27f8b420a0f7cdb3447aac416cd2ac71cf499ee8b5ea719ff05c968fd6f1184e'}, {'path': _HISTORY_BASE+'/s28-f13-root/s1-before.json', 'bytes': 262140, 'sha256': '39b4910b21c51042a5d68f3da51061ce2d57ead51ec5ff4cdcf82554a6493ef0'}, {'path': _HISTORY_BASE+'/s28-f12-root/a318.json', 'bytes': 495558, 'sha256': '1221bf613896fd8f929b0dbfd93c1e4bc8e922f4c873e3869d5a7780d53c6d54'}, {'path': _HISTORY_BASE+'/s28-f14-root/a318.json', 'bytes': 495559, 'sha256': 'fadfb1604441f5c2cc481e15b07eeaf10a7a7529cedc8969bc496196ea221274'}, {'path': _HISTORY_BASE+'/s28-f12-root/actual-tool-return.json', 'bytes': 1846, 'sha256': '71341a5233ecc031b45876ab24d1f7e4305ed79c738aa19f3efa71dc6c250162'}, {'path': _HISTORY_BASE+'/s28-f13-root/c5-before.json', 'bytes': 74674, 'sha256': '3a87bc0f34b8a36ddd5153940f2b17fccbdd990b32459aea28ae92e51be58b0a'}, {'path': _HISTORY_BASE+'/h28.json', 'bytes': 261912, 'sha256': 'e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e'}, {'path': _HISTORY_BASE+'/s28-f12-root/actual-layout-compatibility-diagnostic.json', 'bytes': 1331, 'sha256': 'efc4694c9651f842832549809e1613b9900ab71004da098bb82dc24c29ee8a6d'}, {'path': _HISTORY_BASE+'/s28-catalog-leaves-300021e7370a4dc8b163ef22f0245f31.jsonl', 'bytes': 6393, 'sha256': 'c4bb7e972e5ed5950e554983344f5d34849008f9a2a9a984a874d1ff8795926e'}, {'path': _HISTORY_BASE+'/s28-f12-root/fast-counter-expectation.json', 'bytes': 2331, 'sha256': '99e971e5a752697e8ab86e9fbf461acd4052d1d27da3a5437f05db14fe050b5d'}, {'path': _HISTORY_BASE+'/s28-recovery9.json', 'bytes': 1902, 'sha256': 'a4cf49fb579aeb3114fd800f092080b5e6a54b8f71ae835121e8cd34a1d4a28d'}, {'path': _HISTORY_BASE+'/s28-f12-root/s1-before.json', 'bytes': 262140, 'sha256': '1cf30bee866a147de013d0023a2ea02d8734005178d145d7f06952f058ef4aba'}, {'path': _HISTORY_BASE+'/s27-bootstrap-settled318-HOLD-20261009.json', 'sha256': '70175b4bc56c19412b64500343ebc72bc538a7b3776dd60733d43efeeb59476e', 'bytes': 495417}, {'path': _HISTORY_BASE+'/e28.json', 'bytes': 2443, 'sha256': 'e5e284cbaa1f67f1dd6e14bdf3fe74c5f902d751c387f6850700323c23609c93'}, {'path': _HISTORY_BASE+'/s28-f12-root/h28-before.json', 'bytes': 261912, 'sha256': 'e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e'}, {'path': _HISTORY_BASE+'/s27-actual.json', 'bytes': 3365, 'sha256': 'fcf5b1fa42a8348fa52f85e0259cc3b3554be5d96206fd5e77f4fcc54e01eb94'}, {'path': _HISTORY_BASE+'/s28-f11-root/actual-fast-warm-gate.json', 'bytes': 4977, 'sha256': '9bbf1418212eedfb12b4aaa57cac0d613c9b900b0f9e2a0b4a4d7dfa1f63c0ce'}, {'path': _HISTORY_BASE+'/s28-recovery11.json', 'bytes': 1882, 'sha256': '0098806a84d0c18cf606482710d15e73dbe21defae360badba88a6c909db4c26'}, {'path': _HISTORY_BASE+'/s28-f14-root/s1-before.json', 'bytes': 262140, 'sha256': 'f1b802af060e385f03955b5cbba5830cc9ee42ccf757b9800016a45afd7cabfd'}, {'path': _HISTORY_BASE+'/s28-f13-root/actual-tool-return.json', 'bytes': 2798, 'sha256': 'e76c924ab89dd27752d2f8a1afe68bd6839f363471886c073b3b14fce0d5d2c0'}, {'path': _HISTORY_BASE+'/s28-catalog-leaves-30944ee8fe084e2aac23de9d47e4de80.jsonl', 'bytes': 496, 'sha256': 'b3ebd169ebfc3f700ec42a0c3b7b9292f35b7561cb6f7d100c71bf3835669606'}, {'path': _HISTORY_BASE+'/s28-f14-root/c5-before.json', 'bytes': 97623, 'sha256': '9690bd2ec7a65d33432071d09d9050c6d949c92d6a20852fc92072737f7dcb41'}, {'path': _HISTORY_BASE+'/s28-f13-root/actual-compiler-type-refusal.json', 'bytes': 1968, 'sha256': '448f06a882edd1757895f90dfd4bd1c7bfad38f414fcdda3909a0eacfa34e407'}, {'path': _HISTORY_BASE+'/s28-f13-root/a318.json', 'bytes': 495559, 'sha256': 'fadfb1604441f5c2cc481e15b07eeaf10a7a7529cedc8969bc496196ea221274'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/h28-before.json', 'bytes': 261912, 'sha256': 'e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f12-root/actual-fast-warm-gate.json', 'bytes': 5074, 'sha256': '9c9348756e9dacf4ee53a320e50acc22d25c8fe9028419dccbc35819356d7b7e'}, {'bytes': 495417, 'sha256': '70175b4bc56c19412b64500343ebc72bc538a7b3776dd60733d43efeeb59476e', 'path': 'verification/banc-natural-gait-m3-20260919/s28-anchor318-before-recovery.json'}, {'path': 'verification/banc-natural-gait-m3-20260919/s27-s1.json', 'sha256': '35b6cf63018f61f3fde34d47045d16f5e3c97dc53bac64b1fe7814aed0aa61c6', 'bytes': 262139}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/h28-before.json', 'bytes': 261912, 'sha256': 'e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/actual-fast-warm-gate.json', 'bytes': 2818, 'sha256': '042051f83aaa9c347c72d943e76d0958cf22afeccb5fd3c929bf0241cd5f5a1f'}, {'path': 'docs/M3_28D15C.md', 'bytes': 10675, 'sha256': '1c1337578a8982b240bf743bb31ad421fde9cd71b54c7e5650b32295a7d43055'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-d15c-astra/review.md', 'bytes': 16179, 'sha256': '79b900d090ca7fe61f0322a3dff5099ebbdf50444d91cd113c6e709a3dd21290'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-d15c-astra/review.json', 'bytes': 14885, 'sha256': '69c41861683acb00b7f784fc2d41601c9d8dbe1611345e9337c672c1ca0895d3'}, {'path': 'docs/M3_28R15C.md', 'bytes': 9488, 'sha256': '578b086dfe080b3119ce75404abdcb7143d715a9251d60396ad6fe332bf27a33'}]:
            r.fixture_text_write(self.root,pin['path'],r.raw_pin(pin,admission.ROOT,524288 if pin['bytes']>262144 else 262144))
        for pin in (k.ANCHOR, r.POLICY_PIN, r.J_PIN, k.J_BODY, k.L_FREEZE, k.L_ROSTER, r.ROSTER_PIN, *k.S15_PINS.values()):
            self.copy(pin, k.reader_cap(pin["path"]))
        for row in k.data(k.S15_PINS["catalog"], self.root)["rows"]:
            self.copy(row["pin"], row["cap"])
        self.source = self.copy(r.text_spec(k.SOURCE_PATH, admission.ROOT, 262144))
        self.freeze = k.data(self.source, self.root)
        self.full_source()
        for name in (k.CONTRACT_PATH, k.BODY_PATH, k.ROSTER_PATH):
            self.copy(self.freeze["sources"][name])
        self.copy(self.freeze["reviewed_source"])
        review = k.data(self.freeze["reviewed_source"], self.root)
        for pin in (review["root_reflection"], *review["astra_review"].values(), *review["root_reviews"], *review["sol_reports"]):
            self.copy(pin)
        current_contract = k.data(self.freeze["sources"][k.CONTRACT_PATH], self.root)
        self._stage_paid(current_contract, k)
        direct_provenance = current_contract["provenance"]
        for name,path in (("root_design","docs/M3_28D18.md"),("astra_markdown",k.BASE+"/s28-d18-astra/review.md"),("astra_json",k.BASE+"/s28-d18-astra/review.json"),("root_reflection","docs/M3_28R18.md")):
            pin = direct_provenance[name]
            k.metadata_pin(pin,path)
            self.copy(pin,262144)
        fit_reports = [pin for pin in review["root_reviews"] if pin["path"] == k.BASE+"/s28-f18-root/SECOND.json"]
        r.need(len(fit_reports) == 1, "unique current ROOT SECOND fit report")
        fit_report = k.data(fit_reports[0], self.root)
        fit_pins = fit_report["fit_provenance"]
        fit_paths = ("docs/M3_28D18_FIT.md", k.BASE+"/s28-d18-fit-astra/review.md", k.BASE+"/s28-d18-fit-astra/review.json", "docs/M3_28R18_FIT.md")
        r.need(type(fit_pins) is list and len(fit_pins) == 4, "exact four supplemental fit pins")
        for fit_pin, fit_path in zip(fit_pins, fit_paths):
            k.metadata_pin(fit_pin, fit_path)
            r.need(type(fit_pin) is dict and type(fit_pin["path"]) is str and fit_pin["bytes"] <= 262144, "bounded exact supplemental metadata pin")
            r.fixture_text_write(self.root, fit_path, r.raw_pin(fit_pin, admission.ROOT, 262144))
        self._stage_residual(review, k)
        binding = current_contract["pinned_checkpoint"]
        for pin in (binding["coverage"], binding["latest_failure"], binding["latest_cost"], *binding["provenance"].values()):
            self.copy(pin)
        gate = k.data(k.S15_PINS["technical_refusal"], self.root)
        for name in ("actual_tool_return", "actual_known_compiler_type_refusal", "journal"):
            self.copy(gate[name], k.reader_cap(gate[name]["path"]))
        # Fixed D14 archives, full D12 history and partial D13/R11 raw operands.
        for pin in [{'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/a318.json', 'bytes': 495559, 'sha256': 'fadfb1604441f5c2cc481e15b07eeaf10a7a7529cedc8969bc496196ea221274'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-recovery10.json', 'bytes': 1903, 'sha256': '27f8b420a0f7cdb3447aac416cd2ac71cf499ee8b5ea719ff05c968fd6f1184e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/s1-before.json', 'bytes': 262140, 'sha256': 'f1b802af060e385f03955b5cbba5830cc9ee42ccf757b9800016a45afd7cabfd'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/c5-before.json', 'bytes': 97623, 'sha256': '9690bd2ec7a65d33432071d09d9050c6d949c92d6a20852fc92072737f7dcb41'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/h28-before.json', 'bytes': 261912, 'sha256': 'e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/actual-fast-warm-gate.json', 'bytes': 2818, 'sha256': '042051f83aaa9c347c72d943e76d0958cf22afeccb5fd3c929bf0241cd5f5a1f'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/s1-before.json', 'bytes': 262140, 'sha256': '39b4910b21c51042a5d68f3da51061ce2d57ead51ec5ff4cdcf82554a6493ef0'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/c5-before.json', 'bytes': 74674, 'sha256': '3a87bc0f34b8a36ddd5153940f2b17fccbdd990b32459aea28ae92e51be58b0a'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f12-root/actual-fast-warm-gate.json', 'bytes': 5074, 'sha256': '9c9348756e9dacf4ee53a320e50acc22d25c8fe9028419dccbc35819356d7b7e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f12-root/actual-tool-return.json', 'bytes': 1846, 'sha256': '71341a5233ecc031b45876ab24d1f7e4305ed79c738aa19f3efa71dc6c250162'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f12-root/actual-layout-compatibility-diagnostic.json', 'bytes': 1331, 'sha256': 'efc4694c9651f842832549809e1613b9900ab71004da098bb82dc24c29ee8a6d'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-catalog-leaves-300021e7370a4dc8b163ef22f0245f31.jsonl', 'bytes': 6393, 'sha256': 'c4bb7e972e5ed5950e554983344f5d34849008f9a2a9a984a874d1ff8795926e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/actual-tool-return.json', 'bytes': 2798, 'sha256': 'e76c924ab89dd27752d2f8a1afe68bd6839f363471886c073b3b14fce0d5d2c0'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/actual-compiler-type-refusal.json', 'bytes': 1968, 'sha256': '448f06a882edd1757895f90dfd4bd1c7bfad38f414fcdda3909a0eacfa34e407'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-catalog-leaves-30944ee8fe084e2aac23de9d47e4de80.jsonl', 'bytes': 496, 'sha256': 'b3ebd169ebfc3f700ec42a0c3b7b9292f35b7561cb6f7d100c71bf3835669606'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-recovery11.json', 'bytes': 1882, 'sha256': '0098806a84d0c18cf606482710d15e73dbe21defae360badba88a6c909db4c26'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/a318.json', 'bytes': 495559, 'sha256': 'fadfb1604441f5c2cc481e15b07eeaf10a7a7529cedc8969bc496196ea221274'}, {'path': 'verification/banc-natural-gait-m3-20260919/s27-bootstrap-settled318-HOLD-20261009.json', 'sha256': '70175b4bc56c19412b64500343ebc72bc538a7b3776dd60733d43efeeb59476e', 'bytes': 495417}, {'path': 'verification/banc-natural-gait-m3-20260919/e28.json', 'bytes': 2443, 'sha256': 'e5e284cbaa1f67f1dd6e14bdf3fe74c5f902d751c387f6850700323c23609c93'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/h28-before.json', 'bytes': 261912, 'sha256': 'e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s27-actual.json', 'bytes': 3365, 'sha256': 'fcf5b1fa42a8348fa52f85e0259cc3b3554be5d96206fd5e77f4fcc54e01eb94'}]:
            raw = r.raw_pin(pin, admission.ROOT, 524288 if pin["bytes"]>262144 else 262144)
            target = self.root / pin["path"]
            if target.exists(): r.exact(target.read_bytes(),raw,"same fixed D14 continuity raw")
            else: r.fixture_text_write(self.root,pin["path"],raw)
        historical_event = k.data(k.S15_PINS["previous_recovery"], self.root)
        for name in ("before_ledger","previous_recovery","previous_source","original_ledger","repair","catalog","latest_failure","technical_refusal"):
            pin = historical_event[name]
            self.copy(pin, 524288 if name in ("before_ledger","original_ledger") else 262144)
        bridge=current_contract["actual320"]
        _history=current_contract["terminal_bridge"]
        _aliases={item["logical_current_before"]["path"]:item["logical_current_before"] for item in bridge["archives"]}
        event=r.d19_event(k.__dict__,self.source,self.root)
        self.recovery=self.put(k.RECOVERY_PATH,event)
        self.ledger = k.recovered_snapshot(self.recovery, self.root)
        r.d19_stage(self,"source","END")
        self.ledger_path = k.BASE + "/parent-e358-execution-ledger.json"
        self.publish()
        r.d19_stage(self,"fixture","END")

    def _stage_paid(self, current_contract, k):
        # D18 finite paid stage begin: two ancestors and all three paid steps.
        def stage_paid_pin(pin):
            raw = r.raw_pin(pin, admission.ROOT, k.reader_cap(pin["path"]))
            target = self.root / pin["path"]
            if target.exists(): r.exact(target.read_bytes(), raw, "same finite paid raw")
            else: r.fixture_text_write(self.root, pin["path"], raw)
        paid = current_contract["actual320"]
        k.whole_exact(paid["adoption_before_Source"], k.CONTINUATION_BEFORE_SOURCE)
        for step, cid in enumerate(("direct-science-20261009-contact-s28-bootstrap-r3", "direct-science-20261009-contact-s28-bootstrap-r2", "direct-science-20261009-contact-s28-bootstrap-r1")):
            r.exact(paid["old_failed_call"], cid, "fixed staged paid identity")
            for pin in paid["raw_pins"]: stage_paid_pin(pin)
            for pin in (paid["old_request"], paid["old_cost"]): stage_paid_pin(pin)
            paid_request = k.data(paid["old_request"], self.root)
            paid_cost = k.data(paid["old_cost"], self.root)
            k.whole_exact(paid_cost["request"], paid["old_request"])
            r.exact((paid_request["call_id"], paid_cost["cost_id"]), (cid, cid), "same finite request/cost identity")
            r.exact(paid["old_cost"]["path"], k.BASE+"/parent-"+cid+"-full-return.json", "fixed finite cost path")
            for pin in (paid_cost["trace"], paid_cost["outer"]): stage_paid_pin(pin)
            if step < 2:
                contract_rows = [row for row in paid["archives"] if row["logical_current_before"]["path"] == k.CONTRACT_PATH]
                review_rows = [row for row in paid["archives"] if row["logical_current_before"]["path"] == k.REVIEW_PATH]
                r.need(len(contract_rows) == len(review_rows) == 1, "finite unique predecessor contract/review")
                cr, rr = contract_rows[0], review_rows[0]
                for pin in (paid["adoption_before_Source"], cr["immutable_archive"], rr["immutable_archive"]): stage_paid_pin(pin)
                old_source = k.data(paid["adoption_before_Source"], self.root)
                old_review = k.data(rr["immutable_archive"], self.root)
                k.whole_exact(old_source["reviewed_source"], rr["logical_current_before"])
                k.whole_exact([old_source["checkpoint_contract"], old_source["sources"][k.CONTRACT_PATH], old_review["sources"][k.CONTRACT_PATH]], [cr["logical_current_before"]]*3)
                k.whole_exact(old_source["sources"], old_review["sources"])
                paid = k.data(cr["immutable_archive"], self.root)["actual320"]
        # D18 finite paid stage end.

    def _stage_residual(self, review, k):
        residual_reports = [pin for pin in review["root_reviews"] if pin["path"] == k.BASE+"/s28-f18-root/residual03/final-v2.json"]
        r.need(len(residual_reports) == 1, "unique authenticated residual ROOT final report")
        residual_report = k.data(residual_reports[0], self.root)
        residual_pins = residual_report["residual_provenance"]
        residual_paths = ("docs/M3_28D18_RESIDUAL3.md", k.BASE+"/s28-f18-root/residual03/astra/review.md", k.BASE+"/s28-f18-root/residual03/astra/review.json", "docs/M3_28R18_RESIDUAL3.md")
        r.need(type(residual_pins) is list and len(residual_pins) == 4, "exact four residual provenance pins")
        for residual_pin, residual_path in zip(residual_pins, residual_paths):
            r.need(type(residual_pin) is dict and list(residual_pin) == ["path","bytes","sha256"] and type(residual_pin["path"]) is str and type(residual_pin["bytes"]) is int and 0 < residual_pin["bytes"] <= 262144 and type(residual_pin["sha256"]) is str, "typed bounded residual descriptor")
            k.metadata_pin(residual_pin, residual_path)
            residual_raw = r.raw_pin(residual_pin, admission.ROOT, 262144)
            r.fixture_text_write(self.root, residual_path, residual_raw)

    def copy(self, pin, cap=None):
        if pin["path"] in ('verification/banc-natural-gait-m3-20260919/s28-f18-root/FIRST.md', 'verification/banc-natural-gait-m3-20260919/s28-f18-root/SECOND.md', 'verification/banc-natural-gait-m3-20260919/s28-f18-root/residual03/final-v2.md', 'verification/banc-natural-gait-m3-20260919/s28-f18-root/SECOND.json', 'verification/banc-natural-gait-m3-20260919/s28-f18-root/residual03/final-v2.json', 'verification/banc-natural-gait-m3-20260919/s28-f18-sol/FIRST.json', 'verification/banc-natural-gait-m3-20260919/s28-f18-sol/sole-R1/R1.json'):
            raw = exact_source_operand(pin)
            target = self.root / pin["path"]
            if target.exists():
                r.exact(target.read_bytes(), raw, "same declared current D18 report")
                return pin
            return r.fixture_text_write(self.root, pin["path"], raw)
        if cap is not None:
            r.need(type(cap) is int and cap == k.reader_cap(pin["path"]), "exact fixture reader cap")
        contract = k.data(k.CURRENT_CONTRACT, admission.ROOT)
        predecessor_rows = [row for row in contract["actual320"]["archives"] if row["logical_current_before"]["path"] == k.CONTRACT_PATH]
        r.need(len(predecessor_rows) == 1, "copy unique predecessor contract")
        predecessor_contract = predecessor_rows[0]["immutable_archive"]
        if pin in (*contract["actual320"]["raw_pins"], *k.data(predecessor_contract, admission.ROOT)["actual320"]["raw_pins"]):
            raw=r.raw_pin(pin,admission.ROOT,k.reader_cap(pin["path"]))
        elif pin in [{'path': 'verification/banc-natural-gait-m3-20260919/s28-f15b-root/terminal319-before.json', 'bytes': 394685, 'sha256': 'c8a74a40a0d2bfef0e630524508d819c2b106cbce06f6f2b93364f5e4b4b1482'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f15-root/admitted318-before.json', 'bytes': 496266, 'sha256': 'ea814b9271554155b9f71673ac4d8c5a4e2a1fcb5a3cc5a20cdf57dd59cb405d'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f15-root/settlement-pre-backup.json', 'bytes': 496266, 'sha256': 'ea814b9271554155b9f71673ac4d8c5a4e2a1fcb5a3cc5a20cdf57dd59cb405d'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f15-root/settlement-replace-backup.json', 'bytes': 496266, 'sha256': 'ea814b9271554155b9f71673ac4d8c5a4e2a1fcb5a3cc5a20cdf57dd59cb405d'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-exec-precreate-cost.json', 'bytes': 2452, 'sha256': '359c0dd97f2baf9f063d9a26c72fcd5269c3a66665be7a4004fdf8cf2278abb4'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-exec-precreate-settlement.json', 'bytes': 2916, 'sha256': '79c2967af54a4a28d7dd580c98e068396f1141b61043361c64befaf72190df31'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f15-root/settlement-owner-token.json', 'bytes': 212, 'sha256': 'c8ef54795ae8bc5fe801478578436c66cbc5b11c9b6995bbc5ad10a88236bf02'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f15-root/actual-settlement-audit.json', 'bytes': 2459, 'sha256': '8cb8eeb7decead2e8db9dd4195e571a667d5b6457ff72165721e9d8fdb8b4401'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f15-root/settlement-actual-tool-capture.json', 'bytes': 13865, 'sha256': '6c02d1939b022ced5f7b45f71712d1d97cae4bef54dc4284f93f58c8827f6f63'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/full4-failure.json', 'bytes': 6976, 'sha256': '978021662924b9a1c15bfac7cc5699a58535d43eececd894d58b542d7395ff0c'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/full4-actual-tool-capture.json', 'bytes': 6464, 'sha256': '72ccd3b9621fe0979c3d51737fc8d3f6c818548ed017e09165ee7502b307e247'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/full4-request.json', 'bytes': 1371, 'sha256': 'bdd63628db50c52b0a6ebf9f0f769848d79bcd241558397a55084b25dbdec90b'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f15b-root/before/verification/banc-natural-gait-m3-20260919/c5.json', 'bytes': 124922, 'sha256': '9d65539915dbe98563ae5599df9d91c422541411f51608bc5ebe14fab8655ed8'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f15b-root/before/verification/banc-natural-gait-m3-20260919/s28-s1.json', 'bytes': 262142, 'sha256': 'b8f00cb6fe06b0fa611b10b62d60f39d5a8e130887767f9e6c791b0973e35bbc'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f15b-root/before/verification/banc-natural-gait-m3-20260919/s28-r1.json', 'bytes': 246958, 'sha256': '06479a5a995aa0d0eb7a3b2a7d062506005ae5e2cfe6d2b37a74c9b27f5bca97'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f15b-root/before/verification/banc-natural-gait-m3-20260919/s28-recovery12.json', 'bytes': 1869, 'sha256': '26d3727cb1ade441017cc5dfe98a474a740aec206e55dbb5aa0adedb8d0b9add'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-recovery10.json', 'bytes': 1903, 'sha256': '27f8b420a0f7cdb3447aac416cd2ac71cf499ee8b5ea719ff05c968fd6f1184e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/s1-before.json', 'bytes': 262140, 'sha256': '39b4910b21c51042a5d68f3da51061ce2d57ead51ec5ff4cdcf82554a6493ef0'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f12-root/a318.json', 'bytes': 495558, 'sha256': '1221bf613896fd8f929b0dbfd93c1e4bc8e922f4c873e3869d5a7780d53c6d54'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/a318.json', 'bytes': 495559, 'sha256': 'fadfb1604441f5c2cc481e15b07eeaf10a7a7529cedc8969bc496196ea221274'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f12-root/actual-tool-return.json', 'bytes': 1846, 'sha256': '71341a5233ecc031b45876ab24d1f7e4305ed79c738aa19f3efa71dc6c250162'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/c5-before.json', 'bytes': 74674, 'sha256': '3a87bc0f34b8a36ddd5153940f2b17fccbdd990b32459aea28ae92e51be58b0a'}, {'path': 'verification/banc-natural-gait-m3-20260919/h28.json', 'bytes': 261912, 'sha256': 'e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f12-root/actual-layout-compatibility-diagnostic.json', 'bytes': 1331, 'sha256': 'efc4694c9651f842832549809e1613b9900ab71004da098bb82dc24c29ee8a6d'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-catalog-leaves-300021e7370a4dc8b163ef22f0245f31.jsonl', 'bytes': 6393, 'sha256': 'c4bb7e972e5ed5950e554983344f5d34849008f9a2a9a984a874d1ff8795926e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f12-root/fast-counter-expectation.json', 'bytes': 2331, 'sha256': '99e971e5a752697e8ab86e9fbf461acd4052d1d27da3a5437f05db14fe050b5d'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-recovery9.json', 'bytes': 1902, 'sha256': 'a4cf49fb579aeb3114fd800f092080b5e6a54b8f71ae835121e8cd34a1d4a28d'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f12-root/s1-before.json', 'bytes': 262140, 'sha256': '1cf30bee866a147de013d0023a2ea02d8734005178d145d7f06952f058ef4aba'}, {'path': 'verification/banc-natural-gait-m3-20260919/s27-bootstrap-settled318-HOLD-20261009.json', 'sha256': '70175b4bc56c19412b64500343ebc72bc538a7b3776dd60733d43efeeb59476e', 'bytes': 495417}, {'path': 'verification/banc-natural-gait-m3-20260919/e28.json', 'bytes': 2443, 'sha256': 'e5e284cbaa1f67f1dd6e14bdf3fe74c5f902d751c387f6850700323c23609c93'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f12-root/h28-before.json', 'bytes': 261912, 'sha256': 'e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s27-actual.json', 'bytes': 3365, 'sha256': 'fcf5b1fa42a8348fa52f85e0259cc3b3554be5d96206fd5e77f4fcc54e01eb94'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f11-root/actual-fast-warm-gate.json', 'bytes': 4977, 'sha256': '9bbf1418212eedfb12b4aaa57cac0d613c9b900b0f9e2a0b4a4d7dfa1f63c0ce'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-recovery11.json', 'bytes': 1882, 'sha256': '0098806a84d0c18cf606482710d15e73dbe21defae360badba88a6c909db4c26'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/s1-before.json', 'bytes': 262140, 'sha256': 'f1b802af060e385f03955b5cbba5830cc9ee42ccf757b9800016a45afd7cabfd'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/actual-tool-return.json', 'bytes': 2798, 'sha256': 'e76c924ab89dd27752d2f8a1afe68bd6839f363471886c073b3b14fce0d5d2c0'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-catalog-leaves-30944ee8fe084e2aac23de9d47e4de80.jsonl', 'bytes': 496, 'sha256': 'b3ebd169ebfc3f700ec42a0c3b7b9292f35b7561cb6f7d100c71bf3835669606'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/c5-before.json', 'bytes': 97623, 'sha256': '9690bd2ec7a65d33432071d09d9050c6d949c92d6a20852fc92072737f7dcb41'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/actual-compiler-type-refusal.json', 'bytes': 1968, 'sha256': '448f06a882edd1757895f90dfd4bd1c7bfad38f414fcdda3909a0eacfa34e407'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/a318.json', 'bytes': 495559, 'sha256': 'fadfb1604441f5c2cc481e15b07eeaf10a7a7529cedc8969bc496196ea221274'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/h28-before.json', 'bytes': 261912, 'sha256': 'e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f12-root/actual-fast-warm-gate.json', 'bytes': 5074, 'sha256': '9c9348756e9dacf4ee53a320e50acc22d25c8fe9028419dccbc35819356d7b7e'}, {'bytes': 495417, 'sha256': '70175b4bc56c19412b64500343ebc72bc538a7b3776dd60733d43efeeb59476e', 'path': 'verification/banc-natural-gait-m3-20260919/s28-anchor318-before-recovery.json'}, {'path': 'verification/banc-natural-gait-m3-20260919/s27-s1.json', 'sha256': '35b6cf63018f61f3fde34d47045d16f5e3c97dc53bac64b1fe7814aed0aa61c6', 'bytes': 262139}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/h28-before.json', 'bytes': 261912, 'sha256': 'e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/actual-fast-warm-gate.json', 'bytes': 2818, 'sha256': '042051f83aaa9c347c72d943e76d0958cf22afeccb5fd3c929bf0241cd5f5a1f'}, {'path': 'docs/M3_28D15C.md', 'bytes': 10675, 'sha256': '1c1337578a8982b240bf743bb31ad421fde9cd71b54c7e5650b32295a7d43055'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-d15c-astra/review.md', 'bytes': 16179, 'sha256': '79b900d090ca7fe61f0322a3dff5099ebbdf50444d91cd113c6e709a3dd21290'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-d15c-astra/review.json', 'bytes': 14885, 'sha256': '69c41861683acb00b7f784fc2d41601c9d8dbe1611345e9337c672c1ca0895d3'}, {'path': 'docs/M3_28R15C.md', 'bytes': 9488, 'sha256': '578b086dfe080b3119ce75404abdcb7143d715a9251d60396ad6fe332bf27a33'}] or pin in [{'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/a318.json', 'bytes': 495559, 'sha256': 'fadfb1604441f5c2cc481e15b07eeaf10a7a7529cedc8969bc496196ea221274'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-recovery10.json', 'bytes': 1903, 'sha256': '27f8b420a0f7cdb3447aac416cd2ac71cf499ee8b5ea719ff05c968fd6f1184e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/s1-before.json', 'bytes': 262140, 'sha256': 'f1b802af060e385f03955b5cbba5830cc9ee42ccf757b9800016a45afd7cabfd'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/c5-before.json', 'bytes': 97623, 'sha256': '9690bd2ec7a65d33432071d09d9050c6d949c92d6a20852fc92072737f7dcb41'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f14-root/h28-before.json', 'bytes': 261912, 'sha256': 'e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/actual-fast-warm-gate.json', 'bytes': 2818, 'sha256': '042051f83aaa9c347c72d943e76d0958cf22afeccb5fd3c929bf0241cd5f5a1f'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/s1-before.json', 'bytes': 262140, 'sha256': '39b4910b21c51042a5d68f3da51061ce2d57ead51ec5ff4cdcf82554a6493ef0'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/c5-before.json', 'bytes': 74674, 'sha256': '3a87bc0f34b8a36ddd5153940f2b17fccbdd990b32459aea28ae92e51be58b0a'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f12-root/actual-fast-warm-gate.json', 'bytes': 5074, 'sha256': '9c9348756e9dacf4ee53a320e50acc22d25c8fe9028419dccbc35819356d7b7e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f12-root/actual-tool-return.json', 'bytes': 1846, 'sha256': '71341a5233ecc031b45876ab24d1f7e4305ed79c738aa19f3efa71dc6c250162'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f12-root/actual-layout-compatibility-diagnostic.json', 'bytes': 1331, 'sha256': 'efc4694c9651f842832549809e1613b9900ab71004da098bb82dc24c29ee8a6d'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-catalog-leaves-300021e7370a4dc8b163ef22f0245f31.jsonl', 'bytes': 6393, 'sha256': 'c4bb7e972e5ed5950e554983344f5d34849008f9a2a9a984a874d1ff8795926e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/actual-tool-return.json', 'bytes': 2798, 'sha256': 'e76c924ab89dd27752d2f8a1afe68bd6839f363471886c073b3b14fce0d5d2c0'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/actual-compiler-type-refusal.json', 'bytes': 1968, 'sha256': '448f06a882edd1757895f90dfd4bd1c7bfad38f414fcdda3909a0eacfa34e407'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-catalog-leaves-30944ee8fe084e2aac23de9d47e4de80.jsonl', 'bytes': 496, 'sha256': 'b3ebd169ebfc3f700ec42a0c3b7b9292f35b7561cb6f7d100c71bf3835669606'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-recovery11.json', 'bytes': 1882, 'sha256': '0098806a84d0c18cf606482710d15e73dbe21defae360badba88a6c909db4c26'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/a318.json', 'bytes': 495559, 'sha256': 'fadfb1604441f5c2cc481e15b07eeaf10a7a7529cedc8969bc496196ea221274'}, {'path': 'verification/banc-natural-gait-m3-20260919/s27-bootstrap-settled318-HOLD-20261009.json', 'sha256': '70175b4bc56c19412b64500343ebc72bc538a7b3776dd60733d43efeeb59476e', 'bytes': 495417}, {'path': 'verification/banc-natural-gait-m3-20260919/e28.json', 'bytes': 2443, 'sha256': 'e5e284cbaa1f67f1dd6e14bdf3fe74c5f902d751c387f6850700323c23609c93'}, {'path': 'verification/banc-natural-gait-m3-20260919/s28-f13-root/h28-before.json', 'bytes': 261912, 'sha256': 'e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e'}, {'path': 'verification/banc-natural-gait-m3-20260919/s27-actual.json', 'bytes': 3365, 'sha256': 'fcf5b1fa42a8348fa52f85e0259cc3b3554be5d96206fd5e77f4fcc54e01eb94'}]:
            raw=r.raw_pin(pin,admission.ROOT,524288 if pin['bytes']>262144 else 262144)
        elif pin["bytes"] == 0:
            r.need(pin["path"] in k.EMPTY_PATHS, "only exact eight empty stderr paths")
            r.exact(pin, {"path":pin["path"],"bytes":0,"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}, "exact empty stderr")
            raw = r.raw_pin(pin, admission.ROOT, k.reader_cap(pin["path"]), allow_empty=True)
        elif cap is not None or pin["path"] in k.LEDGER_METADATA_PATHS or pin["path"].startswith((k.BASE+"/s14-before/", k.BASE+"/s15-before/", k.BASE+"/s16-before/", k.BASE+"/s17-before/", k.BASE+"/s18-before/", k.BASE+"/s19-before/", k.BASE+"/s22-before/", k.BASE+"/s23-before/", k.BASE+"/s24-before/", k.BASE+"/s25-before/", k.BASE+"/s26-before/", k.BASE+"/s27-before/", k.BASE+"/s28-before/")):
            raw = r.raw_pin(pin, admission.ROOT, k.reader_cap(pin["path"]))
        elif pin["path"] in ('docs/M3_28D18.md', 'verification/banc-natural-gait-m3-20260919/s28-d18-astra/review.md', 'verification/banc-natural-gait-m3-20260919/s28-d18-astra/review.json', 'docs/M3_28R18.md'):
            raw = r.raw_pin(pin, admission.ROOT, 262144)
        elif pin in tuple(k.S15_PINS[name] for name in ("previous_recovery","previous_source","previous_contract","previous_catalog","technical_refusal")) or pin in tuple(k.data(k.S15_PINS["technical_refusal"], admission.ROOT)[name] for name in ("actual_tool_return","actual_known_compiler_type_refusal","journal")):
            raw = r.raw_pin(pin, admission.ROOT, k.reader_cap(pin["path"]))
        elif pin["path"] in ('verification/banc-natural-gait-m3-20260919/s28-f15c-root/FIRST.md', 'verification/banc-natural-gait-m3-20260919/s28-f15c-root/SECOND.md', 'verification/banc-natural-gait-m3-20260919/s28-f15c-root/final.md', 'verification/banc-natural-gait-m3-20260919/s28-f15c-root/SECOND.json', 'verification/banc-natural-gait-m3-20260919/s28-f15c-root/final.json', 'verification/banc-natural-gait-m3-20260919/s28-f15c-sol/FIRST.json', 'verification/banc-natural-gait-m3-20260919/s28-f15c-sol/R1.json'):
            raw = r.raw_pin(pin, admission.ROOT, 262144)
        else:
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
        for pin in self.freeze["sources"].values():
            if not Path(pin["path"]).is_absolute():
                self.copy(pin)

    def request(self, call=k.BOOTSTRAP_CALL):
        p = k.plan(call, self.freeze, self.root)
        q = {"schema": "flylab.m3.direct-science.supervision.v1", "call_id": call, "mode": p["mode"], "role": p["role"],
             "source_freeze": self.source, "limits": p["limits"], "output": p["output"], "ledger_path": self.ledger_path,
             "shared_original_lock_path": r.SHARED_LOCK, "generated_behavior": "command", "generated_argv": p["argv"]}
        spec = self.put(k.BASE+"/s28-d19bz-full4-request.json" if call==k.BOOTSTRAP_CALL else k.BASE+"/parent-"+call+"-request.json", q)
        return q, spec

    def admit(self):
        self.q, self.request_spec = self.request()
        k.dispatch(self.ledger, self.q, self.request_spec, self.root, phase="closed")
        self.ledger["direct_science_calls"][self.q["call_id"]] = {"request": self.request_spec, "state": "admitted", "mode": "generated", "reserved_s": self.q["limits"]["outer_s"]}
        self.ledger["direct_science_reservations"] = {self.q["call_id"]: {"request": self.request_spec, "reserved_s": self.q["limits"]["outer_s"]}}
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
        trace["exec_args"]={"cmd":"& .venv/Scripts/python.exe -m tools.banc_m3_startup_ray_supervisor --request "+self.request_spec["path"],"workdir":self.root.as_posix()}
        self.trace_spec = self.put(k.BASE + "/parent-" + call + "-genuine-return.json", trace)
        if not (uncertain or no_create) and call in self.ledger["direct_science_calls"] and self.ledger["direct_science_calls"][call]["state"]=="admitted":
            self.ledger["direct_science_reservations"]={};self.ledger["direct_science_workers"]={}
            self.ledger["direct_science_pending_actuals"]={call:{"request":self.request_spec,"state":"actual_pending","ownership_closed":True}}
            self.publish()
        return outer

    def unknown_owner(self):
        call = self.q['call_id']
        identity = {'owner_pid': 411, 'owner_creation_ticks': 412, 'worker_pid': 413, 'worker_creation_ticks': 414}
        token = {'schema': 'flylab.m3.direct-science.lock-owner.v1', 'call_id': call, 'request': self.request_spec,
                 'owner_pid': 411, 'owner_creation_ticks': 412, 'entry_counter_s': 1.0, 'job_name': 'Local\\generated-synthetic-L'}
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
        before_spec = self.put(k.BASE + '/generated-l13r2-original-before.json', before)
        contract = k.data(body['contract'], self.root)
        call = k.CALLS[0]
        self.science = {'schema': 'flylab.m3.contact-learning.request.v1', 'mode': 'original', 'role': 'contact_candidate_body',
            'call_id': call, 'output': k.BASE + '/parent-' + call + '/result', 'source_freeze': self.source,
            'body_retry': self.freeze['sources'][k.BODY_PATH], 'limits': contract['limits']['contact_candidate_body'],
            'ledger_path': self.ledger_path, 'shared_original_lock_path': r.SHARED_LOCK, 'local_debit_offset': 165,
            'resource_increment_s': 0, 'historical_manifest': previous['historical_manifest'], 'qualification': qualified,
            'before_ledger': before_spec, **{key: body[key] for key in ('contract', 'grant', 'inputs', 'roster')}}
        science_spec = self.put(k.BASE + '/generated-l-original-science.json', self.science)
        p = k.plan(call, self.freeze, self.root)
        self.q = {'schema': 'flylab.m3.direct-science.supervision.v1', 'call_id': call, 'mode': 'original', 'role': p['role'],
            'source_freeze': self.source, 'limits': p['limits'], 'output': p['output'], 'ledger_path': self.ledger_path,
            'shared_original_lock_path': r.SHARED_LOCK, 'grant': body['grant'], 'science_request': science_spec}
        self.request_spec = self.put(k.BASE+'/parent-'+self.q['call_id']+'-request.json', self.q)
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
        # Synthetic fixture fee projection stays local; production payment remains Native K-Commit.
        import hashlib
        import math
        import os
        import re
        import struct
        import time

        from tools.banc_m3_startup_ray_supervisor import creation_ticks
        outer = self.evidence(seconds, exit_code, **creation)
        if not k.closure(outer):
            return {"committed":False,"status":"UNCOMMITTED_OWNERSHIP_UNKNOWN","cost_published":False,"owner_and_reservation_retained":True}
        k.integer(exit_code,"actual fixture exit code"); k.integer(outer["exit_code"],"actual outer exit code")
        def whole(a,b):
            r.exact(a,b,"whole independent fixture fee")
            if type(a) is float: r.need(math.isfinite(a) and struct.pack('>d',a)==struct.pack('>d',b),"exact finite binary64")
            if type(a) is dict:
                r.exact(list(a),list(b),"exact object order")
                for key in a: whole(a[key],b[key])
            elif type(a) is list:
                for left,right in zip(a,b,strict=True): whole(left,right)
        def semantic_equal(a,b):
            r.exact(a,b,"whole typed membership/positional arrays")
            if type(a) is float: r.need(math.isfinite(a) and struct.pack('>d',a)==struct.pack('>d',b),"exact finite binary64")
            if type(a) is dict:
                for key in a: semantic_equal(a[key],b[key])
            elif type(a) is list:
                for left,right in zip(a,b,strict=True): semantic_equal(left,right)
        def sorted_state(v):
            if type(v) is dict:return {key:sorted_state(v[key]) for key in sorted(v)}
            if type(v) is list:return [sorted_state(item) for item in v]
            return v
        k.metadata_pin(self.recovery,k.RECOVERY_PATH)
        immutable_event=k.data(self.recovery,self.root)
        return r.d19_fixture_commit(k,self,seconds,exit_code,outer)
        immutable_policy=k.data(r.POLICY_PIN,self.root)
        immutable_grant=k.data(r.J_PIN,self.root)
        contract=k.data(k.CURRENT_CONTRACT,self.root); bridge=contract["actual320"]
        binding=contract["pinned_checkpoint"]
        for operand in (binding["coverage"],binding["latest_failure"],binding["latest_cost"],binding["policy"],binding["grant"],*binding["provenance"].values()): r.raw_pin(operand,self.root,262144)
        old_bridge=contract["terminal_bridge"]
        for name in ("terminal","admitted","pre_backup","replace_backup","cost","receipt","actor","audit","tool","failure","failed_tool","request"):
            operand=old_bridge[name];r.raw_pin(operand,self.root,k.reader_cap(operand["path"]))
        for row in old_bridge["archives"]: r.raw_pin(row["immutable_archive"],self.root,262144)
        for operand in bridge["raw_pins"]: r.raw_pin(operand,self.root,k.reader_cap(operand["path"]))
        whole(bridge["adoption_before_Source"], k.CONTINUATION_BEFORE_SOURCE)
        ancestor = bridge
        for _ in range(2):
            contract_rows = [row for row in ancestor["archives"] if row["logical_current_before"]["path"] == k.CONTRACT_PATH]
            review_rows = [row for row in ancestor["archives"] if row["logical_current_before"]["path"] == k.REVIEW_PATH]
            r.need(len(contract_rows) == len(review_rows) == 1, "fee unique ancestor contract/review")
            cr, rr = contract_rows[0], review_rows[0]
            predecessor_contract = k.data(cr["immutable_archive"], self.root)
            predecessor_source = k.data(ancestor["adoption_before_Source"], self.root)
            predecessor_review = k.data(rr["immutable_archive"], self.root)
            whole(predecessor_source["reviewed_source"], rr["logical_current_before"])
            whole([predecessor_source["checkpoint_contract"], predecessor_source["sources"][k.CONTRACT_PATH], predecessor_review["sources"][k.CONTRACT_PATH]], [cr["logical_current_before"]]*3)
            whole(predecessor_source["sources"], predecessor_review["sources"])
            ancestor = predecessor_contract["actual320"]
            for operand in ancestor["raw_pins"]: r.raw_pin(operand,self.root,k.reader_cap(operand["path"]))
        immutable_baseline=k.data(bridge["terminal"],self.root,524288)
        r.need(len(immutable_baseline["debits"])==322 and len(immutable_baseline["direct_science_calls"])==204,"exact322/204 baseline")
        for owner_key in r.OWNER_KEYS:
            if owner_key!="direct_science_blocking_holds": r.exact(immutable_baseline[owner_key],{},"sealed baseline owner map")
        r.exact((immutable_baseline["original_jobs_running"],immutable_baseline["original_reserved_s"]),(0,0),"sealed no Original liability")
        r.exact(immutable_baseline["direct_science_blocking_holds"],{bridge["old_failed_call"]:bridge["old_failed_hold"]},"exact prior r1 HOLD")
        k.metadata_pin(self.source,k.SOURCE_PATH)
        whole(k.data(self.source,self.root),self.freeze)
        k.metadata_pin(self.freeze["reviewed_source"],k.REVIEW_PATH)
        review=k.data(self.freeze["reviewed_source"],self.root)
        k.provenance(review,self.root)
        r.exact(review["before_source"],k.CONTINUATION_BEFORE_SOURCE,"fee predecessor Source authority")
        predecessor=k.data(k.CONTINUATION_BEFORE_SOURCE,self.root)
        inherited=copy.deepcopy(predecessor)
        inherited.update(qualification_epoch=k.EPOCH,reviewed_source=self.freeze["reviewed_source"],
            body_retry=self.freeze["sources"][k.BODY_PATH],sources=self.freeze["sources"],
            checkpoint_contract=self.freeze["sources"][k.CONTRACT_PATH])
        semantic_equal(self.freeze,inherited)
        r.need(len(predecessor["sources"])==871,"complete871 Source metadata")
        k.keys(self.freeze["sources"],predecessor["sources"],"complete inherited Source namespace")
        semantic_equal(self.freeze["sources"],review["sources"])
        r.exact(review["replacements"],list(k.REPLACEMENTS),"exact ordered11 replacements")
        r.exact(review["additions"],[],"exact empty Source additions")
        for source_name,source_pin in predecessor["sources"].items():
            if source_name not in k.REPLACEMENTS: semantic_equal(self.freeze["sources"][source_name],source_pin)
        for source_name in k.REPLACEMENTS:
            source_pin=self.freeze["sources"][source_name]
            k.keys(source_pin,("path","bytes","sha256"),"whole fee Source descriptor")
            r.exact(source_pin["path"],source_name,"fixed fee Source name")
            k.integer(source_pin["bytes"],"fee Source bytes")
            cap=131072 if source_name in k.REPLACEMENTS[:2] else 393216 if source_name==k.BASE+"/parent-direct-science-account.ps1" else 262144
            r.need(0<=source_pin["bytes"]<=cap and type(source_pin["sha256"]) is str and re.fullmatch("[0-9a-f]{64}",source_pin["sha256"]),"fee Source metadata")
        aliases={row["logical_current_before"]["path"]:row["logical_current_before"] for row in bridge["archives"]}
        expected_event={"schema":contract["terminal_bridge"]["event_schema"],"epoch":k.EPOCH,
            "before_ledger":bridge["terminal"],"source_freeze":self.source,"review":self.freeze["reviewed_source"],
            "settlement":bridge["commit_capture"],"cost":bridge["old_cost"],
            "audit":bridge["failure"],"previous_recovery":bridge["old_continuation"],
            "old_source":aliases[k.SOURCE_PATH],"old_review":aliases[k.REVIEW_PATH],"old_contract":aliases[k.CONTRACT_PATH],
            "root_reviews":review["root_reviews"],"sol_reports":review["sol_reports"],"old_caps":dict(k.CAPS),
            "new_caps":dict(k.CAPS),"new_resource_grant_s":0,"refund_s":0,"reset":False}
        k.keys(immutable_event,contract["terminal_bridge"]["event_keys"],"whole19 continuation event")
        whole(immutable_event,expected_event)
        whole(immutable_event["root_reviews"],review["root_reviews"])
        whole(immutable_event["sol_reports"],review["sol_reports"])
        k.identity(k.data(self.request_spec,self.root),self.request_spec,self.source,self.root)
        trace=k.data(self.trace_spec,self.root); actual_outer=k.data(self.outer_spec,self.root)
        r.exact((trace["phase"],trace["first_error"],trace["request"],trace["source_freeze"],actual_outer["request"]),("FINAL",None,self.request_spec,self.source,self.request_spec),"exact immutable first FINAL and admitted outer")
        r.need(type(trace["dispatch_ms"]) is int and type(trace["first_final_ms"]) is int and trace["first_final_ms"]>trace["dispatch_ms"]>0,"typed genuine elapsed")
        r.exact((trace["first_final_ms"]-trace["dispatch_ms"])/1000.0,seconds,"exact first FINAL fee")
        r.exact((trace["full_return_s"],trace["exit_code"]),(seconds,exit_code),"exact incurred fee/exit")
        whole(actual_outer,outer)
        cost={"schema":"flylab.m3.direct-science.full-return.v1","cost_id":self.q["call_id"],"mode":"generated","request":self.request_spec,"source_freeze":self.source,"trace":self.trace_spec,"outer":self.outer_spec,"actual_outer_full_return_s":seconds,"exit_code":exit_code,"charged_once":True,"cap_exceeded":seconds>self.q["limits"]["outer_s"],**{key:outer[key] for key in ("all_owned_ended","reservation_released","worker_started","worker_creation_state","creation_attempted","resume_attempted","resume_outcome")}}
        raw=json.dumps(cost,ensure_ascii=True,allow_nan=False).encode("utf8"); cost_path=k.BASE+"/parent-"+self.q["call_id"]+"-full-return.json"
        cost_pin={"path":cost_path,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
        debit={"cost_id":self.q["call_id"],"mode":"generated","actual_outer_full_return_s":seconds,"actual_returned_exit_code":exit_code,"descriptor":cost_pin}
        def fee(pin,row,source,root):
            if pin["path"]==cost_path:
                k.metadata_pin(pin,k.BASE+"/parent-"+row["cost_id"]+"-full-return.json")
                r.exact(pin,cost_pin,"exact computed provisional fee tuple")
                request_pin=cost["request"]
                request=k.data(request_pin,root)
                k.identity(request,request_pin,source,root)
                r.exact(request,self.q,"same independently authenticated fixture request")
                for field,value in (("request",request_pin),("source_freeze",source),("schema","flylab.m3.direct-science.genuine-return.v1"),("phase","FINAL"),("first_error",None)):
                    r.exact(trace[field],value,"provisional genuine trace "+field)
                r.exact(actual_outer["request"],request_pin,"provisional actual outer binding")
                k.integer(trace["dispatch_ms"],"genuine dispatch",positive=True)
                k.integer(trace["first_final_ms"],"genuine first FINAL",positive=True)
                fee_s=(trace["first_final_ms"]-trace["dispatch_ms"])/1000.0
                r.need(k.number(fee_s,"positive exact fee")>0,"positive elapsed")
                r.exact(trace["full_return_s"],fee_s,"exact genuine full return")
                for field,value in (("cost_id",request["call_id"]),("mode",request["mode"]),("source_freeze",source),
                    ("actual_outer_full_return_s",fee_s),("exit_code",trace["exit_code"]),("charged_once",True),
                    ("cap_exceeded",fee_s>request["limits"]["outer_s"])):
                    r.exact(cost[field],value,"once provisional cost "+field)
                r.exact(row,{"cost_id":request["call_id"],"mode":request["mode"],"actual_outer_full_return_s":fee_s,
                    "actual_returned_exit_code":trace["exit_code"],"descriptor":pin},"whole provisional paid debit")
                k.integer(trace["exit_code"],"genuine exit code")
                k.integer(actual_outer["exit_code"],"actual exit code")
                shell_pair=(trace["exit_code"]==1 and actual_outer["exit_code"]==124 and actual_outer["worker_exit_code"]==124
                    and actual_outer.get("timeout") is True and (actual_outer.get("first_error") or {}).get("type")=="TimeoutError"
                    and actual_outer["all_owned_ended"] is True and actual_outer["reservation_released"] is True
                    and actual_outer["stdout_eof"] is True and actual_outer["stderr_eof"] is True and not actual_outer["unknown_owned_identities"]
                    and trace.get("exec_args",{}).get("cmd","").endswith("& .venv/Scripts/python.exe -m tools.banc_m3_startup_ray_supervisor --request "+request_pin["path"]))
                r.need(actual_outer["exit_code"]==trace["exit_code"] or shell_pair,"actual exit or proven existing shell deadline pair")
                layout={"schema","cost_id","mode","request","source_freeze","trace","outer","actual_outer_full_return_s",
                    "exit_code","charged_once","all_owned_ended","reservation_released","cap_exceeded","worker_started",
                    "creation_attempted","worker_creation_state","resume_attempted","resume_outcome"}
                if request["mode"]=="original":layout|={"supervisor_request","role","grant","inputs","roster","contract","body_retry","local_debit_offset","resource_increment_s"}
                if shell_pair:layout|={"native_supervisor_exit_code","actual_collector_shell_exit_code","shell_deadline_status_pair_observed"}
                k.keys(cost,layout,"whole exact provisional cost layout")
                r.exact(cost["schema"],"flylab.m3.direct-science.full-return.v1" if request["mode"]=="generated" else "flylab.m3.contact-learning.full-return.v1","exact mode cost schema")
                r.exact(actual_outer["schema"],"flylab.m3.direct-science.outer.v1","actual supervisor outer schema")
                k.fee_values(cost,request,request_pin,trace,actual_outer,root)
                for field in ("all_owned_ended","reservation_released","worker_started","creation_attempted","worker_creation_state","resume_attempted","resume_outcome"):
                    r.exact(cost[field],actual_outer[field],"cost actual creation/closure "+field)
                return request,cost,actual_outer,k.closure(actual_outer),cost_pin
            request,paid,observed,closed=k.fee_proof(pin,row,source,root)
            fixed_pin=r.text_spec(k.BASE+"/parent-"+request["call_id"]+"-full-return.json",root)
            r.exact(pin,fixed_pin,"exact independently computed historical fee tuple")
            return request,paid,observed,closed,fixed_pin
        def project(ledger, root):
         """Return legal accounting state, permitting incurred negative remaining."""
         event, baseline, policy, grant = immutable_event, immutable_baseline, immutable_policy, immutable_grant
         expected = copy.deepcopy(baseline)
         expected[k.KEY] = copy.deepcopy(self.recovery)
         expected["direct_science_blocking_holds"] = {}
         source = event["source_freeze"]
         tail = ledger["debits"][327:]
         r.need(type(tail) is list and len(tail) <= len(k.SEQUENCE), "finite seven-call tail")
         caps = dict(k.CAPS)
         failed = False
         for index, debit in enumerate(tail):
          r.exact(debit["cost_id"], k.SEQUENCE[index], "next literal debit")
          r.need(not failed, "paid failure does not admit successors")
          q, cost, _outer, closed, fee_pin = fee(debit["descriptor"], debit, source, root)
          r.need(closed, "UNCOMMITTED_OWNERSHIP_UNKNOWN: no fee/debit/totals without known closure")
          reserved = q["limits"]["outer_s"]
          paid = {"cost_id":q["call_id"],"mode":q["mode"],"actual_outer_full_return_s":cost["actual_outer_full_return_s"],"actual_returned_exit_code":cost["exit_code"],"descriptor":fee_pin}
          r.exact(debit,paid,"independent whole paid row")
          expected["debits"].append(paid)
          expected["direct_science_calls"][q["call_id"]] = {"request": cost["request"] if q["mode"] == "generated" else cost["supervisor_request"],
                     "state": "charged", "mode": q["mode"], "reserved_s": reserved}
          failed = cost["exit_code"] != 0 or cost["cap_exceeded"] or not closed
          if not closed:
           k.expected_ownership(expected, q, expected["direct_science_calls"][q["call_id"]]["request"], ledger, root, charged=True)
           owner_key = ('direct_science_' if q['mode'] == 'generated' else 'startup_ray_original_') + 'workers'
           for field in ('owner_pid', 'owner_creation_ticks', 'worker_pid', 'worker_creation_ticks'):
            r.exact(_outer[field], expected[owner_key][q['call_id']][field], 'unclosed evidence retains admitted identity')
          if closed and cost['exit_code'] != 0:
           expected[('direct_science_' if q['mode'] == 'generated' else 'startup_ray_original_') + 'blocking_holds'][q['call_id']] = k.FAILURE_HOLD
          if cost["cap_exceeded"]:
           k.accounting_hold(expected, q['mode'], q['call_id'])
         if k.ALLOCATION_KEY in ledger:
          r.need(len(tail) >= 1, "allocation after actual bootstrap")
          k.metadata_pin(ledger[k.ALLOCATION_KEY], k.ALLOCATION_PATH)
          allocation = k.data(ledger[k.ALLOCATION_KEY], root)
          k.keys(allocation, ("schema", "epoch", "before_ledger", "recovery", "source_freeze", "bootstrap", "bootstrap_junit",
              "old_caps", "new_caps", "delta_s", "combined_global_s", "new_resource_grant_s", "refund_s"), "allocation event")
          r.exact(allocation["schema"], "flylab.m3.checkpoint-allocation.v1", "allocation schema")
          for field_name, field_value in (("epoch", k.EPOCH), ("recovery", ledger[k.KEY]), ("source_freeze", source), ("old_caps", k.CAPS),
             ("new_caps", k.CAPS), ("delta_s", {"generated_global": 0, "generated_local": 0, "original_global": 0}),
             ("combined_global_s", 19900), ("new_resource_grant_s", 0), ("refund_s", 0), ("bootstrap", tail[0]["descriptor"])):
           r.exact(allocation[field_name], field_value, "exact allocation " + field_name)
          before = k.data(allocation["before_ledger"], root, 524288)
          r.need(k.ALLOCATION_KEY not in before and len(before["debits"]) == 324, "fresh closed bootstrap before allocation")
          state = project(before, root)
          r.need(state["phase"] == "charged" and not state["failed"], "successful known charged prerequisite")
          for owner_key in r.OWNER_KEYS: r.exact(before[owner_key], {}, "closed eligible owner")
          r.need(before["generated_shared_remaining_s"] >= 0 and before["original_remaining_s"] >= 0 and before["direct_science_generated_used_s"] <= state["caps"]["generated_local"], "execution remaining resource")
          k.success(before["debits"][-1], source, root, "bootstrap", allocation["bootstrap_junit"])
          expected[k.ALLOCATION_KEY] = ledger[k.ALLOCATION_KEY]
          caps = dict(k.CAPS)
          for cap_name, cap_value in caps.items():
           expected[r.CAP_FIELDS[cap_name]] = cap_value
         else:
          r.need(len(tail) <= 1, "qualification cannot use old caps")
         if k.QUALIFICATION_KEY in ledger:
          r.need(k.ALLOCATION_KEY in ledger and len(tail) >= 5, "separate four actual costs before registration")
          k.metadata_pin(ledger[k.QUALIFICATION_KEY], k.QUALIFICATION_PATH)
          k.qualification(k.data(ledger[k.QUALIFICATION_KEY], root), ledger, source, root)
          expected[k.QUALIFICATION_KEY] = ledger[k.QUALIFICATION_KEY]
         r.need(len(tail) <= 5 or k.QUALIFICATION_KEY in ledger, "Original requires registered K qualification")
         expected.update(k.totals(expected, policy, grant, caps))
         active_calls = set(ledger["direct_science_calls"]) - set(expected["direct_science_calls"])
         r.need(len(active_calls) <= 1, "at most next admitted call")
         phase = "charged"
         if active_calls:
          r.need(not failed and len(tail) < len(k.SEQUENCE), "no call after failure/finite end")
          call = k.SEQUENCE[len(tail)]
          r.exact(active_calls, {call}, "no old/replay/alias admission")
          admitted = ledger["direct_science_calls"][call]
          k.keys(admitted, ("request", "state", "mode", "reserved_s"), "whole admission")
          request_path = k.BASE+"/s28-d19bz-full4-request.json" if call == k.BOOTSTRAP_CALL else admitted["request"]["path"]
          k.metadata_pin(admitted["request"],request_path)
          request_pin = r.text_spec(request_path,root)
          q = k.data(request_pin, root)
          k.identity(q, request_pin, source, root)
          r.exact(admitted, {"request": request_pin, "state": "admitted", "mode": q["mode"], "reserved_s": q["limits"]["outer_s"]}, "exact admitted identity")
          r.need(len(tail) != 1 or k.ALLOCATION_KEY in ledger, "no lint before allocation")
          r.need(len(tail) != 5 or k.QUALIFICATION_KEY in ledger, "no body before K qualification")
          if len(tail) == 6:
           k.valid_body(tail[5], source, k.data(q['science_request'], root), root)
          expected["direct_science_calls"][call] = {"request":request_pin,"state":"admitted","mode":q["mode"],"reserved_s":q["limits"]["outer_s"]}
          phase = k.expected_ownership(expected, q, request_pin, ledger, root)
         # A legitimate cumulative breach adds the exact responsible HOLD, never erases a debit.
         if tail:
          last = tail[-1]
          breached = ((last["mode"] == "generated" and (expected["direct_science_generated_used_s"] > caps["generated_local"]
             or expected["generated_shared_remaining_s"] < 0)) or
             (last["mode"] == "original" and (expected["original_remaining_s"] < 0
             or expected["original_stage_used_s"] > 6300 or expected["startup_ray_original_local_used_s"] > 6000)))
          if breached:
           k.accounting_hold(expected, last['mode'], last['cost_id'])
         semantic_equal(ledger, expected)
         return {"caps": caps, "completed": len(tail), "phase": phase, "failed": failed, "source": source, "epoch": k.EPOCH, "producer_state": expected}

        token={"schema":"flylab.m3.direct-science.lock-owner.v1","call_id":self.q["call_id"],"request":self.request_spec,"owner_pid":os.getpid(),"owner_creation_ticks":creation_ticks(os.getpid()),"entry_counter_s":time.perf_counter(),"job_name":"Local\\generated-synthetic-fixture-fee"}
        lock=r.SharedLock(self.root/r.SHARED_LOCK,token)
        try:
            lock.authenticate(); r.need(not lock.uncertain,"certain common fee capability")
            before_pin=r.text_spec(self.ledger_path,self.root,524288);before_raw=r.raw_pin(before_pin,self.root,524288);before=json.loads(before_raw)
            whole(before,self.ledger); project(before,self.root)
            if (self.root/cost_path).exists():
                cost_pin=r.text_spec(cost_path,self.root)
                raw=r.raw_pin(cost_pin,self.root,262144)
                semantic_equal(k.data(cost_pin,self.root),cost)
                debit["descriptor"]=cost_pin
            candidate=copy.deepcopy(before);candidate["debits"].append(debit);candidate["direct_science_calls"][self.q["call_id"]]["state"]="charged"
            for key in ("direct_science_workers","direct_science_reservations","direct_science_pending_actuals"):candidate[key]={}
            state=project(before,self.root);candidate.update(k.totals(candidate,immutable_policy,immutable_grant,state["caps"]))
            if exit_code!=0:candidate["direct_science_blocking_holds"][self.q["call_id"]]=k.FAILURE_HOLD
            if cost["cap_exceeded"] or candidate["generated_shared_remaining_s"]<0 or candidate["direct_science_generated_used_s"]>state["caps"]["generated_local"]: k.accounting_hold(candidate,'generated',self.q['call_id'])
            proved=project(candidate,self.root)["producer_state"]
            candidate_raw=json.dumps(candidate,ensure_ascii=True,allow_nan=False,sort_keys=True).encode("utf8");r.need(len(candidate_raw)<=524288,"bounded complete candidate before either publication")
            lock.authenticate();r.exact(r.raw_pin(before_pin,self.root,524288),before_raw,"fresh complete before both publications")
            target=self.root/cost_path
            if target.exists():r.exact(r.raw_pin(cost_pin,self.root,262144),raw,"only fully proven exact uncommitted cost")
            else:
                lock.uncertain=True;self.put(cost_path,cost);r.exact(r.raw_pin(cost_pin,self.root,262144),raw,"whole exact cost readback");lock.authenticate();r.exact(r.raw_pin(before_pin,self.root,524288),before_raw,"unchanged canonical after cost");lock.uncertain=False
            r.atomic_ledger_publish(self.root/self.ledger_path,before_raw,candidate,lock,validate=lambda value:semantic_equal(value,proved))
            published=r.raw_pin(r.text_spec(self.ledger_path,self.root,524288),self.root,524288);self.ledger=json.loads(published);whole(self.ledger,sorted_state(proved));self.cost_spec=cost_pin
            return debit
        except Exception:
            lock.uncertain=True
            raise
        finally:
            if not lock.uncertain:lock.release(True)

    def native(self, body):
        from tests.test_c_banc_contact_runtime_journal import (
            conserved_native_definitions,
            conserved_run_native,
        )
        host = shutil.which("pwsh")
        r.need(host is not None, "actual Native qualification requires pwsh; never skipped")
        source = conserved_native_definitions(self.root) + "\n" + body
        observation = observation_directory(self)
        wrapper = observation / "generated-native.ps1"
        with wrapper.open("xb") as stream:
            stream.write(source.encode("utf8"))
        return conserved_run_native(self.root, [host, "-NoProfile", "-File", str(wrapper)], observation_root=observation)

    def native_batch(self, blocks):
        """Finite same-root cases; one actual observation and ordered receipts."""
        parts = ["$batchEnd=[long]0;"]
        for ordinal, (name, body) in enumerate(blocks):
            r.need(name.replace("-", "").isalnum(), "literal generated case tag")
            parts.append("$start=[Diagnostics.Stopwatch]::GetTimestamp();try{$payload=@(& {\n" + body + "\n});}catch{$r33Error=$_;try{[Console]::Error.WriteLine('R33 CALLER '+$r33Error.ScriptStackTrace)}catch{};throw $r33Error};$end=[Diagnostics.Stopwatch]::GetTimestamp();"
                         "Need ($null -eq $script:ledgerTransaction) 'Closed case transaction';"
                         "$timing=@{start_ticks=$start;end_ticks=$end;frequency=[Diagnostics.Stopwatch]::Frequency};"
                         "K-Keys $timing @('start_ticks','end_ticks','frequency');"
                         "foreach($n in @('start_ticks','end_ticks','frequency')){$null=K-Int $timing[$n]};"
                         "Need ($start -ge 0 -and $start -ge $batchEnd -and $end -ge $start -and $timing.frequency -gt 0) 'Ordered actual case timing';$batchEnd=$end;"
                         "@{ordinal=[long]" + str(ordinal) + ";case='" + name
                         + "';root=$taskRoot;status='PASS';payload=@($payload);timing=$timing}"
                         "|ConvertTo-Json -Depth 100 -Compress")
        observed = self.native("\n".join(parts))
        observed_pass(observed)
        receipts = [json.loads(line) for line in observed.stdout.splitlines()]
        r.need(len(receipts) == len(blocks), "exact Native batch receipt count")
        previous_end = 0
        frequency = None
        for ordinal, ((name, _body), receipt) in enumerate(zip(blocks, receipts, strict=True)):
            k.keys(receipt, ("ordinal", "case", "root", "status", "payload", "timing"), "Native receipt")
            r.exact({key: receipt[key] for key in ("ordinal", "case", "root", "status")},
                    {"ordinal": ordinal, "case": name, "root": str(self.root.absolute()), "status": "PASS"},
                    "whole ordered actual case receipt")
            timing = receipt["timing"]
            k.keys(timing, ("start_ticks", "end_ticks", "frequency"), "Native timing")
            r.need(all(type(v) is int for v in timing.values()), "Native ticks are non-bool integers")
            r.need(0 <= previous_end <= timing["start_ticks"] <= timing["end_ticks"]
                   and timing["frequency"] > 0, "ordered actual completed-case timing")
            if frequency is not None:
                r.exact(timing["frequency"], frequency, "same Native batch tick frequency")
            previous_end, frequency = timing["end_ticks"], timing["frequency"]
            r.need(type(receipt["payload"]) is list and all(type(v) is str for v in receipt["payload"]),
                   "original Native stdout strings, never invented process status")
        return receipts

    def native_commit(self, *, entry_marker=False):
        marker = "Write-Output 'ENTER Start-LedgerTransaction';" if entry_marker else ""
        commit_marker = "Write-Output 'ENTER K-Commit';" if entry_marker else ""
        return self.native(marker + """
$null=Start-LedgerTransaction 'generated-synthetic-L-commit'
try{$l=Conserved-Json $script:ledgerTransaction.before;MARKER$receipt=K-Commit $l 'TRACE';$receipt|ConvertTo-Json -Depth 100 -Compress}
finally{if($null -ne $script:ledgerTransaction -and -not $script:ledgerTransaction.uncertain){Stop-LedgerTransaction $script:ledgerTransaction}}
""".replace("TRACE", self.trace_spec["path"]).replace("MARKER", commit_marker))


def observed_pass(result):
    assert result.returncode == 0, (result.stdout.decode("utf8", errors="replace") + result.stderr.decode("utf8", errors="replace"), result.first_error)
    assert result.all_owned_ended and result.stdout_eof and result.stderr_eof and not result.unknown_owned_identities
    assert result.first_error is None and not result.secondary_errors and not result.timeout


def observed_rejection(result):
    assert result.returncode != 0
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



def d19_root_binding_controls(k, fixture, resource):
 need,read_pin,raw_pin,d19_root_bind_span,d19_r1_defs,_text_spec,BASE,d19_json,d19_whole=(getattr(resource,n) for n in ('need','read_pin','raw_pin','d19_root_bind_span','d19_r1_defs','text_spec','BASE','d19_json','d19_whole'))
 import re
 import shutil

 from tests import banc_m3_checkpoint_fixture as legacy
 from tests.banc_m3_checkpoint_harness_fixture import observation_directory
 from tests.test_c_banc_contact_runtime_journal import conserved_run_native
 source = read_pin(fixture.source, fixture.root, 262144)
 account = source['sources'][BASE+'/parent-direct-science-account.ps1']
 contract = read_pin(source['checkpoint_contract'], fixture.root, 131072)
 caps=contract['new_caps'];need(type(caps) is dict and set(caps)=={'generated_global','generated_local','original_global'} and all(type(v) is int and v>=0 for v in caps.values()),'D19 declared nonbool caps')
 host = raw_pin(account, fixture.root, 393216)
 span = d19_root_bind_span(contract, host)
 native_path = 'tools/banc_m3_checkpoint_harness_runtime_native.ps1'
 native = source['sources'][native_path]
 raw_pin(native, fixture.root, 131072)
 for malformed_bytes in (True, 262145):
  malformed = dict(legacy.K_ACCOUNT, bytes=malformed_bytes)
  refused = False
  try: d19_r1_defs(fixture.root, malformed, legacy.K_CONTACT)
  except ValueError: refused = True
  need(refused, 'D19 explicit historical nonbool/cap refusal')
 executable = shutil.which('pwsh')
 need(executable is not None, 'D19 cold binding requires real pwsh, never skip')
 for route in ('current', 'explicit-historical'):
  definitions = d19_r1_defs(fixture.root) if route == 'current' else d19_r1_defs(fixture.root, legacy.K_ACCOUNT, legacy.K_CONTACT)
  text = span.decode('utf8')
  exports=sorted(re.findall(r'^function script:(D19-[A-Za-z0-9-]+)(?=[ (\r\n])',text,re.MULTILINE))
  need(len(exports)==len(set(exports)) and len(exports)>0,'D19 exact independently pinned declaration names')
  need(definitions.count(text) == 1, 'D19 exactly one producer definition span')
  cold = definitions.replace(text, '', 1)
  need('function script:D19-' not in cold and 'function D19-' not in cold, 'D19 no preloaded names in cold common context')
  body = r'''
$bindingNames=@(Get-Command D19-* -CommandType Function -ErrorAction SilentlyContinue);Need ($bindingNames.Count -eq 0) 'D19 genuine cold function namespace'
Set-Variable -Name a -Scope Script -Value 'D19 caller scratch a' -Option None -Force
Set-Variable -Name b -Scope Script -Value ([byte[]]@(11,22,33)) -Option None -Force
Need (-not ((Get-Variable a -Scope Script).Options -band [Management.Automation.ScopedItemOptions]::AllScope)) 'D19 ordinary a'
Need (-not ((Get-Variable b -Scope Script).Options -band [Management.Automation.ScopedItemOptions]::AllScope)) 'D19 ordinary b'
$bindingNative=Safe 'tools/banc_m3_checkpoint_harness_runtime_native.ps1'
$bindingBefore=@{};foreach($n in @('KManagedSourceAccepted','KCatalogManifestContract','KManagedCache','KCatalogCache')){$v=Get-Variable -Name $n -Scope Script -ErrorAction SilentlyContinue;$bindingBefore[$n]=if($null -eq $v){$null}else{$v.Value}}
foreach($reload in @(0,1)){
 $bindingOutput=@(. $bindingNative);Need ($bindingOutput.Count -eq 0) 'D19 declaration binding has zero success output'
 Same $script:a 'D19 caller scratch a' 'D19 parent a unchanged';Same $script:b ([byte[]]@(11,22,33)) 'D19 parent b unchanged'
 $bindingExpected=__EXPORTS__;$bindingNames=@(Get-Command D19-* -CommandType Function -ErrorAction SilentlyContinue);Need ($bindingNames.Count -eq $bindingExpected.Count) 'D19 exact surviving script exports';$bindingActual=@($bindingNames.Name);[Array]::Sort($bindingActual,[StringComparer]::Ordinal);D19-R4Whole $bindingActual $bindingExpected
 $positive=@(D19-Caps);Need ($positive.Count -eq 1) 'D19 functional one-value output';Same $positive[0] __CAPS__ 'D19 actual caps function'
 Need (@(D19-R4Need $true 'D19 no output').Count -eq 0) 'D19 assertion output0'
 foreach($n in $bindingBefore.Keys){$v=Get-Variable -Name $n -Scope Script -ErrorAction SilentlyContinue;$now=if($null -eq $v){$null}else{$v.Value};Need ([object]::ReferenceEquals($now,$bindingBefore[$n])) 'D19 binding never accepts Source or initializes backend/cache'}
}
$bindingControls=[Collections.Generic.List[string]]::new();$bindingControls.Add('cold');$bindingControls.Add('reload')
if('__ROUTE__' -ceq 'current'){
 $bindingContractPath=Safe ($kb+'/c5.json');$bindingContractRaw=[IO.File]::ReadAllBytes($bindingContractPath)
 $bindingAccountPath=Safe ($kb+'/parent-direct-science-account.ps1');$bindingAccountRaw=[IO.File]::ReadAllBytes($bindingAccountPath)
 $bindingContract=K-Json $bindingContractRaw;$bindingStart=[int]$bindingContract.d19_bind[1];$bindingEnd=[int]$bindingContract.d19_bind[2]
 foreach($case in @('c5-raw','tuple-shape','tuple-short','tuple-path','tuple-bool','tuple-float','tuple-negative','tuple-range','tuple-hash','tuple-case')){
  $v=K-Json $bindingContractRaw
  switch($case){
   'c5-raw'{$changed=[byte[]]($bindingContractRaw+@([byte]32))}
   'tuple-shape'{$v.d19_bind=@{path=$v.d19_bind[0]}}
   'tuple-short'{$v.d19_bind=@($v.d19_bind[0..2])}
   'tuple-path'{$v.d19_bind[0]='alias.ps1'}
   'tuple-bool'{$v.d19_bind[1]=$true}
   'tuple-float'{$v.d19_bind[1]=[double]$v.d19_bind[1]+0.5}
   'tuple-negative'{$v.d19_bind[1]=-1L}
   'tuple-range'{$v.d19_bind[2]=393216L}
   'tuple-hash'{$v.d19_bind[3]='0'*64}
   'tuple-case'{$v.d19_bind[3]=$v.d19_bind[3].ToLowerInvariant()}
  }
  if($case -cne 'c5-raw'){$changed=$utf8.GetBytes(($v|ConvertTo-Json -Depth 100 -Compress))}
  $beforeFunction=(Get-Command D19-Caps -CommandType Function).ScriptBlock;$refused=$false
  try{[IO.File]::WriteAllBytes($bindingContractPath,$changed);try{$unexpected=@(. $bindingNative)}catch{$refused=$true}}finally{[IO.File]::WriteAllBytes($bindingContractPath,$bindingContractRaw)}
  Need $refused ('D19 actual compiled c5 refusal '+$case);Need ([object]::ReferenceEquals((Get-Command D19-Caps).ScriptBlock,$beforeFunction)) 'D19 no unchecked definition replacement'
  $bindingControls.Add($case)
 }
 foreach($case in @('span-byte','host-truncated','host-over-cap','host-UTF8')){
  $changed=[byte[]]$bindingAccountRaw.Clone()
  switch($case){
   'span-byte'{$changed[$bindingStart]=[byte]($changed[$bindingStart] -bxor 1)}
   'host-truncated'{$changed=[byte[]]$changed[0..($bindingEnd-1)]}
   'host-over-cap'{$large=[byte[]]::new(393217);[Array]::Copy($changed,0,$large,0,$changed.Length);$changed=$large}
   'host-UTF8'{$changed[$bindingStart]=[byte]255}
  }
  $beforeFunction=(Get-Command D19-Caps).ScriptBlock;$refused=$false
  try{[IO.File]::WriteAllBytes($bindingAccountPath,$changed);try{$unexpected=@(. $bindingNative)}catch{$refused=$true}}finally{[IO.File]::WriteAllBytes($bindingAccountPath,$bindingAccountRaw)}
  Need $refused ('D19 actual host reader refusal '+$case);Need ([object]::ReferenceEquals((Get-Command D19-Caps).ScriptBlock,$beforeFunction)) 'D19 host refusal installs no definitions'
  $bindingControls.Add($case)
 }
 $bindingTarget=Safe 'generated-D19-bind-target';$bindingLink=Safe 'generated-D19-bind-link'
 $null=[IO.Directory]::CreateDirectory($bindingTarget);[IO.File]::WriteAllBytes([IO.Path]::Combine($bindingTarget,'account.ps1'),$bindingAccountRaw)
 try{
  if([OperatingSystem]::IsWindows()){$null=New-Item -ItemType Junction -Path $bindingLink -Target $bindingTarget -ErrorAction Stop}else{$null=[IO.Directory]::CreateSymbolicLink($bindingLink,$bindingTarget)}
  Need (([IO.File]::GetAttributes($bindingLink) -band [IO.FileAttributes]::ReparsePoint) -ne 0) 'D19 actual reparse control exists'
  $refused=$false;try{$unexpected=K-SourceRaw (Safe 'generated-D19-bind-link/account.ps1') 393216}catch{$refused=$true};Need $refused 'D19 actual same host reparse reader refusal'
 }finally{if([IO.Directory]::Exists($bindingLink)){Need (([IO.File]::GetAttributes($bindingLink) -band [IO.FileAttributes]::ReparsePoint) -ne 0) 'D19 remove only verified junction';[IO.Directory]::Delete($bindingLink,$false)}}
 $bindingControls.Add('host-reparse')
 $bindingSource=K-BoundedPin (K-Paths).source;$null=K-SourceFence $bindingSource
 try{
  $changed=[byte[]]::new(393216);[Array]::Copy($bindingAccountRaw,0,$changed,0,$bindingAccountRaw.Length)
  for($i=$bindingAccountRaw.Length;$i -lt $changed.Length;$i++){$changed[$i]=[byte]32}
   if($changed.Length -eq $bindingAccountRaw.Length){$pos=[Array]::IndexOf($bindingAccountRaw,[byte]32);Need ($pos -ge 0 -and $pos -lt $bindingStart -and $changed[$pos] -eq 32) 'D19 specified first outside-span space';$changed[$pos]=[byte]9}
   Need ([Convert]::ToBase64String($changed) -cne [Convert]::ToBase64String($bindingAccountRaw)) 'D19 real exact-cap mutation';Same ([Convert]::ToBase64String([byte[]]$changed[$bindingStart..$bindingEnd])) ([Convert]::ToBase64String([byte[]]$bindingAccountRaw[$bindingStart..$bindingEnd])) 'D19 unchanged authenticated span'
  [IO.File]::WriteAllBytes($bindingAccountPath,$changed)
  Need (@(. $bindingNative).Count -eq 0) 'D19 actual exact393216 reader positive'
  $refused=$false;try{$null=K-SourceFence $bindingSource}catch{$refused=$true};Need $refused 'D19 exact-cap binding is not whole Source acceptance'
 }finally{[IO.File]::WriteAllBytes($bindingAccountPath,$bindingAccountRaw)}
 $bindingControls.Add('host-exact-cap-before-SourceFence')
 try{
  $changed=if($bindingAccountRaw.Length -lt 393216){[byte[]]($bindingAccountRaw+@([byte]10))}else{[byte[]]$bindingAccountRaw.Clone()};if($bindingAccountRaw.Length -eq 393216){$pos=[Array]::IndexOf($bindingAccountRaw,[byte]32);Need ($pos -ge 0 -and $pos -lt $bindingStart -and $changed[$pos] -eq 32) 'D19 specified first outside-span space';$changed[$pos]=[byte]9}
   Need ($changed.Length -le 393216 -and [Convert]::ToBase64String($changed) -cne [Convert]::ToBase64String($bindingAccountRaw)) 'D19 bounded real outside-span mutation';Same ([Convert]::ToBase64String([byte[]]$changed[$bindingStart..$bindingEnd])) ([Convert]::ToBase64String([byte[]]$bindingAccountRaw[$bindingStart..$bindingEnd])) 'D19 outside span unchanged'
   [IO.File]::WriteAllBytes($bindingAccountPath,$changed)
  Need (@(. $bindingNative).Count -eq 0) 'D19 outside-span binding alone may succeed'
  $refused=$false;try{$null=K-SourceFence $bindingSource}catch{$refused=$true};Need $refused 'D19 outside-span drift refused by actual whole Source fence before action'
 }finally{[IO.File]::WriteAllBytes($bindingAccountPath,$bindingAccountRaw)}
 $bindingControls.Add('outside-span-SourceFence')
   $bindingOrdered=[ordered]@{path=$bindingSource.path;bytes=$bindingSource.bytes;sha256=$bindingSource.sha256};$bindingPermuted=[ordered]@{sha256=$bindingSource.sha256;bytes=$bindingSource.bytes;path=$bindingSource.path}
  foreach($p in @($bindingOrdered,$bindingPermuted)){$null=D19-R1Closure $p};$bindingControls.Add('Source-pin-order-positive')
  foreach($case in @('bool','float','extra','missing','alias')){
   $p=[ordered]@{path=$bindingSource.path;bytes=$bindingSource.bytes;sha256=$bindingSource.sha256}
   switch($case){'bool'{$p.bytes=$true};'float'{$p.bytes=[double]$p.bytes};'extra'{$p.extra=0L};'missing'{$p.Remove('sha256')};'alias'{$p.path='alias.json'}}
   $refused=$false;try{$null=D19-R1Closure $p}catch{$refused=$true};Need $refused ('D19 Source pin refusal '+$case);$bindingControls.Add('Source-pin-'+$case)
  }
$bindingClosurePath=Safe ($kb+'/s28-f19-root/transition64/ADOPTION.json');$bindingClosureRaw=[IO.File]::ReadAllBytes($bindingClosurePath)
 foreach($case in @('closure-missing','closure-forged')){
  try{
   if($case -ceq 'closure-missing'){[IO.File]::Delete($bindingClosurePath)}else{$v=K-Json $bindingClosureRaw;$v.verdict='FORGED';[IO.File]::WriteAllBytes($bindingClosurePath,$utf8.GetBytes(($v|ConvertTo-Json -Depth 100 -Compress)))}
   if($case -ceq 'closure-missing'){Need (-not [IO.File]::Exists($bindingClosurePath)) 'D19 actual consumed closure missing'}else{Need ([Convert]::ToBase64String([IO.File]::ReadAllBytes($bindingClosurePath)) -cne [Convert]::ToBase64String($bindingClosureRaw)) 'D19 actual consumed closure changed'}
    $refused=$false;try{$null=K-SourceFence $bindingSource}catch{$refused=$true};Need $refused ('D19 actual new closure refusal '+$case)
  }finally{[IO.File]::WriteAllBytes($bindingClosurePath,$bindingClosureRaw)}
  $bindingControls.Add($case)
 }
 Same ([Convert]::ToBase64String([IO.File]::ReadAllBytes($bindingContractPath))) ([Convert]::ToBase64String($bindingContractRaw)) 'D19 full contract control restoration'
 Same ([Convert]::ToBase64String([IO.File]::ReadAllBytes($bindingAccountPath))) ([Convert]::ToBase64String($bindingAccountRaw)) 'D19 full account control restoration'
 Same ([Convert]::ToBase64String([IO.File]::ReadAllBytes($bindingClosurePath))) ([Convert]::ToBase64String($bindingClosureRaw)) 'D19 full closure control restoration'
}
@{schema='flylab.m3.d19.actual-native-binding-controls.v1';route='__ROUTE__';controls=$bindingControls.ToArray();functions=[long]$bindingExpected.Count;exports=$bindingActual;output0=$true;scratch_preserved=$true;runtime_or_gait_pass=$false}|ConvertTo-Json -Depth 10 -Compress
'''.replace('__ROUTE__', route).replace('__EXPORTS__',"@("+','.join("'"+n+"'" for n in exports)+")").replace('__CAPS__','@{'+ ';'.join(n+'='+str(caps[n])+'L' for n in ('generated_global','generated_local','original_global'))+'}')
  observation = observation_directory(fixture)
  wrapper = observation/'generated-D19-binding.ps1'
  with wrapper.open('xb') as stream: stream.write((cold+'\n'+body).encode('utf8'))
  result = conserved_run_native(fixture.root, [executable,'-NoProfile','-File',str(wrapper)], observation_root=observation)
  need(result.returncode == 0 and result.all_owned_ended is True and result.stdout_eof is True and result.stderr_eof is True, 'D19 cold Native controls known real closure '+route+': '+result.stderr.decode('utf8',errors='replace')[-2048:])
  lines = result.stdout.splitlines()
  need(len(lines) == 1, 'D19 exact one actual control receipt, no binder output')
  receipt = d19_json(lines[0])
  expected = ['cold','reload']
  if route == 'current': expected += ['c5-raw','tuple-shape','tuple-short','tuple-path','tuple-bool','tuple-float','tuple-negative','tuple-range','tuple-hash','tuple-case','span-byte','host-truncated','host-over-cap','host-UTF8','host-reparse','host-exact-cap-before-SourceFence','outside-span-SourceFence','Source-pin-order-positive','Source-pin-bool','Source-pin-float','Source-pin-extra','Source-pin-missing','Source-pin-alias','closure-missing','closure-forged']
  d19_whole(receipt,{'schema':'flylab.m3.d19.actual-native-binding-controls.v1','route':route,'controls':expected,'functions':len(exports),'exports':exports,'output0':True,'scratch_preserved':True,'runtime_or_gait_pass':False},False)
  need(raw_pin(account,fixture.root,393216) == host, 'D19 unchanged full host after actual controls')


def d19_fee_refusal(k,error,variant,candidate,before_source,contract,predecessor_source):
    import copy
    import traceback
    expected=copy.deepcopy(contract);source=copy.deepcopy(predecessor_source)
    if variant=='reviewed-source-mismatch':
        assert source['reviewed_source']['sha256']!='0'*64
        source['reviewed_source']['sha256']='0'*64
    else:
        rows=expected['actual320']['archives'];rr=[x for x in rows if x['logical_current_before']['path']==k.REVIEW_PATH]
        assert len(rr)==1 and variant in ('missing-review','duplicate-review')
        expected['actual320']['archives']=[x for x in rows if x['logical_current_before']['path']!=k.REVIEW_PATH]+(rr*2 if variant=='duplicate-review' else [])
    k.whole_exact(candidate,expected);k.whole_exact(before_source,source)
    frames=traceback.extract_tb(error.__traceback__)
    runtime=[x.name for x in frames if x.filename.replace('\\','/').endswith('/tools/banc_m3_checkpoint_harness_runtime.py')]
    if variant!='reviewed-source-mismatch':
        assert str(error) in ('fee unique predecessor review','unique predecessor review') and 'predecessor' in runtime,str(error)
    elif str(error)=='scalar':
        assert 'predecessor' in runtime and runtime[-1]=='whole_exact',str(error)
    else:
        assert str(error)=='whole independent fixture fee' and any(x.name=='whole' and x.filename.replace('\\','/').endswith('/tests/banc_m3_checkpoint_harness_fixture.py') for x in frames),str(error)


def d19_golden(k,fixture):
    return k.data(k.data(k.CURRENT_CONTRACT,fixture.root)['d19_current_golden'],fixture.root)


def d19_independent_totals(k,fixture,ledger):
    g=d19_golden(k,fixture);p=k.data(g['policy'],fixture.root);a=k.data(g['grant'],fixture.root)
    generated=original=0.0
    for ordinal,debit in enumerate(ledger['debits']):
        if ordinal>=118 and debit['mode']=='generated':generated+=float(debit['actual_outer_full_return_s'])
        if ordinal>=165 and debit['mode']=='original':original+=float(debit['actual_outer_full_return_s'])
    stage=float(p['generated_stage_before_s'])+generated
    return {'direct_science_generated_used_s':generated,'generated_stage_used_s':stage,'generated_shared_cumulative_used_s':float(p['generated_shared_before_s'])+generated,'generated_shared_remaining_s':float(g['new_caps']['generated_global'])-float(p['generated_shared_baseline_s'])-stage,'original_cumulative_used_s':float(a['original_cumulative_before_s'])+original,'original_stage_used_s':float(a['original_stage_before_s'])+original,'startup_ray_original_local_used_s':float(a['science_local_used_before_s'])+original,'original_remaining_s':float(g['new_caps']['original_global'])-float(a['original_cumulative_before_s'])-original}


def d19_snapshot_expected(k,fixture,recovery):
    import copy

    from tools import banc_m3_conserved_resource as resource
    g=d19_golden(k,fixture);v=k.data(g['independent_baseline'],fixture.root,524288)
    assert len(v['debits'])==g['baseline_debits'] and len(v['direct_science_calls'])==g['baseline_calls']
    assert v['direct_science_blocking_holds']=={'direct-science-20261009-contact-s28-bootstrap-r63':'L stage unsuccessful; successor blocked'}
    v[k.KEY]=copy.deepcopy(recovery);v['direct_science_blocking_holds']={}
    if k.data(k.CURRENT_CONTRACT,fixture.root)['d19_predecessor_allocation_present']:v.pop(k.ALLOCATION_KEY)
    for name,value in g['new_caps'].items():v[resource.CAP_FIELDS[name]]=value
    cal=d19_independent_totals(k,fixture,v)
    resource.d19_whole(cal,g['totals']);v.update(cal)
    return v


def d19_native_history(k,fixture):
    """Bind the actual finite historical consumer; never replay its producer."""
    import base64
    import hashlib
    g=d19_golden(k,fixture);hc=g['native_historical_context']
    assert hc['seed_only'] is True and hc['producer_execution_claim'] is False
    sf=k.data(fixture.source,fixture.root)
    native=k.r.raw_pin(sf['sources']['tools/banc_m3_checkpoint_harness_runtime_native.ps1'],fixture.root,131072)
    start=native.index(b'function K-Recovery($_p){return D19-R $_p;')
    stop=native.index(b'\nfunction K-RecoveredSnapshot',start)
    lines=native[start:stop].splitlines(True);parts={}
    for name,prefix in (('guard',b'$d3=$false;try{'),('seed',b'$c8=@{anchor=$t;'),('key',b'$key=@($taskRoot,'),('tail',b'if(-not $script:KPC.ContainsKey($key))'),('cleanup',b'}catch{$script:KManagedSourceAccepted=')):
        selected=[v for v in lines if v.startswith(prefix)];assert len(selected)==1
        value=selected[0]
        if name=='guard':value=value[:value.index(b'K-PinShape $_p')]
        expected=hc['spans'][name]
        assert len(value)==expected['bytes'] and hashlib.sha256(value).hexdigest()==expected['sha256'] and value==base64.b64decode(expected['raw_b64'])
        parts[name]=value.decode('utf8')
    for alias in hc['archives'].values():k.r.raw_pin(alias,fixture.root,262144)
    k.r.raw_pin(hc['event_archive'],fixture.root,262144)
    snapshot=fixture.put(k.BASE+'/generated-r14-snapshot.json',d19_snapshot_expected(k,fixture,fixture.recovery))
    prefix="$script:R14SnapshotPin=K-Json '"+json.dumps(snapshot,separators=(',',':'))+"';\n"
    prefix+="$script:R14Legacy=[scriptblock]::Create('"+("param($key,$c8,$v)\n"+parts['guard']+parts['tail']+parts['cleanup']).replace("'","''")+"');\n"
    setup="""
& {
 $g=K-Read (K-Read $script:KCurrentContract).d19_current_golden;$hc=$g.native_historical_context;$cache=$g.historical_cache
 Need ($hc.seed_only -ceq $true -and $hc.producer_execution_claim -ceq $false -and $hc.key_components -eq 16) 'Historical seed only; actual consumer, no producer claim'
 $f=K-Read $hc.archives.source;$frozenContract=K-Read $hc.archives.contract;$rv=K-Read $hc.archives.review;$v=K-Read $hc.event_archive
 Same @($f.checkpoint_contract,$f.reviewed_source,$v.source_freeze,$v.review,$v.before_ledger) @($cache.contract,$cache.review,$cache.source,$cache.review,$cache.historical_value[0]) 'Whole historical logical archive/event linkage'
 K-Same $f.sources $rv.sources 'Historical full Source/review vector'
 Same $zz 'reviewed_source' 'Actual key lexical review field';Same $z8 'anchor' 'Actual tail lexical anchor field'
 $sf=$cache.source;$cp=$cache.contract;$p=K-S15Pins;K-Same $p $frozenContract.pinned_checkpoint.pins 'Historical fixed pin mapping'
 $d4=@(foreach($name in @('root_design','astra_markdown','astra_json','root_reflection')){$pin=$frozenContract.provenance[$name];$null=K-Raw $pin;$pin})
 Same @($d4[1],$d4[2],$d4[3]) @($rv.astra_review.markdown,$rv.astra_review.json,$rv.root_reflection) 'Historical direct provenance linkage'
 foreach($pin in @($rv.root_reviews)+@($rv.sol_reports)){$null=K-Raw $pin}
 $c5=K-Read $p.previous_contract;$b5=$c5.pinned_checkpoint.pins;$c1=K-Read $b5.technical_refusal;$gate=K-Read $p.technical_refusal;$receipt=K-Read $gate.actual_tool_return
 $c6=@($b5.previous_source,$b5.previous_contract,$b5.technical_refusal,$c1.actual_tool_return,$c1.actual_known_layout_compatibility_diagnostic,$c1.journal,$gate.actual_tool_return,$gate.actual_known_compiler_type_refusal,$gate.journal,$receipt.request)
 Need ($c6.Length -eq 10) 'Complete historical c6';foreach($pin in $c6){$null=K-Raw $pin (K-ReaderCap $pin.path)}
 $pins=@(foreach($pin in $frozenContract.actual320.frozen31){$null=K-Raw $pin (K-ReaderCap $pin.path);$row=[ordered]@{};foreach($name in $pin.Keys){$row[$name]=if($name -ceq 'bytes'){[long]$pin[$name]}else{$pin[$name]}};$row});Need ($pins.Length -eq 31) 'Complete ordered historical31'
 $b6=K-Clone $frozenContract.actual320;$b6.event_keys=$frozenContract.terminal_bridge.event_keys;$b6.event_schema=$frozenContract.terminal_bridge.event_schema
 $rp=K-CatalogVerified (K-Read $p.catalog);K-Same $rp $cache.policy_rows 'Whole actual779 catalog policy'
 $t=K-Read $cache.historical_value[0] 524288;Need ($t.debits.Length -eq 322 -and $t.direct_science_calls.Count -eq 204) 'Independently authenticated historical322'
 SEEDKEY
 $script:R14Seed=K-Clone $c8;$script:R14Key=$key;$script:R14Event=$v
}
""".replace('SEEDKEY',parts['seed']+parts['key'])
    check="""
$script:R14Check={param($h)
 $gg=K-Read (K-Read $script:KCurrentContract).d19_current_golden;K-Keys $h @('anchor','policy','grant','event')
 D19-R4Whole $h.anchor (K-Read $gg.historical_cache.historical_value[0] 524288)
 D19-R4Whole $h.policy (K-Read $gg.historical_cache.historical_value[1])
 D19-R4Whole $h.grant (K-Read $gg.historical_cache.historical_value[2])
 D19-R4Whole $h.event (K-Read $gg.native_historical_context.event_archive)
}
"""
    return prefix+setup+check


def d19_native_allocation(k):
    return """$null=Start-LedgerTransaction 'generated-synthetic-L-allocation'
try{
 $l=Conserved-Json $script:ledgerTransaction.before;$p=K-BoundedPin 'ALLOCATION';$e=K-Clone $l;$e[(K-AllocationKey)]=[ordered]@{path=$p.path;bytes=$p.bytes;sha256=$p.sha256}
 foreach($v in @($p,[ordered]@{path=$p.path;bytes=$p.bytes;sha256=$p.sha256},[ordered]@{sha256=$p.sha256;bytes=$p.bytes;path=$p.path})){$a=K-AllocationAction $l $v $script:ledgerTransaction.before;D19-R4Whole $a $e}
 foreach($case in @('extra','missing','alias','bool','double','hash')){
  $v=K-Clone $p
  switch($case){'extra'{$v.extra=0L} 'missing'{$v.Remove('bytes')} 'alias'{$v.path+='-alias'} 'bool'{$v.bytes=$true} 'double'{$v.bytes=[double]$v.bytes} 'hash'{$v.sha256='0'*64}}
  $rejected=$false;try{$null=K-AllocationAction $l $v $script:ledgerTransaction.before}catch{$rejected=$true};Need $rejected ('Exact public allocation refusal '+$case)
 }
 $bad=K-Clone $a;$bad[(K-AllocationKey)]=[ordered]@{sha256=$p.sha256;bytes=$p.bytes;path=$p.path};$message=$null;try{$null=K-Validate $bad}catch{$message=$_.Exception.Message};Same $message 'R4 metadata proof: dictionary order' 'Stored allocation remains strict ordered'
 $null=Save-Ledger $a;Write-Output 'PASS synthetic allocation'
}finally{if($script:ledgerTransaction -and -not $script:ledgerTransaction.uncertain){Stop-LedgerTransaction $script:ledgerTransaction}}""".replace('ALLOCATION',k.ALLOCATION_PATH)



def d19_python_history_raw(k,fixture,admitted_raw):
    from unittest.mock import patch
    r=k.r
    assert (fixture.root/fixture.ledger_path).read_bytes()==admitted_raw
    bridge=k.terminal_bridge(fixture.root)
    pins=(bridge['terminal'],bridge['cost'],bridge['receipt'],bridge['actor'],k.HISTORY_AUTHORITY)
    for pin in pins:
        selected=(lambda:k.recovery(fixture.recovery,fixture.root)) if pin==k.HISTORY_AUTHORITY else (lambda:k.terminal_baseline(fixture.root))
        expected=selected()
        operand=fixture.root/pin['path'];saved=operand.read_bytes()
        for warm in (False,True):
            k._HISTORY_PROOF.clear()
            k.whole_exact(selected(),expected)
            if warm:k.whole_exact(selected(),expected)
            touched=[];original_raw=r.raw_pin
            def bind_observed(pin,touched,original_raw):
                def observed(spec,root,cap=1048576,**kwargs):
                    if spec['path']==pin['path']:
                        k.whole_exact(spec,pin)
                        assert type(root) is type(fixture.root) and str(root)==str(fixture.root)
                        assert type(cap) is int and cap==k.reader_cap(pin['path']) and set(kwargs)<= {'allow_empty'} and kwargs.get('allow_empty',False) is False
                        touched.append(pin['path'])
                    return original_raw(spec,root,cap,**kwargs)
                return observed
            hook=bind_observed(pin,touched,original_raw)
            try:
                operand.write_bytes(saved+b'\n')
                with patch.object(r,'raw_pin',side_effect=hook),patch.object(k,'catalog_auth',side_effect=AssertionError('must refuse before backend')):
                    try:selected()
                    except ValueError as error:assert str(error)=='exact text size before bounded read'
                    else:raise AssertionError('targeted historical raw mutation accepted')
                assert touched==[pin['path']] and not k._HISTORY_PROOF and not k._HISTORY_BUSY
                assert (fixture.root/fixture.ledger_path).read_bytes()==admitted_raw
            finally:operand.write_bytes(saved)
            k.whole_exact(selected(),expected)
            assert (fixture.root/fixture.ledger_path).read_bytes()==admitted_raw


def d19_python_catalog_raw(k,fixture,ledger_raw):
    import os
    from unittest.mock import patch
    r=k.r
    assert (fixture.root/fixture.ledger_path).read_bytes()==ledger_raw
    old_report=k.data(k.FAIL_REPORT,fixture.root)
    names=(k.BASE+'/l9-before/tools/banc_m3_checkpoint_harness_runtime.py',old_report['nested_failure']['path'],k.BASE+'/s14-before/tools/banc_m3_checkpoint_harness_runtime.py',k.EMPTY_PATHS[1])
    catalog=k.data(k.S15_PINS['catalog'],fixture.root)
    selected=lambda:k.catalog_auth(catalog,fixture.root)
    expected=selected()
    for name,ordinal in zip(names,(19,29,116,111),strict=True):
        row=catalog['rows'][ordinal];pin=row['pin'];assert row['ordinal']==ordinal and pin['path']==name and name not in fixture.freeze['sources']
        operand=fixture.root/name;saved=operand.read_bytes()
        damaged=saved[:-1]+bytes([saved[-1]^1]) if saved else b'x';damaged.decode('utf8',errors='strict')
        for warm in (False,True):
            k._HISTORY_PROOF.clear()
            k.whole_exact(selected(),expected)
            if warm:k.whole_exact(selected(),expected)
            for missing in (False,True):
                touched=[];original_raw=r.raw_pin
                def bind_observed(name,pin,row,touched,original_raw):
                    def observed(spec,root,cap=1048576,**kwargs):
                        if spec['path']==name:
                            k.whole_exact(spec,pin)
                            assert type(root) is type(fixture.root) and str(root)==str(fixture.root)
                            assert type(cap) is int and cap==row['cap'] and set(kwargs)<= {'allow_empty'} and kwargs.get('allow_empty',False) is (row['kind']=='raw0')
                            touched.append(name)
                        return original_raw(spec,root,cap,**kwargs)
                    return observed
                hook=bind_observed(name,pin,row,touched,original_raw)
                try:
                    if missing:operand.unlink()
                    else:operand.write_bytes(damaged)
                    with patch.object(r,'raw_pin',side_effect=hook):
                        try:selected()
                        except FileNotFoundError as error:
                            assert missing and error.filename==('\\\\?\\'+str(operand) if os.name=='nt' else str(operand))
                        except ValueError as error:
                            assert not missing and str(error)==('full text repeated bytes/stamp/hash' if saved else 'exact text size before bounded read')
                        else:raise AssertionError('targeted catalog raw damage accepted')
                    assert touched==[name] and not k._HISTORY_PROOF and not k._HISTORY_BUSY
                    assert (fixture.root/fixture.ledger_path).read_bytes()==ledger_raw
                finally:operand.write_bytes(saved)
                k.whole_exact(selected(),expected)
                assert (fixture.root/fixture.ledger_path).read_bytes()==ledger_raw



def d19_history_entry(k,fixture):
    import copy
    import os

    from tools import banc_m3_conserved_resource as resource
    g=d19_golden(k,fixture);c=g['historical_cache'];rows=g['historical_raw_inputs']
    assert c['seed_only'] is True and c['producer_execution_claim'] is False
    for p in c['pin_vector']:
        rr=[r for r in rows if r['logical']==p];assert len(rr)==1
        row=rr[0];resource.d19_whole(p,row['logical'])
        resource.raw_pin(row['archive'],fixture.root,row['cap'])
    key=(os.fspath(fixture.root.absolute()),c['label'],c['epoch'],tuple((p['path'],p['bytes'],p['sha256']) for p in c['pin_vector']),tuple(tuple(row) for row in c['policy_rows']))
    value=tuple(k.data(p,fixture.root,524288 if p['bytes']>262144 else 262144) for p in c['historical_value'])
    assert c['historical_value'][0]==g['historical_paid322'] and len(value[0]['debits'])==322
    return copy.deepcopy({key:value})

def d19_r1_controls(k,fixture,resource):
 _d19_8b=resource.d19_8b;_d19_8r=resource.d19_8r;_d19_8raw=resource.d19_8raw;d19_r1_transfer=resource.d19_r1_transfer;d19_whole=resource.d19_whole;need=resource.need
 POLICY_PIN=resource.POLICY_PIN;J_PIN=resource.J_PIN
 import copy
 import math
 c=k.data(k.data(k.CURRENT_CONTRACT,fixture.root)['d19_fixed_history'],fixture.root)
 paid=k.data(c['independent_expected'],fixture.root,524288)
 raw=resource.raw_pin(c['authenticated_actual'],fixture.root,524288)
 event = k.data(fixture.recovery,fixture.root)
 w = copy.deepcopy(paid)
 if k.data(k.CURRENT_CONTRACT,fixture.root)['d19_predecessor_allocation_present']:w.pop(k.ALLOCATION_KEY)
 w[k.KEY] = copy.deepcopy(fixture.recovery)
 w['direct_science_blocking_holds'] = {}
 w['direct_science_generated_limit_s'] = 87794
 w['generated_shared_cumulative_limit_s'] = 88530
 w['original_cumulative_limit_s'] = 12870
 w['generated_shared_remaining_s'] = paid['generated_shared_remaining_s']+0.0
 w['original_remaining_s'] = paid['original_remaining_s']
 policy = k.data(POLICY_PIN,fixture.root)
 grant = k.data(J_PIN,fixture.root)
 gen = orig = 0.0
 for ordinal,debit in enumerate(paid['debits']):
  if ordinal >= 118 and debit['mode'] == 'generated': gen += float(debit['actual_outer_full_return_s'])
  if ordinal >= 165 and debit['mode'] == 'original': orig += float(debit['actual_outer_full_return_s'])
 w['generated_shared_remaining_s'] = 88530.0-float(policy['generated_shared_baseline_s'])-(float(policy['generated_stage_before_s'])+gen)
 w['original_remaining_s'] = 12870.0-float(grant['original_cumulative_before_s'])-orig
 got = d19_r1_transfer(k.__dict__,paid,raw,event,fixture.recovery,fixture.root)
 d19_whole(got,w)
 need(k.validate(fixture.ledger,fixture.root)['completed'] == 0, 'D19 no-allocation context path')
 cases = ['duplicate','stale','partial-caps','pointer','unknown-owner','unknown-HOLD','missing-provenance','forged-provenance','bool','order','binary64']
 for case in cases:
  p,e,b = copy.deepcopy(paid),copy.deepcopy(event),raw
  if case == 'duplicate': p = copy.deepcopy(got)
  elif case == 'stale': b += b'\n'
  elif case == 'partial-caps': p['direct_science_generated_limit_s'] = 6895
  elif case == 'pointer': p[k.KEY] = copy.deepcopy(fixture.recovery)
  elif case == 'unknown-owner': p['direct_science_workers'] = {'foreign':{}}
  elif case == 'unknown-HOLD': p['direct_science_blocking_holds']['foreign'] = 'unknown'
  elif case == 'missing-provenance': e.pop('adoption_closure')
  elif case == 'forged-provenance': e['adoption_closure'] = dict(e['adoption_closure'],sha256='0'*64)
  elif case == 'bool': e['delta_s']['generated_global'] = True
  elif case == 'order': e = dict(reversed(list(e.items())))
  elif case == 'binary64': p['direct_science_generated_used_s'] = math.nextafter(p['direct_science_generated_used_s'],math.inf)
  refused = False
  try: d19_r1_transfer(k.__dict__,p,b,e,fixture.recovery,fixture.root)
  except ValueError: refused = True
  need(refused,'D19 same-fixed4 transfer refusal '+case)
 result = fixture.native('D19-R1Controls | ConvertTo-Json -Depth 10 -Compress')
 need(result.returncode == 0 and result.all_owned_ended is True and result.stdout_eof is True and result.stderr_eof is True,'D19 same paid Native transfer controls closure')



def d19_fixture_projector(k,fixture,event,baseline,policy,grant,fee):
 import copy
 r=k.r
 r.need(k.ALLOCATION_KEY not in baseline,'D19 already recovered projector baseline')
 whole=lambda a,b:r.d19_whole(a,b)
 semantic_equal=lambda a,b:r.d19_whole(a,b)
 def project(ledger, root):
  """Return legal accounting state, permitting incurred negative remaining."""
  n=len(baseline['debits']);expected = copy.deepcopy(baseline)
  expected[k.KEY] = copy.deepcopy(fixture.recovery)
  expected["direct_science_blocking_holds"] = {}
  source=event['source_freeze']
  def pin(p,path):
   k.metadata_pin(p,path);v={'path':path,'bytes':p['bytes'],'sha256':p['sha256']};whole(p,v);return v
  tail = ledger["debits"][n:]
  r.need(type(tail) is list and len(tail) <= len(k.SEQUENCE), "finite seven-call tail")
  caps = dict(k.CAPS)
  failed = False
  for index, debit in enumerate(tail):
   r.exact(debit["cost_id"], k.SEQUENCE[index], "next literal debit")
   r.need(not failed, "paid failure does not admit successors")
   q, cost, _outer, closed = fee(debit["descriptor"], debit, source, root)
   r.need(closed, "UNCOMMITTED_OWNERSHIP_UNKNOWN: no fee/debit/totals without known closure")
   reserved=q['limits']['outer_s']
   p=cost['request'] if q['mode']=='generated' else cost['supervisor_request'];pin(p,p['path'])
   paid = {"cost_id":q["call_id"],"mode":q["mode"],"actual_outer_full_return_s":cost["actual_outer_full_return_s"],"actual_returned_exit_code":cost["exit_code"],"descriptor":pin(debit["descriptor"],debit["descriptor"]["path"])}
   whole(debit,paid)
   expected["debits"].append(paid)
   expected["direct_science_calls"][q["call_id"]] = {"request": cost["request"] if q["mode"] == "generated" else cost["supervisor_request"],
              "state": "charged", "mode": q["mode"], "reserved_s": reserved}
   failed = cost["exit_code"] != 0 or cost["cap_exceeded"] or not closed
   if not closed:
    k.expected_ownership(expected, q, expected["direct_science_calls"][q["call_id"]]["request"], ledger, root, charged=True)
    owner_key = ('direct_science_' if q['mode'] == 'generated' else 'startup_ray_original_') + 'workers'
    for field in ('owner_pid', 'owner_creation_ticks', 'worker_pid', 'worker_creation_ticks'):
     r.exact(_outer[field], expected[owner_key][q['call_id']][field], 'unclosed evidence retains admitted identity')
   if closed and cost['exit_code'] != 0:
    expected[('direct_science_' if q['mode'] == 'generated' else 'startup_ray_original_') + 'blocking_holds'][q['call_id']] = k.FAILURE_HOLD
   if cost["cap_exceeded"]:
    k.accounting_hold(expected, q['mode'], q['call_id'])
  if k.ALLOCATION_KEY in ledger:
   r.need(len(tail) >= 1, "allocation after actual bootstrap")
   k.metadata_pin(ledger[k.ALLOCATION_KEY], k.ALLOCATION_PATH)
   allocation = k.data(ledger[k.ALLOCATION_KEY], root)
   k.keys(allocation, ("schema", "epoch", "before_ledger", "recovery", "source_freeze", "bootstrap", "bootstrap_junit",
       "old_caps", "new_caps", "delta_s", "combined_global_s", "new_resource_grant_s", "refund_s"), "allocation event")
   r.exact(allocation["schema"], "flylab.m3.checkpoint-allocation.v1", "allocation schema")
   for field_name, field_value in (("epoch", k.EPOCH), ("recovery", ledger[k.KEY]), ("source_freeze", source), ("old_caps", k.CAPS),
      ("new_caps", k.CAPS), ("delta_s", {"generated_global": 0, "generated_local": 0, "original_global": 0}),
      ("combined_global_s", 101400), ("new_resource_grant_s", 0), ("refund_s", 0), ("bootstrap", tail[0]["descriptor"])):
    r.exact(allocation[field_name], field_value, "exact allocation " + field_name)
   before = k.data(allocation["before_ledger"], root, 524288)
   r.need(k.ALLOCATION_KEY not in before and len(before["debits"]) == n+1, "fresh closed bootstrap before allocation")
   state = project(before, root)
   r.need(state["phase"] == "charged" and not state["failed"], "successful known charged prerequisite")
   for owner_key in r.OWNER_KEYS: r.exact(before[owner_key], {}, "closed eligible owner")
   r.need(before["generated_shared_remaining_s"] >= 0 and before["original_remaining_s"] >= 0 and before["direct_science_generated_used_s"] <= state["caps"]["generated_local"], "execution remaining resource")
   k.success(before["debits"][-1], source, root, "bootstrap", allocation["bootstrap_junit"])
   expected[k.ALLOCATION_KEY] = pin(ledger[k.ALLOCATION_KEY],k.ALLOCATION_PATH)
   caps = dict(k.CAPS)
   for cap_name, cap_value in caps.items():
    expected[r.CAP_FIELDS[cap_name]] = cap_value
  else:
   r.need(len(tail) <= 1, "qualification cannot use old caps")
  if k.QUALIFICATION_KEY in ledger:
   r.need(k.ALLOCATION_KEY in ledger and len(tail) >= 5, "separate four actual costs before registration")
   k.metadata_pin(ledger[k.QUALIFICATION_KEY], k.QUALIFICATION_PATH)
   k.qualification(k.data(ledger[k.QUALIFICATION_KEY], root), ledger, source, root)
   expected[k.QUALIFICATION_KEY] = pin(ledger[k.QUALIFICATION_KEY],k.QUALIFICATION_PATH)
  r.need(len(tail) <= 5 or k.QUALIFICATION_KEY in ledger, "Original requires registered K qualification")
  expected.update(k.totals(expected, policy, grant, caps))
  active_calls = set(ledger["direct_science_calls"]) - set(expected["direct_science_calls"])
  r.need(len(active_calls) <= 1, "at most next admitted call")
  phase = "charged"
  if active_calls:
   r.need(not failed and len(tail) < len(k.SEQUENCE), "no call after failure/finite end")
   call = k.SEQUENCE[len(tail)]
   r.exact(active_calls, {call}, "no old/replay/alias admission")
   admitted = ledger["direct_science_calls"][call]
   k.keys(admitted, ("request", "state", "mode", "reserved_s"), "whole admission")
   request_path = admitted["request"]["path"]
   k.metadata_pin(admitted["request"],request_path)
   request_pin = r.text_spec(request_path,root)
   q = k.data(request_pin, root)
   k.identity(q, request_pin, source, root)
   r.exact(admitted, {"request": request_pin, "state": "admitted", "mode": q["mode"], "reserved_s": q["limits"]["outer_s"]}, "exact admitted identity")
   r.need(len(tail) != 1 or k.ALLOCATION_KEY in ledger, "no lint before allocation")
   r.need(len(tail) != 5 or k.QUALIFICATION_KEY in ledger, "no body before K qualification")
   if len(tail) == 6:
    k.valid_body(tail[5], source, k.data(q['science_request'], root), root)
   expected["direct_science_calls"][call] = {"request":request_pin,"state":"admitted","mode":q["mode"],"reserved_s":q["limits"]["outer_s"]}
   phase = k.expected_ownership(expected, q, request_pin, ledger, root)
  # A legitimate cumulative breach adds the exact responsible HOLD, never erases a debit.
  if tail:
   last = tail[-1]
   breached = ((last["mode"] == "generated" and (expected["direct_science_generated_used_s"] > caps["generated_local"]
      or expected["generated_shared_remaining_s"] < 0)) or
      (last["mode"] == "original" and (expected["original_remaining_s"] < 0
      or expected["original_stage_used_s"] > 6300 or expected["startup_ray_original_local_used_s"] > 6000)))
   if breached:
    k.accounting_hold(expected, last['mode'], last['cost_id'])
  semantic_equal(ledger, expected)
  return {"caps": caps, "completed": len(tail), "phase": phase, "failed": failed, "source": source, "epoch": k.EPOCH, "producer_state": expected}
 
 return project

# New definition-only R1 helpers. No import-time work or project execution here.

def d19_fixed4_controls(k, fixture):
 r=k.r
 d19_owner_controls=r.d19_owner_controls;d19_root_binding_controls=r.d19_root_binding_controls;d19_r1_controls=r.d19_r1_controls;d19_8b=r.d19_8b;d19_8r=r.d19_8r;d19_whole=r.d19_whole;need=r.need;d19_capsule=r.d19_capsule;d19_proof=r.d19_proof;d19_historical_exit=r.d19_historical_exit;d19_json=r.d19_json;d19_raw=r.d19_raw
 d19_owner_controls()
 d19_root_binding_controls(k,fixture)
 d19_r1_controls(k,fixture)
 import copy
 b=d19_8b(k.__dict__,fixture.root);real=d19_8r(k.__dict__,b['outer'],fixture.root)
 for name,value in [('worker_pid',1),('worker_exit_code',1),('stdout_eof',True),('creation_attempted',True),('pre_spawn_refused',0),('owner_pid',True),('unknown_owned_identities',{'foreign':1}),('first_error',None)]:
  wrong=copy.deepcopy(real);wrong[name]=value;refused=False
  try:d19_whole(wrong,b['outer_value'])
  except ValueError:refused=True
  need(refused,'Finite r8 no-create typed tampering '+name)
 wanted={'bytes':b['request']['bytes'],'path':b['request']['path'],'sha256':b['request']['sha256']};d19_whole(real['request'],wanted)
 for case in ('bytes','path','hash','bool','float','path-type','hash-type','missing','extra','order'):
  wrong=copy.deepcopy(real['request'])
  if case=='bytes':wrong['bytes']+=1
  elif case=='path':wrong['path']='alias.json'
  elif case=='hash':wrong['sha256']='0'*64
  elif case=='bool':wrong['bytes']=True
  elif case=='float':wrong['bytes']=float(wrong['bytes'])
  elif case=='path-type':wrong['path']=1
  elif case=='hash-type':wrong['sha256']=1
  elif case=='missing':wrong.pop('sha256')
  elif case=='extra':wrong['extra']=0
  else:wrong={'path':wrong['path'],'bytes':wrong['bytes'],'sha256':wrong['sha256']}
  refused=False
  try:d19_whole(wrong,wanted)
  except ValueError:refused=True
  need(refused,'r8 request association refusal '+case)
 c = d19_capsule()
 t, o, _q = d19_proof(fixture.root)
 need(d19_historical_exit(t, o, c['request'], fixture.root) is True, 'D19 actual historical positive')
 cases = [('trace-extra', 'trace', 'extra', 0), ('trace-bool-exit', 'trace', 'exit_code', True), ('trace-command', 'trace', 'exec_args', {'cmd':'relative forged command', 'workdir':c['workdir']}),
   ('trace-source', 'trace', 'source_freeze', dict(c['source'], path='alias.json')), ('trace-first-final', 'trace', 'first_final_ms', c['first_final_ms']+1),
   ('outer-equal-exit', 'outer', 'exit_code', 1), ('outer-EOF', 'outer', 'stdout_eof', False), ('outer-timeout-bool', 'outer', 'timeout', 1), ('outer-unknown', 'outer', 'unknown_owned_identities', {'1': 1})]
 for label, where, key, value in cases:
  a, b = copy.deepcopy(t), copy.deepcopy(o)
  (a if where == 'trace' else b)[key] = value
  refused = False
  try: d19_historical_exit(a, b, c['request'], fixture.root)
  except ValueError: refused = True
  need(refused, 'D19 paid fixed4 historical negative ' + label)
 for raw in (b'{"x":1,"x":2}', b'{"x":1e999}', b'{"x":NaN}'):
  refused = False
  try: d19_json(raw)
  except ValueError: refused = True
  need(refused, 'D19 strict negative JSON')
 for pin in (dict(c['actor'], bytes=357047), dict(c['source'], path='alias.json'), dict(c['source'], bytes=True)):
  refused = False
  try: d19_raw(pin, fixture.root)
  except (ValueError, FileNotFoundError): refused = True
  need(refused, 'D19 finite route negative')
 result = fixture.native("D19-Controls | ConvertTo-Json -Depth 10 -Compress")
 need(result.returncode == 0 and result.all_owned_ended is True and result.stdout_eof is True and result.stderr_eof is True, 'D19 actual paid Native controls known closure')



def d19_atomic_legacy_positive(tmp_path):
 import json
 import os

 from tests import banc_m3_checkpoint_fixture as old
 fixture=old.case(tmp_path,'generated-legacy-writer-positive')
 assert k.KEY not in fixture.ledger
 path=fixture.root/fixture.ledger_path;before=path.read_bytes()
 lock=r.SharedLock(fixture.root/r.SHARED_LOCK,{'schema':'flylab.m3.direct-science.lock-owner.v1','owner_pid':os.getpid()})
 try:
  value=copy.deepcopy(fixture.ledger)
  result=r.atomic_ledger_publish(path,before,value,lock,validate=lambda current:old.k.validate(current,fixture.root))
  assert result==json.dumps(value,ensure_ascii=True,allow_nan=False,sort_keys=True).encode('utf8')
  assert path.read_bytes()==result
 finally:lock.release(True)




def d19_current_fee_refusal(k,error,variant,candidate,contract,changed_source,source):
    import copy
    import traceback
    expected=copy.deepcopy(contract);wanted=copy.deepcopy(source)
    if variant=='reviewed-source-mismatch':
        assert wanted['reviewed_source']['sha256']!='0'*64
        wanted['reviewed_source']['sha256']='0'*64
        message='D19 decoded current Source metadata differs from raw';frame='d19_event'
    else:
        rows=expected['actual320']['archives'];rr=[x for x in rows if x['logical_current_before']['path']==k.REVIEW_PATH]
        assert len(rr)==1 and variant in ('missing-review','duplicate-review')
        expected['actual320']['archives']=[x for x in rows if x['logical_current_before']['path']!=k.REVIEW_PATH]+(rr*2 if variant=='duplicate-review' else [])
        message='D19 decoded current contract metadata differs from raw';frame='d19_current_contract'
    k.whole_exact(candidate,expected);k.whole_exact(changed_source,wanted)
    frames=traceback.extract_tb(error.__traceback__)
    actual=[x.name for x in frames if x.filename.replace('\\','/').endswith('/tools/banc_m3_conserved_resource.py')]
    assert str(error)==message and frame in actual,str(error)


def d19_fee_association_controls(tmp_path):
    from unittest.mock import patch
    actual_data=k.data
    fixture=case(tmp_path,'fee-association-cold')
    contract=actual_data(k.CURRENT_CONTRACT,fixture.root)
    predecessor_source=actual_data(k.CONTINUATION_BEFORE_SOURCE,fixture.root)
    old=k.predecessor(contract['actual320'],fixture.root)
    older=k.predecessor(old['actual320'],fixture.root)
    assert type(old) is dict and type(older) is dict
    for variant in ('missing-review','duplicate-review','reviewed-source-mismatch'):
        negative_fee=case(tmp_path,'fit-fee-'+variant)
        candidate=copy.deepcopy(contract);before_source=copy.deepcopy(predecessor_source)
        source=actual_data(negative_fee.source,negative_fee.root);changed_source=copy.deepcopy(source)
        if variant=='reviewed-source-mismatch':
            before_source['reviewed_source']['sha256']='0'*64
            changed_source['reviewed_source']['sha256']='0'*64
        else:
            rr=[x for x in candidate['actual320']['archives'] if x['logical_current_before']['path']==k.REVIEW_PATH]
            assert len(rr)==1
            candidate['actual320']['archives']=[x for x in candidate['actual320']['archives'] if x['logical_current_before']['path']!=k.REVIEW_PATH]+(rr*2 if variant=='duplicate-review' else [])
        negative_fee.admit();negative_fee.evidence(1.0,0)
        damaged=negative_fee.root/k.REPLACEMENTS[0];damage=damaged.read_bytes()+b'\n';damaged.write_bytes(damage)
        source_raw=(negative_fee.root/k.SOURCE_PATH).read_bytes()
        raw_before=(negative_fee.root/negative_fee.ledger_path).read_bytes()
        def current_decoded(p,root,cap=262144,candidate=candidate,negative_fee=negative_fee,changed_source=changed_source):
            if p==k.CURRENT_CONTRACT:return copy.deepcopy(candidate)
            if p==negative_fee.source:return copy.deepcopy(changed_source)
            return actual_data(p,root,cap)
        with patch.object(k,'data',side_effect=current_decoded):
            try:negative_fee.python_charge()
            except ValueError as error:d19_current_fee_refusal(k,error,variant,candidate,contract,changed_source,source)
            else:raise AssertionError('current fee association guard did not refuse')
        def historical_decoded(p,root,cap=262144,candidate=candidate,before_source=before_source):
            if p==k.CURRENT_CONTRACT:return copy.deepcopy(candidate)
            if p==k.CONTINUATION_BEFORE_SOURCE:return copy.deepcopy(before_source)
            return actual_data(p,root,cap)
        with patch.object(k,'data',side_effect=historical_decoded):
            try:k.predecessor(candidate['actual320'],negative_fee.root)
            except ValueError as error:d19_fee_refusal(k,error,variant,candidate,before_source,contract,predecessor_source)
            else:raise AssertionError('historical predecessor association guard did not refuse')
        assert (negative_fee.root/negative_fee.ledger_path).read_bytes()==raw_before
        assert (negative_fee.root/k.SOURCE_PATH).read_bytes()==source_raw
        assert damaged.read_bytes()==damage


D19_OPAQUE_TAIL = r'''$empty}catch{$e=$_.Exception.GetBaseException().GetType().FullName};Same $e 'System.Text.DecoderFallbackException' 'Default strict UTF8'}
try{if($b.Length){$changed=[byte[]]$b.Clone();$changed[-1]=$changed[-1] -bxor 1;[IO.File]::WriteAllBytes($a,$changed);& $oc {K-Raw $p $r.cap $empty $true} 'Whole K SHA'};[IO.File]::WriteAllBytes($a,[byte[]]($b+@([byte]120)));& $oc {K-Raw $p $r.cap $empty $true} 'Whole K evidence bytes';[IO.File]::Delete($a);& $oc {K-Raw $p $r.cap $empty $true} 'Bounded actual regular Source'}finally{[IO.File]::WriteAllBytes($a,$b)};& $positive}
foreach($pin in @($script:KCurrentContract,(K-Read (K-BoundedPin (K-Paths).source)).sources['tools/banc_m3_checkpoint_harness_runtime.py'])){& $oc {K-Raw $pin 262144 $false $true} 'Opaque stream'}
}}finally{Set-Item Function:K-Raw $origRaw;Set-Item Function:K-SourceRaw $origSource}
'''

def d19_accepted_history(k,fixture):
 return "$null=K-SourceFence (K-BoundedPin (K-Paths).source);\n"+d19_native_history(k,fixture)

