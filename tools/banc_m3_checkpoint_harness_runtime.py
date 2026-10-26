"""Finite L accounting checkpoint. Metadata only; current Source is a cold fence.

Accounting reconstructs every permitted field from the literal closed300 anchor.
It does not invoke the historical epoch proof graph or import scientific workers.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import stat
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from tools import banc_m3_conserved_resource as r

_HISTORY_BASE = 'verification/banc-natural-gait-m3-20260919'
_LOGGER = logging.getLogger(__name__)

BASE = r.BASE
KEY = "direct_science_checkpoint_harness_s28_recovery"
ALLOCATION_KEY = "direct_science_checkpoint_harness_s28_allocation"
QUALIFICATION_KEY = "direct_science_checkpoint_harness_s28_qualification"
EPOCH = "contact-checkpoint-harness-20261009-s28"
COLD302 = {"path":BASE+"/parent-l9-cold-settled302-HOLD-20261008.json","bytes":475996,"sha256":"0bbe985caef363635be5568fcf4e0cdc9619b648b91d707b08f80addda151baa"}
FAILED303 = {"path":BASE+"/parent-l9r2-bootstrap-settled303-HOLD-20261008.json","bytes":477147,"sha256":"dcd13d8bd47e1f878bc0b83acdad6d7fa9b67023102ae16ef629f84a067b1e7c"}
COLD304 = {"path":BASE+"/parent-l10-cold-settled304-HOLD-20261008.json","bytes":478269,"sha256":"8ee279a65965b578841cdd67e028f5e98017a5ad57fa07506c89a765aa8ff461"}
ANCHOR = {"path":BASE+"/parent-l12-cold-settled306-HOLD-20261008.json","bytes":480516,"sha256":"7748aaa0695f12d25b46b613c6d9b8226a1680410b5645d80378aa00393e66a3"}
CLOSED305 = {"path":BASE+"/parent-l11-bootstrap-settled305-HOLD-20261008.json","bytes":479395,"sha256":"039a2cdf42d4dc9cfbe02daafd6487c8199e8da4bd272ded430626c69108c8fa"}
SOURCE11 = {"path":BASE+"/l11-s1.json","bytes":262136,"sha256":"fa8ce9f6d795d1bb5b53ecb131ce302b147c3b2d65742a591c1ea3e251e8d877"}
SOURCE12 = {"path":BASE+"/l12-s1.json","bytes":262136,"sha256":"f89aee03161b4cb6e1916f27d06eaf5a80d3bdbbe60c810fbfe076c5d8243519"}
COLD12 = {"path":BASE+"/l12-cold-actual.json","bytes":2161,"sha256":"1d5e85e7e541807aaba30df8b226ffd0f035dd1b04c10525c0273530771a621d"}
TIMEOUT_REPORT = {"path":BASE+"/l11-actual.json","bytes":1722,"sha256":"042547cf24bee0d9b644eabae32562a0587f8e73297652acf34598d7426c6091"}
LATEST_FAILURE = {"path":BASE+"/l9r2-actual.json","bytes":2263,"sha256":"3a36455df74e5fa367adae1d5775ae6f0115f386e4ae9fb076a2f5aad492223e"}
L8_ANCHOR = {"path":BASE+"/parent-l8-bootstrap-settled301-HOLD-20261008.json","bytes":474869,"sha256":"70872adcfb9654c2219f17b80bcd2acfdfadc5b2195cf40fc3dfcd1fe25f46fc"}
LATEST_COLD = {"sha256":"b71204781cbacd2f3794b8c5440ccdf9ca8f7b5c4d793100bf5f71f7ce87c571","bytes":2055,"path":BASE+"/l10-cold-actual.json"}
ROOT_COLD = {"path":BASE+"/l9-cold-actual.json","bytes":2065,"sha256":"cf10476b8bb4a9546ead22e330466b5113e269e985bd95dc11f6ebde72796bf4"}

BEFORE_MAP = {"path": BASE + "/parent-k-before-20261008/source-before.json", "bytes": 15761,
   "sha256": "26666fb0be28e4b0efc68eda07c6764e63e8e339fe24c903fd81c0a9ec6b5a58"}
J_FREEZE = {"path": BASE + "/parent-contact-learning-source-freeze-20261008-j.json", "bytes": 255412,
   "sha256": "d2ea5057326a9a916dcfe59c3b9cbbdb5b7b61fc852bf23b5d314fb1269f1baa"}
J_BODY = {"bytes":19735,"sha256":"a9628aaf192f3b1085ff7d403f7cb17c9542cab4c32df7bc97c83b60f3a07d77","path":BASE+"/l13r2-before/verification/banc-natural-gait-m3-20260919/b5.json"}
OLD300 = {"path":BASE+"/parent-l7-cold-settled300-HOLD-20261008.json","bytes":473757,"sha256":"19c52087c920c29d2f51c9dc9b7a03ec84127877b6ff70bb8e70c26d72884218"}
OLD_RECOVERY = {"path":BASE+"/l8-recovery1.json","bytes":2410,"sha256":"3a2178d3d9a4fba0d0b31d4cb2fc138db3cfc2e9a09785d36d3422fab5485f3b"}
OLD_SOURCE = {"path":BASE+"/l8-s1.json","bytes":262134,"sha256":"91d3a533e3d73f1aaae7af2bce236724452e7f90cd5e12720f42072948825d12"}
FAIL_REPORT = {"path":BASE+"/l8-actual.json","bytes":2209,"sha256":"6ee7571754780fa913f7462fbf5725c4ad17eeab1ab17997903f3e1140933dc7"}
COLD_PATHS = {"witness":BASE+"/l7-cold-witness.json","source":BASE+"/l7-s2.json","legacy":BASE+"/parent-l6-math-settled299-HOLD-20261008.json","report":BASE+"/l7-cold-actual.json","request":BASE+"/parent-direct-science-20261008-contact-l7-bootstrap-request.json","trace":BASE+"/parent-direct-science-20261008-contact-l7-bootstrap-genuine-return.json","recovery":BASE+"/l7-recovery1.json","cost":BASE+"/parent-direct-science-20261008-contact-l7-bootstrap-full-return.json","before":BASE+"/parent-l7-bootstrap-admitted299-before-cold-settlement-20261008.json"}
S15_PINS = {'original': {'path': _HISTORY_BASE+'/s27-bootstrap-settled318-HOLD-20261009.json', 'sha256': '70175b4bc56c19412b64500343ebc72bc538a7b3776dd60733d43efeeb59476e', 'bytes': 495417}, 'anchor': {'bytes': 495417, 'sha256': '70175b4bc56c19412b64500343ebc72bc538a7b3776dd60733d43efeeb59476e', 'path': _HISTORY_BASE+'/s28-anchor318-before-recovery.json'}, 'catalog': {'path': _HISTORY_BASE+'/h28.json', 'bytes': 261912, 'sha256': 'e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e'}, 'repair': {'path': _HISTORY_BASE+'/e28.json', 'bytes': 2443, 'sha256': 'e5e284cbaa1f67f1dd6e14bdf3fe74c5f902d751c387f6850700323c23609c93'}, 'before': {'path': _HISTORY_BASE+'/s27-s1.json', 'sha256': '35b6cf63018f61f3fde34d47045d16f5e3c97dc53bac64b1fe7814aed0aa61c6', 'bytes': 262139}, 'refinement_anchor': {'path': _HISTORY_BASE+'/s28-f14-root/a318.json', 'bytes': 495559, 'sha256': 'fadfb1604441f5c2cc481e15b07eeaf10a7a7529cedc8969bc496196ea221274'}, 'previous_recovery': {'path': _HISTORY_BASE+'/s28-recovery10.json', 'bytes': 1903, 'sha256': '27f8b420a0f7cdb3447aac416cd2ac71cf499ee8b5ea719ff05c968fd6f1184e'}, 'previous_source': {'path': _HISTORY_BASE+'/s28-f14-root/s1-before.json', 'bytes': 262140, 'sha256': 'f1b802af060e385f03955b5cbba5830cc9ee42ccf757b9800016a45afd7cabfd'}, 'previous_contract': {'path': _HISTORY_BASE+'/s28-f14-root/c5-before.json', 'bytes': 97623, 'sha256': '9690bd2ec7a65d33432071d09d9050c6d949c92d6a20852fc92072737f7dcb41'}, 'previous_catalog': {'path': _HISTORY_BASE+'/s28-f14-root/h28-before.json', 'bytes': 261912, 'sha256': 'e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e'}, 'technical_refusal': {'path': _HISTORY_BASE+'/s28-f13-root/actual-fast-warm-gate.json', 'bytes': 2818, 'sha256': '042051f83aaa9c347c72d943e76d0958cf22afeccb5fd3c929bf0241cd5f5a1f'}}
EMPTY_PATHS = tuple(BASE+"/parent-direct-science-20261008-contact-"+x+"-bootstrap/stderr.bin" for x in ("l11","l13r2")) + (BASE+"/parent-direct-science-20261009-contact-s14-bootstrap/stderr.bin",BASE+"/parent-direct-science-20261009-contact-s15-bootstrap/stderr.bin",BASE+"/parent-direct-science-20261009-contact-s16-bootstrap/stderr.bin",BASE+"/parent-direct-science-20261009-contact-s19-bootstrap/stderr.bin",) + ('verification/banc-natural-gait-m3-20260919/parent-direct-science-20261009-contact-s23-bootstrap/stderr.bin',) + ('verification/banc-natural-gait-m3-20260919/parent-direct-science-20261009-contact-s24-bootstrap/stderr.bin',) + ('verification/banc-natural-gait-m3-20260919/parent-direct-science-20261009-contact-s25-bootstrap/stderr.bin',) + ('verification/banc-natural-gait-m3-20260919/parent-direct-science-20261009-contact-s27-bootstrap/stderr.bin',)
S15_HOLD = "direct-science-20261009-contact-s27-bootstrap"
REPAIR_KEY = "direct_science_checkpoint_extension_s23_2000"
_HISTORY_PROOF = {}
_HISTORY_BUSY = set()
L_FREEZE = {"path":BASE+"/l13-s1.json","bytes":262136,"sha256":"51afdfc45124190a03354cbc1892aada5d105959bd1f74c812f802ef147a6998"}

L_ROSTER = {"bytes":47801,"sha256":"fcf5b40f439941b0b16cd7f87cdb16cd5be80675d97b4cdcd18283f45a55b764","path":BASE+"/l13r2-before/verification/banc-natural-gait-m3-20260919/t5.json"}
HISTORY_CAPS = {"generated_global": 3400, "generated_local": 2664, "original_global": 14500}
OLD_CAPS = {"generated_global": 6530, "generated_local": 5794, "original_global": 13370}
CAPS = {"generated_global":88530,"generated_local":87794,"original_global":12870}
RECOVERY_PATH = BASE + "/s28-d19bz-recovery.json"
ALLOCATION_PATH = BASE + "/s28-allocation-r64.json"
SOURCE_PATH = BASE + "/s28-s1.json"
REVIEW_PATH = BASE + "/s28-r1.json"
CONTRACT_PATH = BASE + "/c5.json"
BODY_PATH = BASE + "/b5.json"
ROSTER_PATH = BASE + "/t5.json"
QUALIFICATION_PATH = BASE + "/s28-qualified-r64.json"
BOOTSTRAP_CALL = "direct-science-20261009-contact-s28-bootstrap-r64"
QUAL_CALLS = tuple("direct-science-20261009-contact-s28-r64-" + x for x in ("lint", "math", "gate-a", "gate-b"))
CALLS = ("contact-learning-20261009-s28-r64-body", "contact-learning-20261009-s28-r64-body-audit")
SEQUENCE = (BOOTSTRAP_CALL, *QUAL_CALLS, *CALLS)
CAP_TRIPLES = ((4000, 3980, 20), (40, 35, 5), (120, 90, 30), (170, 155, 15), (5400, 5380, 20), (300, 285, 15), (120, 100, 20))
BOOTSTRAP_NAMES = (
 "test_gate_l_checkpoint_production_python_native_parity",
 "test_gate_l_checkpoint_exact_recovery_allocation_and_negatives",
 "test_gate_l_checkpoint_actual_atomic_foreign_lock_and_stale_raw",
 "test_gate_l_checkpoint_fee_source_drift_overrun_unknown_duplicate",
)
TEST_MODULE = "tests/test_c_banc_checkpoint_harness_runtime.py"
MATH_NAMES = ("test_math_l_checkpoint_ordered_accounting_negative_remaining",
   "test_math_l_checkpoint_exact_prefix_types_and_legal_states")
REPLACEMENTS = ('tools/banc_m3_checkpoint_harness_runtime.py','tools/banc_m3_checkpoint_harness_runtime_native.ps1','tests/banc_m3_checkpoint_fixture.py','tests/banc_m3_checkpoint_harness_fixture.py','tests/test_c_banc_checkpoint_harness_runtime.py','tests/test_c_banc_contact_runtime_journal.py','tools/banc_m3_conserved_resource.py',BASE + '/parent-direct-science-account.ps1',BASE + '/b5.json',BASE + '/c5.json',BASE + '/t5.json',)
ADDITIONS = ()
LINT_MODULES = ('tools/banc_m3_contact_learning.py','tools/verify_c_banc_m3_contact_learning.py','tools/analyze_c_banc_m3_contact_learning.py','tools/banc_m3_contact_learning_admission.py','tools/banc_m3_startup_ray_supervisor.py','tools/verify_c_banc_m3_startup_ray.py','tools/banc_m3_contact_runtime_journal.py','tools/banc_m3_conserved_resource.py','tests/test_c_banc_contact_runtime_journal.py','tests/test_c_banc_contact_learning.py','tests/test_c_banc_contact_metadata.py','tests/test_c_banc_startup_ray_execution.py','tests/test_c_banc_contact_body_retry.py','tools/banc_m3_checkpoint_runtime.py','tests/test_c_banc_checkpoint_runtime.py','tests/banc_m3_checkpoint_fixture.py','tools/banc_m3_checkpoint_harness_runtime.py','tests/test_c_banc_checkpoint_harness_runtime.py','tests/banc_m3_checkpoint_harness_fixture.py')
HOLD = "Actual closure or resource cap unresolved"
FAILURE_HOLD = 'L stage unsuccessful; successor blocked'


def keys(value, expected, label):
 r.need(type(value) is dict and set(value) == set(expected), label + " exact ordinal keys")


def integer(value, label, positive=False):
 r.need(type(value) is int and (not positive or value > 0), label + " integer, never bool/float")
 return value


def number(value, label, nonnegative=True):
 r.need(type(value) in (int, float) and math.isfinite(value) and (not nonnegative or value >= 0), label)
 return float(value)


LEDGER_METADATA_PATHS = ((BASE+'/s28-f14-root/a318.json'),) + ((BASE+'/s28-f13-root/a318.json'),) + ((BASE+'/s28-f12-root/a318.json'),) + ((BASE+'/s28-f11-root/a318.json'),) + ((BASE+'/s28-f10-root/a318.json'),) + ((BASE+'/s28-f9-root/a318.json'),) + ((BASE+'/s28-f8-root/a318.json'),) + ((BASE+'/s28-f7-root/a318.json'),) + ((BASE+'/s28-f6-root/a318.json'),) + ((BASE+'/s28-f5-root/a318.json'),) + ((BASE+'/s28-f4-root/a318.json'),) + ((BASE+'/s28-anchor318-before-recovery.json'), (BASE+'/s28-recovered318-before-bootstrap.json'), (BASE+'/parent-s28-bootstrap-settled319-before-allocation-20261009.json'), (BASE+'/parent-s28-qualification-settled323-before-body-20261009.json'), (BASE+'/parent-s28-body-settled324-before-audit-20261009.json'), (BASE+'/s28-bootstrap-settled319-HOLD-20261009.json')) + ((BASE+'/s27-anchor317-before-recovery.json'), (BASE+'/s27-recovered317-before-bootstrap.json'), (BASE+'/parent-s27-bootstrap-settled318-before-allocation-20261009.json'), (BASE+'/parent-s27-qualification-settled322-before-body-20261009.json'), (BASE+'/parent-s27-body-settled323-before-audit-20261009.json'), (BASE+'/s27-bootstrap-settled318-HOLD-20261009.json')) + ((BASE+'/s25-bootstrap-settled317-HOLD-20261009.json'), (BASE+'/s26-anchor317-before-recovery.json'), (BASE+'/s26-recovered317-before-bootstrap.json'), (BASE+'/parent-s26-bootstrap-settled318-before-allocation-20261009.json'), (BASE+'/parent-s26-qualification-settled322-before-body-20261009.json'), (BASE+'/parent-s26-body-settled323-before-audit-20261009.json'), (BASE+'/s26-bootstrap-settled318-HOLD-20261009.json')) + ((BASE+'/s24-bootstrap-settled316-HOLD-20261009.json'), (BASE+'/s25-anchor316-before-recovery.json'), (BASE+'/s25-recovered316-before-bootstrap.json'), (BASE+'/parent-s25-bootstrap-settled317-before-allocation-20261009.json'), (BASE+'/parent-s25-qualification-settled321-before-body-20261009.json'), (BASE+'/parent-s25-body-settled322-before-audit-20261009.json'), (BASE+'/s25-bootstrap-settled317-HOLD-20261009.json')) + ((BASE+'/s23-bootstrap-settled315-HOLD-20261009.json'), (BASE+'/s24-anchor315-before-recovery.json'), (BASE+'/s24-recovered315-before-bootstrap.json'), (BASE+'/parent-s24-bootstrap-settled316-before-allocation-20261009.json'), (BASE+'/parent-s24-qualification-settled320-before-body-20261009.json'), (BASE+'/parent-s24-body-settled321-before-audit-20261009.json'), (BASE+'/s24-bootstrap-settled316-HOLD-20261009.json')) + (ANCHOR['path'],CLOSED305['path'],COLD304['path'],FAILED303['path'],COLD302['path'],L8_ANCHOR['path'],OLD300['path'],BASE+'/s17-bootstrap-settled311-HOLD-20261009.json',BASE+'/s18-anchor311-after-repair20.json',*(BASE+"/"+x.replace('^','parent-').replace('@','-20261008').replace('~','-20261009').replace('%','bootstrap-settled').replace('?','qualification-settled').replace('!','body-settled')+".json" for x in ('^l12-bootstrap-admitted305-before-cold-settlement@','^l10-bootstrap-admitted303-before-cold-settlement@','^l9-bootstrap-admitted301-before-cold-settlement@','^e358-execution-ledger','^l13r2-%307-before-allocation@','^l13r2-?311-before-body@','^l13r2-!312-before-audit@','^l7-bootstrap-admitted299-before-cold-settlement@','^l6-math-settled299-HOLD@','^l3-%292-before-allocation@','generated-l13r2-allocation-before','generated-l13r2-original-before','^s15-%309-before-allocation~','^s15-?313-before-body~','^s15-!314-before-audit~','^l13r2-%307-HOLD@','s14-anchor307-after-repair400','^s14-%308-before-allocation~','^s14-?312-before-body~','^s14-!313-before-audit~','s14-%308-HOLD~','s15-%309-HOLD~','^s16-%310-before-allocation~','^s16-?314-before-body~','^s16-!315-before-audit~','s16-anchor309-after-repair400-v2','s16-recovered309-before-bootstrap','^s17-%311-before-allocation~','^s17-?315-before-body~','^s17-!316-before-audit~','s16-%310-HOLD~','s17-recovered310-before-bootstrap','^s18-%312-before-allocation~','^s18-?316-before-body~','^s18-!317-before-audit~','s18-%312-HOLD~','s19-anchor312-after-repair200','s18-recovered311-before-bootstrap','^s19-%313-before-allocation~','^s19-?317-before-body~','^s19-!318-before-audit~','s19-%313-HOLD~','s22-anchor313-after-repair110','s19-recovered312-before-bootstrap','^s22-%314-before-allocation~','^s22-?318-before-body~','^s22-!319-before-audit~','s22-%314-HOLD~','s23-anchor314-after-grant2000','^s23-%315-before-allocation~','^s23-?319-before-body~','^s23-!320-before-audit~')))

CURRENT_CONTRACT = {'path': 'verification/banc-natural-gait-m3-20260919/c5.json', 'bytes': 130832, 'sha256': '9398fcc3a61b0e8019b2599c40ca8dd7e0ebaa4e60302b71a74c2dd7a1a32d78'}
HISTORY_AUTHORITY = {'path': 'verification/banc-natural-gait-m3-20260919/s28-f15b-root/before/verification/banc-natural-gait-m3-20260919/c5.json', 'bytes': 124922, 'sha256': '9d65539915dbe98563ae5599df9d91c422541411f51608bc5ebe14fab8655ed8'}
FROZEN_HISTORY_PINS = json.loads('[{"path":"@/s28-recovery10.json","bytes":1903,"sha256":"27f8b420a0f7cdb3447aac416cd2ac71cf499ee8b5ea719ff05c968fd6f1184e"},{"path":"@/s28-f13-root/s1-before.json","bytes":262140,"sha256":"39b4910b21c51042a5d68f3da51061ce2d57ead51ec5ff4cdcf82554a6493ef0"},{"path":"@/s28-f12-root/a318.json","bytes":495558,"sha256":"1221bf613896fd8f929b0dbfd93c1e4bc8e922f4c873e3869d5a7780d53c6d54"},{"path":"@/s28-f14-root/a318.json","bytes":495559,"sha256":"fadfb1604441f5c2cc481e15b07eeaf10a7a7529cedc8969bc496196ea221274"},{"path":"@/s28-f12-root/actual-tool-return.json","bytes":1846,"sha256":"71341a5233ecc031b45876ab24d1f7e4305ed79c738aa19f3efa71dc6c250162"},{"path":"@/s28-f13-root/c5-before.json","bytes":74674,"sha256":"3a87bc0f34b8a36ddd5153940f2b17fccbdd990b32459aea28ae92e51be58b0a"},{"path":"@/h28.json","bytes":261912,"sha256":"e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e"},{"path":"@/s28-f12-root/actual-layout-compatibility-diagnostic.json","bytes":1331,"sha256":"efc4694c9651f842832549809e1613b9900ab71004da098bb82dc24c29ee8a6d"},{"path":"@/s28-catalog-leaves-300021e7370a4dc8b163ef22f0245f31.jsonl","bytes":6393,"sha256":"c4bb7e972e5ed5950e554983344f5d34849008f9a2a9a984a874d1ff8795926e"},{"path":"@/s28-f12-root/fast-counter-expectation.json","bytes":2331,"sha256":"99e971e5a752697e8ab86e9fbf461acd4052d1d27da3a5437f05db14fe050b5d"},{"path":"@/s28-recovery9.json","bytes":1902,"sha256":"a4cf49fb579aeb3114fd800f092080b5e6a54b8f71ae835121e8cd34a1d4a28d"},{"path":"@/s28-f12-root/s1-before.json","bytes":262140,"sha256":"1cf30bee866a147de013d0023a2ea02d8734005178d145d7f06952f058ef4aba"},{"path":"@/s27-bootstrap-settled318-HOLD-20261009.json","sha256":"70175b4bc56c19412b64500343ebc72bc538a7b3776dd60733d43efeeb59476e","bytes":495417},{"path":"@/e28.json","bytes":2443,"sha256":"e5e284cbaa1f67f1dd6e14bdf3fe74c5f902d751c387f6850700323c23609c93"},{"path":"@/s28-f12-root/h28-before.json","bytes":261912,"sha256":"e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e"},{"path":"@/s27-actual.json","bytes":3365,"sha256":"fcf5b1fa42a8348fa52f85e0259cc3b3554be5d96206fd5e77f4fcc54e01eb94"},{"path":"@/s28-f11-root/actual-fast-warm-gate.json","bytes":4977,"sha256":"9bbf1418212eedfb12b4aaa57cac0d613c9b900b0f9e2a0b4a4d7dfa1f63c0ce"},{"path":"@/s28-recovery11.json","bytes":1882,"sha256":"0098806a84d0c18cf606482710d15e73dbe21defae360badba88a6c909db4c26"},{"path":"@/s28-f14-root/s1-before.json","bytes":262140,"sha256":"f1b802af060e385f03955b5cbba5830cc9ee42ccf757b9800016a45afd7cabfd"},{"path":"@/s28-f13-root/actual-tool-return.json","bytes":2798,"sha256":"e76c924ab89dd27752d2f8a1afe68bd6839f363471886c073b3b14fce0d5d2c0"},{"path":"@/s28-catalog-leaves-30944ee8fe084e2aac23de9d47e4de80.jsonl","bytes":496,"sha256":"b3ebd169ebfc3f700ec42a0c3b7b9292f35b7561cb6f7d100c71bf3835669606"},{"path":"@/s28-f14-root/c5-before.json","bytes":97623,"sha256":"9690bd2ec7a65d33432071d09d9050c6d949c92d6a20852fc92072737f7dcb41"},{"path":"@/s28-f13-root/actual-compiler-type-refusal.json","bytes":1968,"sha256":"448f06a882edd1757895f90dfd4bd1c7bfad38f414fcdda3909a0eacfa34e407"},{"path":"@/s28-f13-root/a318.json","bytes":495559,"sha256":"fadfb1604441f5c2cc481e15b07eeaf10a7a7529cedc8969bc496196ea221274"},{"path":"@/s28-f13-root/h28-before.json","bytes":261912,"sha256":"e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e"},{"path":"@/s28-f12-root/actual-fast-warm-gate.json","bytes":5074,"sha256":"9c9348756e9dacf4ee53a320e50acc22d25c8fe9028419dccbc35819356d7b7e"},{"bytes":495417,"sha256":"70175b4bc56c19412b64500343ebc72bc538a7b3776dd60733d43efeeb59476e","path":"@/s28-anchor318-before-recovery.json"},{"path":"@/s27-s1.json","sha256":"35b6cf63018f61f3fde34d47045d16f5e3c97dc53bac64b1fe7814aed0aa61c6","bytes":262139},{"path":"@/s28-f14-root/h28-before.json","bytes":261912,"sha256":"e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e"},{"path":"@/s28-f13-root/actual-fast-warm-gate.json","bytes":2818,"sha256":"042051f83aaa9c347c72d943e76d0958cf22afeccb5fd3c929bf0241cd5f5a1f"},{"path":"@/s28-f15b-root/before/@/c5.json","bytes":124922,"sha256":"9d65539915dbe98563ae5599df9d91c422541411f51608bc5ebe14fab8655ed8"}]'.replace("@",BASE))
HISTORY_BEFORE_SOURCE = {'path': 'verification/banc-natural-gait-m3-20260919/s28-f15b-root/before/verification/banc-natural-gait-m3-20260919/s28-s1.json', 'bytes': 262142, 'sha256': 'b8f00cb6fe06b0fa611b10b62d60f39d5a8e130887767f9e6c791b0973e35bbc'}
CONTINUATION_BEFORE_SOURCE = {'path': 'verification/banc-natural-gait-m3-20260919/s28-f18-root/residual03/inputs/verification/banc-natural-gait-m3-20260919/s28-s1.json', 'bytes': 262142, 'sha256': '7acb957d85b4987472452f576c2f851ad575b13d6c571b47d9d37f0f55c1c0e3'}
LEDGER_METADATA_PATHS += ((BASE+'/s28-f15b-root/terminal319-before.json'), (BASE+'/s28-f15-root/admitted318-before.json'), (BASE+'/s28-f15-root/settlement-pre-backup.json'), (BASE+'/s28-f15-root/settlement-replace-backup.json'), (BASE+'/s28-f12-root/a318.json'), (BASE+'/s28-f14-root/a318.json'), (BASE+'/s27-bootstrap-settled318-HOLD-20261009.json'), (BASE+'/s28-f13-root/a318.json'), (BASE+'/s28-anchor318-before-recovery.json'))

_OLD_LEDGER_PATHS = LEDGER_METADATA_PATHS
_PAID_LEDGER_PATHS = ((BASE+'/s28-f16-root/before/'+BASE+'/parent-e358-execution-ledger.json'), (BASE+'/s28-f15c-root/before-full4-canonical319.json'), (BASE+'/s28-f15c-root/full4-admitted319.json'), (BASE+'/s28-f18-root/b320.json'), (BASE+'/s28-f18-root/a320.json'), (BASE+'/s28-f18-root/t321.json'))
_PAID_LEDGER_PATHS += tuple(BASE+'/s28-f18-root/residual02/r3-'+p+'.json' for p in ('before-admission-canonical321','admitted321','terminal322'))
LEDGER_METADATA_PATHS += _PAID_LEDGER_PATHS
LEDGER_METADATA_PATHS += (BASE+'/s28-f19-root/transition-cut/independent-expected-paid328.json',BASE+'/s28-f19-root/residual09/v2/paid328.json')
LEDGER_METADATA_PATHS += tuple(BASE+'/s28-f19-root/'+p for n in range(329,362) for p in (f'transition{n-318}/cut/independent-expected-paid{n}.json',f'transition{n-319}/paid{n}.json'))
LEDGER_METADATA_PATHS += (BASE+'/s28-f19-root/transition36/paid353.json',)
LEDGER_METADATA_PATHS += tuple(BASE+'/s28-f19-root/transition'+p for p in ('44/allocation-before.json', '44/original-body-before.json', '44/original-audit-before.json', '44/cut/independent-expected-paid363.json', '43/paid363-pre-spawn.json', '43/paid362.json', '45/allocation-before.json', '45/original-body-before.json', '45/original-audit-before.json', '45/cut/independent-expected-paid364.json', '44/paid364.json', '46/allocation-before.json', '46/original-body-before.json', '46/original-audit-before.json', '46/cut/independent-expected-paid365.json', '45/paid365.json', '47/allocation-before.json', '47/original-body-before.json', '47/original-audit-before.json', '47/cut/independent-expected-paid366.json', '46/paid366.json', '48/allocation-before.json', '48/original-body-before.json', '48/original-audit-before.json', '48/cut/independent-expected-paid367.json', '47/paid367.json', '49/allocation-before.json', '49/original-body-before.json', '49/original-audit-before.json', '49/cut/independent-expected-paid368.json', '48/paid368.json', '50/allocation-before.json', '50/original-body-before.json', '50/original-audit-before.json', '50/cut/independent-expected-paid369.json', '49/paid369.json', '51/allocation-before.json', '51/original-body-before.json', '51/original-audit-before.json', '51/cut/independent-expected-paid370.json', '50/paid370.json', '52/allocation-before.json', '52/original-body-before.json', '52/original-audit-before.json', '52/cut/independent-expected-paid371.json', '51/paid371.json', '53/allocation-before.json', '53/original-body-before.json', '53/original-audit-before.json', '53/cut/independent-expected-paid372.json', '52/paid372.json', '54/allocation-before.json', '54/original-body-before.json', '54/original-audit-before.json', '54/cut/independent-expected-paid373.json', '53/paid373.json', '55/cut/p.json', '54/paid374.json', '56/p.json', '55/paid375.json'))+tuple('verification/banc'+n+'/'+p for n in ('57','58','59','60','61','62','63','64') for p in ('p.json','t.json','a.json','b.json','u.json'))


def data(spec, root, cap=262144):
 lock = type(spec) is dict and spec.get('path') == r.SHARED_LOCK
 r.need(type(spec) is dict and type(spec.get('path')) is str
  and (Path(spec['path']).suffix == '.json' or lock),
  'metadata JSON or exact existing ownership token before any operand read')
 r.need(cap <= 262144 or spec['path'] in LEDGER_METADATA_PATHS, 'only finite named ledger metadata cap524288')
 return r.read_pin(spec, root, min(cap, 16384) if lock else cap)


def metadata_pin(spec, path):
 keys(spec, ("path", "bytes", "sha256"), "pin")
 r.exact(spec["path"], path, "literal metadata path")
 r.need(0 < integer(spec["bytes"], "pin bytes") <= 524288
  and type(spec["sha256"]) is str and len(spec["sha256"]) == 64
  and all(c in "0123456789abcdef" for c in spec["sha256"]), "typed bounded pin")


def totals(ledger, policy, grant, caps):
 generated = original = 0.0
 seen = set()
 for index, debit in enumerate(ledger["debits"]):
  r.need(type(debit["cost_id"]) is str and debit["cost_id"] not in seen, "one debit per identity")
  seen.add(debit["cost_id"])
  r.need(debit["mode"] in ("generated", "original"), "finite debit mode")
  fee = number(debit["actual_outer_full_return_s"], "finite actual fee")
  if index >= 118 and debit["mode"] == "generated":
   generated += fee
  if index >= 165 and debit["mode"] == "original":
   original += fee
 stage = float(policy["generated_stage_before_s"]) + generated
 result = {"direct_science_generated_used_s": generated, "generated_stage_used_s": stage,
   "generated_shared_cumulative_used_s": float(policy["generated_shared_before_s"]) + generated,
   "generated_shared_remaining_s": float(caps["generated_global"]) - float(policy["generated_shared_baseline_s"]) - stage,
   "original_cumulative_used_s": float(grant["original_cumulative_before_s"]) + original,
   "original_stage_used_s": float(grant["original_stage_before_s"]) + original,
   "startup_ray_original_local_used_s": float(grant["science_local_used_before_s"]) + original,
   "original_remaining_s": float(caps["original_global"]) - float(grant["original_cumulative_before_s"]) - original}
 for value in result.values():
  number(value, "finite accounting including incurred negative remaining", nonnegative=False)
 return result


def anchor(root):
 closed = data(ANCHOR, root, 524288)
 r.need(len(closed["debits"]) == 306 and len(closed["direct_science_calls"]) == 188, "literal300/182")
 policy, grant = data(r.POLICY_PIN, root), data(r.J_PIN, root)
 r.exact(closed["direct_science_policy"], r.POLICY_PIN, "original policy pin")
 for k, v in totals(closed, policy, grant, HISTORY_CAPS).items():
  r.exact(closed[k], v, "ordered anchor " + k)
 return closed, policy, grant


def _history_proof(spec, root, read, raw):
 event = read(spec)
 r.exact(event, {"schema":"flylab.m3.checkpoint-recovery.v1", "epoch":"contact-checkpoint-harness-20261008-l13r2",
  "before_ledger":ANCHOR, "source_freeze":event["source_freeze"], "failed_bootstrap":FAIL_REPORT,
  "exact_l9_cold":ROOT_COLD, "latest_failure":LATEST_FAILURE, "exact_l10_cold":LATEST_COLD, "latest_timeout":TIMEOUT_REPORT,
  "exact_l12_cold":COLD12, "released_hold":"direct-science-20261008-contact-l12-bootstrap",
  "reason":"KNOWN_EXACT_ROOT_COLD_L12_FAILURE_RETAINED",
  "new_resource_grant_s":0, "refund_s":0}, "whole finite failed recovery")
 metadata_pin(event["source_freeze"], BASE+"/l13r2-s1.json")
 module = sys.modules[__name__]
 _e, previous, policy, grant = r._l9_cold_proof(module, root, read, raw)
 previous = r._l10_failed_bootstrap(module, FAIL_REPORT, OLD_SOURCE, previous, policy, grant, L8_ANCHOR,
  "direct_science_checkpoint_harness_l8_recovery", OLD_RECOVERY, "l8", "l9-before",
  170.472, 1791495442844, 1791495613316, read, raw)
 previous = r._l9_root_cold_proof(module, previous, policy, grant, read, raw)
 closed = read(FAILED303, 524288)
 prior_pin = closed["direct_science_checkpoint_harness_l9r2_recovery"]
 old_source = read(LATEST_FAILURE)["source_freeze"]
 metadata_pin(old_source, BASE+"/l9r2-s1.json")
 metadata_pin(prior_pin, BASE+"/l9r2-recovery1.json")
 r.exact(read(prior_pin), {'schema':"flylab.m3.checkpoint-recovery.v1",
  'epoch':"contact-checkpoint-harness-20261008-l9r2", 'before_ledger':COLD302,
  'source_freeze':old_source, 'failed_bootstrap':FAIL_REPORT, 'exact_l9_cold':ROOT_COLD,
  'released_hold':"direct-science-20261008-contact-l9-bootstrap",
  'reason':"KNOWN_EXACT_ROOT_COLD_L9_FAILURE_RETAINED", 'new_resource_grant_s':0, 'refund_s':0},
  "whole actual previous recovery")
 closed = r._l10_failed_bootstrap(module, LATEST_FAILURE, old_source, previous, policy, grant, FAILED303,
  "direct_science_checkpoint_harness_l9r2_recovery", prior_pin, "l9r2", "l10-before",
  170.146, 1791499127903, 1791499298049, read, raw)
 closed = r._l9_root_cold_proof(module, closed, policy, grant, read, raw, latest=True)
 closed = r._l10_failed_bootstrap(module, TIMEOUT_REPORT, SOURCE11, closed, policy, grant, CLOSED305,
  "direct_science_checkpoint_harness_l11_recovery", read(CLOSED305, 524288)["direct_science_checkpoint_harness_l11_recovery"],
  "l11", "l12-before", 170.664, 1791503352604, 1791503523268, read, raw)
 closed = r._l9_root_cold_proof(module, closed, policy, grant, read, raw, latest="l12")
 return event, closed, policy, grant


def reader_cap(path):
    if type(path) is str:
        if path in LEDGER_METADATA_PATHS: return 524288
        for prefix in (BASE+"/s28-f16-root/before/",BASE+"/s28-f18-root/before/"):
            if path in (prefix+REPLACEMENTS[0],prefix+REPLACEMENTS[1]): return 131072
            if path == prefix+BASE+"/parent-direct-science-account.ps1": return 393216
    if path == _HISTORY_BASE+'/s28-f14-root/a318.json': return 524288
    if path == _HISTORY_BASE+'/s28-f13-root/a318.json': return 524288
    if path == _HISTORY_BASE+'/s28-f12-root/a318.json': return 524288
    if path == _HISTORY_BASE+'/s28-f11-root/a318.json': return 524288
    if path == _HISTORY_BASE+'/s28-f10-root/a318.json': return 524288
    if path == _HISTORY_BASE+'/s28-f9-root/a318.json': return 524288
    if path == _HISTORY_BASE+'/s28-f8-root/a318.json': return 524288
    if path == _HISTORY_BASE+'/s28-f7-root/a318.json': return 524288
    if path == _HISTORY_BASE+'/s28-f6-root/a318.json': return 524288
    if path == _HISTORY_BASE+'/s28-f5-root/a318.json': return 524288
    if path == _HISTORY_BASE+'/s28-f4-root/a318.json': return 524288
    if path in REPLACEMENTS[:2]: return 131072
    if path == BASE + "/parent-direct-science-account.ps1": return 393216
    for n in ("24","25","26","27","28"):
        prefix="verification/banc-natural-gait-m3-20260919/s"+n+"-before/"
        historical={prefix+"tools/banc_m3_checkpoint_harness_runtime.py":131072, prefix+"tools/banc_m3_checkpoint_harness_runtime_native.ps1":131072, prefix+"verification/banc-natural-gait-m3-20260919/parent-direct-science-account.ps1":393216}
        if path in historical: return historical[path]
    if path == r.SHARED_LOCK: return 16384
    if path in _OLD_LEDGER_PATHS: return 524288
    if any(path == p or path.endswith("/"+p) for p in REPLACEMENTS[:2]): return 65536
    return 262144

def catalog_auth(value,root):
 source_fence(r.text_spec(SOURCE_PATH,root,262144),root)
 keys(value, ("schema","complete_status","project_execution","rows","Original_arrays","trust_basis","deduplicated_path_cap_kind","parent_catalog"),"catalog")
 r.exact(value["schema"],"flylab.m3.s28.ROOT-static-historical-catalog.v1","catalog schema")
 r.need(type(value["rows"]) is list and len(value["rows"]) == 779,"complete actual finite catalog")
 seen, paths = set(), {}
 for i, row in enumerate(value["rows"]):
  keys(row, ("ordinal","pin","cap","kind","reason"),"reader row")
  r.exact(integer(row["ordinal"],"ordinal"), i,"ordered catalog ordinal")
  pin, cap, kind = row["pin"], row["cap"], row["kind"]
  r.need(type(row["reason"]) is str and type(kind) is str,"reader strings")
  r.exact(integer(cap,"cap"), reader_cap(pin["path"]),"path reader cap")
  key = (pin["path"], cap, kind)
  r.need(key not in seen,"unique reader tuple")
  seen.add(key)
  if pin["path"] in paths: r.exact(pin, paths[pin["path"]],"consistent same-path pins")
  paths[pin["path"]] = pin
  _history_operand(pin,root, cap, kind)
 r.need(len(paths) == 583 and sum(row["kind"] == "raw0" for row in value["rows"]) == 10,"actual finite paths/raw0 oldeight plus S25")
 return tuple((row["pin"]["path"],row["pin"]["bytes"],row["pin"]["sha256"],row["cap"],row["kind"]) for row in value["rows"])

def pinned_anchor(root):
 extension = data(S15_PINS["repair"],root)
 r.exact(extension, {'schema': 'flylab.m3.s28.ROOT-zero-resource-source-continuation.v1', 'epoch': 'contact-checkpoint-harness-20261009-s28', 'status': 'PROSPECTIVE_METADATA_ONLY_NO_LEDGER_RESOURCE_DELTA', 'trust': 'ROOT_WHOLE318_ACTUAL_S27_FAILURE_SUPPORT_DECLINED_EXPLICIT_TRUST_NOT_HISTORY_REPLAY', 'before_ledger': {'bytes': 495417, 'sha256': '70175b4bc56c19412b64500343ebc72bc538a7b3776dd60733d43efeeb59476e', 'path': 'verification/banc-natural-gait-m3-20260919/parent-e358-execution-ledger.json'}, 'source_before': {'bytes': 262139, 'sha256': '35b6cf63018f61f3fde34d47045d16f5e3c97dc53bac64b1fe7814aed0aa61c6', 'path': 'verification/banc-natural-gait-m3-20260919/s27-s1.json'}, 'anchor': {'bytes': 495417, 'sha256': '70175b4bc56c19412b64500343ebc72bc538a7b3776dd60733d43efeeb59476e', 'path': 'verification/banc-natural-gait-m3-20260919/s28-anchor318-before-recovery.json'}, 'old_caps': {'generated_global': 6530, 'original_global': 13370, 'generated_local': 5794}, 'new_caps': {'generated_global': 6530, 'original_global': 13370, 'generated_local': 5794}, 'delta_s': {'generated_global': 0, 'original_global': 0, 'generated_local': 0}, 'combined_global_s': 19900, 'new_resource_grant_s': 0, 'refund_s': 0, 'reset': False, 'retained_e23_event': {'bytes': 2130, 'sha256': '1e70cd7f9280407321f40c5c53d2336a4cd2f1f328ad6d743c435b3f9c9f1d2c', 'path': 'verification/banc-natural-gait-m3-20260919/e23.json'}, 'retained_e23_pointer': 'direct_science_checkpoint_extension_s23_2000', 'exact_ledger_resource_delta': [], 'canonical_new_extension_pointer': False, 'provenance': {'root_design': {'bytes': 8964, 'sha256': 'd461a02466a69ae131975590ca2610edeac8b7f661d4c329a5b061b38b8a1e58', 'path': 'docs/M3_28D.md'}, 'astra_json': {'bytes': 107955, 'sha256': '567ff76f3033c612efcb1f56a7e0ee955f4867182581e50030fc67c9aa17ce6e', 'path': 'verification/banc-natural-gait-m3-20260919/s28-aj.json'}, 'root_cause': {'bytes': 11668, 'sha256': 'a3fdc896a0f39060f379b08a433b44bb59be17e49c57e3d0de33dbe4f9b12d33', 'path': 'verification/banc-natural-gait-m3-20260919/s28-root-cause.json'}, 'root_reflection': {'bytes': 11407, 'sha256': '85614debeff6238e5b0516fd303d2179dc01d67ebb7727d11abef5c57c995363', 'path': 'docs/M3_28R.md'}, 'astra_markdown': {'bytes': 30684, 'sha256': '218cf3b65877bb7d47ed9eed16ba99f6f4cf88185c5e3ad9c17818ad4e5f3b39', 'path': 'verification/banc-natural-gait-m3-20260919/s28-am.md'}, 'before_index': {'bytes': 4915, 'sha256': '7c9ca2a35817b59ab1f91bbf699f48e1ff40a8394044b2c9d911b1e305255bd5', 'path': 'verification/banc-natural-gait-m3-20260919/s28-before-index.json'}}},"whole exact zero-resource S28 event including canonical descriptor metadata")
 r.exact((extension["before_ledger"]["bytes"],extension["before_ledger"]["sha256"]),(S15_PINS["original"]["bytes"],S15_PINS["original"]["sha256"]),"canonical before metadata and immutable sealed318 raw identity")
 original = data(S15_PINS["original"],root,524288)
 policy, grant = data(r.POLICY_PIN,root), data(r.J_PIN,root)
 r.need(type(original["debits"]) is list and len(original["debits"]) == 318 and type(original["direct_science_calls"]) is dict and len(original["direct_science_calls"]) == 200,"trusted whole318/paid200")
 r.exact(original["direct_science_policy"],r.POLICY_PIN,"immutable policy")
 r.exact(original["direct_science_blocking_holds"],{S15_HOLD:FAILURE_HOLD},"sole exact retained actual S27 HOLD")
 for key in r.OWNER_KEYS:
  if key != "direct_science_blocking_holds": r.exact(original[key],{},"closed checkpoint owners")
 for key in ("original_jobs_running","original_reserved_s"): r.exact(original[key],0,"closed Original liabilities")
 for key,cap in OLD_CAPS.items(): r.exact(original[r.CAP_FIELDS[key]],cap,"unchanged actual e23 caps")
 for key,value in totals(original,policy,grant,OLD_CAPS).items(): r.exact(original[key],value,"ordered actual316 totals offsets118/165")
 r.exact(original[REPAIR_KEY],extension["retained_e23_event"],"actual e23 resource pointer retained")
 for key,value in (("direct_science_generated_used_s",4264.651),("generated_shared_remaining_s",1537.1369844),("original_remaining_s",1106.9150391004946)):
  r.need(type(original[key]) is float and original[key].hex() == value.hex(),"exact ordered binary64 "+key)
 post = data(S15_PINS["anchor"],root,524288)
 r.exact(r.raw_pin(S15_PINS["original"],root,524288),r.raw_pin(S15_PINS["anchor"],root,524288),"same whole318 raw bytes; mutable canonical is never re-read")
 r.exact(post,original,"whole318 zero resource delta; all history/e23/e25/Original retained")
 return post,policy,grant

PRODUCTION_REASONS = tuple(range(13))+tuple(range(14,20))+tuple(range(22,36))+tuple(range(38,138))

def whole_exact(a,b):
 if type(a) is not type(b):raise ValueError("type")
 if type(a) is dict:
  if tuple(a)!=tuple(b):raise ValueError("keys")
  for k in a:whole_exact(a[k],b[k])
 elif type(a) in (list,tuple):
  if len(a)!=len(b):raise ValueError("array")
  for x,y in zip(a,b):whole_exact(x,y)
 elif type(a) is float:
  if a.hex()!=b.hex():raise ValueError("bits")
 elif a!=b:raise ValueError("scalar")

def terminal_bridge(root):return data(CURRENT_CONTRACT,root)["terminal_bridge"]

def frozen_history(contract,root):
 old=data(HISTORY_AUTHORITY,root)
 whole_exact(contract["checkpoint_continuity_evidence"],{'authority':HISTORY_AUTHORITY,'member':"checkpoint_continuity_evidence"})
 for pin in FROZEN_HISTORY_PINS:r.raw_pin(pin,root,reader_cap(pin["path"]))
 for name in ("d12","d13"):
  e=old["checkpoint_continuity_evidence"][name];gate=e["gate"]
  gp=S15_PINS["technical_refusal"] if name=="d13" else data(S15_PINS["previous_contract"],root)["pinned_checkpoint"]["pins"]["technical_refusal"]
  known=gate["actual_known_compiler_type_refusal" if name=="d13" else "actual_known_layout_compatibility_diagnostic"]
  raw=r.raw_pin(gate["journal"],root,262144)
  r.need(raw.endswith(b"\n") and b"\r" not in raw and len(raw)<=524288,"history LF")
  lines=raw.decode("utf8").splitlines();r.need(all(len(x.encode())+1<=384 for x in lines),"record cap")
  rp=S15_PINS["previous_recovery"] if name=="d12" else data(gate["actual_tool_return"],root)["request"]
  whole_exact([data(gate["actual_tool_return"],root),data(known,root),data(gp,root),data(rp,root),[json.loads(x) for x in lines]],[e["tool"],e["known"],e["gate"],e["recovery"],e["journal_rows"]])
 return tuple(FROZEN_HISTORY_PINS)

def predecessor(nb,root):
 aa=[a for a in nb['archives'] if a['logical_current_before']['path']==CONTRACT_PATH];r.need(len(aa)==1,'unique predecessor contract');a=aa[0];old=data(a['immutable_archive'],root);sf=data(nb['adoption_before_Source'],root);rr=[x for x in nb['archives'] if x['logical_current_before']['path']==REVIEW_PATH];r.need(len(rr)==1,'unique predecessor review');rp=rr[0];rv=data(rp['immutable_archive'],root);whole_exact(sf['reviewed_source'],rp['logical_current_before']);whole_exact([sf['checkpoint_contract'],sf['sources'][CONTRACT_PATH],rv['sources'][CONTRACT_PATH]],[a['logical_current_before']]*3);whole_exact(sf['sources'],rv['sources'])
 return old

def terminal_baseline(root):
 b=terminal_bridge(root)
 for k in ("terminal","admitted","pre_backup","replace_backup","cost","receipt","actor","audit","tool","failure","failed_tool","request"):
  p=b[k];r.raw_pin(p,root,reader_cap(p["path"]))
 for a in b["archives"]:r.raw_pin(a["immutable_archive"],root,262144)
 t=data(b["terminal"],root,524288);a=data(b["admitted"],root,524288)
 r.exact(r.raw_pin(b["admitted"],root,524288),r.raw_pin(b["pre_backup"],root,524288),"admitted")
 r.exact(r.raw_pin(b["pre_backup"],root,524288),r.raw_pin(b["replace_backup"],root,524288),"backups")
 c=data(b["cost"],root);receipt=data(b["receipt"],root);actor=data(b["actor"],root);audit=data(b["audit"],root);tool=data(b["tool"],root)
 whole_exact(receipt["actual_metadata_actor"],actor);whole_exact(audit["actual_actor"],actor)
 r.exact((actor["pid"],actor["creation_filetime_1601"]),(26436,134361160074558301),"actor")
 r.exact((audit["first_FINAL"],audit["session_spent"],audit["exit_code"],tool["genuine_return_kind"],tool["session_id"],tool["session_spent"]),("396ac0",55000,0,"FIRST_WRITE_STDIN_FINAL",55000,True),"settlement")
 for k,n in (("cost","cost"),("immutable_before","pre_backup"),("replace_backup","replace_backup"),("old_request","request")):whole_exact(receipt[k],b[n])
 r.exact((receipt["immutable_after"]["bytes"],receipt["immutable_after"]["sha256"]),(b["terminal"]["bytes"],b["terminal"]["sha256"]),"actual after and terminal raw identity")
 call="direct-science-20261009-contact-s28-bootstrap"
 r.exact((c["call_id"],c["reported_tool_wall_s"],c["measurement_basis"],c["returned_exit_code"],c["scientific_runtime"]),(call,0.4983084,"genuine_exec_return.wall_time_seconds",1,"NOT_RUN"),"mixed fee")
 r.exact((c["original_supervisor"]["status"],c["original_supervisor"]["pid"],c["stdout_eof"],c["stderr_eof"]),("NOT_CAPTURED",None,False,False),"old unknown")
 r.need(len(t["debits"])==319 and len(t["direct_science_calls"])==201 and sum(d["cost_id"]==call for d in t["debits"])==1,"319/201")
 whole_exact(t["direct_science_blocking_holds"],{call:b["hold"]})
 for k in r.OWNER_KEYS:
  if k!="direct_science_blocking_holds":whole_exact(t[k],{})
 whole_exact([t["original_jobs_running"],t["original_reserved_s"]],[0,0])
 whole_exact(t["debits"][-1],{'actual_outer_full_return_s':0.4983084,'mode':"generated",'descriptor':b["cost"],'cost_id':call,'actual_returned_exit_code':1})
 changed=[k for k in dict.fromkeys((*a,*t)) if k not in a or k not in t or a[k]!=t[k]]
 r.exact(changed,b["nine_delta"],"nine")
 rev=copy.deepcopy(t)
 for k in b["nine_delta"]:
  if k in a:rev[k]=copy.deepcopy(a[k])
  else:rev.pop(k)
 whole_exact(rev,a)
 oldsource=data(HISTORY_BEFORE_SOURCE,root)
 archive={x["logical_current_before"]["path"]:x for x in b["archives"]}
 for x in archive.values():r.exact((x["logical_current_before"]["bytes"],x["logical_current_before"]["sha256"]),(x["immutable_archive"]["bytes"],x["immutable_archive"]["sha256"]),"archive")
 oldreview=data(archive[REVIEW_PATH]["immutable_archive"],root)
 whole_exact(oldsource["sources"],oldreview["sources"])
 whole_exact(oldsource["checkpoint_contract"],archive[CONTRACT_PATH]["logical_current_before"])
 whole_exact(t[KEY],archive[BASE+"/s28-recovery12.json"]["logical_current_before"])
 ct=data(CURRENT_CONTRACT,root);nb=ct['actual320'];whole_exact(nb['adoption_before_Source'],CONTINUATION_BEFORE_SOURCE)
 old=predecessor(nb,root);older=predecessor(old['actual320'],root)
 def reorder(value,maps):
  out=copy.deepcopy(value)
  for m in sorted(maps,key=lambda v:len(v['parts']),reverse=True):
   node=out
   for part in m['parts']:node=node[part]
   kk=list(node);r.exact(hashlib.sha256(json.dumps(kk,ensure_ascii=True,separators=(',',':')).encode()).hexdigest(),m['keys_sha256'],'fixed bridge key order');r.exact(sorted(m['indices']),list(range(len(kk))),'exact permutation');ordered={kk[i]:node[kk[i]] for i in m['indices']}
   if not m['parts']:out=ordered
   else:
    parent=out
    for part in m['parts'][:-1]:parent=parent[part]
    parent[m['parts'][-1]]=ordered
  return out
 prior_hold=b["hold"]
 basecall=call
 for paid,amount,cid,nd,nc in ((older['actual320'],115.76,basecall+'-r1',320,202),(old['actual320'],0.672,basecall+'-r2',321,203),(nb,123.633,basecall+'-r3',322,204)):
  for pin in paid['raw_pins']:r.raw_pin(pin,root,reader_cap(pin['path']))
  r.exact(paid['old_failed_call'],cid,'fixed paid identity')
  e=copy.deepcopy(t);whole_exact(e['direct_science_blocking_holds'],{call:prior_hold});del e['direct_science_blocking_holds'][call];e[KEY]=paid['old_continuation'];whole_exact(reorder(e,paid['order_relations']['before']),data(paid['before_full4'],root,524288))
  q=data(paid['old_request'],root);c=data(paid['old_cost'],root);cid=paid['old_failed_call'];r.exact((q['call_id'],q['mode'],q['limits']['outer_s'],c['cost_id'],c['request'],c['charged_once'],c['actual_outer_full_return_s'],c['exit_code'],c['cap_exceeded']),(cid,'generated',180,cid,paid['old_request'],True,amount,1,False),'r1 once')
  e['direct_science_calls'][cid]={'request':paid['old_request'],'state':'admitted','mode':q['mode'],'reserved_s':q['limits']['outer_s']};e['direct_science_reservations'][cid]={'request':paid['old_request'],'reserved_s':q['limits']['outer_s']};whole_exact(reorder(e,paid['order_relations']['admitted']),data(paid['admitted_full4'],root,524288))
  tr=data(c['trace'],root);outer=data(c['outer'],root);r.exact((tr['phase'],tr['first_error'],tr['request'],tr['source_freeze'],tr['full_return_s'],tr['exit_code'],outer['request'],outer['exit_code']),('FINAL',None,paid['old_request'],q['source_freeze'],amount,1,paid['old_request'],1),'real prior FINAL');r.need(closure(outer),'real r1 closure');fee_values(c,q,paid['old_request'],tr,outer,root)
  e['direct_science_calls'][cid]['state']='charged';e['direct_science_reservations']={};e['direct_science_workers']={};e['direct_science_pending_actuals']={};e['direct_science_blocking_holds']={cid:paid['old_failed_hold']};e['debits'].append({'cost_id':cid,'descriptor':paid['old_cost'],'mode':'generated','actual_outer_full_return_s':c['actual_outer_full_return_s'],'actual_returned_exit_code':c['exit_code']});e.update(totals(e,r.read_pin(r.POLICY_PIN,root),r.read_pin(r.J_PIN,root),OLD_CAPS));whole_exact(reorder(e,paid['order_relations']['paid']),data(paid['terminal'],root,524288));t=data(paid['terminal'],root,524288)
  r.exact((len(t['debits']),len(t['direct_science_calls'])),(nd,nc),'fixed paid counts');call,prior_hold=cid,paid['old_failed_hold']
 return t,r.read_pin(r.POLICY_PIN,root),r.read_pin(r.J_PIN,root)

def refinement_before(root):
 c=r.d19_current_contract(globals(),root);return data(data(c['d19_fixed_history'],root)['independent_expected'],root,524288)
 source_fence(r.text_spec(SOURCE_PATH,root,262144),root)
 return terminal_baseline(root)[0]

def refinement_input(ledger,before_raw,root):
 c=r.d19_current_contract(globals(),root);cut=data(c['d19_fixed_history'],root);r.d19_whole(ledger,refinement_before(root));r.exact(before_raw,r.raw_pin(cut['authenticated_actual'],root,524288),'D19 fixed-history raw');return copy.deepcopy(ledger)
 r.need(type(before_raw) is bytes,"own raw")
 r.exact(before_raw,r.raw_pin(data(CURRENT_CONTRACT,root)["actual320"]["terminal"],root,524288),"terminal raw")
 whole_exact(ledger,terminal_baseline(root)[0])
 return copy.deepcopy(ledger)

def recovery(spec,root):
 return r.d19_recovery(globals(),spec,root)
 key = None
 owned = None
 try:
  r.need(not _HISTORY_BUSY,"no reentrant checkpoint proof")
  owned = object()
  _HISTORY_BUSY.add(owned)
  metadata_pin(spec, RECOVERY_PATH)
  event = data(spec,root)
  baseline319,policy319,grant319=terminal_baseline(root)
  history_bridge=terminal_bridge(root)
  bridge=data(CURRENT_CONTRACT,root)["actual320"]
  bridge=dict(bridge,event_keys=history_bridge["event_keys"],event_schema=history_bridge["event_schema"])
  source = event["source_freeze"]
  metadata_pin(source, SOURCE_PATH)
  freeze = data(source,root)
  r.exact(freeze["qualification_epoch"], EPOCH,"accepted Source epoch")
  r.exact(freeze["ROOT_review_cycle_complete"], True,"accepted ROOT Source")
  contract_pin = freeze["checkpoint_contract"]
  r.exact(contract_pin, freeze["sources"][CONTRACT_PATH],"accepted whole contract pin")
  contract = data(contract_pin,root)
  r.exact(contract["epoch"], EPOCH,"contract epoch")
  binding = contract["pinned_checkpoint"]
  keys(binding, ("pins","trust","coverage","latest_failure","latest_cost","policy","grant","provenance","repair_delta","post_counts"),"checkpoint binding")
  r.exact(binding["pins"],S15_PINS,"fixed original/post/catalog/repair/Source")
  r.exact(binding["trust"],"ROOT_WHOLE318_ACTUAL_S27_FAILURE_SUPPORT_DECLINED_EXPLICIT_TRUST_NOT_HISTORY_REPLAY","explicit acceptance boundary")
  r.exact(binding["repair_delta"], [],"no new repair delta")
  r.exact(binding["post_counts"], {'debits': 322, 'paid_calls': 204, 'bootstrap_debits': 323, 'bootstrap_calls': 205, 'qualified_debits': 327, 'qualified_calls': 209, 'body_debits': 328, 'body_calls': 210, 'audit_debits': 329, 'audit_calls': 211},"exact D15B baseline and successor counts")
  for pin in (binding["coverage"], binding["latest_failure"], binding["latest_cost"], binding["policy"], binding["grant"], *binding["provenance"].values()):
   r.raw_pin(pin,root, 262144)
  r.exact((binding["policy"],binding["grant"]), (r.POLICY_PIN,r.J_PIN),"original policy/grant")
  keys(event,bridge["event_keys"],"finite new event keys")
  aliases={a["logical_current_before"]["path"]:a["logical_current_before"] for a in bridge["archives"]}
  expected_event={'schema':bridge["event_schema"],'epoch':EPOCH,'before_ledger':bridge["terminal"],'source_freeze':source,'review':freeze["reviewed_source"],'settlement':bridge["commit_capture"],'cost':bridge["old_cost"],'audit':bridge["failure"],'previous_recovery':bridge["old_continuation"],'old_source':aliases[SOURCE_PATH],'old_review':aliases[REVIEW_PATH],'old_contract':aliases[CONTRACT_PATH],'root_reviews':event["root_reviews"],'sol_reports':event["sol_reports"],'old_caps':OLD_CAPS,'new_caps':CAPS,'new_resource_grant_s':0,'refund_s':0,'reset':False}
  whole_exact(event,expected_event)
  metadata_pin(freeze["reviewed_source"],REVIEW_PATH)
  review = data(freeze["reviewed_source"],root)
  keys(review,('schema','epoch','sources','replacements','additions','before_source','root_reflection','astra_review','root_reviews','sol_reports'),"complete recovery reviewed metadata")
  r.exact((freeze["schema"],review["schema"],review["epoch"]),("flylab.m3.contact-learning.source-freeze.v1","flylab.m3.checkpoint-reviewed-source.v1",EPOCH),"Source review exact schemas and epoch")
  r.exact(contract_pin,review["sources"][CONTRACT_PATH],"Source contract vector equals review")
  r.exact(freeze["sources"],review["sources"],"whole Source vector equals review before first catalog and every cache")
  whole_exact(event['root_reviews'],review['root_reviews']);whole_exact(event['sol_reports'],review['sol_reports'])
  direct_pins = provenance(review,root)
  policy = catalog_auth(data(S15_PINS["catalog"],root),root)
  for name in ("original","anchor","repair"):
   r.raw_pin(S15_PINS[name],root, 262144 if name == "repair" else 524288)
  continuity_gate = data(S15_PINS["technical_refusal"],root)
  prior_contract = data(S15_PINS["previous_contract"],root)
  incoming_pins = prior_contract["pinned_checkpoint"]["pins"]
  old_gate = data(incoming_pins["technical_refusal"],root)
  continuity_pins = (incoming_pins["previous_source"],incoming_pins["previous_contract"],incoming_pins["technical_refusal"],*(old_gate[n] for n in ("actual_tool_return","actual_known_layout_compatibility_diagnostic","journal")),*(continuity_gate[n] for n in ("actual_tool_return","actual_known_compiler_type_refusal","journal")),data(continuity_gate["actual_tool_return"],root)["request"])
  key = (os.fspath(Path(root).absolute()),"terminal320-v1",EPOCH,tuple((p["path"],p["bytes"],p["sha256"]) for p in (source,contract_pin,freeze["reviewed_source"],*review["root_reviews"],*review["sol_reports"],*direct_pins,*S15_PINS.values(),*continuity_pins,*FROZEN_HISTORY_PINS,*bridge["raw_pins"],*[a['immutable_archive'] for a in bridge['archives']],r.POLICY_PIN,r.J_PIN)),policy)
  if key not in _HISTORY_PROOF:
   _HISTORY_PROOF.clear()
   baseline, p, g = baseline319,policy319,grant319
   _HISTORY_PROOF[key] = copy.deepcopy((baseline,p,g))
  baseline, p, g = copy.deepcopy(_HISTORY_PROOF[key])
  return event, baseline, p, g
 except Exception:
  _HISTORY_PROOF.clear()
  raise
 finally:
  if owned is not None: _HISTORY_BUSY.discard(owned)

def _history_operand(pin,root, cap, kind):
 keys(pin, ("path","bytes","sha256"),"historical pin")
 r.need(type(pin["path"]) is str,"literal path type")
 if kind == "raw0":
  r.need(pin["path"] in EMPTY_PATHS,"exact ten empty stderr paths")
  r.exact(pin, {'path':pin["path"],'bytes':0,'sha256':"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},"exact empty stderr pin")
 else: metadata_pin(pin, pin["path"])
 r.need(type(cap) is int and type(kind) is str and kind in ("raw","raw0","json"),"typed reader policy")
 r.exact(cap, reader_cap(pin["path"]),"path cap for every reader kind")
 if kind == "json": r.need(Path(pin["path"]).suffix == ".json","historical JSON namespace")
 return r.raw_pin(pin,root, cap, allow_empty=kind == "raw0")
def recovered_snapshot(spec, root):
 return r.d19_recovered_snapshot(globals(),spec,root)
 _event, closed, _p, _g = recovery(spec, root)
 result = copy.deepcopy(closed)
 result[KEY] = spec
 call="direct-science-20261009-contact-s28-bootstrap-r3"
 r.exact(result["direct_science_blocking_holds"],{call:data(CURRENT_CONTRACT,root)["actual320"]["old_failed_hold"]},"old HOLD")
 del result["direct_science_blocking_holds"][call]
 return result

def roster(freeze, root):
 old = data(dict(L_ROSTER, path=BASE+"/s28-before/"+ROSTER_PATH,bytes=47801,sha256="53cc880df410197b07f32ae15258c1cd062d6f2bd150955bde6d781b1ecec7d7"), root)
 new = data(freeze["sources"][ROSTER_PATH], root)
 expected = copy.deepcopy(old)
 expected["epoch"] = EPOCH
 r.exact(new, expected, "whole tested L23/61 roster and exact A25/B36 unchanged")
 r.need(len(new["math"]) == 23 and len(new["gate"]) == 61, "whole inherited tests")
 for kind in ("math", "gate", "gate_a", "gate_b"):
  rows = new[kind]
  r.need(len({row["node_id"] for row in rows}) == len(rows), "no duplicate node")
  for row in rows:
   r.exact(row["node_id"], row["module"] + "::" + row["name"], "full parameter node")
   r.exact(row["classname"], row["module"][:-3].replace("/", "."), "full classname")
 a, b, whole = ({row["node_id"] for row in new[k]} for k in ("gate_a", "gate_b", "gate"))
 r.need(not a & b and a | b == whole, "whole disjoint gate union")
 for module in new["indivisible_groups"]:
  nodes = {row["node_id"] for row in new["gate"] if row["module"] == module}
  r.need(nodes <= a or nodes <= b, "whole indivisible Native groups")
 r.exact([row["name"] for row in new["bootstrap"]], list(BOOTSTRAP_NAMES), "whole inherited L4")
 return new


def plan(call, freeze, root):
 r.need(call in SEQUENCE, "only seven literal K calls")
 i = SEQUENCE.index(call)
 limits = dict(zip(("outer_s", "child_s", "cleanup_s"), CAP_TRIPLES[i], strict=True))
 limits.update(host_bytes=2147483648 if i < 5 else 8589934592, output_cap_bytes=8388608 if i < 5 else 26214400)
 roles = ("execution", "lint", "numerical", "execution", "execution", "contact_candidate_body", "contact_body_audit")
 result = {"call": call, "role": roles[i], "limits": limits,
   "mode": "generated" if i < 5 else "original",
   "output": BASE + "/parent-" + call + ("" if i < 5 else "/result-supervision")}
 if i < 5:
  if i == 1:
   result["argv"] = ["-m", "ruff", "check", *LINT_MODULES]
  else:
   kind = ("bootstrap", "lint", "math", "gate_a", "gate_b")[i]
   rows = roster(freeze, root)[kind]
   suffix = ("r64-bootstrap", "r64-lint", "r64-math", "r64-gate-a", "r64-gate-b")[i]
   result["argv"] = ["-m", "pytest", *[row["node_id"] for row in rows], "--tb=short", "-q",
       "--junitxml=" + BASE + "/parent-contact-learning-s28-" + suffix + "-junit.xml"]
   if i == 0: result["argv"].append("--maxfail=1")
 else:
  contract = data(freeze["contract"], root)
  result["limits"] = {k: contract["limits"][roles[i]][k] for k in limits}
  r.exact(tuple(result["limits"][k] for k in ("outer_s", "child_s", "cleanup_s")), CAP_TRIPLES[i], "unchanged Original envelope")
 return result


def identity(q, request_spec, source_spec, root):
 r.d19_request_policy(globals(),q,request_spec)
 metadata_pin(source_spec, SOURCE_PATH)
 freeze = data(source_spec, root)
 p = plan(q["call_id"], freeze, root)
 expected_keys = {"schema", "call_id", "mode", "role", "source_freeze", "limits", "output", "ledger_path", "shared_original_lock_path"}
 expected_keys |= {"generated_behavior", "generated_argv"} if p["mode"] == "generated" else {"grant", "science_request"}
 keys(q, expected_keys, "whole K supervision request")
 for k, v in (("schema", "flylab.m3.direct-science.supervision.v1"), ("source_freeze", source_spec),
    ("call_id", p["call"]), ("mode", p["mode"]), ("role", p["role"]), ("limits", p["limits"]),
    ("output", p["output"]), ("ledger_path", BASE + "/parent-e358-execution-ledger.json"),
    ("shared_original_lock_path", r.SHARED_LOCK)):
  r.exact(q[k], v, "exact K request " + k)
 if q["mode"] == "generated":
  r.exact(q["generated_behavior"], "command", "one generated command")
  r.exact(q["generated_argv"], p["argv"], "whole exact K argv")
 else:
  r.exact(q["grant"], r.J_PIN, "unchanged J grant")
  science = data(q["science_request"], root)
  original_metadata(science, q, root)
  for k in ("call_id", "role", "grant", "source_freeze", "limits"):
   value = {x: science["limits"][x] for x in q["limits"]} if k == "limits" else science[k]
   r.exact(q[k], value, "same K science/outer " + k)
 r.exact(data(request_spec, root), q, "intact admitted request bytes")
 return freeze


def closure(outer):
 for key in ("worker_started", "creation_attempted", "job_created", "resume_attempted", "all_owned_ended", "reservation_released", "stdout_eof", "stderr_eof"):
  r.need(type(outer[key]) is bool, "mandatory typed " + key)
 state = outer["worker_creation_state"]
 r.need(state in ("NO_CREATE_ATTEMPT", "CREATED_SUSPENDED_NOT_RESUMED", "RESUMED", "UNCERTAIN_CREATION"), "finite creation state")
 r.need(outer["resume_outcome"] in ("NOT_ATTEMPTED", "RESUMED", "UNCERTAIN"), "finite resume outcome")
 keys(outer["unknown_owned_identities"], outer["unknown_owned_identities"], "unknown identities")
 r.need(len(outer['unknown_owned_identities']) <= 1024, 'bounded owned identities')
 for pid, ticks in outer['unknown_owned_identities'].items():
  r.need(type(pid) is str and pid.isascii() and pid.isdecimal() and int(pid) > 0 and str(int(pid)) == pid, 'typed unknown PID')
  integer(ticks, 'unknown creation ticks', positive=True)
 for k in ('owner_pid', 'owner_creation_ticks'):
  integer(outer[k], 'actual owner ' + k, positive=True)
 r.exact(outer['reservation_released'], outer['all_owned_ended'], 'release only with known whole closure')
 if state == "NO_CREATE_ATTEMPT":
  for k in ("worker_started", "creation_attempted", "resume_attempted", "stdout_eof", "stderr_eof"):
   r.exact(outer[k], False, "lexical no-create never observed EOF")
  for k in ("worker_pid", "worker_creation_ticks", "worker_exit_code"):
   r.exact(outer[k], None, "no synthetic identity/exit")
  r.exact(outer["resume_outcome"], "NOT_ATTEMPTED", "no synthetic resume")
  r.exact(outer['unknown_owned_identities'], {}, 'no invented no-create identities')
  return outer["all_owned_ended"] and outer["reservation_released"] and not outer["unknown_owned_identities"]
 r.exact(outer["creation_attempted"], True, "actual creation attempted")
 if state == "UNCERTAIN_CREATION":
  r.exact(outer["all_owned_ended"], False, "uncertain creation retains liability")
  r.exact(outer['reservation_released'], False, 'unknown reservation retained')
  r.exact(outer['job_created'], True, 'uncertainty belongs to actual owned Job')
  r.exact(outer['worker_started'], False, 'resume never confirmed')
  r.exact(outer['resume_outcome'], 'UNCERTAIN' if outer['resume_attempted'] else 'NOT_ATTEMPTED', 'finite uncertainty phase')
  if outer['resume_attempted']:
   for k in ('worker_pid', 'worker_creation_ticks'):
    integer(outer[k], 'known suspended identity ' + k, positive=True)
   r.exact(outer['unknown_owned_identities'].get(str(outer['worker_pid'])), outer['worker_creation_ticks'], 'retained suspended identity')
   r.need(outer['worker_pid'] != outer['owner_pid'], 'distinct observed suspended worker/owner')
  else:
   for k in ('worker_pid', 'worker_creation_ticks', 'worker_exit_code'):
    r.exact(outer[k], None, 'unconfirmed creation has no invented identity/exit')
   r.exact(outer['unknown_owned_identities'], {}, 'no observed identity before successful CreateProcess')
   for k in ('stdout_eof', 'stderr_eof'):
    r.exact(outer[k], False, 'no installed readers before confirmed creation')
  if outer['worker_exit_code'] is not None:
   integer(outer['worker_exit_code'], 'observed uncertain-resume worker exit')
  return False
 for k in ("worker_pid", "worker_creation_ticks", "owner_pid", "owner_creation_ticks"):
  integer(outer[k], "actual process " + k, positive=True)
 r.exact(outer["job_created"], True, "owned actual Job")
 r.need(outer['worker_pid'] != outer['owner_pid'], 'distinct worker/owner')
 if outer['worker_exit_code'] is not None:
  integer(outer['worker_exit_code'], 'actual worker exit')
 r.exact(outer['resume_attempted'], state == 'RESUMED', 'actual finite resume attempt')
 r.exact(outer["worker_started"], state == "RESUMED", "worker_started means resumed")
 r.exact(outer["resume_outcome"], "RESUMED" if state == "RESUMED" else "NOT_ATTEMPTED", "actual resume outcome")
 if outer['all_owned_ended']:
  integer(outer['worker_exit_code'], 'known closed worker exit')
  r.exact(outer['unknown_owned_identities'], {}, 'closed means no unknown identities')
  for k in ('stdout_eof', 'stderr_eof'):
   r.exact(outer[k], True, 'known closure requires actual two EOF')
 else:
  r.exact(outer['unknown_owned_identities'].get(str(outer['worker_pid'])), outer['worker_creation_ticks'], 'retain every unclosed worker identity')
 return (outer["all_owned_ended"] and outer["reservation_released"] and not outer["unknown_owned_identities"]
   and outer["stdout_eof"] and outer["stderr_eof"])


def fee_proof(cost_spec, debit, source_spec, root):
 if debit['cost_id']==r.d19_capsule()['call_id']:return r.d19_historical_fee(globals(),cost_spec,debit,source_spec,root)
 """Intact immutable evidence, never current executable Source validation."""
 metadata_pin(cost_spec, BASE + "/parent-" + debit["cost_id"] + "-full-return.json")
 cost = data(cost_spec, root)
 request_spec = cost["request"] if debit["mode"] == "generated" else cost["supervisor_request"]
 q = data(request_spec, root)
 identity(q, request_spec, source_spec, root)
 trace, outer = data(cost["trace"], root, 262144), data(cost["outer"], root, 262144)
 for k in ("request", "source_freeze"):
  r.exact(trace[k], request_spec if k == "request" else source_spec, "genuine K trace binding")
 r.exact(outer["request"], request_spec, "immutable outer binding")
 r.exact(trace["schema"], "flylab.m3.direct-science.genuine-return.v1", "actual collector")
 r.exact(trace["phase"], "FINAL", "genuine first FINAL")
 r.exact(trace["first_error"], None, "no collector uncertainty")
 integer(trace["dispatch_ms"], "genuine dispatch", positive=True)
 integer(trace["first_final_ms"], "genuine first FINAL", positive=True)
 fee = (trace["first_final_ms"] - trace["dispatch_ms"]) / 1000.0
 r.need(number(fee, "positive exact fee") > 0, "positive elapsed")
 r.exact(trace["full_return_s"], fee, "genuine full return")
 for k, v in (("cost_id", q["call_id"]), ("mode", q["mode"]), ("source_freeze", source_spec),
    ("actual_outer_full_return_s", fee), ("exit_code", trace["exit_code"]), ("charged_once", True),
    ("cap_exceeded", fee > q["limits"]["outer_s"])):
  r.exact(cost[k], v, "once cost " + k)
 r.exact(debit, {"cost_id": q["call_id"], "mode": q["mode"], "actual_outer_full_return_s": fee,
     "actual_returned_exit_code": trace["exit_code"], "descriptor": cost_spec}, "entire debit")
 integer(trace['exit_code'], 'genuine exit code')
 integer(outer['exit_code'], 'actual exit code')
 shell_pair = (trace['exit_code'] == 1 and outer['exit_code'] == 124 and outer['worker_exit_code'] == 124
    and outer.get('timeout') is True and (outer.get('first_error') or {}).get('type') == 'TimeoutError'
    and outer['all_owned_ended'] is True and outer['reservation_released'] is True
    and outer['stdout_eof'] is True and outer['stderr_eof'] is True and not outer['unknown_owned_identities']
    and trace.get('exec_args', {}).get('cmd', '').endswith('& .venv/Scripts/python.exe -m tools.banc_m3_startup_ray_supervisor --request ' + request_spec['path']))
 r.need(r.d19_current_exit(globals(),trace,outer,request_spec,root), 'D19 exact current request-bound fee policy')
 base_keys = {'schema', 'cost_id', 'mode', 'request', 'source_freeze', 'trace', 'outer', 'actual_outer_full_return_s',
    'exit_code', 'charged_once', 'all_owned_ended', 'reservation_released', 'cap_exceeded', 'worker_started',
    'creation_attempted', 'worker_creation_state', 'resume_attempted', 'resume_outcome'}
 if q['mode'] == 'original':
  base_keys |= {'supervisor_request', 'role', 'grant', 'inputs', 'roster', 'contract', 'body_retry', 'local_debit_offset', 'resource_increment_s'}
 if shell_pair:
  base_keys |= {'native_supervisor_exit_code', 'actual_collector_shell_exit_code', 'shell_deadline_status_pair_observed'}
 keys(cost, base_keys, 'entire exact K cost layout')
 r.exact(cost['schema'], 'flylab.m3.direct-science.full-return.v1' if q['mode'] == 'generated' else 'flylab.m3.contact-learning.full-return.v1', 'exact mode cost schema')
 r.exact(outer['schema'], 'flylab.m3.direct-science.outer.v1', 'actual supervisor outer schema')
 fee_values(cost, q, request_spec, trace, outer, root)
 for field in ('all_owned_ended', 'reservation_released', 'worker_started', 'creation_attempted', 'worker_creation_state', 'resume_attempted', 'resume_outcome'):
  r.exact(cost[field], outer[field], 'cost actual creation/closure ' + field)
 closed = closure(outer)
 return q, cost, outer, closed


def fee_values(cost, q, request_spec, trace, outer, root):
 if q['mode'] == 'original':
  science = data(q['science_request'], root)
  r.exact(cost['request'], q['science_request'], 'same immutable admitted science request')
  r.exact(cost['supervisor_request'], request_spec, 'same immutable admitted supervision request')
  for field in ('role', 'grant', 'inputs', 'roster', 'contract', 'body_retry', 'local_debit_offset', 'resource_increment_s'):
   r.exact(cost[field], science[field], 'cost admitted science ' + field)
 if outer['exit_code'] != trace['exit_code']:
  for field, value in (('native_supervisor_exit_code', outer['exit_code']),
       ('actual_collector_shell_exit_code', trace['exit_code']), ('shell_deadline_status_pair_observed', True)):
   r.exact(cost[field], value, 'actual observed shell pair ' + field)


def accounting_hold(ledger, mode, call):
 key = ('direct_science_' if mode == 'generated' else 'startup_ray_original_') + 'blocking_holds'
 hold = ledger[key].get(call)
 if type(hold) is dict:
  hold['accounting_hold'] = HOLD
 else:
  ledger[key][call] = HOLD


def expected_ownership(expected,q,request_spec,live,root,*,charged=False):
 return r.d19_expected_ownership(globals(),expected,q,request_spec,live,root,charged=charged)


def validate(ledger, root):
 return r.d19_validate(globals(),ledger,root)
 """Return legal accounting state, permitting incurred negative remaining."""
 event, baseline, policy, grant = recovery(ledger[KEY], root)
 expected = copy.deepcopy(baseline)
 expected[KEY] = ledger[KEY]
 expected["direct_science_blocking_holds"] = {}
 source = event["source_freeze"]
 tail = ledger["debits"][322:]
 r.need(type(tail) is list and len(tail) <= len(SEQUENCE), "finite seven-call tail")
 caps = dict(OLD_CAPS)
 failed = False
 for index, debit in enumerate(tail):
  r.exact(debit["cost_id"], SEQUENCE[index], "next literal debit")
  r.need(not failed, "paid failure does not admit successors")
  q, cost, _outer, closed = fee_proof(debit["descriptor"], debit, source, root)
  r.need(closed, "UNCOMMITTED_OWNERSHIP_UNKNOWN: no fee/debit/totals without known closure")
  reserved = q["limits"]["outer_s"]
  expected["debits"].append(debit)
  expected["direct_science_calls"][q["call_id"]] = {"request": cost["request"] if q["mode"] == "generated" else cost["supervisor_request"],
             "state": "charged", "mode": q["mode"], "reserved_s": reserved}
  failed = cost["exit_code"] != 0 or cost["cap_exceeded"] or not closed
  if not closed:
   expected_ownership(expected, q, expected["direct_science_calls"][q["call_id"]]["request"], ledger, root, charged=True)
   owner_key = ('direct_science_' if q['mode'] == 'generated' else 'startup_ray_original_') + 'workers'
   for field in ('owner_pid', 'owner_creation_ticks', 'worker_pid', 'worker_creation_ticks'):
    r.exact(_outer[field], expected[owner_key][q['call_id']][field], 'unclosed evidence retains admitted identity')
  if closed and cost['exit_code'] != 0:
   expected[('direct_science_' if q['mode'] == 'generated' else 'startup_ray_original_') + 'blocking_holds'][q['call_id']] = FAILURE_HOLD
  if cost["cap_exceeded"]:
   accounting_hold(expected, q['mode'], q['call_id'])
 if ALLOCATION_KEY in ledger:
  r.need(len(tail) >= 1, "allocation after actual bootstrap")
  metadata_pin(ledger[ALLOCATION_KEY], ALLOCATION_PATH)
  allocation = data(ledger[ALLOCATION_KEY], root)
  keys(allocation, ("schema", "epoch", "before_ledger", "recovery", "source_freeze", "bootstrap", "bootstrap_junit",
      "old_caps", "new_caps", "delta_s", "combined_global_s", "new_resource_grant_s", "refund_s"), "allocation event")
  r.exact(allocation["schema"], "flylab.m3.checkpoint-allocation.v1", "allocation schema")
  for k, v in (("epoch", EPOCH), ("recovery", ledger[KEY]), ("source_freeze", source), ("old_caps", OLD_CAPS),
     ("new_caps", CAPS), ("delta_s", {"generated_global": 0, "generated_local": 0, "original_global": 0}),
     ("combined_global_s", 19900), ("new_resource_grant_s", 0), ("refund_s", 0), ("bootstrap", tail[0]["descriptor"])):
   r.exact(allocation[k], v, "exact allocation " + k)
  before = data(allocation["before_ledger"], root, 524288)
  r.need(ALLOCATION_KEY not in before and len(before["debits"]) == 323, "fresh closed bootstrap before allocation")
  state = validate(before, root)
  eligibility(before, state)
  success(before["debits"][-1], source, root, "bootstrap", allocation["bootstrap_junit"])
  expected[ALLOCATION_KEY] = ledger[ALLOCATION_KEY]
  caps = dict(CAPS)
  for k, v in caps.items():
   expected[r.CAP_FIELDS[k]] = v
 else:
  r.need(len(tail) <= 1, "qualification cannot use old caps")
 if QUALIFICATION_KEY in ledger:
  r.need(ALLOCATION_KEY in ledger and len(tail) >= 5, "separate four actual costs before registration")
  metadata_pin(ledger[QUALIFICATION_KEY], QUALIFICATION_PATH)
  qualification(data(ledger[QUALIFICATION_KEY], root), ledger, source, root)
  expected[QUALIFICATION_KEY] = ledger[QUALIFICATION_KEY]
 r.need(len(tail) <= 5 or QUALIFICATION_KEY in ledger, "Original requires registered K qualification")
 expected.update(totals(expected, policy, grant, caps))
 active_calls = set(ledger["direct_science_calls"]) - set(expected["direct_science_calls"])
 r.need(len(active_calls) <= 1, "at most next admitted call")
 phase = "charged"
 if active_calls:
  r.need(not failed and len(tail) < len(SEQUENCE), "no call after failure/finite end")
  call = SEQUENCE[len(tail)]
  r.exact(active_calls, {call}, "no old/replay/alias admission")
  admitted = ledger["direct_science_calls"][call]
  keys(admitted, ("request", "state", "mode", "reserved_s"), "whole admission")
  q = data(admitted["request"], root)
  identity(q, admitted["request"], source, root)
  r.exact(admitted, {"request": admitted["request"], "state": "admitted", "mode": q["mode"], "reserved_s": q["limits"]["outer_s"]}, "exact admitted identity")
  r.need(len(tail) != 1 or ALLOCATION_KEY in ledger, "no lint before allocation")
  r.need(len(tail) != 5 or QUALIFICATION_KEY in ledger, "no body before K qualification")
  if len(tail) == 6:
   valid_body(tail[5], source, data(q['science_request'], root), root)
  expected["direct_science_calls"][call] = admitted
  phase = expected_ownership(expected, q, admitted["request"], ledger, root)
 if tail:
  last = tail[-1]
  breached = ((last["mode"] == "generated" and (expected["direct_science_generated_used_s"] > caps["generated_local"]
     or expected["generated_shared_remaining_s"] < 0)) or
     (last["mode"] == "original" and (expected["original_remaining_s"] < 0
     or expected["original_stage_used_s"] > 6300 or expected["startup_ray_original_local_used_s"] > 6000)))
  if breached:
   accounting_hold(expected, last['mode'], last['cost_id'])
 r.exact(ledger, expected, "whole reconstructed K state; all immutable300 fields/calls retained")
 return {"caps": caps, "completed": len(tail), "phase": phase, "failed": failed, "source": source, "epoch": EPOCH}


def eligibility(ledger, state):
 r.need(state["phase"] == "charged" and not state["failed"], "known charged successful predecessor")
 for key in r.OWNER_KEYS:
  r.exact(ledger[key], {}, "closed eligible ownership " + key)
 r.need(ledger["generated_shared_remaining_s"] >= 0 and ledger["original_remaining_s"] >= 0
  and ledger["direct_science_generated_used_s"] <= state["caps"]["generated_local"], "remaining execution capacity")


def successful_junit(spec, rows, root):
 document = ET.fromstring(r.raw_pin(spec, root, 262144))
 suites = document.findall('testsuite')
 r.need(document.tag == 'testsuites' and len(suites) == 1, 'one whole actual suite')
 suite = suites[0]
 r.need(all(suite.get(k) == '0' for k in ('failures', 'errors', 'skipped'))
  and suite.get('tests') == str(len(rows)), 'whole passed suite without subset/skip/error')
 cases = suite.findall('testcase')
 actual = [(case.get('classname'), case.get('name')) for case in cases]
 r.exact(actual, [(row['classname'], row['name']) for row in rows], 'every exact full JUnit identity/parameter')
 r.need(len(set(actual)) == len(actual) and all(not list(case) for case in cases), 'no duplicate/failure/skip/error children')

def success(debit, source, root, kind, junit_spec=None):
 q, cost, _outer, closed = fee_proof(debit["descriptor"], debit, source, root)
 r.need(closed and cost["exit_code"] == 0 and cost["cap_exceeded"] is False and _outer["worker_started"] is True and _outer["worker_creation_state"] == "RESUMED", "actual successful resumed paid known closed role")
 if kind != "lint":
  freeze = data(source, root)
  metadata_pin(junit_spec, BASE + "/parent-contact-learning-s28-" + ("r64-bootstrap" if kind == "bootstrap" else "r64-"+kind.replace("_", "-")) + "-junit.xml")
  successful_junit(junit_spec, roster(freeze, root)[kind], root)
 return q


def qualification(value, ledger, source, root):
 n=data(CURRENT_CONTRACT,root)['baseline_debits'];r.need(type(n) is int and n>0,'Exact current baseline')
 keys(value, ("schema", "epoch", "source_freeze", "actual_costs", "tests"), "K qualification")
 r.exact(value["schema"], "flylab.m3.direct-science.qualification.v8", "distinct K qualification")
 r.exact(value["epoch"], EPOCH, "same K epoch")
 r.exact(value["source_freeze"], source, "one tested Source")
 costs = ledger["debits"][n+1:n+5]
 r.exact(value["actual_costs"], [d["descriptor"] for d in costs], "four separately paid K qualification calls")
 r.need(len(costs) == 4 and len(value["tests"]) == 3, "four full costs/three full JUnits")
 for index, debit in enumerate(costs):
  r.exact(debit["cost_id"], QUAL_CALLS[index], "fixed role order")
  success(debit, source, root, ("lint", "math", "gate_a", "gate_b")[index], None if index == 0 else value["tests"][index-1])


def bounded_source(spec, root, cap):
 """The selected Source cap bounds each actual read, including installed text."""
 keys(spec, ('path', 'bytes', 'sha256'), 'whole Source descriptor')
 integer(spec['bytes'], 'Source bytes')
 r.need(0 <= spec['bytes'] <= cap, 'selected actual Source cap')
 path = Path(spec['path'])
 if not path.is_absolute():
  r.need(path.suffix in ('.py', '.ps1', '.js', '.json', '.txt', '.ini', '.md', '.xml'), 'relative text Source only')
  return r.raw_pin(spec, root, cap, allow_empty=True)
 old = data(L_FREEZE, root)['sources']
 r.need(spec['path'] in old, 'installed roots come only from exact inherited J vector')
 r.exact(spec, old[spec['path']], 'fixed installed Source path and pin')
 r.need(path.suffix in ('.py', '.ps1', '.json', '.txt', '.ini', '.md'), 'installed text extension')
 relative = path.relative_to(path.anchor).as_posix()
 operand = r._text_path(path.anchor, relative)
 before = os.lstat(operand)
 r.need(stat.S_ISREG(before.st_mode) and before.st_size == spec['bytes'], 'actual installed regular Source bytes')
 with operand.open('rb') as stream:
  raw = stream.read(cap + 1)
 raw.decode('utf8', errors='strict')
 with operand.open('rb') as stream:
  again = stream.read(cap + 1)
 r._text_path(path.anchor, relative)
 r.need(len(raw) <= cap and len(raw) == spec['bytes'] and raw == again
  and r._text_stamp(os.lstat(operand)) == r._text_stamp(before)
  and hashlib.sha256(raw).hexdigest() == spec['sha256'], 'whole capped installed Source readback/stamp/SHA')
 return raw


def provenance(review, root):
 keys(review, ('schema','epoch','sources','replacements','additions','before_source','root_reflection','astra_review','root_reviews','sol_reports'), "complete current reviewed metadata")
 r.exact((review["schema"],review["epoch"]),("flylab.m3.checkpoint-reviewed-source.v1",EPOCH),"current review schema and epoch")
 keys(review["astra_review"], ("markdown", "json"), "finite current review provenance")
 cp = review["sources"][CONTRACT_PATH]
 metadata_pin(cp, CONTRACT_PATH)
 r.exact(cp, {'path': 'verification/banc-natural-gait-m3-20260919/c5.json', 'bytes': 130832, 'sha256': '9398fcc3a61b0e8019b2599c40ca8dd7e0ebaa4e60302b71a74c2dd7a1a32d78'},"whole current compiled contract pin")
 contract = data(cp,root)
 r.exact((contract["schema"],contract["epoch"]),('flylab.m3.checkpoint-contract.v1',EPOCH),"current provenance contract schema epoch")
 r.exact(contract["actual320"]["adoption_before_Source"],CONTINUATION_BEFORE_SOURCE,"compiled adoption predecessor")
 p = contract["provenance"]
 keys(p, ('root_design', 'astra_json', 'root_cause', 'root_reflection', 'astra_markdown', 'before_index', 'source_before', 'immutable_inputs', 'root_coverage', 'reader_closure'),"exact ten current provenance keys")
 r.exact((p["astra_markdown"],p["astra_json"],p["root_reflection"],p["root_coverage"]),(review["astra_review"]["markdown"],review["astra_review"]["json"],review["root_reflection"],review["root_reflection"]),"direct descriptors equal review fields")
 for spec,path in ((p["root_design"],"docs/M3_28D18.md"),(p["astra_markdown"],BASE+"/s28-d18-astra/review.md"),(p["astra_json"],BASE+"/s28-d18-astra/review.json"),(p["root_reflection"],"docs/M3_28R18.md")):
  metadata_pin(spec,path)
  r.raw_pin(spec,root,262144)
 contract_freeze = data(r.text_spec(SOURCE_PATH,root,262144),root)
 r.exact((contract_freeze["ROOT_review_cycle_complete"],contract_freeze["qualification_epoch"],contract_freeze["checkpoint_contract"],contract_freeze["sources"]),(True,EPOCH,cp,review["sources"]),"actual accepted Source/review before evidence")
 r.exact(contract_freeze["reviewed_source"],r.text_spec(REVIEW_PATH,root,262144),"actual raw reviewed Source")
 for spec in (*review["root_reviews"],*review["sol_reports"]):
  metadata_pin(spec,spec["path"]);r.raw_pin(spec,root,262144)
 frozen_history(contract,root)
 return tuple(contract["provenance"][k] for k in ("root_design","astra_markdown","astra_json","root_reflection"))

def source_fence(source, root, check=lambda: None):
 return r.d19_source_fence(globals(),source,root,check)
 """Fresh whole current Source: call only after actual SharedLock acquisition."""
 try:
  metadata_pin(source, SOURCE_PATH)
  check()
  f8 = data(source, root)
  r.exact(f8["ROOT_review_cycle_complete"], True, "ROOT adopted K reviewed Source")
  r.exact(f8["qualification_epoch"], EPOCH, "distinct K Source")
  metadata_pin(f8["reviewed_source"], REVIEW_PATH)
  v8 = data(f8["reviewed_source"], root)
  keys(v8, ('schema', 'epoch', 'sources', 'replacements', 'additions', 'before_source', 'root_reflection',
     'astra_review', 'root_reviews', 'sol_reports'), 'complete K reviewed-source metadata')
  r.exact(v8['schema'], 'flylab.m3.checkpoint-reviewed-source.v1', 'distinct K reviewed Source')
  r.exact(v8['epoch'], EPOCH, 'K Source review epoch')
  r.exact(f8["checkpoint_contract"],f8["sources"][CONTRACT_PATH],"Source contract before provenance")
  r.exact(f8["checkpoint_contract"],v8["sources"][CONTRACT_PATH],"Source review contract agreement")
  provenance(v8, root)
  r.exact(v8['before_source'], CONTINUATION_BEFORE_SOURCE, 'exact continuation Source archive')
  o8 = data(CONTINUATION_BEFORE_SOURCE, root)
  old = o8['sources']
  e57 = copy.deepcopy(o8)
  e57.update(qualification_epoch=EPOCH, reviewed_source=f8['reviewed_source'],
       body_retry=f8['sources'][BODY_PATH], sources=f8['sources'],
       checkpoint_contract=f8['sources'][CONTRACT_PATH])
  r.exact(f8, e57, 'whole inherited freeze with only finite K changes')
  r.need(len(old) == 871 and set(f8["sources"]) == set(old) | set(ADDITIONS), "entire inherited871 plus zero flat additions")
  r.exact(v8["sources"], f8["sources"], "same reviewed vector")
  r.exact(v8["replacements"], list(REPLACEMENTS), "all and only declared replacements")
  r.exact(v8["additions"], list(ADDITIONS), "all and only declared new dependencies")
  for n8, s8 in old.items():
   if n8 not in REPLACEMENTS:
    r.exact(f8["sources"][n8], s8, "whole unchanged Source pin " + n8)
  for n8, s8 in f8["sources"].items():
   check()
   r.exact(s8['path'], n8, 'same whole Source namespace')
   c8 = {REPLACEMENTS[0]: 131072, REPLACEMENTS[1]: 131072,
                  BASE + '/parent-direct-science-account.ps1': 393216}.get(n8)
   evidence = c8 is None and n8 in old and old[n8]['bytes'] > 262144
   cap = c8 if c8 is not None else 524288 if evidence else 262144
   r.need(s8['bytes'] <= cap, 'bounded Source/fixture or exact inherited evidence')
   if evidence:
    r.exact(s8, old[n8], 'only fixed J inherited evidence may use524288')
    r.raw_pin(s8, root, cap)
   else:
    bounded_source(s8, root, cap)
  check()
  roster(f8, root)
  b57 = f8["sources"][BODY_PATH]
  r.exact(f8["body_retry"], b57, "static body in acyclic Source vector")
  return f8

 except Exception:
  _HISTORY_PROOF.clear()
  raise

def dispatch(ledger, q, request_spec, root, *, phase):
 n=data(CURRENT_CONTRACT,root)['baseline_debits'];r.need(type(n) is int and n>0,'Exact current baseline')
 state = validate(ledger, root)
 r.need(phase in ("closed", "admitted"), "finite dispatch phase")
 source_fence(state["source"],root)
 if phase == "closed":
  eligibility(ledger, state)
  r.exact(q["call_id"], SEQUENCE[state["completed"]], "next literal call")
 else:
  r.exact(state["phase"], "admitted", "exact root admission before worker")
  r.exact(ledger["direct_science_calls"][q["call_id"]]["request"], request_spec, "same root-admitted K request")
 identity(q, request_spec, state["source"], root)
 if q["mode"] == "generated":
  envelope = (9755, 5755, 5715, 5595, 5425)[SEQUENCE.index(q["call_id"])]
  r.need(ledger["direct_science_generated_used_s"] + envelope <= state["caps"]["generated_local"]
   and ledger["generated_shared_remaining_s"] >= envelope, "complete paired Generated envelope")
 else:
  r.need(QUALIFICATION_KEY in ledger, "registered whole K qualification before Original")
  original_envelope = 600 if q['call_id'] == CALLS[0] else q['limits']['outer_s'] + 180
  for field, cap in (("original_cumulative_used_s", CAPS["original_global"]), ("original_stage_used_s", 6300), ("startup_ray_original_local_used_s", 6000)):
   r.need(ledger[field] + original_envelope <= cap, 'whole body/audit/close600 or conditional audit plus close180')
 for index in range(1, min(state['completed'], 5)):
  kind = ('lint', 'math', 'gate_a', 'gate_b')[index-1]
  junit = None if kind == 'lint' else r.text_spec(BASE + '/parent-contact-learning-s28-r64-' + kind.replace('_', '-') + '-junit.xml', root)
  success(ledger['debits'][n+index], state['source'], root, kind, junit)
 return state


def body_binding(q, contract, read):
 r.need(q["call_id"] in CALLS and q["role"] == ("contact_candidate_body", "contact_body_audit")[CALLS.index(q["call_id"])], "sole K body/audit mapping")
 r.exact(q["source_freeze"]["path"], SOURCE_PATH, "K execution Source")
 freeze = read(q["source_freeze"])
 r.exact(q["body_retry"], freeze["sources"][BODY_PATH], "immutable K body pin")
 prior = read(S15_PINS["before"])["sources"][BODY_PATH]
 r.exact(prior, {'path':BODY_PATH,'bytes':20516,'sha256':"ac3786738fad5fe69e4934c0382c1a4508444402d387e1f113017f7eaec92276"}, "exact executed beforeB5")
 value = read(q["body_retry"])
 expected = copy.deepcopy(read(dict(prior,path=BASE+"/s28-before/"+BODY_PATH)))
 expected.update(calls=list(CALLS), qualification_epoch=EPOCH, source_review_path=REVIEW_PATH,
     source_freeze_path=SOURCE_PATH, checkpoint_contract_path=CONTRACT_PATH,
     checkpoint_recovery_path=RECOVERY_PATH, checkpoint_allocation_path=ALLOCATION_PATH)
 r.exact(value, expected, "whole beforeB5 with only seven routing changes")
 for key in ("contract", "grant", "inputs", "roster"):
  r.exact(q[key], value[key], "same candidate metadata " + key)
 r.exact(q["limits"], contract["limits"][q["role"]], "unchanged body/audit scientific envelopes")
 return value


def original_metadata(science, outer, root):
 """K common Original guarantees from immutable metadata, no J history/Source reads."""
 def read(pin):
  return data(pin, root)
 old = read(J_BODY)
 for key in ('contract', 'grant', 'inputs', 'roster'):
  r.exact(science[key], old[key], 'fixed immutable science pin before reading ' + key)
 metadata_pin(science['source_freeze'], SOURCE_PATH)
 metadata_pin(science['body_retry'], BODY_PATH)
 metadata_pin(science['qualification'], QUALIFICATION_PATH)
 contract = read(science['contract'])
 r.exact(contract['schema'], 'flylab.m3.contact-learning.stage.v1', 'fixed science contract schema')
 r.exact(contract['roles'], ['contact_direction', 'contact_candidate_audit', 'contact_candidate_body', 'contact_body_audit'], 'whole fixed science roles')
 body = body_binding(science, contract, read)
 for key, value in (('schema', 'flylab.m3.contact-learning.request.v1'), ('mode', 'original'),
     ('ledger_path', BASE + '/parent-e358-execution-ledger.json'), ('shared_original_lock_path', r.SHARED_LOCK),
     ('local_debit_offset', 165), ('resource_increment_s', 0),
     ('output', BASE + '/parent-' + science['call_id'] + '/result')):
  r.exact(science[key], value, 'finite K science ' + key)
 r.exact(outer['mode'], 'original', 'same Original outer')
 r.exact(outer['output'], science['output'] + '-supervision', 'whole science/outer output relation')
 for key in ('call_id', 'role', 'grant', 'source_freeze', 'ledger_path', 'shared_original_lock_path'):
  r.exact(outer[key], science[key], 'same K Original ' + key)
 for key, value in outer['limits'].items():
  r.exact(science['limits'][key], value, 'whole exact outer science cap ' + key)
 r.exact(science['limits'], contract['limits'][science['role']], 'whole science limits including internal')
 roster, grant = read(science['roster']), read(science['grant'])
 r.exact(science['grant'], r.J_PIN, 'fixed existing grant')
 for key, value in (('schema', 'flylab.m3.startup-ray.original-grant.v1'), ('grant_id', 'startup-ray-original-20261007-j'),
     ('reset', False), ('resource_increment_s', 0), ('local_debit_offset', 165), ('worker_max', 1), ('automatic_restarts', 0)):
  r.exact(grant[key], value, 'existing finite grant ' + key)
 r.exact(roster['schema'], 'flylab.m3.contact-conditioned-learning.input-roster.v1', 'fixed input roster schema')
 r.exact(contract['roster'], science['roster'], 'fixed contract roster')
 r.exact(roster['grant'], science['grant'], 'same immutable roster grant')
 r.exact(roster['original_inputs'], science['inputs'], 'same immutable input binder')
 r.exact(grant['inputs'], science['inputs'], 'same grant input lineage')
 inputs = read(science['inputs'])
 r.need(type(inputs) is dict, 'intact metadata input binder, no payload reads')
 r.exact(science['historical_manifest'], read(roster['previous_geometry_roster'])['historical_manifest'], 'fixed historical manifest lineage')
 read(science['historical_manifest'])
 freeze = read(science['source_freeze'])
 r.exact(freeze['schema'], 'flylab.m3.contact-learning.source-freeze.v1', 'same K freeze schema')
 r.exact(freeze['ROOT_review_cycle_complete'], True, 'actually reviewed immutable K freeze')
 r.exact(freeze['qualification_epoch'], EPOCH, 'finite K epoch')
 r.exact(freeze['sources'][science['contract']['path']], science['contract'], 'same frozen contract')
 r.exact(freeze['human_authorization'], grant['human_authorization'], 'unchanged human scope')
 qualified = read(science['qualification'])
 r.exact(qualified['schema'], 'flylab.m3.direct-science.qualification.v8', 'same K qualification schema')
 r.exact(qualified['epoch'], EPOCH, 'same K qualification epoch')
 r.exact(qualified['source_freeze'], science['source_freeze'], 'same admitted qualified freeze')
 before = data(science['before_ledger'], root, 524288)
 r.exact(before[QUALIFICATION_KEY], science['qualification'], 'before carries actual K qualification')
 r.exact(data(before[KEY], root)['source_freeze'], science['source_freeze'], 'same recovered immutable Source lineage')
 r.need(ALLOCATION_KEY in before, 'existing conserved allocation before Original')
 return body


def candidate(q, report, read):
 body = body_binding(q, read(q["contract"]), read)
 historic = read(body["historical_manifests"][1])
 audit = read(historic["artifacts"]["report"])
 r.exact(audit["candidate"], body["candidate"], "entire audited candidate-k")
 r.exact(report["candidate"], body["candidate"], "same whole candidate-k")
 r.exact(historic["artifacts"]["candidate_arrays"], body["candidate_arrays"], "literal array descriptor only")
 manifest = read(q["artifacts"]["manifest"])
 r.exact(manifest["source_freeze"], body["source_k"] if q["role"] == "contact_candidate_body" else q["source_freeze"], "immediate predecessor Source")
 r.exact(manifest["role"], "contact_candidate_audit" if q["role"] == "contact_candidate_body" else "contact_candidate_body", "immediate predecessor role")
 r.exact(manifest["run_valid"], True, "technically valid predecessor")
 r.exact(q["artifacts"], dict(manifest["artifacts"], manifest=q["artifacts"]["manifest"]), "all predecessor artifact pins")


def valid_body(debit, source, q, root):
 request, cost, _outer, closed = fee_proof(debit["descriptor"], debit, source, root)
 r.need(closed and cost["exit_code"] == 0 and not cost["cap_exceeded"] and request["call_id"] == CALLS[0], "audit only valid genuine body")
 manifest = data(q["artifacts"]["manifest"], root)
 r.exact(manifest["request"], cost["request"], "same actual body manifest")
 r.exact(manifest["run_valid"], True, "actual body run_valid")
 report = data(manifest["artifacts"]["report"], root)
 r.need(report["run_valid"] is True and report["first_error"] is None and not report["secondary_errors"], "actual report valid")
 candidate(q, report, lambda pin: data(pin, root))


def cold_contact(admission, label):
 """Preserve the existing deadline/Original owner/Job/fence without old history recursion."""
 from tools import banc_m3_startup_ray_supervisor as supervisor
 from tools import verify_c_banc_m3_startup_ray as ray
 q, root = admission.request, admission.root
 admission.deadline()
 for spec, expected, cap in ((admission.request_spec, q, 262144), (q["before_ledger"], admission.before, 524288),
        (q["contract"], admission.contract, 262144), (q["grant"], admission.grant, 262144),
        (q["inputs"], admission.inputs, 262144), (q["roster"], admission.roster, 262144)):
  r.exact(admission.read(spec, cap), expected, "fresh exact K metadata")
 body_binding(q, admission.contract, admission.read)
 source_fence(q["source_freeze"], root, admission.deadline)
 before_state = validate(admission.before, root)
 eligibility(admission.before, before_state)
 live = admission.read(ray.spec_for(q["ledger_path"], root, cap=524288), 524288)
 state = validate(live, root)
 r.exact(live["debits"], admission.before["debits"], "all original/generated actual debit history")
 r.exact(live[QUALIFICATION_KEY], q["qualification"], "registered K qualification")
 admitted = live["direct_science_calls"][q["call_id"]]
 outer = admission.read(admitted["request"])
 r.exact(outer["science_request"], admission.request_spec, "same science worker request")
 if admission.stage == "pre_spawn":
  r.exact(state["phase"], "admitted", "no K worker yet")
  r.need(not ray.path(r.SHARED_LOCK, root).exists() and not admission.output.exists(), "fresh physical exclusion/output")
 else:
  r.exact(state["phase"], "begun", "actual K Original begun owner")
  worker = live["startup_ray_original_workers"][q["call_id"]]
  r.exact(worker["worker_pid"], os.getpid(), "this actual worker")
  for name in ("owner", "worker"):
   r.exact(supervisor.creation_ticks(worker[name + "_pid"]), worker[name + "_creation_ticks"], "live creation identity")
  token = admission.read(worker["lock_token"])
  r.exact(ray.json_load(os.environ["FLYLAB_DIRECT_SCIENCE_OWNER"].encode("utf8")), token, "actual same ENTRY token")
  r.exact(token["entry_counter_s"], admission.entry, "no reset ENTRY")
  r.exact(token["request"], admitted["request"], "same owner request")
  r.need(supervisor.in_named_job(token["job_name"]), "same actual owned Job")
 r.need(label in ("initial", "pre_array", "publication", "pre_construct"), "unchanged cold boundary labels")
 admission.deadline()
 return True


def initial_refusal(root, request_spec, q, ledger_spec, raw, error, *, entry):
 """Independent pre-create reporting; closure is not canonical budget release."""
 from tools import banc_m3_startup_ray_supervisor as supervisor
 def need(test,label):
  if not test:raise ValueError(label)
 def layout(a,b):
  if type(a) is dict and type(b) is dict:
   need(set(a)==set(b),"layout")
   for k in a:layout(a[k],b[k])
  else:whole_exact(a,b)
 def path(name):
  need(type(name) is str and not Path(name).is_absolute() and "\\" not in name and ":" not in name,"path")
  parts=name.split("/");need(all(x and x not in (".","..") and not x.endswith(("."," ")) for x in parts),"parts")
  p=Path(root).absolute();need(p.is_dir() and not p.is_symlink() and not (getattr(p.stat(),"st_file_attributes",0)&1024),"root")
  for x in parts:
   p=p/x
   if p.exists():need(not p.is_symlink() and not (getattr(p.stat(),"st_file_attributes",0)&1024),"reparse")
  return p
 def read(spec,cap):
  need(type(spec) is dict and set(spec)=={"bytes","path","sha256"} and type(spec["bytes"]) is int and 0<spec["bytes"]<=cap and type(spec["sha256"]) is str and len(spec["sha256"])==64,"pin")
  p=path(spec["path"]);need(p.is_file() and p.stat().st_size==spec["bytes"],"file")
  with p.open("rb") as f:value=f.read(cap+1)
  need(len(value)==spec["bytes"] and hashlib.sha256(value).hexdigest()==spec["sha256"],"raw")
  return value
 def pairs(rows):
  d={}
  for k,v in rows:need(k not in d,"dupe");d[k]=v
  return d
 def decode(value):return json.loads(value,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError("finite")))
 def unchanged():
  need(not path(q["shared_original_lock_path"]).exists(),"lock")
  whole_exact(read(ledger_spec,524288),raw);whole_exact(read(request_spec,262144),request_raw)
 try:
  need(type(raw) is bytes and type(entry) in (float,int) and math.isfinite(entry) and entry>0,"entry")
  whole_exact(read(ledger_spec,524288),raw)
  request_raw=read(request_spec,262144);whole_exact(decode(request_raw),q)
  live=decode(raw);call=q["call_id"];need(call in SEQUENCE,"call")
  i=SEQUENCE.index(call);mode="generated" if i<5 else "original"
  role=("execution","lint","numerical","execution","execution","contact_candidate_body","contact_body_audit")[i]
  limits=dict(zip(("outer_s","child_s","cleanup_s"),CAP_TRIPLES[i],strict=True));limits.update(host_bytes=2147483648 if i<5 else 8589934592,output_cap_bytes=8388608 if i<5 else 26214400)
  want={"schema","call_id","mode","role","source_freeze","limits","output","ledger_path","shared_original_lock_path"}|({"generated_behavior","generated_argv"} if i<5 else {"grant","science_request"})
  need(set(q)==want,"keys")
  for k,v in (("schema","flylab.m3.direct-science.supervision.v1"),("mode",mode),("role",role),("limits",limits),("output",BASE+"/parent-"+call+("" if i<5 else "/result-supervision")),("ledger_path",BASE+"/parent-e358-execution-ledger.json"),("shared_original_lock_path",BASE+"/parent-e358-original-active.lock")):layout(q[k],v)
  need(ledger_spec["path"]==q["ledger_path"],"canonical")
  if i==0:need(request_spec["path"]==BASE+"/s28-d19bz-full4-request.json","bootstrap")
  need(type(q["source_freeze"]) is dict and q["source_freeze"]["path"]==SOURCE_PATH,"Source")
  layout(live["direct_science_calls"][call],{'request':request_spec,'state':"admitted",'mode':mode,'reserved_s':limits["outer_s"]})
  need([x for x,v in live["direct_science_calls"].items() if v["state"]!="charged"]==[call],"admitted")
  for k in r.OWNER_KEYS:layout(live[k],{call:{'request':request_spec,'reserved_s':limits["outer_s"]}} if k=="direct_science_reservations" and i<5 else {})
  whole_exact([live["original_jobs_running"],live["original_reserved_s"]],[0,0])
  if i<5:
   whole_exact(q["generated_behavior"],"command");need(type(q["generated_argv"]) is list and all(type(x) is str and len(x)<=1024 for x in q["generated_argv"]),"argv")
   if i==0:whole_exact(q["generated_argv"],["-m","pytest",*[TEST_MODULE+"::"+n for n in BOOTSTRAP_NAMES],"--tb=short","-q","--junitxml="+BASE+"/parent-contact-learning-s28-r64-bootstrap-junit.xml","--maxfail=1"])
  else:layout(q["grant"],r.J_PIN)
  unchanged();output=path(q["output"]);need(not output.exists(),"output")
  pid=os.getpid();ticks=supervisor.creation_ticks(pid);need(type(pid) is int and pid>0 and type(ticks) is int and ticks>0,"actor")
  unchanged();output.mkdir(parents=True,exist_ok=False);unchanged()
  outer={"schema":"flylab.m3.direct-science.outer.v1","request":request_spec,"exit_code":1,"worker_exit_code":None,"worker_pid":None,"worker_creation_ticks":None,"owner_pid":pid,"owner_creation_ticks":ticks,"all_owned_ended":True,"reservation_released":True,"unknown_owned_identities":{},"first_error":{"type":type(error).__name__,"message":str(error)[:8192]},"secondary_errors":[],"timeout":isinstance(error,TimeoutError),"stdout_eof":False,"stderr_eof":False,"stdout":None,"stderr":None,"pre_spawn_refused":True,"job_created":False,"worker_started":False,"creation_attempted":False,"worker_creation_state":"NO_CREATE_ATTEMPT","resume_attempted":False,"resume_outcome":"NOT_ATTEMPTED","checkpoint_recovery":live[KEY],"initial_ledger":ledger_spec,"refusal_origin":"K_INITIAL_GUARD_BEFORE_CREATE","observed_supervisor_interval_s":time.perf_counter()-entry}
  supervisor.durable(output/"outer.json",outer);unchanged();return outer
 except Exception as secondary:
  _LOGGER.exception("Independent refusal uncertainty")
  raise RuntimeError("original "+type(error).__name__+": "+str(error)[:8192]+"; independent refusal uncertainty "+type(secondary).__name__+": "+str(secondary)[:8192]) from error


def d19_opaque_native(k,fixture,tail):
 import base64
 import re
 r=k.r;sf=k.data(fixture.source,fixture.root);raw=r.raw_pin(sf['sources']['tools/banc_m3_checkpoint_harness_runtime_native.ps1'],fixture.root,131072);parts=[]
 for name in ('K-Raw','K-SourceRaw'):
  matches=list(re.finditer(rb'^function '+name.encode()+rb'\(.*?(?=^function )',raw,re.MULTILINE|re.DOTALL));assert len(matches)==1;parts.append(matches[0].group())
 definitions=base64.b64encode(b'\n'.join(parts)).decode()
 return (k.r.D19_OPAQUE_BODY+tail).replace('DEFS',definitions)
