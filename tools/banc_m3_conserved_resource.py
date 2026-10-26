"""Shared physical ledger transaction and conserved cap epoch; metadata only."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import secrets
import stat
import weakref
import xml.etree.ElementTree as ET
from pathlib import Path

_HISTORY_BASE = 'verification/banc-natural-gait-m3-20260919'
BASE = "verification/banc-natural-gait-m3-20260919"
SHARED_LOCK = BASE + "/parent-e358-original-active.lock"
ALLOCATION_KEY = "direct_science_conserved_allocation"
ALLOCATION_PATH = BASE + "/parent-conserved-resource-allocation-20261008-a.json"
E_PATH = BASE + "/parent-contact-body-runtime-20261008-e.json"
E_SOURCE_PATH = BASE + "/parent-contact-learning-source-freeze-20261008-e.json"
E_QUAL_PATH = BASE + "/parent-contact-learning-qualified-20261008-e.json"
E_CALLS = ("contact-learning-20261008-e-body", "contact-learning-20261008-e-body-audit")
EPOCH = "contact-conserved-20261008-e"
E_DESCRIPTOR_PIN = {"path": E_PATH, "bytes": 10332,
                    "sha256": "4259bc545045b6b505291b37ca15e38e1ba8344f314f7292d014d254a06c1849"}
HELPER_SOURCE = "tools/banc_m3_conserved_resource.py"
ROSTER_PIN = {"path": BASE + "/parent-contact-learning-qualification-roster-20261008-e.json", "bytes": 40437,
              "sha256": "be96770622f6fc0584c3f2ff427708f47aa852b7b3a083a3576ff11eb34fcba0"}
HISTORICAL_PIN = {"path": BASE + "/parent-native-count-conformance-closed282-20261008-a.json", "bytes": 453918,
                  "sha256": "405ffe995f1241f2ff3263efc916c9e69897b18c37028da3a9aab82c6d37f661"}
POLICY_PIN = {"path": BASE + "/parent-direct-science-generated-policy.json", "bytes": 1496,
              "sha256": "693b5193d09a83f5de3b48b99c13199fb3eed2a955c044aa1e97a5a8a9546403"}
J_PIN = {"path": BASE + "/parent-startup-ray-original-grant-20261007-j.json", "bytes": 2227,
         "sha256": "7245e218f5c9abae27a849f880f37bc6013df87ae5e514947887183b4a2dba2f"}
HUMAN_PIN = {"path": BASE + "/parent-contact-metadata-current-human-autonomy-20261007-a.json", "bytes": 324,
             "sha256": "476afe58a7d98697e9f6edf94f6ac6d788d5447062d6a980a8ea5e552b5759ad"}
ROOT_PIN = {"path": "docs/BANC_M3_CONSERVED_RESOURCE_REALLOCATION_ROOT_REFLECTED_20261008.md", "bytes": 9222,
            "sha256": "1694f6233a68123aa0f72ce3c5fcc043a1d11f975bd42b008c68be33d77853ed"}
OLD_CAPS = {"generated_global": 1700, "generated_local": 964, "original_global": 16200}
NEW_CAPS = {"generated_global": 2300, "generated_local": 1564, "original_global": 15600}
DELTA = {"generated_global": 600, "generated_local": 600, "original_global": -600}
CAP_FIELDS = {"generated_global": "generated_shared_cumulative_limit_s",
              "generated_local": "direct_science_generated_limit_s", "original_global": "original_cumulative_limit_s"}
TRANSFER_KEYS = ("direct_science_generated_transfer", "direct_science_generated_additional_transfer",
                 "direct_science_generated_final_transfer", "direct_science_generated_continuation_transfer",
                 "direct_science_generated_runtime_transfer", "direct_science_generated_runtime_residual_transfer")
OWNER_KEYS = ("e359_reservations", "e360_reservations", "e360_pending_actuals", "e360_blocking_holds",
              "direct_science_workers", "direct_science_reservations", "direct_science_pending_actuals", "direct_science_blocking_holds",
              "startup_ray_original_workers", "startup_ray_original_reservations", "startup_ray_original_pending_actuals", "startup_ray_original_blocking_holds")
TOTAL_KEYS = ("direct_science_generated_used_s", "generated_stage_used_s", "generated_shared_cumulative_used_s", "generated_shared_remaining_s",
              "original_cumulative_used_s", "original_stage_used_s", "startup_ray_original_local_used_s", "original_remaining_s")
TAIL_KEYS = set(OWNER_KEYS + TOTAL_KEYS + ("debits", "direct_science_calls", "original_jobs_running", "original_reserved_s", "direct_science_qualification"))
QUAL_CALLS = tuple("direct-science-20261008-contact-e-" + role for role in ("lint", "math", "gate-a", "gate-b"))
QUAL_CAPS = ((15, 10, 5), (25, 20, 5), (120, 110, 10), (120, 110, 10))
BOOTSTRAP_CALL = "direct-science-20261008-conserved-bootstrap"
BOOTSTRAP_NODE = "tests/test_c_banc_contact_runtime_journal.py::test_gate_conserved_bootstrap_production_transaction_and_native_parity"
# Permit the first bounded full experiment. Observed fit remains UNKNOWN;
# bootstrap_proof still requires actual success before any allocation.
BOOTSTRAP_DISPATCHABLE = True
BOOTSTRAP_SOURCES = (HELPER_SOURCE, "tools/banc_m3_conserved_resource_native_core.ps1",
                     "tools/banc_m3_conserved_resource_native_epoch.ps1", BASE + "/parent-direct-science-account.ps1",
                     "tools/banc_m3_startup_ray_supervisor.py", "tests/test_c_banc_contact_runtime_journal.py")
BOOTSTRAP_SOURCE_CAP = 262144

# Explicit administrative route; successful v1 bootstrap checks remain separate.
F_SCHEMA = "flylab.m3.conserved-resource-prequalification.v2"
F_EVENT_PATH = BASE + "/parent-conserved-resource-prequalification-20261008-f-fixed.json"
F_REVIEW_PATH = BASE + "/parent-conserved-resource-prequalification-reviewed-source-20261008-f-fixed.json"
F_SOURCE_PATH = BASE + "/parent-contact-learning-source-freeze-20261008-f-fixed.json"
F_QUAL_PATH = BASE + "/parent-contact-learning-qualified-20261008-f-fixed.json"
F_PATH = BASE + "/parent-contact-body-runtime-20261008-f-fixed.json"
F_EPOCH = "contact-prequalification-20261008-f"
F_QUAL_CALLS = tuple("direct-science-20261008-contact-f-" + role for role in ("lint", "math", "gate-a", "gate-b"))
F_CALLS = ("contact-learning-20261008-f-body", "contact-learning-20261008-f-body-audit")
F_DESCRIPTOR_PIN = {"path": F_PATH, "bytes": 11151, "sha256": "43c59780c88db868befa19c132087302007673c1c2cecae02d7526bf94cbb537"}
F_ORIGINAL_DESCRIPTOR_PIN = {"path": BASE + "/parent-contact-body-runtime-20261008-f.json", "bytes": 11133,
                             "sha256": "3266e2fda20702fc0a182269cb8edb8aaa3e08894ddb844de9cb3676ca269ac7"}
F_ORIGINAL_REVIEW_PIN = {"path": BASE + "/parent-conserved-resource-prequalification-reviewed-source-20261008-f.json", "bytes": 205973,
                        "sha256": "7d3d18610e821e512e4618c7048f7ab2e87c01786e4dc80b9d58ba605140e4cc"}
F_ORIGINAL_EVENT_PIN = {"path": BASE + "/parent-conserved-resource-prequalification-20261008-f.json", "bytes": 17051,
                       "sha256": "49617e5236a11b8c367bba1c5ba0dcb0f09051ba02d361139bf744ea87f7e012"}
F_REFUSAL_PIN = {"path": BASE + "/parent-post-bootstrap-resource-route-administrative-allocation-refusal-20261008-f.json", "bytes": 2307,
                 "sha256": "f1435c615c975eb58902f39a1886c30dfd414e031360d37cc176583560f7fcc4"}
F_ADOPTION_PIN = {"path": BASE + "/parent-post-bootstrap-resource-route-ROOT-residual-20261008/ROOT-source-adoption.json", "bytes": 2565,
                  "sha256": "ff6dd02cfcbc59377fdddb51f4efc7a1f44adedf50c7407f8af1eeab5afd0647"}
F_FIXUP_DESIGN_PIN = {"path": "docs/BANC_M3_POST_BOOTSTRAP_RESOURCE_ROUTE_ROOT_FIXUP_REFLECTED_20261008.md", "bytes": 5965,
                      "sha256": "34d4689ee796c50b8dac323a2b833fceceaa53529c6f095c42a30339276f4988"}
F_FIXUP_BEFORE_PIN = {"path": BASE + "/parent-post-bootstrap-root-fixup-ROOT-before-20261008/source-before.json", "bytes": 6778,
                      "sha256": "2123376c2b03589ee964688c34a525a01a75fe057d98ec3479af618ab0fa0fa1"}
F_FIXUP_ASTRA_PINS = {
    "markdown": {"path": BASE + "/parent-post-bootstrap-root-fixup-astra-review-20261008/review.md", "bytes": 11427,
                 "sha256": "e88c28a5f4dbc3eb672ab1c078b2ab65e4ce5e6721e63ffebe01b2dc8c9d5b4a"},
    "json": {"path": BASE + "/parent-post-bootstrap-root-fixup-astra-review-20261008/review.json", "bytes": 22166,
             "sha256": "4dbc6e0c191e2f839cac7d70ab48248e2d6861cd7db18f8c6fc914a97e8910af"},
}
F_FIXUP_REPORT_PATH = "docs/BANC_M3_POST_BOOTSTRAP_RESOURCE_ROUTE_ROOT_FIXUP_20261008.md"
F_FIXUP_SOURCES = (HELPER_SOURCE, "tools/banc_m3_conserved_resource_native_epoch.ps1",
                   BASE + "/parent-direct-science-account.ps1", BASE + "/parent-contact-learning-contract-native.ps1",
                   "tests/test_c_banc_contact_runtime_journal.py")
F_CLOSED_PIN = {"path": BASE + "/parent-conserved-resource-prequalification-closed283-20261008-f.json", "bytes": 454791,
                "sha256": "40c35c8f08f65a036d678db9d562385425068c04e89bbae2367389cca667c7a2"}
F_ROOT_PIN = {"path": "docs/BANC_M3_POST_BOOTSTRAP_RESOURCE_ROUTE_ROOT_REFLECTED_20261008.md", "bytes": 11137,
              "sha256": "1a36744c198707ede56cefde3b393d60c7d3dfef4ddffbd0fb17a49d32c30ba0"}
F_ADDENDUM_PIN = {"path": "docs/BANC_M3_POST_BOOTSTRAP_RESOURCE_ROUTE_ROOT_DEPENDENCY_ADDENDUM_20261008.md", "bytes": 1819,
                  "sha256": "20af9d1e74933c393bea2eaabb8ed13a7ea7cd9713e38187691975072fe226bb"}
F_ASTRA_PINS = {"markdown": {"path": BASE + "/parent-post-bootstrap-resource-route-astra-review-20261008/review.md",
                            "bytes": 11018, "sha256": "82295c54781ddde5150d0c109e49b99ac85cc3d8936e2e8d2b479c4a3ed28da9"},
                "json": {"path": BASE + "/parent-post-bootstrap-resource-route-astra-review-20261008/review.json",
                         "bytes": 12894, "sha256": "5b26e6e749da601c9937f175e9904a9ad6c17af2e873b3a599a9e743dd7911d0"}}
F_ADMIN_PIN = {"path": BASE + "/parent-conserved-bootstrap-administrative-observations-20261008.json", "bytes": 2307,
               "sha256": "4810e9b55110553ce2877208ea4d381845021263bd25249034ae9c37545d149a"}
F_BEFORE_SOURCE_PIN = {"path": BASE + "/parent-post-bootstrap-resource-route-ROOT-review-20261008/source-before-valid.json", "bytes": 5381,
                       "sha256": "ebfd5d975947f47766f17def16c4835f568707aca00c98f91a4ca82dd0d8b97e"}
F_FAILED_PINS = {
    "cost": {"path": BASE + "/parent-direct-science-20261008-conserved-bootstrap-full-return.json", "bytes": 1304,
             "sha256": "cd04912465bad2d267928d8bedf0d446bc51ac5264dedbe4f49d83fe79c6ff65"},
    "request": {"path": BASE + "/parent-direct-science-20261008-conserved-bootstrap-request.json", "bytes": 1231,
                "sha256": "115974fdaec579e0388aa222c287b3c0e7a32c2e8f0b7181fa7958196cbd8354"},
    "outer": {"path": BASE + "/parent-direct-science-20261008-conserved-bootstrap/outer.json", "bytes": 1547,
              "sha256": "673c634d9b1d01a60ad94a8d8442ffd7de45e9fbdbcae31ece280b881a311757"},
    "trace": {"path": BASE + "/parent-direct-science-20261008-conserved-bootstrap-genuine-trace.json", "bytes": 3078,
              "sha256": "b8528a3f5202e9c6147677829ce56f9fbf581d891f259fb83a65fdeb2f7e63b0"},
    "source_freeze": {"path": E_SOURCE_PATH, "bytes": 249298,
                      "sha256": "a0b5c0e1d7f77708754e303dfa7450f1c5c89120bd12ec6663bfe7cb1bcaaebc"},
    "native_observation": {"path": BASE + "/parent-conserved-bootstrap-actual-failure-20261008/generated-conserved-native-observation.json",
                           "bytes": 740, "sha256": "daa175989a4983226ed4157e0c1c80bf9cfd2e424e9cdefd3e4b4450325ee1f2"},
    "junit": {"path": BASE + "/parent-conserved-resource-bootstrap-junit.xml", "bytes": 0,
              "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}}
F_REQUIRED_SOURCES = (*BOOTSTRAP_SOURCES, "tools/banc_m3_contact_learning_admission.py",
                      BASE + "/parent-contact-learning-contract-native.ps1", "tools/verify_c_banc_m3_contact_learning.py",
                      "tools/verify_c_banc_m3_startup_ray.py", "tools/banc_m3_contact_runtime_journal.py",
                      "tests/test_c_banc_contact_body_retry.py", "tests/test_c_banc_startup_ray_execution.py", ROSTER_PIN["path"], F_PATH)

_LOCAL_LOCKS = weakref.WeakSet()


def need(ok, label):
    if not ok:
        raise ValueError(label)


def exact(actual, expected, label):
    need(type(actual) is type(expected), label + " type")
    if type(expected) is dict:
        need(set(actual) == set(expected), label + " keys")
        for key, value in expected.items():
            exact(actual[key], value, label + "." + key)
    elif type(expected) in (list, tuple):
        need(len(actual) == len(expected), label + " length")
        for a, b in zip(actual, expected, strict=True):
            exact(a, b, label)
    else:
        if type(expected) is float:
            need(math.isfinite(actual) and math.isfinite(expected), label + " finite")
        need(actual == expected, label)


def finite(value, label):
    need(type(value) in (int, float) and math.isfinite(value) and value >= 0, label + " typed finite")
    return float(value)


def _text_path(root, name, *, write=False):
    """Private text operands only; lexical validation precedes every filesystem call."""
    trusted = os.fspath(root)
    need(type(trusted) is str and trusted and not trusted.startswith(("\\\\", "//"))
         and "\x00" not in trusted, "normal trusted absolute text root")
    if os.name == "nt":
        need(re.fullmatch(r"[A-Za-z]:[\\/].*", trusted) is not None and ":" not in trusted[2:], "one local rooted drive")
        components = trusted[3:].replace("\\", "/").split("/")
    else:
        need(trusted.startswith("/") and "\\" not in trusted and ":" not in trusted, "normal absolute POSIX text root")
        components = trusted[1:].split("/")
    need(all(p not in (".", "..") for p in components), "no root traversal normalization")
    need(type(name) is str and 0 < len(name) <= 4096 and "\\" not in name
         and re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", name) is not None,
         "literal relative text grammar, no namespace/drive/UNC/ADS")
    parts = name.split("/")
    need(all(p not in (".", "..") and not p.endswith((".", " ")) for p in parts), "no text traversal/Win32 aliases")
    reserved = r"(?i:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?"
    need(all(not re.fullmatch(reserved, p) for p in (*components, *parts)), "no device text operands")
    normal = Path(trusted)
    need(normal.is_absolute(), "lexical absolute root before lstat")
    def operand(p):
        return Path("\\\\?\\" + os.fspath(p)) if os.name == "nt" else p
    def checked(p, directory):
        info = os.lstat(operand(p))  # missing never means no-reparse
        need(not stat.S_ISLNK(info.st_mode) and not (getattr(info, "st_file_attributes", 0) & 0x400), "no text reparse ancestor")
        need(stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode), "real text directory/file")
        return info
    for parent in reversed((normal, *normal.parents)):
        checked(parent, True)
    current = normal
    for i, part in enumerate(parts):
        current = current / part
        need(current.is_relative_to(normal), "same lexical contained text namespace")
        directory = i < len(parts) - 1
        try:
            checked(current, directory)
        except FileNotFoundError:
            if not write:
                raise
            if directory:
                os.mkdir(operand(current))
                checked(current, True)
            # Missing leaf is allowed only after all existing/created parent lstat.
    return operand(current)


def _text_stamp(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
            getattr(info, "st_file_attributes", 0))


def raw_pin(spec, root, cap=1048576, *, allow_empty=False, opaque=False):
    need(type(spec) is dict and set(spec) == {"path", "bytes", "sha256"}, "exact bounded text descriptor")
    need(type(cap) is int and 0 < cap <= 2097152 and type(spec["bytes"]) is int
         and (0 if allow_empty else 1) <= spec["bytes"] <= cap and type(spec["sha256"]) is str
         and re.fullmatch("[0-9a-f]{64}", spec["sha256"]), "typed finite text pin")
    need(type(opaque) is bool and (not opaque or spec['path'].endswith(('/stdout.bin','/stderr.bin'))), 'literal opaque stream option')
    target = _text_path(root, spec["path"])
    before = os.lstat(target)
    need(before.st_size == spec["bytes"], "exact text size before bounded read")
    with target.open("rb") as stream:
        raw = stream.read(cap + 1)
    if not opaque: raw.decode("utf8", errors="strict")
    with target.open("rb") as stream:
        again = stream.read(cap + 1)
    _text_path(root, spec["path"])
    need(len(raw) == spec["bytes"] and raw == again and _text_stamp(os.lstat(target)) == _text_stamp(before)
         and hashlib.sha256(raw).hexdigest() == spec["sha256"], "full text repeated bytes/stamp/hash")
    return raw


def text_spec(name, root, cap=262144, *, allow_empty=False):
    target = _text_path(root, name)
    before = os.lstat(target)
    need(type(cap) is int and 0 < cap <= 2097152 and (0 if allow_empty else 1) <= before.st_size <= cap, "bounded text identity")
    with target.open("rb") as stream:
        raw = stream.read(cap + 1)
    spec = {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    need(_text_stamp(os.lstat(target)) == _text_stamp(before), "text identity stamp")
    raw_pin(spec, root, cap, allow_empty=allow_empty)
    return spec


def fixture_text_write(root, name, raw, *, opaque=False):
    """Generated text fixtures only; never an Original reader or ledger transaction."""
    need(type(raw) is bytes and len(raw) <= 2097152, "bounded generated text write")
    need(type(opaque) is bool and (not opaque or name.endswith(('/stdout.bin','/stderr.bin'))), 'literal opaque stream option')
    if not opaque: raw.decode("utf8", errors="strict")
    target = _text_path(root, name, write=True)
    with target.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    spec = {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    raw_pin(spec, root, 2097152, allow_empty=True, opaque=opaque)
    return spec


def source_text(spec, root):
    from tools import verify_c_banc_m3_startup_ray as ray
    if Path(spec["path"]).is_absolute():
        return ray.source_leaf(spec, root)  # inherited installed Source guard unchanged
    need(Path(spec["path"]).suffix in (".py", ".ps1", ".js", ".json", ".txt", ".ini", ".md", ".xml"), "text Source only")
    return raw_pin(spec, root, 2097152, allow_empty=True)


def read_pin(spec, root, cap=1048576):
    from tools import verify_c_banc_m3_startup_ray as ray
    return ray.json_load(raw_pin(spec, root, cap))

class SharedLock:
    """A locally created capability; token bytes alone never authorize a writer."""
    def __init__(self, target, value):
        self.target, self.value = target, copy.deepcopy(value)
        self._pid, self._local_nonce = os.getpid(), secrets.token_hex(32)
        self.uncertain = self.released = False
        self.raw = json.dumps(value, sort_keys=True, allow_nan=False).encode("utf8")
        need(len(self.raw) <= 16384, "bounded shared exclusion token")
        with target.open("xb") as stream:
            stream.write(self.raw)
            stream.flush()
            os.fsync(stream.fileno())
        need(target.read_bytes() == self.raw, "shared exclusion fsync/full exact readback")
        _LOCAL_LOCKS.add(self)

    def authenticate(self, target=None):
        need(self in _LOCAL_LOCKS and self._pid == os.getpid() and not self.released
             and type(self._local_nonce) is str and len(self._local_nonce) == 64, "actual local owner capability")
        need(target is None or self.target == target, "the same common physical exclusion")
        need(self.target.read_bytes() == self.raw, "authenticated exact locally owned token")
        return True

    def release(self, closed):
        self.authenticate()
        need(closed is True and not self.uncertain, "uncertain persistence/ownership retains hold and lock")
        self.target.unlink()
        need(not self.target.exists(), "exact owner's shared exclusion released")
        self.released = True
        _LOCAL_LOCKS.discard(self)


def _d19_preserve_current_order(path,before_raw,ledger,lock):
 """CURRENT / LEGACY / REFUSE, after the actual validation callback."""
 import os

 from tools import banc_m3_checkpoint_harness_runtime as k
 from tools import banc_m3_startup_ray_supervisor as supervisor
 need(type(before_raw) is bytes,'Typed raw-before at writer boundary')
 before=d19_json(before_raw);key=k.KEY;route=k.RECOVERY_PATH
 value=lock.value;d19_whole(d19_json(lock.raw),d19_owner_wire(value));request=value.get('request');q=None
 if request is not None:
  q=k.data(request,Path(path).resolve().parents[2]);need(type(q) is dict,'Typed owned request')
 intent=any(type(v) is dict and type(v.get(key)) is dict and v[key].get('path')==route for v in (before,ledger))
 intent=intent or (type(q) is dict and q.get('call_id') in k.SEQUENCE)
 if not intent:
  # Legacy authority is the old validator, not absence of a future filename.
  need(type(before) is dict and type(ledger) is dict and key not in before and key not in ledger,'REFUSE unrecognized or conflicting current pointer')
  need(any(name in before and name in ledger for name in (J_KEY,I_KEY,H_KEY,G_KEY,'direct_science_conserved_allocation')),'Recognized legacy epoch')
  current_caps(ledger,Path(path).resolve().parents[2],supervisor.generated_budget_limit)
  return False
 root=Path(path).resolve().parents[2]
 need(Path(path).resolve()==root/BASE/'parent-e358-execution-ledger.json' and lock.target.resolve()==root/SHARED_LOCK,'CURRENT exact canonical and shared exclusion')
 lock.authenticate()
 need(type(ledger) is dict and type(ledger.get(key)) is dict and ledger[key].get('path')==route,'REFUSE missing/altered current pointer')
 need(type(before) is dict and type(before.get(key)) is dict and before[key].get('path')==route,'REFUSE current before namespace')
 d19_whole(before[key],ledger[key]);d19_current_pin(k.__dict__,ledger[key],route)
 _state=k.validate(ledger,root);event=k.data(ledger[key],root)
 need(event['event_id']=='D19_390_R63_0' and event['writer_protocol']=='current-insertion-order-v1','CURRENT authenticated event protocol')
 need(type(value.get('owner_pid')) is int and value['owner_pid']==os.getpid() and type(value.get('owner_creation_ticks')) is int and value['owner_creation_ticks']==supervisor.creation_ticks(os.getpid()),'CURRENT actual local PID/FILETIME')
 need(value.get('schema')=='flylab.m3.direct-science.lock-owner.v1' and q is not None and value.get('call_id')==q.get('call_id') and q['call_id'] in k.SEQUENCE,'CURRENT exact owned literal request')
 k.identity(q,request,event['source_freeze'],root)
 need(type(value.get('entry_counter_s')) is float and 0<value['entry_counter_s']<=__import__('time').perf_counter(),'CURRENT real local ENTRY')
 need(Path(path).read_bytes()==before_raw,'CURRENT stale raw-before')
 return True


def d19_current_pin(k,pin,path):
 k['metadata_pin'](pin,path)
 expected={'path':path,'bytes':pin['bytes'],'sha256':pin['sha256']}
 d19_whole(pin,expected)
 return expected


def atomic_ledger_publish(path, before_raw, ledger, lock, *, validate):
    """Caller owns the shared capability first; no overwrite retry or foreign lease."""
    need(isinstance(lock, SharedLock), "actual SharedLock object required")
    lock.authenticate()
    need(not lock.uncertain and type(before_raw) is bytes and len(before_raw) <= 524288, "known bounded transaction")
    validate(ledger)
    after = json.dumps(ledger, allow_nan=False, ensure_ascii=True, sort_keys=not _d19_preserve_current_order(path,before_raw,ledger,lock)).encode("utf8")
    need(len(after) <= 524288, "bounded full ledger publication")
    need(path.read_bytes() == before_raw, "fresh raw-before comparison")
    temporary = path.with_name(path.name + "." + secrets.token_hex(12) + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(after)
        stream.flush()
        os.fsync(stream.fileno())
    need(temporary.read_bytes() == after, "temporary full fsync readback")
    lock.authenticate()
    need(path.read_bytes() == before_raw, "exact unchanged before bytes immediately before atomic replace")
    lock.uncertain = True
    os.replace(temporary, path)  # one attempt, retain exclusion on uncertain replace/readback
    need(path.read_bytes() == after, "full exact atomic publication readback")
    lock.uncertain = False
    return after


def known_closed(ledger):
    for key in OWNER_KEYS:
        exact(ledger[key], {}, "allocation empty ownership " + key)
    exact(ledger["original_jobs_running"], 0, "no Original job")
    exact(ledger["original_reserved_s"], 0, "no Original reservation")
    need(all(event["state"] == "charged" for event in ledger["direct_science_calls"].values()), "all actual calls once charged")


def prefix(current, before):
    need(len(current["debits"]) >= len(before["debits"]), "entire historical debit prefix")
    exact(current["debits"][:len(before["debits"])], before["debits"], "all actual debit history")
    for call, value in before["direct_science_calls"].items():
        exact(current["direct_science_calls"][call], value, "all charged call history")


def ordered_totals(ledger, policy, grant, caps):
    generated = original = 0.0
    need(type(ledger["debits"]) is list, "ordered actual debit array")
    ids = []
    for i, debit in enumerate(ledger["debits"]):
        need(type(debit["cost_id"]) is str and debit["mode"] in ("generated", "original"), "typed cost ID and historical mode")
        ids.append(debit["cost_id"])
        if i >= 118 and debit["mode"] == "generated":
            generated += finite(debit["actual_outer_full_return_s"], "generated actual debit")
        if i >= 165 and debit["mode"] == "original":
            original += finite(debit["actual_outer_full_return_s"], "Original actual debit")
    need(len(ids) == len(set(ids)), "no duplicate cost event")
    stage = float(policy["generated_stage_before_s"]) + generated
    values = {"direct_science_generated_used_s": generated, "generated_stage_used_s": stage,
              "generated_shared_cumulative_used_s": float(policy["generated_shared_before_s"]) + generated,
              "generated_shared_remaining_s": float(caps["generated_global"]) - float(policy["generated_shared_baseline_s"]) - stage,
              "original_cumulative_used_s": float(grant["original_cumulative_before_s"]) + original,
              "original_stage_used_s": float(grant["original_stage_before_s"]) + original,
              "startup_ray_original_local_used_s": float(grant["science_local_used_before_s"]) + original,
              "original_remaining_s": float(caps["original_global"]) - float(grant["original_cumulative_before_s"]) - original}
    for key, value in values.items():
        need(math.isfinite(value) and value >= 0, "finite nonnegative ordered total " + key)
        exact(ledger[key], value, "exact ordered binary64 " + key)
    return values


def complete_roster(root):
    roster = read_pin(ROSTER_PIN, root)
    exact(roster["schema"], "flylab.m3.contact-learning.qualification-roster.v2", "fixed complete roster")
    exact(roster["epoch"], EPOCH, "explicit new epoch")
    for kind in ("math", "gate_a", "gate_b", "gate"):
        rows = roster[kind]
        need(type(rows) is list and len({row["node_id"] for row in rows}) == len(rows), "no missing or duplicate roster node")
        for row in rows:
            exact(row["node_id"], row["module"] + "::" + row["name"], "module/name/parameterized identity")
            exact(row["classname"], row["module"][:-3].replace("/", "."), "full module classname")
    a, b, whole = ({row["node_id"] for row in roster[kind]} for kind in ("gate_a", "gate_b", "gate"))
    need(not a & b and a | b == whole, "disjoint full gate union")
    need(len(roster["math"]) == 19 and len(whole) == 53 and len(a) == 21 and len(b) == 32, "complete fixed math19/gate53")
    return roster


def generated_plan(role, root):
    roster = complete_roster(root)
    need(role in ("lint", "math", "gate-a", "gate-b"), "four fixed qualification roles")
    index = ("lint", "math", "gate-a", "gate-b").index(role)
    if role == "lint":
        modules = ("tools/banc_m3_contact_learning.py", "tools/verify_c_banc_m3_contact_learning.py",
                   "tools/analyze_c_banc_m3_contact_learning.py", "tools/banc_m3_contact_learning_admission.py",
                   "tools/banc_m3_startup_ray_supervisor.py", "tools/verify_c_banc_m3_startup_ray.py",
                   "tools/banc_m3_contact_runtime_journal.py", HELPER_SOURCE,
                   "tests/test_c_banc_contact_runtime_journal.py", "tests/test_c_banc_contact_learning.py",
                   "tests/test_c_banc_contact_metadata.py", "tests/test_c_banc_startup_ray_execution.py", "tests/test_c_banc_contact_body_retry.py")
        argv = ["-m", "ruff", "check", *modules]
    else:
        kind = role.replace("-", "_")
        argv = ["-m", "pytest", *[row["node_id"] for row in roster[kind]], "--tb=short", "-q",
                "--junitxml=" + BASE + "/parent-contact-learning-e-" + role + "-junit.xml"]
    return {"call": QUAL_CALLS[index], "role": ("lint", "numerical", "execution", "execution")[index],
            "argv": argv, "caps": QUAL_CAPS[index]}


def successful_junit(spec, rows, root):
    raw = raw_pin(spec, root)
    document = ET.fromstring(raw)
    suites = document.findall("testsuite")
    need(document.tag == "testsuites" and len(suites) == 1, "one complete actual JUnit suite")
    suite = suites[0]
    need(all(suite.get(k) == "0" for k in ("failures", "errors", "skipped"))
         and suite.get("tests") == str(len(rows)), "no skipped/error/failed suite")
    cases = suite.findall("testcase")
    actual = [(case.get("classname"), case.get("name")) for case in cases]
    expected = [(row["classname"], row["name"]) for row in rows]
    exact(actual, expected, "every full exact JUnit module/class/name")
    need(len(set(actual)) == len(actual) and all(not list(case) for case in cases), "no duplicates or testcase failure/skip/error children")


def bootstrap_sources(spec, root):
    """Bind the actual tested freeze to the complete current production writer.

    The freeze is created after Source publication and paid by the actual cost.
    Its dynamic descriptors include the account's embedded Native definitions
    and this exact full fixture; no Source embeds its own/future content hash.
    """
    freeze = read_pin(spec, root, BOOTSTRAP_SOURCE_CAP)
    exact(freeze["schema"], "flylab.m3.contact-learning.source-freeze.v1", "actual bootstrap freeze schema")
    exact(freeze["ROOT_review_cycle_complete"], True, "actual reviewed bootstrap Source")
    need(type(freeze["sources"]) is dict, "typed bootstrap Source dictionary")
    for name in BOOTSTRAP_SOURCES:
        need(name in freeze["sources"], "missing critical bootstrap Source " + name)
        pin = freeze["sources"][name]
        need(type(pin) is dict and set(pin) == {"path", "bytes", "sha256"}, "exact critical Source descriptor")
        exact(pin["path"], name, "same critical production Source path")
        need(type(pin["bytes"]) is int and 0 < pin["bytes"] <= BOOTSTRAP_SOURCE_CAP
             and type(pin["sha256"]) is str and len(pin["sha256"]) == 64
             and all(c in "0123456789abcdef" for c in pin["sha256"]), "typed bounded critical Source descriptor")
        # Full current read/hash/readback/stamp, not a cached pin or imported module.
        raw_pin(pin, root, BOOTSTRAP_SOURCE_CAP).decode("utf8", errors="strict")
    return freeze


def bootstrap_proof(spec, before, root):
    from tools.banc_m3_contact_learning_admission import cost_proof
    proof = read_pin(spec, root)
    exact(set(proof), {"schema", "cost", "source_freeze", "fixture_node", "junit"}, "bootstrap exact evidence fields")
    exact(proof["schema"], "flylab.m3.conserved-resource.bootstrap.v1", "actual bootstrap proof")
    exact(proof["fixture_node"], BOOTSTRAP_NODE, "one complete fixed meaningful bootstrap")
    debit = before["debits"][-1]
    exact(debit["cost_id"], BOOTSTRAP_CALL, "sole prospective paid bootstrap")
    exact(debit["descriptor"], proof["cost"], "fresh before includes actual bootstrap once")
    cost = read_pin(proof["cost"], root)
    request = cost_proof(cost, debit, lambda p: read_pin(p, root), generated=True)
    exact(cost["exit_code"], 0, "successful known-closed actual bootstrap")
    exact(cost["source_freeze"], proof["source_freeze"], "actual bootstrap frozen Source")
    bootstrap_sources(proof["source_freeze"], root)  # before any allocation/cap success
    exact(request["generated_argv"], ["-m", "pytest", BOOTSTRAP_NODE, "--tb=short", "-q",
                                     "--junitxml=" + BASE + "/parent-conserved-resource-bootstrap-junit.xml"], "fixed complete bootstrap argv")
    exact(request["limits"]["outer_s"], 11, "old-budget maximum11")
    exact(request["limits"]["child_s"], 8, "same ENTRY child8")
    exact(request["limits"]["cleanup_s"], 3, "cleanup3 plus distinct close20")
    exact(request["generated_behavior"], "command", "generated temp fixture only")
    successful_junit(proof["junit"], [{"classname": "tests.test_c_banc_contact_runtime_journal", "name": BOOTSTRAP_NODE.split("::")[1]}], root)
    return proof


def allocation_before(event, root, legacy_validate):
    expected = {"schema", "event_id", "before_ledger", "historical_closed282", "initial_policy", "transfers", "j_grant",
                "direct_human_auto_authority", "root_decision", "old_caps", "new_caps", "delta_s",
                "combined_global_s", "new_resource_grant_s", "bootstrap"}
    exact(set(event), expected, "one exact conserved allocation event")
    exact(event["schema"], "flylab.m3.conserved-resource-allocation.v1", "event schema")
    exact(event["event_id"], "conserved-resource-20261008-600", "single prospective event ID")
    for key, value in (("historical_closed282", HISTORICAL_PIN), ("initial_policy", POLICY_PIN), ("j_grant", J_PIN),
                       ("direct_human_auto_authority", HUMAN_PIN), ("old_caps", OLD_CAPS), ("new_caps", NEW_CAPS),
                       ("delta_s", DELTA), ("combined_global_s", 17900), ("new_resource_grant_s", 0)):
        exact(event[key], value, "conserved event " + key)
    exact(event["root_decision"], {"source": ROOT_PIN, "pool_transfer_s": 600, "human_numerical_approval": False}, "ROOT choice, not numerical human approval")
    raw_pin(ROOT_PIN, root).decode("utf8", errors="strict")
    authority = read_pin(HUMAN_PIN, root)
    exact(authority, {"schema": "flylab.m3.contact-metadata.current-human-autonomy.v1", "direct_human_authority": True,
                      "automatic_authority": True, "new_resource_grant_s": 0,
                      "literal_latest_user": "앞으로 내 승인 받지 말고 스스로 판단해서 진행해줘. 개입할 상황이면 내가 알아서 개입할게"}, "literal latest delegation")
    before, historical = read_pin(event["before_ledger"], root, 524288), read_pin(HISTORICAL_PIN, root, 524288)
    need(ALLOCATION_KEY not in before and legacy_validate(before, root) == 964, "real legacy before, never reset-cap clone")
    known_closed(before)
    prefix(before, historical)
    need(len(before["debits"]) == len(historical["debits"]) + 1
         and len(before["direct_science_calls"]) == len(historical["direct_science_calls"]) + 1, "fresh once-paid bootstrap tail, historical282 retained")
    for key, value in historical.items():
        if key not in TAIL_KEYS:
            exact(before[key], value, "bootstrap immutable historical " + key)
    for key, value in OLD_CAPS.items():
        exact(before[CAP_FIELDS[key]], value, "before paired old caps")
    exact(before["direct_science_policy"], POLICY_PIN, "original immutable policy")
    exact(event["transfers"], {key: before[key] for key in TRANSFER_KEYS}, "all six immutable transfer bytes")
    exact(before["startup_ray_original_grants"]["startup-ray-original-20261007-j"]["grant"], J_PIN, "historical registered J retained")
    policy, grant = read_pin(POLICY_PIN, root), read_pin(J_PIN, root)
    ordered_totals(before, policy, grant, OLD_CAPS)
    bootstrap_proof(event["bootstrap"], before, root)
    return before, policy, grant


def allocated_snapshot(before, event_spec, policy, grant):
    """Construct the sole permitted forward diff; this never restores legacy caps."""
    after = copy.deepcopy(before)
    for key, value in NEW_CAPS.items():
        after[CAP_FIELDS[key]] = value
    after["generated_shared_remaining_s"] = 2300.0 - float(policy["generated_shared_baseline_s"]) - float(before["generated_stage_used_s"])
    total = 0.0
    for debit in before["debits"][165:]:
        if debit["mode"] == "original":
            total += finite(debit["actual_outer_full_return_s"], "actual Original")
    after["original_remaining_s"] = 15600.0 - float(grant["original_cumulative_before_s"]) - total
    after[ALLOCATION_KEY] = copy.deepcopy(event_spec)
    return after


def current_caps(ledger, root, legacy_validate):
    from tools import banc_m3_checkpoint_harness_runtime as harness
    from tools import banc_m3_checkpoint_runtime as checkpoint
    if harness.KEY in ledger:
        checkpoint = harness
    if checkpoint.KEY in ledger:
        return checkpoint.validate(ledger, root)['caps']
    need(checkpoint.ALLOCATION_KEY not in ledger and checkpoint.QUALIFICATION_KEY not in ledger, 'K pointers require exact checkpoint recovery')
    need(not any(ledger.get(CAP_FIELDS[k]) == v for k, v in checkpoint.CAPS.items()), 'K mixed caps require checkpoint')
    if J_KEY in ledger:
        return j_current_caps(ledger, root, legacy_validate)
    if I_KEY in ledger:
        return i_current_caps(ledger, root, legacy_validate)
    if H_KEY in ledger:
        return h_current_caps(ledger, root, legacy_validate)
    if G_KEY in ledger:
        return g_current_caps(ledger, root, legacy_validate)
    if ALLOCATION_KEY not in ledger:
        need(ledger.get("generated_shared_cumulative_limit_s") != 2300
             and ledger.get("direct_science_generated_limit_s") != 1564
             and ledger.get("original_cumulative_limit_s") != 15600, "new/mixed caps require exact event")
        return dict(OLD_CAPS)
    event_spec = ledger[ALLOCATION_KEY]
    need(event_spec["path"] in (ALLOCATION_PATH, F_EVENT_PATH), "only strict v1/F event paths")
    event = read_pin(event_spec, root)
    before, policy, grant = allocation_route_before(event_spec, event, root, legacy_validate)
    expected = allocated_snapshot(before, event_spec, policy, grant)
    prefix(ledger, before)
    need(set(ledger) == set(expected), "no duplicate/extra allocation field")
    for key, value in expected.items():
        if key not in TAIL_KEYS:
            exact(ledger[key], value, "immutable allocated history/policy/cap " + key)
    ordered_totals(ledger, policy, grant, NEW_CAPS)
    # Active worker/admission dictionaries remain legal here; allocation alone requires emptiness.
    return dict(NEW_CAPS)


def validate_qualification(qualification, ledger, root, legacy_validate):
    from tools import banc_m3_checkpoint_harness_runtime as harness
    from tools import banc_m3_checkpoint_runtime as checkpoint
    if harness.KEY in ledger:
        checkpoint = harness
    if checkpoint.KEY in ledger:
        state = checkpoint.validate(ledger, root)
        checkpoint.qualification(qualification, ledger, state['source'], root)
        checkpoint.source_fence(state['source'], root)
        # D15B retains old failed bootstrap; qualified whole324.
        return 324 if checkpoint is harness else 293
    if J_KEY in ledger:
        return validate_j_qualification(qualification, ledger, root, legacy_validate)
    if I_KEY in ledger:
        return validate_i_qualification(qualification, ledger, root, legacy_validate)
    if H_KEY in ledger:
        return validate_h_qualification(qualification, ledger, root, legacy_validate)
    if G_KEY in ledger:
        return validate_g_qualification(qualification, ledger, root, legacy_validate)
    if ledger.get(ALLOCATION_KEY, {}).get("path") == F_EVENT_PATH:
        return validate_f_qualification(qualification, ledger, root, legacy_validate)
    caps = current_caps(ledger, root, legacy_validate)
    exact(caps, NEW_CAPS, "real paired allocation before four-call qualification")
    exact(qualification["schema"], "flylab.m3.direct-science.qualification.v2", "explicit four-cost/three-JUnit epoch")
    exact(qualification["epoch"], EPOCH, "qualification epoch")
    exact(qualification["old_E360"], "UNPASSED", "old failed qualification unchanged")
    exact(qualification["allocation"], ledger[ALLOCATION_KEY], "same real allocation")
    exact(qualification["qualification_roster"], ROSTER_PIN, "complete reviewed static roster")
    exact(qualification["source_freeze"]["path"], E_SOURCE_PATH, "new full same Source path")
    freeze = read_pin(qualification["source_freeze"], root)
    exact(freeze["ROOT_review_cycle_complete"], True, "reviewed full Source")
    bootstrap_sources(qualification["source_freeze"], root)
    for name in (*BOOTSTRAP_SOURCES, ROSTER_PIN["path"], E_PATH, "tests/test_c_banc_contact_body_retry.py",
                 "tests/test_c_banc_startup_ray_execution.py"):
        need(name in freeze["sources"], "all new production/fixture/roster/E Source frozen")
        raw_pin(freeze["sources"][name], root)
    exact(freeze["sources"][ROSTER_PIN["path"]], ROSTER_PIN, "fixed roster Source identity")
    event = read_pin(ledger[ALLOCATION_KEY], root)
    allocation_anchor = read_pin(event["before_ledger"], root, 524288)
    offset = len(allocation_anchor["debits"])
    need(type(qualification["actual_costs"]) is list and len(qualification["actual_costs"]) == 4, "all four genuine actual costs")
    from tools.banc_m3_contact_learning_admission import cost_proof
    for i, role in enumerate(("lint", "math", "gate-a", "gate-b")):
        plan = generated_plan(role, root)
        debit = ledger["debits"][offset + i]
        exact(debit["cost_id"], plan["call"], "ordered four-call epoch")
        exact(debit["descriptor"], qualification["actual_costs"][i], "actual once debit descriptor")
        cost = read_pin(debit["descriptor"], root)
        request = cost_proof(cost, debit, lambda p: read_pin(p, root), generated=True)
        exact(cost["exit_code"], 0, "genuine successful qualification")
        exact(cost["source_freeze"], qualification["source_freeze"], "all four same executed Source")
        exact(request["role"], plan["role"], "fixed real qualification role")
        exact(request["generated_argv"], plan["argv"], "complete explicit node argv; no -k subset")
        for key, limit in zip(("outer_s", "child_s", "cleanup_s"), plan["caps"], strict=True):
            exact(request["limits"][key], limit, "fixed generated outer/child/cleanup")
        exact(ledger["direct_science_calls"][cost["cost_id"]],
              {"state": "charged", "mode": "generated", "request": cost["request"], "reserved_s": plan["caps"][0]}, "known charged qualification request")
    need(len(qualification["tests"]) == 3, "all three successful complete JUnits")
    roster = complete_roster(root)
    for item, role in zip(qualification["tests"], ("math", "gate-a", "gate-b"), strict=True):
        rows = roster[role.replace("-", "_")]
        exact(item["file"]["path"], BASE + "/parent-contact-learning-e-" + role + "-junit.xml", "fixed complete JUnit")
        exact(item["node_ids"], [row["node_id"] for row in rows], "full fixed parameterized actual nodes")
        exact(item["names"], [row["name"] for row in rows], "every exact name")
        exact(item["classnames"], [row["classname"] for row in rows], "every exact module/class")
        exact(item["count"], len(rows), "complete actual collected count")
        successful_junit(item["file"], rows, root)
    need(len(qualification["evidence"]) >= 4, "four genuine observations, no shard qualifies alone")
    for spec in qualification["evidence"]:
        read_pin(spec, root, 2097152)
    return offset + 4



def f_generated_plan(role, root):
    # The immutable roster's E epoch stays distinct from this request/event F epoch.
    old = generated_plan(role, root)
    index = ("lint", "math", "gate-a", "gate-b").index(role)
    argv = list(old["argv"])
    if role != "lint":
        argv[-1] = "--junitxml=" + BASE + "/parent-contact-learning-f-" + role + "-junit.xml"
    return {"call": F_QUAL_CALLS[index], "role": old["role"], "argv": argv,
            "limits": dict(zip(("outer_s", "child_s", "cleanup_s"), QUAL_CAPS[index], strict=True),
                           host_bytes=2147483648, output_cap_bytes=8388608),
            "output": BASE + "/parent-" + F_QUAL_CALLS[index]}


def f_plan(root):
    return {"epoch": F_EPOCH, "roster_epoch": EPOCH, "qualification_roster": ROSTER_PIN,
            "calls": [f_generated_plan(role, root) for role in ("lint", "math", "gate-a", "gate-b")],
            "generated_close_s": 20, "original_release_calls": list(F_CALLS)}


def f_fixup_provenance(provenance, root):
    exact(set(provenance), {"schema", "design", "astra_review", "before_source", "root_fixup"},
          "exact separate fixup provenance")
    exact(provenance["schema"], "flylab.m3.post-bootstrap-root-fixup.provenance.v1", "fixup provenance schema")
    exact(provenance["design"], F_FIXUP_DESIGN_PIN, "actual ROOT reflected fixup")
    exact(provenance["astra_review"], F_FIXUP_ASTRA_PINS, "actual auxiliary Astra reports")
    exact(provenance["before_source"], F_FIXUP_BEFORE_PIN, "distinct fixup-before14, never bootstrap-before13")
    for pin in (F_FIXUP_DESIGN_PIN, *F_FIXUP_ASTRA_PINS.values()):
        raw_pin(pin, root).decode("utf8", errors="strict")
    before = read_pin(F_FIXUP_BEFORE_PIN, root)
    exact(before["schema"], "flylab.m3.post-bootstrap-root-fixup.source-before.v1", "actual fixup archive schema")
    exact(before["source_metadata_only"], True, "fixup is Source/metadata, not conformance")
    exact(before["scientific_execution"], "NOT_RUN", "before record never qualification")
    exact(before["allocation_committed"], False, "failed original F did not allocate")
    need(type(before["snapshots"]) is list and len(before["snapshots"]) == 14, "entire separate before14")
    names = set()
    for item in before["snapshots"]:
        exact(set(item), {"source", "archive"}, "exact before14 entry")
        pin = item["source"]
        need(pin["path"] not in names, "unique before14 Source")
        names.add(pin["path"])
        exact(item["archive"], dict(pin, path=BASE + "/parent-post-bootstrap-root-fixup-ROOT-before-20261008/before/" + pin["path"]),
              "full exact preserved fixup archive")
        raw_pin(item["archive"], root, 262144).decode("utf8", errors="strict")
    need(all(name in names for name in F_FIXUP_SOURCES), "all five replaced current Sources archived")
    report = provenance["root_fixup"]
    exact(report["path"], F_FIXUP_REPORT_PATH, "only actual final ROOT fixup report path")
    raw_pin(report, root, 65536).decode("utf8", errors="strict")
    return before


def f_administrative_refusal(root):
    refusal = read_pin(F_REFUSAL_PIN, root)
    exact(refusal["schema"], "flylab.m3.post-bootstrap-resource-route.administrative-allocation-refusal.v1", "actual failed administration")
    exact(refusal["actual_action"], "conserved-allocation", "same failed administrative action")
    exact(refusal["actual_event"], F_ORIGINAL_EVENT_PIN, "immutable original unallocated F event")
    exact(refusal["actual_source_adoption"], F_ADOPTION_PIN, "actual Source adoption precedes refusal")
    exact(refusal["actual_tool_first_final"]["exit_code"], 1, "actual administrative first FINAL failed")
    need("session_id" not in refusal["actual_tool_first_final"]
         and "Exactly two schema-bound prospective routes" in refusal["actual_tool_first_final"]["output"],
         "known final route refusal, never ongoing/success")
    exact(refusal["observed_unchanged_ledger"], dict(F_CLOSED_PIN, path=BASE + "/parent-e358-execution-ledger.json"),
          "actual after-read unchanged closed283")
    for key, value in (("observed_allocation_present", False), ("observed_shared_lock_absent", True),
                       ("debits", 283), ("direct_calls", 165), ("scientific_execution", "NOT_RUN"),
                       ("qualification", "NOT_RUN"), ("original", "NOT_RUN"),
                       ("scientific_cost_created", False), ("scientific_fees_rewritten", False)):
        exact(refusal[key], value, "actual refusal " + key)
    read_pin(F_ADOPTION_PIN, root)
    return refusal


def _f_review_metadata(spec, root):
    exact(spec["path"], F_REVIEW_PATH, "one reviewed current Source vector")
    review = read_pin(spec, root)
    exact(set(review), {"schema", "epoch", "sources", "root_reflection", "dependency_addendum", "astra_review", "root_reviews", "fixup_provenance"},
          "exact reviewed Source manifest fields")
    exact(review["schema"], "flylab.m3.conserved-resource-prequalification.reviewed-source.v2", "fixed Source review schema")
    exact(review["epoch"], F_EPOCH, "fresh reviewed F Source epoch")
    for key, pin in (("root_reflection", F_ROOT_PIN), ("dependency_addendum", F_ADDENDUM_PIN)):
        exact(review[key], pin, "current ROOT design binding")
        raw_pin(pin, root)
    exact(review["astra_review"], F_ASTRA_PINS, "actual Astra review binding")
    for pin in F_ASTRA_PINS.values():
        raw_pin(pin, root)
    need(type(review["root_reviews"]) is list and 1 <= len(review["root_reviews"]) <= 3, "actual final ROOT review provenance")
    allowed = {"docs/BANC_M3_POST_BOOTSTRAP_RESOURCE_ROUTE_ROOT_" + suffix + "_20261008.md"
               for suffix in ("FIRST_REVIEW", "SECOND_REVIEW", "RESIDUAL")}
    paths = [pin["path"] for pin in review["root_reviews"]]
    need(len(set(paths)) == len(paths) and all(name in allowed for name in paths), "no foreign/duplicate ROOT review")
    need("docs/BANC_M3_POST_BOOTSTRAP_RESOURCE_ROUTE_ROOT_FIRST_REVIEW_20261008.md" in paths, "actual new ROOT FIRST")
    for pin in review["root_reviews"]:
        raw_pin(pin, root).decode("utf8", errors="strict")
    original = read_pin(F_ORIGINAL_REVIEW_PIN, root)  # historical metadata; superseded Source pins are not cold-read
    exact(original["schema"], "flylab.m3.conserved-resource-prequalification.reviewed-source.v1", "immutable original F manifest")
    exact(review["root_reviews"], original["root_reviews"], "preserve every actual original ROOT review")
    need(type(original["sources"]) is dict and len(original["sources"]) == 722, "full inherited original F722")
    f_fixup_provenance(review["fixup_provenance"], root)
    vector = review["sources"]
    need(type(vector) is dict and all(name in vector for name in F_REQUIRED_SOURCES), "complete production/current fixture Source vector")
    exact(set(vector), set(original["sources"]) | {F_PATH}, "all inherited722 plus fixed static only, no omission/extra")
    for name, pin in original["sources"].items():
        if name not in F_FIXUP_SOURCES:
            exact(vector[name], pin, "unchanged unrelated inherited Source " + name)
    forbidden = {F_REVIEW_PATH, F_EVENT_PATH, F_SOURCE_PATH, F_QUAL_PATH}
    for name, pin in vector.items():
        need(type(name) is str and name not in forbidden and not any(call in name for call in (*F_QUAL_CALLS, *F_CALLS)),
             "acyclic Source vector has no own/event/future request/cost/qualification hash")
        exact(pin["path"], name, "Source vector path identity")
        need(Path(name).suffix in (".py", ".ps1", ".js", ".json", ".txt", ".ini", ".md", ".xml"), "text Source only, never Original arrays")
        need(type(pin["bytes"]) is int and 0 <= pin["bytes"] <= 2097152 and re.fullmatch("[0-9a-f]{64}", pin["sha256"]), "historical Source pin shape")
    exact(vector[ROSTER_PIN["path"]], ROSTER_PIN, "unchanged full E roster")
    exact(vector[F_ORIGINAL_DESCRIPTOR_PIN["path"]], F_ORIGINAL_DESCRIPTOR_PIN, "preserve original static F descriptor")
    exact(vector[F_PATH], F_DESCRIPTOR_PIN, "one static F body scope")
    return review


def f_reviewed_sources(spec, root):
    review = _f_review_metadata(spec, root)
    for pin in review["sources"].values():
        source_text(pin, root)
    return review

def f_failed_history(before, root):
    from tools.banc_m3_contact_learning_admission import cost_proof
    debit = before["debits"][-1]
    exact(debit["cost_id"], BOOTSTRAP_CALL, "actual spent bootstrap retained")
    exact(debit["descriptor"], F_FAILED_PINS["cost"], "actual failed cost descriptor")
    cost = read_pin(F_FAILED_PINS["cost"], root)
    request = cost_proof(cost, debit, lambda pin: read_pin(pin, root), generated=True, expected_timeout=True)
    exact(cost["exit_code"], 124, "failed v1 remains exit124")
    exact(cost["actual_outer_full_return_s"], 9.208, "entire once-paid failed fee")
    for key in ("request", "outer", "trace", "source_freeze"):
        exact(cost[key], F_FAILED_PINS[key], "immutable actual failed evidence")
    exact(request["call_id"], BOOTSTRAP_CALL, "no failed-call alias")
    exact(request["generated_argv"], ["-m", "pytest", BOOTSTRAP_NODE, "--tb=short", "-q",
                                     "--junitxml=" + F_FAILED_PINS["junit"]["path"]], "spent whole fixture, no renamed retry")
    exact(request["limits"], {"outer_s": 11, "child_s": 8, "cleanup_s": 3, "host_bytes": 2147483648,
                              "output_cap_bytes": 8388608}, "failed attempt exact original envelope")
    exact(request["source_freeze"], F_FAILED_PINS["source_freeze"], "as-executed E remains immutable")
    old_freeze = read_pin(F_FAILED_PINS["source_freeze"], root)
    observation = read_pin(F_FAILED_PINS["native_observation"], root)
    exact(observation["returncode"], None, "Native did not start")
    exact(observation["worker_pid"], None, "no Native PID")
    exact(observation["worker_creation_ticks"], None, "no Native creation")
    exact(observation["pre_spawn_refused"], True, "actual parser refusal")
    exact(observation["first_error"], {"type": "AttributeError", "message": "'str' object has no attribute 'decode'"}, "actual failure preserved")
    exact(raw_pin(F_FAILED_PINS["junit"], root, cap=1048576, allow_empty=True), b"", "empty JUnit is failure, never success")
    admin = read_pin(F_ADMIN_PIN, root)
    for key in ("admission", "commit"):
        exact(admin[key]["final_tool_exit_code"], 0, "actual bookkeeping return only")
    exact(admin["admission"]["parsed_actual_stdout"]["admitted"], BOOTSTRAP_CALL, "actual admitted failed call")
    exact(admin["admission"]["parsed_actual_stdout"]["request"], F_FAILED_PINS["request"], "actual admitted request")
    commit = admin["commit"]["parsed_actual_stdout"]
    exact(commit["committed"], True, "actual once settlement")
    exact(commit["cost"], F_FAILED_PINS["cost"], "actual committed failure")
    exact(commit["actual_s"], 9.208, "no administrative credit/refund")
    exact(commit["exit_code"], 124, "bookkeeping success is not scientific success")
    exact(commit["ledger"], dict(F_CLOSED_PIN, path=BASE + "/parent-e358-execution-ledger.json"), "actual final mutable-ledger observation")
    archived = read_pin(F_BEFORE_SOURCE_PIN, root)
    exact(archived["schema"], "flylab.m3.post-bootstrap-resource-route.source-before.v1", "actual before-source archive record")
    need(type(archived["source_snapshots"]) is list and len(archived["source_snapshots"]) == 13, "full recorded before Source13")
    seen = set()
    for item in archived["source_snapshots"]:
        exact(set(item), {"path", "bytes", "sha256", "archive"}, "exact prior Source record")
        need(item["path"] not in seen, "unique prior Source archive")
        seen.add(item["path"])
        exact(item["archive"], BASE + "/parent-post-bootstrap-resource-route-ROOT-review-20261008/before/" + item["path"], "exact preserved Source archive")
        raw_pin({"path": item["archive"], "bytes": item["bytes"], "sha256": item["sha256"]}, root, 262144)
        if item["path"] in (BASE + "/parent-direct-science-account.ps1", "tools/banc_m3_conserved_resource_native_core.ps1",
                            "tools/banc_m3_conserved_resource_native_epoch.ps1"):
            exact(old_freeze["sources"][item["path"]], {k: item[k] for k in ("path", "bytes", "sha256")},
                  "prior writer operations used old writer, never new allocator")


def _f_before_metadata(event, root, legacy_validate):
    exact(set(event), {"schema", "event_id", "scope", "before_ledger", "historical_closed282", "initial_policy", "transfers",
                       "j_grant", "direct_human_auto_authority", "root_decision", "reviewed_source", "astra_review",
                       "administrative_observations", "failed_bootstrap", "source_before", "qualification_plan",
                       "old_caps", "new_caps", "delta_s", "combined_global_s", "new_resource_grant_s", "superseded_administrative_refusal"}, "exact separate fixed F event")
    original = read_pin(F_ORIGINAL_EVENT_PIN, root)
    exact(original["schema"], "flylab.m3.conserved-resource-prequalification.v1", "immutable original F event")
    expected = dict(original, schema=F_SCHEMA, event_id="conserved-prequalification-20261008-f-fixed-600",
                    reviewed_source=event["reviewed_source"], superseded_administrative_refusal=F_REFUSAL_PIN)
    exact(event, expected, "only three replaced fields and one actual-refusal field in fixed companion")
    f_administrative_refusal(root)
    for key, value in (("schema", F_SCHEMA), ("event_id", "conserved-prequalification-20261008-f-fixed-600"), ("scope", "PREQUALIFICATION_ONLY"),
                       ("before_ledger", F_CLOSED_PIN), ("historical_closed282", HISTORICAL_PIN), ("initial_policy", POLICY_PIN),
                       ("j_grant", J_PIN), ("direct_human_auto_authority", HUMAN_PIN), ("astra_review", F_ASTRA_PINS),
                       ("administrative_observations", F_ADMIN_PIN), ("failed_bootstrap", F_FAILED_PINS), ("source_before", F_BEFORE_SOURCE_PIN),
                       ("old_caps", OLD_CAPS), ("new_caps", NEW_CAPS), ("delta_s", DELTA), ("combined_global_s", 17900),
                       ("new_resource_grant_s", 0), ("qualification_plan", f_plan(root)), ("superseded_administrative_refusal", F_REFUSAL_PIN)):
        exact(event[key], value, "F event " + key)
    exact(event["root_decision"], {"source": F_ROOT_PIN, "pool_transfer_s": 600, "human_numerical_approval": False}, "new ROOT choice, no numerical human grant")
    # Reuse the immutable literal authority check without interpreting v1 bootstrap as passed.
    authority = read_pin(HUMAN_PIN, root)
    exact(authority, {"schema": "flylab.m3.contact-metadata.current-human-autonomy.v1", "direct_human_authority": True,
                      "automatic_authority": True, "new_resource_grant_s": 0,
                      "literal_latest_user": "앞으로 내 승인 받지 말고 스스로 판단해서 진행해줘. 개입할 상황이면 내가 알아서 개입할게"}, "literal delegation only")
    before, old = read_pin(F_CLOSED_PIN, root, 524288), read_pin(HISTORICAL_PIN, root, 524288)
    need(ALLOCATION_KEY not in before and legacy_validate(before, root) == 964, "real old-cap closed283, no reset clone")
    known_closed(before)
    prefix(before, old)
    need(len(before["debits"]) == 283 and len(before["direct_science_calls"]) == 165
         and len(old["debits"]) == 282 and len(old["direct_science_calls"]) == 164, "entire282 plus sole actual failed bootstrap")
    for key, value in old.items():
        if key not in TAIL_KEYS:
            exact(before[key], value, "immutable old field " + key)
    for key, value in OLD_CAPS.items():
        exact(before[CAP_FIELDS[key]], value, "strict old paired caps")
    exact(before["direct_science_policy"], POLICY_PIN, "old policy retained")
    exact(event["transfers"], {key: before[key] for key in TRANSFER_KEYS}, "all six exact old transfers")
    exact(before["startup_ray_original_grants"]["startup-ray-original-20261007-j"]["grant"], J_PIN, "unchanged J16200/6300/6000")
    policy, grant = read_pin(POLICY_PIN, root), read_pin(J_PIN, root)
    ordered_totals(before, policy, grant, OLD_CAPS)
    f_failed_history(before, root)
    return before, policy, grant


def prequalification_before(event, root, legacy_validate):
    data = _f_before_metadata(event, root, legacy_validate)
    f_reviewed_sources(event["reviewed_source"], root)
    return data

def allocation_route_before(event_spec, event, root, legacy_validate):
    if event_spec["path"] == ALLOCATION_PATH:
        exact(event["schema"], "flylab.m3.conserved-resource-allocation.v1", "strict old v1 path/schema")
        return allocation_before(event, root, legacy_validate)
    exact(event_spec["path"], F_EVENT_PATH, "only exact second event path")
    exact(event["schema"], F_SCHEMA, "strict F path/schema")
    return prequalification_before(event, root, legacy_validate)


def f_freeze(spec, event, event_spec, root):
    exact(spec["path"], F_SOURCE_PATH, "fresh F freeze only")
    freeze = read_pin(spec, root)
    exact(freeze["schema"], "flylab.m3.contact-learning.source-freeze.v1", "full inherited Source schema")
    exact(freeze["ROOT_review_cycle_complete"], True, "actual current reviewed Source")
    exact(freeze["qualification_epoch"], F_EPOCH, "distinct F epoch")
    exact(freeze["reviewed_source"], event["reviewed_source"], "same current reviewed vector")
    exact(freeze["prequalification_event"], event_spec, "freeze after administrative event, acyclic")
    review = f_reviewed_sources(event["reviewed_source"], root)
    exact(freeze["sources"], review["sources"], "entire same current vector, no legacy-source relabelling")
    return freeze


def f_exact_generated(q, plan, source, root):
    exact(set(q), {"schema", "mode", "call_id", "role", "source_freeze", "shared_original_lock_path",
                   "ledger_path", "limits", "generated_behavior", "generated_argv", "output"}, "exact F generated request, no authorization alias")
    for key, value in (("schema", "flylab.m3.direct-science.supervision.v1"), ("mode", "generated"), ("call_id", plan["call"]),
                       ("role", plan["role"]), ("source_freeze", source), ("shared_original_lock_path", SHARED_LOCK),
                       ("ledger_path", BASE + "/parent-e358-execution-ledger.json"), ("limits", plan["limits"]),
                       ("generated_behavior", "command"), ("generated_argv", plan["argv"]), ("output", plan["output"])):
        exact(q[key], value, "full fixed F request " + key)
    from tools import verify_c_banc_m3_startup_ray as ray
    need(not ray.path(plan["output"], root).exists(), "unused fixed current output before dispatch")


def f_progress(ledger, event, event_spec, source, root):
    from tools.banc_m3_contact_learning_admission import cost_proof
    anchor = read_pin(event["before_ledger"], root, 524288)
    tail = ledger["debits"][len(anchor["debits"]):]
    need(len(tail) <= 6, "no fifth generated/retry/other Original tail")
    completed = min(len(tail), 4)
    for i, debit in enumerate(tail[:4]):
        plan = f_generated_plan(("lint", "math", "gate-a", "gate-b")[i], root)
        exact((debit["cost_id"], debit["mode"]), (plan["call"], "generated"), "ordered exact F fees only")
        cost = read_pin(debit["descriptor"], root)
        request = cost_proof(cost, debit, lambda p: read_pin(p, root), generated=True)
        exact(cost["exit_code"], 0, "a spent failed call blocks every progression/replacement")
        exact(cost["source_freeze"], source, "all F real fees same fresh Source")
        # Historical completed output exists; compare the full fixed request without a fresh-output check.
        expected = {"schema": "flylab.m3.direct-science.supervision.v1", "mode": "generated", "call_id": plan["call"],
                    "role": plan["role"], "source_freeze": source, "shared_original_lock_path": SHARED_LOCK,
                    "ledger_path": BASE + "/parent-e358-execution-ledger.json", "limits": plan["limits"],
                    "generated_behavior": "command", "generated_argv": plan["argv"], "output": plan["output"]}
        exact(request, expected, "complete actual executed F request")
        exact(ledger["direct_science_calls"][plan["call"]],
              {"state": "charged", "mode": "generated", "request": cost["request"], "reserved_s": plan["limits"]["outer_s"]}, "once charged same actual F call")
    for i, debit in enumerate(tail[4:]):
        exact((debit["cost_id"], debit["mode"]), (F_CALLS[i], "original"), "only sole F body then audit")
        cost = read_pin(debit["descriptor"], root)
        request = cost_proof(cost, debit, lambda p: read_pin(p, root))
        exact(cost["exit_code"], 0, "failed body cannot enable audit or repeat")
        exact(cost["source_freeze"], source, "same F physical Source")
        exact(request["call_id"], F_CALLS[i], "actual unchanged F candidate scope")
        exact(ledger["direct_science_calls"][F_CALLS[i]],
              {"state": "charged", "mode": "original", "request": cost["supervisor_request"],
               "reserved_s": (300, 120)[i]}, "once charged actual F body/audit")
    f_freeze(source, event, event_spec, root)
    return completed, len(tail) - completed, anchor


def validate_f_qualification(qualification, ledger, root, legacy_validate):
    exact(current_caps(ledger, root, legacy_validate), NEW_CAPS, "paired F caps")
    event_spec = ledger[ALLOCATION_KEY]
    exact(event_spec["path"], F_EVENT_PATH, "new qualification cannot release old v1")
    event = read_pin(event_spec, root)
    exact(set(qualification), {"schema", "epoch", "old_E360", "allocation", "reviewed_source", "qualification_roster",
                               "source_freeze", "actual_costs", "tests", "evidence"}, "exact F qualification")
    for key, value in (("schema", "flylab.m3.direct-science.qualification.v3"), ("epoch", F_EPOCH), ("old_E360", "UNPASSED"),
                       ("allocation", event_spec), ("reviewed_source", event["reviewed_source"]), ("qualification_roster", ROSTER_PIN)):
        exact(qualification[key], value, "F qualification " + key)
    source = qualification["source_freeze"]
    completed, _bodies, anchor = f_progress(ledger, event, event_spec, source, root)
    exact(completed, 4, "all four genuine successful costs, no subset")
    exact(qualification["actual_costs"], [d["descriptor"] for d in ledger["debits"][len(anchor["debits"]):len(anchor["debits"]) + 4]],
          "every exact actual once cost")
    need(type(qualification["tests"]) is list and len(qualification["tests"]) == 3, "all three complete successful JUnits")
    roster = complete_roster(root)
    for item, role in zip(qualification["tests"], ("math", "gate-a", "gate-b"), strict=True):
        rows = roster[role.replace("-", "_")]
        exact(set(item), {"file", "node_ids", "names", "classnames", "count"}, "exact complete F JUnit descriptor")
        exact(item["file"]["path"], BASE + "/parent-contact-learning-f-" + role + "-junit.xml", "new actual F JUnit path")
        for key, value in (("node_ids", [r["node_id"] for r in rows]), ("names", [r["name"] for r in rows]),
                           ("classnames", [r["classname"] for r in rows]), ("count", len(rows))):
            exact(item[key], value, "full F JUnit " + key)
        successful_junit(item["file"], rows, root)
    exact(qualification["evidence"], qualification["actual_costs"], "four actual cost/owner/trace evidence, no free qualified boolean")
    return len(anchor["debits"]) + 4


def prequalification_dispatch(ledger, q, request_spec, root, legacy_validate, *, phase):
    """Global guard BEFORE any Original Admission; exact closed/admitted states."""
    from tools import banc_m3_checkpoint_harness_runtime as harness
    from tools import banc_m3_checkpoint_runtime as checkpoint
    if harness.KEY in ledger:
        checkpoint = harness
    if checkpoint.KEY in ledger:
        return checkpoint.dispatch(ledger, q, request_spec, root, phase=phase)
    if J_KEY in ledger:
        return j_dispatch(ledger, q, request_spec, root, legacy_validate, phase=phase)
    if I_KEY in ledger:
        return i_dispatch(ledger, q, request_spec, root, legacy_validate, phase=phase)
    if H_KEY in ledger:
        return h_dispatch(ledger, q, request_spec, root, legacy_validate, phase=phase)
    if G_KEY in ledger:
        return g_dispatch(ledger, q, request_spec, root, legacy_validate, phase=phase)
    # Reject missing-pointer new/mixed caps before any request/science access.
    current_caps(ledger, root, legacy_validate)
    if ALLOCATION_KEY not in ledger:
        return None
    if ledger[ALLOCATION_KEY]["path"] != F_EVENT_PATH:
        return None  # strict old v1 is unchanged
    need(phase in ("closed", "admitted"), "exact dispatch phase")
    event_spec = ledger[ALLOCATION_KEY]
    event = read_pin(event_spec, root)
    source = q["source_freeze"]
    completed, bodies, anchor = f_progress(ledger, event, event_spec, source, root)
    if q["mode"] == "generated":
        need(completed < 4 and bodies == 0, "only next unused fixed full F qualification; no fifth or retry")
        plan = f_generated_plan(("lint", "math", "gate-a", "gate-b")[completed], root)
        f_exact_generated(q, plan, source, root)
    else:
        exact(q["mode"], "original", "finite original mode")
        need(completed == 4 and bodies < 2, "all4 before any Original; only single body/audit")
        qualification_spec = ledger.get("direct_science_qualification")
        need(type(qualification_spec) is dict and qualification_spec["path"] == F_QUAL_PATH, "legacy qualification never releases F")
        qualification = read_pin(qualification_spec, root)
        validate_f_qualification(qualification, ledger, root, legacy_validate)
        exact(qualification["source_freeze"], source, "same qualified F Source")
        exact(q["call_id"], F_CALLS[bodies], "only current same-candidate F body then conditional audit")
        exact(q["role"], ("contact_candidate_body", "contact_body_audit")[bodies], "no saved/tracking/replay/geometry/legacy bypass")
        exact(set(q), {"schema", "mode", "call_id", "role", "source_freeze", "shared_original_lock_path",
                       "ledger_path", "limits", "output", "science_request", "grant"}, "exact F Original supervisor fields")
        exact(q["schema"], "flylab.m3.direct-science.supervision.v1", "same actual supervisor protocol")
        exact(q["grant"], J_PIN, "immutable J no restore/rebind")
        contract = read_pin(read_pin(F_DESCRIPTOR_PIN, root)["contract"], root)
        exact(q["limits"], {key: contract["limits"][q["role"]][key] for key in
                           ("outer_s", "child_s", "cleanup_s", "host_bytes", "output_cap_bytes")}, "unchanged actual body/audit limits")
        exact(q["output"], BASE + "/parent-" + q["call_id"] + "/result-supervision", "fresh same F output")
        exact(q["shared_original_lock_path"], SHARED_LOCK, "common F exclusion")
        exact(q["ledger_path"], BASE + "/parent-e358-execution-ledger.json", "same actual F ledger")
    from tools import verify_c_banc_m3_startup_ray as ray
    exact(ray.json_load(raw_pin(request_spec, root)), q, "actual descriptor/current dispatch request")
    current = {q["call_id"]: {"state": "admitted", "mode": q["mode"], "request": request_spec, "reserved_s": q["limits"]["outer_s"]}}
    expected_calls = dict(anchor["direct_science_calls"])
    for debit in ledger["debits"][len(anchor["debits"]):]:
        expected_calls[debit["cost_id"]] = ledger["direct_science_calls"][debit["cost_id"]]
    if phase == "admitted":
        expected_calls.update(current)
    exact(ledger["direct_science_calls"], expected_calls, "no extra admitted alias/stale charged call")
    for key in OWNER_KEYS:
        expected = {q["call_id"]: {"request": request_spec, "reserved_s": q["limits"]["outer_s"]}} if (
            phase == "admitted" and q["mode"] == "generated" and key == "direct_science_reservations") else {}
        exact(ledger[key], expected, "exact phase/own reservation/worker/pending/HOLD " + key)
    exact(ledger["original_jobs_running"], 0, "no worker before actual spawn")
    exact(ledger["original_reserved_s"], 0, "Original reserve starts at actual begin")
    need(not ray.path(q["output"], root).exists(), "unused fixed actual output")
    return {"completed": completed, "bodies": bodies, "epoch": F_EPOCH}


# Dedicated immutable qualification amendment; never a third allocation route.
G_KEY = "direct_science_qualification_amendment"
G_PATH = BASE + "/parent-contact-body-runtime-20261008-g.json"
G_AMENDMENT_PATH = BASE + "/parent-contact-qualification-amendment-20261008-g.json"
G_REVIEW_PATH = BASE + "/parent-contact-qualification-reviewed-source-20261008-g.json"
G_SOURCE_PATH = BASE + "/parent-contact-learning-source-freeze-20261008-g.json"
G_QUAL_PATH = BASE + "/parent-contact-learning-qualified-20261008-g.json"
G_EPOCH = "contact-qualification-repair-20261008-g"
G_QUAL_CALLS = tuple("direct-science-20261008-contact-g-" + r for r in ("lint", "math", "gate-a", "gate-b"))
G_CALLS = ("contact-learning-20261008-g-body", "contact-learning-20261008-g-body-audit")
G_REPLACEMENTS = (HELPER_SOURCE, "tools/banc_m3_conserved_resource_native_epoch.ps1",
                  BASE + "/parent-direct-science-account.ps1", "tools/banc_m3_startup_ray_supervisor.py",
                  "tools/banc_m3_contact_learning_admission.py", BASE + "/parent-contact-learning-contract-native.ps1",
                  "tests/test_c_banc_contact_runtime_journal.py")
G_BEFORE_PIN = {"path": BASE + "/parent-g-before-20261008/source-before.json", "bytes": 12295,
                "sha256": "dbda3224e0c51aec89ba679fd11557b5174b6b170bf6494ba9c943fc6b0689f6"}
G_ROOT_PIN = {"path": "docs/BANC_M3_G_QUALIFICATION_REPAIR_ROOT_REFLECTED_20261008.md", "bytes": 14189,
              "sha256": "66924e4d50c435bdf36e9712357e0e8e78b2e0b20ad63fa660ee69195baa8ed5"}
G_ASTRA_PINS = {
    "markdown": {"path": BASE + "/parent-g-qualification-repair-astra-review-20261008/review.md", "bytes": 14953,
                 "sha256": "4bd957e57b40f3196c1a6873b3ff352889319fdd11696030e009116758956f60"},
    "json": {"path": BASE + "/parent-g-qualification-repair-astra-review-20261008/review.json", "bytes": 18390,
             "sha256": "7b62e24bca5e151444853df69bf6de12dd473a956a961feac13a245e116407e0"},
}
G_CLOSED_PIN = {"path": BASE + "/parent-f-lint-closed284-20261008.json", "bytes": 455909,
                "sha256": "cdfd6e6653c2b6f3a08e0cb9ae60c5947fd0bc0d2a21bdf21b628e5ca64c86d8"}
G_F_PATHS = {"allocation": F_EVENT_PATH, "review": F_REVIEW_PATH, "freeze": F_SOURCE_PATH,
             "trace": BASE + "/parent-direct-science-20261008-contact-f-lint-genuine-return.json",
             "request": BASE + "/parent-direct-science-20261008-contact-f-lint-request.json",
             "admitted": BASE + "/parent-f-lint-actual-admitted455814-20261008.json",
             "outer": BASE + "/parent-direct-science-20261008-contact-f-lint/outer.json",
             "cost": BASE + "/parent-direct-science-20261008-contact-f-lint-full-return.json",
             "result": BASE + "/parent-f-lint-settlement-ROOT-actual-result-20261008.json"}


def g_archives(root):
    before = read_pin(G_BEFORE_PIN, root)
    exact(set(before), {"schema", "source_metadata_only", "actual_f_lint_once_settled", "authoring_note",
                        "scientific_execution", "registration_committed", "snapshots", "closed284"}, "exact pre-G archive map")
    for key, value in (("schema", "flylab.m3.g-qualification-repair.source-before.v1"), ("source_metadata_only", True),
                       ("actual_f_lint_once_settled", True), ("scientific_execution", "NOT_RUN"),
                       ("registration_committed", False), ("closed284", G_CLOSED_PIN)):
        exact(before[key], value, "actual pre-G map " + key)
    need(type(before["snapshots"]) is list and len(before["snapshots"]) == 23, "all23 immutable pre-G mappings")
    mapped = {}
    for i, item in enumerate(before["snapshots"]):
        exact(set(item), {"source", "archive"}, "exact historical source/archive mapping")
        source, archive = item["source"], item["archive"]
        exact(set(source), {"path", "bytes", "sha256"}, "exact historical descriptor")
        need(source["path"] not in mapped, "unique historical mapping")
        exact(archive, dict(source, path=BASE + "/parent-g-before-20261008/" + f"{i:02d}-" + source["sha256"][:16] + ".txt"),
              "literal full-content numbered pre-G archive")
        raw_pin(archive, root, 524288)
        mapped[source["path"]] = item
    need(all(n in mapped for n in (*G_REPLACEMENTS, *G_F_PATHS.values(), F_PATH, G_CLOSED_PIN["path"])), "complete fixed G/F history map")
    return mapped


def historical_f_reviewed_sources(root):
    """Only immutable F provenance/archive content; never live old Source success."""
    mapped = g_archives(root)
    review = _f_review_metadata(mapped[F_REVIEW_PATH]["source"], root)
    exact(len(review["sources"]), 723, "whole historical F723")
    for name in G_REPLACEMENTS:
        exact(review["sources"][name], mapped[name]["source"], "exact superseded F Source identity")
        raw_pin(mapped[name]["archive"], root, BOOTSTRAP_SOURCE_CAP)
    # Unrelated/installed identities remain in the immutable F vector. Full live
    # G checking cold-reads them; accounting never requires live Source success.
    return review, mapped


def g_generated_plan(role, root):
    old = f_generated_plan(role, root)
    index = ("lint", "math", "gate-a", "gate-b").index(role)
    argv = list(old["argv"])
    if role != "lint":
        argv[-1] = "--junitxml=" + BASE + "/parent-contact-learning-g-" + role + "-junit.xml"
    return dict(old, call=G_QUAL_CALLS[index], argv=argv, output=BASE + "/parent-" + G_QUAL_CALLS[index])


def g_plan(root):
    return {"epoch": G_EPOCH, "roster_epoch": EPOCH, "qualification_roster": ROSTER_PIN,
            "calls": [g_generated_plan(r, root) for r in ("lint", "math", "gate-a", "gate-b")],
            "generated_close_s": 20, "original_release_calls": list(G_CALLS)}


def g_review_metadata(spec, root, historical, mapped):
    exact(spec["path"], G_REVIEW_PATH, "one actual reviewed G manifest")
    review = read_pin(spec, root)
    exact(set(review), {"schema", "epoch", "sources", "replacements", "root_reflection", "before_source",
                        "astra_review", "root_reviews", "sol_reports"}, "exact acyclic G review fields")
    for key, value in (("schema", "flylab.m3.contact-qualification.reviewed-source.v1"), ("epoch", G_EPOCH),
                       ("root_reflection", G_ROOT_PIN), ("before_source", G_BEFORE_PIN), ("astra_review", G_ASTRA_PINS)):
        exact(review[key], value, "reviewed G " + key)
    for pin in (G_ROOT_PIN, *G_ASTRA_PINS.values()):
        raw_pin(pin, root, 65536)
    root_names = {"docs/BANC_M3_G_QUALIFICATION_REPAIR_ROOT_" + r + "_20261008.md" for r in ("FIRST_REVIEW", "SECOND_REVIEW", "RESIDUAL")}
    pins = review["root_reviews"]
    need(type(pins) is list and 2 <= len(pins) <= 3, "actual ROOT FIRST and SECOND, optional residual")
    paths = [p["path"] for p in pins]
    need(len(set(paths)) == len(paths) and set(paths) <= root_names
         and root_names - {"docs/BANC_M3_G_QUALIFICATION_REPAIR_ROOT_RESIDUAL_20261008.md"} <= set(paths), "actual final ROOT reviews")
    sol_names = {BASE + "/parent-g-qualification-repair-sol-" + r + "-20261008/" + r + "." + ext
                 for r in ("FIRST", "R1") for ext in ("md", "json")}
    sol = review["sol_reports"]
    need(type(sol) is list and len(sol) in (2, 4) and len({p["path"] for p in sol}) == len(sol)
         and {p["path"] for p in sol} <= sol_names
         and {BASE + "/parent-g-qualification-repair-sol-FIRST-20261008/FIRST." + ext for ext in ("md", "json")} <= {p["path"] for p in sol},
         "actual sole Sol FIRST/optional R1 reports")
    for pin in (*pins, *sol):
        raw_pin(pin, root, 1048576)
    vector, replaced = review["sources"], review["replacements"]
    need(type(vector) is dict and type(replaced) is dict, "typed G vector/replacements")
    exact(set(vector), set(historical["sources"]) | {G_PATH}, "all inherited723 plus only static G key")
    exact(set(replaced), set(G_REPLACEMENTS), "only seven reviewed code replacements")
    for name, pin in vector.items():
        exact(set(pin), {"path", "bytes", "sha256"}, "exact current G descriptor")
        exact(pin["path"], name, "literal G Source key")
        need(type(pin["bytes"]) is int and 0 <= pin["bytes"] <= 2097152 and type(pin["sha256"]) is str
             and re.fullmatch("[0-9a-f]{64}", pin["sha256"]), "unchanged finite Source cap")
        if name in G_REPLACEMENTS:
            need(0 < pin["bytes"] <= BOOTSTRAP_SOURCE_CAP, "existing finite changed Source cap")
            exact(replaced[name], {"before": mapped[name]["source"], "after": pin}, "exact reviewed replacement before/after")
        elif name != G_PATH:
            exact(pin, historical["sources"][name], "all unrelated/installed pins unchanged")
    return review


def g_body(spec, root):
    exact(spec["path"], G_PATH, "one static G body")
    prior = read_pin(F_DESCRIPTOR_PIN, root)
    mapped = g_archives(root)
    expected = dict(prior, schema="flylab.m3.contact-learning.body-runtime.v4", retry_id="contact-learning-20261008-g",
                    calls=list(G_CALLS), qualification_epoch=G_EPOCH, source_review_path=G_REVIEW_PATH,
                    source_freeze_path=G_SOURCE_PATH, qualification_amendment_path=G_AMENDMENT_PATH,
                    historical_closed284=G_CLOSED_PIN, failed_f_lint=mapped[G_F_PATHS["cost"]]["source"])
    body = read_pin(spec, root)
    exact(body, expected, "entire unchanged F candidate/history/limits and literal G scope")
    return body


def g_current_review(spec, root):
    historical, mapped = historical_f_reviewed_sources(root)
    review = g_review_metadata(spec, root, historical, mapped)
    for pin in review["sources"].values():
        source_text(pin, root)
    g_body(review["sources"][G_PATH], root)
    return review


def f_request_identity(q, plan, source):
    expected = {"schema": "flylab.m3.direct-science.supervision.v1", "mode": "generated", "call_id": plan["call"],
                "role": plan["role"], "source_freeze": source, "shared_original_lock_path": SHARED_LOCK,
                "ledger_path": BASE + "/parent-e358-execution-ledger.json", "limits": plan["limits"],
                "generated_behavior": "command", "generated_argv": plan["argv"], "output": plan["output"]}
    exact(q, expected, "full literal generated qualification request")


def g_history(root, legacy_validate):
    """Exact historical allocation/closed284/failure proof, no live G validator."""
    historical, mapped = historical_f_reviewed_sources(root)
    pins = {k: mapped[v]["source"] for k, v in G_F_PATHS.items()}
    event = read_pin(pins["allocation"], root)
    exact(event["reviewed_source"], pins["review"], "unchanged actual F allocation/review")
    before, policy, grant = _f_before_metadata(event, root, legacy_validate)
    allocated = allocated_snapshot(before, pins["allocation"], policy, grant)
    freeze = read_pin(pins["freeze"], root)
    exact(freeze["schema"], "flylab.m3.contact-learning.source-freeze.v1", "actual historical F freeze")
    exact(freeze["ROOT_review_cycle_complete"], True, "actual F review status, never qualification")
    exact(freeze["qualification_epoch"], F_EPOCH, "actual F epoch")
    exact(freeze["reviewed_source"], pins["review"], "actual F freeze/review")
    exact(freeze["prequalification_event"], pins["allocation"], "actual allocated F freeze")
    exact(freeze["sources"], historical["sources"], "every immutable as-executed F Source")
    admitted = read_pin(pins["admitted"], root, 524288)
    q = read_pin(pins["request"], root)
    f_request_identity(q, f_generated_plan("lint", root), pins["freeze"])
    expected_admitted = copy.deepcopy(allocated)
    expected_admitted["direct_science_calls"][F_QUAL_CALLS[0]] = {"state": "admitted", "mode": "generated", "request": pins["request"], "reserved_s": 15}
    expected_admitted["direct_science_reservations"] = {F_QUAL_CALLS[0]: {"request": pins["request"], "reserved_s": 15}}
    exact(admitted, expected_admitted, "actual admitted283/166 snapshot, not mutable live observation")
    anchor = read_pin(G_CLOSED_PIN, root, 524288)
    known_closed(anchor)
    prefix(anchor, allocated)
    need(len(anchor["debits"]) == 284 and len(anchor["direct_science_calls"]) == 166, "real once closed284/166")
    exact(set(anchor), set(allocated), "no historical extra cap/pointer")
    for key, value in allocated.items():
        if key not in TAIL_KEYS:
            exact(anchor[key], value, "immutable F allocation field")
    debit = anchor["debits"][-1]
    exact(debit, {"cost_id": F_QUAL_CALLS[0], "mode": "generated", "actual_outer_full_return_s": 1.755,
                  "actual_returned_exit_code": 1, "descriptor": pins["cost"]}, "sole genuine paid F failure")
    from tools.banc_m3_contact_learning_admission import cost_proof
    cost = read_pin(pins["cost"], root)
    exact(cost_proof(cost, debit, lambda p: read_pin(p, root), generated=True), q, "actual F failed request proof")
    for key in ("trace", "request", "outer"):
        exact(cost[key], pins[key], "immutable F once settlement " + key)
    exact(cost["source_freeze"], pins["freeze"], "immutable failed F Source")
    exact(anchor["direct_science_calls"], dict(allocated["direct_science_calls"], **{
        F_QUAL_CALLS[0]: {"state": "charged", "mode": "generated", "request": pins["request"], "reserved_s": 15}}), "all165 plus sole once charge")
    outer = read_pin(pins["outer"], root)
    exact(outer["origin"], "ROOT_ADMINISTRATIVE_RECONSTRUCTION", "actual ROOT witness, never supervisor receipt")
    exact(outer["current_ledger"], dict(pins["admitted"], path=BASE + "/parent-e358-execution-ledger.json"), "historical ledger observation binds admitted snapshot")
    exact(outer["immutable_admitted_ledger"], pins["admitted"], "immutable actual admitted identity")
    for key, value in (("job_created", False), ("worker_started", False), ("worker_pid", None), ("worker_creation_ticks", None),
                       ("stream_state", "NO_WORKER_STREAMS_OPENED"), ("qualification", "NOT_QUALIFIED")):
        exact(outer[key], value, "actual F pre-Job " + key)
    result = read_pin(pins["result"], root)
    exact(result["actual_administrative_commit"]["first_final_tool"]["exit_code"], 0, "actual unchanged canonical once commit")
    exact(result["independent_actual_readback"]["closed284"], G_CLOSED_PIN, "actual ROOT closed284")
    exact(result["independent_actual_readback"]["actual_cost"], pins["cost"], "actual ROOT cost")
    ordered_totals(anchor, policy, grant, NEW_CAPS)
    return anchor, policy, grant, historical, mapped, pins


def g_amendment(spec, root, legacy_validate):
    exact(spec["path"], G_AMENDMENT_PATH, "one exact immutable G pointer path")
    amendment = read_pin(spec, root)
    anchor, policy, grant, historical, mapped, pins = g_history(root, legacy_validate)
    exact(set(amendment), {"schema", "epoch", "scope", "allocation", "historical_closed284", "failed_f_lint", "source_before",
                          "reviewed_source", "qualification_plan", "new_resource_grant_s", "refund_s"}, "exact G amendment fields")
    for key, value in (("schema", "flylab.m3.contact-qualification-amendment.v1"), ("epoch", G_EPOCH), ("scope", "PREQUALIFICATION_ONLY"),
                       ("allocation", pins["allocation"]), ("historical_closed284", G_CLOSED_PIN), ("failed_f_lint", pins),
                       ("source_before", G_BEFORE_PIN), ("qualification_plan", g_plan(root)), ("new_resource_grant_s", 0), ("refund_s", 0)):
        exact(amendment[key], value, "immutable G amendment " + key)
    g_review_metadata(amendment["reviewed_source"], root, historical, mapped)
    return amendment, anchor, policy, grant


def g_registration(ledger, spec, root, legacy_validate, *, before_raw):
    need(type(before_raw) is bytes and before_raw == raw_pin(G_CLOSED_PIN, root, 524288),
         "actual immutable closed284 raw before pointer registration")
    need(G_KEY not in ledger, "no duplicate/alias G registration")
    amendment, anchor, _policy, _grant = g_amendment(spec, root, legacy_validate)
    known_closed(ledger)
    exact(ledger, anchor, "fresh full exact closed284 before one pointer")
    need(ledger["direct_science_generated_used_s"] + 300 <= 1564 and ledger["generated_shared_remaining_s"] >= 300, "full280 plus separate close20 in both finite pools")
    g_current_review(amendment["reviewed_source"], root)
    after = copy.deepcopy(ledger)
    after[G_KEY] = copy.deepcopy(spec)
    exact(set(after) - set(ledger), {G_KEY}, "sole immutable qualification pointer delta")
    return after


def g_current_caps(ledger, root, legacy_validate):
    amendment, anchor, policy, grant = g_amendment(ledger[G_KEY], root, legacy_validate)
    exact(ledger[ALLOCATION_KEY], amendment["allocation"], "original F allocation unchanged after G")
    expected = dict(anchor, **{G_KEY: ledger[G_KEY]})
    exact(set(ledger), set(expected), "strict G expected keys, pointer not a mutable-tail exception")
    prefix(ledger, anchor)
    for key, value in expected.items():
        if key not in TAIL_KEYS:
            exact(ledger[key], value, "immutable G historical accounting " + key)
    ordered_totals(ledger, policy, grant, NEW_CAPS)
    return dict(NEW_CAPS)


def g_freeze(spec, amendment, amendment_spec, root):
    exact(spec["path"], G_SOURCE_PATH, "fresh literal G freeze")
    freeze = read_pin(spec, root)
    historical = g_archives(root)
    prior = read_pin(historical[F_SOURCE_PATH]["source"], root)
    expected = dict(prior, qualification_epoch=G_EPOCH, reviewed_source=amendment["reviewed_source"],
                    qualification_amendment=amendment_spec, sources=freeze["sources"], body_retry=freeze["body_retry"])
    exact(freeze, expected, "whole inherited F metadata plus fixed G fields; historical markers remain historical")
    for key, value in (("schema", "flylab.m3.contact-learning.source-freeze.v1"), ("ROOT_review_cycle_complete", True),
                       ("qualification_epoch", G_EPOCH), ("reviewed_source", amendment["reviewed_source"]), ("qualification_amendment", amendment_spec)):
        exact(freeze[key], value, "actual G freeze " + key)
    review = g_current_review(amendment["reviewed_source"], root)
    exact(freeze["sources"], review["sources"], "entire current G724, no legacy Source substitution")
    exact(freeze["body_retry"], review["sources"][G_PATH], "one reviewed static G descriptor")
    return freeze


def _epoch_progress(ledger, amendment, amendment_spec, source, root, family):
    from tools.banc_m3_contact_learning_admission import cost_proof
    f = _qualification_family(family)
    tail = ledger["debits"][f["offset"]:]
    need(len(tail) <= 6, "no fifth G role/retry/other Original tail")
    completed = min(len(tail), 4)
    for i, debit in enumerate(tail):
        generated = i < 4
        call = f["qual_calls"][i] if generated else f["calls"][i - 4]
        exact((debit["cost_id"], debit["mode"]), (call, "generated" if generated else "original"), "exact ordered G tail")
        cost = read_pin(debit["descriptor"], root)
        q = cost_proof(cost, debit, lambda p: read_pin(p, root), generated=generated)
        exact(cost["exit_code"], 0, "spent G failure blocks every later role/replacement/retry")
        exact(cost["source_freeze"], source, "same genuinely executed G Source")
        if generated:
            plan = f["plan"](("lint", "math", "gate-a", "gate-b")[i], root)
            f_request_identity(q, plan, source)
            owner = cost["request"]
            reserved = plan["limits"]["outer_s"]
        else:
            exact(q["call_id"], call, "sole same-candidate G body then conditional audit")
            owner, reserved = cost["supervisor_request"], (300, 120)[i - 4]
        exact(ledger["direct_science_calls"][call], {"state": "charged", "mode": debit["mode"], "request": owner, "reserved_s": reserved}, "once charged exact G tail")
    f["freeze"](source, amendment, amendment_spec, root)
    return completed, max(0, len(tail) - 4)


def _epoch_qualification(qualification, ledger, root, legacy_validate, family):
    f = _qualification_family(family)
    exact(f["caps"](ledger, root, legacy_validate), NEW_CAPS, "strict G/H historical caps before release")
    amendment, _anchor, _policy, _grant = f["metadata"](ledger[f["key"]], root, legacy_validate)
    keys = {"schema", "epoch", "old_E360", "allocation", "qualification_amendment", "reviewed_source", "qualification_roster",
            "source_freeze", "actual_costs", "tests", "evidence"}
    if family in ("h", "i", "j"):
        keys.add("qualification_lint_repair")
        exact(qualification["qualification_lint_repair"], ledger[H_KEY], "same immutable H pointer")
    if family in ("i", "j"):
        keys.add("preflight_recovery")
        exact(qualification["preflight_recovery"], ledger[I_KEY], "same immutable I pointer")
    if family == "j":
        keys.add("qualification_lint_cleanup")
        exact(qualification["qualification_lint_cleanup"], ledger[J_KEY], "same immutable J cleanup pointer")
    exact(set(qualification), keys, "exact G/H qualification")
    for key, value in (("schema", f["schema"]), ("epoch", f["epoch"]), ("old_E360", "UNPASSED"),
                       ("allocation", ledger[ALLOCATION_KEY]), ("qualification_amendment", ledger[G_KEY]),
                       ("reviewed_source", amendment["reviewed_source"]), ("qualification_roster", ROSTER_PIN)):
        exact(qualification[key], value, "G qualification " + key)
    completed, _bodies = _epoch_progress(ledger, amendment, ledger[f["key"]], qualification["source_freeze"], root, family)
    exact(completed, 4, "all four fresh full G costs required")
    exact(qualification["actual_costs"], [d["descriptor"] for d in ledger["debits"][f["offset"]:f["end"]]], "four exact once real fees")
    exact(qualification["evidence"], qualification["actual_costs"], "actual cost/owner/trace evidence")
    need(type(qualification["tests"]) is list and len(qualification["tests"]) == 3, "three complete G JUnits")
    roster = complete_roster(root)
    for item, role in zip(qualification["tests"], ("math", "gate-a", "gate-b"), strict=True):
        rows = roster[role.replace("-", "_")]
        exact(set(item), {"file", "node_ids", "names", "classnames", "count"}, "complete G JUnit fields")
        exact(item["file"]["path"], BASE + "/parent-contact-learning-" + family + "-" + role + "-junit.xml", "fresh literal G/H JUnit")
        for key, value in (("node_ids", [r["node_id"] for r in rows]), ("names", [r["name"] for r in rows]),
                           ("classnames", [r["classname"] for r in rows]), ("count", len(rows))):
            exact(item[key], value, "complete unchanged G JUnit " + key)
        successful_junit(item["file"], rows, root)
    return f["end"]


def _epoch_dispatch(ledger, q, request_spec, root, legacy_validate, family, *, phase):
    f = _qualification_family(family)
    f["caps"](ledger, root, legacy_validate)  # accounting alone never permits science
    need(phase in ("closed", "admitted"), "exact G dispatch phase")
    amendment, anchor, _policy, _grant = f["metadata"](ledger[f["key"]], root, legacy_validate)
    source = q["source_freeze"]
    completed, bodies = _epoch_progress(ledger, amendment, ledger[f["key"]], source, root, family)
    if q["mode"] == "generated":
        need(completed < 4 and bodies == 0, "only next unused full G role, no fifth/subset/retry")
        plan = f["plan"](("lint", "math", "gate-a", "gate-b")[completed], root)
        f_request_identity(q, plan, source)
        if completed == 0:
            envelope = f["envelope"]
            need(ledger["direct_science_generated_used_s"] + envelope <= 1564 and ledger["generated_shared_remaining_s"] >= envelope, "entire fixed family plan plus distinct close20")
    else:
        exact(q["mode"], "original", "finite G Original mode")
        need(completed == 4 and bodies < 2, "global Original block before every legacy-specific input")
        qualified = ledger.get("direct_science_qualification")
        need(type(qualified) is dict and qualified["path"] == f["qual_path"], "old qualification cannot release current G/H")
        qualification = read_pin(qualified, root)
        _epoch_qualification(qualification, ledger, root, legacy_validate, family)
        exact(qualification["source_freeze"], source, "same qualified G Source")
        exact(set(q), {"schema", "mode", "call_id", "role", "source_freeze", "shared_original_lock_path", "ledger_path", "limits", "output", "science_request", "grant"}, "exact G Original envelope")
        exact(q["call_id"], f["calls"][bodies], "only literal G/H candidate body/audit")
        exact(q["role"], ("contact_candidate_body", "contact_body_audit")[bodies], "all legacy Original roles stay blocked")
        exact(q["schema"], "flylab.m3.direct-science.supervision.v1", "same actual G supervisor protocol")
        exact(q["grant"], J_PIN, "same immutable J lineage")
        freeze = read_pin(source, root)
        contract = read_pin(f["body"](freeze["body_retry"], root)["contract"], root)
        exact(q["limits"], {k: contract["limits"][q["role"]][k] for k in ("outer_s", "child_s", "cleanup_s", "host_bytes", "output_cap_bytes")}, "unchanged G body/audit limits")
        exact(q["output"], BASE + "/parent-" + q["call_id"] + "/result-supervision", "fresh G physical output")
        exact(q["shared_original_lock_path"], SHARED_LOCK, "same G exclusion")
        exact(q["ledger_path"], BASE + "/parent-e358-execution-ledger.json", "same actual G ledger")
    exact(read_pin(request_spec, root), q, "actual exact G dispatch request")
    calls = dict(anchor["direct_science_calls"])
    for debit in ledger["debits"][f["offset"]:]:
        calls[debit["cost_id"]] = ledger["direct_science_calls"][debit["cost_id"]]
    if phase == "admitted":
        calls[q["call_id"]] = {"state": "admitted", "mode": q["mode"], "request": request_spec, "reserved_s": q["limits"]["outer_s"]}
    exact(ledger["direct_science_calls"], calls, "no G alias/unfinished extra call")
    for key in OWNER_KEYS:
        own = {q["call_id"]: {"request": request_spec, "reserved_s": q["limits"]["outer_s"]}} if (
            phase == "admitted" and q["mode"] == "generated" and key == "direct_science_reservations") else {}
        exact(ledger[key], own, "exact G phase/owner/HOLD " + key)
    exact(ledger["original_jobs_running"], 0, "no G worker before spawn")
    exact(ledger["original_reserved_s"], 0, "no G runtime reserve before begin")
    from tools import verify_c_banc_m3_startup_ray as ray
    need(not ray.path(q["output"], root).exists(), "fresh fixed G output")
    return {"completed": completed, "bodies": bodies, "epoch": f["epoch"]}

def text_new_directory(root, name):
    """Fresh bounded metadata directory for the actual lexical refusal publisher."""
    target = _text_path(root, name, write=True)
    os.mkdir(target)  # CreateNew semantics; any already existing leaf refuses.
    _text_path(root, name + "/outer.json", write=True)
    return target


def _epoch_admitted_identity(ledger, q, request_spec, root, family):
    """Exact admitted G envelope/ownership, without current Source/progress success."""
    f = _qualification_family(family)
    pointer = ledger.get(f["key"])
    need(type(pointer) is dict and pointer.get("path") == f["pointer_path"], "one actual admitted G/H pointer")
    exact(set(pointer), {"path", "bytes", "sha256"}, "exact saved G pointer")
    exact(q["source_freeze"]["path"], f["source_path"], "literal admitted G/H freeze")
    source = q["source_freeze"]
    need(type(source) is dict and set(source) == {"path", "bytes", "sha256"}
         and type(source["bytes"]) is int and 0 < source["bytes"] <= (1048576 if family == "g" else BOOTSTRAP_SOURCE_CAP)
         and type(source["sha256"]) is str and re.fullmatch("[0-9a-f]{64}", source["sha256"]), "exact admitted freeze pin")
    if q["mode"] == "generated":
        need(q["call_id"] in f["qual_calls"], "literal G/H generated refusal family")
        role = ("lint", "math", "gate-a", "gate-b")[f["qual_calls"].index(q["call_id"])]
        f_request_identity(q, f["plan"](role, root), source)
    else:
        exact(q["mode"], "original", "literal G Original refusal")
        need(q["call_id"] in f["calls"], "only actual G/H body/audit")
        i = f["calls"].index(q["call_id"])
        exact(set(q), {"schema", "mode", "call_id", "role", "source_freeze", "shared_original_lock_path", "ledger_path", "limits", "output", "science_request", "grant"}, "exact G Original refusal envelope")
        role = ("contact_candidate_body", "contact_body_audit")[i]
        exact(q["role"], role, "same admitted physical G role")
        exact(q["grant"], J_PIN, "unchanged admitted J")
        contract = read_pin(read_pin(F_DESCRIPTOR_PIN, root)["contract"], root)
        exact(q["limits"], {k: contract["limits"][role][k] for k in ("outer_s", "child_s", "cleanup_s", "host_bytes", "output_cap_bytes")}, "same immutable admitted body/audit bounds")
        exact(q["output"], BASE + "/parent-" + q["call_id"] + "/result-supervision", "literal G Original refusal output")
        exact(q["shared_original_lock_path"], SHARED_LOCK, "same admitted physical lock")
        exact(q["ledger_path"], BASE + "/parent-e358-execution-ledger.json", "same actual ledger")
        exact(q["schema"], "flylab.m3.direct-science.supervision.v1", "same admitted G protocol")
    exact(ledger["direct_science_calls"][q["call_id"]], {"state": "admitted", "mode": q["mode"], "request": request_spec, "reserved_s": q["limits"]["outer_s"]}, "actual admitted G call before lock/Job")
    for key in OWNER_KEYS:
        own = {q["call_id"]: {"request": request_spec, "reserved_s": q["limits"]["outer_s"]}} if (
            q["mode"] == "generated" and key == "direct_science_reservations") else {}
        exact(ledger[key], own, "known initial G no foreign/worker/HOLD " + key)
    exact(ledger["original_jobs_running"], 0, "initial G no Original worker")
    exact(ledger["original_reserved_s"], 0, "initial G no runtime reservation")
    if family in ("h", "i", "j"):
        prior = ledger.get(G_KEY)
        need(type(prior) is dict and set(prior) == {"path", "bytes", "sha256"} and prior.get("path") == G_AMENDMENT_PATH,
             "exact saved immutable G lineage before H refusal")
        repair = read_pin(pointer, root)
        exact(repair["qualification_amendment"], prior, "H refusal preserves the actual saved G pointer")
        if family in ("i", "j"):
            exact(repair["qualification_lint_repair"], ledger[H_KEY], "I refusal preserves the actual saved H pointer")
        if family == "j":
            exact(repair["preflight_recovery"], ledger[I_KEY], "J refusal preserves the actual saved I pointer")
            freeze = read_pin(source, root, BOOTSTRAP_SOURCE_CAP)
            for key, pin in (("qualification_amendment", prior), ("qualification_lint_repair", ledger[H_KEY]),
                             ("preflight_recovery", ledger[I_KEY]), ("qualification_lint_cleanup", pointer)):
                exact(freeze[key], pin, "saved J freeze ancestry before lexical refusal")
            exact(read_pin(request_spec, root), q, "saved whole J request before lexical refusal")
    return pointer


H_KEY = "direct_science_qualification_lint_repair"
H_PATH = BASE + "/parent-contact-body-runtime-20261008-h.json"
H_REPAIR_PATH = BASE + "/parent-contact-qualification-lint-repair-20261008-h.json"
H_REVIEW_PATH = BASE + "/parent-contact-qualification-reviewed-source-20261008-h.json"
H_SOURCE_PATH = BASE + "/parent-contact-learning-source-freeze-20261008-h.json"
H_QUAL_PATH = BASE + "/parent-contact-learning-qualified-20261008-h.json"
H_EPOCH = "contact-qualification-lint-repair-20261008-h"
H_QUAL_CALLS = tuple("direct-science-20261008-contact-h-" + r for r in ("lint", "math", "gate-a", "gate-b"))
H_CALLS = ("contact-learning-20261008-h-body", "contact-learning-20261008-h-body-audit")
H_BEFORE_PIN = {"path": BASE + "/parent-h-before-20261008/source-before.json", "bytes": 13018,
                "sha256": "3f5539edf097e2cd528a65af8b5f5ba82f7482f09369ef01fc5c9caf1747e116"}
H_ROOT_PIN = {"path": "docs/BANC_M3_H_QUALIFICATION_LINT_REPAIR_ROOT_REFLECTED_20261008.md", "bytes": 20929,
              "sha256": "a654f5bfd052282b83817db5b6c36abf18b6c3477f0bb3fd929b46c8ef92c621"}
H_ASTRA_PINS = {
    "markdown": {"path": BASE + "/parent-h-qualification-lint-repair-astra-review-20261008/review.md", "bytes": 15036,
                 "sha256": "2028976c129b1ef9922514ee76af121fbf72b0dc1ddf40909bfc6f783927ab9d"},
    "json": {"path": BASE + "/parent-h-qualification-lint-repair-astra-review-20261008/review.json", "bytes": 28230,
             "sha256": "1086ae05231bd95a550209a4b7cad48176b47d18c44b2d148bff84a296298e71"},
}
H_CLOSED_PIN = {"path": BASE + "/parent-g-lint-closed285-20261008.json", "bytes": 457032,
                "sha256": "f45f87bc2c7a29df94dcc720c80e25201f8340c8136b40d04710fdc650b0be37"}
H_REGISTERED_PIN = {"path": BASE + "/parent-g-qualification-registered-closed284-20261008.json", "bytes": 456179,
                    "sha256": "d211472e6d721fa62283817122fa5ad78bb8a9227a537ee0158decbcec6f20b6"}
H_G_PATHS = {"review": G_REVIEW_PATH, "amendment": G_AMENDMENT_PATH, "freeze": G_SOURCE_PATH,
             "request": BASE + "/parent-direct-science-20261008-contact-g-lint-request.json",
             "trace": BASE + "/parent-direct-science-20261008-contact-g-lint-genuine-return.json",
             "outer": BASE + "/parent-direct-science-20261008-contact-g-lint/outer.json",
             "stdout": BASE + "/parent-direct-science-20261008-contact-g-lint/stdout.bin",
             "stderr": BASE + "/parent-direct-science-20261008-contact-g-lint/stderr.bin",
             "cost": BASE + "/parent-direct-science-20261008-contact-g-lint-full-return.json",
             "result": BASE + "/parent-g-qualification-repair-ROOT-actual-result-20261008.json"}


def h_archives(root):
    before = read_pin(H_BEFORE_PIN, root)
    exact(set(before), {"schema", "source_metadata_only", "actual_g_lint_once_settled", "scientific_execution",
                        "registration_committed", "snapshots", "closed285"}, "exact pre-H archive map")
    for key, value in (("schema", "flylab.m3.h-qualification-repair.source-before.v1"), ("source_metadata_only", True),
                       ("actual_g_lint_once_settled", True), ("scientific_execution", "NOT_RUN"),
                       ("registration_committed", False), ("closed285", H_CLOSED_PIN)):
        exact(before[key], value, "actual pre-H map " + key)
    need(type(before["snapshots"]) is list and len(before["snapshots"]) == 25, "all25 exact before-H pairs")
    mapped = {}
    for i, item in enumerate(before["snapshots"]):
        exact(set(item), {"source", "archive"}, "exact before-H pair")
        source, archive = item["source"], item["archive"]
        exact(set(source), {"path", "bytes", "sha256"}, "exact before-H pin")
        need(source["path"] not in mapped, "unique before-H identity")
        exact(archive, dict(source, path=BASE + "/parent-h-before-20261008/" + f"{i:02d}-" + source["sha256"][:16] + ".txt"),
              "literal full-content before-H archive")
        raw_pin(archive, root, 524288, allow_empty=True)
        mapped[source["path"]] = item
    expected = set(G_REPLACEMENTS) | set(H_G_PATHS.values()) | {H_REGISTERED_PIN["path"], H_CLOSED_PIN["path"], G_PATH,
        "tools/banc_m3_conserved_resource_native_core.ps1", "tools/verify_c_banc_m3_startup_ray.py",
        "tools/banc_m3_contact_runtime_journal.py", "tools/verify_c_banc_m3_contact_learning.py", "tests/test_c_banc_contact_body_retry.py"}
    exact(set(mapped), expected, "entire exact25 Source/metadata scope")
    return mapped


def h_generated_plan(role, root):
    old = g_generated_plan(role, root)
    index = ("lint", "math", "gate-a", "gate-b").index(role)
    argv = list(old["argv"])
    if role != "lint":
        argv[-1] = "--junitxml=" + BASE + "/parent-contact-learning-h-" + role + "-junit.xml"
    return dict(old, call=H_QUAL_CALLS[index], argv=argv, output=BASE + "/parent-" + H_QUAL_CALLS[index])


def h_plan(root):
    return {"epoch": H_EPOCH, "roster_epoch": EPOCH, "qualification_roster": ROSTER_PIN,
            "calls": [h_generated_plan(r, root) for r in ("lint", "math", "gate-a", "gate-b")],
            "generated_close_s": 20, "original_release_calls": list(H_CALLS)}


def h_history(root, legacy_validate):
    """Strict paid G history; no live G Source, registration, freeze or success."""
    mapped = h_archives(root)
    pins = {key: mapped[name]["source"] for key, name in H_G_PATHS.items()}
    amendment, anchor, policy, grant = g_amendment(pins["amendment"], root, legacy_validate)
    historical, old = historical_f_reviewed_sources(root)
    review = g_review_metadata(pins["review"], root, historical, old)
    exact(amendment["reviewed_source"], pins["review"], "immutable reviewed G pointer")
    exact(len(review["sources"]), 724, "as-executed entire G724")
    for name in G_REPLACEMENTS:
        exact(review["sources"][name], mapped[name]["source"], "as-executed seven G archive identities")
    registered = read_pin(H_REGISTERED_PIN, root, 524288)
    exact(registered, dict(anchor, **{G_KEY: pins["amendment"]}), "closed284 anchor plus sole exact immutable G pointer")
    known_closed(registered)
    prior = read_pin(old[F_SOURCE_PATH]["source"], root)
    freeze = read_pin(pins["freeze"], root)
    expected_freeze = dict(prior, qualification_epoch=G_EPOCH, reviewed_source=pins["review"], qualification_amendment=pins["amendment"],
                           sources=review["sources"], body_retry=review["sources"][G_PATH])
    exact(freeze, expected_freeze, "whole immutable executed G freeze, not live G permission")
    g_body(review["sources"][G_PATH], root)
    q = read_pin(pins["request"], root)
    f_request_identity(q, g_generated_plan("lint", root), pins["freeze"])
    closed = read_pin(H_CLOSED_PIN, root, 524288)
    debit = {"cost_id": G_QUAL_CALLS[0], "mode": "generated", "actual_outer_full_return_s": 11.355,
             "actual_returned_exit_code": 1, "descriptor": pins["cost"]}
    from tools.banc_m3_contact_learning_admission import cost_proof
    cost = read_pin(pins["cost"], root)
    exact(cost_proof(cost, debit, lambda p: read_pin(p, root), generated=True), q, "sole genuine known-closed exit1 G cost")
    for key in ("request", "trace", "outer", "freeze"):
        exact(cost["source_freeze" if key == "freeze" else key], pins[key], "exact immutable G once proof " + key)
    outer = read_pin(pins["outer"], root)
    for key, value in (("all_owned_ended", True), ("reservation_released", True), ("unknown_owned_identities", {}),
                       ("stdout_eof", True), ("stderr_eof", True), ("timeout", False), ("first_error", None),
                       ("worker_exit_code", 1), ("stdout", pins["stdout"]), ("stderr", pins["stderr"])):
        exact(outer[key], value, "positive actual G Job/process/EOF closure " + key)
    for key in ("owner_pid", "owner_creation_ticks", "worker_pid", "worker_creation_ticks"):
        need(type(outer[key]) is int and outer[key] > 0, "actual positive G runtime identity")
    for key in ("stdout", "stderr"):
        raw_pin(pins[key], root, 262144, allow_empty=True)
    trace = read_pin(pins["trace"], root)
    exact((trace["phase"], trace["publication"], trace["first_error"], trace["next_role_allowed"]),
          ("FINAL", "PENDING", None, False), "unchanged actual collector FINAL, failure never permission")
    expected = copy.deepcopy(registered)
    expected["debits"].append(debit)
    expected["direct_science_calls"][G_QUAL_CALLS[0]] = {"state": "charged", "mode": "generated", "request": pins["request"], "reserved_s": 15}
    generated = 0.0
    for d in expected["debits"][118:]:
        if d["mode"] == "generated":
            generated += finite(d["actual_outer_full_return_s"], "immutable ordered Generated history")
    stage = float(policy["generated_stage_before_s"]) + generated
    expected.update(direct_science_generated_used_s=generated, generated_stage_used_s=stage,
                    generated_shared_cumulative_used_s=float(policy["generated_shared_before_s"]) + generated,
                    generated_shared_remaining_s=2300.0 - float(policy["generated_shared_baseline_s"]) - stage)
    exact(closed, expected, "all old284debits/166calls plus sole actual G11.355/exit1; every other field unchanged")
    known_closed(closed)
    ordered_totals(closed, policy, grant, NEW_CAPS)
    result = read_pin(pins["result"], root)
    exact(result["actual_once_commit"]["cost"], pins["cost"], "actual Native once commit receipt")
    exact(result["independent_closed_readback"]["closed285"], H_CLOSED_PIN, "actual ROOT immutable closed285")
    return closed, policy, grant, review, mapped, pins


def h_review_metadata(spec, root, historical, mapped):
    exact(spec["path"], H_REVIEW_PATH, "one acyclic reviewed H manifest")
    review = read_pin(spec, root)
    exact(set(review), {"schema", "epoch", "sources", "replacements", "root_reflection", "before_source",
                        "astra_review", "root_reviews", "sol_reports"}, "exact H review fields")
    for key, value in (("schema", "flylab.m3.contact-qualification.reviewed-source.v2"), ("epoch", H_EPOCH),
                       ("root_reflection", H_ROOT_PIN), ("before_source", H_BEFORE_PIN), ("astra_review", H_ASTRA_PINS)):
        exact(review[key], value, "reviewed H " + key)
    for pin in (H_ROOT_PIN, *H_ASTRA_PINS.values()):
        raw_pin(pin, root, 65536)
    names = {"docs/BANC_M3_H_QUALIFICATION_LINT_REPAIR_ROOT_" + r + "_20261008.md" for r in ("FIRST_REVIEW", "SECOND_REVIEW", "RESIDUAL")}
    pins = review["root_reviews"]
    need(type(pins) is list and 2 <= len(pins) <= 3 and len({p["path"] for p in pins}) == len(pins)
         and {p["path"] for p in pins} <= names
         and names - {"docs/BANC_M3_H_QUALIFICATION_LINT_REPAIR_ROOT_RESIDUAL_20261008.md"} <= {p["path"] for p in pins}, "actual ROOT FIRST/SECOND optional residual")
    allowed = {BASE + "/parent-h-qualification-lint-repair-sol-" + r + "-20261008/" + r + "." + ext
               for r in ("FIRST", "R1") for ext in ("md", "json")}
    sol = review["sol_reports"]
    need(type(sol) is list and len(sol) in (2, 4) and len({p["path"] for p in sol}) == len(sol)
         and {p["path"] for p in sol} <= allowed
         and {BASE + "/parent-h-qualification-lint-repair-sol-FIRST-20261008/FIRST." + ext for ext in ("md", "json")} <= {p["path"] for p in sol}, "actual sole Sol FIRST/optional R1")
    for pin in (*pins, *sol):
        raw_pin(pin, root, 1048576)
    vector, replaced = review["sources"], review["replacements"]
    need(type(vector) is dict and type(replaced) is dict, "typed H vector/replacements")
    exact(set(vector), set(historical["sources"]) | {H_PATH}, "entire G724 plus only static H725")
    exact(set(replaced), set(G_REPLACEMENTS), "same seven replacement identities only")
    for name, pin in vector.items():
        exact(set(pin), {"path", "bytes", "sha256"}, "exact H Source pin")
        exact(pin["path"], name, "literal H Source identity")
        need(type(pin["bytes"]) is int and 0 <= pin["bytes"] <= 2097152 and type(pin["sha256"]) is str
             and re.fullmatch("[0-9a-f]{64}", pin["sha256"]), "fixed finite Source cap")
        if name in G_REPLACEMENTS:
            need(0 < pin["bytes"] <= BOOTSTRAP_SOURCE_CAP, "existing changed Source cap")
            exact(replaced[name], {"before": mapped[name]["source"], "after": pin}, "exact before-H/after-H replacement")
        elif name != H_PATH:
            exact(pin, historical["sources"][name], "all unrelated/installed/static G Source unchanged")
    return review


def h_body(spec, root):
    exact(spec["path"], H_PATH, "only static H body")
    mapped = h_archives(root)
    prior = g_body(mapped[G_PATH]["source"], root)
    expected = dict(prior, schema="flylab.m3.contact-learning.body-runtime.v5", retry_id="contact-learning-20261008-h",
                    calls=list(H_CALLS), qualification_epoch=H_EPOCH, source_review_path=H_REVIEW_PATH, source_freeze_path=H_SOURCE_PATH,
                    qualification_lint_repair_path=H_REPAIR_PATH, historical_closed285=H_CLOSED_PIN, failed_g_lint=mapped[H_G_PATHS["cost"]]["source"])
    body = read_pin(spec, root)
    exact(body, expected, "whole G object: exactly six changed fields and three additions; old G/F allocation lineage intact")
    return body


def h_current_review(spec, root, legacy_validate):
    _anchor, _policy, _grant, historical, mapped, _pins = h_history(root, legacy_validate)
    review = h_review_metadata(spec, root, historical, mapped)
    for pin in review["sources"].values():
        source_text(pin, root)
    h_body(review["sources"][H_PATH], root)
    return review


def h_lint_repair(spec, root, legacy_validate):
    exact(spec["path"], H_REPAIR_PATH, "one immutable separately typed H pointer")
    repair = read_pin(spec, root)
    anchor, policy, grant, historical, mapped, pins = h_history(root, legacy_validate)
    exact(set(repair), {"schema", "epoch", "scope", "allocation", "qualification_amendment", "historical_closed285", "failed_g_lint",
                       "source_before", "reviewed_source", "qualification_plan", "new_resource_grant_s", "refund_s"}, "exact H repair fields")
    for key, value in (("schema", "flylab.m3.contact-qualification-lint-repair.v1"), ("epoch", H_EPOCH), ("scope", "PREQUALIFICATION_ONLY"),
                       ("allocation", anchor[ALLOCATION_KEY]), ("qualification_amendment", anchor[G_KEY]), ("historical_closed285", H_CLOSED_PIN),
                       ("failed_g_lint", pins), ("source_before", H_BEFORE_PIN), ("qualification_plan", h_plan(root)), ("new_resource_grant_s", 0), ("refund_s", 0)):
        exact(repair[key], value, "immutable H repair " + key)
    h_review_metadata(repair["reviewed_source"], root, historical, mapped)
    return repair, anchor, policy, grant


def h_registration(ledger, spec, root, legacy_validate, *, before_raw):
    need(type(before_raw) is bytes and before_raw == raw_pin(H_CLOSED_PIN, root, 524288), "exact actual closed285 raw before H registration")
    need(H_KEY not in ledger, "no duplicate/alias/overwrite H registration")
    repair, anchor, _policy, _grant = h_lint_repair(spec, root, legacy_validate)
    known_closed(ledger)
    exact(ledger, anchor, "whole strict closed285 before sole new H pointer")
    need(ledger["direct_science_generated_used_s"] + 300 <= 1564 and ledger["generated_shared_remaining_s"] >= 300, "full280 plus distinct close20 in both finite pools")
    h_current_review(repair["reviewed_source"], root, legacy_validate)
    after = copy.deepcopy(ledger)
    after[H_KEY] = copy.deepcopy(spec)
    exact(set(after) - set(ledger), {H_KEY}, "only one new immutable H pointer; no allocation/grant/refund")
    return after


def h_current_caps(ledger, root, legacy_validate):
    repair, anchor, policy, grant = h_lint_repair(ledger[H_KEY], root, legacy_validate)
    exact(ledger[G_KEY], repair["qualification_amendment"], "immutable G pointer unchanged after H")
    expected = dict(anchor, **{H_KEY: ledger[H_KEY]})
    exact(set(ledger), set(expected), "strict H keyset, neither pointer a generic mutable tail exemption")
    prefix(ledger, anchor)
    for key, value in expected.items():
        if key not in TAIL_KEYS:
            exact(ledger[key], value, "immutable H history/accounting " + key)
    ordered_totals(ledger, policy, grant, NEW_CAPS)
    return dict(NEW_CAPS)


def h_freeze(spec, repair, repair_spec, root):
    exact(spec["path"], H_SOURCE_PATH, "fresh literal H freeze")
    freeze = read_pin(spec, root, BOOTSTRAP_SOURCE_CAP)
    mapped = h_archives(root)
    prior = read_pin(mapped[G_SOURCE_PATH]["source"], root)
    expected = dict(prior, qualification_epoch=H_EPOCH, reviewed_source=repair["reviewed_source"],
                    qualification_amendment=repair["qualification_amendment"], qualification_lint_repair=repair_spec,
                    sources=freeze["sources"], body_retry=freeze["body_retry"])
    exact(freeze, expected, "whole G metadata plus H fields and both immutable pointers")
    from tools.banc_m3_startup_ray_supervisor import generated_budget_limit
    review = h_current_review(repair["reviewed_source"], root, generated_budget_limit)
    exact(freeze["sources"], review["sources"], "entire current H725")
    exact(freeze["body_retry"], review["sources"][H_PATH], "one reviewed static H descriptor")
    return freeze


def _qualification_family(name):
    """Private closed literals only; no caller epoch, offset, cap or skip flags."""
    need(name in ("g", "h", "i", "j"), "only four fixed qualification families")
    return {
        "g": {"key": G_KEY, "pointer_path": G_AMENDMENT_PATH, "source_path": G_SOURCE_PATH, "qual_path": G_QUAL_PATH,
              "epoch": G_EPOCH, "schema": "flylab.m3.direct-science.qualification.v4", "offset": 284, "end": 288,
              "calls": G_CALLS, "qual_calls": G_QUAL_CALLS, "caps": g_current_caps, "metadata": g_amendment,
              "freeze": g_freeze, "body": g_body, "plan": g_generated_plan, "envelope": 300},
        "h": {"key": H_KEY, "pointer_path": H_REPAIR_PATH, "source_path": H_SOURCE_PATH, "qual_path": H_QUAL_PATH,
              "epoch": H_EPOCH, "schema": "flylab.m3.direct-science.qualification.v5", "offset": 285, "end": 289,
              "calls": H_CALLS, "qual_calls": H_QUAL_CALLS, "caps": h_current_caps, "metadata": h_lint_repair,
              "freeze": h_freeze, "body": h_body, "plan": h_generated_plan, "envelope": 300},
        "i": {"key": I_KEY, "pointer_path": I_RECOVERY_PATH, "source_path": I_SOURCE_PATH, "qual_path": I_QUAL_PATH,
              "epoch": I_EPOCH, "schema": "flylab.m3.direct-science.qualification.v6", "offset": 286, "end": 290,
              "calls": I_CALLS, "qual_calls": I_QUAL_CALLS, "caps": i_current_caps, "metadata": i_preflight_recovery,
              "freeze": i_freeze, "body": i_body, "plan": i_generated_plan, "envelope": 575},
        "j": {"key": J_KEY, "pointer_path": J_CLEANUP_PATH, "source_path": J_SOURCE_PATH, "qual_path": J_QUAL_PATH,
              "epoch": J_EPOCH, "schema": "flylab.m3.direct-science.qualification.v7", "offset": 287, "end": 291,
              "calls": J_CALLS, "qual_calls": J_QUAL_CALLS, "caps": j_current_caps, "metadata": j_lint_cleanup,
              "freeze": j_freeze, "body": j_body, "plan": j_generated_plan, "envelope": 575},
    }[name]


def g_progress(ledger, amendment, pointer, source, root):
    return _epoch_progress(ledger, amendment, pointer, source, root, "g")


def h_progress(ledger, repair, pointer, source, root):
    return _epoch_progress(ledger, repair, pointer, source, root, "h")


def validate_g_qualification(qualification, ledger, root, legacy_validate):
    return _epoch_qualification(qualification, ledger, root, legacy_validate, "g")


def validate_h_qualification(qualification, ledger, root, legacy_validate):
    return _epoch_qualification(qualification, ledger, root, legacy_validate, "h")


def g_dispatch(ledger, q, request_spec, root, legacy_validate, *, phase):
    return _epoch_dispatch(ledger, q, request_spec, root, legacy_validate, "g", phase=phase)


def h_dispatch(ledger, q, request_spec, root, legacy_validate, *, phase):
    return _epoch_dispatch(ledger, q, request_spec, root, legacy_validate, "h", phase=phase)


def g_admitted_identity(ledger, q, request_spec, root):
    return _epoch_admitted_identity(ledger, q, request_spec, root, "g")


def h_admitted_identity(ledger, q, request_spec, root):
    return _epoch_admitted_identity(ledger, q, request_spec, root, "h")


I_KEY = "direct_science_preflight_recovery"
I_EPOCH = "contact-preflight-overhead-recovery-20261008-i"
I_PATH = BASE + "/parent-contact-body-runtime-20261008-i.json"
I_RECOVERY_PATH = BASE + "/parent-contact-preflight-recovery-20261008-i.json"
I_REVIEW_PATH = BASE + "/parent-contact-preflight-reviewed-source-20261008-i.json"
I_SOURCE_PATH = BASE + "/parent-contact-learning-source-freeze-20261008-i.json"
I_QUAL_PATH = BASE + "/parent-contact-learning-qualified-20261008-i.json"
I_QUAL_CALLS = tuple("direct-science-20261008-contact-i-" + r for r in ("lint", "math", "gate-a", "gate-b"))
I_CALLS = ("contact-learning-20261008-i-body", "contact-learning-20261008-i-body-audit")
I_BEFORE_PIN = {"path": BASE + "/parent-i-before-20261008/source-before.json", "bytes": 15101,
                "sha256": "32fd13fce6cb183fc268633f5bf6c5fb6ec8ea94c8b0fd2fd0f221122b81a32b"}
I_ROOT_PIN = {"path": "docs/BANC_M3_I_PREFLIGHT_OVERHEAD_RECOVERY_ROOT_REFLECTED_20261008.md", "bytes": 18795,
              "sha256": "b9adaa3cfefe8f6cc4798353134786a4ea8b4d202fdf8a14a6836a3a91b3f694"}
I_ASTRA_PINS = {
    "markdown": {"path": BASE + "/parent-i-preflight-overhead-astra-review-20261008/review.md", "bytes": 29182,
                 "sha256": "f81df8fdac76b298b33f8815207602ac664062df48721cd86011217b6fa39808"},
    "json": {"path": BASE + "/parent-i-preflight-overhead-astra-review-20261008/review.json", "bytes": 33212,
             "sha256": "66763c19f7d7f2934324badfcca9e25421279b6ddf0e7d18849d8fa729849837"},
}
I_CLOSED_PIN = {"path": BASE + "/parent-h-lint-closed286-20261008.json", "bytes": 458157,
                "sha256": "0c5b10a4e68bcc5bbe8b485bce872c3de219b9a9af2f7992a010c831eff25b98"}
I_REGISTERED_PIN = {"path": BASE + "/parent-h-qualification-registered-closed285-20261008.json", "bytes": 457306,
                    "sha256": "9d9ddf1b69073dbf57f2c0882d492314a94a0257cc89ffa3d7600a05d38b16f9"}
I_ADMITTED_PIN = {"path": BASE + "/parent-h-lint-actual-admitted-20261008.json", "bytes": 458051,
                  "sha256": "6365c44eb3c8426604004a97a4be41a8b414972e7f913f9fb6463ec679f3b81c"}
I_H_PATHS = {"review": H_REVIEW_PATH, "repair": H_REPAIR_PATH, "freeze": H_SOURCE_PATH,
             "request": BASE + "/parent-direct-science-20261008-contact-h-lint-request.json",
             "trace": BASE + "/parent-direct-science-20261008-contact-h-lint-genuine-return.json",
             "outer": BASE + "/parent-direct-science-20261008-contact-h-lint/outer.json",
             "stdout": BASE + "/parent-direct-science-20261008-contact-h-lint/stdout.bin",
             "stderr": BASE + "/parent-direct-science-20261008-contact-h-lint/stderr.bin",
             "cost": BASE + "/parent-direct-science-20261008-contact-h-lint-full-return.json",
             "result": BASE + "/parent-h-qualification-lint-repair-ROOT-actual-result-20261008.json",
             "registered": I_REGISTERED_PIN["path"], "admitted": I_ADMITTED_PIN["path"]}


def i_archives(root):
    before = read_pin(I_BEFORE_PIN, root)
    exact(set(before), {"schema", "source_metadata_only", "actual_h_lint_once_settled", "scientific_execution",
                        "registration_committed", "snapshots", "closed286"}, "exact pre-I map")
    for key, value in (("schema", "flylab.m3.i-preflight-overhead-recovery.source-before.v1"), ("source_metadata_only", True),
                       ("actual_h_lint_once_settled", True), ("scientific_execution", "NOT_RUN"),
                       ("registration_committed", False), ("closed286", I_CLOSED_PIN)):
        exact(before[key], value, "immutable pre-I " + key)
    need(type(before["snapshots"]) is list and len(before["snapshots"]) == 29, "exact29 pairs")
    mapped = {}
    for i, item in enumerate(before["snapshots"]):
        exact(set(item), {"source", "archive"}, "exact pre-I pair")
        source, archive = item["source"], item["archive"]
        exact(set(source), {"path", "bytes", "sha256"}, "exact immutable pre-I pin")
        need(source["path"] not in mapped, "unique pre-I identity")
        exact(archive, dict(source, path=BASE + "/parent-i-before-20261008/" + f"{i:02d}-" + source["sha256"][:16] + ".txt"), "full inert pre-I archive")
        raw_pin(archive, root, 524288, allow_empty=True)
        mapped[source["path"]] = item
    expected = set(G_REPLACEMENTS) | set(I_H_PATHS.values()) | {I_CLOSED_PIN["path"], H_BEFORE_PIN["path"], H_CLOSED_PIN["path"], H_PATH, ROSTER_PIN["path"],
        "tools/banc_m3_conserved_resource_native_core.ps1", "tools/verify_c_banc_m3_startup_ray.py", "tools/banc_m3_contact_runtime_journal.py",
        "tools/verify_c_banc_m3_contact_learning.py", "tests/test_c_banc_contact_body_retry.py"}
    exact(set(mapped), expected, "whole immutable29 scope")
    return mapped


def i_generated_plan(role, root):
    old = h_generated_plan(role, root)
    index = ("lint", "math", "gate-a", "gate-b").index(role)
    argv = list(old["argv"])
    if role != "lint":
        argv[-1] = "--junitxml=" + BASE + "/parent-contact-learning-i-" + role + "-junit.xml"
    outer, child, cleanup = ((40, 35, 5), (35, 30, 5), (120, 110, 10), (360, 340, 20))[index]
    return dict(old, call=I_QUAL_CALLS[index], argv=argv, output=BASE + "/parent-" + I_QUAL_CALLS[index],
                limits=dict(old["limits"], outer_s=outer, child_s=child, cleanup_s=cleanup))


def i_plan(root):
    return {"epoch": I_EPOCH, "roster_epoch": EPOCH, "qualification_roster": ROSTER_PIN,
            "calls": [i_generated_plan(r, root) for r in ("lint", "math", "gate-a", "gate-b")],
            "generated_close_s": 20, "original_release_calls": list(I_CALLS)}


def i_history(root, legacy_validate):
    """Whole registered/admitted/closed H failure; never current H Source permission."""
    mapped = i_archives(root)
    pins = {key: mapped[name]["source"] for key, name in I_H_PATHS.items()}
    repair, anchor, policy, grant = h_lint_repair(pins["repair"], root, legacy_validate)
    _old_anchor, _p, _g, historical, old, _pins = h_history(root, legacy_validate)
    review = h_review_metadata(pins["review"], root, historical, old)
    exact(repair["reviewed_source"], pins["review"], "executed H review pointer")
    exact(len(review["sources"]), 725, "entire as-executed H725")
    for name in G_REPLACEMENTS:
        exact(review["sources"][name], mapped[name]["source"], "exact as-executed H archive identity")
    registered = read_pin(I_REGISTERED_PIN, root, 524288)
    exact(registered, dict(anchor, **{H_KEY: pins["repair"]}), "exact closed285 plus H pointer")
    known_closed(registered)
    prior = read_pin(old[G_SOURCE_PATH]["source"], root)
    freeze = read_pin(pins["freeze"], root)
    exact(freeze, dict(prior, qualification_epoch=H_EPOCH, reviewed_source=pins["review"],
                       qualification_amendment=repair["qualification_amendment"], qualification_lint_repair=pins["repair"],
                       sources=review["sources"], body_retry=review["sources"][H_PATH]), "whole immutable executed H freeze")
    h_body(review["sources"][H_PATH], root)
    q = read_pin(pins["request"], root)
    f_request_identity(q, h_generated_plan("lint", root), pins["freeze"])
    admitted = copy.deepcopy(registered)
    admitted["direct_science_calls"][H_QUAL_CALLS[0]] = {"state": "admitted", "mode": "generated", "request": pins["request"], "reserved_s": 15}
    admitted["direct_science_reservations"] = {H_QUAL_CALLS[0]: {"request": pins["request"], "reserved_s": 15}}
    exact(read_pin(I_ADMITTED_PIN, root, 524288), admitted, "whole registered-to-admitted H identity; no other mutation")
    debit = {"cost_id": H_QUAL_CALLS[0], "mode": "generated", "actual_outer_full_return_s": 13.253,
             "actual_returned_exit_code": 124, "descriptor": pins["cost"]}
    from tools.banc_m3_contact_learning_admission import cost_proof
    cost = read_pin(pins["cost"], root)
    exact(cost_proof(cost, debit, lambda p: read_pin(p, root), generated=True, expected_timeout=True), q, "literal historical H timeout only")
    for key in ("request", "trace", "outer", "freeze"):
        exact(cost["source_freeze" if key == "freeze" else key], pins[key], "actual H fee pin " + key)
    outer = read_pin(pins["outer"], root)
    for key, value in (("all_owned_ended", True), ("reservation_released", True), ("unknown_owned_identities", {}),
                       ("stdout_eof", True), ("stderr_eof", True), ("timeout", True), ("worker_exit_code", 124),
                       ("stdout", pins["stdout"]), ("stderr", pins["stderr"])):
        exact(outer[key], value, "actual spawned H closure " + key)
    exact((outer["first_error"]["type"], outer["first_error"]["message"]), ("TimeoutError", "same ENTRY pre-resume child interval expired"), "actual H primary error")
    for key in ("owner_pid", "owner_creation_ticks", "worker_pid", "worker_creation_ticks"):
        need(type(outer[key]) is int and outer[key] > 0, "positive actual H identity")
    for key in ("stdout", "stderr"):
        exact(raw_pin(pins[key], root, 262144, allow_empty=True), b"", "actual empty H streams")
    trace = read_pin(pins["trace"], root)
    exact((trace["phase"], trace["publication"], trace["first_error"], trace["next_role_allowed"]), ("FINAL", "PENDING", None, False), "failure remains non-permission")
    need(type(trace["dispatch_ms"]) is int and type(trace["first_final_ms"]) is int, "typed actual H collector clocks")
    exact((trace["dispatch_ms"], trace["first_final_ms"]), (1791444054523, 1791444067776), "literal actual H timestamps")
    expected = copy.deepcopy(registered)
    expected["debits"].append(debit)
    expected["direct_science_calls"][H_QUAL_CALLS[0]] = {"state": "charged", "mode": "generated", "request": pins["request"], "reserved_s": 15}
    generated = 0.0
    for d in expected["debits"][118:]:
        if d["mode"] == "generated":
            generated += finite(d["actual_outer_full_return_s"], "immutable ordered Generated fee")
    stage = float(policy["generated_stage_before_s"]) + generated
    expected.update(direct_science_generated_used_s=generated, generated_stage_used_s=stage,
                    generated_shared_cumulative_used_s=float(policy["generated_shared_before_s"]) + generated,
                    generated_shared_remaining_s=2300.0 - float(policy["generated_shared_baseline_s"]) - stage)
    closed = read_pin(I_CLOSED_PIN, root, 524288)
    exact(closed, expected, "all285debits/167calls plus only actual H13.253/124; exact ordered totals")
    known_closed(closed)
    ordered_totals(closed, policy, grant, NEW_CAPS)
    result = read_pin(pins["result"], root)
    exact(result["actual_once_commit"]["cost"], pins["cost"], "saved actual Native once receipt")
    exact(result["independent_closed_readback"]["closed286"], I_CLOSED_PIN, "immutable actual closed286")
    exact(result["actual_lint"]["immutable_trace"], pins["trace"], "saved actual FINAL trace")
    return closed, policy, grant, review, mapped, pins


def i_body(spec, root):
    exact(spec["path"], I_PATH, "only static I body")
    mapped = i_archives(root)
    prior = h_body(mapped[H_PATH]["source"], root)
    expected = dict(prior, schema="flylab.m3.contact-learning.body-runtime.v6", retry_id="contact-learning-20261008-i",
                    calls=list(I_CALLS), qualification_epoch=I_EPOCH, source_review_path=I_REVIEW_PATH, source_freeze_path=I_SOURCE_PATH,
                    preflight_recovery_path=I_RECOVERY_PATH, historical_closed286=I_CLOSED_PIN, failed_h_lint=mapped[I_H_PATHS["cost"]]["source"])
    body = read_pin(spec, root)
    exact(body, expected, "whole static H plus six changes and three additions")
    return body


def i_current_review(spec, root, legacy_validate):
    _anchor, _policy, _grant, historical, mapped, _pins = i_history(root, legacy_validate)
    review = i_review_metadata(spec, root, historical, mapped)
    for pin in review["sources"].values():
        source_text(pin, root)
    i_body(review["sources"][I_PATH], root)
    return review


def i_preflight_recovery(spec, root, legacy_validate):
    exact(spec["path"], I_RECOVERY_PATH, "one immutable typed I descriptor")
    recovery = read_pin(spec, root)
    anchor, policy, grant, historical, mapped, pins = i_history(root, legacy_validate)
    exact(set(recovery), {"schema", "epoch", "scope", "allocation", "qualification_amendment", "qualification_lint_repair",
                         "historical_closed286", "failed_h_lint", "source_before", "reviewed_source", "qualification_plan", "new_resource_grant_s", "refund_s"}, "exact I descriptor fields")
    for key, value in (("schema", "flylab.m3.contact-preflight-recovery.v1"), ("epoch", I_EPOCH), ("scope", "PREQUALIFICATION_ONLY"),
                       ("allocation", anchor[ALLOCATION_KEY]), ("qualification_amendment", anchor[G_KEY]), ("qualification_lint_repair", anchor[H_KEY]),
                       ("historical_closed286", I_CLOSED_PIN), ("failed_h_lint", pins), ("source_before", I_BEFORE_PIN),
                       ("qualification_plan", i_plan(root)), ("new_resource_grant_s", 0), ("refund_s", 0)):
        exact(recovery[key], value, "immutable I " + key)
    i_review_metadata(recovery["reviewed_source"], root, historical, mapped)
    return recovery, anchor, policy, grant


def i_registration(ledger, spec, root, legacy_validate, *, before_raw):
    need(type(before_raw) is bytes and before_raw == raw_pin(I_CLOSED_PIN, root, 524288), "exact closed286 raw before I")
    need(I_KEY not in ledger, "no duplicate/alias/overwrite I")
    recovery, anchor, _policy, _grant = i_preflight_recovery(spec, root, legacy_validate)
    known_closed(ledger)
    exact(ledger, anchor, "whole exact closed286 before sole I pointer")
    need(ledger["direct_science_generated_used_s"] + 575 <= 1564 and ledger["generated_shared_remaining_s"] >= 575, "whole555 plus close20 in both pools")
    i_current_review(recovery["reviewed_source"], root, legacy_validate)
    after = copy.deepcopy(ledger)
    after[I_KEY] = copy.deepcopy(spec)
    exact(set(after) - set(ledger), {I_KEY}, "sole immutable I pointer, no grant/refund/transfer")
    return after


def i_current_caps(ledger, root, legacy_validate):
    recovery, anchor, policy, grant = i_preflight_recovery(ledger[I_KEY], root, legacy_validate)
    exact(ledger[G_KEY], recovery["qualification_amendment"], "G pointer unchanged after I")
    exact(ledger[H_KEY], recovery["qualification_lint_repair"], "H pointer unchanged after I")
    expected = dict(anchor, **{I_KEY: ledger[I_KEY]})
    exact(set(ledger), set(expected), "strict I keyset; no pointer mutable-tail exemption")
    prefix(ledger, anchor)
    for key, value in expected.items():
        if key not in TAIL_KEYS:
            exact(ledger[key], value, "immutable I accounting " + key)
    ordered_totals(ledger, policy, grant, NEW_CAPS)
    return dict(NEW_CAPS)


def i_freeze(spec, recovery, recovery_spec, root):
    exact(spec["path"], I_SOURCE_PATH, "literal fresh I freeze")
    freeze = read_pin(spec, root, BOOTSTRAP_SOURCE_CAP)
    prior = read_pin(i_archives(root)[H_SOURCE_PATH]["source"], root)
    expected = dict(prior, qualification_epoch=I_EPOCH, reviewed_source=recovery["reviewed_source"],
                    sources=freeze["sources"], body_retry=freeze["body_retry"], preflight_recovery=recovery_spec)
    exact(freeze, expected, "whole H immutable clone and one I pointer")
    from tools.banc_m3_startup_ray_supervisor import generated_budget_limit
    review = i_current_review(recovery["reviewed_source"], root, generated_budget_limit)
    exact(freeze["sources"], review["sources"], "whole current I726")
    exact(freeze["body_retry"], review["sources"][I_PATH], "one reviewed static I")
    return freeze


def i_progress(ledger, recovery, pointer, source, root):
    return _epoch_progress(ledger, recovery, pointer, source, root, "i")


def validate_i_qualification(qualification, ledger, root, legacy_validate):
    return _epoch_qualification(qualification, ledger, root, legacy_validate, "i")


def i_dispatch(ledger, q, request_spec, root, legacy_validate, *, phase):
    return _epoch_dispatch(ledger, q, request_spec, root, legacy_validate, "i", phase=phase)


def i_admitted_identity(ledger, q, request_spec, root):
    return _epoch_admitted_identity(ledger, q, request_spec, root, "i")


def i_review_metadata(spec, root, historical, mapped):
    exact(spec["path"], I_REVIEW_PATH, "one acyclic reviewed H manifest")
    review = read_pin(spec, root)
    exact(set(review), {"schema", "epoch", "sources", "replacements", "root_reflection", "before_source",
                        "astra_review", "root_reviews", "sol_reports"}, "exact H review fields")
    for key, value in (("schema", "flylab.m3.contact-qualification.reviewed-source.v3"), ("epoch", I_EPOCH),
                       ("root_reflection", I_ROOT_PIN), ("before_source", I_BEFORE_PIN), ("astra_review", I_ASTRA_PINS)):
        exact(review[key], value, "reviewed H " + key)
    for pin in (I_ROOT_PIN, *I_ASTRA_PINS.values()):
        raw_pin(pin, root, 65536)
    names = {"docs/BANC_M3_I_PREFLIGHT_OVERHEAD_RECOVERY_ROOT_" + r + "_20261008.md" for r in ("FIRST_REVIEW", "SECOND_REVIEW", "RESIDUAL")}
    pins = review["root_reviews"]
    need(type(pins) is list and 2 <= len(pins) <= 3 and len({p["path"] for p in pins}) == len(pins)
         and {p["path"] for p in pins} <= names
         and names - {"docs/BANC_M3_I_PREFLIGHT_OVERHEAD_RECOVERY_ROOT_RESIDUAL_20261008.md"} <= {p["path"] for p in pins}, "actual ROOT FIRST/SECOND optional residual")
    allowed = {BASE + "/parent-i-preflight-overhead-sol-" + r + "-20261008/" + r + "." + ext
               for r in ("FIRST", "R1") for ext in ("md", "json")}
    sol = review["sol_reports"]
    need(type(sol) is list and len(sol) in (2, 4) and len({p["path"] for p in sol}) == len(sol)
         and {p["path"] for p in sol} <= allowed
         and {BASE + "/parent-i-preflight-overhead-sol-FIRST-20261008/FIRST." + ext for ext in ("md", "json")} <= {p["path"] for p in sol}, "actual sole Sol FIRST/optional R1")
    for pin in (*pins, *sol):
        raw_pin(pin, root, 1048576)
    vector, replaced = review["sources"], review["replacements"]
    need(type(vector) is dict and type(replaced) is dict, "typed H vector/replacements")
    exact(set(vector), set(historical["sources"]) | {I_PATH}, "entire H725 plus only static I726")
    exact(set(replaced), set(G_REPLACEMENTS), "same seven replacement identities only")
    for name, pin in vector.items():
        exact(set(pin), {"path", "bytes", "sha256"}, "exact H Source pin")
        exact(pin["path"], name, "literal H Source identity")
        need(type(pin["bytes"]) is int and 0 <= pin["bytes"] <= 2097152 and type(pin["sha256"]) is str
             and re.fullmatch("[0-9a-f]{64}", pin["sha256"]), "fixed finite Source cap")
        if name in G_REPLACEMENTS:
            need(0 < pin["bytes"] <= BOOTSTRAP_SOURCE_CAP, "existing changed Source cap")
            exact(replaced[name], {"before": mapped[name]["source"], "after": pin}, "exact before-H/after-H replacement")
        elif name != I_PATH:
            exact(pin, historical["sources"][name], "all unrelated/installed/static G Source unchanged")
    return review



J_KEY = "direct_science_qualification_lint_cleanup"
J_EPOCH = "contact-qualification-lint-cleanup-20261008-j"
J_PATH = BASE + "/parent-contact-body-runtime-20261008-j.json"
J_CLEANUP_PATH = BASE + "/parent-contact-qualification-lint-cleanup-20261008-j.json"
J_REVIEW_PATH = BASE + "/parent-contact-qualification-reviewed-source-20261008-j.json"
J_SOURCE_PATH = BASE + "/parent-contact-learning-source-freeze-20261008-j.json"
J_QUAL_PATH = BASE + "/parent-contact-learning-qualified-20261008-j.json"
J_QUAL_CALLS = tuple("direct-science-20261008-contact-j-" + r for r in ("lint", "math", "gate-a", "gate-b"))
J_CALLS = ("contact-learning-20261008-j-body", "contact-learning-20261008-j-body-audit")
J_BEFORE_PIN = {
    "path": _HISTORY_BASE+"/parent-j-before-20261008/source-before.json",
    "bytes": 15084,
    "sha256": "bd24b181dd6b788d47debc11cc9830c840c7c126b130547a9d1551bb7a985a40"
}
J_ROOT_PIN = {
    "path": "docs/BANC_M3_J_FIXED_RECORDS_R1_ROOT_REFLECTED_20261008.md",
    "bytes": 28250,
    "sha256": "c78899df480ec38c059de7b81417e75ed8aaa649b74f94e633f8ca1fa46a1455"
}
J_ASTRA_PINS = {
    "markdown": {
        "path": _HISTORY_BASE+"/parent-j-fixed-records-R1-design-astra-review-20261008/review.md",
        "bytes": 16146,
        "sha256": "18f46128096cc97027740469d2fd24a0cfeaaaf81e4f250c90ab2c7ef02a0381"
    },
    "json": {
        "path": _HISTORY_BASE+"/parent-j-fixed-records-R1-design-astra-review-20261008/review.json",
        "bytes": 37180,
        "sha256": "5ca5f8e6a9b85d356c0dab005a9d011be94247ef5c2165372188f7afb74c32c7"
    }
}
IMMUTABLE_RECORD_PIN = {
    "bytes": 16830,
    "path": _HISTORY_BASE+"/parent-j-immutable-attempt-records-20261008.json",
    "sha256": "60cb1084cefbb8204178c15ffa319da0683d137b1d9d8da460591a5ab7945db0"
}
J_CLOSED_PIN = {
    "path": _HISTORY_BASE+"/parent-i-lint-closed287-20261008.json",
    "sha256": "56dec9a77fa285e83345d79611fa51181e77b5b682e95b811e5d4d63365d3807",
    "bytes": 459280
}
J_REGISTERED_PIN = {
    "path": _HISTORY_BASE+"/parent-i-preflight-registered-closed286-20261008.json",
    "sha256": "233f8bea1af6b78791c951137e17df8ff78f65dbe9d37125d248da083c3f7917",
    "bytes": 458417
}
J_ADMITTED_PIN = {
    "path": _HISTORY_BASE+"/parent-i-lint-actual-admitted-20261008.json",
    "sha256": "052402ca89ddd3480e7ac96d4527008ed9eb2916393482dabb91d1c07ce4731b",
    "bytes": 459162
}
J_I_PATHS = {
    "review": _HISTORY_BASE+"/parent-contact-preflight-reviewed-source-20261008-i.json",
    "recovery": _HISTORY_BASE+"/parent-contact-preflight-recovery-20261008-i.json",
    "freeze": _HISTORY_BASE+"/parent-contact-learning-source-freeze-20261008-i.json",
    "request": _HISTORY_BASE+"/parent-direct-science-20261008-contact-i-lint-request.json",
    "trace": _HISTORY_BASE+"/parent-direct-science-20261008-contact-i-lint-genuine-return.json",
    "outer": _HISTORY_BASE+"/parent-direct-science-20261008-contact-i-lint/outer.json",
    "stdout": _HISTORY_BASE+"/parent-direct-science-20261008-contact-i-lint/stdout.bin",
    "stderr": _HISTORY_BASE+"/parent-direct-science-20261008-contact-i-lint/stderr.bin",
    "cost": _HISTORY_BASE+"/parent-direct-science-20261008-contact-i-lint-full-return.json",
    "result": _HISTORY_BASE+"/parent-i-preflight-overhead-ROOT-actual-result-20261008.json",
    "registered": _HISTORY_BASE+"/parent-i-preflight-registered-closed286-20261008.json",
    "admitted": _HISTORY_BASE+"/parent-i-lint-actual-admitted-20261008.json"
}


def j_archives(root):
    before = read_pin(J_BEFORE_PIN, root)
    exact(set(before), {"schema", "source_metadata_only", "actual_i_lint_once_settled", "scientific_execution",
                        "registration_committed", "snapshots", "closed287"}, "exact pre-J map")
    for key, value in (("schema", "flylab.m3.j-qualification-lint-cleanup.source-before.v1"), ("source_metadata_only", True),
                       ("actual_i_lint_once_settled", True), ("scientific_execution", "NOT_RUN"),
                       ("registration_committed", False), ("closed287", J_CLOSED_PIN)):
        exact(before[key], value, "immutable pre-J " + key)
    need(type(before["snapshots"]) is list and len(before["snapshots"]) == 29, "exact29 J pairs")
    mapped = {}
    for i, item in enumerate(before["snapshots"]):
        exact(set(item), {"source", "archive"}, "exact pre-J pair")
        source, archive = item["source"], item["archive"]
        exact(set(source), {"path", "bytes", "sha256"}, "exact immutable pre-J pin")
        need(source["path"] not in mapped, "unique pre-J identity")
        exact(archive, dict(source, path=BASE + "/parent-j-before-20261008/" + f"{i:02d}-" + source["sha256"][:16] + ".txt"), "full inert pre-J archive")
        raw_pin(archive, root, 524288, allow_empty=True)
        mapped[source["path"]] = item
    expected = set(G_REPLACEMENTS) | set(J_I_PATHS.values()) | {J_CLOSED_PIN["path"], I_BEFORE_PIN["path"], I_CLOSED_PIN["path"], I_PATH, ROSTER_PIN["path"],
        "tools/banc_m3_conserved_resource_native_core.ps1", "tools/verify_c_banc_m3_startup_ray.py", "tools/banc_m3_contact_runtime_journal.py",
        "tools/verify_c_banc_m3_contact_learning.py", "tests/test_c_banc_contact_body_retry.py"}
    exact(set(mapped), expected, "whole immutable29 J scope")
    return mapped


def j_generated_plan(role, root):
    old = i_generated_plan(role, root)
    index = ("lint", "math", "gate-a", "gate-b").index(role)
    argv = list(old["argv"])
    if role != "lint":
        argv[-1] = "--junitxml=" + BASE + "/parent-contact-learning-j-" + role + "-junit.xml"
    return dict(old, call=J_QUAL_CALLS[index], argv=argv, output=BASE + "/parent-" + J_QUAL_CALLS[index])


def j_plan(root):
    return {"epoch": J_EPOCH, "roster_epoch": EPOCH, "qualification_roster": ROSTER_PIN,
            "calls": [j_generated_plan(r, root) for r in ("lint", "math", "gate-a", "gate-b")],
            "generated_close_s": 20, "original_release_calls": list(J_CALLS)}


def j_history(root, legacy_validate):
    """Actual I non-timeout paid failure; immutable metadata, never current-I authority."""
    mapped = j_archives(root)
    pins = {key: mapped[name]["source"] for key, name in J_I_PATHS.items()}
    recovery, anchor, policy, grant = i_preflight_recovery(pins["recovery"], root, legacy_validate)
    _anchor, _policy, _grant, historical, old, _pins = i_history(root, legacy_validate)
    review = i_review_metadata(pins["review"], root, historical, old)
    exact(recovery["reviewed_source"], pins["review"], "actual I review pointer")
    exact(len(review["sources"]), 726, "whole as-executed I726")
    for name in G_REPLACEMENTS:
        exact(review["sources"][name], mapped[name]["source"], "whole as-executed seven I archives including ROOT residual")
    registered = read_pin(J_REGISTERED_PIN, root, 524288)
    exact(registered, dict(anchor, **{I_KEY: pins["recovery"]}), "whole closed286 plus sole I pointer")
    known_closed(registered)
    prior = read_pin(old[H_SOURCE_PATH]["source"], root)
    exact(read_pin(pins["freeze"], root), dict(prior, qualification_epoch=I_EPOCH, reviewed_source=pins["review"],
          sources=review["sources"], body_retry=review["sources"][I_PATH], preflight_recovery=pins["recovery"]),
          "whole executed I freeze clone of actual H")
    i_body(review["sources"][I_PATH], root)
    q = read_pin(pins["request"], root)
    f_request_identity(q, i_generated_plan("lint", root), pins["freeze"])
    admitted = copy.deepcopy(registered)
    admitted["direct_science_calls"][I_QUAL_CALLS[0]] = {"state": "admitted", "mode": "generated", "request": pins["request"], "reserved_s": 40}
    admitted["direct_science_reservations"] = {I_QUAL_CALLS[0]: {"request": pins["request"], "reserved_s": 40}}
    exact(read_pin(J_ADMITTED_PIN, root, 524288), admitted, "whole real I admission and reservation40")
    debit = {"cost_id": I_QUAL_CALLS[0], "mode": "generated", "actual_outer_full_return_s": 20.284,
             "actual_returned_exit_code": 1, "descriptor": pins["cost"]}
    from tools.banc_m3_contact_learning_admission import cost_proof
    cost = read_pin(pins["cost"], root)
    exact(cost_proof(cost, debit, lambda p: read_pin(p, root), generated=True), q, "ordinary Generated non-timeout actual I fee")
    for key in ("request", "trace", "outer", "freeze"):
        exact(cost["source_freeze" if key == "freeze" else key], pins[key], "exact failed I fee pin")
    outer = read_pin(pins["outer"], root)
    for key, value in (("all_owned_ended", True), ("reservation_released", True), ("unknown_owned_identities", {}),
                       ("stdout_eof", True), ("stderr_eof", True), ("timeout", False), ("worker_exit_code", 1),
                       ("first_error", None), ("stdout", pins["stdout"]), ("stderr", pins["stderr"])):
        exact(outer[key], value, "literal actual non-timeout I closure " + key)
    for key in ("owner_pid", "owner_creation_ticks", "worker_pid", "worker_creation_ticks"):
        need(type(outer[key]) is int and outer[key] > 0, "positive actual I owner/worker identity")
    exact(len(raw_pin(pins["stdout"], root, 262144)), 1067, "actual I stdout1067")
    exact(raw_pin(pins["stderr"], root, 262144, allow_empty=True), b"", "actual I empty stderr")
    trace = read_pin(pins["trace"], root)
    exact((trace["phase"], trace["publication"], trace["first_error"], trace["next_role_allowed"]), ("FINAL", "PENDING", None, False), "I failed, never replay or success")
    need(type(trace["dispatch_ms"]) is int and type(trace["first_final_ms"]) is int, "typed literal I clocks")
    exact((trace["dispatch_ms"], trace["first_final_ms"]), (1791453078261, 1791453098545), "actual I first FINAL timestamps")
    expected = copy.deepcopy(registered)
    expected["debits"].append(debit)
    expected["direct_science_calls"][I_QUAL_CALLS[0]] = {"state": "charged", "mode": "generated", "request": pins["request"], "reserved_s": 40}
    generated = 0.0
    for d in expected["debits"][118:]:
        if d["mode"] == "generated":
            generated += finite(d["actual_outer_full_return_s"], "ordered paid history")
    stage = float(policy["generated_stage_before_s"]) + generated
    expected.update(direct_science_generated_used_s=generated, generated_stage_used_s=stage,
                    generated_shared_cumulative_used_s=float(policy["generated_shared_before_s"]) + generated,
                    generated_shared_remaining_s=2300.0 - float(policy["generated_shared_baseline_s"]) - stage)
    closed = read_pin(J_CLOSED_PIN, root, 524288)
    exact(closed, expected, "whole286debits168calls plus sole I20.284/exit1 and charged40")
    known_closed(closed)
    ordered_totals(closed, policy, grant, NEW_CAPS)
    result = read_pin(pins["result"], root)
    exact(result["actual_once_commit"]["cost"], pins["cost"], "actual once I receipt")
    exact(result["independent_closed_readback"]["closed287"], J_CLOSED_PIN, "actual closed287 full identity")
    exact(result["actual_lint"]["immutable_trace"], pins["trace"], "actual I FINAL trace")
    exact(trace["chunks"], result["actual_lint"]["chunks"], "whole actual I collector chunks no truncation or error")
    need(type(trace["chunks"]) is list and len(trace["chunks"]) == 2, "two actual I collector receipts")
    final = trace["chunks"][-1]
    exact(final["received_ms"], trace["first_final_ms"], "actual first FINAL receipt")
    exact(set(final["result"]), {"chunk_id", "wall_time_seconds", "exit_code", "original_token_count", "output"}, "complete final collector fields")
    exact(json.loads(final["result"]["output"]), outer, "whole actual I supervisor output no collector truncation")
    exact(result["qualification"], "NOT_QUALIFIED", "I permanently failed")
    return closed, policy, grant, review, mapped, pins


def j_body(spec, root):
    exact(spec["path"], J_PATH, "only static J body")
    mapped = j_archives(root)
    prior = i_body(mapped[I_PATH]["source"], root)
    expected = dict(prior, schema="flylab.m3.contact-learning.body-runtime.v7", retry_id="contact-learning-20261008-j",
                    calls=list(J_CALLS), qualification_epoch=J_EPOCH, source_review_path=J_REVIEW_PATH, source_freeze_path=J_SOURCE_PATH,
                    qualification_lint_cleanup_path=J_CLEANUP_PATH, historical_closed287=J_CLOSED_PIN, failed_i_lint=mapped[J_I_PATHS["cost"]]["source"])
    body = read_pin(spec, root)
    exact(body, expected, "whole static I six changes three additions")
    return body


def j_current_review(spec, root, legacy_validate):
    _anchor, _policy, _grant, historical, mapped, _pins = j_history(root, legacy_validate)
    review = j_review_metadata(spec, root, historical, mapped)
    for pin in review["sources"].values():
        source_text(pin, root)
    j_body(review["sources"][J_PATH], root)
    return review


def j_lint_cleanup(spec, root, legacy_validate):
    exact(spec["path"], J_CLEANUP_PATH, "one typed immutable J cleanup descriptor")
    cleanup = read_pin(spec, root)
    anchor, policy, grant, historical, mapped, pins = j_history(root, legacy_validate)
    exact(set(cleanup), {"schema", "epoch", "scope", "allocation", "qualification_amendment", "qualification_lint_repair", "preflight_recovery",
                        "historical_closed287", "failed_i_lint", "source_before", "reviewed_source", "qualification_plan", "new_resource_grant_s", "refund_s"}, "exact14 J fields")
    for key, value in (("schema", "flylab.m3.contact-qualification-lint-cleanup.v1"), ("epoch", J_EPOCH), ("scope", "PREQUALIFICATION_ONLY"),
                       ("allocation", anchor[ALLOCATION_KEY]), ("qualification_amendment", anchor[G_KEY]), ("qualification_lint_repair", anchor[H_KEY]),
                       ("preflight_recovery", anchor[I_KEY]), ("historical_closed287", J_CLOSED_PIN), ("failed_i_lint", pins), ("source_before", J_BEFORE_PIN),
                       ("qualification_plan", j_plan(root)), ("new_resource_grant_s", 0), ("refund_s", 0)):
        exact(cleanup[key], value, "immutable J " + key)
    j_review_metadata(cleanup["reviewed_source"], root, historical, mapped)
    return cleanup, anchor, policy, grant


def j_registration(ledger, spec, root, legacy_validate, *, before_raw):
    need(type(before_raw) is bytes and before_raw == raw_pin(J_CLOSED_PIN, root, 524288), "exact whole closed287 raw before J")
    need(J_KEY not in ledger, "no duplicate alias overwrite J")
    cleanup, anchor, _policy, _grant = j_lint_cleanup(spec, root, legacy_validate)
    known_closed(ledger)
    exact(ledger, anchor, "whole closed287 before sole J pointer")
    need(ledger["direct_science_generated_used_s"] + 575 <= 1564 and ledger["generated_shared_remaining_s"] >= 575, "J555 plus distinct close20 both pools")
    j_current_review(cleanup["reviewed_source"], root, legacy_validate)
    after = copy.deepcopy(ledger)
    after[J_KEY] = copy.deepcopy(spec)
    exact(set(after) - set(ledger), {J_KEY}, "sole J pointer no grant refund transfer")
    return after


def j_current_caps(ledger, root, legacy_validate):
    cleanup, anchor, policy, grant = j_lint_cleanup(ledger[J_KEY], root, legacy_validate)
    for key, field in ((G_KEY, "qualification_amendment"), (H_KEY, "qualification_lint_repair"), (I_KEY, "preflight_recovery")):
        exact(ledger[key], cleanup[field], "unchanged immutable J ancestry")
    expected = dict(anchor, **{J_KEY: ledger[J_KEY]})
    exact(set(ledger), set(expected), "strict J whole keyset")
    prefix(ledger, anchor)
    for key, value in expected.items():
        if key not in TAIL_KEYS:
            exact(ledger[key], value, "immutable J accounting " + key)
    ordered_totals(ledger, policy, grant, NEW_CAPS)
    return dict(NEW_CAPS)


def j_freeze(spec, cleanup, cleanup_spec, root):
    exact(spec["path"], J_SOURCE_PATH, "literal fresh J freeze")
    freeze = read_pin(spec, root, BOOTSTRAP_SOURCE_CAP)
    prior = read_pin(j_archives(root)[I_SOURCE_PATH]["source"], root)
    expected = dict(prior, qualification_epoch=J_EPOCH, reviewed_source=cleanup["reviewed_source"],
                    sources=freeze["sources"], body_retry=freeze["body_retry"], qualification_lint_cleanup=cleanup_spec)
    exact(freeze, expected, "whole immutable I clone plus J pointer")
    from tools.banc_m3_startup_ray_supervisor import generated_budget_limit
    review = j_current_review(cleanup["reviewed_source"], root, generated_budget_limit)
    exact(freeze["sources"], review["sources"], "whole current J727")
    exact(freeze["body_retry"], review["sources"][J_PATH], "only reviewed static J")
    return freeze


def j_progress(ledger, cleanup, pointer, source, root):
    return _epoch_progress(ledger, cleanup, pointer, source, root, "j")


def validate_j_qualification(qualification, ledger, root, legacy_validate):
    return _epoch_qualification(qualification, ledger, root, legacy_validate, "j")


def j_dispatch(ledger, q, request_spec, root, legacy_validate, *, phase):
    return _epoch_dispatch(ledger, q, request_spec, root, legacy_validate, "j", phase=phase)


def j_admitted_identity(ledger, q, request_spec, root):
    return _epoch_admitted_identity(ledger, q, request_spec, root, "j")

def j_review_metadata(spec, root, historical, mapped):
    exact(spec["path"], J_REVIEW_PATH, "one acyclic reviewed H manifest")
    review = read_pin(spec, root)
    exact(set(review), {"schema", "epoch", "sources", "replacements", "root_reflection", "before_source",
                        "astra_review", "root_reviews", "sol_reports"}, "exact H review fields")
    for key, value in (("schema", "flylab.m3.contact-qualification.reviewed-source.v4"), ("epoch", J_EPOCH),
                       ("root_reflection", J_ROOT_PIN), ("before_source", J_BEFORE_PIN), ("astra_review", J_ASTRA_PINS)):
        exact(review[key], value, "reviewed H " + key)
    for pin in (J_ROOT_PIN, *J_ASTRA_PINS.values()):
        raw_pin(pin, root, 65536)
    names = {"docs/BANC_M3_J_QUALIFICATION_LINT_CLEANUP_ROOT_" + r + "_20261008.md" for r in ("FIRST_REVIEW", "SECOND_REVIEW", "RESIDUAL")}
    pins = review["root_reviews"]
    need(type(pins) is list and 2 <= len(pins) <= 3 and len({p["path"] for p in pins}) == len(pins)
         and {p["path"] for p in pins} <= names
         and names - {"docs/BANC_M3_J_QUALIFICATION_LINT_CLEANUP_ROOT_RESIDUAL_20261008.md"} <= {p["path"] for p in pins}, "actual ROOT FIRST/SECOND optional residual")
    allowed = {BASE + "/parent-j-qualification-lint-cleanup-sol-" + r + "-20261008/" + r + "." + ext
               for r in ("FIRST", "R1") for ext in ("md", "json")}
    sol = review["sol_reports"]
    need(type(sol) is list and len(sol) in (2, 4) and len({p["path"] for p in sol}) == len(sol)
         and {p["path"] for p in sol} <= allowed
         and {BASE + "/parent-j-qualification-lint-cleanup-sol-FIRST-20261008/FIRST." + ext for ext in ("md", "json")} <= {p["path"] for p in sol}, "actual sole Sol FIRST/optional R1")
    for pin in (*pins, *sol):
        raw_pin(pin, root, 1048576)
    vector, replaced = review["sources"], review["replacements"]
    need(type(vector) is dict and type(replaced) is dict, "typed H vector/replacements")
    exact(set(vector), set(historical["sources"]) | {J_PATH, IMMUTABLE_RECORD_PIN["path"]}, "entire I726 plus staticJ and exact record728")
    exact(vector[IMMUTABLE_RECORD_PIN["path"]], IMMUTABLE_RECORD_PIN, "compiled immutable DATA pin")
    exact(set(replaced), set(G_REPLACEMENTS), "same seven replacement identities only")
    for name, pin in vector.items():
        exact(set(pin), {"path", "bytes", "sha256"}, "exact H Source pin")
        exact(pin["path"], name, "literal H Source identity")
        need(type(pin["bytes"]) is int and 0 <= pin["bytes"] <= 2097152 and type(pin["sha256"]) is str
             and re.fullmatch("[0-9a-f]{64}", pin["sha256"]), "fixed finite Source cap")
        if name in G_REPLACEMENTS:
            need(0 < pin["bytes"] <= BOOTSTRAP_SOURCE_CAP, "existing changed Source cap")
            exact(replaced[name], {"before": mapped[name]["source"], "after": pin}, "exact before-H/after-H replacement")
        elif name not in (J_PATH, IMMUTABLE_RECORD_PIN["path"]):
            exact(pin, historical["sources"][name], "all unrelated/installed/static G Source unchanged")
    return review



def _l9_cold_proof(k, root, read, raw):
    BASE, OLD_RECOVERY, OLD300 = k.BASE, k.OLD_RECOVERY, k.OLD300
    OLD_SOURCE, REPLACEMENTS, COLD_PATHS = k.OLD_SOURCE, k.REPLACEMENTS, k.COLD_PATHS
    BODY_PATH, ROSTER_PATH, OLD_CAPS = k.BODY_PATH, k.ROSTER_PATH, k.HISTORY_CAPS
    FAILURE_HOLD = k.FAILURE_HOLD
    keys, metadata_pin, totals = k.keys, k.metadata_pin, k.totals
    spec = OLD_RECOVERY
    ANCHOR = OLD300
    EPOCH = 'contact-checkpoint-harness-20261008-l8'
    RECOVERY_PATH = OLD_RECOVERY['path']
    SOURCE_PATH = OLD_SOURCE['path']
    metadata_pin(spec, RECOVERY_PATH)
    event = read(spec)
    COLD_PINS = event['exact_l7_cold']
    keys(COLD_PINS, ('witness','source','legacy','report','request','trace','recovery','cost','before'), 'finite historical operands')
    for name, path in COLD_PATHS.items():
        metadata_pin(COLD_PINS[name], path)
    L_FREEZE = COLD_PINS['source']
    J_BODY = {"path":BASE+"/l7-cold-before/verification/banc-natural-gait-m3-20260919/b5.json","bytes":16895,"sha256":"3ba5da830fdf48d832f7968c156774829c7492536ae779e6eb46b90a7e530e1a"}
    L_ROSTER = {"path":BASE+"/l7-cold-before/verification/banc-natural-gait-m3-20260919/t5.json","bytes":47800,"sha256":"45711482ab9ff137f2bce50469adbd638dc17dd47d1215353ec720407bb80cf4"}
    call = "direct-science-20261008-contact-l7-bootstrap"
    exact(event, {"schema": "flylab.m3.checkpoint-recovery.v1", "epoch": EPOCH,
        "before_ledger": ANCHOR, "source_freeze": event["source_freeze"], "exact_l7_cold": COLD_PINS,
        "released_hold": call, "reason": "KNOWN_EXACT_ROOT_COLD_L7_FAILURE_RETAINED",
        "new_resource_grant_s": 0, "refund_s": 0}, "exact known historical cold recovery")
    metadata_pin(event["source_freeze"], SOURCE_PATH)
    trace = read(COLD_PINS["trace"])
    witness = read(COLD_PINS["witness"])
    AS_EXECUTED = witness["as_executed_sources"]
    keys(AS_EXECUTED, (*REPLACEMENTS[:2], BASE + "/parent-direct-science-account.ps1", "tools/banc_m3_startup_ray_supervisor.py"), "finite actual archived Source")
    for name, pin in AS_EXECUTED.items():
        metadata_pin(pin, BASE + "/l7-cold-before/" + name)
    cost = read(COLD_PINS["cost"])
    old = read(L_FREEZE)
    for pin, path in ((J_BODY, BODY_PATH), (L_ROSTER, ROSTER_PATH)):
        exact(old["sources"][path], dict(pin, path=path), "frozen old body/roster")
    for name, pin in AS_EXECUTED.items():
        cap = 65536 if name in REPLACEMENTS[:2] else 262144
        blob = raw(pin, cap)
        exact(old["sources"][name], dict(pin, path=name), "exact archived execution Source")
        if name == REPLACEMENTS[0]:
            text = blob.decode("utf8")
            need(text.count("KNOWN_CLOSED_ONCE_PAID_MATH6_FAILURE_RETAINED") == 1
                   and 0 <= text.index("state = validate(live, root)") < text.index("output.mkdir(parents=True)"),
                   "exact unreached refusal output")
        if name == "tools/banc_m3_startup_ray_supervisor.py":
            text = blob.decode("utf8")
            start = text.index("def _supervise(")
            guard = text.index("f_dispatch = checkpoint.dispatch", start)
            refuse = text.index("return checkpoint.initial_refusal", guard)
            need(start < guard < refuse < text.index("SharedLock(", refuse)
                   and refuse < text.index("OwnedJob(", refuse), "guard before creation")
    exact(witness, {"schema": "flylab.m3.ROOT-cold-witness.v1", "call_id": call,
        "request": COLD_PINS["request"], "source_freeze": L_FREEZE, "trace": COLD_PINS["trace"],
        "before_ledger": COLD_PINS["before"], "as_executed_sources": AS_EXECUTED,
        "supervisor_returned_exit_code": 1, "no_owned_job_created": True, "administrative_reservation_release_s": 190,
        "worker_creation_state": "NO_CREATE_ATTEMPT", "resume_outcome": "NOT_ATTEMPTED",
        "creation_attempted": False, "job_created": False, "worker_started": False, "resume_attempted": False,
        "owner_pid": None, "owner_creation_ticks": None, "worker_pid": None, "worker_creation_ticks": None,
        "worker_exit_code": None, "stdout_eof": False, "stderr_eof": False, "unknown_owned_identities": {},
        "worker_tests": "NOT_RUN"}, "distinct whole ROOT cold witness, no fabricated EOF/owner")
    for name, value in (("call_id", call), ("schema", "flylab.m3.direct-science.genuine-return.v1"),
        ("mode", "generated"), ("request", COLD_PINS["request"]), ("source_freeze", L_FREEZE),
        ("phase", "FINAL"), ("exit_code", 1), ("session_id", None), ("first_error", None),
        ("dispatch_ms", 1791493566299), ("first_final_ms", 1791493567035), ("full_return_s", 0.736)):
        exact(trace[name], value, "exact genuine cold " + name)
    need(len(trace["chunks"]) == 1 and trace["chunks"][0]["result"]["exit_code"] == 1, "one actual cold return")
    error = trace["chunks"][0]["result"]["output"]
    need(error.count("ValueError: exact recovery.reason") == 2
           and error.endswith("ValueError: exact recovery.reason\r\n"), "exact two authenticated failed stacks")
    command = "$ErrorActionPreference='Stop'\n& .venv/Scripts/python.exe -m tools.banc_m3_startup_ray_supervisor --request " + COLD_PINS["request"]["path"] + "\nexit $LASTEXITCODE"
    exact(trace["exec_args"]["cmd"], command, "exact command without alternate worker")
    exact(trace["exec_args"]["workdir"], "C:/Users/ckthd/Dev/Projects/fly_lab", "actual workspace")
    exact(cost, {"schema": "flylab.m3.ROOT-cold-full-return.v1", "cost_id": call, "mode": "generated",
        "request": COLD_PINS["request"], "source_freeze": L_FREEZE, "trace": COLD_PINS["trace"],
        "witness": COLD_PINS["witness"], "actual_outer_full_return_s": 0.736, "exit_code": 1,
        "charged_once": True, "cap_exceeded": False, "no_owned_job_created": True,
        "administrative_reservation_released": True}, "exact once actual cold cost")
    before = read(COLD_PINS["before"], 524288)
    legacy = read(COLD_PINS["legacy"], 524288)
    prior = read(COLD_PINS["recovery"])
    exact(prior["reason"], "KNOWN_CLOSED_ONCE_PAID_L6_MATH_FAILURE_RETAINED", "semantic literal preserved")
    exact(prior["source_freeze"], L_FREEZE, "old recovery Source")
    exact(prior["before_ledger"], COLD_PINS["legacy"], "old closed299")
    need(len(legacy["debits"]) == 299 and len(legacy["direct_science_calls"]) == 181, "old299/181")
    exact(legacy["direct_science_blocking_holds"], {"direct-science-20261008-contact-l6-math": FAILURE_HOLD}, "old exact HOLD")
    admitted = copy.deepcopy(legacy)
    admitted["direct_science_blocking_holds"] = {}
    admitted["direct_science_checkpoint_harness_l7_recovery"] = COLD_PINS["recovery"]
    admitted["direct_science_calls"][call] = {"request": COLD_PINS["request"], "state": "admitted",
        "mode": "generated", "reserved_s": 190}
    admitted["direct_science_reservations"][call] = {"request": COLD_PINS["request"], "reserved_s": 190}
    exact(before, admitted, "whole admitted299 with only prior known recovery/admission")
    closed = read(OLD300, 524288)
    policy, grant = read(POLICY_PIN), read(J_PIN)
    need(len(closed['debits']) == 300 and len(closed['direct_science_calls']) == 182, 'historical300/182')
    exact(closed['direct_science_policy'], POLICY_PIN, 'historical policy')
    rebuilt = copy.deepcopy(before)
    rebuilt["direct_science_calls"][call]["state"] = "charged"
    del rebuilt["direct_science_reservations"][call]
    rebuilt["debits"].append({"cost_id": call, "mode": "generated", "actual_outer_full_return_s": 0.736,
                             "actual_returned_exit_code": 1, "descriptor": COLD_PINS["cost"]})
    rebuilt["direct_science_blocking_holds"][call] = FAILURE_HOLD
    rebuilt.update(totals(rebuilt, policy, grant, OLD_CAPS))
    exact(closed, rebuilt, "whole typed300 exact cold transition")
    for key in OWNER_KEYS:
        if key != "direct_science_blocking_holds":
            exact(closed[key], {}, "no other liability")
    report = read(COLD_PINS["report"])
    for name, value in (("anchor", ANCHOR), ("before", COLD_PINS["before"]), ("source_freeze", L_FREEZE),
        ("trace", COLD_PINS["trace"]), ("request", COLD_PINS["request"]), ("witness", COLD_PINS["witness"]),
        ("cost", COLD_PINS["cost"]), ("whole300_typed_reconstructed", True), ("old299_preserved", True),
        ("charged_once", True), ("actual_s", 0.736), ("worker_tests", "NOT_RUN"), ("Original", "NOT_RUN")):
        exact(report[name], value, "independent whole cold result")
    return event, closed, policy, grant


def _l9_root_cold_proof(k, previous, policy, grant, read, raw, *, latest=False):
    need(type(latest) is bool or type(latest) is str and latest == 'l12', 'three fixed cold cases')
    current = type(latest) is str
    fee, release, d0, d1 = ((0.852, 180, 1791505192008, 1791505192860) if current else
        (0.659, 190, 1791501442307, 1791501442966) if latest else (0.617, 190, 1791497664621, 1791497665238))
    report = read(k.COLD12 if current else k.LATEST_COLD if latest else k.ROOT_COLD)
    suffix = "l12" if current else "l10" if latest else "l9"
    closed_pin = k.ANCHOR if current else k.COLD304 if latest else k.COLD302
    old_source_path = k.BASE+"/l12-s1.json" if current else k.BASE+"/l10-s2.json" if latest else k.BASE+"/l9-s1.json"
    old_anchor = k.CLOSED305 if current else k.FAILED303 if latest else k.L8_ANCHOR
    call = "direct-science-20261008-contact-"+suffix+"-bootstrap"
    for name, path in (("anchor", closed_pin["path"]),
        ("before", k.BASE+("/parent-l12-bootstrap-admitted305-before-cold-settlement-20261008.json" if current else "/parent-l10-bootstrap-admitted303-before-cold-settlement-20261008.json" if latest else "/parent-l9-bootstrap-admitted301-before-cold-settlement-20261008.json")),
        ("witness", k.BASE+"/"+suffix+"-cold-witness.json"),
        ("cost", k.BASE+"/parent-"+call+"-full-return.json"),
        ("trace", k.BASE+"/parent-"+call+"-genuine-return.json"),
        ("source_freeze", old_source_path)):
        k.metadata_pin(report[name], path)
    before = read(report["before"], 524288)
    witness, cost, trace = read(report["witness"]), read(report["cost"]), read(report["trace"])
    source = report["source_freeze"]
    freeze = read(source)
    archives = witness["as_executed_sources"]
    k.keys(archives, (*k.REPLACEMENTS[:2], "tools/banc_m3_conserved_resource.py",
                     k.BASE+"/parent-direct-science-account.ps1", "tools/banc_m3_startup_ray_supervisor.py"), "exact cold Source set")
    for name, pin in archives.items():
        k.metadata_pin(pin, k.BASE+"/"+suffix+"-cold-before/"+name)
        text = raw(pin, 65536 if name in k.REPLACEMENTS[:2] else 262144).decode("utf8")
        exact(freeze["sources"][name], dict(pin, path=name), "exact executed Source")
        if name == "tools/banc_m3_conserved_resource.py":
            if current:
                start = text.index('def _l10_failed_bootstrap(k,')
                shadow = text.index('for k, v in', start)
                late = text.index('k.integer(o[name]', shadow)
                need(start < shadow < late and text[start:].count('k.integer(o[name]') == 1, 'exact archived module shadow')
            else:
                need((text.count('"/outejson"') == 1 and text.count('"/parent-e358-execution-ledgejson"') == 1)
                     if latest else text.count("        raw = raw(pin, cap)") == 1, "exact archived reader collision")
        if name == k.REPLACEMENTS[0]:
            need(0 <= text.index("state = validate(live, root)") < text.index("output.mkdir(parents=True)"), "refusal before output")
        if name == "tools/banc_m3_startup_ray_supervisor.py":
            start = text.index("def _supervise(")
            dispatch = text.index("f_dispatch = checkpoint.dispatch", start)
            refusal = text.index("return checkpoint.initial_refusal", dispatch)
            need(start < dispatch < refusal < text.index("SharedLock(", refusal)
                 and refusal < text.index("OwnedJob(", refusal), "before Job creation")
    request = witness["request"]
    k.metadata_pin(request, k.BASE+"/parent-"+call+"-request.json")
    q = read(request)
    exact(q["call_id"], call, "actual cold call")
    exact(q["source_freeze"], source, "actual cold Source")
    if current:
        exact(source, k.SOURCE12, "fixed executed S12")
        exact(q, {'schema':"flylab.m3.direct-science.supervision.v1", 'call_id':call, 'mode':"generated", 'role':"execution",
            'source_freeze':source, 'limits':{'outer_s':180, 'child_s':160, 'cleanup_s':20, 'host_bytes':2147483648, 'output_cap_bytes':8388608},
            'output':k.BASE+"/parent-"+call, 'ledger_path':k.BASE+"/parent-e358-execution-ledger.json",
            'shared_original_lock_path':k.BASE+"/parent-e358-original-active.lock", 'generated_behavior':"command",
            'generated_argv':["-m", "pytest", *(k.TEST_MODULE+"::"+x for x in k.BOOTSTRAP_NAMES), "--tb=short", "-q",
                            "--junitxml="+k.BASE+"/parent-contact-learning-l12-bootstrap-junit.xml"]}, "whole fixed cold12 request")
    exact(witness, {'schema':"flylab.m3.ROOT-cold-witness.v1",'call_id':call,'request':request,
        'source_freeze':source,'trace':report["trace"],'before_ledger':report["before"],'as_executed_sources':archives,
        'supervisor_returned_exit_code':1,'no_owned_job_created':True,'administrative_reservation_release_s':release,
        'worker_creation_state':"NO_CREATE_ATTEMPT",'resume_outcome':"NOT_ATTEMPTED",'creation_attempted':False,
        'job_created':False,'worker_started':False,'resume_attempted':False,'owner_pid':None,'owner_creation_ticks':None,
        'worker_pid':None,'worker_creation_ticks':None,'worker_exit_code':None,'stdout_eof':False,'stderr_eof':False,
        'unknown_owned_identities':{},'worker_tests':"NOT_RUN"}, "whole distinct cold witness")
    for name, value in (("call_id",call),("mode","generated"),("request",request),("source_freeze",source),
        ("phase","FINAL"),("first_error",None),("session_id",None),("exit_code",1),
        ("dispatch_ms",d0),
        ("first_final_ms",d1),("full_return_s",fee)):
        exact(trace[name], value, "exact actual ROOT cold trace")
    error = "AttributeError: 'str' object has no attribute 'integer'" if current else "ValueError: literal metadata path" if latest else "TypeError: 'bytes' object is not callable"
    need(len(trace["chunks"]) == 1 and trace["chunks"][0]["result"]["exit_code"] == 1
         and trace["chunks"][0]["result"]["output"].count(error) == 2
         and trace["chunks"][0]["result"]["output"].endswith(error+"\r\n"), "exact double stack")
    exact(trace["exec_args"]["cmd"], "$ErrorActionPreference='Stop'\n& .venv/Scripts/python.exe -m tools.banc_m3_startup_ray_supervisor --request "
          +request["path"]+"\nexit $LASTEXITCODE", "exact command")
    exact(trace["exec_args"]["workdir"], "C:/Users/ckthd/Dev/Projects/fly_lab", "actual root")
    exact(cost, {'schema':"flylab.m3.ROOT-cold-full-return.v1",'cost_id':call,'mode':"generated",
        'request':request,'source_freeze':source,'trace':report["trace"],'witness':report["witness"],
        'actual_outer_full_return_s':fee,'exit_code':1,'charged_once':True,'cap_exceeded':False,
        'no_owned_job_created':True,'administrative_reservation_released':True}, "whole known once cold fee")
    recovery_key = "direct_science_checkpoint_harness_"+suffix+"_recovery"
    prior_pin = before[recovery_key]
    k.metadata_pin(prior_pin, k.BASE+("/l12-recovery1.json" if current else "/l10-recovery2.json" if latest else "/l9-recovery1.json"))
    exact(prior_pin, report["recovery"], "exact old recovery pin")
    prior = read(prior_pin)
    expected_prior = {'schema':"flylab.m3.checkpoint-recovery.v1",
        'epoch':"contact-checkpoint-harness-20261008-"+suffix, 'before_ledger':old_anchor,
        'source_freeze':source, 'failed_bootstrap':k.FAIL_REPORT,
        'released_hold':"direct-science-20261008-contact-l8-bootstrap",
        'reason':"KNOWN_CLOSED_ONCE_PAID_L8_BOOTSTRAP_FAILURE_RETAINED",
        'new_resource_grant_s':0, 'refund_s':0}
    if latest:
        expected_prior.update(exact_l9_cold=k.ROOT_COLD, latest_failure=k.LATEST_FAILURE,
            released_hold="direct-science-20261008-contact-l9r2-bootstrap",
            reason="KNOWN_CLOSED_ONCE_PAID_L9R2_BOOTSTRAP_FAILURE_RETAINED")
    if current:
        expected_prior.update(exact_l10_cold=k.LATEST_COLD, latest_timeout=k.TIMEOUT_REPORT,
            released_hold="direct-science-20261008-contact-l11-bootstrap",
            reason="KNOWN_CLOSED_ONCE_PAID_L11_TIMEOUT_FAILURE_RETAINED")
    exact(prior, expected_prior, "actual old recovery")
    expected = copy.deepcopy(previous)
    expected["direct_science_blocking_holds"] = {}
    expected[recovery_key] = prior_pin
    expected["direct_science_calls"][call] = {'request':request,'state':"admitted",'mode':"generated",'reserved_s':release}
    expected["direct_science_reservations"][call] = {'request':request,'reserved_s':release}
    exact(before, expected, "whole admitted301")
    expected["direct_science_calls"][call]["state"] = "charged"
    expected["direct_science_reservations"].pop(call)
    expected["debits"].append({'cost_id':call,'mode':"generated",'actual_outer_full_return_s':fee,
                                  'actual_returned_exit_code':1,'descriptor':report["cost"]})
    expected["direct_science_blocking_holds"][call] = k.FAILURE_HOLD
    expected.update(k.totals(expected, policy, grant, k.HISTORY_CAPS))
    closed = read(closed_pin, 524288)
    exact(closed, expected, "whole302 exact cold transition")
    return closed


def _l10_failed_bootstrap(k, report_pin, source, previous, policy, grant, closed_pin,
                         recovery_key, recovery_pin, suffix, archive, fee, dispatch, final, read, raw):
    BASE, BODY_PATH, ROSTER_PATH = k.BASE, k.BODY_PATH, k.ROSTER_PATH
    REPLACEMENTS, TEST_MODULE, BOOTSTRAP_NAMES = k.REPLACEMENTS, k.TEST_MODULE, k.BOOTSTRAP_NAMES
    OLD_CAPS, FAILURE_HOLD = k.HISTORY_CAPS, k.FAILURE_HOLD
    metadata_pin, totals, closure = k.metadata_pin, k.totals, k.closure
    integer = k.integer
    from xml.etree import ElementTree as ET
    report = read(report_pin)
    need(suffix in ("l8", "l9r2", "l11"), "three fixed failed cases")
    timeout = suffix == "l11"
    latest = suffix != "l8"
    bootstrap = report if latest else report["bootstrap"]
    mask = (False, False, True, False) if latest else (True, True, True, False)
    call = "direct-science-20261008-contact-" + suffix + "-bootstrap"
    exact(report["anchor"] if latest else report["ledger"], closed_pin, "actual closed301")
    exact(report["source_freeze"], source, "actual executed L8 Source")
    if timeout:
        exact({name: report[name] for name in ("full4", "exit_code", "JUnit", "stdout", "actual_s")},
              {'full4':"INCOMPLETE", 'exit_code':124, 'JUnit':"NOT_CREATED", 'stdout':".F", 'actual_s':fee}, "actual incomplete timeout")
        exact((fee, dispatch, final, source, closed_pin, archive),
              (170.664, 1791503352604, 1791503523268, k.SOURCE11, k.CLOSED305, "l12-before"), "only fixed actual L11 timeout")
        metadata_pin(recovery_pin, BASE+"/l11-recovery1.json")
        exact(read(recovery_pin), {'schema':"flylab.m3.checkpoint-recovery.v1",
            'epoch':"contact-checkpoint-harness-20261008-l11", 'before_ledger':k.COLD304,
            'source_freeze':source, 'failed_bootstrap':k.FAIL_REPORT, 'exact_l9_cold':k.ROOT_COLD,
            'latest_failure':k.LATEST_FAILURE, 'exact_l10_cold':k.LATEST_COLD,
            'released_hold':"direct-science-20261008-contact-l10-bootstrap",
            'reason':"KNOWN_EXACT_ROOT_COLD_L10_FAILURE_RETAINED", 'new_resource_grant_s':0, 'refund_s':0}, "whole actual L11 recovery")
    else:
        exact(({key: report[key] for key in ("passed", "failed", "errors", "skipped", "actual_s")}
                 if latest else bootstrap),
                ({'passed':1, 'failed':3, 'errors':0, 'skipped':0, 'actual_s':fee} if latest
                 else dict(bootstrap, passed=3, failed=1, errors=0, skipped=0, exit_code=1, fee_s=fee)), "actual full4 failed")
    old = read(source)
    for path in (BODY_PATH, ROSTER_PATH):
        pin = dict(old["sources"][path], path=BASE+"/"+archive+"/"+path)
        metadata_pin(pin, BASE+"/"+archive+"/"+path)
        raw(pin)
        exact(old["sources"][path], dict(pin, path=path), "executed old body/roster")
    for name in REPLACEMENTS[:2] + (BASE + "/parent-direct-science-account.ps1",):
        pin = dict(old["sources"][name], path=BASE + "/" + archive + "/" + name)
        raw(pin, 65536 if name in REPLACEMENTS[:2] else 262144)
    cost_pin = bootstrap["cost"]
    junit_path = BASE+"/parent-contact-learning-"+suffix+"-bootstrap-junit.xml"
    junit_pin = None if timeout else bootstrap["junit"]
    metadata_pin(cost_pin, BASE + "/parent-" + call + "-full-return.json")
    if not timeout:
        metadata_pin(junit_pin, junit_path)
    cost = read(cost_pin)
    request, trace, outer = cost["request"], cost["trace"], cost["outer"]
    for p, path in ((request, BASE + "/parent-" + call + "-request.json"),
                    (trace, BASE + "/parent-" + call + "-genuine-return.json"),
                    (outer, BASE + "/parent-" + call + "/outer.json"),
):
        metadata_pin(p, path)
    if not latest:
        metadata_pin(report["nested_failure"], BASE + "/l8-nested-last-observation.json")
    q, t, o = read(request), read(trace), read(outer)
    exact(q, {"schema":"flylab.m3.direct-science.supervision.v1", "call_id":call, "mode":"generated",
        "role":"execution", "source_freeze":source, "limits":{'outer_s':190,'child_s':170,'cleanup_s':20,
        'host_bytes':2147483648,'output_cap_bytes':8388608}, "output":BASE+"/parent-"+call,
        "ledger_path":BASE+"/parent-e358-execution-ledger.json", "shared_original_lock_path":SHARED_LOCK,
        "generated_behavior":"command", "generated_argv":["-m","pytest", *[TEST_MODULE+"::"+n for n in BOOTSTRAP_NAMES],
        "--tb=short","-q","--junitxml="+junit_path]}, "whole actual L8 request")
    exact(t["schema"], "flylab.m3.direct-science.genuine-return.v1", "actual collector")
    for field, v in (("call_id",call),("mode","generated"),("request",request),("source_freeze",source),
                 ("phase","FINAL"),("first_error",None),("dispatch_ms",dispatch),
                 ("first_final_ms",final),("full_return_s",fee),("exit_code",124 if timeout else 1)):
        exact(t[field], v, "actual L8 trace "+field)
    exact(t["exec_args"]["cmd"], "$ErrorActionPreference='Stop'\n& .venv/Scripts/python.exe -m tools.banc_m3_startup_ray_supervisor --request "+request["path"]+"\nexit $LASTEXITCODE", "actual command")
    exact(t["exec_args"]["workdir"], "C:/Users/ckthd/Dev/Projects/fly_lab", "actual root")
    exact(o["request"], request, "actual outer binding")
    need(closure(o) and o["exit_code"] == (124 if timeout else 1) and o["worker_creation_state"] == "RESUMED"
         and o["timeout"] is timeout and (o["first_error"] is None if not timeout else
             o["first_error"]["type"] == "TimeoutError" and o["first_error"]["message"] ==
             "same ENTRY child interval, owned Job termination required"), "actual known parent closure")
    if timeout:
        exact(o["secondary_errors"], [], "no timeout secondary error")
        for name in ("owner_pid", "owner_creation_ticks", "worker_pid", "worker_creation_ticks"):
            integer(o[name], "actual timeout positive identity", positive=True)
        need(o["owner_pid"] != o["worker_pid"], "distinct timeout owner/worker")
        for name, expected in (("stdout", {'path':BASE+"/parent-"+call+"/stdout.bin", 'bytes':2,
              'sha256':"56df4f6ec7b11cb0d5f94b41ec7e21b122f4fdf68c717f43ee40be1003f33d0f"}),
              ("stderr", {'path':BASE+"/parent-"+call+"/stderr.bin", 'bytes':0,
              'sha256':"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"})):
            exact(o[name], expected, "exact timeout raw log")
            exact(raw(o[name]), b".F" if name == "stdout" else b"", "whole actual incomplete log")
        for name in ("cost", "trace", "request", "outer"):
            exact(report[name], {'cost':cost_pin, 'trace':trace, 'request':request, 'outer':outer}[name], "whole timeout correlated record")
    expected_cost = {'schema':"flylab.m3.direct-science.full-return.v1", 'cost_id':call, 'mode':"generated",
        'request':request,'source_freeze':source,'trace':trace,'outer':outer,'actual_outer_full_return_s':fee,
        'exit_code':124 if timeout else 1,'charged_once':True,'cap_exceeded':False}
    expected_cost.update({k:o[k] for k in ("worker_started","reservation_released","resume_outcome",
        "resume_attempted","worker_creation_state","creation_attempted","all_owned_ended")})
    exact(cost, expected_cost, "whole once known L8 cost")
    if not timeout:
        suites = ET.fromstring(raw(junit_pin)).findall("testsuite")
        need(len(suites) == 1 and all(suites[0].get(k) == v for k,v in
               (("tests","4"),("failures","3" if latest else "1"),("errors","0"),("skipped","0"))), "whole failed4 no skips")
        cases = suites[0].findall("testcase")
        exact([(c.get("classname"),c.get("name")) for c in cases],
                [("tests.test_c_banc_checkpoint_harness_runtime",n) for n in BOOTSTRAP_NAMES], "ordered full4")
        need(all((not list(c) if passed else len(list(c)) == 1 and next(iter(c)).tag == "failure")
                   for c, passed in zip(cases, mask, strict=True)), "sole last actual failure")
        if latest:
            need("same parent ENTRY Native deadline; owned descendant termination required" in cases[3].get("message", "")
                   or "same parent ENTRY Native deadline; owned descendant termination required" in ET.tostring(cases[3], encoding="unicode"),
                   "actual nested same ENTRY timeout/closure")
        else:
            nested = read(report["nested_failure"])
            need(nested["returncode"] == 124 and nested["timeout"] is True
                   and nested["first_error"] == {"type":"TimeoutError","message":"same parent ENTRY Native deadline; owned descendant termination required"}
                   and nested["all_owned_ended"] is True and nested["stdout_eof"] is True and nested["stderr_eof"] is True
                   and nested["unknown_owned_identities"] == {}, "actual nested same ENTRY timeout/closure")
    expected = copy.deepcopy(previous)
    expected["direct_science_blocking_holds"] = {}
    expected[recovery_key] = recovery_pin
    expected["direct_science_blocking_holds"] = {call:FAILURE_HOLD}
    expected["direct_science_calls"][call] = {'request':request,'state':"charged",'mode':"generated",'reserved_s':190}
    expected["debits"].append({'cost_id':call,'mode':"generated",'actual_outer_full_return_s':fee,
                                  'actual_returned_exit_code':124 if timeout else 1,'descriptor':cost_pin})
    expected.update(totals(expected, policy, grant, OLD_CAPS))
    closed = read(closed_pin, 524288)
    exact(closed, expected, "whole typed301 exact old300 retained")
    need(len(closed["direct_science_calls"]) == (187 if timeout else (185 if latest else 183)), "actual183")
    return closed



# D19 definitions only. The generator embeds the finite literal capsule in d19_capsule.
def d19_whole(a, b, ordered=True):
 import math
 import struct
 need(type(a) is type(b), 'D19 exact scalar/container type')
 if type(b) is dict:
  need((tuple(a) == tuple(b)) if ordered else (set(a) == set(b)), 'D19 exact keys/order')
  for key in b:
   d19_whole(a[key], b[key], ordered)
 elif type(b) in (list, tuple):
  need(len(a) == len(b), 'D19 exact array')
  for x, y in zip(a, b, strict=True):
   d19_whole(x, y, ordered)
 elif type(b) is float:
  need(math.isfinite(a) and math.isfinite(b) and struct.pack('>d', a) == struct.pack('>d', b), 'D19 binary64')
 else:
  need(a == b, 'D19 exact value')


def d19_json(raw):
 import json
 import math
 def pairs(items):
  result = {}
  for key, value in items:
   need(key not in result, 'D19 duplicate JSON key')
   result[key] = value
  return result
 def constant(value):
  raise ValueError('D19 nonfinite JSON ' + value)
 value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
 def walk(node, depth):
  need(depth <= 64, 'D19 bounded JSON depth')
  if type(node) is dict:
   for item in node.values(): walk(item, depth + 1)
  elif type(node) is list:
   for item in node: walk(item, depth + 1)
  elif type(node) is float:
   need(math.isfinite(node), 'D19 finite JSON number')
 walk(value, 0)
 return value


def d19_raw(pin, root):
 c = d19_capsule()
 need(type(pin) is dict and set(pin) == {'path', 'bytes', 'sha256'}, 'D19 exact evidence pin')
 candidates = [p for p in c['finite_pins'] if p == pin]
 for row in c['archives']:
  if pin == row['logical_current_before']:
   d19_whole(pin, row['logical_current_before'])
   candidates.append(row['immutable_archive'])
  elif pin == row['immutable_archive']:
   candidates.append(pin)
 need(candidates and all(p == candidates[0] for p in candidates), 'D19 finite exact archive resolution only')
 resolved = candidates[0]
 cap = next(item['cap'] for item in c['routes'] if item['pin'] == resolved)
 need(type(pin['bytes']) is int and type(resolved['bytes']) is int and 0 <= resolved['bytes'] <= cap, 'D19 nonbool exact route cap')
 raw = raw_pin(resolved, root, cap, allow_empty=resolved['bytes'] == 0)
 return raw


def d19_read(pin, root):
 return d19_json(d19_raw(pin, root))


def d19_proof(root):
 c = d19_capsule()
 for pin in c['proof_pins'] + c['receipt_pins']:
  d19_raw(pin, root)
 for row in c['archives']:
  d19_whole([row['logical_current_before']['bytes'], row['logical_current_before']['sha256']],
    [row['immutable_archive']['bytes'], row['immutable_archive']['sha256']])
  d19_raw(row['logical_current_before'], root)
 t, o, q, capture = [d19_read(c[name], root) for name in ('trace', 'outer', 'request', 'complete')]
 need(d19_raw(c['wrapper'], root).decode('utf8') == c['wrapper_text'], 'D19 actual archived wrapper bytes')
 d19_whole(q['source_freeze'], c['source'])
 d19_whole(q['generated_argv'], c['generated_argv'])
 d19_whole(q['limits'], {'outer_s':540, 'child_s':520, 'cleanup_s':20, 'host_bytes':2147483648, 'output_cap_bytes':8388608}, False)
 need(q['call_id'] == c['call_id'] and q['mode'] == 'generated' and q['output'] == c['output_path'], 'D19 immutable old request')
 initial = d19_read(c['initial'], root)
 d19_whole(capture['initial'], initial)
 need(type(capture['polls']) is list and len(capture['polls']) == 6 and type(t['chunks']) is list and len(t['chunks']) == 7, 'D19 one initial/six polls')
 d19_whole(t['chunks'][0]['result'], initial['initial'])
 d19_whole(t['chunks'][0]['received_ms'], initial['observed_ms'])
 need('exit_code' not in initial['initial'] and initial['initial']['session_id'] == c['session_id'], 'D19 initial running')
 previous = c['dispatch_ms']
 for i, pin in enumerate(c['polls']):
  poll = d19_read(pin, root)
  d19_whole(capture['polls'][i], poll)
  d19_whole(t['chunks'][i+1]['received_ms'], poll['observed_ms'])
  d19_whole(t['chunks'][i+1]['result'], poll['actual_tool_return'])
  need(type(poll['observed_ms']) is int and previous <= poll['observed_ms'], 'D19 observed timing order')
  previous = poll['observed_ms']
  if i < 5:
   need('exit_code' not in poll['actual_tool_return'] and poll['actual_tool_return']['session_id'] == c['session_id'], 'D19 still same running session')
 final = capture['polls'][-1]
 need(final['observed_ms'] == c['first_final_ms'] and final['actual_tool_return']['exit_code'] == 1 and 'session_id' not in final['actual_tool_return'], 'D19 first actual collector FINAL1')
 d19_whole(d19_json(final['actual_tool_return']['output']), o, False)
 for name, expected in (('session_spent', True), ('action_spent', True), ('automatic_worker_retry', False), ('incurred_once_s', 525.537), ('dispatch_ms', c['dispatch_ms']), ('first_FINAL_observed_ms', c['first_final_ms']), ('session_id', c['session_id']), ('final_collector_exit_code', 1), ('native_supervisor_exit_code', 124)):
  d19_whole(capture[name], expected)
 actor = d19_read(c['actor_capture'], root)
 first = actor['first_FINAL']['actual_tool_return']
 need(actor['initial']['actual_tool_return']['chunk_id'] == 'f808ac' and actor['initial']['actual_tool_return']['session_id'] == 70332 and first['chunk_id'] == 'f46f0c' and type(first['exit_code']) is int and first['exit_code'] == 0 and 'session_id' not in first, 'D19 actual completed metadata actor')
 controls = d19_json(actor['initial']['actual_tool_return']['output'].splitlines()[0].removeprefix('R4_METADATA_CONTROLS '))
 d19_whole(controls, d19_read(c['settlement'], root)['controls'])
 need(controls['passed'] is True and controls['actual_r4_replayed'] is False and type(controls['count']) is int and controls['count'] == 44 and len(controls['cases']) == len(set(controls['cases'])) == 44, 'D19 exact actual44 receipt')
 need(first['output'].splitlines()[0] == 'R4_ACTUAL_PROOF_CALLS 3', 'D19 three actual fee proofs, not runtime PASS')
 d19_whole(d19_json(first['output'].splitlines()[-1]), c['actor_return'])
 launch = d19_read(c['launch'], root)
 d19_whole(launch['actor'], c['actor'])
 old_source = d19_read(c['source'], root)
 old_review = d19_read(c['old_review'], root)
 d19_whole(old_source['reviewed_source'], c['old_review'])
 d19_whole(old_source['sources'], old_review['sources'])
 need(len(old_source['sources']) == 871, 'D19 old871 binding')
 return t, o, q


def d19_historical_exit(t, o, request_pin, root):
 c = d19_capsule()
 d19_whole(request_pin, c['request'])
 actual_t, actual_o, _q = d19_proof(root)
 d19_whole(t, actual_t)
 d19_whole(o, actual_o)
 d19_whole(t['exec_args'], {'cmd':c['command'], 'workdir':c['workdir']})
 need(type(t['dispatch_ms']) is int and type(t['first_final_ms']) is int and t['dispatch_ms'] == c['dispatch_ms'] and t['first_final_ms'] == c['first_final_ms'] > t['dispatch_ms'], 'D19 exact firstFINAL interval')
 d19_whole(t['full_return_s'], (t['first_final_ms'] - t['dispatch_ms'])/1000.0)
 d19_whole(t['full_return_s'], 525.537)
 for name, value in (('exit_code', 1), ('phase', 'FINAL'), ('first_error', None), ('call_id', c['call_id']), ('mode', 'generated'), ('request', request_pin), ('source_freeze', c['source'])):
  d19_whole(t[name], value)
 for name in ('timeout', 'worker_started', 'creation_attempted', 'job_created', 'resume_attempted', 'all_owned_ended', 'reservation_released', 'stdout_eof', 'stderr_eof'):
  d19_whole(o[name], True)
 for name, value in (('exit_code', 124), ('worker_exit_code', 124), ('worker_creation_state', 'RESUMED'), ('resume_outcome', 'RESUMED'), ('unknown_owned_identities', {}), ('secondary_errors', [])):
  d19_whole(o[name], value)
 need(o['first_error']['type'] == 'TimeoutError', 'D19 known timeout')
 need(all(type(o[name]) is int and o[name] > 0 for name in ('owner_pid', 'worker_pid', 'owner_creation_ticks', 'worker_creation_ticks')) and o['owner_pid'] != o['worker_pid'], 'D19 actual PID/FILETIME')
 return True


def d19_historical_fee(k, cost_pin, debit, source, root):
 c = d19_capsule()
 d19_whole(cost_pin, c['cost'])
 d19_whole(source, c['source'])
 t, o, q = d19_proof(root)
 need(d19_historical_exit(t, o, c['request'], root) is True, 'D19 single Boolean historical proof')
 cost = d19_read(cost_pin, root)
 expected = {'schema':'flylab.m3.direct-science.full-return.v1', 'cost_id':c['call_id'], 'mode':'generated', 'request':t['request'], 'source_freeze':t['source_freeze'],
     'trace':{'bytes':c['trace']['bytes'], 'path':c['trace']['path'], 'sha256':c['trace']['sha256']}, 'outer':{'bytes':c['outer']['bytes'], 'path':c['outer']['path'], 'sha256':c['outer']['sha256']},
     'actual_outer_full_return_s':525.537, 'exit_code':1, 'charged_once':True, 'all_owned_ended':True, 'reservation_released':True, 'cap_exceeded':False, 'worker_started':True,
     'worker_creation_state':'RESUMED', 'creation_attempted':True, 'resume_attempted':True, 'resume_outcome':'RESUMED', 'native_supervisor_exit_code':124, 'actual_collector_shell_exit_code':1, 'shell_deadline_status_pair_observed':True}
 d19_whole(cost, expected)
 d19_whole(debit, {'cost_id':c['call_id'], 'mode':'generated', 'actual_outer_full_return_s':525.537, 'actual_returned_exit_code':1, 'descriptor':c['cost']})
 need(k['closure'](o) is True, 'D19 actual known closure')
 return q, cost, o, True


def d19_paid323(k, root):
 import copy
 c = d19_capsule()
 previous, policy, grant = k['terminal_baseline'](root)
 need(len(previous['debits']) == 322 and len(previous['direct_science_calls']) == 204, 'D19 independent previous322/204')
 pre = d19_read(c['before_admission'], root)
 expected_pre = copy.deepcopy(previous)
 need(expected_pre['direct_science_blocking_holds'] == {c['prior_call']: 'L stage unsuccessful; successor blocked'}, 'D19 sole prior r3 failure')
 expected_pre['direct_science_blocking_holds'] = {}
 expected_pre[k['KEY']] = d19_read(c['before'], root)[k['KEY']]
 d19_whole(pre, expected_pre, False)
 admitted = d19_read(c['admitted'], root)
 expected_admitted = copy.deepcopy(pre)
 expected_admitted['direct_science_calls'][c['call_id']] = {'reserved_s':540, 'request':c['request'], 'state':'admitted', 'mode':'generated'}
 expected_admitted['direct_science_reservations'][c['call_id']] = {'request':c['request'], 'reserved_s':540}
 d19_whole(admitted, expected_admitted)
 before = d19_read(c['before'], root)
 expected_before = copy.deepcopy(admitted)
 expected_before['direct_science_reservations'] = {}
 expected_before['direct_science_pending_actuals'] = {c['call_id']: {'request':c['request'], 'state':'actual_pending', 'ownership_closed':True}}
 d19_whole(before, expected_before, False)
 paid = d19_read(c['terminal'], root)
 debit = {'cost_id':c['call_id'], 'mode':'generated', 'actual_outer_full_return_s':525.537, 'actual_returned_exit_code':1, 'descriptor':c['cost']}
 d19_historical_fee(k, c['cost'], debit, c['source'], root)
 expected = copy.deepcopy(before)
 expected['debits'].append(debit)
 expected['direct_science_calls'][c['call_id']]['state'] = 'charged'
 for name in ('direct_science_workers', 'direct_science_reservations', 'direct_science_pending_actuals'):
  expected[name].pop(c['call_id'], None)
 expected['direct_science_blocking_holds'][c['call_id']] = 'L stage unsuccessful; successor blocked'
 totals = k['totals'](expected, policy, grant, k['OLD_CAPS'])
 expected.update(totals)
 d19_whole(paid, expected)
 reverse = copy.deepcopy(paid)
 reverse['debits'].pop()
 reverse['direct_science_calls'][c['call_id']]['state'] = 'admitted'
 for name in ('direct_science_workers', 'direct_science_reservations', 'direct_science_pending_actuals', 'direct_science_blocking_holds', *totals):
  reverse[name] = copy.deepcopy(before[name])
 d19_whole(reverse, before)
 need(len(paid['debits']) == 323 and len(paid['direct_science_calls']) == 205, 'D19 paid323/205 exactly once')
 for name in OWNER_KEYS:
  d19_whole(paid[name], {c['call_id']: 'L stage unsuccessful; successor blocked'} if name == 'direct_science_blocking_holds' else {})
 return paid, policy, grant


def d19_supplement(k, root):
 c = d19_capsule()
 evidence = d19_read(c['supplement'], root)
 d19_whole(evidence, c['supplement_value'])
 need(evidence['status'] == 'PRIVATE_STATIC_IMPLEMENTATION_NOT_ADOPTION', 'D19 honest implementation provenance')
 need(evidence['before_source'] == c['source'] and evidence['reviewed_source_archive'] == c['old_review'], 'D19 unique actual before Source association')
 need(len({p['path'] for p in evidence['ordered_inputs']}) == len(evidence['ordered_inputs']), 'D19 unique finite ordered supplemental inputs')
 for pin in evidence['ordered_inputs']: d19_raw(pin, root)
 return evidence


def d19_event(k,source,root):
 c=d19_capsule();ct=k['data'](k['CURRENT_CONTRACT'],root);b=k['data'](ct['d19_fixed_history'],root)
 f=k['data'](source,root);immutable=d19_json(raw_pin(source,root))
 try:d19_whole(f,immutable)
 except ValueError as error:raise ValueError('D19 decoded current Source metadata differs from raw') from error
 rv=k['data'](f['reviewed_source'],root);d19_whole(rv,d19_json(raw_pin(f['reviewed_source'],root)));k['provenance'](rv,root)
 paid=k['data'](b['independent_expected'],root,524288);d19_supplement(k,root);raw_pin(ct['d19_writer_decision'],root)
 return {'schema':'flylab.m3.s28.d19.resource-amendment.v1','epoch':k['EPOCH'],'event_id':'D19_390_R63_0','before_ledger':b['authenticated_actual'],'source_freeze':source,'review':f['reviewed_source'],'settlement':b['settlement'],'cost':b['fee'],'previous_recovery':paid[k['KEY']],'released_hold':'direct-science-20261009-contact-s28-bootstrap-r63','supplement':c['supplement'],'adoption_closure':d19_r1_closure(source,root)[0],'old_caps':ct['old_caps'],'new_caps':k['CAPS'],'delta_s':{'generated_global':0,'generated_local':0,'original_global':0},'combined_global_s':101400,'new_resource_grant_s':0,'refund_s':0,'reset':False,'resource_decision':ct['d19_resource_decision'],'writer_protocol':'current-insertion-order-v1','writer_decision':ct['d19_writer_decision']}


def d19_recovery(k, spec, root):
 need(not k['_HISTORY_BUSY'], 'D19 no reentrant checkpoint proof')
 owned=object();k['_HISTORY_BUSY'].add(owned)
 try:
  ct=d19_current_contract(k,root);cut=k['data'](ct['d19_fixed_history'],root)
  d19_current_pin(k,spec,k['RECOVERY_PATH']);event=k['data'](spec,root)
  d19_whole(event,d19_event(k,event['source_freeze'],root))
  paid=k['data'](cut['independent_expected'],root,524288)
  need(len(paid['debits'])==ct['baseline_debits'] and len(paid['direct_science_calls'])==ct['baseline_paid_calls'],'Certified fixed history counts')
  return event,paid,k['data'](cut['policy'],root),k['data'](cut['grant'],root)
 except Exception:
  k['_HISTORY_PROOF'].clear()
  raise
 finally:k['_HISTORY_BUSY'].discard(owned)


def d19_current_contract(k, root):
 import json
 ct = k['data'](k['CURRENT_CONTRACT'], root)
 immutable=json.loads(raw_pin(k['CURRENT_CONTRACT'],root,262144))
 try:d19_whole(ct,immutable)
 except ValueError as error:raise ValueError('D19 decoded current contract metadata differs from raw') from error
 need(type(ct['baseline_debits']) is int and ct['baseline_debits']==390 and type(ct['baseline_paid_calls']) is int and ct['baseline_paid_calls']==272, 'D19 sealed current boundary390/272')
 d19_whole(ct['pinned_checkpoint']['post_counts'],{'debits':390,'paid_calls':272,'bootstrap_debits':391,'bootstrap_calls':273,'qualified_debits':395,'qualified_calls':277,'body_debits':396,'body_calls':278,'audit_debits':397,'audit_calls':279})
 d19_whole([x['call'] for x in ct['sequence']],list(k['SEQUENCE']))
 d19_whole(['verification/'+p for p in ct['d19_metadata_paths']],list(k['LEDGER_METADATA_PATHS'][-100:]))
 d19_whole(ct['new_caps'],{'generated_global':88530,'generated_local':87794,'original_global':12870})
 d19_whole(ct['d19_fixed_history'],{'path':'verification/banc64/c.json','bytes':6877,'sha256':'2f94eceef44457c7e13a2a8cada0c76133e01a8a766c4ab680099531b6a4415c'})
 cut=k['data'](ct['d19_fixed_history'],root)
 from concurrent.futures import ThreadPoolExecutor
 def check(row):
  opts={} if row['opaque_stream'] is False else {'opaque':row['opaque_stream']}
  raw_pin(row['archive'],root,row['cap'],allow_empty=row['archive']['bytes']==0,**opts)
 with ThreadPoolExecutor(max_workers=4) as pool:
  for _ in pool.map(check,k['data'](cut['historical_inventory'],root)):pass
 actual=k['data'](cut['authenticated_actual'],root,524288);expected=k['data'](cut['independent_expected'],root,524288)
 d19_whole(actual,expected)
 ap=ct['d19_predecessor_allocation_present'];need(type(ap) is bool and ap==(k['ALLOCATION_KEY'] in actual) and ap==(k['ALLOCATION_KEY'] in expected),'D19 predecessor allocation presence')
 return ct


def d19_recovered_snapshot(k, spec, root):
 event,paid,_policy,_grant=d19_recovery(k,spec,root)
 cut=k['data'](d19_current_contract(k,root)['d19_fixed_history'],root)
 return d19_r1_transfer(k,paid,raw_pin(cut['authenticated_actual'],root,524288),event,spec,root)


def d19_current_exit(k, t, o, request_pin, root):
 q = k['data'](request_pin, root)
 need(q['call_id'] in k['SEQUENCE'], 'D19 only current seven exact calls')
 k['identity'](q, request_pin, q['source_freeze'], root)
 need(type(t['exec_args']) is dict and tuple(t['exec_args']) == ('cmd', 'workdir'), 'D19 exact execution policy shape/order')
 # Current policy is an exact request-bound direct invocation, not the old r4 wrapper.
 expected = {'cmd':'& .venv/Scripts/python.exe -m tools.banc_m3_startup_ray_supervisor --request ' + request_pin['path'], 'workdir':str(__import__('pathlib').Path(root).absolute()).replace('\\', '/')}
 d19_whole(t['exec_args'], expected)
 if t['exit_code'] == o['exit_code']:
  need(type(t['exit_code']) is int and type(o['exit_code']) is int, 'D19 typed equal exits')
  return True
 need(type(t['exit_code']) is int and t['exit_code'] == 1 and type(o['exit_code']) is int and o['exit_code'] == 124 and type(o['worker_exit_code']) is int and o['worker_exit_code'] == 124, 'D19 exact current unequal exits')
 need(o.get('timeout') is True and (o.get('first_error') or {}).get('type') == 'TimeoutError' and k['closure'](o) is True, 'D19 known current shell deadline closure')
 return True


def d19_request_policy(k, q, request_pin):
 call = q['call_id']
 need(call in k['SEQUENCE'], 'D19 exact current role')
 path = k['BASE']+'/s28-d19bz-full4-request.json' if call == k['BOOTSTRAP_CALL'] else k['BASE']+'/parent-'+call+'-request.json'
 k['metadata_pin'](request_pin, path)


def d19_stage(fixture, stage, boundary, command=None, child=None):
 import ctypes
 import os
 import time
 from pathlib import Path

 # This function is called only inside the existing paid fixed4/current Native path.
 from tools import banc_m3_contact_learning_admission as admission
 from tools import banc_m3_startup_ray_supervisor as supervisor
 c = d19_capsule()
 need(stage in ('fixture', 'source', 'native') and boundary in ('BEGIN', 'END', 'BEFORE_SPAWN', 'AFTER_SPAWN'), 'D19 finite timing stages')
 token = d19_json(os.environ['FLYLAB_DIRECT_SCIENCE_OWNER'].encode('utf8'))
 actual = d19_json(raw_pin(text_spec(SHARED_LOCK, admission.ROOT, 16384), admission.ROOT, 16384))
 d19_whole(actual, d19_owner_wire(token))
 need(supervisor.creation_ticks(token['owner_pid']) == token['owner_creation_ticks'], 'D19 live PID/FILETIME parent')
 q = read_pin(token['request'], admission.ROOT)
 need(q['call_id'] == token['call_id'] and q['call_id'] in __import__('tools.banc_m3_checkpoint_harness_runtime',fromlist=['SEQUENCE']).SEQUENCE and q['source_freeze']['path'] == c['source']['path'], 'D19 same exact current request')
 need(type(token['entry_counter_s']) is float and 0 < token['entry_counter_s'] <= time.perf_counter(), 'D19 genuine same parent ENTRY')
 need(supervisor.in_named_job(token['job_name']), 'D19 current fixture in owned parent Job')
 get_command = ctypes.windll.kernel32.GetCommandLineW
 get_command.restype = ctypes.c_wchar_p
 actual_command = get_command()
 need(type(actual_command) is str and actual_command, 'D19 actual current process command')
 row = {'schema':'flylab.m3.s28.d19.current-stage.v1', 'stage':stage, 'boundary':boundary, 'ticks':time.perf_counter_ns(), 'frequency':1000000000,
   'parent_entry_s':token['entry_counter_s'], 'parent_pid':token['owner_pid'], 'parent_filetime':token['owner_creation_ticks'],
   'current_pid':os.getpid(), 'current_filetime':supervisor.creation_ticks(os.getpid()), 'root':os.fspath(Path(fixture.root).absolute()),
   'request':token['request'], 'actual_command':actual_command, 'native_argv':command, 'child':child}
 path = Path(fixture.root) / 'd19-current-timing.jsonl'
 raw = (__import__('json').dumps(row, ensure_ascii=True, allow_nan=False, separators=(',', ':')) + '\n').encode()
 need(len(raw) <= 4096, 'D19 bounded timing row')
 with path.open('ab') as stream: stream.write(raw)
 return row


def d19_spawn(job, argv, root, environment, token):
 from types import SimpleNamespace
 fixture = SimpleNamespace(root=root)
 d19_stage(fixture, 'native', 'BEFORE_SPAWN', list(argv))
 job.spawn(argv, root, environment)
 d19_stage(fixture, 'native', 'AFTER_SPAWN', list(argv), {'pid':job.pid, 'filetime':job.identities[str(job.pid)]})


def d19_body_verdict(report):
 # The production path still executes300*0.005=1.5 simulation seconds.
 # This metadata verdict never upgrades a technical run_valid result to gait PASS.
 return {'screen':'1.5_SIMULATION_SECOND_STARTUP_SCREEN', 'full_F3':'NOT_RUN', 'full_F4':'NOT_RUN', 'full_F5':'NOT_RUN',
    'goal':'ACTIVE_INCOMPLETE', 'biological_mechanism':'UNPROVEN', 'runtime_run_valid':report.get('run_valid')}


def d19_stage_fixture(fixture, source_root):
 from pathlib import Path
 c = d19_capsule()
 d19_stage(fixture, 'fixture', 'BEGIN')
 for pin in c['stage_pins']:
  raw = d19_raw(pin, source_root)
  target = Path(fixture.root) / pin['path']
  if target.exists(): need(target.read_bytes() == raw, 'D19 finite staged raw agreement')
  else: fixture_text_write(fixture.root, pin['path'], raw)
 d19_stage(fixture, "source", "BEGIN")
 from tools import banc_m3_checkpoint_harness_runtime as k
 bp=k.data(k.CURRENT_CONTRACT,source_root)['d19_r5_bridge'];b=d19_5b(k.__dict__,source_root)
 for p,cap in [(bp,262144)]+[(x['archive'],x['cap']) for x in b['inputs']]:
  v=raw_pin(p,source_root,cap,allow_empty=p['bytes']==0);target=Path(fixture.root)/p['path']
  if target.exists():need(target.read_bytes()==v,'Same r5 staged immutable raw')
  else:fixture_text_write(fixture.root,p['path'],v)
 bp=k.data(k.CURRENT_CONTRACT,source_root)['d19_r6_bridge'];b=d19_6b(k.__dict__,source_root)
 for p,cap in [(bp,262144)]+[(x['archive'],x['cap']) for x in b['inputs']]:
  v=raw_pin(p,source_root,cap,allow_empty=p['bytes']==0);target=Path(fixture.root)/p['path']
  if target.exists():need(target.read_bytes()==v,'Same r6 staged immutable raw')
  else:fixture_text_write(fixture.root,p['path'],v)
 bp=k.data(k.CURRENT_CONTRACT,source_root)['d19_r7_bridge'];b=d19_7b(k.__dict__,source_root)
 for p,cap in [(bp,262144)]+[(x['archive'],x['cap']) for x in b['inputs']]:
  v=raw_pin(p,source_root,cap,allow_empty=p['bytes']==0);target=Path(fixture.root)/p['path']
  if target.exists():need(target.read_bytes()==v,'Same r7 staged immutable raw')
  else:fixture_text_write(fixture.root,p['path'],v)
 bp=k.data(k.CURRENT_CONTRACT,source_root)['d19_r8_bridge'];b=d19_8b(k.__dict__,source_root)
 for p,cap in [(bp,262144)]+[(x['archive'],x['cap']) for x in b['inputs']]:
  v=raw_pin(p,source_root,cap,allow_empty=p['bytes']==0);target=Path(fixture.root)/p['path']
  if target.exists():need(target.read_bytes()==v,'Same r8 staged immutable raw')
  else:fixture_text_write(fixture.root,p['path'],v)
 ct=k.data(k.CURRENT_CONTRACT,source_root)
 cut=k.data(ct['d19_fixed_history'],source_root)
 rows=[(ct['d19_fixed_history'],262144,False),(cut['historical_inventory'],262144,False),(ct['d19_current_golden'],262144,False),(ct['d19_resource_decision'],262144,False),(ct['d19_writer_decision'],262144,False),(cut['independent_expected'],524288,False),(cut['authenticated_actual'],524288,False)]
 rows += [(row['archive'],row['cap'],row['opaque_stream']) for row in k.data(cut['historical_inventory'],source_root)]
 g=k.data(ct['d19_current_golden'],source_root);rows.append((g['native_historical_context']['event_archive'],262144,False))
 for p,cap,opaque in rows:
  v=raw_pin(p,source_root,cap,allow_empty=p['bytes']==0,opaque=opaque);target=Path(fixture.root)/p['path']
  if target.exists():need(target.read_bytes()==v,'Same certified current staged immutable raw')
  else:fixture_text_write(fixture.root,p['path'],v,opaque=opaque)
 d19_r1_stage_closure(fixture,source_root)


def d19_fixed4_controls(k,fixture):
 from tests.banc_m3_checkpoint_harness_fixture import d19_fixed4_controls
 return d19_fixed4_controls(k,fixture)


def d19_legacy_fixture_binding(fixture):
 # Legacy explicit account_pin keeps its old262144 cap. The current authenticated
 # host is appended only as a separate new pure-definition slice by the fixture.
 from tools import banc_m3_contact_learning_admission as admission
 current = read_pin(text_spec(BASE+'/s28-s1.json', admission.ROOT, 262144), admission.ROOT)
 d19_supplement({}, admission.ROOT)
 pin = current['sources'][BASE+'/parent-direct-science-account.ps1']
 raw_pin(pin, admission.ROOT, 393216)
 return pin


def d19_legacy_definitions(root):
 from tools import banc_m3_contact_learning_admission as admission
 source = read_pin(text_spec(BASE+'/s28-s1.json', admission.ROOT, 262144), admission.ROOT)
 pin = source['sources'][BASE+'/parent-direct-science-account.ps1']
 raw = raw_pin(pin, admission.ROOT, 393216).decode('utf8')
 begin, end = '# D19 definitions BEGIN', '# D19 definitions END'
 need(raw.count(begin) == raw.count(end) == 1 and raw.index(begin) < raw.index(end), 'D19 exact legacy separate pure definitions')
 return raw[raw.index(begin)+len(begin):raw.index(end)]


def d19_fixture_commit(k, fixture, seconds, exit_code, outer):
 import copy
 import hashlib
 import json

 from tools import banc_m3_startup_ray_supervisor as supervisor

 need(k.closure(outer) is True, 'D19 known fixture closure')
 event, _baseline, policy, grant = d19_recovery(k.__dict__, fixture.recovery, fixture.root)
 recovered = d19_recovered_snapshot(k.__dict__, fixture.recovery, fixture.root)
 trace = k.data(fixture.trace_spec, fixture.root)
 d19_current_exit(k.__dict__, trace, outer, fixture.request_spec, fixture.root)
 d19_whole(trace['full_return_s'], float(seconds))
 d19_whole(trace['exit_code'], exit_code)
 d19_whole((trace['first_final_ms']-trace['dispatch_ms'])/1000.0, float(seconds))
 cost = dict(schema='flylab.m3.direct-science.full-return.v1', cost_id=fixture.q['call_id'], mode='generated', request=fixture.request_spec,
    source_freeze=fixture.source, trace=fixture.trace_spec, outer=fixture.outer_spec, actual_outer_full_return_s=seconds,
    exit_code=exit_code, charged_once=True, cap_exceeded=seconds>fixture.q['limits']['outer_s'],
    **{key:outer[key] for key in ('all_owned_ended','reservation_released','worker_started','worker_creation_state','creation_attempted','resume_attempted','resume_outcome')})
 if outer['exit_code'] != trace['exit_code']:
  cost.update(native_supervisor_exit_code=outer['exit_code'], actual_collector_shell_exit_code=trace['exit_code'], shell_deadline_status_pair_observed=True)
 raw = json.dumps(cost, ensure_ascii=True, allow_nan=False).encode('utf8')
 cost_path = k.BASE+'/parent-'+fixture.q['call_id']+'-full-return.json'
 cost_pin = {'path':cost_path, 'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest()}
 debit = {'cost_id':fixture.q['call_id'], 'mode':'generated', 'actual_outer_full_return_s':seconds, 'actual_returned_exit_code':exit_code, 'descriptor':cost_pin}
 def fee(pin, row, source, root):
  if pin == cost_pin:
   d19_whole(row, debit)
   k.identity(fixture.q, fixture.request_spec, source, root)
   return fixture.q, cost, outer, True
  return k.fee_proof(pin, row, source, root)
 # Retained independently implemented fixture projector is generated as a new
 # unique local closure below; it never patches a production function or global cache.
 project = d19_fixture_projector(k, fixture, event, recovered, policy, grant, fee)
 import os
 import time
 token={'schema':'flylab.m3.direct-science.lock-owner.v1','call_id':fixture.q['call_id'],'request':fixture.request_spec,'owner_pid':os.getpid(),'owner_creation_ticks':supervisor.creation_ticks(os.getpid()),'entry_counter_s':time.perf_counter(),'job_name':'Local\\generated-synthetic-fixture-fee'}
 lock = SharedLock(fixture.root/SHARED_LOCK,token)
 try:
  lock.authenticate()
  before_raw = raw_pin(text_spec(fixture.ledger_path,fixture.root,524288),fixture.root,524288)
  ledger = d19_json(before_raw)
  state = project(ledger, fixture.root)
  need(state['phase'] == 'closed-pending', 'D19 actual fixture closed pending')
  expected = copy.deepcopy(ledger)
  need(fixture.q['call_id'] not in [d['cost_id'] for d in expected['debits']], 'D19 duplicate fee refused')
  expected['debits'].append(debit)
  expected['direct_science_calls'][fixture.q['call_id']]['state'] = 'charged'
  for name in ('direct_science_workers','direct_science_reservations','direct_science_pending_actuals'):
   expected[name].pop(fixture.q['call_id'], None)
  if exit_code != 0: expected['direct_science_blocking_holds'][fixture.q['call_id']] = k.FAILURE_HOLD
  expected.update(k.totals(expected, policy, grant, k.CAPS))
  if cost['cap_exceeded'] or expected['direct_science_generated_used_s'] > 87794 or expected['generated_shared_remaining_s'] < 0:
   k.accounting_hold(expected, 'generated', fixture.q['call_id'])
  project(expected, fixture.root)
  inverse = copy.deepcopy(expected)
  inverse['debits'].pop()
  for name in ('direct_science_calls','direct_science_workers','direct_science_reservations','direct_science_pending_actuals','direct_science_blocking_holds', *k.totals(ledger,policy,grant,k.CAPS)):
   inverse[name] = copy.deepcopy(ledger[name])
  d19_whole(inverse, ledger)
  need((fixture.root/fixture.ledger_path).read_bytes() == before_raw, 'D19 exact stale fixture before refused')
  if (fixture.root/cost_path).exists(): d19_whole(d19_json((fixture.root/cost_path).read_bytes()), cost, False)
  else: fixture_text_write(fixture.root,cost_path,raw)
  lock.authenticate()
  atomic_ledger_publish(fixture.root/fixture.ledger_path,before_raw,expected,lock,validate=lambda value:d19_whole(value,expected))
  fixture.ledger=k.data(text_spec(fixture.ledger_path,fixture.root,524288),fixture.root,524288)
  d19_whole(fixture.ledger,expected)
  fixture.cost_spec=cost_pin
  return debit
 except Exception:
  lock.uncertain=True
  raise
 finally:
  if not lock.uncertain: lock.release(True)

def d19_capsule():
 import json
 from pathlib import Path
 c=json.loads(raw_pin({'path': 'verification/banc-natural-gait-m3-20260919/s28-f19-sol/sole-R1/historical-capsule.json', 'bytes': 30148, 'sha256': '145c8ca6daf2819dbf3c9533b087752e36c8ee8d62495007f589bdb608e9c67a'},Path(__file__).resolve().parents[1],262144))
 pins=[]
 for p in c['proof_pins']+c['receipt_pins']+c['supplement_value']['ordered_inputs']+[c['supplement'],c['before'],c['r1_predecessor']]+c['r1_inputs']+[{'path': 'verification/banc-natural-gait-m3-20260919/s28-f19-sol/sole-R1/historical-capsule.json', 'bytes': 30148, 'sha256': '145c8ca6daf2819dbf3c9533b087752e36c8ee8d62495007f589bdb608e9c67a'}]:
  if p not in pins:pins.append(p)
 logical=[row['logical_current_before'] for row in c['archives']]
 c['finite_pins']=[p for p in pins if p not in logical]
 c['stage_pins']=c['finite_pins']+[row['immutable_archive'] for row in c['archives']]
 c['routes']=[]
 for p in c['stage_pins']:
  if any(row['pin']==p for row in c['routes']):continue
  cap=524288 if p['bytes']>393216 else 393216 if p['path'].endswith('parent-direct-science-account.ps1') and '/s28-f19-root/before/' in p['path'] else 357046 if p==c['actor'] else 262144
  c['routes'].append({'pin':p,'cap':cap})
 return c
def d19_validate(k, ledger, root):
 r=k['r']
 KEY=k['KEY']
 ALLOCATION_KEY=k['ALLOCATION_KEY']
 QUALIFICATION_KEY=k['QUALIFICATION_KEY']
 EPOCH=k['EPOCH']
 CAPS=k['CAPS']
 ALLOCATION_PATH=k['ALLOCATION_PATH']
 QUALIFICATION_PATH=k['QUALIFICATION_PATH']
 SEQUENCE=k['SEQUENCE']
 FAILURE_HOLD=k['FAILURE_HOLD']
 data=k['data']
 metadata_pin=k['metadata_pin']
 keys=k['keys']
 totals=k['totals']
 accounting_hold=k['accounting_hold']
 expected_ownership=k['expected_ownership']
 eligibility=k['eligibility']
 identity=k['identity']
 success=k['success']
 qualification=k['qualification']
 valid_body=k['valid_body']
 fee_proof=k['fee_proof']
 # Recovery authenticates the current immutable cut once for this action.
 """Return legal accounting state, permitting incurred negative remaining."""
 event, _paid, policy, grant = d19_recovery(k, ledger[KEY], root)
 n=len(_paid['debits']);baseline=copy.deepcopy(_paid);baseline[KEY]=copy.deepcopy(ledger[KEY]);baseline['direct_science_blocking_holds']={}
 if d19_current_contract(k,root)['d19_predecessor_allocation_present']:baseline.pop(ALLOCATION_KEY)
 for name,value in CAPS.items():baseline[CAP_FIELDS[name]]=value
 baseline.update(totals(baseline,policy,grant,CAPS))
 expected = copy.deepcopy(baseline)
 expected[KEY] = d19_current_pin(k,ledger[KEY],k['RECOVERY_PATH'])
 expected["direct_science_blocking_holds"] = {}
 source = event["source_freeze"]
 tail = ledger["debits"][n:]
 r.need(type(tail) is list and len(tail) <= len(SEQUENCE), "finite seven-call tail")
 caps = dict(CAPS)
 failed = False
 for index, debit in enumerate(tail):
  r.exact(debit["cost_id"], SEQUENCE[index], "next literal debit")
  r.need(not failed, "paid failure does not admit successors")
  q, cost, _outer, closed = fee_proof(debit["descriptor"], debit, source, root)
  r.need(closed, "UNCOMMITTED_OWNERSHIP_UNKNOWN: no fee/debit/totals without known closure")
  reserved = q['limits']['outer_s']
  request=cost['request'] if q['mode']=='generated' else cost['supervisor_request'];d19_current_pin(k,request,request['path'])
  paid={'cost_id':q['call_id'],'mode':q['mode'],'actual_outer_full_return_s':cost['actual_outer_full_return_s'],'actual_returned_exit_code':cost['exit_code'],'descriptor':d19_current_pin(k,debit['descriptor'],debit['descriptor']['path'])}
  d19_whole(debit,paid);expected['debits'].append(paid)
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
  for field_name, field_value in (("epoch", EPOCH), ("recovery", ledger[KEY]), ("source_freeze", source), ("old_caps", CAPS),
     ("new_caps", CAPS), ("delta_s", {"generated_global": 0, "generated_local": 0, "original_global": 0}),
     ("combined_global_s", CAPS["generated_global"]+CAPS["original_global"]), ("new_resource_grant_s", 0), ("refund_s", 0), ("bootstrap", tail[0]["descriptor"])):
   r.exact(allocation[field_name], field_value, "exact allocation " + field_name)
  before = data(allocation["before_ledger"], root, 524288)
  r.need(ALLOCATION_KEY not in before and len(before["debits"]) == n+1, "fresh closed bootstrap before allocation")
  state = d19_validate(k, before, root)
  eligibility(before, state)
  success(before["debits"][-1], source, root, "bootstrap", allocation["bootstrap_junit"])
  expected[ALLOCATION_KEY] = d19_current_pin(k,ledger[ALLOCATION_KEY],k['ALLOCATION_PATH'])
  caps = dict(CAPS)
  for cap_name, cap_value in caps.items():
   expected[r.CAP_FIELDS[cap_name]] = cap_value
 else:
  r.need(len(tail) <= 1, "qualification cannot use old caps")
 if QUALIFICATION_KEY in ledger:
  r.need(ALLOCATION_KEY in ledger and len(tail) >= 5, "separate four actual costs before registration")
  metadata_pin(ledger[QUALIFICATION_KEY], QUALIFICATION_PATH)
  qualification(data(ledger[QUALIFICATION_KEY], root), ledger, source, root)
  expected[QUALIFICATION_KEY] = d19_current_pin(k,ledger[QUALIFICATION_KEY],k['QUALIFICATION_PATH'])
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
  request=d19_current_pin(k,admitted['request'],admitted['request']['path'])
  wanted={'request':request,'state':'admitted','mode':q['mode'],'reserved_s':q['limits']['outer_s']};d19_whole(admitted,wanted)
  r.need(len(tail) != 1 or ALLOCATION_KEY in ledger, "no lint before allocation")
  r.need(len(tail) != 5 or QUALIFICATION_KEY in ledger, "no body before K qualification")
  if len(tail) == 6:
   valid_body(tail[5], source, data(q['science_request'], root), root)
  expected["direct_science_calls"][call] = wanted
  phase = expected_ownership(expected, q, admitted["request"], ledger, root)
 # A legitimate cumulative breach adds the exact responsible HOLD, never erases a debit.
 if tail:
  last = tail[-1]
  breached = ((last["mode"] == "generated" and (expected["direct_science_generated_used_s"] > caps["generated_local"]
     or expected["generated_shared_remaining_s"] < 0)) or
     (last["mode"] == "original" and (expected["original_remaining_s"] < 0
     or expected["original_stage_used_s"] > 6300 or expected["startup_ray_original_local_used_s"] > 6000)))
  if breached:
   accounting_hold(expected, last['mode'], last['cost_id'])
 d19_whole(ledger, expected)
 return {"caps": caps, "completed": len(tail), "phase": phase, "failed": failed, "source": source, "epoch": EPOCH}


def d19_source_fence(k, source, root, check=lambda: None):
 r=k['r']
 BASE=k['BASE']
 _KEY=k['KEY']
 _ALLOCATION_KEY=k['ALLOCATION_KEY']
 _QUALIFICATION_KEY=k['QUALIFICATION_KEY']
 EPOCH=k['EPOCH']
 _OLD_CAPS=k['OLD_CAPS']
 _CAPS=k['CAPS']
 _RECOVERY_PATH=k['RECOVERY_PATH']
 _ALLOCATION_PATH=k['ALLOCATION_PATH']
 SOURCE_PATH=k['SOURCE_PATH']
 REVIEW_PATH=k['REVIEW_PATH']
 CONTRACT_PATH=k['CONTRACT_PATH']
 BODY_PATH=k['BODY_PATH']
 _ROSTER_PATH=k['ROSTER_PATH']
 _QUALIFICATION_PATH=k['QUALIFICATION_PATH']
 _SEQUENCE=k['SEQUENCE']
 _QUAL_CALLS=k['QUAL_CALLS']
 _CALLS=k['CALLS']
 _FAILURE_HOLD=k['FAILURE_HOLD']
 REPLACEMENTS=k['REPLACEMENTS']
 ADDITIONS=k['ADDITIONS']
 _CURRENT_CONTRACT=k['CURRENT_CONTRACT']
 _HISTORY_BEFORE_SOURCE=k['HISTORY_BEFORE_SOURCE']
 _CONTINUATION_BEFORE_SOURCE=k['CONTINUATION_BEFORE_SOURCE']
 data=k['data']
 metadata_pin=k['metadata_pin']
 keys=k['keys']
 _integer=k['integer']
 _number=k['number']
 _totals=k['totals']
 _terminal_baseline=k['terminal_baseline']
 _terminal_bridge=k['terminal_bridge']
 _frozen_history=k['frozen_history']
 _accounting_hold=k['accounting_hold']
 _expected_ownership=k['expected_ownership']
 _eligibility=k['eligibility']
 _identity=k['identity']
 _success=k['success']
 _qualification=k['qualification']
 roster=k['roster']
 bounded_source=k['bounded_source']
 provenance=k['provenance']
 _body_binding=k['body_binding']
 _original_metadata=k['original_metadata']
 _valid_body=k['valid_body']
 _HISTORY_PROOF=k['_HISTORY_PROOF']
 """Fresh whole current Source: call only after actual SharedLock acquisition."""
 try:
  metadata_pin(source, SOURCE_PATH)
  check()
  freeze = data(source, root)
  r.exact(freeze["ROOT_review_cycle_complete"], True, "ROOT adopted K reviewed Source")
  r.exact(freeze["qualification_epoch"], EPOCH, "distinct K Source")
  metadata_pin(freeze["reviewed_source"], REVIEW_PATH)
  review = data(freeze["reviewed_source"], root)
  keys(review, ('schema', 'epoch', 'sources', 'replacements', 'additions', 'before_source', 'root_reflection',
     'astra_review', 'root_reviews', 'sol_reports'), 'complete K reviewed-source metadata')
  r.exact(review['schema'], 'flylab.m3.checkpoint-reviewed-source.v1', 'distinct K reviewed Source')
  r.exact(review['epoch'], EPOCH, 'K Source review epoch')
  r.exact(freeze["checkpoint_contract"],freeze["sources"][CONTRACT_PATH],"Source contract before provenance")
  r.exact(freeze["checkpoint_contract"],review["sources"][CONTRACT_PATH],"Source review contract agreement")
  provenance(review, root)
  d19_supplement(k, root)
  d19_r1_closure(source,root)
  d19_whole(review['before_source'], next(x['immutable_archive'] for x in d19_capsule()['archives'] if x['logical_current_before'] == d19_capsule()['source']))
  original = d19_read(d19_capsule()['source'], root)
  old = original['sources']
  expected_freeze = copy.deepcopy(original)
  expected_freeze.update(qualification_epoch=EPOCH, reviewed_source=freeze['reviewed_source'],
       body_retry=freeze['sources'][BODY_PATH], sources=freeze['sources'],
       checkpoint_contract=freeze['sources'][CONTRACT_PATH])
  r.exact(freeze, expected_freeze, 'whole inherited freeze with only finite K changes')
  r.need(len(old) == 871 and set(freeze["sources"]) == set(old) | set(ADDITIONS), "entire inherited871 plus zero flat additions")
  r.exact(review["sources"], freeze["sources"], "same reviewed vector")
  r.exact(review["replacements"], list(REPLACEMENTS), "all and only declared replacements")
  r.exact(review["additions"], list(ADDITIONS), "all and only declared new dependencies")
  for name, spec in old.items():
   if name not in REPLACEMENTS:
    r.exact(freeze["sources"][name], spec, "whole unchanged Source pin " + name)
  for name, spec in freeze["sources"].items():
   check()
   r.exact(spec['path'], name, 'same whole Source namespace')
   current_cap = {REPLACEMENTS[0]: 131072, REPLACEMENTS[1]: 131072,
                  BASE + '/parent-direct-science-account.ps1': 393216}.get(name)
   evidence = current_cap is None and name in old and old[name]['bytes'] > 262144
   cap = current_cap if current_cap is not None else 524288 if evidence else 262144
   r.need(spec['bytes'] <= cap, 'bounded Source/fixture or exact inherited evidence')
   if evidence:
    r.exact(spec, old[name], 'only fixed J inherited evidence may use524288')
    r.raw_pin(spec, root, cap)
   else:
    bounded_source(spec, root, cap)
  check()
  roster(freeze, root)
  body_binding_pin = freeze["sources"][BODY_PATH]
  r.exact(freeze["body_retry"], body_binding_pin, "static body in acyclic Source vector")
  return freeze

 except Exception:
  _HISTORY_PROOF.clear()
  raise


def d19_fixture_projector(k,fixture,event,baseline,policy,grant,fee):
 from tests.banc_m3_checkpoint_harness_fixture import d19_fixture_projector
 return d19_fixture_projector(k,fixture,event,baseline,policy,grant,fee)


def d19_r1_closure(source, root):
 c = d19_capsule()
 base = BASE+'/s28-f19-root/transition64/'
 def load(pin, path):
  need(type(pin) is dict and tuple(pin) == ('path','bytes','sha256'), 'D19 closure pin order')
  need(pin['path'] == path and type(pin['bytes']) is int and 0 < pin['bytes'] <= 262144, 'D19 finite closure path/cap')
  return d19_json(raw_pin(pin, root, 262144))
 path = base+'ADOPTION.json'
 spec = text_spec(path, root, 262144)
 v = load(spec, path)
 need(tuple(v) == ('schema','verdict','source_freeze','reviewed_source','predecessor','ordered_reports'), 'D19 closure exact keys/order')
 d19_whole([v['schema'],v['verdict'],v['source_freeze'],v['predecessor']], ['flylab.m3.s28.d19.root-adoption-closure.v1','ADOPT_STATIC_IMPLEMENTATION',source,c['r1_predecessor']])
 freeze = read_pin(source, root, 262144)
 d19_whole(v['reviewed_source'],freeze['reviewed_source'])
 reports = v['ordered_reports']
 need(type(reports) is list and len(reports) == 4, 'D19 four finite report associations')
 d19_whole(reports[:2],c['r1_report_predecessors'])
 for pin in reports[:2]: load(pin,pin['path'])
 sol = load(reports[2],BASE+'/s28-f19-sol/sole-R1/R1.json')
 residual = load(reports[3],base+'RESIDUAL.json')
 d19_whole(sol['root_first'],c['r1_report_predecessors'][1])
 d19_whole(sol['first_seal'],c['r1_first_seal'])
 d19_whole(residual['schema'],'flylab.m3.s28.d19.root-residual.v1')
 d19_whole(residual['verdict'],'ADOPT_STATIC_IMPLEMENTATION')
 d19_whole(residual['sol_r1'],reports[2])
 d19_whole(residual['input_r1_source'],sol['candidate_source'])
 d19_whole(residual['adopted_source'],source)
 d19_whole(residual['reviewed_source'],freeze['reviewed_source'])
 actual = residual['actual_static_authentication']
 need(type(actual) is dict and tuple(actual) == ('chunk_id','exit_code','kind') and type(actual['chunk_id']) is str and actual['chunk_id'] and type(actual['exit_code']) is int and actual['exit_code'] == 0 and actual['kind'] == 'OWN_STDLIB_METADATA', 'D19 actual ROOT closure authentication')
 return spec, reports


def d19_r1_transfer(k, paid, before_raw, event, event_pin, root):
 import copy
 ct=d19_current_contract(k,root);cut=k['data'](ct['d19_fixed_history'],root)
 need(type(before_raw) is bytes and before_raw==raw_pin(cut['authenticated_actual'],root,524288),'D19 exact fixed-history stale raw-before')
 d19_whole(paid,k['data'](cut['independent_expected'],root,524288))
 d19_whole(event,d19_event(k,event['source_freeze'],root));k['metadata_pin'](event_pin,k['RECOVERY_PATH']);d19_whole(k['data'](event_pin,root),event)
 need(len(paid['debits'])==cut['debits'] and len(paid['direct_science_calls'])==cut['paid_calls'],'D19 independent current counts')
 d19_whole(list(cut['failure_holds']),[event['released_hold']])
 for name in OWNER_KEYS:d19_whole(paid[name],cut['failure_holds'] if name=='direct_science_blocking_holds' else {})
 ap=ct['d19_predecessor_allocation_present']
 if ap:need(tuple(paid)[-1]==k['ALLOCATION_KEY'],'D19 last allocation key')
 result=copy.deepcopy(paid);result[k['KEY']]=copy.deepcopy(event_pin);result['direct_science_blocking_holds']={}
 if ap:result.pop(k['ALLOCATION_KEY'])
 for name,value in k['CAPS'].items():result[CAP_FIELDS[name]]=value
 policy=k['data'](POLICY_PIN,root);grant=k['data'](J_PIN,root);result.update(k['totals'](result,policy,grant,k['CAPS']))
 reverse=copy.deepcopy(result)
 for name in (k['KEY'],'direct_science_blocking_holds',*CAP_FIELDS.values(),*k['totals'](paid,policy,grant,ct['old_caps'])):reverse[name]=copy.deepcopy(paid[name])
 if ap:reverse[k['ALLOCATION_KEY']]=copy.deepcopy(paid[k['ALLOCATION_KEY']])
 d19_whole(reverse,paid)
 return result


def d19_r1_controls(k, fixture):
 import sys

 from tests.banc_m3_checkpoint_harness_fixture import d19_r1_controls as control
 return control(k,fixture,sys.modules[__name__])

def d19_r1_stage_closure(fixture, source_root):
 source = text_spec(BASE+'/s28-s1.json',source_root,262144)
 spec,reports = d19_r1_closure(source,source_root)
 for p in [spec,*reports]:
  raw = raw_pin(p,source_root,262144)
  target = fixture.root/p['path']
  if target.exists(): need(target.read_bytes() == raw,'D19 same staged closure raw')
  else: fixture_text_write(fixture.root,p['path'],raw)


def d19_r1_defs(root, account_pin=None, contact_pin=None):
 from tests.banc_m3_checkpoint_fixture import stage_legacy
 from tools import banc_m3_contact_learning_admission as admission
 stage_legacy(root)
 source = text_spec(BASE+'/s28-s1.json',admission.ROOT,262144)
 freeze = read_pin(source,admission.ROOT,262144)
 d19_r1_closure(source,admission.ROOT)
 current = freeze['sources'][BASE+'/parent-direct-science-account.ps1']
 need(current['path'] == BASE+'/parent-direct-science-account.ps1','D19 exact current definition host')
 host = raw_pin(current,admission.ROOT,393216).decode('utf8')
 begin,end = '# D19 definitions BEGIN','# D19 definitions END'
 need(host.count(begin) == host.count(end) == 1 and host.index(begin) < host.index(end),'D19 unique ordered definition-only slice')
 if account_pin is None: text = host
 else: text = raw_pin(account_pin,admission.ROOT,262144).decode('utf8')
 sentinel = '$ledger=Get-Content -LiteralPath (Safe $ledgerName)'
 need(text.count('$ErrorActionPreference=') >= 1 and sentinel in text,'D19 retained pure extraction boundary')
 definitions = text[text.index('$ErrorActionPreference='):text.index(sentinel)]
 if account_pin is not None: definitions += '\n'+host[host.index(begin)+len(begin):host.index(end)]
 anchor = "$taskRoot=[IO.Path]::GetFullPath('C:/Users/ckthd/Dev/Projects/fly_lab')"
 need(definitions.count(anchor) == 1,'D19 actual root anchor once')
 contact = text_spec(BASE+'/parent-contact-learning-contract-native.ps1',admission.ROOT,262144) if contact_pin is None else contact_pin
 contact_text = raw_pin(contact,admission.ROOT,262144).decode('utf8')
 return definitions.replace(anchor,"$taskRoot=[IO.Path]::GetFullPath('"+root.as_posix().replace("'","''")+"')")+'\n'+contact_text

def d19_root_bind_span(contract, host):
 import hashlib
 import re
 need(type(host) is bytes and 0 < len(host) <= 393216, 'D19 actual bounded account host')
 host.decode('utf8', errors='strict')
 selector = contract['d19_bind']
 need(type(selector) is list and len(selector) == 4, 'D19 exact four-item selector')
 path, start, end, sha = selector
 need(type(path) is str and path == BASE+'/parent-direct-science-account.ps1', 'D19 one literal definition host')
 need(type(start) is int and type(end) is int and 0 <= start <= end < len(host), 'D19 nonbool inclusive bounds')
 need(type(sha) is str and re.fullmatch('[0-9A-F]{64}', sha) is not None, 'D19 uppercase exact span SHA')
 begin, finish = b'# D19 definitions BEGIN', b'# D19 definitions END'
 need(host.count(begin) == host.count(finish) == 1, 'D19 unique definition markers')
 need(start == host.index(begin)+len(begin) and end+1 == host.index(finish), 'D19 exact marker-bound span')
 span = host[start:end+1]
 need(hashlib.sha256(span).hexdigest().upper() == sha, 'D19 whole exact definition bytes')
 return span


def d19_root_binding_controls(k,fixture):
 import sys

 from tests.banc_m3_checkpoint_harness_fixture import (
     d19_root_binding_controls as control,
 )
 return control(k,fixture,sys.modules[__name__])

def d19_5b(k,root):
 return d19_json(raw_pin(k['data'](k['CURRENT_CONTRACT'],root)['d19_r5_bridge'],root,262144))

def d19_5raw(k,p,root):
 b=d19_5b(k,root);rows=[x for x in b['inputs'] if x['logical']==p]
 need(len(rows)==1,'D19 exact finite r5 operand');x=rows[0];d19_whole(p,x['logical'],False)
 return raw_pin(x['archive'],root,x['cap'],allow_empty=p['bytes']==0)

def d19_5r(k,p,root):
 return d19_json(d19_5raw(k,p,root))

def d19_paid324(k,root):
 import copy
 b=d19_5b(k,root);rd=lambda p:d19_5r(k,p,root)
 for x in b['inputs']:d19_5raw(k,x['logical'],root)
 old,policy,grant=d19_paid323(k,root);e=copy.deepcopy(old);event=rd(b['old_event'])
 d19_whole(event,b['old_event_value']);need(event['new_resource_grant_s']==0 and event['combined_global_s']==19900,'Old conserved500 immutable')
 d19_whole(event['before_ledger'],d19_capsule()['terminal']);d19_whole(event['delta_s'],{'generated_global':500,'generated_local':500,'original_global':-500})
 e['direct_science_blocking_holds']={};e[k['KEY']]=b['old_event']
 for n,v in b['old_caps'].items():e[CAP_FIELDS[n]]=v
 e.update(k['totals'](e,policy,grant,b['old_caps']));d19_whole(rd(b['recovered']),e)
 e['direct_science_calls'][b['call']]={'reserved_s':540,'mode':'generated','request':b['request'],'state':'admitted'}
 e['direct_science_reservations'][b['call']]={'reserved_s':540,'request':b['request']};d19_whole(rd(b['admitted']),e)
 e['direct_science_reservations']={};e['direct_science_pending_actuals']={b['call']:{'ownership_closed':True,'request':b['request'],'state':'actual_pending'}}
 before=rd(b['before']);d19_whole(before,d19_owner_wire(e));e=copy.deepcopy(before)
 t,o,cost=rd(b['trace']),rd(b['outer']),rd(b['cost']);d19_whole(t,b['trace_value']);d19_whole(o,b['outer_value']);d19_whole(cost,b['cost_value'])
 need(t['exit_code']==o['exit_code']==o['worker_exit_code']==1 and o['timeout'] is False and o['first_error'] is None and k['closure'](o) is True,'Actual r5 failed RESUMED/closure')
 d19_whole(t['full_return_s'],(t['first_final_ms']-t['dispatch_ms'])/1000.0);d19_whole(t['full_return_s'],63.816)
 e['debits'].append({'cost_id':b['call'],'mode':'generated','actual_outer_full_return_s':63.816,'actual_returned_exit_code':1,'descriptor':b['cost']})
 e['direct_science_calls'][b['call']]['state']='charged';e['direct_science_pending_actuals']={};e['direct_science_blocking_holds']={b['call']:'L stage unsuccessful; successor blocked'}
 totals=k['totals'](e,policy,grant,b['old_caps']);e.update(totals);paid=rd(b['paid']);d19_whole(paid,e)
 z=copy.deepcopy(paid);z['debits'].pop();z['direct_science_calls'][b['call']]['state']='admitted'
 for n in ('direct_science_pending_actuals','direct_science_blocking_holds',*totals):z[n]=copy.deepcopy(before[n])
 d19_whole(z,before);need(len(paid['debits'])==324 and len(paid['direct_science_calls'])==206,'Actual once paid324')
 return paid,policy,grant

def d19_owner_wire(token):
 return d19_json(__import__('json').dumps(token,sort_keys=True,allow_nan=False).encode('utf8'))

def d19_owner_controls():
 import copy
 import math
 import os

 from tools import banc_m3_contact_learning_admission as a
 t=d19_json(os.environ['FLYLAB_DIRECT_SCIENCE_OWNER'].encode());real=read_pin(text_spec(SHARED_LOCK,a.ROOT,16384),a.ROOT,16384);d19_whole(real,d19_owner_wire(t))
 for key,value in [('schema',None),('owner_pid',float(t['owner_pid'])),('entry_counter_s',math.nextafter(t['entry_counter_s'],math.inf)),('owner_pid',True),('owner_creation_ticks',0),('entry_counter_s',1),('job_name','forged'),('request',{}),('extra',0)]:
  x=copy.deepcopy(t);x[key]=value
  refused=False
  try:d19_whole(real,d19_owner_wire(x))
  except ValueError:refused=True
  need(refused,'D19 owner control accepted '+key)
 x=copy.deepcopy(t);x.pop('owner_pid')
 refused=False
 try:d19_whole(real,d19_owner_wire(x))
 except ValueError:refused=True
 need(refused,'D19 missing owner key accepted')
 for x in (dict(reversed(list(real.items()))),dict(real,request=dict(reversed(list(real['request'].items()))))):
  refused=False
  try:d19_whole(x,d19_owner_wire(t))
  except ValueError:refused=True
  need(refused,'D19 durable token order accepted')

def d19_6b(k,root):
 return d19_json(raw_pin(k['data'](k['CURRENT_CONTRACT'],root)['d19_r6_bridge'],root,262144))

def d19_6raw(k,p,root):
 b=d19_6b(k,root);rows=[x for x in b['inputs'] if x['logical']==p]
 need(len(rows)==1,'D19 exact finite r6 operand');x=rows[0];d19_whole(p,x['logical'],False)
 return raw_pin(x['archive'],root,x['cap'],allow_empty=p['bytes']==0)

def d19_6r(k,p,root):
 return d19_json(d19_6raw(k,p,root))

def d19_paid325(k,root):
 import copy
 b=d19_6b(k,root);rd=lambda p:d19_6r(k,p,root)
 for x in b['inputs']:d19_6raw(k,x['logical'],root)
 old,policy,grant=d19_paid324(k,root);e=copy.deepcopy(old);event=rd(b['old_event'])
 d19_whole(event,b['old_event_value']);need(event['new_resource_grant_s']==600 and event['combined_global_s']==20500,'Actual generated600 retained')
 d19_whole(event['before_ledger'],d19_5b(k,root)['paid']);d19_whole(event['delta_s'],{'generated_global':600,'generated_local':600,'original_global':0})
 e['direct_science_blocking_holds']={};e[k['KEY']]=b['old_event']
 for n,v in b['old_caps'].items():e[CAP_FIELDS[n]]=v
 e.update(k['totals'](e,policy,grant,b['old_caps']));d19_whole(rd(b['recovered']),e)
 e['direct_science_calls'][b['call']]={'state':'admitted','request':b['request'],'reserved_s':540,'mode':'generated'}
 e['direct_science_reservations'][b['call']]={'reserved_s':540,'request':b['request']};d19_whole(rd(b['admitted']),e)
 e['direct_science_reservations']={};e['direct_science_pending_actuals']={b['call']:{'ownership_closed':True,'request':b['request'],'state':'actual_pending'}}
 before=rd(b['before']);d19_whole(before,d19_owner_wire(e));e=copy.deepcopy(before)
 t,o,cost=rd(b['trace']),rd(b['outer']),rd(b['cost']);d19_whole(t,b['trace_value']);d19_whole(o,b['outer_value']);d19_whole(cost,b['cost_value'])
 need(t['exit_code']==o['exit_code']==o['worker_exit_code']==1 and o['timeout'] is False and o['first_error'] is None and k['closure'](o) is True,'Actual r6 failed RESUMED/closure')
 d19_whole(t['exec_args'],{'cmd':'& .venv/Scripts/python.exe -m tools.banc_m3_startup_ray_supervisor --request '+b['request']['path'],'workdir':b['workdir']})
 d19_whole(t['request'],b['request']);d19_whole(t['source_freeze'],b['source']);d19_whole(rd(b['request'])['source_freeze'],b['source'])
 need(o['owner_pid']>0 and o['owner_creation_ticks']>0 and o['worker_pid']>0 and o['worker_creation_ticks']>0 and o['worker_started'] is True and o['job_created'] is True and o['worker_creation_state']=='RESUMED','Actual r6 positive process identities')
 d19_whole(t['full_return_s'],(t['first_final_ms']-t['dispatch_ms'])/1000.0);d19_whole(t['full_return_s'],352.438)
 e['debits'].append({'cost_id':b['call'],'mode':'generated','actual_outer_full_return_s':352.438,'actual_returned_exit_code':1,'descriptor':b['cost']})
 e['direct_science_calls'][b['call']]['state']='charged';e['direct_science_pending_actuals']={};e['direct_science_blocking_holds']={b['call']:'L stage unsuccessful; successor blocked'}
 totals=k['totals'](e,policy,grant,b['old_caps']);e.update(totals);paid=rd(b['paid']);d19_whole(paid,e)
 z=copy.deepcopy(paid);z['debits'].pop();z['direct_science_calls'][b['call']]['state']='admitted'
 for n in ('direct_science_pending_actuals','direct_science_blocking_holds',*totals):z[n]=copy.deepcopy(before[n])
 d19_whole(z,before);need(len(paid['debits'])==325 and len(paid['direct_science_calls'])==207,'Actual once paid325')
 return paid,policy,grant

def d19_7b(k,root):
 return d19_json(raw_pin(k['data'](k['CURRENT_CONTRACT'],root)['d19_r7_bridge'],root,262144))

def d19_7raw(k,p,root):
 b=d19_7b(k,root);rows=[x for x in b['inputs'] if x['logical']==p]
 need(len(rows)==1,'D19 exact finite r6 operand');x=rows[0];d19_whole(p,x['logical'],False)
 return raw_pin(x['archive'],root,x['cap'],allow_empty=p['bytes']==0)

def d19_7r(k,p,root):
 return d19_json(d19_7raw(k,p,root))

def d19_paid326(k,root):
 import copy
 b=d19_7b(k,root);rd=lambda p:d19_7r(k,p,root)
 for x in b['inputs']:d19_7raw(k,x['logical'],root)
 old,policy,grant=d19_paid325(k,root);e=copy.deepcopy(old);event=rd(b['old_event'])
 d19_whole(event,b['old_event_value']);need(event['new_resource_grant_s']==0 and event['combined_global_s']==20500,'Actual generated600 retained')
 d19_whole(event['before_ledger'],d19_6b(k,root)['paid']);d19_whole(event['delta_s'],{'generated_global':0,'generated_local':0,'original_global':0})
 e['direct_science_blocking_holds']={};e[k['KEY']]=b['old_event']
 for n,v in b['old_caps'].items():e[CAP_FIELDS[n]]=v
 e.update(k['totals'](e,policy,grant,b['old_caps']));d19_whole(rd(b['recovered']),e)
 e['direct_science_calls'][b['call']]={'request':b['request'],'mode':'generated','reserved_s':540,'state':'admitted'}
 e['direct_science_reservations'][b['call']]={'reserved_s':540,'request':b['request']};d19_whole(rd(b['admitted']),e)
 e['direct_science_reservations']={};e['direct_science_pending_actuals']={b['call']:{'ownership_closed':True,'request':b['request'],'state':'actual_pending'}}
 before=rd(b['before']);d19_whole(before,d19_owner_wire(e));e=copy.deepcopy(before)
 t,o,cost=rd(b['trace']),rd(b['outer']),rd(b['cost']);d19_whole(t,b['trace_value']);d19_whole(o,b['outer_value']);d19_whole(cost,b['cost_value'])
 need(t['exit_code']==o['exit_code']==o['worker_exit_code']==1 and o['timeout'] is False and o['first_error'] is None and k['closure'](o) is True,'Actual r6 failed RESUMED/closure')
 d19_whole(t['exec_args'],{'cmd':'& .venv/Scripts/python.exe -m tools.banc_m3_startup_ray_supervisor --request '+b['request']['path'],'workdir':b['workdir']})
 d19_whole(t['request'],b['request']);d19_whole(t['source_freeze'],b['source']);d19_whole(rd(b['request'])['source_freeze'],b['source'])
 need(o['owner_pid']>0 and o['owner_creation_ticks']>0 and o['worker_pid']>0 and o['worker_creation_ticks']>0 and o['worker_started'] is True and o['job_created'] is True and o['worker_creation_state']=='RESUMED','Actual r6 positive process identities')
 d19_whole(t['full_return_s'],(t['first_final_ms']-t['dispatch_ms'])/1000.0);d19_whole(t['full_return_s'],525.62)
 e['debits'].append({'cost_id':b['call'],'mode':'generated','actual_outer_full_return_s':525.62,'actual_returned_exit_code':1,'descriptor':b['cost']})
 e['direct_science_calls'][b['call']]['state']='charged';e['direct_science_pending_actuals']={};e['direct_science_blocking_holds']={b['call']:'L stage unsuccessful; successor blocked'}
 totals=k['totals'](e,policy,grant,b['old_caps']);e.update(totals);paid=rd(b['paid']);d19_whole(paid,e)
 z=copy.deepcopy(paid);z['debits'].pop();z['direct_science_calls'][b['call']]['state']='admitted'
 for n in ('direct_science_pending_actuals','direct_science_blocking_holds',*totals):z[n]=copy.deepcopy(before[n])
 d19_whole(z,before);need(len(paid['debits'])==326 and len(paid['direct_science_calls'])==208,'Actual once paid326')
 return paid,policy,grant

def d19_8b(k,root):
 return d19_json(raw_pin(k['data'](k['CURRENT_CONTRACT'],root)['d19_r8_bridge'],root,262144))

def d19_8raw(k,p,root):
 rows=[x for x in d19_8b(k,root)['inputs'] if x['logical']==p];need(len(rows)==1,'Exact r8 operand');x=rows[0];d19_whole(p,x['logical'],False)
 return raw_pin(x['archive'],root,x['cap'],allow_empty=p['bytes']==0)

def d19_8r(k,p,root):
 return d19_json(d19_8raw(k,p,root))

def d19_paid327(k,root):
 import copy
 b=d19_8b(k,root);rd=lambda p:d19_8r(k,p,root)
 for x in b['inputs']:d19_8raw(k,x['logical'],root)
 old,policy,grant=d19_paid326(k,root);e=copy.deepcopy(old);event=rd(b['old_event'])
 d19_whole(event,b['old_event_value']);d19_whole(event['before_ledger'],d19_7b(k,root)['paid']);d19_whole(event['delta_s'],{'generated_global':2400,'generated_local':2400,'original_global':0});need(event['new_resource_grant_s']==2400 and event['combined_global_s']==22900,'Actual2400 retained')
 e['direct_science_blocking_holds']={};e[k['KEY']]=b['old_event']
 for n,v in b['old_caps'].items():e[CAP_FIELDS[n]]=v
 e.update(k['totals'](e,policy,grant,b['old_caps']));d19_whole(rd(b['recovered']),e)
 e['direct_science_calls'][b['call']]={'mode':'generated','reserved_s':1200,'request':b['request'],'state':'admitted'};e['direct_science_reservations'][b['call']]={'reserved_s':1200,'request':b['request']};d19_whole(rd(b['admitted']),e)
 before=rd(b['before']);d19_whole(before,e);need(d19_8raw(k,b['before'],root)==d19_8raw(k,b['admitted'],root),'No-create exact raw admission')
 t,o,cost=rd(b['trace']),rd(b['outer']),rd(b['cost'])
 for a,z in [(t,b['trace_value']),(o,b['outer_value']),(cost,b['cost_value'])]:d19_whole(a,z)
 need(t['exit_code']==o['exit_code']==1 and t['first_error'] is None and t['session_id'] is None and o['timeout'] is False and o['pre_spawn_refused'] is True and k['closure'](o) is True,'Actual no-create closure')
 d19_whole(o['first_error'],{'message':'Actual once paid326','type':'ValueError'});need(o['refusal_origin']=='K_INITIAL_GUARD_BEFORE_CREATE' and o['owner_pid']>0 and o['owner_creation_ticks']>0,'Actual initial guard')
 d19_whole(t['exec_args'],{'cmd':'& .venv/Scripts/python.exe -m tools.banc_m3_startup_ray_supervisor --request '+b['request']['path'],'workdir':b['workdir']})
 d19_whole(t['request'],b['request'])
 d19_whole(o['request'],{'bytes':b['request']['bytes'],'path':b['request']['path'],'sha256':b['request']['sha256']})
 for a in [t['source_freeze'],rd(b['request'])['source_freeze']]:d19_whole(a,b['source'])
 need(rd(b['request'])['call_id']==b['call'],'Actual r8 role');d19_whole(t['full_return_s'],(t['first_final_ms']-t['dispatch_ms'])/1000.0);d19_whole(t['full_return_s'],1.989)
 e['debits'].append({'cost_id':b['call'],'mode':'generated','actual_outer_full_return_s':1.989,'actual_returned_exit_code':1,'descriptor':b['cost']});e['direct_science_calls'][b['call']]['state']='charged';e['direct_science_reservations']={};e['direct_science_blocking_holds']={b['call']:'L stage unsuccessful; successor blocked'}
 totals=k['totals'](e,policy,grant,b['old_caps']);e.update(totals);paid=rd(b['paid']);d19_whole(paid,e)
 z=copy.deepcopy(paid);z['debits'].pop();z['direct_science_calls'][b['call']]['state']='admitted'
 for n in ('direct_science_reservations','direct_science_blocking_holds',*totals):z[n]=copy.deepcopy(before[n])
 d19_whole(z,before);need(len(paid['debits'])==327 and len(paid['direct_science_calls'])==209,'Actual once paid327')
 return paid,policy,grant

def d19_expected_ownership(k,expected,q,request_spec,live,root,*,charged=False):
 r=k['r'];keys=k['keys'];integer=k['integer'];data=k['data'];metadata_pin=k['metadata_pin'];number=k['number']
 """Exactly admitted/begun/closed-pending/unknown layouts; no broad tail keys."""
 call, mode = q["call_id"], q["mode"]
 prefix = "direct_science_" if mode == "generated" else "startup_ray_original_"
 w, res, pending, holds = (prefix + x for x in ("workers", "reservations", "pending_actuals", "blocking_holds"))
 worker, reservation, actual, hold = (live[k] for k in (w, res, pending, holds))
 if not worker and not actual and not hold and not charged:
  if mode == "generated":
   expected[res] = {call: {"request": request_spec, "reserved_s": q["limits"]["outer_s"]}}
  return "admitted"
 if not worker and not reservation and not hold:
  expected[pending] = {} if charged else {call: {"request": request_spec, "state": "actual_pending", "ownership_closed": True}}
  return "charged" if charged else "closed-pending"
 keys(worker, (call,), "sole actual owner")
 value = worker[call]
 id_keys = ("owner_pid", "owner_creation_ticks", "worker_pid", "worker_creation_ticks")
 allowed = ("request", "state", "lock_token", *id_keys) + (("grant", "worker_max", "reserved_s") if mode == "original" else ())
 keys(value, allowed, "whole typed owner")
 for field in id_keys:
  integer(value[field], "owner " + field, positive=True)
 r.need(value["owner_pid"] != value["worker_pid"], "distinct owned worker")
 metadata_pin(value["lock_token"], r.SHARED_LOCK)
 r.d19_whole(value["lock_token"],r.text_spec(r.SHARED_LOCK,root,16384))
 token = data(value["lock_token"], root, 16384)
 r.exact(token['schema'], 'flylab.m3.direct-science.lock-owner.v1', 'owned token schema')
 r.exact(token['request'], request_spec, 'owned request')
 r.exact(token['call_id'], call, 'same owned literal call')
 for key in ('owner_pid', 'owner_creation_ticks'):
  r.exact(token[key], value[key], 'same owned process identity')
 r.need(type(token['entry_counter_s']) is float and number(token['entry_counter_s'], 'same ENTRY') > 0,
  'positive finite original ENTRY')
 wanted = {"request": request_spec if mode == "generated" else q["science_request"], "state": "begun",
   "lock_token": value["lock_token"], **{k: value[k] for k in id_keys}}
 wanted_res = {"request": request_spec, "reserved_s": q["limits"]["outer_s"]}
 if mode == "original":
  d19_current_pin(k,q['science_request'],q['science_request']['path']);d19_current_pin(k,q['grant'],q['grant']['path'])
  wanted={"request":q["science_request"],"grant":q["grant"],"state":"begun","worker_max":1,"reserved_s":q["limits"]["outer_s"],"lock_token":value["lock_token"],**{k:value[k] for k in id_keys}}
  wanted_res = {"request": q["science_request"], "grant": q["grant"], "reserved_s": q["limits"]["outer_s"]}
  expected["original_jobs_running"] = 1
  expected["original_reserved_s"] = q["limits"]["outer_s"]
 expected[w], expected[res] = {call: wanted}, {call: wanted_res}
 if hold:
  expected[holds] = {call: {"request": request_spec, "state": "ownership_unknown", **{k: value[k] for k in id_keys}}}
  return "unknown"
 r.need(not charged, "charged unresolved owner requires unknown HOLD")
 return "begun"



def d19_opaque_refuse(call,kind,message):
 try:call()
 except kind as error:assert str(error)==message
 else:raise AssertionError(message)

def d19_opaque_write(k,fixture,p,saved,cap):
 for flag in (None,0,'true',[]):d19_opaque_refuse(lambda flag=flag:k.r.fixture_text_write(fixture.root,'unrelated.bin',b'x',opaque=flag),ValueError,'literal opaque stream option')
 r=k.r;target=k.BASE+'/generated-opaque-write/'+p['path'].rsplit('/',1)[1];made=r.fixture_text_write(fixture.root,target,saved,opaque=True);assert made==dict(p,path=target) and r.raw_pin(made,fixture.root,cap,allow_empty=not saved,opaque=True)==saved


D19_OPAQUE_BODY = r'''
$defs=$utf8.GetString([Convert]::FromBase64String('DEFS'));$origRaw=(Get-Item Function:K-Raw).ScriptBlock;$origSource=(Get-Item Function:K-SourceRaw).ScriptBlock
$oc={param($call,$label)$e=$null;try{$null=& $call}catch{$e=$_.Exception.Message};Same $e $label 'Exact opaque failure'}
$cut=K-Read (K-Read $script:KCurrentContract).d19_fixed_history;$all=@(K-Read $cut.historical_inventory);foreach($x in $all){Need ($x.opaque_stream -is [bool]) 'Opaque flag'};$rows=@($all|Where-Object opaque_stream);Need ($rows.Length -gt 0) 'Certified streams';$strict=[Text.UTF8Encoding]::new($false,$true)
try{foreach($mode in 0,1){if($mode){iex $defs};foreach($r in $rows){$p=$r.archive;$a=Safe $p.path;$b=[IO.File]::ReadAllBytes($a);$empty=$p.bytes -eq 0;$positive={param($o=$true)Same ([Convert]::ToBase64String((K-Raw $p $r.cap $empty $o))) ([Convert]::ToBase64String($b)) 'Whole opaque bytes'};& $positive
foreach($bad in @($null,0,'true')){& $oc {K-Raw $p $r.cap $empty $bad} 'Opaque stream'};& $oc {K-Raw $p $r.cap $empty @()} 'Opaque stream'
$ok=$true;try{$null=$strict.GetString($b)}catch{Need ($_.Exception.GetBaseException() -is [Text.DecoderFallbackException]) 'Expected UTF8';$ok=$false};if($ok){& $positive $false}else{$e=$null;try{$null=K-Raw $p $r.cap '''
