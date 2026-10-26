"""Generated-only E182 recovery checks; no original stores or native model runs."""

import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from tools import optimize_c_banc_m3_native_timing as e181
from tools import recover_c_banc_m3_native_timing as recovery


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_reviewed_raw(root, arm):
    result_sha = write(root / arm / "result.json", {"validity": "valid_3s_diagnostic"})
    artifacts = {"result.json": result_sha}
    for name in (
        *recovery.REVIEWED_RAW,
        "start/array_00000.npy",
        "final/array_00000.npy",
    ):
        if name != "result.json":
            artifacts[name] = write(root / arm / name, {"generated": name})
    return result_sha, artifacts


def write_parent_reviews(root, *, include_plus=True):
    baseline_sha, baseline_artifacts = write_reviewed_raw(root, "baseline")
    write(root / "baseline-driver.json", {"error": None, "objective_completed": True})
    comparison_sha = write(
        root / "baseline-comparison.json", {"exact_reproduction": True}
    )
    write(
        root / "parent-baseline-review.json",
        {
            "schema": recovery.BASELINE_REVIEW_SCHEMA,
            "freeze_sha256": "a" * 64,
            "baseline_result_sha256": baseline_sha,
            "baseline_comparison_sha256": comparison_sha,
            "baseline_artifact_sha256": baseline_artifacts,
            "accepts_baseline": True,
            "permits_candidates": True,
            "blocking_defects": [],
        },
    )
    if not include_plus:
        return baseline_artifacts, {}
    plus_sha, plus_artifacts = write_reviewed_raw(root, "plus")
    write(root / "plus-driver.json", {"error": None, "objective_completed": True})
    write(
        root / "parent-plus-review.json",
        {
            "schema": recovery.PLUS_REVIEW_SCHEMA,
            "freeze_sha256": "a" * 64,
            "plus_result_sha256": plus_sha,
            "plus_artifact_sha256": plus_artifacts,
            "accepts_minus": True,
            "blocking_defects": [],
        },
    )
    return baseline_artifacts, plus_artifacts


def test_gate_rejects_unreviewed_freeze_before_outputs(tmp_path):
    freeze = tmp_path / "freeze.json"
    sha = write(freeze, {})
    args = SimpleNamespace(
        freeze=freeze,
        freeze_sha256=sha,
        source_review=tmp_path / "missing-review.json",
        source_review_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="E182 fixed recovery"):
        recovery._source_gate(args)
    assert not (tmp_path / "e182-native-timing-recovery").exists()


def test_capacity_passes_string_path_to_windows_disk_api(monkeypatch, tmp_path):
    def disk_usage(path):
        assert isinstance(path, str)
        assert path == str(tmp_path)
        return SimpleNamespace(free=30 << 30)

    monkeypatch.setattr(recovery.psutil, "disk_usage", disk_usage)
    now = recovery.time.perf_counter()
    monkeypatch.setattr(recovery, "_ENTRY_STARTED", now)
    recovery._capacity(tmp_path / "recovery", "baseline", now, "a" * 64)


@pytest.mark.parametrize("change", ["theta", "source_review_sha256", "arm"])
def test_child_rejects_tampered_request_before_model_load(
    monkeypatch, tmp_path, change
):
    root = tmp_path / "recovery"
    request_path = root / "baseline-request.json"
    request = {
        "schema": "flylab.m3.e182.child-request.v1",
        "arm": "baseline",
        "theta": recovery.THETA["baseline"],
        "freeze_sha256": "a" * 64,
        "source_review_sha256": "b" * 64,
    }
    request[change] = {
        "theta": [0.125] * 6,
        "source_review_sha256": "c" * 64,
        "arm": "plus",
    }[change]
    request_sha = write(request_path, request)
    args = SimpleNamespace(
        arm="baseline",
        child_request=request_path,
        child_request_sha256=request_sha,
        freeze_sha256="a" * 64,
        source_review_sha256="b" * 64,
    )
    monkeypatch.setattr(recovery, "_rel", lambda _name: root)
    with pytest.raises(ValueError, match="Exact live E182 driver"):
        recovery._child(args, {}, None)
    assert not (root / "baseline").exists()


@pytest.mark.parametrize("arm", ["plus", "minus"])
def test_permitted_candidate_child_reaches_generated_model_load_sentinel(tmp_path, arm):
    root = tmp_path / "recovery"
    root.mkdir()
    write_parent_reviews(root, include_plus=arm == "minus")
    preflight_name = f"{arm}-preflight-000.json"
    preflight_sha = write(
        root / preflight_name,
        {
            "schema": "flylab.m3.e182.preflight.v1",
            "arm": arm,
            "freeze_sha256": "a" * 64,
            "source_review_sha256": "b" * 64,
            "native_permitted": True,
            "observations": [{"ready": True}, {"ready": True}],
            "elapsed_wall_s": 0.1,
            "error": None,
        },
    )
    write(
        root / f"{arm}-request.json",
        {
            "schema": "flylab.m3.e182.child-request.v1",
            "arm": arm,
            "theta": recovery.THETA[arm],
            "parent_pid": psutil.Process().pid,
            "parent_created_s": psutil.Process().create_time(),
            "freeze_sha256": "a" * 64,
            "source_review_sha256": "b" * 64,
            "e181_freeze_sha256": "d" * 64,
            "preflight_name": preflight_name,
            "preflight_sha256": preflight_sha,
        },
    )
    code = """
import sys
from pathlib import Path
from types import SimpleNamespace
from tools import recover_c_banc_m3_native_timing as r
root, arm = Path(sys.argv[1]), sys.argv[2]
r._rel = lambda _name: root
r.e181._load_startup = lambda _freeze: {}
def model_load_sentinel(_path):
    raise RuntimeError('MODEL_LOAD_SENTINEL')
r.e181.GraphStore.load = model_load_sentinel
request = root / f'{arm}-request.json'
args = SimpleNamespace(arm=arm, child_request=request,
    child_request_sha256=r.e181._sha256(request), freeze_sha256='a'*64,
    source_review_sha256='b'*64)
gate = SimpleNamespace(freeze_sha256='d'*64, freeze={'graph_path':'synthetic'})
try:
    r._child(args, {}, gate)
except RuntimeError as error:
    print(str(error))
"""
    completed = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", code, str(root), arm],
        cwd=Path(recovery.__file__).parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "MODEL_LOAD_SENTINEL"
    assert not (root / arm).exists()
    with pytest.raises(FileExistsError, match="already permitted"):
        recovery._driver_attempt_gate(root, arm, "a" * 64)


def test_readiness_uses_two_observations_and_fails_closed(monkeypatch):
    queue = iter([{"running": 0, "pending": 0}, {"running": 1, "pending": 0}])
    delays = []
    monkeypatch.setattr(recovery, "_queue_counts", lambda: next(queue))
    monkeypatch.setattr(
        recovery, "_client_status", lambda: {"16248": "absent", "35496": "reused"}
    )
    monkeypatch.setattr(recovery.time, "sleep", delays.append)
    observations = recovery._readiness()
    assert delays == [2]
    assert [row["ready"] for row in observations] == [True, False]
    monkeypatch.setattr(
        recovery, "_queue_counts", lambda: (_ for _ in ()).throw(OSError("offline"))
    )
    assert recovery._observation()["ready"] is False


def test_pid_birthtime_distinguishes_reuse_and_unknown(monkeypatch):
    original = recovery.CLIENTS
    monkeypatch.setattr(recovery, "CLIENTS", {4242: "2026-09-23T14:37:08.417602+00:00"})
    expected = recovery.datetime.fromisoformat(recovery.CLIENTS[4242]).timestamp()

    class Process:
        def __init__(self, born):
            self.born = born

        def create_time(self):
            return self.born

        def is_running(self):
            return True

    monkeypatch.setattr(
        recovery.psutil, "Process", lambda _pid: Process(expected + 100)
    )
    assert recovery._client_status()["4242"] == "reused"
    monkeypatch.setattr(recovery.psutil, "Process", lambda _pid: Process(expected))
    assert recovery._client_status()["4242"] == "live"
    monkeypatch.setattr(
        recovery.psutil,
        "Process",
        lambda _pid: (_ for _ in ()).throw(recovery.psutil.AccessDenied(4242)),
    )
    assert recovery._client_status()["4242"].startswith("unknown:")
    monkeypatch.setattr(recovery, "CLIENTS", original)


def test_blocked_readiness_records_distinct_attempts_without_native(
    monkeypatch, tmp_path
):
    root = tmp_path / "e182-output"
    monkeypatch.setattr(recovery, "_rel", lambda _name: root)
    monkeypatch.setattr(recovery, "_capacity", lambda *_args: None)
    monkeypatch.setattr(
        recovery,
        "_readiness",
        lambda: [
            {"ready": False, "queue": {"running": 1, "pending": 0}},
            {"ready": False, "queue": {"running": 1, "pending": 0}},
        ],
    )
    args = SimpleNamespace(
        arm="baseline", freeze_sha256="a" * 64, source_review_sha256="b" * 64
    )
    for number in range(2):
        with pytest.raises(RuntimeError, match="before native"):
            recovery._driver(args, {}, None)
        receipt = root / f"baseline-preflight-{number:03d}.json"
        assert receipt.is_file()
        assert json.loads(receipt.read_text())["native_permitted"] is False
    assert not (root / "baseline-request.json").exists()
    assert not (root / "baseline-driver.json").exists()
    assert recovery._account(root)["native_arms_upper_bound"] == 0


def test_baseline_and_plus_parent_review_order_with_real_hashes(tmp_path):
    root = tmp_path / "recovery"
    root.mkdir()
    assert recovery._prior_review(root, "baseline", "a" * 64) is None
    with pytest.raises(FileNotFoundError):
        recovery._prior_review(root, "plus", "a" * 64)
    write_parent_reviews(root)
    assert recovery._prior_review(root, "plus", "a" * 64) is None
    assert recovery._prior_review(root, "minus", "a" * 64) is None
    write(root / "plus" / "result.json", {"validity": "incomplete"})
    with pytest.raises(ValueError, match="Complete E182 plus"):
        recovery._prior_review(root, "minus", "a" * 64)


@pytest.mark.parametrize(
    "changed", ["native-contact.npz", "start/array_00000.npy", "final/manifest.json"]
)
def test_review_rejects_changed_contact_and_nested_checkpoint(tmp_path, changed):
    root = tmp_path / "recovery"
    root.mkdir()
    write_parent_reviews(root)
    assert recovery._prior_review(root, "plus", "a" * 64) is None
    write(root / "baseline" / changed, {"tampered": changed})
    with pytest.raises(ValueError, match="Reviewed E182 raw changed"):
        recovery._prior_review(root, "plus", "a" * 64)


def test_review_rejects_traversal_and_unlisted_raw(tmp_path):
    root = tmp_path / "recovery"
    root.mkdir()
    baseline_artifacts, _ = write_parent_reviews(root)
    for unsafe in (
        "../escape",
        "start/../result.json",
        "C:/outside",
        "start\\state.json",
    ):
        with pytest.raises(ValueError, match="Unsafe E182 raw path"):
            recovery._check_reviewed_raw(
                root, "baseline", {**baseline_artifacts, unsafe: "0" * 64}
            )
    write(root / "baseline" / "new-raw.json", {"unreviewed": True})
    with pytest.raises(ValueError, match="inventory changed"):
        recovery._prior_review(root, "plus", "a" * 64)


def test_missing_receipt_and_failed_child_keep_conservative_cost(tmp_path):
    root = tmp_path / "recovery"
    root.mkdir()
    baseline = root / "baseline-request.json"
    write(baseline, {"synthetic": True})
    missing = recovery._account(root)
    assert missing["native_arms_exact"] is None
    assert missing["native_arms_upper_bound"] == 1
    assert missing["native_accounted_wall_s"] == e181.SUPERVISOR_S
    receipt = e181.supervise_command(
        [sys.executable, "-c", "raise RuntimeError('generated failure')"],
        baseline,
        timeout_s=10,
    )
    assert receipt["started"] is True and receipt["status"] == "child_nonzero"
    failed = recovery._account(root)
    assert failed["native_arms_exact"] == 1
    assert failed["native_accounted_wall_s"] >= 110
    assert failed["native_cost_exact"] is False


def test_blocked_preflight_costs_count_once_before_native(monkeypatch, tmp_path):
    root = tmp_path / "recovery"
    root.mkdir()
    for number, elapsed in enumerate((120.0, 111.0)):
        write(
            root / f"baseline-preflight-{number:03d}.json",
            {
                "schema": "flylab.m3.e182.preflight.v1",
                "arm": "baseline",
                "freeze_sha256": "a" * 64,
                "native_permitted": False,
                "elapsed_wall_s": elapsed,
            },
        )
    receipts, total = recovery._preflight_receipts(root, "a" * 64)
    assert len(receipts) == 2 and total == 231.0
    write(
        root / "baseline-preflight-002.json",
        {
            "schema": "flylab.m3.e182.preflight.v1",
            "arm": "baseline",
            "freeze_sha256": "a" * 64,
            "native_permitted": True,
            "elapsed_wall_s": 2.5,
        },
    )
    receipts, total = recovery._preflight_receipts(root, "a" * 64)
    assert len(receipts) == 3 and total == 233.5
    assert recovery._account(root)["native_arms_upper_bound"] == 0
    monkeypatch.setattr(
        recovery.psutil, "disk_usage", lambda _path: SimpleNamespace(free=30 << 30)
    )
    monkeypatch.setattr(e181, "_directory_bytes", lambda _path: 0)
    with pytest.raises(e181.ResourceGuard, match="340-second"):
        recovery._capacity(root, "baseline", time.perf_counter(), "a" * 64)
    assert not (root / "baseline-request.json").exists()


def test_monitor_stops_and_writes_only_resource_rows(monkeypatch, tmp_path):
    count = 0

    def sample():
        nonlocal count
        count += 1
        return {"queue": {"running": 0, "pending": 0}, "gpu_csv": ["generated"]}

    monkeypatch.setattr(recovery, "_sample_resources", sample)
    stop = threading.Event()
    errors = []
    output = tmp_path / "resources.jsonl"
    thread = threading.Thread(target=recovery._monitor, args=(output, stop, errors))
    thread.start()
    while not output.exists() or output.stat().st_size == 0:
        thread.join(timeout=0.01)
    stop.set()
    thread.join(timeout=3)
    assert not thread.is_alive() and not errors
    assert count >= 1
    assert json.loads(output.read_text().splitlines()[0])["gpu_csv"] == ["generated"]


def test_help_is_inert_and_exposes_three_separate_arms():
    script = Path(recovery.__file__)
    completed = subprocess.run(
        [sys.executable, "-X", "utf8", str(script), "--help"],
        cwd=script.parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--arm {baseline,plus,minus}" in completed.stdout
    assert "--child-request" not in completed.stdout
