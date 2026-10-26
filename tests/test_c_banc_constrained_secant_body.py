"""Generated-only E327 gate/classification and failed-inclusive body journal tests."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix

from tools import verify_c_banc_m3_constrained_secant_body as body


def sample_evaluation(*, candidate):
    return {"shape": {"shape_loss": .9 if candidate else 1.,
                      "startup_q_rms": .017, "startup_q_max": .039,
                      "startup_tendon_rms": .008, "startup_tendon_max": .024},
            "short": {"health": {"healthy": True}},
            "summary": {"forward_mm": 1.7, "slip_mm_s": 1.79,
                        "longest_continuous_forward_bout_s": .31,
                        "complete_cycles": {leg: 1 for leg in
                                            ("lf", "lm", "lh", "rf", "rm", "rh")},
                        "health": {"healthy": True}}}


def test_single_conjunction_exposes_each_existing_threshold():
    candidate, neutral = sample_evaluation(candidate=True), sample_evaluation(candidate=False)
    gates, supported = body.classify(candidate, neutral, engineering_valid=True)
    assert supported and all(gates.values())
    candidate["summary"]["slip_mm_s"] = 1.81
    gates, supported = body.classify(candidate, neutral, engineering_valid=True)
    assert not supported and gates["slip"] is False
    assert gates["shape_better_than_saved_neutral"] is True
    candidate["summary"]["slip_mm_s"] = 1.79
    candidate["summary"]["complete_cycles"]["rh"] = 0
    gates, supported = body.classify(candidate, neutral, engineering_valid=True)
    assert not supported and gates["all_six_eligible_cycles"] is False


def test_transitive_input_roster_includes_both_producers(monkeypatch):
    actual = set(body.input_roster())
    e326_freeze = json.loads((body.ROOT / body.PATHS["e326_freeze"]).read_text("utf8"))
    e318_freeze = json.loads((body.ROOT / body.PATHS["e318_freeze"]).read_text("utf8"))
    assert set(e326_freeze["input_sha256"]) <= actual
    assert set(e318_freeze["input_sha256"]) <= actual
    assert set(body.e323._file_roster(body.PATHS, body.ROOT)) <= actual
    assert body.PATHS["graph_bundle"] == body.e318.PATHS["graph_bundle"]


class FakeClock:
    def __init__(self):
        self.events = []
        self.clock_valid = True
        self.neural_s = 0.
        self.physical_s = 0.


class FakeJournal:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self):
        self.control_tick = 0
        self.network = SimpleNamespace(tick=0)
        self.body = SimpleNamespace(fault=False)
        self.fault = False
        self.closed = False
        self.ssp_efficacy_trial_patch = {
            "positions": np.array([0], np.int64),
            "edge_parameters": np.array([0], np.int64),
            "theta64": np.array([1.1], np.float64),
            "original32": np.array([.5], np.float32),
            "actual32": np.array([.55], np.float32)}

    def close(self):
        self.closed = True


class FakeDependencies:
    class health_stop(RuntimeError):
        pass

    def __init__(self, fail_control=None, raw_fail=False):
        self.session = FakeSession()
        self.fail_control = fail_control
        self.raw_fail = raw_fail
        self.detached = False
        self.journal_object = FakeJournal()
        native = SimpleNamespace(
            AdvanceLedger=FakeClock,
            native_arrays=lambda records: {
                "upright": np.ones(len(records), np.float64),
                "nonfoot_contact": np.zeros(len(records), bool)})
        self.base = SimpleNamespace(e310=SimpleNamespace(
            runner=SimpleNamespace(native=native)))
        self.native_records = None
        self.partial_saved = False

    def load(self, _freeze):
        return {"ids": np.array([0], np.int64), "curve": None, "startup": None}

    def construct(self, _data, ledger, _persist, on_return):
        ledger["construction"].append({"unknown_setup_steps": 0,
                                        "unknown_capture_setup_steps": 0})
        ledger["body_construction"].append({"unknown_step_entries": 0,
                                             "active_unreturned_frames": 0})
        on_return(self.session)
        return self.session

    def initial(self, *_args):
        return {"generated": True}

    def save_state(self, path, _session):
        if Path(path).name == "partial-final":
            self.partial_saved = True

    def save_partial(self, *_args):
        pass

    def trace_metadata(self, _session):
        return {}

    def install(self, _session, native_records):
        self.native_records = native_records
        return lambda: setattr(self, "detached", True)

    def journal(self, _path):
        return self.journal_object

    def capture(self, session, raw):
        raw["rows"].append({"control_tick": session.control_tick})
        raw["activity"].append({"control_tick": session.control_tick})

    def advance(self, session, _arm, control, clock, _journal):
        if control == self.fail_control:
            raise RuntimeError("generated primary advance failure")
        session.network.tick += 20
        session.control_tick = control
        self.native_records.extend(range(50))
        clock.neural_s += .005
        clock.physical_s += .005
        clock.events.append({"control": control})
        return {"control": control}

    def readout_receipt(self, *_args):
        pass

    def save_raw(self, *_args):
        return ([{"artifact": "raw", "error": "generated raw failure"}]
                if self.raw_fail else [])


def meter():
    return {"host_rss_bytes": 1, "host_peak_bytes": 1,
            "device_free_bytes": 10 << 30, "device_total_bytes": 16 << 30,
            "torch_reserved_bytes": 0, "torch_peak_reserved_bytes": 0}


def setup_generated(monkeypatch):
    from tools import probe_c_banc_m3_core_rhythm as probe
    monkeypatch.setattr(probe, "source_matrix", lambda _network:
                        csr_matrix(np.array([[.55]], np.float32)))
    monkeypatch.setattr(body, "evaluate_saved", lambda folder, *_args:
                        copy.deepcopy(sample_evaluation(candidate=Path(folder).name
                                                         != "e318-full-drive-capture")))


def invoke(tmp_path, deps, **kwargs):
    return body.run(tmp_path / "freeze", "x", tmp_path / "review", "y",
                    tmp_path / "out", dependencies=deps,
                    gatekeeper=lambda *_args, **_kwargs:
                        {"source_sha256": {}, "input_sha256": {}},
                    meter=meter, **kwargs)


def test_partial_failure_keeps_first_error_and_owns_body(tmp_path, monkeypatch):
    setup_generated(monkeypatch)
    deps = FakeDependencies(fail_control=2, raw_fail=True)
    result = invoke(tmp_path, deps)
    assert not result["run_valid"] and not result["constrained_body_supported"]
    assert result["first_error"]["message"] == "generated primary advance failure"
    assert result["cost"]["controls_completed"] == 1
    assert result["cost"]["physics_steps_completed"] == 50
    assert deps.partial_saved and deps.detached and deps.session.closed
    assert deps.journal_object.closed
    assert any(row["stage"] == "raw_raw" for row in result["errors"])
    assert (tmp_path / "out/publication-failure.json").exists()


def test_late_publication_failure_clears_supported_result(tmp_path, monkeypatch):
    setup_generated(monkeypatch)
    deps = FakeDependencies()
    clock_value = [0.]
    def writer(path, value):
        body._json(path, value)
        if Path(path).name == "output-manifest.json":
            clock_value[0] = 71.
    result = invoke(tmp_path, deps, writer=writer, clock=lambda: clock_value[0])
    assert result["first_error"]["stage"] == "finalization_deadline"
    assert not result["run_valid"] and not result["constrained_body_supported"]
    saved = json.loads((tmp_path / "out/result.json").read_text("utf8"))
    assert not saved["run_valid"] and saved["first_error"] == result["first_error"]
    assert deps.session.closed and deps.detached
