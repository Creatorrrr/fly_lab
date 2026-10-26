"""Generated-only E352 current frozen analyzer and precise clock/reader guards."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools import analyze_c_banc_m3_cycle_coverage as p


def data():
    t = np.arange(301) * 0.005
    window = (t >= 0.5) & (t < 1.5)
    ap = np.tile((0.15 * np.cos(2 * np.pi * 10 * t))[:, None], (1, 6))
    return {
        "t": t,
        "dt": 0.005,
        "ap": ap,
        "window": window,
        "moving": window.copy(),
        "speed": np.ones(301),
        "straight": np.ones(301, bool),
        "source": {},
        "knee": np.ones((301, 6)) * 120,
        "lengths": np.ones(6),
        "start_s": 0.5,
        "end_s": 1.5,
        "yaw_rate": np.zeros(301),
        "position": np.c_[t, np.zeros((301, 2))],
        "upright": np.ones(301),
        "faults": [],
        "contact": np.ones((301, 6), bool),
    }


def native(count=15000):
    return {
        "time_s": np.arange(1, count + 1) * 1e-4,
        "normal_load_bw": np.full((count, 6), 0.1),
        "slip_mm_s": np.full((count, 6), 2.0),
        "slip_valid": np.ones((count, 6), bool),
        "upright": np.ones(count),
        "nonfoot_contact": np.zeros(count, bool),
    }


def test_actual_analyzer_periodic_events_and_mask_not_filter():
    d = data()
    result = p.coverage(d)
    actual = p.gait().phase_and_events(d)
    counts = [len(v) for v in actual[1]]
    assert min(counts) >= 8
    assert [
        result["standard"]["events"][leg]["complete_cycles"] for leg in p.LEGS
    ] == counts
    assert result["phase_unavailable_count"] > 0
    highfloor = p.gait().phase_and_events(d, amplitude_floor=100.0)
    assert np.isnan(highfloor[0]).all()
    assert (
        highfloor[1] == actual[1]
    )  # Actual analyzer keeps peak events without phase validity.
    assert result["moving_runs"][0]["hi"] == 300
    assert result["moving_runs"][0]["end_s"] == 1.495


@pytest.mark.parametrize("length", [1, 2, 3, 4])
def test_short_runs_raw_geometry_without_filter_or_official_count(length):
    d = data()
    d["speed"][:] = 0
    d["speed"][120 : 120 + length] = 1
    d["moving"] = d["window"] & (d["speed"] > 0.5)
    result = p.coverage(d)
    run = result["moving_runs"][0]
    assert run["eligible"] is False
    assert run["duration_s"] == pytest.approx((length - 1) * 0.005)
    assert all(v == 0 for v in run["standard_count_by_leg"].values())
    for row in run["geometry"].values():
        assert row["peaks"] is row["pairs"] is None
        assert row["filter_unavailable_mask"] is True
        assert row["raw_start"] is not None


def test_gaps_whole_window_diagnostic_pairs_never_add_official_cycles():
    d = data()
    d["speed"][::40] = 0
    d["moving"] = d["window"] & (d["speed"] > 0.5)
    r = p.coverage(d)
    assert all(x["complete_cycles"] == 0 for x in r["standard"]["events"].values())
    pairs = r["whole_window_geometry"]["lf"]["pairs"]
    assert len(pairs) >= 8
    assert any(
        x["classification"] == "crosses_nonmoving_or_multiple_runs" for x in pairs
    )
    assert all(x["contributes_to_standard_count"] is False for x in pairs)


def test_phase_wrap_backtracking_tie_and_blend_edges():
    r = p.phase_progress([0.9, 0.1, 0.05, 0.55], [118, 119, 120, 121])
    assert r["net_cycles"] == pytest.approx(-0.35)
    assert r["positive_variation_cycles"] == pytest.approx(0.2)
    assert r["negative_variation_cycles"] == pytest.approx(0.55)
    assert r["span_cycles"] == pytest.approx(0.55)
    assert r["direction_ambiguous"] is True
    assert r["circular_wrap_to_ticks"] == [119]
    assert len(r["blend_bins"]["blend_internal"]) == 1
    assert len(r["blend_bins"]["crosses_blend_boundary"]) == 1
    assert len(r["blend_bins"]["fully_shaped_internal"]) == 1
    with pytest.raises(ValueError, match="contiguous"):
        p.phase_progress([0.1, 0.2], [100, 102])


def test_native_crossing_censor_clipping_halfopen_union_and_endpoint():
    d = data()
    n = native()
    d["speed"][101:103] = 0
    r = p.speed_context(d, n)
    assert len(r["sampled_crossings"]) == 2
    first = r["sampled_crossings"][0]
    assert first["bracket_s"] == [0.5, 0.505]
    assert first["context_half_open_s"] == pytest.approx([0.5, 0.525])
    assert first["summary"]["samples"] == 250
    assert r["union_summary"]["samples"] < sum(
        x["summary"]["samples"] for x in r["sampled_crossings"]
    )
    assert len(r["union_indices"]) == len(set(r["union_indices"]))
    assert 14999 not in r["union_indices"]
    d["speed"][:] = 1
    assert p.speed_context(d, n)["sampled_crossings"] == []


def test_native_rounding_jitter_stable_and_bad_clock_rejected():
    d = data()
    d["speed"][299] = 0
    n = native()
    a = p.speed_context(d, n)
    n["time_s"] += 1e-8
    b = p.speed_context(d, n)
    assert a["union_indices"] == b["union_indices"]
    assert b["sampled_crossings"][0]["last_native_index"] == 14999
    n["time_s"][5] += 4e-8
    with pytest.raises(ValueError, match="clock"):
        p.speed_context(d, n)


def test_weighted_slip_missing_and_zero_support_unavailable():
    n = native(3)
    chosen = np.ones(3, bool)
    n["normal_load_bw"][:, 0] = 0.2
    n["slip_mm_s"][:, 0] = 4
    assert p.contact_summary(n, chosen)["weighted_slip_mm_s"] == pytest.approx(18 / 7)
    n["slip_mm_s"][0, 0] = np.nan
    n["slip_valid"][0, 0] = False
    assert p.contact_summary(n, chosen)["weighted_slip_mm_s"] is None
    assert p.contact_summary(n, chosen)["supported_slip_missing_count"] == 1
    n["normal_load_bw"][:] = 0
    r = p.contact_summary(n, chosen)
    assert r["weighted_slip_mm_s"] is None and r["unavailable_reason"] == "zero_support"
    assert r["support_leg_occupancy"][0] == 1.0


@pytest.mark.parametrize("mutation", ["Inf", "mask", "negative", "dtype"])
def test_native_invalid_not_sanitized(mutation):
    n = native(3)
    if mutation == "Inf":
        n["slip_mm_s"][0, 0] = np.inf
    if mutation == "mask":
        n["slip_valid"][0, 0] = False
    if mutation == "negative":
        n["normal_load_bw"][0, 0] = -0.1
    if mutation == "dtype":
        n["nonfoot_contact"] = np.zeros(3, np.int64)
    with pytest.raises(ValueError):
        p.native_validate(n, expected_count=3)


def six_fixtures(tmp_path):
    specs = []
    for k in range(6):
        file = tmp_path / f"{k}.npz"
        np.savez(file, x=np.arange(4, dtype=np.float64), unselected=np.zeros(3))
        raw = file.read_bytes()
        specs.append(
            {
                "path": file.name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return specs


def test_six_path_reader_capability_exact_descriptor_and_once(tmp_path):
    specs = six_fixtures(tmp_path)
    r = p.SixReader(specs, root=tmp_path)
    with pytest.raises(ValueError, match="gate"):
        r.npz(specs[0], {"x": ((4,), "float64")})
    assert r.reader.read_counts == {}
    r = p.SixReader(specs, root=tmp_path, generated=True)
    # No selected header metadata means raw exact member roster must reject extras.
    with pytest.raises(ValueError, match="roster"):
        r.npz(specs[0], {"x": ((4,), "float64")})
    for _ in range(2):
        r.npz(specs[1], {"x": ((4,), "float64"), "unselected": ((3,), "float64")})
    assert r.reader.read_counts["1.npz"] == 1
    wrong = dict(specs[1], sha256="0" * 64)
    with pytest.raises(ValueError, match="six-path"):
        r.npz(wrong, {})
    with pytest.raises(ValueError, match="six-path"):
        r.npz(dict(specs[1], path="../bad.npz"), {})


def test_selected_extra_unopened_and_bounded_header(tmp_path):
    import zipfile

    specs = six_fixtures(tmp_path)
    r = p.SixReader(specs, root=tmp_path, generated=True)
    with zipfile.ZipFile(tmp_path / specs[0]["path"]) as z:
        info = z.getinfo("x.npy")
        selected = {
            "x": {
                "member": "x.npy",
                "npy_version": [1, 0],
                "header_bytes": 128,
                "fortran_order": False,
                "zip_compression": info.compress_type,
                "compressed_bytes": info.compress_size,
                "uncompressed_bytes": info.file_size,
                "crc32": info.CRC,
            }
        }
    assert np.array_equal(
        r.npz(specs[0], {"x": ((4,), "float64")}, selected=selected)["x"], np.arange(4)
    )
    assert r.reader.opened_members == [("0.npz", "x.npy")]
    bad = copy.deepcopy(selected)
    bad["x"]["header_bytes"] = 129
    with pytest.raises(ValueError, match="header"):
        r.npz(specs[0], {"x": ((4,), "float64")}, selected=bad)


def test_metadata_real_pins_stat_only(monkeypatch):
    original = Path.open

    def checked(self, *args, **kwargs):
        assert self.suffix not in (".npz", ".npy")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", checked)
    ctx = p.metadata_preflight()
    assert len(ctx["descriptors"]) == 6
    assert all(
        not name.endswith((".npz", ".npy")) for name in ctx["reader"].read_counts
    )
    assert ctx["final"]["diagnostic_completion_qualified"] is True
    assert ctx["inherited_report"]["external_completion_qualified"] is False


def test_gate_bad_freeze_before_arrays(tmp_path, monkeypatch):
    ctx = p.metadata_preflight()
    files = {"freeze": {"qualification": {"path": "bad"}}, "review": {}}
    monkeypatch.setattr(p, "metadata_preflight", lambda *a, **kw: ctx)
    monkeypatch.setattr(p, "make_freeze", lambda *a, **kw: {"different": True})
    original = ctx["reader"].json
    monkeypatch.setattr(
        ctx["reader"],
        "json",
        lambda spec: files[spec["path"]] if spec["path"] in files else original(spec),
    )
    (tmp_path / "freeze").write_text("{}")
    (tmp_path / "review").write_text("{}")
    with pytest.raises(ValueError, match="freeze"):
        p.original_gate("freeze", "review", root=tmp_path)
    assert not ctx["reader"].array_authorized


def test_sole_cli_rejects_extras_before_metadata(monkeypatch):
    monkeypatch.setattr(
        p, "original_gate", lambda *a, **kw: pytest.fail("gate touched")
    )
    with pytest.raises(ValueError, match="sole"):
        p.main(["--analyze", "--freeze", "f", "--review", "r"])


def test_strict_publication_late_close_and_caps(tmp_path):
    def late():
        if (tmp_path / "late" / "report.json").exists():
            raise TimeoutError("after close")

    report, code = p.publish_outcome(
        tmp_path / "late",
        lambda: {"diagnostic_verified": True, "value": np.bool_(True)},
        timely=late,
    )
    assert code == 1 and report["provisional_report_present"] is True
    assert report["analysis_completed"] is True
    assert report["late_error_has_precedence"] is True
    assert report["secondary_errors"]
    assert json.loads((tmp_path / "late" / "report.json").read_text())["value"] is True
    with pytest.raises(ValueError, match="cap"):
        p.helper().publish(tmp_path / "oversize", {"x": "x" * 100}, output_cap=20)
    assert not (tmp_path / "oversize").exists()


@pytest.mark.parametrize(
    "value",
    [np.array(0), np.float64(np.inf), float("nan"), complex(1), np.longdouble(0.1)],
)
def test_strict_success_and_error_serialization(value, tmp_path):
    r, code = p.publish_outcome(
        tmp_path / "bad", lambda: {"diagnostic_verified": True, "bad": value}
    )
    assert code == 1 and r["diagnostic_verified"] is False
    assert json.loads((tmp_path / "bad" / "failure.json").read_text())["arms"] is None
