import json
import copy
import os
from pathlib import Path
import tempfile
import unittest

from flylab.c.cuda_campaign import Campaign, allocation_failure
from tests.c_fixtures import graph_fixture, bindings_fixture


class CampaignValidation(unittest.TestCase):
    def test_complete_requires_progress_and_checkpoint_evidence(self):
        graph = graph_fixture()
        bindings = bindings_fixture(graph)
        spec = dict(schema="flylab.cuda-campaign.v1", jobs=[dict(id="a")], seconds=0.01)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "campaign"
            c = Campaign(graph, bindings, spec, out)
            for tick in (0, 100):
                with self.subTest(tick=tick):
                    c.state["status"] = "COMPLETE"
                    c.state["jobs"]["a"].update(status="COMPLETE", tick=tick)
                    c._save()
                    before = c.path.read_bytes()
                    with self.assertRaises(ValueError):
                        Campaign(graph, bindings, spec, out, resume=True)
                    with self.assertRaises(ValueError):
                        c.run()
                    self.assertEqual(c.path.read_bytes(), before)

    def test_cancelled_before_start_is_a_valid_complete_campaign(self):
        graph = graph_fixture()
        bindings = bindings_fixture(graph)
        spec = dict(schema="flylab.cuda-campaign.v1", jobs=[dict(id="a")], seconds=0.01)

        def never_allocate(*args, **kwargs):
            raise AssertionError("Cancelled work must not allocate a session")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "campaign"
            c = Campaign(graph, bindings, spec, out, session_type=never_allocate)
            c.cancel(["a"])
            self.assertEqual(c.run()["status"], "COMPLETE")
            result = Campaign(
                graph, bindings, spec, out, resume=True, session_type=never_allocate
            ).run()
            self.assertEqual(result["jobs"]["a"]["tick"], 0)
            self.assertEqual(result["jobs"]["a"]["status"], "CANCELLED")

    def test_single_world_creation_oom_is_bounded_and_retryable(self):
        graph = graph_fixture()
        bindings = bindings_fixture(graph)
        attempts = []
        spec = dict(
            schema="flylab.cuda-campaign.v1",
            jobs=[dict(id="a")],
            seconds=0.01,
            max_worlds=1,
        )

        def unavailable(*args, **kwargs):
            attempts.append(1)
            raise MemoryError("test allocation unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "campaign"
            for attempt in range(2):
                result = Campaign(
                    graph,
                    bindings,
                    spec,
                    out,
                    resume=attempt > 0,
                    session_type=unavailable,
                ).run()
                self.assertEqual(result["status"], "PAUSED")
                self.assertIn("waiting_for_memory", result)
                self.assertEqual(result["jobs"]["a"]["tick"], 0)
                self.assertEqual(len(attempts), attempt + 1)

    def test_resume_rejects_escaping_checkpoints_and_changed_inventory(self):
        graph = graph_fixture()
        bindings = bindings_fixture(graph)
        spec = dict(
            schema="flylab.cuda-campaign.v1", jobs=[dict(id="a")], seconds=0.005
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "campaign"
            c = Campaign(graph, bindings, spec, out)
            c.state["current"] = dict(
                jobs=["a"],
                group=c.jobs["a"]["group"],
                controls=0,
                checkpoint_controls=0,
                checkpoint="../foreign",
            )
            c._save()
            with self.assertRaisesRegex(ValueError, "escapes"):
                Campaign(graph, bindings, spec, out, resume=True)
            c.state["current"] = None
            c.state["jobs"]["foreign"] = c.state["jobs"]["a"]
            c._save()
            with self.assertRaisesRegex(ValueError, "inventory"):
                Campaign(graph, bindings, spec, out, resume=True)

    def test_rejects_invalid_specs_and_does_not_treat_other_failures_as_oom(self):
        graph = graph_fixture()
        bindings = bindings_fixture(graph)
        with tempfile.TemporaryDirectory() as tmp:
            for seconds in (float("nan"), True, 0.006, -1):
                with self.assertRaises(ValueError):
                    Campaign(
                        graph,
                        bindings,
                        dict(
                            schema="flylab.cuda-campaign.v1",
                            jobs=[dict(id="a")],
                            seconds=seconds,
                        ),
                        Path(tmp) / "invalid",
                    )
        self.assertTrue(allocation_failure(MemoryError()))
        self.assertFalse(
            allocation_failure(ValueError("CUDA checkpoint shape mismatch"))
        )


@unittest.skipUnless(
    os.environ.get("FLYLAB_CUDA_BATCH_TESTS") == "1",
    "Explicit CUDA campaign verification",
)
class NativeCampaign(unittest.TestCase):
    def test_resume_oom_splits_saved_worlds_and_preserves_recordings_and_cancellation(
        self,
    ):
        import cupy as cp
        import numpy as np
        from flylab.c.batch import BatchSession
        from flylab.c.storage import StateStore

        pool = cp.cuda.MemoryPool()
        pool.set_limit(size=16384)
        restored = []

        class LimitedRestore:
            @classmethod
            def restore(cls, graph, bindings, path, *, worlds=None):
                saved = StateStore.load(path, max_files=4096)
                selected = list(saved["engines"]) if worlds is None else worlds
                with cp.cuda.using_allocator(pool.malloc):
                    reservation = cp.empty(len(selected) * 4096, dtype=cp.float32)
                del reservation
                pool.free_all_blocks()
                session = BatchSession.restore(graph, bindings, path, worlds=worlds)
                for i, key in enumerate(selected):
                    actual = session.engines[str(i)].checkpoint()
                    expected = saved["engines"][key]
                    self.assertEqual(actual["control_tick"], expected["control_tick"])
                    self.assertEqual(actual["seed"], expected["seed"])
                    for field in (
                        "v",
                        "h",
                        "rate",
                        "queue",
                        "spike_count",
                        "refractory_until",
                        "suppress",
                        "mute",
                    ):
                        np.testing.assert_array_equal(
                            actual["neural"][field], expected["neural"][field]
                        )
                    np.testing.assert_array_equal(
                        actual["encoder"]["filtered"], expected["encoder"]["filtered"]
                    )
                    restored.append((actual["seed"], actual["control_tick"]))
                return session

        graph = graph_fixture()
        bindings = bindings_fixture(graph)
        spec = dict(
            schema="flylab.cuda-campaign.v1",
            seconds=0.015,
            max_worlds=2,
            record_channels=4,
            jobs=[dict(id="a", seed=42), dict(id="b", seed=43), dict(id="c", seed=44)],
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "campaign"
            original = Campaign(graph, bindings, spec, out).run(max_controls=1)
            source = out / original["current"]["checkpoint"]
            source_bytes = (source / "manifest.json").read_bytes()
            result = Campaign(
                graph, bindings, spec, out, resume=True, session_type=LimitedRestore
            ).run(max_controls=1)
            self.assertEqual(result["status"], "PAUSED")
            self.assertEqual(result["capacity"], 1)
            self.assertEqual(result["jobs"]["a"]["tick"], 100)
            self.assertEqual(result["jobs"]["b"]["tick"], 50)
            self.assertEqual([c["jobs"] for c in result["pending_chunks"]], [["b"]])
            continued = Campaign(
                graph, bindings, spec, out, resume=True, session_type=LimitedRestore
            )
            continued.cancel(["b", "c"])
            result = continued.run()
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(
                {k: (s["status"], s["tick"]) for k, s in result["jobs"].items()},
                {"a": ("COMPLETE", 150), "b": ("CANCELLED", 50), "c": ("CANCELLED", 0)},
            )
            self.assertIn((42, 1), restored)
            self.assertIn((42, 2), restored)
            self.assertIn((43, 1), restored)
            self.assertEqual((source / "manifest.json").read_bytes(), source_bytes)
            self.assertTrue(
                any(
                    a["phase"] == "restore" and a["status"] == "OOM"
                    for a in result["allocation_attempts"]
                )
            )
            self.assertTrue(
                all("regroup_origin" in c for c in result["completed_chunks"])
            )
            for path in (out / "recordings").glob("*/*/manifest.json"):
                manifest = json.loads(path.read_text())
                self.assertEqual(len(manifest["cohort_ids"]), 4)
                self.assertIn(manifest["status"], ("COMPLETE", "CANCELLED"))
            self.assertEqual(
                Campaign(graph, bindings, spec, out, resume=True).run()["status"],
                "COMPLETE",
            )

    def test_single_world_restore_oom_preserves_checkpoint_for_later_retry(self):
        import cupy as cp

        pool = cp.cuda.MemoryPool()
        pool.set_limit(size=8192)

        class UnavailableRestore:
            @classmethod
            def restore(cls, *args, **kwargs):
                with cp.cuda.using_allocator(pool.malloc):
                    cp.empty(4096, dtype=cp.float32)

        graph = graph_fixture()
        bindings = bindings_fixture(graph)
        spec = dict(
            schema="flylab.cuda-campaign.v1",
            seconds=0.01,
            max_worlds=1,
            record_channels=0,
            jobs=[dict(id="a")],
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "campaign"
            saved = Campaign(graph, bindings, spec, out).run(max_controls=1)
            checkpoint = copy.deepcopy(saved["current"])
            for _ in range(2):
                result = Campaign(
                    graph,
                    bindings,
                    spec,
                    out,
                    resume=True,
                    session_type=UnavailableRestore,
                ).run()
                self.assertEqual(result["status"], "PAUSED")
                self.assertIn("waiting_for_memory", result)
                self.assertEqual(result["current"], checkpoint)
                self.assertEqual(result["jobs"]["a"]["tick"], 50)
            result = Campaign(graph, bindings, spec, out, resume=True).run()
            self.assertEqual(result["status"], "COMPLETE")
            self.assertNotIn("waiting_for_memory", result)
            self.assertEqual(result["jobs"]["a"]["tick"], 100)

    def test_resume_rejects_inconsistent_hashed_checkpoint_references(self):
        graph = graph_fixture()
        bindings = bindings_fixture(graph)
        spec = dict(
            schema="flylab.cuda-campaign.v1",
            seconds=0.015,
            max_worlds=2,
            record_channels=0,
            jobs=[dict(id="a", seed=42), dict(id="b", seed=43)],
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "campaign"
            c = Campaign(graph, bindings, spec, out)
            original = copy.deepcopy(c.run(max_controls=1))
            cases = []
            bad = copy.deepcopy(original)
            bad["current"]["checkpoint_hash"] = "0" * 64
            cases.append(("hash", bad))
            bad = copy.deepcopy(original)
            bad["current"]["checkpoint"] = "checkpoints/missing"
            cases.append(("missing", bad))
            bad = copy.deepcopy(original)
            bad["current"]["checkpoint_worlds"] = ["1", "0"]
            cases.append(("identity", bad))
            bad = copy.deepcopy(original)
            bad["pending_chunks"] = [copy.deepcopy(bad["current"])]
            cases.append(("duplicate", bad))
            bad = copy.deepcopy(original)
            bad["status"] = "COMPLETE"
            cases.append(("pending", bad))
            bad = copy.deepcopy(original)
            bad["status"] = "COMPLETE"
            bad["completed_chunks"] = [bad.pop("current")]
            bad["current"] = None
            for job in bad["jobs"].values():
                job.update(status="COMPLETE", tick=150)
            bad["completed_chunks"][0].update(controls=3, checkpoint_controls=3)
            cases.append(("false progress", bad))
            for label, state in cases:
                with self.subTest(case=label):
                    c.state = state
                    c._save()
                    before = c.path.read_bytes()
                    with self.assertRaises(ValueError):
                        Campaign(graph, bindings, spec, out, resume=True)
                    self.assertEqual(c.path.read_bytes(), before)
            c.state = copy.deepcopy(original)
            c._save()
            payload = out / original["current"]["checkpoint"] / "state.json"
            good = payload.read_bytes()
            try:
                payload.write_bytes(good + b" ")
                with self.assertRaisesRegex(ValueError, "file hash"):
                    Campaign(graph, bindings, spec, out, resume=True)
            finally:
                payload.write_bytes(good)
            # Older valid v1 journals have no outer checkpoint hash or pending queue.
            c.state = copy.deepcopy(original)
            c.state["current"].pop("checkpoint_hash")
            c.state.pop("pending_chunks")
            c._save()
            self.assertEqual(
                Campaign(graph, bindings, spec, out, resume=True).state["jobs"]["a"][
                    "tick"
                ],
                50,
            )
            # A journal written after its checkpoint may contain uncommitted
            # progress. Restore from tick 50 and finish at 150, without adding it.
            c.state = copy.deepcopy(original)
            c.state["current"]["controls"] = 2
            for job in c.state["jobs"].values():
                job["tick"] = 100
            c._save()
            result = Campaign(graph, bindings, spec, out, resume=True).run(
                max_controls=1
            )
            self.assertTrue(all(j["tick"] == 100 for j in result["jobs"].values()))

    def test_oom_during_step_remains_non_resumable(self):
        from unittest.mock import patch
        from flylab.c.batch import BatchSession

        graph = graph_fixture()
        bindings = bindings_fixture(graph)
        spec = dict(
            schema="flylab.cuda-campaign.v1",
            seconds=0.01,
            max_worlds=1,
            record_channels=0,
            jobs=[dict(id="a")],
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "campaign"
            c = Campaign(graph, bindings, spec, out)
            with patch.object(
                BatchSession, "step", side_effect=MemoryError("test in-period OOM")
            ):
                with self.assertRaises(MemoryError):
                    c.run()
            self.assertEqual(c.state["status"], "FAILED")
            self.assertNotIn("waiting_for_memory", c.state)
            with self.assertRaises(ValueError):
                Campaign(graph, bindings, spec, out, resume=True)

    def test_allocation_backoff_keeps_every_job_and_model_group(self):
        from flylab.c.batch import BatchSession
        import cupy as cp

        pool = cp.cuda.MemoryPool()
        pool.set_limit(size=16384)

        class LimitedSession:
            def __new__(cls, graph, bindings, seeds, **kwargs):
                # A real CuPy allocation failure within a private 16 KiB pool;
                # this does not exhaust the device or other applications' VRAM.
                with cp.cuda.using_allocator(pool.malloc):
                    reservation = cp.empty(len(seeds) * 4096, dtype=cp.float32)
                del reservation
                pool.free_all_blocks()
                return BatchSession(graph, bindings, seeds, **kwargs)

        graph = graph_fixture()
        bindings = bindings_fixture(graph)
        spec = dict(
            schema="flylab.cuda-campaign.v1",
            seconds=0.005,
            max_worlds=2,
            record_channels=4,
            jobs=[
                dict(id="flat-a", seed=42),
                dict(id="flat-b", seed=43),
                dict(id="slope-a", seed=44, body_options=dict(terrain="slope")),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = Campaign(
                graph,
                bindings,
                spec,
                Path(tmp) / "campaign",
                session_type=LimitedSession,
            ).run()
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(set(result["jobs"]), {"flat-a", "flat-b", "slope-a"})
            self.assertTrue(
                all(
                    j["status"] == "COMPLETE" and j["tick"] == 50
                    for j in result["jobs"].values()
                )
            )
            self.assertEqual(result["allocation_attempts"][0]["status"], "OOM")
            self.assertEqual(len({c["group"] for c in result["completed_chunks"]}), 2)

    def test_resume_and_cancel_preserve_per_world_ticks(self):
        graph = graph_fixture()
        bindings = bindings_fixture(graph)
        spec = dict(
            schema="flylab.cuda-campaign.v1",
            seconds=0.015,
            max_worlds=2,
            record_channels=4,
            jobs=[dict(id="a", seed=42), dict(id="b", seed=43)],
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "campaign"
            paused = Campaign(graph, bindings, spec, out).run(max_controls=1)
            self.assertEqual(paused["status"], "PAUSED")
            continued = Campaign(graph, bindings, spec, out, resume=True)
            continued.cancel(["b"])
            result = continued.run()
            self.assertEqual(result["jobs"]["a"]["tick"], 150)
            self.assertEqual(result["jobs"]["a"]["status"], "COMPLETE")
            self.assertEqual(result["jobs"]["b"]["tick"], 50)
            self.assertEqual(result["jobs"]["b"]["status"], "CANCELLED")
            self.assertEqual(
                json.loads((out / "campaign.json").read_text())["status"], "COMPLETE"
            )


if __name__ == "__main__":
    unittest.main()
