"""Resolve the existing E318 control0/1..99 observations after real admission."""
from __future__ import annotations


def load_e318(admission, core, ledger, inventory):
    # This is an Original reader. Even numerical imports follow its cold gate.
    admission.fence("pre_array")
    import numpy as np

    from tools import verify_c_banc_m3_common_descent_body as body
    from tools import verify_c_banc_m3_startup_ray as ray

    binding = admission.inputs["e318_startup"]
    manifest = ray.json_load(ray.leaf(binding["manifest"], admission.root))
    result = ray.json_load(ray.leaf(binding["result"], admission.root))
    ray.require(result["run_valid"] is True and result["learning_input_approved"] is True
                and result["first_error"] is None and not result["errors"], "valid historical E318 producer")
    ray.require(len(binding["chunks"]) == 4, "only four fixed E318 startup chunks")

    def selected(spec):
        relative = spec["path"].removeprefix(binding["output"] + "/")
        ray.require(relative != spec["path"] and relative in manifest["files"], "E318 member declared by original producer")
        producer = manifest["files"][relative]
        ray.require(spec["sha256"] == producer["sha256"] and spec["bytes"] == producer["bytes"], "same E318 produced bytes")
        admission.fence("pre_array")
        # Recruitment archives also contain Unicode provenance columns. Only the
        # declared numeric members enter the calculation; preserve whole-file hashes.
        p = ray.path(spec["path"], admission.root)
        before = p.stat()
        ray.require(before.st_size == spec["bytes"] and core.digest(p, admission.check) == spec["sha256"], "E318 prehash")
        with np.load(p, allow_pickle=False) as archive:
            values = {}
            for key, schema in spec["selected_schema"].items():
                admission.check("science")
                value = archive[key]
                ray.require(value.dtype == np.dtype(schema["dtype"]) and list(value.shape) == schema["shape"]
                            and value.dtype.kind in "bifu" and np.isfinite(value).all(), "E318 selected numeric member " + key)
                values[key] = value.copy()
        ray.require(p.stat().st_size == before.st_size and p.stat().st_mtime_ns == before.st_mtime_ns
                    and core.digest(p, admission.check) == spec["sha256"], "E318 posthash")
        ledger.append({"path": spec["path"], "expected_sha256": spec["sha256"], "bytes": spec["bytes"],
                       "selected": sorted(values), "posthash_verified": True})
        return values

    chunks = []
    for i, spec in enumerate(binding["chunks"]):
        observed = selected(spec)
        ticks = np.arange(25 * i + 1, 25 * i + 26, dtype=np.int64)
        ray.require(core.exact(observed["control_tick"], ticks)
                    and core.exact(observed["network_tick"], ticks * 20), "E318 contiguous twenty-tick controls")
        receipt = result["chunks"][i]
        ray.require(receipt["start"] == ticks[0] and receipt["end"] == ticks[-1] and receipt["count"] == 25
                    and receipt["sha256"] == spec["sha256"] and receipt["bytes"] == spec["bytes"], "same E318 chunk receipt")
        chunks.append(observed)
    drives = np.concatenate([c["drive"] for c in chunks], axis=0)[:99].copy()
    rates = np.concatenate([c["rate"] for c in chunks], axis=0)[:99].copy()
    adaptations = np.concatenate([c["adaptation"] for c in chunks], axis=0)[:99].copy()
    control = selected(binding["controls"])
    recruitment = selected(binding["recruitment"])
    ticks = np.arange(301, dtype=np.int64)
    for values in (control, recruitment):
        ray.require(core.exact(values["control_tick"], ticks) and core.exact(values["network_tick"], ticks * 20), "E318 complete original boundary rows")
    ray.require(np.allclose(control["physics_time_s"], ticks * .005, rtol=0, atol=2e-8), "E318 original physical clock")
    ray.require(recruitment["complete"].all() and recruitment["bank_available"].all()
                and recruitment["original_activation_available"].all(), "complete decoder boundary observations")

    # Pin and hash the finite original whole roster before StateStore opens it.
    whole_files = binding["whole_files"]
    ray.require(set(whole_files) == {name for name in inventory["files"] if name.startswith(inventory["initial_whole"] + "/")}, "same original body control0 whole roster")
    for name, spec in whole_files.items():
        admission.check("science")
        p = ray.path(name, admission.root)
        ray.require(p.stat().st_size == spec["bytes"] and core.digest(p, admission.check) == spec["sha256"], "original control0 whole bytes")
        ledger.append({"path": name, "expected_sha256": spec["sha256"], "bytes": spec["bytes"]})
    admission.fence("pre_array")
    api = body.resolve_original_api()
    saved = api["StateStore"].load(ray.path(inventory["initial_whole"], admission.root), max_files=123)
    api["recovery"].checked_checkpoint(saved)
    ray.require(saved["control_tick"] == saved["network"]["tick"] == 0
                and saved["network"]["model_hash"] == inventory["source_network_identity"]
                and saved["adapter"]["hash"] == inventory["source_adapter_identity"]
                and saved["graph_hash"] == inventory["graph_hash"], "E318/body same actual control0 identities")
    # The neutral source's utilization/resources are original float64 values.
    selection = saved["network"]["recurrent_resource"]
    utilization = selection["utilization"].copy()
    resource = saved["network"]["synaptic_resource"].copy()
    ray.require(utilization.dtype == resource.dtype == np.float64 and utilization.shape == resource.shape == (7136,)
                and np.isfinite(utilization).all() and np.isfinite(resource).all()
                and np.all(utilization == 0) and np.all(resource == 1), "actual fixed neutral u0/resources")
    control0 = {"rate": saved["network"]["rate"].copy(), "adaptation": saved["network"]["adaptation"].copy(),
                "bank": saved["adapter"]["bank"].copy(), "original_activation": saved["adapter"]["original"]["activation"].copy(),
                "q": control["command"][0].copy(), "tendon": control["tendon"][0].copy(),
                "control_tick": np.asarray([0], np.int64), "network_tick": np.asarray([0], np.int64),
                "utilization": utilization, "resource": resource}
    ray.require(core.exact(control0["bank"], recruitment["bank"][0])
                and core.exact(control0["original_activation"], recruitment["original_activation"][0])
                and core.exact(control0["q"], np.asarray(saved["body"]["action"]["joint_angles"], np.float64))
                and core.exact(control0["tendon"], saved["adapter"]["original"]["original"]["original"]["tendon_activation"]), "E318 and original whole boundary0 equality")
    endpoint = {"q": control["command"][1:100].copy(), "tendon": control["tendon"][1:100].copy(),
                "rate": rates[-1].copy(), "adaptation": adaptations[-1].copy(),
                "bank": recruitment["bank"][99].copy(), "original_activation": recruitment["original_activation"][99].copy()}
    for name, dtype, shape in (("rate", np.float32, (158706,)), ("adaptation", np.float32, (158706,)),
                               ("bank", np.float64, (391, 3)), ("original_activation", np.float64, (96,))):
        for values in (control0, endpoint):
            ray.require(values[name].dtype == dtype and values[name].shape == shape and np.isfinite(values[name]).all(), "actual startup state " + name)
    ray.require(drives.dtype == np.float32 and drives.shape == (99, 158706) and np.isfinite(drives).all(), "actual E318 startup drives")
    startup_target = core.load_selected(admission.root, inventory["selected_archives"]["startup"], ledger,
                                        timely=admission.check, generated=False)
    for item in ledger:
        ray.require(core.digest(ray.path(item["path"], admission.root), admission.check) == item["expected_sha256"], "startup resolution posthash")
    return {"control0": control0, "drive": drives, "endpoint": endpoint, "target": startup_target}, {
        "schema": "flylab.m3.startup-ray.e318-resolution.v1", "controls": [1, 99], "initial_control": 0,
        "network_identity": inventory["source_network_identity"], "graph_hash": inventory["graph_hash"],
        "manifest": binding["manifest"], "result": binding["result"], "control0_original_whole_exact": True,
        "neutral99_reproduction": "REQUIRED_BY_RAY", "biology_proven": False}
