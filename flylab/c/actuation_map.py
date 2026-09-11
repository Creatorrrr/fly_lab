"""Inspectable coverage of the instantiated neural-to-position adapter.

An anatomical motor roster is not a joint binding. Only nonzero vectors in
the active adapter count as mappings, and none imply validated muscle forces.
"""

import numpy as np


def describe_actuation(loop):
    body = loop.body
    whole = getattr(body, "whole_body_control", None)
    leg_names = list(body.joint_names)
    if whole is None:
        names = leg_names.copy()
        leg_indices = np.arange(len(leg_names))
        regions = [None] * len(names)
        passive_count = None
    else:
        from flylab.whole_body import region

        names = list(whole.names)
        leg_indices = np.asarray(body.leg_action_indices)
        regions = [region(dof) for dof in body.full_order]
        passive_count = len(whole.passive_ids)
    if (
        len(names) != len(set(names))
        or len(regions) != len(names)
        or leg_indices.shape != (len(leg_names),)
        or leg_indices.dtype.kind not in "iu"
        or len(set(leg_indices.tolist())) != len(leg_indices)
        or (leg_indices < 0).any()
        or (leg_indices >= len(names)).any()
    ):
        raise ValueError("Ambiguous physical actuation order")
    if whole is not None and any(
        not leg_names[i].endswith("/" + names[int(j)])
        for i, j in enumerate(leg_indices)
    ):
        raise ValueError("Whole-body/leg joint identities disagree")
    positive = [set() for _ in names]
    negative = [set() for _ in names]
    for row, joints, groups in zip(loop.spec["rows"], loop.joints, loop.muscles):
        for axis, leg_index in enumerate(joints):
            index = int(leg_indices[leg_index])
            regions[index] = row["leg"]
            for _, motors, vector in groups:
                if vector[axis] > 0:
                    positive[index].update(map(int, motors))
                elif vector[axis] < 0:
                    negative[index].update(map(int, motors))
    axes = [
        {
            "name": name,
            "region": regions[i],
            "status": "engineering_map" if positive[i] or negative[i] else "unmapped",
            "motor_neurons": len(positive[i] | negative[i]),
            "positive_motor_neurons": len(positive[i]),
            "negative_motor_neurons": len(negative[i]),
        }
        for i, name in enumerate(names)
    ]
    mapped_motors = set().union(*positive, *negative)
    all_motors = {
        i for i, node in enumerate(loop.graph.nodes) if node["super_class"] == "motor"
    }
    if (
        mapped_motors != set(map(int, loop.motor_indices))
        or not mapped_motors <= all_motors
    ):
        raise ValueError("Actuator motor roster differs from neural decoder")
    mapped_count = sum(row["status"] == "engineering_map" for row in axes)
    if mapped_count != loop.spec["covered_dofs"]:
        raise ValueError("Actuator coverage differs from neural decoder")
    return {
        "schema": "flylab.actuation-map.v1",
        "graph_hash": loop.graph.hash,
        "adapter_hash": loop.hash,
        "body_model_hash": getattr(body, "model_hash", None),
        "active_axes": len(axes),
        "mapped_axes": mapped_count,
        "unmapped_axes": len(axes) - mapped_count,
        "passive_joints": passive_count,
        "graph_motor_neurons": len(all_motors),
        "mapped_motor_neurons": len(mapped_motors),
        "unmapped_motor_neurons": len(all_motors - mapped_motors),
        "biological_validation": False,
        "mapping_kind": "signed_rate_to_position_engineering_proxy",
        "axes": axes,
    }
