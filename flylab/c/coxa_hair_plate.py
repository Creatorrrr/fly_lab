"""Explicit front-leg anterior-limit receptor hypothesis on source BANC cells.

The nerve/organ correspondence is anatomical; the one-dimensional tuning and
input scale are engineering choices, not reconstructed receptor physiology.
"""

import numpy as np

SOURCE = "https://www.nature.com/articles/s41467-026-69333-z"
INPUT_GAIN = 144.0
ANTERIOR_RANGE_RAD = 0.5
ROOTS = (
    ("720575941432016009", "720575941442197595", "720575941491974604",
     "720575941532268747", "720575941563170984", "720575941577076072",
     "720575941618732757", "720575941691438299"),
    ("720575941411400815", "720575941437256492", "720575941470187895",
     "720575941487662989", "720575941525329410", "720575941526678093",
     "720575941556634625", "720575941605475774"),
)


def _tokens(value):
    return set(value.split(",")) if isinstance(value, str) else set()


def front_coxa_ports(graph):
    """Keep exact source identities and cell types; reject conflicting anatomy."""
    rows = {}
    for leg, side, roots in zip((0, 3), ("left", "right"), ROOTS):
        groups = {}
        for i in graph.resolve(["flywire:banc:888:" + root for root in roots]):
            node = graph.nodes[i]
            annotation = node.get("source_annotations", {})
            if (
                node.get("super_class") != "sensory"
                or node.get("class") != "hair_plate_neuron"
                or node.get("soma_side") != side
                or node.get("nt_type") != "ACH"
                or not node.get("cell_type")
                or _tokens(annotation.get("body_part_sensory")) != {"coxa", "front_leg"}
                or _tokens(annotation.get("peripheral_target_type")) != {"CoHP8", "hair_plate"}
                or annotation.get("nerve") != side + "_ventral_prothoracic_nerve"
            ):
                raise ValueError("Front coxa hair plate anatomy mismatch")
            groups.setdefault(node["cell_type"], []).append(node["id"])
        rows[leg] = [dict(kind="coxa_limit", cell_type=kind, ids=ids)
                     for kind, ids in sorted(groups.items())]
    return rows


def anterior_feature(q, neutral, sign):
    """Measured anterior yaw from neutral, bounded by the decoder's .5rad range."""
    return np.clip(sign * (q - neutral) / ANTERIOR_RANGE_RAD, 0.0, 1.0)
