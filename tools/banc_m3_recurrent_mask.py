"""Existing premotor VNC recurrence mask; no activity-dependent selection."""

import numpy as np


def recurrent_mask(graph, weights, motor_ids):
    """Select direct motor presynaptic VNC cells and their existing mutual edges.

    Positive efficacy is shared by post cell type, original edge sign and known
    left/right relationship. Unknown soma sides are never inferred as bilateral.
    """
    incoming = []
    for post in motor_ids:
        positions = np.arange(weights.indptr[post], weights.indptr[post + 1])
        incoming.extend(weights.indices[positions[weights.data[positions] != 0]])
    presynaptic = np.unique(incoming)
    population = np.array([i for i in presynaptic if
        graph.nodes[i].get('super_class') == 'ventral_nerve_cord_intrinsic'], dtype=np.int64)
    if np.intersect1d(population, motor_ids).size:
        raise ValueError('Premotor intrinsic population intersects motor roster')
    membership = np.zeros(weights.shape[0], dtype=bool); membership[population] = True
    positions = []; groups = []; keys = []; index = {}
    for post in population:
        row = np.arange(weights.indptr[post], weights.indptr[post + 1])
        row = row[membership[weights.indices[row]] & (weights.data[row] != 0)]
        pnode = graph.nodes[post]; post_side = pnode.get('soma_side')
        for position in row:
            pre = weights.indices[position]; pre_side = graph.nodes[pre].get('soma_side')
            relation = ('same' if post_side == pre_side else 'different') if (
                post_side in ('left', 'right') and pre_side in ('left', 'right')) else 'unknown'
            key = (pnode.get('cell_type') or pnode['id'], int(np.sign(weights.data[position])), relation)
            if key not in index:
                index[key] = len(keys); keys.append(key)
            positions.append(position); groups.append(index[key])
    return population, np.asarray(positions, np.int64), np.asarray(groups, np.int64), keys
