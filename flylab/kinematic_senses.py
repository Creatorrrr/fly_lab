"""Joint receptor geometry independent of model-specific hinge zero/sign."""

import numpy as np


def segment_flexion(points, velocities):
    """Flexion is pi minus the interior angle; positive velocity is flexing."""
    p, v = np.asarray(points, float), np.asarray(velocities, float)
    if (
        p.shape != (3, 3)
        or v.shape != (3, 3)
        or not np.isfinite(p).all()
        or not np.isfinite(v).all()
    ):
        raise ValueError("Three finite physical joint points and velocities required")
    a, b = p[0] - p[1], p[2] - p[1]
    va, vb = v[0] - v[1], v[2] - v[1]
    la, lb = np.linalg.norm(a), np.linalg.norm(b)
    if min(la, lb) < 1e-9:
        raise ValueError("Degenerate leg segments")
    cosine = float(np.clip(a @ b / (la * lb), -1.0, 1.0))
    sine = np.sqrt(max(0.0, 1 - cosine * cosine))
    if sine < 1e-6:
        raise ValueError("Singular straight/folded knee geometry")
    derivative = (va @ b + a @ vb) / (la * lb) - cosine * (
        a @ va / la**2 + b @ vb / lb**2
    )
    return float(np.pi - np.arccos(cosine)), float(derivative / sine)
