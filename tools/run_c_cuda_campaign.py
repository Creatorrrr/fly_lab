#!/usr/bin/env python3
"""Run/resume model-grouped CUDA jobs without dropping jobs on allocation failure."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.cuda_campaign import Campaign
from flylab.c.graph import GraphStore
from flylab.c.ports import PortBindings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("graph", "bindings", "spec", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-controls", type=int)
    p.add_argument("--cancel", nargs="*", default=[])
    a = p.parse_args()
    g = GraphStore.load(a.graph)
    bindings = PortBindings(g, json.loads(a.bindings.read_text(encoding="utf-8")))
    campaign = Campaign(
        g,
        bindings,
        json.loads(a.spec.read_text(encoding="utf-8")),
        a.out,
        resume=a.resume,
    )
    if a.cancel:
        campaign.cancel(a.cancel)
    result = campaign.run(max_controls=a.max_controls)
    print(
        json.dumps(
            dict(
                status=result["status"],
                jobs=result["jobs"],
                allocation_attempts=result["allocation_attempts"],
                waiting_for_memory=result.get("waiting_for_memory"),
                capacity=campaign.limit,
            )
        )
    )


if __name__ == "__main__":
    main()
