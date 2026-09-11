"""Check the real C-engine autonomous path and preserve a live pre-update state."""

import argparse
import asyncio
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flylab.autonomy import AUTONOMOUS_MODE
from flylab.c import PROTOCOL
from flylab.c.engine import CEngine
from flylab.c.graph import GraphStore
from flylab.c.integrity import read_json, write_json
from flylab.c.ports import PortBindings
from flylab.c.storage import StateStore


async def preserve_live(port, out):
    async with aiohttp.ClientSession() as client:
        async with client.get(f"http://127.0.0.1:{port}/api/bootstrap") as response:
            bootstrap = await response.json()
        async with client.ws_connect(
            f"http://127.0.0.1:{port}/ws", origin=f"http://127.0.0.1:{port}"
        ) as socket:
            await socket.send_json({"protocol": PROTOCOL, "token": bootstrap["token"]})
            authenticated = await socket.receive_json(timeout=30)
            if not authenticated.get("authenticated"):
                raise RuntimeError("Could not authenticate local session")
            result = {}
            for rid, op in enumerate(("frame", "checkpoint"), start=1):
                await socket.send_json(
                    {"protocol": PROTOCOL, "requestId": rid, "op": op, "payload": {}}
                )
                while True:
                    message = await socket.receive(timeout=60)
                    if message.type == aiohttp.WSMsgType.BINARY:
                        continue
                    data = message.json()
                    if data.get("requestId") != rid:
                        continue
                    if not data.get("ok"):
                        raise RuntimeError(data.get("error"))
                    result[op] = data["result"]
                    break
            out.mkdir(parents=True, exist_ok=False)
            write_json(out / "preserved.json", result)
            print(
                {
                    "simTime": result["frame"]["simTime"],
                    "mode": result["frame"]["mode"],
                    "checkpoint": result["checkpoint"],
                },
                flush=True,
            )


def verify_engine(args):
    args.out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    graph = GraphStore.load(args.graph)
    binding = PortBindings(graph, read_json(args.bindings))
    engine = CEngine(graph, binding, mode=AUTONOMOUS_MODE)
    restored = None
    result = {"neural_control": False, "biological_validation": False}
    try:
        write_json(
            args.out / "spec.json",
            {
                "mode": AUTONOMOUS_MODE,
                "provenance": engine.provenance(),
                "checkpoint_future_tolerance": 1e-9,
            },
        )
        frame = engine.step(80)
        assert frame["scope"]["simulatedNodes"] == 0 and frame["neural"] is None
        assert not frame["neuromuscular"]["enabled"]
        assert frame["command"]["command_source"] == "flygym_sensory_policy"
        assert frame["command"]["u_final"]["forwardSpeed"] > 0
        assert frame["body"]["travel"] > 1
        saved = engine.checkpoint()
        StateStore.save(args.out / "checkpoint", saved)
        restored = CEngine.from_checkpoint(
            graph, binding, StateStore.load(args.out / "checkpoint")
        )
        engine.step(40)
        restored.step(40)
        errors = {
            name: float(
                np.max(
                    np.abs(
                        getattr(engine.body.d, name) - getattr(restored.body.d, name)
                    )
                )
            )
            for name in ("qpos", "qvel", "ctrl", "qacc_warmstart")
        }
        assert max(errors.values()) <= 1e-9, errors
        assert engine.autonomy.snapshot() == restored.autonomy.snapshot()
        result["checkpoint_future_errors"] = errors
        engine.command("autonomy", {"enabled": False})
        stopped = engine.step(300)
        assert stopped["command"]["u_final"]["forwardSpeed"] == 0
        assert abs(stopped["body"]["speed"]) < 0.1
        result["stopped_speed_mm_s"] = stopped["body"]["speed"]
        engine.command("autonomy", {"enabled": True})
        resumed = engine.step(5)
        assert resumed["command"]["u_final"]["forwardSpeed"] > 0
        engine.schedule({"kind": "motor_disconnect", "duration_controls": 20})
        assert engine.step(1)["command"]["command_source"] == "motor_disconnect"
        assert engine.last_command["u_final"]["forwardSpeed"] == 0
        assert engine.step(20)["command"]["command_source"] == "flygym_sensory_policy"
        engine.schedule(
            {"kind": "sensor_off", "channels": ["odor_mean"], "duration_controls": 20}
        )
        assert engine.step(1)["autonomy"]["odor"] == [0.0, 0.0]
        frame = engine.step(20)
        assert max(frame["autonomy"]["odor"]) > 0
        result.update(
            status="PASS",
            physical=True,
            frame=frame,
            wall_s=time.perf_counter() - began,
        )
        write_json(args.out / "report.json", result)
        print(
            {key: value for key, value in result.items() if key != "frame"}, flush=True
        )
    finally:
        engine.close()
        if restored:
            restored.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--capture-live", type=int)
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("data/acquisitions/banc888-windows-20260910/bundle"),
    )
    parser.add_argument(
        "--bindings",
        type=Path,
        default=Path(
            "data/acquisitions/banc888-windows-20260910/bindings-neuromuscular-v2.json"
        ),
    )
    args = parser.parse_args()
    if args.capture_live:
        asyncio.run(preserve_live(args.capture_live, args.out))
    else:
        verify_engine(args)
