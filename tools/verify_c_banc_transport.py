"""Matched loopback RPC timing for PNG responses versus measured-state streams.

Use a separate idle local test server. Restores the same named checkpoint for
every arm, verifies full saved futures, and leaves that checkpoint restored.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.integrity import file_hash, write_json
from flylab.c.storage import StateStore
from tools.verify_c_banc_navigation import physical_state_hash


async def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(
        args.out / "spec.json",
        {
            "source_hash": file_hash(Path(__file__)),
            "port": args.port,
            "checkpoint": args.checkpoint,
            "pairs": 5,
            "model_seconds": 0.05,
            "scope": "Same current model/source, stream=false PNG versus stream=true 5ms states. Round-trip includes request, server compute/render/serialization, response and client JSON parsing. No browser drawing.",
            "judgment": "Report latency and bytes separately; full futures must match, no performance threshold or neural speedup inferred from removing PNG.",
            "excluded": "restore, warmup and checkpoint save/load",
        },
    )
    base = f"http://127.0.0.1:{args.port}"
    reference = None
    results = []
    async with aiohttp.ClientSession() as client:
        async with client.get(base + "/api/bootstrap") as reply:
            boot = await reply.json()
        async with client.ws_connect(
            base + "/ws", origin=base, max_msg_size=64 * 1024 * 1024
        ) as socket:
            await socket.send_json(
                {"protocol": boot["protocol"], "token": boot["token"]}
            )
            hello = await socket.receive_json()
            if not hello.get("authenticated"):
                raise RuntimeError("Local handshake failed")
            sequence = 0

            async def rpc(op, payload=None):
                nonlocal sequence
                sequence += 1
                start = time.perf_counter()
                await socket.send_json(
                    {
                        "protocol": boot["protocol"],
                        "requestId": sequence,
                        "op": op,
                        "payload": payload or {},
                    }
                )
                raw = await socket.receive()
                if raw.type != aiohttp.WSMsgType.TEXT:
                    raise RuntimeError("Expected BANC JSON response")
                parsed = json.loads(raw.data)
                elapsed = time.perf_counter() - start
                if parsed.get("requestId") != sequence or not parsed.get("ok"):
                    raise RuntimeError(str(parsed.get("error", "Unexpected response")))
                return parsed["result"], elapsed, len(raw.data.encode("utf-8"))

            try:
                for pair in range(5):
                    for stream in [False, True] if pair % 2 == 0 else [True, False]:
                        await rpc("banc_walking_restore", {"name": args.checkpoint})
                        await rpc(
                            "banc_walking_advance", {"steps": 10, "stream": stream}
                        )
                        await rpc("banc_walking_restore", {"name": args.checkpoint})
                        result, elapsed, count = await rpc(
                            "banc_walking_advance", {"steps": 10, "stream": stream}
                        )
                        if result.get("render_error") or result["observation"].get(
                            "trace_error"
                        ):
                            raise RuntimeError("Response observation or render failed")
                        checkpoint, _, _ = await rpc("banc_walking_save")
                        state = StateStore.load(
                            args.artifacts
                            / "banc-walking-checkpoints"
                            / checkpoint["name"]
                        )
                        identity = physical_state_hash(state)
                        if reference is None:
                            reference = identity
                        if identity != reference:
                            raise AssertionError(
                                "Response modes changed the full model future"
                            )
                        row = {
                            "pair": pair,
                            "stream": stream,
                            "wall_seconds": elapsed,
                            "response_bytes": count,
                            "measured_samples": len(
                                result.get("trace", {}).get("samples", [])
                            ),
                            "png_present": bool(result.get("image")),
                            "full_future_bit_exact": True,
                        }
                        results.append(row)
                        write_json(args.out / "runs.json", results)
                        print(json.dumps(row), flush=True)
            finally:
                await rpc("banc_walking_restore", {"name": args.checkpoint})
    summary = {"validity": "valid", "runs": results, "medians": {}}
    for stream in (False, True):
        selected = [r for r in results if r["stream"] == stream]
        summary["medians"]["measured_states" if stream else "png"] = {
            key: statistics.median(row[key] for row in selected)
            for key in ("wall_seconds", "response_bytes")
        }
    write_json(args.out / "result.json", summary)
    print(json.dumps(summary["medians"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8771)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    asyncio.run(run(p.parse_args()))
