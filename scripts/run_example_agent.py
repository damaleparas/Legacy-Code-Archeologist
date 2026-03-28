"""
scripts/run_example_agent.py
A minimal rule-based agent that solves all three tasks sequentially.
Use this to verify the environment works end-to-end before training.

Run:  python scripts/run_example_agent.py
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import textwrap
from env import LegacyCodeEnv
from models import ReadFile, EditCode, CallAPI

FIXED_MAIN_PY = textwrap.dedent("""\
    import time
    from fastapi import FastAPI, Header, HTTPException
    from fastapi.responses import JSONResponse

    app = FastAPI(title="LegacyAPI", version="0.0.1")

    INTERNAL_TOKEN = "arch3olog1st-s3cr3t-2019"

    @app.get("/health")
    async def health():
        return {"status": "healthy", "version": "0.0.1"}

    @app.post("/process")
    async def process(
        x_internal_token: str = Header(None, alias="X-Internal-Token")
    ):
        if x_internal_token != INTERNAL_TOKEN:
            raise HTTPException(status_code=401, detail="Bad token")
        return JSONResponse({"status": "ok", "processed": True})

    @app.get("/compute")
    async def compute():
        # time.sleep(2) — removed by the archeologist agent
        return {"result": sum(i*i for i in range(10_000))}
""")


def run_task(task_id: str, actions) -> None:
    print(f"\n{'='*60}")
    print(f"  Running: {task_id}")
    print(f"{'='*60}")

    with LegacyCodeEnv(task_id=task_id) as env:
        obs = env.reset()
        print(f"[reset]\n{obs.stdout[:200]}")

        for action in actions:
            obs, reward, done, info = env.step(action)
            print(f"\n[step {info['step_count']}]  reward={reward:+.3f}  done={done}")
            if obs.error:
                print(f"  ERROR: {obs.error}")
            if obs.status_code:
                print(f"  HTTP {obs.status_code}  latency={obs.latency*1000:.1f} ms")
            if obs.api_response:
                print(f"  JSON: {obs.api_response}")
            if done:
                print(f"\n✓ Task complete! Cumulative reward: {info['cumulative_reward']:.3f}")
                break
        else:
            print(f"\n✗ Task not solved in allotted steps.")


if __name__ == "__main__":
    # ── Task 1: fix syntax error ──────────────────────────────────────────────
    run_task("task_1_syntax_error", [
        ReadFile(path="main.py"),
        EditCode(path="main.py", patch=FIXED_MAIN_PY),
        CallAPI(url="http://localhost:8001/health"),
    ])

    # ── Task 2: auth header discovery ────────────────────────────────────────
    run_task("task_2_auth_header", [
        ReadFile(path="README.txt"),
        EditCode(path="main.py", patch=FIXED_MAIN_PY),
        CallAPI(
            url     = "http://localhost:8001/process",
            method  = "POST",
            headers = {"X-Internal-Token": "arch3olog1st-s3cr3t-2019"},
        ),
    ])

    # ── Task 3: performance optimisation ─────────────────────────────────────
    run_task("task_3_perf_optimization", [
        ReadFile(path="main.py"),
        EditCode(path="main.py", patch=FIXED_MAIN_PY),
        CallAPI(url="http://localhost:8001/compute"),
    ])
