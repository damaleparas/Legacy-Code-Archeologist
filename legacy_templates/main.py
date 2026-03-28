# legacy_templates/main.py
# ⚠️  LEGACY CODE — DO NOT EDIT (this is the broken template)
# Vintage: 2019. Author: unknown. Last touched: never.
#
# This file is intentionally broken. It contains:
#   Task 1: SyntaxError  — missing colon on a function definition
#   Task 2: Hidden auth  — /process requires X-Internal-Token header
#   Task 3: Bottleneck   — time.sleep(2) inside /compute

import time
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="LegacyAPI", version="0.0.1")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1:  SyntaxError — the colon after 'def health()' is missing
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health()                          # ← SyntaxError: missing ':'
    return {"status": "healthy", "version": "0.0.1"}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2:  Hidden auth header
# ─────────────────────────────────────────────────────────────────────────────
INTERNAL_TOKEN = "arch3olog1st-s3cr3t-2019"

@app.post("/process")
async def process(
    x_internal_token: str = Header(None, alias="X-Internal-Token")
):
    if x_internal_token != INTERNAL_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid X-Internal-Token header. Read the README."
        )
    return JSONResponse({"status": "ok", "processed": True})


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3:  Performance bottleneck
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/compute")
async def compute():
    time.sleep(2)          # TODO: remove this — leftover from integration tests
    result = sum(i * i for i in range(10_000))
    return {"result": result}


# ─────────────────────────────────────────────────────────────────────────────
# Misc legacy cruft (agent does not need to touch these)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/legacy/ping")
async def ping():
    return "pong"


@app.get("/legacy/status")
async def legacy_status():
    # This endpoint intentionally does nothing useful
    return {"deprecated": True, "message": "Use /health instead"}
