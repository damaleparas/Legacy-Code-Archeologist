# legacy_templates/main.py — BROKEN LEGACY CODE
import time
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="LegacyAPI", version="0.0.1")

INTERNAL_TOKEN = "arch3olog1st-s3cr3t-2019"

@app.get("/health")
async def health()                          # SyntaxError: missing ':'
    return {"status": "healthy"}

@app.post("/process")
async def process(
    x_internal_token: str = Header(None, alias="X-Internal-Token")
):
    if x_internal_token != INTERNAL_TOKEN:
        raise HTTPException(status_code=401, detail="Bad token")
    return JSONResponse({"status": "ok", "processed": True})

@app.get("/compute")
async def compute():
    time.sleep(2)          # BOTTLENECK
    return {"result": sum(i*i for i in range(10_000))}
