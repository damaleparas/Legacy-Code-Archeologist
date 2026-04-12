#!/usr/bin/env python3
"""
scripts/generate_templates.py
Regenerate the legacy_templates/ folder from scratch.
Run this once before starting training:  python scripts/generate_templates.py
"""

import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
TEMPLATES = ROOT / "legacy_templates"
TEMPLATES.mkdir(exist_ok=True)

MAIN_PY = '''\
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
'''

README_TXT = '''\
======================================================
  LEGACY API — INTERNAL DOCUMENTATION
======================================================
POST /process requires header:
    X-Internal-Token: arch3olog1st-s3cr3t-2019
GET /compute has a time.sleep() bottleneck (JIRA-4892).
GET /health has a SyntaxError (JIRA-4891).
======================================================
'''

(TEMPLATES / "main.py").write_text(MAIN_PY)
(TEMPLATES / "README.txt").write_text(README_TXT)

# Create app.db for Task 4
import sqlite3
db_path = TEMPLATES / "app.db"
if db_path.exists():
    db_path.unlink()
conn = sqlite3.connect(db_path)
curr = conn.cursor()
curr.execute("CREATE TABLE users (user_id INTEGER PRIMARY KEY, name TEXT)")
curr.execute("INSERT INTO users (user_id, name) VALUES (1, 'Archeologist')")
conn.commit()
conn.close()

print(f"DONE: Templates (including app.db) written to {TEMPLATES}")
