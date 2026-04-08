import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import sys
import os

# Add parent directory to sys.path to import env and models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import LegacyCodeEnv
from models import ActionType, ReadFile, EditCode, RunTest, CallAPI

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("openenv.server")

app = FastAPI(title="LegacyCodeArcheologist OpenEnv Server")

# Global env instance
_env = None

def deserialise_action(data: dict):
    atype = data.get("action_type")
    try:
        if atype == ActionType.READ_FILE:
            return ReadFile(path=data["path"])
        elif atype == ActionType.EDIT_CODE:
            return EditCode(path=data["path"], patch=data["patch"])
        elif atype == ActionType.RUN_TEST:
            return RunTest(command=data["command"])
        elif atype == ActionType.CALL_API:
            return CallAPI(
                url     = data.get("url", "http://localhost:8001/"),
                method  = data.get("method", "GET"),
                headers = data.get("headers", {}),
                payload = data.get("payload", {}),
            )
    except KeyError as e:
        raise ValueError(f"Missing required field for action {atype}: {e}")
    raise ValueError(f"Unknown action_type: {atype}")

@app.post("/reset")
async def reset(request: Request):
    global _env
    body = await request.json()
    task_id = body.get("task_id", "task_1_syntax_error")
    try:
        if _env:
            _env.close()
        _env = LegacyCodeEnv(task_id=task_id)
        obs = _env.reset()
        return {"observation": obs.to_dict()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/step")
async def step(request: Request):
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    
    body = await request.json()
    try:
        action = deserialise_action(body)
        obs, reward, done, info = _env.step(action)
        return {
            "observation": obs.to_dict(),
            "reward":      reward,
            "done":        done,
            "info":        info,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/close")
async def close():
    global _env
    if _env:
        _env.close()
        _env = None
    return {"status": "closed"}

@app.get("/info")
async def info():
    return LegacyCodeEnv.metadata

@app.get("/health")
async def health():
    return {"status": "ok"}

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()

