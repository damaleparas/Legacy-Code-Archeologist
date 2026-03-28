"""
server.py — OpenEnv HTTP wrapper for LegacyCodeArcheologist
Exposes /reset, /step, /close as JSON endpoints so the openenv-core
client (or any HTTP client) can drive the environment remotely.

Run:
    python server.py --port 5000 --task task_1_syntax_error
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

from env import LegacyCodeEnv
from models import ReadFile, EditCode, RunTest, CallAPI, ActionType

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("openenv.server")

# Global env instance (single-threaded server)
_env: LegacyCodeEnv | None = None


# ---------------------------------------------------------------------------
# Action deserialiser
# ---------------------------------------------------------------------------

def deserialise_action(data: dict):
    atype = data.get("action_type")
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
    raise ValueError(f"Unknown action_type: {atype}")


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class EnvHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        log.info(fmt % args)

    def _send_json(self, code: int, body: dict):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw    = self.rfile.read(length) if length else b"{}"
        return json.loads(raw)

    def do_POST(self):
        global _env
        path = urlparse(self.path).path

        if path == "/reset":
            body    = self._read_body()
            task_id = body.get("task_id", "task_1_syntax_error")
            try:
                if _env:
                    _env.close()
                _env = LegacyCodeEnv(task_id=task_id)
                obs  = _env.reset()
                self._send_json(200, {"observation": obs.to_dict()})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

        elif path == "/step":
            if _env is None:
                self._send_json(400, {"error": "Call /reset first."})
                return
            try:
                body   = self._read_body()
                action = deserialise_action(body)
                obs, reward, done, info = _env.step(action)
                self._send_json(200, {
                    "observation": obs.to_dict(),
                    "reward":      reward,
                    "done":        done,
                    "info":        info,
                })
            except Exception as exc:
                self._send_json(400, {"error": str(exc)})

        elif path == "/close":
            if _env:
                _env.close()
                _env = None
            self._send_json(200, {"status": "closed"})

        else:
            self._send_json(404, {"error": f"Unknown route: {path}"})

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/info":
            self._send_json(200, LegacyCodeEnv.metadata)
        elif path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "Not found"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OpenEnv server for LegacyCodeArcheologist")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), EnvHandler)
    log.info(f"OpenEnv server listening on {args.host}:{args.port}")
    log.info(f"Tasks: {list(LegacyCodeEnv.metadata['tasks'])}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down.")
        if _env:
            _env.close()
        sys.exit(0)


if __name__ == "__main__":
    main()
