"""
env.py — LegacyCodeArcheologist
Core RL environment: reset, step, reward, sandbox lifecycle.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

from models import (
    Action, ActionType,
    CallAPI, EditCode, ReadFile, RunTest,
    Observation, State,
)
from task import TASK_REGISTRY
from grader import get_grader

# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------
R_STEP_PROGRESS     =  0.10   # small positive for any valid, non-trivial action
R_TASK_COMPLETE     =  1.00   # terminal success
R_PARTIAL_CREDIT    =  0.30   # grader partial (e.g., server started but wrong body)
R_INVALID_SYNTAX    = -0.05   # patch introduced a Python syntax error
R_INVALID_ACTION    = -0.02   # malformed action (caught by validate())
R_TIMEOUT_PENALTY   = -0.01   # per step when max_steps exceeded
R_SERVER_CRASH      = -0.10   # server died after a patch

LEGACY_TEMPLATES_DIR = Path(__file__).parent / "legacy_templates"
SERVER_STARTUP_WAIT  = 2.5    # seconds after launching uvicorn


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class LegacyCodeEnv:
    """
    OpenEnv-compatible environment for Software Archeology tasks.

    Lifecycle
    ---------
    env = LegacyCodeEnv(task_id="task_1_syntax_error")
    obs = env.reset()
    while not done:
        obs, reward, done, info = env.step(action)
    env.close()
    """

    metadata = {
        "name":    "legacy_code_archeologist",
        "version": "1.0.0",
        "tasks":   [
            {"id": task_id, "has_grader": True} 
            for task_id in TASK_REGISTRY.keys()
        ],
        "max_steps": 30,
    }

    def __init__(self, task_id: str = "task_1_syntax_error", seed: int = 42):
        if task_id not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task '{task_id}'. Valid: {list(TASK_REGISTRY.keys())}"
            )
        self.task_id  = task_id
        self.seed     = seed
        self._state: Optional[State] = None
        self._proc:  Optional[subprocess.Popen] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Restore sandbox to the broken legacy state and start the server."""
        self._teardown_server()
        sandbox = self._fresh_sandbox()
        self._state = State(
            task_id      = self.task_id,
            sandbox_root = str(sandbox),
            max_steps    = 30,
            server_port  = 8001,
        )
        self._deploy_legacy_files(sandbox)
        self._start_server(sandbox, self._state.server_port)

        # Initial observation: show the agent which files exist
        file_listing = self._list_files(sandbox)
        return Observation(
            file_content = file_listing,
            stdout       = f"[reset] Sandbox ready at {sandbox}\n"
                           f"Task: {TASK_REGISTRY[self.task_id]['description']}",
            step_count   = 0,
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        """Apply an action, compute reward, return (obs, reward, done, info)."""
        assert self._state is not None, "Call reset() before step()."

        self._state.step_count += 1
        t0 = time.perf_counter()

        # 1. Validate action
        try:
            action.validate()
        except ValueError as exc:
            obs = Observation(
                error      = f"[invalid_action] {exc}",
                step_count = self._state.step_count,
            )
            reward = R_INVALID_ACTION
            done   = self._check_done(reward)
            self._state.cumulative_reward += reward
            return obs, reward, done, self._info()

        # 2. Dispatch
        obs = self._dispatch(action, t0)

        # 3. Grade
        grader  = get_grader(self.task_id)
        result  = grader.grade(self._state, obs)
        reward  = result["reward"]
        done    = result["done"] or self._check_done(reward)

        self._state.cumulative_reward += reward
        self._state.done = done

        return obs, reward, done, {**self._info(), **result}

    def close(self) -> None:
        self._teardown_server()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Dispatch helpers
    # ------------------------------------------------------------------

    def _dispatch(self, action: Action, t0: float) -> Observation:
        atype = action.action_type
        if atype == ActionType.READ_FILE:
            return self._do_read_file(action, t0)         # type: ignore[arg-type]
        elif atype == ActionType.EDIT_CODE:
            return self._do_edit_code(action, t0)         # type: ignore[arg-type]
        elif atype == ActionType.RUN_TEST:
            return self._do_run_test(action, t0)          # type: ignore[arg-type]
        elif atype == ActionType.CALL_API:
            return self._do_call_api(action, t0)          # type: ignore[arg-type]
        else:
            return Observation(error=f"Unknown action type: {atype}")

    def _do_read_file(self, action: ReadFile, t0: float) -> Observation:
        target = Path(self._state.sandbox_root) / action.path
        try:
            content = target.read_text(encoding="utf-8")
            return Observation(
                file_content = content,
                latency      = time.perf_counter() - t0,
                step_count   = self._state.step_count,
                files_modified = list(self._state.files_modified),
            )
        except FileNotFoundError:
            return Observation(
                error      = f"File not found: {action.path}",
                latency    = time.perf_counter() - t0,
                step_count = self._state.step_count,
            )

    def _do_edit_code(self, action: EditCode, t0: float) -> Observation:
        target = Path(self._state.sandbox_root) / action.path
        if not target.exists():
            return Observation(
                error      = f"Cannot edit non-existent file: {action.path}",
                latency    = time.perf_counter() - t0,
                step_count = self._state.step_count,
            )

        # Write the new content (full replacement for simplicity;
        # patch-mode can be added via `patch` binary if needed)
        target.write_text(action.patch, encoding="utf-8")

        # Syntax check for Python files
        syntax_error = ""
        if target.suffix == ".py":
            syntax_error = self._check_python_syntax(action.patch)
            if syntax_error:
                return Observation(
                    error      = f"[syntax_error] {syntax_error}",
                    latency    = time.perf_counter() - t0,
                    step_count = self._state.step_count,
                )

        # Track modification
        if action.path not in self._state.files_modified:
            self._state.files_modified.append(action.path)

        # Hot-reload: restart server with new code
        self._teardown_server()
        self._start_server(
            Path(self._state.sandbox_root), self._state.server_port
        )

        return Observation(
            stdout         = f"[edit_ok] {action.path} written & server restarted.",
            latency        = time.perf_counter() - t0,
            step_count     = self._state.step_count,
            files_modified = list(self._state.files_modified),
        )

    def _do_run_test(self, action: RunTest, t0: float) -> Observation:
        try:
            result = subprocess.run(
                shlex.split(action.command),
                cwd     = self._state.sandbox_root,
                capture_output = True,
                text    = True,
                timeout = 30,
            )
            stdout = result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            stdout = "[timeout] Command exceeded 30 s."
        except Exception as exc:
            stdout = f"[error] {exc}"

        return Observation(
            stdout         = stdout,
            latency        = time.perf_counter() - t0,
            step_count     = self._state.step_count,
            files_modified = list(self._state.files_modified),
        )

    def _do_call_api(self, action: CallAPI, t0: float) -> Observation:
        try:
            resp = requests.request(
                method  = action.method,
                url     = action.url,
                headers = action.headers,
                json    = action.payload if action.payload else None,
                timeout = action.timeout,
            )
            latency = time.perf_counter() - t0
            try:
                body = resp.json()
            except Exception:
                body = {"raw": resp.text}

            return Observation(
                api_response   = body,
                status_code    = resp.status_code,
                latency        = latency,
                step_count     = self._state.step_count,
                files_modified = list(self._state.files_modified),
            )
        except requests.exceptions.ConnectionError:
            return Observation(
                error      = "[connection_error] Server not reachable.",
                latency    = time.perf_counter() - t0,
                step_count = self._state.step_count,
                status_code = 0,
            )
        except requests.exceptions.Timeout:
            return Observation(
                error      = "[timeout] API call exceeded timeout.",
                latency    = time.perf_counter() - t0,
                step_count = self._state.step_count,
                status_code = 0,
            )

    # ------------------------------------------------------------------
    # Sandbox management
    # ------------------------------------------------------------------

    def _fresh_sandbox(self) -> Path:
        tmp = tempfile.mkdtemp(prefix="lca_sandbox_")
        return Path(tmp)

    def _deploy_legacy_files(self, sandbox: Path) -> None:
        """Copy broken legacy templates into the fresh sandbox."""
        task_cfg = TASK_REGISTRY[self.task_id]
        templates_needed = task_cfg.get("template_files", [])
        for fname in templates_needed:
            src = LEGACY_TEMPLATES_DIR / fname
            if src.exists():
                shutil.copy(src, sandbox / fname)
            else:
                raise FileNotFoundError(
                    f"Missing template file: {src}. "
                    "Run scripts/generate_templates.py first."
                )
        # Always copy the shared README
        readme_src = LEGACY_TEMPLATES_DIR / "README.txt"
        if readme_src.exists():
            shutil.copy(readme_src, sandbox / "README.txt")

    def _start_server(self, sandbox: Path, port: int) -> None:
        """Launch the sandboxed FastAPI app with uvicorn."""
        main_py = sandbox / "main.py"
        if not main_py.exists():
            return  # nothing to start yet
        log_path = sandbox / "server.log"
        log_fh   = open(log_path, "w")
        self._proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "main:app",
                "--host", "127.0.0.1",
                "--port", str(port),
                "--log-level", "warning",
            ],
            cwd    = str(sandbox),
            stdout = log_fh,
            stderr = log_fh,
        )
        self._state.server_pid = self._proc.pid
        time.sleep(SERVER_STARTUP_WAIT)  # give uvicorn time to bind

    def _teardown_server(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                os.kill(self._proc.pid, signal.SIGTERM)
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        self._proc = None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _check_python_syntax(code: str) -> str:
        """Return error string or empty string on success."""
        try:
            compile(code, "<string>", "exec")
            return ""
        except SyntaxError as exc:
            return str(exc)

    @staticmethod
    def _list_files(sandbox: Path) -> str:
        lines = ["=== Sandbox file listing ==="]
        for p in sorted(sandbox.rglob("*")):
            if p.is_file():
                lines.append(f"  {p.relative_to(sandbox)}")
        return "\n".join(lines)

    def _check_done(self, reward: float) -> bool:
        if self._state.step_count >= self._state.max_steps:
            return True
        return self._state.done

    def _info(self) -> Dict[str, Any]:
        return {
            "task_id":           self.task_id,
            "step_count":        self._state.step_count,
            "cumulative_reward": self._state.cumulative_reward,
            "files_modified":    list(self._state.files_modified),
            "server_pid":        self._state.server_pid,
        }
