"""
grader.py — LegacyCodeArcheologist
Deterministic reward graders for each task.
"""

from __future__ import annotations

import abc
import re
from typing import Any, Dict

from models import Observation, State
from task import TASK_REGISTRY

# ---------------------------------------------------------------------------
# Reward constants (shared with env.py — kept here for grader autonomy)
# ---------------------------------------------------------------------------
R_PARTICIPATION   =  0.01
R_TASK_COMPLETE   =  0.49
R_PARTIAL_CREDIT  =  0.20
R_STEP_PROGRESS   =  0.10
R_INVALID_ACTION  = -0.01
R_INVALID_SYNTAX  = -0.05



# ---------------------------------------------------------------------------
# Base grader
# ---------------------------------------------------------------------------

class BaseGrader(abc.ABC):
    def __init__(self, task_id: str):
        self.task_id  = task_id
        self.task_cfg = TASK_REGISTRY[task_id]

    @abc.abstractmethod
    def grade(self, state: State, obs: Observation) -> Dict[str, Any]:
        """
        Returns dict with at minimum:
            reward  : float
            done    : bool
            message : str  (human-readable explanation)
        """

    def _participation_bonus(self, state: State) -> float:
        """One-time participation reward to ensure score > 0."""
        if not self._flagged(state, "participated"):
            self._flag(state, "participated")
            return R_PARTICIPATION
        return 0.0

    # Convenience: store intermediate flags in state.grader_state
    def _flag(self, state: State, key: str, value: Any = True) -> None:
        state.grader_state[key] = value

    def _flagged(self, state: State, key: str) -> bool:
        return bool(state.grader_state.get(key, False))


# ---------------------------------------------------------------------------
# Task 1 — SyntaxError fix
# ---------------------------------------------------------------------------

class Task1Grader(BaseGrader):
    """
    Reward schedule:
        +0.01  Participation (first step)
        +0.05  first ReadFile on main.py
        +0.10  any EditCode on main.py
        +0.10  server survived restart (no error in obs)
        +0.73  CallAPI → /health → HTTP 200  (terminal)
        Total: 0.99
    """

    def grade(self, state: State, obs: Observation) -> Dict[str, Any]:
        reward  = self._participation_bonus(state)
        done    = False
        message = ""

        # Penalise syntax errors from edits
        if obs.error and "[syntax_error]" in obs.error:
            reward += R_INVALID_SYNTAX
            message = "Syntax error introduced — penalty applied."
            return {"reward": reward, "done": False, "message": message}

        # Progress: first file read
        if obs.file_content and not self._flagged(state, "file_read"):
            reward += 0.05
            self._flag(state, "file_read")
            message += " +0.05 file read."

        # Progress: edit applied without crash
        if "main.py" in state.files_modified and not self._flagged(state, "edit_applied"):
            reward += 0.10
            self._flag(state, "edit_applied")
            message += " +0.10 edit applied."

        # Progress: server alive after edit (no connection error)
        if (
            obs.status_code is not None
            and obs.status_code != 0
            and not self._flagged(state, "server_alive")
        ):
            reward += 0.10
            self._flag(state, "server_alive")
            message += " +0.10 server reachable."

        # Terminal: /health → 200
        if obs.status_code == 200 and not self._flagged(state, "task_done"):
            # Adjusted to hit exactly 0.99 total
            reward += 0.73
            done    = True
            self._flag(state, "task_done")
            message += " *** TASK COMPLETE ***"

        return {"reward": reward, "done": done, "message": message.strip()}


# ---------------------------------------------------------------------------
# Task 2 — Auth header discovery
# ---------------------------------------------------------------------------

class Task2Grader(BaseGrader):
    """
    Reward schedule:
        +0.01  Participation
        +0.10  README.txt read
        +0.20  correct header sent
        +0.20  HTTP 200 received
        +0.48  JSON body contains {"status": "ok"}  (terminal)
        Total: 0.99
    """

    EXPECTED_HEADER = "X-Internal-Token"
    EXPECTED_JSON   = {"status": "ok"}

    def grade(self, state: State, obs: Observation) -> Dict[str, Any]:
        reward  = self._participation_bonus(state)
        done    = False
        message = ""

        # Progress: README read
        if (
            obs.file_content
            and "X-Internal-Token" in obs.file_content
            and not self._flagged(state, "readme_read")
        ):
            reward += 0.10
            self._flag(state, "readme_read")
            message += " +0.10 README read."

        # Progress: header present in last API call
        if obs.api_response is not None:
            headers_sent = state.grader_state.get("last_request_headers", {})
            if self.EXPECTED_HEADER in headers_sent and not self._flagged(state, "header_sent"):
                reward += 0.20
                self._flag(state, "header_sent")
                message += " +0.20 correct header sent."

        # Progress: HTTP 200
        if obs.status_code == 200 and not self._flagged(state, "status_200"):
            reward += 0.20
            self._flag(state, "status_200")
            message += " +0.20 HTTP 200."

        # Terminal: JSON validated
        if (
            obs.api_response
            and obs.api_response.get("status") == "ok"
            and not self._flagged(state, "task_done")
        ):
            reward += 0.48
            done    = True
            self._flag(state, "task_done")
            message += " *** TASK COMPLETE ***"

        return {"reward": reward, "done": done, "message": message.strip()}


# ---------------------------------------------------------------------------
# Task 3 — Performance optimisation
# ---------------------------------------------------------------------------

class Task3Grader(BaseGrader):
    """
    Reward schedule:
        +0.01  Participation
        +0.10  bottleneck found
        +0.20  sleep removed
        +0.20  latency < 500 ms
        +0.48  latency < 100 ms (+ bonus)
        Total: 0.99
    """

    LATENCY_FULL_MS    = 100
    LATENCY_PARTIAL_MS = 500

    def grade(self, state: State, obs: Observation) -> Dict[str, Any]:
        reward  = self._participation_bonus(state)
        done    = False
        message = ""

        # Progress: agent found the bottleneck
        if (
            obs.file_content
            and "time.sleep" in obs.file_content
            and not self._flagged(state, "bottleneck_found")
        ):
            reward += 0.10
            self._flag(state, "bottleneck_found")
            message += " +0.10 bottleneck located."

        # Progress: sleep removed
        if (
            "main.py" in state.files_modified
            and not self._flagged(state, "sleep_removed")
        ):
            import os
            fpath = os.path.join(state.sandbox_root, "main.py")
            try:
                code = open(fpath).read()
                if "time.sleep" not in code:
                    reward += 0.20
                    self._flag(state, "sleep_removed")
                    message += " +0.20 sleep removed."
            except OSError:
                pass

        # Performance scoring
        if obs.latency > 0 and obs.status_code and obs.status_code == 200:
            latency_ms = obs.latency * 1000
            if latency_ms < self.LATENCY_FULL_MS:
                if not self._flagged(state, "task_done"):
                    continuous = max(0.0, 1.0 - latency_ms / self.LATENCY_FULL_MS)
                    # Base 0.28 + up to 0.20 bonus = 0.48
                    reward += 0.28 + (continuous * 0.20)
                    done    = True
                    self._flag(state, "task_done")
                    message += f" *** TASK COMPLETE (latency={latency_ms:.1f} ms) ***"
            elif latency_ms < self.LATENCY_PARTIAL_MS:
                if not self._flagged(state, "partial_latency"):
                    reward += 0.20
                    self._flag(state, "partial_latency")
                    message += f" +0.20 partial (latency={latency_ms:.1f} ms)."

        return {"reward": reward, "done": done, "message": message.strip()}


# ---------------------------------------------------------------------------
# Task 4 — DB Schema mismatch
# ---------------------------------------------------------------------------

class Task4Grader(BaseGrader):
    """
    Reward schedule:
        +0.01  Participation
        +0.10  schema read
        +0.40  query fixed (edit applied)
        +0.48  status 200 OK
        Total: 0.99
    """

    def grade(self, state: State, obs: Observation) -> Dict[str, Any]:
        reward  = self._participation_bonus(state)
        done    = False
        message = ""

        # Progress: schema read
        if obs.stdout and "CREATE TABLE" in obs.stdout and not self._flagged(state, "schema_read"):
            reward += 0.10
            self._flag(state, "schema_read")
            message += " +0.10 schema inspected."

        # Progress: edit applied
        if "main.py" in state.files_modified and not self._flagged(state, "edit_applied"):
            reward += 0.40
            self._flag(state, "edit_applied")
            message += " +0.40 edit applied."

        # Terminal: success
        if obs.status_code == 200 and not self._flagged(state, "task_done"):
            reward += 0.48
            done    = True
            self._flag(state, "task_done")
            message += " *** TASK COMPLETE ***"

        return {"reward": reward, "done": done, "message": message.strip()}


# ---------------------------------------------------------------------------
# Task 5 — Env var leak
# ---------------------------------------------------------------------------

class Task5Grader(BaseGrader):
    """
    Reward schedule:
        +0.01  Participation
        +0.10  endpoint checked
        +0.40  mask applied (edit)
        +0.48  JSON body correct
        Total: 0.99
    """

    def grade(self, state: State, obs: Observation) -> Dict[str, Any]:
        reward  = self._participation_bonus(state)
        done    = False
        message = ""

        # Progress: endpoint checked
        if obs.api_response and "DB_PASSWORD" in obs.api_response and not self._flagged(state, "endpoint_checked"):
            reward += 0.10
            self._flag(state, "endpoint_checked")
            message += " +0.10 endpoint verified."

        # Progress: edit applied
        if "main.py" in state.files_modified and not self._flagged(state, "edit_applied"):
            reward += 0.40
            self._flag(state, "edit_applied")
            message += " +0.40 mask implemented."

        # Terminal: success
        if (
            obs.api_response
            and obs.api_response.get("DB_PASSWORD") == "********"
            and not self._flagged(state, "task_done")
        ):
            reward += 0.48
            done    = True
            self._flag(state, "task_done")
            message += " *** TASK COMPLETE ***"

        return {"reward": reward, "done": done, "message": message.strip()}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_GRADER_MAP = {
    "task_1_syntax_error":      Task1Grader,
    "task_2_auth_header":       Task2Grader,
    "task_3_perf_optimization": Task3Grader,
    "task_4_db_schema_mismatch": Task4Grader,
    "task_5_env_var_leak":       Task5Grader,
}


def get_grader(task_id: str) -> BaseGrader:
    if task_id not in _GRADER_MAP:
        raise KeyError(f"No grader registered for task '{task_id}'.")
    return _GRADER_MAP[task_id](task_id)

