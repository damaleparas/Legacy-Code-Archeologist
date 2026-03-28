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
R_TASK_COMPLETE   =  1.00
R_PARTIAL_CREDIT  =  0.30
R_STEP_PROGRESS   =  0.10
R_INVALID_ACTION  = -0.02
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
        +0.05  first ReadFile on main.py
        +0.10  any EditCode on main.py
        +0.20  server survived restart (no error in obs)
        +1.00  CallAPI → /health → HTTP 200  (terminal)
        -0.05  EditCode that introduces a syntax error
    """

    def grade(self, state: State, obs: Observation) -> Dict[str, Any]:
        reward  = 0.0
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
            reward += 0.20
            self._flag(state, "server_alive")
            message += " +0.20 server reachable."

        # Terminal: /health → 200
        if obs.status_code == 200 and not self._flagged(state, "task_done"):
            reward += R_TASK_COMPLETE
            done    = True
            self._flag(state, "task_done")
            message += " *** TASK COMPLETE +1.00 ***"

        return {"reward": reward, "done": done, "message": message.strip()}


# ---------------------------------------------------------------------------
# Task 2 — Auth header discovery
# ---------------------------------------------------------------------------

class Task2Grader(BaseGrader):
    """
    Reward schedule:
        +0.10  README.txt read (agent discovers token)
        +0.20  API call includes correct header key
        +0.30  HTTP 200 received
        +1.00  JSON body contains {"status": "ok"}  (terminal)
    """

    EXPECTED_HEADER = "X-Internal-Token"
    EXPECTED_JSON   = {"status": "ok"}

    def grade(self, state: State, obs: Observation) -> Dict[str, Any]:
        reward  = 0.0
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

        # Progress: header present in last API call (stored in grader_state)
        if obs.api_response is not None:
            # We can't inspect request headers from obs alone;
            # the env stores them in grader_state during dispatch
            headers_sent = state.grader_state.get("last_request_headers", {})
            if self.EXPECTED_HEADER in headers_sent and not self._flagged(state, "header_sent"):
                reward += 0.20
                self._flag(state, "header_sent")
                message += " +0.20 correct header sent."

        # Progress: HTTP 200
        if obs.status_code == 200 and not self._flagged(state, "status_200"):
            reward += 0.30
            self._flag(state, "status_200")
            message += " +0.30 HTTP 200."

        # Terminal: JSON validated
        if (
            obs.api_response
            and obs.api_response.get("status") == "ok"
            and not self._flagged(state, "task_done")
        ):
            reward += R_TASK_COMPLETE
            done    = True
            self._flag(state, "task_done")
            message += " *** TASK COMPLETE +1.00 ***"

        return {"reward": reward, "done": done, "message": message.strip()}


# ---------------------------------------------------------------------------
# Task 3 — Performance optimisation
# ---------------------------------------------------------------------------

class Task3Grader(BaseGrader):
    """
    Reward schedule:
        +0.10  file read and 'time.sleep' found in content
        +0.20  edit applied (sleep removed from file)
        +0.40  latency < 500 ms  (partial credit)
        +1.00  latency < 100 ms  (terminal, full score)
        Continuous bonus:  score = max(0, 1 - latency_ms / 100)
                           applied only when latency < 500 ms
    """

    LATENCY_FULL_MS    = 100    # ms → full score
    LATENCY_PARTIAL_MS = 500    # ms → partial credit

    def grade(self, state: State, obs: Observation) -> Dict[str, Any]:
        reward  = 0.0
        done    = False
        message = ""

        # Progress: agent found the bottleneck by reading the file
        if (
            obs.file_content
            and "time.sleep" in obs.file_content
            and not self._flagged(state, "bottleneck_found")
        ):
            reward += 0.10
            self._flag(state, "bottleneck_found")
            message += " +0.10 bottleneck located."

        # Progress: sleep removed (edit applied)
        if (
            "main.py" in state.files_modified
            and not self._flagged(state, "sleep_removed")
        ):
            # Verify the current file no longer contains time.sleep
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

        # Performance scoring based on observed latency
        if obs.latency > 0 and obs.status_code and obs.status_code == 200:
            latency_ms = obs.latency * 1000
            if latency_ms < self.LATENCY_FULL_MS:
                if not self._flagged(state, "task_done"):
                    # Continuous score component
                    continuous = max(0.0, 1.0 - latency_ms / self.LATENCY_FULL_MS)
                    reward += R_TASK_COMPLETE + continuous * 0.5
                    done    = True
                    self._flag(state, "task_done")
                    message += f" *** TASK COMPLETE +{R_TASK_COMPLETE:.2f} (latency={latency_ms:.1f} ms) ***"
            elif latency_ms < self.LATENCY_PARTIAL_MS:
                if not self._flagged(state, "partial_latency"):
                    reward += R_PARTIAL_CREDIT
                    self._flag(state, "partial_latency")
                    message += f" +0.30 partial (latency={latency_ms:.1f} ms, target<100 ms)."

        return {"reward": reward, "done": done, "message": message.strip()}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_GRADER_MAP = {
    "task_1_syntax_error":      Task1Grader,
    "task_2_auth_header":       Task2Grader,
    "task_3_perf_optimization": Task3Grader,
}


def get_grader(task_id: str) -> BaseGrader:
    if task_id not in _GRADER_MAP:
        raise KeyError(f"No grader registered for task '{task_id}'.")
    return _GRADER_MAP[task_id](task_id)
