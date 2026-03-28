"""
models.py — LegacyCodeArcheologist
Type-safe Action, Observation, and State definitions for the OpenEnv framework.
Follows the openenv-core / env_server specification.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Action primitives
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    READ_FILE   = "ReadFile"
    EDIT_CODE   = "EditCode"
    RUN_TEST    = "RunTest"
    CALL_API    = "CallAPI"


@dataclass
class ReadFile:
    """Read the contents of a file from the sandboxed filesystem."""
    action_type: ActionType = field(default=ActionType.READ_FILE, init=False)
    path: str = ""                        # Relative to sandbox root

    def validate(self) -> None:
        if not self.path:
            raise ValueError("ReadFile.path must be a non-empty string.")
        if ".." in self.path:
            raise ValueError("Path traversal ('..') is not allowed.")

    def to_dict(self) -> Dict[str, Any]:
        return {"action_type": self.action_type.value, "path": self.path}


@dataclass
class EditCode:
    """Apply a unified-diff patch to a file inside the sandbox."""
    action_type: ActionType = field(default=ActionType.EDIT_CODE, init=False)
    path: str  = ""
    patch: str = ""                       # Unified diff OR full replacement text

    def validate(self) -> None:
        if not self.path:
            raise ValueError("EditCode.path must be non-empty.")
        if not self.patch:
            raise ValueError("EditCode.patch must contain at least one change.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "path": self.path,
            "patch": self.patch,
        }


@dataclass
class RunTest:
    """Execute a shell command inside the sandbox (pytest, curl, etc.)."""
    action_type: ActionType = field(default=ActionType.RUN_TEST, init=False)
    command: str = ""                     # e.g. "pytest tests/ -x -q"

    # Safety allow-list — override per deployment
    ALLOWED_PREFIXES: List[str] = field(
        default_factory=lambda: ["pytest", "python", "curl", "cat", "ls"],
        repr=False,
    )

    def validate(self) -> None:
        if not self.command:
            raise ValueError("RunTest.command must be non-empty.")
        cmd_root = self.command.strip().split()[0]
        if cmd_root not in self.ALLOWED_PREFIXES:
            raise ValueError(
                f"Command '{cmd_root}' not in allow-list: {self.ALLOWED_PREFIXES}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {"action_type": self.action_type.value, "command": self.command}


@dataclass
class CallAPI:
    """Send an HTTP GET/POST to the internal sandboxed FastAPI server."""
    action_type: ActionType = field(default=ActionType.CALL_API, init=False)
    url: str                   = "http://localhost:8000/"
    method: str                = "GET"
    headers: Dict[str, str]    = field(default_factory=dict)
    payload: Dict[str, Any]    = field(default_factory=dict)
    timeout: float             = 5.0

    def validate(self) -> None:
        if not self.url.startswith("http://localhost"):
            raise ValueError("CallAPI is restricted to http://localhost.")
        if self.method not in ("GET", "POST", "PUT", "DELETE", "PATCH"):
            raise ValueError(f"Unsupported HTTP method: {self.method}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "url": self.url,
            "method": self.method,
            "headers": self.headers,
            "payload": self.payload,
        }


# Union alias used by the environment step() signature
Action = Union[ReadFile, EditCode, RunTest, CallAPI]


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """
    Everything the agent perceives after each step.

    Fields
    ------
    file_content   : Raw text of the last file read (empty string if N/A).
    stdout         : Combined stdout + stderr from the last subprocess.
    api_response   : Parsed JSON body of the last HTTP call (None if N/A).
    latency        : Wall-clock seconds for the last action.
    status_code    : HTTP status code (None for non-HTTP actions).
    error          : Human-readable error message, empty on success.
    step_count     : Cumulative steps taken in the current episode.
    files_modified : List of sandbox paths edited so far this episode.
    """
    file_content:   str                   = ""
    stdout:         str                   = ""
    api_response:   Optional[Dict]        = None
    latency:        float                 = 0.0
    status_code:    Optional[int]         = None
    error:          str                   = ""
    step_count:     int                   = 0
    files_modified: List[str]             = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_content":   self.file_content,
            "stdout":         self.stdout,
            "api_response":   self.api_response,
            "latency":        self.latency,
            "status_code":    self.status_code,
            "error":          self.error,
            "step_count":     self.step_count,
            "files_modified": self.files_modified,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Observation":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# State  (internal — not directly exposed to the agent)
# ---------------------------------------------------------------------------

@dataclass
class State:
    """
    Full internal state of the environment.
    Serialised by env_server for checkpointing / replay.
    """
    task_id:          str            = "task_1_syntax_error"
    sandbox_root:     str            = "/tmp/sandbox"
    step_count:       int            = 0
    max_steps:        int            = 30
    cumulative_reward: float         = 0.0
    done:             bool           = False
    files_modified:   List[str]      = field(default_factory=list)
    server_pid:       Optional[int]  = None
    server_port:      int            = 8001
    episode_start:    float          = field(default_factory=time.time)

    # Per-task scratch storage (grader can stash intermediate signals here)
    grader_state:     Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id":            self.task_id,
            "sandbox_root":       self.sandbox_root,
            "step_count":         self.step_count,
            "max_steps":          self.max_steps,
            "cumulative_reward":  self.cumulative_reward,
            "done":               self.done,
            "files_modified":     self.files_modified,
            "server_pid":         self.server_pid,
            "server_port":        self.server_port,
            "grader_state":       self.grader_state,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "State":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
