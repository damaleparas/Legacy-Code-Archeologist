"""
gym_wrapper.py — Gymnasium-compatible wrapper around LegacyCodeEnv
Allows training with Stable-Baselines3, RLlib, CleanRL, etc.

Usage:
    from gym_wrapper import LegacyCodeGymEnv
    env = LegacyCodeGymEnv(task_id="task_1_syntax_error")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from env import LegacyCodeEnv
from models import ReadFile, EditCode, RunTest, CallAPI
from task import TASK_REGISTRY


# ---------------------------------------------------------------------------
# Discrete action space design
# ---------------------------------------------------------------------------
# We discretise the action space into a flat integer index for simplicity.
# A real LLM-based agent would use the raw Action dataclasses directly.

DISCRETE_ACTIONS = [
    ReadFile(path="main.py"),
    ReadFile(path="README.txt"),
    EditCode(path="main.py", patch=""),          # patch filled at runtime
    RunTest(command="pytest tests/ -x -q"),
    CallAPI(url="http://localhost:8001/health"),
    CallAPI(url="http://localhost:8001/process", method="POST",
            headers={"X-Internal-Token": "arch3olog1st-s3cr3t-2019"}),
    CallAPI(url="http://localhost:8001/compute"),
]

OBS_DIM = 64   # flattened observation vector length


def obs_to_vector(obs_dict: Dict[str, Any]) -> np.ndarray:
    """Embed the structured observation into a fixed-length float32 vector."""
    vec = np.zeros(OBS_DIM, dtype=np.float32)
    # Simple hand-crafted features — replace with an LLM encoder for real training
    vec[0]  = float(len(obs_dict.get("file_content", "")) > 0)
    vec[1]  = float(len(obs_dict.get("stdout", "")) > 0)
    vec[2]  = float(obs_dict.get("api_response") is not None)
    vec[3]  = min(obs_dict.get("latency", 0.0), 5.0) / 5.0
    vec[4]  = (obs_dict.get("status_code") or 0) / 500.0
    vec[5]  = float("error" in obs_dict.get("error", ""))
    vec[6]  = obs_dict.get("step_count", 0) / 30.0
    vec[7]  = len(obs_dict.get("files_modified", [])) / 5.0
    # Bag-of-words style: did the file content contain key strings?
    fc = obs_dict.get("file_content", "")
    vec[8]  = float("SyntaxError" in fc or "def health()" in fc)
    vec[9]  = float("time.sleep" in fc)
    vec[10] = float("X-Internal-Token" in fc)
    return vec


class LegacyCodeGymEnv(gym.Env):
    """Gymnasium wrapper for single-task episodes."""

    metadata = {"render_modes": []}

    def __init__(self, task_id: str = "task_1_syntax_error"):
        super().__init__()
        self.task_id = task_id
        self._inner  = LegacyCodeEnv(task_id=task_id)

        self.action_space      = gym.spaces.Discrete(len(DISCRETE_ACTIONS))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        obs = self._inner.reset()
        return obs_to_vector(obs.to_dict()), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        act = DISCRETE_ACTIONS[action]
        obs, reward, done, info = self._inner.step(act)
        truncated = info["step_count"] >= TASK_REGISTRY[self.task_id]["max_steps"]
        return obs_to_vector(obs.to_dict()), reward, done, truncated, info

    def close(self) -> None:
        self._inner.close()
