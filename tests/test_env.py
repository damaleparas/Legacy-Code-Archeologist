"""
tests/test_env.py — Full test suite for LegacyCodeArcheologist
Run:  pytest tests/ -v
"""

from __future__ import annotations

import sys
import os
import pytest
from models import (
    ReadFile, EditCode, RunTest, CallAPI,
    Observation, State, ActionType,
)
from grader import Task1Grader, Task2Grader, Task3Grader, Task4Grader
from task import TASK_REGISTRY, get_task


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestModels:

    def test_read_file_valid(self):
        a = ReadFile(path="main.py")
        assert a.action_type == ActionType.READ_FILE
        a.validate()   # should not raise

    def test_read_file_path_traversal(self):
        a = ReadFile(path="../../etc/passwd")
        with pytest.raises(ValueError, match="traversal"):
            a.validate()

    def test_edit_code_empty_patch(self):
        a = EditCode(path="main.py", patch="")
        with pytest.raises(ValueError, match="patch"):
            a.validate()

    def test_run_test_disallowed(self):
        a = RunTest(command="rm -rf /")
        with pytest.raises(ValueError, match="allow-list"):
            a.validate()

    def test_run_test_allowed(self):
        a = RunTest(command="pytest tests/ -q")
        a.validate()

    def test_call_api_external_blocked(self):
        a = CallAPI(url="https://evil.com/steal")
        with pytest.raises(ValueError, match="localhost"):
            a.validate()

    def test_observation_roundtrip(self):
        obs = Observation(
            file_content = "hello",
            status_code  = 200,
            latency      = 0.05,
        )
        d   = obs.to_dict()
        obs2 = Observation.from_dict(d)
        assert obs2.file_content == "hello"
        assert obs2.status_code  == 200

    def test_state_roundtrip(self):
        s  = State(task_id="task_1_syntax_error", step_count=5)
        d  = s.to_dict()
        s2 = State.from_dict(d)
        assert s2.step_count == 5


# ---------------------------------------------------------------------------
# Task registry tests
# ---------------------------------------------------------------------------

class TestTaskRegistry:

    def test_all_tasks_present(self):
        assert "task_1_syntax_error"      in TASK_REGISTRY
        assert "task_2_auth_header"       in TASK_REGISTRY
        assert "task_3_perf_optimization" in TASK_REGISTRY

    def test_task_has_required_fields(self):
        for tid, cfg in TASK_REGISTRY.items():
            assert "description"      in cfg, f"{tid} missing 'description'"
            assert "template_files"   in cfg, f"{tid} missing 'template_files'"
            assert "success_criteria" in cfg, f"{tid} missing 'success_criteria'"
            assert "max_steps"        in cfg, f"{tid} missing 'max_steps'"

    def test_get_task_invalid(self):
        with pytest.raises(KeyError):
            get_task("nonexistent_task")


# ---------------------------------------------------------------------------
# Grader unit tests (no live server needed)
# ---------------------------------------------------------------------------

class TestTask1Grader:

    def _make(self):
        g = Task1Grader("task_1_syntax_error")
        s = State(task_id="task_1_syntax_error")
        return g, s

    def test_syntax_error_penalty(self):
        g, s = self._make()
        obs = Observation(error="[syntax_error] invalid syntax (line 5)")
        result = g.grade(s, obs)
        # 0.01 participation + (-0.05 penalty) = -0.04
        assert result["reward"] == pytest.approx(-0.04)
        assert not result["done"]

    def test_file_read_reward(self):
        g, s = self._make()
        obs = Observation(file_content="def health():")
        result = g.grade(s, obs)
        # 0.01 participation + 0.05 read = 0.06
        assert result["reward"] == pytest.approx(0.06)

    def test_file_read_not_double_counted(self):
        g, s = self._make()
        obs = Observation(file_content="def health():")
        g.grade(s, obs)
        result2 = g.grade(s, obs)   # second read
        assert result2["reward"] == pytest.approx(0.0)

    def test_terminal_200_ok(self):
        g, s = self._make()
        obs = Observation(status_code=200, api_response={"status": "healthy"})
        result = g.grade(s, obs)
        assert result["done"]
        # 0.01 + 0.73 + (server reachable? no, obs didn't have status_code != 0 before terminal)
        # Actually Task1Grader reward math:
        # participation=0.01
        # server_alive=0.10 (if obs.status_code is not None and != 0)
        # status_200_terminal=0.73
        # Total: 0.84
        assert result["reward"] >= 0.70


    def test_server_alive_reward(self):
        g, s = self._make()
        obs = Observation(status_code=404)   # server up but wrong route
        result = g.grade(s, obs)
        # 0.01 participation + 0.10 server alive = 0.11
        assert result["reward"] == pytest.approx(0.11)


class TestTask2Grader:

    def _make(self):
        g = Task2Grader("task_2_auth_header")
        s = State(task_id="task_2_auth_header")
        return g, s

    def test_readme_read_reward(self):
        g, s = self._make()
        obs = Observation(file_content="X-Internal-Token: arch3olog1st-s3cr3t-2019")
        result = g.grade(s, obs)
        # 0.01 + 0.10 = 0.11
        assert result["reward"] == pytest.approx(0.11)

    def test_http_200_reward(self):
        g, s = self._make()
        obs = Observation(status_code=200, api_response={"other": "field"})
        result = g.grade(s, obs)
        # 0.01 + 0.20 = 0.21
        assert result["reward"] == pytest.approx(0.21)

    def test_json_validated_terminal(self):
        g, s = self._make()
        obs = Observation(
            status_code  = 200,
            api_response = {"status": "ok"},
        )
        result = g.grade(s, obs)
        assert result["done"]
        # 0.01 + 0.20 (status 200) + 0.48 (json) = 0.69
        assert result["reward"] == pytest.approx(0.69)


class TestTask3Grader:

    def _make(self, tmp_path=None):
        g = Task3Grader("task_3_perf_optimization")
        s = State(task_id="task_3_perf_optimization")
        if tmp_path:
            s.sandbox_root = str(tmp_path)
        return g, s

    def test_bottleneck_found_reward(self):
        g, s = self._make()
        obs = Observation(file_content="time.sleep(2)  # BOTTLENECK")
        result = g.grade(s, obs)
        # 0.01 + 0.10 = 0.11
        assert result["reward"] == pytest.approx(0.11)

    def test_fast_latency_terminal(self, tmp_path):
        # Create a main.py without sleep
        (tmp_path / "main.py").write_text("# clean code")
        g, s = self._make(tmp_path)
        s.files_modified = ["main.py"]
        obs = Observation(status_code=200, latency=0.010)   # 10 ms
        result = g.grade(s, obs)
        assert result["done"]
        # 0.01 participation + 0.20 sleep removed + 0.20 partial + 0.48 terminal = 0.89
        # Wait, if latency < 100 on first go: 0.01 + 0.48 + 10 ms bonus
        assert result["reward"] >= 0.40

    def test_partial_latency(self, tmp_path):
        (tmp_path / "main.py").write_text("# clean code")
        g, s = self._make(tmp_path)
        s.files_modified = ["main.py"]
        obs = Observation(status_code=200, latency=0.300)   # 300 ms
        result = g.grade(s, obs)
        assert not result["done"]
        # 0.01 + 0.20 + 0.20 = 0.41
        assert result["reward"] >= 0.40

    def test_slow_latency_no_reward(self, tmp_path):
        (tmp_path / "main.py").write_text("time.sleep(2)")
        g, s = self._make(tmp_path)
        obs = Observation(status_code=200, latency=2.100)   # still slow
        result = g.grade(s, obs)
        assert not result["done"]
        # Only participation 0.01
        assert result["reward"] == pytest.approx(0.01)


class TestTask4Grader:
    def _make(self):
        g = Task4Grader("task_4_db_schema_mismatch")
        s = State(task_id="task_4_db_schema_mismatch")
        return g, s

    def test_schema_read_reward(self):
        g, s = self._make()
        obs = Observation(stdout="CREATE TABLE users (user_id INT)")
        result = g.grade(s, obs)
        assert result["reward"] == pytest.approx(0.11)

    def test_terminal_success(self):
        g, s = self._make()
        obs = Observation(status_code=200)
        result = g.grade(s, obs)
        assert result["done"]
        assert result["reward"] == pytest.approx(0.49)



# ---------------------------------------------------------------------------
# Smoke test: env reset (no live server; just checks file deployment)
# ---------------------------------------------------------------------------

class TestEnvSmoke:

    def test_import(self):
        from env import LegacyCodeEnv
        assert LegacyCodeEnv.metadata["name"] == "legacy_code_archeologist"

    def test_invalid_task_id(self):
        from env import LegacyCodeEnv
        with pytest.raises(ValueError, match="Unknown task"):
            LegacyCodeEnv(task_id="bad_task")
