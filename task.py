"""
task.py — LegacyCodeArcheologist
Task registry: 3 deterministic tasks of increasing difficulty.
"""

from __future__ import annotations

from typing import Any, Dict

# ---------------------------------------------------------------------------
# Task registry
# Each entry describes one episode configuration.
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, Dict[str, Any]] = {

    # -----------------------------------------------------------------------
    # Task 1 — EASY
    # A FastAPI route has a SyntaxError (missing colon on function def).
    # The agent must: ReadFile → EditCode (fix it) → CallAPI → 200 OK.
    # -----------------------------------------------------------------------
    "task_1_syntax_error": {
        "id":          "task_1_syntax_error",
        "difficulty":  "easy",
        "description": (
            "A FastAPI route in main.py has a SyntaxError. "
            "Read the file, fix the Python syntax, and verify "
            "the /health endpoint returns HTTP 200."
        ),
        "template_files": ["main.py"],
        "success_criteria": {
            "endpoint":     "/health",
            "method":       "GET",
            "expected_status": 200,
        },
        "max_steps": 15,
        "reward_shaping": {
            "read_file":     0.05,
            "edit_applied":  0.10,
            "server_alive":  0.10,
            "status_200":    0.74,   # 0.05 + 0.10 + 0.10 + 0.74 = 0.99
        },
        "hint": "Look for a missing ':' on a function definition line.",
    },

    # -----------------------------------------------------------------------
    # Task 2 — MEDIUM
    # The /process endpoint requires an 'X-Internal-Token' header.
    # Token is documented only in README.txt. Agent must read README,
    # then call the API with the correct header and validate JSON body.
    # -----------------------------------------------------------------------
    "task_2_auth_header": {
        "id":          "task_2_auth_header",
        "difficulty":  "medium",
        "description": (
            "The /process endpoint is returning 401 Unauthorized. "
            "Read README.txt to discover the required authentication header, "
            "then call the endpoint with the correct 'X-Internal-Token' header "
            "and confirm the JSON response contains {'status': 'ok'}."
        ),
        "template_files": ["main.py"],
        "success_criteria": {
            "endpoint":           "/process",
            "method":             "POST",
            "required_header_key": "X-Internal-Token",
            "expected_json_key":  "status",
            "expected_json_value": "ok",
        },
        "max_steps": 20,
        "reward_shaping": {
            "readme_read":       0.10,
            "header_present":    0.20,
            "status_200":        0.20,
            "json_validated":    0.49,   # 0.1+0.2+0.2+0.49 = 0.99
        },
        "hint": "The token is documented in README.txt under 'Internal Auth'.",
    },

    # -----------------------------------------------------------------------
    # Task 3 — HARD
    # The /compute endpoint has a time.sleep(2) bottleneck.
    # Agent must locate, remove it, and verify latency < 100 ms.
    # Score is continuous: 1.0 if <100 ms, partial if <500 ms.
    # -----------------------------------------------------------------------
    "task_3_perf_optimization": {
        "id":          "task_3_perf_optimization",
        "difficulty":  "hard",
        "description": (
            "The /compute endpoint is unacceptably slow (>2 s). "
            "Profile the code by reading main.py, remove the artificial "
            "bottleneck (time.sleep), redeploy, and verify that the endpoint "
            "responds in under 100 ms."
        ),
        "template_files": ["main.py"],
        "success_criteria": {
            "endpoint":        "/compute",
            "method":          "GET",
            "latency_target_ms": 100,
        },
        "max_steps": 25,
        "reward_shaping": {
            "bottleneck_found":   0.10,
            "sleep_removed":      0.20,
            "latency_lt_500ms":   0.20,
            "latency_lt_100ms":   0.49,   # 0.1+0.2+0.2+0.49 = 0.99
        },
        "hint": "Search for 'time.sleep' inside main.py.",
    },

    # -----------------------------------------------------------------------
    # Task 4 — HARD
    # SQLite schema mismatch. Column renamed from "username" to "user_id".
    # Agent must inspect schema and update the SQL query.
    # -----------------------------------------------------------------------
    "task_4_db_schema_mismatch": {
        "id":          "task_4_db_schema_mismatch",
        "difficulty":  "hard",
        "description": (
            "The /user-data endpoint is crashing during database lookups. "
            "Inspect the SQLite database schema in 'app.db', identify why "
            "the query in main.py is failing, and fix the column name mismatch."
        ),
        "template_files": ["main.py", "app.db"],
        "success_criteria": {
            "endpoint":     "/user-data",
            "method":       "GET",
            "expected_status": 200,
        },
        "max_steps": 25,
        "reward_shaping": {
            "schema_read":     0.10,
            "query_fixed":      0.40,
            "status_200":       0.49,    # 0.1+0.4+0.49 = 0.99
        },
        "hint": "Use 'sqlite3 app.db .schema' to check column names.",
    },

    # -----------------------------------------------------------------------
    # Task 5 — MEDIUM
    # Environment variable leakage. The /env endpoint leaks DB_PASSWORD.
    # Agent must mask it with '********' in the response.
    # -----------------------------------------------------------------------
    "task_5_env_var_leak": {
        "id":          "task_5_env_var_leak",
        "difficulty":  "medium",
        "description": (
            "The /env endpoint is leaking sensitive configuration. "
            "Modify main.py to mask the 'DB_PASSWORD' value with '********' "
            "in the JSON response."
        ),
        "template_files": ["main.py"],
        "success_criteria": {
            "endpoint":     "/env",
            "method":       "GET",
            "expected_json_key": "DB_PASSWORD",
            "expected_json_value": "********",
        },
        "max_steps": 15,
        "reward_shaping": {
            "endpoint_checked":  0.10,
            "mask_applied":      0.40,
            "status_200":         0.49,
        },
        "hint": "Check the /env route and look for where it returns all env variables.",
    },
}



def get_task(task_id: str) -> Dict[str, Any]:
    if task_id not in TASK_REGISTRY:
        raise KeyError(
            f"Task '{task_id}' not found. Available: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_id]
