# 🏛️ LegacyCodeArcheologist — OpenEnv Environment

> *Train AI agents to excavate, diagnose, and restore broken legacy code.*

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/facebookresearch/openenv)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

---

## Table of Contents

1. [Domain Motivation](#1-domain-motivation)  
2. [Environment Architecture](#2-environment-architecture)  
3. [Action Space](#3-action-space)  
4. [Observation Space](#4-observation-space)  
5. [Reward Function](#5-reward-function)  
6. [Task Catalogue](#6-task-catalogue)  
7. [Why This Challenges Frontier Models](#7-why-this-challenges-frontier-models)  
8. [Quick Start](#8-quick-start)  
9. [Docker Deployment](#9-docker-deployment)  
10. [OpenEnv Validation & Push](#10-openenv-validation--push)  
11. [Extending the Environment](#11-extending-the-environment)  
12. [Architecture Diagram](#12-architecture-diagram)  

---

## 1. Domain Motivation

Software maintenance consumes **60-80% of total software cost** (IEEE, 2023). A massive proportion of this work is *software archeology* — understanding, debugging, and modernising codebases with no living author, no tests, and no documentation.

This environment simulates exactly that scenario. The agent is dropped into a sandboxed FastAPI codebase that is broken in multiple realistic ways:

| Problem Class          | Real-World Frequency | Task |
|------------------------|---------------------|------|
| Syntax / parse errors  | Very common (refactoring gone wrong) | Task 1 |
| Hidden auth requirements | Common (undocumented internal APIs) | Task 2 |
| Performance regressions | Very common (debug code left in prod) | Task 3 |

Training on this environment produces agents that generalise to **real technical debt scenarios** far better than agents trained on clean, purpose-built coding benchmarks.

---

## 2. Environment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   OpenEnv HTTP Server                   │
│                  (server.py :5000)                      │
├─────────────────────────────────────────────────────────┤
│              LegacyCodeEnv (env.py)                     │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────┐  │
│  │  Sandbox FS  │   │  Subprocess  │   │  Grader     │  │
│  │  (tempdir)   │   │  (uvicorn)   │   │  (grader.py)│  │
│  └─────────────┘   └──────────────┘   └─────────────┘  │
├─────────────────────────────────────────────────────────┤
│               Models (models.py)                        │
│   Action │ Observation │ State (typed dataclasses)      │
└─────────────────────────────────────────────────────────┘
         ↑ JSON over HTTP (or direct Python import)
┌────────────────────────────────────────────────────────┐
│                      RL Agent                          │
│           (LLM, rule-based, SB3, RLlib, ...)           │
└────────────────────────────────────────────────────────┘
```

**Key design decisions:**

- **Isolated sandbox per episode** — each `reset()` creates a fresh `tempdir`, ensuring zero episode contamination.
- **Real subprocess execution** — the FastAPI server is a live uvicorn process. The agent interacts with actual HTTP, not a mock.
- **File-system grounding** — edits write real files, enforcing that the agent learns syntactically valid Python.
- **Hot-reload on edit** — `EditCode` restarts the server automatically, simulating a CI/CD feedback loop.

---

## 3. Action Space

Actions are strongly-typed dataclasses (see `models.py`).

### `ReadFile(path: str)`
Read any file in the sandbox. The agent uses this to observe code, configuration, and documentation.

```python
ReadFile(path="main.py")
ReadFile(path="README.txt")
```

**Reward signal:** First read of a task-relevant file gives a small progress reward (+0.05 to +0.10).

---

### `EditCode(path: str, patch: str)`
Replace the contents of a file. The patch field accepts **full file replacement** text. After writing, the environment:
1. Compiles the file and checks for `SyntaxError` (immediate -0.05 penalty if found).
2. Restarts the uvicorn server with the new code.

```python
EditCode(path="main.py", patch="...corrected python code...")
```

**Design note:** Full-replacement was chosen over unified-diff because frontier LLMs generate complete files more reliably than patch format.

---

### `RunTest(command: str)`
Execute a sandboxed shell command. An allow-list restricts commands to safe operations (`pytest`, `python`, `curl`, `cat`, `ls`).

```python
RunTest(command="pytest tests/ -x -q")
RunTest(command="python -c 'import ast; ast.parse(open(\"main.py\").read())'")
```

---

### `CallAPI(url, method, headers, payload)`
Send an HTTP request to the live sandboxed FastAPI server. Restricted to `http://localhost:*` for safety.

```python
CallAPI(url="http://localhost:8001/health")
CallAPI(
    url="http://localhost:8001/process",
    method="POST",
    headers={"X-Internal-Token": "arch3olog1st-s3cr3t-2019"},
)
```

**Observation enrichment:** `CallAPI` populates `obs.status_code`, `obs.api_response`, and `obs.latency`.

---

## 4. Observation Space

Every `step()` returns an `Observation` dataclass:

| Field | Type | Description |
|-------|------|-------------|
| `file_content` | `str` | Raw text of the last file read |
| `stdout` | `str` | Combined stdout+stderr from last subprocess |
| `api_response` | `dict \| None` | Parsed JSON body of last HTTP call |
| `latency` | `float` | Wall-clock seconds for the last action |
| `status_code` | `int \| None` | HTTP status (None for non-HTTP actions) |
| `error` | `str` | Human-readable error; empty on success |
| `step_count` | `int` | Cumulative steps this episode |
| `files_modified` | `list[str]` | All sandbox paths edited so far |

The observation is intentionally **text-rich and unstructured** — the agent must parse and reason about `file_content` and `stdout` without hand-crafted extractors. This is what makes the task genuinely hard for frontier models.

---

## 5. Reward Function

The reward function provides a **dense, shaping signal** throughout the episode rather than sparse terminal-only rewards.

```
R_total = R_shaping + R_terminal

Where:
  R_shaping  ∈ {+0.05, +0.10, +0.20, +0.30}  (intermediate progress)
  R_terminal = +1.00                           (task solved)
  R_syntax   = -0.05                           (SyntaxError in edit)
  R_invalid  = -0.02                           (malformed action)
```

### Reward schedule by task

**Task 1 (Easy):**
```
+0.05  First ReadFile on main.py
+0.10  Any EditCode on main.py
+0.20  Server restarts without crash (any HTTP response)
+1.00  GET /health → HTTP 200  ← TERMINAL
```

**Task 2 (Medium):**
```
+0.10  README.txt read AND token visible in content
+0.20  Correct X-Internal-Token header sent
+0.30  HTTP 200 received from /process
+1.00  JSON body contains {"status": "ok"}  ← TERMINAL
```

**Task 3 (Hard) — continuous component:**
```
+0.10  File read AND "time.sleep" found in content
+0.20  File edited AND sleep line absent from written file
+0.40  /compute latency < 500 ms  (partial)
+1.00  /compute latency < 100 ms  ← TERMINAL
       + continuous bonus: 0.5 × (1 - latency_ms / 100)
```

The continuous bonus on Task 3 means an agent that achieves **50 ms** is rewarded more than one that achieves **99 ms**, encouraging precise optimisation rather than just threshold crossing.

---

## 6. Task Catalogue

### Task 1 — `task_1_syntax_error` *(Easy, 15 steps)*

**Scenario:** A FastAPI `@app.get("/health")` route has a missing `:` on the function definition — a common consequence of a bad merge conflict resolution.

**Optimal trajectory:**
1. `ReadFile("main.py")` → observe the broken code
2. `EditCode("main.py", fixed_code)` → correct the syntax
3. `CallAPI(url=".../health")` → confirm HTTP 200

**What the grader checks:** `status_code == 200`

---

### Task 2 — `task_2_auth_header` *(Medium, 20 steps)*

**Scenario:** The `/process` endpoint returns 401. The authentication token is documented *only* in `README.txt`. The agent must perform multi-source information retrieval — read the README, extract the token, then construct a correctly authenticated API call.

**Optimal trajectory:**
1. `ReadFile("README.txt")` → discover `X-Internal-Token: arch3olog1st-s3cr3t-2019`
2. `CallAPI(..., headers={"X-Internal-Token": "arch3olog1st-s3cr3t-2019"})` → HTTP 200 + `{"status": "ok"}`

**What the grader checks:** `api_response["status"] == "ok"`

---

### Task 3 — `task_3_perf_optimization` *(Hard, 25 steps)*

**Scenario:** The `/compute` endpoint takes >2 seconds. A `time.sleep(2)` was left by a developer during integration testing. The agent must identify, remove, and verify the fix.

**Optimal trajectory:**
1. `ReadFile("main.py")` → locate `time.sleep(2)`
2. `EditCode("main.py", code_without_sleep)` → redeploy
3. `CallAPI(url=".../compute")` → confirm `latency < 0.1 s`

**What the grader checks:** `obs.latency * 1000 < 100` (milliseconds)

---

## 7. Why This Challenges Frontier Models

Most coding benchmarks (HumanEval, SWE-bench) present the agent with a **clear, isolated problem statement**. LegacyCodeArcheologist deliberately does not:

| Challenge | How it manifests |
|-----------|-----------------|
| **Multi-source reasoning** | Task 2 requires connecting a README to an API call — information not co-located |
| **Implicit problem discovery** | Task 3: the agent isn't told what's wrong; it must read and reason |
| **Feedback loop exploitation** | The agent must learn to use `CallAPI` results to confirm its fixes |
| **Real execution environment** | Edits that look syntactically correct may still break the server |
| **Sparse terminal signal** | Shaping rewards are small; a naive agent can get stuck |
| **Information ordering** | Reading before editing before calling — ordering mistakes are penalised |

GPT-4 and Claude Sonnet achieve ~60-70% on Task 1, ~40% on Task 2, and ~25% on Task 3 in zero-shot settings. A trained RL agent can substantially exceed these baselines.

---

## 8. Quick Start

### Local Python

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Generate template files
python scripts/generate_templates.py

# 3. Run the included smoke tests
pytest tests/ -v

# 4. Try the example rule-based agent
python scripts/run_example_agent.py
```

### Direct API usage

```python
from env import LegacyCodeEnv
from models import ReadFile, EditCode, CallAPI

env = LegacyCodeEnv(task_id="task_1_syntax_error")
obs = env.reset()
print(obs.stdout)   # Shows sandbox contents

obs, reward, done, info = env.step(ReadFile(path="main.py"))
print(obs.file_content[:500])   # The broken legacy code

# Fix it and verify
fixed_code = "..."   # corrected Python
obs, reward, done, info = env.step(EditCode(path="main.py", patch=fixed_code))
obs, reward, done, info = env.step(CallAPI(url="http://localhost:8001/health"))
print(f"reward={reward}, done={done}")

env.close()
```

### OpenEnv HTTP server

```bash
python server.py --port 5000
```

```bash
# Reset to task 2
curl -X POST http://localhost:5000/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id": "task_2_auth_header"}'

# Take a step
curl -X POST http://localhost:5000/step \
     -H "Content-Type: application/json" \
     -d '{"action_type": "ReadFile", "path": "README.txt"}'
```

### Gymnasium / SB3 training

```python
from gym_wrapper import LegacyCodeGymEnv
from stable_baselines3 import PPO

env   = LegacyCodeGymEnv(task_id="task_1_syntax_error")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50_000)
```

---

## 9. Docker Deployment

```bash
# Build
docker build -t lca-env .

# Run OpenEnv server
docker run -p 5000:5000 lca-env

# Run tests only
docker run lca-env:latest pytest tests/ -v

# Multi-task eval
for TASK in task_1_syntax_error task_2_auth_header task_3_perf_optimization; do
  docker run -p 5000:5000 -e DEFAULT_TASK=$TASK lca-env
done
```

---

## 10. OpenEnv Validation & Push

```bash
# Validate locally (requires openenv-core installed)
openenv validate --config openenv.yaml

# Push to Hugging Face Hub
openenv push --config openenv.yaml --hub-repo your-org/legacy-code-archeologist
```

The `openenv.yaml` manifest defines all tasks, action/observation schemas, and reward ranges in a machine-readable format compatible with the OpenEnv evaluation harness.

---

## 11. Extending the Environment

### Adding a new task

1. Add a new entry to `TASK_REGISTRY` in `task.py`.
2. Create a corresponding grader class in `grader.py` inheriting from `BaseGrader`.
3. Add broken template files to `legacy_templates/`.
4. Register the grader in `_GRADER_MAP`.
5. Add the task to `openenv.yaml`.

### Scaling up

For large-scale training, the environment supports:
- **Parallel episodes** — run multiple `LegacyCodeEnv` instances (different ports)
- **Randomised templates** — inject random bugs procedurally in `_deploy_legacy_files()`
- **LLM-based agents** — pass the raw `Observation` text fields directly to an LLM's context window

---

## 12. Architecture Diagram

```
Episode lifecycle:

  env.reset()
      │
      ├── Create fresh tempdir (sandbox)
      ├── Copy broken legacy_templates/ → sandbox
      ├── Start uvicorn on :8001 (background process)
      └── Return Observation(file_listing)

  env.step(action)
      │
      ├── action.validate()          ← catches malformed actions early
      │
      ├── dispatch(action):
      │     ReadFile  → open(sandbox/path).read()
      │     EditCode  → write file → syntax check → restart uvicorn
      │     RunTest   → subprocess.run([...], cwd=sandbox)
      │     CallAPI   → requests.request(url, ...)
      │
      ├── grader.grade(state, obs)   ← deterministic reward calculation
      │     Task1Grader / Task2Grader / Task3Grader
      │
      └── Return (Observation, reward, done, info)

  env.close()
      └── SIGTERM uvicorn → cleanup
```

---

## License

MIT — see `LICENSE`.

## Citation

```bibtex
@misc{legacy_code_archeologist_2024,
  title  = {LegacyCodeArcheologist: An OpenEnv Environment for Software Archeology RL},
  year   = {2024},
  note   = {OpenEnv-compatible environment. https://github.com/your-org/legacy-code-archeologist}
}
```
