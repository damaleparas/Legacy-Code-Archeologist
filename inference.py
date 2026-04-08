"""
Inference Script Example for LegacyCodeArcheologist
===================================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import List, Optional

# Force stdout to be unbuffered
sys.stdout.reconfigure(line_buffering=True)

from openai import OpenAI

# Import the environment and action schemas from the generated openenv client
try:
    from legacy_code_archeologist import (
        LegacyCodeArcheologistEnv,
        LegacyCodeArcheologistAction,
        LegacyCodeArcheologistReadFileAction,
        LegacyCodeArcheologistRunTestAction,
        LegacyCodeArcheologistEditCodeAction,
        LegacyCodeArcheologistCallApiAction
    )
except ImportError:
    LegacyCodeArcheologistEnv = None
    LegacyCodeArcheologistAction = None


IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("LEGACY_CODE_ARCHEOLOGIST_TASK", "task_1_syntax_error")
BENCHMARK = os.getenv("LEGACY_CODE_ARCHEOLOGIST_BENCHMARK", "legacy_code_archeologist")
MAX_STEPS = 15
TEMPERATURE = 0.7
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.5  

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI Software Archeologist.
    You will debug, document, and refactor legacy code in a sandboxed FastAPI environment.
    
    You can use the following actions:
    1. {"action_type": "ReadFile", "path": "file.py"}
    2. {"action_type": "RunTest", "command": "pytest tests/ -x"}
    3. {"action_type": "EditCode", "path": "file.py", "patch": "full new code..."}
    4. {"action_type": "CallAPI", "url": "http://localhost:8001/...", "method": "GET"}

    Respond ONLY with a valid JSON text representing the next action you want to take.
    Do not output any markdown formatting, just the raw JSON.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    line = f"[START] task={task} env={env} model={model}\n"
    sys.stdout.write(line)
    sys.stdout.flush()


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    line = f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}\n"
    sys.stdout.write(line)
    sys.stdout.flush()


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    line = f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}\n"
    sys.stdout.write(line)
    sys.stdout.flush()


def build_user_prompt(step: int, obs: dict, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    obs_summary = json.dumps({
        "stdout": obs.get("stdout") or "",
        "error": obs.get("error") or "",
        "status_code": obs.get("status_code"),
        "latency": obs.get("latency", 0),
        "files_modified": obs.get("files_modified", []),
        "file_content": obs.get("file_content")[:500] if obs.get("file_content") else ""
    }, indent=2)

    return textwrap.dedent(
        f"""
        Step: {step}
        Observation: {obs_summary}
        Last reward: {last_reward:.2f}
        Previous steps history:
        {history_block}
        
        Send your next action as a raw JSON without any markdown formatting.
        """
    ).strip()


def parse_action(json_text: str):
    try:
        data = json.loads(json_text.strip())
        action_type = data.get("action_type")
        if action_type == "ReadFile":
            return LegacyCodeArcheologistAction(read_file=LegacyCodeArcheologistReadFileAction(path=data.get("path", "")))
        elif action_type == "RunTest":
            return LegacyCodeArcheologistAction(run_test=LegacyCodeArcheologistRunTestAction(command=data.get("command", "")))
        elif action_type == "EditCode":
            return LegacyCodeArcheologistAction(edit_code=LegacyCodeArcheologistEditCodeAction(path=data.get("path", ""), patch=data.get("patch", "")))
        elif action_type == "CallAPI":
            return LegacyCodeArcheologistAction(call_api=LegacyCodeArcheologistCallApiAction(url=data.get("url", ""), method=data.get("method", "GET")))
        # Fallback to pure dict if pydantic unpacking has issues:
        return LegacyCodeArcheologistAction(**{
            "action_type": action_type,
            **{k:v for k,v in data.items() if k != "action_type"}
        })
    except Exception:
        # Default action on parse failure to keep moving
        try:
             return LegacyCodeArcheologistAction(run_test=LegacyCodeArcheologistRunTestAction(command="echo 'Invalid JSON'"))
        except:
             return None


def get_model_action_json(client: OpenAI, step: int, obs: dict, last_reward: float, history: List[str]) -> str:
    user_prompt = build_user_prompt(step, obs, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # strip markdown code blocks if the model wrapped it
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
    except Exception:
        return '{"action_type": "RunTest", "command": "echo Model request failed"}'


async def main() -> None:
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        # Initialize env
        env = await LegacyCodeArcheologistEnv.from_docker_image(IMAGE_NAME)

        result = await env.reset()
        obs = vars(result.observation)
        last_reward = 0.0

        max_steps = getattr(env, "max_steps", MAX_STEPS)
        if hasattr(result, 'state') and hasattr(result.state, 'max_steps'):
            max_steps = result.state.max_steps

        for step in range(1, max_steps + 1):
            if result.done:
                break

            action_json = get_model_action_json(client, step, obs, last_reward, history)
            action = parse_action(action_json)

            if not action:
                action_str = f"Parse failed: {action_json}"
                log_step(step=step, action=action_str, reward=0.0, done=False, error="Action parsing failed")
                continue

            try:
                result = await env.step(action)
                obs = vars(result.observation)
                reward = result.reward or 0.0
                done = result.done
                error = obs.get("error")
                action_str = json.dumps(json.loads(action_json))

            except Exception as e:
                reward = -0.5
                done = False
                error = str(e)
                action_str = "Agent execution exception"

            rewards.append(reward)
            steps_taken = step
            last_reward = reward
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards)
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        sys.stderr.write(f"Inference error: {e}\n")
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    if LegacyCodeArcheologistEnv is None:
        log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
        log_end(success=False, steps=0, score=0.0, rewards=[])
    else:
        asyncio.run(main())
