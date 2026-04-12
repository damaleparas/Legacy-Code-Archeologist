#!/usr/bin/env bash
# validate-submission.sh — pre-submission validator for LegacyCodeArcheologist
# Usage: bash scripts/validate-submission.sh <BASE_URL> <REPO_DIR>
# Example: bash scripts/validate-submission.sh https://your-space.hf.space .

set -euo pipefail

BASE_URL="${1:-http://localhost:7860}"
REPO_DIR="${2:-.}"
PASS=0
FAIL=0

green()  { echo -e "\033[32m[PASS]\033[0m $*"; }
red()    { echo -e "\033[31m[FAIL]\033[0m $*"; }
yellow() { echo -e "\033[33m[INFO]\033[0m $*"; }

check() {
    local desc="$1"; local result="$2"
    if [ "$result" = "ok" ]; then
        green "$desc"
        PASS=$((PASS+1))
    else
        red "$desc — $result"
        FAIL=$((FAIL+1))
    fi
}

yellow "Validating against: $BASE_URL"
echo ""

# ── 1. Health check ───────────────────────────────────────────────────────────
HEALTH=$(curl -sf "$BASE_URL/health" 2>/dev/null || echo "ERROR")
if echo "$HEALTH" | grep -q "ok\|healthy"; then
    check "GET /health returns ok/healthy" "ok"
else
    check "GET /health returns ok/healthy" "$HEALTH"
fi

# ── 2. /tasks endpoint ───────────────────────────────────────────────────────
TASKS_RESP=$(curl -sf "$BASE_URL/tasks" 2>/dev/null || echo "ERROR")
if echo "$TASKS_RESP" | grep -q "tasks\|id"; then
    check "GET /tasks returns task list" "ok"
else
    check "GET /tasks returns task list" "$TASKS_RESP"
fi

# ── 3. At least 3 tasks with graders ─────────────────────────────────────────
GRADER_COUNT=$(echo "$TASKS_RESP" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    tasks = data.get('tasks', data) if isinstance(data, dict) else data
    count = sum(1 for t in tasks if t.get('has_grader') or t.get('grader'))
    print(count)
except Exception as e:
    print(0)
" 2>/dev/null || echo "0")

if [ "$GRADER_COUNT" -ge 3 ]; then
    check "At least 3 tasks have graders (found $GRADER_COUNT)" "ok"
else
    check "At least 3 tasks have graders (found $GRADER_COUNT)" "need >= 3, got $GRADER_COUNT"
fi

# ── 4. /reset works ───────────────────────────────────────────────────────────
RESET_RESP=$(curl -sf -X POST "$BASE_URL/reset" \
    -H "Content-Type: application/json" \
    -d '{"task_id":"task_1_syntax_error"}' 2>/dev/null || echo "ERROR")
if echo "$RESET_RESP" | grep -q "observation"; then
    check "POST /reset returns observation" "ok"
else
    check "POST /reset returns observation" "$RESET_RESP"
fi

# ── 5. /step works ────────────────────────────────────────────────────────────
STEP_RESP=$(curl -sf -X POST "$BASE_URL/step" \
    -H "Content-Type: application/json" \
    -d '{"action_type":"ReadFile","path":"main.py"}' 2>/dev/null || echo "ERROR")
if echo "$STEP_RESP" | grep -q "reward\|observation"; then
    check "POST /step returns reward + observation" "ok"
else
    check "POST /step returns reward + observation" "$STEP_RESP"
fi

# ── 6. openenv.yaml present ──────────────────────────────────────────────────
if [ -f "$REPO_DIR/openenv.yaml" ]; then
    check "openenv.yaml present" "ok"
else
    check "openenv.yaml present" "not found"
fi

# ── 7. Dockerfile present ────────────────────────────────────────────────────
if [ -f "$REPO_DIR/Dockerfile" ]; then
    check "Dockerfile present" "ok"
else
    check "Dockerfile present" "not found"
fi

# ── 8. inference.py present ──────────────────────────────────────────────────
if [ -f "$REPO_DIR/inference.py" ]; then
    check "inference.py present" "ok"
else
    check "inference.py present" "not found"
fi

# ── 9. inference.py has required log markers ─────────────────────────────────
if grep -q "\[START\]" "$REPO_DIR/inference.py" && \
   grep -q "\[STEP\]"  "$REPO_DIR/inference.py" && \
   grep -q "\[END\]"   "$REPO_DIR/inference.py"; then
    check "inference.py has [START]/[STEP]/[END] log markers" "ok"
else
    check "inference.py has [START]/[STEP]/[END] log markers" "missing markers"
fi

# ── 10. Score range in inference.py ──────────────────────────────────────────
if grep -q "0\.99\|0\.999" "$REPO_DIR/inference.py"; then
    check "inference.py uses score threshold >= 0.99" "ok"
else
    check "inference.py uses score threshold >= 0.99" "check SUCCESS_SCORE_THRESHOLD"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "─────────────────────────────────"
echo "Results: $PASS passed, $FAIL failed"
echo "─────────────────────────────────"

if [ "$FAIL" -eq 0 ]; then
    green "All checks passed — ready to submit!"
    exit 0
else
    red "$FAIL check(s) failed — fix before submitting."
    exit 1
fi