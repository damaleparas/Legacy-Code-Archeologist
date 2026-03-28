import sys
import os
from pathlib import Path

# Add the project root to sys.path so that tests can easily import
# project modules like 'models', 'grader', 'env', 'task', etc.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# You can add global pytest fixtures here if needed in the future.
