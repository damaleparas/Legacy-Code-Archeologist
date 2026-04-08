# Legacy Code Archeologist

A benchmark environment for legacy code debugging and refactoring.

## Overview

This project provides a sandboxed environment for testing AI agents on legacy code tasks. It includes a server, an environment wrapper, and various tasks designed to challenge code understanding and fixing capabilities.

## Features

- **Sandboxed Environment**: Run tests and code edits in a safe, isolated container.
- **REST API**: Interact with the environment via simple HTTP endpoints.
- **Task System**: Supports multiple tasks with varying difficulty.
- **OpenEnv Compatible**: Designed to work with the OpenEnv framework.

## Installation

```bash
pip install .
```

## Running the Server

```bash
python server.py --host 0.0.0.0 --port 7860
```

## Running Tests

```bash
pytest tests/
```

## License

MIT
