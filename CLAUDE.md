# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**smolagents** is a minimal agent framework by Hugging Face (~1,000 lines of core code). Key distinction: **Code Agents** generate Python code for actions rather than JSON tool calls, using 30% fewer LLM steps.

- **Python**: >= 3.10
- **Branch**: `feature/creative-orchestration` (local extensions for creative prompt orchestration)

## Build Commands

```bash
# Install for development
pip install -e ".[dev]"

# Check code quality (ruff lint + format check)
make quality

# Auto-format code
make style

# Run tests
make test
```

## Code Style

- **Line length**: 119 characters
- **Linter**: ruff (E, F, I, W rules)
- **Ignored**: F403 (import *), E501 (long lines)
- **Imports**: isort with 2 blank lines after imports, `smolagents` is first-party

## Architecture

### Core Library (`src/smolagents/`)

| File | Purpose |
|------|---------|
| `agents.py` | CodeAgent (code actions) and ToolCallingAgent (JSON actions) |
| `models.py` | LLM providers: InferenceClientModel, LiteLLMModel, OpenAIModel, AzureOpenAIModel, AmazonBedrockModel, TransformersModel, VLLMModel |
| `tools.py` | Tool base class and `@tool` decorator |
| `local_python_executor.py` | Local code execution with security controls |
| `remote_executors.py` | Sandboxed execution via E2B, Docker, Blaxel, Modal, WASM |
| `cli.py` | CLI commands: `smolagent` (general), `webagent` (vision browser) |

### Agent Flow

1. Task added to `agent.memory`
2. ReAct loop: Memory → Generate code → Parse → Execute → Update memory
3. Loop until `final_answer()` tool is called

### Local Extensions (feature/creative-orchestration)

| File | Purpose |
|------|---------|
| `agent_init.py` | Factory functions: `get_gemini_agent()`, `get_claude_agent()`, `get_qwen_agent()`, `get_hf_agent()` |
| `skills_bridge.py` | `generate_synthetic_prompt()` and `push_to_hf_hub()` tools |
| `tools/hf_tools.py` | 11 HuggingFace Hub tools (download, upload, search, query datasets, cloud jobs) |

## Usage Patterns

### Creating Agents

```python
# Using local factory (feature branch)
from agent_init import get_gemini_agent, get_hf_agent
agent = get_gemini_agent(tools=[...])
hf_agent = get_hf_agent(provider="gemini")  # All HF tools

# Standard smolagents
from smolagents import CodeAgent, WebSearchTool, InferenceClientModel
model = InferenceClientModel()
agent = CodeAgent(tools=[WebSearchTool()], model=model)
```

### Defining Tools

```python
from smolagents import tool

@tool
def my_tool(query: str) -> str:
    """Clear docstring with param descriptions."""
    return result
```

## Environment Variables

Required in `.env`:
```
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
HF_TOKEN=...
```

## Testing

```bash
# Run all tests
pytest ./tests/ -sv --durations=0

# Run single test file
pytest tests/test_agents.py -sv

# Run specific test
pytest tests/test_tools.py::test_function_name -sv
```

## CLI

```bash
# General agent with tools
smolagent "Your prompt" --tools web_search --model-id "Qwen/Qwen2.5-72B-Instruct"

# Interactive mode
smolagent

# Vision web browser
webagent "Go to xyz.com and find..."
```
